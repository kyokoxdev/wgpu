//! WGSL-module → GLSL 460 translation for the deko3d backend.
//!
//! Shared by the device backend and the host bake tool (feature
//! `deko3d-bake`, no deko3d-sys), so both produce byte-identical GLSL and
//! therefore the same pack key.

use alloc::{format, string::String, vec::Vec};

use naga::back::glsl;

/// Per-stage resource tables (`DK_NUM_*`).
pub const MAX_UNIFORM_BUFFERS: u8 = 16;
pub const MAX_STORAGE_BUFFERS: u8 = 16;
pub const MAX_TEXTURES: u8 = 32;
pub const MAX_IMAGES: u8 = 8;

/// Slot assignment for one pipeline layout. Each resource class counts from
/// 0 across all groups, in group then entry order (as the GL backend does).
/// A sampler's slot is unused by GLSL: samplers bind into the slot of the
/// texture they are paired with (see [`Glsl::samplers`]).
pub fn binding_map(
    groups: &[Option<&[wgt::BindGroupLayoutEntry]>],
) -> Result<glsl::BindingMap, String> {
    let (mut samplers, mut textures, mut images, mut ubos, mut ssbos) = (0u8, 0u8, 0u8, 0u8, 0u8);
    let mut map = glsl::BindingMap::default();
    for (group, entries) in (0u32..).zip(groups) {
        for entry in entries.iter().flat_map(|e| e.iter()) {
            let (counter, max) = match entry.ty {
                wgt::BindingType::Sampler(_) => (&mut samplers, u8::MAX),
                wgt::BindingType::Texture { .. } => (&mut textures, MAX_TEXTURES),
                wgt::BindingType::StorageTexture { .. } => (&mut images, MAX_IMAGES),
                wgt::BindingType::Buffer {
                    ty: wgt::BufferBindingType::Uniform,
                    ..
                } => (&mut ubos, MAX_UNIFORM_BUFFERS),
                wgt::BindingType::Buffer {
                    ty: wgt::BufferBindingType::Storage { .. },
                    ..
                } => (&mut ssbos, MAX_STORAGE_BUFFERS),
                ref ty => return Err(format!("deko3d: unsupported binding {ty:?}")),
            };
            let count = entry.count.map_or(1, |c| c.get());
            if u32::from(*counter) + count > u32::from(max) {
                return Err(format!("deko3d: too many {:?} bindings", entry.ty));
            }
            let br = naga::ResourceBinding {
                group,
                binding: entry.binding,
            };
            map.insert(br, *counter);
            *counter += count as u8;
        }
    }
    Ok(map)
}

/// Writer options for a layout's binding map.
pub fn options(binding_map: glsl::BindingMap) -> glsl::Options {
    glsl::Options {
        version: glsl::Version::Desktop(460),
        // deko3d's device defaults (depth 0..1, origin upper left) match
        // WebGPU, so no coordinate fix-up. DRAW_PARAMETERS makes
        // instance_index include first_instance.
        writer_flags: glsl::WriterFlags::DRAW_PARAMETERS,
        binding_map,
        zero_initialize_workgroup_memory: true,
    }
}

/// One stage's GLSL.
#[derive(Debug)]
pub struct Glsl {
    pub text: String,
    /// Pack key: `fnv1a64(text)`.
    pub key: u64,
    /// `(texture slot, sampler binding)` per combined sampler: bind time puts
    /// that sampler into the texture's slot. `None`: texel fetch only.
    pub samplers: Vec<(u8, Option<naga::ResourceBinding>)>,
}

pub fn fnv1a64(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325, |h, &b| {
        (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3)
    })
}

/// Translates one entry point, mirroring the GL backend's `create_shader`
/// (overrides, then unchecked bounds, then the GLSL writer).
pub fn translate(
    module: &naga::Module,
    info: &naga::valid::ModuleInfo,
    stage: naga::ShaderStage,
    entry_point: &str,
    constants: &naga::back::PipelineConstants,
    zero_initialize_workgroup_memory: bool,
    options: &glsl::Options,
) -> Result<Glsl, String> {
    let (module, info) = naga::back::pipeline_constants::process_overrides(
        module,
        info,
        Some((stage, entry_point)),
        constants,
    )
    .map_err(|e| format!("{e}"))?;

    let mut temp;
    let options = if options.zero_initialize_workgroup_memory == zero_initialize_workgroup_memory {
        options
    } else {
        temp = options.clone();
        temp.zero_initialize_workgroup_memory = zero_initialize_workgroup_memory;
        &temp
    };
    let pipeline_options = glsl::PipelineOptions {
        shader_stage: stage,
        entry_point: entry_point.into(),
        multiview: None,
    };
    let policies = naga::proc::BoundsCheckPolicies {
        index: naga::proc::BoundsCheckPolicy::Unchecked,
        buffer: naga::proc::BoundsCheckPolicy::Unchecked,
        image_load: naga::proc::BoundsCheckPolicy::Unchecked,
        binding_array: naga::proc::BoundsCheckPolicy::Unchecked,
    };
    let mut text = String::new();
    let reflection = glsl::Writer::new(
        &mut text,
        &module,
        &info,
        options,
        &pipeline_options,
        policies,
    )
    .and_then(|mut w| w.write())
    .map_err(|e| format!("{e}"))?;

    let binding = |h: naga::Handle<naga::GlobalVariable>| module.global_variables[h].binding;
    let mut samplers = Vec::new();
    for mapping in reflection.texture_mapping.values() {
        let texture = binding(mapping.texture).ok_or("deko3d: unbound texture")?;
        let slot = *options
            .binding_map
            .get(&texture)
            .ok_or_else(|| format!("deko3d: texture {texture:?} not in layout"))?;
        let sampler = mapping.sampler.and_then(binding);
        match samplers.iter().find(|&&(s, _)| s == slot) {
            // One texture with two samplers would need two slots.
            Some(&(_, other)) if other != sampler => {
                return Err(format!("deko3d: texture {texture:?} used with two samplers"))
            }
            Some(_) => {}
            None => samplers.push((slot, sampler)),
        }
    }
    samplers.sort_unstable_by_key(|&(slot, _)| slot);

    Ok(Glsl {
        key: fnv1a64(text.as_bytes()),
        text,
        samplers,
    })
}

/// A shader module as the hal receives it: the naga IR, translated per
/// pipeline once the layout is known.
#[derive(Debug)]
pub struct ShaderModule {
    module: alloc::sync::Arc<naga::Module>,
    info: alloc::sync::Arc<naga::valid::ModuleInfo>,
}

impl ShaderModule {
    pub fn new(shader: crate::ShaderInput) -> Result<Self, crate::ShaderError> {
        match shader {
            crate::ShaderInput::Naga(naga) => Ok(Self {
                module: alloc::sync::Arc::new(naga.module.into_owned()),
                info: alloc::sync::Arc::new(naga.info),
            }),
            _ => Err(crate::ShaderError::Compilation(
                "deko3d: only naga shader input".into(),
            )),
        }
    }
}

#[derive(Debug)]
pub struct BindGroupLayout {
    pub(crate) entries: alloc::sync::Arc<[wgt::BindGroupLayoutEntry]>,
}

impl BindGroupLayout {
    pub fn new(desc: &crate::BindGroupLayoutDescriptor) -> Self {
        Self {
            entries: desc.entries.into(),
        }
    }
}

/// Where a bind group's resources of each class start in the slot tables.
#[derive(Clone, Copy, Debug, Default)]
pub struct GroupBase {
    pub uniform: u8,
    pub storage: u8,
    pub texture: u8,
    pub image: u8,
}

#[derive(Debug)]
pub struct PipelineLayout {
    options: glsl::Options,
    /// Per bind group, in `binding_map`'s counting order.
    pub(crate) bases: Vec<GroupBase>,
}

impl PipelineLayout {
    pub fn new(
        desc: &crate::PipelineLayoutDescriptor<BindGroupLayout>,
    ) -> Result<Self, crate::DeviceError> {
        let groups: Vec<_> = desc
            .bind_group_layouts
            .iter()
            .map(|bgl| bgl.map(|bgl| &bgl.entries[..]))
            .collect();
        let map = binding_map(&groups).map_err(|e| {
            log::error!("{e}");
            crate::DeviceError::Unexpected
        })?;
        let mut next = GroupBase::default();
        let bases = groups
            .iter()
            .map(|entries| {
                let base = next;
                for entry in entries.iter().flat_map(|e| e.iter()) {
                    let count = entry.count.map_or(1, |c| c.get()) as u8;
                    match entry.ty {
                        wgt::BindingType::Texture { .. } => next.texture += count,
                        wgt::BindingType::StorageTexture { .. } => next.image += count,
                        wgt::BindingType::Buffer {
                            ty: wgt::BufferBindingType::Uniform,
                            ..
                        } => next.uniform += count,
                        wgt::BindingType::Buffer { .. } => next.storage += count,
                        _ => {}
                    }
                }
                base
            })
            .collect();
        Ok(Self {
            options: options(map),
            bases,
        })
    }
}

crate::impl_dyn_resource!(ShaderModule, BindGroupLayout, PipelineLayout);
impl crate::DynShaderModule for ShaderModule {}
impl crate::DynBindGroupLayout for BindGroupLayout {}
impl crate::DynPipelineLayout for PipelineLayout {}

/// The vertex and fragment GLSL of a render pipeline.
pub fn render_pipeline_glsl<R: crate::DynPipelineCache + ?Sized>(
    desc: &crate::RenderPipelineDescriptor<PipelineLayout, ShaderModule, R>,
) -> Result<(Glsl, Option<Glsl>), crate::PipelineError> {
    let crate::VertexProcessor::Standard { vertex_stage, .. } = &desc.vertex_processor else {
        return Err(crate::PipelineError::Linkage(
            wgt::ShaderStages::MESH,
            "deko3d: no mesh shaders".into(),
        ));
    };
    let stage = |s: &crate::ProgrammableStage<ShaderModule>, naga_stage, wgt_stage| {
        translate(
            &s.module.module,
            &s.module.info,
            naga_stage,
            s.entry_point,
            s.constants,
            s.zero_initialize_workgroup_memory,
            &desc.layout.options,
        )
        .map_err(|e| crate::PipelineError::Linkage(wgt_stage, e))
    };
    let vertex = stage(vertex_stage, naga::ShaderStage::Vertex, wgt::ShaderStages::VERTEX)?;
    let fragment = desc
        .fragment_stage
        .as_ref()
        .map(|f| stage(f, naga::ShaderStage::Fragment, wgt::ShaderStages::FRAGMENT))
        .transpose()?;
    Ok((vertex, fragment))
}

/// The installed `.dkpk` packs (`crates/bake/src/pack.rs` writes them): what
/// the device turns GLSL keys into DKSH with. Global so the game can install
/// one without reaching through wgpu to the hal device.
pub mod packs {
    use alloc::{format, string::String, sync::Arc, vec::Vec};
    use std::collections::HashMap;
    use std::sync::Mutex;

    pub const FORMAT: u32 = 1;
    /// uam 1.1.0, as `major << 16 | minor << 8 | patch`.
    pub const UAM_VERSION: u32 = 0x01_01_00;
    const ALIGN: usize = 0x100;

    /// Key, GLSL length, stage, and the DKSH's byte range in the pack.
    pub type Entry = (u64, u32, u8, core::ops::Range<usize>);

    /// Checks the header and walks the entries.
    pub fn parse(bytes: &[u8]) -> Result<Vec<Entry>, String> {
        let u32_at = |at: usize| {
            bytes
                .get(at..at + 4)
                .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
                .ok_or("truncated")
        };
        if bytes.get(..4) != Some(&b"DKPK"[..]) {
            return Err("not a dkpk".into());
        }
        let (format, uam) = (u32_at(4)?, u32_at(8)?);
        if format != FORMAT || uam != UAM_VERSION {
            return Err(format!(
                "pack format {format} uam {uam:#x}, want {FORMAT} {UAM_VERSION:#x}: re-run iw4l-bake"
            ));
        }
        let count = u32_at(12)? as usize;
        let mut entries = Vec::with_capacity(count);
        let mut at = 16;
        for _ in 0..count {
            let key = bytes.get(at..at + 8).ok_or("truncated")?;
            let key = u64::from_le_bytes(key.try_into().unwrap());
            let glsl_len = u32_at(at + 8)?;
            let stage = *bytes.get(at + 12).ok_or("truncated")?;
            let dksh_len = u32_at(at + 13)? as usize;
            let start = (at + 17).next_multiple_of(ALIGN);
            let end = start + dksh_len;
            if end > bytes.len() {
                return Err("truncated".into());
            }
            entries.push((key, glsl_len, stage, start..end));
            at = end;
        }
        Ok(entries)
    }

    struct Pack {
        name: String,
        bytes: Arc<[u8]>,
        index: HashMap<u64, (u32, core::ops::Range<usize>)>,
    }

    static PACKS: Mutex<Vec<Pack>> = Mutex::new(Vec::new());

    /// Installs a pack, replacing one of the same name. Returns its entry count.
    pub fn install(name: &str, bytes: Vec<u8>) -> Result<usize, String> {
        let index: HashMap<_, _> = parse(&bytes)?
            .into_iter()
            .map(|(key, glsl_len, _, range)| (key, (glsl_len, range)))
            .collect();
        let count = index.len();
        let mut packs = PACKS.lock().unwrap_or_else(|e| e.into_inner());
        packs.retain(|p| p.name != name);
        packs.push(Pack {
            name: name.into(),
            bytes: bytes.into(),
            index,
        });
        Ok(count)
    }

    /// Drops a pack. Code already loaded from it stays until its pipelines go.
    pub fn remove(name: &str) {
        PACKS
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .retain(|p| p.name != name);
    }

    static MISSING: core::sync::atomic::AtomicUsize = core::sync::atomic::AtomicUsize::new(0);

    #[cfg_attr(not(deko3d), allow(dead_code))]
    pub(crate) fn note_missing() {
        MISSING.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
    }

    /// Pipelines created so far whose DKSH no installed pack has.
    pub fn missing() -> usize {
        MISSING.load(core::sync::atomic::Ordering::Relaxed)
    }

    /// Runs `f` on the DKSH for `key`. `glsl_len` guards against a hash collision.
    #[cfg_attr(not(deko3d), allow(dead_code))]
    pub(crate) fn with_dksh<R>(key: u64, glsl_len: u32, f: impl FnOnce(&[u8]) -> R) -> Option<R> {
        let packs = PACKS.lock().unwrap_or_else(|e| e.into_inner());
        packs.iter().find_map(|pack| {
            let (len, range) = pack.index.get(&key)?;
            (*len == glsl_len).then_some(range.clone())
                .map(|range| (&pack.bytes, range))
        })
        .map(|(bytes, range)| f(&bytes[range]))
    }
}

/// What the Noop backend records under `deko3d-bake`: every render
/// pipeline's GLSL, for the host bake to compile into packs.
#[cfg(feature = "deko3d-bake")]
pub mod record {
    use alloc::{string::String, vec::Vec};

    #[derive(Debug)]
    pub struct Recorded {
        pub label: Option<String>,
        pub result: Result<(super::Glsl, Option<super::Glsl>), crate::PipelineError>,
    }

    static RECORDED: std::sync::Mutex<Vec<Recorded>> = std::sync::Mutex::new(Vec::new());

    pub fn push(recorded: Recorded) {
        RECORDED.lock().unwrap_or_else(|e| e.into_inner()).push(recorded);
    }

    pub fn take() -> Vec<Recorded> {
        core::mem::take(&mut *RECORDED.lock().unwrap_or_else(|e| e.into_inner()))
    }
}

#[cfg(test)]
mod tests {
    extern crate std;

    use super::*;
    use alloc::vec;

    /// Same shape as d3d9_sm3's bound-slot output: a storage constant arena
    /// at (0,0) and (texture, sampler) pairs at group 1, bindings 2n/2n+1.
    const SAMPLE: &str = r#"
struct Sm3ConstantArena { c: array<vec4<f32>> }
@group(0) @binding(0) var<storage, read> sm3_constants: Sm3ConstantArena;
@group(1) @binding(0) var sm3_tex_s0: texture_2d<f32>;
@group(1) @binding(1) var sm3_smp_s0: sampler;
@group(1) @binding(2) var sm3_tex_s1: texture_cube<f32>;
@group(1) @binding(3) var sm3_smp_s1: sampler;

struct VsOut { @builtin(position) pos: vec4<f32>, @location(0) uv: vec2<f32> }

@vertex
fn vs_main(@location(0) v0: vec4<f32>, @builtin(instance_index) inst: u32) -> VsOut {
    let m = sm3_constants.c[inst * 4u];
    return VsOut(v0 * m, v0.xy);
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let a = textureSample(sm3_tex_s0, sm3_smp_s0, in.uv);
    let b = textureSample(sm3_tex_s1, sm3_smp_s1, vec3<f32>(in.uv, 1.0));
    return a * b * sm3_constants.c[1];
}
"#;

    fn entries() -> (Vec<wgt::BindGroupLayoutEntry>, Vec<wgt::BindGroupLayoutEntry>) {
        let e = |binding, ty| wgt::BindGroupLayoutEntry {
            binding,
            visibility: wgt::ShaderStages::VERTEX_FRAGMENT,
            ty,
            count: None,
        };
        let tex = |d| wgt::BindingType::Texture {
            sample_type: wgt::TextureSampleType::Float { filterable: true },
            view_dimension: d,
            multisampled: false,
        };
        let smp = wgt::BindingType::Sampler(wgt::SamplerBindingType::Filtering);
        (
            vec![e(
                0,
                wgt::BindingType::Buffer {
                    ty: wgt::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
            )],
            vec![
                e(0, tex(wgt::TextureViewDimension::D2)),
                e(1, smp),
                e(2, tex(wgt::TextureViewDimension::Cube)),
                e(3, smp),
            ],
        )
    }

    fn translate_sample() -> [Glsl; 2] {
        let module = naga::front::wgsl::parse_str(SAMPLE).unwrap();
        let info = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap();
        let (g0, g1) = entries();
        let options = options(binding_map(&[Some(&g0), Some(&g1)]).unwrap());
        let t = |stage, ep| {
            translate(&module, &info, stage, ep, &Default::default(), true, &options).unwrap()
        };
        [
            t(naga::ShaderStage::Vertex, "vs_main"),
            t(naga::ShaderStage::Fragment, "fs_main"),
        ]
    }

    #[test]
    fn sample_glsl_is_stable_and_bound() {
        let [vs, fs] = translate_sample();
        let [vs2, fs2] = translate_sample();
        assert_eq!((vs.key, fs.key), (vs2.key, fs2.key));
        assert_eq!(vs.text, vs2.text);
        assert!(fs.text.starts_with("#version 460"), "{}", fs.text);
        assert!(fs.text.contains("layout(binding = 1) uniform samplerCube"), "{}", fs.text);
        let s = |binding| Some(naga::ResourceBinding { group: 1, binding });
        assert_eq!(fs.samplers, vec![(0, s(1)), (1, s(3))]);
        assert!(vs.samplers.is_empty());
    }

    #[test]
    fn pack_parses_and_refuses() {
        let mut pack = vec![];
        pack.extend_from_slice(b"DKPK");
        for word in [packs::FORMAT, packs::UAM_VERSION, 1] {
            pack.extend_from_slice(&word.to_le_bytes());
        }
        pack.extend_from_slice(&7u64.to_le_bytes());
        pack.extend_from_slice(&9u32.to_le_bytes());
        pack.push(1);
        pack.extend_from_slice(&3u32.to_le_bytes());
        pack.resize(0x100, 0);
        pack.extend_from_slice(&[1, 2, 3]);
        assert_eq!(packs::parse(&pack).unwrap(), vec![(7, 9, 1, 0x100..0x103)]);
        assert!(packs::parse(&pack[..0x101]).is_err());
        pack[4] = 99;
        assert!(packs::parse(&pack).is_err());
    }

    /// Runs devkitPro's uam on the sample when it is installed (B1 check).
    /// `DEKO3D_UAM_DUMP=<dir>` keeps the GLSL there.
    #[test]
    fn uam_accepts_sample() {
        let uam = std::path::Path::new(&std::env::var("DEVKITPRO").unwrap_or_default())
            .join("tools/bin/uam");
        if !uam.exists() {
            std::println!("skip: no uam at {}", uam.display());
            return;
        }
        let dir = std::env::var("DEKO3D_UAM_DUMP")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|_| std::env::temp_dir());
        let [vs, fs] = translate_sample();
        for (glsl, stage) in [(vs, "vert"), (fs, "frag")] {
            let src = dir.join(format!("deko3d_sample.{stage}.glsl"));
            std::fs::write(&src, &glsl.text).unwrap();
            let out = std::process::Command::new(&uam)
                .args(["-s", stage, "-o"])
                .arg(src.with_extension("dksh"))
                .arg(&src)
                .output()
                .unwrap();
            assert!(
                out.status.success(),
                "uam {stage}:\n{}\n{}",
                String::from_utf8_lossy(&out.stderr),
                glsl.text
            );
        }
    }
}
