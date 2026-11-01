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
    entries: alloc::sync::Arc<[wgt::BindGroupLayoutEntry]>,
}

impl BindGroupLayout {
    pub fn new(desc: &crate::BindGroupLayoutDescriptor) -> Self {
        Self {
            entries: desc.entries.into(),
        }
    }
}

#[derive(Debug)]
pub struct PipelineLayout {
    options: glsl::Options,
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
        Ok(Self {
            options: options(map),
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
