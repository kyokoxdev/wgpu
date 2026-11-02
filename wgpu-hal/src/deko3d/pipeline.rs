//! K2: bind groups and render pipelines. deko3d has no pipeline object, so a
//! pipeline is DKSH shaders from the packs plus prebuilt state words.

use alloc::{format, string::String, sync::Arc, vec::Vec};
use core::ptr;

use deko3d_sys as dk;
use hashbrown::HashMap;

use super::memory::{Allocation, Pool};
use crate::deko3d_shader::{packs, render_pipeline_glsl, BindGroupLayout, Glsl, PipelineLayout, ShaderModule};

/// A loaded shader, shared by every pipeline using its key.
pub(super) struct Shader {
    raw: dk::DkShader,
    mem: Allocation,
}
unsafe impl Send for Shader {}
unsafe impl Sync for Shader {}

impl Shader {
    pub(super) fn raw(&self) -> *const dk::DkShader {
        &self.raw
    }
}

/// The DKSH layout: a control section (`DkshHeader` first), then code.
fn control_size(dksh: &[u8]) -> Option<usize> {
    let word = |at: usize| Some(u32::from_le_bytes(dksh.get(at..at + 4)?.try_into().ok()?) as usize);
    (dksh.get(..4)? == b"DKSH").then_some(())?;
    word(8)
}

/// Loaded shaders by pack key. Code stays in the pool while a pipeline holds it.
pub(super) struct ShaderCache {
    code: Pool,
    loaded: HashMap<u64, Arc<Shader>>,
}

impl ShaderCache {
    pub fn new(device: dk::DkDevice) -> Self {
        Self {
            code: Pool::new_code(device),
            loaded: HashMap::new(),
        }
    }

    fn get(&mut self, glsl: &Glsl) -> Result<Arc<Shader>, String> {
        if let Some(shader) = self.loaded.get(&glsl.key) {
            return Ok(Arc::clone(shader));
        }
        let code = &mut self.code;
        let shader = packs::with_dksh(glsl.key, glsl.text.len() as u32, |dksh| {
            let control = control_size(dksh).ok_or("bad dksh")?;
            let bytes = &dksh[control..];
            let mem = code
                .alloc(bytes.len() as u64, dk::DK_SHADER_CODE_ALIGNMENT)
                .ok_or("out of code memory")?;
            let mut raw = dk::DkShader::zeroed();
            unsafe {
                let cpu = dk::dkMemBlockGetCpuAddr(mem.block).cast::<u8>();
                ptr::copy_nonoverlapping(bytes.as_ptr(), cpu.add(mem.offset as usize), bytes.len());
                dk::dkShaderInitialize(
                    &mut raw,
                    &dk::DkShaderMaker {
                        codeMem: mem.block,
                        control: dksh.as_ptr().cast(),
                        codeOffset: mem.offset,
                        programId: 0,
                    },
                );
            }
            Ok::<_, &str>(Arc::new(Shader { raw, mem }))
        })
        .ok_or_else(|| format!("missing dksh {:016x}", glsl.key))?
        .map_err(|e| format!("{e} {:016x}", glsl.key))?;
        self.loaded.insert(glsl.key, Arc::clone(&shader));
        Ok(shader)
    }

    /// Frees the code of shaders only the cache still holds. Called when a
    /// pipeline goes; wgpu-core destroys it after its last submit retired.
    pub fn trim(&mut self, keys: &[u64]) {
        for key in keys {
            let unused = self.loaded.get(key).is_some_and(|s| Arc::strong_count(s) == 1);
            if unused {
                if let Some(shader) = self.loaded.remove(key) {
                    self.code.free(shader.mem);
                }
            }
        }
    }
}

/// A buffer entry of a bind group, in layout order.
pub(super) struct BoundBuffer {
    pub addr: dk::DkGpuAddr,
    pub size: u32,
    pub stages: wgt::ShaderStages,
    pub uniform: bool,
    pub dynamic: bool,
}

#[derive(Debug, Default)]
pub struct BindGroup {
    pub(super) buffers: Vec<BoundBuffer>,
    /// Image descriptors of sampled textures, in layout order.
    pub(super) textures: Vec<u32>,
    /// Image descriptors of storage textures, in layout order.
    pub(super) images: Vec<(u32, wgt::ShaderStages)>,
    /// `(binding, sampler descriptor)`, sorted by binding.
    pub(super) samplers: Vec<(u32, u32)>,
}
unsafe impl Send for BindGroup {}
unsafe impl Sync for BindGroup {}

impl core::fmt::Debug for BoundBuffer {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("BoundBuffer").field("addr", &self.addr).finish()
    }
}

impl BindGroup {
    pub(super) fn new(
        desc: &crate::BindGroupDescriptor<
            BindGroupLayout,
            super::Buffer,
            super::Sampler,
            super::TextureView,
            super::Resource,
        >,
    ) -> Result<Self, crate::DeviceError> {
        let mut group = Self::default();
        for layout in desc.layout.entries.iter() {
            let Some(entry) = desc.entries.iter().find(|e| e.binding == layout.binding) else {
                continue;
            };
            let at = entry.resource_index as usize;
            let view = |at: usize| desc.textures[at].view.descriptor.ok_or(crate::DeviceError::Unexpected);
            match layout.ty {
                wgt::BindingType::Buffer {
                    ty,
                    has_dynamic_offset,
                    ..
                } => {
                    let binding = &desc.buffers[at];
                    let uniform = matches!(ty, wgt::BufferBindingType::Uniform);
                    let size = binding
                        .size
                        .map_or(binding.buffer.size() as u64 - binding.offset, wgt::BufferSize::get);
                    // The hardware takes a uniform buffer in 256-byte units.
                    let size = if uniform {
                        size.next_multiple_of(u64::from(dk::DK_UNIFORM_BUF_ALIGNMENT))
                            .min(u64::from(dk::DK_UNIFORM_BUF_MAX_SIZE))
                    } else {
                        size
                    };
                    group.buffers.push(BoundBuffer {
                        addr: binding.buffer.gpu_addr() + binding.offset,
                        size: size as u32,
                        stages: layout.visibility,
                        uniform,
                        dynamic: has_dynamic_offset,
                    });
                }
                wgt::BindingType::Sampler(_) => group
                    .samplers
                    .push((layout.binding, desc.samplers[at].descriptor)),
                wgt::BindingType::Texture { .. } => group.textures.push(view(at)?),
                wgt::BindingType::StorageTexture { .. } => {
                    group.images.push((view(at)?, layout.visibility))
                }
                _ => return Err(crate::DeviceError::Unexpected),
            }
        }
        group.samplers.sort_unstable_by_key(|&(binding, _)| binding);
        Ok(group)
    }

    pub(super) fn sampler(&self, binding: u32) -> u32 {
        self.samplers
            .binary_search_by_key(&binding, |&(b, _)| b)
            .map_or(0, |at| self.samplers[at].1)
    }
}

/// `(texture slot, (group, sampler binding))` per combined sampler of a stage.
type StageSamplers = Vec<(u8, Option<(u8, u32)>)>;

pub struct RenderPipeline {
    /// `None`: a stage's DKSH is missing, so draws with this pipeline are
    /// skipped (see `packs::missing`).
    pub(super) shaders: Option<(Arc<Shader>, Option<Arc<Shader>>)>,
    keys: [u64; 2],
    /// Vertex, then fragment.
    pub(super) samplers: [StageSamplers; 2],
    pub(super) primitive: dk::DkPrimitive,
    /// The restart index for strips, per index format.
    pub(super) restart: bool,
    pub(super) rasterizer: dk::DkRasterizerState,
    /// Constant, clamp and slope factors.
    pub(super) depth_bias: Option<(f32, f32, f32)>,
    pub(super) color: dk::DkColorState,
    pub(super) color_write: dk::DkColorWriteState,
    pub(super) blends: Vec<dk::DkBlendState>,
    pub(super) depth_stencil: dk::DkDepthStencilState,
    /// Write and read masks, when the stencil test is on.
    pub(super) stencil: Option<(u8, u8)>,
    pub(super) attribs: Vec<dk::DkVtxAttribState>,
    pub(super) vertex_buffers: Vec<dk::DkVtxBufferState>,
}
unsafe impl Send for RenderPipeline {}
unsafe impl Sync for RenderPipeline {}

impl core::fmt::Debug for RenderPipeline {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("RenderPipeline").field("keys", &self.keys).finish()
    }
}

fn blend_op(op: wgt::BlendOperation) -> u32 {
    use wgt::BlendOperation as O;
    match op {
        O::Add => dk::DkBlendOp_Add,
        O::Subtract => dk::DkBlendOp_Sub,
        O::ReverseSubtract => dk::DkBlendOp_RevSub,
        O::Min => dk::DkBlendOp_Min,
        O::Max => dk::DkBlendOp_Max,
    }
}

fn blend_factor(f: wgt::BlendFactor) -> u32 {
    use wgt::BlendFactor as F;
    match f {
        F::Zero => dk::DkBlendFactor_Zero,
        F::One => dk::DkBlendFactor_One,
        F::Src => dk::DkBlendFactor_SrcColor,
        F::OneMinusSrc => dk::DkBlendFactor_InvSrcColor,
        F::SrcAlpha => dk::DkBlendFactor_SrcAlpha,
        F::OneMinusSrcAlpha => dk::DkBlendFactor_InvSrcAlpha,
        F::Dst => dk::DkBlendFactor_DstColor,
        F::OneMinusDst => dk::DkBlendFactor_InvDstColor,
        F::DstAlpha => dk::DkBlendFactor_DstAlpha,
        F::OneMinusDstAlpha => dk::DkBlendFactor_InvDstAlpha,
        F::SrcAlphaSaturated => dk::DkBlendFactor_SrcAlphaSaturate,
        F::Constant => dk::DkBlendFactor_ConstColor,
        F::OneMinusConstant => dk::DkBlendFactor_InvConstColor,
        F::Src1 => dk::DkBlendFactor_Src1Color,
        F::OneMinusSrc1 => dk::DkBlendFactor_InvSrc1Color,
        F::Src1Alpha => dk::DkBlendFactor_Src1Alpha,
        F::OneMinusSrc1Alpha => dk::DkBlendFactor_InvSrc1Alpha,
    }
}

fn stencil_op(op: wgt::StencilOperation) -> u32 {
    use wgt::StencilOperation as O;
    match op {
        O::Keep => dk::DkStencilOp_Keep,
        O::Zero => dk::DkStencilOp_Zero,
        O::Replace => dk::DkStencilOp_Replace,
        O::Invert => dk::DkStencilOp_Invert,
        O::IncrementClamp => dk::DkStencilOp_Incr,
        O::DecrementClamp => dk::DkStencilOp_Decr,
        O::IncrementWrap => dk::DkStencilOp_IncrWrap,
        O::DecrementWrap => dk::DkStencilOp_DecrWrap,
    }
}

fn stencil_face(face: &wgt::StencilFaceState) -> [u32; 4] {
    [
        stencil_op(face.fail_op),
        stencil_op(face.pass_op),
        stencil_op(face.depth_fail_op),
        super::texture::compare_op(face.compare),
    ]
}

fn vertex_format(format: wgt::VertexFormat) -> Option<(u32, u32, bool)> {
    use dk::*;
    use wgt::VertexFormat as V;
    Some(match format {
        V::Uint8 => (DkVtxAttribSize_1x8, DkVtxAttribType_Uint, false),
        V::Uint8x2 => (DkVtxAttribSize_2x8, DkVtxAttribType_Uint, false),
        V::Uint8x4 => (DkVtxAttribSize_4x8, DkVtxAttribType_Uint, false),
        V::Sint8 => (DkVtxAttribSize_1x8, DkVtxAttribType_Sint, false),
        V::Sint8x2 => (DkVtxAttribSize_2x8, DkVtxAttribType_Sint, false),
        V::Sint8x4 => (DkVtxAttribSize_4x8, DkVtxAttribType_Sint, false),
        V::Unorm8 => (DkVtxAttribSize_1x8, DkVtxAttribType_Unorm, false),
        V::Unorm8x2 => (DkVtxAttribSize_2x8, DkVtxAttribType_Unorm, false),
        V::Unorm8x4 => (DkVtxAttribSize_4x8, DkVtxAttribType_Unorm, false),
        V::Unorm8x4Bgra => (DkVtxAttribSize_4x8, DkVtxAttribType_Unorm, true),
        V::Snorm8 => (DkVtxAttribSize_1x8, DkVtxAttribType_Snorm, false),
        V::Snorm8x2 => (DkVtxAttribSize_2x8, DkVtxAttribType_Snorm, false),
        V::Snorm8x4 => (DkVtxAttribSize_4x8, DkVtxAttribType_Snorm, false),
        V::Uint16 => (DkVtxAttribSize_1x16, DkVtxAttribType_Uint, false),
        V::Uint16x2 => (DkVtxAttribSize_2x16, DkVtxAttribType_Uint, false),
        V::Uint16x4 => (DkVtxAttribSize_4x16, DkVtxAttribType_Uint, false),
        V::Sint16 => (DkVtxAttribSize_1x16, DkVtxAttribType_Sint, false),
        V::Sint16x2 => (DkVtxAttribSize_2x16, DkVtxAttribType_Sint, false),
        V::Sint16x4 => (DkVtxAttribSize_4x16, DkVtxAttribType_Sint, false),
        V::Unorm16 => (DkVtxAttribSize_1x16, DkVtxAttribType_Unorm, false),
        V::Unorm16x2 => (DkVtxAttribSize_2x16, DkVtxAttribType_Unorm, false),
        V::Unorm16x4 => (DkVtxAttribSize_4x16, DkVtxAttribType_Unorm, false),
        V::Snorm16 => (DkVtxAttribSize_1x16, DkVtxAttribType_Snorm, false),
        V::Snorm16x2 => (DkVtxAttribSize_2x16, DkVtxAttribType_Snorm, false),
        V::Snorm16x4 => (DkVtxAttribSize_4x16, DkVtxAttribType_Snorm, false),
        V::Float16 => (DkVtxAttribSize_1x16, DkVtxAttribType_Float, false),
        V::Float16x2 => (DkVtxAttribSize_2x16, DkVtxAttribType_Float, false),
        V::Float16x4 => (DkVtxAttribSize_4x16, DkVtxAttribType_Float, false),
        V::Float32 => (DkVtxAttribSize_1x32, DkVtxAttribType_Float, false),
        V::Float32x2 => (DkVtxAttribSize_2x32, DkVtxAttribType_Float, false),
        V::Float32x3 => (DkVtxAttribSize_3x32, DkVtxAttribType_Float, false),
        V::Float32x4 => (DkVtxAttribSize_4x32, DkVtxAttribType_Float, false),
        V::Uint32 => (DkVtxAttribSize_1x32, DkVtxAttribType_Uint, false),
        V::Uint32x2 => (DkVtxAttribSize_2x32, DkVtxAttribType_Uint, false),
        V::Uint32x3 => (DkVtxAttribSize_3x32, DkVtxAttribType_Uint, false),
        V::Uint32x4 => (DkVtxAttribSize_4x32, DkVtxAttribType_Uint, false),
        V::Sint32 => (DkVtxAttribSize_1x32, DkVtxAttribType_Sint, false),
        V::Sint32x2 => (DkVtxAttribSize_2x32, DkVtxAttribType_Sint, false),
        V::Sint32x3 => (DkVtxAttribSize_3x32, DkVtxAttribType_Sint, false),
        V::Sint32x4 => (DkVtxAttribSize_4x32, DkVtxAttribType_Sint, false),
        V::Unorm10_10_10_2 => (DkVtxAttribSize_10_10_10_2, DkVtxAttribType_Unorm, false),
        _ => return None,
    })
}

fn stage_samplers(glsl: &Glsl) -> StageSamplers {
    glsl.samplers
        .iter()
        .map(|&(slot, sampler)| (slot, sampler.map(|b| (b.group as u8, b.binding))))
        .collect()
}

impl RenderPipeline {
    pub(super) fn new(
        cache: &mut ShaderCache,
        desc: &crate::RenderPipelineDescriptor<PipelineLayout, ShaderModule, super::Resource>,
    ) -> Result<Self, crate::PipelineError> {
        let (vs, fs) = render_pipeline_glsl(desc)?;
        let linkage = |stage, e: String| crate::PipelineError::Linkage(stage, e);
        let label = desc.label.unwrap_or("?");
        let loaded = cache.get(&vs).and_then(|vertex| {
            let fragment = fs.as_ref().map(|fs| cache.get(fs)).transpose()?;
            Ok((vertex, fragment))
        });
        let shaders = match loaded {
            Ok(shaders) => Some(shaders),
            Err(e) => {
                packs::note_missing();
                log::error!("deko3d: {e} {label}");
                None
            }
        };

        if desc.multisample.count > 1 {
            // ponytail: every Switch camera runs Msaa::Off; bind a
            // DkMultisampleState here if that changes (F1).
            return Err(linkage(wgt::ShaderStages::FRAGMENT, format!("deko3d: no MSAA yet {label}")));
        }

        let crate::VertexProcessor::Standard { vertex_buffers: layouts, .. } = &desc.vertex_processor else {
            unreachable!("render_pipeline_glsl refuses mesh pipelines");
        };
        let mut attribs = Vec::new();
        let mut vertex_buffers = Vec::with_capacity(layouts.len());
        for (buffer, layout) in layouts.iter().enumerate() {
            let Some(layout) = layout else {
                vertex_buffers.push(dk::DkVtxBufferState { stride: 0, divisor: 0 });
                continue;
            };
            vertex_buffers.push(dk::DkVtxBufferState {
                stride: layout.array_stride as u32,
                divisor: match layout.step_mode {
                    wgt::VertexStepMode::Vertex => 0,
                    wgt::VertexStepMode::Instance => 1,
                },
            });
            for a in layout.attributes {
                let location = a.shader_location as usize;
                if attribs.len() <= location {
                    attribs.resize(location + 1, dk::DkVtxAttribState::FIXED);
                }
                let (size, ty, bgra) = vertex_format(a.format).ok_or_else(|| {
                    linkage(wgt::ShaderStages::VERTEX, format!("deko3d: vertex format {:?}", a.format))
                })?;
                attribs[location] = dk::DkVtxAttribState::new(buffer as u32, a.offset as u32, size, ty, bgra);
            }
        }

        let p = &desc.primitive;
        let (primitive, restart) = match p.topology {
            wgt::PrimitiveTopology::PointList => (dk::DkPrimitive_Points, false),
            wgt::PrimitiveTopology::LineList => (dk::DkPrimitive_Lines, false),
            wgt::PrimitiveTopology::LineStrip => (dk::DkPrimitive_LineStrip, true),
            wgt::PrimitiveTopology::TriangleList => (dk::DkPrimitive_Triangles, false),
            wgt::PrimitiveTopology::TriangleStrip => (dk::DkPrimitive_TriangleStrip, true),
        };

        let bias = desc.depth_stencil.as_ref().map(|ds| ds.bias).filter(|b| b.is_enabled());
        let rasterizer = dk::DkRasterizerState::new(
            p.unclipped_depth,
            match p.polygon_mode {
                wgt::PolygonMode::Fill => dk::DkPolygonMode_Fill,
                wgt::PolygonMode::Line => dk::DkPolygonMode_Line,
                wgt::PolygonMode::Point => dk::DkPolygonMode_Point,
            },
            match p.cull_mode {
                None => dk::DkFace_None,
                Some(wgt::Face::Front) => dk::DkFace_Front,
                Some(wgt::Face::Back) => dk::DkFace_Back,
            },
            // With the device defaults (upper-left origin, clip +Y up) deko3d's
            // winding is what WebGPU's means; see the deko3d cube example.
            match p.front_face {
                wgt::FrontFace::Ccw => dk::DkFrontFace_CCW,
                wgt::FrontFace::Cw => dk::DkFrontFace_CW,
            },
            dk::DkProvokingVertex_First,
            if bias.is_some() { 0b111 } else { 0 },
        );

        let mut blend_mask = 0;
        let mut write_masks = 0;
        let mut blends = Vec::with_capacity(desc.color_targets.len());
        for (i, target) in desc.color_targets.iter().enumerate() {
            let target = target.as_ref();
            write_masks |= target.map_or(0, |t| t.write_mask.bits()) << (i * 4);
            let blend = target.and_then(|t| t.blend);
            if blend.is_some() {
                blend_mask |= 1 << i;
            }
            let b = blend.unwrap_or(wgt::BlendState::REPLACE);
            blends.push(dk::DkBlendState::new(
                blend_op(b.color.operation),
                blend_factor(b.color.src_factor),
                blend_factor(b.color.dst_factor),
                blend_op(b.alpha.operation),
                blend_factor(b.alpha.src_factor),
                blend_factor(b.alpha.dst_factor),
            ));
        }

        let keep = [
            dk::DkStencilOp_Keep,
            dk::DkStencilOp_Keep,
            dk::DkStencilOp_Keep,
            dk::DkCompareOp_Always,
        ];
        let (depth_stencil, stencil) = match &desc.depth_stencil {
            None => (
                dk::DkDepthStencilState::new(false, false, false, dk::DkCompareOp_Always, keep, keep),
                None,
            ),
            Some(ds) => {
                let on = ds.stencil.is_enabled();
                let compare = ds.depth_compare.unwrap_or(wgt::CompareFunction::Always);
                (
                    dk::DkDepthStencilState::new(
                        compare != wgt::CompareFunction::Always || ds.depth_write_enabled.unwrap_or(false),
                        ds.depth_write_enabled.unwrap_or(false),
                        on,
                        super::texture::compare_op(compare),
                        stencil_face(&ds.stencil.front),
                        stencil_face(&ds.stencil.back),
                    ),
                    on.then_some((ds.stencil.write_mask as u8, ds.stencil.read_mask as u8)),
                )
            }
        };

        Ok(Self {
            shaders,
            keys: [vs.key, fs.as_ref().map_or(vs.key, |f| f.key)],
            samplers: [
                stage_samplers(&vs),
                fs.as_ref().map_or_else(Vec::new, stage_samplers),
            ],
            primitive,
            restart,
            rasterizer,
            depth_bias: bias.map(|b| (b.constant as f32, b.clamp, b.slope_scale)),
            color: dk::DkColorState::new(blend_mask),
            color_write: dk::DkColorWriteState { masks: write_masks },
            blends,
            depth_stencil,
            stencil,
            attribs,
            vertex_buffers,
        })
    }

    /// Releases the shaders; their code goes back to the pool if unshared.
    pub(super) fn destroy(self, cache: &mut ShaderCache) {
        let keys = self.keys;
        drop(self);
        cache.trim(&keys);
    }
}
