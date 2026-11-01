//! deko3d backend (Nintendo Switch).
//!
//! S4 state: real device, queue, fences and swapchain; render passes bind
//! and clear swapchain images. Buffers are still CPU memory, other textures
//! have no image, and pipelines/draws are recorded as nothing (K1-K3).

#![allow(unused_variables)]

use alloc::{boxed::Box, collections::VecDeque, string::String, sync::Arc, vec, vec::Vec};
use core::{
    ffi::{c_char, c_void, CStr},
    ptr,
    sync::atomic::Ordering,
    time::Duration,
};

#[cfg(supports_64bit_atomics)]
use core::sync::atomic::AtomicU64;
#[cfg(not(supports_64bit_atomics))]
use portable_atomic::AtomicU64;

use deko3d_sys as dk;
use parking_lot::Mutex;

use crate::TlasInstance;

mod buffer;
pub use buffer::Buffer;
mod command;
pub use command::{CommandBuffer, CommandEncoder};
mod surface;
pub use surface::Surface;

#[derive(Clone, Debug)]
pub struct Api;

/// Owns the `DkDevice`; everything else holds an `Arc` to it.
struct DeviceShared {
    raw: dk::DkDevice,
}
// SAFETY: deko3d device calls used here are thread-safe (object creation).
unsafe impl Send for DeviceShared {}
unsafe impl Sync for DeviceShared {}

impl Drop for DeviceShared {
    fn drop(&mut self) {
        unsafe { dk::dkDeviceDestroy(self.raw) };
    }
}

/// The one graphics queue. deko3d queues are not thread-safe, so every use
/// goes through the lock.
struct QueueShared {
    raw: Mutex<dk::DkQueue>,
    _device: Arc<DeviceShared>,
}
unsafe impl Send for QueueShared {}
unsafe impl Sync for QueueShared {}

impl QueueShared {
    unsafe fn wait_idle(&self) {
        unsafe { dk::dkQueueWaitIdle(*self.raw.lock()) };
    }

    unsafe fn present(&self, swapchain: dk::DkSwapchain, texture: &Texture) {
        unsafe { dk::dkQueuePresentImage(*self.raw.lock(), swapchain, texture.slot) };
    }
}

impl Drop for QueueShared {
    fn drop(&mut self) {
        let raw = *self.raw.get_mut();
        unsafe {
            dk::dkQueueWaitIdle(raw);
            dk::dkQueueDestroy(raw);
        }
    }
}

pub struct Instance {
    shared: Arc<DeviceShared>,
}
pub struct Adapter {
    shared: Arc<DeviceShared>,
}
pub struct Device {
    shared: Arc<DeviceShared>,
    queue: Arc<QueueShared>,
}
pub struct Queue {
    shared: Arc<QueueShared>,
}

macro_rules! opaque_debug {
    ($($t:ty),*) => {$(
        impl core::fmt::Debug for $t {
            fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                f.write_str(stringify!($t))
            }
        }
    )*};
}
opaque_debug!(
    Instance,
    Adapter,
    Device,
    Queue,
    Texture,
    TextureView,
    Fence
);

/// A texture. Only swapchain images have a `DkImage` so far (`image` is null
/// otherwise); `slot`/`acquire` are meaningful only for those.
pub struct Texture {
    image: *const dk::DkImage,
    slot: i32,
    acquire: dk::DkFence,
}
unsafe impl Send for Texture {}
unsafe impl Sync for Texture {}

pub struct TextureView {
    image: *const dk::DkImage,
}
unsafe impl Send for TextureView {}
unsafe impl Sync for TextureView {}

#[derive(Debug)]
pub struct Resource;

/// Signalled by queue submits. Each submit pushes one boxed `DkFence`
/// (deko3d writes through the pointer) with the wgpu value it completes.
pub struct Fence {
    done: AtomicU64,
    pending: Mutex<VecDeque<(crate::FenceValue, Box<dk::DkFence>)>>,
}

impl Fence {
    /// Retires signalled submits; returns the last completed value.
    fn poll(&self) -> crate::FenceValue {
        let mut pending = self.pending.lock();
        while let Some((value, fence)) = pending.front_mut() {
            if unsafe { dk::dkFenceWait(&mut **fence, 0) } != dk::DkResult_Success {
                break;
            }
            self.done.store(*value, Ordering::Release);
            pending.pop_front();
        }
        self.done.load(Ordering::Acquire)
    }
}

type DeviceResult<T> = Result<T, crate::DeviceError>;

impl crate::Api for Api {
    const VARIANT: wgt::Backend = wgt::Backend::Deko3d;

    type Instance = Instance;
    type Surface = Surface;
    type Adapter = Adapter;
    type Device = Device;

    type Queue = Queue;
    type CommandEncoder = CommandEncoder;
    type CommandBuffer = CommandBuffer;

    type Buffer = Buffer;
    type Texture = Texture;
    type SurfaceTexture = Texture;
    type TextureView = TextureView;
    type Sampler = Resource;
    type QuerySet = Resource;
    type Fence = Fence;
    type AccelerationStructure = Resource;
    type PipelineCache = Resource;

    type BindGroupLayout = Resource;
    type BindGroup = Resource;
    type PipelineLayout = Resource;
    type ShaderModule = Resource;
    type RenderPipeline = Resource;
    type RayTracingPipeline = Resource;
    type ComputePipeline = Resource;
}

crate::impl_dyn_resource!(
    Adapter,
    Buffer,
    CommandBuffer,
    CommandEncoder,
    Device,
    Fence,
    Instance,
    Queue,
    Resource,
    Surface,
    Texture,
    TextureView
);

impl crate::DynAccelerationStructure for Resource {}
impl crate::DynBindGroup for Resource {}
impl crate::DynBindGroupLayout for Resource {}
impl crate::DynBuffer for Buffer {}
impl crate::DynCommandBuffer for CommandBuffer {}
impl crate::DynComputePipeline for Resource {}
impl crate::DynFence for Fence {}
impl crate::DynPipelineCache for Resource {}
impl crate::DynPipelineLayout for Resource {}
impl crate::DynQuerySet for Resource {}
impl crate::DynRenderPipeline for Resource {}
impl crate::DynRayTracingPipeline for Resource {}
impl crate::DynSampler for Resource {}
impl crate::DynShaderModule for Resource {}
impl crate::DynSurfaceTexture for Texture {}
impl crate::DynTexture for Texture {}
impl crate::DynTextureView for TextureView {}

impl core::borrow::Borrow<dyn crate::DynTexture> for Texture {
    fn borrow(&self) -> &dyn crate::DynTexture {
        self
    }
}

unsafe extern "C" fn debug_callback(
    _user: *mut c_void,
    context: *const c_char,
    result: dk::DkResult,
    message: *const c_char,
) {
    let text = |p: *const c_char| {
        if p.is_null() {
            "?".into()
        } else {
            unsafe { CStr::from_ptr(p) }.to_string_lossy()
        }
    };
    log::error!("deko3d: {} ({result}): {}", text(context), text(message));
}

impl crate::Instance for Instance {
    type A = Api;

    unsafe fn init(desc: &crate::InstanceDescriptor<'_>) -> Result<Self, crate::InstanceError> {
        if !cfg!(target_os = "horizon") {
            return Err(crate::InstanceError::new(String::from(
                "deko3d: only available on Horizon",
            )));
        }
        let maker = dk::DkDeviceMaker {
            cbDebug: Some(debug_callback),
            ..Default::default()
        };
        let raw = unsafe { dk::dkDeviceCreate(&maker) };
        if raw.is_null() {
            return Err(crate::InstanceError::new(String::from(
                "deko3d: dkDeviceCreate failed",
            )));
        }
        Ok(Instance {
            shared: Arc::new(DeviceShared { raw }),
        })
    }

    unsafe fn create_surface(
        &self,
        _display_handle: raw_window_handle::RawDisplayHandle,
        window_handle: raw_window_handle::RawWindowHandle,
    ) -> Result<Surface, crate::InstanceError> {
        // platform_nx passes libnx's NWindow as an AndroidNdk handle.
        let raw_window_handle::RawWindowHandle::AndroidNdk(handle) = window_handle else {
            return Err(crate::InstanceError::new(String::from(
                "deko3d: expected an NWindow (AndroidNdk) window handle",
            )));
        };
        Ok(Surface {
            window: handle.a_native_window.as_ptr(),
            swapchain: Mutex::new(None),
        })
    }

    unsafe fn enumerate_adapters(
        &self,
        _surface_hint: Option<&Surface>,
    ) -> Vec<crate::ExposedAdapter<Api>> {
        vec![crate::ExposedAdapter {
            adapter: Adapter {
                shared: Arc::clone(&self.shared),
            },
            info: adapter_info(),
            features: features(),
            capabilities: capabilities(),
        }]
    }
}

pub fn adapter_info() -> wgt::AdapterInfo {
    wgt::AdapterInfo {
        name: String::from("Tegra X1 (deko3d)"),
        driver: String::from("deko3d"),
        ..wgt::AdapterInfo::new(wgt::DeviceType::IntegratedGpu, wgt::Backend::Deko3d)
    }
}

/// What the game requests (`bootstrap/src/plugins.rs`) plus what deko3d
/// plainly has. F2 audits this list against the finished backend.
fn features() -> wgt::Features {
    wgt::Features::TEXTURE_COMPRESSION_BC
        | wgt::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES
        | wgt::Features::DEPTH32FLOAT_STENCIL8
        | wgt::Features::DEPTH_CLIP_CONTROL
        | wgt::Features::POLYGON_MODE_LINE
        | wgt::Features::INDIRECT_FIRST_INSTANCE
}

/// Limits from the `DK_*` per-stage binding tables.
fn capabilities() -> crate::Capabilities {
    let limits = wgt::Limits {
        max_sampled_textures_per_shader_stage: dk::DK_NUM_TEXTURE_BINDINGS,
        max_samplers_per_shader_stage: dk::DK_NUM_TEXTURE_BINDINGS,
        max_uniform_buffers_per_shader_stage: dk::DK_NUM_UNIFORM_BUFS,
        max_storage_buffers_per_shader_stage: dk::DK_NUM_STORAGE_BUFS,
        max_storage_textures_per_shader_stage: dk::DK_NUM_IMAGE_BINDINGS,
        max_uniform_buffer_binding_size: dk::DK_UNIFORM_BUF_MAX_SIZE as u64,
        max_color_attachments: dk::DK_MAX_RENDER_TARGETS,
        max_vertex_buffers: dk::DK_MAX_VERTEX_BUFFERS,
        max_vertex_attributes: dk::DK_MAX_VERTEX_ATTRIBS,
        ..wgt::Limits::default()
    };
    let four = wgt::BufferSize::new(4).unwrap();
    crate::Capabilities {
        limits,
        alignments: crate::Alignments {
            buffer_copy_offset: four,
            buffer_copy_pitch: four,
            uniform_bounds_check_alignment: four,
            raw_tlas_instance_size: 0,
            ray_tracing_scratch_buffer_alignment: 1,
            ray_tracing_pipeline_group_data_size: 1,
            ray_tracing_pipeline_group_data_alignment: 1,
            ray_tracing_pipeline_data_offset_alignment: 1,
        },
        downlevel: wgt::DownlevelCapabilities {
            flags: wgt::DownlevelFlags::all(),
            limits: wgt::DownlevelLimits {},
            shader_model: wgt::ShaderModel::Sm5,
        },
        cooperative_matrix_properties: Vec::new(),
    }
}

impl crate::Adapter for Adapter {
    type A = Api;

    unsafe fn open(
        &self,
        features: wgt::Features,
        _limits: &wgt::Limits,
        _memory_hints: &wgt::MemoryHints,
    ) -> DeviceResult<crate::OpenDevice<Api>> {
        let raw = unsafe { dk::dkQueueCreate(&dk::DkQueueMaker::new(self.shared.raw)) };
        if raw.is_null() {
            return Err(crate::DeviceError::Unexpected);
        }
        let queue = Arc::new(QueueShared {
            raw: Mutex::new(raw),
            _device: Arc::clone(&self.shared),
        });
        Ok(crate::OpenDevice {
            device: Device {
                shared: Arc::clone(&self.shared),
                queue: Arc::clone(&queue),
            },
            queue: Queue { shared: queue },
        })
    }

    unsafe fn texture_format_capabilities(
        &self,
        format: wgt::TextureFormat,
    ) -> crate::TextureFormatCapabilities {
        use crate::TextureFormatCapabilities as Tfc;
        // ponytail: blanket answer per format class; F2 replaces it with a
        // per-format table from the deko3d format list.
        let copy = Tfc::COPY_SRC | Tfc::COPY_DST;
        if format.is_compressed() {
            Tfc::SAMPLED | Tfc::SAMPLED_LINEAR | copy
        } else if format.is_depth_stencil_format() {
            Tfc::SAMPLED | Tfc::DEPTH_STENCIL_ATTACHMENT | Tfc::MULTISAMPLE_X4 | copy
        } else {
            Tfc::SAMPLED
                | Tfc::SAMPLED_LINEAR
                | Tfc::COLOR_ATTACHMENT
                | Tfc::COLOR_ATTACHMENT_BLEND
                | Tfc::MULTISAMPLE_X4
                | Tfc::MULTISAMPLE_RESOLVE
                | copy
        }
    }

    unsafe fn surface_capabilities(&self, surface: &Surface) -> Option<crate::SurfaceCapabilities> {
        let formats = [
            wgt::TextureFormat::Rgba8UnormSrgb,
            wgt::TextureFormat::Rgba8Unorm,
        ]
        .into_iter()
        .map(|format| wgt::SurfaceFormatCapabilities {
            format,
            color_spaces: wgt::SurfaceColorSpaces::SRGB,
        })
        .collect();
        Some(crate::SurfaceCapabilities {
            formats,
            maximum_frame_latency: 1..=2,
            current_extent: None,
            usage: wgt::TextureUses::COLOR_TARGET,
            present_modes: vec![wgt::PresentMode::Fifo, wgt::PresentMode::Immediate],
            composite_alpha_modes: vec![wgt::CompositeAlphaMode::Opaque],
        })
    }

    unsafe fn get_presentation_timestamp(&self) -> wgt::PresentationTimestamp {
        wgt::PresentationTimestamp::INVALID_TIMESTAMP
    }

    fn get_ordered_buffer_usages(&self) -> wgt::BufferUses {
        wgt::BufferUses::INCLUSIVE | wgt::BufferUses::MAP_WRITE
    }

    fn get_ordered_texture_usages(&self) -> wgt::TextureUses {
        wgt::TextureUses::INCLUSIVE
            | wgt::TextureUses::COLOR_TARGET
            | wgt::TextureUses::DEPTH_STENCIL_WRITE
    }
}

impl crate::Queue for Queue {
    type A = Api;

    unsafe fn submit(
        &self,
        command_buffers: &[&CommandBuffer],
        surface_textures: &[&Texture],
        (fence, fence_value): (&Fence, crate::FenceValue),
    ) -> DeviceResult<()> {
        let queue = *self.shared.raw.lock();
        // Rendering into a swapchain image waits for its acquire.
        for texture in surface_textures {
            unsafe {
                dk::dkQueueWaitFence(
                    queue,
                    ptr::from_ref::<dk::DkFence>(&texture.acquire).cast_mut(),
                )
            };
        }
        for cb in command_buffers {
            // SAFETY: caller orders this against other access to the buffers.
            unsafe { cb.submit(queue) };
        }
        let mut signal = Box::new(dk::DkFence::zeroed());
        unsafe { dk::dkQueueSignalFence(queue, &mut *signal, true) };
        fence.pending.lock().push_back((fence_value, signal));
        if unsafe { dk::dkQueueIsInErrorState(queue) } {
            return Err(crate::DeviceError::Lost);
        }
        Ok(())
    }

    unsafe fn present(
        &self,
        surface: &Surface,
        texture: Texture,
    ) -> Result<(), crate::SurfaceError> {
        let guard = surface.swapchain.lock();
        let sc = guard.as_ref().ok_or(crate::SurfaceError::Outdated)?;
        unsafe { self.shared.present(sc.raw, &texture) };
        Ok(())
    }

    unsafe fn get_timestamp_period(&self) -> f32 {
        1.0
    }

    unsafe fn wait_for_idle(&self) -> Result<(), crate::DeviceError> {
        unsafe { self.shared.wait_idle() };
        Ok(())
    }
}

impl crate::Device for Device {
    type A = Api;

    unsafe fn create_buffer(&self, desc: &crate::BufferDescriptor) -> DeviceResult<Buffer> {
        Buffer::new(desc)
    }

    unsafe fn destroy_buffer(&self, buffer: Buffer) {}
    unsafe fn add_raw_buffer(&self, _buffer: &Buffer) {}

    unsafe fn map_buffer(
        &self,
        buffer: &Buffer,
        range: crate::MemoryRange,
    ) -> DeviceResult<crate::BufferMapping> {
        Ok(crate::BufferMapping {
            ptr: ptr::NonNull::new(buffer.get_slice_ptr(range).cast::<u8>()).unwrap(),
            is_coherent: true,
        })
    }
    unsafe fn unmap_buffer(&self, buffer: &Buffer) {}
    unsafe fn flush_mapped_ranges<I>(&self, buffer: &Buffer, ranges: I) {}
    unsafe fn invalidate_mapped_ranges<I>(&self, buffer: &Buffer, ranges: I) {}

    unsafe fn create_texture(&self, desc: &crate::TextureDescriptor) -> DeviceResult<Texture> {
        // K1 gives these a DkImage.
        Ok(Texture {
            image: ptr::null(),
            slot: -1,
            acquire: dk::DkFence::zeroed(),
        })
    }
    unsafe fn destroy_texture(&self, texture: Texture) {}
    unsafe fn add_raw_texture(&self, _texture: &Texture) {}

    unsafe fn create_texture_view(
        &self,
        texture: &Texture,
        desc: &crate::TextureViewDescriptor,
    ) -> DeviceResult<TextureView> {
        Ok(TextureView {
            image: texture.image,
        })
    }
    unsafe fn destroy_texture_view(&self, view: TextureView) {}
    unsafe fn create_sampler(&self, desc: &crate::SamplerDescriptor) -> DeviceResult<Resource> {
        Ok(Resource)
    }
    unsafe fn destroy_sampler(&self, sampler: Resource) {}

    unsafe fn create_command_encoder(
        &self,
        desc: &crate::CommandEncoderDescriptor<Queue>,
    ) -> DeviceResult<CommandEncoder> {
        CommandEncoder::new(self.shared.raw)
    }

    unsafe fn create_bind_group_layout(
        &self,
        desc: &crate::BindGroupLayoutDescriptor,
    ) -> DeviceResult<Resource> {
        Ok(Resource)
    }
    unsafe fn destroy_bind_group_layout(&self, bg_layout: Resource) {}
    unsafe fn create_pipeline_layout(
        &self,
        desc: &crate::PipelineLayoutDescriptor<Resource>,
    ) -> DeviceResult<Resource> {
        Ok(Resource)
    }
    unsafe fn destroy_pipeline_layout(&self, pipeline_layout: Resource) {}
    unsafe fn create_bind_group(
        &self,
        desc: &crate::BindGroupDescriptor<Resource, Buffer, Resource, TextureView, Resource>,
    ) -> DeviceResult<Resource> {
        Ok(Resource)
    }
    unsafe fn destroy_bind_group(&self, group: Resource) {}

    unsafe fn create_shader_module(
        &self,
        desc: &crate::ShaderModuleDescriptor,
        shader: crate::ShaderInput,
    ) -> Result<Resource, crate::ShaderError> {
        Ok(Resource)
    }
    unsafe fn destroy_shader_module(&self, module: Resource) {}
    unsafe fn create_render_pipeline(
        &self,
        desc: &crate::RenderPipelineDescriptor<Resource, Resource, Resource>,
    ) -> Result<Resource, crate::PipelineError> {
        Ok(Resource)
    }
    unsafe fn destroy_render_pipeline(&self, pipeline: Resource) {}
    unsafe fn create_compute_pipeline(
        &self,
        desc: &crate::ComputePipelineDescriptor<Resource, Resource, Resource>,
    ) -> Result<Resource, crate::PipelineError> {
        Ok(Resource)
    }
    unsafe fn destroy_compute_pipeline(&self, pipeline: Resource) {}
    unsafe fn create_ray_tracing_pipeline(
        &self,
        desc: &crate::RayTracingPipelineDescriptor<Resource, Resource, Resource>,
    ) -> Result<Resource, crate::PipelineError> {
        Ok(Resource)
    }
    unsafe fn destroy_ray_tracing_pipeline(&self, pipeline: Resource) {}
    unsafe fn get_raytracing_pipeline_group_data(
        &self,
        pipeline: &Resource,
        groups: core::ops::Range<u32>,
    ) -> Result<Vec<u8>, crate::DeviceError> {
        Ok(vec![0; groups.count()])
    }
    unsafe fn create_pipeline_cache(
        &self,
        desc: &crate::PipelineCacheDescriptor<'_>,
    ) -> Result<Resource, crate::PipelineCacheError> {
        Ok(Resource)
    }
    unsafe fn destroy_pipeline_cache(&self, cache: Resource) {}

    unsafe fn create_query_set(
        &self,
        desc: &wgt::QuerySetDescriptor<crate::Label>,
    ) -> DeviceResult<Resource> {
        Ok(Resource)
    }
    unsafe fn destroy_query_set(&self, set: Resource) {}
    unsafe fn create_fence(&self) -> DeviceResult<Fence> {
        Ok(Fence {
            done: AtomicU64::new(0),
            pending: Mutex::new(VecDeque::new()),
        })
    }
    unsafe fn destroy_fence(&self, fence: Fence) {}
    unsafe fn get_fence_value(&self, fence: &Fence) -> DeviceResult<crate::FenceValue> {
        Ok(fence.poll())
    }
    unsafe fn wait(
        &self,
        fence: &Fence,
        value: crate::FenceValue,
        timeout: Option<Duration>,
    ) -> DeviceResult<bool> {
        if fence.poll() >= value {
            return Ok(true);
        }
        let timeout_ns = timeout.map_or(-1, |t| i64::try_from(t.as_nanos()).unwrap_or(i64::MAX));
        // ponytail: holds the fence lock while blocked, so pollers on other
        // threads wait too. Fine with one render thread.
        let result = {
            let mut pending = fence.pending.lock();
            let Some((_, signal)) = pending.iter_mut().find(|(v, _)| *v >= value) else {
                return Ok(false);
            };
            unsafe { dk::dkFenceWait(&mut **signal, timeout_ns) }
        };
        match result {
            dk::DkResult_Success => {
                fence.poll();
                Ok(true)
            }
            dk::DkResult_Timeout => Ok(false),
            _ => Err(crate::DeviceError::Lost),
        }
    }

    unsafe fn start_graphics_debugger_capture(&self) -> bool {
        false
    }
    unsafe fn stop_graphics_debugger_capture(&self) {}
    unsafe fn create_acceleration_structure(
        &self,
        desc: &crate::AccelerationStructureDescriptor,
    ) -> DeviceResult<Resource> {
        Ok(Resource)
    }
    unsafe fn get_acceleration_structure_build_sizes<'a>(
        &self,
        _desc: &crate::GetAccelerationStructureBuildSizesDescriptor<'a, Buffer>,
    ) -> crate::AccelerationStructureBuildSizes {
        Default::default()
    }
    unsafe fn get_acceleration_structure_device_address(
        &self,
        _acceleration_structure: &Resource,
    ) -> wgt::BufferAddress {
        Default::default()
    }
    unsafe fn destroy_acceleration_structure(&self, _acceleration_structure: Resource) {}

    fn tlas_instance_to_bytes(&self, instance: TlasInstance) -> Vec<u8> {
        vec![]
    }

    fn get_internal_counters(&self) -> wgt::HalCounters {
        Default::default()
    }

    fn check_if_oom(&self) -> DeviceResult<()> {
        Ok(())
    }
}
