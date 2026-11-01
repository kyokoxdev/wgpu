use alloc::vec::Vec;
use core::{ffi::c_void, mem, ops::Range, ptr};

use deko3d_sys as dk;

use super::{Api, Buffer, DeviceResult, Resource, Texture, TextureView};

/// Command memory chunk size; `cbAddMem` grabs another when one fills.
const CMD_CHUNK: u32 = 0x10000;

/// Command memory owned by one encoder. Boxed: deko3d calls back into it
/// through `userData`. Chunks are rewound (not freed) in `reset_all`.
struct CmdMem {
    device: dk::DkDevice,
    chunks: Vec<dk::DkMemBlock>,
    next: usize,
}

unsafe extern "C" fn add_mem(user: *mut c_void, cmdbuf: dk::DkCmdBuf, min: usize) {
    let mem = unsafe { &mut *user.cast::<CmdMem>() };
    let fits = |b: dk::DkMemBlock| unsafe { dk::dkMemBlockGetSize(b) } as usize >= min;
    if !mem.chunks.get(mem.next).is_some_and(|&b| fits(b)) {
        let size = (min as u32)
            .max(CMD_CHUNK)
            .next_multiple_of(dk::DK_MEMBLOCK_ALIGNMENT);
        let block = unsafe { dk::dkMemBlockCreate(&dk::DkMemBlockMaker::new(mem.device, size)) };
        // deko3d reports the failure through the debug callback; recording
        // past this point is undefined, so stop here.
        assert!(!block.is_null(), "deko3d: out of command memory");
        mem.chunks.insert(mem.next, block);
    }
    let block = mem.chunks[mem.next];
    mem.next += 1;
    unsafe { dk::dkCmdBufAddMemory(cmdbuf, block, 0, dk::dkMemBlockGetSize(block)) };
}

/// Records straight into a `DkCmdBuf`. Buffer commands still run on the CPU
/// at submit (buffers are CPU memory until K1).
pub struct CommandEncoder {
    raw: dk::DkCmdBuf,
    mem: alloc::boxed::Box<CmdMem>,
    cpu: Vec<Command>,
}

// SAFETY: the encoder is used from one thread at a time (`&mut self`).
unsafe impl Send for CommandEncoder {}
unsafe impl Sync for CommandEncoder {}

impl core::fmt::Debug for CommandEncoder {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("CommandEncoder")
    }
}

impl Drop for CommandEncoder {
    fn drop(&mut self) {
        // wgpu-core drops encoders only after their submissions retire.
        unsafe {
            dk::dkCmdBufDestroy(self.raw);
            for &block in &self.mem.chunks {
                dk::dkMemBlockDestroy(block);
            }
        }
    }
}

impl CommandEncoder {
    pub(super) fn new(device: dk::DkDevice) -> DeviceResult<Self> {
        let mut mem = alloc::boxed::Box::new(CmdMem {
            device,
            chunks: Vec::new(),
            next: 0,
        });
        let raw = unsafe {
            dk::dkCmdBufCreate(&dk::DkCmdBufMaker {
                device,
                userData: ptr::from_mut(&mut *mem).cast(),
                cbAddMem: Some(add_mem),
            })
        };
        if raw.is_null() {
            return Err(crate::DeviceError::Unexpected);
        }
        Ok(Self {
            raw,
            mem,
            cpu: Vec::new(),
        })
    }
}

/// A finished list plus the CPU-side buffer work recorded with it.
#[derive(Debug)]
pub struct CommandBuffer {
    list: dk::DkCmdList,
    cpu: Vec<Command>,
}

#[derive(Debug)]
enum Command {
    ClearBuffer {
        buffer: Buffer,
        range: crate::MemoryRange,
    },
    CopyBufferToBuffer {
        src: Buffer,
        dst: Buffer,
        regions: Vec<crate::BufferCopy>,
    },
}

impl CommandBuffer {
    /// # Safety
    ///
    /// Must be called with appropriate synchronization for the resources affected by the command,
    /// such as ensuring that buffers are not accessed by a command while aliasing references exist.
    pub(super) unsafe fn submit(&self, queue: dk::DkQueue) {
        for command in &self.cpu {
            unsafe { command.execute() };
        }
        if self.list != 0 {
            unsafe { dk::dkQueueSubmitCommands(queue, self.list) };
        }
    }
}

impl crate::CommandEncoder for CommandEncoder {
    type A = Api;

    unsafe fn begin_encoding(&mut self, label: crate::Label) -> DeviceResult<()> {
        assert!(self.cpu.is_empty());
        Ok(())
    }
    unsafe fn discard_encoding(&mut self) {
        // The list is dropped; its memory comes back in `reset_all`.
        unsafe { dk::dkCmdBufFinishList(self.raw) };
        self.cpu.clear();
    }
    unsafe fn end_encoding(&mut self) -> DeviceResult<CommandBuffer> {
        Ok(CommandBuffer {
            list: unsafe { dk::dkCmdBufFinishList(self.raw) },
            cpu: mem::take(&mut self.cpu),
        })
    }
    unsafe fn reset_all<I>(&mut self, command_buffers: I) {
        // Every list from this encoder has retired on the GPU.
        unsafe { dk::dkCmdBufClear(self.raw) };
        self.mem.next = 0;
    }

    unsafe fn transition_buffers<'a, T>(&mut self, barriers: T)
    where
        T: Iterator<Item = crate::BufferBarrier<'a, Buffer>>,
    {
    }

    unsafe fn transition_textures<'a, T>(&mut self, barriers: T)
    where
        T: Iterator<Item = crate::TextureBarrier<'a, Texture>>,
    {
    }

    unsafe fn clear_buffer(&mut self, buffer: &Buffer, range: crate::MemoryRange) {
        self.cpu.push(Command::ClearBuffer {
            buffer: buffer.clone(),
            range,
        })
    }

    unsafe fn copy_buffer_to_buffer<T>(&mut self, src: &Buffer, dst: &Buffer, regions: T)
    where
        T: Iterator<Item = crate::BufferCopy>,
    {
        self.cpu.push(Command::CopyBufferToBuffer {
            src: src.clone(),
            dst: dst.clone(),
            regions: regions.collect(),
        });
    }

    #[cfg(webgl)]
    unsafe fn copy_external_image_to_texture<T>(
        &mut self,
        src: &wgt::CopyExternalImageSourceInfo,
        dst: &Texture,
        dst_premultiplication: bool,
        regions: T,
    ) where
        T: Iterator<Item = crate::TextureCopy>,
    {
    }

    unsafe fn copy_texture_to_texture<T>(
        &mut self,
        src: &Texture,
        src_usage: wgt::TextureUses,
        dst: &Texture,
        regions: T,
    ) {
        // TODO: consider implementing this and other texture manipulation
    }

    unsafe fn copy_buffer_to_texture<T>(&mut self, src: &Buffer, dst: &Texture, regions: T) {
        // TODO: consider implementing this and other texture manipulation
    }

    unsafe fn copy_texture_to_buffer<T>(
        &mut self,
        src: &Texture,
        src_usage: wgt::TextureUses,
        dst: &Buffer,
        regions: T,
    ) {
        // TODO: consider implementing this and other texture manipulation
    }

    unsafe fn begin_query(&mut self, set: &Resource, index: u32) {}
    unsafe fn end_query(&mut self, set: &Resource, index: u32) {}
    unsafe fn write_timestamp(&mut self, set: &Resource, index: u32) {}
    unsafe fn read_acceleration_structure_compact_size(
        &mut self,
        acceleration_structure: &Resource,
        buf: &Buffer,
    ) {
    }
    unsafe fn reset_queries(&mut self, set: &Resource, range: Range<u32>) {}
    unsafe fn copy_query_results(
        &mut self,
        set: &Resource,
        range: Range<u32>,
        buffer: &Buffer,
        offset: wgt::BufferAddress,
        stride: wgt::BufferSize,
    ) {
    }

    // render

    unsafe fn begin_render_pass(
        &mut self,
        desc: &crate::RenderPassDescriptor<Resource, TextureView>,
    ) -> DeviceResult<()> {
        // S4: only targets with a DkImage (swapchain images) bind and clear.
        let mut views = [dk::DkImageView::new(ptr::null()); dk::DK_MAX_RENDER_TARGETS as usize];
        let mut targets = [ptr::null::<dk::DkImageView>(); dk::DK_MAX_RENDER_TARGETS as usize];
        let mut count = 0;
        for (i, at) in desc.color_attachments.iter().enumerate() {
            if let Some(at) = at.as_ref().filter(|at| !at.target.view.image.is_null()) {
                views[i] = dk::DkImageView::new(at.target.view.image);
                targets[i] = &views[i];
                count = i + 1;
            }
        }
        let depth_view = desc
            .depth_stencil_attachment
            .as_ref()
            .filter(|ds| !ds.target.view.image.is_null())
            .map(|ds| dk::DkImageView::new(ds.target.view.image));
        if count == 0 && depth_view.is_none() {
            return Ok(());
        }
        let depth_ptr = depth_view.as_ref().map_or(ptr::null(), ptr::from_ref);
        let (w, h) = (desc.extent.width, desc.extent.height);
        unsafe {
            dk::dkCmdBufBindRenderTargets(self.raw, targets.as_ptr(), count as u32, depth_ptr);
            dk::dkCmdBufSetViewports(
                self.raw,
                0,
                &dk::DkViewport {
                    x: 0.0,
                    y: 0.0,
                    width: w as f32,
                    height: h as f32,
                    near: 0.0,
                    far: 1.0,
                },
                1,
            );
            dk::dkCmdBufSetScissors(
                self.raw,
                0,
                &dk::DkScissor {
                    x: 0,
                    y: 0,
                    width: w,
                    height: h,
                },
                1,
            );
        }
        for (i, at) in desc.color_attachments.iter().enumerate().take(count) {
            let Some(at) = at.as_ref() else { continue };
            if targets[i].is_null() || !at.ops.contains(crate::AttachmentOps::LOAD_CLEAR) {
                continue;
            }
            let c = at.clear_value;
            let data = [c.r as f32, c.g as f32, c.b as f32, c.a as f32];
            unsafe {
                dk::dkCmdBufClearColor(
                    self.raw,
                    i as u32,
                    dk::DkColorMask_RGBA,
                    data.as_ptr().cast(),
                )
            };
        }
        if let Some(ds) = desc
            .depth_stencil_attachment
            .as_ref()
            .filter(|_| depth_view.is_some())
        {
            let depth = ds.depth_ops.contains(crate::AttachmentOps::LOAD_CLEAR);
            let stencil = ds.stencil_ops.contains(crate::AttachmentOps::LOAD_CLEAR);
            if depth || stencil {
                unsafe {
                    dk::dkCmdBufClearDepthStencil(
                        self.raw,
                        depth,
                        ds.clear_value.0,
                        if stencil { 0xff } else { 0 },
                        ds.clear_value.1 as u8,
                    )
                };
            }
        }
        Ok(())
    }
    unsafe fn end_render_pass(&mut self) {}

    unsafe fn set_bind_group(
        &mut self,
        layout: &Resource,
        index: u32,
        group: &Resource,
        dynamic_offsets: &[wgt::DynamicOffset],
    ) {
    }
    unsafe fn set_immediates(&mut self, layout: &Resource, offset_bytes: u32, data: &[u32]) {}

    unsafe fn insert_debug_marker(&mut self, label: &str) {}
    unsafe fn begin_debug_marker(&mut self, group_label: &str) {}
    unsafe fn end_debug_marker(&mut self) {}

    unsafe fn set_render_pipeline(&mut self, pipeline: &Resource) {}

    unsafe fn set_index_buffer<'a>(
        &mut self,
        binding: crate::BufferBinding<'a, Buffer>,
        format: wgt::IndexFormat,
    ) {
    }
    unsafe fn set_vertex_buffer<'a>(
        &mut self,
        index: u32,
        binding: crate::BufferBinding<'a, Buffer>,
    ) {
    }
    unsafe fn set_viewport(&mut self, rect: &crate::Rect<f32>, depth_range: Range<f32>) {}
    unsafe fn set_scissor_rect(&mut self, rect: &crate::Rect<u32>) {}
    unsafe fn set_stencil_reference(&mut self, value: u32) {}
    unsafe fn set_blend_constants(&mut self, color: &[f32; 4]) {}

    unsafe fn draw(
        &mut self,
        first_vertex: u32,
        vertex_count: u32,
        first_instance: u32,
        instance_count: u32,
    ) {
    }
    unsafe fn draw_indexed(
        &mut self,
        first_index: u32,
        index_count: u32,
        base_vertex: i32,
        first_instance: u32,
        instance_count: u32,
    ) {
    }
    unsafe fn draw_mesh_tasks(
        &mut self,
        group_count_x: u32,
        group_count_y: u32,
        group_count_z: u32,
    ) {
    }
    unsafe fn draw_indirect(
        &mut self,
        buffer: &Buffer,
        offset: wgt::BufferAddress,
        draw_count: u32,
    ) {
    }
    unsafe fn draw_indexed_indirect(
        &mut self,
        buffer: &Buffer,
        offset: wgt::BufferAddress,
        draw_count: u32,
    ) {
    }
    unsafe fn draw_mesh_tasks_indirect(
        &mut self,
        buffer: &<Self::A as crate::Api>::Buffer,
        offset: wgt::BufferAddress,
        draw_count: u32,
    ) {
    }
    unsafe fn draw_indirect_count(
        &mut self,
        buffer: &Buffer,
        offset: wgt::BufferAddress,
        count_buffer: &Buffer,
        count_offset: wgt::BufferAddress,
        max_count: u32,
    ) {
    }
    unsafe fn draw_indexed_indirect_count(
        &mut self,
        buffer: &Buffer,
        offset: wgt::BufferAddress,
        count_buffer: &Buffer,
        count_offset: wgt::BufferAddress,
        max_count: u32,
    ) {
    }
    unsafe fn draw_mesh_tasks_indirect_count(
        &mut self,
        buffer: &<Self::A as crate::Api>::Buffer,
        offset: wgt::BufferAddress,
        count_buffer: &<Self::A as crate::Api>::Buffer,
        count_offset: wgt::BufferAddress,
        max_count: u32,
    ) {
    }

    // compute

    unsafe fn begin_compute_pass(&mut self, desc: &crate::ComputePassDescriptor<Resource>) {}
    unsafe fn end_compute_pass(&mut self) {}

    unsafe fn set_compute_pipeline(&mut self, pipeline: &Resource) {}

    unsafe fn dispatch_workgroups(&mut self, count: [u32; 3]) {}
    unsafe fn dispatch_workgroups_indirect(&mut self, buffer: &Buffer, offset: wgt::BufferAddress) {
    }

    unsafe fn build_acceleration_structures<'a, T>(
        &mut self,
        _descriptor_count: u32,
        descriptors: T,
    ) where
        Api: 'a,
        T: IntoIterator<Item = crate::BuildAccelerationStructureDescriptor<'a, Buffer, Resource>>,
    {
    }

    unsafe fn place_acceleration_structure_barrier(
        &mut self,
        _barriers: crate::AccelerationStructureBarrier,
    ) {
    }

    unsafe fn copy_acceleration_structure_to_acceleration_structure(
        &mut self,
        src: &Resource,
        dst: &Resource,
        copy: wgt::AccelerationStructureCopy,
    ) {
    }

    unsafe fn set_acceleration_structure_dependencies(
        command_buffers: &[&CommandBuffer],
        dependencies: &[&Resource],
    ) {
    }

    unsafe fn begin_ray_tracing_pass(&mut self, _desc: &crate::RayTracingPassDescriptor) {}

    unsafe fn end_ray_tracing_pass(&mut self) {}

    unsafe fn set_ray_tracing_pipeline(
        &mut self,
        _pipeline: &<Self::A as crate::Api>::RayTracingPipeline,
    ) {
    }

    unsafe fn trace_rays(
        &mut self,
        _count: [u32; 3],
        _ray_generation_group_data: crate::PipelineGroupData<Buffer>,
        _miss_group_data: crate::PipelineGroupData<Buffer>,
        _intersection_group_data: crate::PipelineGroupData<Buffer>,
    ) {
    }
}

impl Command {
    /// # Safety
    ///
    /// Must be called with appropriate synchronization for the resources affected by the command,
    /// such as ensuring that buffers are not accessed by a command while aliasing references exist.
    unsafe fn execute(&self) {
        match self {
            Command::ClearBuffer { ref buffer, range } => {
                // SAFETY:
                // Caller is responsible for ensuring this does not alias.
                let buffer_slice: &mut [u8] = unsafe { &mut *buffer.get_slice_ptr(range.clone()) };
                buffer_slice.fill(0);
            }

            Command::CopyBufferToBuffer { src, dst, regions } => {
                for &crate::BufferCopy {
                    src_offset,
                    dst_offset,
                    size,
                } in regions
                {
                    // SAFETY:
                    // Caller is responsible for ensuring this does not alias.
                    let src_region: &[u8] =
                        unsafe { &*src.get_slice_ptr(src_offset..src_offset + size.get()) };
                    let dst_region: &mut [u8] =
                        unsafe { &mut *dst.get_slice_ptr(dst_offset..dst_offset + size.get()) };
                    dst_region.copy_from_slice(src_region);
                }
            }
        }
    }
}
