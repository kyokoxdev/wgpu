use alloc::vec::Vec;
use core::{ffi::c_void, mem, ops::Range, ptr};

use deko3d_sys as dk;

use super::{Api, BindGroup, Buffer, DeviceResult, RenderPipeline, Resource, Texture, TextureView};
use crate::deko3d_shader::{PipelineLayout, MAX_STORAGE_BUFFERS, MAX_TEXTURES, MAX_UNIFORM_BUFFERS};

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

/// Vertex, then fragment: the two stages render pipelines have.
const STAGES: [(wgt::ShaderStages, dk::DkStage); 2] = [
    (wgt::ShaderStages::VERTEX, dk::DkStage_Vertex),
    (wgt::ShaderStages::FRAGMENT, dk::DkStage_Fragment),
];

/// What the current command list has bound, to drop redundant binds. The
/// pointers stay valid while encoding: wgpu-core keeps the resources alive.
struct Bound {
    pipeline: *const RenderPipeline,
    /// The set pipeline has no shaders: draws do nothing.
    skip: bool,
    groups: [*const BindGroup; crate::MAX_BIND_GROUPS],
    /// Image descriptor per texture slot, from the bound groups.
    images: [u32; MAX_TEXTURES as usize],
    /// A group or pipeline changed: rebuild the texture handles at the next draw.
    textures_dirty: bool,
    handles: [[dk::DkResHandle; MAX_TEXTURES as usize]; 2],
    uniforms: [[dk::DkBufExtents; MAX_UNIFORM_BUFFERS as usize]; 2],
    storages: [[dk::DkBufExtents; MAX_STORAGE_BUFFERS as usize]; 2],
    vertex_buffers: [dk::DkBufExtents; dk::DK_MAX_VERTEX_BUFFERS as usize],
    index: (dk::DkGpuAddr, u32),
    index_format: wgt::IndexFormat,
    /// `None` until the first strip needs it.
    restart: Option<(bool, u32)>,
    stencil_ref: u8,
    depth_bias: Option<(f32, f32, f32)>,
}

impl Bound {
    fn new() -> Self {
        Self {
            pipeline: ptr::null(),
            skip: false,
            groups: [ptr::null(); crate::MAX_BIND_GROUPS],
            images: [0; MAX_TEXTURES as usize],
            textures_dirty: true,
            // No real handle is all ones (image and sampler at their maximum).
            handles: [[u32::MAX; MAX_TEXTURES as usize]; 2],
            uniforms: Default::default(),
            storages: Default::default(),
            vertex_buffers: Default::default(),
            index: (dk::DK_GPU_ADDR_INVALID, 0),
            index_format: wgt::IndexFormat::Uint16,
            restart: None,
            stencil_ref: 0,
            depth_bias: None,
        }
    }
}

/// Records straight into a `DkCmdBuf`. Buffer clears still run on the CPU at
/// submit, through the buffers' CPU mapping.
pub struct CommandEncoder {
    raw: dk::DkCmdBuf,
    mem: alloc::boxed::Box<CmdMem>,
    cpu: Vec<Command>,
    descriptor_sets: ((dk::DkGpuAddr, u32), (dk::DkGpuAddr, u32)),
    bound: Bound,
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
    pub(super) fn new(
        device: dk::DkDevice,
        descriptor_sets: ((dk::DkGpuAddr, u32), (dk::DkGpuAddr, u32)),
    ) -> DeviceResult<Self> {
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
            descriptor_sets,
            bound: Bound::new(),
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
        self.bound = Bound::new();
        let ((images, num_images), (samplers, num_samplers)) = self.descriptor_sets;
        unsafe {
            dk::dkCmdBufBindImageDescriptorSet(self.raw, images, num_images);
            dk::dkCmdBufBindSamplerDescriptorSet(self.raw, samplers, num_samplers);
        }
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

    unsafe fn transition_buffers<'a, T>(&mut self, mut barriers: T)
    where
        T: Iterator<Item = crate::BufferBarrier<'a, Buffer>>,
    {
        if barriers.any(|b| b.usage.from != b.usage.to) {
            self.barrier();
        }
    }

    unsafe fn transition_textures<'a, T>(&mut self, mut barriers: T)
    where
        T: Iterator<Item = crate::TextureBarrier<'a, Texture>>,
    {
        if barriers.any(|b| b.usage.from != b.usage.to) {
            self.barrier();
        }
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
        for r in regions {
            let size = u32::try_from(r.size.get()).expect("deko3d: buffer copy over 4 GiB");
            unsafe {
                dk::dkCmdBufCopyBuffer(
                    self.raw,
                    src.gpu_addr() + r.src_offset,
                    dst.gpu_addr() + r.dst_offset,
                    size,
                )
            };
        }
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
    ) where
        T: Iterator<Item = crate::TextureCopy>,
    {
        for r in regions {
            let src_view = src.mip_view(r.src_base.mip_level);
            let dst_view = dst.mip_view(r.dst_base.mip_level);
            let src_rect = src.copy_rect(&r.src_base, &r.size);
            let dst_rect = dst.copy_rect(&r.dst_base, &r.size);
            unsafe {
                dk::dkCmdBufCopyImage(self.raw, &src_view, &src_rect, &dst_view, &dst_rect, 0)
            };
        }
    }

    unsafe fn copy_buffer_to_texture<T>(&mut self, src: &Buffer, dst: &Texture, regions: T)
    where
        T: Iterator<Item = crate::BufferTextureCopy>,
    {
        for r in regions {
            let view = dst.mip_view(r.texture_base.mip_level);
            let rect = dst.copy_rect(&r.texture_base, &r.size);
            let buf = copy_buf(src, dst, &r);
            unsafe { dk::dkCmdBufCopyBufferToImage(self.raw, &buf, &view, &rect, 0) };
        }
    }

    unsafe fn copy_texture_to_buffer<T>(
        &mut self,
        src: &Texture,
        src_usage: wgt::TextureUses,
        dst: &Buffer,
        regions: T,
    ) where
        T: Iterator<Item = crate::BufferTextureCopy>,
    {
        for r in regions {
            let view = src.mip_view(r.texture_base.mip_level);
            let rect = src.copy_rect(&r.texture_base, &r.size);
            let buf = copy_buf(dst, src, &r);
            unsafe { dk::dkCmdBufCopyImageToBuffer(self.raw, &view, &rect, &buf, 0) };
        }
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
        let mut targets = [ptr::null::<dk::DkImageView>(); dk::DK_MAX_RENDER_TARGETS as usize];
        let mut count = 0;
        for (i, at) in desc.color_attachments.iter().enumerate() {
            if let Some(at) = at.as_ref() {
                targets[i] = &at.target.view.raw;
                count = i + 1;
            }
        }
        let depth_view = desc
            .depth_stencil_attachment
            .as_ref()
            .map(|ds| ds.target.view.raw);
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
        layout: &PipelineLayout,
        index: u32,
        group: &BindGroup,
        dynamic_offsets: &[wgt::DynamicOffset],
    ) {
        let base = layout.bases[index as usize];
        let (mut uniform, mut storage) = (base.uniform as usize, base.storage as usize);
        let mut offsets = dynamic_offsets.iter();
        for b in &group.buffers {
            let offset = if b.dynamic {
                offsets.next().map_or(0, |&o| u64::from(o))
            } else {
                0
            };
            let extents = dk::DkBufExtents {
                addr: b.addr + offset,
                size: b.size,
            };
            let slot = if b.uniform { &mut uniform } else { &mut storage };
            self.bind_buffer(b.uniform, *slot, b.stages, extents);
            *slot += 1;
        }
        let first = base.texture as usize;
        self.bound.images[first..first + group.textures.len()].copy_from_slice(&group.textures);
        for (i, &(image, stages)) in group.images.iter().enumerate() {
            let handle = dk::dkMakeImageHandle(image);
            for (flag, stage) in STAGES {
                if stages.contains(flag) {
                    let slot = u32::from(base.image) + i as u32;
                    unsafe { dk::dkCmdBufBindImages(self.raw, stage, slot, &handle, 1) };
                }
            }
        }
        self.bound.groups[index as usize] = group;
        self.bound.textures_dirty = true;
    }
    unsafe fn set_immediates(&mut self, layout: &PipelineLayout, offset_bytes: u32, data: &[u32]) {}

    unsafe fn insert_debug_marker(&mut self, label: &str) {}
    unsafe fn begin_debug_marker(&mut self, group_label: &str) {}
    unsafe fn end_debug_marker(&mut self) {}

    unsafe fn set_render_pipeline(&mut self, pipeline: &RenderPipeline) {
        // A skipped pipeline binds nothing, so `bound` still describes the GPU.
        let Some((vertex, fragment)) = &pipeline.shaders else {
            self.bound.skip = true;
            return;
        };
        self.bound.skip = false;
        if ptr::eq(self.bound.pipeline, pipeline) {
            return;
        }
        // SAFETY: non-null `bound.pipeline` is alive while encoding.
        let prev = unsafe { self.bound.pipeline.as_ref() };
        let raw = self.raw;
        let changed = |f: &dyn Fn(&RenderPipeline) -> bool| !prev.is_some_and(f);
        let same_shaders = prev.and_then(|p| p.shaders.as_ref()).is_some_and(|(v, f)| {
            alloc::sync::Arc::ptr_eq(v, vertex)
                && match (f, fragment) {
                    (Some(a), Some(b)) => alloc::sync::Arc::ptr_eq(a, b),
                    (None, None) => true,
                    _ => false,
                }
        });
        unsafe {
            if !same_shaders {
                let shaders = [vertex.raw(), fragment.as_ref().map_or(ptr::null(), |f| f.raw())];
                let count = if fragment.is_some() { 2 } else { 1 };
                dk::dkCmdBufBindShaders(raw, dk::DkStageFlag_GraphicsMask, shaders.as_ptr(), count);
            }
            if changed(&|p| p.rasterizer == pipeline.rasterizer) {
                dk::dkCmdBufBindRasterizerState(raw, &pipeline.rasterizer);
            }
            if changed(&|p| p.color == pipeline.color) {
                dk::dkCmdBufBindColorState(raw, &pipeline.color);
            }
            if changed(&|p| p.color_write == pipeline.color_write) {
                dk::dkCmdBufBindColorWriteState(raw, &pipeline.color_write);
            }
            if !pipeline.blends.is_empty() && changed(&|p| p.blends == pipeline.blends) {
                dk::dkCmdBufBindBlendStates(raw, 0, pipeline.blends.as_ptr(), pipeline.blends.len() as u32);
            }
            if changed(&|p| p.depth_stencil == pipeline.depth_stencil) {
                dk::dkCmdBufBindDepthStencilState(raw, &pipeline.depth_stencil);
            }
            if changed(&|p| p.attribs == pipeline.attribs) {
                dk::dkCmdBufBindVtxAttribState(raw, pipeline.attribs.as_ptr(), pipeline.attribs.len() as u32);
            }
            if changed(&|p| p.vertex_buffers == pipeline.vertex_buffers) {
                dk::dkCmdBufBindVtxBufferState(
                    raw,
                    pipeline.vertex_buffers.as_ptr(),
                    pipeline.vertex_buffers.len() as u32,
                );
            }
            if let Some(bias) = pipeline.depth_bias.filter(|&b| self.bound.depth_bias != Some(b)) {
                dk::dkCmdBufSetDepthBias(raw, bias.0, bias.1, bias.2);
                self.bound.depth_bias = Some(bias);
            }
        }
        self.bound.pipeline = pipeline;
        self.bound.textures_dirty = true;
        self.apply_stencil();
    }

    unsafe fn set_index_buffer<'a>(
        &mut self,
        binding: crate::BufferBinding<'a, Buffer>,
        format: wgt::IndexFormat,
    ) {
        let addr = binding.buffer.gpu_addr() + binding.offset;
        let dk_format = match format {
            wgt::IndexFormat::Uint16 => dk::DkIdxFormat_Uint16,
            wgt::IndexFormat::Uint32 => dk::DkIdxFormat_Uint32,
        };
        self.bound.index_format = format;
        if self.bound.index != (addr, dk_format) {
            unsafe { dk::dkCmdBufBindIdxBuffer(self.raw, dk_format, addr) };
            self.bound.index = (addr, dk_format);
        }
    }
    unsafe fn set_vertex_buffer<'a>(
        &mut self,
        index: u32,
        binding: crate::BufferBinding<'a, Buffer>,
    ) {
        let size = binding
            .size
            .map_or(binding.buffer.size() as u64 - binding.offset, wgt::BufferSize::get);
        let extents = dk::DkBufExtents {
            addr: binding.buffer.gpu_addr() + binding.offset,
            size: size as u32,
        };
        let bound = &mut self.bound.vertex_buffers[index as usize];
        if *bound != extents {
            *bound = extents;
            unsafe { dk::dkCmdBufBindVtxBuffers(self.raw, index, &extents, 1) };
        }
    }
    unsafe fn set_viewport(&mut self, rect: &crate::Rect<f32>, depth_range: Range<f32>) {
        let viewport = dk::DkViewport {
            x: rect.x,
            y: rect.y,
            width: rect.w,
            height: rect.h,
            near: depth_range.start,
            far: depth_range.end,
        };
        unsafe { dk::dkCmdBufSetViewports(self.raw, 0, &viewport, 1) };
    }
    unsafe fn set_scissor_rect(&mut self, rect: &crate::Rect<u32>) {
        let scissor = dk::DkScissor {
            x: rect.x,
            y: rect.y,
            width: rect.w,
            height: rect.h,
        };
        unsafe { dk::dkCmdBufSetScissors(self.raw, 0, &scissor, 1) };
    }
    unsafe fn set_stencil_reference(&mut self, value: u32) {
        self.bound.stencil_ref = value as u8;
        self.apply_stencil();
    }
    unsafe fn set_blend_constants(&mut self, color: &[f32; 4]) {
        unsafe { dk::dkCmdBufSetBlendConst(self.raw, color[0], color[1], color[2], color[3]) };
    }

    unsafe fn draw(
        &mut self,
        first_vertex: u32,
        vertex_count: u32,
        first_instance: u32,
        instance_count: u32,
    ) {
        let Some(prim) = self.prepare_draw(false) else {
            return;
        };
        unsafe {
            dk::dkCmdBufDraw(self.raw, prim, vertex_count, instance_count, first_vertex, first_instance)
        };
    }
    unsafe fn draw_indexed(
        &mut self,
        first_index: u32,
        index_count: u32,
        base_vertex: i32,
        first_instance: u32,
        instance_count: u32,
    ) {
        let Some(prim) = self.prepare_draw(true) else {
            return;
        };
        unsafe {
            dk::dkCmdBufDrawIndexed(
                self.raw,
                prim,
                index_count,
                instance_count,
                first_index,
                base_vertex,
                first_instance,
            )
        };
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
        let Some(prim) = self.prepare_draw(false) else {
            return;
        };
        for i in 0..u64::from(draw_count) {
            let addr = buffer.gpu_addr() + offset + i * size_of::<wgt::DrawIndirectArgs>() as u64;
            unsafe { dk::dkCmdBufDrawIndirect(self.raw, prim, addr) };
        }
    }
    unsafe fn draw_indexed_indirect(
        &mut self,
        buffer: &Buffer,
        offset: wgt::BufferAddress,
        draw_count: u32,
    ) {
        let Some(prim) = self.prepare_draw(true) else {
            return;
        };
        for i in 0..u64::from(draw_count) {
            let addr = buffer.gpu_addr() + offset + i * size_of::<wgt::DrawIndexedIndirectArgs>() as u64;
            unsafe { dk::dkCmdBufDrawIndexedIndirect(self.raw, prim, addr) };
        }
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
        }
    }
}

impl CommandEncoder {
    /// Orders everything before against everything after and drops the
    /// texture and shader caches.
    // ponytail: one full barrier per transition batch, whatever the usages;
    // narrow to Fragments/Tiles from the usage pair if P5 shows them.
    /// Binds to `slot` in each stage the entry is visible to.
    fn bind_buffer(
        &mut self,
        uniform: bool,
        slot: usize,
        stages: wgt::ShaderStages,
        extents: dk::DkBufExtents,
    ) {
        for (i, (flag, stage)) in STAGES.into_iter().enumerate() {
            if !stages.contains(flag) {
                continue;
            }
            let bound = if uniform {
                &mut self.bound.uniforms[i][slot]
            } else {
                &mut self.bound.storages[i][slot]
            };
            if *bound == extents {
                continue;
            }
            *bound = extents;
            unsafe {
                if uniform {
                    dk::dkCmdBufBindUniformBuffers(self.raw, stage, slot as u32, &extents, 1);
                } else {
                    dk::dkCmdBufBindStorageBuffers(self.raw, stage, slot as u32, &extents, 1);
                }
            }
        }
    }

    fn apply_stencil(&mut self) {
        // SAFETY: `bound.pipeline` is alive while encoding.
        let Some(pipeline) = (unsafe { self.bound.pipeline.as_ref() }) else {
            return;
        };
        if let Some((write, read)) = pipeline.stencil {
            let reference = self.bound.stencil_ref;
            unsafe { dk::dkCmdBufSetStencil(self.raw, dk::DkFace_FrontAndBack, write, reference, read) };
        }
    }

    /// Late binds for a draw: texture handles pair each slot's image with the
    /// sampler the current pipeline's GLSL uses it with. Returns the primitive,
    /// or `None` if the draw must be skipped.
    fn prepare_draw(&mut self, indexed: bool) -> Option<dk::DkPrimitive> {
        if self.bound.skip {
            return None;
        }
        // SAFETY: wgpu-core sets a pipeline before any draw, and it and the
        // groups stay alive while encoding.
        let pipeline = unsafe { &*self.bound.pipeline };
        if self.bound.textures_dirty {
            self.bound.textures_dirty = false;
            for (i, (_, stage)) in STAGES.into_iter().enumerate() {
                for &(slot, sampler) in &pipeline.samplers[i] {
                    let sampler = sampler.map_or(0, |(group, binding)| {
                        let group = self.bound.groups[group as usize];
                        unsafe { group.as_ref() }.map_or(0, |g| g.sampler(binding))
                    });
                    let handle = dk::dkMakeTextureHandle(self.bound.images[slot as usize], sampler);
                    let bound = &mut self.bound.handles[i][slot as usize];
                    if *bound != handle {
                        *bound = handle;
                        unsafe { dk::dkCmdBufBindTextures(self.raw, stage, slot.into(), &handle, 1) };
                    }
                }
            }
        }
        if indexed {
            let index = match self.bound.index_format {
                wgt::IndexFormat::Uint16 => 0xffff,
                wgt::IndexFormat::Uint32 => 0xffff_ffff,
            };
            let restart = (pipeline.restart, index);
            // Lists leave it as is unless a strip turned it on before.
            if self.bound.restart.map_or(pipeline.restart, |r| r != restart) {
                unsafe { dk::dkCmdBufSetPrimitiveRestart(self.raw, pipeline.restart, index) };
                self.bound.restart = Some(restart);
            }
        }
        Some(pipeline.primitive)
    }

    fn barrier(&mut self) {
        unsafe {
            dk::dkCmdBufBarrier(
                self.raw,
                dk::DkBarrier_Primitives,
                dk::DkInvalidateFlags_Image | dk::DkInvalidateFlags_Shader,
            )
        };
    }
}

/// The buffer side of a buffer/texture copy. wgpu gives the pitch in bytes
/// and the layer height in block rows; deko3d wants both in bytes.
fn copy_buf(buffer: &Buffer, texture: &Texture, r: &crate::BufferTextureCopy) -> dk::DkCopyBuf {
    let layout = &r.buffer_layout;
    let row = layout.bytes_per_row.unwrap_or_else(|| {
        let (bw, _) = texture.format.block_dimensions();
        let bytes = texture.format.block_copy_size(Some(r.texture_base.aspect.map())).unwrap_or(4);
        r.size.width.div_ceil(bw) * bytes
    });
    let (_, bh) = texture.format.block_dimensions();
    let rows = layout.rows_per_image.unwrap_or(r.size.height.div_ceil(bh));
    dk::DkCopyBuf {
        addr: buffer.gpu_addr() + layout.offset,
        rowLength: row,
        imageHeight: row * rows,
    }
}
