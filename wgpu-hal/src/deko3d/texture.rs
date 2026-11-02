//! K1: images, views, samplers and the device descriptor sets.
//!
//! Every non-swapchain texture is one `DkImage` in pooled image memory.
//! Sampled and storage views, and every sampler, take a slot in the device's
//! image/sampler descriptor sets; each command list binds the sets once and
//! passes `dkMakeTextureHandle(view, sampler)` handles to shaders.

use alloc::{boxed::Box, vec::Vec};
use core::ptr;

use deko3d_sys as dk;

use super::memory::{Allocation, Pool};

const STORAGE: wgt::TextureUses = wgt::TextureUses::STORAGE_READ_ONLY
    .union(wgt::TextureUses::STORAGE_WRITE_ONLY)
    .union(wgt::TextureUses::STORAGE_READ_WRITE)
    .union(wgt::TextureUses::STORAGE_ATOMIC);

/// Image descriptors (the handle has 20 bits for these).
const MAX_IMAGES: u32 = 8192;
/// Sampler descriptors (the handle has 12 bits for these).
const MAX_SAMPLERS: u32 = 4096;

pub(super) fn dk_format(format: wgt::TextureFormat) -> Option<dk::DkImageFormat> {
    use wgt::TextureFormat as F;
    Some(match format {
        F::R8Unorm => dk::DkImageFormat_R8_Unorm,
        F::Rg8Unorm => dk::DkImageFormat_RG8_Unorm,
        F::R16Float => dk::DkImageFormat_R16_Float,
        F::Rg16Float => dk::DkImageFormat_RG16_Float,
        F::R32Float => dk::DkImageFormat_R32_Float,
        F::R32Uint => dk::DkImageFormat_R32_Uint,
        F::Rg32Float => dk::DkImageFormat_RG32_Float,
        F::Rgba8Unorm => dk::DkImageFormat_RGBA8_Unorm,
        F::Rgba8UnormSrgb => dk::DkImageFormat_RGBA8_Unorm_sRGB,
        F::Rgba8Snorm => dk::DkImageFormat_RGBA8_Snorm,
        F::Bgra8Unorm => dk::DkImageFormat_BGRA8_Unorm,
        F::Bgra8UnormSrgb => dk::DkImageFormat_BGRA8_Unorm_sRGB,
        F::Rgb10a2Unorm => dk::DkImageFormat_RGB10A2_Unorm,
        F::Rg11b10Ufloat => dk::DkImageFormat_RG11B10_Float,
        F::Rgba16Float => dk::DkImageFormat_RGBA16_Float,
        F::Rgba32Float => dk::DkImageFormat_RGBA32_Float,
        F::Depth16Unorm => dk::DkImageFormat_Z16,
        F::Depth24Plus => dk::DkImageFormat_Z24X8,
        F::Depth24PlusStencil8 => dk::DkImageFormat_Z24S8,
        F::Depth32Float => dk::DkImageFormat_ZF32,
        F::Depth32FloatStencil8 => dk::DkImageFormat_ZF32_X24S8,
        F::Bc1RgbaUnorm => dk::DkImageFormat_RGBA_BC1,
        F::Bc1RgbaUnormSrgb => dk::DkImageFormat_RGBA_BC1_sRGB,
        F::Bc2RgbaUnorm => dk::DkImageFormat_RGBA_BC2,
        F::Bc2RgbaUnormSrgb => dk::DkImageFormat_RGBA_BC2_sRGB,
        F::Bc3RgbaUnorm => dk::DkImageFormat_RGBA_BC3,
        F::Bc3RgbaUnormSrgb => dk::DkImageFormat_RGBA_BC3_sRGB,
        F::Bc4RUnorm => dk::DkImageFormat_R_BC4_Unorm,
        F::Bc5RgUnorm => dk::DkImageFormat_RG_BC5_Unorm,
        F::Bc5RgSnorm => dk::DkImageFormat_RG_BC5_Snorm,
        F::Bc7RgbaUnorm => dk::DkImageFormat_RGBA_BC7_Unorm,
        F::Bc7RgbaUnormSrgb => dk::DkImageFormat_RGBA_BC7_Unorm_sRGB,
        _ => return None,
    })
}

fn ms_mode(samples: u32) -> dk::DkMsMode {
    match samples {
        2 => dk::DkMsMode_2x,
        4 => dk::DkMsMode_4x,
        8 => dk::DkMsMode_8x,
        _ => dk::DkMsMode_1x,
    }
}

/// The image type a texture is laid out as. Cube-compatible textures are
/// 2D arrays; deko3d lays a cubemap out the same way, and a view may
/// override the type to `Cubemap` because both have base type 2D.
fn layout_type(desc: &crate::TextureDescriptor) -> dk::DkImageType {
    let layers = desc.array_layer_count();
    match (desc.dimension, desc.sample_count > 1, layers > 1) {
        (wgt::TextureDimension::D1, _, false) => dk::DkImageType_1D,
        (wgt::TextureDimension::D1, _, true) => dk::DkImageType_1DArray,
        (wgt::TextureDimension::D3, _, _) => dk::DkImageType_3D,
        (_, true, false) => dk::DkImageType_2DMS,
        (_, true, true) => dk::DkImageType_2DMSArray,
        (_, false, false) => dk::DkImageType_2D,
        (_, false, true) => dk::DkImageType_2DArray,
    }
}

fn view_type(dim: wgt::TextureViewDimension, multisampled: bool) -> dk::DkImageType {
    use wgt::TextureViewDimension as D;
    match (dim, multisampled) {
        (D::D1, _) => dk::DkImageType_1D,
        (D::D2, false) => dk::DkImageType_2D,
        (D::D2, true) => dk::DkImageType_2DMS,
        (D::D2Array, false) => dk::DkImageType_2DArray,
        (D::D2Array, true) => dk::DkImageType_2DMSArray,
        (D::Cube, _) => dk::DkImageType_Cubemap,
        (D::CubeArray, _) => dk::DkImageType_CubemapArray,
        (D::D3, _) => dk::DkImageType_3D,
    }
}

/// A texture. `image` points into `owned` or, for swapchain images, into the
/// swapchain's array; `slot`/`acquire` are meaningful only for those.
pub struct Texture {
    pub(super) image: *const dk::DkImage,
    owned: Option<(Box<dk::DkImage>, Allocation)>,
    pub(super) format: wgt::TextureFormat,
    pub(super) dimension: wgt::TextureDimension,
    pub(super) samples: u32,
    pub(super) slot: i32,
    pub(super) acquire: dk::DkFence,
}
impl core::fmt::Debug for Texture {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Texture").field("format", &self.format).finish()
    }
}
unsafe impl Send for Texture {}
unsafe impl Sync for Texture {}

impl Texture {
    pub(super) fn swapchain(
        image: *const dk::DkImage,
        format: wgt::TextureFormat,
        slot: i32,
        acquire: dk::DkFence,
    ) -> Self {
        Self {
            image,
            owned: None,
            format,
            dimension: wgt::TextureDimension::D2,
            samples: 1,
            slot,
            acquire,
        }
    }

    pub(super) fn new(
        device: dk::DkDevice,
        pool: &mut Pool,
        desc: &crate::TextureDescriptor,
    ) -> Result<Self, crate::DeviceError> {
        let Some(format) = dk_format(desc.format) else {
            log::error!("deko3d: no image format for {:?}", desc.format);
            return Err(crate::DeviceError::Unexpected);
        };
        let mut maker = dk::DkImageLayoutMaker::new(device);
        maker.type_ = layout_type(desc);
        maker.format = format;
        maker.msMode = ms_mode(desc.sample_count);
        maker.mipLevels = desc.mip_level_count;
        maker.dimensions = [
            desc.size.width,
            desc.size.height,
            desc.size.depth_or_array_layers,
        ];
        if desc.usage.intersects(
            wgt::TextureUses::COLOR_TARGET
                | wgt::TextureUses::DEPTH_STENCIL_READ
                | wgt::TextureUses::DEPTH_STENCIL_WRITE,
        ) {
            maker.flags |= dk::DkImageFlags_UsageRender | dk::DkImageFlags_HwCompression;
        }
        if desc.usage.intersects(STORAGE) {
            maker.flags |= dk::DkImageFlags_UsageLoadStore;
        }
        let mut layout = dk::DkImageLayout::zeroed();
        let (size, align) = unsafe {
            dk::dkImageLayoutInitialize(&mut layout, &maker);
            (
                dk::dkImageLayoutGetSize(&layout),
                dk::dkImageLayoutGetAlignment(&layout),
            )
        };
        let mem = pool
            .alloc(size, align)
            .ok_or(crate::DeviceError::OutOfMemory)?;
        let mut image = Box::new(dk::DkImage::zeroed());
        unsafe { dk::dkImageInitialize(&mut *image, &layout, mem.block, mem.offset) };
        Ok(Self {
            image: ptr::from_ref(&*image),
            owned: Some((image, mem)),
            format: desc.format,
            dimension: desc.dimension,
            samples: desc.sample_count,
            slot: -1,
            acquire: dk::DkFence::zeroed(),
        })
    }

    /// Returns the image memory, for `destroy_texture`.
    pub(super) fn into_mem(self) -> Option<Allocation> {
        self.owned.map(|(_, mem)| mem)
    }

    /// The whole image at one mip level: what copies address, with the
    /// layer (or 3D slice) in the rect's z.
    pub(super) fn mip_view(&self, mip: u32) -> dk::DkImageView {
        let mut view = dk::DkImageView::new(self.image);
        view.mipLevelOffset = mip as u8;
        view
    }

    /// A copy rect at `base` of `size` texels.
    pub(super) fn copy_rect(
        &self,
        base: &crate::TextureCopyBase,
        size: &crate::CopyExtent,
    ) -> dk::DkImageRect {
        let z = match self.dimension {
            wgt::TextureDimension::D3 => base.origin.z,
            _ => base.array_layer,
        };
        dk::DkImageRect {
            x: base.origin.x,
            y: base.origin.y,
            z,
            width: size.width,
            height: size.height,
            depth: size.depth,
        }
    }
}

pub struct TextureView {
    pub(super) raw: dk::DkImageView,
    /// Slot in the image descriptor set, for sampled and storage views.
    pub(super) descriptor: Option<u32>,
}
impl core::fmt::Debug for TextureView {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("TextureView").field("descriptor", &self.descriptor).finish()
    }
}
unsafe impl Send for TextureView {}
unsafe impl Sync for TextureView {}

impl TextureView {
    pub(super) fn new(
        texture: &Texture,
        desc: &crate::TextureViewDescriptor,
        sets: &mut DescriptorSets,
    ) -> Result<Self, crate::DeviceError> {
        let mut raw = dk::DkImageView::new(texture.image);
        if texture.owned.is_some() {
            raw.type_ = view_type(desc.dimension, texture.samples > 1);
            raw.format = dk_format(desc.format).ok_or(crate::DeviceError::Unexpected)?;
            raw.mipLevelOffset = desc.range.base_mip_level as u8;
            raw.mipLevelCount = desc.range.mip_level_count.unwrap_or(0) as u8;
            if texture.dimension != wgt::TextureDimension::D3 {
                raw.layerOffset = desc.range.base_array_layer as u16;
                raw.layerCount = desc.range.array_layer_count.unwrap_or(0) as u16;
            }
            if desc.range.aspect == wgt::TextureAspect::StencilOnly {
                raw.dsSource = dk::DkDsSource_Stencil;
            }
        }
        let storage = desc.usage.intersects(STORAGE);
        let descriptor = if storage || desc.usage.contains(wgt::TextureUses::RESOURCE) {
            let slot = sets.images.take().ok_or(crate::DeviceError::OutOfMemory)?;
            unsafe { dk::dkImageDescriptorInitialize(sets.image_ptr(slot), &raw, storage, false) };
            Some(slot)
        } else {
            None
        };
        Ok(Self { raw, descriptor })
    }
}

#[derive(Debug)]
pub struct Sampler {
    pub(super) descriptor: u32,
}

fn filter(mode: wgt::FilterMode) -> u32 {
    match mode {
        wgt::FilterMode::Nearest => dk::DkFilter_Nearest,
        wgt::FilterMode::Linear => dk::DkFilter_Linear,
    }
}

fn wrap(mode: wgt::AddressMode) -> u32 {
    match mode {
        wgt::AddressMode::ClampToEdge => dk::DkWrapMode_ClampToEdge,
        wgt::AddressMode::Repeat => dk::DkWrapMode_Repeat,
        wgt::AddressMode::MirrorRepeat => dk::DkWrapMode_MirroredRepeat,
        wgt::AddressMode::ClampToBorder => dk::DkWrapMode_ClampToBorder,
    }
}

pub(super) fn compare_op(f: wgt::CompareFunction) -> u32 {
    use wgt::CompareFunction as C;
    match f {
        C::Never => dk::DkCompareOp_Never,
        C::Less => dk::DkCompareOp_Less,
        C::Equal => dk::DkCompareOp_Equal,
        C::LessEqual => dk::DkCompareOp_Lequal,
        C::Greater => dk::DkCompareOp_Greater,
        C::NotEqual => dk::DkCompareOp_NotEqual,
        C::GreaterEqual => dk::DkCompareOp_Gequal,
        C::Always => dk::DkCompareOp_Always,
    }
}

impl Sampler {
    pub(super) fn new(
        desc: &crate::SamplerDescriptor,
        sets: &mut DescriptorSets,
    ) -> Result<Self, crate::DeviceError> {
        let mut s = dk::DkSampler::default();
        s.minFilter = filter(desc.min_filter);
        s.magFilter = filter(desc.mag_filter);
        s.mipFilter = match desc.mipmap_filter {
            wgt::MipmapFilterMode::Nearest => dk::DkMipFilter_Nearest,
            wgt::MipmapFilterMode::Linear => dk::DkMipFilter_Linear,
        };
        s.wrapMode = desc.address_modes.map(wrap);
        s.lodClampMin = desc.lod_clamp.start;
        s.lodClampMax = desc.lod_clamp.end;
        if let Some(f) = desc.compare {
            s.compareEnable = true;
            s.compareOp = compare_op(f);
        }
        s.borderColor = match desc.border_color {
            Some(wgt::SamplerBorderColor::OpaqueBlack) => [0.0, 0.0, 0.0, 1.0],
            Some(wgt::SamplerBorderColor::OpaqueWhite) => [1.0; 4],
            _ => [0.0; 4],
        }
        .map(f32::to_bits);
        s.maxAnisotropy = f32::from(desc.anisotropy_clamp);
        let slot = sets.samplers.take().ok_or(crate::DeviceError::OutOfMemory)?;
        unsafe { dk::dkSamplerDescriptorInitialize(sets.sampler_ptr(slot), &s) };
        Ok(Self { descriptor: slot })
    }
}

/// A free list of descriptor slots.
struct Slots {
    free: Vec<u32>,
    next: u32,
    max: u32,
}

impl Slots {
    fn new(max: u32) -> Self {
        Self { free: Vec::new(), next: 0, max }
    }

    fn take(&mut self) -> Option<u32> {
        self.free.pop().or_else(|| {
            (self.next < self.max).then(|| {
                self.next += 1;
                self.next - 1
            })
        })
    }

    fn give(&mut self, slot: u32) {
        self.free.push(slot);
    }
}

/// The image and sampler descriptor sets: one CPU-visible block, images
/// first. Descriptors are written from the CPU at view/sampler creation.
// A reused slot can't be stale: deko3d invalidates the descriptor cache after
// every queue flush, and each submit flushes.
pub(super) struct DescriptorSets {
    mem: dk::DkMemBlock,
    cpu: *mut u8,
    images: Slots,
    samplers: Slots,
}

// SAFETY: only used behind the device's mutex.
unsafe impl Send for DescriptorSets {}

const IMAGE_SET_SIZE: u32 = MAX_IMAGES * 32;

impl DescriptorSets {
    pub fn new(device: dk::DkDevice) -> Option<Self> {
        let size = (IMAGE_SET_SIZE + MAX_SAMPLERS * 32).next_multiple_of(dk::DK_MEMBLOCK_ALIGNMENT);
        let mem = unsafe { dk::dkMemBlockCreate(&dk::DkMemBlockMaker::new(device, size)) };
        if mem.is_null() {
            return None;
        }
        Some(Self {
            mem,
            cpu: unsafe { dk::dkMemBlockGetCpuAddr(mem) }.cast(),
            images: Slots::new(MAX_IMAGES),
            samplers: Slots::new(MAX_SAMPLERS),
        })
    }

    fn image_ptr(&self, slot: u32) -> *mut dk::DkImageDescriptor {
        unsafe { self.cpu.add(slot as usize * 32) }.cast()
    }

    fn sampler_ptr(&self, slot: u32) -> *mut dk::DkSamplerDescriptor {
        unsafe { self.cpu.add((IMAGE_SET_SIZE + slot * 32) as usize) }.cast()
    }

    pub fn free_image(&mut self, slot: u32) {
        self.images.give(slot);
    }

    pub fn free_sampler(&mut self, slot: u32) {
        self.samplers.give(slot);
    }

    /// GPU addresses and sizes of the image and sampler sets, for
    /// `dkCmdBufBind{Image,Sampler}DescriptorSet`.
    pub fn sets(&self) -> ((dk::DkGpuAddr, u32), (dk::DkGpuAddr, u32)) {
        let gpu = unsafe { dk::dkMemBlockGetGpuAddr(self.mem) };
        (
            (gpu, MAX_IMAGES),
            (gpu + IMAGE_SET_SIZE as u64, MAX_SAMPLERS),
        )
    }
}

impl Drop for DescriptorSets {
    fn drop(&mut self) {
        unsafe { dk::dkMemBlockDestroy(self.mem) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slots_reuse_and_cap() {
        let mut s = Slots::new(3);
        assert_eq!([s.take(), s.take(), s.take(), s.take()], [Some(0), Some(1), Some(2), None]);
        s.give(1);
        assert_eq!(s.take(), Some(1));
        assert_eq!(s.take(), None);
    }
}
