//! Swapchain over libnx's `NWindow` (S4).

use alloc::{boxed::Box, sync::Arc, vec::Vec};
use core::{ffi::c_void, ptr, time::Duration};

use deko3d_sys as dk;
use parking_lot::Mutex;

use super::{Api, Device, Fence, QueueShared, Texture};

pub struct Surface {
    pub(super) window: *mut c_void,
    pub(super) swapchain: Mutex<Option<Swapchain>>,
}

// SAFETY: the NWindow is a process-lifetime static; the swapchain is behind a lock.
unsafe impl Send for Surface {}
unsafe impl Sync for Surface {}

impl core::fmt::Debug for Surface {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("Surface")
    }
}

pub(super) struct Swapchain {
    pub(super) raw: dk::DkSwapchain,
    queue: Arc<QueueShared>,
    mem: dk::DkMemBlock,
    /// Boxed: deko3d keeps pointers to these.
    images: Box<[dk::DkImage]>,
}

impl Drop for Swapchain {
    fn drop(&mut self) {
        unsafe {
            self.queue.wait_idle();
            dk::dkSwapchainDestroy(self.raw);
            dk::dkMemBlockDestroy(self.mem);
        }
    }
}

pub(super) fn dk_format(format: wgt::TextureFormat) -> Option<dk::DkImageFormat> {
    Some(match format {
        wgt::TextureFormat::Rgba8UnormSrgb => dk::DkImageFormat_RGBA8_Unorm_sRGB,
        wgt::TextureFormat::Rgba8Unorm => dk::DkImageFormat_RGBA8_Unorm,
        _ => return None,
    })
}

const fn align_up(v: u64, a: u64) -> u64 {
    v.div_ceil(a) * a
}

impl crate::Surface for Surface {
    type A = Api;

    unsafe fn configure(
        &self,
        device: &Device,
        config: &crate::SurfaceConfiguration,
    ) -> Result<(), crate::SurfaceError> {
        let mut slot = self.swapchain.lock();
        *slot = None;

        let dev = device.shared.raw;
        let format = dk_format(config.format).ok_or(crate::SurfaceError::Other(
            "deko3d: unsupported swapchain format",
        ))?;
        let count = config.maximum_frame_latency.clamp(1, 2) + 1;

        let mut maker = dk::DkImageLayoutMaker::new(dev);
        maker.flags = dk::DkImageFlags_UsageRender
            | dk::DkImageFlags_UsagePresent
            | dk::DkImageFlags_HwCompression;
        maker.format = format;
        maker.dimensions = [config.extent.width, config.extent.height, 1];
        let mut layout = dk::DkImageLayout::zeroed();
        let (size, align) = unsafe {
            dk::dkImageLayoutInitialize(&mut layout, &maker);
            (
                dk::dkImageLayoutGetSize(&layout),
                dk::dkImageLayoutGetAlignment(&layout) as u64,
            )
        };
        let stride = align_up(size, align);
        let total = align_up(stride * count as u64, dk::DK_MEMBLOCK_ALIGNMENT as u64);
        let mut mem_maker = dk::DkMemBlockMaker::new(
            dev,
            u32::try_from(total)
                .map_err(|_| crate::SurfaceError::Other("deko3d: swapchain too big"))?,
        );
        mem_maker.flags = dk::DkMemBlockFlags_GpuCached | dk::DkMemBlockFlags_Image;
        let mem = unsafe { dk::dkMemBlockCreate(&mem_maker) };
        if mem.is_null() {
            return Err(crate::SurfaceError::Device(crate::DeviceError::OutOfMemory));
        }

        let mut images: Box<[dk::DkImage]> = (0..count).map(|_| dk::DkImage::zeroed()).collect();
        for (i, image) in images.iter_mut().enumerate() {
            unsafe { dk::dkImageInitialize(image, &layout, mem, (stride * i as u64) as u32) };
        }
        let ptrs: Vec<*const dk::DkImage> = images.iter().map(ptr::from_ref).collect();
        let raw = unsafe {
            dk::dkSwapchainCreate(&dk::DkSwapchainMaker {
                device: dev,
                nativeWindow: self.window,
                pImages: ptrs.as_ptr(),
                numImages: count,
            })
        };
        if raw.is_null() {
            unsafe { dk::dkMemBlockDestroy(mem) };
            return Err(crate::SurfaceError::Other(
                "deko3d: dkSwapchainCreate failed",
            ));
        }
        let interval = match config.present_mode {
            wgt::PresentMode::Immediate | wgt::PresentMode::AutoNoVsync => 0,
            _ => 1,
        };
        unsafe { dk::dkSwapchainSetSwapInterval(raw, interval) };
        log::info!(
            "deko3d: swapchain {}x{} {:?} x{count}, interval {interval}",
            config.extent.width,
            config.extent.height,
            config.format
        );

        *slot = Some(Swapchain {
            raw,
            queue: Arc::clone(&device.queue),
            mem,
            images,
        });
        Ok(())
    }

    unsafe fn unconfigure(&self, _device: &Device) {
        *self.swapchain.lock() = None;
    }

    unsafe fn acquire_texture(
        &self,
        _timeout: Option<Duration>,
        _fence: &Fence,
    ) -> Result<crate::AcquiredSurfaceTexture<Api>, crate::SurfaceError> {
        let guard = self.swapchain.lock();
        let sc = guard.as_ref().ok_or(crate::SurfaceError::Outdated)?;
        let mut slot = -1;
        let mut acquire = dk::DkFence::zeroed();
        // ponytail: blocks until an image is free and ignores `timeout`.
        unsafe { dk::dkSwapchainAcquireImage(sc.raw, &mut slot, &mut acquire) };
        let image = usize::try_from(slot)
            .ok()
            .and_then(|i| sc.images.get(i))
            .ok_or(crate::SurfaceError::Lost)?;
        Ok(crate::AcquiredSurfaceTexture {
            texture: Texture {
                image: ptr::from_ref(image),
                slot,
                acquire,
            },
            suboptimal: false,
        })
    }

    unsafe fn discard_texture(&self, texture: Texture) {
        // deko3d only returns a dequeued image through a present; dropping it
        // would leak the slot and hang the next acquire. Present it as-is.
        if let Some(sc) = self.swapchain.lock().as_ref() {
            unsafe { sc.queue.present(sc.raw, &texture) };
        }
    }
}
