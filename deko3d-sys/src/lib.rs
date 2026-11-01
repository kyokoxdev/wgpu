//! FFI for devkitPro deko3d v0.5 (`libnx/include/deko3d.h`).
//!
//! Hand-written (no libclang in the toolchain), and grown only as the
//! wgpu-hal backend needs it. Layouts follow the C header field for field;
//! `DK_CONSTEXPR` helpers (`dk*Defaults`, `dkMake*Handle`) are ported here
//! because they are `static inline` in C and not exported by the library.
//! Items exist on every target so host builds type-check; only horizon links.

#![no_std]
#![allow(non_camel_case_types, non_snake_case, non_upper_case_globals)]

use core::ffi::{c_char, c_void};

macro_rules! handle {
    ($($name:ident),*) => {$(
        #[repr(transparent)]
        #[derive(Clone, Copy, Debug, PartialEq, Eq)]
        pub struct $name(pub *mut c_void);
        impl $name {
            pub const NULL: Self = Self(core::ptr::null_mut());
            pub fn is_null(self) -> bool { self.0.is_null() }
        }
    )*};
}
handle!(DkDevice, DkMemBlock, DkCmdBuf, DkQueue, DkSwapchain);

macro_rules! opaque {
    ($($name:ident, $align:literal, $size:literal);*) => {$(
        #[repr(C, align($align))]
        #[derive(Clone, Copy)]
        pub struct $name(pub [u8; $size]);
        const _: () = assert!(core::mem::size_of::<$name>() == $size);
        impl $name {
            pub const fn zeroed() -> Self { Self([0; $size]) }
        }
    )*};
}
opaque!(
    DkFence, 8, 64;
    DkVariable, 8, 16;
    DkShader, 8, 128;
    DkImageLayout, 8, 128;
    DkImage, 8, 128;
    DkImageDescriptor, 4, 32;
    DkSamplerDescriptor, 4, 32
);

pub type DkResult = u32;
pub const DkResult_Success: DkResult = 0;
pub const DkResult_Fail: DkResult = 1;
pub const DkResult_Timeout: DkResult = 2;
pub const DkResult_OutOfMemory: DkResult = 3;
pub const DkResult_NotImplemented: DkResult = 4;
pub const DkResult_MisalignedSize: DkResult = 5;
pub const DkResult_MisalignedData: DkResult = 6;
pub const DkResult_BadInput: DkResult = 7;
pub const DkResult_BadFlags: DkResult = 8;
pub const DkResult_BadState: DkResult = 9;

pub const DK_GPU_ADDR_INVALID: DkGpuAddr = u64::MAX;
pub type DkGpuAddr = u64;
pub type DkCmdList = usize;
pub type DkResHandle = u32;
pub type DkDebugFunc =
    Option<unsafe extern "C" fn(*mut c_void, *const c_char, DkResult, *const c_char)>;
pub type DkAllocFunc =
    Option<unsafe extern "C" fn(*mut c_void, usize, usize, *mut *mut c_void) -> DkResult>;
pub type DkFreeFunc = Option<unsafe extern "C" fn(*mut c_void, *mut c_void)>;

pub const DkDeviceFlags_DepthZeroToOne: u32 = 0 << 8;
pub const DkDeviceFlags_DepthMinusOneToOne: u32 = 1 << 8;
pub const DkDeviceFlags_OriginUpperLeft: u32 = 0 << 9;
pub const DkDeviceFlags_OriginLowerLeft: u32 = 1 << 9;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct DkDeviceMaker {
    pub userData: *mut c_void,
    pub cbDebug: DkDebugFunc,
    pub cbAlloc: DkAllocFunc,
    pub cbFree: DkFreeFunc,
    pub flags: u32,
}

impl Default for DkDeviceMaker {
    /// `dkDeviceMakerDefaults`.
    fn default() -> Self {
        Self {
            userData: core::ptr::null_mut(),
            cbDebug: None,
            cbAlloc: None,
            cbFree: None,
            flags: DkDeviceFlags_DepthZeroToOne | DkDeviceFlags_OriginUpperLeft,
        }
    }
}

pub const DK_MEMBLOCK_ALIGNMENT: u32 = 0x1000;
pub const DK_CMDMEM_ALIGNMENT: u32 = 4;
pub const DK_QUEUE_MIN_CMDMEM_SIZE: u32 = 0x10000;
pub const DK_PER_WARP_SCRATCH_MEM_ALIGNMENT: u32 = 0x200;
pub const DK_NUM_UNIFORM_BUFS: u32 = 16;
pub const DK_NUM_STORAGE_BUFS: u32 = 16;
pub const DK_NUM_TEXTURE_BINDINGS: u32 = 32;
pub const DK_NUM_IMAGE_BINDINGS: u32 = 8;
pub const DK_UNIFORM_BUF_ALIGNMENT: u32 = 0x100;
pub const DK_UNIFORM_BUF_MAX_SIZE: u32 = 0x10000;
pub const DK_DEFAULT_MAX_COMPUTE_CONCURRENT_JOBS: u32 = 128;
pub const DK_SHADER_CODE_ALIGNMENT: u32 = 0x100;
pub const DK_SHADER_CODE_UNUSABLE_SIZE: u32 = 0x400;
pub const DK_IMAGE_DESCRIPTOR_ALIGNMENT: u32 = 0x20;
pub const DK_SAMPLER_DESCRIPTOR_ALIGNMENT: u32 = 0x20;
pub const DK_MAX_RENDER_TARGETS: u32 = 8;
pub const DK_NUM_VIEWPORTS: u32 = 16;
pub const DK_NUM_SCISSORS: u32 = 16;
pub const DK_MAX_VERTEX_ATTRIBS: u32 = 32;
pub const DK_MAX_VERTEX_BUFFERS: u32 = 16;
pub const DK_IMAGE_LINEAR_STRIDE_ALIGNMENT: u32 = 32;

pub const DkMemAccess_None: u32 = 0;
pub const DkMemAccess_Uncached: u32 = 1;
pub const DkMemAccess_Cached: u32 = 2;
pub const DkMemAccess_Mask: u32 = 3;

pub const DkMemBlockFlags_CpuAccessShift: u32 = 0;
pub const DkMemBlockFlags_GpuAccessShift: u32 = 2;
pub const DkMemBlockFlags_CpuUncached: u32 = DkMemAccess_Uncached << DkMemBlockFlags_CpuAccessShift;
pub const DkMemBlockFlags_CpuCached: u32 = DkMemAccess_Cached << DkMemBlockFlags_CpuAccessShift;
pub const DkMemBlockFlags_CpuAccessMask: u32 = DkMemAccess_Mask << DkMemBlockFlags_CpuAccessShift;
pub const DkMemBlockFlags_GpuUncached: u32 = DkMemAccess_Uncached << DkMemBlockFlags_GpuAccessShift;
pub const DkMemBlockFlags_GpuCached: u32 = DkMemAccess_Cached << DkMemBlockFlags_GpuAccessShift;
pub const DkMemBlockFlags_GpuAccessMask: u32 = DkMemAccess_Mask << DkMemBlockFlags_GpuAccessShift;
pub const DkMemBlockFlags_Code: u32 = 1 << 4;
pub const DkMemBlockFlags_Image: u32 = 1 << 5;
pub const DkMemBlockFlags_ZeroFillInit: u32 = 1 << 8;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct DkMemBlockMaker {
    pub device: DkDevice,
    pub size: u32,
    pub flags: u32,
    pub storage: *mut c_void,
}

impl DkMemBlockMaker {
    /// `dkMemBlockMakerDefaults`.
    pub fn new(device: DkDevice, size: u32) -> Self {
        Self {
            device,
            size,
            flags: DkMemBlockFlags_CpuUncached | DkMemBlockFlags_GpuCached,
            storage: core::ptr::null_mut(),
        }
    }
}

pub const DkQueueFlags_Graphics: u32 = 1 << 0;
pub const DkQueueFlags_Compute: u32 = 1 << 1;
pub const DkQueueFlags_MediumPrio: u32 = 0 << 2;
pub const DkQueueFlags_EnableZcull: u32 = 0 << 4;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct DkQueueMaker {
    pub device: DkDevice,
    pub flags: u32,
    pub commandMemorySize: u32,
    pub flushThreshold: u32,
    pub perWarpScratchMemorySize: u32,
    pub maxConcurrentComputeJobs: u32,
}

impl DkQueueMaker {
    /// `dkQueueMakerDefaults`.
    pub fn new(device: DkDevice) -> Self {
        Self {
            device,
            flags: DkQueueFlags_Graphics
                | DkQueueFlags_Compute
                | DkQueueFlags_MediumPrio
                | DkQueueFlags_EnableZcull,
            commandMemorySize: DK_QUEUE_MIN_CMDMEM_SIZE,
            flushThreshold: DK_QUEUE_MIN_CMDMEM_SIZE / 8,
            perWarpScratchMemorySize: 4 * DK_PER_WARP_SCRATCH_MEM_ALIGNMENT,
            maxConcurrentComputeJobs: DK_DEFAULT_MAX_COMPUTE_CONCURRENT_JOBS,
        }
    }
}

pub type DkCmdBufAddMemFunc = Option<unsafe extern "C" fn(*mut c_void, DkCmdBuf, usize)>;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct DkCmdBufMaker {
    pub device: DkDevice,
    pub userData: *mut c_void,
    pub cbAddMem: DkCmdBufAddMemFunc,
}

pub type DkImageType = u32;
pub const DkImageType_None: DkImageType = 0;
pub const DkImageType_2D: DkImageType = 2;

pub const DkImageFlags_HwCompression: u32 = 1 << 2;
pub const DkImageFlags_UsageRender: u32 = 1 << 8;
pub const DkImageFlags_UsagePresent: u32 = 1 << 10;

/// Values checked against `deko3d.h` with a host C build.
pub type DkImageFormat = u32;
pub const DkImageFormat_None: DkImageFormat = 0;
pub const DkImageFormat_RGBA8_Unorm: DkImageFormat = 28;
pub const DkImageFormat_RGBA8_Unorm_sRGB: DkImageFormat = 47;

pub type DkMsMode = u32;
pub const DkMsMode_1x: DkMsMode = 0;

pub const DkImageSwizzle_Red: u32 = 2;
pub const DkImageSwizzle_Green: u32 = 3;
pub const DkImageSwizzle_Blue: u32 = 4;
pub const DkImageSwizzle_Alpha: u32 = 5;
pub const DkDsSource_Depth: u32 = 0;

pub const DkColorMask_RGBA: u32 = 15;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct DkImageLayoutMaker {
    pub device: DkDevice,
    pub type_: DkImageType,
    pub flags: u32,
    pub format: DkImageFormat,
    pub msMode: DkMsMode,
    pub dimensions: [u32; 3],
    pub mipLevels: u32,
    /// Union with `tileSize`.
    pub pitchStride: u32,
}
const _: () = assert!(core::mem::size_of::<DkImageLayoutMaker>() == 48);

impl DkImageLayoutMaker {
    /// `dkImageLayoutMakerDefaults`.
    pub fn new(device: DkDevice) -> Self {
        Self {
            device,
            type_: DkImageType_2D,
            flags: 0,
            format: DkImageFormat_None,
            msMode: DkMsMode_1x,
            dimensions: [0; 3],
            mipLevels: 1,
            pitchStride: 0,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct DkImageView {
    pub pImage: *const DkImage,
    pub type_: DkImageType,
    pub format: DkImageFormat,
    pub swizzle: [u32; 4],
    pub dsSource: u32,
    pub layerOffset: u16,
    pub layerCount: u16,
    pub mipLevelOffset: u8,
    pub mipLevelCount: u8,
}
const _: () = assert!(core::mem::size_of::<DkImageView>() == 48);

impl DkImageView {
    /// `dkImageViewDefaults`.
    pub fn new(image: *const DkImage) -> Self {
        Self {
            pImage: image,
            type_: DkImageType_None,
            format: DkImageFormat_None,
            swizzle: [
                DkImageSwizzle_Red,
                DkImageSwizzle_Green,
                DkImageSwizzle_Blue,
                DkImageSwizzle_Alpha,
            ],
            dsSource: DkDsSource_Depth,
            layerOffset: 0,
            layerCount: 0,
            mipLevelOffset: 0,
            mipLevelCount: 0,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct DkViewport {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
    pub near: f32,
    pub far: f32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct DkScissor {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct DkSwapchainMaker {
    pub device: DkDevice,
    pub nativeWindow: *mut c_void,
    pub pImages: *const *const DkImage,
    pub numImages: u32,
}

unsafe extern "C" {
    pub fn dkDeviceCreate(maker: *const DkDeviceMaker) -> DkDevice;
    pub fn dkDeviceDestroy(obj: DkDevice);

    pub fn dkMemBlockCreate(maker: *const DkMemBlockMaker) -> DkMemBlock;
    pub fn dkMemBlockDestroy(obj: DkMemBlock);
    pub fn dkMemBlockGetCpuAddr(obj: DkMemBlock) -> *mut c_void;
    pub fn dkMemBlockGetGpuAddr(obj: DkMemBlock) -> DkGpuAddr;
    pub fn dkMemBlockGetSize(obj: DkMemBlock) -> u32;
    pub fn dkMemBlockFlushCpuCache(obj: DkMemBlock, offset: u32, size: u32) -> DkResult;

    pub fn dkFenceWait(obj: *mut DkFence, timeout_ns: i64) -> DkResult;

    pub fn dkCmdBufCreate(maker: *const DkCmdBufMaker) -> DkCmdBuf;
    pub fn dkCmdBufDestroy(obj: DkCmdBuf);
    pub fn dkCmdBufAddMemory(obj: DkCmdBuf, mem: DkMemBlock, offset: u32, size: u32);
    pub fn dkCmdBufFinishList(obj: DkCmdBuf) -> DkCmdList;
    pub fn dkCmdBufClear(obj: DkCmdBuf);
    pub fn dkCmdBufBindRenderTargets(
        obj: DkCmdBuf,
        colorTargets: *const *const DkImageView,
        numColorTargets: u32,
        depthTarget: *const DkImageView,
    );
    pub fn dkCmdBufSetViewports(
        obj: DkCmdBuf,
        firstId: u32,
        viewports: *const DkViewport,
        num: u32,
    );
    pub fn dkCmdBufSetScissors(obj: DkCmdBuf, firstId: u32, scissors: *const DkScissor, num: u32);
    pub fn dkCmdBufClearColor(
        obj: DkCmdBuf,
        targetId: u32,
        clearMask: u32,
        clearData: *const c_void,
    );
    pub fn dkCmdBufClearDepthStencil(
        obj: DkCmdBuf,
        clearDepth: bool,
        depthValue: f32,
        stencilMask: u8,
        stencilValue: u8,
    );

    pub fn dkQueueCreate(maker: *const DkQueueMaker) -> DkQueue;
    pub fn dkQueueDestroy(obj: DkQueue);
    pub fn dkQueueIsInErrorState(obj: DkQueue) -> bool;
    pub fn dkQueueWaitFence(obj: DkQueue, fence: *mut DkFence);
    pub fn dkQueueSignalFence(obj: DkQueue, fence: *mut DkFence, flush: bool);
    pub fn dkQueueSubmitCommands(obj: DkQueue, cmds: DkCmdList);
    pub fn dkQueueFlush(obj: DkQueue);
    pub fn dkQueueWaitIdle(obj: DkQueue);
    pub fn dkQueuePresentImage(obj: DkQueue, swapchain: DkSwapchain, imageSlot: i32);

    pub fn dkImageLayoutInitialize(obj: *mut DkImageLayout, maker: *const DkImageLayoutMaker);
    pub fn dkImageLayoutGetSize(obj: *const DkImageLayout) -> u64;
    pub fn dkImageLayoutGetAlignment(obj: *const DkImageLayout) -> u32;
    pub fn dkImageInitialize(
        obj: *mut DkImage,
        layout: *const DkImageLayout,
        memBlock: DkMemBlock,
        offset: u32,
    );

    pub fn dkSwapchainCreate(maker: *const DkSwapchainMaker) -> DkSwapchain;
    pub fn dkSwapchainDestroy(obj: DkSwapchain);
    pub fn dkSwapchainAcquireImage(obj: DkSwapchain, imageSlot: *mut i32, fence: *mut DkFence);
    pub fn dkSwapchainSetSwapInterval(obj: DkSwapchain, interval: u32);
}
