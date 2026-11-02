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
pub const DkImageType_1D: DkImageType = 1;
pub const DkImageType_2D: DkImageType = 2;
pub const DkImageType_3D: DkImageType = 3;
pub const DkImageType_1DArray: DkImageType = 4;
pub const DkImageType_2DArray: DkImageType = 5;
pub const DkImageType_2DMS: DkImageType = 6;
pub const DkImageType_2DMSArray: DkImageType = 7;
pub const DkImageType_Cubemap: DkImageType = 9;
pub const DkImageType_CubemapArray: DkImageType = 10;

pub const DkImageFlags_HwCompression: u32 = 1 << 2;
pub const DkImageFlags_UsageRender: u32 = 1 << 8;
pub const DkImageFlags_UsageLoadStore: u32 = 1 << 9;
pub const DkImageFlags_UsagePresent: u32 = 1 << 10;

pub type DkBarrier = u32;
pub const DkBarrier_None: DkBarrier = 0;
pub const DkBarrier_Tiles: DkBarrier = 1;
pub const DkBarrier_Fragments: DkBarrier = 2;
pub const DkBarrier_Primitives: DkBarrier = 3;
pub const DkBarrier_Full: DkBarrier = 4;
pub const DkInvalidateFlags_Image: u32 = 1 << 0;
pub const DkInvalidateFlags_Shader: u32 = 1 << 1;
pub const DkInvalidateFlags_Descriptors: u32 = 1 << 2;
pub const DkInvalidateFlags_L2Cache: u32 = 1 << 4;

/// Values checked against `deko3d.h` with a host C build.
pub type DkImageFormat = u32;
pub const DkImageFormat_None: DkImageFormat = 0;
pub const DkImageFormat_R8_Unorm: DkImageFormat = 1;
pub const DkImageFormat_RG8_Unorm: DkImageFormat = 13;
pub const DkImageFormat_R16_Float: DkImageFormat = 5;
pub const DkImageFormat_RG16_Float: DkImageFormat = 17;
pub const DkImageFormat_R32_Float: DkImageFormat = 10;
pub const DkImageFormat_R32_Uint: DkImageFormat = 11;
pub const DkImageFormat_RG32_Float: DkImageFormat = 22;
pub const DkImageFormat_RGBA8_Snorm: DkImageFormat = 29;
pub const DkImageFormat_RGBA16_Float: DkImageFormat = 32;
pub const DkImageFormat_RGBA32_Float: DkImageFormat = 37;
pub const DkImageFormat_Z16: DkImageFormat = 41;
pub const DkImageFormat_Z24X8: DkImageFormat = 42;
pub const DkImageFormat_ZF32: DkImageFormat = 43;
pub const DkImageFormat_Z24S8: DkImageFormat = 44;
pub const DkImageFormat_ZF32_X24S8: DkImageFormat = 45;
pub const DkImageFormat_RGB10A2_Unorm: DkImageFormat = 52;
pub const DkImageFormat_RG11B10_Float: DkImageFormat = 54;
pub const DkImageFormat_RGBA_BC1: DkImageFormat = 57;
pub const DkImageFormat_RGBA_BC2: DkImageFormat = 58;
pub const DkImageFormat_RGBA_BC3: DkImageFormat = 59;
pub const DkImageFormat_RGBA_BC1_sRGB: DkImageFormat = 61;
pub const DkImageFormat_RGBA_BC2_sRGB: DkImageFormat = 62;
pub const DkImageFormat_RGBA_BC3_sRGB: DkImageFormat = 63;
pub const DkImageFormat_R_BC4_Unorm: DkImageFormat = 64;
pub const DkImageFormat_RG_BC5_Unorm: DkImageFormat = 66;
pub const DkImageFormat_RG_BC5_Snorm: DkImageFormat = 67;
pub const DkImageFormat_RGBA_BC7_Unorm: DkImageFormat = 68;
pub const DkImageFormat_RGBA_BC7_Unorm_sRGB: DkImageFormat = 69;
pub const DkImageFormat_BGRA8_Unorm: DkImageFormat = 117;
pub const DkImageFormat_BGRA8_Unorm_sRGB: DkImageFormat = 119;
pub const DkImageFormat_RGBA8_Unorm: DkImageFormat = 28;
pub const DkImageFormat_RGBA8_Unorm_sRGB: DkImageFormat = 47;

pub type DkMsMode = u32;
pub const DkMsMode_1x: DkMsMode = 0;
pub const DkMsMode_2x: DkMsMode = 1;
pub const DkMsMode_4x: DkMsMode = 2;
pub const DkMsMode_8x: DkMsMode = 3;

pub const DkImageSwizzle_Red: u32 = 2;
pub const DkImageSwizzle_Green: u32 = 3;
pub const DkImageSwizzle_Blue: u32 = 4;
pub const DkImageSwizzle_Alpha: u32 = 5;
pub const DkDsSource_Depth: u32 = 0;
pub const DkDsSource_Stencil: u32 = 1;

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

pub const DkFilter_Nearest: u32 = 1;
pub const DkFilter_Linear: u32 = 2;
pub const DkMipFilter_None: u32 = 1;
pub const DkMipFilter_Nearest: u32 = 2;
pub const DkMipFilter_Linear: u32 = 3;
pub const DkWrapMode_Repeat: u32 = 0;
pub const DkWrapMode_MirroredRepeat: u32 = 1;
pub const DkWrapMode_ClampToEdge: u32 = 2;
pub const DkWrapMode_ClampToBorder: u32 = 3;
pub const DkCompareOp_Never: u32 = 1;
pub const DkCompareOp_Less: u32 = 2;
pub const DkCompareOp_Equal: u32 = 3;
pub const DkCompareOp_Lequal: u32 = 4;
pub const DkCompareOp_Greater: u32 = 5;
pub const DkCompareOp_NotEqual: u32 = 6;
pub const DkCompareOp_Gequal: u32 = 7;
pub const DkCompareOp_Always: u32 = 8;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct DkSampler {
    pub minFilter: u32,
    pub magFilter: u32,
    pub mipFilter: u32,
    pub wrapMode: [u32; 3],
    pub lodClampMin: f32,
    pub lodClampMax: f32,
    pub lodBias: f32,
    pub lodSnap: f32,
    pub compareEnable: bool,
    pub compareOp: u32,
    /// Union of f32/u32/i32 per channel; stored as raw bits.
    pub borderColor: [u32; 4],
    pub maxAnisotropy: f32,
    pub reductionMode: u32,
}
const _: () = assert!(core::mem::size_of::<DkSampler>() == 72);
const _: () = assert!(core::mem::offset_of!(DkSampler, compareEnable) == 40);
const _: () = assert!(core::mem::offset_of!(DkSampler, borderColor) == 48);
const _: () = assert!(core::mem::offset_of!(DkSampler, maxAnisotropy) == 64);

impl Default for DkSampler {
    /// `dkSamplerDefaults`.
    fn default() -> Self {
        Self {
            minFilter: DkFilter_Nearest,
            magFilter: DkFilter_Nearest,
            mipFilter: DkMipFilter_None,
            wrapMode: [DkWrapMode_Repeat; 3],
            lodClampMin: 0.0,
            lodClampMax: 1000.0,
            lodBias: 0.0,
            lodSnap: 0.0,
            compareEnable: false,
            compareOp: DkCompareOp_Less,
            borderColor: [0; 4],
            maxAnisotropy: 1.0,
            reductionMode: 0,
        }
    }
}

pub const fn dkMakeImageHandle(id: u32) -> DkResHandle {
    id & ((1 << 20) - 1)
}
pub const fn dkMakeSamplerHandle(id: u32) -> DkResHandle {
    id << 20
}
pub const fn dkMakeTextureHandle(image: u32, sampler: u32) -> DkResHandle {
    dkMakeImageHandle(image) | dkMakeSamplerHandle(sampler)
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct DkImageRect {
    pub x: u32,
    pub y: u32,
    pub z: u32,
    pub width: u32,
    pub height: u32,
    pub depth: u32,
}

/// `rowLength` is bytes per row and `imageHeight` bytes per layer (0 =
/// tightly packed), per `dk_image.cpp`.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct DkCopyBuf {
    pub addr: DkGpuAddr,
    pub rowLength: u32,
    pub imageHeight: u32,
}
const _: () = assert!(core::mem::size_of::<DkCopyBuf>() == 16);

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

#[repr(C)]
#[derive(Clone, Copy)]
pub struct DkShaderMaker {
    pub codeMem: DkMemBlock,
    pub control: *const c_void,
    pub codeOffset: u32,
    pub programId: u32,
}


pub type DkStage = u32;
pub const DkStage_Vertex: DkStage = 0;
pub const DkStage_Fragment: DkStage = 4;
pub const DkStageFlag_GraphicsMask: u32 = (1 << 5) - 1;

pub type DkPrimitive = u32;
pub const DkPrimitive_Points: DkPrimitive = 0;
pub const DkPrimitive_Lines: DkPrimitive = 1;
pub const DkPrimitive_LineStrip: DkPrimitive = 3;
pub const DkPrimitive_Triangles: DkPrimitive = 4;
pub const DkPrimitive_TriangleStrip: DkPrimitive = 5;

pub const DkIdxFormat_Uint16: u32 = 1;
pub const DkIdxFormat_Uint32: u32 = 2;

pub const DkFace_None: u32 = 0;
pub const DkFace_Front: u32 = 1;
pub const DkFace_Back: u32 = 2;
pub const DkFace_FrontAndBack: u32 = 3;

pub const DkFrontFace_CW: u32 = 0;
pub const DkFrontFace_CCW: u32 = 1;

pub const DkPolygonMode_Point: u32 = 0;
pub const DkPolygonMode_Line: u32 = 1;
pub const DkPolygonMode_Fill: u32 = 2;

pub const DkProvokingVertex_First: u32 = 0;

pub const DkLogicOp_Copy: u32 = 3;

pub const DkBlendOp_Add: u32 = 1;
pub const DkBlendOp_Sub: u32 = 2;
pub const DkBlendOp_RevSub: u32 = 3;
pub const DkBlendOp_Min: u32 = 4;
pub const DkBlendOp_Max: u32 = 5;

pub const DkBlendFactor_Zero: u32 = 1;
pub const DkBlendFactor_One: u32 = 2;
pub const DkBlendFactor_SrcColor: u32 = 3;
pub const DkBlendFactor_InvSrcColor: u32 = 4;
pub const DkBlendFactor_SrcAlpha: u32 = 5;
pub const DkBlendFactor_InvSrcAlpha: u32 = 6;
pub const DkBlendFactor_DstAlpha: u32 = 7;
pub const DkBlendFactor_InvDstAlpha: u32 = 8;
pub const DkBlendFactor_DstColor: u32 = 9;
pub const DkBlendFactor_InvDstColor: u32 = 10;
pub const DkBlendFactor_SrcAlphaSaturate: u32 = 11;
pub const DkBlendFactor_Src1Color: u32 = 16;
pub const DkBlendFactor_InvSrc1Color: u32 = 17;
pub const DkBlendFactor_Src1Alpha: u32 = 18;
pub const DkBlendFactor_InvSrc1Alpha: u32 = 19;
pub const DkBlendFactor_ConstColor: u32 = 1 | 0x20;
pub const DkBlendFactor_InvConstColor: u32 = 2 | 0x20;
pub const DkBlendFactor_ConstAlpha: u32 = 3 | 0x20;
pub const DkBlendFactor_InvConstAlpha: u32 = 4 | 0x20;

pub const DkStencilOp_Keep: u32 = 1;
pub const DkStencilOp_Zero: u32 = 2;
pub const DkStencilOp_Replace: u32 = 3;
pub const DkStencilOp_Incr: u32 = 4;
pub const DkStencilOp_Decr: u32 = 5;
pub const DkStencilOp_Invert: u32 = 6;
pub const DkStencilOp_IncrWrap: u32 = 7;
pub const DkStencilOp_DecrWrap: u32 = 8;

pub const DkVtxAttribSize_1x32: u32 = 0x12;
pub const DkVtxAttribSize_2x32: u32 = 0x04;
pub const DkVtxAttribSize_3x32: u32 = 0x02;
pub const DkVtxAttribSize_4x32: u32 = 0x01;
pub const DkVtxAttribSize_1x16: u32 = 0x1b;
pub const DkVtxAttribSize_2x16: u32 = 0x0f;
pub const DkVtxAttribSize_4x16: u32 = 0x03;
pub const DkVtxAttribSize_1x8: u32 = 0x1d;
pub const DkVtxAttribSize_2x8: u32 = 0x18;
pub const DkVtxAttribSize_4x8: u32 = 0x0a;
pub const DkVtxAttribSize_10_10_10_2: u32 = 0x30;

pub const DkVtxAttribType_Snorm: u32 = 1;
pub const DkVtxAttribType_Unorm: u32 = 2;
pub const DkVtxAttribType_Sint: u32 = 3;
pub const DkVtxAttribType_Uint: u32 = 4;
pub const DkVtxAttribType_Float: u32 = 7;

// The render state structs are C bitfields; each is built here as its packed
// words (GCC fills a little-endian word from bit 0).

#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct DkRasterizerState(pub u32);

impl DkRasterizerState {
    pub const fn new(
        depth_clamp: bool,
        polygon_mode: u32,
        cull_mode: u32,
        front_face: u32,
        provoking_vertex: u32,
        depth_bias_mask: u32,
    ) -> Self {
        Self(
            1 | (depth_clamp as u32) << 1
                | polygon_mode << 3
                | polygon_mode << 5
                | cull_mode << 7
                | front_face << 9
                | provoking_vertex << 10
                | depth_bias_mask << 14,
        )
    }
}

#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct DkColorState(pub u32);

impl DkColorState {
    pub const fn new(blend_enable_mask: u32) -> Self {
        Self(blend_enable_mask | DkLogicOp_Copy << 8 | DkCompareOp_Always << 16)
    }
}

#[repr(C)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct DkColorWriteState {
    pub masks: u32,
}

#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct DkBlendState(pub u32);

impl DkBlendState {
    pub const fn new(
        color_op: u32,
        src_color: u32,
        dst_color: u32,
        alpha_op: u32,
        src_alpha: u32,
        dst_alpha: u32,
    ) -> Self {
        Self(
            color_op
                | src_color << 3
                | dst_color << 9
                | alpha_op << 15
                | src_alpha << 18
                | dst_alpha << 24,
        )
    }
}

#[repr(C)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct DkDepthStencilState(pub [u32; 2]);

impl DkDepthStencilState {
    pub fn new(
        depth_test: bool,
        depth_write: bool,
        stencil_test: bool,
        depth_compare: u32,
        // (fail, pass, depth fail, compare) per face.
        front: [u32; 4],
        back: [u32; 4],
    ) -> Self {
        let face = |f: [u32; 4]| f[0] | f[1] << 4 | f[2] << 8 | f[3] << 12;
        Self([
            depth_test as u32
                | (depth_write as u32) << 1
                | (stencil_test as u32) << 2
                | depth_compare << 4,
            face(front) | face(back) << 16,
        ])
    }
}

#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct DkVtxAttribState(pub u32);

impl DkVtxAttribState {
    pub const fn new(buffer: u32, offset: u32, size: u32, ty: u32, bgra: bool) -> Self {
        Self(buffer | offset << 7 | size << 21 | ty << 27 | (bgra as u32) << 31)
    }
    /// A location no buffer feeds: reads as constant.
    pub const FIXED: Self =
        Self(1 << 6 | DkVtxAttribSize_1x32 << 21 | DkVtxAttribType_Float << 27);
}

#[repr(C)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct DkVtxBufferState {
    pub stride: u32,
    pub divisor: u32,
}

#[repr(C)]
#[derive(Clone, Copy, PartialEq, Eq, Default)]
pub struct DkBufExtents {
    pub addr: DkGpuAddr,
    pub size: u32,
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

    pub fn dkCmdBufBarrier(obj: DkCmdBuf, mode: DkBarrier, invalidateFlags: u32);
    pub fn dkCmdBufBindShaders(
        obj: DkCmdBuf,
        stageMask: u32,
        shaders: *const *const DkShader,
        numShaders: u32,
    );
    pub fn dkCmdBufBindUniformBuffers(
        obj: DkCmdBuf,
        stage: DkStage,
        firstId: u32,
        buffers: *const DkBufExtents,
        numBuffers: u32,
    );
    pub fn dkCmdBufBindStorageBuffers(
        obj: DkCmdBuf,
        stage: DkStage,
        firstId: u32,
        buffers: *const DkBufExtents,
        numBuffers: u32,
    );
    pub fn dkCmdBufBindTextures(
        obj: DkCmdBuf,
        stage: DkStage,
        firstId: u32,
        handles: *const DkResHandle,
        numHandles: u32,
    );
    pub fn dkCmdBufBindImages(
        obj: DkCmdBuf,
        stage: DkStage,
        firstId: u32,
        handles: *const DkResHandle,
        numHandles: u32,
    );
    pub fn dkCmdBufBindRasterizerState(obj: DkCmdBuf, state: *const DkRasterizerState);
    pub fn dkCmdBufBindColorState(obj: DkCmdBuf, state: *const DkColorState);
    pub fn dkCmdBufBindColorWriteState(obj: DkCmdBuf, state: *const DkColorWriteState);
    pub fn dkCmdBufBindBlendStates(
        obj: DkCmdBuf,
        firstId: u32,
        states: *const DkBlendState,
        numStates: u32,
    );
    pub fn dkCmdBufBindDepthStencilState(obj: DkCmdBuf, state: *const DkDepthStencilState);
    pub fn dkCmdBufBindVtxAttribState(
        obj: DkCmdBuf,
        attribs: *const DkVtxAttribState,
        numAttribs: u32,
    );
    pub fn dkCmdBufBindVtxBufferState(
        obj: DkCmdBuf,
        buffers: *const DkVtxBufferState,
        numBuffers: u32,
    );
    pub fn dkCmdBufBindVtxBuffers(
        obj: DkCmdBuf,
        firstId: u32,
        buffers: *const DkBufExtents,
        numBuffers: u32,
    );
    pub fn dkCmdBufBindIdxBuffer(obj: DkCmdBuf, format: u32, address: DkGpuAddr);
    pub fn dkCmdBufSetDepthBias(obj: DkCmdBuf, constant: f32, clamp: f32, slope: f32);
    pub fn dkCmdBufSetBlendConst(obj: DkCmdBuf, r: f32, g: f32, b: f32, a: f32);
    pub fn dkCmdBufSetStencil(obj: DkCmdBuf, face: u32, mask: u8, funcRef: u8, funcMask: u8);
    pub fn dkCmdBufSetPrimitiveRestart(obj: DkCmdBuf, enable: bool, index: u32);
    pub fn dkCmdBufDraw(
        obj: DkCmdBuf,
        prim: DkPrimitive,
        vertexCount: u32,
        instanceCount: u32,
        firstVertex: u32,
        firstInstance: u32,
    );
    pub fn dkCmdBufDrawIndirect(obj: DkCmdBuf, prim: DkPrimitive, indirect: DkGpuAddr);
    pub fn dkCmdBufDrawIndexed(
        obj: DkCmdBuf,
        prim: DkPrimitive,
        indexCount: u32,
        instanceCount: u32,
        firstIndex: u32,
        vertexOffset: i32,
        firstInstance: u32,
    );
    pub fn dkCmdBufDrawIndexedIndirect(obj: DkCmdBuf, prim: DkPrimitive, indirect: DkGpuAddr);
    pub fn dkCmdBufResolveImage(
        obj: DkCmdBuf,
        srcView: *const DkImageView,
        dstView: *const DkImageView,
    );
    pub fn dkShaderInitialize(obj: *mut DkShader, maker: *const DkShaderMaker);
    pub fn dkCmdBufBindImageDescriptorSet(obj: DkCmdBuf, setAddr: DkGpuAddr, numDescriptors: u32);
    pub fn dkCmdBufBindSamplerDescriptorSet(obj: DkCmdBuf, setAddr: DkGpuAddr, numDescriptors: u32);
    pub fn dkCmdBufCopyBuffer(obj: DkCmdBuf, srcAddr: DkGpuAddr, dstAddr: DkGpuAddr, size: u32);
    pub fn dkCmdBufCopyImage(
        obj: DkCmdBuf,
        srcView: *const DkImageView,
        srcRect: *const DkImageRect,
        dstView: *const DkImageView,
        dstRect: *const DkImageRect,
        flags: u32,
    );
    pub fn dkCmdBufCopyBufferToImage(
        obj: DkCmdBuf,
        src: *const DkCopyBuf,
        dstView: *const DkImageView,
        dstRect: *const DkImageRect,
        flags: u32,
    );
    pub fn dkCmdBufCopyImageToBuffer(
        obj: DkCmdBuf,
        srcView: *const DkImageView,
        srcRect: *const DkImageRect,
        dst: *const DkCopyBuf,
        flags: u32,
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

    pub fn dkImageGetGpuAddr(obj: *const DkImage) -> DkGpuAddr;
    pub fn dkImageDescriptorInitialize(
        obj: *mut DkImageDescriptor,
        view: *const DkImageView,
        usesLoadOrStore: bool,
        decayMS: bool,
    );
    pub fn dkSamplerDescriptorInitialize(obj: *mut DkSamplerDescriptor, sampler: *const DkSampler);

    pub fn dkSwapchainCreate(maker: *const DkSwapchainMaker) -> DkSwapchain;
    pub fn dkSwapchainDestroy(obj: DkSwapchain);
    pub fn dkSwapchainAcquireImage(obj: DkSwapchain, imageSlot: *mut i32, fence: *mut DkFence);
    pub fn dkSwapchainSetSwapInterval(obj: DkSwapchain, interval: u32);
}
