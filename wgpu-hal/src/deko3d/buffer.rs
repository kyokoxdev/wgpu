use core::{ops::Range, ptr};

use deko3d_sys as dk;

use super::memory::{Allocation, Pool};

/// Uniform/storage offset alignment; covers every buffer use.
pub(super) const BUFFER_ALIGN: u32 = 0x100;

/// A range of a CPU-uncached, GPU-cached block. Clones share the memory;
/// only `destroy_buffer` frees it.
#[derive(Clone)]
pub struct Buffer {
    pub(super) mem: Allocation,
    cpu: *mut u8,
    size: usize,
}

/// SAFETY:
/// This shared mutable data will not be accessed in a way which causes data races;
/// the obligation to do so is on the caller of the HAL API.
/// For safe code, `wgpu-core` validation manages appropriate access.
unsafe impl Send for Buffer {}
unsafe impl Sync for Buffer {}

impl core::fmt::Debug for Buffer {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Buffer")
            .field("offset", &self.mem.offset)
            .field("size", &self.size)
            .finish()
    }
}

impl Buffer {
    pub(super) fn new(
        pool: &mut Pool,
        desc: &crate::BufferDescriptor,
    ) -> Result<Self, crate::DeviceError> {
        let size = usize::try_from(desc.size).map_err(|_| crate::DeviceError::OutOfMemory)?;
        let mem = pool
            .alloc(desc.size, BUFFER_ALIGN)
            .ok_or(crate::DeviceError::OutOfMemory)?;
        let cpu = unsafe {
            dk::dkMemBlockGetCpuAddr(mem.block)
                .cast::<u8>()
                .add(mem.offset as usize)
        };
        Ok(Buffer { mem, cpu, size })
    }

    /// Returns a pointer to the memory owned by this buffer within the given `range`.
    ///
    /// This may be used to create any number of simultaneous pointers;
    /// aliasing is only a concern when actually reading, writing, or converting the pointer
    /// to a reference.
    pub(super) fn get_slice_ptr(&self, range: crate::MemoryRange) -> *mut [u8] {
        let range = range_to_usize(range, self.size);
        // SAFETY: `range_to_usize` bounds checks this addition.
        ptr::slice_from_raw_parts_mut(unsafe { self.cpu.add(range.start) }, range.len())
    }
}

/// Convert a [`crate::MemoryRange`] to `Range<usize>` and bounds check it.
fn range_to_usize(range: crate::MemoryRange, upper_bound: usize) -> Range<usize> {
    // Note: these assertions should be impossible to trigger from safe code.
    let start = usize::try_from(range.start).expect("range too large");
    let end = usize::try_from(range.end).expect("range too large");
    assert!(start <= end && end <= upper_bound, "range out of bounds");
    start..end
}
