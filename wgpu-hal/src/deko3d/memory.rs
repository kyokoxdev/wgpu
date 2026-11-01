//! GPU memory: fixed-size `DkMemBlock`s, each carved up with
//! `offset-allocator`. Allocations bigger than a block get a block of their
//! own. wgpu-core destroys a resource only after its last submission retires,
//! so `free` is immediate.

use alloc::vec::Vec;

use deko3d_sys as dk;
use offset_allocator::Allocator;

/// Size of one pooled block.
pub(super) const BLOCK_SIZE: u32 = 32 << 20;

/// Allocates `size` bytes at a multiple of `align` (a power of two) from `a`.
/// Over-allocates by `align - 1` and rounds the offset up, since
/// offset-allocator has no alignment of its own.
/// Returns the aligned offset and the raw allocation to free later.
fn alloc_aligned(
    a: &mut Allocator,
    size: u32,
    align: u32,
) -> Option<(u32, offset_allocator::Allocation)> {
    let raw = a.allocate(size.checked_add(align - 1)?)?;
    Some((raw.offset.next_multiple_of(align), raw))
}

/// One suballocation, or a whole dedicated block (`raw` is `None`).
#[derive(Clone, Copy)]
pub(super) struct Allocation {
    pub block: dk::DkMemBlock,
    pub offset: u32,
    raw: Option<offset_allocator::Allocation>,
}

struct Block {
    raw: dk::DkMemBlock,
    alloc: Allocator,
}

pub(super) struct Pool {
    device: dk::DkDevice,
    flags: u32,
    blocks: Vec<Block>,
}

// SAFETY: the pool is only used behind the device's mutex.
unsafe impl Send for Pool {}

impl Pool {
    pub fn new(device: dk::DkDevice, flags: u32) -> Self {
        Self {
            device,
            flags,
            blocks: Vec::new(),
        }
    }

    fn create_block(&self, size: u32) -> Option<dk::DkMemBlock> {
        let mut maker = dk::DkMemBlockMaker::new(self.device, size);
        maker.flags = self.flags;
        let raw = unsafe { dk::dkMemBlockCreate(&maker) };
        (!raw.is_null()).then_some(raw)
    }

    pub fn alloc(&mut self, size: u64, align: u32) -> Option<Allocation> {
        let size = u32::try_from(size.max(1)).ok()?;
        if size > BLOCK_SIZE / 2 {
            let block_size = size.checked_next_multiple_of(dk::DK_MEMBLOCK_ALIGNMENT)?;
            return Some(Allocation {
                block: self.create_block(block_size)?,
                offset: 0,
                raw: None,
            });
        }
        for b in &mut self.blocks {
            if let Some((offset, raw)) = alloc_aligned(&mut b.alloc, size, align) {
                return Some(Allocation { block: b.raw, offset, raw: Some(raw) });
            }
        }
        // ponytail: empty blocks are kept, not returned; trim if the heap line
        // shows pools only grow.
        let mut b = Block {
            raw: self.create_block(BLOCK_SIZE)?,
            alloc: Allocator::new(BLOCK_SIZE),
        };
        let (offset, raw) = alloc_aligned(&mut b.alloc, size, align)?;
        let block = b.raw;
        self.blocks.push(b);
        Some(Allocation { block, offset, raw: Some(raw) })
    }

    pub fn free(&mut self, a: Allocation) {
        match a.raw {
            None => unsafe { dk::dkMemBlockDestroy(a.block) },
            Some(raw) => {
                let b = self.blocks.iter_mut().find(|b| b.raw == a.block);
                b.expect("deko3d: free of unknown block").alloc.free(raw);
            }
        }
    }
}

impl Drop for Pool {
    fn drop(&mut self) {
        for b in &self.blocks {
            unsafe { dk::dkMemBlockDestroy(b.raw) };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aligned_alloc_free() {
        let mut a = Allocator::new(BLOCK_SIZE);
        let mut live = Vec::new();
        for (i, align) in [1u32, 0x100, 0x1000, 0x100, 0x10000].into_iter().cycle().take(200).enumerate() {
            let size = 1 + (i as u32 * 7919) % 0x3000;
            let (off, raw) = alloc_aligned(&mut a, size, align).unwrap();
            assert_eq!(off % align, 0);
            assert!(off + size <= BLOCK_SIZE);
            live.push((off, size, raw));
        }
        // No overlaps.
        let mut spans: Vec<_> = live.iter().map(|&(o, s, _)| (o, o + s)).collect();
        spans.sort();
        assert!(spans.windows(2).all(|w| w[0].1 <= w[1].0));
        for (_, _, raw) in live {
            a.free(raw);
        }
        assert_eq!(a.storage_report().total_free_space, BLOCK_SIZE);
        assert!(alloc_aligned(&mut a, BLOCK_SIZE, 1).is_some());
    }
}
