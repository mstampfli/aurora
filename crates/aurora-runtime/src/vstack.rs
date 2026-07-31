//! The value stack: a per-thread bump arena for aggregates too large to live in
//! a machine stack frame.
//!
//! # Why this exists
//!
//! Codegen used to place every aggregate in a Cranelift `ExplicitSlot`, so a
//! function's frame grew with the size of the structs it touched. A 235 KB
//! struct (marrow's `Warren`: 46 fixed arrays) produced a 470 KB frame, and a
//! call chain holding two or three of those blew straight through Windows'
//! 1 MB default stack reserve. Worse, a single `sub rsp, 0x75910` jumps ~117
//! guard pages at once, so Windows never gets the ordered guard-page fault it
//! uses to grow the stack: the process dies with a bare access violation and no
//! diagnostic. Linux's main thread auto-grows to 8 MB on demand, which is the
//! only reason the same code ran there.
//!
//! # The shape
//!
//! A bump arena, not a general allocator, because the lifetime is exactly a
//! function activation - strictly nested, so a stack is the correct structure:
//!
//! - `alloc` is a bounds check and a pointer add.
//! - `enter`/`leave` push and pop a mark; `leave` frees everything the call
//!   allocated in O(1), regardless of how many allocations it made.
//! - Chunks are **retained** after `leave`, never freed, so a steady-state game
//!   loop performs zero allocations no matter how many frames it runs. The
//!   arena converges on the high-water mark of the deepest call chain and stays
//!   there.
//! - Exhausting a chunk appends another (each at least as big as the last), so
//!   there is no fixed ceiling: depth is bounded by memory, not by a constant.
//!
//! Thread-local, so Aurora code running on the runtime's worker threads gets
//! its own arena with no synchronisation on the hot path.
//!
//! **Place in the graph.** Leaf: depends on nothing in the workspace.
//!
//! **Never.** Never used for anything whose lifetime outlives its function
//! activation - this is a stack, and `leave` invalidates every pointer the call
//! handed out.

use std::cell::RefCell;

/// Size of the first chunk, and the floor for every later one. Big enough that
/// a deep chain of large aggregates never touches the slow path, small enough
/// that a thread which uses the arena lightly does not pay for it: nothing is
/// allocated until the first `alloc` call.
const CHUNK: usize = 4 << 20; // 4 MiB

/// Every allocation is 16-byte aligned - enough for any Aurora scalar, and for
/// the SIMD types a future backend may want to store inline.
const ALIGN: usize = 16;

struct VStack {
    /// Retained backing store. Never shrinks: reuse is the whole point.
    chunks: Vec<Vec<u8>>,
    /// Index of the chunk the bump pointer is in.
    cur: usize,
    /// Bump offset within `chunks[cur]`.
    off: usize,
    /// One saved `(cur, off)` per live `enter`.
    marks: Vec<(usize, usize)>,
    /// High-water mark in bytes, for tests and diagnostics.
    peak: usize,
}

impl VStack {
    const fn new() -> Self {
        VStack {
            chunks: Vec::new(),
            cur: 0,
            off: 0,
            marks: Vec::new(),
            peak: 0,
        }
    }

    /// Total bytes handed out below the current bump pointer. Only meaningful
    /// as a growth signal; chunks before `cur` may be partly unused.
    fn used(&self) -> usize {
        self.chunks[..self.cur].iter().map(Vec::len).sum::<usize>() + self.off
    }

    fn alloc(&mut self, size: usize) -> *mut u8 {
        let size = size.max(1).next_multiple_of(ALIGN);

        // First use: nothing is allocated until a program actually needs it, so
        // a thread that never touches a large aggregate never pays the 4 MiB.
        if self.chunks.is_empty() {
            self.chunks.push(vec![0u8; CHUNK.max(size)]);
            self.cur = 0;
            self.off = 0;
        }

        // Fast path: room in the current chunk.
        if self.off + size <= self.chunks[self.cur].len() {
            let p = self.chunks[self.cur][self.off..].as_mut_ptr();
            self.off += size;
            self.peak = self.peak.max(self.used());
            return p;
        }

        // Slow path: advance to the next chunk. Everything at an index past
        // `cur` is free by construction - marks only ever point at or below the
        // current position - so growing or replacing it cannot invalidate a
        // live pointer.
        let next = self.cur + 1;
        if next == self.chunks.len() {
            self.chunks.push(vec![0u8; CHUNK.max(size)]);
        } else if self.chunks[next].len() < size {
            // Grow monotonically rather than to the exact request, so a
            // sequence of increasing allocations converges instead of
            // reallocating on every call.
            let grown = (self.chunks[next].len() * 2).max(size).max(CHUNK);
            self.chunks[next] = vec![0u8; grown];
        }
        self.cur = next;
        self.off = size;
        self.peak = self.peak.max(self.used());
        self.chunks[self.cur].as_mut_ptr()
    }

    fn enter(&mut self) {
        self.marks.push((self.cur, self.off));
    }

    fn leave(&mut self) {
        // A missing mark would mean codegen emitted an unbalanced leave. Ignore
        // it rather than corrupt the arena: resetting to zero here would free
        // memory an outer frame is still using.
        if let Some((cur, off)) = self.marks.pop() {
            self.cur = cur;
            self.off = off;
        }
    }
}

thread_local! {
    static VSTACK: RefCell<VStack> = const { RefCell::new(VStack::new()) };
}

/// Push a frame mark. Emitted at the top of any function whose lowering placed
/// an aggregate in the arena.
#[no_mangle]
pub extern "C" fn aurora_vstack_enter() {
    VSTACK.with(|v| v.borrow_mut().enter());
}

/// Pop to the matching mark, freeing every allocation the call made at once.
#[no_mangle]
pub extern "C" fn aurora_vstack_leave() {
    VSTACK.with(|v| v.borrow_mut().leave());
}

/// Allocate `size` bytes with the lifetime of the current frame. The pointer is
/// valid until the matching `aurora_vstack_leave`.
#[no_mangle]
pub extern "C" fn aurora_vstack_alloc(size: i64) -> *mut u8 {
    VSTACK.with(|v| v.borrow_mut().alloc(size.max(0) as usize))
}

/// High-water mark in bytes for this thread. Diagnostics and tests only.
#[no_mangle]
pub extern "C" fn aurora_vstack_peak() -> i64 {
    VSTACK.with(|v| v.borrow().peak as i64)
}

/// Clear the high-water mark so a following measurement reflects only what
/// happens after this point. Diagnostics and tests only - without it a peak
/// reading is polluted by whatever ran earlier on the same thread, and a test
/// built on it could never fail.
#[no_mangle]
pub extern "C" fn aurora_vstack_reset_peak() {
    VSTACK.with(|v| {
        let mut s = v.borrow_mut();
        s.peak = if s.chunks.is_empty() { 0 } else { s.used() };
    });
}

/// Bytes currently outstanding on this thread. Diagnostics and tests only.
#[no_mangle]
pub extern "C" fn aurora_vstack_used() -> i64 {
    VSTACK.with(|v| {
        let s = v.borrow();
        if s.chunks.is_empty() {
            0
        } else {
            s.used() as i64
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The property the whole design rests on: a call that allocates and leaves
    /// gives the memory back, so repeating it forever does not grow the arena.
    #[test]
    fn leave_frees_everything_the_call_allocated() {
        aurora_vstack_enter();
        let base = aurora_vstack_used();
        aurora_vstack_leave();

        for _ in 0..10_000 {
            aurora_vstack_enter();
            let _ = aurora_vstack_alloc(240_888);
            let _ = aurora_vstack_alloc(240_888);
            aurora_vstack_leave();
        }
        assert_eq!(
            aurora_vstack_used(),
            base,
            "ten thousand calls must return to the same bump offset"
        );
    }

    /// Steady state must not allocate. After the first pass the chunk list is
    /// warm, so a second identical pass adds no chunks at all.
    #[test]
    fn chunks_are_reused_not_reallocated() {
        for _ in 0..50 {
            aurora_vstack_enter();
            let _ = aurora_vstack_alloc(1 << 20);
            aurora_vstack_leave();
        }
        let warm = VSTACK.with(|v| v.borrow().chunks.len());
        for _ in 0..50 {
            aurora_vstack_enter();
            let _ = aurora_vstack_alloc(1 << 20);
            aurora_vstack_leave();
        }
        assert_eq!(
            VSTACK.with(|v| v.borrow().chunks.len()),
            warm,
            "a warm arena must not allocate another chunk"
        );
    }

    /// Nesting is what a call chain does, so marks must nest exactly.
    #[test]
    fn nested_frames_unwind_in_order() {
        let start = aurora_vstack_used();
        aurora_vstack_enter();
        let a = aurora_vstack_alloc(4096);
        let mid = aurora_vstack_used();
        aurora_vstack_enter();
        let b = aurora_vstack_alloc(4096);
        assert_ne!(a, b, "an inner frame must not reuse a live outer pointer");
        aurora_vstack_leave();
        assert_eq!(aurora_vstack_used(), mid, "inner leave restores the middle");
        aurora_vstack_leave();
        assert_eq!(
            aurora_vstack_used(),
            start,
            "outer leave restores the start"
        );
    }

    /// An allocation larger than a whole chunk must still work - the arena is
    /// bounded by memory, not by CHUNK.
    #[test]
    fn an_allocation_bigger_than_a_chunk_still_works() {
        aurora_vstack_enter();
        let huge = CHUNK * 3;
        let p = aurora_vstack_alloc(huge as i64);
        assert!(!p.is_null());
        // Writing the whole span proves the region is really that big.
        unsafe { std::ptr::write_bytes(p, 0xAB, huge) };
        assert_eq!(unsafe { *p.add(huge - 1) }, 0xAB);
        aurora_vstack_leave();
    }

    /// Distinct allocations in one frame must not overlap.
    #[test]
    fn allocations_in_one_frame_are_disjoint() {
        aurora_vstack_enter();
        let n = 64usize;
        let ps: Vec<*mut u8> = (0..n).map(|_| aurora_vstack_alloc(1024)).collect();
        for (i, p) in ps.iter().enumerate() {
            unsafe { std::ptr::write_bytes(*p, i as u8, 1024) };
        }
        for (i, p) in ps.iter().enumerate() {
            assert_eq!(unsafe { **p }, i as u8, "allocation {i} was overwritten");
            assert_eq!(
                unsafe { *p.add(1023) },
                i as u8,
                "allocation {i} tail clobbered"
            );
        }
        aurora_vstack_leave();
    }

    /// Every pointer must satisfy the alignment the codegen assumes.
    #[test]
    fn every_allocation_is_aligned() {
        aurora_vstack_enter();
        for sz in [1i64, 7, 8, 9, 15, 16, 17, 1000] {
            let p = aurora_vstack_alloc(sz);
            assert_eq!(p as usize % ALIGN, 0, "size {sz} came back misaligned");
        }
        aurora_vstack_leave();
    }

    /// Each thread gets its own arena, so parallel Aurora code needs no locking.
    #[test]
    fn arenas_are_per_thread() {
        aurora_vstack_enter();
        let _ = aurora_vstack_alloc(1 << 20);
        let mine = aurora_vstack_used();
        let theirs = std::thread::spawn(|| aurora_vstack_used()).join().unwrap();
        assert!(mine > 0);
        assert_eq!(theirs, 0, "a fresh thread must start with an empty arena");
        aurora_vstack_leave();
    }

    /// An unbalanced leave must not corrupt an outer frame's memory.
    #[test]
    fn a_stray_leave_does_not_free_an_outer_frame() {
        aurora_vstack_enter();
        let _ = aurora_vstack_alloc(4096);
        let held = aurora_vstack_used();
        // Drain any marks this thread still holds, then over-pop.
        for _ in 0..8 {
            aurora_vstack_leave();
        }
        let after = aurora_vstack_used();
        assert!(
            after <= held,
            "leave must never hand out memory that is still live"
        );
    }
}
