//! Generation-tagged slot storage: stable handles whose freed slots cannot
//! alias.
//!
//! # Why not a `Vec` and an index
//!
//! GPU resources have to be released when the thing that owns them dies -
//! a level change, a terrain reload, an asset unload. A plain `Vec` gives you
//! two bad options: never remove (which is the leak) or remove and shuffle
//! (which silently repoints every handle a game is still holding at a
//! DIFFERENT resource). Reusing a slot without shuffling is no better on its
//! own: the next allocation lands in the hole, and a stale handle now reads a
//! live neighbour's mesh instead of failing.
//!
//! So a handle is not an index. It is `(index, generation)`, and a slot only
//! answers to the generation it is currently at. Freeing bumps the generation,
//! which invalidates every outstanding handle to that slot in O(1) and makes
//! the aliasing state UNREPRESENTABLE rather than merely unlikely.
//!
//! # Generation exhaustion
//!
//! A `u32` generation wraps after ~2.1 billion frees of the SAME slot, and a
//! wrap would resurrect an ancient handle. [`SlotMap`] does not wrap: a slot
//! that reaches [`MAX_GENERATION`] is retired, never returned to the free list,
//! so the guarantee holds for the life of the process rather than for the first
//! two billion frees. Retiring costs one slot of address space out of 2^32,
//! after two billion frees of that one slot.
//!
//! # Handles crossing the ABI
//!
//! Aurora programs hold handles as `i64`. [`Key::to_i64`] packs the generation
//! into bits 32..62 and the index into bits 0..31, so a handle is always
//! non-negative and never collides with the `-1` failure sentinel.
//! [`Key::from_i64`] rejects anything negative and anything with generation 0,
//! which means a zeroed or default-initialized `i64` is invalid by
//! construction rather than being handle 0.

use std::marker::PhantomData;

/// Highest generation a slot may reach before it is retired. Chosen so
/// [`Key::to_i64`] always fits in a non-negative `i64`.
pub const MAX_GENERATION: u32 = i32::MAX as u32;

/// A handle into a [`SlotMap<T>`].
///
/// Carries the slot's generation, so a handle to a freed slot is rejected by
/// [`SlotMap::get`] instead of reading whatever was put there next. The
/// `PhantomData` makes a mesh key and a material key different types, so the
/// compiler catches a swapped pair that two bare `usize`s would not.
pub struct Key<T> {
    index: u32,
    generation: u32,
    /// Invariant in `T` is unnecessary here and `fn() -> T` keeps `Key<T>`
    /// `Send`/`Sync`/`Copy` regardless of what `T` is.
    owner: PhantomData<fn() -> T>,
}

// Derived impls would demand `T: Clone`/`T: Eq`/`T: Debug`, which the stored
// value has no reason to satisfy (a `GpuMesh` is none of them). A key is two
// integers; write the impls by hand and keep the bounds off `T`.
impl<T> Clone for Key<T> {
    fn clone(&self) -> Key<T> {
        *self
    }
}
impl<T> Copy for Key<T> {}
impl<T> PartialEq for Key<T> {
    fn eq(&self, other: &Key<T>) -> bool {
        self.index == other.index && self.generation == other.generation
    }
}
impl<T> Eq for Key<T> {}
impl<T> std::hash::Hash for Key<T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.index.hash(state);
        self.generation.hash(state);
    }
}
impl<T> std::fmt::Debug for Key<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Key({}v{})", self.index, self.generation)
    }
}

impl<T> Key<T> {
    /// Pack into the non-negative `i64` an Aurora program holds.
    pub const fn to_i64(self) -> i64 {
        ((self.generation as i64) << 32) | self.index as i64
    }

    /// Unpack an `i64` handed back by an Aurora program.
    ///
    /// Returns `None` for a negative value (the runtime's `-1` failure
    /// sentinel) and for generation 0 (no slot is ever issued at generation 0,
    /// so a zeroed `i64` is not a valid handle). A well-formed key still has to
    /// clear [`SlotMap::get`]: this only rejects values that could never have
    /// been issued at all.
    pub const fn from_i64(raw: i64) -> Option<Key<T>> {
        if raw < 0 {
            return None;
        }
        let generation = (raw >> 32) as u32;
        if generation == 0 {
            return None;
        }
        Some(Key {
            index: raw as u32,
            generation,
            owner: PhantomData,
        })
    }

    /// A key that no [`SlotMap`] will ever issue, for a field that has to hold
    /// something before its real key exists. Generation 0 is reserved, so this
    /// resolves to nothing and cannot be mistaken for a live handle.
    pub const PLACEHOLDER: Key<T> = Key {
        index: 0,
        generation: 0,
        owner: PhantomData,
    };

    /// The slot this key points at, whether or not it is still live. For
    /// diagnostics and tests; use [`SlotMap::get`] to actually reach the value.
    pub const fn slot(self) -> u32 {
        self.index
    }

    /// The generation this key was issued at.
    pub const fn generation(self) -> u32 {
        self.generation
    }
}

struct Slot<T> {
    /// Generation this slot is currently at. Odd/even carries no meaning; a key
    /// matches only when the numbers are equal.
    generation: u32,
    value: Option<T>,
}

/// A store of `T` addressed by generation-tagged [`Key`]s.
///
/// `insert` is amortized O(1) and reuses a freed slot when one is available, so
/// a create/destroy cycle occupies bounded address space. `get`/`get_mut`/
/// `remove` are O(1) and reject stale keys.
pub struct SlotMap<T> {
    slots: Vec<Slot<T>>,
    /// Indices of slots that are free and not retired, newest first.
    free: Vec<u32>,
    live: usize,
}

impl<T> Default for SlotMap<T> {
    fn default() -> SlotMap<T> {
        SlotMap::new()
    }
}

impl<T> SlotMap<T> {
    pub const fn new() -> SlotMap<T> {
        SlotMap {
            slots: Vec::new(),
            free: Vec::new(),
            live: 0,
        }
    }

    /// Store `value` and return its key. Reuses a previously freed slot when
    /// one is available; the returned key carries that slot's CURRENT
    /// generation, so keys issued before the free stay invalid.
    pub fn insert(&mut self, value: T) -> Key<T> {
        self.live += 1;
        if let Some(index) = self.free.pop() {
            let slot = &mut self.slots[index as usize];
            debug_assert!(slot.value.is_none(), "free list held a live slot");
            slot.value = Some(value);
            return Key {
                index,
                generation: slot.generation,
                owner: PhantomData,
            };
        }
        // A slot index is a u32 in the key, so the store cannot silently wrap
        // past that and start reissuing live indices. Reaching it would take
        // 4 billion simultaneously-live GPU resources, so this is a guard
        // against a future caller, not a case that can arise today.
        assert!(
            self.slots.len() < u32::MAX as usize,
            "slot map exhausted: {} slots",
            self.slots.len()
        );
        let index = self.slots.len() as u32;
        // Generation 1 for a fresh slot: generation 0 is reserved so that a
        // zeroed handle can never be valid.
        self.slots.push(Slot {
            generation: 1,
            value: Some(value),
        });
        Key {
            index,
            generation: 1,
            owner: PhantomData,
        }
    }

    /// The value `key` refers to, or `None` if the key is stale (its slot was
    /// freed, and possibly refilled) or out of range.
    pub fn get(&self, key: Key<T>) -> Option<&T> {
        let slot = self.slots.get(key.index as usize)?;
        if slot.generation != key.generation {
            return None;
        }
        slot.value.as_ref()
    }

    /// Mutable counterpart of [`Self::get`].
    pub fn get_mut(&mut self, key: Key<T>) -> Option<&mut T> {
        let slot = self.slots.get_mut(key.index as usize)?;
        if slot.generation != key.generation {
            return None;
        }
        slot.value.as_mut()
    }

    /// Whether `key` still names a live value.
    pub fn contains(&self, key: Key<T>) -> bool {
        self.get(key).is_some()
    }

    /// Free `key`'s slot and return the value, or `None` if the key was already
    /// stale. Bumping the generation invalidates every outstanding handle to
    /// that slot; a slot at [`MAX_GENERATION`] is retired instead of reused, so
    /// the generation can never wrap back onto a live handle.
    pub fn remove(&mut self, key: Key<T>) -> Option<T> {
        let slot = self.slots.get_mut(key.index as usize)?;
        if slot.generation != key.generation {
            return None;
        }
        let value = slot.value.take()?;
        self.live -= 1;
        if slot.generation < MAX_GENERATION {
            slot.generation += 1;
            self.free.push(key.index);
        }
        Some(value)
    }

    /// Number of live values.
    pub fn len(&self) -> usize {
        self.live
    }

    pub fn is_empty(&self) -> bool {
        self.live == 0
    }

    /// Number of slots ever allocated, live or free. Bounded growth means this
    /// plateaus while [`Self::len`] cycles; a test asserts exactly that.
    pub fn slot_count(&self) -> usize {
        self.slots.len()
    }

    /// Every live value, in slot order.
    pub fn values(&self) -> impl Iterator<Item = &T> {
        self.slots.iter().filter_map(|s| s.value.as_ref())
    }

    /// Every live `(key, value)`, in slot order.
    pub fn iter(&self) -> impl Iterator<Item = (Key<T>, &T)> {
        self.slots.iter().enumerate().filter_map(|(i, s)| {
            s.value.as_ref().map(|v| {
                (
                    Key {
                        index: i as u32,
                        generation: s.generation,
                        owner: PhantomData,
                    },
                    v,
                )
            })
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_freed_slot_is_reused_but_the_old_key_is_rejected() {
        let mut m: SlotMap<i32> = SlotMap::new();
        let a = m.insert(10);
        assert_eq!(m.remove(a), Some(10));
        // The next insert lands in the SAME slot...
        let b = m.insert(20);
        assert_eq!(a.slot(), b.slot(), "the free slot should be reused");
        assert_eq!(m.slot_count(), 1, "reuse must not allocate a second slot");
        // ...and the stale key must not read it.
        assert_eq!(m.get(a), None, "a stale key aliased a live value");
        assert_eq!(m.get(b), Some(&20));
        assert_ne!(a, b, "keys to the same slot must differ by generation");
    }

    #[test]
    fn removing_twice_reports_the_second_removal_as_stale() {
        let mut m: SlotMap<i32> = SlotMap::new();
        let a = m.insert(1);
        assert_eq!(m.remove(a), Some(1));
        assert_eq!(m.remove(a), None, "double free must be rejected");
        assert_eq!(m.len(), 0);
    }

    #[test]
    fn create_and_destroy_cycles_occupy_bounded_slots() {
        let mut m: SlotMap<i32> = SlotMap::new();
        for i in 0..10_000 {
            let k = m.insert(i);
            m.remove(k).unwrap();
        }
        assert_eq!(m.len(), 0);
        assert_eq!(m.slot_count(), 1, "one slot should serve every cycle");
    }

    #[test]
    fn i64_round_trip_is_non_negative_and_rejects_junk() {
        let mut m: SlotMap<i32> = SlotMap::new();
        let a = m.insert(7);
        let raw = a.to_i64();
        assert!(raw > 0, "a handle must be positive, got {raw}");
        assert_eq!(Key::<i32>::from_i64(raw), Some(a));
        // The runtime's failure sentinel and a zeroed handle are both invalid.
        assert_eq!(Key::<i32>::from_i64(-1), None);
        assert_eq!(Key::<i32>::from_i64(0), None);
        // Generation 0 in the high bits can never be issued, whatever the index.
        assert_eq!(Key::<i32>::from_i64(5), None);
    }

    #[test]
    fn a_key_from_a_far_slot_is_still_rejected_when_out_of_range() {
        let m: SlotMap<i32> = SlotMap::new();
        let bogus = Key::<i32>::from_i64((1i64 << 32) | 9999).unwrap();
        assert_eq!(m.get(bogus), None);
        assert!(!m.contains(bogus));
    }

    #[test]
    fn an_exhausted_slot_is_retired_rather_than_wrapped() {
        let mut m: SlotMap<i32> = SlotMap::new();
        let a = m.insert(1);
        // Fast-forward the slot to its last generation.
        m.slots[a.slot() as usize].generation = MAX_GENERATION;
        let last = Key::<i32>::from_i64(((MAX_GENERATION as i64) << 32) | a.slot() as i64).unwrap();
        assert_eq!(m.remove(last), Some(1));
        assert!(
            m.free.is_empty(),
            "an exhausted slot must not go back on the free list"
        );
        let b = m.insert(2);
        assert_ne!(a.slot(), b.slot(), "the retired slot was handed out again");
        // ...and the exhausted key still reads nothing.
        assert_eq!(m.get(last), None);
    }

    #[test]
    fn the_placeholder_key_never_resolves() {
        let mut m: SlotMap<i32> = SlotMap::new();
        let p = Key::<i32>::PLACEHOLDER;
        assert_eq!(m.get(p), None);
        assert!(!m.contains(p));
        assert_eq!(m.remove(p), None);
        // ...even once slot 0 is occupied, which is the case that matters.
        let real = m.insert(5);
        assert_eq!(real.slot(), p.slot(), "the test needs them to share a slot");
        assert_eq!(m.get(p), None, "the placeholder aliased a live value");
        assert_ne!(real, p);
        // It is also not a value any program could have received over the ABI.
        assert_eq!(Key::<i32>::from_i64(p.to_i64()), None);
    }

    #[test]
    fn iteration_sees_only_live_values() {
        let mut m: SlotMap<i32> = SlotMap::new();
        let a = m.insert(1);
        let _b = m.insert(2);
        let c = m.insert(3);
        m.remove(a);
        m.remove(c);
        let seen: Vec<i32> = m.values().copied().collect();
        assert_eq!(seen, vec![2]);
        assert_eq!(m.iter().count(), 1);
        assert_eq!(m.len(), 1);
        assert_eq!(m.slot_count(), 3);
    }
}
