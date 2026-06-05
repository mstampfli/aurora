//! Deterministic, replicated RNG (netcode spec §8.1).
//!
//! Predicted systems may only use a *replicated* RNG: client and server seed it
//! identically, so both draw the same sequence and prediction replay reproduces
//! the server bit-for-bit. This is a SplitMix64 generator — tiny, fast, and
//! fully deterministic across machines (integer-only, no float nondeterminism).

#[derive(Clone)]
pub struct Rng {
    state: u64,
}

impl Rng {
    /// Seed the generator. The same seed always yields the same sequence.
    pub fn new(seed: u64) -> Rng {
        Rng { state: seed }
    }

    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    pub fn next_u32(&mut self) -> u32 {
        (self.next_u64() >> 32) as u32
    }

    /// A float in `[0, 1)` (24 bits of mantissa precision).
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u32 << 24) as f32
    }

    /// A value in `[lo, hi)` (returns `lo` if the range is empty).
    pub fn range(&mut self, lo: i64, hi: i64) -> i64 {
        if hi <= lo {
            return lo;
        }
        let span = (hi - lo) as u64;
        lo + (self.next_u64() % span) as i64
    }
}

#[cfg(test)]
mod rng_tests {
    use super::*;

    #[test]
    fn same_seed_same_sequence() {
        // The replication guarantee: two endpoints seeded alike agree exactly.
        let mut a = Rng::new(0xDEAD_BEEF);
        let mut b = Rng::new(0xDEAD_BEEF);
        for _ in 0..1000 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn different_seeds_diverge() {
        let mut a = Rng::new(1);
        let mut b = Rng::new(2);
        let sa: Vec<u64> = (0..8).map(|_| a.next_u64()).collect();
        let sb: Vec<u64> = (0..8).map(|_| b.next_u64()).collect();
        assert_ne!(sa, sb);
    }

    #[test]
    fn floats_are_in_unit_interval() {
        let mut r = Rng::new(42);
        for _ in 0..10_000 {
            let f = r.next_f32();
            assert!((0.0..1.0).contains(&f), "out of range: {f}");
        }
    }

    #[test]
    fn range_stays_within_bounds() {
        let mut r = Rng::new(7);
        for _ in 0..10_000 {
            let v = r.range(-5, 5);
            assert!((-5..5).contains(&v));
        }
    }

    #[test]
    fn replay_reproduces_after_clone() {
        // Cloning captures exact state — used to snapshot RNG for prediction replay.
        let mut r = Rng::new(99);
        for _ in 0..10 {
            r.next_u64();
        }
        let mut snapshot = r.clone();
        let live: Vec<u64> = (0..5).map(|_| r.next_u64()).collect();
        let replayed: Vec<u64> = (0..5).map(|_| snapshot.next_u64()).collect();
        assert_eq!(live, replayed);
    }
}
