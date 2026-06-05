//! Deterministic lockstep simulation (netcode spec §9, building on §8.2).
//!
//! In lockstep / P2P rollback, peers exchange only *inputs* and each simulates
//! independently — which is only correct if the simulation is bit-identical
//! everywhere. Using [`Fixed`]/[`FVec3`] (integer-backed) makes that true. This
//! module is a minimal physics body demonstrating the two properties rollback
//! depends on:
//!
//! * **Determinism** — same initial state + same inputs ⇒ bit-identical results
//!   on independent instances.
//! * **Rollback reproducibility** — snapshot a state, advance, restore the
//!   snapshot, re-advance the same inputs ⇒ exactly the original result.

use crate::fixed::{FVec3, Fixed};

/// A point-mass body integrated with a fixed timestep.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Body {
    pub pos: FVec3,
    pub vel: FVec3,
}

impl Body {
    pub fn new(pos: FVec3, vel: FVec3) -> Body {
        Body { pos, vel }
    }

    /// Advance one tick: semi-implicit Euler with a per-tick acceleration input.
    pub fn step(&mut self, accel: FVec3, dt: Fixed) {
        self.vel = self.vel.add(accel.scale(dt));
        self.pos = self.pos.add(self.vel.scale(dt));
    }

    /// Run a whole input sequence (one acceleration per tick).
    pub fn run(&mut self, inputs: &[FVec3], dt: Fixed) {
        for &a in inputs {
            self.step(a, dt);
        }
    }
}

#[cfg(test)]
mod lockstep_tests {
    use super::*;

    fn gravity_then_thrust(n: usize) -> Vec<FVec3> {
        // A deterministic, slightly varied input stream.
        (0..n)
            .map(|i| {
                let up = if i % 3 == 0 { 2 } else { -1 };
                FVec3::from_ints((i as i32 % 5) - 2, up, 0)
            })
            .collect()
    }

    #[test]
    fn two_peers_stay_bit_identical() {
        let dt = Fixed::from_f32(0.5);
        let inputs = gravity_then_thrust(200);

        let mut peer_a = Body::new(FVec3::ZERO, FVec3::ZERO);
        let mut peer_b = Body::new(FVec3::ZERO, FVec3::ZERO);
        peer_a.run(&inputs, dt);
        peer_b.run(&inputs, dt);

        // Bit-identical (FVec3 is Eq over integer-backed Fixed).
        assert_eq!(peer_a, peer_b);
    }

    #[test]
    fn rollback_resimulation_reproduces_exactly() {
        let dt = Fixed::from_f32(0.25);
        let inputs = gravity_then_thrust(20);

        // Reference run: advance all 20 ticks.
        let mut reference = Body::new(FVec3::from_ints(1, 2, 3), FVec3::ZERO);
        reference.run(&inputs, dt);

        // Rollback run: advance 12 ticks, snapshot, advance the rest...
        let mut rolled = Body::new(FVec3::from_ints(1, 2, 3), FVec3::ZERO);
        rolled.run(&inputs[..12], dt);
        let snapshot = rolled; // cheap Copy snapshot
        rolled.run(&inputs[12..], dt);
        assert_eq!(rolled, reference);

        // ...now restore the snapshot and replay the same tail — identical.
        let mut replayed = snapshot;
        replayed.run(&inputs[12..], dt);
        assert_eq!(replayed, reference, "replay from a snapshot must reproduce exactly");
    }

    #[test]
    fn divergent_inputs_diverge() {
        // Sanity: different inputs must produce different state (the sim isn't
        // trivially constant).
        let dt = Fixed::from_f32(0.5);
        let mut a = Body::new(FVec3::ZERO, FVec3::ZERO);
        let mut b = Body::new(FVec3::ZERO, FVec3::ZERO);
        a.run(&gravity_then_thrust(50), dt);
        b.run(&[FVec3::from_ints(1, 0, 0); 50], dt);
        assert_ne!(a, b);
    }
}
