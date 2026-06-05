//! Client-side snapshot interpolation for `@interp` fields (netcode spec §5).
//!
//! Remote (non-predicted) entities are rendered a fixed delay behind the latest
//! received snapshot, interpolating between the two samples straddling the
//! render time. This trades a little latency for smooth motion that's robust to
//! jitter. On buffer underrun (a late/dropped snapshot) it extrapolates from the
//! last two samples for a bounded amount before clamping — so a stalled peer
//! doesn't freeze *and* doesn't shoot off across the map.
//!
//! Pairs with `Predictor` (§6): the local player is predicted; everyone else is
//! interpolated.

use std::collections::VecDeque;

use crate::lagcomp::V3;

fn lerp(a: V3, b: V3, t: f32) -> V3 {
    [
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
    ]
}

pub struct InterpBuffer {
    /// (tick, position) samples, kept sorted by ascending tick.
    samples: VecDeque<(f32, V3)>,
    /// How far behind the latest sample to render.
    delay: f32,
    /// Max ticks to extrapolate past the last sample before clamping.
    max_extrapolation: f32,
    capacity: usize,
}

impl InterpBuffer {
    pub fn new(delay: f32) -> InterpBuffer {
        InterpBuffer { samples: VecDeque::new(), delay, max_extrapolation: 2.0, capacity: 32 }
    }

    pub fn with_extrapolation(mut self, ticks: f32) -> InterpBuffer {
        self.max_extrapolation = ticks;
        self
    }

    /// Record an authoritative sample at `tick`.
    pub fn push(&mut self, tick: f32, pos: V3) {
        // Maintain ascending order (snapshots usually arrive in order, but be
        // robust to mild reordering).
        let idx = self.samples.partition_point(|&(t, _)| t < tick);
        if self.samples.get(idx).map(|&(t, _)| t) == Some(tick) {
            self.samples[idx].1 = pos; // replace same-tick sample
        } else {
            self.samples.insert(idx, (tick, pos));
        }
        while self.samples.len() > self.capacity {
            self.samples.pop_front();
        }
    }

    pub fn latest_tick(&self) -> Option<f32> {
        self.samples.back().map(|&(t, _)| t)
    }

    /// Interpolated position to render at simulation time `now`, accounting for
    /// the interpolation delay.
    pub fn sample(&self, now: f32) -> Option<V3> {
        if self.samples.is_empty() {
            return None;
        }
        let render = now - self.delay;
        let first = *self.samples.front().unwrap();
        let last = *self.samples.back().unwrap();

        // Before the buffer: clamp to the earliest sample.
        if render <= first.0 {
            return Some(first.1);
        }

        // After the buffer: extrapolate from the last two, bounded.
        if render >= last.0 {
            if self.samples.len() < 2 {
                return Some(last.1);
            }
            let prev = self.samples[self.samples.len() - 2];
            let span = last.0 - prev.0;
            if span <= 0.0 {
                return Some(last.1);
            }
            let over = (render - last.0).min(self.max_extrapolation);
            let t = 1.0 + over / span;
            return Some(lerp(prev.1, last.1, t));
        }

        // Interpolate between the two straddling samples.
        let i = self.samples.partition_point(|&(t, _)| t <= render);
        let (t0, p0) = self.samples[i - 1];
        let (t1, p1) = self.samples[i];
        let span = t1 - t0;
        let alpha = if span > 0.0 { (render - t0) / span } else { 0.0 };
        Some(lerp(p0, p1, alpha))
    }
}

#[cfg(test)]
mod snapshot_tests {
    use super::*;

    fn approx(a: V3, b: V3) -> bool {
        (0..3).all(|i| (a[i] - b[i]).abs() < 1e-4)
    }

    #[test]
    fn interpolates_between_samples() {
        let mut buf = InterpBuffer::new(0.0);
        buf.push(0.0, [0.0, 0.0, 0.0]);
        buf.push(10.0, [10.0, 20.0, 0.0]);
        // Halfway in time -> halfway in space.
        assert!(approx(buf.sample(5.0).unwrap(), [5.0, 10.0, 0.0]));
        assert!(approx(buf.sample(2.5).unwrap(), [2.5, 5.0, 0.0]));
    }

    #[test]
    fn delay_renders_in_the_past() {
        let mut buf = InterpBuffer::new(10.0);
        buf.push(0.0, [0.0, 0.0, 0.0]);
        buf.push(10.0, [10.0, 0.0, 0.0]);
        // now=15, delay=10 -> render at tick 5 -> [5,0,0].
        assert!(approx(buf.sample(15.0).unwrap(), [5.0, 0.0, 0.0]));
    }

    #[test]
    fn clamps_before_the_first_sample() {
        let mut buf = InterpBuffer::new(0.0);
        buf.push(5.0, [5.0, 0.0, 0.0]);
        buf.push(10.0, [10.0, 0.0, 0.0]);
        assert!(approx(buf.sample(0.0).unwrap(), [5.0, 0.0, 0.0]));
    }

    #[test]
    fn extrapolates_past_the_last_sample_but_is_bounded() {
        let mut buf = InterpBuffer::new(0.0).with_extrapolation(1.0);
        buf.push(0.0, [0.0, 0.0, 0.0]);
        buf.push(10.0, [10.0, 0.0, 0.0]); // velocity 1.0/tick
        // 1 tick past -> extrapolate to 11.
        assert!(approx(buf.sample(11.0).unwrap(), [11.0, 0.0, 0.0]));
        // Far past -> clamped to last + max_extrapolation (1 tick) -> 11, not 100.
        assert!(approx(buf.sample(100.0).unwrap(), [11.0, 0.0, 0.0]));
    }

    #[test]
    fn empty_buffer_yields_none() {
        let buf = InterpBuffer::new(0.0);
        assert!(buf.sample(1.0).is_none());
    }
}
