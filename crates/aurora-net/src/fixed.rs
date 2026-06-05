//! Deterministic fixed-point arithmetic (netcode spec §8.2).
//!
//! Cross-platform lockstep / P2P rollback needs bit-identical math on every
//! machine, which IEEE floats don't guarantee (FMA, rounding, transcendentals).
//! `Fixed` is a Q16.16 fixed-point number backed entirely by `i32`/`i64` integer
//! ops, so `a * b` is identical everywhere. It's the deterministic alternative
//! the `@deterministic` simulation path would use.

use std::ops::{Add, Div, Mul, Neg, Sub};

const SHIFT: u32 = 16;
const ONE: i32 = 1 << SHIFT;

/// A Q16.16 fixed-point number.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct Fixed(pub i32);

impl Fixed {
    pub const ZERO: Fixed = Fixed(0);
    pub const ONE: Fixed = Fixed(ONE);

    pub fn from_int(n: i32) -> Fixed {
        Fixed(n << SHIFT)
    }

    pub fn from_f32(x: f32) -> Fixed {
        Fixed((x * ONE as f32).round() as i32)
    }

    pub fn to_f32(self) -> f32 {
        self.0 as f32 / ONE as f32
    }

    pub fn to_int(self) -> i32 {
        self.0 >> SHIFT
    }

    pub fn abs(self) -> Fixed {
        Fixed(self.0.abs())
    }

    /// Deterministic fixed-point square root via integer `isqrt` (no floats).
    /// For `x = v / 2^16`, `sqrt(x) = isqrt(v << 16) / 2^16`.
    pub fn sqrt(self) -> Fixed {
        if self.0 <= 0 {
            return Fixed::ZERO;
        }
        let widened = (self.0 as u64) << SHIFT;
        Fixed(widened.isqrt() as i32)
    }
}

/// A deterministic 3-D vector of fixed-point numbers (for lockstep physics).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct FVec3 {
    pub x: Fixed,
    pub y: Fixed,
    pub z: Fixed,
}

impl FVec3 {
    pub const ZERO: FVec3 = FVec3 { x: Fixed::ZERO, y: Fixed::ZERO, z: Fixed::ZERO };

    pub fn new(x: Fixed, y: Fixed, z: Fixed) -> FVec3 {
        FVec3 { x, y, z }
    }

    pub fn from_ints(x: i32, y: i32, z: i32) -> FVec3 {
        FVec3::new(Fixed::from_int(x), Fixed::from_int(y), Fixed::from_int(z))
    }

    pub fn add(self, o: FVec3) -> FVec3 {
        FVec3::new(self.x + o.x, self.y + o.y, self.z + o.z)
    }

    pub fn sub(self, o: FVec3) -> FVec3 {
        FVec3::new(self.x - o.x, self.y - o.y, self.z - o.z)
    }

    pub fn scale(self, s: Fixed) -> FVec3 {
        FVec3::new(self.x * s, self.y * s, self.z * s)
    }

    pub fn dot(self, o: FVec3) -> Fixed {
        self.x * o.x + self.y * o.y + self.z * o.z
    }

    pub fn length_sq(self) -> Fixed {
        self.dot(self)
    }

    pub fn length(self) -> Fixed {
        self.length_sq().sqrt()
    }
}

impl Add for Fixed {
    type Output = Fixed;
    fn add(self, rhs: Fixed) -> Fixed {
        Fixed(self.0.wrapping_add(rhs.0))
    }
}

impl Sub for Fixed {
    type Output = Fixed;
    fn sub(self, rhs: Fixed) -> Fixed {
        Fixed(self.0.wrapping_sub(rhs.0))
    }
}

impl Neg for Fixed {
    type Output = Fixed;
    fn neg(self) -> Fixed {
        Fixed(-self.0)
    }
}

impl Mul for Fixed {
    type Output = Fixed;
    fn mul(self, rhs: Fixed) -> Fixed {
        // Widen to i64 to keep the full product, then shift back.
        Fixed(((self.0 as i64 * rhs.0 as i64) >> SHIFT) as i32)
    }
}

impl Div for Fixed {
    type Output = Fixed;
    fn div(self, rhs: Fixed) -> Fixed {
        debug_assert!(rhs.0 != 0, "fixed-point division by zero");
        Fixed((((self.0 as i64) << SHIFT) / rhs.0 as i64) as i32)
    }
}

#[cfg(test)]
mod fixed_tests {
    use super::*;

    #[test]
    fn int_round_trip() {
        assert_eq!(Fixed::from_int(5).to_int(), 5);
        assert_eq!(Fixed::from_int(-3).to_int(), -3);
    }

    #[test]
    fn arithmetic_is_exact_for_representable_values() {
        let a = Fixed::from_int(3);
        let b = Fixed::from_int(4);
        assert_eq!((a + b).to_int(), 7);
        assert_eq!((b - a).to_int(), 1);
        assert_eq!((a * b).to_int(), 12);
        // Division quantizes; close to 1.333..., not bit-exact to the f32 quotient.
        assert!(((b / a).to_f32() - 4.0 / 3.0).abs() < 1e-3);
    }

    #[test]
    fn fractional_round_trip_is_close() {
        for &x in &[0.5f32, -2.25, 3.75, 0.1, -0.0078125] {
            let f = Fixed::from_f32(x);
            assert!((f.to_f32() - x).abs() < 1.0 / ONE as f32);
        }
    }

    #[test]
    fn multiplication_is_deterministic_and_associative_in_integers() {
        // The whole point: identical inputs -> identical bits, anywhere.
        let a = Fixed::from_f32(1.5);
        let b = Fixed::from_f32(2.5);
        let r1 = a * b;
        let r2 = a * b;
        assert_eq!(r1.0, r2.0); // bit-identical
        assert_eq!(r1.to_f32(), 3.75);
    }

    #[test]
    fn ordering_works() {
        assert!(Fixed::from_f32(1.5) < Fixed::from_f32(1.6));
        assert!(Fixed::from_int(-1) < Fixed::ZERO);
        assert_eq!(Fixed::from_int(1), Fixed::ONE);
    }

    #[test]
    fn fixed_sqrt_is_deterministic_and_close() {
        assert_eq!(Fixed::from_int(4).sqrt().to_int(), 2);
        assert_eq!(Fixed::from_int(144).sqrt().to_int(), 12);
        // sqrt(2) ~ 1.4142, within fixed-point precision.
        assert!((Fixed::from_int(2).sqrt().to_f32() - std::f32::consts::SQRT_2).abs() < 1e-2);
        // Bit-identical across calls.
        assert_eq!(Fixed::from_f32(7.5).sqrt().0, Fixed::from_f32(7.5).sqrt().0);
    }

    #[test]
    fn fvec3_length_and_dot() {
        let v = FVec3::from_ints(3, 4, 0);
        assert_eq!(v.length().to_int(), 5); // 3-4-5
        assert_eq!(v.length_sq().to_int(), 25);

        let a = FVec3::from_ints(1, 0, 0);
        let b = FVec3::from_ints(0, 1, 0);
        assert_eq!(a.dot(b), Fixed::ZERO); // orthogonal
    }

    #[test]
    fn fvec3_add_scale_is_deterministic_motion() {
        // pos += vel * dt over ticks, fully deterministic.
        let vel = FVec3::new(Fixed::from_f32(2.0), Fixed::from_f32(-1.0), Fixed::ZERO);
        let dt = Fixed::from_f32(0.5);
        let mut pos = FVec3::ZERO;
        for _ in 0..4 {
            pos = pos.add(vel.scale(dt));
        }
        assert_eq!(pos, FVec3::from_ints(4, -2, 0)); // 4 ticks * (2,-1,0) * 0.5
    }

    #[test]
    fn simulating_motion_stays_integer_deterministic() {
        // pos += vel * dt, repeated — the kind of update a lockstep sim runs.
        let dt = Fixed::from_f32(0.5);
        let vel = Fixed::from_f32(3.0);
        let mut pos = Fixed::ZERO;
        for _ in 0..10 {
            pos = pos + vel * dt;
        }
        assert_eq!(pos.to_f32(), 15.0); // 10 * 3 * 0.5
    }
}
