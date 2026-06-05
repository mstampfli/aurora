//! Bit-level packing and quaternion compression (netcode spec §3.2).
//!
//! Snapshot deltas are packed to the bit, not the byte: a bool costs 1 bit, a
//! quantized field costs exactly its declared width. [`BitWriter`]/[`BitReader`]
//! are the primitive; `write_quat`/`read_quat` implement the **smallest-three**
//! rotation encoding (store the 2-bit index of the largest component and the
//! other three quantized; reconstruct the largest from unit-length).

/// Writes bits LSB-first into a byte buffer.
#[derive(Default)]
pub struct BitWriter {
    bytes: Vec<u8>,
    cur: u8,
    nbits: u8,
}

impl BitWriter {
    pub fn new() -> BitWriter {
        BitWriter::default()
    }

    pub fn write_bool(&mut self, b: bool) {
        self.cur |= (b as u8) << self.nbits;
        self.nbits += 1;
        if self.nbits == 8 {
            self.bytes.push(self.cur);
            self.cur = 0;
            self.nbits = 0;
        }
    }

    /// Write the low `n` bits of `value` (n in 0..=64).
    pub fn write_bits(&mut self, value: u64, n: u32) {
        for i in 0..n {
            self.write_bool((value >> i) & 1 == 1);
        }
    }

    /// Flush and return the packed bytes (zero-padded to a byte boundary).
    pub fn finish(mut self) -> Vec<u8> {
        if self.nbits > 0 {
            self.bytes.push(self.cur);
        }
        self.bytes
    }

    /// Number of bits written so far.
    pub fn bit_len(&self) -> usize {
        self.bytes.len() * 8 + self.nbits as usize
    }
}

/// Reads bits LSB-first from a byte buffer.
pub struct BitReader<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> BitReader<'a> {
    pub fn new(bytes: &'a [u8]) -> BitReader<'a> {
        BitReader { bytes, pos: 0 }
    }

    pub fn read_bool(&mut self) -> bool {
        let byte = self.pos / 8;
        let bit = self.pos % 8;
        self.pos += 1;
        self.bytes.get(byte).is_some_and(|b| (b >> bit) & 1 == 1)
    }

    pub fn read_bits(&mut self, n: u32) -> u64 {
        let mut v = 0u64;
        for i in 0..n {
            if self.read_bool() {
                v |= 1 << i;
            }
        }
        v
    }
}

const INV_SQRT2: f32 = 0.707_106_77;

fn quantize(v: f32, range: f32, bits: u32) -> u64 {
    let max = ((1u64 << bits) - 1) as f32;
    let norm = ((v + range) / (2.0 * range)).clamp(0.0, 1.0);
    (norm * max).round() as u64
}

fn dequantize(q: u64, range: f32, bits: u32) -> f32 {
    let max = ((1u64 << bits) - 1) as f32;
    (q as f32 / max) * 2.0 * range - range
}

/// Encode a unit quaternion `[x, y, z, w]` with the smallest-three method:
/// 2 bits for the largest-component index + `bits` per remaining component.
pub fn write_quat(w: &mut BitWriter, q: [f32; 4], bits: u32) {
    // Index of the largest-magnitude component.
    let mut largest = 0usize;
    for i in 1..4 {
        if q[i].abs() > q[largest].abs() {
            largest = i;
        }
    }
    // q and -q are the same rotation; choose the sign that makes the dropped
    // (largest) component positive, so it reconstructs unambiguously.
    let sign = if q[largest] < 0.0 { -1.0 } else { 1.0 };

    w.write_bits(largest as u64, 2);
    for i in 0..4 {
        if i != largest {
            w.write_bits(quantize(q[i] * sign, INV_SQRT2, bits), bits);
        }
    }
}

/// Decode a quaternion written by [`write_quat`].
pub fn read_quat(r: &mut BitReader, bits: u32) -> [f32; 4] {
    let largest = r.read_bits(2) as usize;
    let mut q = [0.0f32; 4];
    let mut sum_sq = 0.0;
    for i in 0..4 {
        if i != largest {
            let v = dequantize(r.read_bits(bits), INV_SQRT2, bits);
            q[i] = v;
            sum_sq += v * v;
        }
    }
    q[largest] = (1.0 - sum_sq).max(0.0).sqrt();
    q
}

#[cfg(test)]
mod bit_tests {
    use super::*;

    #[test]
    fn bits_round_trip_at_various_widths() {
        let mut w = BitWriter::new();
        w.write_bool(true);
        w.write_bits(5, 3); // 101
        w.write_bits(1000, 10);
        w.write_bool(false);
        w.write_bits(0xABCD, 16);
        let bytes = w.finish();

        let mut r = BitReader::new(&bytes);
        assert!(r.read_bool());
        assert_eq!(r.read_bits(3), 5);
        assert_eq!(r.read_bits(10), 1000);
        assert!(!r.read_bool());
        assert_eq!(r.read_bits(16), 0xABCD);
    }

    #[test]
    fn bool_costs_one_bit() {
        let mut w = BitWriter::new();
        for _ in 0..10 {
            w.write_bool(true);
        }
        assert_eq!(w.bit_len(), 10);
        assert_eq!(w.finish().len(), 2); // 10 bits -> 2 bytes
    }

    fn normalize(q: [f32; 4]) -> [f32; 4] {
        let n = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
        [q[0] / n, q[1] / n, q[2] / n, q[3] / n]
    }

    fn quat_dot(a: [f32; 4], b: [f32; 4]) -> f32 {
        a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]
    }

    #[test]
    fn quaternion_smallest_three_round_trips_closely() {
        let cases = [
            [0.0, 0.0, 0.0, 1.0],          // identity
            normalize([0.3, -0.5, 0.2, 0.78]),
            normalize([-0.6, 0.1, 0.7, -0.3]),
            normalize([0.5, 0.5, 0.5, 0.5]),
        ];
        for q in cases {
            let mut w = BitWriter::new();
            write_quat(&mut w, q, 12);
            let bytes = w.finish();
            let mut r = BitReader::new(&bytes);
            let back = read_quat(&mut r, 12);
            // Same rotation: |dot| ~ 1 (allowing the q/-q double cover).
            assert!(quat_dot(q, back).abs() > 0.999, "q={q:?} back={back:?}");
        }
    }

    #[test]
    fn quaternion_encoding_is_compact() {
        // 2-bit index + 3 * 12-bit components = 38 bits -> 5 bytes, vs 16 raw.
        let mut w = BitWriter::new();
        write_quat(&mut w, [0.0, 0.0, 0.0, 1.0], 12);
        assert_eq!(w.bit_len(), 2 + 3 * 12);
        assert_eq!(w.finish().len(), 5);
    }
}
