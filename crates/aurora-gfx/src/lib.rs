//! CPU rasterization foundation for Aurora's builtin graphics (grammar/engine
//! spec — the renderer's software path).
//!
//! A full GPU renderer (wgpu + SPIR-V shader lowering) is future work and needs
//! a window/device; this is the headless, deterministic foundation: an RGBA
//! [`Framebuffer`] with clear, point, line, and a barycentric **triangle
//! rasterizer** with per-vertex color interpolation, plus binary PPM output so
//! results are inspectable and testable.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Color {
    pub r: u8,
    pub g: u8,
    pub b: u8,
    pub a: u8,
}

impl Color {
    pub const fn rgb(r: u8, g: u8, b: u8) -> Color {
        Color { r, g, b, a: 255 }
    }
    pub const BLACK: Color = Color::rgb(0, 0, 0);
    pub const WHITE: Color = Color::rgb(255, 255, 255);

    fn lerp3(a: Color, b: Color, c: Color, w: [f32; 3]) -> Color {
        let mix = |x: u8, y: u8, z: u8| {
            (x as f32 * w[0] + y as f32 * w[1] + z as f32 * w[2]).round().clamp(0.0, 255.0) as u8
        };
        Color {
            r: mix(a.r, b.r, c.r),
            g: mix(a.g, b.g, c.g),
            b: mix(a.b, b.b, c.b),
            a: 255,
        }
    }
}

/// A 2-D vertex position in pixel space.
pub type P2 = [f32; 2];

pub struct Framebuffer {
    width: u32,
    height: u32,
    pixels: Vec<Color>,
}

impl Framebuffer {
    pub fn new(width: u32, height: u32) -> Framebuffer {
        Framebuffer { width, height, pixels: vec![Color::BLACK; (width * height) as usize] }
    }

    pub fn width(&self) -> u32 {
        self.width
    }
    pub fn height(&self) -> u32 {
        self.height
    }

    pub fn clear(&mut self, color: Color) {
        for p in &mut self.pixels {
            *p = color;
        }
    }

    pub fn get(&self, x: u32, y: u32) -> Color {
        self.pixels[(y * self.width + x) as usize]
    }

    pub fn set(&mut self, x: i32, y: i32, color: Color) {
        if x >= 0 && y >= 0 && (x as u32) < self.width && (y as u32) < self.height {
            self.pixels[(y as u32 * self.width + x as u32) as usize] = color;
        }
    }

    /// Bresenham line.
    pub fn line(&mut self, a: P2, b: P2, color: Color) {
        let (mut x0, mut y0) = (a[0] as i32, a[1] as i32);
        let (x1, y1) = (b[0] as i32, b[1] as i32);
        let dx = (x1 - x0).abs();
        let dy = -(y1 - y0).abs();
        let sx = if x0 < x1 { 1 } else { -1 };
        let sy = if y0 < y1 { 1 } else { -1 };
        let mut err = dx + dy;
        loop {
            self.set(x0, y0, color);
            if x0 == x1 && y0 == y1 {
                break;
            }
            let e2 = 2 * err;
            if e2 >= dy {
                err += dy;
                x0 += sx;
            }
            if e2 <= dx {
                err += dx;
                y0 += sy;
            }
        }
    }

    /// Filled triangle with per-vertex colors (barycentric interpolation).
    pub fn triangle(&mut self, p: [P2; 3], colors: [Color; 3]) {
        let area = edge(p[0], p[1], p[2]);
        if area == 0.0 {
            return; // degenerate
        }

        // Bounding box, clamped to the framebuffer.
        let min_x = p.iter().map(|q| q[0]).fold(f32::INFINITY, f32::min).floor().max(0.0) as i32;
        let max_x = p
            .iter()
            .map(|q| q[0])
            .fold(f32::NEG_INFINITY, f32::max)
            .ceil()
            .min((self.width - 1) as f32) as i32;
        let min_y = p.iter().map(|q| q[1]).fold(f32::INFINITY, f32::min).floor().max(0.0) as i32;
        let max_y = p
            .iter()
            .map(|q| q[1])
            .fold(f32::NEG_INFINITY, f32::max)
            .ceil()
            .min((self.height - 1) as f32) as i32;

        for y in min_y..=max_y {
            for x in min_x..=max_x {
                let pt = [x as f32 + 0.5, y as f32 + 0.5];
                let w0 = edge(p[1], p[2], pt);
                let w1 = edge(p[2], p[0], pt);
                let w2 = edge(p[0], p[1], pt);
                // Inside if all edge functions share the triangle's orientation.
                let inside = if area > 0.0 {
                    w0 >= 0.0 && w1 >= 0.0 && w2 >= 0.0
                } else {
                    w0 <= 0.0 && w1 <= 0.0 && w2 <= 0.0
                };
                if inside {
                    let bw = [w0 / area, w1 / area, w2 / area];
                    self.set(x, y, Color::lerp3(colors[0], colors[1], colors[2], bw));
                }
            }
        }
    }

    /// Overwrite the framebuffer from tightly-packed RGBA8 bytes (ignores alpha).
    /// Bytes beyond the pixel count are ignored; missing pixels stay unchanged.
    pub fn set_rgba(&mut self, rgba: &[u8]) {
        for (i, px) in self.pixels.iter_mut().enumerate() {
            let o = i * 4;
            if o + 2 < rgba.len() {
                *px = Color::rgb(rgba[o], rgba[o + 1], rgba[o + 2]);
            }
        }
    }

    /// Tightly-packed RGBA8 bytes (row-major, top-left origin, alpha = 255).
    /// Suitable for uploading to a GPU texture for real-time presentation.
    pub fn rgba(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.pixels.len() * 4);
        for p in &self.pixels {
            out.extend_from_slice(&[p.r, p.g, p.b, 255]);
        }
        out
    }

    /// Parse a binary PPM (P6) image into a framebuffer. Returns `None` if the
    /// data isn't a well-formed P6 file. Used by the asset pipeline (`load_ppm`).
    pub fn from_ppm(bytes: &[u8]) -> Option<Framebuffer> {
        // Header: "P6", then width, height, maxval — whitespace-separated, then a
        // single whitespace byte, then width*height*3 RGB bytes.
        let mut tokens: Vec<u32> = Vec::new();
        if bytes.len() < 2 || &bytes[0..2] != b"P6" {
            return None;
        }
        let mut pos = 2; // past the "P6" magic
        while tokens.len() < 3 && pos < bytes.len() {
            // Skip whitespace and comments.
            while pos < bytes.len() && (bytes[pos] as char).is_whitespace() {
                pos += 1;
            }
            if pos < bytes.len() && bytes[pos] == b'#' {
                while pos < bytes.len() && bytes[pos] != b'\n' {
                    pos += 1;
                }
                continue;
            }
            let start = pos;
            while pos < bytes.len() && (bytes[pos] as char).is_ascii_digit() {
                pos += 1;
            }
            if pos == start {
                return None;
            }
            let n: u32 = std::str::from_utf8(&bytes[start..pos]).ok()?.parse().ok()?;
            tokens.push(n);
        }
        let (w, h) = (tokens[0], tokens[1]);
        pos += 1; // the single whitespace separator after maxval
        let need = (w * h * 3) as usize;
        if bytes.len() < pos + need {
            return None;
        }
        let mut fb = Framebuffer::new(w, h);
        for (i, px) in fb.pixels.iter_mut().enumerate() {
            let o = pos + i * 3;
            *px = Color::rgb(bytes[o], bytes[o + 1], bytes[o + 2]);
        }
        Some(fb)
    }

    /// Encode as a binary PPM (P6) image.
    pub fn to_ppm(&self) -> Vec<u8> {
        let mut out = format!("P6\n{} {}\n255\n", self.width, self.height).into_bytes();
        for p in &self.pixels {
            out.push(p.r);
            out.push(p.g);
            out.push(p.b);
        }
        out
    }
}

/// Signed area (×2) of the triangle (a, b, c) — the edge function.
fn edge(a: P2, b: P2, c: P2) -> f32 {
    (c[0] - a[0]) * (b[1] - a[1]) - (c[1] - a[1]) * (b[0] - a[0])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ppm_round_trips_through_from_ppm() {
        // The asset pipeline: a framebuffer saved to PPM and reloaded must match.
        let mut fb = Framebuffer::new(3, 2);
        fb.clear(Color::rgb(9, 9, 9));
        fb.set(1, 1, Color::rgb(200, 100, 50));
        let bytes = fb.to_ppm();
        let back = Framebuffer::from_ppm(&bytes).expect("valid PPM");
        assert_eq!(back.width(), 3);
        assert_eq!(back.height(), 2);
        assert_eq!(back.get(1, 1), Color::rgb(200, 100, 50));
        assert_eq!(back.get(0, 0), Color::rgb(9, 9, 9));
    }

    #[test]
    fn from_ppm_rejects_garbage() {
        assert!(Framebuffer::from_ppm(b"not a ppm").is_none());
        assert!(Framebuffer::from_ppm(b"P6 2 2").is_none()); // truncated body
    }

    #[test]
    fn clear_fills_every_pixel() {
        let mut fb = Framebuffer::new(4, 4);
        fb.clear(Color::rgb(10, 20, 30));
        for y in 0..4 {
            for x in 0..4 {
                assert_eq!(fb.get(x, y), Color::rgb(10, 20, 30));
            }
        }
    }

    #[test]
    fn triangle_fills_interior_not_exterior() {
        let mut fb = Framebuffer::new(20, 20);
        fb.clear(Color::BLACK);
        let red = Color::rgb(255, 0, 0);
        fb.triangle([[2.0, 2.0], [18.0, 2.0], [2.0, 18.0]], [red, red, red]);
        // A point well inside the lower-left triangle is filled.
        assert_eq!(fb.get(4, 4), red);
        // A point in the opposite (excluded) corner stays background.
        assert_eq!(fb.get(17, 17), Color::BLACK);
    }

    #[test]
    fn vertex_colors_interpolate() {
        let mut fb = Framebuffer::new(30, 30);
        fb.clear(Color::BLACK);
        fb.triangle(
            [[15.0, 1.0], [1.0, 28.0], [28.0, 28.0]],
            [Color::rgb(255, 0, 0), Color::rgb(0, 255, 0), Color::rgb(0, 0, 255)],
        );
        // Center of the triangle blends all three vertex colors (no channel is
        // 0 or 255).
        let c = fb.get(15, 19);
        assert!(c.r > 0 && c.g > 0 && c.b > 0, "expected a blended pixel, got {c:?}");
    }

    #[test]
    fn ppm_header_and_size() {
        let fb = Framebuffer::new(3, 2);
        let ppm = fb.to_ppm();
        let header = b"P6\n3 2\n255\n";
        assert!(ppm.starts_with(header));
        assert_eq!(ppm.len(), header.len() + 3 * 3 * 2); // RGB * w * h
    }

    #[test]
    fn out_of_bounds_set_is_ignored() {
        let mut fb = Framebuffer::new(4, 4);
        fb.set(-1, 0, Color::WHITE);
        fb.set(0, 100, Color::WHITE);
        fb.set(4, 4, Color::WHITE);
        // Nothing crashed and no in-bounds pixel changed.
        assert_eq!(fb.get(0, 0), Color::BLACK);
    }
}
