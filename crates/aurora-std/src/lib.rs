//! The Aurora standard library — a prelude written in Aurora, appended to every
//! program. It is plain Aurora source (so it compiles natively like user code)
//! and is added *after* the user's source, leaving user line numbers intact for
//! diagnostics and the debugger.

/// The prelude source, appended to every compiled program.
pub const PRELUDE: &str = r#"
// ---- Aurora standard library (auto-included) --------------------------------

// Float helpers.
fn lerp(a: f64, b: f64, t: f64) -> f64 { a + (b - a) * t }
fn signf(x: f64) -> f64 { if x > 0.0 { 1.0 } else { if x < 0.0 { 0.0 - 1.0 } else { 0.0 } } }
fn clampf(x: f64, lo: f64, hi: f64) -> f64 { if x < lo { lo } else { if x > hi { hi } else { x } } }
fn fract(x: f64) -> f64 { x - floor(x) }
fn deg2rad(d: f64) -> f64 { d * 0.017453292519943295 }
fn rad2deg(r: f64) -> f64 { r * 57.29577951308232 }
fn smoothstep(t: f64) -> f64 {
    let c = clampf(t, 0.0, 1.0)
    c * c * (3.0 - 2.0 * c)
}

// Integer helpers.
fn maxi(a: i64, b: i64) -> i64 { if a > b { a } else { b } }
fn mini(a: i64, b: i64) -> i64 { if a < b { a } else { b } }
fn absi(a: i64) -> i64 { if a < 0 { 0 - a } else { a } }
fn clampi(x: i64, lo: i64, hi: i64) -> i64 { maxi(lo, mini(hi, x)) }

// More integer math.
fn signi(x: i64) -> i64 { if x > 0 { 1 } else { if x < 0 { 0 - 1 } else { 0 } } }
fn max3(a: i64, b: i64, c: i64) -> i64 { maxi(a, maxi(b, c)) }
fn min3(a: i64, b: i64, c: i64) -> i64 { mini(a, mini(b, c)) }
fn gcd(a: i64, b: i64) -> i64 {
    let mut x = absi(a)
    let mut y = absi(b)
    while y != 0 {
        let t = y
        y = x - (x / y) * y
        x = t
    }
    x
}
fn lcm(a: i64, b: i64) -> i64 { if a == 0 { 0 } else { absi(a / gcd(a, b) * b) } }
fn ipow(base: i64, exp: i64) -> i64 {
    let mut r = 1
    let mut e = exp
    while e > 0 {
        r = r * base
        e = e - 1
    }
    r
}
fn factorial(n: i64) -> i64 {
    let mut r = 1
    let mut i = 2
    while i <= n {
        r = r * i
        i = i + 1
    }
    r
}
fn isqrt(n: i64) -> i64 {
    if n < 2 { return n }
    let mut x = n
    let mut y = (x + 1) / 2
    while y < x {
        x = y
        y = (x + n / x) / 2
    }
    x
}
// Wrap `x` into the half-open range [lo, hi).
fn wrapi(x: i64, lo: i64, hi: i64) -> i64 {
    let span = hi - lo
    if span <= 0 { return lo }
    let mut v = (x - lo) - ((x - lo) / span) * span
    if v < 0 { v = v + span }
    v + lo
}

// More float math.
fn mixf(a: f64, b: f64, t: f64) -> f64 { lerp(a, b, t) }
fn saturate(x: f64) -> f64 { clampf(x, 0.0, 1.0) }
fn stepf(edge: f64, x: f64) -> f64 { if x < edge { 0.0 } else { 1.0 } }
fn dist2(x0: f64, y0: f64, x1: f64, y1: f64) -> f64 {
    let dx = x1 - x0
    let dy = y1 - y0
    sqrt(dx * dx + dy * dy)
}

// String helpers (strings are first-class values).
fn str_repeat(s: str, n: i64) -> str {
    let mut r = ""
    let mut i = 0
    while i < n {
        r = r + s
        i = i + 1
    }
    r
}
fn yes_no(b: i64) -> str { if b != 0 { "yes" } else { "no" } }
fn labeled(label: str, n: i64) -> str { label + str(n) }
fn join2(a: str, b: str, sep: str) -> str { a + sep + b }

// Input key codes (match the window builtins' codes).
fn key_left() -> i64 { 0 }
fn key_right() -> i64 { 1 }
fn key_up() -> i64 { 2 }
fn key_down_arrow() -> i64 { 3 }
fn key_space() -> i64 { 4 }
fn key_w() -> i64 { 5 }
fn key_a() -> i64 { 6 }
fn key_s() -> i64 { 7 }
fn key_d() -> i64 { 8 }

// 2D vector math (the workhorse of game code).
struct Vec2 { x: f64, y: f64 }
fn vec2(x: f64, y: f64) -> Vec2 { Vec2 { x: x, y: y } }
impl Vec2 {
    fn add(self, o: Vec2) -> Vec2 { Vec2 { x: self.x + o.x, y: self.y + o.y } }
    fn sub(self, o: Vec2) -> Vec2 { Vec2 { x: self.x - o.x, y: self.y - o.y } }
    fn scale(self, s: f64) -> Vec2 { Vec2 { x: self.x * s, y: self.y * s } }
    fn dot(self, o: Vec2) -> f64 { self.x * o.x + self.y * o.y }
    fn length(self) -> f64 { sqrt(self.x * self.x + self.y * self.y) }
    fn dist(self, o: Vec2) -> f64 { dist2(self.x, self.y, o.x, o.y) }
}

// A bounded growable integer list (fixed capacity 32) — a real collection built
// from a fixed array + length, with mutation through `self`.
struct IntList { data: [i64; 32], len: i64 }
fn intlist() -> IntList { IntList { data: [0; 32], len: 0 } }
impl IntList {
    fn push(self, x: i64) {
        self.data[self.len] = x
        self.len = self.len + 1
    }
    fn get(self, i: i64) -> i64 { self.data[i] }
    fn size(self) -> i64 { self.len }
    fn sum(self) -> i64 {
        let mut s = 0
        let mut i = 0
        while i < self.len {
            s = s + self.data[i]
            i = i + 1
        }
        s
    }
    fn maxv(self) -> i64 {
        let mut m = self.data[0]
        let mut i = 1
        while i < self.len {
            m = maxi(m, self.data[i])
            i = i + 1
        }
        m
    }
}

// A bounded growable GENERIC list (fixed capacity 32) — works for any element
// type via monomorphization. Construct with `List { data: [0; 32], len: 0 }`
// (or `[0.0; 32]` for floats).
struct List<T> { data: [T; 32], len: i64 }
impl List<T> {
    fn push(self, x: T) {
        self.data[self.len] = x
        self.len = self.len + 1
    }
    fn get(self, i: i64) -> T { self.data[i] }
    fn size(self) -> i64 { self.len }
}

// A bounded growable float list — for positions, weights, timings, etc.
struct F64List { data: [f64; 32], len: i64 }
fn f64list() -> F64List { F64List { data: [0.0; 32], len: 0 } }
impl F64List {
    fn push(self, x: f64) {
        self.data[self.len] = x
        self.len = self.len + 1
    }
    fn get(self, i: i64) -> f64 { self.data[i] }
    fn size(self) -> i64 { self.len }
    fn sum(self) -> f64 {
        let mut s = 0.0
        let mut i = 0
        while i < self.len {
            s = s + self.data[i]
            i = i + 1
        }
        s
    }
    fn mean(self) -> f64 { if self.len == 0 { 0.0 } else { self.sum() / (self.len as f64) } }
}

// --- more scalar helpers ----------------------------------------------------
fn clamp01(x: f64) -> f64 { clampf(x, 0.0, 1.0) }
fn wrapf(x: f64, lo: f64, hi: f64) -> f64 {
    let r = hi - lo
    if r <= 0.0 { lo } else { lo + fmodp(x - lo, r) }
}
// Positive floating remainder (Aurora `%` is integer-only).
fn fmodp(a: f64, m: f64) -> f64 {
    let k = (a / m) as i64
    let r = a - (k as f64) * m
    if r < 0.0 { r + m } else { r }
}
// Move `cur` toward `target` by at most `step`.
fn approach(cur: f64, target: f64, step: f64) -> f64 {
    if cur < target { minf(cur + step, target) } else { maxf(cur - step, target) }
}
fn minf(a: f64, b: f64) -> f64 { if a < b { a } else { b } }
fn maxf(a: f64, b: f64) -> f64 { if a > b { a } else { b } }

// --- easing curves (t in 0..1) ----------------------------------------------
fn ease_in_quad(t: f64) -> f64 { t * t }
fn ease_out_quad(t: f64) -> f64 { t * (2.0 - t) }
fn ease_in_out_quad(t: f64) -> f64 {
    if t < 0.5 { 2.0 * t * t } else { (4.0 - 2.0 * t) * t - 1.0 }
}
fn ease_in_cubic(t: f64) -> f64 { t * t * t }
fn ease_out_cubic(t: f64) -> f64 {
    let p = t - 1.0
    p * p * p + 1.0
}
fn ease_in_out_cubic(t: f64) -> f64 {
    if t < 0.5 { 4.0 * t * t * t } else {
        let p = 2.0 * t - 2.0
        0.5 * p * p * p + 1.0
    }
}

// --- packed RGB color (0xRRGGBB), matching the framebuffer's `fb_get` ---------
fn rgb(r: i64, g: i64, b: i64) -> i64 { bor(bor(shl(band(r, 255), 16), shl(band(g, 255), 8)), band(b, 255)) }
fn red(c: i64) -> i64 { band(shr(c, 16), 255) }
fn green(c: i64) -> i64 { band(shr(c, 8), 255) }
fn blue(c: i64) -> i64 { band(c, 255) }
fn color_lerp(a: i64, b: i64, t: f64) -> i64 {
    let r = lerp(red(a) as f64, red(b) as f64, t) as i64
    let g = lerp(green(a) as f64, green(b) as f64, t) as i64
    let bl = lerp(blue(a) as f64, blue(b) as f64, t) as i64
    rgb(r, g, bl)
}

// --- 3D vector --------------------------------------------------------------
struct Vec3 { x: f64, y: f64, z: f64 }
fn vec3(x: f64, y: f64, z: f64) -> Vec3 { Vec3 { x: x, y: y, z: z } }
impl Vec3 {
    fn add(self, o: Vec3) -> Vec3 { Vec3 { x: self.x + o.x, y: self.y + o.y, z: self.z + o.z } }
    fn sub(self, o: Vec3) -> Vec3 { Vec3 { x: self.x - o.x, y: self.y - o.y, z: self.z - o.z } }
    fn scale(self, s: f64) -> Vec3 { Vec3 { x: self.x * s, y: self.y * s, z: self.z * s } }
    fn dot(self, o: Vec3) -> f64 { self.x * o.x + self.y * o.y + self.z * o.z }
    fn cross(self, o: Vec3) -> Vec3 {
        Vec3 { x: self.y * o.z - self.z * o.y, y: self.z * o.x - self.x * o.z, z: self.x * o.y - self.y * o.x }
    }
    fn length(self) -> f64 { sqrt(self.dot(self)) }
    fn normalize(self) -> Vec3 {
        let l = self.length()
        if l > 0.0 { self.scale(1.0 / l) } else { self }
    }
}

// --- collision primitives ---------------------------------------------------
struct Rect { x: f64, y: f64, w: f64, h: f64 }
fn rect(x: f64, y: f64, w: f64, h: f64) -> Rect { Rect { x: x, y: y, w: w, h: h } }
impl Rect {
    // 1 if the point is inside, else 0.
    fn contains(self, px: f64, py: f64) -> i64 {
        if px >= self.x and px < self.x + self.w and py >= self.y and py < self.y + self.h { 1 } else { 0 }
    }
    // 1 if two axis-aligned rectangles overlap, else 0.
    fn intersects(self, o: Rect) -> i64 {
        if self.x < o.x + o.w and self.x + self.w > o.x and self.y < o.y + o.h and self.y + self.h > o.y {
            1
        } else {
            0
        }
    }
}
// Circle-vs-circle overlap (1/0), using squared distance (no sqrt).
fn circles_hit(ax: f64, ay: f64, ar: f64, bx: f64, by: f64, br: f64) -> i64 {
    let dx = ax - bx
    let dy = ay - by
    let rs = ar + br
    if dx * dx + dy * dy <= rs * rs { 1 } else { 0 }
}
// Point inside circle (1/0).
fn point_in_circle(px: f64, py: f64, cx: f64, cy: f64, r: f64) -> i64 {
    let dx = px - cx
    let dy = py - cy
    if dx * dx + dy * dy <= r * r { 1 } else { 0 }
}

// --- sprite-sheet addressing ------------------------------------------------
// Source pixel offset of frame `index` in a grid sheet of `cols` columns.
struct SpriteSheet { cols: i64, tile_w: i64, tile_h: i64 }
fn sheet(cols: i64, tile_w: i64, tile_h: i64) -> SpriteSheet { SpriteSheet { cols: cols, tile_w: tile_w, tile_h: tile_h } }
impl SpriteSheet {
    fn src_x(self, index: i64) -> i64 { (index % self.cols) * self.tile_w }
    fn src_y(self, index: i64) -> i64 { (index / self.cols) * self.tile_h }
}
// Current frame of a looping animation given elapsed milliseconds.
fn anim_frame(elapsed_ms: i64, frame_ms: i64, frames: i64) -> i64 {
    if frame_ms <= 0 or frames <= 0 { 0 } else { (elapsed_ms / frame_ms) % frames }
}

// --- particles --------------------------------------------------------------
struct Particle { x: f64, y: f64, vx: f64, vy: f64, life: f64 }
fn particle(x: f64, y: f64, vx: f64, vy: f64, life: f64) -> Particle {
    Particle { x: x, y: y, vx: vx, vy: vy, life: life }
}
impl Particle {
    // Advance by `dt` seconds under gravity `g` (px/s^2); ages by `dt`.
    fn step(self, dt: f64, g: f64) {
        self.vy = self.vy + g * dt
        self.x = self.x + self.vx * dt
        self.y = self.y + self.vy * dt
        self.life = self.life - dt
    }
    fn alive(self) -> i64 { if self.life > 0.0 { 1 } else { 0 } }
}

// --- simple 2D physics helpers ----------------------------------------------
// Semi-implicit Euler step of a 1D velocity under acceleration.
fn integrate_vel(v: f64, accel: f64, dt: f64) -> f64 { v + accel * dt }
// Overlap depth of two 1D segments [a, a+aw] and [b, b+bw] (0 if disjoint).
fn overlap_1d(a: f64, aw: f64, b: f64, bw: f64) -> f64 {
    let lo = maxf(a, b)
    let hi = minf(a + aw, b + bw)
    if hi > lo { hi - lo } else { 0.0 }
}

// --- immediate-mode UI: a button hit-test -----------------------------------
struct Button { r: Rect }
fn button(x: f64, y: f64, w: f64, h: f64) -> Button { Button { r: rect(x, y, w, h) } }
impl Button {
    fn hovered(self, mx: f64, my: f64) -> i64 { self.r.contains(mx, my) }
    // 1 when the mouse is down *and* over the button.
    fn clicked(self, mx: f64, my: f64, mouse_down: i64) -> i64 {
        if mouse_down == 1 and self.r.contains(mx, my) == 1 { 1 } else { 0 }
    }
}

// --- grid / pathfinding helpers ---------------------------------------------
// Manhattan distance — the standard 4-connected grid heuristic for A*/greedy.
fn manhattan(ax: i64, ay: i64, bx: i64, by: i64) -> i64 { absi(ax - bx) + absi(ay - by) }
// Chebyshev distance — the 8-connected grid heuristic.
fn chebyshev(ax: i64, ay: i64, bx: i64, by: i64) -> i64 { maxi(absi(ax - bx), absi(ay - by)) }
// One greedy 4-connected step from (x,y) toward (tx,ty): returns packed
// direction dx*3 + dy in {-1,0,1}, prioritising the larger axis gap.
fn step_toward(x: i64, y: i64, tx: i64, ty: i64) -> i64 {
    let dx = signi(tx - x)
    let dy = signi(ty - y)
    if absi(tx - x) >= absi(ty - y) { dx * 3 + 0 } else { 0 * 3 + dy }
}

// --- grid pathfinding: 4-connected breadth-first search ---------------------
// A grid up to 32x32 (1024 cells). `cells`: 0 = walkable, non-zero = blocked.
// `compute_field(gx, gy)` fills a shortest-path distance field from the goal;
// `next_to(sx, sy)` then returns the index of the neighbouring cell one step
// closer to that goal (or -1 if unreachable) — follow it repeatedly for the
// full shortest path.
struct Grid { cells: [i64; 1024], dist: [i64; 1024], queue: [i64; 1024], w: i64, h: i64 }
fn grid(w: i64, h: i64) -> Grid {
    Grid { cells: [0; 1024], dist: [0; 1024], queue: [0; 1024], w: w, h: h }
}
impl Grid {
    fn set_wall(self, x: i64, y: i64, blocked: i64) { self.cells[y * self.w + x] = blocked }
    fn is_wall(self, x: i64, y: i64) -> i64 { self.cells[y * self.w + x] }

    // BFS from the goal; fills `dist` (steps to goal; -1 = unreachable).
    fn compute_field(self, gx: i64, gy: i64) {
        let n = self.w * self.h
        let mut i = 0
        while i < n {
            self.dist[i] = 0 - 1
            i = i + 1
        }
        let mut head = 0
        let mut tail = 0
        let g = gy * self.w + gx
        self.dist[g] = 0
        self.queue[tail] = g
        tail = tail + 1
        while head < tail {
            let cur = self.queue[head]
            head = head + 1
            let cx = cur % self.w
            let cy = cur / self.w
            let d = self.dist[cur]
            // four neighbours
            let mut k = 0
            while k < 4 {
                let mut nx = cx
                let mut ny = cy
                if k == 0 { nx = cx + 1 }
                if k == 1 { nx = cx - 1 }
                if k == 2 { ny = cy + 1 }
                if k == 3 { ny = cy - 1 }
                if nx >= 0 and ny >= 0 and nx < self.w and ny < self.h {
                    let ni = ny * self.w + nx
                    if self.cells[ni] == 0 and self.dist[ni] < 0 {
                        self.dist[ni] = d + 1
                        self.queue[tail] = ni
                        tail = tail + 1
                    }
                }
                k = k + 1
            }
        }
    }

    // After `compute_field`, the neighbour of (sx,sy) closest to the goal.
    // Returns its cell index, or -1 if (sx,sy) can't reach the goal.
    fn next_to(self, sx: i64, sy: i64) -> i64 {
        let here = self.dist[sy * self.w + sx]
        if here < 0 { 0 - 1 } else {
            let mut best = 0 - 1
            let mut bestd = here
            let mut k = 0
            while k < 4 {
                let mut nx = sx
                let mut ny = sy
                if k == 0 { nx = sx + 1 }
                if k == 1 { nx = sx - 1 }
                if k == 2 { ny = sy + 1 }
                if k == 3 { ny = sy - 1 }
                if nx >= 0 and ny >= 0 and nx < self.w and ny < self.h {
                    let ni = ny * self.w + nx
                    let nd = self.dist[ni]
                    if nd >= 0 and nd < bestd {
                        bestd = nd
                        best = ni
                    }
                }
                k = k + 1
            }
            best
        }
    }
}

// --- 2D AABB physics body ---------------------------------------------------
// Velocity integration under gravity + discrete collision resolution against
// static rectangles (minimum-translation push-out, zeroing the resolved axis) —
// enough for platformer/top-down movement.
struct Body { x: f64, y: f64, w: f64, h: f64, vx: f64, vy: f64 }
fn body(x: f64, y: f64, w: f64, h: f64) -> Body { Body { x: x, y: y, w: w, h: h, vx: 0.0, vy: 0.0 } }
impl Body {
    fn step(self, dt: f64, gravity: f64) {
        self.vy = self.vy + gravity * dt
        self.x = self.x + self.vx * dt
        self.y = self.y + self.vy * dt
    }
    fn bounds(self) -> Rect { rect(self.x, self.y, self.w, self.h) }
    // Resolve overlap with a static rect along the least-penetration axis.
    // Returns 1 if a collision was resolved, else 0.
    fn collide(self, wx: f64, wy: f64, ww: f64, wh: f64) -> i64 {
        let ox = overlap_1d(self.x, self.w, wx, ww)
        let oy = overlap_1d(self.y, self.h, wy, wh)
        if ox <= 0.0 or oy <= 0.0 { 0 } else {
            if ox < oy {
                if self.x + self.w * 0.5 < wx + ww * 0.5 { self.x = self.x - ox } else { self.x = self.x + ox }
                self.vx = 0.0
            } else {
                if self.y + self.h * 0.5 < wy + wh * 0.5 { self.y = self.y - oy } else { self.y = self.y + oy }
                self.vy = 0.0
            }
            1
        }
    }
}

// --- immediate-mode UI widgets ----------------------------------------------
// Draw themselves into the framebuffer each frame and report interaction using
// the live mouse builtins. `draw_text` is a no-op until a font is loaded, so
// widgets still render as shapes without one.
fn fill_rect(x: i64, y: i64, w: i64, h: i64, r: i64, g: i64, b: i64) {
    triangle(x, y, x + w, y, x, y + h, r, g, b)
    triangle(x + w, y, x + w, y + h, x, y + h, r, g, b)
}
fn ui_label(x: i64, y: i64, text: str, px: i64) { draw_text(x, y, text, px, rgb(230, 230, 235)) }
// Returns 1 when the button is hovered and the mouse is down.
fn ui_button(x: i64, y: i64, w: i64, h: i64, label: str) -> i64 {
    let mx = mouse_x()
    let my = mouse_y()
    let over = if mx >= x and mx < x + w and my >= y and my < y + h { 1 } else { 0 }
    if over == 1 { fill_rect(x, y, w, h, 90, 90, 130) } else { fill_rect(x, y, w, h, 55, 55, 70) }
    draw_text(x + 6, y + 4, label, h - 8, rgb(235, 235, 245))
    if over == 1 and mouse_down() == 1 { 1 } else { 0 }
}
// Horizontal slider; returns the (possibly updated) value in 0..1.
fn ui_slider(x: i64, y: i64, w: i64, value: f64) -> f64 {
    fill_rect(x, y + 6, w, 4, 80, 80, 95)
    let kx = x + (value * (w as f64)) as i64
    fill_rect(kx - 4, y, 8, 16, 200, 200, 215)
    let mx = mouse_x()
    let my = mouse_y()
    if mouse_down() == 1 and mx >= x and mx < x + w and my >= y - 4 and my < y + 20 {
        clamp01((mx - x) as f64 / (w as f64))
    } else {
        value
    }
}
"#;

/// Append the standard-library prelude to a user program's source. The prelude
/// goes last so the user's line numbers (and thus diagnostics/breakpoints) are
/// unchanged.
pub fn with_std(user_src: &str) -> String {
    format!("{user_src}\n{PRELUDE}")
}

#[cfg(test)]
mod tests {
    #[test]
    fn prelude_parses_and_checks_clean() {
        // The standard library must itself parse and pass all static checks.
        let (module, pdiags) = aurora_parser::parse_str(super::PRELUDE);
        assert!(!pdiags.iter().any(|d| d.is_error()), "prelude parse errors: {pdiags:?}");
        let mut diags = aurora_check::check(&module);
        diags.extend(aurora_typeck::check_types(&module));
        let errs: Vec<_> = diags.iter().filter(|d| d.is_error()).map(|d| &d.message).collect();
        assert!(errs.is_empty(), "prelude check errors: {errs:?}");
    }
}
