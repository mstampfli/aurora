//! Aurora's native runtime — the host functions compiled Aurora code calls.
//!
//! Every `aurora_*` symbol here is `#[no_mangle] pub extern "C"`, so it is a
//! real, linkable C-ABI symbol. Two consumers use them:
//!
//! * the **JIT** (`aurora-codegen`) registers their addresses as symbols, and
//! * **AOT executables** resolve the undefined `aurora_*` references in the
//!   emitted object file against this crate at link time.
//!
//! State (framebuffer, ECS world) is thread-local, matching the single-threaded
//! `main` the compiled program runs on.

use std::cell::RefCell;
use std::collections::HashSet;

// 3D physics (Rapier 3D) and 3D pathfinding (voxel grid + navmesh) builtins.
mod nav3d;
mod phys3d;
pub use nav3d::*;
pub use phys3d::*;

// Game-ready multiplayer: authoritative server, client prediction, interpolation.
mod netgame;
pub use netgame::*;

// --- printing --------------------------------------------------------------

#[no_mangle]
pub extern "C" fn aurora_print_i64(n: i64) {
    print!("{n}");
}
/// Format an `f64` for display. Whole-valued finite floats get a trailing `.0`
/// (`7.0` not `7`) so floats are visually distinct from ints — Aurora is a
/// float-heavy game-dev language and the ambiguity is a debugging hazard.
/// Non-finite values (`inf`, `NaN`) and already-fractional values are left as
/// Rust's default Display renders them.
fn fmt_f64(x: f64) -> String {
    if x.is_finite() && x == x.trunc() {
        format!("{x}.0")
    } else {
        format!("{x}")
    }
}

#[no_mangle]
pub extern "C" fn aurora_print_f64(x: f64) {
    print!("{}", fmt_f64(x));
}
#[no_mangle]
pub extern "C" fn aurora_print_str(ptr: *const u8, len: i64) {
    let s = unsafe { std::slice::from_raw_parts(ptr, len.max(0) as usize) };
    print!("{}", String::from_utf8_lossy(s));
}
#[no_mangle]
pub extern "C" fn aurora_print_nl() {
    println!();
}

/// Flush buffered stdout — called from the AOT entry shim before exit, since the
/// program does not return through Rust's runtime (which would flush for us).
#[no_mangle]
pub extern "C" fn aurora_runtime_flush() {
    use std::io::Write;
    let _ = std::io::stdout().flush();
}

/// Graceful-shutdown hook: leak the window + GPU/audio state so it is NOT torn down in a
/// thread-local destructor at process exit (wgpu/winit panic if it is). Called by the AOT
/// entry shim right before `process::exit`.
#[no_mangle]
pub extern "C" fn aurora_runtime_shutdown() {
    aurora_window::imm_leak();
    aurora_audio::leak_audio();
}

thread_local! {
    static LAST_FRAME: RefCell<Option<std::time::Instant>> = const { RefCell::new(None) };
}

/// Real elapsed seconds since the previous call (0.016 on the first call),
/// clamped to 0.1 so a stall can't make the game lurch or spiral. Lets the game
/// loop run frame-rate-independent instead of assuming a fixed step.
#[no_mangle]
pub extern "C" fn aurora_frame_dt() -> f64 {
    LAST_FRAME.with(|c| {
        let now = std::time::Instant::now();
        let dt = match c.borrow_mut().replace(now) {
            Some(prev) => now.duration_since(prev).as_secs_f64(),
            None => 1.0 / 60.0,
        };
        dt.clamp(0.0001, 0.1)
    })
}

/// Sleep the calling thread for `ms` milliseconds. For pacing a loop that has no
/// other frame limiter (a headless server tick, or a non-windowed test).
#[no_mangle]
pub extern "C" fn aurora_sleep_ms(ms: i64) {
    if ms > 0 {
        std::thread::sleep(std::time::Duration::from_millis(ms as u64));
    }
}

/// FFI demonstration target (a Rust `extern "C"` function): dot product of two
/// `n`-element `f64` buffers. Aurora arrays/structs of `f64` are contiguous
/// 8-byte slots, so they pass straight through as `const double*` — this is what
/// lets `@extern` bind real C/Rust functions that take buffers and vectors.
#[no_mangle]
pub extern "C" fn aurora_ffi_dot(a: *const f64, b: *const f64, n: i64) -> f64 {
    let n = n.max(0) as usize;
    let (a, b) = unsafe { (std::slice::from_raw_parts(a, n), std::slice::from_raw_parts(b, n)) };
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// `f32` variant — reads two C-packed `float` buffers. Tests that Aurora `f32`
/// aggregates are marshaled to C's 4-byte-packed layout over FFI.
#[no_mangle]
pub extern "C" fn aurora_ffi_dotf(a: *const f32, b: *const f32, n: i64) -> f32 {
    let n = n.max(0) as usize;
    let (a, b) = unsafe { (std::slice::from_raw_parts(a, n), std::slice::from_raw_parts(b, n)) };
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Report an out-of-bounds array access with a clear message and abort. Called
/// by bounds-check code in place of a raw trap, so the failure reads as a panic
/// rather than a cryptic "illegal instruction".
#[no_mangle]
pub extern "C" fn aurora_oob(idx: i64, len: i64) {
    use std::io::Write;
    let _ = std::io::stdout().flush();
    eprintln!("panic: array index {idx} out of bounds (length {len})");
    std::process::exit(101);
}

/// Clean panic for integer division/remainder by zero, in place of a raw CPU
/// trap (SIGFPE / "illegal instruction"), matching the interpreter's behavior.
#[no_mangle]
pub extern "C" fn aurora_divzero() {
    use std::io::Write;
    let _ = std::io::stdout().flush();
    eprintln!("panic: integer division or remainder by zero");
    std::process::exit(101);
}

/// IEEE float remainder (`%` on floats), via libm fmod.
#[no_mangle]
pub extern "C" fn aurora_fmod(x: f64, y: f64) -> f64 {
    x % y
}

// --- graphics: a thread-local CPU framebuffer ------------------------------

thread_local! {
    static FB: RefCell<Option<aurora_gfx::Framebuffer>> = const { RefCell::new(None) };
}

#[no_mangle]
pub extern "C" fn aurora_framebuffer(w: i64, h: i64) {
    FB.with(|fb| {
        *fb.borrow_mut() = Some(aurora_gfx::Framebuffer::new(w.max(0) as u32, h.max(0) as u32))
    });
}
fn color(r: i64, g: i64, b: i64) -> aurora_gfx::Color {
    let c = |v: i64| v.clamp(0, 255) as u8;
    aurora_gfx::Color::rgb(c(r), c(g), c(b))
}
#[no_mangle]
pub extern "C" fn aurora_clear(r: i64, g: i64, b: i64) {
    FB.with(|fb| {
        if let Some(f) = fb.borrow_mut().as_mut() {
            f.clear(color(r, g, b));
        }
    });
}
#[no_mangle]
pub extern "C" fn aurora_pixel(x: i64, y: i64, r: i64, g: i64, b: i64) {
    FB.with(|fb| {
        if let Some(f) = fb.borrow_mut().as_mut() {
            f.set(x as i32, y as i32, color(r, g, b));
        }
    });
}
#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub extern "C" fn aurora_triangle(
    x0: i64, y0: i64, x1: i64, y1: i64, x2: i64, y2: i64, r: i64, g: i64, b: i64,
) {
    FB.with(|fb| {
        if let Some(f) = fb.borrow_mut().as_mut() {
            let c = color(r, g, b);
            f.triangle(
                [[x0 as f32, y0 as f32], [x1 as f32, y1 as f32], [x2 as f32, y2 as f32]],
                [c, c, c],
            );
        }
    });
}
#[no_mangle]
pub extern "C" fn aurora_fb_get(x: i64, y: i64) -> i64 {
    FB.with(|fb| match fb.borrow().as_ref() {
        Some(f) if (x as u32) < f.width() && (y as u32) < f.height() => {
            let c = f.get(x as u32, y as u32);
            ((c.r as i64) << 16) | ((c.g as i64) << 8) | c.b as i64
        }
        _ => 0,
    })
}
#[no_mangle]
pub extern "C" fn aurora_save_ppm(ptr: *const u8, len: i64) {
    let path = {
        let s = unsafe { std::slice::from_raw_parts(ptr, len.max(0) as usize) };
        String::from_utf8_lossy(s).into_owned()
    };
    FB.with(|fb| {
        if let Some(f) = fb.borrow().as_ref() {
            let _ = std::fs::write(&path, f.to_ppm());
        }
    });
}

// --- region arenas ----------------------------------------------------------
//
// A real runtime backing for the language's `#frame`/`#level`/`#perm` regions:
// each is a chunked bump allocator. Dynamic allocations (string concat, int/
// float formatting) come from the `#frame` arena, and `frame_reset()` frees the
// whole frame's allocations at once (O(1)) — so memory is arena-managed and
// reclaimed at frame boundaries instead of leaking. The region *checker*
// (`aurora-check` §8.2) statically prevents storing shorter-lived (frame) data
// where longer-lived data is expected, which is what makes the bulk reset safe.

const CHUNK: usize = 1 << 20; // 1 MiB per chunk

struct Arena {
    chunks: Vec<Vec<u8>>,
    cur: usize,
    used: usize,
}
impl Arena {
    fn new() -> Arena {
        Arena { chunks: vec![vec![0u8; CHUNK]], cur: 0, used: 0 }
    }
    /// Bump-allocate `n` 8-aligned bytes; returns a stable pointer (chunks never
    /// move once allocated). Oversized requests get their own chunk.
    fn alloc(&mut self, n: usize) -> *mut u8 {
        let n = (n + 7) & !7;
        if n > CHUNK {
            let mut c = vec![0u8; n];
            let p = c.as_mut_ptr();
            // Park oversized chunks before the active one so `cur` stays valid.
            self.chunks.insert(self.cur, c);
            self.cur += 1;
            return p;
        }
        if self.used + n > self.chunks[self.cur].len() {
            self.cur += 1;
            if self.cur >= self.chunks.len() {
                self.chunks.push(vec![0u8; CHUNK]);
            }
            self.used = 0;
        }
        let p = unsafe { self.chunks[self.cur].as_mut_ptr().add(self.used) };
        self.used += n;
        p
    }
    /// Free everything (reuse the first chunk; retain capacity for next frame).
    fn reset(&mut self) {
        self.chunks.truncate(1);
        self.cur = 0;
        self.used = 0;
    }
}

thread_local! {
    static FRAME_ARENA: RefCell<Arena> = RefCell::new(Arena::new());
}

fn frame_alloc(bytes: &[u8]) -> *mut u8 {
    FRAME_ARENA.with(|a| {
        let mut a = a.borrow_mut();
        let p = a.alloc(bytes.len().max(1));
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), p, bytes.len()) };
        p
    })
}

/// Free all `#frame` allocations made since the last reset. Call once per frame.
#[no_mangle]
pub extern "C" fn aurora_frame_reset() {
    FRAME_ARENA.with(|a| a.borrow_mut().reset());
}

/// Bytes currently allocated in the frame arena (for tests/introspection).
pub fn frame_arena_used() -> usize {
    FRAME_ARENA.with(|a| {
        let a = a.borrow();
        a.cur * CHUNK + a.used
    })
}

// --- first-class strings ---------------------------------------------------
//
// A string value is a `[data_ptr, len]` pair. These host functions build new
// strings (concat, int/float formatting) from the `#frame` arena and write the
// resulting `[ptr, len]` into a caller-provided 2-slot aggregate `out`.

/// Write a `[ptr, len]` pair (allocated in the frame arena) into `out`.
unsafe fn write_str(out: *mut i64, bytes: Vec<u8>) {
    let ptr = frame_alloc(&bytes) as i64;
    *out = ptr;
    *out.add(1) = bytes.len() as i64;
}

#[no_mangle]
pub extern "C" fn aurora_str_concat(
    out: *mut i64, ap: *const u8, al: i64, bp: *const u8, bl: i64,
) {
    let a = unsafe { std::slice::from_raw_parts(ap, al.max(0) as usize) };
    let b = unsafe { std::slice::from_raw_parts(bp, bl.max(0) as usize) };
    let mut v = Vec::with_capacity(a.len() + b.len());
    v.extend_from_slice(a);
    v.extend_from_slice(b);
    unsafe { write_str(out, v) };
}

#[no_mangle]
pub extern "C" fn aurora_str_eq(ap: *const u8, al: i64, bp: *const u8, bl: i64) -> i64 {
    let a = unsafe { std::slice::from_raw_parts(ap, al.max(0) as usize) };
    let b = unsafe { std::slice::from_raw_parts(bp, bl.max(0) as usize) };
    (a == b) as i64
}

#[no_mangle]
pub extern "C" fn aurora_int_to_str(out: *mut i64, n: i64) {
    unsafe { write_str(out, n.to_string().into_bytes()) };
}

/// Byte at index `i` of the string (0..len), or -1 if out of range.
#[no_mangle]
pub extern "C" fn aurora_str_char_at(ptr: *const u8, len: i64, i: i64) -> i64 {
    if i < 0 || i >= len {
        return -1;
    }
    let s = unsafe { std::slice::from_raw_parts(ptr, len.max(0) as usize) };
    s[i as usize] as i64
}

/// Substring `[start, start+n)` (clamped) written into `out` as a new string.
#[no_mangle]
pub extern "C" fn aurora_str_substr(out: *mut i64, ptr: *const u8, len: i64, start: i64, n: i64) {
    let s = unsafe { std::slice::from_raw_parts(ptr, len.max(0) as usize) };
    let start = start.clamp(0, len) as usize;
    let end = (start + n.max(0) as usize).min(len.max(0) as usize);
    unsafe { write_str(out, s[start..end].to_vec()) };
}

/// 1 if `hay` starts with `needle`, else 0.
#[no_mangle]
pub extern "C" fn aurora_str_starts_with(
    hp: *const u8, hl: i64, np: *const u8, nl: i64,
) -> i64 {
    let hay = unsafe { std::slice::from_raw_parts(hp, hl.max(0) as usize) };
    let needle = unsafe { std::slice::from_raw_parts(np, nl.max(0) as usize) };
    hay.starts_with(needle) as i64
}

#[no_mangle]
pub extern "C" fn aurora_float_to_str(out: *mut i64, x: f64) {
    unsafe { write_str(out, fmt_f64(x).into_bytes()) };
}

// --- asset pipeline --------------------------------------------------------

/// Load a binary PPM image at `path` into the framebuffer (resizing it).
/// Returns 1 on success, 0 on failure. Backs the `load_ppm` builtin.
#[no_mangle]
pub extern "C" fn aurora_load_ppm(ptr: *const u8, len: i64) -> i64 {
    let path = {
        let s = unsafe { std::slice::from_raw_parts(ptr, len.max(0) as usize) };
        String::from_utf8_lossy(s).into_owned()
    };
    match std::fs::read(&path).ok().and_then(|b| aurora_gfx::Framebuffer::from_ppm(&b)) {
        Some(fb) => {
            FB.with(|f| *f.borrow_mut() = Some(fb));
            1
        }
        None => 0,
    }
}

/// Load a PNG/JPEG image at `path` into the framebuffer (resizing it to the
/// image), decoded to RGBA via the `image` crate. Returns 1 on success, 0 on
/// failure. Backs the `load_image` builtin — the asset pipeline beyond PPM.
#[no_mangle]
pub extern "C" fn aurora_load_image(ptr: *const u8, len: i64) -> i64 {
    let path = {
        let s = unsafe { std::slice::from_raw_parts(ptr, len.max(0) as usize) };
        String::from_utf8_lossy(s).into_owned()
    };
    match image::open(&path) {
        Ok(img) => {
            let rgba = img.to_rgba8();
            let (w, h) = rgba.dimensions();
            let mut fb = aurora_gfx::Framebuffer::new(w, h);
            fb.set_rgba(rgba.as_raw());
            FB.with(|f| *f.borrow_mut() = Some(fb));
            1
        }
        Err(_) => 0,
    }
}

// --- text rendering (TrueType via fontdue) ----------------------------------

thread_local! {
    static FONT: RefCell<Option<fontdue::Font>> = const { RefCell::new(None) };
}

/// Load a TrueType/OpenType font from `path` for `draw_text`. Returns 1/0.
#[no_mangle]
pub extern "C" fn aurora_load_font(ptr: *const u8, len: i64) -> i64 {
    let path = {
        let s = unsafe { std::slice::from_raw_parts(ptr, len.max(0) as usize) };
        String::from_utf8_lossy(s).into_owned()
    };
    let Ok(bytes) = std::fs::read(&path) else { return 0 };
    match fontdue::Font::from_bytes(bytes, fontdue::FontSettings::default()) {
        Ok(f) => {
            FONT.with(|x| *x.borrow_mut() = Some(f));
            1
        }
        Err(_) => 0,
    }
}

/// Draw `text` into the framebuffer with its top-left at (x, y), at `px` pixel
/// height, in packed `color` (0xRRGGBB), alpha-blending each glyph's coverage
/// over the existing pixels. No-op if no font was loaded or no framebuffer is
/// active. Backs the `draw_text` builtin.
#[no_mangle]
fn render_text(x: i64, y: i64, text: &str, px: i64, color: i64) {
    let px = px.max(1) as f32;
    let (cr, cg, cb) = (((color >> 16) & 255) as u8, ((color >> 8) & 255) as u8, (color & 255) as u8);
    FONT.with(|font| {
        let font = font.borrow();
        let Some(font) = font.as_ref() else { return };
        FB.with(|fb| {
            let mut fb = fb.borrow_mut();
            let Some(fb) = fb.as_mut() else { return };
            let (w, h) = (fb.width() as i32, fb.height() as i32);
            let baseline = y + px as i64; // `y` is the top; baseline ≈ y + size
            let mut pen = x;
            for ch in text.chars() {
                let (m, bitmap) = font.rasterize(ch, px);
                let gx = pen + m.xmin as i64;
                let gy = baseline - m.height as i64 - m.ymin as i64;
                for row in 0..m.height {
                    for col in 0..m.width {
                        let cov = bitmap[row * m.width + col] as u32;
                        if cov == 0 {
                            continue;
                        }
                        let (sx, sy) = ((gx + col as i64) as i32, (gy + row as i64) as i32);
                        if sx < 0 || sy < 0 || sx >= w || sy >= h {
                            continue;
                        }
                        let bg = fb.get(sx as u32, sy as u32);
                        let blend = |b: u8, f: u8| ((b as u32 * (255 - cov) + f as u32 * cov) / 255) as u8;
                        let out = aurora_gfx::Color::rgb(blend(bg.r, cr), blend(bg.g, cg), blend(bg.b, cb));
                        fb.set(sx, sy, out);
                    }
                }
                pen += m.advance_width as i64;
            }
        });
    });
}

#[no_mangle]
pub extern "C" fn aurora_draw_text(x: i64, y: i64, ptr: *const u8, len: i64, px: i64, color: i64) {
    let text = {
        let s = unsafe { std::slice::from_raw_parts(ptr, len.max(0) as usize) };
        String::from_utf8_lossy(s).into_owned()
    };
    render_text(x, y, &text, px, color);
}

/// Pixel width of `text` at size `px` in the loaded font (sum of glyph advances).
/// Lets a game centre/right-align labels. Returns 0 if no font is loaded.
#[no_mangle]
pub extern "C" fn aurora_text_width(ptr: *const u8, len: i64, px: i64) -> i64 {
    let text = {
        let s = unsafe { std::slice::from_raw_parts(ptr, len.max(0) as usize) };
        String::from_utf8_lossy(s).into_owned()
    };
    let px = px.max(1) as f32;
    FONT.with(|font| {
        let font = font.borrow();
        let Some(font) = font.as_ref() else { return 0 };
        let mut w = 0i64;
        for ch in text.chars() {
            w += font.metrics(ch, px).advance_width as i64;
        }
        w
    })
}

/// Draw an integer as text (formats it in Rust, renders like `draw_text`). Lets a
/// game show dynamic numbers (scores, timers) without string formatting in Aurora.
#[no_mangle]
pub extern "C" fn aurora_draw_int(x: i64, y: i64, n: i64, px: i64, color: i64) {
    render_text(x, y, &n.to_string(), px, color);
}

// --- real 2D physics (Rapier) -----------------------------------------------
//
// A stateful physics world backed by Rapier: rigid bodies, colliders, gravity,
// continuous collision — far beyond the hand-rolled AABB resolver in the stdlib.
// Bodies are referenced by an i64 handle (an index into `handles`). Positions
// are the body centre, in whatever units the program uses (e.g. pixels).

struct Phys {
    gravity: rapier2d::prelude::Vector<rapier2d::prelude::Real>,
    params: rapier2d::prelude::IntegrationParameters,
    pipeline: rapier2d::prelude::PhysicsPipeline,
    islands: rapier2d::prelude::IslandManager,
    broad: rapier2d::prelude::DefaultBroadPhase,
    narrow: rapier2d::prelude::NarrowPhase,
    bodies: rapier2d::prelude::RigidBodySet,
    colliders: rapier2d::prelude::ColliderSet,
    impulse: rapier2d::prelude::ImpulseJointSet,
    multibody: rapier2d::prelude::MultibodyJointSet,
    ccd: rapier2d::prelude::CCDSolver,
    query: rapier2d::prelude::QueryPipeline,
    handles: Vec<rapier2d::prelude::RigidBodyHandle>,
}
thread_local! {
    static PHYS: RefCell<Option<Phys>> = const { RefCell::new(None) };
}

/// Create (or reset) the physics world with gravity (gx, gy).
#[no_mangle]
pub extern "C" fn aurora_phys_init(gx: f64, gy: f64) {
    use rapier2d::prelude::*;
    let p = Phys {
        gravity: vector![gx as Real, gy as Real],
        params: IntegrationParameters::default(),
        pipeline: PhysicsPipeline::new(),
        islands: IslandManager::new(),
        broad: DefaultBroadPhase::new(),
        narrow: NarrowPhase::new(),
        bodies: RigidBodySet::new(),
        colliders: ColliderSet::new(),
        impulse: ImpulseJointSet::new(),
        multibody: MultibodyJointSet::new(),
        ccd: CCDSolver::new(),
        query: QueryPipeline::new(),
        handles: Vec::new(),
    };
    PHYS.with(|x| *x.borrow_mut() = Some(p));
}

/// Add a box body (half-extents hw,hh) at centre (x,y); `dynamic` 1=moving,
/// 0=static. Returns its handle, or -1 if no world.
#[no_mangle]
pub extern "C" fn aurora_phys_add(x: f64, y: f64, hw: f64, hh: f64, dynamic: i64) -> i64 {
    use rapier2d::prelude::*;
    PHYS.with(|p| {
        let mut p = p.borrow_mut();
        let Some(p) = p.as_mut() else { return -1 };
        let rb = if dynamic != 0 {
            RigidBodyBuilder::dynamic().translation(vector![x as Real, y as Real]).build()
        } else {
            RigidBodyBuilder::fixed().translation(vector![x as Real, y as Real]).build()
        };
        let h = p.bodies.insert(rb);
        let col = ColliderBuilder::cuboid(hw as Real, hh as Real).build();
        p.colliders.insert_with_parent(col, h, &mut p.bodies);
        p.handles.push(h);
        (p.handles.len() - 1) as i64
    })
}

/// Advance the simulation by `dt` seconds.
#[no_mangle]
pub extern "C" fn aurora_phys_step(dt: f64) {
    use rapier2d::prelude::Real;
    PHYS.with(|p| {
        let mut p = p.borrow_mut();
        let Some(p) = p.as_mut() else { return };
        p.params.dt = dt as Real;
        let g = p.gravity;
        p.pipeline.step(
            &g, &p.params, &mut p.islands, &mut p.broad, &mut p.narrow,
            &mut p.bodies, &mut p.colliders, &mut p.impulse, &mut p.multibody,
            &mut p.ccd, Some(&mut p.query), &(), &(),
        );
    });
}

fn phys_pos(h: i64, axis: usize) -> f64 {
    PHYS.with(|p| {
        let p = p.borrow();
        let Some(p) = p.as_ref() else { return 0.0 };
        match p.handles.get(h.max(0) as usize).and_then(|&hd| p.bodies.get(hd)) {
            Some(b) => b.translation()[axis] as f64,
            None => 0.0,
        }
    })
}
#[no_mangle]
pub extern "C" fn aurora_phys_x(h: i64) -> f64 { phys_pos(h, 0) }
#[no_mangle]
pub extern "C" fn aurora_phys_y(h: i64) -> f64 { phys_pos(h, 1) }

/// Set a body's linear velocity.
#[no_mangle]
pub extern "C" fn aurora_phys_set_vel(h: i64, vx: f64, vy: f64) {
    use rapier2d::prelude::*;
    PHYS.with(|p| {
        let mut p = p.borrow_mut();
        let Some(p) = p.as_mut() else { return };
        if let Some(b) = p.handles.get(h.max(0) as usize).copied().and_then(|hd| p.bodies.get_mut(hd)) {
            b.set_linvel(vector![vx as Real, vy as Real], true);
        }
    });
}

fn phys_vel(h: i64, axis: usize) -> f64 {
    PHYS.with(|p| {
        let p = p.borrow();
        match p.as_ref().and_then(|p| p.handles.get(h.max(0) as usize).and_then(|&hd| p.bodies.get(hd))) {
            Some(b) => b.linvel()[axis] as f64,
            None => 0.0,
        }
    })
}
#[no_mangle]
pub extern "C" fn aurora_phys_vel_x(h: i64) -> f64 { phys_vel(h, 0) }
#[no_mangle]
pub extern "C" fn aurora_phys_vel_y(h: i64) -> f64 { phys_vel(h, 1) }

/// Apply an instantaneous impulse (e.g. a jump or knockback) to a body.
#[no_mangle]
pub extern "C" fn aurora_phys_apply_impulse(h: i64, ix: f64, iy: f64) {
    use rapier2d::prelude::*;
    PHYS.with(|p| {
        let mut p = p.borrow_mut();
        let Some(p) = p.as_mut() else { return };
        if let Some(b) = p.handles.get(h.max(0) as usize).copied().and_then(|hd| p.bodies.get_mut(hd)) {
            b.apply_impulse(vector![ix as Real, iy as Real], true);
        }
    });
}

/// Apply a continuous force (cleared each step) to a body.
#[no_mangle]
pub extern "C" fn aurora_phys_apply_force(h: i64, fx: f64, fy: f64) {
    use rapier2d::prelude::*;
    PHYS.with(|p| {
        let mut p = p.borrow_mut();
        let Some(p) = p.as_mut() else { return };
        if let Some(b) = p.handles.get(h.max(0) as usize).copied().and_then(|hd| p.bodies.get_mut(hd)) {
            b.add_force(vector![fx as Real, fy as Real], true);
        }
    });
}

/// Teleport a body to (x, y).
#[no_mangle]
pub extern "C" fn aurora_phys_set_pos(h: i64, x: f64, y: f64) {
    use rapier2d::prelude::*;
    PHYS.with(|p| {
        let mut p = p.borrow_mut();
        let Some(p) = p.as_mut() else { return };
        if let Some(b) = p.handles.get(h.max(0) as usize).copied().and_then(|hd| p.bodies.get_mut(hd)) {
            b.set_translation(vector![x as Real, y as Real], true);
        }
    });
}

/// Cast a ray from (x,y) along (dx,dy) up to `max` distance; returns the
/// distance to the first collider hit, or -1 if nothing is hit. Useful for
/// line-of-sight and ground checks. (Run after `phys_step`.)
#[no_mangle]
pub extern "C" fn aurora_phys_raycast(x: f64, y: f64, dx: f64, dy: f64, max: f64) -> f64 {
    use rapier2d::prelude::*;
    PHYS.with(|p| {
        let p = p.borrow();
        let Some(p) = p.as_ref() else { return -1.0 };
        let ray = Ray::new(point![x as Real, y as Real], vector![dx as Real, dy as Real]);
        match p.query.cast_ray(&p.bodies, &p.colliders, &ray, max as Real, true, QueryFilter::default()) {
            Some((_, toi)) => toi as f64,
            None => -1.0,
        }
    })
}

// --- weighted A* pathfinding (the `pathfinding` crate) ----------------------
//
// A 4-connected grid with per-cell walls; `nav_find` runs A* and stores the
// resulting shortest path, read back cell by cell. Real A*, not the stdlib BFS.

struct Nav {
    w: i32,
    h: i32,
    walls: Vec<bool>,
    path: Vec<(i32, i32)>,
}
thread_local! {
    static NAV: RefCell<Option<Nav>> = const { RefCell::new(None) };
}

#[no_mangle]
pub extern "C" fn aurora_nav_init(w: i64, h: i64) {
    let (w, h) = (w.max(0) as i32, h.max(0) as i32);
    let n = Nav { w, h, walls: vec![false; (w * h).max(0) as usize], path: Vec::new() };
    NAV.with(|x| *x.borrow_mut() = Some(n));
}
#[no_mangle]
pub extern "C" fn aurora_nav_wall(x: i64, y: i64, blocked: i64) {
    NAV.with(|n| {
        let mut n = n.borrow_mut();
        let Some(n) = n.as_mut() else { return };
        if x >= 0 && y >= 0 && (x as i32) < n.w && (y as i32) < n.h {
            let idx = (y as i32 * n.w + x as i32) as usize;
            n.walls[idx] = blocked != 0;
        }
    });
}
/// Run A* from (sx,sy) to (gx,gy); returns the path length (cells), or -1.
#[no_mangle]
pub extern "C" fn aurora_nav_find(sx: i64, sy: i64, gx: i64, gy: i64) -> i64 {
    NAV.with(|n| {
        let mut n = n.borrow_mut();
        let Some(n) = n.as_mut() else { return -1 };
        let (w, h) = (n.w, n.h);
        let walls = n.walls.clone();
        let goal = (gx as i32, gy as i32);
        let result = pathfinding::prelude::astar(
            &(sx as i32, sy as i32),
            |&(x, y)| {
                let mut v: Vec<((i32, i32), i32)> = Vec::new();
                for (dx, dy) in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
                    let (nx, ny) = (x + dx, y + dy);
                    if nx >= 0 && ny >= 0 && nx < w && ny < h && !walls[(ny * w + nx) as usize] {
                        v.push(((nx, ny), 1));
                    }
                }
                v
            },
            |&(x, y)| (x - goal.0).abs() + (y - goal.1).abs(),
            |&p| p == goal,
        );
        match result {
            Some((path, _)) => {
                let len = path.len() as i64;
                n.path = path;
                len
            }
            None => {
                n.path.clear();
                -1
            }
        }
    })
}
#[no_mangle]
pub extern "C" fn aurora_nav_x(i: i64) -> i64 {
    NAV.with(|n| n.borrow().as_ref().and_then(|n| n.path.get(i.max(0) as usize)).map(|&(x, _)| x as i64).unwrap_or(-1))
}
#[no_mangle]
pub extern "C" fn aurora_nav_y(i: i64) -> i64 {
    NAV.with(|n| n.borrow().as_ref().and_then(|n| n.path.get(i.max(0) as usize)).map(|&(_, y)| y as i64).unwrap_or(-1))
}

// --- networking (reliable UDP) as a language feature ------------------------
//
// Backs Aurora's `net_bind`/`net_connect`/`net_send`/`net_recv` builtins with the
// reliable-ordered transport from `aurora-net`. Messages are strings.

thread_local! {
    static NET: RefCell<Option<aurora_net::UdpEndpoint>> = const { RefCell::new(None) };
    static NET_INBOX: RefCell<std::collections::VecDeque<Vec<u8>>> =
        const { RefCell::new(std::collections::VecDeque::new()) };
}

/// Bind a UDP endpoint to `127.0.0.1:port`. Returns 1 on success, 0 on failure.
#[no_mangle]
pub extern "C" fn aurora_net_bind(port: i64) -> i64 {
    match aurora_net::UdpEndpoint::bind(("127.0.0.1", port.clamp(0, 65535) as u16)) {
        Ok(ep) => {
            NET.with(|n| *n.borrow_mut() = Some(ep));
            1
        }
        Err(_) => 0,
    }
}

/// Point this endpoint at a peer `"host:port"`. Returns 1/0.
#[no_mangle]
pub extern "C" fn aurora_net_connect(ptr: *const u8, len: i64) -> i64 {
    let addr = {
        let s = unsafe { std::slice::from_raw_parts(ptr, len.max(0) as usize) };
        String::from_utf8_lossy(s).into_owned()
    };
    NET.with(|n| match n.borrow_mut().as_mut() {
        Some(ep) => ep.connect(&addr).is_ok() as i64,
        None => 0,
    })
}

/// Reliably send a string message. Returns 1/0.
#[no_mangle]
pub extern "C" fn aurora_net_send(ptr: *const u8, len: i64) -> i64 {
    let msg = unsafe { std::slice::from_raw_parts(ptr, len.max(0) as usize) }.to_vec();
    NET.with(|n| match n.borrow_mut().as_mut() {
        Some(ep) => {
            ep.queue(msg);
            ep.flush().is_ok() as i64
        }
        None => 0,
    })
}

/// Receive the next delivered message into `out` (empty string if none pending).
/// Pumps the socket first, buffering any newly-delivered messages in order.
#[no_mangle]
pub extern "C" fn aurora_net_recv(out: *mut i64) {
    NET.with(|n| {
        if let Some(ep) = n.borrow_mut().as_mut() {
            let delivered = ep.poll();
            NET_INBOX.with(|q| q.borrow_mut().extend(delivered));
        }
    });
    let msg = NET_INBOX.with(|q| q.borrow_mut().pop_front()).unwrap_or_default();
    unsafe { write_str(out, msg) };
}

// --- data-parallel execution ------------------------------------------------
//
// `par_for(out, f)` fills `out[i] = f(i)` across OS threads. Each thread writes a
// disjoint slice of `out`, and the closure `f` runs as reentrant native code, so
// there's no data race (the only shared state is the read-only closure env and
// disjoint output slots). The closure is `[fn_ptr, env_ptr]`; lambda-lifted
// closures take `(env_ptr, i)` and return i64.

#[no_mangle]
pub extern "C" fn aurora_par_for(out: *mut i64, n: i64, fn_ptr: *const u8, env_ptr: *const u8) {
    let n = n.max(0) as usize;
    if n == 0 {
        return;
    }
    // Raw addresses as `usize` are `Send`; pointers are not.
    let out_addr = out as usize;
    let fn_addr = fn_ptr as usize;
    let env_addr = env_ptr as usize;
    let threads = std::thread::available_parallelism().map(|x| x.get()).unwrap_or(4).min(n);
    let chunk = n.div_ceil(threads);

    std::thread::scope(|scope| {
        for t in 0..threads {
            scope.spawn(move || {
                // SAFETY: `fn_ptr` is finalized JIT code (executable, shared); each
                // thread writes only its disjoint `[start, end)` slice of `out`.
                let f: extern "C" fn(i64, i64) -> i64 = unsafe { std::mem::transmute(fn_addr) };
                let start = t * chunk;
                let end = ((t + 1) * chunk).min(n);
                for i in start..end {
                    let r = f(env_addr as i64, i as i64);
                    unsafe { *(out_addr as *mut i64).add(i) = r };
                }
            });
        }
    });
}

// --- native ECS world ------------------------------------------------------

/// `&mut` components are raw pointers into this storage, so writes from compiled
/// code persist directly.
#[derive(Default)]
struct World {
    next: i64,
    entities: Vec<i64>,
    comps: std::collections::HashMap<(i64, i64), Box<[u8]>>,
}
thread_local! {
    static WORLD: RefCell<World> = RefCell::new(World::default());
    /// Query results are per-thread, so systems running concurrently under the
    /// parallel scheduler each iterate their own match set instead of clobbering
    /// one shared buffer. (Single-threaded execution is unaffected.)
    static QUERY: RefCell<Vec<i64>> = const { RefCell::new(Vec::new()) };
}

// --- scoped shared-world routing (parallel scheduler) ----------------------
//
// Normally every ECS host fn touches the *calling thread's* thread_local
// `WORLD`. During `aurora_run_parallel`, systems run on worker threads whose own
// thread_local world is empty, so their world access is routed to the single
// shared world owned by the main thread. `PAR_WORLD`, when non-null, points at a
// `ParWorld` living on the main thread's stack for the duration of one batch.
//
// SAFETY rests on the §6.2 data-race-freedom theorem the compiler already
// enforces: two systems run concurrently only when their component access sets
// don't conflict, so no two threads ever touch the same component buffer
// mutably. The `Mutex` serialises only *structural* map access (lookup/insert);
// component data is then written through raw pointers into heap-stable
// `Box<[u8]>` buffers, which unrelated inserts never reallocate.
struct ParWorld {
    lock: std::sync::Mutex<()>,
    world: *mut World,
}
static PAR_WORLD: std::sync::atomic::AtomicPtr<ParWorld> =
    std::sync::atomic::AtomicPtr::new(std::ptr::null_mut());

/// Route ECS world access: the shared world under a lock during a parallel
/// batch, otherwise the calling thread's thread_local world.
fn with_world<R>(f: impl FnOnce(&mut World) -> R) -> R {
    let p = PAR_WORLD.load(std::sync::atomic::Ordering::Acquire);
    if p.is_null() {
        WORLD.with(|w| f(&mut w.borrow_mut()))
    } else {
        // SAFETY: `p` points at a `ParWorld` on the main thread's stack that
        // outlives the `thread::scope` in `aurora_run_parallel`; the lock guards
        // concurrent structural access to the shared world.
        let par = unsafe { &*p };
        let _guard = par.lock.lock().unwrap();
        f(unsafe { &mut *par.world })
    }
}

#[no_mangle]
pub extern "C" fn aurora_spawn_entity() -> i64 {
    with_world(|w| {
        let e = w.next;
        w.next += 1;
        w.entities.push(e);
        e
    })
}
#[no_mangle]
pub extern "C" fn aurora_despawn(e: i64) {
    with_world(|w| {
        w.entities.retain(|&x| x != e);
        w.comps.retain(|&(ent, _), _| ent != e);
    });
}
#[no_mangle]
pub extern "C" fn aurora_store_component(e: i64, tid: i64, ptr: *const u8, size: i64) {
    let bytes = unsafe { std::slice::from_raw_parts(ptr, size.max(0) as usize) };
    with_world(|w| {
        w.comps.insert((e, tid), bytes.to_vec().into_boxed_slice());
    });
}
#[no_mangle]
pub extern "C" fn aurora_get_component(e: i64, tid: i64) -> *mut u8 {
    with_world(|w| match w.comps.get_mut(&(e, tid)) {
        Some(b) => b.as_mut_ptr(),
        None => std::ptr::null_mut(),
    })
}
#[no_mangle]
pub extern "C" fn aurora_query_begin(ids: *const i64, n: i64) -> i64 {
    let ids = unsafe { std::slice::from_raw_parts(ids, n.max(0) as usize) };
    let matches: Vec<i64> = with_world(|w| {
        w.entities
            .iter()
            .copied()
            .filter(|&e| ids.iter().all(|&t| w.comps.contains_key(&(e, t))))
            .collect()
    });
    let len = matches.len() as i64;
    QUERY.with(|q| *q.borrow_mut() = matches);
    len
}
#[no_mangle]
pub extern "C" fn aurora_query_entity(i: i64) -> i64 {
    QUERY.with(|q| q.borrow().get(i.max(0) as usize).copied().unwrap_or(-1))
}
#[no_mangle]
pub extern "C" fn aurora_entity_count() -> i64 {
    with_world(|w| w.entities.len() as i64)
}

/// Run a batch of zero-arg system functions concurrently over the shared ECS
/// world. The §6.2 scheduler check guarantees the systems handed to one batch
/// have non-conflicting component access, so concurrent execution is race-free.
/// `fns` is an array of `n` raw function addresses (each an `extern "C" fn()`).
#[no_mangle]
pub extern "C" fn aurora_run_parallel(fns: *const usize, n: i64) {
    let n = n.max(0) as usize;
    if n == 0 {
        return;
    }
    let addrs: Vec<usize> = unsafe { std::slice::from_raw_parts(fns, n) }.to_vec();
    if n == 1 {
        // One system in the layer: run it inline, no threads or routing needed.
        let f: extern "C" fn() = unsafe { std::mem::transmute(addrs[0]) };
        f();
        return;
    }
    WORLD.with(|w| {
        // `as_ptr` yields `*mut World` without taking a RefCell borrow, so the
        // worker threads (which route through `PAR_WORLD` + lock) are the only
        // accessors during the scope.
        let mut par = ParWorld { lock: std::sync::Mutex::new(()), world: w.as_ptr() };
        let prev = PAR_WORLD.swap(&mut par as *mut ParWorld, std::sync::atomic::Ordering::AcqRel);
        std::thread::scope(|scope| {
            for &a in &addrs {
                scope.spawn(move || {
                    // SAFETY: `a` is a finalized native function address; `usize`
                    // is `Send`. System bodies access the world only through the
                    // routing layer above.
                    let f: extern "C" fn() = unsafe { std::mem::transmute(a) };
                    f();
                });
            }
        });
        PAR_WORLD.store(prev, std::sync::atomic::Ordering::Release);
    });
}

// --- scene system: persist/restore the ECS world ---------------------------

fn put_i64(buf: &mut Vec<u8>, n: i64) {
    buf.extend_from_slice(&n.to_le_bytes());
}
fn get_i64(buf: &[u8], pos: &mut usize) -> Option<i64> {
    let end = *pos + 8;
    let v = i64::from_le_bytes(buf.get(*pos..end)?.try_into().ok()?);
    *pos = end;
    Some(v)
}

/// Save the entire ECS world (entities + components) to `path`. Returns 1/0.
#[no_mangle]
pub extern "C" fn aurora_scene_save(ptr: *const u8, len: i64) -> i64 {
    let path = {
        let s = unsafe { std::slice::from_raw_parts(ptr, len.max(0) as usize) };
        String::from_utf8_lossy(s).into_owned()
    };
    let bytes = WORLD.with(|w| {
        let w = w.borrow();
        let mut b = Vec::new();
        b.extend_from_slice(b"ASCN"); // magic
        put_i64(&mut b, w.next);
        put_i64(&mut b, w.entities.len() as i64);
        for &e in &w.entities {
            put_i64(&mut b, e);
        }
        put_i64(&mut b, w.comps.len() as i64);
        for (&(ent, tid), data) in &w.comps {
            put_i64(&mut b, ent);
            put_i64(&mut b, tid);
            put_i64(&mut b, data.len() as i64);
            b.extend_from_slice(data);
        }
        b
    });
    if std::fs::write(&path, bytes).is_ok() {
        1
    } else {
        0
    }
}

/// Replace the ECS world with the scene saved at `path`. Returns 1/0.
#[no_mangle]
pub extern "C" fn aurora_scene_load(ptr: *const u8, len: i64) -> i64 {
    let path = {
        let s = unsafe { std::slice::from_raw_parts(ptr, len.max(0) as usize) };
        String::from_utf8_lossy(s).into_owned()
    };
    let Ok(b) = std::fs::read(&path) else { return 0 };
    if b.len() < 4 || &b[0..4] != b"ASCN" {
        return 0;
    }
    let mut pos = 4;
    let mut parse = || -> Option<World> {
        let mut world = World::default();
        world.next = get_i64(&b, &mut pos)?;
        let n_ent = get_i64(&b, &mut pos)?;
        for _ in 0..n_ent {
            world.entities.push(get_i64(&b, &mut pos)?);
        }
        let n_comp = get_i64(&b, &mut pos)?;
        for _ in 0..n_comp {
            let ent = get_i64(&b, &mut pos)?;
            let tid = get_i64(&b, &mut pos)?;
            let size = get_i64(&b, &mut pos)?.max(0) as usize;
            let data = b.get(pos..pos + size)?.to_vec().into_boxed_slice();
            pos += size;
            world.comps.insert((ent, tid), data);
        }
        Some(world)
    };
    match parse() {
        Some(w) => {
            WORLD.with(|world| *world.borrow_mut() = w);
            1
        }
        None => 0,
    }
}

// --- profiler: per-function call counts + time ------------------------------
//
// In profiling builds the compiler emits `aurora_prof_enter(name)` at each
// function entry and `aurora_prof_exit()` at each return, accumulating call
// counts and wall-clock time per function — a real instrumenting profiler over
// the native code.

#[derive(Default)]
struct Profiler {
    stack: Vec<(String, std::time::Instant)>,
    totals: std::collections::HashMap<String, (u64, u128)>, // name -> (calls, nanos)
}
thread_local! {
    static PROF: RefCell<Profiler> = RefCell::new(Profiler::default());
}

/// One profiler sample: function name, call count, total nanoseconds.
#[derive(Clone, Debug)]
pub struct ProfRow {
    pub func: String,
    pub calls: u64,
    pub nanos: u128,
}

pub fn prof_reset() {
    PROF.with(|p| {
        let mut p = p.borrow_mut();
        p.stack.clear();
        p.totals.clear();
    });
}

/// Per-function profile rows, sorted by total time descending.
pub fn prof_report() -> Vec<ProfRow> {
    PROF.with(|p| {
        let mut rows: Vec<ProfRow> = p
            .borrow()
            .totals
            .iter()
            .map(|(f, &(calls, nanos))| ProfRow { func: f.clone(), calls, nanos })
            .collect();
        rows.sort_by(|a, b| b.nanos.cmp(&a.nanos));
        rows
    })
}

#[no_mangle]
pub extern "C" fn aurora_prof_enter(name_ptr: *const u8, name_len: i64) {
    let name = {
        let s = unsafe { std::slice::from_raw_parts(name_ptr, name_len.max(0) as usize) };
        String::from_utf8_lossy(s).into_owned()
    };
    PROF.with(|p| p.borrow_mut().stack.push((name, std::time::Instant::now())));
}

#[no_mangle]
pub extern "C" fn aurora_prof_exit() {
    PROF.with(|p| {
        let mut p = p.borrow_mut();
        if let Some((name, start)) = p.stack.pop() {
            let ns = start.elapsed().as_nanos();
            let e = p.totals.entry(name).or_insert((0, 0));
            e.0 += 1;
            e.1 += ns;
        }
    });
}

// --- audio + windowing builtins --------------------------------------------
//
// These back Aurora's `play_note`, `window_open`, `window_present`, and
// `key_down` builtins, wiring the language to real audio output (cpal) and a
// real-time window (winit + wgpu) that presents the builtin framebuffer.

/// Synthesize and play one note: `semitone` is relative to A4, `dur_ms` ms long.
/// Blocks until the note finishes (so notes sequence naturally).
#[no_mangle]
pub extern "C" fn aurora_play_note(semitone: i64, dur_ms: i64) {
    let sr = 44_100;
    let dur = (dur_ms.max(0) as f32) / 1000.0;
    let note = aurora_audio::Note::new(aurora_audio::pitch(semitone as i32), dur)
        .wave(aurora_audio::Wave::Triangle)
        .gain(0.5);
    let _ = aurora_audio::play(&note.render(sr), sr);
}

/// Run a user fragment shader on the GPU into the builtin framebuffer. `wgsl` is
/// a fragment shader body (defining `fs_main`, reading `uv` and `u.time`).
/// `time_ms` animates it. The result replaces the framebuffer, so the next
/// `window_present`/`save_ppm` shows the GPU-rendered image.
#[no_mangle]
pub extern "C" fn aurora_gpu_render(ptr: *const u8, len: i64, time_ms: i64) {
    let wgsl = {
        let s = unsafe { std::slice::from_raw_parts(ptr, len.max(0) as usize) };
        String::from_utf8_lossy(s).into_owned()
    };
    FB.with(|fb| {
        let mut fb = fb.borrow_mut();
        if let Some(f) = fb.as_mut() {
            let (w, h) = (f.width(), f.height());
            let rgba = aurora_gpu::render_shader(&wgsl, w, h, time_ms as f32 / 1000.0);
            if !rgba.is_empty() {
                f.set_rgba(&rgba);
            }
        }
    });
}

/// Run a compute shader on the GPU over an `[f64; n]` array, in place. `wgsl`
/// operates on a `read_write array<f32>` at binding 0. Values are converted
/// f64→f32 for the GPU and back. Backs the `gpu_compute` builtin.
#[no_mangle]
pub extern "C" fn aurora_gpu_compute(wptr: *const u8, wlen: i64, data: *mut f64, n: i64) {
    let wgsl = {
        let s = unsafe { std::slice::from_raw_parts(wptr, wlen.max(0) as usize) };
        String::from_utf8_lossy(s).into_owned()
    };
    let n = n.max(0) as usize;
    let slice = unsafe { std::slice::from_raw_parts_mut(data, n) };
    let input: Vec<f32> = slice.iter().map(|&x| x as f32).collect();
    let out = aurora_gpu::compute(&wgsl, &input);
    for (dst, &v) in slice.iter_mut().zip(out.iter()) {
        *dst = v as f64;
    }
}

/// Open a real-time window backing a `w`×`h` builtin framebuffer.
#[no_mangle]
pub extern "C" fn aurora_window_open(w: i64, h: i64) {
    aurora_window::imm_open(w.max(0) as u32, h.max(0) as u32);
}

/// Present the current framebuffer and pump events; returns 1 while open, 0 when
/// the window has been closed.
#[no_mangle]
pub extern "C" fn aurora_window_present() -> i64 {
    let rgba = FB.with(|fb| fb.borrow().as_ref().map(|f| f.rgba()).unwrap_or_default());
    if aurora_window::imm_present(&rgba) {
        1
    } else {
        0
    }
}

/// Whether the given Aurora key code is currently held (1) or not (0).
#[no_mangle]
pub extern "C" fn aurora_key_down(code: i64) -> i64 {
    if aurora_window::imm_key_down(code.max(0) as u32) {
        1
    } else {
        0
    }
}

/// Pop the next typed character code (0 if none); Backspace = 8. For text fields.
#[no_mangle]
pub extern "C" fn aurora_input_char() -> i64 {
    aurora_window::imm_input_char()
}
/// Set fullscreen mode: 0 windowed, 1 borderless, 2 exclusive.
#[no_mangle]
pub extern "C" fn aurora_window_fullscreen(mode: i64) {
    aurora_window::imm_window_fullscreen(mode);
}

/// Mouse X in framebuffer pixels.
#[no_mangle]
pub extern "C" fn aurora_mouse_x() -> i64 {
    aurora_window::imm_mouse().0
}

/// Mouse Y in framebuffer pixels.
#[no_mangle]
pub extern "C" fn aurora_mouse_y() -> i64 {
    aurora_window::imm_mouse().1
}

/// Whether the left mouse button is held (1) or not (0).
#[no_mangle]
pub extern "C" fn aurora_mouse_down() -> i64 {
    if aurora_window::imm_mouse().2 {
        1
    } else {
        0
    }
}

// --- 3D rendering (the `r3d_*` builtins) -----------------------------------
//
// These drive the GPU 3D renderer that lives in the window (`aurora-render3d`),
// sharing the window's wgpu device. Colors are 0..1 floats; angles are radians.

/// Load a glTF/GLB/OBJ model; returns a handle (>= 0) or -1.
#[no_mangle]
pub extern "C" fn aurora_r3d_load_model(ptr: *const u8, len: i64) -> i64 {
    let s = unsafe { std::slice::from_raw_parts(ptr, len.max(0) as usize) };
    let path = String::from_utf8_lossy(s);
    aurora_window::imm_r3d_load_model(&path)
}
#[no_mangle]
pub extern "C" fn aurora_r3d_make_box(r: f64, g: f64, b: f64) -> i64 {
    aurora_window::imm_r3d_make_box(r as f32, g as f32, b as f32)
}
/// A box mesh sized by half-extents (matching a physics box collider), colored.
#[no_mangle]
pub extern "C" fn aurora_r3d_make_box_sized(
    hx: f64, hy: f64, hz: f64, r: f64, g: f64, b: f64,
) -> i64 {
    aurora_window::imm_r3d_make_box_sized(hx as f32, hy as f32, hz as f32, r as f32, g as f32, b as f32)
}
/// An emissive (self-lit, glowing) box mesh. Color is the emissive RGB.
#[no_mangle]
pub extern "C" fn aurora_r3d_make_box_emissive(
    hx: f64, hy: f64, hz: f64, r: f64, g: f64, b: f64,
) -> i64 {
    aurora_window::imm_r3d_make_box_emissive(hx as f32, hy as f32, hz as f32, r as f32, g as f32, b as f32)
}
#[no_mangle]
pub extern "C" fn aurora_r3d_make_sphere(segments: i64, r: f64, g: f64, b: f64) -> i64 {
    aurora_window::imm_r3d_make_sphere(segments, r as f32, g as f32, b as f32)
}
#[no_mangle]
pub extern "C" fn aurora_r3d_make_plane(size: f64, tiles: f64, r: f64, g: f64, b: f64) -> i64 {
    aurora_window::imm_r3d_make_plane(size as f32, tiles as f32, r as f32, g as f32, b as f32)
}
#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub extern "C" fn aurora_r3d_camera(ex: f64, ey: f64, ez: f64, tx: f64, ty: f64, tz: f64, fov: f64) {
    aurora_window::imm_r3d_camera(
        ex as f32, ey as f32, ez as f32, tx as f32, ty as f32, tz as f32, fov as f32,
    );
}
/// Set the camera roll (banking) in radians, for wallrun lean / strafe tilt.
#[no_mangle]
pub extern "C" fn aurora_r3d_camera_roll(roll: f64) {
    aurora_window::imm_r3d_camera_roll(roll as f32);
}
#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub extern "C" fn aurora_r3d_light(dx: f64, dy: f64, dz: f64, r: f64, g: f64, b: f64, ambient: f64) {
    aurora_window::imm_r3d_light(
        dx as f32, dy as f32, dz as f32, r as f32, g as f32, b as f32, ambient as f32,
    );
}
#[no_mangle]
pub extern "C" fn aurora_r3d_clear(r: f64, g: f64, b: f64) {
    aurora_window::imm_r3d_clear(r as f32, g as f32, b as f32);
}
#[no_mangle]
pub extern "C" fn aurora_r3d_begin() {
    aurora_window::imm_r3d_begin();
}
#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub extern "C" fn aurora_r3d_draw(
    h: i64, px: f64, py: f64, pz: f64, yaw: f64, pitch: f64, roll: f64, scale: f64,
) {
    aurora_window::imm_r3d_draw(
        h, px as f32, py as f32, pz as f32, yaw as f32, pitch as f32, roll as f32, scale as f32,
    );
}
#[no_mangle]
pub extern "C" fn aurora_r3d_draw_tint(
    h: i64, px: f64, py: f64, pz: f64, yaw: f64, pitch: f64, roll: f64, scale: f64, r: f64, g: f64, b: f64,
) {
    aurora_window::imm_r3d_draw_tint(
        h, px as f32, py as f32, pz as f32, yaw as f32, pitch as f32, roll as f32, scale as f32,
        r as f32, g as f32, b as f32,
    );
}
#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub extern "C" fn aurora_r3d_draw_shield(
    h: i64, px: f64, py: f64, pz: f64, yaw: f64, pitch: f64, roll: f64, scale: f64, strength: f64, time: f64,
) {
    aurora_window::imm_r3d_draw_shield(
        h, px as f32, py as f32, pz as f32, yaw as f32, pitch as f32, roll as f32, scale as f32,
        strength as f32, time as f32,
    );
}
#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub extern "C" fn aurora_r3d_draw_on_joint(
    weapon: i64, host: i64, joint: i64,
    hx: f64, hy: f64, hz: f64, hyaw: f64, hpitch: f64, hroll: f64, hscale: f64,
    ox: f64, oy: f64, oz: f64, oyaw: f64, opitch: f64, oroll: f64, oscale: f64,
) {
    aurora_window::imm_r3d_draw_on_joint(
        weapon, host, joint,
        hx as f32, hy as f32, hz as f32, hyaw as f32, hpitch as f32, hroll as f32, hscale as f32,
        ox as f32, oy as f32, oz as f32, oyaw as f32, opitch as f32, oroll as f32, oscale as f32,
    );
}
#[no_mangle]
pub extern "C" fn aurora_r3d_joint_dump(host: i64) {
    aurora_window::imm_r3d_joint_dump(host);
}
#[no_mangle]
pub extern "C" fn aurora_r3d_anim_play(h: i64, clip: i64, looping: i64, speed: f64, fade: f64) {
    aurora_window::imm_r3d_anim_play(h, clip, looping, speed as f32, fade as f32);
}
#[no_mangle]
pub extern "C" fn aurora_r3d_anim_update(h: i64, dt: f64) {
    aurora_window::imm_r3d_anim_update(h, dt as f32);
}
#[no_mangle]
pub extern "C" fn aurora_r3d_anim_play_upper(h: i64, clip: i64, looping: i64, speed: f64, fade: f64, mask_root: i64) {
    aurora_window::imm_r3d_anim_play_upper(h, clip, looping, speed as f32, fade as f32, mask_root);
}
#[no_mangle]
pub extern "C" fn aurora_r3d_anim_stop_upper(h: i64, fade: f64) {
    aurora_window::imm_r3d_anim_stop_upper(h, fade as f32);
}
#[no_mangle]
pub extern "C" fn aurora_r3d_clip_count(h: i64) -> i64 {
    aurora_window::imm_r3d_clip_count(h)
}
#[no_mangle]
pub extern "C" fn aurora_r3d_present() -> i64 {
    // Overlay the CPU framebuffer (HUD: text/crosshair/2D) over the 3D scene. Pass
    // the framebuffer dimensions so the HUD texture can track its size (a game can
    // size its HUD framebuffer to the live window for a crisp 1:1 overlay).
    let (rgba, w, h) = FB.with(|fb| {
        fb.borrow()
            .as_ref()
            .map(|f| (f.rgba(), f.width(), f.height()))
            .unwrap_or((Vec::new(), 0, 0))
    });
    if aurora_window::imm_r3d_present(&rgba, w, h) {
        1
    } else {
        0
    }
}

/// Current window/surface size in physical pixels (0 before the window exists).
#[no_mangle]
pub extern "C" fn aurora_surface_w() -> i64 {
    aurora_window::imm_surface_w() as i64
}
#[no_mangle]
pub extern "C" fn aurora_surface_h() -> i64 {
    aurora_window::imm_surface_h() as i64
}
#[no_mangle]
pub extern "C" fn aurora_r3d_fog(r: f64, g: f64, b: f64, density: f64) {
    aurora_window::imm_r3d_fog(r as f32, g as f32, b as f32, density as f32);
}
/// Set the procedural speed/wind-lines overlay (intensity 0..1, animation time).
#[no_mangle]
pub extern "C" fn aurora_r3d_speedlines(intensity: f64, time: f64) {
    aurora_window::imm_speedlines(intensity as f32, time as f32);
}
/// Set the damage overlay: low-health vignette (0..1), directional hit glow (0..1),
/// the hit direction in screen space (dx, dy), and a gold overclock tint `oc` (0..1).
#[no_mangle]
pub extern "C" fn aurora_r3d_damage(vig: f64, hit: f64, dx: f64, dy: f64, oc: f64) {
    aurora_window::imm_damage(vig as f32, hit as f32, dx as f32, dy as f32, oc as f32);
}
/// Set the fullscreen blur radius in pixels (0 = off): the paused/menu backdrop.
#[no_mangle]
pub extern "C" fn aurora_r3d_blur(radius: f64) {
    aurora_window::imm_blur(radius as f32);
}
#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub extern "C" fn aurora_r3d_sky(on: i64, tr: f64, tg: f64, tb: f64, hr: f64, hg: f64, hb: f64) {
    aurora_window::imm_r3d_sky(on, tr as f32, tg as f32, tb as f32, hr as f32, hg as f32, hb as f32);
}
#[no_mangle]
pub extern "C" fn aurora_r3d_shadows(on: i64) {
    aurora_window::imm_r3d_shadows(on);
}
#[no_mangle]
pub extern "C" fn aurora_r3d_ssao(on: i64) {
    aurora_window::imm_r3d_ssao(on);
}
#[no_mangle]
pub extern "C" fn aurora_r3d_point_shadows(on: i64) {
    aurora_window::imm_r3d_point_shadows(on);
}
#[no_mangle]
pub extern "C" fn aurora_r3d_clear_lights() {
    aurora_window::imm_r3d_clear_lights();
}
#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub extern "C" fn aurora_r3d_point_light(x: f64, y: f64, z: f64, r: f64, g: f64, b: f64, range: f64, intensity: f64) {
    aurora_window::imm_r3d_point_light(
        x as f32, y as f32, z as f32, r as f32, g as f32, b as f32, range as f32, intensity as f32,
    );
}
#[no_mangle]
pub extern "C" fn aurora_r3d_make_sprite(r: f64, g: f64, b: f64) -> i64 {
    aurora_window::imm_r3d_make_sprite(r as f32, g as f32, b as f32)
}
#[no_mangle]
pub extern "C" fn aurora_r3d_draw_billboard(h: i64, x: f64, y: f64, z: f64, size: f64) {
    aurora_window::imm_r3d_draw_billboard(h, x as f32, y as f32, z as f32, size as f32);
}
#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub extern "C" fn aurora_r3d_debug_line(ax: f64, ay: f64, az: f64, bx: f64, by: f64, bz: f64, r: f64, g: f64, b: f64) {
    aurora_window::imm_r3d_debug_line(
        ax as f32, ay as f32, az as f32, bx as f32, by as f32, bz as f32, r as f32, g as f32, b as f32,
    );
}
#[no_mangle]
pub extern "C" fn aurora_r3d_frustum_cull(on: i64) {
    aurora_window::imm_r3d_frustum_cull(on);
}
#[no_mangle]
pub extern "C" fn aurora_r3d_screen_x(wx: f64, wy: f64, wz: f64) -> f64 {
    let (x, _, vis) = aurora_window::imm_r3d_world_to_screen(wx as f32, wy as f32, wz as f32);
    if vis { x as f64 } else { -1.0 }
}
#[no_mangle]
pub extern "C" fn aurora_r3d_screen_y(wx: f64, wy: f64, wz: f64) -> f64 {
    let (_, y, vis) = aurora_window::imm_r3d_world_to_screen(wx as f32, wy as f32, wz as f32);
    if vis { y as f64 } else { -1.0 }
}

// --- FPS input ---
#[no_mangle]
pub extern "C" fn aurora_mouse_dx() -> f64 {
    aurora_window::imm_mouse_delta().0
}
#[no_mangle]
pub extern "C" fn aurora_mouse_dy() -> f64 {
    aurora_window::imm_mouse_delta().1
}
#[no_mangle]
pub extern "C" fn aurora_mouse_scroll() -> f64 {
    aurora_window::imm_scroll()
}
#[no_mangle]
pub extern "C" fn aurora_mouse_button(b: i64) -> i64 {
    if aurora_window::imm_mouse_button(b.max(0) as u32) {
        1
    } else {
        0
    }
}
#[no_mangle]
pub extern "C" fn aurora_grab_mouse(on: i64) {
    aurora_window::imm_grab_mouse(on != 0);
}

// --- rebindable input actions ----------------------------------------------
//
// Decouple the game from physical keys: it binds abstract ACTIONS to input codes
// (rebindable at runtime, e.g. from a settings menu) and queries actions, never
// raw keys. Codes 0..65 are keyboard (the `key_down` codes); 100/101/102 are the
// left/right/middle mouse buttons.

thread_local! {
    static BINDINGS: RefCell<std::collections::HashMap<i64, i64>> =
        RefCell::new(std::collections::HashMap::new());
    // When set, the bind-layer reads (input_down / input_axis) all report "not held",
    // so a game can freeze player actions in one call (e.g. a pause overlay) without
    // touching the raw mouse used by menus.
    static INPUT_SUPPRESS: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Suppress (1) or restore (0) all bound-action input. While suppressed, every
/// `input_down`/`input_axis` reads as zero; the raw mouse/keyboard queries are
/// untouched so menus still work.
#[no_mangle]
pub extern "C" fn aurora_input_suppress(on: i64) {
    INPUT_SUPPRESS.with(|s| s.set(on != 0));
}

fn code_is_down(code: i64) -> bool {
    if code < 0 {
        false
    } else if code >= 100 {
        aurora_window::imm_mouse_button((code - 100) as u32)
    } else {
        aurora_window::imm_key_down(code as u32)
    }
}

/// Bind an action id to an input code (rebindable any time).
#[no_mangle]
pub extern "C" fn aurora_input_bind(action: i64, code: i64) {
    BINDINGS.with(|b| {
        b.borrow_mut().insert(action, code);
    });
}

/// The input code currently bound to an action, or -1 if unbound.
#[no_mangle]
pub extern "C" fn aurora_input_binding(action: i64) -> i64 {
    BINDINGS.with(|b| b.borrow().get(&action).copied().unwrap_or(-1))
}

/// Whether an action's bound input is currently held (1) or not (0).
#[no_mangle]
pub extern "C" fn aurora_input_down(action: i64) -> i64 {
    if INPUT_SUPPRESS.with(|s| s.get()) {
        return 0;
    }
    let code = BINDINGS.with(|b| b.borrow().get(&action).copied().unwrap_or(-1));
    code_is_down(code) as i64
}

/// A -1/0/+1 axis from two opposing actions (e.g. back/forward).
#[no_mangle]
pub extern "C" fn aurora_input_axis(neg: i64, pos: i64) -> f64 {
    let p = aurora_input_down(pos) as f64;
    let n = aurora_input_down(neg) as f64;
    p - n
}

/// Read the `i`-th `f32` at a raw pointer (passed as integer bits), widened to
/// `f64`. Lets Aurora sim code read the opaque `f32` state/input blobs the
/// netcode framework hands it (see `aurora_net_sim`).
#[no_mangle]
pub extern "C" fn aurora_f32_load(ptr: i64, i: i64) -> f64 {
    if ptr == 0 || i < 0 {
        return 0.0;
    }
    unsafe { *(ptr as *const f32).add(i as usize) as f64 }
}

/// Write `v` (narrowed to `f32`) as the `i`-th `f32` at a raw pointer.
#[no_mangle]
pub extern "C" fn aurora_f32_store(ptr: i64, i: i64, v: f64) {
    if ptr == 0 || i < 0 {
        return;
    }
    unsafe { *(ptr as *mut f32).add(i as usize) = v as f32 };
}

// Transcendental math builtins. Cranelift has no native instruction for these,
// so they are host calls into Rust's libm (a correct, ABI-safe path, unlike a
// raw libcall import). `sqrt`/`floor`/`abs`/`min`/`max`/`clamp` stay native in
// codegen; these are the ones that need a real function call.
#[no_mangle]
pub extern "C" fn aurora_sin(x: f64) -> f64 {
    x.sin()
}
#[no_mangle]
pub extern "C" fn aurora_cos(x: f64) -> f64 {
    x.cos()
}
#[no_mangle]
pub extern "C" fn aurora_tan(x: f64) -> f64 {
    x.tan()
}
#[no_mangle]
pub extern "C" fn aurora_pow(x: f64, y: f64) -> f64 {
    x.powf(y)
}
#[no_mangle]
pub extern "C" fn aurora_log(x: f64) -> f64 {
    x.ln()
}
#[no_mangle]
pub extern "C" fn aurora_exp(x: f64) -> f64 {
    x.exp()
}
#[no_mangle]
pub extern "C" fn aurora_atan2(y: f64, x: f64) -> f64 {
    y.atan2(x)
}

/// Play a note WITHOUT blocking — mixed into the persistent audio engine, so
/// sounds and music overlap. `looped` != 0 repeats it until volume/stop.
#[no_mangle]
pub extern "C" fn aurora_play_sound(semitone: i64, dur_ms: i64, looped: i64) {
    let sr = 44_100;
    let dur = (dur_ms.max(0) as f32) / 1000.0;
    let mut note = aurora_audio::Note::new(aurora_audio::pitch(semitone as i32), dur)
        .wave(aurora_audio::Wave::Triangle)
        .gain(0.4);
    // One-shot SFX get a percussive pluck envelope (fast attack, no sustain) so
    // they read as a crisp tick instead of a flat held beep. Looped sounds keep
    // the default sustained envelope (for tones/music).
    if looped == 0 {
        note.adsr = aurora_audio::Adsr {
            attack: 0.001,
            decay: (dur * 0.6).max(0.004),
            sustain: 0.0,
            release: 0.02,
        };
    }
    aurora_audio::play_mixed(&note.render(sr), sr, looped != 0);
}

/// Play a short white-noise burst (percussive, pitch-less) for impact/hit SFX
/// that should read as a "thwack/click" rather than a tone. `gain_pct` is 0..200.
#[no_mangle]
pub extern "C" fn aurora_play_noise(dur_ms: i64, gain_pct: i64) {
    let sr = 44_100;
    let dur = (dur_ms.max(1) as f32) / 1000.0;
    let g = (gain_pct.clamp(0, 200) as f32) / 100.0;
    let mut note =
        aurora_audio::Note::new(440.0, dur).wave(aurora_audio::Wave::Noise).gain(g);
    note.adsr = aurora_audio::Adsr {
        attack: 0.003,                  // soft attack (no click) for a smooth onset
        decay: (dur * 0.85).max(0.004), // long gentle fade
        sustain: 0.0,
        release: 0.02,
    };
    // Heavily low-pass the white noise so it reads as a soft, smooth "pf/pap" (like
    // a muffled hit on paper/cloth), not a piercing high hiss. Lower coefficient =
    // darker/smoother.
    let raw = note.render(sr);
    let mut buf = Vec::with_capacity(raw.len());
    let mut lp = 0.0f32;
    for s in raw {
        lp += 0.09 * (s - lp);
        buf.push(lp);
    }
    aurora_audio::play_mixed(&buf, sr, false);
}

// --- 3D positional audio ---------------------------------------------------

thread_local! {
    // Listener pose: position and forward direction (for panning).
    static LISTENER: RefCell<([f64; 3], [f64; 3])> = const { RefCell::new(([0.0; 3], [0.0, 0.0, -1.0])) };
}

/// Set the audio listener's world position and forward direction. Spatial sounds
/// are attenuated by distance and panned left/right relative to this pose.
#[no_mangle]
pub extern "C" fn aurora_audio_listener(x: f64, y: f64, z: f64, fx: f64, fy: f64, fz: f64) {
    LISTENER.with(|l| *l.borrow_mut() = ([x, y, z], [fx, fy, fz]));
}

/// Compute (gain, pan) for a sound at `pos` relative to the current listener.
/// `max_dist` is the audible range; falloff is quadratic.
fn spatialize(pos: [f64; 3]) -> (f32, f32) {
    LISTENER.with(|l| {
        let (lp, fwd) = *l.borrow();
        let to = [pos[0] - lp[0], pos[1] - lp[1], pos[2] - lp[2]];
        let dist = (to[0] * to[0] + to[1] * to[1] + to[2] * to[2]).sqrt();
        let max_dist = 35.0;
        let g = (1.0 - dist / max_dist).clamp(0.0, 1.0);
        let gain = (g * g) as f32;
        // Pan by the listener's right vector (forward x up).
        let f = norm3(fwd);
        let right = norm3([f[2], 0.0, -f[0]]); // cross(forward, up=+Y), flattened
        let dir = if dist > 1e-4 { [to[0] / dist, to[1] / dist, to[2] / dist] } else { [0.0; 3] };
        let pan = (right[0] * dir[0] + right[1] * dir[1] + right[2] * dir[2]).clamp(-1.0, 1.0) as f32;
        (gain, pan)
    })
}

fn norm3(v: [f64; 3]) -> [f64; 3] {
    let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if l > 1e-6 {
        [v[0] / l, v[1] / l, v[2] / l]
    } else {
        [0.0, 0.0, -1.0]
    }
}

/// Play a synthesized note at a world position, spatialized by distance + pan.
#[no_mangle]
pub extern "C" fn aurora_play_sound_at(
    semitone: i64, dur_ms: i64, gain_pct: i64, x: f64, y: f64, z: f64,
) {
    let (gain, pan) = spatialize([x, y, z]);
    if gain <= 0.001 {
        return;
    }
    let sr = 44_100;
    let dur = (dur_ms.max(0) as f32) / 1000.0;
    // gain_pct lets callers mix levels: quiet background ticks (e.g. gunfire) vs loud
    // foreground hits (explosions). 100 = the old default.
    let g = 0.5 * (gain_pct.max(0) as f32) / 100.0;
    let note = aurora_audio::Note::new(aurora_audio::pitch(semitone as i32), dur)
        .wave(aurora_audio::Wave::Triangle)
        .gain(g);
    aurora_audio::play_mixed_spatial(&note.render(sr), sr, false, gain, pan);
}

/// Persist a small settings blob (`len` f64 values) to a fixed file on disk, one
/// value per line. Backs the `save_settings` builtin (keybinds, sensitivity, volume).
#[no_mangle]
pub extern "C" fn aurora_save_settings(data: *const f64, len: i64) -> i64 {
    if data.is_null() || len <= 0 {
        return 0;
    }
    let slice = unsafe { std::slice::from_raw_parts(data, len as usize) };
    let mut s = String::new();
    for v in slice {
        s.push_str(&format!("{}\n", v));
    }
    let _ = std::fs::write("overclock_settings.txt", s);
    0
}

/// Read the settings blob back into `data` (up to `len` values); returns the count
/// read, or -1 if the file is missing. Backs the `load_settings` builtin.
#[no_mangle]
pub extern "C" fn aurora_load_settings(data: *mut f64, len: i64) -> i64 {
    if data.is_null() || len <= 0 {
        return -1;
    }
    let Ok(s) = std::fs::read_to_string("overclock_settings.txt") else {
        return -1;
    };
    let slice = unsafe { std::slice::from_raw_parts_mut(data, len as usize) };
    let mut n = 0usize;
    for line in s.lines() {
        if n >= len as usize {
            break;
        }
        if let Ok(v) = line.trim().parse::<f64>() {
            slice[n] = v;
            n += 1;
        }
    }
    n as i64
}

/// Load and play a WAV file at `path` through the audio mixer (downmixed to
/// mono, normalized to f32). Returns 1 on success, 0 on failure. Backs the
/// `play_wav` builtin — audio asset playback beyond the synth.
#[no_mangle]
pub extern "C" fn aurora_play_wav(ptr: *const u8, len: i64) -> i64 {
    let path = {
        let s = unsafe { std::slice::from_raw_parts(ptr, len.max(0) as usize) };
        String::from_utf8_lossy(s).into_owned()
    };
    let Ok(mut reader) = hound::WavReader::open(&path) else { return 0 };
    let spec = reader.spec();
    let ch = spec.channels.max(1) as usize;
    let raw: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().filter_map(|s| s.ok()).collect(),
        hound::SampleFormat::Int => {
            let max = (1i64 << (spec.bits_per_sample - 1).max(1)) as f32;
            reader.samples::<i32>().filter_map(|s| s.ok()).map(|s| s as f32 / max).collect()
        }
    };
    let mono: Vec<f32> = if ch <= 1 {
        raw
    } else {
        raw.chunks(ch).map(|c| c.iter().sum::<f32>() / ch as f32).collect()
    };
    if mono.is_empty() {
        return 0;
    }
    aurora_audio::play_mixed(&mono, spec.sample_rate, false);
    1
}

/// Set the master audio volume from a 0..=100 percentage.
#[no_mangle]
pub extern "C" fn aurora_audio_volume(percent: i64) {
    aurora_audio::set_volume(percent.clamp(0, 200) as f32 / 100.0);
}

/// Stop all currently-playing sounds.
#[no_mangle]
pub extern "C" fn aurora_audio_stop() {
    aurora_audio::stop_all();
}

// --- native debugger support ----------------------------------------------
//
// In debug builds the compiler instruments the *native* code: a call to
// `aurora_dbg_enter` at each function entry, `aurora_dbg_stmt(line)` before each
// statement, and `aurora_dbg_var(name, value)` after each scalar binding. The
// program runs at full native speed; these hooks just maintain a little state
// here so a debugger front-end can set breakpoints and inspect locals.

/// A local variable's value as seen by the debugger. Aggregates are reported
/// field-by-field with dotted names (e.g. `v.x`), so only scalar leaves appear.
#[derive(Clone, Debug, PartialEq)]
pub enum DbgVal {
    Int(i64),
    Float(f64),
}

impl std::fmt::Display for DbgVal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DbgVal::Int(n) => write!(f, "{n}"),
            DbgVal::Float(x) => write!(f, "{x}"),
        }
    }
}

/// A recorded pause: the source line, the locals in the current (innermost)
/// frame, and the call stack (outermost first, innermost last).
#[derive(Clone, Debug, PartialEq)]
pub struct Stop {
    pub line: u32,
    pub vars: Vec<(String, DbgVal)>,
    pub stack: Vec<String>,
}

/// What the interactive front-end wants to do after a stop.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DbgCmd {
    Continue,
    Step,
    Quit,
}

/// One call frame: the function name and its scalar locals.
#[derive(Default)]
struct Frame {
    func: String,
    vars: Vec<(String, DbgVal)>,
}

#[derive(Default)]
struct DebugState {
    breakpoints: HashSet<u32>,
    step: bool,
    frames: Vec<Frame>,
    stops: Vec<Stop>,
    handler: Option<Box<dyn FnMut(&Stop) -> DbgCmd>>,
}
thread_local! {
    static DEBUG: RefCell<DebugState> = RefCell::new(DebugState::default());
}

/// Configure the debugger before a run: which lines break, whether to single-
/// step every statement, and clear any prior recorded stops/locals.
pub fn dbg_reset(breakpoints: HashSet<u32>, step: bool) {
    DEBUG.with(|d| {
        let mut d = d.borrow_mut();
        d.breakpoints = breakpoints;
        d.step = step;
        d.frames.clear();
        d.stops.clear();
        d.handler = None;
    });
}

/// Install an interactive handler invoked at every stop (it decides whether to
/// continue, step, or quit). Without one, stops are simply recorded.
pub fn dbg_set_handler(handler: Box<dyn FnMut(&Stop) -> DbgCmd>) {
    DEBUG.with(|d| d.borrow_mut().handler = Some(handler));
}

/// Take the recorded stops after a run.
pub fn dbg_take_stops() -> Vec<Stop> {
    DEBUG.with(|d| std::mem::take(&mut d.borrow_mut().stops))
}

#[no_mangle]
pub extern "C" fn aurora_dbg_enter(name_ptr: *const u8, name_len: i64) {
    let func = {
        let s = unsafe { std::slice::from_raw_parts(name_ptr, name_len.max(0) as usize) };
        String::from_utf8_lossy(s).into_owned()
    };
    DEBUG.with(|d| d.borrow_mut().frames.push(Frame { func, vars: Vec::new() }));
}

#[no_mangle]
pub extern "C" fn aurora_dbg_leave() {
    DEBUG.with(|d| {
        d.borrow_mut().frames.pop();
    });
}

#[no_mangle]
pub extern "C" fn aurora_dbg_stmt(line: i64) {
    let line = line.max(0) as u32;
    // Decide whether this statement is a pause point, and capture a snapshot of
    // the innermost frame's locals plus the call stack.
    let (paused, snapshot) = DEBUG.with(|d| {
        let d = d.borrow();
        let paused = d.step || d.breakpoints.contains(&line);
        let snap = if paused {
            let vars = d.frames.last().map(|f| f.vars.clone()).unwrap_or_default();
            let stack = d.frames.iter().map(|f| f.func.clone()).collect();
            Some(Stop { line, vars, stack })
        } else {
            None
        };
        (paused, snap)
    });
    let Some(stop) = snapshot else { return };
    if !paused {
        return;
    }
    // Record it, then let any interactive handler steer the run.
    let cmd = DEBUG.with(|d| {
        let mut d = d.borrow_mut();
        d.stops.push(stop.clone());
        d.handler.take()
    });
    if let Some(mut h) = cmd {
        let decision = h(&stop);
        DEBUG.with(|d| {
            let mut d = d.borrow_mut();
            d.handler = Some(h);
            match decision {
                DbgCmd::Step => d.step = true,
                DbgCmd::Continue => d.step = false,
                DbgCmd::Quit => {}
            }
        });
        if decision == DbgCmd::Quit {
            std::process::exit(0);
        }
    }
}

fn dbg_record_var(name_ptr: *const u8, name_len: i64, value: DbgVal) {
    let name = {
        let s = unsafe { std::slice::from_raw_parts(name_ptr, name_len.max(0) as usize) };
        String::from_utf8_lossy(s).into_owned()
    };
    DEBUG.with(|d| {
        let mut d = d.borrow_mut();
        // Update the innermost frame's locals (recursion stays isolated).
        if let Some(frame) = d.frames.last_mut() {
            if let Some(slot) = frame.vars.iter_mut().find(|(n, _)| *n == name) {
                slot.1 = value;
            } else {
                frame.vars.push((name, value));
            }
        }
    });
}

#[no_mangle]
pub extern "C" fn aurora_dbg_var(name_ptr: *const u8, name_len: i64, value: i64) {
    dbg_record_var(name_ptr, name_len, DbgVal::Int(value));
}

#[no_mangle]
pub extern "C" fn aurora_dbg_var_f64(name_ptr: *const u8, name_len: i64, value: f64) {
    dbg_record_var(name_ptr, name_len, DbgVal::Float(value));
}

/// Touch every host symbol so the linker keeps this crate's object in an AOT
/// link even when the Rust driver references nothing from it directly.
pub fn force_link() -> usize {
    let fns: [*const (); 228] = [
        aurora_r3d_draw_shield as *const (),
        aurora_net_player_state as *const (),
        aurora_r3d_draw_on_joint as *const (),
        aurora_r3d_joint_dump as *const (),
        aurora_r3d_blur as *const (),
        aurora_input_suppress as *const (),
        aurora_text_width as *const (),
        aurora_phys3d_add_box_rot as *const (),
        aurora_save_settings as *const (),
        aurora_load_settings as *const (),
        aurora_r3d_ssao as *const (),
        aurora_r3d_point_shadows as *const (),
        // Multiplayer (generic framework: the game registers its Aurora sim).
        aurora_net_host as *const (),
        aurora_net_join as *const (),
        aurora_net_sim as *const (),
        aurora_net_send_input as *const (),
        aurora_net_update as *const (),
        aurora_net_interest as *const (),
        aurora_net_hit_radius as *const (),
        aurora_net_spawn_at as *const (),
        aurora_net_my_id as *const (),
        aurora_net_is_server as *const (),
        aurora_net_player_count as *const (),
        aurora_net_player_id_at as *const (),
        aurora_net_player_x as *const (),
        aurora_net_player_y as *const (),
        aurora_net_player_z as *const (),
        aurora_net_player_yaw as *const (),
        aurora_net_local_x as *const (),
        aurora_net_local_y as *const (),
        aurora_net_local_z as *const (),
        aurora_net_local_yaw as *const (),
        aurora_net_state as *const (),
        aurora_net_local_state as *const (),
        aurora_net_fire as *const (),
        aurora_net_hit_player as *const (),
        aurora_net_hit_x as *const (),
        aurora_net_hit_y as *const (),
        aurora_net_hit_z as *const (),
        // Rebindable input-action layer.
        aurora_input_bind as *const (),
        aurora_input_binding as *const (),
        aurora_input_down as *const (),
        aurora_input_axis as *const (),
        // Raw f32-blob accessors (for the Aurora net sim).
        aurora_f32_load as *const (),
        aurora_f32_store as *const (),
        // Transcendental math builtins.
        aurora_sin as *const (),
        aurora_cos as *const (),
        aurora_tan as *const (),
        aurora_pow as *const (),
        aurora_log as *const (),
        aurora_exp as *const (),
        aurora_atan2 as *const (),
        // 3D rendering extras.
        aurora_r3d_fog as *const (),
        aurora_r3d_sky as *const (),
        aurora_r3d_shadows as *const (),
        aurora_r3d_clear_lights as *const (),
        aurora_r3d_point_light as *const (),
        aurora_r3d_make_sprite as *const (),
        aurora_r3d_draw_billboard as *const (),
        aurora_r3d_debug_line as *const (),
        aurora_r3d_frustum_cull as *const (),
        aurora_r3d_screen_x as *const (),
        aurora_r3d_screen_y as *const (),
        // FPS input.
        aurora_mouse_dx as *const (),
        aurora_mouse_dy as *const (),
        aurora_mouse_scroll as *const (),
        aurora_mouse_button as *const (),
        aurora_grab_mouse as *const (),
        // 3D positional audio.
        aurora_audio_listener as *const (),
        aurora_play_sound_at as *const (),
        // Rich 3D physics queries.
        aurora_phys3d_raycast_full as *const (),
        aurora_phys3d_raycast_ex as *const (),
        aurora_phys3d_hit_x as *const (),
        aurora_phys3d_hit_y as *const (),
        aurora_phys3d_hit_z as *const (),
        aurora_phys3d_hit_nx as *const (),
        aurora_phys3d_hit_ny as *const (),
        aurora_phys3d_hit_nz as *const (),
        aurora_phys3d_hit_body as *const (),
        aurora_phys3d_spherecast as *const (),
        aurora_phys3d_overlap_sphere as *const (),
        aurora_phys3d_apply_force as *const (),
        aurora_phys3d_apply_torque as *const (),
        aurora_phys3d_set_angvel as *const (),
        aurora_phys3d_set_rot as *const (),
        aurora_phys3d_rot_qx as *const (),
        aurora_phys3d_rot_qy as *const (),
        aurora_phys3d_rot_qz as *const (),
        aurora_phys3d_rot_qw as *const (),
        // 3D physics (Rapier 3D).
        aurora_phys3d_init as *const (),
        aurora_phys3d_add_box as *const (),
        aurora_phys3d_add_sphere as *const (),
        aurora_phys3d_add_capsule as *const (),
        aurora_phys3d_add_character as *const (),
        aurora_phys3d_add_trimesh as *const (),
        aurora_phys3d_step as *const (),
        aurora_phys3d_x as *const (),
        aurora_phys3d_y as *const (),
        aurora_phys3d_z as *const (),
        aurora_phys3d_vel_x as *const (),
        aurora_phys3d_vel_y as *const (),
        aurora_phys3d_vel_z as *const (),
        aurora_phys3d_set_vel as *const (),
        aurora_phys3d_set_pos as *const (),
        aurora_phys3d_apply_impulse as *const (),
        aurora_phys3d_move_character as *const (),
        aurora_phys3d_grounded as *const (),
        aurora_phys3d_raycast as *const (),
        // 3D pathfinding (voxel grid + navmesh).
        aurora_nav3d_init as *const (),
        aurora_nav3d_wall as *const (),
        aurora_nav3d_find as *const (),
        aurora_nav3d_x as *const (),
        aurora_nav3d_y as *const (),
        aurora_nav3d_z as *const (),
        aurora_navmesh_build as *const (),
        aurora_navmesh_find as *const (),
        aurora_navmesh_x as *const (),
        aurora_navmesh_y as *const (),
        aurora_navmesh_z as *const (),
        // 3D rendering.
        aurora_r3d_load_model as *const (),
        aurora_r3d_make_box as *const (),
        aurora_r3d_make_box_sized as *const (),
        aurora_r3d_make_box_emissive as *const (),
        aurora_r3d_make_sphere as *const (),
        aurora_r3d_make_plane as *const (),
        aurora_r3d_camera as *const (),
        aurora_r3d_camera_roll as *const (),
        aurora_r3d_light as *const (),
        aurora_r3d_clear as *const (),
        aurora_r3d_begin as *const (),
        aurora_r3d_draw as *const (),
        aurora_r3d_draw_tint as *const (),
        aurora_r3d_anim_play as *const (),
        aurora_r3d_anim_update as *const (),
        aurora_r3d_anim_play_upper as *const (),
        aurora_r3d_anim_stop_upper as *const (),
        aurora_r3d_clip_count as *const (),
        aurora_r3d_present as *const (),
        aurora_oob as *const (),
        aurora_divzero as *const (),
        aurora_fmod as *const (),
        aurora_ffi_dot as *const (),
        aurora_ffi_dotf as *const (),
        aurora_phys_vel_x as *const (),
        aurora_phys_vel_y as *const (),
        aurora_phys_apply_impulse as *const (),
        aurora_phys_apply_force as *const (),
        aurora_phys_set_pos as *const (),
        aurora_phys_raycast as *const (),
        aurora_load_image as *const (),
        aurora_load_font as *const (),
        aurora_draw_text as *const (),
        aurora_play_wav as *const (),
        aurora_phys_init as *const (),
        aurora_phys_add as *const (),
        aurora_phys_step as *const (),
        aurora_phys_x as *const (),
        aurora_phys_y as *const (),
        aurora_phys_set_vel as *const (),
        aurora_nav_init as *const (),
        aurora_nav_wall as *const (),
        aurora_nav_find as *const (),
        aurora_nav_x as *const (),
        aurora_nav_y as *const (),
        aurora_par_for as *const (),
        aurora_run_parallel as *const (),
        aurora_gpu_compute as *const (),
        aurora_net_bind as *const (),
        aurora_net_connect as *const (),
        aurora_net_send as *const (),
        aurora_net_recv as *const (),
        aurora_frame_reset as *const (),
        aurora_load_ppm as *const (),
        aurora_scene_save as *const (),
        aurora_scene_load as *const (),
        aurora_prof_enter as *const (),
        aurora_prof_exit as *const (),
        aurora_str_concat as *const (),
        aurora_str_eq as *const (),
        aurora_str_char_at as *const (),
        aurora_str_substr as *const (),
        aurora_str_starts_with as *const (),
        aurora_int_to_str as *const (),
        aurora_float_to_str as *const (),
        aurora_play_note as *const (),
        aurora_play_sound as *const (),
        aurora_play_noise as *const (),
        aurora_surface_w as *const (),
        aurora_surface_h as *const (),
        aurora_r3d_speedlines as *const (),
        aurora_r3d_damage as *const (),
        aurora_draw_int as *const (),
        aurora_audio_volume as *const (),
        aurora_audio_stop as *const (),
        aurora_gpu_render as *const (),
        aurora_window_open as *const (),
        aurora_window_present as *const (),
        aurora_key_down as *const (),
        aurora_input_char as *const (),
        aurora_window_fullscreen as *const (),
        aurora_mouse_x as *const (),
        aurora_mouse_y as *const (),
        aurora_mouse_down as *const (),
        aurora_dbg_enter as *const (),
        aurora_dbg_leave as *const (),
        aurora_dbg_stmt as *const (),
        aurora_dbg_var as *const (),
        aurora_dbg_var_f64 as *const (),
        aurora_print_i64 as *const (),
        aurora_print_f64 as *const (),
        aurora_print_str as *const (),
        aurora_print_nl as *const (),
        aurora_runtime_flush as *const (),
        aurora_frame_dt as *const (),
        aurora_sleep_ms as *const (),
        aurora_framebuffer as *const (),
        aurora_clear as *const (),
        aurora_pixel as *const (),
        aurora_triangle as *const (),
        aurora_fb_get as *const (),
        aurora_save_ppm as *const (),
        aurora_spawn_entity as *const (),
        aurora_despawn as *const (),
        aurora_store_component as *const (),
        aurora_get_component as *const (),
        aurora_query_begin as *const (),
        aurora_query_entity as *const (),
        aurora_entity_count as *const (),
    ];
    std::hint::black_box(fns.iter().map(|p| *p as usize).sum())
}

#[cfg(test)]
mod arena_tests {
    use super::*;

    #[test]
    fn floats_display_with_trailing_decimal() {
        // Whole-valued floats keep a `.0` so they read as floats, not ints.
        assert_eq!(fmt_f64(7.0), "7.0");
        assert_eq!(fmt_f64(4.0), "4.0");
        assert_eq!(fmt_f64(0.0), "0.0");
        assert_eq!(fmt_f64(-3.0), "-3.0");
        // Fractional values are unchanged.
        assert_eq!(fmt_f64(3.25), "3.25");
        assert_eq!(fmt_f64(-1.5), "-1.5");
        // Non-finite values are left as Rust renders them (no bogus `.0`).
        assert_eq!(fmt_f64(f64::INFINITY), "inf");
        assert_eq!(fmt_f64(f64::NAN), "NaN");
    }

    #[test]
    fn frame_arena_allocates_then_resets() {
        aurora_frame_reset();
        let base = frame_arena_used();
        let p = frame_alloc(b"hello");
        let used = unsafe { std::slice::from_raw_parts(p, 5) };
        assert_eq!(used, b"hello");
        assert!(frame_arena_used() > base, "allocation advances the arena");
        aurora_frame_reset();
        assert_eq!(frame_arena_used(), 0, "reset frees the whole frame");
    }

    #[test]
    fn arena_pointers_stay_valid_across_many_allocs() {
        aurora_frame_reset();
        let first = frame_alloc(b"abcd");
        // Force growth past a chunk so reallocation would move a naive Vec.
        for _ in 0..300_000 {
            let _ = frame_alloc(b"xxxxxxxx");
        }
        // The first pointer must still hold its bytes (chunks never move).
        let bytes = unsafe { std::slice::from_raw_parts(first, 4) };
        assert_eq!(bytes, b"abcd");
        aurora_frame_reset();
    }
}
