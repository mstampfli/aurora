//! Aurora's native runtime - the host functions compiled Aurora code calls.
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
//!
//! # Raw pointers and `unsafe`
//!
//! Aurora passes a `str` as its two `[data, len]` slots, an array as its data
//! pointer plus a length the COMPILER derives from the array's type, and a
//! closure as an `[fn_ptr, env_ptr]` pair - so every host function taking a
//! pointer reads or writes through it. Nothing the function itself can check
//! makes that sound: the pointer's validity is the caller's to guarantee.
//!
//! Every such function is therefore `pub unsafe extern "C" fn` with a `# Safety`
//! section naming exactly what it requires. `unsafe` is a Rust-level property
//! only: the emitted symbol, its C ABI, and both consumers above are unchanged.
//! What it buys is that safe Rust can no longer call one of these with a pointer
//! it invented. A host function that takes no pointer stays safe.
//!
//! **Place in the graph.** Sits on `abi`, `slot`, `gfx`, `audio`, `window`, `render3d`, `gpu`, `net`. Called by compiled code.
//!
//! **Never.** Never parses or type-checks. Every entry point here is an `extern "C"` fn whose row lives in `aurora-abi`.

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

// Determinism + data: seeded RNG, fixed dt, file I/O, JSON.
mod data;
pub use data::*;

// Heightmap terrain: one heightfield behind the renderer, the collider, and the
// height query.
mod terrain;
pub use terrain::*;

// The value stack: per-thread bump arena for aggregates too large to sit in a
// machine stack frame.
mod font;
mod vstack;
pub use vstack::*;

// --- printing --------------------------------------------------------------

#[no_mangle]
pub extern "C" fn aurora_print_i64(n: i64) {
    print!("{n}");
}
/// Format an `f64` for display. Whole-valued finite floats get a trailing `.0`
/// (`7.0` not `7`) so floats are visually distinct from ints - Aurora is a
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
/// # Safety
/// `ptr` must point to `len` initialized bytes.
#[no_mangle]
pub unsafe extern "C" fn aurora_print_str(ptr: *const u8, len: i64) {
    let s = unsafe { std::slice::from_raw_parts(ptr, len.max(0) as usize) };
    print!("{}", String::from_utf8_lossy(s));
}
#[no_mangle]
pub extern "C" fn aurora_print_nl() {
    println!();
}

/// Flush buffered stdout - called from the AOT entry shim before exit, since the
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
    /// This frame's delta, once measured.
    ///
    /// `frame_dt` used to be a DESTRUCTIVE read: every call reset the frame
    /// timer, so the second caller in a frame got roughly zero. That matters
    /// because `run_systems` calls it too - it is how the fixed stage learns how
    /// much time it owes - so a game that did the ordinary thing
    ///
    /// ```text
    /// let dt = frame_dt()
    /// ...
    /// run_systems()
    /// ```
    ///
    /// starved its own simulation. Played, that is a boss that takes minutes to
    /// throw its first attack, stamina that never comes back, and an attack
    /// frozen on its last frame, while every headless test passes - because a
    /// test pins the step with `set_fixed_dt`, which makes `frame_dt` a constant
    /// and hides the whole thing.
    ///
    /// Measured once per frame now and reused until the frame is presented,
    /// which is what every engine means by delta time.
    static FRAME_DT: std::cell::Cell<Option<f64>> = const { std::cell::Cell::new(None) };
}

/// Real elapsed seconds since the previous call (0.016 on the first call),
/// clamped to 0.1 so a stall can't make the game lurch or spiral. Lets the game
/// loop run frame-rate-independent instead of assuming a fixed step.
///
/// Under a fixed step (`set_fixed_dt` builtin or `AURORA_FIXED_DT` env var)
/// this returns the scripted dt and advances the virtual clock instead - the
/// determinism hook replays and headless runs rely on.
#[no_mangle]
pub extern "C" fn aurora_frame_dt() -> f64 {
    // Already answered this frame: the same answer, not a fresh near-zero one
    // and not a second advance of the clock. Checked BEFORE the fixed-step
    // branch, because "how long was this frame" has one answer per frame however
    // it is measured.
    //
    // The fixed path used to advance virtual time on every call while the
    // wall-clock path cached, so a frame that asked the time twice - and the
    // game's does, once for the frame and once for the camera - ran the virtual
    // clock at twice the rate it claimed. Everything hanging off that clock
    // (offline audio capture, telemetry timestamps) was stamped at a time that
    // depended on how many callers happened to ask.
    if let Some(dt) = FRAME_DT.with(|c| c.get()) {
        return dt;
    }
    let fixed = data::fixed_dt_override();
    if fixed > 0.0 {
        data::advance_virtual_time(fixed);
        FRAME_DT.with(|c| c.set(Some(fixed)));
        return fixed;
    }
    let dt = LAST_FRAME.with(|c| {
        let now = std::time::Instant::now();
        let dt = match c.borrow_mut().replace(now) {
            Some(prev) => now.duration_since(prev).as_secs_f64(),
            None => 1.0 / 60.0,
        };
        dt.clamp(0.0001, 0.1)
    });
    FRAME_DT.with(|c| c.set(Some(dt)));
    dt
}

/// Forget this frame's delta so the next one is measured afresh. Called when a
/// frame is presented, which is what ends a frame.
pub(crate) fn end_frame_dt() {
    FRAME_DT.with(|c| c.set(None));
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
/// 8-byte slots, so they pass straight through as `const double*` - this is what
/// lets `@extern` bind real C/Rust functions that take buffers and vectors.
///
/// # Safety
/// `a` and `b` must each point to `n` initialized `f64`s.
#[no_mangle]
pub unsafe extern "C" fn aurora_ffi_dot(a: *const f64, b: *const f64, n: i64) -> f64 {
    let n = n.max(0) as usize;
    let (a, b) = unsafe {
        (
            std::slice::from_raw_parts(a, n),
            std::slice::from_raw_parts(b, n),
        )
    };
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// `f32` variant - reads two C-packed `float` buffers. Tests that Aurora `f32`
/// aggregates are marshaled to C's 4-byte-packed layout over FFI.
///
/// # Safety
/// `a` and `b` must each point to `n` initialized `f32`s.
#[no_mangle]
pub unsafe extern "C" fn aurora_ffi_dotf(a: *const f32, b: *const f32, n: i64) -> f32 {
    let n = n.max(0) as usize;
    let (a, b) = unsafe {
        (
            std::slice::from_raw_parts(a, n),
            std::slice::from_raw_parts(b, n),
        )
    };
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Stop the program with a clear message, the way every runtime failure does.
///
/// NOT `panic!`. Every builtin is `extern "C"`, which Rust treats as nounwind:
/// a panic crossing that boundary aborts the process with
/// STATUS_STACK_BUFFER_OVERRUN and no message at all. Found the hard way -
/// making a stale physics handle "loud" with `panic!` turned a silent no-op
/// into a silent crash, which is strictly worse, and `should_panic` could not
/// catch it either.
///
/// So there is one way to die, and it is this: flush what the program already
/// printed (stdout is buffered and stderr is not, so the message would
/// otherwise arrive before the output it refers to), say what happened, and
/// exit 101 - the code `assert` has always used.
pub(crate) fn fatal(what: std::fmt::Arguments<'_>) -> ! {
    use std::io::Write;
    let _ = std::io::stdout().flush();
    eprintln!("panic: {what}");
    std::process::exit(101);
}

/// Report an out-of-bounds array access with a clear message and abort. Called
/// by bounds-check code in place of a raw trap, so the failure reads as a panic
/// rather than a cryptic "illegal instruction".
#[no_mangle]
pub extern "C" fn aurora_oob(idx: i64, len: i64) {
    fatal(format_args!(
        "array index {idx} out of bounds (length {len})"
    ));
}

/// `assert(cond)`: abort unless `cond` holds, matching the interpreter's
/// "assertion failed" and the documented "abort if `cond` is 0".
#[no_mangle]
pub extern "C" fn aurora_assert(cond: i64) {
    if cond != 0 {
        return;
    }
    fatal(format_args!("assertion failed"));
}

/// Clean panic for integer division/remainder by zero, in place of a raw CPU
/// trap (SIGFPE / "illegal instruction"), matching the interpreter's behavior.
#[no_mangle]
pub extern "C" fn aurora_divzero() {
    fatal(format_args!("integer division or remainder by zero"));
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
        *fb.borrow_mut() = Some(aurora_gfx::Framebuffer::new(
            w.max(0) as u32,
            h.max(0) as u32,
        ))
    });
}
fn color(r: i64, g: i64, b: i64) -> aurora_gfx::Color {
    let c = |v: i64| v.clamp(0, 255) as u8;
    aurora_gfx::Color::rgb(c(r), c(g), c(b))
}
/// Clear the framebuffer to a colour, TRANSPARENT.
///
/// Alpha 0 rather than opaque, because the dominant use of `clear` is erasing a
/// HUD that composites over a 3D scene, and a HUD that cleared to an opaque
/// colour would hide the game. The 2D path presents the framebuffer as the
/// image and forces alpha opaque in its blit, so a 2D game that clears to a
/// background colour is unaffected.
#[no_mangle]
pub extern "C" fn aurora_clear(r: i64, g: i64, b: i64) {
    let c = color(r, g, b);
    FB.with(|fb| {
        if let Some(f) = fb.borrow_mut().as_mut() {
            f.erase(c);
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
/// A pixel with explicit coverage: 0 invisible, 255 opaque.
///
/// The alpha-taking counterpart to `pixel`, and the primitive every translucent
/// 2D thing is built from - `fill_rect_a` in the prelude is a loop over it.
/// Needed because a HUD plate that is readable over a bright scene has to be
/// dark AND see-through, which a colour key cannot express.
#[no_mangle]
pub extern "C" fn aurora_pixel_alpha(x: i64, y: i64, r: i64, g: i64, b: i64, a: i64) {
    FB.with(|fb| {
        if let Some(f) = fb.borrow_mut().as_mut() {
            let c = color(r, g, b);
            f.set(
                x as i32,
                y as i32,
                aurora_gfx::Color::rgba(c.r, c.g, c.b, a.clamp(0, 255) as u8),
            );
        }
    });
}
/// Fill an axis-aligned rectangle with an explicit alpha.
///
/// A builtin rather than a loop over `pixel_alpha` in the prelude because this
/// is a per-frame HUD call: a full-width dialogue plate is tens of thousands of
/// pixels, and that is a span fill here versus tens of thousands of native
/// calls there. Clipped to the framebuffer, so a plate wider than the screen is
/// not an error - `w`/`h` of 0 or less draw nothing.
#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub extern "C" fn aurora_fill_rect_alpha(
    x: i64,
    y: i64,
    w: i64,
    h: i64,
    r: i64,
    g: i64,
    b: i64,
    a: i64,
) {
    if w <= 0 || h <= 0 {
        return;
    }
    let c = color(r, g, b);
    let c = aurora_gfx::Color::rgba(c.r, c.g, c.b, a.clamp(0, 255) as u8);
    FB.with(|fb| {
        if let Some(f) = fb.borrow_mut().as_mut() {
            f.fill_rect(x, y, w, h, c);
        }
    });
}
#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub extern "C" fn aurora_triangle(
    x0: i64,
    y0: i64,
    x1: i64,
    y1: i64,
    x2: i64,
    y2: i64,
    r: i64,
    g: i64,
    b: i64,
) {
    FB.with(|fb| {
        if let Some(f) = fb.borrow_mut().as_mut() {
            let c = color(r, g, b);
            f.triangle(
                [
                    [x0 as f32, y0 as f32],
                    [x1 as f32, y1 as f32],
                    [x2 as f32, y2 as f32],
                ],
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
/// # Safety
/// `ptr` must point to `len` initialized bytes.
#[no_mangle]
pub unsafe extern "C" fn aurora_save_ppm(ptr: *const u8, len: i64) {
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

/// Save the 2D framebuffer as a PNG (the format vision tooling reads).
/// Creates parent directories. Backs the `save_png` builtin.
///
/// `fb_width()` / `fb_height()`: the framebuffer's size, or 0 if there is none.
///
/// A HUD cannot lay itself out without these. Poly Souls' dialogue banner was
/// positioned for a 640x360 surface and sat across the middle of a 960x540 one,
/// because the only size a program could know was the one it had typed into its
/// own constants - which is right until anything opens a different window.
///
/// 0 rather than a guess when no framebuffer exists, so `fb_width() / 2` is a
/// harmless zero rather than a plausible wrong number.
#[no_mangle]
pub extern "C" fn aurora_fb_width() -> i64 {
    FB.with(|fb| fb.borrow().as_ref().map(|f| f.width() as i64).unwrap_or(0))
}

#[no_mangle]
pub extern "C" fn aurora_fb_height() -> i64 {
    FB.with(|fb| fb.borrow().as_ref().map(|f| f.height() as i64).unwrap_or(0))
}

/// # Safety
/// `ptr` must point to `len` initialized bytes.
#[no_mangle]
pub unsafe extern "C" fn aurora_save_png(ptr: *const u8, len: i64) {
    let path = {
        let s = unsafe { std::slice::from_raw_parts(ptr, len.max(0) as usize) };
        String::from_utf8_lossy(s).into_owned()
    };
    FB.with(|fb| {
        if let Some(f) = fb.borrow().as_ref() {
            let (w, h) = (f.width(), f.height());
            let mut rgba = Vec::with_capacity((w * h * 4) as usize);
            for y in 0..h {
                for x in 0..w {
                    let c = f.get(x, y);
                    rgba.extend_from_slice(&[c.r, c.g, c.b, 255]);
                }
            }
            if let Some(parent) = std::path::Path::new(&path).parent() {
                if !parent.as_os_str().is_empty() {
                    let _ = std::fs::create_dir_all(parent);
                }
            }
            if let Err(e) = image::save_buffer(&path, &rgba, w, h, image::ExtendedColorType::Rgba8)
            {
                eprintln!("aurora: save_png {path}: {e}");
            }
        }
    });
}

// --- region arenas ----------------------------------------------------------
//
// A real runtime backing for the language's `#frame`/`#level`/`#perm` regions:
// each is a chunked bump allocator. Dynamic allocations (string concat, int/
// float formatting) come from the `#frame` arena, and `frame_reset()` frees the
// whole frame's allocations at once (O(1)) - so memory is arena-managed and
// reclaimed at frame boundaries instead of leaking. The region *checker*
// (`aurora-check` section 8.2) statically prevents storing shorter-lived
// (frame) data where longer-lived data is expected, which is what makes the
// bulk reset safe.

const CHUNK: usize = 1 << 20; // 1 MiB per chunk

struct Arena {
    chunks: Vec<Vec<u8>>,
    cur: usize,
    used: usize,
}
impl Arena {
    fn new() -> Arena {
        Arena {
            chunks: vec![vec![0u8; CHUNK]],
            cur: 0,
            used: 0,
        }
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
pub(crate) unsafe fn write_str(out: *mut i64, bytes: Vec<u8>) {
    let ptr = frame_alloc(&bytes) as i64;
    *out = ptr;
    *out.add(1) = bytes.len() as i64;
}

/// # Safety
/// `ap` must point to `al` initialized bytes and `bp` to `bl`. `out` must
/// be valid for writes of two `i64`s.
#[no_mangle]
pub unsafe extern "C" fn aurora_str_concat(
    out: *mut i64,
    ap: *const u8,
    al: i64,
    bp: *const u8,
    bl: i64,
) {
    let a = unsafe { std::slice::from_raw_parts(ap, al.max(0) as usize) };
    let b = unsafe { std::slice::from_raw_parts(bp, bl.max(0) as usize) };
    let mut v = Vec::with_capacity(a.len() + b.len());
    v.extend_from_slice(a);
    v.extend_from_slice(b);
    unsafe { write_str(out, v) };
}

/// # Safety
/// `ap` must point to `al` initialized bytes and `bp` to `bl`.
#[no_mangle]
pub unsafe extern "C" fn aurora_str_eq(ap: *const u8, al: i64, bp: *const u8, bl: i64) -> i64 {
    let a = unsafe { std::slice::from_raw_parts(ap, al.max(0) as usize) };
    let b = unsafe { std::slice::from_raw_parts(bp, bl.max(0) as usize) };
    (a == b) as i64
}

/// # Safety
/// `out` must be valid for writes of two `i64`s.
#[no_mangle]
pub unsafe extern "C" fn aurora_int_to_str(out: *mut i64, n: i64) {
    unsafe { write_str(out, n.to_string().into_bytes()) };
}

/// Byte at index `i` of the string (0..len), or -1 if out of range.
///
/// # Safety
/// `ptr` must point to `len` initialized bytes.
#[no_mangle]
pub unsafe extern "C" fn aurora_str_char_at(ptr: *const u8, len: i64, i: i64) -> i64 {
    if i < 0 || i >= len {
        return -1;
    }
    let s = unsafe { std::slice::from_raw_parts(ptr, len.max(0) as usize) };
    s[i as usize] as i64
}

/// Substring `[start, start+n)` (clamped) written into `out` as a new string.
///
/// # Safety
/// `ptr` must point to `len` initialized bytes. `out` must be valid for
/// writes of two `i64`s.
#[no_mangle]
pub unsafe extern "C" fn aurora_str_substr(
    out: *mut i64,
    ptr: *const u8,
    len: i64,
    start: i64,
    n: i64,
) {
    let s = unsafe { std::slice::from_raw_parts(ptr, len.max(0) as usize) };
    let start = start.clamp(0, len) as usize;
    let end = (start + n.max(0) as usize).min(len.max(0) as usize);
    unsafe { write_str(out, s[start..end].to_vec()) };
}

/// 1 if `hay` starts with `needle`, else 0.
///
/// # Safety
/// `hp` must point to `hl` initialized bytes and `np` to `nl`.
#[no_mangle]
pub unsafe extern "C" fn aurora_str_starts_with(
    hp: *const u8,
    hl: i64,
    np: *const u8,
    nl: i64,
) -> i64 {
    let hay = unsafe { std::slice::from_raw_parts(hp, hl.max(0) as usize) };
    let needle = unsafe { std::slice::from_raw_parts(np, nl.max(0) as usize) };
    hay.starts_with(needle) as i64
}

/// # Safety
/// `out` must be valid for writes of two `i64`s.
#[no_mangle]
pub unsafe extern "C" fn aurora_float_to_str(out: *mut i64, x: f64) {
    unsafe { write_str(out, fmt_f64(x).into_bytes()) };
}

// --- asset pipeline --------------------------------------------------------

/// Load a binary PPM image at `path` into the framebuffer (resizing it).
/// Returns 1 on success, 0 on failure. Backs the `load_ppm` builtin.
///
/// # Safety
/// `ptr` must point to `len` initialized bytes.
#[no_mangle]
pub unsafe extern "C" fn aurora_load_ppm(ptr: *const u8, len: i64) -> i64 {
    let path = {
        let s = unsafe { std::slice::from_raw_parts(ptr, len.max(0) as usize) };
        String::from_utf8_lossy(s).into_owned()
    };
    match std::fs::read(&path)
        .ok()
        .and_then(|b| aurora_gfx::Framebuffer::from_ppm(&b))
    {
        Some(fb) => {
            FB.with(|f| *f.borrow_mut() = Some(fb));
            1
        }
        None => 0,
    }
}

/// Load a PNG/JPEG image at `path` into the framebuffer (resizing it to the
/// image), decoded to RGBA via the `image` crate. Returns 1 on success, 0 on
/// failure. Backs the `load_image` builtin - the asset pipeline beyond PPM.
///
/// REPLACES THE FRAMEBUFFER, and the framebuffer is also the HUD layer that
/// `r3d_capture` composites over the 3D frame. Loading a full-screen image and
/// then capturing therefore photographs the loaded image, not the scene. To read
/// an image back in order to MEASURE it, use `image_open` and friends below,
/// which hold their pixels off to one side and touch no drawing state.
///
/// # Safety
/// `ptr` must point to `len` initialized bytes.
#[no_mangle]
pub unsafe extern "C" fn aurora_load_image(ptr: *const u8, len: i64) -> i64 {
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

// --- images as values, for MEASURING what was rendered ----------------------
//
// The framebuffer is a place you draw. These are images you ask questions about.
// Keeping them apart is the whole point: `load_image` is the only way to read a
// PNG back today, and it does so by REPLACING the framebuffer - which is also
// the HUD layer `r3d_capture` composites. So a program that photographed a frame
// and then measured it destroyed the layer the next frame drew through, and the
// next capture came out as the image that had been loaded to measure. Silently,
// with `r3d_capture` still answering success.
//
// A renderer that cannot be asked what it drew can only be checked by a human
// looking at a file, which is why capture scripts tend to assert nothing. These
// are the primitives that let a check make a claim about pixels.
//
// Handles are indices into a push-only table, like sounds. Every accessor
// answers a NEGATIVE number for a bad handle or an out-of-range region rather
// than a plausible one - a black pixel and "no such image" must not be the same
// answer, or a check reads zero and calls it a dark frame.

thread_local! {
    static IMAGES: RefCell<Vec<image::RgbaImage>> = const { RefCell::new(Vec::new()) };
}

/// Read the pixels of `img` for `f`, or `None` if the handle is not live.
fn with_image<T>(img: i64, f: impl FnOnce(&image::RgbaImage) -> T) -> Option<T> {
    IMAGES.with(|s| {
        let v = s.borrow();
        let i = usize::try_from(img).ok()?;
        v.get(i).filter(|im| im.width() > 0).map(f)
    })
}

/// The region `(x, y, w, h)` as `u32`s if it lies wholly inside `im`, else
/// `None`. Rejects rather than clamps: a check that asked for the wrong
/// rectangle should fail, not quietly measure a different one.
fn image_region(
    im: &image::RgbaImage,
    x: i64,
    y: i64,
    w: i64,
    h: i64,
) -> Option<(u32, u32, u32, u32)> {
    if w <= 0 || h <= 0 || x < 0 || y < 0 {
        return None;
    }
    let (x1, y1) = (x.checked_add(w)?, y.checked_add(h)?);
    if x1 > im.width() as i64 || y1 > im.height() as i64 {
        return None;
    }
    Some((x as u32, y as u32, w as u32, h as u32))
}

/// Rec.709 luminance of one RGBA sample, 0..255. The one place this project
/// converts colour to brightness, so two checks cannot disagree about what
/// "darker" means.
fn luma(p: &image::Rgba<u8>) -> f64 {
    0.2126 * p.0[0] as f64 + 0.7152 * p.0[1] as f64 + 0.0722 * p.0[2] as f64
}

/// Open a PNG/JPEG at `path` for measurement. Returns a handle, or -1.
///
/// Touches no drawing state, unlike `load_image`.
///
/// # Safety
/// `ptr` must point to `len` initialized bytes.
#[no_mangle]
pub unsafe extern "C" fn aurora_image_open(ptr: *const u8, len: i64) -> i64 {
    let path = {
        let s = unsafe { std::slice::from_raw_parts(ptr, len.max(0) as usize) };
        String::from_utf8_lossy(s).into_owned()
    };
    match image::open(&path) {
        Ok(img) => IMAGES.with(|s| {
            let mut v = s.borrow_mut();
            v.push(img.to_rgba8());
            (v.len() - 1) as i64
        }),
        Err(_) => -1,
    }
}

/// Width of `img` in pixels, or -1 for a handle that is not live.
#[no_mangle]
pub extern "C" fn aurora_image_width(img: i64) -> i64 {
    with_image(img, |im| im.width() as i64).unwrap_or(-1)
}

/// Height of `img` in pixels, or -1 for a handle that is not live.
#[no_mangle]
pub extern "C" fn aurora_image_height(img: i64) -> i64 {
    with_image(img, |im| im.height() as i64).unwrap_or(-1)
}

/// The pixel at `(x, y)` packed `0xRRGGBB`, or -1 if the handle is not live or
/// the point is outside the image.
///
/// Packed the same way `fb_get` packs, because two conventions for a colour is
/// one more than a program can keep straight.
#[no_mangle]
pub extern "C" fn aurora_image_pixel(img: i64, x: i64, y: i64) -> i64 {
    with_image(img, |im| {
        if x < 0 || y < 0 || x >= im.width() as i64 || y >= im.height() as i64 {
            return -1;
        }
        let p = im.get_pixel(x as u32, y as u32);
        ((p.0[0] as i64) << 16) | ((p.0[1] as i64) << 8) | p.0[2] as i64
    })
    .unwrap_or(-1)
}

/// Mean Rec.709 luminance over a region of `img`, 0..255, or -1 if the handle is
/// not live or the region is not wholly inside the image.
///
/// One call rather than a loop of `image_pixel` over hundreds of thousands of
/// pixels: the question "how bright is this part of the frame" is asked by every
/// visual check there is, and answering it a pixel at a time across the FFI
/// boundary would make the checks too slow to keep.
#[no_mangle]
pub extern "C" fn aurora_image_mean_luma(img: i64, x: i64, y: i64, w: i64, h: i64) -> f64 {
    with_image(img, |im| {
        let Some((x, y, w, h)) = image_region(im, x, y, w, h) else {
            return -1.0;
        };
        let mut total = 0.0;
        for py in y..y + h {
            for px in x..x + w {
                total += luma(im.get_pixel(px, py));
            }
        }
        total / (w as f64 * h as f64)
    })
    .unwrap_or(-1.0)
}

/// Mean absolute per-channel difference between the same region of two images,
/// 0..255, or -1 if either handle is not live or the region does not fit BOTH.
///
/// The direct answer to "are these two frames the same picture", which is the
/// question a capture script has to be able to ask before it can claim anything
/// about what changed between them.
#[no_mangle]
pub extern "C" fn aurora_image_diff(a: i64, b: i64, x: i64, y: i64, w: i64, h: i64) -> f64 {
    let Some(pixels) = with_image(a, |ia| {
        with_image(b, |ib| {
            let ra = image_region(ia, x, y, w, h);
            let rb = image_region(ib, x, y, w, h);
            match (ra, rb) {
                (Some((x, y, w, h)), Some(_)) => {
                    let mut total = 0.0;
                    for py in y..y + h {
                        for px in x..x + w {
                            let pa = ia.get_pixel(px, py).0;
                            let pb = ib.get_pixel(px, py).0;
                            for c in 0..3 {
                                total += (pa[c] as f64 - pb[c] as f64).abs();
                            }
                        }
                    }
                    total / (w as f64 * h as f64 * 3.0)
                }
                _ => -1.0,
            }
        })
    }) else {
        return -1.0;
    };
    pixels.unwrap_or(-1.0)
}

/// Release `img`. Answers 1 if it was live, 0 if it was not.
///
/// The handle is not reused, so a stale one keeps reading as dead rather than
/// silently naming somebody else's image.
#[no_mangle]
pub extern "C" fn aurora_image_free(img: i64) -> i64 {
    IMAGES.with(|s| {
        let mut v = s.borrow_mut();
        match usize::try_from(img).ok().and_then(|i| v.get_mut(i)) {
            Some(slot) if slot.width() > 0 => {
                *slot = image::RgbaImage::new(0, 0);
                1
            }
            _ => 0,
        }
    })
}

// --- text rendering (TrueType via fontdue) ----------------------------------

thread_local! {
    static FONT: RefCell<Option<fontdue::Font>> = const { RefCell::new(None) };
}

/// Load a TrueType/OpenType font from `path` for `draw_text`. Returns 1/0.
///
/// # Safety
/// `ptr` must point to `len` initialized bytes.
#[no_mangle]
pub unsafe extern "C" fn aurora_load_font(ptr: *const u8, len: i64) -> i64 {
    let path = {
        let s = unsafe { std::slice::from_raw_parts(ptr, len.max(0) as usize) };
        String::from_utf8_lossy(s).into_owned()
    };
    let Ok(bytes) = std::fs::read(&path) else {
        return 0;
    };
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
/// Draw with the built-in 5x7 font, clipped to the framebuffer.
///
/// Opaque rather than blended: the glyphs are 1-bit, so there is no coverage to
/// blend with, and a HUD wants its label to survive whatever is behind it.
fn render_text_builtin(x: i64, y: i64, text: &str, px: i64, cr: u8, cg: u8, cb: u8) {
    let scale = font::scale_for(px);
    FB.with(|fb| {
        let mut fb = fb.borrow_mut();
        let Some(fb) = fb.as_mut() else { return };
        let (w, h) = (fb.width() as i64, fb.height() as i64);
        let colour = aurora_gfx::Color::rgb(cr, cg, cb);
        font::blit(x, y, text, scale, |px_x, px_y| {
            if px_x >= 0 && px_y >= 0 && px_x < w && px_y < h {
                fb.set(px_x as i32, px_y as i32, colour);
            }
        });
    });
}

fn render_text(x: i64, y: i64, text: &str, px: i64, color: i64) {
    let px = px.max(1) as f32;
    let (cr, cg, cb) = (
        ((color >> 16) & 255) as u8,
        ((color >> 8) & 255) as u8,
        (color & 255) as u8,
    );
    let loaded = FONT.with(|f| f.borrow().is_some());
    if !loaded {
        // No TTF: draw with the font that ships in the binary rather than
        // returning silently, which is what this did and which makes "no font"
        // and "text drawn off-screen" the same picture.
        render_text_builtin(x, y, text, px as i64, cr, cg, cb);
        return;
    }
    FONT.with(|font| {
        let font = font.borrow();
        let Some(font) = font.as_ref() else { return };
        FB.with(|fb| {
            let mut fb = fb.borrow_mut();
            let Some(fb) = fb.as_mut() else { return };
            let (w, h) = (fb.width() as i32, fb.height() as i32);
            let baseline = y + px as i64; // `y` is the top; baseline ~= y + size
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
                        let blend =
                            |b: u8, f: u8| ((b as u32 * (255 - cov) + f as u32 * cov) / 255) as u8;
                        let out = aurora_gfx::Color::rgb(
                            blend(bg.r, cr),
                            blend(bg.g, cg),
                            blend(bg.b, cb),
                        );
                        fb.set(sx, sy, out);
                    }
                }
                pen += m.advance_width as i64;
            }
        });
    });
}

/// # Safety
/// `ptr` must point to `len` initialized bytes.
#[no_mangle]
pub unsafe extern "C" fn aurora_draw_text(
    x: i64,
    y: i64,
    ptr: *const u8,
    len: i64,
    px: i64,
    color: i64,
) {
    let text = {
        let s = unsafe { std::slice::from_raw_parts(ptr, len.max(0) as usize) };
        String::from_utf8_lossy(s).into_owned()
    };
    render_text(x, y, &text, px, color);
}

/// Pixel width of `text` at size `px`, so a game can centre or right-align it.
///
/// Answers for the BUILT-IN font when no TTF is loaded, which is the only
/// version that is safe to divide by: it used to return 0, so a centred label
/// on a program with no font was centred at zero width - drawn hard against the
/// left edge, in the one case where the text itself was also invisible.
///
/// # Safety
/// `ptr` must point to `len` initialized bytes.
#[no_mangle]
pub unsafe extern "C" fn aurora_text_width(ptr: *const u8, len: i64, px: i64) -> i64 {
    let text = {
        let s = unsafe { std::slice::from_raw_parts(ptr, len.max(0) as usize) };
        String::from_utf8_lossy(s).into_owned()
    };
    let loaded = FONT.with(|f| f.borrow().is_some());
    if !loaded {
        return font::width(&text, font::scale_for(px.max(1)));
    }
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
// continuous collision - far beyond the hand-rolled AABB resolver in the stdlib.
// Positions are the body centre, in whatever units the program uses (pixels,
// say).
//
// A body handle is a generation-tagged `aurora_slot::Key` packed into an i64,
// not an index: `phys_remove` bumps its slot's generation, so the handle is
// refused from then on instead of being inherited by the next `phys_add` that
// lands in the freed slot. Same reasoning, same primitive, as `phys3d`.

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
    /// Aurora-visible bodies, keyed by the handle the program holds.
    registry: aurora_slot::SlotMap<rapier2d::prelude::RigidBodyHandle>,
}
thread_local! {
    static PHYS_OWN: RefCell<Option<Phys>> = const { RefCell::new(None) };
}

/// The 2D physics world, routed to the batch owner's while this thread is a
/// worker. See `ROUTED_CELLS`.
struct PhysSlot;

impl PhysSlot {
    fn with<R>(&self, f: impl FnOnce(&RefCell<Option<Phys>>) -> R) -> R {
        let batch = par_batch();
        if batch.is_null() {
            return PHYS_OWN.with(f);
        }
        // SAFETY: as for the world - the owner is blocked in `thread::scope`
        // until this worker joins, so its cell is alive and untouched.
        unsafe {
            with_par_cell(
                batch,
                par_cell(batch, CELL_PHYS) as *const RefCell<Option<Phys>>,
                f,
            )
        }
    }
}

const PHYS: PhysSlot = PhysSlot;

/// The `i64` an Aurora program holds for a 2D body.
type Body2 = aurora_slot::Key<rapier2d::prelude::RigidBodyHandle>;

/// The Rapier body `h` names, or `None` for a stale, never-issued, or negative
/// handle. Every 2D accessor goes through this.
fn rb2_of(p: &Phys, h: i64) -> Option<rapier2d::prelude::RigidBodyHandle> {
    p.registry.get(Body2::from_i64(h)?).copied()
}

/// Create (or reset) the physics world with gravity (gx, gy).
///
/// Handles from the previous world are invalidated rather than carried over:
/// the registry is cleared, which bumps every live slot's generation. See
/// `aurora_phys3d_init` for why a fresh registry would be wrong.
#[no_mangle]
pub extern "C" fn aurora_phys_init(gx: f64, gy: f64) {
    use rapier2d::prelude::*;
    PHYS.with(|x| {
        let mut cell = x.borrow_mut();
        let mut registry = cell.take().map(|p| p.registry).unwrap_or_default();
        registry.clear();
        *cell = Some(Phys {
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
            registry,
        });
    });
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
            RigidBodyBuilder::dynamic()
                .translation(vector![x as Real, y as Real])
                .build()
        } else {
            RigidBodyBuilder::fixed()
                .translation(vector![x as Real, y as Real])
                .build()
        };
        let h = p.bodies.insert(rb);
        let col = ColliderBuilder::cuboid(hw as Real, hh as Real).build();
        p.colliders.insert_with_parent(col, h, &mut p.bodies);
        p.registry.insert(h).to_i64()
    })
}

/// Destroy a body and the collider attached to it, and invalidate `h`. Returns
/// 1 if a body was destroyed, 0 if `h` was already freed or never named one.
///
/// Without this, a 2D game that spawns and kills bullets or enemies grows
/// Rapier's body and collider sets for as long as it runs. `h` is not recycled:
/// a later `phys_add` may land in the same slot, but at a higher generation, so
/// the old handle keeps reading as dead.
#[no_mangle]
pub extern "C" fn aurora_phys_remove(h: i64) -> i64 {
    PHYS.with(|p| {
        let mut p = p.borrow_mut();
        let Some(p) = p.as_mut() else { return 0 };
        let Some(key) = Body2::from_i64(h) else {
            return 0;
        };
        let Some(body) = p.registry.remove(key) else {
            return 0;
        };
        // `true` = remove the attached colliders with it, so nothing is left
        // in the collider set parented to a body that no longer exists.
        p.bodies.remove(
            body,
            &mut p.islands,
            &mut p.colliders,
            &mut p.impulse,
            &mut p.multibody,
            true,
        );
        1
    })
}

/// Whether `h` still names a live body (1) or has been removed / was never
/// valid (0). `phys_x`/`phys_y` answer 0.0 for a dead handle, which a body at
/// the origin also answers, so this is how the two are told apart.
#[no_mangle]
pub extern "C" fn aurora_phys_alive(h: i64) -> i64 {
    PHYS.with(|p| {
        p.borrow()
            .as_ref()
            .map_or(0, |p| rb2_of(p, h).is_some() as i64)
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
            &g,
            &p.params,
            &mut p.islands,
            &mut p.broad,
            &mut p.narrow,
            &mut p.bodies,
            &mut p.colliders,
            &mut p.impulse,
            &mut p.multibody,
            &mut p.ccd,
            Some(&mut p.query),
            &(),
            &(),
        );
    });
}

fn phys_pos(h: i64, axis: usize) -> f64 {
    PHYS.with(|p| {
        let p = p.borrow();
        let Some(p) = p.as_ref() else { return 0.0 };
        match rb2_of(p, h).and_then(|hd| p.bodies.get(hd)) {
            Some(b) => b.translation()[axis] as f64,
            None => 0.0,
        }
    })
}
#[no_mangle]
pub extern "C" fn aurora_phys_x(h: i64) -> f64 {
    phys_pos(h, 0)
}
#[no_mangle]
pub extern "C" fn aurora_phys_y(h: i64) -> f64 {
    phys_pos(h, 1)
}

/// Set a body's linear velocity.
#[no_mangle]
pub extern "C" fn aurora_phys_set_vel(h: i64, vx: f64, vy: f64) {
    use rapier2d::prelude::*;
    PHYS.with(|p| {
        let mut p = p.borrow_mut();
        let Some(p) = p.as_mut() else { return };
        if let Some(b) = rb2_of(p, h).and_then(|hd| p.bodies.get_mut(hd)) {
            b.set_linvel(vector![vx as Real, vy as Real], true);
        }
    });
}

fn phys_vel(h: i64, axis: usize) -> f64 {
    PHYS.with(|p| {
        let p = p.borrow();
        match p
            .as_ref()
            .and_then(|p| rb2_of(p, h).and_then(|hd| p.bodies.get(hd)))
        {
            Some(b) => b.linvel()[axis] as f64,
            None => 0.0,
        }
    })
}
#[no_mangle]
pub extern "C" fn aurora_phys_vel_x(h: i64) -> f64 {
    phys_vel(h, 0)
}
#[no_mangle]
pub extern "C" fn aurora_phys_vel_y(h: i64) -> f64 {
    phys_vel(h, 1)
}

/// Apply an instantaneous impulse (e.g. a jump or knockback) to a body.
#[no_mangle]
pub extern "C" fn aurora_phys_apply_impulse(h: i64, ix: f64, iy: f64) {
    use rapier2d::prelude::*;
    PHYS.with(|p| {
        let mut p = p.borrow_mut();
        let Some(p) = p.as_mut() else { return };
        if let Some(b) = rb2_of(p, h).and_then(|hd| p.bodies.get_mut(hd)) {
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
        if let Some(b) = rb2_of(p, h).and_then(|hd| p.bodies.get_mut(hd)) {
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
        if let Some(b) = rb2_of(p, h).and_then(|hd| p.bodies.get_mut(hd)) {
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
        let ray = Ray::new(
            point![x as Real, y as Real],
            vector![dx as Real, dy as Real],
        );
        match p.query.cast_ray(
            &p.bodies,
            &p.colliders,
            &ray,
            max as Real,
            true,
            QueryFilter::default(),
        ) {
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
    static NAV_OWN: RefCell<Option<Nav>> = const { RefCell::new(None) };
}

/// The navigation grid, routed to the batch owner's when this thread is a
/// worker.
///
/// A shim with `LocalKey`'s shape, so the call sites below read exactly as they
/// did when it was a plain `thread_local!` - which is the point: the bug was
/// that they all looked fine.
struct NavSlot;

impl NavSlot {
    fn with<R>(&self, f: impl FnOnce(&RefCell<Option<Nav>>) -> R) -> R {
        let batch = par_batch();
        if batch.is_null() {
            return NAV_OWN.with(f);
        }
        // SAFETY: the pointer came from `aurora_run_parallel` on the owner
        // thread, which is blocked in `thread::scope` until this worker joins.
        unsafe {
            with_par_cell(
                batch,
                par_cell(batch, CELL_NAV) as *const RefCell<Option<Nav>>,
                f,
            )
        }
    }
}

const NAV: NavSlot = NavSlot;

#[no_mangle]
pub extern "C" fn aurora_nav_init(w: i64, h: i64) {
    let (w, h) = (w.max(0) as i32, h.max(0) as i32);
    let n = Nav {
        w,
        h,
        walls: vec![false; (w * h).max(0) as usize],
        path: Vec::new(),
    };
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
    NAV.with(|n| {
        n.borrow()
            .as_ref()
            .and_then(|n| n.path.get(i.max(0) as usize))
            .map(|&(x, _)| x as i64)
            .unwrap_or(-1)
    })
}
#[no_mangle]
pub extern "C" fn aurora_nav_y(i: i64) -> i64 {
    NAV.with(|n| {
        n.borrow()
            .as_ref()
            .and_then(|n| n.path.get(i.max(0) as usize))
            .map(|&(_, y)| y as i64)
            .unwrap_or(-1)
    })
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
///
/// # Safety
/// `ptr` must point to `len` initialized bytes.
#[no_mangle]
pub unsafe extern "C" fn aurora_net_connect(ptr: *const u8, len: i64, port: i64) -> i64 {
    // host and PORT, matching the documented net_connect(host, port) and net_join's row.
    //
    // The row used to be [Ptr, I64] - a bare string - so the port was dropped on the floor and
    // the host alone was parsed as a socket address. "127.0.0.1" is not one, so every call
    // failed with InvalidInput and the low-level API could never connect to anything.
    let addr = {
        let s = unsafe { std::slice::from_raw_parts(ptr, len.max(0) as usize) };
        format!("{}:{}", String::from_utf8_lossy(s), port.clamp(0, 65535))
    };
    NET.with(|n| match n.borrow_mut().as_mut() {
        Some(ep) => match ep.connect(&addr) {
            Ok(()) => 1,
            Err(e) => {
                // The error used to be discarded, which made a refused connect
                // indistinguishable from "no endpoint" - report it, since the OS reason
                // (permission, unreachable, bad address) is the whole diagnosis.
                eprintln!(
                    "aurora: net_connect to {addr} failed: {e} (kind {:?})",
                    e.kind()
                );
                0
            }
        },
        None => {
            eprintln!("aurora: net_connect called before net_bind - no local endpoint");
            0
        }
    })
}

/// Reliably send a string message. Returns 1/0.
///
/// # Safety
/// `ptr` must point to `len` initialized bytes.
#[no_mangle]
pub unsafe extern "C" fn aurora_net_send(ptr: *const u8, len: i64) -> i64 {
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
///
/// # Safety
/// `out` must be valid for writes of two `i64`s.
#[no_mangle]
pub unsafe extern "C" fn aurora_net_recv(out: *mut i64) {
    NET.with(|n| {
        if let Some(ep) = n.borrow_mut().as_mut() {
            let delivered = ep.poll();
            NET_INBOX.with(|q| q.borrow_mut().extend(delivered));
        }
    });
    let msg = NET_INBOX
        .with(|q| q.borrow_mut().pop_front())
        .unwrap_or_default();
    unsafe { write_str(out, msg) };
}

// --- data-parallel execution ------------------------------------------------
//
// `par_for(out, f)` fills `out[i] = f(i)` across OS threads. Each thread writes a
// disjoint slice of `out`, and the closure `f` runs as reentrant native code, so
// there's no data race (the only shared state is the read-only closure env and
// disjoint output slots). The closure is `[fn_ptr, env_ptr]`; lambda-lifted
// closures take `(env_ptr, i)` and return i64.

/// # Safety
/// `out` must be valid for writes of `n` `i64`s. `fn_ptr` must be a live
/// `extern "C" fn(i64, i64) -> i64` (a lambda-lifted Aurora closure) and
/// `env_ptr` its matching environment. both must stay valid for the whole
/// call, which runs the closure on several threads at once.
#[no_mangle]
pub unsafe extern "C" fn aurora_par_for(
    out: *mut i64,
    n: i64,
    fn_ptr: *const u8,
    env_ptr: *const u8,
) {
    let n = n.max(0) as usize;
    if n == 0 {
        return;
    }
    // Raw addresses as `usize` are `Send`; pointers are not.
    let out_addr = out as usize;
    let fn_addr = fn_ptr as usize;
    let env_addr = env_ptr as usize;
    let threads = std::thread::available_parallelism()
        .map(|x| x.get())
        .unwrap_or(4)
        .min(n);
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
    /// The match sets of every query loop currently RUNNING, innermost last.
    ///
    /// A stack, not a single list. There used to be one, and a nested query
    /// overwrote it: the outer loop then read entities out of the inner query's
    /// matches, ran off the end, got -1 from `query_entity`, and dereferenced
    /// the null that `get_component(-1, ..)` returns. A segmentation fault, from
    /// source the checker accepted.
    ///
    /// It survived because it needs TWO entities in the outer loop to show. With
    /// one, the body runs once and the corrupted set is never read again - which
    /// is every test anyone had written, and every fight in the game that has
    /// ever had exactly one boss in it.
    ///
    /// Pushes and pops are emitted by codegen and are balanced across returns
    /// and breaks, so the innermost set is always the one the reading loop owns.
    static QUERY: RefCell<Vec<Vec<i64>>> = const { RefCell::new(Vec::new()) };
}

// --- scoped shared-world routing (parallel scheduler) ----------------------
//
// Normally every ECS host fn touches the *calling thread's* thread_local
// `WORLD`. During `aurora_run_parallel`, systems run on worker threads whose own
// thread_local world is empty, so their world access must be routed back to the
// world owned by the thread that started the batch.
//
// That routing is a property of *the thread doing the work*, never of the
// process: a thread that is not part of a batch must keep using its own world.
// `PAR_WORLD` is therefore thread-local and is written only by the scoped
// workers `aurora_run_parallel` spawns, which inherit their parent's world
// pointer explicitly. A process-global slot would reroute every unrelated
// thread's ECS access into whichever world happened to be running a batch, and
// its save/restore would be racy across threads (leaving a pointer to a dead
// stack frame behind).
//
// Lifetime: the pointer is only ever observed by threads created inside the
// `thread::scope` that owns the `ParWorld`, and `thread::scope` joins them all
// before that owner's frame returns. So a live `PAR_WORLD` always points at a
// live `ParWorld`, by construction, with no save/restore protocol at all.
//
// SAFETY also rests on the section 6.2 data-race-freedom theorem the compiler
// already enforces: two systems run concurrently only when their component
// access sets don't conflict, so no two threads ever touch the same component
// buffer mutably. The `Mutex` serialises only *structural* map access
// (lookup/insert); component data is then written through raw pointers into
// heap-stable `Box<[u8]>` buffers, which unrelated inserts never reallocate.
struct ParWorld {
    lock: std::sync::Mutex<()>,
    world: *mut World,
    /// The owner's OTHER runtime state, as opaque cells.
    ///
    /// A worker thread routed the ECS world and nothing else, so every other
    /// subsystem the runtime owns - all of them their own `thread_local!` - was
    /// a freshly zeroed copy on that thread. A system that pathfinds got "no
    /// route" from a grid the program had built and filled; a system that
    /// raycast got "nothing there" from a world full of colliders.
    ///
    /// Not an error and not a crash: the exact answer a caller gets when the
    /// thing genuinely is not there, which is the worst failure mode available
    /// because every caller already handles it. A game found it four iterations
    /// after giving its creatures navigation they had never once used.
    ///
    /// Opaque because the types live in their own modules and this struct has no
    /// business knowing them; each module casts its own back.
    ///
    /// An array rather than a field each, so adding a subsystem is one constant
    /// and one line in `aurora_run_parallel` rather than a new shape here.
    cells: [*const (); ROUTED_CELLS],
}

// Which slot each routed subsystem occupies.
//
// Routed means "one simulation shared by the whole batch". Not everything the
// runtime holds per-thread belongs here, and the distinction is worth stating:
//
//   - The query stack and the frame arena are per-thread ON PURPOSE. Two workers
//     iterating two queries need two iteration states, and a worker's scratch
//     allocations are its own.
//   - The window, the framebuffer, the font and the audio mixer are the
//     frontend's. A worker thread has no business drawing.
//   - Everything below is the simulation, and there is exactly one of it. A
//     worker that cannot see it does not fail - it reports an empty world, which
//     is a legal answer to every question anyone asks of it.
pub(crate) const CELL_NAV: usize = 0;
pub(crate) const CELL_PHYS3: usize = 1;
pub(crate) const CELL_PHYS: usize = 2;
pub(crate) const CELL_GRID3: usize = 3;
pub(crate) const CELL_NAVMESH: usize = 4;
pub(crate) const CELL_RNG: usize = 5;
pub(crate) const CELL_FIXED_DT: usize = 6;
pub(crate) const CELL_VIRTUAL_TIME: usize = 7;
pub(crate) const CELL_FIXED: usize = 8;
pub(crate) const ROUTED_CELLS: usize = 9;

/// A `*const ParWorld` that may be moved into a scoped worker thread.
///
/// SAFETY: the pointee is only reached through `ParWorld`'s own `Mutex`, and
/// `thread::scope` joins every holder before the pointee's frame ends.
#[derive(Clone, Copy)]
struct ParWorldPtr(*const ParWorld);
unsafe impl Send for ParWorldPtr {}
impl ParWorldPtr {
    /// Read the pointer back out. Taking `self` by value keeps closures
    /// capturing the whole `Send` wrapper rather than the bare pointer field.
    fn get(self) -> *const ParWorld {
        self.0
    }
}

thread_local! {
    /// Non-null only while *this* thread is executing systems for a parallel
    /// batch; then it points at the `ParWorld` of the thread that owns the
    /// batch. Threads outside a batch always see null and use their own world.
    static PAR_WORLD: std::cell::Cell<*const ParWorld> =
        const { std::cell::Cell::new(std::ptr::null()) };
}

/// Route ECS world access: the batch's shared world under a lock while this
/// thread is inside a parallel batch, otherwise this thread's own world.
/// The batch this thread is working for, or null when it is its own owner.
///
/// Exposed so a subsystem in another module can route its own state the same
/// way the world does, without that module needing to know how batches work.
pub(crate) fn par_batch() -> *const ParWorld {
    PAR_WORLD.with(|c| c.get())
}

/// Borrow a subsystem cell belonging to the batch's owner, under the batch lock.
///
/// SAFETY: `p` must be `ParWorld::nav` or `ParWorld::phys3` from `par_batch()`,
/// cast back to the type that module put in. Both were taken from the owner's
/// own `thread_local!`, and `thread::scope` cannot return until every worker is
/// joined, so the owner - blocked in that join - keeps them alive and untouched.
pub(crate) unsafe fn with_par_cell<T, R>(
    p: *const ParWorld,
    cell: *const T,
    f: impl FnOnce(&T) -> R,
) -> R {
    let par = unsafe { &*p };
    let _guard = par.lock.lock().unwrap();
    f(unsafe { &*cell })
}

/// The batch owner's cell for a routed subsystem.
pub(crate) fn par_cell(p: *const ParWorld, slot: usize) -> *const () {
    unsafe { &*p }.cells[slot]
}

fn with_world<R>(f: impl FnOnce(&mut World) -> R) -> R {
    let p = PAR_WORLD.with(|c| c.get());
    if p.is_null() {
        WORLD.with(|w| f(&mut w.borrow_mut()))
    } else {
        // SAFETY: `p` was installed by `run_batch` on this very thread and the
        // owning `thread::scope` cannot return until this thread is joined, so
        // the `ParWorld` and its world are still alive. The lock guards
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
/// # Safety
/// `ptr` must point to `size` initialized bytes.
#[no_mangle]
pub unsafe extern "C" fn aurora_store_component(e: i64, tid: i64, ptr: *const u8, size: i64) {
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
/// # Safety
/// `ids` must point to `n` initialized `i64`s.
#[no_mangle]
pub unsafe extern "C" fn aurora_query_begin(ids: *const i64, n: i64) -> i64 {
    let ids = unsafe { std::slice::from_raw_parts(ids, n.max(0) as usize) };
    let matches: Vec<i64> = with_world(|w| {
        w.entities
            .iter()
            .copied()
            .filter(|&e| ids.iter().all(|&t| w.comps.contains_key(&(e, t))))
            .collect()
    });
    let len = matches.len() as i64;
    QUERY.with(|q| {
        let mut q = q.borrow_mut();
        q.push(matches);
    });
    len
}

/// Finish the innermost query loop, so the one enclosing it reads its own
/// matches again. Emitted by codegen on every path out of a query loop.
#[no_mangle]
pub extern "C" fn aurora_query_end() {
    QUERY.with(|q| {
        let mut q = q.borrow_mut();
        // An unbalanced end means codegen closed a query that was never open,
        // which would leave an ENCLOSING loop reading a set it does not own -
        // and the symptom of that is a segmentation fault several frames later.
        // Better to say so here, where the cause is.
        assert!(
            q.pop().is_some(),
            "query_end with no query open: the loop stack is unbalanced"
        );
    });
}

#[no_mangle]
pub extern "C" fn aurora_query_entity(i: i64) -> i64 {
    QUERY.with(|q| {
        let q = q.borrow();
        let out = match q.last() {
            Some(cur) => cur.get(i.max(0) as usize).copied().unwrap_or(-1),
            None => -1,
        };
        out
    })
}
#[no_mangle]
pub extern "C" fn aurora_entity_count() -> i64 {
    with_world(|w| w.entities.len() as i64)
}

/// Despawn every entity and drop all component storage, leaving an empty world.
///
/// What a level transition needs, and what a test suite needs between cases: a
/// world carried over from the last one is a world whose contents no assertion
/// mentioned.
///
/// Entity ids keep counting up rather than restarting at zero. An id held from
/// before the clear then names nothing, instead of silently naming whatever new
/// entity happened to take its number - a stale handle is a bug either way, and
/// the version that resolves to nothing is the one that stays findable.
///
/// The pending query match set is cleared too. A clear from inside a query loop
/// is a mistake, but the loop reads its matches by index and stops at the end,
/// so emptying the set ends that iteration rather than walking entities that no
/// longer exist.
#[no_mangle]
pub extern "C" fn aurora_world_clear() {
    with_world(|w| {
        w.entities.clear();
        w.comps.clear();
    });
    QUERY.with(|q| q.borrow_mut().clear());
}

/// Run a batch of zero-arg system functions concurrently over the shared ECS
/// world. The section 6.2 scheduler check guarantees the systems handed to one
/// batch have non-conflicting component access, so concurrent execution is
/// race-free. Each worker is bound to the caller's world for the batch.
/// `fns` is an array of `n` raw function addresses (each an `extern "C" fn()`).
///
/// # Safety
/// `fns` must point to `n` initialized addresses, each a live `extern "C"
/// fn(i64) -> i64` compiled system, and all of them must stay valid for the
/// whole call.
#[no_mangle]
pub unsafe extern "C" fn aurora_run_parallel(fns: *const usize, n: i64) {
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
    // A batch nested inside another batch keeps running against the world this
    // thread is already bound to, reusing the *same* `ParWorld` so every worker
    // under it contends on one lock (two locks over one world would not exclude
    // each other).
    let inherited = PAR_WORLD.with(|c| c.get());
    if !inherited.is_null() {
        run_batch(ParWorldPtr(inherited), &addrs);
        return;
    }
    WORLD.with(|w| {
        // `as_ptr` yields `*mut World` without taking a RefCell borrow, so the
        // worker threads (which route through `PAR_WORLD` + lock) are the only
        // accessors during the scope; this thread just blocks in the join.
        //
        // The other subsystems are handed over the same way and for the same
        // reason: a worker that cannot see them answers every question about
        // them with "nothing there".
        let par = ParWorld {
            lock: std::sync::Mutex::new(()),
            world: w.as_ptr(),
            cells: routed_cells(),
        };
        run_batch(ParWorldPtr(&par), &addrs);
    });
}

// --- fixed-timestep simulation ---------------------------------------------

/// The fixed-step clock: how long a tick is, how much real time is owed, and how
/// many ticks have run.
///
/// Thread-local for the same reason the world is: two threads each driving their
/// own simulation must not share a clock.
struct FixedClock {
    step: f64,
    owed: f64,
    ticks: i64,
}

thread_local! {
    static FIXED_OWN: RefCell<FixedClock> = const {
        RefCell::new(FixedClock { step: 1.0 / 60.0, owed: 0.0, ticks: 0 })
    };
}

/// The fixed-step clock, routed to the batch owner's while this thread is a
/// worker. See `ROUTED_CELLS`.
///
/// `tick_count` and `fixed_step` are things a system may reasonably ask, so they
/// are marked shared in the ABI table - and a shared builtin whose state is NOT
/// routed is exactly the silent bug that column exists to prevent. Either it is
/// routed or it is owner-only; it may not be neither.
struct FixedSlot;

impl FixedSlot {
    fn with<R>(&self, f: impl FnOnce(&RefCell<FixedClock>) -> R) -> R {
        let batch = par_batch();
        if batch.is_null() {
            return FIXED_OWN.with(f);
        }
        unsafe {
            with_par_cell(
                batch,
                par_cell(batch, CELL_FIXED) as *const RefCell<FixedClock>,
                f,
            )
        }
    }
}

const FIXED: FixedSlot = FixedSlot;

/// Most fixed steps one frame may run before the rest of the debt is written off.
///
/// Without a ceiling a frame that ran long owes several steps, and running them
/// all makes the next frame longer still - the simulation falls further behind
/// the harder it tries to catch up, and the program locks solid. Dropping the
/// excess makes a stalled frame lose simulated time, which is visible and
/// recoverable, instead of never returning.
const MAX_CATCHUP_STEPS: i64 = 8;

/// `tick_rate() -> f64`: the fixed simulation rate, in ticks per second.
///
/// The reader for `set_tick_rate`. Without it a program that needs to know the
/// rate has to remember the number it passed in, which is a copy of a fact the
/// engine already holds - and the copy is what drifts.
#[no_mangle]
pub extern "C" fn aurora_tick_rate() -> f64 {
    let step = FIXED.with(|f| f.borrow().step);
    if step > 0.0 {
        1.0 / step
    } else {
        0.0
    }
}

/// `pin_frame_to_tick()`: pin `frame_dt` to exactly one fixed step, so the frame
/// clock and the simulation clock cannot disagree.
///
/// The pair `set_tick_rate(hz)` + `set_fixed_dt(1.0 / hz)` is one intention
/// written as two calls that must agree, and every caller spelling the second
/// half is a place the two can drift. This states the relation once, here, where
/// both clocks live - a caller says WHAT it wants rather than recomputing HOW.
#[no_mangle]
pub extern "C" fn aurora_pin_frame_to_tick() {
    let step = FIXED.with(|f| f.borrow().step);
    crate::data::aurora_set_fixed_dt(step);
}

/// Set the fixed simulation rate in ticks per second. Values outside a sane
/// range are ignored rather than allowed to produce a zero or negative step.
#[no_mangle]
pub extern "C" fn aurora_set_tick_rate(hz: f64) {
    if hz.is_finite() && (1.0..=1000.0).contains(&hz) {
        FIXED.with(|f| f.borrow_mut().step = 1.0 / hz);
    }
}

/// Fixed ticks simulated since the program started.
///
/// This, not a frame counter, is what game rules should be written against: it
/// advances at exactly the configured rate regardless of how long frames take.
#[no_mangle]
pub extern "C" fn aurora_tick_count() -> i64 {
    FIXED.with(|f| f.borrow().ticks)
}

/// How far the current frame sits between the last fixed tick and the next, in
/// `0..1`. Render positions interpolated by this do not judder when the frame
/// rate and the tick rate disagree.
#[no_mangle]
pub extern "C" fn aurora_tick_alpha() -> f64 {
    FIXED.with(|f| {
        let f = f.borrow();
        if f.step > 0.0 {
            (f.owed / f.step).clamp(0.0, 1.0)
        } else {
            0.0
        }
    })
}

/// The fixed step length in seconds - what a `FixedUpdate` system should use as
/// its delta rather than the frame time.
#[no_mangle]
pub extern "C" fn aurora_tick_delta() -> f64 {
    FIXED.with(|f| f.borrow().step)
}

/// Advance the fixed clock by `dt` seconds and run the fixed schedule once per
/// whole step owed. Returns the number of steps run.
///
/// `fns` is a flat array of system addresses and `lens` gives the length of each
/// layer within it, so one call carries the whole schedule: layers run in order,
/// and a layer with several systems runs them concurrently exactly as
/// [`aurora_run_parallel`] does for the frame schedule.
///
/// # Safety
/// `fns` must point to the sum of `lens[0..n_layers]` initialized addresses, each
/// a live compiled system, and `lens` to `n_layers` initialized lengths. All must
/// stay valid for the whole call.
#[no_mangle]
pub unsafe extern "C" fn aurora_run_fixed(
    fns: *const usize,
    lens: *const i64,
    n_layers: i64,
    dt: f64,
) -> i64 {
    let n_layers = n_layers.max(0) as usize;
    let steps = FIXED.with(|f| {
        let mut f = f.borrow_mut();
        // A frame time that is negative, NaN or absurd must not move the clock;
        // a paused or hitching host should resume, not teleport.
        if dt.is_finite() && dt > 0.0 {
            f.owed += dt;
        }
        let want = if f.step > 0.0 {
            (f.owed / f.step) as i64
        } else {
            0
        };
        let run = want.min(MAX_CATCHUP_STEPS);
        f.owed -= run as f64 * f.step;
        if want > run {
            // Debt beyond the ceiling is written off, not banked.
            f.owed = 0.0;
        }
        f.ticks += run;
        run
    });

    if n_layers == 0 || steps == 0 {
        return steps;
    }
    let lens = unsafe { std::slice::from_raw_parts(lens, n_layers) };
    let total: usize = lens.iter().map(|&l| l.max(0) as usize).sum();
    let all = unsafe { std::slice::from_raw_parts(fns, total) };

    for _ in 0..steps {
        let mut at = 0usize;
        for &len in lens {
            let len = len.max(0) as usize;
            if len > 0 {
                // SAFETY: `all` is a live slice of system addresses and this
                // window lies inside it by construction of `total`.
                unsafe { aurora_run_parallel(all[at..].as_ptr(), len as i64) };
            }
            at += len;
        }
    }
    steps
}

/// Run `addrs` concurrently, each worker bound to `par` for the duration.
///
/// The binding is installed on the worker threads only, so no thread outside
/// this scope can ever observe it, and `thread::scope` guarantees every worker
/// is joined before `par`'s frame goes away.
/// Every routed subsystem's cell on THIS thread, for handing to workers.
///
/// One line per subsystem, in one place. A subsystem missing from here is a
/// subsystem that answers "nothing there" to every system in a parallel layer,
/// so the tests in `systems_see_the_runtime` carry one case each and a new
/// entry without a case is a gap that shows.
fn routed_cells() -> [*const (); ROUTED_CELLS] {
    let mut c = [std::ptr::null(); ROUTED_CELLS];
    c[CELL_NAV] = NAV_OWN.with(|n| n as *const _ as *const ());
    c[CELL_PHYS3] = crate::phys3d::own_cell();
    c[CELL_PHYS] = PHYS.with(|n| n as *const _ as *const ());
    c[CELL_GRID3] = crate::nav3d::own_grid3();
    c[CELL_NAVMESH] = crate::nav3d::own_navmesh();
    c[CELL_RNG] = crate::data::own_rng();
    c[CELL_FIXED_DT] = crate::data::own_fixed_dt();
    c[CELL_VIRTUAL_TIME] = crate::data::own_virtual_time();
    c[CELL_FIXED] = FIXED_OWN.with(|c| c as *const _ as *const ());
    c
}

fn run_batch(par: ParWorldPtr, addrs: &[usize]) {
    std::thread::scope(|scope| {
        for &a in addrs {
            scope.spawn(move || {
                PAR_WORLD.with(|c| c.set(par.get()));
                // SAFETY: `a` is a finalized native function address; `usize` is
                // `Send`. System bodies access the world only through the
                // routing layer above.
                let f: extern "C" fn() = unsafe { std::mem::transmute(a) };
                f();
            });
        }
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
///
/// # Safety
/// `ptr` must point to `len` initialized bytes.
#[no_mangle]
pub unsafe extern "C" fn aurora_scene_save(ptr: *const u8, len: i64) -> i64 {
    let path = {
        let s = unsafe { std::slice::from_raw_parts(ptr, len.max(0) as usize) };
        String::from_utf8_lossy(s).into_owned()
    };
    // Routed like every other world access, so saving from inside a parallel
    // batch snapshots the batch's world instead of an empty thread-local one.
    let bytes = with_world(|w| {
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
///
/// # Safety
/// `ptr` must point to `len` initialized bytes.
#[no_mangle]
pub unsafe extern "C" fn aurora_scene_load(ptr: *const u8, len: i64) -> i64 {
    let path = {
        let s = unsafe { std::slice::from_raw_parts(ptr, len.max(0) as usize) };
        String::from_utf8_lossy(s).into_owned()
    };
    let Ok(b) = std::fs::read(&path) else {
        return 0;
    };
    if b.len() < 4 || &b[0..4] != b"ASCN" {
        return 0;
    }
    let mut pos = 4;
    let mut parse = || -> Option<World> {
        let mut world = World {
            next: get_i64(&b, &mut pos)?,
            ..Default::default()
        };
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
            with_world(|world| *world = w);
            1
        }
        None => 0,
    }
}

// --- profiler: per-function call counts + time ------------------------------
//
// In profiling builds the compiler emits `aurora_prof_enter(name)` at each
// function entry and `aurora_prof_exit()` at each return, accumulating call
// counts and wall-clock time per function - a real instrumenting profiler over
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
            .map(|(f, &(calls, nanos))| ProfRow {
                func: f.clone(),
                calls,
                nanos,
            })
            .collect();
        rows.sort_by(|a, b| b.nanos.cmp(&a.nanos));
        rows
    })
}

/// # Safety
/// `name_ptr` must point to `name_len` initialized bytes.
#[no_mangle]
pub unsafe extern "C" fn aurora_prof_enter(name_ptr: *const u8, name_len: i64) {
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

/// Whether audio output should skip the device: true under `AURORA_HEADLESS=1`,
/// so verification runs never contend for cpal (deterministic, no audio thread).
/// Read once and cached.
pub(crate) fn headless_audio() -> bool {
    use std::sync::OnceLock;
    static H: OnceLock<bool> = OnceLock::new();
    *H.get_or_init(|| {
        std::env::var("AURORA_HEADLESS")
            .map(|v| v == "1")
            .unwrap_or(false)
    })
}

// Offline audio capture: under headless, play_note/play_sound record the note
// (semitone, seconds, virtual start time) instead of touching the device, so
// `audio_capture_save` can render them to a WAV that `wav-audit` verifies -
// closing the loop on synthesized/procedural audio (library WAVs are audited
// directly).
thread_local! {
    static AUDIO_CAP: RefCell<Vec<(i32, f32, f64)>> = const { RefCell::new(Vec::new()) };
}

// Library sound plays, so a headless run can be asked whether the game actually
// made a noise.
//
// `audio_capture_save` above renders SYNTHESIZED notes only - its own comment
// says library WAVs are audited directly - and both play functions returned
// early under headless before they had even looked at the handle. So nothing a
// script could ask distinguished "the fight played twenty-six sounds" from "the
// assets were never staged and the game ran in silence", and a check written
// against that could only assert the files LOADED, which is not the thing that
// matters.
//
// A negative handle deliberately does NOT count. `load_sound` answers -1 for a
// missing file and every play function treats that as a no-op, so the silent
// build is exactly the one that reports zero plays - which is the failure a
// check needs to be able to see.
thread_local! {
    static SOUND_PLAYS: std::cell::Cell<i64> = const { std::cell::Cell::new(0) };
    static SOUND_LAST: std::cell::Cell<i64> = const { std::cell::Cell::new(-1) };
    static MUSIC_NOW: std::cell::Cell<i64> = const { std::cell::Cell::new(-1) };
    static MUSIC_STARTS: std::cell::Cell<i64> = const { std::cell::Cell::new(0) };
}

/// `music_starts() -> i64`: how many times a bed has been STARTED.
///
/// `play_music` replaces whatever is playing, so calling it every frame restarts
/// the track every frame and the bed never advances past its first fraction of a
/// second - the audio twin of an animation re-seeded each frame, which this
/// engine already made `r3d_anim_play` idempotent to prevent. `music_now` cannot
/// see that: it reports the same handle whether the bed is playing or being
/// relaunched sixty times a second. This is what makes "started once and left
/// alone" an assertable property rather than a hope.
#[no_mangle]
pub extern "C" fn aurora_music_starts() -> i64 {
    MUSIC_STARTS.with(|c| c.get())
}

/// `music_now() -> i64`: the handle of the bed currently playing, or -1.
///
/// Music is a STATE, not an event - one bed at a time, replaced rather than
/// layered - so this answers which one rather than counting starts. It is the
/// counterpart of `audio_plays` and exists for the same reason: `play_music`
/// returned early under headless before touching anything, so a game could not
/// be asked whether its boss theme actually started, and a check written against
/// that could only assert the file LOADED.
#[no_mangle]
pub extern "C" fn aurora_music_now() -> i64 {
    MUSIC_NOW.with(|c| c.get())
}

fn record_sound_play(handle: i64) {
    SOUND_PLAYS.with(|c| c.set(c.get() + 1));
    SOUND_LAST.with(|c| c.set(handle));
}

/// `audio_plays() -> i64`: how many library sounds have been played, ever.
///
/// Counts the request the engine ACCEPTED, not what reached a speaker: headless
/// has no device and a distant sound is culled by the mixer, and neither of those
/// is what a game is asking about when it asks whether it played a sound.
#[no_mangle]
pub extern "C" fn aurora_audio_plays() -> i64 {
    SOUND_PLAYS.with(|c| c.get())
}

/// `audio_last_sound() -> i64`: the handle most recently played, or -1.
///
/// The half that makes a check non-decorative. A count says a noise happened; a
/// game wants to assert that the DEFLECT played a clash rather than a hit, and
/// that question needs to name which buffer went out.
#[no_mangle]
pub extern "C" fn aurora_audio_last_sound() -> i64 {
    SOUND_LAST.with(|c| c.get())
}

fn audio_capture_note(semitone: i64, dur_ms: i64) {
    let t = crate::data::virtual_time_seconds();
    AUDIO_CAP.with(|c| {
        c.borrow_mut()
            .push((semitone as i32, (dur_ms.max(0) as f32) / 1000.0, t))
    });
}

/// Render the captured note events into a 16-bit mono WAV at 44.1 kHz, placing
/// each note at its virtual start time. Returns 1 on success, 0 on failure.
///
/// # Safety
/// `ptr` must point to `len` initialized bytes.
#[no_mangle]
pub unsafe extern "C" fn aurora_audio_capture_save(ptr: *const u8, len: i64) -> i64 {
    let path = {
        let s = unsafe { std::slice::from_raw_parts(ptr, len.max(0) as usize) };
        String::from_utf8_lossy(s).into_owned()
    };
    let sr = 44_100u32;
    let events = AUDIO_CAP.with(|c| c.borrow().clone());
    if events.is_empty() {
        return 0;
    }
    // Buffer spans from 0 to the latest note end.
    let end_s = events
        .iter()
        .map(|(_, d, t)| t + *d as f64)
        .fold(0.0, f64::max);
    let total = ((end_s * sr as f64).ceil() as usize).max(1) + sr as usize / 10;
    let mut buf = vec![0.0f32; total];
    for (semi, dur, t) in &events {
        let note = aurora_audio::Note::new(aurora_audio::pitch(*semi), *dur)
            .wave(aurora_audio::Wave::Triangle)
            .gain(0.4);
        let samples = note.render(sr);
        let off = (t * sr as f64) as usize;
        for (i, s) in samples.iter().enumerate() {
            if off + i < buf.len() {
                buf[off + i] += *s;
            }
        }
    }
    // Soft-clip to keep peaks in range, then write 16-bit PCM.
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: sr,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    if let Some(parent) = std::path::Path::new(&path).parent() {
        if !parent.as_os_str().is_empty() {
            let _ = std::fs::create_dir_all(parent);
        }
    }
    let Ok(mut w) = hound::WavWriter::create(&path, spec) else {
        return 0;
    };
    for s in &buf {
        let v = (s.clamp(-1.0, 1.0) * 32767.0) as i16;
        if w.write_sample(v).is_err() {
            return 0;
        }
    }
    w.finalize().is_ok() as i64
}

/// Synthesize and play one note: `semitone` is relative to A4, `dur_ms` ms long.
/// Blocks until the note finishes (so notes sequence naturally).
#[no_mangle]
pub extern "C" fn aurora_play_note(semitone: i64, dur_ms: i64) {
    if headless_audio() {
        audio_capture_note(semitone, dur_ms);
        return;
    }
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
///
/// # Safety
/// `ptr` must point to `len` initialized bytes.
#[no_mangle]
pub unsafe extern "C" fn aurora_gpu_render(ptr: *const u8, len: i64, time_ms: i64) {
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
/// f64->f32 for the GPU and back. Backs the `gpu_compute` builtin.
///
/// # Safety
/// `wptr` must point to `wlen` initialized bytes. `data` must be valid for
/// reads and writes of `n` `f64`s.
#[no_mangle]
pub unsafe extern "C" fn aurora_gpu_compute(wptr: *const u8, wlen: i64, data: *mut f64, n: i64) {
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

/// Open a real-time window backing a `w` x `h` builtin framebuffer.
#[no_mangle]
pub extern "C" fn aurora_window_open(w: i64, h: i64) {
    aurora_window::imm_open(w.max(0) as u32, h.max(0) as u32);
}

/// Present the current framebuffer and pump events; returns 1 while open, 0 when
/// the window has been closed.
#[no_mangle]
pub extern "C" fn aurora_window_present() -> i64 {
    let rgba = FB.with(|fb| fb.borrow().as_ref().map(|f| f.rgba()).unwrap_or_default());
    // A frame just ended: `input_step` is the boundary, and it advances the edge
    // snapshot and spends this frame's delta together. Done here rather than left
    // to the game - an edge snapshot that is only advanced when someone remembers
    // to is one that reports every held button as a fresh press forever.
    //
    // BEFORE the present, which is what pumps the event loop: snapshotting after
    // it records this frame's new presses as already-seen and kills every
    // `input_pressed` in a windowed game. See `input_step`.
    aurora_input_step();
    let open = aurora_window::imm_present(&rgba);
    if open {
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
///
/// # Safety
/// `ptr` must point to `len` initialized bytes.
#[no_mangle]
pub unsafe extern "C" fn aurora_r3d_load_model(ptr: *const u8, len: i64) -> i64 {
    let s = unsafe { std::slice::from_raw_parts(ptr, len.max(0) as usize) };
    let path = String::from_utf8_lossy(s);
    aurora_window::imm_r3d_load_model(&path)
}

/// Borrow an Aurora string argument as a `&str`.
///
/// # Safety
/// `ptr` must point to `len` initialized bytes that stay valid for the call.
unsafe fn arg_str<'a>(ptr: *const u8, len: i64) -> std::borrow::Cow<'a, str> {
    String::from_utf8_lossy(unsafe { std::slice::from_raw_parts(ptr, len.max(0) as usize) })
}

/// Name the rig the clips gathered so far were authored on.
///
/// # Safety
/// `ptr` must point to `len` initialized bytes.
#[no_mangle]
pub unsafe extern "C" fn aurora_r3d_clip_rig(ptr: *const u8, len: i64) {
    aurora_window::imm_r3d_clip_rig(&unsafe { arg_str(ptr, len) });
}

/// Add one clip file to the moveset being gathered.
///
/// # Safety
/// `ptr` must point to `len` initialized bytes.
#[no_mangle]
pub unsafe extern "C" fn aurora_r3d_clip_add(ptr: *const u8, len: i64) {
    aurora_window::imm_r3d_clip_add(&unsafe { arg_str(ptr, len) });
}

/// Map a bone name on the clips' rig to its name on the character.
///
/// # Safety
/// Both pointers must point to their stated number of initialized bytes.
#[no_mangle]
pub unsafe extern "C" fn aurora_r3d_bone_map(
    from: *const u8,
    from_len: i64,
    to: *const u8,
    to_len: i64,
) {
    aurora_window::imm_r3d_bone_map(&unsafe { arg_str(from, from_len) }, &unsafe {
        arg_str(to, to_len)
    });
}

/// Allow one character bone to take translation from a clip - the root.
///
/// # Safety
/// `ptr` must point to `len` initialized bytes.
#[no_mangle]
pub unsafe extern "C" fn aurora_r3d_clip_root(ptr: *const u8, len: i64) {
    aurora_window::imm_r3d_clip_root(&unsafe { arg_str(ptr, len) });
}

/// Load a character with the moveset gathered since the last load; -1 on
/// failure. Clears the gathered recipe.
///
/// # Safety
/// `ptr` must point to `len` initialized bytes.
#[no_mangle]
pub unsafe extern "C" fn aurora_r3d_load_character(ptr: *const u8, len: i64) -> i64 {
    aurora_window::imm_r3d_load_character(&unsafe { arg_str(ptr, len) })
}

/// Load a mesh as a part of `host`'s body, rebound onto its skeleton; -1 if it
/// cannot be bound.
///
/// # Safety
/// `ptr` must point to `len` initialized bytes.
#[no_mangle]
pub unsafe extern "C" fn aurora_r3d_load_part(ptr: *const u8, len: i64, host: i64) -> i64 {
    aurora_window::imm_r3d_load_part(&unsafe { arg_str(ptr, len) }, host)
}

/// Add one mesh file to the body being gathered for `r3d_load_assembly`.
///
/// # Safety
/// `ptr` must point to `len` initialized bytes.
#[no_mangle]
pub unsafe extern "C" fn aurora_r3d_part_add(ptr: *const u8, len: i64) {
    aurora_window::imm_r3d_part_add(&unsafe { arg_str(ptr, len) });
}

/// Assemble a character from the gathered parts, deriving the rig from them;
/// -1 if they do not share one. Clears the gathered recipe.
#[no_mangle]
pub extern "C" fn aurora_r3d_load_assembly() -> i64 {
    aurora_window::imm_r3d_load_assembly()
}

/// Attach a texture to every mesh whose material is named `material` and that
/// carries none of its own.
///
/// # Safety
/// Both pointers must point to their stated number of initialized bytes.
#[no_mangle]
pub unsafe extern "C" fn aurora_r3d_material_texture(
    material: *const u8,
    material_len: i64,
    path: *const u8,
    path_len: i64,
) {
    aurora_window::imm_r3d_material_texture(&unsafe { arg_str(material, material_len) }, &unsafe {
        arg_str(path, path_len)
    });
}
/// Half-extent of a loaded model's bounding box on one axis (0 = x, 1 = y,
/// 2 = z), in model space, before any draw scale.
///
/// This exists so a game can size a collider to the art instead of to a guess.
/// A hand-typed box is a number that silently stops matching the moment an asset
/// is swapped, and a collider that is not where the model is reads to a player
/// as the world being broken rather than as a wrong constant.
#[no_mangle]
pub extern "C" fn aurora_r3d_model_extent(handle: i64, axis: i64) -> f64 {
    aurora_window::imm_r3d_model_extent(handle, axis) as f64
}
/// Centre of a loaded model's bounding box on one axis, relative to the model's
/// origin. A model authored standing on its origin has a positive `y` centre,
/// which is the offset its collider needs to sit on the same ground.
#[no_mangle]
pub extern "C" fn aurora_r3d_model_centre(handle: i64, axis: i64) -> f64 {
    aurora_window::imm_r3d_model_centre(handle, axis) as f64
}
/// Release a model/primitive handle and every GPU buffer behind it.
///
/// Returns 1 when the handle was live and is now freed, 0 when it was already
/// freed or was never valid. The handle is dead afterwards: drawing with it
/// does nothing rather than drawing whatever is loaded into its slot next,
/// because handles carry a generation the freed slot no longer answers to.
///
/// This is what lets a level change actually release its assets. Without it the
/// only way to reclaim a model was to end the process.
#[no_mangle]
pub extern "C" fn aurora_r3d_free_model(handle: i64) -> i64 {
    aurora_window::imm_r3d_free_model(handle)
}
#[no_mangle]
pub extern "C" fn aurora_r3d_make_box(r: f64, g: f64, b: f64) -> i64 {
    aurora_window::imm_r3d_make_box(r as f32, g as f32, b as f32)
}
/// A box mesh sized by half-extents (matching a physics box collider), colored.
#[no_mangle]
pub extern "C" fn aurora_r3d_make_box_sized(
    hx: f64,
    hy: f64,
    hz: f64,
    r: f64,
    g: f64,
    b: f64,
) -> i64 {
    aurora_window::imm_r3d_make_box_sized(
        hx as f32, hy as f32, hz as f32, r as f32, g as f32, b as f32,
    )
}
/// An emissive (self-lit, glowing) box mesh. Color is the emissive RGB.
#[no_mangle]
pub extern "C" fn aurora_r3d_make_box_emissive(
    hx: f64,
    hy: f64,
    hz: f64,
    r: f64,
    g: f64,
    b: f64,
) -> i64 {
    aurora_window::imm_r3d_make_box_emissive(
        hx as f32, hy as f32, hz as f32, r as f32, g as f32, b as f32,
    )
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
pub extern "C" fn aurora_r3d_camera(
    ex: f64,
    ey: f64,
    ez: f64,
    tx: f64,
    ty: f64,
    tz: f64,
    fov: f64,
) {
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
pub extern "C" fn aurora_r3d_light(
    dx: f64,
    dy: f64,
    dz: f64,
    r: f64,
    g: f64,
    b: f64,
    ambient: f64,
) {
    aurora_window::imm_r3d_light(
        dx as f32,
        dy as f32,
        dz as f32,
        r as f32,
        g as f32,
        b as f32,
        ambient as f32,
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
    h: i64,
    px: f64,
    py: f64,
    pz: f64,
    yaw: f64,
    pitch: f64,
    roll: f64,
    scale: f64,
) {
    aurora_window::imm_r3d_draw(
        h,
        px as f32,
        py as f32,
        pz as f32,
        yaw as f32,
        pitch as f32,
        roll as f32,
        scale as f32,
    );
}
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub extern "C" fn aurora_r3d_draw_quat(
    h: i64,
    px: f64,
    py: f64,
    pz: f64,
    qx: f64,
    qy: f64,
    qz: f64,
    qw: f64,
    scale: f64,
) {
    aurora_window::imm_r3d_draw_quat(
        h,
        px as f32,
        py as f32,
        pz as f32,
        qx as f32,
        qy as f32,
        qz as f32,
        qw as f32,
        scale as f32,
    );
}
#[no_mangle]
pub extern "C" fn aurora_r3d_draw_tint(
    h: i64,
    px: f64,
    py: f64,
    pz: f64,
    yaw: f64,
    pitch: f64,
    roll: f64,
    scale: f64,
    r: f64,
    g: f64,
    b: f64,
) {
    aurora_window::imm_r3d_draw_tint(
        h,
        px as f32,
        py as f32,
        pz as f32,
        yaw as f32,
        pitch as f32,
        roll as f32,
        scale as f32,
        r as f32,
        g as f32,
        b as f32,
    );
}
#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub extern "C" fn aurora_r3d_draw_shield(
    h: i64,
    px: f64,
    py: f64,
    pz: f64,
    yaw: f64,
    pitch: f64,
    roll: f64,
    scale: f64,
    strength: f64,
    time: f64,
) {
    aurora_window::imm_r3d_draw_shield(
        h,
        px as f32,
        py as f32,
        pz as f32,
        yaw as f32,
        pitch as f32,
        roll as f32,
        scale as f32,
        strength as f32,
        time as f32,
    );
}
#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub extern "C" fn aurora_r3d_draw_on_joint(
    weapon: i64,
    host: i64,
    joint: i64,
    hx: f64,
    hy: f64,
    hz: f64,
    hyaw: f64,
    hpitch: f64,
    hroll: f64,
    hscale: f64,
    ox: f64,
    oy: f64,
    oz: f64,
    oyaw: f64,
    opitch: f64,
    oroll: f64,
    oscale: f64,
) {
    aurora_window::imm_r3d_draw_on_joint(
        weapon,
        host,
        joint,
        hx as f32,
        hy as f32,
        hz as f32,
        hyaw as f32,
        hpitch as f32,
        hroll as f32,
        hscale as f32,
        ox as f32,
        oy as f32,
        oz as f32,
        oyaw as f32,
        opitch as f32,
        oroll as f32,
        oscale as f32,
    );
}
/// Per-axis scaled draw: one unit mesh, any box size. Lets a streamed level draw
/// every wall from a single handle instead of uploading a mesh per size.
#[no_mangle]
pub extern "C" fn aurora_r3d_draw_scaled(
    h: i64,
    x: f64,
    y: f64,
    z: f64,
    yaw: f64,
    pitch: f64,
    roll: f64,
    sx: f64,
    sy: f64,
    sz: f64,
) {
    aurora_window::imm_r3d_draw_scaled(
        h,
        x as f32,
        y as f32,
        z as f32,
        yaw as f32,
        pitch as f32,
        roll as f32,
        sx as f32,
        sy as f32,
        sz as f32,
    );
}
#[no_mangle]
pub extern "C" fn aurora_r3d_draw_skinned(
    armor: i64,
    host: i64,
    x: f64,
    y: f64,
    z: f64,
    yaw: f64,
    pitch: f64,
    roll: f64,
    scale: f64,
) {
    aurora_window::imm_r3d_draw_skinned(
        armor,
        host,
        x as f32,
        y as f32,
        z as f32,
        yaw as f32,
        pitch as f32,
        roll as f32,
        scale as f32,
    );
}
#[no_mangle]
pub extern "C" fn aurora_r3d_joint_dump(host: i64) {
    aurora_window::imm_r3d_joint_dump(host);
}
/// Model-space position of a joint (axis 0=x/1=y/2=z) in the host's current pose.
#[no_mangle]
pub extern "C" fn aurora_r3d_joint_pos(host: i64, joint: i64, axis: i64) -> f64 {
    aurora_window::imm_r3d_joint_pos(host, joint, axis) as f64
}
/// A joint's position IN THE WORLD, for a host at a given placement.
#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub extern "C" fn aurora_r3d_joint_world(
    h: i64,
    joint: i64,
    axis: i64,
    x: f64,
    y: f64,
    z: f64,
    yaw: f64,
    scale: f64,
) -> f64 {
    aurora_window::imm_r3d_joint_world(
        h,
        joint,
        axis,
        x as f32,
        y as f32,
        z as f32,
        yaw as f32,
        scale as f32,
    ) as f64
}
/// `r3d_clip_joint_world(h, clip, t, joint, axis, x, y, z, yaw, scale) -> f64`:
/// where a joint IS on one frame of one clip, in world space.
///
/// The asset-side twin of `r3d_joint_world`, which answers about the pose on
/// screen. A rule may ask this and may not ask that: this is a pure function of
/// authored data, so a simulation that owns which move is out and how far
/// through it can place a hitbox on the weapon without asking the animator
/// anything.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub extern "C" fn aurora_r3d_clip_joint_world(
    h: i64,
    clip: i64,
    t: f64,
    joint: i64,
    axis: i64,
    x: f64,
    y: f64,
    z: f64,
    yaw: f64,
    scale: f64,
) -> f64 {
    aurora_window::imm_r3d_clip_joint_world(
        h,
        clip,
        t as f32,
        joint,
        axis,
        x as f32,
        y as f32,
        z as f32,
        yaw as f32,
        scale as f32,
    ) as f64
}
/// `r3d_clip_joint_basis(h, clip, t, joint, axis, comp, yaw) -> f64`: which way
/// a joint POINTS on one frame of one clip.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub extern "C" fn aurora_r3d_clip_joint_basis(
    h: i64,
    clip: i64,
    t: f64,
    joint: i64,
    axis: i64,
    comp: i64,
    yaw: f64,
) -> f64 {
    aurora_window::imm_r3d_clip_joint_basis(h, clip, t as f32, joint, axis, comp, yaw as f32) as f64
}
/// `r3d_clip_joint_ok(h, clip, joint) -> i64`: can this rig answer about that
/// joint in that clip at all? The two above fall back to the origin, which is a
/// plausible answer, so this is how a caller tells a missing clip from a joint
/// that really is at the body's feet.
#[no_mangle]
pub extern "C" fn aurora_r3d_clip_joint_ok(h: i64, clip: i64, joint: i64) -> i64 {
    aurora_window::imm_r3d_clip_joint_ok(h, clip, joint)
}
/// `r3d_yaw_rotate(yaw, lx, lz, axis) -> f64`: turn a LOCAL offset by a yaw into
/// world space, in the renderer's own handedness. `axis` 0 = x, 2 = z.
///
/// The rotation `r3d_draw` applies, on its own. Without it a caller placing
/// anything relative to a turned object writes `x*cos + z*sin` / `-x*sin + z*cos`
/// by hand, and the sign of the second row looks obviously right in both
/// directions - this project shipped it in both.
#[no_mangle]
pub extern "C" fn aurora_r3d_yaw_rotate(yaw: f64, lx: f64, lz: f64, axis: i64) -> f64 {
    aurora_window::imm_r3d_yaw_rotate(yaw as f32, lx as f32, lz as f32, axis) as f64
}
/// One column of a joint's world rotation basis: which way the joint faces.
#[no_mangle]
pub extern "C" fn aurora_r3d_joint_basis(
    h: i64,
    joint: i64,
    axis: i64,
    comp: i64,
    yaw: f64,
) -> f64 {
    aurora_window::imm_r3d_joint_basis(h, joint, axis, comp, yaw as f32) as f64
}
#[no_mangle]
pub extern "C" fn aurora_r3d_anim_play(h: i64, clip: i64, looping: i64, speed: f64, fade: f64) {
    aurora_window::imm_r3d_anim_play(h, clip, looping, speed as f32, fade as f32);
}
/// Start a clip from the top even if it is already the one playing.
///
/// `r3d_anim_play` states what SHOULD be playing and is idempotent, so a frame
/// loop can call it every frame. This is the explicit "again", for the combo
/// step that reuses a clip.
#[no_mangle]
pub extern "C" fn aurora_r3d_anim_restart(h: i64, clip: i64, looping: i64, speed: f64, fade: f64) {
    aurora_window::imm_r3d_anim_restart(h, clip, looping, speed as f32, fade as f32);
}
#[no_mangle]
pub extern "C" fn aurora_r3d_anim_update(h: i64, dt: f64) {
    aurora_window::imm_r3d_anim_update(h, dt as f32);
}
#[no_mangle]
pub extern "C" fn aurora_r3d_anim_play_upper(
    h: i64,
    clip: i64,
    looping: i64,
    speed: f64,
    fade: f64,
    mask_root: i64,
) {
    aurora_window::imm_r3d_anim_play_upper(h, clip, looping, speed as f32, fade as f32, mask_root);
}
#[no_mangle]
pub extern "C" fn aurora_r3d_anim_aim_upper(
    h: i64,
    clip_a: i64,
    clip_b: i64,
    weight: f64,
    speed: f64,
    fade: f64,
    mask_root: i64,
) {
    aurora_window::imm_r3d_anim_aim_upper(
        h,
        clip_a,
        clip_b,
        weight as f32,
        speed as f32,
        fade as f32,
        mask_root,
    );
}
#[no_mangle]
pub extern "C" fn aurora_r3d_anim_blend(
    h: i64,
    clip_a: i64,
    clip_b: i64,
    weight: f64,
    speed: f64,
    fade: f64,
    looping: i64,
) {
    aurora_window::imm_r3d_anim_blend(
        h,
        clip_a,
        clip_b,
        weight as f32,
        speed as f32,
        fade as f32,
        looping != 0,
    );
}
/// `r3d_anim_blend_space(h, a0, a1, b0, b1, dir, weight, speed, fade, looping)`:
/// a 2x2 locomotion blend space.
///
/// `a0`/`a1` are the lower tier and `b0`/`b1` the upper - idle and walk, or walk
/// and run - and `dir` mixes within BOTH at once. `r3d_anim_blend` is this with
/// each tier collapsed to one clip, so a caller that has no direction axis is
/// using the same mechanism rather than a different one.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub extern "C" fn aurora_r3d_anim_blend_space(
    h: i64,
    a0: i64,
    a1: i64,
    b0: i64,
    b1: i64,
    dir: f64,
    weight: f64,
    speed: f64,
    fade: f64,
    looping: i64,
) {
    aurora_window::imm_r3d_anim_blend_space(
        h,
        a0,
        a1,
        b0,
        b1,
        dir as f32,
        weight as f32,
        speed as f32,
        fade as f32,
        looping != 0,
    );
}
#[no_mangle]
pub extern "C" fn aurora_r3d_anim_seek(h: i64, t: f64) {
    aurora_window::imm_r3d_anim_seek(h, t as f32);
}
/// Jump the upper-body overlay to `t` seconds.
#[no_mangle]
pub extern "C" fn aurora_r3d_anim_seek_upper(h: i64, t: f64) {
    aurora_window::imm_r3d_anim_seek_upper(h, t as f32);
}
#[no_mangle]
pub extern "C" fn aurora_r3d_pose_bone(h: i64, joint: i64, rx: f64, ry: f64, rz: f64) {
    aurora_window::imm_r3d_pose_bone(h, joint, rx as f32, ry as f32, rz as f32);
}
#[no_mangle]
pub extern "C" fn aurora_r3d_clear_pose(h: i64) {
    aurora_window::imm_r3d_clear_pose(h);
}
/// Hide one skin joint's geometry on a model (first-person arms drop the body this way).
#[no_mangle]
pub extern "C" fn aurora_r3d_hide_joint(h: i64, joint: i64) {
    aurora_window::imm_r3d_hide_joint(h, joint);
}
/// Undo every `r3d_hide_joint` on a model, so a pooled character can be reused
/// without reloading it from disk.
#[no_mangle]
pub extern "C" fn aurora_r3d_show_joints(h: i64) {
    aurora_window::imm_r3d_show_joints(h);
}
#[no_mangle]
pub extern "C" fn aurora_r3d_anim_stop_upper(h: i64, fade: f64) {
    aurora_window::imm_r3d_anim_stop_upper(h, fade as f32);
}
#[no_mangle]
pub extern "C" fn aurora_r3d_clip_count(h: i64) -> i64 {
    aurora_window::imm_r3d_clip_count(h)
}
/// `r3d_material_count(h) -> i64`: how many materials a model carries.
#[no_mangle]
pub extern "C" fn aurora_r3d_material_count(h: i64) -> i64 {
    aurora_window::imm_r3d_material_count(h)
}
/// `r3d_material_name(h, i) -> str`: the material name mesh `i` declares, or "".
///
/// `r3d_material_texture` attaches an atlas BY NAME and there was no way to ask
/// what the names were, so binding a new art pack was guesswork - list every
/// name you have ever seen and hope. When none matched, the model drew flat
/// grey, which is indistinguishable from a textured model unless you look.
///
/// # Safety
/// `out` must be valid for writes of two `i64`s.
#[no_mangle]
pub unsafe extern "C" fn aurora_r3d_material_name(out: *mut i64, h: i64, i: i64) {
    let name = aurora_window::imm_r3d_material_name(h, i);
    unsafe { write_str(out, name.into_bytes()) };
}
/// `r3d_clip_duration(h, i) -> f64`: how long clip `i` runs, in seconds.
///
/// For making an animation agree with the rules that own the move. A game whose
/// attack lasts 42 ticks can play a 1.4-second swing at `1.4 / (42/60)` and have
/// the blade land on the frame its hitbox opens. Without this the choice is to
/// play everything at 1.0 and let each clip drift out of sync with its own frame
/// data, or to guess a speed per clip by eye - which is how a jump attack ends
/// up indistinguishable from a heavy.
#[no_mangle]
pub extern "C" fn aurora_r3d_clip_duration(h: i64, i: i64) -> f64 {
    aurora_window::imm_r3d_clip_duration(h, i)
}
/// `r3d_anim_done(h) -> i64`: 1 once the current one-shot clip has played out.
///
/// The question every game asks about a one-shot - is the swing over, is the
/// guard up, is the roll finished - and it had no answer, so callers kept their
/// own timer beside the player's and hoped the two never drifted. The player
/// already knew: it clamps its own time to the clip's duration.
#[no_mangle]
pub extern "C" fn aurora_r3d_anim_done(h: i64) -> i64 {
    aurora_window::imm_r3d_anim_done(h)
}
/// `r3d_anim_done_upper(h) -> i64`: 1 once the upper-body overlay's one-shot has
/// played out.
///
/// The overlay keeps its own clock, so `r3d_anim_done` cannot answer for it. A
/// masked overlay could be started and stopped but never sequenced - a guard
/// built as begin/hold/end on the arms had no way to learn its raise was over.
#[no_mangle]
pub extern "C" fn aurora_r3d_anim_done_upper(h: i64) -> i64 {
    aurora_window::imm_r3d_anim_done_upper(h)
}
/// `r3d_anim_time(h) -> f64`: seconds into the current clip.
#[no_mangle]
pub extern "C" fn aurora_r3d_anim_time(h: i64) -> f64 {
    aurora_window::imm_r3d_anim_time(h)
}
/// `r3d_anim_speed(h) -> f64`: the rate the base clip is actually being played
/// at (1.0 = as authored), or -1.0 for a handle that is not a model.
///
/// The third of the base-layer readers, beside `r3d_anim_clip` (which clip) and
/// `r3d_anim_time` (how far in). A game that retimes a clip to meet a tick
/// budget has to be able to check the result against the renderer rather than
/// against its own arithmetic: an assertion whose expected value comes from the
/// same function it is testing holds whatever that function does.
#[no_mangle]
pub extern "C" fn aurora_r3d_anim_speed(h: i64) -> f64 {
    aurora_window::imm_r3d_anim_speed(h)
}
/// `r3d_root_dx/dy/dz(h) -> f64`: ROOT MOTION - how far the last
/// `r3d_anim_update` moved this character, in metres in the model's own space.
///
/// The distance an attack was authored to cover. Aurora lifts a clip's travel
/// out of the pose at import, so the body animates in place and this is what
/// moves it: add these to the character's position (rotated by the yaw it is
/// drawn with) and the mesh, its collider and its hitbox travel together, over
/// exactly the ground the animator laid down. Without it a lunge is a swing on
/// the spot, and the alternative - a per-clip speed guessed by eye - drifts away
/// from the animation the first time either is retimed.
///
/// Zero before the first update, and zero for a clip authored in place, which
/// every locomotion loop is: a walk cycle's ground speed belongs to the game.
#[no_mangle]
pub extern "C" fn aurora_r3d_root_dx(h: i64) -> f64 {
    aurora_window::imm_r3d_root_dx(h)
}
/// `r3d_root_dy(h) -> f64`: the vertical part of the same delta.
#[no_mangle]
pub extern "C" fn aurora_r3d_root_dy(h: i64) -> f64 {
    aurora_window::imm_r3d_root_dy(h)
}
/// `r3d_root_dz(h) -> f64`: the forward part of the same delta.
#[no_mangle]
pub extern "C" fn aurora_r3d_root_dz(h: i64) -> f64 {
    aurora_window::imm_r3d_root_dz(h)
}
/// `r3d_anim_clip(h) -> i64`: which clip is playing, or -1.
///
/// Completes the set beside `anim_done` and `anim_time`, which both answer
/// about the current clip without ever saying which one it is. A state machine
/// without this has to remember what it last asked for, and that copy goes
/// stale the moment anything else plays a clip on the same model.
#[no_mangle]
pub extern "C" fn aurora_r3d_anim_clip(h: i64) -> i64 {
    aurora_window::imm_r3d_anim_clip(h)
}
/// `r3d_mesh_bytes() -> i64`: bytes of GPU vertex + index buffer the live
/// meshes hold.
///
/// What the loaded art COSTS, so a game can assert on it. Standing a room up is
/// the largest allocation a game makes, and the only symptom of getting it
/// wrong is memory: a scene that uploads forty copies of one atlas renders
/// exactly like a scene that uploads one, and no still frame, timing or rule
/// assertion can tell them apart.
#[no_mangle]
pub extern "C" fn aurora_r3d_mesh_bytes() -> i64 {
    aurora_window::imm_r3d_mesh_bytes()
}
/// `r3d_mesh_count() -> i64`: how many meshes are live.
#[no_mangle]
pub extern "C" fn aurora_r3d_mesh_count() -> i64 {
    aurora_window::imm_r3d_mesh_count()
}
/// `r3d_mesh_slots() -> i64`: mesh slots ever allocated, live or free.
///
/// A load/free workload keeps this flat while `r3d_mesh_count` cycles. A gap
/// that grows is the signature of a leak.
#[no_mangle]
pub extern "C" fn aurora_r3d_mesh_slots() -> i64 {
    aurora_window::imm_r3d_mesh_slots()
}
/// `r3d_texture_bytes() -> i64`: bytes of GPU texture held, counted ONCE per
/// distinct image rather than once per material that names it.
#[no_mangle]
pub extern "C" fn aurora_r3d_texture_bytes() -> i64 {
    aurora_window::imm_r3d_texture_bytes()
}
/// `r3d_texture_count() -> i64`: how many distinct images are uploaded.
#[no_mangle]
pub extern "C" fn aurora_r3d_texture_count() -> i64 {
    aurora_window::imm_r3d_texture_count()
}
/// `r3d_anim_blend_clip(h) -> i64`: the SECOND clip of a sustained base blend,
/// or -1 when the base layer is a single clip.
#[no_mangle]
pub extern "C" fn aurora_r3d_anim_blend_clip(h: i64) -> i64 {
    aurora_window::imm_r3d_anim_blend_clip(h)
}
/// `r3d_anim_blend_weight(h) -> f64`: how far through a sustained base blend the
/// base layer is (0 = entirely the first clip, 1 = entirely the second), or -1
/// when it is not blending.
///
/// A walking character IS a two-clip blend, and its WEIGHT is what the body
/// looks like. Without this a caller can read which clips are loaded and how far
/// each clock has run and still not see a body being thrown between a standing
/// pose and a sprint on alternate frames - no clip changes, no clock jumps, and
/// the character is unwatchable.
#[no_mangle]
pub extern "C" fn aurora_r3d_anim_blend_weight(h: i64) -> f64 {
    aurora_window::imm_r3d_anim_blend_weight(h) as f64
}
/// `r3d_anim_clip_upper(h) -> i64`: which clip the upper-body overlay is
/// playing, or -1 when no overlay is running.
#[no_mangle]
pub extern "C" fn aurora_r3d_anim_clip_upper(h: i64) -> i64 {
    aurora_window::imm_r3d_anim_clip_upper(h)
}
/// `r3d_clip_name(h, i) -> str`: the asset's own name for clip `i`, or "" for a
/// stale handle or an out-of-range index.
///
/// # Safety
/// `out` must be valid for writes of two `i64`s.
#[no_mangle]
pub unsafe extern "C" fn aurora_r3d_clip_name(out: *mut i64, h: i64, i: i64) {
    let name = aurora_window::imm_r3d_clip_name(h, i);
    unsafe { write_str(out, name.into_bytes()) };
}
/// `r3d_joint_index(h, name) -> i64`: the index of the joint called `name`, or
/// -1. Lets a game attach props and hitboxes by BONE NAME instead of by a magic
/// index that silently moves when a rig is re-exported.
///
/// # Safety
/// `ptr` must point to `len` initialized bytes.
#[no_mangle]
pub unsafe extern "C" fn aurora_r3d_joint_index(h: i64, ptr: *const u8, len: i64) -> i64 {
    let s = unsafe { std::slice::from_raw_parts(ptr, len.max(0) as usize) };
    aurora_window::imm_r3d_joint_index(h, &String::from_utf8_lossy(s))
}
/// `r3d_joint_name(h, i) -> str`: the name of joint `i`, or "".
///
/// # Safety
/// `out` must be valid for writes of two `i64`s.
#[no_mangle]
pub unsafe extern "C" fn aurora_r3d_joint_name(out: *mut i64, h: i64, i: i64) {
    let name = aurora_window::imm_r3d_joint_name(h, i);
    unsafe { write_str(out, name.into_bytes()) };
}
/// `r3d_clip_index(h, name) -> i64`: the index of the clip called `name`, or -1.
/// Lets a game bind animations by NAME instead of by a magic index that silently
/// selects the wrong motion when a model is re-exported.
///
/// # Safety
/// `ptr` must point to `len` initialized bytes.
#[no_mangle]
pub unsafe extern "C" fn aurora_r3d_clip_index(h: i64, ptr: *const u8, len: i64) -> i64 {
    let s = unsafe { std::slice::from_raw_parts(ptr, len.max(0) as usize) };
    aurora_window::imm_r3d_clip_index(h, &String::from_utf8_lossy(s))
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
    // The frame is over: edge snapshot and this frame's delta both roll. See
    // `input_step`, which is the one place a frame ends.
    //
    // BEFORE the present, because presenting is what PUMPS the window's event
    // loop. Snapshotting after it records this frame's fresh key presses as
    // "already seen", and `input_pressed` never fires for a real keyboard - the
    // bug that left a shipped game unable to attack or drink while walking and
    // guarding worked fine.
    aurora_input_step();
    let open = aurora_window::imm_r3d_present(&rgba, w, h);
    if open {
        1
    } else {
        0
    }
}

/// Capture the queued 3D scene to a PNG at the framebuffer's size (headless
/// only), with the HUD framebuffer composited on top. Returns 1 on success.
///
/// # Safety
/// `ptr` must point to `len` initialized bytes.
#[no_mangle]
pub unsafe extern "C" fn aurora_r3d_capture(ptr: *const u8, len: i64) -> i64 {
    let path = {
        let s = unsafe { std::slice::from_raw_parts(ptr, len.max(0) as usize) };
        String::from_utf8_lossy(s).into_owned()
    };
    let (rgba, w, h) = FB.with(|fb| {
        fb.borrow()
            .as_ref()
            .map(|f| (f.rgba(), f.width(), f.height()))
            .unwrap_or((Vec::new(), 0, 0))
    });
    let (ow, oh) = if w > 0 && h > 0 { (w, h) } else { (1280, 720) };
    aurora_window::imm_r3d_capture(&path, &rgba, w, h, ow, oh)
}

/// Like `r3d_capture` but at an explicit output resolution.
///
/// # Safety
/// `ptr` must point to `len` initialized bytes.
#[no_mangle]
pub unsafe extern "C" fn aurora_r3d_capture_size(
    ptr: *const u8,
    len: i64,
    ow: i64,
    oh: i64,
) -> i64 {
    let path = {
        let s = unsafe { std::slice::from_raw_parts(ptr, len.max(0) as usize) };
        String::from_utf8_lossy(s).into_owned()
    };
    let (rgba, w, h) = FB.with(|fb| {
        fb.borrow()
            .as_ref()
            .map(|f| (f.rgba(), f.width(), f.height()))
            .unwrap_or((Vec::new(), 0, 0))
    });
    aurora_window::imm_r3d_capture(&path, &rgba, w, h, ow.max(16) as u32, oh.max(16) as u32)
}

/// Input injection builtins: scripted input indistinguishable from a player.
#[no_mangle]
pub extern "C" fn aurora_inject_key(code: i64, down: i64) {
    aurora_window::imm_inject_key(code.max(0) as u32, down != 0);
}
#[no_mangle]
pub extern "C" fn aurora_inject_mouse_move(dx: f64, dy: f64) {
    aurora_window::imm_inject_mouse_move(dx, dy);
}
#[no_mangle]
pub extern "C" fn aurora_inject_mouse_pos(x: i64, y: i64) {
    aurora_window::imm_inject_mouse_pos(x, y);
}
#[no_mangle]
pub extern "C" fn aurora_inject_mouse_button(b: i64, down: i64) {
    aurora_window::imm_inject_mouse_button(b.max(0) as u32, down != 0);
}
#[no_mangle]
pub extern "C" fn aurora_inject_scroll(dy: f64) {
    aurora_window::imm_inject_scroll(dy);
}
#[no_mangle]
pub extern "C" fn aurora_inject_char(c: i64) {
    aurora_window::imm_inject_char(c.max(0) as u32);
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
    aurora_window::imm_r3d_sky(
        on, tr as f32, tg as f32, tb as f32, hr as f32, hg as f32, hb as f32,
    );
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
pub extern "C" fn aurora_r3d_viewmodel(on: i64) {
    aurora_window::imm_r3d_viewmodel(on);
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
pub extern "C" fn aurora_r3d_point_light(
    x: f64,
    y: f64,
    z: f64,
    r: f64,
    g: f64,
    b: f64,
    range: f64,
    intensity: f64,
) {
    aurora_window::imm_r3d_point_light(
        x as f32,
        y as f32,
        z as f32,
        r as f32,
        g as f32,
        b as f32,
        range as f32,
        intensity as f32,
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
pub extern "C" fn aurora_r3d_debug_line(
    ax: f64,
    ay: f64,
    az: f64,
    bx: f64,
    by: f64,
    bz: f64,
    r: f64,
    g: f64,
    b: f64,
) {
    aurora_window::imm_r3d_debug_line(
        ax as f32, ay as f32, az as f32, bx as f32, by as f32, bz as f32, r as f32, g as f32,
        b as f32,
    );
}
/// Draw a model's skeleton as debug bone lines (headless rig/hitbox audits).
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub extern "C" fn aurora_r3d_debug_skeleton(
    handle: i64,
    px: f64,
    py: f64,
    pz: f64,
    yaw: f64,
    scale: f64,
    r: f64,
    g: f64,
    b: f64,
) {
    aurora_window::imm_r3d_debug_skeleton(
        handle,
        px as f32,
        py as f32,
        pz as f32,
        yaw as f32,
        scale as f32,
        r as f32,
        g as f32,
        b as f32,
    );
}
#[no_mangle]
pub extern "C" fn aurora_r3d_frustum_cull(on: i64) {
    aurora_window::imm_r3d_frustum_cull(on);
}
#[no_mangle]
pub extern "C" fn aurora_r3d_screen_x(wx: f64, wy: f64, wz: f64) -> f64 {
    let (x, _, vis) = aurora_window::imm_r3d_world_to_screen(wx as f32, wy as f32, wz as f32);
    if vis {
        x as f64
    } else {
        -1.0
    }
}
#[no_mangle]
pub extern "C" fn aurora_r3d_screen_y(wx: f64, wy: f64, wz: f64) -> f64 {
    let (_, y, vis) = aurora_window::imm_r3d_world_to_screen(wx as f32, wy as f32, wz as f32);
    if vis {
        y as f64
    } else {
        -1.0
    }
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
    // A LIST of codes per action, not one.
    //
    // A game that supports a controller has to answer to the keyboard AND the
    // pad at the same time - nobody should have to open a rebind screen before
    // their controller does anything - and "the action fires on any of these"
    // is the only shape that expresses it. One code per action forced a choice
    // between the two, or a second parallel binding table beside this one,
    // which is the same fact in two places and the defect this project has the
    // most history with.
    static BINDINGS: RefCell<std::collections::HashMap<i64, Vec<i64>>> =
        RefCell::new(std::collections::HashMap::new());
    // When set, the bind-layer reads (input_down / input_axis) all report "not held",
    // so a game can freeze player actions in one call (e.g. a pause overlay) without
    // touching the raw mouse used by menus.
    static INPUT_SUPPRESS: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    // Which input codes were held at the last frame boundary, one bit each, so
    // `input_pressed` and `input_released` answer about the EDGE and not the
    // level.
    //
    // Codes rather than actions: rebinding an action while its old key is held
    // must not manufacture a press on the new one. And EVERY code, not only the
    // bound ones - the first version snapshotted just what was bound, so a key
    // that was already down when an action moved onto it read as freshly
    // pressed, which is the exact case a rebind screen produces.
    //
    // A bitmask rather than a set: the per-frame snapshot costs no allocation
    // and the lookup is a shift. It was one `u128` and the gamepad codes took
    // the space past 128, so the width is DERIVED from the code space now -
    // a snapshot narrower than the codes it must hold would silently stop
    // reporting edges for everything above the cut.
    static INPUT_PREV: std::cell::Cell<[u64; INPUT_WORDS]> =
        const { std::cell::Cell::new([0; INPUT_WORDS]) };
}

/// How many 64-bit words the edge snapshot needs to cover every input code.
const INPUT_WORDS: usize = (INPUT_CODE_MAX as usize).div_ceil(64);

fn bit_get(mask: &[u64; INPUT_WORDS], code: i64) -> bool {
    if !(0..INPUT_CODE_MAX).contains(&code) {
        return false;
    }
    let c = code as usize;
    mask[c / 64] >> (c % 64) & 1 == 1
}

fn bit_set(mask: &mut [u64; INPUT_WORDS], code: i64) {
    if !(0..INPUT_CODE_MAX).contains(&code) {
        return;
    }
    let c = code as usize;
    mask[c / 64] |= 1u64 << (c % 64);
}

/// The first mouse-button input code. Below it a code is a KEY (the
/// `code_to_key` codes); at it and above, `code - MOUSE_CODE_BASE` is the button
/// index `mouse_button` takes.
///
/// Named because this rule was being re-derived at every site that had to tell a
/// key from a button - including in game code, where getting it wrong pressed
/// nothing at all and ran a whole test fight with no guard up. Anything that
/// needs it calls `code_is_down` or `inject_action` instead of writing 100.
const MOUSE_CODE_BASE: i64 = 100;

/// The mouse buttons `input_code`/`input_name` know, indexed by button number -
/// the order `mouse_button` and `inject_mouse_button` use.
const MOUSE_NAMES: [&str; 5] = [
    "MouseLeft",
    "MouseRight",
    "MouseMiddle",
    "MouseBack",
    "MouseForward",
];

/// The first gamepad input code. Above it, `code - PAD_CODE_BASE` indexes
/// `PAD_INPUTS`.
const PAD_CODE_BASE: i64 = 200;

/// What a gamepad input code reads.
#[derive(Clone, Copy)]
enum PadSource {
    /// A digital button.
    Button(usize),
    /// ONE HALF of an analog axis: which axis, and the sign that counts.
    ///
    /// A stick axis is bipolar and an action is not, so a stick is four named
    /// directions rather than two signed axes. `PadLeftStickUp` gives 0 when
    /// the stick is pulled back, which is what lets it be bound to "forward"
    /// beside the W key and behave like a better W key.
    Half(usize, f64),
}

/// Every gamepad input by name, in code order.
///
/// The names are the Xbox letters because that is what is printed on the pad in
/// the player's hands and what a prompt has to say; underneath they are
/// POSITIONS (`BTN_SOUTH`), so the same code is the bottom face button on a
/// Nintendo pad rather than the one labelled A there.
///
/// One table, so `input_code`, `input_name`, `code_value` and `inject_action`
/// cannot disagree about what a code is. The keyboard learned this the hard
/// way: a soulslike shipped with the dodge on the LEFT ARROW because the game
/// wrote a number down and the test read the same number back.
const PAD_INPUTS: [(&str, PadSource); 24] = [
    ("PadA", PadSource::Button(aurora_window::pad::BTN_SOUTH)),
    ("PadB", PadSource::Button(aurora_window::pad::BTN_EAST)),
    ("PadX", PadSource::Button(aurora_window::pad::BTN_WEST)),
    ("PadY", PadSource::Button(aurora_window::pad::BTN_NORTH)),
    (
        "PadLB",
        PadSource::Button(aurora_window::pad::BTN_LEFT_BUMPER),
    ),
    (
        "PadRB",
        PadSource::Button(aurora_window::pad::BTN_RIGHT_BUMPER),
    ),
    ("PadBack", PadSource::Button(aurora_window::pad::BTN_SELECT)),
    ("PadStart", PadSource::Button(aurora_window::pad::BTN_START)),
    (
        "PadLS",
        PadSource::Button(aurora_window::pad::BTN_LEFT_STICK),
    ),
    (
        "PadRS",
        PadSource::Button(aurora_window::pad::BTN_RIGHT_STICK),
    ),
    ("PadUp", PadSource::Button(aurora_window::pad::BTN_DPAD_UP)),
    (
        "PadDown",
        PadSource::Button(aurora_window::pad::BTN_DPAD_DOWN),
    ),
    (
        "PadLeft",
        PadSource::Button(aurora_window::pad::BTN_DPAD_LEFT),
    ),
    (
        "PadRight",
        PadSource::Button(aurora_window::pad::BTN_DPAD_RIGHT),
    ),
    (
        "PadLT",
        PadSource::Half(aurora_window::pad::AXIS_LEFT_TRIGGER, 1.0),
    ),
    (
        "PadRT",
        PadSource::Half(aurora_window::pad::AXIS_RIGHT_TRIGGER, 1.0),
    ),
    (
        "PadLeftStickUp",
        PadSource::Half(aurora_window::pad::AXIS_LEFT_Y, 1.0),
    ),
    (
        "PadLeftStickDown",
        PadSource::Half(aurora_window::pad::AXIS_LEFT_Y, -1.0),
    ),
    (
        "PadLeftStickRight",
        PadSource::Half(aurora_window::pad::AXIS_LEFT_X, 1.0),
    ),
    (
        "PadLeftStickLeft",
        PadSource::Half(aurora_window::pad::AXIS_LEFT_X, -1.0),
    ),
    (
        "PadRightStickUp",
        PadSource::Half(aurora_window::pad::AXIS_RIGHT_Y, 1.0),
    ),
    (
        "PadRightStickDown",
        PadSource::Half(aurora_window::pad::AXIS_RIGHT_Y, -1.0),
    ),
    (
        "PadRightStickRight",
        PadSource::Half(aurora_window::pad::AXIS_RIGHT_X, 1.0),
    ),
    (
        "PadRightStickLeft",
        PadSource::Half(aurora_window::pad::AXIS_RIGHT_X, -1.0),
    ),
];

/// One past the highest input code. 0..65 are keyboard, then the mouse buttons,
/// then the gamepad; the gaps between cost a few bits and nothing else.
const INPUT_CODE_MAX: i64 = PAD_CODE_BASE + PAD_INPUTS.len() as i64;

/// End the frame: advance the input edge snapshot (what is held now becomes "was
/// held" for the next frame's `input_pressed` / `input_released`) and spend this
/// frame's delta, so the next `frame_dt` measures a fresh one.
///
/// This is THE frame boundary, and the only one. Called automatically by
/// `window_present` and `r3d_present`, because that is where a frame ends and a
/// game with a window never has to remember. Headless programs that inject input
/// and step the simulation without presenting have no frame boundary of their
/// own, and call this where theirs is - which is also what keeps `frame_dt` from
/// freezing in a loop that never presents.
///
/// The snapshot records the RAW key state even while input is suppressed. A pause
/// menu opened with attack held and closed with attack still held must not fire
/// an attack on the way out - which it would if suppression made the snapshot
/// read "not held".
#[no_mangle]
pub extern "C" fn aurora_input_step() {
    // SNAPSHOT FIRST, then take new input. The order is the whole correctness of
    // `input_pressed`, and having it backwards made every edge action in a
    // windowed game dead.
    //
    // `prev` has to be the state THIS frame's logic actually saw. Polling the
    // pads (and, in `r3d_present`, pumping the window's key events) before the
    // snapshot writes the new press into `prev` in the same breath that first
    // makes it visible - so `now && !was` is never true and the press is
    // consumed before a single line of game code can read it.
    //
    // The symptom in a real game is exact and was reported from play: held
    // actions work - moving, sprinting, guarding - and every EDGE action is
    // dead. No attack, no flask, no jump, no lock-on. Scripted checks could not
    // see it because they inject input and call this themselves, so their press
    // lands between two snapshots rather than inside one.
    let mut held = [0u64; INPUT_WORDS];
    let mut c = 0;
    while c < INPUT_CODE_MAX {
        if code_is_down(c) {
            bit_set(&mut held, c);
        }
        c += 1;
    }
    INPUT_PREV.with(|p| p.set(held));

    // The gamepads are read HERE, at the one frame boundary, and nowhere else -
    // now AFTER the snapshot, so a button that goes down on this poll is an edge
    // for the next frame rather than a press nobody was told about.
    //
    // Polling them lazily inside each read would let a pad change state halfway
    // through a frame's logic - the button that was down when movement was
    // resolved is up by the time the attack is.
    // Through `poll_pads`, which reads the hardware only when this process
    // should be receiving it - see its comment. Calling `pad::poll` here read
    // whatever the player was doing on a pad in ANOTHER game.
    aurora_window::poll_pads();
    end_frame_dt();
}

/// Whether an action went down THIS frame (1) or not (0).
///
/// The distinction every action game needs and every one of them re-implements:
/// a held button is one press, not sixty. Without it a game keeps its own
/// `was_down` beside each call site, and those copies drift - the flask that
/// empties five charges in a second, the menu that scrolls the whole list on one
/// tap. It is a property of the input layer, so it lives here.
#[no_mangle]
pub extern "C" fn aurora_input_pressed(action: i64) -> i64 {
    input_edge(action, true)
}

/// Whether an action came up THIS frame (1) or not (0).
#[no_mangle]
pub extern "C" fn aurora_input_released(action: i64) -> i64 {
    input_edge(action, false)
}

fn input_edge(action: i64, want_press: bool) -> i64 {
    if INPUT_SUPPRESS.with(|s| s.get()) {
        return 0;
    }
    // The edge is the ACTION's, over every code bound to it: an action held on
    // the keyboard and then also pressed on the pad has not been pressed twice.
    // Per-code edges would fire a second attack for the second input.
    let prev = INPUT_PREV.with(|p| p.get());
    let mut was = false;
    let mut now = false;
    for_each_binding(action, |code| {
        was |= bit_get(&prev, code);
        now |= code_is_down(code);
    });
    if want_press {
        (now && !was) as i64
    } else {
        (was && !now) as i64
    }
}

/// Run `f` over every input code bound to `action`, in bind order.
fn for_each_binding(action: i64, mut f: impl FnMut(i64)) {
    BINDINGS.with(|b| {
        if let Some(codes) = b.borrow().get(&action) {
            for c in codes {
                f(*c);
            }
        }
    });
}

/// Suppress (1) or restore (0) all bound-action input. While suppressed, every
/// `input_down`/`input_axis` reads as zero; the raw mouse/keyboard queries are
/// untouched so menus still work.
#[no_mangle]
pub extern "C" fn aurora_input_suppress(on: i64) {
    INPUT_SUPPRESS.with(|s| s.set(on != 0));
}

/// HOW MUCH an input is giving, in 0..1.
///
/// The primitive the whole layer is built on, and the reason analog movement
/// needs no second mechanism. A key or a button answers 1.0 or 0.0; a trigger
/// answers its pull; a stick direction answers its lean. `input_axis` subtracts
/// two of these, so binding a stick direction as an alternate for "forward"
/// makes every existing `input_axis` call analog with no change at the call
/// site and no branch anywhere asking whether a controller is plugged in.
fn code_value(code: i64) -> f64 {
    if code < 0 {
        return 0.0;
    }
    if code >= PAD_CODE_BASE {
        let Some((_, src)) = PAD_INPUTS.get((code - PAD_CODE_BASE) as usize) else {
            return 0.0;
        };
        return match src {
            PadSource::Button(b) => aurora_window::pad::any_button(*b) as i64 as f64,
            // The half that is not being pushed gives nothing, rather than a
            // negative - an action is not bipolar.
            PadSource::Half(a, sign) => (aurora_window::pad::any_axis(*a) * sign).max(0.0),
        };
    }
    if code >= MOUSE_CODE_BASE {
        return aurora_window::imm_mouse_button((code - MOUSE_CODE_BASE) as u32) as i64 as f64;
    }
    aurora_window::imm_key_down(code as u32) as i64 as f64
}

/// Whether an input counts as HELD - a digital read of an analog world.
///
/// The threshold is well past the deadzone, so resting a thumb on the stick
/// does not fire an action bound to it and a trigger bound to attack needs a
/// deliberate pull. A key is 1.0 and is unaffected.
fn code_is_down(code: i64) -> bool {
    code_value(code) >= aurora_window::pad::DIGITAL_THRESHOLD
}

/// `input_code(name) -> i64`: the input code for a NAMED key or mouse button, or
/// -1 if there is no such input.
///
/// Keys are the W3C `KeyboardEvent.code` names winit's `KeyCode` already uses -
/// "Space", "KeyW", "ShiftLeft", "Tab" - answered from `code_to_key` itself, so
/// this cannot disagree with the key the code presses. Mouse buttons are
/// "MouseLeft"/"MouseRight"/"MouseMiddle"/"MouseBack"/"MouseForward".
///
/// It exists so that no program ever writes an input code down. Every one that
/// did got one wrong: a soulslike shipped with `KEY_SPACE = 0`, which is the
/// LEFT ARROW, and the dodge roll could not be rolled. Nothing caught it,
/// because the test asked `input_binding` for the code and pressed the same
/// wrong number back.
///
/// # Safety
/// `ptr` must point to `len` initialized bytes.
#[no_mangle]
pub unsafe extern "C" fn aurora_input_code(ptr: *const u8, len: i64) -> i64 {
    let name = unsafe { arg_str(ptr, len) };
    if let Some(b) = MOUSE_NAMES.iter().position(|n| *n == name) {
        return MOUSE_CODE_BASE + b as i64;
    }
    if let Some(b) = PAD_INPUTS.iter().position(|(n, _)| *n == name) {
        return PAD_CODE_BASE + b as i64;
    }
    aurora_window::imm_key_code(&name)
        .map(|c| c as i64)
        .unwrap_or(-1)
}

/// `input_name(code) -> str`: the inverse of `input_code`, and "" for a code
/// that names no input - including -1, which is what `input_binding` answers for
/// an unbound action.
///
/// This is how a test states what a control IS: `input_name(input_binding(a))`
/// is "Space" or it is not, and the answer comes from the engine's own table
/// rather than from the number the program bound.
///
/// # Safety
/// `out` must be valid for writes of two `i64`s.
#[no_mangle]
pub unsafe extern "C" fn aurora_input_name(out: *mut i64, code: i64) {
    let name = if code < 0 {
        ""
    } else if code >= PAD_CODE_BASE {
        PAD_INPUTS
            .get((code - PAD_CODE_BASE) as usize)
            .map(|(n, _)| *n)
            .unwrap_or("")
    } else if code >= MOUSE_CODE_BASE {
        MOUSE_NAMES
            .get((code - MOUSE_CODE_BASE) as usize)
            .copied()
            .unwrap_or("")
    } else {
        u32::try_from(code)
            .ok()
            .and_then(aurora_window::imm_key_name)
            .unwrap_or("")
    };
    unsafe { write_str(out, name.as_bytes().to_vec()) };
}

/// `inject_action(action, down)`: press or release whatever input an ACTION is
/// bound to, key or mouse button. A no-op for an unbound action.
///
/// The one place the key-or-button decision is made for injection, matching
/// `code_is_down` on the reading side. A test that hand-rolled it instead sent a
/// KEY press for an action bound to the right mouse button - not an error, just
/// a no-op - and then ran an entire fight with the guard never up, reporting the
/// player dying as a balance problem.
#[no_mangle]
pub extern "C" fn aurora_inject_action(action: i64, down: i64) {
    let code = aurora_input_binding(action);
    if code < 0 {
        return;
    }
    inject_code(code, down != 0);
}

/// `inject_input(code, value)`: drive a named input to a VALUE, 0..1.
///
/// The analog counterpart of `inject_action`, and the one a controller test
/// wants: `inject_input(input_code("PadLeftStickUp"), 0.33)` is a stick pushed
/// a third of the way, which no press can express and which is the whole thing
/// analog movement has to be checked against.
///
/// Through the input CODE, so a test never writes an axis index down - the same
/// rule that stopped a soulslike shipping its dodge on the Left Arrow. A
/// digital input takes the value as a press past the same threshold a real one
/// is read at, so one call drives either kind.
#[no_mangle]
pub extern "C" fn aurora_inject_input(code: i64, value: f64) {
    if code < 0 {
        return;
    }
    if code >= PAD_CODE_BASE {
        let Some((_, src)) = PAD_INPUTS.get((code - PAD_CODE_BASE) as usize) else {
            return;
        };
        match src {
            PadSource::Button(b) => aurora_window::pad::inject_button(
                0,
                *b,
                value >= aurora_window::pad::DIGITAL_THRESHOLD,
            ),
            PadSource::Half(a, sign) => {
                aurora_window::pad::inject_axis(0, *a, value.clamp(0.0, 1.0) * *sign)
            }
        }
        return;
    }
    inject_code(code, value >= aurora_window::pad::DIGITAL_THRESHOLD);
}

/// Press or release a raw input code, whatever kind it is.
///
/// The one place the key-or-button-or-pad decision is made for injection,
/// matching `code_value` on the reading side. A test that hand-rolled it sent a
/// KEY press for an action bound to the right mouse button - not an error, just
/// a no-op - and then ran an entire fight with the guard never up, reporting
/// the player dying as a balance problem.
fn inject_code(code: i64, down: bool) {
    if code < 0 {
        return;
    }
    if code >= PAD_CODE_BASE {
        let Some((_, src)) = PAD_INPUTS.get((code - PAD_CODE_BASE) as usize) else {
            return;
        };
        // Pad 0: a test driving "the controller" means the one somebody would
        // be holding.
        match src {
            PadSource::Button(b) => aurora_window::pad::inject_button(0, *b, down),
            // Injected to FULL lean, so a digital press of an analog input is
            // the whole input rather than something that happens to clear the
            // threshold - `inject_pad_axis` is there for a partial one.
            PadSource::Half(a, sign) => {
                aurora_window::pad::inject_axis(0, *a, if down { *sign } else { 0.0 })
            }
        }
        return;
    }
    if code >= MOUSE_CODE_BASE {
        aurora_window::imm_inject_mouse_button((code - MOUSE_CODE_BASE) as u32, down);
    } else {
        aurora_window::imm_inject_key(code as u32, down);
    }
}

/// Bind an action id to an input code, REPLACING everything it was bound to.
///
/// The primary binding, and the one `input_binding` and `inject_action` speak
/// about. Use `input_bind_also` to add a second input that fires the same
/// action - a pad button beside a key - without taking the first one away.
#[no_mangle]
pub extern "C" fn aurora_input_bind(action: i64, code: i64) {
    BINDINGS.with(|b| {
        b.borrow_mut().insert(action, vec![code]);
    });
}

/// Add another input that fires `action`, keeping the ones already bound.
///
/// Idempotent: binding the same code twice leaves one. Without that, a setup
/// function called on every room transition would grow the list forever, and
/// `input_binding_count` - which a rebind screen lists from - would fill with
/// duplicates of the same key.
#[no_mangle]
pub extern "C" fn aurora_input_bind_also(action: i64, code: i64) {
    BINDINGS.with(|b| {
        let mut b = b.borrow_mut();
        let codes = b.entry(action).or_default();
        if !codes.contains(&code) {
            codes.push(code);
        }
    });
}

/// The PRIMARY input code bound to an action, or -1 if unbound.
#[no_mangle]
pub extern "C" fn aurora_input_binding(action: i64) -> i64 {
    aurora_input_binding_at(action, 0)
}

/// The `i`-th input code bound to an action, or -1 past the end. With
/// `input_binding_count`, this is what a controls screen lists from.
#[no_mangle]
pub extern "C" fn aurora_input_binding_at(action: i64, i: i64) -> i64 {
    if i < 0 {
        return -1;
    }
    BINDINGS.with(|b| {
        b.borrow()
            .get(&action)
            .and_then(|c| c.get(i as usize))
            .copied()
            .unwrap_or(-1)
    })
}

/// How many inputs fire this action.
#[no_mangle]
pub extern "C" fn aurora_input_binding_count(action: i64) -> i64 {
    BINDINGS.with(|b| b.borrow().get(&action).map(|c| c.len()).unwrap_or(0) as i64)
}

/// Whether ANY input bound to this action is currently held (1) or not (0).
#[no_mangle]
pub extern "C" fn aurora_input_down(action: i64) -> i64 {
    if INPUT_SUPPRESS.with(|s| s.get()) {
        return 0;
    }
    let mut down = false;
    for_each_binding(action, |code| down |= code_is_down(code));
    down as i64
}

/// HOW MUCH this action is being asked for, in 0..1.
///
/// 1.0 for a held key, the pull for a trigger, the lean for a stick direction -
/// and the LARGEST of them when an action has several inputs bound, so a hand
/// resting on the keyboard cannot cap what the stick is asking for.
#[no_mangle]
pub extern "C" fn aurora_input_value(action: i64) -> f64 {
    if INPUT_SUPPRESS.with(|s| s.get()) {
        return 0.0;
    }
    let mut v = 0.0f64;
    for_each_binding(action, |code| v = v.max(code_value(code)));
    v
}

/// A -1..+1 axis from two opposing actions (e.g. back/forward).
///
/// ANALOG, and that is the whole design: it subtracts two `input_value`s rather
/// than two held/not-held reads, so a key answers the 1.0 it always did and a
/// stick pushed a third of the way answers 0.33. Every caller that already
/// asked for a movement axis became analog the moment a stick direction was
/// bound beside the key, without one of them changing.
#[no_mangle]
pub extern "C" fn aurora_input_axis(neg: i64, pos: i64) -> f64 {
    aurora_input_value(pos) - aurora_input_value(neg)
}

// --- gamepads, raw -----------------------------------------------------------
//
// Below the action layer. A game should bind `input_code("PadA")` and read
// actions like it reads keys; these are for the cases the action layer cannot
// express - "is a controller plugged in", telling pad 1 from pad 2 for local
// co-op, and shaking the thing in the player's hands.

/// `pad_count()`: how many controllers are connected.
#[no_mangle]
pub extern "C" fn aurora_pad_count() -> i64 {
    aurora_window::pad::count() as i64
}

/// `pad_connected(i)`: is controller `i` plugged in?
///
/// What a HUD asks to decide whether to draw "A" or "Space" on a prompt. A game
/// that guesses from a config file tells half its players the wrong button.
#[no_mangle]
pub extern "C" fn aurora_pad_connected(i: i64) -> i64 {
    if i < 0 {
        return 0;
    }
    aurora_window::pad::connected(i as usize) as i64
}

/// `pad_button(i, b)`: is button `b` held on controller `i`?
#[no_mangle]
pub extern "C" fn aurora_pad_button(i: i64, b: i64) -> i64 {
    if i < 0 || b < 0 {
        return 0;
    }
    aurora_window::pad::button(i as usize, b as usize) as i64
}

/// `pad_axis(i, a)`: axis `a` of controller `i`, deadzoned - sticks in -1..1,
/// triggers in 0..1.
///
/// The deadzone is applied HERE rather than by the game, radially, with the
/// remainder rescaled to the full range. Every game that does its own gets a
/// character that drifts at rest or cannot walk slowly, and most do it per-axis,
/// which lets a slow diagonal through the corner of a square deadzone.
#[no_mangle]
pub extern "C" fn aurora_pad_axis(i: i64, a: i64) -> f64 {
    if i < 0 || a < 0 {
        return 0.0;
    }
    aurora_window::pad::axis(i as usize, a as usize)
}

/// `pad_rumble(i, strong, weak, seconds)`: shake controller `i`. `strong` is the
/// heavy low-frequency motor and `weak` the light one, both 0..1. Answers 1 if
/// the pad actually shook - a pad with no motors, or none plugged in, is 0.
///
/// The duration goes to the driver, so nothing in the game has to remember to
/// stop it: a motor set spinning keeps spinning, and a design that depends on a
/// later frame arriving leaves the pad buzzing when that frame does not.
#[no_mangle]
pub extern "C" fn aurora_pad_rumble(i: i64, strong: f64, weak: f64, seconds: f64) -> i64 {
    if i < 0 {
        return 0;
    }
    aurora_window::pad::rumble(i as usize, strong, weak, seconds) as i64
}

/// `inject_pad_button(i, b, down)`: press or release a pad button, writing the
/// same state a real controller writes.
///
/// The pad half of `inject_key`, and the basis of every headless controller
/// test. `poll` leaves a slot alone when no physical device is at it, so an
/// injected pad survives on a build machine with nothing plugged in.
#[no_mangle]
pub extern "C" fn aurora_inject_pad_button(i: i64, b: i64, down: i64) {
    if i < 0 || b < 0 {
        return;
    }
    aurora_window::pad::inject_button(i as usize, b as usize, down != 0);
}

/// `inject_pad_axis(i, a, v)`: set an axis, already deadzoned. This is how a
/// test asks for HALF forward rather than for forward.
#[no_mangle]
pub extern "C" fn aurora_inject_pad_axis(i: i64, a: i64, v: f64) {
    if i < 0 || a < 0 {
        return;
    }
    aurora_window::pad::inject_axis(i as usize, a as usize, v);
}

/// `inject_pad_disconnect(i)`: unplug a pad and forget its state, for a test
/// that needs "no controller" - including the one that proves a prompt says
/// "Space" when nothing is plugged in.
#[no_mangle]
pub extern "C" fn aurora_inject_pad_disconnect(i: i64) {
    if i < 0 {
        return;
    }
    aurora_window::pad::inject_disconnect(i as usize);
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

/// Allocate a zeroed, LEAKED `f32` blob of `len` floats and return its raw pointer (as i64),
/// usable with `f32_load`/`f32_store` and as a `sim_step` state/input blob. The allocation lives
/// for the whole program on purpose - it's how a game gives a non-networked actor (e.g. a bot) its
/// own persistent sim state, so the SAME sim_step that moves players can move it too.
#[no_mangle]
pub extern "C" fn aurora_f32_blob(len: i64) -> i64 {
    let n = if len < 0 { 0 } else { len as usize };
    let mut v = vec![0.0f32; n];
    let ptr = v.as_mut_ptr() as i64;
    std::mem::forget(v);
    ptr
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
/// Inverse trig, so an ANGLE can be measured rather than eyeballed.
///
/// A game needs these constantly - the angle between a blade and a forearm, a
/// facing against a target, a slope. Their absence is why a weapon grip was
/// judged from renders instead of computed, and got the wrong answer repeatedly.
/// Inputs are clamped to the valid domain, so a dot product that lands at
/// 1.0000001 through rounding returns 0 rather than NaN.
#[no_mangle]
pub extern "C" fn aurora_acos(x: f64) -> f64 {
    x.clamp(-1.0, 1.0).acos()
}
#[no_mangle]
pub extern "C" fn aurora_asin(x: f64) -> f64 {
    x.clamp(-1.0, 1.0).asin()
}
#[no_mangle]
pub extern "C" fn aurora_atan(x: f64) -> f64 {
    x.atan()
}

/// Play a note WITHOUT blocking - mixed into the persistent audio engine, so
/// sounds and music overlap. `looped` != 0 repeats it until volume/stop.
#[no_mangle]
pub extern "C" fn aurora_play_sound(semitone: i64, dur_ms: i64, looped: i64) {
    if headless_audio() {
        if looped == 0 {
            audio_capture_note(semitone, dur_ms);
        }
        return;
    }
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
    if headless_audio() {
        return;
    }
    let sr = 44_100;
    let dur = (dur_ms.max(1) as f32) / 1000.0;
    let g = (gain_pct.clamp(0, 200) as f32) / 100.0;
    let mut note = aurora_audio::Note::new(440.0, dur)
        .wave(aurora_audio::Wave::Noise)
        .gain(g);
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
        // Pan by the listener's right vector = cross(forward, up=+Y), flattened to the XZ plane:
        // cross((Fx,Fy,Fz),(0,1,0)) = (-Fz, 0, Fx). The earlier form ([Fz,0,-Fx]) was this NEGATED,
        // which mirrored the stereo image (sounds on your right played on the left).
        let f = norm3(fwd);
        let right = norm3([-f[2], 0.0, f[0]]);
        let dir = if dist > 1e-4 {
            [to[0] / dist, to[1] / dist, to[2] / dist]
        } else {
            [0.0; 3]
        };
        let pan =
            (right[0] * dir[0] + right[1] * dir[1] + right[2] * dir[2]).clamp(-1.0, 1.0) as f32;
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
    semitone: i64,
    dur_ms: i64,
    gain_pct: i64,
    x: f64,
    y: f64,
    z: f64,
) {
    if headless_audio() {
        return;
    }
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
///
/// # Safety
/// `data` must point to `len` initialized `f64`s.
#[no_mangle]
pub unsafe extern "C" fn aurora_save_settings(data: *const f64, len: i64) -> i64 {
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
///
/// # Safety
/// `data` must be valid for writes of `len` `f64`s.
#[no_mangle]
pub unsafe extern "C" fn aurora_load_settings(data: *mut f64, len: i64) -> i64 {
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
/// `play_wav` builtin - audio asset playback beyond the synth.
///
/// # Safety
/// `ptr` must point to `len` initialized bytes.
#[no_mangle]
pub unsafe extern "C" fn aurora_play_wav(ptr: *const u8, len: i64) -> i64 {
    let path = {
        let s = unsafe { std::slice::from_raw_parts(ptr, len.max(0) as usize) };
        String::from_utf8_lossy(s).into_owned()
    };
    // Headless: don't touch the audio device (deterministic verification runs
    // must not contend for cpal). Report success if the file is readable so
    // the wiring is still exercised.
    if headless_audio() {
        return std::path::Path::new(&path).exists() as i64;
    }
    // Same decoder as load_sound, so a format that one accepts the other does too.
    let Some((mono, rate)) = decode_audio_mono(&path) else {
        return 0;
    };
    if mono.is_empty() {
        return 0;
    }
    aurora_audio::play_mixed(&mono, rate, false);
    1
}

thread_local! {
    // Decoded sound cache: mono f32 samples RESAMPLED to the device rate, shared by Arc so every
    // later play is copy-free (no per-shot resample/alloc). Indexed by handle.
    static SOUNDS: RefCell<Vec<std::sync::Arc<Vec<f32>>>> = const { RefCell::new(Vec::new()) };
}

/// Linear-resample a mono buffer to a new rate, done ONCE at load so playback never re-resamples.
fn resample_mono(src: &[f32], src_rate: u32, dst_rate: u32) -> Vec<f32> {
    if src_rate == dst_rate || src.is_empty() {
        return src.to_vec();
    }
    let ratio = src_rate as f64 / dst_rate as f64;
    let n = (src.len() as f64 / ratio) as usize;
    (0..n)
        .map(|i| {
            let pos = i as f64 * ratio;
            let i0 = pos as usize;
            let frac = (pos - i0 as f64) as f32;
            let s0 = src.get(i0).copied().unwrap_or(0.0);
            let s1 = src.get(i0 + 1).copied().unwrap_or(s0);
            s0 + (s1 - s0) * frac
        })
        .collect()
}

/// Fold interleaved frames down to one channel, APPENDING to `out`. Music and SFX are
/// played through a mono mixer, so this is where a stereo file loses its image - once,
/// at load. One implementation, shared by every decoder below.
fn fold_into(interleaved: &[f32], channels: usize, out: &mut Vec<f32>) {
    if channels <= 1 {
        out.extend_from_slice(interleaved);
        return;
    }
    out.reserve(interleaved.len() / channels);
    out.extend(
        interleaved
            .chunks(channels)
            .map(|c| c.iter().sum::<f32>() / c.len() as f32),
    );
}

/// Owning form of [`fold_into`], which keeps an already-mono buffer copy-free.
fn fold_to_mono(interleaved: Vec<f32>, channels: usize) -> Vec<f32> {
    if channels <= 1 {
        return interleaved;
    }
    let mut out = Vec::with_capacity(interleaved.len() / channels);
    fold_into(&interleaved, channels, &mut out);
    out
}

/// Decode a WAV via hound: lossless, no probing, and the format most SFX ship in.
fn decode_wav_mono(path: &str) -> Option<(Vec<f32>, u32)> {
    let mut reader = hound::WavReader::open(path).ok()?;
    let spec = reader.spec();
    let raw: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().filter_map(|s| s.ok()).collect(),
        hound::SampleFormat::Int => {
            let max = (1i64 << (spec.bits_per_sample - 1).max(1)) as f32;
            reader
                .samples::<i32>()
                .filter_map(|s| s.ok())
                .map(|s| s as f32 / max)
                .collect()
        }
    };
    Some((
        fold_to_mono(raw, spec.channels.max(1) as usize),
        spec.sample_rate,
    ))
}

/// Decode any compressed format Symphonia can read - MP3, OGG/Vorbis, FLAC, M4A/AAC.
///
/// Music is distributed compressed, so requiring WAV meant every track had to be
/// converted by hand before a game could load it. The container is identified by
/// CONTENT, not by extension, so a mislabelled file still loads.
fn decode_compressed_mono(path: &str) -> Option<(Vec<f32>, u32)> {
    use symphonia::core::codecs::audio::AudioDecoderOptions;
    use symphonia::core::formats::probe::Hint;
    use symphonia::core::formats::{FormatOptions, TrackType};
    use symphonia::core::io::MediaSourceStream;
    use symphonia::core::meta::MetadataOptions;

    let file = std::fs::File::open(path).ok()?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());
    // The extension is only a HINT that speeds up probing; content still decides.
    let mut hint = Hint::new();
    if let Some(ext) = std::path::Path::new(path)
        .extension()
        .and_then(|e| e.to_str())
    {
        hint.with_extension(ext);
    }
    let mut format = symphonia::default::get_probe()
        .probe(
            &hint,
            mss,
            FormatOptions::default(),
            MetadataOptions::default(),
        )
        .ok()?;
    let track = format.default_track(TrackType::Audio)?;
    let track_id = track.id;
    let codec_params = track.codec_params.as_ref()?.audio()?;
    let mut rate = codec_params.sample_rate.unwrap_or(0);
    let mut decoder = symphonia::default::get_codecs()
        .make_audio_decoder(codec_params, &AudioDecoderOptions::default())
        .ok()?;

    let mut mono: Vec<f32> = Vec::new();
    // One scratch buffer, reused for every packet: copy_to_vec_interleaved resizes
    // rather than appending, so this decodes a whole track with no per-packet alloc.
    let mut inter: Vec<f32> = Vec::new();
    loop {
        let packet = match format.next_packet() {
            Ok(Some(p)) => p,
            // End of stream, or a container that ends mid-packet: keep what decoded.
            Ok(None) | Err(_) => break,
        };
        if packet.track_id != track_id {
            continue;
        }
        let decoded = match decoder.decode(&packet) {
            Ok(d) => d,
            // A corrupt frame is survivable in audio: skip it rather than lose the file.
            Err(_) => continue,
        };
        if rate == 0 {
            rate = decoded.spec().rate();
        }
        let channels = decoded.spec().channels().count().max(1);
        // Symphonia owns the sample-format conversion (every integer width, signed and
        // unsigned, normalized correctly). Hand-rolling that per variant is exactly the
        // kind of arithmetic that silently gets one case wrong.
        decoded.copy_to_vec_interleaved(&mut inter);
        fold_into(&inter, channels, &mut mono);
    }
    if rate == 0 {
        return None;
    }
    Some((mono, rate))
}

/// Decode an audio file to mono f32 at its own rate. WAV takes the direct path;
/// anything else goes through Symphonia.
fn decode_audio_mono(path: &str) -> Option<(Vec<f32>, u32)> {
    let is_wav = std::path::Path::new(path)
        .extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| e.eq_ignore_ascii_case("wav") || e.eq_ignore_ascii_case("wave"));
    if is_wav {
        if let Some(pcm) = decode_wav_mono(path) {
            return Some(pcm);
        }
        // A .wav hound cannot read (e.g. a compressed payload in a RIFF wrapper, or a
        // misnamed file) still gets the Symphonia attempt rather than failing outright.
    }
    decode_compressed_mono(path)
}

/// Decode an audio file ONCE (mono, normalized f32) and cache it, returning a handle for
/// play_sound_handle / play_sound_handle_at. Returns -1 on failure. Backs `load_sound` - this is
/// how a game loads real SFX and music at startup without re-opening/decoding on every play.
/// WAV, MP3, OGG/Vorbis, FLAC and M4A/AAC are all accepted.
///
/// # Safety
/// `ptr` must point to `len` initialized bytes.
#[no_mangle]
pub unsafe extern "C" fn aurora_load_sound(ptr: *const u8, len: i64) -> i64 {
    let path = {
        let s = unsafe { std::slice::from_raw_parts(ptr, len.max(0) as usize) };
        String::from_utf8_lossy(s).into_owned()
    };
    let Some((mono, src_rate)) = decode_audio_mono(&path) else {
        return -1;
    };
    if mono.is_empty() {
        return -1;
    }
    // Match the device rate ONCE so every play is a copy-free Arc share (fixes the sustained-fire hitch).
    let buf = resample_mono(&mono, src_rate, aurora_audio::device_rate());
    if buf.is_empty() {
        return -1;
    }
    SOUNDS.with(|s| {
        let mut v = s.borrow_mut();
        v.push(std::sync::Arc::new(buf));
        (v.len() - 1) as i64
    })
}

/// Play a cached sound (a load_sound handle) NON-positionally at `gain_pct` (0..200, 100 = unity).
/// Backs `play_sound_handle` - no re-decode, so it is safe on the hot path (every shot/footstep).
#[no_mangle]
pub extern "C" fn aurora_play_sound_handle(handle: i64, gain_pct: i64) {
    if handle < 0 {
        return;
    }
    record_sound_play(handle);
    if headless_audio() {
        return;
    }
    let arc = SOUNDS.with(|s| s.borrow().get(handle as usize).cloned());
    if let Some(a) = arc {
        let g = 0.7 * (gain_pct.max(0) as f32) / 100.0;
        aurora_audio::play_mixed_arc(a, false, g, 0.0);
    }
}

/// Play a cached sound (a load_sound handle) SPATIALIZED at a world position:
/// distance attenuation and stereo pan from the listener pose, like
/// play_sound_at but for a real WAV. Backs `play_sound_handle_at`.
#[no_mangle]
pub extern "C" fn aurora_play_sound_handle_at(handle: i64, gain_pct: i64, x: f64, y: f64, z: f64) {
    if handle < 0 {
        return;
    }
    record_sound_play(handle);
    if headless_audio() {
        return;
    }
    let (sgain, pan) = spatialize([x, y, z]);
    if sgain <= 0.001 {
        return;
    }
    let arc = SOUNDS.with(|s| s.borrow().get(handle as usize).cloned());
    if let Some(a) = arc {
        let g = sgain * (gain_pct.max(0) as f32) / 100.0;
        aurora_audio::play_mixed_arc(a, false, g, pan);
    }
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

/// Start looping a cached sound (a load_sound handle) as background music at `gain_pct` (0..200,
/// 100 = unity). Replaces any current music. Backs `play_music` - a game loads a track once and
/// loops it under the action.
#[no_mangle]
pub extern "C" fn aurora_play_music(handle: i64, gain_pct: i64) {
    if handle < 0 {
        return;
    }
    MUSIC_NOW.with(|c| c.set(handle));
    MUSIC_STARTS.with(|c| c.set(c.get() + 1));
    if headless_audio() {
        return;
    }
    let arc = SOUNDS.with(|s| s.borrow().get(handle as usize).cloned());
    if let Some(a) = arc {
        let g = 0.7 * (gain_pct.max(0) as f32) / 100.0;
        aurora_audio::play_music(a, g);
    }
}

/// Set the background-music gain live from a 0..=200 percentage (a music-volume slider), without
/// restarting the track. Backs `music_volume`.
#[no_mangle]
pub extern "C" fn aurora_music_volume(percent: i64) {
    aurora_audio::set_music_gain(0.7 * (percent.clamp(0, 200) as f32) / 100.0);
}

/// Stop the background music, leaving SFX untouched. Backs `music_stop`.
#[no_mangle]
pub extern "C" fn aurora_music_stop() {
    MUSIC_NOW.with(|c| c.set(-1));
    aurora_audio::stop_music();
}

/// Start looping a cached sound (a load_sound handle) as the AMBIENCE bed at `gain_pct` (0..200).
/// A second looping channel, independent of the music. Backs `play_ambience`.
#[no_mangle]
pub extern "C" fn aurora_play_ambience(handle: i64, gain_pct: i64) {
    if handle < 0 || headless_audio() {
        return;
    }
    let arc = SOUNDS.with(|s| s.borrow().get(handle as usize).cloned());
    if let Some(a) = arc {
        let g = 0.7 * (gain_pct.max(0) as f32) / 100.0;
        aurora_audio::play_ambience(a, g);
    }
}

/// Set the ambience-bed gain live from a 0..=200 percentage. Backs `ambience_volume`.
#[no_mangle]
pub extern "C" fn aurora_ambience_volume(percent: i64) {
    aurora_audio::set_ambience_gain(0.7 * (percent.clamp(0, 200) as f32) / 100.0);
}

/// Stop the ambience bed, leaving music + SFX untouched. Backs `ambience_stop`.
#[no_mangle]
pub extern "C" fn aurora_ambience_stop() {
    aurora_audio::stop_ambience();
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

/// The interactive stepper's callback: it is handed each [`Stop`] as it
/// happens and answers with the [`DbgCmd`] that resumes the program.
pub type StopHandler = Box<dyn FnMut(&Stop) -> DbgCmd>;

#[derive(Default)]
struct DebugState {
    breakpoints: HashSet<u32>,
    step: bool,
    frames: Vec<Frame>,
    stops: Vec<Stop>,
    handler: Option<StopHandler>,
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
pub fn dbg_set_handler(handler: StopHandler) {
    DEBUG.with(|d| d.borrow_mut().handler = Some(handler));
}

/// Take the recorded stops after a run.
pub fn dbg_take_stops() -> Vec<Stop> {
    DEBUG.with(|d| std::mem::take(&mut d.borrow_mut().stops))
}

/// # Safety
/// `name_ptr` must point to `name_len` initialized bytes.
#[no_mangle]
pub unsafe extern "C" fn aurora_dbg_enter(name_ptr: *const u8, name_len: i64) {
    let func = {
        let s = unsafe { std::slice::from_raw_parts(name_ptr, name_len.max(0) as usize) };
        String::from_utf8_lossy(s).into_owned()
    };
    DEBUG.with(|d| {
        d.borrow_mut().frames.push(Frame {
            func,
            vars: Vec::new(),
        })
    });
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

/// # Safety
/// `name_ptr` must point to `name_len` initialized bytes.
unsafe fn dbg_record_var(name_ptr: *const u8, name_len: i64, value: DbgVal) {
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

/// # Safety
/// `name_ptr` must point to `name_len` initialized bytes.
#[no_mangle]
pub unsafe extern "C" fn aurora_dbg_var(name_ptr: *const u8, name_len: i64, value: i64) {
    dbg_record_var(name_ptr, name_len, DbgVal::Int(value));
}

/// # Safety
/// `name_ptr` must point to `name_len` initialized bytes.
#[no_mangle]
pub unsafe extern "C" fn aurora_dbg_var_f64(name_ptr: *const u8, name_len: i64, value: f64) {
    dbg_record_var(name_ptr, name_len, DbgVal::Float(value));
}

macro_rules! rust_ty {
    (I64) => {
        i64
    };
    (F64) => {
        f64
    };
    (Bool) => {
        i64
    };
    (Ptr) => {
        *const u8
    };
}

/// The Rust function-pointer type a row describes, built one parameter at a
/// time so a `Str` can contribute TWO of them - the data pointer and the length
/// its value carries. The accumulator leads because a `Str` RESULT is written
/// through a caller-allocated out-pointer passed first.
macro_rules! rust_fn_ptr {
    ([$($out:ty),*], [], void) => {
        unsafe extern "C" fn($($out),*)
    };
    ([$($out:ty),*], [], $ret:ident) => {
        unsafe extern "C" fn($($out),*) -> rust_ty!($ret)
    };
    ([$($out:ty),*], [Str $(, $rest:ident)*], $ret:ident) => {
        rust_fn_ptr!([$($out,)* *const u8, i64], [$($rest),*], $ret)
    };
    ([$($out:ty),*], [$p:ident $(, $rest:ident)*], $ret:ident) => {
        rust_fn_ptr!([$($out,)* rust_ty!($p)], [$($rest),*], $ret)
    };
}

/// Take a host function's address THROUGH a function pointer spelled from its
/// table row. Rust then checks the row against the real definition, so a row
/// that claims the wrong parameter or return type does not compile - the only
/// thing that can otherwise catch it is a program miscompiled at run time. A
/// `Str` result is the caller-allocated 2-slot out-pointer, passed first.
///
/// The pointer type is spelled `unsafe extern "C" fn` so it accepts BOTH kinds
/// of host function: a safe one coerces to it, and one that takes a raw pointer
/// (and is therefore `unsafe`) matches it directly. The parameter and return
/// types are still checked either way, which is what this macro exists for.
macro_rules! checked_addr {
    // A `Str` result returns nothing: it is written through the out-pointer the
    // accumulator starts with.
    ($sym:ident, [$($p:ident),*], Str) => {{
        let f: rust_fn_ptr!([*mut i64], [$($p),*], void) = $sym;
        f as usize
    }};
    ($sym:ident, [$($p:ident),*], $ret:ident) => {{
        let f: rust_fn_ptr!([], [$($p),*], $ret) = $sym;
        f as usize
    }};
}

// An `inline` builtin has no runtime function to keep. `scalar` and `text` rows
// are entirely table-driven - nothing else describes their signature - so they
// are the kinds whose row is checked against the definition. The rest pass
// arrays and closures, whose Rust spelling the table does not model.
macro_rules! force_link_one {
    ($acc:ident, inline, $sym:ident, [$($p:ident),*], $ret:ident) => {};
    ($acc:ident, scalar, $sym:ident, [$($p:ident),*], $ret:ident) => {
        $acc = $acc.wrapping_add(checked_addr!($sym, [$($p),*], $ret));
    };
    ($acc:ident, text, $sym:ident, [$($p:ident),*], $ret:ident) => {
        $acc = $acc.wrapping_add(checked_addr!($sym, [$($p),*], $ret));
    };
    ($acc:ident, $kind:ident, $sym:ident, [$($p:ident),*], $ret:ident) => {
        $acc = $acc.wrapping_add($sym as *const () as usize);
    };
}

macro_rules! gen_force_link {
    ($([$kind:ident, $name:ident, $sym:ident, [$($p:ident),*], $ret:ident, $home:ident])*) => {
        /// Touch every host symbol so the linker keeps this crate's object in an
        /// AOT link even when the Rust driver references nothing from it
        /// directly. Generated from `aurora-abi`'s builtin table, so it cannot
        /// fall behind the runtime the way a hand-written list did - and a row
        /// naming a function that does not exist, or giving a `scalar` builtin a
        /// signature its definition does not have, fails to COMPILE here.
        pub fn force_link() -> usize {
            let mut acc = 0usize;
            $( force_link_one!(acc, $kind, $sym, [$($p),*], $ret); )*
            acc
        }
    };
}

aurora_abi::for_each_builtin!(gen_force_link);

#[cfg(test)]
mod input_edge_tests {
    use super::*;

    const ACT: i64 = 1;
    const OTHER: i64 = 2;
    const KEY_A: i64 = 40;
    const KEY_B: i64 = 41;

    /// Injected input is written into the window's own key set, so there has to
    /// be a window for it to land in. Headless, which needs no event loop and no
    /// GPU adapter - and per test thread, since the window state is thread-local.
    fn reset() {
        std::env::set_var("AURORA_HEADLESS", "1");
        aurora_window_open(1, 1);
        BINDINGS.with(|b| b.borrow_mut().clear());
        INPUT_PREV.with(|p| p.set([0; INPUT_WORDS]));
        aurora_input_suppress(0);
        aurora_inject_key(KEY_A, 0);
        aurora_inject_key(KEY_B, 0);
        aurora_input_step();
        // The harness has to be able to move a key at all, or every assertion
        // below passes by reading "nothing is held" forever.
        aurora_input_bind(0, KEY_A);
        aurora_inject_key(KEY_A, 1);
        assert_eq!(
            aurora_input_down(0),
            1,
            "injected input never reached a key"
        );
        aurora_inject_key(KEY_A, 0);
        BINDINGS.with(|b| b.borrow_mut().clear());
        aurora_input_step();
    }

    /// A held button is ONE press. This is the whole reason the builtin exists:
    /// without it a game reads `input_down` every frame, a flask belt empties in
    /// a second and a half, and a menu scrolls its whole list on one tap.
    /// A press that arrives AFTER the frame boundary is still an edge.
    ///
    /// This is the ordering a real keyboard uses and the one that shipped
    /// broken. `r3d_present` pumped the window's event loop and then called
    /// `input_step`, so a key going down during the pump was written into the
    /// "previously held" snapshot in the same call that first made it visible.
    /// `now && !was` was never true, and every edge action in a windowed game
    /// was dead - no attack, no flask, no jump, no lock-on - while held actions
    /// like moving and guarding worked perfectly.
    ///
    /// Every existing test injects input and then calls `input_step`, so the
    /// press lands BETWEEN two snapshots. That is the one order that worked and
    /// the only one that was exercised. This drives the other.
    #[test]
    fn a_press_arriving_after_the_boundary_is_still_an_edge() {
        reset();
        aurora_input_bind(ACT, KEY_A);

        // A frame in which nothing is held, ended properly.
        assert_eq!(aurora_input_pressed(ACT), 0, "nothing is held yet");
        aurora_input_step();

        // The key goes down HERE - after the boundary, where a real key's event
        // lands, because presenting is what pumps the event loop.
        aurora_inject_key(KEY_A, 1);
        assert_eq!(
            aurora_input_pressed(ACT),
            1,
            "a key that went down after the last boundary must read as pressed"
        );

        // And it is one press, not sixty.
        aurora_input_step();
        assert_eq!(
            aurora_input_pressed(ACT),
            0,
            "a held key must not read as pressed again"
        );
        assert_eq!(aurora_input_down(ACT), 1, "it is still down");
    }

    /// The release edge, the same way round.
    #[test]
    fn a_release_arriving_after_the_boundary_is_still_an_edge() {
        reset();
        aurora_input_bind(ACT, KEY_A);
        aurora_inject_key(KEY_A, 1);
        aurora_input_step();

        aurora_inject_key(KEY_A, 0);
        assert_eq!(
            aurora_input_released(ACT),
            1,
            "a key that came up after the last boundary must read as released"
        );
        aurora_input_step();
        assert_eq!(aurora_input_released(ACT), 0, "and only once");
    }

    #[test]
    fn a_held_button_is_one_press() {
        reset();
        aurora_input_bind(ACT, KEY_A);

        aurora_inject_key(KEY_A, 1);
        assert_eq!(aurora_input_pressed(ACT), 1, "the press was missed");
        // Asked twice in the same frame, it must answer the same. A read that
        // consumed the edge would make the order of two callers matter.
        assert_eq!(
            aurora_input_pressed(ACT),
            1,
            "the edge was consumed by a read"
        );

        aurora_input_step();
        assert_eq!(aurora_input_down(ACT), 1, "the key is still held");
        assert_eq!(
            aurora_input_pressed(ACT),
            0,
            "a hold reported a second press"
        );

        aurora_inject_key(KEY_A, 0);
        assert_eq!(aurora_input_released(ACT), 1);
        assert_eq!(aurora_input_pressed(ACT), 0);
        aurora_input_step();
        assert_eq!(aurora_input_released(ACT), 0, "a release repeated");

        // And it can happen again.
        aurora_inject_key(KEY_A, 1);
        assert_eq!(aurora_input_pressed(ACT), 1);
    }

    /// Rebinding while the old key is held must not manufacture a press on the
    /// new one - the reason the snapshot is keyed by code and not by action.
    #[test]
    fn rebinding_under_a_held_key_does_not_fire() {
        reset();
        aurora_input_bind(ACT, KEY_A);
        aurora_inject_key(KEY_B, 1);
        aurora_input_step();
        assert_eq!(aurora_input_pressed(ACT), 0);

        // B was already down when the action moved onto it.
        aurora_input_bind(ACT, KEY_B);
        assert_eq!(
            aurora_input_pressed(ACT),
            0,
            "rebinding onto a held key fired it"
        );
    }

    /// A pause menu opened with attack held and closed with attack still held
    /// must not fire an attack on the way out, so the snapshot has to record the
    /// RAW state rather than the suppressed reading.
    #[test]
    fn unsuppressing_a_held_action_does_not_fire_it() {
        reset();
        aurora_input_bind(ACT, KEY_A);
        aurora_inject_key(KEY_A, 1);
        aurora_input_step();

        aurora_input_suppress(1);
        assert_eq!(
            aurora_input_pressed(ACT),
            0,
            "suppressed input reported a press"
        );
        assert_eq!(aurora_input_down(ACT), 0);
        // Frames pass while paused.
        aurora_input_step();
        aurora_input_step();
        aurora_input_suppress(0);
        assert_eq!(
            aurora_input_pressed(ACT),
            0,
            "unpausing fired the button that was already held"
        );
        assert_eq!(aurora_input_down(ACT), 1);
    }

    /// An unbound action is never pressed, rather than being pressed by whatever
    /// key happens to answer for code -1.
    #[test]
    fn an_unbound_action_is_never_pressed() {
        reset();
        aurora_input_bind(ACT, KEY_A);
        aurora_inject_key(KEY_A, 1);
        assert_eq!(aurora_input_pressed(OTHER), 0);
        assert_eq!(aurora_input_released(OTHER), 0);
    }
}

#[cfg(test)]
mod pad_input_tests {
    use super::*;
    use aurora_window::pad;

    const FORWARD: i64 = 10;
    const BACK: i64 = 11;
    const DODGE: i64 = 12;

    fn reset() {
        std::env::set_var("AURORA_HEADLESS", "1");
        aurora_window_open(1, 1);
        BINDINGS.with(|b| b.borrow_mut().clear());
        INPUT_PREV.with(|p| p.set([0; INPUT_WORDS]));
        aurora_input_suppress(0);
        for i in 0..pad::PAD_MAX {
            aurora_inject_pad_disconnect(i as i64);
        }
    }

    fn code(name: &str) -> i64 {
        let c = unsafe { aurora_input_code(name.as_ptr(), name.len() as i64) };
        assert!(c >= 0, "no input code named {name}");
        c
    }

    fn name_of(c: i64) -> String {
        let mut out = [0i64; 2];
        unsafe { aurora_input_name(out.as_mut_ptr(), c) };
        if out[1] <= 0 {
            return String::new();
        }
        unsafe {
            std::str::from_utf8_unchecked(std::slice::from_raw_parts(
                out[0] as *const u8,
                out[1] as usize,
            ))
            .to_string()
        }
    }

    /// The SPEC, in a vocabulary the input layer does not own: a name a human
    /// reads off the pad in their hands, against the number the layer answers.
    ///
    /// This is the shape the dodge-on-Space defect was fixed by. A test that
    /// asks `input_binding` for a code and presses the same code back agrees
    /// with itself however wrong the number is.
    #[test]
    fn every_pad_input_round_trips_through_its_name() {
        reset();
        for (n, _) in PAD_INPUTS.iter() {
            let c = code(n);
            assert_eq!(&name_of(c), n, "{n} did not survive a round trip");
        }
        // And a name nothing answers to is -1 rather than something plausible.
        let missing = "PadTurbo";
        assert_eq!(
            unsafe { aurora_input_code(missing.as_ptr(), missing.len() as i64) },
            -1
        );
        assert_eq!(name_of(INPUT_CODE_MAX), "");
        assert_eq!(name_of(-1), "");
    }

    /// A key and a pad button on the SAME action, which is the whole point:
    /// nobody should have to open a rebind screen before their controller does
    /// anything.
    #[test]
    fn an_action_answers_to_the_keyboard_and_the_pad_at_once() {
        reset();
        let key = code("Space");
        let pad_a = code("PadA");
        aurora_input_bind(DODGE, key);
        aurora_input_bind_also(DODGE, pad_a);
        assert_eq!(aurora_input_binding_count(DODGE), 2);
        // The PRIMARY is still the key, so `input_binding` and every existing
        // caller of it are unchanged.
        assert_eq!(aurora_input_binding(DODGE), key);
        assert_eq!(aurora_input_binding_at(DODGE, 1), pad_a);
        assert_eq!(aurora_input_binding_at(DODGE, 2), -1);

        assert_eq!(aurora_input_down(DODGE), 0);
        aurora_inject_pad_button(0, pad::BTN_SOUTH as i64, 1);
        assert_eq!(
            aurora_input_down(DODGE),
            1,
            "the pad did not fire the action"
        );
        aurora_inject_pad_button(0, pad::BTN_SOUTH as i64, 0);
        assert_eq!(aurora_input_down(DODGE), 0);
        aurora_inject_key(key, 1);
        assert_eq!(aurora_input_down(DODGE), 1, "the key stopped firing it");
        aurora_inject_key(key, 0);

        // Binding again REPLACES, so a rebind screen does not accumulate.
        aurora_input_bind(DODGE, pad_a);
        assert_eq!(aurora_input_binding_count(DODGE), 1);
        // And an alternate cannot be added twice - a setup function called on
        // every room transition would otherwise grow the list forever.
        aurora_input_bind_also(DODGE, key);
        aurora_input_bind_also(DODGE, key);
        assert_eq!(aurora_input_binding_count(DODGE), 2);
    }

    /// ANALOG. The claim the whole design turns on: `input_axis` is the same
    /// call the keyboard has always made, and a stick makes it continuous.
    #[test]
    fn a_stick_gives_a_movement_axis_between_zero_and_one() {
        reset();
        aurora_input_bind(FORWARD, code("KeyW"));
        aurora_input_bind_also(FORWARD, code("PadLeftStickUp"));
        aurora_input_bind(BACK, code("KeyS"));
        aurora_input_bind_also(BACK, code("PadLeftStickDown"));

        assert_eq!(aurora_input_axis(BACK, FORWARD), 0.0);

        // A third of the way forward is a third, not "forward".
        aurora_inject_pad_axis(0, pad::AXIS_LEFT_Y as i64, 0.33);
        let v = aurora_input_axis(BACK, FORWARD);
        assert!(
            (v - 0.33).abs() < 1e-9,
            "a stick a third forward asked for {v}"
        );
        // And it does not read as HELD - a walk is not a press.
        assert_eq!(
            aurora_input_down(FORWARD),
            0,
            "a third of a stick counted as holding the key down"
        );

        // Past the digital threshold it is both: analog for movement, held for
        // anything that only asks yes or no.
        aurora_inject_pad_axis(0, pad::AXIS_LEFT_Y as i64, 1.0);
        assert!((aurora_input_axis(BACK, FORWARD) - 1.0).abs() < 1e-9);
        assert_eq!(aurora_input_down(FORWARD), 1);

        // The other half of the same axis is the opposing action, and gives
        // NOTHING while the stick is pushed forward. Without that, pushing
        // forward would drive back as well and the axis would always be zero.
        assert_eq!(aurora_input_value(BACK), 0.0);
        aurora_inject_pad_axis(0, pad::AXIS_LEFT_Y as i64, -0.5);
        assert!((aurora_input_axis(BACK, FORWARD) + 0.5).abs() < 1e-9);
        assert_eq!(aurora_input_value(FORWARD), 0.0);

        // A KEY on the same action is still worth exactly 1.0, so the keyboard
        // is not quietly downgraded by living beside a stick.
        aurora_inject_pad_axis(0, pad::AXIS_LEFT_Y as i64, 0.0);
        aurora_inject_key(code("KeyW"), 1);
        assert!((aurora_input_axis(BACK, FORWARD) - 1.0).abs() < 1e-9);
        // ...and the LARGEST wins rather than the last one read: a hand resting
        // on the keyboard must not cap what the stick is asking for, and a
        // stick at rest must not cap the key.
        aurora_inject_pad_axis(0, pad::AXIS_LEFT_Y as i64, 0.2);
        assert!((aurora_input_value(FORWARD) - 1.0).abs() < 1e-9);
        aurora_inject_key(code("KeyW"), 0);
    }

    /// A trigger is analog too, and it is where a souls game puts attack and
    /// block. Bound digitally it must still need a deliberate pull.
    #[test]
    fn a_trigger_is_analog_and_needs_a_deliberate_pull() {
        reset();
        let lt = code("PadLT");
        aurora_input_bind(DODGE, lt);
        aurora_inject_pad_axis(0, pad::AXIS_LEFT_TRIGGER as i64, 0.2);
        assert!((aurora_input_value(DODGE) - 0.2).abs() < 1e-9);
        assert_eq!(
            aurora_input_down(DODGE),
            0,
            "a fifth of a trigger counted as a press"
        );
        aurora_inject_pad_axis(0, pad::AXIS_LEFT_TRIGGER as i64, 0.9);
        assert_eq!(aurora_input_down(DODGE), 1);
    }

    /// The EDGE is the action's, not each code's. An action already held on the
    /// keyboard and then also pressed on the pad has not been pressed twice -
    /// which as a second attack is a real one.
    #[test]
    fn two_inputs_on_one_action_are_still_one_press() {
        reset();
        let key = code("Space");
        let pad_a = code("PadA");
        aurora_input_bind(DODGE, key);
        aurora_input_bind_also(DODGE, pad_a);
        aurora_input_step();
        assert_eq!(aurora_input_pressed(DODGE), 0);

        aurora_inject_key(key, 1);
        assert_eq!(aurora_input_pressed(DODGE), 1, "the key press was missed");
        aurora_input_step();
        assert_eq!(aurora_input_pressed(DODGE), 0, "a held key pressed twice");

        // The pad joins in while the key is still held: no second press.
        aurora_inject_pad_button(0, pad::BTN_SOUTH as i64, 1);
        assert_eq!(
            aurora_input_pressed(DODGE),
            0,
            "a second input on a held action manufactured a press"
        );
        // And the action is not RELEASED until both are up.
        aurora_input_step();
        aurora_inject_key(key, 0);
        assert_eq!(
            aurora_input_released(DODGE),
            0,
            "letting go of one of two inputs released the action"
        );
        aurora_input_step();
        aurora_inject_pad_button(0, pad::BTN_SOUTH as i64, 0);
        assert_eq!(aurora_input_released(DODGE), 1);
    }

    /// The edge snapshot has to reach the pad codes at all. It was one `u128`
    /// and the pad codes start at 200: every assertion above would pass while
    /// `input_pressed` silently answered 0 forever for a pad-bound action.
    #[test]
    fn the_edge_snapshot_covers_the_highest_input_code() {
        reset();
        let last = PAD_CODE_BASE + PAD_INPUTS.len() as i64 - 1;
        assert!(
            (INPUT_WORDS * 64) as i64 > last,
            "the snapshot is {} bits and the codes run to {last}",
            INPUT_WORDS * 64
        );
        aurora_input_bind(DODGE, last);
        aurora_input_step();
        // Injected through the ACTION, which is what proves `inject_code` knows
        // a pad code from a key - injecting the number as a key is a silent
        // no-op and the test would pass by never pressing anything.
        aurora_inject_action(DODGE, 1);
        assert_eq!(
            aurora_input_down(DODGE),
            1,
            "inject_action missed a pad code"
        );
        assert_eq!(aurora_input_pressed(DODGE), 1);
        aurora_input_step();
        assert_eq!(aurora_input_pressed(DODGE), 0);
        aurora_inject_action(DODGE, 0);
        assert_eq!(aurora_input_released(DODGE), 1);
    }

    /// Suppression covers the pad too. A pause menu that freezes the keyboard
    /// and not the controller is a pause menu you can be killed through.
    #[test]
    fn suppression_covers_the_controller() {
        reset();
        aurora_input_bind(FORWARD, code("PadLeftStickUp"));
        aurora_inject_pad_axis(0, pad::AXIS_LEFT_Y as i64, 1.0);
        assert_eq!(aurora_input_down(FORWARD), 1);
        aurora_input_suppress(1);
        assert_eq!(aurora_input_down(FORWARD), 0);
        assert_eq!(aurora_input_value(FORWARD), 0.0);
        assert_eq!(aurora_input_axis(BACK, FORWARD), 0.0);
        aurora_input_suppress(0);
        assert_eq!(aurora_input_down(FORWARD), 1);
    }

    /// With nothing plugged in, every pad read is nothing - so a prompt asking
    /// "is there a controller" gets a truthful no, and a game on a machine with
    /// no gamepad support at all still runs.
    #[test]
    fn no_controller_is_answered_honestly() {
        reset();
        assert_eq!(aurora_pad_count(), 0);
        assert_eq!(aurora_pad_connected(0), 0);
        assert_eq!(aurora_pad_button(0, pad::BTN_SOUTH as i64), 0);
        assert_eq!(aurora_pad_axis(0, pad::AXIS_LEFT_X as i64), 0.0);
        assert_eq!(aurora_pad_rumble(0, 1.0, 1.0, 0.2), 0);
        // Negative indexes answer nothing rather than panicking.
        assert_eq!(aurora_pad_connected(-1), 0);
        assert_eq!(aurora_pad_axis(-1, -1), 0.0);
        aurora_inject_pad_button(-1, 0, 1);
        aurora_inject_pad_disconnect(-1);

        aurora_input_bind(FORWARD, code("PadLeftStickUp"));
        assert_eq!(aurora_input_value(FORWARD), 0.0);
        // And an unbound action is nothing, not a panic on an empty list.
        assert_eq!(aurora_input_value(999), 0.0);
        assert_eq!(aurora_input_down(999), 0);
        assert_eq!(aurora_input_binding(999), -1);
        assert_eq!(aurora_input_binding_count(999), 0);
        assert_eq!(aurora_input_binding_at(999, -1), -1);
    }
}

#[cfg(test)]
mod arena_tests {
    use super::*;

    #[test]
    fn spatial_pan_is_not_mirrored() {
        // Listener at the origin looking down -Z (yaw 0 in-game: forward = (sin0, 0, -cos0)).
        aurora_audio_listener(0.0, 0.0, 0.0, 0.0, 0.0, -1.0);
        // A sound to the player's RIGHT (+X when looking -Z) must pan RIGHT (pan > 0), not left.
        let (_, pan_right) = spatialize([10.0, 0.0, 0.0]);
        assert!(
            pan_right > 0.9,
            "sound on the right should pan right, got {pan_right}"
        );
        // ...and a sound to the LEFT (-X) must pan LEFT.
        let (_, pan_left) = spatialize([-10.0, 0.0, 0.0]);
        assert!(
            pan_left < -0.9,
            "sound on the left should pan left, got {pan_left}"
        );
    }

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

/// Parallel-batch world routing must stay bound to the thread doing the work.
///
/// These pin down the ownership rule that `PAR_WORLD` is per-thread: a batch
/// running on one thread must never redirect ECS access on any other thread.
/// Both tests hold the racy window open deliberately, so they fail every time
/// against a process-global routing slot instead of only under load.
#[cfg(test)]
mod par_world_tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Systems are bare `extern "C" fn()`, so the batches below coordinate
    /// through statics. Spin waits are bounded so a regression fails the test
    /// instead of hanging the suite.
    fn wait_for(what: &str, cond: impl Fn() -> bool) {
        let start = std::time::Instant::now();
        while !cond() {
            assert!(
                start.elapsed() < std::time::Duration::from_secs(30),
                "timed out waiting for {what}"
            );
            std::thread::yield_now();
        }
    }

    static HELD: AtomicUsize = AtomicUsize::new(0);
    static RELEASE: AtomicUsize = AtomicUsize::new(0);

    /// Announce arrival, then park inside the batch so the window stays open.
    extern "C" fn hold_open() {
        HELD.fetch_add(1, Ordering::SeqCst);
        wait_for("the test to release the batch", || {
            RELEASE.load(Ordering::SeqCst) != 0
        });
    }

    #[test]
    fn parallel_batch_does_not_capture_another_threads_world() {
        // Thread A owns a world with 5 entities and parks two systems inside a
        // parallel batch. While that batch is wide open, an unrelated thread B
        // does ordinary ECS work. B's entities belong in B's world, and A's
        // world must not absorb them.
        let a = std::thread::spawn(|| {
            for _ in 0..5 {
                aurora_spawn_entity();
            }
            let fns = [
                hold_open as *const () as usize,
                hold_open as *const () as usize,
            ];
            // SAFETY: `fns` points to a live local array of addresses of `extern "C" fn`s
            // that outlive the call.
            unsafe {
                aurora_run_parallel(fns.as_ptr(), 2);
            }
            aurora_entity_count()
        });
        wait_for("both systems to enter the batch", || {
            HELD.load(Ordering::SeqCst) == 2
        });

        let b = std::thread::spawn(|| {
            for _ in 0..3 {
                aurora_spawn_entity();
            }
            aurora_entity_count()
        });
        let b_count = b.join().expect("bystander thread panicked");

        // Release before asserting, so a failure cannot wedge thread A.
        RELEASE.store(1, Ordering::SeqCst);
        let a_count = a.join().expect("batch owner thread panicked");

        assert_eq!(
            b_count, 3,
            "a thread outside the batch must see only its own entities"
        );
        assert_eq!(
            a_count, 5,
            "the batch's world must not absorb another thread's entities"
        );
    }

    static GATE: std::sync::OnceLock<std::sync::Barrier> = std::sync::OnceLock::new();

    /// Wait until every system of both batches is live, then spawn into
    /// whichever world this thread is routed to.
    extern "C" fn spawn_two() {
        GATE.get_or_init(|| std::sync::Barrier::new(4)).wait();
        aurora_spawn_entity();
        aurora_spawn_entity();
    }

    #[test]
    fn concurrent_batches_keep_their_worlds_separate() {
        // Two threads run a two-system batch each, forced to overlap by the
        // barrier. Each system spawns 2 entities into its own batch's world, so
        // both owners must end with exactly 4: one shared routing slot would
        // funnel all 8 spawns into a single world.
        let batch = || {
            std::thread::spawn(|| {
                let fns = [
                    spawn_two as *const () as usize,
                    spawn_two as *const () as usize,
                ];
                // SAFETY: `fns` points to a live local array of addresses of `extern "C" fn`s
                // that outlive the call.
                unsafe {
                    aurora_run_parallel(fns.as_ptr(), 2);
                }
                aurora_entity_count()
            })
        };
        let a = batch();
        let b = batch();
        let a_count = a.join().expect("batch A panicked");
        let b_count = b.join().expect("batch B panicked");
        assert_eq!(
            (a_count, b_count),
            (4, 4),
            "each concurrent batch must stay inside its own owner's world"
        );
    }

    /// Innermost system: one entity into whatever world it is routed to.
    extern "C" fn spawn_one() {
        aurora_spawn_entity();
    }

    /// A system that itself opens a batch, so the workers below are nested.
    extern "C" fn nested_batch() {
        let fns = [
            spawn_one as *const () as usize,
            spawn_one as *const () as usize,
        ];
        // SAFETY: `fns` points to a live local array of addresses of `extern "C" fn`s
        // that outlive the call.
        unsafe {
            aurora_run_parallel(fns.as_ptr(), 2);
        }
    }

    #[test]
    fn nested_batches_reach_the_owning_threads_world() {
        // A batch started from inside a batch must keep writing to the world of
        // the thread that opened the outer one, not to the empty thread-local
        // world of the worker that happens to be running the outer system.
        // 2 outer systems * 2 inner systems = 4 entities, all in the owner.
        let owner = std::thread::spawn(|| {
            let fns = [
                nested_batch as *const () as usize,
                nested_batch as *const () as usize,
            ];
            // SAFETY: `fns` points to a live local array of addresses of `extern "C" fn`s
            // that outlive the call.
            unsafe {
                aurora_run_parallel(fns.as_ptr(), 2);
            }
            aurora_entity_count()
        });
        let count = owner.join().expect("nested batch owner panicked");
        assert_eq!(
            count, 4,
            "nested batch writes must land in the outer owner's world"
        );
    }
}

#[cfg(test)]
mod phys2d_tests {
    use super::*;

    /// Rapier's own sets plus the handle store, which is where the memory is.
    fn census() -> (usize, usize, usize, usize) {
        PHYS.with(|p| {
            let p = p.borrow();
            match p.as_ref() {
                Some(p) => (
                    p.bodies.len(),
                    p.colliders.len(),
                    p.registry.len(),
                    p.registry.slot_count(),
                ),
                None => (0, 0, 0, 0),
            }
        })
    }

    /// 500 spawn/despawn cycles - a bullet or an enemy per frame - must leave
    /// the world exactly as they found it. Before `phys_remove` existed there
    /// was no way to take a 2D body back out at all.
    #[test]
    fn create_and_destroy_cycles_leave_the_world_bounded() {
        aurora_phys_init(0.0, 900.0);
        aurora_phys_add(50.0, 200.0, 60.0, 10.0, 0);
        let start = census();
        assert_eq!(start, (1, 1, 1, 1), "just the floor");

        for i in 0..500 {
            let bullet = aurora_phys_add(10.0, 10.0, 2.0, 2.0, 1);
            aurora_phys_set_vel(bullet, 100.0, 0.0);
            aurora_phys_step(0.016);
            assert_eq!(aurora_phys_remove(bullet), 1, "cycle {i}: not removed");
        }

        let end = census();
        assert_eq!(end.0, start.0, "Rapier rigid bodies grew to {}", end.0);
        assert_eq!(end.1, start.1, "Rapier colliders grew to {}", end.1);
        assert_eq!(end.2, start.2, "live handles grew to {}", end.2);
        assert_eq!(end.3, 2, "handle slots grew to {}", end.3);
    }

    #[test]
    fn removing_a_body_takes_its_collider_down_with_it() {
        aurora_phys_init(0.0, 0.0);
        let wall = aurora_phys_add(0.0, 0.0, 5.0, 5.0, 0);
        aurora_phys_step(0.016);
        assert_eq!(census(), (1, 1, 1, 1));
        assert!(aurora_phys_raycast(-50.0, 0.0, 1.0, 0.0, 100.0) > 0.0);

        assert_eq!(aurora_phys_remove(wall), 1);
        assert_eq!(census(), (0, 0, 0, 1), "an orphaned collider was left");
        assert_eq!(
            aurora_phys_raycast(-50.0, 0.0, 1.0, 0.0, 100.0),
            -1.0,
            "a removed body still answered a raycast"
        );
    }

    /// The old handle must be refused, not aliased onto whatever takes its slot.
    #[test]
    fn a_removed_handle_is_refused_by_every_accessor_that_takes_one() {
        aurora_phys_init(0.0, 0.0);
        let dead = aurora_phys_add(1.0, 2.0, 1.0, 1.0, 1);
        assert_eq!(aurora_phys_remove(dead), 1);
        let live = aurora_phys_add(30.0, 40.0, 1.0, 1.0, 1);
        assert_eq!(
            Body2::from_i64(dead).unwrap().slot(),
            Body2::from_i64(live).unwrap().slot(),
            "the freed slot must be reused for this test to mean anything"
        );
        assert_ne!(dead, live);
        aurora_phys_step(0.016);

        assert_eq!(aurora_phys_alive(dead), 0);
        assert_eq!(aurora_phys_alive(live), 1);
        assert_eq!(aurora_phys_x(dead), 0.0, "a dead handle read a live body");
        assert_eq!(aurora_phys_y(dead), 0.0);
        assert_eq!(aurora_phys_vel_x(dead), 0.0);
        assert_eq!(aurora_phys_vel_y(dead), 0.0);
        assert_eq!(aurora_phys_x(live), 30.0);
        assert_eq!(aurora_phys_y(live), 40.0);

        let before = (
            aurora_phys_x(live),
            aurora_phys_y(live),
            aurora_phys_vel_x(live),
            aurora_phys_vel_y(live),
        );
        aurora_phys_set_pos(dead, -900.0, -900.0);
        aurora_phys_set_vel(dead, 77.0, 77.0);
        aurora_phys_apply_impulse(dead, 500.0, 500.0);
        aurora_phys_apply_force(dead, 500.0, 500.0);
        assert_eq!(
            (
                aurora_phys_x(live),
                aurora_phys_y(live),
                aurora_phys_vel_x(live),
                aurora_phys_vel_y(live),
            ),
            before,
            "a write through a dead handle reached the body that took its slot"
        );

        assert_eq!(
            aurora_phys_remove(dead),
            0,
            "double free must report nothing"
        );
        assert_eq!(aurora_phys_remove(-1), 0);
        assert_eq!(aurora_phys_remove(0), 0);
        assert_eq!(aurora_phys_alive(live), 1, "the removal hit the wrong body");
    }

    #[test]
    fn a_handle_from_the_previous_world_is_refused() {
        aurora_phys_init(0.0, 0.0);
        let old = aurora_phys_add(1.0, 2.0, 1.0, 1.0, 0);
        aurora_phys_init(0.0, 0.0);
        assert_eq!(aurora_phys_alive(old), 0, "a handle outlived its world");
        let new = aurora_phys_add(60.0, 70.0, 1.0, 1.0, 0);
        assert_ne!(old, new, "the new world reissued the old world's handle");
        assert_eq!(aurora_phys_x(old), 0.0, "the old handle read a new body");
        assert_eq!(aurora_phys_x(new), 60.0);
        assert_eq!(aurora_phys_remove(old), 0);
        assert_eq!(aurora_phys_alive(new), 1);
    }

    #[test]
    fn resetting_the_world_in_a_loop_is_bounded() {
        for _ in 0..200 {
            aurora_phys_init(0.0, 900.0);
            aurora_phys_add(0.0, 0.0, 5.0, 5.0, 0);
            aurora_phys_add(0.0, 50.0, 2.0, 2.0, 1);
            aurora_phys_step(0.016);
        }
        assert_eq!(census(), (2, 2, 2, 2), "a world reset grew the store");
    }
}

#[cfg(test)]
mod audio_decode_tests {
    use super::{decode_audio_mono, fold_into, fold_to_mono};

    fn fixture(name: &str) -> String {
        format!("{}/tests/fixtures/{}", env!("CARGO_MANIFEST_DIR"), name)
    }

    /// Root-mean-square of the decoded signal. A decoder that returns silence, or
    /// noise, or the wrong sample scaling, fails this where a "did it return Some"
    /// check would pass.
    fn rms(v: &[f32]) -> f32 {
        (v.iter().map(|s| s * s).sum::<f32>() / v.len().max(1) as f32).sqrt()
    }

    #[test]
    fn fold_averages_channels_and_keeps_mono_intact() {
        let mut out = Vec::new();
        fold_into(&[1.0, 0.0, 0.0, 1.0], 2, &mut out);
        assert_eq!(out, vec![0.5, 0.5]);

        // Mono passes through untouched, including the owning form's copy-free path.
        let mono = vec![0.25, -0.75];
        assert_eq!(fold_to_mono(mono.clone(), 1), mono);
        assert_eq!(fold_to_mono(vec![1.0, 0.0, 0.0, 1.0], 2), vec![0.5, 0.5]);

        // A trailing partial frame is averaged over what is actually there, never
        // divided by the channel count (which would attenuate the last frame).
        let mut odd = Vec::new();
        fold_into(&[1.0, 1.0, 1.0], 2, &mut odd);
        assert_eq!(odd, vec![1.0, 1.0]);
    }

    /// The same half-second 440 Hz tone in four containers must decode to the same
    /// signal. Comparing the compressed formats AGAINST the WAV is what makes this a
    /// real check: a decoder that produced silence, half-speed audio, or samples off
    /// by a scale factor would diverge from the reference.
    #[test]
    fn decodes_wav_mp3_ogg_and_flac_to_the_same_tone() {
        let (wav, wav_rate) = decode_audio_mono(&fixture("tone.wav")).expect("wav decodes");
        assert_eq!(wav_rate, 44100);
        // 0.5 s at 44.1 kHz.
        assert!(
            (wav.len() as i64 - 22050).abs() < 128,
            "wav length {}",
            wav.len()
        );
        let reference = rms(&wav);
        // A 0.5-amplitude sine has RMS 0.354 - confirmed against an independent read
        // of the fixture outside this crate, so it pins the sample SCALING, not just
        // "something came back". An off-by-a-power-of-two normalization fails here.
        assert!(
            (reference - 0.354).abs() < 0.02,
            "reference rms {reference}"
        );

        for name in ["tone.mp3", "tone.ogg", "tone.flac"] {
            let (pcm, rate) = decode_audio_mono(&fixture(name)).unwrap_or_else(|| {
                panic!("{name} must decode - load_sound accepts compressed audio")
            });
            assert_eq!(rate, 44100, "{name} sample rate");
            // Lossy encoders pad the stream, so length is close rather than exact.
            assert!(
                (pcm.len() as i64 - 22050).abs() < 4410,
                "{name} length {}",
                pcm.len()
            );
            let got = rms(&pcm);
            assert!(
                (got - reference).abs() < 0.05,
                "{name} rms {got} vs reference {reference}"
            );
        }
    }

    #[test]
    fn rejects_what_it_cannot_decode() {
        assert!(decode_audio_mono(&fixture("does_not_exist.mp3")).is_none());
        // A real file that is not audio must fail rather than yield garbage samples.
        assert!(decode_audio_mono(&format!("{}/Cargo.toml", env!("CARGO_MANIFEST_DIR"))).is_none());
    }
}
