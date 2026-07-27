//! Immediate-mode windowing for the Aurora language builtins.
//!
//! `run` owns the event loop, but Aurora's `while window_present() { .. }` game
//! loop needs to keep control of the thread. winit's `pump_app_events` lets us
//! pump pending events on each `present` call without surrendering the loop, so
//! an Aurora program can open a window, draw a framebuffer, and poll input from
//! its own loop. State lives in a thread-local (the program runs on one thread).

use std::cell::RefCell;
use std::collections::HashSet;
use std::sync::Arc;
use std::time::Duration;

use winit::application::ApplicationHandler;
use winit::event::{ElementState, MouseButton, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::platform::pump_events::EventLoopExtPumpEvents;
use winit::window::{Window, WindowId};

use crate::Gfx;

struct ImmApp {
    width: u32,
    height: u32,
    window: Option<Arc<Window>>,
    gfx: Option<Gfx>,
    keys: HashSet<KeyCode>,
    /// Queue of typed character codes for text fields (Backspace pushes 8).
    typed: Vec<u32>,
    open: bool,
    /// Mouse position in framebuffer pixels, and button states.
    mouse: (i64, i64),
    mouse_down: bool,
    mouse_right: bool,
    mouse_middle: bool,
    mouse_back: bool,
    mouse_forward: bool,
    /// Raw mouse motion accumulated since the last present (for FPS look).
    mouse_dx: f64,
    mouse_dy: f64,
    /// Scroll accumulated since the last present.
    scroll: f64,
    /// Whether the cursor is currently grabbed + hidden (FPS look).
    grabbed: bool,
    /// Whether the game asked for a grab at all (so a click can re-capture after
    /// Escape releases it).
    grab_wanted: bool,
    /// Window inner size (to map cursor coords back to framebuffer pixels).
    win_size: (f64, f64),
    /// Speed/wind lines overlay state (intensity 0..1, animation time).
    sl_intensity: f32,
    sl_time: f32,
    /// Damage overlay: low-health vignette, hit-glow intensity, hit direction.
    dmg_vig: f32,
    dmg_hit: f32,
    dmg_dx: f32,
    dmg_dy: f32,
    /// Gold overclock tint intensity (0..1).
    dmg_oc: f32,
    /// Fullscreen blur radius in pixels (0 = off); used for the paused/menu backdrop.
    blur: f32,
    /// Headless mode (AURORA_HEADLESS=1): no window, no event loop, no surface.
    /// 3D renders offscreen on demand (`r3d_capture`); input comes from the
    /// inject_* builtins or a replay tape.
    headless: bool,
    /// Lazily-created headless GPU + scene (only when 3D builtins are used).
    hgfx: Option<HeadlessGfx>,
    /// Present counter (frames elapsed), for AURORA_MAX_FRAMES and tapes.
    frame: u64,
    /// Present returns "closed" past this many frames (0 = unlimited). The
    /// harness's universal timeout: any unmodified game loop exits cleanly.
    max_frames: u64,
    /// Input record/replay tape (AURORA_INPUT_RECORD / AURORA_INPUT_REPLAY).
    tape: Tape,
}

impl ApplicationHandler for ImmApp {
    fn resumed(&mut self, el: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        let attrs = Window::default_attributes()
            .with_title("Aurora")
            // PHYSICAL-pixel size so the surface is EXACTLY width x height regardless
            // of the display's DPI scaling. The framebuffer/HUD is the same size, so
            // it blits 1:1 - pixel-sharp and perfectly centered, no DPI upscaling.
            .with_inner_size(winit::dpi::PhysicalSize::new(self.width, self.height));
        match el.create_window(attrs) {
            Ok(w) => {
                let w = Arc::new(w);
                match Gfx::new(w.clone(), self.width, self.height) {
                    Ok(g) => {
                        self.gfx = Some(g);
                        // The window is created lazily on the first frame, so a
                        // grab requested at startup (before it existed) is applied
                        // now that we have a window.
                        if self.grabbed {
                            apply_grab(&w, true);
                        }
                        self.window = Some(w);
                    }
                    Err(e) => {
                        eprintln!("aurora-window: GPU init failed: {e}");
                        self.open = false;
                    }
                }
            }
            Err(e) => {
                eprintln!("aurora-window: window creation failed: {e}");
                self.open = false;
            }
        }
    }

    fn window_event(&mut self, _el: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => self.open = false,
            WindowEvent::Resized(size) => {
                self.win_size = (size.width.max(1) as f64, size.height.max(1) as f64);
                // Track the REAL window size so the cursor mapping + surface_w()/_h()
                // (and any framebuffer sized to them) all agree - otherwise the reported
                // mouse position drifts from the OS cursor when the window isn't exactly
                // the requested size (DPI scaling, resize, etc.).
                self.width = size.width.max(1);
                self.height = size.height.max(1);
                if let Some(g) = self.gfx.as_mut() {
                    g.resize(size.width, size.height);
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                // Map window pixels back to framebuffer pixels.
                let fx = position.x / self.win_size.0 * self.width as f64;
                let fy = position.y / self.win_size.1 * self.height as f64;
                self.mouse = (fx as i64, fy as i64);
            }
            WindowEvent::MouseInput { state, button, .. } => {
                let down = state == ElementState::Pressed;
                match button {
                    MouseButton::Left => self.mouse_down = down,
                    MouseButton::Right => self.mouse_right = down,
                    MouseButton::Middle => self.mouse_middle = down,
                    MouseButton::Back => self.mouse_back = down,
                    MouseButton::Forward => self.mouse_forward = down,
                    _ => {}
                }
                // Clicking back into the window re-captures the cursor after
                // Escape released it (standard FPS / pointer-lock behaviour).
                if down && self.grab_wanted && !self.grabbed {
                    if let Some(w) = &self.window {
                        apply_grab(w, true);
                        self.grabbed = true;
                    }
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                self.scroll += match delta {
                    winit::event::MouseScrollDelta::LineDelta(_, y) => y as f64,
                    winit::event::MouseScrollDelta::PixelDelta(p) => p.y / 40.0,
                };
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if let PhysicalKey::Code(code) = event.physical_key {
                    // Escape releases the mouse (so you can reach other windows)
                    // instead of quitting; click back in to re-capture. Close the
                    // window to quit.
                    if code == KeyCode::Escape && event.state == ElementState::Pressed {
                        if let Some(w) = &self.window {
                            apply_grab(w, false);
                        }
                        self.grabbed = false;
                    }
                    if event.state == ElementState::Pressed {
                        self.keys.insert(code);
                        if code == KeyCode::Backspace {
                            self.typed.push(8);
                        }
                    } else {
                        self.keys.remove(&code);
                    }
                }
                if event.state == ElementState::Pressed {
                    if let Some(t) = &event.text {
                        for ch in t.chars() {
                            let c = ch as u32;
                            if (32..127).contains(&c) {
                                self.typed.push(c);
                            }
                        }
                    }
                    // Bound the queue: it's only drained by text fields, so cap it so held keys
                    // during normal gameplay can't grow it without limit.
                    while self.typed.len() > 256 {
                        self.typed.remove(0);
                    }
                }
            }
            _ => {}
        }
    }

    fn device_event(
        &mut self,
        _el: &ActiveEventLoop,
        _id: winit::event::DeviceId,
        event: winit::event::DeviceEvent,
    ) {
        // Raw mouse motion: the unaccelerated delta an FPS camera wants.
        if let winit::event::DeviceEvent::MouseMotion { delta } = event {
            self.mouse_dx += delta.0;
            self.mouse_dy += delta.1;
        }
    }
}

thread_local! {
    static IMM: RefCell<Option<(Option<EventLoop<()>>, ImmApp)>> = const { RefCell::new(None) };
}

/// Leak the window + GPU state instead of dropping it. Call right before the process
/// exits: wgpu/winit panic if their state is torn down in a thread-local destructor at
/// process exit ("thread local panicked on drop"). Leaking it makes shutdown graceful.
pub fn imm_leak() {
    IMM.with(|s| {
        if let Some(inner) = s.borrow_mut().take() {
            // Headless state has no winit window/event loop; dropping the wgpu
            // device DELIBERATELY (before thread-local teardown) lets its worker
            // threads shut down cleanly instead of panicking at process exit.
            if inner.1.headless {
                drop(inner);
            } else {
                std::mem::forget(inner);
            }
        }
    });
}

/// Open a window backing a `width`×`height` framebuffer. Replaces any prior one.
/// With `AURORA_HEADLESS=1` no window or event loop is created: presents just
/// advance the frame counter (and the replay tape), and 3D renders offscreen
/// on demand via `r3d_capture`.
pub fn open(width: u32, height: u32) {
    let headless = std::env::var("AURORA_HEADLESS")
        .map(|v| v == "1")
        .unwrap_or(false);
    let event_loop = if headless {
        None
    } else {
        match crate::new_event_loop() {
            Ok(e) => Some(e),
            Err(e) => {
                eprintln!("aurora-window: event loop creation failed: {e}");
                return;
            }
        }
    };
    let max_frames = std::env::var("AURORA_MAX_FRAMES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    let tape = Tape::from_env();
    let app = ImmApp {
        width: width.max(1),
        height: height.max(1),
        window: None,
        gfx: None,
        keys: HashSet::new(),
        typed: Vec::new(),
        open: true,
        mouse: (0, 0),
        mouse_down: false,
        mouse_right: false,
        mouse_middle: false,
        mouse_back: false,
        mouse_forward: false,
        mouse_dx: 0.0,
        mouse_dy: 0.0,
        scroll: 0.0,
        grabbed: false,
        grab_wanted: false,
        win_size: (width.max(1) as f64, height.max(1) as f64),
        sl_intensity: 0.0,
        sl_time: 0.0,
        dmg_vig: 0.0,
        dmg_hit: 0.0,
        dmg_dx: 0.0,
        dmg_dy: 0.0,
        dmg_oc: 0.0,
        blur: 0.0,
        headless,
        hgfx: None,
        frame: 0,
        max_frames,
        tape,
    };
    IMM.with(|s| *s.borrow_mut() = Some((event_loop, app)));
}

/// The raw mouse motion accumulated this frame. Reset at the next present.
pub fn mouse_delta() -> (f64, f64) {
    IMM.with(|s| {
        s.borrow()
            .as_ref()
            .map(|(_, app)| (app.mouse_dx, app.mouse_dy))
            .unwrap_or((0.0, 0.0))
    })
}

/// The scroll-wheel delta accumulated this frame. Reset at the next present.
pub fn scroll() -> f64 {
    IMM.with(|s| {
        s.borrow()
            .as_ref()
            .map(|(_, app)| app.scroll)
            .unwrap_or(0.0)
    })
}

fn reset_frame_input(app: &mut ImmApp) {
    app.mouse_dx = 0.0;
    app.mouse_dy = 0.0;
    app.scroll = 0.0;
}

/// Whether mouse button `b` is held: 0 = left, 1 = right, 2 = middle.
pub fn mouse_button(b: u32) -> bool {
    IMM.with(|s| {
        s.borrow()
            .as_ref()
            .map(|(_, app)| match b {
                1 => app.mouse_right,
                2 => app.mouse_middle,
                3 => app.mouse_back,
                4 => app.mouse_forward,
                _ => app.mouse_down,
            })
            .unwrap_or(false)
    })
}

/// Grab + hide the cursor for FPS mouse-look (or release it). Falls back from
/// locked to confined grab if the platform requires it.
/// Apply (or release) the cursor grab + visibility on a window. Locked is the
/// FPS ideal; fall back to Confined where the platform requires it.
fn apply_grab(w: &Window, on: bool) {
    if on {
        let _ = w
            .set_cursor_grab(winit::window::CursorGrabMode::Locked)
            .or_else(|_| w.set_cursor_grab(winit::window::CursorGrabMode::Confined));
        w.set_cursor_visible(false);
    } else {
        let _ = w.set_cursor_grab(winit::window::CursorGrabMode::None);
        w.set_cursor_visible(true);
    }
}

pub fn grab_mouse(on: bool) {
    IMM.with(|s| {
        let mut slot = s.borrow_mut();
        let Some((_, app)) = slot.as_mut() else {
            return;
        };
        app.grabbed = on;
        // Track intent both ways: releasing for a menu (on=false) must clear
        // grab_wanted, or the click-to-recapture path would re-grab on the first
        // menu click. (Escape leaves grab_wanted set, so click-back still works in play.)
        app.grab_wanted = on;
        // If the window exists, apply now; otherwise `resumed` applies it when the
        // window is created on the first frame.
        if let Some(w) = &app.window {
            apply_grab(w, on);
        }
    })
}

/// Pump events, present `rgba` (tight `width*height*4` bytes), and return whether
/// the window is still open. Returns `false` if no window was opened.
pub fn present(rgba: &[u8]) -> bool {
    IMM.with(|s| {
        let mut slot = s.borrow_mut();
        let Some((event_loop, app)) = slot.as_mut() else {
            return false;
        };
        // Clear last frame's per-frame input (mouse delta, scroll) BEFORE pumping,
        // so the deltas accumulated this pump survive for the caller to read after
        // present returns. Resetting after the pump would zero them first.
        reset_frame_input(app);
        if let Some(el) = event_loop.as_mut() {
            el.pump_app_events(Some(Duration::ZERO), app);
        }
        end_frame(app);
        if app.open && !app.headless {
            // Only upload when the buffer matches the window's framebuffer size.
            let expected = (app.width * app.height * 4) as usize;
            if let Some(g) = app.gfx.as_mut() {
                if rgba.len() >= expected {
                    g.present_rgba(&rgba[..expected]);
                }
            }
        }
        app.open
    })
}

/// Whether the key with the given Aurora key code is currently held.
pub fn key_down(code: u32) -> bool {
    let Some(key) = code_to_key(code) else {
        return false;
    };
    IMM.with(|s| {
        s.borrow()
            .as_ref()
            .map(|(_, app)| app.keys.contains(&key))
            .unwrap_or(false)
    })
}

/// Set the window's fullscreen mode: 0 = windowed, 1 = borderless (windowed) fullscreen,
/// 2 = exclusive fullscreen (falls back to borderless if no exclusive mode is available).
pub fn window_fullscreen(mode: i64) {
    IMM.with(|s| {
        if let Some((_, app)) = s.borrow().as_ref() {
            if let Some(w) = &app.window {
                let fs = match mode {
                    1 => Some(winit::window::Fullscreen::Borderless(None)),
                    2 => w
                        .current_monitor()
                        .and_then(|m| m.video_modes().next())
                        .map(winit::window::Fullscreen::Exclusive)
                        .or(Some(winit::window::Fullscreen::Borderless(None))),
                    _ => None,
                };
                w.set_fullscreen(fs);
            }
        }
    });
}

/// Pop the next typed character code from the queue (0 if none). Backspace = 8.
pub fn input_char() -> i64 {
    IMM.with(|s| {
        s.borrow_mut()
            .as_mut()
            .map(|(_, app)| {
                if app.typed.is_empty() {
                    0
                } else {
                    app.typed.remove(0) as i64
                }
            })
            .unwrap_or(0)
    })
}

/// Current mouse position in framebuffer pixels, and left-button state.
pub fn mouse() -> (i64, i64, bool) {
    IMM.with(|s| {
        s.borrow()
            .as_ref()
            .map(|(_, app)| (app.mouse.0, app.mouse.1, app.mouse_down))
            .unwrap_or((0, 0, false))
    })
}

// --- 3D scene API (the `r3d_*` builtins) -----------------------------------
//
// These drive the GPU 3D renderer that lives inside `Gfx`, sharing the window's
// wgpu device. Resource creation needs the device, which exists only once the
// window has been resumed, so `with_gfx` pumps one round of events to force
// window/device creation on first use.

use glam::{EulerRot, Mat4, Quat, Vec3};

/// Anything that can hand out the wgpu device/queue and the 3D scene: the
/// window's `Gfx`, or the surface-free `HeadlessGfx`. Every r3d_* builtin is
/// written against this, so both backends share one code path.
trait SceneHost {
    fn scene_mut(&mut self) -> (&wgpu::Device, &wgpu::Queue, &mut aurora_render3d::Scene);
}

impl SceneHost for Gfx {
    fn scene_mut(&mut self) -> (&wgpu::Device, &wgpu::Queue, &mut aurora_render3d::Scene) {
        Gfx::scene_mut(self)
    }
}

/// Headless GPU + scene: a device from `headless_device()` (no surface) and a
/// `Scene` in the offscreen-proven format (Rgba8Unorm, no MSAA).
struct HeadlessGfx {
    device: wgpu::Device,
    queue: wgpu::Queue,
    scene: Option<aurora_render3d::Scene>,
    w: u32,
    h: u32,
}

impl HeadlessGfx {
    fn new(w: u32, h: u32) -> Option<HeadlessGfx> {
        let (device, queue) = aurora_render3d::headless_device()?;
        Some(HeadlessGfx {
            device,
            queue,
            scene: None,
            w: w.max(1),
            h: h.max(1),
        })
    }
}

impl SceneHost for HeadlessGfx {
    fn scene_mut(&mut self) -> (&wgpu::Device, &wgpu::Queue, &mut aurora_render3d::Scene) {
        if self.scene.is_none() {
            self.scene = Some(aurora_render3d::Scene::new(
                &self.device,
                &self.queue,
                wgpu::TextureFormat::Rgba8Unorm,
                self.w,
                self.h,
                1,
            ));
        }
        (&self.device, &self.queue, self.scene.as_mut().unwrap())
    }
}

fn with_gfx<R>(default: R, f: impl FnOnce(&mut dyn SceneHost) -> R) -> R {
    IMM.with(|s| {
        let mut slot = s.borrow_mut();
        let Some((event_loop, app)) = slot.as_mut() else {
            return default;
        };
        if app.headless {
            if app.hgfx.is_none() && app.open {
                match HeadlessGfx::new(app.width, app.height) {
                    Some(h) => app.hgfx = Some(h),
                    None => {
                        // Contract: a blocked visual path is loud, never silent -
                        // runners treat this marker as BLOCKED, not as a pass.
                        eprintln!("aurora: HEADLESS-NO-GPU");
                        app.open = false;
                        return default;
                    }
                }
            }
            return match app.hgfx.as_mut() {
                Some(h) => f(h),
                None => default,
            };
        }
        if app.gfx.is_none() && app.open {
            if let Some(el) = event_loop.as_mut() {
                el.pump_app_events(Some(Duration::ZERO), app);
            }
        }
        match app.gfx.as_mut() {
            Some(g) => f(g),
            None => default,
        }
    })
}

/// Load a glTF/GLB/OBJ model; returns a handle (>= 0) or -1 on failure.
pub fn r3d_load_model(path: &str) -> i64 {
    with_gfx(-1, |g| {
        let (d, q, s) = g.scene_mut();
        s.load_model(d, q, path)
    })
}

/// Release a model or primitive handle and every GPU buffer it owns. Returns 1
/// when something was freed, 0 for a handle that was already freed or never
/// valid.
pub fn r3d_free_model(handle: i64) -> i64 {
    with_gfx(0, |g| {
        let (_, _, s) = g.scene_mut();
        s.free_model(handle) as i64
    })
}

pub fn r3d_make_box(r: f32, g: f32, b: f32) -> i64 {
    with_gfx(-1, |gf| {
        let (d, q, s) = gf.scene_mut();
        s.make_box(d, q, [r, g, b, 1.0])
    })
}
pub fn r3d_make_box_sized(hx: f32, hy: f32, hz: f32, r: f32, g: f32, b: f32) -> i64 {
    with_gfx(-1, |gf| {
        let (d, q, s) = gf.scene_mut();
        s.make_box_sized(d, q, hx, hy, hz, [r, g, b, 1.0])
    })
}
pub fn r3d_make_box_emissive(hx: f32, hy: f32, hz: f32, r: f32, g: f32, b: f32) -> i64 {
    with_gfx(-1, |gf| {
        let (d, q, s) = gf.scene_mut();
        s.make_box_emissive(d, q, hx, hy, hz, [r, g, b])
    })
}
pub fn r3d_make_sphere(segments: i64, r: f32, g: f32, b: f32) -> i64 {
    with_gfx(-1, |gf| {
        let (d, q, s) = gf.scene_mut();
        s.make_sphere(d, q, segments.max(3) as u32, [r, g, b, 1.0])
    })
}
pub fn r3d_make_plane(size: f32, tiles: f32, r: f32, g: f32, b: f32) -> i64 {
    with_gfx(-1, |gf| {
        let (d, q, s) = gf.scene_mut();
        s.make_plane(d, q, size, tiles.max(1.0), [r, g, b, 1.0])
    })
}

pub fn r3d_camera(ex: f32, ey: f32, ez: f32, tx: f32, ty: f32, tz: f32, fov_deg: f32) {
    with_gfx((), |gf| {
        let (_, _, s) = gf.scene_mut();
        s.set_camera(Vec3::new(ex, ey, ez), Vec3::new(tx, ty, tz), fov_deg);
    });
}
pub fn r3d_camera_roll(roll: f32) {
    with_gfx((), |gf| {
        let (_, _, s) = gf.scene_mut();
        s.set_camera_roll(roll);
    });
}
pub fn r3d_light(dx: f32, dy: f32, dz: f32, r: f32, g: f32, b: f32, ambient: f32) {
    with_gfx((), |gf| {
        let (_, _, s) = gf.scene_mut();
        s.set_light(Vec3::new(dx, dy, dz), Vec3::new(r, g, b), ambient);
    });
}
pub fn r3d_clear(r: f32, g: f32, b: f32) {
    with_gfx((), |gf| {
        let (_, _, s) = gf.scene_mut();
        s.set_clear(r, g, b);
    });
}
pub fn r3d_fog(r: f32, g: f32, b: f32, density: f32) {
    with_gfx((), |gf| {
        let (_, _, s) = gf.scene_mut();
        s.set_fog(Vec3::new(r, g, b), density);
    });
}
#[allow(clippy::too_many_arguments)]
pub fn r3d_sky(on: i64, tr: f32, tg: f32, tb: f32, hr: f32, hg: f32, hb: f32) {
    with_gfx((), |gf| {
        let (_, _, s) = gf.scene_mut();
        s.set_sky(on != 0, Vec3::new(tr, tg, tb), Vec3::new(hr, hg, hb));
    });
}
pub fn r3d_shadows(on: i64) {
    with_gfx((), |gf| {
        let (_, _, s) = gf.scene_mut();
        s.set_shadows(on != 0);
    });
}
pub fn r3d_ssao(on: i64) {
    with_gfx((), |gf| {
        let (_, _, s) = gf.scene_mut();
        s.set_ssao(on != 0);
    });
}
pub fn r3d_viewmodel(on: i64) {
    with_gfx((), |gf| {
        let (_, _, s) = gf.scene_mut();
        s.set_viewmodel(on != 0);
    });
}
pub fn r3d_point_shadows(on: i64) {
    with_gfx((), |gf| {
        let (_, _, s) = gf.scene_mut();
        s.set_point_shadows(on != 0);
    });
}
pub fn r3d_clear_lights() {
    with_gfx((), |gf| {
        let (_, _, s) = gf.scene_mut();
        s.clear_point_lights();
    });
}
#[allow(clippy::too_many_arguments)]
pub fn r3d_point_light(x: f32, y: f32, z: f32, r: f32, g: f32, b: f32, range: f32, intensity: f32) {
    with_gfx((), |gf| {
        let (_, _, s) = gf.scene_mut();
        s.add_point_light(Vec3::new(x, y, z), Vec3::new(r, g, b), range, intensity);
    });
}
pub fn r3d_make_sprite(r: f32, g: f32, b: f32) -> i64 {
    with_gfx(-1, |gf| {
        let (d, q, s) = gf.scene_mut();
        s.make_sprite(d, q, [r, g, b])
    })
}
pub fn r3d_draw_billboard(handle: i64, x: f32, y: f32, z: f32, size: f32) {
    with_gfx((), |gf| {
        let (_, _, s) = gf.scene_mut();
        s.draw_billboard(handle, Vec3::new(x, y, z), size);
    });
}
#[allow(clippy::too_many_arguments)]
pub fn r3d_debug_line(
    ax: f32,
    ay: f32,
    az: f32,
    bx: f32,
    by: f32,
    bz: f32,
    r: f32,
    g: f32,
    b: f32,
) {
    with_gfx((), |gf| {
        let (_, _, s) = gf.scene_mut();
        s.renderer.debug_line(
            Vec3::new(ax, ay, az),
            Vec3::new(bx, by, bz),
            Vec3::new(r, g, b),
        );
    });
}
pub fn r3d_frustum_cull(on: i64) {
    with_gfx((), |gf| {
        let (_, _, s) = gf.scene_mut();
        s.renderer.set_frustum_cull(on != 0);
    });
}

/// Draw a model's skeleton as debug bone lines at (px,py,pz)/yaw/scale, in the
/// current pose - for headless rig/hitbox visual audits.
#[allow(clippy::too_many_arguments)]
pub fn r3d_debug_skeleton(
    handle: i64,
    px: f32,
    py: f32,
    pz: f32,
    yaw: f32,
    scale: f32,
    r: f32,
    g: f32,
    b: f32,
) {
    with_gfx((), |gf| {
        let (_, _, s) = gf.scene_mut();
        let m = Mat4::from_scale_rotation_translation(
            Vec3::splat(scale),
            Quat::from_euler(EulerRot::YXZ, yaw, 0.0, 0.0),
            Vec3::new(px, py, pz),
        );
        s.debug_skeleton(handle, m, Vec3::new(r, g, b));
    });
}
pub fn r3d_begin() {
    with_gfx((), |gf| {
        let (_, _, s) = gf.scene_mut();
        s.begin();
    });
}

/// Queue the heightmap terrain for this frame, at the level of detail the
/// current camera calls for. Like `r3d_draw`, it belongs between `r3d_begin` and
/// `r3d_present`.
///
/// The runtime hands its heightfield and albedo in on EVERY call rather than
/// once at load: the scene does not exist until the window (or the headless
/// device) does, so a program that loads its terrain before opening a window
/// would otherwise hand it to nothing. Both installs are no-ops once the scene
/// already holds that heightfield and color, so the steady-state cost is two
/// comparisons.
pub fn terrain_draw(field: Arc<aurora_render3d::Heightfield>, color: [f32; 3]) {
    with_gfx((), |gf| {
        let (d, q, s) = gf.scene_mut();
        s.set_terrain(d, q, field);
        s.set_terrain_color(d, q, color);
        s.draw_terrain(d, q);
    });
}

/// Queue a model at position (px,py,pz), Euler rotation (yaw,pitch,roll radians),
/// and uniform `scale`.
#[allow(clippy::too_many_arguments)]
pub fn r3d_draw(
    handle: i64,
    px: f32,
    py: f32,
    pz: f32,
    yaw: f32,
    pitch: f32,
    roll: f32,
    scale: f32,
) {
    with_gfx((), |gf| {
        let (_, _, s) = gf.scene_mut();
        let m = Mat4::from_scale_rotation_translation(
            Vec3::splat(scale),
            Quat::from_euler(EulerRot::YXZ, yaw, pitch, roll),
            Vec3::new(px, py, pz),
        );
        s.draw(handle, m);
    });
}

/// Queue a model at position (px,py,pz) with an explicit unit quaternion (qx,qy,qz,qw) and uniform
/// `scale`. Used for free-tumbling rigid bodies (crates) where a euler triple would be lossy/ambiguous.
#[allow(clippy::too_many_arguments)]
pub fn r3d_draw_quat(
    handle: i64,
    px: f32,
    py: f32,
    pz: f32,
    qx: f32,
    qy: f32,
    qz: f32,
    qw: f32,
    scale: f32,
) {
    with_gfx((), |gf| {
        let (_, _, s) = gf.scene_mut();
        let m = Mat4::from_scale_rotation_translation(
            Vec3::splat(scale),
            Quat::from_xyzw(qx, qy, qz, qw).normalize(),
            Vec3::new(px, py, pz),
        );
        s.draw(handle, m);
    });
}

// The parameter list mirrors this builtin's row in `aurora-abi`, which is
// the single source of truth for its signature; grouping the arguments
// would break the 1:1 correspondence the table is built on.
#[allow(clippy::too_many_arguments)]
pub fn r3d_draw_tint(
    handle: i64,
    px: f32,
    py: f32,
    pz: f32,
    yaw: f32,
    pitch: f32,
    roll: f32,
    scale: f32,
    r: f32,
    g: f32,
    b: f32,
) {
    with_gfx((), |gf| {
        let (_, _, s) = gf.scene_mut();
        let m = Mat4::from_scale_rotation_translation(
            Vec3::splat(scale),
            Quat::from_euler(EulerRot::YXZ, yaw, pitch, roll),
            Vec3::new(px, py, pz),
        );
        s.draw_tint(handle, m, [r, g, b]);
    });
}

/// Draw a model with an energy-shield Fresnel rim (cyan crackle): strength 0..1, animated by time.
#[allow(clippy::too_many_arguments)]
pub fn r3d_draw_shield(
    handle: i64,
    px: f32,
    py: f32,
    pz: f32,
    yaw: f32,
    pitch: f32,
    roll: f32,
    scale: f32,
    strength: f32,
    time: f32,
) {
    with_gfx((), |gf| {
        let (_, _, s) = gf.scene_mut();
        let m = Mat4::from_scale_rotation_translation(
            Vec3::splat(scale),
            Quat::from_euler(EulerRot::YXZ, yaw, pitch, roll),
            Vec3::new(px, py, pz),
        );
        s.draw_shield(handle, m, strength, time);
    });
}

/// Draw `weapon` attached to `joint` of `host` (posed at the h* transform), with the
/// weapon's own o* offset relative to that bone. So a 3rd-person weapon rides the hand.
#[allow(clippy::too_many_arguments)]
pub fn r3d_draw_on_joint(
    weapon: i64,
    host: i64,
    joint: i64,
    hx: f32,
    hy: f32,
    hz: f32,
    hyaw: f32,
    hpitch: f32,
    hroll: f32,
    hscale: f32,
    ox: f32,
    oy: f32,
    oz: f32,
    oyaw: f32,
    opitch: f32,
    oroll: f32,
    oscale: f32,
) {
    with_gfx((), |gf| {
        let (_, _, s) = gf.scene_mut();
        let host_xform = Mat4::from_scale_rotation_translation(
            Vec3::splat(hscale),
            Quat::from_euler(EulerRot::YXZ, hyaw, hpitch, hroll),
            Vec3::new(hx, hy, hz),
        );
        let local = Mat4::from_scale_rotation_translation(
            Vec3::splat(oscale),
            Quat::from_euler(EulerRot::YXZ, oyaw, opitch, oroll),
            Vec3::new(ox, oy, oz),
        );
        s.draw_on_joint(weapon, host, joint, host_xform, local);
    });
}

/// Draw `armor` skinned by `host`'s current pose (armour carries per-vertex
/// weights in the host's joint order). Lets fitted gear deform with the body.
#[allow(clippy::too_many_arguments)]
pub fn r3d_draw_skinned(
    armor: i64,
    host: i64,
    x: f32,
    y: f32,
    z: f32,
    yaw: f32,
    pitch: f32,
    roll: f32,
    scale: f32,
) {
    with_gfx((), |gf| {
        let (_, _, s) = gf.scene_mut();
        let xf = Mat4::from_scale_rotation_translation(
            Vec3::splat(scale),
            Quat::from_euler(EulerRot::YXZ, yaw, pitch, roll),
            Vec3::new(x, y, z),
        );
        s.draw_skinned(armor, host, xf);
    });
}

/// Print every joint index + name of a model to stdout (bone-discovery helper).
pub fn r3d_joint_dump(host: i64) {
    with_gfx((), |gf| {
        let (_, _, s) = gf.scene_mut();
        s.dump_joints(host);
    });
}
/// Model-space position of `joint` in the host's current pose, component `axis` (0=x,1=y,2=z).
pub fn r3d_joint_pos(host: i64, joint: i64, axis: i64) -> f32 {
    with_gfx(0.0f32, |gf| {
        let (_, _, s) = gf.scene_mut();
        let p = s.joint_pos(host, joint).unwrap_or([0.0, 0.0, 0.0]);
        p[(axis.max(0) as usize).min(2)]
    })
}

pub fn r3d_anim_play(handle: i64, clip: i64, looping: i64, speed: f32, fade: f32) {
    with_gfx((), |gf| {
        let (_, _, s) = gf.scene_mut();
        s.anim_play(handle, clip, looping != 0, speed, fade);
    });
}
pub fn r3d_anim_update(handle: i64, dt: f32) {
    with_gfx((), |gf| {
        let (_, _, s) = gf.scene_mut();
        s.anim_update(handle, dt);
    });
}
pub fn r3d_anim_play_upper(
    handle: i64,
    clip: i64,
    looping: i64,
    speed: f32,
    fade: f32,
    mask_root: i64,
) {
    with_gfx((), |gf| {
        let (_, _, s) = gf.scene_mut();
        s.anim_play_upper(handle, clip, looping != 0, speed, fade, mask_root);
    });
}
pub fn r3d_anim_aim_upper(
    handle: i64,
    clip_a: i64,
    clip_b: i64,
    weight: f32,
    speed: f32,
    fade: f32,
    mask_root: i64,
) {
    with_gfx((), |gf| {
        let (_, _, s) = gf.scene_mut();
        s.anim_aim_upper(handle, clip_a, clip_b, weight, speed, fade, mask_root);
    });
}
pub fn r3d_anim_blend(handle: i64, clip_a: i64, clip_b: i64, weight: f32, speed: f32, fade: f32) {
    with_gfx((), |gf| {
        let (_, _, s) = gf.scene_mut();
        s.anim_blend(handle, clip_a, clip_b, weight, speed, fade);
    });
}
pub fn r3d_anim_seek_upper(handle: i64, t: f32) {
    with_gfx((), |gf| {
        let (_, _, s) = gf.scene_mut();
        s.anim_seek_upper(handle, t);
    });
}
pub fn r3d_pose_bone(handle: i64, joint: i64, rx: f32, ry: f32, rz: f32) {
    with_gfx((), |gf| {
        let (_, _, s) = gf.scene_mut();
        s.pose_bone(handle, joint, rx, ry, rz);
    });
}
pub fn r3d_clear_pose(handle: i64) {
    with_gfx((), |gf| {
        let (_, _, s) = gf.scene_mut();
        s.clear_pose(handle);
    });
}
pub fn r3d_hide_joint(handle: i64, joint: i64) {
    with_gfx((), |gf| {
        let (_, _, s) = gf.scene_mut();
        s.hide_joint(handle, joint);
    });
}

/// Show every joint again, undoing `r3d_hide_joint`. Hiding accumulates into a
/// mask, so without this a model handle could never get its body back and a
/// pooled character had to be reloaded from disk to be reused.
pub fn r3d_show_joints(handle: i64) {
    with_gfx((), |gf| {
        let (_, _, s) = gf.scene_mut();
        s.show_joints(handle);
    });
}
pub fn r3d_anim_stop_upper(handle: i64, fade: f32) {
    with_gfx((), |gf| {
        let (_, _, s) = gf.scene_mut();
        s.anim_stop_upper(handle, fade);
    });
}
pub fn r3d_clip_count(handle: i64) -> i64 {
    with_gfx(0, |gf| {
        let (_, _, s) = gf.scene_mut();
        s.clip_count(handle)
    })
}

/// The asset's own name for clip `i`, or "" for a stale handle / bad index.
pub fn r3d_clip_name(handle: i64, i: i64) -> String {
    with_gfx(String::new(), |gf| {
        let (_, _, s) = gf.scene_mut();
        s.clip_name(handle, i).unwrap_or("").to_string()
    })
}

/// Index of the clip called `name`, or -1 when the model has no such clip.
pub fn r3d_clip_index(handle: i64, name: &str) -> i64 {
    with_gfx(-1, |gf| {
        let (_, _, s) = gf.scene_mut();
        s.clip_index(handle, name)
    })
}

/// Index of the joint called `name`, or -1 when the model has no such joint.
pub fn r3d_joint_index(handle: i64, name: &str) -> i64 {
    with_gfx(-1, |gf| {
        let (_, _, s) = gf.scene_mut();
        s.joint_index(handle, name)
    })
}

/// The name of joint `i`, or "" for a stale handle / bad index.
pub fn r3d_joint_name(handle: i64, i: i64) -> String {
    with_gfx(String::new(), |gf| {
        let (_, _, s) = gf.scene_mut();
        s.joint_name(handle, i).unwrap_or("").to_string()
    })
}

/// Render the queued 3D scene to the window and overlay `hud_rgba` (the CPU
/// framebuffer; black is transparent), pump events, and return whether the
/// window is still open.
pub fn r3d_present(hud_rgba: &[u8], hud_w: u32, hud_h: u32) -> bool {
    IMM.with(|s| {
        let mut slot = s.borrow_mut();
        let Some((event_loop, app)) = slot.as_mut() else {
            return false;
        };
        // Reset per-frame input before pumping so this frame's mouse/scroll delta
        // survives for the caller to read after present returns (see `present`).
        reset_frame_input(app);
        if let Some(el) = event_loop.as_mut() {
            el.pump_app_events(Some(Duration::ZERO), app);
        }
        end_frame(app);
        if app.open && !app.headless {
            let (sli, slt) = (app.sl_intensity, app.sl_time);
            let (dv, dh, ddx, ddy, doc) =
                (app.dmg_vig, app.dmg_hit, app.dmg_dx, app.dmg_dy, app.dmg_oc);
            let blur = app.blur;
            if let Some(g) = app.gfx.as_mut() {
                g.present_scene(
                    hud_rgba, hud_w, hud_h, sli, slt, dv, dh, ddx, ddy, doc, blur,
                );
            }
        }
        app.open
    })
}

/// Per-present bookkeeping shared by both present paths: run the input tape
/// (record after the pump so real events are captured; replay overwrites the
/// input state), advance the frame counter, and enforce AURORA_MAX_FRAMES.
fn end_frame(app: &mut ImmApp) {
    tape_tick(app);
    app.frame += 1;
    if app.max_frames > 0 && app.frame >= app.max_frames {
        app.open = false;
    }
}

// --- input injection + record/replay tape -----------------------------------
//
// inject_* write the same state real events write, so scripted input is
// indistinguishable from a player (works windowed too - demo playback). A tape
// (AURORA_INPUT_RECORD=path / AURORA_INPUT_REPLAY=path) captures/replays the
// full per-frame input state; combined with srand + fixed dt a replayed
// session reproduces bit-for-bit. Replay ends -> present reports "closed", so
// the game loop exits on its own.
//
// Tape format (line-based, deterministic, greppable):
//   AURTAPE 1
//   f=<n>;k=<code,code,..>;dx=<f>;dy=<f>;mx=<i>;my=<i>;s=<f>;b=<bitmask>;t=<c,c,..>

enum Tape {
    Off,
    Record {
        out: std::io::BufWriter<std::fs::File>,
    },
    Replay {
        frames: Vec<TapeFrame>,
        next: usize,
    },
}

#[derive(Default)]
struct TapeFrame {
    keys: Vec<u32>,
    dx: f64,
    dy: f64,
    mx: i64,
    my: i64,
    scroll: f64,
    buttons: u32,
    typed: Vec<u32>,
}

impl Tape {
    fn from_env() -> Tape {
        if let Ok(path) = std::env::var("AURORA_INPUT_REPLAY") {
            match std::fs::read_to_string(&path) {
                Ok(text) => {
                    let frames = parse_tape(&text);
                    eprintln!(
                        "aurora: replaying input tape {path} ({} frames)",
                        frames.len()
                    );
                    return Tape::Replay { frames, next: 0 };
                }
                Err(e) => eprintln!("aurora: cannot read input tape {path}: {e}"),
            }
        }
        if let Ok(path) = std::env::var("AURORA_INPUT_RECORD") {
            match std::fs::File::create(&path) {
                Ok(f) => {
                    use std::io::Write;
                    let mut out = std::io::BufWriter::new(f);
                    let _ = writeln!(out, "AURTAPE 1");
                    eprintln!("aurora: recording input tape to {path}");
                    return Tape::Record { out };
                }
                Err(e) => eprintln!("aurora: cannot create input tape {path}: {e}"),
            }
        }
        Tape::Off
    }
}

fn parse_tape(text: &str) -> Vec<TapeFrame> {
    let mut frames = Vec::new();
    for line in text.lines().skip(1) {
        if line.trim().is_empty() {
            continue;
        }
        let mut fr = TapeFrame::default();
        for field in line.split(';') {
            let Some((k, v)) = field.split_once('=') else {
                continue;
            };
            match k {
                "k" => fr.keys = v.split(',').filter_map(|s| s.parse().ok()).collect(),
                "dx" => fr.dx = v.parse().unwrap_or(0.0),
                "dy" => fr.dy = v.parse().unwrap_or(0.0),
                "mx" => fr.mx = v.parse().unwrap_or(0),
                "my" => fr.my = v.parse().unwrap_or(0),
                "s" => fr.scroll = v.parse().unwrap_or(0.0),
                "b" => fr.buttons = v.parse().unwrap_or(0),
                "t" => fr.typed = v.split(',').filter_map(|s| s.parse().ok()).collect(),
                _ => {}
            }
        }
        frames.push(fr);
    }
    frames
}

/// Aurora key-code range covered by snapshots/tapes (0..=65, see code_to_key).
const KEY_CODE_MAX: u32 = 65;

fn tape_tick(app: &mut ImmApp) {
    match &mut app.tape {
        Tape::Off => {}
        Tape::Record { out } => {
            use std::io::Write;
            let keys: Vec<String> = (0..=KEY_CODE_MAX)
                .filter(|&c| {
                    code_to_key(c)
                        .map(|k| app.keys.contains(&k))
                        .unwrap_or(false)
                })
                .map(|c| c.to_string())
                .collect();
            let buttons = (app.mouse_down as u32)
                | (app.mouse_right as u32) << 1
                | (app.mouse_middle as u32) << 2
                | (app.mouse_back as u32) << 3
                | (app.mouse_forward as u32) << 4;
            let typed: Vec<String> = app.typed.iter().map(|c| c.to_string()).collect();
            let _ = writeln!(
                out,
                "f={};k={};dx={};dy={};mx={};my={};s={};b={};t={}",
                app.frame,
                keys.join(","),
                app.mouse_dx,
                app.mouse_dy,
                app.mouse.0,
                app.mouse.1,
                app.scroll,
                buttons,
                typed.join(",")
            );
        }
        Tape::Replay { frames, next } => {
            if *next >= frames.len() {
                app.open = false;
                return;
            }
            let fr = &frames[*next];
            *next += 1;
            app.keys.clear();
            for &c in &fr.keys {
                if let Some(k) = code_to_key(c) {
                    app.keys.insert(k);
                }
            }
            app.mouse_dx = fr.dx;
            app.mouse_dy = fr.dy;
            app.mouse = (fr.mx, fr.my);
            app.scroll = fr.scroll;
            app.mouse_down = fr.buttons & 1 != 0;
            app.mouse_right = fr.buttons & 2 != 0;
            app.mouse_middle = fr.buttons & 4 != 0;
            app.mouse_back = fr.buttons & 8 != 0;
            app.mouse_forward = fr.buttons & 16 != 0;
            app.typed = fr.typed.clone();
        }
    }
}

/// Press or release a key by Aurora key code (same state real events write).
pub fn inject_key(code: u32, down: bool) {
    IMM.with(|s| {
        if let Some((_, app)) = s.borrow_mut().as_mut() {
            if let Some(k) = code_to_key(code) {
                if down {
                    app.keys.insert(k);
                } else {
                    app.keys.remove(&k);
                }
            }
        }
    });
}

/// Add raw mouse-look motion (accumulates until the next present).
pub fn inject_mouse_move(dx: f64, dy: f64) {
    IMM.with(|s| {
        if let Some((_, app)) = s.borrow_mut().as_mut() {
            app.mouse_dx += dx;
            app.mouse_dy += dy;
        }
    });
}

/// Set the cursor position in framebuffer pixels.
pub fn inject_mouse_pos(x: i64, y: i64) {
    IMM.with(|s| {
        if let Some((_, app)) = s.borrow_mut().as_mut() {
            app.mouse = (x, y);
        }
    });
}

/// Press or release a mouse button (0 left, 1 right, 2 middle, 3 back, 4 forward).
pub fn inject_mouse_button(b: u32, down: bool) {
    IMM.with(|s| {
        if let Some((_, app)) = s.borrow_mut().as_mut() {
            match b {
                1 => app.mouse_right = down,
                2 => app.mouse_middle = down,
                3 => app.mouse_back = down,
                4 => app.mouse_forward = down,
                _ => app.mouse_down = down,
            }
        }
    });
}

/// Add scroll-wheel delta (accumulates until the next present).
pub fn inject_scroll(dy: f64) {
    IMM.with(|s| {
        if let Some((_, app)) = s.borrow_mut().as_mut() {
            app.scroll += dy;
        }
    });
}

/// Queue a typed character code for text fields (Backspace = 8).
pub fn inject_char(c: u32) {
    IMM.with(|s| {
        if let Some((_, app)) = s.borrow_mut().as_mut() {
            app.typed.push(c);
        }
    });
}

/// Render the queued 3D scene offscreen and save it as a PNG with the HUD
/// framebuffer composited on top (black = transparent, like the live overlay).
/// Headless-only: the harness's eyes. Returns 1 on success, 0 on failure.
pub fn r3d_capture(
    path: &str,
    hud_rgba: &[u8],
    hud_w: u32,
    hud_h: u32,
    out_w: u32,
    out_h: u32,
) -> i64 {
    IMM.with(|s| {
        let mut slot = s.borrow_mut();
        let Some((_, app)) = slot.as_mut() else {
            return 0;
        };
        if !app.headless {
            eprintln!("aurora: r3d_capture is headless-only (set AURORA_HEADLESS=1)");
            return 0;
        }
        if app.hgfx.is_none() && app.open {
            match HeadlessGfx::new(app.width, app.height) {
                Some(h) => app.hgfx = Some(h),
                None => {
                    eprintln!("aurora: HEADLESS-NO-GPU");
                    app.open = false;
                    return 0;
                }
            }
        }
        let Some(h) = app.hgfx.as_mut() else { return 0 };
        let (w, hh) = (out_w.clamp(16, 4096), out_h.clamp(16, 4096));
        let (device, queue, scene) = h.scene_mut();
        // Match the camera aspect to the capture size before rendering.
        scene.resize(device, w, hh);
        let clear = scene.clear_color();
        let mut px =
            aurora_render3d::render_offscreen(&mut scene.renderer, device, queue, w, hh, clear);
        // Composite the HUD (CPU framebuffer) over the render: nearest-neighbor
        // scale, near-black is the transparent key (same threshold as HUD_WGSL).
        if hud_w > 0 && hud_h > 0 && hud_rgba.len() >= (hud_w * hud_h * 4) as usize {
            for y in 0..hh {
                let sy = (y as u64 * hud_h as u64 / hh as u64) as u32;
                for x in 0..w {
                    let sx = (x as u64 * hud_w as u64 / w as u64) as u32;
                    let so = ((sy * hud_w + sx) * 4) as usize;
                    let (r, g, b) = (hud_rgba[so], hud_rgba[so + 1], hud_rgba[so + 2]);
                    if r as u32 + g as u32 + b as u32 >= 3 {
                        let o = ((y * w + x) * 4) as usize;
                        px[o] = r;
                        px[o + 1] = g;
                        px[o + 2] = b;
                        px[o + 3] = 255;
                    }
                }
            }
        }
        for p in px.chunks_exact_mut(4) {
            p[3] = 255;
        }
        if let Some(parent) = std::path::Path::new(path).parent() {
            if !parent.as_os_str().is_empty() {
                let _ = std::fs::create_dir_all(parent);
            }
        }
        match image::save_buffer(path, &px, w, hh, image::ExtendedColorType::Rgba8) {
            Ok(()) => 1,
            Err(e) => {
                eprintln!("aurora: r3d_capture {path}: {e}");
                0
            }
        }
    })
}

/// Set the fullscreen blur radius in pixels (0 = off). Used for the paused/menu
/// backdrop: the scene keeps rendering (and, in multiplayer, simulating) but is
/// blurred so the menu reads on top.
pub fn blur(radius: f32) {
    IMM.with(|s| {
        if let Some((_, app)) = s.borrow_mut().as_mut() {
            app.blur = radius;
        }
    });
}

/// Set the speed/wind-lines overlay intensity (0..1) and animation time.
pub fn speedlines(intensity: f32, time: f32) {
    IMM.with(|s| {
        if let Some((_, app)) = s.borrow_mut().as_mut() {
            app.sl_intensity = intensity;
            app.sl_time = time;
        }
    });
}

/// Set the damage overlay: low-health vignette (0..1), directional hit glow (0..1),
/// and the hit direction in screen space (dx, dy).
pub fn damage(vig: f32, hit: f32, dx: f32, dy: f32, oc: f32) {
    IMM.with(|s| {
        if let Some((_, app)) = s.borrow_mut().as_mut() {
            app.dmg_vig = vig;
            app.dmg_hit = hit;
            app.dmg_dx = dx;
            app.dmg_dy = dy;
            app.dmg_oc = oc;
        }
    });
}

/// Current window inner size in physical pixels (the surface size). 0 before the
/// window exists. Lets a game size its HUD framebuffer to the live window.
pub fn surface_w() -> u32 {
    IMM.with(|s| {
        s.borrow()
            .as_ref()
            .map(|(_, a)| a.win_size.0 as u32)
            .unwrap_or(0)
    })
}
pub fn surface_h() -> u32 {
    IMM.with(|s| {
        s.borrow()
            .as_ref()
            .map(|(_, a)| a.win_size.1 as u32)
            .unwrap_or(0)
    })
}

/// Project a world point to framebuffer pixel coords; returns `(x, y, visible)`
/// where `visible` is 0 if the point is behind the camera or off-screen.
pub fn r3d_world_to_screen(wx: f32, wy: f32, wz: f32) -> (f32, f32, bool) {
    with_gfx((0.0, 0.0, false), |gf| {
        let (_, _, s) = gf.scene_mut();
        match s.world_to_screen(Vec3::new(wx, wy, wz)) {
            Some((x, y)) => (x, y, true),
            None => (0.0, 0.0, false),
        }
    })
}

/// Aurora key codes (stable integers passed from `.aur` code). 0-9 are the
/// classic movement/action keys; 10-19 modifiers/common action keys; 30-39 the
/// number row (1..9,0); 40-65 the letters A..Z.
fn code_to_key(code: u32) -> Option<KeyCode> {
    use KeyCode::*;
    const LETTERS: [KeyCode; 26] = [
        KeyA, KeyB, KeyC, KeyD, KeyE, KeyF, KeyG, KeyH, KeyI, KeyJ, KeyK, KeyL, KeyM, KeyN, KeyO,
        KeyP, KeyQ, KeyR, KeyS, KeyT, KeyU, KeyV, KeyW, KeyX, KeyY, KeyZ,
    ];
    const DIGITS: [KeyCode; 10] = [
        Digit1, Digit2, Digit3, Digit4, Digit5, Digit6, Digit7, Digit8, Digit9, Digit0,
    ];
    Some(match code {
        0 => ArrowLeft,
        1 => ArrowRight,
        2 => ArrowUp,
        3 => ArrowDown,
        4 => Space,
        5 => KeyW,
        6 => KeyA,
        7 => KeyS,
        8 => KeyD,
        9 => Enter,
        10 => ShiftLeft,
        11 => ControlLeft,
        12 => AltLeft,
        13 => Tab,
        14 => KeyR,
        15 => KeyE,
        16 => KeyQ,
        17 => KeyF,
        18 => KeyC,
        19 => KeyV,
        20 => Escape,
        30..=39 => DIGITS[(code - 30) as usize],
        40..=65 => LETTERS[(code - 40) as usize],
        _ => return None,
    })
}
