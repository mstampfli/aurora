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
    static IMM: RefCell<Option<(EventLoop<()>, ImmApp)>> = const { RefCell::new(None) };
}

/// Leak the window + GPU state instead of dropping it. Call right before the process
/// exits: wgpu/winit panic if their state is torn down in a thread-local destructor at
/// process exit ("thread local panicked on drop"). Leaking it makes shutdown graceful.
pub fn imm_leak() {
    IMM.with(|s| {
        if let Some(inner) = s.borrow_mut().take() {
            std::mem::forget(inner);
        }
    });
}

/// Open a window backing a `width`×`height` framebuffer. Replaces any prior one.
pub fn open(width: u32, height: u32) {
    let event_loop = match EventLoop::new() {
        Ok(e) => e,
        Err(e) => {
            eprintln!("aurora-window: event loop creation failed: {e}");
            return;
        }
    };
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
    };
    IMM.with(|s| *s.borrow_mut() = Some((event_loop, app)));
}

/// The raw mouse motion accumulated this frame. Reset at the next present.
pub fn mouse_delta() -> (f64, f64) {
    IMM.with(|s| s.borrow().as_ref().map(|(_, app)| (app.mouse_dx, app.mouse_dy)).unwrap_or((0.0, 0.0)))
}

/// The scroll-wheel delta accumulated this frame. Reset at the next present.
pub fn scroll() -> f64 {
    IMM.with(|s| s.borrow().as_ref().map(|(_, app)| app.scroll).unwrap_or(0.0))
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
        let Some((_, app)) = slot.as_mut() else { return };
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
        let Some((event_loop, app)) = slot.as_mut() else { return false };
        // Clear last frame's per-frame input (mouse delta, scroll) BEFORE pumping,
        // so the deltas accumulated this pump survive for the caller to read after
        // present returns. Resetting after the pump would zero them first.
        reset_frame_input(app);
        event_loop.pump_app_events(Some(Duration::ZERO), app);
        if app.open {
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
    let Some(key) = code_to_key(code) else { return false };
    IMM.with(|s| s.borrow().as_ref().map(|(_, app)| app.keys.contains(&key)).unwrap_or(false))
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
            .map(|(_, app)| if app.typed.is_empty() { 0 } else { app.typed.remove(0) as i64 })
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

fn with_gfx<R>(default: R, f: impl FnOnce(&mut Gfx) -> R) -> R {
    IMM.with(|s| {
        let mut slot = s.borrow_mut();
        let Some((event_loop, app)) = slot.as_mut() else { return default };
        if app.gfx.is_none() && app.open {
            event_loop.pump_app_events(Some(Duration::ZERO), app);
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
pub fn r3d_debug_line(ax: f32, ay: f32, az: f32, bx: f32, by: f32, bz: f32, r: f32, g: f32, b: f32) {
    with_gfx((), |gf| {
        let (_, _, s) = gf.scene_mut();
        s.renderer.debug_line(Vec3::new(ax, ay, az), Vec3::new(bx, by, bz), Vec3::new(r, g, b));
    });
}
pub fn r3d_frustum_cull(on: i64) {
    with_gfx((), |gf| {
        let (_, _, s) = gf.scene_mut();
        s.renderer.set_frustum_cull(on != 0);
    });
}
pub fn r3d_begin() {
    with_gfx((), |gf| {
        let (_, _, s) = gf.scene_mut();
        s.begin();
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
pub fn r3d_anim_play_upper(handle: i64, clip: i64, looping: i64, speed: f32, fade: f32, mask_root: i64) {
    with_gfx((), |gf| {
        let (_, _, s) = gf.scene_mut();
        s.anim_play_upper(handle, clip, looping != 0, speed, fade, mask_root);
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

/// Render the queued 3D scene to the window and overlay `hud_rgba` (the CPU
/// framebuffer; black is transparent), pump events, and return whether the
/// window is still open.
pub fn r3d_present(hud_rgba: &[u8], hud_w: u32, hud_h: u32) -> bool {
    IMM.with(|s| {
        let mut slot = s.borrow_mut();
        let Some((event_loop, app)) = slot.as_mut() else { return false };
        // Reset per-frame input before pumping so this frame's mouse/scroll delta
        // survives for the caller to read after present returns (see `present`).
        reset_frame_input(app);
        event_loop.pump_app_events(Some(Duration::ZERO), app);
        if app.open {
            let (sli, slt) = (app.sl_intensity, app.sl_time);
            let (dv, dh, ddx, ddy, doc) =
                (app.dmg_vig, app.dmg_hit, app.dmg_dx, app.dmg_dy, app.dmg_oc);
            let blur = app.blur;
            if let Some(g) = app.gfx.as_mut() {
                g.present_scene(hud_rgba, hud_w, hud_h, sli, slt, dv, dh, ddx, ddy, doc, blur);
            }
        }
        app.open
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
    IMM.with(|s| s.borrow().as_ref().map(|(_, a)| a.win_size.0 as u32).unwrap_or(0))
}
pub fn surface_h() -> u32 {
    IMM.with(|s| s.borrow().as_ref().map(|(_, a)| a.win_size.1 as u32).unwrap_or(0))
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
    const DIGITS: [KeyCode; 10] =
        [Digit1, Digit2, Digit3, Digit4, Digit5, Digit6, Digit7, Digit8, Digit9, Digit0];
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
