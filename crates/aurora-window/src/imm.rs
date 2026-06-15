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
    open: bool,
    /// Mouse position in framebuffer pixels, and left-button state.
    mouse: (i64, i64),
    mouse_down: bool,
    /// Window inner size (to map cursor coords back to framebuffer pixels).
    win_size: (f64, f64),
}

impl ApplicationHandler for ImmApp {
    fn resumed(&mut self, el: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        let attrs = Window::default_attributes()
            .with_title("Aurora")
            .with_inner_size(winit::dpi::LogicalSize::new(self.width * 3, self.height * 3));
        if let Ok(w) = el.create_window(attrs) {
            let w = Arc::new(w);
            if let Ok(g) = Gfx::new(w.clone(), self.width, self.height) {
                self.gfx = Some(g);
                self.window = Some(w);
            } else {
                self.open = false;
            }
        } else {
            self.open = false;
        }
    }

    fn window_event(&mut self, _el: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => self.open = false,
            WindowEvent::Resized(size) => {
                self.win_size = (size.width.max(1) as f64, size.height.max(1) as f64);
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
            WindowEvent::MouseInput { state, button: MouseButton::Left, .. } => {
                self.mouse_down = state == ElementState::Pressed;
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if let PhysicalKey::Code(code) = event.physical_key {
                    if code == KeyCode::Escape {
                        self.open = false;
                    }
                    if event.state == ElementState::Pressed {
                        self.keys.insert(code);
                    } else {
                        self.keys.remove(&code);
                    }
                }
            }
            _ => {}
        }
    }
}

thread_local! {
    static IMM: RefCell<Option<(EventLoop<()>, ImmApp)>> = const { RefCell::new(None) };
}

/// Open a window backing a `width`×`height` framebuffer. Replaces any prior one.
pub fn open(width: u32, height: u32) {
    let event_loop = match EventLoop::new() {
        Ok(e) => e,
        Err(_) => return,
    };
    let app = ImmApp {
        width: width.max(1),
        height: height.max(1),
        window: None,
        gfx: None,
        keys: HashSet::new(),
        open: true,
        mouse: (0, 0),
        mouse_down: false,
        win_size: ((width.max(1) * 3) as f64, (height.max(1) * 3) as f64),
    };
    IMM.with(|s| *s.borrow_mut() = Some((event_loop, app)));
}

/// Pump events, present `rgba` (tight `width*height*4` bytes), and return whether
/// the window is still open. Returns `false` if no window was opened.
pub fn present(rgba: &[u8]) -> bool {
    IMM.with(|s| {
        let mut slot = s.borrow_mut();
        let Some((event_loop, app)) = slot.as_mut() else { return false };
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

pub fn r3d_anim_play(handle: i64, clip: i64, looping: i64, speed: f32) {
    with_gfx((), |gf| {
        let (_, _, s) = gf.scene_mut();
        s.anim_play(handle, clip, looping != 0, speed);
    });
}
pub fn r3d_anim_update(handle: i64, dt: f32) {
    with_gfx((), |gf| {
        let (_, _, s) = gf.scene_mut();
        s.anim_update(handle, dt);
    });
}
pub fn r3d_clip_count(handle: i64) -> i64 {
    with_gfx(0, |gf| {
        let (_, _, s) = gf.scene_mut();
        s.clip_count(handle)
    })
}

/// Render the queued 3D scene to the window and pump events; returns whether the
/// window is still open.
pub fn r3d_present() -> bool {
    IMM.with(|s| {
        let mut slot = s.borrow_mut();
        let Some((event_loop, app)) = slot.as_mut() else { return false };
        event_loop.pump_app_events(Some(Duration::ZERO), app);
        if app.open {
            if let Some(g) = app.gfx.as_mut() {
                g.present_scene();
            }
        }
        app.open
    })
}

/// Aurora key codes (stable integers passed from `.aur` code).
fn code_to_key(code: u32) -> Option<KeyCode> {
    Some(match code {
        0 => KeyCode::ArrowLeft,
        1 => KeyCode::ArrowRight,
        2 => KeyCode::ArrowUp,
        3 => KeyCode::ArrowDown,
        4 => KeyCode::Space,
        5 => KeyCode::KeyW,
        6 => KeyCode::KeyA,
        7 => KeyCode::KeyS,
        8 => KeyCode::KeyD,
        9 => KeyCode::Enter,
        _ => return None,
    })
}
