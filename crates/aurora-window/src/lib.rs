//! Real-time window + input for Aurora games (winit + wgpu).
//!
//! [`run`] opens a window and drives a frame loop: each frame your callback gets
//! the current [`Input`] and delta-time and draws into a [`Framebuffer`], which
//! is uploaded to the GPU and presented. The input/timing core lives in
//! [`input`] and is unit-tested without a window; this module is the windowing
//! and presentation glue.
//!
//! ```no_run
//! use aurora_window::{run, Key};
//! let mut x = 0.0f32;
//! run("demo", 320, 180, move |input, dt, fb| {
//!     if input.is_down(Key::Right) { x += 60.0 * dt; }
//!     fb.clear(aurora_gfx::Color::rgb(10, 12, 20));
//!     fb.set(x as i32, 90, aurora_gfx::Color::WHITE);
//! }).unwrap();
//! ```

mod imm;
mod input;
pub use imm::{
    imm_leak,
    grab_mouse as imm_grab_mouse, input_char as imm_input_char, key_down as imm_key_down,
    window_fullscreen as imm_window_fullscreen, mouse as imm_mouse,
    mouse_button as imm_mouse_button, mouse_delta as imm_mouse_delta, open as imm_open,
    present as imm_present, scroll as imm_scroll,
    r3d_anim_play as imm_r3d_anim_play, r3d_anim_play_upper as imm_r3d_anim_play_upper,
    r3d_anim_aim_upper as imm_r3d_anim_aim_upper, r3d_anim_blend as imm_r3d_anim_blend,
    r3d_anim_seek_upper as imm_r3d_anim_seek_upper,
    r3d_pose_bone as imm_r3d_pose_bone, r3d_clear_pose as imm_r3d_clear_pose,
    r3d_hide_joint as imm_r3d_hide_joint,
    r3d_anim_stop_upper as imm_r3d_anim_stop_upper, r3d_anim_update as imm_r3d_anim_update,
    r3d_begin as imm_r3d_begin, r3d_camera as imm_r3d_camera,
    r3d_camera_roll as imm_r3d_camera_roll, r3d_clear as imm_r3d_clear,
    r3d_clear_lights as imm_r3d_clear_lights, r3d_clip_count as imm_r3d_clip_count,
    r3d_debug_line as imm_r3d_debug_line, r3d_draw as imm_r3d_draw, r3d_draw_quat as imm_r3d_draw_quat,
    r3d_draw_tint as imm_r3d_draw_tint,
    r3d_draw_on_joint as imm_r3d_draw_on_joint, r3d_joint_dump as imm_r3d_joint_dump,
    r3d_joint_pos as imm_r3d_joint_pos,
    r3d_draw_shield as imm_r3d_draw_shield,
    r3d_draw_billboard as imm_r3d_draw_billboard, r3d_fog as imm_r3d_fog,
    r3d_frustum_cull as imm_r3d_frustum_cull, r3d_light as imm_r3d_light,
    r3d_load_model as imm_r3d_load_model, r3d_make_box as imm_r3d_make_box,
    r3d_make_box_emissive as imm_r3d_make_box_emissive,
    r3d_make_box_sized as imm_r3d_make_box_sized,
    r3d_make_plane as imm_r3d_make_plane, r3d_make_sphere as imm_r3d_make_sphere,
    r3d_make_sprite as imm_r3d_make_sprite, r3d_point_light as imm_r3d_point_light,
    r3d_point_shadows as imm_r3d_point_shadows, r3d_present as imm_r3d_present,
    r3d_shadows as imm_r3d_shadows, r3d_sky as imm_r3d_sky, r3d_ssao as imm_r3d_ssao,
    blur as imm_blur,
    damage as imm_damage, r3d_world_to_screen as imm_r3d_world_to_screen,
    speedlines as imm_speedlines, surface_h as imm_surface_h, surface_w as imm_surface_w,
};
pub use input::{Input, Key};

use std::sync::Arc;
use std::time::Instant;

use aurora_gfx::Framebuffer;
use winit::application::ApplicationHandler;
use winit::event::{ElementState, MouseButton, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

/// Build a winit event loop. On the free-unix backends (X11/Wayland) winit panics if the event loop
/// is created off the main thread, but the flag is just advisory there, so the Aurora runtime runs
/// the JIT'd program on a worker thread and opts into `any_thread`. macOS is the OPPOSITE: the event
/// loop MUST own the OS main thread and there is no opt-out, so on macOS aurorac runs the program on
/// the main thread (see aurorac/src/main.rs) and we build the loop plainly here. Windows is relaxed.
pub(crate) fn new_event_loop() -> Result<EventLoop<()>, winit::error::EventLoopError> {
    #[allow(unused_mut)]
    let mut builder = EventLoop::builder();
    #[cfg(any(
        target_os = "linux",
        target_os = "freebsd",
        target_os = "dragonfly",
        target_os = "netbsd",
        target_os = "openbsd"
    ))]
    {
        use winit::platform::wayland::EventLoopBuilderExtWayland;
        builder.with_any_thread(true);
    }
    builder.build()
}

/// Open a window of `width`×`height` (the framebuffer resolution) titled `title`
/// and run the frame loop until the user closes it (or presses Escape). `frame`
/// is called once per presented frame with input, delta-seconds, and the
/// framebuffer to draw into.
pub fn run(
    title: &str,
    width: u32,
    height: u32,
    frame: impl FnMut(&Input, f32, &mut Framebuffer) + 'static,
) -> Result<(), String> {
    let event_loop = new_event_loop().map_err(|e| format!("event loop: {e}"))?;
    let mut app = App {
        title: title.to_string(),
        width,
        height,
        window: None,
        gfx: None,
        fb: Framebuffer::new(width, height),
        input: Input::new(),
        last: Instant::now(),
        frame: Box::new(frame),
    };
    event_loop.run_app(&mut app).map_err(|e| format!("run: {e}"))
}

/// A built-in interactive demo: an arrow-key/WASD-controlled box with simple
/// momentum and wall bounces, drawn each frame. Opens a window and blocks until
/// closed. Useful as a smoke test that the real-time path works end to end.
pub fn demo() -> Result<(), String> {
    use aurora_gfx::Color;
    let (w, h) = (200u32, 150u32);
    let mut x = (w / 2) as f32;
    let mut y = (h / 2) as f32;
    let (mut vx, mut vy) = (48.0f32, 33.0f32);
    run("Aurora — live window (arrows/WASD move, Esc quits)", w, h, move |input, dt, fb| {
        let dt = dt.min(0.05); // clamp huge first-frame dt
        let accel = 320.0;
        if input.is_down(Key::Left) || input.is_down(Key::A) {
            vx -= accel * dt;
        }
        if input.is_down(Key::Right) || input.is_down(Key::D) {
            vx += accel * dt;
        }
        if input.is_down(Key::Up) || input.is_down(Key::W) {
            vy -= accel * dt;
        }
        if input.is_down(Key::Down) || input.is_down(Key::S) {
            vy += accel * dt;
        }
        x += vx * dt;
        y += vy * dt;
        // Bounce off the walls.
        if x < 4.0 {
            x = 4.0;
            vx = vx.abs();
        }
        if x > (w - 4) as f32 {
            x = (w - 4) as f32;
            vx = -vx.abs();
        }
        if y < 4.0 {
            y = 4.0;
            vy = vy.abs();
        }
        if y > (h - 4) as f32 {
            y = (h - 4) as f32;
            vy = -vy.abs();
        }

        fb.clear(Color::rgb(10, 12, 22));
        let c = Color::rgb(120, 200, 255);
        for dy in -3..=3 {
            for dx in -3..=3 {
                fb.set(x as i32 + dx, y as i32 + dy, c);
            }
        }
    })
}

struct App {
    title: String,
    width: u32,
    height: u32,
    window: Option<Arc<Window>>,
    gfx: Option<Gfx>,
    fb: Framebuffer,
    input: Input,
    last: Instant,
    frame: Box<dyn FnMut(&Input, f32, &mut Framebuffer)>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, el: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        let attrs = Window::default_attributes()
            .with_title(&self.title)
            .with_inner_size(winit::dpi::LogicalSize::new(self.width * 2, self.height * 2));
        let window = match el.create_window(attrs) {
            Ok(w) => Arc::new(w),
            Err(e) => {
                eprintln!("aurora-window: cannot create window: {e}");
                el.exit();
                return;
            }
        };
        match Gfx::new(window.clone(), self.width, self.height) {
            Ok(gfx) => self.gfx = Some(gfx),
            Err(e) => {
                eprintln!("aurora-window: GPU init failed: {e}");
                el.exit();
                return;
            }
        }
        self.window = Some(window);
        self.last = Instant::now();
    }

    fn window_event(&mut self, el: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => {
                self.input.close = true;
                el.exit();
            }
            WindowEvent::Resized(size) => {
                if let Some(g) = self.gfx.as_mut() {
                    g.resize(size.width, size.height);
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                self.input.mouse = (position.x as f32, position.y as f32);
            }
            WindowEvent::MouseInput { state, button: MouseButton::Left, .. } => {
                self.input.mouse_down = state == ElementState::Pressed;
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if let PhysicalKey::Code(code) = event.physical_key {
                    if let Some(key) = map_key(code) {
                        self.input.set_key(key, event.state == ElementState::Pressed);
                    }
                    if code == KeyCode::Escape {
                        el.exit();
                    }
                }
            }
            WindowEvent::RedrawRequested => {
                let now = Instant::now();
                let dt = (now - self.last).as_secs_f32();
                self.last = now;

                (self.frame)(&self.input, dt, &mut self.fb);
                self.input.end_frame();

                if let Some(g) = self.gfx.as_mut() {
                    g.present(&self.fb);
                }
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _el: &ActiveEventLoop) {
        // Drive a continuous animation loop.
        if let Some(w) = &self.window {
            w.request_redraw();
        }
    }
}

fn map_key(code: KeyCode) -> Option<Key> {
    Some(match code {
        KeyCode::KeyW => Key::W,
        KeyCode::KeyA => Key::A,
        KeyCode::KeyS => Key::S,
        KeyCode::KeyD => Key::D,
        KeyCode::KeyQ => Key::Q,
        KeyCode::KeyE => Key::E,
        KeyCode::KeyR => Key::R,
        KeyCode::KeyF => Key::F,
        KeyCode::ArrowUp => Key::Up,
        KeyCode::ArrowDown => Key::Down,
        KeyCode::ArrowLeft => Key::Left,
        KeyCode::ArrowRight => Key::Right,
        KeyCode::Space => Key::Space,
        KeyCode::Enter => Key::Enter,
        KeyCode::Escape => Key::Escape,
        _ => return None,
    })
}

/// wgpu surface + a blit pipeline that presents the CPU framebuffer texture,
/// and (for 3D programs) a [`Scene`](aurora_render3d::Scene) rendered straight
/// to the surface with a depth buffer.
pub(crate) struct Gfx {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    pipeline: wgpu::RenderPipeline,
    texture: wgpu::Texture,
    bind_group: wgpu::BindGroup,
    tex_w: u32,
    tex_h: u32,
    /// Lazily-created 3D scene (only for programs that use the 3D builtins).
    pub(crate) scene: Option<aurora_render3d::Scene>,
    /// HUD overlay: blits the CPU framebuffer over the 3D scene, treating pure
    /// black as transparent (a color key).
    hud_pipeline: wgpu::RenderPipeline,
    hud_bind_group: wgpu::BindGroup,
    /// Procedural speed/wind lines overlay (a uniform-driven fullscreen pass).
    sl_pipeline: wgpu::RenderPipeline,
    sl_bind_group: wgpu::BindGroup,
    sl_buf: wgpu::Buffer,
    /// Damage feedback overlay (low-health vignette + directional hit glow).
    dmg_pipeline: wgpu::RenderPipeline,
    dmg_bind_group: wgpu::BindGroup,
    dmg_buf: wgpu::Buffer,
    /// Offscreen colour target the 3D scene renders into, so the blur pass has a
    /// texture to sample. Then a fullscreen blur/blit pass copies it to the surface.
    post_tex: wgpu::Texture,
    post_view: wgpu::TextureView,
    post_w: u32,
    post_h: u32,
    blur_pipeline: wgpu::RenderPipeline,
    blur_bind_group: wgpu::BindGroup,
    blur_sampler: wgpu::Sampler,
    blur_buf: wgpu::Buffer,
}

const HUD_WGSL: &str = r#"
struct VOut { @builtin(position) pos: vec4<f32>, @location(0) uv: vec2<f32> };
@vertex
fn vs(@builtin(vertex_index) i: u32) -> VOut {
    var p = array<vec2<f32>, 3>(vec2(-1.0, -1.0), vec2(3.0, -1.0), vec2(-1.0, 3.0));
    var o: VOut;
    let xy = p[i];
    o.pos = vec4<f32>(xy, 0.0, 1.0);
    o.uv = vec2<f32>(xy.x * 0.5 + 0.5, 1.0 - (xy.y * 0.5 + 0.5));
    return o;
}
@group(0) @binding(0) var tex: texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;
@fragment
fn fs(in: VOut) -> @location(0) vec4<f32> {
    let c = textureSample(tex, samp, in.uv);
    // Pure black is the transparent key; everything else is HUD.
    if (c.r + c.g + c.b < 0.012) { discard; }
    return vec4<f32>(c.rgb, 1.0);
}
"#;

// Procedural radial speed/wind lines: soft, diffuse white streaks anchored at the
// screen edges, fading to a clear centre. Driven by a wind intensity + time uniform.
const SPEEDLINES_WGSL: &str = r#"
struct SL { wind: f32, time: f32, aspect: f32, pad: f32 };
@group(0) @binding(0) var<uniform> u: SL;
struct VOut { @builtin(position) pos: vec4<f32>, @location(0) uv: vec2<f32> };
@vertex
fn vs(@builtin(vertex_index) i: u32) -> VOut {
    var p = array<vec2<f32>, 3>(vec2(-1.0, -1.0), vec2(3.0, -1.0), vec2(-1.0, 3.0));
    var o: VOut;
    let xy = p[i];
    o.pos = vec4<f32>(xy, 0.0, 1.0);
    o.uv = vec2<f32>(xy.x * 0.5 + 0.5, 1.0 - (xy.y * 0.5 + 0.5));
    return o;
}
// Periodic value noise over the circle (so streaks are IRREGULAR, no seam at +-pi).
fn hashp(n: f32, period: f32) -> f32 {
    let m = n - floor(n / period) * period;
    return fract(sin(m * 17.23) * 43758.5453);
}
fn vnoisep(x: f32, period: f32) -> f32 {
    let i = floor(x);
    let f = fract(x);
    let w = f * f * (3.0 - 2.0 * f);
    return mix(hashp(i, period), hashp(i + 1.0, period), w);
}
@fragment
fn fs(in: VOut) -> @location(0) vec4<f32> {
    var p = in.uv * 2.0 - 1.0;
    p.x = p.x * u.aspect;
    let r = length(p);
    let a = atan2(p.y, p.x);
    let ta = a / 6.2831853 + 0.5;          // 0..1 around the circle
    // Irregular multi-scale angular streaks (value noise -> random widths/spacing).
    var s = vnoisep(ta * 23.0, 23.0) * 0.6
          + vnoisep(ta * 57.0, 57.0) * 0.3
          + vnoisep(ta * 131.0, 131.0) * 0.1;
    s = pow(clamp(s, 0.0, 1.0), 9.5);      // high contrast -> crisp, defined streaks
    s = s * (0.7 + 0.3 * sin(u.time * 2.0 + ta * 60.0));   // subtle shimmer
    // Clear centre, ramping to the edges; wind pushes the inner edge inward (longer).
    // Per-angle noise varies how far IN each streak reaches, so lengths are irregular.
    let lenvar = vnoisep(ta * 19.0, 19.0);
    let inner = 1.0 - 0.6 * u.wind * (0.4 + 0.6 * lenvar);
    let radial = smoothstep(inner, 1.25, r);
    // Per-angle brightness so some streaks are much stronger than others (irregular).
    let intvar = 0.35 + 1.0 * vnoisep(ta * 13.0, 13.0);
    let alpha = clamp(s * radial * intvar * u.wind * 5.2, 0.0, 1.0);
    return vec4<f32>(1.0, 1.0, 1.0, alpha);
}
"#;

// Damage feedback: a red low-health vignette (edges, scaled by `vig`) plus a
// directional red glow at the edge in the hit direction (`dir`, scaled by `hit`).
const DAMAGE_WGSL: &str = r#"
struct DMG { vig: f32, hit: f32, dirx: f32, diry: f32, aspect: f32, oc: f32, p1: f32, p2: f32 };
@group(0) @binding(0) var<uniform> u: DMG;
struct VOut { @builtin(position) pos: vec4<f32>, @location(0) uv: vec2<f32> };
@vertex
fn vs(@builtin(vertex_index) i: u32) -> VOut {
    var p = array<vec2<f32>, 3>(vec2(-1.0, -1.0), vec2(3.0, -1.0), vec2(-1.0, 3.0));
    var o: VOut;
    let xy = p[i];
    o.pos = vec4<f32>(xy, 0.0, 1.0);
    o.uv = vec2<f32>(xy.x * 0.5 + 0.5, 1.0 - (xy.y * 0.5 + 0.5));
    return o;
}
@fragment
fn fs(in: VOut) -> @location(0) vec4<f32> {
    let p = in.uv * 2.0 - 1.0;
    let pa = vec2<f32>(p.x * u.aspect, p.y);
    let r = length(pa);
    // Low-health vignette: red glow at the edges, stronger as vig rises.
    let vig = smoothstep(0.35, 1.15, r) * u.vig;
    // Directional hit glow: a red cone at the edge pointing at the attacker.
    var dirg = 0.0;
    if (u.hit > 0.001 && length(p) > 0.001) {
        let pd = p / length(p);
        let d = dot(pd, vec2<f32>(u.dirx, u.diry));
        dirg = pow(max(d, 0.0), 3.0) * smoothstep(0.15, 1.1, r) * u.hit;
    }
    let dmg_a = clamp(vig + dirg, 0.0, 1.0);
    // Overclock: gently DARKEN the surroundings (a bit more at the edges) so glowing
    // enemies pop - a Reyna/Empress-style highlight rather than a full screen tint.
    let oc_a = clamp(u.oc * (0.22 + 0.12 * smoothstep(0.2, 1.25, r)), 0.0, 1.0);
    let tot = dmg_a + oc_a;
    if (tot < 0.001) { return vec4<f32>(0.0, 0.0, 0.0, 0.0); }
    // red damage tint + near-black overclock dim, blended by weight.
    let col = (vec3<f32>(0.92, 0.06, 0.06) * dmg_a + vec3<f32>(0.02, 0.02, 0.05) * oc_a) / tot;
    return vec4<f32>(col, clamp(tot, 0.0, 1.0));
}
"#;

const BLIT_WGSL: &str = r#"
struct VOut { @builtin(position) pos: vec4<f32>, @location(0) uv: vec2<f32> };
@vertex
fn vs(@builtin(vertex_index) i: u32) -> VOut {
    var p = array<vec2<f32>, 3>(vec2(-1.0, -1.0), vec2(3.0, -1.0), vec2(-1.0, 3.0));
    var o: VOut;
    let xy = p[i];
    o.pos = vec4<f32>(xy, 0.0, 1.0);
    o.uv = vec2<f32>(xy.x * 0.5 + 0.5, 1.0 - (xy.y * 0.5 + 0.5));
    return o;
}
@group(0) @binding(0) var tex: texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;
@fragment
fn fs(in: VOut) -> @location(0) vec4<f32> {
    return textureSample(tex, samp, in.uv);
}
"#;

// Fullscreen blur of the rendered scene. Samples a 24-tap golden-angle spiral disc
// (linear filtering) at a uniform-driven pixel radius. At radius 0 it is a plain copy,
// so the scene looks identical when the blur is off. Used for the paused/menu backdrop.
const BLUR_WGSL: &str = r#"
struct BL { radius: f32, texx: f32, texy: f32, pad: f32 };
@group(0) @binding(0) var tex: texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;
@group(0) @binding(2) var<uniform> u: BL;
struct VOut { @builtin(position) pos: vec4<f32>, @location(0) uv: vec2<f32> };
@vertex
fn vs(@builtin(vertex_index) i: u32) -> VOut {
    var p = array<vec2<f32>, 3>(vec2(-1.0, -1.0), vec2(3.0, -1.0), vec2(-1.0, 3.0));
    var o: VOut;
    let xy = p[i];
    o.pos = vec4<f32>(xy, 0.0, 1.0);
    o.uv = vec2<f32>(xy.x * 0.5 + 0.5, 1.0 - (xy.y * 0.5 + 0.5));
    return o;
}
@fragment
fn fs(in: VOut) -> @location(0) vec4<f32> {
    if (u.radius < 0.25) { return textureSampleLevel(tex, samp, in.uv, 0.0); }
    var acc = vec3<f32>(0.0);
    let texel = vec2<f32>(u.texx, u.texy);
    for (var k: i32 = 0; k < 24; k = k + 1) {
        let fk = f32(k);
        let ang = fk * 2.39996323;                       // golden angle -> even spread
        let rad = sqrt((fk + 0.5) / 24.0) * u.radius;    // even disc coverage
        let off = vec2<f32>(cos(ang), sin(ang)) * rad * texel;
        acc = acc + textureSampleLevel(tex, samp, in.uv + off, 0.0).rgb;
    }
    return vec4<f32>(acc / 24.0, 1.0);
}
"#;

impl Gfx {
    pub(crate) fn new(window: Arc<Window>, tex_w: u32, tex_h: u32) -> Result<Gfx, String> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
        let surface = instance
            .create_surface(window.clone())
            .map_err(|e| format!("create surface: {e}"))?;
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        }))
        .ok_or("no GPU adapter")?;
        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("aurora-window"),
                required_features: wgpu::Features::empty(),
                // Use the adapter's full limits (desktop GPUs) so the 3D renderer
                // gets real performance and large scenes, not the conservative
                // downlevel fallback.
                required_limits: adapter.limits(),
                memory_hints: wgpu::MemoryHints::Performance,
            },
            None,
        ))
        .map_err(|e| format!("request device: {e}"))?;

        let size = window.inner_size();
        let caps = surface.get_capabilities(&adapter);
        let format = caps
            .formats
            .iter()
            .copied()
            .find(|f| !f.is_srgb())
            .unwrap_or(caps.formats[0]);
        // Present mode: DEFAULT to plain Fifo (the long-standing baseline - vsync-capped, GPU idles
        // between frames, coolest/most stable on a throttled laptop). Override at runtime with
        // AURORA_PRESENT=mailbox|relaxed|immediate|fifo to A/B which feels best on this machine
        // (Mailbox = uncapped/lowest-latency but maxes the GPU; relaxed = adaptive vsync, no hard
        // cliff; immediate = uncapped, may tear). Falls back to Fifo if the pick is unsupported.
        let present_mode = {
            let want = std::env::var("AURORA_PRESENT").unwrap_or_default().to_lowercase();
            let pick = match want.as_str() {
                "mailbox" => wgpu::PresentMode::Mailbox,
                "relaxed" | "fiforelaxed" | "adaptive" => wgpu::PresentMode::FifoRelaxed,
                "immediate" | "nosync" => wgpu::PresentMode::Immediate,
                _ => wgpu::PresentMode::Fifo,
            };
            if caps.present_modes.contains(&pick) { pick } else { wgpu::PresentMode::Fifo }
        };
        eprintln!("[aurora] present mode: {:?}", present_mode);
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode,
            desired_maximum_frame_latency: 2,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
        };
        surface.configure(&device, &config);

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("framebuffer"),
            size: wgpu::Extent3d { width: tex_w, height: tex_h, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("blit"),
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("blit"),
            source: wgpu::ShaderSource::Wgsl(BLIT_WGSL.into()),
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("blit"),
            layout: None,
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: "vs",
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: "fs",
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("blit"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&sampler) },
            ],
        });

        // HUD overlay pipeline (alpha-blended, black = transparent key).
        let hud_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("hud"),
            source: wgpu::ShaderSource::Wgsl(HUD_WGSL.into()),
        });
        let hud_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("hud"),
            layout: None,
            vertex: wgpu::VertexState {
                module: &hud_module,
                entry_point: "vs",
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &hud_module,
                entry_point: "fs",
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });
        // The HUD overlay is a low-res framebuffer stretched over the full surface,
        // so sample it LINEARLY (the 2D retro `present` path keeps the Nearest
        // `sampler` for crisp pixel art). Linear smooths the upscale so the HUD
        // reads clean instead of chunky/blocky.
        let hud_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("hud-linear"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let hud_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("hud"),
            layout: &hud_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&hud_sampler) },
            ],
        });

        // Speed/wind lines: a uniform-driven procedural fullscreen pass (alpha-blended).
        let sl_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("speedlines"),
            source: wgpu::ShaderSource::Wgsl(SPEEDLINES_WGSL.into()),
        });
        let sl_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("speedlines-uniform"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let sl_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("speedlines"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let sl_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("speedlines"),
            bind_group_layouts: &[&sl_layout],
            push_constant_ranges: &[],
        });
        let sl_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("speedlines"),
            layout: Some(&sl_pl),
            vertex: wgpu::VertexState {
                module: &sl_module,
                entry_point: "vs",
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &sl_module,
                entry_point: "fs",
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });
        let sl_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("speedlines"),
            layout: &sl_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: sl_buf.as_entire_binding(),
            }],
        });

        // Damage feedback overlay (same uniform-driven fullscreen pattern).
        let dmg_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("damage"),
            source: wgpu::ShaderSource::Wgsl(DAMAGE_WGSL.into()),
        });
        let dmg_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("damage-uniform"),
            size: 32,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let dmg_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("damage"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let dmg_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("damage"),
            bind_group_layouts: &[&dmg_layout],
            push_constant_ranges: &[],
        });
        let dmg_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("damage"),
            layout: Some(&dmg_pl),
            vertex: wgpu::VertexState {
                module: &dmg_module,
                entry_point: "vs",
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &dmg_module,
                entry_point: "fs",
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });
        let dmg_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("damage"),
            layout: &dmg_layout,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: dmg_buf.as_entire_binding() }],
        });

        // Offscreen scene target + fullscreen blur/blit pass (the paused/menu backdrop).
        let (post_w, post_h) = (config.width.max(1), config.height.max(1));
        let post_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("post-scene"),
            size: wgpu::Extent3d { width: post_w, height: post_h, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let post_view = post_tex.create_view(&wgpu::TextureViewDescriptor::default());
        let blur_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("blur"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let blur_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("blur"),
            source: wgpu::ShaderSource::Wgsl(BLUR_WGSL.into()),
        });
        let blur_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("blur-uniform"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let blur_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("blur"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let blur_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("blur"),
            bind_group_layouts: &[&blur_layout],
            push_constant_ranges: &[],
        });
        let blur_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("blur"),
            layout: Some(&blur_pl),
            vertex: wgpu::VertexState {
                module: &blur_module,
                entry_point: "vs",
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &blur_module,
                entry_point: "fs",
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });
        let blur_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("blur"),
            layout: &blur_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&post_view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&blur_sampler) },
                wgpu::BindGroupEntry { binding: 2, resource: blur_buf.as_entire_binding() },
            ],
        });

        Ok(Gfx {
            surface,
            device,
            queue,
            config,
            pipeline,
            texture,
            bind_group,
            tex_w,
            tex_h,
            scene: None,
            hud_pipeline,
            hud_bind_group,
            sl_pipeline,
            sl_bind_group,
            sl_buf,
            dmg_pipeline,
            dmg_bind_group,
            dmg_buf,
            post_tex,
            post_view,
            post_w,
            post_h,
            blur_pipeline,
            blur_bind_group,
            blur_sampler,
            blur_buf,
        })
    }

    /// Create the 3D scene on first use, sized to the current surface, and return
    /// it along with the device/queue (disjoint borrows).
    pub(crate) fn scene_mut(
        &mut self,
    ) -> (&wgpu::Device, &wgpu::Queue, &mut aurora_render3d::Scene) {
        if self.scene.is_none() {
            self.scene = Some(aurora_render3d::Scene::new(
                &self.device,
                &self.queue,
                self.config.format,
                self.config.width.max(1),
                self.config.height.max(1),
                4, // 4x MSAA for the live window
            ));
        }
        (&self.device, &self.queue, self.scene.as_mut().unwrap())
    }

    /// Render the 3D scene to the surface, then overlay the HUD framebuffer
    /// (`hud_rgba`, the CPU framebuffer; pure-black pixels are transparent).
    /// Recreate the HUD/blit texture (and its bind groups) at a new size, so the
    /// HUD framebuffer can track the window size dynamically.
    fn resize_hud_texture(&mut self, w: u32, h: u32) {
        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("framebuffer"),
            size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let nearest = self.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("blit"),
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });
        let linear = self.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("hud-linear"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        self.bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("blit"),
            layout: &self.pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&nearest) },
            ],
        });
        self.hud_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("hud"),
            layout: &self.hud_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&linear) },
            ],
        });
        self.texture = texture;
        self.tex_w = w;
        self.tex_h = h;
    }

    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::too_many_arguments)]
    fn present_scene(
        &mut self, hud_rgba: &[u8], hud_w: u32, hud_h: u32, sl_wind: f32, sl_time: f32,
        dmg_vig: f32, dmg_hit: f32, dmg_dx: f32, dmg_dy: f32, dmg_oc: f32, blur: f32,
    ) {
        let (w, h) = (self.config.width.max(1), self.config.height.max(1));
        // Keep the offscreen scene target sized to the surface, recreating its bind group
        // so the blur pass samples a matching texture.
        if w != self.post_w || h != self.post_h {
            self.post_tex = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("post-scene"),
                size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: self.config.format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            self.post_view = self.post_tex.create_view(&wgpu::TextureViewDescriptor::default());
            self.blur_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("blur"),
                layout: &self.blur_pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&self.post_view) },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.blur_sampler) },
                    wgpu::BindGroupEntry { binding: 2, resource: self.blur_buf.as_entire_binding() },
                ],
            });
            self.post_w = w;
            self.post_h = h;
        }
        // Blur uniform: radius in pixels + texel size.
        let mut blu = [0u8; 16];
        blu[0..4].copy_from_slice(&blur.to_ne_bytes());
        blu[4..8].copy_from_slice(&(1.0f32 / w as f32).to_ne_bytes());
        blu[8..12].copy_from_slice(&(1.0f32 / h as f32).to_ne_bytes());
        self.queue.write_buffer(&self.blur_buf, 0, &blu);
        // Update the speed-lines uniform (wind, time, aspect).
        let aspect = w as f32 / h.max(1) as f32;
        let mut slu = [0u8; 16];
        slu[0..4].copy_from_slice(&sl_wind.to_ne_bytes());
        slu[4..8].copy_from_slice(&sl_time.to_ne_bytes());
        slu[8..12].copy_from_slice(&aspect.to_ne_bytes());
        self.queue.write_buffer(&self.sl_buf, 0, &slu);
        // Update the damage uniform (vig, hit, dir, aspect).
        let mut dmu = [0u8; 32];
        dmu[0..4].copy_from_slice(&dmg_vig.to_ne_bytes());
        dmu[4..8].copy_from_slice(&dmg_hit.to_ne_bytes());
        dmu[8..12].copy_from_slice(&dmg_dx.to_ne_bytes());
        dmu[12..16].copy_from_slice(&dmg_dy.to_ne_bytes());
        dmu[16..20].copy_from_slice(&aspect.to_ne_bytes());
        dmu[20..24].copy_from_slice(&dmg_oc.to_ne_bytes());
        self.queue.write_buffer(&self.dmg_buf, 0, &dmu);
        if let Some(scene) = self.scene.as_mut() {
            scene.resize(&self.device, w, h);
        } else {
            return;
        }
        // Resize the HUD texture to match the framebuffer the game gave us, so a
        // game can size its HUD to the live window and have it blit 1:1 (crisp).
        if hud_w > 0 && hud_h > 0 && (hud_w != self.tex_w || hud_h != self.tex_h) {
            self.resize_hud_texture(hud_w, hud_h);
        }
        // Upload the HUD framebuffer for the overlay pass.
        let hud_bytes = (self.tex_w * self.tex_h * 4) as usize;
        let has_hud = hud_rgba.len() >= hud_bytes && hud_bytes > 0;
        if has_hud {
            self.queue.write_texture(
                wgpu::ImageCopyTexture {
                    texture: &self.texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                &hud_rgba[..hud_bytes],
                wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(self.tex_w * 4),
                    rows_per_image: Some(self.tex_h),
                },
                wgpu::Extent3d { width: self.tex_w, height: self.tex_h, depth_or_array_layers: 1 },
            );
        }
        let surface_tex = match self.surface.get_current_texture() {
            Ok(t) => t,
            Err(e) => {
                // Lost/Outdated/Timeout are transient (e.g. during a resize):
                // reconfigure and skip the frame quietly. Surface out of memory
                // is serious and would otherwise be a silent black screen.
                if matches!(e, wgpu::SurfaceError::OutOfMemory) {
                    eprintln!("aurora-window: surface error (out of memory)");
                }
                self.surface.configure(&self.device, &self.config);
                return;
            }
        };
        let view = surface_tex.texture.create_view(&wgpu::TextureViewDescriptor::default());
        let mut enc =
            self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        // Render the 3D scene into the offscreen target, then a fullscreen pass copies it
        // to the surface - blurred when `blur` > 0 (paused/menu), an exact copy otherwise.
        if let Some(scene) = self.scene.as_mut() {
            scene.render(&self.device, &self.queue, &mut enc, &self.post_view);
        }
        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("blur"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_pipeline(&self.blur_pipeline);
            pass.set_bind_group(0, &self.blur_bind_group, &[]);
            pass.draw(0..3, 0..1);
        }
        // Speed/wind lines pass (over the 3D, under the HUD) when wind is active.
        if sl_wind > 0.001 {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("speedlines"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_pipeline(&self.sl_pipeline);
            pass.set_bind_group(0, &self.sl_bind_group, &[]);
            pass.draw(0..3, 0..1);
        }
        // Damage feedback pass (red vignette + hit glow + gold overclock) when active.
        if dmg_vig > 0.001 || dmg_hit > 0.001 || dmg_oc > 0.001 {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("damage"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_pipeline(&self.dmg_pipeline);
            pass.set_bind_group(0, &self.dmg_bind_group, &[]);
            pass.draw(0..3, 0..1);
        }
        // HUD overlay pass (load the 3D result, blend the HUD on top).
        if has_hud {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("hud"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_pipeline(&self.hud_pipeline);
            pass.set_bind_group(0, &self.hud_bind_group, &[]);
            pass.draw(0..3, 0..1);
        }
        self.queue.submit(Some(enc.finish()));
        surface_tex.present();
    }

    pub(crate) fn resize(&mut self, w: u32, h: u32) {
        if w > 0 && h > 0 {
            self.config.width = w;
            self.config.height = h;
            self.surface.configure(&self.device, &self.config);
        }
    }

    fn present(&mut self, fb: &Framebuffer) {
        self.present_rgba(&fb.rgba());
    }

    /// Present tightly-packed RGBA8 bytes (`tex_w * tex_h * 4`).
    pub(crate) fn present_rgba(&mut self, rgba: &[u8]) {
        // Upload the pixels into the GPU texture.
        self.queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &self.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            rgba,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(self.tex_w * 4),
                rows_per_image: Some(self.tex_h),
            },
            wgpu::Extent3d { width: self.tex_w, height: self.tex_h, depth_or_array_layers: 1 },
        );

        let surface_tex = match self.surface.get_current_texture() {
            Ok(t) => t,
            Err(e) => {
                // Lost/Outdated/Timeout are transient (e.g. during a resize):
                // reconfigure and skip the frame quietly. Surface out of memory
                // is serious and would otherwise be a silent black screen.
                if matches!(e, wgpu::SurfaceError::OutOfMemory) {
                    eprintln!("aurora-window: surface error (out of memory)");
                }
                self.surface.configure(&self.device, &self.config);
                return;
            }
        };
        let view = surface_tex.texture.create_view(&wgpu::TextureViewDescriptor::default());
        let mut enc =
            self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.draw(0..3, 0..1);
        }
        self.queue.submit(Some(enc.finish()));
        surface_tex.present();
    }
}
