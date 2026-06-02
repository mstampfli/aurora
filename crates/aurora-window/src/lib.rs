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
    key_down as imm_key_down, mouse as imm_mouse, open as imm_open, present as imm_present,
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
    let event_loop = EventLoop::new().map_err(|e| format!("event loop: {e}"))?;
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

/// wgpu surface + a blit pipeline that presents the CPU framebuffer texture.
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
}

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
                required_limits: wgpu::Limits::downlevel_defaults(),
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
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: wgpu::PresentMode::Fifo,
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

        Ok(Gfx { surface, device, queue, config, pipeline, texture, bind_group, tex_w, tex_h })
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
            Err(_) => {
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
