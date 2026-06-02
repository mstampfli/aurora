//! A persistent live renderer: runs a user fragment shader on the GPU every
//! frame into an offscreen texture and reads the pixels back, reusing the device
//! and pipeline across frames (recompiling only when the shader source changes).
//! A `time` uniform lets shaders animate. This is what backs Aurora's
//! `gpu_render` builtin, so a game loop can do GPU-accelerated rendering.

use std::sync::{Mutex, OnceLock};

use crate::Gpu;

/// Header prepended to the user's fragment source: a fullscreen-triangle vertex
/// stage that passes `uv`, plus a `u` uniform with `time` and `res`. User
/// shaders define `@fragment fn fs_main(@location(0) uv: vec2<f32>) ->
/// @location(0) vec4<f32>` and may read `u.time` / `u.res`.
const HEADER: &str = r#"
struct Uniforms { time: f32, _pad: f32, res: vec2<f32> };
@group(0) @binding(0) var<uniform> u: Uniforms;
struct VOut { @builtin(position) pos: vec4<f32>, @location(0) uv: vec2<f32> };
@vertex
fn vs_main(@builtin(vertex_index) i: u32) -> VOut {
    var p = array<vec2<f32>, 3>(vec2(-1.0, -1.0), vec2(3.0, -1.0), vec2(-1.0, 3.0));
    var o: VOut;
    let xy = p[i];
    o.pos = vec4<f32>(xy, 0.0, 1.0);
    o.uv = vec2<f32>(xy.x * 0.5 + 0.5, 1.0 - (xy.y * 0.5 + 0.5));
    return o;
}
"#;

struct Live {
    gpu: Gpu,
    w: u32,
    h: u32,
    shader: String,
    pipeline: wgpu::RenderPipeline,
    uniform: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    texture: wgpu::Texture,
    readback: wgpu::Buffer,
    padded_row: u32,
}

// Persistent renderer state. A `static` (not thread-local) so its wgpu objects
// are leaked at process exit rather than dropped — dropping them during TLS
// teardown trips wgpu's internal thread-locals ("access during destruction").
static LIVE: OnceLock<Mutex<Option<Live>>> = OnceLock::new();

fn live_cell() -> &'static Mutex<Option<Live>> {
    LIVE.get_or_init(|| Mutex::new(None))
}

/// Render `fragment_src` (a fragment shader body referencing `uv` and `u.time`)
/// at `time` seconds, returning tightly-packed RGBA8 (`w*h*4`). Returns an empty
/// vec if no GPU is available. Reuses state across calls; recompiles only when
/// the shader source or size changes.
pub fn render_shader(fragment_src: &str, w: u32, h: u32, time: f32) -> Vec<u8> {
    let mut slot = live_cell().lock().unwrap();
    // (Re)build if first call, size changed, or shader changed.
    let needs_build = match slot.as_ref() {
        Some(l) => l.w != w || l.h != h || l.shader != fragment_src,
        None => true,
    };
    if needs_build {
        match Live::new(fragment_src, w, h) {
            Some(l) => *slot = Some(l),
            None => return Vec::new(),
        }
    }
    slot.as_ref().map(|l| l.render(time)).unwrap_or_default()
}

impl Live {
    fn new(fragment_src: &str, w: u32, h: u32) -> Option<Live> {
        let gpu = Gpu::new()?;
        let device = gpu.device();
        let fmt = wgpu::TextureFormat::Rgba8Unorm;
        let wgsl = format!("{HEADER}\n{fragment_src}");
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("live"),
            source: wgpu::ShaderSource::Wgsl(wgsl.into()),
        });
        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("u"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
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
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("live"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: "vs_main",
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: "fs_main",
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: fmt,
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
            label: None,
            layout: &bgl,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: uniform.as_entire_binding() }],
        });
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("live-target"),
            size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: fmt,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let padded_row = (w * 4).div_ceil(align) * align;
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("live-readback"),
            size: (padded_row * h) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Some(Live {
            gpu,
            w,
            h,
            shader: fragment_src.to_string(),
            pipeline,
            uniform,
            bind_group,
            texture,
            readback,
            padded_row,
        })
    }

    fn render(&self, time: f32) -> Vec<u8> {
        let device = self.gpu.device();
        let queue = self.gpu.queue();
        // time, _pad, res.x, res.y
        let u: [f32; 4] = [time, 0.0, self.w as f32, self.h as f32];
        queue.write_buffer(&self.uniform, 0, bytemuck(&u));

        let view = self.texture.create_view(&wgpu::TextureViewDescriptor::default());
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
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
        enc.copy_texture_to_buffer(
            wgpu::ImageCopyTexture {
                texture: &self.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::ImageCopyBuffer {
                buffer: &self.readback,
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(self.padded_row),
                    rows_per_image: Some(self.h),
                },
            },
            wgpu::Extent3d { width: self.w, height: self.h, depth_or_array_layers: 1 },
        );
        queue.submit(Some(enc.finish()));

        let slice = self.readback.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        device.poll(wgpu::Maintain::Wait);
        if rx.recv().is_err() {
            return Vec::new();
        }
        let data = slice.get_mapped_range();
        let unpadded = (self.w * 4) as usize;
        let mut out = Vec::with_capacity(unpadded * self.h as usize);
        for row in 0..self.h {
            let start = (row * self.padded_row) as usize;
            out.extend_from_slice(&data[start..start + unpadded]);
        }
        drop(data);
        self.readback.unmap();
        out
    }
}

fn bytemuck(u: &[f32; 4]) -> &[u8] {
    // SAFETY: f32 array → bytes, exact length, lifetime tied to input.
    unsafe { std::slice::from_raw_parts(u.as_ptr() as *const u8, 16) }
}
