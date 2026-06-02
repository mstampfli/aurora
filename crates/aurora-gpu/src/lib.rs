//! Live GPU execution for Aurora via [`wgpu`].
//!
//! Aurora's `@vertex`/`@fragment` functions lower to WGSL (`aurora-shader`); this
//! crate is where that WGSL — and compute WGSL — actually runs on the GPU. It is
//! **headless**: it renders to offscreen textures and reads pixels/buffers back,
//! so GPU execution is verifiable with no window or surface.
//!
//! A real adapter is required at runtime. Acquisition is fallible (no GPU / no
//! driver), so the entry points return `Result` and tests skip gracefully when
//! [`Gpu::new`] returns `None`.

use std::borrow::Cow;

mod live;
pub use live::render_shader;

use std::sync::OnceLock;

// A leaked, cached GPU for compute dispatches (avoids re-initializing the device
// each call, and avoids TLS-teardown issues — see `live`).
static COMPUTE_GPU: OnceLock<Option<Gpu>> = OnceLock::new();

/// Run a compute shader over `data` (in/out, f32) on the GPU, returning the
/// result. If no GPU is available, returns the input unchanged. The `wgsl` must
/// define `@compute @workgroup_size(64) fn main(...)` over a `read_write`
/// `array<f32>` at `@group(0) @binding(0)`.
pub fn compute(wgsl: &str, data: &[f32]) -> Vec<f32> {
    match COMPUTE_GPU.get_or_init(Gpu::new) {
        Some(g) => g.compute_f32(wgsl, data).unwrap_or_else(|_| data.to_vec()),
        None => data.to_vec(),
    }
}

/// An initialized headless GPU context (device + queue).
pub struct Gpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
    adapter_name: String,
}

impl Gpu {
    /// Acquire a GPU. Returns `None` if no adapter/device is available.
    pub fn new() -> Option<Gpu> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))?;
        let adapter_name = adapter.get_info().name;
        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("aurora-gpu"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::downlevel_defaults(),
                memory_hints: wgpu::MemoryHints::Performance,
            },
            None,
        ))
        .ok()?;
        Some(Gpu { device, queue, adapter_name })
    }

    /// The underlying adapter's reported name (e.g. "NVIDIA GeForce ...").
    pub fn adapter_name(&self) -> &str {
        &self.adapter_name
    }

    pub(crate) fn device(&self) -> &wgpu::Device {
        &self.device
    }

    pub(crate) fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    /// Run a compute shader over `data` in place on the GPU and read it back.
    ///
    /// `wgsl` must define `@compute @workgroup_size(64) fn main(...)` operating on
    /// a single `var<storage, read_write> data: array<f32>` at `@group(0)
    /// @binding(0)`. Dispatches `ceil(len / 64)` workgroups.
    pub fn compute_f32(&self, wgsl: &str, data: &[f32]) -> Result<Vec<f32>, String> {
        use wgpu::util::DeviceExt;
        let n = data.len();
        if n == 0 {
            return Ok(Vec::new());
        }
        let bytes = (n * std::mem::size_of::<f32>()) as wgpu::BufferAddress;

        let storage = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("data"),
            contents: bytemuck_cast(data),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        });
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size: bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let module = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("compute"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(wgsl)),
        });
        let pipeline = self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("compute"),
            layout: None,
            module: &module,
            entry_point: "main",
            compilation_options: Default::default(),
            cache: None,
        });
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: storage.as_entire_binding() }],
        });

        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass =
                enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let groups = n.div_ceil(64) as u32;
            pass.dispatch_workgroups(groups, 1, 1);
        }
        enc.copy_buffer_to_buffer(&storage, 0, &staging, 0, bytes);
        self.queue.submit(Some(enc.finish()));

        // Map and read back.
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv()
            .map_err(|e| format!("map channel: {e}"))?
            .map_err(|e| format!("buffer map failed: {e:?}"))?;
        let view = slice.get_mapped_range();
        let out: Vec<f32> =
            view.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        drop(view);
        staging.unmap();
        Ok(out)
    }

    /// Render `wgsl` to an offscreen RGBA8 texture and read the pixels back.
    ///
    /// `wgsl` must define `@vertex fn vs_main(@builtin(vertex_index) i: u32) ->
    /// @builtin(position) vec4<f32>` and `@fragment fn fs_main(...) ->
    /// @location(0) vec4<f32>`. Three vertices are drawn (a fullscreen triangle).
    /// Returns `width * height * 4` bytes, row-major, top-left origin.
    pub fn render_rgba(&self, wgsl: &str, width: u32, height: u32) -> Result<Vec<u8>, String> {
        self.render_rgba_entries(wgsl, width, height, "vs_main", "fs_main")
    }

    /// Like [`render_rgba`](Self::render_rgba) but with explicit vertex/fragment
    /// entry-point names — used to drive `aurora-shader`'s lowered WGSL, whose
    /// entry names come from the Aurora function names.
    pub fn render_rgba_entries(
        &self,
        wgsl: &str,
        width: u32,
        height: u32,
        vs_entry: &str,
        fs_entry: &str,
    ) -> Result<Vec<u8>, String> {
        let fmt = wgpu::TextureFormat::Rgba8Unorm;
        let tex = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("target"),
            size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: fmt,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = tex.create_view(&wgpu::TextureViewDescriptor::default());

        let module = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("render"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(wgsl)),
        });
        let pipeline = self.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("render"),
            layout: None,
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: vs_entry,
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: fs_entry,
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

        // Padded row length (wgpu requires 256-byte row alignment for copies).
        let unpadded = width * 4;
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let padded = unpadded.div_ceil(align) * align;
        let out_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: (padded * height) as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.0, g: 0.0, b: 0.0, a: 1.0 }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_pipeline(&pipeline);
            pass.draw(0..3, 0..1);
        }
        enc.copy_texture_to_buffer(
            wgpu::ImageCopyTexture {
                texture: &tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::ImageCopyBuffer {
                buffer: &out_buf,
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(padded),
                    rows_per_image: Some(height),
                },
            },
            wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        );
        self.queue.submit(Some(enc.finish()));

        let slice = out_buf.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv()
            .map_err(|e| format!("map channel: {e}"))?
            .map_err(|e| format!("buffer map failed: {e:?}"))?;
        let data = slice.get_mapped_range();
        // Strip row padding back to a tight RGBA buffer.
        let mut out = Vec::with_capacity((unpadded * height) as usize);
        for row in 0..height {
            let start = (row * padded) as usize;
            out.extend_from_slice(&data[start..start + unpadded as usize]);
        }
        drop(data);
        out_buf.unmap();
        Ok(out)
    }
}

/// Reinterpret an `&[f32]` as bytes (little-endian; native on all wgpu targets).
fn bytemuck_cast(data: &[f32]) -> &[u8] {
    // SAFETY: f32 has no padding/invalid bit patterns as bytes; lifetime tied to
    // the input slice; length is exact.
    unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, std::mem::size_of_val(data)) }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Serialize GPU tests: creating several devices concurrently can contend on
    // some drivers. Each test takes this lock first.
    static GPU_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    fn gpu_guard() -> std::sync::MutexGuard<'static, ()> {
        GPU_LOCK.lock().unwrap_or_else(|p| p.into_inner())
    }

    const SQUARE: &str = r#"
@group(0) @binding(0) var<storage, read_write> data: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i < arrayLength(&data)) { data[i] = data[i] * data[i]; }
}
"#;

    // A fullscreen triangle that outputs a solid color, so we can verify the
    // render path executed real WGSL on the GPU.
    const SOLID: &str = r#"
@vertex
fn vs_main(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    var p = array<vec2<f32>, 3>(vec2(-1.0, -3.0), vec2(-1.0, 1.0), vec2(3.0, 1.0));
    return vec4<f32>(p[i], 0.0, 1.0);
}
@fragment
fn fs_main() -> @location(0) vec4<f32> {
    return vec4<f32>(0.2, 0.4, 0.8, 1.0);
}
"#;

    #[test]
    fn gpu_compute_squares_on_real_hardware() {
        let _g = gpu_guard();
        let Some(gpu) = Gpu::new() else {
            eprintln!("no GPU adapter available — skipping live GPU compute test");
            return;
        };
        eprintln!("running on GPU adapter: {}", gpu.adapter_name());
        let input: Vec<f32> = (1..=8).map(|i| i as f32).collect();
        let out = gpu.compute_f32(SQUARE, &input).expect("compute failed");
        let want: Vec<f32> = input.iter().map(|x| x * x).collect();
        assert_eq!(out, want, "GPU compute must square each element");
    }

    #[test]
    fn live_shader_animates_with_time() {
        let _g = gpu_guard();
        if Gpu::new().is_none() {
            eprintln!("no GPU adapter — skipping live shader test");
            return;
        }
        // A shader whose red channel follows `u.time`; two different times must
        // produce different pixels (proving the time uniform reaches the GPU).
        let frag = "@fragment fn fs_main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {\n\
            return vec4<f32>(u.time, uv.x, 0.0, 1.0);\n}";
        let a = super::render_shader(frag, 8, 8, 0.0);
        let b = super::render_shader(frag, 8, 8, 1.0);
        assert_eq!(a.len(), 8 * 8 * 4);
        assert_eq!(b.len(), 8 * 8 * 4);
        assert_eq!(a[0], 0, "red follows time=0");
        assert!(b[0] > 200, "red follows time=1 (got {})", b[0]);
    }

    #[test]
    fn gpu_renders_wgsl_to_texture() {
        let _g = gpu_guard();
        let Some(gpu) = Gpu::new() else {
            eprintln!("no GPU adapter available — skipping live GPU render test");
            return;
        };
        let px = gpu.render_rgba(SOLID, 16, 16).expect("render failed");
        assert_eq!(px.len(), 16 * 16 * 4);
        // Every pixel should be the solid fragment color (0.2,0.4,0.8) in u8.
        let (r, g, b) = (px[0], px[1], px[2]);
        assert!((r as i32 - 51).abs() <= 2, "r={r}");
        assert!((g as i32 - 102).abs() <= 2, "g={g}");
        assert!((b as i32 - 204).abs() <= 2, "b={b}");
    }

    /// The full chain: **Aurora shader source → WGSL (`aurora-shader`) → GPU**.
    /// An Aurora `@fragment` is lowered and its WGSL drives the real render
    /// pipeline; the read-back pixels must match the Aurora-authored color.
    #[test]
    fn aurora_fragment_lowers_and_runs_on_gpu() {
        let _g = gpu_guard();
        let Some(gpu) = Gpu::new() else {
            eprintln!("no GPU adapter available — skipping Aurora→GPU test");
            return;
        };
        // Lower an Aurora fragment shader to WGSL.
        let aurora_src = "@fragment fn aurora_solid() -> Color { vec4(0.6, 0.3, 0.1, 1.0) }";
        let (module, diags) = aurora_parser::parse_str(aurora_src);
        assert!(!diags.iter().any(|d| d.is_error()), "Aurora shader failed to parse");
        let fs_wgsl = aurora_shader::lower_module(&module);

        // A fixed fullscreen-triangle vertex stage, plus the lowered fragment.
        let vs = "@vertex fn vs_main(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {\n\
            var p = array<vec2<f32>, 3>(vec2(-1.0, -3.0), vec2(-1.0, 1.0), vec2(3.0, 1.0));\n\
            return vec4<f32>(p[i], 0.0, 1.0);\n}\n";
        let combined = format!("{vs}\n{fs_wgsl}");

        let px = gpu
            .render_rgba_entries(&combined, 8, 8, "vs_main", "aurora_solid")
            .expect("render of Aurora-lowered shader failed");
        let (r, g, b) = (px[0], px[1], px[2]);
        // vec4(0.6, 0.3, 0.1) -> ~ (153, 76, 26) in u8.
        assert!((r as i32 - 153).abs() <= 3, "r={r}");
        assert!((g as i32 - 76).abs() <= 3, "g={g}");
        assert!((b as i32 - 26).abs() <= 3, "b={b}");
    }
}
