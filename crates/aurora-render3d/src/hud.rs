//! The HUD overlay: the CPU framebuffer composited over the 3D scene.
//!
//! **One implementation, deliberately.** This pass exists in two situations -
//! presenting to a live window, and reading a frame back offscreen for a
//! headless capture - and it used to be written twice: a WGSL pass for the
//! window and a hand-rolled pixel loop in `aurora-window::imm` for the capture.
//! Two copies of a blend rule drift, and when they drift the screenshot stops
//! being evidence about the game: it shows a frame composited by a rule the
//! game does not use. So the pass lives here, in the renderer both paths
//! already own, and there is exactly one place that says how a HUD pixel meets
//! the scene behind it.
//!
//! **Place in the graph.** Part of `aurora-render3d`; used by `aurora-window`
//! for both `present` and `r3d_capture`.

/// The overlay's shader. Alpha is real coverage and the pipeline blends
/// source-over, so a HUD can put a translucent plate behind text.
///
/// This was a colour key - pure black discarded - which could express neither a
/// translucent plate nor a black HUD pixel, so dialogue arrived on an opaque
/// slab and black text was invisible.
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
    if (c.a <= 0.0) { discard; }
    return c;
}
"#;

/// A HUD overlay pass, owning the texture the framebuffer is uploaded into and
/// the pipeline that blends it over a target view.
pub struct HudOverlay {
    pipeline: wgpu::RenderPipeline,
    sampler: wgpu::Sampler,
    texture: wgpu::Texture,
    bind_group: wgpu::BindGroup,
    w: u32,
    h: u32,
}

impl HudOverlay {
    /// Build the pass for a target of `format`. The window's surface format and
    /// the offscreen capture's `Rgba8Unorm` differ, so the format is a
    /// parameter rather than a constant - it is the only thing that varies
    /// between the two callers.
    pub fn new(device: &wgpu::Device, format: wgpu::TextureFormat, w: u32, h: u32) -> HudOverlay {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("hud"),
            source: wgpu::ShaderSource::Wgsl(HUD_WGSL.into()),
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("hud"),
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
        // A low-res framebuffer stretched over the full target, so sample it
        // LINEARLY - the 2D retro blit keeps Nearest for crisp pixel art, but a
        // HUD upscaled with Nearest reads chunky.
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("hud-linear"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let (texture, bind_group) = Self::make(
            device,
            &pipeline,
            &sampler,
            crate::tex_dim(w),
            crate::tex_dim(h),
        );
        HudOverlay {
            pipeline,
            sampler,
            texture,
            bind_group,
            w: crate::tex_dim(w),
            h: crate::tex_dim(h),
        }
    }

    fn make(
        device: &wgpu::Device,
        pipeline: &wgpu::RenderPipeline,
        sampler: &wgpu::Sampler,
        w: u32,
        h: u32,
    ) -> (wgpu::Texture, wgpu::BindGroup) {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("hud-framebuffer"),
            size: wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            // Unorm, not UnormSrgb: the framebuffer's bytes are the colours the
            // game asked for and must reach the blend unconverted.
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("hud"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
            ],
        });
        (texture, bind_group)
    }

    /// Upload a tightly-packed RGBA8 framebuffer, resizing to match it.
    ///
    /// Returns false when there is nothing to composite - no framebuffer, or
    /// fewer bytes than `w * h * 4` - so the caller can skip the pass rather
    /// than blend stale pixels.
    pub fn upload(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        rgba: &[u8],
        w: u32,
        h: u32,
    ) -> bool {
        if w == 0 || h == 0 {
            return false;
        }
        if w != self.w || h != self.h {
            let (texture, bind_group) = Self::make(device, &self.pipeline, &self.sampler, w, h);
            self.texture = texture;
            self.bind_group = bind_group;
            self.w = w;
            self.h = h;
        }
        let bytes = (self.w * self.h * 4) as usize;
        if rgba.len() < bytes || bytes == 0 {
            return false;
        }
        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &self.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &rgba[..bytes],
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(self.w * 4),
                rows_per_image: Some(self.h),
            },
            wgpu::Extent3d {
                width: self.w,
                height: self.h,
                depth_or_array_layers: 1,
            },
        );
        true
    }

    /// Blend the uploaded framebuffer over `target`, which must already hold the
    /// rendered scene (the pass LOADs it).
    pub fn composite(&self, enc: &mut wgpu::CommandEncoder, target: &wgpu::TextureView) {
        let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("hud"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
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
}
