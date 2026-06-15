//! The forward 3D renderer: one pipeline drawing indexed, lit, textured,
//! optionally skinned meshes with a depth buffer.

use bytemuck::{Pod, Zeroable};
use glam::{Mat3, Mat4, Vec3};

use crate::mesh::{GpuMesh, MeshData, Vertex};

/// Maximum joints per skinned mesh (fits a 16 KiB uniform: 128 * 64 B = 8 KiB).
pub const MAX_JOINTS: usize = 128;
const OBJ_ALIGN: u64 = 256; // dynamic-uniform offset alignment (>= 256 everywhere)
const JOINT_BYTES: u64 = (MAX_JOINTS * 64) as u64; // 8192, already a multiple of 256

pub const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GlobalsU {
    view_proj: [[f32; 4]; 4],
    cam_pos: [f32; 4],
    light_dir: [f32; 4],
    light_color: [f32; 4], // rgb + ambient in w
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ObjU {
    model: [[f32; 4]; 4],
    normal_mat: [[f32; 4]; 4],
    base_color: [f32; 4],
    params: [f32; 4], // x = has_texture, y = skinned
}

/// A material: a base color and an optional base-color texture.
pub struct Material {
    pub base_color: [f32; 4],
    pub bind_group: wgpu::BindGroup,
    pub has_texture: bool,
}

struct DrawCmd {
    mesh: usize,
    material: usize,
    model: Mat4,
    joints: Option<Vec<Mat4>>,
}

/// A GPU forward renderer that owns its pipeline and resource registries but
/// borrows the wgpu device/queue for each operation, so it can target an
/// offscreen texture (tests) or the window surface, sharing one device.
pub struct Renderer3D {
    pipeline: wgpu::RenderPipeline,
    globals_buf: wgpu::Buffer,
    globals_bg: wgpu::BindGroup,
    obj_layout: wgpu::BindGroupLayout,
    mat_layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,

    // Per-frame dynamic-uniform ring buffers + their bind group.
    obj_buf: wgpu::Buffer,
    joint_buf: wgpu::Buffer,
    obj_bg: wgpu::BindGroup,
    obj_cap: u64,   // capacity in entries
    joint_cap: u64, // capacity in blocks

    depth: wgpu::TextureView,
    depth_size: (u32, u32),

    meshes: Vec<GpuMesh>,
    materials: Vec<Material>,

    globals: GlobalsU,
    queue_cmds: Vec<DrawCmd>,
}

impl Renderer3D {
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        color_format: wgpu::TextureFormat,
        w: u32,
        h: u32,
    ) -> Renderer3D {
        let globals_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("globals"),
            entries: &[uniform_entry(0, wgpu::ShaderStages::VERTEX_FRAGMENT, false, None)],
        });
        let obj_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("object"),
            entries: &[
                uniform_entry(0, wgpu::ShaderStages::VERTEX_FRAGMENT, true, Some(std::mem::size_of::<ObjU>() as u64)),
                uniform_entry(1, wgpu::ShaderStages::VERTEX, true, Some(JOINT_BYTES)),
            ],
        });
        let mat_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("material"),
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
            ],
        });

        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("render3d"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("render3d"),
            bind_group_layouts: &[&globals_layout, &obj_layout, &mat_layout],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("render3d"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: "vs",
                compilation_options: Default::default(),
                buffers: &[Vertex::LAYOUT],
            },
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: "fs",
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: color_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: Some(wgpu::Face::Back),
                front_face: wgpu::FrontFace::Ccw,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        let globals_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("globals"),
            size: std::mem::size_of::<GlobalsU>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let globals_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("globals"),
            layout: &globals_layout,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: globals_buf.as_entire_binding() }],
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("tex"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            ..Default::default()
        });

        let (obj_cap, joint_cap) = (256u64, 16u64);
        let (obj_buf, joint_buf, obj_bg) =
            make_ring(device, &obj_layout, obj_cap, joint_cap);
        let depth = make_depth(device, w.max(1), h.max(1));

        let globals = GlobalsU {
            view_proj: Mat4::IDENTITY.to_cols_array_2d(),
            cam_pos: [0.0, 0.0, 0.0, 1.0],
            light_dir: Vec3::new(0.4, 1.0, 0.3).normalize().extend(0.0).into(),
            light_color: [1.0, 1.0, 1.0, 0.25],
        };

        let mut r = Renderer3D {
            pipeline,
            globals_buf,
            globals_bg,
            obj_layout,
            mat_layout,
            sampler,
            obj_buf,
            joint_buf,
            obj_bg,
            obj_cap,
            joint_cap,
            depth,
            depth_size: (w.max(1), h.max(1)),
            meshes: Vec::new(),
            materials: Vec::new(),
            globals,
            queue_cmds: Vec::new(),
        };
        // Material 0 is a plain white, untextured material: it backs every draw
        // that doesn't name a material, and provides a default texture binding.
        let default_mat = r.make_material(device, queue, [1.0; 4], None, 0, 0);
        r.materials.push(default_mat);
        r
    }

    pub fn resize(&mut self, device: &wgpu::Device, w: u32, h: u32) {
        if (w, h) != self.depth_size && w > 0 && h > 0 {
            self.depth = make_depth(device, w, h);
            self.depth_size = (w, h);
        }
    }

    pub fn set_camera(&mut self, view_proj: Mat4, cam_pos: Vec3) {
        self.globals.view_proj = view_proj.to_cols_array_2d();
        self.globals.cam_pos = cam_pos.extend(1.0).into();
    }

    pub fn set_light(&mut self, dir: Vec3, color: Vec3, ambient: f32) {
        self.globals.light_dir = dir.normalize_or_zero().extend(0.0).into();
        self.globals.light_color = color.extend(ambient).into();
    }

    pub fn add_mesh(&mut self, device: &wgpu::Device, mesh: &MeshData) -> usize {
        self.meshes.push(GpuMesh::upload(device, mesh));
        self.meshes.len() - 1
    }

    /// Register a material. `texture` is optional tightly-packed RGBA8 of size
    /// `tw*th*4`. Returns the material id.
    pub fn add_material(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        base_color: [f32; 4],
        texture: Option<(&[u8], u32, u32)>,
    ) -> usize {
        let mat = match texture {
            Some((rgba, tw, th)) => self.make_material(device, queue, base_color, Some(rgba), tw, th),
            None => self.make_material(device, queue, base_color, None, 0, 0),
        };
        self.materials.push(mat);
        self.materials.len() - 1
    }

    fn make_material(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        base_color: [f32; 4],
        rgba: Option<&[u8]>,
        tw: u32,
        th: u32,
    ) -> Material {
        let (w, h, pixels, has_texture) = match rgba {
            Some(px) if tw > 0 && th > 0 => (tw, th, px.to_vec(), true),
            _ => (1, 1, vec![255u8, 255, 255, 255], false),
        };
        let tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("material-tex"),
            size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &pixels,
            wgpu::ImageDataLayout { offset: 0, bytes_per_row: Some(w * 4), rows_per_image: Some(h) },
            wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        );
        let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("material"),
            layout: &self.mat_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.sampler) },
            ],
        });
        Material { base_color, bind_group, has_texture }
    }

    pub fn material_count(&self) -> usize {
        self.materials.len()
    }
    pub fn mesh_count(&self) -> usize {
        self.meshes.len()
    }

    /// Clear the per-frame draw queue.
    pub fn begin(&mut self) {
        self.queue_cmds.clear();
    }

    /// Enqueue a draw of `mesh` with `material` at `model`. If `joints` is given,
    /// the mesh is skinned by those matrices.
    pub fn draw(&mut self, mesh: usize, material: usize, model: Mat4, joints: Option<Vec<Mat4>>) {
        if mesh < self.meshes.len() {
            let material = if material < self.materials.len() { material } else { 0 };
            self.queue_cmds.push(DrawCmd { mesh, material, model, joints });
        }
    }

    /// Render the queued draws into `color_view` (clearing color + depth).
    pub fn render(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        color_view: &wgpu::TextureView,
        clear: [f32; 4],
    ) {
        queue.write_buffer(&self.globals_buf, 0, bytemuck::bytes_of(&self.globals));

        // Lay out per-object uniforms (each padded to OBJ_ALIGN) and joint blocks.
        let stride = OBJ_ALIGN;
        let n = self.queue_cmds.len() as u64;
        let skinned: u64 = self.queue_cmds.iter().filter(|c| c.joints.is_some()).count() as u64;
        // block 0 is the identity joint block used by every non-skinned draw.
        let joint_blocks = skinned + 1;
        if n > self.obj_cap || joint_blocks > self.joint_cap {
            self.obj_cap = (n.max(1)).next_power_of_two().max(self.obj_cap);
            self.joint_cap = (joint_blocks.max(1)).next_power_of_two().max(self.joint_cap);
            let (o, j, bg) = make_ring(device, &self.obj_layout, self.obj_cap, self.joint_cap);
            self.obj_buf = o;
            self.joint_buf = j;
            self.obj_bg = bg;
        }

        let mut obj_bytes = vec![0u8; (self.obj_cap * stride) as usize];
        let mut joint_bytes = vec![0u8; (self.joint_cap * JOINT_BYTES) as usize];
        // Identity joint block at index 0.
        write_joint_block(&mut joint_bytes, 0, &[Mat4::IDENTITY]);

        let mut offsets: Vec<(u32, u32)> = Vec::with_capacity(self.queue_cmds.len());
        let mut next_joint = 1u64;
        for (i, cmd) in self.queue_cmds.iter().enumerate() {
            let normal_mat = Mat4::from_mat3(Mat3::from_mat4(cmd.model).inverse().transpose());
            let (skinned_flag, joint_off) = match &cmd.joints {
                Some(mats) => {
                    let bi = next_joint;
                    next_joint += 1;
                    write_joint_block(&mut joint_bytes, bi, mats);
                    (1.0f32, bi * JOINT_BYTES)
                }
                None => (0.0f32, 0),
            };
            let mat = &self.materials[cmd.material];
            let obj = ObjU {
                model: cmd.model.to_cols_array_2d(),
                normal_mat: normal_mat.to_cols_array_2d(),
                base_color: mat.base_color,
                params: [if mat.has_texture { 1.0 } else { 0.0 }, skinned_flag, 0.0, 0.0],
            };
            let off = i as u64 * stride;
            obj_bytes[off as usize..off as usize + std::mem::size_of::<ObjU>()]
                .copy_from_slice(bytemuck::bytes_of(&obj));
            offsets.push((off as u32, joint_off as u32));
        }
        queue.write_buffer(&self.obj_buf, 0, &obj_bytes);
        queue.write_buffer(&self.joint_buf, 0, &joint_bytes);

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("render3d"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: color_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: clear[0] as f64,
                        g: clear[1] as f64,
                        b: clear[2] as f64,
                        a: clear[3] as f64,
                    }),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &self.depth,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.globals_bg, &[]);
        for (cmd, (obj_off, joint_off)) in self.queue_cmds.iter().zip(offsets) {
            let m = &self.meshes[cmd.mesh];
            pass.set_bind_group(1, &self.obj_bg, &[obj_off, joint_off]);
            pass.set_bind_group(2, &self.materials[cmd.material].bind_group, &[]);
            pass.set_vertex_buffer(0, m.vbuf.slice(..));
            pass.set_index_buffer(m.ibuf.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..m.index_count, 0, 0..1);
        }
    }
}

fn write_joint_block(buf: &mut [u8], block: u64, mats: &[Mat4]) {
    let base = (block * JOINT_BYTES) as usize;
    for (i, m) in mats.iter().take(MAX_JOINTS).enumerate() {
        let off = base + i * 64;
        buf[off..off + 64].copy_from_slice(bytemuck::bytes_of(&m.to_cols_array()));
    }
}

fn make_ring(
    device: &wgpu::Device,
    obj_layout: &wgpu::BindGroupLayout,
    obj_cap: u64,
    joint_cap: u64,
) -> (wgpu::Buffer, wgpu::Buffer, wgpu::BindGroup) {
    let obj_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("obj-ring"),
        size: obj_cap * OBJ_ALIGN,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let joint_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("joint-ring"),
        size: joint_cap * JOINT_BYTES,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let obj_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("obj-ring"),
        layout: obj_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &obj_buf,
                    offset: 0,
                    size: Some(std::num::NonZeroU64::new(std::mem::size_of::<ObjU>() as u64).unwrap()),
                }),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &joint_buf,
                    offset: 0,
                    size: Some(std::num::NonZeroU64::new(JOINT_BYTES).unwrap()),
                }),
            },
        ],
    });
    (obj_buf, joint_buf, obj_bg)
}

fn make_depth(device: &wgpu::Device, w: u32, h: u32) -> wgpu::TextureView {
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("depth"),
        size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: DEPTH_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    tex.create_view(&wgpu::TextureViewDescriptor::default())
}

fn uniform_entry(
    binding: u32,
    visibility: wgpu::ShaderStages,
    dynamic: bool,
    min_size: Option<u64>,
) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: dynamic,
            min_binding_size: min_size.and_then(std::num::NonZeroU64::new),
        },
        count: None,
    }
}

const SHADER: &str = r#"
struct Globals {
    view_proj: mat4x4<f32>,
    cam_pos: vec4<f32>,
    light_dir: vec4<f32>,
    light_color: vec4<f32>,
};
@group(0) @binding(0) var<uniform> globals: Globals;

struct ObjU {
    model: mat4x4<f32>,
    normal_mat: mat4x4<f32>,
    base_color: vec4<f32>,
    params: vec4<f32>,
};
@group(1) @binding(0) var<uniform> obj: ObjU;

struct Joints { m: array<mat4x4<f32>, 128> };
@group(1) @binding(1) var<uniform> joints: Joints;

@group(2) @binding(0) var tex: texture_2d<f32>;
@group(2) @binding(1) var samp: sampler;

struct VsIn {
    @location(0) pos: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) joints: vec4<u32>,
    @location(4) weights: vec4<f32>,
};
struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) world_normal: vec3<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) world_pos: vec3<f32>,
};

@vertex
fn vs(in: VsIn) -> VsOut {
    var local = vec4<f32>(in.pos, 1.0);
    var nrm = in.normal;
    if (obj.params.y > 0.5) {
        let skin = joints.m[in.joints.x] * in.weights.x
                 + joints.m[in.joints.y] * in.weights.y
                 + joints.m[in.joints.z] * in.weights.z
                 + joints.m[in.joints.w] * in.weights.w;
        local = skin * local;
        nrm = (skin * vec4<f32>(in.normal, 0.0)).xyz;
    }
    let world = obj.model * local;
    var out: VsOut;
    out.clip = globals.view_proj * world;
    out.world_normal = (obj.normal_mat * vec4<f32>(nrm, 0.0)).xyz;
    out.uv = in.uv;
    out.world_pos = world.xyz;
    return out;
}

@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
    var albedo = obj.base_color;
    if (obj.params.x > 0.5) {
        albedo = albedo * textureSample(tex, samp, in.uv);
    }
    let N = normalize(in.world_normal);
    let L = normalize(globals.light_dir.xyz);
    let diff = max(dot(N, L), 0.0);
    let ambient = globals.light_color.w;
    let lit = albedo.rgb * (globals.light_color.rgb * diff + vec3<f32>(ambient, ambient, ambient));
    return vec4<f32>(lit, albedo.a);
}
"#;
