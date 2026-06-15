//! The forward 3D renderer: a PBR pipeline (Cook-Torrance metallic/roughness,
//! normal mapping, emissive) lit by a directional light plus point lights, with
//! fog and a depth buffer, drawing indexed, optionally skinned meshes.

use bytemuck::{Pod, Zeroable};
use glam::{Mat3, Mat4, Vec3};

use crate::mesh::{GpuMesh, MeshData, Vertex};

/// Maximum joints per skinned mesh (fits a 16 KiB uniform: 128 * 64 B = 8 KiB).
pub const MAX_JOINTS: usize = 128;
/// Maximum simultaneous point lights.
pub const MAX_LIGHTS: usize = 16;
const OBJ_ALIGN: u64 = 256;
const JOINT_BYTES: u64 = (MAX_JOINTS * 64) as u64;
const SHADOW_SIZE: u32 = 2048;

pub const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PointLightU {
    pos_range: [f32; 4],  // xyz position, w range
    color_int: [f32; 4],  // rgb color, w intensity
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GlobalsU {
    view_proj: [[f32; 4]; 4],
    light_vp: [[f32; 4]; 4],     // directional shadow light view-projection
    inv_view_proj: [[f32; 4]; 4], // for reconstructing skybox view rays
    cam_pos: [f32; 4],
    dir_dir: [f32; 4],    // xyz direction toward the light, w intensity
    dir_color: [f32; 4],  // rgb color, w ambient
    fog_color: [f32; 4],  // rgb, w density (0 = no fog)
    sky_top: [f32; 4],    // zenith color
    sky_horizon: [f32; 4], // horizon color
    counts: [f32; 4],     // x = point light count, y = shadows on, z = sky on
    lights: [PointLightU; MAX_LIGHTS],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ObjU {
    model: [[f32; 4]; 4],
    normal_mat: [[f32; 4]; 4],
    params: [f32; 4], // x = skinned
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct MatU {
    base_color: [f32; 4],
    emissive: [f32; 4],
    mr: [f32; 4],    // metallic, roughness, normal_scale, occlusion
    flags: [f32; 4], // has_base, has_normal, has_mr, has_emissive
}

/// Describes a material to register.
pub struct MaterialDesc<'a> {
    pub base_color: [f32; 4],
    pub metallic: f32,
    pub roughness: f32,
    pub emissive: [f32; 3],
    pub base_tex: Option<(&'a [u8], u32, u32)>,
    pub normal_tex: Option<(&'a [u8], u32, u32)>,
    pub mr_tex: Option<(&'a [u8], u32, u32)>,
    pub emissive_tex: Option<(&'a [u8], u32, u32)>,
}

impl<'a> MaterialDesc<'a> {
    /// A flat (untextured) material from a base color.
    pub fn flat(color: [f32; 4]) -> MaterialDesc<'a> {
        MaterialDesc {
            base_color: color,
            metallic: 0.0,
            roughness: 0.8,
            emissive: [0.0; 3],
            base_tex: None,
            normal_tex: None,
            mr_tex: None,
            emissive_tex: None,
        }
    }
}

pub struct Material {
    pub bind_group: wgpu::BindGroup,
    pub transparent: bool,
}

struct DrawCmd {
    mesh: usize,
    material: usize,
    model: Mat4,
    joints: Option<Vec<Mat4>>,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct LineVert {
    pos: [f32; 3],
    color: [f32; 3],
}
impl LineVert {
    const LAYOUT: wgpu::VertexBufferLayout<'static> = wgpu::VertexBufferLayout {
        array_stride: 24,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3],
    };
}

pub struct Renderer3D {
    pipeline: wgpu::RenderPipeline,
    globals_buf: wgpu::Buffer,
    globals_bg: wgpu::BindGroup,
    obj_layout: wgpu::BindGroupLayout,
    mat_layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,

    obj_buf: wgpu::Buffer,
    joint_buf: wgpu::Buffer,
    obj_bg: wgpu::BindGroup,
    obj_cap: u64,
    joint_cap: u64,

    depth: wgpu::TextureView,
    depth_size: (u32, u32),
    sample_count: u32,
    color_format: wgpu::TextureFormat,
    msaa_color: Option<wgpu::TextureView>,

    shadow_view: wgpu::TextureView,
    shadow_pipeline: wgpu::RenderPipeline,
    shadow_globals_bg: wgpu::BindGroup,
    shadow_extent: f32,
    shadows_on: bool,
    sky_pipeline: wgpu::RenderPipeline,
    line_pipeline: wgpu::RenderPipeline,
    line_buf: wgpu::Buffer,
    line_cap: u64,
    line_verts: Vec<LineVert>,

    meshes: Vec<GpuMesh>,
    mesh_radius: Vec<f32>,
    materials: Vec<Material>,

    frustum_cull: bool,
    last_drawn: usize,

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
        samples: u32,
    ) -> Renderer3D {
        let sample_count = samples.max(1);
        let globals_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("globals"),
            entries: &[
                uniform_entry(0, wgpu::ShaderStages::VERTEX_FRAGMENT, false, None),
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Comparison),
                    count: None,
                },
            ],
        });
        // Shadow pass uses just the globals uniform (no shadow-map binding, since
        // it renders INTO the shadow map).
        let shadow_g_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("shadow-globals"),
            entries: &[uniform_entry(0, wgpu::ShaderStages::VERTEX, false, None)],
        });
        let obj_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("object"),
            entries: &[
                uniform_entry(0, wgpu::ShaderStages::VERTEX_FRAGMENT, true, Some(std::mem::size_of::<ObjU>() as u64)),
                uniform_entry(1, wgpu::ShaderStages::VERTEX, true, Some(JOINT_BYTES)),
            ],
        });
        let mut mat_entries = vec![uniform_entry(0, wgpu::ShaderStages::FRAGMENT, false, Some(std::mem::size_of::<MatU>() as u64))];
        for b in 1..=4 {
            mat_entries.push(wgpu::BindGroupLayoutEntry {
                binding: b,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            });
        }
        mat_entries.push(wgpu::BindGroupLayoutEntry {
            binding: 5,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
            count: None,
        });
        let mat_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("material"),
            entries: &mat_entries,
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
            multisample: wgpu::MultisampleState { count: sample_count, ..Default::default() },
            multiview: None,
            cache: None,
        });

        let globals_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("globals"),
            size: std::mem::size_of::<GlobalsU>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Shadow map (a depth texture rendered from the light's POV) + a
        // comparison sampler for PCF filtering.
        let shadow_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("shadow-map"),
            size: wgpu::Extent3d { width: SHADOW_SIZE, height: SHADOW_SIZE, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: DEPTH_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let shadow_view = shadow_tex.create_view(&wgpu::TextureViewDescriptor::default());
        let shadow_cmp_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("shadow-cmp"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            compare: Some(wgpu::CompareFunction::LessEqual),
            ..Default::default()
        });

        let globals_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("globals"),
            layout: &globals_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: globals_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&shadow_view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&shadow_cmp_sampler) },
            ],
        });
        let shadow_globals_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("shadow-globals"),
            layout: &shadow_g_layout,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: globals_buf.as_entire_binding() }],
        });

        // Depth-only shadow pipeline (vertex transforms by light_vp; skinned too).
        let shadow_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("shadow"),
            bind_group_layouts: &[&shadow_g_layout, &obj_layout],
            push_constant_ranges: &[],
        });
        let shadow_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("shadow"),
            layout: Some(&shadow_pl),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: "vs_shadow",
                compilation_options: Default::default(),
                buffers: &[Vertex::LAYOUT],
            },
            fragment: None,
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: None, // render both faces to reduce peter-panning
                front_face: wgpu::FrontFace::Ccw,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: Default::default(),
                bias: wgpu::DepthBiasState { constant: 2, slope_scale: 2.0, clamp: 0.0 },
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        // Procedural skybox: a fullscreen triangle, depth-always, no depth write.
        let sky_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("sky"),
            bind_group_layouts: &[&globals_layout],
            push_constant_ranges: &[],
        });
        let sky_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("sky"),
            layout: Some(&sky_pl),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: "vs_sky",
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: "fs_sky",
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: color_format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::Always,
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: wgpu::MultisampleState { count: sample_count, ..Default::default() },
            multiview: None,
            cache: None,
        });

        // Debug line pipeline (line list, depth-tested, no depth write).
        let line_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("lines"),
            bind_group_layouts: &[&globals_layout],
            push_constant_ranges: &[],
        });
        let line_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("lines"),
            layout: Some(&line_pl),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: "vs_line",
                compilation_options: Default::default(),
                buffers: &[LineVert::LAYOUT],
            },
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: "fs_line",
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: color_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::LineList,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::LessEqual,
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: wgpu::MultisampleState { count: sample_count, ..Default::default() },
            multiview: None,
            cache: None,
        });
        let line_cap = 1024u64;
        let line_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("lines"),
            size: line_cap * std::mem::size_of::<LineVert>() as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
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
        let (obj_buf, joint_buf, obj_bg) = make_ring(device, &obj_layout, obj_cap, joint_cap);
        let depth = make_depth(device, w.max(1), h.max(1), sample_count);
        let msaa_color = if sample_count > 1 {
            Some(make_msaa_color(device, color_format, w.max(1), h.max(1), sample_count))
        } else {
            None
        };

        let globals = GlobalsU {
            view_proj: Mat4::IDENTITY.to_cols_array_2d(),
            light_vp: Mat4::IDENTITY.to_cols_array_2d(),
            inv_view_proj: Mat4::IDENTITY.to_cols_array_2d(),
            cam_pos: [0.0, 0.0, 0.0, 1.0],
            dir_dir: Vec3::new(0.4, 1.0, 0.3).normalize().extend(1.0).into(),
            dir_color: [1.0, 1.0, 1.0, 0.15],
            fog_color: [0.0, 0.0, 0.0, 0.0],
            sky_top: [0.20, 0.40, 0.75, 1.0],
            sky_horizon: [0.70, 0.80, 0.92, 1.0],
            counts: [0.0, 1.0, 0.0, 0.0],
            lights: [PointLightU { pos_range: [0.0; 4], color_int: [0.0; 4] }; MAX_LIGHTS],
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
            sample_count,
            color_format,
            msaa_color,
            shadow_view,
            shadow_pipeline,
            shadow_globals_bg,
            shadow_extent: 50.0,
            shadows_on: true,
            sky_pipeline,
            line_pipeline,
            line_buf,
            line_cap,
            line_verts: Vec::new(),
            meshes: Vec::new(),
            mesh_radius: Vec::new(),
            materials: Vec::new(),
            frustum_cull: true,
            last_drawn: 0,
            globals,
            queue_cmds: Vec::new(),
        };
        // Material 0: a plain white default.
        let m0 = r.build_material(device, queue, &MaterialDesc::flat([1.0; 4]));
        r.materials.push(m0);
        r
    }

    pub fn resize(&mut self, device: &wgpu::Device, w: u32, h: u32) {
        if (w, h) != self.depth_size && w > 0 && h > 0 {
            self.depth = make_depth(device, w, h, self.sample_count);
            if self.sample_count > 1 {
                self.msaa_color = Some(make_msaa_color(device, self.color_format, w, h, self.sample_count));
            }
            self.depth_size = (w, h);
        }
    }

    pub fn set_camera(&mut self, view_proj: Mat4, cam_pos: Vec3) {
        self.globals.view_proj = view_proj.to_cols_array_2d();
        self.globals.inv_view_proj = view_proj.inverse().to_cols_array_2d();
        self.globals.cam_pos = cam_pos.extend(1.0).into();
    }

    /// Enable a procedural sky and set its zenith/horizon colors.
    pub fn set_sky(&mut self, on: bool, top: Vec3, horizon: Vec3) {
        self.globals.counts[2] = if on { 1.0 } else { 0.0 };
        self.globals.sky_top = top.extend(1.0).into();
        self.globals.sky_horizon = horizon.extend(1.0).into();
    }

    pub fn set_light(&mut self, dir: Vec3, color: Vec3, ambient: f32) {
        let d = dir.normalize_or_zero();
        self.globals.dir_dir = [d.x, d.y, d.z, 1.0];
        self.globals.dir_color = color.extend(ambient).into();
    }

    pub fn set_fog(&mut self, color: Vec3, density: f32) {
        self.globals.fog_color = color.extend(density.max(0.0)).into();
    }

    pub fn set_shadows(&mut self, on: bool) {
        self.shadows_on = on;
    }

    /// Half-size of the orthographic shadow frustum (world units around camera).
    pub fn set_shadow_extent(&mut self, extent: f32) {
        self.shadow_extent = extent.max(1.0);
    }

    pub fn clear_point_lights(&mut self) {
        self.globals.counts[0] = 0.0;
    }

    pub fn add_point_light(&mut self, pos: Vec3, color: Vec3, range: f32, intensity: f32) {
        let n = self.globals.counts[0] as usize;
        if n < MAX_LIGHTS {
            self.globals.lights[n] = PointLightU {
                pos_range: [pos.x, pos.y, pos.z, range.max(0.001)],
                color_int: [color.x, color.y, color.z, intensity],
            };
            self.globals.counts[0] = (n + 1) as f32;
        }
    }

    pub fn add_mesh(&mut self, device: &wgpu::Device, mesh: &MeshData) -> usize {
        let radius = mesh
            .vertices
            .iter()
            .map(|v| Vec3::from(v.pos).length())
            .fold(0.0f32, f32::max);
        self.meshes.push(GpuMesh::upload(device, mesh));
        self.mesh_radius.push(radius.max(0.001));
        self.meshes.len() - 1
    }

    pub fn set_frustum_cull(&mut self, on: bool) {
        self.frustum_cull = on;
    }
    /// Number of draws that survived frustum culling in the last `render`.
    pub fn last_drawn(&self) -> usize {
        self.last_drawn
    }

    /// Queue a world-space debug line segment.
    pub fn debug_line(&mut self, a: Vec3, b: Vec3, color: Vec3) {
        self.line_verts.push(LineVert { pos: a.into(), color: color.into() });
        self.line_verts.push(LineVert { pos: b.into(), color: color.into() });
    }

    pub fn add_material(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        desc: &MaterialDesc,
    ) -> usize {
        let m = self.build_material(device, queue, desc);
        self.materials.push(m);
        self.materials.len() - 1
    }

    fn build_material(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        desc: &MaterialDesc,
    ) -> Material {
        // Each material binds its own textures; missing ones get a 1x1 default
        // (white base, flat normal, white metallic-roughness, white emissive).
        let base_v = match desc.base_tex {
            Some((px, w, h)) if w > 0 && h > 0 => make_tex(device, queue, px, w, h, true),
            _ => make_pixel_tex(device, queue, [255, 255, 255, 255], true),
        };
        let normal_v = match desc.normal_tex {
            Some((px, w, h)) if w > 0 && h > 0 => make_tex(device, queue, px, w, h, false),
            _ => make_pixel_tex(device, queue, [128, 128, 255, 255], false),
        };
        let mr_v = match desc.mr_tex {
            Some((px, w, h)) if w > 0 && h > 0 => make_tex(device, queue, px, w, h, false),
            _ => make_pixel_tex(device, queue, [255, 255, 255, 255], false),
        };
        let em_v = match desc.emissive_tex {
            Some((px, w, h)) if w > 0 && h > 0 => make_tex(device, queue, px, w, h, true),
            _ => make_pixel_tex(device, queue, [255, 255, 255, 255], true),
        };

        let u = MatU {
            base_color: desc.base_color,
            emissive: [desc.emissive[0], desc.emissive[1], desc.emissive[2], 0.0],
            mr: [desc.metallic, desc.roughness, 1.0, 1.0],
            flags: [
                desc.base_tex.is_some() as i32 as f32,
                desc.normal_tex.is_some() as i32 as f32,
                desc.mr_tex.is_some() as i32 as f32,
                desc.emissive_tex.is_some() as i32 as f32,
            ],
        };
        let ubuf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("material"),
            size: std::mem::size_of::<MatU>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&ubuf, 0, bytemuck::bytes_of(&u));

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("material"),
            layout: &self.mat_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: ubuf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&base_v) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&normal_v) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&mr_v) },
                wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(&em_v) },
                wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::Sampler(&self.sampler) },
            ],
        });
        Material { bind_group, transparent: desc.base_color[3] < 0.999 }
    }

    /// The current camera view-projection matrix.
    pub fn view_proj(&self) -> Mat4 {
        Mat4::from_cols_array_2d(&self.globals.view_proj)
    }

    pub fn material_count(&self) -> usize {
        self.materials.len()
    }
    pub fn mesh_count(&self) -> usize {
        self.meshes.len()
    }

    pub fn begin(&mut self) {
        self.queue_cmds.clear();
        self.line_verts.clear();
    }

    pub fn draw(&mut self, mesh: usize, material: usize, model: Mat4, joints: Option<Vec<Mat4>>) {
        if mesh < self.meshes.len() {
            let material = if material < self.materials.len() { material } else { 0 };
            self.queue_cmds.push(DrawCmd { mesh, material, model, joints });
        }
    }

    pub fn render(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        color_view: &wgpu::TextureView,
        clear: [f32; 4],
    ) {
        // Directional shadow light matrix: an orthographic box centered on the
        // camera, looking along the light direction.
        let d = Vec3::new(self.globals.dir_dir[0], self.globals.dir_dir[1], self.globals.dir_dir[2])
            .normalize_or_zero();
        let center = Vec3::new(self.globals.cam_pos[0], self.globals.cam_pos[1], self.globals.cam_pos[2]);
        let e = self.shadow_extent;
        let eye = center + d * (e * 2.0);
        let up = if d.y.abs() > 0.95 { Vec3::Z } else { Vec3::Y };
        let light_vp = Mat4::orthographic_rh(-e, e, -e, e, 0.1, e * 4.5)
            * Mat4::look_at_rh(eye, center, up);
        self.globals.light_vp = light_vp.to_cols_array_2d();
        self.globals.counts[1] = if self.shadows_on { 1.0 } else { 0.0 };

        queue.write_buffer(&self.globals_buf, 0, bytemuck::bytes_of(&self.globals));

        let stride = OBJ_ALIGN;
        let n = self.queue_cmds.len() as u64;
        let skinned: u64 = self.queue_cmds.iter().filter(|c| c.joints.is_some()).count() as u64;
        let joint_blocks = skinned + 1;
        if n > self.obj_cap || joint_blocks > self.joint_cap {
            self.obj_cap = n.max(1).next_power_of_two().max(self.obj_cap);
            self.joint_cap = joint_blocks.max(1).next_power_of_two().max(self.joint_cap);
            let (o, j, bg) = make_ring(device, &self.obj_layout, self.obj_cap, self.joint_cap);
            self.obj_buf = o;
            self.joint_buf = j;
            self.obj_bg = bg;
        }

        let mut obj_bytes = vec![0u8; (self.obj_cap * stride) as usize];
        let mut joint_bytes = vec![0u8; (self.joint_cap * JOINT_BYTES) as usize];
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
            let obj = ObjU {
                model: cmd.model.to_cols_array_2d(),
                normal_mat: normal_mat.to_cols_array_2d(),
                params: [skinned_flag, 0.0, 0.0, 0.0],
            };
            let off = i as u64 * stride;
            obj_bytes[off as usize..off as usize + std::mem::size_of::<ObjU>()]
                .copy_from_slice(bytemuck::bytes_of(&obj));
            offsets.push((off as u32, joint_off as u32));
        }
        queue.write_buffer(&self.obj_buf, 0, &obj_bytes);
        queue.write_buffer(&self.joint_buf, 0, &joint_bytes);

        // Draw opaque first, then transparent back-to-front.
        let cam = Vec3::from_slice(&self.globals.cam_pos[..3]);
        let mut order: Vec<usize> = (0..self.queue_cmds.len()).collect();
        order.sort_by(|&a, &b| {
            let ta = self.materials[self.queue_cmds[a].material].transparent;
            let tb = self.materials[self.queue_cmds[b].material].transparent;
            if ta != tb {
                return ta.cmp(&tb); // opaque (false) first
            }
            if ta {
                // back-to-front for transparent
                let da = (cam - self.queue_cmds[a].model.w_axis.truncate()).length_squared();
                let db = (cam - self.queue_cmds[b].model.w_axis.truncate()).length_squared();
                db.partial_cmp(&da).unwrap_or(std::cmp::Ordering::Equal)
            } else {
                std::cmp::Ordering::Equal
            }
        });

        // Shadow pass: render scene depth from the light's POV into the shadow map.
        if self.shadows_on {
            let mut sp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("shadow"),
                color_attachments: &[],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.shadow_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            sp.set_pipeline(&self.shadow_pipeline);
            sp.set_bind_group(0, &self.shadow_globals_bg, &[]);
            for (ci, &(obj_off, joint_off)) in offsets.iter().enumerate() {
                let cmd = &self.queue_cmds[ci];
                let m = &self.meshes[cmd.mesh];
                sp.set_bind_group(1, &self.obj_bg, &[obj_off, joint_off]);
                sp.set_vertex_buffer(0, m.vbuf.slice(..));
                sp.set_index_buffer(m.ibuf.slice(..), wgpu::IndexFormat::Uint32);
                sp.draw_indexed(0..m.index_count, 0, 0..1);
            }
        }

        // With MSAA, render into the multisampled color target and resolve into
        // the caller's view; otherwise render straight into it.
        let (attach_view, resolve) = match &self.msaa_color {
            Some(msaa) if self.sample_count > 1 => (msaa, Some(color_view)),
            _ => (color_view, None),
        };
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("render3d"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: attach_view,
                resolve_target: resolve,
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
        if self.globals.counts[2] > 0.5 {
            pass.set_pipeline(&self.sky_pipeline);
            pass.set_bind_group(0, &self.globals_bg, &[]);
            pass.draw(0..3, 0..1);
        }
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.globals_bg, &[]);
        let planes = frustum_planes(Mat4::from_cols_array_2d(&self.globals.view_proj));
        let mut drawn = 0usize;
        for &ci in &order {
            let cmd = &self.queue_cmds[ci];
            // Frustum cull by the mesh's bounding sphere (scaled by the model).
            if self.frustum_cull {
                let center = cmd.model.w_axis.truncate();
                let scale = cmd.model.x_axis.truncate().length()
                    .max(cmd.model.y_axis.truncate().length())
                    .max(cmd.model.z_axis.truncate().length());
                let radius = self.mesh_radius[cmd.mesh] * scale;
                if !sphere_in_frustum(&planes, center, radius) {
                    continue;
                }
            }
            let (obj_off, joint_off) = offsets[ci];
            let m = &self.meshes[cmd.mesh];
            pass.set_bind_group(1, &self.obj_bg, &[obj_off, joint_off]);
            pass.set_bind_group(2, &self.materials[cmd.material].bind_group, &[]);
            pass.set_vertex_buffer(0, m.vbuf.slice(..));
            pass.set_index_buffer(m.ibuf.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..m.index_count, 0, 0..1);
            drawn += 1;
        }

        // Debug lines, drawn last (depth-tested against the scene).
        if !self.line_verts.is_empty() {
            let needed = self.line_verts.len() as u64;
            if needed > self.line_cap {
                self.line_cap = needed.next_power_of_two();
                self.line_buf = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("lines"),
                    size: self.line_cap * std::mem::size_of::<LineVert>() as u64,
                    usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
            }
            queue.write_buffer(&self.line_buf, 0, bytemuck::cast_slice(&self.line_verts));
            pass.set_pipeline(&self.line_pipeline);
            pass.set_bind_group(0, &self.globals_bg, &[]);
            pass.set_vertex_buffer(0, self.line_buf.slice(..));
            pass.draw(0..self.line_verts.len() as u32, 0..1);
        }
        drop(pass);
        self.last_drawn = drawn;
    }
}

/// The six frustum planes (a,b,c,d) from a view-projection matrix.
fn frustum_planes(vp: Mat4) -> [glam::Vec4; 6] {
    let r = vp.to_cols_array_2d();
    // Row-vector extraction (column-major storage: m[col][row]).
    let row = |i: usize| glam::Vec4::new(r[0][i], r[1][i], r[2][i], r[3][i]);
    let (r0, r1, r2, r3) = (row(0), row(1), row(2), row(3));
    [
        r3 + r0, // left
        r3 - r0, // right
        r3 + r1, // bottom
        r3 - r1, // top
        r3 + r2, // near
        r3 - r2, // far
    ]
}

fn sphere_in_frustum(planes: &[glam::Vec4; 6], center: Vec3, radius: f32) -> bool {
    for p in planes {
        let n = p.truncate();
        let len = n.length().max(1e-6);
        let dist = (n.dot(center) + p.w) / len;
        if dist < -radius {
            return false;
        }
    }
    true
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

fn make_depth(device: &wgpu::Device, w: u32, h: u32, samples: u32) -> wgpu::TextureView {
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("depth"),
        size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: samples,
        dimension: wgpu::TextureDimension::D2,
        format: DEPTH_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    tex.create_view(&wgpu::TextureViewDescriptor::default())
}

fn make_msaa_color(device: &wgpu::Device, format: wgpu::TextureFormat, w: u32, h: u32, samples: u32) -> wgpu::TextureView {
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("msaa-color"),
        size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: samples,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    tex.create_view(&wgpu::TextureViewDescriptor::default())
}

fn make_tex(device: &wgpu::Device, queue: &wgpu::Queue, px: &[u8], w: u32, h: u32, srgb: bool) -> wgpu::TextureView {
    let format = if srgb { wgpu::TextureFormat::Rgba8UnormSrgb } else { wgpu::TextureFormat::Rgba8Unorm };
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("tex"),
        size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    queue.write_texture(
        wgpu::ImageCopyTexture { texture: &tex, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
        px,
        wgpu::ImageDataLayout { offset: 0, bytes_per_row: Some(w * 4), rows_per_image: Some(h) },
        wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
    );
    tex.create_view(&wgpu::TextureViewDescriptor::default())
}

fn make_pixel_tex(device: &wgpu::Device, queue: &wgpu::Queue, rgba: [u8; 4], srgb: bool) -> wgpu::TextureView {
    make_tex(device, queue, &rgba, 1, 1, srgb)
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
struct PointLight { pos_range: vec4<f32>, color_int: vec4<f32> };
struct Globals {
    view_proj: mat4x4<f32>,
    light_vp: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    cam_pos: vec4<f32>,
    dir_dir: vec4<f32>,
    dir_color: vec4<f32>,
    fog_color: vec4<f32>,
    sky_top: vec4<f32>,
    sky_horizon: vec4<f32>,
    counts: vec4<f32>,
    lights: array<PointLight, 16>,
};
@group(0) @binding(0) var<uniform> g: Globals;
@group(0) @binding(1) var shadow_map: texture_depth_2d;
@group(0) @binding(2) var shadow_samp: sampler_comparison;

struct ObjU { model: mat4x4<f32>, normal_mat: mat4x4<f32>, params: vec4<f32> };
@group(1) @binding(0) var<uniform> obj: ObjU;
struct Joints { m: array<mat4x4<f32>, 128> };
@group(1) @binding(1) var<uniform> joints: Joints;

struct MatU { base_color: vec4<f32>, emissive: vec4<f32>, mr: vec4<f32>, flags: vec4<f32> };
@group(2) @binding(0) var<uniform> mat: MatU;
@group(2) @binding(1) var base_tex: texture_2d<f32>;
@group(2) @binding(2) var normal_tex: texture_2d<f32>;
@group(2) @binding(3) var mr_tex: texture_2d<f32>;
@group(2) @binding(4) var emissive_tex: texture_2d<f32>;
@group(2) @binding(5) var samp: sampler;

struct VsIn {
    @location(0) pos: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) j: vec4<u32>,
    @location(4) w: vec4<f32>,
    @location(5) tangent: vec4<f32>,
};
struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) world_normal: vec3<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) world_pos: vec3<f32>,
    @location(3) world_tangent: vec3<f32>,
    @location(4) tangent_w: f32,
};

@vertex
fn vs(in: VsIn) -> VsOut {
    var local = vec4<f32>(in.pos, 1.0);
    var nrm = in.normal;
    var tan = in.tangent.xyz;
    if (obj.params.x > 0.5) {
        let skin = joints.m[in.j.x] * in.w.x + joints.m[in.j.y] * in.w.y
                 + joints.m[in.j.z] * in.w.z + joints.m[in.j.w] * in.w.w;
        local = skin * local;
        nrm = (skin * vec4<f32>(in.normal, 0.0)).xyz;
        tan = (skin * vec4<f32>(in.tangent.xyz, 0.0)).xyz;
    }
    let world = obj.model * local;
    var out: VsOut;
    out.clip = g.view_proj * world;
    out.world_normal = (obj.normal_mat * vec4<f32>(nrm, 0.0)).xyz;
    out.world_tangent = (obj.model * vec4<f32>(tan, 0.0)).xyz;
    out.tangent_w = in.tangent.w;
    out.uv = in.uv;
    out.world_pos = world.xyz;
    return out;
}

// Debug lines: transform by the camera and pass the vertex color through.
struct LineOut { @builtin(position) clip: vec4<f32>, @location(0) color: vec3<f32> };
@vertex
fn vs_line(@location(0) pos: vec3<f32>, @location(1) color: vec3<f32>) -> LineOut {
    var o: LineOut;
    o.clip = g.view_proj * vec4<f32>(pos, 1.0);
    o.color = color;
    return o;
}
@fragment
fn fs_line(in: LineOut) -> @location(0) vec4<f32> {
    return vec4<f32>(in.color, 1.0);
}

// Procedural skybox: gradient between horizon and zenith plus a sun disk.
struct SkyOut { @builtin(position) clip: vec4<f32>, @location(0) ndc: vec2<f32> };
@vertex
fn vs_sky(@builtin(vertex_index) i: u32) -> SkyOut {
    var p = array<vec2<f32>, 3>(vec2(-1.0, -1.0), vec2(3.0, -1.0), vec2(-1.0, 3.0));
    var o: SkyOut;
    o.clip = vec4<f32>(p[i], 1.0, 1.0);
    o.ndc = p[i];
    return o;
}
@fragment
fn fs_sky(in: SkyOut) -> @location(0) vec4<f32> {
    let far = g.inv_view_proj * vec4<f32>(in.ndc, 1.0, 1.0);
    let world = far.xyz / far.w;
    let dir = normalize(world - g.cam_pos.xyz);
    let t = clamp(dir.y * 0.5 + 0.5, 0.0, 1.0);
    var col = mix(g.sky_horizon.rgb, g.sky_top.rgb, pow(t, 0.6));
    let sun = max(dot(dir, normalize(g.dir_dir.xyz)), 0.0);
    col = col + g.dir_color.rgb * pow(sun, 600.0) * 3.0;       // sun disk
    col = col + g.dir_color.rgb * pow(sun, 8.0) * 0.15;        // glow
    return vec4<f32>(col, 1.0);
}

@vertex
fn vs_shadow(in: VsIn) -> @builtin(position) vec4<f32> {
    var local = vec4<f32>(in.pos, 1.0);
    if (obj.params.x > 0.5) {
        let skin = joints.m[in.j.x] * in.w.x + joints.m[in.j.y] * in.w.y
                 + joints.m[in.j.z] * in.w.z + joints.m[in.j.w] * in.w.w;
        local = skin * local;
    }
    return g.light_vp * (obj.model * local);
}

// Percentage-closer-filtered shadow factor (1 = lit, 0 = fully shadowed).
fn shadow_factor(world_pos: vec3<f32>, ndl: f32) -> f32 {
    if (g.counts.y < 0.5) { return 1.0; }
    let lc = g.light_vp * vec4<f32>(world_pos, 1.0);
    var proj = lc.xyz / lc.w;
    if (proj.z > 1.0 || proj.z < 0.0) { return 1.0; }
    let uv = vec2<f32>(proj.x * 0.5 + 0.5, 1.0 - (proj.y * 0.5 + 0.5));
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) { return 1.0; }
    let bias = max(0.0015 * (1.0 - ndl), 0.0004);
    let texel = 1.0 / 2048.0;
    var sum = 0.0;
    for (var dy = -1; dy <= 1; dy = dy + 1) {
        for (var dx = -1; dx <= 1; dx = dx + 1) {
            let o = vec2<f32>(f32(dx), f32(dy)) * texel;
            sum = sum + textureSampleCompare(shadow_map, shadow_samp, uv + o, proj.z - bias);
        }
    }
    return sum / 9.0;
}

const PI: f32 = 3.14159265;

fn distribution_ggx(n: vec3<f32>, h: vec3<f32>, rough: f32) -> f32 {
    let a = rough * rough;
    let a2 = a * a;
    let ndh = max(dot(n, h), 0.0);
    let d = ndh * ndh * (a2 - 1.0) + 1.0;
    return a2 / max(PI * d * d, 1e-5);
}
fn geometry_schlick(ndv: f32, rough: f32) -> f32 {
    let r = rough + 1.0;
    let k = (r * r) / 8.0;
    return ndv / (ndv * (1.0 - k) + k);
}
fn geometry_smith(n: vec3<f32>, v: vec3<f32>, l: vec3<f32>, rough: f32) -> f32 {
    return geometry_schlick(max(dot(n, v), 0.0), rough) * geometry_schlick(max(dot(n, l), 0.0), rough);
}
fn fresnel(cos_t: f32, f0: vec3<f32>) -> vec3<f32> {
    return f0 + (vec3<f32>(1.0) - f0) * pow(clamp(1.0 - cos_t, 0.0, 1.0), 5.0);
}

fn brdf(n: vec3<f32>, v: vec3<f32>, l: vec3<f32>, radiance: vec3<f32>, albedo: vec3<f32>, metallic: f32, rough: f32) -> vec3<f32> {
    let h = normalize(v + l);
    let ndl = max(dot(n, l), 0.0);
    if (ndl <= 0.0) { return vec3<f32>(0.0); }
    let f0 = mix(vec3<f32>(0.04), albedo, metallic);
    let ndf = distribution_ggx(n, h, rough);
    let geo = geometry_smith(n, v, l, rough);
    let f = fresnel(max(dot(h, v), 0.0), f0);
    let spec = (ndf * geo * f) / max(4.0 * max(dot(n, v), 0.0) * ndl, 1e-4);
    let kd = (vec3<f32>(1.0) - f) * (1.0 - metallic);
    return (kd * albedo / PI + spec) * radiance * ndl;
}

@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
    var albedo = mat.base_color;
    if (mat.flags.x > 0.5) { albedo = albedo * textureSample(base_tex, samp, in.uv); }
    var metallic = mat.mr.x;
    var rough = mat.mr.y;
    if (mat.flags.z > 0.5) {
        let m = textureSample(mr_tex, samp, in.uv);
        rough = rough * m.g;
        metallic = metallic * m.b;
    }
    rough = clamp(rough, 0.04, 1.0);

    var n = normalize(in.world_normal);
    if (mat.flags.y > 0.5) {
        let t = normalize(in.world_tangent - n * dot(n, in.world_tangent));
        let b = cross(n, t) * in.tangent_w;
        let tn = textureSample(normal_tex, samp, in.uv).xyz * 2.0 - 1.0;
        n = normalize(mat3x3<f32>(t, b, n) * tn);
    }
    let v = normalize(g.cam_pos.xyz - in.world_pos);

    // Directional light, attenuated by the shadow map.
    let ndl = max(dot(n, normalize(g.dir_dir.xyz)), 0.0);
    let shadow = shadow_factor(in.world_pos, ndl);
    var lo = brdf(n, v, normalize(g.dir_dir.xyz), g.dir_color.rgb * g.dir_dir.w, albedo.rgb, metallic, rough) * shadow;
    // Point lights.
    let count = i32(g.counts.x);
    for (var i = 0; i < count; i = i + 1) {
        let lp = g.lights[i].pos_range.xyz;
        let range = g.lights[i].pos_range.w;
        let to = lp - in.world_pos;
        let d = length(to);
        let l = to / max(d, 1e-4);
        let att = clamp(1.0 - (d / range), 0.0, 1.0);
        let radiance = g.lights[i].color_int.rgb * g.lights[i].color_int.w * att * att;
        lo = lo + brdf(n, v, l, radiance, albedo.rgb, metallic, rough);
    }
    let ambient = albedo.rgb * g.dir_color.w;
    var emissive = mat.emissive.rgb;
    if (mat.flags.w > 0.5) { emissive = emissive + textureSample(emissive_tex, samp, in.uv).rgb; }
    var color = lo + ambient + emissive;

    // Fog (exponential by camera distance).
    if (g.fog_color.w > 0.0) {
        let dist = length(g.cam_pos.xyz - in.world_pos);
        let f = clamp(exp(-dist * g.fog_color.w), 0.0, 1.0);
        color = mix(g.fog_color.rgb, color, f);
    }
    return vec4<f32>(color, albedo.a);
}
"#;
