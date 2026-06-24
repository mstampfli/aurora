//! The forward 3D renderer: a PBR pipeline (Cook-Torrance metallic/roughness,
//! normal mapping, emissive) lit by a directional light plus point lights, with
//! fog and a depth buffer, drawing indexed, optionally skinned meshes.

use std::sync::Arc;

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
const NUM_CASCADES: usize = 3;
const PCUBE_SIZE: u32 = 1024;

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
    csm_vp: [[[f32; 4]; 4]; 4],   // per-cascade light view-projection (3 used)
    inv_view_proj: [[f32; 4]; 4], // for reconstructing skybox view rays
    csm_splits: [f32; 4],         // cascade radii (x,y,z); selection by distance
    cam_pos: [f32; 4],
    dir_dir: [f32; 4],    // xyz direction toward the light, w intensity
    dir_color: [f32; 4],  // rgb color, w ambient
    fog_color: [f32; 4],  // rgb, w density (0 = no fog)
    sky_top: [f32; 4],    // zenith color
    sky_horizon: [f32; 4], // horizon color
    counts: [f32; 4],     // x = point light count, y = shadows on, z = sky on
    screen: [f32; 4],     // x = width, y = height, z = ssao on
    lights: [PointLightU; MAX_LIGHTS],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ObjU {
    model: [[f32; 4]; 4],
    normal_mat: [[f32; 4]; 4],
    params: [f32; 4],  // x = skinned, yzw = tint offset
    params2: [f32; 4], // x = shield Fresnel strength, y = time
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
    // Skinning matrices, shared via Arc so a model's primitives reference ONE allocation instead
    // of deep-copying the full 128-matrix array per primitive each frame.
    joints: Option<Arc<Vec<Mat4>>>,
    tint: [f32; 3],
    /// Energy-shield Fresnel rim: [strength, time]. strength 0 = off.
    shield: [f32; 2],
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

/// Per-instance data for GPU instancing: a model matrix and a color tint.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct InstanceRaw {
    model: [[f32; 4]; 4],
    tint: [f32; 4],
}
impl InstanceRaw {
    pub fn new(model: Mat4, tint: [f32; 4]) -> InstanceRaw {
        InstanceRaw { model: model.to_cols_array_2d(), tint }
    }
    const LAYOUT: wgpu::VertexBufferLayout<'static> = wgpu::VertexBufferLayout {
        array_stride: 80,
        step_mode: wgpu::VertexStepMode::Instance,
        attributes: &wgpu::vertex_attr_array![
            6 => Float32x4, 7 => Float32x4, 8 => Float32x4, 9 => Float32x4, // model rows
            10 => Float32x4, // tint
        ],
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
    // Reused CPU scratch for per-frame uploads (avoids re-allocating + zeroing
    // these every frame, which caused frame-time spikes with many objects).
    obj_scratch: Vec<u8>,
    joint_scratch: Vec<u8>,
    inst_scratch: Vec<InstanceRaw>,

    depth: wgpu::TextureView,
    depth_size: (u32, u32),
    sample_count: u32,
    color_format: wgpu::TextureFormat,
    msaa_color: Option<wgpu::TextureView>,

    shadow_layer_views: Vec<wgpu::TextureView>,
    csm_buf: wgpu::Buffer,
    shadow_pipeline: wgpu::RenderPipeline,
    shadow_globals_bg: wgpu::BindGroup,
    shadow_extent: f32,
    shadows_on: bool,
    sky_pipeline: wgpu::RenderPipeline,
    line_pipeline: wgpu::RenderPipeline,
    line_buf: wgpu::Buffer,
    line_cap: u64,
    line_verts: Vec<LineVert>,
    inst_pipeline: wgpu::RenderPipeline,
    inst_shadow_pipeline: wgpu::RenderPipeline,
    inst_buf: wgpu::Buffer,
    inst_cap: u64,
    inst_cmds: Vec<(usize, usize, Vec<InstanceRaw>)>,
    // SSAO.
    ssao_on: bool,
    prepass_pipeline: wgpu::RenderPipeline,
    prepass_inst_pipeline: wgpu::RenderPipeline,
    ssao_pipeline: wgpu::RenderPipeline,
    blur_pipeline: wgpu::RenderPipeline,
    ssao: Ssao,
    ao_bg_white: wgpu::BindGroup,
    ao_layout: wgpu::BindGroupLayout,
    ssao_layout: wgpu::BindGroupLayout,
    blur_layout: wgpu::BindGroupLayout,
    // Point-light shadows.
    point_shadows_on: bool,
    pshadow_pipeline: wgpu::RenderPipeline,
    pshadow_g_bg: wgpu::BindGroup,
    pshadow_buf: wgpu::Buffer,
    pshadow_face_views: Vec<wgpu::TextureView>,
    pshadow_depth: wgpu::TextureView,
    pshadow_bg: wgpu::BindGroup,

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
                        view_dimension: wgpu::TextureViewDimension::D2Array,
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
        // Shadow pass binds the current cascade's light matrix (dynamic offset
        // into the per-cascade buffer) at binding 3, so it doesn't collide with
        // the main pipeline's globals at binding 0.
        let shadow_g_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("shadow-globals"),
            entries: &[uniform_entry(3, wgpu::ShaderStages::VERTEX, true, Some(64))],
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
        // AO (group 3): the (blurred) SSAO texture sampled in the lighting pass.
        let tex_entry = |b: u32| wgpu::BindGroupLayoutEntry {
            binding: b,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };
        let samp_entry = |b: u32| wgpu::BindGroupLayoutEntry {
            binding: b,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
            count: None,
        };
        let ao_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ao"),
            entries: &[tex_entry(0), samp_entry(1)],
        });
        let ssao_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ssao-in"),
            entries: &[
                uniform_entry(0, wgpu::ShaderStages::FRAGMENT, false, None),
                tex_entry(1),
                samp_entry(2),
            ],
        });
        let blur_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("blur-in"),
            // Match the SSAO module's binding numbers (src=1, samp=2); group 0 (g)
            // is unused by fs_blur and pruned.
            entries: &[tex_entry(1), samp_entry(2)],
        });
        // Point-light shadow cube (group 4) + the per-face uniform (binding 5).
        let pshadow_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("pshadow"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::Cube,
                        multisampled: false,
                    },
                    count: None,
                },
                samp_entry(1),
            ],
        });
        let pshadow_g_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("pshadow-face"),
            entries: &[uniform_entry(5, wgpu::ShaderStages::VERTEX_FRAGMENT, true, Some(80))],
        });

        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("render3d"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });
        let ssao_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ssao"),
            source: wgpu::ShaderSource::Wgsl(SSAO_WGSL.into()),
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("render3d"),
            bind_group_layouts: &[&globals_layout, &obj_layout, &mat_layout, &ao_layout, &pshadow_layout],
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

        // Cascaded shadow maps: a depth-texture ARRAY (one layer per cascade) +
        // a comparison sampler for PCF filtering.
        let shadow_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("shadow-map"),
            size: wgpu::Extent3d {
                width: SHADOW_SIZE,
                height: SHADOW_SIZE,
                depth_or_array_layers: NUM_CASCADES as u32,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: DEPTH_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        // One D2 view per cascade (render targets) + one array view (sampling).
        let shadow_layer_views: Vec<wgpu::TextureView> = (0..NUM_CASCADES as u32)
            .map(|i| {
                shadow_tex.create_view(&wgpu::TextureViewDescriptor {
                    label: Some("shadow-layer"),
                    dimension: Some(wgpu::TextureViewDimension::D2),
                    base_array_layer: i,
                    array_layer_count: Some(1),
                    ..Default::default()
                })
            })
            .collect();
        let shadow_array_view = shadow_tex.create_view(&wgpu::TextureViewDescriptor {
            label: Some("shadow-array"),
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });
        let shadow_cmp_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("shadow-cmp"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            compare: Some(wgpu::CompareFunction::LessEqual),
            ..Default::default()
        });
        // Per-cascade light matrices, one 256-aligned block each (dynamic offset).
        let csm_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cascades"),
            size: (NUM_CASCADES as u64) * 256,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let globals_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("globals"),
            layout: &globals_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: globals_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&shadow_array_view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&shadow_cmp_sampler) },
            ],
        });
        let shadow_globals_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("shadow-globals"),
            layout: &shadow_g_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &csm_buf,
                    offset: 0,
                    size: Some(std::num::NonZeroU64::new(64).unwrap()),
                }),
            }],
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

        // Instanced pipeline: one draw for many copies of a mesh, with per-instance
        // model matrix + tint read from an instance buffer.
        let inst_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("inst"),
            // Same group indices as the main pipeline (globals=0, object=1 unused,
            // material=2, ao=3, point-shadow=4) so shared bindings keep their @group.
            bind_group_layouts: &[&globals_layout, &obj_layout, &mat_layout, &ao_layout, &pshadow_layout],
            push_constant_ranges: &[],
        });
        let inst_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("inst"),
            layout: Some(&inst_pl),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: "vs_inst",
                compilation_options: Default::default(),
                buffers: &[Vertex::LAYOUT, InstanceRaw::LAYOUT],
            },
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: "fs_inst",
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
        let inst_shadow_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("inst-shadow"),
            bind_group_layouts: &[&shadow_g_layout],
            push_constant_ranges: &[],
        });
        let inst_shadow_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("inst-shadow"),
            layout: Some(&inst_shadow_pl),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: "vs_shadow_inst",
                compilation_options: Default::default(),
                buffers: &[Vertex::LAYOUT, InstanceRaw::LAYOUT],
            },
            fragment: None,
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: None,
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
        let inst_cap = 256u64;
        let inst_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("instances"),
            size: inst_cap * std::mem::size_of::<InstanceRaw>() as u64,
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

        // SSAO pipelines (geometry prepass -> occlusion -> blur). The prepass
        // reuses the main vertex stages; the SSAO/blur passes are fullscreen.
        let prepass_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("prepass"),
            bind_group_layouts: &[&globals_layout, &obj_layout],
            push_constant_ranges: &[],
        });
        let prepass_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("prepass"),
            layout: Some(&prepass_pl),
            vertex: wgpu::VertexState { module: &module, entry_point: "vs", compilation_options: Default::default(), buffers: &[Vertex::LAYOUT] },
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: "fs_prepass",
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState { format: wgpu::TextureFormat::Rgba16Float, blend: None, write_mask: wgpu::ColorWrites::ALL })],
            }),
            primitive: wgpu::PrimitiveState { topology: wgpu::PrimitiveTopology::TriangleList, cull_mode: Some(wgpu::Face::Back), front_face: wgpu::FrontFace::Ccw, ..Default::default() },
            depth_stencil: Some(wgpu::DepthStencilState { format: DEPTH_FORMAT, depth_write_enabled: true, depth_compare: wgpu::CompareFunction::Less, stencil: Default::default(), bias: Default::default() }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });
        let prepass_inst_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("prepass-inst"),
            bind_group_layouts: &[&globals_layout],
            push_constant_ranges: &[],
        });
        let prepass_inst_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("prepass-inst"),
            layout: Some(&prepass_inst_pl),
            vertex: wgpu::VertexState { module: &module, entry_point: "vs_inst", compilation_options: Default::default(), buffers: &[Vertex::LAYOUT, InstanceRaw::LAYOUT] },
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: "fs_prepass_inst",
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState { format: wgpu::TextureFormat::Rgba16Float, blend: None, write_mask: wgpu::ColorWrites::ALL })],
            }),
            primitive: wgpu::PrimitiveState { topology: wgpu::PrimitiveTopology::TriangleList, cull_mode: Some(wgpu::Face::Back), front_face: wgpu::FrontFace::Ccw, ..Default::default() },
            depth_stencil: Some(wgpu::DepthStencilState { format: DEPTH_FORMAT, depth_write_enabled: true, depth_compare: wgpu::CompareFunction::Less, stencil: Default::default(), bias: Default::default() }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });
        let fullscreen_pipe = |layout: &wgpu::PipelineLayout, fs: &str| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("ssao-pass"),
                layout: Some(layout),
                vertex: wgpu::VertexState { module: &ssao_module, entry_point: "vs_fs", compilation_options: Default::default(), buffers: &[] },
                fragment: Some(wgpu::FragmentState {
                    module: &ssao_module,
                    entry_point: fs,
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState { format: wgpu::TextureFormat::R8Unorm, blend: None, write_mask: wgpu::ColorWrites::ALL })],
                }),
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
                cache: None,
            })
        };
        let ssao_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("ssao"), bind_group_layouts: &[&ssao_layout], push_constant_ranges: &[] });
        let blur_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("blur"), bind_group_layouts: &[&blur_layout], push_constant_ranges: &[] });
        let ssao_pipeline = fullscreen_pipe(&ssao_pl, "fs_ssao");
        let blur_pipeline = fullscreen_pipe(&blur_pl, "fs_blur");

        // A 1x1 white AO texture used when SSAO is off (ao = 1, no change).
        let white_ao = make_pixel_tex(device, queue, [255, 255, 255, 255], false);
        let ao_bg_white = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ao-white"),
            layout: &ao_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&white_ao) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&sampler) },
            ],
        });
        let ssao = build_ssao(device, &ssao_layout, &blur_layout, &ao_layout, &globals_buf, &sampler, w.max(1), h.max(1));

        // Point-light shadow cube: 6 faces of distance-to-light (R16Float).
        let pcube_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("pcube"),
            size: wgpu::Extent3d { width: PCUBE_SIZE, height: PCUBE_SIZE, depth_or_array_layers: 6 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R16Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let pshadow_face_views: Vec<wgpu::TextureView> = (0..6)
            .map(|i| {
                pcube_tex.create_view(&wgpu::TextureViewDescriptor {
                    label: Some("pcube-face"),
                    dimension: Some(wgpu::TextureViewDimension::D2),
                    base_array_layer: i,
                    array_layer_count: Some(1),
                    ..Default::default()
                })
            })
            .collect();
        let pcube_view = pcube_tex.create_view(&wgpu::TextureViewDescriptor {
            label: Some("pcube"),
            dimension: Some(wgpu::TextureViewDimension::Cube),
            ..Default::default()
        });
        let pshadow_depth = device
            .create_texture(&wgpu::TextureDescriptor {
                label: Some("pcube-depth"),
                size: wgpu::Extent3d { width: PCUBE_SIZE, height: PCUBE_SIZE, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: DEPTH_FORMAT,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            })
            .create_view(&wgpu::TextureViewDescriptor::default());
        let pshadow_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pcube-faces"),
            size: 6 * 256,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let pshadow_g_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("pcube-face"),
            layout: &pshadow_g_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 5,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &pshadow_buf,
                    offset: 0,
                    size: Some(std::num::NonZeroU64::new(80).unwrap()),
                }),
            }],
        });
        let pshadow_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("pcube"),
            layout: &pshadow_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&pcube_view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&sampler) },
            ],
        });
        let pshadow_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("pshadow"),
            bind_group_layouts: &[&pshadow_g_layout, &obj_layout],
            push_constant_ranges: &[],
        });
        let pshadow_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("pshadow"),
            layout: Some(&pshadow_pl),
            vertex: wgpu::VertexState { module: &module, entry_point: "vs_pshadow", compilation_options: Default::default(), buffers: &[Vertex::LAYOUT] },
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: "fs_pshadow",
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState { format: wgpu::TextureFormat::R16Float, blend: None, write_mask: wgpu::ColorWrites::ALL })],
            }),
            primitive: wgpu::PrimitiveState { topology: wgpu::PrimitiveTopology::TriangleList, cull_mode: None, front_face: wgpu::FrontFace::Ccw, ..Default::default() },
            depth_stencil: Some(wgpu::DepthStencilState { format: DEPTH_FORMAT, depth_write_enabled: true, depth_compare: wgpu::CompareFunction::Less, stencil: Default::default(), bias: Default::default() }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        let globals = GlobalsU {
            view_proj: Mat4::IDENTITY.to_cols_array_2d(),
            csm_vp: [Mat4::IDENTITY.to_cols_array_2d(); 4],
            inv_view_proj: Mat4::IDENTITY.to_cols_array_2d(),
            csm_splits: [0.0; 4],
            cam_pos: [0.0, 0.0, 0.0, 1.0],
            dir_dir: Vec3::new(0.4, 1.0, 0.3).normalize().extend(1.0).into(),
            dir_color: [1.0, 1.0, 1.0, 0.15],
            fog_color: [0.0, 0.0, 0.0, 0.0],
            sky_top: [0.20, 0.40, 0.75, 1.0],
            sky_horizon: [0.70, 0.80, 0.92, 1.0],
            counts: [0.0, 1.0, 0.0, 0.0],
            screen: [w.max(1) as f32, h.max(1) as f32, 1.0, 0.0],
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
            obj_scratch: Vec::new(),
            joint_scratch: Vec::new(),
            inst_scratch: Vec::new(),
            depth,
            depth_size: (w.max(1), h.max(1)),
            sample_count,
            color_format,
            msaa_color,
            shadow_layer_views,
            csm_buf,
            shadow_pipeline,
            shadow_globals_bg,
            shadow_extent: 50.0,
            shadows_on: true,
            sky_pipeline,
            line_pipeline,
            line_buf,
            line_cap,
            line_verts: Vec::new(),
            inst_pipeline,
            inst_shadow_pipeline,
            inst_buf,
            inst_cap,
            inst_cmds: Vec::new(),
            ssao_on: false,
            prepass_pipeline,
            prepass_inst_pipeline,
            ssao_pipeline,
            blur_pipeline,
            ssao,
            ao_bg_white,
            ao_layout,
            ssao_layout,
            blur_layout,
            point_shadows_on: false,
            pshadow_pipeline,
            pshadow_g_bg,
            pshadow_buf,
            pshadow_face_views,
            pshadow_depth,
            pshadow_bg,
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
            self.ssao = build_ssao(
                device, &self.ssao_layout, &self.blur_layout, &self.ao_layout, &self.globals_buf,
                &self.sampler, w, h,
            );
            self.depth_size = (w, h);
        }
    }

    /// Toggle screen-space ambient occlusion.
    pub fn set_ssao(&mut self, on: bool) {
        self.ssao_on = on;
    }

    /// Toggle omnidirectional shadows for the first (key) point light.
    pub fn set_point_shadows(&mut self, on: bool) {
        self.point_shadows_on = on;
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
        self.inst_cmds.clear();
    }

    pub fn draw(&mut self, mesh: usize, material: usize, model: Mat4, joints: Option<Arc<Vec<Mat4>>>) {
        self.draw_tint(mesh, material, model, joints, [0.0, 0.0, 0.0]);
    }

    /// Like [`draw`] but adds a per-draw RGB `tint` OFFSET to the albedo (identity (0,0,0)).
    pub fn draw_tint(&mut self, mesh: usize, material: usize, model: Mat4, joints: Option<Arc<Vec<Mat4>>>, tint: [f32; 3]) {
        if mesh < self.meshes.len() {
            let material = if material < self.materials.len() { material } else { 0 };
            self.queue_cmds.push(DrawCmd { mesh, material, model, joints, tint, shield: [0.0, 0.0] });
        }
    }

    /// Like [`draw`] but adds an energy-shield Fresnel rim (cyan crackle, `strength` 0..1,
    /// animated by `time`). Tint stays neutral.
    pub fn draw_shield(&mut self, mesh: usize, material: usize, model: Mat4, joints: Option<Arc<Vec<Mat4>>>, strength: f32, time: f32) {
        if mesh < self.meshes.len() {
            let material = if material < self.materials.len() { material } else { 0 };
            self.queue_cmds.push(DrawCmd { mesh, material, model, joints, tint: [0.0, 0.0, 0.0], shield: [strength, time] });
        }
    }

    /// Draw `mesh`/`material` once per instance in a single instanced draw call.
    pub fn draw_instanced(&mut self, mesh: usize, material: usize, instances: Vec<InstanceRaw>) {
        if mesh < self.meshes.len() && !instances.is_empty() {
            let material = if material < self.materials.len() { material } else { 0 };
            self.inst_cmds.push((mesh, material, instances));
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
        // Cascaded shadow maps: concentric orthographic boxes of increasing size
        // centered on the camera, each crisper near and coarser far.
        let d = Vec3::new(self.globals.dir_dir[0], self.globals.dir_dir[1], self.globals.dir_dir[2])
            .normalize_or_zero();
        let center = Vec3::new(self.globals.cam_pos[0], self.globals.cam_pos[1], self.globals.cam_pos[2]);
        let up = if d.y.abs() > 0.95 { Vec3::Z } else { Vec3::Y };
        let factors = [0.12f32, 0.4, 1.0];
        let mut csm_bytes = vec![0u8; NUM_CASCADES * 256];
        for i in 0..NUM_CASCADES {
            let e = self.shadow_extent * factors[i];
            let eye = center + d * (e * 2.0);
            let vp = Mat4::orthographic_rh(-e, e, -e, e, 0.1, e * 4.5) * Mat4::look_at_rh(eye, center, up);
            self.globals.csm_vp[i] = vp.to_cols_array_2d();
            self.globals.csm_splits[i] = e;
            csm_bytes[i * 256..i * 256 + 64].copy_from_slice(bytemuck::bytes_of(&vp.to_cols_array()));
        }
        self.globals.counts[1] = if self.shadows_on { 1.0 } else { 0.0 };
        let do_pshadow = self.point_shadows_on && self.globals.counts[0] >= 1.0;
        self.globals.counts[3] = if do_pshadow { 1.0 } else { 0.0 };
        self.globals.screen = [
            self.depth_size.0 as f32,
            self.depth_size.1 as f32,
            if self.ssao_on { 1.0 } else { 0.0 },
            0.0,
        ];

        queue.write_buffer(&self.globals_buf, 0, bytemuck::bytes_of(&self.globals));
        queue.write_buffer(&self.csm_buf, 0, &csm_bytes);

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

        // Reuse last frame's scratch buffers (taken out so the loop below can
        // still borrow &self.queue_cmds); restored after the uploads.
        let mut obj_bytes = std::mem::take(&mut self.obj_scratch);
        obj_bytes.clear();
        obj_bytes.resize((self.obj_cap * stride) as usize, 0);
        let mut joint_bytes = std::mem::take(&mut self.joint_scratch);
        joint_bytes.clear();
        joint_bytes.resize((self.joint_cap * JOINT_BYTES) as usize, 0);
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
                params: [skinned_flag, cmd.tint[0], cmd.tint[1], cmd.tint[2]],
                params2: [cmd.shield[0], cmd.shield[1], 0.0, 0.0],
            };
            let off = i as u64 * stride;
            obj_bytes[off as usize..off as usize + std::mem::size_of::<ObjU>()]
                .copy_from_slice(bytemuck::bytes_of(&obj));
            offsets.push((off as u32, joint_off as u32));
        }
        queue.write_buffer(&self.obj_buf, 0, &obj_bytes);
        queue.write_buffer(&self.joint_buf, 0, &joint_bytes);
        self.obj_scratch = obj_bytes;
        self.joint_scratch = joint_bytes;

        // Flatten instanced batches into one instance buffer (reused scratch).
        let mut inst_flat: Vec<InstanceRaw> = std::mem::take(&mut self.inst_scratch);
        inst_flat.clear();
        let mut inst_ranges: Vec<(usize, usize, u32, u32)> = Vec::new();
        for (mesh, material, insts) in &self.inst_cmds {
            let start = inst_flat.len() as u32;
            inst_flat.extend_from_slice(insts);
            inst_ranges.push((*mesh, *material, start, insts.len() as u32));
        }
        if !inst_flat.is_empty() {
            let need = inst_flat.len() as u64;
            if need > self.inst_cap {
                self.inst_cap = need.next_power_of_two();
                self.inst_buf = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("instances"),
                    size: self.inst_cap * std::mem::size_of::<InstanceRaw>() as u64,
                    usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
            }
            queue.write_buffer(&self.inst_buf, 0, bytemuck::cast_slice(&inst_flat));
        }
        // The instance data now lives in the GPU buffer; the draw loop uses
        // inst_ranges, so return the scratch Vec for reuse next frame.
        self.inst_scratch = inst_flat;
        let cam = Vec3::from_slice(&self.globals.cam_pos[..3]);
        let mut order: Vec<usize> = (0..self.queue_cmds.len()).collect();
        order.sort_by(|&a, &b| {
            let ta = self.materials[self.queue_cmds[a].material].transparent;
            let tb = self.materials[self.queue_cmds[b].material].transparent;
            if ta != tb {
                return ta.cmp(&tb); // opaque (false) first
            }
            let da = (cam - self.queue_cmds[a].model.w_axis.truncate()).length_squared();
            let db = (cam - self.queue_cmds[b].model.w_axis.truncate()).length_squared();
            if ta {
                // Transparent: back-to-front so alpha blending composites correctly.
                db.partial_cmp(&da).unwrap_or(std::cmp::Ordering::Equal)
            } else {
                // Opaque: front-to-back so the GPU's early-z rejects covered fragments BEFORE the
                // PBR shader runs - kills overdraw shading (same goal as a depth pre-pass, but free
                // and with no MSAA/transparency/precision pitfalls). Final image is identical.
                da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
            }
        });

        // Camera frustum planes - reused to cull the SSAO prepass AND the main pass (the same
        // camera view). NOT used for the shadow cascades: those are rendered from the LIGHT's view,
        // where an object behind the camera can still cast a shadow into frame, so camera-culling
        // them would drop valid shadows.
        let cam_planes = frustum_planes(Mat4::from_cols_array_2d(&self.globals.view_proj));

        // Shadow pass: render scene depth into each cascade layer from its light
        // matrix (selected by dynamic offset into the per-cascade buffer). Each object is drawn
        // ONLY into the cascades its bounding sphere actually reaches (per-cascade cull below), so
        // near objects no longer get rasterized into all three concentric cascades.
        let shadow_light = Vec3::new(self.globals.dir_dir[0], self.globals.dir_dir[1], self.globals.dir_dir[2]).normalize_or_zero();
        let shadow_cam = Vec3::new(self.globals.cam_pos[0], self.globals.cam_pos[1], self.globals.cam_pos[2]);
        if self.shadows_on {
            for cascade in 0..NUM_CASCADES {
                let off = (cascade * 256) as u32;
                let e = self.globals.csm_splits[cascade];
                let mut sp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("shadow"),
                    color_attachments: &[],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &self.shadow_layer_views[cascade],
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
                sp.set_bind_group(0, &self.shadow_globals_bg, &[off]);
                for (ci, &(obj_off, joint_off)) in offsets.iter().enumerate() {
                    let cmd = &self.queue_cmds[ci];
                    // Skip objects whose shadow can't land in this cascade's footprint (kept for
                    // every cascade they DO reach, so no shadow is lost - see caster_in_cascade).
                    if self.frustum_cull {
                        let (center, radius) = cull_bounds(&cmd.model, self.mesh_radius[cmd.mesh]);
                        if !caster_in_cascade(center, radius, shadow_cam, shadow_light, e) {
                            continue;
                        }
                    }
                    let m = &self.meshes[cmd.mesh];
                    sp.set_bind_group(1, &self.obj_bg, &[obj_off, joint_off]);
                    sp.set_vertex_buffer(0, m.vbuf.slice(..));
                    sp.set_index_buffer(m.ibuf.slice(..), wgpu::IndexFormat::Uint32);
                    sp.draw_indexed(0..m.index_count, 0, 0..1);
                }
                if !inst_ranges.is_empty() {
                    sp.set_pipeline(&self.inst_shadow_pipeline);
                    sp.set_bind_group(0, &self.shadow_globals_bg, &[off]);
                    sp.set_vertex_buffer(1, self.inst_buf.slice(..));
                    for (mesh, _, start, count) in &inst_ranges {
                        let m = &self.meshes[*mesh];
                        sp.set_vertex_buffer(0, m.vbuf.slice(..));
                        sp.set_index_buffer(m.ibuf.slice(..), wgpu::IndexFormat::Uint32);
                        sp.draw_indexed(0..m.index_count, 0, *start..*start + *count);
                    }
                }
            }
        }

        // SSAO: geometry prepass (normal + distance) -> occlusion -> blur.
        if self.ssao_on {
            {
                let mut pp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("prepass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &self.ssao.prepass_color,
                        resolve_target: None,
                        ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT), store: wgpu::StoreOp::Store },
                    })],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &self.ssao.prepass_depth,
                        depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Store }),
                        stencil_ops: None,
                    }),
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });
                pp.set_pipeline(&self.prepass_pipeline);
                pp.set_bind_group(0, &self.globals_bg, &[]);
                for (ci, &(obj_off, joint_off)) in offsets.iter().enumerate() {
                    let cmd = &self.queue_cmds[ci];
                    // SSAO only affects on-screen pixels, so skip objects outside the camera
                    // frustum exactly like the main pass does (safe, no visual change).
                    if self.frustum_cull {
                        let (center, radius) = cull_bounds(&cmd.model, self.mesh_radius[cmd.mesh]);
                        if !sphere_in_frustum(&cam_planes, center, radius) {
                            continue;
                        }
                    }
                    let m = &self.meshes[cmd.mesh];
                    pp.set_bind_group(1, &self.obj_bg, &[obj_off, joint_off]);
                    pp.set_vertex_buffer(0, m.vbuf.slice(..));
                    pp.set_index_buffer(m.ibuf.slice(..), wgpu::IndexFormat::Uint32);
                    pp.draw_indexed(0..m.index_count, 0, 0..1);
                }
                if !inst_ranges.is_empty() {
                    pp.set_pipeline(&self.prepass_inst_pipeline);
                    pp.set_bind_group(0, &self.globals_bg, &[]);
                    pp.set_vertex_buffer(1, self.inst_buf.slice(..));
                    for (mesh, _, start, count) in &inst_ranges {
                        let m = &self.meshes[*mesh];
                        pp.set_vertex_buffer(0, m.vbuf.slice(..));
                        pp.set_index_buffer(m.ibuf.slice(..), wgpu::IndexFormat::Uint32);
                        pp.draw_indexed(0..m.index_count, 0, *start..*start + *count);
                    }
                }
            }
            // Occlusion + blur (fullscreen).
            for (pipe, bg, target) in [
                (&self.ssao_pipeline, &self.ssao.ssao_input_bg, &self.ssao.ssao_view),
                (&self.blur_pipeline, &self.ssao.blur_input_bg, &self.ssao.blur_view),
            ] {
                let mut fp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("ssao-fs"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: target,
                        resolve_target: None,
                        ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::WHITE), store: wgpu::StoreOp::Store },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });
                fp.set_pipeline(pipe);
                fp.set_bind_group(0, bg, &[]);
                fp.draw(0..3, 0..1);
            }
        }

        // Point-light shadow cube: render distance-to-light from light 0 across
        // the 6 cube faces.
        if do_pshadow {
            let lp = Vec3::new(
                self.globals.lights[0].pos_range[0],
                self.globals.lights[0].pos_range[1],
                self.globals.lights[0].pos_range[2],
            );
            let far = self.globals.lights[0].pos_range[3].max(1.0);
            let proj = Mat4::perspective_rh(std::f32::consts::FRAC_PI_2, 1.0, 0.05, far);
            let faces = [
                (Vec3::X, -Vec3::Y),
                (-Vec3::X, -Vec3::Y),
                (Vec3::Y, Vec3::Z),
                (-Vec3::Y, -Vec3::Z),
                (Vec3::Z, -Vec3::Y),
                (-Vec3::Z, -Vec3::Y),
            ];
            let mut pf_bytes = vec![0u8; 6 * 256];
            for (i, (dir, up)) in faces.iter().enumerate() {
                let vp = proj * Mat4::look_at_rh(lp, lp + *dir, *up);
                pf_bytes[i * 256..i * 256 + 64].copy_from_slice(bytemuck::bytes_of(&vp.to_cols_array()));
                pf_bytes[i * 256 + 64..i * 256 + 80]
                    .copy_from_slice(bytemuck::bytes_of(&[lp.x, lp.y, lp.z, 1.0f32]));
            }
            queue.write_buffer(&self.pshadow_buf, 0, &pf_bytes);
            for face in 0..6 {
                let off = (face * 256) as u32;
                let mut fp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("pshadow"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &self.pshadow_face_views[face],
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color { r: far as f64, g: far as f64, b: far as f64, a: far as f64 }),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &self.pshadow_depth,
                        depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Store }),
                        stencil_ops: None,
                    }),
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });
                fp.set_pipeline(&self.pshadow_pipeline);
                fp.set_bind_group(0, &self.pshadow_g_bg, &[off]);
                for (ci, &(obj_off, joint_off)) in offsets.iter().enumerate() {
                    let m = &self.meshes[self.queue_cmds[ci].mesh];
                    fp.set_bind_group(1, &self.obj_bg, &[obj_off, joint_off]);
                    fp.set_vertex_buffer(0, m.vbuf.slice(..));
                    fp.set_index_buffer(m.ibuf.slice(..), wgpu::IndexFormat::Uint32);
                    fp.draw_indexed(0..m.index_count, 0, 0..1);
                }
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
        let ao_bg = if self.ssao_on { &self.ssao.ao_bg } else { &self.ao_bg_white };
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.globals_bg, &[]);
        pass.set_bind_group(3, ao_bg, &[]);
        pass.set_bind_group(4, &self.pshadow_bg, &[]);
        let mut drawn = 0usize;
        for &ci in &order {
            let cmd = &self.queue_cmds[ci];
            // Frustum cull by the mesh's bounding sphere (scaled by the model).
            if self.frustum_cull {
                let (center, radius) = cull_bounds(&cmd.model, self.mesh_radius[cmd.mesh]);
                if !sphere_in_frustum(&cam_planes, center, radius) {
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

        // Instanced batches (one draw call each), lit + shadowed like the rest.
        if !inst_ranges.is_empty() {
            pass.set_pipeline(&self.inst_pipeline);
            pass.set_bind_group(0, &self.globals_bg, &[]);
            // group1 (object) is unused by the instanced shaders but must be bound
            // to satisfy the layout; the ring's slot 0 is a valid dummy.
            pass.set_bind_group(1, &self.obj_bg, &[0, 0]);
            pass.set_bind_group(3, ao_bg, &[]);
            pass.set_bind_group(4, &self.pshadow_bg, &[]);
            pass.set_vertex_buffer(1, self.inst_buf.slice(..));
            for (mesh, material, start, count) in &inst_ranges {
                let m = &self.meshes[*mesh];
                pass.set_bind_group(2, &self.materials[*material].bind_group, &[]);
                pass.set_vertex_buffer(0, m.vbuf.slice(..));
                pass.set_index_buffer(m.ibuf.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..m.index_count, 0, *start..*start + *count);
            }
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
/// Whether a caster (bounding sphere `center`/`radius`) can cast into a camera-centred cascade of
/// footprint half-width `e`. The cascades are concentric squares centred on the camera, and the
/// shader selects a fragment's cascade by its distance from the camera - so a caster only matters
/// to cascade i if its distance PERPENDICULAR to the light is within that footprint. That
/// perpendicular distance is invariant along the light direction, so a caster occluding any
/// in-cascade fragment is always kept (no shadow is ever dropped); only genuinely out-of-footprint
/// casters are skipped. `light` must be normalized. Square footprint -> circumscribed radius e*sqrt2.
fn caster_in_cascade(center: Vec3, radius: f32, cam: Vec3, light: Vec3, e: f32) -> bool {
    let off = center - cam;
    let perp = off - light * off.dot(light);
    perp.length() <= e * std::f32::consts::SQRT_2 + radius
}

/// Bounding-sphere center + world-space radius for a draw command's model, used by every pass's
/// frustum cull (single source of truth so the camera and main passes test identical bounds).
fn cull_bounds(model: &Mat4, mesh_radius: f32) -> (Vec3, f32) {
    let center = model.w_axis.truncate();
    let scale = model.x_axis.truncate().length()
        .max(model.y_axis.truncate().length())
        .max(model.z_axis.truncate().length());
    (center, mesh_radius * scale)
}

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

/// Size-dependent SSAO resources (recreated on resize).
struct Ssao {
    prepass_color: wgpu::TextureView,
    prepass_depth: wgpu::TextureView,
    ssao_view: wgpu::TextureView,
    blur_view: wgpu::TextureView,
    ssao_input_bg: wgpu::BindGroup,
    blur_input_bg: wgpu::BindGroup,
    ao_bg: wgpu::BindGroup,
}

#[allow(clippy::too_many_arguments)]
fn build_ssao(
    device: &wgpu::Device,
    ssao_layout: &wgpu::BindGroupLayout,
    blur_layout: &wgpu::BindGroupLayout,
    ao_layout: &wgpu::BindGroupLayout,
    globals_buf: &wgpu::Buffer,
    sampler: &wgpu::Sampler,
    w: u32,
    h: u32,
) -> Ssao {
    let target = |format: wgpu::TextureFormat, label: &str| {
        device
            .create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            })
            .create_view(&wgpu::TextureViewDescriptor::default())
    };
    let prepass_color = target(wgpu::TextureFormat::Rgba16Float, "prepass");
    let prepass_depth = target(DEPTH_FORMAT, "prepass-depth");
    let ssao_view = target(wgpu::TextureFormat::R8Unorm, "ssao");
    let blur_view = target(wgpu::TextureFormat::R8Unorm, "ssao-blur");
    let ssao_input_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("ssao-in"),
        layout: ssao_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: globals_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&prepass_color) },
            wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(sampler) },
        ],
    });
    let blur_input_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("blur-in"),
        layout: blur_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&ssao_view) },
            wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(sampler) },
        ],
    });
    let ao_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("ao"),
        layout: ao_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&blur_view) },
            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(sampler) },
        ],
    });
    Ssao { prepass_color, prepass_depth, ssao_view, blur_view, ssao_input_bg, blur_input_bg, ao_bg }
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
    csm_vp: array<mat4x4<f32>, 4>,
    inv_view_proj: mat4x4<f32>,
    csm_splits: vec4<f32>,
    cam_pos: vec4<f32>,
    dir_dir: vec4<f32>,
    dir_color: vec4<f32>,
    fog_color: vec4<f32>,
    sky_top: vec4<f32>,
    sky_horizon: vec4<f32>,
    counts: vec4<f32>,
    screen: vec4<f32>,
    lights: array<PointLight, 16>,
};
@group(0) @binding(0) var<uniform> g: Globals;
@group(0) @binding(1) var shadow_map: texture_depth_2d_array;
@group(0) @binding(2) var shadow_samp: sampler_comparison;
// The current cascade's light matrix during the shadow pass (binding 3 so it
// doesn't collide with `g`; pruned from pipelines that don't use it).
struct CascadeU { vp: mat4x4<f32> };
@group(0) @binding(3) var<uniform> csm: CascadeU;

struct ObjU { model: mat4x4<f32>, normal_mat: mat4x4<f32>, params: vec4<f32>, params2: vec4<f32> };
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
@group(3) @binding(0) var ao_tex: texture_2d<f32>;
@group(3) @binding(1) var ao_samp: sampler;
@group(4) @binding(0) var pcube: texture_cube<f32>;
@group(4) @binding(1) var pcube_samp: sampler;
// The current cube face's matrix + light position (point-shadow pass, binding 5).
struct PFace { vp: mat4x4<f32>, light_pos: vec4<f32> };
@group(0) @binding(5) var<uniform> pface: PFace;

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
// The sky/environment radiance in a direction (gradient + sun). Used to draw the
// skybox AND as the image-based lighting environment.
fn sky_color(dir: vec3<f32>) -> vec3<f32> {
    let t = clamp(dir.y * 0.5 + 0.5, 0.0, 1.0);
    var col = mix(g.sky_horizon.rgb, g.sky_top.rgb, pow(t, 0.6));
    let sun = max(dot(dir, normalize(g.dir_dir.xyz)), 0.0);
    col = col + g.dir_color.rgb * pow(sun, 600.0) * 3.0; // sun disk
    col = col + g.dir_color.rgb * pow(sun, 8.0) * 0.15;  // glow
    return col;
}

@fragment
fn fs_sky(in: SkyOut) -> @location(0) vec4<f32> {
    let far = g.inv_view_proj * vec4<f32>(in.ndc, 1.0, 1.0);
    let world = far.xyz / far.w;
    let dir = normalize(world - g.cam_pos.xyz);
    return vec4<f32>(sky_color(dir), 1.0);
}

@vertex
fn vs_shadow(in: VsIn) -> @builtin(position) vec4<f32> {
    var local = vec4<f32>(in.pos, 1.0);
    if (obj.params.x > 0.5) {
        let skin = joints.m[in.j.x] * in.w.x + joints.m[in.j.y] * in.w.y
                 + joints.m[in.j.z] * in.w.z + joints.m[in.j.w] * in.w.w;
        local = skin * local;
    }
    return csm.vp * (obj.model * local);
}

// Percentage-closer-filtered cascaded shadow factor (1 = lit, 0 = shadowed).
fn shadow_factor(world_pos: vec3<f32>, ndl: f32) -> f32 {
    if (g.counts.y < 0.5) { return 1.0; }
    // Select the cascade by distance from the camera.
    let dist = length(world_pos - g.cam_pos.xyz);
    var ci = 0;
    if (dist > g.csm_splits.x) { ci = 1; }
    if (dist > g.csm_splits.y) { ci = 2; }
    let lc = g.csm_vp[ci] * vec4<f32>(world_pos, 1.0);
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
            sum = sum + textureSampleCompare(shadow_map, shadow_samp, uv + o, ci, proj.z - bias);
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

// Omnidirectional shadow for the key point light: compare the fragment's
// distance to the light against the nearest occluder distance in the cube map.
fn point_shadow(world_pos: vec3<f32>, light_pos: vec3<f32>) -> f32 {
    let dir = world_pos - light_pos;
    let cur = length(dir);
    let stored = textureSample(pcube, pcube_samp, dir).r;
    let bias = 0.08 + cur * 0.02;
    if (cur - bias > stored) { return 0.0; }
    return 1.0;
}

// The point-shadow pass: output distance-to-light into the cube face.
@vertex
fn vs_pshadow(in: VsIn) -> VsOut {
    var local = vec4<f32>(in.pos, 1.0);
    if (obj.params.x > 0.5) {
        let skin = joints.m[in.j.x] * in.w.x + joints.m[in.j.y] * in.w.y
                 + joints.m[in.j.z] * in.w.z + joints.m[in.j.w] * in.w.w;
        local = skin * local;
    }
    let world = obj.model * local;
    var out: VsOut;
    out.clip = pface.vp * world;
    out.world_pos = world.xyz;
    out.world_normal = vec3<f32>(0.0, 1.0, 0.0);
    out.uv = vec2<f32>(0.0);
    out.world_tangent = vec3<f32>(1.0, 0.0, 0.0);
    out.tangent_w = 1.0;
    return out;
}
@fragment
fn fs_pshadow(in: VsOut) -> @location(0) vec4<f32> {
    let dist = length(in.world_pos - pface.light_pos.xyz);
    return vec4<f32>(dist, dist, dist, dist);
}

// Shared lighting: PBR direct (directional + shadow + point lights) + ambient +
// emissive + fog. Used by both the per-object and instanced fragment stages.
fn shade(world_pos: vec3<f32>, n_in: vec3<f32>, albedo: vec3<f32>, alpha: f32, metallic: f32, rough: f32, emissive: vec3<f32>, ao: f32) -> vec4<f32> {
    let n = normalize(n_in);
    let v = normalize(g.cam_pos.xyz - world_pos);
    let ndl = max(dot(n, normalize(g.dir_dir.xyz)), 0.0);
    let shadow = shadow_factor(world_pos, ndl);
    var lo = brdf(n, v, normalize(g.dir_dir.xyz), g.dir_color.rgb * g.dir_dir.w, albedo, metallic, rough) * shadow;
    let count = i32(g.counts.x);
    for (var i = 0; i < count; i = i + 1) {
        let lp = g.lights[i].pos_range.xyz;
        let range = g.lights[i].pos_range.w;
        let to = lp - world_pos;
        let d = length(to);
        let l = to / max(d, 1e-4);
        let att = clamp(1.0 - (d / range), 0.0, 1.0);
        var psh = 1.0;
        if (i == 0 && g.counts.w > 0.5) { psh = point_shadow(world_pos, lp); }
        let radiance = g.lights[i].color_int.rgb * g.lights[i].color_int.w * att * att * psh;
        lo = lo + brdf(n, v, l, radiance, albedo, metallic, rough);
    }
    // Image-based ambient + specular reflection from the sky environment.
    let amb = g.dir_color.w;
    let f0 = mix(vec3<f32>(0.04), albedo, metallic);
    let ndv = max(dot(n, v), 0.0);
    let irradiance = sky_color(n) * albedo * (1.0 - metallic);
    let env = sky_color(reflect(-v, n));
    let fr = f0 + (max(vec3<f32>(1.0 - rough), f0) - f0) * pow(1.0 - ndv, 5.0);
    let ambient = (irradiance + env * fr) * amb * ao;
    var color = lo + ambient + emissive;
    if (g.fog_color.w > 0.0) {
        let f = clamp(exp(-length(g.cam_pos.xyz - world_pos) * g.fog_color.w), 0.0, 1.0);
        color = mix(g.fog_color.rgb, color, f);
    }
    return vec4<f32>(color, alpha);
}

@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
    // params.yzw is an additive colour OFFSET (not a multiply): it shifts the albedo
    // toward a hue without crushing the other channels to black. Identity is (0,0,0).
    var albedo = vec4<f32>(max(mat.base_color.rgb + vec3<f32>(obj.params.y, obj.params.z, obj.params.w), vec3<f32>(0.0)), mat.base_color.a);
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
    var emissive = mat.emissive.rgb;
    if (mat.flags.w > 0.5) { emissive = emissive + textureSample(emissive_tex, samp, in.uv).rgb; }
    // Energy-shield FIELD: a cyan shell enveloping the whole body (base sheen + Fresnel rim +
    // crackle + scrolling scan bands), not just a thin edge. params2.x = strength, .y = time.
    let shield_str = obj.params2.x;
    if (shield_str > 0.001) {
        let vdir = normalize(g.cam_pos.xyz - in.world_pos);
        let ndotv = max(dot(n, vdir), 0.0);
        // Softer Fresnel (pow 1.8 not 2.5) so the glow spreads off the silhouette, PLUS a base
        // sheen over the whole surface -> an energy FIELD around them, not a hairline rim.
        let fres = pow(1.0 - ndotv, 1.8);
        let tm = obj.params2.y;
        let crackle = 0.5 + 0.5 * sin(in.world_pos.y * 24.0 + tm * 10.0) * sin(in.world_pos.x * 18.0 - tm * 7.0);
        let scan = 0.5 + 0.5 * sin(in.world_pos.y * 9.0 - tm * 3.5);   // scrolling energy bands
        let pulse = 0.8 + 0.2 * sin(tm * 4.0);
        // base fill (0.45) lights the whole body; the Fresnel term brightens the rim on top.
        let field = (0.45 + 1.0 * fres) * (0.65 + 0.35 * crackle) * (0.75 + 0.25 * scan) * pulse;
        emissive = emissive + vec3<f32>(0.3, 0.85, 1.0) * field * shield_str * 4.5;
    }
    let ao = textureSample(ao_tex, ao_samp, in.clip.xy / g.screen.xy).r;
    return shade(in.world_pos, n, albedo.rgb, albedo.a, metallic, rough, emissive, ao);
}

// Geometry prepass: world normal (rgb) + distance-from-camera (a) for SSAO.
@fragment
fn fs_prepass(in: VsOut) -> @location(0) vec4<f32> {
    return vec4<f32>(normalize(in.world_normal), length(g.cam_pos.xyz - in.world_pos));
}

// --- instanced (per-instance model matrix + tint) ---
struct InstIn {
    @location(0) pos: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) j: vec4<u32>,
    @location(4) w: vec4<f32>,
    @location(5) tangent: vec4<f32>,
    @location(6) m0: vec4<f32>,
    @location(7) m1: vec4<f32>,
    @location(8) m2: vec4<f32>,
    @location(9) m3: vec4<f32>,
    @location(10) tint: vec4<f32>,
};
struct InstOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) world_normal: vec3<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) world_pos: vec3<f32>,
    @location(3) tint: vec4<f32>,
};
@vertex
fn vs_inst(in: InstIn) -> InstOut {
    let model = mat4x4<f32>(in.m0, in.m1, in.m2, in.m3);
    let world = model * vec4<f32>(in.pos, 1.0);
    var out: InstOut;
    out.clip = g.view_proj * world;
    out.world_normal = (model * vec4<f32>(in.normal, 0.0)).xyz;
    out.uv = in.uv;
    out.world_pos = world.xyz;
    out.tint = in.tint;
    return out;
}
@fragment
fn fs_inst(in: InstOut) -> @location(0) vec4<f32> {
    var albedo = mat.base_color * in.tint;
    if (mat.flags.x > 0.5) { albedo = albedo * textureSample(base_tex, samp, in.uv); }
    let rough = clamp(mat.mr.y, 0.04, 1.0);
    let ao = textureSample(ao_tex, ao_samp, in.clip.xy / g.screen.xy).r;
    return shade(in.world_pos, in.world_normal, albedo.rgb, albedo.a, mat.mr.x, rough, mat.emissive.rgb, ao);
}
@fragment
fn fs_prepass_inst(in: InstOut) -> @location(0) vec4<f32> {
    return vec4<f32>(normalize(in.world_normal), length(g.cam_pos.xyz - in.world_pos));
}
@vertex
fn vs_shadow_inst(in: InstIn) -> @builtin(position) vec4<f32> {
    let model = mat4x4<f32>(in.m0, in.m1, in.m2, in.m3);
    return csm.vp * (model * vec4<f32>(in.pos, 1.0));
}
"#;

// Screen-space ambient occlusion: hemisphere-kernel SSAO from the normal+depth
// prepass, then a box blur. Separate module (fullscreen passes).
const SSAO_WGSL: &str = r#"
struct PointLight { pos_range: vec4<f32>, color_int: vec4<f32> };
struct Globals {
    view_proj: mat4x4<f32>,
    csm_vp: array<mat4x4<f32>, 4>,
    inv_view_proj: mat4x4<f32>,
    csm_splits: vec4<f32>,
    cam_pos: vec4<f32>,
    dir_dir: vec4<f32>,
    dir_color: vec4<f32>,
    fog_color: vec4<f32>,
    sky_top: vec4<f32>,
    sky_horizon: vec4<f32>,
    counts: vec4<f32>,
    screen: vec4<f32>,
    lights: array<PointLight, 16>,
};
@group(0) @binding(0) var<uniform> g: Globals;
@group(0) @binding(1) var src: texture_2d<f32>;
@group(0) @binding(2) var samp: sampler;

struct FsOut { @builtin(position) clip: vec4<f32>, @location(0) uv: vec2<f32> };
@vertex
fn vs_fs(@builtin(vertex_index) i: u32) -> FsOut {
    var p = array<vec2<f32>, 3>(vec2(-1.0, -1.0), vec2(3.0, -1.0), vec2(-1.0, 3.0));
    var o: FsOut;
    let xy = p[i];
    o.clip = vec4<f32>(xy, 0.0, 1.0);
    o.uv = vec2<f32>(xy.x * 0.5 + 0.5, 1.0 - (xy.y * 0.5 + 0.5));
    return o;
}

var<private> KERNEL: array<vec3<f32>, 16> = array<vec3<f32>, 16>(
    vec3<f32>(0.05, 0.04, 0.06), vec3<f32>(-0.10, 0.08, 0.10), vec3<f32>(0.12, -0.09, 0.14),
    vec3<f32>(-0.15, -0.12, 0.10), vec3<f32>(0.18, 0.14, 0.20), vec3<f32>(-0.06, 0.22, 0.16),
    vec3<f32>(0.24, -0.06, 0.26), vec3<f32>(-0.28, 0.10, 0.20), vec3<f32>(0.10, 0.30, 0.30),
    vec3<f32>(-0.20, -0.26, 0.28), vec3<f32>(0.34, 0.18, 0.10), vec3<f32>(-0.12, 0.36, 0.34),
    vec3<f32>(0.40, -0.20, 0.30), vec3<f32>(-0.42, -0.10, 0.40), vec3<f32>(0.22, 0.44, 0.40),
    vec3<f32>(-0.30, 0.30, 0.50),
);

@fragment
fn fs_ssao(in: FsOut) -> @location(0) vec4<f32> {
    let data = textureSample(src, samp, in.uv);
    let n = data.xyz;
    let dist = data.w;
    if (dist <= 0.0) { return vec4<f32>(1.0); } // background = unoccluded
    // Reconstruct the world position of this pixel.
    let ndc = vec2<f32>(in.uv.x * 2.0 - 1.0, (1.0 - in.uv.y) * 2.0 - 1.0);
    let far = g.inv_view_proj * vec4<f32>(ndc, 1.0, 1.0);
    let ray = normalize(far.xyz / far.w - g.cam_pos.xyz);
    let pos = g.cam_pos.xyz + ray * dist;
    // A randomized tangent frame to rotate the kernel per pixel.
    let rnd = fract(sin(dot(in.uv, vec2<f32>(12.9898, 78.233))) * 43758.5453) * 6.2831853;
    let randv = normalize(vec3<f32>(cos(rnd), sin(rnd), 0.0));
    let t = normalize(randv - n * dot(randv, n));
    let b = cross(n, t);
    let tbn = mat3x3<f32>(t, b, n);
    let radius = 0.7;
    var occ = 0.0;
    for (var i = 0; i < 16; i = i + 1) {
        let sp = pos + (tbn * KERNEL[i]) * radius;
        let clip = g.view_proj * vec4<f32>(sp, 1.0);
        let sndc = clip.xyz / clip.w;
        let suv = vec2<f32>(sndc.x * 0.5 + 0.5, 1.0 - (sndc.y * 0.5 + 0.5));
        if (suv.x < 0.0 || suv.x > 1.0 || suv.y < 0.0 || suv.y > 1.0) { continue; }
        let sdist = textureSample(src, samp, suv).w;
        let sample_dist = length(sp - g.cam_pos.xyz);
        if (sdist > 0.0 && sdist < sample_dist - 0.02) {
            occ = occ + smoothstep(0.0, 1.0, radius / max(abs(dist - sdist), 1e-3));
        }
    }
    let ao = clamp(1.0 - occ / 16.0, 0.0, 1.0);
    return vec4<f32>(ao, ao, ao, 1.0);
}

@fragment
fn fs_blur(in: FsOut) -> @location(0) vec4<f32> {
    let dim = vec2<f32>(textureDimensions(src));
    let texel = 1.0 / dim;
    var sum = 0.0;
    for (var y = -2; y <= 1; y = y + 1) {
        for (var x = -2; x <= 1; x = x + 1) {
            sum = sum + textureSample(src, samp, in.uv + vec2<f32>(f32(x), f32(y)) * texel).r;
        }
    }
    let v = sum / 16.0;
    return vec4<f32>(v, v, v, 1.0);
}
"#;
