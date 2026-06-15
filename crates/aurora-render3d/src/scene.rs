//! A high-level scene: a registry of drawable models (file-loaded or primitive),
//! per-model animation players, and a camera, on top of [`Renderer3D`]. This is
//! the surface the engine/runtime drives; it owns no device and borrows one per
//! call so the same scene renders offscreen or to the window.

use glam::{Mat4, Vec3};

use crate::anim::AnimPlayer;
use crate::mesh::MeshData;
use crate::model::Model;
use crate::render::{MaterialDesc, Renderer3D};

/// One drawable: a set of (mesh, material) primitives, with an optional skeleton
/// and animation player when it came from an animated model.
struct Renderable {
    prims: Vec<(usize, usize)>, // (mesh id, material id) in the renderer
    model: Option<Model>,
    player: AnimPlayer,
    skinned: bool,
}

struct Camera {
    eye: Vec3,
    target: Vec3,
    up: Vec3,
    fov_y: f32,
    near: f32,
    far: f32,
}

pub struct Scene {
    pub renderer: Renderer3D,
    items: Vec<Renderable>,
    cam: Camera,
    size: (u32, u32),
    clear: [f32; 4],
}

impl Scene {
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        format: wgpu::TextureFormat,
        w: u32,
        h: u32,
        samples: u32,
    ) -> Scene {
        let mut s = Scene {
            renderer: Renderer3D::new(device, queue, format, w, h, samples),
            items: Vec::new(),
            cam: Camera {
                eye: Vec3::new(0.0, 2.0, 6.0),
                target: Vec3::ZERO,
                up: Vec3::Y,
                fov_y: 60f32.to_radians(),
                near: 0.05,
                far: 500.0,
            },
            size: (w.max(1), h.max(1)),
            clear: [0.05, 0.06, 0.09, 1.0],
        };
        s.update_camera();
        s.renderer.set_light(Vec3::new(0.4, 1.0, 0.3), Vec3::ONE, 0.25);
        s
    }

    pub fn resize(&mut self, device: &wgpu::Device, w: u32, h: u32) {
        if w > 0 && h > 0 {
            self.size = (w, h);
            self.renderer.resize(device, w, h);
            self.update_camera();
        }
    }

    fn update_camera(&mut self) {
        let aspect = self.size.0 as f32 / self.size.1.max(1) as f32;
        let proj = crate::perspective(self.cam.fov_y, aspect, self.cam.near, self.cam.far);
        let view = crate::look_at(self.cam.eye, self.cam.target, self.cam.up);
        self.renderer.set_camera(proj * view, self.cam.eye);
    }

    pub fn set_camera(&mut self, eye: Vec3, target: Vec3, fov_deg: f32) {
        self.cam.eye = eye;
        self.cam.target = target;
        self.cam.fov_y = fov_deg.to_radians().clamp(0.05, std::f32::consts::PI - 0.05);
        self.update_camera();
    }

    pub fn set_light(&mut self, dir: Vec3, color: Vec3, ambient: f32) {
        self.renderer.set_light(dir, color, ambient);
    }

    pub fn set_fog(&mut self, color: Vec3, density: f32) {
        self.renderer.set_fog(color, density);
    }
    pub fn set_sky(&mut self, on: bool, top: Vec3, horizon: Vec3) {
        self.renderer.set_sky(on, top, horizon);
    }
    pub fn set_shadows(&mut self, on: bool) {
        self.renderer.set_shadows(on);
    }
    pub fn clear_point_lights(&mut self) {
        self.renderer.clear_point_lights();
    }
    pub fn add_point_light(&mut self, pos: Vec3, color: Vec3, range: f32, intensity: f32) {
        self.renderer.add_point_light(pos, color, range, intensity);
    }

    pub fn set_clear(&mut self, r: f32, g: f32, b: f32) {
        self.clear = [r, g, b, 1.0];
    }

    /// Load a model file (glTF/GLB/OBJ). Returns a handle or -1 on failure.
    pub fn load_model(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, path: &str) -> i64 {
        let model = match Model::load(path) {
            Ok(m) => m,
            Err(e) => {
                eprintln!("aurora: {e}");
                return -1;
            }
        };
        let mut prims = Vec::new();
        let mut skinned = false;
        for p in &model.primitives {
            let mesh = self.renderer.add_mesh(device, &p.mesh);
            let desc = MaterialDesc {
                base_color: p.base_color,
                metallic: p.metallic,
                roughness: p.roughness,
                emissive: p.emissive,
                base_tex: p.texture.as_ref().map(|(px, w, h)| (px.as_slice(), *w, *h)),
                normal_tex: p.normal_tex.as_ref().map(|(px, w, h)| (px.as_slice(), *w, *h)),
                mr_tex: p.mr_tex.as_ref().map(|(px, w, h)| (px.as_slice(), *w, *h)),
                emissive_tex: p.emissive_tex.as_ref().map(|(px, w, h)| (px.as_slice(), *w, *h)),
            };
            let mat = self.renderer.add_material(device, queue, &desc);
            prims.push((mesh, mat));
            skinned |= p.skinned;
        }
        self.items.push(Renderable { prims, model: Some(model), player: AnimPlayer::new(), skinned });
        (self.items.len() - 1) as i64
    }

    /// Register a primitive mesh with a flat color. Returns a handle.
    pub fn add_primitive(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        mesh: &MeshData,
        color: [f32; 4],
    ) -> i64 {
        let m = self.renderer.add_mesh(device, mesh);
        let mat = self.renderer.add_material(device, queue, &MaterialDesc::flat(color));
        self.items.push(Renderable {
            prims: vec![(m, mat)],
            model: None,
            player: AnimPlayer::new(),
            skinned: false,
        });
        (self.items.len() - 1) as i64
    }

    pub fn make_box(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, color: [f32; 4]) -> i64 {
        self.add_primitive(device, queue, &MeshData::cube(), color)
    }
    pub fn make_sphere(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        segments: u32,
        color: [f32; 4],
    ) -> i64 {
        self.add_primitive(device, queue, &MeshData::sphere(1.0, segments), color)
    }
    pub fn make_plane(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        size: f32,
        tiles: f32,
        color: [f32; 4],
    ) -> i64 {
        self.add_primitive(device, queue, &MeshData::plane(size, tiles), color)
    }

    /// Project a world point to framebuffer pixel coords (origin top-left), or
    /// `None` if it is behind the camera.
    pub fn world_to_screen(&self, p: Vec3) -> Option<(f32, f32)> {
        let clip = self.renderer.view_proj() * p.extend(1.0);
        if clip.w <= 0.0001 {
            return None;
        }
        let ndc = clip.truncate() / clip.w;
        let x = (ndc.x * 0.5 + 0.5) * self.size.0 as f32;
        let y = (1.0 - (ndc.y * 0.5 + 0.5)) * self.size.1 as f32;
        Some((x, y))
    }

    /// A camera-facing sprite: a quad with an unlit (emissive) color. Draw it
    /// with `draw_billboard`. Good for particles, muzzle flashes, and markers.
    pub fn make_sprite(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, color: [f32; 3]) -> i64 {
        let m = self.renderer.add_mesh(device, &MeshData::quad());
        let desc = MaterialDesc {
            base_color: [0.0, 0.0, 0.0, 1.0],
            metallic: 0.0,
            roughness: 1.0,
            emissive: color,
            base_tex: None,
            normal_tex: None,
            mr_tex: None,
            emissive_tex: None,
        };
        let mat = self.renderer.add_material(device, queue, &desc);
        self.items.push(Renderable {
            prims: vec![(m, mat)],
            model: None,
            player: AnimPlayer::new(),
            skinned: false,
        });
        (self.items.len() - 1) as i64
    }

    /// Draw a sprite handle as a camera-facing billboard of side `size` at `pos`.
    pub fn draw_billboard(&mut self, handle: i64, pos: Vec3, size: f32) {
        let to_cam = (self.cam.eye - pos).normalize_or_zero();
        let mut right = Vec3::Y.cross(to_cam);
        if right.length_squared() < 1e-6 {
            right = Vec3::X;
        }
        right = right.normalize();
        let up = to_cam.cross(right);
        let model = Mat4::from_cols(
            (right * size).extend(0.0),
            (up * size).extend(0.0),
            to_cam.extend(0.0),
            pos.extend(1.0),
        );
        self.draw(handle, model);
    }

    /// Draw a handle once per transform (batched instancing).
    pub fn draw_instances(&mut self, handle: i64, transforms: &[Mat4]) {
        for &t in transforms {
            self.draw(handle, t);
        }
    }

    /// Number of animation clips on a model handle.
    pub fn clip_count(&self, handle: i64) -> i64 {
        self.item(handle).and_then(|r| r.model.as_ref()).map(|m| m.clips.len() as i64).unwrap_or(0)
    }

    /// Start (or crossfade to) an animation clip on a model handle, blending from
    /// the current pose over `fade` seconds (0 = instant).
    pub fn anim_play(&mut self, handle: i64, clip: i64, looping: bool, speed: f32, fade: f32) {
        if let Some(r) = self.item_mut(handle) {
            r.player.play(clip.max(0) as usize, looping, speed, fade);
        }
    }

    /// Advance a model's current animation by `dt` seconds.
    pub fn anim_update(&mut self, handle: i64, dt: f32) {
        // Split borrow: take the model out by reference for sampling.
        if let Some(r) = self.item_mut(handle) {
            if let Some(model) = &r.model {
                r.player.advance(model, dt);
            }
        }
    }

    pub fn begin(&mut self) {
        self.renderer.begin();
    }

    /// Queue a model for drawing at `transform`.
    pub fn draw(&mut self, handle: i64, transform: Mat4) {
        let idx = match self.resolve(handle) {
            Some(i) => i,
            None => return,
        };
        // Compute skinning matrices once if needed.
        let joints = {
            let r = &self.items[idx];
            if r.skinned {
                r.model.as_ref().map(|m| r.player.matrices(m))
            } else {
                None
            }
        };
        let prims = self.items[idx].prims.clone();
        for (mesh, mat) in prims {
            let j = joints.clone().filter(|v| !v.is_empty());
            self.renderer.draw(mesh, mat, transform, j);
        }
    }

    pub fn render(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
    ) {
        self.renderer.render(device, queue, encoder, view, self.clear);
    }

    fn resolve(&self, handle: i64) -> Option<usize> {
        let i = handle as usize;
        if handle >= 0 && i < self.items.len() {
            Some(i)
        } else {
            None
        }
    }
    fn item(&self, handle: i64) -> Option<&Renderable> {
        self.resolve(handle).map(|i| &self.items[i])
    }
    fn item_mut(&mut self, handle: i64) -> Option<&mut Renderable> {
        self.resolve(handle).map(|i| &mut self.items[i])
    }
}
