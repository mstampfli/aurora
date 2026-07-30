//! A high-level scene: a registry of drawable models (file-loaded or primitive),
//! per-model animation players, and a camera, on top of [`Renderer3D`]. This is
//! the surface the engine/runtime drives; it owns no device and borrows one per
//! call so the same scene renders offscreen or to the window.

use std::sync::Arc;

use glam::{Mat4, Vec3};

use crate::anim::AnimPlayer;
use crate::mesh::MeshData;
use crate::model::Model;
use crate::render::{MaterialDesc, MaterialId, MeshId, Renderer3D};
use aurora_slot::{Key, SlotMap};

/// Resolve a name against a model's clip or joint names, returning an index or -1.
///
/// The rule, in order: an exact (case-insensitive) match, then a match on the
/// segment after the last `|`. Exporters prefix clips with the armature they came
/// from (`CharacterArmature|Walk`), and that prefix is an export setting rather
/// than authored intent, so a game asking for `"Walk"` must find it. Exact wins
/// over suffix so a model that really does have a clip named `Walk` is never
/// beaten by some other armature's `Rig|Walk`.
///
/// Split out from [`Scene::clip_index`] / [`Scene::joint_index`] so the rule is
/// testable without a GPU, and so clips and joints cannot drift to two rules.
fn match_name<'a, I>(names: I, want: &str) -> i64
where
    I: Iterator<Item = &'a str> + Clone,
{
    let want = want.trim();
    for (i, n) in names.clone().enumerate() {
        if n.eq_ignore_ascii_case(want) {
            return i as i64;
        }
    }
    for (i, n) in names.enumerate() {
        let tail = n.rsplit('|').next().unwrap_or(n);
        if tail.eq_ignore_ascii_case(want) {
            return i as i64;
        }
    }
    -1
}

/// The heavy half of a drawable: uploaded GPU meshes and materials, plus the parsed
/// model (skeleton, clips, CPU mesh data). SHARED between every handle that loaded the
/// same file, and reference-counted so the last handle to go frees the GPU memory.
///
/// This split exists because animation state is per-handle but GPU data is not, and
/// conflating them made loading the same file N times cost N uploads. A horde of 24
/// bodies over 5 distinct models needs 24 animation players and 5 uploads; before this
/// it did 24 uploads and spent about 4.7 GB of VRAM to do it, which is more than a
/// second copy of the game could fit on an 8 GB card.
struct Asset {
    prims: Vec<(MeshId, MaterialId)>,
    model: Option<Model>,
    skinned: bool,
    /// The file this came from, and the key it is cached under. `None` for a primitive
    /// built in code, which is cheap and never shared.
    path: Option<String>,
}

/// One drawable: a shared [`Asset`] plus the state that must NOT be shared - where this
/// particular body is in its animation, and which of its joints are hidden.
struct Renderable {
    asset: Arc<Asset>,
    player: AnimPlayer,
    /// Bitmask of skin joints to HIDE: their skinning matrix is zeroed before drawing, so
    /// geometry weighted to them collapses to the model origin (used for first-person arms -
    /// hide the torso/head/legs so only the arms render). Bit i = joint i. 0 = show all.
    hidden_joints: u64,
}

struct Camera {
    eye: Vec3,
    target: Vec3,
    up: Vec3,
    /// View roll about the forward axis, in radians (camera banking).
    roll: f32,
    fov_y: f32,
    near: f32,
    far: f32,
}

/// Internal handle to a scene item. Aurora programs hold the `i64` form (see
/// [`Key::to_i64`]); nothing outside this module ever sees the typed key.
type ItemId = Key<Renderable>;

pub struct Scene {
    pub renderer: Renderer3D,
    /// Registered drawables, generation-tagged: [`Scene::free_model`] releases
    /// an item's GPU meshes and materials and invalidates its handle, and a
    /// later load reusing that slot cannot be reached by the old handle.
    items: SlotMap<Renderable>,
    /// Path-keyed asset cache. One entry per distinct model FILE, holding the only other
    /// reference besides the live handles, so a file loaded twice is uploaded once.
    assets: std::collections::HashMap<String, Arc<Asset>>,
    cam: Camera,
    size: (u32, u32),
    clear: [f32; 4],
    /// The heightmap terrain, if one has been loaded. It sits beside the items
    /// rather than being one of them, because its geometry is re-chosen each
    /// frame from the camera this scene already owns.
    terrain: Option<crate::terrain::TerrainRender>,
    /// Albedo the terrain is built with.
    terrain_color: [f32; 3],
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
            items: SlotMap::new(),
            assets: std::collections::HashMap::new(),
            cam: Camera {
                eye: Vec3::new(0.0, 2.0, 6.0),
                target: Vec3::ZERO,
                up: Vec3::Y,
                roll: 0.0,
                fov_y: 60f32.to_radians(),
                near: 0.05,
                far: 500.0,
            },
            size: (w.max(1), h.max(1)),
            clear: [0.05, 0.06, 0.09, 1.0],
            terrain: None,
            terrain_color: [0.32, 0.40, 0.24],
        };
        s.update_camera();
        s.renderer
            .set_light(Vec3::new(0.4, 1.0, 0.3), Vec3::ONE, 0.25);
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
        // Bank the camera by rolling the up vector about the forward axis. Forward
        // is unchanged, so the centre of the screen still aims where you look.
        let fwd = (self.cam.target - self.cam.eye).normalize_or_zero();
        let up = if self.cam.roll.abs() > 1e-5 && fwd.length_squared() > 0.0 {
            glam::Quat::from_axis_angle(fwd, self.cam.roll) * self.cam.up
        } else {
            self.cam.up
        };
        let view = crate::look_at(self.cam.eye, self.cam.target, up);
        self.renderer.set_camera(proj * view, self.cam.eye);
    }

    /// Set the camera roll (banking) in radians; applied on the next camera update.
    pub fn set_camera_roll(&mut self, roll: f32) {
        self.cam.roll = roll;
        self.update_camera();
    }

    pub fn set_camera(&mut self, eye: Vec3, target: Vec3, fov_deg: f32) {
        self.cam.eye = eye;
        self.cam.target = target;
        self.cam.fov_y = fov_deg
            .to_radians()
            .clamp(0.05, std::f32::consts::PI - 0.05);
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
    pub fn set_ssao(&mut self, on: bool) {
        self.renderer.set_ssao(on);
    }
    pub fn set_point_shadows(&mut self, on: bool) {
        self.renderer.set_point_shadows(on);
    }
    pub fn set_viewmodel(&mut self, on: bool) {
        self.renderer.set_viewmodel(on);
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

    /// The current clear color (offscreen capture renders with the same
    /// background the live window would).
    pub fn clear_color(&self) -> [f32; 4] {
        self.clear
    }

    /// Load a model file (glTF/GLB/OBJ). Returns a handle or -1 on failure.
    ///
    /// Loading the SAME file again is cheap: the parsed model and its uploaded meshes and
    /// materials are shared, and only a fresh animation player is created. So the natural
    /// way to write a horde - one handle per body, so each animates independently - costs
    /// one upload per distinct file instead of one per body.
    pub fn load_model(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, path: &str) -> i64 {
        let key = Scene::asset_key(path);
        if let Some(asset) = self.assets.get(&key) {
            return self
                .items
                .insert(Renderable {
                    asset: Arc::clone(asset),
                    player: AnimPlayer::new(),
                    hidden_joints: 0,
                })
                .to_i64();
        }
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
                normal_tex: p
                    .normal_tex
                    .as_ref()
                    .map(|(px, w, h)| (px.as_slice(), *w, *h)),
                mr_tex: p.mr_tex.as_ref().map(|(px, w, h)| (px.as_slice(), *w, *h)),
                emissive_tex: p
                    .emissive_tex
                    .as_ref()
                    .map(|(px, w, h)| (px.as_slice(), *w, *h)),
            };
            let mat = self.renderer.add_material(device, queue, &desc);
            prims.push((mesh, mat));
            skinned |= p.skinned;
        }
        let asset = Arc::new(Asset {
            prims,
            model: Some(model),
            skinned,
            path: Some(key.clone()),
        });
        self.assets.insert(key, Arc::clone(&asset));
        self.items
            .insert(Renderable {
                asset,
                player: AnimPlayer::new(),
                hidden_joints: 0,
            })
            .to_i64()
    }

    /// The cache key for a model path: the canonical filesystem path when it resolves, and
    /// the string as given otherwise.
    ///
    /// Canonicalising matters because `models/x.glb` and `./models/x.glb` are the same
    /// upload, and a cache that missed on the spelling would quietly hand back the old
    /// per-call cost for no visible reason.
    fn asset_key(path: &str) -> String {
        match std::fs::canonicalize(path) {
            Ok(p) => p.to_string_lossy().into_owned(),
            Err(_) => path.to_string(),
        }
    }

    /// Wrap freshly built GPU primitives as a one-off, unshared asset.
    fn own_asset(&mut self, prims: Vec<(MeshId, MaterialId)>) -> i64 {
        self.items
            .insert(Renderable {
                asset: Arc::new(Asset {
                    prims,
                    model: None,
                    skinned: false,
                    path: None,
                }),
                player: AnimPlayer::new(),
                hidden_joints: 0,
            })
            .to_i64()
    }

    /// Release a model or primitive handle.
    ///
    /// Returns whether anything was freed. A stale handle - one already freed, or never
    /// issued - returns `false` and touches nothing; it cannot free whatever was loaded
    /// into that slot afterwards, because the handle carries the generation the slot was
    /// at when it was issued.
    ///
    /// GPU meshes and materials are SHARED between handles that loaded the same file, so
    /// they are reference-counted: this always drops the handle, and frees the upload only
    /// when the handle was the last user of it. Freeing one body of a horde therefore does
    /// not pull the mesh out from under the other twenty-three.
    pub fn free_model(&mut self, handle: i64) -> bool {
        let Some(key) = ItemId::from_i64(handle) else {
            return false;
        };
        let Some(item) = self.items.remove(key) else {
            return false;
        };
        let asset = item.asset;
        // Two references left (this one and the cache's) means no other handle is using it,
        // so the cache entry goes too - otherwise the upload would outlive its last user
        // and never be reclaimed.
        if let Some(path) = asset.path.clone() {
            if Arc::strong_count(&asset) == 2 {
                self.assets.remove(&path);
            }
        }
        if Arc::strong_count(&asset) == 1 {
            for &(mesh, mat) in &asset.prims {
                self.renderer.free_mesh(mesh);
                self.renderer.free_material(mat);
            }
        }
        true
    }

    /// How many distinct model FILES are uploaded. Handles can outnumber this freely: a
    /// horde of 24 bodies over 5 files is 24 handles and 5 assets.
    pub fn asset_count(&self) -> usize {
        self.assets.len()
    }

    /// Number of live model/primitive handles.
    pub fn model_count(&self) -> usize {
        self.items.len()
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
        let mat = self
            .renderer
            .add_material(device, queue, &MaterialDesc::flat(color));
        self.own_asset(vec![(m, mat)])
    }

    pub fn make_box(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, color: [f32; 4]) -> i64 {
        self.add_primitive(device, queue, &MeshData::cube(), color)
    }
    pub fn make_box_sized(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        hx: f32,
        hy: f32,
        hz: f32,
        color: [f32; 4],
    ) -> i64 {
        self.add_primitive(device, queue, &MeshData::box_dims(hx, hy, hz), color)
    }
    /// A box that GLOWS (emissive material, self-lit regardless of scene lighting).
    pub fn make_box_emissive(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        hx: f32,
        hy: f32,
        hz: f32,
        color: [f32; 3],
    ) -> i64 {
        let m = self
            .renderer
            .add_mesh(device, &MeshData::box_dims(hx, hy, hz));
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
        self.own_asset(vec![(m, mat)])
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
    pub fn make_sprite(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        color: [f32; 3],
    ) -> i64 {
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
        self.own_asset(vec![(m, mat)])
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

    /// Draw a handle many times in a single GPU instanced draw call per
    /// primitive (one draw for all `transforms`, not N draws).
    pub fn draw_instances(&mut self, handle: i64, transforms: &[Mat4]) {
        let Some(item) = self.item(handle) else {
            return;
        };
        let prims = item.asset.prims.clone();
        let insts: Vec<crate::render::InstanceRaw> = transforms
            .iter()
            .map(|&t| crate::render::InstanceRaw::new(t, [1.0; 4]))
            .collect();
        for (mesh, mat) in prims {
            self.renderer.draw_instanced(mesh, mat, insts.clone());
        }
    }

    /// Number of animation clips on a model handle.
    pub fn clip_count(&self, handle: i64) -> i64 {
        self.item(handle)
            .and_then(|r| r.asset.model.as_ref())
            .map(|m| m.clips.len() as i64)
            .unwrap_or(0)
    }

    /// The name of clip `i` as the asset declares it, or `None` for a stale
    /// handle or an out-of-range index. Clip names are loaded already; without a
    /// way to read them a game can only address animations by bare index, which
    /// silently plays the WRONG motion the moment an artist re-exports the model
    /// with its clips in a different order.
    pub fn clip_name(&self, handle: i64, i: i64) -> Option<&str> {
        let m = self.item(handle)?.asset.model.as_ref()?;
        if i < 0 {
            return None;
        }
        m.clips.get(i as usize).map(|c| c.name.as_str())
    }

    /// Index of the clip called `name`, or -1 if this model has no such clip.
    ///
    /// Exporters routinely prefix a clip with its armature (Blender/glTF emit
    /// `CharacterArmature|Walk`), so an exact match is tried first and then the
    /// segment after the last `|`. Matching is case-insensitive because that
    /// prefix and the casing are export settings, not authored intent.
    pub fn clip_index(&self, handle: i64, name: &str) -> i64 {
        let Some(m) = self.item(handle).and_then(|r| r.asset.model.as_ref()) else {
            return -1;
        };
        match_name(m.clips.iter().map(|c| c.name.as_str()), name)
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
            if let Some(model) = &r.asset.model {
                r.player.advance(model, dt);
            }
        }
    }

    /// Start an upper-body overlay clip on a model, masked to joint `mask_root` and its
    /// descendants (so the legs keep the base clip). Fades in over `fade` seconds.
    pub fn anim_play_upper(
        &mut self,
        handle: i64,
        clip: i64,
        looping: bool,
        speed: f32,
        fade: f32,
        mask_root: i64,
    ) {
        if let Some(r) = self.item_mut(handle) {
            r.player.play_upper(
                clip.max(0) as usize,
                looping,
                speed,
                fade,
                mask_root.max(0) as usize,
            );
        }
    }

    /// Drive the FULL-BODY base as a sustained weighted blend of two clips (`clip_a` at weight 0,
    /// `clip_b` at weight 1) - e.g. idle <-> run by speed. Call every frame to update the weight; the
    /// first call crossfades in over `fade` so jump->land and similar transitions stay smooth.
    pub fn anim_blend(
        &mut self,
        handle: i64,
        clip_a: i64,
        clip_b: i64,
        weight: f32,
        speed: f32,
        fade: f32,
    ) {
        if let Some(r) = self.item_mut(handle) {
            r.player.blend(
                clip_a.max(0) as usize,
                clip_b.max(0) as usize,
                weight,
                speed,
                fade,
            );
        }
    }

    /// Drive the upper-body overlay as a weighted BLEND of two clips (`clip_a` at weight 0, `clip_b`
    /// at weight 1), masked to `mask_root`. Call every frame to track a continuous value such as aim
    /// pitch (look down -> up); only the first call fades in, so per-frame weight updates stay smooth.
    // The parameter list mirrors this builtin's row in `aurora-abi`, which is
    // the single source of truth for its signature; grouping the arguments
    // would break the 1:1 correspondence the table is built on.
    #[allow(clippy::too_many_arguments)]
    pub fn anim_aim_upper(
        &mut self,
        handle: i64,
        clip_a: i64,
        clip_b: i64,
        weight: f32,
        speed: f32,
        fade: f32,
        mask_root: i64,
    ) {
        if let Some(r) = self.item_mut(handle) {
            r.player.aim_upper(
                clip_a.max(0) as usize,
                clip_b.max(0) as usize,
                weight,
                speed,
                fade,
                mask_root.max(0) as usize,
            );
        }
    }

    /// Set a per-bone pose override (extra local XYZ-Euler rotation on `joint`), e.g. to author a
    /// slide the clips don't have. Set each frame; clear_pose() resets a model to its pure clip pose.
    pub fn pose_bone(&mut self, handle: i64, joint: i64, rx: f32, ry: f32, rz: f32) {
        if let Some(r) = self.item_mut(handle) {
            let q = glam::Quat::from_euler(glam::EulerRot::XYZ, rx, ry, rz);
            r.player.set_pose(joint.max(0) as usize, q);
        }
    }

    /// Drop all per-bone pose overrides on a model.
    pub fn clear_pose(&mut self, handle: i64) {
        if let Some(r) = self.item_mut(handle) {
            r.player.clear_pose();
        }
    }

    /// Fade out a model's upper-body overlay over `fade` seconds.
    pub fn anim_stop_upper(&mut self, handle: i64, fade: f32) {
        if let Some(r) = self.item_mut(handle) {
            r.player.stop_upper(fade);
        }
    }

    /// Jump a model's upper-body overlay playback to `t` seconds (skip a clip wind-up).
    pub fn anim_seek_upper(&mut self, handle: i64, t: f32) {
        if let Some(r) = self.item_mut(handle) {
            r.player.seek_upper(t);
        }
    }

    pub fn begin(&mut self) {
        self.renderer.begin();
    }

    // --- terrain ----------------------------------------------------------

    /// Install (or replace) the heightmap terrain. The heightfield is shared
    /// with the runtime's height query and physics collider through the `Arc`,
    /// so the surface drawn here is the same data those answer from.
    ///
    /// Installing the SAME heightfield again is free, which is what lets the
    /// runtime re-offer its terrain on every draw. It has to: the scene does not
    /// exist until the window (or the headless device) does, so a program that
    /// loads its terrain before opening a window would otherwise hand it to
    /// nothing and render an empty world.
    ///
    /// Installing a DIFFERENT heightfield releases the outgoing terrain's GPU
    /// meshes and material first. Dropping the old [`TerrainRender`] is not
    /// enough on its own: its tile meshes live in the renderer's store, which
    /// outlives any single terrain, so the terrain has to hand them back
    /// explicitly or every reload leaks a whole terrain's worth of geometry.
    pub fn set_terrain(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        field: Arc<crate::terrain::Heightfield>,
    ) {
        if let Some(t) = self.terrain.as_mut() {
            if Arc::ptr_eq(t.field(), &field) {
                return;
            }
            t.release(&mut self.renderer);
        }
        self.terrain = Some(crate::terrain::TerrainRender::new(
            device,
            queue,
            &mut self.renderer,
            field,
            self.terrain_color,
        ));
    }

    /// Drop the terrain entirely, releasing its GPU meshes and material.
    pub fn clear_terrain(&mut self) {
        if let Some(mut t) = self.terrain.take() {
            t.release(&mut self.renderer);
        }
    }

    /// GPU bytes held by the terrain's resident tile meshes, and the budget
    /// that bounds them. `(0, 0)` when no terrain is loaded.
    pub fn terrain_tile_bytes(&self) -> (u64, u64) {
        self.terrain
            .as_ref()
            .map_or((0, 0), |t| (t.resident_bytes(), t.budget_bytes()))
    }

    /// The live terrain, for callers that need to tune its tile cache.
    pub fn terrain_mut(&mut self) -> Option<&mut crate::terrain::TerrainRender> {
        self.terrain.as_mut()
    }

    /// Set the terrain albedo. Safe to call before the terrain is loaded.
    pub fn set_terrain_color(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        color: [f32; 3],
    ) {
        self.terrain_color = color;
        if let Some(t) = self.terrain.as_mut() {
            t.set_color(device, queue, &mut self.renderer, color);
        }
    }

    /// Queue the terrain for this frame at the level of detail the current
    /// camera calls for. No-op when no terrain is loaded.
    pub fn draw_terrain(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        let eye = self.cam.eye;
        if let Some(t) = self.terrain.as_mut() {
            t.draw(device, queue, &mut self.renderer, eye);
        }
    }

    /// The loaded terrain's heightfield, if any.
    pub fn terrain_field(&self) -> Option<&Arc<crate::terrain::Heightfield>> {
        self.terrain.as_ref().map(|t| t.field())
    }

    /// `(tiles queued, finest sample step, coarsest sample step)` of the last
    /// [`Self::draw_terrain`]. Lets a caller confirm a level-of-detail seam was
    /// actually on screen before concluding anything from the pixels.
    pub fn terrain_last_draw(&self) -> Option<(usize, u32, u32)> {
        self.terrain.as_ref().map(|t| t.last_draw())
    }

    /// Queue a model for drawing at `transform`.
    /// Hide one skin joint's geometry on a model (its skinning matrix is zeroed, collapsing that
    /// geometry to the model origin). Accumulates; clear with [`show_joints`]. Used by first-person
    /// arms to drop the torso/head/legs so only the arms render.
    pub fn hide_joint(&mut self, handle: i64, joint: i64) {
        if let Some(r) = self.item_mut(handle) {
            if (0..64).contains(&joint) {
                r.hidden_joints |= 1u64 << joint;
            }
        }
    }

    /// Show all joints again (clear the hidden mask).
    pub fn show_joints(&mut self, handle: i64) {
        if let Some(r) = self.item_mut(handle) {
            r.hidden_joints = 0;
        }
    }

    /// What every `draw*` variant needs from a handle: the shared skinning
    /// matrices and the primitives to issue. `None` for a stale handle, which
    /// is how a freed model stops drawing instead of drawing something else.
    ///
    /// The skinning matrices are computed ONCE and shared across the item's
    /// primitives via `Arc` (a refcount bump per primitive instead of a deep
    /// copy of the 128-matrix array).
    #[allow(clippy::type_complexity)]
    fn draw_parts(
        &self,
        handle: i64,
    ) -> Option<(Option<Arc<Vec<Mat4>>>, Vec<(MeshId, MaterialId)>)> {
        let r = self.item(handle)?;
        let joints = if r.asset.skinned {
            r.asset
                .model
                .as_ref()
                .map(|m| r.player.matrices(m, r.hidden_joints))
                .filter(|v| !v.is_empty())
                .map(Arc::new)
        } else {
            None
        };
        Some((joints, r.asset.prims.clone()))
    }

    pub fn draw(&mut self, handle: i64, transform: Mat4) {
        let Some((joints, prims)) = self.draw_parts(handle) else {
            return;
        };
        for (mesh, mat) in prims {
            self.renderer.draw(mesh, mat, transform, joints.clone());
        }
    }

    /// Draw `armor`'s mesh skinned by `host`'s CURRENT pose. The armor mesh
    /// carries per-vertex joint indices/weights (in the host's skinning order);
    /// this feeds the HOST's skin matrices to those weights, so an attached piece
    /// of gear deforms exactly with the character without owning a skeleton.
    pub fn draw_skinned(&mut self, armor: i64, host: i64, transform: Mat4) {
        // Skin the armour from the host's FULL pose - never the host's hidden mask.
        // hide_joint hides the host's OWN covered mesh (so the body can't clip through
        // the armour); the armour worn over it must still render in full, otherwise
        // hiding the body under a gauntlet would collapse the gauntlet too.
        let host_joints = self.item(host).and_then(|r| {
            r.asset
                .model
                .as_ref()
                .map(|m| r.player.matrices(m, 0))
                .filter(|v| !v.is_empty())
                .map(Arc::new)
        });
        let Some(prims) = self.item(armor).map(|r| r.asset.prims.clone()) else {
            return;
        };
        for (mesh, mat) in prims {
            self.renderer
                .draw(mesh, mat, transform, host_joints.clone());
        }
    }

    /// Like [`draw`] but shifts the model's albedo by `tint` (RGB additive offset).
    pub fn draw_tint(&mut self, handle: i64, transform: Mat4, tint: [f32; 3]) {
        let Some((joints, prims)) = self.draw_parts(handle) else {
            return;
        };
        for (mesh, mat) in prims {
            self.renderer
                .draw_tint(mesh, mat, transform, joints.clone(), tint);
        }
    }

    /// Like [`draw`] but with an energy-shield Fresnel rim (cyan, `strength` 0..1, animated
    /// by `time`).
    pub fn draw_shield(&mut self, handle: i64, transform: Mat4, strength: f32, time: f32) {
        let Some((joints, prims)) = self.draw_parts(handle) else {
            return;
        };
        for (mesh, mat) in prims {
            self.renderer
                .draw_shield(mesh, mat, transform, joints.clone(), strength, time);
        }
    }

    /// Draw `weapon` attached to `joint` of `host` (posed at `host_xform`), with the
    /// weapon's own `local` offset relative to that bone:
    ///   weapon_world = host_xform * joint_global(host pose) * local.
    /// Falls back to host_xform * local if the joint/skeleton is missing.
    pub fn draw_on_joint(
        &mut self,
        weapon: i64,
        host: i64,
        joint: i64,
        host_xform: Mat4,
        local: Mat4,
    ) {
        let g = self
            .item(host)
            .and_then(|r| {
                r.asset
                    .model
                    .as_ref()
                    .and_then(|m| r.player.joint_global(m, joint.max(0) as usize))
            })
            .unwrap_or(Mat4::IDENTITY);
        self.draw(weapon, host_xform * g * local);
    }

    /// The full model-space global transform of `joint` in the host's CURRENT
    /// pose (what `draw_on_joint` composes with). Tooling uses it to draw
    /// attachment gnomons and to solve socket transforms.
    pub fn joint_global_mat(&self, host: i64, joint: i64) -> Option<Mat4> {
        let r = self.item(host)?;
        r.asset
            .model
            .as_ref()
            .and_then(|m| r.player.joint_global(m, joint.max(0) as usize))
    }

    /// Index of the joint called `name`, or -1 when this model has no such joint.
    ///
    /// The counterpart of [`Scene::clip_index`], and the same argument applies
    /// with more force: a game that hardcodes `hand_joint = 29` breaks silently
    /// into a weapon welded to a shin the first time the rig changes. Matching
    /// tolerates an armature prefix and ignores case.
    pub fn joint_index(&self, host: i64, name: &str) -> i64 {
        let Some(skel) = self
            .item(host)
            .and_then(|r| r.asset.model.as_ref())
            .and_then(|m| m.skeleton.as_ref())
        else {
            return -1;
        };
        match_name(skel.joints.iter().map(|j| j.name.as_str()), name)
    }

    /// The name of joint `i`, or `None` for a stale handle or a bad index. The
    /// discovery counterpart of [`Scene::joint_index`].
    pub fn joint_name(&self, host: i64, i: i64) -> Option<&str> {
        let skel = self.item(host)?.asset.model.as_ref()?.skeleton.as_ref()?;
        if i < 0 {
            return None;
        }
        skel.joints.get(i as usize).map(|j| j.name.as_str())
    }

    /// The model-space position of `joint` in the host's CURRENT pose (the translation of its
    /// global transform, before the draw transform). Lets a first-person rig cancel the bone offset
    /// so a bone-attached weapon lands at a fixed camera-space spot. None if missing.
    pub fn joint_pos(&self, host: i64, joint: i64) -> Option<[f32; 3]> {
        let r = self.item(host)?;
        let g = r
            .asset
            .model
            .as_ref()
            .and_then(|m| r.player.joint_global(m, joint.max(0) as usize))?;
        let t = g.w_axis;
        Some([t.x, t.y, t.z])
    }

    /// Draw the host's skeleton as debug lines (parent->child bones) at the
    /// given world transform, for headless rig/hitbox visual audits. Uses the
    /// current animation pose. No-op if the model has no skeleton.
    pub fn debug_skeleton(&mut self, host: i64, host_xform: Mat4, color: Vec3) {
        // Collect (parent_world, child_world) segments first (immutable borrow),
        // then draw (mutable borrow of the renderer).
        let mut segs: Vec<(Vec3, Vec3)> = Vec::new();
        {
            let Some(r) = self.item(host) else {
                return;
            };
            let Some(model) = r.asset.model.as_ref() else {
                return;
            };
            let Some(skel) = model.skeleton.as_ref() else {
                return;
            };
            for (ji, joint) in skel.joints.iter().enumerate() {
                let Some(parent) = joint.parent else { continue };
                let (Some(cg), Some(pg)) = (
                    r.player.joint_global(model, ji),
                    r.player.joint_global(model, parent),
                ) else {
                    continue;
                };
                let cp = host_xform.transform_point3(cg.w_axis.truncate());
                let pp = host_xform.transform_point3(pg.w_axis.truncate());
                segs.push((pp, cp));
            }
        }
        for (a, b) in segs {
            self.renderer.debug_line(a, b, color);
        }
    }

    /// Print every joint index + name of `host` to stdout (bone-discovery helper).
    pub fn dump_joints(&self, host: i64) {
        let Some(item) = self.item(host) else {
            println!("joint dump: bad handle {host}");
            return;
        };
        let Some(model) = item.asset.model.as_ref() else {
            println!("joint dump: no model");
            return;
        };
        let Some(skel) = model.skeleton.as_ref() else {
            println!("joint dump: no skeleton");
            return;
        };
        println!("== joint dump: {} joints ==", skel.joints.len());
        for (i, j) in skel.joints.iter().enumerate() {
            println!("  [{i}] '{}' (parent {:?})", j.name, j.parent);
        }
    }

    pub fn render(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
    ) {
        self.renderer
            .render(device, queue, encoder, view, self.clear);
    }

    /// The item `handle` names, or `None` when it was never issued or has been
    /// freed. Every handle-taking method goes through here or its `_mut`
    /// counterpart, so a stale handle is refused in exactly one place.
    fn item(&self, handle: i64) -> Option<&Renderable> {
        self.items.get(ItemId::from_i64(handle)?)
    }
    fn item_mut(&mut self, handle: i64) -> Option<&mut Renderable> {
        self.items.get_mut(ItemId::from_i64(handle)?)
    }
}

#[cfg(test)]
mod tests;
