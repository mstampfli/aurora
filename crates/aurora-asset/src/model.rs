//! glTF (.gltf/.glb) and OBJ model loading: meshes, materials (base color +
//! texture), skeletons (joints, inverse-bind matrices, hierarchy), and skeletal
//! animation clips.

use std::collections::{HashMap, HashSet};

use glam::{Mat4, Quat, Vec3};

use crate::mesh::{MeshData, Vertex};

/// A tightly-packed RGBA8 texture: `(pixels, w, h)`.
pub type Tex = (Vec<u8>, u32, u32);

/// Decode an image file (PNG or JPEG) to tightly-packed RGBA8.
///
/// Lives here rather than in the renderer so that decoding art is the asset
/// layer's job wherever it is triggered from, and so a caller that needs a
/// texture without a GPU - a baker, a test - can still get one.
pub fn load_texture_file(path: &str) -> Result<Tex, String> {
    let img = image::open(path).map_err(|e| format!("load texture {path}: {e}"))?;
    let rgba = img.to_rgba8();
    let (w, h) = rgba.dimensions();
    Ok((rgba.into_raw(), w, h))
}

/// Decode an image already in memory (an embedded texture) to RGBA8.
pub fn decode_texture(bytes: &[u8]) -> Option<Tex> {
    let img = image::load_from_memory(bytes).ok()?;
    let rgba = img.to_rgba8();
    let (w, h) = rgba.dimensions();
    Some((rgba.into_raw(), w, h))
}

/// A drawable piece of a model: geometry plus a PBR material.
pub struct Primitive {
    pub mesh: MeshData,
    /// The material's name in the source file, or empty when it has none.
    ///
    /// Kept because a stylised pack routinely ships meshes with no texture
    /// bound, expecting the engine to attach a shared atlas chosen by material
    /// name. Discarding the name would leave nothing to choose it by.
    pub material: String,
    pub base_color: [f32; 4],
    pub metallic: f32,
    pub roughness: f32,
    pub emissive: [f32; 3],
    pub texture: Option<Tex>,
    pub normal_tex: Option<Tex>,
    pub mr_tex: Option<Tex>,
    pub emissive_tex: Option<Tex>,
    /// Whether this primitive carries skinning weights.
    pub skinned: bool,
}

/// One bone.
#[derive(Clone)]
pub struct Joint {
    pub parent: Option<usize>,
    pub inverse_bind: Mat4,
    pub t: Vec3,
    pub r: Quat,
    pub s: Vec3,
    /// glTF node name (e.g. "Hand.R") - lets a game find a bone to attach props to.
    pub name: String,
}

/// A skeleton: joints in skinning order with their default local transforms.
#[derive(Clone)]
pub struct Skeleton {
    pub joints: Vec<Joint>,
}

impl Skeleton {
    pub fn joint_count(&self) -> usize {
        self.joints.len()
    }

    /// Model-space transform of every joint in the rest (bind) pose.
    ///
    /// Joints are stored parent-before-child by both importers, but this does
    /// not rely on that: each joint resolves through its parent chain, so a
    /// skeleton in any order gives the same answer.
    pub fn rest_globals(&self) -> Vec<Mat4> {
        let mut out = vec![None; self.joints.len()];
        for i in 0..self.joints.len() {
            resolve_rest(self, i, &mut out);
        }
        out.into_iter().map(|m| m.unwrap_or(Mat4::IDENTITY)).collect()
    }

    /// Per-joint matrices that take bind-space geometry to model space in the
    /// rest pose - the identity transform of the skinning pipeline.
    pub fn bind_matrices(&self) -> Vec<Mat4> {
        let globals = self.rest_globals();
        self.joints
            .iter()
            .enumerate()
            .map(|(i, j)| globals[i] * j.inverse_bind)
            .collect()
    }

    /// Per-joint local TRS with `clip` applied at `time`.
    ///
    /// Joints the clip does not drive keep their authored rest transform, which
    /// is what makes a clip authored for part of a body composable with the
    /// rest of it. `None` yields the rest pose unchanged.
    ///
    /// This is the one clip sampler. It is pure math over a skeleton and a
    /// clip - no playback state, no device - so it lives with the data rather
    /// than in the renderer, and the importer's own verification uses exactly
    /// the code that will pose the character on screen.
    pub fn sample(&self, clip: Option<&Clip>, time: f32) -> (Vec<Vec3>, Vec<Quat>, Vec<Vec3>) {
        let n = self.joints.len();
        let mut t: Vec<Vec3> = self.joints.iter().map(|j| j.t).collect();
        let mut r: Vec<Quat> = self.joints.iter().map(|j| j.r).collect();
        let mut s: Vec<Vec3> = self.joints.iter().map(|j| j.s).collect();
        if let Some(clip) = clip {
            for ch in &clip.channels {
                if ch.joint >= n {
                    continue;
                }
                match ch.path {
                    Path::Translation => t[ch.joint] = sample_vec3(ch, time),
                    Path::Scale => s[ch.joint] = sample_vec3(ch, time),
                    Path::Rotation => r[ch.joint] = sample_quat(ch, time),
                }
            }
        }
        (t, r, s)
    }

    /// Model-space transform of every joint for a local TRS pose.
    pub fn globals(&self, t: &[Vec3], r: &[Quat], s: &[Vec3]) -> Vec<Mat4> {
        let local: Vec<Mat4> = (0..self.joints.len())
            .map(|i| Mat4::from_scale_rotation_translation(s[i], r[i], t[i]))
            .collect();
        let mut out = vec![None; self.joints.len()];
        for i in 0..self.joints.len() {
            resolve_pose(self, &local, i, &mut out);
        }
        out.into_iter().map(|m| m.unwrap_or(Mat4::IDENTITY)).collect()
    }
}

/// Resolve one joint's global for a posed skeleton, memoizing into `out`.
///
/// Iterative for the same reason as [`resolve_rest`]: a self-parented joint in a
/// malformed file must not take the stack down with it.
fn resolve_pose(skel: &Skeleton, local: &[Mat4], joint: usize, out: &mut Vec<Option<Mat4>>) {
    let mut chain = Vec::new();
    let mut cur = Some(joint);
    while let Some(i) = cur {
        if out[i].is_some() || chain.contains(&i) {
            break;
        }
        chain.push(i);
        cur = skel.joints[i].parent;
    }
    for &i in chain.iter().rev() {
        // A root joint's local already folds in every non-joint ancestor above
        // it, so it composes against the world rather than against a parent.
        out[i] = Some(match skel.joints[i].parent {
            Some(p) if p != i => out[p].unwrap_or(Mat4::IDENTITY) * local[i],
            _ => local[i],
        });
    }
}

/// Find the key interval `[i, i+1]` containing `time` and the fraction within.
fn locate(times: &[f32], time: f32) -> (usize, usize, f32) {
    if times.is_empty() {
        return (0, 0, 0.0);
    }
    if time <= times[0] {
        return (0, 0, 0.0);
    }
    let last = times.len() - 1;
    if time >= times[last] {
        return (last, last, 0.0);
    }
    let mut i = 0;
    while i + 1 < times.len() && times[i + 1] < time {
        i += 1;
    }
    let (a, b) = (times[i], times[i + 1]);
    let f = if b > a { (time - a) / (b - a) } else { 0.0 };
    (i, i + 1, f)
}

fn sample_vec3(ch: &Channel, time: f32) -> Vec3 {
    let (i0, i1, f) = locate(&ch.times, time);
    let get = |k: usize| Vec3::new(ch.values[k * 3], ch.values[k * 3 + 1], ch.values[k * 3 + 2]);
    if ch.interp == Interp::Step || i0 == i1 {
        get(i0)
    } else {
        get(i0).lerp(get(i1), f)
    }
}

fn sample_quat(ch: &Channel, time: f32) -> Quat {
    let (i0, i1, f) = locate(&ch.times, time);
    let get = |k: usize| {
        Quat::from_xyzw(
            ch.values[k * 4],
            ch.values[k * 4 + 1],
            ch.values[k * 4 + 2],
            ch.values[k * 4 + 3],
        )
        .normalize()
    };
    if ch.interp == Interp::Step || i0 == i1 {
        get(i0)
    } else {
        get(i0).slerp(get(i1), f)
    }
}

/// Resolve one joint's rest global, memoizing into `out`.
///
/// Iterative rather than recursive: a corrupt file can name a joint its own
/// ancestor, and a recursive walk would blow the stack instead of reporting a
/// bad asset. The visited set makes a cycle terminate at identity.
fn resolve_rest(skel: &Skeleton, joint: usize, out: &mut Vec<Option<Mat4>>) {
    let mut chain = Vec::new();
    let mut cur = Some(joint);
    while let Some(i) = cur {
        if out[i].is_some() || chain.contains(&i) {
            break;
        }
        chain.push(i);
        cur = skel.joints[i].parent;
    }
    for &i in chain.iter().rev() {
        let j = &skel.joints[i];
        let local = Mat4::from_scale_rotation_translation(j.s, j.r, j.t);
        out[i] = Some(match j.parent {
            Some(p) => out[p].unwrap_or(Mat4::IDENTITY) * local,
            None => local,
        });
    }
}

/// Which transform component an animation channel drives.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Path {
    Translation,
    Rotation,
    Scale,
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Interp {
    Linear,
    Step,
}

#[derive(Clone, Debug)]
pub struct Channel {
    pub joint: usize,
    pub path: Path,
    pub interp: Interp,
    pub times: Vec<f32>,
    /// Flattened values: 3 per key for T/S, 4 per key for R (xyzw).
    pub values: Vec<f32>,
}

/// A named animation: a set of per-joint TRS channels.
#[derive(Clone, Debug)]
pub struct Clip {
    pub name: String,
    pub duration: f32,
    pub channels: Vec<Channel>,
}

/// Everything a retarget needs to move a clip from one rig to another.
pub struct Retarget<'a> {
    /// The skeleton the clip's channel indices address - the one that came out
    /// of the clip's own file.
    pub source: &'a Skeleton,
    /// The source rig's TRUE rest pose, by bone name.
    ///
    /// Usually this is `source`, but not for a clip-only export: those ship no
    /// bind data, so every joint's rest transform is a placeholder and cannot be
    /// used as the reference the motion was authored against. An animation pack
    /// that ships clips without a rig also ships the rig they were authored on,
    /// and that is what belongs here.
    pub source_rest: &'a Skeleton,
    /// The skeleton to retarget onto.
    pub target: &'a Skeleton,
    /// Source bone name to target bone name, for the bones whose names differ.
    pub rename: &'a [(&'a str, &'a str)],
    /// Target bones allowed to take translation from the clip - normally just
    /// the root or the hips.
    pub translate: &'a [&'a str],
}

impl Retarget<'_> {
    /// The target bone a source bone maps to.
    fn target_name<'n>(&self, source_name: &'n str) -> &'n str
    where
        Self: 'n,
    {
        self.rename
            .iter()
            .find(|(a, _)| a.eq_ignore_ascii_case(source_name))
            .map(|(_, b)| *b)
            .unwrap_or(source_name)
    }

    fn joint_by_name<'s>(skel: &'s Skeleton, name: &str) -> Option<(usize, &'s Joint)> {
        skel.joints
            .iter()
            .position(|j| j.name.eq_ignore_ascii_case(name))
            .map(|i| (i, &skel.joints[i]))
    }
}

impl Clip {
    /// Rewrite this clip's channels to address `target`'s joints instead of
    /// `source`'s, matching by bone name.
    ///
    /// This is the whole of retargeting when two rigs share a rest pose - the
    /// usual case for a pack whose animations and characters were built to the
    /// same proportions, and the case worth checking for before reaching for
    /// anything heavier. A channel holds a rotation in its joint's parent frame,
    /// so if the joint sits in the same place on both skeletons the rotation
    /// means the same thing on both and only the addressing has to change. Rigs
    /// that genuinely differ in proportion need rest-relative transfer and limb
    /// scaling; this is not that, and does not pretend to be.
    ///
    /// `rename` maps a source bone name to a target bone name. Names absent from
    /// it are matched as they are, case-insensitively, so a map need only list
    /// the bones whose names actually differ.
    ///
    /// `translate` names the TARGET bones permitted to take their translation
    /// from the clip. Everything else keeps the target skeleton's own bone
    /// offsets, and only rotation transfers.
    ///
    /// That restriction is the difference between an animated character and a
    /// heap. A clip-only export carries no bone offsets at all - every joint's
    /// local translation is zero and the motion lives entirely in rotations,
    /// because the file is meant to drive a skeleton it does not ship. Copying
    /// those translations across replaces a character's real bone lengths with
    /// zero and collapses the whole body onto its hip. Bone offsets belong to
    /// the skeleton; only the root's travel belongs to the clip, which is why
    /// that one bone has to be named explicitly.
    ///
    /// Channels naming a joint the target lacks are dropped - a source rig
    /// routinely drives bones no character has, like a jaw or a weapon socket.
    /// A clip where NOTHING matched is an error rather than an empty clip,
    /// because that means the map is wrong and silence would hide it.
    pub fn retarget(&self, opts: &Retarget) -> Result<Clip, String> {
        let mut channels = Vec::with_capacity(self.channels.len());
        let mut dropped = Vec::new();

        for ch in &self.channels {
            let Some(from) = opts.source.joints.get(ch.joint) else {
                continue;
            };
            let want = opts.target_name(&from.name);

            let Some((joint, to_rest)) = Retarget::joint_by_name(opts.target, want) else {
                if !dropped.contains(&from.name) {
                    dropped.push(from.name.clone());
                }
                continue;
            };
            // The rest the clip was authored against, looked up by the SOURCE
            // name: `source_rest` is the source rig, not the target.
            let from_rest = Retarget::joint_by_name(opts.source_rest, &from.name).map(|(_, j)| j);

            let values = match ch.path {
                // Scale is never transferred. A clip that does not author it
                // still reports the source rig's own scale, and on a rig whose
                // root carries a unit conversion that resizes the character.
                Path::Scale => continue,

                Path::Rotation => {
                    // A clip stores each joint's LOCAL rotation, and a local
                    // rotation only means anything against the rest orientation
                    // it was authored from. Two rigs can agree on every joint's
                    // world position and still bake different orientations into
                    // their locals, so copying the value across replaces the
                    // target's bind orientation with the source's and folds the
                    // skeleton up.
                    //
                    // What transfers is the motion AWAY from rest, re-expressed
                    // against the target's own rest:
                    //     delta  = inverse(source_rest) * clip
                    //     result = target_rest * delta
                    let Some(from_rest) = from_rest else {
                        return Err(format!(
                            "bone {} is animated but absent from the source rest skeleton, so \
                             there is nothing to measure its motion against",
                            from.name
                        ));
                    };
                    let basis = to_rest.r * from_rest.r.inverse();
                    ch.values
                        .chunks_exact(4)
                        .flat_map(|q| {
                            let r = basis
                                * Quat::from_xyzw(q[0], q[1], q[2], q[3]).normalize();
                            [r.x, r.y, r.z, r.w]
                        })
                        .collect()
                }

                Path::Translation => {
                    if !opts.translate.iter().any(|n| n.eq_ignore_ascii_case(want)) {
                        continue;
                    }
                    // Root travel is authored in the source rig's own local
                    // units, which need not be the target's - one rig may put a
                    // unit conversion on its root and the other not. Rescaling
                    // by the ratio of the two rest offsets carries the motion
                    // over as the same distance relative to the body.
                    let scale = from_rest
                        .map(|f| {
                            let (a, b) = (to_rest.t.length(), f.t.length());
                            if b > 1e-6 {
                                a / b
                            } else {
                                1.0
                            }
                        })
                        .unwrap_or(1.0);
                    ch.values.iter().map(|v| v * scale).collect()
                }
            };

            channels.push(Channel {
                joint,
                path: ch.path,
                interp: ch.interp,
                times: ch.times.clone(),
                values,
            });
        }

        if channels.is_empty() {
            return Err(format!(
                "clip {} retargeted to nothing: none of its {} joints matched the target \
                 skeleton, so the bone map is wrong",
                self.name,
                dropped.len()
            ));
        }

        Ok(Clip {
            name: self.name.clone(),
            duration: self.duration,
            channels,
        })
    }
}

/// A loaded model: drawable primitives, an optional skeleton, and clips.
pub struct Model {
    pub primitives: Vec<Primitive>,
    pub skeleton: Option<Skeleton>,
    pub clips: Vec<Clip>,
}

impl Model {
    /// Axis-aligned bounds of this model as it is actually drawn, in model
    /// space: `[min_x, min_y, min_z, max_x, max_y, max_z]`.
    ///
    /// Use this, not [`MeshData::bounds`], to size anything that must match what
    /// appears on screen - a collider, a culling volume, a camera framing. A
    /// skinned mesh's vertices live in the source file's bind space, which for
    /// an FBX authored in centimetres is a hundred times the model's real size;
    /// only after the bind matrices are applied is the geometry in model space.
    /// Static primitives are already there and are measured as they are.
    pub fn bind_pose_bounds(&self) -> [f32; 6] {
        let bind = self.skeleton.as_ref().map(|s| s.bind_matrices());
        let mut lo = Vec3::splat(f32::MAX);
        let mut hi = Vec3::splat(f32::MIN);
        let mut any = false;

        for prim in &self.primitives {
            for v in &prim.mesh.vertices {
                let p = Vec3::from(v.pos);
                let p = match (prim.skinned, &bind) {
                    (true, Some(bind)) => {
                        let mut acc = Vec3::ZERO;
                        let mut total = 0.0;
                        for k in 0..4 {
                            let w = v.weights[k];
                            let j = v.joints[k] as usize;
                            if w > 0.0 && j < bind.len() {
                                acc += w * bind[j].transform_point3(p);
                                total += w;
                            }
                        }
                        // An unweighted vertex would collapse to the origin and
                        // drag the bounds with it; leave it where it lies.
                        if total > 0.0 {
                            acc / total
                        } else {
                            p
                        }
                    }
                    _ => p,
                };
                lo = lo.min(p);
                hi = hi.max(p);
                any = true;
            }
        }

        if !any {
            return [0.0; 6];
        }
        [lo.x, lo.y, lo.z, hi.x, hi.y, hi.z]
    }

    /// Rebind this model's skinning onto `target`, matching joints by name.
    ///
    /// This is what makes a modular character possible. Each part is authored as
    /// its own file with its own skeleton - a head exports eight joints, a
    /// forearm eight different ones - and its vertices index that private list.
    /// Rewriting those indices into a shared skeleton's order lets one pose
    /// drive every part, so a character assembled from a dozen meshes animates
    /// as one body and costs one pose evaluation rather than a dozen.
    ///
    /// A part may cover any subset of the target's joints; that is the normal
    /// case. What it may not do is disagree about where those joints sit at
    /// bind time, because a shared pose is applied as
    /// `target_global[joint] * inverse_bind[joint]` and only one inverse bind
    /// per joint survives. Parts whose bind pose differs by more than
    /// `tolerance` are rejected by name rather than silently skinned to a
    /// slightly wrong body, which reads on screen as a seam that pulls apart
    /// only in certain poses and is miserable to track down later.
    ///
    /// On success the model adopts `target` as its skeleton and reports how many
    /// influences were rewritten. On failure the model is left untouched.
    pub fn rebind_skin(&mut self, target: &Skeleton, tolerance: f32) -> Result<usize, String> {
        let Some(source) = &self.skeleton else {
            return Err("model has no skeleton to rebind".into());
        };

        let index_of: std::collections::HashMap<&str, usize> = target
            .joints
            .iter()
            .enumerate()
            .map(|(i, j)| (j.name.as_str(), i))
            .collect();

        let remap: Vec<Option<u32>> = source
            .joints
            .iter()
            .map(|j| index_of.get(j.name.as_str()).map(|&i| i as u32))
            .collect();

        // Which joints this part actually deforms with.
        //
        // Only these matter. A part's skeleton also carries the chain above what
        // it deforms - a head exports the spine and pelvis purely so its neck
        // has somewhere to hang from - and those joints have no skin cluster,
        // hence no bind matrix, only a placeholder identity. Checking them
        // against the target's real bind matrices compares a measurement to a
        // placeholder and rejects every part that is doing nothing wrong.
        let mut weighted = vec![false; source.joints.len()];
        for prim in self.primitives.iter().filter(|p| p.skinned) {
            for v in &prim.mesh.vertices {
                for k in 0..4 {
                    if v.weights[k] > 0.0 {
                        let j = v.joints[k] as usize;
                        if j >= source.joints.len() {
                            return Err(format!(
                                "a vertex is weighted to joint {j}, past the end of this part's \
                                 {}-joint skeleton",
                                source.joints.len()
                            ));
                        }
                        weighted[j] = true;
                    }
                }
            }
        }

        // Validate before touching a vertex, so a rejected part is left exactly
        // as it was.
        for (i, joint) in source.joints.iter().enumerate() {
            if !weighted[i] {
                continue;
            }
            let Some(to) = remap[i] else {
                return Err(format!(
                    "joint {} carries weight in this part but is absent from the target skeleton",
                    joint.name
                ));
            };
            let drift = (joint.inverse_bind - target.joints[to as usize].inverse_bind)
                .to_cols_array()
                .iter()
                .fold(0.0f32, |acc, d| acc.max(d.abs()));
            if drift > tolerance {
                return Err(format!(
                    "joint {} binds differently in this part than in the target skeleton \
                     (largest difference {drift}, tolerance {tolerance})",
                    joint.name
                ));
            }
        }

        let mut rewritten = 0;
        for prim in self.primitives.iter_mut().filter(|p| p.skinned) {
            for v in &mut prim.mesh.vertices {
                for k in 0..4 {
                    if v.weights[k] > 0.0 {
                        v.joints[k] = remap[v.joints[k] as usize].expect("validated above");
                        rewritten += 1;
                    } else {
                        // An unweighted slot indexes nothing meaningful. Point it
                        // at a joint that exists so the palette lookup stays in
                        // bounds whatever the source file left here.
                        v.joints[k] = 0;
                    }
                }
            }
        }

        self.skeleton = Some(target.clone());
        Ok(rewritten)
    }

    /// Load the clips from `path` and add them to this model, retargeted onto
    /// its skeleton.
    ///
    /// A moveset ships as one file per clip, authored against a rig that is not
    /// the character's - a library of a few hundred animations is exported once
    /// and every character in the project is expected to borrow from it. This
    /// reads such a file, discards its geometry and its skeleton, and keeps the
    /// motion addressed to bones this model actually has.
    ///
    /// Returns the number of clips added. Clips that retarget to nothing are
    /// reported and skipped rather than aborting the whole file, so one bad
    /// export does not cost a library.
    pub fn add_clips_from(
        &mut self,
        path: &str,
        source_rest: &Skeleton,
        rename: &[(&str, &str)],
        translate: &[&str],
    ) -> Result<usize, String> {
        let Some(target) = &self.skeleton else {
            return Err(format!("cannot add clips to {path}: model has no skeleton"));
        };
        let library = Model::load(path)?;
        let Some(source) = &library.skeleton else {
            return Err(format!("{path} has no skeleton to retarget from"));
        };
        let opts = Retarget {
            source,
            source_rest,
            target,
            rename,
            translate,
        };

        let mut added = Vec::new();
        for clip in &library.clips {
            match clip.retarget(&opts) {
                Ok(c) => added.push(c),
                Err(e) => eprintln!("aurora: {e}"),
            }
        }
        let n = added.len();
        self.clips.extend(added);
        Ok(n)
    }

    /// Load just the skeleton from a file, for use as a retargeting reference.
    ///
    /// An animation pack ships the rig its clips were authored on precisely so
    /// this is possible; the clips themselves carry no usable rest pose.
    pub fn load_skeleton(path: &str) -> Result<Skeleton, String> {
        Model::load(path)?
            .skeleton
            .ok_or_else(|| format!("{path} has no skeleton"))
    }

    /// Load a model by file extension (`.gltf`/`.glb`, `.obj`, or `.fbx`).
    pub fn load(path: &str) -> Result<Model, String> {
        let lower = path.to_ascii_lowercase();
        if lower.ends_with(".obj") {
            Self::load_obj(path)
        } else if lower.ends_with(".fbx") {
            crate::fbx::load(path)
        } else {
            Self::load_gltf(path)
        }
    }

    /// Load a static OBJ mesh (no skeleton/animation).
    pub fn load_obj(path: &str) -> Result<Model, String> {
        let (models, materials) = tobj::load_obj(
            path,
            &tobj::LoadOptions {
                triangulate: true,
                single_index: true,
                ..Default::default()
            },
        )
        .map_err(|e| format!("load obj {path}: {e}"))?;
        let materials = materials.unwrap_or_default();

        let mut primitives = Vec::new();
        for m in &models {
            let mesh = &m.mesh;
            let mut data = MeshData::default();
            let count = mesh.positions.len() / 3;
            let has_normals = mesh.normals.len() == mesh.positions.len();
            let has_uv = mesh.texcoords.len() / 2 == count;
            for i in 0..count {
                let pos = [
                    mesh.positions[i * 3],
                    mesh.positions[i * 3 + 1],
                    mesh.positions[i * 3 + 2],
                ];
                let normal = if has_normals {
                    [
                        mesh.normals[i * 3],
                        mesh.normals[i * 3 + 1],
                        mesh.normals[i * 3 + 2],
                    ]
                } else {
                    [0.0, 1.0, 0.0]
                };
                let uv = if has_uv {
                    [mesh.texcoords[i * 2], 1.0 - mesh.texcoords[i * 2 + 1]]
                } else {
                    [0.0, 0.0]
                };
                data.vertices.push(Vertex::new(pos, normal, uv));
            }
            data.indices = mesh.indices.clone();
            if !has_normals {
                data.compute_flat_normals();
            }
            data.compute_tangents();
            let base_color = mesh
                .material_id
                .and_then(|id| materials.get(id))
                .and_then(|mat| mat.diffuse)
                .map(|d| [d[0], d[1], d[2], 1.0])
                .unwrap_or([0.8, 0.8, 0.8, 1.0]);
            primitives.push(Primitive {
                mesh: data,
                material: mesh
                    .material_id
                    .and_then(|id| materials.get(id))
                    .map(|m| m.name.clone())
                    .unwrap_or_default(),
                base_color,
                metallic: 0.0,
                roughness: 0.9,
                emissive: [0.0; 3],
                texture: None,
                normal_tex: None,
                mr_tex: None,
                emissive_tex: None,
                skinned: false,
            });
        }
        Ok(Model {
            primitives,
            skeleton: None,
            clips: Vec::new(),
        })
    }

    /// Load a glTF/GLB model with materials, skeleton, and animation clips.
    pub fn load_gltf(path: &str) -> Result<Model, String> {
        let (doc, buffers, images) =
            gltf::import(path).map_err(|e| format!("load gltf {path}: {e}"))?;
        let buf = |b: gltf::Buffer| buffers.get(b.index()).map(|d| &d.0[..]);

        // Node world transforms, needed both for baking static geometry and for
        // the transform above the joint hierarchy (see Skeleton::root).
        let globals = node_global_transforms(&doc);

        // --- skeleton (first skin) ---
        // Map glTF node index -> joint index, and record each joint's parent.
        let mut node_to_joint: HashMap<usize, usize> = HashMap::new();
        let mut skeleton = None;
        if let Some(skin) = doc.skins().next() {
            let joints_nodes: Vec<gltf::Node> = skin.joints().collect();
            for (ji, n) in joints_nodes.iter().enumerate() {
                node_to_joint.insert(n.index(), ji);
            }
            // Parent of each node (only matters within the joint set).
            let mut node_parent: HashMap<usize, usize> = HashMap::new();
            for n in doc.nodes() {
                for c in n.children() {
                    node_parent.insert(c.index(), n.index());
                }
            }
            let reader = skin.reader(buf);
            let ibm: Vec<Mat4> = reader
                .read_inverse_bind_matrices()
                .map(|it| it.map(|m| Mat4::from_cols_array_2d(&m)).collect())
                .unwrap_or_else(|| vec![Mat4::IDENTITY; joints_nodes.len()]);

            // Local transform of one node, as a matrix.
            let node_local = |ni: usize| -> (Vec3, Quat, Vec3) {
                let (t, r, sc) = doc
                    .nodes()
                    .nth(ni)
                    .map(|n| n.transform().decomposed())
                    .unwrap_or(([0.0; 3], [0.0, 0.0, 0.0, 1.0], [1.0; 3]));
                (Vec3::from(t), Quat::from_array(r), Vec3::from(sc))
            };
            let node_name = |ni: usize| -> String {
                doc.nodes()
                    .nth(ni)
                    .and_then(|n| n.name().map(String::from))
                    .unwrap_or_default()
            };

            // The skeleton spans the FULL BONE TREE, not just the skin's joint list.
            //
            // A skin lists only the joints that deform vertices. Real rigs put other
            // nodes in between: an exporter emits L_Foot under L_Calf under L_Thigh under
            // Pelvis while weighting vertices only to the twist bones, so L_Thigh and
            // L_Calf are absent from the joint list. Building the skeleton from that list
            // alone left every limb bone a SIBLING of the hip - the chain was gone, so an
            // animation channel on L_Thigh had nowhere to go and posing a thigh could not
            // carry its calf and foot.
            //
            // Skin joints keep indices 0..N-1 exactly as the skin orders them, because
            // that is what the mesh's JOINTS_0 attribute indexes. Intermediate and
            // ancestor nodes are APPENDED after them, so skinning is untouched while the
            // parent chain becomes real and every bone is addressable by name.
            let mut order: Vec<usize> = joints_nodes.iter().map(|n| n.index()).collect();
            let mut seen: HashSet<usize> = order.iter().copied().collect();
            let mut queue: Vec<usize> = order.clone();
            while let Some(ni) = queue.pop() {
                let mut up = node_parent.get(&ni).copied();
                while let Some(pi) = up {
                    if !seen.insert(pi) {
                        break;
                    }
                    order.push(pi);
                    up = node_parent.get(&pi).copied();
                }
            }
            let index_of: HashMap<usize, usize> =
                order.iter().enumerate().map(|(i, n)| (*n, i)).collect();
            // Animation channels are resolved through this map too. It held only the
            // SKIN's joints, so a channel targeting a bone that deforms nothing itself -
            // an upper arm or a thigh, whose twist children carry the weights - was
            // silently dropped and that limb never moved. The skeleton spans those bones,
            // so the channel map must span them as well or the clip is half applied.
            node_to_joint = index_of.clone();

            let mut joints = Vec::with_capacity(order.len());
            for (ji, ni) in order.iter().enumerate() {
                let (t, r, sc) = node_local(*ni);
                joints.push(Joint {
                    parent: node_parent.get(ni).and_then(|pi| index_of.get(pi)).copied(),
                    // Only skin joints deform anything; the appended bones exist to carry
                    // the chain, so their bind matrix is never read.
                    inverse_bind: if ji < joints_nodes.len() {
                        ibm.get(ji).copied().unwrap_or(Mat4::IDENTITY)
                    } else {
                        Mat4::IDENTITY
                    },
                    t,
                    r,
                    s: sc,
                    name: node_name(*ni),
                });
            }

            skeleton = Some(Skeleton { joints });
        }

        // --- primitives ---
        let mut primitives = Vec::new();
        for node in doc.nodes() {
            let Some(mesh) = node.mesh() else { continue };
            let is_skinned = node.skin().is_some();
            let world = globals
                .get(&node.index())
                .copied()
                .unwrap_or(Mat4::IDENTITY);
            let normal_world = Mat4::from_mat3(glam::Mat3::from_mat4(world).inverse().transpose());
            for prim in mesh.primitives() {
                let reader = prim.reader(buf);
                let positions: Vec<[f32; 3]> = match reader.read_positions() {
                    Some(p) => p.collect(),
                    None => continue,
                };
                let normals: Vec<[f32; 3]> = reader
                    .read_normals()
                    .map(|n| n.collect())
                    .unwrap_or_else(|| vec![[0.0, 1.0, 0.0]; positions.len()]);
                let uvs: Vec<[f32; 2]> = reader
                    .read_tex_coords(0)
                    .map(|t| t.into_f32().collect())
                    .unwrap_or_else(|| vec![[0.0, 0.0]; positions.len()]);
                let joints_attr: Option<Vec<[u16; 4]>> =
                    reader.read_joints(0).map(|j| j.into_u16().collect());
                let weights_attr: Option<Vec<[f32; 4]>> =
                    reader.read_weights(0).map(|w| w.into_f32().collect());

                let mut data = MeshData::default();
                for i in 0..positions.len() {
                    let (pos, normal) = if is_skinned {
                        (positions[i], normals[i])
                    } else {
                        // Bake the node's world transform into static geometry so
                        // the model sits where the file places it; the caller's
                        // object matrix is then applied on top.
                        let p = world.transform_point3(Vec3::from(positions[i]));
                        let n = normal_world
                            .transform_vector3(Vec3::from(normals[i]))
                            .normalize_or_zero();
                        (p.into(), n.into())
                    };
                    let mut v = Vertex::new(pos, normal, uvs[i]);
                    if let (Some(j), Some(w)) = (&joints_attr, &weights_attr) {
                        v.joints = [
                            j[i][0] as u32,
                            j[i][1] as u32,
                            j[i][2] as u32,
                            j[i][3] as u32,
                        ];
                        // Remap glTF skin-local joint indices: read_joints already
                        // indexes into the skin's joint list, which is our order.
                        let ww = w[i];
                        let sum = ww[0] + ww[1] + ww[2] + ww[3];
                        v.weights = if sum > 0.0 {
                            [ww[0] / sum, ww[1] / sum, ww[2] / sum, ww[3] / sum]
                        } else {
                            [1.0, 0.0, 0.0, 0.0]
                        };
                    }
                    data.vertices.push(v);
                }
                data.indices = match reader.read_indices() {
                    Some(idx) => idx.into_u32().collect(),
                    None => (0..positions.len() as u32).collect(),
                };
                // Read tangents if present; otherwise compute them from UVs.
                match reader.read_tangents() {
                    Some(ts) => {
                        for (v, t) in data.vertices.iter_mut().zip(ts) {
                            v.tangent = t;
                        }
                    }
                    None => data.compute_tangents(),
                }

                let material = prim.material();
                let pbr = material.pbr_metallic_roughness();
                let tex_of = |info: gltf::texture::Texture| -> Option<crate::model::Tex> {
                    rgba_from_gltf(images.get(info.source().index())?)
                };
                let texture = pbr.base_color_texture().and_then(|i| tex_of(i.texture()));
                let mr_tex = pbr
                    .metallic_roughness_texture()
                    .and_then(|i| tex_of(i.texture()));
                let normal_tex = material.normal_texture().and_then(|i| tex_of(i.texture()));
                let emissive_tex = material
                    .emissive_texture()
                    .and_then(|i| tex_of(i.texture()));
                primitives.push(Primitive {
                    mesh: data,
                    material: material.name().unwrap_or_default().to_string(),
                    base_color: pbr.base_color_factor(),
                    metallic: pbr.metallic_factor(),
                    roughness: pbr.roughness_factor(),
                    emissive: material.emissive_factor(),
                    texture,
                    normal_tex,
                    mr_tex,
                    emissive_tex,
                    skinned: is_skinned,
                });
            }
        }

        // --- animation clips ---
        let mut clips = Vec::new();
        for anim in doc.animations() {
            let mut channels = Vec::new();
            let mut duration = 0.0f32;
            for ch in anim.channels() {
                let target = ch.target();
                let Some(&joint) = node_to_joint.get(&target.node().index()) else {
                    continue;
                };
                let path = match target.property() {
                    gltf::animation::Property::Translation => Path::Translation,
                    gltf::animation::Property::Rotation => Path::Rotation,
                    gltf::animation::Property::Scale => Path::Scale,
                    gltf::animation::Property::MorphTargetWeights => continue,
                };
                let interp = match ch.sampler().interpolation() {
                    gltf::animation::Interpolation::Step => Interp::Step,
                    _ => Interp::Linear,
                };
                let reader = ch.reader(buf);
                let times: Vec<f32> = match reader.read_inputs() {
                    Some(t) => t.collect(),
                    None => continue,
                };
                let values: Vec<f32> = match reader.read_outputs() {
                    Some(gltf::animation::util::ReadOutputs::Translations(it)) => {
                        it.flat_map(|v| v.into_iter()).collect()
                    }
                    Some(gltf::animation::util::ReadOutputs::Scales(it)) => {
                        it.flat_map(|v| v.into_iter()).collect()
                    }
                    Some(gltf::animation::util::ReadOutputs::Rotations(it)) => {
                        it.into_f32().flat_map(|v| v.into_iter()).collect()
                    }
                    _ => continue,
                };
                if let Some(&last) = times.last() {
                    duration = duration.max(last);
                }
                channels.push(Channel {
                    joint,
                    path,
                    interp,
                    times,
                    values,
                });
            }
            let name = anim.name().unwrap_or("clip").to_string();
            clips.push(Clip {
                name,
                duration,
                channels,
            });
        }

        Ok(Model {
            primitives,
            skeleton,
            clips,
        })
    }
}

/// Compute every node's global transform by walking the scene hierarchy.
fn node_global_transforms(doc: &gltf::Document) -> HashMap<usize, Mat4> {
    let mut out = HashMap::new();
    for scene in doc.scenes() {
        for node in scene.nodes() {
            walk(&node, Mat4::IDENTITY, &mut out);
        }
    }
    out
}

fn walk(node: &gltf::Node, parent: Mat4, out: &mut HashMap<usize, Mat4>) {
    let local = Mat4::from_cols_array_2d(&node.transform().matrix());
    let global = parent * local;
    out.insert(node.index(), global);
    for child in node.children() {
        walk(&child, global, out);
    }
}

/// Convert a decoded glTF image to tightly-packed RGBA8.
fn rgba_from_gltf(img: &gltf::image::Data) -> Option<(Vec<u8>, u32, u32)> {
    use gltf::image::Format;
    let (w, h) = (img.width, img.height);
    let px = &img.pixels;
    let rgba = match img.format {
        Format::R8G8B8A8 => px.clone(),
        Format::R8G8B8 => px
            .chunks_exact(3)
            .flat_map(|c| [c[0], c[1], c[2], 255])
            .collect(),
        Format::R8 => px.iter().flat_map(|&v| [v, v, v, 255]).collect(),
        Format::R8G8 => px
            .chunks_exact(2)
            .flat_map(|c| [c[0], c[0], c[0], c[1]])
            .collect(),
        _ => return None,
    };
    Some((rgba, w, h))
}
