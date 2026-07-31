//! FBX import, via the `ufbx` parser.
//!
//! FBX is the format art actually arrives in - every Synty, Mixamo and DCC
//! export is FBX - so Aurora reads it directly rather than making the artist
//! round-trip through another tool. The output is the same [`Model`] the glTF
//! and OBJ paths produce, so nothing downstream knows or cares which importer
//! ran.
//!
//! Two shapes of file matter and both are first class:
//!
//! - **A skinned character**: meshes with skin deformers, plus the bone tree.
//! - **A clip with no geometry at all**: an animation-only export, which has
//!   anim stacks and a bone tree but zero meshes. Synty's animation packs are
//!   all of this shape, so a loader that assumed geometry would reject the
//!   entire animation library.

use std::collections::{HashMap, HashSet};
use std::path::{Path as FsPath, PathBuf};

use glam::{Mat4, Quat, Vec3};

use crate::mesh::{MeshData, Vertex};
use crate::model::{Channel, Clip, Interp, Joint, Model, Path, Primitive, Skeleton, Tex};

/// Bone influences kept per vertex. Matches the renderer's vertex format; the
/// lowest-weight influences beyond this are dropped and the rest renormalized,
/// which is what every engine does and what the source art is authored against.
const MAX_INFLUENCES: usize = 4;

/// Load an FBX file into a [`Model`].
pub fn load(path: &str) -> Result<Model, String> {
    let opts = ufbx::LoadOpts {
        // Aurora is right-handed, +Y up, +Z toward the viewer - the same basis
        // the glTF path produces, so a model imported either way sits the same
        // way up and one animation set drives both.
        target_axes: ufbx::CoordinateAxes {
            right: ufbx::CoordinateAxis::PositiveX,
            up: ufbx::CoordinateAxis::PositiveY,
            front: ufbx::CoordinateAxis::PositiveZ,
        },
        // Source art is authored in centimetres far more often than not.
        // Normalising to metres here means physics extents, camera distances and
        // movement speeds are all in one unit and nobody scales by 0.01 by hand.
        target_unit_meters: 1.0,
        // Fold the conversion into node transforms rather than rewriting vertex
        // data, which `ModifyGeometry` cannot do for geometry instanced at more
        // than one transform.
        //
        // A consequence worth knowing before it surprises someone: a skinned
        // mesh's vertices stay in the source file's own bind space, so raw
        // geometry from a centimetre export still reads in centimetres while
        // joint transforms read in metres. That is not a mismatch - the skin
        // clusters' bind matrices carry the conversion, so `global *
        // inverse_bind` lands in metres. [`Model::bind_pose_bounds`] is what
        // measures a skinned model; `MeshData::bounds` on one would report the
        // unskinned bind-space extent and size a collider a hundred times too
        // large.
        space_conversion: ufbx::SpaceConversion::AdjustTransforms,
        generate_missing_normals: true,
        ..Default::default()
    };
    let root = ufbx::load_file(path, opts)
        .map_err(|e| format!("load fbx {path}: {}", &*e.description))?;
    let scene: &ufbx::Scene = &root;
    let dir = FsPath::new(path).parent().unwrap_or(FsPath::new("."));

    let (skeleton, joint_of) = build_skeleton(scene);
    let primitives = build_primitives(scene, &joint_of, dir);
    let clips = build_clips(scene, &joint_of);

    Ok(Model {
        primitives,
        skeleton,
        clips,
    })
}

/// Element ids of every node that belongs in the skeleton, mapped to its joint
/// index, alongside the skeleton itself.
type JointIndex = HashMap<u32, usize>;

/// Build the bone tree.
///
/// The skeleton spans more than the bones that deform vertices. A rig routinely
/// puts plain transform nodes between real bones, and a skin cluster only names
/// the ones carrying weights - so collecting just those leaves every limb a
/// sibling of the hip and an animation channel on an intermediate node lands
/// nowhere. Seed from skin clusters *and* nodes flagged as bones, then close the
/// set upward over parents so the chain is intact.
///
/// Nodes are numbered in depth-first tree order, which keeps a parent's index
/// below its children's. The pose solver memoizes and so does not require that,
/// but it makes a dumped skeleton readable and diffable against the source rig.
fn build_skeleton(scene: &ufbx::Scene) -> (Option<Skeleton>, JointIndex) {
    let mut wanted: HashSet<u32> = HashSet::new();

    for skin in &scene.skin_deformers {
        for cluster in &skin.clusters {
            if let Some(bone) = &cluster.bone_node {
                wanted.insert(bone.element.element_id);
            }
        }
    }
    for node in &scene.nodes {
        if !node.is_root && node.bone.is_some() {
            wanted.insert(node.element.element_id);
        }
    }
    if wanted.is_empty() {
        return (None, HashMap::new());
    }

    // Close upward over parents. The synthetic scene root is never a joint - it
    // is ufbx's container, not part of the rig.
    let seeds: Vec<u32> = wanted.iter().copied().collect();
    for id in seeds {
        let Some(node) = scene.nodes.iter().find(|n| n.element.element_id == id) else {
            continue;
        };
        let mut up: Option<&ufbx::Node> = node.parent.as_ref().map(|p| &**p);
        while let Some(p) = up {
            if p.is_root || !wanted.insert(p.element.element_id) {
                break;
            }
            up = p.parent.as_ref().map(|p| &**p);
        }
    }

    let mut order: Vec<&ufbx::Node> = Vec::new();
    let mut joint_of: JointIndex = HashMap::new();
    walk_joints(&scene.root_node, &wanted, &mut order, &mut joint_of);

    // A skin cluster's `geometry_to_bone` is exactly the inverse-bind matrix the
    // skinning shader wants: it takes a vertex from mesh space into the bone's
    // space at bind time. Joints that deform nothing (the intermediate nodes
    // added above) never have their bind matrix read, so identity is correct.
    let mut inverse_bind: HashMap<u32, Mat4> = HashMap::new();
    for skin in &scene.skin_deformers {
        for cluster in &skin.clusters {
            if let Some(bone) = &cluster.bone_node {
                inverse_bind.insert(bone.element.element_id, matrix(&cluster.geometry_to_bone));
            }
        }
    }

    let bind_of: Vec<Option<Mat4>> = order
        .iter()
        .map(|n| inverse_bind.get(&n.element.element_id).copied())
        .collect();
    let parent_of: Vec<Option<usize>> = order
        .iter()
        .map(|n| {
            n.parent
                .as_ref()
                .and_then(|p| joint_of.get(&p.element.element_id))
                .copied()
        })
        .collect();

    let joints: Vec<Joint> = order
        .iter()
        .enumerate()
        .map(|(i, n)| {
            let x = &n.local_transform;
            let mut t = vec3(x.translation);
            let mut r = Quat::from_xyzw(
                x.rotation.x as f32,
                x.rotation.y as f32,
                x.rotation.z as f32,
                x.rotation.w as f32,
            );
            let mut s = vec3(x.scale);

            // Prefer the bind pose recorded by the skin clusters over the node's
            // own transform.
            //
            // An FBX node transform is whatever pose the rig happened to be in
            // when it was exported, which is frequently not the bind pose and is
            // sometimes nothing at all: Synty's reference character ships with
            // every bone collapsed onto the hip, so a skeleton read from node
            // transforms alone skins the whole mesh to a single point. The skin
            // clusters always carry the truth, because that is what the weights
            // were authored against.
            //
            // Deriving the local from a parent and child bind matrix keeps the
            // result in bind space, where it is directly comparable with the
            // node-derived transforms it replaces. Going through world space
            // instead would mix bind-space centimetres with the metres the node
            // chain has already been converted to.
            if let (Some(bind), Some(parent)) = (bind_of[i], parent_of[i]) {
                if let Some(parent_bind) = bind_of[parent] {
                    let (ls, lr, lt) = (parent_bind * bind.inverse()).to_scale_rotation_translation();
                    t = lt;
                    r = lr;
                    s = ls;
                }
            }

            Joint {
                parent: parent_of[i],
                inverse_bind: bind_of[i].unwrap_or(Mat4::IDENTITY),
                t,
                r,
                s,
                name: n.element.name.to_string(),
            }
        })
        .collect();

    (Some(Skeleton { joints }), joint_of)
}

fn walk_joints<'a>(
    node: &'a ufbx::Node,
    wanted: &HashSet<u32>,
    order: &mut Vec<&'a ufbx::Node>,
    joint_of: &mut JointIndex,
) {
    for child in &node.children {
        if wanted.contains(&child.element.element_id) {
            joint_of.insert(child.element.element_id, order.len());
            order.push(child);
        }
        walk_joints(child, wanted, order, joint_of);
    }
}

/// Triangulate every mesh in the scene, split by material, with skin weights
/// resolved against the skeleton.
fn build_primitives(scene: &ufbx::Scene, joint_of: &JointIndex, dir: &FsPath) -> Vec<Primitive> {
    let mut out = Vec::new();

    for node in &scene.nodes {
        if node.is_root {
            continue;
        }
        let Some(mesh) = &node.mesh else { continue };
        let mesh: &ufbx::Mesh = mesh;

        // One skin deformer is the norm; a second is a stacked rig Aurora does
        // not model, so the first is authoritative.
        let skin: Option<&ufbx::SkinDeformer> = if mesh.skin_deformers.count > 0 {
            Some(&mesh.skin_deformers[0])
        } else {
            None
        };
        let skinned = skin.is_some();

        // Static geometry is baked into world space so the model sits where the
        // file places it. Skinned geometry must stay in its own space, because
        // the bind matrices above are expressed relative to exactly that space.
        let to_world = matrix(&node.geometry_to_world);
        let normal_to_world = Mat4::from_mat3(glam::Mat3::from_mat4(to_world).inverse().transpose());

        let bucket_count = mesh.materials.count.max(1);
        let mut buckets = vec![MeshData::default(); bucket_count];
        let mut dedup: Vec<HashMap<(u32, u32, u32), u32>> = vec![HashMap::new(); bucket_count];

        let mut tri = vec![0u32; (mesh.max_face_triangles * 3).max(3)];
        for (fi, face) in mesh.faces.iter().enumerate() {
            let tris = ufbx::triangulate_face(&mut tri, mesh, *face);
            let bucket = if mesh.face_material.count > 0 {
                (mesh.face_material[fi] as usize).min(bucket_count - 1)
            } else {
                0
            };
            for corner in &tri[..tris as usize * 3] {
                let ci = *corner as usize;
                let key = (
                    mesh.vertex_position.indices[ci],
                    if mesh.vertex_normal.exists {
                        mesh.vertex_normal.indices[ci]
                    } else {
                        u32::MAX
                    },
                    if mesh.vertex_uv.exists {
                        mesh.vertex_uv.indices[ci]
                    } else {
                        u32::MAX
                    },
                );
                let next = buckets[bucket].vertices.len() as u32;
                let index = match dedup[bucket].get(&key) {
                    Some(&i) => i,
                    None => {
                        let v = make_vertex(mesh, ci, skin, joint_of, skinned, to_world, normal_to_world);
                        buckets[bucket].vertices.push(v);
                        dedup[bucket].insert(key, next);
                        next
                    }
                };
                buckets[bucket].indices.push(index);
            }
        }

        for (i, mut data) in buckets.into_iter().enumerate() {
            if data.vertices.is_empty() {
                continue;
            }
            if !mesh.vertex_normal.exists {
                data.compute_flat_normals();
            }
            data.compute_tangents();
            let material = mesh.materials.get(i).map(|m| &**m);
            let (base_color, texture) = material_of(material, dir);
            out.push(Primitive {
                mesh: data,
                material: material.map(|m| m.element.name.to_string()).unwrap_or_default(),
                base_color,
                metallic: 0.0,
                roughness: 0.9,
                emissive: [0.0; 3],
                texture,
                normal_tex: None,
                mr_tex: None,
                emissive_tex: None,
                skinned,
            });
        }
    }

    out
}

fn make_vertex(
    mesh: &ufbx::Mesh,
    ci: usize,
    skin: Option<&ufbx::SkinDeformer>,
    joint_of: &JointIndex,
    skinned: bool,
    to_world: Mat4,
    normal_to_world: Mat4,
) -> Vertex {
    let p = vec3(mesh.vertex_position[ci]);
    let n = if mesh.vertex_normal.exists {
        vec3(mesh.vertex_normal[ci])
    } else {
        Vec3::Y
    };
    // FBX puts the UV origin at the bottom left; wgpu samples from the top
    // left, so V is flipped exactly as the OBJ path does.
    let uv = if mesh.vertex_uv.exists {
        let t = mesh.vertex_uv[ci];
        [t.x as f32, 1.0 - t.y as f32]
    } else {
        [0.0, 0.0]
    };

    let (pos, normal) = if skinned {
        (p.into(), n.into())
    } else {
        (
            to_world.transform_point3(p).into(),
            normal_to_world.transform_vector3(n).normalize_or_zero().into(),
        )
    };

    let mut v = Vertex::new(pos, normal, uv);
    if let Some(skin) = skin {
        let vertex_id = mesh.vertex_indices[ci] as usize;
        if vertex_id < skin.vertices.count {
            let sv = &skin.vertices[vertex_id];
            let begin = sv.weight_begin as usize;
            let mut influences: Vec<(u32, f32)> = (0..sv.num_weights as usize)
                .filter_map(|k| {
                    let w = &skin.weights[begin + k];
                    let cluster = &skin.clusters[w.cluster_index as usize];
                    let joint = cluster
                        .bone_node
                        .as_ref()
                        .and_then(|b| joint_of.get(&b.element.element_id))
                        .copied()?;
                    Some((joint as u32, w.weight as f32))
                })
                .collect();
            // Keep the heaviest influences: dropping the smallest is the least
            // visible truncation, and renormalising afterwards keeps the vertex
            // fully weighted rather than shrinking toward the origin.
            influences.sort_by(|a, b| b.1.total_cmp(&a.1));
            influences.truncate(MAX_INFLUENCES);
            let sum: f32 = influences.iter().map(|(_, w)| w).sum();
            if sum > 0.0 {
                for (slot, (joint, weight)) in influences.iter().enumerate() {
                    v.joints[slot] = *joint;
                    v.weights[slot] = weight / sum;
                }
                for slot in influences.len()..MAX_INFLUENCES {
                    v.joints[slot] = 0;
                    v.weights[slot] = 0.0;
                }
            }
        }
    }
    v
}

/// Bake every animation stack into per-joint TRS channels.
///
/// Baking rather than reading raw curves is deliberate: FBX curves carry
/// pre/post rotations, rotation orders and inherit modes that must be composed
/// before they mean anything as a local transform. `ufbx` already resolves all
/// of that, and it hands back exactly the sampled TRS tracks Aurora's clip
/// format stores.
fn build_clips(scene: &ufbx::Scene, joint_of: &JointIndex) -> Vec<Clip> {
    let mut clips = Vec::new();

    for stack in &scene.anim_stacks {
        let opts = ufbx::BakeOpts {
            // Clips are addressed from zero, so a stack that happens to start at
            // frame 2 does not make every consumer subtract an offset.
            trim_start_time: true,
            ..Default::default()
        };
        let Ok(baked) = ufbx::bake_anim(scene, &stack.anim, opts) else {
            continue;
        };

        let mut channels = Vec::new();
        let mut duration = 0.0f32;
        for node in &baked.nodes {
            let Some(scene_node) = scene.nodes.get(node.typed_id as usize) else {
                continue;
            };
            let Some(&joint) = joint_of.get(&scene_node.element.element_id) else {
                continue;
            };

            let mut push = |path: Path, times: Vec<f32>, values: Vec<f32>| {
                if let Some(&last) = times.last() {
                    duration = duration.max(last);
                }
                if !times.is_empty() {
                    channels.push(Channel {
                        joint,
                        path,
                        interp: Interp::Linear,
                        times,
                        values,
                    });
                }
            };

            push(
                Path::Translation,
                node.translation_keys.iter().map(|k| k.time as f32).collect(),
                node.translation_keys
                    .iter()
                    .flat_map(|k| [k.value.x as f32, k.value.y as f32, k.value.z as f32])
                    .collect(),
            );
            push(
                Path::Rotation,
                node.rotation_keys.iter().map(|k| k.time as f32).collect(),
                node.rotation_keys
                    .iter()
                    .flat_map(|k| {
                        [
                            k.value.x as f32,
                            k.value.y as f32,
                            k.value.z as f32,
                            k.value.w as f32,
                        ]
                    })
                    .collect(),
            );
            push(
                Path::Scale,
                node.scale_keys.iter().map(|k| k.time as f32).collect(),
                node.scale_keys
                    .iter()
                    .flat_map(|k| [k.value.x as f32, k.value.y as f32, k.value.z as f32])
                    .collect(),
            );
        }

        if !channels.is_empty() {
            clips.push(Clip {
                name: stack.element.name.to_string(),
                duration,
                channels,
            });
        }
    }

    clips
}

/// Base colour and base-colour texture for one material.
fn material_of(material: Option<&ufbx::Material>, dir: &FsPath) -> ([f32; 4], Option<Tex>) {
    let Some(m) = material else {
        return ([0.8, 0.8, 0.8, 1.0], None);
    };

    // Prefer the PBR interpretation; fall back to the classic FBX Lambert/Phong
    // diffuse, which is all an older export carries.
    let map = if m.pbr.base_color.has_value {
        Some(&m.pbr.base_color)
    } else if m.fbx.diffuse_color.has_value {
        Some(&m.fbx.diffuse_color)
    } else {
        None
    };
    let base_color = map
        .map(|c| {
            [
                c.value_vec4.x as f32,
                c.value_vec4.y as f32,
                c.value_vec4.z as f32,
                1.0,
            ]
        })
        .unwrap_or([0.8, 0.8, 0.8, 1.0]);

    let tex = map
        .and_then(|c| c.texture.as_ref().map(|t| &**t))
        .or_else(|| {
            m.textures
                .iter()
                .find(|t| {
                    let p = &*t.material_prop;
                    p == "DiffuseColor" || p == "BaseColor"
                })
                .map(|t| &*t.texture)
        })
        .and_then(|t| load_texture(t, dir));

    (base_color, tex)
}

/// How far up from the model file to look for a texture directory.
///
/// A pack commonly nests models a couple of levels below the folder its textures
/// sit in - `Source_Files/FBX/ModularParts_Unreal/x.fbx` beside
/// `Source_Files/Textures/atlas.png` is three. Walking up rather than hardcoding
/// a depth means one rule covers every layout; the bound stops a model at a
/// drive root from scanning the whole filesystem.
const TEXTURE_SEARCH_DEPTH: usize = 4;

/// Find and decode a texture referenced by an FBX file.
///
/// The path recorded in an FBX is the one that existed on the machine that
/// exported it, so it almost never resolves here. What does survive is the file
/// NAME, so the search is: the recorded paths first, in case the file really was
/// authored in place, then that bare name in each ancestor directory and in a
/// `Textures` folder beside it.
fn load_texture(tex: &ufbx::Texture, dir: &FsPath) -> Option<Tex> {
    // Embedded content wins outright: it needs no search and cannot go stale.
    if !tex.content.is_empty() {
        if let Some(t) = crate::model::decode_texture(&tex.content) {
            return Some(t);
        }
    }

    let name = FsPath::new(&*tex.relative_filename)
        .file_name()
        .or_else(|| FsPath::new(&*tex.filename).file_name())
        .or_else(|| FsPath::new(&*tex.absolute_filename).file_name())?;

    let mut candidates: Vec<PathBuf> = vec![
        PathBuf::from(&*tex.absolute_filename),
        dir.join(&*tex.relative_filename),
    ];
    let mut up = Some(dir);
    for _ in 0..TEXTURE_SEARCH_DEPTH {
        let Some(at) = up else { break };
        candidates.push(at.join(name));
        candidates.push(at.join("Textures").join(name));
        up = at.parent();
    }

    candidates
        .iter()
        .filter(|p| p.is_file())
        .find_map(|p| crate::model::load_texture_file(&p.to_string_lossy()).ok())
}

/// ufbx stores a 3x4 affine matrix column by column; widen it to a 4x4.
fn matrix(m: &ufbx::Matrix) -> Mat4 {
    Mat4::from_cols_array(&[
        m.m00 as f32,
        m.m10 as f32,
        m.m20 as f32,
        0.0,
        m.m01 as f32,
        m.m11 as f32,
        m.m21 as f32,
        0.0,
        m.m02 as f32,
        m.m12 as f32,
        m.m22 as f32,
        0.0,
        m.m03 as f32,
        m.m13 as f32,
        m.m23 as f32,
        1.0,
    ])
}

fn vec3(v: ufbx::Vec3) -> Vec3 {
    Vec3::new(v.x as f32, v.y as f32, v.z as f32)
}
