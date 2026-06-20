//! glTF (.gltf/.glb) and OBJ model loading: meshes, materials (base color +
//! texture), skeletons (joints, inverse-bind matrices, hierarchy), and skeletal
//! animation clips.

use std::collections::HashMap;

use glam::{Mat4, Quat, Vec3};

use crate::mesh::{MeshData, Vertex};

/// A tightly-packed RGBA8 texture: `(pixels, w, h)`.
pub type Tex = (Vec<u8>, u32, u32);

/// A drawable piece of a model: geometry plus a PBR material.
pub struct Primitive {
    pub mesh: MeshData,
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
pub struct Skeleton {
    pub joints: Vec<Joint>,
}

impl Skeleton {
    pub fn joint_count(&self) -> usize {
        self.joints.len()
    }
}

/// Which transform component an animation channel drives.
#[derive(Clone, Copy, PartialEq)]
pub enum Path {
    Translation,
    Rotation,
    Scale,
}

#[derive(Clone, Copy, PartialEq)]
pub enum Interp {
    Linear,
    Step,
}

pub struct Channel {
    pub joint: usize,
    pub path: Path,
    pub interp: Interp,
    pub times: Vec<f32>,
    /// Flattened values: 3 per key for T/S, 4 per key for R (xyzw).
    pub values: Vec<f32>,
}

/// A named animation: a set of per-joint TRS channels.
pub struct Clip {
    pub name: String,
    pub duration: f32,
    pub channels: Vec<Channel>,
}

/// A loaded model: drawable primitives, an optional skeleton, and clips.
pub struct Model {
    pub primitives: Vec<Primitive>,
    pub skeleton: Option<Skeleton>,
    pub clips: Vec<Clip>,
}

impl Model {
    /// Load a model by file extension (`.gltf`/`.glb` or `.obj`).
    pub fn load(path: &str) -> Result<Model, String> {
        let lower = path.to_ascii_lowercase();
        if lower.ends_with(".obj") {
            Self::load_obj(path)
        } else {
            Self::load_gltf(path)
        }
    }

    /// Load a static OBJ mesh (no skeleton/animation).
    pub fn load_obj(path: &str) -> Result<Model, String> {
        let (models, materials) = tobj::load_obj(
            path,
            &tobj::LoadOptions { triangulate: true, single_index: true, ..Default::default() },
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
                let pos = [mesh.positions[i * 3], mesh.positions[i * 3 + 1], mesh.positions[i * 3 + 2]];
                let normal = if has_normals {
                    [mesh.normals[i * 3], mesh.normals[i * 3 + 1], mesh.normals[i * 3 + 2]]
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
        Ok(Model { primitives, skeleton: None, clips: Vec::new() })
    }

    /// Load a glTF/GLB model with materials, skeleton, and animation clips.
    pub fn load_gltf(path: &str) -> Result<Model, String> {
        let (doc, buffers, images) =
            gltf::import(path).map_err(|e| format!("load gltf {path}: {e}"))?;
        let buf = |b: gltf::Buffer| buffers.get(b.index()).map(|d| &d.0[..]);

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

            let mut joints = Vec::with_capacity(joints_nodes.len());
            for (ji, n) in joints_nodes.iter().enumerate() {
                let (t, r, s) = n.transform().decomposed();
                let parent = node_parent.get(&n.index()).and_then(|pi| node_to_joint.get(pi)).copied();
                joints.push(Joint {
                    parent,
                    inverse_bind: ibm.get(ji).copied().unwrap_or(Mat4::IDENTITY),
                    t: Vec3::from(t),
                    r: Quat::from_array(r),
                    s: Vec3::from(s),
                    name: n.name().unwrap_or("").to_string(),
                });
            }
            skeleton = Some(Skeleton { joints });
        }

        // --- node global transforms (for baking static geometry) ---
        let globals = node_global_transforms(&doc);

        // --- primitives ---
        let mut primitives = Vec::new();
        for node in doc.nodes() {
            let Some(mesh) = node.mesh() else { continue };
            let is_skinned = node.skin().is_some();
            let world = globals.get(&node.index()).copied().unwrap_or(Mat4::IDENTITY);
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
                        let n = normal_world.transform_vector3(Vec3::from(normals[i])).normalize_or_zero();
                        (p.into(), n.into())
                    };
                    let mut v = Vertex::new(pos, normal, uvs[i]);
                    if let (Some(j), Some(w)) = (&joints_attr, &weights_attr) {
                        v.joints = [j[i][0] as u32, j[i][1] as u32, j[i][2] as u32, j[i][3] as u32];
                        // Remap glTF skin-local joint indices: read_joints already
                        // indexes into the skin's joint list, which is our order.
                        let ww = w[i];
                        let sum = ww[0] + ww[1] + ww[2] + ww[3];
                        v.weights = if sum > 0.0 { [ww[0] / sum, ww[1] / sum, ww[2] / sum, ww[3] / sum] } else { [1.0, 0.0, 0.0, 0.0] };
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
                let mr_tex = pbr.metallic_roughness_texture().and_then(|i| tex_of(i.texture()));
                let normal_tex = material.normal_texture().and_then(|i| tex_of(i.texture()));
                let emissive_tex = material.emissive_texture().and_then(|i| tex_of(i.texture()));
                primitives.push(Primitive {
                    mesh: data,
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
                let Some(&joint) = node_to_joint.get(&target.node().index()) else { continue };
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
                channels.push(Channel { joint, path, interp, times, values });
            }
            let name = anim.name().unwrap_or("clip").to_string();
            clips.push(Clip { name, duration, channels });
        }

        Ok(Model { primitives, skeleton, clips })
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
        Format::R8G8B8 => px.chunks_exact(3).flat_map(|c| [c[0], c[1], c[2], 255]).collect(),
        Format::R8 => px.iter().flat_map(|&v| [v, v, v, 255]).collect(),
        Format::R8G8 => px.chunks_exact(2).flat_map(|c| [c[0], c[0], c[0], c[1]]).collect(),
        _ => return None,
    };
    Some((rgba, w, h))
}
