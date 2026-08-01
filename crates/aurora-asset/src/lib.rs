//! Aurora's asset layer: authored art, in memory, before it reaches a GPU.
//!
//! [`mesh`] is the vertex format and CPU geometry; [`model`] is a loaded model -
//! drawable primitives with PBR materials, an optional skeleton, and animation
//! clips - together with the importers that produce one from glTF/GLB, OBJ, and
//! FBX.
//!
//! **Place in the graph.** Depends on nothing of Aurora's. `aurora-render3d`
//! sits on top and uploads what lands here; `aurorac` uses it to bake source art
//! offline.
//!
//! **Never.** Never touches wgpu, opens a window, or reads a GPU adapter. That
//! separation is the point: an importer is a pure function from bytes to
//! [`model::Model`], so it is testable on a machine with no graphics device and
//! the offline baker does not drag a renderer in behind it.

pub mod fbx;
pub mod mesh;
pub mod model;

pub use mesh::{MeshData, Vertex};
pub use model::{
    decode_texture, load_texture_file, Channel, Clip, Interp, Joint, Model, Path, Primitive,
    RootMotion, Skeleton, Tex,
};
