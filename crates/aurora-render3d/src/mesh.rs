//! GPU mesh buffers and the wgpu vertex layout.
//!
//! The vertex format and CPU-side geometry live in `aurora-asset`, which has no
//! graphics dependency; this module is the half that needs a device. Both are
//! re-exported here so callers keep naming them through `crate::mesh`.

pub use aurora_asset::mesh::{MeshData, Vertex};

use wgpu::util::DeviceExt;

/// How a [`Vertex`] is fed to the vertex shader.
///
/// This is a free constant rather than an associated one because `Vertex` is
/// defined in `aurora-asset`, which must not depend on wgpu - and an inherent
/// impl may only live in the crate that defines the type.
pub const VERTEX_LAYOUT: wgpu::VertexBufferLayout<'static> = wgpu::VertexBufferLayout {
    array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
    step_mode: wgpu::VertexStepMode::Vertex,
    attributes: &wgpu::vertex_attr_array![
        0 => Float32x3, // pos
        1 => Float32x3, // normal
        2 => Float32x2, // uv
        3 => Uint32x4,  // joints
        4 => Float32x4, // weights
        5 => Float32x4, // tangent
    ],
};

/// GPU-resident mesh buffers.
pub struct GpuMesh {
    pub vbuf: wgpu::Buffer,
    pub ibuf: wgpu::Buffer,
    pub index_count: u32,
    /// Radius of the origin-centred sphere containing this geometry, for the
    /// frustum and shadow-cascade culls. It lives HERE rather than in a
    /// parallel `Vec` beside the mesh store: a parallel array is one more thing
    /// that can fall out of step with the store it indexes, and the store now
    /// reuses freed slots.
    pub radius: f32,
    /// Allocated byte capacity of `vbuf` and `ibuf`. Geometry that is rebuilt
    /// while the game runs (terrain LOD tiles) is rewritten in place whenever
    /// the new data still fits, so a steady-state level-of-detail change costs
    /// two queue writes and no allocation at all.
    vcap: u64,
    icap: u64,
}

impl GpuMesh {
    /// Bytes of GPU buffer this mesh holds: the vertex and index allocations.
    ///
    /// This is the ALLOCATED capacity, not the bytes currently in use, because
    /// that is what the driver is actually holding. A tile rebuilt at a coarser
    /// level of detail keeps its buffers (see [`GpuMesh::write`]), so its cost
    /// stays at the high-water mark until the mesh is freed.
    pub fn bytes(&self) -> u64 {
        self.vcap + self.icap
    }

    pub fn upload(device: &wgpu::Device, mesh: &MeshData) -> GpuMesh {
        let vdata: &[u8] = bytemuck::cast_slice(&mesh.vertices);
        let idata: &[u8] = bytemuck::cast_slice(&mesh.indices);
        let vbuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("mesh-verts"),
            contents: vdata,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });
        let ibuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("mesh-indices"),
            contents: idata,
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
        });
        GpuMesh {
            vbuf,
            ibuf,
            index_count: mesh.indices.len() as u32,
            radius: mesh.bounding_radius(),
            vcap: vdata.len() as u64,
            icap: idata.len() as u64,
        }
    }

    /// Overwrite this mesh's geometry in place, or report `false` when the new
    /// data does not fit the buffers (the caller then re-uploads).
    pub fn write(&mut self, queue: &wgpu::Queue, mesh: &MeshData) -> bool {
        let vdata: &[u8] = bytemuck::cast_slice(&mesh.vertices);
        let idata: &[u8] = bytemuck::cast_slice(&mesh.indices);
        if vdata.len() as u64 > self.vcap || idata.len() as u64 > self.icap {
            return false;
        }
        queue.write_buffer(&self.vbuf, 0, vdata);
        queue.write_buffer(&self.ibuf, 0, idata);
        self.index_count = mesh.indices.len() as u32;
        self.radius = mesh.bounding_radius();
        true
    }
}
