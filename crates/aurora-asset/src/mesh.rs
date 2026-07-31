//! Vertex format and CPU-side mesh geometry.
//!
//! The GPU-resident counterpart (`GpuMesh`) and the wgpu vertex layout live in
//! `aurora-render3d`, which uploads what is built here.

use bytemuck::{Pod, Zeroable};

/// One vertex. Skinning attributes are always present; static meshes set
/// `weights = [1,0,0,0]` and `joints = [0,0,0,0]` and are drawn with skinning
/// disabled (the joint matrices are ignored), so a single pipeline serves both.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct Vertex {
    pub pos: [f32; 3],
    pub normal: [f32; 3],
    pub uv: [f32; 2],
    pub joints: [u32; 4],
    pub weights: [f32; 4],
    /// Tangent xyz + handedness in w (for normal mapping).
    pub tangent: [f32; 4],
}

impl Vertex {
    pub fn new(pos: [f32; 3], normal: [f32; 3], uv: [f32; 2]) -> Vertex {
        Vertex {
            pos,
            normal,
            uv,
            joints: [0; 4],
            weights: [1.0, 0.0, 0.0, 0.0],
            tangent: [1.0, 0.0, 0.0, 1.0],
        }
    }
}

/// CPU-side mesh: vertices + 32-bit indices.
#[derive(Clone, Debug, Default)]
pub struct MeshData {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
}

impl MeshData {
    /// Axis-aligned bounds as `[min_x, min_y, min_z, max_x, max_y, max_z]` in the
    /// mesh's own space.
    ///
    /// This is what lets a caller size a collider to the art rather than to a guess,
    /// so a swapped asset moves its collider with it instead of silently leaving one
    /// behind at the old size. An empty mesh reports all zeroes rather than the
    /// infinities an unseeded fold would produce, so arithmetic on a failed load
    /// stays finite.
    pub fn bounds(&self) -> [f32; 6] {
        if self.vertices.is_empty() {
            return [0.0; 6];
        }
        let mut b = [f32::MAX, f32::MAX, f32::MAX, f32::MIN, f32::MIN, f32::MIN];
        for v in &self.vertices {
            for a in 0..3 {
                b[a] = b[a].min(v.pos[a]);
                b[3 + a] = b[3 + a].max(v.pos[a]);
            }
        }
        b
    }

    /// Radius of the origin-centred sphere containing this geometry, as the
    /// frustum and shadow-cascade culls use it. Never zero, so a single-point
    /// mesh still has a testable bound.
    pub fn bounding_radius(&self) -> f32 {
        self.vertices
            .iter()
            .map(|v| glam::Vec3::from(v.pos).length())
            .fold(0.0f32, f32::max)
            .max(0.001)
    }

    /// Recompute flat per-face normals (used when a source mesh lacks normals).
    pub fn compute_flat_normals(&mut self) {
        for v in &mut self.vertices {
            v.normal = [0.0, 0.0, 0.0];
        }
        for tri in self.indices.chunks_exact(3) {
            let (a, b, c) = (tri[0] as usize, tri[1] as usize, tri[2] as usize);
            let p = |i: usize| glam::Vec3::from(self.vertices[i].pos);
            let n = (p(b) - p(a)).cross(p(c) - p(a)).normalize_or_zero();
            for &i in &[a, b, c] {
                let cur = glam::Vec3::from(self.vertices[i].normal) + n;
                self.vertices[i].normal = cur.into();
            }
        }
        for v in &mut self.vertices {
            v.normal = glam::Vec3::from(v.normal).normalize_or_zero().into();
        }
    }

    /// Compute per-vertex tangents from positions, normals, and UVs (for normal
    /// mapping). Handedness is stored in `tangent.w`.
    pub fn compute_tangents(&mut self) {
        let n = self.vertices.len();
        let mut tan = vec![glam::Vec3::ZERO; n];
        let mut bit = vec![glam::Vec3::ZERO; n];
        for tri in self.indices.chunks_exact(3) {
            let (i0, i1, i2) = (tri[0] as usize, tri[1] as usize, tri[2] as usize);
            let p0 = glam::Vec3::from(self.vertices[i0].pos);
            let p1 = glam::Vec3::from(self.vertices[i1].pos);
            let p2 = glam::Vec3::from(self.vertices[i2].pos);
            let uv0 = glam::Vec2::from(self.vertices[i0].uv);
            let uv1 = glam::Vec2::from(self.vertices[i1].uv);
            let uv2 = glam::Vec2::from(self.vertices[i2].uv);
            let e1 = p1 - p0;
            let e2 = p2 - p0;
            let d1 = uv1 - uv0;
            let d2 = uv2 - uv0;
            let denom = d1.x * d2.y - d2.x * d1.y;
            let r = if denom.abs() > 1e-8 { 1.0 / denom } else { 0.0 };
            let t = (e1 * d2.y - e2 * d1.y) * r;
            let bt = (e2 * d1.x - e1 * d2.x) * r;
            for &i in &[i0, i1, i2] {
                tan[i] += t;
                bit[i] += bt;
            }
        }
        for i in 0..n {
            let nrm = glam::Vec3::from(self.vertices[i].normal);
            let t = tan[i];
            // Gram-Schmidt orthogonalize, fall back to an arbitrary basis.
            let tangent = (t - nrm * nrm.dot(t)).normalize_or_zero();
            let tangent = if tangent.length_squared() < 1e-8 {
                nrm.cross(glam::Vec3::X).normalize_or_zero()
            } else {
                tangent
            };
            let w = if nrm.cross(tangent).dot(bit[i]) < 0.0 {
                -1.0
            } else {
                1.0
            };
            self.vertices[i].tangent = [tangent.x, tangent.y, tangent.z, w];
        }
    }

    /// An axis-aligned unit cube centered at the origin (side length 2), with
    /// per-face normals and UVs. Handy as a primitive and for tests.
    pub fn cube() -> MeshData {
        MeshData::box_dims(1.0, 1.0, 1.0)
    }

    /// An axis-aligned box with the given half-extents (full size `2*hx` by
    /// `2*hy` by `2*hz`, matching a physics box collider's extents).
    pub fn box_dims(hx: f32, hy: f32, hz: f32) -> MeshData {
        let mut m = MeshData::default();
        // (normal, four corners ccw) per face, on the unit cube.
        let faces: [([f32; 3], [[f32; 3]; 4]); 6] = [
            (
                [0.0, 0.0, 1.0],
                [
                    [-1.0, -1.0, 1.0],
                    [1.0, -1.0, 1.0],
                    [1.0, 1.0, 1.0],
                    [-1.0, 1.0, 1.0],
                ],
            ),
            (
                [0.0, 0.0, -1.0],
                [
                    [1.0, -1.0, -1.0],
                    [-1.0, -1.0, -1.0],
                    [-1.0, 1.0, -1.0],
                    [1.0, 1.0, -1.0],
                ],
            ),
            (
                [1.0, 0.0, 0.0],
                [
                    [1.0, -1.0, 1.0],
                    [1.0, -1.0, -1.0],
                    [1.0, 1.0, -1.0],
                    [1.0, 1.0, 1.0],
                ],
            ),
            (
                [-1.0, 0.0, 0.0],
                [
                    [-1.0, -1.0, -1.0],
                    [-1.0, -1.0, 1.0],
                    [-1.0, 1.0, 1.0],
                    [-1.0, 1.0, -1.0],
                ],
            ),
            (
                [0.0, 1.0, 0.0],
                [
                    [-1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0],
                    [1.0, 1.0, -1.0],
                    [-1.0, 1.0, -1.0],
                ],
            ),
            (
                [0.0, -1.0, 0.0],
                [
                    [-1.0, -1.0, -1.0],
                    [1.0, -1.0, -1.0],
                    [1.0, -1.0, 1.0],
                    [-1.0, -1.0, 1.0],
                ],
            ),
        ];
        for (normal, corners) in faces {
            let base = m.vertices.len() as u32;
            let uvs = [[0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]];
            for (corner, uv) in corners.iter().zip(uvs) {
                let p = [corner[0] * hx, corner[1] * hy, corner[2] * hz];
                m.vertices.push(Vertex::new(p, normal, uv));
            }
            m.indices
                .extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
        }
        m.compute_tangents();
        m
    }

    /// A UV sphere of `radius` centered at the origin, `segments` around and
    /// `segments/2` rings.
    pub fn sphere(radius: f32, segments: u32) -> MeshData {
        let mut m = MeshData::default();
        let seg = segments.max(3);
        let rings = (seg / 2).max(2);
        for ring in 0..=rings {
            let v = ring as f32 / rings as f32;
            let phi = v * std::f32::consts::PI;
            for s in 0..=seg {
                let u = s as f32 / seg as f32;
                let theta = u * std::f32::consts::TAU;
                let n = [phi.sin() * theta.cos(), phi.cos(), phi.sin() * theta.sin()];
                m.vertices.push(Vertex::new(
                    [n[0] * radius, n[1] * radius, n[2] * radius],
                    n,
                    [u, v],
                ));
            }
        }
        let stride = seg + 1;
        for ring in 0..rings {
            for s in 0..seg {
                let a = ring * stride + s;
                let b = a + stride;
                // Wound so the OUTWARD faces are CCW (front-facing) to match the
                // engine's back-face culling, otherwise the outside is culled and
                // you see straight through to the inner surface.
                m.indices.extend_from_slice(&[a, a + 1, b, a + 1, b + 1, b]);
            }
        }
        m.compute_tangents();
        m
    }

    /// A flat ground plane of side `size` centered at the origin in the XZ plane,
    /// normal pointing +Y. `tiles` controls UV tiling.
    pub fn plane(size: f32, tiles: f32) -> MeshData {
        let h = size * 0.5;
        let n = [0.0, 1.0, 0.0];
        let mut m = MeshData::default();
        m.vertices.push(Vertex::new([-h, 0.0, -h], n, [0.0, 0.0]));
        m.vertices.push(Vertex::new([-h, 0.0, h], n, [0.0, tiles]));
        m.vertices.push(Vertex::new([h, 0.0, h], n, [tiles, tiles]));
        m.vertices.push(Vertex::new([h, 0.0, -h], n, [tiles, 0.0]));
        m.indices.extend_from_slice(&[0, 1, 2, 0, 2, 3]);
        m.compute_tangents();
        m
    }

    /// A unit quad in the XY plane (corners -0.5..0.5), normal +Z. Used for
    /// camera-facing billboards (the orientation comes from the model matrix).
    pub fn quad() -> MeshData {
        let n = [0.0, 0.0, 1.0];
        let mut m = MeshData::default();
        m.vertices
            .push(Vertex::new([-0.5, -0.5, 0.0], n, [0.0, 1.0]));
        m.vertices
            .push(Vertex::new([0.5, -0.5, 0.0], n, [1.0, 1.0]));
        m.vertices.push(Vertex::new([0.5, 0.5, 0.0], n, [1.0, 0.0]));
        m.vertices
            .push(Vertex::new([-0.5, 0.5, 0.0], n, [0.0, 0.0]));
        m.indices.extend_from_slice(&[0, 1, 2, 0, 2, 3]);
        m.compute_tangents();
        m
    }
}
