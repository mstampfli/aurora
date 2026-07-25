//! Heightmap terrain: the heightfield, and the level-of-detail mesher that turns
//! it into crack-free GPU tiles.
//!
//! # One surface, three consumers
//!
//! A terrain is wrong in a way that is miserable to debug when the surface you
//! SEE, the surface you WALK on, and the surface a height query reports are
//! three slightly different things. So there is exactly one definition here:
//!
//! > cell `(i, j)` (row `i` along +Z, column `j` along +X) is two triangles,
//! > `(p00, p10, p01)` and `(p10, p11, p01)`, where `pRC` is the sample at row
//! > `i + R`, column `j + C`. The shared diagonal runs from `p10` to `p01`.
//!
//! That is exactly [`parry3d`'s heightfield triangulation], so the Rapier
//! collider built from a [`Heightfield`] is the same surface, and
//! [`Heightfield::height_at`] evaluates that surface analytically (planar
//! interpolation inside the triangle the query lands in, NOT bilinear over the
//! quad - bilinear disagrees with either diagonal by up to a quarter of the
//! cell's height range at the cell centre). The full-resolution render mesh uses
//! the same split. Coarser LOD tiles are, by definition, a simplification of it.
//!
//! [`parry3d`'s heightfield triangulation]: parry3d's `HeightField::triangles_vids_at`
//!
//! # Level of detail
//!
//! The field is cut into square tiles of [`TILE_CELLS`] cells. Each tile picks a
//! sample `step` (a power of two) from its distance to the camera, so a distant
//! tile spends a quarter of the triangles per LOD level.
//!
//! Seams are handled by **edge stitching**, not skirts: a tile whose neighbour
//! is coarser builds that edge at the NEIGHBOUR's step, so both tiles emit the
//! exact same vertex positions along their shared boundary. There is no
//! T-junction to crack open and no skirt wall to show at grazing angles. The
//! interior of such a tile is joined to its coarsened border by a ring of
//! transition triangles (see [`Heightfield::tile_mesh`]).

use std::sync::Arc;

use glam::Vec3;

use crate::mesh::{MeshData, Vertex};

/// Magic bytes at the start of an `.aterr` heightfield file.
pub const MAGIC: &[u8; 8] = b"AURTERR1";

/// Bytes of `.aterr` header before the first height sample.
pub const HEADER_BYTES: usize = 24;

/// Largest accepted heightfield side, in samples (4096 cells).
pub const MAX_DIM: u32 = 4097;

/// Cells along one side of a full-resolution terrain tile. A power of two, so
/// every LOD step divides it evenly.
pub const TILE_CELLS: u32 = 32;

/// Edge slots of a tile, in the order [`Heightfield::tile_mesh`] expects.
pub const EDGE_NEG_X: usize = 0;
pub const EDGE_POS_X: usize = 1;
pub const EDGE_NEG_Z: usize = 2;
pub const EDGE_POS_Z: usize = 3;

/// A regular grid of terrain heights.
///
/// Immutable once built, so the renderer, the physics collider, and the height
/// query can share one instance behind an [`Arc`] instead of holding copies that
/// can drift apart.
///
/// Sample `(row, col)` sits at world `(origin_x + col * spacing, h, origin_z +
/// row * spacing)`: `col` runs along +X and `row` along +Z.
#[derive(Debug)]
pub struct Heightfield {
    dim: u32,
    spacing: f32,
    origin_x: f32,
    origin_z: f32,
    /// `dim * dim` samples, row-major: `heights[row * dim + col]`.
    heights: Vec<f32>,
}

impl Heightfield {
    /// Build a heightfield from raw samples.
    ///
    /// `dim` must be `2^k + 1` (5, 9, 17, ... 4097) so the tile grid and every
    /// LOD step divide it exactly - the same constraint mainstream terrain
    /// engines put on a heightmap. Everything is validated, because a silently
    /// mis-shaped terrain is a bug that only shows up as players falling
    /// through the world.
    pub fn new(
        dim: u32,
        spacing: f32,
        origin_x: f32,
        origin_z: f32,
        heights: Vec<f32>,
    ) -> Result<Heightfield, String> {
        if !(3..=MAX_DIM).contains(&dim) {
            return Err(format!("terrain: dim {dim} out of range (3..={MAX_DIM})"));
        }
        if !(dim - 1).is_power_of_two() {
            return Err(format!(
                "terrain: dim {dim} is not 2^k + 1 (use 5, 9, 17, 33, 65, 129, 257, 513, 1025, 2049 or 4097)"
            ));
        }
        if !spacing.is_finite() || spacing <= 0.0 {
            return Err(format!("terrain: spacing {spacing} must be finite and > 0"));
        }
        if !origin_x.is_finite() || !origin_z.is_finite() {
            return Err("terrain: origin must be finite".to_string());
        }
        let want = dim as usize * dim as usize;
        if heights.len() != want {
            return Err(format!(
                "terrain: expected {want} samples for dim {dim}, got {}",
                heights.len()
            ));
        }
        if let Some(bad) = heights.iter().position(|h| !h.is_finite()) {
            return Err(format!("terrain: sample {bad} is not finite"));
        }
        Ok(Heightfield {
            dim,
            spacing,
            origin_x,
            origin_z,
            heights,
        })
    }

    /// Samples along one side.
    pub fn dim(&self) -> u32 {
        self.dim
    }
    /// World distance between adjacent samples.
    pub fn spacing(&self) -> f32 {
        self.spacing
    }
    /// World X of sample column 0.
    pub fn origin_x(&self) -> f32 {
        self.origin_x
    }
    /// World Z of sample row 0.
    pub fn origin_z(&self) -> f32 {
        self.origin_z
    }
    /// World side length of the terrain footprint: `(dim - 1) * spacing`.
    pub fn extent(&self) -> f32 {
        (self.dim - 1) as f32 * self.spacing
    }
    /// All samples, row-major.
    pub fn heights(&self) -> &[f32] {
        &self.heights
    }

    /// The sample at `(row, col)`, clamped to the footprint.
    pub fn sample(&self, row: i64, col: i64) -> f32 {
        let last = self.dim as i64 - 1;
        let r = row.clamp(0, last) as usize;
        let c = col.clamp(0, last) as usize;
        self.heights[r * self.dim as usize + c]
    }

    /// Surface height at world `(x, z)`.
    ///
    /// Evaluates the same triangulated surface the collider uses, so a downward
    /// raycast onto the terrain and this query agree to float precision.
    ///
    /// Outside the footprint the query CLAMPS to the nearest edge sample, so it
    /// is always defined (the surface reads as if the border extended outward
    /// forever). A non-finite coordinate clamps to the terrain's origin corner.
    /// Note that the physics collider stops at the footprint, so beyond the
    /// border this height has no collider behind it.
    pub fn height_at(&self, x: f64, z: f64) -> f64 {
        let last = (self.dim - 1) as f64;
        let sp = self.spacing as f64;
        let clamp01 = |v: f64| {
            if v.is_nan() {
                0.0
            } else {
                v.clamp(0.0, last)
            }
        };
        let fx = clamp01((x - self.origin_x as f64) / sp);
        let fz = clamp01((z - self.origin_z as f64) / sp);
        // The last row/column of samples has no cell of its own, so a query
        // exactly on the far border evaluates the last cell at u/v = 1.
        let j = (fx.floor() as i64).clamp(0, self.dim as i64 - 2);
        let i = (fz.floor() as i64).clamp(0, self.dim as i64 - 2);
        let u = fx - j as f64;
        let v = fz - i as f64;
        let h00 = self.sample(i, j) as f64;
        let h01 = self.sample(i, j + 1) as f64;
        let h10 = self.sample(i + 1, j) as f64;
        let h11 = self.sample(i + 1, j + 1) as f64;
        if u + v <= 1.0 {
            // Triangle (p00, p10, p01): the half of the cell with u + v <= 1.
            h00 + u * (h01 - h00) + v * (h10 - h00)
        } else {
            // Triangle (p10, p11, p01).
            h11 + (1.0 - u) * (h10 - h11) + (1.0 - v) * (h01 - h11)
        }
    }

    /// Unit surface normal at sample `(row, col)`, from the height gradient.
    ///
    /// Central differences in the interior and one-sided at the border (the
    /// actual sample span is divided out either way), so lighting is smooth and
    /// does not change with the LOD a vertex happens to be drawn at.
    pub fn normal_at_sample(&self, row: i64, col: i64) -> Vec3 {
        let last = self.dim as i64 - 1;
        let (c0, c1) = ((col - 1).max(0), (col + 1).min(last));
        let (r0, r1) = ((row - 1).max(0), (row + 1).min(last));
        let dx = (self.sample(row, c1) - self.sample(row, c0)) / ((c1 - c0) as f32 * self.spacing);
        let dz = (self.sample(r1, col) - self.sample(r0, col)) / ((r1 - r0) as f32 * self.spacing);
        Vec3::new(-dx, 1.0, -dz).normalize()
    }

    /// A procedurally generated heightfield: seeded value-noise fBm centred on
    /// the world origin, with heights spanning `[0, amplitude]`.
    ///
    /// Deterministic for a given `(seed, dim, spacing, amplitude)` on every
    /// platform: the hash is integer-only and the octave sum is normalized, so a
    /// test can build a terrain without shipping an asset.
    pub fn generate(
        seed: i64,
        dim: u32,
        spacing: f32,
        amplitude: f32,
    ) -> Result<Heightfield, String> {
        if !amplitude.is_finite() || amplitude < 0.0 {
            return Err(format!(
                "terrain: amplitude {amplitude} must be finite and >= 0"
            ));
        }
        if !(3..=MAX_DIM).contains(&dim) || !(dim - 1).is_power_of_two() {
            // Let `new` produce the one authoritative message.
            return Heightfield::new(dim, spacing, 0.0, 0.0, Vec::new());
        }
        let seed = seed as u64;
        let n = dim as usize;
        let mut heights = vec![0.0f32; n * n];
        // Four base features across the terrain, five octaves.
        let base = 4.0 / (dim - 1) as f32;
        for row in 0..n {
            for col in 0..n {
                let (mut f, mut a, mut sum, mut norm) = (base, 1.0f32, 0.0f32, 0.0f32);
                for oct in 0..5u64 {
                    sum += a * value_noise(seed ^ (oct << 32), col as f32 * f, row as f32 * f);
                    norm += a;
                    f *= 2.0;
                    a *= 0.5;
                }
                heights[row * n + col] = (sum / norm) * amplitude;
            }
        }
        let half = (dim - 1) as f32 * spacing * 0.5;
        Heightfield::new(dim, spacing, -half, -half, heights)
    }

    /// Decode the `.aterr` byte layout documented in
    /// `docs/04-stdlib-and-builtins.md`.
    pub fn decode(bytes: &[u8]) -> Result<Heightfield, String> {
        if bytes.len() < HEADER_BYTES {
            return Err(format!(
                "terrain: file is {} bytes, shorter than the {HEADER_BYTES}-byte header",
                bytes.len()
            ));
        }
        if &bytes[0..8] != MAGIC {
            return Err("terrain: bad magic (expected AURTERR1)".to_string());
        }
        let u32_at =
            |o: usize| u32::from_le_bytes([bytes[o], bytes[o + 1], bytes[o + 2], bytes[o + 3]]);
        let f32_at =
            |o: usize| f32::from_le_bytes([bytes[o], bytes[o + 1], bytes[o + 2], bytes[o + 3]]);
        let dim = u32_at(8);
        let spacing = f32_at(12);
        let origin_x = f32_at(16);
        let origin_z = f32_at(20);
        // Check the size BEFORE allocating: `dim` comes from the file.
        if !(3..=MAX_DIM).contains(&dim) {
            return Err(format!("terrain: dim {dim} out of range (3..={MAX_DIM})"));
        }
        let want = HEADER_BYTES + dim as usize * dim as usize * 4;
        if bytes.len() != want {
            return Err(format!(
                "terrain: dim {dim} needs a {want}-byte file, got {}",
                bytes.len()
            ));
        }
        let heights = bytes[HEADER_BYTES..]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        Heightfield::new(dim, spacing, origin_x, origin_z, heights)
    }

    /// Encode to the `.aterr` byte layout.
    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(HEADER_BYTES + self.heights.len() * 4);
        out.extend_from_slice(MAGIC);
        out.extend_from_slice(&self.dim.to_le_bytes());
        out.extend_from_slice(&self.spacing.to_le_bytes());
        out.extend_from_slice(&self.origin_x.to_le_bytes());
        out.extend_from_slice(&self.origin_z.to_le_bytes());
        for h in &self.heights {
            out.extend_from_slice(&h.to_le_bytes());
        }
        out
    }

    /// Read an `.aterr` file.
    pub fn load(path: &str) -> Result<Heightfield, String> {
        let bytes = std::fs::read(path).map_err(|e| format!("terrain: cannot read {path}: {e}"))?;
        Heightfield::decode(&bytes).map_err(|e| format!("{e} (in {path})"))
    }

    /// Write an `.aterr` file.
    pub fn save(&self, path: &str) -> Result<(), String> {
        std::fs::write(path, self.encode())
            .map_err(|e| format!("terrain: cannot write {path}: {e}"))
    }

    // --- tiling -----------------------------------------------------------

    /// Cells along one side of a tile: [`TILE_CELLS`], or the whole terrain when
    /// it is smaller than that.
    pub fn tile_cells(&self) -> u32 {
        TILE_CELLS.min(self.dim - 1)
    }

    /// Tiles along one side.
    pub fn tiles_per_side(&self) -> u32 {
        (self.dim - 1) / self.tile_cells()
    }

    /// Coarsest sample step a tile may use: half the tile, so a tile always has
    /// at least 2x2 cells and the transition ring is never degenerate.
    pub fn max_step(&self) -> u32 {
        (self.tile_cells() / 2).max(1)
    }

    /// World position of a tile's centre, at the midpoint of its height range,
    /// and the radius of a sphere around it that contains the whole tile.
    ///
    /// Tile meshes are built in coordinates relative to this centre and drawn
    /// with a translation, so the renderer's bounding sphere (built from the
    /// mesh's own extent) is tight and per-tile frustum and shadow-cascade
    /// culling actually work.
    pub fn tile_bounds(&self, tx: u32, tz: u32) -> (Vec3, f32) {
        let t = self.tile_cells() as i64;
        let (c0, r0) = (tx as i64 * t, tz as i64 * t);
        let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
        for r in r0..=r0 + t {
            for c in c0..=c0 + t {
                let h = self.sample(r, c);
                lo = lo.min(h);
                hi = hi.max(h);
            }
        }
        let half = t as f32 * self.spacing * 0.5;
        let centre = Vec3::new(
            self.origin_x + c0 as f32 * self.spacing + half,
            (lo + hi) * 0.5,
            self.origin_z + r0 as f32 * self.spacing + half,
        );
        let radius = (half * half * 2.0 + ((hi - lo) * 0.5).powi(2)).sqrt();
        (centre, radius.max(1e-3))
    }

    /// Build the mesh for tile `(tx, tz)` at the given level of detail, in
    /// coordinates relative to the tile centre reported by [`Self::tile_bounds`].
    ///
    /// `lod.step` is the interior sample step; `lod.edge_step[e]` is the step of
    /// edge `e` (`EDGE_NEG_X` .. `EDGE_POS_Z`), which is `step` for an edge whose
    /// neighbour is equally fine or finer, and the NEIGHBOUR's coarser step
    /// otherwise. Every edge step is a multiple of `step` and divides the tile,
    /// so both sides of a seam land on exactly the same sample positions.
    ///
    /// When no edge is coarsened this is a plain grid with the collider's own
    /// diagonal. Otherwise the interior `[1, n-1]` sub-grid is meshed as a grid
    /// and the surrounding ring is stitched: each outer segment fans to the
    /// inner-ring vertices it spans, which covers the ring exactly, introduces
    /// no T-junction, and works for any power-of-two step ratio.
    pub fn tile_mesh(&self, tx: u32, tz: u32, lod: TileLod) -> MeshData {
        let t = self.tile_cells();
        let step = lod
            .step
            .clamp(1, t)
            .next_power_of_two()
            .min(self.max_step());
        let n = t / step;
        let (c0, r0) = (tx as i64 * t as i64, tz as i64 * t as i64);
        let (centre, _) = self.tile_bounds(tx, tz);

        // Vertices: the full (n+1)^2 grid. A coarsened edge simply does not
        // index its odd vertices; leaving them in the buffer keeps the index
        // arithmetic a single multiply-add and wastes under 2% of the data.
        let mut m = MeshData {
            vertices: Vec::with_capacity(((n + 1) * (n + 1)) as usize),
            indices: Vec::new(),
        };
        for r in 0..=n {
            for c in 0..=n {
                let (row, col) = (r0 + (r * step) as i64, c0 + (c * step) as i64);
                let h = self.sample(row, col);
                let pos = Vec3::new(
                    self.origin_x + col as f32 * self.spacing,
                    h,
                    self.origin_z + row as f32 * self.spacing,
                ) - centre;
                let nrm = self.normal_at_sample(row, col);
                m.vertices.push(Vertex::new(
                    pos.into(),
                    nrm.into(),
                    // One UV unit per sample cell, continuous across tiles.
                    [col as f32, row as f32],
                ));
            }
        }
        let vid = |r: u32, c: u32| r * (n + 1) + c;

        if lod.edge_step.iter().all(|&e| e == step) {
            // No coarser neighbour: the collider's own triangulation, exactly.
            for r in 0..n {
                for c in 0..n {
                    push_cell(
                        &mut m,
                        vid(r, c),
                        vid(r + 1, c),
                        vid(r, c + 1),
                        vid(r + 1, c + 1),
                    );
                }
            }
        } else {
            // Interior grid over the cells whose four corners are all inside the
            // ring. Empty for n == 2, where the ring covers the whole tile.
            for r in 1..n.saturating_sub(1) {
                for c in 1..n.saturating_sub(1) {
                    push_cell(
                        &mut m,
                        vid(r, c),
                        vid(r + 1, c),
                        vid(r, c + 1),
                        vid(r + 1, c + 1),
                    );
                }
            }
            stitch_ring(&mut m, n, step, &lod.edge_step, &vid);
        }
        m.compute_tangents();
        m
    }
}

/// The two triangles of one full cell, with the collider's diagonal
/// (`p10`-`p01`). `p00` is `(row, col)`, `p10` is `(row+1, col)`, `p01` is
/// `(row, col+1)`, `p11` is `(row+1, col+1)`.
fn push_cell(m: &mut MeshData, p00: u32, p10: u32, p01: u32, p11: u32) {
    push_tri(m, p00, p10, p01);
    push_tri(m, p10, p11, p01);
}

/// Append a triangle wound so its normal points UP (+Y), swapping the last two
/// indices when the caller handed them the other way round.
///
/// Terrain triangles always have a non-zero footprint in XZ, so "which way is
/// up" is decidable per triangle. Making it decidable HERE means no ring or fan
/// in this file can be wound backwards and get silently back-face culled: the
/// generator cannot express the illegal state.
fn push_tri(m: &mut MeshData, a: u32, b: u32, c: u32) {
    let p = |i: u32| m.vertices[i as usize].pos;
    let (pa, pb, pc) = (p(a), p(b), p(c));
    // The +Y component of (pb - pa) x (pc - pa).
    let ny = (pb[2] - pa[2]) * (pc[0] - pa[0]) - (pb[0] - pa[0]) * (pc[2] - pa[2]);
    if ny > 0.0 {
        m.indices.extend_from_slice(&[a, b, c]);
    } else if ny < 0.0 {
        m.indices.extend_from_slice(&[a, c, b]);
    }
    // ny == 0 is a triangle with no footprint: it cannot tile any area, so
    // dropping it removes a degenerate rather than opening a hole.
}

/// Mesh the ring between a tile's outer boundary (at each edge's own step) and
/// the inner `[1, n-1]` sub-grid boundary (at the interior step).
///
/// Both boundaries are walked as cycles in the same rotational direction. Each
/// outer vertex projects onto the inner cycle by clamping its grid coordinates
/// into `[1, n-1]`, and that projection advances monotonically around the cycle,
/// so an outer segment maps to a contiguous run of inner vertices and fans onto
/// it. Every inner vertex and every outer vertex is used exactly once, which is
/// what makes the ring watertight for any edge-step ratio, including the
/// degenerate `n == 2` case where the whole inner cycle is a single vertex.
fn stitch_ring(
    m: &mut MeshData,
    n: u32,
    step: u32,
    edge_step: &[u32; 4],
    vid: &impl Fn(u32, u32) -> u32,
) {
    let seg = |e: usize| (edge_step[e] / step).clamp(1, n);
    let (m_nx, m_px, m_nz, m_pz) = (
        seg(EDGE_NEG_X),
        seg(EDGE_POS_X),
        seg(EDGE_NEG_Z),
        seg(EDGE_POS_Z),
    );
    // Outer cycle, counter-clockwise in grid space: -Z edge, +X edge, +Z edge,
    // -X edge, each contributing its own vertices and sharing the corners.
    let mut outer: Vec<(u32, u32)> = Vec::new();
    let mut c = 0;
    while c <= n {
        outer.push((0, c));
        c += m_nz;
    }
    let mut r = m_px;
    while r <= n {
        outer.push((r, n));
        r += m_px;
    }
    let mut c = n as i64 - m_pz as i64;
    while c >= 0 {
        outer.push((n, c as u32));
        c -= m_pz as i64;
    }
    let mut r = n as i64 - m_nx as i64;
    while r >= m_nx as i64 {
        outer.push((r as u32, 0));
        r -= m_nx as i64;
    }

    // Inner cycle: the boundary of the [1, n-1] square, same direction. For
    // n == 2 that square is the single vertex (1, 1).
    let k = n.saturating_sub(2);
    let inner_len = if k == 0 { 1 } else { 4 * k };
    let inner = |t: u32| -> (u32, u32) {
        if k == 0 {
            return (1, 1);
        }
        match t / k {
            0 => (1, 1 + t),
            1 => (1 + (t - k), n - 1),
            2 => (n - 1, n - 1 - (t - 2 * k)),
            _ => (n - 1 - (t - 3 * k), 1),
        }
    };
    // Position of an outer vertex's projection on the inner cycle.
    let pos = |(r, c): (u32, u32)| -> u32 {
        if k == 0 {
            return 0;
        }
        let (r, c) = (r.clamp(1, n - 1), c.clamp(1, n - 1));
        if r == 1 && c < n - 1 {
            c - 1
        } else if c == n - 1 && r < n - 1 {
            k + (r - 1)
        } else if r == n - 1 && c > 1 {
            2 * k + (n - 1 - c)
        } else {
            3 * k + (n - 1 - r)
        }
    };

    for i in 0..outer.len() {
        let a = outer[i];
        let b = outer[(i + 1) % outer.len()];
        let (ia, ib) = (pos(a), pos(b));
        let (va, vb) = (vid(a.0, a.1), vid(b.0, b.1));
        push_tri(m, va, vb, vid(inner(ib).0, inner(ib).1));
        // Fan back across every inner vertex the segment spans.
        let mut t = ib;
        while t != ia {
            let prev = (t + inner_len - 1) % inner_len;
            let (cr, cc) = inner(t);
            let (pr, pc) = inner(prev);
            push_tri(m, va, vid(cr, cc), vid(pr, pc));
            t = prev;
        }
    }
}

/// The level of detail one tile is meshed at.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct TileLod {
    /// Sample step of the tile interior: a power of two, 1 = full resolution.
    pub step: u32,
    /// Sample step of each edge, indexed by `EDGE_*`. Always a multiple of
    /// `step`: an edge is coarsened to match a coarser neighbour, never
    /// refined (the coarser side cannot follow a finer edge).
    pub edge_step: [u32; 4],
}

impl Default for TileLod {
    fn default() -> TileLod {
        TileLod {
            step: 1,
            edge_step: [1; 4],
        }
    }
}

/// GPU bytes of a worst-case (full-detail) tile mesh: the whole `(n+1)^2`
/// vertex grid plus two triangles per cell. Derived from the vertex layout
/// rather than written down, so it cannot drift when a vertex attribute is
/// added.
pub const MAX_TILE_BYTES: u64 = {
    let vertices = ((TILE_CELLS + 1) * (TILE_CELLS + 1)) as u64;
    let indices = (TILE_CELLS * TILE_CELLS * 6) as u64;
    vertices * std::mem::size_of::<Vertex>() as u64 + indices * 4
};

/// Default GPU budget for resident terrain tile geometry, in bytes.
///
/// # Where the number comes from
///
/// A full-detail tile is `(TILE_CELLS + 1)^2 = 1089` vertices of 80 bytes plus
/// `TILE_CELLS^2 * 2 * 3 = 6144` 32-bit indices: about 109 KiB. A tile that has
/// ever been full detail keeps that allocation while it is resident, because
/// [`crate::GpuMesh::write`] deliberately reuses the buffers rather than
/// reallocating on every level-of-detail change. So 32 MiB is roughly 300
/// worst-case tiles, and far more in practice because the tiles behind the
/// first LOD threshold are a quarter the size per step.
///
/// For scale: the 4097-sample cap is 128x128 = 16384 tiles, which at full
/// detail would be about 1.7 GiB. The budget is what stands between a player
/// crossing that map and every tile they have ever seen staying resident.
pub const TILE_CACHE_BUDGET: u64 = 32 << 20;

/// GPU-side terrain: one mesh per tile, rebuilt in place when that tile's level
/// of detail changes, and evicted when the cache is over budget.
///
/// Memory is O(tiles), not O(tiles * LOD levels * edge configurations): a tile
/// keeps ONE mesh whose contents are rewritten, so a camera move costs a queue
/// write for the handful of tiles that crossed a distance threshold and nothing
/// at all for the rest.
///
/// # Eviction
///
/// Tile meshes are built lazily, the first time a tile is actually drawn, so
/// without eviction the resident set is O(tiles ever visited) and a player
/// crossing a large map accumulates the whole terrain. The cache is bounded by
/// [`TILE_CACHE_BUDGET`] bytes of GPU mesh and evicts least-recently-drawn
/// first (which, since a tile is drawn exactly when it is in the frustum, is
/// the tile the camera turned away from longest ago).
///
/// The set drawn in the CURRENT frame is never evicted: its meshes are already
/// queued, and dropping one would punch a hole in the frame being rendered. If
/// one frame's visible tiles alone exceed the budget, they all stay and the
/// cache is over budget for that frame - a hole in the world is a worse
/// outcome than a temporary overshoot. [`Self::over_budget_frames`] counts
/// those frames so it is measurable rather than silent.
pub struct TerrainRender {
    field: Arc<Heightfield>,
    material: crate::render::MaterialId,
    color: [f32; 3],
    tiles_per_side: u32,
    tiles: Vec<TerrainTile>,
    /// Distance at which the first LOD step happens, in world units.
    lod_range: f32,
    /// `(tiles queued, finest step, coarsest step)` of the last [`Self::draw`].
    /// A visual seam check needs this: "no crack was rendered" proves nothing
    /// unless a level-of-detail boundary was actually on screen.
    last_draw: (usize, u32, u32),
    /// Monotone draw counter, the "time" the LRU is ordered by.
    frame: u64,
    /// GPU bytes currently held by tile meshes, maintained incrementally so the
    /// eviction check is O(1) per frame rather than a scan.
    resident: u64,
    budget: u64,
    evicted: u64,
    over_budget_frames: u64,
    /// Reused scratch for the eviction pass, so a frame that evicts does not
    /// also allocate.
    evict_scratch: Vec<(u64, usize)>,
}

struct TerrainTile {
    /// Renderer mesh id, allocated the first time the tile is drawn and
    /// released again when the cache evicts it.
    mesh: Option<crate::render::MeshId>,
    /// The level of detail currently IN that mesh.
    built: TileLod,
    centre: Vec3,
    radius: f32,
    /// [`TerrainRender::frame`] at this tile's last draw; 0 when not resident.
    last_used: u64,
    /// GPU bytes of this tile's mesh, mirrored here so eviction can subtract
    /// without a renderer lookup per candidate.
    bytes: u64,
}

impl TerrainRender {
    /// Prepare a terrain for drawing. The material is created once; only the
    /// per-tile meshes change from frame to frame.
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        renderer: &mut crate::render::Renderer3D,
        field: Arc<Heightfield>,
        color: [f32; 3],
    ) -> TerrainRender {
        let per_side = field.tiles_per_side();
        let mut tiles = Vec::with_capacity((per_side * per_side) as usize);
        for tz in 0..per_side {
            for tx in 0..per_side {
                let (centre, radius) = field.tile_bounds(tx, tz);
                tiles.push(TerrainTile {
                    mesh: None,
                    built: TileLod::default(),
                    centre,
                    radius,
                    last_used: 0,
                    bytes: 0,
                });
            }
        }
        let material = renderer.add_material(
            device,
            queue,
            &crate::render::MaterialDesc::flat([color[0], color[1], color[2], 1.0]),
        );
        // Full detail out to two tile widths, then a step per doubling.
        let lod_range = field.tile_cells() as f32 * field.spacing() * 2.0;
        TerrainRender {
            field,
            material,
            color,
            tiles_per_side: per_side,
            tiles,
            lod_range,
            last_draw: (0, 1, 1),
            frame: 0,
            resident: 0,
            budget: TILE_CACHE_BUDGET,
            evicted: 0,
            over_budget_frames: 0,
            evict_scratch: Vec::new(),
        }
    }

    /// Release every GPU resource this terrain owns: each resident tile mesh and
    /// the terrain material.
    ///
    /// A terrain is REPLACED, not mutated, when a game reloads or regenerates
    /// one, and the renderer's stores outlive any single terrain - so the new
    /// terrain installing itself is not enough, the old one has to hand its
    /// buffers back. Idempotent: a second call frees nothing, because the
    /// generation-tagged ids no longer resolve.
    pub fn release(&mut self, renderer: &mut crate::render::Renderer3D) {
        for tile in &mut self.tiles {
            if let Some(id) = tile.mesh.take() {
                renderer.free_mesh(id);
            }
            tile.bytes = 0;
            tile.last_used = 0;
        }
        renderer.free_material(self.material);
        self.resident = 0;
    }

    /// The heightfield this was built from.
    pub fn field(&self) -> &Arc<Heightfield> {
        &self.field
    }

    /// GPU bytes currently held by this terrain's tile meshes.
    pub fn resident_bytes(&self) -> u64 {
        self.resident
    }

    /// The tile cache budget in bytes ([`TILE_CACHE_BUDGET`] unless overridden).
    pub fn budget_bytes(&self) -> u64 {
        self.budget
    }

    /// Override the tile cache budget. Clamped up to one worst-case full-detail
    /// tile, because a budget that cannot hold a single tile would evict on
    /// every frame and thrash instead of caching.
    pub fn set_budget_bytes(&mut self, bytes: u64) {
        self.budget = bytes.max(MAX_TILE_BYTES);
    }

    /// Tile meshes evicted since this terrain was built. Zero across a traverse
    /// means the cache never had to work, so a budget test that expects
    /// eviction should assert this is non-zero before trusting its result.
    pub fn evictions(&self) -> u64 {
        self.evicted
    }

    /// Frames whose own visible tile set exceeded the budget, so nothing could
    /// be evicted. Non-zero means the budget is too small for the view
    /// distance in use, not that the cache failed.
    pub fn over_budget_frames(&self) -> u64 {
        self.over_budget_frames
    }

    /// `(tiles queued, finest sample step, coarsest sample step)` of the last
    /// [`Self::draw`]. `finest != coarsest` means a level-of-detail seam was on
    /// screen, which is what makes a rendered crack check meaningful.
    pub fn last_draw(&self) -> (usize, u32, u32) {
        self.last_draw
    }

    /// Replace the albedo. A new material is only created when the color
    /// actually changes, so calling this every frame does not leak materials -
    /// and the material it replaces is released, so calling it with a hundred
    /// different colors does not either.
    pub fn set_color(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        renderer: &mut crate::render::Renderer3D,
        color: [f32; 3],
    ) {
        if color == self.color {
            return;
        }
        self.color = color;
        let old = self.material;
        self.material = renderer.add_material(
            device,
            queue,
            &crate::render::MaterialDesc::flat([color[0], color[1], color[2], 1.0]),
        );
        renderer.free_material(old);
    }

    /// The sample step every tile wants, from its distance to `eye`.
    ///
    /// Computed for ALL tiles, including culled ones, because a visible tile's
    /// edges have to match what its neighbour WOULD build. Deriving a seam from
    /// a partially-filled table is exactly how stitched terrain cracks.
    ///
    /// Distance is the real 3D distance to the tile's bounding sphere, so a
    /// camera high above the ground coarsens what is far below it, the way
    /// screen-space error actually behaves.
    fn desired_steps(&self, eye: Vec3) -> Vec<u32> {
        let max_step = self.field.max_step();
        self.tiles
            .iter()
            .map(|t| {
                let d = (t.centre - eye).length() - t.radius;
                let ratio = (d / self.lod_range).max(1.0);
                let level = ratio.log2().floor().max(0.0) as u32;
                (1u32 << level.min(16)).clamp(1, max_step)
            })
            .collect()
    }

    /// Choose per-tile detail, refresh the meshes whose detail changed, and
    /// queue every visible tile.
    pub fn draw(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        renderer: &mut crate::render::Renderer3D,
        eye: Vec3,
    ) {
        self.frame += 1;
        let steps = self.desired_steps(eye);
        let planes = crate::render::frustum_planes(renderer.view_proj());
        let per = self.tiles_per_side;
        let (mut drawn, mut finest, mut coarsest) = (0usize, u32::MAX, 1u32);
        for tz in 0..per {
            for tx in 0..per {
                let i = (tz * per + tx) as usize;
                if !crate::render::sphere_in_frustum(
                    &planes,
                    self.tiles[i].centre,
                    self.tiles[i].radius,
                ) {
                    continue;
                }
                let step = steps[i];
                let neighbour = |dx: i32, dz: i32| -> u32 {
                    let (nx, nz) = (tx as i32 + dx, tz as i32 + dz);
                    if nx < 0 || nz < 0 || nx >= per as i32 || nz >= per as i32 {
                        step
                    } else {
                        steps[(nz as u32 * per + nx as u32) as usize]
                    }
                };
                let lod = TileLod {
                    step,
                    edge_step: [
                        step.max(neighbour(-1, 0)),
                        step.max(neighbour(1, 0)),
                        step.max(neighbour(0, -1)),
                        step.max(neighbour(0, 1)),
                    ],
                };
                let tile = &mut self.tiles[i];
                if tile.mesh.is_none() || tile.built != lod {
                    let mesh = self.field.tile_mesh(tx, tz, lod);
                    match tile.mesh {
                        Some(id) => renderer.update_mesh(device, queue, id, &mesh),
                        None => tile.mesh = Some(renderer.add_mesh(device, &mesh)),
                    }
                    tile.built = lod;
                    // The mesh may have grown (a coarse tile refined back to
                    // full detail reallocates); re-read rather than assume.
                    let now = tile.mesh.map_or(0, |id| renderer.mesh_bytes_of(id));
                    self.resident = self.resident - tile.bytes + now;
                    tile.bytes = now;
                }
                tile.last_used = self.frame;
                let Some(id) = tile.mesh else { continue };
                renderer.draw(
                    id,
                    self.material,
                    glam::Mat4::from_translation(tile.centre),
                    None,
                );
                drawn += 1;
                finest = finest.min(step);
                coarsest = coarsest.max(step);
            }
        }
        self.last_draw = (drawn, finest.min(coarsest), coarsest);
        self.evict_to_budget(renderer);
    }

    /// Drop least-recently-drawn tile meshes until the cache fits its budget.
    ///
    /// Runs after the draw so this frame's tiles are already stamped with the
    /// current frame and are therefore skipped: their meshes are queued for
    /// rendering, and freeing one would drop that geometry from the frame.
    fn evict_to_budget(&mut self, renderer: &mut crate::render::Renderer3D) {
        if self.resident <= self.budget {
            return;
        }
        // Candidates: resident tiles NOT drawn this frame, oldest first.
        let frame = self.frame;
        let mut scratch = std::mem::take(&mut self.evict_scratch);
        scratch.clear();
        scratch.extend(
            self.tiles
                .iter()
                .enumerate()
                .filter(|(_, t)| t.mesh.is_some() && t.last_used != frame)
                .map(|(i, t)| (t.last_used, i)),
        );
        scratch.sort_unstable();
        for &(_, i) in &scratch {
            if self.resident <= self.budget {
                break;
            }
            let tile = &mut self.tiles[i];
            if let Some(id) = tile.mesh.take() {
                renderer.free_mesh(id);
                self.resident -= tile.bytes;
                tile.bytes = 0;
                tile.last_used = 0;
                self.evicted += 1;
            }
        }
        self.evict_scratch = scratch;
        if self.resident > self.budget {
            // Every remaining tile is in this frame's view. Keeping them is the
            // right call, but it is worth counting: it means the budget is
            // smaller than the view distance actually needs.
            self.over_budget_frames += 1;
        }
    }
}

/// A 24-bit hash of a lattice point, in `[0, 1)`.
fn hash01(seed: u64, x: i64, y: i64) -> f32 {
    let mut h = seed
        ^ (x as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ (y as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F);
    h ^= h >> 30;
    h = h.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    h ^= h >> 27;
    h = h.wrapping_mul(0x94D0_49BB_1331_11EB);
    h ^= h >> 31;
    (h >> 40) as f32 / (1u32 << 24) as f32
}

/// Smooth value noise in `[0, 1]` on the integer lattice.
fn value_noise(seed: u64, x: f32, y: f32) -> f32 {
    let (x0, y0) = (x.floor(), y.floor());
    let (ix, iy) = (x0 as i64, y0 as i64);
    let smooth = |t: f32| t * t * (3.0 - 2.0 * t);
    let (tx, ty) = (smooth(x - x0), smooth(y - y0));
    let mix = |a: f32, b: f32, t: f32| a + (b - a) * t;
    let top = mix(hash01(seed, ix, iy), hash01(seed, ix + 1, iy), tx);
    let bot = mix(hash01(seed, ix, iy + 1), hash01(seed, ix + 1, iy + 1), tx);
    mix(top, bot, ty)
}

#[cfg(test)]
mod tests;
