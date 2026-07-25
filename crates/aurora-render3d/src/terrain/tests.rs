//! Terrain surface, file format, and level-of-detail meshing.
//!
//! The meshing tests are the ones that matter: a terrain that cracks at a level
//! of detail seam is obvious on screen and painful to find in code, so the
//! properties are asserted geometrically rather than eyeballed. Two independent
//! ones together rule cracks out:
//!
//! * every tile mesh covers its own footprint EXACTLY once (no hole, no
//!   overlap), checked by summing signed triangle area in XZ;
//! * two neighbouring tiles reference the exact same vertex positions along
//!   their shared boundary, whatever levels of detail they picked.

use super::*;

/// A terrain with a distinctive, non-planar surface, so a wrong triangle
/// diagonal or a dropped vertex changes a height rather than cancelling out.
fn bumpy(dim: u32) -> Heightfield {
    let n = dim as usize;
    let mut h = vec![0.0f32; n * n];
    for r in 0..n {
        for c in 0..n {
            let (x, z) = (c as f32 * 0.37, r as f32 * 0.29);
            h[r * n + c] = x.sin() * 3.0 + z.cos() * 2.0 + (x * z * 0.05).sin() * 4.0;
        }
    }
    Heightfield::new(dim, 1.5, -10.0, 7.0, h).expect("valid heightfield")
}

/// Total footprint area of a mesh in the XZ plane, counting overlaps twice and
/// holes not at all.
fn xz_area(m: &MeshData) -> f64 {
    m.indices
        .chunks_exact(3)
        .map(|t| {
            let p = |i: u32| m.vertices[i as usize].pos;
            let (a, b, c) = (p(t[0]), p(t[1]), p(t[2]));
            let ny = (b[2] - a[2]) as f64 * (c[0] - a[0]) as f64
                - (b[0] - a[0]) as f64 * (c[2] - a[2]) as f64;
            ny * 0.5
        })
        .sum()
}

/// World-space positions of the vertices a mesh actually REFERENCES, on the
/// plane `axis == value` (axis 0 = x, 2 = z), rounded so two tiles that computed
/// the same corner through different arithmetic still compare equal.
fn seam_vertices(m: &MeshData, centre: Vec3, axis: usize, value: f32) -> Vec<(i64, i64, i64)> {
    let mut used = vec![false; m.vertices.len()];
    for &i in &m.indices {
        used[i as usize] = true;
    }
    let q = |v: f32| (v as f64 * 4096.0).round() as i64;
    let mut out: Vec<(i64, i64, i64)> = m
        .vertices
        .iter()
        .enumerate()
        .filter(|(i, _)| used[*i])
        .map(|(_, v)| {
            [
                v.pos[0] + centre.x,
                v.pos[1] + centre.y,
                v.pos[2] + centre.z,
            ]
        })
        .filter(|p| (p[axis] - value).abs() < 1e-3)
        .map(|p| (q(p[0]), q(p[1]), q(p[2])))
        .collect();
    out.sort_unstable();
    out.dedup();
    out
}

/// Height of the surface a mesh describes at `(x, z)`, or `None` when no
/// triangle covers that point. Uses barycentric coordinates in XZ, so it reads
/// the mesh's own planar interpolation.
fn mesh_height(m: &MeshData, centre: Vec3, x: f64, z: f64) -> Option<f64> {
    for t in m.indices.chunks_exact(3) {
        let p = |i: u32| {
            let v = m.vertices[i as usize].pos;
            [
                (v[0] + centre.x) as f64,
                (v[1] + centre.y) as f64,
                (v[2] + centre.z) as f64,
            ]
        };
        let (a, b, c) = (p(t[0]), p(t[1]), p(t[2]));
        let det = (b[2] - a[2]) * (c[0] - a[0]) - (b[0] - a[0]) * (c[2] - a[2]);
        if det.abs() < 1e-12 {
            continue;
        }
        let w1 = ((z - a[2]) * (c[0] - a[0]) - (x - a[0]) * (c[2] - a[2])) / det;
        let w2 = ((b[2] - a[2]) * (x - a[0]) - (b[0] - a[0]) * (z - a[2])) / det;
        if w1 >= -1e-9 && w2 >= -1e-9 && w1 + w2 <= 1.0 + 1e-9 {
            return Some(a[1] + w1 * (b[1] - a[1]) + w2 * (c[1] - a[1]));
        }
    }
    None
}

// --- the surface ------------------------------------------------------------

/// `height_at` must evaluate the collider's triangulation, not a bilinear patch
/// over the quad. On this cell the two differ by 1.0 at the centre, so a
/// bilinear implementation cannot pass.
#[test]
fn height_evaluates_the_collider_triangulation_not_bilinear() {
    // One cell, heights h00=0, h01=0, h10=0, h11=4 (row-major: row 0 is z=0).
    let f = Heightfield::new(
        3,
        1.0,
        0.0,
        0.0,
        vec![0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0],
    )
    .expect("valid");
    // Cell (0,0) has h00 = h(0,0) = 0, h01 = h(0,1) = 0, h10 = h(1,0) = 0,
    // h11 = h(1,1) = 4. The diagonal p10-p01 runs from (x=0,z=1) to (x=1,z=0),
    // so the centre (0.5, 0.5) sits exactly on it, at height 0.
    assert!(
        f.height_at(0.5, 0.5).abs() < 1e-9,
        "cell centre is on the p10-p01 diagonal, so height 0; bilinear would say 1.0, got {}",
        f.height_at(0.5, 0.5)
    );
    // Just inside the u+v > 1 half, the surface climbs toward h11 = 4.
    assert!(
        (f.height_at(0.75, 0.75) - 2.0).abs() < 1e-9,
        "got {}",
        f.height_at(0.75, 0.75)
    );
    // The other half is flat at 0.
    assert!(f.height_at(0.25, 0.25).abs() < 1e-9);
    // Corners are exact samples.
    assert!((f.height_at(1.0, 1.0) - 4.0).abs() < 1e-6);
}

/// A fine sweep must be continuous: no step bigger than the surface's own
/// steepest slope allows, which is what "interpolated, not nearest neighbour"
/// means in practice. A nearest-neighbour lookup would jump a whole sample.
#[test]
fn height_is_continuous_along_a_fine_sweep() {
    let f = bumpy(65);
    let (x0, z0) = (f.origin_x() as f64, f.origin_z() as f64);
    let span = f.extent() as f64;
    let steps = 20_000;
    let dx = span / steps as f64;
    // The steepest the surface can be between two samples.
    let mut max_slope = 0.0f64;
    for r in 0..f.dim() as i64 {
        for c in 0..f.dim() as i64 - 1 {
            let d = (f.sample(r, c + 1) - f.sample(r, c)).abs() as f64;
            max_slope = max_slope.max(d / f.spacing() as f64);
        }
    }
    let z = z0 + span * 0.5;
    let mut prev = f.height_at(x0, z);
    let mut biggest = 0.0f64;
    for i in 1..=steps {
        let h = f.height_at(x0 + i as f64 * dx, z);
        biggest = biggest.max((h - prev).abs());
        prev = h;
    }
    let bound = max_slope * dx * 1.000_001 + 1e-9;
    assert!(
        biggest <= bound,
        "sweep jumped {biggest} in one {dx}-unit step; the surface's own slope allows at most {bound}"
    );
    assert!(biggest > 0.0, "a flat sweep would prove nothing");
}

/// Out of bounds is defined: the border height, clamped, in every direction and
/// for non-finite input.
#[test]
fn out_of_bounds_clamps_to_the_border() {
    let f = bumpy(17);
    let (x0, z0) = (f.origin_x() as f64, f.origin_z() as f64);
    let e = f.extent() as f64;
    assert_eq!(f.height_at(x0 - 1e6, z0 - 1e6), f.height_at(x0, z0));
    assert_eq!(
        f.height_at(x0 + e + 1e6, z0 + e + 1e6),
        f.height_at(x0 + e, z0 + e)
    );
    assert_eq!(
        f.height_at(x0 - 5.0, z0 + e * 0.5),
        f.height_at(x0, z0 + e * 0.5)
    );
    assert_eq!(
        f.height_at(f64::INFINITY, f64::INFINITY),
        f.height_at(x0 + e, z0 + e)
    );
    assert_eq!(
        f.height_at(f64::NEG_INFINITY, f64::NEG_INFINITY),
        f.height_at(x0, z0)
    );
    let nan = f.height_at(f64::NAN, f64::NAN);
    assert!(nan.is_finite(), "NaN input must not produce NaN, got {nan}");
    assert_eq!(nan, f.height_at(x0, z0));
}

// --- the file format --------------------------------------------------------

#[test]
fn aterr_round_trips_exactly() {
    let f = bumpy(33);
    let bytes = f.encode();
    assert_eq!(bytes.len(), HEADER_BYTES + 33 * 33 * 4);
    assert_eq!(&bytes[0..8], MAGIC);
    let g = Heightfield::decode(&bytes).expect("decode");
    assert_eq!(g.dim(), f.dim());
    assert_eq!(g.spacing(), f.spacing());
    assert_eq!(g.origin_x(), f.origin_x());
    assert_eq!(g.origin_z(), f.origin_z());
    assert_eq!(g.heights(), f.heights(), "samples must survive bit-exactly");
}

#[test]
fn aterr_rejects_malformed_files() {
    let good = bumpy(9).encode();

    let mut bad_magic = good.clone();
    bad_magic[0] = b'X';
    assert!(
        Heightfield::decode(&bad_magic).is_err(),
        "bad magic accepted"
    );

    assert!(
        Heightfield::decode(&good[..10]).is_err(),
        "truncated accepted"
    );
    assert!(Heightfield::decode(&[]).is_err(), "empty accepted");

    let mut short = good.clone();
    short.truncate(good.len() - 4);
    assert!(
        Heightfield::decode(&short).is_err(),
        "a file one sample short was accepted"
    );

    // A dim that is not 2^k + 1 cannot tile, so it must be refused rather than
    // quietly meshed wrong.
    let mut odd = good.clone();
    odd[8..12].copy_from_slice(&10u32.to_le_bytes());
    assert!(Heightfield::decode(&odd).is_err(), "dim 10 accepted");

    // A huge dim must be rejected on the header, before anything is allocated.
    let mut huge = good.clone();
    huge[8..12].copy_from_slice(&u32::MAX.to_le_bytes());
    assert!(Heightfield::decode(&huge).is_err(), "dim u32::MAX accepted");
}

#[test]
fn constructor_rejects_bad_shapes() {
    assert!(
        Heightfield::new(2, 1.0, 0.0, 0.0, vec![0.0; 4]).is_err(),
        "dim 2"
    );
    assert!(
        Heightfield::new(10, 1.0, 0.0, 0.0, vec![0.0; 100]).is_err(),
        "dim 10"
    );
    assert!(
        Heightfield::new(5, 0.0, 0.0, 0.0, vec![0.0; 25]).is_err(),
        "spacing 0"
    );
    assert!(
        Heightfield::new(5, -1.0, 0.0, 0.0, vec![0.0; 25]).is_err(),
        "negative spacing"
    );
    assert!(
        Heightfield::new(5, 1.0, 0.0, 0.0, vec![0.0; 24]).is_err(),
        "short samples"
    );
    let mut nan = vec![0.0f32; 25];
    nan[7] = f32::NAN;
    assert!(
        Heightfield::new(5, 1.0, 0.0, 0.0, nan).is_err(),
        "NaN sample"
    );
    assert!(
        Heightfield::new(5, 1.0, 0.0, 0.0, vec![0.0; 25]).is_ok(),
        "dim 5 is valid"
    );
}

#[test]
fn generate_is_deterministic_and_bounded() {
    let a = Heightfield::generate(1234, 65, 2.0, 30.0).expect("generate");
    let b = Heightfield::generate(1234, 65, 2.0, 30.0).expect("generate");
    assert_eq!(
        a.heights(),
        b.heights(),
        "same seed must give the same terrain"
    );
    let c = Heightfield::generate(1235, 65, 2.0, 30.0).expect("generate");
    assert_ne!(a.heights(), c.heights(), "a different seed must differ");
    let lo = a.heights().iter().cloned().fold(f32::INFINITY, f32::min);
    let hi = a
        .heights()
        .iter()
        .cloned()
        .fold(f32::NEG_INFINITY, f32::max);
    assert!(
        lo >= 0.0 && hi <= 30.0,
        "heights must stay in [0, amplitude], got {lo}..{hi}"
    );
    assert!(
        hi - lo > 5.0,
        "the terrain must actually have relief, got {lo}..{hi}"
    );
    // Centred on the world origin.
    assert!((a.origin_x() + a.extent() * 0.5).abs() < 1e-3);
    assert!(
        Heightfield::generate(1, 64, 1.0, 1.0).is_err(),
        "dim 64 is not 2^k + 1"
    );
}

// --- the level-of-detail mesher --------------------------------------------

/// Every LOD, and every combination of coarsened edges, must cover the tile
/// footprint exactly once. Signed area catches BOTH failure modes at once: a
/// hole subtracts, an overlap or a backwards triangle adds.
#[test]
fn every_tile_configuration_covers_its_footprint_exactly() {
    let f = bumpy(65);
    let t = f.tile_cells();
    let want = (t as f64 * f.spacing() as f64).powi(2);
    let steps = [1u32, 2, 4, 8, 16];
    for &step in &steps {
        if step > f.max_step() {
            continue;
        }
        for mask in 0..16u32 {
            let mut edge_step = [step; 4];
            for (e, es) in edge_step.iter_mut().enumerate() {
                if mask & (1 << e) != 0 {
                    // A coarser neighbour: two levels up, which also exercises
                    // ratios above 2:1.
                    *es = (step * if e % 2 == 0 { 2 } else { 4 }).min(t);
                }
            }
            let lod = TileLod { step, edge_step };
            let m = f.tile_mesh(1, 1, lod);
            let area = xz_area(&m);
            assert!(
                (area - want).abs() < want * 1e-4,
                "step {step} mask {mask}: covered {area}, footprint is {want}"
            );
            assert!(
                m.indices.len().is_multiple_of(3) && !m.indices.is_empty(),
                "step {step} mask {mask}: produced no triangles"
            );
        }
    }
}

/// The seam proof: two adjacent tiles at different levels of detail must
/// reference the SAME vertex positions along their shared edge. Same positions
/// means no T-junction and no gap, at any camera angle.
#[test]
fn neighbouring_tiles_share_their_seam_vertices() {
    let f = bumpy(129);
    let t = f.tile_cells();
    let per = f.tiles_per_side();
    assert!(per >= 3, "need interior tiles to test seams, got {per}");
    let mut checked = 0;
    for &(fine, coarse) in &[(1u32, 2u32), (1, 4), (2, 4), (2, 8), (4, 16), (1, 16)] {
        if coarse > f.max_step() {
            continue;
        }
        // Tile (1,1) is fine and tile (2,1) to its +X is coarse, so the fine
        // tile's +X edge is built at the coarse tile's step.
        let a = f.tile_mesh(
            1,
            1,
            TileLod {
                step: fine,
                edge_step: [fine, coarse, fine, fine],
            },
        );
        let b = f.tile_mesh(
            2,
            1,
            TileLod {
                step: coarse,
                edge_step: [coarse; 4],
            },
        );
        let seam_x = f.origin_x() + (2 * t) as f32 * f.spacing();
        let (ca, _) = f.tile_bounds(1, 1);
        let (cb, _) = f.tile_bounds(2, 1);
        let va = seam_vertices(&a, ca, 0, seam_x);
        let vb = seam_vertices(&b, cb, 0, seam_x);
        assert_eq!(
            va,
            vb,
            "fine step {fine} vs coarse step {coarse}: the shared +X edge has {} vertices on \
             the fine side and {} on the coarse side",
            va.len(),
            vb.len()
        );
        assert_eq!(
            va.len(),
            (t / coarse) as usize + 1,
            "the seam must be built at the coarse step {coarse}"
        );

        // The same across a -Z / +Z seam, so the other axis is not just
        // accidentally right.
        let c = f.tile_mesh(
            1,
            1,
            TileLod {
                step: fine,
                edge_step: [fine, fine, fine, coarse],
            },
        );
        let d = f.tile_mesh(
            1,
            2,
            TileLod {
                step: coarse,
                edge_step: [coarse; 4],
            },
        );
        let seam_z = f.origin_z() + (2 * t) as f32 * f.spacing();
        let (cc, _) = f.tile_bounds(1, 1);
        let (cd, _) = f.tile_bounds(1, 2);
        assert_eq!(
            seam_vertices(&c, cc, 2, seam_z),
            seam_vertices(&d, cd, 2, seam_z),
            "fine step {fine} vs coarse step {coarse}: +Z seam differs"
        );
        checked += 1;
    }
    assert!(checked >= 4, "only {checked} step pairs were exercised");
}

/// Every terrain triangle must face up, or the renderer's back-face culling
/// would punch holes in the ground that look exactly like LOD cracks.
#[test]
fn every_terrain_triangle_faces_up() {
    let f = bumpy(65);
    for &step in &[1u32, 2, 4, 8] {
        let m = f.tile_mesh(
            0,
            0,
            TileLod {
                step,
                edge_step: [step, step * 2, step * 4, step],
            },
        );
        for (n, t) in m.indices.chunks_exact(3).enumerate() {
            let p = |i: u32| m.vertices[i as usize].pos;
            let (a, b, c) = (p(t[0]), p(t[1]), p(t[2]));
            let ny = (b[2] - a[2]) * (c[0] - a[0]) - (b[0] - a[0]) * (c[2] - a[2]);
            assert!(
                ny > 0.0,
                "step {step} triangle {n} is wound downward (ny = {ny})"
            );
        }
    }
}

/// At full detail the drawn surface must BE the collider surface, so what a
/// player sees under their feet is what they stand on.
#[test]
fn the_full_detail_mesh_is_the_collider_surface() {
    let f = bumpy(65);
    let t = f.tile_cells();
    let lod = TileLod {
        step: 1,
        edge_step: [1; 4],
    };
    let m = f.tile_mesh(1, 1, lod);
    let (centre, _) = f.tile_bounds(1, 1);
    let x0 = f.origin_x() as f64 + (t as f64) * f.spacing() as f64;
    let z0 = f.origin_z() as f64 + (t as f64) * f.spacing() as f64;
    let span = t as f64 * f.spacing() as f64;
    let mut worst = 0.0f64;
    let mut n = 0;
    for i in 0..=40 {
        for j in 0..=40 {
            // Keep clear of the outer boundary, where a neighbouring tile owns
            // the surface and this mesh's triangles simply end.
            let x = x0 + span * (0.01 + 0.98 * i as f64 / 40.0);
            let z = z0 + span * (0.01 + 0.98 * j as f64 / 40.0);
            let Some(mh) = mesh_height(&m, centre, x, z) else {
                panic!("no triangle covers ({x}, {z}) inside the tile");
            };
            worst = worst.max((mh - f.height_at(x, z)).abs());
            n += 1;
        }
    }
    assert!(n > 1000, "only {n} samples");
    assert!(
        worst < 1e-3,
        "the full-detail mesh disagrees with height_at by {worst}"
    );
}

/// Every tile vertex has to sit on a real sample, not on an interpolated or
/// stale height: that is what keeps the LOD skeleton anchored to the data.
#[test]
fn tile_vertices_sit_on_heightfield_samples() {
    let f = bumpy(65);
    let t = f.tile_cells();
    let step = 4;
    let m = f.tile_mesh(
        1,
        2,
        TileLod {
            step,
            edge_step: [step, step * 2, step, step],
        },
    );
    let (centre, _) = f.tile_bounds(1, 2);
    let mut used = vec![false; m.vertices.len()];
    for &i in &m.indices {
        used[i as usize] = true;
    }
    for (i, v) in m.vertices.iter().enumerate() {
        if !used[i] {
            continue;
        }
        let x = v.pos[0] + centre.x;
        let z = v.pos[2] + centre.z;
        let col = ((x - f.origin_x()) / f.spacing()).round() as i64;
        let row = ((z - f.origin_z()) / f.spacing()).round() as i64;
        assert!(
            (x - (f.origin_x() + col as f32 * f.spacing())).abs() < 1e-3,
            "vertex {i} is not on a sample column"
        );
        assert!(
            (v.pos[1] + centre.y - f.sample(row, col)).abs() < 1e-3,
            "vertex {i} height {} is not sample ({row},{col}) = {}",
            v.pos[1] + centre.y,
            f.sample(row, col)
        );
        // ...and inside the tile it belongs to.
        assert!(
            col >= t as i64 && col <= 2 * t as i64,
            "column {col} escaped the tile"
        );
    }
}

/// Normals come from the heightfield gradient, so a slope leans and a flat is
/// straight up - and the SAME vertex gets the same normal at every level of
/// detail, which is what stops lighting from popping at an LOD change.
#[test]
fn normals_follow_the_surface_and_do_not_depend_on_lod() {
    let flat = Heightfield::new(5, 1.0, 0.0, 0.0, vec![3.0; 25]).expect("flat");
    let n = flat.normal_at_sample(2, 2);
    assert!((n - Vec3::Y).length() < 1e-6, "flat ground normal is {n:?}");

    // A constant slope of 1 in x: the normal tilts against +X by 45 degrees.
    let mut h = vec![0.0f32; 25];
    for r in 0..5 {
        for c in 0..5 {
            h[r * 5 + c] = c as f32;
        }
    }
    let ramp = Heightfield::new(5, 1.0, 0.0, 0.0, h).expect("ramp");
    let n = ramp.normal_at_sample(2, 2);
    assert!(
        (n.x + std::f32::consts::FRAC_1_SQRT_2).abs() < 1e-5,
        "got {n:?}"
    );
    assert!(
        (n.y - std::f32::consts::FRAC_1_SQRT_2).abs() < 1e-5,
        "got {n:?}"
    );
    // One-sided at the border: the slope is constant, so the normal is too.
    let edge = ramp.normal_at_sample(0, 0);
    assert!(
        (edge - n).length() < 1e-5,
        "border normal {edge:?} != interior {n:?}"
    );

    // Same world position, two levels of detail, one normal.
    let f = bumpy(65);
    let fine = f.tile_mesh(
        1,
        1,
        TileLod {
            step: 1,
            edge_step: [1; 4],
        },
    );
    let coarse = f.tile_mesh(
        1,
        1,
        TileLod {
            step: 4,
            edge_step: [4; 4],
        },
    );
    let (centre, _) = f.tile_bounds(1, 1);
    let corner = |m: &MeshData| {
        m.vertices
            .iter()
            .find(|v| (v.pos[0] + centre.x - f.origin_x() - 48.0 * 1.5).abs() < 1e-3)
            .map(|v| Vec3::from(v.normal))
    };
    if let (Some(a), Some(b)) = (corner(&fine), corner(&coarse)) {
        assert!(
            (a - b).length() < 1e-5,
            "normal changed with LOD: {a:?} vs {b:?}"
        );
    }
}

/// The rendered seam check.
///
/// A top-down view over a terrain whose tiles span two levels of detail, drawn
/// headless against a clear color no lit surface can produce. Any background
/// pixel is a hole through the ground, which is what an LOD crack looks like.
///
/// The test proves its own premise both ways before concluding anything:
/// a control render pointed away from the terrain MUST report background (so
/// the detector works), and the real render MUST report more than one sample
/// step among the tiles it queued (so a seam was actually on screen).
#[test]
fn a_view_across_an_lod_seam_renders_with_no_hole() {
    let _g = crate::gpu_guard();
    let Some((device, queue)) = crate::headless_device() else {
        eprintln!("no GPU adapter - skipping the terrain seam render");
        return;
    };
    let (w, h) = (320u32, 320u32);
    let mut scene = crate::Scene::new(&device, &queue, wgpu::TextureFormat::Rgba8Unorm, w, h, 1);
    let field = Heightfield::generate(20_260_725, 513, 1.0, 30.0).expect("generate");
    scene.set_terrain(&device, &queue, std::sync::Arc::new(field));
    scene.set_terrain_color(&device, &queue, [0.35, 0.42, 0.25]);
    scene.set_light(Vec3::new(0.4, 1.0, 0.3), Vec3::ONE, 0.35);

    // Magenta: the terrain material is a dull green under a white light, so no
    // shaded terrain pixel can be mistaken for the background.
    let clear = [1.0f32, 0.0, 1.0, 1.0];
    scene.set_clear(clear[0], clear[1], clear[2]);
    let is_background = |p: &[u8]| p[0] > 200 && p[1] < 60 && p[2] > 200;
    let count_background = |img: &[u8]| img.chunks_exact(4).filter(|p| is_background(p)).count();

    // Control: look up at nothing. If this does not find background, the
    // detector is broken and the real check below would pass vacuously.
    scene.set_camera(Vec3::new(0.0, 200.0, 0.0), Vec3::new(0.0, 400.0, 0.0), 90.0);
    scene.begin();
    scene.draw_terrain(&device, &queue);
    let control = crate::render_offscreen(&mut scene.renderer, &device, &queue, w, h, clear);
    assert_eq!(
        count_background(&control),
        (w * h) as usize,
        "the background detector does not recognise an empty frame"
    );

    // The real view: straight down from 110 units up with a 90 degree field of
    // view. The frustum corners reach a radius of 156 on a terrain that runs to
    // 256, so every pixel is over ground, and the near and far tiles land on
    // opposite sides of the first level-of-detail threshold.
    scene.set_camera(Vec3::new(0.0, 110.0, 0.0), Vec3::new(0.0, 0.0, 0.001), 90.0);
    scene.begin();
    scene.draw_terrain(&device, &queue);
    let (drawn, finest, coarsest) = scene.terrain_last_draw().expect("terrain is loaded");
    assert!(drawn > 16, "only {drawn} tiles were queued");
    assert!(
        finest < coarsest,
        "no LOD seam was in view (every tile used step {finest}), so this proves nothing"
    );

    let img = crate::render_offscreen(&mut scene.renderer, &device, &queue, w, h, clear);
    let holes = count_background(&img);
    assert_eq!(
        holes, 0,
        "{holes} background pixels showed through the terrain across an LOD seam \
         ({drawn} tiles, steps {finest}..{coarsest})"
    );
}

/// A terrain smaller than one tile still tiles, meshes, and covers itself: the
/// smallest legal field must not be a special case that only fails in the field.
#[test]
fn a_single_tile_terrain_still_meshes() {
    let f = bumpy(5);
    assert_eq!(f.tiles_per_side(), 1);
    assert_eq!(f.tile_cells(), 4);
    let want = (4.0 * f.spacing() as f64).powi(2);
    for step in [1u32, 2] {
        for mask in 0..16u32 {
            let mut edge_step = [step; 4];
            for (e, es) in edge_step.iter_mut().enumerate() {
                if mask & (1 << e) != 0 {
                    *es = (step * 2).min(4);
                }
            }
            let m = f.tile_mesh(0, 0, TileLod { step, edge_step });
            let area = xz_area(&m);
            assert!(
                (area - want).abs() < want * 1e-4,
                "step {step} mask {mask}: covered {area} of {want}"
            );
        }
    }
}
