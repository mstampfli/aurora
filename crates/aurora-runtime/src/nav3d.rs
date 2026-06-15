//! 3D pathfinding for Aurora: a 26-connected voxel grid A* (`nav3d_*`) and a
//! polygon **navmesh** pathfinder (`navmesh_*`) that runs A* over a triangle
//! adjacency graph and then string-pulls the corridor with the Simple Stupid
//! Funnel algorithm to produce a smooth path of waypoints.

use std::cell::RefCell;

use pathfinding::prelude::astar;

type V3 = [f64; 3];

fn sub(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn dist(a: V3, b: V3) -> f64 {
    let d = sub(a, b);
    (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt()
}

// --- 26-connected voxel grid A* --------------------------------------------

struct Grid3 {
    w: i32,
    h: i32,
    d: i32,
    walls: Vec<bool>,
    path: Vec<(i32, i32, i32)>,
}
thread_local! {
    static GRID3: RefCell<Option<Grid3>> = const { RefCell::new(None) };
}

fn gidx(g: &Grid3, x: i32, y: i32, z: i32) -> usize {
    ((z * g.h + y) * g.w + x) as usize
}

#[no_mangle]
pub extern "C" fn aurora_nav3d_init(w: i64, h: i64, d: i64) {
    let (w, h, d) = (w.max(0) as i32, h.max(0) as i32, d.max(0) as i32);
    let g = Grid3 { w, h, d, walls: vec![false; (w * h * d).max(0) as usize], path: Vec::new() };
    GRID3.with(|x| *x.borrow_mut() = Some(g));
}

#[no_mangle]
pub extern "C" fn aurora_nav3d_wall(x: i64, y: i64, z: i64, blocked: i64) {
    GRID3.with(|g| {
        let mut g = g.borrow_mut();
        let Some(g) = g.as_mut() else { return };
        let (x, y, z) = (x as i32, y as i32, z as i32);
        if x >= 0 && y >= 0 && z >= 0 && x < g.w && y < g.h && z < g.d {
            let i = gidx(g, x, y, z);
            g.walls[i] = blocked != 0;
        }
    });
}

/// A* from (sx,sy,sz) to (gx,gy,gz) over a 26-connected grid; returns path length
/// in cells, or -1. Costs are scaled Euclidean distances.
#[no_mangle]
pub extern "C" fn aurora_nav3d_find(sx: i64, sy: i64, sz: i64, gx: i64, gy: i64, gz: i64) -> i64 {
    GRID3.with(|g| {
        let mut g = g.borrow_mut();
        let Some(g) = g.as_mut() else { return -1 };
        let (w, h, d) = (g.w, g.h, g.d);
        let walls = g.walls.clone();
        let goal = (gx as i32, gy as i32, gz as i32);
        let blocked = |x: i32, y: i32, z: i32| -> bool {
            x < 0 || y < 0 || z < 0 || x >= w || y >= h || z >= d || walls[((z * h + y) * w + x) as usize]
        };
        let result = astar(
            &(sx as i32, sy as i32, sz as i32),
            |&(x, y, z)| {
                let mut out: Vec<((i32, i32, i32), i64)> = Vec::new();
                for dz in -1..=1 {
                    for dy in -1..=1 {
                        for dx in -1..=1 {
                            if dx == 0 && dy == 0 && dz == 0 {
                                continue;
                            }
                            let (nx, ny, nz) = (x + dx, y + dy, z + dz);
                            if blocked(nx, ny, nz) {
                                continue;
                            }
                            let step = (((dx * dx + dy * dy + dz * dz) as f64).sqrt() * 1000.0) as i64;
                            out.push(((nx, ny, nz), step));
                        }
                    }
                }
                out
            },
            |&(x, y, z)| {
                let (dx, dy, dz) = ((x - goal.0) as f64, (y - goal.1) as f64, (z - goal.2) as f64);
                ((dx * dx + dy * dy + dz * dz).sqrt() * 1000.0) as i64
            },
            |&p| p == goal,
        );
        match result {
            Some((path, _)) => {
                let n = path.len() as i64;
                g.path = path;
                n
            }
            None => {
                g.path.clear();
                -1
            }
        }
    })
}

fn grid_cell(i: i64, axis: usize) -> i64 {
    GRID3.with(|g| {
        g.borrow()
            .as_ref()
            .and_then(|g| g.path.get(i.max(0) as usize))
            .map(|&(x, y, z)| [x, y, z][axis] as i64)
            .unwrap_or(-1)
    })
}
#[no_mangle]
pub extern "C" fn aurora_nav3d_x(i: i64) -> i64 { grid_cell(i, 0) }
#[no_mangle]
pub extern "C" fn aurora_nav3d_y(i: i64) -> i64 { grid_cell(i, 1) }
#[no_mangle]
pub extern "C" fn aurora_nav3d_z(i: i64) -> i64 { grid_cell(i, 2) }

// --- polygon navmesh + funnel ----------------------------------------------

struct NavMesh {
    tris: Vec<[V3; 3]>,
    centroids: Vec<V3>,
    /// For each triangle, its neighbors as (neighbor index, shared edge a, b).
    neighbors: Vec<Vec<(usize, V3, V3)>>,
    path: Vec<V3>,
}
thread_local! {
    static NAVMESH: RefCell<Option<NavMesh>> = const { RefCell::new(None) };
}

/// Build a navmesh from `vcount*3` vertex floats and `icount` triangle indices.
/// Triangles sharing an edge (two vertices) become neighbors.
#[no_mangle]
pub extern "C" fn aurora_navmesh_build(
    verts: *const f64, vcount: i64, indices: *const i64, icount: i64,
) -> i64 {
    if verts.is_null() || indices.is_null() || vcount <= 0 || icount < 3 {
        return -1;
    }
    let vs = unsafe { std::slice::from_raw_parts(verts, (vcount * 3) as usize) };
    let is = unsafe { std::slice::from_raw_parts(indices, icount as usize) };
    let vert = |i: i64| -> V3 {
        let i = i.max(0) as usize;
        [vs[i * 3], vs[i * 3 + 1], vs[i * 3 + 2]]
    };
    let mut tris: Vec<[V3; 3]> = Vec::new();
    let mut idx_tris: Vec<[i64; 3]> = Vec::new();
    for c in is.chunks_exact(3) {
        tris.push([vert(c[0]), vert(c[1]), vert(c[2])]);
        idx_tris.push([c[0], c[1], c[2]]);
    }
    let centroids: Vec<V3> = tris
        .iter()
        .map(|t| {
            [
                (t[0][0] + t[1][0] + t[2][0]) / 3.0,
                (t[0][1] + t[1][1] + t[2][1]) / 3.0,
                (t[0][2] + t[1][2] + t[2][2]) / 3.0,
            ]
        })
        .collect();

    // Two triangles are neighbors if they share two vertex indices.
    let mut neighbors = vec![Vec::new(); tris.len()];
    for a in 0..idx_tris.len() {
        for b in (a + 1)..idx_tris.len() {
            let shared: Vec<i64> =
                idx_tris[a].iter().copied().filter(|v| idx_tris[b].contains(v)).collect();
            if shared.len() == 2 {
                let (e0, e1) = (vert(shared[0]), vert(shared[1]));
                neighbors[a].push((b, e0, e1));
                neighbors[b].push((a, e0, e1));
            }
        }
    }

    NAVMESH.with(|n| *n.borrow_mut() = Some(NavMesh { tris, centroids, neighbors, path: Vec::new() }));
    0
}

fn nearest_tri(nm: &NavMesh, p: V3) -> Option<usize> {
    (0..nm.centroids.len()).min_by(|&a, &b| {
        dist(nm.centroids[a], p).partial_cmp(&dist(nm.centroids[b], p)).unwrap()
    })
}

/// Find a smooth path from (sx,sy,sz) to (gx,gy,gz) across the navmesh; returns
/// the number of waypoints, or -1. Read them with `navmesh_x/y/z`.
#[no_mangle]
pub extern "C" fn aurora_navmesh_find(sx: f64, sy: f64, sz: f64, gx: f64, gy: f64, gz: f64) -> i64 {
    NAVMESH.with(|n| {
        let mut n = n.borrow_mut();
        let Some(nm) = n.as_mut() else { return -1 };
        let start = [sx, sy, sz];
        let goal = [gx, gy, gz];
        let (Some(st), Some(gt)) = (nearest_tri(nm, start), nearest_tri(nm, goal)) else {
            return -1;
        };
        // A* over the triangle adjacency graph (centroid distances).
        let result = astar(
            &st,
            |&t| {
                nm.neighbors[t]
                    .iter()
                    .map(|&(nb, _, _)| (nb, (dist(nm.centroids[t], nm.centroids[nb]) * 1000.0) as i64))
                    .collect::<Vec<_>>()
            },
            |&t| (dist(nm.centroids[t], goal) * 1000.0) as i64,
            |&t| t == gt,
        );
        let Some((corridor, _)) = result else {
            nm.path.clear();
            return -1;
        };

        // Build portals (left,right) from the shared edges between consecutive
        // corridor triangles, oriented by travel direction in the XZ plane.
        let mut portals: Vec<(V3, V3)> = vec![(start, start)];
        for w in corridor.windows(2) {
            let (a, b) = (w[0], w[1]);
            if let Some(&(_, e0, e1)) = nm.neighbors[a].iter().find(|&&(nb, _, _)| nb == b) {
                let travel = sub(nm.centroids[b], nm.centroids[a]);
                // cross of travel x (e0-centroid_a) in XZ decides left/right.
                let rel = sub(e0, nm.centroids[a]);
                let cross = travel[0] * rel[2] - travel[2] * rel[0];
                if cross > 0.0 {
                    portals.push((e0, e1));
                } else {
                    portals.push((e1, e0));
                }
            }
        }
        portals.push((goal, goal));

        let xz = funnel(&portals);
        // Reattach height by interpolating onto the nearest corridor triangle.
        nm.path = xz.iter().map(|&p| with_height(nm, &corridor, p)).collect();
        nm.path.len() as i64
    })
}

/// Simple Stupid Funnel over portals given as (left, right), in the XZ plane.
#[allow(unused_assignments)]
fn funnel(portals: &[(V3, V3)]) -> Vec<V3> {
    let tri_area2 = |a: V3, b: V3, c: V3| (b[0] - a[0]) * (c[2] - a[2]) - (c[0] - a[0]) * (b[2] - a[2]);
    let eq = |a: V3, b: V3| (a[0] - b[0]).abs() < 1e-5 && (a[2] - b[2]).abs() < 1e-5;

    let mut pts: Vec<V3> = Vec::new();
    if portals.is_empty() {
        return pts;
    }
    let mut apex = portals[0].0;
    let mut left = portals[0].0;
    let mut right = portals[0].1;
    let (mut apex_i, mut left_i, mut right_i) = (0usize, 0usize, 0usize);
    pts.push(apex);

    let mut i = 1;
    while i < portals.len() {
        let (l, r) = portals[i];

        // Update right side.
        if tri_area2(apex, right, r) <= 0.0 {
            if eq(apex, right) || tri_area2(apex, left, r) > 0.0 {
                right = r;
                right_i = i;
            } else {
                pts.push(left);
                apex = left;
                apex_i = left_i;
                left = apex;
                right = apex;
                left_i = apex_i;
                right_i = apex_i;
                i = apex_i + 1;
                continue;
            }
        }
        // Update left side.
        if tri_area2(apex, left, l) >= 0.0 {
            if eq(apex, left) || tri_area2(apex, right, l) < 0.0 {
                left = l;
                left_i = i;
            } else {
                pts.push(right);
                apex = right;
                apex_i = right_i;
                left = apex;
                right = apex;
                left_i = apex_i;
                right_i = apex_i;
                i = apex_i + 1;
                continue;
            }
        }
        i += 1;
    }
    let goal = portals[portals.len() - 1].0;
    if pts.last().map(|&p| !eq(p, goal)).unwrap_or(true) {
        pts.push(goal);
    }
    pts
}

/// Interpolate the navmesh height for an XZ point by barycentric coordinates on
/// the nearest corridor triangle.
fn with_height(nm: &NavMesh, corridor: &[usize], p: V3) -> V3 {
    let mut best = (f64::INFINITY, p[1]);
    for &t in corridor {
        let tri = nm.tris[t];
        if let Some(y) = bary_height(tri, p[0], p[2]) {
            return [p[0], y, p[2]];
        }
        // Fallback: distance to centroid, take its height if nothing contains p.
        let dc = dist(nm.centroids[t], [p[0], nm.centroids[t][1], p[2]]);
        if dc < best.0 {
            best = (dc, nm.centroids[t][1]);
        }
    }
    [p[0], best.1, p[2]]
}

fn bary_height(tri: [V3; 3], x: f64, z: f64) -> Option<f64> {
    let (a, b, c) = (tri[0], tri[1], tri[2]);
    let det = (b[2] - c[2]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[2] - c[2]);
    if det.abs() < 1e-9 {
        return None;
    }
    let l1 = ((b[2] - c[2]) * (x - c[0]) + (c[0] - b[0]) * (z - c[2])) / det;
    let l2 = ((c[2] - a[2]) * (x - c[0]) + (a[0] - c[0]) * (z - c[2])) / det;
    let l3 = 1.0 - l1 - l2;
    let eps = -1e-4;
    if l1 >= eps && l2 >= eps && l3 >= eps {
        Some(l1 * a[1] + l2 * b[1] + l3 * c[1])
    } else {
        None
    }
}

fn nav_wp(i: i64, axis: usize) -> f64 {
    NAVMESH.with(|n| {
        n.borrow().as_ref().and_then(|n| n.path.get(i.max(0) as usize)).map(|p| p[axis]).unwrap_or(0.0)
    })
}
#[no_mangle]
pub extern "C" fn aurora_navmesh_x(i: i64) -> f64 { nav_wp(i, 0) }
#[no_mangle]
pub extern "C" fn aurora_navmesh_y(i: i64) -> f64 { nav_wp(i, 1) }
#[no_mangle]
pub extern "C" fn aurora_navmesh_z(i: i64) -> f64 { nav_wp(i, 2) }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grid3_finds_a_straight_path() {
        aurora_nav3d_init(5, 5, 5);
        let n = aurora_nav3d_find(0, 0, 0, 4, 0, 0);
        assert!(n >= 2, "expected a path, got {n}");
        assert_eq!(aurora_nav3d_x(0), 0);
        assert_eq!(aurora_nav3d_x(n - 1), 4);
    }

    #[test]
    fn navmesh_paths_across_two_quads() {
        // Two quads (4 triangles) forming a 4x2 strip on the ground (y=0).
        // Vertices: a grid of 3x2 points along x.
        let verts: Vec<f64> = vec![
            0.0, 0.0, 0.0, // 0
            0.0, 0.0, 2.0, // 1
            2.0, 0.0, 0.0, // 2
            2.0, 0.0, 2.0, // 3
            4.0, 0.0, 0.0, // 4
            4.0, 0.0, 2.0, // 5
        ];
        let idx: Vec<i64> = vec![0, 1, 2, 2, 1, 3, 2, 3, 4, 4, 3, 5];
        let built = aurora_navmesh_build(verts.as_ptr(), 6, idx.as_ptr(), idx.len() as i64);
        assert_eq!(built, 0);
        let n = aurora_navmesh_find(0.5, 0.0, 1.0, 3.5, 0.0, 1.0);
        assert!(n >= 2, "expected waypoints, got {n}");
        // Path should progress in +x from near 0.5 to near 3.5.
        assert!(aurora_navmesh_x(0) < aurora_navmesh_x(n - 1));
        assert!((aurora_navmesh_x(n - 1) - 3.5).abs() < 0.6);
    }
}
