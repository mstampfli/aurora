//! 3D physics for Aurora, backed by Rapier 3D: rigid bodies (box/sphere/capsule
//! and arbitrary static trimeshes), impulses (jumps/knockback), raycasts, and a
//! kinematic capsule character controller that slides along walls - the core of
//! a fluid 3D movement shooter.
//!
//! Bodies are referenced by an `i64` handle (insertion order). State lives in a
//! thread-local, matching the single-threaded program the runtime serves.

use std::cell::RefCell;

use rapier3d::control::{CharacterLength, KinematicCharacterController};
use rapier3d::na::{Quaternion, UnitQuaternion};
use rapier3d::parry::query::ShapeCastOptions;
use rapier3d::prelude::*;

struct Phys3 {
    gravity: Vector<Real>,
    params: IntegrationParameters,
    pipeline: PhysicsPipeline,
    islands: IslandManager,
    broad: DefaultBroadPhase,
    narrow: NarrowPhase,
    bodies: RigidBodySet,
    colliders: ColliderSet,
    impulse: ImpulseJointSet,
    multibody: MultibodyJointSet,
    ccd: CCDSolver,
    query: QueryPipeline,
    handles: Vec<RigidBodyHandle>,
    cols: Vec<ColliderHandle>,
    grounded: Vec<bool>,
    controller: KinematicCharacterController,
    // Last raycast/shapecast hit (for `phys3d_hit_*`).
    hit_point: [f64; 3],
    hit_normal: [f64; 3],
    hit_body: i64,
}

thread_local! {
    static PHYS3: RefCell<Option<Phys3>> = const { RefCell::new(None) };
}

/// Create (or reset) the 3D physics world with gravity `(gx, gy, gz)`.
#[no_mangle]
pub extern "C" fn aurora_phys3d_init(gx: f64, gy: f64, gz: f64) {
    let mut controller = KinematicCharacterController::default();
    controller.up = Vector::y_axis();
    controller.offset = CharacterLength::Absolute(0.02);
    controller.slide = true;
    controller.snap_to_ground = Some(CharacterLength::Absolute(0.3));
    let p = Phys3 {
        gravity: vector![gx as Real, gy as Real, gz as Real],
        params: IntegrationParameters::default(),
        pipeline: PhysicsPipeline::new(),
        islands: IslandManager::new(),
        broad: DefaultBroadPhase::new(),
        narrow: NarrowPhase::new(),
        bodies: RigidBodySet::new(),
        colliders: ColliderSet::new(),
        impulse: ImpulseJointSet::new(),
        multibody: MultibodyJointSet::new(),
        ccd: CCDSolver::new(),
        query: QueryPipeline::new(),
        handles: Vec::new(),
        cols: Vec::new(),
        grounded: Vec::new(),
        controller,
        hit_point: [0.0; 3],
        hit_normal: [0.0; 3],
        hit_body: -1,
    };
    PHYS3.with(|x| *x.borrow_mut() = Some(p));
}

fn push_body(p: &mut Phys3, rb: RigidBody, col: Collider) -> i64 {
    let h = p.bodies.insert(rb);
    let c = p.colliders.insert_with_parent(col, h, &mut p.bodies);
    p.handles.push(h);
    p.cols.push(c);
    p.grounded.push(false);
    (p.handles.len() - 1) as i64
}

fn body_builder(x: f64, y: f64, z: f64, dynamic: i64) -> RigidBodyBuilder {
    let b = if dynamic != 0 { RigidBodyBuilder::dynamic() } else { RigidBodyBuilder::fixed() };
    b.translation(vector![x as Real, y as Real, z as Real])
}

/// Add a box (half-extents hx,hy,hz) at (x,y,z). `dynamic` 1=moving, 0=static.
#[no_mangle]
pub extern "C" fn aurora_phys3d_add_box(
    x: f64, y: f64, z: f64, hx: f64, hy: f64, hz: f64, dynamic: i64,
) -> i64 {
    PHYS3.with(|p| {
        let mut p = p.borrow_mut();
        let Some(p) = p.as_mut() else { return -1 };
        let rb = body_builder(x, y, z, dynamic).build();
        let col = ColliderBuilder::cuboid(hx as Real, hy as Real, hz as Real).build();
        push_body(p, rb, col)
    })
}

/// Add a box rotated by the axis-angle vector (rx,ry,rz) - e.g. a tilt about X gives a
/// ramp/slope. Pass the same angles to `r3d_draw`'s euler to make the visual match.
#[no_mangle]
pub extern "C" fn aurora_phys3d_add_box_rot(
    x: f64, y: f64, z: f64, hx: f64, hy: f64, hz: f64, rx: f64, ry: f64, rz: f64, dynamic: i64,
) -> i64 {
    PHYS3.with(|p| {
        let mut p = p.borrow_mut();
        let Some(p) = p.as_mut() else { return -1 };
        let b = if dynamic != 0 { RigidBodyBuilder::dynamic() } else { RigidBodyBuilder::fixed() };
        let rb = b
            .translation(vector![x as Real, y as Real, z as Real])
            .rotation(vector![rx as Real, ry as Real, rz as Real])
            .build();
        let col = ColliderBuilder::cuboid(hx as Real, hy as Real, hz as Real).build();
        push_body(p, rb, col)
    })
}

/// Add a sphere of `radius` at (x,y,z).
#[no_mangle]
pub extern "C" fn aurora_phys3d_add_sphere(x: f64, y: f64, z: f64, radius: f64, dynamic: i64) -> i64 {
    PHYS3.with(|p| {
        let mut p = p.borrow_mut();
        let Some(p) = p.as_mut() else { return -1 };
        let rb = body_builder(x, y, z, dynamic).build();
        let col = ColliderBuilder::ball(radius as Real).build();
        push_body(p, rb, col)
    })
}

/// Add an upright capsule (cylinder half-height `hh`, end radius `r`) at (x,y,z).
#[no_mangle]
pub extern "C" fn aurora_phys3d_add_capsule(x: f64, y: f64, z: f64, hh: f64, r: f64, dynamic: i64) -> i64 {
    PHYS3.with(|p| {
        let mut p = p.borrow_mut();
        let Some(p) = p.as_mut() else { return -1 };
        let rb = body_builder(x, y, z, dynamic).build();
        let col = ColliderBuilder::capsule_y(hh as Real, r as Real).build();
        push_body(p, rb, col)
    })
}

/// Add a kinematic capsule character controller at (x,y,z). Move it with
/// `phys3d_move_character`, which slides along walls and reports grounding.
#[no_mangle]
pub extern "C" fn aurora_phys3d_add_character(x: f64, y: f64, z: f64, hh: f64, r: f64) -> i64 {
    PHYS3.with(|p| {
        let mut p = p.borrow_mut();
        let Some(p) = p.as_mut() else { return -1 };
        let rb = RigidBodyBuilder::kinematic_position_based()
            .translation(vector![x as Real, y as Real, z as Real])
            .build();
        // Characters are in group 2; the move query (below) only collides with group 1
        // (world), so characters never stand on / trap each other - they pass through.
        let col = ColliderBuilder::capsule_y(hh as Real, r as Real)
            .collision_groups(InteractionGroups::new(Group::GROUP_2, Group::ALL))
            .build();
        push_body(p, rb, col)
    })
}

/// Add a static triangle-mesh collider from `vcount*3` vertex floats and
/// `icount` triangle indices. For arbitrary level collision geometry.
#[no_mangle]
pub extern "C" fn aurora_phys3d_add_trimesh(
    verts: *const f64, vcount: i64, indices: *const i64, icount: i64,
) -> i64 {
    if verts.is_null() || indices.is_null() || vcount <= 0 || icount < 3 {
        return -1;
    }
    let vs = unsafe { std::slice::from_raw_parts(verts, (vcount * 3) as usize) };
    let is = unsafe { std::slice::from_raw_parts(indices, icount as usize) };
    let points: Vec<Point<Real>> = (0..vcount as usize)
        .map(|i| point![vs[i * 3] as Real, vs[i * 3 + 1] as Real, vs[i * 3 + 2] as Real])
        .collect();
    let tris: Vec<[u32; 3]> = is
        .chunks_exact(3)
        .map(|c| [c[0] as u32, c[1] as u32, c[2] as u32])
        .collect();
    PHYS3.with(|p| {
        let mut p = p.borrow_mut();
        let Some(p) = p.as_mut() else { return -1 };
        let rb = RigidBodyBuilder::fixed().build();
        let col = ColliderBuilder::trimesh(points, tris).build();
        push_body(p, rb, col)
    })
}

/// Advance the simulation by `dt` seconds (also flushes kinematic moves).
#[no_mangle]
pub extern "C" fn aurora_phys3d_step(dt: f64) {
    PHYS3.with(|p| {
        let mut p = p.borrow_mut();
        let Some(p) = p.as_mut() else { return };
        p.params.dt = dt as Real;
        let g = p.gravity;
        p.pipeline.step(
            &g, &p.params, &mut p.islands, &mut p.broad, &mut p.narrow, &mut p.bodies,
            &mut p.colliders, &mut p.impulse, &mut p.multibody, &mut p.ccd, Some(&mut p.query), &(), &(),
        );
        p.query.update(&p.colliders);
    });
}

fn axis(h: i64, i: usize) -> f64 {
    PHYS3.with(|p| {
        let p = p.borrow();
        match p.as_ref().and_then(|p| p.handles.get(h.max(0) as usize).and_then(|&hd| p.bodies.get(hd))) {
            Some(b) => b.translation()[i] as f64,
            None => 0.0,
        }
    })
}
#[no_mangle]
pub extern "C" fn aurora_phys3d_x(h: i64) -> f64 { axis(h, 0) }
#[no_mangle]
pub extern "C" fn aurora_phys3d_y(h: i64) -> f64 { axis(h, 1) }
#[no_mangle]
pub extern "C" fn aurora_phys3d_z(h: i64) -> f64 { axis(h, 2) }

fn vaxis(h: i64, i: usize) -> f64 {
    PHYS3.with(|p| {
        let p = p.borrow();
        match p.as_ref().and_then(|p| p.handles.get(h.max(0) as usize).and_then(|&hd| p.bodies.get(hd))) {
            Some(b) => b.linvel()[i] as f64,
            None => 0.0,
        }
    })
}
#[no_mangle]
pub extern "C" fn aurora_phys3d_vel_x(h: i64) -> f64 { vaxis(h, 0) }
#[no_mangle]
pub extern "C" fn aurora_phys3d_vel_y(h: i64) -> f64 { vaxis(h, 1) }
#[no_mangle]
pub extern "C" fn aurora_phys3d_vel_z(h: i64) -> f64 { vaxis(h, 2) }

#[no_mangle]
pub extern "C" fn aurora_phys3d_set_vel(h: i64, vx: f64, vy: f64, vz: f64) {
    PHYS3.with(|p| {
        let mut p = p.borrow_mut();
        let Some(p) = p.as_mut() else { return };
        if let Some(b) = p.handles.get(h.max(0) as usize).copied().and_then(|hd| p.bodies.get_mut(hd)) {
            b.set_linvel(vector![vx as Real, vy as Real, vz as Real], true);
        }
    });
}

#[no_mangle]
pub extern "C" fn aurora_phys3d_set_pos(h: i64, x: f64, y: f64, z: f64) {
    PHYS3.with(|p| {
        let mut p = p.borrow_mut();
        let Some(p) = p.as_mut() else { return };
        let idx = h.max(0) as usize;
        if let Some(b) = p.handles.get(idx).copied().and_then(|hd| p.bodies.get_mut(hd)) {
            let t = vector![x as Real, y as Real, z as Real];
            if b.is_kinematic() {
                b.set_next_kinematic_translation(t.into());
                b.set_translation(t, true);
            } else {
                b.set_translation(t, true);
            }
        }
    });
}

/// Apply an instantaneous impulse (jump/knockback) to a dynamic body.
#[no_mangle]
pub extern "C" fn aurora_phys3d_apply_impulse(h: i64, ix: f64, iy: f64, iz: f64) {
    PHYS3.with(|p| {
        let mut p = p.borrow_mut();
        let Some(p) = p.as_mut() else { return };
        if let Some(b) = p.handles.get(h.max(0) as usize).copied().and_then(|hd| p.bodies.get_mut(hd)) {
            b.apply_impulse(vector![ix as Real, iy as Real, iz as Real], true);
        }
    });
}

/// Move a character capsule by `(dx,dy,dz)` this frame, sliding along walls.
/// Sets the body's next kinematic position; read it back after `phys3d_step`.
#[no_mangle]
pub extern "C" fn aurora_phys3d_move_character(h: i64, dx: f64, dy: f64, dz: f64, dt: f64) {
    PHYS3.with(|p| {
        let mut p = p.borrow_mut();
        let Some(p) = p.as_mut() else { return };
        let idx = h.max(0) as usize;
        let (Some(&col_h), Some(&body_h)) = (p.cols.get(idx), p.handles.get(idx)) else { return };
        let desired = vector![dx as Real, dy as Real, dz as Real];
        // Use the BODY's current translation as the shape's start position, not
        // the collider's cached pose. The collider pose only syncs during a step,
        // so if the caller just teleported the body with `phys3d_set_pos` (the
        // rollback-safe pattern: write the authoritative position in each tick),
        // the collider is still stale. The body translation reflects `set_pos`
        // immediately, so the slide starts from the right place.
        let body_t = p.bodies.get(body_h).map(|b| *b.translation()).unwrap_or(desired);
        let (new_t, grounded, hit_cols) = {
            let Some(collider) = p.colliders.get(col_h) else { return };
            let shape = collider.shape();
            let mut pos = *collider.position();
            pos.translation.vector = body_t;
            // Group 1 (world) only: a character slides on the world but not on other
            // characters, so no stacking/trapping. Raycasts (default filter) still hit
            // characters, so shooting is unaffected.
            let filter = QueryFilter::default()
                .exclude_collider(col_h)
                .groups(InteractionGroups::new(Group::GROUP_2, Group::GROUP_1));
            // Collect the colliders we ran into so we can SHOVE the dynamic ones (crates) afterwards -
            // a kinematic controller otherwise just slides off them and they never move.
            let mut hits = Vec::new();
            let mvt = p.controller.move_shape(
                dt as Real, &p.bodies, &p.colliders, &p.query, shape, &pos, desired, filter,
                |coll| hits.push(coll.handle),
            );
            (pos.translation.vector + mvt.translation, mvt.grounded, hits)
        };
        p.grounded[idx] = grounded;
        // Resolve the dynamic bodies (crates) we ran into + read their velocities, so we can do BOTH
        // directions: the character shoves the box, AND a fast-moving box shoves the character a bit
        // (a flying crate "kinda blocks you but not like a hard wall" - it carries you along).
        let mut dyn_hits = Vec::new();
        for ch in hit_cols {
            if let Some(bh) = p.colliders.get(ch).and_then(|c| c.parent()) {
                if let Some(b) = p.bodies.get(bh) {
                    if b.is_dynamic() {
                        let v = *b.linvel();
                        dyn_hits.push((bh, v));
                    }
                }
            }
        }
        // BOX -> CHARACTER: a fast crate carries the character a fraction of its horizontal speed
        // (capped, soft) rather than being a perfect wall.
        let mut carry = vector![0.0 as Real, 0.0, 0.0];
        for (_, v) in &dyn_hits {
            let vh = vector![v.x, 0.0 as Real, v.z];
            let vl = vh.norm();
            if vl > 2.0 {
                let s = vl.min(8.0); // cap how hard a flung box can shove you
                carry += vh / vl * (s * 0.5 * dt as Real);
            }
        }
        if let Some(b) = p.bodies.get_mut(body_h) {
            b.set_next_kinematic_translation((new_t + carry).into());
        }
        // CHARACTER -> BOX: shove the dynamic ones along the move direction (a firm nudge, not a launch).
        let hdir = vector![dx as Real, 0.0, dz as Real];
        let hl = hdir.norm();
        if hl > 0.001 {
            let imp = hdir / hl * 0.5 as Real;
            for (bh, _) in dyn_hits {
                if let Some(b) = p.bodies.get_mut(bh) {
                    b.apply_impulse(imp, true);
                }
            }
        }
    });
}

/// Whether a character is touching the ground (1) or airborne (0).
#[no_mangle]
pub extern "C" fn aurora_phys3d_grounded(h: i64) -> i64 {
    PHYS3.with(|p| {
        p.borrow().as_ref().and_then(|p| p.grounded.get(h.max(0) as usize)).map(|&g| g as i64).unwrap_or(0)
    })
}

/// Cast a ray from (x,y,z) along (dx,dy,dz) up to `max`; returns the distance to
/// the first hit, or -1. Run after `phys3d_step`. Good for shooting and ground
/// checks.
#[no_mangle]
pub extern "C" fn aurora_phys3d_raycast(
    x: f64, y: f64, z: f64, dx: f64, dy: f64, dz: f64, max: f64,
) -> f64 {
    PHYS3.with(|p| {
        let p = p.borrow();
        let Some(p) = p.as_ref() else { return -1.0 };
        let dir = vector![dx as Real, dy as Real, dz as Real];
        let ray = Ray::new(point![x as Real, y as Real, z as Real], dir);
        match p.query.cast_ray(&p.bodies, &p.colliders, &ray, max as Real, true, QueryFilter::default()) {
            Some((_, toi)) => toi as f64,
            None => -1.0,
        }
    })
}

fn col_index(p: &Phys3, ch: ColliderHandle) -> i64 {
    p.cols.iter().position(|&c| c == ch).map(|i| i as i64).unwrap_or(-1)
}

/// Cast a ray and record the hit: returns the hit body handle (or -1) and stores
/// the hit point + surface normal for `phys3d_hit_*`. For shooting and grapples.
#[no_mangle]
pub extern "C" fn aurora_phys3d_raycast_full(
    x: f64, y: f64, z: f64, dx: f64, dy: f64, dz: f64, max: f64,
) -> i64 {
    PHYS3.with(|p| {
        let mut p = p.borrow_mut();
        let Some(p) = p.as_mut() else { return -1 };
        let ray = Ray::new(point![x as Real, y as Real, z as Real], vector![dx as Real, dy as Real, dz as Real]);
        let hit = p.query.cast_ray_and_get_normal(
            &p.bodies, &p.colliders, &ray, max as Real, true, QueryFilter::default(),
        );
        match hit {
            Some((ch, inter)) => {
                let pt = ray.point_at(inter.time_of_impact);
                p.hit_point = [pt.x as f64, pt.y as f64, pt.z as f64];
                p.hit_normal = [inter.normal.x as f64, inter.normal.y as f64, inter.normal.z as f64];
                p.hit_body = col_index(p, ch);
                p.hit_body
            }
            None => {
                p.hit_body = -1;
                -1
            }
        }
    })
}

/// Like `raycast_full`, but excludes one character/body's own collider (by its
/// handle). Lets a body probe outward from its own centre - e.g. a wallrun side
/// cast - without immediately hitting itself. Records hit point + normal too.
#[no_mangle]
pub extern "C" fn aurora_phys3d_raycast_ex(
    exclude: i64, x: f64, y: f64, z: f64, dx: f64, dy: f64, dz: f64, max: f64,
) -> i64 {
    PHYS3.with(|p| {
        let mut p = p.borrow_mut();
        let Some(p) = p.as_mut() else { return -1 };
        let filter = match p.cols.get(exclude.max(0) as usize).copied() {
            Some(ch) => QueryFilter::default().exclude_collider(ch),
            None => QueryFilter::default(),
        };
        let ray = Ray::new(point![x as Real, y as Real, z as Real], vector![dx as Real, dy as Real, dz as Real]);
        let hit = p.query.cast_ray_and_get_normal(&p.bodies, &p.colliders, &ray, max as Real, true, filter);
        match hit {
            Some((ch, inter)) => {
                let pt = ray.point_at(inter.time_of_impact);
                p.hit_point = [pt.x as f64, pt.y as f64, pt.z as f64];
                p.hit_normal = [inter.normal.x as f64, inter.normal.y as f64, inter.normal.z as f64];
                p.hit_body = col_index(p, ch);
                p.hit_body
            }
            None => {
                p.hit_body = -1;
                -1
            }
        }
    })
}

/// Like `raycast_ex`, but only hits the WORLD (static/dynamic level geometry, group 1) and
/// IGNORES other character capsules (group 2). For MOVEMENT probes - ground checks, wall
/// detection, mantle - where standing/sliding is resolved against the world only (matching
/// `move_character`). Using the plain raycast there made a player read as "grounded" when
/// another player's capsule happened to be below them, cancelling gravity (float + infinite
/// jump). Records hit point + normal like `raycast_ex`. Shooting still uses the plain raycast
/// (which DOES hit characters).
#[no_mangle]
pub extern "C" fn aurora_phys3d_raycast_world(
    exclude: i64, x: f64, y: f64, z: f64, dx: f64, dy: f64, dz: f64, max: f64,
) -> i64 {
    PHYS3.with(|p| {
        let mut p = p.borrow_mut();
        let Some(p) = p.as_mut() else { return -1 };
        // Collide with world (group 1) only - characters (group 2) are skipped. See the group
        // reasoning in `move_character`.
        let mut filter = QueryFilter::default()
            .groups(InteractionGroups::new(Group::GROUP_1, Group::GROUP_1));
        if let Some(&ch) = p.cols.get(exclude.max(0) as usize) {
            filter = filter.exclude_collider(ch);
        }
        let ray = Ray::new(point![x as Real, y as Real, z as Real], vector![dx as Real, dy as Real, dz as Real]);
        let hit = p.query.cast_ray_and_get_normal(&p.bodies, &p.colliders, &ray, max as Real, true, filter);
        match hit {
            Some((ch, inter)) => {
                let pt = ray.point_at(inter.time_of_impact);
                p.hit_point = [pt.x as f64, pt.y as f64, pt.z as f64];
                p.hit_normal = [inter.normal.x as f64, inter.normal.y as f64, inter.normal.z as f64];
                p.hit_body = col_index(p, ch);
                p.hit_body
            }
            None => {
                p.hit_body = -1;
                -1
            }
        }
    })
}

fn hit_pt(i: usize) -> f64 {
    PHYS3.with(|p| p.borrow().as_ref().map(|p| p.hit_point[i]).unwrap_or(0.0))
}
fn hit_nrm(i: usize) -> f64 {
    PHYS3.with(|p| p.borrow().as_ref().map(|p| p.hit_normal[i]).unwrap_or(0.0))
}
#[no_mangle]
pub extern "C" fn aurora_phys3d_hit_x() -> f64 { hit_pt(0) }
#[no_mangle]
pub extern "C" fn aurora_phys3d_hit_y() -> f64 { hit_pt(1) }
#[no_mangle]
pub extern "C" fn aurora_phys3d_hit_z() -> f64 { hit_pt(2) }
#[no_mangle]
pub extern "C" fn aurora_phys3d_hit_nx() -> f64 { hit_nrm(0) }
#[no_mangle]
pub extern "C" fn aurora_phys3d_hit_ny() -> f64 { hit_nrm(1) }
#[no_mangle]
pub extern "C" fn aurora_phys3d_hit_nz() -> f64 { hit_nrm(2) }
#[no_mangle]
pub extern "C" fn aurora_phys3d_hit_body() -> i64 {
    PHYS3.with(|p| p.borrow().as_ref().map(|p| p.hit_body).unwrap_or(-1))
}

/// Sweep a sphere of `radius` from (x,y,z) along (dx,dy,dz); returns the distance
/// to the first hit, or -1. Thick projectiles, character probes.
#[no_mangle]
pub extern "C" fn aurora_phys3d_spherecast(
    x: f64, y: f64, z: f64, dx: f64, dy: f64, dz: f64, radius: f64, max: f64,
) -> f64 {
    PHYS3.with(|p| {
        let p = p.borrow();
        let Some(p) = p.as_ref() else { return -1.0 };
        let dir = vector![dx as Real, dy as Real, dz as Real];
        let len = dir.norm();
        if len < 1e-6 {
            return -1.0;
        }
        let vel = dir / len; // unit direction -> time_of_impact is distance
        let shape = Ball::new(radius as Real);
        let pos = Isometry::translation(x as Real, y as Real, z as Real);
        let opts = ShapeCastOptions::with_max_time_of_impact(max as Real);
        match p.query.cast_shape(&p.bodies, &p.colliders, &pos, &vel, &shape, opts, QueryFilter::default()) {
            Some((_, hit)) => hit.time_of_impact as f64,
            None => -1.0,
        }
    })
}

/// First body whose collider overlaps a sphere at (x,y,z); -1 if none. Triggers,
/// pickups, explosion queries.
#[no_mangle]
pub extern "C" fn aurora_phys3d_overlap_sphere(x: f64, y: f64, z: f64, radius: f64) -> i64 {
    PHYS3.with(|p| {
        let p = p.borrow();
        let Some(p) = p.as_ref() else { return -1 };
        let shape = Ball::new(radius as Real);
        let pos = Isometry::translation(x as Real, y as Real, z as Real);
        match p.query.intersection_with_shape(&p.bodies, &p.colliders, &pos, &shape, QueryFilter::default()) {
            Some(ch) => col_index(p, ch),
            None => -1,
        }
    })
}

/// Apply a continuous force (cleared each step) to a dynamic body.
#[no_mangle]
pub extern "C" fn aurora_phys3d_apply_force(h: i64, fx: f64, fy: f64, fz: f64) {
    PHYS3.with(|p| {
        let mut p = p.borrow_mut();
        let Some(p) = p.as_mut() else { return };
        if let Some(b) = p.handles.get(h.max(0) as usize).copied().and_then(|hd| p.bodies.get_mut(hd)) {
            b.add_force(vector![fx as Real, fy as Real, fz as Real], true);
        }
    });
}

/// Apply a torque to a dynamic body.
#[no_mangle]
pub extern "C" fn aurora_phys3d_apply_torque(h: i64, tx: f64, ty: f64, tz: f64) {
    PHYS3.with(|p| {
        let mut p = p.borrow_mut();
        let Some(p) = p.as_mut() else { return };
        if let Some(b) = p.handles.get(h.max(0) as usize).copied().and_then(|hd| p.bodies.get_mut(hd)) {
            b.add_torque(vector![tx as Real, ty as Real, tz as Real], true);
        }
    });
}

/// Set a body's angular velocity.
#[no_mangle]
pub extern "C" fn aurora_phys3d_set_angvel(h: i64, ax: f64, ay: f64, az: f64) {
    PHYS3.with(|p| {
        let mut p = p.borrow_mut();
        let Some(p) = p.as_mut() else { return };
        if let Some(b) = p.handles.get(h.max(0) as usize).copied().and_then(|hd| p.bodies.get_mut(hd)) {
            b.set_angvel(vector![ax as Real, ay as Real, az as Real], true);
        }
    });
}

/// Set a body's orientation from a quaternion (x,y,z,w).
#[no_mangle]
pub extern "C" fn aurora_phys3d_set_rot(h: i64, qx: f64, qy: f64, qz: f64, qw: f64) {
    PHYS3.with(|p| {
        let mut p = p.borrow_mut();
        let Some(p) = p.as_mut() else { return };
        if let Some(b) = p.handles.get(h.max(0) as usize).copied().and_then(|hd| p.bodies.get_mut(hd)) {
            let q = UnitQuaternion::from_quaternion(Quaternion::new(
                qw as Real, qx as Real, qy as Real, qz as Real,
            ));
            b.set_rotation(q, true);
        }
    });
}

fn rot_comp(h: i64, i: usize) -> f64 {
    PHYS3.with(|p| {
        let p = p.borrow();
        match p.as_ref().and_then(|p| p.handles.get(h.max(0) as usize).and_then(|&hd| p.bodies.get(hd))) {
            Some(b) => {
                let q = b.rotation();
                [q.i, q.j, q.k, q.w][i] as f64
            }
            None => [0.0, 0.0, 0.0, 1.0][i],
        }
    })
}
#[no_mangle]
pub extern "C" fn aurora_phys3d_rot_qx(h: i64) -> f64 { rot_comp(h, 0) }
#[no_mangle]
pub extern "C" fn aurora_phys3d_rot_qy(h: i64) -> f64 { rot_comp(h, 1) }
#[no_mangle]
pub extern "C" fn aurora_phys3d_rot_qz(h: i64) -> f64 { rot_comp(h, 2) }
#[no_mangle]
pub extern "C" fn aurora_phys3d_rot_qw(h: i64) -> f64 { rot_comp(h, 3) }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn raycast_full_reports_body_point_and_normal() {
        aurora_phys3d_init(0.0, -9.81, 0.0);
        let ground = aurora_phys3d_add_box(0.0, 0.0, 0.0, 5.0, 1.0, 5.0, 0); // top at y=1
        aurora_phys3d_step(0.016);
        // Ray straight down from above the box.
        let body = aurora_phys3d_raycast_full(0.0, 5.0, 0.0, 0.0, -1.0, 0.0, 20.0);
        assert_eq!(body, ground, "should hit the ground box");
        assert!((aurora_phys3d_hit_y() - 1.0).abs() < 0.05, "hit point on top face, got {}", aurora_phys3d_hit_y());
        assert!(aurora_phys3d_hit_ny() > 0.9, "normal should point up, got {}", aurora_phys3d_hit_ny());
    }

    #[test]
    fn overlap_sphere_finds_a_body() {
        aurora_phys3d_init(0.0, 0.0, 0.0);
        let b = aurora_phys3d_add_sphere(0.0, 0.0, 0.0, 1.0, 0);
        aurora_phys3d_step(0.016);
        assert_eq!(aurora_phys3d_overlap_sphere(0.5, 0.0, 0.0, 0.5), b, "overlapping sphere found");
        assert_eq!(aurora_phys3d_overlap_sphere(20.0, 20.0, 20.0, 0.5), -1, "far query finds nothing");
    }
}
