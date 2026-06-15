//! 3D physics for Aurora, backed by Rapier 3D: rigid bodies (box/sphere/capsule
//! and arbitrary static trimeshes), impulses (jumps/knockback), raycasts, and a
//! kinematic capsule character controller that slides along walls - the core of
//! a fluid 3D movement shooter.
//!
//! Bodies are referenced by an `i64` handle (insertion order). State lives in a
//! thread-local, matching the single-threaded program the runtime serves.

use std::cell::RefCell;

use rapier3d::control::{CharacterLength, KinematicCharacterController};
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
        let col = ColliderBuilder::capsule_y(hh as Real, r as Real).build();
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
        let (new_t, grounded) = {
            let Some(collider) = p.colliders.get(col_h) else { return };
            let shape = collider.shape();
            let pos = *collider.position();
            let filter = QueryFilter::default().exclude_collider(col_h);
            let mvt = p.controller.move_shape(
                dt as Real, &p.bodies, &p.colliders, &p.query, shape, &pos, desired, filter, |_| {},
            );
            (pos.translation.vector + mvt.translation, mvt.grounded)
        };
        p.grounded[idx] = grounded;
        if let Some(b) = p.bodies.get_mut(body_h) {
            b.set_next_kinematic_translation(new_t.into());
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
