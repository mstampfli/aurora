//! 3D physics for Aurora, backed by Rapier 3D: rigid bodies (box/sphere/capsule
//! and arbitrary static trimeshes), impulses (jumps/knockback), raycasts, and a
//! kinematic capsule character controller that slides along walls - the core of
//! a fluid 3D movement shooter.
//!
//! State lives in a thread-local, matching the single-threaded program the
//! runtime serves.
//!
//! # Body handles
//!
//! A body handle is a generation-tagged [`aurora_slot::Key`] packed into the
//! `i64` a program holds, NOT an index. `phys3d_remove` bumps its slot's
//! generation, so the handle is rejected by every accessor from then on, even
//! once a later `phys3d_add_*` lands in that same slot. The alternative - an
//! index into a `Vec` - has only bad endings: never remove (a world that grows
//! for as long as the process runs) or remove and let the next body inherit the
//! hole, which silently turns one actor's handle into another actor's position.
//! Rapier's own handles work the same way, so this layer matches rather than
//! fights the engine underneath it.
//!
//! Handles are therefore no longer small integers. They are `i64` and must be
//! kept in one; a program that stashes a handle in an `f32` (a netcode state
//! blob, say) loses the generation bits and gets a handle that is REJECTED
//! rather than one that silently points somewhere else.

use std::cell::RefCell;

use aurora_slot::{Key, SlotMap};
use rapier3d::control::{CharacterAutostep, CharacterLength, KinematicCharacterController};
use rapier3d::na::{DMatrix, Quaternion, UnitQuaternion};
use rapier3d::parry::query::ShapeCastOptions;
use rapier3d::prelude::*;

/// Everything one Aurora-visible body owns. Held in a [`SlotMap`], so freeing
/// one invalidates its handle instead of shuffling every other body's.
struct Body3 {
    body: RigidBodyHandle,
    /// The single collider attached to `body`. Kept so a query can be told to
    /// skip it and so removal can be checked to have taken it down.
    collider: ColliderHandle,
    /// Last result from the character controller, read by `phys3d_grounded`.
    grounded: bool,
    /// Does this character's own movement collide with OTHER characters?
    ///
    /// Off by default, and that default is deliberate: characters pass through
    /// each other so a crowd cannot stack, trap, or wedge itself in a doorway,
    /// which is what a shooter's bots want.
    ///
    /// It is exactly wrong for the mover you can push against. A game whose
    /// enemies are characters - so that they slide along the level instead of
    /// walking through it - needs the player to still meet them as bodies, or
    /// the enemies gain the world's collision and lose their own. The flag is on
    /// the mover rather than on the obstacle so both can be true at once: the
    /// player is solid and is stopped by a crowd, the crowd is not solid and
    /// does not jam itself.
    solid: bool,
    /// The capsule this character was built with, or `None` for anything that
    /// is not a character.
    ///
    /// Stored because SEPARATION has to measure an overlap, and an overlap
    /// needs both shapes. Reading it back off the collider would work and would
    /// also be a second answer to "how big is this character" living beside the
    /// one the program gave.
    capsule: Option<(Real, Real)>,
}

/// The `i64` an Aurora program holds for a body.
type BodyId = Key<Body3>;

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
    /// Aurora-visible bodies, keyed by the handle the program holds.
    registry: SlotMap<Body3>,
    /// Handles the program destroyed ON PURPOSE, since the last `init`.
    ///
    /// Holding a handle to something you removed is normal - you destroy a
    /// pickup and the code that referred to it runs once more - so reading one
    /// answers "nothing" and that is right. Holding a handle from BEFORE an
    /// `init` is never right: it is a body built into a world that was thrown
    /// away, and the slot generation makes it indistinguishable at the call
    /// site from the first case.
    ///
    /// So the two are told apart by remembering which of them the program asked
    /// for. See `body_of`.
    removed: std::collections::HashSet<i64>,
    controller: KinematicCharacterController,
    // Last raycast/shapecast hit (for `phys3d_hit_*`).
    hit_point: [f64; 3],
    hit_normal: [f64; 3],
    hit_body: i64,
    /// Colliders have changed since the query structure was last rebuilt.
    ///
    /// Rapier's `QueryPipeline` only knows about colliders it has been updated
    /// with, and that update used to happen only in `step`. A program that added
    /// bodies and then queried without stepping got truthful answers about an
    /// EMPTY world - indistinguishable at the call site from open space, since
    /// "nothing there" and "nothing indexed" both come back as -1.
    ///
    /// That cost six iterations of chasing a camera that was working perfectly:
    /// its spherecast reported a clear ray aimed straight at a wall, and every
    /// guard built on the answer was correct and inert. A query that cannot
    /// answer must rebuild or refuse, never invent.
    query_dirty: bool,
}

impl Phys3 {
    /// Bring the query structure up to date if colliders have moved or changed.
    ///
    /// Called by every spatial query rather than by the caller, so forgetting to
    /// step can no longer be mistaken for an empty world. The cost lands once per
    /// batch of mutations, which is where it belongs.
    fn sync_queries(&mut self) {
        if self.query_dirty {
            self.query.update(&self.colliders);
            self.query_dirty = false;
        }
    }
}

/// This thread's own physics world, and the shim every call site goes through.
///
/// Routed to the batch owner's while this thread is a worker running systems.
/// Without it a system that raycasts, overlaps or moves a character sees an
/// empty world and reports "nothing there" - which is a legal answer, so every
/// caller believed it.
///
/// The shim has `LocalKey`'s shape so the three dozen call sites below are
/// unchanged. They were never wrong; what they reached for was.
pub(crate) fn own_cell() -> *const () {
    PHYS3_OWN.with(|c| c as *const _ as *const ())
}

struct Phys3Slot;

impl Phys3Slot {
    fn with<R>(&self, f: impl FnOnce(&RefCell<Option<Phys3>>) -> R) -> R {
        let batch = crate::par_batch();
        if batch.is_null() {
            return PHYS3_OWN.with(f);
        }
        // SAFETY: as for the world - the owner is blocked in `thread::scope`
        // until this worker joins, so its cell is alive and untouched.
        unsafe {
            crate::with_par_cell(
                batch,
                crate::par_cell(batch, crate::CELL_PHYS3) as *const RefCell<Option<Phys3>>,
                f,
            )
        }
    }
}

const PHYS3: Phys3Slot = Phys3Slot;

thread_local! {
    static PHYS3_OWN: RefCell<Option<Phys3>> = const { RefCell::new(None) };
}

/// Create (or reset) the 3D physics world with gravity `(gx, gy, gz)`.
///
/// Handles issued by the PREVIOUS world are invalidated, not silently carried
/// over: the registry is cleared rather than replaced, which bumps every live
/// slot's generation. A fresh registry would restart generations at 1 and hand
/// the new world's first body exactly the `i64` the old world's first body had.
#[no_mangle]
pub extern "C" fn aurora_phys3d_init(gx: f64, gy: f64, gz: f64) {
    let controller = KinematicCharacterController {
        up: Vector::y_axis(),
        offset: CharacterLength::Absolute(0.02),
        slide: true,
        snap_to_ground: Some(CharacterLength::Absolute(0.3)),
        // Stairs, in metres rather than as a fraction of the character.
        //
        // The default is Relative(0.25), which for a human-sized capsule works out around 0.2 m -
        // below the riser of any staircase anyone would build, so a character walked to the foot
        // of a flight and stopped dead against the first step. It cost MARROW a whole feature:
        // its stairwells could be walked DOWN (gravity does that) and never climbed, so every
        // level below was one-way.
        //
        // Absolute, because "what can this body step onto" is a fact about the world's geometry,
        // not about how tall the body is: a kerb is a kerb for everyone.
        autostep: Some(CharacterAutostep {
            max_height: CharacterLength::Absolute(0.55),
            min_width: CharacterLength::Absolute(0.2),
            include_dynamic_bodies: false,
        }),
        ..Default::default()
    };
    PHYS3.with(|x| {
        let mut cell = x.borrow_mut();
        let mut registry = cell.take().map(|p| p.registry).unwrap_or_default();
        registry.clear();
        *cell = Some(Phys3 {
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
            registry,
            // A new world: nothing in it has been removed from it.
            removed: std::collections::HashSet::new(),
            controller,
            hit_point: [0.0; 3],
            hit_normal: [0.0; 3],
            hit_body: -1,
            // Nothing indexed yet, and nothing to index.
            query_dirty: false,
        });
    });
}

fn push_body(p: &mut Phys3, rb: RigidBody, col: Collider) -> i64 {
    let body = p.bodies.insert(rb);
    let collider = p.colliders.insert_with_parent(col, body, &mut p.bodies);
    let id = p.registry.insert(Body3 {
        body,
        collider,
        grounded: false,
        solid: false,
        capsule: None,
    });
    // Stamp the handle into the collider. A query answers with a collider, and
    // the program wants a body handle back; reading it out of `user_data` is
    // O(1), where the linear scan of every body this replaces cost O(n) on
    // every single raycast.
    if let Some(c) = p.colliders.get_mut(collider) {
        c.user_data = id.to_i64() as u128;
    }
    p.query_dirty = true;
    id.to_i64()
}

/// The body `h` names, or `None` when `h` is stale (its body was removed, and
/// possibly replaced), was never issued, or is the runtime's `-1` "no body"
/// sentinel.
///
/// Every accessor goes through this or one of the two helpers below, which is
/// what makes "a freed handle cannot read a live body" a property of the file
/// rather than of each function remembering to check.
fn body_of(p: &Phys3, h: i64) -> Option<&Body3> {
    let key = Key::from_i64(h)?;
    if let Some(b) = p.registry.get(key) {
        return Some(b);
    }
    // A handle that PARSES but names nothing is one of two very different
    // things, and answering `None` to both is how a whole feature disappeared
    // without a word.
    //
    // `init` clears the registry, which bumps every live slot's generation, so
    // every handle issued by the previous world goes stale at once. Nothing said
    // so. In Poly Souls a boss's collider was built two lines before the arena
    // called `init`, so `boss_body` named nothing for the rest of the process:
    // the player walked through the boss, the body never followed it, and
    // `shove_player` - a mechanic built on purpose so a creature pushes you out
    // of the way - never ran once. Every call took a handle, resolved it to
    // `None`, and returned successfully.
    //
    // Removing a body yourself is the other case and it is legitimate, so it
    // stays quiet: you destroy something and the code that referred to it runs
    // one more time.
    if outlived_its_world(p, h) {
        crate::fatal(format_args!(
            "phys3d: body handle {h} names nothing. It was not removed by this program, so it was issued before the last `phys3d_init` - which destroys the world and every handle into it. Build bodies AFTER the world they live in, and use `phys3d_alive` to ask whether a handle is still good; it answers rather than stops."
        ));
    }
    None
}

/// Is `h` a handle from a world that no longer exists?
///
/// Split out from the reporting so the DECISION can be tested. The report ends
/// the process, which a unit test cannot survive, and a rule nothing exercises
/// is the thing this whole change is about.
///
/// True only for a handle that parses, resolves to nothing, and was never handed
/// to `phys3d_remove` in the current world. Removing a body yourself and then
/// reading it back is ordinary and stays quiet.
fn outlived_its_world(p: &Phys3, h: i64) -> bool {
    let Some(key) = Key::from_i64(h) else {
        return false;
    };
    p.registry.get(key).is_none() && !p.removed.contains(&h)
}

/// The Rapier rigid body `h` names. Copied out so the caller can then borrow
/// `p.bodies` mutably.
fn rb_of(p: &Phys3, h: i64) -> Option<RigidBodyHandle> {
    body_of(p, h).map(|b| b.body)
}

/// The collider `h`'s body owns. `None` for a stale or negative handle, which
/// is what "I have no body to skip" has to mean for a query filter.
fn col_of(p: &Phys3, h: i64) -> Option<ColliderHandle> {
    body_of(p, h).map(|b| b.collider)
}

fn body_builder(x: f64, y: f64, z: f64, dynamic: i64) -> RigidBodyBuilder {
    let b = if dynamic != 0 {
        RigidBodyBuilder::dynamic()
    } else {
        RigidBodyBuilder::fixed()
    };
    b.translation(vector![x as Real, y as Real, z as Real])
}

/// Draw every physics collider as a debug wireframe (box/sphere/capsule) in
/// its current world pose, via the r3d debug-line path so it appears in
/// captures. For headless HITBOX visual audits: the physics world and the
/// rendered world can be checked to agree. `(r,g,b)` is the line color.
#[no_mangle]
pub extern "C" fn aurora_phys3d_debug_draw(r: f64, g: f64, b: f64) {
    PHYS3.with(|p| {
        let p = p.borrow();
        let Some(p) = p.as_ref() else { return };
        let (rf, gf, bf) = (r as f32, g as f32, b as f32);
        let line = |a: [f32; 3], b: [f32; 3]| {
            aurora_window::imm_r3d_debug_line(a[0], a[1], a[2], b[0], b[1], b[2], rf, gf, bf);
        };
        for (_, col) in p.colliders.iter() {
            let iso = col.position();
            let t = iso.translation.vector;
            let rot = iso.rotation;
            // Transform a shape-local point to world.
            let w = |lx: f32, ly: f32, lz: f32| -> [f32; 3] {
                let pt = rot * point![lx, ly, lz];
                [(pt.x + t.x), (pt.y + t.y), (pt.z + t.z)]
            };
            let shape = col.shape();
            if let Some(cb) = shape.as_cuboid() {
                let e = cb.half_extents;
                let (hx, hy, hz) = (e.x, e.y, e.z);
                // 8 corners, 12 edges.
                let c = [
                    w(-hx, -hy, -hz),
                    w(hx, -hy, -hz),
                    w(hx, hy, -hz),
                    w(-hx, hy, -hz),
                    w(-hx, -hy, hz),
                    w(hx, -hy, hz),
                    w(hx, hy, hz),
                    w(-hx, hy, hz),
                ];
                let edges = [
                    (0, 1),
                    (1, 2),
                    (2, 3),
                    (3, 0),
                    (4, 5),
                    (5, 6),
                    (6, 7),
                    (7, 4),
                    (0, 4),
                    (1, 5),
                    (2, 6),
                    (3, 7),
                ];
                for (i, j) in edges {
                    line(c[i], c[j]);
                }
            } else if let Some(ball) = shape.as_ball() {
                debug_rings(&w, ball.radius, 0.0, &line);
            } else if let Some(cap) = shape.as_capsule() {
                let rad = cap.radius;
                let half = (cap.segment.b - cap.segment.a).norm() * 0.5;
                // Rings at both cap centers + connecting verticals.
                debug_rings(&w, rad, half, &line);
                debug_rings(&w, rad, -half, &line);
                let n = 8;
                for k in 0..n {
                    let ang = k as f32 / n as f32 * std::f32::consts::TAU;
                    let (cx, cz) = (rad * ang.cos(), rad * ang.sin());
                    line(w(cx, -half, cz), w(cx, half, cz));
                }
            }
        }
    });
}

/// Draw a horizontal ring of `radius` at local height `y` (a wireframe circle
/// in the XZ plane), transformed to world by `w`.
fn debug_rings(
    w: &impl Fn(f32, f32, f32) -> [f32; 3],
    radius: f32,
    y: f32,
    line: &impl Fn([f32; 3], [f32; 3]),
) {
    let n = 16;
    let mut prev = w(radius, y, 0.0);
    for k in 1..=n {
        let ang = k as f32 / n as f32 * std::f32::consts::TAU;
        let cur = w(radius * ang.cos(), y, radius * ang.sin());
        line(prev, cur);
        prev = cur;
    }
    // A vertical ring too, so a sphere reads as a sphere.
    let mut pv = w(radius, y, 0.0);
    for k in 1..=n {
        let ang = k as f32 / n as f32 * std::f32::consts::TAU;
        let cur = w(radius * ang.cos(), y + radius * ang.sin(), 0.0);
        line(pv, cur);
        pv = cur;
    }
}

/// Add a box (half-extents hx,hy,hz) at (x,y,z). `dynamic` 1=moving, 0=static.
/// The world, or a clear death if there is not one.
///
/// Creating a body before `phys3d_init` used to answer -1 and carry on. -1 is
/// also the runtime's "no body" sentinel, and every function that takes a handle
/// treats it as nothing to do - so the body was never built, nothing referred to
/// it, and no call anywhere returned an error.
///
/// It cost Poly Souls its boss collider. `frame::open` asked for the creature's
/// body before the arena had stood the world up, so the handle was -1 for the
/// life of the process: the player walked through the boss, the body never
/// followed it, and `shove_player` - a mechanic built on purpose so a creature
/// pushes you out of the way rather than being stopped by you - never ran once.
/// Every fight script staged its own world first and so was unaffected, and a
/// boss with no collider fights exactly as well as one with, so nothing in a
/// large suite could see it.
///
/// Queries are left alone deliberately: asking an empty world what is at a point
/// and being told "nothing" is a fair answer. Being handed a body that does not
/// exist is not.
fn world_for<'a>(p: &'a mut Option<Phys3>, what: &str) -> &'a mut Phys3 {
    match p.as_mut() {
        Some(p) => p,
        None => crate::fatal(format_args!(
            "phys3d: {what} before `phys3d_init`. There is no world to put it in, so there is no body, and the handle would be -1 - the same value every accessor reads as \"nothing to do\". Call `phys3d_init` first."
        )),
    }
}

#[no_mangle]
pub extern "C" fn aurora_phys3d_add_box(
    x: f64,
    y: f64,
    z: f64,
    hx: f64,
    hy: f64,
    hz: f64,
    dynamic: i64,
) -> i64 {
    PHYS3.with(|p| {
        let mut p = p.borrow_mut();
        let p = world_for(&mut p, "add_box");
        let rb = body_builder(x, y, z, dynamic).build();
        let col = ColliderBuilder::cuboid(hx as Real, hy as Real, hz as Real).build();
        push_body(p, rb, col)
    })
}

/// Add a box rotated by the axis-angle vector (rx,ry,rz) - e.g. a tilt about X gives a
/// ramp/slope. Pass the same angles to `r3d_draw`'s euler to make the visual match.
#[no_mangle]
pub extern "C" fn aurora_phys3d_add_box_rot(
    x: f64,
    y: f64,
    z: f64,
    hx: f64,
    hy: f64,
    hz: f64,
    rx: f64,
    ry: f64,
    rz: f64,
    dynamic: i64,
) -> i64 {
    PHYS3.with(|p| {
        let mut p = p.borrow_mut();
        let p = world_for(&mut p, "add_box_rot");
        let b = if dynamic != 0 {
            RigidBodyBuilder::dynamic()
        } else {
            RigidBodyBuilder::fixed()
        };
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
pub extern "C" fn aurora_phys3d_add_sphere(
    x: f64,
    y: f64,
    z: f64,
    radius: f64,
    dynamic: i64,
) -> i64 {
    PHYS3.with(|p| {
        let mut p = p.borrow_mut();
        let p = world_for(&mut p, "add_sphere");
        let rb = body_builder(x, y, z, dynamic).build();
        let col = ColliderBuilder::ball(radius as Real).build();
        push_body(p, rb, col)
    })
}

/// Add an upright capsule (cylinder half-height `hh`, end radius `r`) at (x,y,z).
#[no_mangle]
pub extern "C" fn aurora_phys3d_add_capsule(
    x: f64,
    y: f64,
    z: f64,
    hh: f64,
    r: f64,
    dynamic: i64,
) -> i64 {
    PHYS3.with(|p| {
        let mut p = p.borrow_mut();
        let p = world_for(&mut p, "add_capsule");
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
        let p = world_for(&mut p, "add_character");
        let rb = RigidBodyBuilder::kinematic_position_based()
            .translation(vector![x as Real, y as Real, z as Real])
            .build();
        // Group 1 is the world, group 2 is characters that stop others, group 3
        // is characters they walk through. A character BLOCKS by default and is
        // not itself blocked, which is the crowd behaviour: bots do not wedge in
        // a doorway, and the player still bumps into them.
        //
        // The two halves are separate flags because they are separate questions
        // - see `phys3d_character_solid` (am I stopped by others?) and
        // `phys3d_character_blocking` (do others stop at me?).
        let col = ColliderBuilder::capsule_y(hh as Real, r as Real)
            .collision_groups(InteractionGroups::new(Group::GROUP_2, Group::ALL))
            .build();
        let id = push_body(p, rb, col);
        if let Some(k) = Key::from_i64(id) {
            if let Some(b) = p.registry.get_mut(k) {
                b.capsule = Some((hh as Real, r as Real));
            }
        }
        id
    })
}

/// NO TWO KINEMATIC CHARACTERS MAY OCCUPY THE SAME SPACE.
///
/// Run at the end of every step, so it is a property of the world rather than
/// something a game has to remember to ask for. That distinction is the whole
/// point: a soulslike had one hand-written call that pushed the player out of
/// one creature - the one they had LOCKED - and it went unnoticed for the life
/// of the project that it pushed against a capsule which was never moved to
/// where its creature was. A rule enforced by a call site is off whenever
/// nobody calls it.
///
/// WHO MOVES is the `solid` flag, unchanged in meaning: solid asks "am I
/// stopped by other characters", so a solid character is one that respects
/// them and is therefore the one that gives way. A non-solid character passes
/// through by design and is left where it is. If both are solid the correction
/// is split, so neither is privileged. If neither is, nothing happens - a crowd
/// of ghosts is allowed to overlap, which is what that flag is for.
///
/// A character is only pushed out of a BLOCKING one (group 2). A ghost (group
/// 3) is something you walk through, so it cannot displace anybody.
///
/// Horizontal only. These are upright capsules on a floor; correcting along Y
/// would lift a character off the ground or push it through one, and standing
/// on someone's head is not a thing this resolves.
///
/// Answers how many pairs it separated, so a test can assert it did something.
#[no_mangle]
pub extern "C" fn aurora_phys3d_separate_characters() -> i64 {
    PHYS3.with(|p| {
        let mut p = p.borrow_mut();
        let Some(p) = p.as_mut() else { return 0 };
        // Snapshot first: the correction below writes positions, and reading
        // them while writing would make the result depend on iteration order.
        let mut chars: Vec<(Key<Body3>, Vector<Real>, Real, Real, bool, bool)> = Vec::new();
        for (k, b) in p.registry.iter() {
            let Some((hh, r)) = b.capsule else { continue };
            let Some(rb) = p.bodies.get(b.body) else { continue };
            let blocking = p
                .colliders
                .get(b.collider)
                .map(|c| c.collision_groups().memberships.contains(Group::GROUP_2))
                .unwrap_or(false);
            chars.push((k, *rb.translation(), hh, r, b.solid, blocking));
        }
        let mut moved: Vec<(Key<Body3>, Vector<Real>)> = Vec::new();
        let mut pairs = 0i64;
        for i in 0..chars.len() {
            for j in (i + 1)..chars.len() {
                let (ka, pa, hha, ra, sa, ba) = chars[i];
                let (kb, pb, hhb, rb_, sb, bb) = chars[j];
                // Neither gives way: they are meant to pass through each other.
                if !sa && !sb {
                    continue;
                }
                let want = ra + rb_;
                let dx = pa.x - pb.x;
                let dz = pa.z - pb.z;
                let d2 = dx * dx + dz * dz;
                if d2 >= want * want {
                    continue;
                }
                // And they have to be at the same HEIGHT to be in each other.
                let (top_a, bot_a) = (pa.y + hha + ra, pa.y - hha - ra);
                let (top_b, bot_b) = (pb.y + hhb + rb_, pb.y - hhb - rb_);
                if top_a <= bot_b || top_b <= bot_a {
                    continue;
                }
                let d = d2.sqrt();
                // Dead centre: no separating direction exists. Pick one rather
                // than dividing by zero, and pick it deterministically so a
                // stack does not jitter between two answers on consecutive
                // frames.
                let (ux, uz) = if d > 1.0e-4 {
                    (dx / d, dz / d)
                } else {
                    (1.0, 0.0)
                };
                let push = want - d;
                // Only a BLOCKING character can displace somebody.
                let a_gives = sa && bb;
                let b_gives = sb && ba;
                let (fa, fb) = match (a_gives, b_gives) {
                    (true, true) => (0.5, 0.5),
                    (true, false) => (1.0, 0.0),
                    (false, true) => (0.0, 1.0),
                    (false, false) => continue,
                };
                if fa > 0.0 {
                    let mut t = pa;
                    t.x += ux * push * fa;
                    t.z += uz * push * fa;
                    moved.push((ka, t));
                }
                if fb > 0.0 {
                    let mut t = pb;
                    t.x -= ux * push * fb;
                    t.z -= uz * push * fb;
                    moved.push((kb, t));
                }
                pairs += 1;
            }
        }
        for (k, t) in moved {
            let Some(b) = p.registry.get(k) else { continue };
            let h = b.body;
            if let Some(rb) = p.bodies.get_mut(h) {
                rb.set_next_kinematic_translation(t);
                rb.set_translation(t, true);
            }
        }
        // The query structures have to see the new positions, or the next
        // overlap test answers about where these bodies used to be - the exact
        // class of bug that made a teleport invisible to the camera.
        p.query_dirty = true;
        pairs
    })
}

/// How many kinematic CHARACTERS the world holds.
///
/// A character is a body built by `phys3d_add_character`, and this counts the
/// ones separation can see. It exists because "the separation found no
/// overlapping pair" and "the separation does not think these are characters"
/// look identical from a game, and the second is the one that has happened.
#[no_mangle]
pub extern "C" fn aurora_phys3d_character_count() -> i64 {
    PHYS3.with(|p| {
        let p = p.borrow();
        let Some(p) = p.as_ref() else { return 0 };
        // Exactly what `separate_characters` collects, including the live-body
        // check - otherwise this answers 2 while the separation sees 1, and the
        // two numbers disagreeing is the thing it exists to rule out.
        p.registry
            .iter()
            .filter(|(_, b)| b.capsule.is_some() && p.bodies.get(b.body).is_some())
            .count() as i64
    })
}

/// Characters that other characters pass straight through.
///
/// Group 1 is the world and group 2 is characters that block; this is the third
/// state, and it is what "not solid" has to mean if the word is to mean one
/// thing. A ghost is still in a group, so raycasts and overlaps - which use the
/// default filter - still find it. Only the character controller's move query
/// skips it.
const GHOST_GROUP: Group = Group::GROUP_3;

/// Whether THIS character is stopped by other characters when it moves.
///
/// Off by default. See `Body3::solid`. The other half of the question is
/// `phys3d_character_blocking`.
#[no_mangle]
pub extern "C" fn aurora_phys3d_character_solid(h: i64, on: i64) {
    PHYS3.with(|p| {
        let mut p = p.borrow_mut();
        let Some(p) = p.as_mut() else { return };
        let Some(k) = Key::from_i64(h) else { return };
        if let Some(b) = p.registry.get_mut(k) {
            b.solid = on != 0;
        }
    })
}

/// Whether OTHER characters are stopped by this one. On by default.
///
/// The companion to `phys3d_character_solid`, and a separate flag because it is
/// a separate question. "Am I stopped by others" is about the mover's own query;
/// "do others stop at me" is about this collider's membership, and no amount of
/// setting the first can express the second.
///
/// Collapsing the two is a real cost either way round. With only the mover's
/// flag, a body can never be made transparent: a game whose creatures block the
/// player - so a boss stops at you instead of walking through and swallowing you
/// - cannot then let you walk over a corpse, because the corpse goes on blocking
/// whatever anyone sets. Measured downstream: three dead soldiers sealed a
/// courtyard and the boss behind them could not be reached, 20000 ticks, nobody
/// landed a blow. Making the flag symmetric instead just moves the loss - a
/// crowd of non-solid bots would stop stopping the player, which is the whole
/// reason the mover's flag exists.
///
/// So: two bits, because there are two facts. A corpse is `blocking(0)` and is
/// walked through while still being findable by a raycast or an overlap - it
/// moves to a group the character controller does not ask about, not out of the
/// world.
#[no_mangle]
pub extern "C" fn aurora_phys3d_character_blocking(h: i64, on: i64) {
    PHYS3.with(|p| {
        let mut p = p.borrow_mut();
        let Some(p) = p.as_mut() else { return };
        let Some(col_h) = col_of(p, h) else { return };
        if let Some(c) = p.colliders.get_mut(col_h) {
            let memberships = if on != 0 { Group::GROUP_2 } else { GHOST_GROUP };
            c.set_collision_groups(InteractionGroups::new(memberships, Group::ALL));
        }
    })
}

/// Add a static triangle-mesh collider from `vcount*3` vertex floats and
/// `icount` triangle indices. For arbitrary level collision geometry.
///
/// # Safety
/// `verts` must point to `vcount * 3` initialized `f64`s and `indices` to
/// `icount` initialized `i64`s. a null `verts` or `indices` is rejected
/// rather than dereferenced.
#[no_mangle]
pub unsafe extern "C" fn aurora_phys3d_add_trimesh(
    verts: *const f64,
    vcount: i64,
    indices: *const i64,
    icount: i64,
) -> i64 {
    if verts.is_null() || indices.is_null() || vcount <= 0 || icount < 3 {
        return -1;
    }
    let vs = unsafe { std::slice::from_raw_parts(verts, (vcount * 3) as usize) };
    let is = unsafe { std::slice::from_raw_parts(indices, icount as usize) };
    let points: Vec<Point<Real>> = (0..vcount as usize)
        .map(|i| {
            point![
                vs[i * 3] as Real,
                vs[i * 3 + 1] as Real,
                vs[i * 3 + 2] as Real
            ]
        })
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

/// Add a static collider shaped like a LOADED MODEL's own mesh, placed at
/// `(x, y, z)`, turned `yaw` radians about Y and scaled by `(sx, sy, sz)`.
/// Returns the body handle, or -1 if the model has no mesh or physics is not up.
///
/// The point is that the collider IS the art. A box typed beside an `r3d_draw`
/// call is a second description of the same object, and the two drift the moment
/// an asset changes - which reads to a player as the world being broken rather
/// than as a wrong constant. Here there is one description.
///
/// Concave by construction: a triangle mesh, not a hull, so a table is solid
/// where the table is and open underneath. Static only, which is all a concave
/// shape can be - Rapier cannot use one for a moving body.
///
/// It joins `GROUP_1`, the world group, so the movement and ground probes that
/// deliberately ignore players and enemies still see it.
///
/// The transform is baked into the points rather than left on the collider, so a
/// non-uniform scale works: an isometry cannot express one.
#[no_mangle]
pub extern "C" fn aurora_phys3d_add_model_collider(
    model: i64,
    x: f64,
    y: f64,
    z: f64,
    yaw: f64,
    sx: f64,
    sy: f64,
    sz: f64,
) -> i64 {
    let Some((pos, idx)) = aurora_window::imm_r3d_model_mesh(model) else {
        return -1;
    };
    let (sn, cs) = (yaw as Real).sin_cos();
    let points: Vec<Point<Real>> = pos
        .chunks_exact(3)
        .map(|p| {
            let px = p[0] as Real * sx as Real;
            let py = p[1] as Real * sy as Real;
            let pz = p[2] as Real * sz as Real;
            point![
                px * cs + pz * sn + x as Real,
                py + y as Real,
                pz * cs - px * sn + z as Real
            ]
        })
        .collect();
    let tris: Vec<[u32; 3]> = idx.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect();
    if points.is_empty() || tris.is_empty() {
        return -1;
    }
    PHYS3.with(|p| {
        let mut p = p.borrow_mut();
        let p = world_for(&mut p, "add_model_collider");
        let rb = RigidBodyBuilder::fixed().build();
        let col = ColliderBuilder::trimesh(points, tris)
            .collision_groups(InteractionGroups::new(Group::GROUP_1, Group::ALL))
            .build();
        push_body(p, rb, col)
    })
}

/// Add the heightmap terrain as a static collider and return its body handle
/// (or -1 if the physics world does not exist yet).
///
/// The shape is Rapier's own heightfield, whose triangulation is exactly the one
/// [`aurora_render3d::Heightfield::height_at`] evaluates and the one the render
/// mesh uses at full detail: what you see, what you walk on, and what a height
/// query reports are one surface.
///
/// # Collision groups
///
/// Terrain is WORLD geometry, so it goes in group 1, exactly where a box added
/// by `phys3d_add_box` sits. That is the group `phys3d_move_character` and
/// `phys3d_raycast_world` filter to, so a character walks and ground-probes on
/// terrain while character capsules (group 2) stay invisible to those probes.
/// Getting this wrong is the bug where a player reads as "grounded" because
/// another player's capsule happened to be underneath them, and floats.
pub(crate) fn add_heightfield(field: &aurora_render3d::Heightfield) -> i64 {
    // The heightfield stores `f32`, which is Rapier's `Real`, so the samples go
    // across without a conversion that could round the surface away from the one
    // `height_at` evaluates.
    let dim = field.dim() as usize;
    let extent: Real = field.extent();
    let half = extent * 0.5;
    // Rapier's heightfield is centred on its collider, indexed [row, col] with
    // the row running along local +Z and the column along local +X, and scaled
    // to `extent` on each horizontal axis (`scale.y = 1` keeps heights in
    // metres). Translating by half an extent puts sample (0,0) back on the
    // heightfield's own origin corner.
    let heights = DMatrix::from_fn(dim, dim, |row, col| field.sample(row as i64, col as i64));
    PHYS3.with(|p| {
        let mut p = p.borrow_mut();
        let Some(p) = p.as_mut() else { return -1 };
        let rb = RigidBodyBuilder::fixed()
            .translation(vector![
                field.origin_x() + half,
                0.0,
                field.origin_z() + half
            ])
            .build();
        let col = ColliderBuilder::heightfield(heights, vector![extent, 1.0, extent])
            .collision_groups(InteractionGroups::new(Group::GROUP_1, Group::ALL))
            .build();
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
            &g,
            &p.params,
            &mut p.islands,
            &mut p.broad,
            &mut p.narrow,
            &mut p.bodies,
            &mut p.colliders,
            &mut p.impulse,
            &mut p.multibody,
            &mut p.ccd,
            Some(&mut p.query),
            &(),
            &(),
        );
        p.query.update(&p.colliders);
        // Just rebuilt, so the next query has nothing to do. Leaving this set
        // would make every step cost a second rebuild on the first query after it.
        p.query_dirty = false;
    });
    // AND NO TWO CHARACTERS ARE LEFT INSIDE EACH OTHER.
    //
    // Part of the step rather than something a game calls, because the version
    // where a game calls it is the version that is off whenever somebody
    // forgets - which is exactly how a soulslike shipped a shove that pushed
    // against a capsule nobody had moved.
    //
    // After the solver, not before: the step is what moves bodies, so
    // separating first resolves last frame's overlaps and leaves this frame's.
    let _ = aurora_phys3d_separate_characters();
}

fn axis(h: i64, i: usize) -> f64 {
    PHYS3.with(|p| {
        let p = p.borrow();
        match p
            .as_ref()
            .and_then(|p| rb_of(p, h).and_then(|hd| p.bodies.get(hd)))
        {
            Some(b) => b.translation()[i] as f64,
            None => 0.0,
        }
    })
}
#[no_mangle]
pub extern "C" fn aurora_phys3d_x(h: i64) -> f64 {
    axis(h, 0)
}
#[no_mangle]
pub extern "C" fn aurora_phys3d_y(h: i64) -> f64 {
    axis(h, 1)
}
#[no_mangle]
pub extern "C" fn aurora_phys3d_z(h: i64) -> f64 {
    axis(h, 2)
}

fn vaxis(h: i64, i: usize) -> f64 {
    PHYS3.with(|p| {
        let p = p.borrow();
        match p
            .as_ref()
            .and_then(|p| rb_of(p, h).and_then(|hd| p.bodies.get(hd)))
        {
            Some(b) => b.linvel()[i] as f64,
            None => 0.0,
        }
    })
}
#[no_mangle]
pub extern "C" fn aurora_phys3d_vel_x(h: i64) -> f64 {
    vaxis(h, 0)
}
#[no_mangle]
pub extern "C" fn aurora_phys3d_vel_y(h: i64) -> f64 {
    vaxis(h, 1)
}
#[no_mangle]
pub extern "C" fn aurora_phys3d_vel_z(h: i64) -> f64 {
    vaxis(h, 2)
}

#[no_mangle]
pub extern "C" fn aurora_phys3d_set_vel(h: i64, vx: f64, vy: f64, vz: f64) {
    PHYS3.with(|p| {
        let mut p = p.borrow_mut();
        let Some(p) = p.as_mut() else { return };
        if let Some(b) = rb_of(p, h).and_then(|hd| p.bodies.get_mut(hd)) {
            b.set_linvel(vector![vx as Real, vy as Real, vz as Real], true);
        }
    });
}

/// Teleport a body, and move what queries can see along with it.
///
/// Rapier copies a body's pose down onto its colliders during `step`, so a
/// teleport alone leaves the collider - and the query tree built from colliders -
/// describing where the body USED to be. A program that moves a body every tick
/// and queries without stepping (an actor driven by game rules rather than by the
/// solver, which is most of them) then gets answers about a world one teleport
/// stale, and a body that never steps stays at its spawn point forever as far as
/// every raycast is concerned.
///
/// That is the same failure the query-dirty flag exists for: an index that
/// silently disagrees with the world, answering confidently and wrongly. So the
/// collider is moved here too and the tree marked for rebuild - a write to a
/// position is a write to everything that reads positions.
#[no_mangle]
pub extern "C" fn aurora_phys3d_set_pos(h: i64, x: f64, y: f64, z: f64) {
    PHYS3.with(|p| {
        let mut p = p.borrow_mut();
        let Some(p) = p.as_mut() else { return };
        let (Some(body_h), Some(col_h)) = (rb_of(p, h), col_of(p, h)) else {
            return;
        };
        let t = vector![x as Real, y as Real, z as Real];
        if let Some(b) = p.bodies.get_mut(body_h) {
            if b.is_kinematic() {
                b.set_next_kinematic_translation(t);
            }
            b.set_translation(t, true);
        }
        // Composed through the collider's offset from its body rather than
        // written flat, exactly as Rapier's own step does it: a collider mounted
        // off-centre must stay off-centre after a teleport, or the shape queries
        // answer about a body that has quietly re-centred itself.
        let parent = p.bodies.get(body_h).map(|b| *b.position());
        if let (Some(parent), Some(c)) = (parent, p.colliders.get_mut(col_h)) {
            let wrt = c
                .position_wrt_parent()
                .copied()
                .unwrap_or_else(Isometry::identity);
            c.set_position(parent * wrt);
        }
        p.query_dirty = true;
    });
}

/// Apply an instantaneous impulse (jump/knockback) to a dynamic body.
#[no_mangle]
pub extern "C" fn aurora_phys3d_apply_impulse(h: i64, ix: f64, iy: f64, iz: f64) {
    PHYS3.with(|p| {
        let mut p = p.borrow_mut();
        let Some(p) = p.as_mut() else { return };
        if let Some(b) = rb_of(p, h).and_then(|hd| p.bodies.get_mut(hd)) {
            b.apply_impulse(vector![ix as Real, iy as Real, iz as Real], true);
        }
    });
}

/// Destroy a body: its Rapier rigid body, the collider attached to it, and the
/// handle. Returns 1 if a body was destroyed, 0 if `h` was already freed or
/// never named one, so a double free reads as 0 rather than tearing down
/// whatever moved into that slot in between.
///
/// `h` is invalidated, not recycled: the slot's generation is bumped, so every
/// later `phys3d_x`, `phys3d_grounded`, raycast filter and so on refuses it,
/// including after a `phys3d_add_*` lands in the same slot. Without a removal
/// path, a game that respawns actors or reloads a level grows the Rapier sets
/// for as long as the process runs.
///
/// A query run between this and the next `phys3d_step` simply does not see the
/// body: the query pipeline's tree still holds the collider handle until the
/// next step rebuilds it, but Rapier's handles are generation-tagged too, so a
/// removed one resolves to nothing rather than to whatever took its place.
#[no_mangle]
pub extern "C" fn aurora_phys3d_remove(h: i64) -> i64 {
    PHYS3.with(|p| {
        let mut p = p.borrow_mut();
        let Some(p) = p.as_mut() else { return 0 };
        let Some(key) = Key::from_i64(h) else {
            return 0;
        };
        let Some(body) = p.registry.remove(key) else {
            return 0;
        };
        // Deliberate. Reading `h` after this answers "nothing", quietly, which
        // is what a program that destroyed something should get back.
        p.removed.insert(h);
        // `true` = take the attached colliders down with the body. Removing the
        // body alone leaves its collider in the set parented to a dead body -
        // an orphan that still answers raycasts and still costs broad-phase
        // work, which is most of what makes an unremoved body expensive.
        p.bodies.remove(
            body.body,
            &mut p.islands,
            &mut p.colliders,
            &mut p.impulse,
            &mut p.multibody,
            true,
        );
        // The tree still holds the dead collider's handle. Rapier's generation
        // tags make that answer nothing rather than answer wrongly, but it is
        // still work done on every query for a body that no longer exists, and a
        // world that respawns actors pays it forever.
        p.query_dirty = true;
        1
    })
}

/// Whether `h` still names a live body (1) or has been removed / was never
/// valid (0).
///
/// Position and velocity reads answer 0.0 for a dead handle, which a body
/// genuinely sitting at the origin also answers, so this is how a program tells
/// "gone" from "at the origin" without guessing.
#[no_mangle]
pub extern "C" fn aurora_phys3d_alive(h: i64) -> i64 {
    PHYS3.with(|p| {
        p.borrow().as_ref().map_or(0, |p| {
            // Deliberately NOT through `body_of`, which refuses a handle from a
            // world that no longer exists by panicking. This is the function you
            // call to find out whether that would happen, so it has to be able
            // to answer - a predicate that explodes on the case it exists to
            // report leaves no safe way to ask the question at all.
            Key::from_i64(h).is_some_and(|k| p.registry.get(k).is_some()) as i64
        })
    })
}

/// Move a character capsule by `(dx,dy,dz)` this frame, sliding along walls.
/// Sets the body's next kinematic position; read it back after `phys3d_step`.
#[no_mangle]
pub extern "C" fn aurora_phys3d_move_character(h: i64, dx: f64, dy: f64, dz: f64, dt: f64) {
    PHYS3.with(|p| {
        let mut p = p.borrow_mut();
        let Some(p) = p.as_mut() else { return };
        let (Some(col_h), Some(body_h)) = (col_of(p, h), rb_of(p, h)) else {
            return;
        };
        // The controller reads the query pipeline, so it is a spatial query and
        // has to sync like one.
        //
        // It did not, and it is the only one of them that did not - because it
        // does not LOOK like a query, it looks like movement. A character
        // therefore walked straight through every collider added since the last
        // step: build a room, put an actor in it, move the actor, and the walls
        // are not there yet. In a game loop a step always happens first and it
        // never shows; the assertion that found it built an arena and walked a
        // creature into a block in the same breath, which is exactly what a test
        // does and gameplay never does.
        //
        // `sync_queries` already documents itself as called by every spatial
        // query so that forgetting to step cannot be mistaken for an empty world.
        // That promise was three-quarters true.
        p.sync_queries();
        let desired = vector![dx as Real, dy as Real, dz as Real];
        // Use the BODY's current translation as the shape's start position, not
        // the collider's cached pose. The collider pose only syncs during a step,
        // so if the caller just teleported the body with `phys3d_set_pos` (the
        // rollback-safe pattern: write the authoritative position in each tick),
        // the collider is still stale. The body translation reflects `set_pos`
        // immediately, so the slide starts from the right place.
        let body_t = p
            .bodies
            .get(body_h)
            .map(|b| *b.translation())
            .unwrap_or(desired);
        let (new_t, grounded, hit_cols) = {
            let Some(collider) = p.colliders.get(col_h) else {
                return;
            };
            let shape = collider.shape();
            let mut pos = *collider.position();
            pos.translation.vector = body_t;
            // Group 1 (world) only: a character slides on the world but not on other
            // characters, so no stacking/trapping. Raycasts (default filter) still hit
            // characters, so shooting is unaffected.
            // Group 1 is the world. A solid mover adds group 2 so it also meets
            // other characters.
            let solid = Key::from_i64(h)
                .and_then(|k| p.registry.get(k))
                .map(|b| b.solid)
                .unwrap_or(false);
            let mask = if solid {
                Group::GROUP_1 | Group::GROUP_2
            } else {
                Group::GROUP_1
            };
            let filter = QueryFilter::default()
                .exclude_collider(col_h)
                .groups(InteractionGroups::new(Group::GROUP_2, mask));
            // Collect the colliders we ran into so we can SHOVE the dynamic ones (crates) afterwards -
            // a kinematic controller otherwise just slides off them and they never move.
            let mut hits = Vec::new();
            let mvt = p.controller.move_shape(
                dt as Real,
                &p.bodies,
                &p.colliders,
                &p.query,
                shape,
                &pos,
                desired,
                filter,
                |coll| hits.push(coll.handle),
            );
            (pos.translation.vector + mvt.translation, mvt.grounded, hits)
        };
        if let Some(k) = Key::from_i64(h) {
            if let Some(b) = p.registry.get_mut(k) {
                b.grounded = grounded;
            }
        }
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
        let mut carry = vector![0.0_f32, 0.0, 0.0];
        for (_, v) in &dyn_hits {
            let vh = vector![v.x, 0.0_f32, v.z];
            let vl = vh.norm();
            if vl > 2.0 {
                let s = vl.min(8.0); // cap how hard a flung box can shove you
                carry += vh / vl * (s * 0.5 * dt as Real);
            }
        }
        if let Some(b) = p.bodies.get_mut(body_h) {
            let target = new_t + carry;
            b.set_next_kinematic_translation(target);
            // Apply the resolved move IMMEDIATELY too, so reading the body's position right after
            // move_character (with NO phys3d_step in between) reflects it. This lets sim_step move the
            // character without stepping the whole world per-actor - the world (crates) is now advanced
            // by exactly ONE phys3d_step per tick by the caller, so dynamic bodies no longer fly N-times
            // too fast. The controller already resolved collisions into `new_t`, so a direct set is safe.
            b.set_translation(target, false);
        }
        // CHARACTER -> BOX: shove the dynamic ones along the move direction (a firm nudge, not a launch).
        let hdir = vector![dx as Real, 0.0, dz as Real];
        let hl = hdir.norm();
        if hl > 0.001 {
            let imp = hdir / hl * 0.5_f32;
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
        p.borrow()
            .as_ref()
            .and_then(|p| body_of(p, h))
            .map(|b| b.grounded as i64)
            .unwrap_or(0)
    })
}

/// Cast a ray from (x,y,z) along (dx,dy,dz) up to `max`; returns the distance to
/// the first hit, or -1. Run after `phys3d_step`. Good for shooting and ground
/// checks.
#[no_mangle]
pub extern "C" fn aurora_phys3d_raycast(
    x: f64,
    y: f64,
    z: f64,
    dx: f64,
    dy: f64,
    dz: f64,
    max: f64,
) -> f64 {
    PHYS3.with(|p| {
        let mut p = p.borrow_mut();
        let Some(p) = p.as_mut() else { return -1.0 };
        p.sync_queries();
        let dir = vector![dx as Real, dy as Real, dz as Real];
        let ray = Ray::new(point![x as Real, y as Real, z as Real], dir);
        match p.query.cast_ray(
            &p.bodies,
            &p.colliders,
            &ray,
            max as Real,
            true,
            QueryFilter::default(),
        ) {
            Some((_, toi)) => toi as f64,
            None => -1.0,
        }
    })
}

/// The body handle a query result belongs to, or -1.
///
/// [`push_body`] stamps the handle into the collider's `user_data`, so this is
/// O(1) rather than the linear scan of every body in the world it replaces.
/// The value is re-validated against the registry so that a collider Aurora did
/// not create (`user_data` 0) reads as "no body" instead of as handle 0.
fn body_handle_of(p: &Phys3, ch: ColliderHandle) -> i64 {
    let Some(col) = p.colliders.get(ch) else {
        return -1;
    };
    let raw = col.user_data as i64;
    match BodyId::from_i64(raw) {
        Some(k) if p.registry.contains(k) => raw,
        _ => -1,
    }
}

/// Cast a ray and record the hit: returns the hit body handle (or -1) and stores
/// the hit point + surface normal for `phys3d_hit_*`. For shooting and grapples.
#[no_mangle]
pub extern "C" fn aurora_phys3d_raycast_full(
    x: f64,
    y: f64,
    z: f64,
    dx: f64,
    dy: f64,
    dz: f64,
    max: f64,
) -> i64 {
    PHYS3.with(|p| {
        let mut p = p.borrow_mut();
        let Some(p) = p.as_mut() else { return -1 };
        let ray = Ray::new(
            point![x as Real, y as Real, z as Real],
            vector![dx as Real, dy as Real, dz as Real],
        );
        let hit = p.query.cast_ray_and_get_normal(
            &p.bodies,
            &p.colliders,
            &ray,
            max as Real,
            true,
            QueryFilter::default(),
        );
        match hit {
            Some((ch, inter)) => {
                let pt = ray.point_at(inter.time_of_impact);
                p.hit_point = [pt.x as f64, pt.y as f64, pt.z as f64];
                p.hit_normal = [
                    inter.normal.x as f64,
                    inter.normal.y as f64,
                    inter.normal.z as f64,
                ];
                p.hit_body = body_handle_of(p, ch);
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
///
/// A NEGATIVE handle excludes nothing, which is what "I have no body to skip"
/// has to mean: clamping it to 0 instead would silently drop body 0 from the
/// cast, and body 0 is usually the ground (or, now, the terrain).
#[no_mangle]
pub extern "C" fn aurora_phys3d_raycast_ex(
    exclude: i64,
    x: f64,
    y: f64,
    z: f64,
    dx: f64,
    dy: f64,
    dz: f64,
    max: f64,
) -> i64 {
    PHYS3.with(|p| {
        let mut p = p.borrow_mut();
        let Some(p) = p.as_mut() else { return -1 };
        let filter = match col_of(p, exclude) {
            Some(ch) => QueryFilter::default().exclude_collider(ch),
            None => QueryFilter::default(),
        };
        let ray = Ray::new(
            point![x as Real, y as Real, z as Real],
            vector![dx as Real, dy as Real, dz as Real],
        );
        let hit = p.query.cast_ray_and_get_normal(
            &p.bodies,
            &p.colliders,
            &ray,
            max as Real,
            true,
            filter,
        );
        match hit {
            Some((ch, inter)) => {
                let pt = ray.point_at(inter.time_of_impact);
                p.hit_point = [pt.x as f64, pt.y as f64, pt.z as f64];
                p.hit_normal = [
                    inter.normal.x as f64,
                    inter.normal.y as f64,
                    inter.normal.z as f64,
                ];
                p.hit_body = body_handle_of(p, ch);
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
/// (which DOES hit characters). As in `raycast_ex`, a NEGATIVE `exclude` skips nothing.
#[no_mangle]
pub extern "C" fn aurora_phys3d_raycast_world(
    exclude: i64,
    x: f64,
    y: f64,
    z: f64,
    dx: f64,
    dy: f64,
    dz: f64,
    max: f64,
) -> i64 {
    PHYS3.with(|p| {
        let mut p = p.borrow_mut();
        let Some(p) = p.as_mut() else { return -1 };
        // Collide with world (group 1) only - characters (group 2) are skipped. See the group
        // reasoning in `move_character`.
        let mut filter =
            QueryFilter::default().groups(InteractionGroups::new(Group::GROUP_1, Group::GROUP_1));
        if let Some(ch) = col_of(p, exclude) {
            filter = filter.exclude_collider(ch);
        }
        let ray = Ray::new(
            point![x as Real, y as Real, z as Real],
            vector![dx as Real, dy as Real, dz as Real],
        );
        let hit = p.query.cast_ray_and_get_normal(
            &p.bodies,
            &p.colliders,
            &ray,
            max as Real,
            true,
            filter,
        );
        match hit {
            Some((ch, inter)) => {
                let pt = ray.point_at(inter.time_of_impact);
                p.hit_point = [pt.x as f64, pt.y as f64, pt.z as f64];
                p.hit_normal = [
                    inter.normal.x as f64,
                    inter.normal.y as f64,
                    inter.normal.z as f64,
                ];
                p.hit_body = body_handle_of(p, ch);
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
pub extern "C" fn aurora_phys3d_hit_x() -> f64 {
    hit_pt(0)
}
#[no_mangle]
pub extern "C" fn aurora_phys3d_hit_y() -> f64 {
    hit_pt(1)
}
#[no_mangle]
pub extern "C" fn aurora_phys3d_hit_z() -> f64 {
    hit_pt(2)
}
#[no_mangle]
pub extern "C" fn aurora_phys3d_hit_nx() -> f64 {
    hit_nrm(0)
}
#[no_mangle]
pub extern "C" fn aurora_phys3d_hit_ny() -> f64 {
    hit_nrm(1)
}
#[no_mangle]
pub extern "C" fn aurora_phys3d_hit_nz() -> f64 {
    hit_nrm(2)
}
#[no_mangle]
pub extern "C" fn aurora_phys3d_hit_body() -> i64 {
    PHYS3.with(|p| p.borrow().as_ref().map(|p| p.hit_body).unwrap_or(-1))
}

/// Sweep a sphere of `radius` from (x,y,z) along (dx,dy,dz); returns the distance
/// to the first hit, or -1. Thick projectiles, character probes.
///
/// `ignore` is a body handle the sweep passes through, or -1 for none. A sweep
/// that starts INSIDE a body otherwise hits that body at zero distance, which is
/// the common case rather than an exotic one: a third-person camera probe begins
/// at the character's own head, and without this it reports the character and
/// the camera is pulled inside it. Every other cast in this engine can say what
/// it is not interested in; this one could not.
#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub extern "C" fn aurora_phys3d_spherecast(
    x: f64,
    y: f64,
    z: f64,
    dx: f64,
    dy: f64,
    dz: f64,
    radius: f64,
    max: f64,
    ignore: i64,
) -> f64 {
    PHYS3.with(|p| {
        let mut p = p.borrow_mut();
        let Some(p) = p.as_mut() else { return -1.0 };
        p.sync_queries();
        let dir = vector![dx as Real, dy as Real, dz as Real];
        let len = dir.norm();
        if len < 1e-6 {
            return -1.0;
        }
        let vel = dir / len; // unit direction -> time_of_impact is distance
        let shape = Ball::new(radius as Real);
        let pos = Isometry::translation(x as Real, y as Real, z as Real);
        let opts = ShapeCastOptions::with_max_time_of_impact(max as Real);
        match p.query.cast_shape(
            &p.bodies,
            &p.colliders,
            &pos,
            &vel,
            &shape,
            opts,
            match rb_of(p, ignore) {
                Some(hd) => QueryFilter::default().exclude_rigid_body(hd),
                None => QueryFilter::default(),
            },
        ) {
            Some((_, hit)) => hit.time_of_impact as f64,
            None => -1.0,
        }
    })
}

/// Like `phys3d_overlap_sphere`, but only the WORLD - never a character.
///
/// A third-person camera needs room around itself and pulls in when it does not
/// have it. Asking the unfiltered query means a creature walking behind the
/// player is a wall: the camera lunges in, the creature steps aside, the camera
/// springs back, and at sixty frames a second that reads as the view shaking
/// rather than as anything to do with the fight.
///
/// Characters are group 2 and the world is group 1, so the distinction already
/// exists in the physics - it was simply not askable. Which is the same shape as
/// every other gap this engine has had: the information was there and the
/// question was not.
#[no_mangle]
pub extern "C" fn aurora_phys3d_overlap_world(x: f64, y: f64, z: f64, radius: f64) -> i64 {
    PHYS3.with(|p| {
        let mut p = p.borrow_mut();
        let Some(p) = p.as_mut() else { return -1 };
        p.sync_queries();
        let shape = Ball::new(radius as Real);
        let pos = Isometry::translation(x as Real, y as Real, z as Real);
        let filter = QueryFilter::default()
            .groups(InteractionGroups::new(Group::ALL, Group::GROUP_1));
        match p
            .query
            .intersection_with_shape(&p.bodies, &p.colliders, &pos, &shape, filter)
        {
            Some(ch) => body_handle_of(p, ch),
            None => -1,
        }
    })
}

/// First body whose collider overlaps a sphere at (x,y,z); -1 if none. Triggers,
/// pickups, explosion queries.
#[no_mangle]
pub extern "C" fn aurora_phys3d_overlap_sphere(x: f64, y: f64, z: f64, radius: f64) -> i64 {
    PHYS3.with(|p| {
        let mut p = p.borrow_mut();
        let Some(p) = p.as_mut() else { return -1 };
        p.sync_queries();
        let shape = Ball::new(radius as Real);
        let pos = Isometry::translation(x as Real, y as Real, z as Real);
        match p.query.intersection_with_shape(
            &p.bodies,
            &p.colliders,
            &pos,
            &shape,
            QueryFilter::default(),
        ) {
            Some(ch) => body_handle_of(p, ch),
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
        if let Some(b) = rb_of(p, h).and_then(|hd| p.bodies.get_mut(hd)) {
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
        if let Some(b) = rb_of(p, h).and_then(|hd| p.bodies.get_mut(hd)) {
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
        if let Some(b) = rb_of(p, h).and_then(|hd| p.bodies.get_mut(hd)) {
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
        if let Some(b) = rb_of(p, h).and_then(|hd| p.bodies.get_mut(hd)) {
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
        match p
            .as_ref()
            .and_then(|p| rb_of(p, h).and_then(|hd| p.bodies.get(hd)))
        {
            Some(b) => {
                let q = b.rotation();
                [q.i, q.j, q.k, q.w][i] as f64
            }
            None => [0.0, 0.0, 0.0, 1.0][i],
        }
    })
}
#[no_mangle]
pub extern "C" fn aurora_phys3d_rot_qx(h: i64) -> f64 {
    rot_comp(h, 0)
}
#[no_mangle]
pub extern "C" fn aurora_phys3d_rot_qy(h: i64) -> f64 {
    rot_comp(h, 1)
}
#[no_mangle]
pub extern "C" fn aurora_phys3d_rot_qz(h: i64) -> f64 {
    rot_comp(h, 2)
}
#[no_mangle]
pub extern "C" fn aurora_phys3d_rot_qw(h: i64) -> f64 {
    rot_comp(h, 3)
}

/// What the world is holding: Rapier's own rigid-body and collider counts, the
/// number of live handles, and the number of handle slots ever allocated.
///
/// The leak tests are stated in these units because Rapier's sets are where the
/// memory actually is: a registry that plateaus while `RigidBodySet` grows would
/// still be a leak, so both are asserted rather than one standing in for the
/// other. `slot_count` is the fourth because a store that reused nothing would
/// keep the first three flat and still grow.
#[cfg(test)]
pub(crate) fn census() -> (usize, usize, usize, usize) {
    PHYS3.with(|p| {
        let p = p.borrow();
        match p.as_ref() {
            Some(p) => (
                p.bodies.len(),
                p.colliders.len(),
                p.registry.len(),
                p.registry.slot_count(),
            ),
            None => (0, 0, 0, 0),
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A handle from before `init` is LOUD; one you removed yourself is quiet.
    ///
    /// Both directions, because they are the same `None` at the bottom of
    /// `body_of` and only one of them is a bug.
    ///
    /// The quiet one first: destroying something and then reading its handle is
    /// ordinary - the pickup is gone and the code that referred to it runs once
    /// more - so it must not panic.
    #[test]
    fn a_handle_you_removed_yourself_stays_quiet() {
        aurora_phys3d_init(0.0, -9.81, 0.0);
        let ball = aurora_phys3d_add_sphere(0.0, 5.0, 0.0, 0.5, 1);
        assert_eq!(aurora_phys3d_remove(ball), 1);
        // Reads answer "nothing" rather than exploding.
        assert_eq!(aurora_phys3d_x(ball), 0.0);
        assert_eq!(aurora_phys3d_grounded(ball), 0);
        aurora_phys3d_move_character(ball, 1.0, 0.0, 0.0, 0.016);
        assert_eq!(aurora_phys3d_remove(ball), 0, "removing twice is not an error");
    }

    /// And the loud one, which is the whole point.
    ///
    /// `init` clears the registry, bumping every live slot's generation, so a
    /// handle issued by the previous world names nothing. That was silent, and
    /// it cost Poly Souls a mechanic: the boss's collider was built two lines
    /// before the arena called `init`, so for the life of the process the player
    /// walked through the boss and the shove that pushes them out never ran.
    /// Every call took the handle, resolved it to `None`, and returned happily.
    #[test]
    fn a_handle_from_before_init_is_refused_loudly() {
        aurora_phys3d_init(0.0, -9.81, 0.0);
        let doomed = aurora_phys3d_add_character(0.0, 1.0, 0.0, 0.5, 0.3);
        let removed_properly = aurora_phys3d_add_sphere(0.0, 5.0, 0.0, 0.5, 1);
        assert_eq!(aurora_phys3d_remove(removed_properly), 1);
        // Both are stale in the same way from the registry's point of view.
        // Only one of them is a bug.
        PHYS3.with(|p| {
            let p = p.borrow();
            let p = p.as_ref().unwrap();
            assert!(
                !outlived_its_world(p, removed_properly),
                "a body this program removed itself must stay quiet"
            );
            assert!(
                !outlived_its_world(p, doomed),
                "a live body must not be reported as stale"
            );
        });

        // A new world. Everything above is gone, including `doomed`.
        aurora_phys3d_init(0.0, -9.81, 0.0);
        aurora_phys3d_add_box(0.0, -0.5, 0.0, 50.0, 0.5, 50.0, 0);
        PHYS3.with(|p| {
            let p = p.borrow();
            let p = p.as_ref().unwrap();
            assert!(
                outlived_its_world(p, doomed),
                "a handle issued before `init` must be reported, not answered                  with a quiet nothing - that silence cost Poly Souls its boss                  collider for the life of the process"
            );
            // The removal list is cleared with the world, so a handle removed in
            // the OLD world is stale for the new reason now, and reported.
            assert!(outlived_its_world(p, removed_properly));
            // -1 is the runtime's "no body" sentinel and must never be reported.
            assert!(!outlived_its_world(p, -1));
        });

        // And the supported way to ask never stops the program.
        assert_eq!(aurora_phys3d_alive(doomed), 0);
        assert_eq!(aurora_phys3d_alive(-1), 0);
    }

    /// A character delivers the distance it was ASKED for, at ANY timestep.
    ///
    /// The same 1.12 m of forward travel across an empty floor, split into
    /// finer and finer steps. The total request is identical every time; only
    /// the number of calls changes. A controller that answers a different
    /// distance at 2400 Hz than at 60 Hz is not simulating movement, it is
    /// leaking a fixed cost per call.
    ///
    /// Found from the game end. Root motion feeds the frame the distance a clip
    /// authored, and `A_Attack_LightCombo01A_RootMotion_Sword` authors exactly
    /// 1.12 m. The frame was measured feeding the full 1.12 - vector sum
    /// (0.0, 1.1200000047) - and the capsule measured arriving at 0.7303. So the
    /// animation, the retarget, the root-delta accumulation and the game's own
    /// arithmetic are all exonerated. Headless frames take about 0.4 ms, so the
    /// game was calling this two thousand times for that 1.12 m where a 60 Hz
    /// build calls it forty-eight times.
    ///
    /// It matters far beyond root motion: dividing a walk into smaller steps
    /// must not slow the walk down. Every velocity in the game is spent through
    /// this call, so a frame-rate-dependent loss means the character is slower
    /// on a fast machine - the exact opposite of what a player expects, and
    /// invisible until something measures travel against a target.
    #[test]
    fn a_character_arrives_at_any_timestep() {
        // Total travel, and the timesteps to spend it in. 1/60 down to the
        // ~2400 fps a headless capture actually runs at.
        let want = 1.12_f64;
        let rates = [60.0_f64, 120.0, 240.0, 600.0, 2400.0];
        let mut worst = (0.0_f64, 0.0_f64);
        let mut report = String::new();

        for rate in rates {
            let dt = 1.0 / rate;
            // 0.8 s of clip, however many steps that is at this rate.
            let steps = (0.8 * rate).round() as i64;
            let per = want / (steps as f64);

            aurora_phys3d_init(0.0, -22.0, 0.0);
            // A wide flat floor with its top at y = 0, and nothing else at all.
            aurora_phys3d_add_box(0.0, -1.0, 0.0, 100.0, 1.0, 100.0, 0);
            let half = 0.55_f64;
            let radius = 0.35_f64;
            let actor = aurora_phys3d_add_character(0.0, half + radius, 0.0, half, radius);
            // Settle onto the floor first, so the walk is not paying for a fall.
            for _ in 0..(rate as i64 / 2) {
                aurora_phys3d_move_character(actor, 0.0, -2.0 * dt, 0.0, dt);
                aurora_phys3d_step(dt);
            }
            let z0 = aurora_phys3d_z(actor);
            for _ in 0..steps {
                // The game's own ground stick: held down, not zeroed.
                aurora_phys3d_move_character(actor, 0.0, -2.0 * dt, per, dt);
                aurora_phys3d_step(dt);
            }
            let got = aurora_phys3d_z(actor) - z0;

            report.push_str(&format!(
                "
  {rate:>6.0} Hz, {steps:>4} steps of {per:.6} m -> travelled {got:.4} m ({:.0}%)",
                got / want * 100.0
            ));
            if worst.1 == 0.0 || got < worst.1 {
                worst = (rate, got);
            }
        }

        assert!(
            (worst.1 - want).abs() < 0.05,
            "the same {want:.2} m of travel across an empty floor arrives short when              it is split into more steps. Worst at {:.0} Hz: {:.4} m.{report}
             Nothing is in the way - no walls, no slope, one flat box - so the              loss is a per-call cost inside the character controller.",
            worst.0,
            worst.1
        );
    }

    /// 500 create/destroy cycles must leave the world exactly as they found it.
    ///
    /// Before removal existed, `handles`/`cols`/`grounded` and Rapier's own
    /// `RigidBodySet`/`ColliderSet` all grew by one per body for the life of the
    /// process, because nothing in the file could take a body back out.
    #[test]
    fn create_and_destroy_cycles_leave_the_world_bounded() {
        aurora_phys3d_init(0.0, -9.81, 0.0);
        aurora_phys3d_add_box(0.0, -0.5, 0.0, 50.0, 0.5, 50.0, 0);
        let start = census();
        assert_eq!(start, (1, 1, 1, 1), "just the ground");

        for i in 0..500 {
            let ball = aurora_phys3d_add_sphere(0.0, 5.0, 0.0, 0.5, 1);
            let actor = aurora_phys3d_add_character(2.0, 2.0, 0.0, 0.7, 0.4);
            aurora_phys3d_move_character(actor, 0.1, -0.1, 0.0, 0.016);
            aurora_phys3d_step(0.016);
            assert_eq!(aurora_phys3d_remove(ball), 1, "cycle {i}: ball not removed");
            assert_eq!(
                aurora_phys3d_remove(actor),
                1,
                "cycle {i}: actor not removed"
            );
        }

        let end = census();
        assert_eq!(
            end.0, start.0,
            "Rapier rigid bodies grew from {} to {}",
            start.0, end.0
        );
        assert_eq!(
            end.1, start.1,
            "Rapier colliders grew from {} to {}",
            start.1, end.1
        );
        assert_eq!(end.2, start.2, "live handles grew");
        // Three slots serve all 500 cycles: the ground plus the two the cycle
        // allocates once and then reuses forever.
        assert_eq!(end.3, 3, "handle slots grew to {}", end.3);
    }

    /// Removal must take the collider down WITH the body. A collider left in
    /// the set still answers raycasts and still costs broad-phase work, which
    /// is most of the cost of a body that was supposed to be gone.
    #[test]
    fn removing_a_body_takes_its_collider_down_with_it() {
        aurora_phys3d_init(0.0, 0.0, 0.0);
        let wall = aurora_phys3d_add_box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0);
        aurora_phys3d_step(0.016);
        assert_eq!(census(), (1, 1, 1, 1));
        assert!(
            aurora_phys3d_raycast(0.0, 5.0, 0.0, 0.0, -1.0, 0.0, 20.0) > 0.0,
            "the ray should hit the wall while it exists"
        );

        assert_eq!(aurora_phys3d_remove(wall), 1);
        assert_eq!(
            census(),
            (0, 0, 0, 1),
            "an orphaned collider was left behind"
        );
        // Immediately, with NO step in between: the query pipeline's tree still
        // holds the collider handle, but it must resolve to nothing.
        assert_eq!(
            aurora_phys3d_raycast(0.0, 5.0, 0.0, 0.0, -1.0, 0.0, 20.0),
            -1.0,
            "a removed body still answered a raycast"
        );
        assert_eq!(
            aurora_phys3d_raycast_full(0.0, 5.0, 0.0, 0.0, -1.0, 0.0, 20.0),
            -1
        );
        assert_eq!(aurora_phys3d_overlap_sphere(0.0, 0.0, 0.0, 0.5), -1);
        assert_eq!(
            aurora_phys3d_spherecast(0.0, 5.0, 0.0, 0.0, -1.0, 0.0, 0.25, 20.0, -1),
            -1.0
        );
        // ...and after a step too, once the tree has actually been rebuilt.
        aurora_phys3d_step(0.016);
        assert_eq!(
            aurora_phys3d_raycast(0.0, 5.0, 0.0, 0.0, -1.0, 0.0, 20.0),
            -1.0
        );
    }

    #[test]
    fn a_double_remove_is_refused() {
        aurora_phys3d_init(0.0, 0.0, 0.0);
        let b = aurora_phys3d_add_sphere(0.0, 0.0, 0.0, 1.0, 1);
        assert_eq!(aurora_phys3d_alive(b), 1);
        assert_eq!(aurora_phys3d_remove(b), 1);
        assert_eq!(aurora_phys3d_alive(b), 0);
        assert_eq!(
            aurora_phys3d_remove(b),
            0,
            "double free must report nothing"
        );
        assert_eq!(aurora_phys3d_remove(-1), 0, "the -1 sentinel is not a body");
        assert_eq!(aurora_phys3d_remove(0), 0, "a zeroed handle is not a body");
        assert_eq!(aurora_phys3d_alive(-1), 0);
        assert_eq!(aurora_phys3d_alive(0), 0);
    }

    /// The property the whole change turns on: after a body is removed and its
    /// SLOT is handed to a new body, the old handle must be REFUSED by every
    /// accessor - never quietly answer with the new body's state.
    #[test]
    fn a_removed_handle_is_refused_by_every_accessor_that_takes_one() {
        aurora_phys3d_init(0.0, 0.0, 0.0);
        let dead = aurora_phys3d_add_box(1.0, 2.0, 3.0, 0.5, 0.5, 0.5, 1);
        assert_eq!(aurora_phys3d_remove(dead), 1);
        // The replacement lands in the freed slot. That is exactly the case a
        // bare index gets wrong, so the test is worthless unless it happens.
        let live = aurora_phys3d_add_box(7.0, 8.0, 9.0, 0.5, 0.5, 0.5, 1);
        assert_eq!(
            BodyId::from_i64(dead).unwrap().slot(),
            BodyId::from_i64(live).unwrap().slot(),
            "the freed slot must be reused for this test to mean anything"
        );
        assert_ne!(dead, live, "reuse must still change the handle");
        aurora_phys3d_step(0.016);

        // Readers answer the documented "no such body" value, not `live`'s.
        assert_eq!(aurora_phys3d_alive(dead), 0);
        assert_eq!(aurora_phys3d_alive(live), 1);
        for (name, read) in [
            ("x", aurora_phys3d_x as extern "C" fn(i64) -> f64),
            ("y", aurora_phys3d_y),
            ("z", aurora_phys3d_z),
            ("vel_x", aurora_phys3d_vel_x),
            ("vel_y", aurora_phys3d_vel_y),
            ("vel_z", aurora_phys3d_vel_z),
            ("rot_qx", aurora_phys3d_rot_qx),
            ("rot_qy", aurora_phys3d_rot_qy),
            ("rot_qz", aurora_phys3d_rot_qz),
        ] {
            assert_eq!(read(dead), 0.0, "phys3d_{name} answered for a dead handle");
        }
        assert_eq!(
            aurora_phys3d_rot_qw(dead),
            1.0,
            "identity rotation expected"
        );
        assert_eq!(aurora_phys3d_grounded(dead), 0);
        // `live` is genuinely somewhere else, so "0.0" above cannot be `live`.
        assert_eq!(aurora_phys3d_x(live), 7.0);
        assert_eq!(aurora_phys3d_y(live), 8.0);
        assert_eq!(aurora_phys3d_z(live), 9.0);

        // Writers must not land on `live` either.
        let before = (
            aurora_phys3d_x(live),
            aurora_phys3d_y(live),
            aurora_phys3d_z(live),
            aurora_phys3d_vel_x(live),
            aurora_phys3d_rot_qw(live),
        );
        aurora_phys3d_set_pos(dead, -100.0, -100.0, -100.0);
        aurora_phys3d_set_vel(dead, 50.0, 50.0, 50.0);
        aurora_phys3d_apply_impulse(dead, 500.0, 500.0, 500.0);
        aurora_phys3d_apply_force(dead, 500.0, 500.0, 500.0);
        aurora_phys3d_apply_torque(dead, 500.0, 500.0, 500.0);
        aurora_phys3d_set_angvel(dead, 9.0, 9.0, 9.0);
        aurora_phys3d_set_rot(dead, 1.0, 0.0, 0.0, 0.0);
        aurora_phys3d_move_character(dead, 5.0, 5.0, 5.0, 0.016);
        assert_eq!(
            (
                aurora_phys3d_x(live),
                aurora_phys3d_y(live),
                aurora_phys3d_z(live),
                aurora_phys3d_vel_x(live),
                aurora_phys3d_rot_qw(live),
            ),
            before,
            "a write through a dead handle reached the body that took its slot"
        );
        assert_eq!(aurora_phys3d_grounded(dead), 0, "move_character wrote back");

        // Query filters: excluding a dead body must exclude NOTHING, exactly as
        // the -1 sentinel does. Excluding it as if it were `live` would hide a
        // body the caller never asked to hide.
        let over = |h: i64| aurora_phys3d_raycast_ex(h, 7.0, 40.0, 9.0, 0.0, -1.0, 0.0, 80.0);
        assert_eq!(over(-1), live, "the ray must reach `live` at all");
        assert_eq!(over(dead), live, "a dead exclude hid a live body");
        assert_eq!(over(live), -1, "a live exclude must still exclude");
        let world = |h: i64| aurora_phys3d_raycast_world(h, 7.0, 40.0, 9.0, 0.0, -1.0, 0.0, 80.0);
        assert_eq!(world(dead), live, "a dead exclude hid a live body");
        assert_eq!(world(live), -1);

        // Queries hand back the LIVE handle, never the stale one.
        assert_eq!(
            aurora_phys3d_raycast_full(7.0, 40.0, 9.0, 0.0, -1.0, 0.0, 80.0),
            live
        );
        assert_eq!(aurora_phys3d_hit_body(), live);
        assert_eq!(aurora_phys3d_overlap_sphere(7.0, 8.0, 9.0, 0.3), live);
    }

    /// A teleported body must be where queries say it is, without a step.
    ///
    /// Rapier syncs a body's pose down onto its colliders during `step`, and the
    /// query tree is built from colliders - so `set_pos` alone left every
    /// raycast, spherecast and overlap answering about the body's OLD position.
    /// Silent, and confidently wrong in both directions at once: open space where
    /// the body now stands, and a phantom where it used to.
    ///
    /// This is what an actor driven by game rules rather than by the solver does
    /// every single tick, so the world it walked through was one teleport stale
    /// forever.
    #[test]
    fn a_teleported_body_moves_for_queries_too() {
        aurora_phys3d_init(0.0, 0.0, 0.0);
        let b = aurora_phys3d_add_box(0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0);
        aurora_phys3d_step(0.016);
        assert_eq!(aurora_phys3d_overlap_sphere(0.0, 0.0, 0.0, 0.2), b);

        // No step between the move and the questions: that is the whole point.
        aurora_phys3d_set_pos(b, 10.0, 0.0, 0.0);
        assert_eq!(aurora_phys3d_x(b), 10.0, "the body itself did not move");
        assert_eq!(
            aurora_phys3d_overlap_sphere(0.0, 0.0, 0.0, 0.2),
            -1,
            "a phantom remained where the body used to be"
        );
        assert_eq!(
            aurora_phys3d_overlap_sphere(10.0, 0.0, 0.0, 0.2),
            b,
            "the body is invisible to queries at its new position"
        );
        // And the shape is really there, not just its centre point: a ray fired
        // down the axis must stop at its face.
        let hit = aurora_phys3d_raycast(4.0, 0.0, 0.0, 1.0, 0.0, 0.0, 20.0);
        assert!(
            (hit - 5.5).abs() < 1e-3,
            "ray met the teleported box at {hit}, expected 5.5"
        );

        // Repeated moves keep working - the flag must not latch clean.
        aurora_phys3d_set_pos(b, 0.0, 0.0, 7.0);
        assert_eq!(aurora_phys3d_overlap_sphere(10.0, 0.0, 0.0, 0.2), -1);
        assert_eq!(aurora_phys3d_overlap_sphere(0.0, 0.0, 7.0, 0.2), b);

        // A step must not undo it either: the collider pose written here has to
        // agree with what the solver then propagates from the body.
        aurora_phys3d_step(0.016);
        assert_eq!(aurora_phys3d_overlap_sphere(0.0, 0.0, 7.0, 0.2), b);
    }

    /// A character walks through another character, unless it is solid.
    ///
    /// The default is what a crowd of bots wants: no stacking, no wedging in a
    /// doorway. It is wrong for the one mover a player pushes against, and the
    /// two cannot be reconciled by a global switch - a game needs its enemies to
    /// slide along the level (so they must be characters) AND to stop the player
    /// (so something must collide with them). The flag is on the MOVER, so the
    /// player can be solid while the crowd is not.
    #[test]
    fn a_solid_character_is_stopped_by_another_character() {
        aurora_phys3d_init(0.0, 0.0, 0.0);
        // Something to walk into, standing still at the origin.
        let wall = aurora_phys3d_add_character(0.0, 1.0, 0.0, 0.5, 0.4);
        assert!(wall >= 0);
        aurora_phys3d_step(0.016);

        // Default: straight through it.
        let ghost = aurora_phys3d_add_character(0.0, 1.0, -3.0, 0.5, 0.4);
        aurora_phys3d_move_character(ghost, 0.0, 0.0, 6.0, 1.0);
        assert!(
            aurora_phys3d_z(ghost) > 2.0,
            "a non-solid character was stopped at {} - it should pass through",
            aurora_phys3d_z(ghost)
        );

        // Solid: stopped short of it.
        let solid = aurora_phys3d_add_character(0.0, 1.0, -3.0, 0.5, 0.4);
        aurora_phys3d_character_solid(solid, 1);
        aurora_phys3d_move_character(solid, 0.0, 0.0, 6.0, 1.0);
        let z = aurora_phys3d_z(solid);
        assert!(
            z < 0.0,
            "a solid character reached {z} - it walked through the one at the origin"
        );
        assert!(
            z > -3.0,
            "a solid character did not move at all ({z}) - it should close the gap"
        );

        // And the flag is not one-way: turn it off and the same body passes.
        aurora_phys3d_character_solid(solid, 0);
        aurora_phys3d_move_character(solid, 0.0, 0.0, 6.0, 1.0);
        assert!(
            aurora_phys3d_z(solid) > 2.0,
            "clearing the flag left the character solid at {}",
            aurora_phys3d_z(solid)
        );
    }

    /// A character meets a wall that was built since the last step.
    ///
    /// `move_character` reads the query pipeline but was not calling
    /// `sync_queries`, so a world assembled and then walked in - with no step
    /// between - had no walls in it. Every other spatial query synced; this one
    /// did not, because it does not read like a query.
    ///
    /// No `phys3d_step` anywhere in this test. That is the whole point.
    #[test]
    fn a_character_meets_a_wall_built_since_the_last_step() {
        aurora_phys3d_init(0.0, 0.0, 0.0);
        aurora_phys3d_add_box(0.0, 1.0, 0.0, 2.0, 2.0, 0.5, 0);
        let c = aurora_phys3d_add_character(0.0, 1.0, -3.0, 0.5, 0.4);
        aurora_phys3d_move_character(c, 0.0, 0.0, 6.0, 1.0);
        let z = aurora_phys3d_z(c);
        assert!(
            z < 0.0,
            "character reached {z}: the wall was invisible because nothing stepped"
        );
        assert!(z > -3.0, "character did not move at all ({z})");
    }

    /// A solid character still slides along the WORLD. The flag adds characters
    /// to what stops it; it must not replace what already did.
    #[test]
    fn a_solid_character_still_meets_the_world() {
        aurora_phys3d_init(0.0, 0.0, 0.0);
        aurora_phys3d_add_box(0.0, 1.0, 0.0, 2.0, 2.0, 0.5, 0);
        aurora_phys3d_step(0.016);
        let c = aurora_phys3d_add_character(0.0, 1.0, -3.0, 0.5, 0.4);
        aurora_phys3d_character_solid(c, 1);
        aurora_phys3d_move_character(c, 0.0, 0.0, 6.0, 1.0);
        assert!(
            aurora_phys3d_z(c) < 0.0,
            "a solid character walked through a wall at {}",
            aurora_phys3d_z(c)
        );
    }

    /// `phys3d_init` builds a new world; handles from the old one must not
    /// resolve in it. A fresh registry would restart generations at 1 and hand
    /// the new world's first body the exact `i64` the old world's first had.
    #[test]
    fn a_handle_from_the_previous_world_is_refused() {
        aurora_phys3d_init(0.0, 0.0, 0.0);
        let old = aurora_phys3d_add_box(1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0);
        aurora_phys3d_init(0.0, 0.0, 0.0);
        assert_eq!(aurora_phys3d_alive(old), 0, "a handle outlived its world");
        let new = aurora_phys3d_add_box(4.0, 5.0, 6.0, 0.5, 0.5, 0.5, 0);
        assert_ne!(old, new, "the new world reissued the old world's handle");
        assert_eq!(aurora_phys3d_alive(old), 0);
        // `x(old)` used to be asserted here as a quiet 0.0. It now panics - see
        // `a_handle_from_before_init_panics` - because reading a position off a
        // body in a destroyed world is a program bug and answering 0.0 hid one
        // for the life of a process. `alive` above is the supported way to ask,
        // and this test's point is unchanged: the old handle must never resolve
        // to the new body.
        assert_eq!(aurora_phys3d_x(new), 4.0);
        assert_eq!(aurora_phys3d_remove(old), 0);
        assert_eq!(
            aurora_phys3d_alive(new),
            1,
            "the removal hit the wrong body"
        );
    }

    /// Resetting the world in a loop must not grow the handle store either.
    #[test]
    fn resetting_the_world_in_a_loop_is_bounded() {
        for _ in 0..200 {
            aurora_phys3d_init(0.0, -9.81, 0.0);
            aurora_phys3d_add_box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0);
            aurora_phys3d_add_sphere(0.0, 3.0, 0.0, 0.5, 1);
            aurora_phys3d_step(0.016);
        }
        assert_eq!(census(), (2, 2, 2, 2), "a world reset grew the store");
    }

    #[test]
    fn raycast_full_reports_body_point_and_normal() {
        aurora_phys3d_init(0.0, -9.81, 0.0);
        let ground = aurora_phys3d_add_box(0.0, 0.0, 0.0, 5.0, 1.0, 5.0, 0); // top at y=1
        aurora_phys3d_step(0.016);
        // Ray straight down from above the box.
        let body = aurora_phys3d_raycast_full(0.0, 5.0, 0.0, 0.0, -1.0, 0.0, 20.0);
        assert_eq!(body, ground, "should hit the ground box");
        assert!(
            (aurora_phys3d_hit_y() - 1.0).abs() < 0.05,
            "hit point on top face, got {}",
            aurora_phys3d_hit_y()
        );
        assert!(
            aurora_phys3d_hit_ny() > 0.9,
            "normal should point up, got {}",
            aurora_phys3d_hit_ny()
        );
    }

    /// A negative "exclude" handle means "skip nothing". Clamping it to 0 hid
    /// the FIRST body added - normally the ground - from every probe that had no
    /// body of its own to skip.
    #[test]
    fn a_negative_exclude_handle_skips_nothing() {
        aurora_phys3d_init(0.0, -9.81, 0.0);
        let ground = aurora_phys3d_add_box(0.0, 0.0, 0.0, 20.0, 1.0, 20.0, 0); // handle 0
        aurora_phys3d_step(0.016);
        assert_eq!(
            aurora_phys3d_raycast_ex(-1, 0.0, 5.0, 0.0, 0.0, -1.0, 0.0, 20.0),
            ground,
            "exclude = -1 must not hide body 0"
        );
        assert_eq!(
            aurora_phys3d_raycast_world(-1, 0.0, 5.0, 0.0, 0.0, -1.0, 0.0, 20.0),
            ground,
            "exclude = -1 must not hide body 0 from a world probe either"
        );
        // ...but a real handle still excludes that body.
        assert_eq!(
            aurora_phys3d_raycast_ex(ground, 0.0, 5.0, 0.0, 0.0, -1.0, 0.0, 20.0),
            -1,
            "excluding the ground must make the ray miss"
        );
    }

    #[test]
    fn overlap_sphere_finds_a_body() {
        aurora_phys3d_init(0.0, 0.0, 0.0);
        let b = aurora_phys3d_add_sphere(0.0, 0.0, 0.0, 1.0, 0);
        aurora_phys3d_step(0.016);
        assert_eq!(
            aurora_phys3d_overlap_sphere(0.5, 0.0, 0.0, 0.5),
            b,
            "overlapping sphere found"
        );
        assert_eq!(
            aurora_phys3d_overlap_sphere(20.0, 20.0, 20.0, 0.5),
            -1,
            "far query finds nothing"
        );
    }
    /// `blocking` decides whether OTHERS stop at this character.
    ///
    /// It used to set only the mover's own query filter while the collider's
    /// membership stayed nailed to group 2 forever, so a character marked
    /// non-solid still blocked everything else. That made "walk through a
    /// corpse" impossible to express: a game that marked its creatures solid so
    /// a boss would stop at the player instead of swallowing them also turned
    /// every dead body into a permanent wall, and three corpses sealed a
    /// courtyard.
    ///
    /// Two movers, same start, same push, one blocker each. The only difference
    /// is whether the BLOCKER is solid, and the mover is solid in both cases -
    /// so anything that gets through is the blocker's membership talking.
    #[test]
    fn a_mover_passes_through_a_non_blocking_character() {
        fn run(blocker_solid: bool) -> f64 {
            aurora_phys3d_init(0.0, 0.0, 0.0);
            // A blocker two metres along +x, and a mover at the origin walking
            // into it. Radius 0.5 each, so solid contact happens around x = 1.0.
            let blocker = aurora_phys3d_add_character(2.0, 0.0, 0.0, 0.5, 0.5);
            aurora_phys3d_character_blocking(blocker, if blocker_solid { 1 } else { 0 });
            let mover = aurora_phys3d_add_character(0.0, 0.0, 0.0, 0.5, 0.5);
            aurora_phys3d_character_solid(mover, 1);
            // Walk into it in small steps, the way a controller actually does.
            let mut n = 0;
            while n < 40 {
                aurora_phys3d_move_character(mover, 0.1, 0.0, 0.0, 1.0);
                n += 1;
            }
            aurora_phys3d_x(mover)
        }

        let through = run(false);
        let stopped = run(true);
        assert!(
            through > 2.5,
            "a solid mover should pass straight through a non-solid character,              but stopped at x={through}"
        );
        assert!(
            stopped < 1.5,
            "a solid mover should be stopped by a solid character, but reached              x={stopped}"
        );
    }

    /// A ghost is still findable. Only the character controller skips it -
    /// raycasts and overlaps use the default filter and must still see it, or
    /// marking a creature non-solid would also make it unshootable.
    #[test]
    fn a_ghost_character_is_still_found_by_an_overlap() {
        aurora_phys3d_init(0.0, 0.0, 0.0);
        let ghost = aurora_phys3d_add_character(5.0, 0.0, 0.0, 0.5, 0.5);
        aurora_phys3d_character_blocking(ghost, 0);
        assert_eq!(
            aurora_phys3d_overlap_sphere(5.0, 0.0, 0.0, 0.4),
            ghost,
            "a non-solid character must still be findable by a query"
        );
        // And it is NOT world geometry, so a camera looking for room ignores it.
        assert_eq!(
            aurora_phys3d_overlap_world(5.0, 0.0, 0.0, 0.4),
            -1,
            "a character is not a wall"
        );
    }

    /// Two characters standing INSIDE each other come apart, and the one that
    /// gives way is the SOLID one.
    ///
    /// The rule the whole depenetration exists for, stated on the primitive
    /// rather than on a game's shove. A soulslike had this as one hand-written
    /// call that pushed against a capsule nobody had moved, and it measured a
    /// perfect result while doing nothing.
    #[test]
    fn characters_do_not_stand_inside_each_other() {
        aurora_phys3d_init(0.0, -22.0, 0.0);
        // A floor, or gravity walks them out of the test.
        aurora_phys3d_add_box(0.0, -0.5, 0.0, 60.0, 0.5, 60.0, 0);
        // Dead centre on each other. The player is SOLID - stopped by others -
        // so the player is the one that moves.
        let player = aurora_phys3d_add_character(0.0, 1.0, 0.0, 0.55, 0.35);
        let creature = aurora_phys3d_add_character(0.0, 1.0, 0.0, 0.9, 0.6);
        aurora_phys3d_character_solid(player, 1);

        let want = 0.35 + 0.6;
        let gap0 = gap(player, creature);
        assert!(gap0 < 0.01, "the two did not start on top of each other: {gap0}");

        aurora_phys3d_step(1.0 / 60.0);
        let gap1 = gap(player, creature);
        assert!(
            gap1 >= want - 0.001,
            "one step left them {gap1}m apart, wanted {want}m - separation is              not running, or is not finishing in one step"
        );

        // The creature did NOT move: it is not solid, so it does not give way.
        assert!(
            aurora_phys3d_x(creature).abs() < 1e-4 && aurora_phys3d_z(creature).abs() < 1e-4,
            "a non-solid character was displaced"
        );

        // And it STAYS apart: a correction that fights the next step's solver
        // oscillates, which reads as a jitter rather than as a push.
        for _ in 0..30 {
            aurora_phys3d_step(1.0 / 60.0);
        }
        let gap2 = gap(player, creature);
        assert!(
            gap2 >= want - 0.001,
            "thirty steps later they are {gap2}m apart, wanted {want}m"
        );
    }

    /// Neither solid: they are meant to pass through each other, and a crowd
    /// that separates itself is a crowd that cannot walk through a doorway.
    #[test]
    fn two_ghosts_are_left_alone() {
        aurora_phys3d_init(0.0, -22.0, 0.0);
        aurora_phys3d_add_box(0.0, -0.5, 0.0, 60.0, 0.5, 60.0, 0);
        let a = aurora_phys3d_add_character(0.0, 1.0, 0.0, 0.55, 0.35);
        let b = aurora_phys3d_add_character(0.0, 1.0, 0.0, 0.55, 0.35);
        aurora_phys3d_step(1.0 / 60.0);
        assert!(
            gap(a, b) < 0.01,
            "two non-solid characters were pushed apart; neither gives way"
        );
    }

    fn gap(a: i64, b: i64) -> f64 {
        let dx = aurora_phys3d_x(a) - aurora_phys3d_x(b);
        let dz = aurora_phys3d_z(a) - aurora_phys3d_z(b);
        (dx * dx + dz * dz).sqrt()
    }
}
