//! Terrain builtins against the physics world they have to agree with.
//!
//! The load-bearing check is `height_query_agrees_with_a_physics_raycast`: if
//! the surface a game reads and the surface a game collides with disagree,
//! players float or sink and the cause is invisible from either side alone.

use super::*;
use crate::phys3d::*;

/// Install a heightfield straight from Rust, for shapes a generator would not
/// reliably produce (an exact ramp, a perfect flat).
fn install_field(f: Heightfield) {
    assert_eq!(install(Ok(f)), 1, "install failed");
}

/// A terrain that is perfectly flat at `y`.
fn flat(dim: u32, spacing: f32, y: f32) -> Heightfield {
    Heightfield::new(
        dim,
        spacing,
        -0.5 * (dim - 1) as f32 * spacing,
        -0.5 * (dim - 1) as f32 * spacing,
        vec![y; (dim * dim) as usize],
    )
    .expect("flat field")
}

/// A terrain that rises along +X at a constant `slope` (rise per world unit).
fn ramp(dim: u32, spacing: f32, slope: f32) -> Heightfield {
    let n = dim as usize;
    let mut h = vec![0.0f32; n * n];
    for r in 0..n {
        for c in 0..n {
            h[r * n + c] = c as f32 * spacing * slope;
        }
    }
    let half = -0.5 * (dim - 1) as f32 * spacing;
    Heightfield::new(dim, spacing, half, half, h).expect("ramp field")
}

/// Drop a downward ray from well above the terrain and report the surface Y it
/// hits, plus the body it belongs to.
fn raycast_surface(x: f64, z: f64, top: f64) -> Option<(i64, f64)> {
    let body = aurora_phys3d_raycast_full(x, top, z, 0.0, -1.0, 0.0, top * 4.0 + 100.0);
    (body >= 0).then(|| (body, aurora_phys3d_hit_y()))
}

/// THE agreement check. A grid of downward raycasts onto the registered
/// heightfield collider, each compared with `terrain_height` at the same spot.
#[test]
fn height_query_agrees_with_a_physics_raycast() {
    aurora_phys3d_init(0.0, -9.81, 0.0);
    assert_eq!(aurora_terrain_generate(20_260_725, 129, 1.5, 40.0), 1);
    let ground = aurora_terrain_collider();
    assert!(ground >= 0, "terrain collider was not registered");
    aurora_phys3d_step(0.016);

    let x0 = aurora_terrain_origin_x();
    let z0 = aurora_terrain_origin_z();
    let span = (aurora_terrain_size() - 1) as f64 * aurora_terrain_spacing();
    let n = 64;
    let (mut worst, mut samples, mut misses) = (0.0f64, 0, 0);
    for i in 0..=n {
        for j in 0..=n {
            // Stay a hair inside the footprint: exactly on the border a ray can
            // slip past the collider's edge, which says nothing about the
            // interior surface the query is being checked against.
            let x = x0 + span * (0.002 + 0.996 * i as f64 / n as f64);
            let z = z0 + span * (0.002 + 0.996 * j as f64 / n as f64);
            let want = aurora_terrain_height(x, z);
            match raycast_surface(x, z, 200.0) {
                Some((body, y)) => {
                    assert_eq!(body, ground, "the ray hit body {body}, not the terrain");
                    worst = worst.max((y - want).abs());
                    samples += 1;
                }
                None => misses += 1,
            }
        }
    }
    assert_eq!(misses, 0, "{misses} rays missed the terrain collider");
    assert_eq!(samples, (n + 1) * (n + 1));
    assert!(samples >= 4000, "only {samples} samples");
    assert!(
        worst < 1.0e-3,
        "terrain_height and the collider disagree by up to {worst} m over {samples} samples"
    );
}

/// The collider must be WORLD geometry (group 1): a movement/ground probe has
/// to find the terrain and must not find another player's capsule.
#[test]
fn terrain_is_on_the_world_group_so_ground_probes_ignore_players() {
    aurora_phys3d_init(0.0, -9.81, 0.0);
    install_field(flat(33, 1.0, 0.0));
    let ground = aurora_terrain_collider();
    assert!(ground >= 0);
    // The player doing the probing, and a SECOND player standing between it and
    // the ground. Only the prober is excluded by handle, so anything that keeps
    // the ray off the blocker has to be the collision group.
    let prober = aurora_phys3d_add_character(0.0, 8.0, 0.0, 0.9, 0.4);
    let blocker = aurora_phys3d_add_character(0.0, 5.0, 0.0, 0.9, 0.4);
    assert!(prober >= 0 && blocker >= 0);
    aurora_phys3d_step(0.016);

    // A movement probe (group 1 only) must fall straight through the blocker
    // and land on the terrain.
    let hit = aurora_phys3d_raycast_world(prober, 0.0, 8.0, 0.0, 0.0, -1.0, 0.0, 50.0);
    assert_eq!(
        hit, ground,
        "the world probe hit body {hit}, not the terrain ({ground})"
    );
    assert!(
        aurora_phys3d_hit_y().abs() < 1e-3,
        "world probe landed at y = {}, not on the flat terrain at 0",
        aurora_phys3d_hit_y()
    );
    assert!(
        aurora_phys3d_hit_ny() > 0.99,
        "terrain normal should point up, got {}",
        aurora_phys3d_hit_ny()
    );

    // ...and the blocker really is in the way. The same ray with the same
    // exclusion, differing ONLY in the group filter, finds it: so the group
    // filter is what kept the movement probe off it, not luck about where the
    // capsule happened to be.
    let shot = aurora_phys3d_raycast_ex(prober, 0.0, 8.0, 0.0, 0.0, -1.0, 0.0, 50.0);
    assert_eq!(
        shot, blocker,
        "the ungrouped ray should hit the blocking capsule"
    );
    assert!(
        aurora_phys3d_hit_y() > 4.0,
        "the capsule hit should be up at the capsule, got {}",
        aurora_phys3d_hit_y()
    );
}

/// A character controller dropped onto terrain has to settle ON the surface, at
/// several places including a steep slope, and must never sink through it.
///
/// A capsule at rest on a plane of slope `s` has its centre at
/// `surface + half_height + radius * sqrt(1 + s*s)`: it touches the incline
/// tangentially, so it sits HIGHER than `radius` above the ground directly
/// below it. Asserting the flat-ground offset on a slope would fail for a
/// perfectly correct collider, so the expected pose is derived from the slope.
#[test]
fn characters_settle_on_the_terrain_surface() {
    let (hh, r) = (0.9f64, 0.4f64);
    for (label, slope) in [("flat", 0.0f64), ("slope", 0.4), ("steep slope", 0.8)] {
        // 0.4 is 22 degrees, 0.8 is 39 degrees: past the controller's slide
        // threshold, so the character creeps downhill and has to stay glued to
        // the surface while it does.
        let lift = hh + r * (1.0 + slope * slope).sqrt();
        aurora_phys3d_init(0.0, -20.0, 0.0);
        install_field(if slope == 0.0 {
            flat(65, 1.0, 3.0)
        } else {
            ramp(65, 1.0, slope as f32)
        });
        assert!(aurora_terrain_collider() >= 0);
        aurora_phys3d_step(0.016);

        for &(x, z) in &[(-12.0f64, -8.0f64), (0.0, 0.0), (9.0, 11.0)] {
            let start = aurora_terrain_height(x, z) + lift + 4.0;
            let c = aurora_phys3d_add_character(x, start, z, hh, r);
            let mut vy = 0.0f64;
            let dt = 1.0 / 120.0;
            let mut deepest = f64::INFINITY;
            let mut landed = false;
            for _ in 0..240 {
                vy = if aurora_phys3d_grounded(c) != 0 {
                    landed = true;
                    -0.5
                } else {
                    vy - 20.0 * dt
                };
                aurora_phys3d_move_character(c, 0.0, vy * dt, 0.0, dt);
                aurora_phys3d_step(dt);
                if landed {
                    let (cx, cy, cz) = (aurora_phys3d_x(c), aurora_phys3d_y(c), aurora_phys3d_z(c));
                    deepest = deepest.min(cy - hh - r - aurora_terrain_height(cx, cz));
                }
            }
            let (cx, cy, cz) = (aurora_phys3d_x(c), aurora_phys3d_y(c), aurora_phys3d_z(c));
            let surface = aurora_terrain_height(cx, cz);
            assert!(
                aurora_phys3d_grounded(c) != 0,
                "{label} at ({x},{z}): character never touched down (y = {cy})"
            );
            assert!(
                (cx - x).abs() < 20.0 && (cz - z).abs() < 20.0,
                "{label} at ({x},{z}): slid off to ({cx},{cz}), outside the test's terrain"
            );
            assert!(
                (cy - lift - surface).abs() < 0.1,
                "{label} at ({x},{z}): centre at {cy}, expected {} for a surface at {surface}",
                surface + lift
            );
            assert!(
                deepest > -0.05,
                "{label} at ({x},{z}): the capsule dipped {deepest} below the terrain"
            );
        }
    }
}

/// The documented out-of-bounds contract, through the builtin.
#[test]
fn out_of_bounds_queries_are_defined() {
    install_field(ramp(33, 2.0, 0.25));
    let x0 = aurora_terrain_origin_x();
    let z0 = aurora_terrain_origin_z();
    let span = (aurora_terrain_size() - 1) as f64 * aurora_terrain_spacing();
    assert_eq!(
        aurora_terrain_height(x0 - 1e7, z0 - 1e7),
        aurora_terrain_height(x0, z0)
    );
    assert_eq!(
        aurora_terrain_height(x0 + span + 1e7, z0),
        aurora_terrain_height(x0 + span, z0)
    );
    // The ramp climbs along +X, so those two are genuinely different values:
    // the clamp is following the border, not returning a constant.
    assert!(aurora_terrain_height(x0 + span + 1e7, z0) - aurora_terrain_height(x0 - 1e7, z0) > 1.0);
    assert!(aurora_terrain_height(f64::NAN, f64::NAN).is_finite());
    assert!(aurora_terrain_height(f64::INFINITY, f64::NEG_INFINITY).is_finite());
}

/// Every builtin has to be safe before a terrain exists, because a game that
/// asks early should get a defined answer, not a crash.
#[test]
fn builtins_are_defined_with_no_terrain_loaded() {
    TERRAIN.with(|t| *t.borrow_mut() = None);
    assert_eq!(aurora_terrain_height(3.0, 4.0), 0.0);
    assert_eq!(aurora_terrain_size(), 0);
    assert_eq!(aurora_terrain_spacing(), 0.0);
    assert_eq!(aurora_terrain_origin_x(), 0.0);
    assert_eq!(aurora_terrain_origin_z(), 0.0);
    assert_eq!(aurora_terrain_collider(), -1);
    aurora_terrain_draw();
    let p = "qa_tmp_never_written.aterr";
    assert_eq!(
        unsafe { aurora_terrain_save(p.as_ptr(), p.len() as i64) },
        0
    );
    assert!(!std::path::Path::new(p).exists());
    // A null or empty path must be refused, not dereferenced.
    assert_eq!(unsafe { aurora_terrain_load(std::ptr::null(), 0) }, 0);
    assert_eq!(unsafe { aurora_terrain_load(p.as_ptr(), 0) }, 0);
}

/// Generate, write, and read back: the `.aterr` layout has to survive a real
/// file, and the reloaded terrain has to answer identically.
#[test]
fn a_generated_terrain_round_trips_through_a_file() {
    let dir = std::env::temp_dir().join(format!("aurora_terrain_{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("temp dir");
    let path = dir.join("rt.aterr");
    let p = path.to_str().expect("utf-8 path");

    assert_eq!(aurora_terrain_generate(99, 65, 1.25, 12.0), 1);
    let probes: Vec<(f64, f64, f64)> = (0..40)
        .map(|i| {
            let x = aurora_terrain_origin_x() + i as f64 * 1.7;
            let z = aurora_terrain_origin_z() + i as f64 * 1.1;
            (x, z, aurora_terrain_height(x, z))
        })
        .collect();
    assert_eq!(
        unsafe { aurora_terrain_save(p.as_ptr(), p.len() as i64) },
        1
    );
    let bytes = std::fs::metadata(&path).expect("stat").len();
    assert_eq!(bytes, (24 + 65 * 65 * 4) as u64, "unexpected .aterr size");

    // Drop it and read it back.
    TERRAIN.with(|t| *t.borrow_mut() = None);
    assert_eq!(
        unsafe { aurora_terrain_load(p.as_ptr(), p.len() as i64) },
        1
    );
    assert_eq!(aurora_terrain_size(), 65);
    assert_eq!(aurora_terrain_spacing(), 1.25);
    for (x, z, want) in probes {
        assert_eq!(
            aurora_terrain_height(x, z),
            want,
            "reloaded height differs at ({x},{z})"
        );
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// A file that is not an `.aterr` must fail loudly and leave the previous
/// terrain alone rather than half-replacing it.
#[test]
fn a_bad_file_fails_without_disturbing_the_loaded_terrain() {
    let dir = std::env::temp_dir().join(format!("aurora_terrain_bad_{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("temp dir");
    let path = dir.join("junk.aterr");
    std::fs::write(&path, b"not a terrain at all").expect("write junk");
    let p = path.to_str().expect("utf-8 path");

    install_field(flat(17, 1.0, 7.5));
    assert_eq!(
        unsafe { aurora_terrain_load(p.as_ptr(), p.len() as i64) },
        0
    );
    assert_eq!(
        aurora_terrain_size(),
        17,
        "a failed load replaced the terrain"
    );
    assert_eq!(aurora_terrain_height(0.0, 0.0), 7.5);

    let missing = dir.join("does_not_exist.aterr");
    let mp = missing.to_str().expect("utf-8 path");
    assert_eq!(
        unsafe { aurora_terrain_load(mp.as_ptr(), mp.len() as i64) },
        0
    );
    assert_eq!(aurora_terrain_size(), 17);
    let _ = std::fs::remove_dir_all(&dir);
}

/// A malformed `terrain_generate` must be refused rather than producing a
/// terrain that cannot be tiled.
#[test]
fn generate_rejects_a_dim_that_cannot_tile() {
    TERRAIN.with(|t| *t.borrow_mut() = None);
    assert_eq!(aurora_terrain_generate(1, 64, 1.0, 5.0), 0, "dim 64");
    assert_eq!(aurora_terrain_generate(1, 0, 1.0, 5.0), 0, "dim 0");
    assert_eq!(aurora_terrain_generate(1, -5, 1.0, 5.0), 0, "negative dim");
    assert_eq!(
        aurora_terrain_generate(1, 1 << 40, 1.0, 5.0),
        0,
        "absurd dim"
    );
    assert_eq!(aurora_terrain_generate(1, 33, 0.0, 5.0), 0, "zero spacing");
    assert_eq!(
        aurora_terrain_size(),
        0,
        "a rejected generate installed a terrain"
    );
    assert_eq!(
        aurora_terrain_generate(1, 33, 2.0, 5.0),
        1,
        "dim 33 is valid"
    );
    assert_eq!(aurora_terrain_size(), 33);
}

/// Reloading terrain and re-registering its collider must not stack colliders.
///
/// `terrain_collider` used to add a Rapier body per call, and a heightfield
/// collider is one of the largest a world holds: `dim*dim` samples plus its
/// acceleration structure. The documented mitigation was "call it once per
/// terrain", which a reload loop breaks silently - and the old collider still
/// answered raycasts, so the world kept the previous surface underneath the new
/// one. It now replaces the collider it issued last.
#[test]
fn re_registering_terrain_replaces_its_collider_instead_of_stacking_them() {
    aurora_phys3d_init(0.0, -9.81, 0.0);
    install_field(flat(33, 1.0, 2.0));
    let first = aurora_terrain_collider();
    assert!(first >= 0, "the first registration must succeed");
    assert_eq!(
        crate::phys3d::census(),
        (1, 1, 1, 1),
        "one terrain body, one collider"
    );

    // 200 reloads, each re-registering, exactly as a level-streaming loop does.
    let mut handle = first;
    for i in 0..200 {
        install_field(flat(33, 1.0, 2.0 + i as f32 * 0.01));
        let next = aurora_terrain_collider();
        assert!(next >= 0, "reload {i} failed to register");
        assert_eq!(
            aurora_phys3d_alive(handle),
            0,
            "reload {i} left the previous collider alive"
        );
        handle = next;
        let (bodies, colliders, live, slots) = crate::phys3d::census();
        assert_eq!(
            (bodies, colliders, live),
            (1, 1, 1),
            "reload {i} stacked a collider"
        );
        // TWO handle slots, not one: the new collider is built BEFORE the old
        // one is dropped, so a failed registration leaves the world with the
        // surface it had rather than with none. The two slots then alternate
        // forever, which is the plateau this test is really about.
        assert_eq!(slots, 2, "reload {i} grew the handle store to {slots}");
    }

    // The surviving collider is the CURRENT terrain, not the first one.
    aurora_phys3d_step(0.016);
    let (body, y) = raycast_surface(0.0, 0.0, 50.0).expect("terrain must be hit");
    assert_eq!(body, handle, "the ray hit a stale terrain body");
    let expected = 2.0 + 199.0 * 0.01;
    assert!(
        (y - expected).abs() < 1e-3,
        "hit the old surface: got {y}, want {expected}"
    );
}

/// A world reset between registrations must not make the stored handle remove
/// some unrelated body that inherited its slot.
#[test]
fn re_registering_after_a_world_reset_removes_nothing_else() {
    aurora_phys3d_init(0.0, -9.81, 0.0);
    install_field(flat(33, 1.0, 1.0));
    let stale = aurora_terrain_collider();
    assert!(stale >= 0);

    aurora_phys3d_init(0.0, -9.81, 0.0);
    // This body takes the slot the old terrain collider had.
    let bystander = aurora_phys3d_add_box(9.0, 9.0, 9.0, 1.0, 1.0, 1.0, 0);
    let terrain = aurora_terrain_collider();
    assert!(terrain >= 0);
    assert_ne!(terrain, stale);
    assert_eq!(
        aurora_phys3d_alive(bystander),
        1,
        "the stale terrain handle removed an unrelated body"
    );
    assert_eq!(aurora_phys3d_x(bystander), 9.0);
    assert_eq!(crate::phys3d::census(), (2, 2, 2, 2));
}
