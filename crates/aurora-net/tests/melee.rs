//! Lag-compensated melee: a swing judged against where targets were on the
//! attacker's screen.
//!
//! A sword is not a ray. Validating a swing with `raycast_at_tick` asks whether
//! an infinitely thin line passed through a target, which misses what melee is
//! mostly made of - a wide blade sweeping past a body slightly off the centre
//! line - and cannot express a cleave through two of them at once. Reach and arc
//! are what the frame data specifies, so they are what these cover, along with
//! the ways a caller can get them wrong.

use aurora_net::{LagComp, V3};

fn one_target_at(pos: V3) -> LagComp {
    let mut lc = LagComp::new(64);
    lc.record(0, 1, pos, 0.5, 0.9);
    lc
}

const FORWARD: V3 = [0.0, 0.0, 1.0];
const ORIGIN: V3 = [0.0, 1.0, 0.0];

#[test]
fn a_swing_reaches_a_target_in_front() {
    let lc = one_target_at([0.0, 1.0, 2.0]);
    let hits = lc.melee_at_tick(ORIGIN, FORWARD, 2.5, 120.0, 0, 0);
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].entity, 1);
}

/// Reach is measured to the body, not its centre, so a target's size counts:
/// being big should mean being easier to touch.
#[test]
fn reach_is_measured_to_the_capsule_surface() {
    let lc = one_target_at([0.0, 1.0, 2.0]);
    // The centre is 2.0 away; the surface is 1.5 away through a 0.5 radius.
    let long = lc.melee_at_tick(ORIGIN, FORWARD, 1.6, 120.0, 0, 0);
    assert_eq!(long.len(), 1, "a reach past the surface must connect");
    let short = lc.melee_at_tick(ORIGIN, FORWARD, 1.4, 120.0, 0, 0);
    assert!(short.is_empty(), "a reach short of the surface must not");
}

#[test]
fn a_target_out_of_reach_is_missed() {
    let lc = one_target_at([0.0, 1.0, 9.0]);
    assert!(lc.melee_at_tick(ORIGIN, FORWARD, 2.5, 120.0, 0, 0).is_empty());
}

/// The arc is the point of using a swing rather than a ray.
#[test]
fn a_target_behind_the_swing_is_missed() {
    let lc = one_target_at([0.0, 1.0, -2.0]);
    let hits = lc.melee_at_tick(ORIGIN, FORWARD, 2.5, 120.0, 0, 0);
    assert!(hits.is_empty(), "a swing must not hit behind the swinger");
}

#[test]
fn a_flanking_target_is_inside_a_wide_arc_and_outside_a_narrow_one() {
    // Exactly 90 degrees off the facing.
    let lc = one_target_at([2.0, 1.0, 0.0]);
    let wide = lc.melee_at_tick(ORIGIN, FORWARD, 2.5, 200.0, 0, 0);
    assert_eq!(wide.len(), 1, "a 200 degree sweep covers the flanks");
    let narrow = lc.melee_at_tick(ORIGIN, FORWARD, 2.5, 60.0, 0, 0);
    assert!(narrow.is_empty(), "a 60 degree thrust does not");
}

/// Melee cleaves. Reporting only the nearest would make a weapon's reach through
/// a crowd a property of the netcode rather than of the weapon.
#[test]
fn a_swing_reports_every_target_it_covers_nearest_first() {
    let mut lc = LagComp::new(64);
    lc.record(0, 1, [0.0, 1.0, 3.0], 0.5, 0.9);
    lc.record(0, 2, [0.0, 1.0, 1.5], 0.5, 0.9);
    lc.record(0, 3, [0.0, 1.0, -3.0], 0.5, 0.9);
    let hits = lc.melee_at_tick(ORIGIN, FORWARD, 4.0, 120.0, 0, 0);
    assert_eq!(hits.len(), 2, "both targets in front; the one behind excluded");
    assert_eq!(hits[0].entity, 2, "nearest first");
    assert_eq!(hits[1].entity, 1);
}

#[test]
fn the_swinger_is_never_hit_by_their_own_swing() {
    let mut lc = LagComp::new(64);
    lc.record(0, 7, [0.0, 1.0, 0.0], 0.5, 0.9);
    lc.record(0, 8, [0.0, 1.0, 1.5], 0.5, 0.9);
    let hits = lc.melee_at_tick(ORIGIN, FORWARD, 3.0, 180.0, 0, 7);
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].entity, 8);
}

/// The whole reason this lives in lag compensation.
#[test]
fn the_swing_is_judged_against_the_rewound_position() {
    let mut lc = LagComp::new(64);
    // The target walks away along +Z, one metre per tick, starting at z = 2.
    for tick in 0..20u64 {
        lc.record(tick, 1, [0.0, 1.0, 2.0 + tick as f32], 0.5, 0.9);
    }

    let then = lc.melee_at_tick(ORIGIN, FORWARD, 2.5, 120.0, 0, 0);
    assert_eq!(then.len(), 1, "rewound to tick 0 the target was in reach");

    let now = lc.melee_at_tick(ORIGIN, FORWARD, 2.5, 120.0, 19, 0);
    assert!(now.is_empty(), "by the latest tick it has long since walked off");
}

/// A view older than anything recorded clamps to where the target first
/// appeared, matching the raycast on a fresh spawn.
#[test]
fn a_view_older_than_the_history_clamps_rather_than_missing() {
    let mut lc = LagComp::new(64);
    lc.record(100, 1, [0.0, 1.0, 1.5], 0.5, 0.9);
    let hits = lc.melee_at_tick(ORIGIN, FORWARD, 2.5, 120.0, 5, 0);
    assert_eq!(hits.len(), 1, "a target that just spawned must still be hittable");
}

/// A target overlapping the swinger has no meaningful direction, and is the one
/// thing the swing is definitely touching.
#[test]
fn a_target_on_top_of_the_swinger_is_hit_whatever_the_arc() {
    let lc = one_target_at([0.0, 1.0, 0.0]);
    let hits = lc.melee_at_tick(ORIGIN, FORWARD, 1.0, 30.0, 0, 0);
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].distance, 0.0, "inside the body is zero, not negative");
}

/// Height is the capsule's job: a swing must not miss because the target stood
/// on a step, nor connect with someone on a roof.
#[test]
fn height_is_decided_by_the_capsule_not_the_arc() {
    let near = one_target_at([0.0, 1.6, 1.5]);
    assert_eq!(
        near.melee_at_tick(ORIGIN, FORWARD, 2.0, 120.0, 0, 0).len(),
        1,
        "a small height difference must not save a target"
    );

    let above = one_target_at([0.0, 9.0, 1.5]);
    assert!(
        above.melee_at_tick(ORIGIN, FORWARD, 2.0, 120.0, 0, 0).is_empty(),
        "reach is three-dimensional"
    );
}

#[test]
fn a_swing_with_no_reach_hits_nothing() {
    let lc = one_target_at([0.0, 1.0, 0.5]);
    assert!(lc.melee_at_tick(ORIGIN, FORWARD, 0.0, 120.0, 0, 0).is_empty());
    assert!(lc.melee_at_tick(ORIGIN, FORWARD, -3.0, 120.0, 0, 0).is_empty());
}

/// A caller with no horizontal facing has not described a swing. Answering
/// "everything" would turn that bug into a free hit on the whole arena.
#[test]
fn a_swing_with_no_horizontal_facing_hits_nothing() {
    let lc = one_target_at([0.0, 1.0, 1.5]);
    assert!(lc.melee_at_tick(ORIGIN, [0.0, 1.0, 0.0], 2.5, 120.0, 0, 0).is_empty());
}

#[test]
fn a_full_circle_arc_covers_every_direction() {
    let mut lc = LagComp::new(64);
    lc.record(0, 1, [0.0, 1.0, 1.5], 0.5, 0.9);
    lc.record(0, 2, [0.0, 1.0, -1.5], 0.5, 0.9);
    lc.record(0, 3, [1.5, 1.0, 0.0], 0.5, 0.9);
    let hits = lc.melee_at_tick(ORIGIN, FORWARD, 2.5, 360.0, 0, 0);
    assert_eq!(hits.len(), 3, "a spin hits all round");
}

/// Ties must resolve the same way every time, or two servers disagree about who
/// a cleave killed first.
#[test]
fn equal_distances_resolve_deterministically() {
    let mut lc = LagComp::new(64);
    lc.record(0, 9, [0.6, 1.0, 1.5], 0.5, 0.9);
    lc.record(0, 4, [-0.6, 1.0, 1.5], 0.5, 0.9);
    let a = lc.melee_at_tick(ORIGIN, FORWARD, 3.0, 180.0, 0, 0);
    let b = lc.melee_at_tick(ORIGIN, FORWARD, 3.0, 180.0, 0, 0);
    assert_eq!(a.len(), 2);
    assert_eq!(a[0].entity, 4, "equal distance breaks by entity id");
    assert_eq!(
        a.iter().map(|h| h.entity).collect::<Vec<_>>(),
        b.iter().map(|h| h.entity).collect::<Vec<_>>()
    );
}

/// A removed entity cannot be hit: a target that died or disconnected must stop
/// absorbing swings.
#[test]
fn a_removed_entity_is_no_longer_a_target() {
    let mut lc = LagComp::new(64);
    lc.record(0, 1, [0.0, 1.0, 1.5], 0.5, 0.9);
    assert_eq!(lc.melee_at_tick(ORIGIN, FORWARD, 2.5, 120.0, 0, 0).len(), 1);
    lc.remove(1);
    assert!(lc.melee_at_tick(ORIGIN, FORWARD, 2.5, 120.0, 0, 0).is_empty());
}
