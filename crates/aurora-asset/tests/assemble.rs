//! Building one rig out of modular parts.
//!
//! A modular pack ships no whole body and no skeleton file: every part carries
//! only the bones it deforms with, plus the chain above them to hang from. The
//! rig exists only as the union, so these cover what that union has to get right
//! - and what it has to refuse, because a part quietly averaged onto the wrong
//! bone is a seam that opens in some poses and not others.

use aurora_asset::model::{Joint, Skeleton};
use glam::{Mat4, Quat, Vec3};

fn joint(name: &str, parent: Option<usize>, t: Vec3) -> Joint {
    Joint {
        parent,
        inverse_bind: Mat4::IDENTITY,
        t,
        r: Quat::IDENTITY,
        s: Vec3::ONE,
        name: name.into(),
    }
}

fn skel(joints: Vec<Joint>) -> Skeleton {
    Skeleton { joints }
}

/// root -> spine -> arm, the shared trunk two parts would each carry.
fn trunk() -> Skeleton {
    skel(vec![
        joint("root", None, Vec3::ZERO),
        joint("spine", Some(0), Vec3::Y),
        joint("arm", Some(1), Vec3::Y),
    ])
}

#[test]
fn merging_into_an_empty_rig_takes_the_whole_part() {
    let mut rig = skel(vec![]);
    let added = rig
        .merge(&trunk(), 1e-3)
        .expect("first part defines the rig");
    assert_eq!(added, 3);
    assert_eq!(rig.joint_count(), 3);
}

#[test]
fn a_shared_trunk_is_not_duplicated() {
    let mut rig = skel(vec![]);
    rig.merge(&trunk(), 1e-3).unwrap();

    // A second part carrying the same trunk plus one bone of its own.
    let mut hand = trunk();
    hand.joints.push(joint("finger", Some(2), Vec3::Y));
    let added = rig.merge(&hand, 1e-3).expect("same rig");

    assert_eq!(added, 1, "only the new bone should be added");
    assert_eq!(rig.joint_count(), 4);
    assert_eq!(rig.index_of("finger"), Some(3));
}

/// The point of the whole exercise: two parts that each know half the rig
/// produce the complete one.
#[test]
fn two_partial_parts_compose_into_the_full_rig() {
    // A hand knows its fingers but stops at the spine.
    let hand = skel(vec![
        joint("root", None, Vec3::ZERO),
        joint("spine", Some(0), Vec3::Y),
        joint("arm", Some(1), Vec3::Y),
        joint("finger", Some(2), Vec3::Y),
    ]);
    // A helmet knows the head but nothing below the neck.
    let helm = skel(vec![
        joint("root", None, Vec3::ZERO),
        joint("spine", Some(0), Vec3::Y),
        joint("neck", Some(1), Vec3::new(0.0, 1.0, 0.5)),
        joint("head", Some(2), Vec3::Y),
    ]);

    let mut rig = skel(vec![]);
    rig.merge(&hand, 1e-3).unwrap();
    rig.merge(&helm, 1e-3).unwrap();

    assert_eq!(rig.joint_count(), 6, "root, spine, arm, finger, neck, head");
    for bone in ["root", "spine", "arm", "finger", "neck", "head"] {
        assert!(
            rig.index_of(bone).is_some(),
            "{bone} missing from the union"
        );
    }
}

/// Parents must be rewritten into the merged rig's own indices, not carried over
/// from the part. A part whose parent index happens to be valid in the union but
/// names a different bone is the subtle version of this bug.
#[test]
fn parents_are_remapped_into_the_merged_rig() {
    let mut rig = skel(vec![]);
    rig.merge(&trunk(), 1e-3).unwrap();

    // Same bones, declared in a different order, plus one new leaf.
    let other = skel(vec![
        joint("arm", Some(2), Vec3::Y),
        joint("hand", Some(0), Vec3::Y),
        joint("spine", Some(3), Vec3::Y),
        joint("root", None, Vec3::ZERO),
    ]);
    rig.merge(&other, 1e-3).unwrap();

    let hand = rig.index_of("hand").expect("hand merged");
    let arm = rig.index_of("arm").expect("arm present");
    assert_eq!(
        rig.joints[hand].parent,
        Some(arm),
        "hand must hang off the union's arm, not the part's index for it"
    );
}

/// Order within a part is arbitrary: a child may be listed before its parent.
#[test]
fn a_part_listing_children_before_parents_still_merges() {
    let reversed = skel(vec![
        joint("tip", Some(2), Vec3::Y),
        joint("mid", Some(2), Vec3::Y),
        joint("base", None, Vec3::ZERO),
    ]);
    let mut rig = skel(vec![]);
    rig.merge(&reversed, 1e-3).expect("order must not matter");
    assert_eq!(rig.joint_count(), 3);
    let base = rig.index_of("base").unwrap();
    assert_eq!(rig.joints[rig.index_of("mid").unwrap()].parent, Some(base));
}

/// The refusal that matters. Same bone names, different rest positions: these are
/// not the same rig, and averaging them would bind a limb to the wrong place.
#[test]
fn a_part_from_another_body_is_refused() {
    let mut rig = skel(vec![]);
    rig.merge(&trunk(), 1e-3).unwrap();

    let stretched = skel(vec![
        joint("root", None, Vec3::ZERO),
        joint("spine", Some(0), Vec3::new(0.0, 3.0, 0.0)),
    ]);
    let err = rig
        .merge(&stretched, 1e-3)
        .expect_err("a bone two units out of place must be refused");
    assert!(
        err.contains("spine"),
        "the error should name the bone: {err}"
    );
}

/// Tolerance absorbs an exporter's rounding rather than rejecting on it.
#[test]
fn rounding_within_tolerance_is_accepted() {
    let mut rig = skel(vec![]);
    rig.merge(&trunk(), 1e-3).unwrap();

    let rounded = skel(vec![
        joint("root", None, Vec3::ZERO),
        joint("spine", Some(0), Vec3::new(0.0, 1.000_2, 0.0)),
    ]);
    assert!(rig.merge(&rounded, 1e-3).is_ok());
    assert_eq!(rig.joint_count(), 3, "nothing new, nothing duplicated");
}

/// A real measured bind matrix must survive a later part's placeholder, and must
/// also replace one. A joint no part deforms with carries identity; whichever
/// order the parts arrive in, the measurement is what has to end up in the rig.
#[test]
fn a_measured_bind_matrix_beats_a_placeholder_either_way() {
    let measured = Mat4::from_translation(Vec3::new(1.0, 2.0, 3.0));

    // Placeholder first, measurement second.
    let mut rig = skel(vec![]);
    rig.merge(&trunk(), 1e-3).unwrap();
    let mut real = trunk();
    real.joints[1].inverse_bind = measured;
    rig.merge(&real, 1e-3).unwrap();
    assert_eq!(
        rig.joints[1].inverse_bind, measured,
        "placeholder must yield"
    );

    // Measurement first, placeholder second.
    let mut rig = skel(vec![]);
    let mut real = trunk();
    real.joints[1].inverse_bind = measured;
    rig.merge(&real, 1e-3).unwrap();
    rig.merge(&trunk(), 1e-3).unwrap();
    assert_eq!(
        rig.joints[1].inverse_bind, measured,
        "a placeholder must not overwrite a measurement"
    );
}

/// A part whose parent chain never reaches the rig is an error, not an infinite
/// loop. This is the case that would hang the merge if progress were not checked.
#[test]
fn an_unreachable_parent_is_reported_rather_than_looping() {
    let mut rig = skel(vec![]);
    let orphaned = skel(vec![
        joint("a", Some(1), Vec3::Y),
        joint("b", Some(0), Vec3::Y),
    ]);
    let err = rig
        .merge(&orphaned, 1e-3)
        .expect_err("a cycle has no root and cannot be merged");
    assert!(err.contains("not reachable"), "{err}");
}

#[test]
fn merging_an_empty_part_changes_nothing() {
    let mut rig = trunk();
    let added = rig.merge(&skel(vec![]), 1e-3).unwrap();
    assert_eq!(added, 0);
    assert_eq!(rig.joint_count(), 3);
}

/// A part that spells the shared trunk differently is still the SAME trunk.
///
/// `index_of` compared with `==`, which made it the third rule for one name:
/// the renderer resolves a joint case-insensitively and tolerating an armature
/// prefix, the retarget does too, and the door `merge` builds the union through
/// insisted on exact bytes. The packs genuinely disagree - the character parts
/// spell it `spine_01` and the clip rig `Spine_01` - so a rig assembled from
/// parts that disagreed would have carried the bone twice, while `joint_index`
/// reported only the first and the second never animated.
///
/// Exporters differ on both axes, so both are covered: case, and the
/// `Armature|bone` decoration an FBX exporter adds.
#[test]
fn a_trunk_spelled_differently_is_the_same_trunk() {
    let mut rig = skel(vec![]);
    rig.merge(&trunk(), 1e-3).unwrap();

    let mut shouty = skel(vec![
        joint("ROOT", None, Vec3::ZERO),
        joint("Spine", Some(0), Vec3::Y),
        joint("Armature|arm", Some(1), Vec3::Y),
        joint("finger", Some(2), Vec3::Y),
    ]);
    shouty.joints[3].parent = Some(2);

    let added = rig
        .merge(&shouty, 1e-3)
        .expect("same rig, spelled differently");
    assert_eq!(added, 1, "only `finger` is new - the trunk is the trunk");
    assert_eq!(rig.joint_count(), 4);

    // And the new bone hangs off the EXISTING arm, not off a second copy of it.
    let arm = rig.index_of("arm").expect("arm present");
    let finger = rig.index_of("finger").expect("finger merged");
    assert_eq!(rig.joints[finger].parent, Some(arm));
}
