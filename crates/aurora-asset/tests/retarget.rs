//! Retargeting a clip from one rig's bone names to another's.

use aurora_asset::model::{Channel, Clip, Interp, Joint, Path, Skeleton};
use glam::{Mat4, Quat, Vec3};

fn joint(name: &str, parent: Option<usize>) -> Joint {
    Joint {
        parent,
        inverse_bind: Mat4::IDENTITY,
        t: Vec3::Y,
        r: Quat::IDENTITY,
        s: Vec3::ONE,
        name: name.into(),
    }
}

/// The animation rig's names.
fn source() -> Skeleton {
    Skeleton {
        joints: vec![
            joint("Hips", None),
            joint("Spine_01", Some(0)),
            joint("Shoulder_L", Some(1)),
            joint("Jaw", Some(1)),
            joint("Prop_L", Some(2)),
        ],
    }
}

/// The character rig's names, deliberately in a different order so a correct
/// retarget must renumber rather than happen to line up.
fn target() -> Skeleton {
    Skeleton {
        joints: vec![
            joint("UpperArm_L", None),
            joint("spine_01", None),
            joint("Pelvis", None),
        ],
    }
}

const MAP: &[(&str, &str)] = &[("Hips", "Pelvis"), ("Shoulder_L", "UpperArm_L")];

fn channel(joint: usize, path: Path) -> Channel {
    Channel {
        joint,
        path,
        interp: Interp::Linear,
        times: vec![0.0, 1.0],
        values: match path {
            Path::Rotation => vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            _ => vec![0.0, 1.0, 0.0, 0.0, 2.0, 0.0],
        },
    }
}

fn clip(channels: Vec<Channel>) -> Clip {
    Clip {
        name: "A_Attack".into(),
        duration: 1.0,
        channels,
    }
}

#[test]
fn channels_are_renumbered_onto_the_target() {
    let c = clip(vec![
        channel(0, Path::Rotation),   // Hips       -> Pelvis      (2)
        channel(2, Path::Rotation),   // Shoulder_L -> UpperArm_L  (0)
    ]);
    let out = c.retarget(&source(), &target(), MAP).expect("retargets");

    assert_eq!(out.channels.len(), 2);
    assert_eq!(out.channels[0].joint, 2);
    assert_eq!(out.channels[1].joint, 0);
    assert_eq!(out.name, "A_Attack");
    assert_eq!(out.duration, 1.0);
}

#[test]
fn a_name_absent_from_the_map_is_matched_as_it_stands() {
    // Spine_01 -> spine_01 by case-insensitive match, with no map entry.
    let c = clip(vec![channel(1, Path::Rotation)]);
    let out = c.retarget(&source(), &target(), MAP).expect("retargets");
    assert_eq!(out.channels[0].joint, 1);
}

#[test]
fn keyframe_data_survives_untouched() {
    // Retargeting changes addressing, never motion. If this drifts, every clip
    // in the library is subtly wrong and nothing says so.
    let c = clip(vec![channel(0, Path::Translation)]);
    let out = c.retarget(&source(), &target(), MAP).expect("retargets");
    assert_eq!(out.channels[0].times, c.channels[0].times);
    assert_eq!(out.channels[0].values, c.channels[0].values);
    assert_eq!(out.channels[0].path, Path::Translation);
    assert_eq!(out.channels[0].interp, Interp::Linear);
}

#[test]
fn joints_the_target_lacks_are_dropped_not_fatal() {
    // A source rig drives bones no character has - a jaw, a weapon socket. The
    // rest of the clip must survive them.
    let c = clip(vec![
        channel(0, Path::Rotation), // Hips   -> Pelvis
        channel(3, Path::Rotation), // Jaw    -> nothing
        channel(4, Path::Rotation), // Prop_L -> nothing
    ]);
    let out = c.retarget(&source(), &target(), MAP).expect("retargets");
    assert_eq!(out.channels.len(), 1);
    assert_eq!(out.channels[0].joint, 2);
}

#[test]
fn a_clip_that_matches_nothing_is_an_error() {
    // Means the bone map is wrong. An empty clip would animate nothing and say
    // nothing, which is the worst of both.
    let c = clip(vec![channel(3, Path::Rotation)]);
    let err = c.retarget(&source(), &target(), MAP).expect_err("must fail");
    assert!(err.contains("bone map is wrong"), "unhelpful error: {err}");
}

#[test]
fn a_channel_naming_a_joint_outside_the_source_is_skipped() {
    let mut c = clip(vec![channel(0, Path::Rotation)]);
    c.channels.push(channel(99, Path::Rotation));
    let out = c.retarget(&source(), &target(), MAP).expect("retargets");
    assert_eq!(out.channels.len(), 1);
}

#[test]
fn retargeting_is_reversible_through_the_inverse_map() {
    // A round trip must land back on the original indices. Catches a map applied
    // in the wrong direction, which otherwise produces plausible-looking motion
    // on the wrong limbs.
    let (s, t) = (source(), target());
    let c = clip(vec![channel(0, Path::Rotation), channel(2, Path::Rotation)]);
    let there = c.retarget(&s, &t, MAP).expect("forward");

    let back_map: Vec<(&str, &str)> = MAP.iter().map(|(a, b)| (*b, *a)).collect();
    let back = there.retarget(&t, &s, &back_map).expect("reverse");

    assert_eq!(back.channels[0].joint, 0);
    assert_eq!(back.channels[1].joint, 2);
}
