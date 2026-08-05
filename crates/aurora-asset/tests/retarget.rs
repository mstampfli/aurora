//! Retargeting a clip from one rig to another.
//!
//! Retargeting transfers world-space motion, not channel values. The rigs here
//! deliberately disagree about local bone frames, because that is the case a
//! value copy gets wrong and the whole reason the transfer exists.

use aurora_asset::model::{Channel, Clip, Interp, Joint, Path, Retarget, RootMotion, Skeleton};
use glam::{Mat4, Quat, Vec3};

fn joint(name: &str, parent: Option<usize>, rest: Quat) -> Joint {
    Joint {
        parent,
        inverse_bind: Mat4::IDENTITY,
        t: Vec3::Y,
        r: rest,
        s: Vec3::ONE,
        name: name.into(),
    }
}

/// The animation rig: bones rest at identity, as a clip-only export has them.
fn source() -> Skeleton {
    Skeleton {
        joints: vec![
            joint("Hips", None, Quat::IDENTITY),
            joint("Spine_01", Some(0), Quat::IDENTITY),
            joint("Jaw", Some(1), Quat::IDENTITY),
        ],
    }
}

/// The character rig: same chain, but every bone rests a quarter turn about Z,
/// standing in for a rig that runs its bones along a different axis. It also
/// stores the child BEFORE its parent, so nothing may depend on storage order.
fn target() -> Skeleton {
    let bent = Quat::from_rotation_z(std::f32::consts::FRAC_PI_2);
    Skeleton {
        joints: vec![
            joint("spine_01", Some(1), bent),
            joint("Pelvis", None, bent),
        ],
    }
}

const MAP: &[(&str, &str)] = &[("Hips", "Pelvis")];

fn rotate_clip(joint: usize, angle: f32) -> Clip {
    let q = Quat::from_rotation_y(angle);
    Clip {
        name: "A_Attack".into(),
        duration: 1.0,
        channels: vec![Channel {
            joint,
            path: Path::Rotation,
            interp: Interp::Linear,
            times: vec![0.0, 1.0],
            values: vec![0.0, 0.0, 0.0, 1.0, q.x, q.y, q.z, q.w],
        }],
        root: None,
    }
}

fn run(c: &Clip, translate: &[&str]) -> Result<Clip, String> {
    let (s, t) = (source(), target());
    c.retarget(&Retarget {
        source: &s,
        source_rest: &s,
        target: &t,
        rename: MAP,
        translate,
    })
}

/// World-space motion survives the move between rigs.
///
/// This is what retargeting means, and it is the assertion a value copy cannot
/// pass: the target rests a quarter turn away from the source, so copying the
/// clip's quaternions would land the bone somewhere else entirely. What must
/// hold is that the bone turns by the SAME amount, about the same world axis,
/// away from its own rest as the source did from its.
#[test]
fn world_space_motion_is_preserved_across_differing_rest_frames() {
    let angle = 0.7f32;
    let out = run(&rotate_clip(0, angle), &[]).expect("retargets");
    let t = target();

    let pelvis = t.joints.iter().position(|j| j.name == "Pelvis").unwrap();
    let rest = t.joints[pelvis].r;

    let (_, r, _) = t.sample(Some(&out), 1.0);
    // Pelvis is a root, so its local rotation is also its world rotation.
    let delta = r[pelvis] * rest.inverse();

    let (axis, turned) = delta.to_axis_angle();
    assert!(
        (turned - angle).abs() < 1e-3,
        "bone turned {turned} rad, source turned {angle}"
    );
    assert!(
        axis.dot(Vec3::Y).abs() > 0.99,
        "bone turned about {axis:?}, expected the world Y axis"
    );
}

/// At rest the clip is a no-op, so the target must hold its own rest pose.
#[test]
fn a_zero_motion_clip_leaves_the_target_at_rest() {
    let out = run(&rotate_clip(0, 0.0), &[]).expect("retargets");
    let t = target();
    let (_, r, _) = t.sample(Some(&out), 0.0);
    for (i, j) in t.joints.iter().enumerate() {
        let drift = (r[i].inverse() * j.r).to_axis_angle().1;
        assert!(drift < 1e-3, "{} drifted {drift} rad from rest", j.name);
    }
}

#[test]
fn every_mapped_bone_gets_a_track_addressed_to_the_target() {
    let out = run(&rotate_clip(0, 0.5), &[]).expect("retargets");
    let t = target();

    // Hips->Pelvis and Spine_01->spine_01 both map; Jaw does not.
    let driven: std::collections::HashSet<usize> = out.channels.iter().map(|c| c.joint).collect();
    assert_eq!(driven.len(), 2, "expected both mapped bones to be driven");
    for name in ["Pelvis", "spine_01"] {
        let i = t.joints.iter().position(|j| j.name == name).unwrap();
        assert!(driven.contains(&i), "{name} has no track");
    }
    for c in &out.channels {
        assert!(
            c.joint < t.joints.len(),
            "track addresses joint {}",
            c.joint
        );
    }
}

#[test]
fn a_clip_that_matches_nothing_is_an_error() {
    // Means the bone map is wrong. An empty clip would animate nothing and say
    // nothing, which is the worst of both.
    let s = source();
    let t = Skeleton {
        joints: vec![joint("unrelated", None, Quat::IDENTITY)],
    };
    let err = rotate_clip(0, 0.5)
        .retarget(&Retarget {
            source: &s,
            source_rest: &s,
            target: &t,
            rename: &[],
            translate: &[],
        })
        .expect_err("must fail");
    assert!(err.contains("bone map is wrong"), "unhelpful error: {err}");
}

#[test]
fn scale_is_never_transferred() {
    // A clip that does not author scale still reports the source rig's own, and
    // on a rig whose root carries a unit conversion that resizes the character.
    let mut c = rotate_clip(0, 0.5);
    c.channels.push(Channel {
        joint: 0,
        path: Path::Scale,
        interp: Interp::Linear,
        times: vec![0.0],
        values: vec![100.0, 100.0, 100.0],
    });
    let out = run(&c, &[]).expect("retargets");
    assert!(out.channels.iter().all(|c| c.path != Path::Scale));
}

fn translation_clip(joint: usize) -> Clip {
    Clip {
        name: "A_Walk".into(),
        duration: 1.0,
        channels: vec![Channel {
            joint,
            path: Path::Translation,
            interp: Interp::Linear,
            times: vec![0.0, 1.0],
            values: vec![0.0, 0.0, 0.0, 0.0, 0.0, 2.0],
        }],
        root: None,
    }
}

#[test]
fn translation_only_reaches_the_bones_the_caller_names() {
    // A clip-only export has no bone offsets, so its translations are zeroes
    // that would wipe the target's bone lengths. Only root travel belongs to the
    // clip, and the caller says which bone that is.
    let without = run(&translation_clip(0), &[]).expect("retargets");
    assert!(without.channels.iter().all(|c| c.path != Path::Translation));

    let with = run(&translation_clip(0), &["Pelvis"]).expect("retargets");
    assert_eq!(
        with.channels
            .iter()
            .filter(|c| c.path == Path::Translation)
            .count(),
        1
    );
}

/// An exporter decorates a source bone with the namespace or armature it came from
/// (`Armature|Hips`, `mixamorig:Spine_01`). That is an export setting, not a different
/// bone - and this used to compare names EXACTLY, so every decorated bone silently
/// dropped its channel and the clip animated a corpse. Same class of failure as the
/// authored travel that was thrown away for the same reason, on the next line down.
#[test]
fn a_decorated_source_rig_still_lands_every_channel() {
    let mut decorated = source();
    decorated.joints[0].name = "Armature|Hips".into();
    decorated.joints[1].name = "mixamorig:Spine_01".into();
    let t = target();

    let clip = rotate_clip(0, 0.6);
    let out = clip
        .retarget(&Retarget {
            source: &decorated,
            source_rest: &decorated,
            target: &t,
            rename: MAP,
            translate: &[],
        })
        .expect("retargets");

    // Same bones driven, and the same pose, as the undecorated rig produces: the
    // decoration must make no difference at all, not merely fail to error.
    let plain = run(&clip, &[]).expect("retargets");
    let driven: std::collections::HashSet<usize> = out.channels.iter().map(|c| c.joint).collect();
    assert_eq!(
        driven.len(),
        2,
        "both decorated bones must map onto the target"
    );
    for time in [0.0, 0.5, 1.0] {
        let (_, want, _) = t.sample(Some(&plain), time);
        let (_, got, _) = t.sample(Some(&out), time);
        for (i, j) in t.joints.iter().enumerate() {
            let drift = (got[i].inverse() * want[i]).to_axis_angle().1;
            assert!(drift < 1e-3, "{} drifted {drift} rad at {time}s", j.name);
        }
    }

    // And the translate whitelist answers a decorated name too, or the one channel
    // that carries travel is the one that gets dropped.
    let moved = translation_clip(0)
        .retarget(&Retarget {
            source: &decorated,
            source_rest: &decorated,
            target: &t,
            rename: MAP,
            translate: &["Pelvis"],
        })
        .expect("retargets");
    assert_eq!(
        moved
            .channels
            .iter()
            .filter(|c| c.path == Path::Translation)
            .count(),
        1,
        "the hips' translation must survive a decorated source name"
    );
}

#[test]
fn a_bone_the_clip_does_not_drive_still_holds_its_rest_orientation() {
    // Only Hips is animated, but spine_01 maps too. Its track must carry the
    // rest orientation rather than identity, or every undriven bone snaps.
    let out = run(&rotate_clip(0, 0.5), &[]).expect("retargets");
    let t = target();
    let spine = t.joints.iter().position(|j| j.name == "spine_01").unwrap();
    let (_, r, _) = t.sample(Some(&out), 0.0);
    let drift = (r[spine].inverse() * t.joints[spine].r).to_axis_angle().1;
    assert!(drift < 1e-3, "spine_01 drifted {drift} rad at rest");
}

#[test]
fn every_emitted_rotation_is_a_unit_quaternion() {
    // The target stores a child before its parent, so composing a chain must not
    // depend on storage order; a mis-ordered compose shows up here as drift or
    // as a non-unit result feeding the skinning matrices.
    let out = run(&rotate_clip(0, 0.4), &[]).expect("retargets");
    let t = target();
    for step in 0..5 {
        let (_, r, _) = t.sample(Some(&out), step as f32 * 0.25);
        assert!(r
            .iter()
            .all(|q| q.is_finite() && (q.length() - 1.0).abs() < 1e-3));
    }
}

// ---------------------------------------------------------------------------
// Root motion
// ---------------------------------------------------------------------------

/// A clip that covers ground: three units forward over its length, on its own
/// track rather than on a bone.
fn travelling_clip() -> Clip {
    let mut c = rotate_clip(0, 0.5);
    c.root = Some(RootMotion {
        interp: Interp::Linear,
        times: vec![0.0, 1.0],
        values: vec![0.0, 0.0, 0.0, 0.0, 0.0, 3.0],
    });
    c
}

/// The travel MUST survive the move between rigs.
///
/// This is the defect the feature was built for: an animation pack authors a
/// lunge on a `Root` bone the character rig does not have, so matching by name
/// found nothing and every authored distance in the moveset was silently thrown
/// away here. Every attack then played on the spot.
#[test]
fn root_motion_survives_the_move_between_rigs() {
    let out = run(&travelling_clip(), &[]).expect("retargets");
    let travel = out.root_pass();
    // These two rigs hang their body at the same height, so the stride is the
    // one the animator authored, undiminished.
    assert!(
        (travel.z - 3.0).abs() < 1e-4,
        "expected the authored 3.0, got {travel:?}"
    );
    assert!(
        travel.x.abs() < 1e-6 && travel.y.abs() < 1e-6,
        "only forward: {travel:?}"
    );
}

#[test]
fn a_clip_authored_in_place_gains_no_travel() {
    let out = run(&rotate_clip(0, 0.5), &[]).expect("retargets");
    assert!(
        out.root.is_none(),
        "a clip that does not travel must not start"
    );
    assert_eq!(out.root_pass(), Vec3::ZERO);
    assert_eq!(out.root_pos(0.5), Vec3::ZERO);
}

/// Travel is not a bone channel, so naming bones cannot suppress it and cannot
/// duplicate it into the pose.
#[test]
fn travel_is_independent_of_the_translate_whitelist() {
    for names in [&[][..], &["Pelvis"][..]] {
        let out = run(&travelling_clip(), names).expect("retargets");
        assert!(
            (out.root_pass().z - 3.0).abs() < 1e-4,
            "travel lost with {names:?}"
        );
    }
}

/// A clip that is nothing but travel is still a clip, and must not come out with
/// keyless bone channels that would panic the first sampler to touch them.
#[test]
fn a_clip_that_is_only_travel_still_retargets() {
    let mut c = travelling_clip();
    c.channels.clear();
    let out = run(&c, &[]).expect("a clip with travel and no pose is not nothing");
    assert!(out.channels.is_empty());
    assert!((out.root_pass().z - 3.0).abs() < 1e-4);
    let (t, r, s) = target().sample(Some(&out), 0.5);
    assert!(t.iter().chain(s.iter()).all(|v| v.is_finite()));
    assert!(r.iter().all(|q| q.is_finite()));
}

/// A stride belongs to the body that walks it. The same clip on a body whose
/// legs are twice as long has to cover twice the ground, or a boss built at
/// twice the scale minces on the spot while its feet skate.
#[test]
fn a_bigger_body_covers_proportionally_more_ground() {
    let (s, mut t) = (source(), target());
    // Lift the whole target rig to twice the hip height.
    for j in &mut t.joints {
        j.t *= 2.0;
    }
    let out = travelling_clip()
        .retarget(&Retarget {
            source: &s,
            source_rest: &s,
            target: &t,
            rename: MAP,
            translate: &[],
        })
        .expect("retargets");
    assert!(
        (out.root_pass().z - 6.0).abs() < 1e-4,
        "twice the leg should cover twice the ground, got {:?}",
        out.root_pass()
    );
}
