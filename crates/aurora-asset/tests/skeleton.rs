//! Skeleton, pose and bounds behaviour, with the weight on what a caller can do
//! wrong or a file can get wrong: cycles, empty inputs, out-of-range channels,
//! unweighted vertices, and sampling outside a clip's range.

use aurora_asset::mesh::{MeshData, Vertex};
use aurora_asset::model::{Channel, Clip, Interp, Joint, Model, Path, Primitive, Skeleton};
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

/// A chain of three joints each one unit further along +Y.
fn chain() -> Skeleton {
    Skeleton {
        joints: vec![
            joint("root", None, Vec3::ZERO),
            joint("mid", Some(0), Vec3::Y),
            joint("tip", Some(1), Vec3::Y),
        ],
    }
}

#[test]
fn rest_globals_accumulate_down_the_chain() {
    let g = chain().rest_globals();
    assert_eq!(g[0].w_axis.truncate(), Vec3::ZERO);
    assert_eq!(g[1].w_axis.truncate(), Vec3::new(0.0, 1.0, 0.0));
    assert_eq!(g[2].w_axis.truncate(), Vec3::new(0.0, 2.0, 0.0));
}

#[test]
fn empty_skeleton_has_no_globals() {
    let skel = Skeleton { joints: vec![] };
    assert!(skel.rest_globals().is_empty());
    assert!(skel.bind_matrices().is_empty());
}

#[test]
fn joint_order_does_not_have_to_be_parent_first() {
    // Same chain, stored child-first. Resolution walks parents, so the answer
    // must not depend on the order an importer happened to emit.
    let skel = Skeleton {
        joints: vec![
            joint("tip", Some(2), Vec3::Y),
            joint("mid", Some(2), Vec3::Y),
            joint("root", None, Vec3::ZERO),
        ],
    };
    let g = skel.rest_globals();
    assert_eq!(g[2].w_axis.truncate(), Vec3::ZERO);
    assert_eq!(g[1].w_axis.truncate(), Vec3::new(0.0, 1.0, 0.0));
    assert_eq!(g[0].w_axis.truncate(), Vec3::new(0.0, 1.0, 0.0));
}

#[test]
fn self_parented_joint_terminates_instead_of_recursing() {
    // A malformed file can name a joint its own parent. That must produce a
    // finite answer, not a blown stack.
    let skel = Skeleton {
        joints: vec![joint("loop", Some(0), Vec3::Y)],
    };
    let g = skel.rest_globals();
    assert_eq!(g.len(), 1);
    assert!(g[0].w_axis.is_finite());
}

#[test]
fn mutual_parent_cycle_terminates() {
    let skel = Skeleton {
        joints: vec![joint("a", Some(1), Vec3::Y), joint("b", Some(0), Vec3::Y)],
    };
    let g = skel.rest_globals();
    assert_eq!(g.len(), 2);
    assert!(g.iter().all(|m| m.w_axis.is_finite()));
}

#[test]
fn sampling_without_a_clip_is_the_rest_pose() {
    let skel = chain();
    let (t, r, s) = skel.sample(None, 0.0);
    for (i, j) in skel.joints.iter().enumerate() {
        assert_eq!(t[i], j.t);
        assert_eq!(r[i], j.r);
        assert_eq!(s[i], j.s);
    }
}

fn translation_clip(joint: usize, times: Vec<f32>, ys: Vec<f32>) -> Clip {
    Clip {
        name: "c".into(),
        duration: *times.last().unwrap(),
        channels: vec![Channel {
            joint,
            path: Path::Translation,
            interp: Interp::Linear,
            times,
            values: ys.into_iter().flat_map(|y| [0.0, y, 0.0]).collect(),
        }],
        root: None,
    }
}

#[test]
fn sampling_clamps_outside_the_clip_range() {
    let skel = chain();
    let clip = translation_clip(1, vec![1.0, 2.0], vec![10.0, 20.0]);

    // Before the first key and after the last, hold the end values rather than
    // extrapolating off into space.
    let (before, _, _) = skel.sample(Some(&clip), -5.0);
    assert_eq!(before[1].y, 10.0);
    let (after, _, _) = skel.sample(Some(&clip), 99.0);
    assert_eq!(after[1].y, 20.0);
}

#[test]
fn sampling_interpolates_between_keys() {
    let skel = chain();
    let clip = translation_clip(1, vec![0.0, 2.0], vec![0.0, 10.0]);
    let (t, _, _) = skel.sample(Some(&clip), 1.0);
    assert!((t[1].y - 5.0).abs() < 1e-5, "got {}", t[1].y);
}

#[test]
fn step_interpolation_holds_the_earlier_key() {
    let skel = chain();
    let mut clip = translation_clip(1, vec![0.0, 2.0], vec![0.0, 10.0]);
    clip.channels[0].interp = Interp::Step;
    let (t, _, _) = skel.sample(Some(&clip), 1.9);
    assert_eq!(t[1].y, 0.0);
}

#[test]
fn a_clip_only_moves_the_joints_it_names() {
    // Joints the clip does not drive keep their rest transform. This is what
    // lets a clip authored for part of a body compose with the rest of it.
    let skel = chain();
    let clip = translation_clip(1, vec![0.0], vec![7.0]);
    let (t, _, _) = skel.sample(Some(&clip), 0.0);
    assert_eq!(t[0], skel.joints[0].t);
    assert_eq!(t[1].y, 7.0);
    assert_eq!(t[2], skel.joints[2].t);
}

/// A channel with no complete key is a HOLE, not a pose.
///
/// The sampler used to substitute a value for one: ZERO for a 3-per-key track and
/// IDENTITY for a rotation. A keyless SCALE channel therefore collapsed its joint -
/// and every vertex bound to it - to a point, a keyless translation snapped the bone
/// onto its parent's origin, and a keyless rotation threw the authored orientation
/// away. There is no value that is right for all three, so none is substituted: the
/// joint keeps the rest transform it would have had if the channel were not there.
///
/// The retarget documents a standing obligation on producers never to emit one. An
/// obligation every producer has to remember is the fix that is wrong.
#[test]
fn a_channel_with_no_keys_leaves_the_joint_at_its_rest_transform() {
    let mut skel = chain();
    skel.joints[1].s = Vec3::new(2.0, 3.0, 4.0);
    skel.joints[1].r = Quat::from_rotation_z(0.5);

    let keyless = |path, times: Vec<f32>, values: Vec<f32>| Clip {
        name: "c".into(),
        duration: 1.0,
        channels: vec![Channel {
            joint: 1,
            path,
            interp: Interp::Linear,
            times,
            values,
        }],
        root: None,
    };
    for path in [Path::Scale, Path::Translation, Path::Rotation] {
        // A rotation key is 4 floats, a T/S key 3. No keys at all, and one float short
        // of a key, are both "no complete key" - a truncated track is a hole too.
        let need = if path == Path::Rotation { 4 } else { 3 };
        for values in [Vec::new(), vec![0.0f32; need - 1]] {
            let times = if values.is_empty() {
                Vec::new()
            } else {
                vec![0.0]
            };
            let clip = keyless(path, times, values);
            let (t, r, s) = skel.sample(Some(&clip), 0.5);
            assert_eq!(
                s[1], skel.joints[1].s,
                "{path:?} collapsed the joint's scale"
            );
            assert_eq!(
                t[1], skel.joints[1].t,
                "{path:?} moved the joint off its rest"
            );
            assert_eq!(
                r[1], skel.joints[1].r,
                "{path:?} lost the joint's rest orientation"
            );
        }
    }
}

#[test]
fn a_channel_naming_a_joint_that_does_not_exist_is_ignored() {
    // Retargeting and hand-edited clips both produce these; an out-of-range
    // joint index must be skipped, not panic on the index.
    let skel = chain();
    let clip = translation_clip(99, vec![0.0], vec![7.0]);
    let (t, _, _) = skel.sample(Some(&clip), 0.0);
    for (i, j) in skel.joints.iter().enumerate() {
        assert_eq!(t[i], j.t);
    }
}

fn vertex_at(pos: [f32; 3]) -> Vertex {
    Vertex::new(pos, [0.0, 1.0, 0.0], [0.0, 0.0])
}

fn model_with(primitives: Vec<Primitive>, skeleton: Option<Skeleton>) -> Model {
    Model {
        primitives,
        skeleton,
        clips: vec![],
    }
}

fn primitive(vertices: Vec<Vertex>, skinned: bool) -> Primitive {
    let indices = (0..vertices.len() as u32).collect();
    Primitive {
        mesh: MeshData { vertices, indices },
        material: String::new(),
        base_color: [1.0; 4],
        metallic: 0.0,
        roughness: 1.0,
        emissive: [0.0; 3],
        texture: None,
        normal_tex: None,
        mr_tex: None,
        emissive_tex: None,
        skinned,
    }
}

#[test]
fn bounds_of_a_model_with_no_geometry_are_zero() {
    // An unseeded fold would report infinities here and poison any arithmetic
    // done on a failed load.
    assert_eq!(model_with(vec![], None).bind_pose_bounds(), [0.0; 6]);
    assert_eq!(MeshData::default().bounds(), [0.0; 6]);
}

#[test]
fn static_geometry_is_measured_where_it_sits() {
    let m = model_with(
        vec![primitive(
            vec![vertex_at([-1.0, 0.0, 2.0]), vertex_at([3.0, 4.0, -5.0])],
            false,
        )],
        None,
    );
    assert_eq!(m.bind_pose_bounds(), [-1.0, 0.0, -5.0, 3.0, 4.0, 2.0]);
}

#[test]
fn skinned_geometry_is_measured_through_its_bind_matrices() {
    // The whole point of bind_pose_bounds: a skinned vertex lives in bind space
    // and only means something once the bind matrix has been applied. Here the
    // bind matrix scales by 1/100, standing in for a centimetre export.
    let skel = Skeleton {
        joints: vec![Joint {
            parent: None,
            inverse_bind: Mat4::from_scale(Vec3::splat(0.01)),
            t: Vec3::ZERO,
            r: Quat::IDENTITY,
            s: Vec3::ONE,
            name: "root".into(),
        }],
    };
    let mut v = vertex_at([0.0, 180.0, 0.0]);
    v.joints = [0; 4];
    v.weights = [1.0, 0.0, 0.0, 0.0];

    let m = model_with(vec![primitive(vec![v], true)], Some(skel));
    let b = m.bind_pose_bounds();
    assert!((b[4] - 1.8).abs() < 1e-4, "expected 1.8, got {}", b[4]);

    // And the raw mesh bounds must still report bind space, so the two are not
    // silently interchangeable.
    assert_eq!(m.primitives[0].mesh.bounds()[4], 180.0);
}

#[test]
fn an_unweighted_skinned_vertex_stays_put_instead_of_collapsing_to_the_origin() {
    // Zero total weight used to fall through to a zero accumulator, dragging the
    // bounds back to the origin and quietly inflating every collider built from
    // them.
    let skel = Skeleton {
        joints: vec![joint("root", None, Vec3::ZERO)],
    };
    let mut v = vertex_at([5.0, 6.0, 7.0]);
    v.weights = [0.0; 4];

    let m = model_with(vec![primitive(vec![v], true)], Some(skel));
    assert_eq!(m.bind_pose_bounds(), [5.0, 6.0, 7.0, 5.0, 6.0, 7.0]);
}

#[test]
fn a_skinned_vertex_naming_a_missing_joint_does_not_panic() {
    let skel = Skeleton {
        joints: vec![joint("root", None, Vec3::ZERO)],
    };
    let mut v = vertex_at([1.0, 2.0, 3.0]);
    v.joints = [7, 0, 0, 0];
    v.weights = [1.0, 0.0, 0.0, 0.0];

    let m = model_with(vec![primitive(vec![v], true)], Some(skel));
    assert!(m.bind_pose_bounds().iter().all(|f| f.is_finite()));
}

#[test]
fn a_skinned_model_with_no_skeleton_falls_back_to_raw_positions() {
    let m = model_with(
        vec![primitive(vec![vertex_at([0.0, 9.0, 0.0])], true)],
        None,
    );
    assert_eq!(m.bind_pose_bounds()[4], 9.0);
}

#[test]
fn bounding_radius_is_never_zero() {
    // A degenerate mesh must still have a testable bound, or the frustum and
    // shadow culls divide by nothing.
    let mut m = MeshData::default();
    m.vertices.push(vertex_at([0.0, 0.0, 0.0]));
    assert!(m.bounding_radius() > 0.0);
    assert!(MeshData::default().bounding_radius() > 0.0);
}

// ---------------------------------------------------------------------------
// The motion root: which bone carries a clip's travel
// ---------------------------------------------------------------------------

/// A UE-style rig: a `Root` at the character's feet, the body hanging off it,
/// and the IK roots sitting beside it at the same origin.
fn rooted_rig() -> Skeleton {
    Skeleton {
        joints: vec![
            joint("Root", None, Vec3::ZERO),
            joint("Pelvis", Some(0), Vec3::new(0.0, 0.9, 0.0)),
            joint("spine_01", Some(1), Vec3::new(0.0, 0.2, 0.0)),
            joint("head", Some(2), Vec3::new(0.0, 0.5, 0.0)),
            joint("ik_foot_root", None, Vec3::ZERO),
            joint("ik_foot_l", Some(4), Vec3::ZERO),
        ],
    }
}

#[test]
fn the_motion_root_is_the_bone_the_body_hangs_off() {
    let skel = rooted_rig();
    assert_eq!(
        skel.motion_root(),
        Some(0),
        "Root carries the body; ik_foot_root does not"
    );
}

/// A hips-rooted rig (Mixamo, most glTF characters) has no bone whose motion is
/// pure travel: the hips' translation is the character's bob and lean. Lifting
/// that out of the pose would flatten the animation, so nothing is lifted.
#[test]
fn a_hips_rooted_rig_has_no_separable_travel() {
    let skel = Skeleton {
        joints: vec![
            joint("Hips", None, Vec3::new(0.0, 0.9, 0.0)),
            joint("Spine", Some(0), Vec3::new(0.0, 0.2, 0.0)),
        ],
    };
    assert_eq!(skel.motion_root(), None);
}

#[test]
fn a_lone_root_bone_is_not_a_motion_root() {
    // Nothing hangs off it, so there is no body for it to move.
    let skel = Skeleton {
        joints: vec![joint("Root", None, Vec3::ZERO)],
    };
    assert_eq!(skel.motion_root(), None);
}

/// The rest pose of an animation-only export is a placeholder: it ships no bind
/// data, so its bones sit wherever the exporter left them - for a root-motion
/// clip, out along the travel itself.
///
/// This is not hypothetical. "The motion root is the bone at the origin" is the
/// obvious rule, and it identifies the bone correctly on every character rig and
/// on exactly none of the clips that have travel to give: the real pack's
/// `Root` rests 1.52 m down its own +Z.
#[test]
fn a_placeholder_rest_pose_does_not_hide_the_motion_root() {
    let mut skel = rooted_rig();
    skel.joints[0].t = Vec3::new(0.0, 0.0, 1.52);
    assert_eq!(skel.motion_root(), Some(0));
}

/// An exporter decorates a bone name with the namespace or armature it came
/// from. That is an export setting, not a different bone.
#[test]
fn a_namespaced_root_is_still_the_root() {
    let mut skel = rooted_rig();
    skel.joints[0].name = "mixamorig:root".into();
    assert_eq!(skel.motion_root(), Some(0));
    skel.joints[0].name = "Armature|Root".into();
    assert_eq!(skel.motion_root(), Some(0));
}

/// A bone that only sounds like the root is not it.
#[test]
fn a_bone_merely_named_like_the_root_is_not_the_root() {
    let mut skel = rooted_rig();
    skel.joints[0].name = "root_motion_helper".into();
    assert_eq!(skel.motion_root(), None);
}

#[test]
fn a_rig_with_no_joints_has_no_motion_root() {
    assert_eq!(Skeleton { joints: vec![] }.motion_root(), None);
}

#[test]
fn a_parent_cycle_does_not_hang_the_motion_root_search() {
    let skel = Skeleton {
        joints: vec![
            joint("Root", None, Vec3::ZERO),
            joint("a", Some(2), Vec3::Y),
            joint("b", Some(1), Vec3::Y),
        ],
    };
    // The cycle is unreachable from Root, so Root carries nothing and is not a
    // motion root. The point is that this ANSWERS.
    assert_eq!(skel.motion_root(), None);
}

/// A stride scales with the body's leg length, and hip height is what stands in
/// for it.
#[test]
fn hip_height_is_where_the_body_hangs_from_the_root() {
    assert!((rooted_rig().hip_height() - 0.9).abs() < 1e-5);
}

/// A rig assembled from modular parts is only as tall as the parts it was given.
/// Measured at the top, a body loaded without its head would report a shorter
/// character and shorten every step it takes; measured at the hip - which every
/// part carries, because every part carries the chain up to the root - it does
/// not.
#[test]
fn hip_height_does_not_depend_on_which_parts_were_assembled() {
    let whole = rooted_rig().hip_height();
    let mut headless = rooted_rig();
    headless.joints.truncate(3); // Root, Pelvis, spine_01: no head
    assert!((headless.hip_height() - whole).abs() < 1e-5);
}

/// A hips-rooted rig has no root bone to hang from, so the topmost body joint is
/// the measurement.
#[test]
fn hip_height_of_a_hips_rooted_rig_is_its_own_root() {
    let skel = Skeleton {
        joints: vec![
            joint("Hips", None, Vec3::new(0.0, 0.95, 0.0)),
            joint("Spine", Some(0), Vec3::new(0.0, 0.2, 0.0)),
        ],
    };
    assert!((skel.hip_height() - 0.95).abs() < 1e-5);
}
