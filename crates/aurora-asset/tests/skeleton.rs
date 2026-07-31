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
        joints: vec![
            joint("a", Some(1), Vec3::Y),
            joint("b", Some(0), Vec3::Y),
        ],
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
    let m = model_with(vec![primitive(vec![vertex_at([0.0, 9.0, 0.0])], true)], None);
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
