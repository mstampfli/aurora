//! Rebinding a part's skinning onto a shared skeleton.
//!
//! Most of these drive the rejection paths. A rebind that silently accepts a
//! mismatched part produces a character that looks right in the bind pose and
//! comes apart at the seams only in certain poses, which is exactly the kind of
//! defect that survives to a build.

use aurora_asset::mesh::{MeshData, Vertex};
use aurora_asset::model::{Joint, Model, Primitive, Skeleton};
use glam::{Mat4, Quat, Vec3};

fn joint(name: &str, parent: Option<usize>, bind: Mat4) -> Joint {
    Joint {
        parent,
        inverse_bind: bind,
        t: Vec3::ZERO,
        r: Quat::IDENTITY,
        s: Vec3::ONE,
        name: name.into(),
    }
}

/// The shared body: four named joints.
fn target() -> Skeleton {
    Skeleton {
        joints: vec![
            joint("Pelvis", None, Mat4::IDENTITY),
            joint(
                "spine_01",
                Some(0),
                Mat4::from_translation(Vec3::new(0.0, -1.0, 0.0)),
            ),
            joint(
                "head",
                Some(1),
                Mat4::from_translation(Vec3::new(0.0, -2.0, 0.0)),
            ),
            joint(
                "Hand_L",
                Some(1),
                Mat4::from_translation(Vec3::new(-1.0, -1.5, 0.0)),
            ),
        ],
    }
}

/// A part carrying only `spine_01` and `head`, in its own order, so a correct
/// rebind must actually renumber rather than happen to line up.
fn part() -> Model {
    let skeleton = Skeleton {
        joints: vec![
            joint(
                "head",
                Some(1),
                Mat4::from_translation(Vec3::new(0.0, -2.0, 0.0)),
            ),
            joint(
                "spine_01",
                None,
                Mat4::from_translation(Vec3::new(0.0, -1.0, 0.0)),
            ),
        ],
    };
    let mut a = Vertex::new([0.0, 2.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0]);
    a.joints = [0, 1, 0, 0];
    a.weights = [0.75, 0.25, 0.0, 0.0];
    let mut b = Vertex::new([0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0]);
    b.joints = [1, 0, 0, 0];
    b.weights = [1.0, 0.0, 0.0, 0.0];

    Model {
        primitives: vec![Primitive {
            mesh: MeshData {
                vertices: vec![a, b],
                indices: vec![0, 1, 0],
            },
            material: String::new(),
            base_color: [1.0; 4],
            metallic: 0.0,
            roughness: 1.0,
            emissive: [0.0; 3],
            texture: None,
            normal_tex: None,
            mr_tex: None,
            emissive_tex: None,
            skinned: true,
        }],
        skeleton: Some(skeleton),
        clips: vec![],
    }
}

#[test]
fn a_part_is_renumbered_into_the_shared_skeletons_order() {
    let mut m = part();
    let rewritten = m.rebind_skin(&target(), 1e-5).expect("part rebinds");

    // Two weighted influences on the first vertex, one on the second.
    assert_eq!(rewritten, 3);

    // "head" was local index 0 and is shared index 2; "spine_01" was 1 and is 1.
    let v = &m.primitives[0].mesh.vertices;
    assert_eq!(v[0].joints[0], 2);
    assert_eq!(v[0].joints[1], 1);
    assert_eq!(v[1].joints[0], 1);

    // Weights are untouched - only the addressing changed.
    assert_eq!(v[0].weights[0], 0.75);
    assert_eq!(v[0].weights[1], 0.25);

    assert_eq!(m.skeleton.as_ref().unwrap().joints.len(), 4);
}

#[test]
fn covering_only_part_of_the_target_is_normal() {
    // The whole point: a head does not carry hand or pelvis weights.
    let mut m = part();
    assert!(m.rebind_skin(&target(), 1e-5).is_ok());
}

#[test]
fn a_weighted_joint_missing_from_the_target_is_rejected_by_name() {
    let mut m = part();
    m.skeleton.as_mut().unwrap().joints[0].name = "tail_01".into();

    let err = m.rebind_skin(&target(), 1e-5).expect_err("must reject");
    assert!(err.contains("tail_01"), "unhelpful error: {err}");
}

#[test]
fn a_part_that_binds_differently_is_rejected_by_name() {
    // Same bone name, different bind pose. One shared palette cannot serve both,
    // so this must fail loudly rather than skin to a slightly wrong body.
    let mut m = part();
    m.skeleton.as_mut().unwrap().joints[0].inverse_bind =
        Mat4::from_translation(Vec3::new(0.0, -2.5, 0.0));

    let err = m.rebind_skin(&target(), 1e-5).expect_err("must reject");
    assert!(err.contains("head"), "unhelpful error: {err}");
}

#[test]
fn bind_drift_within_tolerance_is_accepted() {
    let mut m = part();
    m.skeleton.as_mut().unwrap().joints[0].inverse_bind =
        Mat4::from_translation(Vec3::new(0.0, -2.0 + 1e-4, 0.0));
    assert!(m.rebind_skin(&target(), 1e-3).is_ok());
}

#[test]
fn a_rejected_rebind_leaves_the_model_untouched() {
    // Validation happens before any vertex is written, so a part that fails
    // cannot leave a half-renumbered mesh behind.
    let mut m = part();
    m.skeleton.as_mut().unwrap().joints[0].name = "tail_01".into();
    let before: Vec<[u32; 4]> = m.primitives[0]
        .mesh
        .vertices
        .iter()
        .map(|v| v.joints)
        .collect();

    assert!(m.rebind_skin(&target(), 1e-5).is_err());

    let after: Vec<[u32; 4]> = m.primitives[0]
        .mesh
        .vertices
        .iter()
        .map(|v| v.joints)
        .collect();
    assert_eq!(before, after);
    assert_eq!(m.skeleton.as_ref().unwrap().joints.len(), 2);
}

#[test]
fn an_unweighted_slot_naming_a_missing_joint_is_harmless() {
    // Importers leave junk in slots that carry no weight. It must not veto a
    // part, and it must not be left pointing past the end of the palette.
    let mut m = part();
    m.primitives[0].mesh.vertices[1].joints = [1, 9, 9, 9];

    assert!(m.rebind_skin(&target(), 1e-5).is_ok());
    for v in &m.primitives[0].mesh.vertices {
        for k in 0..4 {
            assert!((v.joints[k] as usize) < 4, "slot {k} out of palette range");
        }
    }
}

#[test]
fn an_unweighted_ancestor_joint_is_not_held_to_the_targets_bind_pose() {
    // A part's skeleton carries the chain above what it deforms - a head exports
    // the spine and pelvis purely so its neck has somewhere to hang from. Those
    // joints have no skin cluster and therefore only a placeholder identity bind
    // matrix. Comparing that placeholder against the target's real bind matrix
    // compares a measurement to a blank and rejects a part doing nothing wrong.
    //
    // Caught by assembling eleven real Synty parts: every one failed on Pelvis,
    // by exactly the target's own bind translation.
    let mut t = target();
    t.joints[0].inverse_bind = Mat4::from_translation(Vec3::new(-86.6, -13.1, 0.0));

    let mut m = part();
    m.skeleton
        .as_mut()
        .unwrap()
        .joints
        .push(joint("Pelvis", None, Mat4::IDENTITY));

    let rewritten = m
        .rebind_skin(&t, 1e-5)
        .expect("an unweighted ancestor must not veto the part");
    assert_eq!(rewritten, 3);
}

#[test]
fn a_vertex_weighted_past_the_end_of_its_own_skeleton_is_rejected() {
    let mut m = part();
    m.primitives[0].mesh.vertices[0].joints = [42, 1, 0, 0];
    let err = m.rebind_skin(&target(), 1e-5).expect_err("must reject");
    assert!(err.contains("42"), "unhelpful error: {err}");
}

#[test]
fn a_model_with_no_skeleton_cannot_be_rebound() {
    let mut m = part();
    m.skeleton = None;
    assert!(m.rebind_skin(&target(), 1e-5).is_err());
}

#[test]
fn static_geometry_in_a_part_is_left_alone() {
    let mut m = part();
    m.primitives[0].skinned = false;
    let before: Vec<[u32; 4]> = m.primitives[0]
        .mesh
        .vertices
        .iter()
        .map(|v| v.joints)
        .collect();

    let rewritten = m
        .rebind_skin(&target(), 1e-5)
        .expect("still adopts the skeleton");
    assert_eq!(rewritten, 0);

    let after: Vec<[u32; 4]> = m.primitives[0]
        .mesh
        .vertices
        .iter()
        .map(|v| v.joints)
        .collect();
    assert_eq!(before, after);
}

#[test]
fn rebinding_a_part_puts_it_where_the_shared_body_puts_it() {
    // The behavioural claim, not just the bookkeeping one: after rebinding, the
    // part measures the same through the shared skeleton as it did through its
    // own. If the renumbering were wrong this would move.
    let mut m = part();
    let before = m.bind_pose_bounds();
    m.rebind_skin(&target(), 1e-5).expect("rebinds");
    let after = m.bind_pose_bounds();

    for i in 0..6 {
        assert!(
            (before[i] - after[i]).abs() < 1e-5,
            "bounds moved on axis {i}: {} -> {}",
            before[i],
            after[i]
        );
    }
}
