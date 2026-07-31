//! FBX import against real source art.
//!
//! These need licensed pack files that cannot live in the repository, so they
//! read a directory from `AURORA_TEST_FBX_DIR` and skip when it is absent. They
//! are still worth having: every number asserted here was established by hand
//! against Blender, and each one has already caught a real defect.
//!
//! Expected layout inside that directory:
//!   SK_Character_Male_King.fbx        a skinned character
//!   SK_Chr_Head_Male_00.fbx           a modular part
//!   PolygonSyntyCharacter.fbx         the animation packs' reference rig
//!   A_Attack_LightCombo01A_RootMotion_Sword.fbx   a clip with no geometry

use aurora_asset::model::Model;

fn fixture(name: &str) -> Option<Model> {
    let dir = std::env::var("AURORA_TEST_FBX_DIR").ok()?;
    let path = std::path::Path::new(&dir).join(name);
    if !path.is_file() {
        return None;
    }
    Some(Model::load(path.to_str()?).expect("fixture must import"))
}

macro_rules! model {
    ($name:expr) => {
        match fixture($name) {
            Some(m) => m,
            None => return,
        }
    };
}

fn joint_named<'a>(m: &'a Model, name: &str) -> Option<glam::Vec3> {
    let skel = m.skeleton.as_ref()?;
    let i = skel.joints.iter().position(|j| j.name == name)?;
    Some(skel.rest_globals()[i].w_axis.truncate())
}

/// The character stands about 1.84 m tall with its feet on the ground.
///
/// This is the bind-pose self-test: skinning a mesh with its own rest pose has
/// to reproduce that pose. It fails loudly whenever geometry, bind matrices and
/// joint transforms stop agreeing about units or space, which is exactly the
/// class of bug that is otherwise noticed three features later.
#[test]
fn a_skinned_character_bind_poses_to_human_scale() {
    let m = model!("SK_Character_Male_King.fbx");
    let b = m.bind_pose_bounds();

    assert!((b[1] - -0.003).abs() < 0.01, "feet at y {}", b[1]);
    assert!((b[4] - 1.836).abs() < 0.01, "head at y {}", b[4]);
    assert!((b[3] - 1.024).abs() < 0.01, "arm reach {}", b[3]);

    // The raw geometry is in bind space - centimetres for this export - so the
    // two measurements must stay far apart. If they ever converge, someone has
    // started baking skinned geometry and bind_pose_bounds is now a no-op.
    let raw = m.primitives[0].mesh.bounds();
    assert!(raw[4] > 100.0, "raw bounds should be bind space, got {}", raw[4]);
}

/// A part skinned to a subset of the shared skeleton lands where that part
/// belongs on the body, not at the origin.
#[test]
fn a_modular_part_sits_where_it_belongs_on_the_body() {
    let m = model!("SK_Chr_Head_Male_00.fbx");
    let b = m.bind_pose_bounds();
    assert!(b[1] > 1.3, "a head should not reach below y 1.3, got {}", b[1]);
    assert!(b[4] < 1.9, "a head should not reach above y 1.9, got {}", b[4]);

    let skel = m.skeleton.expect("modular part carries a skeleton");
    assert!(skel.joints.len() < 20, "a part carries only its own chain");
    assert!(skel.joints.iter().any(|j| j.name == "head"));
}

/// The rest pose comes from the skin clusters, not from node transforms.
///
/// This file's node transforms have every bone collapsed onto the hip. Read
/// naively it imports 4896 triangles into a four-millimetre blob.
#[test]
fn a_rig_whose_node_transforms_are_not_its_bind_pose_still_imports() {
    let m = model!("PolygonSyntyCharacter.fbx");
    let b = m.bind_pose_bounds();
    assert!(
        b[4] - b[1] > 1.5,
        "character collapsed: y spans {}..{}",
        b[1],
        b[4]
    );
    let hips = joint_named(&m, "Hips").expect("Hips");
    let head = joint_named(&m, "Head").expect("Head");
    assert!(head.y - hips.y > 0.5, "head sits on top of the hip");
}

/// An animation export has no geometry at all. A loader that assumed a mesh
/// would reject an entire animation library.
#[test]
fn a_clip_with_no_geometry_imports_as_animation() {
    let m = model!("A_Attack_LightCombo01A_RootMotion_Sword.fbx");
    assert!(m.primitives.is_empty(), "clip file carries no geometry");
    assert!(m.skeleton.is_some(), "clip file carries a skeleton");

    let clip = m.clips.first().expect("clip file carries a clip");
    // 24 frames at 30fps, matching the source.
    assert!((clip.duration - 0.8).abs() < 0.02, "duration {}", clip.duration);
    assert!(clip.channels.len() > 100, "channels {}", clip.channels.len());

    // Root motion: the attack travels forward over its length.
    let skel = m.skeleton.as_ref().unwrap();
    let hips = skel.joints.iter().position(|j| j.name == "Hips").unwrap();
    let at = |t: f32| {
        let (tr, r, s) = skel.sample(Some(clip), t);
        skel.globals(&tr, &r, &s)[hips].w_axis.truncate()
    };
    let travel = (at(clip.duration * 0.5) - at(0.0)).length();
    assert!(travel > 0.3, "root motion travelled only {travel}");
}

/// Bone-name correspondence between the animation rig and the character rig.
const BONE_MAP: &[(&str, &str)] = &[
    ("Hips", "Pelvis"),
    ("Spine_01", "spine_01"),
    ("Spine_02", "spine_02"),
    ("Spine_03", "spine_03"),
    ("Neck", "neck_01"),
    ("Head", "head"),
    ("Clavicle_L", "clavicle_l"),
    ("Shoulder_L", "UpperArm_L"),
    ("Elbow_L", "lowerarm_l"),
    ("Hand_L", "Hand_L"),
    ("Clavicle_R", "clavicle_r"),
    ("Shoulder_R", "UpperArm_R"),
    ("Elbow_R", "lowerarm_r"),
    ("Hand_R", "Hand_R"),
    ("UpperLeg_L", "Thigh_L"),
    ("LowerLeg_L", "calf_l"),
    ("Ankle_L", "Foot_L"),
    ("Ball_L", "ball_l"),
    ("Toes_L", "toes_l"),
    ("UpperLeg_R", "Thigh_R"),
    ("LowerLeg_R", "calf_r"),
    ("Ankle_R", "Foot_R"),
    ("Ball_R", "ball_r"),
    ("Toes_R", "toes_r"),
];

/// The animation rig and the character rig are the same skeleton under two
/// naming conventions.
///
/// This is the assumption the whole animation plan rests on. If it holds, a clip
/// retargets by renaming bones and nothing else - no rotation correction, no
/// limb scaling. If it ever stops holding, retargeting silently starts producing
/// subtly wrong poses, so it is asserted rather than believed.
#[test]
fn the_animation_rig_and_the_character_rig_share_a_rest_pose() {
    let anim = model!("PolygonSyntyCharacter.fbx");
    let character = model!("SK_Character_Male_King.fbx");

    let mut compared = 0;
    for (from, to) in BONE_MAP {
        let (Some(a), Some(b)) = (joint_named(&anim, from), joint_named(&character, to)) else {
            panic!("bone map names a joint that does not exist: {from} -> {to}");
        };
        assert!(
            (a - b).length() < 0.002,
            "{from} at {a:?} but {to} at {b:?}"
        );
        compared += 1;
    }
    assert_eq!(compared, BONE_MAP.len());
}
