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

/// The eleven slots that make up a whole modular body.
const SLOTS: &[&str] = &[
    "Head",
    "Torso",
    "Hips",
    "ArmUpperLeft",
    "ArmUpperRight",
    "ArmLowerLeft",
    "ArmLowerRight",
    "HandLeft",
    "HandRight",
    "LegLeft",
    "LegRight",
];

/// A character assembled from eleven separately authored parts rebinds onto one
/// skeleton and forms a single whole body.
///
/// This is the modular character system end to end. Each part exports its own
/// private joint list, so this passing means the renumbering is right; and the
/// assembled bounds being a plausible human means the parts landed on the body
/// rather than at the origin or inside each other.
#[test]
fn eleven_modular_parts_assemble_onto_one_skeleton() {
    let donor = model!("SK_Character_Male_King.fbx");
    let skeleton = donor.skeleton.expect("donor carries the shared skeleton");

    let mut lo = [f32::MAX; 3];
    let mut hi = [f32::MIN; 3];
    let mut assembled = 0;

    for slot in SLOTS {
        let Some(mut part) = fixture(&format!("modular/SK_Chr_{slot}_Male_00.fbx")) else {
            return;
        };
        let joints_before = part.skeleton.as_ref().map(|s| s.joints.len()).unwrap_or(0);
        assert!(
            joints_before < skeleton.joints.len(),
            "{slot} should carry only its own chain, got {joints_before}"
        );

        part.rebind_skin(&skeleton, 1e-4)
            .unwrap_or_else(|e| panic!("{slot} failed to rebind: {e}"));

        // Every influence must now address the shared palette.
        for prim in &part.primitives {
            for v in &prim.mesh.vertices {
                for k in 0..4 {
                    assert!(
                        (v.joints[k] as usize) < skeleton.joints.len(),
                        "{slot} vertex addresses joint {} of {}",
                        v.joints[k],
                        skeleton.joints.len()
                    );
                }
            }
        }

        let b = part.bind_pose_bounds();
        for a in 0..3 {
            lo[a] = lo[a].min(b[a]);
            hi[a] = hi[a].max(b[3 + a]);
        }
        assembled += 1;
    }

    assert_eq!(assembled, SLOTS.len());

    // One body: feet near the ground, head near 1.8m, arms spread about a metre
    // either side. Parts left unrebound would pile up at the origin and collapse
    // the vertical span.
    assert!(lo[1] > -0.05 && lo[1] < 0.05, "feet at y {}", lo[1]);
    assert!(hi[1] > 1.6 && hi[1] < 2.0, "head at y {}", hi[1]);
    assert!(hi[0] > 0.5, "arms reach only to x {}", hi[0]);
    assert!(lo[0] < -0.5, "arms reach only to x {}", lo[0]);
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

/// The full correspondence between Synty's animation rig and its character rig.
///
/// Fingers are generated rather than listed: the animation rig suffixes the right
/// hand with `_1` where the character rig suffixes with `_r`, and spelling out
/// thirty entries by hand is thirty chances to transpose a digit.
fn synty_bone_map() -> Vec<(String, String)> {
    let mut m: Vec<(String, String)> = [
        ("Hips", "Pelvis"),
        ("Neck", "neck_01"),
        ("Head", "head"),
        ("Shoulder_L", "UpperArm_L"),
        ("Elbow_L", "lowerarm_l"),
        ("Shoulder_R", "UpperArm_R"),
        ("Elbow_R", "lowerarm_r"),
        ("UpperLeg_L", "Thigh_L"),
        ("LowerLeg_L", "calf_l"),
        ("Ankle_L", "Foot_L"),
        ("UpperLeg_R", "Thigh_R"),
        ("LowerLeg_R", "calf_r"),
        ("Ankle_R", "Foot_R"),
    ]
    .iter()
    .map(|(a, b)| (a.to_string(), b.to_string()))
    .collect();

    for (anim_suffix, char_suffix) in [("", "_l"), ("_1", "_r")] {
        for digit in ["01", "02", "03"] {
            m.push((
                format!("Thumb_{digit}{anim_suffix}"),
                format!("thumb_{digit}{char_suffix}"),
            ));
        }
        for digit in ["01", "02", "03", "04"] {
            m.push((
                format!("IndexFinger_{digit}{anim_suffix}"),
                format!("indexFinger_{digit}{char_suffix}"),
            ));
            m.push((
                format!("Finger_{digit}{anim_suffix}"),
                format!("finger_{digit}{char_suffix}"),
            ));
        }
    }
    m
}

/// A real sword clip drives a real character.
///
/// The end of the animation pipeline: a clip authored against one rig, loaded
/// from a file with no geometry, addressed to a character built from a different
/// pack. If the map or the renumbering were wrong this poses a person into a
/// knot, so the assertions are about anatomy rather than about counts.
#[test]
#[ignore = "needs rest-relative rotation retargeting; a plain name remap discards the \
            target's bind orientation and collapses the spine. See the decisions log."]
fn a_sword_clip_retargets_onto_a_character() {
    let mut character = model!("SK_Character_Male_King.fbx");
    let dir = std::env::var("AURORA_TEST_FBX_DIR").unwrap();

    let owned = synty_bone_map();
    let map: Vec<(&str, &str)> = owned.iter().map(|(a, b)| (a.as_str(), b.as_str())).collect();

    let before = character.clips.len();
    let added = character
        .add_clips_from(
            &format!("{dir}/A_Attack_LightCombo01A_RootMotion_Sword.fbx"),
            &map,
            &["Pelvis"],
        )
        .expect("clip library loads");
    assert_eq!(added, 1, "expected one clip from the file");
    assert_eq!(character.clips.len(), before + added);

    let clip = character.clips.last().unwrap();
    let skel = character.skeleton.as_ref().unwrap();

    // Most of the source rig's joints should have found a home. A map that only
    // half works still animates, badly, so a low count has to fail.
    let driven: std::collections::HashSet<usize> = clip.channels.iter().map(|c| c.joint).collect();
    assert!(
        driven.len() >= 40,
        "only {} joints are driven; the bone map is incomplete",
        driven.len()
    );

    {
        use aurora_asset::model::Path as P;
        let n = |p: P| clip.channels.iter().filter(|c| c.path == p).count();
        println!(
            "retargeted clip: rot={} trans={} scale={}  joints={}",
            n(P::Rotation),
            n(P::Translation),
            n(P::Scale),
            driven.len()
        );
        let rest = skel.rest_globals();
        let name_of = |i: usize| skel.joints[i].name.as_str();
        for b in ["Pelvis", "spine_01", "neck_01", "head"] {
            let i = skel.joints.iter().position(|j| j.name == b).unwrap();
            println!(
                "  {b}: rest_y={:.3} local_t=({:.3},{:.3},{:.3}) parent={:?} driven={}",
                rest[i].w_axis.y,
                skel.joints[i].t.x,
                skel.joints[i].t.y,
                skel.joints[i].t.z,
                skel.joints[i].parent.map(name_of),
                driven.contains(&i)
            );
        }
    }

    let index = |n: &str| skel.joints.iter().position(|j| j.name == n).unwrap();
    let (pelvis, head, hand_r, foot_l) = (
        index("Pelvis"),
        index("head"),
        index("Hand_R"),
        index("Foot_L"),
    );

    let mut moved = 0;
    let mut prev: Option<glam::Vec3> = None;
    for step in 0..=8 {
        let t = clip.duration * step as f32 / 8.0;
        let (tr, r, s) = skel.sample(Some(clip), t);
        let g = skel.globals(&tr, &r, &s);
        let at = |i: usize| g[i].w_axis.truncate();

        // Anatomy holds at every instant of the swing, with real distance
        // between the joints.
        //
        // The magnitudes are the point. An earlier version asserted only that
        // the head was ABOVE the hip, and passed on a completely collapsed
        // skeleton where every joint sat within a millimetre of the hip - 0.819
        // is greater than 0.818. Ordering alone proves nothing about a body.
        assert!(
            at(head).y - at(pelvis).y > 0.5,
            "at t={t:.2} the head is only {:.3} above the hip; the skeleton has collapsed",
            at(head).y - at(pelvis).y
        );
        assert!(
            at(pelvis).y - at(foot_l).y > 0.5,
            "at t={t:.2} the hip is only {:.3} above the foot; the skeleton has collapsed",
            at(pelvis).y - at(foot_l).y
        );
        assert!(
            at(pelvis).y > 0.4 && at(pelvis).y < 1.4,
            "at t={t:.2} the hip is at y {}",
            at(pelvis).y
        );

        // And the sword hand actually swings.
        if let Some(p) = prev {
            if (at(hand_r) - p).length() > 0.05 {
                moved += 1;
            }
        }
        prev = Some(at(hand_r));
    }
    assert!(
        moved >= 3,
        "the sword hand barely moved across the clip ({moved} of 8 steps)"
    );
}

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
