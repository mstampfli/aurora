//! Root motion, end to end, against the real source art.
//!
//! The synthetic tests beside `AnimPlayer` prove the arithmetic. This proves the
//! number a game actually gets: a licensed pack's `_RootMotion_` clip, imported,
//! retargeted onto a character rig that shares none of its bone names, and
//! played frame by frame, has to hand back the distance the animator authored.
//!
//! Needs the licensed pack files, so it reads `AURORA_TEST_FBX_DIR` and skips
//! when that is unset.

use aurora_render3d::model::Model;
use aurora_render3d::{AnimPlayer, Vec3};

const SWING: &str = "A_Attack_LightCombo01A_RootMotion_Sword.fbx";
/// The step the swing was authored to take, measured off the source file.
const AUTHORED: f32 = 1.12;

fn fixture(name: &str) -> Option<Model> {
    // Through `aurora_fixtures`, which PANICS when the pack directory is unset
    // rather than handing back None. That None is why all six tests in this file
    // had never run: each one did `else { return }` and reported ok.
    let path = aurora_fixtures::file(name)?;
    Some(Model::load(path.to_str()?).expect("fixture must import"))
}

/// Play the whole clip at 60 Hz and add up what each update reported.
fn walk_the_clip(model: &Model, clip: usize, looping: bool, frames: usize) -> Vec3 {
    let mut p = AnimPlayer::default();
    p.play(clip, looping, 1.0, 0.0);
    let mut total = Vec3::ZERO;
    for _ in 0..frames {
        p.advance(model, 1.0 / 60.0);
        total += p.root_delta();
    }
    total
}

/// The sum of the per-frame deltas is the distance in the file.
///
/// Integrating a delta is what a game does with it, so integrating it is what
/// has to come out right - a per-frame number that looks plausible but does not
/// add up is a lunge that lands short every time.
#[test]
fn a_real_attack_reports_the_ground_it_was_authored_to_cover() {
    let Some(m) = fixture(SWING) else { return };
    let total = walk_the_clip(&m, 0, false, 120);
    assert!(
        (total.z - AUTHORED).abs() < 0.02,
        "the swing should step {AUTHORED} m forward, got {total:?}"
    );
    assert!(total.x.abs() < 0.02, "and only forward: {total:?}");
}

/// The same clip on a character from another pack, through the retarget the game
/// uses. This is the path the bug was on: the pack authors the travel on a
/// `Root` bone the character rig does not have under that name, and matching by
/// name found nothing, so every authored distance was dropped and every attack
/// played on the spot.
#[test]
fn the_distance_survives_the_retarget_onto_a_character() {
    let Some(dir) = aurora_fixtures::dir() else {
        return;
    };
    let Some(mut character) = fixture("SK_Character_Male_King.fbx") else {
        return;
    };
    let rest = match Model::load_skeleton(&format!("{}/PolygonSyntyCharacter.fbx", dir.display())) {
        Ok(s) => s,
        Err(_) => return,
    };
    let before = character.clips.len();
    character
        .add_clips_from(
            &format!("{}/{SWING}", dir.display()),
            &rest,
            &[("Hips", "Pelvis")],
            &["Pelvis"],
        )
        .expect("clip library loads");

    let total = walk_the_clip(&character, before, false, 120);
    // Scaled by the two rigs' proportions, which for two bodies from the same
    // pack is 1.0 - so the authored distance, not a fraction of it.
    assert!(
        (total.z - AUTHORED).abs() < 0.05,
        "the retargeted swing should still step {AUTHORED} m, got {total:?}"
    );
}

const ROLL: &str = "A_DodgeRoll_F_RootMotion_Sword.fbx";
const IDLE: &str = "A_Idle_Standing_Masc.fbx";
const DT: f32 = 1.0 / 60.0;

/// A character carrying both clips the game switches between, through the retarget
/// the game uses: `(model, roll clip, idle clip)`.
fn roll_and_idle() -> Option<(Model, usize, usize)> {
    let dir = aurora_fixtures::dir()?;
    let mut character = fixture("SK_Character_Male_King.fbx")?;
    let rest =
        Model::load_skeleton(&format!("{}/PolygonSyntyCharacter.fbx", dir.display())).ok()?;
    let roll = character.clips.len();
    character
        .add_clips_from(
            &format!("{}/{ROLL}", dir.display()),
            &rest,
            &[("Hips", "Pelvis")],
            &["Pelvis"],
        )
        .expect("the roll loads");
    let idle = character.clips.len();
    character
        .add_clips_from(
            &format!("{}/{IDLE}", dir.display()),
            &rest,
            &[("Hips", "Pelvis")],
            &["Pelvis"],
        )
        .expect("the idle loads");
    assert!(character.clips.len() > idle, "both clips must arrive");
    Some((character, roll, idle))
}

/// Play `clip` to the end, returning the ground it covered.
fn play_out(p: &mut AnimPlayer, m: &Model, clip: usize, fade: f32) -> Vec3 {
    p.restart(clip, false, 1.0, fade);
    let frames = (m.clips[clip].duration / DT).ceil() as usize + 2;
    let mut total = Vec3::ZERO;
    for _ in 0..frames {
        p.advance(m, DT);
        total += p.root_delta();
    }
    total
}

/// A roll that has PLAYED OUT, faded into an idle authored in place, on the real
/// art. The fabricated distance the synthetic pin catches, in metres of level.
#[test]
fn a_finished_roll_fabricates_no_distance_while_it_fades() {
    let Some((m, roll, idle)) = roll_and_idle() else {
        return;
    };
    let mut p = AnimPlayer::default();
    let travelled = play_out(&mut p, &m, roll, 0.0);
    assert!(
        travelled.z > 0.5,
        "the roll must actually travel: {travelled:?}"
    );

    // The idle covers no ground, and the roll has none left, so every frame of the
    // fade is worth exactly nothing. Before the fix this paid out 0.1734 m.
    p.play(idle, true, 1.0, 0.2);
    let mut added = Vec3::ZERO;
    for _ in 0..60 {
        p.advance(&m, DT);
        added += p.root_delta();
    }
    assert_eq!(
        added,
        Vec3::ZERO,
        "fading a played-out roll into an in-place idle moved the character"
    );
}

/// A player standing on the idle, which is where the game is between actions.
fn on_the_idle(m: &Model, idle: usize) -> AnimPlayer {
    let mut p = AnimPlayer::default();
    p.play(idle, true, 1.0, 0.0);
    for _ in 0..30 {
        p.advance(m, DT);
    }
    p
}

/// One dodge as the game performs it: into the roll over `fade_in`, played out,
/// then back to the idle over `fade_out` and held there. The ground it covered.
fn dodge(
    p: &mut AnimPlayer,
    m: &Model,
    roll: usize,
    idle: usize,
    fade_in: f32,
    fade_out: f32,
) -> f32 {
    let mut total = play_out(p, m, roll, fade_in).z;
    p.play(idle, true, 1.0, fade_out);
    for _ in 0..30 {
        p.advance(m, DT);
        total += p.root_delta().z;
    }
    total
}

/// Five rolls in a row cover five rolls' worth of ground.
///
/// The error compounded once per action, so this is where it showed: the game's own
/// loop is exactly this, dodge after dodge, and every metre above five rolls is
/// distance no animator authored. Entered instantly, so the answer is the authored
/// distance itself and not a number this test derived from the code under test.
#[test]
fn five_rolls_cover_five_rolls_worth_of_ground() {
    let Some((m, roll, idle)) = roll_and_idle() else {
        return;
    };
    let authored = play_out(&mut AnimPlayer::default(), &m, roll, 0.0).z;

    let mut p = on_the_idle(&m, idle);
    let mut total = 0.0;
    for _ in 0..5 {
        total += dodge(&mut p, &m, roll, idle, 0.0, 0.2);
    }
    assert!(
        (total - 5.0 * authored).abs() < 1e-3,
        "five rolls of {authored} m should cover {} m, got {total} m",
        5.0 * authored
    );
}

/// The same at the fade lengths the game actually passes (`scene.aur` uses 0.05
/// into an action and 0.12 back to the idle), because the error is clip-shape
/// dependent rather than a constant and one fade proves one fade.
///
/// A fade INTO the roll legitimately discounts its opening travel - the character is
/// not yet in the roll, and travel is weighted exactly as the pose is - so the
/// reference is the same dodge with the fade OUT removed. Anything the fade out adds
/// on top of that is fabricated, which is the whole defect.
#[test]
fn the_games_own_fade_lengths_fabricate_nothing_either() {
    let Some((m, roll, idle)) = roll_and_idle() else {
        return;
    };
    let reference = dodge(&mut on_the_idle(&m, idle), &m, roll, idle, 0.05, 0.0);

    let mut p = on_the_idle(&m, idle);
    let mut total = 0.0;
    for _ in 0..3 {
        total += dodge(&mut p, &m, roll, idle, 0.05, 0.12);
    }
    assert!(
        (total - 3.0 * reference).abs() < 1e-3,
        "three dodges should cover {} m, got {total} m",
        3.0 * reference
    );
}

/// A locomotion loop is authored in place on purpose: its ground speed belongs
/// to the game, and travel invented for it would double every step.
#[test]
fn a_real_locomotion_loop_moves_the_character_nowhere() {
    let Some(m) = fixture("A_Walk_F_Masc.fbx") else {
        return;
    };
    let total = walk_the_clip(&m, 0, true, 300);
    assert!(
        total.length() < 0.01,
        "a walk cycle travels nowhere, got {total:?}"
    );
}
