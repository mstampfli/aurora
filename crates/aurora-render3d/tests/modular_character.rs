//! A modular character, rendered.
//!
//! The asset-side tests prove the numbers line up. This proves the pixels do:
//! eleven separately authored parts, rebound onto one skeleton, drawn from a
//! single pose, have to come out as one human-shaped silhouette. A part left
//! unrebound would skin to the wrong joints and smear across the frame, which no
//! amount of correct bookkeeping would have caught.
//!
//! Needs the licensed pack files, so it reads `AURORA_TEST_FBX_DIR` and skips
//! when that is unset. Writes the frame it judged to `AURORA_TEST_FRAME_DIR`
//! when that is set, so a human can look at what the assertions saw.

use aurora_render3d::{headless_device, render_offscreen, Scene, Vec3};

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

/// Magenta. Nothing lit can be mistaken for it, so a pixel either is the
/// character or is not.
const CLEAR: [f32; 4] = [1.0, 0.0, 1.0, 1.0];

fn is_background(p: &[u8]) -> bool {
    p[0] > 200 && p[1] < 60 && p[2] > 200
}

fn write_png(dir: &str, name: &str, rgba: &[u8], w: u32, h: u32) {
    // Written as a PPM and left for the caller to convert; this crate has no
    // image encoder and does not need one to be inspectable.
    let mut out = format!("P6\n{w} {h}\n255\n").into_bytes();
    out.extend(rgba.chunks_exact(4).flat_map(|p| [p[0], p[1], p[2]]));
    let _ = std::fs::create_dir_all(dir);
    let _ = std::fs::write(std::path::Path::new(dir).join(name), out);
}

#[test]
fn eleven_modular_parts_render_as_one_body() {
    let Ok(dir) = std::env::var("AURORA_TEST_FBX_DIR") else {
        return;
    };
    let Some((device, queue)) = headless_device() else {
        eprintln!("no GPU adapter - skipping the modular character render");
        return;
    };

    let (w, h) = (256u32, 384u32);
    let mut scene = Scene::new(&device, &queue, wgpu::TextureFormat::Rgba8Unorm, w, h, 1);
    scene.set_clear(CLEAR[0], CLEAR[1], CLEAR[2]);
    // Lit from in front and slightly above, like an asset review rather than a
    // level: a front-facing chest under an overhead key sits at pure ambient and
    // reads as far darker than the art actually is.
    scene.set_light(Vec3::new(0.3, 0.5, 1.0), Vec3::ONE, 0.5);
    // Front-on, framing a person about 1.8m tall standing at the origin.
    scene.set_camera(Vec3::new(0.0, 0.9, 3.0), Vec3::new(0.0, 0.9, 0.0), 45.0);

    // Synty meshes carry no texture of their own; the pack ships one atlas for
    // the whole cast, named by the material every mesh in it uses.
    if let Ok(atlas) = std::env::var("AURORA_TEST_ATLAS") {
        scene.set_material_texture("ModularFantasyHeroCharacters", &atlas);
    }

    let host = scene.load_model(
        &device,
        &queue,
        &format!("{dir}/SK_Character_Male_King.fbx"),
    );
    assert!(host >= 0, "host failed to load");

    // Control: the detector must see an empty frame as empty, or the real
    // assertion below passes vacuously.
    scene.begin();
    let empty = render_offscreen(&mut scene.renderer, &device, &queue, w, h, CLEAR);
    assert_eq!(
        empty.chunks_exact(4).filter(|p| is_background(p)).count(),
        (w * h) as usize,
        "the background detector does not recognise an empty frame"
    );

    let mut parts = Vec::new();
    for slot in SLOTS {
        let p = scene.load_part(
            &device,
            &queue,
            &format!("{dir}/modular/SK_Chr_{slot}_Male_00.fbx"),
            host,
        );
        assert!(p >= 0, "{slot} failed to load as a part of the host");
        parts.push(p);
    }
    assert_eq!(parts.len(), SLOTS.len());

    scene.begin();
    for part in &parts {
        scene.draw_skinned(*part, host, glam::Mat4::IDENTITY);
    }
    let img = render_offscreen(&mut scene.renderer, &device, &queue, w, h, CLEAR);

    if let Ok(out) = std::env::var("AURORA_TEST_FRAME_DIR") {
        write_png(&out, "modular_character.ppm", &img, w, h);
    }

    let drawn = img.chunks_exact(4).filter(|p| !is_background(p)).count();
    assert!(
        drawn > 2000,
        "only {drawn} pixels of character rendered; the body is missing or tiny"
    );

    // A standing person is taller than wide and roughly centred. Measuring the
    // silhouette's extent catches a part skinned to the wrong joint, which
    // stretches geometry off toward the origin and blows the bounds out.
    let (mut min_x, mut max_x, mut min_y, mut max_y) = (w, 0u32, h, 0u32);
    for y in 0..h {
        for x in 0..w {
            let p = &img[((y * w + x) * 4) as usize..][..4];
            if !is_background(p) {
                min_x = min_x.min(x);
                max_x = max_x.max(x);
                min_y = min_y.min(y);
                max_y = max_y.max(y);
            }
        }
    }
    let (bw, bh) = (max_x - min_x, max_y - min_y);
    assert!(bh > bw, "silhouette is wider ({bw}) than tall ({bh})");
    assert!(
        bh > h / 2,
        "silhouette spans only {bh} of {h} rows; the body is not filling the frame"
    );
    let cx = (min_x + max_x) / 2;
    assert!(
        cx.abs_diff(w / 2) < w / 6,
        "silhouette centred at x={cx}, expected near {}",
        w / 2
    );
}
