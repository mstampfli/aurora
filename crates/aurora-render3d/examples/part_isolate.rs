//! Render each modular part on its own and report bright outliers.
//!
//! ```text
//! cargo run -p aurora-render3d --example part_isolate -- <fixture-dir> [atlas.png]
//! ```
//!
//! An artifact on an assembled character says nothing about which of a dozen
//! meshes owns it. Drawing each in isolation and counting pixels far brighter
//! than the rest of that part narrows it to one slot in a single pass.

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

const CLEAR: [f32; 4] = [1.0, 0.0, 1.0, 1.0];

fn is_background(p: &[u8]) -> bool {
    p[0] > 200 && p[1] < 60 && p[2] > 200
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let Some(dir) = args.first() else {
        eprintln!("usage: part_isolate <fixture-dir> [atlas.png]");
        std::process::exit(2);
    };
    let Some((device, queue)) = headless_device() else {
        eprintln!("no GPU adapter");
        return;
    };

    let (w, h) = (256u32, 384u32);
    let mut scene = Scene::new(&device, &queue, wgpu::TextureFormat::Rgba8Unorm, w, h, 1);
    scene.set_clear(CLEAR[0], CLEAR[1], CLEAR[2]);
    scene.set_light(Vec3::new(0.3, 0.5, 1.0), Vec3::ONE, 0.5);
    scene.set_camera(Vec3::new(0.0, 0.9, 3.0), Vec3::new(0.0, 0.9, 0.0), 45.0);
    if let Some(atlas) = args.get(1) {
        scene.set_material_texture("ModularFantasyHeroCharacters", atlas);
    }

    let host = scene.load_model(&device, &queue, &format!("{dir}/SK_Character_Male_King.fbx"));
    assert!(host >= 0, "host failed to load");

    println!("{:<16} {:>7} {:>7} {:>8}  brightest pixel", "slot", "pixels", "bright", "bright%");
    for slot in SLOTS {
        let part = scene.load_part(
            &device,
            &queue,
            &format!("{dir}/modular/SK_Chr_{slot}_Male_00.fbx"),
            host,
        );
        if part < 0 {
            println!("{slot:<16} FAILED TO LOAD");
            continue;
        }

        scene.begin();
        scene.draw_skinned(part, host, glam::Mat4::IDENTITY);
        let img = render_offscreen(&mut scene.renderer, &device, &queue, w, h, CLEAR);

        let body: Vec<&[u8]> = img.chunks_exact(4).filter(|p| !is_background(p)).collect();
        if body.is_empty() {
            println!("{slot:<16} nothing drawn");
            continue;
        }
        let lum = |p: &[u8]| (p[0] as u32 + p[1] as u32 + p[2] as u32) / 3;
        let mean: u32 = body.iter().map(|p| lum(p)).sum::<u32>() / body.len() as u32;
        // "Bright" means far above this part's own average, so a naturally pale
        // mesh is not flagged wholesale.
        let cut = (mean + 90).min(250);
        let bright = body.iter().filter(|p| lum(p) > cut).count();
        let peak = body.iter().map(|p| lum(p)).max().unwrap_or(0);
        println!(
            "{slot:<16} {:>7} {:>7} {:>7.2}%  peak={peak} mean={mean}",
            body.len(),
            bright,
            100.0 * bright as f32 / body.len() as f32,
        );

        scene.free_model(part);
    }

    // Assembled: an artifact absent from every part alone can only come from how
    // they meet, so locate the outliers and say where they are and what they are.
    let mut parts = Vec::new();
    for slot in SLOTS {
        let p = scene.load_part(
            &device,
            &queue,
            &format!("{dir}/modular/SK_Chr_{slot}_Male_00.fbx"),
            host,
        );
        if p >= 0 {
            parts.push((*slot, p));
        }
    }
    scene.begin();
    for (_, p) in &parts {
        scene.draw_skinned(*p, host, glam::Mat4::IDENTITY);
    }
    let img = render_offscreen(&mut scene.renderer, &device, &queue, w, h, CLEAR);

    let lum = |p: &[u8]| (p[0] as u32 + p[1] as u32 + p[2] as u32) / 3;
    let body: Vec<&[u8]> = img.chunks_exact(4).filter(|p| !is_background(p)).collect();
    let mean: u32 = body.iter().map(|p| lum(p)).sum::<u32>() / body.len().max(1) as u32;
    let cut = mean + 90;
    println!("\nassembled: {} pixels, mean luminance {mean}, flagging above {cut}", body.len());

    let mut flagged = Vec::new();
    for y in 0..h {
        for x in 0..w {
            let p = &img[((y * w + x) * 4) as usize..][..4];
            if !is_background(p) && lum(p) > cut {
                flagged.push((x, y, p[0], p[1], p[2]));
            }
        }
    }
    println!("flagged {} pixels", flagged.len());
    for (x, y, r, g, b) in flagged.iter().take(12) {
        println!("  ({x:>3},{y:>3}) rgb=({r},{g},{b})");
    }

    // Which part owns them: redraw with one slot withheld and see if they vanish.
    if !flagged.is_empty() {
        for (slot, skip) in &parts {
            scene.begin();
            for (_, p) in &parts {
                if p != skip {
                    scene.draw_skinned(*p, host, glam::Mat4::IDENTITY);
                }
            }
            let without = render_offscreen(&mut scene.renderer, &device, &queue, w, h, CLEAR);
            let still = flagged
                .iter()
                .filter(|(x, y, ..)| {
                    let p = &without[((y * w + x) * 4) as usize..][..4];
                    !is_background(p) && lum(p) > cut
                })
                .count();
            if still < flagged.len() {
                println!(
                    "  withholding {slot:<14} removes {} of {} flagged pixels",
                    flagged.len() - still,
                    flagged.len()
                );
            }
        }
    }
}
