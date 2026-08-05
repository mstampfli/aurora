//! Dump what an importer actually produced, so an import can be checked against
//! the source file rather than eyeballed on screen.
//!
//! ```text
//! cargo run -p aurora-asset --example fbx_dump -- <file.fbx> [more.fbx ...]
//! ```
//!
//! Prints geometry counts, the bone tree with rest positions, and every clip
//! with its duration and channel coverage.

use aurora_asset::model::Model;

fn main() {
    let files: Vec<String> = std::env::args().skip(1).collect();
    if files.is_empty() {
        eprintln!("usage: fbx_dump <file> [file ...]");
        std::process::exit(2);
    }

    for path in &files {
        println!("{}", "=".repeat(78));
        println!("{path}");
        println!("{}", "=".repeat(78));
        let model = match Model::load(path) {
            Ok(m) => m,
            Err(e) => {
                println!("  LOAD FAILED: {e}");
                continue;
            }
        };

        let verts: usize = model.primitives.iter().map(|p| p.mesh.vertices.len()).sum();
        let tris: usize = model
            .primitives
            .iter()
            .map(|p| p.mesh.indices.len() / 3)
            .sum();
        let skinned = model.primitives.iter().filter(|p| p.skinned).count();
        let textured = model.primitives.iter().filter(|p| p.texture.is_some()).count();
        println!(
            "  primitives={}  verts={verts}  tris={tris}  skinned={skinned}  textured={textured}",
            model.primitives.len()
        );
        for p in &model.primitives {
            let b = p.mesh.bounds();
            println!(
                "    prim: {:>6} tris  mat={:<28} base_color=[{:.2} {:.2} {:.2}]  bounds y {:.3}..{:.3}  tex={}",
                p.mesh.indices.len() / 3,
                if p.material.is_empty() { "<none>" } else { &p.material },
                p.base_color[0],
                p.base_color[1],
                p.base_color[2],
                b[1],
                b[4],
                p.texture
                    .as_ref()
                    .map(|(_, w, h)| format!("{w}x{h}"))
                    .unwrap_or_else(|| "none".into()),
            );
        }

        match &model.skeleton {
            None => println!("  skeleton: NONE"),
            Some(sk) => {
                println!("  skeleton: {} joints", sk.joints.len());
                // The inverse-bind maps mesh space into bone space. Its scale
                // therefore reveals whether the geometry shares the skeleton's
                // units or is being rescaled implicitly during skinning.
                for j in sk.joints.iter().filter(|j| j.inverse_bind != glam::Mat4::IDENTITY).take(3)
                {
                    let (s, _, t) = j.inverse_bind.to_scale_rotation_translation();
                    println!(
                        "    inverse_bind[{}]: scale=({:.4},{:.4},{:.4}) translation=({:.3},{:.3},{:.3})",
                        j.name, s.x, s.y, s.z, t.x, t.y, t.z
                    );
                }
                // Rest-pose model-space position of each joint, so bone lengths
                // and overall scale can be compared against the source rig.
                let mut world = vec![glam::Mat4::IDENTITY; sk.joints.len()];
                for (i, j) in sk.joints.iter().enumerate() {
                    let local = glam::Mat4::from_scale_rotation_translation(j.s, j.r, j.t);
                    world[i] = match j.parent {
                        Some(p) => world[p] * local,
                        None => local,
                    };
                }
                for (i, j) in sk.joints.iter().enumerate() {
                    let depth = {
                        let mut d = 0;
                        let mut p = j.parent;
                        while let Some(k) = p {
                            d += 1;
                            p = sk.joints[k].parent;
                        }
                        d
                    };
                    let pos = world[i].w_axis;
                    println!(
                        "    {:>3} {}{:<24} pos=({:7.3},{:7.3},{:7.3})",
                        i,
                        "  ".repeat(depth),
                        j.name,
                        pos.x,
                        pos.y,
                        pos.z
                    );
                }
            }
        }

        // Bind-pose self-test.
        //
        // Skinning the mesh with the rest pose must reproduce the rest pose:
        // that is what a bind matrix means. If geometry, bind matrices and joint
        // transforms disagree - different units, a missed geometry transform -
        // this is where it shows, and it shows as a number rather than as a
        // mangled silhouette noticed three features later.
        if !model.primitives.is_empty() {
            let b = model.bind_pose_bounds();
            println!(
                "  bind-pose bounds: x {:.3}..{:.3}  y {:.3}..{:.3}  z {:.3}..{:.3}",
                b[0], b[3], b[1], b[4], b[2], b[5]
            );
        }

        // Where the bones actually go once the first clip is applied. A rest
        // pose can be a placeholder - clip-only exports often leave the static
        // transforms at identity and carry the real offsets in the curves - so
        // the posed skeleton, not the rest one, says whether a file is sane.
        if let (Some(sk), Some(clip)) = (&model.skeleton, model.clips.first()) {
            for &time in &[0.0f32, clip.duration * 0.5] {
                let (t, r, s) = sk.sample(Some(clip), time);
                let g = sk.globals(&t, &r, &s);
                let named = |want: &str| sk.index_of(want).map(|i| g[i].w_axis);
                let show = |want: &str| {
                    named(want)
                        .map(|p| format!("{want}=({:.2},{:.2},{:.2})", p.x, p.y, p.z))
                        .unwrap_or_default()
                };
                let lo = g.iter().map(|m| m.w_axis.y).fold(f32::MAX, f32::min);
                let hi = g.iter().map(|m| m.w_axis.y).fold(f32::MIN, f32::max);
                println!(
                    "  posed t={time:.2}s  y {:.2}..{:.2}  {} {} {} {}",
                    lo,
                    hi,
                    show("Hips"),
                    show("Pelvis"),
                    show("Head"),
                    show("head"),
                );
            }
        }

        println!("  clips: {}", model.clips.len());
        for c in &model.clips {
            let joints: std::collections::HashSet<usize> =
                c.channels.iter().map(|ch| ch.joint).collect();
            let keys: usize = c.channels.iter().map(|ch| ch.times.len()).sum();
            println!(
                "    {:<40} dur={:.3}s  channels={}  joints={}  keys={}",
                c.name,
                c.duration,
                c.channels.len(),
                joints.len(),
                keys
            );
        }
    }
}
