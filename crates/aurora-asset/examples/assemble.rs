//! Probe modular assembly: union the parts' skeletons, then rebind each part.
//!
//!     cargo run -p aurora-asset --example assemble -- part.fbx [part.fbx ...]
//!
//! Reports the rig the parts add up to and whether every one of them binds onto
//! it. A part that will not bind is the failure this exists to surface, because
//! the alternative is a body with a limb silently skinned to the wrong bones.

use aurora_asset::model::{Model, Skeleton};

fn main() {
    let paths: Vec<String> = std::env::args().skip(1).collect();
    if paths.is_empty() {
        eprintln!("usage: assemble <part.fbx> [part.fbx ...]");
        std::process::exit(2);
    }

    let mut parts = Vec::new();
    for p in &paths {
        match Model::load(p) {
            Ok(m) => parts.push((p.clone(), m)),
            Err(e) => {
                eprintln!("{p}: {e}");
                std::process::exit(1);
            }
        }
    }

    let mut rig = Skeleton { joints: Vec::new() };
    for (path, m) in &parts {
        let Some(s) = &m.skeleton else {
            println!("{:<44} no skeleton, skipped", short(path));
            continue;
        };
        match rig.merge(s, 0.001) {
            Ok(added) => println!(
                "{:<44} {:>3} joints, +{added} new, rig now {}",
                short(path),
                s.joint_count(),
                rig.joint_count()
            ),
            Err(e) => {
                eprintln!("{}: {e}", short(path));
                std::process::exit(1);
            }
        }
    }

    println!("\nassembled rig: {} joints", rig.joint_count());

    let mut failed = 0;
    for (path, m) in &mut parts {
        match m.rebind_skin(&rig, 0.001) {
            Ok(n) => println!("  bind {:<42} {n} joint slots rewritten", short(path)),
            Err(e) => {
                println!("  BIND FAILED {:<36} {e}", short(path));
                failed += 1;
            }
        }
    }

    if failed > 0 {
        eprintln!("\n{failed} part(s) could not bind to the assembled rig");
        std::process::exit(1);
    }
    println!("\nevery part binds to the assembled rig");
}

fn short(p: &str) -> String {
    p.rsplit(['/', '\\']).next().unwrap_or(p).to_string()
}
