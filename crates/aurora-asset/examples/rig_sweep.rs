//! Check a whole directory tree of character files against one reference rig.
//!
//! ```text
//! cargo run -p aurora-asset --example rig_sweep -- <reference.fbx> <dir> [dir ...]
//! ```
//!
//! For every model found, reports whether its bones are a subset of the
//! reference's and whether the ones they share sit in the same place at rest.
//! That is the property a shared animation set depends on, and it is worth
//! establishing over a whole library rather than over the handful of files
//! someone happened to open.

use aurora_asset::model::Model;

/// Rest positions this far apart are the same joint. Generous enough for
/// exporter rounding, tight enough that a genuinely different body fails.
const TOLERANCE: f32 = 0.002;

fn models_under(dir: &std::path::Path, out: &mut Vec<std::path::PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for e in entries.flatten() {
        let p = e.path();
        if p.is_dir() {
            models_under(&p, out);
        } else if p
            .extension()
            .and_then(|s| s.to_str())
            .is_some_and(|s| s.eq_ignore_ascii_case("fbx"))
        {
            out.push(p);
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let Some((reference, dirs)) = args.split_first() else {
        eprintln!("usage: rig_sweep <reference.fbx> <dir> [dir ...]");
        std::process::exit(2);
    };

    let reference_model = Model::load(reference).expect("reference rig must load");
    let reference_skel = reference_model.skeleton.expect("reference rig has no skeleton");
    let reference_rest = reference_skel.rest_globals();
    let reference_pos = |name: &str| {
        reference_skel
            .joints
            .iter()
            .position(|j| j.name.eq_ignore_ascii_case(name))
            .map(|i| reference_rest[i].w_axis.truncate())
    };
    println!("reference: {reference} ({} joints)", reference_skel.joints.len());

    let mut files = Vec::new();
    for d in dirs {
        models_under(std::path::Path::new(d), &mut files);
    }
    files.sort();
    println!("sweeping {} files\n", files.len());

    let (mut conforming, mut no_skeleton, mut failed) = (0usize, 0usize, 0usize);
    let mut unknown_bones: std::collections::BTreeMap<String, usize> = Default::default();
    let mut worst: (f32, String, String) = (0.0, String::new(), String::new());

    for path in &files {
        let name = path.file_name().unwrap().to_string_lossy().into_owned();
        let model = match Model::load(&path.to_string_lossy()) {
            Ok(m) => m,
            Err(e) => {
                println!("  LOAD FAILED  {name}: {e}");
                failed += 1;
                continue;
            }
        };
        let Some(skel) = &model.skeleton else {
            no_skeleton += 1;
            continue;
        };

        let rest = skel.rest_globals();
        let mut ok = true;
        for (i, j) in skel.joints.iter().enumerate() {
            match reference_pos(&j.name) {
                None => {
                    *unknown_bones.entry(j.name.clone()).or_default() += 1;
                    ok = false;
                }
                Some(want) => {
                    let d = (rest[i].w_axis.truncate() - want).length();
                    if d > worst.0 {
                        worst = (d, name.clone(), j.name.clone());
                    }
                    if d > TOLERANCE {
                        println!("  MOVED  {name}: {} is {d:.4} from the reference", j.name);
                        ok = false;
                    }
                }
            }
        }
        if ok {
            conforming += 1;
        } else {
            failed += 1;
        }
    }

    println!();
    println!("conforming     {conforming}");
    println!("non-conforming {failed}");
    println!("no skeleton    {no_skeleton}");
    println!(
        "largest rest difference among shared bones: {:.5} ({} / {})",
        worst.0, worst.1, worst.2
    );
    if !unknown_bones.is_empty() {
        println!("\nbones absent from the reference rig:");
        for (bone, count) in &unknown_bones {
            println!("  {bone:<28} on {count} files");
        }
    }
}
