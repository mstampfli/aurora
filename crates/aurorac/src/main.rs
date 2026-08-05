//! `aurorac` — the Aurora compiler driver.
//!
//! Phase A surface: `aurorac lex <file>` tokenizes a source file and prints the
//! token stream (or any diagnostics). More subcommands (`parse`, `check`) land
//! as those phases come online.
//!
//! **Place in the graph.** The top of the graph: it depends on the front end, the back end and the runtime, and nothing depends on it.
//!
//! **Never.** Never contains language semantics - it wires stages together, resolves dependencies and reports.

use std::process::ExitCode;

use aurora_lexer::lex;
use aurora_span::SourceFile;

fn main() -> ExitCode {
    // Run the whole compiler on a large stack so deeply-nested source (handled by
    // the recursive parser and every later recursive pass: typeck, checks,
    // codegen) yields a diagnostic instead of an uncatchable stack-overflow abort.
    // macOS requires the window event loop (NSApplication) to own the OS MAIN thread - it panics if
    // created off it, and there is no `any_thread` escape hatch. So on macOS we run the program (and
    // the compiler that JIT-executes it) ON the main thread, giving that thread the big stack via a
    // linker flag (see build.rs) instead of a worker. Other platforms keep the worker thread.
    #[cfg(target_os = "macos")]
    {
        return run_cli();
    }
    #[cfg(not(target_os = "macos"))]
    {
        std::thread::Builder::new()
            .stack_size(256 * 1024 * 1024)
            .spawn(run_cli)
            .expect("spawn compiler thread")
            .join()
            .unwrap_or(ExitCode::FAILURE)
    }
}

fn run_cli() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("lex") => match args.get(1) {
            Some(path) => cmd_lex(path),
            None => {
                eprintln!("usage: aurorac lex <file>");
                ExitCode::from(2)
            }
        },
        Some("parse") => match args.get(1) {
            Some(path) => cmd_parse(path),
            None => {
                eprintln!("usage: aurorac parse <file>");
                ExitCode::from(2)
            }
        },
        Some("check") => match args.get(1) {
            Some(path) => cmd_check(path),
            None => {
                eprintln!("usage: aurorac check <file>");
                ExitCode::from(2)
            }
        },
        Some("new") => match args.get(1) {
            Some(name) => cmd_new(name),
            None => {
                eprintln!("usage: aurorac new <name>");
                ExitCode::from(2)
            }
        },
        Some("asset") => match (args.get(1).map(String::as_str), args.get(2)) {
            (Some("info"), Some(path)) => cmd_asset_info(path),
            (Some("check"), Some(reference)) => cmd_asset_check(reference, &args[3..]),
            (Some("import"), Some(_)) => cmd_asset_import(&args[2..]),
            _ => {
                eprintln!("usage: aurorac asset info <model>");
                eprintln!("       aurorac asset check <reference-rig> <dir>...");
                eprintln!("       aurorac asset import <model-or-dir>...");
                ExitCode::from(2)
            }
        },
        Some("run") => {
            // First positional arg may be the file, or omitted to use the
            // manifest; everything after it belongs to the PROGRAM, not to us.
            let explicit = args
                .get(1)
                .filter(|a| !a.starts_with('-'))
                .map(String::as_str);
            let rest_start = if explicit.is_some() { 2 } else { 1 };
            match resolve_entry(explicit) {
                Ok(path) => cmd_run(&path, &args[rest_start..]),
                Err(e) => {
                    eprintln!("{e}");
                    ExitCode::from(2)
                }
            }
        }
        Some("jit") => match args.get(1) {
            Some(path) => cmd_jit(path, &args[2..]),
            None => {
                eprintln!("usage: aurorac jit <file> <function> [int args...]");
                ExitCode::from(2)
            }
        },
        Some("native") => match args.get(1) {
            Some(path) => cmd_native(path, &args[2..]),
            None => {
                eprintln!("usage: aurorac native <file>");
                ExitCode::from(2)
            }
        },
        Some("build") => {
            // First positional arg may be a file, or omitted to use the manifest.
            let explicit = args
                .get(1)
                .filter(|a| !a.starts_with('-'))
                .map(String::as_str);
            let rest_start = if explicit.is_some() { 2 } else { 1 };
            match resolve_entry(explicit) {
                Ok(path) => cmd_build(&path, &args[rest_start..]),
                Err(e) => {
                    eprintln!("{e}");
                    ExitCode::from(2)
                }
            }
        }
        Some("render") => match args.get(1) {
            Some(out) => cmd_render(out),
            None => {
                eprintln!("usage: aurorac render <out.ppm>");
                ExitCode::from(2)
            }
        },
        Some("wgsl") => match args.get(1) {
            Some(path) => cmd_wgsl(path),
            None => {
                eprintln!("usage: aurorac wgsl <file>");
                ExitCode::from(2)
            }
        },
        Some("gpu") => match args.get(1) {
            Some(path) => cmd_gpu(path, &args[2..]),
            None => {
                eprintln!("usage: aurorac gpu <file> [-o <out.ppm>]");
                ExitCode::from(2)
            }
        },
        Some("debug") => match args.get(1) {
            Some(path) => cmd_debug(path, &args[2..]),
            None => {
                eprintln!("usage: aurorac debug <file> [--break <line>]...");
                ExitCode::from(2)
            }
        },
        Some("window") => match aurora_window::demo() {
            Ok(()) => ExitCode::SUCCESS,
            Err(e) => {
                eprintln!("window error: {e}");
                ExitCode::FAILURE
            }
        },
        Some("profile") => match args.get(1) {
            Some(path) => cmd_profile(path),
            None => {
                eprintln!("usage: aurorac profile <file>");
                ExitCode::from(2)
            }
        },
        Some("watch") => match args.get(1) {
            Some(path) => cmd_watch(path),
            None => {
                eprintln!("usage: aurorac watch <file>");
                ExitCode::from(2)
            }
        },
        Some("sound") => {
            let sr = 44_100;
            println!("playing a demo melody on the default audio device...");
            match aurora_audio::play(&aurora_audio::demo_melody(sr), sr) {
                Ok(()) => ExitCode::SUCCESS,
                Err(e) => {
                    eprintln!("audio error: {e}");
                    ExitCode::FAILURE
                }
            }
        }
        Some("--help") | Some("-h") | None => {
            println!("aurorac — the Aurora compiler\n");
            println!("usage:");
            println!(
                "  aurorac new <name>      scaffold a new Aurora project (aurora.toml + src/)"
            );
            println!("  aurorac lex <file>      tokenize a source file");
            println!("  aurorac parse <file>    parse a source file to an AST");
            println!("  aurorac check <file>    parse and run static checks");
            println!("  aurorac run <file> [args...]  check, compile `main` to native code & run");
            println!("                          (args after the file go to the PROGRAM: sys_arg)");
            println!(
                "  aurorac native <file>   compile `main` to native code & run (no interpreter)"
            );
            println!("  aurorac build <file> [-o <out>] compile to a standalone native executable");
            println!("  aurorac jit <file> <fn> [args]  compile a fn to native code & run");
            println!("  aurorac render <out.ppm>        render a demo scene (CPU rasterizer)");
            println!("  aurorac wgsl <file>             lower @vertex/@fragment fns to WGSL");
            println!(
                "  aurorac gpu <file> [-o out.ppm]  run an Aurora @fragment shader on the GPU"
            );
            println!(
                "  aurorac window                  open a live real-time window (interactive demo)"
            );
            println!(
                "  aurorac debug <file> [--break L] native debugger (breakpoints, step, locals)"
            );
            println!(
                "  aurorac sound                   play a demo melody (synthesis + audio output)"
            );
            println!(
                "  aurorac profile <file>          run with the native profiler (per-fn time)"
            );
            println!("  aurorac watch <file>            re-run on file change (hot reload)");
            ExitCode::SUCCESS
        }
        Some(other) => {
            eprintln!("unknown subcommand `{other}` (try `aurorac --help`)");
            ExitCode::from(2)
        }
    }
}

/// Install the argument vector the RUNNING PROGRAM sees, so `sys_argc`/`sys_arg`
/// report the same thing whichever way it was compiled: argv[0] is the program
/// as invoked - the source file under `aurorac run`, the executable itself for
/// `aurorac build` output - and argv[1..] are its own arguments. Without this
/// the JIT-run program would read `aurorac`'s command line instead of its own.
///
/// A leading `--` is dropped, so `aurorac run game.aur -- --host 45123` and
/// `aurorac run game.aur --host 45123` pass the same vector.
fn set_program_args(path: &str, extra: &[String]) {
    let extra = match extra.first() {
        Some(first) if first == "--" => &extra[1..],
        _ => extra,
    };
    let mut argv = Vec::with_capacity(1 + extra.len());
    argv.push(path.to_string());
    argv.extend(extra.iter().cloned());
    aurora_runtime::set_program_args(argv);
}

/// Resolve which source file to compile: an explicit path if given, otherwise
/// the `entry` of an `aurora.toml` manifest in the current directory.
fn resolve_entry(explicit: Option<&str>) -> Result<String, String> {
    if let Some(p) = explicit {
        return Ok(p.to_string());
    }
    let manifest = std::fs::read_to_string("aurora.toml").map_err(|_| {
        "no source file given and no `aurora.toml` in the current directory \
         (try `aurorac run <file>` or `aurorac new <name>`)"
            .to_string()
    })?;
    match manifest_value(&manifest, "entry") {
        Some(entry) => Ok(entry),
        None => Err("`aurora.toml` is missing an `entry = \"...\"` key".to_string()),
    }
}

/// Parse the `[dependencies]` table of a manifest into `(name, spec)` pairs.
/// Each line is `name = "spec"` where spec is a path or `git:<url>`.
fn manifest_deps(toml: &str) -> Vec<(String, String)> {
    let mut deps = Vec::new();
    let mut in_deps = false;
    for line in toml.lines() {
        let line = line.trim();
        if line.starts_with('[') {
            in_deps = line == "[dependencies]";
            continue;
        }
        if !in_deps || line.is_empty() || line.starts_with('#') {
            continue;
        }
        if let Some((name, rest)) = line.split_once('=') {
            let spec = rest.trim().trim_matches('"').to_string();
            deps.push((name.trim().to_string(), spec));
        }
    }
    deps
}

/// Resolve every dependency (transitively) from `./aurora.toml`, wrapping each
/// in a `mod <name> { .. }` so they namespace cleanly, and writing an
/// `aurora.lock` of the resolved set. Each dependency's own dependencies are
/// emitted first, so a dep can reference them as `<dep>::item`.
///
/// Returns the sources, plus what each module DECLARED it depends on.
///
/// The second half is what makes a module boundary real. Flattening puts every module in one
/// scope, so after it `map::room_at` and a local `room_at` are both just mangled names and the
/// checker cannot tell a declared dependency from a reach into a module nobody said this one
/// knew about. The manifests know; this carries them to where the check happens.
fn collect_deps() -> (
    String,
    std::collections::HashMap<String, std::collections::HashSet<String>>,
) {
    use std::collections::HashSet;
    use std::path::Path;
    let mut visited = HashSet::new();
    let mut out = String::new();
    let mut lock: Vec<(String, String)> = Vec::new();
    let mut declared = std::collections::HashMap::new();
    add_deps(
        Path::new("."),
        &mut visited,
        &mut out,
        &mut lock,
        &mut declared,
    );
    if !lock.is_empty() {
        let mut body = String::from("# Auto-generated by aurorac. Resolved dependencies.\n");
        for (name, spec) in &lock {
            body.push_str(&format!("{name} = \"{spec}\"\n"));
        }
        let _ = std::fs::write("aurora.lock", body);
    }
    (out, declared)
}

/// Diagnostics for any module reaching into one it never declared.
///
/// Only names that ARE modules are judged: a path head like `String::` or an enum from the
/// prelude is not a dependency and never appears in a manifest.
fn undeclared_module_uses(
    module: &aurora_parser::ast::Module,
    declared: &std::collections::HashMap<String, std::collections::HashSet<String>>,
) -> Vec<aurora_diag::Diagnostic> {
    use aurora_diag::Diagnostic;
    let mut out = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for (from, to, span) in &module.cross_refs {
        // A nested module inherits its root's manifest: `a::b` is declared by `a`.
        let root = from.split("::").next().unwrap_or(from);
        if root == to || !declared.contains_key(to) {
            continue;
        }
        let allowed = declared.get(root);
        if allowed.map(|d| d.contains(to)).unwrap_or(false) {
            continue;
        }
        if !seen.insert((root.to_string(), to.clone())) {
            continue;
        }
        out.push(
            Diagnostic::error(format!(
                "module `{root}` uses `{to}` without depending on it"
            ))
            .with_code("E0330")
            .primary(
                *span,
                format!("add `{to}` to the [dependencies] of modules/{root}/aurora.toml"),
            ),
        );
    }
    out
}

/// Recursively resolve the `[dependencies]` of the manifest in `base`, appending
/// each dependency's module-wrapped source to `out` (transitive deps first).
fn add_deps(
    base: &std::path::Path,
    visited: &mut std::collections::HashSet<String>,
    out: &mut String,
    lock: &mut Vec<(String, String)>,
    declared: &mut std::collections::HashMap<String, std::collections::HashSet<String>>,
) {
    let Ok(manifest) = std::fs::read_to_string(base.join("aurora.toml")) else {
        return;
    };
    for (name, spec) in manifest_deps(&manifest) {
        if !visited.insert(name.clone()) {
            continue; // already resolved (dedup across the graph)
        }
        match locate_dep(&name, &spec, base) {
            Ok((dir, src)) => {
                // What THIS dependency declares, before its own are resolved.
                if let Ok(dep_manifest) = std::fs::read_to_string(dir.join("aurora.toml")) {
                    declared.insert(
                        name.clone(),
                        manifest_deps(&dep_manifest)
                            .into_iter()
                            .map(|(n, _)| n)
                            .collect(),
                    );
                }
                // Resolve this dependency's own dependencies first.
                add_deps(&dir, visited, out, lock, declared);
                out.push_str(&format!("\nmod {name} {{\n{src}\n}}\n"));
                lock.push((name, spec));
            }
            Err(e) => eprintln!("warning: skipping dependency `{name}`: {e}"),
        }
    }
}

/// Locate a dependency relative to `base`: returns its directory and library
/// source. `git:<url>` clones into `target/aurora-deps/<name>` once; otherwise
/// `spec` is a path relative to `base`. The dep manifest's `lib`/`entry` names
/// its library file.
fn locate_dep(
    name: &str,
    spec: &str,
    base: &std::path::Path,
) -> Result<(std::path::PathBuf, String), String> {
    let dir = if let Some(url) = spec.strip_prefix("git:") {
        let dest = std::path::PathBuf::from("target")
            .join("aurora-deps")
            .join(name);
        if !dest.exists() {
            let _ = std::fs::create_dir_all(dest.parent().unwrap());
            let status = std::process::Command::new("git")
                .args(["clone", "--depth", "1", url])
                .arg(&dest)
                .status()
                .map_err(|e| format!("git not available: {e}"))?;
            if !status.success() {
                return Err(format!("git clone failed for {url}"));
            }
        }
        dest
    } else {
        base.join(spec)
    };

    let manifest = std::fs::read_to_string(dir.join("aurora.toml"))
        .map_err(|e| format!("no aurora.toml in `{}`: {e}", dir.display()))?;
    let lib = manifest_value(&manifest, "lib")
        .or_else(|| manifest_value(&manifest, "entry"))
        .ok_or("dependency manifest has no `lib`/`entry`")?;
    let lib_path = dir.join(&lib);
    let src =
        std::fs::read_to_string(&lib_path).map_err(|e| format!("cannot read `{lib}`: {e}"))?;
    // A dependency's library may itself be split across files with `mod NAME;`.
    // Loading them here means they land inside the `mod <dep> { .. }` wrapper the
    // caller adds, so they stay namespaced under the dependency.
    let (src, diags) = aurora_parser::load_file_modules(&src, &lib_path);
    if let Some(d) = diags.iter().find(|d| d.is_error()) {
        return Err(format!("in `{lib}`: {}", d.message));
    }
    Ok((dir, src))
}

/// Read a top-level `key = "value"` string from a minimal TOML manifest.
fn manifest_value(toml: &str, key: &str) -> Option<String> {
    for line in toml.lines() {
        let line = line.trim();
        if let Some(rest) = line.strip_prefix(key) {
            if let Some(eq) = rest.trim_start().strip_prefix('=') {
                return Some(eq.trim().trim_matches('"').to_string());
            }
        }
    }
    None
}

/// Read `path` and resolve its file-based `mod NAME;` declarations by loading
/// `NAME.aur` (see `aurora_parser::load_file_modules`). Returns `None` after
/// reporting, so an unresolvable module is a hard error instead of a module that
/// silently contributes nothing.
///
/// The loader only appends, so byte offsets in `path`'s own text are unchanged
/// and the prelude/dependency sources every caller concatenates afterwards keep
/// lining up with the spans reported here.
/// Report the functions that failed to compile to native code, and say whether
/// the command must refuse.
///
/// A body that fails to lower is replaced with a stub returning 0. Left
/// unreported that is the worst possible failure mode: the program builds and
/// runs, and the broken function just quietly evaluates to nothing. So every
/// path that compiles or executes a program routes through here, not just
/// `main` (only `main` used to be checked when running, which is how a missing
/// language feature could hide inside a helper for a long time).
///
/// `verb` completes "refusing to ...".
fn report_stub_failures(failed: &std::collections::HashMap<String, String>, verb: &str) -> bool {
    if failed.is_empty() {
        return false;
    }
    let mut names: Vec<&String> = failed.keys().collect();
    names.sort();
    eprintln!(
        "error: {} function(s) failed to compile to native code:",
        failed.len()
    );
    for n in names {
        eprintln!("  - {n}: {}", failed[n]);
    }
    eprintln!("refusing to {verb}: a stubbed function silently does nothing and returns 0.");
    true
}

/// Say why there is no runnable `main`, and mean it.
///
/// Two entirely different situations wore the same sentence, "`main` did not
/// compile to native code (codegen gap)":
///
///   - the program HAS a `main` and the backend could not lower it, in which case
///     the reason was sitting in `compile_error` and was never printed; or
///   - the file has no `main` at all, because it is a module other programs
///     import, and there was never a gap to report.
///
/// The second is what a library module run by mistake looks like, and blaming the
/// compiler for it sends whoever reads the message looking for a missing language
/// feature. Both answers were already in hand. This asks for them.
fn report_no_main(jit: &aurora_codegen::Jit, path: &str) {
    match jit.compile_error("main") {
        Some(why) => eprintln!("error: `main` did not compile to native code: {why}"),
        None => eprintln!(
            "error: `{path}` has no `main` function, so there is nothing to run.\n\
             note: a file without `main` is a module - import it with `mod` from a \
             program that has one."
        ),
    }
}

fn read_program(path: &str) -> Option<String> {
    let src = match std::fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("error: cannot read `{path}`: {e}");
            return None;
        }
    };
    let (expanded, diags) = aurora_parser::load_file_modules(&src, std::path::Path::new(path));
    if !diags.is_empty() {
        let file = SourceFile::new(path, expanded.clone());
        for d in &diags {
            eprintln!("{}", d.render(&file));
        }
        if diags.iter().any(|d| d.is_error()) {
            return None;
        }
    }
    Some(expanded)
}

/// Scaffold a new project directory with a manifest and a hello-world program.
/// Report what an importer actually produced from a model file.
///
/// Art arrives wrong far more often than code does, and the failures are quiet:
/// a rig whose bind pose is not where its node transforms say, a clip file with
/// no geometry, a mesh whose vertices are in centimetres. Reading the numbers
/// out of the importer answers those without writing a program or opening a DCC.
/// Bake source art into Aurora's runtime format.
///
/// A source model is an INTERCHANGE file - FBX, glTF, OBJ - and reading one
/// means walking a node graph and rebuilding buffers. That happens on every
/// load, in every run. Poly Souls' bailey is 105 distinct files and parsing them
/// is what standing the room up costs: 2.2 GB of peak memory against 69 MiB of
/// mesh and 88 MiB of texture actually uploaded.
///
/// A bake is written once, beside the source, and after that a load is a read.
///
/// Directories are walked, so baking a whole pack is one command. A file that
/// fails is reported and the rest still bake: one bad export in a library of
/// hundreds should cost that file, not the pack.
fn cmd_asset_import(paths: &[String]) -> ExitCode {
    fn is_source(p: &std::path::Path) -> bool {
        matches!(
            p.extension()
                .and_then(|e| e.to_str())
                .map(str::to_ascii_lowercase)
                .as_deref(),
            Some("fbx") | Some("gltf") | Some("glb") | Some("obj")
        )
    }
    fn gather(p: &std::path::Path, out: &mut Vec<std::path::PathBuf>) {
        if p.is_dir() {
            let Ok(rd) = std::fs::read_dir(p) else { return };
            let mut entries: Vec<_> = rd.filter_map(|e| e.ok()).map(|e| e.path()).collect();
            entries.sort();
            for e in entries {
                gather(&e, out);
            }
        } else if is_source(p) {
            out.push(p.to_path_buf());
        }
    }

    let mut sources = Vec::new();
    for p in paths {
        gather(std::path::Path::new(p), &mut sources);
    }
    if sources.is_empty() {
        eprintln!("error: nothing to import - no .fbx, .gltf, .glb or .obj found");
        return ExitCode::FAILURE;
    }

    let mut baked = 0usize;
    let mut skipped = 0usize;
    let mut failed = 0usize;
    let mut src_bytes = 0u64;
    let mut out_bytes = 0u64;
    for src in &sources {
        let s = src.to_string_lossy().to_string();
        let dst = aurora_asset::bake::baked_path(&s);
        // Up to date already: baking is cheap but not free, and a pack is
        // thousands of files.
        if aurora_asset::bake::usable(&s, &dst) {
            skipped += 1;
            continue;
        }
        // `parse`, not `load`: this is the thing that MAKES the bake, so it must
        // read the source even when a stale bake is sitting beside it.
        let model = match aurora_asset::model::Model::parse(&s) {
            Ok(m) => m,
            Err(e) => {
                eprintln!("  FAILED {s}: {e}");
                failed += 1;
                continue;
            }
        };
        let bytes = aurora_asset::bake::write(&model);
        if let Err(e) = std::fs::write(&dst, &bytes) {
            eprintln!("  FAILED {}: {e}", dst.display());
            failed += 1;
            continue;
        }
        src_bytes += std::fs::metadata(src).map(|m| m.len()).unwrap_or(0);
        out_bytes += bytes.len() as u64;
        baked += 1;
    }

    println!(
        "baked {baked}, up to date {skipped}, failed {failed} ({:.1} MiB source -> {:.1} MiB baked)",
        src_bytes as f64 / (1 << 20) as f64,
        out_bytes as f64 / (1 << 20) as f64
    );
    if failed > 0 {
        return ExitCode::FAILURE;
    }
    ExitCode::SUCCESS
}

fn cmd_asset_info(path: &str) -> ExitCode {
    let model = match aurora_asset::model::Model::load(path) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::FAILURE;
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
    println!("{path}");
    println!(
        "  primitives {}  verts {verts}  tris {tris}  skinned {skinned}  textured {textured}",
        model.primitives.len()
    );

    let materials: std::collections::BTreeSet<&str> = model
        .primitives
        .iter()
        .map(|p| p.material.as_str())
        .filter(|m| !m.is_empty())
        .collect();
    if !materials.is_empty() {
        println!("  materials  {}", materials.into_iter().collect::<Vec<_>>().join(", "));
    }

    match &model.skeleton {
        None => println!("  skeleton   none"),
        Some(skel) => {
            let rest = skel.rest_globals();
            let lo = rest.iter().map(|m| m.w_axis.y).fold(f32::MAX, f32::min);
            let hi = rest.iter().map(|m| m.w_axis.y).fold(f32::MIN, f32::max);
            println!(
                "  skeleton   {} joints, rest spans y {lo:.3}..{hi:.3}",
                skel.joints.len()
            );
        }
    }

    if !model.primitives.is_empty() {
        // Through the bind matrices: a skinned mesh's raw vertices are in the
        // file's own bind space, which for a centimetre export is a hundred
        // times the size the model actually appears.
        let b = model.bind_pose_bounds();
        println!(
            "  bounds     x {:.3}..{:.3}  y {:.3}..{:.3}  z {:.3}..{:.3}",
            b[0], b[3], b[1], b[4], b[2], b[5]
        );
    }

    println!("  clips      {}", model.clips.len());
    for c in &model.clips {
        let joints: std::collections::BTreeSet<usize> =
            c.channels.iter().map(|ch| ch.joint).collect();
        println!(
            "    {:<44} {:.3}s  {} channels over {} joints",
            c.name,
            c.duration,
            c.channels.len(),
            joints.len()
        );
    }
    ExitCode::SUCCESS
}

/// Check a library of models against one reference rig.
///
/// A shared animation set only works if every character agrees with the rig the
/// clips were authored on. That is a property of a whole directory, not of the
/// one file someone happened to open, and it is cheap enough to check on every
/// asset drop.
fn cmd_asset_check(reference: &str, dirs: &[String]) -> ExitCode {
    // Rest positions this far apart are the same joint: loose enough for an
    // exporter's rounding, tight enough that a different body fails.
    const TOLERANCE: f32 = 0.002;

    if dirs.is_empty() {
        eprintln!("usage: aurorac asset check <reference-rig> <dir>...");
        return ExitCode::from(2);
    }
    let reference_model = match aurora_asset::model::Model::load(reference) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("error: reference rig: {e}");
            return ExitCode::FAILURE;
        }
    };
    let Some(reference_skel) = reference_model.skeleton else {
        eprintln!("error: reference rig `{reference}` has no skeleton");
        return ExitCode::FAILURE;
    };
    let reference_rest = reference_skel.rest_globals();

    fn models_under(dir: &std::path::Path, out: &mut Vec<std::path::PathBuf>) {
        let Ok(entries) = std::fs::read_dir(dir) else {
            return;
        };
        for e in entries.flatten() {
            let p = e.path();
            if p.is_dir() {
                models_under(&p, out);
            } else if p.extension().and_then(|s| s.to_str()).is_some_and(|s| {
                matches!(s.to_ascii_lowercase().as_str(), "fbx" | "gltf" | "glb")
            }) {
                out.push(p);
            }
        }
    }

    let mut files = Vec::new();
    for d in dirs {
        models_under(std::path::Path::new(d), &mut files);
    }
    files.sort();
    println!(
        "checking {} model(s) against {reference} ({} joints)",
        files.len(),
        reference_skel.joints.len()
    );

    let (mut ok, mut bad) = (0usize, 0usize);
    for path in &files {
        let name = path.file_name().unwrap_or_default().to_string_lossy();
        let model = match aurora_asset::model::Model::load(&path.to_string_lossy()) {
            Ok(m) => m,
            Err(e) => {
                println!("  FAILED  {name}: {e}");
                bad += 1;
                continue;
            }
        };
        let Some(skel) = &model.skeleton else { continue };

        let rest = skel.rest_globals();
        let mut conforms = true;
        for (i, joint) in skel.joints.iter().enumerate() {
            let found = reference_skel.index_of(&joint.name);
            match found {
                None => {
                    println!("  EXTRA   {name}: {} is not on the reference rig", joint.name);
                    conforms = false;
                }
                Some(r) => {
                    let d = (rest[i].w_axis.truncate() - reference_rest[r].w_axis.truncate())
                        .length();
                    if d > TOLERANCE {
                        println!("  MOVED   {name}: {} sits {d:.4} away", joint.name);
                        conforms = false;
                    }
                }
            }
        }
        if conforms {
            ok += 1;
        } else {
            bad += 1;
        }
    }

    println!("conforming {ok}, non-conforming {bad}");
    if bad > 0 {
        // A non-conforming library is a real failure: the shared animation set
        // will not drive it. Reporting it as success would make this checkable
        // in a pipeline and useless in one.
        return ExitCode::FAILURE;
    }
    ExitCode::SUCCESS
}

fn cmd_new(name: &str) -> ExitCode {
    use std::path::Path;
    let root = Path::new(name);
    if root.exists() {
        eprintln!("error: `{name}` already exists");
        return ExitCode::FAILURE;
    }
    let src_dir = root.join("src");
    if let Err(e) = std::fs::create_dir_all(&src_dir) {
        eprintln!("error: cannot create `{}`: {e}", src_dir.display());
        return ExitCode::FAILURE;
    }
    let manifest = format!("name = \"{name}\"\nversion = \"0.1.0\"\nentry = \"src/main.aur\"\n");
    let main_src = "// A new Aurora project. Build a native binary with `aurorac build`,\n\
        // or run it directly with `aurorac run`.\n\n\
        fn main() {\n    println(\"Hello from Aurora!\")\n}\n";
    let ok = std::fs::write(root.join("aurora.toml"), manifest).is_ok()
        && std::fs::write(src_dir.join("main.aur"), main_src).is_ok();
    if !ok {
        eprintln!("error: failed to write project files");
        return ExitCode::FAILURE;
    }
    println!("created project `{name}`");
    println!("  cd {name} && aurorac run");
    ExitCode::SUCCESS
}

fn cmd_lex(path: &str) -> ExitCode {
    let src = match std::fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("error: cannot read `{path}`: {e}");
            return ExitCode::FAILURE;
        }
    };
    let file = SourceFile::new(path, src);
    let result = lex(&file.src);

    for token in &result.tokens {
        let lc = file.line_col(token.span.lo);
        println!("{:>4}:{:<3} {:?}", lc.line, lc.col, token.kind);
    }

    if !result.diagnostics.is_empty() {
        eprintln!();
        for d in &result.diagnostics {
            eprintln!("{}", d.render(&file));
        }
        eprintln!("{} error(s)", result.diagnostics.len());
        return ExitCode::FAILURE;
    }

    ExitCode::SUCCESS
}

fn cmd_wgsl(path: &str) -> ExitCode {
    let Some(src) = read_program(path) else {
        return ExitCode::FAILURE;
    };
    let file = SourceFile::new(path, src);
    let (module, diags) = aurora_parser::parse_str(&file.src);
    if diags.iter().any(|d| d.is_error()) {
        for d in &diags {
            eprintln!("{}", d.render(&file));
        }
        return ExitCode::FAILURE;
    }
    print!("{}", aurora_shader::lower_module(&module));
    ExitCode::SUCCESS
}

/// Run an Aurora `@fragment` shader on the real GPU and save the result as PPM.
///
/// Lowers the shader to WGSL (`aurora-shader`), pairs it with a fullscreen-
/// triangle vertex stage, executes it headless via `aurora-gpu` (wgpu), and
/// writes the read-back pixels to an image.
fn cmd_gpu(path: &str, rest: &[String]) -> ExitCode {
    let mut out: Option<String> = None;
    let mut i = 0;
    while i < rest.len() {
        match rest[i].as_str() {
            "-o" | "--output" => {
                if let Some(v) = rest.get(i + 1) {
                    out = Some(v.clone());
                    i += 2;
                    continue;
                }
                eprintln!("usage: aurorac gpu <file> [-o <out.ppm>]");
                return ExitCode::from(2);
            }
            other => {
                eprintln!("gpu: unexpected argument `{other}`");
                return ExitCode::from(2);
            }
        }
    }

    let Some(src) = read_program(path) else {
        return ExitCode::FAILURE;
    };
    let file = SourceFile::new(path, src);
    let (module, diags) = aurora_parser::parse_str(&file.src);
    if diags.iter().any(|d| d.is_error()) {
        for d in &diags {
            eprintln!("{}", d.render(&file));
        }
        return ExitCode::FAILURE;
    }

    let frag = match aurora_shader::fragment_entries(&module).into_iter().next() {
        Some(name) => name,
        None => {
            eprintln!("gpu: no `@fragment` function found in `{path}`");
            return ExitCode::FAILURE;
        }
    };
    let fs_wgsl = aurora_shader::lower_module(&module);
    // Fullscreen triangle, so the fragment shader covers every pixel.
    let vs =
        "@vertex fn vs_main(@builtin(vertex_index) idx: u32) -> @builtin(position) vec4<f32> {\n\
        var p = array<vec2<f32>, 3>(vec2(-1.0, -3.0), vec2(-1.0, 1.0), vec2(3.0, 1.0));\n\
        return vec4<f32>(p[idx], 0.0, 1.0);\n}\n";
    let wgsl = format!("{vs}\n{fs_wgsl}");

    let gpu = match aurora_gpu::Gpu::new() {
        Some(g) => g,
        None => {
            eprintln!("gpu: no GPU adapter available");
            return ExitCode::FAILURE;
        }
    };
    let (w, h) = (256u32, 256u32);
    let pixels = match gpu.render_rgba_entries(&wgsl, w, h, "vs_main", &frag) {
        Ok(px) => px,
        Err(e) => {
            eprintln!("gpu error: {e}");
            return ExitCode::FAILURE;
        }
    };

    // Encode RGBA8 pixels as a binary PPM (P6, dropping alpha).
    let out_path = out.unwrap_or_else(|| {
        let stem = std::path::Path::new(path)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("shader");
        format!("{stem}.ppm")
    });
    let mut ppm = format!("P6\n{w} {h}\n255\n").into_bytes();
    for px in pixels.chunks_exact(4) {
        ppm.extend_from_slice(&px[..3]);
    }
    match std::fs::write(&out_path, &ppm) {
        Ok(()) => {
            println!(
                "ran `{frag}` on {} → {out_path} ({w}x{h})",
                gpu.adapter_name()
            );
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("gpu: cannot write `{out_path}`: {e}");
            ExitCode::FAILURE
        }
    }
}

fn cmd_render(out: &str) -> ExitCode {
    use aurora_gfx::{Color, Framebuffer};
    let mut fb = Framebuffer::new(256, 256);
    fb.clear(Color::rgb(18, 18, 28));
    // A Gouraud-shaded triangle (per-vertex RGB).
    fb.triangle(
        [[128.0, 20.0], [20.0, 230.0], [236.0, 230.0]],
        [
            Color::rgb(255, 40, 40),
            Color::rgb(40, 255, 40),
            Color::rgb(40, 40, 255),
        ],
    );
    // A white outline.
    fb.line([128.0, 20.0], [20.0, 230.0], Color::WHITE);
    fb.line([20.0, 230.0], [236.0, 230.0], Color::WHITE);
    fb.line([236.0, 230.0], [128.0, 20.0], Color::WHITE);

    match std::fs::write(out, fb.to_ppm()) {
        Ok(()) => {
            println!("wrote {}x{} image to {out}", fb.width(), fb.height());
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("error: cannot write `{out}`: {e}");
            ExitCode::FAILURE
        }
    }
}

fn cmd_native(path: &str, args: &[String]) -> ExitCode {
    set_program_args(path, args);
    let Some(src) = read_program(path) else {
        return ExitCode::FAILURE;
    };
    let file = SourceFile::new(path, aurora_std::with_std(&src));
    let (module, mut diags) = aurora_parser::parse_str(&file.src);
    diags.extend(aurora_check::check(&module));
    diags.extend(aurora_typeck::check_types(&module));
    if diags.iter().any(|d| d.is_error()) {
        for d in &diags {
            eprintln!("{}", d.render(&file));
        }
        return ExitCode::FAILURE;
    }

    // Verify every function actually compiled to native code (not stubbed).
    match aurora_codegen::build(&module) {
        Ok(jit) => {
            if report_stub_failures(jit.failures(), "run") {
                return ExitCode::FAILURE;
            }
            if !jit.compiled("main") {
                report_no_main(&jit, path);
                return ExitCode::FAILURE;
            }
            match jit.call_i64("main", &[]) {
                Ok(_) => {
                    use std::io::Write;
                    let _ = std::io::stdout().flush();
                    ExitCode::SUCCESS
                }
                Err(e) => {
                    eprintln!("native error: {e}");
                    ExitCode::FAILURE
                }
            }
        }
        Err(e) => {
            eprintln!("native error: {e}");
            ExitCode::FAILURE
        }
    }
}

/// Deletes a file when dropped, so a build leaves no object behind on any exit
/// path including an early error return.
struct TempFile(std::path::PathBuf);

impl Drop for TempFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

/// A cross-process lock guarding the link-and-copy step of `aurorac build`.
///
/// Held from before `cargo build -p aurora-exe` until after the produced binary
/// has been copied to its destination, because cargo writes a single shared
/// `target/release/aurora-exe` that concurrent builds would otherwise overwrite
/// under each other.
///
/// Implemented with an exclusive-create lock file rather than a crate, to avoid
/// a dependency for one call site. A lock older than `STALE` is broken, so a
/// build killed mid-link cannot wedge every later build forever.
struct LinkLock(std::path::PathBuf);

impl LinkLock {
    const STALE: std::time::Duration = std::time::Duration::from_secs(900);
    const TIMEOUT: std::time::Duration = std::time::Duration::from_secs(1800);

    fn acquire(dir: &std::path::Path) -> Result<LinkLock, String> {
        let path = dir.join("link.lock");
        let start = std::time::Instant::now();
        loop {
            match std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&path)
            {
                Ok(mut f) => {
                    use std::io::Write;
                    let _ = writeln!(f, "{}", std::process::id());
                    return Ok(LinkLock(path));
                }
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                    // Break a lock left behind by a build that died mid-link.
                    if let Ok(md) = std::fs::metadata(&path) {
                        if let Ok(age) = md.modified().and_then(|m| {
                            m.elapsed()
                                .map_err(|_| std::io::Error::other("clock went backwards"))
                        }) {
                            if age > Self::STALE {
                                let _ = std::fs::remove_file(&path);
                                continue;
                            }
                        }
                    }
                    if start.elapsed() > Self::TIMEOUT {
                        return Err(format!(
                            "timed out waiting for the link lock at {}",
                            path.display()
                        ));
                    }
                    std::thread::sleep(std::time::Duration::from_millis(100));
                }
                Err(e) => return Err(format!("cannot create {}: {e}", path.display())),
            }
        }
    }
}

impl Drop for LinkLock {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

/// Compile an Aurora program to a standalone native executable.
///
/// Pipeline: parse + check → emit a native object (`aurora-codegen::build_object`)
/// → link it with the `aurora-runtime` host functions and an entry shim by
/// `cargo build`ing the `aurora-exe` crate with `AURORA_OBJ` pointing at the
/// object → copy the resulting binary to the output path.
fn cmd_build(path: &str, rest: &[String]) -> ExitCode {
    use std::path::{Path, PathBuf};

    // Parse `-o <out>`.
    let mut out: Option<String> = None;
    let mut i = 0;
    while i < rest.len() {
        match rest[i].as_str() {
            "-o" | "--output" => {
                if let Some(v) = rest.get(i + 1) {
                    out = Some(v.clone());
                    i += 2;
                    continue;
                }
                eprintln!("usage: aurorac build <file> [-o <out>]");
                return ExitCode::from(2);
            }
            other => {
                eprintln!("build: unexpected argument `{other}`");
                return ExitCode::from(2);
            }
        }
    }

    let Some(src) = read_program(path) else {
        return ExitCode::FAILURE;
    };
    let (dep_src, declared) = collect_deps();
    let file = SourceFile::new(path, aurora_std::with_std(&format!("{src}{dep_src}")));
    let (module, mut diags) = aurora_parser::parse_str(&file.src);
    diags.extend(aurora_check::check(&module));
    diags.extend(aurora_typeck::check_types(&module));
    diags.extend(undeclared_module_uses(&module, &declared));
    if diags.iter().any(|d| d.is_error()) {
        for d in &diags {
            eprintln!("{}", d.render(&file));
        }
        return ExitCode::FAILURE;
    }

    // Emit the native object file.
    let (obj, failed) = match aurora_codegen::build_object(&module) {
        Ok(out) => out,
        Err(e) => {
            eprintln!("build error: {e}");
            return ExitCode::FAILURE;
        }
    };

    // A function that failed to compile was replaced with a no-op stub: the
    // binary would silently do the wrong thing. Refuse to build if `main` (or any
    // function) was stubbed, and report exactly which and why.
    if report_stub_failures(&failed, "emit a binary") {
        return ExitCode::FAILURE;
    }

    let stem = Path::new(path)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("a");
    // Workspace root, relative to this crate's manifest (crates/aurorac).
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));
    let obj_dir = root.join("target").join("aurora-build");
    if let Err(e) = std::fs::create_dir_all(&obj_dir) {
        eprintln!("build error: cannot create {}: {e}", obj_dir.display());
        return ExitCode::FAILURE;
    }
    // The object file name is unique per invocation. It used to be `<stem>.obj`,
    // so two builds of programs that happened to share a file stem - `main.aur`
    // is the common case - overwrote each other's object.
    let obj_ext = if cfg!(windows) { "obj" } else { "o" };
    let obj_path = obj_dir.join(format!("{stem}-{}.{obj_ext}", std::process::id()));
    if let Err(e) = std::fs::write(&obj_path, &obj) {
        eprintln!("build error: cannot write {}: {e}", obj_path.display());
        return ExitCode::FAILURE;
    }
    // Remove the object however this function exits.
    let _obj_guard = TempFile(obj_path.clone());

    let exe_name = if cfg!(windows) {
        "aurora-exe.exe"
    } else {
        "aurora-exe"
    };
    let built = root.join("target").join("release").join(exe_name);
    let out_path = out.unwrap_or_else(|| {
        if cfg!(windows) {
            format!("{stem}.exe")
        } else {
            stem.to_string()
        }
    });

    // Linking and copying are ONE critical section, held across both steps.
    //
    // Unique object names alone do not make this safe. Every build shells out to
    // `cargo build -p aurora-exe`, and cargo writes one shared
    // `target/release/aurora-exe`. Two concurrent builds therefore race even
    // with different source names: A links its object, B relinks over the same
    // output, then A copies and ships B's program. That is not theoretical - it
    // produced a game binary that ran an unrelated test program and failed every
    // check for reasons that had nothing to do with the game.
    let _link_guard = match LinkLock::acquire(&obj_dir) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("build error: {e}");
            return ExitCode::FAILURE;
        }
    };

    let status = std::process::Command::new("cargo")
        .current_dir(&root)
        .args(["build", "--release", "-p", "aurora-exe"])
        .env("AURORA_OBJ", &obj_path)
        .status();
    match status {
        Ok(s) if s.success() => {}
        Ok(s) => {
            eprintln!("build error: linking failed (cargo exited with {s})");
            return ExitCode::FAILURE;
        }
        Err(e) => {
            eprintln!("build error: failed to invoke cargo: {e}");
            return ExitCode::FAILURE;
        }
    }

    if let Err(e) = std::fs::copy(&built, &out_path) {
        eprintln!(
            "build error: cannot copy {} -> {out_path}: {e}",
            built.display()
        );
        return ExitCode::FAILURE;
    }
    println!("wrote native executable `{out_path}`");
    ExitCode::SUCCESS
}

/// Native source-level debugger. With `--trace`/`--step` it compiles the program
/// with debug instrumentation, runs it at native speed, and prints the line +
/// locals at each breakpoint (or every statement when stepping). Without `-i`,
/// breakpoints just print; with `-i` it drops into an interactive stdin REPL.
fn cmd_debug(path: &str, rest: &[String]) -> ExitCode {
    // `rest` is the debugger's own flags, so the program gets just its name.
    set_program_args(path, &[]);
    let mut breakpoints: Vec<u32> = Vec::new();
    let mut step = false;
    let mut interactive = false;
    let mut i = 0;
    while i < rest.len() {
        match rest[i].as_str() {
            "--break" | "-b" => match rest.get(i + 1).and_then(|s| s.parse::<u32>().ok()) {
                Some(line) => {
                    breakpoints.push(line);
                    i += 2;
                }
                None => {
                    eprintln!("debug: `--break` needs a line number");
                    return ExitCode::from(2);
                }
            },
            "--step" | "--trace" | "-s" => {
                step = true;
                i += 1;
            }
            "-i" | "--interactive" => {
                interactive = true;
                i += 1;
            }
            other => {
                eprintln!("debug: unexpected argument `{other}`");
                return ExitCode::from(2);
            }
        }
    }

    let Some(src) = read_program(path) else {
        return ExitCode::FAILURE;
    };

    // Include the standard library, like the other execution paths.
    let src = aurora_std::with_std(&src);

    if interactive {
        return match aurora_debug::debug_interactive(&src, &breakpoints) {
            Ok(()) => ExitCode::SUCCESS,
            Err(e) => {
                eprintln!("debug error: {e}");
                ExitCode::FAILURE
            }
        };
    }

    // Non-interactive: record and print the trace. Default to stepping if no
    // breakpoints were given.
    if breakpoints.is_empty() {
        step = true;
    }
    match aurora_debug::debug_trace(&src, &breakpoints, step) {
        Ok(stops) => {
            println!("native debug trace ({} stops):", stops.len());
            for s in &stops {
                let vars = if s.vars.is_empty() {
                    "(no locals)".to_string()
                } else {
                    s.vars
                        .iter()
                        .map(|(n, v)| format!("{n}={v}"))
                        .collect::<Vec<_>>()
                        .join(", ")
                };
                let frame = s.stack.last().map(String::as_str).unwrap_or("?");
                let indent = "  ".repeat(s.stack.len().saturating_sub(1));
                println!("  line {:>3} {indent}[{frame}]  {vars}", s.line);
            }
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("debug error: {e}");
            ExitCode::FAILURE
        }
    }
}

/// Run a program under the native profiler and print a per-function report
/// (call counts + wall-clock time), sorted by time.
fn cmd_profile(path: &str) -> ExitCode {
    set_program_args(path, &[]);
    let Some(src) = read_program(path) else {
        return ExitCode::FAILURE;
    };
    let file = SourceFile::new(path, aurora_std::with_std(&src));
    let (module, mut diags) = aurora_parser::parse_str(&file.src);
    diags.extend(aurora_check::check(&module));
    diags.extend(aurora_typeck::check_types(&module));
    if diags.iter().any(|d| d.is_error()) {
        for d in &diags {
            eprintln!("{}", d.render(&file));
        }
        return ExitCode::FAILURE;
    }
    let jit = match aurora_codegen::build_profile(&module) {
        Ok(j) => {
            if report_stub_failures(j.failures(), "profile") {
                return ExitCode::FAILURE;
            }
            if !j.compiled("main") {
                eprintln!("profile: `main` did not compile to native code");
                return ExitCode::FAILURE;
            }
            j
        }
        Err(e) => {
            eprintln!("profile error: {e}");
            return ExitCode::FAILURE;
        }
    };
    aurora_runtime::prof_reset();
    if let Err(e) = jit.call_i64("main", &[]) {
        eprintln!("profile: runtime error: {e}");
        return ExitCode::FAILURE;
    }
    use std::io::Write;
    let _ = std::io::stdout().flush();
    let rows = aurora_runtime::prof_report();
    println!("\n=== profile (by total time) ===");
    println!("{:>10}  {:>14}  function", "calls", "total (µs)");
    for r in rows {
        println!(
            "{:>10}  {:>14.3}  {}",
            r.calls,
            r.nanos as f64 / 1000.0,
            r.func
        );
    }
    ExitCode::SUCCESS
}

/// Watch `path` and re-run it whenever it changes (a simple hot-reload loop).
fn cmd_watch(path: &str) -> ExitCode {
    use std::time::{Duration, SystemTime};
    fn mtime(p: &str) -> Option<SystemTime> {
        std::fs::metadata(p).ok().and_then(|m| m.modified().ok())
    }
    println!("watching `{path}` — re-running on change (Ctrl-C to stop)");
    let mut last: Option<SystemTime> = None;
    loop {
        let cur = mtime(path);
        if cur != last {
            last = cur;
            println!("\n── running {path} ──");
            // Run in a child process so a crash doesn't kill the watcher, and so
            // the program's `process::exit` doesn't end the loop.
            let exe = std::env::current_exe().unwrap_or_else(|_| "aurorac".into());
            let _ = std::process::Command::new(exe)
                .arg("run")
                .arg(path)
                .status();
        }
        std::thread::sleep(Duration::from_millis(300));
    }
}

fn cmd_jit(path: &str, rest: &[String]) -> ExitCode {
    // `rest` names the function to call and its integer arguments, not the
    // program's, so the program sees only its own name.
    set_program_args(path, &[]);
    let Some(func) = rest.first() else {
        eprintln!("usage: aurorac jit <file> <function> [int args...]");
        return ExitCode::from(2);
    };
    let raw = &rest[1..];
    let is_float = raw.iter().any(|a| a.contains('.'));

    let Some(src) = read_program(path) else {
        return ExitCode::FAILURE;
    };
    let file = SourceFile::new(path, src);
    let (module, diags) = aurora_parser::parse_str(&file.src);
    if diags.iter().any(|d| d.is_error()) {
        for d in &diags {
            eprintln!("{}", d.render(&file));
        }
        return ExitCode::FAILURE;
    }

    // Float entries take/return f64; integer entries take/return i64.
    let result = if is_float {
        match raw
            .iter()
            .map(|a| a.parse::<f64>())
            .collect::<Result<Vec<_>, _>>()
        {
            Ok(args) => aurora_codegen::jit_call_f64(&module, func, &args).map(|r| r.to_string()),
            Err(_) => Err("jit arguments must be numbers".into()),
        }
    } else {
        match raw
            .iter()
            .map(|a| a.parse::<i64>())
            .collect::<Result<Vec<_>, _>>()
        {
            Ok(args) => aurora_codegen::jit_call(&module, func, &args).map(|r| r.to_string()),
            Err(_) => Err("jit arguments must be integers".into()),
        }
    };

    match result {
        Ok(r) => {
            println!("{r}");
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("jit error: {e}");
            ExitCode::FAILURE
        }
    }
}

fn cmd_run(path: &str, args: &[String]) -> ExitCode {
    set_program_args(path, args);
    let Some(src) = read_program(path) else {
        return ExitCode::FAILURE;
    };
    let (deps, declared) = collect_deps();
    let file = SourceFile::new(path, aurora_std::with_std(&format!("{src}{deps}")));
    let (module, mut diags) = aurora_parser::parse_str(&file.src);
    diags.extend(aurora_check::check(&module));
    diags.extend(aurora_typeck::check_types(&module));
    diags.extend(undeclared_module_uses(&module, &declared));

    let errors = diags.iter().filter(|d| d.is_error()).count();
    if errors > 0 {
        for d in &diags {
            eprintln!("{}", d.render(&file));
        }
        eprintln!("{errors} error(s); not running");
        return ExitCode::FAILURE;
    }

    // Aurora is a compiled language: always lower `main` to native machine code
    // and run it. No interpreter fallback.
    match aurora_codegen::build(&module) {
        Ok(jit) => {
            // Every stubbed function, not only `main`: running a program whose
            // helper was replaced by `return 0` produces plausible-looking output
            // with the real behaviour missing.
            if report_stub_failures(jit.failures(), "run") {
                return ExitCode::FAILURE;
            }
            if !jit.compiled("main") {
                report_no_main(&jit, path);
                return ExitCode::FAILURE;
            }
            match jit.call_i64("main", &[]) {
                Ok(_) => {
                    use std::io::Write;
                    let _ = std::io::stdout().flush();
                    // Leak windowed GPU state / drop headless GPU state deliberately,
                    // then exit directly so nothing is torn down during thread-local
                    // teardown (which trips wgpu's internals).
                    aurora_runtime::aurora_runtime_shutdown();
                    std::process::exit(0);
                }
                Err(e) => {
                    eprintln!("native error: {e}");
                    ExitCode::FAILURE
                }
            }
        }
        Err(e) => {
            eprintln!("native error: {e}");
            ExitCode::FAILURE
        }
    }
}

fn cmd_check(path: &str) -> ExitCode {
    let Some(src) = read_program(path) else {
        return ExitCode::FAILURE;
    };
    // Check exactly the program `run` and `build` compile: the user's source,
    // its dependencies, and the standard prelude. Checking a different program
    // than the one that runs is how `check` could pass a call to `lerp` in one
    // command and reject it in another.
    let (dep_src, declared) = collect_deps();
    let user_src = format!("{src}{dep_src}");
    let boundary = user_src.len() as u32;
    let file = SourceFile::new(path, aurora_std::with_std(&user_src));
    let (module, mut diags) = aurora_parser::parse_str(&file.src);
    diags.extend(aurora_check::check(&module));
    diags.extend(aurora_typeck::check_types(&module));
    diags.extend(undeclared_module_uses(&module, &declared));

    let errors = diags.iter().filter(|d| d.is_error()).count();
    for d in &diags {
        eprintln!("{}", d.render(&file));
    }
    if errors == 0 {
        // The prelude is appended after the user's source, so an item that
        // starts beyond the boundary belongs to the standard library. Report the
        // user's item count: that number is about their program.
        let items = module
            .items
            .iter()
            .filter(|it| it.span.lo < boundary)
            .count();
        // A file with NO items of its own is almost never what somebody meant,
        // and reporting it as a success is a silent failure with a green tick
        // on it. Measured: a script truncated a source file to zero bytes and
        // `check` answered "ok: checked 0 item(s), no errors" for it - the only
        // signal that 597 lines had just been destroyed was a number nobody
        // reads.
        //
        // A comment-only file is the one legitimate case and it is not a thing
        // anybody checks, so this refuses rather than warning.
        if items == 0 {
            eprintln!(
                "error: {} declares no items - an empty or comment-only file is                  almost always a truncated one, and reporting it as checked is                  how a deleted file passes",
                path
            );
            return ExitCode::FAILURE;
        }
        println!("ok: checked {items} item(s), no errors");
        ExitCode::SUCCESS
    } else {
        eprintln!("{errors} error(s)");
        ExitCode::FAILURE
    }
}

fn cmd_parse(path: &str) -> ExitCode {
    let Some(src) = read_program(path) else {
        return ExitCode::FAILURE;
    };
    let file = SourceFile::new(path, src);
    let (module, diags) = aurora_parser::parse_str(&file.src);

    let errors = diags.iter().filter(|d| d.is_error()).count();
    if diags.is_empty() {
        println!("{:#?}", module);
        println!("\nparsed {} item(s), no diagnostics", module.items.len());
        ExitCode::SUCCESS
    } else {
        for d in &diags {
            eprintln!("{}", d.render(&file));
        }
        eprintln!("{errors} error(s)");
        if errors > 0 {
            ExitCode::FAILURE
        } else {
            ExitCode::SUCCESS
        }
    }
}
