//! `aurorac` — the Aurora compiler driver.
//!
//! Phase A surface: `aurorac lex <file>` tokenizes a source file and prints the
//! token stream (or any diagnostics). More subcommands (`parse`, `check`) land
//! as those phases come online.

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
        Some("run") => match resolve_entry(args.get(1).map(String::as_str)) {
            Ok(path) => cmd_run(&path),
            Err(e) => {
                eprintln!("{e}");
                ExitCode::from(2)
            }
        },
        Some("jit") => match args.get(1) {
            Some(path) => cmd_jit(path, &args[2..]),
            None => {
                eprintln!("usage: aurorac jit <file> <function> [int args...]");
                ExitCode::from(2)
            }
        },
        Some("native") => match args.get(1) {
            Some(path) => cmd_native(path),
            None => {
                eprintln!("usage: aurorac native <file>");
                ExitCode::from(2)
            }
        },
        Some("build") => {
            // First positional arg may be a file, or omitted to use the manifest.
            let explicit = args.get(1).filter(|a| !a.starts_with('-')).map(String::as_str);
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
            println!("  aurorac new <name>      scaffold a new Aurora project (aurora.toml + src/)");
            println!("  aurorac lex <file>      tokenize a source file");
            println!("  aurorac parse <file>    parse a source file to an AST");
            println!("  aurorac check <file>    parse and run static checks");
            println!("  aurorac run <file>      check, then compile `main` to native code & run");
            println!("  aurorac native <file>   compile `main` to native code & run (no interpreter)");
            println!("  aurorac build <file> [-o <out>] compile to a standalone native executable");
            println!("  aurorac jit <file> <fn> [args]  compile a fn to native code & run");
            println!("  aurorac render <out.ppm>        render a demo scene (CPU rasterizer)");
            println!("  aurorac wgsl <file>             lower @vertex/@fragment fns to WGSL");
            println!("  aurorac gpu <file> [-o out.ppm]  run an Aurora @fragment shader on the GPU");
            println!("  aurorac window                  open a live real-time window (interactive demo)");
            println!("  aurorac debug <file> [--break L] native debugger (breakpoints, step, locals)");
            println!("  aurorac sound                   play a demo melody (synthesis + audio output)");
            println!("  aurorac profile <file>          run with the native profiler (per-fn time)");
            println!("  aurorac watch <file>            re-run on file change (hot reload)");
            ExitCode::SUCCESS
        }
        Some(other) => {
            eprintln!("unknown subcommand `{other}` (try `aurorac --help`)");
            ExitCode::from(2)
        }
    }
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
fn collect_dep_sources() -> String {
    use std::collections::HashSet;
    use std::path::Path;
    let mut visited = HashSet::new();
    let mut out = String::new();
    let mut lock: Vec<(String, String)> = Vec::new();
    add_deps(Path::new("."), &mut visited, &mut out, &mut lock);
    if !lock.is_empty() {
        let mut body = String::from("# Auto-generated by aurorac. Resolved dependencies.\n");
        for (name, spec) in &lock {
            body.push_str(&format!("{name} = \"{spec}\"\n"));
        }
        let _ = std::fs::write("aurora.lock", body);
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
                // Resolve this dependency's own dependencies first.
                add_deps(&dir, visited, out, lock);
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
        let dest = std::path::PathBuf::from("target").join("aurora-deps").join(name);
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
    let src = std::fs::read_to_string(dir.join(&lib))
        .map_err(|e| format!("cannot read `{lib}`: {e}"))?;
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

/// Scaffold a new project directory with a manifest and a hello-world program.
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
    let src = match std::fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("error: cannot read `{path}`: {e}");
            return ExitCode::FAILURE;
        }
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

    let src = match std::fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("error: cannot read `{path}`: {e}");
            return ExitCode::FAILURE;
        }
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
    let vs = "@vertex fn vs_main(@builtin(vertex_index) idx: u32) -> @builtin(position) vec4<f32> {\n\
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
        let stem = std::path::Path::new(path).file_stem().and_then(|s| s.to_str()).unwrap_or("shader");
        format!("{stem}.ppm")
    });
    let mut ppm = format!("P6\n{w} {h}\n255\n").into_bytes();
    for px in pixels.chunks_exact(4) {
        ppm.extend_from_slice(&px[..3]);
    }
    match std::fs::write(&out_path, &ppm) {
        Ok(()) => {
            println!("ran `{frag}` on {} → {out_path} ({w}x{h})", gpu.adapter_name());
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
        [Color::rgb(255, 40, 40), Color::rgb(40, 255, 40), Color::rgb(40, 40, 255)],
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

fn cmd_native(path: &str) -> ExitCode {
    let src = match std::fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("error: cannot read `{path}`: {e}");
            return ExitCode::FAILURE;
        }
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

    // Verify `main` actually compiled to native code (not stubbed).
    match aurora_codegen::build(&module) {
        Ok(jit) if jit.compiled("main") => match jit.call_i64("main", &[]) {
            Ok(_) => {
                use std::io::Write;
                let _ = std::io::stdout().flush();
                ExitCode::SUCCESS
            }
            Err(e) => {
                eprintln!("native error: {e}");
                ExitCode::FAILURE
            }
        },
        Ok(_) => {
            eprintln!(
                "native: `main` uses constructs not yet compiled (struct/array/ECS). \
                 Use `aurorac run` to interpret it for now."
            );
            ExitCode::FAILURE
        }
        Err(e) => {
            eprintln!("native error: {e}");
            ExitCode::FAILURE
        }
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

    let src = match std::fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("error: cannot read `{path}`: {e}");
            return ExitCode::FAILURE;
        }
    };
    let file = SourceFile::new(path, aurora_std::with_std(&format!("{src}{}", collect_dep_sources())));
    let (module, mut diags) = aurora_parser::parse_str(&file.src);
    diags.extend(aurora_check::check(&module));
    diags.extend(aurora_typeck::check_types(&module));
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
    if !failed.is_empty() {
        let mut names: Vec<&String> = failed.keys().collect();
        names.sort();
        eprintln!(
            "build error: {} function(s) failed to compile and would be replaced \
             with a no-op stub:",
            failed.len()
        );
        for n in names {
            eprintln!("  - {n}: {}", failed[n]);
        }
        eprintln!("refusing to emit a binary that silently does nothing for these.");
        return ExitCode::FAILURE;
    }

    let stem = Path::new(path).file_stem().and_then(|s| s.to_str()).unwrap_or("a");
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
    let obj_ext = if cfg!(windows) { "obj" } else { "o" };
    let obj_path = obj_dir.join(format!("{stem}.{obj_ext}"));
    if let Err(e) = std::fs::write(&obj_path, &obj) {
        eprintln!("build error: cannot write {}: {e}", obj_path.display());
        return ExitCode::FAILURE;
    }

    // Link via cargo: build the `aurora-exe` crate with our object spliced in.
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

    let exe_name = if cfg!(windows) { "aurora-exe.exe" } else { "aurora-exe" };
    let built = root.join("target").join("release").join(exe_name);
    let out_path = out.unwrap_or_else(|| {
        if cfg!(windows) { format!("{stem}.exe") } else { stem.to_string() }
    });
    if let Err(e) = std::fs::copy(&built, &out_path) {
        eprintln!("build error: cannot copy {} -> {out_path}: {e}", built.display());
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

    let src = match std::fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("error: cannot read `{path}`: {e}");
            return ExitCode::FAILURE;
        }
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
                    s.vars.iter().map(|(n, v)| format!("{n}={v}")).collect::<Vec<_>>().join(", ")
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
    let src = match std::fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("error: cannot read `{path}`: {e}");
            return ExitCode::FAILURE;
        }
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
        Ok(j) if j.compiled("main") => j,
        Ok(_) => {
            eprintln!("profile: `main` did not compile to native code");
            return ExitCode::FAILURE;
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
        println!("{:>10}  {:>14.3}  {}", r.calls, r.nanos as f64 / 1000.0, r.func);
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
            let _ = std::process::Command::new(exe).arg("run").arg(path).status();
        }
        std::thread::sleep(Duration::from_millis(300));
    }
}

fn cmd_jit(path: &str, rest: &[String]) -> ExitCode {
    let Some(func) = rest.first() else {
        eprintln!("usage: aurorac jit <file> <function> [int args...]");
        return ExitCode::from(2);
    };
    let raw = &rest[1..];
    let is_float = raw.iter().any(|a| a.contains('.'));

    let src = match std::fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("error: cannot read `{path}`: {e}");
            return ExitCode::FAILURE;
        }
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
        match raw.iter().map(|a| a.parse::<f64>()).collect::<Result<Vec<_>, _>>() {
            Ok(args) => aurora_codegen::jit_call_f64(&module, func, &args).map(|r| r.to_string()),
            Err(_) => Err("jit arguments must be numbers".into()),
        }
    } else {
        match raw.iter().map(|a| a.parse::<i64>()).collect::<Result<Vec<_>, _>>() {
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

fn cmd_run(path: &str) -> ExitCode {
    let src = match std::fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("error: cannot read `{path}`: {e}");
            return ExitCode::FAILURE;
        }
    };
    let deps = collect_dep_sources();
    let file = SourceFile::new(path, aurora_std::with_std(&format!("{src}{deps}")));
    let (module, mut diags) = aurora_parser::parse_str(&file.src);
    diags.extend(aurora_check::check(&module));
    diags.extend(aurora_typeck::check_types(&module));

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
        Ok(jit) if jit.compiled("main") => match jit.call_i64("main", &[]) {
            Ok(_) => {
                use std::io::Write;
                let _ = std::io::stdout().flush();
                // Exit directly so leaked GPU/audio resources aren't dropped
                // during thread-local teardown (which trips wgpu's internals).
                std::process::exit(0);
            }
            Err(e) => {
                eprintln!("native error: {e}");
                ExitCode::FAILURE
            }
        },
        Ok(jit) => {
            match jit.compile_error("main") {
                Some(reason) => eprintln!("error: `main` did not compile to native code: {reason}"),
                None => eprintln!("error: `main` did not compile to native code (codegen gap)"),
            }
            ExitCode::FAILURE
        }
        Err(e) => {
            eprintln!("native error: {e}");
            ExitCode::FAILURE
        }
    }
}

fn cmd_check(path: &str) -> ExitCode {
    let src = match std::fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("error: cannot read `{path}`: {e}");
            return ExitCode::FAILURE;
        }
    };
    let file = SourceFile::new(path, src);
    let (module, mut diags) = aurora_parser::parse_str(&file.src);
    diags.extend(aurora_check::check(&module));
    diags.extend(aurora_typeck::check_types(&module));

    let errors = diags.iter().filter(|d| d.is_error()).count();
    for d in &diags {
        eprintln!("{}", d.render(&file));
    }
    if errors == 0 {
        println!("ok: checked {} item(s), no errors", module.items.len());
        ExitCode::SUCCESS
    } else {
        eprintln!("{errors} error(s)");
        ExitCode::FAILURE
    }
}

fn cmd_parse(path: &str) -> ExitCode {
    let src = match std::fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("error: cannot read `{path}`: {e}");
            return ExitCode::FAILURE;
        }
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
