//! Driver-level tests: the failure modes where `aurorac` used to report success.
//!
//! Two families live here, both about the compiler staying silent when it should
//! not. File-based modules (`mod NAME;`): the loader is wired into every
//! subcommand through `read_program`, so an unresolvable module has to fail the
//! command, the item count has to include what the modules brought in, and `run`
//! has to execute across files. And unresolved names: a function that fails to
//! compile used to be replaced with a stub returning 0 everywhere except `main`,
//! and `check` never resolved callees at all, so a call to a function that does
//! not exist passed with a green light.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

/// Serializes the tests that run a REAL `aurorac build`.
///
/// `build` stages its object at `target/aurora-build/<stem>.obj` and links it by
/// `cargo build -p aurora-exe`, whose output is the single
/// `target/release/aurora-exe`. Both are shared, and both of these programs are
/// called `main.aur`, so two builds at once overwrite each other's object and
/// each other's binary: one test then runs the other test's program. That is a
/// limitation of the driver, not of these tests - it would bite two developers
/// building in one checkout too - so until `build` stages per-invocation
/// artifacts, the suite takes them one at a time.
fn build_lock() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    LOCK.lock().unwrap_or_else(|p| p.into_inner())
}

/// Write a throwaway multi-file program to its own temp directory and return the
/// path of its entry file. `files[0]` is the entry.
fn program(tag: &str, files: &[(&str, &str)]) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("aurora_cli_{}_{tag}", std::process::id()));
    // Fresh every run, so a stale file cannot mask a failure.
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create temp dir");
    for (name, src) in files {
        std::fs::write(dir.join(name), src).expect("write module file");
    }
    dir.join(files[0].0)
}

/// Run `aurorac <cmd> <entry>`. Headless, so nothing can try to open a window.
fn aurorac(cmd: &str, entry: &Path) -> Output {
    Command::new(env!("CARGO_BIN_EXE_aurorac"))
        .args([cmd, entry.to_str().unwrap()])
        .env("AURORA_HEADLESS", "1")
        .output()
        .expect("run aurorac")
}

/// A three-file program: `main.aur` -> `mid.aur` -> `leaf.aur`.
fn nested_program(tag: &str) -> PathBuf {
    program(
        tag,
        &[
            (
                "main.aur",
                "mod mid;\nfn main() { println(mid::doubled()) }",
            ),
            (
                "mid.aur",
                "mod leaf;\nfn doubled() -> i64 { leaf::base() * 2 }",
            ),
            ("leaf.aur", "fn base() -> i64 { 10 }"),
        ],
    )
}

/// `check` used to count only the entry file's own items (a `mod` declaration
/// contributed nothing), so it reported 1 item for a two-item program. It must
/// now count everything it actually checked.
#[test]
fn check_counts_items_pulled_in_from_file_modules() {
    let out = aurorac("check", &nested_program("count"));
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        out.status.success(),
        "check failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    // main + mid::doubled + leaf::base
    assert!(
        stdout.contains("checked 3 item(s)"),
        "expected 3 checked items, got: {stdout}"
    );
}

/// The worst failure mode: `check` reporting no errors for a program whose module
/// could not be resolved. It has to fail, and say which path it looked for.
#[test]
fn check_fails_on_an_unresolvable_module() {
    let entry = program(
        "badmod",
        &[("main.aur", "mod nope;\nfn main() { println(1) }")],
    );
    let out = aurorac("check", &entry);
    let stderr = String::from_utf8_lossy(&out.stderr);
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        !out.status.success(),
        "check passed an unresolvable module: {stdout}"
    );
    assert!(
        !stdout.contains("no errors"),
        "check reported a false green: {stdout}"
    );
    assert!(
        stderr.contains("E0110"),
        "expected an E0110 error, got: {stderr}"
    );
    assert!(
        stderr.contains("nope.aur"),
        "error must name the path looked for: {stderr}"
    );
}

/// Items inside a file module are really checked, not merely loaded: a type error
/// in a module file has to fail the entry program's `check`.
#[test]
fn check_reports_a_type_error_inside_a_file_module() {
    let entry = program(
        "typeerr",
        &[
            ("main.aur", "mod bad;\nfn main() { println(bad::oops()) }"),
            ("bad.aur", "fn oops() -> i64 { \"not an int\" }"),
        ],
    );
    let out = aurorac("check", &entry);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        !out.status.success(),
        "a type error in a module file was not reported"
    );
    assert!(
        stderr.contains("expected `i64`"),
        "expected a type error, got: {stderr}"
    );
}

/// End-to-end through the driver: a multi-file program compiles to native code
/// and prints the right answer.
#[test]
fn run_executes_a_nested_multi_file_program() {
    let out = aurorac("run", &nested_program("run"));
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        out.status.success(),
        "run failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert_eq!(stdout.trim(), "20", "unexpected program output: {stdout}");
}

/// `run` must not execute a program with an unresolvable module either.
#[test]
fn run_fails_on_an_unresolvable_module() {
    let entry = program(
        "runbad",
        &[("main.aur", "mod nope;\nfn main() { println(1) }")],
    );
    let out = aurorac("run", &entry);
    assert!(
        !out.status.success(),
        "run executed a program with an unresolved module"
    );
    assert!(
        String::from_utf8_lossy(&out.stderr).contains("E0110"),
        "expected an E0110 error, got: {}",
        String::from_utf8_lossy(&out.stderr)
    );
}

// --- silent compilation failures -------------------------------------------

/// A program calling a function that does not exist, from a helper rather than
/// from `main`. The helper's body cannot compile.
const UNKNOWN_CALLEE_IN_HELPER: &str =
    "fn helper() -> i64 { no_such_fn(1) }\nfn main() { println(str(helper())) }";

/// A program that passes every static check but that the BACKEND cannot lower:
/// `C::make` is an associated function, which the native backend does not
/// compile, and a multi-segment callee is not resolved by the type checker. So
/// this reaches codegen clean and fails there, in a helper rather than in
/// `main`: exactly the shape that used to be stubbed away in silence.
const HELPER_THE_BACKEND_CANNOT_LOWER: &str = "struct C { v: i64 }\n\
     impl C { fn make(v: i64) -> C { C { v: v } } }\n\
     fn helper() -> i64 { let c = C::make(4); c.v }\n\
     fn main() { println(\"main ran\") }";

/// The headline bug: a non-`main` function that fails to compile was replaced
/// with a stub returning 0, and the program ran to completion with exit 0. A
/// gameplay system that fails to compile has to break the build, not quietly
/// evaluate to nothing.
#[test]
fn run_refuses_a_program_whose_helper_failed_to_compile() {
    let entry = program(
        "stubhelper",
        &[("main.aur", HELPER_THE_BACKEND_CANNOT_LOWER)],
    );
    let out = aurorac("run", &entry);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        !out.status.success(),
        "run executed a program with a stubbed function: {stdout}"
    );
    assert!(
        !stdout.contains("main ran"),
        "the program must not run at all when a helper was stubbed: {stdout}"
    );
    assert!(
        stderr.contains("helper"),
        "the error must name the function: {stderr}"
    );
    assert!(
        stderr.contains("C::make"),
        "the error must say why the function failed: {stderr}"
    );
}

/// The same program through the AOT path. `build` already refused; it must keep
/// refusing, and for the same stated reason.
#[test]
fn build_refuses_a_program_whose_helper_failed_to_compile() {
    let entry = program(
        "stubhelperaot",
        &[("main.aur", HELPER_THE_BACKEND_CANNOT_LOWER)],
    );
    let out = aurorac("build", &entry);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        !out.status.success(),
        "build emitted a binary with a stubbed function"
    );
    assert!(
        stderr.contains("helper"),
        "the error must name the function: {stderr}"
    );
    assert!(
        stderr.contains("C::make"),
        "the error must say why: {stderr}"
    );
}

/// The unresolved-name form of the same bug reaches the type checker first, so
/// it is rejected before codegen, by both the JIT and the AOT driver.
#[test]
fn run_and_build_reject_an_unknown_callee_in_a_helper() {
    for cmd in ["run", "build"] {
        let entry = program(
            &format!("unknown_{cmd}"),
            &[("main.aur", UNKNOWN_CALLEE_IN_HELPER)],
        );
        let out = aurorac(cmd, &entry);
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(
            !out.status.success(),
            "`{cmd}` accepted a call to a function that does not exist"
        );
        assert!(
            stderr.contains("E0313"),
            "`{cmd}`: expected E0313, got: {stderr}"
        );
        assert!(
            stderr.contains("no_such_fn"),
            "`{cmd}`: must name the callee: {stderr}"
        );
    }
}

/// `check` used to answer `ok: checked 2 item(s), no errors` for this program:
/// the type checker never resolved callees at all.
#[test]
fn check_rejects_an_unknown_function_called_from_a_helper() {
    let entry = program("checkhelper", &[("main.aur", UNKNOWN_CALLEE_IN_HELPER)]);
    let out = aurorac("check", &entry);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        !out.status.success(),
        "check passed an unknown function: {stdout}"
    );
    assert!(
        !stdout.contains("no errors"),
        "check reported a false green: {stdout}"
    );
    assert!(
        stderr.contains("E0313"),
        "expected an E0313 error, got: {stderr}"
    );
    assert!(
        stderr.contains("no_such_fn"),
        "the error must name the callee: {stderr}"
    );
}

/// The same unknown call directly in `main`.
#[test]
fn check_rejects_an_unknown_function_called_from_main() {
    let entry = program(
        "checkmain",
        &[(
            "main.aur",
            "fn main() { println(no_such_function_anywhere(1)) }",
        )],
    );
    let out = aurorac("check", &entry);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        !out.status.success(),
        "check passed an unknown function in main: {stdout}"
    );
    assert!(
        !stdout.contains("no errors"),
        "check reported a false green: {stdout}"
    );
    assert!(
        stderr.contains("no_such_function_anywhere"),
        "the error must name the callee: {stderr}"
    );
}

/// `run` must reject it too, and agree with `check`: one program, one meaning.
#[test]
fn run_rejects_an_unknown_function_called_from_main() {
    let entry = program(
        "runmain",
        &[(
            "main.aur",
            "fn main() { println(no_such_function_anywhere(1)) }",
        )],
    );
    let out = aurorac("run", &entry);
    assert!(
        !out.status.success(),
        "run executed a call to a function that does not exist"
    );
    assert!(
        String::from_utf8_lossy(&out.stderr).contains("no_such_function_anywhere"),
        "the error must name the callee: {}",
        String::from_utf8_lossy(&out.stderr)
    );
}

/// The leniency that must survive: the several hundred runtime builtins are not
/// user-defined functions, and `check`/`run` must not report them as unknown.
#[test]
fn builtins_and_prelude_functions_still_compile_and_run() {
    // Exercises a builtin family each (RNG, bitwise) plus a prelude function.
    // The RNG value itself is clamped away so the expected output is exact.
    let src = "fn main() {\n\
        \x20   srand(7)\n\
        \x20   let r = rand_int(0, 3)\n\
        \x20   println(str(clampi(r, 5, 5)))\n\
        \x20   println(str(lerp(0.0, 10.0, 0.5)))\n\
        \x20   println(str(band(6, 3)))\n\
        }";
    let entry = program("builtins", &[("main.aur", src)]);
    let check = aurorac("check", &entry);
    assert!(
        check.status.success(),
        "check rejected builtins/prelude calls: {}",
        String::from_utf8_lossy(&check.stderr)
    );
    let run = aurorac("run", &entry);
    assert!(
        run.status.success(),
        "run rejected builtins/prelude calls: {}",
        String::from_utf8_lossy(&run.stderr)
    );
    let stdout = String::from_utf8_lossy(&run.stdout);
    let lines: Vec<&str> = stdout.lines().collect();
    assert_eq!(lines, vec!["5", "5.0", "2"], "unexpected output: {stdout}");
}

/// An `@extern` import has no body and resolves at link/registration time, so it
/// must stay accepted by the name resolver.
#[test]
fn an_extern_declaration_still_checks_and_runs() {
    let src = "@extern fn hypot(x: f64, y: f64) -> f64\n\
               fn main() { println(str(hypot(3.0, 4.0))) }";
    let entry = program("externfn", &[("main.aur", src)]);
    let check = aurorac("check", &entry);
    assert!(
        check.status.success(),
        "check rejected an `@extern` import: {}",
        String::from_utf8_lossy(&check.stderr)
    );
    let run = aurorac("run", &entry);
    assert!(
        run.status.success(),
        "run rejected an `@extern` import: {}",
        String::from_utf8_lossy(&run.stderr)
    );
    assert_eq!(String::from_utf8_lossy(&run.stdout).trim(), "5.0");
}

/// `check` used to type-check a DIFFERENT program from the one `run` compiles:
/// it left out the standard prelude. A program calling a prelude function had to
/// be accepted by both, and a program calling nothing real rejected by both.
#[test]
fn check_and_run_agree_on_the_same_program() {
    let good = program(
        "parity_ok",
        &[("main.aur", "fn main() { println(str(clamp01(2.0))) }")],
    );
    let bad = program(
        "parity_bad",
        &[("main.aur", "fn main() { println(str(clamp01x(2.0))) }")],
    );
    for (entry, want_ok, label) in [(&good, true, "prelude call"), (&bad, false, "typo'd call")] {
        let check = aurorac("check", entry);
        let run = aurorac("run", entry);
        assert_eq!(
            check.status.success(),
            want_ok,
            "check disagreed on the {label}: {}{}",
            String::from_utf8_lossy(&check.stdout),
            String::from_utf8_lossy(&check.stderr)
        );
        assert_eq!(
            run.status.success(),
            want_ok,
            "run disagreed on the {label}: {}{}",
            String::from_utf8_lossy(&run.stdout),
            String::from_utf8_lossy(&run.stderr)
        );
    }
}

/// Adding the prelude to `check` must not change the item count it reports: that
/// number is about the user's program, not the standard library.
#[test]
fn check_counts_only_the_users_items_not_the_prelude() {
    let entry = program(
        "countuser",
        &[(
            "main.aur",
            "fn helper() -> i64 { 1 }\nfn main() { println(helper()) }",
        )],
    );
    let out = aurorac("check", &entry);
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        out.status.success(),
        "check failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(
        stdout.contains("checked 2 item(s)"),
        "expected 2 checked items, got: {stdout}"
    );
}

/// `sys_arg` must mean the same thing whichever way a program was compiled, or
/// role dispatch (`--host`, `--dedicated`) works in one and not the other. Under
/// `run` the compiler owns `std::env::args()`, so it installs the program's own
/// vector: argv[0] the source file, argv[1..] whatever followed it.
#[test]
fn run_passes_the_programs_own_arguments() {
    let entry = program("sysargs", &[("main.aur", ECHO_ARGS)]);
    let out = Command::new(env!("CARGO_BIN_EXE_aurorac"))
        .args(["run", entry.to_str().unwrap(), "--host", "45123"])
        .env("AURORA_HEADLESS", "1")
        .env("AURORA_TEST_ROLE", "host")
        .output()
        .expect("run aurorac");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        out.status.success(),
        "run failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert_eq!(
        stdout.replace('\r', ""),
        expected_echo(entry.to_str().unwrap())
    );
}

/// The same program built as a standalone executable, run with the same
/// arguments: everything but argv[0] (the program as invoked) must match.
#[test]
fn a_built_executable_echoes_its_own_arguments() {
    let _lock = build_lock();
    let entry = program("sysargsaot", &[("main.aur", ECHO_ARGS)]);
    let exe = entry.with_file_name(if cfg!(windows) { "echo.exe" } else { "echo" });
    let build = Command::new(env!("CARGO_BIN_EXE_aurorac"))
        .args([
            "build",
            entry.to_str().unwrap(),
            "-o",
            exe.to_str().unwrap(),
        ])
        .env("AURORA_HEADLESS", "1")
        .output()
        .expect("run aurorac build");
    assert!(
        build.status.success(),
        "build failed: {}{}",
        String::from_utf8_lossy(&build.stdout),
        String::from_utf8_lossy(&build.stderr)
    );
    let out = Command::new(&exe)
        .args(["--host", "45123"])
        .env("AURORA_HEADLESS", "1")
        .env("AURORA_TEST_ROLE", "host")
        .output()
        .expect("run the built executable");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        out.status.success(),
        "exe failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert_eq!(
        stdout.replace('\r', ""),
        expected_echo(exe.to_str().unwrap())
    );
}

// --- terrain end to end ------------------------------------------------------

/// Exercises the terrain builtins that need no GPU: generate, the documented
/// `.aterr` round trip, the height query against a physics raycast on the
/// registered collider, the collision group a ground probe filters to, and the
/// out-of-bounds contract. `terrain_draw` is called with no scene open, which
/// has to be a silent no-op rather than a crash.
///
/// The output is a fixed set of lines, so `run` and `build` can each be compared
/// against it AND against each other: a builtin that works under one and not the
/// other is exactly the failure this guards.
const TERRAIN_PROGRAM: &str = r#"
fn f(i: i64) -> f64 { i as f64 }

fn main() {
    let dir = sys_arg(1)
    if terrain_generate(20260726, 129, 1.5, 30.0) != 1 { println("generate failed"); return }
    println("size=" + str(terrain_size()))
    println("spacing=" + str(terrain_spacing()))

    let path = dir + "/t.aterr"
    let probe = terrain_height(4.25, 0.0 - 6.5)
    if terrain_save(path) != 1 { println("save failed"); return }
    if terrain_load(path) != 1 { println("load failed"); return }
    println("roundtrip=" + str(terrain_height(4.25, 0.0 - 6.5) == probe))

    phys3d_init(0.0, 0.0 - 20.0, 0.0)
    let ground = terrain_collider()
    if ground < 0 { println("collider failed"); return }
    phys3d_step(0.016)
    let mut worst = 0.0
    let mut hits = 0
    let mut i = 0
    while i < 20 {
        let mut j = 0
        while j < 20 {
            let x = 0.0 - 80.0 + 8.1 * f(i)
            let z = 0.0 - 80.0 + 8.3 * f(j)
            if phys3d_raycast_full(x, 200.0, z, 0.0, 0.0 - 1.0, 0.0, 500.0) == ground {
                let d = abs(phys3d_hit_y() - terrain_height(x, z))
                if d > worst { worst = d }
                hits = hits + 1
            }
            j = j + 1
        }
        i = i + 1
    }
    println("rays=" + str(hits))
    println("agree=" + str(worst < 0.001))

    let prober = phys3d_add_character(0.0, 40.0, 0.0, 0.9, 0.4)
    let blocker = phys3d_add_character(0.0, 30.0, 0.0, 0.9, 0.4)
    phys3d_step(0.016)
    let world = phys3d_raycast_world(prober, 0.0, 40.0, 0.0, 0.0, 0.0 - 1.0, 0.0, 300.0)
    let any = phys3d_raycast_ex(prober, 0.0, 40.0, 0.0, 0.0, 0.0 - 1.0, 0.0, 300.0)
    println("worldprobe=" + str(world == ground))
    println("anyprobe=" + str(any == blocker))

    let x0 = terrain_origin_x()
    let z0 = terrain_origin_z()
    println("oob=" + str(terrain_height(x0 - 1000000.0, z0 - 1000000.0) == terrain_height(x0, z0)))
    terrain_draw()
    println("done")
}
"#;

/// What `TERRAIN_PROGRAM` prints when every check holds. A `bool` stringifies as
/// `1`/`0`, so every `1` below is a passing comparison.
const TERRAIN_EXPECTED: &[&str] = &[
    "size=129",
    "spacing=1.5",
    "roundtrip=1",
    "rays=400",
    "agree=1",
    "worldprobe=1",
    "anyprobe=1",
    "oob=1",
    "done",
];

/// The terrain builtins have to behave identically through the JIT and through a
/// standalone executable. Each is one table row, but the two paths resolve those
/// symbols differently (JIT symbol registration vs the AOT link keeper), so only
/// running both proves the row is complete.
#[test]
fn terrain_builtins_work_under_both_run_and_build() {
    let _lock = build_lock();
    let entry = program("terrain", &[("main.aur", TERRAIN_PROGRAM)]);
    let dir = entry.parent().expect("temp dir").to_path_buf();

    let jit = Command::new(env!("CARGO_BIN_EXE_aurorac"))
        .args(["run", entry.to_str().unwrap(), dir.to_str().unwrap()])
        .env("AURORA_HEADLESS", "1")
        .output()
        .expect("run aurorac");
    assert!(
        jit.status.success(),
        "run failed: {}{}",
        String::from_utf8_lossy(&jit.stdout),
        String::from_utf8_lossy(&jit.stderr)
    );
    let jit_out = String::from_utf8_lossy(&jit.stdout).replace('\r', "");
    assert_eq!(
        jit_out.lines().collect::<Vec<_>>(),
        TERRAIN_EXPECTED,
        "JIT output"
    );

    let exe = dir.join(if cfg!(windows) { "terr.exe" } else { "terr" });
    let build = Command::new(env!("CARGO_BIN_EXE_aurorac"))
        .args([
            "build",
            entry.to_str().unwrap(),
            "-o",
            exe.to_str().unwrap(),
        ])
        .env("AURORA_HEADLESS", "1")
        .output()
        .expect("run aurorac build");
    assert!(
        build.status.success(),
        "build failed: {}{}",
        String::from_utf8_lossy(&build.stdout),
        String::from_utf8_lossy(&build.stderr)
    );
    assert!(exe.exists(), "build reported success but wrote no {exe:?}");

    let aot = Command::new(&exe)
        .arg(dir.to_str().unwrap())
        .env("AURORA_HEADLESS", "1")
        .output()
        .expect("run the built executable");
    assert!(
        aot.status.success(),
        "the built executable failed: {}",
        String::from_utf8_lossy(&aot.stderr)
    );
    let aot_out = String::from_utf8_lossy(&aot.stdout).replace('\r', "");
    assert_eq!(
        aot_out.lines().collect::<Vec<_>>(),
        TERRAIN_EXPECTED,
        "AOT output"
    );
    assert_eq!(jit_out, aot_out, "JIT and AOT disagree about the terrain");
}

/// Prints its whole argument vector, both out-of-range ends, and three env
/// lookups, so one comparison covers every edge at once.
const ECHO_ARGS: &str = "fn main() {\n\
     println(str(sys_argc()))\n\
     let mut i = 0\n\
     while i < sys_argc() {\n\
         println(\"[\" + sys_arg(i) + \"]\")\n\
         i = i + 1\n\
     }\n\
     println(\"past=[\" + sys_arg(sys_argc()) + \"]\")\n\
     println(\"far=[\" + sys_arg(1000000) + \"]\")\n\
     println(\"neg=[\" + sys_arg(0 - 1) + \"]\")\n\
     println(\"negfar=[\" + sys_arg(0 - 1000000) + \"]\")\n\
     println(\"role=[\" + sys_env(\"AURORA_TEST_ROLE\") + \"]\")\n\
     println(\"unset=[\" + sys_env(\"AURORA_TEST_DEFINITELY_UNSET\") + \"]\")\n\
     println(\"noname=[\" + sys_env(\"\") + \"]\")\n\
 }";

/// What `ECHO_ARGS` prints for a program invoked as `argv0 --host 45123`.
fn expected_echo(argv0: &str) -> String {
    format!(
        "3\n[{argv0}]\n[--host]\n[45123]\n\
         past=[]\nfar=[]\nneg=[]\nnegfar=[]\n\
         role=[host]\nunset=[]\nnoname=[]\n"
    )
}
