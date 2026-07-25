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

use std::path::PathBuf;
use std::process::{Command, Output};

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
fn aurorac(cmd: &str, entry: &PathBuf) -> Output {
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
            ("main.aur", "mod mid;\nfn main() { println(mid::doubled()) }"),
            ("mid.aur", "mod leaf;\nfn doubled() -> i64 { leaf::base() * 2 }"),
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
    assert!(out.status.success(), "check failed: {}", String::from_utf8_lossy(&out.stderr));
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
    let entry = program("badmod", &[("main.aur", "mod nope;\nfn main() { println(1) }")]);
    let out = aurorac("check", &entry);
    let stderr = String::from_utf8_lossy(&out.stderr);
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(!out.status.success(), "check passed an unresolvable module: {stdout}");
    assert!(!stdout.contains("no errors"), "check reported a false green: {stdout}");
    assert!(stderr.contains("E0110"), "expected an E0110 error, got: {stderr}");
    assert!(stderr.contains("nope.aur"), "error must name the path looked for: {stderr}");
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
    assert!(!out.status.success(), "a type error in a module file was not reported");
    assert!(stderr.contains("expected `i64`"), "expected a type error, got: {stderr}");
}

/// End-to-end through the driver: a multi-file program compiles to native code
/// and prints the right answer.
#[test]
fn run_executes_a_nested_multi_file_program() {
    let out = aurorac("run", &nested_program("run"));
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(out.status.success(), "run failed: {}", String::from_utf8_lossy(&out.stderr));
    assert_eq!(stdout.trim(), "20", "unexpected program output: {stdout}");
}

/// `run` must not execute a program with an unresolvable module either.
#[test]
fn run_fails_on_an_unresolvable_module() {
    let entry = program("runbad", &[("main.aur", "mod nope;\nfn main() { println(1) }")]);
    let out = aurorac("run", &entry);
    assert!(!out.status.success(), "run executed a program with an unresolved module");
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
    let entry = program("stubhelper", &[("main.aur", HELPER_THE_BACKEND_CANNOT_LOWER)]);
    let out = aurorac("run", &entry);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(!out.status.success(), "run executed a program with a stubbed function: {stdout}");
    assert!(
        !stdout.contains("main ran"),
        "the program must not run at all when a helper was stubbed: {stdout}"
    );
    assert!(stderr.contains("helper"), "the error must name the function: {stderr}");
    assert!(
        stderr.contains("C::make"),
        "the error must say why the function failed: {stderr}"
    );
}

/// The same program through the AOT path. `build` already refused; it must keep
/// refusing, and for the same stated reason.
#[test]
fn build_refuses_a_program_whose_helper_failed_to_compile() {
    let entry = program("stubhelperaot", &[("main.aur", HELPER_THE_BACKEND_CANNOT_LOWER)]);
    let out = aurorac("build", &entry);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(!out.status.success(), "build emitted a binary with a stubbed function");
    assert!(stderr.contains("helper"), "the error must name the function: {stderr}");
    assert!(stderr.contains("C::make"), "the error must say why: {stderr}");
}

/// The unresolved-name form of the same bug reaches the type checker first, so
/// it is rejected before codegen, by both the JIT and the AOT driver.
#[test]
fn run_and_build_reject_an_unknown_callee_in_a_helper() {
    for cmd in ["run", "build"] {
        let entry = program(&format!("unknown_{cmd}"), &[("main.aur", UNKNOWN_CALLEE_IN_HELPER)]);
        let out = aurorac(cmd, &entry);
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(!out.status.success(), "`{cmd}` accepted a call to a function that does not exist");
        assert!(stderr.contains("E0313"), "`{cmd}`: expected E0313, got: {stderr}");
        assert!(stderr.contains("no_such_fn"), "`{cmd}`: must name the callee: {stderr}");
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
    assert!(!out.status.success(), "check passed an unknown function: {stdout}");
    assert!(!stdout.contains("no errors"), "check reported a false green: {stdout}");
    assert!(stderr.contains("E0313"), "expected an E0313 error, got: {stderr}");
    assert!(stderr.contains("no_such_fn"), "the error must name the callee: {stderr}");
}

/// The same unknown call directly in `main`.
#[test]
fn check_rejects_an_unknown_function_called_from_main() {
    let entry =
        program("checkmain", &[("main.aur", "fn main() { println(no_such_function_anywhere(1)) }")]);
    let out = aurorac("check", &entry);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(!out.status.success(), "check passed an unknown function in main: {stdout}");
    assert!(!stdout.contains("no errors"), "check reported a false green: {stdout}");
    assert!(
        stderr.contains("no_such_function_anywhere"),
        "the error must name the callee: {stderr}"
    );
}

/// `run` must reject it too, and agree with `check`: one program, one meaning.
#[test]
fn run_rejects_an_unknown_function_called_from_main() {
    let entry =
        program("runmain", &[("main.aur", "fn main() { println(no_such_function_anywhere(1)) }")]);
    let out = aurorac("run", &entry);
    assert!(!out.status.success(), "run executed a call to a function that does not exist");
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
    let good = program("parity_ok", &[("main.aur", "fn main() { println(str(clamp01(2.0))) }")]);
    let bad = program("parity_bad", &[("main.aur", "fn main() { println(str(clamp01x(2.0))) }")]);
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
    let entry = program("countuser", &[("main.aur", "fn helper() -> i64 { 1 }\nfn main() { println(helper()) }")]);
    let out = aurorac("check", &entry);
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(out.status.success(), "check failed: {}", String::from_utf8_lossy(&out.stderr));
    assert!(stdout.contains("checked 2 item(s)"), "expected 2 checked items, got: {stdout}");
}
