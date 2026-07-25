//! Driver-level tests: the failure modes where `aurorac` used to report success.
//!
//! Two families live here, both about the compiler staying silent when it should
//! not. File-based modules (`mod NAME;`): the loader is wired into every
//! subcommand through `read_program`, so an unresolvable module has to fail the
//! command, the item count has to include what the modules brought in, and `run`
//! has to execute across files. And stubbed functions: a function that fails to
//! compile is replaced with a stub returning 0, and that was reported for `main`
//! only, so a broken helper ran as a silent no-op.

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
