//! Every shipped example must lower to native code with NO stubbed function.
//!
//! A body that fails to lower is replaced with a stub returning 0, so a
//! regression here does not surface as a build failure: it surfaces as a
//! program that runs and quietly computes nothing. This test compiles each
//! example (it never executes one, so nothing can open a window) and reports the
//! offending function names and reasons.
//!
//! [`KNOWN_STUBBED`] lists the examples that do NOT compile cleanly today; they
//! are pre-existing gaps, not regressions. The list is a ratchet: a new example
//! is covered automatically, and an example that starts compiling cleanly fails
//! the test until it is removed from the list, so the debt can only shrink.

use std::path::{Path, PathBuf};

/// Examples with a known-stubbed function, and why. Do not grow this list.
const KNOWN_STUBBED: &[(&str, &str)] = &[
    // Spec showcases written ahead of the backend.
    (
        "netcode.aur",
        "uses region/field forms the backend does not lower yet",
    ),
    (
        "spinner.aur",
        "uses the unimplemented `App` API and `Transform::rotate`",
    ),
];

fn examples_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("examples")
}

#[test]
fn every_example_lowers_with_no_stubbed_functions() {
    let dir = examples_dir();
    let mut sources: Vec<PathBuf> = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", dir.display()))
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "aur"))
        .collect();
    sources.sort();
    assert!(
        !sources.is_empty(),
        "no examples found in {}",
        dir.display()
    );

    let mut regressions: Vec<String> = Vec::new();
    for path in &sources {
        let src = std::fs::read_to_string(path).expect("read example");
        // The same program the driver compiles: user source plus the prelude.
        let (module, diags) = aurora_parser::parse_str(&aurora_std::with_std(&src));
        let name = path.file_name().unwrap().to_string_lossy().to_string();
        assert!(
            !diags.iter().any(|d| d.is_error()),
            "{name}: example does not parse: {diags:?}"
        );
        let known = KNOWN_STUBBED.iter().find(|(n, _)| *n == name);
        let jit = aurora_codegen::build(&module)
            .unwrap_or_else(|e| panic!("{name}: codegen failed outright: {e}"));
        let failed = jit.failures();
        match known {
            Some((_, why)) if failed.is_empty() => regressions.push(format!(
                "{name}: now compiles cleanly ({why} was fixed) - remove it from KNOWN_STUBBED"
            )),
            Some(_) => {}
            None => {
                let mut names: Vec<&String> = failed.keys().collect();
                names.sort();
                for n in names {
                    regressions.push(format!("{name}: `{n}` was stubbed ({})", failed[n]));
                }
            }
        }
    }
    assert!(
        regressions.is_empty(),
        "examples no longer compile cleanly:\n  {}",
        regressions.join("\n  ")
    );
}

/// A GPU shader stage is not CPU code: the native backend must skip it entirely
/// rather than try, fail, and stub it. Without this a program that ships a
/// `@fragment` shader beside its game code could not be run at all once stubbed
/// functions became a hard error.
#[test]
fn shader_stages_are_not_compiled_as_cpu_code() {
    let src = "@fragment fn shade() -> Color { vec4(0.9, 0.2, 0.5, 1.0) }\n\
               fn main() { println(7) }";
    let (module, diags) = aurora_parser::parse_str(src);
    assert!(
        !diags.iter().any(|d| d.is_error()),
        "parse failed: {diags:?}"
    );
    let jit = aurora_codegen::build(&module).expect("codegen failed");
    assert!(
        jit.failures().is_empty(),
        "a shader stage was compiled as CPU code and stubbed: {:?}",
        jit.failures()
    );
    assert!(jit.compiled("main"), "`main` must still compile");
    // The stage is absent from the native module, so calling it from CPU code is
    // a hard error rather than a silent no-op.
    assert!(
        !jit.compiled("shade"),
        "a shader stage must not be a callable CPU function"
    );
}
