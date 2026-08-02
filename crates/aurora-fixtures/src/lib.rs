//! Where the licensed source art lives for tests, and what happens when it does not.
//!
//! **This crate exists because eight tests were passing without running.**
//!
//! Every test that needs a real FBX read `AURORA_TEST_FBX_DIR` and, when it was
//! unset, did `let Some(m) = fixture(..) else { return };` - a silent early
//! return that reports the test as `ok`. The variable was unset on the machine
//! this engine is developed on, so all six root-motion tests and both modular
//! character tests had never executed an assertion. `cargo test` said 811 passed
//! and eight of those were empty.
//!
//! That is the worst shape a check can have: it is not missing, so nobody adds
//! it; and it is not failing, so nobody looks at it. A test that cannot run
//! should be as loud as a test that fails.
//!
//! So the default is inverted here. Fixtures ABSENT is an error, and skipping
//! them is something you have to ask for by name.

/// The directory holding the licensed pack FBX files.
///
/// # Panics
///
/// When `AURORA_TEST_FBX_DIR` is unset or does not exist, unless
/// `AURORA_SKIP_FIXTURE_TESTS=1` says the caller knows and accepts it. The
/// panic is the point: it is the difference between "this machine cannot run
/// these" and "these have not run for months and nobody noticed".
pub fn dir() -> Option<std::path::PathBuf> {
    let skip_allowed = std::env::var("AURORA_SKIP_FIXTURE_TESTS").as_deref() == Ok("1");
    match std::env::var("AURORA_TEST_FBX_DIR") {
        Ok(d) if std::path::Path::new(&d).is_dir() => Some(std::path::PathBuf::from(d)),
        Ok(d) => {
            if skip_allowed {
                return None;
            }
            panic!(
                "AURORA_TEST_FBX_DIR is set to `{d}`, which is not a directory.\n\
                 These tests read real pack art and cannot check anything without it.\n\
                 Set it correctly, or set AURORA_SKIP_FIXTURE_TESTS=1 to say so on purpose."
            )
        }
        Err(_) => {
            if skip_allowed {
                return None;
            }
            panic!(
                "AURORA_TEST_FBX_DIR is not set, so this test would check NOTHING and \
                 still report ok.\n\
                 That is exactly how six root-motion tests and two character tests went \
                 months without running.\n\
                 Point it at the staged pack files (the game repo keeps them in \
                 `_staging/fixtures`), or set AURORA_SKIP_FIXTURE_TESTS=1 if you \
                 genuinely do not have them."
            )
        }
    }
}

/// One fixture file, or `None` when fixtures are legitimately absent.
///
/// A file MISSING from a directory that exists is always an error: the caller
/// named a specific clip, and quietly testing nothing instead is the behaviour
/// this crate was written to remove.
pub fn file(name: &str) -> Option<std::path::PathBuf> {
    let d = dir()?;
    let p = d.join(name);
    assert!(
        p.is_file(),
        "fixture `{name}` is not in {}. The directory is there, so this is a \
         missing or renamed file rather than a machine without the packs - and \
         skipping it would mean this test silently checks nothing.",
        d.display()
    );
    Some(p)
}
