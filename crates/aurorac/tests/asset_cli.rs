//! The `aurorac asset` subcommands.
//!
//! The interesting cases need licensed art that cannot live in the repository,
//! so what is covered here is the contract a pipeline depends on: the exit codes
//! and the refusals. A checker that reports success on a library it could not
//! read is worse than no checker, because it is the one a build trusts.

use std::process::{Command, Output};

fn aurorac(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_aurorac"))
        .args(args)
        .output()
        .expect("aurorac should run")
}

#[test]
fn asset_with_no_subcommand_reports_usage() {
    let out = aurorac(&["asset"]);
    assert_eq!(out.status.code(), Some(2), "usage errors exit 2");
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(err.contains("asset info"), "{err}");
    assert!(err.contains("asset check"), "{err}");
}

#[test]
fn an_unknown_asset_subcommand_reports_usage() {
    let out = aurorac(&["asset", "bake", "x.fbx"]);
    assert_eq!(out.status.code(), Some(2));
}

#[test]
fn info_on_a_missing_file_fails_rather_than_reporting_nothing() {
    let out = aurorac(&["asset", "info", "no/such/model.fbx"]);
    assert_eq!(
        out.status.code(),
        Some(1),
        "a file that cannot be read must fail, not print an empty report"
    );
    assert!(
        !out.stderr.is_empty(),
        "the failure must say what went wrong"
    );
}

#[test]
fn check_without_a_directory_reports_usage() {
    let out = aurorac(&["asset", "check", "rig.fbx"]);
    assert_eq!(out.status.code(), Some(2));
}

#[test]
fn check_with_an_unreadable_reference_fails() {
    // Silently treating a missing reference as "everything conforms" would make
    // this pass in exactly the situation it exists to catch.
    let out = aurorac(&["asset", "check", "no/such/rig.fbx", "."]);
    assert_eq!(out.status.code(), Some(1));
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(err.contains("reference rig"), "{err}");
}
