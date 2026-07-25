//! File-based modules: `mod NAME;` must load `NAME.aur` from the declaring
//! file's directory and feed its items through module flattening, so every later
//! pass sees them as ordinary `NAME::`-prefixed top-level items.
//!
//! These tests cover the resolution rule itself (see `docs/01-grammar-and-types.md`
//! §3.1). Cross-module *execution* through the JIT and the AOT object backend is
//! covered in `aurora-codegen`'s tests.

use std::path::{Path, PathBuf};

use aurora_diag::Diagnostic;
use aurora_parser::ast::ItemKind;

/// Write a throwaway multi-file program to its own temp directory and return the
/// path of its entry file. `files[0]` is the entry.
fn program(tag: &str, files: &[(&str, &str)]) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("aurora_modload_{}_{tag}", std::process::id()));
    // Fresh every run, so a stale file from an earlier run cannot mask a failure.
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create temp dir");
    for (name, src) in files {
        std::fs::write(dir.join(name), src).expect("write module file");
    }
    dir.join(files[0].0)
}

/// Expand `entry`'s file modules, parse (which flattens), and report the names of
/// the resulting top-level items plus every diagnostic from either phase.
fn load(entry: &Path) -> (Vec<String>, Vec<Diagnostic>) {
    let src = std::fs::read_to_string(entry).expect("read entry file");
    let (expanded, mut diags) = aurora_parser::load_file_modules(&src, entry);
    let (module, parse_diags) = aurora_parser::parse_str(&expanded);
    diags.extend(parse_diags);
    let names = module
        .items
        .iter()
        .filter_map(|it| match &it.kind {
            ItemKind::Fn(f) => Some(f.name.name.clone()),
            ItemKind::Struct(s) | ItemKind::Component(s) => Some(s.name.name.clone()),
            ItemKind::Enum(e) => Some(e.name.name.clone()),
            ItemKind::Const(c) => Some(c.name.name.clone()),
            _ => None,
        })
        .collect();
    (names, diags)
}

fn assert_no_errors(diags: &[Diagnostic]) {
    let errors: Vec<_> = diags
        .iter()
        .filter(|d| d.is_error())
        .map(|d| &d.message)
        .collect();
    assert!(errors.is_empty(), "unexpected errors: {errors:?}");
}

/// The whole point: a bodiless `mod NAME;` must actually pull in `NAME.aur`. It
/// used to be parsed and then dropped, so the module contributed nothing.
#[test]
fn file_module_function_is_loaded_and_namespaced() {
    let entry = program(
        "fn",
        &[
            (
                "main.aur",
                "mod helper;\nfn main() { println(helper::add(2, 3)) }",
            ),
            ("helper.aur", "fn add(a: i64, b: i64) -> i64 { a + b }"),
        ],
    );
    let (names, diags) = load(&entry);
    assert_no_errors(&diags);
    assert!(
        names.contains(&"helper::add".to_string()),
        "module fn missing: {names:?}"
    );
    assert!(
        names.contains(&"main".to_string()),
        "entry fn missing: {names:?}"
    );
}

/// Every item kind a module can define has to come across, not just functions.
#[test]
fn file_module_struct_enum_and_const_are_loaded() {
    let entry = program(
        "items",
        &[
            ("main.aur", "mod shape;\nfn main() { println(1) }"),
            (
                "shape.aur",
                "struct P { x: f64, y: f64 }\nenum Kind { Small, Big }\nconst LIMIT: i64 = 7\n",
            ),
        ],
    );
    let (names, diags) = load(&entry);
    assert_no_errors(&diags);
    for want in ["shape::P", "shape::Kind", "shape::LIMIT"] {
        assert!(
            names.contains(&want.to_string()),
            "`{want}` missing: {names:?}"
        );
    }
}

/// A loaded file may declare file modules of its own, resolved against its own
/// directory. Modules form a flat namespace, so `leaf` lands as `leaf::`, not
/// `mid::leaf::`.
#[test]
fn nested_file_module_is_loaded_transitively() {
    let entry = program(
        "nested",
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
    );
    let (names, diags) = load(&entry);
    assert_no_errors(&diags);
    assert!(
        names.contains(&"mid::doubled".to_string()),
        "mid missing: {names:?}"
    );
    assert!(
        names.contains(&"leaf::base".to_string()),
        "transitive leaf missing: {names:?}"
    );
}

/// A module whose file does not exist must be a hard error naming the path that
/// was looked for. Silently contributing nothing is the failure mode this whole
/// feature exists to remove.
#[test]
fn missing_module_file_is_a_hard_error() {
    let entry = program(
        "missing",
        &[("main.aur", "mod nope;\nfn main() { println(1) }")],
    );
    let (_, diags) = load(&entry);
    let d = diags
        .iter()
        .find(|d| d.is_error() && d.code.as_deref() == Some("E0110"))
        .expect("missing module file must produce an E0110 error");
    assert!(
        d.message.contains("nope"),
        "error must name the module: {}",
        d.message
    );
    assert!(
        d.notes.iter().any(|n| n.contains("nope.aur")),
        "error must name the path looked for: {:?}",
        d.notes
    );
}

/// Directory modules (`NAME/mod.aur`) are deliberately not supported; the error
/// has to say so rather than just claiming the file is absent.
#[test]
fn directory_module_is_reported_as_unsupported() {
    let entry = program(
        "dirmod",
        &[("main.aur", "mod pack;\nfn main() { println(1) }")],
    );
    std::fs::create_dir_all(entry.parent().unwrap().join("pack")).expect("create module dir");
    let (_, diags) = load(&entry);
    let d = diags
        .iter()
        .find(|d| d.is_error())
        .expect("a directory is not a module");
    assert!(
        d.notes
            .iter()
            .any(|n| n.contains("directory modules are not supported")),
        "expected a directory-module note, got {:?}",
        d.notes
    );
}

/// Two modules that both declare a third (a diamond) must load it exactly once:
/// twice would define every item in it under two identical names.
#[test]
fn diamond_import_loads_the_shared_module_once() {
    let entry = program(
        "diamond",
        &[
            (
                "main.aur",
                "mod l;\nmod r;\nfn main() { println(l::lv() + r::rv()) }",
            ),
            (
                "l.aur",
                "mod shared;\nfn lv() -> i64 { shared::base() + 1 }",
            ),
            (
                "r.aur",
                "mod shared;\nfn rv() -> i64 { shared::base() + 2 }",
            ),
            ("shared.aur", "fn base() -> i64 { 4 }"),
        ],
    );
    let (names, diags) = load(&entry);
    assert_no_errors(&diags);
    let loads = names.iter().filter(|n| *n == "shared::base").count();
    assert_eq!(
        loads, 1,
        "shared module loaded {loads} times, want exactly 1: {names:?}"
    );
}

/// Two files that declare each other terminate instead of recursing forever, and
/// both are loaded once. Load-once is what bounds the walk.
#[test]
fn mutual_module_declarations_terminate_and_load_once() {
    let entry = program(
        "cycle",
        &[
            ("main.aur", "mod ca;\nfn main() { println(ca::av()) }"),
            ("ca.aur", "mod cb;\nfn av() -> i64 { cb::bv() + 3 }"),
            ("cb.aur", "mod ca;\nfn bv() -> i64 { 4 }"),
        ],
    );
    let (names, diags) = load(&entry);
    assert_no_errors(&diags);
    assert_eq!(
        names.iter().filter(|n| *n == "ca::av").count(),
        1,
        "{names:?}"
    );
    assert_eq!(
        names.iter().filter(|n| *n == "cb::bv").count(),
        1,
        "{names:?}"
    );
}

/// The entry file is the root module. Loading it again under its own name (a file
/// that declares itself, directly or through a cycle) would give every item it
/// defines a second, prefixed name.
#[test]
fn entry_file_is_not_reloaded_under_its_own_name() {
    let entry = program(
        "selfmod",
        &[("main.aur", "mod main;\nfn only() -> i64 { 1 }")],
    );
    let (names, diags) = load(&entry);
    assert_no_errors(&diags);
    assert_eq!(
        names,
        vec!["only".to_string()],
        "root module was re-loaded: {names:?}"
    );
}

/// One file is one module, so a module name is a single segment. `mod a::b;` used
/// to parse as `mod a` plus junk; it must be an explicit error instead.
#[test]
fn path_module_declaration_is_rejected() {
    let entry = program(
        "pathmod",
        &[("main.aur", "mod a::b;\nfn main() { println(1) }")],
    );
    let (_, diags) = load(&entry);
    let d = diags
        .iter()
        .find(|d| d.is_error())
        .expect("`mod a::b;` must be rejected");
    assert!(
        d.message.contains("path modules"),
        "expected a path-module error, got: {}",
        d.message
    );
}

/// Expansion only appends, so every byte offset in the entry file is unchanged
/// and diagnostics reported against it keep pointing at the right place.
#[test]
fn expansion_appends_and_preserves_entry_offsets() {
    let entry = program(
        "offsets",
        &[
            (
                "main.aur",
                "mod helper;\nfn main() { println(helper::add(2, 3)) }",
            ),
            ("helper.aur", "fn add(a: i64, b: i64) -> i64 { a + b }"),
        ],
    );
    let src = std::fs::read_to_string(&entry).unwrap();
    let (expanded, diags) = aurora_parser::load_file_modules(&src, &entry);
    assert_no_errors(&diags);
    assert!(
        expanded.starts_with(&src),
        "expansion must not move the entry file's bytes"
    );
    assert!(
        expanded.len() > src.len(),
        "the module's text was never appended"
    );
}
