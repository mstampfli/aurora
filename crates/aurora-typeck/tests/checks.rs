//! Type-checker regression tests for the call-arity and return-type checks.

use aurora_parser::parse_str;
use aurora_typeck::check_types;

fn errors(src: &str) -> Vec<String> {
    let (module, pdiags) = parse_str(src);
    assert!(
        !pdiags.iter().any(|d| d.is_error()),
        "parse error in test source"
    );
    check_types(&module)
        .iter()
        .filter(|d| d.is_error())
        .map(|d| d.message.clone())
        .collect()
}

#[test]
fn arg_count_mismatch_is_reported() {
    let errs = errors("fn add(a: i64, b: i64) -> i64 { a + b }\nfn main() { let x = add(1) }");
    assert!(
        errs.iter().any(|e| e.contains("expects 2 argument")),
        "got: {errs:?}"
    );

    let errs = errors("fn id(a: i64) -> i64 { a }\nfn main() { let x = id(1, 2, 3) }");
    assert!(
        errs.iter().any(|e| e.contains("expects 1 argument")),
        "got: {errs:?}"
    );
}

#[test]
fn correct_arg_count_is_accepted() {
    let errs = errors("fn add(a: i64, b: i64) -> i64 { a + b }\nfn main() { let x = add(1, 2) }");
    assert!(errs.is_empty(), "unexpected errors: {errs:?}");
}

#[test]
fn return_type_mismatch_is_reported() {
    let errs = errors("fn f() -> i64 { return true }");
    assert!(!errs.is_empty(), "expected a return-type error, got none");

    // An early return of the right type is fine.
    let errs = errors("fn f(x: i64) -> i64 { if x > 0 { return 1 } 2 }");
    assert!(errs.is_empty(), "unexpected errors: {errs:?}");
}

// --- calls ACROSS modules ------------------------------------------------
//
// These were checked for EXISTENCE only. A call into a submodule is collapsed by
// the flattener to one segment and so was checked; a call out to a sibling
// top-level module keeps two segments and fell to the existence branch. In a
// program where every file is its own top-level module - which is how the game
// on this compiler is written - that is nearly every call in the codebase going
// unchecked for arity AND for argument types.
//
// It shipped: a capture script passed two arguments to a four-parameter function
// and `aurorac check` reported no errors.

/// Two sibling modules, one calling the other - the game's exact shape.
const SIBLINGS: &str = r#"
mod lib {
    struct Box2 { w: f64, h: f64 }

    fn four(a: i64, b: i64, c: i64, d: i64) -> i64 { a + b + c + d }
    fn area(b: Box2) -> f64 { b.w * b.h }
    fn make() -> Box2 { Box2 { w: 2.0, h: 3.0 } }
}

mod app {
    fn wrong_count() -> i64 { lib::four(1, 2) }
    fn right_count() -> i64 { lib::four(1, 2, 3, 4) }
    fn wrong_type() -> f64 { lib::area(7) }
    // A value of lib::Box2, handed straight back to lib. The two spellings of
    // that struct - `lib::Box2` written here, `lib::Box2` mangled there - have to
    // be ONE type or this is a false error.
    fn round_trip() -> f64 { lib::area(lib::make()) }
    fn through_param(b: lib::Box2) -> f64 { lib::area(b) }
}

fn main() { }
"#;

#[test]
fn a_cross_module_call_checks_its_arity() {
    let errs = errors(SIBLINGS);
    assert!(
        errs.iter().any(|e| e.contains("expects 4 argument")),
        "two arguments to a four-parameter function in another module went \
         unreported; got: {errs:?}"
    );
}

#[test]
fn a_cross_module_call_checks_its_argument_types() {
    let errs = errors(SIBLINGS);
    assert!(
        errs.iter().any(|e| e.contains("type mismatch in function argument")),
        "an i64 passed where another module wanted a struct went unreported; \
         got: {errs:?}"
    );
}

#[test]
fn a_qualified_type_is_the_same_type_as_its_declaration() {
    // The false-positive guard, and the reason this needed a second fix.
    //
    // `type_to_ty` read only the LAST segment of a path, so a parameter written
    // `lib::Box2` resolved to bare `Box2` while the struct it names had been
    // mangled to `lib::Box2`. Turning argument checking on lit up fifty files in
    // the game at once, every one of them this. Nothing here may complain about
    // `round_trip` or `through_param`.
    // Match on what was FOUND, not on the name appearing anywhere: the
    // deliberate error in SIBLINGS is `lib::area(7)`, which reports "found
    // `i64`" and legitimately names Box2 as the EXPECTED type. The false
    // positive's shape is a Box2 arriving where a Box2 was wanted.
    let errs = errors(SIBLINGS);
    for e in &errs {
        assert!(
            !e.contains("found `Box2`") && !e.contains("found `lib::Box2`"),
            "a struct passed back to its own module was rejected as the wrong \
             type - the qualified and mangled spellings are not unified: {e}"
        );
    }
}

#[test]
fn correct_cross_module_calls_are_accepted() {
    // Exactly one arity error and one type error, both deliberate. More than
    // that means the new check is firing on something correct.
    let errs = errors(SIBLINGS);
    let arity = errs.iter().filter(|e| e.contains("argument(s), found")).count();
    let types = errs
        .iter()
        .filter(|e| e.contains("type mismatch in function argument"))
        .count();
    assert_eq!(arity, 1, "expected exactly one arity error; got: {errs:?}");
    assert_eq!(types, 1, "expected exactly one type error; got: {errs:?}");
}

#[test]
fn an_enum_variant_is_not_mistaken_for_a_call() {
    // `Opt::Some(1)` is a two-segment callee that is not a function. It must not
    // be arity-checked against nothing and reported.
    let errs = errors(
        "enum Opt { Some(i64), None }\nfn main() { let x = Opt::Some(1) }",
    );
    assert!(errs.is_empty(), "enum variant construction was rejected: {errs:?}");
}

/// A value derived from a BUILTIN call carries the builtin's return type.
///
/// It did not, and that was not a cosmetic gap. Every such value was a fresh
/// type variable that unified with anything, so a struct shadowed by a float
/// could be passed where the struct was required: `check` reported no errors and
/// codegen died with "invalid field access in JIT". That breaks the invariant
/// ARCHITECTURE.md states - `check` compiles the same program `run` and `build`
/// do - and it cost a real debugging session in the game.
///
/// The literal case was ALREADY caught, which is what made the hole confusing:
/// `let p = 1.0` was rejected and `let p = cos(p.a)` was not.
#[test]
fn a_builtin_result_is_typed_not_unknown() {
    let src = "struct P { a: f64 }\n\
               fn takes(p: P) -> f64 { p.a }\n\
               fn main() {\n\
                   let p = P { a: 1.0 }\n\
                   let p = cos(p.a)\n\
                   let x = takes(p)\n\
               }";
    let errs = errors(src);
    assert!(
        errs.iter().any(|e| e.contains("expected `P`, found `f64`")),
        "a float from a builtin was accepted where a struct was required: {errs:?}"
    );
}

/// And the same program without the shadowing still type-checks, so the rule
/// above is not simply rejecting builtin results everywhere.
#[test]
fn a_builtin_result_still_unifies_where_it_should() {
    let src = "fn main() {\n\
                   let a = cos(1.0)\n\
                   let b: f64 = a + sin(2.0)\n\
                   let n = abs(0 - 3)\n\
               }";
    let errs = errors(src);
    assert!(errs.is_empty(), "unexpected errors: {errs:?}");
}
