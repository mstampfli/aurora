//! Type-checker tests: catch real mismatches between known types, while staying
//! silent on unresolved external names (the leniency contract).

use crate::check_types;
use aurora_parser::parse_str;

fn errors(src: &str) -> Vec<String> {
    let (module, pdiags) = parse_str(src);
    assert!(
        !pdiags.iter().any(|d| d.is_error()),
        "source failed to parse: {:?}",
        pdiags.iter().map(|d| &d.message).collect::<Vec<_>>()
    );
    check_types(&module)
        .into_iter()
        .filter(|d| d.is_error())
        .map(|d| d.message)
        .collect()
}

#[test]
fn let_annotation_mismatch_is_caught() {
    let errs = errors("fn f() { let x: bool = 1 }");
    assert!(errs.iter().any(|e| e.contains("let binding")), "got {errs:?}");
}

#[test]
fn matching_let_annotation_is_ok() {
    assert!(errors("fn f() { let x: i32 = 1 }").is_empty());
    assert!(errors("fn f() { let x: f32 = 1.0 }").is_empty());
    assert!(errors("fn f() { let ok: bool = true }").is_empty());
}

#[test]
fn wrong_return_type_is_caught() {
    let errs = errors("fn f() -> bool { 1 }");
    assert!(errs.iter().any(|e| e.contains("return value")), "got {errs:?}");
}

#[test]
fn correct_return_type_is_ok() {
    assert!(errors("fn add(a: i32, b: i32) -> i32 { a + b }").is_empty());
}

#[test]
fn mixed_scalar_arithmetic_is_caught() {
    // i32 + f32 with both operands known is a real error.
    let errs = errors("fn f() { let a: i32 = 1\n let b: f32 = 2.0\n let c = a + b }");
    assert!(errs.iter().any(|e| e.contains("arithmetic")), "got {errs:?}");
}

#[test]
fn vector_scalar_arithmetic_is_allowed() {
    // Vec3 * f32 is overloaded algebra, not an error.
    let errs = errors("fn f() { let v: Vec3 = make()\n let s: f32 = 2.0\n let r = v * s }");
    assert!(errs.is_empty(), "vector*scalar should be allowed, got {errs:?}");
}

#[test]
fn unknown_names_do_not_false_positive() {
    // App.new / load / texture are all unresolved externs; no errors expected.
    let errs = errors(
        "fn main() {
            let app = App.new(\"x\", 1, 2)
            let cube: Handle = load(\"c.glb\")
            app.run()
         }",
    );
    assert!(errs.is_empty(), "externs must not false-positive, got {errs:?}");
}

#[test]
fn function_argument_mismatch_is_caught() {
    let errs = errors(
        "fn takes_int(x: i32) -> i32 { x }
         fn f() { takes_int(true) }",
    );
    assert!(errs.iter().any(|e| e.contains("argument")), "got {errs:?}");
}

#[test]
fn if_branches_must_agree_when_known() {
    let errs = errors("fn f() { let x = if cond() { 1 } else { true } }");
    assert!(errs.iter().any(|e| e.contains("if branches")), "got {errs:?}");
}

#[test]
fn struct_field_type_checked_for_local_type() {
    let errs = errors(
        "struct P { x: f32, y: f32 }
         fn f() { let p = P { x: true, y: 2.0 } }",
    );
    assert!(errs.iter().any(|e| e.contains("struct field")), "got {errs:?}");
}

#[test]
fn user_struct_shadows_builtin_name() {
    // A user `struct Vec3` is its own type, not the builtin vector — using it
    // consistently type-checks.
    let errs = errors(
        "struct Vec3 { x: i64, y: i64 }
         fn id(v: Vec3) -> Vec3 { v }
         fn f() { let a = Vec3 { x: 1, y: 2 }\n let b = id(a) }",
    );
    assert!(errs.is_empty(), "user Vec3 should shadow the builtin, got {errs:?}");
}

#[test]
fn missing_required_field_is_caught() {
    let errs = errors(
        "struct P { x: f32, y: f32 }
         fn f() { let p = P { x: 1.0 } }", // y missing, no default
    );
    assert!(errs.iter().any(|e| e.contains("missing field `y`")), "got {errs:?}");
}

#[test]
fn field_with_default_may_be_omitted() {
    // `speed` has a default, so omitting it is fine.
    let errs = errors(
        "component Spinner { speed: f32 = 1.0, name: str }
         fn f() { let s = Spinner { name: \"x\" } }",
    );
    assert!(errs.is_empty(), "defaulted field omission must be ok, got {errs:?}");
}

#[test]
fn base_spread_satisfies_missing_fields() {
    let errs = errors(
        "struct P { x: f32, y: f32 }
         fn f(orig: P) { let p = P { x: 1.0, ..orig } }",
    );
    assert!(errs.is_empty(), "..base should cover the rest, got {errs:?}");
}

#[test]
fn unknown_struct_field_is_caught() {
    let errs = errors(
        "struct P { x: f32 }
         fn f() { let p = P { x: 1.0, z: 2.0 } }",
    );
    assert!(errs.iter().any(|e| e.contains("no field `z`")), "got {errs:?}");
}

#[test]
fn generic_function_calls_instantiate_independently() {
    // `pair<A,B>` accepts any types per call; `first<T>` requires both args to
    // unify to the same `T`.
    assert!(errors(
        "fn pair<A, B>(a: A, b: B) -> (A, B) { (a, b) }
         fn first<T>(a: T, b: T) -> T { a }
         fn f() {
             let p = pair(1, true)
             let q = pair(\"x\", 2.0)
             let r = first(10, 20)
         }"
    )
    .is_empty());
}

#[test]
fn generic_same_param_must_unify() {
    // `first<T>(T, T)` called with mismatched arg types is an error.
    let errs = errors("fn first<T>(a: T, b: T) -> T { a }\nfn f() { first(1, true) }");
    assert!(errs.iter().any(|e| e.contains("argument")), "got {errs:?}");
}

#[test]
fn trait_bound_satisfied_is_ok() {
    let errs = errors(
        "trait Speaker { fn speak(self) -> i64 }
         struct Dog { hp: i64 }
         impl Speaker for Dog { fn speak(self) -> i64 { 7 } }
         fn yell<T: Speaker>(x: T) -> i64 { x.speak() }
         fn f() { yell(Dog { hp: 1 }) }",
    );
    assert!(errs.is_empty(), "Dog implements Speaker, got {errs:?}");
}

#[test]
fn unsatisfied_trait_bound_is_caught() {
    let errs = errors(
        "trait Speaker { fn speak(self) -> i64 }
         struct Rock { w: i64 }
         fn yell<T: Speaker>(x: T) -> i64 { x.speak() }
         fn f() { yell(Rock { w: 1 }) }",
    );
    assert!(
        errs.iter().any(|e| e.contains("does not implement trait `Speaker`")),
        "Rock lacks Speaker, got {errs:?}"
    );
}

#[test]
fn condition_must_be_bool_when_known() {
    let errs = errors("fn f() { if 1 { } }");
    assert!(errs.iter().any(|e| e.contains("condition")), "got {errs:?}");
}
