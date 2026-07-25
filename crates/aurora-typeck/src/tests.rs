//! Type-checker tests: catch real mismatches between known types, while staying
//! silent on unresolved external *types* (the leniency contract).
//!
//! Leniency about a type is not leniency about a name: a direct call to a name
//! that is declared nowhere is an error (`E0313`). The tests at the bottom pin
//! both sides of that line, because getting it wrong in either direction is bad
//! a false green hides a whole broken function, and a false error would fire
//! on every one of the several hundred runtime builtins.

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
    assert!(
        errs.iter().any(|e| e.contains("let binding")),
        "got {errs:?}"
    );
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
    assert!(
        errs.iter().any(|e| e.contains("return value")),
        "got {errs:?}"
    );
}

#[test]
fn correct_return_type_is_ok() {
    assert!(errors("fn add(a: i32, b: i32) -> i32 { a + b }").is_empty());
}

#[test]
fn mixed_scalar_arithmetic_is_caught() {
    // i32 + f32 with both operands known is a real error.
    let errs = errors("fn f() { let a: i32 = 1\n let b: f32 = 2.0\n let c = a + b }");
    assert!(
        errs.iter().any(|e| e.contains("arithmetic")),
        "got {errs:?}"
    );
}

#[test]
fn vector_scalar_arithmetic_is_allowed() {
    // Vec3 * f32 is overloaded algebra, not an error.
    let errs = errors("fn f(v: Vec3) { let s: f32 = 2.0\n let r = v * s }");
    assert!(
        errs.is_empty(),
        "vector*scalar should be allowed, got {errs:?}"
    );
}

#[test]
fn unknown_names_do_not_false_positive() {
    // Unresolved method/field accesses (`App.new`, `app.run()`) stay lenient,
    // and a DECLARED `@extern` import is a real name. Neither may error. A call
    // to a name declared nowhere is a different case, covered below.
    let errs = errors(
        "@extern fn load(path: str) -> Handle
         fn main() {
            let app = App.new(\"x\", 1, 2)
            let cube: Handle = load(\"c.glb\")
            app.run()
         }",
    );
    assert!(
        errs.is_empty(),
        "externs must not false-positive, got {errs:?}"
    );
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
    assert!(
        errs.iter().any(|e| e.contains("if branches")),
        "got {errs:?}"
    );
}

#[test]
fn struct_field_type_checked_for_local_type() {
    let errs = errors(
        "struct P { x: f32, y: f32 }
         fn f() { let p = P { x: true, y: 2.0 } }",
    );
    assert!(
        errs.iter().any(|e| e.contains("struct field")),
        "got {errs:?}"
    );
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
    assert!(
        errs.is_empty(),
        "user Vec3 should shadow the builtin, got {errs:?}"
    );
}

#[test]
fn missing_required_field_is_caught() {
    let errs = errors(
        "struct P { x: f32, y: f32 }
         fn f() { let p = P { x: 1.0 } }", // y missing, no default
    );
    assert!(
        errs.iter().any(|e| e.contains("missing field `y`")),
        "got {errs:?}"
    );
}

#[test]
fn field_with_default_may_be_omitted() {
    // `speed` has a default, so omitting it is fine.
    let errs = errors(
        "component Spinner { speed: f32 = 1.0, name: str }
         fn f() { let s = Spinner { name: \"x\" } }",
    );
    assert!(
        errs.is_empty(),
        "defaulted field omission must be ok, got {errs:?}"
    );
}

#[test]
fn base_spread_satisfies_missing_fields() {
    let errs = errors(
        "struct P { x: f32, y: f32 }
         fn f(orig: P) { let p = P { x: 1.0, ..orig } }",
    );
    assert!(
        errs.is_empty(),
        "..base should cover the rest, got {errs:?}"
    );
}

#[test]
fn unknown_struct_field_is_caught() {
    let errs = errors(
        "struct P { x: f32 }
         fn f() { let p = P { x: 1.0, z: 2.0 } }",
    );
    assert!(
        errs.iter().any(|e| e.contains("no field `z`")),
        "got {errs:?}"
    );
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
        errs.iter()
            .any(|e| e.contains("does not implement trait `Speaker`")),
        "Rock lacks Speaker, got {errs:?}"
    );
}

#[test]
fn condition_must_be_bool_when_known() {
    let errs = errors("fn f() { if 1 { } }");
    assert!(errs.iter().any(|e| e.contains("condition")), "got {errs:?}");
}

// --- callee resolution (E0313) ----------------------------------------------

/// The defect this guards: a call to a function that does not exist used to type
/// check clean, and the backend then replaced the whole enclosing function with
/// a stub returning 0. It has to be an error, wherever it appears.
#[test]
fn unknown_function_is_caught() {
    for src in [
        "fn main() { println(no_such_function_anywhere(1)) }",
        "fn helper() -> i64 { no_such_fn(1) }\nfn main() { println(str(helper())) }",
    ] {
        let errs = errors(src);
        assert!(
            errs.iter().any(|e| e.contains("unknown function")),
            "an undefined callee was accepted in {src:?}, got {errs:?}"
        );
    }
}

/// A near-miss on a real name is the realistic version of the bug: it must not
/// be waved through just because a similarly named function exists.
#[test]
fn a_typo_of_a_real_function_is_caught() {
    let errs = errors("fn tick(dt: f64) -> f64 { dt }\nfn main() { println(str(tikc(1.0))) }");
    assert!(errs.iter().any(|e| e.contains("`tikc`")), "got {errs:?}");
}

/// The leniency that MUST survive: runtime builtins are not `fn` items, and
/// there are several hundred of them. Reporting these would make the compiler
/// unusable, so a sample across the builtin families must stay silent.
#[test]
fn runtime_builtins_are_not_unknown_functions() {
    let errs = errors(
        "fn main() {
            println(str(abs(0 - 3)))
            srand(1)
            let r = rand_int(0, 2)
            r3d_camera(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 60.0)
            phys3d_init(0.0, 0.0 - 9.8, 0.0)
            net_host(45123)
            nav_init(8, 8)
            par_for(r, |i| i)
         }",
    );
    assert!(
        errs.is_empty(),
        "a builtin was reported as unknown, got {errs:?}"
    );
}

/// A bodiless `@extern` import is a declaration, not a definition: it still
/// resolves.
#[test]
fn an_extern_import_resolves() {
    let errs = errors("@extern fn hypot(x: f64, y: f64) -> f64\nfn f() -> f64 { hypot(3.0, 4.0) }");
    assert!(
        errs.is_empty(),
        "an `@extern` import was reported as unknown, got {errs:?}"
    );
}

/// A local holding a closure is a legitimate callee that is not a `fn` item.
#[test]
fn a_local_closure_is_a_legitimate_callee() {
    let errs = errors("fn main() { let f = |x: i64| x + 1\n println(str(f(2))) }");
    assert!(
        errs.is_empty(),
        "calling a local closure was reported, got {errs:?}"
    );
}

/// A shader stage is GPU code: its intrinsics and bound globals have no CPU
/// declaration and must not be resolved against one.
#[test]
fn shader_stage_intrinsics_are_not_unknown_functions() {
    let errs = errors("@fragment fn shade() -> Color { vec4(0.9, 0.2, 0.5, 1.0) }");
    assert!(
        errs.is_empty(),
        "a shader intrinsic was reported as unknown, got {errs:?}"
    );
    // ...and leaving the stage re-arms the check.
    let errs = errors(
        "@fragment fn shade() -> Color { vec4(1.0, 1.0, 1.0, 1.0) }\n\
                       fn main() { println(str(nope(1))) }",
    );
    assert!(
        errs.iter().any(|e| e.contains("`nope`")),
        "the check must resume after a shader stage, got {errs:?}"
    );
}

/// A name brought in by `use` comes from a module we cannot see.
#[test]
fn a_used_import_is_not_an_unknown_function() {
    let errs = errors("use engine::spawn_actor\nfn main() { spawn_actor(1) }");
    assert!(
        errs.is_empty(),
        "a `use`d name was reported as unknown, got {errs:?}"
    );
}
