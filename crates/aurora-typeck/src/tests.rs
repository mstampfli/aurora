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
            let cube = load(\"c.glb\")
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

/// A call to a function a MODULE does not have must be caught by the checker.
///
/// Module items are flattened to `module::name`, and only single-segment callees
/// were validated - so `hud::banner()` with no such function passed `check` and
/// failed later in the backend, pointing at a function rather than a line.
#[test]
fn a_missing_module_function_is_caught() {
    let errs = errors(
        "mod hud { fn draw() { } fn label() { } }
         fn main() { hud::banner() }",
    );
    assert!(
        errs.iter().any(|e| e.contains("has no function `banner`")),
        "a missing module fn must be reported, got {errs:?}"
    );
}

/// ...and a call that DOES exist in the module must stay silent.
#[test]
fn a_present_module_function_is_not_reported() {
    let errs = errors(
        "mod hud { fn draw() { } }
         fn main() { hud::draw() }",
    );
    assert!(
        errs.is_empty(),
        "a real module call must not error, got {errs:?}"
    );
}

/// The guard must not fire on qualified paths that are not module calls at all:
/// an enum variant constructor, or an unknown prefix with no module behind it.
#[test]
fn qualified_paths_that_are_not_module_calls_are_left_alone() {
    let errs = errors(
        "enum Opt { Some(i32), None }
         fn main() { let a = Opt::Some(1) }",
    );
    assert!(
        errs.is_empty(),
        "enum variant construction must not error, got {errs:?}"
    );

    // No function anywhere is qualified with `Thing`, so `Thing` is not known to be
    // a module and the checker must stay lenient rather than guess.
    let errs2 = errors("fn main() { let x = Thing::make() }");
    assert!(
        !errs2.iter().any(|e| e.contains("has no function")),
        "an unknown prefix must not be reported as a module, got {errs2:?}"
    );
}

/// A qualified path used as a VALUE had the same hole, and it is the one a
/// rename produces.
///
/// `mod::CONST` where the const does not exist reached the backend and failed
/// there as "unsupported path expression in JIT" - no line, no column, and only
/// if you ran it. Deleting or renaming a const leaves the dangling reference in
/// some OTHER file, one the author was never editing, so nothing prompts them to
/// look, and the one tool whose job is to answer "does this compile" said yes.
#[test]
fn a_missing_module_const_is_caught() {
    let errs = errors(
        "mod cfg { const SPEED: f64 = 3.2 }
         fn main() { println(str(cfg::REACH)) }",
    );
    assert!(
        errs.iter().any(|e| e.contains("has no value `REACH`")),
        "a missing module const must be reported, got {errs:?}"
    );
}

/// ...and every legitimate shape of the same syntax must stay silent. This is
/// the half that decides whether the guard is usable: a compiler that cries wolf
/// on real code gets its check switched off, and then catches nothing at all.
#[test]
fn qualified_values_that_do_exist_are_left_alone() {
    // The const is really there.
    let errs = errors(
        "mod cfg { const SPEED: f64 = 3.2 }
         fn main() { println(str(cfg::SPEED)) }",
    );
    assert!(errs.is_empty(), "a real module const errored, got {errs:?}");

    // A function of that module, named rather than called.
    let errs = errors(
        "mod cfg { fn tune() -> i64 { 1 } }
         fn main() { let f = cfg::tune }",
    );
    assert!(
        errs.is_empty(),
        "a module fn used as a value errored, got {errs:?}"
    );

    // A unit enum variant: a qualified value that is not a module member.
    let errs = errors(
        "enum Side { Left, Right }
         fn main() { let s = Side::Left }",
    );
    assert!(errs.is_empty(), "an enum variant errored, got {errs:?}");

    // A prefix with nothing behind it is not known to be a module, so the
    // checker stays lenient rather than guessing.
    let errs = errors("fn main() { let x = Whatever::THING }");
    assert!(
        !errs.iter().any(|e| e.contains("has no value")),
        "an unknown prefix was reported as a module, got {errs:?}"
    );

    // One module's const read from inside another, which is the whole reason
    // the syntax exists.
    let errs = errors(
        "mod a { const N: i64 = 2 }
         mod b { fn twice() -> i64 { a::N * 2 } }
         fn main() { println(str(b::twice())) }",
    );
    assert!(
        errs.is_empty(),
        "a cross-module const read errored, got {errs:?}"
    );
}

/// An undefined variable must be caught by the checker, not by the backend.
///
/// This used to pass `check` and then fail as "unknown variable `x` in JIT" - no
/// line, no column, and only if you ran it. A misspelled or out-of-order local is
/// among the easiest mistakes to make in a large program.
#[test]
fn an_undefined_value_is_caught() {
    let errs = errors("fn main() { let a = 1\n let b = a + nope }");
    assert!(
        errs.iter().any(|e| e.contains("cannot find value `nope`")),
        "an undefined value must be reported, got {errs:?}"
    );
}

/// Using a local BEFORE it is declared is the same mistake and must also be caught.
#[test]
fn a_value_used_before_declaration_is_caught() {
    let errs = errors("fn main() { let a = later\n let later = 3 }");
    assert!(
        errs.iter().any(|e| e.contains("cannot find value `later`")),
        "use-before-declaration must be reported, got {errs:?}"
    );
}

/// Everything a bare name may legitimately be must stay silent: a const, a
/// function used as a value, a parameter, a type name, and a declared extern.
#[test]
fn legitimate_bare_names_are_not_reported() {
    let errs = errors(
        "const LIMIT: i32 = 4
         @extern fn ext_thing() -> i32
         fn helper(q: i32) -> i32 { q }
         fn main() {
            let a = LIMIT
            let b = helper
            let c = ext_thing()
            let d = helper(LIMIT)
         }",
    );
    assert!(
        !errs.iter().any(|e| e.contains("cannot find value")),
        "consts, fn values, params and externs must not be reported, got {errs:?}"
    );

    // A closure's parameter and a loop variable are locals too.
    let errs2 = errors("fn main() { let f = |q| q + 1\n for i in 0..3 { let z = i } }");
    assert!(
        !errs2.iter().any(|e| e.contains("cannot find value")),
        "closure params and loop variables must not be reported, got {errs2:?}"
    );
}

/// A local that shadows a PARAMETER of a different type must be warned about.
///
/// This is legal Aurora and usually deliberate, so it is a warning - but the
/// different-type case silently breaks calls. A `let m = h.model[i]` (an i64)
/// inside a function taking `m: Warren` made every later `m` the handle, so a
/// callee expecting a struct pointer received an integer and the program
/// segfaulted with no diagnostic anywhere.
#[test]
fn a_local_shadowing_a_parameter_of_another_type_warns() {
    let src = "struct Warren { n: i32 }
               fn go(m: Warren, i: i32) -> i32 {
                  let m = i
                  m
               }";
    let (module, _) = parse_str(src);
    let diags = check_types(&module);
    assert!(
        diags
            .iter()
            .any(|d| !d.is_error() && d.message.contains("shadows the parameter")),
        "expected a shadowing warning, got {:?}",
        diags.iter().map(|d| d.message.clone()).collect::<Vec<_>>()
    );
    // And it must NOT be an error: shadowing stays legal.
    assert!(
        !diags.iter().any(|d| d.is_error()),
        "shadowing must warn, not fail the build"
    );
}

/// Shadowing with the SAME type is idiomatic and must stay silent.
#[test]
fn shadowing_with_the_same_type_is_silent() {
    let errs = errors("fn go(n: i32) -> i32 { let n = n + 1\n n }");
    assert!(
        errs.is_empty(),
        "same-type shadowing must not complain, got {errs:?}"
    );
}

// --- array length in a diagnostic ---------------------------------------
//
// A length mismatch has to name BOTH lengths. It once said
//
//     expected `[str]`, found `[str; 43]`
//
// which is not a formatting quirk - the expected side had no length because the
// const naming it had not RESOLVED, and an unsized slice is what an unresolved
// length degrades to. So the message that was meant to report a miscount was
// simultaneously the symptom of a compiler bug, and read as neither. Three
// separate passes dropped an array type's length over this project's life; each
// time the report looked like this.
//
// These pin the message rather than the resolution, because the resolution has
// its own tests (`aurora-codegen/tests/const_array_length.rs`) and this is the
// part a human reads.

fn expects_both_lengths(errs: &[String], want: &str, got: &str) {
    assert!(
        errs.iter().any(|e| e.contains(want) && e.contains(got)),
        "the diagnostic must name BOTH lengths - want {want:?} and {got:?}, got {errs:?}"
    );
}

#[test]
fn a_length_mismatch_names_the_literal_length_it_expected() {
    let errs = errors("const T: [i64; 5] = [1, 2, 3]");
    expects_both_lengths(&errs, "[i64; 5]", "; 3]");
}

/// The case that regressed: the length is a CONST, so reporting it at all means
/// the const resolved.
#[test]
fn a_length_mismatch_resolves_a_const_length_before_reporting_it() {
    let errs = errors("const N: i64 = 4\nconst T: [str; N] = [\"a\", \"b\", \"c\"]");
    expects_both_lengths(&errs, "[str; 4]", "[str; 3]");
    assert!(
        !errs.iter().any(|e| e.contains("expected `[str]`")),
        "an unsized `[str]` means the const length was never resolved: {errs:?}"
    );
}

/// And an arithmetic const, which has to be folded first.
#[test]
fn a_length_mismatch_folds_an_arithmetic_const_length() {
    let errs = errors("const N: i64 = 4\nconst W: i64 = N * 2\nconst T: [i64; W] = [1, 2, 3]");
    expects_both_lengths(&errs, "[i64; 8]", "; 3]");
}

/// A correct table is silent, so the tests above are reporting a real mismatch
/// rather than a checker that complains about every array.
#[test]
fn a_correctly_sized_table_is_accepted() {
    assert!(errors("const N: i64 = 3\nconst T: [i64; N] = [1, 2, 3]").is_empty());
}

// --- naming a type that does not exist (E0315) --------------------------------
//
// The leniency contract above is about UNRESOLVED types - a name a module was
// handed by `use` and cannot see the definition of. It was never meant to cover
// a name that resolves to nothing anywhere, and it did: an undefined type became
// an opaque nominal that unified with itself, so `a: Nonexistent` was accepted
// and `fn takes(x: AlsoMissing)` produced no diagnostic at all. The errors that
// did appear read "expected `StillMissing`, found `{integer}`", which asserts
// the missing type exists.
//
// Both sides are pinned here, because a false error would fire on every generic
// parameter and every engine resource.

#[test]
fn an_undefined_type_is_an_error_in_every_position() {
    // One position per line, and each must be reported exactly once.
    for (src, name) in [
        ("struct S { a: Nope }", "Nope"),
        ("fn f(x: Nope) {}", "Nope"),
        ("fn f() -> Nope { 0 }", "Nope"),
        ("fn f() { let x: Nope = 0 }", "Nope"),
        ("struct S { a: [Nope; 4] }", "Nope"),
        ("struct S { a: (Nope, i64) }", "Nope"),
        ("fn f(x: &Nope) {}", "Nope"),
        ("fn f(x: &mut Nope) {}", "Nope"),
        ("struct W<T> { i: T }\nstruct S { a: W<Nope> }", "Nope"),
    ] {
        let errs = errors(src);
        let hits = errs
            .iter()
            .filter(|e| e.contains("unknown type") && e.contains(name))
            .count();
        assert_eq!(hits, 1, "in `{src}` expected one unknown-type error, got {errs:?}");
    }
}

#[test]
fn a_type_parameter_is_not_an_undefined_type() {
    // The first false positive this produced, and the reason `enter_generics`
    // extends rather than replaces: an impl method has to see the impl's
    // parameter as well as its own.
    for src in [
        "fn id<T>(x: T) -> T { x }",
        "struct Pair<T> { a: T, b: T }",
        "struct P<T> { a: T }\nimpl<T> P<T> { fn first(self) -> T { self.a } }",
        "fn two<A, B>(a: A, b: B) -> A { a }",
    ] {
        let errs = errors(src);
        assert!(
            !errs.iter().any(|e| e.contains("unknown type")),
            "`{src}` reported a declared type parameter as unknown: {errs:?}"
        );
    }
}

#[test]
fn the_engine_s_own_types_are_not_undefined() {
    // These are provided by the language rather than the program, and the list
    // that says so lives in `aurora_ast::is_builtin_type` - one list, read by
    // this pass and by `aurora-check`. It was two, and the second was missing
    // Transform, Handle, Option, Result and Time.
    for name in [
        "i64", "f64", "str", "bool", "Vec3", "Quat", "Color", "Mat4", "Transform", "Time", "Tick",
        "Entity", "Handle",
    ] {
        let src = format!("fn f(x: {name}) {{}}");
        let errs = errors(&src);
        assert!(
            !errs.iter().any(|e| e.contains("unknown type")),
            "`{name}` is a language-provided type and was reported unknown: {errs:?}"
        );
    }
}

#[test]
fn one_typo_in_one_place_is_one_error() {
    // A struct field's type is converted by the collection pass and again when
    // the struct is used, so the naive version reported the same typo twice at
    // the same span.
    let errs = errors("struct S { a: Nope }\nfn f() { let s = S { a: 0 } }");
    let hits = errs.iter().filter(|e| e.contains("unknown type")).count();
    assert_eq!(hits, 1, "expected one error for one typo, got {errs:?}");
}

// --- reading a field that is not there (E0301) --------------------------------
//
// `ExprKind::Field` answered `Ty::Error` unconditionally with a note that field
// resolution would come later, so nothing about a dotted access was checked at
// all: `s.nonexistent` type-checked green and failed in the backend as "no field
// `nonexistent` in JIT" - no line, no column, and only if you ran it.

#[test]
fn reading_a_field_that_does_not_exist_is_an_error() {
    let errs = errors("struct S { a: i64 }\nfn f() { let s = S { a: 1 }\n println(str(s.b)) }");
    assert!(
        errs.iter().any(|e| e.contains("no field `b`")),
        "expected a missing-field error, got {errs:?}"
    );
}

#[test]
fn reading_a_field_through_a_reference_still_resolves() {
    // The game passes `&mut Session` everywhere, so a checker that only looked
    // at bare struct types would go quiet on nearly every real access.
    let errs = errors("struct S { a: i64 }\nfn f(s: &mut S) { println(str(s.b)) }");
    assert!(
        errs.iter().any(|e| e.contains("no field `b`")),
        "a field read through &mut was not checked: {errs:?}"
    );
}

#[test]
fn a_field_that_exists_is_not_reported_and_carries_its_type() {
    // Both halves: no false positive, and the field's TYPE flows out - a bool
    // field assigned to an i64 has to be caught by the mismatch, which only
    // works if the access answers something better than `Error`.
    let errs = errors("struct S { a: bool }\nfn f(s: &S) { let x: i64 = s.a }");
    assert!(
        !errs.iter().any(|e| e.contains("no field")),
        "a real field was reported missing: {errs:?}"
    );
    assert!(
        errs.iter().any(|e| e.contains("type mismatch")),
        "the field's type did not reach the let annotation: {errs:?}"
    );
}

#[test]
fn a_qualified_struct_literal_checks_its_fields() {
    // The fourth place that read a path by its last segment only. The flattener
    // mangles another module's struct to `m::S`, so `m::S { bogus: 1 }` missed
    // the table entirely and was accepted, while the identical same-module
    // literal was rejected.
    let errs = errors("mod m { struct S { a: i64 } }\nfn f() { let s = m::S { a: 1, bogus: 2 } }");
    assert!(
        errs.iter().any(|e| e.contains("no field `bogus`")),
        "a qualified struct literal went unchecked: {errs:?}"
    );
}

#[test]
fn a_tuple_index_is_left_alone() {
    // `.0` is a different lookup; judging it with the struct-field table would
    // reject every tuple in the language.
    let errs = errors("fn f() { let t = (1, 2)\n println(str(t.0)) }");
    assert!(
        !errs.iter().any(|e| e.contains("no field")),
        "tuple access was judged as a struct field: {errs:?}"
    );
}

#[test]
fn a_method_call_is_not_a_missing_field() {
    // `a.b(c)` is a Call whose callee is a Field. Judging that Field as a struct
    // member reports every method in the language as missing, which is exactly
    // what happened: the prelude's own examples lit up with `no field `scale` on
    // `Vec2``, `no field `intersects` on `Rect``, `no field `step` on
    // `Particle`` the moment field checking was switched on.
    let errs = errors(
        "struct S { a: i64 }\nimpl S { fn twice(self) -> i64 { self.a * 2 } }\n\
         fn f(s: &S) { println(str(s.twice())) }",
    );
    assert!(
        !errs.iter().any(|e| e.contains("no field")),
        "a method call was judged as a field read: {errs:?}"
    );
}
