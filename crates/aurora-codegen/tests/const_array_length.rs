//! An array's length may be a const, and it has to survive every pass.
//!
//! The length of an array TYPE is an expression hiding inside a type, and each
//! pass that walks types has forgotten it in turn:
//!
//!   * `[expr; N]` as an INITIALISER built a zero-length array, so the first
//!     read panicked with "index 0 out of bounds (length 0)" at the read rather
//!     than at the declaration.
//!   * `[T; N]` as a TYPE resolved N only when it was a literal; a const name
//!     fell to 0 in codegen and to an unsized `[T]` in the type checker.
//!   * The module flattener rewrote the ELEMENT type and dropped the length, so
//!     a table that worked in one file broke the moment it moved into a module:
//!     the const was mangled to `m::N` and the type still said `N`.
//!
//! Every one of them failed silently, which is why they were found three
//! separate times, by three different symptoms, none of them the declaration.
//! These lock all three shut - including the one that only appears with a `mod`.

use aurora_parser::parse_str;

/// Sum an array so a wrong length shows up as a wrong number rather than a
/// panic, and report the length separately via `len`.
const SRC: &str = r#"
const N: i64 = 4
const WIDE: i64 = N * 2

// A const in a type position.
const TABLE: [i64; N] = [10, 20, 30, 40]

// An arithmetic const in a type position.
const BIG: [i64; WIDE] = [1, 1, 1, 1, 1, 1, 1, 1]

fn table_len() -> i64 { len(TABLE) }
fn table_sum() -> i64 {
    let mut s = 0
    let mut i = 0
    while i < len(TABLE) { s = s + TABLE[i]; i = i + 1 }
    s
}
fn big_len() -> i64 { len(BIG) }

// A const count in an array REPEAT initialiser.
fn repeat_len() -> i64 {
    let v: [i64; N] = [7; N]
    let mut s = 0
    let mut i = 0
    while i < len(v) { s = s + v[i]; i = i + 1 }
    s
}

// A local declared with a const length, filled and read back.
fn local_len() -> i64 {
    let mut v: [i64; N] = [0, 0, 0, 0]
    let mut i = 0
    while i < N { v[i] = i * i; i = i + 1 }
    v[3]
}
"#;

/// The same tables, one module deep. This is the case the flattener broke: it
/// mangles `N` to `m::N` on the declaration and used to leave the length in
/// `[i64; N]` saying bare `N`, which then resolved to nothing.
const MOD_SRC: &str = r#"
mod m {
    const N: i64 = 4
    const TABLE: [i64; N] = [10, 20, 30, 40]
    const NAMES: [str; N] = ["a", "b", "c", "d"]

    fn table_len() -> i64 { len(TABLE) }
    fn table_sum() -> i64 {
        let mut s = 0
        let mut i = 0
        while i < len(TABLE) { s = s + TABLE[i]; i = i + 1 }
        s
    }
    fn name_len() -> i64 { len(NAMES) }
    fn third_name_len() -> i64 { len(NAMES[2]) }
}

fn table_len() -> i64 { m::table_len() }
fn table_sum() -> i64 { m::table_sum() }
fn name_len() -> i64 { m::name_len() }
fn third_name_len() -> i64 { m::third_name_len() }
"#;

fn call_in(src: &'static str, f: &'static str) -> i64 {
    std::thread::spawn(move || {
        let (module, diags) = parse_str(src);
        assert!(!diags.iter().any(|d| d.is_error()), "parse failed: {diags:?}");
        let jit = aurora_codegen::build(&module).expect("must compile natively");
        jit.call_i64(f, &[]).expect("run")
    })
    .join()
    .expect("worker panicked")
}

#[test]
fn a_const_names_an_array_length_in_a_type_position() {
    assert_eq!(call_in(SRC, "table_len"), 4, "`[i64; N]` lost its length");
    assert_eq!(call_in(SRC, "table_sum"), 100, "the table did not survive");
    assert_eq!(call_in(SRC, "big_len"), 8, "`N * 2` did not fold");
    assert_eq!(call_in(SRC, "local_len"), 9, "a local with a const length");
}

#[test]
fn a_const_counts_an_array_repeat() {
    // `[7; N]` used to build a ZERO-length array.
    assert_eq!(call_in(SRC, "repeat_len"), 28);
}

#[test]
fn a_const_length_survives_module_flattening() {
    // The whole point: identical tables, one `mod` deep.
    assert_eq!(call_in(MOD_SRC, "table_len"), 4);
    assert_eq!(call_in(MOD_SRC, "table_sum"), 100);
    assert_eq!(call_in(MOD_SRC, "name_len"), 4, "a [str; N] table in a module");
    assert_eq!(call_in(MOD_SRC, "third_name_len"), 1, "and its contents");
}

/// Every OTHER position a length can appear in.
///
/// The three bugs above were each fixed where their symptom appeared, and the
/// file above tests exactly those positions. Two more were left: a STRUCT FIELD
/// and a RETURN TYPE both silently became length ZERO, because codegen lowered
/// struct layouts and function signatures BEFORE it filled its const-length
/// table, so those positions looked a name up in an empty map and took the
/// `unwrap_or(0)`. Declarations are lowered after the fill, which is exactly why
/// they worked and these did not.
///
/// It cost a fourth discovery, in game code that was then forced to carry
/// literal lengths and hand-written asserts as a workaround. So this covers
/// every position a length can appear in, not only the ones known to have
/// broken - the whole point of the family is that the next one is somewhere
/// nobody has looked.
const POSITIONS: &str = r#"
const N: i64 = 3

struct Bag {
    v: [i64; N],
}

fn from_return() -> [i64; N] {
    let mut a: [i64; N] = [7, 8, 9]
    a
}

fn through_param(a: [i64; N]) -> i64 {
    len(a)
}

fn nested() -> i64 {
    let g: [[i64; N]; N] = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    len(g) * len(g[0])
}

fn return_len() -> i64 { len(from_return()) }
fn return_sum() -> i64 {
    let r = from_return()
    r[0] + r[1] + r[2]
}
fn field_len() -> i64 {
    let b = Bag { v: [1, 2, 3] }
    len(b.v)
}
fn field_sum() -> i64 {
    let b = Bag { v: [4, 5, 6] }
    b.v[0] + b.v[1] + b.v[2]
}
fn param_len() -> i64 { through_param([1, 2, 3]) }
fn nested_len() -> i64 { nested() }
"#;

#[test]
fn a_const_length_survives_a_return_type() {
    assert_eq!(call_in(POSITIONS, "return_len"), 3, "a returned [i64; N] lost its length");
    assert_eq!(call_in(POSITIONS, "return_sum"), 24, "its contents were dropped");
}

#[test]
fn a_const_length_survives_a_struct_field() {
    assert_eq!(call_in(POSITIONS, "field_len"), 3, "a [i64; N] field lost its length");
    assert_eq!(call_in(POSITIONS, "field_sum"), 15, "its contents were dropped");
}

#[test]
fn a_const_length_survives_a_parameter_and_a_nested_array() {
    assert_eq!(call_in(POSITIONS, "param_len"), 3, "a [i64; N] parameter");
    assert_eq!(call_in(POSITIONS, "nested_len"), 9, "[[i64; N]; N]");
}
