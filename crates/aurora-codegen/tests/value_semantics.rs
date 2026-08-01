//! An aggregate is a VALUE. Binding one to a new name copies it.
//!
//! It did not. `struct`, `[T; N]`, tuples and `str` are held by pointer, and
//! every binding form except a struct field and an array element simply rebound
//! that pointer - so `let b = a`, `b = a` and a by-value parameter all made two
//! names for one piece of storage, and a write through either showed up in both.
//!
//! What that looks like from the outside, in the order the game hit them:
//!
//!   * `prev = now` in a change detector never reports a change. `now` is a
//!     string concatenation, whose `[ptr, len]` pair lives in a stack slot the
//!     expression site reuses every iteration, so `prev` reads the CURRENT
//!     value and the two are always equal. Four state changes, one reported.
//!   * `let best = items[0]` followed by `items[2].x = -7` moves `best`.
//!   * `fn poke(p: P) { p.x = 1000 }` rewrites the caller's `p`.
//!
//! None of it is visible at the call or in the signature, and none of it fails
//! loudly - the program runs and quietly computes the wrong thing.
//!
//! Sharing is still available; it just has to be written down. `&mut T` on a
//! parameter is the caller's storage, `&mut x` at the call site says the caller
//! agreed, and a method's `self` stays a reference so `c.bump()` can change `c`.
//!
//! Every case below fails on the old codegen and passes on the new one.

use aurora_parser::parse_str;

const SRC: &str = r#"
struct P { x: i64, y: i64 }
struct Bag { v: [i64; 2], n: i64 }
struct Counter { n: i64 }

impl Counter {
    fn bump(self) { self.n += 1 }
}

// --- binding an aggregate copies it -------------------------------------

// A change detector over a string that is rebuilt each iteration. Four
// distinct states, so four changes.
fn str_detector() -> i64 {
    let mut prev = "start"
    let mut changes = 0
    let mut i = 0
    while i < 4 {
        let now = "state-" + str(i)
        if prev != now { changes += 1 }
        prev = now
        i += 1
    }
    changes
}

fn let_struct() -> i64 {
    let mut a = P { x: 1, y: 2 }
    let b = a
    a.x = 99
    b.x
}

fn let_array() -> i64 {
    let mut a: [i64; 3] = [1, 2, 3]
    let b = a
    a[0] = 99
    b[0]
}

fn assign_struct() -> i64 {
    let mut a = P { x: 1, y: 2 }
    let mut b = P { x: 0, y: 0 }
    b = a
    a.x = 99
    b.x
}

// The running-best pattern: `best` must snapshot the element, not follow it.
fn running_best() -> i64 {
    let mut items: [P; 3] = [P { x: 1, y: 0 }, P { x: 2, y: 0 }, P { x: 3, y: 0 }]
    let mut best = items[0]
    let mut i = 1
    while i < 3 {
        if items[i].x > best.x { best = items[i] }
        i += 1
    }
    items[2].x = 0 - 7
    best.x
}

// A const is shared by every reader, so a local copied out of one must not be
// able to write back into it.
const SEED: [i64; 3] = [4, 5, 6]

fn read_seed(i: i64) -> i64 { SEED[i] }

fn const_array() -> i64 {
    let mut a = SEED
    a[0] = 99
    read_seed(0)
}

// `s = s + x` reads s and writes s in the same statement.
fn str_append() -> i64 {
    let mut s = ""
    let mut i = 0
    while i < 3 {
        s = s + str(i)
        i += 1
    }
    len(s)
}

// Two structs built from one array must not share it: struct fields are inline
// storage, so the array is copied in at construction.
fn struct_array_field() -> i64 {
    let src: [i64; 2] = [1, 2]
    let mut one = Bag { v: src, n: 0 }
    let two = Bag { v: src, n: 0 }
    one.v[0] = 42
    two.v[0]
}

// --- a parameter is the callee's own value ------------------------------

fn poke_field(p: P) -> i64 { p.x = 1000; return p.x }

fn param_field() -> i64 {
    let mut a = P { x: 7, y: 0 }
    let r = poke_field(a)
    if r != 1000 { return 0 - 1 }
    a.x
}

fn poke_whole(p: P) -> i64 {
    p = P { x: 1000, y: 1000 }
    return p.x
}

fn param_whole() -> i64 {
    let mut a = P { x: 7, y: 0 }
    let r = poke_whole(a)
    if r != 1000 { return 0 - 1 }
    a.x
}

fn poke_array(v: [i64; 3]) -> i64 { v[0] = 1000; return v[0] }

fn param_array() -> i64 {
    let mut a: [i64; 3] = [7, 8, 9]
    let r = poke_array(a)
    if r != 1000 { return 0 - 1 }
    a[0]
}

// The copy must be OF the argument - a fresh but empty slot would pass every
// case above and break every function that reads its parameters.
fn read_it(p: P) -> i64 { return p.x + p.y }

fn param_read() -> i64 {
    let p = P { x: 3, y: 4 }
    read_it(p)
}

fn take_str(s: str) -> i64 {
    s = s + "!"
    return len(s)
}

fn param_str() -> i64 {
    let mut s = "abc"
    let r = take_str(s)
    if r != 4 { return 0 - 1 }
    len(s)
}

// --- `&mut` shares on purpose -------------------------------------------

fn split(v: i64, out: &mut [i64; 2]) -> i64 {
    out[0] = v / 10
    out[1] = v - (v / 10) * 10
    return 1
}

fn array_out() -> i64 {
    let mut got: [i64; 2] = [0, 0]
    let ok = split(47, &mut got)
    if ok != 1 { return 0 - 1 }
    got[0] * 100 + got[1]
}

fn shift(p: &mut P, dx: i64) -> i64 {
    p.x += dx
    return p.x
}

fn struct_out() -> i64 {
    let mut p = P { x: 5, y: 0 }
    let r = shift(&mut p, 3)
    if r != 8 { return 0 - 1 }
    p.x
}

// A `&mut` handed straight on to another `&mut` is still the same storage.
fn shift_twice(p: &mut P) -> i64 {
    let a = shift(p, 1)
    let b = shift(p, 1)
    return b - a
}

fn forwarded_ref() -> i64 {
    let mut p = P { x: 0, y: 0 }
    let r = shift_twice(&mut p)
    if r != 1 { return 0 - 1 }
    p.x
}

// `&T` shares too; it just promises not to write.
fn total(p: &P) -> i64 { return p.x + p.y }

fn shared_ref() -> i64 {
    let p = P { x: 3, y: 4 }
    total(&p)
}

// And the receiver is a reference, which is the whole reason a method can
// change the thing it was called on.
fn self_is_a_reference() -> i64 {
    let mut c = Counter { n: 0 }
    c.bump()
    c.bump()
    c.n
}
"#;

/// Compile once per case on its own thread, matching the other codegen tests:
/// the runtime's world and clocks are per-thread, so no case can be affected by
/// the order the harness picks.
fn call(f: &'static str) -> i64 {
    std::thread::spawn(move || {
        let (module, diags) = parse_str(SRC);
        assert!(!diags.iter().any(|d| d.is_error()), "parse failed: {diags:?}");
        let jit = aurora_codegen::build(&module).expect("must compile natively");
        jit.call_i64(f, &[]).expect("run")
    })
    .join()
    .expect("worker panicked")
}

#[test]
fn a_change_detector_over_a_rebuilt_string_sees_every_change() {
    // The report that started this. `now` is a concatenation whose [ptr, len]
    // pair lives in a slot the expression site reuses, so an aliasing `prev`
    // reads the current state and scores 1.
    assert_eq!(call("str_detector"), 4);
}

#[test]
fn let_from_a_place_expression_copies() {
    assert_eq!(call("let_struct"), 1, "`let b = a` must not alias a struct");
    assert_eq!(call("let_array"), 1, "`let b = a` must not alias an array");
}

#[test]
fn assignment_from_a_place_expression_copies() {
    assert_eq!(call("assign_struct"), 1);
    assert_eq!(
        call("running_best"),
        3,
        "`best = items[i]` must snapshot the element, not follow it"
    );
}

#[test]
fn a_local_copied_out_of_a_const_cannot_write_back_into_it() {
    assert_eq!(call("const_array"), 4);
}

#[test]
fn a_string_can_be_appended_to_itself() {
    assert_eq!(call("str_append"), 3);
}

#[test]
fn a_struct_copies_an_array_field_in_at_construction() {
    assert_eq!(call("struct_array_field"), 1);
}

#[test]
fn a_parameter_is_the_callees_own_value() {
    assert_eq!(call("param_field"), 7, "`p.x = 1` reached the caller");
    assert_eq!(call("param_whole"), 7, "`p = ...` reached the caller");
    assert_eq!(call("param_array"), 7, "`v[0] = 1` reached the caller");
    assert_eq!(call("param_str"), 3, "`s = s + x` reached the caller");
}

#[test]
fn a_parameter_still_carries_the_argument_in() {
    // The half a "always take a fresh slot" fix would break.
    assert_eq!(call("param_read"), 7);
}

#[test]
fn a_mut_reference_parameter_writes_through_to_the_caller() {
    assert_eq!(call("array_out"), 407, "the out-parameter shape");
    assert_eq!(call("struct_out"), 8);
    assert_eq!(call("forwarded_ref"), 2, "a forwarded &mut is one storage");
}

#[test]
fn a_shared_reference_parameter_reads_the_callers_value() {
    assert_eq!(call("shared_ref"), 7);
}

#[test]
fn a_method_receiver_is_a_reference() {
    // The other side of the parameter rule: if `self` were copied like a
    // parameter, no `impl` method could ever change anything.
    assert_eq!(call("self_is_a_reference"), 2);
}
