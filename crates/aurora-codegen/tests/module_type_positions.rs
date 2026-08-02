//! A module-local type name has to survive EVERY nested type position.
//!
//! The flattener mangles a module's items - `Session` declared in `frame`
//! becomes `frame::Session` - and then rewrites the type positions that name it.
//! It has missed a position six separate times, and every miss was silent until
//! something far away broke:
//!
//!   * an array ELEMENT (`[Actor; 9]`) - field access failed with "no field"
//!   * an array LENGTH (`[str; CLIP_COUNT]`) - the array silently became empty
//!   * a tuple member, a fn parameter/return, a `dyn Trait`
//!   * and a REFERENCE (`&mut Session`), which is what this file was added for
//!
//! The reference case is the nastiest of the set because it looks like a bug
//! about `&mut`. A by-value parameter of the very same type goes through the
//! Path arm and is rewritten correctly, so `fn read(s: Session)` compiles and
//! `fn step(s: &mut Session)` does not - which sends you looking at reference
//! handling in codegen instead of at the one match that did not list `Ref`.
//!
//! The real fix is that `rewrite_type` no longer has a `_` arm. A wildcard makes
//! "a type position nobody thought about" look exactly like "a type position
//! with nothing to do", which is how the same bug shipped six times.

use aurora_parser::parse_str;

/// A struct declared INSIDE a module, reached through every wrapper that can
/// hold a type. Each function returns a number that is wrong (or fails to
/// compile at all) if the name was not mangled in that position.
const SRC: &str = r#"
mod inner {
    struct Leaf { a: i64, b: i64 }

    fn leaf() -> Leaf { Leaf { a: 3, b: 4 } }
}

mod m {
    struct Session {
        arena: i64,
        room: inner::Leaf,
        held: [i64; 3],
        body: i64,
    }

    fn open() -> Session {
        Session { arena: 7, room: inner::leaf(), held: [1, 2, 3], body: 10 }
    }

    // BY VALUE - this always worked, and is here so a regression that breaks
    // everything is distinguishable from one that breaks only references.
    fn by_value(s: Session) -> i64 { s.body }

    // BY MUTABLE REFERENCE - the case that failed with "no field `body` in JIT".
    fn by_ref(s: &mut Session) -> i64 {
        s.body = s.body + 1
        s.body
    }

    // BY SHARED REFERENCE.
    fn by_shared_ref(s: &Session) -> i64 { s.arena }

    // A reference to a struct that itself came from ANOTHER module, so the
    // rewrite has to reach a qualified path behind a reference too.
    fn leaf_by_ref(l: &inner::Leaf) -> i64 { l.a + l.b }

    // A nested field and an array element, read through a reference.
    fn deep_by_ref(s: &mut Session) -> i64 { s.room.a + s.held[2] }

    fn drive() -> i64 {
        let mut s = open()
        let bumped = by_ref(&mut s)
        // Read the SAME value back by value: if the reference wrote to a copy,
        // this is 10 and not 11, which no "does it compile" check would catch.
        let seen = by_value(s)
        bumped * 100 + seen
    }

    fn drive_shared() -> i64 {
        let s = open()
        by_shared_ref(&s)
    }

    fn drive_leaf() -> i64 {
        let l = inner::leaf()
        leaf_by_ref(&l)
    }

    fn drive_deep() -> i64 {
        let mut s = open()
        deep_by_ref(&mut s)
    }
}

fn drive() -> i64 { m::drive() }
fn drive_shared() -> i64 { m::drive_shared() }
fn drive_leaf() -> i64 { m::drive_leaf() }
fn drive_deep() -> i64 { m::drive_deep() }
"#;

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
fn a_module_struct_survives_behind_a_mutable_reference() {
    // 11 from the call, 11 read back afterwards: the write went through the
    // reference to the caller's value, not to a copy.
    assert_eq!(
        call("drive"),
        1111,
        "`&mut Session` inside a module lost the mangled name, or wrote to a copy"
    );
}

#[test]
fn a_module_struct_survives_behind_a_shared_reference() {
    assert_eq!(call("drive_shared"), 7);
}

#[test]
fn a_qualified_type_survives_behind_a_reference() {
    // `&inner::Leaf` - already qualified, and still has to be reachable.
    assert_eq!(call("drive_leaf"), 7);
}

#[test]
fn a_nested_field_and_an_array_element_read_through_a_reference() {
    // room.a is 3, held[2] is 3. A dropped array length would make the index
    // panic instead of answering 6.
    assert_eq!(call("drive_deep"), 6);
}
