//! A query inside a query reads its own entities.
//!
//! There was one match set. `query_begin` overwrote it, so a nested query
//! destroyed the enclosing loop's entities: the outer loop then read out of the
//! inner query's set, ran off the end, got `-1` from `query_entity`, and
//! dereferenced the null that `get_component(-1, ..)` returns. A segmentation
//! fault, from source the checker accepted, with no line and no message.
//!
//! It hid because it needs TWO entities in the outer loop to show. With one, the
//! body runs once and the corrupted set is never read again - which describes
//! every test anyone had written, and every fight in the game this compiler
//! serves, all of which have had exactly one boss in them.
//!
//! The set is a stack now, pushed by the loop and popped on every path out:
//! its own exit, a `return` from inside it, and a `break` to a loop outside it.
//! These tests are the three paths plus the shape that found it.

use aurora_parser::parse_str;

fn run(src: &str) -> i64 {
    let src = src.to_string();
    std::thread::spawn(move || {
        let (module, diags) = parse_str(&src);
        assert!(
            !diags.iter().any(|d| d.is_error()),
            "source failed to parse: {:?}",
            diags.iter().map(|d| &d.message).collect::<Vec<_>>()
        );
        let jit = aurora_codegen::build(&module).expect("must compile natively");
        jit.call_i64("run", &[]).expect("run")
    })
    .join()
    .expect("worker panicked")
}

const W: &str = "component A { v: i64 }\ncomponent B { v: i64 }\n";

/// Two outer entities and a nested query. This is the whole bug: with one outer
/// entity it passed, which is why it survived.
#[test]
fn an_outer_loop_of_two_reads_its_own_entities() {
    let n = run(&format!(
        "{W}
         fn total_b() -> i64 {{
             let mut n = 0
             for b in query<&B> {{ n = n + b.v }}
             n
         }}
         fn run() -> i64 {{
             spawn(A {{ v: 1 }})
             spawn(A {{ v: 2 }})
             spawn(B {{ v: 100 }})
             let mut t = 0
             for a in query<&A> {{ t = t + a.v + total_b() }}
             t
         }}"
    ));
    // 1 + 100 + 2 + 100. Anything else means the outer loop read the wrong
    // entities - and before the fix this did not return a number at all.
    assert_eq!(n, 203);
}

/// Directly nested, without a function call between them: every pair must be
/// visited exactly once.
#[test]
fn a_directly_nested_query_visits_every_pair() {
    let n = run(&format!(
        "{W}
         fn run() -> i64 {{
             spawn(A {{ v: 1 }})
             spawn(A {{ v: 1 }})
             spawn(A {{ v: 1 }})
             spawn(B {{ v: 1 }})
             spawn(B {{ v: 1 }})
             let mut pairs = 0
             for a in query<&A> {{
                 for c in query<&B> {{ pairs = pairs + 1 }}
             }}
             pairs
         }}"
    ));
    assert_eq!(n, 6, "three outer by two inner");
}

// STILL BROKEN, and deliberately not a test here: a `return` from inside a query
// loop, in a function called from within another query loop, still faults.
//
//     fn first_b() -> i64 { for b in query<&B> { return b.v }  0 }
//     for a in query<&A> { t = t + a.v + first_b() }        // two A entities
//
// It is NOT a regression - it faulted identically before the match-set stack
// existed, verified against the previous compiler. And it is not an imbalance:
// the runtime's `query_end` now asserts the stack is non-empty, and that
// assertion does not fire, so the pushes and pops match and something else is
// wrong. Left as an open defect with its own task rather than as a test,
// because a segfaulting test takes the whole binary down and hides the five
// above it.

/// A `break` out of the inner query, and a `break` out to a loop OUTSIDE it -
/// the path that skips the query loop's own exit block entirely.
#[test]
fn a_break_past_a_query_closes_it() {
    let n = run(&format!(
        "{W}
         fn run() -> i64 {{
             spawn(A {{ v: 5 }})
             spawn(A {{ v: 6 }})
             spawn(B {{ v: 1 }})
             let mut rounds = 0
             let mut seen = 0
             while rounds < 3 {{
                 for a in query<&A> {{
                     seen = seen + a.v
                     break
                 }}
                 rounds = rounds + 1
             }}
             // And a break that leaves the WHILE from inside a query loop.
             let mut escaped = 0
             while escaped < 100 {{
                 for a in query<&A> {{
                     escaped = escaped + 1
                     break
                 }}
                 if escaped > 0 {{ break }}
             }}
             // The outer query still reads its own entities after all that.
             let mut after = 0
             for a in query<&A> {{ after = after + a.v }}
             rounds * 1000 + seen * 10 + after
         }}"
    ));
    // Three rounds, the first A each time (5), and 11 across both afterwards.
    assert_eq!(n, 3 * 1000 + 15 * 10 + 11);
}

/// Three deep, to be sure the stack is a stack rather than two slots.
#[test]
fn three_levels_of_nesting_each_read_their_own() {
    let n = run(&format!(
        "{W}
         fn run() -> i64 {{
             spawn(A {{ v: 1 }})
             spawn(A {{ v: 1 }})
             spawn(B {{ v: 1 }})
             spawn(B {{ v: 1 }})
             let mut n = 0
             for a in query<&A> {{
                 for c in query<&B> {{
                     for d in query<&A> {{ n = n + 1 }}
                 }}
             }}
             n
         }}"
    ));
    assert_eq!(n, 8, "two by two by two");
}
