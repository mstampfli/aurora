//! `break` and `continue` inside a `for .. in query<..>` loop.
//!
//! A query loop is a loop, and it did not push a loop frame - so the two control
//! statements found whatever loop was OUTSIDE it. That failed in two ways, and
//! the quiet one is why this file exists:
//!
//! * At the top level of a system, `continue` was rejected as "used outside of a
//!   loop". Wrong, but loud, and it stopped the build.
//! * Nested inside a `while`, it COMPILED and jumped to the while's step block.
//!   The rest of the query iteration and the rest of the while body were both
//!   skipped, silently, with no diagnostic and no runtime error. `break` left
//!   the outer loop entirely.
//!
//! Exercised through the compiled path, because that is the one games run, and
//! each case on its own thread because the ECS world is per-thread.

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

const MARK: &str = "component Mark { v: i64 }\n";

/// `continue` skips one entity and moves to the next, rather than ending the
/// loop. Three entities, one skipped: the other two must both be reached.
#[test]
fn continue_advances_to_the_next_entity() {
    let n = run(&format!(
        "{MARK}
         fn run() -> i64 {{
             spawn(Mark {{ v: 0 }})
             spawn(Mark {{ v: 1 }})
             spawn(Mark {{ v: 2 }})
             let mut seen = 0
             for m in query<&Mark> {{
                 if m.v == 1 {{ continue }}
                 seen = seen + 1
             }}
             seen
         }}"
    ));
    assert_eq!(n, 2, "`continue` did not resume the query");
}

/// And it really does skip: a write guarded by `continue` must not happen.
#[test]
fn continue_skips_the_rest_of_the_body() {
    let n = run(&format!(
        "{MARK}
         fn run() -> i64 {{
             spawn(Mark {{ v: 0 }})
             spawn(Mark {{ v: 1 }})
             spawn(Mark {{ v: 2 }})
             for m in query<&mut Mark> {{
                 if m.v == 1 {{ continue }}
                 m.v = m.v + 10
             }}
             let mut total = 0
             for m in query<&Mark> {{ total = total + m.v }}
             total
         }}"
    ));
    // 10 + 1 + 12. Anything else means the skip hit the wrong entities.
    assert_eq!(n, 23, "`continue` skipped the wrong work");
}

/// The silent case. `continue` inside a query loop that is itself inside a
/// `while` must advance the QUERY, leaving the while's own body to finish.
///
/// Before the fix this compiled and jumped to the while's step, so the rounds
/// counter still reached its limit and the inner count was wrong - a result that
/// looks entirely plausible unless you know what it should have been.
#[test]
fn continue_inside_a_while_advances_the_query_not_the_while() {
    let n = run(&format!(
        "{MARK}
         fn run() -> i64 {{
             spawn(Mark {{ v: 0 }})
             spawn(Mark {{ v: 1 }})
             spawn(Mark {{ v: 2 }})
             let mut rounds = 0
             let mut seen = 0
             while rounds < 3 {{
                 for m in query<&Mark> {{
                     if m.v == 1 {{ continue }}
                     seen = seen + 1
                 }}
                 rounds = rounds + 1
             }}
             rounds * 100 + seen
         }}"
    ));
    // Three rounds of two entities each.
    assert_eq!(n, 306, "`continue` reached the enclosing while");
}

/// `break` leaves the query and nothing else. The enclosing loop must run its
/// full count, and the query must stop after one entity every time.
#[test]
fn break_leaves_the_query_not_the_enclosing_loop() {
    let n = run(&format!(
        "{MARK}
         fn run() -> i64 {{
             spawn(Mark {{ v: 0 }})
             spawn(Mark {{ v: 1 }})
             spawn(Mark {{ v: 2 }})
             let mut outer = 0
             let mut hits = 0
             while outer < 4 {{
                 for m in query<&Mark> {{
                     hits = hits + 1
                     break
                 }}
                 outer = outer + 1
             }}
             outer * 100 + hits
         }}"
    ));
    // Four rounds, one entity each.
    assert_eq!(n, 404, "`break` left the enclosing while");
}

/// A body whose every path breaks leaves the step block unreachable. It still
/// needs a terminator, or the function does not verify.
#[test]
fn a_body_that_always_breaks_still_compiles() {
    let n = run(&format!(
        "{MARK}
         fn run() -> i64 {{
             spawn(Mark {{ v: 7 }})
             spawn(Mark {{ v: 8 }})
             let mut first = 0 - 1
             for m in query<&Mark> {{
                 first = m.v
                 break
             }}
             first
         }}"
    ));
    assert_eq!(n, 7);
}

/// Nested query loops: the inner one's `continue` must not disturb the outer.
#[test]
fn continue_in_a_nested_query_stays_in_the_inner_one() {
    let n = run(&format!(
        "{MARK}
         fn run() -> i64 {{
             spawn(Mark {{ v: 0 }})
             spawn(Mark {{ v: 1 }})
             let mut pairs = 0
             let mut outer = 0
             for a in query<&Mark> {{
                 outer = outer + 1
                 for c in query<&Mark> {{
                     if c.v == 1 {{ continue }}
                     pairs = pairs + 1
                 }}
             }}
             outer * 100 + pairs
         }}"
    ));
    // Two outer entities, each seeing one inner entity that is not skipped.
    assert_eq!(n, 202, "a nested query's `continue` escaped it");
}
