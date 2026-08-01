//! `after`/`before` decide execution order, not merely layer membership.
//!
//! These run real programs and read back the order the systems actually executed,
//! because that is the claim the annotation makes. An earlier scheduler split
//! layers in declaration order and treated `after` only as a hint to split, so
//! `a after(b)` with `a` declared first ran `a` first - the annotation was
//! accepted, the checker credited it, and the runtime did the opposite. Nothing
//! caught it, because nothing asserted on the order itself.
//!
//! Each system folds its own digit into a shared accumulator, so the final number
//! spells the execution order left to right.

use aurora_parser::parse_str;

/// Compile and run `src`, returning the order digits that `run` reports.
///
/// On a dedicated thread because the ECS world and simulation clock are
/// per-thread: sharing them would make one test's result depend on another's.
fn order(src: String) -> i64 {
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

/// A program whose systems are declared in `decls` order and each append a digit.
fn program(decls: &str) -> String {
    format!(
        r#"
component Log {{ v: i64 }}
{decls}
fn run() -> i64 {{
    spawn(Log {{ v: 0 }})
    run_systems()
    let mut out = 0
    for l in query<&Log> {{ out = l.v }}
    out
}}
"#
    )
}

fn sys(name: &str, sched: &str, digit: i64) -> String {
    format!(
        "system {name}() stage(Update) {sched} {{ \
           for l in query<&mut Log> {{ l.v = l.v * 10 + {digit} }} \
         }}\n"
    )
}

#[test]
fn after_runs_the_named_system_first_even_when_declared_later() {
    // `first` is declared first but claims to run after `second`.
    let src = program(&format!(
        "{}{}",
        sys("first", "after(second)", 1),
        sys("second", "", 2)
    ));
    assert_eq!(
        order(src),
        21,
        "`after(second)` must put second first; 12 means the annotation was ignored"
    );
}

#[test]
fn before_runs_the_named_system_last_even_when_declared_earlier() {
    let src = program(&format!(
        "{}{}",
        sys("early", "", 1),
        sys("late", "before(early)", 2)
    ));
    assert_eq!(
        order(src),
        21,
        "`late before(early)` must run late first, whatever the declaration order"
    );
}

#[test]
fn a_chain_orders_transitively_against_declaration_order() {
    // Declared a, b, c; ordered c, b, a. Every edge points backwards, so a
    // scheduler that leans on declaration order gets this exactly reversed.
    let src = program(&format!(
        "{}{}{}",
        sys("a", "after(b)", 1),
        sys("b", "after(c)", 2),
        sys("c", "", 3)
    ));
    assert_eq!(order(src), 321, "the chain c -> b -> a must run in that order");
}

#[test]
fn declaration_order_still_decides_between_unordered_conflicting_systems() {
    // No annotations: the two conflict, so they cannot share a layer, and the
    // tie-break is declaration order. This is the case the ranking must leave
    // alone - reordering it would silently change existing programs.
    let src = program(&format!("{}{}", sys("one", "", 1), sys("two", "", 2)));
    assert_eq!(order(src), 12);
}

#[test]
fn an_ordering_is_honoured_across_a_longer_gap() {
    // The ordered pair is separated by an unrelated system, so the "split the
    // current layer" behaviour alone cannot produce the right answer.
    let src = program(&format!(
        "{}{}{}",
        sys("last", "after(mid)", 1),
        sys("filler", "", 2),
        sys("mid", "", 3)
    ));
    let got = order(src);
    let (a, b) = (digits(got, 1), digits(got, 3));
    assert!(
        b < a,
        "mid (3) must precede last (1), got {got} - `after` is not reaching across the gap"
    );
}

/// Position of `digit` within `n`, left to right.
fn digits(n: i64, digit: i64) -> usize {
    let s = n.to_string();
    s.find(char::from_digit(digit as u32, 10).unwrap())
        .unwrap_or_else(|| panic!("digit {digit} missing from execution order {n}"))
}
