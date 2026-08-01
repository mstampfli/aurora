//! `world_clear()` empties the ECS world.
//!
//! Exercised through the compiled path rather than the interpreter, because that
//! is the one games run. Each case gets its own thread: the world is per-thread,
//! so sharing one would make these tests depend on each other in exactly the way
//! `world_clear` exists to prevent.

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

const COMPONENTS: &str = "component A { v: i64 }\ncomponent B { v: i64 }\n";

#[test]
fn clearing_removes_every_entity() {
    let n = run(&format!(
        "{COMPONENTS}
         fn run() -> i64 {{
             spawn(A {{ v: 1 }})
             spawn(A {{ v: 2 }})
             spawn(B {{ v: 3 }})
             world_clear()
             entity_count()
         }}"
    ));
    assert_eq!(n, 0);
}

/// The component storage must go too. An entity count of zero with live
/// component data behind it is the failure mode that looks fine right up until a
/// query walks it.
#[test]
fn clearing_drops_component_storage_so_queries_find_nothing() {
    let n = run(&format!(
        "{COMPONENTS}
         fn run() -> i64 {{
             spawn(A {{ v: 7 }})
             spawn(A {{ v: 8 }})
             world_clear()
             let mut seen = 0
             for a in query<&A> {{ seen = seen + 1 }}
             seen
         }}"
    ));
    assert_eq!(n, 0, "a query after a clear must match nothing");
}

/// A cleared world is usable, not merely empty.
#[test]
fn the_world_works_again_after_a_clear() {
    let n = run(&format!(
        "{COMPONENTS}
         fn run() -> i64 {{
             spawn(A {{ v: 1 }})
             world_clear()
             spawn(A {{ v: 41 }})
             let mut total = 0
             for a in query<&A> {{ total = total + a.v }}
             total * 10 + entity_count()
         }}"
    ));
    assert_eq!(n, 411, "one entity with v=41 after the clear");
}

/// Entity ids keep counting up. An id captured before the clear must resolve to
/// nothing afterwards rather than aliasing a fresh entity - the property that
/// makes a stale handle a findable bug instead of a silent one.
#[test]
fn ids_are_not_reused_after_a_clear() {
    let n = run(&format!(
        "{COMPONENTS}
         fn run() -> i64 {{
             let old = spawn(A {{ v: 1 }})
             world_clear()
             let fresh = spawn(A {{ v: 2 }})
             if fresh == old {{ 1 }} else {{ 0 }}
         }}"
    ));
    assert_eq!(n, 0, "a fresh entity reused the cleared id, so a stale handle now aliases it");
}

/// Despawning a stale id after a clear must be a no-op, not damage to whatever
/// occupies the world now.
#[test]
fn despawning_a_stale_id_after_a_clear_harms_nothing() {
    let n = run(&format!(
        "{COMPONENTS}
         fn run() -> i64 {{
             let old = spawn(A {{ v: 1 }})
             world_clear()
             spawn(A {{ v: 2 }})
             despawn(old)
             entity_count()
         }}"
    ));
    assert_eq!(n, 1, "the stale despawn removed a live entity");
}

/// Clearing an already-empty world is fine, and clearing twice is not different
/// from clearing once.
#[test]
fn clearing_is_idempotent_and_safe_when_empty() {
    let n = run(&format!(
        "{COMPONENTS}
         fn run() -> i64 {{
             world_clear()
             spawn(A {{ v: 1 }})
             world_clear()
             world_clear()
             entity_count()
         }}"
    ));
    assert_eq!(n, 0);
}

/// Systems see the cleared world, so a suite can reset between cases and the
/// schedule still runs against what it just spawned.
#[test]
fn systems_run_against_the_world_left_after_a_clear() {
    let n = run(
        "component A { v: i64 }
         system bump() stage(Update) { for a in query<&mut A> { a.v = a.v + 1 } }
         fn run() -> i64 {
             spawn(A { v: 100 })
             spawn(A { v: 100 })
             run_systems()
             world_clear()
             spawn(A { v: 5 })
             run_systems()
             let mut total = 0
             for a in query<&A> { total = total + a.v }
             total
         }",
    );
    assert_eq!(n, 6, "only the post-clear entity should exist, bumped once");
}
