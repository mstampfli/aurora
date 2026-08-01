//! A system running in a parallel layer can still see the runtime.
//!
//! Systems in one layer run on worker threads. Those threads route the ECS
//! WORLD through `PAR_WORLD`, so components work - and nothing else does,
//! because every other subsystem the runtime owns lives in its own
//! `thread_local!` and a freshly spawned worker's copy is empty.
//!
//! So a system that pathfinds gets "no route" from a grid the main thread built
//! and filled. A system that raycasts gets "nothing there" from a world full of
//! colliders. Not an error, not a crash: the answer a caller gets when the thing
//! genuinely is not there, which is the worst possible failure mode because
//! every caller already handles it.
//!
//! Found from a game, four iterations after the feature was written. Creatures
//! were given navigation and never used it once: `nav_next` called from ordinary
//! code returned a correct route round an obstacle, and the identical call
//! inside a system returned the fallback, in the same run, microseconds apart.
//! The fights all passed, because none of them was fought along a line with
//! anything on it.
//!
//! The tests need TWO systems in the layer. One runs inline on the owning
//! thread - `aurora_run_parallel` says so in as many words - so a single-system
//! layer sees the runtime perfectly and proves nothing.

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

/// A grid built on the main thread is the same grid a system searches.
///
/// Four open cells in a row: a search across them is a path of four. A worker
/// with its own empty `NAV` answers -1, which is the same -1 a caller gets for
/// "there is no way through" - so the creature that asked shrugs and walks
/// straight at the wall.
#[test]
fn a_system_pathfinds_on_the_grid_the_program_built() {
    let n = run("
         component Ask { v: i64 }
         component Other { v: i64 }

         system search() stage(Update) {
             for a in query<&mut Ask> { a.v = nav_find(0, 0, 3, 0) }
         }
         system alongside() stage(Update) {
             for o in query<&mut Other> { o.v = 1 }
         }

         fn run() -> i64 {
             nav_init(4, 1)
             spawn(Ask { v: 0 })
             spawn(Other { v: 0 })
             run_systems()
             let mut r = 0
             for a in query<&Ask> { r = a.v }
             r
         }");
    assert_eq!(
        n, 4,
        "a system searching a four-cell grid got {n}: the worker thread has its own empty NAV"
    );
}

/// The same, for the physics world: a collider the program added is a collider
/// a system can see.
///
/// Overlap rather than a raycast, because it answers with a body handle and a
/// missing world answers -1 just the same.
#[test]
fn a_system_queries_the_physics_world_the_program_built() {
    let n = run("
         component Ask { v: i64 }
         component Other { v: i64 }

         system probe() stage(Update) {
             for a in query<&mut Ask> {
                 a.v = phys3d_overlap_sphere(0.0, 0.0, 0.0, 1.0)
             }
         }
         system alongside() stage(Update) {
             for o in query<&mut Other> { o.v = 1 }
         }

         fn run() -> i64 {
             phys3d_init(0.0, 0.0, 0.0)
             let b = phys3d_add_box(0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0)
             phys3d_step(0.016)
             spawn(Ask { v: 0 - 2 })
             spawn(Other { v: 0 })
             run_systems()
             let mut r = 0
             for a in query<&Ask> { r = a.v }
             // The handle it should have found, or the -1 of an empty world.
             if r == b { return 1 }
             r
         }");
    assert_eq!(
        n, 1,
        "a system probing for a box it can see got {n}: the worker thread has its own empty PHYS3"
    );
}

/// And a layer of ONE still works, which is what made this invisible: every
/// hand-written test of a runtime call from a system had a single system in it.
#[test]
fn a_lone_system_sees_the_runtime_too() {
    let n = run("
         component Ask { v: i64 }

         system search() stage(Update) {
             for a in query<&mut Ask> { a.v = nav_find(0, 0, 3, 0) }
         }

         fn run() -> i64 {
             nav_init(4, 1)
             spawn(Ask { v: 0 })
             run_systems()
             let mut r = 0
             for a in query<&Ask> { r = a.v }
             r
         }");
    assert_eq!(n, 4, "a lone system runs inline and must see everything");
}
