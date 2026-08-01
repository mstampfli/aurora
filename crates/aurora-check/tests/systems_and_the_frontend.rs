//! A system may not reach the frontend.
//!
//! Systems in one stage layer run on worker threads. The world and the
//! simulation subsystems are routed to the thread that owns the program, so a
//! worker sees the program's own - but the window, the framebuffer, the font,
//! the audio mixer and the GPU are not, and never will be. Sharing a window
//! between threads is not a thing to fix.
//!
//! Refused for EVERY system rather than only for the ones that share a layer
//! today, because "happens to share a layer" is not a property anyone can see. A
//! lone system runs inline and works; add an unrelated second system and the
//! first silently starts drawing into a worker's empty framebuffer.
//!
//! The failure this replaces was silent, which is why it is worth a compile
//! error rather than a note in a comment: a worker that cannot see a subsystem
//! does not fail, it reports an empty one, and "nothing there" is a legal answer
//! every caller already handles.

use aurora_parser::parse_str;

fn errors(src: &str) -> Vec<String> {
    let (module, mut diags) = parse_str(src);
    assert!(
        !diags.iter().any(|d| d.is_error()),
        "source failed to parse: {:?}",
        diags.iter().map(|d| &d.message).collect::<Vec<_>>()
    );
    diags.clear();
    let found = aurora_check::check(&module);
    found
        .iter()
        .filter(|d| d.is_error())
        .map(|d| d.message.clone())
        .collect()
}

#[test]
fn a_system_that_draws_is_refused() {
    let errs = errors(
        "component P { x: i64 }
         system paint() stage(Update) {
             for p in query<&mut P> { pixel(p.x, 0, 255, 255, 255) }
         }
         fn main() { run_systems() }",
    );
    assert!(
        errs.iter().any(|e| e.contains("`pixel`")),
        "drawing from a system was allowed: {errs:?}"
    );
}

/// One call deep counts. A helper that draws has drawn.
#[test]
fn a_system_that_draws_through_a_helper_is_refused() {
    let errs = errors(
        "component P { x: i64 }
         fn paint_it(x: i64) { pixel(x, 0, 255, 255, 255) }
         system paint() stage(Update) {
             for p in query<&mut P> { paint_it(p.x) }
         }
         fn main() { run_systems() }",
    );
    assert!(
        errs.iter().any(|e| e.contains("`pixel`")),
        "a system drawing one call deep was allowed: {errs:?}"
    );
}

/// And the simulation is fine, which is the whole point of the distinction:
/// physics, navigation, the random stream and the clock ARE shared with workers,
/// so a system may use them freely.
#[test]
fn a_system_that_only_simulates_is_allowed() {
    let errs = errors(
        "component P { x: i64 }
         system think() stage(Update) {
             for p in query<&mut P> {
                 p.x = nav_find(0, 0, 3, 0) + phys3d_overlap_sphere(0.0, 0.0, 0.0, 1.0)
                     + rand_int(0, 10)
             }
         }
         fn main() { run_systems() }",
    );
    assert!(
        errs.is_empty(),
        "a system doing simulation work was refused: {errs:?}"
    );
}

/// The frame may draw as much as it likes. The rule is about systems, not about
/// the builtins themselves.
#[test]
fn the_frame_may_draw() {
    let errs = errors(
        "component P { x: i64 }
         system think() stage(Update) {
             for p in query<&mut P> { p.x = p.x + 1 }
         }
         fn main() {
             framebuffer(64, 64)
             run_systems()
             for p in query<&P> { pixel(p.x, 0, 255, 255, 255) }
         }",
    );
    assert!(errs.is_empty(), "the frame was refused: {errs:?}");
}
