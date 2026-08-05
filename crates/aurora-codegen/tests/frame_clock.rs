//! `frame_dt` is measured once per frame, not once per caller.
//!
//! It used to reset the frame timer on every call. `run_systems` calls it too -
//! that is how the fixed stage learns how much time it owes - so the ordinary
//! shape of a game loop,
//!
//!     let dt = frame_dt()
//!     ...
//!     run_systems()
//!
//! handed the frame's whole delta to the first reader and roughly zero to the
//! second. Played, that is a boss that takes minutes to throw its first attack,
//! stamina that never comes back, and damage that arrives late or not at all.
//!
//! Every other test of the fixed stage pins the step with `set_fixed_dt`, which
//! turns `frame_dt` into a constant and hides this completely. So these do not
//! pin it. They spend real milliseconds and assert against the wall clock, which
//! is the only way the bug is visible at all.
//!
//! The frame's time is spent BEFORE the game reads dt, because that is where a
//! real frame spends it: the present call blocks on vsync, then the next frame
//! asks how long that took. Sleeping after the read instead would hand the
//! elapsed time to `run_systems` and the bug would pass.

use aurora_parser::parse_str;

/// Sleep, read dt, run the schedule, end the frame - the loop a game actually
/// writes. `input_step` is the frame boundary when there is no window to present
/// to. Returns `sim * 1000 + frames`, so one call reports both clocks.
const LOOP_SRC: &str = r#"
component Sim { n: i64 }
component Frame { n: i64 }

system step_sim() stage(FixedUpdate) {
    for s in query<&mut Sim> { s.n += 1 }
}
system step_frame() stage(Update) {
    for f in query<&mut Frame> { f.n += 1 }
}

fn run(frames: i64, ms: i64) -> i64 {
    spawn(Sim { n: 0 }, Frame { n: 0 })
    set_tick_rate(60.0)
    let mut i = 0
    while i < frames {
        sleep_ms(ms)
        let dt = frame_dt()
        if dt <= 0.0 { return 0 - 1 }
        run_systems()
        input_step()
        i = i + 1
    }
    let mut sim = 0
    for s in query<&Sim> { sim = s.n }
    let mut fr = 0
    for f in query<&Frame> { fr = f.n }
    sim * 1000 + fr
}

// Two reads inside one frame must agree: the second caller is `run_systems`, and
// what it is owed is the frame's delta, not what is left of it. Returns how many
// frames out of `frames` agreed.
fn agree(frames: i64, ms: i64) -> i64 {
    let mut same = 0
    let mut i = 0
    while i < frames {
        sleep_ms(ms)
        let a = frame_dt()
        let b = frame_dt()
        if a == b { same = same + 1 }
        input_step()
        i = i + 1
    }
    same
}

// And the boundary has to actually end the frame, or dt freezes at whatever the
// first one measured and the simulation stops tracking real time. Same loop
// without the `input_step`, reported as milliseconds of dt summed over the run.
fn frozen(frames: i64, ms: i64) -> i64 {
    let mut total = 0.0
    let mut i = 0
    while i < frames {
        sleep_ms(ms)
        total = total + frame_dt()
        input_step()
        i = i + 1
    }
    // f64 out through an i64 entry point: milliseconds, rounded down.
    floor(total * 1000.0) as i64
}
"#;

/// Compile and run on a dedicated thread: the frame clock and the simulation
/// clock are both per-thread, so each case starts from zero regardless of the
/// order the test harness happens to run these in.
fn call(f: &'static str, args: Vec<i64>) -> i64 {
    std::thread::spawn(move || {
        let (module, diags) = parse_str(LOOP_SRC);
        assert!(
            !diags.iter().any(|d| d.is_error()),
            "parse failed: {diags:?}"
        );
        let jit = aurora_codegen::build(&module).expect("must compile natively");
        jit.call_i64(f, &args).expect("run")
    })
    .join()
    .expect("worker panicked")
}

/// The regression itself: a game that reads dt before running its systems must
/// not starve them.
///
/// Ten frames of ~20ms is ~200ms of real time, which is ~12 steps at 60Hz. The
/// first frame reports 1/60 rather than 20ms (nothing has been measured yet), and
/// a loaded machine sleeps longer than asked, so the bar is a wide band - but the
/// destructive read scores ZERO here, not eleven. Sleep is never shorter than
/// requested, so the lower bound is the safe side of the slop.
#[test]
fn reading_dt_before_run_systems_does_not_starve_the_fixed_stage() {
    let packed = call("run", vec![10, 20]);
    assert!(packed >= 0, "frame_dt returned a non-positive delta");
    let (sim, frames) = (packed / 1000, packed % 1000);

    // First: the loop ran at all. Without this, a sim of 0 could mean the body
    // never executed and the real assertion below would be measuring nothing.
    assert_eq!(frames, 10, "the frame schedule should run once per frame");

    // ~200ms of real time owed to a 60Hz simulation. The old destructive read
    // delivered ~0.1ms of it in total and this lands on 0.
    assert!(
        sim >= 8,
        "the fixed stage got {sim} steps out of ~12 for ~200ms of real time - \
         it is being starved of the frame's delta"
    );
    // And it must not be running away either: a step per millisecond would mean
    // the delta is being counted more than once.
    assert!(
        sim <= 30,
        "the fixed stage over-ran: {sim} steps for ~200ms"
    );
}

/// Both readers in a frame get the same answer. This is the property itself,
/// stated without reference to any schedule.
#[test]
fn two_reads_in_one_frame_are_the_same_delta() {
    assert_eq!(
        call("agree", vec![6, 10]),
        6,
        "a second read inside the same frame returned a different delta"
    );
}

/// The cached delta is spent at the frame boundary, so a long run still tracks
/// the wall clock. If `input_step` did not clear it, every frame after the first
/// would report the first frame's delta and this would come out near 6 * 16ms
/// (the initial 1/60) instead of near 6 * 30ms.
#[test]
fn the_frame_boundary_spends_the_delta() {
    let ms = call("frozen", vec![6, 30]);
    // Frame 1 reports 1/60 (nothing measured yet), frames 2..6 report ~30ms:
    // ~16 + 5*30 = ~166ms. Frozen at 1/60 it would be ~100ms.
    assert!(
        ms >= 140,
        "summed dt was {ms}ms over ~180ms of sleeping - the frame delta is not \
         being spent at the boundary"
    );
    assert!(ms <= 400, "summed dt was {ms}ms over ~180ms of sleeping");
}
