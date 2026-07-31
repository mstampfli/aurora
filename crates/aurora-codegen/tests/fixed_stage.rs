//! `stage(FixedUpdate)` systems run on the simulation clock, not the frame.
//!
//! The whole point is that a rule written in ticks means the same thing however
//! long frames take, so these drive `run_systems()` at different frame times and
//! compare the two counts, rather than checking that a system ran at all.

use aurora_parser::parse_str;

/// A fixed system and a frame system counting into separate components, so one
/// program reports both clocks. `dt` is baked in because the JIT entry points
/// used here take only integers.
fn program(dt: &str) -> String {
    format!(
        r#"
component Sim {{ n: i64 }}
component Frame {{ n: i64 }}

system step_sim() stage(FixedUpdate) {{
    for s in query<&mut Sim> {{ s.n += 1 }}
}}

system step_frame() stage(Update) {{
    for f in query<&mut Frame> {{ f.n += 1 }}
}}

// Returns sim_ticks * 1000 + frames, so one call reports both clocks.
fn run(frames: i64) -> i64 {{
    spawn(Sim {{ n: 0 }}, Frame {{ n: 0 }})
    set_tick_rate(60.0)
    set_fixed_dt({dt})
    let mut i = 0
    while i < frames {{ run_systems(); i = i + 1 }}
    let mut sim = 0
    for s in query<&Sim> {{ sim = s.n }}
    let mut fr = 0
    for f in query<&Frame> {{ fr = f.n }}
    sim * 1000 + fr
}}
"#
    )
}

/// Compile and run on a dedicated thread: the simulation clock is per-thread,
/// so each case starts from zero without depending on test ordering.
fn counts(dt: &'static str, frames: i64) -> (i64, i64) {
    std::thread::spawn(move || {
        let src = program(dt);
        let (module, diags) = parse_str(&src);
        assert!(!diags.iter().any(|d| d.is_error()), "parse failed");
        let jit = aurora_codegen::build(&module).expect("must compile natively");
        let packed = jit.call_i64("run", &[frames]).expect("run");
        (packed / 1000, packed % 1000)
    })
    .join()
    .expect("worker panicked")
}

#[test]
fn a_fixed_system_ticks_by_time_and_a_frame_system_by_frame() {
    // Sixty frames of exactly one step each: both clocks agree.
    let (sim, frame) = counts("0.016666666666666666", 60);
    assert_eq!(frame, 60, "the frame system runs once per frame");
    assert_eq!(sim, 60, "one step per frame at the tick rate");
}

#[test]
fn a_slow_frame_runs_the_fixed_schedule_more_than_once() {
    // Ten frames at four steps each. The frame system runs ten times; the
    // simulation still advances forty ticks, which is what keeps a rule written
    // in ticks meaning the same thing on a slower machine.
    let (sim, frame) = counts("0.06666666666666667", 10);
    assert_eq!(frame, 10);
    assert_eq!(sim, 40);
}

#[test]
fn a_fast_frame_does_not_run_the_fixed_schedule_every_time() {
    // Twelve frames at a third of a step: four whole steps come due over the
    // run, so the simulation advances four times while the frame system runs
    // twelve. Sub-step time is banked, not discarded.
    let (sim, frame) = counts("0.005555555555555556", 12);
    assert_eq!(frame, 12);
    assert_eq!(sim, 4);
}

#[test]
fn a_program_with_no_fixed_systems_still_runs_its_frame_schedule() {
    // The fixed dispatch must be inert when nothing declares the stage, rather
    // than costing a call or, worse, swallowing the frame schedule.
    let src = r#"
component Frame { n: i64 }
system step_frame() { for f in query<&mut Frame> { f.n += 1 } }
fn run(frames: i64) -> i64 {
    spawn(Frame { n: 0 })
    let mut i = 0
    while i < frames { run_systems(); i = i + 1 }
    let mut fr = 0
    for f in query<&Frame> { fr = f.n }
    fr
}
"#;
    let (module, diags) = parse_str(src);
    assert!(!diags.iter().any(|d| d.is_error()), "parse failed");
    let jit = aurora_codegen::build(&module).expect("must compile natively");
    assert_eq!(jit.call_i64("run", &[7]).expect("run"), 7);
}
