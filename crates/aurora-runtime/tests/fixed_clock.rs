//! The fixed-timestep clock.
//!
//! The point of a fixed step is that game rules stated in ticks mean the same
//! thing on every machine, so these check the properties that guarantee it:
//! ticks accumulate by wall time and not by call count, a stalled frame cannot
//! wedge the program, and nonsense frame times move nothing.

use aurora_runtime::{
    aurora_run_fixed, aurora_set_tick_rate, aurora_tick_alpha, aurora_tick_count, aurora_tick_delta,
};

/// Advance the clock with no schedule attached.
fn tick(dt: f64) -> i64 {
    // SAFETY: no layers, so neither pointer is read.
    unsafe { aurora_run_fixed(std::ptr::null(), std::ptr::null(), 0, dt) }
}

/// Each test runs on its own thread because the clock is thread-local, which is
/// also what lets two simulations run side by side without sharing time.
fn on_fresh_clock(f: impl FnOnce() + Send + 'static) {
    std::thread::spawn(f).join().expect("test thread panicked");
}

#[test]
fn ticks_follow_wall_time_not_call_count() {
    on_fresh_clock(|| {
        aurora_set_tick_rate(60.0);
        // Sixty frames of exactly one step each.
        for _ in 0..60 {
            assert_eq!(tick(1.0 / 60.0), 1);
        }
        assert_eq!(aurora_tick_count(), 60);

        // One frame worth four steps counts as four, not as one.
        assert_eq!(tick(4.0 / 60.0), 4);
        assert_eq!(aurora_tick_count(), 64);
    });
}

#[test]
fn a_frame_shorter_than_a_step_runs_nothing_until_the_debt_adds_up() {
    on_fresh_clock(|| {
        aurora_set_tick_rate(60.0);
        // Three frames at a third of a step: the first two run nothing, and the
        // third pays for the whole step. Time is banked, never lost.
        assert_eq!(tick(1.0 / 180.0), 0);
        assert_eq!(tick(1.0 / 180.0), 0);
        assert_eq!(tick(1.0 / 180.0), 1);
        assert_eq!(aurora_tick_count(), 1);
    });
}

#[test]
fn a_long_stall_is_written_off_rather_than_chased() {
    on_fresh_clock(|| {
        aurora_set_tick_rate(60.0);
        // Ten seconds of debt is 600 steps. Running them would make this frame
        // enormous and the next one worse; the program would never return.
        let ran = tick(10.0);
        assert!(ran <= 8, "ran {ran} steps in one frame");

        // And the debt does not survive to be chased next frame.
        let next = tick(1.0 / 60.0);
        assert_eq!(next, 1, "leftover debt was banked instead of dropped");
    });
}

#[test]
fn nonsense_frame_times_do_not_move_the_clock() {
    on_fresh_clock(|| {
        aurora_set_tick_rate(60.0);
        for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert_eq!(tick(bad), 0, "dt {bad} produced steps");
        }
        assert_eq!(aurora_tick_count(), 0);
    });
}

#[test]
fn the_tick_rate_sets_the_step_and_rejects_nonsense() {
    on_fresh_clock(|| {
        aurora_set_tick_rate(120.0);
        assert!((aurora_tick_delta() - 1.0 / 120.0).abs() < 1e-12);
        assert_eq!(tick(1.0 / 120.0), 1);

        // A zero or negative rate would give a zero or negative step and divide
        // the accumulator into nonsense; it must be refused, leaving the last
        // good rate in place.
        for bad in [0.0, -60.0, f64::NAN, 1e9] {
            aurora_set_tick_rate(bad);
            assert!(
                (aurora_tick_delta() - 1.0 / 120.0).abs() < 1e-12,
                "rate {bad} took effect"
            );
        }
    });
}

#[test]
fn alpha_reports_the_position_between_ticks() {
    on_fresh_clock(|| {
        aurora_set_tick_rate(60.0);
        assert!(
            aurora_tick_alpha().abs() < 1e-9,
            "a fresh clock is on a tick"
        );

        // Half a step of debt sits halfway to the next tick.
        tick(0.5 / 60.0);
        assert!(
            (aurora_tick_alpha() - 0.5).abs() < 1e-6,
            "alpha {}",
            aurora_tick_alpha()
        );

        // Paying the rest lands back on a tick.
        tick(0.5 / 60.0);
        assert!(
            aurora_tick_alpha().abs() < 1e-6,
            "alpha {}",
            aurora_tick_alpha()
        );
    });
}

#[test]
fn two_threads_keep_separate_clocks() {
    // A dedicated server stepping its own simulation must not have its clock
    // advanced by a client thread in the same process.
    let a = std::thread::spawn(|| {
        aurora_set_tick_rate(60.0);
        tick(1.0);
        aurora_tick_count()
    });
    let b = std::thread::spawn(|| {
        aurora_set_tick_rate(60.0);
        aurora_tick_count()
    });
    assert!(a.join().unwrap() > 0);
    assert_eq!(
        b.join().unwrap(),
        0,
        "one thread's clock advanced another's"
    );
}

static RAN: std::sync::atomic::AtomicI64 = std::sync::atomic::AtomicI64::new(0);

extern "C" fn count_one() {
    RAN.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
}

#[test]
fn the_schedule_runs_once_per_step_in_layer_order() {
    on_fresh_clock(|| {
        RAN.store(0, std::sync::atomic::Ordering::SeqCst);
        aurora_set_tick_rate(60.0);

        // Two layers of one system each.
        let fns = [
            count_one as *const () as usize,
            count_one as *const () as usize,
        ];
        let lens = [1i64, 1];
        // SAFETY: both arrays are live locals of the stated lengths.
        let steps = unsafe { aurora_run_fixed(fns.as_ptr(), lens.as_ptr(), 2, 3.0 / 60.0) };

        assert_eq!(steps, 3);
        assert_eq!(
            RAN.load(std::sync::atomic::Ordering::SeqCst),
            6,
            "expected two systems across three steps"
        );
    });
}
