//! A server-owned non-player entity, over a real socket.
//!
//! Replication used to cover players only, so a boss - which nobody predicts and
//! everybody must agree about - had no channel. These drive a host and a client
//! over loopback UDP and check that what the host decides the boss is doing is
//! what the client sees, because the failure this guards against is silent: the
//! second player watches a statue while the first fights something.
//!
//! Slot meanings here are this test's, standing in for a game's: 0 = which
//! attack is out, 1 = how far into it, 2 = health.

use std::time::{Duration, Instant};

/// Keep every socket this test opens on loopback.
///
/// The host and the client both bind the wildcard `0.0.0.0` by default, which is
/// right for a real session - a loopback-bound host is reachable only from its
/// own machine, and that failure is silent. It is wrong for a test: binding the
/// wildcard raises the OS firewall dialog, and a suite that pops a dialog at the
/// person running it is a suite they stop running.
///
/// Set once per process, before any socket is opened. Both tests want the same
/// value, so `Once` rather than per-test bookkeeping.
fn loopback_only() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| std::env::set_var("AURORA_NET_BIND", "127.0.0.1"));
}

/// Pump both sides until `done` holds, or give up. UDP over loopback is quick
/// but not instant, and a fixed sleep would be either flaky or slow.
fn pump_until(step: &mut impl FnMut(), done: impl Fn() -> bool, what: &str) {
    let deadline = Instant::now() + Duration::from_secs(5);
    while Instant::now() < deadline {
        step();
        if done() {
            return;
        }
        std::thread::sleep(Duration::from_millis(5));
    }
    panic!("timed out waiting for {what}");
}

/// The host and a client on loopback, each in its own thread because the session
/// is thread-local. Communication is by channel so the assertions run on the
/// client's side where its view lives.
#[test]
fn a_boss_the_host_owns_is_seen_by_a_client() {
    use std::sync::mpsc;

    loopback_only();
    let port = 46101u16;
    let (to_host, host_rx) = mpsc::channel::<u8>();
    let (host_ready, ready_rx) = mpsc::channel::<()>();

    let host = std::thread::spawn(move || {
        aurora_runtime::aurora_net_host(port as i64);
        // One object: the boss. Two metres tall, so a swing meets it where it
        // looks rather than only near its centre.
        aurora_runtime::aurora_net_set_object_count(1);
        aurora_runtime::aurora_net_set_object(0, 0.0, 0.0, -7.0);
        aurora_runtime::aurora_net_set_object_size(0, 0.6, 1.0);
        aurora_runtime::aurora_net_set_object_state(0, 0, -1.0); // idle
        aurora_runtime::aurora_net_set_object_state(0, 1, 0.0);
        aurora_runtime::aurora_net_set_object_state(0, 2, 1200.0);
        let _ = host_ready.send(());

        let mut phase = 0.0f64;
        loop {
            // The boss winds up WITHOUT moving: the case that broke change
            // detection which only watched the pose.
            if host_rx.try_recv().is_ok() {
                return;
            }
            aurora_runtime::aurora_net_set_object_state(0, 0, 0.0); // attack 0 out
            aurora_runtime::aurora_net_set_object_state(0, 1, phase);
            phase += 1.0;
            aurora_runtime::aurora_net_update(1.0 / 60.0);
            std::thread::sleep(Duration::from_millis(4));
        }
    });

    ready_rx.recv_timeout(Duration::from_secs(5)).expect("host started");

    let host_name = "127.0.0.1";
    unsafe {
        aurora_runtime::aurora_net_join(host_name.as_ptr(), host_name.len() as i64, port as i64);
    }
    let mut step = || {
        // fwd, strafe, yaw, jump, dt - the shape net_send_input takes as an array.
        let input = [0.0f64, 0.0, 0.0, 0.0, 1.0 / 60.0];
        unsafe {
            aurora_runtime::aurora_net_send_input(input.as_ptr(), input.len() as i64);
        }
        aurora_runtime::aurora_net_update(1.0 / 60.0);
    };

    pump_until(&mut step, || aurora_runtime::aurora_net_object_count() == 1,
               "the client to learn the boss exists");

    // Its position crossed intact.
    assert!(
        (aurora_runtime::aurora_net_object_z(0) - (-7.0)).abs() < 0.01,
        "the client should see the boss where the host put it, got z={}",
        aurora_runtime::aurora_net_object_z(0)
    );

    // And so did the state that is NOT its pose - the telegraph.
    pump_until(&mut step, || aurora_runtime::aurora_net_object_state(0, 1) > 2.0,
               "the boss's windup phase to reach the client");
    assert_eq!(
        aurora_runtime::aurora_net_object_state(0, 0),
        0.0,
        "the client must see which attack is out"
    );
    assert_eq!(
        aurora_runtime::aurora_net_object_state(0, 2),
        1200.0,
        "and its health, untouched"
    );

    let _ = to_host.send(0);
    let _ = host.join();
}

/// A stationary boss's state must arrive FRESH, not merely eventually.
///
/// This measures how many distinct values the client actually observes rather
/// than whether the last one ever turns up. The distinction matters and was
/// nearly missed: objects are also re-sent on a keyframe, so even change
/// detection that watched only the pose would deliver the state in the end. What
/// it would not do is deliver it in time - keyframes are 30 ticks apart, longer
/// than the 24-tick window the design requires an attack to be readable for, so
/// a remote player could watch a windup begin after the blade had already
/// landed.
///
/// An "did it eventually arrive" test passes against that bug. Counting
/// distinct observations does not.
#[test]
fn a_stationary_objects_state_streams_rather_than_waiting_for_keyframes() {
    use std::sync::mpsc;

    loopback_only();
    let port = 46102u16;
    let (to_host, host_rx) = mpsc::channel::<u8>();
    let (host_ready, ready_rx) = mpsc::channel::<()>();

    let host = std::thread::spawn(move || {
        aurora_runtime::aurora_net_host(port as i64);
        aurora_runtime::aurora_net_set_object_count(1);
        // Never moved after this line.
        aurora_runtime::aurora_net_set_object(0, 4.0, 0.0, 4.0);
        let _ = host_ready.send(());

        let mut posture = 0.0f64;
        loop {
            if host_rx.try_recv().is_ok() {
                return;
            }
            posture += 1.0;
            aurora_runtime::aurora_net_set_object_state(0, 3, posture);
            aurora_runtime::aurora_net_update(1.0 / 60.0);
            std::thread::sleep(Duration::from_millis(4));
        }
    });

    ready_rx.recv_timeout(Duration::from_secs(5)).expect("host started");
    let host_name = "127.0.0.1";
    unsafe {
        aurora_runtime::aurora_net_join(host_name.as_ptr(), host_name.len() as i64, port as i64);
    }
    let mut step = || {
        // fwd, strafe, yaw, jump, dt - the shape net_send_input takes as an array.
        let input = [0.0f64, 0.0, 0.0, 0.0, 1.0 / 60.0];
        unsafe {
            aurora_runtime::aurora_net_send_input(input.as_ptr(), input.len() as i64);
        }
        aurora_runtime::aurora_net_update(1.0 / 60.0);
    };

    pump_until(&mut step, || aurora_runtime::aurora_net_object_count() == 1,
               "the client to learn the object exists");

    // Watch for a while and count how many different values actually landed.
    //
    // Counting distinct arrivals rather than checking that the last one turned
    // up eventually. Objects are ALSO re-sent on a keyframe every 30 ticks, so
    // "did it arrive" is satisfied even by change detection that ignores state
    // entirely - and a boss telegraph delivered twice a second is a lie, since
    // 30 ticks is longer than the 24-tick window an attack must stay readable
    // for. Freshness is the property worth asserting, so freshness is measured.
    let mut seen = std::collections::BTreeSet::new();
    for _ in 0..200 {
        step();
        let v = aurora_runtime::aurora_net_object_state(0, 3);
        seen.insert(v.to_bits());
        std::thread::sleep(Duration::from_millis(3));
    }

    // Measured at ~155 distinct values over ~660 ms, against 5 keyframes in the
    // same window. Fifty is far above anything keyframes alone could produce and
    // far below the observed rate, so this catches a regression to keyframe-only
    // delivery without failing on a slow machine.
    assert!(
        seen.len() >= 50,
        "a stationary object's state must stream, not arrive on keyframes: \
         saw only {} distinct values",
        seen.len()
    );

    let _ = to_host.send(0);
    let _ = host.join();
}
