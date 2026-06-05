//! Client-side prediction & server reconciliation (netcode spec §6).
//!
//! The drift-proofing argument is structural: the *same* Aurora movement
//! function advances state on the predicting client, on the authoritative
//! server, and during reconciliation replay. There is no separate "client
//! formula" to diverge from the server's — which is exactly the bug class this
//! is meant to prevent ([[feedback_consistent_coord_space]]).
//!
//! Here the movement function is an Aurora `fn step(state, input) -> state` run
//! through the interpreter. The [`Predictor`] simulates the client: it predicts
//! locally each tick, and on an authoritative snapshot reconciles by snapping to
//! the server state and replaying unacked inputs through the same `step`.

use aurora_ast::Module;
use aurora_interp::{call_fn, Value};

/// One recorded predicted tick.
#[derive(Clone, Copy, Debug)]
struct Frame {
    tick: u64,
    input: i128,
    predicted: i128,
}

pub struct Predictor<'a> {
    module: &'a Module,
    step: String,
    /// The current predicted state (1-D position, for the demo).
    state: i128,
    tick: u64,
    history: Vec<Frame>,
    /// Highest tick the server has already confirmed (for ignoring stale acks).
    last_acked: u64,
}

impl<'a> Predictor<'a> {
    /// Create a predictor over `module`'s `step` function, starting at `initial`.
    pub fn new(module: &'a Module, step: &str, initial: i128) -> Predictor<'a> {
        Predictor {
            module,
            step: step.to_string(),
            state: initial,
            tick: 0,
            history: Vec::new(),
            last_acked: 0,
        }
    }

    pub fn state(&self) -> i128 {
        self.state
    }

    pub fn tick(&self) -> u64 {
        self.tick
    }

    /// Run `step(state, input)` once via the interpreter.
    fn apply(&self, state: i128, input: i128) -> i128 {
        match call_fn(self.module, &self.step, vec![Value::Int(state), Value::Int(input)]) {
            Ok(Value::Int(n)) => n,
            _ => state, // non-int / error: leave unchanged (lenient)
        }
    }

    /// Advance one tick locally with `input`, recording it for later reconcile.
    pub fn predict(&mut self, input: i128) -> i128 {
        self.tick += 1;
        let next = self.apply(self.state, input);
        self.history.push(Frame { tick: self.tick, input, predicted: next });
        self.state = next;
        next
    }

    /// Apply an authoritative server snapshot: the server says that *after*
    /// `confirmed_tick` the state was `authoritative`. Returns `true` if this was
    /// a misprediction (a correction was applied), `false` if the prediction
    /// already matched (no visible correction — the common case when nothing
    /// diverged, proving same-function = no drift).
    pub fn reconcile(&mut self, confirmed_tick: u64, authoritative: i128) -> bool {
        // Ignore stale or duplicate snapshots for an already-acked tick.
        if confirmed_tick <= self.last_acked {
            return false;
        }
        self.last_acked = confirmed_tick;
        let predicted_at = self.history.iter().find(|f| f.tick == confirmed_tick).map(|f| f.predicted);

        let mispredicted = predicted_at != Some(authoritative);

        if mispredicted {
            // Snap to the authoritative state, then replay every unacked input
            // (tick > confirmed_tick) through the SAME step function.
            let mut s = authoritative;
            for f in self.history.iter_mut().filter(|f| f.tick > confirmed_tick) {
                s = call_fn(self.module, &self.step, vec![Value::Int(s), Value::Int(f.input)])
                    .ok()
                    .and_then(|v| match v {
                        Value::Int(n) => Some(n),
                        _ => None,
                    })
                    .unwrap_or(s);
                f.predicted = s;
            }
            self.state = s;
        }

        // Drop acked history.
        self.history.retain(|f| f.tick > confirmed_tick);
        mispredicted
    }
}

/// The authoritative server: it advances state with the same `step` function.
/// Used in tests to produce snapshots the client reconciles against.
pub fn server_advance(module: &Module, step: &str, initial: i128, inputs: &[i128]) -> i128 {
    let mut s = initial;
    for &input in inputs {
        s = match call_fn(module, step, vec![Value::Int(s), Value::Int(input)]) {
            Ok(Value::Int(n)) => n,
            _ => s,
        };
    }
    s
}

#[cfg(test)]
mod predict_tests {
    use super::*;
    use aurora_parser::parse_str;

    const MOVE: &str = "fn step(pos: i32, input: i32) -> i32 { pos + input }";

    #[test]
    fn matching_prediction_needs_no_correction() {
        // Client and server run the identical `step`; with the same inputs the
        // server confirms exactly what the client predicted — zero drift.
        let (module, _) = parse_str(MOVE);
        let mut client = Predictor::new(&module, "step", 0);
        let inputs = [1, 2, 3];
        for &i in &inputs {
            client.predict(i);
        }
        assert_eq!(client.state(), 6);

        // Server authoritative state after tick 2 = step over inputs[..2] = 3.
        let server_at_2 = server_advance(&module, "step", 0, &inputs[..2]);
        let corrected = client.reconcile(2, server_at_2);
        assert!(!corrected, "identical functions must not mispredict");
        assert_eq!(client.state(), 6);
    }

    #[test]
    fn misprediction_snaps_and_replays() {
        // Simulate divergence (e.g. an unacked server-side force): the server's
        // authoritative state at tick 2 is 10, not the predicted 3.
        let (module, _) = parse_str(MOVE);
        let mut client = Predictor::new(&module, "step", 0);
        for &i in &[1, 2, 3] {
            client.predict(i); // predicts 1, 3, 6
        }

        let corrected = client.reconcile(2, 10);
        assert!(corrected, "divergent server state must trigger a correction");
        // Snap to 10, then replay the unacked tick-3 input (+3) -> 13.
        assert_eq!(client.state(), 13);
    }

    #[test]
    fn reconcile_acks_and_trims_history() {
        let (module, _) = parse_str(MOVE);
        let mut client = Predictor::new(&module, "step", 0);
        for &i in &[5, 5, 5] {
            client.predict(i);
        }
        // Confirm through tick 3: nothing left to replay; state unchanged.
        let corrected = client.reconcile(3, 15);
        assert!(!corrected);
        assert_eq!(client.state(), 15);
        // A subsequent reconcile at an old tick is a no-op (history trimmed).
        assert!(!client.reconcile(3, 15));
    }
}
