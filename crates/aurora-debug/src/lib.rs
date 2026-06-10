//! Native source-level debugger for Aurora.
//!
//! Unlike an interpreter-based debugger (far too slow), this compiles the program
//! to **machine code** with lightweight instrumentation: the codegen emits calls
//! to the runtime debugger before each statement and after each scalar binding.
//! The program runs at native speed; the hooks only fire bookkeeping. A debugger
//! can therefore set breakpoints by source line, single-step, and read locals
//! while the real compiled code executes.
//!
//! * [`debug_trace`] runs to completion recording a [`Stop`] at each pause —
//!   deterministic and unit-testable.
//! * [`debug_interactive`] drives a stdin REPL (`step`/`continue`/`quit`).

use std::collections::HashSet;
use std::io::{BufRead, Write};

pub use aurora_runtime::{DbgCmd, DbgVal, Stop};

/// Parse, check, and compile `src` with debug instrumentation, then run `main`,
/// recording a stop at each breakpoint line (and at every statement if `step`).
/// Returns the ordered trace of stops.
pub fn debug_trace(src: &str, breakpoints: &[u32], step: bool) -> Result<Vec<Stop>, String> {
    let jit = compile_debug(src)?;
    aurora_runtime::dbg_reset(breakpoints.iter().copied().collect::<HashSet<_>>(), step);
    jit.call_i64("main", &[]).map_err(|e| format!("runtime error: {e}"))?;
    Ok(aurora_runtime::dbg_take_stops())
}

/// Run `src` under an interactive debugger reading commands from stdin. Stops at
/// each breakpoint (or every line if `breakpoints` is empty); at each stop prints
/// the line and locals and waits for: `s`tep, `c`ontinue, or `q`uit.
pub fn debug_interactive(src: &str, breakpoints: &[u32]) -> Result<(), String> {
    let jit = compile_debug(src)?;
    let step_all = breakpoints.is_empty();
    aurora_runtime::dbg_reset(breakpoints.iter().copied().collect::<HashSet<_>>(), step_all);
    aurora_runtime::dbg_set_handler(Box::new(|stop: &Stop| {
        let vars = if stop.vars.is_empty() {
            "(no locals)".to_string()
        } else {
            stop.vars.iter().map(|(n, v)| format!("{n} = {v}")).collect::<Vec<_>>().join(", ")
        };
        println!("\n⏸  line {}: {vars}", stop.line);
        print!("(adbg) [s]tep / [c]ontinue / [q]uit > ");
        let _ = std::io::stdout().flush();
        let mut line = String::new();
        if std::io::stdin().lock().read_line(&mut line).unwrap_or(0) == 0 {
            return DbgCmd::Continue; // EOF → let it run out
        }
        match line.trim() {
            "c" | "continue" => DbgCmd::Continue,
            "q" | "quit" => DbgCmd::Quit,
            _ => DbgCmd::Step,
        }
    }));
    println!("aurora native debugger — running `main`");
    jit.call_i64("main", &[]).map_err(|e| format!("runtime error: {e}"))?;
    println!("program exited.");
    Ok(())
}

fn compile_debug(src: &str) -> Result<aurora_codegen::Jit, String> {
    let (module, mut diags) = aurora_parser::parse_str(src);
    diags.extend(aurora_check::check(&module));
    diags.extend(aurora_typeck::check_types(&module));
    if diags.iter().any(|d| d.is_error()) {
        return Err("source has errors; fix them before debugging".to_string());
    }
    let jit = aurora_codegen::build_debug(&module, src)?;
    if !jit.compiled("main") {
        return Err("`main` did not compile to native code".to_string());
    }
    Ok(jit)
}

#[cfg(test)]
mod tests {
    use super::*;

    const PROG: &str = "fn main() {
    let mut total = 0
    for i in 1..=3 {
        total = total + i
    }
    println(total)
}";

    #[test]
    fn stepping_records_native_execution_trace() {
        // Single-step: every statement should produce a stop, in execution order.
        let stops = debug_trace(PROG, &[], true).expect("debug run failed");
        assert!(!stops.is_empty(), "stepping should record stops");
        // The loop body (`total = total + i`) runs 3 times.
        let body_line = 4; // `total = total + i`
        let body_hits = stops.iter().filter(|s| s.line == body_line).count();
        assert_eq!(body_hits, 3, "loop body should be hit once per iteration, got {body_hits}");
    }

    #[test]
    fn breakpoint_reports_changing_locals() {
        // Break only on the loop-body line; `total` should grow 0→1→3 across the
        // three hits as the native code runs (instrumented assignment reporting).
        let stops = debug_trace(PROG, &[4], false).expect("debug run failed");
        assert_eq!(stops.len(), 3, "three breakpoint hits, got {}", stops.len());
        let totals: Vec<i64> = stops
            .iter()
            .map(|s| {
                match s.vars.iter().find(|(n, _)| n == "total").map(|(_, v)| v) {
                    Some(DbgVal::Int(n)) => *n,
                    _ => -1,
                }
            })
            .collect();
        // On entry to each iteration: before adding 1, 2, 3 respectively.
        assert_eq!(totals, vec![0, 1, 3], "total at each loop-body entry");
    }

    #[test]
    fn inspects_floats_and_struct_fields() {
        // Floats report as DbgVal::Float; struct locals report leaf-by-leaf with
        // dotted names (`p.x`, `p.y`).
        let prog = "struct P { x: i64, y: i64 }
fn main() {
    let pi = 3.5
    let p = P { x: 7, y: 9 }
    println(p.x)
}";
        let stops = debug_trace(prog, &[], true).expect("debug run failed");
        // The last stop (at `println`) sees all bindings made so far.
        let last = stops.last().expect("at least one stop");
        let get = |n: &str| last.vars.iter().find(|(name, _)| name == n).map(|(_, v)| v.clone());
        assert_eq!(get("pi"), Some(DbgVal::Float(3.5)), "float local: {:?}", last.vars);
        assert_eq!(get("p.x"), Some(DbgVal::Int(7)), "struct field x: {:?}", last.vars);
        assert_eq!(get("p.y"), Some(DbgVal::Int(9)), "struct field y: {:?}", last.vars);
    }

    #[test]
    fn recursion_has_isolated_per_frame_locals_and_a_call_stack() {
        // `fact` recurses; the debugger must keep each frame's `n` separate and
        // expose the growing call stack (per-frame model, not a single table).
        let prog = "fn fact(n: i64) -> i64 {
    if n <= 1 { return 1 }
    let r = n * fact(n - 1)
    r
}
fn main() {
    let x = fact(3)
    println(x)
}";
        // Break on the `let r` line (line 3).
        let stops = debug_trace(prog, &[3], false).expect("debug run failed");
        assert!(!stops.is_empty(), "should hit the breakpoint");
        // The deepest stop's stack is [main, fact, fact] (n=3 calls fact(2)).
        let deepest = stops.iter().max_by_key(|s| s.stack.len()).unwrap();
        assert!(deepest.stack.len() >= 3, "call stack should nest: {:?}", deepest.stack);
        assert_eq!(deepest.stack.last().map(String::as_str), Some("fact"));
        // Across stops, `n` takes distinct per-frame values (3 and 2).
        let ns: Vec<i64> = stops
            .iter()
            .filter_map(|s| s.vars.iter().find(|(n, _)| n == "n"))
            .filter_map(|(_, v)| if let DbgVal::Int(n) = v { Some(*n) } else { None })
            .collect();
        assert!(ns.contains(&3) && ns.contains(&2), "isolated frame locals, got {ns:?}");
    }

    #[test]
    fn enum_variant_tag_is_reported() {
        let prog = "enum Shape { Dot, Box, Seg }
fn main() {
    let s = Shape::Seg
    println(0)
}";
        let stops = debug_trace(prog, &[], true).expect("debug run failed");
        let last = stops.last().unwrap();
        let tag = last.vars.iter().find(|(n, _)| n == "s.tag").map(|(_, v)| v.clone());
        assert_eq!(tag, Some(DbgVal::Int(2)), "Seg is variant index 2: {:?}", last.vars);
    }

    #[test]
    fn no_breakpoints_no_step_runs_clean() {
        let stops = debug_trace(PROG, &[], false).expect("debug run failed");
        assert!(stops.is_empty(), "no breakpoints + no step → no stops");
    }
}
