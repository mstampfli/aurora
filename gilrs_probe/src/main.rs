// Does `Gilrs::new()` return on this machine, and how fast?
//
// The suspicion: it hangs, and `aurora_input_step` calls it every frame, so the
// game would hang on its first frame with nobody able to say why.
fn main() {
    let t = std::time::Instant::now();
    eprintln!("calling Gilrs::new()");
    match gilrs::Gilrs::new() {
        Ok(mut g) => {
            eprintln!("ok in {:?}", t.elapsed());
            let n = g.gamepads().count();
            eprintln!("{n} gamepad(s)");
            let t2 = std::time::Instant::now();
            while g.next_event().is_some() {}
            eprintln!("drained in {:?}", t2.elapsed());
        }
        Err(e) => eprintln!("err in {:?}: {e}", t.elapsed()),
    }
    eprintln!("PROBE DONE");
}
