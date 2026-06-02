//! Run the built-in live-window demo:  cargo run -p aurora-window --example demo
fn main() {
    if let Err(e) = aurora_window::demo() {
        eprintln!("demo failed: {e}");
        std::process::exit(1);
    }
}
