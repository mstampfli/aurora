// On macOS the compiler runs on the OS main thread (so the window event loop can own it - see
// main.rs), and the main thread's default 8MB stack would overflow on Aurora's recursive compiler
// passes for big programs. Give the main thread a 256MB stack via ld64's -stack_size, matching the
// worker-thread stack other platforms use. No effect on Linux/Windows.
fn main() {
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") {
        println!("cargo:rustc-link-arg=-Wl,-stack_size,0x10000000");
    }
}
