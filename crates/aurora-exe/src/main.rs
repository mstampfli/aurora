//! Entry shim for AOT-compiled Aurora programs.
//!
//! The Aurora object exports `aurora_user_main`; we call it, flush stdout (the
//! program does not return through Rust's runtime), and exit with its result.
#![allow(unused_unsafe)]

// Real build: the symbol comes from the linked Aurora object.
#[cfg(not(aurora_stub))]
extern "C" {
    fn aurora_user_main() -> i64;
}

// Stub build (no AURORA_OBJ): define it so the workspace still links.
#[cfg(aurora_stub)]
#[no_mangle]
extern "C" fn aurora_user_main() -> i64 {
    0
}

fn main() {
    // Keep the runtime's host symbols in the link even if unreferenced by Rust.
    let _ = aurora_runtime::force_link();
    let code = unsafe { aurora_user_main() };
    aurora_runtime::aurora_runtime_flush();
    std::process::exit(code as i32);
}
