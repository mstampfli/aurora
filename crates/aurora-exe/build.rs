//! Link the Aurora-compiled object (pointed to by `AURORA_OBJ`) into this
//! executable. When the variable is unset — e.g. a plain `cargo build` of the
//! workspace — compile a stub instead so the crate still builds.

fn main() {
    println!("cargo:rustc-check-cfg=cfg(aurora_stub)");
    println!("cargo:rerun-if-env-changed=AURORA_OBJ");
    match std::env::var("AURORA_OBJ") {
        Ok(obj) if !obj.is_empty() => {
            // Relink whenever the compiled object changes.
            println!("cargo:rerun-if-changed={obj}");
            println!("cargo:rustc-link-arg={obj}");
        }
        _ => println!("cargo:rustc-cfg=aurora_stub"),
    }
}
