//! Aurora's runtime builtins, as seen by the front end.
//!
//! These are not user-defined functions: they never appear as `fn` items, and
//! the backend lowers each one to a native runtime call (or expands it inline).
//! The front end must not report them as unresolved names, so it needs the same
//! list the backend lowers from - a builtin missing from it would be reported as
//! an unknown function by `aurorac check`, and a stale extra entry would let a
//! real typo through.
//!
//! That list is not kept here: it is one row per builtin in [`aurora_abi`],
//! which the backend and the runtime expand as well. This module only re-exports
//! the front end's view of it, so `aurora-ast` stays where every front-end pass
//! already looks for it.

pub use aurora_abi::{builtin_names, is_builtin};

/// The type names that are real without being declared in Aurora source.
///
/// The same hazard as the builtin FUNCTION list above, one layer up: a name
/// missing from here is reported as an unknown type, and a stale extra entry
/// lets a real typo through. So there is one list and both front-end passes read
/// it - `aurora-check`'s resolver and `aurora-typeck`'s conversion.
///
/// They were two lists. `aurora-check` had this one; typeck grew a second within
/// minutes of needing one, and it was already missing `Transform`, `Handle`,
/// `Option`, `Result` and `Time` - so a program naming any of them type-checked
/// under one pass and was rejected by the other.
///
/// Sources, so an entry can be justified rather than remembered:
/// primitives and math from grammar spec 2.2; `Transform`/`Time`/`Entity`/
/// `Handle` are engine resources (`system spin(dt: Time)` is in the spec at
/// 01-grammar-and-types.md:250); `Tick` is the netcode clock, and is the type of
/// `Time`'s own `tick` field (02-netcode-replication.md:40); `rc`/`weak` are the
/// pointer shapes; `Self` is the receiver's type inside an `impl`.
const BUILTIN_TYPES: &[&str] = &[
    "f32", "f64", "i8", "i16", "i32", "i64", "u8", "u16", "u32", "u64", "bool", "char", "str",
    "void", "Vec2", "Vec3", "Vec4", "Mat2", "Mat3", "Mat4", "Quat", "Color", "Transform", "Time",
    "Tick", "Entity", "Handle", "Option", "Result", "rc", "weak", "Self",
];

/// Is `name` a type the language provides rather than the program?
pub fn is_builtin_type(name: &str) -> bool {
    BUILTIN_TYPES.contains(&name)
}
