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
