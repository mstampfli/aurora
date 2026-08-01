//! Integer constants usable as ARRAY LENGTHS, resolved from the AST alone.
//!
//! **One resolver, deliberately.** This lived in `aurora-typeck` while
//! `aurora-codegen` kept a second copy of the same idea, and the two disagreed
//! about which POSITIONS get an answer: a const-named length resolved in a
//! declaration and silently became ZERO in a struct field or a return type,
//! because codegen filled its table after those positions were already lowered.
//! The program then compiled, ran, and failed a hundred lines away with
//! "index 0 out of bounds (length 0)".
//!
//! That was three separately-filed bugs with one cause. It lives here now
//! because `aurora-ast` is what both crates already depend on, and because a
//! length is a fact about the source, not about any one back end.
//!
//! **Place in the graph.** Bottom: depends on nothing, used by `typeck` and
//! `codegen`.

use crate::{BinOp, ExprKind, ItemKind, Module, UnOp};
use std::collections::HashMap;

/// Evaluate a constant integer expression against already-known constants.
pub fn eval_const(k: &ExprKind, known: &HashMap<String, u64>) -> Option<i64> {
    match k {
        ExprKind::Int(v, _) => i64::try_from(*v).ok(),
        ExprKind::Path(p) => {
            let joined = p
                .segments
                .iter()
                .map(|s| s.ident.name.as_str())
                .collect::<Vec<_>>()
                .join("::");
            known.get(&joined).map(|v| *v as i64)
        }
        ExprKind::Unary(UnOp::Neg, inner) => eval_const(&inner.kind, known)?.checked_neg(),
        ExprKind::Binary(op, lhs, rhs) => {
            let a = eval_const(&lhs.kind, known)?;
            let b = eval_const(&rhs.kind, known)?;
            match op {
                BinOp::Add => a.checked_add(b),
                BinOp::Sub => a.checked_sub(b),
                BinOp::Mul => a.checked_mul(b),
                BinOp::Div if b != 0 => a.checked_div(b),
                _ => None,
            }
        }
        _ => None,
    }
}

/// Every const in the module that evaluates to a non-negative integer.
///
/// Iterated to a fixpoint, because a const may be written in terms of another
/// (`const WIDE: i64 = N * 2`) and items are in source order. The bound is a
/// guard against a cycle rather than a depth limit - a chain longer than this
/// simply does not resolve, and an unresolved length must be reported by the
/// caller, never defaulted to zero.
pub fn const_lengths(module: &Module) -> HashMap<String, u64> {
    let mut out: HashMap<String, u64> = HashMap::new();
    for _ in 0..8 {
        let before = out.len();
        for item in &module.items {
            if let ItemKind::Const(c) = &item.kind {
                if let Some(v) = eval_const(&c.value.kind, &out) {
                    if v >= 0 {
                        out.insert(c.name.name.to_string(), v as u64);
                    }
                }
            }
        }
        if out.len() == before {
            break;
        }
    }
    out
}

/// How long an array-length expression says the array is.
///
/// `None` means "cannot be resolved", and the caller MUST treat that as an
/// error. Returning a default here is what made a broken program compile.
pub fn array_len(k: &ExprKind, known: &HashMap<String, u64>) -> Option<u64> {
    eval_const(k, known).and_then(|v| u64::try_from(v).ok())
}
