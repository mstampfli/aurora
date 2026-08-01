//! Conversion from AST type syntax to the checker's [`Ty`] representation,
//! recognizing the builtin types from grammar spec §2.2 / §7. A user-defined
//! type with the same name as a builtin (e.g. a `struct Vec3`) shadows the
//! builtin.

use std::collections::HashSet;

use aurora_ast::{Type, TypeKind};
use aurora_lexer::{FloatTy, IntTy};
use aurora_types::{InferCtx, Ty};

thread_local! {
    static CONST_LENS: std::cell::RefCell<std::collections::HashMap<String, u64>> =
        std::cell::RefCell::new(std::collections::HashMap::new());
}

fn const_len(name: &str) -> Option<u64> {
    CONST_LENS.with(|c| c.borrow().get(name).copied())
}

/// Fold a constant integer expression: a literal, a name already known, or
/// arithmetic over them. Mirrors codegen's `const_int` so an array length means
/// the same thing to both layers.
fn eval_const(
    k: &aurora_ast::ExprKind,
    known: &std::collections::HashMap<String, u64>,
) -> Option<i64> {
    use aurora_ast::{BinOp, ExprKind, UnOp};
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

/// How long an array-length expression says the array is, or None.
///
/// The ONE place typeck answers that question - both the type position and the
/// repeat expression call it, so the two cannot drift apart. They did: the
/// annotation resolved a const while `[7; N]` did not, and the mismatch was the
/// error.
pub(crate) fn array_len_of(k: &aurora_ast::ExprKind) -> Option<u64> {
    match k {
        aurora_ast::ExprKind::Int(v, _) => Some(*v as u64),
        aurora_ast::ExprKind::Path(p) => {
            let joined = p
                .segments
                .iter()
                .map(|s| s.ident.name.as_str())
                .collect::<Vec<_>>()
                .join("::");
            const_len(&joined)
        }
        other => CONST_LENS.with(|c| {
            eval_const(other, &c.borrow()).and_then(|v| u64::try_from(v).ok())
        }),
    }
}

/// Publish every const that evaluates to a non-negative integer.
///
/// Two passes, because a const may be defined in terms of another
/// (`const WIDE: i64 = N * 2`) and items are in source order.
pub fn set_const_lens(module: &aurora_ast::Module) {
    let mut out: std::collections::HashMap<String, u64> = std::collections::HashMap::new();
    for _ in 0..8 {
        let before = out.len();
        for item in &module.items {
            if let aurora_ast::ItemKind::Const(c) = &item.kind {
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
    if std::env::var("AURORA_DEBUG_LENS").is_ok() {
        let mut k: Vec<&String> = out.keys().collect();
        k.sort();
        eprintln!("  const-len table: {} entries {:?}", out.len(), k);
    }
    CONST_LENS.with(|c| *c.borrow_mut() = out);
}

pub(crate) fn type_to_ty(t: &Type, cx: &mut InferCtx, user: &HashSet<String>) -> Ty {
    match &t.kind {
        TypeKind::Path(p) => {
            let last = p.segments.last();
            let name = last.map(|s| s.ident.name.as_str()).unwrap_or("");
            // `rc<T>` is special: a refcounted box.
            if name == "rc" {
                if let Some(arg) = last.and_then(|s| s.args.first()) {
                    return Ty::Rc(Box::new(type_to_ty(arg, cx, user)));
                }
            }
            // A user-defined type shadows any builtin of the same name.
            if user.contains(name) {
                return Ty::Named(name.to_string());
            }
            builtin_or_named(name)
        }
        TypeKind::Owned(inner) => Ty::Owned(Box::new(type_to_ty(inner, cx, user))),
        TypeKind::Ref { mutable, inner } => Ty::reference(*mutable, type_to_ty(inner, cx, user)),
        TypeKind::Array { elem, len } => {
            // A length may be a literal OR a const - `[i64; TOWN_COUNT]`.
            let n = len.as_ref().and_then(|e| array_len_of(&e.kind));
            Ty::Array(Box::new(type_to_ty(elem, cx, user)), n)
        }
        TypeKind::Tuple(ts) => Ty::Tuple(ts.iter().map(|t| type_to_ty(t, cx, user)).collect()),
        TypeKind::Fn { params, ret } => Ty::Fn(
            params.iter().map(|t| type_to_ty(t, cx, user)).collect(),
            Box::new(type_to_ty(ret, cx, user)),
        ),
        // Trait objects aren't modelled yet; treat as unknown.
        TypeKind::Dyn(_) => Ty::Error,
        // A region annotation (`#perm T`) is checking-only — the type is `T`.
        TypeKind::Region(_, inner) => type_to_ty(inner, cx, user),
        TypeKind::Infer => cx.fresh(),
        TypeKind::Error => Ty::Error,
    }
}

fn builtin_or_named(name: &str) -> Ty {
    match name {
        "f32" => Ty::Float(FloatTy::F32),
        "f64" => Ty::Float(FloatTy::F64),
        "i8" => Ty::Int(IntTy::I8),
        "i16" => Ty::Int(IntTy::I16),
        "i32" => Ty::Int(IntTy::I32),
        "i64" => Ty::Int(IntTy::I64),
        "u8" => Ty::Int(IntTy::U8),
        "u16" => Ty::Int(IntTy::U16),
        "u32" => Ty::Int(IntTy::U32),
        "u64" => Ty::Int(IntTy::U64),
        "bool" => Ty::Bool,
        "char" => Ty::Char,
        "str" => Ty::Str,
        "void" => Ty::Unit,
        "Vec2" => Ty::Vec(2),
        "Vec3" => Ty::Vec(3),
        "Vec4" => Ty::Vec(4),
        "Mat2" => Ty::Mat(2),
        "Mat3" => Ty::Mat(3),
        "Mat4" => Ty::Mat(4),
        "Quat" => Ty::Quat,
        "Color" => Ty::Color,
        // Everything else (Transform, Time, Entity, Handle, Option, local types,
        // imported names, ...) is nominal and unified by name.
        other => Ty::Named(other.to_string()),
    }
}
