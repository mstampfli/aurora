//! Conversion from AST type syntax to the checker's [`Ty`] representation,
//! recognizing the builtin types from grammar spec §2.2 / §7. A user-defined
//! type with the same name as a builtin (e.g. a `struct Vec3`) shadows the
//! builtin.

use std::collections::HashSet;

use aurora_ast::{Type, TypeKind};
use aurora_lexer::{FloatTy, IntTy};
use aurora_types::{InferCtx, Ty};

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
            // Only literal integer lengths are captured for now.
            let n = len.as_ref().and_then(|e| match &e.kind {
                aurora_ast::ExprKind::Int(v, _) => Some(*v as u64),
                _ => None,
            });
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
