//! Conversion from AST type syntax to the checker's [`Ty`] representation,
//! recognizing the builtin types from grammar spec §2.2 / §7. A user-defined
//! type with the same name as a builtin (e.g. a `struct Vec3`) shadows the
//! builtin.

use std::collections::HashSet;

use aurora_ast::{Span, Type, TypeKind};
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
// The const evaluator lives in `aurora-ast` so codegen resolves lengths the SAME
// way this does. It was duplicated, and the copies disagreed about which
// POSITIONS get an answer - a struct field or a return type silently became a
// zero-length array while a declaration worked.
use aurora_ast::eval_const;

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
    let out = aurora_ast::const_lengths(module);
    CONST_LENS.with(|c| *c.borrow_mut() = out);
}

/// Everything a type annotation is resolved AGAINST, in one place.
///
/// This was three loose arguments and it was about to be four. More to the
/// point, an unknown type name has to be REPORTED, and the only position that
/// can see one is inside the conversion - so the sink travels with the scope
/// rather than being something each of the eleven callers remembers to check.
pub(crate) struct TyScope<'a> {
    /// Every struct, component and enum declared in the (flattened) module.
    pub user: &'a HashSet<String>,
    /// The type parameters legal right here: a fn's or a struct's own generics.
    /// `fn id<T>(x: T)` must not report `T` as undefined.
    pub generics: &'a HashSet<String>,
    /// Names that resolved to nothing, with where they were written.
    pub unknown: &'a mut Vec<(String, Span)>,
}

/// Is this name a type that exists, one way or another?
fn resolves(name: &str, scope: &TyScope) -> bool {
    scope.user.contains(name)
        || scope.generics.contains(name)
        // The one list of names the language provides rather than the program.
        || aurora_ast::is_builtin_type(name)
}

pub(crate) fn type_to_ty(t: &Type, cx: &mut InferCtx, scope: &mut TyScope) -> Ty {
    match &t.kind {
        TypeKind::Path(p) => {
            let last = p.segments.last();
            let name = last.map(|s| s.ident.name.as_str()).unwrap_or("");
            // `rc<T>` is special: a refcounted box.
            if name == "rc" {
                if let Some(arg) = last.and_then(|s| s.args.first()) {
                    return Ty::Rc(Box::new(type_to_ty(arg, cx, scope)));
                }
            }
            // Generic ARGUMENTS are types too, and nothing converted them, so
            // the `Foo` in `Option<Foo>` was never resolved by anything. The
            // results are discarded - the checker models `Option<T>` nominally -
            // but converting them is what makes an undefined name inside one
            // visible at all.
            if let Some(seg) = last {
                for arg in &seg.args {
                    let _ = type_to_ty(arg, cx, scope);
                }
            }
            // A QUALIFIED type names an item in ANOTHER module, and the flattener
            // mangles that item's declaration to `module::Name` - one identifier
            // with the `::` inside it. So resolve the JOINED path first, or the
            // two spellings of one struct become two types.
            //
            // They did. A parameter written `arena::Meshes` read as bare `Meshes`
            // here, because only the last segment was consulted, while the struct
            // it names had been rewritten to `arena::Meshes`. Nothing noticed for
            // as long as that existed, because argument TYPES were only checked
            // for unqualified callees - where both sides come from one module and
            // therefore agree. Switching argument checking on for `mod::f(..)`
            // calls lit up fifty files at once, and every one of them was this.
            //
            // `array_len_of`, in this same file, already joins the segments. Two
            // rules for reading one path, twenty lines apart.
            if p.segments.len() > 1 {
                let joined = p
                    .segments
                    .iter()
                    .map(|s| s.ident.name.as_str())
                    .collect::<Vec<_>>()
                    .join("::");
                if scope.user.contains(joined.as_str()) {
                    return Ty::Named(joined);
                }
                // A qualified name that resolves to nothing is reported under
                // the spelling it was WRITTEN with. `arena::Meshes` is not a
                // missing `Meshes`.
                if !scope.generics.contains(joined.as_str()) {
                    scope.unknown.push((joined, t.span));
                }
                return builtin_or_named(name);
            }
            // A user-defined type shadows any builtin of the same name.
            if scope.user.contains(name) {
                return Ty::Named(name.to_string());
            }
            // NAMING NOTHING IS AN ERROR, and until this line it was not.
            //
            // `builtin_or_named` falls through to `Ty::Named(other)`, which is
            // exactly right for the engine's nominal types and was silently
            // right for typos too: a field declared `a: Nonexistent` became an
            // opaque type that unified with itself and nothing else. A parameter
            // written `fn takes(x: AlsoMissing)` produced no diagnostic at all,
            // and the errors that DID appear read "expected `StillMissing`,
            // found `{integer}`" - which asserts the missing type exists and
            // sends the reader looking for a conversion.
            if !resolves(name, scope) {
                scope.unknown.push((name.to_string(), t.span));
            }
            builtin_or_named(name)
        }
        TypeKind::Owned(inner) => Ty::Owned(Box::new(type_to_ty(inner, cx, scope))),
        TypeKind::Ref { mutable, inner } => Ty::reference(*mutable, type_to_ty(inner, cx, scope)),
        TypeKind::Array { elem, len } => {
            // A length may be a literal OR a const - `[i64; TOWN_COUNT]`.
            let n = len.as_ref().and_then(|e| array_len_of(&e.kind));
            Ty::Array(Box::new(type_to_ty(elem, cx, scope)), n)
        }
        TypeKind::Tuple(ts) => Ty::Tuple(ts.iter().map(|t| type_to_ty(t, cx, scope)).collect()),
        TypeKind::Fn { params, ret } => Ty::Fn(
            params.iter().map(|t| type_to_ty(t, cx, scope)).collect(),
            Box::new(type_to_ty(ret, cx, scope)),
        ),
        // Trait objects aren't modelled yet; treat as unknown.
        TypeKind::Dyn(_) => Ty::Error,
        // A region annotation (`#perm T`) is checking-only — the type is `T`.
        TypeKind::Region(_, inner) => type_to_ty(inner, cx, scope),
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
