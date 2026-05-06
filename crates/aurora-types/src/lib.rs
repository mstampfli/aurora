//! Type representation and a Hindley-Milner-style unification engine (grammar
//! spec §7). This is the core the bidirectional type checker will drive; it is
//! deliberately independent of the AST so it can be tested in isolation.
//!
//! Inference variables live in [`InferCtx`] as a union-find of substitutions.
//! `unify` makes two types equal (binding variables, occurs-checked), and
//! `resolve_deep` reads back a fully-substituted type once inference settles.
//!
//! Note (current limitation): numeric literals are given concrete default types
//! (`i32`/`f32`) at construction rather than via numeric inference variables, so
//! `let x: u8 = 1` defaulting is not modelled yet — that arrives with the
//! bidirectional checker's literal handling.

use aurora_lexer::{FloatTy, IntTy};

/// A type. `Var` is an inference variable resolved through [`InferCtx`].
#[derive(Clone, Debug, PartialEq)]
pub enum Ty {
    Unit,
    Bool,
    Char,
    Str,
    Int(IntTy),
    Float(FloatTy),
    /// An unsuffixed integer literal — unifies with any concrete integer type
    /// (but not bool/float). Defaults to `i32` if never constrained.
    IntLit,
    /// An unsuffixed float literal — unifies with any concrete float type.
    FloatLit,
    /// `VecN` for N in 2..=4.
    Vec(u8),
    /// Square `MatN`.
    Mat(u8),
    Quat,
    Color,
    /// A nominal type: struct, enum, or component, referenced by name.
    Named(String),
    Tuple(Vec<Ty>),
    Ref { mutable: bool, inner: Box<Ty> },
    Owned(Box<Ty>),
    Rc(Box<Ty>),
    Array(Box<Ty>, Option<u64>),
    Fn(Vec<Ty>, Box<Ty>),
    /// Inference variable (index into `InferCtx`).
    Var(u32),
    /// Propagated type error; unifies with anything to avoid error cascades.
    Error,
}

impl Ty {
    pub fn unit() -> Ty {
        Ty::Unit
    }
    pub fn reference(mutable: bool, inner: Ty) -> Ty {
        Ty::Ref { mutable, inner: Box::new(inner) }
    }

    /// A short human-facing rendering for diagnostics.
    pub fn display(&self) -> String {
        match self {
            Ty::Unit => "()".into(),
            Ty::Bool => "bool".into(),
            Ty::Char => "char".into(),
            Ty::Str => "str".into(),
            Ty::Int(t) => format!("{t:?}").to_lowercase(),
            Ty::Float(t) => format!("{t:?}").to_lowercase(),
            Ty::IntLit => "{integer}".into(),
            Ty::FloatLit => "{float}".into(),
            Ty::Vec(n) => format!("Vec{n}"),
            Ty::Mat(n) => format!("Mat{n}"),
            Ty::Quat => "Quat".into(),
            Ty::Color => "Color".into(),
            Ty::Named(n) => n.clone(),
            Ty::Tuple(ts) => {
                let inner: Vec<_> = ts.iter().map(Ty::display).collect();
                format!("({})", inner.join(", "))
            }
            Ty::Ref { mutable, inner } => {
                format!("&{}{}", if *mutable { "mut " } else { "" }, inner.display())
            }
            Ty::Owned(t) => format!("~{}", t.display()),
            Ty::Rc(t) => format!("rc<{}>", t.display()),
            Ty::Array(t, Some(n)) => format!("[{}; {n}]", t.display()),
            Ty::Array(t, None) => format!("[{}]", t.display()),
            Ty::Fn(params, ret) => {
                let ps: Vec<_> = params.iter().map(Ty::display).collect();
                format!("fn({}) -> {}", ps.join(", "), ret.display())
            }
            Ty::Var(id) => format!("?{id}"),
            Ty::Error => "<error>".into(),
        }
    }
}

/// A unification failure: the two types could not be made equal.
#[derive(Clone, Debug, PartialEq)]
pub struct TypeError {
    pub expected: String,
    pub found: String,
    pub message: String,
}

/// The inference context: a growable union-find of variable substitutions.
#[derive(Default)]
pub struct InferCtx {
    subst: Vec<Option<Ty>>,
}

impl InferCtx {
    pub fn new() -> InferCtx {
        InferCtx::default()
    }

    /// Allocate a fresh inference variable.
    pub fn fresh(&mut self) -> Ty {
        let id = self.subst.len() as u32;
        self.subst.push(None);
        Ty::Var(id)
    }

    /// Follow variable bindings one level (does not recurse into compounds).
    pub fn resolve_shallow(&self, ty: &Ty) -> Ty {
        let mut cur = ty.clone();
        while let Ty::Var(id) = cur {
            match &self.subst[id as usize] {
                Some(bound) => cur = bound.clone(),
                None => break,
            }
        }
        cur
    }

    /// Fully substitute all variables, recursing into compound types. Remaining
    /// unbound variables are left as `Var`.
    pub fn resolve_deep(&self, ty: &Ty) -> Ty {
        let shallow = self.resolve_shallow(ty);
        match shallow {
            Ty::Tuple(ts) => Ty::Tuple(ts.iter().map(|t| self.resolve_deep(t)).collect()),
            Ty::Ref { mutable, inner } => {
                Ty::Ref { mutable, inner: Box::new(self.resolve_deep(&inner)) }
            }
            Ty::Owned(t) => Ty::Owned(Box::new(self.resolve_deep(&t))),
            Ty::Rc(t) => Ty::Rc(Box::new(self.resolve_deep(&t))),
            Ty::Array(t, n) => Ty::Array(Box::new(self.resolve_deep(&t)), n),
            Ty::Fn(params, ret) => Ty::Fn(
                params.iter().map(|t| self.resolve_deep(t)).collect(),
                Box::new(self.resolve_deep(&ret)),
            ),
            other => other,
        }
    }

    /// Make `a` and `b` equal, binding inference variables as needed.
    pub fn unify(&mut self, a: &Ty, b: &Ty) -> Result<(), TypeError> {
        let ra = self.resolve_shallow(a);
        let rb = self.resolve_shallow(b);
        match (&ra, &rb) {
            // Errors absorb to prevent cascades.
            (Ty::Error, _) | (_, Ty::Error) => Ok(()),

            (Ty::Var(i), Ty::Var(j)) if i == j => Ok(()),
            (Ty::Var(i), other) | (other, Ty::Var(i)) => self.bind(*i, other),

            (Ty::Unit, Ty::Unit)
            | (Ty::Bool, Ty::Bool)
            | (Ty::Char, Ty::Char)
            | (Ty::Str, Ty::Str)
            | (Ty::Quat, Ty::Quat)
            | (Ty::Color, Ty::Color) => Ok(()),

            (Ty::Int(x), Ty::Int(y)) if x == y => Ok(()),
            (Ty::Float(x), Ty::Float(y)) if x == y => Ok(()),

            // Numeric literals adapt to any concrete numeric type of the right
            // kind (but not across kinds, and not to bool).
            (Ty::IntLit, Ty::IntLit) => Ok(()),
            (Ty::FloatLit, Ty::FloatLit) => Ok(()),
            (Ty::IntLit, Ty::Int(_)) | (Ty::Int(_), Ty::IntLit) => Ok(()),
            (Ty::FloatLit, Ty::Float(_)) | (Ty::Float(_), Ty::FloatLit) => Ok(()),
            (Ty::Vec(x), Ty::Vec(y)) if x == y => Ok(()),
            (Ty::Mat(x), Ty::Mat(y)) if x == y => Ok(()),
            (Ty::Named(x), Ty::Named(y)) if x == y => Ok(()),

            (Ty::Tuple(xs), Ty::Tuple(ys)) if xs.len() == ys.len() => {
                for (x, y) in xs.iter().zip(ys.iter()) {
                    self.unify(x, y)?;
                }
                Ok(())
            }
            (
                Ty::Ref { mutable: m1, inner: i1 },
                Ty::Ref { mutable: m2, inner: i2 },
            ) if m1 == m2 => self.unify(i1, i2),
            (Ty::Owned(x), Ty::Owned(y)) | (Ty::Rc(x), Ty::Rc(y)) => self.unify(x, y),
            (Ty::Array(x, n1), Ty::Array(y, n2)) if n1 == n2 => self.unify(x, y),
            (Ty::Fn(p1, r1), Ty::Fn(p2, r2)) if p1.len() == p2.len() => {
                for (x, y) in p1.iter().zip(p2.iter()) {
                    self.unify(x, y)?;
                }
                self.unify(r1, r2)
            }

            _ => Err(TypeError {
                expected: ra.display(),
                found: rb.display(),
                message: format!("expected `{}`, found `{}`", ra.display(), rb.display()),
            }),
        }
    }

    fn bind(&mut self, var: u32, ty: &Ty) -> Result<(), TypeError> {
        if self.occurs(var, ty) {
            return Err(TypeError {
                expected: format!("?{var}"),
                found: ty.display(),
                message: format!("infinite type: `?{var}` occurs in `{}`", ty.display()),
            });
        }
        self.subst[var as usize] = Some(ty.clone());
        Ok(())
    }

    /// Occurs check: does `var` appear anywhere inside `ty` (after resolution)?
    fn occurs(&self, var: u32, ty: &Ty) -> bool {
        match self.resolve_shallow(ty) {
            Ty::Var(id) => id == var,
            Ty::Tuple(ts) => ts.iter().any(|t| self.occurs(var, t)),
            Ty::Ref { inner, .. } | Ty::Owned(inner) | Ty::Rc(inner) | Ty::Array(inner, _) => {
                self.occurs(var, &inner)
            }
            Ty::Fn(params, ret) => {
                params.iter().any(|t| self.occurs(var, t)) || self.occurs(var, &ret)
            }
            _ => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aurora_lexer::IntTy;

    #[test]
    fn unify_concrete_equal_and_mismatch() {
        let mut cx = InferCtx::new();
        assert!(cx.unify(&Ty::Bool, &Ty::Bool).is_ok());
        assert!(cx.unify(&Ty::Int(IntTy::I32), &Ty::Int(IntTy::I32)).is_ok());
        let err = cx.unify(&Ty::Bool, &Ty::Int(IntTy::I32)).unwrap_err();
        assert_eq!(err.expected, "bool");
        assert_eq!(err.found, "i32");
    }

    #[test]
    fn variable_binds_and_resolves() {
        let mut cx = InferCtx::new();
        let v = cx.fresh();
        cx.unify(&v, &Ty::Vec(3)).unwrap();
        assert_eq!(cx.resolve_deep(&v), Ty::Vec(3));
    }

    #[test]
    fn transitive_variable_unification() {
        let mut cx = InferCtx::new();
        let a = cx.fresh();
        let b = cx.fresh();
        cx.unify(&a, &b).unwrap();
        cx.unify(&b, &Ty::Float(FloatTy::F32)).unwrap();
        assert_eq!(cx.resolve_deep(&a), Ty::Float(FloatTy::F32));
    }

    #[test]
    fn tuple_structural_unification() {
        let mut cx = InferCtx::new();
        let v = cx.fresh();
        let lhs = Ty::Tuple(vec![Ty::Bool, v.clone()]);
        let rhs = Ty::Tuple(vec![Ty::Bool, Ty::Vec(2)]);
        cx.unify(&lhs, &rhs).unwrap();
        assert_eq!(cx.resolve_deep(&v), Ty::Vec(2));
    }

    #[test]
    fn tuple_arity_mismatch_fails() {
        let mut cx = InferCtx::new();
        let lhs = Ty::Tuple(vec![Ty::Bool]);
        let rhs = Ty::Tuple(vec![Ty::Bool, Ty::Bool]);
        assert!(cx.unify(&lhs, &rhs).is_err());
    }

    #[test]
    fn ref_mutability_must_match() {
        let mut cx = InferCtx::new();
        let shared = Ty::reference(false, Ty::Bool);
        let mutable = Ty::reference(true, Ty::Bool);
        assert!(cx.unify(&shared, &mutable).is_err());
    }

    #[test]
    fn occurs_check_prevents_infinite_type() {
        let mut cx = InferCtx::new();
        let v = cx.fresh();
        // v = (v, bool) would be infinite.
        let recursive = Ty::Tuple(vec![v.clone(), Ty::Bool]);
        assert!(cx.unify(&v, &recursive).is_err());
    }

    #[test]
    fn error_absorbs() {
        let mut cx = InferCtx::new();
        assert!(cx.unify(&Ty::Error, &Ty::Bool).is_ok());
        assert!(cx.unify(&Ty::Int(IntTy::I32), &Ty::Error).is_ok());
    }

    #[test]
    fn fn_type_unification() {
        let mut cx = InferCtx::new();
        let r = cx.fresh();
        let f1 = Ty::Fn(vec![Ty::Int(IntTy::I32)], Box::new(r.clone()));
        let f2 = Ty::Fn(vec![Ty::Int(IntTy::I32)], Box::new(Ty::Bool));
        cx.unify(&f1, &f2).unwrap();
        assert_eq!(cx.resolve_deep(&r), Ty::Bool);
    }
}
