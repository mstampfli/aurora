//! Bidirectional type checker (grammar spec §7), driving the unification engine
//! in `aurora-types`.
//!
//! ## Leniency strategy (the prelude problem)
//!
//! Without a full stdlib we cannot know the type of `texture(...)`, `App.new`,
//! or a field like `t.pos`. So unresolved external names, method/field accesses,
//! and unknown calls evaluate to [`Ty::Error`], which unifies with anything. The
//! effect: we never false-positive on builtins we haven't modelled, yet we still
//! catch real mismatches between *known* types — `let x: bool = 1`, a function
//! returning the wrong type, mismatched `if` branches, `i32 + f32`, and so on.
//!
//! Vector/matrix algebra (`Vec3 * f32`, `Mat4 * Vec4`) is intentionally treated
//! permissively rather than strictly unified, matching the overloaded operators
//! in spec §7.5.

mod convert;

use std::collections::HashMap;

use aurora_ast::{
    AssocItem, BinOp, Block, Expr, ExprKind, ItemKind, Module, Param, Pat, PatKind, QTerm,
    QueryExpr, Stmt, UnOp,
};
use aurora_diag::Diagnostic;
use aurora_span::Span;
use aurora_types::{InferCtx, Ty};

/// Type-check a module, returning diagnostics.
pub fn check_types(module: &Module) -> Vec<Diagnostic> {
    let mut tc = Typeck::new();
    tc.collect(module);
    tc.run(module);
    tc.diags
}

struct Typeck {
    cx: InferCtx,
    diags: Vec<Diagnostic>,
    scopes: Vec<HashMap<String, Ty>>,
    /// Local struct/component fields: name -> [(field, type)].
    structs: HashMap<String, Vec<(String, Ty)>>,
    /// Per struct, the fields that have no default (must be given in a literal).
    struct_required: HashMap<String, Vec<String>>,
    /// Top-level function signatures.
    fns: HashMap<String, Ty>,
    /// Generic type-parameter names per function (for call-site instantiation).
    fn_generics: HashMap<String, Vec<String>>,
    /// Generic type-parameter names per struct (fields of these types accept any
    /// value — generic structs are monomorphized later, in codegen).
    struct_generics: HashMap<String, std::collections::HashSet<String>>,
    /// `(type, trait)` pairs from `impl Trait for Type` — the implemented traits.
    trait_impls: std::collections::HashSet<(String, String)>,
    /// Per generic function: `(param, trait)` bounds to enforce at call sites.
    fn_bounds: HashMap<String, Vec<(String, String)>>,
    /// User-defined type names (structs/components/enums) — shadow builtins.
    user_types: std::collections::HashSet<String>,
}

impl Typeck {
    fn new() -> Typeck {
        Typeck {
            cx: InferCtx::new(),
            diags: Vec::new(),
            scopes: vec![HashMap::new()],
            structs: HashMap::new(),
            struct_required: HashMap::new(),
            fns: HashMap::new(),
            fn_generics: HashMap::new(),
            struct_generics: HashMap::new(),
            trait_impls: std::collections::HashSet::new(),
            fn_bounds: HashMap::new(),
            user_types: std::collections::HashSet::new(),
        }
    }

    // --- collection pass -----------------------------------------------------

    fn collect(&mut self, module: &Module) {
        // User-defined type names first (they shadow builtins in conversions).
        for item in &module.items {
            match &item.kind {
                ItemKind::Struct(s) | ItemKind::Component(s) => {
                    self.user_types.insert(s.name.name.clone());
                }
                ItemKind::Enum(e) => {
                    self.user_types.insert(e.name.name.clone());
                }
                _ => {}
            }
        }
        for item in &module.items {
            match &item.kind {
                ItemKind::Struct(s) | ItemKind::Component(s) => {
                    if let aurora_ast::StructBody::Named(fields) = &s.body {
                        let fs = fields
                            .iter()
                            .map(|f| (f.name.name.clone(), convert::type_to_ty(&f.ty, &mut self.cx, &self.user_types)))
                            .collect();
                        self.structs.insert(s.name.name.clone(), fs);
                        if !s.generics.is_empty() {
                            self.struct_generics.insert(
                                s.name.name.clone(),
                                s.generics.iter().map(|g| g.name.name.clone()).collect(),
                            );
                        }
                        let required = fields
                            .iter()
                            .filter(|f| f.default.is_none())
                            .map(|f| f.name.name.clone())
                            .collect();
                        self.struct_required.insert(s.name.name.clone(), required);
                    }
                }
                ItemKind::Fn(f) => {
                    let ty = self.fn_type(f);
                    self.fns.insert(f.name.name.clone(), ty);
                    let gens = f.generics.iter().map(|g| g.name.name.clone()).collect();
                    self.fn_generics.insert(f.name.name.clone(), gens);
                    // Record each generic param's trait bounds for call-site checks.
                    let bounds: Vec<(String, String)> = f
                        .generics
                        .iter()
                        .flat_map(|g| {
                            g.bounds.iter().filter_map(move |b| {
                                b.segments.last().map(|s| (g.name.name.clone(), s.ident.name.clone()))
                            })
                        })
                        .collect();
                    if !bounds.is_empty() {
                        self.fn_bounds.insert(f.name.name.clone(), bounds);
                    }
                }
                ItemKind::Impl(i) => {
                    // `impl Trait for Type` registers that Type implements Trait.
                    if let (Some(tr), Some(ty)) = (&i.trait_, type_head_name(&i.self_ty)) {
                        if let Some(tname) = tr.segments.last() {
                            self.trait_impls.insert((ty, tname.ident.name.clone()));
                        }
                    }
                }
                _ => {}
            }
        }
    }

    fn fn_type(&mut self, f: &aurora_ast::FnDecl) -> Ty {
        let params = f
            .params
            .iter()
            .filter_map(|p| match p {
                Param::Normal { ty, .. } => Some(convert::type_to_ty(ty, &mut self.cx, &self.user_types)),
                Param::SelfParam { .. } => None,
            })
            .collect();
        let ret = f
            .ret
            .as_ref()
            .map(|t| convert::type_to_ty(t, &mut self.cx, &self.user_types))
            .unwrap_or(Ty::Unit);
        Ty::Fn(params, Box::new(ret))
    }

    // --- driver --------------------------------------------------------------

    fn run(&mut self, module: &Module) {
        for item in &module.items {
            match &item.kind {
                ItemKind::Fn(f) => self.check_fn(f),
                ItemKind::System(s) => self.check_system(&s.body, &s.params),
                ItemKind::Const(c) => {
                    let value_ty = self.infer(&c.value);
                    if let Some(ann) = &c.ty {
                        let ann_ty = convert::type_to_ty(ann, &mut self.cx, &self.user_types);
                        self.expect(c.value.span, &ann_ty, &value_ty, "constant");
                    }
                }
                ItemKind::Impl(i) => {
                    for it in &i.items {
                        if let AssocItem::Fn(f) = it {
                            self.check_fn(f);
                        }
                    }
                }
                _ => {}
            }
        }
    }

    fn check_fn(&mut self, f: &aurora_ast::FnDecl) {
        let Some(body) = &f.body else { return };
        self.push();
        for p in &f.params {
            match p {
                Param::Normal { name, ty, .. } => {
                    let t = convert::type_to_ty(ty, &mut self.cx, &self.user_types);
                    self.bind(&name.name, t);
                }
                Param::SelfParam { .. } => self.bind("self", Ty::Named("Self".into())),
            }
        }
        let body_ty = self.check_block_no_scope(body);
        // Only enforce the return type when it was written explicitly.
        if let Some(ret) = &f.ret {
            let ret_ty = convert::type_to_ty(ret, &mut self.cx, &self.user_types);
            let span = body.tail.as_ref().map(|e| e.span).unwrap_or(body.span);
            self.expect(span, &ret_ty, &body_ty, "function return value");
        }
        self.pop();
    }

    fn check_system(&mut self, body: &Block, params: &[aurora_ast::SysParam]) {
        self.push();
        for p in params {
            let t = convert::type_to_ty(&p.ty, &mut self.cx, &self.user_types);
            self.bind(&p.name.name, t);
        }
        self.check_block_no_scope(body);
        self.pop();
    }

    // --- scopes --------------------------------------------------------------

    fn push(&mut self) {
        self.scopes.push(HashMap::new());
    }
    fn pop(&mut self) {
        self.scopes.pop();
    }
    fn bind(&mut self, name: &str, ty: Ty) {
        self.scopes.last_mut().unwrap().insert(name.to_string(), ty);
    }
    fn lookup(&self, name: &str) -> Option<Ty> {
        self.scopes.iter().rev().find_map(|s| s.get(name).cloned())
    }

    // --- diagnostics ---------------------------------------------------------

    fn expect(&mut self, span: Span, expected: &Ty, actual: &Ty, ctx: &str) {
        if let Err(e) = self.cx.unify(expected, actual) {
            self.diags.push(
                Diagnostic::error(format!("type mismatch in {ctx}: {}", e.message))
                    .with_code("E0300")
                    .primary(span, format!("expected `{}`, found `{}`", e.expected, e.found)),
            );
        }
    }

    // --- blocks & statements -------------------------------------------------

    fn check_block(&mut self, block: &Block) -> Ty {
        self.push();
        let t = self.check_block_no_scope(block);
        self.pop();
        t
    }

    fn check_block_no_scope(&mut self, block: &Block) -> Ty {
        for stmt in &block.stmts {
            match stmt {
                Stmt::Let(l) => self.check_let(l),
                Stmt::Defer(e) | Stmt::Expr(e) => {
                    self.infer(e);
                }
            }
        }
        match &block.tail {
            Some(e) => self.infer(e),
            None => Ty::Unit,
        }
    }

    fn check_let(&mut self, l: &aurora_ast::LetStmt) {
        let declared = l.ty.as_ref().map(|t| convert::type_to_ty(t, &mut self.cx, &self.user_types));
        let init_ty = l.init.as_ref().map(|e| (e.span, self.infer(e)));

        let bind_ty = match (&declared, &init_ty) {
            (Some(d), Some((span, i))) => {
                self.expect(*span, d, i, "let binding");
                d.clone()
            }
            (Some(d), None) => d.clone(),
            (None, Some((_, i))) => i.clone(),
            (None, None) => self.cx.fresh(),
        };
        self.bind_pat(&l.pat, &bind_ty);
    }

    // --- expression inference ------------------------------------------------

    fn infer(&mut self, e: &Expr) -> Ty {
        match &e.kind {
            ExprKind::Int(_, suffix) => suffix.map(Ty::Int).unwrap_or(Ty::IntLit),
            ExprKind::Float(_, suffix) => suffix.map(Ty::Float).unwrap_or(Ty::FloatLit),
            ExprKind::Bool(_) => Ty::Bool,
            ExprKind::Char(_) => Ty::Char,
            ExprKind::Str(_) => Ty::Str,
            ExprKind::SelfExpr => self.lookup("self").unwrap_or(Ty::Error),
            ExprKind::Path(p) => {
                if p.is_single() {
                    self.lookup(&p.segments[0].ident.name).unwrap_or(Ty::Error)
                } else {
                    Ty::Error
                }
            }
            ExprKind::Dot(_) => Ty::Error,
            ExprKind::Paren(inner) => self.infer(inner),
            ExprKind::Unary(op, inner) => self.infer_unary(*op, inner),
            ExprKind::Binary(op, a, b) => self.infer_binary(*op, a, b, e.span),
            ExprKind::Assign(_, lhs, rhs) => {
                let lt = self.infer(lhs);
                let rt = self.infer(rhs);
                // Lenient: only flag when both sides are known and incompatible.
                if !is_unknown(&self.cx.resolve_deep(&lt)) && !is_unknown(&self.cx.resolve_deep(&rt))
                {
                    self.expect(rhs.span, &lt, &rt, "assignment");
                }
                Ty::Unit
            }
            ExprKind::Cast(inner, ty) => {
                self.infer(inner);
                convert::type_to_ty(ty, &mut self.cx, &self.user_types)
            }
            ExprKind::Call { callee, args, .. } => self.infer_call(callee, args),
            ExprKind::Index { base, index } => {
                self.infer(base);
                self.infer(index);
                self.cx.fresh()
            }
            ExprKind::Field { base, .. } => {
                self.infer(base);
                Ty::Error // field types require nominal field resolution (later)
            }
            ExprKind::Range { start, end, .. } => {
                if let Some(s) = start {
                    self.infer(s);
                }
                if let Some(en) = end {
                    self.infer(en);
                }
                Ty::Error
            }
            ExprKind::Pipe { value, func } => {
                self.infer(value);
                self.infer(func);
                Ty::Error
            }
            ExprKind::Struct { path, fields, base } => self.infer_struct(path, fields, base.as_deref()),
            ExprKind::Array(items) => {
                let elem = self.cx.fresh();
                for it in items {
                    let t = self.infer(it);
                    self.expect(it.span, &elem, &t, "array element");
                }
                Ty::Array(Box::new(self.cx.resolve_deep(&elem)), Some(items.len() as u64))
            }
            ExprKind::ArrayRepeat { value, count } => {
                self.infer(count);
                let e = self.infer(value);
                // A literal repeat count gives a known array size, so `[0; 32]`
                // matches a `[i64; 32]` field/annotation.
                let n = if let ExprKind::Int(v, _) = &count.kind { Some(*v as u64) } else { None };
                Ty::Array(Box::new(self.cx.resolve_deep(&e)), n)
            }
            ExprKind::Tuple(items) => {
                Ty::Tuple(items.iter().map(|it| self.infer(it)).collect())
            }
            ExprKind::If(ifx) => {
                self.check_cond(&ifx.cond);
                let then_ty = self.check_block(&ifx.then_branch);
                match &ifx.else_branch {
                    Some(else_e) => {
                        let else_ty = self.infer(else_e);
                        // Branches must agree only when both are known.
                        if !is_unknown(&self.cx.resolve_deep(&then_ty))
                            && !is_unknown(&self.cx.resolve_deep(&else_ty))
                        {
                            self.expect(else_e.span, &then_ty, &else_ty, "if branches");
                        }
                        then_ty
                    }
                    None => Ty::Unit,
                }
            }
            ExprKind::Match { scrutinee, arms } => {
                self.infer(scrutinee);
                let mut result: Option<Ty> = None;
                for arm in arms {
                    self.push();
                    self.bind_pat(&arm.pat, &Ty::Error);
                    if let Some(g) = &arm.guard {
                        self.check_cond(g);
                    }
                    let body_ty = self.infer(&arm.body);
                    self.pop();
                    result = Some(body_ty);
                }
                result.unwrap_or(Ty::Unit)
            }
            ExprKind::For { pat, iter, body } => {
                let elem = self.iter_elem_ty(iter);
                self.push();
                self.bind_pat(pat, &elem);
                self.check_block_no_scope(body);
                self.pop();
                Ty::Unit
            }
            ExprKind::While { cond, body } => {
                self.check_cond(cond);
                self.check_block(body);
                Ty::Unit
            }
            ExprKind::Loop(b) | ExprKind::Unsafe(b) | ExprKind::Block(b) => self.check_block(b),
            ExprKind::Closure { params, body } => {
                self.push();
                let mut param_tys = Vec::new();
                for p in params {
                    if let Param::Normal { name, ty, .. } = p {
                        let t = convert::type_to_ty(ty, &mut self.cx, &self.user_types);
                        self.bind(&name.name, t.clone());
                        param_tys.push(t);
                    }
                }
                let ret = self.infer(body);
                self.pop();
                Ty::Fn(param_tys, Box::new(ret))
            }
            ExprKind::Query(_) => Ty::Named("Query".into()),
            ExprKind::Spawn(args) => {
                for a in args {
                    self.infer(&a.value);
                }
                Ty::Named("Entity".into())
            }
            ExprKind::Despawn(inner) => {
                self.infer(inner);
                Ty::Unit
            }
            ExprKind::Region { value, .. } => self.infer(value),
            ExprKind::Return(opt) | ExprKind::Break(opt) => {
                if let Some(inner) = opt {
                    self.infer(inner);
                }
                // Diverging expressions: a fresh var unifies with any context.
                self.cx.fresh()
            }
            ExprKind::Continue => self.cx.fresh(),
            // `expr?` unwraps the success payload; its type is left to inference
            // (the enum is monomorphized in codegen). Infer the inner for checks.
            ExprKind::Try(inner) => {
                self.infer(inner);
                self.cx.fresh()
            }
            ExprKind::Error => Ty::Error,
        }
    }

    fn infer_unary(&mut self, op: UnOp, inner: &Expr) -> Ty {
        let t = self.infer(inner);
        match op {
            UnOp::Neg => t,
            UnOp::Not => {
                if !is_unknown(&self.cx.resolve_deep(&t)) {
                    self.expect(inner.span, &Ty::Bool, &t, "`not` operand");
                }
                Ty::Bool
            }
            UnOp::RefShared => Ty::reference(false, t),
            UnOp::RefMut => Ty::reference(true, t),
            UnOp::Own => Ty::Owned(Box::new(t)),
            UnOp::Deref => match self.cx.resolve_deep(&t) {
                Ty::Ref { inner, .. } | Ty::Owned(inner) | Ty::Rc(inner) => *inner,
                _ => Ty::Error,
            },
        }
    }

    fn infer_binary(&mut self, op: BinOp, a: &Expr, b: &Expr, _span: Span) -> Ty {
        let ta = self.infer(a);
        let tb = self.infer(b);
        let ra = self.cx.resolve_deep(&ta);
        let rb = self.cx.resolve_deep(&tb);
        match op {
            BinOp::And | BinOp::Or => {
                if !is_unknown(&ra) {
                    self.expect(a.span, &Ty::Bool, &ra, "logical operand");
                }
                if !is_unknown(&rb) {
                    self.expect(b.span, &Ty::Bool, &rb, "logical operand");
                }
                Ty::Bool
            }
            BinOp::Eq | BinOp::Ne | BinOp::Lt | BinOp::Gt | BinOp::Le | BinOp::Ge => {
                if !is_unknown(&ra) && !is_unknown(&rb) && !is_vectorish(&ra) && !is_vectorish(&rb) {
                    self.expect(b.span, &ra, &rb, "comparison");
                }
                Ty::Bool
            }
            BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Rem => {
                self.arith(&ra, &rb, b.span)
            }
        }
    }

    /// Arithmetic result type. Scalar+scalar must match; anything involving a
    /// vector/matrix/quat/color is permitted (overloaded algebra, §7.5);
    /// unknowns propagate leniently.
    fn arith(&mut self, a: &Ty, b: &Ty, span: Span) -> Ty {
        if is_unknown(a) {
            return if is_unknown(b) { self.cx.fresh() } else { b.clone() };
        }
        if is_unknown(b) {
            return a.clone();
        }
        if is_vectorish(a) {
            return a.clone();
        }
        if is_vectorish(b) {
            return b.clone();
        }
        // Integer literals adapt to a concrete integer operand (and vice versa).
        if is_int_like(a) && is_int_like(b) {
            return if matches!(a, Ty::Int(_)) { a.clone() } else { b.clone() };
        }
        if is_float_like(a) && is_float_like(b) {
            return if matches!(a, Ty::Float(_)) { a.clone() } else { b.clone() };
        }
        // Both scalar/known: require equality.
        if a == b {
            a.clone()
        } else {
            self.expect(span, a, b, "arithmetic operands");
            Ty::Error
        }
    }

    fn check_cond(&mut self, cond: &Expr) {
        let t = self.infer(cond);
        let rt = self.cx.resolve_deep(&t);
        if !is_unknown(&rt) {
            self.expect(cond.span, &Ty::Bool, &rt, "condition");
        }
    }

    fn infer_call(&mut self, callee: &Expr, args: &[aurora_ast::Arg]) -> Ty {
        let arg_tys: Vec<(Span, Ty)> = args.iter().map(|a| (a.value.span, self.infer(&a.value))).collect();

        // Only known top-level function paths get argument checking; everything
        // else (methods, builtins, imports) is treated as unknown.
        if let ExprKind::Path(p) = &callee.kind {
            if p.is_single() {
                let name = &p.segments[0].ident.name;
                if let Some(Ty::Fn(params, ret)) = self.fns.get(name).cloned() {
                    // Instantiate generic type parameters with fresh variables so
                    // each call is checked independently (e.g. `pair(1, true)`).
                    let generics = self.fn_generics.get(name).cloned().unwrap_or_default();
                    let subst: HashMap<String, Ty> =
                        generics.iter().map(|g| (g.clone(), self.cx.fresh())).collect();
                    let params: Vec<Ty> = params.iter().map(|p| subst_ty(p, &subst)).collect();
                    let ret = subst_ty(&ret, &subst);

                    if params.len() == arg_tys.len() {
                        for (param, (span, actual)) in params.iter().zip(&arg_tys) {
                            self.expect(*span, param, actual, "function argument");
                        }
                    }
                    // Enforce trait bounds: each bounded generic param's resolved
                    // concrete (named) type must `impl` the required trait.
                    if let Some(bounds) = self.fn_bounds.get(name).cloned() {
                        let call_span = p.span;
                        for (param, trait_name) in bounds {
                            if let Some(tv) = subst.get(&param) {
                                if let Ty::Named(ty) = self.cx.resolve_deep(tv) {
                                    if !self.trait_impls.contains(&(ty.clone(), trait_name.clone())) {
                                        self.diags.push(
                                            Diagnostic::error(format!(
                                                "`{ty}` does not implement trait `{trait_name}`"
                                            ))
                                            .with_code("E0320")
                                            .primary(call_span, "required by this call's bound"),
                                        );
                                    }
                                }
                            }
                        }
                    }
                    return ret;
                }
            }
        }
        self.infer(callee);
        Ty::Error
    }

    fn infer_struct(
        &mut self,
        path: &aurora_ast::Path,
        fields: &[aurora_ast::FieldInit],
        base: Option<&Expr>,
    ) -> Ty {
        let name = path.segments.last().map(|s| s.ident.name.clone()).unwrap_or_default();
        let known = self.structs.get(&name).cloned();

        for f in fields {
            let value_ty = match &f.value {
                Some(v) => self.infer(v),
                None => self.lookup(&f.name.name).unwrap_or(Ty::Error), // shorthand
            };
            if let Some(decls) = &known {
                match decls.iter().find(|(n, _)| *n == f.name.name) {
                    Some((_, declared)) => {
                        let span = f.value.as_ref().map(|v| v.span).unwrap_or(f.name.span);
                        // A field whose declared type mentions one of the struct's
                        // generic params (bare `T` or e.g. `[T; N]`) accepts any
                        // value — generic structs are monomorphized later.
                        let empty = std::collections::HashSet::new();
                        let gens = self.struct_generics.get(&name).unwrap_or(&empty);
                        let is_generic_field = ty_mentions_generic(declared, gens);
                        if !is_generic_field && !is_unknown(&self.cx.resolve_deep(&value_ty)) {
                            self.expect(span, declared, &value_ty, "struct field");
                        }
                    }
                    None => self.diags.push(
                        Diagnostic::error(format!("no field `{}` on `{name}`", f.name.name))
                            .with_code("E0301")
                            .primary(f.name.span, "unknown field"),
                    ),
                }
            }
        }
        // Missing-field check: a literal of a known local struct must supply
        // every field without a default (unless it spreads a `..base`).
        if known.is_some() && base.is_none() {
            if let Some(required) = self.struct_required.get(&name).cloned() {
                let provided: std::collections::HashSet<&str> =
                    fields.iter().map(|f| f.name.name.as_str()).collect();
                for req in required {
                    if !provided.contains(req.as_str()) {
                        let span = path.segments.last().map(|s| s.ident.span).unwrap_or(path.span);
                        self.diags.push(
                            Diagnostic::error(format!("missing field `{req}` in `{name}`"))
                                .with_code("E0302")
                                .primary(span, "this field has no default and must be set"),
                        );
                    }
                }
            }
        }

        if let Some(b) = base {
            self.infer(b);
        }
        Ty::Named(name)
    }

    // --- iteration & patterns ------------------------------------------------

    /// Element type produced by iterating `iter` (special-cased for queries).
    fn iter_elem_ty(&mut self, iter: &Expr) -> Ty {
        if let ExprKind::Query(q) = &iter.kind {
            return query_elem_ty(q);
        }
        self.infer(iter);
        self.cx.fresh()
    }

    fn bind_pat(&mut self, pat: &Pat, ty: &Ty) {
        match &pat.kind {
            PatKind::Wild | PatKind::Rest | PatKind::Lit(_) | PatKind::Path(_) => {}
            PatKind::Binding { name, sub } => {
                self.bind(&name.name, ty.clone());
                if let Some(s) = sub {
                    self.bind_pat(s, ty);
                }
            }
            PatKind::Tuple(pats) => {
                let resolved = self.cx.resolve_deep(ty);
                if let Ty::Tuple(elems) = &resolved {
                    if elems.len() == pats.len() {
                        for (p, t) in pats.iter().zip(elems.iter()) {
                            self.bind_pat(p, t);
                        }
                        return;
                    }
                }
                for p in pats {
                    let fresh = self.cx.fresh();
                    self.bind_pat(p, &fresh);
                }
            }
            PatKind::TupleStruct { elems, .. } => {
                for p in elems {
                    let fresh = self.cx.fresh();
                    self.bind_pat(p, &fresh);
                }
            }
            PatKind::Struct { fields, .. } => {
                for fp in fields {
                    let t = self.cx.fresh();
                    match &fp.pat {
                        Some(p) => self.bind_pat(p, &t),
                        None => self.bind(&fp.name.name, t),
                    }
                }
            }
            PatKind::Error => {}
        }
    }
}

/// The tuple type yielded by iterating a `query<...>` (data terms only).
fn query_elem_ty(q: &QueryExpr) -> Ty {
    let mut parts = Vec::new();
    for term in &q.terms {
        let part = match term {
            QTerm::Read(p) => Ty::reference(false, named(p)),
            QTerm::Write(p) => Ty::reference(true, named(p)),
            QTerm::OptRead(_) | QTerm::OptWrite(_) => Ty::Named("Option".into()),
            QTerm::Entity => Ty::Named("Entity".into()),
            QTerm::With(_) | QTerm::Without(_) => continue, // filters: no binding
        };
        parts.push(part);
    }
    match parts.len() {
        0 => Ty::Unit,
        1 => parts.into_iter().next().unwrap(),
        _ => Ty::Tuple(parts),
    }
}

fn named(p: &aurora_ast::Path) -> Ty {
    Ty::Named(p.segments.last().map(|s| s.ident.name.clone()).unwrap_or_default())
}

/// Substitute generic type-parameter names with their instantiated types.
fn subst_ty(ty: &Ty, subst: &HashMap<String, Ty>) -> Ty {
    match ty {
        Ty::Named(n) => subst.get(n).cloned().unwrap_or_else(|| ty.clone()),
        Ty::Tuple(ts) => Ty::Tuple(ts.iter().map(|t| subst_ty(t, subst)).collect()),
        Ty::Ref { mutable, inner } => {
            Ty::Ref { mutable: *mutable, inner: Box::new(subst_ty(inner, subst)) }
        }
        Ty::Owned(t) => Ty::Owned(Box::new(subst_ty(t, subst))),
        Ty::Rc(t) => Ty::Rc(Box::new(subst_ty(t, subst))),
        Ty::Array(t, n) => Ty::Array(Box::new(subst_ty(t, subst)), *n),
        Ty::Fn(ps, r) => Ty::Fn(
            ps.iter().map(|t| subst_ty(t, subst)).collect(),
            Box::new(subst_ty(r, subst)),
        ),
        other => other.clone(),
    }
}

fn is_unknown(t: &Ty) -> bool {
    matches!(t, Ty::Error | Ty::Var(_))
}

/// The head type name of an `impl`'s self type, e.g. `Cat` or `Box<i64>` → "Box".
fn type_head_name(t: &aurora_ast::Type) -> Option<String> {
    match &t.kind {
        aurora_ast::TypeKind::Path(p) => p.segments.last().map(|s| s.ident.name.clone()),
        _ => None,
    }
}

/// Whether `t` mentions any of the given generic-param names (bare or nested in
/// an array/tuple/ref). Used to relax struct-field checks for generic structs.
fn ty_mentions_generic(t: &Ty, generics: &std::collections::HashSet<String>) -> bool {
    match t {
        Ty::Named(n) => generics.contains(n),
        Ty::Array(e, _) | Ty::Ref { inner: e, .. } | Ty::Owned(e) | Ty::Rc(e) => {
            ty_mentions_generic(e, generics)
        }
        Ty::Tuple(ts) => ts.iter().any(|t| ty_mentions_generic(t, generics)),
        Ty::Fn(ps, r) => {
            ps.iter().any(|t| ty_mentions_generic(t, generics)) || ty_mentions_generic(r, generics)
        }
        _ => false,
    }
}

fn is_int_like(t: &Ty) -> bool {
    matches!(t, Ty::Int(_) | Ty::IntLit)
}

fn is_float_like(t: &Ty) -> bool {
    matches!(t, Ty::Float(_) | Ty::FloatLit)
}

fn is_vectorish(t: &Ty) -> bool {
    matches!(t, Ty::Vec(_) | Ty::Mat(_) | Ty::Quat | Ty::Color)
}

#[cfg(test)]
mod tests;
