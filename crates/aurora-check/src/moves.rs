//! Move checking for owned (`~T`) values (grammar spec §8.1, §8.3).
//!
//! Owned heap values have move semantics: passing one by value, assigning it,
//! returning it, or putting it in a struct/tuple *moves* it; using it afterward
//! is an error. Borrowing (`&x`, `&mut x`) and field/index reads do not move.
//! Reassigning a moved binding revives it.
//!
//! This is sound and self-contained (no cross-function region inference). Only
//! `~T` bindings are tracked; `rc<T>`, references, and Copy values are ignored,
//! so code that doesn't use `~` is never affected.
//!
//! Branches (`if`/`match`) are merged conservatively — a binding moved on *any*
//! path is considered moved afterward — and a move of an outer binding inside a
//! loop body is reported (it would be a use-after-move on the next iteration).

use std::collections::HashMap;

use aurora_ast::{
    AssocItem, Block, Expr, ExprKind, FnDecl, ItemKind, Module, Param, Stmt, TypeKind, UnOp,
};
use aurora_diag::Diagnostic;
use aurora_span::Span;

#[derive(Clone, Copy, PartialEq)]
enum Ctx {
    Move,
    Borrow,
}

struct VarInfo {
    owned: bool,
    moved: Option<Span>,
    decl_loop_depth: usize,
}

pub(crate) fn check_moves(module: &Module, diags: &mut Vec<Diagnostic>) {
    for item in &module.items {
        match &item.kind {
            ItemKind::Fn(f) => check_fn(f, diags),
            ItemKind::System(s) => {
                let mut mc = MoveChk::new(diags);
                for p in &s.params {
                    mc.declare(&p.name.name, is_owned_type(&p.ty.kind));
                }
                mc.block(&s.body);
            }
            ItemKind::Impl(i) => {
                for it in &i.items {
                    if let AssocItem::Fn(f) = it {
                        check_fn(f, diags);
                    }
                }
            }
            _ => {}
        }
    }
}

fn check_fn(f: &FnDecl, diags: &mut Vec<Diagnostic>) {
    let Some(body) = &f.body else { return };
    let mut mc = MoveChk::new(diags);
    for p in &f.params {
        if let Param::Normal { name, ty, .. } = p {
            mc.declare(&name.name, is_owned_type(&ty.kind));
        }
    }
    mc.block(body);
}

struct MoveChk<'d> {
    vars: HashMap<String, VarInfo>,
    diags: &'d mut Vec<Diagnostic>,
    loop_depth: usize,
}

type MovedSnapshot = HashMap<String, Option<Span>>;

impl<'d> MoveChk<'d> {
    fn new(diags: &'d mut Vec<Diagnostic>) -> MoveChk<'d> {
        MoveChk { vars: HashMap::new(), diags, loop_depth: 0 }
    }

    fn declare(&mut self, name: &str, owned: bool) {
        self.vars.insert(
            name.to_string(),
            VarInfo { owned, moved: None, decl_loop_depth: self.loop_depth },
        );
    }

    fn snapshot(&self) -> MovedSnapshot {
        self.vars.iter().map(|(k, v)| (k.clone(), v.moved)).collect()
    }

    fn restore(&mut self, snap: &MovedSnapshot) {
        for (k, v) in self.vars.iter_mut() {
            if let Some(m) = snap.get(k) {
                v.moved = *m;
            }
        }
    }

    /// Merge two branch outcomes: moved if moved on either path.
    fn merge(&mut self, a: &MovedSnapshot, b: &MovedSnapshot) {
        for (k, v) in self.vars.iter_mut() {
            let ma = a.get(k).copied().flatten();
            let mb = b.get(k).copied().flatten();
            v.moved = ma.or(mb);
        }
    }

    fn block(&mut self, block: &Block) {
        for stmt in &block.stmts {
            match stmt {
                Stmt::Let(l) => {
                    if let Some(init) = &l.init {
                        self.expr(init, Ctx::Move);
                    }
                    let owned = l
                        .ty
                        .as_ref()
                        .map(|t| is_owned_type(&t.kind))
                        .unwrap_or(false)
                        || l.init.as_ref().is_some_and(|e| self.produces_owned(e));
                    if let Some(name) = binding_name(&l.pat) {
                        self.declare(&name, owned);
                    }
                }
                Stmt::Defer(e) | Stmt::Expr(e) => self.expr(e, Ctx::Move),
            }
        }
        if let Some(tail) = &block.tail {
            self.expr(tail, Ctx::Move);
        }
    }

    fn expr(&mut self, e: &Expr, ctx: Ctx) {
        match &e.kind {
            ExprKind::Path(p) if p.is_single() => {
                self.use_var(&p.segments[0].ident.name, ctx, e.span);
            }
            ExprKind::Unary(UnOp::RefShared | UnOp::RefMut, inner) => self.expr(inner, Ctx::Borrow),
            ExprKind::Unary(_, inner) => self.expr(inner, Ctx::Move),
            ExprKind::Binary(_, a, b) => {
                self.expr(a, Ctx::Move);
                self.expr(b, Ctx::Move);
            }
            ExprKind::Assign(op, lhs, rhs) => {
                self.expr(rhs, Ctx::Move);
                match &lhs.kind {
                    ExprKind::Path(p) if p.is_single() => {
                        let name = &p.segments[0].ident.name;
                        if op.is_some() {
                            // compound assign reads the old value first
                            self.use_var(name, Ctx::Borrow, lhs.span);
                        }
                        self.revive(name); // plain or compound: lhs is now valid
                    }
                    _ => self.expr(lhs, Ctx::Borrow),
                }
            }
            ExprKind::Cast(inner, _) => self.expr(inner, Ctx::Move),
            ExprKind::Paren(inner) => self.expr(inner, ctx),
            ExprKind::Call { callee, args, .. } => {
                self.expr(callee, Ctx::Borrow);
                for a in args {
                    self.expr(&a.value, Ctx::Move);
                }
            }
            ExprKind::Index { base, index } => {
                self.expr(base, Ctx::Borrow);
                self.expr(index, Ctx::Move);
            }
            ExprKind::Field { base, .. } => self.expr(base, Ctx::Borrow),
            ExprKind::Range { start, end, .. } => {
                if let Some(s) = start {
                    self.expr(s, Ctx::Move);
                }
                if let Some(en) = end {
                    self.expr(en, Ctx::Move);
                }
            }
            ExprKind::Pipe { value, func } => {
                self.expr(value, Ctx::Move);
                self.expr(func, Ctx::Borrow);
            }
            ExprKind::Struct { fields, base, .. } => {
                for f in fields {
                    if let Some(v) = &f.value {
                        self.expr(v, Ctx::Move);
                    } else {
                        self.use_var(&f.name.name, Ctx::Move, f.name.span); // shorthand
                    }
                }
                if let Some(b) = base {
                    self.expr(b, Ctx::Move);
                }
            }
            ExprKind::Array(items) | ExprKind::Tuple(items) => {
                for it in items {
                    self.expr(it, Ctx::Move);
                }
            }
            ExprKind::ArrayRepeat { value, count } => {
                self.expr(value, Ctx::Move);
                self.expr(count, Ctx::Move);
            }
            ExprKind::If(ifx) => {
                self.expr(&ifx.cond, Ctx::Move);
                let base = self.snapshot();
                self.block(&ifx.then_branch);
                let then_snap = self.snapshot();
                self.restore(&base);
                if let Some(else_e) = &ifx.else_branch {
                    self.expr(else_e, ctx);
                }
                let else_snap = self.snapshot();
                self.merge(&then_snap, &else_snap);
            }
            ExprKind::Match { scrutinee, arms } => {
                self.expr(scrutinee, Ctx::Borrow);
                let base = self.snapshot();
                let mut acc = base.clone();
                for arm in arms {
                    self.restore(&base);
                    if let Some(g) = &arm.guard {
                        self.expr(g, Ctx::Borrow);
                    }
                    self.expr(&arm.body, ctx);
                    let arm_snap = self.snapshot();
                    let acc_clone = acc.clone();
                    self.merge(&acc_clone, &arm_snap);
                    acc = self.snapshot();
                }
            }
            ExprKind::For { iter, body, .. } => {
                self.expr(iter, Ctx::Borrow);
                self.loop_depth += 1;
                self.block(body);
                self.loop_depth -= 1;
            }
            ExprKind::While { cond, body } => {
                self.expr(cond, Ctx::Move);
                self.loop_depth += 1;
                self.block(body);
                self.loop_depth -= 1;
            }
            ExprKind::Loop(body) => {
                self.loop_depth += 1;
                self.block(body);
                self.loop_depth -= 1;
            }
            ExprKind::Block(b) | ExprKind::Unsafe(b) => self.block(b),
            ExprKind::Closure { body, .. } => self.expr(body, Ctx::Move),
            ExprKind::Spawn(args) => {
                for a in args {
                    self.expr(&a.value, Ctx::Move);
                }
            }
            ExprKind::Despawn(inner) | ExprKind::Region { value: inner, .. } => {
                self.expr(inner, Ctx::Move);
            }
            ExprKind::Return(opt) | ExprKind::Break(opt) => {
                if let Some(inner) = opt {
                    self.expr(inner, Ctx::Move);
                }
            }
            _ => {}
        }
    }

    fn use_var(&mut self, name: &str, ctx: Ctx, span: Span) {
        let loop_depth = self.loop_depth;
        let Some(v) = self.vars.get_mut(name) else { return };
        if !v.owned {
            return;
        }
        if let Some(move_span) = v.moved {
            self.diags.push(
                Diagnostic::error(format!("use of moved value `{name}`"))
                    .with_code("E0400")
                    .primary(span, "used here after it was moved")
                    .secondary(move_span, "value moved here")
                    .note("`~T` owned values move on use; borrow with `&` to use without moving"),
            );
            return;
        }
        if ctx == Ctx::Move {
            if v.decl_loop_depth < loop_depth {
                self.diags.push(
                    Diagnostic::error(format!(
                        "owned value `{name}` is moved inside a loop"
                    ))
                    .with_code("E0401")
                    .primary(span, "moved here, but the loop repeats")
                    .note("it would be used after move on the next iteration; move a fresh value or borrow"),
                );
            }
            v.moved = Some(span);
        }
    }

    fn revive(&mut self, name: &str) {
        if let Some(v) = self.vars.get_mut(name) {
            if v.owned {
                v.moved = None;
            }
        }
    }

    /// Whether `e` yields an owned value (so a `let` binding of it is owned).
    fn produces_owned(&self, e: &Expr) -> bool {
        match &e.kind {
            ExprKind::Unary(UnOp::Own, _) => true,
            ExprKind::Paren(inner) | ExprKind::Region { value: inner, .. } => {
                self.produces_owned(inner)
            }
            ExprKind::Path(p) if p.is_single() => {
                self.vars.get(&p.segments[0].ident.name).is_some_and(|v| v.owned)
            }
            _ => false,
        }
    }
}

fn is_owned_type(kind: &TypeKind) -> bool {
    matches!(kind, TypeKind::Owned(_))
}

fn binding_name(pat: &aurora_ast::Pat) -> Option<String> {
    match &pat.kind {
        aurora_ast::PatKind::Binding { name, .. } => Some(name.name.clone()),
        _ => None,
    }
}
