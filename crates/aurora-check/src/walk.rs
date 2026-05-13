//! Shared read-only AST walkers used by multiple check passes.

use aurora_ast::{Block, Expr, ExprKind, QueryExpr, Stmt};

/// Collect references to every `query<...>` expression reachable from `block`.
pub(crate) fn queries_in_block<'a>(block: &'a Block) -> Vec<&'a QueryExpr> {
    let mut out = Vec::new();
    walk_block(block, &mut out);
    out
}

fn walk_block<'a>(block: &'a Block, out: &mut Vec<&'a QueryExpr>) {
    for stmt in &block.stmts {
        match stmt {
            Stmt::Let(l) => {
                if let Some(e) = &l.init {
                    walk_expr(e, out);
                }
            }
            Stmt::Defer(e) | Stmt::Expr(e) => walk_expr(e, out),
        }
    }
    if let Some(tail) = &block.tail {
        walk_expr(tail, out);
    }
}

fn walk_expr<'a>(e: &'a Expr, out: &mut Vec<&'a QueryExpr>) {
    match &e.kind {
        ExprKind::Query(q) => {
            out.push(q);
            if let Some(f) = &q.filter {
                walk_expr(f, out);
            }
        }
        ExprKind::Unary(_, a) | ExprKind::Cast(a, _) | ExprKind::Paren(a) => walk_expr(a, out),
        ExprKind::Binary(_, a, b) | ExprKind::Assign(_, a, b) => {
            walk_expr(a, out);
            walk_expr(b, out);
        }
        ExprKind::Pipe { value, func } => {
            walk_expr(value, out);
            walk_expr(func, out);
        }
        ExprKind::Call { callee, args, .. } => {
            walk_expr(callee, out);
            for arg in args {
                walk_expr(&arg.value, out);
            }
        }
        ExprKind::Index { base, index } => {
            walk_expr(base, out);
            walk_expr(index, out);
        }
        ExprKind::Field { base, .. } => walk_expr(base, out),
        ExprKind::Range { start, end, .. } => {
            if let Some(s) = start {
                walk_expr(s, out);
            }
            if let Some(en) = end {
                walk_expr(en, out);
            }
        }
        ExprKind::Struct { fields, base, .. } => {
            for f in fields {
                if let Some(v) = &f.value {
                    walk_expr(v, out);
                }
            }
            if let Some(b) = base {
                walk_expr(b, out);
            }
        }
        ExprKind::Array(items) | ExprKind::Tuple(items) => {
            for it in items {
                walk_expr(it, out);
            }
        }
        ExprKind::ArrayRepeat { value, count } => {
            walk_expr(value, out);
            walk_expr(count, out);
        }
        ExprKind::If(ifx) => {
            walk_expr(&ifx.cond, out);
            walk_block(&ifx.then_branch, out);
            if let Some(e) = &ifx.else_branch {
                walk_expr(e, out);
            }
        }
        ExprKind::Match { scrutinee, arms } => {
            walk_expr(scrutinee, out);
            for arm in arms {
                if let Some(g) = &arm.guard {
                    walk_expr(g, out);
                }
                walk_expr(&arm.body, out);
            }
        }
        ExprKind::For { iter, body, .. } => {
            walk_expr(iter, out);
            walk_block(body, out);
        }
        ExprKind::While { cond, body } => {
            walk_expr(cond, out);
            walk_block(body, out);
        }
        ExprKind::Loop(b) | ExprKind::Block(b) | ExprKind::Unsafe(b) => walk_block(b, out),
        ExprKind::Closure { body, .. } => walk_expr(body, out),
        ExprKind::Spawn(args) => {
            for arg in args {
                walk_expr(&arg.value, out);
            }
        }
        ExprKind::Despawn(e) | ExprKind::Region { value: e, .. } => walk_expr(e, out),
        ExprKind::Try(e) => walk_expr(e, out),
        ExprKind::Return(Some(e)) | ExprKind::Break(Some(e)) => walk_expr(e, out),
        ExprKind::Int(..)
        | ExprKind::Float(..)
        | ExprKind::Str(_)
        | ExprKind::Char(_)
        | ExprKind::Bool(_)
        | ExprKind::Path(_)
        | ExprKind::SelfExpr
        | ExprKind::Dot(_)
        | ExprKind::Return(None)
        | ExprKind::Break(None)
        | ExprKind::Continue
        | ExprKind::Error => {}
    }
}
