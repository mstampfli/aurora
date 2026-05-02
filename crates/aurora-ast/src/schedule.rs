//! Parallel system scheduling (grammar spec §6.2).
//!
//! Groups a module's `system`s into ordered *layers* of mutually-independent
//! systems that may execute concurrently. Two systems land in different layers —
//! preserving their declaration order — whenever they (a) have conflicting
//! component access (one writes a component the other reads or writes) or (b)
//! are explicitly ordered with `after`/`before`. Everything else commutes, so
//! fusing it into one concurrent layer cannot change results. This is the
//! runtime realisation of the data-race-freedom theorem the checker enforces.

use std::collections::BTreeSet;

use crate::{Block, Expr, ExprKind, ItemKind, Module, Path, QTerm, QueryExpr, Stmt, SysSched, SystemDecl};

#[derive(Default)]
struct Access {
    reads: BTreeSet<String>,
    writes: BTreeSet<String>,
}

struct SysInfo {
    name: String,
    access: Access,
    /// Systems this one is explicitly ordered against (either direction).
    ordered_with: BTreeSet<String>,
}

fn last_seg(p: &Path) -> Option<String> {
    p.segments.last().map(|s| s.ident.name.clone())
}

/// Component read/write sets derived from every `query<...>` in the body.
fn access_of(sys: &SystemDecl) -> Access {
    let mut queries = Vec::new();
    walk_block(&sys.body, &mut queries);
    let mut a = Access::default();
    for q in queries {
        for term in &q.terms {
            match term {
                QTerm::Read(p) | QTerm::OptRead(p) => {
                    if let Some(n) = last_seg(p) {
                        a.reads.insert(n);
                    }
                }
                QTerm::Write(p) | QTerm::OptWrite(p) => {
                    if let Some(n) = last_seg(p) {
                        a.writes.insert(n);
                    }
                }
                // Filters / entity id are not data access.
                QTerm::With(_) | QTerm::Without(_) | QTerm::Entity => {}
            }
        }
    }
    a
}

fn ordering_of(sys: &SystemDecl) -> BTreeSet<String> {
    let mut set = BTreeSet::new();
    for s in &sys.schedule {
        if let SysSched::After(ps) | SysSched::Before(ps) = s {
            for p in ps {
                if let Some(n) = last_seg(p) {
                    set.insert(n);
                }
            }
        }
    }
    set
}

/// Two systems conflict when one writes a component the other reads or writes.
fn conflict(a: &Access, b: &Access) -> bool {
    a.writes.iter().any(|c| b.reads.contains(c) || b.writes.contains(c))
        || b.writes.iter().any(|c| a.reads.contains(c) || a.writes.contains(c))
}

/// Group the module's systems (declaration order) into ordered parallel layers.
/// Returns, for each layer, the indices into the declaration-ordered system
/// list — index `k` is the k-th `system` item in `module`. A layer with one
/// index runs sequentially; a layer with several runs them concurrently.
pub fn parallel_layers(module: &Module) -> Vec<Vec<usize>> {
    let infos: Vec<SysInfo> = module
        .items
        .iter()
        .filter_map(|it| match &it.kind {
            ItemKind::System(s) => Some(SysInfo {
                name: s.name.name.clone(),
                access: access_of(s),
                ordered_with: ordering_of(s),
            }),
            _ => None,
        })
        .collect();

    let mut layers: Vec<Vec<usize>> = Vec::new();
    let mut cur: Vec<usize> = Vec::new();
    for i in 0..infos.len() {
        // `i` may join the current layer only if it is independent of, and
        // unordered relative to, every system already placed in it. This keeps
        // every conflicting or explicitly-ordered pair in declaration order.
        let joins = cur.iter().all(|&j| {
            !conflict(&infos[i].access, &infos[j].access)
                && !infos[i].ordered_with.contains(&infos[j].name)
                && !infos[j].ordered_with.contains(&infos[i].name)
        });
        if !joins && !cur.is_empty() {
            layers.push(std::mem::take(&mut cur));
        }
        cur.push(i);
    }
    if !cur.is_empty() {
        layers.push(cur);
    }
    layers
}

// --- query collection (read-only walk over a system body) ------------------

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
