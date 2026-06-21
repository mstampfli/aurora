//! Module flattening: lower `mod NAME { items }` into top-level items with
//! `NAME::`-mangled names, rewriting intra-module references so that modules
//! provide real namespacing (two modules may define same-named items without
//! colliding). Runs automatically after parsing.
//!
//! * A definition `mod m { fn f }` becomes a top-level `fn` named `m::f`.
//! * A reference to a sibling (`f` → `m::f`) or a submodule path (`s::g` →
//!   `m::s::g`) inside the module is rewritten to the mangled name.
//! * Qualified references from *outside* (`m::f`) are resolved by the backend,
//!   which joins multi-segment call paths with `::`.
//!
//! Nesting is supported (`mod a { mod b { fn f } }` → `a::b::f`).

use std::collections::HashSet;

use aurora_ast::{
    AssocItem, Block, Expr, ExprKind, Item, ItemKind, MatchArm, Param, Pat, PatKind, Stmt, Type,
    TypeKind,
};

/// Replace every module in `items` with its flattened, mangled contents.
pub fn flatten_modules(items: Vec<Item>) -> Vec<Item> {
    let mut out = Vec::new();
    for item in items {
        match item.kind {
            ItemKind::Mod(name, Some(inner)) => out.extend(flatten_mod(&name.name, inner)),
            ItemKind::Mod(_, None) => {} // external module declaration: nothing to inline
            _ => out.push(item),
        }
    }
    out
}

fn flatten_mod(prefix: &str, items: Vec<Item>) -> Vec<Item> {
    let mut flat = Vec::new();
    let mut own = Vec::new();
    let mut locals = HashSet::new();
    let mut submods = HashSet::new();

    // Separate nested modules (flattened recursively) from own items, and
    // collect the names visible at this module level.
    for item in items {
        match item.kind {
            ItemKind::Mod(sub, Some(inner)) => {
                submods.insert(sub.name.clone());
                flat.extend(flatten_mod(&format!("{prefix}::{}", sub.name), inner));
            }
            ItemKind::Mod(_, None) => {}
            _ => {
                if let Some(n) = item_name(&item) {
                    locals.insert(n);
                }
                own.push(item);
            }
        }
    }

    // Rewrite references inside each own item, then mangle its defined name.
    for mut item in own {
        let cx = Cx { prefix, locals: &locals, submods: &submods };
        rewrite_item(&mut item, &cx);
        mangle_item(&mut item, prefix);
        flat.push(item);
    }
    flat
}

struct Cx<'a> {
    prefix: &'a str,
    locals: &'a HashSet<String>,
    submods: &'a HashSet<String>,
}

fn item_name(item: &Item) -> Option<String> {
    match &item.kind {
        ItemKind::Fn(f) => Some(f.name.name.clone()),
        ItemKind::Struct(s) | ItemKind::Component(s) => Some(s.name.name.clone()),
        ItemKind::Enum(e) => Some(e.name.name.clone()),
        ItemKind::Const(c) => Some(c.name.name.clone()),
        _ => None,
    }
}

fn mangle_item(item: &mut Item, prefix: &str) {
    let set = |id: &mut aurora_ast::Ident| id.name = format!("{prefix}::{}", id.name);
    match &mut item.kind {
        ItemKind::Fn(f) => set(&mut f.name),
        ItemKind::Struct(s) | ItemKind::Component(s) => set(&mut s.name),
        ItemKind::Enum(e) => set(&mut e.name),
        ItemKind::Const(c) => set(&mut c.name),
        _ => {}
    }
}

// --- reference rewriting ----------------------------------------------------

fn rewrite_item(item: &mut Item, cx: &Cx) {
    match &mut item.kind {
        ItemKind::Fn(f) => {
            for p in &mut f.params {
                if let Param::Normal { ty, .. } = p {
                    rewrite_type(ty, cx);
                }
            }
            if let Some(t) = &mut f.ret {
                rewrite_type(t, cx);
            }
            if let Some(b) = &mut f.body {
                rewrite_block(b, cx);
            }
        }
        ItemKind::Struct(s) | ItemKind::Component(s) => {
            if let aurora_ast::StructBody::Named(fields) = &mut s.body {
                for fd in fields {
                    rewrite_type(&mut fd.ty, cx);
                }
            }
        }
        ItemKind::Const(c) => rewrite_expr(&mut c.value, cx),
        ItemKind::Impl(im) => {
            rewrite_type(&mut im.self_ty, cx);
            for it in &mut im.items {
                if let AssocItem::Fn(f) = it {
                    if let Some(b) = &mut f.body {
                        rewrite_block(b, cx);
                    }
                }
            }
        }
        _ => {}
    }
}

fn rewrite_type(ty: &mut Type, cx: &Cx) {
    // Rewrite module-local type names to their mangled form in EVERY nested type position, not just
    // a bare path: an array element (`[Actor; 9]`), tuple member, fn param/return, dyn trait, or
    // region inner type can all name a sibling struct. Missing these left a module struct field
    // like `[Actor; 9]` with an unmangled element, so field access on it failed ("no field ...").
    match &mut ty.kind {
        TypeKind::Path(p) => {
            rewrite_path(p, cx);
            for seg in &mut p.segments {
                for a in &mut seg.args {
                    rewrite_type(a, cx);
                }
            }
        }
        TypeKind::Dyn(p) => rewrite_path(p, cx),
        TypeKind::Array { elem, .. } => rewrite_type(elem, cx),
        TypeKind::Tuple(ts) => {
            for t in ts {
                rewrite_type(t, cx);
            }
        }
        TypeKind::Fn { params, ret } => {
            for t in params {
                rewrite_type(t, cx);
            }
            rewrite_type(ret, cx);
        }
        TypeKind::Region(_, inner) => rewrite_type(inner, cx),
        _ => {}
    }
}

/// Rewrite a path that names a sibling item or reaches into a submodule.
fn rewrite_path(p: &mut aurora_ast::Path, cx: &Cx) {
    if p.segments.len() == 1 {
        let n = &p.segments[0].ident.name;
        if cx.locals.contains(n) {
            p.segments[0].ident.name = format!("{}::{}", cx.prefix, n);
        }
    } else if cx.submods.contains(&p.segments[0].ident.name) {
        let joined =
            p.segments.iter().map(|s| s.ident.name.as_str()).collect::<Vec<_>>().join("::");
        p.segments[0].ident.name = format!("{}::{}", cx.prefix, joined);
        p.segments.truncate(1);
    }
}

fn rewrite_block(b: &mut Block, cx: &Cx) {
    for stmt in &mut b.stmts {
        match stmt {
            Stmt::Let(l) => {
                if let Some(t) = &mut l.ty {
                    rewrite_type(t, cx);
                }
                if let Some(e) = &mut l.init {
                    rewrite_expr(e, cx);
                }
            }
            Stmt::Defer(e) | Stmt::Expr(e) => rewrite_expr(e, cx),
        }
    }
    if let Some(t) = &mut b.tail {
        rewrite_expr(t, cx);
    }
}

fn rewrite_pat(pat: &mut Pat, cx: &Cx) {
    match &mut pat.kind {
        PatKind::Path(p) => rewrite_path(p, cx),
        PatKind::TupleStruct { path, elems } => {
            rewrite_path(path, cx);
            for e in elems {
                rewrite_pat(e, cx);
            }
        }
        PatKind::Struct { path, fields, .. } => {
            rewrite_path(path, cx);
            for f in fields {
                if let Some(sub) = &mut f.pat {
                    rewrite_pat(sub, cx);
                }
            }
        }
        PatKind::Tuple(ps) => {
            for p in ps {
                rewrite_pat(p, cx);
            }
        }
        PatKind::Binding { sub: Some(s), .. } => rewrite_pat(s, cx),
        _ => {}
    }
}

fn rewrite_expr(e: &mut Expr, cx: &Cx) {
    match &mut e.kind {
        ExprKind::Path(p) => rewrite_path(p, cx),
        ExprKind::Struct { path, fields, base } => {
            rewrite_path(path, cx);
            for f in fields {
                if let Some(v) = &mut f.value {
                    rewrite_expr(v, cx);
                }
            }
            if let Some(b) = base {
                rewrite_expr(b, cx);
            }
        }
        ExprKind::Unary(_, x)
        | ExprKind::Paren(x)
        | ExprKind::Despawn(x)
        | ExprKind::Try(x)
        | ExprKind::Region { value: x, .. } => rewrite_expr(x, cx),
        ExprKind::Cast(x, t) => {
            rewrite_expr(x, cx);
            rewrite_type(t, cx);
        }
        ExprKind::Binary(_, a, b) | ExprKind::Assign(_, a, b) | ExprKind::Index { base: a, index: b } => {
            rewrite_expr(a, cx);
            rewrite_expr(b, cx);
        }
        ExprKind::Pipe { value, func } => {
            rewrite_expr(value, cx);
            rewrite_expr(func, cx);
        }
        ExprKind::Call { callee, type_args, args } => {
            rewrite_expr(callee, cx);
            for t in type_args {
                rewrite_type(t, cx);
            }
            for a in args {
                rewrite_expr(&mut a.value, cx);
            }
        }
        ExprKind::Field { base, .. } => rewrite_expr(base, cx),
        ExprKind::Range { start, end, .. } => {
            if let Some(s) = start {
                rewrite_expr(s, cx);
            }
            if let Some(en) = end {
                rewrite_expr(en, cx);
            }
        }
        ExprKind::Array(xs) | ExprKind::Tuple(xs) => {
            for x in xs {
                rewrite_expr(x, cx);
            }
        }
        ExprKind::ArrayRepeat { value, count } => {
            rewrite_expr(value, cx);
            rewrite_expr(count, cx);
        }
        ExprKind::If(ifx) => rewrite_if(ifx, cx),
        ExprKind::Match { scrutinee, arms } => {
            rewrite_expr(scrutinee, cx);
            for arm in arms {
                rewrite_arm(arm, cx);
            }
        }
        ExprKind::For { pat, iter, body } => {
            rewrite_pat(pat, cx);
            rewrite_expr(iter, cx);
            rewrite_block(body, cx);
        }
        ExprKind::While { cond, body } => {
            rewrite_expr(cond, cx);
            rewrite_block(body, cx);
        }
        ExprKind::Loop(b) | ExprKind::Block(b) | ExprKind::Unsafe(b) => rewrite_block(b, cx),
        ExprKind::Closure { body, .. } => rewrite_expr(body, cx),
        ExprKind::Spawn(args) => {
            for a in args {
                rewrite_expr(&mut a.value, cx);
            }
        }
        ExprKind::Return(o) | ExprKind::Break(o) => {
            if let Some(x) = o {
                rewrite_expr(x, cx);
            }
        }
        _ => {}
    }
}

fn rewrite_if(ifx: &mut aurora_ast::IfExpr, cx: &Cx) {
    rewrite_expr(&mut ifx.cond, cx);
    rewrite_block(&mut ifx.then_branch, cx);
    if let Some(e) = &mut ifx.else_branch {
        rewrite_expr(e, cx);
    }
}

fn rewrite_arm(arm: &mut MatchArm, cx: &Cx) {
    rewrite_pat(&mut arm.pat, cx);
    if let Some(g) = &mut arm.guard {
        rewrite_expr(g, cx);
    }
    rewrite_expr(&mut arm.body, cx);
}
