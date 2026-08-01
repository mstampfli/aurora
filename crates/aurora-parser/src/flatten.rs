//! Module flattening: lower `mod NAME { items }` into top-level items with
//! `NAME::`-mangled names, rewriting intra-module references so that modules
//! provide real namespacing (two modules may define same-named items without
//! colliding). Runs automatically after parsing.
//!
//! * A definition `mod m { fn f }` becomes a top-level `fn` named `m::f`.
//! * A reference to a sibling (`f` â†’ `m::f`) or a submodule path (`s::g` â†’
//!   `m::s::g`) inside the module is rewritten to the mangled name.
//! * Qualified references from *outside* (`m::f`) are resolved by the backend,
//!   which joins multi-segment call paths with `::`.
//!
//! Nesting is supported (`mod a { mod b { fn f } }` â†’ `a::b::f`).

use std::collections::HashSet;

use aurora_ast::{
    AssocItem, Block, Expr, ExprKind, Item, ItemKind, MatchArm, Param, Pat, PatKind, Stmt, Type,
    TypeKind,
};
use aurora_span::Span;

/// Replace every module in `items` with its flattened, mangled contents, also
/// reporting every reference one module makes to another.
///
/// This pass is the only place that knows the boundary: afterwards `map::room_at` and a local
/// `room_at` are both just mangled names in one flat list, which is why the compiler could not
/// tell a declared dependency from an undeclared reach. The caller holds the manifests and can.
pub fn flatten_modules_tracked(items: Vec<Item>) -> (Vec<Item>, Vec<(String, String, Span)>) {
    let mut refs = Vec::new();
    let out = flatten_modules_into(items, &mut refs);
    (out, refs)
}

fn flatten_modules_into(items: Vec<Item>, refs: &mut Vec<(String, String, Span)>) -> Vec<Item> {
    let mut out = Vec::new();
    for item in items {
        match item.kind {
            ItemKind::Mod(name, Some(inner)) => {
                out.extend(flatten_mod_into(&name.name, inner, refs))
            }
            // A bodiless `mod NAME;` carries no items of its own: the file module
            // loader (`modload.rs`) has already appended `NAME.aur` as an inline
            // `mod NAME { .. }` block, which is flattened above. The declaration
            // itself is therefore nothing left to inline.
            ItemKind::Mod(_, None) => {}
            _ => out.push(item),
        }
    }
    out
}

fn flatten_mod_into(
    prefix: &str,
    items: Vec<Item>,
    refs: &mut Vec<(String, String, Span)>,
) -> Vec<Item> {
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
                flat.extend(flatten_mod_into(
                    &format!("{prefix}::{}", sub.name),
                    inner,
                    refs,
                ));
            }
            // Resolved by the file module loader into a top-level block (see above).
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
    let seen = std::cell::RefCell::new(Vec::new());
    for mut item in own {
        let cx = Cx {
            prefix,
            refs: &seen,
            locals: &locals,
            submods: &submods,
            bound: HashSet::new(),
        };
        rewrite_item(&mut item, &cx);
        mangle_item(&mut item, prefix);
        flat.push(item);
    }
    refs.extend(seen.into_inner());
    flat
}

struct Cx<'a> {
    prefix: &'a str,
    /// Where a reference to ANOTHER module is recorded, as `(from, to, span)`.
    refs: &'a std::cell::RefCell<Vec<(String, String, Span)>>,
    /// Module-level item names (functions/structs/enums/consts) defined here.
    locals: &'a HashSet<String>,
    submods: &'a HashSet<String>,
    /// Names bound by enclosing params / `let`s / closure params / pattern
    /// binders. A reference to one of these is a LOCAL, not a module item, so
    /// it must NOT be module-qualified even if it shares a name with an item.
    bound: HashSet<String>,
}

impl<'a> Cx<'a> {
    /// A child context with additional local bindings in scope.
    fn with_bound(&self, extra: impl IntoIterator<Item = String>) -> Cx<'a> {
        let mut bound = self.bound.clone();
        bound.extend(extra);
        Cx {
            prefix: self.prefix,
            refs: self.refs,
            locals: self.locals,
            submods: self.submods,
            bound,
        }
    }
}

/// Names a binding pattern introduces (so references to them stay local).
fn pat_binding_names(pat: &Pat, out: &mut Vec<String>) {
    match &pat.kind {
        PatKind::Binding { name, sub, .. } => {
            out.push(name.name.clone());
            if let Some(s) = sub {
                pat_binding_names(s, out);
            }
        }
        PatKind::TupleStruct { elems, .. } => {
            for e in elems {
                pat_binding_names(e, out);
            }
        }
        PatKind::Struct { fields, .. } => {
            for f in fields {
                match &f.pat {
                    Some(sub) => pat_binding_names(sub, out),
                    // Shorthand `Struct { x }` binds `x`.
                    None => out.push(f.name.name.clone()),
                }
            }
        }
        PatKind::Tuple(ps) => {
            for p in ps {
                pat_binding_names(p, out);
            }
        }
        _ => {}
    }
}

fn param_names(params: &[Param]) -> Vec<String> {
    params
        .iter()
        .filter_map(|p| match p {
            Param::Normal { name, .. } => Some(name.name.clone()),
            _ => None,
        })
        .collect()
}

fn item_name(item: &Item) -> Option<String> {
    match &item.kind {
        ItemKind::Fn(f) => Some(f.name.name.clone()),
        ItemKind::Struct(s) | ItemKind::Component(s) => Some(s.name.name.clone()),
        ItemKind::Enum(e) => Some(e.name.name.clone()),
        ItemKind::Const(c) => Some(c.name.name.clone()),
        ItemKind::System(s) => Some(s.name.name.clone()),
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
        // A system is an item like any other. Leaving it unmangled let two
        // modules declare the same system name and collide silently, and left
        // `after(other)` in one module able to name a system in another.
        ItemKind::System(s) => set(&mut s.name),
        _ => {}
    }
}

// --- reference rewriting ----------------------------------------------------

/// Parameter and return types, wherever a function is written - top level, in
/// an `impl`, or in a `trait`. One place, because three copies is how the
/// `impl` arm ended up rewriting bodies and not signatures.
fn rewrite_fn_signature(f: &mut aurora_ast::FnDecl, cx: &Cx) {
    for p in &mut f.params {
        if let Param::Normal { ty, .. } = p {
            rewrite_type(ty, cx);
        }
    }
    if let Some(t) = &mut f.ret {
        rewrite_type(t, cx);
    }
}

fn rewrite_item(item: &mut Item, cx: &Cx) {
    match &mut item.kind {
        ItemKind::Fn(f) => {
            rewrite_fn_signature(f, cx);
            // Parameters are in scope for the body: they shadow module items.
            let body_cx = cx.with_bound(param_names(&f.params));
            if let Some(b) = &mut f.body {
                rewrite_block(b, &body_cx);
            }
        }
        ItemKind::Struct(s) | ItemKind::Component(s) => {
            if let aurora_ast::StructBody::Named(fields) = &mut s.body {
                for fd in fields {
                    rewrite_type(&mut fd.ty, cx);
                    // A field default is an expression and can name a sibling
                    // const: `component Spinner { speed: f32 = BASE_SPEED }`.
                    if let Some(d) = &mut fd.default {
                        rewrite_expr(d, cx);
                    }
                }
            }
        }
        // The DECLARED TYPE counts as much as the value. `const T: [str; N]`
        // and `const HOME: Actor` both name sibling items inside the type, and
        // rewriting only the value left the annotation pointing at names that no
        // longer exist after mangling - so the length silently went unresolved
        // and the type silently went unknown.
        ItemKind::Const(c) => {
            if let Some(t) = &mut c.ty {
                rewrite_type(t, cx);
            }
            rewrite_expr(&mut c.value, cx);
        }
        // An enum variant's payload types are types too. Latent rather than
        // reported, because no module has declared `Hit(Actor)` yet - but it is
        // the same miss, and finding it a fourth time by symptom is not a plan.
        ItemKind::Enum(e) => {
            for v in &mut e.variants {
                match &mut v.data {
                    aurora_ast::VariantData::Tuple(ts) => {
                        for t in ts {
                            rewrite_type(t, cx);
                        }
                    }
                    aurora_ast::VariantData::Struct(fields) => {
                        for fd in fields {
                            rewrite_type(&mut fd.ty, cx);
                        }
                    }
                    aurora_ast::VariantData::Unit => {}
                }
                if let Some(d) = &mut v.discriminant {
                    rewrite_expr(d, cx);
                }
            }
        }
        // A system's body names components, and its schedule names sibling
        // systems. Neither was being rewritten, so a system declared in a module
        // could not see its own components: the declaration became `m::Player`
        // while the `query<&mut Player>` inside it did not, and the checker
        // reported a component that was right there in the same file.
        //
        // The stage is deliberately left alone. `stage(FixedUpdate)` names a
        // schedule the runtime owns, not an item in this module, and prefixing
        // it would put every module's systems in a stage of their own.
        ItemKind::System(s) => {
            for sched in &mut s.schedule {
                match sched {
                    aurora_ast::SysSched::After(ps) | aurora_ast::SysSched::Before(ps) => {
                        for p in ps {
                            rewrite_path(p, cx);
                        }
                    }
                    aurora_ast::SysSched::Stage(_) => {}
                }
            }
            rewrite_block(&mut s.body, cx);
        }
        ItemKind::Impl(im) => {
            rewrite_type(&mut im.self_ty, cx);
            if let Some(t) = &mut im.trait_ {
                rewrite_path(t, cx);
            }
            for it in &mut im.items {
                if let AssocItem::Fn(f) = it {
                    rewrite_fn_signature(f, cx);
                    let mut names = param_names(&f.params);
                    names.push("self".into());
                    let body_cx = cx.with_bound(names);
                    if let Some(b) = &mut f.body {
                        rewrite_block(b, &body_cx);
                    }
                }
            }
        }
        // A trait's method signatures name types the same way an impl's do, and
        // a default body is a body.
        ItemKind::Trait(t) => {
            for p in &mut t.supertraits {
                rewrite_path(p, cx);
            }
            for it in &mut t.items {
                if let AssocItem::Fn(f) = it {
                    rewrite_fn_signature(f, cx);
                    let mut names = param_names(&f.params);
                    names.push("self".into());
                    let body_cx = cx.with_bound(names);
                    if let Some(b) = &mut f.body {
                        rewrite_block(b, &body_cx);
                    }
                }
            }
        }
        ItemKind::Pipeline(p) => {
            for f in &mut p.fields {
                rewrite_expr(&mut f.value, cx);
            }
        }
        ItemKind::Comptime(b) => rewrite_block(b, cx),
        ItemKind::Use(_) | ItemKind::Mod(..) | ItemKind::Error => {}
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
        // The LENGTH counts too. `[str; CLIP_COUNT]` inside a module names a
        // sibling const, and dropping it here left the type saying bare
        // `CLIP_COUNT` while the const had been mangled to `scene::CLIP_COUNT`.
        // Neither the type checker nor codegen could then resolve the length, so
        // the array silently became unsized / zero-length - the same class of
        // miss as the two bugs before it, and for the same reason: an array
        // type's length is an EXPRESSION hiding inside a type, and every pass
        // that walks types forgets it.
        TypeKind::Array { elem, len } => {
            rewrite_type(elem, cx);
            if let Some(n) = len {
                rewrite_expr(n, cx);
            }
        }
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
        // Only qualify a bare name that names a MODULE ITEM and is NOT shadowed
        // by a local binding (param / let / pattern binder). Without the shadow
        // check, a parameter that shares a name with a module function (e.g.
        // `fn pick(phase: ...)` alongside `fn phase()`) was wrongly rewritten to
        // `mod::phase`, silently reading the wrong value.
        if cx.locals.contains(n) && !cx.bound.contains(n) {
            p.segments[0].ident.name = format!("{}::{}", cx.prefix, n);
        }
    } else if cx.submods.contains(&p.segments[0].ident.name) {
        let joined = p
            .segments
            .iter()
            .map(|s| s.ident.name.as_str())
            .collect::<Vec<_>>()
            .join("::");
        p.segments[0].ident.name = format!("{}::{}", cx.prefix, joined);
        p.segments.truncate(1);
    } else if p.segments.len() > 1
        && !cx.locals.contains(&p.segments[0].ident.name)
        && !cx.bound.contains(&p.segments[0].ident.name)
    {
        // Neither a sibling item nor a submodule nor a local: this is a reference OUT of this
        // module, which is exactly the edge a dependency graph is made of. Recorded rather than
        // rejected here - the parser has no manifest; the driver does.
        let head = &p.segments[0].ident;
        cx.refs
            .borrow_mut()
            .push((cx.prefix.to_string(), head.name.clone(), head.span));
    } else if cx.locals.contains(&p.segments[0].ident.name)
        && !cx.bound.contains(&p.segments[0].ident.name)
    {
        // A qualified reference THROUGH a module item: `E::Variant` for a sibling
        // enum, or `T::assoc` for a sibling type's associated function. The
        // definition was mangled to `prefix::E`, so only the first segment moves;
        // the rest (the variant / method name) stays a separate segment.
        let n = &p.segments[0].ident.name;
        p.segments[0].ident.name = format!("{}::{}", cx.prefix, n);
    }
}

fn rewrite_block(b: &mut Block, cx: &Cx) {
    // `let` bindings come into scope for SUBSEQUENT statements, so grow the
    // bound set as we go (a later `phase` referring to `let phase = ...` must
    // stay local).
    let mut scope = cx.with_bound(std::iter::empty());
    for stmt in &mut b.stmts {
        match stmt {
            Stmt::Let(l) => {
                if let Some(t) = &mut l.ty {
                    rewrite_type(t, &scope);
                }
                // The initializer is evaluated BEFORE the binding exists.
                if let Some(e) = &mut l.init {
                    rewrite_expr(e, &scope);
                }
                let mut names = Vec::new();
                pat_binding_names(&l.pat, &mut names);
                scope = scope.with_bound(names);
            }
            Stmt::Defer(e) | Stmt::Expr(e) => rewrite_expr(e, &scope),
        }
    }
    if let Some(t) = &mut b.tail {
        rewrite_expr(t, &scope);
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
        ExprKind::Binary(_, a, b)
        | ExprKind::Assign(_, a, b)
        | ExprKind::Index { base: a, index: b } => {
            rewrite_expr(a, cx);
            rewrite_expr(b, cx);
        }
        ExprKind::Pipe { value, func } => {
            rewrite_expr(value, cx);
            rewrite_expr(func, cx);
        }
        ExprKind::Call {
            callee,
            type_args,
            args,
        } => {
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
            // The loop pattern's bindings are in scope for the body.
            rewrite_pat(pat, cx);
            rewrite_expr(iter, cx);
            let mut names = Vec::new();
            pat_binding_names(pat, &mut names);
            rewrite_block(body, &cx.with_bound(names));
        }
        ExprKind::While { cond, body } => {
            rewrite_expr(cond, cx);
            rewrite_block(body, cx);
        }
        ExprKind::Loop(b) | ExprKind::Block(b) | ExprKind::Unsafe(b) => rewrite_block(b, cx),
        ExprKind::Closure { params, body } => {
            // Closure parameters shadow module items inside the body.
            rewrite_expr(body, &cx.with_bound(param_names(params)));
        }
        ExprKind::Spawn(args) => {
            for a in args {
                rewrite_expr(&mut a.value, cx);
            }
        }
        // A query names components, and a component is an item like any other -
        // so a query inside a module has to reach its module's components.
        // Without this, declaring `component Player` and `query<&mut Player>` in
        // one file failed the moment that file became a module: the declaration
        // was mangled and the query was not, and the checker reported a
        // component missing that was three lines above.
        ExprKind::Query(q) => {
            for term in &mut q.terms {
                match term {
                    aurora_ast::QTerm::Read(p)
                    | aurora_ast::QTerm::Write(p)
                    | aurora_ast::QTerm::OptRead(p)
                    | aurora_ast::QTerm::OptWrite(p)
                    | aurora_ast::QTerm::Without(p)
                    | aurora_ast::QTerm::With(p) => rewrite_path(p, cx),
                    aurora_ast::QTerm::Entity => {}
                }
            }
            if let Some(f) = &mut q.filter {
                rewrite_expr(f, cx);
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
    // The arm pattern's bindings are in scope for the guard and body.
    let mut names = Vec::new();
    pat_binding_names(&arm.pat, &mut names);
    let arm_cx = cx.with_bound(names);
    if let Some(g) = &mut arm.guard {
        rewrite_expr(g, &arm_cx);
    }
    rewrite_expr(&mut arm.body, &arm_cx);
}

#[cfg(test)]
mod tests {
    use super::*;
    use aurora_ast::ItemKind;

    fn fn_body_tail_name(items: &[Item], fn_name: &str) -> Option<String> {
        for it in items {
            if let ItemKind::Fn(f) = &it.kind {
                if f.name.name == fn_name {
                    if let Some(tail) = f.body.as_ref().and_then(|b| b.tail.as_ref()) {
                        if let ExprKind::Path(p) = &tail.kind {
                            return Some(p.segments[0].ident.name.clone());
                        }
                    }
                }
            }
        }
        None
    }

    /// A parameter that shares a name with a sibling module function must NOT be
    /// module-qualified: it is a local, and qualifying it silently read the
    /// function instead (regression: boss::pick(phase: ...) beside fn phase()).
    #[test]
    fn param_shadowing_a_module_fn_is_not_qualified() {
        let src =
            "mod m {\n  fn phase(x: i64) -> i64 { x }\n  fn pick(phase: i64) -> i64 { phase }\n}";
        let (module, diags) = crate::parse_str(src);
        assert!(
            !diags.iter().any(|d| d.is_error()),
            "parse errors: {diags:?}"
        );
        let flat = flatten_modules_tracked(module.items).0;
        // The body of m::pick must still reference the bare local `phase`,
        // not the qualified `m::phase`.
        assert_eq!(
            fn_body_tail_name(&flat, "m::pick").as_deref(),
            Some("phase"),
            "the parameter `phase` was wrongly qualified to a module item"
        );
        // Sanity: a genuine reference to the sibling function IS still qualified.
        let src2 = "mod m {\n  fn phase(x: i64) -> i64 { x }\n  fn caller() -> i64 { phase(3) }\n}";
        let (m2, _) = crate::parse_str(src2);
        let flat2 = flatten_modules_tracked(m2.items).0;
        let calls_qualified = flat2.iter().any(|it| {
            if let ItemKind::Fn(f) = &it.kind {
                if f.name.name == "m::caller" {
                    if let Some(ExprKind::Call { callee, .. }) = f
                        .body
                        .as_ref()
                        .and_then(|b| b.tail.as_ref())
                        .map(|t| &t.kind)
                    {
                        if let ExprKind::Path(p) = &callee.kind {
                            return p.segments[0].ident.name == "m::phase";
                        }
                    }
                }
            }
            false
        });
        assert!(
            calls_qualified,
            "a real sibling-fn call must stay qualified"
        );
    }

    /// `[str; N]` inside a module must carry the mangled `m::N`.
    ///
    /// The length of an array type is an expression hiding inside a type, and
    /// `rewrite_type` walked the element and dropped it. The const was mangled
    /// to `m::N` and the type still said `N`, so nothing downstream could
    /// resolve the length: the type checker read an unsized `[str]` and codegen
    /// read zero. In one file it worked, which is why the game only found it
    /// once a real table moved into a module.
    #[test]
    fn an_array_length_naming_a_sibling_const_is_qualified() {
        let src = "mod m {\n  const N: i64 = 3\n  const T: [str; N] = [\"a\", \"b\", \"c\"]\n}";
        let (module, diags) = crate::parse_str(src);
        assert!(
            !diags.iter().any(|d| d.is_error()),
            "parse errors: {diags:?}"
        );
        let flat = flatten_modules_tracked(module.items).0;
        let mut seen = None;
        for it in &flat {
            if let ItemKind::Const(c) = &it.kind {
                if c.name.name == "m::T" {
                    if let aurora_ast::TypeKind::Array { len: Some(n), .. } = &c.ty.as_ref().unwrap().kind {
                        if let ExprKind::Path(p) = &n.kind {
                            seen = Some(p.segments[0].ident.name.clone());
                        }
                    }
                }
            }
        }
        assert_eq!(
            seen.as_deref(),
            Some("m::N"),
            "the array length `N` was left unqualified, so its const cannot be found"
        );
    }
}
