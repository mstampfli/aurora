//! Region-escape checking (grammar spec §8.2): sound, and now spanning function
//! calls via inferred region-parameterized signatures (each parameter's
//! stored-into region is inferred and enforced at call sites).
//!
//! The region lattice is `'perm ⊐ 'level ⊐ 'frame` (outlives). A value in region
//! `R` may be stored into a location of region `R'` only if `R'` does not outlive
//! `R`. The clearest violation, detectable purely structurally, is **nesting a
//! shorter-lived allocation inside a longer-lived one** — e.g. storing a
//! `#frame` value in a field of a `#perm` allocation: the `#perm` data outlives
//! the frame, so the stored pointer would dangle.
//!
//! This is a *subset* of full region checking (which needs region-parameterized
//! function signatures and per-expression region inference); it does not chase
//! regions through bindings or calls. It only descends *storage positions*
//! (struct fields, array/tuple elements, the value of a region expression, and
//! value-producing tails), so a `#frame` value merely *passed* to a call inside
//! a `#perm` expression is not flagged. Region-free code is never affected.

use aurora_ast::{
    AssocItem, Block, Expr, ExprKind, FnDecl, ItemKind, Module, PatKind, RegionKind, Stmt, Type,
    TypeKind,
};
use aurora_diag::Diagnostic;
use aurora_span::Span;

/// Outlives rank: higher outlives lower.
fn rank(r: RegionKind) -> u8 {
    match r {
        RegionKind::Frame => 0,
        RegionKind::Level => 1,
        RegionKind::Perm => 2,
    }
}

fn region_name(r: RegionKind) -> &'static str {
    match r {
        RegionKind::Frame => "#frame",
        RegionKind::Level => "#level",
        RegionKind::Perm => "#perm",
    }
}

pub(crate) fn check_regions(module: &Module, diags: &mut Vec<Diagnostic>) {
    // First pass: infer each function's *return-region spec* — either a fixed
    // region allocation, or "the region of parameter N" (a region-polymorphic
    // signature like `fn id(x: T) -> T`). This lets calls carry a region across
    // function boundaries, including through region-generic passthroughs.
    let mut fn_regions: std::collections::HashMap<String, RetRegion> =
        std::collections::HashMap::new();
    let mut fn_param_cons: std::collections::HashMap<String, Vec<Option<RegionKind>>> =
        std::collections::HashMap::new();
    for item in &module.items {
        match &item.kind {
            ItemKind::Fn(f) => {
                if let Some(r) = fn_return_spec(f) {
                    fn_regions.insert(f.name.name.clone(), r);
                }
                let cons = compute_param_constraints(f);
                if cons.iter().any(Option::is_some) {
                    fn_param_cons.insert(f.name.name.clone(), cons);
                }
            }
            ItemKind::Impl(i) => {
                for it in &i.items {
                    if let AssocItem::Fn(f) = it {
                        // Only fixed-region returns for methods (arg alignment of
                        // a region-polymorphic `self`/params is out of scope).
                        if let Some(r @ RetRegion::Fixed(_)) = fn_return_spec(f) {
                            fn_regions.insert(f.name.name.clone(), r);
                        }
                    }
                }
            }
            _ => {}
        }
    }
    let mut chk = RegionChk { diags, scopes: Vec::new(), fn_regions, fn_param_cons };
    for item in &module.items {
        match &item.kind {
            ItemKind::Fn(f) => chk.fn_body(f),
            ItemKind::System(s) => chk.block(&s.body, None),
            ItemKind::Const(c) => chk.expr(&c.value, None),
            ItemKind::Impl(i) => {
                for it in &i.items {
                    if let AssocItem::Fn(f) = it {
                        chk.fn_body(f);
                    }
                }
            }
            _ => {}
        }
    }
}

struct RegionChk<'d> {
    diags: &'d mut Vec<Diagnostic>,
    /// Lexical scopes mapping a binding to the region it was allocated in (only
    /// for bindings whose initializer is a known region allocation). This lets
    /// the check chase a region value through a `let` into a later storage
    /// position — a step beyond the purely-inline check.
    scopes: Vec<std::collections::HashMap<String, (RegionKind, Span)>>,
    /// Inferred return-region spec per function, so a call's result carries a
    /// region (fixed, or inherited from an argument).
    fn_regions: std::collections::HashMap<String, RetRegion>,
    /// Inferred per-parameter region constraints per function (the region each
    /// parameter is stored into), so call sites can reject arguments that don't
    /// outlive what the callee stores them in.
    fn_param_cons: std::collections::HashMap<String, Vec<Option<RegionKind>>>,
}

/// How a function's return region is determined.
#[derive(Clone, Copy)]
enum RetRegion {
    /// Always a fixed region (the body allocates in `#region { .. }`).
    Fixed(RegionKind),
    /// The region of the call's Nth argument (region-polymorphic passthrough).
    Param(usize),
}

/// Infer a function's return-region spec from its body tail: a fixed-region
/// allocation, or a bare reference to one of its parameters (so the result
/// inherits that argument's region). `None` when not statically determinable —
/// callers then make no assumption, avoiding false positives.
fn fn_return_spec(f: &FnDecl) -> Option<RetRegion> {
    // An explicit return annotation (`-> #perm T`) is the declared contract and
    // applies even with no body (trait signature / `@extern`).
    if let Some(r) = f.ret.as_ref().and_then(type_region) {
        return Some(RetRegion::Fixed(r));
    }
    let tail = f.body.as_ref()?.tail.as_ref()?;
    if let Some(r) = fixed_region_of(tail) {
        return Some(RetRegion::Fixed(r));
    }
    // `fn id(x: T) -> T { x }`: the tail names parameter N.
    if let ExprKind::Path(p) = &tail.kind {
        if p.segments.len() == 1 {
            let name = &p.segments[0].ident.name;
            let mut idx = 0;
            for prm in &f.params {
                if let aurora_ast::Param::Normal { name: pname, .. } = prm {
                    if &pname.name == name {
                        return Some(RetRegion::Param(idx));
                    }
                    idx += 1;
                }
            }
        }
    }
    None
}

/// An explicit region annotation on a type (`#perm T`), if any.
fn type_region(t: &Type) -> Option<RegionKind> {
    match &t.kind {
        TypeKind::Region(r, _) => Some(*r),
        _ => None,
    }
}

/// The fixed region of an expression that is a region allocation literal,
/// looking through parentheses and block tails. Does not consult bindings.
fn fixed_region_of(e: &Expr) -> Option<RegionKind> {
    match &e.kind {
        ExprKind::Region { region, .. } => Some(*region),
        ExprKind::Paren(inner) => fixed_region_of(inner),
        ExprKind::Block(b) | ExprKind::Unsafe(b) => b.tail.as_ref().and_then(|t| fixed_region_of(t)),
        _ => None,
    }
}

/// Region-parameterized signature inference: for each parameter, the
/// longest-lived region it is *stored into* anywhere in the body (`None` if it
/// never escapes into a region). A call must then pass an argument that outlives
/// that region — e.g. `fn keep(x) { let c = #perm Box { v: x } }` constrains its
/// parameter to `#perm`, so `keep(#frame ..)` is rejected at the call site.
fn compute_param_constraints(f: &FnDecl) -> Vec<Option<RegionKind>> {
    let params: Vec<String> = f
        .params
        .iter()
        .filter_map(|p| match p {
            aurora_ast::Param::Normal { name, .. } => Some(name.name.clone()),
            _ => None,
        })
        .collect();
    let mut cons: Vec<Option<RegionKind>> = vec![None; params.len()];
    // Seed from explicit `#region` parameter annotations — the declared contract,
    // available with no body (trait signature / `@extern`).
    let mut idx = 0;
    for prm in &f.params {
        if let aurora_ast::Param::Normal { ty, .. } = prm {
            if let Some(r) = type_region(ty) {
                cons[idx] = Some(r);
            }
            idx += 1;
        }
    }
    // A visible body can only raise the requirement (`note_store` takes the max).
    if let Some(body) = &f.body {
        cons_block(body, None, &params, &mut cons);
    }
    cons
}

fn param_index(e: &Expr, params: &[String]) -> Option<usize> {
    match &e.kind {
        ExprKind::Path(p) if p.segments.len() == 1 => {
            params.iter().position(|n| n == &p.segments[0].ident.name)
        }
        ExprKind::Paren(inner) => param_index(inner, params),
        _ => None,
    }
}

// Record that `value` (if a bare parameter) is stored into region `enclosing`.
fn note_store(value: &Expr, enclosing: Option<RegionKind>, params: &[String], cons: &mut [Option<RegionKind>]) {
    if let (Some(idx), Some(r)) = (param_index(value, params), enclosing) {
        let keep = matches!(cons[idx], Some(p) if rank(p) >= rank(r));
        if !keep {
            cons[idx] = Some(r);
        }
    }
}

fn cons_block(b: &Block, enclosing: Option<RegionKind>, params: &[String], cons: &mut [Option<RegionKind>]) {
    for stmt in &b.stmts {
        match stmt {
            Stmt::Let(l) => {
                if let Some(e) = &l.init {
                    cons_expr(e, None, params, cons);
                }
            }
            Stmt::Defer(e) | Stmt::Expr(e) => cons_expr(e, None, params, cons),
        }
    }
    if let Some(t) = &b.tail {
        cons_expr(t, enclosing, params, cons);
    }
}

fn cons_expr(e: &Expr, enclosing: Option<RegionKind>, params: &[String], cons: &mut [Option<RegionKind>]) {
    match &e.kind {
        ExprKind::Region { region, value } => cons_expr(value, Some(*region), params, cons),
        ExprKind::Struct { fields, base, .. } => {
            for f in fields {
                if let Some(v) = &f.value {
                    note_store(v, enclosing, params, cons);
                    cons_expr(v, enclosing, params, cons);
                }
            }
            if let Some(b) = base {
                note_store(b, enclosing, params, cons);
                cons_expr(b, enclosing, params, cons);
            }
        }
        ExprKind::Array(items) | ExprKind::Tuple(items) => {
            for it in items {
                note_store(it, enclosing, params, cons);
                cons_expr(it, enclosing, params, cons);
            }
        }
        ExprKind::ArrayRepeat { value, count } => {
            note_store(value, enclosing, params, cons);
            cons_expr(value, enclosing, params, cons);
            cons_expr(count, None, params, cons);
        }
        ExprKind::Paren(inner) => cons_expr(inner, enclosing, params, cons),
        ExprKind::Block(b) | ExprKind::Unsafe(b) => cons_block(b, enclosing, params, cons),
        ExprKind::If(ifx) => {
            cons_expr(&ifx.cond, None, params, cons);
            cons_block(&ifx.then_branch, enclosing, params, cons);
            if let Some(el) = &ifx.else_branch {
                cons_expr(el, enclosing, params, cons);
            }
        }
        ExprKind::Match { scrutinee, arms } => {
            cons_expr(scrutinee, None, params, cons);
            for arm in arms {
                cons_expr(&arm.body, enclosing, params, cons);
            }
        }
        ExprKind::Unary(_, x)
        | ExprKind::Cast(x, _)
        | ExprKind::Try(x)
        | ExprKind::Field { base: x, .. }
        | ExprKind::Despawn(x) => cons_expr(x, None, params, cons),
        ExprKind::Assign(_, a, c)
        | ExprKind::Binary(_, a, c)
        | ExprKind::Index { base: a, index: c }
        | ExprKind::Pipe { value: a, func: c } => {
            cons_expr(a, None, params, cons);
            cons_expr(c, None, params, cons);
        }
        ExprKind::Call { callee, args, .. } => {
            cons_expr(callee, None, params, cons);
            for a in args {
                cons_expr(&a.value, None, params, cons);
            }
        }
        ExprKind::For { iter, body, .. } => {
            cons_expr(iter, None, params, cons);
            cons_block(body, None, params, cons);
        }
        ExprKind::While { cond, body } => {
            cons_expr(cond, None, params, cons);
            cons_block(body, None, params, cons);
        }
        ExprKind::Loop(b) => cons_block(b, None, params, cons),
        ExprKind::Closure { body, .. } => cons_expr(body, None, params, cons),
        ExprKind::Range { start, end, .. } => {
            if let Some(s) = start {
                cons_expr(s, None, params, cons);
            }
            if let Some(en) = end {
                cons_expr(en, None, params, cons);
            }
        }
        ExprKind::Return(Some(x)) | ExprKind::Break(Some(x)) => cons_expr(x, None, params, cons),
        ExprKind::Spawn(args) => {
            for a in args {
                cons_expr(&a.value, None, params, cons);
            }
        }
        _ => {}
    }
}

/// The enclosing region a value is being stored into (region + its span).
type Enclosing = Option<(RegionKind, Span)>;

impl RegionChk<'_> {
    fn fn_body(&mut self, f: &FnDecl) {
        if let Some(body) = &f.body {
            self.block(body, None);
        }
    }

    fn block(&mut self, block: &Block, enclosing: Enclosing) {
        self.scopes.push(std::collections::HashMap::new());
        for stmt in &block.stmts {
            match stmt {
                Stmt::Let(l) => {
                    if let Some(init) = &l.init {
                        self.expr(init, None);
                        // If the initializer is a known region allocation and the
                        // binding is a plain name, remember its region so a later
                        // store of this binding can be checked.
                        if let (Some(reg), PatKind::Binding { name, sub: None }) =
                            (self.region_of(init), &l.pat.kind)
                        {
                            self.scopes
                                .last_mut()
                                .unwrap()
                                .insert(name.name.clone(), reg);
                        }
                    }
                }
                Stmt::Defer(e) | Stmt::Expr(e) => self.expr(e, None),
            }
        }
        // The tail expression is the block's value, so it inherits the storage
        // context.
        if let Some(tail) = &block.tail {
            self.expr(tail, enclosing);
        }
        self.scopes.pop();
    }

    /// The region a binding currently in scope was allocated in, if known.
    fn binding_region(&self, name: &str) -> Option<(RegionKind, Span)> {
        self.scopes.iter().rev().find_map(|s| s.get(name).copied())
    }

    /// The region of an assignable location (`x`, `x.field`, `x[i]`): the region
    /// of the root binding it reaches into. `None` if not a known region binding.
    fn lvalue_region(&self, e: &Expr) -> Option<(RegionKind, Span)> {
        match &e.kind {
            ExprKind::Path(p) if p.segments.len() == 1 => {
                self.binding_region(&p.segments[0].ident.name)
            }
            ExprKind::Field { base, .. } | ExprKind::Index { base, .. } => self.lvalue_region(base),
            ExprKind::Paren(inner) => self.lvalue_region(inner),
            _ => None,
        }
    }

    /// Infer the region of an expression that *is* a region value: a region
    /// allocation literal, a parenthesized/block form of one, or a binding
    /// previously bound to such an allocation. `None` if not statically known.
    fn region_of(&self, e: &Expr) -> Option<(RegionKind, Span)> {
        match &e.kind {
            ExprKind::Region { region, .. } => Some((*region, e.span)),
            ExprKind::Paren(inner) => self.region_of(inner),
            ExprKind::Block(b) | ExprKind::Unsafe(b) => {
                b.tail.as_ref().and_then(|t| self.region_of(t))
            }
            ExprKind::Path(p) if p.segments.len() == 1 => {
                self.binding_region(&p.segments[0].ident.name)
            }
            // A call carries its callee's inferred return region: either a fixed
            // region, or the region of the corresponding argument.
            ExprKind::Call { callee, args, .. } => {
                let ExprKind::Path(p) = &callee.kind else { return None };
                if p.segments.len() != 1 {
                    return None;
                }
                match self.fn_regions.get(&p.segments[0].ident.name).copied() {
                    Some(RetRegion::Fixed(r)) => Some((r, e.span)),
                    Some(RetRegion::Param(i)) => {
                        // Inherit the i-th argument's region (recursively).
                        args.get(i).and_then(|a| self.region_of(&a.value))
                    }
                    None => None,
                }
            }
            _ => None,
        }
    }

    /// At a call site, reject an argument that doesn't outlive the region the
    /// callee stores it into (region-parameterized signature inference, §8.2).
    fn check_call_args(&mut self, callee: &Expr, args: &[aurora_ast::Arg]) {
        let ExprKind::Path(p) = &callee.kind else { return };
        if p.segments.len() != 1 {
            return;
        }
        let Some(cons) = self.fn_param_cons.get(&p.segments[0].ident.name).cloned() else { return };
        for (i, a) in args.iter().enumerate() {
            let Some(Some(need)) = cons.get(i).copied() else { continue };
            if let Some((arg_reg, arg_span)) = self.region_of(&a.value) {
                if rank(arg_reg) < rank(need) {
                    self.diags.push(
                        Diagnostic::error(format!(
                            "{} argument passed where the callee stores it in a longer-lived {} region",
                            region_name(arg_reg),
                            region_name(need)
                        ))
                        .with_code("E0410")
                        .primary(a.value.span, "this value does not live long enough")
                        .secondary(arg_span, "allocated in a shorter-lived region here")
                        .note("the called function keeps this parameter in a longer-lived region, so the argument must outlive it (`'perm ⊐ 'level ⊐ 'frame`)"),
                    );
                }
            }
        }
    }

    /// In a storage position, flag a value whose region is shorter-lived than the
    /// destination (`enclosing`). Covers both inline allocations and bindings.
    fn check_store(&mut self, value: &Expr, enclosing: Enclosing) {
        if let (Some((inner, inner_span)), Some((outer, outer_span))) =
            (self.region_of(value), enclosing)
        {
            // Inline `#region { .. }` allocations are reported by `expr` directly;
            // here we catch *bindings* of a shorter-lived region being stored.
            if rank(inner) < rank(outer) && !matches!(value.kind, ExprKind::Region { .. }) {
                self.diags.push(
                    Diagnostic::error(format!(
                        "{} value stored inside a longer-lived {} allocation",
                        region_name(inner),
                        region_name(outer)
                    ))
                    .with_code("E0410")
                    .primary(value.span, "this value does not live long enough")
                    .secondary(inner_span, "allocated in a shorter-lived region here")
                    .secondary(outer_span, "stored into this longer-lived allocation")
                    .note("a value may only be stored where the destination does not outlive it (`'perm ⊐ 'level ⊐ 'frame`)"),
                );
            }
        }
    }

    fn expr(&mut self, e: &Expr, enclosing: Enclosing) {
        match &e.kind {
            ExprKind::Region { region, value } => {
                // If this allocation is being stored into a longer-lived region,
                // it would dangle.
                if let Some((outer, outer_span)) = enclosing {
                    if rank(*region) < rank(outer) {
                        self.diags.push(
                            Diagnostic::error(format!(
                                "{} value stored inside a longer-lived {} allocation",
                                region_name(*region),
                                region_name(outer)
                            ))
                            .with_code("E0410")
                            .primary(e.span, "this allocation does not live long enough")
                            .secondary(outer_span, "stored into this longer-lived allocation")
                            .note("a value may only be stored where the destination does not outlive it (`'perm ⊐ 'level ⊐ 'frame`)"),
                        );
                    }
                }
                // The region's own value is stored into the region.
                self.expr(value, Some((*region, e.span)));
            }
            // Storage positions: propagate the enclosing region.
            ExprKind::Struct { fields, base, .. } => {
                for f in fields {
                    if let Some(v) = &f.value {
                        self.check_store(v, enclosing);
                        self.expr(v, enclosing);
                    }
                }
                if let Some(b) = base {
                    self.check_store(b, enclosing);
                    self.expr(b, enclosing);
                }
            }
            ExprKind::Array(items) | ExprKind::Tuple(items) => {
                for it in items {
                    self.check_store(it, enclosing);
                    self.expr(it, enclosing);
                }
            }
            ExprKind::ArrayRepeat { value, count } => {
                self.expr(value, enclosing);
                self.expr(count, None);
            }
            ExprKind::Paren(inner) => self.expr(inner, enclosing),
            ExprKind::Block(b) | ExprKind::Unsafe(b) => self.block(b, enclosing),
            ExprKind::If(ifx) => {
                self.expr(&ifx.cond, None);
                self.block(&ifx.then_branch, enclosing);
                if let Some(else_e) = &ifx.else_branch {
                    self.expr(else_e, enclosing);
                }
            }
            ExprKind::Match { scrutinee, arms } => {
                self.expr(scrutinee, None);
                for arm in arms {
                    if let Some(g) = &arm.guard {
                        self.expr(g, None);
                    }
                    self.expr(&arm.body, enclosing);
                }
            }
            // Non-storage positions: recurse to find nested regions, but reset
            // the enclosing context (these values aren't stored into it).
            ExprKind::Unary(_, x) | ExprKind::Cast(x, _) | ExprKind::Try(x) => self.expr(x, None),
            ExprKind::Assign(_, lhs, rhs) => {
                // Storing a shorter-lived value into a longer-lived destination
                // (e.g. `perm_thing.field = frame_value`) escapes — this catches
                // loop-carried and closure-captured stores the literal check
                // can't see.
                if let (Some((dst, dst_span)), Some((src, src_span))) =
                    (self.lvalue_region(lhs), self.region_of(rhs))
                {
                    if rank(src) < rank(dst) {
                        self.diags.push(
                            Diagnostic::error(format!(
                                "{} value assigned into a longer-lived {} location",
                                region_name(src),
                                region_name(dst)
                            ))
                            .with_code("E0410")
                            .primary(rhs.span, "this value does not live long enough")
                            .secondary(src_span, "allocated in a shorter-lived region here")
                            .secondary(dst_span, "assigned into this longer-lived location")
                            .note("a value may only be stored where the destination does not outlive it (`'perm ⊐ 'level ⊐ 'frame`)"),
                        );
                    }
                }
                self.expr(lhs, None);
                self.expr(rhs, None);
            }
            ExprKind::Binary(_, a, b) => {
                self.expr(a, None);
                self.expr(b, None);
            }
            ExprKind::Pipe { value, func } => {
                self.expr(value, None);
                self.expr(func, None);
            }
            ExprKind::Call { callee, args, .. } => {
                self.check_call_args(callee, args);
                self.expr(callee, None);
                for a in args {
                    self.expr(&a.value, None);
                }
            }
            ExprKind::Index { base, index } => {
                self.expr(base, None);
                self.expr(index, None);
            }
            ExprKind::Field { base, .. } => self.expr(base, None),
            ExprKind::Range { start, end, .. } => {
                if let Some(s) = start {
                    self.expr(s, None);
                }
                if let Some(en) = end {
                    self.expr(en, None);
                }
            }
            ExprKind::For { iter, body, .. } => {
                self.expr(iter, None);
                self.block(body, None);
            }
            ExprKind::While { cond, body } => {
                self.expr(cond, None);
                self.block(body, None);
            }
            ExprKind::Loop(b) => self.block(b, None),
            ExprKind::Closure { body, .. } => self.expr(body, None),
            ExprKind::Spawn(args) => {
                for a in args {
                    self.expr(&a.value, None);
                }
            }
            ExprKind::Despawn(inner) => self.expr(inner, None),
            ExprKind::Return(opt) | ExprKind::Break(opt) => {
                if let Some(inner) = opt {
                    self.expr(inner, None);
                }
            }
            _ => {}
        }
    }
}
