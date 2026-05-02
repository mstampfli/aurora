//! Generic monomorphization: specialize each generic **function** and **struct**
//! per concrete type it's used with, so the backend only sees fully-concrete
//! definitions.
//!
//! * `fn id<T>(x: T) -> T` called `id(7)`/`id(3.5)` → `id$i64`/`id$f64`.
//! * `struct Box<T> { val: T }` + `impl Box<T> { .. }` constructed as
//!   `Box { val: 7 }` → `struct Box$i64` + `impl Box$i64`, with the construction
//!   rewritten to the mangled name (so a real `List<T>` works).
//!
//! Concrete type args are inferred from call-argument and struct-field
//! expressions (literals, struct literals, casts) and from explicit `Box<i64>`
//! type references. Only generic items + their use sites are touched; runs in
//! the codegen path (after type-checking, so the checker still reports generic
//! mismatches). Traits and array-of-`T` fields remain future work.

use std::collections::{HashMap, HashSet};

use crate::{
    AssocItem, Block, Expr, ExprKind, FnDecl, ImplDecl, Item, ItemKind, MatchArm, Pat, PatKind,
    Stmt, StructBody, StructDecl, Type, TypeKind, Vis,
};

/// Generic definitions available to specialize.
struct Gen {
    fns: HashMap<String, FnDecl>,
    structs: HashMap<String, StructDecl>,
    impls: HashMap<String, Vec<ImplDecl>>, // struct name -> impls on it
}

/// Instantiations discovered during rewriting.
#[derive(Default)]
struct Wanted {
    fns: HashSet<(String, Vec<String>)>,
    structs: HashSet<(String, Vec<String>)>,
}

pub fn monomorphize(items: Vec<Item>) -> Result<Vec<Item>, String> {
    let mut gen = Gen { fns: HashMap::new(), structs: HashMap::new(), impls: HashMap::new() };
    for it in &items {
        match &it.kind {
            ItemKind::Fn(f) if !f.generics.is_empty() => {
                gen.fns.insert(f.name.name.clone(), f.clone());
            }
            ItemKind::Struct(s) if !s.generics.is_empty() => {
                gen.structs.insert(s.name.name.clone(), s.clone());
            }
            _ => {}
        }
    }
    // Impls on generic structs (grouped by struct name).
    for it in &items {
        if let ItemKind::Impl(im) = &it.kind {
            if let Some(name) = type_head(&im.self_ty) {
                if gen.structs.contains_key(&name) {
                    gen.impls.entry(name).or_default().push(im.clone());
                }
            }
        }
    }
    if gen.fns.is_empty() && gen.structs.is_empty() {
        // No generic fns/structs, but there may still be generic enums.
        return monomorphize_enums(items);
    }


    let mut wanted = Wanted::default();
    let mut out: Vec<Item> = Vec::new();

    // Pass 1: keep concrete items (rewriting their generic uses); drop generic
    // defs and impls on generic structs.
    for mut it in items {
        let drop = match &it.kind {
            ItemKind::Fn(f) => !f.generics.is_empty(),
            ItemKind::Struct(s) => !s.generics.is_empty(),
            ItemKind::Impl(im) => {
                type_head(&im.self_ty).map(|n| gen.structs.contains_key(&n)).unwrap_or(false)
            }
            _ => false,
        };
        if drop {
            continue;
        }
        rewrite_item(&mut it, &gen, &mut wanted);
        out.push(it);
    }

    // Pass 2: generate specializations until the worklist is dry (a fn body can
    // construct a struct; a method can call a fn).
    let mut done_fns: HashSet<(String, Vec<String>)> = HashSet::new();
    let mut done_structs: HashSet<(String, Vec<String>)> = HashSet::new();
    loop {
        let pend_fns: Vec<_> =
            wanted.fns.iter().filter(|k| !done_fns.contains(*k)).cloned().collect();
        let pend_structs: Vec<_> =
            wanted.structs.iter().filter(|k| !done_structs.contains(*k)).cloned().collect();
        if pend_fns.is_empty() && pend_structs.is_empty() {
            break;
        }
        for inst in pend_fns {
            done_fns.insert(inst.clone());
            if let Some(item) = specialize_fn(&inst, &gen, &mut wanted) {
                out.push(item);
            }
        }
        for inst in pend_structs {
            done_structs.insert(inst.clone());
            specialize_struct(&inst, &gen, &mut wanted, &mut out);
        }
    }
    monomorphize_enums(out)
}

// --- generic enums -----------------------------------------------------------
//
// Specialize each generic enum (e.g. `Result<T, E>`) per concrete instantiation
// the program uses. A single instantiation is rewritten uniformly by name. For
// *multiple* instantiations, a value construction like `Result::Ok(x)` carries
// no type arguments, so we resolve each construction / match / type site to its
// instantiation by propagating concrete types from annotations (function
// returns + params, `let` annotations, call-return types). Each resolution is
// backed by a type the checker already validated, so it cannot be wrong; a site
// that *can't* be resolved becomes a clear error (never silent wrong code), and
// a final scan guarantees no bare generic-enum reference survives into codegen.

/// A concrete enum instantiation: the enum name plus its type-argument heads,
/// e.g. `("Opt", ["f64"])`. Its specialized name is `mangle(enum, targs)`.
type Inst = (String, Vec<String>);

fn monomorphize_enums(items: Vec<Item>) -> Result<Vec<Item>, String> {
    use crate::EnumDecl;

    let genums: HashMap<String, EnumDecl> = items
        .iter()
        .filter_map(|it| match &it.kind {
            ItemKind::Enum(e) if !e.generics.is_empty() => Some((e.name.name.clone(), e.clone())),
            _ => None,
        })
        .collect();
    if genums.is_empty() {
        return Ok(items);
    }

    // Collect concrete instantiations from all type references, plus those
    // inferable from value constructions' payloads (`Opt::Some(2.5)` ⇒ `Opt<f64>`).
    // `ctor_payload` records enums constructed *with* a payload — those must end
    // up specialized or be a clear error, never left generic (which miscompiles
    // an f64 payload as i64 bits).
    let mut insts: HashMap<String, HashSet<Vec<String>>> = HashMap::new();
    for it in &items {
        collect_enum_insts_item(it, &genums, &mut insts);
    }
    let mut ctor_payload: HashSet<String> = HashSet::new();
    for it in &items {
        CtorCollect {
            genums: &genums,
            insts: &mut insts,
            ctor: &mut ctor_payload,
            var_ty: HashMap::new(),
        }
        .item(it);
    }

    // Single-instantiation enums: rewritten uniformly by name (the proven path).
    let single: HashMap<String, (String, Vec<String>)> = insts
        .iter()
        .filter(|(_, set)| set.len() == 1)
        .map(|(name, set)| {
            let targs = set.iter().next().unwrap().clone();
            (name.clone(), (mangle(name, &targs), targs))
        })
        .collect();
    // Multi-instantiation enums: each construction/match resolved by context.
    let multi: HashMap<String, Vec<Vec<String>>> = insts
        .iter()
        .filter(|(_, set)| set.len() >= 2)
        .map(|(name, set)| {
            let mut v: Vec<Vec<String>> = set.iter().cloned().collect();
            v.sort();
            (name.clone(), v)
        })
        .collect();

    // Safety guard: a generic enum constructed with a payload but with no
    // resolvable instantiation (e.g. `Opt::Some(v)` for an unknown-typed `v`,
    // no annotation anywhere) would otherwise pass through generic and
    // miscompile. Reject it cleanly instead.
    let mut guard_errors: Vec<String> = Vec::new();
    for name in &ctor_payload {
        if !single.contains_key(name) && !multi.contains_key(name) {
            guard_errors.push(unresolved_msg(name));
        }
    }

    if single.is_empty() && multi.is_empty() {
        if !guard_errors.is_empty() {
            guard_errors.sort();
            guard_errors.dedup();
            return Err(guard_errors.join("; "));
        }
        return Ok(items);
    }

    let multi_names: HashSet<String> = multi.keys().cloned().collect();
    let tables = (!multi.is_empty()).then(|| build_tables(&items, &genums, &multi_names));

    let mut out = Vec::new();
    let mut errors: Vec<String> = guard_errors;
    for it in items {
        match &it.kind {
            // Enum declarations: emit one specialized copy per instantiation.
            ItemKind::Enum(e) if single.contains_key(&e.name.name) => {
                let (mangled, targs) = &single[&e.name.name];
                out.push(specialize_enum_decl(&it, e, mangled, targs));
            }
            ItemKind::Enum(e) if multi.contains_key(&e.name.name) => {
                for targs in &multi[&e.name.name] {
                    let mangled = mangle(&e.name.name, targs);
                    out.push(specialize_enum_decl(&it, e, &mangled, targs));
                }
            }
            _ => {
                let mut it = it;
                // Multi-instantiation: context-resolve each construction/match,
                // then rewrite explicit-`<T>` type references.
                if let Some(tables) = &tables {
                    let mut r = Resolver { t: tables, cur_ret: None, errors: Vec::new() };
                    r.resolve_item(&mut it);
                    errors.extend(r.errors);
                    rewrite_item_multi_types(&mut it, &multi_names);
                    scan_unresolved_item(&mut it, &multi_names, &mut errors);
                }
                // Single-instantiation: uniform rewrite by name.
                if !single.is_empty() {
                    remap_item_enums(&mut it, &single);
                }
                out.push(it);
            }
        }
    }

    if !errors.is_empty() {
        errors.sort();
        errors.dedup();
        return Err(errors.join("; "));
    }
    Ok(out)
}

fn unresolved_msg(enum_name: &str) -> String {
    format!(
        "could not determine which instantiation of generic enum `{enum_name}` is meant here; \
         annotate the binding or function signature with a concrete type \
         (e.g. `{enum_name}<i64>`) so the compiler can pick the right specialization"
    )
}

/// Build one specialized enum declaration: substitute the generics with `targs`
/// and rename to `mangled`.
fn specialize_enum_decl(it: &Item, e: &crate::EnumDecl, mangled: &str, targs: &[String]) -> Item {
    use crate::VariantData;
    let subst: HashMap<String, String> =
        e.generics.iter().map(|g| g.name.name.clone()).zip(targs.iter().cloned()).collect();
    let mut spec = e.clone();
    spec.name.name = mangled.to_string();
    spec.generics.clear();
    for v in &mut spec.variants {
        match &mut v.data {
            VariantData::Tuple(tys) => {
                for t in tys {
                    subst_type(t, &subst);
                }
            }
            VariantData::Struct(fs) => {
                for f in fs {
                    subst_type(&mut f.ty, &subst);
                }
            }
            VariantData::Unit => {}
        }
    }
    Item { attrs: it.attrs.clone(), vis: it.vis, kind: ItemKind::Enum(spec), span: it.span }
}

/// If `t` names a multi-instantiation generic enum with concrete args, return
/// its instantiation (`("Opt", ["f64"])`).
fn type_to_inst(t: &Type, multi: &HashSet<String>) -> Option<Inst> {
    if let TypeKind::Path(p) = &t.kind {
        if let Some(seg) = p.segments.last() {
            if multi.contains(&seg.ident.name) && !seg.args.is_empty() {
                let targs: Vec<String> = seg.args.iter().filter_map(type_head).collect();
                if targs.len() == seg.args.len() {
                    return Some((seg.ident.name.clone(), targs));
                }
            }
        }
    }
    None
}

/// Infer a generic enum's instantiation from a value construction, by matching
/// the payload argument types against the variant's declared payload types
/// (`Opt::Some(2.5)` ⇒ `Opt<f64>`). Returns `Some` only when every generic of
/// the enum is determined by the payload — otherwise the caller must rely on an
/// annotation (and error if there is none), never guess.
fn infer_inst_from_construction(
    edecl: &crate::EnumDecl,
    variant: &str,
    args: &[&Expr],
    vars: &HashMap<String, String>,
) -> Option<Inst> {
    use crate::VariantData;
    if edecl.generics.is_empty() {
        return None;
    }
    let gnames: HashSet<&str> = edecl.generics.iter().map(|g| g.name.name.as_str()).collect();
    let var = edecl.variants.iter().find(|v| v.name.name == variant)?;
    let mut map: HashMap<String, String> = HashMap::new();
    // Post-monomorphization there are no generic structs left, so `guess_type`
    // needs no generic context here; `vars` resolves variable payloads.
    let cgen = Gen { fns: HashMap::new(), structs: HashMap::new(), impls: HashMap::new() };
    if let VariantData::Tuple(tys) = &var.data {
        for (ty, arg) in tys.iter().zip(args.iter()) {
            bind_generics(ty, arg, &gnames, &mut map, &cgen, vars);
        }
    }
    let targs: Vec<String> =
        edecl.generics.iter().map(|g| map.get(&g.name.name).cloned()).collect::<Option<_>>()?;
    Some((edecl.name.name.clone(), targs))
}

/// Walk an item's value constructions of generic enums. For each, add any
/// payload-inferable instantiation to `insts`, and record the enum in `ctor`
/// (it has a payload-carrying construction, so it must end up specialized or be
/// a clear error — never left generic to miscompile).
struct CtorCollect<'a> {
    genums: &'a HashMap<String, crate::EnumDecl>,
    insts: &'a mut HashMap<String, HashSet<Vec<String>>>,
    ctor: &'a mut HashSet<String>,
    /// Variable → type-head within the function being walked, so a construction
    /// with a *variable* payload (`Opt::Some(v)`) can infer its instantiation.
    var_ty: HashMap<String, String>,
}

impl CtorCollect<'_> {
    fn item(&mut self, it: &Item) {
        let body = match &it.kind {
            ItemKind::Fn(f) => f.body.as_ref(),
            ItemKind::Impl(im) => {
                for ai in &im.items {
                    if let AssocItem::Fn(f) = ai {
                        if let Some(b) = &f.body {
                            self.var_ty.clear();
                            self.bind_params(&f.params);
                            self.block(b);
                        }
                    }
                }
                None
            }
            ItemKind::System(s) => Some(&s.body),
            ItemKind::Const(c) => {
                self.expr(&c.value);
                None
            }
            _ => None,
        };
        if let Some(b) = body {
            self.var_ty.clear();
            if let ItemKind::Fn(f) = &it.kind {
                self.bind_params(&f.params);
            }
            self.block(b);
        }
    }

    fn bind_params(&mut self, params: &[crate::Param]) {
        for p in params {
            if let crate::Param::Normal { name, ty, .. } = p {
                if let Some(h) = type_head(ty) {
                    self.var_ty.insert(name.name.clone(), h);
                }
            }
        }
    }

    fn block(&mut self, b: &Block) {
        for stmt in &b.stmts {
            match stmt {
                Stmt::Let(l) => {
                    if let Some(e) = &l.init {
                        self.expr(e);
                    }
                    // Track the binding's type so later constructions can use it.
                    if let PatKind::Binding { name, sub: None } = &l.pat.kind {
                        let cgen = Gen { fns: HashMap::new(), structs: HashMap::new(), impls: HashMap::new() };
                        let t = l
                            .ty
                            .as_ref()
                            .and_then(type_head)
                            .or_else(|| l.init.as_ref().and_then(|e| guess_type(e, &cgen, &self.var_ty)));
                        if let Some(t) = t {
                            self.var_ty.insert(name.name.clone(), t);
                        }
                    }
                }
                Stmt::Defer(e) | Stmt::Expr(e) => self.expr(e),
            }
        }
        if let Some(t) = &b.tail {
            self.expr(t);
        }
    }

    fn expr(&mut self, e: &Expr) {
        // Inspect a generic-enum tuple construction at this node.
        if let ExprKind::Call { callee, args, .. } = &e.kind {
            if let ExprKind::Path(p) = &callee.kind {
                if p.segments.len() >= 2 {
                    if let Some(edecl) = self.genums.get(&p.segments[0].ident.name) {
                        let variant = &p.segments.last().unwrap().ident.name;
                        let argrefs: Vec<&Expr> = args.iter().map(|a| &a.value).collect();
                        if !argrefs.is_empty() {
                            self.ctor.insert(edecl.name.name.clone());
                        }
                        if let Some((name, targs)) =
                            infer_inst_from_construction(edecl, variant, &argrefs, &self.var_ty)
                        {
                            self.insts.entry(name).or_default().insert(targs);
                        }
                    }
                }
            }
        }
        // Recurse into children.
        match &e.kind {
            ExprKind::Unary(_, x)
            | ExprKind::Paren(x)
            | ExprKind::Cast(x, _)
            | ExprKind::Try(x)
            | ExprKind::Despawn(x)
            | ExprKind::Region { value: x, .. }
            | ExprKind::Field { base: x, .. } => self.expr(x),
            ExprKind::Binary(_, a, b)
            | ExprKind::Assign(_, a, b)
            | ExprKind::Index { base: a, index: b } => {
                self.expr(a);
                self.expr(b);
            }
            ExprKind::Pipe { value, func } => {
                self.expr(value);
                self.expr(func);
            }
            ExprKind::Call { callee, args, .. } => {
                self.expr(callee);
                for a in args {
                    self.expr(&a.value);
                }
            }
            ExprKind::Array(xs) | ExprKind::Tuple(xs) => {
                for x in xs {
                    self.expr(x);
                }
            }
            ExprKind::Spawn(args) => {
                for a in args {
                    self.expr(&a.value);
                }
            }
            ExprKind::ArrayRepeat { value, count } => {
                self.expr(value);
                self.expr(count);
            }
            ExprKind::Struct { fields, base, .. } => {
                for fld in fields {
                    if let Some(v) = &fld.value {
                        self.expr(v);
                    }
                }
                if let Some(b) = base {
                    self.expr(b);
                }
            }
            ExprKind::If(ifx) => {
                self.expr(&ifx.cond);
                self.block(&ifx.then_branch);
                if let Some(el) = &ifx.else_branch {
                    self.expr(el);
                }
            }
            ExprKind::Match { scrutinee, arms } => {
                self.expr(scrutinee);
                for arm in arms {
                    if let Some(g) = &arm.guard {
                        self.expr(g);
                    }
                    self.expr(&arm.body);
                }
            }
            ExprKind::For { iter, body, .. } => {
                self.expr(iter);
                self.block(body);
            }
            ExprKind::While { cond, body } => {
                self.expr(cond);
                self.block(body);
            }
            ExprKind::Loop(b) | ExprKind::Block(b) | ExprKind::Unsafe(b) => self.block(b),
            ExprKind::Closure { body, .. } => self.expr(body),
            ExprKind::Range { start, end, .. } => {
                if let Some(s) = start {
                    self.expr(s);
                }
                if let Some(en) = end {
                    self.expr(en);
                }
            }
            ExprKind::Return(Some(x)) | ExprKind::Break(Some(x)) => self.expr(x),
            _ => {}
        }
    }
}

/// Function signature instantiation tables, used to propagate enum types across
/// call boundaries (a call's result type; an argument's expected type).
struct Tables {
    multi: HashSet<String>,
    fn_ret: HashMap<String, Inst>,
    fn_params: HashMap<String, Vec<Option<Inst>>>,
    /// Generic enum declarations (for payload-based instantiation inference).
    enums: HashMap<String, crate::EnumDecl>,
}

fn build_tables(
    items: &[Item],
    genums: &HashMap<String, crate::EnumDecl>,
    multi: &HashSet<String>,
) -> Tables {
    let mut fn_ret = HashMap::new();
    let mut fn_params = HashMap::new();
    let mut add = |f: &FnDecl| {
        if let Some(ret) = &f.ret {
            if let Some(inst) = type_to_inst(ret, multi) {
                fn_ret.insert(f.name.name.clone(), inst);
            }
        }
        let params: Vec<Option<Inst>> = f
            .params
            .iter()
            .map(|p| match p {
                crate::Param::Normal { ty, .. } => type_to_inst(ty, multi),
                _ => None,
            })
            .collect();
        fn_params.insert(f.name.name.clone(), params);
    };
    for it in items {
        match &it.kind {
            ItemKind::Fn(f) => add(f),
            ItemKind::Impl(im) => {
                for ai in &im.items {
                    if let AssocItem::Fn(f) = ai {
                        add(f);
                    }
                }
            }
            _ => {}
        }
    }
    Tables { multi: multi.clone(), fn_ret, fn_params, enums: genums.clone() }
}

/// Context resolver for multi-instantiation enums.
struct Resolver<'a> {
    t: &'a Tables,
    cur_ret: Option<Inst>,
    errors: Vec<String>,
}

impl Resolver<'_> {
    fn resolve_item(&mut self, it: &mut Item) {
        match &mut it.kind {
            ItemKind::Fn(f) => self.resolve_fn(f),
            ItemKind::Impl(im) => {
                for ai in &mut im.items {
                    if let AssocItem::Fn(f) = ai {
                        self.resolve_fn(f);
                    }
                }
            }
            ItemKind::System(s) => {
                self.cur_ret = None;
                let mut vars = HashMap::new();
                self.resolve_block(&mut s.body, None, &mut vars);
            }
            ItemKind::Const(c) => {
                self.cur_ret = None;
                let mut vars = HashMap::new();
                self.resolve_expr(&mut c.value, None, &mut vars);
            }
            _ => {}
        }
    }

    fn resolve_fn(&mut self, f: &mut FnDecl) {
        let ret = f.ret.as_ref().and_then(|t| type_to_inst(t, &self.t.multi));
        let mut vars: HashMap<String, Inst> = HashMap::new();
        for p in &f.params {
            if let crate::Param::Normal { name, ty, .. } = p {
                if let Some(inst) = type_to_inst(ty, &self.t.multi) {
                    vars.insert(name.name.clone(), inst);
                }
            }
        }
        self.cur_ret = ret.clone();
        if let Some(b) = &mut f.body {
            self.resolve_block(b, ret, &mut vars);
        }
    }

    /// Resolve a block; `tail_expected` is the expected instantiation of the
    /// block's value (tail expression). Variables are lexically scoped.
    fn resolve_block(&mut self, b: &mut Block, tail_expected: Option<Inst>, vars: &mut HashMap<String, Inst>) {
        let saved = vars.clone();
        for stmt in &mut b.stmts {
            match stmt {
                Stmt::Let(l) => {
                    let ann = l.ty.as_ref().and_then(|t| type_to_inst(t, &self.t.multi));
                    // Determine the binding's instantiation from the annotation
                    // or the *not-yet-rewritten* initializer (so a construction's
                    // payload is still visible), before resolving rewrites it.
                    let bind_inst =
                        ann.clone().or_else(|| l.init.as_ref().and_then(|e| self.expr_inst(e, vars)));
                    if let Some(init) = &mut l.init {
                        self.resolve_expr(init, bind_inst.clone(), vars);
                    }
                    if let PatKind::Binding { name, sub: None } = &l.pat.kind {
                        if let Some(inst) = bind_inst {
                            vars.insert(name.name.clone(), inst);
                        }
                    }
                }
                Stmt::Defer(e) | Stmt::Expr(e) => self.resolve_expr(e, None, vars),
            }
        }
        if let Some(t) = &mut b.tail {
            self.resolve_expr(t, tail_expected, vars);
        }
        *vars = saved;
    }

    /// The instantiation a value expression evaluates to, if statically known
    /// from a variable's tracked type, a function's return type, or a variant
    /// construction's payload (`Opt::Some(2.5)` ⇒ `Opt<f64>`).
    fn expr_inst(&self, e: &Expr, vars: &HashMap<String, Inst>) -> Option<Inst> {
        match &e.kind {
            ExprKind::Path(p) if p.segments.len() == 1 => {
                vars.get(&p.segments[0].ident.name).cloned()
            }
            ExprKind::Call { callee, args, .. } => {
                if let ExprKind::Path(p) = &callee.kind {
                    if self.is_variant_path(p) {
                        let variant = &p.segments.last().unwrap().ident.name;
                        let edecl = self.t.enums.get(&p.segments[0].ident.name)?;
                        let argrefs: Vec<&Expr> = args.iter().map(|a| &a.value).collect();
                        return infer_inst_from_construction(edecl, variant, &argrefs, &HashMap::new());
                    }
                    if p.segments.len() == 1 {
                        return self.t.fn_ret.get(&p.segments[0].ident.name).cloned();
                    }
                }
                None
            }
            ExprKind::Paren(inner) => self.expr_inst(inner, vars),
            _ => None,
        }
    }

    /// Rewrite a construction/variant path head to its mangled instantiation;
    /// record an error if it can't be determined.
    fn resolve_variant_path(&mut self, p: &mut crate::Path, inst: Option<Inst>) {
        let enum_name = p.segments[0].ident.name.clone();
        match inst {
            Some((en, targs)) if en == enum_name => {
                p.segments[0].ident.name = mangle(&enum_name, &targs);
            }
            _ => self.errors.push(unresolved_msg(&enum_name)),
        }
    }

    /// Infer a construction's instantiation from its payload (no annotation).
    fn payload_inst(&self, p: &crate::Path, args: &[crate::Arg]) -> Option<Inst> {
        let variant = &p.segments.last()?.ident.name;
        let edecl = self.t.enums.get(&p.segments[0].ident.name)?;
        let argrefs: Vec<&Expr> = args.iter().map(|a| &a.value).collect();
        infer_inst_from_construction(edecl, variant, &argrefs, &HashMap::new())
    }

    fn is_variant_path(&self, p: &crate::Path) -> bool {
        p.segments.len() >= 2 && self.t.multi.contains(&p.segments[0].ident.name)
    }

    fn resolve_expr(&mut self, e: &mut Expr, expected: Option<Inst>, vars: &mut HashMap<String, Inst>) {
        match &mut e.kind {
            ExprKind::Call { callee, args, .. } => {
                if let ExprKind::Path(p) = &mut callee.kind {
                    if self.is_variant_path(p) {
                        // Resolve from the expected type, else infer from payload.
                        let inst = expected.clone().or_else(|| self.payload_inst(p, args));
                        self.resolve_variant_path(p, inst);
                        for a in args {
                            self.resolve_expr(&mut a.value, None, vars);
                        }
                        return;
                    }
                    if p.segments.len() == 1 {
                        if let Some(params) = self.t.fn_params.get(&p.segments[0].ident.name).cloned()
                        {
                            for (i, a) in args.iter_mut().enumerate() {
                                let exp = params.get(i).cloned().flatten();
                                self.resolve_expr(&mut a.value, exp, vars);
                            }
                            return;
                        }
                    }
                }
                self.resolve_expr(callee, None, vars);
                for a in args {
                    self.resolve_expr(&mut a.value, None, vars);
                }
            }
            ExprKind::Path(p) => {
                if self.is_variant_path(p) {
                    // Unit variant (e.g. `Opt::None`): only the expected type can
                    // pin it — there is no payload to infer from.
                    self.resolve_variant_path(p, expected.clone());
                }
            }
            ExprKind::Match { scrutinee, arms } => {
                self.resolve_expr(scrutinee, None, vars);
                let sinst = self.expr_inst(scrutinee, vars);
                for arm in arms.iter_mut() {
                    self.resolve_pat(&mut arm.pat, sinst.as_ref());
                    if let Some(g) = &mut arm.guard {
                        self.resolve_expr(g, None, vars);
                    }
                    let mut scoped = vars.clone();
                    self.resolve_expr(&mut arm.body, expected.clone(), &mut scoped);
                }
            }
            ExprKind::If(ifx) => {
                self.resolve_expr(&mut ifx.cond, None, vars);
                self.resolve_block(&mut ifx.then_branch, expected.clone(), vars);
                if let Some(el) = &mut ifx.else_branch {
                    self.resolve_expr(el, expected, vars);
                }
            }
            ExprKind::Block(b) => self.resolve_block(b, expected, vars),
            ExprKind::Loop(b) | ExprKind::Unsafe(b) => {
                self.resolve_block(b, None, vars);
            }
            ExprKind::Paren(inner) => self.resolve_expr(inner, expected, vars),
            ExprKind::Return(Some(inner)) => {
                let r = self.cur_ret.clone();
                self.resolve_expr(inner, r, vars);
            }
            ExprKind::Binary(_, a, b) | ExprKind::Assign(_, a, b) | ExprKind::Index { base: a, index: b } => {
                self.resolve_expr(a, None, vars);
                self.resolve_expr(b, None, vars);
            }
            ExprKind::Pipe { value, func } => {
                self.resolve_expr(value, None, vars);
                self.resolve_expr(func, None, vars);
            }
            ExprKind::Unary(_, x)
            | ExprKind::Cast(x, _)
            | ExprKind::Try(x)
            | ExprKind::Despawn(x)
            | ExprKind::Region { value: x, .. }
            | ExprKind::Field { base: x, .. } => self.resolve_expr(x, None, vars),
            ExprKind::Array(xs) | ExprKind::Tuple(xs) => {
                for x in xs {
                    self.resolve_expr(x, None, vars);
                }
            }
            ExprKind::Spawn(args) => {
                for a in args {
                    self.resolve_expr(&mut a.value, None, vars);
                }
            }
            ExprKind::ArrayRepeat { value, count } => {
                self.resolve_expr(value, None, vars);
                self.resolve_expr(count, None, vars);
            }
            ExprKind::Struct { fields, base, .. } => {
                for f in fields {
                    if let Some(v) = &mut f.value {
                        self.resolve_expr(v, None, vars);
                    }
                }
                if let Some(b) = base {
                    self.resolve_expr(b, None, vars);
                }
            }
            ExprKind::For { iter, body, .. } => {
                self.resolve_expr(iter, None, vars);
                self.resolve_block(body, None, vars);
            }
            ExprKind::While { cond, body } => {
                self.resolve_expr(cond, None, vars);
                self.resolve_block(body, None, vars);
            }
            ExprKind::Closure { body, .. } => {
                let mut scoped = vars.clone();
                self.resolve_expr(body, None, &mut scoped);
            }
            ExprKind::Range { start, end, .. } => {
                if let Some(s) = start {
                    self.resolve_expr(s, None, vars);
                }
                if let Some(en) = end {
                    self.resolve_expr(en, None, vars);
                }
            }
            ExprKind::Return(None)
            | ExprKind::Break(_)
            | ExprKind::Int(..)
            | ExprKind::Float(..)
            | ExprKind::Str(_)
            | ExprKind::Char(_)
            | ExprKind::Bool(_)
            | ExprKind::SelfExpr
            | ExprKind::Dot(_)
            | ExprKind::Continue
            | ExprKind::Query(_)
            | ExprKind::Error => {}
        }
    }

    fn resolve_pat(&mut self, pat: &mut Pat, sinst: Option<&Inst>) {
        let head = match &pat.kind {
            PatKind::Path(p) | PatKind::TupleStruct { path: p, .. } | PatKind::Struct { path: p, .. }
                if self.is_variant_path(p) =>
            {
                Some(p.segments[0].ident.name.clone())
            }
            _ => None,
        };
        if let Some(en) = head {
            match sinst {
                Some((sen, targs)) if sen == &en => {
                    if let PatKind::Path(p)
                    | PatKind::TupleStruct { path: p, .. }
                    | PatKind::Struct { path: p, .. } = &mut pat.kind
                    {
                        p.segments[0].ident.name = mangle(&en, targs);
                    }
                }
                _ => self.errors.push(unresolved_msg(&en)),
            }
        }
        match &mut pat.kind {
            PatKind::TupleStruct { elems, .. } => {
                for e in elems {
                    self.resolve_pat(e, None);
                }
            }
            PatKind::Struct { fields, .. } => {
                for f in fields {
                    if let Some(s) = &mut f.pat {
                        self.resolve_pat(s, None);
                    }
                }
            }
            PatKind::Tuple(ps) => {
                for p in ps {
                    self.resolve_pat(p, None);
                }
            }
            PatKind::Binding { sub: Some(s), .. } => self.resolve_pat(s, None),
            _ => {}
        }
    }
}

/// Rewrite explicit-`<T>` type references to multi-instantiation enums (which
/// carry concrete args) to their mangled names, throughout an item.
fn rewrite_item_multi_types(it: &mut Item, multi: &HashSet<String>) {
    walk_item_types_mut(it, &mut |t: &mut Type| rewrite_type_multi(t, multi));
}

fn rewrite_type_multi(t: &mut Type, multi: &HashSet<String>) {
    if let TypeKind::Path(p) = &mut t.kind {
        if let Some(seg) = p.segments.last_mut() {
            if multi.contains(&seg.ident.name) && !seg.args.is_empty() {
                let targs: Vec<String> = seg.args.iter().filter_map(type_head).collect();
                if targs.len() == seg.args.len() {
                    seg.ident.name = mangle(&seg.ident.name, &targs);
                    seg.args.clear();
                    return;
                }
            }
            for a in &mut seg.args {
                rewrite_type_multi(a, multi);
            }
        }
    }
}

/// Safety net: after resolution + type rewriting, any *bare* reference to a
/// multi-instantiation enum left in a type annotation is unresolved — record an
/// error so it can never reach codegen and miscompile. (Constructions and match
/// patterns are caught directly by the resolver's complete expression walk.)
fn scan_unresolved_item(it: &mut Item, multi: &HashSet<String>, errors: &mut Vec<String>) {
    let mut found: HashSet<String> = HashSet::new();
    walk_item_types_mut(it, &mut |t: &mut Type| scan_type(t, multi, &mut found));
    for name in found {
        errors.push(unresolved_msg(&name));
    }
}

fn scan_type(t: &Type, multi: &HashSet<String>, found: &mut HashSet<String>) {
    if let TypeKind::Path(p) = &t.kind {
        if let Some(seg) = p.segments.last() {
            if multi.contains(&seg.ident.name) {
                found.insert(seg.ident.name.clone());
            }
            for a in &seg.args {
                scan_type(a, multi, found);
            }
        }
    } else {
        match &t.kind {
            TypeKind::Owned(i) | TypeKind::Ref { inner: i, .. } | TypeKind::Array { elem: i, .. } => {
                scan_type(i, multi, found)
            }
            TypeKind::Tuple(ts) => {
                for x in ts {
                    scan_type(x, multi, found);
                }
            }
            TypeKind::Fn { params, ret } => {
                for x in params {
                    scan_type(x, multi, found);
                }
                scan_type(ret, multi, found);
            }
            _ => {}
        }
    }
}

// --- mutable walk over every type annotation in an item --------------------

fn walk_item_types_mut(it: &mut Item, f: &mut impl FnMut(&mut Type)) {
    match &mut it.kind {
        ItemKind::Fn(fd) => walk_fn_types_mut(fd, f),
        ItemKind::Impl(im) => {
            for ai in &mut im.items {
                if let AssocItem::Fn(fd) = ai {
                    walk_fn_types_mut(fd, f);
                }
            }
        }
        ItemKind::Struct(s) | ItemKind::Component(s) => {
            if let StructBody::Named(fields) = &mut s.body {
                for fld in fields {
                    f(&mut fld.ty);
                }
            }
        }
        ItemKind::Const(c) => {
            if let Some(t) = &mut c.ty {
                f(t);
            }
            walk_expr_types_mut(&mut c.value, f);
        }
        ItemKind::System(s) => walk_block_types_mut(&mut s.body, f),
        _ => {}
    }
}

fn walk_fn_types_mut(fd: &mut FnDecl, f: &mut impl FnMut(&mut Type)) {
    for p in &mut fd.params {
        if let crate::Param::Normal { ty, .. } = p {
            f(ty);
        }
    }
    if let Some(t) = &mut fd.ret {
        f(t);
    }
    if let Some(b) = &mut fd.body {
        walk_block_types_mut(b, f);
    }
}

fn walk_block_types_mut(b: &mut Block, f: &mut impl FnMut(&mut Type)) {
    for stmt in &mut b.stmts {
        match stmt {
            Stmt::Let(l) => {
                if let Some(t) = &mut l.ty {
                    f(t);
                }
                if let Some(e) = &mut l.init {
                    walk_expr_types_mut(e, f);
                }
            }
            Stmt::Defer(e) | Stmt::Expr(e) => walk_expr_types_mut(e, f),
        }
    }
    if let Some(t) = &mut b.tail {
        walk_expr_types_mut(t, f);
    }
}

/// Walk type annotations embedded inside an expression (casts, closure params)
/// and recurse through expression-nested blocks.
fn walk_expr_types_mut(e: &mut Expr, f: &mut impl FnMut(&mut Type)) {
    match &mut e.kind {
        ExprKind::Cast(x, ty) => {
            f(ty);
            walk_expr_types_mut(x, f);
        }
        ExprKind::Closure { params, body } => {
            for p in params {
                if let crate::Param::Normal { ty, .. } = p {
                    f(ty);
                }
            }
            walk_expr_types_mut(body, f);
        }
        ExprKind::Unary(_, x)
        | ExprKind::Paren(x)
        | ExprKind::Try(x)
        | ExprKind::Despawn(x)
        | ExprKind::Region { value: x, .. }
        | ExprKind::Field { base: x, .. } => walk_expr_types_mut(x, f),
        ExprKind::Binary(_, a, b) | ExprKind::Assign(_, a, b) | ExprKind::Index { base: a, index: b } => {
            walk_expr_types_mut(a, f);
            walk_expr_types_mut(b, f);
        }
        ExprKind::Pipe { value, func } => {
            walk_expr_types_mut(value, f);
            walk_expr_types_mut(func, f);
        }
        ExprKind::Call { callee, args, .. } => {
            walk_expr_types_mut(callee, f);
            for a in args {
                walk_expr_types_mut(&mut a.value, f);
            }
        }
        ExprKind::Array(xs) | ExprKind::Tuple(xs) => {
            for x in xs {
                walk_expr_types_mut(x, f);
            }
        }
        ExprKind::Spawn(args) => {
            for a in args {
                walk_expr_types_mut(&mut a.value, f);
            }
        }
        ExprKind::ArrayRepeat { value, count } => {
            walk_expr_types_mut(value, f);
            walk_expr_types_mut(count, f);
        }
        ExprKind::Struct { fields, base, .. } => {
            for fld in fields {
                if let Some(v) = &mut fld.value {
                    walk_expr_types_mut(v, f);
                }
            }
            if let Some(b) = base {
                walk_expr_types_mut(b, f);
            }
        }
        ExprKind::If(ifx) => {
            walk_expr_types_mut(&mut ifx.cond, f);
            walk_block_types_mut(&mut ifx.then_branch, f);
            if let Some(el) = &mut ifx.else_branch {
                walk_expr_types_mut(el, f);
            }
        }
        ExprKind::Match { scrutinee, arms } => {
            walk_expr_types_mut(scrutinee, f);
            for arm in arms {
                if let Some(g) = &mut arm.guard {
                    walk_expr_types_mut(g, f);
                }
                walk_expr_types_mut(&mut arm.body, f);
            }
        }
        ExprKind::For { iter, body, .. } => {
            walk_expr_types_mut(iter, f);
            walk_block_types_mut(body, f);
        }
        ExprKind::While { cond, body } => {
            walk_expr_types_mut(cond, f);
            walk_block_types_mut(body, f);
        }
        ExprKind::Loop(b) | ExprKind::Block(b) | ExprKind::Unsafe(b) => walk_block_types_mut(b, f),
        ExprKind::Return(Some(x)) | ExprKind::Break(Some(x)) => walk_expr_types_mut(x, f),
        _ => {}
    }
}

/// Walk an item's type references, recording `Enum<concrete...>` instantiations.
fn collect_enum_insts_item(
    it: &Item,
    genums: &HashMap<String, crate::EnumDecl>,
    insts: &mut HashMap<String, HashSet<Vec<String>>>,
) {
    let mut on_type = |t: &Type| collect_enum_insts_type(t, genums, insts);
    match &it.kind {
        ItemKind::Fn(f) => fn_types(f, &mut on_type),
        ItemKind::Struct(s) | ItemKind::Component(s) => {
            if let crate::StructBody::Named(fields) = &s.body {
                for f in fields {
                    on_type(&f.ty);
                }
            }
        }
        ItemKind::Impl(im) => {
            for it in &im.items {
                if let AssocItem::Fn(f) = it {
                    fn_types(f, &mut on_type);
                }
            }
        }
        _ => {}
    }
}

fn fn_types(f: &FnDecl, on_type: &mut impl FnMut(&Type)) {
    for p in &f.params {
        if let crate::Param::Normal { ty, .. } = p {
            on_type(ty);
        }
    }
    if let Some(t) = &f.ret {
        on_type(t);
    }
    if let Some(b) = &f.body {
        block_types(b, on_type);
    }
}

fn block_types(b: &Block, on_type: &mut impl FnMut(&Type)) {
    for stmt in &b.stmts {
        if let Stmt::Let(l) = stmt {
            if let Some(t) = &l.ty {
                on_type(t);
            }
        }
    }
}

fn collect_enum_insts_type(
    t: &Type,
    genums: &HashMap<String, crate::EnumDecl>,
    insts: &mut HashMap<String, HashSet<Vec<String>>>,
) {
    if let TypeKind::Path(p) = &t.kind {
        if let Some(seg) = p.segments.last() {
            if genums.contains_key(&seg.ident.name) && !seg.args.is_empty() {
                if let Some(targs) = seg.args.iter().map(type_head).collect::<Option<Vec<_>>>() {
                    insts.entry(seg.ident.name.clone()).or_default().insert(targs);
                }
            }
            for a in &seg.args {
                collect_enum_insts_type(a, genums, insts);
            }
        }
    }
}

/// Rewrite enum references in an item: type refs `Enum<..>` and value/pattern
/// paths `Enum::Variant` whose head is a single-instantiation generic enum.
fn remap_item_enums(it: &mut Item, remap: &HashMap<String, (String, Vec<String>)>) {
    match &mut it.kind {
        ItemKind::Fn(f) => remap_fn_enums(f, remap),
        ItemKind::Const(c) => remap_expr_enums(&mut c.value, remap),
        ItemKind::Impl(im) => {
            for it in &mut im.items {
                if let AssocItem::Fn(f) = it {
                    remap_fn_enums(f, remap);
                }
            }
        }
        _ => {}
    }
}

fn remap_fn_enums(f: &mut FnDecl, remap: &HashMap<String, (String, Vec<String>)>) {
    for p in &mut f.params {
        if let crate::Param::Normal { ty, .. } = p {
            remap_type_enums(ty, remap);
        }
    }
    if let Some(t) = &mut f.ret {
        remap_type_enums(t, remap);
    }
    if let Some(b) = &mut f.body {
        remap_block_enums(b, remap);
    }
}

fn remap_type_enums(t: &mut Type, remap: &HashMap<String, (String, Vec<String>)>) {
    if let TypeKind::Path(p) = &mut t.kind {
        if let Some(seg) = p.segments.last_mut() {
            if let Some((mangled, _)) = remap.get(&seg.ident.name) {
                seg.ident.name = mangled.clone();
                seg.args.clear();
            }
            for a in &mut seg.args {
                remap_type_enums(a, remap);
            }
        }
    }
}

fn remap_path_head(p: &mut crate::Path, remap: &HashMap<String, (String, Vec<String>)>) {
    if let Some(seg) = p.segments.first_mut() {
        if let Some((mangled, _)) = remap.get(&seg.ident.name) {
            seg.ident.name = mangled.clone();
        }
    }
}

fn remap_block_enums(b: &mut Block, remap: &HashMap<String, (String, Vec<String>)>) {
    for stmt in &mut b.stmts {
        match stmt {
            Stmt::Let(l) => {
                if let Some(t) = &mut l.ty {
                    remap_type_enums(t, remap);
                }
                if let Some(e) = &mut l.init {
                    remap_expr_enums(e, remap);
                }
            }
            Stmt::Defer(e) | Stmt::Expr(e) => remap_expr_enums(e, remap),
        }
    }
    if let Some(t) = &mut b.tail {
        remap_expr_enums(t, remap);
    }
}

fn remap_pat_enums(pat: &mut Pat, remap: &HashMap<String, (String, Vec<String>)>) {
    match &mut pat.kind {
        PatKind::Path(p) => remap_path_head(p, remap),
        PatKind::TupleStruct { path, elems } => {
            remap_path_head(path, remap);
            for e in elems {
                remap_pat_enums(e, remap);
            }
        }
        PatKind::Struct { path, fields, .. } => {
            remap_path_head(path, remap);
            for f in fields {
                if let Some(sub) = &mut f.pat {
                    remap_pat_enums(sub, remap);
                }
            }
        }
        PatKind::Tuple(ps) => {
            for p in ps {
                remap_pat_enums(p, remap);
            }
        }
        PatKind::Binding { sub: Some(s), .. } => remap_pat_enums(s, remap),
        _ => {}
    }
}

fn remap_expr_enums(e: &mut Expr, remap: &HashMap<String, (String, Vec<String>)>) {
    match &mut e.kind {
        ExprKind::Path(p) => remap_path_head(p, remap),
        ExprKind::Struct { path, fields, base } => {
            remap_path_head(path, remap);
            for f in fields {
                if let Some(v) = &mut f.value {
                    remap_expr_enums(v, remap);
                }
            }
            if let Some(b) = base {
                remap_expr_enums(b, remap);
            }
        }
        ExprKind::Call { callee, args, .. } => {
            remap_expr_enums(callee, remap);
            for a in args {
                remap_expr_enums(&mut a.value, remap);
            }
        }
        ExprKind::Unary(_, x)
        | ExprKind::Paren(x)
        | ExprKind::Despawn(x)
        | ExprKind::Cast(x, _)
        | ExprKind::Try(x)
        | ExprKind::Region { value: x, .. } => remap_expr_enums(x, remap),
        ExprKind::Binary(_, a, c)
        | ExprKind::Assign(_, a, c)
        | ExprKind::Index { base: a, index: c } => {
            remap_expr_enums(a, remap);
            remap_expr_enums(c, remap);
        }
        ExprKind::Pipe { value, func } => {
            remap_expr_enums(value, remap);
            remap_expr_enums(func, remap);
        }
        ExprKind::Field { base, .. } => remap_expr_enums(base, remap),
        ExprKind::Array(xs) | ExprKind::Tuple(xs) => {
            for x in xs {
                remap_expr_enums(x, remap);
            }
        }
        ExprKind::ArrayRepeat { value, count } => {
            remap_expr_enums(value, remap);
            remap_expr_enums(count, remap);
        }
        ExprKind::If(ifx) => {
            remap_expr_enums(&mut ifx.cond, remap);
            remap_block_enums(&mut ifx.then_branch, remap);
            if let Some(el) = &mut ifx.else_branch {
                remap_expr_enums(el, remap);
            }
        }
        ExprKind::Match { scrutinee, arms } => {
            remap_expr_enums(scrutinee, remap);
            for arm in arms {
                remap_pat_enums(&mut arm.pat, remap);
                if let Some(g) = &mut arm.guard {
                    remap_expr_enums(g, remap);
                }
                remap_expr_enums(&mut arm.body, remap);
            }
        }
        ExprKind::For { pat, iter, body } => {
            remap_pat_enums(pat, remap);
            remap_expr_enums(iter, remap);
            remap_block_enums(body, remap);
        }
        ExprKind::While { cond, body } => {
            remap_expr_enums(cond, remap);
            remap_block_enums(body, remap);
        }
        ExprKind::Loop(b) | ExprKind::Block(b) | ExprKind::Unsafe(b) => remap_block_enums(b, remap),
        ExprKind::Closure { body, .. } => remap_expr_enums(body, remap),
        ExprKind::Return(o) | ExprKind::Break(o) => {
            if let Some(x) = o {
                remap_expr_enums(x, remap);
            }
        }
        _ => {}
    }
}

fn mangle(name: &str, args: &[String]) -> String {
    format!("{name}${}", args.join("$"))
}

/// The head type name of a (possibly generic) path type, e.g. `Box<i64>` → "Box".
fn type_head(t: &Type) -> Option<String> {
    match &t.kind {
        TypeKind::Path(p) => p.segments.last().map(|s| s.ident.name.clone()),
        // A region annotation is transparent to monomorphization.
        TypeKind::Region(_, inner) => type_head(inner),
        _ => None,
    }
}

// --- type inference from expressions ----------------------------------------

fn guess_type(e: &Expr, gen: &Gen, vars: &HashMap<String, String>) -> Option<String> {
    match &e.kind {
        ExprKind::Int(..) | ExprKind::Bool(_) => Some("i64".to_string()),
        ExprKind::Float(..) => Some("f64".to_string()),
        ExprKind::Str(_) => Some("str".to_string()),
        ExprKind::Paren(inner) => guess_type(inner, gen, vars),
        ExprKind::Cast(_, ty) => type_head(ty),
        // A variable whose type we've tracked (parameter / `let`).
        ExprKind::Path(p) if p.segments.len() == 1 => {
            vars.get(&p.segments[0].ident.name).cloned()
        }
        ExprKind::Struct { path, fields, .. } => {
            let name = path.segments.last()?.ident.name.clone();
            // A *generic* struct construction's type is its specialized name,
            // recursively (so `Box<Box<i64>>` → `Box$Box$i64` and the inner
            // `Box$i64` instantiation is what the outer field is typed as).
            if let Some(sd) = gen.structs.get(&name) {
                let targs = infer_struct_args(sd, fields, gen)?;
                Some(mangle(&name, &targs))
            } else {
                Some(name)
            }
        }
        _ => None,
    }
}

// --- specialization ---------------------------------------------------------

fn specialize_fn(
    inst: &(String, Vec<String>),
    gen: &Gen,
    wanted: &mut Wanted,
) -> Option<Item> {
    let (fname, args) = inst;
    let decl = gen.fns.get(fname)?;
    let subst: HashMap<String, String> =
        decl.generics.iter().map(|g| g.name.name.clone()).zip(args.iter().cloned()).collect();
    let mut spec = decl.clone();
    spec.name.name = mangle(fname, args);
    spec.generics.clear();
    subst_fn(&mut spec, &subst);
    let mut item =
        Item { attrs: Vec::new(), vis: Vis::Private, kind: ItemKind::Fn(spec), span: decl.name.span };
    rewrite_item(&mut item, gen, wanted);
    Some(item)
}

fn specialize_struct(
    inst: &(String, Vec<String>),
    gen: &Gen,
    wanted: &mut Wanted,
    out: &mut Vec<Item>,
) {
    let (sname, args) = inst;
    let Some(decl) = gen.structs.get(sname) else { return };
    let mangled = mangle(sname, args);
    let subst: HashMap<String, String> =
        decl.generics.iter().map(|g| g.name.name.clone()).zip(args.iter().cloned()).collect();

    // Specialized struct.
    let mut sd = decl.clone();
    sd.name.name = mangled.clone();
    sd.generics.clear();
    if let StructBody::Named(fields) = &mut sd.body {
        for f in fields {
            subst_type(&mut f.ty, &subst);
        }
    }
    let span = decl.name.span;
    out.push(Item { attrs: Vec::new(), vis: Vis::Private, kind: ItemKind::Struct(sd), span });

    // Specialized impls. The impl's own generic params are bound by matching the
    // generic names in its `self_ty` args against this instantiation's args.
    if let Some(impls) = gen.impls.get(sname) {
        for im in impls {
            let isubst = impl_subst(im, args);
            let mut spec = im.clone();
            spec.generics.clear();
            spec.self_ty = Type { kind: TypeKind::Path(crate::Path {
                segments: vec![crate::PathSeg {
                    ident: crate::Ident { name: mangled.clone(), span },
                    args: Vec::new(),
                }],
                span,
            }), span };
            for it in &mut spec.items {
                if let AssocItem::Fn(f) = it {
                    subst_fn(f, &isubst);
                }
            }
            let mut item =
                Item { attrs: Vec::new(), vis: Vis::Private, kind: ItemKind::Impl(spec), span };
            rewrite_item(&mut item, gen, wanted);
            out.push(item);
        }
    }
}

/// Map an impl's generic params to concrete types by aligning the generic names
/// appearing in its `self_ty` args with the instantiation's args.
fn impl_subst(im: &ImplDecl, args: &[String]) -> HashMap<String, String> {
    let mut subst = HashMap::new();
    if let TypeKind::Path(p) = &im.self_ty.kind {
        if let Some(seg) = p.segments.last() {
            for (i, a) in seg.args.iter().enumerate() {
                if let (TypeKind::Path(ap), Some(concrete)) = (&a.kind, args.get(i)) {
                    if ap.segments.len() == 1 {
                        subst.insert(ap.segments[0].ident.name.clone(), concrete.clone());
                    }
                }
            }
        }
    }
    subst
}

// --- substituting generic params with concrete types ------------------------

fn subst_fn(f: &mut FnDecl, subst: &HashMap<String, String>) {
    for prm in &mut f.params {
        if let crate::Param::Normal { ty, .. } = prm {
            subst_type(ty, subst);
        }
    }
    if let Some(t) = &mut f.ret {
        subst_type(t, subst);
    }
    if let Some(b) = &mut f.body {
        subst_block(b, subst);
    }
}

fn subst_type(t: &mut Type, subst: &HashMap<String, String>) {
    match &mut t.kind {
        TypeKind::Path(p) => {
            if p.segments.len() == 1 {
                if let Some(c) = subst.get(&p.segments[0].ident.name) {
                    p.segments[0].ident.name = c.clone();
                }
            }
            for seg in &mut p.segments {
                for a in &mut seg.args {
                    subst_type(a, subst);
                }
            }
        }
        // Recurse into compound types so `[T; N]`, `(T, U)`, `&T`, etc. specialize.
        TypeKind::Owned(inner) | TypeKind::Ref { inner, .. } => subst_type(inner, subst),
        TypeKind::Array { elem, .. } => subst_type(elem, subst),
        TypeKind::Tuple(ts) => {
            for t in ts {
                subst_type(t, subst);
            }
        }
        TypeKind::Fn { params, ret } => {
            for p in params {
                subst_type(p, subst);
            }
            subst_type(ret, subst);
        }
        TypeKind::Region(_, inner) => subst_type(inner, subst),
        _ => {}
    }
}

fn subst_block(b: &mut Block, subst: &HashMap<String, String>) {
    for stmt in &mut b.stmts {
        match stmt {
            Stmt::Let(l) => {
                if let Some(t) = &mut l.ty {
                    subst_type(t, subst);
                }
                if let Some(e) = &mut l.init {
                    subst_expr(e, subst);
                }
            }
            Stmt::Defer(e) | Stmt::Expr(e) => subst_expr(e, subst),
        }
    }
    if let Some(t) = &mut b.tail {
        subst_expr(t, subst);
    }
}

fn subst_expr(e: &mut Expr, subst: &HashMap<String, String>) {
    // Substitute a generic struct construction's path head, then recurse.
    if let ExprKind::Struct { path, .. } = &mut e.kind {
        if path.segments.len() == 1 {
            if let Some(c) = subst.get(&path.segments[0].ident.name) {
                path.segments[0].ident.name = c.clone();
            }
        }
    }
    match &mut e.kind {
        ExprKind::Cast(x, ty) => {
            subst_expr(x, subst);
            subst_type(ty, subst);
        }
        ExprKind::Unary(_, x)
        | ExprKind::Paren(x)
        | ExprKind::Despawn(x)
        | ExprKind::Try(x)
        | ExprKind::Region { value: x, .. } => subst_expr(x, subst),
        ExprKind::Binary(_, a, c)
        | ExprKind::Assign(_, a, c)
        | ExprKind::Index { base: a, index: c } => {
            subst_expr(a, subst);
            subst_expr(c, subst);
        }
        ExprKind::Pipe { value, func } => {
            subst_expr(value, subst);
            subst_expr(func, subst);
        }
        ExprKind::Call { callee, type_args, args } => {
            subst_expr(callee, subst);
            for t in type_args {
                subst_type(t, subst);
            }
            for a in args {
                subst_expr(&mut a.value, subst);
            }
        }
        ExprKind::Field { base, .. } => subst_expr(base, subst),
        ExprKind::Struct { fields, base, .. } => {
            for fd in fields {
                if let Some(v) = &mut fd.value {
                    subst_expr(v, subst);
                }
            }
            if let Some(b) = base {
                subst_expr(b, subst);
            }
        }
        ExprKind::Array(xs) | ExprKind::Tuple(xs) => {
            for x in xs {
                subst_expr(x, subst);
            }
        }
        ExprKind::ArrayRepeat { value, count } => {
            subst_expr(value, subst);
            subst_expr(count, subst);
        }
        ExprKind::If(ifx) => {
            subst_expr(&mut ifx.cond, subst);
            subst_block(&mut ifx.then_branch, subst);
            if let Some(el) = &mut ifx.else_branch {
                subst_expr(el, subst);
            }
        }
        ExprKind::Match { scrutinee, arms } => {
            subst_expr(scrutinee, subst);
            for arm in arms {
                subst_arm(arm, subst);
            }
        }
        ExprKind::For { iter, body, .. } => {
            subst_expr(iter, subst);
            subst_block(body, subst);
        }
        ExprKind::While { cond, body } => {
            subst_expr(cond, subst);
            subst_block(body, subst);
        }
        ExprKind::Loop(b) | ExprKind::Block(b) | ExprKind::Unsafe(b) => subst_block(b, subst),
        ExprKind::Closure { body, .. } => subst_expr(body, subst),
        ExprKind::Return(o) | ExprKind::Break(o) => {
            if let Some(x) = o {
                subst_expr(x, subst);
            }
        }
        _ => {}
    }
}

fn subst_arm(arm: &mut MatchArm, subst: &HashMap<String, String>) {
    if let Some(g) = &mut arm.guard {
        subst_expr(g, subst);
    }
    subst_expr(&mut arm.body, subst);
}

// --- rewriting generic uses to mangled names --------------------------------

fn rewrite_item(item: &mut Item, gen: &Gen, w: &mut Wanted) {
    match &mut item.kind {
        ItemKind::Fn(f) => {
            for p in &mut f.params {
                if let crate::Param::Normal { ty, .. } = p {
                    rewrite_type(ty, gen, w);
                }
            }
            if let Some(t) = &mut f.ret {
                rewrite_type(t, gen, w);
            }
            if let Some(b) = &mut f.body {
                rewrite_block(b, gen, w);
            }
        }
        ItemKind::Const(c) => rewrite_expr(&mut c.value, gen, w),
        ItemKind::Impl(im) => {
            for it in &mut im.items {
                if let AssocItem::Fn(f) = it {
                    if let Some(b) = &mut f.body {
                        rewrite_block(b, gen, w);
                    }
                }
            }
        }
        _ => {}
    }
}

/// Rewrite an explicit generic-struct type reference `Box<i64>` → `Box$i64`.
fn rewrite_type(t: &mut Type, gen: &Gen, w: &mut Wanted) {
    if let TypeKind::Path(p) = &mut t.kind {
        if let Some(seg) = p.segments.last_mut() {
            if gen.structs.contains_key(&seg.ident.name) && !seg.args.is_empty() {
                let targs: Option<Vec<String>> = seg.args.iter().map(type_head).collect();
                if let Some(targs) = targs {
                    let name = seg.ident.name.clone();
                    seg.ident.name = mangle(&name, &targs);
                    seg.args.clear();
                    w.structs.insert((name, targs));
                }
            }
            for a in &mut seg.args {
                rewrite_type(a, gen, w);
            }
        }
    }
}

fn rewrite_block(b: &mut Block, gen: &Gen, w: &mut Wanted) {
    for stmt in &mut b.stmts {
        match stmt {
            Stmt::Let(l) => {
                if let Some(t) = &mut l.ty {
                    rewrite_type(t, gen, w);
                }
                if let Some(e) = &mut l.init {
                    rewrite_expr(e, gen, w);
                }
            }
            Stmt::Defer(e) | Stmt::Expr(e) => rewrite_expr(e, gen, w),
        }
    }
    if let Some(t) = &mut b.tail {
        rewrite_expr(t, gen, w);
    }
}

fn rewrite_expr(e: &mut Expr, gen: &Gen, w: &mut Wanted) {
    // Rewrite a generic function call.
    if let ExprKind::Call { callee, args, .. } = &mut e.kind {
        if let Some((name, targs)) = resolve_fn_call(callee, args, gen) {
            if let ExprKind::Path(p) = &mut callee.kind {
                p.segments[0].ident.name = mangle(&name, &targs);
            }
            w.fns.insert((name, targs));
        }
    }
    // Rewrite a generic struct construction.
    if let ExprKind::Struct { path, fields, .. } = &mut e.kind {
        if path.segments.len() == 1 {
            let name = path.segments[0].ident.name.clone();
            if let Some(sd) = gen.structs.get(&name) {
                if let Some(targs) = infer_struct_args(sd, fields, gen) {
                    path.segments[0].ident.name = mangle(&name, &targs);
                    w.structs.insert((name, targs));
                }
            }
        }
    }
    // Recurse into children.
    match &mut e.kind {
        ExprKind::Unary(_, x)
        | ExprKind::Paren(x)
        | ExprKind::Despawn(x)
        | ExprKind::Region { value: x, .. }
        | ExprKind::Try(x)
        | ExprKind::Cast(x, _) => rewrite_expr(x, gen, w),
        ExprKind::Binary(_, a, c)
        | ExprKind::Assign(_, a, c)
        | ExprKind::Index { base: a, index: c } => {
            rewrite_expr(a, gen, w);
            rewrite_expr(c, gen, w);
        }
        ExprKind::Pipe { value, func } => {
            rewrite_expr(value, gen, w);
            rewrite_expr(func, gen, w);
        }
        ExprKind::Call { callee, args, .. } => {
            rewrite_expr(callee, gen, w);
            for a in args {
                rewrite_expr(&mut a.value, gen, w);
            }
        }
        ExprKind::Field { base, .. } => rewrite_expr(base, gen, w),
        ExprKind::Struct { fields, base, .. } => {
            for fd in fields {
                if let Some(v) = &mut fd.value {
                    rewrite_expr(v, gen, w);
                }
            }
            if let Some(b) = base {
                rewrite_expr(b, gen, w);
            }
        }
        ExprKind::Array(xs) | ExprKind::Tuple(xs) => {
            for x in xs {
                rewrite_expr(x, gen, w);
            }
        }
        ExprKind::ArrayRepeat { value, count } => {
            rewrite_expr(value, gen, w);
            rewrite_expr(count, gen, w);
        }
        ExprKind::If(ifx) => {
            rewrite_expr(&mut ifx.cond, gen, w);
            rewrite_block(&mut ifx.then_branch, gen, w);
            if let Some(el) = &mut ifx.else_branch {
                rewrite_expr(el, gen, w);
            }
        }
        ExprKind::Match { scrutinee, arms } => {
            rewrite_expr(scrutinee, gen, w);
            for arm in arms {
                if let Some(g) = &mut arm.guard {
                    rewrite_expr(g, gen, w);
                }
                rewrite_expr(&mut arm.body, gen, w);
            }
        }
        ExprKind::For { iter, body, .. } => {
            rewrite_expr(iter, gen, w);
            rewrite_block(body, gen, w);
        }
        ExprKind::While { cond, body } => {
            rewrite_expr(cond, gen, w);
            rewrite_block(body, gen, w);
        }
        ExprKind::Loop(b) | ExprKind::Block(b) | ExprKind::Unsafe(b) => rewrite_block(b, gen, w),
        ExprKind::Closure { body, .. } => rewrite_expr(body, gen, w),
        ExprKind::Return(o) | ExprKind::Break(o) => {
            if let Some(x) = o {
                rewrite_expr(x, gen, w);
            }
        }
        _ => {}
    }
}

fn resolve_fn_call(
    callee: &Expr,
    args: &[crate::Arg],
    gen: &Gen,
) -> Option<(String, Vec<String>)> {
    let ExprKind::Path(p) = &callee.kind else { return None };
    if p.segments.len() != 1 {
        return None;
    }
    let name = &p.segments[0].ident.name;
    let decl = gen.fns.get(name)?;
    let gnames: HashSet<&str> = decl.generics.iter().map(|g| g.name.name.as_str()).collect();
    let mut map: HashMap<String, String> = HashMap::new();
    let mut ai = args.iter();
    for prm in &decl.params {
        if let crate::Param::Normal { ty, .. } = prm {
            let arg = ai.next()?;
            if let TypeKind::Path(tp) = &ty.kind {
                if tp.segments.len() == 1 && gnames.contains(tp.segments[0].ident.name.as_str()) {
                    map.insert(tp.segments[0].ident.name.clone(), guess_type(&arg.value, gen, &HashMap::new())?);
                }
            }
        }
    }
    let resolved: Option<Vec<String>> =
        decl.generics.iter().map(|g| map.get(&g.name.name).cloned()).collect();
    Some((name.clone(), resolved?))
}

/// Infer a generic struct's type args from a construction's field values,
/// matching the declared field type structure against the value (handles bare
/// `T` and `[T; N]` fields).
fn infer_struct_args(sd: &StructDecl, fields: &[crate::FieldInit], gen: &Gen) -> Option<Vec<String>> {
    let StructBody::Named(decl_fields) = &sd.body else { return None };
    let gnames: HashSet<&str> = sd.generics.iter().map(|g| g.name.name.as_str()).collect();
    let mut map: HashMap<String, String> = HashMap::new();
    for df in decl_fields {
        if let Some(fi) = fields.iter().find(|f| f.name.name == df.name.name) {
            if let Some(val) = &fi.value {
                bind_generics(&df.ty, val, &gnames, &mut map, gen, &HashMap::new());
            }
        }
    }
    sd.generics.iter().map(|g| map.get(&g.name.name).cloned()).collect()
}

/// Bind generic params by matching a declared type's shape against a value
/// expression (bare `T` ← value's type; `[T; N]` ← the element value's type).
fn bind_generics(
    ty: &Type,
    val: &Expr,
    gnames: &HashSet<&str>,
    map: &mut HashMap<String, String>,
    gen: &Gen,
    vars: &HashMap<String, String>,
) {
    match &ty.kind {
        TypeKind::Path(p) if p.segments.len() == 1 && gnames.contains(p.segments[0].ident.name.as_str()) => {
            if let Some(c) = guess_type(val, gen, vars) {
                map.insert(p.segments[0].ident.name.clone(), c);
            }
        }
        TypeKind::Array { elem, .. } => match &val.kind {
            ExprKind::ArrayRepeat { value, .. } => bind_generics(elem, value, gnames, map, gen, vars),
            ExprKind::Array(items) if !items.is_empty() => {
                bind_generics(elem, &items[0], gnames, map, gen, vars)
            }
            _ => {}
        },
        _ => {}
    }
}
