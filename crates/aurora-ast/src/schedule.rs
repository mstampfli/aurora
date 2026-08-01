//! Parallel system scheduling (grammar spec Â§6.2).
//!
//! Groups a module's `system`s into ordered *layers* of mutually-independent
//! systems that may execute concurrently. Everything else commutes, so fusing it
//! into one concurrent layer cannot change results. This is the runtime
//! realisation of the data-race-freedom theorem the checker enforces.
//!
//! Layering happens in two steps, and the order of the two is what makes
//! `after`/`before` mean what they say:
//!
//! 1. **Rank by ordering.** `after`/`before` form a DAG over the stage's systems,
//!    and each system is ranked one past the longest chain that must precede it.
//!    Ordering is therefore transitive by construction, and independent of
//!    declaration order - `a after(b)` puts b first even when a is declared
//!    first. Ranking only after the edges are known is the point: an earlier
//!    version split layers in declaration order and used the annotation merely as
//!    a hint to split, so `a after(b)` with a declared first ran a first, exactly
//!    backwards, and silently.
//! 2. **Split conflicts within a rank.** Same-rank systems are unordered relative
//!    to each other, so they may share a layer only where their component access
//!    does not conflict; conflicting ones are split into consecutive layers in
//!    declaration order. No pair inside a layer can race, which is what makes a
//!    layer safe to hand to the thread pool.

use std::collections::BTreeSet;

use crate::{
    Block, Expr, ExprKind, ItemKind, Module, Path, QTerm, QueryExpr, Stmt, SysSched, SystemDecl,
};

#[derive(Default)]
struct Access {
    reads: BTreeSet<String>,
    writes: BTreeSet<String>,
}

struct SysInfo {
    name: String,
    access: Access,
    /// Systems named by `after(..)`: each must run strictly before this one.
    after: BTreeSet<String>,
    /// Systems named by `before(..)`: each must run strictly after this one.
    before: BTreeSet<String>,
}

/// The name a call expression names, joined so a module-qualified path matches
/// the mangled item name flattening produced.
fn callee_name(e: &Expr) -> Option<String> {
    match &e.kind {
        ExprKind::Path(p) => Some(
            p.segments
                .iter()
                .map(|s| s.ident.name.as_str())
                .collect::<Vec<_>>()
                .join("::"),
        ),
        _ => None,
    }
}

/// A path written out in full, matching the mangled name flattening gives an item.
///
/// Ordering names must join rather than take the last segment: `after(sim::tick)`
/// names a system called `sim::tick` once modules are flattened, and matching on
/// `tick` alone would either miss it or, worse, match a same-named system in some
/// other module.
fn joined(p: &Path) -> String {
    p.segments
        .iter()
        .map(|s| s.ident.name.as_str())
        .collect::<Vec<_>>()
        .join("::")
}

/// Component read/write sets derived from every `query<...>` the system reaches,
/// its own and those of the functions it calls.
fn access_of(module: &Module, sys: &SystemDecl) -> Access {
    let queries = reachable_queries(module, &sys.body);
    let mut a = Access::default();
    for q in queries {
        for term in &q.terms {
            match term {
                // Joined for the same reason ordering names are: `sim::Foe` and
                // a different module's `Foe` are different components, and the
                // layering must not fuse or split systems on a name collision.
                QTerm::Read(p) | QTerm::OptRead(p) => {
                    a.reads.insert(joined(p));
                }
                QTerm::Write(p) | QTerm::OptWrite(p) => {
                    a.writes.insert(joined(p));
                }
                // Filters / entity id are not data access.
                QTerm::With(_) | QTerm::Without(_) | QTerm::Entity => {}
            }
        }
    }
    a
}

/// The two ordering sets, kept apart because direction is the whole point.
///
/// Folding `after` and `before` into one "ordered with" set loses which side of
/// the edge a system is on, and an ordering that does not know its own direction
/// can only ever be used to keep two systems apart - never to put them in the
/// right sequence.
fn ordering_of(sys: &SystemDecl) -> (BTreeSet<String>, BTreeSet<String>) {
    let mut after = BTreeSet::new();
    let mut before = BTreeSet::new();
    for s in &sys.schedule {
        match s {
            SysSched::After(ps) => after.extend(ps.iter().map(joined)),
            SysSched::Before(ps) => before.extend(ps.iter().map(joined)),
            _ => {}
        }
    }
    (after, before)
}

/// Rank every system by the longest chain of ordering constraints reaching it.
///
/// `after`/`before` form a DAG; a system's rank is one past the highest-ranked
/// system that must precede it, which makes the ordering transitive by
/// construction: `a after(b)` and `b after(c)` puts c, b and a in three ascending
/// ranks whatever order they were declared in.
///
/// The ready set is drained lowest-index-first so the result depends only on the
/// program, not on hash iteration order - a schedule that varied between runs
/// would defeat the point of having one.
///
/// Systems caught in an ordering cycle are unreachable here and keep rank 0. The
/// checker rejects those programs; this only has to stay deterministic and
/// race-free in the meantime, and the conflict split below guarantees both.
fn ranks(edges: &[BTreeSet<usize>], n: usize) -> Vec<usize> {
    let mut indegree = vec![0usize; n];
    for succs in edges.iter().take(n) {
        for &v in succs {
            indegree[v] += 1;
        }
    }
    let mut ready: BTreeSet<usize> = (0..n).filter(|&i| indegree[i] == 0).collect();
    let mut rank = vec![0usize; n];
    while let Some(&u) = ready.iter().next() {
        ready.remove(&u);
        for &v in &edges[u] {
            rank[v] = rank[v].max(rank[u] + 1);
            indegree[v] -= 1;
            if indegree[v] == 0 {
                ready.insert(v);
            }
        }
    }
    rank
}

/// Two systems conflict when one writes a component the other reads or writes.
fn conflict(a: &Access, b: &Access) -> bool {
    a.writes
        .iter()
        .any(|c| b.reads.contains(c) || b.writes.contains(c))
        || b.writes
            .iter()
            .any(|c| a.reads.contains(c) || a.writes.contains(c))
}

/// Group the module's systems (declaration order) into ordered parallel layers.
/// Returns, for each layer, the indices into the declaration-ordered system
/// list â€” index `k` is the k-th `system` item in `module`. A layer with one
/// index runs sequentially; a layer with several runs them concurrently.
pub fn parallel_layers(module: &Module) -> Vec<Vec<usize>> {
    layers_matching(module, |_| true)
}

/// The stage a system runs in when it names none.
pub const DEFAULT_STAGE: &str = "Update";

/// The stage driven by a fixed-timestep accumulator rather than by the frame.
///
/// Simulation that must be reproducible belongs here. A rule stated in frames -
/// an invulnerability window from frame 6 to frame 27 - only means anything if
/// the frames are a fixed length; under a variable frame time the same input
/// produces different outcomes on different machines, and on the same machine
/// under load.
pub const FIXED_STAGE: &str = "FixedUpdate";

/// The stage a system declares, or [`DEFAULT_STAGE`].
pub fn stage_of(sys: &SystemDecl) -> String {
    sys.schedule
        .iter()
        .find_map(|s| match s {
            SysSched::Stage(id) => Some(id.name.clone()),
            _ => None,
        })
        .unwrap_or_else(|| DEFAULT_STAGE.to_string())
}

/// Layers containing only the systems in `stage`.
///
/// Indices still address the module's full declaration-ordered system list, so a
/// caller holding that list can use them directly. Systems outside the stage are
/// skipped rather than renumbered: two stages that each renumbered would both be
/// right about a different list, and the caller has only one.
pub fn parallel_layers_in(module: &Module, stage: &str) -> Vec<Vec<usize>> {
    layers_matching(module, |s| stage_of(s) == stage)
}

fn layers_matching(module: &Module, keep: impl Fn(&SystemDecl) -> bool) -> Vec<Vec<usize>> {
    let decls: Vec<&SystemDecl> = module
        .items
        .iter()
        .filter_map(|it| match &it.kind {
            ItemKind::System(s) => Some(s),
            _ => None,
        })
        .collect();
    let infos: Vec<SysInfo> = decls
        .iter()
        .map(|s| {
            let (after, before) = ordering_of(s);
            SysInfo {
                name: s.name.name.clone(),
                access: access_of(module, s),
                after,
                before,
            }
        })
        .collect();
    let wanted: Vec<usize> = (0..decls.len()).filter(|&i| keep(decls[i])).collect();

    // Ordering edges over positions within `wanted`. A system may only be ordered
    // against one in the same stage: stages already run in sequence, so an edge
    // across them is either redundant or unsatisfiable, and silently honouring it
    // here would let a stage boundary be reordered.
    let n = wanted.len();
    let mut at: std::collections::HashMap<&str, usize> = std::collections::HashMap::new();
    for (pos, &i) in wanted.iter().enumerate() {
        at.insert(infos[i].name.as_str(), pos);
    }
    let mut edges: Vec<BTreeSet<usize>> = vec![BTreeSet::new(); n];
    for (pos, &i) in wanted.iter().enumerate() {
        for name in &infos[i].after {
            if let Some(&other) = at.get(name.as_str()) {
                edges[other].insert(pos);
            }
        }
        for name in &infos[i].before {
            if let Some(&other) = at.get(name.as_str()) {
                edges[pos].insert(other);
            }
        }
    }
    let rank = ranks(&edges, n);

    // Systems of equal rank are unordered with respect to each other, so they may
    // share a layer - but only where they do not conflict. Conflicting systems
    // are split into consecutive layers in declaration order, which is what makes
    // a layer safe to run concurrently: no pair inside one can race.
    let mut by_rank: std::collections::BTreeMap<usize, Vec<usize>> =
        std::collections::BTreeMap::new();
    for pos in 0..n {
        by_rank.entry(rank[pos]).or_default().push(pos);
    }

    let mut layers: Vec<Vec<usize>> = Vec::new();
    for group in by_rank.values() {
        let mut cur: Vec<usize> = Vec::new();
        for &pos in group {
            let joins = cur
                .iter()
                .all(|&other| !conflict(&infos[wanted[pos]].access, &infos[wanted[other]].access));
            if !joins && !cur.is_empty() {
                layers.push(cur.drain(..).map(|p| wanted[p]).collect());
            }
            cur.push(pos);
        }
        if !cur.is_empty() {
            layers.push(cur.into_iter().map(|p| wanted[p]).collect());
        }
    }
    layers
}

// --- query collection (read-only walk over a system body) ------------------

/// What a read-only walk finds: the queries written here, and the functions
/// called from here.
///
/// The calls matter as much as the queries. A system that calls a helper which
/// queries `&mut Player` accesses `Player` just as surely as if it had written
/// the query inline, and a race-freedom proof that only reads the system body
/// cannot see it. Collecting both in one pass keeps the two in step: a walker
/// that learned about a new expression form for queries but not for calls would
/// go quietly blind on exactly the programs that nest deepest.
#[derive(Default)]
pub struct Collected<'a> {
    pub queries: Vec<&'a QueryExpr>,
    pub calls: Vec<String>,
}

/// Every query a system can reach, including through the functions it calls.
///
/// Follows calls transitively, so component access is attributed to the system
/// that ultimately performs it. Recursion terminates on the visited set; an
/// unresolved callee (a builtin, or an `@extern`) contributes nothing, which is
/// correct - neither runs a query.
/// Every function name a body can reach, including through the helpers it calls.
///
/// The same walk `reachable_queries` does, for the same reason: a system that
/// calls a helper that draws has drawn, exactly as surely as an inline call
/// would, and attributing only the body would let anything hide one call deep.
/// Names that are not functions in this module are builtins - which is what the
/// caller is usually looking for.
pub fn reachable_calls(module: &Module, body: &Block) -> Vec<String> {
    let mut bodies: std::collections::HashMap<&str, &Block> = std::collections::HashMap::new();
    for item in &module.items {
        if let ItemKind::Fn(f) = &item.kind {
            if let Some(b) = &f.body {
                bodies.insert(f.name.name.as_str(), b);
            }
        }
    }

    let mut out = Vec::new();
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut queue = vec![body];
    while let Some(b) = queue.pop() {
        let mut found = Collected::default();
        walk_block(b, &mut found);
        for name in found.calls {
            if !seen.insert(name.clone()) {
                continue;
            }
            if let Some(next) = bodies.get(name.as_str()) {
                queue.push(next);
            }
            out.push(name);
        }
    }
    out
}

pub fn reachable_queries<'a>(module: &'a Module, body: &'a Block) -> Vec<&'a QueryExpr> {
    let mut bodies: std::collections::HashMap<&str, &Block> = std::collections::HashMap::new();
    for item in &module.items {
        // A bodiless `@extern fn` has nothing to walk, and cannot run a query.
        if let ItemKind::Fn(f) = &item.kind {
            if let Some(body) = &f.body {
                bodies.insert(f.name.name.as_str(), body);
            }
        }
    }

    let mut out = Vec::new();
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut queue = vec![body];
    while let Some(b) = queue.pop() {
        let mut found = Collected::default();
        walk_block(b, &mut found);
        out.append(&mut found.queries);
        for name in found.calls {
            if !seen.insert(name.clone()) {
                continue;
            }
            if let Some(next) = bodies.get(name.as_str()) {
                queue.push(next);
            }
        }
    }
    out
}

fn walk_block<'a>(block: &'a Block, out: &mut Collected<'a>) {
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

fn walk_expr<'a>(e: &'a Expr, out: &mut Collected<'a>) {
    match &e.kind {
        ExprKind::Query(q) => {
            out.queries.push(q);
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
            if let Some(name) = callee_name(callee) {
                out.calls.push(name);
            }
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
