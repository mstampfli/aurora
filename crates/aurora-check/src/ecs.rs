//! ECS query and scheduler analysis (grammar spec Â§5.4, Â§6.2).

use std::collections::BTreeSet;

use aurora_ast::{ItemKind, Module, QTerm, QueryExpr, SysSched, SystemDecl};
use aurora_diag::Diagnostic;
use aurora_span::Span;

/// Read/write component access derived from a system's queries.
#[derive(Default)]
struct Access {
    reads: BTreeSet<String>,
    writes: BTreeSet<String>,
}

struct SysInfo {
    name: String,
    span: Span,
    stage: Option<String>,
    /// Systems named by `after(..)`: each must run strictly before this one.
    after: BTreeSet<String>,
    /// Systems named by `before(..)`: each must run strictly after this one.
    before: BTreeSet<String>,
    access: Access,
}

pub(crate) fn check_queries_and_schedule(module: &Module, diags: &mut Vec<Diagnostic>) {
    let mut systems = Vec::new();

    for item in &module.items {
        let ItemKind::System(sys) = &item.kind else {
            continue;
        };

        // Every query the system can reach, including through the functions it
        // calls. A helper that queries `&mut Player` gives its caller that access
        // just as surely as an inline query would, and attributing it only to the
        // body would let two systems that both reach `Player` through helpers be
        // declared independent - and then run concurrently over it.
        let queries = aurora_ast::reachable_queries(module, &sys.body);

        // Intra-query aliasing + access-set union.
        let mut access = Access::default();
        for q in &queries {
            check_query_aliasing(q, diags);
            accumulate(q, &mut access);
        }

        // A system may not reach the frontend.
        //
        // Systems in one stage layer run on worker threads. The world and the
        // simulation subsystems are routed to the thread that owns the program,
        // so a worker sees the program's own - but the window, the framebuffer,
        // the font, the audio mixer and the GPU are not, and never will be:
        // sharing a window between threads is not a thing to fix.
        //
        // Refused for EVERY system rather than only for the ones that happen to
        // share a layer today, because "happens to share a layer" is not a
        // property anyone can see. A lone system runs inline and works; add an
        // unrelated second system and the first silently starts drawing into a
        // worker's empty framebuffer. A rule that holds only until the next
        // system is added is a trap, not a rule.
        //
        // The failure this replaces was silent: a worker that cannot see a
        // subsystem reports an empty one, and "nothing there" is a legal answer
        // every caller already handles. A game shipped four iterations of
        // creatures that had navigation and never once used it.
        for call in aurora_ast::reachable_calls(module, &sys.body) {
            if !aurora_abi::is_owner_only(&call) {
                continue;
            }
            diags.push(
                Diagnostic::error(format!(
                    "system `{}` reaches `{call}`, which belongs to the thread that owns the program",
                    sys.name.name
                ))
                .with_code("E0204")
                .primary(sys.name.span, format!("reaches `{call}`"))
                .note(
                    "systems in one stage layer run on worker threads, and the window,                      framebuffer, font, audio mixer and GPU are not shared with them - a call                      from a worker would draw into an empty copy and report success"
                )
                .note(
                    "do it from the frame instead: run_systems() first, then draw what the                      systems decided"
                ),
            );
        }

        systems.push(SysInfo {
            name: sys.name.name.clone(),
            span: sys.name.span,
            stage: stage_of(sys),
            after: directed_of(sys, true),
            before: directed_of(sys, false),
            access,
        });
    }

    check_schedule(&systems, diags);
}

fn stage_of(sys: &SystemDecl) -> Option<String> {
    sys.schedule.iter().find_map(|s| match s {
        SysSched::Stage(id) => Some(id.name.clone()),
        _ => None,
    })
}

/// The systems named by one direction of ordering annotation.
///
/// Kept directional rather than merged into a single "ordered with" set: the
/// direction is what lets the ordering be composed transitively, and a set that
/// has forgotten which way its edges point can only say whether two systems were
/// mentioned together, not which of them runs first.
fn directed_of(sys: &SystemDecl, want_after: bool) -> BTreeSet<String> {
    let mut set = BTreeSet::new();
    for s in &sys.schedule {
        let paths = match s {
            SysSched::After(paths) if want_after => paths,
            SysSched::Before(paths) if !want_after => paths,
            _ => continue,
        };
        // Joined, not last-segment: `after(sim::tick)` names the system that
        // module flattening called `sim::tick`, and matching on `tick` alone
        // would miss it, or match a same-named system in another module.
        for p in paths {
            set.insert(
                p.segments
                    .iter()
                    .map(|s| s.ident.name.as_str())
                    .collect::<Vec<_>>()
                    .join("::"),
            );
        }
    }
    set
}

/// The component a query-term path names, in full.
///
/// Joined rather than reduced to the last segment: after module flattening a
/// component is called `sim::Foe`, and two modules may each define a `Foe`.
/// Comparing on `Foe` would treat those as one component and report a conflict
/// between systems that never touch the same data.
fn comp_name(path: &aurora_ast::Path) -> Option<String> {
    if path.segments.is_empty() {
        return None;
    }
    Some(
        path.segments
            .iter()
            .map(|s| s.ident.name.as_str())
            .collect::<Vec<_>>()
            .join("::"),
    )
}

fn accumulate(q: &QueryExpr, access: &mut Access) {
    for term in &q.terms {
        match term {
            QTerm::Read(p) | QTerm::OptRead(p) => {
                if let Some(n) = comp_name(p) {
                    access.reads.insert(n);
                }
            }
            QTerm::Write(p) | QTerm::OptWrite(p) => {
                if let Some(n) = comp_name(p) {
                    access.writes.insert(n);
                }
            }
            // `+T` / `!T` are archetype filters, not data access. `Entity` is
            // an id, no component access.
            QTerm::With(_) | QTerm::Without(_) | QTerm::Entity => {}
        }
    }
}

/// A single query may not borrow the same component twice when one borrow is
/// mutable (it would alias mutable state within one iteration).
fn check_query_aliasing(q: &QueryExpr, diags: &mut Vec<Diagnostic>) {
    let mut reads: BTreeSet<String> = BTreeSet::new();
    let mut writes: BTreeSet<String> = BTreeSet::new();
    for term in &q.terms {
        let (set_is_write, path) = match term {
            QTerm::Read(p) | QTerm::OptRead(p) => (false, p),
            QTerm::Write(p) | QTerm::OptWrite(p) => (true, p),
            _ => continue,
        };
        let Some(name) = comp_name(path) else {
            continue;
        };

        let conflict = writes.contains(&name) || (set_is_write && reads.contains(&name));
        if conflict {
            diags.push(
                Diagnostic::error(format!(
                    "component `{name}` is borrowed more than once in a single query"
                ))
                .with_code("E0201")
                .primary(path.span, "conflicting borrow here")
                .note(
                    "a query may have any number of `&T`, or exactly one `&mut T`, per component",
                ),
            );
        }
        if set_is_write {
            writes.insert(name);
        } else {
            reads.insert(name);
        }
    }
}

/// Who must run before whom, transitively, within one stage.
///
/// `precedes[i][j]` is true when system `i` is forced ahead of system `j` by some
/// chain of `after`/`before`. Transitivity is the point: `a after(b)` and
/// `b after(c)` fixes c ahead of a just as firmly as writing `a after(c)` would,
/// and requiring the direct edge as well would be asking the programmer to
/// restate something the annotations already say. Small graphs, so the closure is
/// computed the obvious way.
fn precedes(systems: &[SysInfo], stage: &Option<String>) -> Vec<Vec<bool>> {
    let n = systems.len();
    let mut reach = vec![vec![false; n]; n];
    let at = |name: &str| (0..n).find(|&k| systems[k].name == name && &systems[k].stage == stage);
    for (i, sys) in systems.iter().enumerate() {
        if &sys.stage != stage {
            continue;
        }
        for name in &sys.after {
            if let Some(k) = at(name) {
                reach[k][i] = true;
            }
        }
        for name in &sys.before {
            if let Some(k) = at(name) {
                reach[i][k] = true;
            }
        }
    }
    for k in 0..n {
        for i in 0..n {
            if reach[i][k] {
                for j in 0..n {
                    if reach[k][j] {
                        reach[i][j] = true;
                    }
                }
            }
        }
    }
    reach
}

/// Within each stage, any two systems with conflicting access sets must be
/// ordered, so that which of them observes the other's writes is a property of
/// the program rather than of the declaration order they happen to sit in.
/// (Grammar spec Â§6.2.)
///
/// An ordering *cycle* is rejected too: it cannot be satisfied, and left alone it
/// would quietly degrade into "whatever rank the layering fell back to".
fn check_schedule(systems: &[SysInfo], diags: &mut Vec<Diagnostic>) {
    let stages: BTreeSet<Option<String>> = systems.iter().map(|s| s.stage.clone()).collect();
    let mut order: Vec<Vec<bool>> = vec![vec![false; systems.len()]; systems.len()];
    for stage in &stages {
        let reach = precedes(systems, stage);
        for (i, row) in reach.iter().enumerate() {
            for (j, &r) in row.iter().enumerate() {
                if r {
                    order[i][j] = true;
                }
            }
        }
    }

    for (i, sys) in systems.iter().enumerate() {
        if order[i][i] {
            diags.push(
                Diagnostic::error(format!(
                    "system `{}` is ordered before itself through a cycle of `after`/`before`",
                    sys.name
                ))
                .with_code("E0203")
                .primary(sys.span, "this ordering cannot be satisfied")
                .note(
                    "every `after`/`before` chain must run in one direction; \
                     remove one edge of the cycle"
                        .to_string(),
                ),
            );
        }
    }

    for i in 0..systems.len() {
        for j in (i + 1)..systems.len() {
            let (a, b) = (&systems[i], &systems[j]);

            // Systems in different stages run sequentially; never conflict.
            if a.stage != b.stage {
                continue;
            }

            let Some(component) = conflicting_component(&a.access, &b.access) else {
                continue;
            };

            if order[i][j] || order[j][i] {
                continue;
            }

            diags.push(
                Diagnostic::error(format!(
                    "systems `{}` and `{}` conflict on component `{component}` but are not ordered",
                    a.name, b.name
                ))
                .with_code("E0202")
                .primary(b.span, format!("conflicts with `{}`", a.name))
                .secondary(a.span, "the other system")
                .note(format!(
                    "add `after({})` or `before({})` to one of them to make execution deterministic",
                    a.name, b.name
                )),
            );
        }
    }
}

/// Returns a component the two access sets race on, if any: one writes it while
/// the other reads or writes it.
fn conflicting_component(a: &Access, b: &Access) -> Option<String> {
    a.writes
        .iter()
        .find(|c| b.reads.contains(*c) || b.writes.contains(*c))
        .or_else(|| {
            b.writes
                .iter()
                .find(|c| a.reads.contains(*c) || a.writes.contains(*c))
        })
        .cloned()
}
