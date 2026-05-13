//! ECS query and scheduler analysis (grammar spec §5.4, §6.2).

use std::collections::BTreeSet;

use aurora_ast::{ItemKind, Module, QTerm, QueryExpr, SysSched, SystemDecl};
use aurora_diag::Diagnostic;
use aurora_span::Span;

use crate::walk::queries_in_block;

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
    /// Names of systems this one is ordered relative to (either direction).
    ordered_with: BTreeSet<String>,
    access: Access,
}

pub(crate) fn check_queries_and_schedule(module: &Module, diags: &mut Vec<Diagnostic>) {
    let mut systems = Vec::new();

    for item in &module.items {
        let ItemKind::System(sys) = &item.kind else { continue };

        // Collect every query in the system body.
        let queries = queries_in_block(&sys.body);

        // Intra-query aliasing + access-set union.
        let mut access = Access::default();
        for q in &queries {
            check_query_aliasing(q, diags);
            accumulate(q, &mut access);
        }

        systems.push(SysInfo {
            name: sys.name.name.clone(),
            span: sys.name.span,
            stage: stage_of(sys),
            ordered_with: ordering_of(sys),
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

fn ordering_of(sys: &SystemDecl) -> BTreeSet<String> {
    let mut set = BTreeSet::new();
    for s in &sys.schedule {
        match s {
            SysSched::After(paths) | SysSched::Before(paths) => {
                for p in paths {
                    if let Some(seg) = p.segments.last() {
                        set.insert(seg.ident.name.clone());
                    }
                }
            }
            SysSched::Stage(_) => {}
        }
    }
    set
}

/// The component named by a query-term path (its last segment).
fn comp_name(path: &aurora_ast::Path) -> Option<String> {
    path.segments.last().map(|s| s.ident.name.clone())
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
        let Some(name) = comp_name(path) else { continue };

        let conflict = writes.contains(&name) || (set_is_write && reads.contains(&name));
        if conflict {
            diags.push(
                Diagnostic::error(format!(
                    "component `{name}` is borrowed more than once in a single query"
                ))
                .with_code("E0201")
                .primary(path.span, "conflicting borrow here")
                .note("a query may have any number of `&T`, or exactly one `&mut T`, per component"),
            );
        }
        if set_is_write {
            writes.insert(name);
        } else {
            reads.insert(name);
        }
    }
}

/// Within each stage, any two systems with conflicting access sets must be
/// explicitly ordered. Otherwise the scheduler would run them in parallel and
/// race. (Grammar spec §6.2.)
fn check_schedule(systems: &[SysInfo], diags: &mut Vec<Diagnostic>) {
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

            let ordered =
                a.ordered_with.contains(&b.name) || b.ordered_with.contains(&a.name);
            if ordered {
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
        .or_else(|| b.writes.iter().find(|c| a.reads.contains(*c) || a.writes.contains(*c)))
        .cloned()
}
