//! Lightweight name resolution (Phase A3, single-file scope).
//!
//! Full expression-local resolution waits on a real prelude/stdlib model (so it
//! doesn't false-positive on graphics builtins like `texture`/`vec4`). For now
//! we resolve the references that are unambiguous and valuable:
//!
//! * **Query components** must name a `component` — not a `struct`/`enum`, and
//!   not an undefined local name. Imported and builtin type names are accepted
//!   (we can't see other modules, and the builtins like `Transform` are real).
//! * **`after`/`before`** must name a real `system`.

use std::collections::HashSet;

use aurora_ast::{ItemKind, Module, QTerm, SysSched};
use aurora_diag::Diagnostic;

use crate::walk::queries_in_block;

/// Builtin type names from grammar spec §2.2 (primitives + math + ECS leaves).
const BUILTIN_TYPES: &[&str] = &[
    "f32", "f64", "i8", "i16", "i32", "i64", "u8", "u16", "u32", "u64", "bool", "char", "str",
    "void", "Vec2", "Vec3", "Vec4", "Mat2", "Mat3", "Mat4", "Quat", "Color", "Transform", "Time",
    "Entity", "Handle", "Option", "Result", "rc", "weak",
];

struct Tables {
    components: HashSet<String>,
    /// Local `struct`/`enum` types (not components).
    value_types: HashSet<String>,
    systems: HashSet<String>,
    /// Names brought into scope by `use` (we can't verify their kind).
    imported: HashSet<String>,
}

impl Tables {
    fn collect(module: &Module) -> Tables {
        let mut t = Tables {
            components: HashSet::new(),
            value_types: HashSet::new(),
            systems: HashSet::new(),
            imported: HashSet::new(),
        };
        for item in &module.items {
            match &item.kind {
                ItemKind::Component(c) => {
                    t.components.insert(c.name.name.clone());
                }
                ItemKind::Struct(s) => {
                    t.value_types.insert(s.name.name.clone());
                }
                ItemKind::Enum(e) => {
                    t.value_types.insert(e.name.name.clone());
                }
                ItemKind::System(s) => {
                    t.systems.insert(s.name.name.clone());
                }
                ItemKind::Use(u) => match &u.kind {
                    aurora_ast::UseKind::Single(alias) => {
                        let name = alias
                            .as_ref()
                            .map(|a| a.name.clone())
                            .or_else(|| u.path.segments.last().map(|s| s.ident.name.clone()));
                        if let Some(n) = name {
                            t.imported.insert(n);
                        }
                    }
                    aurora_ast::UseKind::Group(names) => {
                        for n in names {
                            t.imported.insert(n.name.clone());
                        }
                    }
                },
                _ => {}
            }
        }
        t
    }

    /// Could `name` plausibly be a component type? (local component, imported,
    /// or a builtin type — anything we can't disprove.)
    fn maybe_component(&self, name: &str) -> bool {
        self.components.contains(name)
            || self.imported.contains(name)
            || BUILTIN_TYPES.contains(&name)
    }
}

pub(crate) fn resolve(module: &Module, diags: &mut Vec<Diagnostic>) {
    let tables = Tables::collect(module);

    for item in &module.items {
        let ItemKind::System(sys) = &item.kind else { continue };

        // `after`/`before` must reference real systems.
        for sched in &sys.schedule {
            if let SysSched::After(paths) | SysSched::Before(paths) = sched {
                for p in paths {
                    let Some(seg) = p.segments.last() else { continue };
                    if !tables.systems.contains(&seg.ident.name) {
                        diags.push(
                            Diagnostic::error(format!(
                                "`{}` orders against unknown system `{}`",
                                sys.name.name, seg.ident.name
                            ))
                            .with_code("E0210")
                            .primary(p.span, "no system with this name"),
                        );
                    }
                }
            }
        }

        // Query terms must name components.
        for q in queries_in_block(&sys.body) {
            for term in &q.terms {
                let path = match term {
                    QTerm::Read(p)
                    | QTerm::Write(p)
                    | QTerm::OptRead(p)
                    | QTerm::OptWrite(p)
                    | QTerm::With(p)
                    | QTerm::Without(p) => p,
                    QTerm::Entity => continue,
                };
                // Multi-segment paths (`foo::Bar`) are assumed valid; we only
                // judge single, local-looking names.
                if path.segments.len() != 1 {
                    continue;
                }
                let name = &path.segments[0].ident.name;

                if tables.components.contains(name) || tables.maybe_component(name) {
                    continue;
                }
                if tables.value_types.contains(name) {
                    diags.push(
                        Diagnostic::error(format!(
                            "`{name}` is a struct/enum, not a component"
                        ))
                        .with_code("E0211")
                        .primary(path.span, "queries iterate components")
                        .note(format!("declare it with `component {name} {{ ... }}` to query it")),
                    );
                } else {
                    diags.push(
                        Diagnostic::error(format!("unknown component `{name}`"))
                            .with_code("E0212")
                            .primary(path.span, "not a component in scope"),
                    );
                }
            }
        }
    }
}
