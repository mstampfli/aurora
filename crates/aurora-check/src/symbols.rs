//! Duplicate-definition detection across namespaces.

use std::collections::HashMap;

use aurora_ast::{Ident, ItemKind, Module};
use aurora_diag::Diagnostic;
use aurora_span::Span;

/// Which namespace a definition occupies. Types and values are separate (a fn
/// and a struct may share a name), but `struct`/`enum`/`component`/`trait` all
/// share the type namespace.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum Ns {
    Type,
    Value,
    System,
    Pipeline,
}

type SeenMap = HashMap<(Ns, String), Span>;

pub(crate) fn check_duplicates(module: &Module, diags: &mut Vec<Diagnostic>) {
    let mut seen: SeenMap = HashMap::new();

    for item in &module.items {
        match &item.kind {
            ItemKind::Struct(s) => record(&mut seen, Ns::Type, &s.name, "type", diags),
            ItemKind::Enum(e) => record(&mut seen, Ns::Type, &e.name, "type", diags),
            ItemKind::Component(c) => record(&mut seen, Ns::Type, &c.name, "component", diags),
            ItemKind::Trait(t) => record(&mut seen, Ns::Type, &t.name, "trait", diags),
            ItemKind::Fn(f) => record(&mut seen, Ns::Value, &f.name, "function", diags),
            ItemKind::Const(c) => record(&mut seen, Ns::Value, &c.name, "constant", diags),
            ItemKind::System(s) => record(&mut seen, Ns::System, &s.name, "system", diags),
            ItemKind::Pipeline(p) => record(&mut seen, Ns::Pipeline, &p.name, "pipeline", diags),
            _ => {}
        }
    }
}

fn record(seen: &mut SeenMap, ns: Ns, ident: &Ident, kind: &str, diags: &mut Vec<Diagnostic>) {
    let key = (ns, ident.name.clone());
    if let Some(&prev) = seen.get(&key) {
        diags.push(
            Diagnostic::error(format!("duplicate {kind} `{}`", ident.name))
                .with_code("E0200")
                .primary(ident.span, "redefined here")
                .secondary(prev, "first defined here"),
        );
    } else {
        seen.insert(key, ident.span);
    }
}
