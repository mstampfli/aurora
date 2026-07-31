//! ECS declarations inside a module.
//!
//! A component and the query that reads it are ordinary items and ordinary
//! references, so putting them in a module has to keep them pointing at each
//! other. It did not: the declaration was mangled and the query was not, which
//! made modules unusable for any program with an ECS in it - the checker
//! reported a component missing that was declared three lines above.

use aurora_parser::parse_str;

fn flat(src: &str) -> Vec<String> {
    let (module, diags) = parse_str(src);
    assert!(
        !diags.iter().any(|d| d.is_error()),
        "parse errors: {:?}",
        diags.iter().filter(|d| d.is_error()).collect::<Vec<_>>()
    );
    module
        .items
        .iter()
        .map(|i| format!("{:?}", i.kind))
        .collect()
}

/// The flattened program, rendered as debug text, so a test can ask whether a
/// name survived rewriting without reaching into every AST node by hand.
fn dump(src: &str) -> String {
    flat(src).join("\n")
}

#[test]
fn a_query_in_a_module_reaches_that_modules_components() {
    let text = dump(
        r#"
mod sim {
    component Player { hp: i64 }
    fn count() -> i64 {
        let mut n = 0
        for p in query<&Player> { n = n + 1 }
        n
    }
}
"#,
    );
    assert!(text.contains("sim::Player"), "component was not mangled");
    // The query must name the mangled component, not the bare one.
    assert!(
        !text.contains("\"Player\""),
        "a bare `Player` survived rewriting, so the query still points outside the module:\n{text}"
    );
}

#[test]
fn a_system_in_a_module_reaches_that_modules_components() {
    let text = dump(
        r#"
mod sim {
    component Player { hp: i64 }
    system tick() stage(FixedUpdate) {
        for p in query<&mut Player> { p.hp = p.hp + 1 }
    }
}
"#,
    );
    assert!(text.contains("sim::Player"));
    assert!(
        !text.contains("\"Player\""),
        "the system body was not rewritten:\n{text}"
    );
}

#[test]
fn a_system_is_mangled_like_any_other_item() {
    // Two modules may define the same system name. Leaving systems unmangled
    // let them collide silently.
    let text = dump(
        r#"
mod a { system tick() { } }
mod b { system tick() { } }
"#,
    );
    assert!(text.contains("a::tick"), "{text}");
    assert!(text.contains("b::tick"), "{text}");
}

#[test]
fn system_ordering_resolves_to_the_sibling_in_the_same_module() {
    let text = dump(
        r#"
mod sim {
    system first() { }
    system second() after(first) { }
}
"#,
    );
    assert!(text.contains("sim::first"), "{text}");
    assert!(text.contains("sim::second"), "{text}");
    assert!(
        !text.contains("\"first\""),
        "the after() path was not rewritten, so ordering points outside the module:\n{text}"
    );
}

#[test]
fn a_stage_name_is_left_alone() {
    // A stage names a schedule the runtime owns, not an item in this module.
    // Prefixing it would give every module a private stage of its own, and a
    // `FixedUpdate` system in a module would stop running on the fixed clock.
    let text = dump(
        r#"
mod sim {
    component P { x: i64 }
    system tick() stage(FixedUpdate) { for p in query<&P> { } }
}
"#,
    );
    assert!(
        text.contains("FixedUpdate") && !text.contains("sim::FixedUpdate"),
        "the stage name was mangled:\n{text}"
    );
}
