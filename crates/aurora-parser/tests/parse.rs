//! Behavioural tests for the parser: structure of the produced AST, operator
//! precedence/associativity, and error recovery.

use aurora_ast::*;
use aurora_parser::parse_str;

/// Parse, asserting no error diagnostics, and return the module.
fn ok(src: &str) -> Module {
    let (module, diags) = parse_str(src);
    let errors: Vec<_> = diags.iter().filter(|d| d.is_error()).map(|d| &d.message).collect();
    assert!(errors.is_empty(), "unexpected errors parsing {src:?}: {errors:?}");
    module
}

/// Parse a single expression by wrapping it in a `const`, returning the value.
fn expr(src: &str) -> Expr {
    let m = ok(&format!("const C = {src}"));
    match &m.items[0].kind {
        ItemKind::Const(c) => c.value.clone(),
        other => panic!("expected const, got {other:?}"),
    }
}

#[test]
fn empty_module() {
    let m = ok("");
    assert!(m.items.is_empty());
}

#[test]
fn function_with_params_and_return() {
    let m = ok("fn add(a: i32, b: i32) -> i32 { a + b }");
    let ItemKind::Fn(f) = &m.items[0].kind else { panic!() };
    assert_eq!(f.name.name, "add");
    assert_eq!(f.params.len(), 2);
    assert!(f.ret.is_some());
    assert!(f.body.as_ref().unwrap().tail.is_some());
}

#[test]
fn struct_with_default_and_attrs() {
    let m = ok("component Spinner { @interp speed: f32 = 1.0, label: str }");
    let ItemKind::Component(s) = &m.items[0].kind else { panic!() };
    let StructBody::Named(fields) = &s.body else { panic!() };
    assert_eq!(fields.len(), 2);
    assert_eq!(fields[0].name.name, "speed");
    assert_eq!(fields[0].attrs[0].name.name, "interp");
    assert!(fields[0].default.is_some());
}

#[test]
fn replicated_component_attribute_args() {
    // `@replicated(authority: .server)` — named arg with a `.variant` value.
    let m = ok("@replicated(authority: .server) component A { health: f32 }");
    let attr = &m.items[0].attrs[0];
    assert_eq!(attr.name.name, "replicated");
    match &attr.args[0] {
        AttrArg::Named(name, value) => {
            assert_eq!(name.name, "authority");
            assert!(matches!(value.kind, ExprKind::Dot(_)));
        }
        other => panic!("expected named arg, got {other:?}"),
    }
}

#[test]
fn system_with_query_and_schedule() {
    let m = ok(
        "system mv(dt: Time) stage(Update) after(spin) {
            for (a, e) in query<&mut Avatar, Entity> where owns(e) { a.x = 1 }
         }",
    );
    let ItemKind::System(s) = &m.items[0].kind else { panic!() };
    assert_eq!(s.name.name, "mv");
    assert_eq!(s.schedule.len(), 2);
    assert!(matches!(s.schedule[0], SysSched::Stage(_)));
    assert!(matches!(s.schedule[1], SysSched::After(_)));
}

#[test]
fn query_terms_classified() {
    let e = expr("query<&mut Transform, &Spinner, Entity, ?&Vel, !Frozen, +Active>");
    let ExprKind::Query(q) = e.kind else { panic!() };
    assert!(matches!(q.terms[0], QTerm::Write(_)));
    assert!(matches!(q.terms[1], QTerm::Read(_)));
    assert!(matches!(q.terms[2], QTerm::Entity));
    assert!(matches!(q.terms[3], QTerm::OptRead(_)));
    assert!(matches!(q.terms[4], QTerm::Without(_)));
    assert!(matches!(q.terms[5], QTerm::With(_)));
}

#[test]
fn precedence_mul_binds_tighter_than_add() {
    // 1 + 2 * 3  ==  1 + (2 * 3)
    let ExprKind::Binary(BinOp::Add, _l, r) = expr("1 + 2 * 3").kind else {
        panic!("expected add at root")
    };
    assert!(matches!(r.kind, ExprKind::Binary(BinOp::Mul, _, _)));
}

#[test]
fn precedence_cmp_and_logical() {
    // a == b and c == d  ==  (a == b) and (c == d)
    let ExprKind::Binary(BinOp::And, l, r) = expr("a == b and c == d").kind else {
        panic!("expected `and` at root")
    };
    assert!(matches!(l.kind, ExprKind::Binary(BinOp::Eq, _, _)));
    assert!(matches!(r.kind, ExprKind::Binary(BinOp::Eq, _, _)));
}

#[test]
fn assignment_is_right_associative() {
    // a = b = c  ==  a = (b = c)
    let ExprKind::Assign(None, _l, r) = expr("a = b = c").kind else {
        panic!("expected assignment at root")
    };
    assert!(matches!(r.kind, ExprKind::Assign(None, _, _)));
}

#[test]
fn method_chain_and_turbofish() {
    // hit.get_mut::<Avatar>().health
    let e = expr("hit.get_mut::<Avatar>().health");
    // Outermost is a field access `.health`.
    let ExprKind::Field { base, field } = e.kind else { panic!("expected field") };
    assert!(matches!(field, FieldAccess::Named(ref i) if i.name == "health"));
    // Its base is a turbofish call.
    let ExprKind::Call { type_args, .. } = base.kind else { panic!("expected call") };
    assert_eq!(type_args.len(), 1);
}

#[test]
fn struct_literal_vs_block_in_condition() {
    // In `if cond { ... }`, the `{` is the block, not a struct literal.
    let m = ok("fn f() { if player { return } }");
    let ItemKind::Fn(f) = &m.items[0].kind else { panic!() };
    // The `if` is the block's tail expression; its cond is a plain path, not a
    // struct literal (the `{` was read as the block).
    let tail = f.body.as_ref().unwrap().tail.as_ref().expect("expected tail if-expr");
    let ExprKind::If(ifx) = &tail.kind else { panic!("expected if") };
    assert!(matches!(ifx.cond.kind, ExprKind::Path(_)));
}

#[test]
fn struct_literal_allowed_in_call_args() {
    // Inside parens, struct literals ARE allowed.
    let e = expr("spawn(Spinner { speed: 1.5 })");
    let ExprKind::Call { args, .. } = e.kind else { panic!() };
    assert!(matches!(args[0].value.kind, ExprKind::Struct { .. }));
}

#[test]
fn use_group_and_alias() {
    let m = ok("use engine::{App, Camera, load}\nuse math::Vec3 as V3");
    let ItemKind::Use(u0) = &m.items[0].kind else { panic!() };
    assert!(matches!(&u0.kind, UseKind::Group(names) if names.len() == 3));
    let ItemKind::Use(u1) = &m.items[1].kind else { panic!() };
    assert!(matches!(&u1.kind, UseKind::Single(Some(a)) if a.name == "V3"));
}

#[test]
fn trait_and_impl() {
    let m = ok(
        "trait Drawable { fn draw(&self) }
         impl Drawable for Sprite { fn draw(&self) { render(self) } }",
    );
    let ItemKind::Trait(t) = &m.items[0].kind else { panic!() };
    assert_eq!(t.items.len(), 1);
    let ItemKind::Impl(i) = &m.items[1].kind else { panic!() };
    assert!(i.trait_.is_some());
    assert_eq!(i.items.len(), 1);
}

#[test]
fn region_and_owned_types() {
    let m = ok("fn f() { let bullets = #frame make() }");
    let ItemKind::Fn(f) = &m.items[0].kind else { panic!() };
    let Stmt::Let(l) = &f.body.as_ref().unwrap().stmts[0] else { panic!() };
    assert!(matches!(l.init.as_ref().unwrap().kind, ExprKind::Region { .. }));
}

#[test]
fn error_recovery_continues_after_bad_item() {
    // A broken item should not swallow the following good one.
    let (module, diags) = parse_str("fn good1() {}\n@@@ junk\nfn good2() {}");
    assert!(diags.iter().any(|d| d.is_error()), "expected at least one error");
    let fn_names: Vec<_> = module
        .items
        .iter()
        .filter_map(|i| match &i.kind {
            ItemKind::Fn(f) => Some(f.name.name.as_str()),
            _ => None,
        })
        .collect();
    assert!(fn_names.contains(&"good1"));
    assert!(fn_names.contains(&"good2"));
}

#[test]
fn does_not_panic_on_truncated_input() {
    // Each of these is incomplete; the parser must terminate with diagnostics
    // rather than loop or panic.
    for src in ["fn", "fn f(", "struct S {", "if x", "query<", "match v {"] {
        let _ = parse_str(src);
    }
}

#[test]
fn juxtaposed_statements_without_separator_are_rejected() {
    // Two value expressions on one line with no `;`/newline is a syntax error,
    // not a silent split that discards the second (`let x = 5 6` must not
    // quietly bind `x = 5`). Catches C-style `a || b` / `!x` mistakes too,
    // since Aurora spells these `or`/`not`.
    for src in ["fn f() { let x = 5 6 }", "fn f() { let c = true || 999 }"] {
        let (_, diags) = parse_str(src);
        assert!(
            diags.iter().any(|d| d.is_error()),
            "expected a separator error for {src:?}"
        );
    }
}

#[test]
fn block_statements_need_no_separator_before_a_tail() {
    // A block-form statement (`if {…}`) is self-delimiting, so a tail
    // expression may follow on the same line without a `;`.
    let m = ok("fn f(x: i64) -> i64 { if x < 0 { return 0 - x } x }");
    let ItemKind::Fn(f) = &m.items[0].kind else { panic!() };
    let body = f.body.as_ref().unwrap();
    assert_eq!(body.stmts.len(), 1, "the `if` is a statement");
    assert!(body.tail.is_some(), "`x` is the block tail");
}

#[test]
fn region_annotated_parameter_type_parses() {
    // `#perm T` on a parameter parses as a region-annotated type wrapping `T`.
    let m = ok("fn keep(t: #perm Thing) {}");
    let ItemKind::Fn(f) = &m.items[0].kind else { panic!() };
    let Param::Normal { ty, .. } = &f.params[0] else { panic!() };
    let TypeKind::Region(region, inner) = &ty.kind else {
        panic!("expected a region-annotated type, got {:?}", ty.kind)
    };
    assert!(matches!(region, RegionKind::Perm));
    assert!(matches!(&inner.kind, TypeKind::Path(_)));
}
