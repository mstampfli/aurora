//! Tests for the AST-level checks: duplicate definitions, intra-query
//! aliasing, and — the headline feature — ECS scheduler conflict detection.

use aurora_check::check;
use aurora_parser::parse_str;

/// Parse `src` (asserting it parses without error) and return check diagnostics.
fn check_src(src: &str) -> Vec<String> {
    let (module, pdiags) = parse_str(src);
    assert!(
        !pdiags.iter().any(|d| d.is_error()),
        "source failed to parse: {:?}",
        pdiags.iter().map(|d| &d.message).collect::<Vec<_>>()
    );
    check(&module)
        .into_iter()
        .filter(|d| d.is_error())
        .map(|d| d.message)
        .collect()
}

#[test]
fn clean_program_has_no_errors() {
    let errs = check_src(
        "component Transform { x: f32 }
         component Spinner { s: f32 }
         system spin(dt: Time) stage(Update) {
             for (t, s) in query<&mut Transform, &Spinner> { t.x = s.s }
         }",
    );
    assert!(errs.is_empty(), "expected clean, got {errs:?}");
}

#[test]
fn unordered_conflicting_systems_error() {
    // Both write Transform in the same stage, with no ordering: a data race.
    let errs = check_src(
        "component Transform { x: f32 }
         system a(dt: Time) stage(Update) {
             for t in query<&mut Transform> { t.x = 1 }
         }
         system b(dt: Time) stage(Update) {
             for t in query<&mut Transform> { t.x = 2 }
         }",
    );
    assert_eq!(errs.len(), 1, "got {errs:?}");
    assert!(errs[0].contains("conflict on component `Transform`"));
}

#[test]
fn ordering_resolves_conflict() {
    // The same two systems, but `b after(a)` makes the schedule deterministic.
    let errs = check_src(
        "component Transform { x: f32 }
         system a(dt: Time) stage(Update) {
             for t in query<&mut Transform> { t.x = 1 }
         }
         system b(dt: Time) stage(Update) after(a) {
             for t in query<&mut Transform> { t.x = 2 }
         }",
    );
    assert!(
        errs.is_empty(),
        "ordering should resolve conflict, got {errs:?}"
    );
}

#[test]
fn read_read_does_not_conflict() {
    // Two systems that only read the same component can run in parallel safely.
    let errs = check_src(
        "component Transform { x: f32 }
         system a(dt: Time) stage(Update) {
             for t in query<&Transform> { read(t) }
         }
         system b(dt: Time) stage(Update) {
             for t in query<&Transform> { read(t) }
         }",
    );
    assert!(errs.is_empty(), "read/read must not conflict, got {errs:?}");
}

#[test]
fn different_stages_never_conflict() {
    let errs = check_src(
        "component Transform { x: f32 }
         system a(dt: Time) stage(Update) {
             for t in query<&mut Transform> { t.x = 1 }
         }
         system b(dt: Time) stage(Render) {
             for t in query<&mut Transform> { t.x = 2 }
         }",
    );
    assert!(
        errs.is_empty(),
        "different stages run sequentially, got {errs:?}"
    );
}

#[test]
fn write_read_conflict_detected() {
    // a writes Transform, b reads it: still a race without ordering.
    let errs = check_src(
        "component Transform { x: f32 }
         system a(dt: Time) stage(Update) {
             for t in query<&mut Transform> { t.x = 1 }
         }
         system b(dt: Time) stage(Update) {
             for t in query<&Transform> { read(t) }
         }",
    );
    assert_eq!(errs.len(), 1, "got {errs:?}");
    assert!(errs[0].contains("Transform"));
}

#[test]
fn intra_query_aliasing_error() {
    // Same component borrowed mutably twice in one query.
    let errs = check_src(
        "component T { x: f32 }
         system s(dt: Time) {
             for (a, b) in query<&mut T, &mut T> { use2(a, b) }
         }",
    );
    assert!(
        errs.iter().any(|e| e.contains("borrowed more than once")),
        "got {errs:?}"
    );
}

#[test]
fn duplicate_component_error() {
    let errs = check_src("component A { x: f32 }\ncomponent A { y: f32 }");
    assert!(
        errs.iter().any(|e| e.contains("duplicate component `A`")),
        "got {errs:?}"
    );
}

#[test]
fn fn_and_struct_may_share_name() {
    // Different namespaces: no error.
    let errs = check_src("struct Mesh { n: i32 }\nfn Mesh() {}");
    assert!(
        errs.is_empty(),
        "type and value namespaces are separate, got {errs:?}"
    );
}

#[test]
fn query_on_struct_is_rejected() {
    // Querying a `struct` (not a `component`) is a resolution error.
    let errs = check_src(
        "struct NotAComp { x: f32 }
         system s(dt: Time) { for n in query<&NotAComp> { use1(n) } }",
    );
    assert!(
        errs.iter().any(|e| e.contains("not a component")),
        "got {errs:?}"
    );
}

#[test]
fn query_on_unknown_component_is_rejected() {
    let errs = check_src("system s(dt: Time) { for n in query<&Ghost> { use1(n) } }");
    assert!(
        errs.iter().any(|e| e.contains("unknown component `Ghost`")),
        "got {errs:?}"
    );
}

#[test]
fn query_on_builtin_and_imported_is_accepted() {
    // `Transform` is a builtin; `Foo` is imported — both accepted without a
    // local `component` declaration.
    let errs = check_src(
        "use engine::{Foo}
         system s(dt: Time) { for (t, f) in query<&mut Transform, &Foo> { go(t, f) } }",
    );
    assert!(
        errs.is_empty(),
        "builtin/imported components should resolve, got {errs:?}"
    );
}

// --- region escape checking ----------------------------------------------

#[test]
fn frame_inside_perm_is_rejected() {
    // Storing a `#frame` allocation in a field of a `#perm` allocation: the
    // perm data outlives the frame, so the pointer would dangle.
    let errs = check_src(
        "fn f() {
             let cache = #perm Holder { inner: #frame Thing { x: 1 } }
         }",
    );
    assert!(
        errs.iter()
            .any(|e| e.contains("stored inside a longer-lived")),
        "got {errs:?}"
    );
}

#[test]
fn longer_inside_shorter_is_fine() {
    // A `#perm` value stored inside a `#frame` allocation is fine (perm
    // outlives frame).
    let errs = check_src(
        "fn f() {
             let tmp = #frame Holder { inner: #perm Thing { x: 1 } }
         }",
    );
    assert!(
        errs.is_empty(),
        "longer-lived inside shorter-lived is ok, got {errs:?}"
    );
}

#[test]
fn frame_passed_to_call_in_perm_is_not_flagged() {
    // A `#frame` value merely passed to a call (not stored) inside a `#perm`
    // expression must not false-positive.
    let errs = check_src(
        "fn f() {
             let x = #perm build(#frame Thing { x: 1 })
         }",
    );
    assert!(
        errs.is_empty(),
        "transient frame arg must not be flagged, got {errs:?}"
    );
}

#[test]
fn frame_binding_stored_in_perm_is_rejected() {
    // The region flows through a `let`: `tmp` is a `#frame` allocation, then it
    // is stored into a `#perm` allocation. The escape must be caught even though
    // the allocation isn't written inline at the storage site.
    let errs = check_src(
        "fn f() {
             let tmp = #frame Thing { x: 1 }
             let cache = #perm Holder { inner: tmp }
         }",
    );
    assert!(
        errs.iter()
            .any(|e| e.contains("stored inside a longer-lived")),
        "a frame-region binding stored into perm should be flagged, got {errs:?}"
    );
}

#[test]
fn perm_binding_stored_in_frame_is_fine() {
    // The symmetric legal case through a binding must not false-positive.
    let errs = check_src(
        "fn f() {
             let keep = #perm Thing { x: 1 }
             let tmp = #frame Holder { inner: keep }
         }",
    );
    assert!(
        errs.is_empty(),
        "perm binding inside frame is ok, got {errs:?}"
    );
}

#[test]
fn frame_returning_call_stored_in_perm_is_rejected() {
    // The region crosses a function boundary: `make_frame` returns a `#frame`
    // allocation, and its result is stored into a `#perm` allocation. Inferring
    // the callee's return region catches the escape.
    let errs = check_src(
        "fn make_frame() -> Thing { #frame Thing { x: 1 } }
         fn f() { let cache = #perm Holder { inner: make_frame() } }",
    );
    assert!(
        errs.iter()
            .any(|e| e.contains("stored inside a longer-lived")),
        "a frame-returning call stored into perm should be flagged, got {errs:?}"
    );
}

#[test]
fn perm_returning_call_in_frame_is_fine() {
    // Symmetric legal case across a call boundary must not false-positive.
    let errs = check_src(
        "fn make_perm() -> Thing { #perm Thing { x: 1 } }
         fn f() { let tmp = #frame Holder { inner: make_perm() } }",
    );
    assert!(
        errs.is_empty(),
        "perm-returning call inside frame is ok, got {errs:?}"
    );
}

#[test]
fn region_polymorphic_passthrough_propagates_arg_region() {
    // `id` returns its argument, so it inherits the argument's region. Passing a
    // `#frame` allocation through `id` and storing the result in `#perm` must
    // still be caught — region flows through the polymorphic signature.
    let errs = check_src(
        "fn id(x: Thing) -> Thing { x }
         fn f() {
             let tmp = #frame Thing { x: 1 }
             let cache = #perm Holder { inner: id(tmp) }
         }",
    );
    assert!(
        errs.iter()
            .any(|e| e.contains("stored inside a longer-lived")),
        "region must flow through a passthrough fn, got {errs:?}"
    );
}

#[test]
fn assignment_of_frame_into_perm_field_is_rejected() {
    // A store via assignment (not a struct literal) into a longer-lived field —
    // the kind of escape that happens inside loops.
    let errs = check_src(
        "fn f() {
             let cache = #perm Holder { x: 0 }
             let tmp = #frame Thing { x: 1 }
             cache.inner = tmp
         }",
    );
    assert!(
        errs.iter().any(|e| e.contains("longer-lived")),
        "assigning a frame value into a perm field should be flagged, got {errs:?}"
    );
}

#[test]
fn loop_carried_frame_store_into_perm_is_rejected() {
    // The escape is inside a loop body; the destination outlives the loop.
    let errs = check_src(
        "fn f() {
             let cache = #perm Holder { x: 0 }
             for i in 0..3 {
                 let tmp = #frame Thing { x: 1 }
                 cache.slot = tmp
             }
         }",
    );
    assert!(
        errs.iter().any(|e| e.contains("longer-lived")),
        "a loop-carried frame store should be flagged, got {errs:?}"
    );
}

#[test]
fn assignment_of_perm_into_frame_field_is_fine() {
    let errs = check_src(
        "fn f() {
             let tmp = #frame Holder { x: 0 }
             let keep = #perm Thing { x: 1 }
             tmp.inner = keep
         }",
    );
    assert!(errs.is_empty(), "perm into frame field is ok, got {errs:?}");
}

#[test]
fn region_polymorphic_passthrough_legal_case_is_fine() {
    // The reverse (perm through id, stored in frame) is legal.
    let errs = check_src(
        "fn id(x: Thing) -> Thing { x }
         fn f() {
             let keep = #perm Thing { x: 1 }
             let tmp = #frame Holder { inner: id(keep) }
         }",
    );
    assert!(
        errs.is_empty(),
        "perm through id inside frame is ok, got {errs:?}"
    );
}

#[test]
fn call_with_unknown_region_is_not_flagged() {
    // A call whose return region can't be inferred makes no assumption.
    let errs = check_src("fn f() { let cache = #perm Holder { inner: build() } }");
    assert!(
        errs.is_empty(),
        "unknown call region must not be flagged, got {errs:?}"
    );
}

#[test]
fn non_region_binding_in_perm_is_not_flagged() {
    // An ordinary (non-region) binding stored into a `#perm` allocation must not
    // be flagged — we only constrain values with a known shorter-lived region.
    let errs = check_src(
        "fn f() {
             let v = Thing { x: 1 }
             let cache = #perm Holder { inner: v }
         }",
    );
    assert!(
        errs.is_empty(),
        "plain binding must not be flagged, got {errs:?}"
    );
}

// --- move checking (owned `~T`) ------------------------------------------

#[test]
fn use_after_move_is_caught() {
    let errs = check_src(
        "fn consume(x: ~Mesh) {}
         fn f(m: ~Mesh) {
             consume(m)
             consume(m)
         }",
    );
    assert!(
        errs.iter().any(|e| e.contains("use of moved value `m`")),
        "got {errs:?}"
    );
}

#[test]
fn borrow_does_not_move() {
    let errs = check_src(
        "fn peek(x: &Mesh) {}
         fn consume(x: ~Mesh) {}
         fn f(m: ~Mesh) {
             peek(&m)
             consume(m)
         }",
    );
    assert!(errs.is_empty(), "borrowing must not move, got {errs:?}");
}

#[test]
fn reassignment_revives_a_moved_binding() {
    let errs = check_src(
        "fn consume(x: ~Mesh) {}
         fn make() -> ~Mesh { make() }
         fn f() {
             let mut m: ~Mesh = make()
             consume(m)
             m = make()
             consume(m)
         }",
    );
    assert!(errs.is_empty(), "reassignment should revive, got {errs:?}");
}

#[test]
fn move_in_loop_is_caught() {
    let errs = check_src(
        "fn consume(x: ~Mesh) {}
         fn f(m: ~Mesh) {
             while true { consume(m) }
         }",
    );
    assert!(
        errs.iter().any(|e| e.contains("moved inside a loop")),
        "got {errs:?}"
    );
}

#[test]
fn move_in_one_if_branch_only_is_ok_after() {
    // Moving in a branch then using only within that branch is fine; the binding
    // is used before the move on the else path.
    let errs = check_src(
        "fn consume(x: ~Mesh) {}
         fn f(m: ~Mesh, cond: bool) {
             if cond { consume(m) } else { consume(m) }
         }",
    );
    assert!(
        errs.is_empty(),
        "mutually-exclusive branch moves are fine, got {errs:?}"
    );
}

#[test]
fn non_owned_values_are_never_move_checked() {
    // `Mesh` (not `~Mesh`) is a plain value; using it twice is fine here.
    let errs = check_src(
        "fn consume(x: Mesh) {}
         fn f(m: Mesh) {
             consume(m)
             consume(m)
         }",
    );
    assert!(errs.is_empty(), "only ~T is move-tracked, got {errs:?}");
}

#[test]
fn after_unknown_system_is_rejected() {
    let errs = check_src(
        "component T { x: f32 }
         system a(dt: Time) stage(Update) after(ghost) {
             for t in query<&mut T> { t.x = 1 }
         }",
    );
    assert!(
        errs.iter().any(|e| e.contains("unknown system `ghost`")),
        "got {errs:?}"
    );
}

#[test]
fn frame_arg_into_perm_storing_callee_is_rejected() {
    // Region-parameterized inference: `keep` stores its parameter in `#perm`, so
    // passing it a `#frame` value would dangle — caught at the call site.
    let errs = check_src(
        "struct Box { v: Thing }
         struct Thing { x: i64 }
         fn keep(t: Thing) { let cache = #perm Box { v: t } }
         fn use_it() { keep(#frame Thing { x: 1 }) }",
    );
    assert!(
        errs.iter()
            .any(|e| e.contains("longer-lived") && e.contains("#perm")),
        "expected a cross-call region-escape error, got {errs:?}"
    );
}

#[test]
fn perm_arg_into_perm_storing_callee_is_fine() {
    // Passing a `#perm` argument to the same function is sound — no false positive.
    let errs = check_src(
        "struct Box { v: Thing }
         struct Thing { x: i64 }
         fn keep(t: Thing) { let cache = #perm Box { v: t } }
         fn use_it() { keep(#perm Thing { x: 1 }) }",
    );
    assert!(errs.is_empty(), "perm arg must be allowed, got {errs:?}");
}

#[test]
fn extern_region_param_annotation_is_enforced() {
    // A bodiless `@extern` declares its parameter's region with `#perm`; the
    // contract is enforced at call sites even though there's no body to infer
    // from. `#frame` is rejected; `#perm` is accepted.
    let bad = check_src(
        "struct Thing { x: i64 }
         @extern fn ffi_keep(t: #perm Thing)
         fn use_it() { ffi_keep(#frame Thing { x: 1 }) }",
    );
    assert!(
        bad.iter()
            .any(|e| e.contains("longer-lived") && e.contains("#perm")),
        "expected a region-contract error, got {bad:?}"
    );
    let ok = check_src(
        "struct Thing { x: i64 }
         @extern fn ffi_keep(t: #perm Thing)
         fn use_it() { ffi_keep(#perm Thing { x: 1 }) }",
    );
    assert!(ok.is_empty(), "perm arg satisfies the contract, got {ok:?}");
}

#[test]
fn extern_region_return_annotation_is_enforced() {
    // A `-> #frame` return annotation makes the result short-lived, so storing
    // it in a `#perm` allocation is rejected — across a bodiless boundary.
    let errs = check_src(
        "struct Thing { x: i64 }
         struct Holder { inner: Thing }
         @extern fn ffi_tmp() -> #frame Thing
         fn use_it() { let c = #perm Holder { inner: ffi_tmp() } }",
    );
    assert!(
        errs.iter()
            .any(|e| e.contains("stored inside a longer-lived")),
        "expected a return-region escape error, got {errs:?}"
    );
}

// --- transitive ordering (§6.2) --------------------------------------------
//
// `after`/`before` compose. Demanding a direct edge between every conflicting
// pair would mean restating an order the annotations already fix, and the
// restatement is the kind that rots: the chain gets rearranged and the redundant
// edge is left behind, now saying something different from the chain.

/// Three systems in a chain, all writing the same component. `a -> b -> c` fixes
/// a ahead of c, so nothing here is ambiguous and nothing should be reported.
#[test]
fn a_transitive_chain_orders_the_ends() {
    let errs = check_src(
        "component P { x: f32 }
         system a() stage(Update) { for p in query<&mut P> { p.x = 1.0 } }
         system b() stage(Update) after(a) { for p in query<&mut P> { p.x = 2.0 } }
         system c() stage(Update) after(b) { for p in query<&mut P> { p.x = 3.0 } }",
    );
    assert!(
        errs.is_empty(),
        "a and c are ordered through b; requiring the direct edge too is noise: {errs:?}"
    );
}

/// The same chain built from `before` instead, and one built from both. Direction
/// has to survive being written from either end or the closure is only half right.
#[test]
fn a_chain_composes_from_either_direction() {
    for chain in [
        "system a() stage(Update) before(b) { for p in query<&mut P> { p.x = 1.0 } }
         system b() stage(Update) before(c) { for p in query<&mut P> { p.x = 2.0 } }
         system c() stage(Update) { for p in query<&mut P> { p.x = 3.0 } }",
        "system a() stage(Update) before(b) { for p in query<&mut P> { p.x = 1.0 } }
         system b() stage(Update) { for p in query<&mut P> { p.x = 2.0 } }
         system c() stage(Update) after(b) { for p in query<&mut P> { p.x = 3.0 } }",
    ] {
        let errs = check_src(&format!("component P {{ x: f32 }}\n{chain}"));
        assert!(
            errs.is_empty(),
            "chain should order a and c: {errs:?}\n{chain}"
        );
    }
}

/// The relaxation must not swallow the case it exists to catch: two conflicting
/// systems with no chain between them at all are still an error. Without this the
/// transitive test above would pass just as well against a checker that reported
/// nothing.
#[test]
fn an_unordered_conflicting_pair_is_still_rejected() {
    let errs = check_src(
        "component P { x: f32 }
         system a() stage(Update) { for p in query<&mut P> { p.x = 1.0 } }
         system b() stage(Update) { for p in query<&mut P> { p.x = 2.0 } }",
    );
    assert_eq!(errs.len(), 1, "expected exactly one conflict: {errs:?}");
    assert!(errs[0].contains("not ordered"), "{errs:?}");
}

/// A chain that reaches a third system does not order the two ends of a *different*
/// branch. `b after(a)` and `c after(a)` leaves b and c siblings, still ambiguous.
#[test]
fn siblings_of_a_common_predecessor_are_not_ordered() {
    let errs = check_src(
        "component P { x: f32 }
         system a() stage(Update) { for p in query<&mut P> { p.x = 1.0 } }
         system b() stage(Update) after(a) { for p in query<&mut P> { p.x = 2.0 } }
         system c() stage(Update) after(a) { for p in query<&mut P> { p.x = 3.0 } }",
    );
    assert_eq!(errs.len(), 1, "b and c are siblings, not ordered: {errs:?}");
    assert!(
        errs[0].contains("`b`") && errs[0].contains("`c`"),
        "{errs:?}"
    );
}

/// A chain does not leak across a stage boundary. Stages already run in sequence,
/// so an edge between them cannot be honoured, and treating it as ordering would
/// let a conflicting same-stage pair through on the strength of it.
#[test]
fn ordering_does_not_compose_across_stages() {
    let errs = check_src(
        "component P { x: f32 }
         system a() stage(Update) { for p in query<&mut P> { p.x = 1.0 } }
         system mid() stage(FixedUpdate) after(a) { for p in query<&mut P> { p.x = 9.0 } }
         system c() stage(Update) after(mid) { for p in query<&mut P> { p.x = 3.0 } }",
    );
    assert!(
        errs.iter().any(|e| e.contains("not ordered")),
        "a and c are both in Update with only a cross-stage hop between them: {errs:?}"
    );
}

/// An ordering cycle cannot be satisfied. Previously it was accepted and then
/// quietly resolved to whatever the layering fell back to.
#[test]
fn an_ordering_cycle_is_rejected() {
    let errs = check_src(
        "component P { x: f32 }
         system a() stage(Update) after(c) { for p in query<&mut P> { p.x = 1.0 } }
         system b() stage(Update) after(a) { for p in query<&mut P> { p.x = 2.0 } }
         system c() stage(Update) after(b) { for p in query<&mut P> { p.x = 3.0 } }",
    );
    assert!(
        errs.iter().any(|e| e.contains("cycle")),
        "a -> b -> c -> a is unsatisfiable: {errs:?}"
    );
}

/// A system ordered against itself is the degenerate cycle, and is caught too.
#[test]
fn a_self_ordering_is_rejected() {
    let errs = check_src(
        "component P { x: f32 }
         system a() stage(Update) after(a) { for p in query<&mut P> { p.x = 1.0 } }",
    );
    assert!(errs.iter().any(|e| e.contains("cycle")), "{errs:?}");
}

// --- access through called functions (§6.2) ---------------------------------
//
// A system's component access is what it *reaches*, not what it spells out. The
// analysis used to read only the system body, so moving a query into a helper
// hid it from the race-freedom proof: two systems that both wrote `Player`
// through helpers were declared independent and handed to the thread pool
// together. The escape hatch was a refactor away, and refactors are exactly when
// nobody re-reads the scheduler.

#[test]
fn a_query_inside_a_called_function_still_counts_as_access() {
    let errs = check_src(
        "component P { x: f32 }
         fn bump() { for p in query<&mut P> { p.x = 1.0 } }
         system a() stage(Update) { bump() }
         system b() stage(Update) { for p in query<&mut P> { p.x = 2.0 } }",
    );
    assert_eq!(
        errs.len(),
        1,
        "`a` writes P through `bump`; hiding a query in a helper must not hide the conflict: {errs:?}"
    );
    assert!(errs[0].contains("not ordered"), "{errs:?}");
}

#[test]
fn access_is_followed_through_a_chain_of_calls() {
    let errs = check_src(
        "component P { x: f32 }
         fn inner() { for p in query<&mut P> { p.x = 1.0 } }
         fn outer() { inner() }
         system a() stage(Update) { outer() }
         system b() stage(Update) { for p in query<&mut P> { p.x = 2.0 } }",
    );
    assert_eq!(
        errs.len(),
        1,
        "access must follow the whole call chain: {errs:?}"
    );
}

#[test]
fn both_systems_reaching_a_component_only_through_helpers_conflict() {
    let errs = check_src(
        "component P { x: f32 }
         fn write_it() { for p in query<&mut P> { p.x = 1.0 } }
         fn read_it() -> f32 { let mut v = 0.0; for p in query<&P> { v = p.x } v }
         system a() stage(Update) { write_it() }
         system b() stage(Update) { let v = read_it() }",
    );
    assert_eq!(
        errs.len(),
        1,
        "neither body names a query, but both reach P: {errs:?}"
    );
}

/// Ordering the pair resolves it, exactly as it would for inline queries -
/// the new analysis must not become impossible to satisfy.
#[test]
fn ordering_resolves_a_conflict_reached_through_a_call() {
    let errs = check_src(
        "component P { x: f32 }
         fn bump() { for p in query<&mut P> { p.x = 1.0 } }
         system a() stage(Update) { bump() }
         system b() stage(Update) after(a) { for p in query<&mut P> { p.x = 2.0 } }",
    );
    assert!(errs.is_empty(), "{errs:?}");
}

/// Recursion must terminate rather than chase itself forever.
#[test]
fn a_recursive_helper_terminates() {
    let errs = check_src(
        "component P { x: f32 }
         fn spin(n: i64) { if n > 0 { spin(n - 1) } for p in query<&mut P> { p.x = 1.0 } }
         system a() stage(Update) { spin(3) }
         system b() stage(Update) { for p in query<&mut P> { p.x = 2.0 } }",
    );
    assert_eq!(errs.len(), 1, "{errs:?}");
}

/// Mutually recursive helpers likewise.
#[test]
fn mutually_recursive_helpers_terminate() {
    let errs = check_src(
        "component P { x: f32 }
         fn ping(n: i64) { if n > 0 { pong(n - 1) } }
         fn pong(n: i64) { for p in query<&mut P> { p.x = 1.0 } if n > 0 { ping(n - 1) } }
         system a() stage(Update) { ping(3) }
         system b() stage(Update) { for p in query<&mut P> { p.x = 2.0 } }",
    );
    assert_eq!(errs.len(), 1, "{errs:?}");
}

/// A helper that touches a *different* component is not a conflict. Without this
/// the analysis could pass every test above by simply reporting everything.
#[test]
fn a_helper_touching_another_component_is_not_a_conflict() {
    let errs = check_src(
        "component P { x: f32 }
         component Q { y: f32 }
         fn bump_q() { for q in query<&mut Q> { q.y = 1.0 } }
         system a() stage(Update) { bump_q() }
         system b() stage(Update) { for p in query<&mut P> { p.x = 2.0 } }",
    );
    assert!(errs.is_empty(), "P and Q are independent: {errs:?}");
}
