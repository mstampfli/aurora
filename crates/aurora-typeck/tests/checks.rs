//! Type-checker regression tests for the call-arity and return-type checks.

use aurora_parser::parse_str;
use aurora_typeck::check_types;

fn errors(src: &str) -> Vec<String> {
    let (module, pdiags) = parse_str(src);
    assert!(!pdiags.iter().any(|d| d.is_error()), "parse error in test source");
    check_types(&module).iter().filter(|d| d.is_error()).map(|d| d.message.clone()).collect()
}

#[test]
fn arg_count_mismatch_is_reported() {
    let errs = errors("fn add(a: i64, b: i64) -> i64 { a + b }\nfn main() { let x = add(1) }");
    assert!(errs.iter().any(|e| e.contains("expects 2 argument")), "got: {errs:?}");

    let errs = errors("fn id(a: i64) -> i64 { a }\nfn main() { let x = id(1, 2, 3) }");
    assert!(errs.iter().any(|e| e.contains("expects 1 argument")), "got: {errs:?}");
}

#[test]
fn correct_arg_count_is_accepted() {
    let errs = errors("fn add(a: i64, b: i64) -> i64 { a + b }\nfn main() { let x = add(1, 2) }");
    assert!(errs.is_empty(), "unexpected errors: {errs:?}");
}

#[test]
fn return_type_mismatch_is_reported() {
    let errs = errors("fn f() -> i64 { return true }");
    assert!(!errs.is_empty(), "expected a return-type error, got none");

    // An early return of the right type is fine.
    let errs = errors("fn f(x: i64) -> i64 { if x > 0 { return 1 } 2 }");
    assert!(errs.is_empty(), "unexpected errors: {errs:?}");
}
