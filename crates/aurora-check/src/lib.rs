//! Static checks that operate directly on the AST (no full type inference yet):
//!
//! 1. **Duplicate definitions** — two items sharing a name in one namespace.
//! 2. **Intra-query aliasing** — a single `query<...>` that borrows the same
//!    component both mutably and (mutably or immutably) again.
//! 3. **ECS scheduler soundness** (grammar spec §6.2) — within a stage, two
//!    systems whose access sets conflict must be explicitly ordered, otherwise
//!    their parallel execution would race. This is the data-race-freedom
//!    theorem, enforced at compile time.
//!
//! These are the checks that don't require resolving expression types, so they
//! land before the full type checker. `query` access sets drive (2) and (3).

mod ecs;
mod moves;
mod regions;
mod resolve;
mod symbols;
mod walk;

use aurora_ast::Module;
use aurora_diag::Diagnostic;

/// Run all AST-level checks and return the diagnostics found.
pub fn check(module: &Module) -> Vec<Diagnostic> {
    let mut diags = Vec::new();
    symbols::check_duplicates(module, &mut diags);
    resolve::resolve(module, &mut diags);
    ecs::check_queries_and_schedule(module, &mut diags);
    moves::check_moves(module, &mut diags);
    regions::check_regions(module, &mut diags);
    diags
}
