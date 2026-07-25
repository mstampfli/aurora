# Aurora — Implementation Roadmap

> Decision (2026-06-14): build a **real, production-grade self-hosting toolchain**,
> written in **Rust**, **frontend-first**. Goal is a language you can actually ship
> games in (native codegen + GPU + netcode runtime), not a toy.

This doc is the build plan. Specs `01` (grammar/types) and `02` (netcode) are the
normative references each phase implements against.

---

## Guiding principles

- **Frontend before backend.** A fully type/region/borrow-checked IR is the
  contract every later phase consumes. We don't write codegen until `.aur` files
  pass/fail correctly.
- **Each crate testable in isolation.** Snapshot tests at every phase boundary
  (token stream, AST, resolved IR, typed IR, diagnostics).
- **Diagnostics are a first-class output, not an afterthought.** Elm/Rust-quality
  errors from day one — they're what makes the language "easy to use" and what we
  rely on while dogfooding.
- **Self-hosting is the long-term goal**, but v0 is in Rust; we revisit bootstrap
  only after the native backend is stable.

---

## Cargo workspace layout (planned)

```
aurora/
  Cargo.toml                 # workspace
  crates/
    aurora-span/             # SourceId, Span, source map, line/col mapping
    aurora-diag/             # Diagnostic, severity, labels, pretty renderer
    aurora-lexer/            # logos-based tokenizer -> Token stream
    aurora-ast/              # AST node types + arena/interner
    aurora-parser/           # recursive-descent + Pratt expr parser (EBNF from spec 01)
    aurora-hir/              # resolved IR: names bound, items registered
    aurora-resolve/          # name resolution, module/use graph, ComponentId/SystemId
    aurora-types/            # type representation, inference, trait solving
    aurora-check/            # region checker + borrow checker + ECS scheduler check
    aurora-driver/           # ties phases together; query/incremental layer (salsa?)
    aurorac/                 # the `aurorac` CLI binary
  tests/                     # end-to-end .aur fixtures (pass/ + fail/)
  docs/                      # specs 01/02/03 (this file)
```

Backend crates (`aurora-mir`, `aurora-codegen-cranelift`, `aurora-shader`,
`aurora-runtime`, `aurora-net`) are added in Phases B/C.

---

## Phase A — Frontend (current target)

Deliverable: `aurorac check foo.aur` fully validates a program against spec 01 and
emits precise diagnostics. No execution.

### A0. Scaffolding
- Workspace, CI, `aurora-span` + `aurora-diag` with a snapshot-friendly renderer.
- Test harness: `tests/pass/*.aur` must check clean; `tests/fail/*.aur` must produce
  an expected diagnostic (compared via `.stderr` snapshots, `insta`).

### A1. Lexer (`aurora-lexer`)
- `logos` tokenizer covering spec §2: idents/keywords, all literal forms + suffixes,
  sigils (`~ & &mut # @ -> => .. ..= |>`), comments (incl. nested block), swizzle
  handled at parse time (lexed as ident, classified later).
- Emits tokens with spans; recovers on bad chars (error token, keep going).

### A2. Parser (`aurora-parser` + `aurora-ast`)
- Recursive descent for items/statements; **Pratt/precedence-climbing** for the
  expression grammar (spec §5.2 operator table is the source of truth).
- Implements the conservative ASI rule (spec §3.3).
- **Error recovery**: synchronize on `}`/`;`/item-keyword so one syntax error
  doesn't cascade.
- Parses *all* constructs incl. `component`/`system`/`query`/`pipeline`/shader attrs/
  `#region` exprs/`comptime`. Attributes attach to the following item/field.

### A3. Resolution (`aurora-resolve` -> `aurora-hir`)
- Build module/`use` graph; resolve every `Path` to a definition.
- Assign stable `ComponentId`/`SystemId`/`RpcId`; register replicated components.
- Lower AST -> HIR (names bound, sugar desugared: `|>` pipe, field shorthand,
  `for` over query, struct-update `..base`).
- Diagnostics: unresolved name, duplicate def, visibility violation.

### A4. Type system (`aurora-types`)
- Type representation incl. `~T`, `rc<T>`, `&/&mut`, regions, generics, `Handle<T>`,
  builtin vec/mat algebra.
- **Bidirectional inference** (spec §7.2): synthesize for `let x = e`, check against
  annotation/return type otherwise. Numeric literal default resolution.
- Vector/swizzle typing (§7.5), coercion rules (§7.4 — explicit numeric widening,
  implicit reborrow/borrow-of-owned).
- Trait solving: monomorphic resolution + `dyn` fat pointers; `@derive` hooks.
- **GPU mode checking** (§7.6): functions reachable from `@vertex/@fragment/@compute`
  checked against the restricted type env; binding inference from free vars.

### A5. Region + borrow + ECS checks (`aurora-check`)
- **Region escape analysis** (§8.2): region tags flow through assignments/returns;
  error on `R' ⊑ R` violations. Region-generic fns (`'r`) supported.
- **Borrow checker** scoped to owned/region memory only (§8.3): aliasing/mutation
  discipline; move/use-after-move for `~T`.
- **ECS scheduler check** (§6.2): compute each system's read/write access sets from
  its `query<...>` and params; verify intra-query disjointness; build the schedule
  graph; **error on unordered write-write conflicts**. This is the data-race-freedom
  theorem, enforced.

### A6. Driver + CLI (`aurora-driver`, `aurorac`)
- Wire phases; consider `salsa` for incremental recompute (sets up hot-reload later).
- `aurorac check <file>` (full pipeline), `aurorac parse`/`aurorac ast`/`aurorac hir`
  debug dumps for development.

**Phase A exit criteria:** the worked examples in spec 01 §9 and spec 02 §11 type-,
region-, and borrow-check correctly; a curated `tests/fail/` suite produces the
right diagnostics (conflicting systems, region escapes, GPU-mode violations,
use-after-move, schema-shape errors).

---

## Phase B — Runtime & semantics (after A)

- `aurora-runtime`: ECS world (archetype SoA storage), the parallel scheduler that
  consumes Phase A's access sets, region allocators (frame/level/perm bump arenas).
- Tree-walking **interpreter** over typed IR -> programs *run*. This doubles as the
  dev/hot-reload execution mode in the final design.
- Minimal builtin std: math, `App`, window/input stub, asset loading.

## Phase C — Backends & full engine (after B)

- `aurora-mir` -> `aurora-codegen-cranelift` (native AOT), LLVM later for `-O`.
- `aurora-shader`: lower GPU-mode IR -> SPIR-V/WGSL; bind-group layout generation.
- `aurora-runtime` graphics: wgpu-style renderer (PBR, shadows, AO, sprite/text).
- `aurora-net`: the generated replication runtime from spec 02 (channels, delta
  snapshots, interpolation, prediction/reconciliation, lag comp, interest mgmt).
- Hot reload: swap function bodies via the incremental driver; preserve `#perm`/
  `#level` state.

---

## Progress log

- **A0 + A1 — DONE.** Workspace builds; `aurora-span`, `aurora-diag` (rustc-style
  caret renderer), hand-rolled `aurora-lexer` (nestable comments, all
  operators/suffixes/escapes, error recovery), `aurorac lex`.
- **A2 — DONE.** `aurora-ast` (full tree) + `aurora-parser` (recursive-descent
  items + Pratt expressions, struct-literal restriction in conditions, ASI for
  statements & declaration bodies, error recovery + progress guards). `aurorac
  parse`. Both spec examples (`spinner.aur`, `netcode.aur`) parse clean.
- **A5 (partial, pulled forward) — DONE.** `aurora-check`: duplicate-definition
  detection, intra-query aliasing, and the **ECS scheduler data-race-freedom
  theorem** (§6.2) — unordered same-stage systems with conflicting access sets
  error with a fix suggestion. `aurorac check`.
- Contextual-keyword decisions made along the way: `in`, `spawn`, `despawn` are
  NOT reserved (used as binding/method names in examples); added `?` and `const`
  tokens. 44 tests passing.

- **A3 (resolution-lite) — DONE.** `aurora-check::resolve`: query terms must name
  components (not structs/unknowns), `after`/`before` must name real systems;
  builtins+imports accepted (single-file scope).
- **A4 (type inference) — DONE (subset).** `aurora-types` (union-find unifier) +
  `aurora-typeck` (bidirectional checker). Lenient prelude strategy: unresolved
  externs/methods/fields → absorbing `Error` type, so real mismatches are caught
  (`let x: bool = 1`, wrong return type, `i32 + f32`, bad args, struct fields, bad
  conditions) without false-positiving on graphics/engine builtins. Both spec
  examples type-check clean.
- **Phase B (interpreter core) — DONE.** `aurora-interp`: tree-walking evaluator
  for the computational subset (fns, recursion, control flow, `for` over ranges/
  arrays, structs, tuples, arrays, assignment/places, `print`). `aurorac run`
  executes `examples/compute.aur`. Caught + fixed a real bug: path parser ate `<`
  as generics, breaking comparison operators in expressions (now turbofish-only in
  expr position).

**84 tests green.** Pipeline: `aurorac lex|parse|check|run`.

- **Phase B (ECS runtime) — DONE (interpreter).** `spawn`/`despawn`/`query`
  iteration with `&mut` write-back + filters (`!T`/`+T`/`Entity`/`where`),
  `run_systems`/`entity_count` harness builtins, `match`. Runs `examples/ecs.aur`.
- **A5 (ownership/move checking) — DONE.** `aurora-check::moves`: use-after-move
  for `~T` (E0400), move-in-loop (E0401), consuming-vs-borrow contexts,
  reassignment revival, branch merging. Only `~T` is tracked, so `~`-free code is
  unaffected. Region *escape* checking still pending (needs real region inference
  with region-param signatures — a naive version is unsound, so deferred).

**99 tests, 0 warnings.** Frontend pipeline: lex → parse → resolve → type-check →
ECS-safety → move-check; interpreter runs scripts + ECS + match.

- **Netcode serialization (spec 02 §2–3, §8.3) — DONE (schema-driven).**
  `aurora-net::CompSchema` derives a wire format from a `@replicated component`:
  honors `@quantize` (f32→2-byte fixed point) and `@noreplicate` (excluded);
  provides `serialize`/`deserialize`, dirty-mask `delta`/`apply_delta`, and a
  `schema_hash` for the handshake version check. This is what the real compiler
  would *generate* per component, done at runtime over interpreter values.

- **Prediction & reconciliation (spec §6) — DONE (harness).** `aurora-net::Predictor`
  runs an Aurora `fn step(state, input) -> state` through the interpreter for
  client prediction; on an authoritative snapshot it reconciles by snapping +
  replaying unacked inputs through the *same* function. Tests show: identical
  function on both sides ⇒ zero corrections (the drift-proofing of §6.3); a
  divergent server state ⇒ snap + replay. Stale/duplicate acks ignored.

**108 tests, 0 warnings, 11 crates.**

## Next action

(1) **Region escape checking** — last memory-safety check; needs real region
inference (`'r` params, per-expression region flow). (2) **Phase C** — Cranelift
native codegen, SPIR-V shader lowering, wgpu renderer, and the live netcode UDP
*transport* (channels, ack/nack, interest management) wrapping the existing
serialization + prediction. (3) Interpreter breadth: closures, method/UFCS
dispatch, enum runtime values, generics.
