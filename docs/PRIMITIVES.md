# Primitives, and the recipes that have a right order

Reach for these instead of re-deriving them. Each exists because a hand-rolled
version somewhere was wrong, and each is the ONE place its problem is solved -
so a second copy is a bug waiting to drift, not a convenience.

`ARCHITECTURE.md` says what the crates are and what must stay true.
`docs/04-stdlib-and-builtins.md` is the surface a program sees. This is for the
people working ON the compiler and runtime.

## Handles

**`aurora-slot::SlotMap<T>` is the one handle primitive.** Every Aurora resource
a program holds as an `i64` and the runtime later takes back goes through it:
GPU meshes and materials, 2D and 3D physics bodies, JSON nodes, models.

It is generation-tagged, so a freed handle is REFUSED rather than aliased to
whatever took its slot. That is not a nicety - a stale mesh handle that silently
draws whichever asset loaded after it is a bug you find by looking at pixels,
weeks later, in a scene that changed for unrelated reasons.

Do not store resources in a `Vec` and hand out indices.

## Aggregates in generated code

| Want | Use | Never |
|---|---|---|
| Space for an aggregate with call lifetime | `alloc(b, env, slots)` | a bare `create_sized_stack_slot` - `alloc` routes anything past `VSTACK_THRESHOLD` to the value stack so a big struct does not blow the frame |
| Copy an aggregate | `copy_agg(b, env, dst, off, src, cty)` | an unrolled load/store chain; it drops to `memcpy` for large ones |
| Does this expression own its storage? | `produces_fresh_storage(&e.kind)` | assuming - guessing "fresh" wrong ALIASES silently, guessing "place" wrong costs one copy |
| Is this an aggregate at all? | `is_aggregate(&cty)` | `!c.is_scalar()` written out again |

**An aggregate is a VALUE.** `let b = a`, `b = a`, `s.f = a`, `v[i] = a` and a
by-value parameter all copy. `&T` / `&mut T` and a method's `self` are the only
things that share. See §8.0 of `docs/01-grammar-and-types.md`; the eleven tests
that pin it are in `crates/aurora-codegen/tests/value_semantics.rs`.

## Types that hide expressions

**An array type's length is an expression inside a type.** `[T; N]` where `N` is
a const has been forgotten by three separate passes - the initialiser, `ty_to_cty`,
and the module flattener - and every one failed SILENTLY, producing a
zero-length or unsized array whose first read panicked somewhere else entirely.

- Resolving a length: `aurora_typeck::convert::array_len_of`, and codegen's
  `ty_const_len` against the `TY_CONSTS` table.
- Walking a type in a new pass: handle `TypeKind::Array`'s `len`, not just its
  `elem`.
- Locked by `crates/aurora-codegen/tests/const_array_length.rs`, which runs the
  literal, const, arithmetic-const and repeat-count forms, and all of them one
  module deep.

**A pass that walks items must not end in `_ => {}`.** The flattener did, and it
silently skipped const type annotations, `impl` method signatures, trait
signatures, enum payloads and struct field defaults. Explicit arms make a new
`ItemKind` a compile error instead of a silent omission.

## Transforms

**`without_scale(m)` for a socket.** A socket PLACES and ORIENTS; it does not
resize. Divides each basis column by its own length and leaves the translation
column alone.

Do NOT decompose to a quaternion and recompose. A mirrored bone has a negative
determinant, no quaternion can represent a reflection, and the round trip drops
it silently - a prop on the rig's mirrored side comes back inside-out. That
version was written once and reverted for exactly that. Six tests in
`crates/aurora-render3d/src/scene/tests.rs`.

**Measure a skinned model through its bind matrices** (`Model::bind_pose_bounds`),
never from raw vertices: skinned geometry stays in the source file's bind space,
so a centimetre export sizes a collider a hundred times too large.

## Adding a builtin

Six places, in this order. Missing one fails at a different layer each time,
and only the last two are caught automatically.

1. `crates/aurora-render3d/src/scene.rs` (or whichever crate owns the state) -
   the implementation, as a method on `Scene`.
2. `crates/aurora-window/src/imm.rs` - the `with_gfx` wrapper, with the default
   the caller gets when there is no graphics context.
3. `crates/aurora-window/src/lib.rs` - re-export it as `imm_*`.
4. `crates/aurora-runtime/src/lib.rs` - a `#[no_mangle] pub extern "C"` shim.
5. `crates/aurora-abi/src/lib.rs` - a row in the `for_each_builtin!` table.
6. `docs/04-stdlib-and-builtins.md` - a row in the right table.

Step 6 is not optional and is enforced: `documentation_debt_only_shrinks` fails
the build for an undocumented builtin, and `documented_arity_matches_the_table`
fails if the documented signature disagrees with the row. There is an
`UNDOCUMENTED` list; adding to it is the wrong answer almost every time.

The table's other tests are worth knowing before you write a row:
`names_are_unique`, `symbols_follow_the_naming_convention`,
`scalar_rows_take_only_scalars`, `text_rows_pair_each_pointer_with_a_length`,
`str_is_only_a_text_rows_return`.

## Running the tests

    cargo test --workspace --profile release-test -j 4

`release-test` inherits `release` and turns LTO off. `--release` runs the same
tests in ninety-plus minutes, almost all of it linking: every integration test
is its own binary and each links the whole runtime, so thin LTO runs once per
binary over that graph. Same `opt-level`, so the timing-sensitive suites
(`frame_clock`, `fixed_stage`) still measure an optimized build.

`-j 4` is not arbitrary either - the default is one rustc per core and makes the
machine unusable while it runs.

## Rules that are not functions

- **A frame ends at `input_step`**, where the input edge snapshot rolls and the
  frame's delta is spent. `frame_dt` is measured once per frame and reused;
  reading it is not destructive, because `run_systems` reads it too.
- **A `stage(FixedUpdate)` system advances by simulated time**, never by frame
  count, and at most `MAX_CATCHUP_STEPS` run in one frame.
- **`r3d_material_texture` binds BY NAME.** Ask the mesh what its materials are
  called (`r3d_material_count` / `r3d_material_name`) rather than guessing; a
  wrong name renders white and says nothing.
- **A whole-sheet UV is not a bug.** `0..1` is one repeat against a TILING
  texture and the entire atlas against an ATLAS. Which is correct is a fact
  about the material, not the mesh.
