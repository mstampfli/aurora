# Aurora - how it works

## Theory of operation

Aurora is a compiled language for games: one source file plus its dependencies become native code
through Cranelift, and everything a game needs - windowing, a 3D renderer, physics, audio,
netcode - is a **builtin** rather than a library, so a program calls `r3d_draw` or `net_host` the
way it calls `sin`. There is no interpreter in the shipping path; `aurorac run` compiles and runs.
The builtin surface is generated from one table (`aurora-abi`), which is what keeps the compiler's
idea of a builtin and the runtime's implementation of it from drifting apart.

## The pipeline

Source text to a running program, in order. Each stage is a crate, and each hands the next a
value rather than mutating shared state.

`read_program` (+ `modload` for `mod NAME;`, + manifest `[dependencies]`, + `aurora-std`'s prelude)
-> `aurora-lexer` -> `aurora-parser` (-> `flatten` lowers `mod` blocks to mangled top-level items)
-> `aurora-check` (AST-level static checks) -> `aurora-typeck` (+ `aurora-types` for unification)
-> `aurora-codegen` (Cranelift) -> native code calling `aurora-runtime`.

`aurora-shader` and `aurora-gpu` branch off the same AST for `@vertex`/`@fragment`/`@compute`
functions, lowering them to WGSL instead of machine code.

## Codemap

**Front end**

| Crate | Owns |
|---|---|
| `aurora-span` | source positions. Depends on nothing |
| `aurora-diag` | the user-facing errors, warnings and notes |
| `aurora-lexer` | text to tokens |
| `aurora-ast` | the tree, and the monomorphizer over it |
| `aurora-parser` | tokens to AST; module flattening and file-module loading |
| `aurora-check` | static checks that need no inference |
| `aurora-types` | type representation and the unification engine |
| `aurora-typeck` | bidirectional type checking |
| `aurora-std` | the prelude, written in Aurora, appended to every program |

**Back end**

| Crate | Owns |
|---|---|
| `aurora-abi` | the builtin table: the single source of truth for every builtin's name, signature and symbol |
| `aurora-codegen` | AST to native code via Cranelift |
| `aurora-interp` | a tree-walking interpreter for the computational subset; used by tooling, not by `run` |
| `aurora-exe` | the tiny binary a built program links against |

**Runtime and engine**

| Crate | Owns |
|---|---|
| `aurora-runtime` | every host function compiled code calls: `phys3d_*`, `net_*` (netgame), audio, input, the r3d bridge, and the value stack that holds aggregates too large for a machine frame |
| `aurora-gfx` | CPU rasterization: the 2D framebuffer, text, sprites |
| `aurora-asset` | authored art in memory - geometry, materials, skeletons, clips - and every importer that produces it (glTF/GLB, OBJ, FBX). No GPU dependency, so the offline baker reads a file without linking a renderer and importers are testable with no adapter |
| `aurora-render3d` | the GPU 3D renderer: meshes, materials, skinning, lights, shadows, the scene. Sits on `aurora-asset` and uploads what lands there |
| `aurora-window` | a real window and input (winit + wgpu), and the immediate-mode surface the builtins call |
| `aurora-audio` | synthesis, mixing and decoding |
| `aurora-net` | the reliable-UDP transport, snapshots, interest, lag compensation |
| `aurora-gpu` | live GPU execution of Aurora compute/render shaders |
| `aurora-shader` | Aurora shader functions to WGSL |
| `aurora-slot` | generation-tagged slot storage: stable handles a freed slot cannot answer to |

**Tools**

| Crate | Owns |
|---|---|
| `aurorac` | the driver: `check`, `run`, `build`, `new`, and dependency resolution |
| `aurora-debug` | the native source-level debugger |
| `aurora-lsp` | the language server core |

## Dependency direction

Dependencies point **down** this list, never up: `span` and `diag` at the bottom, then the front
end, then codegen, then the runtime and engine, with `aurorac` on top. `aurora-abi` sits beside
the front end and is depended on by both codegen and the runtime - that is the point of it, and
the reason a builtin cannot exist on one side only.

Each crate's `lib.rs` header states its own job, its place in this graph, and what it must never
do. That is the per-unit contract; there are no separate crate READMEs, because a rustdoc header
is the one place a reader is already looking.

## Invariants

| Invariant | Enforced by |
|---|---|
| Every builtin has exactly one definition of its name, signature and symbol | the `aurora-abi` table; its tests check arity, uniqueness, naming, and that every builtin is documented |
| A builtin that is documented must exist, and one that exists must be documented | `aurora-abi`'s `documentation_debt_only_shrinks` test, against `docs/04-stdlib-and-builtins.md` |
| `check` compiles the same program `run` and `build` do - the user's source, its dependencies and the prelude | all three assemble the program identically in `aurorac`; a call to a missing function used to pass `check` and fail `run` |
| A module may only use modules it declares in its manifest | the compiler: `E0330`, tested in `aurorac`'s CLI suite |
| A freed handle is refused rather than aliased to whatever took its slot | `aurora-slot`'s generation tag, tested at the pixels in `aurora-render3d` |
| A model file's GPU upload is shared between every handle that loads it, and released when the last one goes | `aurora-render3d`'s asset cache, tested by byte counts |
| A rig's rest pose is whatever its skin clusters say, not whatever pose its node transforms were exported in | `aurora-asset`'s FBX importer derives a joint's local from the parent and child bind matrices; `a_rig_whose_node_transforms_are_not_its_bind_pose_still_imports` |
| Skinning a mesh with its own rest pose reproduces that rest pose | the bind-pose self-test in `aurora-asset`'s `fbx_import` suite; this is what catches geometry, bind matrices and joint transforms disagreeing about units or space |
| A skinned model is measured through its bind matrices, never from raw vertices | `Model::bind_pose_bounds`. Skinned geometry stays in the source file's bind space, so `MeshData::bounds` on it would size a collider a hundred times too large for a centimetre export |
| There is one clip sampler, and it lives with the clip format | `Skeleton::sample` in `aurora-asset`; `aurora-render3d` delegates rather than keeping a copy that can drift |
| Under owned movement a body has exactly one simulator | `aurora-runtime::netgame` skips the sim for a networked client, and skips the client's own prediction and reconciliation |
| A function's frame size does not scale with the size of the aggregates it holds | `aurora-codegen`'s `alloc`: at or above `VSTACK_THRESHOLD` an aggregate comes from `aurora-runtime`'s per-thread value stack instead of a Cranelift stack slot |
| A value-stack allocation is made once per call SITE, never once per execution | `aurora-codegen` collects each request during lowering and emits it in the function's `preamble` block, which runs once per activation - so a site inside a loop yields one buffer reused across iterations, exactly as the frame slot it replaces did. Tested by `a_loop_does_not_grow_the_arena_with_its_trip_count` |
| Every value-stack frame is released on every return path | the single `epilogue` block every `return` jumps to, in `compile_body`, `compile_lambda` and `compile_system` - there is one exit, so there is one `vstack_leave` |
| A stack overflow faults cleanly instead of silently corrupting | `enable_probestack` (`STACK_PROBE_FLAGS`, applied to both the JIT and the AOT object) makes a large frame touch guard pages in order rather than jumping past them |

## Key flows

**Compiling** (`aurorac::cmd_run`)
`read_program` -> `collect_deps` -> `aurora_std::with_std` -> `aurora_parser::parse_str` ->
`aurora_check::check` -> `aurora_typeck::check_types` -> `undeclared_module_uses` ->
`aurora_codegen::build` -> run. (`build_object` for `aurorac build`.)

**A drawn frame** (a game's `r3d_*` calls)
`aurora-runtime`'s r3d builtins -> `aurora_window::imm_*` -> `aurora_render3d::Scene` ->
`Renderer3D` -> wgpu.

**A replicated frame** (`net_send_input` / `net_update`)
`Session::send_input` (runs the registered sim, or not, under owned movement) -> `encode_input` ->
the host's `on_server_packet` -> `Session::update` -> `broadcast` -> the client's
`on_client_packet` -> reconcile or adopt.

## Entry points

- `crates/aurorac/src/main.rs` - the driver. `check`, `run`, `build`, `new`.
- `crates/aurora-exe` - what a built program becomes.
- `crates/aurora-lsp`, `crates/aurora-debug` - the editor and debugger front ends.

## Where the other facts live

- **The language**: `docs/01-grammar-and-types.md`. **The netcode model**:
  `docs/02-netcode-replication.md`. **The roadmap**: `docs/03-implementation-roadmap.md`.
- **Every builtin**: `docs/04-stdlib-and-builtins.md`, and `crates/aurora-abi`, which is the table
  that document describes.
- **A crate's contract**: that crate's `lib.rs` header.
