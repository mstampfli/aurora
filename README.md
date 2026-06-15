# Aurora

A from-scratch programming language designed for game development: **fast**
(compiles to native machine code - JIT *and* standalone executables),
**batteries-included** (ECS, graphics, and netcode are language/runtime features),
**GPU-native** (shaders are Aurora functions that run on real hardware), and
**safe where it's free** (the compiler proves systems can't data-race).

> Status: a working compiler toolchain. **22 crates, 313 tests, 0 warnings.**
> Every capability below is backed by passing tests and runnable examples.
> Design specs live in [`docs/`](docs/).

Aurora programs drive graphics, a real-time window, keyboard + mouse input,
audio (mixed/non-blocking), and **GPU fragment shaders** through language
builtins - `window_open`/`window_present`/`key_down`/`mouse_x`/`play_sound`/
`gpu_render` (see [`examples/game_window.aur`](examples/game_window.aur)).
**Strings are first-class** (`let s = "a" + str(n)`, `len`, `char_at`, `substr`,
`starts_with`, `==`, params/returns), and dynamic memory is **arena-managed** -
`#frame` allocations are reclaimed in O(1) by `frame_reset()` rather than leaked.
**Networking is a builtin** (`net_bind`/`net_connect`/`net_send`/`net_recv` over
reliable UDP). **Generics** are monomorphized - `fn id<T>`, `struct Stack<T>`, and
a generic `List<T>` collection all specialize per concrete type. A standard
library (math, easing, string helpers, `Vec2`, generic `List<T>`) is
auto-included, and code is organized with **modules** (`mod m { … }` →
`m::item`) and **dependencies** (`[dependencies]` in `aurora.toml`, path or git,
resolved transitively with an `aurora.lock`).

```sh
aurorac new    mygame                        # scaffold a project (aurora.toml + src/)
aurorac run    examples/showcase.aur         # compile main to native code & run
aurorac build  examples/native.aur -o game.exe   # → a standalone native .exe
aurorac debug  examples/native.aur --break 9 # native debugger (breakpoints, step, locals)
aurorac gpu    examples/gpu_shader.aur -o out.ppm  # run a shader on the real GPU
aurorac window                               # open a live real-time window (interactive)
aurorac sound                                # synthesize + play audio
aurorac check  examples/game.aur             # type-check + safety checks
aurora-lsp                                   # language server (editor diagnostics over stdio)
```

## What makes Aurora different

| Pillar | How | Where |
|---|---|---|
| **Fast (JIT)** | Cranelift compiles the *whole* language (scalars, structs/enums/arrays, methods, closures, ECS) to machine code | `aurora-codegen` |
| **Fast (AOT)** | `aurorac build` emits a native object and links a standalone `.exe` - no interpreter, no runtime VM | `aurora-codegen`, `aurora-runtime`, `aurora-exe` |
| **ECS is the language** | `component`/`system`/`query` are syntax; the compiler proves systems race-free (§6.2) | `aurora-parser`, `aurora-check` |
| **Parallel scheduler** | `run_systems()` layers systems by access conflict + ordering and runs each layer of independent systems concurrently over a shared world - the §6.2 race-freedom proof is what makes it safe | `aurora-ast` (`parallel_layers`), `aurora-codegen`, `aurora-runtime` |
| **Built-in graphics** | Aurora code drives a CPU rasterizer; shaders are Aurora functions lowered to WGSL | `aurora-gfx`, `aurora-shader` |
| **Live GPU** | lowered WGSL runs headless on the real GPU (compute + render), pixels read back | `aurora-gpu` (wgpu) |
| **Real-time window** | winit + wgpu window presents a framebuffer each frame with keyboard/mouse input | `aurora-window` |
| **Built-in audio** | synthesis (oscillators, ADSR, note pitches, mixing) + real output via cpal | `aurora-audio` |
| **FFI (C + Rust)** | `@extern fn hypot(x: f64, y: f64) -> f64` binds a C-ABI symbol - any C library or Rust `#[no_mangle] extern "C"` function. Resolves at link time (AOT) or against registered symbols (JIT) | `aurora-codegen` |
| **Asset loading** | `load_image(path)` decodes PNG/JPEG into the framebuffer (via the `image` crate), beyond the built-in PPM loader | `aurora-runtime` |
| **Physics (Rapier)** | `phys_init`/`phys_add`/`phys_step`/`phys_x`/`phys_set_vel` - real 2D rigid-body simulation (gravity, colliders, continuous collision) via the Rapier crate | `aurora-runtime` |
| **Pathfinding (A\*)** | `nav_init`/`nav_wall`/`nav_find`/`nav_x`/`nav_y` - weighted grid A* via the `pathfinding` crate | `aurora-runtime` |
| **Text rendering** | `load_font(path)` + `draw_text(x, y, str, px, color)` rasterize TrueType/OpenType glyphs into the framebuffer (via `fontdue`), alpha-blended | `aurora-runtime` |
| **Native debugger** | machine-code instrumentation (not an interpreter) - breakpoints, step, and locals incl. floats + struct fields, at native speed | `aurora-debug` |
| **Profiler** | `aurorac profile` instruments the native code for per-function call counts + wall-clock time | `aurora-runtime` |
| **Assets + scenes** | `load_ppm` loads images into the framebuffer; `scene_save`/`scene_load` persist the whole ECS world; `aurorac watch` hot-reloads on change | `aurora-runtime` |
| **Standard library** | a prelude auto-included in every program: math/easing, `Vec2`/`Vec3` + vector math, generic `List<T>`, **collision** (`Rect`/circle), **color** packing + lerp, **sprite-sheet** + animation helpers, **particles**, **grid BFS pathfinding** (`Grid`), **2D AABB physics** (`Body` integrate + collision resolve), immediate-mode **UI widgets** (`ui_button`/`ui_slider`/`ui_label`/`fill_rect`), integer **bitwise** builtins; packages depend on each other by path or git | `aurora-std`, `aurorac` |
| **Built-in netcode** | serialization, prediction/reconciliation, lag-comp, interest, **real UDP transport** (now exposed as language builtins), fixed-point determinism | `aurora-net` |
| **Arena memory** | per-region bump allocator; `#frame` data reclaimed O(1), not leaked | `aurora-runtime` |
| **Data parallelism** | `par_for(out, closure)` runs the closure across OS threads, each writing disjoint output slots | `aurora-runtime`, `aurora-codegen` |
| **Generics + traits + enums** | functions, structs (incl. **nested** like `Box<Box<i64>>`), collections, and enums (`Result<T,E>`, **multiple instantiations**) monomorphize per type; `trait`s give static dispatch with enforced bounds (`fn f<T: Speaker>`) and dynamic dispatch (`dyn Speaker`) | `aurora-ast`, `aurora-typeck`, `aurora-codegen` |
| **Error handling** | `Result<T,E>`/`Option<T>` + `match`, with the `?` operator for early-return error propagation | `aurora-codegen` |
| **Safe where it's free** | data-race (§6.2), ownership/move (§8.1/3), region-escape through bindings, return values, *and* **inferred region-parameterized signatures** - a call passing a `#frame` value to a function that stores it in `#perm` is rejected; optional **`#region` type annotations** (`fn keep(t: #perm Thing)`) declare the contract across bodiless boundaries (`@extern`, trait sigs) (§8.2); **array indexing is bounds-checked** (out-of-range/negative traps with a clear panic) | `aurora-check`, `aurora-codegen` |
| **Editor tooling** | a Language Server (LSP) streams diagnostics AND completion (keywords, builtins, symbols) to any editor | `aurora-lsp` |

### "Safe where it's free", concretely

```aurora
system integrate() stage(Update) { for (p, v) in query<&mut Position, &Velocity> { p.x += v.x } }
system render()    stage(Update) after(integrate) { for (p, s) in query<&Position, &Sprite> { triangle(...) } }
```

`render` reads `Position` while `integrate` writes it. Without `after(integrate)`,
`aurorac check` reports `E0202: systems conflict on component 'Position' but are
not ordered` - the data-race-freedom theorem, enforced at compile time. Add the
ordering and it checks clean.

### One language for CPU and GPU

```aurora
@vertex fn vs(vin: VsIn) -> VsOut { VsOut { clip: view_proj * vec4(vin.pos, 1.0), uv: vin.uv } }
```

`aurorac wgsl` lowers this to valid WGSL - the same `Vec3`/`Mat4` types work on
both sides.

### Using C/Rust libraries and assets

```aurora
@extern fn hypot(x: f64, y: f64) -> f64     // bind any C-ABI symbol

fn main() {
    println(hypot(3.0, 4.0))                 // 5.0 - calls libm

    load_image("hero.png")                   // PNG/JPEG -> framebuffer
    load_font("C:/Windows/Fonts/arial.ttf")
    draw_text(8, 8, "Score: 1234", 28, rgb(255, 255, 255))
}
```

`@extern` declarations bind external **C** symbols (and **Rust** functions
exported as `#[no_mangle] extern "C"`), resolved at link time for `aurorac build`
and against registered symbols for `aurorac run`. `load_image` (via the `image`
crate) and `load_font`/`draw_text` (via `fontdue`) give a real asset pipeline and
on-screen text.

## Architecture (22 crates)

```
aurora-span      spans + source maps
aurora-diag      diagnostics + caret renderer
aurora-lexer     hand-rolled lexer (nested comments, newline-aware ASI)
aurora-ast       the syntax tree
aurora-parser    recursive-descent + Pratt expressions
aurora-types     type representation + union-find unifier
aurora-typeck    bidirectional type checker (lenient prelude, generics)
aurora-check     ECS scheduler safety, move checking, region escape, resolution
aurora-interp    tree-walking interpreter (dev/reference; the compiled path is primary)
aurora-codegen   Cranelift backend - JIT + AOT object emission (whole language)
aurora-runtime   native host functions (print/graphics/ECS) linked into executables
aurora-exe       link target + entry shim for standalone `.exe` output
aurora-gfx       CPU rasterizer (framebuffer, triangles, PPM)
aurora-shader    Aurora @vertex/@fragment/@compute -> WGSL
aurora-gpu       live GPU execution via wgpu (headless compute + render)
aurora-window    real-time winit+wgpu window + keyboard/mouse input
aurora-audio     synthesis (oscillators/ADSR/mixing) + cpal playback
aurora-runtime   native host functions (print/graphics/ECS/debug) for compiled code
aurora-debug     native source-level debugger (machine-code instrumentation)
aurora-std       standard-library prelude (Aurora source, auto-included)
aurora-net       netcode: serialization, prediction, lag-comp, interest, real UDP transport, fixed-point
aurora-lsp       Language Server (LSP diagnostics over stdio)
aurorac          the CLI driver
```

Pipeline: `lex → parse → resolve → typecheck → ECS-safety → move-check → region-check`,
then JIT-run or emit a standalone native executable.

## Examples

| File | Shows |
|---|---|
| `compute.aur` | functions, recursion, control flow, structs, arrays, tuples |
| `showcase.aur` | enums, `impl` methods, closures, the pipe operator, `match` |
| `ecs.aur` | spawn/query/systems with `&mut` write-back |
| `parallel.aur` | independent systems run concurrently; a conflicting one is `after`-ordered into a later layer |
| `generic_enum.aur` | one generic enum used at two instantiations (`Opt<i64>` + `Opt<f64>`), resolved per construction/match |
| `ffi.aur` | calling C library functions directly with `@extern` |
| `gamedev.aur` | the game-dev stdlib: vectors, collision, easing, color, particles, UI, pathfinding |
| `physics.aur` | library-backed Rapier 2D physics + `pathfinding`-crate A*, via builtins |
| `game.aur` | ECS + scheduler ordering + graphics, all in one program |
| `draw.aur` | Aurora code driving the builtin rasterizer |
| `shader.aur` | shaders that lower to WGSL |
| `spinner.aur`, `netcode.aur` | the grammar/netcode spec examples |

## Building

```sh
cargo build --workspace      # builds everything (Cranelift takes a moment first time)
cargo test  --workspace      # 313 tests
```

## Known limitations (honest list)

The live GPU renderer, real UDP transport, audio, the real-time window, and the
parallel system scheduler that earlier drafts of this list called "not yet" are
all done now (see the table above). What genuinely remains:

- **Backend / performance**: Cranelift only (JIT + AOT; release builds use
  `opt_level=speed`). No LLVM backend or autovectorization yet - a designed-but-
  deferred performance item; runtime speed is good, not maximal.
- **Young & evolving**: a capable foundation with a CLI debugger/profiler and an
  LSP, but no editor-integrated debugger UI or central package registry (path/git
  dependencies do work), and not yet battle-tested in a shipped game.

## Design docs

- [`docs/01-grammar-and-types.md`](docs/01-grammar-and-types.md) - full EBNF + type system
- [`docs/02-netcode-replication.md`](docs/02-netcode-replication.md) - replication model
- [`docs/03-implementation-roadmap.md`](docs/03-implementation-roadmap.md) - phase-by-phase build log
- [`docs/04-stdlib-and-builtins.md`](docs/04-stdlib-and-builtins.md) - **practical reference**: every builtin (graphics/audio/physics/pathfinding/FFI) + the stdlib prelude
