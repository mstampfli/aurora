# Aurora

A from-scratch language for game development. It compiles to native machine code
(Cranelift JIT *and* standalone executables), treats ECS, graphics, and netcode as
language and runtime features rather than libraries, lowers shaders to real WGSL
that runs on the GPU, and proves at compile time that parallel systems cannot
data-race.

Aurora is opinionated about a small set of things:

- **The compiler is the engine.** `component`, `system`, and `query` are syntax,
  not a library. `spawn`/`despawn`/`run_systems` are builtins. So are graphics, a
  real-time window, keyboard/mouse input, audio, networking, physics, and
  pathfinding. You write a game, not glue code.
- **Parallel systems are proven safe, not hoped safe.** `run_systems()` groups
  systems into layers by access conflict and ordering, then runs each layer
  concurrently over one shared world. The race-freedom proof (spec section 6.2) is
  what makes that sound: two systems that touch the same component must be ordered
  or the compiler rejects them.
- **One language for CPU and GPU.** A shader is an Aurora function with `@vertex` /
  `@fragment` / `@compute`. The same `Vec3`/`Mat4`/`Color` types work on both
  sides; the shader crate lowers them to valid WGSL and the GPU crate runs it.
- **Memory is regions, not a GC.** Values live in `#frame`, `#level`, or `#perm`
  arenas. `#frame` data is reclaimed in O(1) by `frame_reset()` at the end of a
  tick instead of leaking, and the checker rejects storing a short-lived value
  where a longer-lived one is required.
- **Safe where it is free.** Data-race freedom, move/ownership checking, region
  escape, and bounds-checked array indexing are all compile-time or cheap-runtime,
  with no borrow-checker ceremony for the common case.

## Hello, Aurora

```aurora
fn main() {
    println("hello, world")
}
```

```sh
aurorac run hello.aur
```

## A taste

Race-free ECS. `render` reads `Position` while `integrate` writes it, so the
ordering is required or `aurorac check` reports `E0202`:

```aurora
component Position { x: f64, y: f64 }
component Velocity { x: f64, y: f64 }

system integrate() stage(Update) {
    for (p, v) in query<&mut Position, &Velocity> { p.x += v.x; p.y += v.y }
}
system render() stage(Update) after(integrate) {
    for (p, s) in query<&Position, &Sprite> { triangle(/* ... */) }
}

fn main() {
    spawn(Position { x: 0.0, y: 0.0 }, Velocity { x: 1.0, y: 0.0 })
    run_systems()
}
```

A shader is just an Aurora function. `aurorac wgsl` lowers it to WGSL and
`aurorac gpu` runs it on real hardware:

```aurora
@vertex fn vs(vin: VsIn) -> VsOut {
    VsOut { clip: view_proj * vec4(vin.pos, 1.0), uv: vin.uv }
}
```

Generics monomorphize per concrete type (functions, structs, nested types, and
enums used at multiple instantiations):

```aurora
enum Opt<T> { Some(T), None }

fn unwrap_or<T>(o: Opt<T>, d: T) -> T {
    match o { Opt::Some(x) => x, Opt::None => d }
}
```

Call into any C or Rust C-ABI function with `@extern`:

```aurora
@extern fn hypot(x: f64, y: f64) -> f64     // binds a C-ABI symbol (here, libm)

fn main() {
    println(hypot(3.0, 4.0))                 // 5.0
    load_image("hero.png")                   // PNG/JPEG into the framebuffer
    load_font("C:/Windows/Fonts/arial.ttf")
    draw_text(8, 8, "Score: 1234", 28, rgb(255, 255, 255))
}
```

## Project layout

```
crates/
  aurora-span      spans + source maps
  aurora-diag      diagnostics + caret renderer
  aurora-lexer     hand-rolled lexer (nested comments, newline-aware ASI)
  aurora-ast       the syntax tree
  aurora-parser    recursive-descent + Pratt expressions
  aurora-types     type representation + union-find unifier
  aurora-typeck    bidirectional type checker (generics, traits, enums)
  aurora-check     ECS scheduler safety, move checking, region escape, resolution
  aurora-interp    tree-walking interpreter (reference path; compiled path is primary)
  aurora-codegen   Cranelift backend: JIT + AOT object emission (whole language)
  aurora-runtime   native host functions (print/graphics/ECS/physics/nav) for compiled code
  aurora-exe       link target + entry shim for standalone .exe output
  aurora-gfx       CPU rasterizer (framebuffer, triangles, PPM)
  aurora-shader    Aurora @vertex/@fragment/@compute -> WGSL
  aurora-gpu       live GPU execution via wgpu (headless compute + render)
  aurora-window    real-time winit + wgpu window with keyboard/mouse input
  aurora-audio     synthesis (oscillators/ADSR/mixing) + cpal playback
  aurora-debug     native source-level debugger (machine-code instrumentation)
  aurora-net       netcode: replication, prediction, lag-comp, interest, reliable UDP
  aurora-std       standard-library prelude (Aurora source, auto-included)
  aurora-lsp       Language Server (diagnostics + completion over stdio)
  aurorac          the CLI driver
docs/
  01-grammar-and-types.md      full EBNF + type system
  02-netcode-replication.md    replication model
  03-implementation-roadmap.md phase-by-phase build log
  04-stdlib-and-builtins.md    practical reference: every builtin + the stdlib prelude
examples/                      runnable .aur programs (start with showcase.aur)
```

The pipeline is `lex -> parse -> resolve -> typecheck -> ECS-safety -> move-check
-> region-check`, then JIT-run or emit a standalone native executable.

## Building

```sh
cargo build --workspace      # builds the toolchain (Cranelift takes a moment first time)
cargo test  --workspace      # 313 tests
```

## CLI

```sh
aurorac new    mygame                       # scaffold a project (aurora.toml + src/)
aurorac run    examples/showcase.aur        # compile main to native code and run (JIT)
aurorac build  examples/native.aur -o game.exe   # standalone optimized native executable
aurorac check  examples/game.aur            # type + ECS + move + region checks only
aurorac debug  examples/native.aur --break 9     # native debugger (breakpoints, step, locals)
aurorac gpu    examples/gpu_shader.aur -o out.ppm # run a shader on the real GPU
aurorac window                              # open a live real-time window
aurorac sound                               # synthesize and play audio
aurora-lsp                                  # language server over stdio
```

## Spec and docs

The design docs in [`docs/`](docs/) are the authoritative description of the
grammar, type system, region rules, netcode model, and every builtin. Start with
[`docs/04-stdlib-and-builtins.md`](docs/04-stdlib-and-builtins.md) for the
practical reference (the CLI, ECS, graphics, audio, physics, pathfinding, FFI, and
the auto-included standard library).

## Tests

```sh
cargo test --workspace       # 313 tests across 22 crates, 0 warnings
```

Every capability above is backed by passing tests and a runnable example in
[`examples/`](examples/).

## Status

Real and working: a full compiler toolchain that JITs and AOT-compiles the whole
language, with ECS, a CPU rasterizer, live GPU shaders, a real-time window, audio,
reliable-UDP netcode, Rapier 2D physics, A* pathfinding, an asset pipeline
(PNG/JPEG/TTF/WAV), C and Rust FFI, a native debugger and profiler, and an LSP.

What is intentionally not here yet, but is on the road:

- **An LLVM backend.** Codegen is Cranelift only (JIT + AOT, `opt_level=speed` for
  builds). Runtime speed is good, not maximal, and there is no autovectorization.
- **A 3D mesh pipeline.** The shader path already speaks `Vec3`/`Mat4`/`vec4`, but
  there is no model loader (glTF/OBJ) or vertex/index-buffer mesh draw yet; physics
  is 2D (Rapier 2D).
- **Editor-integrated debugger UI** and a **central package registry** (path and
  git dependencies do work today).
- **Battle-testing.** This is a capable foundation, not yet a shipped-game-proven
  production engine.

## License

MIT.
