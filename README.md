# Aurora

A from-scratch language for game development. It compiles to native machine code
(Cranelift JIT *and* standalone executables), treats ECS, graphics, and netcode as
language and runtime features rather than libraries, lowers shaders to real WGSL
that runs on the GPU, and proves at compile time that parallel systems cannot
data-race.

Aurora is opinionated about a small set of things:

- **The compiler is the engine.** `component`, `system`, and `query` are syntax,
  not a library. `spawn`/`despawn`/`run_systems` are builtins. So are 2D and 3D
  graphics (a GPU renderer with depth, lighting, textures, and skeletal
  animation), glTF/OBJ model loading, a real-time window, keyboard/mouse input,
  audio, networking, 2D/3D physics, and 2D/3D pathfinding. You write a game, not
  glue code.
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

A GPU 3D scene: load a model, set a camera and light, and draw each frame. The
renderer does the depth buffer, lighting, textures, and skeletal animation:

```aurora
window_open(960, 540)
let hero = r3d_load_model("hero.glb")   // glTF/GLB/OBJ; or r3d_make_box(r,g,b)
r3d_anim_play(hero, 0, 1, 1.0)          // clip 0, looping, 1x speed

while r3d_present() {
    r3d_begin()
    r3d_anim_update(hero, 0.016)
    r3d_camera(0.0, 2.0, 6.0,  0.0, 1.0, 0.0,  70.0)   // eye, target, fov
    r3d_light(0.4, 1.0, 0.3,  1.0, 1.0, 1.0,  0.3)     // dir, color, ambient
    r3d_draw(hero, 0.0, 0.0, 0.0,  0.0, 0.0, 0.0,  1.0) // pos, euler, scale
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
    draw_text(8, 8, "Score:", 28, rgb(255, 255, 255))
    draw_int(110, 8, 1234, 28, rgb(255, 255, 255))   // integers without allocating
}
```

## Project layout

`ARCHITECTURE.md` has the codemap of all 25 crates, the pipeline from source to native code, the
dependency direction, the invariants and where each is enforced. Each crate's `lib.rs` header
states that crate's own job, its place in the graph, and what it must never do.

`CONTRIBUTING.md` has the dev loop and what to touch to add a builtin, syntax, a static rule or a
renderer feature.

## Building

```sh
cargo build --release --workspace    # the toolchain (Cranelift takes a moment the first time)
cargo test  --release                # the gate
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
cargo test --release          # the whole workspace, including GPU and socket suites
```

Every capability above is backed by passing tests and a runnable example in
[`examples/`](examples/).

## Status

Real and working: a full compiler toolchain that JITs and AOT-compiles the whole
language, with ECS, a CPU rasterizer, live GPU shaders, a **GPU 3D renderer**
(PBR metallic/roughness materials, normal maps, emissive, **image-based lighting +
reflections** from the sky, a directional light with **cascaded shadow maps** plus
up to 16 point lights with **omnidirectional shadows**, **SSAO**, a procedural sky,
fog, 4x MSAA, transparency, **true GPU instancing**, billboards, debug lines,
frustum culling, GPU vertex skinning, and **full-screen post effects** (speed
lines, a directional damage vignette)),
**glTF/OBJ model loading with skeletal animation and crossfade blending**, a
real-time window with **FPS mouse-look** (cursor capture + raw delta) and a HUD
overlay, audio (including **3D positional** sound), reliable-UDP netcode plus a
**game-ready multiplayer layer** (authoritative server, client-side prediction +
reconciliation, remote-player interpolation, **world-collided movement**,
**lag-compensated hit validation**, interest management, and delta-compressed
snapshots), **2D and 3D physics** (Rapier, with a kinematic capsule character controller, raycasts that
return hit point/normal/body, shapecasts, and trigger overlaps), **2D and 3D
pathfinding** (grid A* plus a navmesh with funnel string-pulling), an asset
pipeline (PNG/JPEG/TTF/WAV), C and Rust FFI, a native debugger and profiler, and
an LSP.

What is intentionally not here yet, but is on the road:

- **An LLVM backend.** Codegen is Cranelift only (JIT + AOT, `opt_level=speed` for
  builds). Runtime speed is good, not maximal, and there is no autovectorization.
- **Editor-integrated debugger UI** and a **central package registry** (path and
  git dependencies do work today).
- **Battle-testing.** This is a capable foundation, not yet a shipped-game-proven
  production engine.

## License

MIT.
