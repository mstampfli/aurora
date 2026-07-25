# Aurora - Builtins & Standard Library Reference

This is the practical reference for writing Aurora programs: the **builtins**
(functions the compiler lowers to native runtime calls) and the **standard
library prelude** (Aurora source auto-included in every program). For the
grammar and type system see [`01-grammar-and-types.md`](01-grammar-and-types.md).

```sh
aurorac run    game.aur              # compile main to native code & run (JIT)
aurorac build  game.aur -o game.exe  # standalone optimized native executable
aurorac check  game.aur              # type + safety checks only
```

`main` is the entry point. Top-level `fn`, `struct`, `component`, `system`,
`enum`, `trait`, `impl`, `const`, and `mod` are all items; statements end at a
newline or `;` (block-form `if`/`while`/`for`/`match` need no separator).

Split a program across files with `mod NAME;`, which loads `NAME.aur` from the
declaring file's directory and namespaces its items as `NAME::item`. Only the entry
file is passed to `aurorac`; see
[`01-grammar-and-types.md`](01-grammar-and-types.md) §3.1 for the full rule.

### Compilation failures are never silent

A call to a name that is not a function, a builtin, an `@extern` import, a local
holding a closure, or a `use`d name is a hard error (`E0313`) - a typo in a call
does not compile. `check`, `run`, and `build` all check the same program (your
source, its dependencies, and the prelude), so they agree.

If a function still fails to lower to native code, `run` and `build` both refuse
and name every failing function and the reason. Neither falls back to running it:
a function that failed to compile is otherwise replaced with a stub returning 0,
and the program runs with that behaviour silently missing.

A `@vertex` / `@fragment` / `@compute` function is GPU code lowered to WGSL, so
it is exempt: it is not compiled as CPU code, and its intrinsics are not resolved
against CPU declarations.

### Where builtins come from

Every builtin below is one row of a single table, `for_each_builtin!` in
`crates/aurora-abi/src/lib.rs`: its Aurora name, the `aurora_*` runtime symbol
that implements it, and its parameter/return types. The front end's name list,
the backend's JIT symbol table, host imports and call-site signatures, and the
AOT link keeper are all generated from that one table, so they cannot drift
apart, and a row that names a runtime function that does not exist - or gives it
a signature it does not have - fails to compile.

Adding a builtin is therefore: write the `aurora_<name>` function in
`aurora-runtime`, add one row, and document it here. A test fails if a new row
is left undocumented, and another checks the argument list written here against
the table for every builtin whose arguments are plain numbers.

---

## Core builtins

| Builtin | Signature | Notes |
|---|---|---|
| `print` / `println` | `(value)` | print a scalar/string (with/without newline) |
| `assert` | `(cond)` | abort if `cond` is 0 (`panic: assertion failed`, exit 101) |
| `str` | `(int\|float) -> str` | format a number |
| `len` | `(str\|array) -> i64` | length |
| `char_at` / `substr` / `starts_with` | string ops | |
| `abs`/`min`/`max`/`clamp`/`sqrt`/`sin`/`cos`/`tan`/`floor`/`ceil`/`round`/`pow` | math | float-typed |
| `band`/`bor`/`bxor`/`shl`/`shr`/`bnot` | `(i64, i64) -> i64` | integer bitwise (`&`/`\|` are taken by refs/closures) |

Arrays are fixed-size (`[T; N]`) and **bounds-checked** - an out-of-range or
negative index panics with `array index N out of bounds (length L)`.

## ECS (the language)

`component Position { x: f64 }` declares storage; `spawn(Position { .. }, ..)`
creates an entity; `system move() { for (p, v) in query<&mut Position, &Velocity> { .. } }`
defines behaviour. `run_systems()` runs them - **independent systems in a stage
run in parallel** (the §6.2 checker proves they can't race). `despawn(e)`,
`entity_count()`.

## Graphics, window, input

| Builtin | Signature |
|---|---|
| `framebuffer(w, h)` / `clear(r,g,b)` | create / clear the CPU framebuffer |
| `pixel(x,y,r,g,b)` / `triangle(x0,y0,x1,y1,x2,y2,r,g,b)` | draw |
| `fb_get(x,y) -> i64` | read a packed `0xRRGGBB` pixel |
| `save_ppm(path)` | write the framebuffer to a PPM |
| `window_open(w,h)` / `window_present() -> i64` | real-time window (1=open) |
| `key_down(code) -> i64` | keyboard (see `key_*` helpers) |
| `mouse_x()` / `mouse_y()` / `mouse_down() -> i64` | mouse |
| `gpu_render("<wgsl>", time_ms)` / `gpu_compute(...)` | run shaders on the GPU |

## Assets & text

| Builtin | Signature | Backed by |
|---|---|---|
| `load_ppm(path) -> i64` | PPM → framebuffer | built-in |
| `load_image(path) -> i64` | **PNG/JPEG** → framebuffer | `image` crate |
| `load_font(path) -> i64` | load a TrueType/OpenType font | `fontdue` |
| `draw_text(x, y, str, px, color)` | rasterize text (alpha-blended) | `fontdue` |
| `play_note(semitone, ms)` / `play_sound(...)` | synth audio | `aurora-audio` |
| `play_wav(path) -> i64` | decode + play a **WAV** file | `hound` |
| `scene_save(path)` / `scene_load(path)` | persist the ECS world | built-in |

## Determinism & data

Seeded RNG (deterministic BY DEFAULT - a fixed seed unless `srand` is called):

| Builtin | Signature |
|---|---|
| `srand(seed)` | reseed the stream (same seed = same sequence, any machine) |
| `rand() -> f64` | uniform in `[0, 1)` (53 random bits, SplitMix64) |
| `rand_range(lo, hi) -> f64` | uniform in `[lo, hi)` |
| `rand_int(lo, hi) -> i64` | uniform integer, **inclusive** both ends |

Fixed timestep: `set_fixed_dt(dt)` pins `frame_dt()` to exactly `dt` per call
(and advances a virtual clock); `set_fixed_dt(0.0)` restores the wall clock.
The `AURORA_FIXED_DT` env var does the same for unmodified programs - the
test-harness hook for reproducible runs.

Text files: `read_file(path) -> str` ("" if unreadable - discriminate with
`file_exists(path) -> 1|0`), `write_file(path, contents) -> 1|0` (creates
parent directories). `save_png(path)` writes the 2D framebuffer as a PNG
(`save_ppm`'s tool-friendly sibling).

JSON (backed by `serde_json`; handles are `i64`, 0 = invalid/absent, reading a
bad handle is always safe). Load content as data at boot instead of hardcoding
tables:

| Builtin | Meaning |
|---|---|
| `json_parse(text) -> h` / `json_load(path) -> h` | parse (0 + stderr diagnostic on error) |
| `json_get(h, key) -> h` / `json_at(h, i) -> h` | O(1) child handles, no copying |
| `json_len(h)` | array length / object entry count / string bytes |
| `json_num(h) -> f64`, `json_int(h)`, `json_bool(h)`, `json_str(h) -> str` | leaf reads |
| `json_kind(h)` | -1 invalid, 0 null, 1 bool, 2 number, 3 string, 4 array, 5 object |
| `json_has(h, key)`, `json_key(h, i) -> str` | probing / key iteration (document order) |
| `json_new_obj()`, `json_new_arr()` | mutable builders (saves, telemetry) |
| `json_set(h, key, child)`, `json_set_num/str/bool(h, key, v)` | object writes |
| `json_push(h, child)`, `json_push_num/str(h, v)` | array appends |
| `json_to_str(h) -> str`, `json_write(h, path) -> 1|0` | pretty serialization |
| `json_free(h)` | release a handle (sub-handles keep the document alive) |

## Headless harness (capture, scripted input, tapes)

`AURORA_HEADLESS=1` runs any windowed game with NO window/event loop: presents
just advance a frame counter, 3D lives on a surface-free device, and
`AURORA_MAX_FRAMES=N` makes present report "closed" after N frames so any
unmodified `while r3d_present() { }` loop exits cleanly. If no GPU adapter
exists the run prints `aurora: HEADLESS-NO-GPU` and closes - runners must treat
that as BLOCKED, never as a pass.

- `r3d_capture(path) -> 1|0` / `r3d_capture_size(path, w, h)`: render the
  queued scene offscreen to a PNG with the HUD framebuffer composited on top
  (black = transparent, same as the live overlay). Headless-only; call it
  INSTEAD of `r3d_present` for a captured frame.
- Input injection (indistinguishable from a player; works windowed too):
  `inject_key(code, down)`, `inject_mouse_move(dx, dy)`,
  `inject_mouse_pos(x, y)`, `inject_mouse_button(b, down)`,
  `inject_scroll(dy)`, `inject_char(c)`.
- Tapes: `AURORA_INPUT_RECORD=file` writes one line of full input state per
  present; `AURORA_INPUT_REPLAY=file` replays it (real input is overridden)
  and CLOSES the window when the tape ends. Replay + `srand` defaults +
  `AURORA_FIXED_DT` reproduce a session bit-for-bit
  (see `examples/headless_capture.aur` - captures hash-identical on replay).
  `AURORA_FIXED_DT` takes precedence over `set_fixed_dt`, so a game runs at a
  fixed step under verification even if it requests wall-clock in play.
- Debug overlays (appear in captures, for rig/hitbox audits):
  `r3d_debug_skeleton(h, px,py,pz, yaw, scale, r,g,b)` draws a model's bones;
  `phys3d_debug_draw(r,g,b)` draws every physics collider as a wireframe
  (box/sphere/capsule) so you can verify hitboxes align with the mesh.
- Offline audio: under headless, `play_note`/`play_sound` record their events
  (not the device); `audio_capture_save(path) -> 1|0` renders them to a 16-bit
  WAV at their virtual timestamps, so synthesized audio can be `wav-audit`ed.
  Audio playback is otherwise a device no-op under headless (deterministic).

## Networking (reliable UDP)

`net_bind(port)`, `net_connect(host, port)`, `net_send(msg)`, `net_recv() -> str`.

## Multiplayer (authoritative server + client prediction)

A generic, game-agnostic framework for a multiplayer movement shooter: an
authoritative UDP server with N clients, **client-side prediction** of the local
player, **server reconciliation** (snap to authoritative + replay unacknowledged
inputs), and **snapshot interpolation** of remote players. The engine owns the
machinery but **no gameplay**: each tick it runs your own simulation step,
registered from Aurora with `net_sim`, over an opaque per-player state blob.
Prediction, rollback replay, and server authority all call that same step, so
they cannot drift.

The contract: a player's state is a block of `f32` floats. The engine reads only
`state[0..3]` = x,y,z and `state[3]` = yaw (for transforms / interpolation /
lag-comp); every other float is yours (velocity, timers, flags). Read and write
the blob from your sim with the raw `f32_load(ptr, i)` / `f32_store(ptr, i, v)`
accessors.

| Builtin | Signature | Notes |
|---|---|---|
| `net_host(port) -> i64` | start an authoritative server | the host is also player 0 |
| `net_join(host, port) -> i64` | join as a predicting client | 1 on success |
| `net_sim(\|state, input\| {...}, state_len, input_len)` | register the game's sim step | a closure run natively over `f32` state/input blobs; `state_len`/`input_len` floats each |
| `net_send_input(input_array) -> i64` | submit a frame's input | from an `[f64; n]` blob; predicts locally + sends; returns the input seq |
| `net_update(dt)` | pump the network | server simulates + broadcasts; client reconciles + interpolates |
| `net_spawn_at(x, y, z)` | set the local spawn point | |
| `net_my_id() -> i64` / `net_is_server() -> i64` | identity | |
| `net_player_count() -> i64` / `net_player_id_at(i) -> i64` | iterate players | |
| `net_player_x/y/z/yaw(id) -> f64` | a player's transform | predicted for the local player, interpolated for remotes |
| `net_local_x/y/z/yaw() -> f64` | the local player's transform | shorthand for the predicted self |
| `net_state(id, i) -> f64` / `net_local_state(i) -> f64` | read any game-defined state float | velocity, flags, etc. |
| `net_interest(radius)` | relevancy radius | clients are only told about players within it |
| `net_hit_radius(r)` | per-player hit sphere radius | used by the lag-compensated raycast |
| `net_fire(ox,oy,oz, dx,dy,dz, weapon)` | lag-compensated hitscan | server rewinds targets to the shooter's view; `weapon` is a 0..255 id carried through to `net_server_hit_weapon` so the server can apply per-weapon damage |
| `net_hit_player() -> i64` / `net_hit_x/y/z() -> f64` | last validated hit | player id (-1 none) + world point |

Snapshots are **delta-compressed** (only changed, in-interest players, with periodic
keyframes), so idle players cost almost nothing.

A loop: once, register your movement step with `net_sim` and host/join; then each
frame build the input blob, `net_send_input(blob)`, `net_update(dt)`, point the
camera at `net_local_*`, and draw every player with `net_player_*`. See
[`examples/mp_shooter3d.aur`](../examples/mp_shooter3d.aur) and the full
momentum controller in [`game/overclock/playground.aur`](../game/overclock/playground.aur).

## Physics - Rapier 2D (`phys_*`)

Real rigid-body simulation. Positions are body centres; units are whatever your
game uses (e.g. pixels). Bodies are referenced by an `i64` handle.

| Builtin | Signature | Notes |
|---|---|---|
| `phys_init(gx, gy)` | create/reset the world with gravity | |
| `phys_add(x, y, hw, hh, dynamic) -> i64` | box (half-extents); `dynamic` 1/0 | returns a handle |
| `phys_step(dt)` | advance the simulation | |
| `phys_x(h) -> f64` / `phys_y(h) -> f64` | body centre | |
| `phys_vel_x(h)` / `phys_vel_y(h) -> f64` | linear velocity | |
| `phys_set_vel(h, vx, vy)` / `phys_set_pos(h, x, y)` | set state | |
| `phys_apply_impulse(h, ix, iy)` | instantaneous (jumps, knockback) | |
| `phys_apply_force(h, fx, fy)` | continuous force | |
| `phys_raycast(x, y, dx, dy, max) -> f64` | distance to first hit, or `-1` | run after `phys_step` |

## Pathfinding - weighted A\* (`nav_*`)

| Builtin | Signature |
|---|---|
| `nav_init(w, h)` | create a grid |
| `nav_wall(x, y, blocked)` | mark a cell blocked (1) / open (0) |
| `nav_find(sx, sy, gx, gy) -> i64` | A* search; returns path length in cells, or `-1` |
| `nav_x(i) -> i64` / `nav_y(i) -> i64` | read the i-th path cell |

## 3D rendering, models, and animation (`r3d_*`)

A real GPU forward renderer (wgpu): indexed meshes with a depth buffer, a
perspective camera, directional + ambient lighting, base-color textures, and GPU
vertex skinning. It shares the live window's device, so 3D draws straight to the
window. Colors are 0..1 floats; angles are radians; handles are `i64`.

| Builtin | Signature | Notes |
|---|---|---|
| `r3d_load_model(path) -> i64` | load `.gltf`/`.glb`/`.obj` | meshes, materials, skeleton, clips; -1 on failure |
| `r3d_make_box(r,g,b) -> i64` | unit cube primitive | greybox geometry |
| `r3d_make_sphere(segments,r,g,b) -> i64` | UV sphere primitive | |
| `r3d_make_plane(size,tiles,r,g,b) -> i64` | ground plane in XZ | `tiles` repeats the UVs |
| `r3d_camera(ex,ey,ez, tx,ty,tz, fov_deg)` | eye, look-at target, vertical FOV | |
| `r3d_light(dx,dy,dz, r,g,b, ambient)` | directional light + ambient | |
| `r3d_clear(r,g,b)` | background color | |
| `r3d_begin()` | start a frame (clear the draw queue) | call once per frame |
| `r3d_draw(h, px,py,pz, yaw,pitch,roll, scale)` | queue a model at a transform | Euler radians, uniform scale |
| `r3d_anim_play(h, clip, looping, speed, fade)` | start an animation clip | `looping`/`speed`; `fade` crossfades from the current clip over that many seconds (0 = snap) |
| `r3d_anim_update(h, dt)` | advance the current clip | per frame |
| `r3d_clip_count(h) -> i64` | number of animation clips | |
| `r3d_present() -> i64` | render the queue to the window | 1 while open, 0 when closed |

A frame loop is `while r3d_present() { r3d_begin(); ...camera/draw...; }`. See
[`examples/shooter3d.aur`](../examples/shooter3d.aur). Materials are physically
based (metallic/roughness, normal maps, emissive, all read from glTF) with
image-based lighting + reflections from the sky; the renderer applies 4x MSAA,
cascaded shadows, and (optionally) SSAO automatically.

More rendering controls:

| Builtin | Signature | Notes |
|---|---|---|
| `r3d_sky(on, tr,tg,tb, hr,hg,hb)` | procedural sky | zenith + horizon colors, sun from the light |
| `r3d_shadows(on)` | cascaded directional shadow maps (PCF) | on by default |
| `r3d_ssao(on)` | screen-space ambient occlusion | off by default |
| `r3d_point_shadows(on)` | omnidirectional shadows for the key point light | off by default |
| `r3d_fog(r,g,b, density)` | exponential distance fog | density 0 = off |
| `r3d_point_light(x,y,z, r,g,b, range, intensity)` | add a point light | up to 16; `r3d_clear_lights()` resets |
| `r3d_make_sprite(r,g,b) -> i64` / `r3d_draw_billboard(h, x,y,z, size)` | camera-facing billboard | particles, markers |
| `r3d_debug_line(ax,ay,az, bx,by,bz, r,g,b)` | world-space debug line | aim rays, nav debug |
| `r3d_frustum_cull(on)` | toggle frustum culling | on by default |
| `r3d_screen_x/y(wx,wy,wz) -> f64` | project a world point to pixels | -1 if behind the camera |

The CPU framebuffer (`clear`/`pixel`/`triangle`/`draw_text`) is composited over
the 3D scene as a **HUD** each `r3d_present()`, with pure black as the
transparent key (clear to black, draw the crosshair/ammo in color).

## FPS input

| Builtin | Signature | Notes |
|---|---|---|
| `grab_mouse(on)` | capture + hide the cursor | for mouse-look |
| `mouse_dx() / mouse_dy() -> f64` | raw mouse motion this frame | the look delta |
| `mouse_scroll() -> f64` | scroll-wheel delta this frame | |
| `mouse_button(b) -> i64` | held: 0 = left, 1 = right, 2 = middle | |
| `key_down(code)` | extended codes | 0-9 movement/action, 10-13 Shift/Ctrl/Alt/Tab, 30-39 digits, 40-65 A-Z |

### Rebindable input actions

Decouple the game from physical keys: bind abstract **actions** (your own integer
ids) to input codes, then query actions, never raw keys. Rebind any time (e.g.
from a settings menu). Codes are the `key_down` codes for the keyboard; 100/101/102
are the left/right/middle mouse buttons.

| Builtin | Signature | Notes |
|---|---|---|
| `input_bind(action, code)` | bind an action id to an input code | rebindable at runtime |
| `input_binding(action) -> i64` | the code bound to an action | -1 if unbound |
| `input_down(action) -> i64` | is the action's input held? | 1/0 |
| `input_axis(neg, pos) -> f64` | a -1/0/+1 axis from two actions | e.g. back vs forward |

### Raw float-blob accessors

For reading and writing the opaque `f32` state/input blobs the netcode framework
hands a `net_sim` step (the pointer is passed as integer bits).

| Builtin | Signature | Notes |
|---|---|---|
| `f32_load(ptr, i) -> f64` | read the `i`-th `f32` at `ptr` | widened to `f64` |
| `f32_store(ptr, i, v)` | write `v` as the `i`-th `f32` at `ptr` | narrowed to `f32` |

## 3D positional audio

| Builtin | Signature | Notes |
|---|---|---|
| `audio_listener(x,y,z, fx,fy,fz)` | set listener pose | position + forward |
| `play_sound_at(semitone, ms, gain_pct, x,y,z)` | spatialized note | distance attenuation + stereo pan; `gain_pct` mixes the level (100 = default) |

## 3D physics - Rapier 3D (`phys3d_*`)

Real 3D rigid bodies plus a kinematic capsule character controller that slides
along walls (the core of a fluid movement shooter). Bodies are `i64` handles.

| Builtin | Signature | Notes |
|---|---|---|
| `phys3d_init(gx,gy,gz)` | create/reset the world with gravity | |
| `phys3d_add_box(x,y,z, hx,hy,hz, dynamic) -> i64` | box (half-extents) | `dynamic` 1/0 |
| `phys3d_add_sphere(x,y,z, r, dynamic) -> i64` | sphere | |
| `phys3d_add_capsule(x,y,z, hh, r, dynamic) -> i64` | upright capsule | |
| `phys3d_add_character(x,y,z, hh, r) -> i64` | kinematic character capsule | move with `move_character` |
| `phys3d_add_trimesh(verts, indices) -> i64` | static mesh collider | `[f64;N]` xyz verts, `[i64;M]` indices |
| `phys3d_step(dt)` | advance the simulation | |
| `phys3d_x/y/z(h) -> f64` | body position | |
| `phys3d_vel_x/y/z(h) -> f64` | linear velocity | |
| `phys3d_set_vel(h, vx,vy,vz)` / `phys3d_set_pos(h, x,y,z)` | set state | |
| `phys3d_apply_impulse(h, ix,iy,iz)` | instantaneous (jumps, knockback) | dynamic bodies |
| `phys3d_move_character(h, dx,dy,dz, dt)` | move + slide a character | read position after `step` |
| `phys3d_grounded(h) -> i64` | is the character on the ground | 1/0 |
| `phys3d_raycast(x,y,z, dx,dy,dz, max) -> f64` | distance to first hit, or -1 | shooting, ground checks |
| `phys3d_raycast_full(x,y,z, dx,dy,dz, max) -> i64` | hit body handle (-1 none) | then read the hit below |
| `phys3d_hit_x/y/z() -> f64` / `phys3d_hit_nx/ny/nz() -> f64` | last hit point + normal | decals, impacts |
| `phys3d_hit_body() -> i64` | last hit body handle | |
| `phys3d_spherecast(x,y,z, dx,dy,dz, r, max) -> f64` | swept-sphere distance, or -1 | thick projectiles |
| `phys3d_overlap_sphere(x,y,z, r) -> i64` | first overlapping body, or -1 | triggers, pickups, blasts |
| `phys3d_apply_force/apply_torque(h, x,y,z)` / `phys3d_set_angvel(h, x,y,z)` | dynamic forces | |
| `phys3d_set_rot(h, qx,qy,qz,qw)` / `phys3d_rot_qx/qy/qz/qw(h) -> f64` | orientation quaternion | |

## 3D pathfinding (`nav3d_*` grid, `navmesh_*` navmesh)

A 26-connected voxel grid A*, and a polygon navmesh that runs A* over a triangle
adjacency graph then string-pulls the corridor with the funnel algorithm for a
smooth path.

| Builtin | Signature | Notes |
|---|---|---|
| `nav3d_init(w,h,d)` / `nav3d_wall(x,y,z,blocked)` | build a voxel grid | |
| `nav3d_find(sx,sy,sz, gx,gy,gz) -> i64` | A* path length in cells, or -1 | |
| `nav3d_x/y/z(i) -> i64` | the i-th path cell | |
| `navmesh_build(verts, indices) -> i64` | build a navmesh from triangles | `[f64;N]` verts, `[i64;M]` indices |
| `navmesh_find(sx,sy,sz, gx,gy,gz) -> i64` | smooth path; waypoint count, or -1 | funnel string-pulled |
| `navmesh_x/y/z(i) -> f64` | the i-th waypoint | |

## Data parallelism

`par_for(out_array, |i| ...)` fills `out[i]` across OS threads (disjoint writes).

---

## Foreign function interface (`@extern`)

Bind external **C** symbols, and **Rust** functions exported as
`#[no_mangle] extern "C"`:

```aurora
@extern fn hypot(x: f64, y: f64) -> f64       // C symbol = function name
@extern("SDL_Delay") fn delay(ms: i64)        // or name the symbol explicitly
```

A bodiless `@extern fn` is declared as an import. It resolves **at link time**
for `aurorac build` (against the C runtime and anything linked into the
executable) and **against registered symbols** for `aurorac run`.

**Supported parameter/return types:** all scalars - `i64`, `f64`, `f32` - plus
**structs and arrays of scalars**, passed **by pointer**. `i64`/`f64` aggregates
read straight through (their 8-byte-slot layout matches C); aggregates containing
`f32` are **marshaled to C's packed layout** at the call site (so an Aurora
`[f32; 16]` matrix is passed as a `const float[16]`). This covers the buffers,
vectors, and matrices that real C/Rust graphics and math APIs take.

**Region contracts at the boundary.** Because an `@extern` function has no body
to infer from, you can declare its region contract with `#region` annotations on
parameter/return types - `@extern fn keep(t: #perm Thing)` or
`@extern fn tmp() -> #frame Buf`. The checker then enforces it at call sites
(passing a `#frame` value where `#perm` is required is an `E0410` error), exactly
as if the body had been visible.

To use your own C/Rust library with `aurorac build`, link it into the
`aurora-exe` crate (its `build.rs` is the hook); the runtime already bundles
`image`, `fontdue`, `hound`, Rapier, and the `pathfinding` crate this way.

---

## Standard library prelude

Auto-included Aurora source. Highlights:

**Math:** `lerp`, `clampf`, `clamp01`, `smoothstep`, `deg2rad`/`rad2deg`,
`gcd`/`lcm`/`ipow`/`factorial`/`isqrt`, `wrapf`/`fmodp`, `approach`, `minf`/`maxf`,
`maxi`/`mini`/`absi`/`clampi`/`signi`.

**Easing:** `ease_in_quad`, `ease_out_quad`, `ease_in_out_quad`, `ease_in_cubic`,
`ease_out_cubic`, `ease_in_out_cubic`.

**Vectors:** `Vec2` (`add`/`sub`/`scale`/`dot`/`length`/`dist`) and `Vec3`
(`add`/`sub`/`scale`/`dot`/`cross`/`length`/`normalize`).

**Color (packed `0xRRGGBB`):** `rgb(r,g,b)`, `red`/`green`/`blue`, `color_lerp`.

**Collision:** `Rect` (`contains`/`intersects`), `circles_hit`, `point_in_circle`,
`overlap_1d`.

**Sprites & animation:** `SpriteSheet` (`src_x`/`src_y`), `anim_frame`.

**Particles:** `Particle` (`step(dt, gravity)`, `alive`).

**Collections:** generic `List<T>` (`push`/`get`/`size`), `IntList`, `F64List`.

**Lightweight engines** (zero-dependency defaults; for serious use prefer the
`phys_*`/`nav_*` library builtins above):
- `Grid` - 4-connected BFS pathfinding (`compute_field`/`next_to`).
- `Body` - AABB physics (`step`/`collide`).
- Immediate-mode UI - `fill_rect`, `ui_button`, `ui_label`, `ui_slider`.

See [`examples/`](../examples/) - `gamedev.aur`, `physics.aur`, `ffi.aur`.

---

## Known limitations

- **FFI structs with sub-8-byte fields** (e.g. `{i32, i32}`) aren't passed by
  value - they'd need layout packing. Scalars, pointers, and structs/arrays of
  `i64`/`f64` (by pointer) all work.
- **Performance** is Cranelift-level (release builds use `opt_level=speed`); there
  is no LLVM backend or autovectorization yet.
- **Tooling**: there is a CLI debugger, profiler, and LSP (diagnostics + completion),
  but no editor-integrated debugger UI, no package registry, and the language is
  young - treat it as a capable foundation, not a battle-tested production engine.
