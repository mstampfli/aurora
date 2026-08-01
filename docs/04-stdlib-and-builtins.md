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
[`01-grammar-and-types.md`](01-grammar-and-types.md) Â§3.1 for the full rule.

### Compilation failures are never silent

A call to a name that is not a function, a builtin, an `@extern` import, a local
holding a closure, or a `use`d name is a hard error (`E0313`) - a typo in a call
does not compile. A name used as a VALUE that resolves to nothing is `E0314`,
including a qualified one: `cfg::REACH` where module `cfg` has no such const is
rejected with a source span, not left to fail in the backend. That is the shape
a rename produces - the dangling reference is in a file you were not editing -
so it is the one most likely to be missed. `check`, `run`, and `build` all check
the same program (your source, its dependencies, and the prelude), so they agree.

Both qualified guards are deliberately narrow: they fire only when the prefix
demonstrably IS a module, because something else in the program is already
qualified with it. An enum variant, an associated function or const on a type,
and a trait path are never at risk.

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
run in parallel** (the Â§6.2 checker proves they can't race). `despawn(e)`,
`entity_count()`, `world_clear()`.

`world_clear()` despawns every entity and drops all component storage - what a
level transition needs, and what a test suite needs between cases. Entity ids
keep counting up rather than restarting, so an id held from before the clear
names nothing instead of silently naming whatever entity later takes its number.

A system's access set is what it **reaches**, not what its body spells out:
queries inside the functions it calls count as its own, transitively. Ordering
with `after`/`before` is transitive and independent of declaration order (Â§6.2).

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
| `load_ppm(path) -> i64` | PPM â†’ framebuffer | built-in |
| `load_image(path) -> i64` | **PNG/JPEG** â†’ framebuffer | `image` crate |
| `load_font(path) -> i64` | load a TrueType/OpenType font | `fontdue` |
| `draw_text(x, y, str, px, color)` | rasterize text (alpha-blended) | `fontdue` |
| `play_note(semitone, ms)` / `play_sound(...)` | synth audio | `aurora-audio` |
| `play_wav(path) -> i64` | decode + play an audio file once (1 = played) | `hound` / `symphonia` |
| `scene_save(path)` / `scene_load(path)` | persist the ECS world | built-in |

### Real audio assets

`load_sound` decodes a file ONCE into a cached mono buffer and hands back a handle, so
playing a sound costs no decode and no allocation. Accepted formats: **WAV, MP3,
OGG/Vorbis, FLAC, M4A/AAC, MKV, AIFF, CAF, ALAC, ADPCM** - the container is identified
by content, so a mislabelled extension still loads. WAV goes through `hound`; everything
else through `symphonia`. Returns `-1` if the file is missing or undecodable, and every
play function treats a negative handle as a no-op, so a missing asset degrades to silence
instead of a crash.

Buffers are resampled to the device rate at load, and playback shares them by `Arc`, so
sustained fire re-plays the same buffer with no copy.

| Builtin | Meaning |
|---|---|
| `load_sound(path) -> i64` | decode + cache, returns a handle (`-1` on failure) |
| `play_sound_handle(h, vol)` | play a loaded sound (`vol` 0..100) |
| `play_sound_handle_at(h, vol, x, y, z)` | the same, positioned for the 3D listener |
| `play_music(h, vol)` | start a looping music bed (replaces any current bed) |
| `music_volume(vol)` | change the bed's level WITHOUT restarting it |
| `music_stop()` | stop the bed |
| `play_ambience(h, vol)` | start a looping ambience layer, mixed under music |

Because `music_volume` does not restart playback, a long track can stay continuous while
the game moves its level - which is what you want for a score that reacts to state.

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

**The simulation clock.** Frame time is what the display did; tick time is what
the simulation did, and game rules belong on the second. A window stated in
ticks - invulnerable from tick 6 to tick 27 - means the same thing on every
machine only because ticks are a fixed length.

| Builtin | Meaning |
|---|---|
| `set_tick_rate(hz)` | ticks per second, default 60. Values outside `1..=1000` are ignored rather than allowed to produce a zero or negative step |
| `tick_count() -> i64` | fixed ticks simulated so far; advances at the configured rate however long frames take |
| `tick_delta() -> f64` | the fixed step in seconds - what a fixed-rate system integrates with, never `frame_dt()` |
| `tick_alpha() -> f64` | position between the last tick and the next, `0..1`. Interpolating render positions by it removes the judder you get when frame rate and tick rate disagree |

The clock is per-thread, so a server and a client in one process keep separate
simulated time. A stalled frame owes many steps at once; at most 8 run and the
remaining debt is dropped, because chasing all of it makes the next frame longer
still and the program never catches up. Losing simulated time after a stall is
visible and survivable; locking up is not.

Text files: `read_file(path) -> str` ("" if unreadable - discriminate with
`file_exists(path) -> 1|0`), `write_file(path, contents) -> 1|0` (creates
parent directories). `save_png(path)` writes the 2D framebuffer as a PNG
(`save_ppm`'s tool-friendly sibling).

## Process environment

The program's own command line and environment, so one binary can dispatch its
own role (`--host`, `--dedicated`, `--verify <name>`).

| Builtin | Signature | Notes |
|---|---|---|
| `sys_argc()` | `-> i64` | argument count, **including** argv[0], so always >= 1 |
| `sys_arg(i)` | `-> str` | the i-th argument; `""` when `i` is out of range either way |
| `sys_env(name)` | `-> str` | an environment variable, or `""` when unset |

`sys_arg(0)` is the program **as invoked**: the executable's path for a binary
built with `aurorac build`, and the source file's path under `aurorac run`.
`sys_arg(1..)` are the program's own arguments and are identical either way, so
a program reads the same command line however it was compiled:

```sh
aurorac build game.aur -o game.exe && ./game.exe --host 45123
aurorac run   game.aur --host 45123        # same sys_arg(1) / sys_arg(2)
aurorac run   game.aur -- --host 45123     # a leading `--` is dropped
```

An unset variable and one set to the empty string both read as `""`, so use a
sentinel value (not emptiness) if you must tell them apart.

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

**Handle lifetime.** `json_free` returns the handle's slot for reuse, so
parse-and-free in a loop - a server handling one document per request - stays
bounded. A freed handle is invalidated rather than recycled: the next
`json_parse` may land in the same slot, but the old handle keeps reading as kind
`-1` instead of quietly answering with the new document. Every handle you are
given is yours to free, including the child handles from `json_get`/`json_at`;
holding them all while walking a large array keeps that many nodes (and the
document behind them) alive, exactly as it would in C.

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
`net_bind` must come first: `net_connect` points an existing local endpoint at a peer, so
without one there is nothing to point.

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
| `net_hit_radius(r)` | per-player hit sphere radius | used by the lag-compensated raycast and melee sweep |
| `net_fire(ox,oy,oz, dx,dy,dz, weapon)` | lag-compensated hitscan | server rewinds targets to the shooter's view; `weapon` is a 0..255 id carried through to `net_server_hit_weapon` so the server can apply per-weapon damage |
| `net_melee(ox,oy,oz, fx,fy,fz, reach, arc_degrees, weapon)` | lag-compensated melee swing | everything within `reach` of the origin and inside `arc_degrees` about the facing, rewound to the swinger's view. **Cleaves**: every target covered reaches the host's validated-hit queue, and `net_hit_player` reports the nearest for the hitmarker |
| `net_hit_player() -> i64` / `net_hit_x/y/z() -> f64` | last validated hit | player id (-1 none) + world point |
| `net_set_object_size(i, radius, half_h)` | size an object's lag-comp collider | a vertical capsule; `half_h` 0 is a sphere. Default is crate-sized, so anything larger must say so or swings that visibly connect will miss |
| `net_set_object_state(i, slot, v)` (host) / `net_object_state(i, slot) -> f64` | per-object game state | six floats an object carries besides its pose, replicated with it. What they mean is the game's business, as with `net_state` for players |

**World objects** are the channel for anything server-owned that is not a player:
a crate, or a boss. Nobody predicts them - the host decides what they are doing
and every other machine is told - which is what makes a boss's telegraph the same
telegraph on every screen. They are recorded into lag compensation each tick, so
`net_fire` and `net_melee` can hit them, and they are re-sent whenever any slot
changes, state included: a boss winding up does not move, and change detection
that watched only the pose would show every other player a statue.
| `net_set_meta(slot, v)` / `net_player_meta(id, slot) -> f64` | per-player gameplay scalars (hp, shield, kills) | **f32 on the wire**: a value needing more than 24 bits of mantissa comes back rounded |
| `net_player_input(id, i) -> f64` | what a player is TRYING to do | the authority acts on this for doors, purchases and revives; answers for the local player too, so one code path drives the whole squad |
| `net_set_local_state(slot, v)` | publish where THIS peer's body is | for a mover that lives in a physics world; call before `net_send_input` |
| `net_owned_movement(on)` (host) | 1 = every peer owns its own body and the host relays it; 0 = the host re-simulates each client | default 0. Owned movement suits co-op PvE: it cannot drift, since a body has one simulator. Combat stays host-authoritative either way |
| `net_set_world_len(ch, n)` / `net_set_world(ch, i, v)` (host) | publish world array `ch` (0..3) | state every peer must agree on that no transform implies - level layout, entity tables. Separate channels so state that changes every frame does not drag along state that changes once a round |
| `net_world_len(ch) -> i64` / `net_world(ch, i) -> f64` / `net_world_gen(ch) -> i64` | read one | on a client the length is 0 until a COMPLETE version of that channel arrives, so it doubles as "do I have it yet?" |
| `net_set_tag(v)` / `net_player_tag(id) -> i64` | one EXACT 64-bit token per player | replicated verbatim - use it for fingerprints, seeds, ids and bitmasks, which an f32 meta slot silently rounds |

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
| `phys_remove(h) -> i64` | destroy a body and its collider | 1 if removed, 0 if `h` was already dead; `h` stays dead afterwards |
| `phys_alive(h) -> i64` | is `h` still a live body | 1/0; tells "removed" from "sitting at the origin" |
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
| `r3d_load_model(path) -> i64` | load `.gltf`/`.glb`/`.obj`/`.fbx` | meshes, materials, skeleton, clips; -1 on failure |
| `r3d_free_model(h) -> i64` | release a model/primitive handle | frees its GPU meshes and materials; 1 if freed, 0 if the handle was already dead |
| `r3d_model_extent(h,axis) -> f64` | half-extent of the model's bounding box | axis 0/1/2 = x/y/z, in model space and before draw scale; 0.0 for a dead handle or bad axis |
| `r3d_model_centre(h,axis) -> f64` | centre of the model's bounding box | relative to the model's origin, so a model standing on its origin reports a positive `y`; 0.0 for a dead handle |
| `r3d_make_box(r,g,b) -> i64` | unit cube primitive | greybox geometry |
| `r3d_make_sphere(segments,r,g,b) -> i64` | UV sphere primitive | |
| `r3d_make_plane(size,tiles,r,g,b) -> i64` | ground plane in XZ | `tiles` repeats the UVs |
| `r3d_camera(ex,ey,ez, tx,ty,tz, fov_deg)` | eye, look-at target, vertical FOV | |
| `r3d_light(dx,dy,dz, r,g,b, ambient)` | directional light + ambient | |
| `r3d_clear(r,g,b)` | background color | |
| `r3d_begin()` | start a frame (clear the draw queue) | call once per frame |
| `r3d_draw(h, px,py,pz, yaw,pitch,roll, scale)` | queue a model at a transform | Euler radians, uniform scale |
| `r3d_draw_scaled(h, px,py,pz, yaw,pitch,roll, sx,sy,sz)` | queue a model with a PER-AXIS scale | one unit-cube mesh can then be every wall, floor and pillar in a level. Without it a box-built level needs a mesh per distinct size, which turns making a room resident into GPU uploads - exactly what a streamed level must avoid |
| `r3d_draw_skinned(armor, host, px,py,pz, yaw,pitch,roll, scale)` | queue `armor` deformed by `host`'s current pose | for gear that must move with a character without owning a skeleton: the `armor` mesh carries per-vertex joint weights in the HOST's joint order, and this feeds it the host's skin matrices. Skins from the host's FULL pose, so `r3d_hide_joint` on the host (hiding covered body parts) never collapses the gear worn over them |
| `r3d_clip_name(h, i) -> str` | the asset's own name for clip `i` | `""` for a stale handle or an out-of-range index. Use it to discover what a model actually contains |
| `r3d_clip_index(h, name) -> i64` | find a clip BY NAME | -1 when the model has no such clip. Prefer this to a literal index: exporters emit clips in whatever order they like, and a stale index silently plays the WRONG animation instead of failing. An armature prefix is tolerated, so `"Walk"` matches `"CharacterArmature|Walk"`, and matching is case-insensitive |
| `r3d_show_joints(h)` | undo every `r3d_hide_joint` on a model | lets a pooled character be reused without reloading it |
| `r3d_joint_index(h, name) -> i64` | find a JOINT by name | -1 when the model has no such joint. The bone-attachment counterpart of `r3d_clip_index`, and the reason to prefer it is the same: a hardcoded joint index welds a weapon to the wrong bone the moment the rig is re-exported. Tolerates an armature prefix, ignores case |
| `r3d_joint_name(h, i) -> str` | the name of joint `i` | `""` for a stale handle or a bad index; pair it with `r3d_joint_dump` when discovering a rig |
| `r3d_joint_dump(h)` | print every joint index, name and parent to stdout | rig discovery: run it once to see what a model's skeleton is called, then bind by name |
| `r3d_hide_joint(h, joint)` | hide one skin joint's geometry | zeroes that joint's skinning matrix, collapsing its geometry to the model origin: first-person arms drop the torso/head/legs, and a body part covered by gear stops clipping through it. Joints 0..63 only (the mask is 64 bits), and it ACCUMULATES: clear it with `r3d_show_joints` |
| `r3d_anim_play(h, clip, looping, speed, fade)` | start an animation clip | `looping`/`speed`; `fade` crossfades from the current clip over that many seconds (0 = snap) |
| `r3d_anim_update(h, dt)` | advance the current clip | per frame |
| `r3d_anim_seek(h, t)` | jump the current clip to `t` seconds | for state that is already true when you first see it - a body that went down ten seconds ago should be lying on the floor, not starting to fall over again. Cancels any crossfade in progress |
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

### Modular characters

A stylised character pack ships a body as a dozen separate meshes over one
skeleton, one shared texture atlas for the whole cast, and its animations as
hundreds of single-clip files authored on a rig that is not the character's.
These assemble that into one animated body.

| Builtin | Signature | Notes |
|---|---|---|
| `r3d_load_part(path, host) -> i64` | load a mesh as part of `host`'s body | rebinds its skinning onto the host's skeleton by bone name, so one pose drives them together. -1 if a bone it deforms with is missing from the host, or if the two disagree about where a bone rests |
| `r3d_material_texture(material, path)` | attach an atlas by material name | for meshes that carry no texture of their own. Applies to models loaded *after* the call |
| `r3d_clip_rig(path)` | the rig the clips were authored on | a clip-only export has no usable rest pose of its own, and a joint's local rotation means nothing without one |
| `r3d_clip_add(path)` | add one clip file to the moveset | |
| `r3d_bone_map(from, to)` | rename a bone between the clips' rig and the character's | only bones whose names differ need an entry |
| `r3d_clip_root(bone)` | let this bone take translation from a clip | the root, so locomotion travels. Every other bone keeps the character's own offsets: a clip-only export has none to give, and its zeroes would collapse the body onto its hip |
| `r3d_load_character(path) -> i64` | load a character with the moveset gathered so far | retargets each clip onto this skeleton, then clears the gathering so one character's moveset cannot leak into the next |
| `r3d_part_add(path)` | add one mesh file to the body being gathered | for `r3d_load_assembly` |
| `r3d_load_assembly() -> i64` | assemble one character from the gathered parts | derives the rig as the union of the parts' skeletons, rebinds each part onto it, and uploads the result as a single character. -1 if the parts do not share a rig |

`r3d_load_assembly` exists because a modular pack ships **no whole body and no
skeleton file**. Every part carries only the bones it deforms with plus the chain
above them to hang from: a hand knows its fingers, a helmet knows the spine, and
no single file knows both. There is therefore nothing to pass `r3d_load_part` as
a host, and the rig has to be built before anything can bind to it. Bones shared
between parts must agree on where they rest, so a part authored for a different
body is refused rather than averaged into a seam that opens only in some poses.

Use `r3d_load_part` when a whole-body character already exists and you are adding
to it; use `r3d_load_assembly` when the parts are all there is.

The moveset is gathered call by call rather than passed in one go because a
builtin cannot take a list of strings, and it is attached at load rather than
afterwards because an uploaded asset is shared between every handle that loaded
the same file - attaching to one later would rewrite the moveset of every
character already drawing from it.

```aurora
r3d_material_texture("ModularFantasyHeroCharacters", "art/atlas_01_A.png")

r3d_clip_rig("anim/ReferenceRig.fbx")
r3d_bone_map("Hips", "Pelvis")
r3d_bone_map("Shoulder_L", "UpperArm_L")
r3d_clip_root("Pelvis")
r3d_clip_add("anim/Attack_LightCombo01A.fbx")
let hero = r3d_load_character("art/Character.fbx")

let torso = r3d_load_part("art/parts/Torso_00.fbx", hero)
let legs  = r3d_load_part("art/parts/LegLeft_00.fbx", hero)

// Or, when the pack ships parts and nothing else, assemble the rig from them:
r3d_part_add("art/parts/Hips_20.fbx")
r3d_part_add("art/parts/Torso_20.fbx")
r3d_part_add("art/parts/HandLeft_17.fbx")
let boss = r3d_load_assembly()

let swing = r3d_clip_index(hero, "Attack_LightCombo01A")
r3d_anim_play(hero, swing, 1, 1.0, 0.15)

// Each frame: advance the host, then draw every part from its pose.
r3d_anim_update(hero, tick_delta())
r3d_draw_skinned(torso, hero, 0.0,0.0,0.0, 0.0,0.0,0.0, 1.0)
r3d_draw_skinned(legs,  hero, 0.0,0.0,0.0, 0.0,0.0,0.0, 1.0)
```

### Asset lifetime

`r3d_load_model` and the `r3d_make_*` primitives each upload their own GPU
buffers, so a program that loads assets in a loop - or a game that changes level
- has to release them, with `r3d_free_model(h)`. Nothing is freed implicitly;
handles live until you free them or the process ends.

A handle is not an index. It carries the generation of the slot it was issued
from, so after `r3d_free_model(h)` the value in `h` is **dead**: drawing,
animating, or freeing with it does nothing and returns 0, even after a later
load has reused that slot. A stale handle can never resolve to a different
asset, which is what makes freeing safe to do mid-game.

```aurora
let level = r3d_load_model("assets/level1.glb")
// ...play the level...
r3d_free_model(level)          // 1: the GPU buffers are gone
r3d_draw(level, 0.0,0.0,0.0, 0.0,0.0,0.0, 1.0)   // no-op, not a wrong model
```

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
| `input_pressed(action) -> i64` | did it go down THIS frame? | 1/0 |
| `input_released(action) -> i64` | did it come up THIS frame? | 1/0 |
| `input_suppress(on)` | freeze all bound-action reads | raw key/mouse untouched |
| `input_step()` | advance the edge snapshot | automatic in `present` |

`input_pressed` is the difference between "drink one flask" and "drink five": a
held button is one press, not sixty. The snapshot it compares against is advanced
by `window_present` and `r3d_present`, so a game with a window never has to think
about it. A headless program that injects input and steps a simulation without
presenting has no frame boundary of its own, and calls `input_step()` where its
frame ends.

Edges are tracked per input CODE, not per action, so rebinding an action while
its old key is held cannot manufacture a press on the new one. The snapshot
records the raw key state even while `input_suppress` is on, so a pause menu
opened and closed with attack held does not fire an attack on the way out.

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
| `phys3d_add_model_collider(model,x,y,z,yaw,sx,sy,sz) -> i64` | static collider shaped like a loaded model | concave (triangle mesh, not a hull), joins the world group so movement probes see it; -1 if the handle has no mesh. Placed and scaled like the matching `r3d_draw_scaled`, so the collider IS the art |
| `phys3d_remove(h) -> i64` | destroy a body and its collider | 1 if removed, 0 if `h` was already dead; `h` stays dead afterwards |
| `phys3d_alive(h) -> i64` | is `h` still a live body | 1/0; tells "removed" from "sitting at the origin" |
| `phys3d_step(dt)` | advance the simulation | |
| `phys3d_x/y/z(h) -> f64` | body position | |
| `phys3d_vel_x/y/z(h) -> f64` | linear velocity | |
| `phys3d_set_vel(h, vx,vy,vz)` / `phys3d_set_pos(h, x,y,z)` | set state | |
| `phys3d_apply_impulse(h, ix,iy,iz)` | instantaneous (jumps, knockback) | dynamic bodies |
| `phys3d_move_character(h, dx,dy,dz, dt)` | move + slide a character | read position after `step` |
| `phys3d_grounded(h) -> i64` | is the character on the ground | 1/0 |
| `phys3d_character_solid(h, on)` | does this character's movement collide with other characters | off by default |
| `phys3d_raycast(x,y,z, dx,dy,dz, max) -> f64` | distance to first hit, or -1 | it hits ANY body, INCLUDING the one the ray starts inside: fired from a character's own centre it returns 0 and every shot silently stops at the muzzle. For shooting or ground probes from a body, use `phys3d_raycast_ex` / `phys3d_raycast_world` and pass that body as `exclude` |
| `phys3d_raycast_full(x,y,z, dx,dy,dz, max) -> i64` | hit body handle (-1 none) | then read the hit below |
| `phys3d_raycast_ex(exclude, x,y,z, dx,dy,dz, max) -> i64` | like `raycast_full`, skipping one body | probe outward from your own centre; a NEGATIVE `exclude` skips nothing |
| `phys3d_raycast_world(exclude, x,y,z, dx,dy,dz, max) -> i64` | like `raycast_ex`, but WORLD geometry only | for movement: ground checks, walls, mantle. Ignores other character capsules, so a player cannot stand on a player |
| `phys3d_hit_x/y/z() -> f64` / `phys3d_hit_nx/ny/nz() -> f64` | last hit point + normal | decals, impacts |
| `phys3d_hit_body() -> i64` | last hit body handle | |
| `phys3d_spherecast(x,y,z, dx,dy,dz, r, max, ignore) -> f64` | swept-sphere distance, or -1 | thick projectiles, camera probes. `ignore` is a body handle the sweep passes through, or -1 for none - a sweep starting inside a body otherwise hits it at zero distance, which is what a camera probe from the character head always does |
| `phys3d_overlap_sphere(x,y,z, r) -> i64` | first overlapping body, or -1 | triggers, pickups, blasts |
| `phys3d_apply_force/apply_torque(h, x,y,z)` / `phys3d_set_angvel(h, x,y,z)` | dynamic forces | |
| `phys3d_set_rot(h, qx,qy,qz,qw)` / `phys3d_rot_qx/qy/qz/qw(h) -> f64` | orientation quaternion | |

### Body handles and removal

A body handle is an **opaque `i64`**, not an index. It carries a generation
alongside the slot, so `phys3d_remove` (and `phys_remove` in 2D) does not just
free the body - it makes the handle **invalid for good**. A later
`phys3d_add_*` may land in the freed slot, but it gets a different handle, and
the old one keeps reading as dead:

```aurora
let bullet = phys3d_add_sphere(0.0, 2.0, 0.0, 0.1, 1)
phys3d_remove(bullet)          // 1
let enemy = phys3d_add_box(9.0, 9.0, 9.0, 1.0, 1.0, 1.0, 1)
phys3d_alive(bullet)           // 0  - not "the enemy"
phys3d_x(bullet)               // 0.0, never enemy's 9.0
phys3d_remove(bullet)          // 0  - a double free takes nothing with it
```

That is the point: with a plain index, `bullet` would silently become a second
name for `enemy`, and a stray `phys3d_set_pos(bullet, ...)` would teleport the
wrong actor. Reads on a dead handle answer the same "nothing there" value they
answer for `-1` (0.0 for position and velocity, identity for rotation, 0 for
`grounded`), so use `phys3d_alive` when you need to tell that apart from a body
genuinely at the origin.

Two consequences worth knowing:

* **Keep a handle in an `i64`.** A handle no longer fits in an `f32` (only
  integers below 2^24 do), so stashing one in a float array - a netcode state
  blob, say - truncates the generation and the handle is REJECTED. Put a small
  dense actor id in the blob and keep the real handle in your own table.
* **Removal is immediate.** A raycast run between `phys3d_remove` and the next
  `phys3d_step` already does not see the body.

Nothing changes for a program that never removes anything: without a
`phys3d_remove` no slot is ever reused, and the simulation is bit-identical.

## Heightmap terrain (`terrain_*`)

An open-world ground surface: a heightfield that is rendered with distance-based
level of detail, registered with the physics world as a Rapier heightfield
collider, and sampled by a height query. All three read **one** heightfield, so
the surface you see, the surface you walk on, and the number `terrain_height`
returns are the same triangles. There is one terrain at a time.

| Builtin | Signature | Notes |
|---|---|---|
| `terrain_generate(seed, dim, spacing, amplitude) -> i64` | build a procedural heightfield | value-noise fBm centred on the origin, heights in `[0, amplitude]`; deterministic; 1 on success, 0 on failure |
| `terrain_load(path) -> i64` | read an `.aterr` file | 1 on success, 0 on failure (a failed load leaves the previous terrain alone) |
| `terrain_save(path) -> i64` | write the loaded terrain as `.aterr` | 1/0; author a terrain once, ship it as an asset |
| `terrain_color(r,g,b)` | terrain albedo, 0..1 | applies from the next `terrain_draw` |
| `terrain_draw()` | queue the terrain for this frame | between `r3d_begin` and `r3d_present`, like `r3d_draw`; picks per-tile detail from the current camera |
| `terrain_height(x,z) -> f64` | surface height at a world position | interpolated across the collider's own triangles |
| `terrain_collider() -> i64` | register the terrain with 3D physics | call after `phys3d_init`; returns a `phys3d_*` body handle, or -1; REPLACES the collider it issued last |
| `terrain_size() -> i64` | samples per side (`dim`) | 0 if no terrain is loaded |
| `terrain_spacing() -> f64` | world units between samples | |
| `terrain_origin_x/z() -> f64` | world position of sample (0,0) | the terrain's -X / -Z border |

`dim` must be **`2^k + 1`** (5, 9, 17, 33, 65, 129, 257, 513, 1025, 2049, 4097) -
the usual heightmap constraint - so the tile grid and every level of detail divide
it exactly. Anything else is refused with a message rather than meshed wrong. The
terrain occupies `x` in `[origin_x, origin_x + (dim-1)*spacing]`, and `z`
likewise.

**Out of bounds.** `terrain_height` outside the footprint CLAMPS to the nearest
edge sample, so it is always defined and never returns garbage: the surface reads
as if the border extended outward forever. A non-finite coordinate clamps to the
`(origin_x, origin_z)` corner. With no terrain loaded it returns 0. Note that the
COLLIDER stops at the footprint, so past the border that height has no collision
behind it; compare against `terrain_origin_x/z()` and `terrain_size()` if your
game needs to know it has left the map.

**Collision groups.** The terrain collider is world geometry (group 1), the same
group `phys3d_add_box` uses, so `phys3d_move_character` walks on it and
`phys3d_raycast_world` ground probes hit it, while other players' character
capsules (group 2) stay invisible to those probes.

**Level of detail.** The field is cut into 32-cell tiles; each picks a sample step
from its distance to the camera, so a distant tile costs a quarter of the
triangles per level. Seams are edge-stitched, not skirted: a tile whose neighbour
is coarser builds that edge at the neighbour's step, so both sides emit identical
vertex positions and there is no crack and no T-junction. Normals come from the
heightfield gradient at full resolution, so lighting does not pop across a level
change.

**Tile memory.** A tile's GPU mesh is built the first time that tile is actually
drawn, so without a bound the resident set would be every tile the camera has
ever seen - about 1.7 GiB for the 4097-sample cap. The tile cache is bounded at
**32 MiB** of GPU vertex + index data (roughly 300 worst-case full-detail tiles,
and many more in practice, since a tile past the first LOD threshold is a
quarter the size per step). Over budget, the least-recently-drawn tiles are
evicted and rebuilt if the camera comes back. Tiles drawn in the current frame
are never evicted, so a single frame whose visible tiles exceed the budget keeps
them all rather than rendering a hole.

**Reloading.** `terrain_generate` / `terrain_load` REPLACE the terrain, and the
outgoing one's tile meshes and material are released as the new one is
installed, so reloading in a loop does not accumulate GPU memory.
`terrain_collider()` likewise REPLACES the collider it issued last rather than
stacking another heightfield on top of it, so a reload loop is bounded on the
physics side too and a stale surface never answers raycasts underneath the new
one. The handle it returned before is dead afterwards (`phys3d_alive` reports
0); calling it once per terrain is still the tidy thing to do, but it is no
longer load-bearing.

### The `.aterr` heightfield format

Little-endian throughout, exactly `24 + dim*dim*4` bytes:

| offset | size | field |
|---|---|---|
| 0 | 8 | magic: the ASCII bytes `AURTERR1` |
| 8 | 4 | `u32` `dim`, samples per side (`2^k + 1`, 5..=4097) |
| 12 | 4 | `f32` `spacing`, world units between samples (> 0) |
| 16 | 4 | `f32` `origin_x`, world X of sample column 0 |
| 20 | 4 | `f32` `origin_z`, world Z of sample row 0 |
| 24 | `dim*dim*4` | `f32` heights, row-major: sample `(row, col)` at byte `24 + (row*dim + col)*4` |

Column indices run along **+X**, row indices along **+Z**, and a height is a world
**Y** in the same units as everything else, so sample `(row, col)` sits at
`(origin_x + col*spacing, height, origin_z + row*spacing)`. A file whose magic,
`dim`, or length does not check out is rejected with a message.

```aurora
fn main() {
    terrain_generate(1234, 513, 1.0, 40.0)
    phys3d_init(0.0, 0.0 - 20.0, 0.0)
    terrain_collider()
    let ground = terrain_height(0.0, 0.0)
    while r3d_present() {
        r3d_begin()
        r3d_camera(0.0, ground + 30.0, 40.0, 0.0, ground, 0.0, 70.0)
        terrain_draw()
    }
}
```

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
