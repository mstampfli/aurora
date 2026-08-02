# Contributing to Aurora

`ARCHITECTURE.md` says how the compiler and runtime work and which crate owns what. This says how
to work in them. Neither restates the other.

## The dev loop

```sh
cargo check -p aurora-typeck                              # the inner loop. Seconds.
cargo test --profile release-test -j 4 -p aurora-abi      # one crate's tests
cargo test --workspace --profile release-test -j 4        # THE GATE. About two minutes.
cargo fmt --check                                         # part of the gate; run cargo fmt first
```

### The suite needs the licensed pack art

Some tests read real FBX files - root motion, the modular character assembly, the
FBX importer - and they need `AURORA_TEST_FBX_DIR` pointing at a directory of
them:

```sh
AURORA_TEST_FBX_DIR=/path/to/staged/fbx cargo test --workspace --profile release-test -j 4
```

**Unset, those tests now FAIL rather than skip.** That default is deliberate and
it was expensive to learn: they used to do `let Some(m) = fixture(..) else
{ return };`, which reports a test as `ok` having asserted nothing. The variable
was unset on the machine this engine is developed on, so all six root-motion
tests and both modular-character tests had never run - and the suite counted
them as passing the whole time, so the test COUNT could never reveal it either.

If you genuinely do not have the packs, say so explicitly:

```sh
AURORA_SKIP_FIXTURE_TESTS=1 cargo test --workspace --profile release-test -j 4
```

Having to type that is the point. A check that cannot run should be as loud as
one that fails.


**`release-test`, not `release`.** It inherits `release` and turns LTO off. The suite used to be
run with `--release` and took ninety-plus minutes, almost all of it LINKING: every integration test
is its own binary, each links the whole runtime (wgpu, rapier, symphonia, cranelift), and thin LTO
runs once per binary over that graph. That is the shipping binary's trade, not a test's. Same
`opt-level`, so the timing-sensitive suites still measure an optimized build, and
`debug-assertions` stays on because a test is exactly where an overflow should abort.

Not debug either: several suites drive the GPU or compile real programs.

**`-j 4` is part of the recipe, not a laptop concession.** Cargo's default is one rustc per core and
will take the machine with it.

A ninety-minute gate is a gate that stops being run, and that is not hypothetical - the first green
run after the profile landed found ten undocumented builtins and a broken doctest that had been
sitting in the tree.

Before a commit: `cargo fmt --check` and the workspace gate above, both clean. Both must pass; the
suite includes tests that drive a real GPU and real sockets, and they are the ones that catch the
failures unit tests cannot.

`.github/workflows/gate.yml` runs those same two commands on every push to `main` and every pull
request. It runs them on a machine with **no GPU adapter**, and every test that needs one skips
instead of failing, so a green run there is not evidence the renderer is correct. The pixel tests
are yours to run locally, which is the reason the paragraph above says both must pass before the
commit rather than after it.

To try a change against a real program rather than a test:

```sh
cargo build --release -p aurorac
AURORA_HEADLESS=1 AURORA_MAX_FRAMES=60 ./target/release/aurorac run examples/shooter3d.aur
```

`AURORA_HEADLESS=1` skips the window and the audio device, which is what makes a run scriptable.
It does NOT skip the GPU: an offscreen device is still created, so two heavy programs at once still
compete for VRAM.

## Adding to the language

| Adding | Touch |
|---|---|
| a builtin | one row in `crates/aurora-abi/src/lib.rs`, the `extern "C"` fn in `aurora-runtime`, and a row in `docs/04-stdlib-and-builtins.md` - the ABI tests fail if the table and the docs disagree |
| syntax | `aurora-lexer` (if it needs a token), `aurora-parser`, `aurora-ast`, then whichever of `check`/`typeck`/`codegen` gives it meaning |
| a static rule | `aurora-check` if it needs no types, `aurora-typeck` if it does. A new error code goes in the `E0xxx` range already in use |
| a renderer feature | `aurora-render3d` (the algorithm), `aurora-window` (the immediate-mode entry), then a builtin row as above |
| anything in the prelude | `crates/aurora-std` - it is Aurora source, so it is bound by the same rules a game is |

New behaviour lands with a test that fails without it. Where a claim is about pixels, bytes or
another process, the test measures those - `aurora-render3d` compares GPU byte counts and rendered
frames, and the netcode suite runs real `Session`s over loopback rather than mocking them.

Commits are conventional and descriptive, small and focused, plain ASCII, and never credit a tool
as an author.

## Docs travel with the change

A change that alters a crate's responsibility, the pipeline, a boundary or an invariant updates
`ARCHITECTURE.md` in the same commit. A change to a crate's contract updates that crate's `lib.rs`
header. A new or changed builtin updates `docs/04-stdlib-and-builtins.md` - and there the ABI test
enforces it rather than trusting anyone to remember.
