# Contributing to Aurora

`ARCHITECTURE.md` says how the compiler and runtime work and which crate owns what. This says how
to work in them. Neither restates the other.

## The dev loop

```sh
cargo check -p aurora-typeck          # the inner loop: the crate you are in. Seconds.
cargo test  --release -p aurora-abi   # one crate's tests
cargo test  --release                 # THE GATE: the whole workspace
cargo fmt --check                     # part of the gate; run cargo fmt first
```

Release, not debug: several suites drive the GPU or compile real programs, and a debug build turns
minutes into tens of minutes. On a laptop, cap the parallelism so a build cannot starve the
machine:

```sh
CARGO_BUILD_JOBS=4 nice -n19 cargo test --release
```

Before a commit: `cargo fmt --check` and `cargo test --release`, both clean. Both must pass; the
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
