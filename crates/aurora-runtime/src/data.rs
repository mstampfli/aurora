//! Determinism + data builtins: seeded RNG, fixed-timestep override, text file
//! I/O, and a JSON API. These exist so games load content as data (manifests,
//! dialogue, quests, levels, tuning) and so any session can be reproduced
//! bit-for-bit (fixed seed + fixed dt).
//!
//! JSON handles are thread-local `i64`s (0 = invalid/absent). Parsed documents
//! are immutable; child lookups return sub-handles in O(1) without cloning
//! (the root is refcounted, nodes are interior pointers). Builder handles
//! (`json_new_obj`/`json_new_arr`) own mutable values for constructing saves /
//! telemetry, serialized with `json_to_str`/`json_write`.

use std::cell::{Cell, RefCell};
use std::rc::Rc;

use serde_json::Value;

// --- seeded RNG (SplitMix64) ------------------------------------------------
//
// Deterministic BY DEFAULT: a fixed seed unless the program calls `srand`.
// Games that want entropy must opt in with e.g. `srand(time-derived value)`;
// the factory's harness relies on the default being reproducible.

const DEFAULT_SEED: u64 = 0x9E37_79B9_7F4A_7C15;

thread_local! {
    static RNG_OWN: Cell<u64> = const { Cell::new(DEFAULT_SEED) };
}

/// The random stream, routed to the batch owner's while this thread is a
/// worker. See `ROUTED_CELLS` in the crate root.
///
/// Worse here than a missing world would be: a worker does not get an empty
/// RNG, it gets a fresh one seeded identically. So every worker draws the SAME
/// numbers, the owner's stream never advances, and a game whose peers replay
/// each other's rules - which is what this runtime's netcode does - diverges
/// silently the first time a system rolls anything.
pub(crate) fn own_rng() -> *const () {
    RNG_OWN.with(|c| c as *const _ as *const ())
}

struct RngSlot;

impl RngSlot {
    fn with<R>(&self, f: impl FnOnce(&Cell<u64>) -> R) -> R {
        let batch = crate::par_batch();
        if batch.is_null() {
            return RNG_OWN.with(f);
        }
        unsafe {
            crate::with_par_cell(
                batch,
                crate::par_cell(batch, crate::CELL_RNG) as *const Cell<u64>,
                f,
            )
        }
    }
}

const RNG: RngSlot = RngSlot;

fn next_u64() -> u64 {
    RNG.with(|s| {
        let mut z = s.get().wrapping_add(0x9E37_79B9_7F4A_7C15);
        s.set(z);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    })
}

/// Reseed the RNG stream. Same seed => same sequence, on any machine.
#[no_mangle]
pub extern "C" fn aurora_srand(seed: i64) {
    RNG.with(|s| s.set(seed as u64));
}

/// Uniform f64 in [0, 1) with 53 random bits.
#[no_mangle]
pub extern "C" fn aurora_rand() -> f64 {
    (next_u64() >> 11) as f64 * (1.0 / 9_007_199_254_740_992.0)
}

/// Uniform f64 in [lo, hi) (degenerate ranges return lo).
#[no_mangle]
pub extern "C" fn aurora_rand_range(lo: f64, hi: f64) -> f64 {
    if hi > lo {
        lo + aurora_rand() * (hi - lo)
    } else {
        lo
    }
}

/// Uniform integer in [lo, hi] inclusive (degenerate ranges return lo).
/// Modulo bias is ~range/2^64 - negligible for gameplay use.
#[no_mangle]
pub extern "C" fn aurora_rand_int(lo: i64, hi: i64) -> i64 {
    if hi > lo {
        let span = (hi - lo) as u64 + 1;
        lo + (next_u64() % span) as i64
    } else {
        lo
    }
}

// --- fixed timestep ---------------------------------------------------------
//
// When active, `frame_dt()` returns the scripted step and advances a virtual
// clock instead of reading the wall clock, so replays and headless runs are
// time-deterministic. Activated by the `set_fixed_dt` builtin or the
// `AURORA_FIXED_DT` env var (harness-friendly: works on unmodified games).

thread_local! {
    /// The program's own `set_fixed_dt` override: Some(dt)=fixed, None=wall clock.
    static FIXED_DT_OWN: Cell<Option<f64>> = const { Cell::new(None) };
    static VIRTUAL_TIME_OWN: Cell<f64> = const { Cell::new(0.0) };
}

/// The pinned timestep and the virtual clock it advances, routed to the batch
/// owner's while this thread is a worker. See `ROUTED_CELLS` in the crate root.
///
/// A worker with its own copy sees no pinned step at all, so `frame_dt()` falls
/// through to the wall clock and answers 1/60 - a plausible number, on a run the
/// program had pinned precisely so that it would be reproducible. Determinism
/// under a fixed step is what the replay tape and the netcode are both built on.
pub(crate) fn own_fixed_dt() -> *const () {
    FIXED_DT_OWN.with(|c| c as *const _ as *const ())
}

pub(crate) fn own_virtual_time() -> *const () {
    VIRTUAL_TIME_OWN.with(|c| c as *const _ as *const ())
}

struct FixedDtSlot;

impl FixedDtSlot {
    fn with<R>(&self, f: impl FnOnce(&Cell<Option<f64>>) -> R) -> R {
        let batch = crate::par_batch();
        if batch.is_null() {
            return FIXED_DT_OWN.with(f);
        }
        unsafe {
            crate::with_par_cell(
                batch,
                crate::par_cell(batch, crate::CELL_FIXED_DT) as *const Cell<Option<f64>>,
                f,
            )
        }
    }
}

struct VirtualTimeSlot;

impl VirtualTimeSlot {
    fn with<R>(&self, f: impl FnOnce(&Cell<f64>) -> R) -> R {
        let batch = crate::par_batch();
        if batch.is_null() {
            return VIRTUAL_TIME_OWN.with(f);
        }
        unsafe {
            crate::with_par_cell(
                batch,
                crate::par_cell(batch, crate::CELL_VIRTUAL_TIME) as *const Cell<f64>,
                f,
            )
        }
    }
}

const FIXED_DT: FixedDtSlot = FixedDtSlot;
const VIRTUAL_TIME: VirtualTimeSlot = VirtualTimeSlot;

/// The harness reproducibility override, read once from `AURORA_FIXED_DT`.
/// It WINS over any `set_fixed_dt` the program makes, so a game that requests
/// wall-clock in normal play still runs at a fixed step under verification.
fn env_fixed_dt() -> Option<f64> {
    use std::sync::OnceLock;
    static E: OnceLock<Option<f64>> = OnceLock::new();
    *E.get_or_init(|| {
        std::env::var("AURORA_FIXED_DT")
            .ok()
            .and_then(|v| v.parse::<f64>().ok())
            .filter(|d| *d > 0.0)
    })
}

/// The active fixed step (> 0) or 0.0 when wall-clock time is in effect.
/// Priority: the `AURORA_FIXED_DT` env override, then the program's
/// `set_fixed_dt`, then wall clock.
pub(crate) fn fixed_dt_override() -> f64 {
    if let Some(d) = env_fixed_dt() {
        return d;
    }
    FIXED_DT.with(|f| f.get().unwrap_or(0.0))
}

pub(crate) fn advance_virtual_time(dt: f64) {
    VIRTUAL_TIME.with(|t| t.set(t.get() + dt));
}

/// Seconds of virtual time accumulated under a fixed step (for offline audio
/// capture and telemetry timestamps).
pub fn virtual_time_seconds() -> f64 {
    VIRTUAL_TIME.with(|t| t.get())
}

/// `set_fixed_dt(dt)`: dt > 0 pins `frame_dt()` to exactly dt per call;
/// dt <= 0 restores wall-clock behavior. The `AURORA_FIXED_DT` env override,
/// when set, takes precedence over this (see `fixed_dt_override`).
#[no_mangle]
pub extern "C" fn aurora_set_fixed_dt(dt: f64) {
    FIXED_DT.with(|f| f.set(if dt > 0.0 { Some(dt) } else { None }));
}

// --- text file I/O ----------------------------------------------------------

/// # Safety
/// `ptr` must point to `len` initialized bytes.
unsafe fn arg_str(ptr: *const u8, len: i64) -> String {
    let s = unsafe { std::slice::from_raw_parts(ptr, len.max(0) as usize) };
    String::from_utf8_lossy(s).into_owned()
}

// --- process environment ----------------------------------------------------
//
// The program's OWN argument vector, which is not always the process's: under
// `aurorac run` the compiler owns `std::env::args()`, so it installs the
// program's vector (the source file it was asked to run, then everything after
// it) with `set_program_args`. Both paths therefore agree on argv[0] = the
// program as invoked and argv[1..] = its arguments.

static PROGRAM_ARGS: std::sync::OnceLock<Vec<String>> = std::sync::OnceLock::new();

/// Install the argument vector `sys_argc`/`sys_arg` report. Called by the
/// compiler driver before it runs a program; ignored if the vector is already
/// set, and never needed by an AOT executable, which owns the real one.
pub fn set_program_args(args: Vec<String>) {
    let _ = PROGRAM_ARGS.set(args);
}

fn program_args() -> &'static [String] {
    PROGRAM_ARGS.get_or_init(|| std::env::args().collect())
}

/// `sys_argc() -> i64`: how many arguments the program was invoked with,
/// including argv[0] (the program itself), so it is always at least 1.
#[no_mangle]
pub extern "C" fn aurora_sys_argc() -> i64 {
    program_args().len() as i64
}

/// `sys_arg(i) -> str`: the i-th argument, or "" when `i` is out of range in
/// either direction. Never reads out of bounds and never aborts, so a program
/// can probe for optional arguments without checking `sys_argc` first.
///
/// # Safety
/// `out` must be valid for writes of two `i64`s.
#[no_mangle]
pub unsafe extern "C" fn aurora_sys_arg(out: *mut i64, i: i64) {
    let args = program_args();
    let s = usize::try_from(i)
        .ok()
        .and_then(|i| args.get(i))
        .map(|s| s.as_str())
        .unwrap_or("");
    unsafe { crate::write_str(out, s.as_bytes().to_vec()) };
}

/// `sys_env(name) -> str`: an environment variable's value, or "" when it is
/// unset (or holds non-UTF-8). An empty value is reported as "" too, so use a
/// sentinel value rather than emptiness to mean "set".
///
/// # Safety
/// `ptr` must point to `len` initialized bytes. `out` must be valid for
/// writes of two `i64`s.
#[no_mangle]
pub unsafe extern "C" fn aurora_sys_env(out: *mut i64, ptr: *const u8, len: i64) {
    let v = std::env::var(arg_str(ptr, len)).unwrap_or_default();
    unsafe { crate::write_str(out, v.into_bytes()) };
}

/// `is_i64(s) -> 1|0`: whether `s` parses as a whole number.
///
/// The predicate half of parsing, and it exists so the fallback half cannot lie.
/// A `parse` that folds "not a number" into a returned 0 makes a corrupt save
/// file read as a level-0 character with no souls, confidently - the shape this
/// engine's own callers keep getting bitten by. Ask this first when the answer
/// has to be trusted; pass a fallback when it does not.
///
/// # Safety
/// `ptr` must point to `len` initialized bytes.
#[no_mangle]
pub unsafe extern "C" fn aurora_is_i64(ptr: *const u8, len: i64) -> i64 {
    i64::from(arg_str(ptr, len).trim().parse::<i64>().is_ok())
}

/// `parse_i64(s, fallback) -> i64`: `s` as a whole number, or `fallback`.
///
/// The fallback is an ARGUMENT rather than a built-in zero so the guess is
/// written at the call site, where a reader can see what "unparseable" was taken
/// to mean. Pair with `is_i64` when a wrong answer would be worse than none.
///
/// # Safety
/// `ptr` must point to `len` initialized bytes.
#[no_mangle]
pub unsafe extern "C" fn aurora_parse_i64(ptr: *const u8, len: i64, fallback: i64) -> i64 {
    arg_str(ptr, len).trim().parse::<i64>().unwrap_or(fallback)
}

/// `is_f64(s) -> 1|0`: whether `s` parses as a number. See `is_i64`.
///
/// # Safety
/// `ptr` must point to `len` initialized bytes.
#[no_mangle]
pub unsafe extern "C" fn aurora_is_f64(ptr: *const u8, len: i64) -> i64 {
    i64::from(arg_str(ptr, len).trim().parse::<f64>().is_ok())
}

/// `parse_f64(s, fallback) -> f64`: `s` as a number, or `fallback`. See
/// `parse_i64` for why the fallback is an argument.
///
/// # Safety
/// `ptr` must point to `len` initialized bytes.
#[no_mangle]
pub unsafe extern "C" fn aurora_parse_f64(ptr: *const u8, len: i64, fallback: f64) -> f64 {
    arg_str(ptr, len).trim().parse::<f64>().unwrap_or(fallback)
}

/// `read_file(path) -> str`: the file's contents, or "" if unreadable
/// (discriminate with `file_exists`). The string lives in the frame arena.
///
/// # Safety
/// `ptr` must point to `len` initialized bytes. `out` must be valid for
/// writes of two `i64`s.
#[no_mangle]
pub unsafe extern "C" fn aurora_read_file(out: *mut i64, ptr: *const u8, len: i64) {
    let bytes = std::fs::read(arg_str(ptr, len)).unwrap_or_default();
    unsafe { crate::write_str(out, bytes) };
}

/// `write_file(path, contents) -> 1|0`. Creates parent directories.
///
/// # Safety
/// `pp` must point to `pl` initialized bytes and `dp` to `dl`.
#[no_mangle]
pub unsafe extern "C" fn aurora_write_file(pp: *const u8, pl: i64, dp: *const u8, dl: i64) -> i64 {
    let path = arg_str(pp, pl);
    let data = unsafe { std::slice::from_raw_parts(dp, dl.max(0) as usize) };
    if let Some(parent) = std::path::Path::new(&path).parent() {
        if !parent.as_os_str().is_empty() {
            let _ = std::fs::create_dir_all(parent);
        }
    }
    std::fs::write(&path, data).is_ok() as i64
}

/// `file_exists(path) -> 1|0`.
///
/// # Safety
/// `ptr` must point to `len` initialized bytes.
#[no_mangle]
pub unsafe extern "C" fn aurora_file_exists(ptr: *const u8, len: i64) -> i64 {
    std::path::Path::new(&arg_str(ptr, len)).exists() as i64
}

// --- JSON -------------------------------------------------------------------

enum JNode {
    /// A node inside an immutable parsed document. `node` points into the tree
    /// owned by `root`; parsed values are never mutated, and the Rc keeps the
    /// tree alive for as long as any sub-handle exists, so the pointer is
    /// stable. O(1) child access with zero clones.
    Parsed { root: Rc<Value>, node: *const Value },
    /// A mutable value being built by the program (saves, telemetry).
    Owned(Value),
}

/// The `i64` an Aurora program holds for a JSON node.
type NodeId = aurora_slot::Key<JNode>;

thread_local! {
    /// Live JSON handles.
    ///
    /// This used to be a `Vec<Option<JNode>>` that only ever grew: `json_free`
    /// set a slot to `None` and nothing ever reused it, so a server parsing one
    /// document per request grew the index vector without bound even though it
    /// freed every handle it took. The slot store reuses a freed slot AND bumps
    /// its generation, so the spine plateaus and the freed handle still reads
    /// as invalid rather than as whatever document landed there next.
    static JSON: RefCell<aurora_slot::SlotMap<JNode>> =
        const { RefCell::new(aurora_slot::SlotMap::new()) };
}

fn push_node(n: JNode) -> i64 {
    // A key is never 0 (generation 0 is never issued), so 0 stays the
    // "invalid/absent" answer it has always been.
    JSON.with(|j| j.borrow_mut().insert(n).to_i64())
}

/// Run `f` on the value behind `h` (parsed nodes and owned values look the
/// same to readers). None for invalid/freed handles.
fn with_value<R>(h: i64, f: impl FnOnce(&Value) -> R) -> Option<R> {
    JSON.with(|j| {
        let j = j.borrow();
        match j.get(NodeId::from_i64(h)?)? {
            JNode::Parsed { node, .. } => Some(f(unsafe { &**node })),
            JNode::Owned(v) => Some(f(v)),
        }
    })
}

fn with_owned<R>(h: i64, f: impl FnOnce(&mut Value) -> R) -> Option<R> {
    JSON.with(|j| {
        let mut j = j.borrow_mut();
        match j.get_mut(NodeId::from_i64(h)?)? {
            JNode::Owned(v) => Some(f(v)),
            _ => None,
        }
    })
}

/// Wrap a child of `h` as a new handle: parsed children stay zero-copy
/// pointer nodes; owned children are cloned (builder trees are small).
fn child_handle(h: i64, pick: impl Fn(&Value) -> Option<&Value>) -> i64 {
    JSON.with(|j| {
        let node = {
            let j = j.borrow();
            match NodeId::from_i64(h).and_then(|k| j.get(k)) {
                Some(JNode::Parsed { root, node }) => {
                    pick(unsafe { &**node }).map(|c| JNode::Parsed {
                        root: root.clone(),
                        node: c,
                    })
                }
                Some(JNode::Owned(v)) => pick(v).map(|c| JNode::Owned(c.clone())),
                None => None,
            }
        };
        match node {
            Some(n) => j.borrow_mut().insert(n).to_i64(),
            None => 0,
        }
    })
}

/// `json_parse(text) -> handle` (0 on parse error, with a diagnostic).
///
/// # Safety
/// `ptr` must point to `len` initialized bytes.
#[no_mangle]
pub unsafe extern "C" fn aurora_json_parse(ptr: *const u8, len: i64) -> i64 {
    let text = arg_str(ptr, len);
    match serde_json::from_str::<Value>(&text) {
        Ok(v) => {
            let root = Rc::new(v);
            let node: *const Value = &*root;
            push_node(JNode::Parsed { root, node })
        }
        Err(e) => {
            eprintln!("aurora: json_parse: {e}");
            0
        }
    }
}

/// `json_load(path) -> handle`: read + parse a file (0 on failure).
///
/// # Safety
/// `ptr` must point to `len` initialized bytes.
#[no_mangle]
pub unsafe extern "C" fn aurora_json_load(ptr: *const u8, len: i64) -> i64 {
    let path = arg_str(ptr, len);
    let text = match std::fs::read_to_string(&path) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("aurora: json_load {path}: {e}");
            return 0;
        }
    };
    match serde_json::from_str::<Value>(&text) {
        Ok(v) => {
            let root = Rc::new(v);
            let node: *const Value = &*root;
            push_node(JNode::Parsed { root, node })
        }
        Err(e) => {
            eprintln!("aurora: json_load {path}: {e}");
            0
        }
    }
}

/// `json_get(h, key) -> handle` (0 if missing / not an object).
///
/// # Safety
/// `kp` must point to `kl` initialized bytes.
#[no_mangle]
pub unsafe extern "C" fn aurora_json_get(h: i64, kp: *const u8, kl: i64) -> i64 {
    let key = arg_str(kp, kl);
    child_handle(h, |v| v.get(&key))
}

/// `json_at(h, i) -> handle` (0 if out of range / not an array).
#[no_mangle]
pub extern "C" fn aurora_json_at(h: i64, i: i64) -> i64 {
    child_handle(h, |v| v.get(i.max(0) as usize))
}

/// `json_len(h)`: array length, object entry count, or string byte length.
#[no_mangle]
pub extern "C" fn aurora_json_len(h: i64) -> i64 {
    with_value(h, |v| match v {
        Value::Array(a) => a.len() as i64,
        Value::Object(o) => o.len() as i64,
        Value::String(s) => s.len() as i64,
        _ => 0,
    })
    .unwrap_or(0)
}

/// `json_num(h) -> f64` (numbers only; else 0.0).
#[no_mangle]
pub extern "C" fn aurora_json_num(h: i64) -> f64 {
    with_value(h, |v| v.as_f64().unwrap_or(0.0)).unwrap_or(0.0)
}

/// `json_int(h) -> i64` (truncates fractional numbers).
#[no_mangle]
pub extern "C" fn aurora_json_int(h: i64) -> i64 {
    with_value(h, |v| {
        v.as_i64()
            .unwrap_or_else(|| v.as_f64().unwrap_or(0.0) as i64)
    })
    .unwrap_or(0)
}

/// `json_bool(h) -> 1|0`.
#[no_mangle]
pub extern "C" fn aurora_json_bool(h: i64) -> i64 {
    with_value(h, |v| v.as_bool().unwrap_or(false) as i64).unwrap_or(0)
}

/// `json_str(h) -> str`: string contents ("" for non-strings - use
/// `json_to_str` to serialize any value).
///
/// # Safety
/// `out` must be valid for writes of two `i64`s.
#[no_mangle]
pub unsafe extern "C" fn aurora_json_str(out: *mut i64, h: i64) {
    let s = with_value(h, |v| v.as_str().map(|s| s.to_string()).unwrap_or_default())
        .unwrap_or_default();
    unsafe { crate::write_str(out, s.into_bytes()) };
}

/// `json_kind(h)`: -1 invalid, 0 null, 1 bool, 2 number, 3 string, 4 array,
/// 5 object.
#[no_mangle]
pub extern "C" fn aurora_json_kind(h: i64) -> i64 {
    with_value(h, |v| match v {
        Value::Null => 0,
        Value::Bool(_) => 1,
        Value::Number(_) => 2,
        Value::String(_) => 3,
        Value::Array(_) => 4,
        Value::Object(_) => 5,
    })
    .unwrap_or(-1)
}

/// `json_has(h, key) -> 1|0`.
///
/// # Safety
/// `kp` must point to `kl` initialized bytes.
#[no_mangle]
pub unsafe extern "C" fn aurora_json_has(h: i64, kp: *const u8, kl: i64) -> i64 {
    let key = arg_str(kp, kl);
    with_value(h, |v| v.get(&key).is_some() as i64).unwrap_or(0)
}

/// `json_key(h, i) -> str`: the i-th key of an object (document order).
///
/// # Safety
/// `out` must be valid for writes of two `i64`s.
#[no_mangle]
pub unsafe extern "C" fn aurora_json_key(out: *mut i64, h: i64, i: i64) {
    let s = with_value(h, |v| match v {
        Value::Object(o) => o.keys().nth(i.max(0) as usize).cloned().unwrap_or_default(),
        _ => String::new(),
    })
    .unwrap_or_default();
    unsafe { crate::write_str(out, s.into_bytes()) };
}

/// `json_free(h)`: release a handle (and, for parsed roots, its share of the
/// document). Reading a freed handle yields kind -1, never a crash.
///
/// The slot is reused by the next handle, so parse-and-free in a loop occupies
/// bounded memory, and its generation is bumped, so `h` keeps reading as
/// invalid rather than as the next document that lands there.
#[no_mangle]
pub extern "C" fn aurora_json_free(h: i64) {
    JSON.with(|j| {
        if let Some(k) = NodeId::from_i64(h) {
            j.borrow_mut().remove(k);
        }
    });
}

// --- JSON building (saves / telemetry) --------------------------------------

/// `json_new_obj() -> handle` (mutable).
#[no_mangle]
pub extern "C" fn aurora_json_new_obj() -> i64 {
    push_node(JNode::Owned(Value::Object(Default::default())))
}

/// `json_new_arr() -> handle` (mutable).
#[no_mangle]
pub extern "C" fn aurora_json_new_arr() -> i64 {
    push_node(JNode::Owned(Value::Array(Vec::new())))
}

fn snapshot(h: i64) -> Option<Value> {
    with_value(h, |v| v.clone())
}

/// `json_set(h, key, child)`: store a deep copy of `child` under `key`.
///
/// # Safety
/// `kp` must point to `kl` initialized bytes.
#[no_mangle]
pub unsafe extern "C" fn aurora_json_set(h: i64, kp: *const u8, kl: i64, child: i64) {
    let key = arg_str(kp, kl);
    let Some(v) = snapshot(child) else { return };
    with_owned(h, |o| {
        if let Value::Object(map) = o {
            map.insert(key, v);
        }
    });
}

/// `json_set_num(h, key, x)`.
///
/// # Safety
/// `kp` must point to `kl` initialized bytes.
#[no_mangle]
pub unsafe extern "C" fn aurora_json_set_num(h: i64, kp: *const u8, kl: i64, x: f64) {
    let key = arg_str(kp, kl);
    with_owned(h, |o| {
        if let Value::Object(map) = o {
            map.insert(key, num_value(x));
        }
    });
}

/// `json_set_str(h, key, s)`.
///
/// # Safety
/// `kp` must point to `kl` initialized bytes and `sp` to `sl`.
#[no_mangle]
pub unsafe extern "C" fn aurora_json_set_str(
    h: i64,
    kp: *const u8,
    kl: i64,
    sp: *const u8,
    sl: i64,
) {
    let key = arg_str(kp, kl);
    let s = arg_str(sp, sl);
    with_owned(h, |o| {
        if let Value::Object(map) = o {
            map.insert(key, Value::String(s));
        }
    });
}

/// `json_set_bool(h, key, b)`.
///
/// # Safety
/// `kp` must point to `kl` initialized bytes.
#[no_mangle]
pub unsafe extern "C" fn aurora_json_set_bool(h: i64, kp: *const u8, kl: i64, b: i64) {
    let key = arg_str(kp, kl);
    with_owned(h, |o| {
        if let Value::Object(map) = o {
            map.insert(key, Value::Bool(b != 0));
        }
    });
}

/// `json_push(h, child)`: append a deep copy of `child` to an array.
#[no_mangle]
pub extern "C" fn aurora_json_push(h: i64, child: i64) {
    let Some(v) = snapshot(child) else { return };
    with_owned(h, |o| {
        if let Value::Array(a) = o {
            a.push(v);
        }
    });
}

/// `json_push_num(h, x)`.
#[no_mangle]
pub extern "C" fn aurora_json_push_num(h: i64, x: f64) {
    with_owned(h, |o| {
        if let Value::Array(a) = o {
            a.push(num_value(x));
        }
    });
}

/// `json_push_str(h, s)`.
///
/// # Safety
/// `sp` must point to `sl` initialized bytes.
#[no_mangle]
pub unsafe extern "C" fn aurora_json_push_str(h: i64, sp: *const u8, sl: i64) {
    let s = arg_str(sp, sl);
    with_owned(h, |o| {
        if let Value::Array(a) = o {
            a.push(Value::String(s));
        }
    });
}

/// Whole-valued floats serialize as integers (7 not 7.0) so counters and ids
/// round-trip exactly.
fn num_value(x: f64) -> Value {
    if x.is_finite() && x == x.trunc() && x.abs() < 9.0e15 {
        Value::Number((x as i64).into())
    } else {
        serde_json::Number::from_f64(x)
            .map(Value::Number)
            .unwrap_or(Value::Null)
    }
}

/// `json_to_str(h) -> str`: pretty-printed JSON of any handle.
///
/// # Safety
/// `out` must be valid for writes of two `i64`s.
#[no_mangle]
pub unsafe extern "C" fn aurora_json_to_str(out: *mut i64, h: i64) {
    let s =
        with_value(h, |v| serde_json::to_string_pretty(v).unwrap_or_default()).unwrap_or_default();
    unsafe { crate::write_str(out, s.into_bytes()) };
}

/// `json_write(h, path) -> 1|0`: pretty-print to a file (parent dirs created).
///
/// # Safety
/// `pp` must point to `pl` initialized bytes.
#[no_mangle]
pub unsafe extern "C" fn aurora_json_write(h: i64, pp: *const u8, pl: i64) -> i64 {
    let path = arg_str(pp, pl);
    let Some(text) = with_value(h, |v| serde_json::to_string_pretty(v).unwrap_or_default()) else {
        return 0;
    };
    if let Some(parent) = std::path::Path::new(&path).parent() {
        if !parent.as_os_str().is_empty() {
            let _ = std::fs::create_dir_all(parent);
        }
    }
    std::fs::write(&path, text).is_ok() as i64
}

#[cfg(test)]
mod tests {
    use super::*;

    fn str_of(f: impl Fn(*mut i64)) -> String {
        let mut out = [0i64; 2];
        f(out.as_mut_ptr());
        let s = unsafe { std::slice::from_raw_parts(out[0] as *const u8, out[1] as usize) };
        String::from_utf8_lossy(s).into_owned()
    }

    #[test]
    fn rng_is_deterministic_and_reseedable() {
        aurora_srand(42);
        let a: Vec<f64> = (0..5).map(|_| aurora_rand()).collect();
        aurora_srand(42);
        let b: Vec<f64> = (0..5).map(|_| aurora_rand()).collect();
        assert_eq!(a, b, "same seed must give the same sequence");
        assert!(a.iter().all(|x| (0.0..1.0).contains(x)));
        aurora_srand(43);
        let c: Vec<f64> = (0..5).map(|_| aurora_rand()).collect();
        assert_ne!(a, c, "different seeds must diverge");
        for _ in 0..1000 {
            let v = aurora_rand_int(3, 7);
            assert!((3..=7).contains(&v));
        }
        assert_eq!(aurora_rand_int(5, 5), 5);
    }

    #[test]
    fn fixed_dt_overrides_and_restores() {
        crate::end_frame_dt();
        aurora_set_fixed_dt(1.0 / 120.0);
        let before = virtual_time_seconds();
        let d1 = crate::aurora_frame_dt();
        let d2 = crate::aurora_frame_dt();
        assert_eq!(d1, 1.0 / 120.0);
        assert_eq!(d2, 1.0 / 120.0);
        // ONE step per frame, however many callers ask. The clock belongs to the
        // frame, not to the question - it advanced per CALL, so a frame that
        // asked twice (the game's does: once for the frame, once for the camera)
        // ran the virtual clock at double speed, and everything stamped against
        // it drifted by however many callers there happened to be.
        let one = virtual_time_seconds() - before;
        assert!(
            (one - 1.0 / 120.0).abs() < 1e-9,
            "two asks in one frame advance the clock once, got {one}"
        );
        // And the next frame does advance it again - the cache is per frame, not
        // a latch that stops the clock forever.
        crate::end_frame_dt();
        let _ = crate::aurora_frame_dt();
        let two = virtual_time_seconds() - before;
        assert!(
            (two - 2.0 / 120.0).abs() < 1e-9,
            "a second frame is a second step, got {two}"
        );
        crate::end_frame_dt();
        aurora_set_fixed_dt(0.0);
        let d3 = crate::aurora_frame_dt();
        assert!(d3 > 0.0 && d3 <= 0.1, "wall clock restored, got {d3}");
    }

    #[test]
    fn json_parse_navigate_and_types() {
        // SAFETY: every pointer below is `as_ptr()` of a live local byte string or
        // `String`, and every length is that value's real length; `o` is a live
        // 2-slot `[i64; 2]` owned by `str_of`.
        unsafe {
            let text =
                br#"{"name":"grunt","hp":40,"fast":true,"tags":["melee","dumb"],"pos":{"x":1.5}}"#;
            let h = aurora_json_parse(text.as_ptr(), text.len() as i64);
            assert!(h > 0);
            assert_eq!(aurora_json_kind(h), 5);
            let name = aurora_json_get(h, b"name".as_ptr(), 4);
            assert_eq!(aurora_json_kind(name), 3);
            assert_eq!(str_of(|o| aurora_json_str(o, name)), "grunt");
            let hp = aurora_json_get(h, b"hp".as_ptr(), 2);
            assert_eq!(aurora_json_int(hp), 40);
            assert_eq!(aurora_json_num(hp), 40.0);
            let fast = aurora_json_get(h, b"fast".as_ptr(), 4);
            assert_eq!(aurora_json_bool(fast), 1);
            let tags = aurora_json_get(h, b"tags".as_ptr(), 4);
            assert_eq!(aurora_json_kind(tags), 4);
            assert_eq!(aurora_json_len(tags), 2);
            let t1 = aurora_json_at(tags, 1);
            assert_eq!(str_of(|o| aurora_json_str(o, t1)), "dumb");
            let pos = aurora_json_get(h, b"pos".as_ptr(), 3);
            let x = aurora_json_get(pos, b"x".as_ptr(), 1);
            assert_eq!(aurora_json_num(x), 1.5);
            // Missing key / out-of-range / freed handles degrade to 0 / -1.
            assert_eq!(aurora_json_get(h, b"nope".as_ptr(), 4), 0);
            assert_eq!(aurora_json_at(tags, 99), 0);
            assert_eq!(aurora_json_has(h, b"hp".as_ptr(), 2), 1);
            assert_eq!(aurora_json_has(h, b"nope".as_ptr(), 4), 0);
            aurora_json_free(h);
            assert_eq!(aurora_json_kind(h), -1);
            // Sub-handles survive freeing the root handle (Rc keeps the tree).
            assert_eq!(aurora_json_int(hp), 40);
        }
    }

    /// Handle slots ever allocated by the JSON store. The leak is stated in
    /// these units because the payload was already freed correctly: it was the
    /// index vector holding it that grew forever.
    fn spine_slots() -> usize {
        JSON.with(|j| j.borrow().slot_count())
    }

    /// Parse-and-free in a loop - a server handling one JSON request per
    /// connection - must occupy bounded memory.
    ///
    /// It did not: `push_node` only ever appended, and `json_free` wrote `None`
    /// into a slot that nothing reused, so the spine grew by one entry per
    /// parse (plus one per child handle) for the life of the process even
    /// though the program freed every handle it was given.
    #[test]
    fn parse_and_free_in_a_loop_leaves_the_handle_spine_bounded() {
        // SAFETY: `doc` is a live local byte string and the length passed is
        // its own; the key pointers are literals with their own lengths.
        unsafe {
            let doc = br#"{"hp":40,"tags":["melee","dumb"],"pos":{"x":1.5}}"#;
            let cycle = || {
                let h = aurora_json_parse(doc.as_ptr(), doc.len() as i64);
                assert!(h > 0);
                let tags = aurora_json_get(h, b"tags".as_ptr(), 4);
                let first = aurora_json_at(tags, 0);
                let pos = aurora_json_get(h, b"pos".as_ptr(), 3);
                assert_eq!(aurora_json_len(tags), 2);
                for handle in [first, tags, pos, h] {
                    aurora_json_free(handle);
                }
            };
            // One cycle first, so the slots it needs are already allocated and
            // the measurement below is about REUSE rather than about warm-up.
            cycle();
            let after_one = spine_slots();
            for _ in 0..500 {
                cycle();
            }
            assert_eq!(
                spine_slots(),
                after_one,
                "the handle spine grew across 500 parse/free cycles"
            );
            assert_eq!(JSON.with(|j| j.borrow().len()), 0, "handles leaked");
        }
    }

    /// A freed handle must keep reading as invalid even once its slot has been
    /// handed to a different document. With a plain index it would read the new
    /// one - a save file answering with another save file's fields.
    #[test]
    fn a_freed_json_handle_never_aliases_the_document_that_reuses_its_slot() {
        // SAFETY: both pointers are `as_ptr()` of live local byte strings and
        // the lengths passed are their own.
        unsafe {
            let a = br#"{"hp":40}"#;
            let b = br#"{"hp":99}"#;
            let first = aurora_json_parse(a.as_ptr(), a.len() as i64);
            assert_eq!(
                aurora_json_int(aurora_json_get(first, b"hp".as_ptr(), 2)),
                40
            );
            aurora_json_free(first);
            let second = aurora_json_parse(b.as_ptr(), b.len() as i64);
            assert_ne!(first, second, "the reused slot must change the handle");
            assert_eq!(
                aurora_json_kind(first),
                -1,
                "a freed handle read a live doc"
            );
            assert_eq!(aurora_json_get(first, b"hp".as_ptr(), 2), 0);
            assert_eq!(aurora_json_len(first), 0);
            assert_eq!(
                aurora_json_int(aurora_json_get(second, b"hp".as_ptr(), 2)),
                99
            );
            // Freeing twice must not take the document that took the slot.
            aurora_json_free(first);
            assert_eq!(
                aurora_json_kind(second),
                5,
                "a double free hit the wrong doc"
            );
            // 0 has always meant "invalid/absent" and must still not resolve.
            assert_eq!(aurora_json_kind(0), -1);
            assert_eq!(aurora_json_kind(-1), -1);
            aurora_json_free(0);
            aurora_json_free(-1);
            assert_eq!(aurora_json_kind(second), 5);
        }
    }

    #[test]
    fn json_build_and_roundtrip() {
        // SAFETY: every pointer below is `as_ptr()` of a live local byte string or
        // `String`, and every length is that value's real length; `o` is a live
        // 2-slot `[i64; 2]` owned by `str_of`.
        unsafe {
            let obj = aurora_json_new_obj();
            aurora_json_set_str(obj, b"name".as_ptr(), 4, b"save1".as_ptr(), 5);
            aurora_json_set_num(obj, b"hp".as_ptr(), 2, 77.0);
            aurora_json_set_num(obj, b"x".as_ptr(), 1, 1.25);
            aurora_json_set_bool(obj, b"hard".as_ptr(), 4, 1);
            let arr = aurora_json_new_arr();
            aurora_json_push_num(arr, 3.0);
            aurora_json_push_str(arr, b"sword".as_ptr(), 5);
            aurora_json_set(obj, b"items".as_ptr(), 5, arr);
            let text = str_of(|o| aurora_json_to_str(o, obj));
            let h = aurora_json_parse(text.as_bytes().as_ptr(), text.len() as i64);
            assert!(h > 0, "round-trip parse of: {text}");
            assert_eq!(aurora_json_int(aurora_json_get(h, b"hp".as_ptr(), 2)), 77);
            assert_eq!(aurora_json_num(aurora_json_get(h, b"x".as_ptr(), 1)), 1.25);
            let items = aurora_json_get(h, b"items".as_ptr(), 5);
            assert_eq!(aurora_json_len(items), 2);
            assert_eq!(aurora_json_int(aurora_json_at(items, 0)), 3);
            // Whole numbers serialize without a fractional suffix.
            assert!(text.contains("\"hp\": 77"), "got: {text}");
        }
    }

    #[test]
    fn file_roundtrip_and_exists() {
        // SAFETY: `p` is a live local `String` and the lengths passed are its own
        // and the literal's; `out` is a live local `[i64; 2]`.
        unsafe {
            let dir = std::env::temp_dir().join("aurora_data_test");
            let path = dir.join("nested").join("t.txt");
            let p = path.to_string_lossy().into_owned();
            assert_eq!(aurora_file_exists(p.as_ptr(), p.len() as i64), 0);
            assert_eq!(
                aurora_write_file(p.as_ptr(), p.len() as i64, b"hello".as_ptr(), 5),
                1,
                "write_file creates parent dirs"
            );
            assert_eq!(aurora_file_exists(p.as_ptr(), p.len() as i64), 1);
            let mut out = [0i64; 2];
            aurora_read_file(out.as_mut_ptr(), p.as_ptr(), p.len() as i64);
            let s = std::slice::from_raw_parts(out[0] as *const u8, out[1] as usize);
            assert_eq!(s, b"hello");
            let _ = std::fs::remove_dir_all(&dir);
        }
    }
}
