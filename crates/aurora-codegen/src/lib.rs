//! Native code generation via Cranelift — the **compiled** execution path.
//!
//! Aurora is a compiled language: this lowers programs to real machine code and
//! runs them, with no interpreter. It compiles:
//!
//! * scalar functions (`i*`/`u*`/`bool`/`f32`/`f64`) with arithmetic, comparisons,
//!   `&&`/`||`, numeric `as` casts, `if` (value + early-return), `while`/`for`,
//!   assignment, recursion, calls, and native math intrinsics;
//! * **aggregates** — structs, tuples, and fixed arrays — as stack-allocated
//!   memory, with field/index access, mutation, destructuring, and array
//!   iteration;
//! * native `print`/`println` via linked host functions.
//!
//! Values are either *scalars* (in registers) or *aggregates* (a pointer to a
//! stack slot). Aggregate fields/elements occupy 8-byte slots. Constructs still
//! ahead (closures, enums, ECS) cause the function to be stubbed; the tree-
//! walking interpreter covers those until their codegen lands.

use std::collections::{HashMap, HashSet};

use aurora_ast::{
    BinOp, Block, Expr, ExprKind, FieldAccess, ItemKind, Module as AstModule, Pat, PatKind, Stmt, StructBody,
    TypeKind, UnOp,
};
use aurora_lexer::FloatTy;

use cranelift::codegen;
use cranelift::prelude::*;
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{DataDescription, FuncId, Linkage, Module};
use cranelift_object::{ObjectBuilder, ObjectModule};

const SLOT: u32 = 8; // bytes per aggregate field/element

// Host functions live in the `aurora-runtime` crate as real C-ABI symbols, so
// the same code backs both the JIT (addresses registered below) and AOT
// executables (the emitted object's `aurora_*` imports resolve against it).
// Each is named by `aurora-abi`'s table and reached as `aurora_runtime::<sym>`.

/// A compiled value's type. Scalars live in registers; aggregates are pointers
/// to stack memory (each field/element an 8-byte slot of scalars only).
#[derive(Clone, PartialEq, Debug)]
enum Cty {
    I64,
    F32,
    F64,
    /// A string value: a 16-byte `[data_ptr, len]` aggregate.
    Str,
    /// A trait object `dyn Trait`: a 16-byte `[data_ptr, type_id]` fat pointer.
    /// Method calls dispatch dynamically on `type_id`.
    Dyn(String),
    Struct(String),
    Enum(String),
    Tuple(Vec<Cty>),
    Array(Box<Cty>, usize),
    /// A closure value (a `[fn_ptr, env_ptr]` pointer pair): its parameter types
    /// and return type. Args/captures cross the call as raw 64-bit slots and are
    /// reinterpreted to/from `f64` at the boundary (see the closure ABI in codegen).
    Fn(Vec<Cty>, Box<Cty>),
}

impl Cty {
    fn is_scalar(&self) -> bool {
        matches!(self, Cty::I64 | Cty::F32 | Cty::F64)
    }
    /// The cranelift type used to hold this value (aggregates are pointers).
    fn clif(&self, ptr: Type) -> Type {
        match self {
            Cty::I64 => types::I64,
            Cty::F32 => types::F32,
            Cty::F64 => types::F64,
            _ => ptr,
        }
    }
}

/// An enum's memory layout: a tag (slot 0) + payload slots. Each variant's
/// fields occupy slots `1..`. `slots` = 1 + max variant arity.
struct EnumLayout {
    variants: Vec<EnumVariant>,
    slots: usize,
}
struct EnumVariant {
    name: String,
    /// (optional field name for struct variants, type) in slot order.
    fields: Vec<(Option<String>, Cty)>,
}

/// A closure's typed signature for the bitcasting ABI: the `Cty` of each
/// parameter and captured variable (in order), and the return `Cty`.
#[derive(Clone)]
struct ClosureSig {
    params: Vec<Cty>,
    captures: Vec<Cty>,
}

/// Result of translating an expression: a typed value, or a diverging path.
enum Term {
    Val(Value, Cty),
    Diverged,
}

struct FnInfo {
    id: FuncId,
    /// Codegen types of the (non-self for methods are included) parameters.
    params: Vec<Cty>,
    ret: Cty,
    /// True if the return is an aggregate, passed via a leading sret pointer.
    sret: bool,
}

/// Whether a type is an aggregate (passed/returned by pointer + sret).
fn is_aggregate(c: &Cty) -> bool {
    !c.is_scalar()
}

/// Reinterpret an 8-byte value as another 8-byte type (via a stack slot, so it
/// doesn't depend on the exact `bitcast` API). Used by the closure ABI to move
/// `f64` payloads through `i64` argument/capture/return slots without changing
/// their bits.
fn reinterpret(b: &mut FunctionBuilder, v: Value, to: Type) -> Value {
    let slot = b.create_sized_stack_slot(StackSlotData::new(StackSlotKind::ExplicitSlot, 8, 3));
    b.ins().stack_store(v, slot, 0);
    b.ins().stack_load(to, slot, 0)
}

/// `f64`/`f32` → its raw bits in an `i64` slot for the closure ABI; other
/// scalars/pointers pass unchanged (they already occupy an i64-sized slot). For
/// `f32` the low 4 bytes carry the value and the high bytes are unused.
fn to_i64_bits(b: &mut FunctionBuilder, v: Value, cty: &Cty) -> Value {
    if matches!(cty, Cty::F32 | Cty::F64) {
        reinterpret(b, v, types::I64)
    } else {
        v
    }
}

/// Raw `i64` bits → `f64`/`f32` (or unchanged for other scalars/pointers).
fn from_i64_bits(b: &mut FunctionBuilder, raw: Value, cty: &Cty) -> Value {
    match cty {
        Cty::F64 => reinterpret(b, raw, types::F64),
        Cty::F32 => reinterpret(b, raw, types::F32),
        _ => raw,
    }
}

/// Infer the type of an unannotated closure parameter `name` from how the body
/// uses it: in `name <op> other` the parameter shares `other`'s (known) scalar
/// type; passed as a function argument it takes that parameter's type. `scope`
/// holds the closure's captures/outer variables (not the parameter itself).
/// Returns `None` when the use doesn't pin it (caller defaults to `i64`).
fn infer_param_cty(name: &str, e: &Expr, scope: &HashMap<String, Cty>, env: &Env) -> Option<Cty> {
    let is_p = |x: &Expr| {
        matches!(&x.kind, ExprKind::Path(p) if p.segments.len() == 1 && p.segments[0].ident.name == name)
    };
    match &e.kind {
        ExprKind::Binary(_, a, c) => {
            if is_p(a) {
                if let Some(t) = infer_cty(c, scope, env) {
                    if t.is_scalar() {
                        return Some(t);
                    }
                }
            }
            if is_p(c) {
                if let Some(t) = infer_cty(a, scope, env) {
                    if t.is_scalar() {
                        return Some(t);
                    }
                }
            }
            infer_param_cty(name, a, scope, env).or_else(|| infer_param_cty(name, c, scope, env))
        }
        ExprKind::Call { callee, args, .. } => {
            if let ExprKind::Path(p) = &callee.kind {
                if p.segments.len() == 1 {
                    if let Some(info) = env.fns.get(&p.segments[0].ident.name) {
                        for (i, a) in args.iter().enumerate() {
                            if is_p(&a.value) {
                                if let Some(t) = info.params.get(i) {
                                    if t.is_scalar() {
                                        return Some(t.clone());
                                    }
                                }
                            }
                        }
                    }
                }
            }
            args.iter().find_map(|a| infer_param_cty(name, &a.value, scope, env))
        }
        ExprKind::Paren(x)
        | ExprKind::Unary(_, x)
        | ExprKind::Cast(x, _)
        | ExprKind::Field { base: x, .. }
        | ExprKind::Try(x)
        | ExprKind::Region { value: x, .. }
        | ExprKind::Despawn(x) => infer_param_cty(name, x, scope, env),
        ExprKind::Assign(_, a, c)
        | ExprKind::Index { base: a, index: c }
        | ExprKind::Pipe { value: a, func: c } => {
            infer_param_cty(name, a, scope, env).or_else(|| infer_param_cty(name, c, scope, env))
        }
        ExprKind::If(ifx) => infer_param_cty(name, &ifx.cond, scope, env)
            .or_else(|| ifx.then_branch.tail.as_ref().and_then(|t| infer_param_cty(name, t, scope, env)))
            .or_else(|| ifx.else_branch.as_ref().and_then(|e| infer_param_cty(name, e, scope, env))),
        ExprKind::Block(blk) | ExprKind::Unsafe(blk) | ExprKind::Loop(blk) => {
            blk.stmts
                .iter()
                .find_map(|s| match s {
                    Stmt::Expr(e) | Stmt::Defer(e) => infer_param_cty(name, e, scope, env),
                    Stmt::Let(l) => l.init.as_ref().and_then(|e| infer_param_cty(name, e, scope, env)),
                })
                .or_else(|| blk.tail.as_ref().and_then(|t| infer_param_cty(name, t, scope, env)))
        }
        ExprKind::Match { scrutinee, arms } => infer_param_cty(name, scrutinee, scope, env)
            .or_else(|| arms.iter().find_map(|a| infer_param_cty(name, &a.body, scope, env))),
        _ => None,
    }
}

/// Best-effort inference of an expression's `Cty` from a name→`Cty` scope, used
/// to learn a closure's return type at its construction site. Returns `None`
/// when uncertain — the caller then falls back to the plain all-i64 closure path.
fn infer_cty(e: &Expr, scope: &HashMap<String, Cty>, env: &Env) -> Option<Cty> {
    use aurora_ast::BinOp;
    match &e.kind {
        ExprKind::Int(..) | ExprKind::Bool(_) => Some(Cty::I64),
        ExprKind::Float(..) => Some(Cty::F64),
        ExprKind::Str(_) => Some(Cty::Str),
        ExprKind::Paren(x) => infer_cty(x, scope, env),
        ExprKind::Cast(_, ty) => Some(ty_to_cty(&ty.kind)),
        ExprKind::Path(p) if p.segments.len() == 1 => {
            scope.get(&p.segments[0].ident.name).cloned()
        }
        ExprKind::Unary(_, x) => infer_cty(x, scope, env),
        ExprKind::Binary(op, a, c) => match op {
            BinOp::Eq | BinOp::Ne | BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge
            | BinOp::And | BinOp::Or => Some(Cty::I64),
            _ => match (infer_cty(a, scope, env), infer_cty(c, scope, env)) {
                (Some(Cty::F64), _) | (_, Some(Cty::F64)) => Some(Cty::F64),
                (Some(Cty::F32), _) | (_, Some(Cty::F32)) => Some(Cty::F32),
                (Some(Cty::I64), Some(Cty::I64)) => Some(Cty::I64),
                _ => None,
            },
        },
        ExprKind::Call { callee, .. } => {
            if let ExprKind::Path(p) = &callee.kind {
                let name = &p.segments.last()?.ident.name;
                if let Some(info) = env.fns.get(name) {
                    return Some(info.ret.clone());
                }
            }
            None
        }
        ExprKind::Field { base, field: FieldAccess::Named(f) } => {
            if let Cty::Struct(sname) = infer_cty(base, scope, env)? {
                let fields = env.structs.get(&sname)?;
                return fields.iter().find(|(n, _)| n == &f.name).map(|(_, c)| c.clone());
            }
            None
        }
        ExprKind::If(ifx) => ifx.then_branch.tail.as_ref().and_then(|t| infer_cty(t, scope, env)),
        ExprKind::Block(blk) | ExprKind::Unsafe(blk) => {
            blk.tail.as_ref().and_then(|t| infer_cty(t, scope, env))
        }
        _ => None,
    }
}

struct Env {
    fns: HashMap<String, FnInfo>,
    hosts: HashMap<&'static str, FuncId>,
    /// Struct name -> ordered (field, type).
    structs: HashMap<String, Vec<(String, Cty)>>,
    /// Enum name -> layout.
    enums: HashMap<String, EnumLayout>,
    /// Top-level `const` name -> its initializer expression. A const is not a
    /// runtime global: each use lowers the initializer inline, so a literal folds
    /// to an immediate (no load, and no initialization order to get wrong) and a
    /// const is readable from every function.
    consts: HashMap<String, Expr>,
    /// Consts whose initializer is currently being lowered, so `const A = B`
    /// alongside `const B = A` is a clear error instead of unbounded recursion.
    /// `RefCell` because it is used while bodies compile (like `closure_sigs`).
    const_stack: std::cell::RefCell<std::collections::HashSet<String>>,
    /// (receiver type, method name) -> compiled function key.
    methods: HashMap<(String, String), String>,
    /// Closure expression span -> lambda-lifted function name.
    closures: HashMap<aurora_span::Span, String>,
    /// Lambda name -> ordered captured variable names.
    lambda_captures: HashMap<String, Vec<String>>,
    /// Lambda name -> typed closure signature, recorded at the construction site
    /// (where capture types are known) when the closure involves `f64`. Drives
    /// the bitcasting closure ABI in `compile_lambda`; absent => the plain
    /// all-i64 path. `RefCell` because it's filled while bodies compile.
    closure_sigs: std::cell::RefCell<HashMap<String, ClosureSig>>,
    /// Systems in declaration order (compiled fn keys), for `run_systems()`.
    system_order: Vec<String>,
    /// `run_systems()` schedule: ordered layers of indices into `system_order`.
    /// A layer with one system runs sequentially; a multi-system layer runs its
    /// systems concurrently (they are provably non-conflicting and unordered).
    system_layers: Vec<Vec<usize>>,
    ptr_ty: Type,
    /// When true, emit native debug hooks (statement-line + variable reporting).
    debug: bool,
    /// When true, emit profiler hooks (per-function enter/exit timing).
    profile: bool,
    /// Trait name -> concrete types implementing it (for `dyn Trait` dispatch).
    trait_types: HashMap<String, Vec<String>>,
    /// Names of `@extern` (foreign C-ABI) functions, so calls can marshal
    /// aggregate arguments to C's packed layout.
    extern_fns: std::collections::HashSet<String>,
    /// Byte offset of each source line start, for mapping spans → line numbers
    /// at compile time (empty unless debugging).
    line_starts: Vec<u32>,
}

impl Env {
    /// 1-based source line containing byte offset `off` (0 if unknown).
    fn line_of(&self, off: u32) -> u32 {
        if self.line_starts.is_empty() {
            return 0;
        }
        match self.line_starts.binary_search(&off) {
            Ok(i) => (i + 1) as u32,
            Err(i) => i as u32, // i = count of starts <= off
        }
    }

    /// Convert an AST type to a `Cty`, classifying known enum names as enums
    /// (not structs) so sizing/sret are correct.
    fn cty(&self, kind: &TypeKind) -> Cty {
        let c = ty_to_cty(kind);
        let names: HashSet<String> = self.enums.keys().cloned().collect();
        fix_enums(c, &names)
    }
}

// Builtin names come from `aurora-abi` (re-exported by `aurora-ast`) so the
// front end (which must not report them as unresolved names) and this backend
// (which lowers them to runtime calls) share one list.
use aurora_ast::builtin_names;

/// Byte size of a type (always a multiple of 8). Aggregates lay out their
/// fields/elements contiguously, so nesting is supported.
fn byte_size(env: &Env, cty: &Cty) -> u32 {
    match cty {
        Cty::I64 | Cty::F32 | Cty::F64 => 8,
        Cty::Str => 16,    // [data_ptr, len]
        Cty::Dyn(_) => 16, // [data_ptr, type_id]
        Cty::Struct(n) => env
            .structs
            .get(n)
            .map(|fs| fs.iter().map(|(_, c)| byte_size(env, c)).sum())
            .unwrap_or(8),
        Cty::Tuple(ts) => ts.iter().map(|c| byte_size(env, c)).sum(),
        Cty::Array(e, n) => *n as u32 * byte_size(env, e),
        Cty::Enum(n) => {
            let payload = env
                .enums
                .get(n)
                .map(|e| {
                    e.variants
                        .iter()
                        .map(|v| v.fields.iter().map(|(_, c)| byte_size(env, c)).sum::<u32>())
                        .max()
                        .unwrap_or(0)
                })
                .unwrap_or(0);
            8 + payload // tag + largest payload
        }
        Cty::Fn(..) => 16, // [fn_ptr, env_ptr]
    }
}

/// Byte offset of a struct field, plus its type.
fn struct_field(env: &Env, name: &str, field: &str) -> Option<(u32, Cty)> {
    let layout = env.structs.get(name)?;
    let mut off = 0u32;
    for (fname, fcty) in layout {
        if fname == field {
            return Some((off, fcty.clone()));
        }
        off += byte_size(env, fcty);
    }
    None
}

/// Number of 8-byte slots an aggregate occupies (for sret copies).
fn agg_slots(env: &Env, cty: &Cty) -> usize {
    (byte_size(env, cty) / 8) as usize
}

/// Stable id for a component type (FNV-1a of its name).
fn comp_id(name: &str) -> i64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in name.bytes() {
        h ^= byte as u64;
        h = h.wrapping_mul(0x100_0000_01b3);
    }
    h as i64
}

impl Env {
    /// If `path` is `Enum::Variant`, return (enum name, variant index).
    ///
    /// The enum may be reached through a module path (`m::E::Variant`, or
    /// `a::b::E::Variant` for a nested module): module flattening mangles the
    /// declaration's name to `m::E`, so every segment but the last joins back into
    /// the enum name. A path that names no known enum yields `None`, which leaves
    /// it to be resolved as an ordinary call/path.
    fn enum_variant(&self, path: &aurora_ast::Path) -> Option<(String, usize)> {
        if path.segments.len() < 2 {
            return None;
        }
        let (var_seg, enum_segs) = path.segments.split_last()?;
        let enm = enum_segs.iter().map(|s| s.ident.name.as_str()).collect::<Vec<_>>().join("::");
        let var = &var_seg.ident.name;
        let layout = self.enums.get(&enm)?;
        let idx = layout.variants.iter().position(|v| &v.name == var)?;
        Some((enm, idx))
    }
}

/// JIT-compile and call an **all-integer** entry function.
pub fn jit_call(module: &AstModule, entry: &str, args: &[i64]) -> Result<i64, String> {
    build(module)?.call_i64(entry, args)
}

/// JIT-compile and call an **all-f64** entry function.
pub fn jit_call_f64(module: &AstModule, entry: &str, args: &[f64]) -> Result<f64, String> {
    build(module)?.call_f64(entry, args)
}

/// Compile `main` to native code and run it (producing native output).
pub fn run_main(module: &AstModule) -> Result<(), String> {
    let jit = build(module)?;
    jit.call_i64("main", &[])?;
    use std::io::Write;
    let _ = std::io::stdout().flush();
    Ok(())
}

/// Compile the eligible functions for in-process JIT execution.
pub fn build(module: &AstModule) -> Result<Jit, String> {
    build_inner(module, false, false, Vec::new())
}

/// Like [`build`], but emit native **debug instrumentation**: hooks into the
/// compiled code report each statement's line and scalar locals to the runtime
/// debugger (see `aurora-runtime`). `src` is the program text, for line mapping.
/// The program still runs as real machine code — only instrumented.
pub fn build_debug(module: &AstModule, src: &str) -> Result<Jit, String> {
    build_inner(module, true, false, line_starts(src))
}

/// Like [`build`], but emit **profiler instrumentation**: each function records
/// its call count and wall-clock time to the runtime profiler.
pub fn build_profile(module: &AstModule) -> Result<Jit, String> {
    build_inner(module, false, true, Vec::new())
}

fn build_inner(
    module: &AstModule,
    debug: bool,
    profile: bool,
    line_starts: Vec<u32>,
) -> Result<Jit, String> {
    let mut builder = JITBuilder::new(cranelift_module::default_libcall_names())
        .map_err(|e| format!("failed to create JIT: {e}"))?;
    register_host_symbols(&mut builder);
    let mut jmod = JITModule::new(builder);
    let module = monomorphized(module)?;
    let (env, failed) = lower(&module, &mut jmod, false, debug, profile, line_starts)?;
    jmod.finalize_definitions().map_err(|e| format!("finalize: {e}"))?;
    Ok(Jit { module: jmod, env, failed })
}

/// Specialize generic functions for the concrete types they're called with,
/// so the backend only sees concrete functions. (Runs after type-checking, so
/// generic mismatches are still reported by the type checker.)
fn monomorphized(module: &AstModule) -> Result<AstModule, String> {
    Ok(AstModule { items: aurora_ast::monomorphize(module.items.clone())? })
}

/// Byte offsets of each source line start (line 1 begins at offset 0).
fn line_starts(src: &str) -> Vec<u32> {
    let mut starts = vec![0u32];
    for (i, c) in src.char_indices() {
        if c == '\n' {
            starts.push((i + 1) as u32);
        }
    }
    starts
}

// The JIT symbol table, the backend's host imports, and the call-site
// signature lookup are all generated from `aurora-abi`'s one builtin table, so
// they cannot drift apart. It is also where a bad row is caught: a table row
// naming a runtime function that does not exist fails to COMPILE here.
macro_rules! cl_ty {
    ($ptr:ident, I64) => {
        types::I64
    };
    ($ptr:ident, F64) => {
        types::F64
    };
    ($ptr:ident, Ptr) => {
        $ptr
    };
}

// A `Str` result is not returned: the caller allocates its two slots and passes
// their address as a leading pointer parameter, which the row does not spell.
macro_rules! cl_ret {
    ($ptr:ident, void) => {
        None
    };
    ($ptr:ident, Str) => {
        None
    };
    ($ptr:ident, $t:ident) => {
        Some(cl_ty!($ptr, $t))
    };
}

macro_rules! cl_params {
    ($ptr:ident, [$($p:ident),*], Str) => {
        &[$ptr, $(cl_ty!($ptr, $p)),*]
    };
    ($ptr:ident, [$($p:ident),*], $r:ident) => {
        &[$(cl_ty!($ptr, $p)),*]
    };
}

// An `inline` builtin is expanded by the backend and has no runtime function to
// take the address of.
macro_rules! host_symbol {
    ($b:ident, inline, $sym:ident) => {};
    ($b:ident, $kind:ident, $sym:ident) => {
        $b.symbol(stringify!($sym), aurora_runtime::$sym as *const u8);
    };
}

macro_rules! gen_register_host_symbols {
    ($([$kind:ident, $name:ident, $sym:ident, [$($p:ident),*], $ret:ident])*) => {
        /// Register the addresses of every `aurora_*` host function with the JIT.
        fn register_host_symbols(builder: &mut JITBuilder) {
            $( host_symbol!(builder, $kind, $sym); )*
            register_ffi_symbols(builder);
        }
    };
}

aurora_abi::for_each_builtin!(gen_register_host_symbols);

// An `inline` builtin has no host function, and a `linkonly` one is never
// called by name from Aurora, so neither is imported into the backend's table.
macro_rules! host_import {
    ($h:ident, $m:ident, $ptr:ident, inline, $n:ident, $s:ident, [$($p:ident),*], $r:ident) => {};
    ($h:ident, $m:ident, $ptr:ident, linkonly, $n:ident, $s:ident, [$($p:ident),*], $r:ident) => {};
    ($h:ident, $m:ident, $ptr:ident, $k:ident, $n:ident, $s:ident, [$($p:ident),*], $r:ident) => {
        $h.insert(
            stringify!($n),
            import($m, stringify!($s), cl_params!($ptr, [$($p),*], $r), cl_ret!($ptr, $r)),
        );
    };
}

macro_rules! gen_host_imports {
    ($([$kind:ident, $name:ident, $sym:ident, [$($p:ident),*], $ret:ident])*) => {
        /// Declare every runtime host function as an import in `jmod`, keyed by
        /// the name the call lowering looks it up by. An Aurora `str` argument
        /// is passed as the `Ptr, I64` pair its two slots hold.
        fn host_imports(jmod: &mut dyn Module) -> HashMap<&'static str, FuncId> {
            let ptr_ty = jmod.target_config().pointer_type();
            let mut hosts = HashMap::with_capacity(aurora_abi::TABLE.len());
            $( host_import!(hosts, jmod, ptr_ty, $kind, $name, $sym, [$($p),*], $ret); )*
            hosts
        }
    };
}

aurora_abi::for_each_builtin!(gen_host_imports);

/// Register common C-standard-library symbols so programs that bind them with
/// `@extern` resolve under the JIT (AOT resolves them at link time against the C
/// runtime). Referencing each `extern "C"` symbol here links it into `aurorac`
/// and yields its address. This is the curated set; bundled libraries (e.g. the
/// image loader) register their own symbols alongside the runtime's.
fn register_ffi_symbols(builder: &mut JITBuilder) {
    extern "C" {
        fn hypot(x: f64, y: f64) -> f64;
        fn cbrt(x: f64) -> f64;
        fn atan2(y: f64, x: f64) -> f64;
        fn log(x: f64) -> f64;
        fn log2(x: f64) -> f64;
        fn log10(x: f64) -> f64;
        fn exp(x: f64) -> f64;
        fn exp2(x: f64) -> f64;
        fn tan(x: f64) -> f64;
        fn asin(x: f64) -> f64;
        fn acos(x: f64) -> f64;
        fn atan(x: f64) -> f64;
        fn sinh(x: f64) -> f64;
        fn cosh(x: f64) -> f64;
        fn tanh(x: f64) -> f64;
        fn fmod(x: f64, y: f64) -> f64;
        // 32-bit variants, to exercise `f32` over FFI.
        fn sqrtf(x: f32) -> f32;
        fn cbrtf(x: f32) -> f32;
    }
    builder.symbol("hypot", hypot as *const u8);
    builder.symbol("cbrt", cbrt as *const u8);
    builder.symbol("atan2", atan2 as *const u8);
    builder.symbol("log", log as *const u8);
    builder.symbol("log2", log2 as *const u8);
    builder.symbol("log10", log10 as *const u8);
    builder.symbol("exp", exp as *const u8);
    builder.symbol("exp2", exp2 as *const u8);
    builder.symbol("tan", tan as *const u8);
    builder.symbol("asin", asin as *const u8);
    builder.symbol("acos", acos as *const u8);
    builder.symbol("atan", atan as *const u8);
    builder.symbol("sinh", sinh as *const u8);
    builder.symbol("cosh", cosh as *const u8);
    builder.symbol("tanh", tanh as *const u8);
    builder.symbol("fmod", fmod as *const u8);
    builder.symbol("sqrtf", sqrtf as *const u8);
    builder.symbol("cbrtf", cbrtf as *const u8);
    // `aurora_ffi_dot`/`aurora_ffi_dotf` - the Rust `extern "C"` functions that
    // exercise struct/array FFI by pointer - are `linkonly` rows in the builtin
    // table, so they are registered with the rest of the runtime's symbols.
}

/// Compile `module` to a native **object file** for the host target (COFF on
/// Windows, ELF on Linux, Mach-O on macOS). The user's `main` is emitted as the
/// symbol `aurora_user_main` so a tiny entry shim can wrap it; the program's
/// `aurora_*` host calls are left as undefined imports, resolved against the
/// `aurora-runtime` crate when the object is linked into an executable.
/// Compile to a native object. Returns the object bytes plus the map of
/// functions that FAILED to compile and were replaced with a no-op stub body
/// (name -> reason). A non-empty map means the produced binary will silently do
/// the wrong thing for those functions, so callers must surface it.
pub fn build_object(module: &AstModule) -> Result<(Vec<u8>, HashMap<String, String>), String> {
    let mut flags = codegen::settings::builder();
    // Statically linked into an executable, not a shared object. On Windows the
    // PE links non-PIC absolute relocations fine. On Linux/macOS the system
    // linker produces a position-independent executable (PIE) by default and
    // rejects absolute R_X86_64_64 relocations against symbols, so emit
    // position-independent code there (GOT/PLT-relative), which links into a PIE.
    let _ = flags.set("is_pic", if cfg!(windows) { "false" } else { "true" });
    // AOT is the release path (`aurorac build`): optimize for speed. The JIT
    // keeps Cranelift's default (fast compile) for quick `aurorac run` turnaround.
    let _ = flags.set("opt_level", "speed");
    let isa = cranelift_native::builder()
        .map_err(|e| format!("host isa unavailable: {e}"))?
        .finish(codegen::settings::Flags::new(flags))
        .map_err(|e| format!("isa finish: {e}"))?;
    let builder = ObjectBuilder::new(isa, "aurora", cranelift_module::default_libcall_names())
        .map_err(|e| format!("object builder: {e}"))?;
    let mut omod = ObjectModule::new(builder);
    let module = monomorphized(module)?;
    let (_, failed) = lower(&module, &mut omod, true, false, false, Vec::new())?;
    let product = omod.finish();
    let bytes = product.emit().map_err(|e| format!("emit object: {e}"))?;
    Ok((bytes, failed))
}

/// Declare and compile every function/method/closure/system in `module` into
/// `jmod` (a JIT or object module). When `aot`, the user `main` is exported
/// under the symbol `aurora_user_main`. Returns the populated environment and
/// the set of functions that fell back to a stub body.
fn lower(
    module: &AstModule,
    jmod: &mut dyn Module,
    aot: bool,
    debug: bool,
    profile: bool,
    line_starts: Vec<u32>,
) -> Result<(Env, HashMap<String, String>), String> {
    let ptr_ty = jmod.target_config().pointer_type();

    let hosts = host_imports(jmod);

    // Enum names, so types can be classified as enums (not structs) below.
    let enum_names: HashSet<String> = module
        .items
        .iter()
        .filter_map(|it| match &it.kind {
            ItemKind::Enum(e) => Some(e.name.name.clone()),
            _ => None,
        })
        .collect();

    // Struct layouts: fields are laid out inline by byte_size, so NESTED AGGREGATES (array and
    // struct fields) ARE supported - stored inline as VALUE fields. Construction copies the field
    // in (see copy_agg in tr_struct); a struct passed to a fn is by-ref, so its inline arrays/
    // structs mutate in place. The copy-on-construct (rather than aliasing the source array's
    // pointer) is deliberate: a struct is returned BY VALUE (sret copy), so an aliased pointer
    // field would dangle once the constructing frame returns. (Surprise vs `let q = p` aliasing:
    // that aliases only within a frame and is likewise copied across the sret return boundary.)
    let mut structs = HashMap::new();
    for item in &module.items {
        if let ItemKind::Struct(s) | ItemKind::Component(s) = &item.kind {
            if let StructBody::Named(fields) = &s.body {
                let mut layout = Vec::new();
                for f in fields {
                    layout.push((f.name.name.clone(), fix_enums(ty_to_cty(&f.ty.kind), &enum_names)));
                }
                structs.insert(s.name.name.clone(), layout);
            }
        }
    }

    // Enum layouts.
    let mut enums = HashMap::new();
    for item in &module.items {
        if let ItemKind::Enum(en) = &item.kind {
            let variants: Vec<EnumVariant> = en
                .variants
                .iter()
                .map(|v| {
                    let fields = match &v.data {
                        aurora_ast::VariantData::Unit => Vec::new(),
                        aurora_ast::VariantData::Tuple(tys) => tys
                            .iter()
                            .map(|t| (None, fix_enums(ty_to_cty(&t.kind), &enum_names)))
                            .collect(),
                        aurora_ast::VariantData::Struct(fs) => fs
                            .iter()
                            .map(|f| {
                                (Some(f.name.name.clone()), fix_enums(ty_to_cty(&f.ty.kind), &enum_names))
                            })
                            .collect(),
                    };
                    EnumVariant { name: v.name.name.clone(), fields }
                })
                .collect();
            let max_arity = variants.iter().map(|v| v.fields.len()).max().unwrap_or(0);
            enums.insert(en.name.name.clone(), EnumLayout { variants, slots: 1 + max_arity });
        }
    }

    // Top-level consts. Module flattening mangles a module's const to `m::NAME`,
    // so a module const is keyed by its qualified name here and needs no special
    // case at the use site.
    let mut consts = HashMap::new();
    for item in &module.items {
        if let ItemKind::Const(c) = &item.kind {
            consts.insert(c.name.name.clone(), c.value.clone());
        }
    }

    // Declare top-level functions and struct/enum `impl` methods. `compile_list`
    // pairs each (decl, key, optional self-receiver type) for pass 2.
    let mut fns: HashMap<String, FnInfo> = HashMap::new();
    let mut extern_fns: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut methods: HashMap<(String, String), String> = HashMap::new();
    let mut compile_list: Vec<(&aurora_ast::FnDecl, String, Option<Cty>)> = Vec::new();

    let declare = |jmod: &mut dyn Module,
                   fns: &mut HashMap<String, FnInfo>,
                   key: &str,
                   param_ctys: Vec<Cty>,
                   ret_cty: Cty|
     -> Result<(), String> {
        let sret = is_aggregate(&ret_cty);
        let mut sig = jmod.make_signature();
        if sret {
            sig.params.push(AbiParam::new(ptr_ty)); // leading sret pointer
        }
        for c in &param_ctys {
            sig.params.push(AbiParam::new(c.clif(ptr_ty)));
        }
        sig.returns
            .push(AbiParam::new(if sret { ptr_ty } else { ret_cty.clif(ptr_ty) }));
        // For AOT, expose `main` as `aurora_user_main` so the entry shim wraps
        // it instead of clashing with the C runtime's `main`.
        let sym = if aot && key == "main" { "aurora_user_main" } else { key };
        let id = jmod
            .declare_function(sym, Linkage::Export, &sig)
            .map_err(|e| format!("declare `{key}`: {e}"))?;
        fns.insert(key.to_string(), FnInfo { id, params: param_ctys, ret: ret_cty, sret });
        Ok(())
    };

    // FFI: an `@extern` (or `@extern("c_symbol")`) function with no body binds to
    // an external C-ABI symbol (a C library, or a Rust `#[no_mangle] extern "C"`
    // function). It's declared as an import; calls lower to a normal call, and the
    // symbol is resolved at link time (AOT) or via the registered/looked-up
    // symbols (JIT). Scalars and pointers (passed as `i64`) cross the boundary;
    // aggregates-by-value are not supported.
    let declare_import = |jmod: &mut dyn Module,
                          fns: &mut HashMap<String, FnInfo>,
                          key: &str,
                          sym: &str,
                          param_ctys: Vec<Cty>,
                          ret_cty: Cty|
     -> Result<(), String> {
        let mut sig = jmod.make_signature();
        for c in &param_ctys {
            // Scalars pass by value; structs/arrays pass as a pointer to their
            // (C-layout-compatible) storage — `const Foo*` / buffer parameters.
            let ct = if is_aggregate(c) { ptr_ty } else { c.clif(ptr_ty) };
            sig.params.push(AbiParam::new(ct));
        }
        sig.returns.push(AbiParam::new(ret_cty.clif(ptr_ty)));
        let id = jmod
            .declare_function(sym, Linkage::Import, &sig)
            .map_err(|e| format!("declare extern `{sym}`: {e}"))?;
        fns.insert(key.to_string(), FnInfo { id, params: param_ctys, ret: ret_cty, sret: false });
        Ok(())
    };

    for item in &module.items {
        // A `@vertex`/`@fragment`/`@compute` function is GPU code: `aurora-shader`
        // lowers it to WGSL, and its body legitimately names shader-only globals
        // and intrinsics (`model`, `albedo`, `vec4`, `texture`) that have no CPU
        // meaning. Compiling it here could only ever fail, and reporting that
        // failure would block every program shipping a shader beside its game
        // code. It is not CPU code, so it is not this backend's to compile.
        if aurora_ast::is_shader_stage(&item.attrs) {
            continue;
        }
        // FFI imports: `@extern` bodiless functions.
        if let ItemKind::Fn(f) = &item.kind {
            if f.body.is_none() && has_attr(&item.attrs, "extern") {
                let (p, r) = fn_abi(f);
                let p: Vec<Cty> = p.into_iter().map(|c| fix_enums(c, &enum_names)).collect();
                let r = fix_enums(r, &enum_names);
                // Struct/array parameters cross by pointer, so their layout must
                // match C's: every leaf an 8-byte `i64`/`f64` (Aurora stores each
                // field/element in an 8-byte slot). Reject anything else clearly.
                for c in &p {
                    if is_aggregate(c) && !ffi_layout_ok(c, &structs) {
                        return Err(format!(
                            "`@extern fn {}`: a struct/array parameter must have a \
                             C-compatible layout (fields of `i64`/`f64`); got a type \
                             with smaller or non-scalar fields",
                            f.name.name
                        ));
                    }
                }
                if is_aggregate(&r) {
                    return Err(format!(
                        "`@extern fn {}`: returning an aggregate by value isn't \
                         supported; return a scalar (or a pointer)",
                        f.name.name
                    ));
                }
                let sym = extern_symbol(&item.attrs, &f.name.name);
                declare_import(jmod, &mut fns, &f.name.name, &sym, p, r)?;
                extern_fns.insert(f.name.name.clone());
                continue;
            }
        }
        match &item.kind {
            ItemKind::Fn(f) if f.body.is_some() => {
                let (p, r) = fn_abi(f);
                let p = p.into_iter().map(|c| fix_enums(c, &enum_names)).collect();
                let r = fix_enums(r, &enum_names);
                declare(jmod, &mut fns, &f.name.name, p, r)?;
                compile_list.push((f, f.name.name.clone(), None));
            }
            ItemKind::Impl(im) => {
                let TypeKind::Path(p) = &im.self_ty.kind else { continue };
                let recv = p.segments.last().map(|s| s.ident.name.clone()).unwrap_or_default();
                if !structs.contains_key(&recv) && !enums.contains_key(&recv) {
                    continue; // receiver must be a known struct or enum
                }
                let recv_is_enum = enums.contains_key(&recv);
                let self_cty =
                    if recv_is_enum { Cty::Enum(recv.clone()) } else { Cty::Struct(recv.clone()) };
                for it in &im.items {
                    let aurora_ast::AssocItem::Fn(f) = it else { continue };
                    if !matches!(f.params.first(), Some(aurora_ast::Param::SelfParam { .. }))
                        || f.body.is_none()
                    {
                        continue;
                    }
                    // self is the first param; then the normal params.
                    let mut p = vec![self_cty.clone()];
                    for prm in &f.params {
                        if let aurora_ast::Param::Normal { ty, .. } = prm {
                            p.push(fix_enums(ty_to_cty(&ty.kind), &enum_names));
                        }
                    }
                    let r = match &f.ret {
                        Some(t) => fix_enums(ty_to_cty(&t.kind), &enum_names),
                        None => Cty::I64,
                    };
                    let key = format!("{recv}#{}", f.name.name);
                    declare(jmod, &mut fns, &key, p, r)?;
                    methods.insert((recv.clone(), f.name.name.clone()), key.clone());
                    compile_list.push((f, key, Some(self_cty.clone())));
                }
            }
            _ => {}
        }
    }

    // Concrete types implementing each trait, for `dyn Trait` dispatch.
    let mut trait_types: HashMap<String, Vec<String>> = HashMap::new();
    for item in &module.items {
        if let ItemKind::Impl(im) = &item.kind {
            if let (Some(tr), TypeKind::Path(p)) = (&im.trait_, &im.self_ty.kind) {
                let trait_name = tr.segments.last().map(|s| s.ident.name.clone()).unwrap_or_default();
                let ty = p.segments.last().map(|s| s.ident.name.clone()).unwrap_or_default();
                trait_types.entry(trait_name).or_default().push(ty);
            }
        }
    }

    // Lambda-lift closures: each becomes a top-level function `__lambda_N`
    // taking `(env_ptr, params...)`; captured variables are loaded from `env`.
    let mut closures: HashMap<aurora_span::Span, String> = HashMap::new();
    let mut lambda_captures: HashMap<String, Vec<String>> = HashMap::new();
    let mut lambdas: Vec<(String, Vec<String>, Vec<String>, &Expr)> = Vec::new();
    let mut found: Vec<&Expr> = Vec::new();
    for item in &module.items {
        match &item.kind {
            ItemKind::Fn(f) => {
                if let Some(body) = &f.body {
                    collect_closures(body, &mut found);
                }
            }
            ItemKind::Impl(im) => {
                for it in &im.items {
                    if let aurora_ast::AssocItem::Fn(f) = it {
                        if let Some(body) = &f.body {
                            collect_closures(body, &mut found);
                        }
                    }
                }
            }
            ItemKind::System(s) => collect_closures(&s.body, &mut found),
            _ => {}
        }
    }
    // Names that are NOT captures: top-level fns/methods and builtins.
    let mut exclude: HashSet<String> = fns.keys().cloned().collect();
    for bn in builtin_names() {
        exclude.insert(bn.to_string());
    }
    for (n, ce) in found.iter().enumerate() {
        if let ExprKind::Closure { params, body } = &ce.kind {
            let name = format!("__lambda_{n}");
            let pnames: Vec<String> = params
                .iter()
                .filter_map(|p| match p {
                    aurora_ast::Param::Normal { name, .. } => Some(name.name.clone()),
                    _ => None,
                })
                .collect();
            let captures = closure_captures(body, &pnames, &exclude);
            // Signature: env pointer (i64), then one i64 per param.
            let mut sig_ctys = vec![Cty::I64];
            sig_ctys.extend(std::iter::repeat_n(Cty::I64, pnames.len()));
            declare(jmod, &mut fns, &name, sig_ctys, Cty::I64)?;
            closures.insert(ce.span, name.clone());
            lambda_captures.insert(name.clone(), captures.clone());
            lambdas.push((name, pnames, captures, body));
        }
    }

    // Systems compile to zero-arg functions; `run_systems()` calls them in order.
    let mut system_order = Vec::new();
    let mut system_list: Vec<(&aurora_ast::SystemDecl, String)> = Vec::new();
    for item in &module.items {
        if let ItemKind::System(s) = &item.kind {
            let key = format!("system#{}", s.name.name);
            declare(jmod, &mut fns, &key, vec![], Cty::I64)?;
            system_order.push(key.clone());
            system_list.push((s, key));
        }
    }
    // Partition systems into ordered layers of mutually-independent systems
    // (§6.2): each multi-system layer is safe to run concurrently.
    let system_layers = aurora_ast::parallel_layers(module);

    let env = Env {
        fns,
        hosts,
        structs,
        enums,
        consts,
        const_stack: std::cell::RefCell::new(std::collections::HashSet::new()),
        methods,
        closures,
        lambda_captures,
        closure_sigs: std::cell::RefCell::new(HashMap::new()),
        extern_fns,
        system_order,
        system_layers,
        ptr_ty,
        debug,
        profile,
        trait_types,
        line_starts,
    };

    let mut ctx = jmod.make_context();
    // Maps a function/lambda/system that failed to compile to native code → the
    // specific reason, so callers can report *why* (not just "codegen gap").
    let mut failed: HashMap<String, String> = HashMap::new();
    let p = ptr_ty;
    for (f, key, self_cty) in &compile_list {
        let (params, ret, sret) = {
            let info = &env.fns[key];
            (info.params.clone(), info.ret.clone(), info.sret)
        };
        set_sig(&mut ctx, &*jmod, &params, &ret, p);
        if let Err(e) = compile_body(jmod, &mut ctx, f, &env, self_cty.as_ref(), sret, &ret) {
            jmod.clear_context(&mut ctx);
            set_sig(&mut ctx, &*jmod, &params, &ret, p);
            stub_body(&mut ctx, sret, &ret, p);
            failed.insert(key.clone(), e);
        }
        let id = env.fns[key].id;
        jmod.define_function(id, &mut ctx).map_err(|e| format!("define `{key}`: {e}"))?;
        jmod.clear_context(&mut ctx);
    }

    // Compile lambda-lifted closures.
    for (name, pnames, captures, body) in &lambdas {
        let mut ctys = vec![Cty::I64];
        ctys.extend(std::iter::repeat_n(Cty::I64, pnames.len()));
        set_sig(&mut ctx, &*jmod, &ctys, &Cty::I64, p);
        if let Err(e) = compile_lambda(jmod, &mut ctx, &env, name, pnames, captures, body) {
            jmod.clear_context(&mut ctx);
            set_sig(&mut ctx, &*jmod, &ctys, &Cty::I64, p);
            stub_body(&mut ctx, false, &Cty::I64, p);
            failed.insert(name.clone(), e);
        }
        let id = env.fns[name].id;
        jmod.define_function(id, &mut ctx).map_err(|e| format!("define `{name}`: {e}"))?;
        jmod.clear_context(&mut ctx);
    }

    // Compile system bodies (params bound to 0 — no resource providers yet).
    for (s, key) in &system_list {
        set_sig(&mut ctx, &*jmod, &[], &Cty::I64, p);
        let pnames: Vec<String> = s.params.iter().map(|p| p.name.name.clone()).collect();
        if let Err(e) = compile_system(jmod, &mut ctx, &env, &pnames, &s.body) {
            jmod.clear_context(&mut ctx);
            set_sig(&mut ctx, &*jmod, &[], &Cty::I64, p);
            stub_body(&mut ctx, false, &Cty::I64, p);
            failed.insert(key.clone(), e);
        }
        let id = env.fns[key].id;
        jmod.define_function(id, &mut ctx).map_err(|e| format!("define `{key}`: {e}"))?;
        jmod.clear_context(&mut ctx);
    }

    Ok((env, failed))
}

fn import(jmod: &mut dyn Module, name: &str, params: &[Type], ret: Option<Type>) -> FuncId {
    let mut sig = jmod.make_signature();
    for &p in params {
        sig.params.push(AbiParam::new(p));
    }
    if let Some(r) = ret {
        sig.returns.push(AbiParam::new(r));
    }
    jmod.declare_function(name, Linkage::Import, &sig).expect("declare host import")
}

fn set_sig(ctx: &mut codegen::Context, jmod: &dyn Module, params: &[Cty], ret: &Cty, ptr: Type) {
    let sret = is_aggregate(ret);
    ctx.func.signature = jmod.make_signature();
    if sret {
        ctx.func.signature.params.push(AbiParam::new(ptr));
    }
    for c in params {
        ctx.func.signature.params.push(AbiParam::new(c.clif(ptr)));
    }
    ctx.func
        .signature
        .returns
        .push(AbiParam::new(if sret { ptr } else { ret.clif(ptr) }));
}

fn compile_body(
    jmod: &mut dyn Module,
    ctx: &mut codegen::Context,
    f: &aurora_ast::FnDecl,
    env: &Env,
    self_cty: Option<&Cty>,
    sret: bool,
    ret_cty: &Cty,
) -> Result<(), String> {
    let mut fb_ctx = FunctionBuilderContext::new();
    let mut b = FunctionBuilder::new(&mut ctx.func, &mut fb_ctx);
    let entry = b.create_block();
    b.append_block_params_for_function_params(entry);
    b.switch_to_block(entry);

    let mut locals = Locals { scope: HashMap::new(), sret: None, loops: Vec::new() };
    if sret {
        // Record the caller's result slot so early returns can copy into it.
        locals.sret = Some((b.block_params(entry)[0], ret_cty.clone()));
    }
    let mut pi = if sret { 1 } else { 0 }; // leading sret pointer if aggregate return
    // Method receiver: bind `self` (a pointer to the aggregate) as first param.
    if let Some(cty) = self_cty {
        let var = b.declare_var(cty.clif(env.ptr_ty));
        b.def_var(var, b.block_params(entry)[pi]);
        locals.scope.insert("self".into(), (var, cty.clone()));
        pi += 1;
    }
    for p in &f.params {
        if let aurora_ast::Param::Normal { name, ty, .. } = p {
            let cty = env.cty(&ty.kind);
            let var = b.declare_var(cty.clif(env.ptr_ty));
            b.def_var(var, b.block_params(entry)[pi]);
            locals.scope.insert(name.name.clone(), (var, cty));
            pi += 1;
        }
    }
    b.seal_block(entry);
    // Push a debugger call frame on entry; report parameters too.
    emit_dbg_enter(jmod, &mut b, env, &f.name.name);
    emit_prof_enter(jmod, &mut b, env, &f.name.name);
    if env.debug {
        for p in &f.params {
            if let aurora_ast::Param::Normal { name, ty, .. } = p {
                let cty = env.cty(&ty.kind);
                let (var, _) = locals.scope[&name.name];
                let v = b.use_var(var);
                emit_dbg_value(jmod, &mut b, env, &name.name, v, &cty);
            }
        }
    }

    match tr_block(jmod, &mut b, &mut locals, env, f.body.as_ref().unwrap())? {
        Term::Val(v, _) => {
            emit_dbg_leave(jmod, &mut b, env); // pop frame on the normal return path
            emit_prof_exit(jmod, &mut b, env);
            if sret {
                // Copy the aggregate result into the caller's sret slot.
                let sret_ptr = b.block_params(entry)[0];
                for i in 0..agg_slots(env, ret_cty) {
                    let x = load_at(&mut b, v, i, types::I64);
                    store_at(&mut b, sret_ptr, i, x);
                }
                b.ins().return_(&[sret_ptr]);
            } else {
                b.ins().return_(&[v]);
            }
        }
        Term::Diverged => {}
    }
    b.finalize();
    Ok(())
}

/// Compile a lambda-lifted closure (no captures): i64 params -> i64 body.
fn compile_lambda(
    jmod: &mut dyn Module,
    ctx: &mut codegen::Context,
    env: &Env,
    name: &str,
    pnames: &[String],
    captures: &[String],
    body: &Expr,
) -> Result<(), String> {
    // A typed signature (recorded at the construction site) means this closure
    // involves `f64`: params/captures arrive as raw i64 slots and are
    // reinterpreted to their real type, and the result is returned as i64 bits.
    // Without one, the plain all-i64 convention applies.
    let sig = env.closure_sigs.borrow().get(name).cloned();
    let mut fb_ctx = FunctionBuilderContext::new();
    let mut b = FunctionBuilder::new(&mut ctx.func, &mut fb_ctx);
    let entry = b.create_block();
    b.append_block_params_for_function_params(entry);
    b.switch_to_block(entry);
    let mut locals = Locals { scope: HashMap::new(), sret: None, loops: Vec::new() };
    // Param 0 is the env pointer; load each captured value from it.
    let env_ptr = b.block_params(entry)[0];
    for (i, cap) in captures.iter().enumerate() {
        let cty = sig.as_ref().map(|s| s.captures[i].clone()).unwrap_or(Cty::I64);
        let raw = load_at(&mut b, env_ptr, i, types::I64);
        let v = from_i64_bits(&mut b, raw, &cty);
        let var = b.declare_var(cty.clif(env.ptr_ty));
        b.def_var(var, v);
        locals.scope.insert(cap.clone(), (var, cty));
    }
    for (i, pn) in pnames.iter().enumerate() {
        let cty = sig.as_ref().map(|s| s.params[i].clone()).unwrap_or(Cty::I64);
        let raw = b.block_params(entry)[1 + i];
        let v = from_i64_bits(&mut b, raw, &cty);
        let var = b.declare_var(cty.clif(env.ptr_ty));
        b.def_var(var, v);
        locals.scope.insert(pn.clone(), (var, cty));
    }
    b.seal_block(entry);
    match tr_expr(jmod, &mut b, &mut locals, env, body)? {
        Term::Val(v, vty) => {
            // Return as raw i64 bits (the signature's return is i64).
            let raw = to_i64_bits(&mut b, v, &vty);
            b.ins().return_(&[raw]);
        }
        Term::Diverged => {}
    }
    b.finalize();
    Ok(())
}

/// Compile a system body (zero-arg; named params bound to 0).
fn compile_system(
    jmod: &mut dyn Module,
    ctx: &mut codegen::Context,
    env: &Env,
    pnames: &[String],
    body: &Block,
) -> Result<(), String> {
    let mut fb_ctx = FunctionBuilderContext::new();
    let mut b = FunctionBuilder::new(&mut ctx.func, &mut fb_ctx);
    let entry = b.create_block();
    b.switch_to_block(entry);
    b.seal_block(entry);
    let mut locals = Locals { scope: HashMap::new(), sret: None, loops: Vec::new() };
    for pn in pnames {
        let var = b.declare_var(types::I64);
        let zero = b.ins().iconst(types::I64, 0);
        b.def_var(var, zero);
        locals.scope.insert(pn.clone(), (var, Cty::I64));
    }
    match tr_block(jmod, &mut b, &mut locals, env, body)? {
        Term::Val(v, _) => {
            b.ins().return_(&[v]);
        }
        Term::Diverged => {}
    }
    b.finalize();
    Ok(())
}

fn stub_body(ctx: &mut codegen::Context, sret: bool, ret_cty: &Cty, ptr: Type) {
    let mut fb_ctx = FunctionBuilderContext::new();
    let mut b = FunctionBuilder::new(&mut ctx.func, &mut fb_ctx);
    let entry = b.create_block();
    b.append_block_params_for_function_params(entry);
    b.switch_to_block(entry);
    b.seal_block(entry);
    let ret = if sret {
        b.block_params(entry)[0] // return the caller's sret pointer
    } else if *ret_cty == Cty::F32 {
        b.ins().f32const(0.0)
    } else if *ret_cty == Cty::F64 {
        b.ins().f64const(0.0)
    } else {
        b.ins().iconst(if ret_cty.is_scalar() { types::I64 } else { ptr }, 0)
    };
    b.ins().return_(&[ret]);
    b.finalize();
}

pub struct Jit {
    module: JITModule,
    env: Env,
    failed: HashMap<String, String>,
}

impl Jit {
    fn entry_ptr(&self, name: &str, want_int: bool) -> Result<(*const u8, usize), String> {
        if let Some(reason) = self.failed.get(name) {
            return Err(format!("`{name}` did not compile: {reason}"));
        }
        let info = self
            .env
            .fns
            .get(name)
            .ok_or_else(|| format!("`{name}` was not compiled (not a scalar function?)"))?;
        let ok = if want_int {
            info.ret == Cty::I64 && info.params.iter().all(|t| *t == Cty::I64)
        } else {
            info.ret == Cty::F64 && info.params.iter().all(|t| *t == Cty::F64)
        };
        if !ok {
            return Err(format!("`{name}` is not callable through this entry helper"));
        }
        Ok((self.module.get_finalized_function(info.id), info.params.len()))
    }

    pub fn call_i64(&self, name: &str, args: &[i64]) -> Result<i64, String> {
        let (ptr, arity) = self.entry_ptr(name, true)?;
        if args.len() != arity {
            return Err(format!("`{name}` expects {arity} args, got {}", args.len()));
        }
        // SAFETY: verified all-i64 signature of `arity` params.
        unsafe {
            Ok(match args {
                [] => std::mem::transmute::<_, extern "C" fn() -> i64>(ptr)(),
                [a] => std::mem::transmute::<_, extern "C" fn(i64) -> i64>(ptr)(*a),
                [a, b] => std::mem::transmute::<_, extern "C" fn(i64, i64) -> i64>(ptr)(*a, *b),
                [a, b, c] => {
                    std::mem::transmute::<_, extern "C" fn(i64, i64, i64) -> i64>(ptr)(*a, *b, *c)
                }
                _ => return Err("JIT entry supports up to 3 args".into()),
            })
        }
    }

    pub fn call_f64(&self, name: &str, args: &[f64]) -> Result<f64, String> {
        let (ptr, arity) = self.entry_ptr(name, false)?;
        if args.len() != arity {
            return Err(format!("`{name}` expects {arity} args, got {}", args.len()));
        }
        // SAFETY: verified all-f64 signature of `arity` params.
        unsafe {
            Ok(match args {
                [] => std::mem::transmute::<_, extern "C" fn() -> f64>(ptr)(),
                [a] => std::mem::transmute::<_, extern "C" fn(f64) -> f64>(ptr)(*a),
                [a, b] => std::mem::transmute::<_, extern "C" fn(f64, f64) -> f64>(ptr)(*a, *b),
                [a, b, c] => {
                    std::mem::transmute::<_, extern "C" fn(f64, f64, f64) -> f64>(ptr)(*a, *b, *c)
                }
                _ => return Err("JIT entry supports up to 3 args".into()),
            })
        }
    }

    pub fn compiled(&self, name: &str) -> bool {
        self.env.fns.contains_key(name) && !self.failed.contains_key(name)
    }

    /// The specific reason a function failed to compile to native code, if it
    /// did — so callers can report *why* instead of a generic "codegen gap".
    pub fn compile_error(&self, name: &str) -> Option<&str> {
        self.failed.get(name).map(|s| s.as_str())
    }

    /// Every function/lambda/system that failed to compile, name -> reason.
    ///
    /// A failed body is replaced with a stub that returns 0, so a non-empty map
    /// means the program would silently compute nothing for those functions.
    /// Anything that RUNS the program must refuse instead: the same rule
    /// `build_object`'s failure map enforces for `aurorac build`.
    pub fn failures(&self) -> &HashMap<String, String> {
        &self.failed
    }
}

struct Locals {
    scope: HashMap<String, (Variable, Cty)>,
    /// For an sret (aggregate-returning) function: the caller's result pointer
    /// and the return type, so an early `return <aggregate>` can copy into it.
    sret: Option<(Value, Cty)>,
    /// Stack of enclosing loops so `break`/`continue` know where to jump. The
    /// innermost loop is last. `continue_to` is the loop's latch (the increment
    /// step for `for`, the condition header for `while`/`loop`); `break_to` is
    /// the exit block. `cont_used` records whether `continue` actually targeted
    /// this loop, so a `for`'s step block isn't left as a dead block.
    loops: Vec<LoopFrame>,
}

#[derive(Clone)]
struct LoopFrame {
    // cranelift IR blocks (the bare `Block` name resolves to `aurora_ast::Block`).
    continue_to: cranelift::prelude::Block,
    break_to: cranelift::prelude::Block,
    cont_used: std::rc::Rc<std::cell::Cell<bool>>,
}

// --- memory helpers --------------------------------------------------------

/// Allocate `slots` 8-byte slots on the stack; return a pointer to slot 0.
fn alloc(b: &mut FunctionBuilder, env: &Env, slots: usize) -> Value {
    let size = (slots as u32).max(1) * SLOT;
    let slot = b.create_sized_stack_slot(StackSlotData::new(StackSlotKind::ExplicitSlot, size, 3));
    b.ins().stack_addr(env.ptr_ty, slot, 0)
}

fn store_at(b: &mut FunctionBuilder, ptr: Value, index: usize, v: Value) {
    b.ins().store(MemFlags::new(), v, ptr, (index as i32) * SLOT as i32);
}

fn load_at(b: &mut FunctionBuilder, ptr: Value, index: usize, ty: Type) -> Value {
    b.ins().load(ty, MemFlags::new(), ptr, (index as i32) * SLOT as i32)
}

fn store_b(b: &mut FunctionBuilder, ptr: Value, off: u32, v: Value) {
    b.ins().store(MemFlags::new(), v, ptr, off as i32);
}

fn load_b(b: &mut FunctionBuilder, ptr: Value, off: u32, ty: Type) -> Value {
    b.ins().load(ty, MemFlags::new(), ptr, off as i32)
}

/// A pointer to byte offset `off` within the aggregate at `base`.
fn agg_ptr(b: &mut FunctionBuilder, base: Value, off: u32) -> Value {
    if off == 0 {
        base
    } else {
        b.ins().iadd_imm(base, off as i64)
    }
}

/// Copy an aggregate (`byte_size` bytes, 8-byte chunks) from `src` to `dst+off`.
fn copy_agg(b: &mut FunctionBuilder, env: &Env, dst: Value, off: u32, src: Value, cty: &Cty) {
    let bytes = byte_size(env, cty);
    let mut k = 0;
    while k < bytes {
        let x = load_b(b, src, k, types::I64);
        store_b(b, dst, off + k, x);
        k += 8;
    }
}

// --- translation -----------------------------------------------------------

/// Emit `aurora_dbg_enter(name)` — push a debugger call frame for `func`.
fn emit_dbg_enter(m: &mut dyn Module, b: &mut FunctionBuilder, env: &Env, func: &str) {
    if !env.debug {
        return;
    }
    if let Ok((ptr, len)) = emit_str_data(m, b, env, func) {
        let f = m.declare_func_in_func(env.hosts["dbg_enter"], b.func);
        b.ins().call(f, &[ptr, len]);
    }
}

/// Emit `aurora_dbg_leave()` — pop the debugger's current call frame.
fn emit_dbg_leave(m: &mut dyn Module, b: &mut FunctionBuilder, env: &Env) {
    if !env.debug {
        return;
    }
    let f = m.declare_func_in_func(env.hosts["dbg_leave"], b.func);
    b.ins().call(f, &[]);
}

/// Emit `aurora_prof_enter(name)` at a function's entry (profiling builds).
fn emit_prof_enter(m: &mut dyn Module, b: &mut FunctionBuilder, env: &Env, func: &str) {
    if !env.profile {
        return;
    }
    if let Ok((ptr, len)) = emit_str_data(m, b, env, func) {
        let f = m.declare_func_in_func(env.hosts["prof_enter"], b.func);
        b.ins().call(f, &[ptr, len]);
    }
}

/// Emit `aurora_prof_exit()` at a function's exit (profiling builds).
fn emit_prof_exit(m: &mut dyn Module, b: &mut FunctionBuilder, env: &Env) {
    if !env.profile {
        return;
    }
    let f = m.declare_func_in_func(env.hosts["prof_exit"], b.func);
    b.ins().call(f, &[]);
}

/// Emit a `aurora_dbg_stmt(line)` call before a statement at source `line`.
fn emit_dbg_stmt(m: &mut dyn Module, b: &mut FunctionBuilder, env: &Env, line: u32) {
    if !env.debug || line == 0 {
        return;
    }
    let f = m.declare_func_in_func(env.hosts["dbg_stmt"], b.func);
    let ln = b.ins().iconst(types::I64, line as i64);
    b.ins().call(f, &[ln]);
}

/// Report a local `name` of type `cty` (value `val`) to the debugger. Scalars
/// are reported directly; aggregates are reported leaf-by-leaf with dotted /
/// indexed names (`v.x`, `t.0`, `a[2]`), so floats and nested data are visible.
fn emit_dbg_value(m: &mut dyn Module, b: &mut FunctionBuilder, env: &Env, name: &str, val: Value, cty: &Cty) {
    if !env.debug {
        return;
    }
    match cty {
        Cty::I64 => {
            if let Ok((ptr, len)) = emit_str_data(m, b, env, name) {
                let f = m.declare_func_in_func(env.hosts["dbg_var"], b.func);
                b.ins().call(f, &[ptr, len, val]);
            }
        }
        Cty::F32 | Cty::F64 => {
            let v = if *cty == Cty::F32 { b.ins().fpromote(types::F64, val) } else { val };
            if let Ok((ptr, len)) = emit_str_data(m, b, env, name) {
                let f = m.declare_func_in_func(env.hosts["dbg_var_f64"], b.func);
                b.ins().call(f, &[ptr, len, v]);
            }
        }
        Cty::Struct(sname) => {
            if let Some(fields) = env.structs.get(sname).cloned() {
                let mut off = 0u32;
                for (fname, fcty) in &fields {
                    emit_dbg_field(m, b, env, &format!("{name}.{fname}"), val, off, fcty);
                    off += byte_size(env, fcty);
                }
            }
        }
        Cty::Tuple(elems) => {
            let mut off = 0u32;
            for (idx, ecty) in elems.iter().enumerate() {
                emit_dbg_field(m, b, env, &format!("{name}.{idx}"), val, off, ecty);
                off += byte_size(env, ecty);
            }
        }
        Cty::Array(elem, n) => {
            let stride = byte_size(env, elem);
            for idx in 0..*n {
                emit_dbg_field(m, b, env, &format!("{name}[{idx}]"), val, idx as u32 * stride, elem);
            }
        }
        // Report an enum's active variant tag (slot 0) as an integer.
        Cty::Enum(_) => {
            let tag = load_b(b, val, 0, types::I64);
            if let Ok((ptr, len)) = emit_str_data(m, b, env, &format!("{name}.tag")) {
                let f = m.declare_func_in_func(env.hosts["dbg_var"], b.func);
                b.ins().call(f, &[ptr, len, tag]);
            }
        }
        // Trait objects / function values aren't decomposed for inspection.
        Cty::Dyn(_) => {}
        // Strings report their length; function values aren't inspected.
        Cty::Str => {
            let len = load_at(b, val, 1, types::I64);
            if let Ok((ptr, nlen)) = emit_str_data(m, b, env, &format!("{name}.len")) {
                let f = m.declare_func_in_func(env.hosts["dbg_var"], b.func);
                b.ins().call(f, &[ptr, nlen, len]);
            }
        }
        Cty::Fn(..) => {}
    }
}

/// Report one field/element at byte offset `off` from aggregate pointer `base`.
fn emit_dbg_field(m: &mut dyn Module, b: &mut FunctionBuilder, env: &Env, name: &str, base: Value, off: u32, cty: &Cty) {
    if cty.is_scalar() {
        let v = load_b(b, base, off, cty.clif(env.ptr_ty));
        emit_dbg_value(m, b, env, name, v, cty);
    } else {
        let sub = if off == 0 {
            base
        } else {
            let o = b.ins().iconst(types::I64, off as i64);
            b.ins().iadd(base, o)
        };
        emit_dbg_value(m, b, env, name, sub, cty);
    }
}

fn tr_block(
    m: &mut dyn Module,
    b: &mut FunctionBuilder,
    l: &mut Locals,
    env: &Env,
    block: &Block,
) -> Result<Term, String> {
    // Proper lexical BLOCK SCOPING: names bound with `let` inside this block live only
    // for the block. Snapshot the scope on entry and restore it on exit, so block-local
    // bindings are dropped and any outer names they shadowed come back. Reassignments to
    // outer mutables (`x = ...`) go through the same Cranelift variable, so their values
    // still persist - only NAME resolution is unwound. A name declared inside a block and
    // referenced after it is now a compile error, as it should be.
    let outer = l.scope.clone();
    let result = tr_block_inner(m, b, l, env, block);
    l.scope = outer;
    result
}

fn tr_block_inner(
    m: &mut dyn Module,
    b: &mut FunctionBuilder,
    l: &mut Locals,
    env: &Env,
    block: &Block,
) -> Result<Term, String> {
    for stmt in &block.stmts {
        if env.debug {
            let span = match stmt {
                Stmt::Let(s) => s.init.as_ref().map(|e| e.span).unwrap_or(s.pat.span),
                Stmt::Defer(e) | Stmt::Expr(e) => e.span,
            };
            emit_dbg_stmt(m, b, env, env.line_of(span.lo));
        }
        match stmt {
            Stmt::Let(let_stmt) => {
                let init = match &let_stmt.init {
                    Some(e) => val(m, b, l, env, e)?,
                    None => (b.ins().iconst(types::I64, 0), Cty::I64),
                };
                // Report a simple binding (scalar or aggregate) to the debugger.
                if env.debug {
                    if let PatKind::Binding { name, .. } = &let_stmt.pat.kind {
                        emit_dbg_value(m, b, env, &name.name, init.0, &init.1);
                    }
                }
                bind_let(b, l, env, &let_stmt.pat, init)?;
            }
            Stmt::Expr(e) => {
                if let ExprKind::If(ifx) = &e.kind {
                    tr_stmt_if(m, b, l, env, ifx)?;
                    continue;
                }
                if let Term::Diverged = tr_expr(m, b, l, env, e)? {
                    return Ok(Term::Diverged);
                }
            }
            Stmt::Defer(_) => return Err("`defer` is not supported by the JIT".into()),
        }
    }
    match &block.tail {
        Some(e) => {
            if env.debug {
                emit_dbg_stmt(m, b, env, env.line_of(e.span.lo));
            }
            // An `if` without `else` in tail position is a statement, not a value
            // (it has no `else` branch to produce one), so lower it as such.
            if let ExprKind::If(ifx) = &e.kind {
                if ifx.else_branch.is_none() {
                    tr_stmt_if(m, b, l, env, ifx)?;
                    return Ok(Term::Val(b.ins().iconst(types::I64, 0), Cty::I64));
                }
            }
            tr_expr(m, b, l, env, e)
        }
        None => Ok(Term::Val(b.ins().iconst(types::I64, 0), Cty::I64)),
    }
}

/// Bind a `let` pattern (a simple name or a tuple destructure).
fn bind_let(
    b: &mut FunctionBuilder,
    l: &mut Locals,
    env: &Env,
    pat: &Pat,
    (v, cty): (Value, Cty),
) -> Result<(), String> {
    match &pat.kind {
        PatKind::Binding { name, .. } => {
            let var = b.declare_var(cty.clif(env.ptr_ty));
            b.def_var(var, v);
            l.scope.insert(name.name.clone(), (var, cty));
            Ok(())
        }
        PatKind::Tuple(pats) => {
            let Cty::Tuple(elems) = &cty else {
                return Err("tuple pattern requires a tuple value (JIT)".into());
            };
            for (i, p) in pats.iter().enumerate() {
                let ety = elems[i].clone();
                let ev = load_at(b, v, i, ety.clif(env.ptr_ty));
                bind_let(b, l, env, p, (ev, ety))?;
            }
            Ok(())
        }
        PatKind::Wild => Ok(()),
        _ => Err("unsupported let-pattern in JIT".into()),
    }
}

fn tr_stmt_if(
    m: &mut dyn Module,
    b: &mut FunctionBuilder,
    l: &mut Locals,
    env: &Env,
    ifx: &aurora_ast::IfExpr,
) -> Result<(), String> {
    let (cond, _) = val(m, b, l, env, &ifx.cond)?;
    let then_b = b.create_block();
    let cont_b = b.create_block();
    let else_b = if ifx.else_branch.is_some() { Some(b.create_block()) } else { None };
    let false_target = else_b.unwrap_or(cont_b);
    b.ins().brif(cond, then_b, &[], false_target, &[]);

    b.switch_to_block(then_b);
    b.seal_block(then_b);
    if let Term::Val(..) = tr_block(m, b, l, env, &ifx.then_branch)? {
        b.ins().jump(cont_b, &[]);
    }

    if let (Some(else_b), Some(else_e)) = (else_b, &ifx.else_branch) {
        b.switch_to_block(else_b);
        b.seal_block(else_b);
        if let Term::Val(..) = tr_expr(m, b, l, env, else_e)? {
            b.ins().jump(cont_b, &[]);
        }
    }

    b.switch_to_block(cont_b);
    b.seal_block(cont_b);
    Ok(())
}

/// Lower a use of `const NAME` by lowering its initializer inline at this use
/// site. Consts have no runtime storage, so this is both the cheapest lowering (a
/// literal folds to an immediate) and the reason a const needs no initialization
/// order. `name` is the mangled name, so a module's const arrives as `m::NAME`.
fn tr_const(
    m: &mut dyn Module,
    b: &mut FunctionBuilder,
    l: &mut Locals,
    env: &Env,
    name: &str,
) -> Result<Term, String> {
    let Some(value) = env.consts.get(name) else {
        return Err(format!("unknown variable `{name}` in JIT"));
    };
    // A const defined in terms of itself has no value; recursing would exhaust
    // the stack, so report it instead.
    if !env.const_stack.borrow_mut().insert(name.to_string()) {
        return Err(format!("`const {name}` is defined in terms of itself"));
    }
    let r = tr_expr(m, b, l, env, value);
    env.const_stack.borrow_mut().remove(name);
    r
}

fn tr_expr(
    m: &mut dyn Module,
    b: &mut FunctionBuilder,
    l: &mut Locals,
    env: &Env,
    e: &Expr,
) -> Result<Term, String> {
    let (v, t) = match &e.kind {
        ExprKind::Int(n, _) => (b.ins().iconst(types::I64, *n as i64), Cty::I64),
        ExprKind::Bool(x) => (b.ins().iconst(types::I64, *x as i64), Cty::I64),
        ExprKind::Float(x, suffix) => match suffix {
            Some(FloatTy::F32) => (b.ins().f32const(*x as f32), Cty::F32),
            _ => (b.ins().f64const(*x), Cty::F64),
        },
        // A string literal is a first-class value: a `[data_ptr, len]` aggregate.
        ExprKind::Str(s) => {
            let (data_ptr, len) = emit_str_data(m, b, env, s)?;
            let ptr = alloc(b, env, 2);
            store_at(b, ptr, 0, data_ptr);
            store_at(b, ptr, 1, len);
            (ptr, Cty::Str)
        }
        ExprKind::Paren(inner) => return tr_expr(m, b, l, env, inner),
        ExprKind::SelfExpr => {
            let (var, cty) = l.scope.get("self").cloned().ok_or("`self` not bound in JIT")?;
            (b.use_var(var), cty)
        }
        ExprKind::Path(p) if p.is_single() => {
            let name = &p.segments[0].ident.name;
            match l.scope.get(name).cloned() {
                Some((var, cty)) => (b.use_var(var), cty),
                // Not a local, so it may be a `const` (a module's const was
                // mangled to the single name `m::NAME`), lowered inline.
                None => return tr_const(m, b, l, env, name),
            }
        }
        ExprKind::Path(p) => {
            // Enum unit-variant `Enum::Variant`, or a const reached through a
            // module path (`m::NAME`) from outside that module.
            let Some((enm, idx)) = env.enum_variant(p) else {
                let joined =
                    p.segments.iter().map(|s| s.ident.name.as_str()).collect::<Vec<_>>().join("::");
                if env.consts.contains_key(&joined) {
                    return tr_const(m, b, l, env, &joined);
                }
                return Err("unsupported path expression in JIT".into());
            };
            let ptr = alloc(b, env, env.enums[&enm].slots);
            let tag = b.ins().iconst(types::I64, idx as i64);
            store_at(b, ptr, 0, tag);
            (ptr, Cty::Enum(enm))
        }
        ExprKind::Match { scrutinee, arms } => return tr_match(m, b, l, env, scrutinee, arms),
        ExprKind::Closure { params, body } => {
            // The closure was lambda-lifted. Build an env holding captured values,
            // then a [fn_ptr, env_ptr] closure pair; yield a pointer to it.
            let name = env.closures.get(&e.span).ok_or("closure not lifted (JIT)")?.clone();
            let captures = env.lambda_captures.get(&name).cloned().unwrap_or_default();
            let arity = env.fns[&name].params.len().saturating_sub(1); // minus env param
            let id = env.fns[&name].id;

            // Capture types (from the enclosing scope) and parameter types (from
            // annotations) are known here. If the closure involves `f64`, infer
            // its return type and record a typed signature so `compile_lambda`
            // and the call site use the bitcasting ABI; otherwise the plain
            // all-i64 path applies (unchanged).
            let capture_ctys: Vec<Cty> = captures
                .iter()
                .map(|cap| {
                    l.scope
                        .get(cap)
                        .map(|(_, c)| c.clone())
                        .ok_or_else(|| format!("closure captures unknown `{cap}` (JIT)"))
                })
                .collect::<Result<_, _>>()?;
            // Parameter types: an annotation if present, otherwise inferred from
            // how the body uses the parameter (e.g. `|x| x * scale` ⇒ `x: f64`
            // when `scale` is an f64 capture). Falls back to `i64` when the use
            // doesn't pin it — same as before, so never a miscompile.
            let outer_scope: HashMap<String, Cty> =
                l.scope.iter().map(|(k, (_, c))| (k.clone(), c.clone())).collect();
            let param_ctys: Vec<Cty> = params
                .iter()
                .filter_map(|p| match p {
                    aurora_ast::Param::Normal { name, ty, .. } => Some(if matches!(ty.kind, TypeKind::Infer) {
                        infer_param_cty(&name.name, body, &outer_scope, env).unwrap_or(Cty::I64)
                    } else {
                        ty_to_cty(&ty.kind)
                    }),
                    _ => None,
                })
                .collect();
            let mut infer_scope = outer_scope.clone();
            for (p, c) in params.iter().zip(param_ctys.iter()) {
                if let aurora_ast::Param::Normal { name, .. } = p {
                    infer_scope.insert(name.name.clone(), c.clone());
                }
            }
            // The return type completes the closure's signature. If the body's
            // type can't be inferred, default to i64 (the bitcasting ABI is then
            // a no-op, matching the legacy integer convention).
            let ret_cty = infer_cty(body, &infer_scope, env).unwrap_or(Cty::I64);
            // Record the full signature so `compile_lambda` reinterprets params
            // and captures consistently with how the call site passes them.
            env.closure_sigs.borrow_mut().insert(
                name.clone(),
                ClosureSig { params: param_ctys.clone(), captures: capture_ctys.clone() },
            );

            let env_ptr = alloc(b, env, captures.len().max(1));
            for (i, cap) in captures.iter().enumerate() {
                let (var, _) = l.scope.get(cap).cloned().unwrap();
                let cv = b.use_var(var);
                // Store float captures as raw bits; the lambda reinterprets them.
                let raw = to_i64_bits(b, cv, &capture_ctys[i]);
                store_at(b, env_ptr, i, raw);
            }
            let fref = m.declare_func_in_func(id, b.func);
            let faddr = b.ins().func_addr(env.ptr_ty, fref);
            let cl = alloc(b, env, 2);
            store_at(b, cl, 0, faddr);
            store_at(b, cl, 1, env_ptr);
            let _ = arity;
            (cl, Cty::Fn(param_ctys, Box::new(ret_cty)))
        }
        ExprKind::Pipe { value, func } => {
            // `x |> f(a)` == `f(x, a)`; `x |> f` == `f(x)`. Desugar to a call.
            let (callee, extra) = match &func.kind {
                ExprKind::Call { callee, args, .. } => ((**callee).clone(), args.clone()),
                _ => ((**func).clone(), Vec::new()),
            };
            let mut args = vec![aurora_ast::Arg { name: None, value: (**value).clone() }];
            args.extend(extra);
            let call = Expr {
                kind: ExprKind::Call { callee: Box::new(callee), type_args: Vec::new(), args },
                span: e.span,
            };
            return tr_expr(m, b, l, env, &call);
        }
        ExprKind::Unary(op, x) => {
            let (xv, xt) = val(m, b, l, env, x)?;
            match op {
                UnOp::Neg if xt == Cty::I64 => (b.ins().ineg(xv), xt),
                UnOp::Neg => (b.ins().fneg(xv), xt),
                UnOp::Not => {
                    let zero = b.ins().iconst(types::I64, 0);
                    let c = b.ins().icmp(IntCC::Equal, xv, zero);
                    (b.ins().uextend(types::I64, c), Cty::I64)
                }
                _ => return Err("unsupported unary operator in JIT".into()),
            }
        }
        ExprKind::Cast(inner, ty) => {
            let (xv, from) = val(m, b, l, env, inner)?;
            let to = ty_to_cty(&ty.kind);
            (cast(b, xv, &from, &to)?, to)
        }
        ExprKind::Binary(op, a, c) => {
            return tr_binary(m, b, l, env, *op, a, c).map(|(v, t)| Term::Val(v, t))
        }
        ExprKind::Assign(op, lhs, rhs) => {
            let (rv, rt) = val(m, b, l, env, rhs)?;
            assign(m, b, l, env, lhs, op, rv, rt)?;
            (b.ins().iconst(types::I64, 0), Cty::I64)
        }
        ExprKind::While { cond, body } => {
            let header = b.create_block();
            let body_b = b.create_block();
            let exit = b.create_block();
            b.ins().jump(header, &[]);
            b.switch_to_block(header);
            let (c, _) = val(m, b, l, env, cond)?;
            b.ins().brif(c, body_b, &[], exit, &[]);
            b.switch_to_block(body_b);
            b.seal_block(body_b);
            // `continue` jumps back to the condition header; `break` to the exit.
            l.loops.push(LoopFrame {
                continue_to: header,
                break_to: exit,
                cont_used: std::rc::Rc::new(std::cell::Cell::new(false)),
            });
            let term = tr_block(m, b, l, env, body)?;
            l.loops.pop();
            if let Term::Val(..) = term {
                b.ins().jump(header, &[]);
            }
            b.seal_block(header);
            b.switch_to_block(exit);
            b.seal_block(exit);
            (b.ins().iconst(types::I64, 0), Cty::I64)
        }
        ExprKind::Loop(body) => {
            // `loop { .. }` runs forever until a `break`. The exit block is only
            // reachable from a `break`, so it is dead if the loop has none.
            let header = b.create_block();
            let exit = b.create_block();
            b.ins().jump(header, &[]);
            b.switch_to_block(header);
            l.loops.push(LoopFrame {
                continue_to: header,
                break_to: exit,
                cont_used: std::rc::Rc::new(std::cell::Cell::new(false)),
            });
            let term = tr_block(m, b, l, env, body)?;
            l.loops.pop();
            if let Term::Val(..) = term {
                b.ins().jump(header, &[]);
            }
            b.seal_block(header);
            b.switch_to_block(exit);
            b.seal_block(exit);
            (b.ins().iconst(types::I64, 0), Cty::I64)
        }
        ExprKind::Break(opt) => {
            // Evaluate any break value for its side effects (value-bearing
            // `break expr` is not yet wired to a loop result).
            if let Some(x) = opt {
                val(m, b, l, env, x)?;
            }
            let target = l
                .loops
                .last()
                .ok_or("`break` used outside of a loop")?
                .break_to;
            b.ins().jump(target, &[]);
            return Ok(Term::Diverged);
        }
        ExprKind::Continue => {
            let frame = l.loops.last().ok_or("`continue` used outside of a loop")?;
            frame.cont_used.set(true);
            let target = frame.continue_to;
            b.ins().jump(target, &[]);
            return Ok(Term::Diverged);
        }
        ExprKind::For { pat, iter, body } => return tr_for(m, b, l, env, pat, iter, body),
        ExprKind::Call { callee, args, .. } => return tr_call(m, b, l, env, callee, args),
        ExprKind::Block(block) => return tr_block(m, b, l, env, block),
        ExprKind::If(ifx) => return tr_value_if(m, b, l, env, ifx),
        ExprKind::Struct { path, fields, .. } => return tr_struct(m, b, l, env, path, fields),
        ExprKind::Tuple(items) => return tr_tuple(m, b, l, env, items),
        ExprKind::Array(items) => return tr_array(m, b, l, env, items),
        ExprKind::ArrayRepeat { value, count } => return tr_array_repeat(m, b, l, env, value, count),
        ExprKind::Field { base, field } => return tr_field(m, b, l, env, base, field),
        ExprKind::Index { base, index } => return tr_index(m, b, l, env, base, index),
        ExprKind::Return(opt) => {
            let rv = match opt {
                Some(inner) => val(m, b, l, env, inner)?.0,
                None => b.ins().iconst(types::I64, 0),
            };
            emit_dbg_leave(m, b, env); // pop frame on the early-return path
            emit_prof_exit(m, b, env);
            // Aggregate-returning (sret) function: copy the value into the
            // caller's result slot and return that pointer.
            if let Some((sret_ptr, ret_cty)) = l.sret.clone() {
                for i in 0..agg_slots(env, &ret_cty) {
                    let x = load_at(b, rv, i, types::I64);
                    store_at(b, sret_ptr, i, x);
                }
                b.ins().return_(&[sret_ptr]);
            } else {
                b.ins().return_(&[rv]);
            }
            return Ok(Term::Diverged);
        }
        ExprKind::Try(inner) => return tr_try(m, b, l, env, inner),
        _ => return Err("unsupported expression in JIT".into()),
    };
    Ok(Term::Val(v, t))
}

fn tr_struct(
    m: &mut dyn Module,
    b: &mut FunctionBuilder,
    l: &mut Locals,
    env: &Env,
    path: &aurora_ast::Path,
    fields: &[aurora_ast::FieldInit],
) -> Result<Term, String> {
    // Enum struct-variant `Enum::Variant { field: .. }`.
    if let Some((enm, idx)) = env.enum_variant(path) {
        let layout = &env.enums[&enm];
        let slots = layout.slots;
        let var_fields = layout.variants[idx].fields.clone();
        let ptr = alloc(b, env, slots);
        let tag = b.ins().iconst(types::I64, idx as i64);
        store_at(b, ptr, 0, tag);
        for (i, (fname, _)) in var_fields.iter().enumerate() {
            let init = fields.iter().find(|fi| Some(&fi.name.name) == fname.as_ref());
            let v = match init.and_then(|fi| fi.value.as_ref()) {
                Some(e) => val(m, b, l, env, e)?.0,
                None => b.ins().iconst(types::I64, 0),
            };
            store_at(b, ptr, 1 + i, v);
        }
        return Ok(Term::Val(ptr, Cty::Enum(enm)));
    }

    // A module-qualified struct path (`world::World`) must join its segments with `::` to match
    // the flattened, mangled layout key - exactly like a module-qualified call. Taking only the
    // last segment dropped the module prefix, so a struct defined in one module could not be
    // constructed from another ("unknown struct").
    let name = if path.segments.len() > 1 {
        path.segments.iter().map(|s| s.ident.name.as_str()).collect::<Vec<_>>().join("::")
    } else {
        path.segments.first().map(|s| s.ident.name.clone()).unwrap_or_default()
    };
    let layout = env
        .structs
        .get(&name)
        .ok_or_else(|| format!("unknown struct `{name}` in JIT"))?
        .clone();
    let cty = Cty::Struct(name.clone());
    let ptr = alloc(b, env, agg_slots(env, &cty));
    let mut off = 0u32;
    for (fname, fcty) in &layout {
        let init = fields.iter().find(|fi| &fi.name.name == fname);
        match init.and_then(|fi| fi.value.as_ref()) {
            Some(e) => {
                let (v, _) = val(m, b, l, env, e)?;
                if fcty.is_scalar() {
                    store_b(b, ptr, off, v);
                } else {
                    copy_agg(b, env, ptr, off, v, fcty);
                }
            }
            None => match init {
                // shorthand `{ x }` -> variable `x`
                Some(fi) => {
                    let (var, _) = l
                        .scope
                        .get(&fi.name.name)
                        .cloned()
                        .ok_or_else(|| format!("unknown field init `{}`", fi.name.name))?;
                    let v = b.use_var(var);
                    if fcty.is_scalar() {
                        store_b(b, ptr, off, v);
                    } else {
                        copy_agg(b, env, ptr, off, v, fcty);
                    }
                }
                None if fcty.is_scalar() => {
                    let z = zero_scalar(b, fcty);
                    store_b(b, ptr, off, z);
                }
                None => return Err("missing aggregate field in JIT".into()),
            },
        }
        off += byte_size(env, fcty);
    }
    Ok(Term::Val(ptr, cty))
}

fn tr_tuple(
    m: &mut dyn Module,
    b: &mut FunctionBuilder,
    l: &mut Locals,
    env: &Env,
    items: &[Expr],
) -> Result<Term, String> {
    let mut vals = Vec::new();
    let mut tys = Vec::new();
    let mut total = 0u32;
    for e in items {
        let (v, t) = val(m, b, l, env, e)?;
        total += byte_size(env, &t);
        vals.push((v, t.clone()));
        tys.push(t);
    }
    let ptr = alloc(b, env, (total / 8).max(1) as usize);
    let mut off = 0u32;
    for (v, t) in &vals {
        if t.is_scalar() {
            store_b(b, ptr, off, *v);
        } else {
            copy_agg(b, env, ptr, off, *v, t);
        }
        off += byte_size(env, t);
    }
    Ok(Term::Val(ptr, Cty::Tuple(tys)))
}

/// `[value; count]` — a fixed array of `count` copies of `value`. `count` must
/// be a constant integer literal (arrays are fixed-size in codegen).
fn tr_array_repeat(
    m: &mut dyn Module,
    b: &mut FunctionBuilder,
    l: &mut Locals,
    env: &Env,
    value: &Expr,
    count: &Expr,
) -> Result<Term, String> {
    let n = match &count.kind {
        ExprKind::Int(v, _) => *v as usize,
        _ => return Err("array-repeat count must be a constant in JIT".into()),
    };
    let (v, elem) = val(m, b, l, env, value)?;
    let stride = byte_size(env, &elem);
    let ptr = alloc(b, env, ((stride * n as u32) / 8).max(1) as usize);
    for i in 0..n {
        let off = stride * i as u32;
        if elem.is_scalar() {
            store_b(b, ptr, off, v);
        } else {
            copy_agg(b, env, ptr, off, v, &elem);
        }
    }
    Ok(Term::Val(ptr, Cty::Array(Box::new(elem), n)))
}

fn tr_array(
    m: &mut dyn Module,
    b: &mut FunctionBuilder,
    l: &mut Locals,
    env: &Env,
    items: &[Expr],
) -> Result<Term, String> {
    let mut vals = Vec::new();
    let mut elem = Cty::I64;
    for e in items {
        let (v, t) = val(m, b, l, env, e)?;
        elem = t.clone();
        vals.push((v, t));
    }
    let stride = byte_size(env, &elem);
    let ptr = alloc(b, env, ((stride * items.len() as u32) / 8).max(1) as usize);
    for (i, (v, t)) in vals.iter().enumerate() {
        let off = stride * i as u32;
        if t.is_scalar() {
            store_b(b, ptr, off, *v);
        } else {
            copy_agg(b, env, ptr, off, *v, t);
        }
    }
    Ok(Term::Val(ptr, Cty::Array(Box::new(elem), items.len())))
}

fn tr_field(
    m: &mut dyn Module,
    b: &mut FunctionBuilder,
    l: &mut Locals,
    env: &Env,
    base: &Expr,
    field: &FieldAccess,
) -> Result<Term, String> {
    let (ptr, cty) = val(m, b, l, env, base)?;
    let (off, fcty) = field_offset(env, &cty, field)?;
    if fcty.is_scalar() {
        Ok(Term::Val(load_b(b, ptr, off, fcty.clif(env.ptr_ty)), fcty))
    } else {
        // Aggregate field: a pointer into the parent.
        Ok(Term::Val(agg_ptr(b, ptr, off), fcty))
    }
}

/// Emit an array bounds check: if `iv` (unsigned, so negatives wrap huge) is
/// `>= len`, call the runtime's `aurora_oob` to print a clear panic and exit;
/// otherwise fall through. `len` is the static array length.
fn emit_bounds_check(m: &mut dyn Module, b: &mut FunctionBuilder, env: &Env, iv: Value, len: usize) {
    let n = b.ins().iconst(types::I64, len as i64);
    let oob = b.ins().icmp(IntCC::UnsignedGreaterThanOrEqual, iv, n);
    let fail = b.create_block();
    let ok = b.create_block();
    b.ins().brif(oob, fail, &[], ok, &[]);
    b.switch_to_block(fail);
    b.seal_block(fail);
    let f = m.declare_func_in_func(env.hosts["oob"], b.func);
    b.ins().call(f, &[iv, n]);
    // `aurora_oob` exits the process, so this is unreachable — but the block
    // still needs a terminator.
    b.ins().trap(TrapCode::HEAP_OUT_OF_BOUNDS);
    b.switch_to_block(ok);
    b.seal_block(ok);
}

fn tr_index(
    m: &mut dyn Module,
    b: &mut FunctionBuilder,
    l: &mut Locals,
    env: &Env,
    base: &Expr,
    index: &Expr,
) -> Result<Term, String> {
    let (ptr, cty) = val(m, b, l, env, base)?;
    let Cty::Array(elem, len) = &cty else {
        return Err("indexing a non-array in JIT".into());
    };
    let len = *len;
    let elem = (**elem).clone();
    let stride = byte_size(env, &elem);
    let (iv, _) = val(m, b, l, env, index)?;
    emit_bounds_check(m, b, env, iv, len);
    let stridev = b.ins().iconst(types::I64, stride as i64);
    let off = b.ins().imul(iv, stridev);
    let addr = b.ins().iadd(ptr, off);
    if elem.is_scalar() {
        let v = b.ins().load(elem.clif(env.ptr_ty), MemFlags::new(), addr, 0);
        Ok(Term::Val(v, elem))
    } else {
        Ok(Term::Val(addr, elem)) // pointer to the element
    }
}

/// `for v in <array | range> { body }`.
fn tr_for(
    m: &mut dyn Module,
    b: &mut FunctionBuilder,
    l: &mut Locals,
    env: &Env,
    pat: &Pat,
    iter: &Expr,
    body: &Block,
) -> Result<Term, String> {
    // ECS query form: `for (p, v) in query<&mut P, &V> { ... }`.
    if let ExprKind::Query(q) = &iter.kind {
        return tr_query_loop(m, b, l, env, pat, q, body);
    }

    let name = binding_name(pat).ok_or("JIT for-loop needs a simple binding")?;

    // Integer range form.
    if let ExprKind::Range { start: Some(s), end: Some(e), inclusive } = &iter.kind {
        let (sv, _) = val(m, b, l, env, s)?;
        let (ev, _) = val(m, b, l, env, e)?;
        let var = b.declare_var(types::I64);
        b.def_var(var, sv);
        l.scope.insert(name, (var, Cty::I64));
        let end_var = b.declare_var(types::I64);
        b.def_var(end_var, ev);
        let cc = if *inclusive { IntCC::SignedLessThanOrEqual } else { IntCC::SignedLessThan };
        loop_count(m, b, l, env, var, |b| b.use_var(end_var), cc, body)?;
        return Ok(Term::Val(b.ins().iconst(types::I64, 0), Cty::I64));
    }

    // Array form: iterate elements by index.
    let (ptr, cty) = val(m, b, l, env, iter)?;
    let Cty::Array(elem, n) = cty else {
        return Err("JIT for-loops support integer ranges and arrays".into());
    };
    let elem = (*elem).clone();
    let idx = b.declare_var(types::I64);
    let zero = b.ins().iconst(types::I64, 0);
    b.def_var(idx, zero);
    let elem_var = b.declare_var(elem.clif(env.ptr_ty));
    l.scope.insert(name, (elem_var, elem.clone()));
    let len = b.ins().iconst(types::I64, n as i64);
    let len_var = b.declare_var(types::I64);
    b.def_var(len_var, len);

    let header = b.create_block();
    let body_b = b.create_block();
    let step = b.create_block();
    let exit = b.create_block();
    b.ins().jump(header, &[]);
    b.switch_to_block(header);
    let i = b.use_var(idx);
    let ln = b.use_var(len_var);
    let c = b.ins().icmp(IntCC::SignedLessThan, i, ln);
    b.ins().brif(c, body_b, &[], exit, &[]);
    b.switch_to_block(body_b);
    b.seal_block(body_b);
    // elem_var = &ptr[i] (scalars are loaded; aggregates bind the element pointer)
    let stride = b.ins().iconst(types::I64, byte_size(env, &elem) as i64);
    let off = b.ins().imul(i, stride);
    let addr = b.ins().iadd(ptr, off);
    let ev = if elem.is_scalar() {
        b.ins().load(elem.clif(env.ptr_ty), MemFlags::new(), addr, 0)
    } else {
        addr
    };
    b.def_var(elem_var, ev);
    // `continue` advances the index via the step block; `break` exits.
    let cont_used = std::rc::Rc::new(std::cell::Cell::new(false));
    l.loops.push(LoopFrame { continue_to: step, break_to: exit, cont_used: cont_used.clone() });
    let term = tr_block(m, b, l, env, body)?;
    l.loops.pop();
    let body_falls = matches!(term, Term::Val(..));
    if body_falls {
        b.ins().jump(step, &[]);
    }
    // Step block: bump the index and re-test. If the body always diverges and no
    // `continue` reaches here, the block is dead - send it straight to the exit so
    // the index variable is never read in an unreachable block.
    b.seal_block(step);
    b.switch_to_block(step);
    if body_falls || cont_used.get() {
        let i2 = b.use_var(idx);
        let one = b.ins().iconst(types::I64, 1);
        let next = b.ins().iadd(i2, one);
        b.def_var(idx, next);
        b.ins().jump(header, &[]);
    } else {
        b.ins().jump(exit, &[]);
    }
    b.seal_block(header);
    b.switch_to_block(exit);
    b.seal_block(exit);
    Ok(Term::Val(b.ins().iconst(types::I64, 0), Cty::I64))
}

/// `match scrut { pat => body, ... }` as a value. Enums dispatch on their tag;
/// scalars compare by value. Each arm tests in sequence, binds payload in its
/// body block, and writes its value to a shared result variable.
/// `expr?` — evaluate `expr` (an enum like `Result`); if its tag is the success
/// variant (index 0) yield its payload, otherwise early-return the whole enum
/// from the enclosing function (which must return a compatible enum type).
fn tr_try(
    m: &mut dyn Module,
    b: &mut FunctionBuilder,
    l: &mut Locals,
    env: &Env,
    inner: &Expr,
) -> Result<Term, String> {
    let (eptr, ecty) = val(m, b, l, env, inner)?;
    let Cty::Enum(name) = &ecty else {
        return Err("`?` applies only to enum (Result/Option) values in JIT".into());
    };
    // Success variant is index 0; its (single) payload is the unwrapped value.
    let ok_payload = env.enums[name]
        .variants
        .first()
        .and_then(|v| v.fields.first())
        .map(|(_, c)| c.clone())
        .unwrap_or(Cty::I64);

    let tag = load_at(b, eptr, 0, types::I64);
    let is_ok = b.ins().icmp_imm(IntCC::Equal, tag, 0);
    let err_b = b.create_block();
    let ok_b = b.create_block();
    b.ins().brif(is_ok, ok_b, &[], err_b, &[]);

    // Error path: propagate by returning the whole enum (sret copy).
    b.switch_to_block(err_b);
    b.seal_block(err_b);
    emit_dbg_leave(m, b, env);
    emit_prof_exit(m, b, env);
    if let Some((sret_ptr, ret_cty)) = l.sret.clone() {
        for i in 0..agg_slots(env, &ret_cty) {
            let x = load_at(b, eptr, i, types::I64);
            store_at(b, sret_ptr, i, x);
        }
        b.ins().return_(&[sret_ptr]);
    } else {
        b.ins().return_(&[eptr]);
    }

    // Success path: yield the payload (scalar loaded, aggregate as sub-pointer).
    b.switch_to_block(ok_b);
    b.seal_block(ok_b);
    let val = if ok_payload.is_scalar() {
        load_at(b, eptr, 1, ok_payload.clif(env.ptr_ty))
    } else {
        // Payload begins at slot 1 (byte offset 8).
        let off = b.ins().iconst(types::I64, 8);
        b.ins().iadd(eptr, off)
    };
    Ok(Term::Val(val, ok_payload))
}

fn tr_match(
    m: &mut dyn Module,
    b: &mut FunctionBuilder,
    l: &mut Locals,
    env: &Env,
    scrut: &Expr,
    arms: &[aurora_ast::MatchArm],
) -> Result<Term, String> {
    let (sv, scty) = val(m, b, l, env, scrut)?;
    let merge = b.create_block();
    let mut result: Option<(Variable, Cty)> = None;

    for arm in arms {
        let body_blk = b.create_block();
        let next_blk = b.create_block();
        pattern_test(m, b, l, env, &arm.pat, sv, &scty, body_blk, next_blk)?;

        b.switch_to_block(body_blk);
        b.seal_block(body_blk);
        bind_pattern(b, l, env, &arm.pat, sv, &scty)?;
        if let Some(g) = &arm.guard {
            let (gv, _) = val(m, b, l, env, g)?;
            let real = b.create_block();
            b.ins().brif(gv, real, &[], next_blk, &[]);
            b.switch_to_block(real);
            b.seal_block(real);
        }
        match tr_expr(m, b, l, env, &arm.body)? {
            Term::Val(bv, bcty) => {
                let rvar = match &result {
                    Some((v, _)) => *v,
                    None => {
                        let v = b.declare_var(bcty.clif(env.ptr_ty));
                        result = Some((v, bcty.clone()));
                        v
                    }
                };
                b.def_var(rvar, bv);
                b.ins().jump(merge, &[]);
            }
            Term::Diverged => {} // arm returned; no value to merge
        }

        b.switch_to_block(next_blk);
        b.seal_block(next_blk);
    }

    // Non-exhaustive fall-through: provide a default so the merge is well-formed.
    if let Some((v, cty)) = &result {
        let z = zero_scalar(b, cty);
        b.def_var(*v, z);
    }
    b.ins().jump(merge, &[]);

    b.switch_to_block(merge);
    b.seal_block(merge);
    match result {
        Some((v, cty)) => Ok(Term::Val(b.use_var(v), cty)),
        None => Ok(Term::Val(b.ins().iconst(types::I64, 0), Cty::I64)),
    }
}

/// Emit the branch deciding whether `pat` matches `sv` (-> body or next).
fn pattern_test(
    m: &mut dyn Module,
    b: &mut FunctionBuilder,
    l: &mut Locals,
    env: &Env,
    pat: &Pat,
    sv: Value,
    scty: &Cty,
    body: cranelift::prelude::Block,
    next: cranelift::prelude::Block,
) -> Result<(), String> {
    match &pat.kind {
        PatKind::Wild | PatKind::Binding { .. } => {
            b.ins().jump(body, &[]);
            Ok(())
        }
        PatKind::Path(path)
        | PatKind::TupleStruct { path, .. }
        | PatKind::Struct { path, .. } => {
            let (_, idx) = env.enum_variant(path).ok_or("non-enum variant pattern in JIT match")?;
            let tag = load_at(b, sv, 0, types::I64);
            let want = b.ins().iconst(types::I64, idx as i64);
            let c = b.ins().icmp(IntCC::Equal, tag, want);
            b.ins().brif(c, body, &[], next, &[]);
            Ok(())
        }
        PatKind::Lit(litexpr) => {
            let (lv, lty) = val(m, b, l, env, litexpr)?;
            let c = if lty == Cty::F32 || lty == Cty::F64 {
                b.ins().fcmp(FloatCC::Equal, sv, lv)
            } else {
                b.ins().icmp(IntCC::Equal, sv, lv)
            };
            b.ins().brif(c, body, &[], next, &[]);
            let _ = scty;
            Ok(())
        }
        _ => Err("unsupported match pattern in JIT".into()),
    }
}

/// Bind any variables a (matched) pattern introduces.
fn bind_pattern(
    b: &mut FunctionBuilder,
    l: &mut Locals,
    env: &Env,
    pat: &Pat,
    sv: Value,
    scty: &Cty,
) -> Result<(), String> {
    match &pat.kind {
        PatKind::Binding { name, .. } => {
            bind_name(b, l, env, name.name.clone(), sv, scty.clone());
            Ok(())
        }
        PatKind::TupleStruct { path, elems } => {
            let (enm, idx) = env.enum_variant(path).ok_or("not an enum variant")?;
            let fields = env.enums[&enm].variants[idx].fields.clone();
            for (i, ep) in elems.iter().enumerate() {
                let fcty = fields.get(i).map(|f| f.1.clone()).unwrap_or(Cty::I64);
                let pv = load_at(b, sv, 1 + i, fcty.clif(env.ptr_ty));
                if let PatKind::Binding { name, .. } = &ep.kind {
                    bind_name(b, l, env, name.name.clone(), pv, fcty);
                }
            }
            Ok(())
        }
        PatKind::Struct { path, fields: fpats, .. } => {
            let (enm, idx) = env.enum_variant(path).ok_or("not an enum variant")?;
            let vfields = env.enums[&enm].variants[idx].fields.clone();
            for fp in fpats {
                if let Some(pos) =
                    vfields.iter().position(|(n, _)| n.as_deref() == Some(fp.name.name.as_str()))
                {
                    let fcty = vfields[pos].1.clone();
                    let pv = load_at(b, sv, 1 + pos, fcty.clif(env.ptr_ty));
                    let target = match &fp.pat {
                        Some(inner) => match &inner.kind {
                            PatKind::Binding { name, .. } => Some(name.name.clone()),
                            _ => None,
                        },
                        None => Some(fp.name.name.clone()),
                    };
                    if let Some(n) = target {
                        bind_name(b, l, env, n, pv, fcty);
                    }
                }
            }
            Ok(())
        }
        _ => Ok(()),
    }
}

fn bind_name(b: &mut FunctionBuilder, l: &mut Locals, env: &Env, name: String, v: Value, cty: Cty) {
    let var = b.declare_var(cty.clif(env.ptr_ty));
    b.def_var(var, v);
    l.scope.insert(name, (var, cty));
}

/// `for <pat> in query<...> { body }` over the native ECS world. `&mut`
/// components are pointers into world storage, so writes persist directly.
fn tr_query_loop(
    m: &mut dyn Module,
    b: &mut FunctionBuilder,
    l: &mut Locals,
    env: &Env,
    pat: &Pat,
    q: &aurora_ast::QueryExpr,
    body: &Block,
) -> Result<Term, String> {
    use aurora_ast::QTerm;
    let comp_name_of = |p: &aurora_ast::Path| {
        p.segments.last().map(|s| s.ident.name.clone()).unwrap_or_default()
    };
    let mut required: Vec<String> = Vec::new();
    let mut data: Vec<Option<String>> = Vec::new(); // Some(comp) or None=Entity
    for term in &q.terms {
        match term {
            QTerm::Read(p) | QTerm::Write(p) => {
                let c = comp_name_of(p);
                required.push(c.clone());
                data.push(Some(c));
            }
            QTerm::With(p) => required.push(comp_name_of(p)),
            QTerm::OptRead(p) | QTerm::OptWrite(p) => data.push(Some(comp_name_of(p))),
            QTerm::Entity => data.push(None),
            QTerm::Without(_) => {}
        }
    }
    let bindings = pattern_names(pat);

    // Required component ids array on the stack.
    let ids_ptr = alloc(b, env, required.len().max(1));
    for (i, c) in required.iter().enumerate() {
        let tid = b.ins().iconst(types::I64, comp_id(c));
        store_at(b, ids_ptr, i, tid);
    }
    let n = b.ins().iconst(types::I64, required.len() as i64);
    let qb = m.declare_func_in_func(env.hosts["query_begin"], b.func);
    let qbcall = b.ins().call(qb, &[ids_ptr, n]);
    let count = b.inst_results(qbcall)[0];

    let idx = b.declare_var(types::I64);
    let zero = b.ins().iconst(types::I64, 0);
    b.def_var(idx, zero);
    let count_var = b.declare_var(types::I64);
    b.def_var(count_var, count);

    let header = b.create_block();
    let body_b = b.create_block();
    let exit = b.create_block();
    b.ins().jump(header, &[]);
    b.switch_to_block(header);
    let iv = b.use_var(idx);
    let cv = b.use_var(count_var);
    let c = b.ins().icmp(IntCC::SignedLessThan, iv, cv);
    b.ins().brif(c, body_b, &[], exit, &[]);

    b.switch_to_block(body_b);
    b.seal_block(body_b);
    let qe = m.declare_func_in_func(env.hosts["query_entity"], b.func);
    let qecall = b.ins().call(qe, &[iv]);
    let e = b.inst_results(qecall)[0];
    for (di, term) in data.iter().enumerate() {
        if let Some(Some(bname)) = bindings.get(di) {
            match term {
                Some(comp) => {
                    let tid = b.ins().iconst(types::I64, comp_id(comp));
                    let gc = m.declare_func_in_func(env.hosts["get_component"], b.func);
                    let gccall = b.ins().call(gc, &[e, tid]);
                    let ptr = b.inst_results(gccall)[0];
                    bind_name(b, l, env, bname.clone(), ptr, Cty::Struct(comp.clone()));
                }
                None => bind_name(b, l, env, bname.clone(), e, Cty::I64),
            }
        }
    }
    if let Term::Val(..) = tr_block(m, b, l, env, body)? {
        let i2 = b.use_var(idx);
        let one = b.ins().iconst(types::I64, 1);
        let next = b.ins().iadd(i2, one);
        b.def_var(idx, next);
        b.ins().jump(header, &[]);
    }
    b.seal_block(header);
    b.switch_to_block(exit);
    b.seal_block(exit);
    Ok(Term::Val(b.ins().iconst(types::I64, 0), Cty::I64))
}

/// Positional binding names from a for/match pattern.
fn pattern_names(pat: &Pat) -> Vec<Option<String>> {
    match &pat.kind {
        PatKind::Tuple(pats) => pats
            .iter()
            .map(|p| match &p.kind {
                PatKind::Binding { name, .. } => Some(name.name.clone()),
                _ => None,
            })
            .collect(),
        PatKind::Binding { name, .. } => vec![Some(name.name.clone())],
        _ => vec![None],
    }
}

/// Counting loop with an incrementing `var` while `cmp(var, end())` holds.
fn loop_count(
    m: &mut dyn Module,
    b: &mut FunctionBuilder,
    l: &mut Locals,
    env: &Env,
    var: Variable,
    end: impl Fn(&mut FunctionBuilder) -> Value,
    cc: IntCC,
    body: &Block,
) -> Result<(), String> {
    let header = b.create_block();
    let body_b = b.create_block();
    let step = b.create_block();
    let exit = b.create_block();
    b.ins().jump(header, &[]);
    b.switch_to_block(header);
    let cur = b.use_var(var);
    let e = end(b);
    let c = b.ins().icmp(cc, cur, e);
    b.ins().brif(c, body_b, &[], exit, &[]);
    b.switch_to_block(body_b);
    b.seal_block(body_b);
    // `continue` advances the counter via the step block; `break` exits.
    let cont_used = std::rc::Rc::new(std::cell::Cell::new(false));
    l.loops.push(LoopFrame { continue_to: step, break_to: exit, cont_used: cont_used.clone() });
    let term = tr_block(m, b, l, env, body)?;
    l.loops.pop();
    let body_falls = matches!(term, Term::Val(..));
    if body_falls {
        b.ins().jump(step, &[]);
    }
    b.seal_block(step);
    b.switch_to_block(step);
    if body_falls || cont_used.get() {
        let cur2 = b.use_var(var);
        let one = b.ins().iconst(types::I64, 1);
        let next = b.ins().iadd(cur2, one);
        b.def_var(var, next);
        b.ins().jump(header, &[]);
    } else {
        b.ins().jump(exit, &[]);
    }
    b.seal_block(header);
    b.switch_to_block(exit);
    b.seal_block(exit);
    Ok(())
}

/// Build a `dyn Trait` fat pointer `[data_ptr, type_id]` from a concrete value.
/// `val` must be an aggregate pointer (structs/enums); `type_name` identifies the
/// concrete type for runtime dispatch.
fn coerce_to_dyn(b: &mut FunctionBuilder, env: &Env, val: Value, type_name: &str) -> Value {
    let ptr = alloc(b, env, 2);
    store_at(b, ptr, 0, val); // data pointer
    let tid = b.ins().iconst(types::I64, comp_id(type_name));
    store_at(b, ptr, 1, tid); // type id
    ptr
}

/// Dynamic method dispatch on a `dyn Trait` value: switch on the runtime type id
/// and call the matching concrete `Type#method`. Args/return are scalar i64.
fn tr_dyn_call(
    m: &mut dyn Module,
    b: &mut FunctionBuilder,
    l: &mut Locals,
    env: &Env,
    trait_name: &str,
    method: &str,
    dyn_val: Value,
    args: &[aurora_ast::Arg],
) -> Result<Term, String> {
    let data_ptr = load_at(b, dyn_val, 0, env.ptr_ty);
    let type_id = load_at(b, dyn_val, 1, types::I64);
    let mut argv = vec![data_ptr]; // self = data pointer
    for a in args {
        argv.push(val(m, b, l, env, &a.value)?.0);
    }
    let types = env.trait_types.get(trait_name).cloned().unwrap_or_default();
    if types.is_empty() {
        return Err(format!("no impls of trait `{trait_name}` for dynamic dispatch"));
    }
    // Return type from the first impl's method (all impls share the signature).
    let ret = env
        .methods
        .get(&(types[0].clone(), method.to_string()))
        .map(|k| env.fns[k].ret.clone())
        .unwrap_or(Cty::I64);
    let result = b.declare_var(ret.clif(env.ptr_ty));
    let cont = b.create_block();

    for tn in &types {
        let Some(key) = env.methods.get(&(tn.clone(), method.to_string())) else { continue };
        let id = env.fns[key].id;
        let want = b.ins().iconst(types::I64, comp_id(tn));
        let is_t = b.ins().icmp(IntCC::Equal, type_id, want);
        let call_b = b.create_block();
        let next_b = b.create_block();
        b.ins().brif(is_t, call_b, &[], next_b, &[]);
        b.switch_to_block(call_b);
        b.seal_block(call_b);
        let fref = m.declare_func_in_func(id, b.func);
        let call = b.ins().call(fref, &argv);
        let rv = b.inst_results(call)[0];
        b.def_var(result, rv);
        b.ins().jump(cont, &[]);
        b.switch_to_block(next_b);
        b.seal_block(next_b);
    }
    // No match (shouldn't happen): default to a zero of the right type. Must
    // pick the constant instruction by type — `iconst` with a float type is
    // invalid IR (it panicked the verifier for `dyn` methods returning `f64`).
    let zero = match ret {
        Cty::F64 => b.ins().f64const(0.0),
        Cty::F32 => b.ins().f32const(0.0),
        _ => b.ins().iconst(ret.clif(env.ptr_ty), 0),
    };
    b.def_var(result, zero);
    b.ins().jump(cont, &[]);
    b.switch_to_block(cont);
    b.seal_block(cont);
    Ok(Term::Val(b.use_var(result), ret))
}

fn tr_call(
    m: &mut dyn Module,
    b: &mut FunctionBuilder,
    l: &mut Locals,
    env: &Env,
    callee: &Expr,
    args: &[aurora_ast::Arg],
) -> Result<Term, String> {
    // Method call `recv.method(args)` -> compiled `Type#method(self, args)`.
    if let ExprKind::Field { base, field: FieldAccess::Named(mname) } = &callee.kind {
        let (recv, cty) = val(m, b, l, env, base)?;
        // `dyn Trait` receiver: dispatch dynamically on the runtime type id.
        if let Cty::Dyn(trait_name) = &cty {
            return tr_dyn_call(m, b, l, env, trait_name, &mname.name, recv, args);
        }
        let (Cty::Struct(tyname) | Cty::Enum(tyname)) = &cty else {
            return Err("method receiver must be a struct/enum in JIT".into());
        };
        let key = env
            .methods
            .get(&(tyname.clone(), mname.name.clone()))
            .ok_or_else(|| format!("method `{}::{}` not compiled", tyname, mname.name))?;
        let (id, ret, sret) = {
            let info = &env.fns[key];
            (info.id, info.ret.clone(), info.sret)
        };
        let sret_ptr = if sret { Some(alloc(b, env, agg_slots(env, &ret))) } else { None };
        let mut argv = Vec::new();
        if let Some(sp) = sret_ptr {
            argv.push(sp);
        }
        argv.push(recv); // self
        for a in args {
            argv.push(val(m, b, l, env, &a.value)?.0);
        }
        let fref = m.declare_func_in_func(id, b.func);
        let call = b.ins().call(fref, &argv);
        let result = sret_ptr.unwrap_or_else(|| b.inst_results(call)[0]);
        return Ok(Term::Val(result, ret));
    }

    let ExprKind::Path(p) = &callee.kind else {
        return Err("JIT supports only direct function calls".into());
    };

    // Enum tuple-variant construction `Enum::Variant(args)`.
    if let Some((enm, idx)) = env.enum_variant(p) {
        let slots = env.enums[&enm].slots;
        let ptr = alloc(b, env, slots);
        let tag = b.ins().iconst(types::I64, idx as i64);
        store_at(b, ptr, 0, tag);
        for (i, a) in args.iter().enumerate() {
            let (v, _) = val(m, b, l, env, &a.value)?;
            store_at(b, ptr, 1 + i, v);
        }
        return Ok(Term::Val(ptr, Cty::Enum(enm)));
    }

    // A multi-segment path is a module-qualified call (`math::square`); join the
    // segments with `::` to match the flattened, mangled function name. A single
    // segment is a plain name (and the only form builtins/print take).
    let name = if p.segments.len() > 1 {
        p.segments.iter().map(|s| s.ident.name.as_str()).collect::<Vec<_>>().join("::")
    } else {
        p.segments.first().map(|s| s.ident.name.to_string()).unwrap_or_default()
    };

    if name == "print" || name == "println" {
        emit_print(m, b, l, env, args)?;
        if name == "println" {
            let f = m.declare_func_in_func(env.hosts["print_nl"], b.func);
            b.ins().call(f, &[]);
        }
        return Ok(Term::Val(b.ins().iconst(types::I64, 0), Cty::I64));
    }

    // Builtin graphics — native calls into the host rasterizer.
    if matches!(name.as_str(), "framebuffer" | "clear" | "pixel" | "triangle" | "fb_get") {
        let mut argv = Vec::with_capacity(args.len());
        for a in args {
            let (v, t) = val(m, b, l, env, &a.value)?;
            // Coords/colors are integers; coerce a stray float to int.
            argv.push(if t == Cty::F32 || t == Cty::F64 {
                b.ins().fcvt_to_sint_sat(types::I64, v)
            } else {
                v
            });
        }
        let host = env.hosts[name.as_str()];
        let f = m.declare_func_in_func(host, b.func);
        let call = b.ins().call(f, &argv);
        let result = if name == "fb_get" {
            b.inst_results(call)[0]
        } else {
            b.ins().iconst(types::I64, 0)
        };
        return Ok(Term::Val(result, Cty::I64));
    }
    if name == "save_ppm" || name == "save_png" {
        if let Some(a) = args.first() {
            let (ptr, len) = str_arg(m, b, l, env, &a.value)?;
            let f = m.declare_func_in_func(env.hosts[name.as_str()], b.func);
            b.ins().call(f, &[ptr, len]);
        }
        return Ok(Term::Val(b.ins().iconst(types::I64, 0), Cty::I64));
    }

    // Data builtins with string arguments and/or string results.
    // `read_file(path) -> str` (out-param [ptr,len] like substr).
    if name == "read_file" {
        let (pp, pl) = str_arg(m, b, l, env, &args[0].value)?;
        let out = alloc(b, env, 2);
        let f = m.declare_func_in_func(env.hosts["read_file"], b.func);
        b.ins().call(f, &[out, pp, pl]);
        return Ok(Term::Val(out, Cty::Str));
    }
    // `write_file(path, contents) -> 1|0`.
    if name == "write_file" {
        let (pp, pl) = str_arg(m, b, l, env, &args[0].value)?;
        let (dp, dl) = str_arg(m, b, l, env, &args[1].value)?;
        let f = m.declare_func_in_func(env.hosts["write_file"], b.func);
        let call = b.ins().call(f, &[pp, pl, dp, dl]);
        return Ok(Term::Val(b.inst_results(call)[0], Cty::I64));
    }
    // `file_exists(path)` / `json_parse(text)` / `json_load(path)` /
    // `r3d_capture(path)` -> i64.
    if matches!(name.as_str(), "file_exists" | "json_parse" | "json_load" | "r3d_capture" | "audio_capture_save") {
        let (pp, pl) = str_arg(m, b, l, env, &args[0].value)?;
        let f = m.declare_func_in_func(env.hosts[name.as_str()], b.func);
        let call = b.ins().call(f, &[pp, pl]);
        return Ok(Term::Val(b.inst_results(call)[0], Cty::I64));
    }
    // `r3d_capture_size(path, w, h) -> 1|0`.
    if name == "r3d_capture_size" {
        let (pp, pl) = str_arg(m, b, l, env, &args[0].value)?;
        let (w, wt) = val(m, b, l, env, &args[1].value)?;
        let w = cast(b, w, &wt, &Cty::I64)?;
        let (h, ht) = val(m, b, l, env, &args[2].value)?;
        let h = cast(b, h, &ht, &Cty::I64)?;
        let f = m.declare_func_in_func(env.hosts["r3d_capture_size"], b.func);
        let call = b.ins().call(f, &[pp, pl, w, h]);
        return Ok(Term::Val(b.inst_results(call)[0], Cty::I64));
    }
    // `json_get(h, key)` / `json_has(h, key)` -> i64.
    if name == "json_get" || name == "json_has" {
        let (h, _) = val(m, b, l, env, &args[0].value)?;
        let (kp, kl) = str_arg(m, b, l, env, &args[1].value)?;
        let f = m.declare_func_in_func(env.hosts[name.as_str()], b.func);
        let call = b.ins().call(f, &[h, kp, kl]);
        return Ok(Term::Val(b.inst_results(call)[0], Cty::I64));
    }
    // `json_str(h)` / `json_to_str(h)` -> str.
    if name == "json_str" || name == "json_to_str" {
        let (h, _) = val(m, b, l, env, &args[0].value)?;
        let out = alloc(b, env, 2);
        let f = m.declare_func_in_func(env.hosts[name.as_str()], b.func);
        b.ins().call(f, &[out, h]);
        return Ok(Term::Val(out, Cty::Str));
    }
    // `json_key(h, i) -> str`.
    if name == "json_key" {
        let (h, _) = val(m, b, l, env, &args[0].value)?;
        let (idx, _) = val(m, b, l, env, &args[1].value)?;
        let out = alloc(b, env, 2);
        let f = m.declare_func_in_func(env.hosts["json_key"], b.func);
        b.ins().call(f, &[out, h, idx]);
        return Ok(Term::Val(out, Cty::Str));
    }
    // `json_set(h, key, child)` / `json_set_bool(h, key, b)` (both end in i64).
    if name == "json_set" || name == "json_set_bool" {
        let (h, _) = val(m, b, l, env, &args[0].value)?;
        let (kp, kl) = str_arg(m, b, l, env, &args[1].value)?;
        let (v, vt) = val(m, b, l, env, &args[2].value)?;
        let v = cast(b, v, &vt, &Cty::I64)?;
        let f = m.declare_func_in_func(env.hosts[name.as_str()], b.func);
        b.ins().call(f, &[h, kp, kl, v]);
        return Ok(Term::Val(b.ins().iconst(types::I64, 0), Cty::I64));
    }
    // `json_set_num(h, key, x)`.
    if name == "json_set_num" {
        let (h, _) = val(m, b, l, env, &args[0].value)?;
        let (kp, kl) = str_arg(m, b, l, env, &args[1].value)?;
        let (v, vt) = val(m, b, l, env, &args[2].value)?;
        let v = cast(b, v, &vt, &Cty::F64)?;
        let f = m.declare_func_in_func(env.hosts["json_set_num"], b.func);
        b.ins().call(f, &[h, kp, kl, v]);
        return Ok(Term::Val(b.ins().iconst(types::I64, 0), Cty::I64));
    }
    // `json_set_str(h, key, s)`.
    if name == "json_set_str" {
        let (h, _) = val(m, b, l, env, &args[0].value)?;
        let (kp, kl) = str_arg(m, b, l, env, &args[1].value)?;
        let (sp, sl) = str_arg(m, b, l, env, &args[2].value)?;
        let f = m.declare_func_in_func(env.hosts["json_set_str"], b.func);
        b.ins().call(f, &[h, kp, kl, sp, sl]);
        return Ok(Term::Val(b.ins().iconst(types::I64, 0), Cty::I64));
    }
    // `json_push_str(h, s)`.
    if name == "json_push_str" {
        let (h, _) = val(m, b, l, env, &args[0].value)?;
        let (sp, sl) = str_arg(m, b, l, env, &args[1].value)?;
        let f = m.declare_func_in_func(env.hosts["json_push_str"], b.func);
        b.ins().call(f, &[h, sp, sl]);
        return Ok(Term::Val(b.ins().iconst(types::I64, 0), Cty::I64));
    }
    // `json_write(h, path) -> 1|0`.
    if name == "json_write" {
        let (h, _) = val(m, b, l, env, &args[0].value)?;
        let (pp, pl) = str_arg(m, b, l, env, &args[1].value)?;
        let f = m.declare_func_in_func(env.hosts["json_write"], b.func);
        let call = b.ins().call(f, &[h, pp, pl]);
        return Ok(Term::Val(b.inst_results(call)[0], Cty::I64));
    }
    // `net_set_name("...")` - set the local player's replicated display name from a string.
    if name == "net_set_name" {
        if let Some(a) = args.first() {
            let (ptr, len) = str_arg(m, b, l, env, &a.value)?;
            let f = m.declare_func_in_func(env.hosts["net_set_name"], b.func);
            b.ins().call(f, &[ptr, len]);
        }
        return Ok(Term::Val(b.ins().iconst(types::I64, 0), Cty::I64));
    }
    // `net_set_bot_name(i, "...")` - host sets bot i's replicated display name.
    if name == "net_set_bot_name" {
        if args.len() == 2 {
            let (idx, _) = val(m, b, l, env, &args[0].value)?;
            let (ptr, len) = str_arg(m, b, l, env, &args[1].value)?;
            let f = m.declare_func_in_func(env.hosts["net_set_bot_name"], b.func);
            b.ins().call(f, &[idx, ptr, len]);
        }
        return Ok(Term::Val(b.ins().iconst(types::I64, 0), Cty::I64));
    }

    // Asset + scene builtins: take a path string, return an i64 status.
    if name == "draw_text" {
        // draw_text(x, y, text, px, color)
        if args.len() == 5 {
            let (x, _) = val(m, b, l, env, &args[0].value)?;
            let (y, _) = val(m, b, l, env, &args[1].value)?;
            let (tp, tl) = str_arg(m, b, l, env, &args[2].value)?;
            let (px, _) = val(m, b, l, env, &args[3].value)?;
            let (col, _) = val(m, b, l, env, &args[4].value)?;
            let f = m.declare_func_in_func(env.hosts["draw_text"], b.func);
            b.ins().call(f, &[x, y, tp, tl, px, col]);
        }
        return Ok(Term::Val(b.ins().iconst(types::I64, 0), Cty::I64));
    }
    if matches!(name.as_str(), "load_ppm" | "load_image" | "load_font" | "play_wav" | "load_sound" | "scene_save" | "scene_load" | "r3d_load_model") {
        let result = if let Some(a) = args.first() {
            let (ptr, len) = str_arg(m, b, l, env, &a.value)?;
            let f = m.declare_func_in_func(env.hosts[name.as_str()], b.func);
            let call = b.ins().call(f, &[ptr, len]);
            b.inst_results(call)[0]
        } else {
            b.ins().iconst(types::I64, 0)
        };
        return Ok(Term::Val(result, Cty::I64));
    }

    // `gpu_render("<wgsl>", time_ms)` — run a fragment shader into the framebuffer.
    if name == "gpu_render" {
        if let Some(ExprKind::Str(s)) = args.first().map(|a| &a.value.kind) {
            let (ptr, len) = emit_str_data(m, b, env, s)?;
            let time = if let Some(a) = args.get(1) {
                let (v, t) = val(m, b, l, env, &a.value)?;
                if t == Cty::F32 || t == Cty::F64 {
                    b.ins().fcvt_to_sint_sat(types::I64, v)
                } else {
                    v
                }
            } else {
                b.ins().iconst(types::I64, 0)
            };
            let f = m.declare_func_in_func(env.hosts["gpu_render"], b.func);
            b.ins().call(f, &[ptr, len, time]);
        }
        return Ok(Term::Val(b.ins().iconst(types::I64, 0), Cty::I64));
    }

    // String operations on `Str` values.
    if name == "char_at" {
        let (sp, sl) = str_arg(m, b, l, env, &args[0].value)?;
        let (iv, _) = val(m, b, l, env, &args[1].value)?;
        let f = m.declare_func_in_func(env.hosts["str_char_at"], b.func);
        let call = b.ins().call(f, &[sp, sl, iv]);
        return Ok(Term::Val(b.inst_results(call)[0], Cty::I64));
    }
    if name == "starts_with" {
        let (sp, sl) = str_arg(m, b, l, env, &args[0].value)?;
        let (pp, pl) = str_arg(m, b, l, env, &args[1].value)?;
        let f = m.declare_func_in_func(env.hosts["str_starts_with"], b.func);
        let call = b.ins().call(f, &[sp, sl, pp, pl]);
        return Ok(Term::Val(b.inst_results(call)[0], Cty::I64));
    }
    if name == "substr" {
        let (sp, sl) = str_arg(m, b, l, env, &args[0].value)?;
        let (start, _) = val(m, b, l, env, &args[1].value)?;
        let (n, _) = val(m, b, l, env, &args[2].value)?;
        let out = alloc(b, env, 2);
        let f = m.declare_func_in_func(env.hosts["str_substr"], b.func);
        b.ins().call(f, &[out, sp, sl, start, n]);
        return Ok(Term::Val(out, Cty::Str));
    }

    // `par_for(out_array, closure)` — fill `out[i] = closure(i)` across threads.
    if name == "par_for" {
        let (out, oct) = val(m, b, l, env, &args[0].value)?;
        let n = match &oct {
            Cty::Array(_, n) => *n as i64,
            _ => return Err("par_for expects an array as its first argument in JIT".into()),
        };
        let (cl, _) = val(m, b, l, env, &args[1].value)?;
        let fn_ptr = load_at(b, cl, 0, env.ptr_ty);
        let env_ptr = load_at(b, cl, 1, env.ptr_ty);
        let nv = b.ins().iconst(types::I64, n);
        let f = m.declare_func_in_func(env.hosts["par_for"], b.func);
        b.ins().call(f, &[out, nv, fn_ptr, env_ptr]);
        return Ok(Term::Val(out, oct));
    }

    // `navmesh_build(verts, indices)` / `phys3d_add_trimesh(verts, indices)` -
    // take an `[f64; N]` vertex array (N/3 vertices, xyz each) and an `[i64; M]`
    // triangle-index array; the counts are derived from the array lengths.
    if name == "navmesh_build" || name == "phys3d_add_trimesh" {
        let (vp, vt) = val(m, b, l, env, &args[0].value)?;
        let vcount = match &vt {
            Cty::Array(_, n) => (*n as i64) / 3,
            _ => return Err(format!("{name} expects an [f64; N] vertex array in JIT")),
        };
        let (ip, it) = val(m, b, l, env, &args[1].value)?;
        let icount = match &it {
            Cty::Array(_, n) => *n as i64,
            _ => return Err(format!("{name} expects an [i64; M] index array in JIT")),
        };
        let vn = b.ins().iconst(types::I64, vcount);
        let inn = b.ins().iconst(types::I64, icount);
        let f = m.declare_func_in_func(env.hosts[name.as_str()], b.func);
        let call = b.ins().call(f, &[vp, vn, ip, inn]);
        return Ok(Term::Val(b.inst_results(call)[0], Cty::I64));
    }

    // `gpu_compute(wgsl, arr)` — run a compute shader over an `[f64; n]` array
    // in place on the GPU; returns the (mutated) array.
    if name == "gpu_compute" {
        let (wp, wl) = str_arg(m, b, l, env, &args[0].value)?;
        let (arr, at) = val(m, b, l, env, &args[1].value)?;
        let n = match &at {
            Cty::Array(_, n) => *n as i64,
            _ => return Err("gpu_compute expects an array argument in JIT".into()),
        };
        let nv = b.ins().iconst(types::I64, n);
        let f = m.declare_func_in_func(env.hosts["gpu_compute"], b.func);
        b.ins().call(f, &[wp, wl, arr, nv]);
        return Ok(Term::Val(arr, at));
    }

    // Networking builtins (reliable UDP messaging).
    if name == "net_bind" {
        let (port, _) = val(m, b, l, env, &args[0].value)?;
        let f = m.declare_func_in_func(env.hosts["net_bind"], b.func);
        let call = b.ins().call(f, &[port]);
        return Ok(Term::Val(b.inst_results(call)[0], Cty::I64));
    }
    if name == "net_connect" || name == "net_send" {
        let (p, len) = str_arg(m, b, l, env, &args[0].value)?;
        let f = m.declare_func_in_func(env.hosts[name.as_str()], b.func);
        let call = b.ins().call(f, &[p, len]);
        return Ok(Term::Val(b.inst_results(call)[0], Cty::I64));
    }
    // `net_join(host_str, port)` - a host string plus a port int.
    if name == "net_join" {
        let (p, len) = str_arg(m, b, l, env, &args[0].value)?;
        let (port, pt) = val(m, b, l, env, &args[1].value)?;
        let port = if pt == Cty::F32 || pt == Cty::F64 {
            b.ins().fcvt_to_sint_sat(types::I64, port)
        } else {
            port
        };
        let f = m.declare_func_in_func(env.hosts["net_join"], b.func);
        let call = b.ins().call(f, &[p, len, port]);
        return Ok(Term::Val(b.inst_results(call)[0], Cty::I64));
    }
    if name == "net_recv" {
        let out = alloc(b, env, 2);
        let f = m.declare_func_in_func(env.hosts["net_recv"], b.func);
        b.ins().call(f, &[out]);
        return Ok(Term::Val(out, Cty::Str));
    }
    // `net_sim(move_closure, state_len, input_len)` - register the game's Aurora
    // simulation step. The closure is a `[fn_ptr, env_ptr]` pair (called natively
    // by the netcode each tick over a raw state/input blob, just like par_for).
    if name == "net_sim" {
        // SAFETY GUARD: the netcode STORES this closure and calls it every tick, long after
        // the current function returns. Aurora allocates a closure's captured environment on
        // the creating function's stack frame (see `alloc`), so any capture becomes a dangling
        // pointer once we return -> use-after-free (a segfault that typically strikes the first
        // time the stored callback runs). Reject captures up front with an actionable message.
        if let Some(a0) = args.first() {
            if let Some(lam) = env.closures.get(&a0.value.span) {
                if let Some(caps) = env.lambda_captures.get(lam) {
                    if !caps.is_empty() {
                        return Err(format!(
                            "closure passed to `net_sim` captures outer variable(s) [{}]; the netcode \
                             stores it and runs it on later ticks, after this function returns, so the \
                             captured value (held on this stack frame) dangles. Capture nothing: pass \
                             data through the state/input blob, or recompute it inside the closure.",
                            caps.join(", ")
                        ));
                    }
                }
            }
        }
        let (cl, _) = val(m, b, l, env, &args[0].value)?;
        let fn_ptr = load_at(b, cl, 0, env.ptr_ty);
        let env_ptr = load_at(b, cl, 1, env.ptr_ty);
        let to_i = |b: &mut FunctionBuilder, v: Value, t: &Cty| {
            if *t == Cty::F32 || *t == Cty::F64 {
                b.ins().fcvt_to_sint_sat(types::I64, v)
            } else {
                v
            }
        };
        let (sl, slt) = val(m, b, l, env, &args[1].value)?;
        let (il, ilt) = val(m, b, l, env, &args[2].value)?;
        let sl = to_i(b, sl, &slt);
        let il = to_i(b, il, &ilt);
        let f = m.declare_func_in_func(env.hosts["net_sim"], b.func);
        b.ins().call(f, &[fn_ptr, env_ptr, sl, il]);
        return Ok(Term::Val(b.ins().iconst(types::I64, 0), Cty::I64));
    }
    // `net_serve(server_closure)` - run the authoritative server loop on its own thread. Like
    // net_sim it takes a `[fn_ptr, env_ptr]` closure pair; the engine spawns a thread that calls it.
    if name == "net_serve" {
        // SAFETY GUARD: the server closure runs on ITS OWN thread and must capture nothing (it
        // cannot reference the spawning thread's stack, which may vanish under it). Same dangling
        // stack-env problem as net_sim - reject captures with a clear message.
        if let Some(a0) = args.first() {
            if let Some(lam) = env.closures.get(&a0.value.span) {
                if let Some(caps) = env.lambda_captures.get(lam) {
                    if !caps.is_empty() {
                        return Err(format!(
                            "closure passed to `net_serve` captures outer variable(s) [{}]; it runs on \
                             a separate thread after this function returns, so the captured value (held \
                             on this stack frame) dangles. Capture nothing: the server closure must set \
                             up its own state inside itself.",
                            caps.join(", ")
                        ));
                    }
                }
            }
        }
        let (cl, _) = val(m, b, l, env, &args[0].value)?;
        let fn_ptr = load_at(b, cl, 0, env.ptr_ty);
        let env_ptr = load_at(b, cl, 1, env.ptr_ty);
        let f = m.declare_func_in_func(env.hosts["net_serve"], b.func);
        b.ins().call(f, &[fn_ptr, env_ptr]);
        return Ok(Term::Val(b.ins().iconst(types::I64, 0), Cty::I64));
    }
    // `text_width(text, px)` - pixel width for centering. Works for a string LITERAL
    // or any runtime string value (str_arg handles both), so e.g. text_width(name, 18)
    // on a dynamic label/username measures correctly instead of returning 0.
    if name == "text_width" {
        if let Some(a) = args.first() {
            let (ptr, len) = str_arg(m, b, l, env, &a.value)?;
            let px = if let Some(a) = args.get(1) {
                let (v, t) = val(m, b, l, env, &a.value)?;
                if t == Cty::F32 || t == Cty::F64 {
                    b.ins().fcvt_to_sint_sat(types::I64, v)
                } else {
                    v
                }
            } else {
                b.ins().iconst(types::I64, 0)
            };
            let f = m.declare_func_in_func(env.hosts["text_width"], b.func);
            let call = b.ins().call(f, &[ptr, len, px]);
            return Ok(Term::Val(b.inst_results(call)[0], Cty::I64));
        }
        return Ok(Term::Val(b.ins().iconst(types::I64, 0), Cty::I64));
    }

    // `net_send_input(input_array)` - submit this tick's input blob from an
    // `[f64; n]` array (the length is taken from the array type).
    if name == "net_send_input" {
        let (arr, at) = val(m, b, l, env, &args[0].value)?;
        let n = match &at {
            Cty::Array(_, n) => *n as i64,
            _ => return Err("net_send_input expects an [f64; n] array in JIT".into()),
        };
        let nv = b.ins().iconst(types::I64, n);
        let f = m.declare_func_in_func(env.hosts["net_send_input"], b.func);
        let call = b.ins().call(f, &[arr, nv]);
        return Ok(Term::Val(b.inst_results(call)[0], Cty::I64));
    }

    // `save_settings(arr)` / `load_settings(arr)` - persist/restore an [f64; n] blob to
    // a fixed file. Length is taken from the array type (like net_send_input).
    if name == "save_settings" || name == "load_settings" {
        let (arr, at) = val(m, b, l, env, &args[0].value)?;
        let n = match &at {
            Cty::Array(_, n) => *n as i64,
            _ => return Err("save_settings/load_settings expect an [f64; n] array".into()),
        };
        let nv = b.ins().iconst(types::I64, n);
        let f = m.declare_func_in_func(env.hosts[name.as_str()], b.func);
        let call = b.ins().call(f, &[arr, nv]);
        return Ok(Term::Val(b.inst_results(call)[0], Cty::I64));
    }

    // Type-aware scalar builtins (physics, pathfinding): each argument is
    // coerced to the host function's declared parameter type and the result is
    // returned with the right `Cty` (e.g. `phys_x` returns `f64`).
    if let Some((params, ret)) = scalar_builtin_sig(name.as_str()) {
        if args.len() == params.len() {
            let mut argv = Vec::with_capacity(args.len());
            for (a, pc) in args.iter().zip(params.iter()) {
                let (v, t) = val(m, b, l, env, &a.value)?;
                argv.push(cast(b, v, &t, &abi_cty(*pc))?);
            }
            let f = m.declare_func_in_func(env.hosts[name.as_str()], b.func);
            let call = b.ins().call(f, &argv);
            let result = match ret {
                Some(_) => b.inst_results(call)[0],
                None => b.ins().iconst(types::I64, 0),
            };
            return Ok(Term::Val(result, ret.map_or(Cty::I64, abi_cty)));
        }
    }

    // Text builtins: same table-driven dispatch, extended to `str`. A `Ptr`
    // parameter is one Aurora `str` argument passed as its two slots, and a
    // `Str` result is a caller-allocated 2-slot out-pointer passed first.
    if let Some(row) = aurora_abi::text_row(name.as_str()) {
        if Some(args.len()) == row.arity() {
            let mut argv = Vec::new();
            if row.ret == Some(aurora_abi::Ty::Str) {
                argv.push(alloc(b, env, 2));
            }
            let mut arg = args.iter();
            let mut p = row.params.iter();
            while let Some(pc) = p.next() {
                let a = arg.next().expect("arity checked above");
                if *pc == aurora_abi::Ty::Ptr {
                    // Its `I64` length slot belongs to the same argument.
                    p.next();
                    let (ptr, len) = str_arg(m, b, l, env, &a.value)?;
                    argv.push(ptr);
                    argv.push(len);
                } else {
                    let (v, t) = val(m, b, l, env, &a.value)?;
                    argv.push(cast(b, v, &t, &abi_cty(*pc))?);
                }
            }
            let f = m.declare_func_in_func(env.hosts[name.as_str()], b.func);
            let call = b.ins().call(f, &argv);
            return Ok(match row.ret {
                Some(aurora_abi::Ty::Str) => Term::Val(argv[0], Cty::Str),
                Some(t) => Term::Val(b.inst_results(call)[0], abi_cty(t)),
                None => Term::Val(b.ins().iconst(types::I64, 0), Cty::I64),
            });
        }
    }

    // `frame_reset()` — free the frame arena (no args, no result).
    if name == "frame_reset" {
        let f = m.declare_func_in_func(env.hosts["frame_reset"], b.func);
        b.ins().call(f, &[]);
        return Ok(Term::Val(b.ins().iconst(types::I64, 0), Cty::I64));
    }

    // Audio + windowing + input builtins (integer args; some return an integer).
    if matches!(
        name.as_str(),
        "play_note"
            | "play_sound"
            | "play_noise"
            | "draw_int"
            | "audio_volume"
            | "window_fullscreen"
            | "audio_stop"
            | "window_open"
            | "window_present"
            | "surface_w"
            | "surface_h"
            | "key_down"
            | "input_char"
            | "mouse_x"
            | "mouse_y"
            | "mouse_down"
    ) {
        let mut argv = Vec::with_capacity(args.len());
        for a in args {
            let (v, t) = val(m, b, l, env, &a.value)?;
            argv.push(if t == Cty::F32 || t == Cty::F64 {
                b.ins().fcvt_to_sint_sat(types::I64, v)
            } else {
                v
            });
        }
        let f = m.declare_func_in_func(env.hosts[name.as_str()], b.func);
        let call = b.ins().call(f, &argv);
        let returns_int = matches!(
            name.as_str(),
            "window_present" | "surface_w" | "surface_h" | "key_down" | "input_char" | "mouse_x" | "mouse_y" | "mouse_down"
        );
        let result = if returns_int {
            b.inst_results(call)[0]
        } else {
            b.ins().iconst(types::I64, 0)
        };
        return Ok(Term::Val(result, Cty::I64));
    }

    // ECS builtins (native runtime).
    if name == "spawn" {
        let f = m.declare_func_in_func(env.hosts["spawn_entity"], b.func);
        let call = b.ins().call(f, &[]);
        let e = b.inst_results(call)[0];
        for a in args {
            let (cptr, cty) = val(m, b, l, env, &a.value)?;
            let Cty::Struct(cname) = &cty else {
                return Err("spawn arguments must be components (JIT)".into());
            };
            let tid = b.ins().iconst(types::I64, comp_id(cname));
            let nbytes = byte_size(env, &cty);
            let size = b.ins().iconst(types::I64, nbytes as i64);
            let sf = m.declare_func_in_func(env.hosts["store_component"], b.func);
            b.ins().call(sf, &[e, tid, cptr, size]);
        }
        return Ok(Term::Val(e, Cty::I64));
    }
    if name == "despawn" {
        let (e, _) = val(m, b, l, env, &args[0].value)?;
        let f = m.declare_func_in_func(env.hosts["despawn"], b.func);
        b.ins().call(f, &[e]);
        return Ok(Term::Val(b.ins().iconst(types::I64, 0), Cty::I64));
    }
    if name == "entity_count" {
        let f = m.declare_func_in_func(env.hosts["entity_count"], b.func);
        let call = b.ins().call(f, &[]);
        let n = b.inst_results(call)[0];
        return Ok(Term::Val(n, Cty::I64));
    }
    if name == "run_systems" {
        // Run the schedule layer by layer. A single-system layer is a direct
        // call; a multi-system layer is handed to `aurora_run_parallel`, which
        // runs its (provably non-conflicting, unordered) systems concurrently
        // over the shared world. Layer order preserves declaration order for
        // every conflicting or explicitly-ordered pair.
        for layer in &env.system_layers {
            if layer.len() == 1 {
                let id = env.fns[&env.system_order[layer[0]]].id;
                let fref = m.declare_func_in_func(id, b.func);
                b.ins().call(fref, &[]);
            } else {
                let arr = alloc(b, env, layer.len());
                for (k, &si) in layer.iter().enumerate() {
                    let id = env.fns[&env.system_order[si]].id;
                    let fref = m.declare_func_in_func(id, b.func);
                    let faddr = b.ins().func_addr(env.ptr_ty, fref);
                    store_at(b, arr, k, faddr);
                }
                let n = b.ins().iconst(types::I64, layer.len() as i64);
                let run_par = m.declare_func_in_func(env.hosts["run_parallel"], b.func);
                b.ins().call(run_par, &[arr, n]);
            }
        }
        return Ok(Term::Val(b.ins().iconst(types::I64, 0), Cty::I64));
    }

    // Indirect call: a local variable holding a closure pair [fn_ptr, env_ptr].
    if let Some((var, Cty::Fn(param_ctys, ret_cty))) = l.scope.get(&name).cloned() {
        let cl = b.use_var(var);
        let fn_ptr = load_at(b, cl, 0, env.ptr_ty);
        let env_ptr = load_at(b, cl, 1, env.ptr_ty);
        let mut argv = vec![env_ptr];
        for (idx, a) in args.iter().enumerate() {
            let (av, at) = val(m, b, l, env, &a.value)?;
            // Coerce the argument to the closure's declared parameter type
            // (a real numeric conversion, e.g. i64→f64), then pass it in a raw
            // i64 slot; the lambda reinterprets it. This keeps the call and the
            // lambda body in agreement even for inferred/unannotated params.
            let pc = param_ctys.get(idx).cloned().unwrap_or(at.clone());
            let coerced = cast(b, av, &at, &pc)?;
            argv.push(to_i64_bits(b, coerced, &pc));
        }
        let mut sig = cranelift::codegen::ir::Signature::new(m.target_config().default_call_conv);
        for _ in &argv {
            sig.params.push(AbiParam::new(types::I64));
        }
        sig.returns.push(AbiParam::new(types::I64));
        let sigref = b.import_signature(sig);
        let call = b.ins().call_indirect(sigref, fn_ptr, &argv);
        // The result comes back as i64 bits; reinterpret to the closure's
        // declared return type.
        let raw = b.inst_results(call)[0];
        let result = from_i64_bits(b, raw, &ret_cty);
        return Ok(Term::Val(result, (*ret_cty).clone()));
    }

    let mut typed = Vec::with_capacity(args.len());
    for a in args {
        typed.push(val(m, b, l, env, &a.value)?);
    }
    if let Some(result) = math_builtin(b, &name, &typed) {
        return Ok(Term::Val(result.0, result.1));
    }

    // Transcendental math (sin/cos/tan/pow/log/exp/atan2): no native Cranelift
    // instruction, so these are host calls into libm. Args are coerced to f64;
    // the result is demoted back to f32 if the (first) argument was f32, so the
    // builtin is float-width-preserving like the native ones.
    if matches!(name.as_str(), "sin" | "cos" | "tan" | "pow" | "log" | "exp" | "atan2") {
        let want = if matches!(name.as_str(), "pow" | "atan2") { 2 } else { 1 };
        if typed.len() == want && typed.iter().all(|(_, t)| *t == Cty::F32 || *t == Cty::F64) {
            let was_f32 = typed[0].1 == Cty::F32;
            let mut argv = Vec::with_capacity(want);
            for (v, t) in &typed {
                argv.push(if *t == Cty::F32 { b.ins().fpromote(types::F64, *v) } else { *v });
            }
            let f = m.declare_func_in_func(env.hosts[name.as_str()], b.func);
            let call = b.ins().call(f, &argv);
            let r = b.inst_results(call)[0];
            return if was_f32 {
                Ok(Term::Val(b.ins().fdemote(types::F32, r), Cty::F32))
            } else {
                Ok(Term::Val(r, Cty::F64))
            };
        }
    }

    // `len(x)` — string length (slot 1) or fixed-array length.
    if name == "len" {
        if let Some((v, t)) = typed.first() {
            let n = match t {
                Cty::Str => load_at(b, *v, 1, types::I64),
                Cty::Array(_, n) => b.ins().iconst(types::I64, *n as i64),
                _ => b.ins().iconst(types::I64, 0),
            };
            return Ok(Term::Val(n, Cty::I64));
        }
    }

    // `str(x)` — convert a value to a string.
    if name == "str" {
        if let Some((v, t)) = typed.first().cloned() {
            if t == Cty::Str {
                return Ok(Term::Val(v, Cty::Str));
            }
            let out = alloc(b, env, 2);
            if t == Cty::F32 || t == Cty::F64 {
                let v64 = if t == Cty::F32 { b.ins().fpromote(types::F64, v) } else { v };
                let f = m.declare_func_in_func(env.hosts["float_to_str"], b.func);
                b.ins().call(f, &[out, v64]);
            } else {
                let f = m.declare_func_in_func(env.hosts["int_to_str"], b.func);
                b.ins().call(f, &[out, v]);
            }
            return Ok(Term::Val(out, Cty::Str));
        }
    }

    let (id, ret, sret, params) = {
        let info = env
            .fns
            .get(&name)
            .ok_or_else(|| format!("call to non-scalar/uncompiled function `{name}`"))?;
        (info.id, info.ret.clone(), info.sret, info.params.clone())
    };
    if typed.len() != params.len() {
        return Err(format!("`{name}` arity mismatch in JIT"));
    }
    // Aggregate return uses a caller-allocated sret slot (leading argument).
    let sret_ptr = if sret { Some(alloc(b, env, agg_slots(env, &ret))) } else { None };
    let mut argv: Vec<Value> = Vec::new();
    if let Some(sp) = sret_ptr {
        argv.push(sp);
    }
    let is_extern = env.extern_fns.contains(&name);
    // Coerce a concrete argument to `dyn Trait` where the parameter expects one.
    for ((v, vt), pt) in typed.iter().zip(&params) {
        let arg = match (pt, vt) {
            // An `@extern` call whose parameter is an aggregate containing `f32`
            // gets it repacked into C's layout first.
            _ if is_extern && is_aggregate(pt) && ffi_needs_marshal(pt, &env.structs) => {
                marshal_to_c(b, env, *v, pt)
            }
            (Cty::Dyn(_), Cty::Struct(tn)) | (Cty::Dyn(_), Cty::Enum(tn)) => {
                coerce_to_dyn(b, env, *v, tn)
            }
            _ => *v,
        };
        argv.push(arg);
    }
    let callee_ref = m.declare_func_in_func(id, b.func);
    let call = b.ins().call(callee_ref, &argv);
    let result = sret_ptr.unwrap_or_else(|| b.inst_results(call)[0]);
    Ok(Term::Val(result, ret))
}

/// `lhs op= rhs` / `lhs = rhs`, where lhs is a variable, field, or index.
fn assign(
    m: &mut dyn Module,
    b: &mut FunctionBuilder,
    l: &mut Locals,
    env: &Env,
    lhs: &Expr,
    op: &Option<BinOp>,
    rv: Value,
    rt: Cty,
) -> Result<(), String> {
    match &lhs.kind {
        ExprKind::Path(p) if p.is_single() => {
            let name = &p.segments[0].ident.name;
            let (var, vty) = l
                .scope
                .get(name)
                .cloned()
                .ok_or_else(|| format!("assignment to unbound variable `{name}` in JIT"))?;
            let report_cty = vty.clone();
            let newv = match op {
                None => rv,
                Some(binop) => {
                    let cur = b.use_var(var);
                    apply_bin(b, *binop, cur, vty, rv, rt)?.0
                }
            };
            b.def_var(var, newv);
            // Report the updated value to an attached debugger.
            if env.debug {
                emit_dbg_value(m, b, env, name, newv, &report_cty);
            }
            Ok(())
        }
        ExprKind::Field { base, field } => {
            let (ptr, cty) = val(m, b, l, env, base)?;
            let (off, fcty) = field_offset(env, &cty, field)?;
            if !fcty.is_scalar() {
                if op.is_some() {
                    return Err("compound assign to an aggregate field in JIT".into());
                }
                copy_agg(b, env, ptr, off, rv, &fcty); // rv is the rhs pointer
                return Ok(());
            }
            let newv = match op {
                None => rv,
                Some(binop) => {
                    let cur = load_b(b, ptr, off, fcty.clif(env.ptr_ty));
                    apply_bin(b, *binop, cur, fcty, rv, rt)?.0
                }
            };
            store_b(b, ptr, off, newv);
            Ok(())
        }
        ExprKind::Index { base, index } => {
            let (ptr, cty) = val(m, b, l, env, base)?;
            let Cty::Array(elem, len) = &cty else {
                return Err("indexed assignment to a non-array in JIT".into());
            };
            let len = *len;
            let elem = (**elem).clone();
            let stride = byte_size(env, &elem);
            let (iv, _) = val(m, b, l, env, index)?;
            emit_bounds_check(m, b, env, iv, len);
            let stridev = b.ins().iconst(types::I64, stride as i64);
            let off = b.ins().imul(iv, stridev);
            let addr = b.ins().iadd(ptr, off);
            if !elem.is_scalar() {
                if op.is_some() {
                    return Err("compound assign to an aggregate element in JIT".into());
                }
                copy_agg(b, env, addr, 0, rv, &elem);
                return Ok(());
            }
            let newv = match op {
                None => rv,
                Some(binop) => {
                    let cur = b.ins().load(elem.clif(env.ptr_ty), MemFlags::new(), addr, 0);
                    apply_bin(b, *binop, cur, elem, rv, rt)?.0
                }
            };
            b.ins().store(MemFlags::new(), newv, addr, 0);
            Ok(())
        }
        _ => Err("unsupported assignment target in JIT".into()),
    }
}

/// Byte offset + type of a struct field or tuple element.
fn field_offset(env: &Env, cty: &Cty, field: &FieldAccess) -> Result<(u32, Cty), String> {
    match (cty, field) {
        (Cty::Struct(name), FieldAccess::Named(id)) => {
            struct_field(env, name, &id.name).ok_or_else(|| format!("no field `{}` in JIT", id.name))
        }
        (Cty::Tuple(tys), FieldAccess::Index(i)) => {
            let i = *i as usize;
            if i >= tys.len() {
                return Err("tuple index out of range in JIT".into());
            }
            let off: u32 = tys[..i].iter().map(|t| byte_size(env, t)).sum();
            Ok((off, tys[i].clone()))
        }
        _ => Err("invalid field access in JIT".into()),
    }
}

fn emit_print(
    m: &mut dyn Module,
    b: &mut FunctionBuilder,
    l: &mut Locals,
    env: &Env,
    args: &[aurora_ast::Arg],
) -> Result<(), String> {
    for (idx, a) in args.iter().enumerate() {
        // Separate multiple arguments with a space, matching the interpreter.
        if idx > 0 {
            let (sp, sl) = emit_str_data(m, b, env, " ")?;
            let f = m.declare_func_in_func(env.hosts["print_str"], b.func);
            b.ins().call(f, &[sp, sl]);
        }
        if let ExprKind::Str(s) = &a.value.kind {
            let (ptr, len) = emit_str_data(m, b, env, s)?;
            let f = m.declare_func_in_func(env.hosts["print_str"], b.func);
            b.ins().call(f, &[ptr, len]);
        } else {
            let (v, t) = val(m, b, l, env, &a.value)?;
            if t == Cty::Str {
                // A string value: load `[data_ptr, len]` and print the bytes.
                let dptr = load_at(b, v, 0, env.ptr_ty);
                let len = load_at(b, v, 1, types::I64);
                let f = m.declare_func_in_func(env.hosts["print_str"], b.func);
                b.ins().call(f, &[dptr, len]);
            } else if t == Cty::F32 || t == Cty::F64 {
                let v64 = if t == Cty::F32 { b.ins().fpromote(types::F64, v) } else { v };
                let f = m.declare_func_in_func(env.hosts["print_f64"], b.func);
                b.ins().call(f, &[v64]);
            } else {
                let f = m.declare_func_in_func(env.hosts["print_i64"], b.func);
                b.ins().call(f, &[v]);
            }
        }
    }
    Ok(())
}

/// Emit a string's bytes into a data object; return (pointer, length) values.
fn emit_str_data(
    m: &mut dyn Module,
    b: &mut FunctionBuilder,
    env: &Env,
    s: &str,
) -> Result<(Value, Value), String> {
    let data_id = m.declare_anonymous_data(false, false).map_err(|e| format!("data: {e}"))?;
    let mut desc = DataDescription::new();
    desc.define(s.to_string().into_bytes().into_boxed_slice());
    m.define_data(data_id, &desc).map_err(|e| format!("data: {e}"))?;
    let gv = m.declare_data_in_func(data_id, b.func);
    let ptr = b.ins().global_value(env.ptr_ty, gv);
    let len = b.ins().iconst(types::I64, s.len() as i64);
    Ok((ptr, len))
}

/// Produce `(data_ptr, len)` for a string argument — either a literal (emitted
/// as static data) or a `Str` value (its `[ptr, len]` slots loaded).
fn str_arg(
    m: &mut dyn Module,
    b: &mut FunctionBuilder,
    l: &mut Locals,
    env: &Env,
    e: &Expr,
) -> Result<(Value, Value), String> {
    if let ExprKind::Str(s) = &e.kind {
        return emit_str_data(m, b, env, s);
    }
    let (v, t) = val(m, b, l, env, e)?;
    if t != Cty::Str {
        return Err("expected a string argument in JIT".into());
    }
    Ok((load_at(b, v, 0, env.ptr_ty), load_at(b, v, 1, types::I64)))
}

fn tr_binary(
    m: &mut dyn Module,
    b: &mut FunctionBuilder,
    l: &mut Locals,
    env: &Env,
    op: BinOp,
    a: &Expr,
    c: &Expr,
) -> Result<(Value, Cty), String> {
    // Short-circuit `and`/`or`: evaluate the right side only when needed (so
    // `i < len and arr[i] > 0` never indexes when `i >= len`), and yield a
    // canonical 0/1 (so `2 and 1` is true, not `band(2,1)=0`).
    if matches!(op, BinOp::And | BinOp::Or) {
        let (av, _) = val(m, b, l, env, a)?;
        let a_true = b.ins().icmp_imm(IntCC::NotEqual, av, 0);
        let result = b.declare_var(types::I64);
        let rhs_b = b.create_block();
        let short_b = b.create_block();
        let merge = b.create_block();
        // `and`: if a is true, result = (rhs != 0); else result = 0.
        // `or`:  if a is true, result = 1;          else result = (rhs != 0).
        if op == BinOp::And {
            b.ins().brif(a_true, rhs_b, &[], short_b, &[]);
        } else {
            b.ins().brif(a_true, short_b, &[], rhs_b, &[]);
        }
        b.switch_to_block(rhs_b);
        b.seal_block(rhs_b);
        let (cv, _) = val(m, b, l, env, c)?;
        let c_true = b.ins().icmp_imm(IntCC::NotEqual, cv, 0);
        let c_i64 = b.ins().uextend(types::I64, c_true);
        b.def_var(result, c_i64);
        b.ins().jump(merge, &[]);
        b.switch_to_block(short_b);
        b.seal_block(short_b);
        let short_val = b.ins().iconst(types::I64, if op == BinOp::And { 0 } else { 1 });
        b.def_var(result, short_val);
        b.ins().jump(merge, &[]);
        b.switch_to_block(merge);
        b.seal_block(merge);
        return Ok((b.use_var(result), Cty::I64));
    }

    let (av, at) = val(m, b, l, env, a)?;
    let (cv, ct) = val(m, b, l, env, c)?;
    // String operations: `+` concatenates, `==`/`!=` compare by bytes.
    if at == Cty::Str || ct == Cty::Str {
        let (ap, al) = (load_at(b, av, 0, env.ptr_ty), load_at(b, av, 1, types::I64));
        let (cp, cl) = (load_at(b, cv, 0, env.ptr_ty), load_at(b, cv, 1, types::I64));
        match op {
            BinOp::Add => {
                let out = alloc(b, env, 2);
                let f = m.declare_func_in_func(env.hosts["str_concat"], b.func);
                b.ins().call(f, &[out, ap, al, cp, cl]);
                return Ok((out, Cty::Str));
            }
            BinOp::Eq | BinOp::Ne => {
                let f = m.declare_func_in_func(env.hosts["str_eq"], b.func);
                let call = b.ins().call(f, &[ap, al, cp, cl]);
                let mut eq = b.inst_results(call)[0];
                if op == BinOp::Ne {
                    let one = b.ins().iconst(types::I64, 1);
                    eq = b.ins().bxor(eq, one);
                }
                return Ok((eq, Cty::I64));
            }
            _ => return Err("unsupported string operator in JIT".into()),
        }
    }

    // Division / remainder need care: integer div/rem by zero must panic cleanly
    // (not a raw CPU trap), and float remainder has no Cranelift instruction so it
    // goes through libm fmod.
    if matches!(op, BinOp::Div | BinOp::Rem) && at == ct {
        let is_float = at == Cty::F32 || at == Cty::F64;
        if is_float && op == BinOp::Rem {
            let (a64, c64) = if at == Cty::F32 {
                (b.ins().fpromote(types::F64, av), b.ins().fpromote(types::F64, cv))
            } else {
                (av, cv)
            };
            let f = m.declare_func_in_func(env.hosts["fmod"], b.func);
            let call = b.ins().call(f, &[a64, c64]);
            let mut r = b.inst_results(call)[0];
            if at == Cty::F32 {
                r = b.ins().fdemote(types::F32, r);
            }
            return Ok((r, at));
        }
        if !is_float {
            // Guard divisor != 0 -> clean panic via the runtime.
            let is_zero = b.ins().icmp_imm(IntCC::Equal, cv, 0);
            let fail = b.create_block();
            let ok = b.create_block();
            b.ins().brif(is_zero, fail, &[], ok, &[]);
            b.switch_to_block(fail);
            b.seal_block(fail);
            let f = m.declare_func_in_func(env.hosts["divzero"], b.func);
            b.ins().call(f, &[]);
            b.ins().trap(TrapCode::INTEGER_DIVISION_BY_ZERO);
            b.switch_to_block(ok);
            b.seal_block(ok);
            // fall through to apply_bin for the actual sdiv/srem
        }
    }
    apply_bin(b, op, av, at, cv, ct)
}

fn apply_bin(
    b: &mut FunctionBuilder,
    op: BinOp,
    av: Value,
    at: Cty,
    cv: Value,
    ct: Cty,
) -> Result<(Value, Cty), String> {
    if at != ct || !at.is_scalar() {
        return Err("binary op needs matching scalar operands (JIT)".into());
    }
    let is_float = at == Cty::F32 || at == Cty::F64;
    let v = match op {
        BinOp::Add if is_float => b.ins().fadd(av, cv),
        BinOp::Sub if is_float => b.ins().fsub(av, cv),
        BinOp::Mul if is_float => b.ins().fmul(av, cv),
        BinOp::Div if is_float => b.ins().fdiv(av, cv),
        BinOp::Rem if is_float => return Err("float remainder not supported in JIT".into()),
        BinOp::Add => b.ins().iadd(av, cv),
        BinOp::Sub => b.ins().isub(av, cv),
        BinOp::Mul => b.ins().imul(av, cv),
        BinOp::Div => b.ins().sdiv(av, cv),
        BinOp::Rem => b.ins().srem(av, cv),
        BinOp::And => return Ok((b.ins().band(av, cv), Cty::I64)),
        BinOp::Or => return Ok((b.ins().bor(av, cv), Cty::I64)),
        BinOp::Eq | BinOp::Ne | BinOp::Lt | BinOp::Gt | BinOp::Le | BinOp::Ge => {
            let cmp = if is_float {
                let cc = match op {
                    BinOp::Eq => FloatCC::Equal,
                    BinOp::Ne => FloatCC::NotEqual,
                    BinOp::Lt => FloatCC::LessThan,
                    BinOp::Gt => FloatCC::GreaterThan,
                    BinOp::Le => FloatCC::LessThanOrEqual,
                    BinOp::Ge => FloatCC::GreaterThanOrEqual,
                    _ => unreachable!(),
                };
                b.ins().fcmp(cc, av, cv)
            } else {
                let cc = match op {
                    BinOp::Eq => IntCC::Equal,
                    BinOp::Ne => IntCC::NotEqual,
                    BinOp::Lt => IntCC::SignedLessThan,
                    BinOp::Gt => IntCC::SignedGreaterThan,
                    BinOp::Le => IntCC::SignedLessThanOrEqual,
                    BinOp::Ge => IntCC::SignedGreaterThanOrEqual,
                    _ => unreachable!(),
                };
                b.ins().icmp(cc, av, cv)
            };
            return Ok((b.ins().uextend(types::I64, cmp), Cty::I64));
        }
    };
    Ok((v, at))
}

fn math_builtin(b: &mut FunctionBuilder, name: &str, args: &[(Value, Cty)]) -> Option<(Value, Cty)> {
    let is_float = |t: &Cty| *t == Cty::F32 || *t == Cty::F64;
    match (name, args) {
        ("sqrt", [(v, t)]) if is_float(t) => Some((b.ins().sqrt(*v), t.clone())),
        ("floor", [(v, t)]) if is_float(t) => Some((b.ins().floor(*v), t.clone())),
        ("ceil", [(v, t)]) if is_float(t) => Some((b.ins().ceil(*v), t.clone())),
        ("round", [(v, t)]) if is_float(t) => Some((b.ins().nearest(*v), t.clone())),
        ("abs", [(v, t)]) if is_float(t) => Some((b.ins().fabs(*v), t.clone())),
        ("min", [(a, t), (c, _)]) if is_float(t) => Some((b.ins().fmin(*a, *c), t.clone())),
        ("max", [(a, t), (c, _)]) if is_float(t) => Some((b.ins().fmax(*a, *c), t.clone())),
        // clamp(x, lo, hi) = min(max(x, lo), hi), all native.
        ("clamp", [(x, t), (lo, _), (hi, _)]) if is_float(t) => {
            let lower = b.ins().fmax(*x, *lo);
            Some((b.ins().fmin(lower, *hi), t.clone()))
        }
        // Integer abs/min/max/clamp (these were unhandled, so any call on i64 args
        // silently stubbed the whole function). Signed throughout.
        ("abs", [(v, t)]) if !is_float(t) => {
            let neg = b.ins().ineg(*v);
            let is_neg = b.ins().icmp_imm(IntCC::SignedLessThan, *v, 0);
            Some((b.ins().select(is_neg, neg, *v), Cty::I64))
        }
        ("min", [(a, t), (c, _)]) if !is_float(t) => {
            let a_lt = b.ins().icmp(IntCC::SignedLessThan, *a, *c);
            Some((b.ins().select(a_lt, *a, *c), Cty::I64))
        }
        ("max", [(a, t), (c, _)]) if !is_float(t) => {
            let a_lt = b.ins().icmp(IntCC::SignedLessThan, *a, *c);
            Some((b.ins().select(a_lt, *c, *a), Cty::I64))
        }
        ("clamp", [(x, t), (lo, _), (hi, _)]) if !is_float(t) => {
            let below = b.ins().icmp(IntCC::SignedLessThan, *x, *lo);
            let lower = b.ins().select(below, *lo, *x);
            let above = b.ins().icmp(IntCC::SignedLessThan, *hi, lower);
            Some((b.ins().select(above, *hi, lower), Cty::I64))
        }
        // Integer bitwise ops (flags, masks, packing). `&`/`|` are taken by
        // references and closures, so these are spelled as functions.
        ("band", [(a, t), (c, _)]) if !is_float(t) => Some((b.ins().band(*a, *c), Cty::I64)),
        ("bor", [(a, t), (c, _)]) if !is_float(t) => Some((b.ins().bor(*a, *c), Cty::I64)),
        ("bxor", [(a, t), (c, _)]) if !is_float(t) => Some((b.ins().bxor(*a, *c), Cty::I64)),
        ("shl", [(a, t), (c, _)]) if !is_float(t) => Some((b.ins().ishl(*a, *c), Cty::I64)),
        ("shr", [(a, t), (c, _)]) if !is_float(t) => Some((b.ins().sshr(*a, *c), Cty::I64)),
        ("bnot", [(v, t)]) if !is_float(t) => Some((b.ins().bnot(*v), Cty::I64)),
        _ => None,
    }
}

fn cast(b: &mut FunctionBuilder, v: Value, from: &Cty, to: &Cty) -> Result<Value, String> {
    if from == to {
        return Ok(v);
    }
    let f = |t: &Cty| *t == Cty::F32 || *t == Cty::F64;
    Ok(match (f(from), f(to)) {
        (false, true) => b.ins().fcvt_from_sint(to.clif(types::I64), v),
        (true, false) => b.ins().fcvt_to_sint_sat(types::I64, v),
        (true, true) => {
            if *to == Cty::F64 {
                b.ins().fpromote(types::F64, v)
            } else {
                b.ins().fdemote(types::F32, v)
            }
        }
        (false, false) => v,
    })
}

fn tr_value_if(
    m: &mut dyn Module,
    b: &mut FunctionBuilder,
    l: &mut Locals,
    env: &Env,
    ifx: &aurora_ast::IfExpr,
) -> Result<Term, String> {
    let Some(else_e) = &ifx.else_branch else {
        return Err("`if` used as a value must have an `else` in the JIT".into());
    };
    let (cond, _) = val(m, b, l, env, &ifx.cond)?;
    let then_b = b.create_block();
    let else_b = b.create_block();
    let merge_b = b.create_block();
    b.ins().brif(cond, then_b, &[], else_b, &[]);

    b.switch_to_block(then_b);
    b.seal_block(then_b);
    let (tv, ty) = block_val(m, b, l, env, &ifx.then_branch)?;
    let result = b.declare_var(ty.clif(env.ptr_ty));
    b.def_var(result, tv);
    b.ins().jump(merge_b, &[]);

    b.switch_to_block(else_b);
    b.seal_block(else_b);
    let (ev, ety) = val(m, b, l, env, else_e)?;
    if ety != ty {
        return Err("`if` branches have different types in JIT".into());
    }
    b.def_var(result, ev);
    b.ins().jump(merge_b, &[]);

    b.switch_to_block(merge_b);
    b.seal_block(merge_b);
    Ok(Term::Val(b.use_var(result), ty))
}

fn block_val(
    m: &mut dyn Module,
    b: &mut FunctionBuilder,
    l: &mut Locals,
    env: &Env,
    block: &Block,
) -> Result<(Value, Cty), String> {
    match tr_block(m, b, l, env, block)? {
        Term::Val(v, t) => Ok((v, t)),
        Term::Diverged => Err("diverging branch where a value is required (JIT)".into()),
    }
}

fn val(
    m: &mut dyn Module,
    b: &mut FunctionBuilder,
    l: &mut Locals,
    env: &Env,
    e: &Expr,
) -> Result<(Value, Cty), String> {
    match tr_expr(m, b, l, env, e)? {
        Term::Val(v, t) => Ok((v, t)),
        Term::Diverged => Err("diverging expression used where a value is required".into()),
    }
}

fn zero_scalar(b: &mut FunctionBuilder, cty: &Cty) -> Value {
    match cty {
        Cty::F32 => b.ins().f32const(0.0),
        Cty::F64 => b.ins().f64const(0.0),
        _ => b.ins().iconst(types::I64, 0),
    }
}

/// AST type -> codegen type (scalars; named types become struct/agg descriptors).
fn ty_to_cty(kind: &TypeKind) -> Cty {
    match kind {
        TypeKind::Path(p) => match p.segments.last().map(|s| s.ident.name.as_str()).unwrap_or("") {
            "f32" => Cty::F32,
            "f64" => Cty::F64,
            "i8" | "i16" | "i32" | "i64" | "u8" | "u16" | "u32" | "u64" | "bool" => Cty::I64,
            "str" | "String" => Cty::Str,
            // A struct/enum type: use the FULL module-qualified path (joined with `::`) so it
            // matches the flattened, mangled layout key (e.g. `world::World`). Primitives are
            // always unqualified, so they're matched on the last segment above.
            _ => {
                let joined = if p.segments.len() > 1 {
                    p.segments.iter().map(|s| s.ident.name.as_str()).collect::<Vec<_>>().join("::")
                } else {
                    p.segments.last().map(|s| s.ident.name.clone()).unwrap_or_default()
                };
                Cty::Struct(joined)
            }
        },
        TypeKind::Dyn(p) => {
            Cty::Dyn(p.segments.last().map(|s| s.ident.name.clone()).unwrap_or_default())
        }
        TypeKind::Fn { params, ret } => {
            Cty::Fn(params.iter().map(|t| ty_to_cty(&t.kind)).collect(), Box::new(ty_to_cty(&ret.kind)))
        }
        TypeKind::Tuple(ts) => Cty::Tuple(ts.iter().map(|t| ty_to_cty(&t.kind)).collect()),
        TypeKind::Array { elem, len } => {
            let n = match len.as_ref().map(|e| &e.kind) {
                Some(ExprKind::Int(v, _)) => *v as usize,
                _ => 0,
            };
            Cty::Array(Box::new(ty_to_cty(&elem.kind)), n)
        }
        // A region annotation is checking-only; the representation is the inner type.
        TypeKind::Region(_, inner) => ty_to_cty(&inner.kind),
        _ => Cty::I64,
    }
}

/// Codegen ABI for a top-level function: parameter types + return type. A unit
/// return maps to i64 (returns 0).
/// Reclassify `Cty::Struct(n)` as `Cty::Enum(n)` when `n` names an enum.
/// `ty_to_cty` can't tell structs from enums (no env), so enum types in function
/// signatures / fields would otherwise be mis-sized (e.g. for sret returns).
fn fix_enums(c: Cty, enums: &HashSet<String>) -> Cty {
    match c {
        Cty::Struct(n) if enums.contains(&n) => Cty::Enum(n),
        Cty::Tuple(ts) => Cty::Tuple(ts.into_iter().map(|t| fix_enums(t, enums)).collect()),
        Cty::Array(e, n) => Cty::Array(Box::new(fix_enums(*e, enums)), n),
        other => other,
    }
}

/// Parameter types and return type of a "scalar" host builtin - one whose
/// arguments pass through with a simple per-type coercion, so the call site
/// needs no bespoke lowering at all. `None` return means void.
///
/// Read straight off the `aurora-abi` table, which is also what declares the
/// import the call will target, so the coercion and the callee's signature
/// cannot disagree.
fn scalar_builtin_sig(name: &str) -> Option<(&'static [aurora_abi::Ty], Option<aurora_abi::Ty>)> {
    aurora_abi::scalar_sig(name)
}

/// The compiled type an ABI type lowers to. `Ptr`/`Str` never reach here - the
/// text dispatch handles both before coercing anything - but a string value IS
/// a pointer to its two slots, and an address is an `i64` to the backend.
fn abi_cty(t: aurora_abi::Ty) -> Cty {
    match t {
        aurora_abi::Ty::F64 => Cty::F64,
        aurora_abi::Ty::Str => Cty::Str,
        aurora_abi::Ty::I64 | aurora_abi::Ty::Ptr => Cty::I64,
    }
}

/// Whether a struct/array/tuple type has a C-compatible memory layout for FFI:
/// every leaf field/element is an 8-byte `i64`/`f64` (Aurora stores each in an
/// 8-byte slot, so such aggregates match C's layout). `f32`/strings/enums and
/// other non-8-byte leaves would need packing, so they're excluded.
fn ffi_layout_ok(c: &Cty, structs: &HashMap<String, Vec<(String, Cty)>>) -> bool {
    match c {
        // `f32` leaves are allowed too: the aggregate is marshaled to C's packed
        // layout at the call site (see `marshal_to_c`).
        Cty::I64 | Cty::F64 | Cty::F32 => true,
        Cty::Struct(n) => {
            structs.get(n).map(|fs| fs.iter().all(|(_, t)| ffi_layout_ok(t, structs))).unwrap_or(false)
        }
        Cty::Array(e, _) => ffi_layout_ok(e, structs),
        Cty::Tuple(ts) => ts.iter().all(|t| ffi_layout_ok(t, structs)),
        _ => false,
    }
}

/// Whether an FFI aggregate argument must be repacked to C layout before the
/// call — true when it contains an `f32` leaf (Aurora stores `f32` in 8-byte
/// slots; C packs it to 4). Pure 8-byte-leaf aggregates already match C.
fn ffi_needs_marshal(c: &Cty, structs: &HashMap<String, Vec<(String, Cty)>>) -> bool {
    match c {
        Cty::F32 => true,
        Cty::Struct(n) => {
            structs.get(n).map(|fs| fs.iter().any(|(_, t)| ffi_needs_marshal(t, structs))).unwrap_or(false)
        }
        Cty::Array(e, _) => ffi_needs_marshal(e, structs),
        Cty::Tuple(ts) => ts.iter().any(|t| ffi_needs_marshal(t, structs)),
        _ => false,
    }
}

fn align_up(x: u32, a: u32) -> u32 {
    (x + a - 1) / a * a
}

/// Flatten an aggregate's scalar leaves with their Aurora byte offset (each leaf
/// in an 8-byte slot) and C byte offset (packed, naturally aligned). Tracks the
/// running C offset and the aggregate's C alignment.
fn flatten_ffi(
    cty: &Cty,
    structs: &HashMap<String, Vec<(String, Cty)>>,
    aurora_off: &mut u32,
    c_off: &mut u32,
    c_align: &mut u32,
    out: &mut Vec<(u32, u32, Cty)>,
) {
    match cty {
        Cty::I64 | Cty::F64 => {
            *c_off = align_up(*c_off, 8);
            out.push((*aurora_off, *c_off, cty.clone()));
            *c_off += 8;
            *aurora_off += 8;
            *c_align = (*c_align).max(8);
        }
        Cty::F32 => {
            *c_off = align_up(*c_off, 4);
            out.push((*aurora_off, *c_off, Cty::F32));
            *c_off += 4;
            *aurora_off += 8;
            *c_align = (*c_align).max(4);
        }
        Cty::Struct(n) => {
            if let Some(fields) = structs.get(n) {
                for (_, ft) in fields {
                    flatten_ffi(ft, structs, aurora_off, c_off, c_align, out);
                }
            }
        }
        Cty::Array(elem, n) => {
            for _ in 0..*n {
                flatten_ffi(elem, structs, aurora_off, c_off, c_align, out);
            }
        }
        Cty::Tuple(ts) => {
            for t in ts {
                flatten_ffi(t, structs, aurora_off, c_off, c_align, out);
            }
        }
        _ => {}
    }
}

/// Copy an Aurora aggregate (8-byte-slot layout) at `aurora_ptr` into a freshly
/// allocated, C-packed buffer and return a pointer to it — so an `@extern`
/// function reads it with C's layout (e.g. a `[f32; 16]` matrix as `float[16]`).
fn marshal_to_c(b: &mut FunctionBuilder, env: &Env, aurora_ptr: Value, cty: &Cty) -> Value {
    let mut leaves = Vec::new();
    let (mut a_off, mut c_off, mut c_align) = (0u32, 0u32, 1u32);
    flatten_ffi(cty, &env.structs, &mut a_off, &mut c_off, &mut c_align, &mut leaves);
    let size = align_up(c_off.max(1), c_align);
    let slot = b.create_sized_stack_slot(StackSlotData::new(StackSlotKind::ExplicitSlot, size, 3));
    let buf = b.ins().stack_addr(env.ptr_ty, slot, 0);
    for (ao, co, lt) in leaves {
        let v = b.ins().load(lt.clif(env.ptr_ty), MemFlags::new(), aurora_ptr, ao as i32);
        b.ins().store(MemFlags::new(), v, buf, co as i32);
    }
    buf
}

/// Does `attrs` contain `@name`?
fn has_attr(attrs: &[aurora_ast::Attr], name: &str) -> bool {
    attrs.iter().any(|a| a.name.name == name)
}

/// The external C symbol an `@extern` function binds to: the string in
/// `@extern("symbol")` if given, else the function's own name.
fn extern_symbol(attrs: &[aurora_ast::Attr], fn_name: &str) -> String {
    for a in attrs {
        if a.name.name == "extern" {
            if let Some(aurora_ast::AttrArg::Positional(e)) = a.args.first() {
                if let ExprKind::Str(s) = &e.kind {
                    return s.clone();
                }
            }
        }
    }
    fn_name.to_string()
}

fn fn_abi(f: &aurora_ast::FnDecl) -> (Vec<Cty>, Cty) {
    let params = f
        .params
        .iter()
        .filter_map(|p| match p {
            aurora_ast::Param::Normal { ty, .. } => Some(ty_to_cty(&ty.kind)),
            aurora_ast::Param::SelfParam { .. } => None,
        })
        .collect();
    let ret = match &f.ret {
        Some(t) => ty_to_cty(&t.kind),
        None => Cty::I64,
    };
    (params, ret)
}

/// Names a closure body references that it must capture from the enclosing
/// scope: free single-name references minus params, minus names bound inside the
/// body, minus the `exclude` set (top-level fns + builtins).
fn closure_captures(body: &Expr, params: &[String], exclude: &HashSet<String>) -> Vec<String> {
    let mut refs = Vec::new();
    let mut bound = HashSet::new();
    refs_and_binds(body, &mut refs, &mut bound);
    let pset: HashSet<&str> = params.iter().map(|s| s.as_str()).collect();
    let mut caps = Vec::new();
    for r in refs {
        if !pset.contains(r.as_str())
            && !exclude.contains(&r)
            && !bound.contains(&r)
            && !caps.contains(&r)
        {
            caps.push(r);
        }
    }
    caps
}

fn refs_and_binds_block(block: &Block, refs: &mut Vec<String>, bound: &mut HashSet<String>) {
    for stmt in &block.stmts {
        match stmt {
            Stmt::Let(le) => {
                if let Some(e) = &le.init {
                    refs_and_binds(e, refs, bound);
                }
                for n in pattern_names(&le.pat).into_iter().flatten() {
                    bound.insert(n);
                }
            }
            Stmt::Defer(e) | Stmt::Expr(e) => refs_and_binds(e, refs, bound),
        }
    }
    if let Some(t) = &block.tail {
        refs_and_binds(t, refs, bound);
    }
}

fn refs_and_binds(e: &Expr, refs: &mut Vec<String>, bound: &mut HashSet<String>) {
    match &e.kind {
        ExprKind::Path(p) if p.is_single() => refs.push(p.segments[0].ident.name.clone()),
        ExprKind::Paren(x) | ExprKind::Unary(_, x) | ExprKind::Cast(x, _) | ExprKind::Try(x) => {
            refs_and_binds(x, refs, bound)
        }
        ExprKind::Binary(_, a, c) | ExprKind::Assign(_, a, c) => {
            refs_and_binds(a, refs, bound);
            refs_and_binds(c, refs, bound);
        }
        ExprKind::Pipe { value, func } => {
            refs_and_binds(value, refs, bound);
            refs_and_binds(func, refs, bound);
        }
        ExprKind::Call { callee, args, .. } => {
            refs_and_binds(callee, refs, bound);
            for a in args {
                refs_and_binds(&a.value, refs, bound);
            }
        }
        ExprKind::Index { base, index } => {
            refs_and_binds(base, refs, bound);
            refs_and_binds(index, refs, bound);
        }
        ExprKind::Field { base, .. } => refs_and_binds(base, refs, bound),
        ExprKind::Struct { fields, base, .. } => {
            for f in fields {
                if let Some(v) = &f.value {
                    refs_and_binds(v, refs, bound);
                }
            }
            if let Some(bse) = base {
                refs_and_binds(bse, refs, bound);
            }
        }
        ExprKind::Tuple(xs) | ExprKind::Array(xs) => {
            for x in xs {
                refs_and_binds(x, refs, bound);
            }
        }
        ExprKind::If(ifx) => {
            refs_and_binds(&ifx.cond, refs, bound);
            refs_and_binds_block(&ifx.then_branch, refs, bound);
            if let Some(el) = &ifx.else_branch {
                refs_and_binds(el, refs, bound);
            }
        }
        ExprKind::Match { scrutinee, arms } => {
            refs_and_binds(scrutinee, refs, bound);
            for arm in arms {
                refs_and_binds(&arm.body, refs, bound);
            }
        }
        ExprKind::For { pat, iter, body } => {
            refs_and_binds(iter, refs, bound);
            for n in pattern_names(pat).into_iter().flatten() {
                bound.insert(n);
            }
            refs_and_binds_block(body, refs, bound);
        }
        ExprKind::While { cond, body } => {
            refs_and_binds(cond, refs, bound);
            refs_and_binds_block(body, refs, bound);
        }
        ExprKind::Loop(b) | ExprKind::Block(b) | ExprKind::Unsafe(b) => {
            refs_and_binds_block(b, refs, bound)
        }
        ExprKind::Return(Some(x)) | ExprKind::Break(Some(x)) => refs_and_binds(x, refs, bound),
        _ => {}
    }
}

/// Collect every closure expression reachable from `block` (for lambda lifting).
fn collect_closures<'a>(block: &'a Block, out: &mut Vec<&'a Expr>) {
    for stmt in &block.stmts {
        match stmt {
            Stmt::Let(le) => {
                if let Some(e) = &le.init {
                    cc_expr(e, out);
                }
            }
            Stmt::Defer(e) | Stmt::Expr(e) => cc_expr(e, out),
        }
    }
    if let Some(t) = &block.tail {
        cc_expr(t, out);
    }
}

fn cc_expr<'a>(e: &'a Expr, out: &mut Vec<&'a Expr>) {
    match &e.kind {
        ExprKind::Closure { body, .. } => {
            out.push(e);
            cc_expr(body, out);
        }
        ExprKind::Paren(x) | ExprKind::Unary(_, x) | ExprKind::Cast(x, _) | ExprKind::Try(x) => {
            cc_expr(x, out)
        }
        ExprKind::Binary(_, a, c) | ExprKind::Assign(_, a, c) => {
            cc_expr(a, out);
            cc_expr(c, out);
        }
        ExprKind::Pipe { value, func } => {
            cc_expr(value, out);
            cc_expr(func, out);
        }
        ExprKind::Call { callee, args, .. } => {
            cc_expr(callee, out);
            for a in args {
                cc_expr(&a.value, out);
            }
        }
        ExprKind::Index { base, index } => {
            cc_expr(base, out);
            cc_expr(index, out);
        }
        ExprKind::Field { base, .. } => cc_expr(base, out),
        ExprKind::Struct { fields, base, .. } => {
            for f in fields {
                if let Some(v) = &f.value {
                    cc_expr(v, out);
                }
            }
            if let Some(b) = base {
                cc_expr(b, out);
            }
        }
        ExprKind::Tuple(xs) | ExprKind::Array(xs) => {
            for x in xs {
                cc_expr(x, out);
            }
        }
        ExprKind::If(ifx) => {
            cc_expr(&ifx.cond, out);
            collect_closures(&ifx.then_branch, out);
            if let Some(el) = &ifx.else_branch {
                cc_expr(el, out);
            }
        }
        ExprKind::Match { scrutinee, arms } => {
            cc_expr(scrutinee, out);
            for arm in arms {
                cc_expr(&arm.body, out);
            }
        }
        ExprKind::For { iter, body, .. } => {
            cc_expr(iter, out);
            collect_closures(body, out);
        }
        ExprKind::While { cond, body } => {
            cc_expr(cond, out);
            collect_closures(body, out);
        }
        ExprKind::Loop(b) | ExprKind::Block(b) | ExprKind::Unsafe(b) => collect_closures(b, out),
        ExprKind::Return(Some(x)) | ExprKind::Break(Some(x)) => cc_expr(x, out),
        _ => {}
    }
}

fn binding_name(pat: &aurora_ast::Pat) -> Option<String> {
    match &pat.kind {
        aurora_ast::PatKind::Binding { name, .. } => Some(name.name.clone()),
        _ => None,
    }
}

#[cfg(test)]
mod abi_tests;
#[cfg(test)]
mod tests;
