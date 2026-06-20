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
use aurora_runtime::{
    aurora_clear, aurora_despawn, aurora_entity_count, aurora_fb_get, aurora_framebuffer,
    aurora_get_component, aurora_pixel, aurora_print_f64, aurora_print_i64, aurora_print_nl,
    aurora_print_str, aurora_query_begin, aurora_query_entity, aurora_save_ppm, aurora_spawn_entity,
    aurora_store_component, aurora_triangle,
};

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

/// Builtin function names (handled specially, never user-defined / captured).
const BUILTINS: &[&str] = &[
    "print", "println", "assert", "sqrt", "sin", "cos", "tan", "floor", "ceil", "round", "pow",
    "log", "exp", "atan2",
    "abs", "min", "max", "clamp", "len", "str", "spawn", "despawn", "run_systems", "entity_count",
    "band", "bor", "bxor", "shl", "shr", "bnot",
    "framebuffer", "clear", "pixel", "triangle", "fb_get", "save_ppm",
    "play_note", "play_sound", "play_noise", "audio_volume", "audio_stop", "window_fullscreen", "window_open", "window_present",
    "surface_w", "surface_h",
    "key_down", "input_char", "mouse_x", "mouse_y", "mouse_down", "gpu_render",
    "load_ppm", "load_image", "load_font", "draw_text", "draw_int", "text_width", "play_wav", "scene_save", "scene_load", "frame_reset",
    "phys_init", "phys_add", "phys_step", "phys_x", "phys_y", "phys_set_vel",
    "phys_vel_x", "phys_vel_y", "phys_apply_impulse", "phys_apply_force", "phys_set_pos", "phys_raycast",
    "nav_init", "nav_wall", "nav_find", "nav_x", "nav_y",
    "char_at", "substr", "starts_with", "net_bind", "net_connect", "net_send", "net_recv",
    "gpu_compute", "par_for",
    // 3D physics (Rapier 3D).
    "phys3d_init", "phys3d_add_box", "phys3d_add_box_rot", "phys3d_add_sphere", "phys3d_add_capsule",
    "phys3d_add_character", "phys3d_add_trimesh", "phys3d_step",
    "phys3d_x", "phys3d_y", "phys3d_z", "phys3d_vel_x", "phys3d_vel_y", "phys3d_vel_z",
    "phys3d_set_vel", "phys3d_set_pos", "phys3d_apply_impulse", "phys3d_move_character",
    "phys3d_grounded", "phys3d_raycast",
    // 3D pathfinding (voxel grid + navmesh).
    "nav3d_init", "nav3d_wall", "nav3d_find", "nav3d_x", "nav3d_y", "nav3d_z",
    "navmesh_build", "navmesh_find", "navmesh_x", "navmesh_y", "navmesh_z",
    // 3D rendering.
    "r3d_load_model", "r3d_make_box", "r3d_make_box_sized", "r3d_make_box_emissive", "r3d_make_sphere", "r3d_make_plane",
    "r3d_camera", "r3d_camera_roll", "r3d_light", "r3d_clear", "r3d_begin", "r3d_draw", "r3d_draw_tint",
    "r3d_draw_on_joint", "r3d_joint_dump", "r3d_draw_shield",
    "r3d_anim_play", "r3d_anim_update", "r3d_anim_play_upper", "r3d_anim_stop_upper", "r3d_clip_count", "r3d_present",
    "r3d_fog", "r3d_speedlines", "r3d_damage", "r3d_blur", "r3d_sky", "r3d_shadows", "r3d_ssao", "r3d_point_shadows", "r3d_clear_lights", "r3d_point_light",
    "r3d_make_sprite", "r3d_draw_billboard", "r3d_debug_line", "r3d_frustum_cull",
    "r3d_screen_x", "r3d_screen_y",
    // FPS input.
    "mouse_dx", "mouse_dy", "mouse_scroll", "mouse_button", "grab_mouse", "frame_dt", "sleep_ms",
    // 3D positional audio.
    "audio_listener", "play_sound_at",
    // Rich 3D physics queries.
    "phys3d_raycast_full", "phys3d_raycast_ex", "phys3d_hit_x", "phys3d_hit_y", "phys3d_hit_z",
    "phys3d_hit_nx", "phys3d_hit_ny", "phys3d_hit_nz", "phys3d_hit_body",
    "phys3d_spherecast", "phys3d_overlap_sphere", "phys3d_apply_force",
    "phys3d_apply_torque", "phys3d_set_angvel", "phys3d_set_rot",
    "phys3d_rot_qx", "phys3d_rot_qy", "phys3d_rot_qz", "phys3d_rot_qw",
    // Multiplayer (generic framework: the game registers its Aurora sim).
    "net_host", "net_join", "net_sim", "net_send_input", "net_update",
    "net_my_id", "net_is_server", "net_player_count", "net_player_id_at",
    "net_player_x", "net_player_y", "net_player_z", "net_player_yaw", "net_player_state",
    "net_set_meta", "net_player_meta", "net_set_name", "net_player_name_len", "net_player_name_char",
    "net_local_x", "net_local_y", "net_local_z", "net_local_yaw",
    "net_state", "net_local_state", "net_interest", "net_hit_radius", "net_max_clients", "net_rejected",
    "net_set_bot_count", "net_set_bot", "net_set_bot_meta", "net_set_bot_name", "net_bot_count",
    "net_set_object_count", "net_set_object", "net_object_count", "net_object_x", "net_object_y", "net_object_z",
    "net_spawn_at", "net_fire",
    "net_hit_player", "net_hit_x", "net_hit_y", "net_hit_z",
    "net_server_hit_count", "net_server_hit_shooter", "net_server_hit_victim", "net_server_hit_weapon",
    "net_server_hit_x", "net_server_hit_y", "net_server_hit_z", "net_server_hits_clear",
    "net_push_kill", "net_kill_count", "net_kill_killer", "net_kill_victim", "net_kills_clear",
    // Rebindable input-action layer + raw f32-blob accessors.
    "input_bind", "input_binding", "input_down", "input_axis", "input_suppress",
    "save_settings", "load_settings",
    "f32_load", "f32_store",
];

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
    fn enum_variant(&self, path: &aurora_ast::Path) -> Option<(String, usize)> {
        if path.segments.len() != 2 {
            return None;
        }
        let enm = &path.segments[0].ident.name;
        let var = &path.segments[1].ident.name;
        let layout = self.enums.get(enm)?;
        let idx = layout.variants.iter().position(|v| &v.name == var)?;
        Some((enm.clone(), idx))
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

/// Register the addresses of every `aurora_*` host function with the JIT.
fn register_host_symbols(builder: &mut JITBuilder) {
    builder.symbol("aurora_print_i64", aurora_print_i64 as *const u8);
    builder.symbol("aurora_print_f64", aurora_print_f64 as *const u8);
    builder.symbol("aurora_print_str", aurora_print_str as *const u8);
    builder.symbol("aurora_print_nl", aurora_print_nl as *const u8);
    builder.symbol("aurora_framebuffer", aurora_framebuffer as *const u8);
    builder.symbol("aurora_clear", aurora_clear as *const u8);
    builder.symbol("aurora_pixel", aurora_pixel as *const u8);
    builder.symbol("aurora_triangle", aurora_triangle as *const u8);
    builder.symbol("aurora_fb_get", aurora_fb_get as *const u8);
    builder.symbol("aurora_save_ppm", aurora_save_ppm as *const u8);
    builder.symbol("aurora_spawn_entity", aurora_spawn_entity as *const u8);
    builder.symbol("aurora_despawn", aurora_despawn as *const u8);
    builder.symbol("aurora_store_component", aurora_store_component as *const u8);
    builder.symbol("aurora_get_component", aurora_get_component as *const u8);
    builder.symbol("aurora_query_begin", aurora_query_begin as *const u8);
    builder.symbol("aurora_query_entity", aurora_query_entity as *const u8);
    builder.symbol("aurora_entity_count", aurora_entity_count as *const u8);
    builder.symbol("aurora_par_for", aurora_runtime::aurora_par_for as *const u8);
    builder.symbol("aurora_run_parallel", aurora_runtime::aurora_run_parallel as *const u8);
    builder.symbol("aurora_gpu_compute", aurora_runtime::aurora_gpu_compute as *const u8);
    builder.symbol("aurora_net_bind", aurora_runtime::aurora_net_bind as *const u8);
    builder.symbol("aurora_net_connect", aurora_runtime::aurora_net_connect as *const u8);
    builder.symbol("aurora_net_send", aurora_runtime::aurora_net_send as *const u8);
    builder.symbol("aurora_net_recv", aurora_runtime::aurora_net_recv as *const u8);
    builder.symbol("aurora_frame_reset", aurora_runtime::aurora_frame_reset as *const u8);
    builder.symbol("aurora_load_ppm", aurora_runtime::aurora_load_ppm as *const u8);
    builder.symbol("aurora_oob", aurora_runtime::aurora_oob as *const u8);
    builder.symbol("aurora_frame_dt", aurora_runtime::aurora_frame_dt as *const u8);
    builder.symbol("aurora_sleep_ms", aurora_runtime::aurora_sleep_ms as *const u8);
    builder.symbol("aurora_divzero", aurora_runtime::aurora_divzero as *const u8);
    builder.symbol("aurora_fmod", aurora_runtime::aurora_fmod as *const u8);
    builder.symbol("aurora_load_image", aurora_runtime::aurora_load_image as *const u8);
    builder.symbol("aurora_load_font", aurora_runtime::aurora_load_font as *const u8);
    builder.symbol("aurora_draw_text", aurora_runtime::aurora_draw_text as *const u8);
    builder.symbol("aurora_draw_int", aurora_runtime::aurora_draw_int as *const u8);
    builder.symbol("aurora_text_width", aurora_runtime::aurora_text_width as *const u8);
    builder.symbol("aurora_play_wav", aurora_runtime::aurora_play_wav as *const u8);
    builder.symbol("aurora_phys_init", aurora_runtime::aurora_phys_init as *const u8);
    builder.symbol("aurora_phys_add", aurora_runtime::aurora_phys_add as *const u8);
    builder.symbol("aurora_phys_step", aurora_runtime::aurora_phys_step as *const u8);
    builder.symbol("aurora_phys_x", aurora_runtime::aurora_phys_x as *const u8);
    builder.symbol("aurora_phys_y", aurora_runtime::aurora_phys_y as *const u8);
    builder.symbol("aurora_phys_set_vel", aurora_runtime::aurora_phys_set_vel as *const u8);
    builder.symbol("aurora_phys_vel_x", aurora_runtime::aurora_phys_vel_x as *const u8);
    builder.symbol("aurora_phys_vel_y", aurora_runtime::aurora_phys_vel_y as *const u8);
    builder.symbol("aurora_phys_apply_impulse", aurora_runtime::aurora_phys_apply_impulse as *const u8);
    builder.symbol("aurora_phys_apply_force", aurora_runtime::aurora_phys_apply_force as *const u8);
    builder.symbol("aurora_phys_set_pos", aurora_runtime::aurora_phys_set_pos as *const u8);
    builder.symbol("aurora_phys_raycast", aurora_runtime::aurora_phys_raycast as *const u8);
    builder.symbol("aurora_nav_init", aurora_runtime::aurora_nav_init as *const u8);
    builder.symbol("aurora_nav_wall", aurora_runtime::aurora_nav_wall as *const u8);
    builder.symbol("aurora_nav_find", aurora_runtime::aurora_nav_find as *const u8);
    builder.symbol("aurora_nav_x", aurora_runtime::aurora_nav_x as *const u8);
    builder.symbol("aurora_nav_y", aurora_runtime::aurora_nav_y as *const u8);
    // 3D physics (Rapier 3D).
    builder.symbol("aurora_phys3d_init", aurora_runtime::aurora_phys3d_init as *const u8);
    builder.symbol("aurora_phys3d_add_box", aurora_runtime::aurora_phys3d_add_box as *const u8);
    builder.symbol("aurora_phys3d_add_box_rot", aurora_runtime::aurora_phys3d_add_box_rot as *const u8);
    builder.symbol("aurora_phys3d_add_sphere", aurora_runtime::aurora_phys3d_add_sphere as *const u8);
    builder.symbol("aurora_phys3d_add_capsule", aurora_runtime::aurora_phys3d_add_capsule as *const u8);
    builder.symbol("aurora_phys3d_add_character", aurora_runtime::aurora_phys3d_add_character as *const u8);
    builder.symbol("aurora_phys3d_add_trimesh", aurora_runtime::aurora_phys3d_add_trimesh as *const u8);
    builder.symbol("aurora_phys3d_step", aurora_runtime::aurora_phys3d_step as *const u8);
    builder.symbol("aurora_phys3d_x", aurora_runtime::aurora_phys3d_x as *const u8);
    builder.symbol("aurora_phys3d_y", aurora_runtime::aurora_phys3d_y as *const u8);
    builder.symbol("aurora_phys3d_z", aurora_runtime::aurora_phys3d_z as *const u8);
    builder.symbol("aurora_phys3d_vel_x", aurora_runtime::aurora_phys3d_vel_x as *const u8);
    builder.symbol("aurora_phys3d_vel_y", aurora_runtime::aurora_phys3d_vel_y as *const u8);
    builder.symbol("aurora_phys3d_vel_z", aurora_runtime::aurora_phys3d_vel_z as *const u8);
    builder.symbol("aurora_phys3d_set_vel", aurora_runtime::aurora_phys3d_set_vel as *const u8);
    builder.symbol("aurora_phys3d_set_pos", aurora_runtime::aurora_phys3d_set_pos as *const u8);
    builder.symbol("aurora_phys3d_apply_impulse", aurora_runtime::aurora_phys3d_apply_impulse as *const u8);
    builder.symbol("aurora_phys3d_move_character", aurora_runtime::aurora_phys3d_move_character as *const u8);
    builder.symbol("aurora_phys3d_grounded", aurora_runtime::aurora_phys3d_grounded as *const u8);
    builder.symbol("aurora_phys3d_raycast", aurora_runtime::aurora_phys3d_raycast as *const u8);
    // 3D pathfinding.
    builder.symbol("aurora_nav3d_init", aurora_runtime::aurora_nav3d_init as *const u8);
    builder.symbol("aurora_nav3d_wall", aurora_runtime::aurora_nav3d_wall as *const u8);
    builder.symbol("aurora_nav3d_find", aurora_runtime::aurora_nav3d_find as *const u8);
    builder.symbol("aurora_nav3d_x", aurora_runtime::aurora_nav3d_x as *const u8);
    builder.symbol("aurora_nav3d_y", aurora_runtime::aurora_nav3d_y as *const u8);
    builder.symbol("aurora_nav3d_z", aurora_runtime::aurora_nav3d_z as *const u8);
    builder.symbol("aurora_navmesh_build", aurora_runtime::aurora_navmesh_build as *const u8);
    builder.symbol("aurora_navmesh_find", aurora_runtime::aurora_navmesh_find as *const u8);
    builder.symbol("aurora_navmesh_x", aurora_runtime::aurora_navmesh_x as *const u8);
    builder.symbol("aurora_navmesh_y", aurora_runtime::aurora_navmesh_y as *const u8);
    builder.symbol("aurora_navmesh_z", aurora_runtime::aurora_navmesh_z as *const u8);
    // 3D rendering.
    builder.symbol("aurora_r3d_load_model", aurora_runtime::aurora_r3d_load_model as *const u8);
    builder.symbol("aurora_r3d_make_box", aurora_runtime::aurora_r3d_make_box as *const u8);
    builder.symbol("aurora_r3d_make_box_sized", aurora_runtime::aurora_r3d_make_box_sized as *const u8);
    builder.symbol("aurora_r3d_make_box_emissive", aurora_runtime::aurora_r3d_make_box_emissive as *const u8);
    builder.symbol("aurora_r3d_make_sphere", aurora_runtime::aurora_r3d_make_sphere as *const u8);
    builder.symbol("aurora_r3d_make_plane", aurora_runtime::aurora_r3d_make_plane as *const u8);
    builder.symbol("aurora_r3d_camera", aurora_runtime::aurora_r3d_camera as *const u8);
    builder.symbol("aurora_r3d_camera_roll", aurora_runtime::aurora_r3d_camera_roll as *const u8);
    builder.symbol("aurora_r3d_light", aurora_runtime::aurora_r3d_light as *const u8);
    builder.symbol("aurora_r3d_clear", aurora_runtime::aurora_r3d_clear as *const u8);
    builder.symbol("aurora_r3d_begin", aurora_runtime::aurora_r3d_begin as *const u8);
    builder.symbol("aurora_r3d_draw", aurora_runtime::aurora_r3d_draw as *const u8);
    builder.symbol("aurora_r3d_draw_tint", aurora_runtime::aurora_r3d_draw_tint as *const u8);
    builder.symbol("aurora_r3d_draw_shield", aurora_runtime::aurora_r3d_draw_shield as *const u8);
    builder.symbol("aurora_r3d_draw_on_joint", aurora_runtime::aurora_r3d_draw_on_joint as *const u8);
    builder.symbol("aurora_r3d_joint_dump", aurora_runtime::aurora_r3d_joint_dump as *const u8);
    builder.symbol("aurora_r3d_anim_play", aurora_runtime::aurora_r3d_anim_play as *const u8);
    builder.symbol("aurora_r3d_anim_update", aurora_runtime::aurora_r3d_anim_update as *const u8);
    builder.symbol("aurora_r3d_anim_play_upper", aurora_runtime::aurora_r3d_anim_play_upper as *const u8);
    builder.symbol("aurora_r3d_anim_stop_upper", aurora_runtime::aurora_r3d_anim_stop_upper as *const u8);
    builder.symbol("aurora_r3d_clip_count", aurora_runtime::aurora_r3d_clip_count as *const u8);
    builder.symbol("aurora_r3d_present", aurora_runtime::aurora_r3d_present as *const u8);
    builder.symbol("aurora_r3d_fog", aurora_runtime::aurora_r3d_fog as *const u8);
    builder.symbol("aurora_r3d_speedlines", aurora_runtime::aurora_r3d_speedlines as *const u8);
    builder.symbol("aurora_r3d_damage", aurora_runtime::aurora_r3d_damage as *const u8);
    builder.symbol("aurora_r3d_blur", aurora_runtime::aurora_r3d_blur as *const u8);
    builder.symbol("aurora_r3d_sky", aurora_runtime::aurora_r3d_sky as *const u8);
    builder.symbol("aurora_r3d_shadows", aurora_runtime::aurora_r3d_shadows as *const u8);
    builder.symbol("aurora_r3d_ssao", aurora_runtime::aurora_r3d_ssao as *const u8);
    builder.symbol("aurora_r3d_point_shadows", aurora_runtime::aurora_r3d_point_shadows as *const u8);
    builder.symbol("aurora_r3d_clear_lights", aurora_runtime::aurora_r3d_clear_lights as *const u8);
    builder.symbol("aurora_r3d_point_light", aurora_runtime::aurora_r3d_point_light as *const u8);
    builder.symbol("aurora_r3d_make_sprite", aurora_runtime::aurora_r3d_make_sprite as *const u8);
    builder.symbol("aurora_r3d_draw_billboard", aurora_runtime::aurora_r3d_draw_billboard as *const u8);
    builder.symbol("aurora_r3d_debug_line", aurora_runtime::aurora_r3d_debug_line as *const u8);
    builder.symbol("aurora_r3d_frustum_cull", aurora_runtime::aurora_r3d_frustum_cull as *const u8);
    builder.symbol("aurora_r3d_screen_x", aurora_runtime::aurora_r3d_screen_x as *const u8);
    builder.symbol("aurora_r3d_screen_y", aurora_runtime::aurora_r3d_screen_y as *const u8);
    builder.symbol("aurora_mouse_dx", aurora_runtime::aurora_mouse_dx as *const u8);
    builder.symbol("aurora_mouse_dy", aurora_runtime::aurora_mouse_dy as *const u8);
    builder.symbol("aurora_mouse_scroll", aurora_runtime::aurora_mouse_scroll as *const u8);
    builder.symbol("aurora_mouse_button", aurora_runtime::aurora_mouse_button as *const u8);
    builder.symbol("aurora_grab_mouse", aurora_runtime::aurora_grab_mouse as *const u8);
    builder.symbol("aurora_audio_listener", aurora_runtime::aurora_audio_listener as *const u8);
    builder.symbol("aurora_play_sound_at", aurora_runtime::aurora_play_sound_at as *const u8);
    builder.symbol("aurora_phys3d_raycast_full", aurora_runtime::aurora_phys3d_raycast_full as *const u8);
    builder.symbol("aurora_phys3d_raycast_ex", aurora_runtime::aurora_phys3d_raycast_ex as *const u8);
    builder.symbol("aurora_phys3d_hit_x", aurora_runtime::aurora_phys3d_hit_x as *const u8);
    builder.symbol("aurora_phys3d_hit_y", aurora_runtime::aurora_phys3d_hit_y as *const u8);
    builder.symbol("aurora_phys3d_hit_z", aurora_runtime::aurora_phys3d_hit_z as *const u8);
    builder.symbol("aurora_phys3d_hit_nx", aurora_runtime::aurora_phys3d_hit_nx as *const u8);
    builder.symbol("aurora_phys3d_hit_ny", aurora_runtime::aurora_phys3d_hit_ny as *const u8);
    builder.symbol("aurora_phys3d_hit_nz", aurora_runtime::aurora_phys3d_hit_nz as *const u8);
    builder.symbol("aurora_phys3d_hit_body", aurora_runtime::aurora_phys3d_hit_body as *const u8);
    builder.symbol("aurora_phys3d_spherecast", aurora_runtime::aurora_phys3d_spherecast as *const u8);
    builder.symbol("aurora_phys3d_overlap_sphere", aurora_runtime::aurora_phys3d_overlap_sphere as *const u8);
    builder.symbol("aurora_phys3d_apply_force", aurora_runtime::aurora_phys3d_apply_force as *const u8);
    builder.symbol("aurora_phys3d_apply_torque", aurora_runtime::aurora_phys3d_apply_torque as *const u8);
    builder.symbol("aurora_phys3d_set_angvel", aurora_runtime::aurora_phys3d_set_angvel as *const u8);
    builder.symbol("aurora_phys3d_set_rot", aurora_runtime::aurora_phys3d_set_rot as *const u8);
    builder.symbol("aurora_phys3d_rot_qx", aurora_runtime::aurora_phys3d_rot_qx as *const u8);
    builder.symbol("aurora_phys3d_rot_qy", aurora_runtime::aurora_phys3d_rot_qy as *const u8);
    builder.symbol("aurora_phys3d_rot_qz", aurora_runtime::aurora_phys3d_rot_qz as *const u8);
    builder.symbol("aurora_phys3d_rot_qw", aurora_runtime::aurora_phys3d_rot_qw as *const u8);
    builder.symbol("aurora_net_host", aurora_runtime::aurora_net_host as *const u8);
    builder.symbol("aurora_net_join", aurora_runtime::aurora_net_join as *const u8);
    builder.symbol("aurora_net_sim", aurora_runtime::aurora_net_sim as *const u8);
    builder.symbol("aurora_net_send_input", aurora_runtime::aurora_net_send_input as *const u8);
    builder.symbol("aurora_save_settings", aurora_runtime::aurora_save_settings as *const u8);
    builder.symbol("aurora_load_settings", aurora_runtime::aurora_load_settings as *const u8);
    builder.symbol("aurora_net_update", aurora_runtime::aurora_net_update as *const u8);
    builder.symbol("aurora_net_my_id", aurora_runtime::aurora_net_my_id as *const u8);
    builder.symbol("aurora_net_is_server", aurora_runtime::aurora_net_is_server as *const u8);
    builder.symbol("aurora_net_player_count", aurora_runtime::aurora_net_player_count as *const u8);
    builder.symbol("aurora_net_player_id_at", aurora_runtime::aurora_net_player_id_at as *const u8);
    builder.symbol("aurora_net_player_x", aurora_runtime::aurora_net_player_x as *const u8);
    builder.symbol("aurora_net_player_y", aurora_runtime::aurora_net_player_y as *const u8);
    builder.symbol("aurora_net_player_z", aurora_runtime::aurora_net_player_z as *const u8);
    builder.symbol("aurora_net_player_yaw", aurora_runtime::aurora_net_player_yaw as *const u8);
    builder.symbol("aurora_net_player_state", aurora_runtime::aurora_net_player_state as *const u8);
    builder.symbol("aurora_net_set_meta", aurora_runtime::aurora_net_set_meta as *const u8);
    builder.symbol("aurora_net_player_meta", aurora_runtime::aurora_net_player_meta as *const u8);
    builder.symbol("aurora_net_set_name", aurora_runtime::aurora_net_set_name as *const u8);
    builder.symbol("aurora_net_player_name_len", aurora_runtime::aurora_net_player_name_len as *const u8);
    builder.symbol("aurora_net_player_name_char", aurora_runtime::aurora_net_player_name_char as *const u8);
    builder.symbol("aurora_net_local_x", aurora_runtime::aurora_net_local_x as *const u8);
    builder.symbol("aurora_net_local_y", aurora_runtime::aurora_net_local_y as *const u8);
    builder.symbol("aurora_net_local_z", aurora_runtime::aurora_net_local_z as *const u8);
    builder.symbol("aurora_net_local_yaw", aurora_runtime::aurora_net_local_yaw as *const u8);
    builder.symbol("aurora_net_state", aurora_runtime::aurora_net_state as *const u8);
    builder.symbol("aurora_net_local_state", aurora_runtime::aurora_net_local_state as *const u8);
    builder.symbol("aurora_net_interest", aurora_runtime::aurora_net_interest as *const u8);
    builder.symbol("aurora_net_max_clients", aurora_runtime::aurora_net_max_clients as *const u8);
    builder.symbol("aurora_net_rejected", aurora_runtime::aurora_net_rejected as *const u8);
    builder.symbol("aurora_net_set_bot_count", aurora_runtime::aurora_net_set_bot_count as *const u8);
    builder.symbol("aurora_net_set_bot", aurora_runtime::aurora_net_set_bot as *const u8);
    builder.symbol("aurora_net_set_bot_meta", aurora_runtime::aurora_net_set_bot_meta as *const u8);
    builder.symbol("aurora_net_set_bot_name", aurora_runtime::aurora_net_set_bot_name as *const u8);
    builder.symbol("aurora_net_bot_count", aurora_runtime::aurora_net_bot_count as *const u8);
    builder.symbol("aurora_net_set_object_count", aurora_runtime::aurora_net_set_object_count as *const u8);
    builder.symbol("aurora_net_set_object", aurora_runtime::aurora_net_set_object as *const u8);
    builder.symbol("aurora_net_object_count", aurora_runtime::aurora_net_object_count as *const u8);
    builder.symbol("aurora_net_object_x", aurora_runtime::aurora_net_object_x as *const u8);
    builder.symbol("aurora_net_object_y", aurora_runtime::aurora_net_object_y as *const u8);
    builder.symbol("aurora_net_object_z", aurora_runtime::aurora_net_object_z as *const u8);
    builder.symbol("aurora_net_hit_radius", aurora_runtime::aurora_net_hit_radius as *const u8);
    builder.symbol("aurora_net_spawn_at", aurora_runtime::aurora_net_spawn_at as *const u8);
    builder.symbol("aurora_net_fire", aurora_runtime::aurora_net_fire as *const u8);
    builder.symbol("aurora_net_server_hit_count", aurora_runtime::aurora_net_server_hit_count as *const u8);
    builder.symbol("aurora_net_server_hit_shooter", aurora_runtime::aurora_net_server_hit_shooter as *const u8);
    builder.symbol("aurora_net_server_hit_victim", aurora_runtime::aurora_net_server_hit_victim as *const u8);
    builder.symbol("aurora_net_server_hit_weapon", aurora_runtime::aurora_net_server_hit_weapon as *const u8);
    builder.symbol("aurora_net_server_hit_x", aurora_runtime::aurora_net_server_hit_x as *const u8);
    builder.symbol("aurora_net_server_hit_y", aurora_runtime::aurora_net_server_hit_y as *const u8);
    builder.symbol("aurora_net_server_hit_z", aurora_runtime::aurora_net_server_hit_z as *const u8);
    builder.symbol("aurora_net_server_hits_clear", aurora_runtime::aurora_net_server_hits_clear as *const u8);
    builder.symbol("aurora_net_push_kill", aurora_runtime::aurora_net_push_kill as *const u8);
    builder.symbol("aurora_net_kill_count", aurora_runtime::aurora_net_kill_count as *const u8);
    builder.symbol("aurora_net_kill_killer", aurora_runtime::aurora_net_kill_killer as *const u8);
    builder.symbol("aurora_net_kill_victim", aurora_runtime::aurora_net_kill_victim as *const u8);
    builder.symbol("aurora_net_kills_clear", aurora_runtime::aurora_net_kills_clear as *const u8);
    builder.symbol("aurora_net_hit_player", aurora_runtime::aurora_net_hit_player as *const u8);
    builder.symbol("aurora_net_hit_x", aurora_runtime::aurora_net_hit_x as *const u8);
    builder.symbol("aurora_net_hit_y", aurora_runtime::aurora_net_hit_y as *const u8);
    builder.symbol("aurora_net_hit_z", aurora_runtime::aurora_net_hit_z as *const u8);
    builder.symbol("aurora_input_bind", aurora_runtime::aurora_input_bind as *const u8);
    builder.symbol("aurora_input_binding", aurora_runtime::aurora_input_binding as *const u8);
    builder.symbol("aurora_input_down", aurora_runtime::aurora_input_down as *const u8);
    builder.symbol("aurora_input_suppress", aurora_runtime::aurora_input_suppress as *const u8);
    builder.symbol("aurora_input_axis", aurora_runtime::aurora_input_axis as *const u8);
    builder.symbol("aurora_f32_load", aurora_runtime::aurora_f32_load as *const u8);
    builder.symbol("aurora_f32_store", aurora_runtime::aurora_f32_store as *const u8);
    builder.symbol("aurora_sin", aurora_runtime::aurora_sin as *const u8);
    builder.symbol("aurora_cos", aurora_runtime::aurora_cos as *const u8);
    builder.symbol("aurora_tan", aurora_runtime::aurora_tan as *const u8);
    builder.symbol("aurora_pow", aurora_runtime::aurora_pow as *const u8);
    builder.symbol("aurora_log", aurora_runtime::aurora_log as *const u8);
    builder.symbol("aurora_exp", aurora_runtime::aurora_exp as *const u8);
    builder.symbol("aurora_atan2", aurora_runtime::aurora_atan2 as *const u8);
    builder.symbol("aurora_scene_save", aurora_runtime::aurora_scene_save as *const u8);
    builder.symbol("aurora_scene_load", aurora_runtime::aurora_scene_load as *const u8);
    builder.symbol("aurora_prof_enter", aurora_runtime::aurora_prof_enter as *const u8);
    builder.symbol("aurora_prof_exit", aurora_runtime::aurora_prof_exit as *const u8);
    builder.symbol("aurora_str_concat", aurora_runtime::aurora_str_concat as *const u8);
    builder.symbol("aurora_str_eq", aurora_runtime::aurora_str_eq as *const u8);
    builder.symbol("aurora_str_char_at", aurora_runtime::aurora_str_char_at as *const u8);
    builder.symbol("aurora_str_substr", aurora_runtime::aurora_str_substr as *const u8);
    builder.symbol("aurora_str_starts_with", aurora_runtime::aurora_str_starts_with as *const u8);
    builder.symbol("aurora_int_to_str", aurora_runtime::aurora_int_to_str as *const u8);
    builder.symbol("aurora_float_to_str", aurora_runtime::aurora_float_to_str as *const u8);
    builder.symbol("aurora_play_note", aurora_runtime::aurora_play_note as *const u8);
    builder.symbol("aurora_play_sound", aurora_runtime::aurora_play_sound as *const u8);
    builder.symbol("aurora_play_noise", aurora_runtime::aurora_play_noise as *const u8);
    builder.symbol("aurora_surface_w", aurora_runtime::aurora_surface_w as *const u8);
    builder.symbol("aurora_surface_h", aurora_runtime::aurora_surface_h as *const u8);
    builder.symbol("aurora_audio_volume", aurora_runtime::aurora_audio_volume as *const u8);
    builder.symbol("aurora_window_fullscreen", aurora_runtime::aurora_window_fullscreen as *const u8);
    builder.symbol("aurora_audio_stop", aurora_runtime::aurora_audio_stop as *const u8);
    builder.symbol("aurora_gpu_render", aurora_runtime::aurora_gpu_render as *const u8);
    builder.symbol("aurora_window_open", aurora_runtime::aurora_window_open as *const u8);
    builder.symbol("aurora_window_present", aurora_runtime::aurora_window_present as *const u8);
    builder.symbol("aurora_key_down", aurora_runtime::aurora_key_down as *const u8);
    builder.symbol("aurora_input_char", aurora_runtime::aurora_input_char as *const u8);
    builder.symbol("aurora_mouse_x", aurora_runtime::aurora_mouse_x as *const u8);
    builder.symbol("aurora_mouse_y", aurora_runtime::aurora_mouse_y as *const u8);
    builder.symbol("aurora_mouse_down", aurora_runtime::aurora_mouse_down as *const u8);
    builder.symbol("aurora_dbg_enter", aurora_runtime::aurora_dbg_enter as *const u8);
    builder.symbol("aurora_dbg_leave", aurora_runtime::aurora_dbg_leave as *const u8);
    builder.symbol("aurora_dbg_stmt", aurora_runtime::aurora_dbg_stmt as *const u8);
    builder.symbol("aurora_dbg_var", aurora_runtime::aurora_dbg_var as *const u8);
    builder.symbol("aurora_dbg_var_f64", aurora_runtime::aurora_dbg_var_f64 as *const u8);
    register_ffi_symbols(builder);
}

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
    // A Rust `extern "C"` function, to exercise struct/array FFI by pointer.
    builder.symbol("aurora_ffi_dot", aurora_runtime::aurora_ffi_dot as *const u8);
    builder.symbol("aurora_ffi_dotf", aurora_runtime::aurora_ffi_dotf as *const u8);
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
    // Statically linked into an executable, not a shared object.
    let _ = flags.set("is_pic", "false");
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

    let i = types::I64;
    let mut hosts = HashMap::new();
    hosts.insert("print_i64", import(jmod, "aurora_print_i64", &[i], None));
    hosts.insert("print_f64", import(jmod, "aurora_print_f64", &[types::F64], None));
    hosts.insert("print_str", import(jmod, "aurora_print_str", &[ptr_ty, i], None));
    hosts.insert("print_nl", import(jmod, "aurora_print_nl", &[], None));
    hosts.insert("framebuffer", import(jmod, "aurora_framebuffer", &[i, i], None));
    hosts.insert("clear", import(jmod, "aurora_clear", &[i, i, i], None));
    hosts.insert("pixel", import(jmod, "aurora_pixel", &[i, i, i, i, i], None));
    hosts.insert("triangle", import(jmod, "aurora_triangle", &[i, i, i, i, i, i, i, i, i], None));
    hosts.insert("fb_get", import(jmod, "aurora_fb_get", &[i, i], Some(i)));
    hosts.insert("save_ppm", import(jmod, "aurora_save_ppm", &[ptr_ty, i], None));
    hosts.insert("spawn_entity", import(jmod, "aurora_spawn_entity", &[], Some(i)));
    hosts.insert("despawn", import(jmod, "aurora_despawn", &[i], None));
    hosts.insert("store_component", import(jmod, "aurora_store_component", &[i, i, ptr_ty, i], None));
    hosts.insert("get_component", import(jmod, "aurora_get_component", &[i, i], Some(ptr_ty)));
    hosts.insert("query_begin", import(jmod, "aurora_query_begin", &[ptr_ty, i], Some(i)));
    hosts.insert("query_entity", import(jmod, "aurora_query_entity", &[i], Some(i)));
    hosts.insert("entity_count", import(jmod, "aurora_entity_count", &[], Some(i)));
    // Audio + windowing builtins.
    hosts.insert("gpu_compute", import(jmod, "aurora_gpu_compute", &[ptr_ty, i, ptr_ty, i], None));
    hosts.insert("par_for", import(jmod, "aurora_par_for", &[ptr_ty, i, ptr_ty, ptr_ty], None));
    hosts.insert("run_parallel", import(jmod, "aurora_run_parallel", &[ptr_ty, i], None));
    hosts.insert("net_bind", import(jmod, "aurora_net_bind", &[i], Some(i)));
    hosts.insert("net_connect", import(jmod, "aurora_net_connect", &[ptr_ty, i], Some(i)));
    hosts.insert("net_send", import(jmod, "aurora_net_send", &[ptr_ty, i], Some(i)));
    hosts.insert("net_recv", import(jmod, "aurora_net_recv", &[ptr_ty], None));
    hosts.insert("frame_reset", import(jmod, "aurora_frame_reset", &[], None));
    hosts.insert("load_ppm", import(jmod, "aurora_load_ppm", &[ptr_ty, i], Some(i)));
    hosts.insert("oob", import(jmod, "aurora_oob", &[i, i], None));
    hosts.insert("frame_dt", import(jmod, "aurora_frame_dt", &[], Some(types::F64)));
    hosts.insert("sleep_ms", import(jmod, "aurora_sleep_ms", &[types::I64], None));
    hosts.insert("divzero", import(jmod, "aurora_divzero", &[], None));
    hosts.insert("fmod", import(jmod, "aurora_fmod", &[types::F64, types::F64], Some(types::F64)));
    hosts.insert("load_image", import(jmod, "aurora_load_image", &[ptr_ty, i], Some(i)));
    hosts.insert("load_font", import(jmod, "aurora_load_font", &[ptr_ty, i], Some(i)));
    hosts.insert("play_wav", import(jmod, "aurora_play_wav", &[ptr_ty, i], Some(i)));
    let f64t = types::F64;
    hosts.insert("phys_init", import(jmod, "aurora_phys_init", &[f64t, f64t], None));
    hosts.insert("phys_add", import(jmod, "aurora_phys_add", &[f64t, f64t, f64t, f64t, i], Some(i)));
    hosts.insert("phys_step", import(jmod, "aurora_phys_step", &[f64t], None));
    hosts.insert("phys_x", import(jmod, "aurora_phys_x", &[i], Some(f64t)));
    hosts.insert("phys_y", import(jmod, "aurora_phys_y", &[i], Some(f64t)));
    hosts.insert("phys_set_vel", import(jmod, "aurora_phys_set_vel", &[i, f64t, f64t], None));
    hosts.insert("phys_vel_x", import(jmod, "aurora_phys_vel_x", &[i], Some(f64t)));
    hosts.insert("phys_vel_y", import(jmod, "aurora_phys_vel_y", &[i], Some(f64t)));
    hosts.insert("phys_apply_impulse", import(jmod, "aurora_phys_apply_impulse", &[i, f64t, f64t], None));
    hosts.insert("phys_apply_force", import(jmod, "aurora_phys_apply_force", &[i, f64t, f64t], None));
    hosts.insert("phys_set_pos", import(jmod, "aurora_phys_set_pos", &[i, f64t, f64t], None));
    hosts.insert("phys_raycast", import(jmod, "aurora_phys_raycast", &[f64t, f64t, f64t, f64t, f64t], Some(f64t)));
    hosts.insert("nav_init", import(jmod, "aurora_nav_init", &[i, i], None));
    hosts.insert("nav_wall", import(jmod, "aurora_nav_wall", &[i, i, i], None));
    hosts.insert("nav_find", import(jmod, "aurora_nav_find", &[i, i, i, i], Some(i)));
    hosts.insert("nav_x", import(jmod, "aurora_nav_x", &[i], Some(i)));
    hosts.insert("nav_y", import(jmod, "aurora_nav_y", &[i], Some(i)));
    // 3D physics (Rapier 3D).
    hosts.insert("phys3d_init", import(jmod, "aurora_phys3d_init", &[f64t, f64t, f64t], None));
    hosts.insert("phys3d_add_box", import(jmod, "aurora_phys3d_add_box", &[f64t, f64t, f64t, f64t, f64t, f64t, i], Some(i)));
    hosts.insert("phys3d_add_box_rot", import(jmod, "aurora_phys3d_add_box_rot", &[f64t, f64t, f64t, f64t, f64t, f64t, f64t, f64t, f64t, i], Some(i)));
    hosts.insert("phys3d_add_sphere", import(jmod, "aurora_phys3d_add_sphere", &[f64t, f64t, f64t, f64t, i], Some(i)));
    hosts.insert("phys3d_add_capsule", import(jmod, "aurora_phys3d_add_capsule", &[f64t, f64t, f64t, f64t, f64t, i], Some(i)));
    hosts.insert("phys3d_add_character", import(jmod, "aurora_phys3d_add_character", &[f64t, f64t, f64t, f64t, f64t], Some(i)));
    hosts.insert("phys3d_add_trimesh", import(jmod, "aurora_phys3d_add_trimesh", &[ptr_ty, i, ptr_ty, i], Some(i)));
    hosts.insert("phys3d_step", import(jmod, "aurora_phys3d_step", &[f64t], None));
    hosts.insert("phys3d_x", import(jmod, "aurora_phys3d_x", &[i], Some(f64t)));
    hosts.insert("phys3d_y", import(jmod, "aurora_phys3d_y", &[i], Some(f64t)));
    hosts.insert("phys3d_z", import(jmod, "aurora_phys3d_z", &[i], Some(f64t)));
    hosts.insert("phys3d_vel_x", import(jmod, "aurora_phys3d_vel_x", &[i], Some(f64t)));
    hosts.insert("phys3d_vel_y", import(jmod, "aurora_phys3d_vel_y", &[i], Some(f64t)));
    hosts.insert("phys3d_vel_z", import(jmod, "aurora_phys3d_vel_z", &[i], Some(f64t)));
    hosts.insert("phys3d_set_vel", import(jmod, "aurora_phys3d_set_vel", &[i, f64t, f64t, f64t], None));
    hosts.insert("phys3d_set_pos", import(jmod, "aurora_phys3d_set_pos", &[i, f64t, f64t, f64t], None));
    hosts.insert("phys3d_apply_impulse", import(jmod, "aurora_phys3d_apply_impulse", &[i, f64t, f64t, f64t], None));
    hosts.insert("phys3d_move_character", import(jmod, "aurora_phys3d_move_character", &[i, f64t, f64t, f64t, f64t], None));
    hosts.insert("phys3d_grounded", import(jmod, "aurora_phys3d_grounded", &[i], Some(i)));
    hosts.insert("phys3d_raycast", import(jmod, "aurora_phys3d_raycast", &[f64t, f64t, f64t, f64t, f64t, f64t, f64t], Some(f64t)));
    // 3D pathfinding.
    hosts.insert("nav3d_init", import(jmod, "aurora_nav3d_init", &[i, i, i], None));
    hosts.insert("nav3d_wall", import(jmod, "aurora_nav3d_wall", &[i, i, i, i], None));
    hosts.insert("nav3d_find", import(jmod, "aurora_nav3d_find", &[i, i, i, i, i, i], Some(i)));
    hosts.insert("nav3d_x", import(jmod, "aurora_nav3d_x", &[i], Some(i)));
    hosts.insert("nav3d_y", import(jmod, "aurora_nav3d_y", &[i], Some(i)));
    hosts.insert("nav3d_z", import(jmod, "aurora_nav3d_z", &[i], Some(i)));
    hosts.insert("navmesh_build", import(jmod, "aurora_navmesh_build", &[ptr_ty, i, ptr_ty, i], Some(i)));
    hosts.insert("navmesh_find", import(jmod, "aurora_navmesh_find", &[f64t, f64t, f64t, f64t, f64t, f64t], Some(i)));
    hosts.insert("navmesh_x", import(jmod, "aurora_navmesh_x", &[i], Some(f64t)));
    hosts.insert("navmesh_y", import(jmod, "aurora_navmesh_y", &[i], Some(f64t)));
    hosts.insert("navmesh_z", import(jmod, "aurora_navmesh_z", &[i], Some(f64t)));
    // 3D rendering.
    hosts.insert("r3d_load_model", import(jmod, "aurora_r3d_load_model", &[ptr_ty, i], Some(i)));
    hosts.insert("r3d_make_box", import(jmod, "aurora_r3d_make_box", &[f64t, f64t, f64t], Some(i)));
    hosts.insert("r3d_make_box_sized", import(jmod, "aurora_r3d_make_box_sized", &[f64t, f64t, f64t, f64t, f64t, f64t], Some(i)));
    hosts.insert("r3d_make_box_emissive", import(jmod, "aurora_r3d_make_box_emissive", &[f64t, f64t, f64t, f64t, f64t, f64t], Some(i)));
    hosts.insert("r3d_make_sphere", import(jmod, "aurora_r3d_make_sphere", &[i, f64t, f64t, f64t], Some(i)));
    hosts.insert("r3d_make_plane", import(jmod, "aurora_r3d_make_plane", &[f64t, f64t, f64t, f64t, f64t], Some(i)));
    hosts.insert("r3d_camera", import(jmod, "aurora_r3d_camera", &[f64t, f64t, f64t, f64t, f64t, f64t, f64t], None));
    hosts.insert("r3d_camera_roll", import(jmod, "aurora_r3d_camera_roll", &[f64t], None));
    hosts.insert("r3d_light", import(jmod, "aurora_r3d_light", &[f64t, f64t, f64t, f64t, f64t, f64t, f64t], None));
    hosts.insert("r3d_clear", import(jmod, "aurora_r3d_clear", &[f64t, f64t, f64t], None));
    hosts.insert("r3d_begin", import(jmod, "aurora_r3d_begin", &[], None));
    hosts.insert("r3d_draw", import(jmod, "aurora_r3d_draw", &[i, f64t, f64t, f64t, f64t, f64t, f64t, f64t], None));
    hosts.insert("r3d_draw_tint", import(jmod, "aurora_r3d_draw_tint", &[i, f64t, f64t, f64t, f64t, f64t, f64t, f64t, f64t, f64t, f64t], None));
    hosts.insert("r3d_draw_shield", import(jmod, "aurora_r3d_draw_shield", &[i, f64t, f64t, f64t, f64t, f64t, f64t, f64t, f64t, f64t], None));
    hosts.insert("r3d_draw_on_joint", import(jmod, "aurora_r3d_draw_on_joint", &[i, i, i, f64t, f64t, f64t, f64t, f64t, f64t, f64t, f64t, f64t, f64t, f64t, f64t, f64t, f64t], None));
    hosts.insert("r3d_joint_dump", import(jmod, "aurora_r3d_joint_dump", &[i], None));
    hosts.insert("r3d_anim_play", import(jmod, "aurora_r3d_anim_play", &[i, i, i, f64t, f64t], None));
    hosts.insert("r3d_anim_update", import(jmod, "aurora_r3d_anim_update", &[i, f64t], None));
    hosts.insert("r3d_anim_play_upper", import(jmod, "aurora_r3d_anim_play_upper", &[i, i, i, f64t, f64t, i], None));
    hosts.insert("r3d_anim_stop_upper", import(jmod, "aurora_r3d_anim_stop_upper", &[i, f64t], None));
    hosts.insert("r3d_clip_count", import(jmod, "aurora_r3d_clip_count", &[i], Some(i)));
    hosts.insert("r3d_present", import(jmod, "aurora_r3d_present", &[], Some(i)));
    hosts.insert("r3d_fog", import(jmod, "aurora_r3d_fog", &[f64t, f64t, f64t, f64t], None));
    hosts.insert("r3d_speedlines", import(jmod, "aurora_r3d_speedlines", &[f64t, f64t], None));
    hosts.insert("r3d_damage", import(jmod, "aurora_r3d_damage", &[f64t, f64t, f64t, f64t, f64t], None));
    hosts.insert("r3d_blur", import(jmod, "aurora_r3d_blur", &[f64t], None));
    hosts.insert("r3d_sky", import(jmod, "aurora_r3d_sky", &[i, f64t, f64t, f64t, f64t, f64t, f64t], None));
    hosts.insert("r3d_shadows", import(jmod, "aurora_r3d_shadows", &[i], None));
    hosts.insert("r3d_ssao", import(jmod, "aurora_r3d_ssao", &[i], None));
    hosts.insert("r3d_point_shadows", import(jmod, "aurora_r3d_point_shadows", &[i], None));
    hosts.insert("r3d_clear_lights", import(jmod, "aurora_r3d_clear_lights", &[], None));
    hosts.insert("r3d_point_light", import(jmod, "aurora_r3d_point_light", &[f64t, f64t, f64t, f64t, f64t, f64t, f64t, f64t], None));
    hosts.insert("r3d_make_sprite", import(jmod, "aurora_r3d_make_sprite", &[f64t, f64t, f64t], Some(i)));
    hosts.insert("r3d_draw_billboard", import(jmod, "aurora_r3d_draw_billboard", &[i, f64t, f64t, f64t, f64t], None));
    hosts.insert("r3d_debug_line", import(jmod, "aurora_r3d_debug_line", &[f64t, f64t, f64t, f64t, f64t, f64t, f64t, f64t, f64t], None));
    hosts.insert("r3d_frustum_cull", import(jmod, "aurora_r3d_frustum_cull", &[i], None));
    hosts.insert("r3d_screen_x", import(jmod, "aurora_r3d_screen_x", &[f64t, f64t, f64t], Some(f64t)));
    hosts.insert("r3d_screen_y", import(jmod, "aurora_r3d_screen_y", &[f64t, f64t, f64t], Some(f64t)));
    hosts.insert("mouse_dx", import(jmod, "aurora_mouse_dx", &[], Some(f64t)));
    hosts.insert("mouse_dy", import(jmod, "aurora_mouse_dy", &[], Some(f64t)));
    hosts.insert("mouse_scroll", import(jmod, "aurora_mouse_scroll", &[], Some(f64t)));
    hosts.insert("mouse_button", import(jmod, "aurora_mouse_button", &[i], Some(i)));
    hosts.insert("grab_mouse", import(jmod, "aurora_grab_mouse", &[i], None));
    hosts.insert("audio_listener", import(jmod, "aurora_audio_listener", &[f64t, f64t, f64t, f64t, f64t, f64t], None));
    hosts.insert("play_sound_at", import(jmod, "aurora_play_sound_at", &[i, i, i, f64t, f64t, f64t], None));
    hosts.insert("phys3d_raycast_full", import(jmod, "aurora_phys3d_raycast_full", &[f64t, f64t, f64t, f64t, f64t, f64t, f64t], Some(i)));
    hosts.insert("phys3d_raycast_ex", import(jmod, "aurora_phys3d_raycast_ex", &[i, f64t, f64t, f64t, f64t, f64t, f64t, f64t], Some(i)));
    hosts.insert("phys3d_hit_x", import(jmod, "aurora_phys3d_hit_x", &[], Some(f64t)));
    hosts.insert("phys3d_hit_y", import(jmod, "aurora_phys3d_hit_y", &[], Some(f64t)));
    hosts.insert("phys3d_hit_z", import(jmod, "aurora_phys3d_hit_z", &[], Some(f64t)));
    hosts.insert("phys3d_hit_nx", import(jmod, "aurora_phys3d_hit_nx", &[], Some(f64t)));
    hosts.insert("phys3d_hit_ny", import(jmod, "aurora_phys3d_hit_ny", &[], Some(f64t)));
    hosts.insert("phys3d_hit_nz", import(jmod, "aurora_phys3d_hit_nz", &[], Some(f64t)));
    hosts.insert("phys3d_hit_body", import(jmod, "aurora_phys3d_hit_body", &[], Some(i)));
    hosts.insert("phys3d_spherecast", import(jmod, "aurora_phys3d_spherecast", &[f64t, f64t, f64t, f64t, f64t, f64t, f64t, f64t], Some(f64t)));
    hosts.insert("phys3d_overlap_sphere", import(jmod, "aurora_phys3d_overlap_sphere", &[f64t, f64t, f64t, f64t], Some(i)));
    hosts.insert("phys3d_apply_force", import(jmod, "aurora_phys3d_apply_force", &[i, f64t, f64t, f64t], None));
    hosts.insert("phys3d_apply_torque", import(jmod, "aurora_phys3d_apply_torque", &[i, f64t, f64t, f64t], None));
    hosts.insert("phys3d_set_angvel", import(jmod, "aurora_phys3d_set_angvel", &[i, f64t, f64t, f64t], None));
    hosts.insert("phys3d_set_rot", import(jmod, "aurora_phys3d_set_rot", &[i, f64t, f64t, f64t, f64t], None));
    hosts.insert("phys3d_rot_qx", import(jmod, "aurora_phys3d_rot_qx", &[i], Some(f64t)));
    hosts.insert("phys3d_rot_qy", import(jmod, "aurora_phys3d_rot_qy", &[i], Some(f64t)));
    hosts.insert("phys3d_rot_qz", import(jmod, "aurora_phys3d_rot_qz", &[i], Some(f64t)));
    hosts.insert("phys3d_rot_qw", import(jmod, "aurora_phys3d_rot_qw", &[i], Some(f64t)));
    hosts.insert("net_host", import(jmod, "aurora_net_host", &[i], Some(i)));
    hosts.insert("net_join", import(jmod, "aurora_net_join", &[ptr_ty, i, i], Some(i)));
    hosts.insert("net_sim", import(jmod, "aurora_net_sim", &[ptr_ty, ptr_ty, i, i], None));
    hosts.insert("net_send_input", import(jmod, "aurora_net_send_input", &[ptr_ty, i], Some(i)));
    hosts.insert("save_settings", import(jmod, "aurora_save_settings", &[ptr_ty, i], Some(i)));
    hosts.insert("load_settings", import(jmod, "aurora_load_settings", &[ptr_ty, i], Some(i)));
    hosts.insert("net_update", import(jmod, "aurora_net_update", &[f64t], None));
    hosts.insert("net_my_id", import(jmod, "aurora_net_my_id", &[], Some(i)));
    hosts.insert("net_is_server", import(jmod, "aurora_net_is_server", &[], Some(i)));
    hosts.insert("net_player_count", import(jmod, "aurora_net_player_count", &[], Some(i)));
    hosts.insert("net_player_id_at", import(jmod, "aurora_net_player_id_at", &[i], Some(i)));
    hosts.insert("net_player_x", import(jmod, "aurora_net_player_x", &[i], Some(f64t)));
    hosts.insert("net_player_y", import(jmod, "aurora_net_player_y", &[i], Some(f64t)));
    hosts.insert("net_player_z", import(jmod, "aurora_net_player_z", &[i], Some(f64t)));
    hosts.insert("net_player_yaw", import(jmod, "aurora_net_player_yaw", &[i], Some(f64t)));
    hosts.insert("net_player_state", import(jmod, "aurora_net_player_state", &[i, i], Some(f64t)));
    hosts.insert("net_set_meta", import(jmod, "aurora_net_set_meta", &[i, f64t], None));
    hosts.insert("net_player_meta", import(jmod, "aurora_net_player_meta", &[i, i], Some(f64t)));
    hosts.insert("net_set_name", import(jmod, "aurora_net_set_name", &[ptr_ty, i], None));
    hosts.insert("net_player_name_len", import(jmod, "aurora_net_player_name_len", &[i], Some(i)));
    hosts.insert("net_player_name_char", import(jmod, "aurora_net_player_name_char", &[i, i], Some(i)));
    hosts.insert("net_local_x", import(jmod, "aurora_net_local_x", &[], Some(f64t)));
    hosts.insert("net_local_y", import(jmod, "aurora_net_local_y", &[], Some(f64t)));
    hosts.insert("net_local_z", import(jmod, "aurora_net_local_z", &[], Some(f64t)));
    hosts.insert("net_local_yaw", import(jmod, "aurora_net_local_yaw", &[], Some(f64t)));
    hosts.insert("net_state", import(jmod, "aurora_net_state", &[i, i], Some(f64t)));
    hosts.insert("net_local_state", import(jmod, "aurora_net_local_state", &[i], Some(f64t)));
    hosts.insert("net_interest", import(jmod, "aurora_net_interest", &[f64t], None));
    hosts.insert("net_max_clients", import(jmod, "aurora_net_max_clients", &[i], None));
    hosts.insert("net_rejected", import(jmod, "aurora_net_rejected", &[], Some(i)));
    hosts.insert("net_set_bot_count", import(jmod, "aurora_net_set_bot_count", &[i], None));
    hosts.insert("net_set_bot", import(jmod, "aurora_net_set_bot", &[i, f64t, f64t, f64t, f64t], None));
    hosts.insert("net_set_bot_meta", import(jmod, "aurora_net_set_bot_meta", &[i, i, f64t], None));
    hosts.insert("net_set_bot_name", import(jmod, "aurora_net_set_bot_name", &[i, ptr_ty, i], None));
    hosts.insert("net_bot_count", import(jmod, "aurora_net_bot_count", &[], Some(i)));
    hosts.insert("net_set_object_count", import(jmod, "aurora_net_set_object_count", &[i], None));
    hosts.insert("net_set_object", import(jmod, "aurora_net_set_object", &[i, f64t, f64t, f64t], None));
    hosts.insert("net_object_count", import(jmod, "aurora_net_object_count", &[], Some(i)));
    hosts.insert("net_object_x", import(jmod, "aurora_net_object_x", &[i], Some(f64t)));
    hosts.insert("net_object_y", import(jmod, "aurora_net_object_y", &[i], Some(f64t)));
    hosts.insert("net_object_z", import(jmod, "aurora_net_object_z", &[i], Some(f64t)));
    hosts.insert("net_hit_radius", import(jmod, "aurora_net_hit_radius", &[f64t], None));
    hosts.insert("net_spawn_at", import(jmod, "aurora_net_spawn_at", &[f64t, f64t, f64t], None));
    hosts.insert("net_fire", import(jmod, "aurora_net_fire", &[f64t, f64t, f64t, f64t, f64t, f64t, i], None));
    hosts.insert("net_server_hit_count", import(jmod, "aurora_net_server_hit_count", &[], Some(i)));
    hosts.insert("net_server_hit_shooter", import(jmod, "aurora_net_server_hit_shooter", &[i], Some(i)));
    hosts.insert("net_server_hit_victim", import(jmod, "aurora_net_server_hit_victim", &[i], Some(i)));
    hosts.insert("net_server_hit_weapon", import(jmod, "aurora_net_server_hit_weapon", &[i], Some(i)));
    hosts.insert("net_server_hit_x", import(jmod, "aurora_net_server_hit_x", &[i], Some(f64t)));
    hosts.insert("net_server_hit_y", import(jmod, "aurora_net_server_hit_y", &[i], Some(f64t)));
    hosts.insert("net_server_hit_z", import(jmod, "aurora_net_server_hit_z", &[i], Some(f64t)));
    hosts.insert("net_server_hits_clear", import(jmod, "aurora_net_server_hits_clear", &[], None));
    hosts.insert("net_push_kill", import(jmod, "aurora_net_push_kill", &[i, i], None));
    hosts.insert("net_kill_count", import(jmod, "aurora_net_kill_count", &[], Some(i)));
    hosts.insert("net_kill_killer", import(jmod, "aurora_net_kill_killer", &[i], Some(i)));
    hosts.insert("net_kill_victim", import(jmod, "aurora_net_kill_victim", &[i], Some(i)));
    hosts.insert("net_kills_clear", import(jmod, "aurora_net_kills_clear", &[], None));
    hosts.insert("net_hit_player", import(jmod, "aurora_net_hit_player", &[], Some(i)));
    hosts.insert("net_hit_x", import(jmod, "aurora_net_hit_x", &[], Some(f64t)));
    hosts.insert("net_hit_y", import(jmod, "aurora_net_hit_y", &[], Some(f64t)));
    hosts.insert("net_hit_z", import(jmod, "aurora_net_hit_z", &[], Some(f64t)));
    hosts.insert("input_bind", import(jmod, "aurora_input_bind", &[i, i], None));
    hosts.insert("input_binding", import(jmod, "aurora_input_binding", &[i], Some(i)));
    hosts.insert("input_down", import(jmod, "aurora_input_down", &[i], Some(i)));
    hosts.insert("input_axis", import(jmod, "aurora_input_axis", &[i, i], Some(f64t)));
    hosts.insert("input_suppress", import(jmod, "aurora_input_suppress", &[i], None));
    hosts.insert("f32_load", import(jmod, "aurora_f32_load", &[i, i], Some(f64t)));
    hosts.insert("f32_store", import(jmod, "aurora_f32_store", &[i, i, f64t], None));
    hosts.insert("sin", import(jmod, "aurora_sin", &[f64t], Some(f64t)));
    hosts.insert("cos", import(jmod, "aurora_cos", &[f64t], Some(f64t)));
    hosts.insert("tan", import(jmod, "aurora_tan", &[f64t], Some(f64t)));
    hosts.insert("pow", import(jmod, "aurora_pow", &[f64t, f64t], Some(f64t)));
    hosts.insert("log", import(jmod, "aurora_log", &[f64t], Some(f64t)));
    hosts.insert("exp", import(jmod, "aurora_exp", &[f64t], Some(f64t)));
    hosts.insert("atan2", import(jmod, "aurora_atan2", &[f64t, f64t], Some(f64t)));
    hosts.insert("draw_text", import(jmod, "aurora_draw_text", &[i, i, ptr_ty, i, i, i], None));
    hosts.insert("draw_int", import(jmod, "aurora_draw_int", &[i, i, i, i, i], None));
    hosts.insert("text_width", import(jmod, "aurora_text_width", &[ptr_ty, i, i], Some(i)));
    hosts.insert("scene_save", import(jmod, "aurora_scene_save", &[ptr_ty, i], Some(i)));
    hosts.insert("scene_load", import(jmod, "aurora_scene_load", &[ptr_ty, i], Some(i)));
    hosts.insert("prof_enter", import(jmod, "aurora_prof_enter", &[ptr_ty, i], None));
    hosts.insert("prof_exit", import(jmod, "aurora_prof_exit", &[], None));
    hosts.insert("str_concat", import(jmod, "aurora_str_concat", &[ptr_ty, ptr_ty, i, ptr_ty, i], None));
    hosts.insert("str_eq", import(jmod, "aurora_str_eq", &[ptr_ty, i, ptr_ty, i], Some(i)));
    hosts.insert("str_char_at", import(jmod, "aurora_str_char_at", &[ptr_ty, i, i], Some(i)));
    hosts.insert("str_substr", import(jmod, "aurora_str_substr", &[ptr_ty, ptr_ty, i, i, i], None));
    hosts.insert("str_starts_with", import(jmod, "aurora_str_starts_with", &[ptr_ty, i, ptr_ty, i], Some(i)));
    hosts.insert("int_to_str", import(jmod, "aurora_int_to_str", &[ptr_ty, i], None));
    hosts.insert("float_to_str", import(jmod, "aurora_float_to_str", &[ptr_ty, types::F64], None));
    hosts.insert("play_note", import(jmod, "aurora_play_note", &[i, i], None));
    hosts.insert("play_sound", import(jmod, "aurora_play_sound", &[i, i, i], None));
    hosts.insert("play_noise", import(jmod, "aurora_play_noise", &[i, i], None));
    hosts.insert("audio_volume", import(jmod, "aurora_audio_volume", &[i], None));
    hosts.insert("window_fullscreen", import(jmod, "aurora_window_fullscreen", &[i], None));
    hosts.insert("audio_stop", import(jmod, "aurora_audio_stop", &[], None));
    hosts.insert("gpu_render", import(jmod, "aurora_gpu_render", &[ptr_ty, i, i], None));
    hosts.insert("window_open", import(jmod, "aurora_window_open", &[i, i], None));
    hosts.insert("window_present", import(jmod, "aurora_window_present", &[], Some(i)));
    hosts.insert("surface_w", import(jmod, "aurora_surface_w", &[], Some(i)));
    hosts.insert("surface_h", import(jmod, "aurora_surface_h", &[], Some(i)));
    hosts.insert("key_down", import(jmod, "aurora_key_down", &[i], Some(i)));
    hosts.insert("input_char", import(jmod, "aurora_input_char", &[], Some(i)));
    hosts.insert("mouse_x", import(jmod, "aurora_mouse_x", &[], Some(i)));
    hosts.insert("mouse_y", import(jmod, "aurora_mouse_y", &[], Some(i)));
    hosts.insert("mouse_down", import(jmod, "aurora_mouse_down", &[], Some(i)));
    // Native debugger hooks (only *called* when `debug`, but always importable).
    hosts.insert("dbg_enter", import(jmod, "aurora_dbg_enter", &[ptr_ty, i], None));
    hosts.insert("dbg_leave", import(jmod, "aurora_dbg_leave", &[], None));
    hosts.insert("dbg_stmt", import(jmod, "aurora_dbg_stmt", &[i], None));
    hosts.insert("dbg_var", import(jmod, "aurora_dbg_var", &[ptr_ty, i, i], None));
    hosts.insert("dbg_var_f64", import(jmod, "aurora_dbg_var_f64", &[ptr_ty, i, types::F64], None));

    // Enum names, so types can be classified as enums (not structs) below.
    let enum_names: HashSet<String> = module
        .items
        .iter()
        .filter_map(|it| match &it.kind {
            ItemKind::Enum(e) => Some(e.name.name.clone()),
            _ => None,
        })
        .collect();

    // Struct layouts (scalar fields only; nested aggregates are unsupported).
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
    for bn in BUILTINS {
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
            let (var, cty) = l
                .scope
                .get(name)
                .cloned()
                .ok_or_else(|| format!("unknown variable `{name}` in JIT"))?;
            (b.use_var(var), cty)
        }
        ExprKind::Path(p) => {
            // Enum unit-variant `Enum::Variant`.
            let (enm, idx) = env
                .enum_variant(p)
                .ok_or("unsupported path expression in JIT")?;
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

    let name = path.segments.last().map(|s| s.ident.name.clone()).unwrap_or_default();
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
    if name == "save_ppm" {
        if let Some(a) = args.first() {
            let (ptr, len) = str_arg(m, b, l, env, &a.value)?;
            let f = m.declare_func_in_func(env.hosts["save_ppm"], b.func);
            b.ins().call(f, &[ptr, len]);
        }
        return Ok(Term::Val(b.ins().iconst(types::I64, 0), Cty::I64));
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
    if matches!(name.as_str(), "load_ppm" | "load_image" | "load_font" | "play_wav" | "scene_save" | "scene_load" | "r3d_load_model") {
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
                argv.push(cast(b, v, &t, pc)?);
            }
            let f = m.declare_func_in_func(env.hosts[name.as_str()], b.func);
            let call = b.ins().call(f, &argv);
            let result = match &ret {
                Some(_) => b.inst_results(call)[0],
                None => b.ins().iconst(types::I64, 0),
            };
            return Ok(Term::Val(result, ret.unwrap_or(Cty::I64)));
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
            other => Cty::Struct(other.to_string()),
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

/// Parameter types and return type of a "scalar" host builtin (physics /
/// pathfinding) — those whose args pass through with a simple per-type
/// coercion. `None` return means void. Keeps the call-site dispatch table-driven.
fn scalar_builtin_sig(name: &str) -> Option<(Vec<Cty>, Option<Cty>)> {
    use Cty::{F64, I64};
    let sig = match name {
        "phys_init" => (vec![F64, F64], None),
        "phys_add" => (vec![F64, F64, F64, F64, I64], Some(I64)),
        "phys_step" => (vec![F64], None),
        "phys_x" => (vec![I64], Some(F64)),
        "phys_y" => (vec![I64], Some(F64)),
        "phys_set_vel" => (vec![I64, F64, F64], None),
        "phys_vel_x" => (vec![I64], Some(F64)),
        "phys_vel_y" => (vec![I64], Some(F64)),
        "phys_apply_impulse" => (vec![I64, F64, F64], None),
        "phys_apply_force" => (vec![I64, F64, F64], None),
        "phys_set_pos" => (vec![I64, F64, F64], None),
        "phys_raycast" => (vec![F64, F64, F64, F64, F64], Some(F64)),
        "nav_init" => (vec![I64, I64], None),
        "nav_wall" => (vec![I64, I64, I64], None),
        "nav_find" => (vec![I64, I64, I64, I64], Some(I64)),
        "nav_x" => (vec![I64], Some(I64)),
        "nav_y" => (vec![I64], Some(I64)),
        // 3D physics.
        "phys3d_init" => (vec![F64, F64, F64], None),
        "phys3d_add_box" => (vec![F64, F64, F64, F64, F64, F64, I64], Some(I64)),
        "phys3d_add_box_rot" => (vec![F64, F64, F64, F64, F64, F64, F64, F64, F64, I64], Some(I64)),
        "phys3d_add_sphere" => (vec![F64, F64, F64, F64, I64], Some(I64)),
        "phys3d_add_capsule" => (vec![F64, F64, F64, F64, F64, I64], Some(I64)),
        "phys3d_add_character" => (vec![F64, F64, F64, F64, F64], Some(I64)),
        "phys3d_step" => (vec![F64], None),
        "phys3d_x" | "phys3d_y" | "phys3d_z" => (vec![I64], Some(F64)),
        "phys3d_vel_x" | "phys3d_vel_y" | "phys3d_vel_z" => (vec![I64], Some(F64)),
        "phys3d_set_vel" => (vec![I64, F64, F64, F64], None),
        "phys3d_set_pos" => (vec![I64, F64, F64, F64], None),
        "phys3d_apply_impulse" => (vec![I64, F64, F64, F64], None),
        "phys3d_move_character" => (vec![I64, F64, F64, F64, F64], None),
        "phys3d_grounded" => (vec![I64], Some(I64)),
        "phys3d_raycast" => (vec![F64, F64, F64, F64, F64, F64, F64], Some(F64)),
        // 3D pathfinding.
        "nav3d_init" => (vec![I64, I64, I64], None),
        "nav3d_wall" => (vec![I64, I64, I64, I64], None),
        "nav3d_find" => (vec![I64, I64, I64, I64, I64, I64], Some(I64)),
        "nav3d_x" | "nav3d_y" | "nav3d_z" => (vec![I64], Some(I64)),
        "navmesh_find" => (vec![F64, F64, F64, F64, F64, F64], Some(I64)),
        "navmesh_x" | "navmesh_y" | "navmesh_z" => (vec![I64], Some(F64)),
        // 3D rendering.
        "r3d_make_box" => (vec![F64, F64, F64], Some(I64)),
        "r3d_make_box_sized" => (vec![F64, F64, F64, F64, F64, F64], Some(I64)),
        "r3d_make_box_emissive" => (vec![F64, F64, F64, F64, F64, F64], Some(I64)),
        "r3d_make_sphere" => (vec![I64, F64, F64, F64], Some(I64)),
        "r3d_make_plane" => (vec![F64, F64, F64, F64, F64], Some(I64)),
        "r3d_camera" => (vec![F64, F64, F64, F64, F64, F64, F64], None),
        "r3d_camera_roll" => (vec![F64], None),
        "r3d_light" => (vec![F64, F64, F64, F64, F64, F64, F64], None),
        "r3d_clear" => (vec![F64, F64, F64], None),
        "r3d_begin" => (vec![], None),
        "r3d_draw" => (vec![I64, F64, F64, F64, F64, F64, F64, F64], None),
        "r3d_draw_tint" => (vec![I64, F64, F64, F64, F64, F64, F64, F64, F64, F64, F64], None),
        "r3d_draw_shield" => (vec![I64, F64, F64, F64, F64, F64, F64, F64, F64, F64], None),
        "r3d_draw_on_joint" => (vec![I64, I64, I64, F64, F64, F64, F64, F64, F64, F64, F64, F64, F64, F64, F64, F64, F64], None),
        "r3d_joint_dump" => (vec![I64], None),
        "r3d_anim_play" => (vec![I64, I64, I64, F64, F64], None),
        "r3d_anim_update" => (vec![I64, F64], None),
        "r3d_anim_play_upper" => (vec![I64, I64, I64, F64, F64, I64], None),
        "r3d_anim_stop_upper" => (vec![I64, F64], None),
        "r3d_clip_count" => (vec![I64], Some(I64)),
        "r3d_present" => (vec![], Some(I64)),
        "r3d_fog" => (vec![F64, F64, F64, F64], None),
        "r3d_speedlines" => (vec![F64, F64], None),
        "r3d_damage" => (vec![F64, F64, F64, F64, F64], None),
        "r3d_blur" => (vec![F64], None),
        "r3d_sky" => (vec![I64, F64, F64, F64, F64, F64, F64], None),
        "r3d_shadows" | "r3d_ssao" | "r3d_point_shadows" => (vec![I64], None),
        "r3d_clear_lights" => (vec![], None),
        "r3d_point_light" => (vec![F64, F64, F64, F64, F64, F64, F64, F64], None),
        "r3d_make_sprite" => (vec![F64, F64, F64], Some(I64)),
        "r3d_draw_billboard" => (vec![I64, F64, F64, F64, F64], None),
        "r3d_debug_line" => (vec![F64, F64, F64, F64, F64, F64, F64, F64, F64], None),
        "r3d_frustum_cull" => (vec![I64], None),
        "r3d_screen_x" | "r3d_screen_y" => (vec![F64, F64, F64], Some(F64)),
        // FPS input.
        "mouse_dx" | "mouse_dy" | "mouse_scroll" | "frame_dt" => (vec![], Some(F64)),
        "sleep_ms" => (vec![I64], None),
        "mouse_button" => (vec![I64], Some(I64)),
        "grab_mouse" => (vec![I64], None),
        // 3D positional audio.
        "audio_listener" => (vec![F64, F64, F64, F64, F64, F64], None),
        "play_sound_at" => (vec![I64, I64, I64, F64, F64, F64], None),
        // Rich 3D physics.
        "phys3d_raycast_full" => (vec![F64, F64, F64, F64, F64, F64, F64], Some(I64)),
        "phys3d_raycast_ex" => (vec![I64, F64, F64, F64, F64, F64, F64, F64], Some(I64)),
        "phys3d_hit_x" | "phys3d_hit_y" | "phys3d_hit_z" => (vec![], Some(F64)),
        "phys3d_hit_nx" | "phys3d_hit_ny" | "phys3d_hit_nz" => (vec![], Some(F64)),
        "phys3d_hit_body" => (vec![], Some(I64)),
        "phys3d_spherecast" => (vec![F64, F64, F64, F64, F64, F64, F64, F64], Some(F64)),
        "phys3d_overlap_sphere" => (vec![F64, F64, F64, F64], Some(I64)),
        "phys3d_apply_force" | "phys3d_apply_torque" | "phys3d_set_angvel" => {
            (vec![I64, F64, F64, F64], None)
        }
        "phys3d_set_rot" => (vec![I64, F64, F64, F64, F64], None),
        "phys3d_rot_qx" | "phys3d_rot_qy" | "phys3d_rot_qz" | "phys3d_rot_qw" => {
            (vec![I64], Some(F64))
        }
        // Multiplayer. net_join (string), net_sim (closure), and net_send_input
        // (array) are dispatched separately.
        "net_host" => (vec![I64], Some(I64)),
        "net_update" => (vec![F64], None),
        "net_my_id" | "net_is_server" | "net_player_count" | "net_rejected" => (vec![], Some(I64)),
        "net_max_clients" => (vec![I64], None),
        "net_bot_count" => (vec![], Some(I64)),
        "net_object_count" => (vec![], Some(I64)),
        "net_set_object_count" => (vec![I64], None),
        "net_set_object" => (vec![I64, F64, F64, F64], None),
        "net_object_x" | "net_object_y" | "net_object_z" => (vec![I64], Some(F64)),
        "net_set_bot_count" => (vec![I64], None),
        "net_set_bot" => (vec![I64, F64, F64, F64, F64], None),
        "net_set_bot_meta" => (vec![I64, I64, F64], None),
        "net_player_id_at" => (vec![I64], Some(I64)),
        "net_player_state" => (vec![I64, I64], Some(F64)),
        "net_set_meta" => (vec![I64, F64], None),
        "net_player_meta" => (vec![I64, I64], Some(F64)),
        "net_player_name_len" => (vec![I64], Some(I64)),
        "net_player_name_char" => (vec![I64, I64], Some(I64)),
        "net_player_x" | "net_player_y" | "net_player_z" | "net_player_yaw" => {
            (vec![I64], Some(F64))
        }
        "net_local_x" | "net_local_y" | "net_local_z" | "net_local_yaw" => (vec![], Some(F64)),
        "net_state" => (vec![I64, I64], Some(F64)),
        "net_local_state" => (vec![I64], Some(F64)),
        "net_interest" => (vec![F64], None),
        "net_hit_radius" => (vec![F64], None),
        "net_spawn_at" => (vec![F64, F64, F64], None),
        "net_fire" => (vec![F64, F64, F64, F64, F64, F64, I64], None),
        "net_server_hit_count" | "net_server_hits_clear" => {
            if name == "net_server_hit_count" { (vec![], Some(I64)) } else { (vec![], None) }
        }
        "net_server_hit_shooter" | "net_server_hit_victim" | "net_server_hit_weapon" => (vec![I64], Some(I64)),
        "net_server_hit_x" | "net_server_hit_y" | "net_server_hit_z" => (vec![I64], Some(F64)),
        "net_push_kill" => (vec![I64, I64], None),
        "net_kill_count" => (vec![], Some(I64)),
        "net_kill_killer" | "net_kill_victim" => (vec![I64], Some(I64)),
        "net_kills_clear" => (vec![], None),
        "net_hit_player" => (vec![], Some(I64)),
        "net_hit_x" | "net_hit_y" | "net_hit_z" => (vec![], Some(F64)),
        // Rebindable input-action layer + raw f32-blob accessors.
        "input_bind" => (vec![I64, I64], None),
        "input_suppress" => (vec![I64], None),
        "input_binding" | "input_down" => (vec![I64], Some(I64)),
        "input_axis" => (vec![I64, I64], Some(F64)),
        "f32_load" => (vec![I64, I64], Some(F64)),
        "f32_store" => (vec![I64, I64, F64], None),
        _ => return None,
    };
    Some(sig)
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
mod tests;
