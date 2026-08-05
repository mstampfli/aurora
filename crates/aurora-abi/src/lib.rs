//! The single source of truth for Aurora's runtime builtins.
//!
//! A builtin used to be registered in up to eight hand-maintained places (name
//! list, signature match arms, JIT symbol table, host import table, the AOT
//! link-keeper, the docs), which drifted apart silently: `net_fire` grew a
//! seventh parameter and the docs, the example, and the shipped multiplayer
//! program kept passing six, compiling to a program that ran and did nothing.
//!
//! There is now ONE table. [`for_each_builtin!`] is an x-macro: it hands the
//! whole table to a callback macro, and every consumer generates its own view:
//!
//! * this crate: [`TABLE`], [`is_builtin`], [`builtin_names`], [`lookup`],
//!   [`scalar_sig`];
//! * `aurora-codegen`: the JIT symbol registrations, the backend's host import
//!   table, and the call-site signature lookup;
//! * `aurora-runtime`: `force_link`, which keeps every host symbol in an AOT link.
//!
//! This crate depends on NOTHING, so the front end, the backend, and the runtime
//! can all sit above it without leaking their dependencies into each other. The
//! table names Rust functions only as *identifiers*, never as function pointers,
//! so it does not need `aurora-runtime` itself; `aurora-codegen` and
//! `aurora-runtime` are what resolve `symbol` to a real function. That is also
//! the safety net: a row naming a runtime function that does not exist fails to
//! COMPILE, in both of those crates, instead of failing at run time.
//!
//! # Adding a builtin
//!
//! 1. write `pub extern "C" fn aurora_<name>(..)` in `aurora-runtime` - or
//!    `pub unsafe extern "C" fn`, with a `# Safety` section, if it takes a
//!    pointer (a `str` or array argument) and reads or writes through it;
//! 2. add one row to the table below;
//! 3. document it in `docs/04-stdlib-and-builtins.md` (a test enforces this).
//!
//! Nothing else. A `scalar` row needs no backend code at all: the shared
//! call-site dispatch lowers it from the table.
//!
//! # Row format
//!
//! `[kind, aurora_name, c_symbol, [parameter types], return type]`
//!
//! | `kind` | meaning |
//! |---|---|
//! | `scalar` | an Aurora builtin taking plain `i64`/`f64` arguments; the backend's generic dispatch lowers the call, so it needs no bespoke code |
//! | `text` | an Aurora builtin that takes and/or returns a `str`, also lowered by a generic dispatch and so needing no bespoke code either |
//! | `special` | an Aurora builtin with bespoke lowering in `aurora-codegen` (arrays, closures, and the string builtins that predate `text`) |
//! | `raster` | a CPU rasterizer builtin: all-integer arguments, lowered as a plain host call into `aurora-gfx` |
//! | `internal` | a runtime host function that Aurora source cannot call: printing primitives, string helpers, ECS plumbing, debugger hooks |
//! | `inline` | an Aurora builtin the backend expands inline, with no runtime call at all; its symbol column is `none` and it has no signature |
//! | `linkonly` | a runtime function that only needs a JIT symbol and an AOT link edge, so an `@extern` declaration can bind it |
//!
//! Parameter types are `I64`, `F64`, or `Ptr` (the target pointer type); the
//! return column is those, `Str`, or `void`. An Aurora `str` argument is passed
//! as the pair `Ptr, I64` (data, length), and a `Str` RETURN is passed as a
//! caller-allocated 2-slot out-pointer prepended to the parameter list, which
//! the row does not spell (see [`Builtin::abi_params`]). Only a `text` row may
//! return `Str`, because only its dispatch knows to allocate that slot.
//!
//! **Place in the graph.** Depends on nothing. `codegen` and `runtime` both read it, which is the whole point.
//!
//! **Never.** Never contains an implementation. A row here is a name, a signature and a symbol; the body lives in `aurora-runtime`.

use std::collections::{HashMap, HashSet};
use std::sync::OnceLock;

/// An ABI-level parameter or return type of a runtime host function.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Ty {
    I64,
    F64,
    /// The target's pointer type (`i64` on every backend Aurora supports today).
    Ptr,
    /// An Aurora `str`.
    ///
    /// As a RESULT: the caller allocates its two slots and passes their address
    /// as a leading [`Ty::Ptr`] parameter, so the host function itself returns
    /// nothing.
    ///
    /// As a PARAMETER: one Aurora argument occupying the two ABI slots its value
    /// holds, `Ptr, I64`. Spelling those slots out as `[Ptr, I64]` is what the
    /// table used to do, and it could not then tell a string from an array or an
    /// out-pointer - so every backend needed a HAND-WRITTEN LIST of which
    /// builtins take a string, and a builtin missing from that list compiled to
    /// "cannot find function" despite being a row here. Say `Str` and the
    /// lowering follows from the type.
    Str,
    /// An Aurora array argument, passed as its address and its length: two ABI
    /// slots, `Ptr, I64`, exactly like [`Ty::Str`], and one argument at the call
    /// site because the length comes from the array's TYPE rather than from a
    /// second argument.
    ///
    /// Distinct from `Str` because they are lowered differently - a string
    /// argument is read out of a string value, an array's length is a constant
    /// the compiler knows - and telling them apart is the whole point: they were
    /// both `[Ptr, I64]`, so nothing could.
    Arr,
    /// A PREDICATE result. The host function returns an `i64` 0 or 1 exactly as
    /// [`Ty::I64`] does - the difference is what the CHECKER is told, so
    /// `if starts_with(a, b)` type-checks as a condition instead of being an
    /// `i64` where a `bool` belongs.
    ///
    /// Added because the table had no word for it: a predicate declared `I64`
    /// rejects its own idiomatic use, and one declared `void` is unchecked
    /// entirely - which is how `str + char_at(..)` reached codegen and
    /// segfaulted rather than being a type error.
    Bool,
}

/// How the backend treats a table row. See the crate docs for the full table.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Kind {
    /// Aurora builtin lowered by the generic scalar call-site dispatch.
    Scalar,
    /// Aurora builtin taking and/or returning a `str`, lowered by the generic
    /// text call-site dispatch.
    Text,
    /// Aurora builtin with bespoke lowering in `aurora-codegen`.
    Special,
    /// CPU rasterizer builtin: all-integer arguments, lowered as a plain host
    /// call into `aurora-gfx` through the framebuffer.
    ///
    /// Its own kind rather than a name list beside the lowering, because that
    /// list existed and drifted: `fill_rect_alpha` was a row in this table and
    /// the compiler still refused to find it. The row is the only place that
    /// says a builtin is one of these.
    Raster,
    /// Runtime host function that Aurora source cannot call.
    Internal,
    /// Aurora builtin expanded inline, with no runtime call.
    Inline,
    /// Runtime function that only needs a JIT symbol and an AOT link edge.
    LinkOnly,
}

impl Kind {
    /// Can Aurora source call a builtin of this kind by name?
    pub const fn is_aurora_visible(self) -> bool {
        matches!(
            self,
            Kind::Scalar | Kind::Text | Kind::Special | Kind::Raster | Kind::Inline
        )
    }

    /// Is this row backed by an `aurora_*` function in `aurora-runtime`? Those
    /// rows get a JIT symbol registration and an AOT link edge.
    pub const fn has_host_fn(self) -> bool {
        !matches!(self, Kind::Inline)
    }

    /// Is this row declared as an import in the backend's host table?
    pub const fn is_host_import(self) -> bool {
        matches!(
            self,
            Kind::Scalar | Kind::Text | Kind::Special | Kind::Raster | Kind::Internal
        )
    }
}

/// One row of the builtin table.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Builtin {
    /// The name Aurora source calls, or - for [`Kind::Internal`] /
    /// [`Kind::LinkOnly`] - the key the backend looks the import up by.
    pub name: &'static str,
    pub kind: Kind,
    /// The C symbol of the runtime function, or `""` for [`Kind::Inline`].
    pub symbol: &'static str,
    /// ABI parameter types, WITHOUT the out-pointer a [`Ty::Str`] return adds
    /// (see [`Builtin::abi_params`]). Empty for [`Kind::Inline`], which has no
    /// single signature (`min`/`max`/`abs` are polymorphic over `i64`/`f64`).
    pub params: &'static [Ty],
    /// Return type, or `None` for a function that returns nothing.
    pub ret: Option<Ty>,
    /// Which thread's runtime state this touches.
    pub home: Home,
}

/// Where a builtin's state lives, and therefore which threads may call it.
///
/// Systems in one stage layer run on worker threads. The world and the
/// simulation subsystems are routed to the batch owner, so a worker sees the
/// program's own; everything the FRONTEND owns is not, and never will be -
/// sharing a window between threads is not a thing to fix, it is a thing not to
/// do.
///
/// This column exists because the failure it prevents is silent. A worker that
/// cannot see a subsystem does not error: it reports an empty one, and "no
/// route", "nothing there", "no fixed step" are all legal answers that every
/// caller already handles. A game shipped four iterations of creatures that had
/// navigation and never used it.
///
/// Every row must say which it is. That is the point of the column: a builtin
/// added without a thought about it will not compile, rather than joining the
/// silent set.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Home {
    /// Pure, or reaches only state the batch shares with its workers.
    Shared,
    /// Reaches state belonging to the thread that owns the program: the window,
    /// the framebuffer, the font, the audio mixer, the GPU. Calling it from a
    /// system that may run on a worker is refused at compile time.
    Owner,
}

macro_rules! home_of {
    (shared) => {
        Home::Shared
    };
    (owner) => {
        Home::Owner
    };
}

impl Builtin {
    /// The parameter types the host function is actually declared with: a
    /// [`Ty::Str`] result is a caller-allocated 2-slot out-pointer passed FIRST,
    /// so it is not a Cranelift return value at all.
    pub fn abi_params(&self) -> Vec<Ty> {
        // A `Str` or `Arr` argument is one Aurora value in two machine slots.
        let declared = self.params.iter().flat_map(|t| match t {
            Ty::Str | Ty::Arr => [Some(Ty::Ptr), Some(Ty::I64)],
            other => [Some(*other), None],
        });
        let declared = declared.flatten();
        match self.ret {
            Some(Ty::Str) => std::iter::once(Ty::Ptr).chain(declared).collect(),
            _ => declared.collect(),
        }
    }

    /// The type the host function returns, or `None` when it returns nothing -
    /// including a [`Ty::Str`] result, which is written through the out-pointer.
    pub fn abi_ret(&self) -> Option<Ty> {
        match self.ret {
            Some(Ty::Str) => None,
            other => other,
        }
    }

    /// How many arguments an Aurora call site passes, for the kinds whose call
    /// the table fully describes ([`Kind::Scalar`] and [`Kind::Text`]): a `str`
    /// or array argument takes two ABI slots but is ONE argument in the source,
    /// and a bare [`Ty::Ptr`] is plumbing the call site does not spell at all -
    /// an out-pointer or a closure. `None` for the kinds it does not model.
    ///
    /// This used to subtract every `Ptr`, which was a guess that happened to be
    /// right: it counted the pointer half of a string pair and left the length
    /// half standing in for it. With `Str` and `Arr` spelled the count is what
    /// it says it is.
    pub fn arity(&self) -> Option<usize> {
        match self.kind {
            Kind::Scalar | Kind::Text => {
                Some(self.params.iter().filter(|t| **t != Ty::Ptr).count())
            }
            _ => None,
        }
    }
}

/// The builtin table. Hands every row to the callback macro `$m` in one
/// invocation, so a consumer is a single `$( .. )*` repetition with no recursion.
///
/// ```ignore
/// macro_rules! names {
///     ($([$kind:ident, $name:ident, $sym:ident, [$($p:ident),*], $ret:ident, $home:ident])*) => {
///         &[$(stringify!($name)),*]
///     };
/// }
/// let all: &[&str] = aurora_abi::for_each_builtin!(names);
/// ```
#[macro_export]
macro_rules! for_each_builtin {
    ($m:ident) => {
        $m! {
        [internal, vstack_enter,                  aurora_vstack_enter,                   [],                                              void, shared]
        [internal, vstack_leave,                  aurora_vstack_leave,                   [],                                              void, shared]
        [internal, vstack_alloc,                  aurora_vstack_alloc,                   [I64],                                           Ptr, shared]
        [internal, print_i64,                     aurora_print_i64,                      [I64],                                           void, shared]
        [internal, print_f64,                     aurora_print_f64,                      [F64],                                           void, shared]
        [internal, print_str,                     aurora_print_str,                      [Str],                                           void, shared]
        [internal, print_nl,                      aurora_print_nl,                       [],                                              void, shared]
        [raster,  framebuffer,                   aurora_framebuffer,                    [I64, I64],                                      void, owner]
        [raster,  clear,                         aurora_clear,                          [I64, I64, I64],                                 void, owner]
        [raster,  pixel,                         aurora_pixel,                          [I64, I64, I64, I64, I64],                       void, owner]
        [raster,  pixel_alpha,                   aurora_pixel_alpha,                    [I64, I64, I64, I64, I64, I64],                  void, owner]
        [raster,  fill_rect_alpha,               aurora_fill_rect_alpha,                [I64, I64, I64, I64, I64, I64, I64, I64],        void, owner]
        [raster,  triangle,                      aurora_triangle,                       [I64, I64, I64, I64, I64, I64, I64, I64, I64],   void, owner]
        [raster,  fb_get,                        aurora_fb_get,                         [I64, I64],                                      I64, owner]
        [special, save_ppm,                      aurora_save_ppm,                       [Str],                                           void, owner]
        [internal, spawn_entity,                  aurora_spawn_entity,                   [],                                              I64, shared]
        [special,  despawn,                       aurora_despawn,                        [I64],                                           void, shared]
        [internal, store_component,               aurora_store_component,                [I64, I64, Ptr, I64],                            void, shared]
        [internal, get_component,                 aurora_get_component,                  [I64, I64],                                      Ptr, shared]
        [internal, query_begin,                   aurora_query_begin,                    [Arr],                                           I64, shared]
        [internal, query_entity,                  aurora_query_entity,                   [I64],                                           I64, shared]
        [internal, query_end,                     aurora_query_end,                      [],                                              void, shared]
        [special,  entity_count,                  aurora_entity_count,                   [],                                              I64, shared]
        [special,  world_clear,                   aurora_world_clear,                    [],                                              void, shared]

        // Audio + windowing builtins.
        [special, gpu_compute,                   aurora_gpu_compute,                    [Str, Arr],                                      void, owner]
        [special, par_for,                       aurora_par_for,                        [Arr, Ptr, Ptr],                                 void, shared]
        [internal, run_parallel,                  aurora_run_parallel,                   [Arr],                                           void, shared]
        [special,  net_bind,                      aurora_net_bind,                       [I64],                                           I64, shared]
        [special, net_connect,                   aurora_net_connect,                    [Str, I64],                                      I64, shared]
        [special, net_send,                      aurora_net_send,                       [Str],                                           I64, shared]
        [special,  net_recv,                      aurora_net_recv,                       [Ptr],                                           void, shared]
        [special,  frame_reset,                   aurora_frame_reset,                    [],                                              void, shared]
        [special, load_ppm,                      aurora_load_ppm,                       [Str],                                           I64, owner]

        // Determinism + data builtins.
        [scalar,   srand,                         aurora_srand,                          [I64],                                           void, shared]
        [scalar,   rand,                          aurora_rand,                           [],                                              F64, shared]
        [scalar,   rand_range,                    aurora_rand_range,                     [F64, F64],                                      F64, shared]
        [scalar,   rand_int,                      aurora_rand_int,                       [I64, I64],                                      I64, shared]
        [scalar,   set_fixed_dt,                  aurora_set_fixed_dt,                   [F64],                                           void, shared]
        [special, save_png,                      aurora_save_png,                       [Str],                                           void, shared]
        [scalar,   fb_width,                      aurora_fb_width,                       [],                                              I64, owner]
        [scalar,   fb_height,                     aurora_fb_height,                      [],                                              I64, owner]
        [special, read_file,                     aurora_read_file,                      [Ptr, Str],                                      void, shared]
        [special, write_file,                    aurora_write_file,                     [Str, Str],                                      I64, shared]
        [special, file_exists,                   aurora_file_exists,                    [Str],                                           I64, shared]
        [text, is_i64,                        aurora_is_i64,                         [Str],                                           I64, shared]
        [text, parse_i64,                     aurora_parse_i64,                      [Str, I64],                                      I64, shared]
        [text, is_f64,                        aurora_is_f64,                         [Str],                                           I64, shared]
        [text, parse_f64,                     aurora_parse_f64,                      [Str, F64],                                      F64, shared]
        [special, json_parse,                    aurora_json_parse,                     [Str],                                           I64, shared]
        [special, json_load,                     aurora_json_load,                      [Str],                                           I64, shared]
        [special, json_get,                      aurora_json_get,                       [I64, Str],                                      I64, shared]
        [scalar,   json_at,                       aurora_json_at,                        [I64, I64],                                      I64, shared]
        [scalar,   json_len,                      aurora_json_len,                       [I64],                                           I64, shared]
        [scalar,   json_num,                      aurora_json_num,                       [I64],                                           F64, shared]
        [scalar,   json_int,                      aurora_json_int,                       [I64],                                           I64, shared]
        [scalar,   json_bool,                     aurora_json_bool,                      [I64],                                           I64, shared]
        [special,  json_str,                      aurora_json_str,                       [Ptr, I64],                                      void, shared]
        [scalar,   json_kind,                     aurora_json_kind,                      [I64],                                           I64, shared]
        [special, json_has,                      aurora_json_has,                       [I64, Str],                                      I64, shared]
        [special,  json_key,                      aurora_json_key,                       [Ptr, I64, I64],                                 void, shared]
        [scalar,   json_free,                     aurora_json_free,                      [I64],                                           void, shared]
        [scalar,   json_new_obj,                  aurora_json_new_obj,                   [],                                              I64, shared]
        [scalar,   json_new_arr,                  aurora_json_new_arr,                   [],                                              I64, shared]
        [special, json_set,                      aurora_json_set,                       [I64, Str, I64],                                 void, shared]
        [special, json_set_num,                  aurora_json_set_num,                   [I64, Str, F64],                                 void, shared]
        [special, json_set_str,                  aurora_json_set_str,                   [I64, Str, Str],                                 void, shared]
        [special, json_set_bool,                 aurora_json_set_bool,                  [I64, Str, I64],                                 void, shared]
        [scalar,   json_push,                     aurora_json_push,                      [I64, I64],                                      void, shared]
        [scalar,   json_push_num,                 aurora_json_push_num,                  [I64, F64],                                      void, shared]
        [special, json_push_str,                 aurora_json_push_str,                  [I64, Str],                                      void, shared]
        [special,  json_to_str,                   aurora_json_to_str,                    [Ptr, I64],                                      void, shared]
        [special, json_write,                    aurora_json_write,                     [I64, Str],                                      I64, shared]
        [special, audio_capture_save,            aurora_audio_capture_save,             [Str],                                           I64, owner]
        [special, r3d_capture,                   aurora_r3d_capture,                    [Str],                                           I64, owner]
        [special, r3d_capture_size,              aurora_r3d_capture_size,               [Str, I64, I64],                                 I64, owner]
        [scalar,   inject_key,                    aurora_inject_key,                     [I64, I64],                                      void, owner]
        [scalar,   inject_mouse_move,             aurora_inject_mouse_move,              [F64, F64],                                      void, owner]
        [scalar,   inject_mouse_pos,              aurora_inject_mouse_pos,               [I64, I64],                                      void, owner]
        [scalar,   inject_mouse_button,           aurora_inject_mouse_button,            [I64, I64],                                      void, owner]
        [scalar,   inject_scroll,                 aurora_inject_scroll,                  [F64],                                           void, owner]
        [scalar,   inject_char,                   aurora_inject_char,                    [I64],                                           void, owner]
        [internal, oob,                           aurora_oob,                            [I64, I64],                                      void, shared]
        [scalar,   frame_dt,                      aurora_frame_dt,                       [],                                              F64, shared]
        [internal, run_fixed,                     aurora_run_fixed,                      [Ptr, Ptr, I64, F64],                            I64, shared]
        [scalar,   set_tick_rate,                 aurora_set_tick_rate,                  [F64],                                           void, shared]
        [scalar,   tick_count,                    aurora_tick_count,                     [],                                              I64, shared]
        [scalar,   tick_rate,                     aurora_tick_rate,                      [],                                              F64, shared]
        [scalar,   pin_frame_to_tick,             aurora_pin_frame_to_tick,              [],                                              void, shared]
        [scalar,   tick_delta,                    aurora_tick_delta,                     [],                                              F64, shared]
        [scalar,   tick_alpha,                    aurora_tick_alpha,                     [],                                              F64, shared]
        [scalar,   sleep_ms,                      aurora_sleep_ms,                       [I64],                                           void, shared]
        [internal, divzero,                       aurora_divzero,                        [],                                              void, shared]
        [internal, fmod,                          aurora_fmod,                           [F64, F64],                                      F64, shared]
        [special, load_image,                    aurora_load_image,                     [Str],                                           I64, owner]
        [special, image_open,                    aurora_image_open,                     [Str],                                           I64, owner]
        [scalar,   image_width,                   aurora_image_width,                    [I64],                                           I64, owner]
        [scalar,   image_height,                  aurora_image_height,                   [I64],                                           I64, owner]
        [scalar,   image_pixel,                   aurora_image_pixel,                    [I64, I64, I64],                                 I64, owner]
        [scalar,   image_mean_luma,               aurora_image_mean_luma,                [I64, I64, I64, I64, I64],                       F64, owner]
        [scalar,   image_diff,                    aurora_image_diff,                     [I64, I64, I64, I64, I64, I64],                  F64, owner]
        [scalar,   image_free,                    aurora_image_free,                     [I64],                                           I64, owner]
        [special, load_font,                     aurora_load_font,                      [Str],                                           I64, owner]
        [special, play_wav,                      aurora_play_wav,                       [Str],                                           I64, owner]
        [special, load_sound,                    aurora_load_sound,                     [Str],                                           I64, shared]
        [scalar,   phys_init,                     aurora_phys_init,                      [F64, F64],                                      void, shared]
        [scalar,   phys_add,                      aurora_phys_add,                       [F64, F64, F64, F64, I64],                       I64, shared]
        [scalar,   phys_remove,                   aurora_phys_remove,                    [I64],                                           I64, shared]
        [scalar,   phys_alive,                    aurora_phys_alive,                     [I64],                                           I64, shared]
        [scalar,   phys_step,                     aurora_phys_step,                      [F64],                                           void, shared]
        [scalar,   phys_x,                        aurora_phys_x,                         [I64],                                           F64, shared]
        [scalar,   phys_y,                        aurora_phys_y,                         [I64],                                           F64, shared]
        [scalar,   phys_set_vel,                  aurora_phys_set_vel,                   [I64, F64, F64],                                 void, shared]
        [scalar,   phys_vel_x,                    aurora_phys_vel_x,                     [I64],                                           F64, shared]
        [scalar,   phys_vel_y,                    aurora_phys_vel_y,                     [I64],                                           F64, shared]
        [scalar,   phys_apply_impulse,            aurora_phys_apply_impulse,             [I64, F64, F64],                                 void, shared]
        [scalar,   phys_apply_force,              aurora_phys_apply_force,               [I64, F64, F64],                                 void, shared]
        [scalar,   phys_set_pos,                  aurora_phys_set_pos,                   [I64, F64, F64],                                 void, shared]
        [scalar,   phys_raycast,                  aurora_phys_raycast,                   [F64, F64, F64, F64, F64],                       F64, shared]
        [scalar,   nav_init,                      aurora_nav_init,                       [I64, I64],                                      void, shared]
        [scalar,   nav_wall,                      aurora_nav_wall,                       [I64, I64, I64],                                 void, shared]
        [scalar,   nav_find,                      aurora_nav_find,                       [I64, I64, I64, I64],                            I64, shared]
        [scalar,   nav_x,                         aurora_nav_x,                          [I64],                                           I64, shared]
        [scalar,   nav_y,                         aurora_nav_y,                          [I64],                                           I64, shared]

        // 3D physics (Rapier 3D).
        [scalar,   phys3d_init,                   aurora_phys3d_init,                    [F64, F64, F64],                                 void, shared]
        [scalar,   phys3d_add_box,                aurora_phys3d_add_box,                 [F64, F64, F64, F64, F64, F64, I64],             I64, shared]
        [scalar,   phys3d_add_box_rot,            aurora_phys3d_add_box_rot,             [F64, F64, F64, F64, F64, F64, F64, F64, F64, I64], I64, shared]
        [scalar,   phys3d_add_sphere,             aurora_phys3d_add_sphere,              [F64, F64, F64, F64, I64],                       I64, shared]
        [scalar,   phys3d_add_capsule,            aurora_phys3d_add_capsule,             [F64, F64, F64, F64, F64, I64],                  I64, shared]
        [scalar,   phys3d_add_character,          aurora_phys3d_add_character,           [F64, F64, F64, F64, F64],                       I64, shared]
        [special, phys3d_add_trimesh,            aurora_phys3d_add_trimesh,             [Arr, Arr],                                      I64, shared]
        [scalar,   phys3d_add_model_collider,     aurora_phys3d_add_model_collider,      [I64, F64, F64, F64, F64, F64, F64, F64],        I64, shared]
        [scalar,   phys3d_remove,                 aurora_phys3d_remove,                  [I64],                                           I64, shared]
        [scalar,   phys3d_character_blocking,     aurora_phys3d_character_blocking,      [I64, I64],                                      void, shared]
        [scalar,   phys3d_alive,                  aurora_phys3d_alive,                   [I64],                                           I64, shared]
        [scalar,   phys3d_step,                   aurora_phys3d_step,                    [F64],                                           void, shared]
        [scalar,   phys3d_x,                      aurora_phys3d_x,                       [I64],                                           F64, shared]
        [scalar,   phys3d_y,                      aurora_phys3d_y,                       [I64],                                           F64, shared]
        [scalar,   phys3d_z,                      aurora_phys3d_z,                       [I64],                                           F64, shared]
        [scalar,   phys3d_vel_x,                  aurora_phys3d_vel_x,                   [I64],                                           F64, shared]
        [scalar,   phys3d_vel_y,                  aurora_phys3d_vel_y,                   [I64],                                           F64, shared]
        [scalar,   phys3d_vel_z,                  aurora_phys3d_vel_z,                   [I64],                                           F64, shared]
        [scalar,   phys3d_set_vel,                aurora_phys3d_set_vel,                 [I64, F64, F64, F64],                            void, shared]
        [scalar,   phys3d_set_pos,                aurora_phys3d_set_pos,                 [I64, F64, F64, F64],                            void, shared]
        [scalar,   phys3d_apply_impulse,          aurora_phys3d_apply_impulse,           [I64, F64, F64, F64],                            void, shared]
        [scalar,   phys3d_move_character,         aurora_phys3d_move_character,          [I64, F64, F64, F64, F64],                       void, shared]
        [scalar,   phys3d_grounded,               aurora_phys3d_grounded,                [I64],                                           I64, shared]
        [scalar,   phys3d_character_solid,       aurora_phys3d_character_solid,         [I64, I64],                                      void, shared]
        [scalar,   phys3d_raycast,                aurora_phys3d_raycast,                 [F64, F64, F64, F64, F64, F64, F64],             F64, shared]

        // Heightmap terrain: one heightfield behind the renderer, the physics
        // collider, and the height query.
        [scalar,   terrain_generate,              aurora_terrain_generate,               [I64, I64, F64, F64],                            I64, shared]
        [text, terrain_load,                  aurora_terrain_load,                   [Str],                                           I64, shared]
        [text, terrain_save,                  aurora_terrain_save,                   [Str],                                           I64, shared]
        [scalar,   terrain_color,                 aurora_terrain_color,                  [F64, F64, F64],                                 void, shared]
        [scalar,   terrain_draw,                  aurora_terrain_draw,                   [],                                              void, shared]
        [scalar,   terrain_height,                aurora_terrain_height,                 [F64, F64],                                      F64, shared]
        [scalar,   terrain_collider,              aurora_terrain_collider,               [],                                              I64, shared]
        [scalar,   terrain_size,                  aurora_terrain_size,                   [],                                              I64, shared]
        [scalar,   terrain_spacing,               aurora_terrain_spacing,                [],                                              F64, shared]
        [scalar,   terrain_origin_x,              aurora_terrain_origin_x,               [],                                              F64, shared]
        [scalar,   terrain_origin_z,              aurora_terrain_origin_z,               [],                                              F64, shared]

        // 3D pathfinding.
        [scalar,   nav3d_init,                    aurora_nav3d_init,                     [I64, I64, I64],                                 void, shared]
        [scalar,   nav3d_wall,                    aurora_nav3d_wall,                     [I64, I64, I64, I64],                            void, shared]
        [scalar,   nav3d_find,                    aurora_nav3d_find,                     [I64, I64, I64, I64, I64, I64],                  I64, shared]
        [scalar,   nav3d_x,                       aurora_nav3d_x,                        [I64],                                           I64, shared]
        [scalar,   nav3d_y,                       aurora_nav3d_y,                        [I64],                                           I64, shared]
        [scalar,   nav3d_z,                       aurora_nav3d_z,                        [I64],                                           I64, shared]
        [special, navmesh_build,                 aurora_navmesh_build,                  [Arr, Arr],                                      I64, shared]
        [scalar,   navmesh_find,                  aurora_navmesh_find,                   [F64, F64, F64, F64, F64, F64],                  I64, shared]
        [scalar,   navmesh_x,                     aurora_navmesh_x,                      [I64],                                           F64, shared]
        [scalar,   navmesh_y,                     aurora_navmesh_y,                      [I64],                                           F64, shared]
        [scalar,   navmesh_z,                     aurora_navmesh_z,                      [I64],                                           F64, shared]

        // 3D rendering.
        [special, r3d_load_model,                aurora_r3d_load_model,                 [Str],                                           I64, owner]
        [special, r3d_load_character,            aurora_r3d_load_character,             [Str],                                           I64, owner]
        [special, r3d_load_part,                 aurora_r3d_load_part,                  [Str, I64],                                      I64, owner]
        [special, r3d_clip_rig,                  aurora_r3d_clip_rig,                   [Str],                                           void, owner]
        [special, r3d_clip_add,                  aurora_r3d_clip_add,                   [Str],                                           void, owner]
        [text, r3d_part_add,                  aurora_r3d_part_add,                   [Str],                                           void, owner]
        [scalar,   r3d_load_assembly,             aurora_r3d_load_assembly,              [],                                              I64, owner]
        [special, r3d_clip_root,                 aurora_r3d_clip_root,                  [Str],                                           void, owner]
        [special, r3d_bone_map,                  aurora_r3d_bone_map,                   [Str, Str],                                      void, owner]
        [special, r3d_material_texture,          aurora_r3d_material_texture,           [Str, Str],                                      void, owner]
        [scalar,   r3d_free_model,                aurora_r3d_free_model,                 [I64],                                           I64, owner]
        [scalar,   r3d_model_extent,              aurora_r3d_model_extent,               [I64, I64],                                      F64, owner]
        [scalar,   r3d_model_centre,              aurora_r3d_model_centre,               [I64, I64],                                      F64, owner]
        [scalar,   r3d_make_box,                  aurora_r3d_make_box,                   [F64, F64, F64],                                 I64, owner]
        [scalar,   r3d_make_box_sized,            aurora_r3d_make_box_sized,             [F64, F64, F64, F64, F64, F64],                  I64, owner]
        [scalar,   r3d_make_box_emissive,         aurora_r3d_make_box_emissive,          [F64, F64, F64, F64, F64, F64],                  I64, owner]
        [scalar,   r3d_make_sphere,               aurora_r3d_make_sphere,                [I64, F64, F64, F64],                            I64, owner]
        [scalar,   r3d_make_plane,                aurora_r3d_make_plane,                 [F64, F64, F64, F64, F64],                       I64, owner]
        [scalar,   r3d_camera,                    aurora_r3d_camera,                     [F64, F64, F64, F64, F64, F64, F64],             void, owner]
        [scalar,   r3d_camera_roll,               aurora_r3d_camera_roll,                [F64],                                           void, owner]
        [scalar,   r3d_light,                     aurora_r3d_light,                      [F64, F64, F64, F64, F64, F64, F64],             void, owner]
        [scalar,   r3d_clear,                     aurora_r3d_clear,                      [F64, F64, F64],                                 void, owner]
        [scalar,   r3d_begin,                     aurora_r3d_begin,                      [],                                              void, owner]
        [scalar,   r3d_draw,                      aurora_r3d_draw,                       [I64, F64, F64, F64, F64, F64, F64, F64],        void, owner]
        [scalar,   r3d_draw_quat,                 aurora_r3d_draw_quat,                  [I64, F64, F64, F64, F64, F64, F64, F64, F64],   void, owner]
        [scalar,   r3d_draw_scaled,               aurora_r3d_draw_scaled,                [I64, F64, F64, F64, F64, F64, F64, F64, F64, F64], void, owner]
        [scalar,   r3d_draw_tint,                 aurora_r3d_draw_tint,                  [I64, F64, F64, F64, F64, F64, F64, F64, F64, F64, F64], void, owner]
        [scalar,   r3d_draw_shield,               aurora_r3d_draw_shield,                [I64, F64, F64, F64, F64, F64, F64, F64, F64, F64], void, owner]
        [scalar,   r3d_draw_on_joint,             aurora_r3d_draw_on_joint,              [I64, I64, I64, F64, F64, F64, F64, F64, F64, F64, F64, F64, F64, F64, F64, F64, F64], void, owner]
        [scalar,   r3d_draw_skinned,              aurora_r3d_draw_skinned,               [I64, I64, F64, F64, F64, F64, F64, F64, F64], void, owner]
        [scalar,   r3d_joint_dump,                aurora_r3d_joint_dump,                 [I64],                                           void, owner]
        [scalar,   r3d_joint_pos,                 aurora_r3d_joint_pos,                  [I64, I64, I64],                                 F64, owner]
        [scalar,   r3d_joint_world,               aurora_r3d_joint_world,                [I64, I64, I64, F64, F64, F64, F64, F64],        F64, owner]
        [scalar,   r3d_yaw_rotate,                aurora_r3d_yaw_rotate,                 [F64, F64, F64, I64],                            F64, shared]
        [scalar,   r3d_joint_basis,               aurora_r3d_joint_basis,                [I64, I64, I64, I64, F64],                       F64, owner]
        [scalar,   r3d_anim_play,                 aurora_r3d_anim_play,                  [I64, I64, I64, F64, F64],                       void, owner]
        [scalar,   r3d_anim_restart,              aurora_r3d_anim_restart,               [I64, I64, I64, F64, F64],                       void, owner]
        [scalar,   r3d_anim_update,               aurora_r3d_anim_update,                [I64, F64],                                      void, owner]
        [scalar,   r3d_anim_play_upper,           aurora_r3d_anim_play_upper,            [I64, I64, I64, F64, F64, I64],                  void, owner]
        [scalar,   r3d_anim_aim_upper,            aurora_r3d_anim_aim_upper,             [I64, I64, I64, F64, F64, F64, I64],             void, owner]
        [scalar,   r3d_anim_blend,                aurora_r3d_anim_blend,                 [I64, I64, I64, F64, F64, F64],                  void, owner]
        [scalar,   r3d_anim_seek,                 aurora_r3d_anim_seek,                  [I64, F64],                                      void, owner]
        [scalar,   r3d_anim_seek_upper,           aurora_r3d_anim_seek_upper,            [I64, F64],                                      void, owner]
        [scalar,   r3d_pose_bone,                 aurora_r3d_pose_bone,                  [I64, I64, F64, F64, F64],                       void, owner]
        [scalar,   r3d_hide_joint,                aurora_r3d_hide_joint,                 [I64, I64],                                      void, owner]
        [scalar,   r3d_clear_pose,                aurora_r3d_clear_pose,                 [I64],                                           void, owner]
        [scalar,   r3d_anim_stop_upper,           aurora_r3d_anim_stop_upper,            [I64, F64],                                      void, owner]
        [scalar,   r3d_clip_count,                aurora_r3d_clip_count,                 [I64],                                           I64, owner]
        [scalar,   r3d_material_count,            aurora_r3d_material_count,             [I64],                                           I64, owner]
        [text,     r3d_material_name,             aurora_r3d_material_name,              [I64, I64],                                      Str, owner]
        [scalar,   r3d_clip_duration,             aurora_r3d_clip_duration,              [I64, I64],                                      F64, owner]
        [scalar,   r3d_anim_done,                 aurora_r3d_anim_done,                  [I64],                                           I64, owner]
        [scalar,   r3d_anim_done_upper,           aurora_r3d_anim_done_upper,            [I64],                                           I64, owner]
        [scalar,   r3d_anim_time,                 aurora_r3d_anim_time,                  [I64],                                           F64, owner]
        [scalar,   r3d_anim_speed,                aurora_r3d_anim_speed,                 [I64],                                           F64, owner]
        [scalar,   r3d_root_dx,                   aurora_r3d_root_dx,                    [I64],                                           F64, owner]
        [scalar,   r3d_root_dy,                   aurora_r3d_root_dy,                    [I64],                                           F64, owner]
        [scalar,   r3d_root_dz,                   aurora_r3d_root_dz,                    [I64],                                           F64, owner]
        [scalar,   r3d_anim_clip,                 aurora_r3d_anim_clip,                  [I64],                                           I64, owner]
        [scalar,   r3d_anim_clip_upper,           aurora_r3d_anim_clip_upper,            [I64],                                           I64, owner]
        [scalar,   r3d_anim_blend_clip,           aurora_r3d_anim_blend_clip,            [I64],                                           I64, owner]
        [scalar,   r3d_mesh_bytes,                aurora_r3d_mesh_bytes,                 [],                                              I64, owner]
        [scalar,   r3d_mesh_count,                aurora_r3d_mesh_count,                 [],                                              I64, owner]
        [scalar,   r3d_mesh_slots,                aurora_r3d_mesh_slots,                 [],                                              I64, owner]
        [scalar,   r3d_texture_bytes,             aurora_r3d_texture_bytes,              [],                                              I64, owner]
        [scalar,   r3d_texture_count,             aurora_r3d_texture_count,              [],                                              I64, owner]
        [scalar,   r3d_anim_blend_weight,         aurora_r3d_anim_blend_weight,          [I64],                                           F64, owner]
        [text,     r3d_clip_name,                 aurora_r3d_clip_name,                  [I64, I64],                                      Str, owner]
        [text, r3d_clip_index,                aurora_r3d_clip_index,                 [I64, Str],                                      I64, owner]
        [text, r3d_joint_index,               aurora_r3d_joint_index,                [I64, Str],                                      I64, owner]
        [text,     r3d_joint_name,                aurora_r3d_joint_name,                 [I64, I64],                                      Str, owner]
        [scalar,   r3d_show_joints,               aurora_r3d_show_joints,                [I64],                                           void, owner]
        [scalar,   r3d_present,                   aurora_r3d_present,                    [],                                              I64, owner]
        [scalar,   r3d_fog,                       aurora_r3d_fog,                        [F64, F64, F64, F64],                            void, owner]
        [scalar,   r3d_speedlines,                aurora_r3d_speedlines,                 [F64, F64],                                      void, owner]
        [scalar,   r3d_damage,                    aurora_r3d_damage,                     [F64, F64, F64, F64, F64],                       void, owner]
        [scalar,   r3d_blur,                      aurora_r3d_blur,                       [F64],                                           void, owner]
        [scalar,   r3d_sky,                       aurora_r3d_sky,                        [I64, F64, F64, F64, F64, F64, F64],             void, owner]
        [scalar,   r3d_shadows,                   aurora_r3d_shadows,                    [I64],                                           void, owner]
        [scalar,   r3d_ssao,                      aurora_r3d_ssao,                       [I64],                                           void, owner]
        [scalar,   r3d_viewmodel,                 aurora_r3d_viewmodel,                  [I64],                                           void, owner]
        [scalar,   r3d_point_shadows,             aurora_r3d_point_shadows,              [I64],                                           void, owner]
        [scalar,   r3d_clear_lights,              aurora_r3d_clear_lights,               [],                                              void, owner]
        [scalar,   r3d_point_light,               aurora_r3d_point_light,                [F64, F64, F64, F64, F64, F64, F64, F64],        void, owner]
        [scalar,   r3d_make_sprite,               aurora_r3d_make_sprite,                [F64, F64, F64],                                 I64, owner]
        [scalar,   r3d_draw_billboard,            aurora_r3d_draw_billboard,             [I64, F64, F64, F64, F64],                       void, owner]
        [scalar,   r3d_debug_line,                aurora_r3d_debug_line,                 [F64, F64, F64, F64, F64, F64, F64, F64, F64],   void, owner]
        [scalar,   r3d_debug_skeleton,            aurora_r3d_debug_skeleton,             [I64, F64, F64, F64, F64, F64, F64, F64, F64],   void, owner]
        [scalar,   r3d_frustum_cull,              aurora_r3d_frustum_cull,               [I64],                                           void, owner]
        [scalar,   r3d_screen_x,                  aurora_r3d_screen_x,                   [F64, F64, F64],                                 F64, owner]
        [scalar,   r3d_screen_y,                  aurora_r3d_screen_y,                   [F64, F64, F64],                                 F64, owner]
        [scalar,   mouse_dx,                      aurora_mouse_dx,                       [],                                              F64, owner]
        [scalar,   mouse_dy,                      aurora_mouse_dy,                       [],                                              F64, owner]
        [scalar,   mouse_scroll,                  aurora_mouse_scroll,                   [],                                              F64, owner]
        [scalar,   mouse_button,                  aurora_mouse_button,                   [I64],                                           I64, owner]
        [scalar,   grab_mouse,                    aurora_grab_mouse,                     [I64],                                           void, owner]
        [scalar,   audio_listener,                aurora_audio_listener,                 [F64, F64, F64, F64, F64, F64],                  void, owner]
        [scalar,   play_sound_at,                 aurora_play_sound_at,                  [I64, I64, I64, F64, F64, F64],                  void, owner]
        [scalar,   play_sound_handle,             aurora_play_sound_handle,              [I64, I64],                                      void, owner]
        [scalar,   play_sound_handle_at,          aurora_play_sound_handle_at,           [I64, I64, F64, F64, F64],                       void, owner]
        [scalar,   audio_plays,                   aurora_audio_plays,                    [],                                              I64, owner]
        [scalar,   audio_last_sound,              aurora_audio_last_sound,               [],                                              I64, owner]
        [scalar,   music_now,                     aurora_music_now,                      [],                                              I64, owner]
        [scalar,   music_starts,                  aurora_music_starts,                   [],                                              I64, owner]
        [scalar,   play_music,                    aurora_play_music,                     [I64, I64],                                      void, shared]
        [scalar,   music_volume,                  aurora_music_volume,                   [I64],                                           void, shared]
        [scalar,   music_stop,                    aurora_music_stop,                     [],                                              void, shared]
        [scalar,   play_ambience,                 aurora_play_ambience,                  [I64, I64],                                      void, shared]
        [scalar,   ambience_volume,               aurora_ambience_volume,                [I64],                                           void, shared]
        [scalar,   ambience_stop,                 aurora_ambience_stop,                  [],                                              void, shared]
        [scalar,   phys3d_raycast_full,           aurora_phys3d_raycast_full,            [F64, F64, F64, F64, F64, F64, F64],             I64, shared]
        [scalar,   phys3d_raycast_ex,             aurora_phys3d_raycast_ex,              [I64, F64, F64, F64, F64, F64, F64, F64],        I64, shared]
        [scalar,   phys3d_raycast_world,          aurora_phys3d_raycast_world,           [I64, F64, F64, F64, F64, F64, F64, F64],        I64, shared]
        [scalar,   phys3d_hit_x,                  aurora_phys3d_hit_x,                   [],                                              F64, shared]
        [scalar,   phys3d_hit_y,                  aurora_phys3d_hit_y,                   [],                                              F64, shared]
        [scalar,   phys3d_hit_z,                  aurora_phys3d_hit_z,                   [],                                              F64, shared]
        [scalar,   phys3d_hit_nx,                 aurora_phys3d_hit_nx,                  [],                                              F64, shared]
        [scalar,   phys3d_hit_ny,                 aurora_phys3d_hit_ny,                  [],                                              F64, shared]
        [scalar,   phys3d_hit_nz,                 aurora_phys3d_hit_nz,                  [],                                              F64, shared]
        [scalar,   phys3d_hit_body,               aurora_phys3d_hit_body,                [],                                              I64, shared]
        [scalar,   phys3d_spherecast,             aurora_phys3d_spherecast,              [F64, F64, F64, F64, F64, F64, F64, F64, I64],   F64, shared]
        [scalar,   phys3d_overlap_sphere,         aurora_phys3d_overlap_sphere,          [F64, F64, F64, F64],                            I64, shared]
        [scalar,   phys3d_overlap_world,          aurora_phys3d_overlap_world,           [F64, F64, F64, F64],                            I64, shared]
        [scalar,   phys3d_debug_draw,             aurora_phys3d_debug_draw,              [F64, F64, F64],                                 void, shared]
        [scalar,   phys3d_apply_force,            aurora_phys3d_apply_force,             [I64, F64, F64, F64],                            void, shared]
        [scalar,   phys3d_apply_torque,           aurora_phys3d_apply_torque,            [I64, F64, F64, F64],                            void, shared]
        [scalar,   phys3d_set_angvel,             aurora_phys3d_set_angvel,              [I64, F64, F64, F64],                            void, shared]
        [scalar,   phys3d_set_rot,                aurora_phys3d_set_rot,                 [I64, F64, F64, F64, F64],                       void, shared]
        [scalar,   phys3d_rot_qx,                 aurora_phys3d_rot_qx,                  [I64],                                           F64, shared]
        [scalar,   phys3d_rot_qy,                 aurora_phys3d_rot_qy,                  [I64],                                           F64, shared]
        [scalar,   phys3d_rot_qz,                 aurora_phys3d_rot_qz,                  [I64],                                           F64, shared]
        [scalar,   phys3d_rot_qw,                 aurora_phys3d_rot_qw,                  [I64],                                           F64, shared]
        [scalar,   net_host,                      aurora_net_host,                       [I64],                                           I64, shared]
        [special, net_join,                      aurora_net_join,                       [Str, I64],                                      I64, shared]
        [special,  net_sim,                       aurora_net_sim,                        [Ptr, Ptr, I64, I64],                            void, shared]
        [special,  net_serve,                     aurora_net_serve,                      [Ptr, Ptr],                                      void, shared]
        [special, net_send_input,                aurora_net_send_input,                 [Arr],                                           I64, shared]
        [special, save_settings,                 aurora_save_settings,                  [Arr],                                           I64, shared]
        [special, load_settings,                 aurora_load_settings,                  [Arr],                                           I64, shared]
        [scalar,   net_update,                    aurora_net_update,                     [F64],                                           void, shared]
        [scalar,   net_leave,                     aurora_net_leave,                      [],                                              void, shared]
        [scalar,   net_my_id,                     aurora_net_my_id,                      [],                                              I64, shared]
        [scalar,   net_is_server,                 aurora_net_is_server,                  [],                                              I64, shared]
        [scalar,   net_player_count,              aurora_net_player_count,               [],                                              I64, shared]
        [scalar,   net_player_id_at,              aurora_net_player_id_at,               [I64],                                           I64, shared]
        [scalar,   net_player_x,                  aurora_net_player_x,                   [I64],                                           F64, shared]
        [scalar,   net_player_y,                  aurora_net_player_y,                   [I64],                                           F64, shared]
        [scalar,   net_player_z,                  aurora_net_player_z,                   [I64],                                           F64, shared]
        [scalar,   net_player_yaw,                aurora_net_player_yaw,                 [I64],                                           F64, shared]
        [scalar,   net_player_state,              aurora_net_player_state,               [I64, I64],                                      F64, shared]
        [scalar,   net_set_meta,                  aurora_net_set_meta,                   [I64, F64],                                      void, shared]
        [scalar,   net_player_meta,               aurora_net_player_meta,                [I64, I64],                                      F64, shared]
        [scalar,   net_player_input,              aurora_net_player_input,               [I64, I64],                                      F64, shared]
        [scalar,   net_set_local_state,           aurora_net_set_local_state,            [I64, F64],                                      void, shared]
        [scalar,   net_owned_movement,            aurora_net_owned_movement,             [I64],                                           void, shared]
        [scalar,   net_set_world_len,             aurora_net_set_world_len,              [I64, I64],                                      void, shared]
        [scalar,   net_set_world,                 aurora_net_set_world,                  [I64, I64, F64],                                 void, shared]
        [scalar,   net_world_len,                 aurora_net_world_len,                  [I64],                                           I64, shared]
        [scalar,   net_world,                     aurora_net_world,                      [I64, I64],                                      F64, shared]
        [scalar,   net_world_gen,                 aurora_net_world_gen,                  [I64],                                           I64, shared]
        [scalar,   net_set_tag,                   aurora_net_set_tag,                    [I64],                                           void, shared]
        [scalar,   net_player_tag,                aurora_net_player_tag,                 [I64],                                           I64, shared]
        [special, net_set_name,                  aurora_net_set_name,                   [Str],                                           void, shared]
        [scalar,   net_player_name_len,           aurora_net_player_name_len,            [I64],                                           I64, shared]
        [scalar,   net_player_name_char,          aurora_net_player_name_char,           [I64, I64],                                      I64, shared]
        [scalar,   net_local_x,                   aurora_net_local_x,                    [],                                              F64, shared]
        [scalar,   net_local_y,                   aurora_net_local_y,                    [],                                              F64, shared]
        [scalar,   net_local_z,                   aurora_net_local_z,                    [],                                              F64, shared]
        [scalar,   net_local_yaw,                 aurora_net_local_yaw,                  [],                                              F64, shared]
        [scalar,   net_state,                     aurora_net_state,                      [I64, I64],                                      F64, shared]
        [scalar,   net_local_state,               aurora_net_local_state,                [I64],                                           F64, shared]
        [scalar,   net_interest,                  aurora_net_interest,                   [F64],                                           void, shared]
        [scalar,   net_max_clients,               aurora_net_max_clients,                [I64],                                           void, shared]
        [scalar,   net_rejected,                  aurora_net_rejected,                   [],                                              I64, shared]
        [scalar,   net_connected,                 aurora_net_connected,                  [],                                              I64, shared]
        [scalar,   net_dedicated,                 aurora_net_dedicated,                  [],                                              void, shared]
        [scalar,   net_cfg_set,                   aurora_net_cfg_set,                    [I64, F64],                                      void, shared]
        [scalar,   net_cfg_get,                   aurora_net_cfg_get,                    [I64],                                           F64, shared]
        [scalar,   net_set_bot_count,             aurora_net_set_bot_count,              [I64],                                           void, shared]
        [scalar,   net_set_bot,                   aurora_net_set_bot,                    [I64, F64, F64, F64, F64],                       void, shared]
        [scalar,   net_set_bot_input,             aurora_net_set_bot_input,              [I64, I64],                                      void, shared]
        [scalar,   net_set_bot_state,             aurora_net_set_bot_state,              [I64, I64],                                      void, shared]
        [scalar,   net_set_bot_alive,             aurora_net_set_bot_alive,              [I64, I64],                                      void, shared]
        [scalar,   net_set_bot_meta,              aurora_net_set_bot_meta,               [I64, I64, F64],                                 void, shared]
        [special, net_set_bot_name,              aurora_net_set_bot_name,               [I64, Str],                                      void, shared]
        [scalar,   net_bot_count,                 aurora_net_bot_count,                  [],                                              I64, shared]
        [scalar,   net_set_object_count,          aurora_net_set_object_count,           [I64],                                           void, shared]
        [scalar,   net_set_object,                aurora_net_set_object,                 [I64, F64, F64, F64],                            void, shared]
        [scalar,   net_object_count,              aurora_net_object_count,               [],                                              I64, shared]
        [scalar,   net_object_x,                  aurora_net_object_x,                   [I64],                                           F64, shared]
        [scalar,   net_object_y,                  aurora_net_object_y,                   [I64],                                           F64, shared]
        [scalar,   net_object_z,                  aurora_net_object_z,                   [I64],                                           F64, shared]
        [scalar,   net_set_object_rot,            aurora_net_set_object_rot,             [I64, F64, F64, F64, F64],                       void, shared]
        [scalar,   net_object_qx,                 aurora_net_object_qx,                  [I64],                                           F64, shared]
        [scalar,   net_object_qy,                 aurora_net_object_qy,                  [I64],                                           F64, shared]
        [scalar,   net_object_qz,                 aurora_net_object_qz,                  [I64],                                           F64, shared]
        [scalar,   net_object_qw,                 aurora_net_object_qw,                  [I64],                                           F64, shared]
        [scalar,   net_set_object_vel,            aurora_net_set_object_vel,             [I64, F64, F64, F64],                            void, shared]
        [scalar,   net_set_object_size,           aurora_net_set_object_size,            [I64, F64, F64],                                 void, shared]
        [scalar,   net_set_object_state,          aurora_net_set_object_state,           [I64, I64, F64],                                 void, shared]
        [scalar,   net_object_state,              aurora_net_object_state,               [I64, I64],                                      F64, shared]
        [scalar,   net_object_vx,                 aurora_net_object_vx,                  [I64],                                           F64, shared]
        [scalar,   net_object_vy,                 aurora_net_object_vy,                  [I64],                                           F64, shared]
        [scalar,   net_object_vz,                 aurora_net_object_vz,                  [I64],                                           F64, shared]
        [scalar,   net_set_fx_count,              aurora_net_set_fx_count,               [I64],                                           void, shared]
        [scalar,   net_set_fx,                    aurora_net_set_fx,                     [I64, F64, F64, F64, F64],                       void, shared]
        [scalar,   net_fx_count,                  aurora_net_fx_count,                   [],                                              I64, shared]
        [scalar,   net_fx_x,                      aurora_net_fx_x,                       [I64],                                           F64, shared]
        [scalar,   net_fx_y,                      aurora_net_fx_y,                       [I64],                                           F64, shared]
        [scalar,   net_fx_z,                      aurora_net_fx_z,                       [I64],                                           F64, shared]
        [scalar,   net_fx_kind,                   aurora_net_fx_kind,                    [I64],                                           F64, shared]
        [scalar,   net_hit_radius,                aurora_net_hit_radius,                 [F64],                                           void, shared]
        [scalar,   net_spawn_at,                  aurora_net_spawn_at,                   [F64, F64, F64],                                 void, shared]
        [scalar,   net_spawn_input_slot,          aurora_net_spawn_input_slot,           [I64],                                           void, shared]
        [scalar,   net_respawn_client,            aurora_net_respawn_client,             [I64, F64, F64, F64],                            void, shared]
        [scalar,   net_impulse_input_slot,        aurora_net_impulse_input_slot,         [I64],                                           void, shared]
        [scalar,   net_push_impulse,              aurora_net_push_impulse,               [I64, F64, F64, F64],                            void, shared]
        [scalar,   net_respawn_trigger_slot,      aurora_net_respawn_trigger_slot,       [I64],                                           void, shared]
        [scalar,   net_force_respawn,             aurora_net_force_respawn,              [I64],                                           void, shared]
        [scalar,   net_fire,                      aurora_net_fire,                       [F64, F64, F64, F64, F64, F64, I64],             void, shared]
        [scalar,   net_melee,                     aurora_net_melee,                      [F64, F64, F64, F64, F64, F64, F64, F64, I64],   void, shared]
        [scalar,   net_server_hit_count,          aurora_net_server_hit_count,           [],                                              I64, shared]
        [scalar,   net_server_hit_shooter,        aurora_net_server_hit_shooter,         [I64],                                           I64, shared]
        [scalar,   net_server_hit_victim,         aurora_net_server_hit_victim,          [I64],                                           I64, shared]
        [scalar,   net_server_hit_weapon,         aurora_net_server_hit_weapon,          [I64],                                           I64, shared]
        [scalar,   net_server_hit_x,              aurora_net_server_hit_x,               [I64],                                           F64, shared]
        [scalar,   net_server_hit_y,              aurora_net_server_hit_y,               [I64],                                           F64, shared]
        [scalar,   net_server_hit_z,              aurora_net_server_hit_z,               [I64],                                           F64, shared]
        [scalar,   net_server_hits_clear,         aurora_net_server_hits_clear,          [],                                              void, shared]
        [scalar,   net_push_kill,                 aurora_net_push_kill,                  [I64, I64],                                      void, shared]
        [scalar,   net_kill_count,                aurora_net_kill_count,                 [],                                              I64, shared]
        [scalar,   net_kill_killer,               aurora_net_kill_killer,                [I64],                                           I64, shared]
        [scalar,   net_kill_victim,               aurora_net_kill_victim,                [I64],                                           I64, shared]
        [scalar,   net_kills_clear,               aurora_net_kills_clear,                [],                                              void, shared]
        [scalar,   net_push_shot,                 aurora_net_push_shot,                  [I64, F64, F64, F64, F64, F64, F64, I64],        void, shared]
        [scalar,   net_shot_count,                aurora_net_shot_count,                 [],                                              I64, shared]
        [scalar,   net_shot_shooter,              aurora_net_shot_shooter,               [I64],                                           I64, shared]
        [scalar,   net_shot_field,                aurora_net_shot_field,                 [I64, I64],                                      F64, shared]
        [scalar,   net_shot_weapon,               aurora_net_shot_weapon,                [I64],                                           I64, shared]
        [scalar,   net_shots_clear,               aurora_net_shots_clear,                [],                                              void, shared]
        [scalar,   net_push_boom,                 aurora_net_push_boom,                  [I64, F64, F64, F64, F64],                       void, shared]
        [scalar,   net_boom_count,                aurora_net_boom_count,                 [],                                              I64, shared]
        [scalar,   net_boom_source,               aurora_net_boom_source,                [I64],                                           I64, shared]
        [scalar,   net_boom_field,                aurora_net_boom_field,                 [I64, I64],                                      F64, shared]
        [scalar,   net_booms_clear,               aurora_net_booms_clear,                [],                                              void, shared]
        [scalar,   net_projectile_intent,         aurora_net_projectile_intent,          [I64, F64, F64, F64, F64, F64, F64],             void, shared]
        [scalar,   net_server_projectile_count,   aurora_net_server_projectile_count,    [],                                              I64, shared]
        [scalar,   net_server_projectile_shooter, aurora_net_server_projectile_shooter,  [I64],                                           I64, shared]
        [scalar,   net_server_projectile_kind,    aurora_net_server_projectile_kind,     [I64],                                           I64, shared]
        [scalar,   net_server_projectile_ox,      aurora_net_server_projectile_ox,       [I64],                                           F64, shared]
        [scalar,   net_server_projectile_oy,      aurora_net_server_projectile_oy,       [I64],                                           F64, shared]
        [scalar,   net_server_projectile_oz,      aurora_net_server_projectile_oz,       [I64],                                           F64, shared]
        [scalar,   net_server_projectile_vx,      aurora_net_server_projectile_vx,       [I64],                                           F64, shared]
        [scalar,   net_server_projectile_vy,      aurora_net_server_projectile_vy,       [I64],                                           F64, shared]
        [scalar,   net_server_projectile_vz,      aurora_net_server_projectile_vz,       [I64],                                           F64, shared]
        [scalar,   net_server_projectiles_clear,  aurora_net_server_projectiles_clear,   [],                                              void, shared]
        [scalar,   net_set_player_meta,           aurora_net_set_player_meta,            [I64, I64, F64],                                 void, shared]
        [scalar,   net_hit_player,                aurora_net_hit_player,                 [],                                              I64, shared]
        [scalar,   net_hit_seq,                   aurora_net_hit_seq,                    [],                                              I64, shared]
        [scalar,   net_hit_x,                     aurora_net_hit_x,                      [],                                              F64, shared]
        [scalar,   net_hit_y,                     aurora_net_hit_y,                      [],                                              F64, shared]
        [scalar,   net_hit_z,                     aurora_net_hit_z,                      [],                                              F64, shared]
        [scalar,   input_bind,                    aurora_input_bind,                     [I64, I64],                                      void, shared]
        [scalar,   input_binding,                 aurora_input_binding,                  [I64],                                           I64, shared]
        [scalar,   input_down,                    aurora_input_down,                     [I64],                                           I64, shared]
        [scalar,   input_axis,                    aurora_input_axis,                     [I64, I64],                                      F64, shared]
        [scalar,   input_suppress,                aurora_input_suppress,                 [I64],                                           void, shared]
        [scalar,   input_pressed,                 aurora_input_pressed,                  [I64],                                           I64, shared]
        [scalar,   input_released,                aurora_input_released,                 [I64],                                           I64, shared]
        [scalar,   input_step,                    aurora_input_step,                     [],                                              void, owner]
        [text, input_code,                    aurora_input_code,                     [Str],                                           I64, shared]
        [text,     input_name,                    aurora_input_name,                     [I64],                                           Str, shared]
        [scalar,   inject_action,                 aurora_inject_action,                  [I64, I64],                                      void, owner]
        [scalar,   input_bind_also,               aurora_input_bind_also,                [I64, I64],                                      void, shared]
        [scalar,   input_binding_at,              aurora_input_binding_at,               [I64, I64],                                      I64, shared]
        [scalar,   input_binding_count,           aurora_input_binding_count,            [I64],                                           I64, shared]
        [scalar,   input_value,                   aurora_input_value,                    [I64],                                           F64, shared]
        [scalar,   pad_count,                     aurora_pad_count,                      [],                                              I64, shared]
        [scalar,   pad_connected,                 aurora_pad_connected,                  [I64],                                           I64, shared]
        [scalar,   pad_button,                    aurora_pad_button,                     [I64, I64],                                      I64, shared]
        [scalar,   pad_axis,                      aurora_pad_axis,                       [I64, I64],                                      F64, shared]
        [scalar,   pad_rumble,                    aurora_pad_rumble,                     [I64, F64, F64, F64],                            I64, shared]
        [scalar,   inject_pad_button,             aurora_inject_pad_button,              [I64, I64, I64],                                 void, owner]
        [scalar,   inject_pad_axis,               aurora_inject_pad_axis,                [I64, I64, F64],                                 void, owner]
        [scalar,   inject_pad_disconnect,         aurora_inject_pad_disconnect,          [I64],                                           void, owner]
        [scalar,   inject_input,                  aurora_inject_input,                   [I64, F64],                                      void, owner]
        [scalar,   phys3d_separate_characters,    aurora_phys3d_separate_characters,     [],                                              I64, shared]
        [scalar,   phys3d_character_count,        aurora_phys3d_character_count,         [],                                              I64, shared]
        [scalar,   f32_load,                      aurora_f32_load,                       [I64, I64],                                      F64, shared]
        [scalar,   f32_store,                     aurora_f32_store,                      [I64, I64, F64],                                 void, shared]
        [scalar,   f32_blob,                      aurora_f32_blob,                       [I64],                                           I64, shared]
        [special,  sin,                           aurora_sin,                            [F64],                                           F64, shared]
        [special,  cos,                           aurora_cos,                            [F64],                                           F64, shared]
        [special,  tan,                           aurora_tan,                            [F64],                                           F64, shared]
        [special,  pow,                           aurora_pow,                            [F64, F64],                                      F64, shared]
        [special,  log,                           aurora_log,                            [F64],                                           F64, shared]
        [special,  exp,                           aurora_exp,                            [F64],                                           F64, shared]
        [special,  atan2,                         aurora_atan2,                          [F64, F64],                                      F64, shared]
        [special,  acos,                          aurora_acos,                           [F64],                                           F64, shared]
        [special,  asin,                          aurora_asin,                           [F64],                                           F64, shared]
        [special,  atan,                          aurora_atan,                           [F64],                                           F64, shared]
        [special, draw_text,                     aurora_draw_text,                      [I64, I64, Str, I64, I64],                       void, owner]
        [special,  draw_int,                      aurora_draw_int,                       [I64, I64, I64, I64, I64],                       void, shared]
        [special, text_width,                    aurora_text_width,                     [Str, I64],                                      I64, shared]
        [special, scene_save,                    aurora_scene_save,                     [Str],                                           I64, shared]
        [special, scene_load,                    aurora_scene_load,                     [Str],                                           I64, shared]
        [internal, prof_enter,                    aurora_prof_enter,                     [Str],                                           void, shared]
        [internal, prof_exit,                     aurora_prof_exit,                      [],                                              void, shared]
        [internal, str_concat,                    aurora_str_concat,                     [Ptr, Str, Str],                                 void, shared]
        [internal, str_eq,                        aurora_str_eq,                         [Str, Str],                                      I64, shared]
        [internal, str_char_at,                   aurora_str_char_at,                    [Str, I64],                                      I64, shared]
        [internal, str_substr,                    aurora_str_substr,                     [Ptr, Str, I64, I64],                            void, shared]
        [internal, str_starts_with,               aurora_str_starts_with,                [Str, Str],                                      I64, shared]
        [internal, int_to_str,                    aurora_int_to_str,                     [Ptr, I64],                                      void, shared]
        [internal, float_to_str,                  aurora_float_to_str,                   [Ptr, F64],                                      void, shared]
        [special,  play_note,                     aurora_play_note,                      [I64, I64],                                      void, owner]
        [special,  play_sound,                    aurora_play_sound,                     [I64, I64, I64],                                 void, owner]
        [special,  play_noise,                    aurora_play_noise,                     [I64, I64],                                      void, shared]
        [special,  audio_volume,                  aurora_audio_volume,                   [I64],                                           void, owner]
        [special,  window_fullscreen,             aurora_window_fullscreen,              [I64],                                           void, owner]
        [special,  audio_stop,                    aurora_audio_stop,                     [],                                              void, owner]
        [special, gpu_render,                    aurora_gpu_render,                     [Str, I64],                                      void, owner]
        [special,  window_open,                   aurora_window_open,                    [I64, I64],                                      void, owner]
        [special,  window_present,                aurora_window_present,                 [],                                              I64, owner]
        [special,  surface_w,                     aurora_surface_w,                      [],                                              I64, shared]
        [special,  surface_h,                     aurora_surface_h,                      [],                                              I64, shared]
        [special,  key_down,                      aurora_key_down,                       [I64],                                           I64, owner]
        [special,  input_char,                    aurora_input_char,                     [],                                              I64, shared]
        [special,  mouse_x,                       aurora_mouse_x,                        [],                                              I64, owner]
        [special,  mouse_y,                       aurora_mouse_y,                        [],                                              I64, owner]
        [special,  mouse_down,                    aurora_mouse_down,                     [],                                              I64, owner]

        // Process environment: the program's own argument vector and env vars.
        [scalar,   sys_argc,                      aurora_sys_argc,                       [],                                              I64, shared]
        [text,     sys_arg,                       aurora_sys_arg,                        [I64],                                           Str, shared]
        [text, sys_env,                       aurora_sys_env,                        [Str],                                           Str, shared]

        // Assertions.
        [scalar,   assert,                        aurora_assert,                         [I64],                                           void, shared]

        // Native debugger hooks (only *called* when `debug`, but always importable).
        [internal, dbg_enter,                     aurora_dbg_enter,                      [Str],                                           void, shared]
        [internal, dbg_leave,                     aurora_dbg_leave,                      [],                                              void, shared]
        [internal, dbg_stmt,                      aurora_dbg_stmt,                       [I64],                                           void, shared]
        [internal, dbg_var,                       aurora_dbg_var,                        [Str, I64],                                      void, shared]
        [internal, dbg_var_f64,                   aurora_dbg_var_f64,                    [Str, F64],                                      void, shared]

        // Builtins the backend lowers inline (no runtime call): printing,
        // polymorphic math/bit ops, string ops, ECS spawn, and `run_systems`.
        [inline,   print,                         none,                                  [],                                              void, shared]
        [inline,   println,                       none,                                  [],                                              void, shared]
        [inline,   sqrt,                          none,                                  [],                                              F64, shared]
        [inline,   floor,                         none,                                  [],                                              F64, shared]
        [inline,   ceil,                          none,                                  [],                                              F64, shared]
        [inline,   round,                         none,                                  [],                                              F64, shared]
        [inline,   abs,                           none,                                  [],                                              void, shared]
        [inline,   min,                           none,                                  [],                                              void, shared]
        [inline,   max,                           none,                                  [],                                              void, shared]
        [inline,   clamp,                         none,                                  [],                                              void, shared]
        [inline,   len,                           none,                                  [],                                              I64, shared]
        [inline,   str,                           none,                                  [],                                              Str, shared]
        [inline,   spawn,                         none,                                  [],                                              void, shared]
        [inline,   run_systems,                   none,                                  [],                                              void, shared]
        [inline,   band,                          none,                                  [],                                              I64, shared]
        [inline,   bor,                           none,                                  [],                                              I64, shared]
        [inline,   bxor,                          none,                                  [],                                              I64, shared]
        [inline,   shl,                           none,                                  [],                                              I64, shared]
        [inline,   shr,                           none,                                  [],                                              I64, shared]
        [inline,   bnot,                          none,                                  [],                                              I64, shared]
        [inline,   char_at,                       none,                                  [],                                              I64, shared]
        [inline,   substr,                        none,                                  [],                                              Str, shared]
        [inline,   starts_with,                   none,                                  [],                                              Bool, shared]

        // Runtime functions that are not builtins: they only need a JIT symbol
        // and an AOT link edge, so `@extern` declarations can bind them.
        [linkonly, ffi_dot,                       aurora_ffi_dot,                        [],                                              void, shared]
        [linkonly, ffi_dotf,                      aurora_ffi_dotf,                       [],                                              void, shared]
        }
    };
}

macro_rules! kind_of {
    (scalar) => {
        $crate::Kind::Scalar
    };
    (text) => {
        $crate::Kind::Text
    };
    (special) => {
        $crate::Kind::Special
    };
    (raster) => {
        $crate::Kind::Raster
    };
    (internal) => {
        $crate::Kind::Internal
    };
    (inline) => {
        $crate::Kind::Inline
    };
    (linkonly) => {
        $crate::Kind::LinkOnly
    };
}

macro_rules! ty_of {
    (I64) => {
        $crate::Ty::I64
    };
    (F64) => {
        $crate::Ty::F64
    };
    (Ptr) => {
        $crate::Ty::Ptr
    };
    (Str) => {
        $crate::Ty::Str
    };
    (Arr) => {
        $crate::Ty::Arr
    };
    (Bool) => {
        $crate::Ty::Bool
    };
}

macro_rules! ret_of {
    (void) => {
        None
    };
    ($t:ident) => {
        Some(ty_of!($t))
    };
}

// An `inline` row has no runtime function, so its symbol column is the literal
// `none` and it reports no symbol at all.
macro_rules! sym_str {
    (inline, none) => {
        ""
    };
    ($kind:ident, $sym:ident) => {
        stringify!($sym)
    };
}

macro_rules! build_table {
    ($([$kind:ident, $name:ident, $sym:ident, [$($p:ident),*], $ret:ident, $home:ident])*) => {
        /// Every builtin, in table order.
        pub const TABLE: &[Builtin] = &[
            $(Builtin {
                name: stringify!($name),
                kind: kind_of!($kind),
                symbol: sym_str!($kind, $sym),
                params: &[$(ty_of!($p)),*],
                ret: ret_of!($ret),
                home: home_of!($home),
            }),*
        ];
    };
}

for_each_builtin!(build_table);

/// Is this builtin one only the owning thread may call?
///
/// Unknown names answer `false`: a name that is not a builtin is somebody else's
/// error to report, and answering `true` here would blame the wrong thing.
pub fn is_owner_only(name: &str) -> bool {
    TABLE
        .iter()
        .any(|b| b.name == name && b.home == Home::Owner)
}

/// Every builtin name Aurora source may call, in table order.
pub fn builtin_names() -> &'static [&'static str] {
    static NAMES: OnceLock<Vec<&'static str>> = OnceLock::new();
    NAMES.get_or_init(|| {
        TABLE
            .iter()
            .filter(|b| b.kind.is_aurora_visible())
            .map(|b| b.name)
            .collect()
    })
}

/// Is `name` a builtin Aurora source may call? O(1); the front end asks this of
/// every unresolved call, so a call to a builtin is not reported as a typo and a
/// real typo still is.
pub fn is_builtin(name: &str) -> bool {
    static SET: OnceLock<HashSet<&'static str>> = OnceLock::new();
    SET.get_or_init(|| builtin_names().iter().copied().collect())
        .contains(name)
}

/// The table row for `name`, including rows Aurora source cannot call
/// (`internal` / `linkonly`), which the backend looks up by the same key. O(1).
pub fn lookup(name: &str) -> Option<&'static Builtin> {
    static BY_NAME: OnceLock<HashMap<&'static str, &'static Builtin>> = OnceLock::new();
    BY_NAME
        .get_or_init(|| TABLE.iter().map(|b| (b.name, b)).collect())
        .get(name)
        .copied()
}

/// Parameter and return types of a builtin the backend lowers with its generic
/// scalar call-site dispatch, or `None` for anything else. Every parameter is
/// `I64` or `F64`, so the Aurora-level and ABI-level arities are the same.
pub fn scalar_sig(name: &str) -> Option<(&'static [Ty], Option<Ty>)> {
    let b = lookup(name)?;
    (b.kind == Kind::Scalar).then_some((b.params, b.ret))
}

/// The row for a builtin the backend lowers with its generic TEXT call-site
/// dispatch (a `str` argument and/or a `str` result), or `None` for anything
/// else. A `Ptr` parameter is the first slot of a `str` argument and is always
/// followed by its `I64` length.
pub fn text_row(name: &str) -> Option<&'static Builtin> {
    let b = lookup(name)?;
    (b.kind == Kind::Text).then_some(b)
}

#[cfg(test)]
mod tests;
