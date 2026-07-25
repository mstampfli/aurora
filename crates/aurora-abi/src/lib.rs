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
//! 1. write `pub extern "C" fn aurora_<name>(..)` in `aurora-runtime`;
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

use std::collections::{HashMap, HashSet};
use std::sync::OnceLock;

/// An ABI-level parameter or return type of a runtime host function.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Ty {
    I64,
    F64,
    /// The target's pointer type (`i64` on every backend Aurora supports today).
    Ptr,
    /// An Aurora `str` RESULT: the caller allocates its two slots and passes
    /// their address as a leading [`Ty::Ptr`] parameter, so the host function
    /// itself returns nothing. Only ever a return type, only on a `text` row.
    Str,
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
        matches!(self, Kind::Scalar | Kind::Text | Kind::Special | Kind::Inline)
    }

    /// Is this row backed by an `aurora_*` function in `aurora-runtime`? Those
    /// rows get a JIT symbol registration and an AOT link edge.
    pub const fn has_host_fn(self) -> bool {
        !matches!(self, Kind::Inline)
    }

    /// Is this row declared as an import in the backend's host table?
    pub const fn is_host_import(self) -> bool {
        matches!(self, Kind::Scalar | Kind::Text | Kind::Special | Kind::Internal)
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
}

impl Builtin {
    /// The parameter types the host function is actually declared with: a
    /// [`Ty::Str`] result is a caller-allocated 2-slot out-pointer passed FIRST,
    /// so it is not a Cranelift return value at all.
    pub fn abi_params(&self) -> Vec<Ty> {
        match self.ret {
            Some(Ty::Str) => std::iter::once(Ty::Ptr).chain(self.params.iter().copied()).collect(),
            _ => self.params.to_vec(),
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
    /// argument takes two ABI slots (`Ptr, I64`) but is one argument in the
    /// source. `None` for the kinds it does not model - `special` (arrays,
    /// closures), `inline`, and the rows Aurora cannot call at all.
    pub fn arity(&self) -> Option<usize> {
        match self.kind {
            Kind::Scalar | Kind::Text => {
                Some(self.params.len() - self.params.iter().filter(|t| **t == Ty::Ptr).count())
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
///     ($([$kind:ident, $name:ident, $sym:ident, [$($p:ident),*], $ret:ident])*) => {
///         &[$(stringify!($name)),*]
///     };
/// }
/// let all: &[&str] = aurora_abi::for_each_builtin!(names);
/// ```
#[macro_export]
macro_rules! for_each_builtin {
    ($m:ident) => {
        $m! {
        [internal, print_i64,                     aurora_print_i64,                      [I64],                                           void]
        [internal, print_f64,                     aurora_print_f64,                      [F64],                                           void]
        [internal, print_str,                     aurora_print_str,                      [Ptr, I64],                                      void]
        [internal, print_nl,                      aurora_print_nl,                       [],                                              void]
        [special,  framebuffer,                   aurora_framebuffer,                    [I64, I64],                                      void]
        [special,  clear,                         aurora_clear,                          [I64, I64, I64],                                 void]
        [special,  pixel,                         aurora_pixel,                          [I64, I64, I64, I64, I64],                       void]
        [special,  triangle,                      aurora_triangle,                       [I64, I64, I64, I64, I64, I64, I64, I64, I64],   void]
        [special,  fb_get,                        aurora_fb_get,                         [I64, I64],                                      I64]
        [special,  save_ppm,                      aurora_save_ppm,                       [Ptr, I64],                                      void]
        [internal, spawn_entity,                  aurora_spawn_entity,                   [],                                              I64]
        [special,  despawn,                       aurora_despawn,                        [I64],                                           void]
        [internal, store_component,               aurora_store_component,                [I64, I64, Ptr, I64],                            void]
        [internal, get_component,                 aurora_get_component,                  [I64, I64],                                      Ptr]
        [internal, query_begin,                   aurora_query_begin,                    [Ptr, I64],                                      I64]
        [internal, query_entity,                  aurora_query_entity,                   [I64],                                           I64]
        [special,  entity_count,                  aurora_entity_count,                   [],                                              I64]

        // Audio + windowing builtins.
        [special,  gpu_compute,                   aurora_gpu_compute,                    [Ptr, I64, Ptr, I64],                            void]
        [special,  par_for,                       aurora_par_for,                        [Ptr, I64, Ptr, Ptr],                            void]
        [internal, run_parallel,                  aurora_run_parallel,                   [Ptr, I64],                                      void]
        [special,  net_bind,                      aurora_net_bind,                       [I64],                                           I64]
        [special,  net_connect,                   aurora_net_connect,                    [Ptr, I64],                                      I64]
        [special,  net_send,                      aurora_net_send,                       [Ptr, I64],                                      I64]
        [special,  net_recv,                      aurora_net_recv,                       [Ptr],                                           void]
        [special,  frame_reset,                   aurora_frame_reset,                    [],                                              void]
        [special,  load_ppm,                      aurora_load_ppm,                       [Ptr, I64],                                      I64]

        // Determinism + data builtins.
        [scalar,   srand,                         aurora_srand,                          [I64],                                           void]
        [scalar,   rand,                          aurora_rand,                           [],                                              F64]
        [scalar,   rand_range,                    aurora_rand_range,                     [F64, F64],                                      F64]
        [scalar,   rand_int,                      aurora_rand_int,                       [I64, I64],                                      I64]
        [scalar,   set_fixed_dt,                  aurora_set_fixed_dt,                   [F64],                                           void]
        [special,  save_png,                      aurora_save_png,                       [Ptr, I64],                                      void]
        [special,  read_file,                     aurora_read_file,                      [Ptr, Ptr, I64],                                 void]
        [special,  write_file,                    aurora_write_file,                     [Ptr, I64, Ptr, I64],                            I64]
        [special,  file_exists,                   aurora_file_exists,                    [Ptr, I64],                                      I64]
        [special,  json_parse,                    aurora_json_parse,                     [Ptr, I64],                                      I64]
        [special,  json_load,                     aurora_json_load,                      [Ptr, I64],                                      I64]
        [special,  json_get,                      aurora_json_get,                       [I64, Ptr, I64],                                 I64]
        [scalar,   json_at,                       aurora_json_at,                        [I64, I64],                                      I64]
        [scalar,   json_len,                      aurora_json_len,                       [I64],                                           I64]
        [scalar,   json_num,                      aurora_json_num,                       [I64],                                           F64]
        [scalar,   json_int,                      aurora_json_int,                       [I64],                                           I64]
        [scalar,   json_bool,                     aurora_json_bool,                      [I64],                                           I64]
        [special,  json_str,                      aurora_json_str,                       [Ptr, I64],                                      void]
        [scalar,   json_kind,                     aurora_json_kind,                      [I64],                                           I64]
        [special,  json_has,                      aurora_json_has,                       [I64, Ptr, I64],                                 I64]
        [special,  json_key,                      aurora_json_key,                       [Ptr, I64, I64],                                 void]
        [scalar,   json_free,                     aurora_json_free,                      [I64],                                           void]
        [scalar,   json_new_obj,                  aurora_json_new_obj,                   [],                                              I64]
        [scalar,   json_new_arr,                  aurora_json_new_arr,                   [],                                              I64]
        [special,  json_set,                      aurora_json_set,                       [I64, Ptr, I64, I64],                            void]
        [special,  json_set_num,                  aurora_json_set_num,                   [I64, Ptr, I64, F64],                            void]
        [special,  json_set_str,                  aurora_json_set_str,                   [I64, Ptr, I64, Ptr, I64],                       void]
        [special,  json_set_bool,                 aurora_json_set_bool,                  [I64, Ptr, I64, I64],                            void]
        [scalar,   json_push,                     aurora_json_push,                      [I64, I64],                                      void]
        [scalar,   json_push_num,                 aurora_json_push_num,                  [I64, F64],                                      void]
        [special,  json_push_str,                 aurora_json_push_str,                  [I64, Ptr, I64],                                 void]
        [special,  json_to_str,                   aurora_json_to_str,                    [Ptr, I64],                                      void]
        [special,  json_write,                    aurora_json_write,                     [I64, Ptr, I64],                                 I64]
        [special,  audio_capture_save,            aurora_audio_capture_save,             [Ptr, I64],                                      I64]
        [special,  r3d_capture,                   aurora_r3d_capture,                    [Ptr, I64],                                      I64]
        [special,  r3d_capture_size,              aurora_r3d_capture_size,               [Ptr, I64, I64, I64],                            I64]
        [scalar,   inject_key,                    aurora_inject_key,                     [I64, I64],                                      void]
        [scalar,   inject_mouse_move,             aurora_inject_mouse_move,              [F64, F64],                                      void]
        [scalar,   inject_mouse_pos,              aurora_inject_mouse_pos,               [I64, I64],                                      void]
        [scalar,   inject_mouse_button,           aurora_inject_mouse_button,            [I64, I64],                                      void]
        [scalar,   inject_scroll,                 aurora_inject_scroll,                  [F64],                                           void]
        [scalar,   inject_char,                   aurora_inject_char,                    [I64],                                           void]
        [internal, oob,                           aurora_oob,                            [I64, I64],                                      void]
        [scalar,   frame_dt,                      aurora_frame_dt,                       [],                                              F64]
        [scalar,   sleep_ms,                      aurora_sleep_ms,                       [I64],                                           void]
        [internal, divzero,                       aurora_divzero,                        [],                                              void]
        [internal, fmod,                          aurora_fmod,                           [F64, F64],                                      F64]
        [special,  load_image,                    aurora_load_image,                     [Ptr, I64],                                      I64]
        [special,  load_font,                     aurora_load_font,                      [Ptr, I64],                                      I64]
        [special,  play_wav,                      aurora_play_wav,                       [Ptr, I64],                                      I64]
        [special,  load_sound,                    aurora_load_sound,                     [Ptr, I64],                                      I64]
        [scalar,   phys_init,                     aurora_phys_init,                      [F64, F64],                                      void]
        [scalar,   phys_add,                      aurora_phys_add,                       [F64, F64, F64, F64, I64],                       I64]
        [scalar,   phys_step,                     aurora_phys_step,                      [F64],                                           void]
        [scalar,   phys_x,                        aurora_phys_x,                         [I64],                                           F64]
        [scalar,   phys_y,                        aurora_phys_y,                         [I64],                                           F64]
        [scalar,   phys_set_vel,                  aurora_phys_set_vel,                   [I64, F64, F64],                                 void]
        [scalar,   phys_vel_x,                    aurora_phys_vel_x,                     [I64],                                           F64]
        [scalar,   phys_vel_y,                    aurora_phys_vel_y,                     [I64],                                           F64]
        [scalar,   phys_apply_impulse,            aurora_phys_apply_impulse,             [I64, F64, F64],                                 void]
        [scalar,   phys_apply_force,              aurora_phys_apply_force,               [I64, F64, F64],                                 void]
        [scalar,   phys_set_pos,                  aurora_phys_set_pos,                   [I64, F64, F64],                                 void]
        [scalar,   phys_raycast,                  aurora_phys_raycast,                   [F64, F64, F64, F64, F64],                       F64]
        [scalar,   nav_init,                      aurora_nav_init,                       [I64, I64],                                      void]
        [scalar,   nav_wall,                      aurora_nav_wall,                       [I64, I64, I64],                                 void]
        [scalar,   nav_find,                      aurora_nav_find,                       [I64, I64, I64, I64],                            I64]
        [scalar,   nav_x,                         aurora_nav_x,                          [I64],                                           I64]
        [scalar,   nav_y,                         aurora_nav_y,                          [I64],                                           I64]

        // 3D physics (Rapier 3D).
        [scalar,   phys3d_init,                   aurora_phys3d_init,                    [F64, F64, F64],                                 void]
        [scalar,   phys3d_add_box,                aurora_phys3d_add_box,                 [F64, F64, F64, F64, F64, F64, I64],             I64]
        [scalar,   phys3d_add_box_rot,            aurora_phys3d_add_box_rot,             [F64, F64, F64, F64, F64, F64, F64, F64, F64, I64], I64]
        [scalar,   phys3d_add_sphere,             aurora_phys3d_add_sphere,              [F64, F64, F64, F64, I64],                       I64]
        [scalar,   phys3d_add_capsule,            aurora_phys3d_add_capsule,             [F64, F64, F64, F64, F64, I64],                  I64]
        [scalar,   phys3d_add_character,          aurora_phys3d_add_character,           [F64, F64, F64, F64, F64],                       I64]
        [special,  phys3d_add_trimesh,            aurora_phys3d_add_trimesh,             [Ptr, I64, Ptr, I64],                            I64]
        [scalar,   phys3d_step,                   aurora_phys3d_step,                    [F64],                                           void]
        [scalar,   phys3d_x,                      aurora_phys3d_x,                       [I64],                                           F64]
        [scalar,   phys3d_y,                      aurora_phys3d_y,                       [I64],                                           F64]
        [scalar,   phys3d_z,                      aurora_phys3d_z,                       [I64],                                           F64]
        [scalar,   phys3d_vel_x,                  aurora_phys3d_vel_x,                   [I64],                                           F64]
        [scalar,   phys3d_vel_y,                  aurora_phys3d_vel_y,                   [I64],                                           F64]
        [scalar,   phys3d_vel_z,                  aurora_phys3d_vel_z,                   [I64],                                           F64]
        [scalar,   phys3d_set_vel,                aurora_phys3d_set_vel,                 [I64, F64, F64, F64],                            void]
        [scalar,   phys3d_set_pos,                aurora_phys3d_set_pos,                 [I64, F64, F64, F64],                            void]
        [scalar,   phys3d_apply_impulse,          aurora_phys3d_apply_impulse,           [I64, F64, F64, F64],                            void]
        [scalar,   phys3d_move_character,         aurora_phys3d_move_character,          [I64, F64, F64, F64, F64],                       void]
        [scalar,   phys3d_grounded,               aurora_phys3d_grounded,                [I64],                                           I64]
        [scalar,   phys3d_raycast,                aurora_phys3d_raycast,                 [F64, F64, F64, F64, F64, F64, F64],             F64]

        // 3D pathfinding.
        [scalar,   nav3d_init,                    aurora_nav3d_init,                     [I64, I64, I64],                                 void]
        [scalar,   nav3d_wall,                    aurora_nav3d_wall,                     [I64, I64, I64, I64],                            void]
        [scalar,   nav3d_find,                    aurora_nav3d_find,                     [I64, I64, I64, I64, I64, I64],                  I64]
        [scalar,   nav3d_x,                       aurora_nav3d_x,                        [I64],                                           I64]
        [scalar,   nav3d_y,                       aurora_nav3d_y,                        [I64],                                           I64]
        [scalar,   nav3d_z,                       aurora_nav3d_z,                        [I64],                                           I64]
        [special,  navmesh_build,                 aurora_navmesh_build,                  [Ptr, I64, Ptr, I64],                            I64]
        [scalar,   navmesh_find,                  aurora_navmesh_find,                   [F64, F64, F64, F64, F64, F64],                  I64]
        [scalar,   navmesh_x,                     aurora_navmesh_x,                      [I64],                                           F64]
        [scalar,   navmesh_y,                     aurora_navmesh_y,                      [I64],                                           F64]
        [scalar,   navmesh_z,                     aurora_navmesh_z,                      [I64],                                           F64]

        // 3D rendering.
        [special,  r3d_load_model,                aurora_r3d_load_model,                 [Ptr, I64],                                      I64]
        [scalar,   r3d_make_box,                  aurora_r3d_make_box,                   [F64, F64, F64],                                 I64]
        [scalar,   r3d_make_box_sized,            aurora_r3d_make_box_sized,             [F64, F64, F64, F64, F64, F64],                  I64]
        [scalar,   r3d_make_box_emissive,         aurora_r3d_make_box_emissive,          [F64, F64, F64, F64, F64, F64],                  I64]
        [scalar,   r3d_make_sphere,               aurora_r3d_make_sphere,                [I64, F64, F64, F64],                            I64]
        [scalar,   r3d_make_plane,                aurora_r3d_make_plane,                 [F64, F64, F64, F64, F64],                       I64]
        [scalar,   r3d_camera,                    aurora_r3d_camera,                     [F64, F64, F64, F64, F64, F64, F64],             void]
        [scalar,   r3d_camera_roll,               aurora_r3d_camera_roll,                [F64],                                           void]
        [scalar,   r3d_light,                     aurora_r3d_light,                      [F64, F64, F64, F64, F64, F64, F64],             void]
        [scalar,   r3d_clear,                     aurora_r3d_clear,                      [F64, F64, F64],                                 void]
        [scalar,   r3d_begin,                     aurora_r3d_begin,                      [],                                              void]
        [scalar,   r3d_draw,                      aurora_r3d_draw,                       [I64, F64, F64, F64, F64, F64, F64, F64],        void]
        [scalar,   r3d_draw_quat,                 aurora_r3d_draw_quat,                  [I64, F64, F64, F64, F64, F64, F64, F64, F64],   void]
        [scalar,   r3d_draw_tint,                 aurora_r3d_draw_tint,                  [I64, F64, F64, F64, F64, F64, F64, F64, F64, F64, F64], void]
        [scalar,   r3d_draw_shield,               aurora_r3d_draw_shield,                [I64, F64, F64, F64, F64, F64, F64, F64, F64, F64], void]
        [scalar,   r3d_draw_on_joint,             aurora_r3d_draw_on_joint,              [I64, I64, I64, F64, F64, F64, F64, F64, F64, F64, F64, F64, F64, F64, F64, F64, F64], void]
        [scalar,   r3d_joint_dump,                aurora_r3d_joint_dump,                 [I64],                                           void]
        [scalar,   r3d_joint_pos,                 aurora_r3d_joint_pos,                  [I64, I64, I64],                                 F64]
        [scalar,   r3d_anim_play,                 aurora_r3d_anim_play,                  [I64, I64, I64, F64, F64],                       void]
        [scalar,   r3d_anim_update,               aurora_r3d_anim_update,                [I64, F64],                                      void]
        [scalar,   r3d_anim_play_upper,           aurora_r3d_anim_play_upper,            [I64, I64, I64, F64, F64, I64],                  void]
        [scalar,   r3d_anim_aim_upper,            aurora_r3d_anim_aim_upper,             [I64, I64, I64, F64, F64, F64, I64],             void]
        [scalar,   r3d_anim_blend,                aurora_r3d_anim_blend,                 [I64, I64, I64, F64, F64, F64],                  void]
        [scalar,   r3d_anim_seek_upper,           aurora_r3d_anim_seek_upper,            [I64, F64],                                      void]
        [scalar,   r3d_pose_bone,                 aurora_r3d_pose_bone,                  [I64, I64, F64, F64, F64],                       void]
        [scalar,   r3d_hide_joint,                aurora_r3d_hide_joint,                 [I64, I64],                                      void]
        [scalar,   r3d_clear_pose,                aurora_r3d_clear_pose,                 [I64],                                           void]
        [scalar,   r3d_anim_stop_upper,           aurora_r3d_anim_stop_upper,            [I64, F64],                                      void]
        [scalar,   r3d_clip_count,                aurora_r3d_clip_count,                 [I64],                                           I64]
        [scalar,   r3d_present,                   aurora_r3d_present,                    [],                                              I64]
        [scalar,   r3d_fog,                       aurora_r3d_fog,                        [F64, F64, F64, F64],                            void]
        [scalar,   r3d_speedlines,                aurora_r3d_speedlines,                 [F64, F64],                                      void]
        [scalar,   r3d_damage,                    aurora_r3d_damage,                     [F64, F64, F64, F64, F64],                       void]
        [scalar,   r3d_blur,                      aurora_r3d_blur,                       [F64],                                           void]
        [scalar,   r3d_sky,                       aurora_r3d_sky,                        [I64, F64, F64, F64, F64, F64, F64],             void]
        [scalar,   r3d_shadows,                   aurora_r3d_shadows,                    [I64],                                           void]
        [scalar,   r3d_ssao,                      aurora_r3d_ssao,                       [I64],                                           void]
        [scalar,   r3d_viewmodel,                 aurora_r3d_viewmodel,                  [I64],                                           void]
        [scalar,   r3d_point_shadows,             aurora_r3d_point_shadows,              [I64],                                           void]
        [scalar,   r3d_clear_lights,              aurora_r3d_clear_lights,               [],                                              void]
        [scalar,   r3d_point_light,               aurora_r3d_point_light,                [F64, F64, F64, F64, F64, F64, F64, F64],        void]
        [scalar,   r3d_make_sprite,               aurora_r3d_make_sprite,                [F64, F64, F64],                                 I64]
        [scalar,   r3d_draw_billboard,            aurora_r3d_draw_billboard,             [I64, F64, F64, F64, F64],                       void]
        [scalar,   r3d_debug_line,                aurora_r3d_debug_line,                 [F64, F64, F64, F64, F64, F64, F64, F64, F64],   void]
        [scalar,   r3d_debug_skeleton,            aurora_r3d_debug_skeleton,             [I64, F64, F64, F64, F64, F64, F64, F64, F64],   void]
        [scalar,   r3d_frustum_cull,              aurora_r3d_frustum_cull,               [I64],                                           void]
        [scalar,   r3d_screen_x,                  aurora_r3d_screen_x,                   [F64, F64, F64],                                 F64]
        [scalar,   r3d_screen_y,                  aurora_r3d_screen_y,                   [F64, F64, F64],                                 F64]
        [scalar,   mouse_dx,                      aurora_mouse_dx,                       [],                                              F64]
        [scalar,   mouse_dy,                      aurora_mouse_dy,                       [],                                              F64]
        [scalar,   mouse_scroll,                  aurora_mouse_scroll,                   [],                                              F64]
        [scalar,   mouse_button,                  aurora_mouse_button,                   [I64],                                           I64]
        [scalar,   grab_mouse,                    aurora_grab_mouse,                     [I64],                                           void]
        [scalar,   audio_listener,                aurora_audio_listener,                 [F64, F64, F64, F64, F64, F64],                  void]
        [scalar,   play_sound_at,                 aurora_play_sound_at,                  [I64, I64, I64, F64, F64, F64],                  void]
        [scalar,   play_sound_handle,             aurora_play_sound_handle,              [I64, I64],                                      void]
        [scalar,   play_sound_handle_at,          aurora_play_sound_handle_at,           [I64, I64, F64, F64, F64],                       void]
        [scalar,   play_music,                    aurora_play_music,                     [I64, I64],                                      void]
        [scalar,   music_volume,                  aurora_music_volume,                   [I64],                                           void]
        [scalar,   music_stop,                    aurora_music_stop,                     [],                                              void]
        [scalar,   play_ambience,                 aurora_play_ambience,                  [I64, I64],                                      void]
        [scalar,   ambience_volume,               aurora_ambience_volume,                [I64],                                           void]
        [scalar,   ambience_stop,                 aurora_ambience_stop,                  [],                                              void]
        [scalar,   phys3d_raycast_full,           aurora_phys3d_raycast_full,            [F64, F64, F64, F64, F64, F64, F64],             I64]
        [scalar,   phys3d_raycast_ex,             aurora_phys3d_raycast_ex,              [I64, F64, F64, F64, F64, F64, F64, F64],        I64]
        [scalar,   phys3d_raycast_world,          aurora_phys3d_raycast_world,           [I64, F64, F64, F64, F64, F64, F64, F64],        I64]
        [scalar,   phys3d_hit_x,                  aurora_phys3d_hit_x,                   [],                                              F64]
        [scalar,   phys3d_hit_y,                  aurora_phys3d_hit_y,                   [],                                              F64]
        [scalar,   phys3d_hit_z,                  aurora_phys3d_hit_z,                   [],                                              F64]
        [scalar,   phys3d_hit_nx,                 aurora_phys3d_hit_nx,                  [],                                              F64]
        [scalar,   phys3d_hit_ny,                 aurora_phys3d_hit_ny,                  [],                                              F64]
        [scalar,   phys3d_hit_nz,                 aurora_phys3d_hit_nz,                  [],                                              F64]
        [scalar,   phys3d_hit_body,               aurora_phys3d_hit_body,                [],                                              I64]
        [scalar,   phys3d_spherecast,             aurora_phys3d_spherecast,              [F64, F64, F64, F64, F64, F64, F64, F64],        F64]
        [scalar,   phys3d_overlap_sphere,         aurora_phys3d_overlap_sphere,          [F64, F64, F64, F64],                            I64]
        [scalar,   phys3d_debug_draw,             aurora_phys3d_debug_draw,              [F64, F64, F64],                                 void]
        [scalar,   phys3d_apply_force,            aurora_phys3d_apply_force,             [I64, F64, F64, F64],                            void]
        [scalar,   phys3d_apply_torque,           aurora_phys3d_apply_torque,            [I64, F64, F64, F64],                            void]
        [scalar,   phys3d_set_angvel,             aurora_phys3d_set_angvel,              [I64, F64, F64, F64],                            void]
        [scalar,   phys3d_set_rot,                aurora_phys3d_set_rot,                 [I64, F64, F64, F64, F64],                       void]
        [scalar,   phys3d_rot_qx,                 aurora_phys3d_rot_qx,                  [I64],                                           F64]
        [scalar,   phys3d_rot_qy,                 aurora_phys3d_rot_qy,                  [I64],                                           F64]
        [scalar,   phys3d_rot_qz,                 aurora_phys3d_rot_qz,                  [I64],                                           F64]
        [scalar,   phys3d_rot_qw,                 aurora_phys3d_rot_qw,                  [I64],                                           F64]
        [scalar,   net_host,                      aurora_net_host,                       [I64],                                           I64]
        [special,  net_join,                      aurora_net_join,                       [Ptr, I64, I64],                                 I64]
        [special,  net_sim,                       aurora_net_sim,                        [Ptr, Ptr, I64, I64],                            void]
        [special,  net_serve,                     aurora_net_serve,                      [Ptr, Ptr],                                      void]
        [special,  net_send_input,                aurora_net_send_input,                 [Ptr, I64],                                      I64]
        [special,  save_settings,                 aurora_save_settings,                  [Ptr, I64],                                      I64]
        [special,  load_settings,                 aurora_load_settings,                  [Ptr, I64],                                      I64]
        [scalar,   net_update,                    aurora_net_update,                     [F64],                                           void]
        [scalar,   net_leave,                     aurora_net_leave,                      [],                                              void]
        [scalar,   net_my_id,                     aurora_net_my_id,                      [],                                              I64]
        [scalar,   net_is_server,                 aurora_net_is_server,                  [],                                              I64]
        [scalar,   net_player_count,              aurora_net_player_count,               [],                                              I64]
        [scalar,   net_player_id_at,              aurora_net_player_id_at,               [I64],                                           I64]
        [scalar,   net_player_x,                  aurora_net_player_x,                   [I64],                                           F64]
        [scalar,   net_player_y,                  aurora_net_player_y,                   [I64],                                           F64]
        [scalar,   net_player_z,                  aurora_net_player_z,                   [I64],                                           F64]
        [scalar,   net_player_yaw,                aurora_net_player_yaw,                 [I64],                                           F64]
        [scalar,   net_player_state,              aurora_net_player_state,               [I64, I64],                                      F64]
        [scalar,   net_set_meta,                  aurora_net_set_meta,                   [I64, F64],                                      void]
        [scalar,   net_player_meta,               aurora_net_player_meta,                [I64, I64],                                      F64]
        [special,  net_set_name,                  aurora_net_set_name,                   [Ptr, I64],                                      void]
        [scalar,   net_player_name_len,           aurora_net_player_name_len,            [I64],                                           I64]
        [scalar,   net_player_name_char,          aurora_net_player_name_char,           [I64, I64],                                      I64]
        [scalar,   net_local_x,                   aurora_net_local_x,                    [],                                              F64]
        [scalar,   net_local_y,                   aurora_net_local_y,                    [],                                              F64]
        [scalar,   net_local_z,                   aurora_net_local_z,                    [],                                              F64]
        [scalar,   net_local_yaw,                 aurora_net_local_yaw,                  [],                                              F64]
        [scalar,   net_state,                     aurora_net_state,                      [I64, I64],                                      F64]
        [scalar,   net_local_state,               aurora_net_local_state,                [I64],                                           F64]
        [scalar,   net_interest,                  aurora_net_interest,                   [F64],                                           void]
        [scalar,   net_max_clients,               aurora_net_max_clients,                [I64],                                           void]
        [scalar,   net_rejected,                  aurora_net_rejected,                   [],                                              I64]
        [scalar,   net_connected,                 aurora_net_connected,                  [],                                              I64]
        [scalar,   net_dedicated,                 aurora_net_dedicated,                  [],                                              void]
        [scalar,   net_cfg_set,                   aurora_net_cfg_set,                    [I64, F64],                                      void]
        [scalar,   net_cfg_get,                   aurora_net_cfg_get,                    [I64],                                           F64]
        [scalar,   net_set_bot_count,             aurora_net_set_bot_count,              [I64],                                           void]
        [scalar,   net_set_bot,                   aurora_net_set_bot,                    [I64, F64, F64, F64, F64],                       void]
        [scalar,   net_set_bot_input,             aurora_net_set_bot_input,              [I64, I64],                                      void]
        [scalar,   net_set_bot_state,             aurora_net_set_bot_state,              [I64, I64],                                      void]
        [scalar,   net_set_bot_alive,             aurora_net_set_bot_alive,              [I64, I64],                                      void]
        [scalar,   net_set_bot_meta,              aurora_net_set_bot_meta,               [I64, I64, F64],                                 void]
        [special,  net_set_bot_name,              aurora_net_set_bot_name,               [I64, Ptr, I64],                                 void]
        [scalar,   net_bot_count,                 aurora_net_bot_count,                  [],                                              I64]
        [scalar,   net_set_object_count,          aurora_net_set_object_count,           [I64],                                           void]
        [scalar,   net_set_object,                aurora_net_set_object,                 [I64, F64, F64, F64],                            void]
        [scalar,   net_object_count,              aurora_net_object_count,               [],                                              I64]
        [scalar,   net_object_x,                  aurora_net_object_x,                   [I64],                                           F64]
        [scalar,   net_object_y,                  aurora_net_object_y,                   [I64],                                           F64]
        [scalar,   net_object_z,                  aurora_net_object_z,                   [I64],                                           F64]
        [scalar,   net_set_object_rot,            aurora_net_set_object_rot,             [I64, F64, F64, F64, F64],                       void]
        [scalar,   net_object_qx,                 aurora_net_object_qx,                  [I64],                                           F64]
        [scalar,   net_object_qy,                 aurora_net_object_qy,                  [I64],                                           F64]
        [scalar,   net_object_qz,                 aurora_net_object_qz,                  [I64],                                           F64]
        [scalar,   net_object_qw,                 aurora_net_object_qw,                  [I64],                                           F64]
        [scalar,   net_set_object_vel,            aurora_net_set_object_vel,             [I64, F64, F64, F64],                            void]
        [scalar,   net_object_vx,                 aurora_net_object_vx,                  [I64],                                           F64]
        [scalar,   net_object_vy,                 aurora_net_object_vy,                  [I64],                                           F64]
        [scalar,   net_object_vz,                 aurora_net_object_vz,                  [I64],                                           F64]
        [scalar,   net_set_fx_count,              aurora_net_set_fx_count,               [I64],                                           void]
        [scalar,   net_set_fx,                    aurora_net_set_fx,                     [I64, F64, F64, F64, F64],                       void]
        [scalar,   net_fx_count,                  aurora_net_fx_count,                   [],                                              I64]
        [scalar,   net_fx_x,                      aurora_net_fx_x,                       [I64],                                           F64]
        [scalar,   net_fx_y,                      aurora_net_fx_y,                       [I64],                                           F64]
        [scalar,   net_fx_z,                      aurora_net_fx_z,                       [I64],                                           F64]
        [scalar,   net_fx_kind,                   aurora_net_fx_kind,                    [I64],                                           F64]
        [scalar,   net_hit_radius,                aurora_net_hit_radius,                 [F64],                                           void]
        [scalar,   net_spawn_at,                  aurora_net_spawn_at,                   [F64, F64, F64],                                 void]
        [scalar,   net_spawn_input_slot,          aurora_net_spawn_input_slot,           [I64],                                           void]
        [scalar,   net_respawn_client,            aurora_net_respawn_client,             [I64, F64, F64, F64],                            void]
        [scalar,   net_impulse_input_slot,        aurora_net_impulse_input_slot,         [I64],                                           void]
        [scalar,   net_push_impulse,              aurora_net_push_impulse,               [I64, F64, F64, F64],                            void]
        [scalar,   net_respawn_trigger_slot,      aurora_net_respawn_trigger_slot,       [I64],                                           void]
        [scalar,   net_force_respawn,             aurora_net_force_respawn,              [I64],                                           void]
        [scalar,   net_fire,                      aurora_net_fire,                       [F64, F64, F64, F64, F64, F64, I64],             void]
        [scalar,   net_server_hit_count,          aurora_net_server_hit_count,           [],                                              I64]
        [scalar,   net_server_hit_shooter,        aurora_net_server_hit_shooter,         [I64],                                           I64]
        [scalar,   net_server_hit_victim,         aurora_net_server_hit_victim,          [I64],                                           I64]
        [scalar,   net_server_hit_weapon,         aurora_net_server_hit_weapon,          [I64],                                           I64]
        [scalar,   net_server_hit_x,              aurora_net_server_hit_x,               [I64],                                           F64]
        [scalar,   net_server_hit_y,              aurora_net_server_hit_y,               [I64],                                           F64]
        [scalar,   net_server_hit_z,              aurora_net_server_hit_z,               [I64],                                           F64]
        [scalar,   net_server_hits_clear,         aurora_net_server_hits_clear,          [],                                              void]
        [scalar,   net_push_kill,                 aurora_net_push_kill,                  [I64, I64],                                      void]
        [scalar,   net_kill_count,                aurora_net_kill_count,                 [],                                              I64]
        [scalar,   net_kill_killer,               aurora_net_kill_killer,                [I64],                                           I64]
        [scalar,   net_kill_victim,               aurora_net_kill_victim,                [I64],                                           I64]
        [scalar,   net_kills_clear,               aurora_net_kills_clear,                [],                                              void]
        [scalar,   net_push_shot,                 aurora_net_push_shot,                  [I64, F64, F64, F64, F64, F64, F64, I64],        void]
        [scalar,   net_shot_count,                aurora_net_shot_count,                 [],                                              I64]
        [scalar,   net_shot_shooter,              aurora_net_shot_shooter,               [I64],                                           I64]
        [scalar,   net_shot_field,                aurora_net_shot_field,                 [I64, I64],                                      F64]
        [scalar,   net_shot_weapon,               aurora_net_shot_weapon,                [I64],                                           I64]
        [scalar,   net_shots_clear,               aurora_net_shots_clear,                [],                                              void]
        [scalar,   net_push_boom,                 aurora_net_push_boom,                  [I64, F64, F64, F64, F64],                       void]
        [scalar,   net_boom_count,                aurora_net_boom_count,                 [],                                              I64]
        [scalar,   net_boom_source,               aurora_net_boom_source,                [I64],                                           I64]
        [scalar,   net_boom_field,                aurora_net_boom_field,                 [I64, I64],                                      F64]
        [scalar,   net_booms_clear,               aurora_net_booms_clear,                [],                                              void]
        [scalar,   net_projectile_intent,         aurora_net_projectile_intent,          [I64, F64, F64, F64, F64, F64, F64],             void]
        [scalar,   net_server_projectile_count,   aurora_net_server_projectile_count,    [],                                              I64]
        [scalar,   net_server_projectile_shooter, aurora_net_server_projectile_shooter,  [I64],                                           I64]
        [scalar,   net_server_projectile_kind,    aurora_net_server_projectile_kind,     [I64],                                           I64]
        [scalar,   net_server_projectile_ox,      aurora_net_server_projectile_ox,       [I64],                                           F64]
        [scalar,   net_server_projectile_oy,      aurora_net_server_projectile_oy,       [I64],                                           F64]
        [scalar,   net_server_projectile_oz,      aurora_net_server_projectile_oz,       [I64],                                           F64]
        [scalar,   net_server_projectile_vx,      aurora_net_server_projectile_vx,       [I64],                                           F64]
        [scalar,   net_server_projectile_vy,      aurora_net_server_projectile_vy,       [I64],                                           F64]
        [scalar,   net_server_projectile_vz,      aurora_net_server_projectile_vz,       [I64],                                           F64]
        [scalar,   net_server_projectiles_clear,  aurora_net_server_projectiles_clear,   [],                                              void]
        [scalar,   net_set_player_meta,           aurora_net_set_player_meta,            [I64, I64, F64],                                 void]
        [scalar,   net_hit_player,                aurora_net_hit_player,                 [],                                              I64]
        [scalar,   net_hit_seq,                   aurora_net_hit_seq,                    [],                                              I64]
        [scalar,   net_hit_x,                     aurora_net_hit_x,                      [],                                              F64]
        [scalar,   net_hit_y,                     aurora_net_hit_y,                      [],                                              F64]
        [scalar,   net_hit_z,                     aurora_net_hit_z,                      [],                                              F64]
        [scalar,   input_bind,                    aurora_input_bind,                     [I64, I64],                                      void]
        [scalar,   input_binding,                 aurora_input_binding,                  [I64],                                           I64]
        [scalar,   input_down,                    aurora_input_down,                     [I64],                                           I64]
        [scalar,   input_axis,                    aurora_input_axis,                     [I64, I64],                                      F64]
        [scalar,   input_suppress,                aurora_input_suppress,                 [I64],                                           void]
        [scalar,   f32_load,                      aurora_f32_load,                       [I64, I64],                                      F64]
        [scalar,   f32_store,                     aurora_f32_store,                      [I64, I64, F64],                                 void]
        [scalar,   f32_blob,                      aurora_f32_blob,                       [I64],                                           I64]
        [special,  sin,                           aurora_sin,                            [F64],                                           F64]
        [special,  cos,                           aurora_cos,                            [F64],                                           F64]
        [special,  tan,                           aurora_tan,                            [F64],                                           F64]
        [special,  pow,                           aurora_pow,                            [F64, F64],                                      F64]
        [special,  log,                           aurora_log,                            [F64],                                           F64]
        [special,  exp,                           aurora_exp,                            [F64],                                           F64]
        [special,  atan2,                         aurora_atan2,                          [F64, F64],                                      F64]
        [special,  draw_text,                     aurora_draw_text,                      [I64, I64, Ptr, I64, I64, I64],                  void]
        [special,  draw_int,                      aurora_draw_int,                       [I64, I64, I64, I64, I64],                       void]
        [special,  text_width,                    aurora_text_width,                     [Ptr, I64, I64],                                 I64]
        [special,  scene_save,                    aurora_scene_save,                     [Ptr, I64],                                      I64]
        [special,  scene_load,                    aurora_scene_load,                     [Ptr, I64],                                      I64]
        [internal, prof_enter,                    aurora_prof_enter,                     [Ptr, I64],                                      void]
        [internal, prof_exit,                     aurora_prof_exit,                      [],                                              void]
        [internal, str_concat,                    aurora_str_concat,                     [Ptr, Ptr, I64, Ptr, I64],                       void]
        [internal, str_eq,                        aurora_str_eq,                         [Ptr, I64, Ptr, I64],                            I64]
        [internal, str_char_at,                   aurora_str_char_at,                    [Ptr, I64, I64],                                 I64]
        [internal, str_substr,                    aurora_str_substr,                     [Ptr, Ptr, I64, I64, I64],                       void]
        [internal, str_starts_with,               aurora_str_starts_with,                [Ptr, I64, Ptr, I64],                            I64]
        [internal, int_to_str,                    aurora_int_to_str,                     [Ptr, I64],                                      void]
        [internal, float_to_str,                  aurora_float_to_str,                   [Ptr, F64],                                      void]
        [special,  play_note,                     aurora_play_note,                      [I64, I64],                                      void]
        [special,  play_sound,                    aurora_play_sound,                     [I64, I64, I64],                                 void]
        [special,  play_noise,                    aurora_play_noise,                     [I64, I64],                                      void]
        [special,  audio_volume,                  aurora_audio_volume,                   [I64],                                           void]
        [special,  window_fullscreen,             aurora_window_fullscreen,              [I64],                                           void]
        [special,  audio_stop,                    aurora_audio_stop,                     [],                                              void]
        [special,  gpu_render,                    aurora_gpu_render,                     [Ptr, I64, I64],                                 void]
        [special,  window_open,                   aurora_window_open,                    [I64, I64],                                      void]
        [special,  window_present,                aurora_window_present,                 [],                                              I64]
        [special,  surface_w,                     aurora_surface_w,                      [],                                              I64]
        [special,  surface_h,                     aurora_surface_h,                      [],                                              I64]
        [special,  key_down,                      aurora_key_down,                       [I64],                                           I64]
        [special,  input_char,                    aurora_input_char,                     [],                                              I64]
        [special,  mouse_x,                       aurora_mouse_x,                        [],                                              I64]
        [special,  mouse_y,                       aurora_mouse_y,                        [],                                              I64]
        [special,  mouse_down,                    aurora_mouse_down,                     [],                                              I64]

        // Process environment: the program's own argument vector and env vars.
        [scalar,   sys_argc,                      aurora_sys_argc,                       [],                                              I64]
        [text,     sys_arg,                       aurora_sys_arg,                        [I64],                                           Str]
        [text,     sys_env,                       aurora_sys_env,                        [Ptr, I64],                                      Str]

        // Assertions.
        [scalar,   assert,                        aurora_assert,                         [I64],                                           void]

        // Native debugger hooks (only *called* when `debug`, but always importable).
        [internal, dbg_enter,                     aurora_dbg_enter,                      [Ptr, I64],                                      void]
        [internal, dbg_leave,                     aurora_dbg_leave,                      [],                                              void]
        [internal, dbg_stmt,                      aurora_dbg_stmt,                       [I64],                                           void]
        [internal, dbg_var,                       aurora_dbg_var,                        [Ptr, I64, I64],                                 void]
        [internal, dbg_var_f64,                   aurora_dbg_var_f64,                    [Ptr, I64, F64],                                 void]

        // Builtins the backend lowers inline (no runtime call): printing,
        // polymorphic math/bit ops, string ops, ECS spawn, and `run_systems`.
        [inline,   print,                         none,                                  [],                                              void]
        [inline,   println,                       none,                                  [],                                              void]
        [inline,   sqrt,                          none,                                  [],                                              void]
        [inline,   floor,                         none,                                  [],                                              void]
        [inline,   ceil,                          none,                                  [],                                              void]
        [inline,   round,                         none,                                  [],                                              void]
        [inline,   abs,                           none,                                  [],                                              void]
        [inline,   min,                           none,                                  [],                                              void]
        [inline,   max,                           none,                                  [],                                              void]
        [inline,   clamp,                         none,                                  [],                                              void]
        [inline,   len,                           none,                                  [],                                              void]
        [inline,   str,                           none,                                  [],                                              void]
        [inline,   spawn,                         none,                                  [],                                              void]
        [inline,   run_systems,                   none,                                  [],                                              void]
        [inline,   band,                          none,                                  [],                                              void]
        [inline,   bor,                           none,                                  [],                                              void]
        [inline,   bxor,                          none,                                  [],                                              void]
        [inline,   shl,                           none,                                  [],                                              void]
        [inline,   shr,                           none,                                  [],                                              void]
        [inline,   bnot,                          none,                                  [],                                              void]
        [inline,   char_at,                       none,                                  [],                                              void]
        [inline,   substr,                        none,                                  [],                                              void]
        [inline,   starts_with,                   none,                                  [],                                              void]

        // Runtime functions that are not builtins: they only need a JIT symbol
        // and an AOT link edge, so `@extern` declarations can bind them.
        [linkonly, ffi_dot,                       aurora_ffi_dot,                        [],                                              void]
        [linkonly, ffi_dotf,                      aurora_ffi_dotf,                       [],                                              void]
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
    ($([$kind:ident, $name:ident, $sym:ident, [$($p:ident),*], $ret:ident])*) => {
        /// Every builtin, in table order.
        pub const TABLE: &[Builtin] = &[
            $(Builtin {
                name: stringify!($name),
                kind: kind_of!($kind),
                symbol: sym_str!($kind, $sym),
                params: &[$(ty_of!($p)),*],
                ret: ret_of!($ret),
            }),*
        ];
    };
}

for_each_builtin!(build_table);

/// Every builtin name Aurora source may call, in table order.
pub fn builtin_names() -> &'static [&'static str] {
    static NAMES: OnceLock<Vec<&'static str>> = OnceLock::new();
    NAMES.get_or_init(|| {
        TABLE.iter().filter(|b| b.kind.is_aurora_visible()).map(|b| b.name).collect()
    })
}

/// Is `name` a builtin Aurora source may call? O(1); the front end asks this of
/// every unresolved call, so a call to a builtin is not reported as a typo and a
/// real typo still is.
pub fn is_builtin(name: &str) -> bool {
    static SET: OnceLock<HashSet<&'static str>> = OnceLock::new();
    SET.get_or_init(|| builtin_names().iter().copied().collect()).contains(name)
}

/// The table row for `name`, including rows Aurora source cannot call
/// (`internal` / `linkonly`), which the backend looks up by the same key. O(1).
pub fn lookup(name: &str) -> Option<&'static Builtin> {
    static BY_NAME: OnceLock<HashMap<&'static str, &'static Builtin>> = OnceLock::new();
    BY_NAME.get_or_init(|| TABLE.iter().map(|b| (b.name, b)).collect()).get(name).copied()
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
