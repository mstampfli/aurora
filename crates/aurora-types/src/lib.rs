//! Type representation and a Hindley-Milner-style unification engine (grammar
//! spec §7). This is the core the bidirectional type checker will drive; it is
//! deliberately independent of the AST so it can be tested in isolation.
//!
//! Inference variables live in [`InferCtx`] as a union-find of substitutions.
//! `unify` makes two types equal (binding variables, occurs-checked), and
//! `resolve_deep` reads back a fully-substituted type once inference settles.
//!
//! Note (current limitation): numeric literals are given concrete default types
//! (`i32`/`f32`) at construction rather than via numeric inference variables, so
//! `let x: u8 = 1` defaulting is not modelled yet — that arrives with the
//! bidirectional checker's literal handling.

use aurora_lexer::{FloatTy, IntTy};

/// A type. `Var` is an inference variable resolved through [`InferCtx`].
#[derive(Clone, Debug, PartialEq)]
pub enum Ty {
    Unit,
    Bool,
    Char,
    Str,
    Int(IntTy),
    Float(FloatTy),
    /// An unsuffixed integer literal — unifies with any concrete integer type
    /// (but not bool/float). Defaults to `i32` if never constrained.
    IntLit,
    /// An unsuffixed float literal — unifies with any concrete float type.
    FloatLit,
    /// `VecN` for N in 2..=4.
    Vec(u8),
    /// Square `MatN`.
    Mat(u8),
    Quat,
    Color,
    /// A nominal type: struct, enum, or component, referenced by name.
    Named(String),
    Tuple(Vec<Ty>),
    Ref { mutable: bool, inner: Box<Ty> },
    Owned(Box<Ty>),
    Rc(Box<Ty>),
    Array(Box<Ty>, Option<u64>),
    Fn(Vec<Ty>, Box<Ty>),
    /// Inference variable (index into `InferCtx`).
    Var(u32),
    /// Propagated type error; unifies with anything to avoid error cascades.
    Error,
}

impl Ty {
    pub fn unit() -> Ty {
        Ty::Unit
    }
    pub fn reference(mutable: bool, inner: Ty) -> Ty {
        Ty::Ref { mutable, inner: Box::new(inner) }
    }

    /// A short human-facing rendering for diagnostics.
    pub fn display(&self) -> String {
        match self {
            Ty::Unit => "()".into(),
            Ty::Bool => "bool".into(),
            Ty::Char => "char".into(),
            Ty::Str => "str".into(),
            Ty::Int(t) => format!("{t:?}").to_lowercase(),
            Ty::Float(t) => format!("{t:?}").to_lowercase(),
            Ty::IntLit => "{integer}".into(),
            Ty::FloatLit => "{float}".into(),
            Ty::Vec(n) => format!("Vec{n}"),
            Ty::Mat(n) => format!("Mat{n}"),
            Ty::Quat => "Quat".into(),
            Ty::Color => "Color".into(),
            Ty::Named(n) => n.clone(),
            Ty::Tuple(ts) => {
                let inner: Vec<_> = ts.iter().map(Ty::display).collect();
                format!("({})", inner.join(", "))
            }
            Ty::Ref { mutable, inner } => {
                format!("&{}{}", if *mutable { "mut " } else { "" }, inner.display())
            }
            Ty::Owned(t) => format!("~{}", t.display()),
            Ty::Rc(t) => format!("rc<{}>", t.display()),
            Ty::Array(t, Some(n)) => format!("[{}; {n}]", t.display()),
            Ty::Array(t, None) => format!("[{}]", t.display()),
            Ty::Fn(params, ret) => {
                let ps: Vec<_> = params.iter().map(Ty::display).collect();
                format!("fn({}) -> {}", ps.join(", "), ret.display())
            }
            Ty::Var(id) => format!("?{id}"),
            Ty::Error => "<error>".into(),
        }
    }
}

/// A unification failure: the two types could not be made equal.
#[derive(Clone, Debug, PartialEq)]
pub struct TypeError {
    pub expected: String,
    pub found: String,
    pub message: String,
}

/// The inference context: a growable union-find of variable substitutions.
#[derive(Default)]
pub struct InferCtx {
    subst: Vec<Option<Ty>>,
}

impl InferCtx {
    pub fn new() -> InferCtx {
        InferCtx::default()
    }

    /// Allocate a fresh inference variable.
    pub fn fresh(&mut self) -> Ty {
        let id = self.subst.len() as u32;
        self.subst.push(None);
        Ty::Var(id)
    }

    /// Follow variable bindings one level (does not recurse into compounds).
    pub fn resolve_shallow(&self, ty: &Ty) -> Ty {
        let mut cur = ty.clone();
        while let Ty::Var(id) = cur {
            match &self.subst[id as usize] {
                Some(bound) => cur = bound.clone(),
                None => break,
            }
        }
        cur
    }

    /// Fully substitute all variables, recursing into compound types. Remaining
    /// unbound variables are left as `Var`.
    pub fn resolve_deep(&self, ty: &Ty) -> Ty {
        let shallow = self.resolve_shallow(ty);
        match shallow {
            Ty::Tuple(ts) => Ty::Tuple(ts.iter().map(|t| self.resolve_deep(t)).collect()),
            Ty::Ref { mutable, inner } => {
                Ty::Ref { mutable, inner: Box::new(self.resolve_deep(&inner)) }
            }
            Ty::Owned(t) => Ty::Owned(Box::new(self.resolve_deep(&t))),
            Ty::Rc(t) => Ty::Rc(Box::new(self.resolve_deep(&t))),
            Ty::Array(t, n) => Ty::Array(Box::new(self.resolve_deep(&t)), n),
            Ty::Fn(params, ret) => Ty::Fn(
                params.iter().map(|t| self.resolve_deep(t)).collect(),
                Box::new(self.resolve_deep(&ret)),
            ),
            other => other,
        }
    }

    /// Make `a` and `b` equal, binding inference variables as needed.
    pub fn unify(&mut self, a: &Ty, b: &Ty) -> Result<(), TypeError> {
        let ra = self.resolve_shallow(a);
        let rb = self.resolve_shallow(b);
        match (&ra, &rb) {
            // Errors absorb to prevent cascades.
            (Ty::Error, _) | (_, Ty::Error) => Ok(()),

            (Ty::Var(i), Ty::Var(j)) if i == j => Ok(()),
            (Ty::Var(i), other) | (other, Ty::Var(i)) => self.bind(*i, other),

            (Ty::Unit, Ty::Unit)
            | (Ty::Bool, Ty::Bool)
            | (Ty::Char, Ty::Char)
            | (Ty::Str, Ty::Str)
            | (Ty::Quat, Ty::Quat)
            | (Ty::Color, Ty::Color) => Ok(()),

            (Ty::Int(x), Ty::Int(y)) if x == y => Ok(()),
            (Ty::Float(x), Ty::Float(y)) if x == y => Ok(()),

            // Numeric literals adapt to any concrete numeric type of the right
            // kind (but not across kinds, and not to bool).
            (Ty::IntLit, Ty::IntLit) => Ok(()),
            (Ty::FloatLit, Ty::FloatLit) => Ok(()),
            (Ty::IntLit, Ty::Int(_)) | (Ty::Int(_), Ty::IntLit) => Ok(()),
            (Ty::FloatLit, Ty::Float(_)) | (Ty::Float(_), Ty::FloatLit) => Ok(()),
            (Ty::Vec(x), Ty::Vec(y)) if x == y => Ok(()),
            (Ty::Mat(x), Ty::Mat(y)) if x == y => Ok(()),
            (Ty::Named(x), Ty::Named(y)) if x == y => Ok(()),

            (Ty::Tuple(xs), Ty::Tuple(ys)) if xs.len() == ys.len() => {
                for (x, y) in xs.iter().zip(ys.iter()) {
                    self.unify(x, y)?;
                }
                Ok(())
            }
            (
                Ty::Ref { mutable: m1, inner: i1 },
                Ty::Ref { mutable: m2, inner: i2 },
            ) if m1 == m2 => self.unify(i1, i2),
            (Ty::Owned(x), Ty::Owned(y)) | (Ty::Rc(x), Ty::Rc(y)) => self.unify(x, y),
            (Ty::Array(x, n1), Ty::Array(y, n2)) if n1 == n2 => self.unify(x, y),
            (Ty::Fn(p1, r1), Ty::Fn(p2, r2)) if p1.len() == p2.len() => {
                for (x, y) in p1.iter().zip(p2.iter()) {
                    self.unify(x, y)?;
                }
                self.unify(r1, r2)
            }

            _ => Err(TypeError {
                expected: ra.display(),
                found: rb.display(),
                message: format!("expected `{}`, found `{}`", ra.display(), rb.display()),
            }),
        }
    }

    fn bind(&mut self, var: u32, ty: &Ty) -> Result<(), TypeError> {
        if self.occurs(var, ty) {
            return Err(TypeError {
                expected: format!("?{var}"),
                found: ty.display(),
                message: format!("infinite type: `?{var}` occurs in `{}`", ty.display()),
            });
        }
        self.subst[var as usize] = Some(ty.clone());
        Ok(())
    }

    /// Occurs check: does `var` appear anywhere inside `ty` (after resolution)?
    fn occurs(&self, var: u32, ty: &Ty) -> bool {
        match self.resolve_shallow(ty) {
            Ty::Var(id) => id == var,
            Ty::Tuple(ts) => ts.iter().any(|t| self.occurs(var, t)),
            Ty::Ref { inner, .. } | Ty::Owned(inner) | Ty::Rc(inner) | Ty::Array(inner, _) => {
                self.occurs(var, &inner)
            }
            Ty::Fn(params, ret) => {
                params.iter().any(|t| self.occurs(var, t)) || self.occurs(var, &ret)
            }
            _ => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aurora_lexer::IntTy;

    #[test]
    fn unify_concrete_equal_and_mismatch() {
        let mut cx = InferCtx::new();
        assert!(cx.unify(&Ty::Bool, &Ty::Bool).is_ok());
        assert!(cx.unify(&Ty::Int(IntTy::I32), &Ty::Int(IntTy::I32)).is_ok());
        let err = cx.unify(&Ty::Bool, &Ty::Int(IntTy::I32)).unwrap_err();
        assert_eq!(err.expected, "bool");
        assert_eq!(err.found, "i32");
    }

    #[test]
    fn variable_binds_and_resolves() {
        let mut cx = InferCtx::new();
        let v = cx.fresh();
        cx.unify(&v, &Ty::Vec(3)).unwrap();
        assert_eq!(cx.resolve_deep(&v), Ty::Vec(3));
    }

    #[test]
    fn transitive_variable_unification() {
        let mut cx = InferCtx::new();
        let a = cx.fresh();
        let b = cx.fresh();
        cx.unify(&a, &b).unwrap();
        cx.unify(&b, &Ty::Float(FloatTy::F32)).unwrap();
        assert_eq!(cx.resolve_deep(&a), Ty::Float(FloatTy::F32));
    }

    #[test]
    fn tuple_structural_unification() {
        let mut cx = InferCtx::new();
        let v = cx.fresh();
        let lhs = Ty::Tuple(vec![Ty::Bool, v.clone()]);
        let rhs = Ty::Tuple(vec![Ty::Bool, Ty::Vec(2)]);
        cx.unify(&lhs, &rhs).unwrap();
        assert_eq!(cx.resolve_deep(&v), Ty::Vec(2));
    }

    #[test]
    fn tuple_arity_mismatch_fails() {
        let mut cx = InferCtx::new();
        let lhs = Ty::Tuple(vec![Ty::Bool]);
        let rhs = Ty::Tuple(vec![Ty::Bool, Ty::Bool]);
        assert!(cx.unify(&lhs, &rhs).is_err());
    }

    #[test]
    fn ref_mutability_must_match() {
        let mut cx = InferCtx::new();
        let shared = Ty::reference(false, Ty::Bool);
        let mutable = Ty::reference(true, Ty::Bool);
        assert!(cx.unify(&shared, &mutable).is_err());
    }

    #[test]
    fn occurs_check_prevents_infinite_type() {
        let mut cx = InferCtx::new();
        let v = cx.fresh();
        // v = (v, bool) would be infinite.
        let recursive = Ty::Tuple(vec![v.clone(), Ty::Bool]);
        assert!(cx.unify(&v, &recursive).is_err());
    }

    #[test]
    fn error_absorbs() {
        let mut cx = InferCtx::new();
        assert!(cx.unify(&Ty::Error, &Ty::Bool).is_ok());
        assert!(cx.unify(&Ty::Int(IntTy::I32), &Ty::Error).is_ok());
    }

    #[test]
    fn fn_type_unification() {
        let mut cx = InferCtx::new();
        let r = cx.fresh();
        let f1 = Ty::Fn(vec![Ty::Int(IntTy::I32)], Box::new(r.clone()));
        let f2 = Ty::Fn(vec![Ty::Int(IntTy::I32)], Box::new(Ty::Bool));
        cx.unify(&f1, &f2).unwrap();
        assert_eq!(cx.resolve_deep(&r), Ty::Bool);
    }
}

/// Every builtin function name the compiler lowers to a runtime call.
///
/// SINGLE SOURCE OF TRUTH, shared by codegen (which imports the host symbol)
/// and the type checker (which uses it to decide whether a call resolves).
/// Keeping one list is what lets an unresolved call be a hard error instead of
/// silently reaching codegen: if a name is neither a user fn, a local binding,
/// nor listed here, it does not exist.
pub const BUILTINS: &[&str] = &[
    "print", "println", "assert", "sqrt", "sin", "cos", "tan", "floor", "ceil", "round", "pow",
    "log", "exp", "atan2",
    "abs", "min", "max", "clamp", "len", "str", "spawn", "despawn", "run_systems", "entity_count",
    "band", "bor", "bxor", "shl", "shr", "bnot",
    "framebuffer", "clear", "pixel", "triangle", "fb_get", "save_ppm",
    "play_note", "play_sound", "play_noise", "audio_volume", "audio_stop", "window_fullscreen", "window_open", "window_present",
    "surface_w", "surface_h",
    "key_down", "input_char", "mouse_x", "mouse_y", "mouse_down", "gpu_render",
    "load_ppm", "load_image", "load_font", "draw_text", "draw_int", "text_width", "play_wav", "load_sound", "scene_save", "scene_load", "frame_reset",
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
    "r3d_camera", "r3d_camera_roll", "r3d_light", "r3d_clear", "r3d_begin", "r3d_draw", "r3d_draw_quat", "r3d_draw_tint",
    "r3d_draw_on_joint", "r3d_draw_skinned", "r3d_joint_dump", "r3d_joint_pos", "r3d_draw_shield",
    "r3d_anim_play", "r3d_anim_update", "r3d_anim_play_upper", "r3d_anim_aim_upper", "r3d_anim_blend", "r3d_anim_seek_upper", "r3d_pose_bone", "r3d_clear_pose", "r3d_hide_joint", "r3d_anim_stop_upper", "r3d_clip_count", "r3d_present",
    "r3d_fog", "r3d_speedlines", "r3d_damage", "r3d_blur", "r3d_sky", "r3d_shadows", "r3d_ssao", "r3d_viewmodel", "r3d_point_shadows", "r3d_clear_lights", "r3d_point_light",
    "r3d_make_sprite", "r3d_draw_billboard", "r3d_debug_line", "r3d_debug_skeleton", "r3d_frustum_cull",
    "r3d_screen_x", "r3d_screen_y",
    // FPS input.
    "mouse_dx", "mouse_dy", "mouse_scroll", "mouse_button", "grab_mouse", "frame_dt", "sleep_ms",
    // 3D positional audio.
    "audio_listener", "play_sound_at", "play_sound_handle", "play_sound_handle_at",
    // Background music + ambience (looping channels).
    "play_music", "music_volume", "music_stop", "audio_capture_save",
    "play_ambience", "ambience_volume", "ambience_stop",
    // Rich 3D physics queries.
    "phys3d_raycast_full", "phys3d_raycast_ex", "phys3d_raycast_world", "phys3d_hit_x", "phys3d_hit_y", "phys3d_hit_z",
    "phys3d_hit_nx", "phys3d_hit_ny", "phys3d_hit_nz", "phys3d_hit_body",
    "phys3d_spherecast", "phys3d_overlap_sphere", "phys3d_debug_draw", "phys3d_apply_force",
    "phys3d_apply_torque", "phys3d_set_angvel", "phys3d_set_rot",
    "phys3d_rot_qx", "phys3d_rot_qy", "phys3d_rot_qz", "phys3d_rot_qw",
    // Multiplayer (generic framework: the game registers its Aurora sim).
    "net_host", "net_join", "net_sim", "net_serve", "net_send_input", "net_update", "net_leave",
    "net_my_id", "net_is_server", "net_player_count", "net_player_id_at",
    "net_player_x", "net_player_y", "net_player_z", "net_player_yaw", "net_player_state",
    "net_set_meta", "net_player_meta", "net_set_name", "net_player_name_len", "net_player_name_char",
    "net_local_x", "net_local_y", "net_local_z", "net_local_yaw",
    "net_state", "net_local_state", "net_interest", "net_hit_radius", "net_max_clients", "net_rejected", "net_connected", "net_dedicated", "net_cfg_set", "net_cfg_get",
    "net_set_bot_count", "net_set_bot", "net_set_bot_input", "net_set_bot_state", "net_set_bot_alive", "net_set_bot_meta", "net_set_bot_name", "net_bot_count",
    "net_set_object_count", "net_set_object", "net_object_count", "net_object_x", "net_object_y", "net_object_z",
    "net_set_object_rot", "net_object_qx", "net_object_qy", "net_object_qz", "net_object_qw",
    "net_set_object_vel", "net_object_vx", "net_object_vy", "net_object_vz",
    "net_set_fx_count", "net_set_fx", "net_fx_count", "net_fx_x", "net_fx_y", "net_fx_z", "net_fx_kind",
    "net_spawn_at", "net_spawn_input_slot", "net_respawn_client", "net_impulse_input_slot", "net_push_impulse", "net_respawn_trigger_slot", "net_force_respawn", "net_fire",
    "net_hit_player", "net_hit_seq", "net_hit_x", "net_hit_y", "net_hit_z",
    "net_server_hit_count", "net_server_hit_shooter", "net_server_hit_victim", "net_server_hit_weapon",
    "net_server_hit_x", "net_server_hit_y", "net_server_hit_z", "net_server_hits_clear",
    "net_push_kill", "net_kill_count", "net_kill_killer", "net_kill_victim", "net_kills_clear",
    "net_push_shot", "net_shot_count", "net_shot_shooter", "net_shot_field", "net_shot_weapon", "net_shots_clear",
    "net_push_boom", "net_boom_count", "net_boom_source", "net_boom_field", "net_booms_clear",
    "net_projectile_intent", "net_server_projectile_count", "net_server_projectile_shooter",
    "net_server_projectile_kind", "net_server_projectile_ox", "net_server_projectile_oy",
    "net_server_projectile_oz", "net_server_projectile_vx", "net_server_projectile_vy",
    "net_server_projectile_vz", "net_server_projectiles_clear", "net_set_player_meta",
    // Rebindable input-action layer + raw f32-blob accessors.
    "input_bind", "input_binding", "input_down", "input_axis", "input_suppress",
    "save_settings", "load_settings",
    "f32_load", "f32_store", "f32_blob",
    // Determinism: seeded RNG + fixed timestep.
    "srand", "rand", "rand_range", "rand_int", "set_fixed_dt",
    // Data: PNG framebuffer capture, text file I/O, JSON parse/build.
    "save_png", "read_file", "write_file", "file_exists",
    "json_parse", "json_load", "json_get", "json_at", "json_len", "json_num", "json_int",
    "json_bool", "json_str", "json_kind", "json_has", "json_key", "json_free",
    "json_new_obj", "json_new_arr", "json_set", "json_set_num", "json_set_str", "json_set_bool",
    "json_push", "json_push_num", "json_push_str", "json_to_str", "json_write",
    // Headless capture + scripted input (the verification harness's hands and eyes).
    "r3d_capture", "r3d_capture_size",
    "inject_key", "inject_mouse_move", "inject_mouse_pos", "inject_mouse_button",
    "inject_scroll", "inject_char",
];

/// Is `name` a compiler builtin?
pub fn is_builtin(name: &str) -> bool {
    BUILTINS.contains(&name)
}
