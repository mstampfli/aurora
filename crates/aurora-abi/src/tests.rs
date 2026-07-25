//! The table's own invariants, plus the ratchet that keeps it and
//! `docs/04-stdlib-and-builtins.md` from drifting apart.

use super::*;

/// Builtins that have no entry in `docs/04-stdlib-and-builtins.md` yet. This
/// list may only SHRINK: `documentation_debt_only_shrinks` fails both when a new
/// builtin is added without documentation and when an entry here becomes
/// documented but is not removed.
const UNDOCUMENTED: &[&str] = &[
    "frame_reset", "json_set_str", "json_push_str", "sleep_ms", "load_sound",
    "phys3d_add_box_rot", "r3d_make_box_sized", "r3d_make_box_emissive", "r3d_camera_roll",
    "r3d_draw_quat", "r3d_draw_tint", "r3d_draw_shield", "r3d_draw_on_joint", "r3d_joint_dump",
    "r3d_joint_pos", "r3d_anim_play_upper", "r3d_anim_aim_upper", "r3d_anim_blend",
    "r3d_anim_seek_upper", "r3d_pose_bone", "r3d_hide_joint", "r3d_clear_pose",
    "r3d_anim_stop_upper", "r3d_speedlines", "r3d_damage", "r3d_blur", "r3d_viewmodel",
    "play_sound_handle", "play_sound_handle_at", "play_music", "music_volume", "music_stop",
    "play_ambience", "ambience_volume", "ambience_stop", "phys3d_raycast_ex",
    "phys3d_raycast_world", "net_serve", "save_settings", "load_settings", "net_leave",
    "net_player_state", "net_set_meta", "net_player_meta", "net_set_name",
    "net_player_name_len", "net_player_name_char", "net_max_clients", "net_rejected",
    "net_connected", "net_dedicated", "net_cfg_set", "net_cfg_get", "net_set_bot_count",
    "net_set_bot", "net_set_bot_input", "net_set_bot_state", "net_set_bot_alive",
    "net_set_bot_meta", "net_set_bot_name", "net_bot_count", "net_set_object_count",
    "net_set_object", "net_object_count", "net_object_x", "net_object_y", "net_object_z",
    "net_set_object_rot", "net_object_qx", "net_object_qy", "net_object_qz", "net_object_qw",
    "net_set_object_vel", "net_object_vx", "net_object_vy", "net_object_vz", "net_set_fx_count",
    "net_set_fx", "net_fx_count", "net_fx_x", "net_fx_y", "net_fx_z", "net_fx_kind",
    "net_spawn_input_slot", "net_respawn_client", "net_impulse_input_slot", "net_push_impulse",
    "net_respawn_trigger_slot", "net_force_respawn", "net_server_hit_count",
    "net_server_hit_shooter", "net_server_hit_victim", "net_server_hit_x", "net_server_hit_y",
    "net_server_hit_z", "net_server_hits_clear", "net_push_kill", "net_kill_count",
    "net_kill_killer", "net_kill_victim", "net_kills_clear", "net_push_shot", "net_shot_count",
    "net_shot_shooter", "net_shot_field", "net_shot_weapon", "net_shots_clear", "net_push_boom",
    "net_boom_count", "net_boom_source", "net_boom_field", "net_booms_clear",
    "net_projectile_intent", "net_server_projectile_count", "net_server_projectile_shooter",
    "net_server_projectile_kind", "net_server_projectile_ox", "net_server_projectile_oy",
    "net_server_projectile_oz", "net_server_projectile_vx", "net_server_projectile_vy",
    "net_server_projectile_vz", "net_server_projectiles_clear", "net_set_player_meta",
    "net_hit_seq", "input_suppress", "f32_blob", "log", "exp", "atan2", "draw_int",
    "text_width", "play_noise", "audio_volume", "window_fullscreen", "audio_stop", "surface_w",
    "surface_h", "input_char",
];

#[test]
fn names_are_unique() {
    let mut seen = HashSet::new();
    for b in TABLE {
        assert!(seen.insert(b.name), "duplicate table row `{}`", b.name);
    }
    assert_eq!(seen.len(), TABLE.len());
}

#[test]
fn symbols_follow_the_naming_convention() {
    for b in TABLE {
        if b.kind.has_host_fn() {
            assert_eq!(
                b.symbol,
                format!("aurora_{}", b.name),
                "`{}` must be backed by `aurora_{}`",
                b.name,
                b.name
            );
        } else {
            assert_eq!(b.symbol, "", "inline builtin `{}` must have no symbol", b.name);
        }
    }
}

#[test]
fn inline_rows_carry_no_signature() {
    for b in TABLE.iter().filter(|b| b.kind == Kind::Inline) {
        assert!(b.params.is_empty(), "inline builtin `{}` must have no parameters", b.name);
        assert_eq!(b.ret, None, "inline builtin `{}` must have no return type", b.name);
    }
}

/// A `scalar` row is lowered by coercing each argument to its declared type, so
/// a pointer parameter there would be lowered as a bare integer.
#[test]
fn scalar_rows_take_only_scalars() {
    for b in TABLE.iter().filter(|b| b.kind == Kind::Scalar) {
        assert!(
            !b.params.contains(&Ty::Ptr) && b.ret != Some(Ty::Ptr),
            "`{}` takes or returns a pointer, so it cannot use the scalar dispatch",
            b.name
        );
    }
}

#[test]
fn lookup_finds_every_row() {
    for b in TABLE {
        assert_eq!(lookup(b.name), Some(b), "lookup missed `{}`", b.name);
    }
    assert_eq!(lookup("definitely_not_a_builtin"), None);
}

#[test]
fn builtin_names_agree_with_is_builtin() {
    for b in TABLE {
        assert_eq!(
            is_builtin(b.name),
            b.kind.is_aurora_visible(),
            "`{}` ({:?}) is misreported by is_builtin",
            b.name,
            b.kind
        );
    }
    assert!(!is_builtin("definitely_not_a_builtin"));
    let names: HashSet<&str> = builtin_names().iter().copied().collect();
    assert_eq!(names.len(), builtin_names().len(), "duplicate builtin name");
}

#[test]
fn scalar_sig_matches_the_table() {
    for b in TABLE {
        match scalar_sig(b.name) {
            Some((p, r)) => {
                assert_eq!(b.kind, Kind::Scalar);
                assert_eq!(p, b.params);
                assert_eq!(r, b.ret);
            }
            None => assert_ne!(b.kind, Kind::Scalar, "`{}` lost its signature", b.name),
        }
    }
}

// ---------------------------------------------------------------------------
// Documentation ratchet
// ---------------------------------------------------------------------------

const DOC: &str = include_str!("../../../docs/04-stdlib-and-builtins.md");

/// Names documented in `docs/04-stdlib-and-builtins.md`, and the argument counts
/// the docs give for them.
///
/// The reference writes a builtin inside a code span, either alone
/// (`` `net_leave` ``), with its arguments (`` `net_fire(ox,oy,oz, dx,dy,dz, weapon)` ``),
/// or as a slash-joined family sharing a prefix and one argument list
/// (`` `phys3d_x/y/z(h)` ``, `` `json_set_num/str/bool(h, key, v)` ``). Anything
/// that does not resolve to a table row is ignored, so prose is harmless.
fn documented() -> (HashSet<&'static str>, HashMap<&'static str, usize>) {
    let mut names = HashSet::new();
    let mut arity: HashMap<&'static str, usize> = HashMap::new();
    for span in code_spans(DOC) {
        let mut rest = span;
        while let Some(start) = rest.find(|c: char| c.is_ascii_alphabetic() || c == '_') {
            let tail = &rest[start..];
            let end = tail
                .find(|c: char| !(c.is_ascii_alphanumeric() || c == '_' || c == '/'))
                .unwrap_or(tail.len());
            let (group, after) = tail.split_at(end);
            let args = arg_count(after);
            for name in expand_family(group) {
                names.insert(name);
                if let Some(n) = args {
                    arity.insert(name, n);
                }
            }
            rest = after;
        }
    }
    (names, arity)
}

/// The contents of every `` `..` `` span in `md`.
fn code_spans(md: &str) -> Vec<&str> {
    let mut spans = Vec::new();
    let mut rest = md;
    while let Some(open) = rest.find('`') {
        rest = &rest[open + 1..];
        match rest.find('`') {
            Some(close) => {
                spans.push(&rest[..close]);
                rest = &rest[close + 1..];
            }
            None => break,
        }
    }
    spans
}

/// The number of comma-separated arguments in a leading `(..)`, if `s` starts
/// with one. An empty list is zero arguments; an elided one (`(...)`) is unknown.
fn arg_count(s: &str) -> Option<usize> {
    let inner = s.strip_prefix('(')?.split(')').next()?;
    if inner.contains("..") {
        return None;
    }
    if inner.trim().is_empty() {
        return Some(0);
    }
    Some(inner.split(',').count())
}

/// Expand a slash-joined family such as `phys3d_x/y/z` or
/// `phys3d_apply_force/apply_torque` into the table names it refers to. Each
/// alternative is tried whole, then under the first alternative's prefix (up to
/// its last `_`, then up to its first `_`); only names that exist in the table
/// are kept, so an unrelated `a/b` in prose expands to nothing.
fn expand_family(group: &str) -> Vec<&'static str> {
    let mut out = Vec::new();
    let mut parts = group.split('/');
    let Some(head) = parts.next() else { return out };
    if let Some(b) = lookup(head) {
        out.push(b.name);
    }
    let prefixes = [
        &head[..head.rfind('_').map(|i| i + 1).unwrap_or(0)],
        &head[..head.find('_').map(|i| i + 1).unwrap_or(0)],
    ];
    for alt in parts {
        let mut candidates = vec![alt.to_string()];
        for p in prefixes {
            candidates.push(format!("{p}{alt}"));
        }
        for c in candidates {
            if let Some(b) = lookup(&c) {
                out.push(b.name);
                break;
            }
        }
    }
    out
}

#[test]
fn documentation_debt_only_shrinks() {
    let (documented, _) = documented();
    let missing: Vec<&str> =
        builtin_names().iter().copied().filter(|n| !documented.contains(n)).collect();
    let allowed: HashSet<&str> = UNDOCUMENTED.iter().copied().collect();

    let new: Vec<&str> = missing.iter().copied().filter(|n| !allowed.contains(n)).collect();
    assert!(
        new.is_empty(),
        "{} builtin(s) have no entry in docs/04-stdlib-and-builtins.md: {:?}\n\
         Document them, or (only if you must) add them to UNDOCUMENTED.",
        new.len(),
        new
    );

    let missing_set: HashSet<&str> = missing.iter().copied().collect();
    let now_documented: Vec<&str> =
        UNDOCUMENTED.iter().copied().filter(|n| !missing_set.contains(n)).collect();
    assert!(
        now_documented.is_empty(),
        "{:?} are documented now - remove them from UNDOCUMENTED so the debt cannot grow back.",
        now_documented
    );

    let stale: Vec<&str> =
        UNDOCUMENTED.iter().copied().filter(|n| lookup(n).is_none()).collect();
    assert!(stale.is_empty(), "UNDOCUMENTED names no builtin: {stale:?}");
}

/// The bug that started all of this: `net_fire` grew a `weapon` parameter and
/// the documented call kept six arguments. For a builtin whose parameters are
/// all scalars the Aurora-level arity IS the ABI arity, so the docs can be
/// checked against the table directly.
#[test]
fn documented_arity_matches_the_table() {
    let (_, arity) = documented();
    let mut wrong = Vec::new();
    for b in TABLE.iter().filter(|b| b.kind.is_aurora_visible()) {
        if b.kind == Kind::Inline || b.params.contains(&Ty::Ptr) {
            continue; // no ABI arity, or a `str`/array argument spanning two slots
        }
        if let Some(&n) = arity.get(b.name) {
            if n != b.params.len() {
                wrong.push(format!("{} documented with {n} args, table has {}", b.name, b.params.len()));
            }
        }
    }
    assert!(wrong.is_empty(), "docs/04-stdlib-and-builtins.md disagrees with the table:\n{}", wrong.join("\n"));
}
