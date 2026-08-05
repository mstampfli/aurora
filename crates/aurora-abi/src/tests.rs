//! The table's own invariants, plus the ratchet that keeps it and
//! `docs/04-stdlib-and-builtins.md` from drifting apart.

use super::*;

/// Builtins that have no entry in `docs/04-stdlib-and-builtins.md` yet. This
/// list may only SHRINK: `documentation_debt_only_shrinks` fails both when a new
/// builtin is added without documentation and when an entry here becomes
/// documented but is not removed.
const UNDOCUMENTED: &[&str] = &[
    "frame_reset",
    "json_set_str",
    "json_push_str",
    "sleep_ms",
    "phys3d_add_box_rot",
    "r3d_make_box_sized",
    "r3d_make_box_emissive",
    "r3d_camera_roll",
    "r3d_draw_quat",
    "r3d_draw_tint",
    "r3d_draw_shield",
    "r3d_draw_on_joint",
    "r3d_anim_play_upper",
    "r3d_anim_aim_upper",
    "r3d_anim_blend",
    "r3d_anim_seek_upper",
    "r3d_pose_bone",
    "r3d_clear_pose",
    "r3d_anim_stop_upper",
    "r3d_speedlines",
    "r3d_damage",
    "r3d_blur",
    "r3d_viewmodel",
    "ambience_volume",
    "ambience_stop",
    "net_serve",
    "save_settings",
    "load_settings",
    "net_leave",
    "net_player_state",
    "net_set_name",
    "net_player_name_len",
    "net_player_name_char",
    "net_max_clients",
    "net_rejected",
    "net_connected",
    "net_dedicated",
    "net_cfg_set",
    "net_cfg_get",
    "net_set_bot_count",
    "net_set_bot",
    "net_set_bot_input",
    "net_set_bot_state",
    "net_set_bot_alive",
    "net_set_bot_meta",
    "net_set_bot_name",
    "net_bot_count",
    "net_set_object_count",
    "net_set_object",
    "net_object_count",
    "net_object_x",
    "net_object_y",
    "net_object_z",
    "net_set_object_rot",
    "net_object_qx",
    "net_object_qy",
    "net_object_qz",
    "net_object_qw",
    "net_set_object_vel",
    "net_object_vx",
    "net_object_vy",
    "net_object_vz",
    "net_set_fx_count",
    "net_set_fx",
    "net_fx_count",
    "net_fx_x",
    "net_fx_y",
    "net_fx_z",
    "net_fx_kind",
    "net_spawn_input_slot",
    "net_respawn_client",
    "net_impulse_input_slot",
    "net_push_impulse",
    "net_respawn_trigger_slot",
    "net_force_respawn",
    "net_server_hit_count",
    "net_server_hit_shooter",
    "net_server_hit_victim",
    "net_server_hit_x",
    "net_server_hit_y",
    "net_server_hit_z",
    "net_server_hits_clear",
    "net_push_kill",
    "net_kill_count",
    "net_kill_killer",
    "net_kill_victim",
    "net_kills_clear",
    "net_push_shot",
    "net_shot_count",
    "net_shot_shooter",
    "net_shot_field",
    "net_shot_weapon",
    "net_shots_clear",
    "net_push_boom",
    "net_boom_count",
    "net_boom_source",
    "net_boom_field",
    "net_booms_clear",
    "net_projectile_intent",
    "net_server_projectile_count",
    "net_server_projectile_shooter",
    "net_server_projectile_kind",
    "net_server_projectile_ox",
    "net_server_projectile_oy",
    "net_server_projectile_oz",
    "net_server_projectile_vx",
    "net_server_projectile_vy",
    "net_server_projectile_vz",
    "net_server_projectiles_clear",
    "net_set_player_meta",
    "net_hit_seq",
    "f32_blob",
    "log",
    "exp",
    "draw_int",
    "play_noise",
    "audio_volume",
    "window_fullscreen",
    "audio_stop",
    "surface_w",
    "surface_h",
    "input_char",
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
            assert_eq!(
                b.symbol, "",
                "inline builtin `{}` must have no symbol",
                b.name
            );
        }
    }
}

#[test]
fn inline_rows_carry_no_signature() {
    for b in TABLE.iter().filter(|b| b.kind == Kind::Inline) {
        assert!(
            b.params.is_empty(),
            "inline builtin `{}` must have no parameters",
            b.name
        );
        assert_eq!(
            b.ret, None,
            "inline builtin `{}` must have no return type",
            b.name
        );
    }
}

/// A `scalar` row is lowered by coercing each argument to its declared type, so
/// a pointer parameter there would be lowered as a bare integer.
#[test]
fn scalar_rows_take_only_scalars() {
    for b in TABLE.iter().filter(|b| b.kind == Kind::Scalar) {
        assert!(
            !b.params.contains(&Ty::Ptr) && !matches!(b.ret, Some(Ty::Ptr) | Some(Ty::Str)),
            "`{}` takes or returns a pointer, so it cannot use the scalar dispatch",
            b.name
        );
    }
}

/// `Str` is a RESULT convention (a caller-allocated out-pointer), never a slot.
/// Only the text dispatch allocates that slot, so only a `text` row may use it.
#[test]
fn str_is_only_a_text_rows_return() {
    for b in TABLE {
        assert!(
            !b.params.contains(&Ty::Str),
            "`{}` has a Str parameter slot",
            b.name
        );
        if b.ret == Some(Ty::Str) {
            assert_eq!(
                b.kind,
                Kind::Text,
                "`{}` returns Str but is not a text row",
                b.name
            );
        }
    }
}

/// The text dispatch reads a `Ptr` slot as the first half of one Aurora `str`
/// argument and consumes the `I64` after it as its length. A `Ptr` that is last,
/// or followed by anything else, would silently eat the next argument.
#[test]
fn text_rows_pair_each_pointer_with_a_length() {
    for b in TABLE.iter().filter(|b| b.kind == Kind::Text) {
        let mut i = 0;
        while i < b.params.len() {
            if b.params[i] == Ty::Ptr {
                assert_eq!(
                    b.params.get(i + 1),
                    Some(&Ty::I64),
                    "`{}`: a str argument must be `Ptr, I64` (data, length)",
                    b.name
                );
                i += 1;
            }
            i += 1;
        }
        assert_eq!(
            b.arity(),
            Some(b.params.len() - b.params.iter().filter(|t| **t == Ty::Ptr).count())
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
    let missing: Vec<&str> = builtin_names()
        .iter()
        .copied()
        .filter(|n| !documented.contains(n))
        .collect();
    let allowed: HashSet<&str> = UNDOCUMENTED.iter().copied().collect();

    let new: Vec<&str> = missing
        .iter()
        .copied()
        .filter(|n| !allowed.contains(n))
        .collect();
    assert!(
        new.is_empty(),
        "{} builtin(s) have no entry in docs/04-stdlib-and-builtins.md: {:?}\n\
         Document them, or (only if you must) add them to UNDOCUMENTED.",
        new.len(),
        new
    );

    let missing_set: HashSet<&str> = missing.iter().copied().collect();
    let now_documented: Vec<&str> = UNDOCUMENTED
        .iter()
        .copied()
        .filter(|n| !missing_set.contains(n))
        .collect();
    assert!(
        now_documented.is_empty(),
        "{:?} are documented now - remove them from UNDOCUMENTED so the debt cannot grow back.",
        now_documented
    );

    let stale: Vec<&str> = UNDOCUMENTED
        .iter()
        .copied()
        .filter(|n| lookup(n).is_none())
        .collect();
    assert!(stale.is_empty(), "UNDOCUMENTED names no builtin: {stale:?}");
}

/// The bug that started all of this: `net_fire` grew a `weapon` parameter and
/// the documented call kept six arguments. `Builtin::arity` is the number of
/// arguments a call site passes, for every kind whose call the table fully
/// describes, so the docs can be checked against the table directly.
#[test]
fn documented_arity_matches_the_table() {
    let (_, doc_arity) = documented();
    let mut wrong = Vec::new();
    let mut checked = 0;
    for b in TABLE.iter().filter(|b| b.kind.is_aurora_visible()) {
        let (Some(want), Some(&n)) = (b.arity(), doc_arity.get(b.name)) else {
            continue;
        };
        checked += 1;
        if n != want {
            wrong.push(format!(
                "{} documented with {n} args, table has {want}",
                b.name
            ));
        }
    }
    assert!(
        wrong.is_empty(),
        "docs/04-stdlib-and-builtins.md disagrees with the table:\n{}",
        wrong.join("\n")
    );
    // A parser that silently stopped matching would pass vacuously.
    assert!(
        checked > 100,
        "only {checked} builtins had a documented argument list"
    );
}

/// The libm calls are exactly these ten, and the SHAPE identifies them.
///
/// `aurora-codegen` used to carry this list by hand to decide which builtins are
/// host calls into libm, plus a second list deciding which take two arguments.
/// It selects on the row now - `special`, every parameter `F64`, returns `F64` -
/// so adding a transcendental is one table row rather than a row and two edits
/// in a backend, and forgetting the edits would have compiled the call at the
/// wrong float width with no diagnostic.
///
/// This is where the membership is STATED, because a predicate that quietly
/// starts matching a new row is the failure the name list at least made visible.
/// The names here are the manual's, not the code's.
#[test]
fn transcendental_rows_are_exactly_the_libm_calls() {
    let expected = [
        "sin", "cos", "tan", "pow", "log", "exp", "atan2", "acos", "asin", "atan",
    ];
    let mut found: Vec<&str> = builtin_names()
        .iter()
        .copied()
        .filter(|n| {
            lookup(n).is_some_and(|b| {
                b.kind == Kind::Special
                    && b.ret == Some(Ty::F64)
                    && !b.params.is_empty()
                    && b.params.iter().all(|t| *t == Ty::F64)
            })
        })
        .collect();
    found.sort_unstable();
    let mut want: Vec<&str> = expected.to_vec();
    want.sort_unstable();
    assert_eq!(
        found, want,
        "the `special` + all-F64 + returns-F64 shape no longer picks out exactly \
         the libm calls. aurora-codegen lowers whatever this matches as a host \
         call into libm, so a new row of this shape would be lowered as one - \
         either give it a different signature or teach the backend about it."
    );

    // And each one's arity comes from the row, which is what the backend now
    // reads instead of a second hand-written list.
    for n in expected {
        let b = lookup(n).expect("declared above");
        let want_args = if n == "pow" || n == "atan2" { 2 } else { 1 };
        assert_eq!(
            b.params.len(),
            want_args,
            "`{n}` takes {want_args} argument(s)"
        );
    }
}
