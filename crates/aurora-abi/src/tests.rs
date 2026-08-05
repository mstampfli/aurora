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

/// An inline row declares no CALL - no parameters, and (asserted next to the
/// symbol check above) no symbol, because codegen emits these rather than
/// calling a host function.
///
/// Its `ret` is a different thing and IS allowed: the type of the expression the
/// builtin produces, which is what the checker needs. That distinction used to
/// be missing - every inline row said `void` - so `char_at`'s result typed as
/// unknown, `str + char_at(..)` type-checked leniently, and codegen emitted a
/// pointer dereference of an integer. A segfault from a program `check` passed.
///
/// The four that stay `None` are genuinely overloaded: `abs`, `min`, `max` and
/// `clamp` each answer int OR float depending on their argument, and a single
/// declared type would be a guess. Unknown is the honest answer for those.
#[test]
fn inline_rows_declare_no_call() {
    for b in TABLE.iter().filter(|b| b.kind == Kind::Inline) {
        assert!(
            b.params.is_empty(),
            "inline builtin `{}` must have no parameters",
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
            !b.params.contains(&Ty::Ptr)
                && !b.params.contains(&Ty::Str)
                && !b.params.contains(&Ty::Arr)
                && !matches!(b.ret, Some(Ty::Ptr) | Some(Ty::Str) | Some(Ty::Arr)),
            "`{}` takes or returns something that is not a scalar, so it cannot use the scalar dispatch",
            b.name
        );
    }
}

/// A string or an array argument is SPELLED, never encoded as a raw pointer.
///
/// This is the invariant that stops the bug it replaced. `Str` used to be a
/// result convention only, so a string ARGUMENT was written `[Ptr, I64]` - and
/// so was an array, and so was an out-pointer next to a length. The table could
/// not tell them apart, which meant no backend could either: lowering a call
/// that takes a string needs to know it takes a string, so each backend carried
/// a HAND-WRITTEN LIST of which builtins those were. A row added without also
/// being added to the list compiled to "cannot find function `image_open`",
/// which is how this was found.
///
/// With `Str` and `Arr` in the table the lowering follows from the row and the
/// lists are gone. This keeps them gone: a bare `Ptr` parameter on a row Aurora
/// can call is allowed only for the handful of rows where a pointer is really
/// what the call site passes - an out-pointer it allocates, or a closure - and
/// that list may only ever SHRINK.
#[test]
fn a_string_or_array_argument_is_spelled_rather_than_encoded() {
    // Each of these takes a genuine machine pointer, not an Aurora value:
    //   net_recv, json_str, json_key, json_to_str - write their result through a
    //     caller-allocated out-pointer, so the pointer IS the argument.
    //   net_sim, net_serve, par_for - take a closure as a code pointer and an
    //     environment pointer.
    const RAW_POINTER_IS_THE_ARGUMENT: &[&str] = &[
        "net_recv",
        "json_str",
        "json_key",
        "json_to_str",
        "net_sim",
        "net_serve",
        "par_for",
        "read_file",
    ];
    for b in TABLE.iter().filter(|b| b.kind.is_aurora_visible()) {
        if b.params.contains(&Ty::Ptr) {
            assert!(
                RAW_POINTER_IS_THE_ARGUMENT.contains(&b.name),
                "`{}` declares a bare `Ptr` parameter. If that is a string say \
                 `Str`, if it is an array say `Arr` - spelling it out is what \
                 lets every backend lower the call without a list of names",
                b.name
            );
        }
    }
    // And the allowlist may only shrink: a name on it that no longer needs to be
    // is the reason a list like this rots into a permanent exemption.
    for name in RAW_POINTER_IS_THE_ARGUMENT {
        let b = lookup(name).expect("an allowlisted row must exist");
        assert!(
            b.params.contains(&Ty::Ptr),
            "`{name}` no longer takes a raw pointer - take it off the list"
        );
    }
}

/// `Str` as a RESULT is a caller-allocated out-pointer, and only the text
/// dispatch allocates that slot, so only a `text` row may return one.
#[test]
fn only_a_text_row_returns_str() {
    for b in TABLE {
        if b.ret == Some(Ty::Str) {
            // An INLINE row may also say `Str`, because it declares no call: no
            // host function is imported for it, so `abi_params` never turns the
            // result into a leading out-pointer and no slot is involved. What it
            // says is the type of the expression, for the checker - `str(n)` and
            // `substr(s, i, n)` produce strings and the compiler should know it.
            assert!(
                b.kind == Kind::Text || b.kind == Kind::Inline,
                "`{}` returns Str but is neither a text nor an inline row",
                b.name
            );
        }
        assert_ne!(b.ret, Some(Ty::Arr), "`{}` cannot return an array", b.name);
    }
}

/// A `Str` or `Arr` argument occupies two ABI slots - the address and the
/// length - and `abi_params` is the one place that expansion happens. This pins
/// it, and pins `arity` counting such an argument ONCE.
#[test]
fn a_two_slot_argument_expands_to_exactly_its_slots() {
    for b in TABLE {
        let want: Vec<Ty> = b
            .params
            .iter()
            .flat_map(|t| match t {
                Ty::Str | Ty::Arr => vec![Ty::Ptr, Ty::I64],
                other => vec![*other],
            })
            .collect();
        let got = b.abi_params();
        let got = if b.ret == Some(Ty::Str) {
            assert_eq!(
                got.first(),
                Some(&Ty::Ptr),
                "`{}` returns Str, so its slots must lead with the out-pointer",
                b.name
            );
            got[1..].to_vec()
        } else {
            got
        };
        assert_eq!(
            got, want,
            "`{}` does not expand to its declared slots",
            b.name
        );
        if let Some(n) = b.arity() {
            assert_eq!(
                n,
                b.params.iter().filter(|t| **t != Ty::Ptr).count(),
                "`{}` counts a two-slot argument more than once",
                b.name
            );
        }
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
