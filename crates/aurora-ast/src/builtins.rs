//! The names of Aurora's runtime builtins.
//!
//! These are not user-defined functions: they never appear as `fn` items, and
//! the backend lowers each one to a native runtime call. They live here, in the
//! crate every front-end pass already depends on, so that the backend (which
//! lowers them) and the front end (which must not report them as unresolved
//! names) share ONE list. A builtin missing from this list would be reported as
//! an unknown function by `aurorac check`; a stale extra entry would let a real
//! typo through, so this is the single source of truth for both.

/// Builtin function names (handled specially, never user-defined / captured).
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
    "r3d_draw_on_joint", "r3d_joint_dump", "r3d_joint_pos", "r3d_draw_shield",
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

/// Is `name` a runtime builtin?
pub fn is_builtin(name: &str) -> bool {
    BUILTINS.contains(&name)
}
