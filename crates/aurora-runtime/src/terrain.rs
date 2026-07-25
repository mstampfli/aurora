//! The `terrain_*` builtins: a heightmap world that is drawn, walked on, and
//! queried from ONE heightfield.
//!
//! The loaded [`Heightfield`] lives here, behind an [`Arc`], and the same `Arc`
//! is handed to the renderer. The physics collider is built from it. So the
//! surface a player sees, the surface the character controller stands on, and
//! the number `terrain_height` returns are the same triangles by construction,
//! not by three implementations agreeing.
//!
//! The heightfield deliberately does NOT live behind the renderer's device: a
//! headless physics-only program never opens a window, and `terrain_height` and
//! `terrain_collider` still have to work there.
//!
//! # File format (`.aterr`)
//!
//! Little-endian throughout, `24 + dim*dim*4` bytes:
//!
//! | offset | size | field |
//! |---|---|---|
//! | 0 | 8 | magic, the ASCII bytes `AURTERR1` |
//! | 8 | 4 | `u32` `dim`, samples per side; must be `2^k + 1`, 5..=4097 |
//! | 12 | 4 | `f32` `spacing`, world units between samples (> 0) |
//! | 16 | 4 | `f32` `origin_x`, world X of sample column 0 |
//! | 20 | 4 | `f32` `origin_z`, world Z of sample row 0 |
//! | 24 | `dim*dim*4` | `f32` heights, row-major (`row * dim + col`) |
//!
//! Column indices run along +X and row indices along +Z, so sample `(row, col)`
//! is at world `(origin_x + col*spacing, height, origin_z + row*spacing)` and a
//! height is a world Y in the same units as everything else.

use std::cell::{Cell, RefCell};
use std::sync::Arc;

use aurora_render3d::Heightfield;

thread_local! {
    /// The one loaded terrain. `None` until `terrain_load`/`terrain_generate`.
    static TERRAIN: RefCell<Option<Arc<Heightfield>>> = const { RefCell::new(None) };
    /// Terrain albedo, handed to the renderer with the heightfield on each draw.
    static COLOR: Cell<[f32; 3]> = const { Cell::new([0.32, 0.40, 0.24]) };
}

/// Take a `str` argument's (pointer, length) pair as a path.
///
/// # Safety
/// `ptr` must point to `len` initialized bytes, or be null with `len <= 0`.
unsafe fn path_arg(ptr: *const u8, len: i64) -> String {
    if ptr.is_null() || len <= 0 {
        return String::new();
    }
    let s = unsafe { std::slice::from_raw_parts(ptr, len as usize) };
    String::from_utf8_lossy(s).into_owned()
}

/// Install a heightfield, replacing any previous one. Reports failure loudly,
/// because a terrain that silently did not load looks exactly like a terrain
/// that is simply flat.
///
/// The renderer picks it up on the next `terrain_draw` rather than here: at load
/// time there may be no scene to give it to yet (no window has been opened), and
/// handing it over then would drop it on the floor.
fn install(built: Result<Heightfield, String>) -> i64 {
    match built {
        Ok(field) => {
            TERRAIN.with(|t| *t.borrow_mut() = Some(Arc::new(field)));
            1
        }
        Err(e) => {
            eprintln!("aurora: {e}");
            0
        }
    }
}

/// Run `f` on the loaded heightfield, or return `default` when none is loaded.
fn with_field<R>(default: R, f: impl FnOnce(&Heightfield) -> R) -> R {
    TERRAIN.with(|t| match t.borrow().as_ref() {
        Some(field) => f(field),
        None => default,
    })
}

/// Build a procedural heightfield: `dim` x `dim` samples `spacing` apart,
/// centred on the world origin, heights spanning `[0, amplitude]`.
///
/// `dim` must be `2^k + 1` (5, 9, 17, ... 4097). Deterministic for a given
/// `(seed, dim, spacing, amplitude)`. Returns 1 on success, 0 on failure.
#[no_mangle]
pub extern "C" fn aurora_terrain_generate(
    seed: i64,
    dim: i64,
    spacing: f64,
    amplitude: f64,
) -> i64 {
    if !(0..=i64::from(u32::MAX)).contains(&dim) {
        eprintln!("aurora: terrain: dim {dim} out of range");
        return 0;
    }
    install(Heightfield::generate(
        seed,
        dim as u32,
        spacing as f32,
        amplitude as f32,
    ))
}

/// Load an `.aterr` heightfield file. Returns 1 on success, 0 on failure.
///
/// # Safety
/// `ptr` must point to `len` initialized bytes (an Aurora `str`).
#[no_mangle]
pub unsafe extern "C" fn aurora_terrain_load(ptr: *const u8, len: i64) -> i64 {
    let path = unsafe { path_arg(ptr, len) };
    install(Heightfield::load(&path))
}

/// Write the loaded heightfield as an `.aterr` file. Returns 1 on success, 0 on
/// failure (including when no terrain is loaded).
///
/// # Safety
/// `ptr` must point to `len` initialized bytes (an Aurora `str`).
#[no_mangle]
pub unsafe extern "C" fn aurora_terrain_save(ptr: *const u8, len: i64) -> i64 {
    let path = unsafe { path_arg(ptr, len) };
    with_field(0, |f| match f.save(&path) {
        Ok(()) => 1,
        Err(e) => {
            eprintln!("aurora: {e}");
            0
        }
    })
}

/// Set the terrain albedo (0..1 per channel). Takes effect on the next
/// `terrain_draw`, so it may be called before the terrain or the window exists.
#[no_mangle]
pub extern "C" fn aurora_terrain_color(r: f64, g: f64, b: f64) {
    COLOR.with(|c| c.set([r as f32, g as f32, b as f32]));
}

/// Queue the terrain for this frame, at the level of detail the current camera
/// calls for. Goes between `r3d_begin` and `r3d_present`, like `r3d_draw`.
/// No-op when no terrain is loaded.
#[no_mangle]
pub extern "C" fn aurora_terrain_draw() {
    let Some(field) = TERRAIN.with(|t| t.borrow().clone()) else {
        return;
    };
    aurora_window::imm_terrain_draw(field, COLOR.with(|c| c.get()));
}

/// Surface height at world `(x, z)`.
///
/// Interpolated across the same triangles the collider uses, so it agrees with a
/// downward raycast onto the terrain. Outside the footprint it clamps to the
/// nearest edge sample; with no terrain loaded it is 0.
#[no_mangle]
pub extern "C" fn aurora_terrain_height(x: f64, z: f64) -> f64 {
    with_field(0.0, |f| f.height_at(x, z))
}

/// Register the terrain with the 3D physics world (call after `phys3d_init`) and
/// return its body handle, or -1 if there is no terrain or no physics world.
#[no_mangle]
pub extern "C" fn aurora_terrain_collider() -> i64 {
    with_field(-1, crate::phys3d::add_heightfield)
}

/// Samples along one side of the loaded terrain, or 0 if none is loaded.
#[no_mangle]
pub extern "C" fn aurora_terrain_size() -> i64 {
    with_field(0, |f| f.dim() as i64)
}

/// World distance between adjacent samples, or 0 if no terrain is loaded.
#[no_mangle]
pub extern "C" fn aurora_terrain_spacing() -> f64 {
    with_field(0.0, |f| f.spacing() as f64)
}

/// World X of sample column 0 (the terrain's -X border), or 0 if none.
#[no_mangle]
pub extern "C" fn aurora_terrain_origin_x() -> f64 {
    with_field(0.0, |f| f.origin_x() as f64)
}

/// World Z of sample row 0 (the terrain's -Z border), or 0 if none.
#[no_mangle]
pub extern "C" fn aurora_terrain_origin_z() -> f64 {
    with_field(0.0, |f| f.origin_z() as f64)
}

#[cfg(test)]
mod tests;
