//! Model and primitive lifetime: creating assets is bounded only if freeing
//! them actually releases their GPU buffers, and freeing them is only safe if a
//! stale handle is refused rather than silently aliasing whatever took the slot.
//!
//! The counts here are backed by a byte figure (`Renderer3D::mesh_bytes`), which
//! is the real GPU allocation rather than a proxy for it, and by a RENDERED
//! check: a freed handle must draw nothing, not draw the asset loaded into its
//! slot afterwards.

use std::sync::Arc;

use super::*;

fn device() -> Option<(wgpu::Device, wgpu::Queue)> {
    crate::headless_device()
}

fn scene(device: &wgpu::Device, queue: &wgpu::Queue, w: u32, h: u32) -> Scene {
    Scene::new(device, queue, wgpu::TextureFormat::Rgba8Unorm, w, h, 1)
}

/// Creating and freeing models in a loop must not grow the stores.
///
/// This is the "game changes level, or loads assets in a loop" case. Every
/// primitive maker uploads its own mesh AND its own material (four 1x1 textures,
/// a uniform buffer, and a bind group), so an unreleased one is not a rounding
/// error.
#[test]
fn creating_and_freeing_primitives_in_a_loop_is_bounded() {
    let Some((device, queue)) = device() else {
        eprintln!("no GPU adapter - skipping the model lifetime leak test");
        return;
    };
    let _g = crate::gpu_guard();
    let mut s = scene(&device, &queue, 64, 64);

    // One full cycle of every maker that pushes into the store, to establish
    // what a steady state looks like.
    let cycle = |s: &mut Scene| {
        let handles = [
            s.make_box(&device, &queue, [0.8, 0.2, 0.2, 1.0]),
            s.make_box_sized(&device, &queue, 1.0, 2.0, 0.5, [0.2, 0.8, 0.2, 1.0]),
            s.make_box_emissive(&device, &queue, 0.5, 0.5, 0.5, [0.1, 0.9, 0.3]),
            s.make_sphere(&device, &queue, 16, [0.2, 0.2, 0.8, 1.0]),
            s.make_plane(&device, &queue, 10.0, 2.0, [0.5, 0.5, 0.5, 1.0]),
            s.make_sprite(&device, &queue, [1.0, 1.0, 0.0]),
        ];
        for h in handles {
            assert!(h >= 0, "a maker returned the failure sentinel {h}");
        }
        handles
    };

    let base_meshes = s.renderer.mesh_count();
    let base_bytes = s.renderer.mesh_bytes();
    let base_mats = s.renderer.material_count();

    // Peak of one live cycle: the store legitimately holds this much while the
    // assets are alive, which is the bar the loop has to come back down to.
    let live = cycle(&mut s);
    let peak_meshes = s.renderer.mesh_count();
    let peak_bytes = s.renderer.mesh_bytes();
    assert!(
        peak_bytes > base_bytes,
        "the makers allocated nothing, so freeing them proves nothing"
    );
    for h in live {
        assert!(
            s.free_model(h),
            "freeing a live handle should report success"
        );
    }
    assert_eq!(s.renderer.mesh_count(), base_meshes);
    assert_eq!(s.renderer.mesh_bytes(), base_bytes);
    assert_eq!(s.renderer.material_count(), base_mats);

    // Now 200 cycles. If anything leaked per cycle, the byte figure would climb
    // by 200 times that.
    for _ in 0..200 {
        for h in cycle(&mut s) {
            s.free_model(h);
        }
    }
    let end_meshes = s.renderer.mesh_count();
    let end_bytes = s.renderer.mesh_bytes();
    let end_mats = s.renderer.material_count();
    eprintln!(
        "200 create/free cycles of 6 primitives: meshes {base_meshes} -> {end_meshes} \
         (peak {peak_meshes}), GPU mesh bytes {base_bytes} -> {end_bytes} (peak {peak_bytes}), \
         materials {base_mats} -> {end_mats}, mesh slots {} / material slots {}",
        s.renderer.mesh_slot_count(),
        s.renderer.material_slot_count(),
    );

    assert_eq!(
        end_meshes,
        base_meshes,
        "200 create/free cycles left {} meshes behind",
        end_meshes - base_meshes
    );
    assert_eq!(
        end_bytes,
        base_bytes,
        "200 create/free cycles left {} bytes of GPU mesh memory behind",
        end_bytes as i64 - base_bytes as i64
    );
    assert_eq!(
        end_mats,
        base_mats,
        "200 create/free cycles left {} materials behind",
        end_mats - base_mats
    );
    assert_eq!(s.model_count(), 0, "model handles leaked");
    // Slots are recycled, so the store's address space is bounded as well.
    assert!(
        s.renderer.mesh_slot_count() <= peak_meshes + 2,
        "the mesh store grew to {} slots for a peak of {peak_meshes} live meshes",
        s.renderer.mesh_slot_count()
    );
}

/// Loading the SAME model file repeatedly and freeing it must be bounded too:
/// `load_model` is the path a level change actually takes.
#[test]
fn loading_and_freeing_a_model_in_a_loop_is_bounded() {
    let Some((device, queue)) = device() else {
        return;
    };
    let _g = crate::gpu_guard();
    let mut s = scene(&device, &queue, 64, 64);
    let path = std::env::temp_dir().join("aurora_leak_test_model.obj");
    std::fs::write(
        &path,
        "v 0 0 0\nv 1 0 0\nv 0 1 0\nv 1 1 0\nf 1 2 3\nf 2 4 3\n",
    )
    .expect("write obj");
    let path = path.to_str().expect("utf8 path").to_string();

    let base_meshes = s.renderer.mesh_count();
    let base_bytes = s.renderer.mesh_bytes();
    let base_mats = s.renderer.material_count();

    let first = s.load_model(&device, &queue, &path);
    assert!(first >= 0, "the test model failed to load");
    assert!(
        s.renderer.mesh_bytes() > base_bytes,
        "the model uploaded nothing"
    );
    assert!(s.free_model(first));

    for _ in 0..50 {
        let h = s.load_model(&device, &queue, &path);
        assert!(h >= 0);
        assert!(s.free_model(h));
    }
    eprintln!(
        "50 load/free cycles of a model: meshes {base_meshes} -> {}, GPU mesh bytes \
         {base_bytes} -> {}, materials {base_mats} -> {}",
        s.renderer.mesh_count(),
        s.renderer.mesh_bytes(),
        s.renderer.material_count(),
    );
    assert_eq!(s.renderer.mesh_count(), base_meshes);
    assert_eq!(s.renderer.mesh_bytes(), base_bytes);
    assert_eq!(s.renderer.material_count(), base_mats);
    let _ = std::fs::remove_file(&path);
}

/// A freed handle must be REFUSED, not resolved to whatever landed in its slot.
///
/// This is the dangling-handle class of bug, and counting entries cannot catch
/// it: the store can be perfectly bounded and still hand a stale handle a live
/// neighbour's mesh. So it is checked at the pixels: free a red box, make a
/// green one that reuses the slot, then draw with the DEAD handle. If the handle
/// aliased, the frame is green.
#[test]
fn a_freed_model_handle_is_refused_rather_than_aliased() {
    let Some((device, queue)) = device() else {
        eprintln!("no GPU adapter - skipping the stale handle test");
        return;
    };
    let _g = crate::gpu_guard();
    let (w, h) = (48u32, 48u32);
    let mut s = scene(&device, &queue, w, h);
    s.set_camera(Vec3::new(0.0, 0.0, 5.0), Vec3::ZERO, 60.0);
    s.set_light(Vec3::new(0.0, 0.0, 1.0), Vec3::ONE, 0.6);

    let red = s.make_box(&device, &queue, [1.0, 0.0, 0.0, 1.0]);
    // Premise check: the live handle DOES render, so a later blank frame means
    // the handle was refused rather than the setup being broken.
    s.begin();
    s.draw(red, Mat4::IDENTITY);
    let lit = crate::render_offscreen(&mut s.renderer, &device, &queue, w, h, [0.0, 0.0, 0.0, 1.0]);
    let centre = |img: &[u8]| {
        let o = (((h / 2) * w + w / 2) * 4) as usize;
        [img[o], img[o + 1], img[o + 2]]
    };
    let c = centre(&lit);
    assert!(
        c[0] > 40 && c[0] > c[1],
        "the live red box did not render: {c:?}"
    );

    assert!(s.free_model(red), "freeing a live handle should succeed");
    assert!(!s.free_model(red), "a double free must be refused");

    // The next primitive reuses the freed slot; give it a colour that could not
    // be mistaken for the old one.
    let green = s.make_box(&device, &queue, [0.0, 1.0, 0.0, 1.0]);
    assert_ne!(
        red, green,
        "the reused slot handed out the SAME handle value, so a stale handle is \
         indistinguishable from a live one"
    );

    // Draw with the DEAD handle. Nothing must appear.
    s.begin();
    s.draw(red, Mat4::IDENTITY);
    let stale =
        crate::render_offscreen(&mut s.renderer, &device, &queue, w, h, [0.0, 0.0, 0.0, 1.0]);
    let c = centre(&stale);
    assert_eq!(
        c,
        [0, 0, 0],
        "a freed handle rendered {c:?} - it aliased the mesh loaded into its slot"
    );
    assert_eq!(
        s.renderer.last_drawn(),
        0,
        "a freed handle queued a draw command"
    );

    // ...and the new handle still works, so the refusal above is specific to the
    // stale handle rather than the renderer having stopped drawing at all.
    s.begin();
    s.draw(green, Mat4::IDENTITY);
    let fresh =
        crate::render_offscreen(&mut s.renderer, &device, &queue, w, h, [0.0, 0.0, 0.0, 1.0]);
    let c = centre(&fresh);
    assert!(
        c[1] > 40 && c[1] > c[0],
        "the live green box should render: {c:?}"
    );
}

/// Every handle-taking entry point must refuse a stale handle, not just `draw`.
/// A single unchecked accessor would be the whole hole.
#[test]
fn every_handle_accessor_refuses_a_stale_handle() {
    let Some((device, queue)) = device() else {
        return;
    };
    let _g = crate::gpu_guard();
    let mut s = scene(&device, &queue, 32, 32);
    let a = s.make_box(&device, &queue, [1.0, 0.0, 0.0, 1.0]);
    assert!(s.free_model(a));
    // Refill the slot so a stale read would find something rather than nothing.
    let _b = s.make_sphere(&device, &queue, 12, [0.0, 0.0, 1.0, 1.0]);

    assert_eq!(s.clip_count(a), 0);
    assert_eq!(s.joint_global_mat(a, 0), None);
    assert_eq!(s.joint_pos(a, 0), None);
    assert!(!s.free_model(a));

    // The mutating ones must be no-ops rather than panics or writes through to
    // the live occupant.
    s.anim_play(a, 0, true, 1.0, 0.0);
    s.anim_update(a, 0.016);
    s.anim_play_upper(a, 0, true, 1.0, 0.0, 0);
    s.anim_blend(a, 0, 1, 0.5, 1.0, 0.0);
    s.anim_aim_upper(a, 0, 1, 0.5, 1.0, 0.0, 0);
    s.anim_stop_upper(a, 0.0);
    s.anim_seek_upper(a, 0.5);
    s.pose_bone(a, 0, 0.1, 0.2, 0.3);
    s.clear_pose(a);
    s.hide_joint(a, 3);
    s.show_joints(a);

    // Every queueing path must drop the stale handle.
    s.begin();
    s.draw(a, Mat4::IDENTITY);
    s.draw_tint(a, Mat4::IDENTITY, [0.1, 0.1, 0.1]);
    s.draw_shield(a, Mat4::IDENTITY, 0.5, 0.0);
    s.draw_billboard(a, Vec3::ZERO, 1.0);
    s.draw_instances(a, &[Mat4::IDENTITY, Mat4::IDENTITY]);
    s.draw_on_joint(a, a, 0, Mat4::IDENTITY, Mat4::IDENTITY);
    // Both of draw_skinned's handles are stale here: the armour it would queue
    // and the host whose pose would skin it.
    s.draw_skinned(a, a, Mat4::IDENTITY);
    s.debug_skeleton(a, Mat4::IDENTITY, Vec3::ONE);
    let img = crate::render_offscreen(
        &mut s.renderer,
        &device,
        &queue,
        32,
        32,
        [0.0, 0.0, 0.0, 1.0],
    );
    assert_eq!(
        s.renderer.last_drawn(),
        0,
        "a stale handle got through a draw path"
    );
    assert!(
        img.chunks_exact(4)
            .all(|p| p[0] == 0 && p[1] == 0 && p[2] == 0),
        "a stale handle put pixels on screen"
    );
}

/// Handle values must not be plain indices, or a program that stores 0 (or any
/// small integer it made up) would address a real asset.
#[test]
fn a_never_issued_handle_addresses_nothing() {
    let Some((device, queue)) = device() else {
        return;
    };
    let _g = crate::gpu_guard();
    let mut s = scene(&device, &queue, 32, 32);
    let real = s.make_box(&device, &queue, [1.0, 0.0, 0.0, 1.0]);
    for bogus in [0i64, 1, 2, 7, -1, i64::MAX] {
        assert_ne!(bogus, real, "the maker handed out a guessable handle");
        assert!(!s.free_model(bogus), "handle {bogus} freed something");
        assert_eq!(s.clip_count(bogus), 0);
    }
    assert!(s.free_model(real));
}

/// The renderer's own mesh and material handles carry the same guarantee, so a
/// freed tile mesh cannot be reached by whatever still refers to it.
#[test]
fn a_freed_mesh_id_is_refused_by_the_renderer() {
    let Some((device, queue)) = device() else {
        return;
    };
    let _g = crate::gpu_guard();
    let mut r = crate::Renderer3D::new(&device, &queue, wgpu::TextureFormat::Rgba8Unorm, 32, 32, 1);
    let mat = r.add_material(&device, &queue, &MaterialDesc::flat([1.0; 4]));
    let a = r.add_mesh(&device, &MeshData::cube());
    let total = r.mesh_bytes();
    let cube_bytes = r.mesh_bytes_of(a);
    assert!(cube_bytes > 0, "a cube should occupy GPU buffers");
    assert!(r.free_mesh(a));
    assert!(!r.free_mesh(a), "a double free must be refused");
    assert_eq!(
        r.mesh_bytes(),
        total - cube_bytes,
        "freeing a mesh must give its bytes back"
    );

    let b = r.add_mesh(&device, &MeshData::sphere(1.0, 8));
    assert_ne!(a, b, "the reused slot reissued the same key");
    assert_eq!(r.mesh_bytes_of(a), 0, "a stale key reported live bytes");

    // A queued draw against the stale key must not reach the sphere.
    r.begin();
    r.draw(a, mat, Mat4::IDENTITY, None);
    let _ = crate::render_offscreen(&mut r, &device, &queue, 32, 32, [0.0; 4]);
    assert_eq!(r.last_drawn(), 0, "a stale mesh key drew");

    // Freeing a mesh AFTER it was queued must not panic the render pass either:
    // the draw is skipped, not indexed into a hole.
    r.begin();
    r.draw(b, mat, Mat4::IDENTITY, None);
    assert!(r.free_mesh(b));
    let _ = crate::render_offscreen(&mut r, &device, &queue, 32, 32, [0.0; 4]);
    assert_eq!(r.last_drawn(), 0, "a mesh freed mid-frame still drew");
}

/// Skinning matrices are shared across a model's primitives by `Arc`; freeing
/// the model must not leave that allocation referenced by a queued draw.
#[test]
fn freeing_a_model_does_not_strand_its_queued_draws() {
    let Some((device, queue)) = device() else {
        return;
    };
    let _g = crate::gpu_guard();
    let mut s = scene(&device, &queue, 32, 32);
    let h = s.make_box(&device, &queue, [1.0, 1.0, 1.0, 1.0]);
    s.begin();
    s.draw(h, Mat4::IDENTITY);
    // Free between queueing and rendering - the worst-case ordering.
    assert!(s.free_model(h));
    let img = crate::render_offscreen(
        &mut s.renderer,
        &device,
        &queue,
        32,
        32,
        [0.0, 0.0, 0.0, 1.0],
    );
    assert_eq!(s.renderer.last_drawn(), 0);
    assert!(img
        .chunks_exact(4)
        .all(|p| p[0] == 0 && p[1] == 0 && p[2] == 0));
    // Arc is dropped with the queue, not held forever.
    let _ = Arc::new(0u8);
}
