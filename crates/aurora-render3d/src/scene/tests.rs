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

/// The same file loaded N times must upload ONCE.
///
/// A game gives each body its own handle so each animates on its own clock, which is the
/// only way to write a horde. That made the natural spelling of "24 zombies over 5 files"
/// cost 24 uploads: MARROW spent about 4.7 GB of VRAM on it, enough that a second copy of
/// the game could not fit on an 8 GB card at all - it stalled inside the driver's allocator
/// instead of failing, which looks like a hang and reads like a netcode problem.
///
/// So the bytes are the assertion, not the handle count: loading again must add nothing.
#[test]
fn loading_one_file_many_times_uploads_it_once() {
    let Some((device, queue)) = device() else {
        eprintln!("no GPU adapter - skipping the shared-asset test");
        return;
    };
    let _g = crate::gpu_guard();
    let mut s = scene(&device, &queue, 64, 64);
    let path = std::env::temp_dir().join("aurora_shared_asset_model.obj");
    std::fs::write(
        &path,
        "v 0 0 0\nv 1 0 0\nv 0 1 0\nv 1 1 0\nf 1 2 3\nf 2 4 3\n",
    )
    .expect("write obj");
    let path = path.to_str().expect("utf8 path").to_string();

    let base_bytes = s.renderer.mesh_bytes();
    let base_mats = s.renderer.material_count();
    let first = s.load_model(&device, &queue, &path);
    assert!(first >= 0, "the test model failed to load");
    let one_copy = s.renderer.mesh_bytes();
    assert!(one_copy > base_bytes, "the first load uploaded nothing");
    let one_mats = s.renderer.material_count();

    let mut handles = vec![first];
    for _ in 0..23 {
        let h = s.load_model(&device, &queue, &path);
        assert!(h >= 0);
        handles.push(h);
    }
    assert_eq!(
        s.renderer.mesh_bytes(),
        one_copy,
        "24 handles uploaded more than one copy of the mesh"
    );
    assert_eq!(
        s.renderer.material_count(),
        one_mats,
        "24 handles uploaded more than one copy of the material"
    );
    assert_eq!(s.asset_count(), 1, "one file should be one asset");
    assert_eq!(s.model_count(), 24, "each body still needs its own handle");

    // Freeing all but one must keep the survivor's geometry: sharing is only safe if the
    // last user owns the release, not the first.
    for h in handles.drain(1..) {
        assert!(s.free_model(h));
    }
    assert_eq!(
        s.renderer.mesh_bytes(),
        one_copy,
        "freeing a sharer released geometry the survivor still needs"
    );
    assert_eq!(s.asset_count(), 1);
    assert!(s.free_model(first));
    assert_eq!(
        s.renderer.mesh_bytes(),
        base_bytes,
        "the last handle did not release the shared upload"
    );
    assert_eq!(
        s.renderer.material_count(),
        base_mats,
        "the last handle did not release the shared material"
    );
    assert_eq!(s.asset_count(), 0, "the cache kept a dead asset alive");
    assert_eq!(s.model_count(), 0);

    // And the cache must not resurrect a freed asset by handing back stale ids: a reload
    // after a full release has to upload again.
    let again = s.load_model(&device, &queue, &path);
    assert!(again >= 0);
    assert_eq!(
        s.renderer.mesh_bytes(),
        one_copy,
        "a reload after release did not re-upload"
    );
    assert!(s.free_model(again));
    let _ = std::fs::remove_file(&path);
}

/// Two spellings of one path are one asset. `models/x.obj` and `./models/x.obj` name the
/// same file, and a cache that missed on the spelling would silently restore the old cost.
#[test]
fn the_asset_cache_is_keyed_by_the_resolved_file() {
    let Some((device, queue)) = device() else {
        return;
    };
    let _g = crate::gpu_guard();
    let mut s = scene(&device, &queue, 64, 64);
    let dir = std::env::temp_dir();
    let path = dir.join("aurora_asset_key_model.obj");
    std::fs::write(&path, "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n").expect("write obj");
    let direct = path.to_str().expect("utf8").to_string();
    let indirect = dir
        .join(".")
        .join("aurora_asset_key_model.obj")
        .to_str()
        .expect("utf8")
        .to_string();

    let a = s.load_model(&device, &queue, &direct);
    let bytes = s.renderer.mesh_bytes();
    let b = s.load_model(&device, &queue, &indirect);
    assert!(a >= 0 && b >= 0);
    assert_eq!(
        s.renderer.mesh_bytes(),
        bytes,
        "the same file spelled two ways uploaded twice"
    );
    assert_eq!(s.asset_count(), 1);
    assert!(s.free_model(a));
    assert!(s.free_model(b));
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
    assert_eq!(s.root_delta(a), [0.0; 3], "a dead handle moves nothing");
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
    s.show_joints(a);
    assert_eq!(s.clip_name(a, 0), None);
    assert_eq!(s.clip_index(a, "Walk"), -1);
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

/// The clip-name matching rule, which is what lets a game bind animations by
/// name instead of by a magic index. Needs no GPU, so it runs everywhere.
///
/// The rule now lives with the asset format ([`crate::model::find_name`]) and is
/// the SAME function the retarget and the motion-root test use. It was three
/// functions: this one stripped `|`, the asset side stripped `:` as well, and a
/// comment there claimed they agreed - so a `mixamorig:` name resolved here and
/// not there. The `mixamorig:` case below is that drift, pinned.
#[test]
fn names_resolve_by_prefix_and_case() {
    // What a Quaternius/Blender glTF export actually looks like.
    let names = ["CharacterArmature|Death", "CharacterArmature|Walk", "Idle"];
    let at = |want: &str| super::index_or_missing(super::find_name(names.iter().copied(), want));

    // The armature prefix is an export setting, so the bare name must match.
    assert_eq!(at("Walk"), 1);
    assert_eq!(at("walk"), 1, "matching must not depend on export casing");
    assert_eq!(
        at("CharacterArmature|Death"),
        0,
        "the full name must match too"
    );
    assert_eq!(at("Idle"), 2);
    assert_eq!(
        at("  Walk  "),
        1,
        "surrounding whitespace must not defeat a match"
    );

    // A clip that is not there must FAIL rather than resolve to something else:
    // silently playing clip 0 is the bug this whole builtin exists to prevent.
    assert_eq!(at("Sprint"), -1);
    assert_eq!(at(""), -1);

    // An exact name beats another armature's suffix.
    let shadowed = ["Rig|Walk", "Walk"];
    assert_eq!(
        super::find_name(shadowed.iter().copied(), "Walk"),
        Some(1),
        "an exact match must win over a suffix match"
    );

    // A namespace is the same kind of export decoration as an armature prefix, on
    // whichever side of the question it turns up.
    let namespaced = ["mixamorig:Hips", "mixamorig:Spine_01"];
    let ns =
        |want: &str| super::index_or_missing(super::find_name(namespaced.iter().copied(), want));
    assert_eq!(ns("Spine_01"), 1, "a namespaced rig must answer to the bare name");
    assert_eq!(ns("mixamorig:Hips"), 0, "and to its own full name");
    assert_eq!(
        at("mixamorig:Walk"),
        1,
        "a decorated request must find an undecorated name too"
    );
    assert_eq!(ns("Neck"), -1, "and a bone that is not there is still not there");
}

/// A skinned model whose joints hang off an ARMATURE node must be posed through
/// that node's transform.
///
/// This is the Blender/glTF shape: the exporter parents the skeleton to an
/// `Armature` node carrying the Z-up -> Y-up rotation and the unit scale, and
/// glTF resolves a joint's global transform through the whole node tree. Walking
/// only the joint subtree drops it, and the symptom is silent - the character
/// renders lying on its back at 1/100th size, with no error anywhere.
///
/// The fixture is a one-joint skin under an armature node scaled 4x and rotated
/// -90 degrees about X, so the expected skin matrix is exactly that transform.
#[test]
fn a_skeleton_under_an_armature_node_inherits_its_transform() {
    use crate::anim::skin_matrices;
    use crate::model::Model;

    // Minimal glTF: RootNode -> Armature(scale 4, rot -90 X) -> Joint, plus a
    // skinned triangle. Buffers are inline base64 so the test needs no asset.
    // 12 floats of position (3 verts), then 4 joint indices + 4 weights per vert.
    let gltf = r#"{
      "asset": {"version": "2.0"},
      "scene": 0,
      "scenes": [{"nodes": [0]}],
      "nodes": [
        {"name": "RootNode", "children": [1, 3]},
        {"name": "Armature", "children": [2],
         "scale": [4.0, 4.0, 4.0],
         "rotation": [-0.7071068, 0.0, 0.0, 0.7071068]},
        {"name": "Joint"},
        {"name": "Mesh", "mesh": 0, "skin": 0}
      ],
      "skins": [{"skeleton": 2, "joints": [2], "inverseBindMatrices": 3}],
      "meshes": [{"primitives": [{"attributes":
        {"POSITION": 0, "JOINTS_0": 1, "WEIGHTS_0": 2}}]}],
      "accessors": [
        {"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3",
         "min": [0,0,0], "max": [1,1,0]},
        {"bufferView": 1, "componentType": 5123, "count": 3, "type": "VEC4"},
        {"bufferView": 2, "componentType": 5126, "count": 3, "type": "VEC4"},
        {"bufferView": 3, "componentType": 5126, "count": 1, "type": "MAT4"}
      ],
      "bufferViews": [
        {"buffer": 0, "byteOffset": 0,   "byteLength": 36},
        {"buffer": 0, "byteOffset": 36,  "byteLength": 24},
        {"buffer": 0, "byteOffset": 60,  "byteLength": 48},
        {"buffer": 0, "byteOffset": 108, "byteLength": 64}
      ],
      "buffers": [{"byteLength": 172, "uri": "data:application/octet-stream;base64,BUF"}]
    }"#;

    // Positions (3 verts), joint indices (u16 x4 each), weights (f32 x4 each),
    // and an identity inverse-bind matrix.
    let mut buf: Vec<u8> = Vec::new();
    for v in [[0.0f32, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]] {
        for c in v {
            buf.extend_from_slice(&c.to_le_bytes());
        }
    }
    for _ in 0..3 {
        for j in [0u16, 0, 0, 0] {
            buf.extend_from_slice(&j.to_le_bytes());
        }
    }
    for _ in 0..3 {
        for w in [1.0f32, 0.0, 0.0, 0.0] {
            buf.extend_from_slice(&w.to_le_bytes());
        }
    }
    for c in Mat4::IDENTITY.to_cols_array() {
        buf.extend_from_slice(&c.to_le_bytes());
    }
    assert_eq!(
        buf.len(),
        172,
        "fixture buffer length must match the header"
    );

    let b64 = {
        // Small inline base64 encoder: avoids adding a dependency for one fixture.
        const T: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        let mut out = String::new();
        for c in buf.chunks(3) {
            let (b0, b1, b2) = (
                c[0] as u32,
                *c.get(1).unwrap_or(&0) as u32,
                *c.get(2).unwrap_or(&0) as u32,
            );
            let n = (b0 << 16) | (b1 << 8) | b2;
            out.push(T[(n >> 18 & 63) as usize] as char);
            out.push(T[(n >> 12 & 63) as usize] as char);
            out.push(if c.len() > 1 {
                T[(n >> 6 & 63) as usize] as char
            } else {
                '='
            });
            out.push(if c.len() > 2 {
                T[(n & 63) as usize] as char
            } else {
                '='
            });
        }
        out
    };

    let path = std::env::temp_dir().join("aurora_armature_fixture.gltf");
    std::fs::write(&path, gltf.replace("BUF", &b64)).expect("write fixture");
    let model = Model::load(path.to_str().expect("utf8")).expect("fixture must load");
    let _ = std::fs::remove_file(&path);

    let skel = model.skeleton.as_ref().expect("fixture has a skin");
    // The skeleton spans the whole bone tree, so it holds the armature above the joint as
    // well as the joint itself. What must hold is that SKIN joints keep the skin's own
    // ordering at indices 0..N-1, because that is what the mesh's JOINTS_0 indexes.
    assert!(!skel.joints.is_empty());
    assert_eq!(skel.joints[0].name, "Joint");

    // With no clip, the single joint's skin matrix IS the armature transform.
    let m = skin_matrices(skel, None, 0.0)[0];

    // The 4x scale must survive: a point 1 unit out along the joint's X lands 4 out.
    let px = m.transform_point3(Vec3::X);
    assert!(
        (px.length() - 4.0).abs() < 1e-3,
        "armature scale was dropped: |{px:?}| = {}, want 4",
        px.length()
    );

    // The -90 degrees about X must survive: +Y maps to -Z (up becomes forward),
    // which is exactly the difference between standing and lying on your back.
    let py = m.transform_point3(Vec3::Y);
    assert!(
        (py - Vec3::new(0.0, 0.0, -4.0)).length() < 1e-3,
        "armature rotation was dropped: +Y mapped to {py:?}, want (0,0,-4)"
    );
}

// --- socket scale ---------------------------------------------------------
//
// A socket places and orients; it does not resize. `draw_on_joint` divides the
// bone's own scale out, which is what lets a weapon be drawn at the size it was
// authored on a rig whose joints carry a unit-conversion factor.
//
// The obvious implementation - decompose to a quaternion and recompose - is
// wrong and was reverted once for being wrong: a mirrored bone has a negative
// determinant, no quaternion can represent a reflection, and the round trip
// drops it silently so a prop on the rig's left-hand side comes back flipped.
// These pin the behaviour that replaced it.

/// A uniform 0.01 (the centimetre rig's factor) is removed.
#[test]
fn a_sockets_uniform_scale_is_divided_out() {
    let m = Mat4::from_scale(Vec3::splat(0.01));
    let out = super::without_scale(m);
    let px = out.transform_point3(Vec3::X);
    assert!(
        (px.length() - 1.0).abs() < 1e-6,
        "a 0.01 bone scale survived: |{px:?}| = {}, want 1",
        px.length()
    );
}

/// So is a NON-uniform one, per axis rather than by one averaged factor.
#[test]
fn each_axis_is_normalised_separately() {
    let m = Mat4::from_scale(Vec3::new(0.5, 2.0, 4.0));
    let out = super::without_scale(m);
    for (axis, name) in [(Vec3::X, "X"), (Vec3::Y, "Y"), (Vec3::Z, "Z")] {
        let p = out.transform_point3(axis);
        assert!(
            (p.length() - 1.0).abs() < 1e-6,
            "{name} kept a scale: |{p:?}| = {}, want 1",
            p.length()
        );
    }
}

/// Rotation is untouched: only the LENGTH of each basis vector changes.
#[test]
fn orientation_and_position_survive() {
    let m = Mat4::from_scale_rotation_translation(
        Vec3::splat(0.01),
        glam::Quat::from_rotation_x(std::f32::consts::FRAC_PI_2),
        Vec3::new(1.0, 2.0, 3.0),
    );
    let out = super::without_scale(m);

    // The socket is still where the bone is. Normalising the translation column
    // would have dragged it to within a metre of the origin.
    let at = out.transform_point3(Vec3::ZERO);
    assert!(
        (at - Vec3::new(1.0, 2.0, 3.0)).length() < 1e-6,
        "the socket moved: {at:?}, want (1,2,3)"
    );

    // +90 degrees about X still maps +Y to +Z.
    let dir = out.transform_point3(Vec3::Y) - at;
    assert!(
        (dir - Vec3::Z).length() < 1e-6,
        "orientation was lost: +Y mapped to {dir:?}, want +Z"
    );
}

/// The case that broke the previous attempt: a bone mirrored on one axis.
///
/// A negative determinant must stay negative. Through a quaternion it cannot,
/// so the reflection is dropped and a left-hand prop is drawn inside-out.
#[test]
fn a_mirrored_bone_keeps_its_reflection() {
    let m = Mat4::from_scale(Vec3::new(-0.01, 0.01, 0.01));
    assert!(m.determinant() < 0.0, "the fixture is not actually mirrored");

    let out = super::without_scale(m);
    assert!(
        out.determinant() < 0.0,
        "the reflection was dropped: determinant {} , want negative",
        out.determinant()
    );
    // And it is a unit reflection now, not a 0.01 one.
    assert!(
        (out.determinant() + 1.0).abs() < 1e-6,
        "determinant {} , want -1",
        out.determinant()
    );
    // The mirrored axis still points the other way.
    let px = out.transform_point3(Vec3::X);
    assert!(
        (px - Vec3::new(-1.0, 0.0, 0.0)).length() < 1e-6,
        "the mirrored axis moved: {px:?}, want (-1,0,0)"
    );
}

/// A collapsed bone must not become NaN. Wrong but finite beats invisible.
#[test]
fn a_degenerate_bone_does_not_divide_by_zero() {
    let m = Mat4::from_scale(Vec3::new(0.0, 1.0, 1.0));
    let out = super::without_scale(m);
    for c in [out.x_axis, out.y_axis, out.z_axis, out.w_axis] {
        assert!(
            c.is_finite(),
            "a zero-length basis column produced {c:?}"
        );
    }
}

/// An already-unit socket is left exactly alone, so normalising is idempotent
/// and costs nothing on a rig that never needed it.
#[test]
fn a_unit_socket_is_unchanged() {
    let m = Mat4::from_rotation_translation(
        glam::Quat::from_rotation_y(0.7),
        Vec3::new(4.0, 0.0, -2.0),
    );
    let out = super::without_scale(m);
    assert!(
        (out - m).to_cols_array().iter().all(|v| v.abs() < 1e-6),
        "a unit socket was changed"
    );
}

// --- the material table's generation is a CACHE KEY ---------------------------
//
// `material_generation` is part of the asset cache key, so bumping it throws
// away every uploaded mesh: the next load of a file already in memory re-reads
// it from disk and uploads it again. `set_material_texture` bumped it
// unconditionally, and binding an atlas before loading art - the same atlas for
// every mesh in a pack - is the COMMON case, not a strange one.
//
// Poly Souls stages a room by binding the pack atlas and loading its walls,
// props and buildings. Every staging re-uploaded the lot, so a doorway that
// re-stages on each trip between two rooms leaked a room's textures per trip,
// and `melee` - which stages four times - died in `Device::create_texture` with
// "Not enough memory left" while twenty gigabytes of system RAM sat free.
//
// The decision is tested rather than the Scene, because a Scene needs a real GPU
// device to build and the only symptom of getting this wrong is memory.

#[test]
fn rebinding_a_material_to_the_texture_it_already_has_changes_nothing() {
    let mut table = std::collections::HashMap::new();
    table.insert("lambert1".to_string(), "atlas_01.png".to_string());
    assert!(
        !super::binding_changes(&table, "lambert1", "atlas_01.png"),
        "an identical rebind was treated as a change, which throws away every \
         uploaded mesh"
    );
}

#[test]
fn a_real_binding_change_is_a_change() {
    let mut table = std::collections::HashMap::new();
    table.insert("lambert1".to_string(), "atlas_01.png".to_string());
    // A different texture on the same material.
    assert!(super::binding_changes(&table, "lambert1", "atlas_02.png"));
    // And a material nothing has bound yet.
    assert!(super::binding_changes(&table, "Wall71", "atlas_01.png"));
}

/// An atlas named by MANY materials is READ and uploaded ONCE.
///
/// The case that made this necessary: an art pack is one 4096 x 4096 image -
/// 64 MiB as RGBA - and every mesh in it carries its own material name, all
/// pointing at that one file. Poly Souls' bailey has dozens. Before textures
/// were shared by source, each name decoded its own copy and uploaded its own
/// GPU texture, and standing up one courtyard cost 4.3 GB; the allocation that
/// finally failed was 67108864 bytes, which is that atlas exactly.
///
/// Asserted in BYTES rather than in "it worked", because the only symptom of
/// this regressing is memory: a scene that uploads forty copies of an atlas
/// renders identically to one that uploads a single copy.
#[test]
fn one_atlas_named_by_many_materials_uploads_once() {
    let Some((device, queue)) = device() else {
        eprintln!("no GPU adapter - skipping the shared texture test");
        return;
    };
    let _g = crate::gpu_guard();
    let mut s = scene(&device, &queue, 64, 64);

    // A real file on disk, because naming a file is now the whole mechanism -
    // a test that handed pixels over could not exercise it.
    let dir = std::env::temp_dir().join("aurora-shared-tex");
    let _ = std::fs::create_dir_all(&dir);
    let atlas = dir.join("atlas.png");
    let other = dir.join("other.png");
    let img = image::RgbaImage::from_pixel(64, 64, image::Rgba([200, 180, 60, 255]));
    img.save(&atlas).expect("write the test atlas");
    // A DIFFERENT picture, because identical pixels are deliberately shared
    // whatever they are called - see the twin assertion at the end.
    let other_img = image::RgbaImage::from_pixel(64, 64, image::Rgba([10, 20, 30, 255]));
    other_img.save(&other).expect("write the second test atlas");
    let twin = dir.join("twin.png");
    img.save(&twin).expect("write a copy of the atlas under another name");
    let twin = twin.to_string_lossy().to_string();
    let atlas = atlas.to_string_lossy().to_string();
    let other = other.to_string_lossy().to_string();
    let one = 64u64 * 64 * 4;

    let base = s.renderer.tex_bytes();
    assert_eq!(s.renderer.tex_count(), 0, "nothing shared yet");

    // A free function rather than a closure: a closure capturing `s` would hold
    // the borrow across every assertion below it.
    fn named(s: &mut Scene, d: &wgpu::Device, q: &wgpu::Queue, file: &str, srgb_slot: bool) {
        let src = crate::render::TexSrc::File(file);
        let desc = MaterialDesc {
            base_color: [1.0, 1.0, 1.0, 1.0],
            metallic: 0.0,
            roughness: 1.0,
            emissive: [0.0; 3],
            base_tex: if srgb_slot { Some(src) } else { None },
            normal_tex: if srgb_slot { None } else { Some(src) },
            mr_tex: None,
            emissive_tex: None,
        };
        s.renderer.add_material(d, q, &desc);
    }

    // Twenty materials, every one of them naming the same file.
    for _ in 0..20 {
        named(&mut s, &device, &queue, &atlas, true);
    }
    assert_eq!(
        s.renderer.tex_bytes() - base,
        one,
        "twenty materials naming one file must upload it once, not twenty times"
    );
    assert_eq!(s.renderer.tex_count(), 1, "one distinct image");

    // A different PICTURE is a second upload.
    named(&mut s, &device, &queue, &other, true);
    assert_eq!(s.renderer.tex_bytes() - base, one * 2, "a second picture");

    // The SAME file in a different colour space is a different GPU texture and
    // must not be handed over. An sRGB atlas served as a linear normal map is a
    // silently WRONG picture, which is worse than a missing one.
    named(&mut s, &device, &queue, &atlas, false);
    assert_eq!(
        s.renderer.tex_bytes() - base,
        one * 3,
        "the same file as sRGB and as linear are two textures"
    );

    // THE FILE IS NOT READ AGAIN once it is uploaded. Deleting it and asking for
    // it a twenty-first time must still work: if the read still happened this
    // would fall back to a 1x1 white pixel and the count would not move.
    std::fs::remove_file(&atlas).expect("remove the atlas");
    named(&mut s, &device, &queue, &atlas, true);
    assert_eq!(
        s.renderer.tex_bytes() - base,
        one * 3,
        "an uploaded file is never read again - the decode is skipped too"
    );
    assert_eq!(s.renderer.tex_count(), 3);

    // And pixels with NO named source are never entered into the shared cache -
    // nothing else can be holding them, and sharing them would alias two
    // models' embedded textures onto whichever loaded first.
    let px = vec![200u8; 64 * 64 * 4];
    let before = s.renderer.tex_bytes();
    for _ in 0..5 {
        let desc = MaterialDesc {
            base_color: [1.0, 1.0, 1.0, 1.0],
            metallic: 0.0,
            roughness: 1.0,
            emissive: [0.0; 3],
            base_tex: Some(crate::render::TexSrc::own(&px, 64, 64)),
            normal_tex: None,
            mr_tex: None,
            emissive_tex: None,
        };
        s.renderer.add_material(&device, &queue, &desc);
    }
    assert_eq!(
        s.renderer.tex_bytes(),
        before,
        "unnamed pixels are not entered into the shared cache at all"
    );

    // THE SAME PICTURE UNDER ANOTHER NAME IS THE SAME UPLOAD.
    //
    // A file shares by path, which is enough for an atlas a game names. It is
    // not enough for one a PACK EMBEDS into every module it ships: those pixels
    // arrive with no name of their own, get called after the model they came out
    // of, and are then one upload per model. Six castle pieces carrying the same
    // 4096 x 4096 image is 384 MiB of one picture - measured, before this.
    let before_twin = s.renderer.tex_bytes();
    named(&mut s, &device, &queue, &twin, true);
    assert_eq!(
        s.renderer.tex_bytes(),
        before_twin,
        "identical pixels under a second name must not upload again"
    );

    let _ = std::fs::remove_file(&other);
    let _ = std::fs::remove_file(&twin);
}
