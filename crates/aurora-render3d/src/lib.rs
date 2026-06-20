//! Aurora's GPU 3D renderer.
//!
//! A real [`wgpu`] forward renderer: indexed meshes with a depth buffer, a
//! perspective camera, directional + ambient lighting, base-color textures, and
//! GPU vertex skinning. It borrows a wgpu device/queue, so the same renderer
//! drives an offscreen target (headless tests, read back and asserted) or the
//! live window surface. [`model`] loads glTF/OBJ meshes, materials, skeletons,
//! and animation clips; [`anim`] samples those clips into skinning matrices.

mod anim;
mod mesh;
mod model;
mod render;
mod scene;

pub use anim::{skin_matrices, skin_matrices_blended, AnimPlayer};
pub use glam::{Mat4, Quat, Vec3};
pub use mesh::{GpuMesh, MeshData, Vertex};
pub use model::{Clip, Joint, Model, Primitive, Skeleton};
pub use render::{InstanceRaw, Material, MaterialDesc, Renderer3D, DEPTH_FORMAT, MAX_JOINTS, MAX_LIGHTS};
pub use scene::Scene;

/// A right-handed perspective projection with a wgpu-style depth range (z in
/// `[0, 1]`). `fov_y` is in radians.
pub fn perspective(fov_y: f32, aspect: f32, near: f32, far: f32) -> Mat4 {
    Mat4::perspective_rh(fov_y, aspect.max(0.0001), near, far)
}

/// A right-handed look-at view matrix.
pub fn look_at(eye: Vec3, target: Vec3, up: Vec3) -> Mat4 {
    Mat4::look_at_rh(eye, target, up)
}

/// Acquire a headless GPU (device + queue) with the adapter's full limits, for
/// offscreen rendering and tests. Returns `None` if no adapter is available.
pub fn headless_device() -> Option<(wgpu::Device, wgpu::Queue)> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))?;
    let limits = adapter.limits();
    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("aurora-render3d"),
            required_features: wgpu::Features::empty(),
            required_limits: limits,
            memory_hints: wgpu::MemoryHints::Performance,
        },
        None,
    ))
    .ok()?;
    Some((device, queue))
}

/// Render the renderer's queued draws into a fresh offscreen `Rgba8Unorm`
/// texture and read the pixels back (tight `w*h*4` bytes). For tests/tools.
pub fn render_offscreen(
    r: &mut Renderer3D,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    w: u32,
    h: u32,
    clear: [f32; 4],
) -> Vec<u8> {
    let fmt = wgpu::TextureFormat::Rgba8Unorm;
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("offscreen"),
        size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: fmt,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
    r.resize(device, w, h);

    let unpadded = w * 4;
    let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let padded = unpadded.div_ceil(align) * align;
    let out_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size: (padded * h) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    r.render(device, queue, &mut enc, &view, clear);
    enc.copy_texture_to_buffer(
        wgpu::ImageCopyTexture {
            texture: &tex,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::ImageCopyBuffer {
            buffer: &out_buf,
            layout: wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(padded),
                rows_per_image: Some(h),
            },
        },
        wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
    );
    queue.submit(Some(enc.finish()));

    let slice = out_buf.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |res| {
        let _ = tx.send(res);
    });
    device.poll(wgpu::Maintain::Wait);
    let _ = rx.recv();
    let data = slice.get_mapped_range();
    let mut out = Vec::with_capacity((unpadded * h) as usize);
    for row in 0..h {
        let start = (row * padded) as usize;
        out.extend_from_slice(&data[start..start + unpadded as usize]);
    }
    drop(data);
    out_buf.unmap();
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    static GPU_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    fn guard() -> std::sync::MutexGuard<'static, ()> {
        GPU_LOCK.lock().unwrap_or_else(|p| p.into_inner())
    }

    fn px(buf: &[u8], w: u32, x: u32, y: u32) -> [u8; 4] {
        let o = ((y * w + x) * 4) as usize;
        [buf[o], buf[o + 1], buf[o + 2], buf[o + 3]]
    }

    #[test]
    fn renders_a_lit_depth_tested_cube() {
        let _g = guard();
        let Some((device, queue)) = headless_device() else {
            eprintln!("no GPU adapter - skipping 3D render test");
            return;
        };
        let (w, h) = (96u32, 96u32);
        let mut r = Renderer3D::new(&device, &queue, wgpu::TextureFormat::Rgba8Unorm, w, h, 1);
        let cube = r.add_mesh(&device, &MeshData::cube());
        let red = r.add_material(&device, &queue, &MaterialDesc::flat([0.9, 0.1, 0.1, 1.0]));

        let view = look_at(Vec3::new(3.0, 2.5, 4.0), Vec3::ZERO, Vec3::Y);
        let proj = perspective(60f32.to_radians(), w as f32 / h as f32, 0.1, 100.0);
        r.set_camera(proj * view, Vec3::new(3.0, 2.5, 4.0));
        r.set_light(Vec3::new(0.5, 1.0, 0.4), Vec3::ONE, 0.2);

        r.begin();
        r.draw(cube, red, Mat4::IDENTITY, None);
        let img = render_offscreen(&mut r, &device, &queue, w, h, [0.0, 0.0, 0.0, 1.0]);

        // The cube fills the center; that pixel must be a lit red, not the black
        // clear color.
        let c = px(&img, w, w / 2, h / 2);
        assert!(c[0] > 32, "center should be lit red, got {c:?}");
        assert!(c[0] > c[2], "red channel should dominate, got {c:?}");
        // A corner pixel should still be the background clear color.
        let corner = px(&img, w, 1, 1);
        assert!(corner[0] < 20 && corner[1] < 20, "corner should be background, got {corner:?}");
    }

    #[test]
    fn emissive_material_glows_without_lights() {
        let _g = guard();
        let Some((device, queue)) = headless_device() else { return };
        let (w, h) = (48u32, 48u32);
        let mut r = Renderer3D::new(&device, &queue, wgpu::TextureFormat::Rgba8Unorm, w, h, 1);
        let cube = r.add_mesh(&device, &MeshData::cube());
        let glow = r.add_material(
            &device,
            &queue,
            &MaterialDesc { base_color: [0.0, 0.0, 0.0, 1.0], metallic: 0.0, roughness: 1.0, emissive: [0.0, 1.0, 0.0], base_tex: None, normal_tex: None, mr_tex: None, emissive_tex: None },
        );
        let view = look_at(Vec3::new(0.0, 0.0, 5.0), Vec3::ZERO, Vec3::Y);
        r.set_camera(perspective(60f32.to_radians(), 1.0, 0.1, 100.0) * view, Vec3::new(0.0, 0.0, 5.0));
        // No lighting at all: black directional, zero ambient.
        r.set_light(Vec3::new(0.0, 1.0, 0.0), Vec3::ZERO, 0.0);
        r.begin();
        r.draw(cube, glow, Mat4::IDENTITY, None);
        let img = render_offscreen(&mut r, &device, &queue, w, h, [0.0, 0.0, 0.0, 1.0]);
        let c = px(&img, w, w / 2, h / 2);
        assert!(c[1] > 150 && c[0] < 60, "emissive green should glow with no lights, got {c:?}");
    }

    #[test]
    fn point_light_illuminates_an_unlit_scene() {
        let _g = guard();
        let Some((device, queue)) = headless_device() else { return };
        let (w, h) = (48u32, 48u32);
        let mut r = Renderer3D::new(&device, &queue, wgpu::TextureFormat::Rgba8Unorm, w, h, 1);
        let cube = r.add_mesh(&device, &MeshData::cube());
        let white = r.add_material(&device, &queue, &MaterialDesc::flat([1.0, 1.0, 1.0, 1.0]));
        let view = look_at(Vec3::new(0.0, 0.0, 5.0), Vec3::ZERO, Vec3::Y);
        r.set_camera(perspective(60f32.to_radians(), 1.0, 0.1, 100.0) * view, Vec3::new(0.0, 0.0, 5.0));
        r.set_light(Vec3::new(0.0, 1.0, 0.0), Vec3::ZERO, 0.0); // no directional, no ambient
        r.clear_point_lights();
        r.add_point_light(Vec3::new(0.0, 0.0, 3.0), Vec3::ONE, 20.0, 5.0);
        r.begin();
        r.draw(cube, white, Mat4::IDENTITY, None);
        let img = render_offscreen(&mut r, &device, &queue, w, h, [0.0, 0.0, 0.0, 1.0]);
        let c = px(&img, w, w / 2, h / 2);
        assert!(c[0] > 40, "point light should illuminate the cube face, got {c:?}");
    }

    #[test]
    fn ssao_darkens_creases() {
        let _g = guard();
        let Some((device, queue)) = headless_device() else { return };
        let (w, h) = (96u32, 96u32);
        let mut r = Renderer3D::new(&device, &queue, wgpu::TextureFormat::Rgba8Unorm, w, h, 1);
        let plane = r.add_mesh(&device, &MeshData::plane(20.0, 1.0));
        let cube = r.add_mesh(&device, &MeshData::cube());
        let white = r.add_material(&device, &queue, &MaterialDesc::flat([1.0, 1.0, 1.0, 1.0]));
        r.set_camera(perspective(55f32.to_radians(), 1.0, 0.1, 100.0) * look_at(Vec3::new(4.0, 5.0, 6.0), Vec3::new(0.0, 0.5, 0.0), Vec3::Y), Vec3::new(4.0, 5.0, 6.0));
        // Ambient-dominant lighting so AO (which only affects ambient) is visible.
        r.set_light(Vec3::new(0.3, 1.0, 0.2), Vec3::splat(0.2), 0.8);
        r.set_shadows(false); // isolate AO from shadowing
        let scene = |r: &mut Renderer3D| {
            r.begin();
            r.draw(plane, white, Mat4::IDENTITY, None);
            r.draw(cube, white, Mat4::from_translation(Vec3::new(0.0, 1.0, 0.0)), None);
        };
        let luma = |img: &[u8]| -> u64 { img.chunks_exact(4).map(|p| p[0] as u64 + p[1] as u64 + p[2] as u64).sum() };
        r.set_ssao(true);
        scene(&mut r);
        let with = render_offscreen(&mut r, &device, &queue, w, h, [0.0, 0.0, 0.0, 1.0]);
        r.set_ssao(false);
        scene(&mut r);
        let without = render_offscreen(&mut r, &device, &queue, w, h, [0.0, 0.0, 0.0, 1.0]);
        let (a, b) = (luma(&with), luma(&without));
        assert!(a < b, "SSAO must darken (occlude) the scene: with={a} without={b}");
        assert!(b - a > 500, "occlusion should be clearly present: with={a} without={b}");
    }

    #[test]
    fn point_light_casts_a_shadow() {
        let _g = guard();
        let Some((device, queue)) = headless_device() else { return };
        let (w, h) = (96u32, 96u32);
        let mut r = Renderer3D::new(&device, &queue, wgpu::TextureFormat::Rgba8Unorm, w, h, 1);
        let plane = r.add_mesh(&device, &MeshData::plane(30.0, 1.0));
        let cube = r.add_mesh(&device, &MeshData::cube());
        let white = r.add_material(&device, &queue, &MaterialDesc::flat([1.0, 1.0, 1.0, 1.0]));
        r.set_camera(perspective(60f32.to_radians(), 1.0, 0.1, 100.0) * look_at(Vec3::new(6.0, 5.0, 6.0), Vec3::new(0.0, 0.5, 0.0), Vec3::Y), Vec3::new(6.0, 5.0, 6.0));
        // Only a point light above an occluder box; no directional, no ambient.
        r.set_light(Vec3::new(0.0, 1.0, 0.0), Vec3::ZERO, 0.0);
        r.set_shadows(false);
        r.clear_point_lights();
        r.add_point_light(Vec3::new(0.0, 5.0, 0.0), Vec3::ONE, 30.0, 8.0);
        let scene = |r: &mut Renderer3D| {
            r.begin();
            r.draw(plane, white, Mat4::IDENTITY, None);
            r.draw(cube, white, Mat4::from_translation(Vec3::new(0.0, 2.0, 0.0)), None);
        };
        let luma = |img: &[u8]| -> u64 { img.chunks_exact(4).map(|p| p[0] as u64 + p[1] as u64 + p[2] as u64).sum() };
        let maxch = |img: &[u8]| -> u8 { img.iter().copied().max().unwrap_or(0) };

        r.set_point_shadows(true);
        scene(&mut r);
        let with = render_offscreen(&mut r, &device, &queue, w, h, [0.0, 0.0, 0.0, 1.0]);
        r.set_point_shadows(false);
        scene(&mut r);
        let without = render_offscreen(&mut r, &device, &queue, w, h, [0.0, 0.0, 0.0, 1.0]);

        let (a, b) = (luma(&with), luma(&without));
        assert!(a < b, "the point light's shadow must darken the scene: with={a} without={b}");
        assert!(b - a > 1000, "the cast shadow should be clearly present: with={a} without={b}");
        // ...but the point light still lights the unoccluded floor (not all dark).
        assert!(maxch(&with) > 100, "some floor must remain point-lit, max={}", maxch(&with));
    }

    #[test]
    fn shadows_darken_the_scene() {
        let _g = guard();
        let Some((device, queue)) = headless_device() else { return };
        let (w, h) = (96u32, 96u32);
        let mut r = Renderer3D::new(&device, &queue, wgpu::TextureFormat::Rgba8Unorm, w, h, 1);
        let plane = r.add_mesh(&device, &MeshData::plane(20.0, 1.0));
        let cube = r.add_mesh(&device, &MeshData::cube());
        let white = r.add_material(&device, &queue, &MaterialDesc::flat([1.0, 1.0, 1.0, 1.0]));
        let view = look_at(Vec3::new(4.0, 6.0, 8.0), Vec3::new(0.0, 0.5, 0.0), Vec3::Y);
        r.set_camera(perspective(55f32.to_radians(), 1.0, 0.1, 200.0) * view, Vec3::new(4.0, 6.0, 8.0));
        r.set_light(Vec3::new(0.35, 1.0, 0.2), Vec3::ONE, 0.05);
        r.set_shadow_extent(10.0);

        let draw_scene = |r: &mut Renderer3D| {
            r.begin();
            r.draw(plane, white, Mat4::IDENTITY, None);
            r.draw(cube, white, Mat4::from_translation(Vec3::new(0.0, 1.0, 0.0)), None);
        };
        let luma = |img: &[u8]| -> u64 {
            img.chunks_exact(4).map(|p| p[0] as u64 + p[1] as u64 + p[2] as u64).sum()
        };

        r.set_shadows(true);
        draw_scene(&mut r);
        let with = render_offscreen(&mut r, &device, &queue, w, h, [0.1, 0.1, 0.12, 1.0]);
        r.set_shadows(false);
        draw_scene(&mut r);
        let without = render_offscreen(&mut r, &device, &queue, w, h, [0.1, 0.1, 0.12, 1.0]);

        // The only difference between the two renders is the shadow map, so the
        // entire luma drop is the cast shadow.
        let (a, b) = (luma(&with), luma(&without));
        assert!(a < b, "shadowed scene must be darker: with={a} without={b}");
        assert!(b - a > 400, "cast shadow should be clearly present: with={a} without={b}");
    }

    #[test]
    fn msaa_renders_and_resolves() {
        let _g = guard();
        let Some((device, queue)) = headless_device() else { return };
        let (w, h) = (64u32, 64u32);
        // 4x MSAA: render() uses a multisampled target and resolves into the
        // single-sample offscreen texture.
        let mut r = Renderer3D::new(&device, &queue, wgpu::TextureFormat::Rgba8Unorm, w, h, 4);
        let cube = r.add_mesh(&device, &MeshData::cube());
        let red = r.add_material(&device, &queue, &MaterialDesc::flat([0.9, 0.1, 0.1, 1.0]));
        let view = look_at(Vec3::new(2.5, 2.0, 3.0), Vec3::ZERO, Vec3::Y);
        r.set_camera(perspective(60f32.to_radians(), 1.0, 0.1, 100.0) * view, Vec3::new(2.5, 2.0, 3.0));
        r.set_light(Vec3::new(0.5, 1.0, 0.4), Vec3::ONE, 0.2);
        r.begin();
        r.draw(cube, red, Mat4::IDENTITY, None);
        let img = render_offscreen(&mut r, &device, &queue, w, h, [0.0, 0.0, 0.0, 1.0]);
        let c = px(&img, w, w / 2, h / 2);
        assert!(c[0] > 32 && c[0] > c[2], "MSAA-resolved cube should be lit red, got {c:?}");
    }

    #[test]
    fn metallic_surface_reflects_the_sky() {
        let _g = guard();
        let Some((device, queue)) = headless_device() else { return };
        let (w, h) = (64u32, 64u32);
        let mut r = Renderer3D::new(&device, &queue, wgpu::TextureFormat::Rgba8Unorm, w, h, 1);
        let sphere = r.add_mesh(&device, &MeshData::sphere(1.0, 32));
        // A polished metal: it has no diffuse, so its color is the reflected sky.
        let metal = r.add_material(
            &device,
            &queue,
            &MaterialDesc { base_color: [0.6, 0.6, 0.6, 1.0], metallic: 1.0, roughness: 0.15, emissive: [0.0; 3], base_tex: None, normal_tex: None, mr_tex: None, emissive_tex: None },
        );
        r.set_camera(perspective(50f32.to_radians(), 1.0, 0.1, 100.0) * look_at(Vec3::new(0.0, 0.0, 4.0), Vec3::ZERO, Vec3::Y), Vec3::new(0.0, 0.0, 4.0));
        // A strongly blue sky, near-zero direct light, IBL strength up.
        r.set_sky(true, Vec3::new(0.1, 0.2, 0.95), Vec3::new(0.3, 0.45, 0.9));
        r.set_light(Vec3::new(0.0, 1.0, 0.2), Vec3::splat(0.05), 1.6);
        r.begin();
        r.draw(sphere, metal, Mat4::IDENTITY, None);
        let img = render_offscreen(&mut r, &device, &queue, w, h, [0.0, 0.0, 0.0, 1.0]);
        let c = px(&img, w, w / 2, h / 2);
        assert!(c[2] > c[0] + 15, "metal should reflect the blue sky, got {c:?}");
    }

    #[test]
    fn skybox_fills_the_background() {
        let _g = guard();
        let Some((device, queue)) = headless_device() else { return };
        let (w, h) = (64u32, 64u32);
        let mut r = Renderer3D::new(&device, &queue, wgpu::TextureFormat::Rgba8Unorm, w, h, 1);
        let view = look_at(Vec3::new(0.0, 1.0, 0.0), Vec3::new(0.0, 1.2, -1.0), Vec3::Y);
        r.set_camera(perspective(70f32.to_radians(), 1.0, 0.1, 100.0) * view, Vec3::new(0.0, 1.0, 0.0));
        r.set_sky(true, Vec3::new(0.2, 0.4, 0.8), Vec3::new(0.7, 0.8, 0.95));
        r.begin();
        // No geometry: the whole frame is sky. Clear color is red so we can tell
        // the sky actually drew over it.
        let img = render_offscreen(&mut r, &device, &queue, w, h, [1.0, 0.0, 0.0, 1.0]);
        let top = px(&img, w, w / 2, 2);
        assert!(top[2] > top[0], "sky should be blue-dominant, not the red clear, got {top:?}");
        assert!(top[2] > 100, "sky should be bright blue, got {top:?}");
    }

    #[test]
    fn gpu_instancing_draws_many_in_one_call() {
        let _g = guard();
        let Some((device, queue)) = headless_device() else { return };
        let (w, h) = (64u32, 64u32);
        let mut r = Renderer3D::new(&device, &queue, wgpu::TextureFormat::Rgba8Unorm, w, h, 1);
        let cube = r.add_mesh(&device, &MeshData::cube());
        let red = r.add_material(&device, &queue, &MaterialDesc::flat([0.9, 0.2, 0.2, 1.0]));
        r.set_camera(perspective(60f32.to_radians(), 1.0, 0.1, 100.0) * look_at(Vec3::new(0.0, 1.0, 8.0), Vec3::ZERO, Vec3::Y), Vec3::new(0.0, 1.0, 8.0));
        r.set_light(Vec3::new(0.3, 1.0, 0.4), Vec3::ONE, 0.3);
        r.begin();
        // Three cubes in a row via ONE instanced draw call.
        let insts = vec![
            InstanceRaw::new(Mat4::from_translation(Vec3::new(-2.5, 0.0, 0.0)), [1.0, 1.0, 1.0, 1.0]),
            InstanceRaw::new(Mat4::from_translation(Vec3::new(0.0, 0.0, 0.0)), [1.0, 1.0, 1.0, 1.0]),
            InstanceRaw::new(Mat4::from_translation(Vec3::new(2.5, 0.0, 0.0)), [1.0, 1.0, 1.0, 1.0]),
        ];
        r.draw_instanced(cube, red, insts);
        let img = render_offscreen(&mut r, &device, &queue, w, h, [0.0, 0.0, 0.0, 1.0]);
        // The center and both flanking instances should be lit red.
        assert!(px(&img, w, w / 2, h / 2)[0] > 32, "center instance should render");
        assert!(px(&img, w, 8, h / 2)[0] > 32, "left instance should render");
        assert!(px(&img, w, w - 8, h / 2)[0] > 32, "right instance should render");
    }

    #[test]
    fn frustum_culls_offscreen_draws() {
        let _g = guard();
        let Some((device, queue)) = headless_device() else { return };
        let (w, h) = (48u32, 48u32);
        let mut r = Renderer3D::new(&device, &queue, wgpu::TextureFormat::Rgba8Unorm, w, h, 1);
        let cube = r.add_mesh(&device, &MeshData::cube());
        let white = r.add_material(&device, &queue, &MaterialDesc::flat([1.0, 1.0, 1.0, 1.0]));
        r.set_camera(perspective(60f32.to_radians(), 1.0, 0.1, 100.0) * look_at(Vec3::new(0.0, 0.0, 5.0), Vec3::ZERO, Vec3::Y), Vec3::new(0.0, 0.0, 5.0));
        r.set_light(Vec3::Y, Vec3::ONE, 0.3);

        let draw = |r: &mut Renderer3D| {
            r.begin();
            r.draw(cube, white, Mat4::IDENTITY, None);                                 // in front
            r.draw(cube, white, Mat4::from_translation(Vec3::new(0.0, 0.0, 1000.0)), None); // behind camera
        };
        r.set_frustum_cull(true);
        draw(&mut r);
        let _ = render_offscreen(&mut r, &device, &queue, w, h, [0.0; 4]);
        assert_eq!(r.last_drawn(), 1, "the cube behind the camera must be culled");

        r.set_frustum_cull(false);
        draw(&mut r);
        let _ = render_offscreen(&mut r, &device, &queue, w, h, [0.0; 4]);
        assert_eq!(r.last_drawn(), 2, "with culling off, both draws issue");
    }

    #[test]
    fn debug_lines_draw() {
        let _g = guard();
        let Some((device, queue)) = headless_device() else { return };
        let (w, h) = (48u32, 48u32);
        let mut r = Renderer3D::new(&device, &queue, wgpu::TextureFormat::Rgba8Unorm, w, h, 1);
        r.set_camera(perspective(60f32.to_radians(), 1.0, 0.1, 100.0) * look_at(Vec3::new(0.0, 0.0, 5.0), Vec3::ZERO, Vec3::Y), Vec3::new(0.0, 0.0, 5.0));
        let luma = |img: &[u8]| -> u64 { img.chunks_exact(4).map(|p| p[0] as u64 + p[1] as u64 + p[2] as u64).sum() };
        r.begin();
        let blank = render_offscreen(&mut r, &device, &queue, w, h, [0.0, 0.0, 0.0, 1.0]);
        r.begin();
        r.debug_line(Vec3::new(-3.0, 0.0, 0.0), Vec3::new(3.0, 0.0, 0.0), Vec3::new(1.0, 1.0, 1.0));
        let withln = render_offscreen(&mut r, &device, &queue, w, h, [0.0, 0.0, 0.0, 1.0]);
        assert!(luma(&withln) > luma(&blank) + 100, "a debug line should add bright pixels");
    }

    #[test]
    fn billboard_quad_renders_unlit() {
        let _g = guard();
        let Some((device, queue)) = headless_device() else { return };
        let (w, h) = (48u32, 48u32);
        let mut r = Renderer3D::new(&device, &queue, wgpu::TextureFormat::Rgba8Unorm, w, h, 1);
        let quad = r.add_mesh(&device, &MeshData::quad());
        let green = r.add_material(
            &device,
            &queue,
            &MaterialDesc { base_color: [0.0, 0.0, 0.0, 1.0], metallic: 0.0, roughness: 1.0, emissive: [0.0, 1.0, 0.0], base_tex: None, normal_tex: None, mr_tex: None, emissive_tex: None },
        );
        // A quad facing the camera (identity model, camera on +z looking at it).
        r.set_camera(perspective(60f32.to_radians(), 1.0, 0.1, 100.0) * look_at(Vec3::new(0.0, 0.0, 3.0), Vec3::ZERO, Vec3::Y), Vec3::new(0.0, 0.0, 3.0));
        r.set_light(Vec3::Y, Vec3::ZERO, 0.0); // no lighting; emissive only
        r.begin();
        r.draw(quad, green, Mat4::from_scale(Vec3::splat(2.0)), None);
        let img = render_offscreen(&mut r, &device, &queue, w, h, [0.0, 0.0, 0.0, 1.0]);
        let c = px(&img, w, w / 2, h / 2);
        assert!(c[1] > 150 && c[0] < 60, "emissive billboard quad should be green, got {c:?}");
    }

    #[test]
    fn obj_loads_geometry_and_normals() {
        let dir = std::env::temp_dir();
        let path = dir.join("aurora_test_tri.obj");
        std::fs::write(
            &path,
            "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n",
        )
        .unwrap();
        let model = Model::load(path.to_str().unwrap()).expect("load obj");
        assert_eq!(model.primitives.len(), 1);
        let p = &model.primitives[0];
        assert_eq!(p.mesh.indices.len(), 3);
        // No normals in the file -> flat normals computed; the triangle lies in
        // the z=0 plane so its normal points along +/- Z.
        let n = p.mesh.vertices[0].normal;
        assert!(n[2].abs() > 0.9, "expected a z-facing normal, got {n:?}");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn animation_samples_and_interpolates_translation() {
        use crate::model::{Channel, Clip, Interp, Joint, Path, Skeleton};
        let skel = Skeleton {
            joints: vec![Joint {
                parent: None,
                inverse_bind: Mat4::IDENTITY,
                t: Vec3::ZERO,
                r: Quat::IDENTITY,
                s: Vec3::ONE,
            }],
        };
        let clip = Clip {
            name: "move".into(),
            duration: 1.0,
            channels: vec![Channel {
                joint: 0,
                path: Path::Translation,
                interp: Interp::Linear,
                times: vec![0.0, 1.0],
                values: vec![0.0, 0.0, 0.0, 0.0, 2.0, 0.0], // (0,0,0) -> (0,2,0)
            }],
        };
        // Halfway should interpolate to (0,1,0).
        let m = skin_matrices(&skel, Some(&clip), 0.5);
        let p = m[0].transform_point3(Vec3::ZERO);
        assert!((p - Vec3::new(0.0, 1.0, 0.0)).length() < 1e-4, "got {p:?}");
        // At the end, (0,2,0).
        let m1 = skin_matrices(&skel, Some(&clip), 1.0);
        let p1 = m1[0].transform_point3(Vec3::ZERO);
        assert!((p1 - Vec3::new(0.0, 2.0, 0.0)).length() < 1e-4, "got {p1:?}");
    }

    #[test]
    fn animation_blends_two_clips() {
        use crate::model::{Channel, Clip, Interp, Joint, Path, Skeleton};
        let skel = Skeleton {
            joints: vec![Joint { parent: None, inverse_bind: Mat4::IDENTITY, t: Vec3::ZERO, r: Quat::IDENTITY, s: Vec3::ONE }],
        };
        let still = Clip { name: "a".into(), duration: 1.0, channels: vec![] }; // stays at default (0,0,0)
        let up = Clip {
            name: "b".into(),
            duration: 1.0,
            channels: vec![Channel { joint: 0, path: Path::Translation, interp: Interp::Linear, times: vec![0.0], values: vec![0.0, 2.0, 0.0] }],
        };
        // Blend halfway between "still" (0,0,0) and "up" (0,2,0) -> (0,1,0).
        let m = skin_matrices_blended(&skel, Some(&still), 0.0, Some(&up), 0.0, 0.5);
        let p = m[0].transform_point3(Vec3::ZERO);
        assert!((p - Vec3::new(0.0, 1.0, 0.0)).length() < 1e-4, "blended pose should be halfway, got {p:?}");
    }

    #[test]
    fn gpu_skinning_applies_joint_matrix() {
        let _g = guard();
        let Some((device, queue)) = headless_device() else {
            return;
        };
        let (w, h) = (64u32, 64u32);
        let mut r = Renderer3D::new(&device, &queue, wgpu::TextureFormat::Rgba8Unorm, w, h, 1);
        let cube = r.add_mesh(&device, &MeshData::cube());
        let red = r.add_material(&device, &queue, &MaterialDesc::flat([1.0, 0.2, 0.2, 1.0]));
        let view = look_at(Vec3::new(0.0, 0.0, 6.0), Vec3::ZERO, Vec3::Y);
        r.set_camera(perspective(60f32.to_radians(), 1.0, 0.1, 100.0) * view, Vec3::new(0.0, 0.0, 6.0));
        r.set_light(Vec3::new(0.0, 0.0, 1.0), Vec3::ONE, 0.5);

        // Skinned with an identity joint: the cube renders normally.
        r.begin();
        r.draw(cube, red, Mat4::IDENTITY, Some(vec![Mat4::IDENTITY]));
        let lit = render_offscreen(&mut r, &device, &queue, w, h, [0.0, 0.0, 0.0, 1.0]);
        assert!(px(&lit, w, w / 2, h / 2)[0] > 60, "identity-skinned cube should render");

        // Skinned with a near-zero scale joint: the GPU skinning collapses the
        // cube, so the center is background -> proves the joint matrix is applied
        // in the vertex shader.
        r.begin();
        r.draw(cube, red, Mat4::IDENTITY, Some(vec![Mat4::from_scale(Vec3::splat(0.0001))]));
        let collapsed = render_offscreen(&mut r, &device, &queue, w, h, [0.0, 0.0, 0.0, 1.0]);
        assert!(px(&collapsed, w, w / 2, h / 2)[0] < 20, "collapsed-skin cube should vanish");
    }

    #[test]
    fn depth_test_occludes_far_cube() {
        let _g = guard();
        let Some((device, queue)) = headless_device() else {
            return;
        };
        let (w, h) = (64u32, 64u32);
        let mut r = Renderer3D::new(&device, &queue, wgpu::TextureFormat::Rgba8Unorm, w, h, 1);
        let cube = r.add_mesh(&device, &MeshData::cube());
        let near_red = r.add_material(&device, &queue, &MaterialDesc::flat([1.0, 0.0, 0.0, 1.0]));
        let far_green = r.add_material(&device, &queue, &MaterialDesc::flat([0.0, 1.0, 0.0, 1.0]));

        let view = look_at(Vec3::new(0.0, 0.0, 6.0), Vec3::ZERO, Vec3::Y);
        let proj = perspective(60f32.to_radians(), 1.0, 0.1, 100.0);
        r.set_camera(proj * view, Vec3::new(0.0, 0.0, 6.0));
        r.set_light(Vec3::new(0.0, 0.0, 1.0), Vec3::ONE, 0.4);

        r.begin();
        // Draw the far green cube first, then the near red one: depth must keep
        // the near (red) cube in front regardless of submission order.
        r.draw(cube, far_green, Mat4::from_translation(Vec3::new(0.0, 0.0, -2.0)), None);
        r.draw(cube, near_red, Mat4::from_translation(Vec3::new(0.0, 0.0, 1.0)) * Mat4::from_scale(Vec3::splat(0.8)), None);
        let img = render_offscreen(&mut r, &device, &queue, w, h, [0.0, 0.0, 0.0, 1.0]);

        let c = px(&img, w, w / 2, h / 2);
        assert!(c[0] > c[1], "near red cube must occlude the far green one, got {c:?}");
    }
}
