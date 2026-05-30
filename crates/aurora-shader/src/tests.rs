//! Tests for Aurora → WGSL shader lowering.

use crate::lower_module;
use aurora_parser::parse_str;

const SHADER: &str = "
struct VsIn { pos: Vec3, uv: Vec2 }
struct VsOut { clip: Vec4, uv: Vec2 }

@vertex fn vs(vin: VsIn) -> VsOut {
    let scaled = vin.pos * 2.0
    VsOut { clip: vec4(scaled, 1.0), uv: vin.uv }
}

@fragment fn fs(vin: VsOut) -> Color {
    texture(albedo, vin.uv)
}
";

fn wgsl() -> String {
    let (module, diags) = parse_str(SHADER);
    assert!(!diags.iter().any(|d| d.is_error()), "shader failed to parse");
    lower_module(&module)
}

#[test]
fn emits_struct_definitions_in_wgsl_types() {
    let w = wgsl();
    assert!(w.contains("struct VsIn {"), "{w}");
    assert!(w.contains("pos: vec3<f32>"), "{w}");
    assert!(w.contains("uv: vec2<f32>"), "{w}");
    assert!(w.contains("clip: vec4<f32>"), "{w}");
}

#[test]
fn vertex_stage_and_signature() {
    let w = wgsl();
    assert!(w.contains("@vertex"), "{w}");
    assert!(w.contains("fn vs(vin: VsIn) -> VsOut"), "{w}");
}

#[test]
fn vec_constructor_and_let_lowering() {
    let w = wgsl();
    assert!(w.contains("let scaled = (vin.pos * 2.0);"), "{w}");
    assert!(w.contains("vec4<f32>(scaled, 1.0)"), "{w}");
}

#[test]
fn struct_literal_becomes_positional_constructor() {
    // Fields reordered to declaration order (clip, uv) regardless of literal order.
    let w = wgsl();
    assert!(w.contains("return VsOut(vec4<f32>(scaled, 1.0), vin.uv);"), "{w}");
}

#[test]
fn fragment_stage_output_location_and_texture_sample() {
    let w = wgsl();
    assert!(w.contains("@fragment"), "{w}");
    assert!(w.contains("-> @location(0) vec4<f32>"), "{w}");
    assert!(w.contains("textureSample(albedo, albedo_sampler, vin.uv)"), "{w}");
}

#[test]
fn struct_literal_reorders_out_of_order_fields() {
    let src = "
        struct Out { a: f32, b: f32 }
        @fragment fn f() -> Color { Out { b: 2.0, a: 1.0 } }
    ";
    let (m, _) = parse_str(src);
    let w = lower_module(&m);
    // Declared order is (a, b), so the literal `{ b, a }` must come out (1.0, 2.0).
    assert!(w.contains("return Out(1.0, 2.0);"), "{w}");
}

#[test]
fn compute_stage_lowers_with_workgroup_size() {
    let src = "
        @compute fn cs(gid: Vec3) -> Color {
            let v = gid * 2.0
            vec4(v, 1.0)
        }
    ";
    let (m, _) = parse_str(src);
    let w = lower_module(&m);
    assert!(w.contains("@compute @workgroup_size(8, 8, 1)"), "{w}");
    assert!(w.contains("fn cs(gid: vec3<f32>)"), "{w}");
    assert!(w.contains("vec4<f32>(v, 1.0)"), "{w}");
}

#[test]
fn non_shader_functions_are_ignored() {
    let src = "fn helper(x: i32) -> i32 { x }";
    let (m, _) = parse_str(src);
    let w = lower_module(&m);
    assert!(!w.contains("fn helper"), "non-shader fns must not be emitted: {w}");
}
