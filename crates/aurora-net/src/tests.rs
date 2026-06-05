//! Tests for replication serialization derived from `@replicated` components.

use std::collections::BTreeMap;

use crate::{CompSchema, FieldKind};
use aurora_interp::Value;
use aurora_parser::parse_str;

const AVATAR: &str = "@replicated(authority: .server)
component Avatar {
    @interp pos: Vec3
    health: f32
    @quantize(0.1) ammo: f32
    alive: bool
    @noreplicate fx_timer: f32
}";

fn schema(src: &str, name: &str) -> CompSchema {
    let (module, diags) = parse_str(src);
    assert!(!diags.iter().any(|d| d.is_error()), "parse failed");
    CompSchema::from_module(&module, name).expect("component not found")
}

fn strukt(name: &str, fields: &[(&str, Value)]) -> Value {
    let mut map = BTreeMap::new();
    for (k, v) in fields {
        map.insert(k.to_string(), v.clone());
    }
    Value::Struct(name.to_string(), map)
}

#[test]
fn schema_excludes_noreplicate_and_marks_quantize() {
    let s = schema(AVATAR, "Avatar");
    let names: Vec<&str> = s.fields.iter().map(|f| f.name.as_str()).collect();
    // `pos` is Vec3 (Opaque here), `fx_timer` is excluded.
    assert!(names.contains(&"health"));
    assert!(names.contains(&"ammo"));
    assert!(names.contains(&"alive"));
    assert!(!names.contains(&"fx_timer"), "noreplicate field must be excluded");

    let ammo = s.fields.iter().find(|f| f.name == "ammo").unwrap();
    assert_eq!(ammo.kind, FieldKind::Float);
    assert!(ammo.quantize.is_some(), "ammo should be quantized");
}

#[test]
fn round_trip_preserves_replicated_fields() {
    let s = schema(AVATAR, "Avatar");
    let v = strukt(
        "Avatar",
        &[
            ("pos", Value::Unit),
            ("health", Value::Float(75.0)),
            ("ammo", Value::Float(2.5)),
            ("alive", Value::Bool(true)),
            ("fx_timer", Value::Float(9.0)),
        ],
    );
    let bytes = s.serialize(&v);
    let back = s.deserialize(&bytes).unwrap();

    let Value::Struct(_, m) = back else { panic!() };
    assert_eq!(m.get("health"), Some(&Value::Float(75.0)));
    assert_eq!(m.get("ammo"), Some(&Value::Float(2.5))); // quantize(0.1) preserves 2.5
    assert_eq!(m.get("alive"), Some(&Value::Bool(true)));
    // The non-replicated field never crosses the wire.
    assert!(m.get("fx_timer").is_none());
}

#[test]
fn quantization_shrinks_the_float() {
    // A quantized f32 costs 2 bytes; an unquantized one costs 4.
    let quant = schema("component Q { @quantize(0.1) x: f32 }", "Q");
    let plain = schema("component P { x: f32 }", "P");
    let qv = strukt("Q", &[("x", Value::Float(3.3))]);
    let pv = strukt("P", &[("x", Value::Float(3.3))]);
    assert_eq!(quant.serialize(&qv).len(), 2);
    assert_eq!(plain.serialize(&pv).len(), 4);
}

#[test]
fn delta_only_sends_changed_fields_and_applies() {
    let s = schema(AVATAR, "Avatar");
    let baseline = strukt(
        "Avatar",
        &[
            ("pos", Value::Unit),
            ("health", Value::Float(100.0)),
            ("ammo", Value::Float(5.0)),
            ("alive", Value::Bool(true)),
        ],
    );
    // Only `health` changes.
    let current = strukt(
        "Avatar",
        &[
            ("pos", Value::Unit),
            ("health", Value::Float(80.0)),
            ("ammo", Value::Float(5.0)),
            ("alive", Value::Bool(true)),
        ],
    );

    let delta = s.delta(&baseline, &current);
    let full = s.serialize(&current);
    assert!(delta.len() < full.len(), "delta ({}) should be smaller than full ({})", delta.len(), full.len());

    let rebuilt = s.apply_delta(&baseline, &delta).unwrap();
    let Value::Struct(_, m) = rebuilt else { panic!() };
    assert_eq!(m.get("health"), Some(&Value::Float(80.0)));
    assert_eq!(m.get("ammo"), Some(&Value::Float(5.0)));
}

#[test]
fn empty_delta_when_nothing_changed() {
    let s = schema(AVATAR, "Avatar");
    let v = strukt(
        "Avatar",
        &[("health", Value::Float(50.0)), ("ammo", Value::Float(1.0)), ("alive", Value::Bool(false))],
    );
    let delta = s.delta(&v, &v);
    // Just the (all-zero) dirty mask, no payload.
    assert!(delta.iter().all(|&b| b == 0));
    // An empty delta applied to the baseline reproduces it unchanged.
    let rebuilt = s.apply_delta(&v, &delta).unwrap();
    assert_eq!(rebuilt, v);
}

#[test]
fn bools_pack_to_bits_not_bytes() {
    // Eight bool fields fit in one byte (bit-packed), not eight.
    let s = schema(
        "component Flags { a: bool, b: bool, c: bool, d: bool, e: bool, f: bool, g: bool, h: bool }",
        "Flags",
    );
    let v = strukt(
        "Flags",
        &[
            ("a", Value::Bool(true)),
            ("b", Value::Bool(false)),
            ("c", Value::Bool(true)),
            ("d", Value::Bool(true)),
            ("e", Value::Bool(false)),
            ("f", Value::Bool(false)),
            ("g", Value::Bool(true)),
            ("h", Value::Bool(false)),
        ],
    );
    assert_eq!(s.serialize(&v).len(), 1, "8 bools should pack into 1 byte");
    // And still round-trips.
    let back = s.deserialize(&s.serialize(&v)).unwrap();
    let Value::Struct(_, m) = back else { panic!() };
    assert_eq!(m.get("a"), Some(&Value::Bool(true)));
    assert_eq!(m.get("g"), Some(&Value::Bool(true)));
    assert_eq!(m.get("b"), Some(&Value::Bool(false)));
}

#[test]
fn schema_hash_detects_layout_change() {
    let a = schema("component C { x: f32, y: i32 }", "C");
    let b = schema("component C { x: f32, y: f32 }", "C"); // y changed type
    let c = schema("component C { x: f32, y: i32 }", "C"); // same as a
    assert_ne!(a.schema_hash(), b.schema_hash());
    assert_eq!(a.schema_hash(), c.schema_hash());
}
