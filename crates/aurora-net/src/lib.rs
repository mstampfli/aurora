//! Replication wire format derived from `@replicated` components (netcode spec
//! §2–§3, §8.3).
//!
//! A [`CompSchema`] is built from a component declaration: it honors `@quantize`
//! (fixed-point encoding) and excludes `@noreplicate` fields. From the schema we
//! get, *without any hand-written (de)serializer*:
//!
//! * `serialize` / `deserialize` — full component round-trip;
//! * `delta` / `apply_delta` — dirty-mask delta encoding against a baseline;
//! * `schema_hash` — a layout fingerprint for the handshake version check.
//!
//! This mirrors what the real compiler would *generate* per replicated
//! component; here it is schema-driven at runtime over interpreter [`Value`]s.

mod bitpack;
mod channel;
mod fixed;
mod interest;
mod lagcomp;
mod lockstep;
mod predict;
mod rng;
mod snapshot;
mod transport;
pub use bitpack::{read_quat, write_quat, BitReader, BitWriter};
pub use channel::Reliable;
pub use transport::UdpEndpoint;
pub use fixed::{Fixed, FVec3};
pub use lockstep::Body;
pub use interest::{interest_delta, InterestGrid};
pub use lagcomp::{Hit, LagComp, V3};
pub use predict::{server_advance, Predictor};
pub use rng::Rng;
pub use snapshot::InterpBuffer;

use std::collections::BTreeMap;

use aurora_ast::{AttrArg, ExprKind, Field, Item, ItemKind, StructBody, TypeKind};
use aurora_interp::Value;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FieldKind {
    Bool,
    Int,
    Float,
    Str,
    /// Unmodelled type; skipped on the wire.
    Opaque,
}

#[derive(Clone, Debug)]
pub struct FieldSchema {
    pub name: String,
    pub kind: FieldKind,
    /// `@quantize(step)` — encode floats as fixed-point multiples of `step`.
    pub quantize: Option<f64>,
}

#[derive(Clone, Debug)]
pub struct CompSchema {
    pub name: String,
    pub fields: Vec<FieldSchema>,
}

impl CompSchema {
    /// Build a schema from a `component` item, honoring `@noreplicate`/`@quantize`.
    pub fn from_item(item: &Item) -> Option<CompSchema> {
        let ItemKind::Component(decl) = &item.kind else { return None };
        let StructBody::Named(fields) = &decl.body else { return None };
        let fields = fields.iter().filter_map(field_schema).collect();
        Some(CompSchema { name: decl.name.name.clone(), fields })
    }

    /// Find and build the schema for the named component in a module.
    pub fn from_module(module: &aurora_ast::Module, name: &str) -> Option<CompSchema> {
        module.items.iter().find_map(|item| {
            let s = CompSchema::from_item(item)?;
            (s.name == name).then_some(s)
        })
    }

    fn field_value(v: &Value, name: &str) -> Value {
        match v {
            Value::Struct(_, map) => map.get(name).cloned().unwrap_or(Value::Unit),
            _ => Value::Unit,
        }
    }

    /// Serialize all replicated fields into a bit-packed buffer.
    pub fn serialize(&self, v: &Value) -> Vec<u8> {
        let mut w = BitWriter::new();
        for f in &self.fields {
            encode_field(f, &Self::field_value(v, &f.name), &mut w);
        }
        w.finish()
    }

    /// Reconstruct a component value from a bit-packed buffer.
    pub fn deserialize(&self, bytes: &[u8]) -> Result<Value, String> {
        let mut r = BitReader::new(bytes);
        let mut map = BTreeMap::new();
        for f in &self.fields {
            map.insert(f.name.clone(), decode_field(f, &mut r));
        }
        Ok(Value::Struct(self.name.clone(), map))
    }

    /// Encode only the fields that differ from `baseline`: a 1-bit-per-field
    /// dirty mask, then the changed fields' values — all bit-packed.
    pub fn delta(&self, baseline: &Value, current: &Value) -> Vec<u8> {
        let mut w = BitWriter::new();
        let mut changed = Vec::new();
        for f in &self.fields {
            let old = Self::field_value(baseline, &f.name);
            let new = Self::field_value(current, &f.name);
            let dirty = old != new;
            w.write_bool(dirty);
            if dirty {
                changed.push((f, new));
            }
        }
        for (f, v) in changed {
            encode_field(f, &v, &mut w);
        }
        w.finish()
    }

    /// Apply a delta produced by [`delta`] on top of `baseline`.
    pub fn apply_delta(&self, baseline: &Value, delta: &[u8]) -> Result<Value, String> {
        let mut r = BitReader::new(delta);
        let dirty: Vec<bool> = self.fields.iter().map(|_| r.read_bool()).collect();
        let mut map = match baseline {
            Value::Struct(_, m) => m.clone(),
            _ => BTreeMap::new(),
        };
        for (i, f) in self.fields.iter().enumerate() {
            if dirty[i] {
                map.insert(f.name.clone(), decode_field(f, &mut r));
            }
        }
        Ok(Value::Struct(self.name.clone(), map))
    }

    /// A layout fingerprint exchanged at handshake to detect schema mismatch.
    pub fn schema_hash(&self) -> u64 {
        // FNV-1a over (name, kind, quantize-flag) of each field.
        let mut h: u64 = 0xcbf29ce484222325;
        let mut feed = |bytes: &[u8]| {
            for &b in bytes {
                h ^= b as u64;
                h = h.wrapping_mul(0x100000001b3);
            }
        };
        feed(self.name.as_bytes());
        for f in &self.fields {
            feed(f.name.as_bytes());
            feed(&[f.kind as u8, f.quantize.is_some() as u8]);
        }
        h
    }
}

fn field_schema(f: &Field) -> Option<FieldSchema> {
    if f.attrs.iter().any(|a| a.name.name == "noreplicate") {
        return None;
    }
    let quantize = f.attrs.iter().find(|a| a.name.name == "quantize").and_then(|a| {
        a.args.first().and_then(|arg| match arg {
            AttrArg::Positional(e) | AttrArg::Named(_, e) => match &e.kind {
                ExprKind::Float(v, _) => Some(*v),
                ExprKind::Int(v, _) => Some(*v as f64),
                _ => None,
            },
        })
    });
    Some(FieldSchema { name: f.name.name.clone(), kind: type_kind(&f.ty.kind), quantize })
}

fn type_kind(t: &TypeKind) -> FieldKind {
    let TypeKind::Path(p) = t else { return FieldKind::Opaque };
    match p.segments.last().map(|s| s.ident.name.as_str()).unwrap_or("") {
        "bool" => FieldKind::Bool,
        "i8" | "i16" | "i32" | "i64" | "u8" | "u16" | "u32" | "u64" => FieldKind::Int,
        "f32" | "f64" => FieldKind::Float,
        "str" => FieldKind::Str,
        _ => FieldKind::Opaque,
    }
}

// --- field codecs ----------------------------------------------------------

fn as_i64(v: &Value) -> i64 {
    match v {
        Value::Int(n) => *n as i64,
        Value::Bool(b) => *b as i64,
        _ => 0,
    }
}

fn as_f64(v: &Value) -> f64 {
    match v {
        Value::Float(x) => *x,
        Value::Int(n) => *n as f64,
        _ => 0.0,
    }
}

fn encode_field(f: &FieldSchema, v: &Value, w: &mut BitWriter) {
    match f.kind {
        FieldKind::Bool => w.write_bool(matches!(v, Value::Bool(true))),
        FieldKind::Int => w.write_bits(as_i64(v) as u64, 64),
        FieldKind::Float => match f.quantize {
            // Fixed-point: 16 bits instead of 32 — the bandwidth win.
            Some(step) => {
                let q = (as_f64(v) / step).round() as i16;
                w.write_bits(q as u16 as u64, 16);
            }
            None => w.write_bits((as_f64(v) as f32).to_bits() as u64, 32),
        },
        FieldKind::Str => {
            let s = match v {
                Value::Str(s) => s.as_str(),
                _ => "",
            };
            let len = s.len().min(u16::MAX as usize);
            w.write_bits(len as u64, 16);
            for &byte in &s.as_bytes()[..len] {
                w.write_bits(byte as u64, 8);
            }
        }
        FieldKind::Opaque => {}
    }
}

fn decode_field(f: &FieldSchema, r: &mut BitReader) -> Value {
    match f.kind {
        FieldKind::Bool => Value::Bool(r.read_bool()),
        FieldKind::Int => Value::Int(r.read_bits(64) as i64 as i128),
        FieldKind::Float => match f.quantize {
            Some(step) => {
                let q = r.read_bits(16) as u16 as i16;
                Value::Float(q as f64 * step)
            }
            None => Value::Float(f32::from_bits(r.read_bits(32) as u32) as f64),
        },
        FieldKind::Str => {
            let len = r.read_bits(16) as usize;
            let bytes: Vec<u8> = (0..len).map(|_| r.read_bits(8) as u8).collect();
            Value::Str(String::from_utf8_lossy(&bytes).into_owned())
        }
        FieldKind::Opaque => Value::Unit,
    }
}

#[cfg(test)]
mod tests;
