//! Runtime values for the tree-walking interpreter.

use std::collections::BTreeMap;
use std::fmt;

use aurora_ast::Expr;

#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    Unit,
    Int(i128),
    Float(f64),
    Bool(bool),
    Char(char),
    Str(String),
    Tuple(Vec<Value>),
    Array(Vec<Value>),
    /// A struct/component instance: type name + named fields.
    Struct(String, BTreeMap<String, Value>),
    /// An ECS entity handle.
    Entity(u64),
    /// A closure: parameter names, body, and a snapshot of captured bindings.
    Closure { params: Vec<String>, body: Box<Expr>, env: BTreeMap<String, Value> },
    /// An enum value: `enum_name::variant` with its payload.
    Enum { enm: String, variant: String, payload: Payload },
}

/// The data carried by an enum variant.
#[derive(Clone, Debug, PartialEq)]
pub enum Payload {
    Unit,
    Tuple(Vec<Value>),
    Struct(BTreeMap<String, Value>),
}

impl Value {
    pub fn type_name(&self) -> &'static str {
        match self {
            Value::Unit => "()",
            Value::Int(_) => "int",
            Value::Float(_) => "float",
            Value::Bool(_) => "bool",
            Value::Char(_) => "char",
            Value::Str(_) => "str",
            Value::Tuple(_) => "tuple",
            Value::Array(_) => "array",
            Value::Struct(..) => "struct",
            Value::Entity(_) => "entity",
            Value::Closure { .. } => "closure",
            Value::Enum { .. } => "enum",
        }
    }

    pub fn truthy(&self) -> Option<bool> {
        match self {
            Value::Bool(b) => Some(*b),
            _ => None,
        }
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::Unit => write!(f, "()"),
            Value::Int(n) => write!(f, "{n}"),
            // Match the native backend's float formatting: whole values keep a
            // trailing `.0` (so `7.0` prints `7.0`, not `7`).
            Value::Float(x) => {
                if x.is_finite() && *x == x.trunc() {
                    write!(f, "{x}.0")
                } else {
                    write!(f, "{x}")
                }
            }
            Value::Bool(b) => write!(f, "{b}"),
            Value::Char(c) => write!(f, "{c}"),
            Value::Str(s) => write!(f, "{s}"),
            Value::Tuple(items) => {
                write!(f, "(")?;
                for (i, v) in items.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{v}")?;
                }
                write!(f, ")")
            }
            Value::Array(items) => {
                write!(f, "[")?;
                for (i, v) in items.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{v}")?;
                }
                write!(f, "]")
            }
            Value::Struct(name, fields) => {
                write!(f, "{name} {{ ")?;
                for (i, (k, v)) in fields.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{k}: {v}")?;
                }
                write!(f, " }}")
            }
            Value::Entity(id) => write!(f, "Entity#{id}"),
            Value::Closure { params, .. } => write!(f, "<closure/{}>", params.len()),
            Value::Enum { enm, variant, payload } => {
                write!(f, "{enm}::{variant}")?;
                match payload {
                    Payload::Unit => Ok(()),
                    Payload::Tuple(items) => {
                        write!(f, "(")?;
                        for (i, v) in items.iter().enumerate() {
                            if i > 0 {
                                write!(f, ", ")?;
                            }
                            write!(f, "{v}")?;
                        }
                        write!(f, ")")
                    }
                    Payload::Struct(fields) => {
                        write!(f, " {{ ")?;
                        for (i, (k, v)) in fields.iter().enumerate() {
                            if i > 0 {
                                write!(f, ", ")?;
                            }
                            write!(f, "{k}: {v}")?;
                        }
                        write!(f, " }}")
                    }
                }
            }
        }
    }
}
