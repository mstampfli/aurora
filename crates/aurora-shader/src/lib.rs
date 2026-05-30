//! Shader lowering: Aurora `@vertex`/`@fragment`/`@compute` functions → WGSL
//! (grammar spec invariant #1 and §7.6 — "one language for CPU and GPU").
//!
//! Shaders in Aurora are ordinary functions tagged with a stage attribute, using
//! the same types (`Vec3`, `Mat4`, `Color`) as CPU code. This module emits WGSL
//! source for them (and the structs they use), so the same source the CPU
//! type-checks can target the GPU. It covers the expression subset shaders need;
//! IO location/builtin annotations are simplified (a full backend would derive
//! them from usage). The point is to prove the unified-language claim — testably,
//! without a GPU device.

use std::collections::HashMap;

use aurora_ast::{
    BinOp, Block, Expr, ExprKind, FnDecl, Item, ItemKind, Module, Param, Stmt, StructBody, Type,
    TypeKind, UnOp,
};

/// Lower every shader-stage function in `module` to a single WGSL source string
/// (with the structs they reference emitted first).
pub fn lower_module(module: &Module) -> String {
    let structs = collect_structs(module);
    let mut out = String::new();

    // Emit struct definitions used by shaders.
    for item in &module.items {
        if let ItemKind::Struct(s) = &item.kind {
            if let StructBody::Named(fields) = &s.body {
                out.push_str(&format!("struct {} {{\n", s.name.name));
                for f in fields {
                    out.push_str(&format!("    {}: {},\n", f.name.name, ty_to_wgsl(&f.ty)));
                }
                out.push_str("}\n\n");
            }
        }
    }

    for item in &module.items {
        if let Some(stage) = shader_stage(item) {
            if let ItemKind::Fn(f) = &item.kind {
                out.push_str(&lower_fn(stage, f, &structs));
                out.push('\n');
            }
        }
    }
    out
}

/// Names of the `@fragment` functions in `module`, in source order — the entry
/// points a render pipeline binds as its fragment stage.
pub fn fragment_entries(module: &Module) -> Vec<String> {
    module
        .items
        .iter()
        .filter(|it| matches!(shader_stage(it), Some(Stage::Fragment)))
        .filter_map(|it| match &it.kind {
            ItemKind::Fn(f) => Some(f.name.name.clone()),
            _ => None,
        })
        .collect()
}

#[derive(Clone, Copy)]
enum Stage {
    Vertex,
    Fragment,
    Compute,
}

fn shader_stage(item: &Item) -> Option<Stage> {
    for attr in &item.attrs {
        match attr.name.name.as_str() {
            "vertex" => return Some(Stage::Vertex),
            "fragment" => return Some(Stage::Fragment),
            "compute" => return Some(Stage::Compute),
            _ => {}
        }
    }
    None
}

type Structs = HashMap<String, Vec<String>>; // name -> field order

fn collect_structs(module: &Module) -> Structs {
    let mut m = HashMap::new();
    for item in &module.items {
        if let ItemKind::Struct(s) = &item.kind {
            if let StructBody::Named(fields) = &s.body {
                m.insert(s.name.name.clone(), fields.iter().map(|f| f.name.name.clone()).collect());
            }
        }
    }
    m
}

fn lower_fn(stage: Stage, f: &FnDecl, structs: &Structs) -> String {
    let stage_attr = match stage {
        Stage::Vertex => "@vertex",
        Stage::Fragment => "@fragment",
        Stage::Compute => "@compute @workgroup_size(8, 8, 1)",
    };

    let params: Vec<String> = f
        .params
        .iter()
        .filter_map(|p| match p {
            Param::Normal { name, ty, .. } => Some(format!("{}: {}", name.name, ty_to_wgsl(ty))),
            Param::SelfParam { .. } => None,
        })
        .collect();

    // Fragment output goes to @location(0); other stages keep their named type.
    let ret = match (&f.ret, stage) {
        (Some(t), Stage::Fragment) => format!("@location(0) {}", ty_to_wgsl(t)),
        (Some(t), _) => ty_to_wgsl(t),
        (None, _) => "()".to_string(),
    };

    let mut body = String::new();
    if let Some(block) = &f.body {
        lower_block(block, structs, &mut body);
    }

    format!(
        "{stage_attr}\nfn {}({}) -> {ret} {{\n{body}}}\n",
        f.name.name,
        params.join(", ")
    )
}

fn lower_block(block: &Block, structs: &Structs, out: &mut String) {
    for stmt in &block.stmts {
        match stmt {
            Stmt::Let(l) => {
                let name = match &l.pat.kind {
                    aurora_ast::PatKind::Binding { name, .. } => name.name.clone(),
                    _ => "_".into(),
                };
                let init = l.init.as_ref().map(|e| lower_expr(e, structs)).unwrap_or_default();
                out.push_str(&format!("    let {name} = {init};\n"));
            }
            Stmt::Expr(e) => {
                if let ExprKind::Return(opt) = &e.kind {
                    let v = opt.as_ref().map(|e| lower_expr(e, structs)).unwrap_or_default();
                    out.push_str(&format!("    return {v};\n"));
                } else {
                    out.push_str(&format!("    {};\n", lower_expr(e, structs)));
                }
            }
            Stmt::Defer(_) => {}
        }
    }
    if let Some(tail) = &block.tail {
        out.push_str(&format!("    return {};\n", lower_expr(tail, structs)));
    }
}

fn lower_expr(e: &Expr, structs: &Structs) -> String {
    match &e.kind {
        ExprKind::Int(n, _) => n.to_string(),
        ExprKind::Float(x, _) => fmt_float(*x),
        ExprKind::Bool(b) => b.to_string(),
        ExprKind::Path(p) => {
            p.segments.iter().map(|s| s.ident.name.clone()).collect::<Vec<_>>().join(".")
        }
        ExprKind::Paren(inner) => format!("({})", lower_expr(inner, structs)),
        ExprKind::Field { base, field } => {
            let f = match field {
                aurora_ast::FieldAccess::Named(i) => i.name.clone(),
                aurora_ast::FieldAccess::Index(i) => i.to_string(),
            };
            format!("{}.{f}", lower_expr(base, structs))
        }
        ExprKind::Unary(op, x) => {
            let s = lower_expr(x, structs);
            match op {
                UnOp::Neg => format!("-{s}"),
                UnOp::Not => format!("!{s}"),
                _ => s,
            }
        }
        ExprKind::Binary(op, a, b) => {
            format!("({} {} {})", lower_expr(a, structs), bin_op(*op), lower_expr(b, structs))
        }
        ExprKind::Call { callee, args, .. } => lower_call(callee, args, structs),
        ExprKind::Struct { path, fields, .. } => lower_struct(path, fields, structs),
        _ => "/* unsupported */".to_string(),
    }
}

fn lower_call(callee: &Expr, args: &[aurora_ast::Arg], structs: &Structs) -> String {
    let name = match &callee.kind {
        ExprKind::Path(p) => p.segments.last().map(|s| s.ident.name.clone()).unwrap_or_default(),
        _ => return "/* indirect call */".into(),
    };
    let a: Vec<String> = args.iter().map(|x| lower_expr(&x.value, structs)).collect();

    match name.as_str() {
        "vec2" => format!("vec2<f32>({})", a.join(", ")),
        "vec3" => format!("vec3<f32>({})", a.join(", ")),
        "vec4" => format!("vec4<f32>({})", a.join(", ")),
        // `texture(t, uv)` -> a sampled texture (sampler named `<t>_sampler`).
        "texture" if a.len() == 2 => format!("textureSample({}, {}_sampler, {})", a[0], a[0], a[1]),
        // Math builtins share names with WGSL.
        _ => format!("{name}({})", a.join(", ")),
    }
}

/// Lower a struct literal to a WGSL positional constructor, reordering fields to
/// the struct's declaration order when known.
fn lower_struct(path: &aurora_ast::Path, fields: &[aurora_ast::FieldInit], structs: &Structs) -> String {
    let name = path.segments.last().map(|s| s.ident.name.clone()).unwrap_or_default();
    let values: HashMap<&str, String> = fields
        .iter()
        .map(|f| {
            let v = f.value.as_ref().map(|e| lower_expr(e, structs)).unwrap_or_else(|| f.name.name.clone());
            (f.name.name.as_str(), v)
        })
        .collect();

    let ordered: Vec<String> = match structs.get(&name) {
        Some(order) => order
            .iter()
            .map(|fname| values.get(fname.as_str()).cloned().unwrap_or_else(|| "0.0".into()))
            .collect(),
        // Unknown struct: emit in written order.
        None => fields
            .iter()
            .map(|f| values.get(f.name.name.as_str()).cloned().unwrap_or_default())
            .collect(),
    };
    format!("{name}({})", ordered.join(", "))
}

fn bin_op(op: BinOp) -> &'static str {
    match op {
        BinOp::Add => "+",
        BinOp::Sub => "-",
        BinOp::Mul => "*",
        BinOp::Div => "/",
        BinOp::Rem => "%",
        BinOp::Eq => "==",
        BinOp::Ne => "!=",
        BinOp::Lt => "<",
        BinOp::Gt => ">",
        BinOp::Le => "<=",
        BinOp::Ge => ">=",
        BinOp::And => "&&",
        BinOp::Or => "||",
    }
}

fn ty_to_wgsl(t: &Type) -> String {
    let TypeKind::Path(p) = &t.kind else { return "f32".into() };
    match p.segments.last().map(|s| s.ident.name.as_str()).unwrap_or("") {
        "f32" | "f64" => "f32".into(),
        "i32" | "i64" => "i32".into(),
        "u32" | "u64" => "u32".into(),
        "bool" => "bool".into(),
        "Vec2" => "vec2<f32>".into(),
        "Vec3" => "vec3<f32>".into(),
        "Vec4" | "Color" => "vec4<f32>".into(),
        "Mat3" => "mat3x3<f32>".into(),
        "Mat4" => "mat4x4<f32>".into(),
        other => other.into(),
    }
}

fn fmt_float(x: f64) -> String {
    // WGSL float literals need a decimal point.
    if x == x.trunc() && x.is_finite() {
        format!("{x:.1}")
    } else {
        format!("{x}")
    }
}

#[cfg(test)]
mod tests;
