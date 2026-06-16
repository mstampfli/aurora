//! A tree-walking interpreter for Aurora's computational subset (Phase B core).
//!
//! This executes functions, `let` bindings, arithmetic/logic, control flow
//! (`if`/`while`/`loop`/`for` over ranges and arrays), recursion, tuples,
//! arrays, and struct literals — enough to *run* real Aurora programs. The
//! ECS/GPU/region runtime (systems, queries, shaders) is the larger second half
//! of Phase B and is not handled here; encountering those yields a clear runtime
//! error rather than a panic.
//!
//! Output from the `print`/`println` builtins is captured in [`Interp::output`].

mod value;

use std::collections::{BTreeMap, HashMap, HashSet};

use aurora_ast::{
    AssocItem, BinOp, Block, Expr, ExprKind, FieldAccess, FnDecl, ItemKind, Module, Pat, PatKind,
    QTerm, QueryExpr, Stmt, SystemDecl, TypeKind, UnOp,
};

pub use value::{Payload, Value};

/// A minimal ECS world: entities plus per-component instance storage.
#[derive(Default)]
struct World {
    next_id: u64,
    entities: Vec<u64>,
    /// component name -> (entity id -> component value).
    comps: HashMap<String, HashMap<u64, Value>>,
}

impl World {
    fn spawn(&mut self, components: Vec<Value>) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        self.entities.push(id);
        for c in components {
            if let Value::Struct(name, _) = &c {
                self.comps.entry(name.clone()).or_default().insert(id, c);
            }
        }
        id
    }

    fn despawn(&mut self, id: u64) {
        self.entities.retain(|&e| e != id);
        for table in self.comps.values_mut() {
            table.remove(&id);
        }
    }

    fn has(&self, comp: &str, id: u64) -> bool {
        self.comps.get(comp).is_some_and(|t| t.contains_key(&id))
    }

    fn get(&self, comp: &str, id: u64) -> Option<Value> {
        self.comps.get(comp).and_then(|t| t.get(&id)).cloned()
    }

    fn set(&mut self, comp: &str, id: u64, value: Value) {
        self.comps.entry(comp.to_string()).or_default().insert(id, value);
    }
}

/// Control-flow signals that unwind expression evaluation.
enum Signal {
    Return(Value),
    Break(Value),
    Continue,
    Error(String),
}

type EvalResult = Result<Value, Signal>;

/// Run `entry` (default `main`) in `module`, returning its value and captured
/// output, or a runtime error string.
pub fn run(module: &Module, entry: &str) -> (Result<Value, String>, String) {
    let mut interp = Interp::new(module);
    if let Err(e) = interp.eval_globals() {
        return (Err(e), interp.output);
    }
    let result = match interp.call_named(entry, Vec::new()) {
        Ok(v) => Ok(v),
        Err(Signal::Error(e)) => Err(e),
        Err(Signal::Return(v)) => Ok(v),
        Err(_) => Err("`break`/`continue` outside of a loop".into()),
    };
    (result, interp.output)
}

/// Call a named function with pre-built argument values, returning its result.
/// Each call runs in a fresh world (suitable for pure functions).
pub fn call_fn(module: &Module, name: &str, args: Vec<Value>) -> Result<Value, String> {
    let mut interp = Interp::new(module);
    interp.eval_globals()?;
    match interp.call_named(name, args) {
        Ok(v) | Err(Signal::Return(v)) => Ok(v),
        Err(Signal::Error(e)) => Err(e),
        Err(_) => Err("`break`/`continue` outside of a loop".into()),
    }
}

struct Interp<'a> {
    fns: HashMap<String, &'a FnDecl>,
    /// Systems in declaration order (run by the `run_systems` builtin).
    systems: Vec<&'a SystemDecl>,
    /// Enum name -> set of its variant names (for constructing/matching).
    enums: HashMap<String, HashSet<String>>,
    /// (type name, method name) -> method, from `impl` blocks.
    methods: HashMap<(String, String), &'a FnDecl>,
    consts: Vec<(&'a str, &'a Expr)>,
    globals: HashMap<String, Value>,
    scopes: Vec<HashMap<String, Value>>,
    world: World,
    /// Software framebuffer for the builtin graphics commands.
    gfx: Option<aurora_gfx::Framebuffer>,
    pub output: String,
}

impl<'a> Interp<'a> {
    fn new(module: &'a Module) -> Interp<'a> {
        let mut fns = HashMap::new();
        let mut systems = Vec::new();
        let mut enums = HashMap::new();
        let mut methods = HashMap::new();
        let mut consts = Vec::new();
        for item in &module.items {
            match &item.kind {
                ItemKind::Fn(f) => {
                    fns.insert(f.name.name.clone(), f);
                }
                ItemKind::System(s) => systems.push(s),
                ItemKind::Enum(e) => {
                    let variants = e.variants.iter().map(|v| v.name.name.clone()).collect();
                    enums.insert(e.name.name.clone(), variants);
                }
                ItemKind::Impl(i) => {
                    if let TypeKind::Path(p) = &i.self_ty.kind {
                        if let Some(seg) = p.segments.last() {
                            let ty = seg.ident.name.clone();
                            for it in &i.items {
                                if let AssocItem::Fn(f) = it {
                                    methods.insert((ty.clone(), f.name.name.clone()), f);
                                }
                            }
                        }
                    }
                }
                ItemKind::Const(c) => consts.push((c.name.name.as_str(), &c.value)),
                _ => {}
            }
        }
        Interp {
            fns,
            systems,
            enums,
            methods,
            consts,
            globals: HashMap::new(),
            scopes: Vec::new(),
            world: World::default(),
            gfx: None,
            output: String::new(),
        }
    }

    fn eval_globals(&mut self) -> Result<(), String> {
        let consts = self.consts.clone();
        for (name, expr) in consts {
            self.scopes.push(HashMap::new());
            let v = self.eval(expr).map_err(signal_to_err)?;
            self.scopes.pop();
            self.globals.insert(name.to_string(), v);
        }
        Ok(())
    }

    // --- function calls ------------------------------------------------------

    fn call_named(&mut self, name: &str, args: Vec<Value>) -> EvalResult {
        let f = *self
            .fns
            .get(name)
            .ok_or_else(|| Signal::Error(format!("call to unknown function `{name}`")))?;
        let Some(body) = &f.body else {
            return Err(Signal::Error(format!("`{name}` has no body")));
        };

        // Bind positional params (self/typed params alike, by position).
        let mut frame = HashMap::new();
        let mut ai = args.into_iter();
        for p in &f.params {
            if let aurora_ast::Param::Normal { name, .. } = p {
                let v = ai.next().unwrap_or(Value::Unit);
                frame.insert(name.name.clone(), v);
            }
        }

        // Functions don't close over caller locals; swap in a fresh stack.
        let saved = std::mem::replace(&mut self.scopes, vec![frame]);
        let result = self.eval_block(body);
        self.scopes = saved;

        match result {
            Ok(v) => Ok(v),
            Err(Signal::Return(v)) => Ok(v),
            Err(other) => Err(other),
        }
    }

    // --- scopes --------------------------------------------------------------

    fn lookup(&self, name: &str) -> Option<Value> {
        self.scopes
            .iter()
            .rev()
            .find_map(|s| s.get(name).cloned())
            .or_else(|| self.globals.get(name).cloned())
    }

    fn bind(&mut self, name: &str, v: Value) {
        self.scopes.last_mut().unwrap().insert(name.to_string(), v);
    }

    fn var_mut(&mut self, name: &str) -> Result<&mut Value, Signal> {
        for s in self.scopes.iter_mut().rev() {
            if s.contains_key(name) {
                return Ok(s.get_mut(name).unwrap());
            }
        }
        if self.globals.contains_key(name) {
            return Ok(self.globals.get_mut(name).unwrap());
        }
        Err(Signal::Error(format!("assignment to unbound variable `{name}`")))
    }

    // --- blocks & statements -------------------------------------------------

    fn eval_block(&mut self, block: &Block) -> EvalResult {
        self.scopes.push(HashMap::new());
        let r = self.eval_block_inner(block);
        self.scopes.pop();
        r
    }

    fn eval_block_inner(&mut self, block: &Block) -> EvalResult {
        for stmt in &block.stmts {
            match stmt {
                Stmt::Let(l) => {
                    let v = match &l.init {
                        Some(e) => self.eval(e)?,
                        None => Value::Unit,
                    };
                    self.bind_pat(&l.pat, v)?;
                }
                Stmt::Defer(e) | Stmt::Expr(e) => {
                    self.eval(e)?;
                }
            }
        }
        match &block.tail {
            Some(e) => self.eval(e),
            None => Ok(Value::Unit),
        }
    }

    fn bind_pat(&mut self, pat: &Pat, v: Value) -> Result<(), Signal> {
        match &pat.kind {
            PatKind::Wild => Ok(()),
            PatKind::Binding { name, .. } => {
                self.bind(&name.name, v);
                Ok(())
            }
            PatKind::Tuple(pats) => match v {
                Value::Tuple(vals) if vals.len() == pats.len() => {
                    for (p, val) in pats.iter().zip(vals) {
                        self.bind_pat(p, val)?;
                    }
                    Ok(())
                }
                _ => Err(Signal::Error("tuple pattern does not match value".into())),
            },
            _ => Err(Signal::Error("unsupported pattern in interpreter".into())),
        }
    }

    // --- expressions ---------------------------------------------------------

    fn eval(&mut self, e: &Expr) -> EvalResult {
        match &e.kind {
            ExprKind::Int(n, _) => Ok(Value::Int(*n as i128)),
            ExprKind::Float(x, _) => Ok(Value::Float(*x)),
            ExprKind::Bool(b) => Ok(Value::Bool(*b)),
            ExprKind::Char(c) => Ok(Value::Char(*c)),
            ExprKind::Str(s) => Ok(Value::Str(s.clone())),
            ExprKind::Paren(inner) => self.eval(inner),
            ExprKind::SelfExpr => self
                .lookup("self")
                .ok_or_else(|| Signal::Error("`self` is not bound here".into())),
            ExprKind::Path(p) if p.is_single() => {
                let name = &p.segments[0].ident.name;
                self.lookup(name)
                    .ok_or_else(|| Signal::Error(format!("unknown name `{name}`")))
            }
            ExprKind::Path(p) => {
                // A qualified `Enum::Variant` unit variant.
                if let Some((enm, variant)) = self.enum_variant_of(p) {
                    Ok(Value::Enum { enm, variant, payload: Payload::Unit })
                } else {
                    Err(Signal::Error("unsupported path expression".into()))
                }
            }
            ExprKind::Unary(op, inner) => {
                let v = self.eval(inner)?;
                self.unary(*op, v)
            }
            ExprKind::Binary(op, a, b) => self.binary(*op, a, b),
            ExprKind::Assign(op, lhs, rhs) => {
                let rv = self.eval(rhs)?;
                let newv = match op {
                    None => rv,
                    Some(binop) => {
                        let cur = self.eval(lhs)?;
                        apply_arith(*binop, cur, rv).map_err(Signal::Error)?
                    }
                };
                self.assign(lhs, newv)?;
                Ok(Value::Unit)
            }
            ExprKind::Block(b) | ExprKind::Unsafe(b) => self.eval_block(b),
            ExprKind::If(ifx) => {
                let cond = self.eval(&ifx.cond)?;
                if self.is_true(cond, &ifx.cond)? {
                    self.eval_block(&ifx.then_branch)
                } else if let Some(else_e) = &ifx.else_branch {
                    self.eval(else_e)
                } else {
                    Ok(Value::Unit)
                }
            }
            ExprKind::While { cond, body } => {
                loop {
                    let c = self.eval(cond)?;
                    if !self.is_true(c, cond)? {
                        break;
                    }
                    match self.eval_block(body) {
                        Ok(_) => {}
                        Err(Signal::Break(_)) => break,
                        Err(Signal::Continue) => continue,
                        Err(other) => return Err(other),
                    }
                }
                Ok(Value::Unit)
            }
            ExprKind::Loop(body) => loop {
                match self.eval_block(body) {
                    Ok(_) => {}
                    Err(Signal::Break(v)) => return Ok(v),
                    Err(Signal::Continue) => continue,
                    Err(other) => return Err(other),
                }
            },
            ExprKind::For { pat, iter, body } => self.eval_for(pat, iter, body),
            ExprKind::Match { scrutinee, arms } => {
                let v = self.eval(scrutinee)?;
                for arm in arms {
                    self.scopes.push(HashMap::new());
                    let matched = self.pat_matches(&arm.pat, &v)?;
                    let guard_ok = if matched {
                        match &arm.guard {
                            Some(g) => self.eval(g)?.truthy().unwrap_or(false),
                            None => true,
                        }
                    } else {
                        false
                    };
                    if guard_ok {
                        let r = self.eval(&arm.body);
                        self.scopes.pop();
                        return r;
                    }
                    self.scopes.pop();
                }
                Err(Signal::Error("no match arm matched the value".into()))
            }
            ExprKind::Call { callee, args, .. } => self.eval_call(callee, args),
            ExprKind::Tuple(items) => {
                let vals: Result<Vec<_>, _> = items.iter().map(|e| self.eval(e)).collect();
                Ok(Value::Tuple(vals?))
            }
            ExprKind::Array(items) => {
                let vals: Result<Vec<_>, _> = items.iter().map(|e| self.eval(e)).collect();
                Ok(Value::Array(vals?))
            }
            ExprKind::ArrayRepeat { value, count } => {
                let v = self.eval(value)?;
                let n = match self.eval(count)? {
                    Value::Int(n) if n >= 0 => n as usize,
                    _ => return Err(Signal::Error("array repeat count must be a non-negative int".into())),
                };
                Ok(Value::Array(vec![v; n]))
            }
            ExprKind::Index { base, index } => {
                let b = self.eval(base)?;
                let i = self.eval(index)?;
                index_value(b, i).map_err(Signal::Error)
            }
            ExprKind::Field { base, field } => {
                let b = self.eval(base)?;
                field_value(b, field).map_err(Signal::Error)
            }
            ExprKind::Struct { path, fields, base } => self.eval_struct(path, fields, base.as_deref()),
            ExprKind::Return(opt) => {
                let v = match opt {
                    Some(e) => self.eval(e)?,
                    None => Value::Unit,
                };
                Err(Signal::Return(v))
            }
            ExprKind::Break(opt) => {
                let v = match opt {
                    Some(e) => self.eval(e)?,
                    None => Value::Unit,
                };
                Err(Signal::Break(v))
            }
            ExprKind::Continue => Err(Signal::Continue),
            ExprKind::Pipe { value, func } => {
                // `x |> f(a)` == `f(x, a)`; `x |> f` == `f(x)`.
                let piped = self.eval(value)?;
                match &func.kind {
                    ExprKind::Call { callee, args, .. } => {
                        let mut argv = vec![piped];
                        for a in args {
                            argv.push(self.eval(&a.value)?);
                        }
                        self.dispatch_call(callee, argv)
                    }
                    // `x |> f` — f is a callee (fn name or closure expr).
                    _ => self.dispatch_call(func, vec![piped]),
                }
            }
            ExprKind::Closure { params, body } => {
                // Capture the currently-visible bindings by value.
                let mut env = BTreeMap::new();
                for scope in &self.scopes {
                    for (k, v) in scope {
                        env.insert(k.clone(), v.clone());
                    }
                }
                let names = params
                    .iter()
                    .filter_map(|p| match p {
                        aurora_ast::Param::Normal { name, .. } => Some(name.name.clone()),
                        aurora_ast::Param::SelfParam { .. } => None,
                    })
                    .collect();
                Ok(Value::Closure { params: names, body: Box::new((**body).clone()), env })
            }
            ExprKind::Spawn(args) => {
                let mut comps = Vec::with_capacity(args.len());
                for a in args {
                    comps.push(self.eval(&a.value)?);
                }
                Ok(Value::Entity(self.world.spawn(comps)))
            }
            ExprKind::Despawn(inner) => {
                match self.eval(inner)? {
                    Value::Entity(id) => self.world.despawn(id),
                    other => {
                        return Err(Signal::Error(format!(
                            "despawn expects an entity, got {}",
                            other.type_name()
                        )))
                    }
                }
                Ok(Value::Unit)
            }
            other => Err(Signal::Error(format!("unsupported expression: {}", describe(other)))),
        }
    }

    fn eval_for(&mut self, pat: &Pat, iter: &Expr, body: &Block) -> EvalResult {
        if let ExprKind::Query(q) = &iter.kind {
            return self.eval_query_loop(pat, q, body);
        }
        let items: Vec<Value> = match &iter.kind {
            ExprKind::Range { start, end, inclusive } => {
                let s = match start.as_ref().map(|e| self.eval(e)).transpose()? {
                    Some(Value::Int(n)) => n,
                    _ => return Err(Signal::Error("range start must be an int".into())),
                };
                let e = match end.as_ref().map(|e| self.eval(e)).transpose()? {
                    Some(Value::Int(n)) => n,
                    _ => return Err(Signal::Error("range end must be an int".into())),
                };
                let hi = if *inclusive { e + 1 } else { e };
                (s..hi).map(Value::Int).collect()
            }
            _ => match self.eval(iter)? {
                Value::Array(items) => items,
                other => {
                    return Err(Signal::Error(format!(
                        "cannot iterate a {} (interpreter supports ranges and arrays)",
                        other.type_name()
                    )))
                }
            },
        };

        for item in items {
            self.scopes.push(HashMap::new());
            let bound = self.bind_pat(pat, item);
            if let Err(e) = bound {
                self.scopes.pop();
                return Err(e);
            }
            let r = self.eval_block_inner(body);
            self.scopes.pop();
            match r {
                Ok(_) => {}
                Err(Signal::Break(_)) => break,
                Err(Signal::Continue) => continue,
                Err(other) => return Err(other),
            }
        }
        Ok(Value::Unit)
    }

    /// Execute a `for <pat> in query<...> { body }` over the world: match
    /// entities holding the required components, bind the pattern, run the body,
    /// then write back any `&mut` components.
    fn eval_query_loop(&mut self, pat: &Pat, q: &QueryExpr, body: &Block) -> EvalResult {
        // Classify terms.
        struct DataTerm {
            comp: Option<String>, // None for Entity
            write: bool,
        }
        let mut required: Vec<String> = Vec::new();
        let mut excluded: Vec<String> = Vec::new();
        let mut data: Vec<DataTerm> = Vec::new();
        for term in &q.terms {
            match term {
                QTerm::Read(p) => {
                    let c = comp_name(p);
                    required.push(c.clone());
                    data.push(DataTerm { comp: Some(c), write: false });
                }
                QTerm::Write(p) => {
                    let c = comp_name(p);
                    required.push(c.clone());
                    data.push(DataTerm { comp: Some(c), write: true });
                }
                QTerm::OptRead(p) => data.push(DataTerm { comp: Some(comp_name(p)), write: false }),
                QTerm::OptWrite(p) => data.push(DataTerm { comp: Some(comp_name(p)), write: true }),
                QTerm::Entity => data.push(DataTerm { comp: None, write: false }),
                QTerm::With(p) => required.push(comp_name(p)),
                QTerm::Without(p) => excluded.push(comp_name(p)),
            }
        }

        let bindings = pattern_bindings(pat);

        // Snapshot matching entity ids first (body may spawn/despawn).
        let matches: Vec<u64> = self
            .world
            .entities
            .iter()
            .copied()
            .filter(|&id| {
                required.iter().all(|c| self.world.has(c, id))
                    && !excluded.iter().any(|c| self.world.has(c, id))
            })
            .collect();

        let filter = q.filter.clone();

        for id in matches {
            self.scopes.push(HashMap::new());
            // Bind each data term to its pattern position.
            for (i, term) in data.iter().enumerate() {
                let value = match &term.comp {
                    Some(c) => self.world.get(c, id).unwrap_or(Value::Unit),
                    None => Value::Entity(id),
                };
                if let Some(Some(name)) = bindings.get(i) {
                    self.bind(name, value);
                }
            }

            // Optional `where` filter.
            let run = match &filter {
                Some(f) => match self.eval(f) {
                    Ok(v) => v.truthy().unwrap_or(true),
                    Err(e) => {
                        self.scopes.pop();
                        return Err(e);
                    }
                },
                None => true,
            };

            let mut flow = Ok(Value::Unit);
            if run {
                flow = self.eval_block_inner(body);
            }

            // Write back mutated components (only for `&mut` terms with a name).
            for (i, term) in data.iter().enumerate() {
                if !term.write {
                    continue;
                }
                if let (Some(c), Some(Some(name))) = (&term.comp, bindings.get(i)) {
                    if let Some(v) = self.lookup(name) {
                        self.world.set(c, id, v);
                    }
                }
            }
            self.scopes.pop();

            match flow {
                Ok(_) => {}
                Err(Signal::Break(_)) => break,
                Err(Signal::Continue) => continue,
                Err(other) => return Err(other),
            }
        }
        Ok(Value::Unit)
    }

    fn eval_call(&mut self, callee: &Expr, args: &[aurora_ast::Arg]) -> EvalResult {
        let mut argv = Vec::with_capacity(args.len());
        for a in args {
            argv.push(self.eval(&a.value)?);
        }
        self.dispatch_call(callee, argv)
    }

    /// Dispatch a call given an already-evaluated argument list. Shared by
    /// direct calls and the pipe operator.
    fn dispatch_call(&mut self, callee: &Expr, argv: Vec<Value>) -> EvalResult {
        // Method call `recv.method(args)`: resolve against `impl` blocks by the
        // receiver's runtime type; otherwise the field may hold a closure.
        if let ExprKind::Field { base, field: FieldAccess::Named(m) } = &callee.kind {
            let recv = self.eval(base)?;
            if let Some(ty) = value_type_name(&recv) {
                if let Some(method) = self.methods.get(&(ty, m.name.clone())).copied() {
                    return self.call_method(method, recv, argv);
                }
            }
            let field = field_value(recv, &FieldAccess::Named(m.clone())).map_err(Signal::Error)?;
            return self.invoke_value(field, argv);
        }

        // A qualified `Enum::Variant(args)` tuple-variant constructor.
        if let ExprKind::Path(p) = &callee.kind {
            if let Some((enm, variant)) = self.enum_variant_of(p) {
                return Ok(Value::Enum { enm, variant, payload: Payload::Tuple(argv) });
            }
        }

        // A single-name callee may be a builtin, a top-level fn, or a local
        // binding holding a closure.
        let name = match &callee.kind {
            ExprKind::Path(p) if p.is_single() => Some(p.segments[0].ident.name.clone()),
            _ => None,
        };

        if name.is_none() {
            // Indirect call: the callee expression must evaluate to a closure.
            let v = self.eval(callee)?;
            return self.invoke_value(v, argv);
        }
        let name = name.unwrap();

        match name.as_str() {
            "print" | "println" => {
                let line: Vec<String> = argv.iter().map(|v| v.to_string()).collect();
                self.output.push_str(&line.join(" "));
                if name == "println" {
                    self.output.push('\n');
                }
                Ok(Value::Unit)
            }
            "assert" => match argv.first() {
                Some(Value::Bool(true)) => Ok(Value::Unit),
                Some(Value::Bool(false)) => Err(Signal::Error("assertion failed".into())),
                _ => Err(Signal::Error("assert expects a bool".into())),
            },
            // ECS builtins. `spawn`/`despawn` are ordinary call syntax in the
            // surface language (not keywords), so they're dispatched here.
            "spawn" => Ok(Value::Entity(self.world.spawn(argv))),
            "despawn" => match argv.into_iter().next() {
                Some(Value::Entity(id)) => {
                    self.world.despawn(id);
                    Ok(Value::Unit)
                }
                _ => Err(Signal::Error("despawn expects an entity".into())),
            },
            // The engine loop isn't modelled in the interpreter, so these drive
            // the world manually.
            "run_systems" => {
                let systems: Vec<&SystemDecl> = self.systems.clone();
                for sys in systems {
                    self.run_system(sys)?;
                }
                Ok(Value::Unit)
            }
            // Builtin graphics — an Aurora program drives the CPU rasterizer.
            "framebuffer" => {
                let w = int_arg(&argv, 0).max(0) as u32;
                let h = int_arg(&argv, 1).max(0) as u32;
                self.gfx = Some(aurora_gfx::Framebuffer::new(w, h));
                Ok(Value::Unit)
            }
            "clear" => {
                let c = color_arg(&argv, 0);
                if let Some(fb) = &mut self.gfx {
                    fb.clear(c);
                }
                Ok(Value::Unit)
            }
            "pixel" => {
                let (x, y) = (int_arg(&argv, 0) as i32, int_arg(&argv, 1) as i32);
                let c = color_arg(&argv, 2);
                if let Some(fb) = &mut self.gfx {
                    fb.set(x, y, c);
                }
                Ok(Value::Unit)
            }
            "triangle" => {
                let p = [
                    [int_arg(&argv, 0) as f32, int_arg(&argv, 1) as f32],
                    [int_arg(&argv, 2) as f32, int_arg(&argv, 3) as f32],
                    [int_arg(&argv, 4) as f32, int_arg(&argv, 5) as f32],
                ];
                let c = color_arg(&argv, 6);
                if let Some(fb) = &mut self.gfx {
                    fb.triangle(p, [c, c, c]);
                }
                Ok(Value::Unit)
            }
            "fb_get" => {
                // Packed 0xRRGGBB of the pixel — lets programs inspect the image.
                let (x, y) = (int_arg(&argv, 0) as u32, int_arg(&argv, 1) as u32);
                let packed = match &self.gfx {
                    Some(fb) if x < fb.width() && y < fb.height() => {
                        let c = fb.get(x, y);
                        ((c.r as i128) << 16) | ((c.g as i128) << 8) | c.b as i128
                    }
                    _ => 0,
                };
                Ok(Value::Int(packed))
            }
            "save_ppm" => {
                let path = match argv.first() {
                    Some(Value::Str(s)) => s.clone(),
                    _ => return Err(Signal::Error("save_ppm expects a path string".into())),
                };
                match &self.gfx {
                    Some(fb) => std::fs::write(&path, fb.to_ppm())
                        .map(|_| Value::Unit)
                        .map_err(|e| Signal::Error(format!("save_ppm: {e}"))),
                    None => Err(Signal::Error("no framebuffer; call framebuffer(w, h) first".into())),
                }
            }
            // Math builtins (games need real math). Float-returning unless the
            // inputs are all integers for abs/min/max.
            "sqrt" => Ok(Value::Float(f64_arg(&argv, 0).sqrt())),
            "sin" => Ok(Value::Float(f64_arg(&argv, 0).sin())),
            "cos" => Ok(Value::Float(f64_arg(&argv, 0).cos())),
            "tan" => Ok(Value::Float(f64_arg(&argv, 0).tan())),
            "floor" => Ok(Value::Float(f64_arg(&argv, 0).floor())),
            "ceil" => Ok(Value::Float(f64_arg(&argv, 0).ceil())),
            // round-half-to-even, matching the native backend's `nearest`.
            "round" => Ok(Value::Float(f64_arg(&argv, 0).round_ties_even())),
            "pow" => Ok(Value::Float(f64_arg(&argv, 0).powf(f64_arg(&argv, 1)))),
            "abs" => Ok(match argv.first() {
                Some(Value::Int(n)) => Value::Int(n.abs()),
                v => Value::Float(v.map(|v| as_f64(v).abs()).unwrap_or(0.0)),
            }),
            "min" | "max" => {
                let is_max = name == "max";
                Ok(match (argv.first(), argv.get(1)) {
                    (Some(Value::Int(a)), Some(Value::Int(b))) => {
                        Value::Int(if is_max { *a.max(b) } else { *a.min(b) })
                    }
                    _ => {
                        let (a, b) = (f64_arg(&argv, 0), f64_arg(&argv, 1));
                        Value::Float(if is_max { a.max(b) } else { a.min(b) })
                    }
                })
            }
            "clamp" => {
                let (v, lo, hi) = (f64_arg(&argv, 0), f64_arg(&argv, 1), f64_arg(&argv, 2));
                Ok(Value::Float(v.clamp(lo, hi)))
            }
            // Integer bitwise ops (the native backend has these; the bundled
            // stdlib `rgb`/`red`/... call them, so the interpreter needs them too).
            "band" | "bor" | "bxor" | "shl" | "shr" => {
                let a = int_arg(&argv, 0);
                let b = int_arg(&argv, 1);
                let r = match name.as_str() {
                    "band" => a & b,
                    "bor" => a | b,
                    "bxor" => a ^ b,
                    "shl" => a.wrapping_shl(b as u32),
                    _ => a.wrapping_shr(b as u32),
                };
                Ok(Value::Int(r as i128))
            }
            "bnot" => Ok(Value::Int(!int_arg(&argv, 0) as i128)),
            "str" => Ok(Value::Str(argv.first().map(|v| v.to_string()).unwrap_or_default())),
            "len" => Ok(Value::Int(match argv.first() {
                Some(Value::Array(v)) | Some(Value::Tuple(v)) => v.len() as i128,
                Some(Value::Str(s)) => s.chars().count() as i128,
                _ => 0,
            })),
            "entity_count" => Ok(Value::Int(self.world.entities.len() as i128)),
            _ => {
                // A top-level function, or a local binding holding a closure.
                if self.fns.contains_key(&name) {
                    self.call_named(&name, argv)
                } else if let Some(v @ Value::Closure { .. }) = self.lookup(&name) {
                    self.invoke_value(v, argv)
                } else {
                    self.call_named(&name, argv) // reports "unknown function"
                }
            }
        }
    }

    /// Call an `impl` method with `recv` bound to `self`.
    fn call_method(&mut self, method: &FnDecl, recv: Value, argv: Vec<Value>) -> EvalResult {
        let Some(body) = &method.body else {
            return Err(Signal::Error(format!("method `{}` has no body", method.name.name)));
        };
        let mut frame = HashMap::new();
        frame.insert("self".to_string(), recv);
        let mut ai = argv.into_iter();
        for p in &method.params {
            if let aurora_ast::Param::Normal { name, .. } = p {
                frame.insert(name.name.clone(), ai.next().unwrap_or(Value::Unit));
            }
        }
        let saved = std::mem::replace(&mut self.scopes, vec![frame]);
        let result = self.eval_block(body);
        self.scopes = saved;
        match result {
            Ok(v) | Err(Signal::Return(v)) => Ok(v),
            Err(other) => Err(other),
        }
    }

    /// Invoke a value as a function (it must be a closure).
    fn invoke_value(&mut self, callee: Value, argv: Vec<Value>) -> EvalResult {
        let Value::Closure { params, body, env } = callee else {
            return Err(Signal::Error(format!("value of type {} is not callable", callee.type_name())));
        };
        let mut frame: HashMap<String, Value> = env.into_iter().collect();
        for (p, v) in params.iter().zip(argv.into_iter()) {
            frame.insert(p.clone(), v);
        }
        let saved = std::mem::replace(&mut self.scopes, vec![frame]);
        let result = self.eval(&body);
        self.scopes = saved;
        match result {
            Ok(v) | Err(Signal::Return(v)) => Ok(v),
            Err(other) => Err(other),
        }
    }

    /// Run one system body once, with its parameters bound to `Unit` (the
    /// interpreter has no resource/time providers yet).
    fn run_system(&mut self, sys: &SystemDecl) -> Result<(), Signal> {
        let mut frame = HashMap::new();
        for p in &sys.params {
            frame.insert(p.name.name.clone(), Value::Unit);
        }
        let saved = std::mem::replace(&mut self.scopes, vec![frame]);
        let result = self.eval_block(&sys.body);
        self.scopes = saved;
        match result {
            Ok(_) | Err(Signal::Return(_)) => Ok(()),
            Err(other) => Err(other),
        }
    }

    fn eval_struct(
        &mut self,
        path: &aurora_ast::Path,
        fields: &[aurora_ast::FieldInit],
        base: Option<&Expr>,
    ) -> EvalResult {
        let enum_variant = self.enum_variant_of(path);
        let name = path.segments.last().map(|s| s.ident.name.clone()).unwrap_or_default();
        let mut map = BTreeMap::new();
        if let Some(b) = base {
            if let Value::Struct(_, base_fields) = self.eval(b)? {
                map = base_fields;
            }
        }
        for f in fields {
            let v = match &f.value {
                Some(e) => self.eval(e)?,
                None => self
                    .lookup(&f.name.name)
                    .ok_or_else(|| Signal::Error(format!("unknown name `{}`", f.name.name)))?,
            };
            map.insert(f.name.name.clone(), v);
        }
        match enum_variant {
            Some((enm, variant)) => Ok(Value::Enum { enm, variant, payload: Payload::Struct(map) }),
            None => Ok(Value::Struct(name, map)),
        }
    }

    /// If `path` is a qualified `Enum::Variant` for a known enum, return the
    /// `(enum, variant)` names.
    fn enum_variant_of(&self, path: &aurora_ast::Path) -> Option<(String, String)> {
        if path.segments.len() == 2 {
            let a = &path.segments[0].ident.name;
            let b = &path.segments[1].ident.name;
            if self.enums.get(a).is_some_and(|vs| vs.contains(b)) {
                return Some((a.clone(), b.clone()));
            }
        }
        None
    }

    /// Try to match `value` against `pat`, binding names into the current scope
    /// on (partial) success. Returns whether the whole pattern matched.
    fn pat_matches(&mut self, pat: &Pat, value: &Value) -> Result<bool, Signal> {
        match &pat.kind {
            PatKind::Wild | PatKind::Rest => Ok(true),
            PatKind::Binding { name, sub } => {
                self.bind(&name.name, value.clone());
                match sub {
                    Some(s) => self.pat_matches(s, value),
                    None => Ok(true),
                }
            }
            PatKind::Lit(expr) => {
                let lit = self.eval(expr)?;
                Ok(&lit == value)
            }
            PatKind::Tuple(pats) => match value {
                Value::Tuple(vals) if vals.len() == pats.len() => {
                    for (p, v) in pats.iter().zip(vals) {
                        if !self.pat_matches(p, v)? {
                            return Ok(false);
                        }
                    }
                    Ok(true)
                }
                _ => Ok(false),
            },
            PatKind::Struct { path, fields, .. } => {
                // Field map of a struct value, or an enum struct-variant payload.
                let ev = self.enum_variant_of(path);
                let map = match (value, &ev) {
                    (Value::Struct(name, map), None)
                        if Some(name.as_str()) == path.segments.last().map(|s| s.ident.name.as_str()) =>
                    {
                        map.clone()
                    }
                    (Value::Enum { enm, variant, payload: Payload::Struct(map) }, Some((e, v)))
                        if enm == e && variant == v =>
                    {
                        map.clone()
                    }
                    _ => return Ok(false),
                };
                for fp in fields {
                    let Some(fv) = map.get(&fp.name.name) else {
                        return Ok(false);
                    };
                    match &fp.pat {
                        Some(p) => {
                            if !self.pat_matches(p, &fv.clone())? {
                                return Ok(false);
                            }
                        }
                        None => self.bind(&fp.name.name, fv.clone()),
                    }
                }
                Ok(true)
            }
            PatKind::Path(p) => {
                // Qualified unit variant, e.g. `Color::Red`.
                match (self.enum_variant_of(p), value) {
                    (Some((e, v)), Value::Enum { enm, variant, .. }) => {
                        Ok(enm == &e && variant == &v)
                    }
                    _ => Ok(false),
                }
            }
            PatKind::TupleStruct { path, elems } => {
                match (self.enum_variant_of(path), value) {
                    (Some((e, v)), Value::Enum { enm, variant, payload: Payload::Tuple(vals) })
                        if enm == &e && variant == &v && vals.len() == elems.len() =>
                    {
                        for (p, val) in elems.iter().zip(vals.clone()) {
                            if !self.pat_matches(p, &val)? {
                                return Ok(false);
                            }
                        }
                        Ok(true)
                    }
                    _ => Ok(false),
                }
            }
            PatKind::Error => Ok(false),
        }
    }

    // --- assignment ----------------------------------------------------------

    fn assign(&mut self, lhs: &Expr, newv: Value) -> Result<(), Signal> {
        let (root, accessors) = self.place(lhs)?;
        let slot = self.var_mut(&root)?;
        let mut cur = slot;
        for acc in &accessors {
            cur = match (cur, acc) {
                (Value::Struct(_, m), Acc::Field(f)) => m
                    .get_mut(f)
                    .ok_or_else(|| Signal::Error(format!("no field `{f}`")))?,
                (Value::Tuple(items), Acc::Index(i)) | (Value::Array(items), Acc::Index(i)) => items
                    .get_mut(*i)
                    .ok_or_else(|| Signal::Error("index out of bounds".into()))?,
                _ => return Err(Signal::Error("invalid assignment target".into())),
            };
        }
        *cur = newv;
        Ok(())
    }

    fn place(&mut self, e: &Expr) -> Result<(String, Vec<Acc>), Signal> {
        match &e.kind {
            ExprKind::Path(p) if p.is_single() => Ok((p.segments[0].ident.name.clone(), Vec::new())),
            ExprKind::Field { base, field } => {
                let (root, mut accs) = self.place(base)?;
                accs.push(match field {
                    FieldAccess::Named(id) => Acc::Field(id.name.clone()),
                    FieldAccess::Index(i) => Acc::Index(*i as usize),
                });
                Ok((root, accs))
            }
            ExprKind::Index { base, index } => {
                let idx = match self.eval(index)? {
                    Value::Int(n) if n >= 0 => n as usize,
                    _ => return Err(Signal::Error("index must be a non-negative int".into())),
                };
                let (root, mut accs) = self.place(base)?;
                accs.push(Acc::Index(idx));
                Ok((root, accs))
            }
            _ => Err(Signal::Error("invalid assignment target".into())),
        }
    }

    // --- operators -----------------------------------------------------------

    fn unary(&mut self, op: UnOp, v: Value) -> EvalResult {
        match (op, v) {
            (UnOp::Neg, Value::Int(n)) => Ok(Value::Int(-n)),
            (UnOp::Neg, Value::Float(x)) => Ok(Value::Float(-x)),
            (UnOp::Not, Value::Bool(b)) => Ok(Value::Bool(!b)),
            (op, v) => Err(Signal::Error(format!("cannot apply {op:?} to {}", v.type_name()))),
        }
    }

    fn binary(&mut self, op: BinOp, a: &Expr, b: &Expr) -> EvalResult {
        // Short-circuit logical operators.
        if matches!(op, BinOp::And | BinOp::Or) {
            let lv = self.eval(a)?;
            let lb = lv
                .truthy()
                .ok_or_else(|| Signal::Error("logical operator needs a bool".into()))?;
            if op == BinOp::And && !lb {
                return Ok(Value::Bool(false));
            }
            if op == BinOp::Or && lb {
                return Ok(Value::Bool(true));
            }
            let rv = self.eval(b)?;
            let rb = rv
                .truthy()
                .ok_or_else(|| Signal::Error("logical operator needs a bool".into()))?;
            return Ok(Value::Bool(rb));
        }

        let lv = self.eval(a)?;
        let rv = self.eval(b)?;
        match op {
            BinOp::Eq => Ok(Value::Bool(lv == rv)),
            BinOp::Ne => Ok(Value::Bool(lv != rv)),
            BinOp::Lt | BinOp::Gt | BinOp::Le | BinOp::Ge => {
                compare(op, lv, rv).map_err(Signal::Error)
            }
            _ => apply_arith(op, lv, rv).map_err(Signal::Error),
        }
    }

    fn is_true(&self, v: Value, _at: &Expr) -> Result<bool, Signal> {
        v.truthy().ok_or_else(|| Signal::Error("condition must be a bool".into()))
    }
}

enum Acc {
    Field(String),
    Index(usize),
}

fn apply_arith(op: BinOp, a: Value, b: Value) -> Result<Value, String> {
    match (a, b) {
        (Value::Int(x), Value::Int(y)) => {
            // Match the native backend: wrap at i64 (not i128 overflow-panic).
            let (xi, yi) = (x as i64, y as i64);
            let r = match op {
                BinOp::Add => xi.wrapping_add(yi),
                BinOp::Sub => xi.wrapping_sub(yi),
                BinOp::Mul => xi.wrapping_mul(yi),
                BinOp::Div => {
                    if yi == 0 {
                        return Err("division by zero".into());
                    }
                    xi.wrapping_div(yi)
                }
                BinOp::Rem => {
                    if yi == 0 {
                        return Err("remainder by zero".into());
                    }
                    xi.wrapping_rem(yi)
                }
                _ => return Err("not an arithmetic operator".into()),
            };
            Ok(Value::Int(r as i128))
        }
        (Value::Float(x), Value::Float(y)) => Ok(Value::Float(match op {
            BinOp::Add => x + y,
            BinOp::Sub => x - y,
            BinOp::Mul => x * y,
            BinOp::Div => x / y,
            BinOp::Rem => x % y,
            _ => return Err("not an arithmetic operator".into()),
        })),
        (Value::Str(x), Value::Str(y)) if op == BinOp::Add => Ok(Value::Str(x + &y)),
        (a, b) => Err(format!("cannot apply {op:?} to {} and {}", a.type_name(), b.type_name())),
    }
}

fn compare(op: BinOp, a: Value, b: Value) -> Result<Value, String> {
    let ord = match (&a, &b) {
        (Value::Int(x), Value::Int(y)) => x.partial_cmp(y),
        (Value::Float(x), Value::Float(y)) => x.partial_cmp(y),
        (Value::Char(x), Value::Char(y)) => x.partial_cmp(y),
        (Value::Str(x), Value::Str(y)) => x.partial_cmp(y),
        _ => return Err(format!("cannot compare {} and {}", a.type_name(), b.type_name())),
    };
    let Some(ord) = ord else {
        return Ok(Value::Bool(false));
    };
    use std::cmp::Ordering::*;
    Ok(Value::Bool(match op {
        BinOp::Lt => ord == Less,
        BinOp::Gt => ord == Greater,
        BinOp::Le => ord != Greater,
        BinOp::Ge => ord != Less,
        _ => false,
    }))
}

fn index_value(base: Value, index: Value) -> Result<Value, String> {
    let i = match index {
        Value::Int(n) if n >= 0 => n as usize,
        _ => return Err("index must be a non-negative int".into()),
    };
    match base {
        Value::Array(items) | Value::Tuple(items) => {
            items.into_iter().nth(i).ok_or_else(|| "index out of bounds".into())
        }
        other => Err(format!("cannot index a {}", other.type_name())),
    }
}

fn field_value(base: Value, field: &FieldAccess) -> Result<Value, String> {
    match (base, field) {
        (Value::Struct(_, fields), FieldAccess::Named(id)) => {
            fields.get(&id.name).cloned().ok_or_else(|| format!("no field `{}`", id.name))
        }
        (Value::Tuple(items), FieldAccess::Index(i)) => {
            items.into_iter().nth(*i as usize).ok_or_else(|| "tuple index out of bounds".into())
        }
        (other, _) => Err(format!("cannot access a field of {}", other.type_name())),
    }
}

/// A numeric value as f64 (0.0 if not numeric).
fn as_f64(v: &Value) -> f64 {
    match v {
        Value::Float(x) => *x,
        Value::Int(n) => *n as f64,
        _ => 0.0,
    }
}

/// Read argument `i` as f64 (0.0 if missing / wrong type).
fn f64_arg(argv: &[Value], i: usize) -> f64 {
    argv.get(i).map(as_f64).unwrap_or(0.0)
}

/// Read argument `i` as an integer (0 if missing / wrong type).
fn int_arg(argv: &[Value], i: usize) -> i64 {
    match argv.get(i) {
        Some(Value::Int(n)) => *n as i64,
        Some(Value::Float(x)) => *x as i64,
        _ => 0,
    }
}

/// Read three consecutive integer args starting at `i` as an RGB color.
fn color_arg(argv: &[Value], i: usize) -> aurora_gfx::Color {
    let ch = |k: usize| int_arg(argv, k).clamp(0, 255) as u8;
    aurora_gfx::Color::rgb(ch(i), ch(i + 1), ch(i + 2))
}

/// The nominal type name of a value, for method resolution.
fn value_type_name(v: &Value) -> Option<String> {
    match v {
        Value::Struct(name, _) => Some(name.clone()),
        Value::Enum { enm, .. } => Some(enm.clone()),
        _ => None,
    }
}

/// The component name a query path refers to (its last segment).
fn comp_name(p: &aurora_ast::Path) -> String {
    p.segments.last().map(|s| s.ident.name.clone()).unwrap_or_default()
}

/// Positional binding names from a for-loop pattern: `(t, s)` -> [t, s],
/// a single `x` -> [x], `_` -> [None].
fn pattern_bindings(pat: &Pat) -> Vec<Option<String>> {
    match &pat.kind {
        PatKind::Tuple(pats) => pats.iter().map(binding_name).collect(),
        _ => vec![binding_name(pat)],
    }
}

fn binding_name(pat: &Pat) -> Option<String> {
    match &pat.kind {
        PatKind::Binding { name, .. } => Some(name.name.clone()),
        _ => None,
    }
}

fn signal_to_err(s: Signal) -> String {
    match s {
        Signal::Error(e) => e,
        _ => "unexpected control flow at top level".into(),
    }
}

fn describe(kind: &ExprKind) -> &'static str {
    match kind {
        ExprKind::Query(_) => "a bare query (use it as `for .. in query<..>`)",
        ExprKind::Region { .. } => "region allocation (needs the region runtime)",
        ExprKind::Closure { .. } => "closure",
        ExprKind::Pipe { .. } => "pipe",
        ExprKind::Cast(..) => "cast",
        _ => "this construct",
    }
}

#[cfg(test)]
mod tests;
