# Aurora — Grammar & Type System Specification

> Version 0.1 (design). Target: a statically-typed, AOT-compiled, GC-free language
> for game development with first-class ECS, GPU shaders, and replication.

This document is the normative reference for **syntax** (EBNF) and **static
semantics** (the type system). It is written to be *implementable*: every rule
here maps to a phase in the frontend (lexer → parser → resolver → typer → region
checker → borrow checker).

---

## 0. Design invariants the grammar must protect

1. **One source language for CPU and GPU.** Shader functions are ordinary
   functions tagged with a stage attribute; they parse with the same grammar but
   are type-checked against a restricted GPU sub-environment (§7.6).
2. **No GC.** All heap data is either single-owner (`~T`), refcounted (`rc<T>`),
   or region-bound (`#frame`/`#level`/`#perm`). The grammar makes the region a
   syntactic property of allocation, so the checker never has to guess.
3. **Borrow checking is opt-in.** Value types (POD structs, primitives) copy
   freely. Borrow/lifetime rules apply *only* to references into owned/region
   memory (§8). Everyday gameplay code never sees a lifetime annotation.
4. **ECS is syntax, not a library.** `component`, `system`, and `query` are
   keywords so the scheduler/codegen can reason about data access (§6).

---

## 1. Notation

EBNF conventions used below:

```
A ::= B          definition
A | B            alternation
A B              concatenation (sequence)
[ A ]            optional
{ A }            zero or more
( A )            grouping
"kw"             terminal keyword / literal
A?               equivalent to [ A ]
A*               equivalent to { A }
A+               one or more
A ~ sep          one or more A separated by sep (trailing sep allowed)
```

Whitespace and comments are insignificant except as token separators. Newlines
are **not** statement terminators; statements end at `;` *or* are inferred by the
parser via offside/expression-completion (ASI-style but conservative — see §3.3).

---

## 2. Lexical grammar

### 2.1 Comments
```
LineComment  ::= "//" {any char except newline}
BlockComment ::= "/*" {any char} "*/"          // nestable
DocComment   ::= "///" {any char except newline}
```

### 2.2 Identifiers & keywords
```
ident   ::= (letter | "_") {letter | digit | "_"}
letter  ::= "a".."z" | "A".."Z"
digit   ::= "0".."9"
```

Reserved keywords:
```
let mut fn return if else match for while loop break continue
struct enum component system trait impl pipeline use mod pub
true false comptime unsafe self Self as in where defer spawn despawn
query and or not
```

Type-level / contextual keywords (not reserved as identifiers everywhere):
```
f32 f64 i8 i16 i32 i64 u8 u16 u32 u64 bool char str void
Vec2 Vec3 Vec4 Mat2 Mat3 Mat4 Quat Color Transform Time Entity
```

### 2.3 Literals
```
IntLit    ::= digit {digit | "_"} [IntSuffix]
            | "0x" hexdigit {hexdigit | "_"}
            | "0b" ("0"|"1") {("0"|"1")|"_"}
IntSuffix ::= "i8"|"i16"|"i32"|"i64"|"u8"|"u16"|"u32"|"u64"
FloatLit  ::= digit {digit|"_"} "." digit {digit|"_"} [Exp] [FloatSuffix]
            | digit {digit|"_"} Exp [FloatSuffix]
Exp       ::= ("e"|"E") ["+"|"-"] digit+
FloatSuffix ::= "f32" | "f64"
BoolLit   ::= "true" | "false"
CharLit   ::= "'" (escape | any-char) "'"
StrLit    ::= '"' { escape | any-char-except-quote } '"'
RawStrLit ::= 'r"' {any char} '"'            // no escapes
```

Default numeric types: an unsuffixed `FloatLit` defaults to **`f32`** (games want
f32 by default), an unsuffixed `IntLit` defaults to **`i32`**. These defaults are
overridden by bidirectional inference from context (§7.2).

### 2.4 Sigils & operators (the distinctive ones)
```
"~"   owned-heap pointer type / move
"&"   shared reference        "&mut"  mutable reference
"#"   region prefix: #frame #level #perm
"@"   attribute introducer
"->"  return type             "=>" match arm / closure body
".."  range (exclusive)       "..=" range (inclusive)
"|>"  pipe operator (UFCS sugar)
swizzle access via "." on vector types: v.xyz, v.xxzw, v.rgb
```

---

## 3. Top-level structure

### 3.1 Program / module
```
Program     ::= { Item }
Item        ::= Attr* Vis? ( UseDecl | ModDecl | FnDecl | StructDecl
                           | EnumDecl | ComponentDecl | SystemDecl
                           | TraitDecl | ImplDecl | PipelineDecl
                           | ConstDecl | ComptimeBlock )
Vis         ::= "pub"
Attr        ::= "@" ident [ "(" AttrArgs ")" ]
AttrArgs    ::= AttrArg ~ ","
AttrArg     ::= ident "=" Expr | "." ident | Expr
UseDecl     ::= "use" Path [ "as" ident ] ";"
ModDecl     ::= "mod" ident ( "{" {Item} "}" | ";" )
Path        ::= ident { "::" ident }
ConstDecl   ::= "const" ident ":" Type "=" Expr ";"
ComptimeBlock ::= "comptime" Block
```

### 3.2 Functions
```
FnDecl   ::= "fn" ident [ Generics ] "(" [ Params ] ")" [ "->" Type ]
             [ WhereClause ] Block
Params   ::= Param ~ ","
Param    ::= ["mut"] ident ":" Type
           | "self" | "&self" | "&mut self"
Generics ::= "<" GenericParam ~ "," ">"
GenericParam ::= ident [ ":" TraitBound ]
TraitBound   ::= Path { "+" Path }
WhereClause  ::= "where" ( ident ":" TraitBound ) ~ ","
```

### 3.3 Statement termination (ASI rule)
A statement is terminated by `;`. When `;` is absent, the parser inserts one iff:
the current token sequence forms a complete statement/expression **and** the next
token cannot continue it (it is not an infix/postfix operator, `.`, `(`, `[`, or
`{` in expression position). A trailing expression with no `;` in a block is the
block's value (§5.1). This keeps the language brace-delimited and unambiguous
while letting most lines omit `;`.

---

## 4. Aggregate & nominal declarations

### 4.1 Structs (value types by default)
```
StructDecl ::= "struct" ident [ Generics ] "{" [ FieldList ] "}"
             | "struct" ident [ Generics ] "(" [ Type ~ "," ] ")" ";"  // tuple struct
FieldList  ::= Field ~ ","
Field      ::= Attr* Vis? ident ":" Type [ "=" Expr ]      // default value optional
```

Structs are **POD value types**: stack-allocated, copied on assignment unless they
(transitively) contain an `~T` or `rc<T>` field, which makes them *move/own* types
(§8.1). Field defaults enable `Foo { .. }` partial init.

### 4.2 Enums (tagged unions, exhaustive)
```
EnumDecl  ::= "enum" ident [ Generics ] "{" Variant ~ "," "}"
Variant   ::= ident [ "(" Type ~ "," ")" | "{" FieldList "}" ]
            | ident "=" IntLit                              // C-like discriminant
```

### 4.3 Components (ECS storage units)
```
ComponentDecl ::= "component" ident [ Generics ] "{" [ FieldList ] "}"
```
A `component` is structurally a struct but:
- It is registered in the world's component table (gets a stable `ComponentId`).
- Its instances are stored **SoA** (struct-of-arrays) per archetype by default;
  `@aos` overrides to array-of-structs for components always accessed together.
- It may carry replication attributes (`@replicated`, `@interp`, `@predict`) — see
  the netcode spec.
- It may **not** contain `~T` owned heap pointers unless tagged `@boxed`
  (keeps archetype storage trivially relocatable). `rc<T>` and `Handle<T>` are fine.

### 4.4 Systems (scheduled functions over queries)
```
SystemDecl ::= "system" ident "(" [ SysParams ] ")" [ SysSched ] Block
SysParams  ::= SysParam ~ ","
SysParam   ::= ident ":" Type                  // resources: dt: Time, res: &mut Audio, etc.
SysSched   ::= "after" "(" Path ~ "," ")"
             | "before" "(" Path ~ "," ")"
             | "stage" "(" ident ")"            // e.g. stage(Update), stage(Render)
```
The system body references components only through `query<...>` expressions (§5.4).
The scheduler computes each system's **access set** (which components are read vs
written, which resources are touched) from its query and param types, and runs
systems with disjoint write-sets in parallel — statically, no locks (§6.2).

### 4.5 Traits & impls (interfaces / behavior)
```
TraitDecl ::= "trait" ident [ Generics ] [ ":" TraitBound ] "{" { TraitItem } "}"
TraitItem ::= FnSig ";" | FnDecl                // default methods allowed
FnSig     ::= "fn" ident [Generics] "(" [Params] ")" [ "->" Type ]
ImplDecl  ::= "impl" [ Generics ] ( Path "for" )? Type [ WhereClause ]
              "{" { FnDecl | ConstDecl } "}"
```
Dispatch is **static (monomorphized) by default**. Dynamic dispatch is explicit via
`dyn Trait` (a fat pointer), used sparingly because it defeats inlining/SIMD.

### 4.6 Pipelines & shader stages (graphics as language)
```
PipelineDecl ::= "pipeline" ident "{" PipelineField ~ "," "}"
PipelineField ::= "vs"     ":" Path
                | "fs"     ":" Path
                | "cs"     ":" Path            // compute
                | "blend"  ":" BlendExpr
                | "depth"  ":" DepthExpr
                | "cull"   ":" ("." ident)
                | "topology" ":" ("." ident)
```
Shader entry points are normal `FnDecl`s with a stage attribute:
```
@vertex   fn vs(in: VertexIn) -> VertexOut { ... }
@fragment fn fs(in: VertexOut) -> Color    { ... }
@compute(workgroup = (8,8,1)) fn cs(id: GlobalId) { ... }
```
Bindings (`uniform`, `texture`, `buffer`) are **inferred from free variable usage**
inside shader functions and lifted into an auto-generated bind-group layout (§7.6).

---

## 5. Expressions & statements

### 5.1 Blocks (expression-oriented)
```
Block ::= "{" { Stmt } [ Expr ] "}"            // trailing Expr is block value
Stmt  ::= LetStmt | ExprStmt | DeferStmt | ";"
LetStmt   ::= "let" ["mut"] Pattern [ ":" Type ] [ "=" Expr ] ";"
ExprStmt  ::= Expr ";"  |  BlockExpr           // if/match/for/while/loop need no ";"
DeferStmt ::= "defer" Expr ";"                 // runs on scope exit (LIFO)
```

### 5.2 Expression grammar (precedence-climbing)
```
Expr        ::= Assign
Assign      ::= OrExpr [ AssignOp Assign ]
AssignOp    ::= "=" | "+=" | "-=" | "*=" | "/=" | "%=" | "|=" | "&=" | "^="
OrExpr      ::= AndExpr   { "or"  AndExpr }
AndExpr     ::= CmpExpr   { "and" CmpExpr }
CmpExpr     ::= AddExpr   { ("=="|"!="|"<"|">"|"<="|">=") AddExpr }
AddExpr     ::= MulExpr   { ("+"|"-") MulExpr }
MulExpr     ::= UnaryExpr { ("*"|"/"|"%") UnaryExpr }
UnaryExpr   ::= ("-"|"not"|"&"|"&mut"|"~"|"*") UnaryExpr | PipeExpr
PipeExpr    ::= Postfix { "|>" Postfix }        // x |> f(a)  ==  f(x, a)
Postfix     ::= Primary { CallSuffix | IndexSuffix | FieldSuffix | RangeSuffix }
CallSuffix  ::= [ "::" "<" Type ~ "," ">" ] "(" [ Args ] ")"
IndexSuffix ::= "[" Expr "]"
FieldSuffix ::= "." ( ident | IntLit | Swizzle )
RangeSuffix ::= (".."|"..=") Expr
Swizzle     ::= /[xyzw]{1,4}/ | /[rgba]{1,4}/
Args        ::= Arg ~ ","
Arg         ::= [ ident ":" ] Expr              // named args allowed
Primary     ::= Literal | Path | "self" | "Self"
              | "(" Expr ")" | TupleExpr | ArrayExpr | StructLit
              | IfExpr | MatchExpr | ForExpr | WhileExpr | LoopExpr | Block
              | QueryExpr | SpawnExpr | RegionExpr | ClosureExpr | "unsafe" Block
```

### 5.3 Compound expressions
```
StructLit   ::= Path "{" [ FieldInit ~ "," ] [ ".." Expr ] "}"   // ..base spread
FieldInit   ::= ident [ ":" Expr ]                               // shorthand: { x } == { x: x }
ArrayExpr   ::= "[" [ Expr ~ "," ] "]" | "[" Expr ";" Expr "]"   // [v; n] repeat
TupleExpr   ::= "(" Expr "," [ Expr ~ "," ] ")"
ClosureExpr ::= "|" [ Param ~ "," ] "|" ( Expr | Block )
IfExpr      ::= "if" Expr Block { "else" "if" Expr Block } [ "else" Block ]
WhileExpr   ::= "while" Expr Block
LoopExpr    ::= "loop" Block
ForExpr     ::= "for" Pattern "in" Expr Block
MatchExpr   ::= "match" Expr "{" MatchArm ~ "," "}"
MatchArm    ::= Pattern [ "if" Expr ] "=>" ( Expr | Block )
SpawnExpr   ::= "spawn" "(" Args ")"            // spawn(CompA{..}, CompB{..}) -> Entity
RegionExpr  ::= "#" ("frame"|"level"|"perm") Expr   // allocate Expr in region
```

### 5.4 Query expressions (the ECS access primitive)
```
QueryExpr ::= "query" "<" QTerm ~ "," ">" [ "where" Expr ]
QTerm     ::= "&" Path             // read component
            | "&mut" Path          // write component
            | "Entity"             // the entity id
            | "?" "&"  Path        // optional component (Option<&T>)
            | "?" "&mut" Path
            | "!" Path             // filter: NOT having component
            | "+" Path             // filter: must have (no data binding)
```
A `query<...>` is iterable (yields tuples in declaration order, minus filter-only
terms). The set of `Path`s with `&mut` is the system's **write set**; with `&` the
**read set**. These drive scheduling (§6.2) and are checked for self-conflict
(a single query may not contain both `&mut T` and `&T`/`&mut T` for the same `T`).

### 5.5 Patterns
```
Pattern   ::= "_" | Literal | ident [ "@" Pattern ]
            | Path [ "(" Pattern ~ "," ")" | "{" FieldPat ~ "," "}" ]
            | "(" Pattern ~ "," ")"
            | ".." 
FieldPat  ::= ident [ ":" Pattern ]
```

---

## 6. Static semantics — ECS

### 6.1 Component & system registration
Each `component` declaration introduces a distinct nominal type and a globally
unique `ComponentId` assigned at link time. `system` declarations are collected
into a scheduler graph at compile time.

### 6.2 Scheduling & data-race freedom (the core safety theorem)
For systems `S1`, `S2` in the same stage with access sets
`(R1, W1)` and `(R2, W2)`:

- They **may run in parallel** iff `W1 ∩ (R2 ∪ W2) = ∅` and `W2 ∩ (R1 ∪ W1) = ∅`.
- Otherwise an edge `S1 → S2` is added per declared `after`/`before`, or, absent an
  ordering, the compiler **emits an error** demanding the programmer disambiguate.

Because access sets are derived from `query<...>` types, the compiler proves the
absence of data races *without* a borrow checker over component storage. This is
the "safe where it's free" payoff: parallelism is the default and it's sound.

### 6.3 Structural commands
- `spawn(c1, c2, ...)` → `Entity`. Component types must be distinct.
- `despawn(e)` schedules removal (applied at command-buffer flush, end of stage).
- Structural changes inside a system are buffered, not immediate, to keep iteration
  invalidation impossible during a query loop.

---

## 7. Static semantics — types & inference

### 7.1 Type syntax
```
Type      ::= Primitive | Path [ "<" Type ~ "," ">" ]
            | "~" Type                          // owned heap
            | "rc" "<" Type ">"                 // refcounted heap
            | "&" Type | "&mut" Type            // borrows
            | "[" Type [ ";" Expr ] "]"         // slice / fixed array
            | "(" Type ~ "," ")"                // tuple
            | "dyn" Path                         // trait object
            | "fn" "(" [Type ~ ","] ")" "->" Type
            | "Option" "<" Type ">" | "Result" "<" Type "," Type ">"
            | "Handle" "<" Type ">"              // opaque asset handle (copyable)
```

### 7.2 Inference algorithm
Aurora uses **local type inference with bidirectional checking**:
- `let x = e;` infers `x`'s type from `e` (synthesis mode).
- `let x: T = e;` checks `e` against `T` (checking mode), which propagates `T`
  inward — e.g. numeric literal default selection, struct-literal field types,
  empty-collection element types.
- Function signatures are **always fully annotated** (no whole-program inference);
  this keeps inference fast and errors local — essential for hot-reload latency.
- Generic calls infer type args from argument types; explicit `::<T>` overrides.

### 7.3 Value vs. owning vs. borrowing types
A type `T` is classified:
- **Copy** if it is a primitive, a `Handle<_>`, a `&_`, or an aggregate whose
  fields are all Copy. Copy types are duplicated on assignment/pass.
- **Move** if it (transitively) contains `~_` or `rc<_>` (note: `rc` clones the
  refcount rather than the payload — it is Move with a cheap clone via `.share()`).
- **Region-bound** if produced by a `RegionExpr`; carries an implicit region tag
  (§8.2) and may not escape its region.

### 7.4 Coercions
- `i32 → i64`, `f32 → f64` widening: **explicit only** (`as`), no implicit numeric
  coercion (prevents silent precision bugs in physics/netcode).
- `&mut T → &T` reborrow: implicit.
- `~T → &T` / `~T → &mut T`: implicit borrow of owned data.
- `T → Option<T>` via `Some`: not implicit; `none`/`some(x)` constructors.

### 7.5 Vector/matrix algebra typing
`VecN`/`MatN`/`Quat`/`Color` are builtin and overloaded:
- `VecN + VecN → VecN`, `VecN * f32 → VecN` (and `f32 * VecN`), componentwise `*`.
- `MatN * VecN → VecN`, `MatN * MatN → MatN`, `Quat * Quat → Quat`,
  `Quat * Vec3 → Vec3` (rotate).
- **Swizzles** type as: reading `n` lanes yields `VecN` (`n∈1..4`, where `Vec1==f32`);
  swizzle *assignment* (`v.xy = ...`) requires no repeated lane.
- These ops are intrinsics that lower to SIMD on CPU and native vector ops on GPU.

### 7.6 GPU sub-environment (shader typing)
A function reachable from a `@vertex`/`@fragment`/`@compute` entry is checked in
**GPU mode**, where:
- Allowed types: primitives, `VecN`/`MatN`, fixed arrays, structs of allowed types,
  `texture`, `sampler`, `buffer<T>`. **Disallowed:** `~T`, `rc<T>`, `str`, `dyn`,
  recursion, function pointers, heap allocation.
- Free variables of `texture`/`sampler`/`buffer`/`uniform`-typed bindings are
  collected to synthesize the bind-group layout; a `uniform` is any module-level
  `let` referenced from a shader fn.
- Control flow is unrestricted (the backend handles uniform vs. non-uniform), but
  the type checker forbids host-only intrinsics (file IO, spawn, query) — calling
  one from GPU code is a compile error with a "not available on GPU" diagnostic.

### 7.7 `comptime`
A `comptime` block/expression is evaluated by the compiler in a sandboxed
interpreter; its result is spl-iced in as a constant. Used for baking lookup
tables, validating asset paths, and generating specialized code. `comptime`
values must be Copy and may not reference runtime resources.

---

## 8. Static semantics — memory, regions & borrows

### 8.1 Ownership
- `~T` is a uniquely-owned heap box. Assigning/passing it **moves** it; the source
  binding is invalidated (use-after-move is a compile error). Dropped automatically
  at end of owning scope (deterministic destruction; `Drop` trait runs).
- `rc<T>` is reference-counted; `.share()` increments, drop decrements, payload
  freed at zero. No cycle collector — cycles are a (documented) leak; break them
  with `weak<T>`.

### 8.2 Regions (the GC replacement)
Three built-in region lifetimes, ordered by outlivedness:
```
'perm  ⊐  'level  ⊐  'frame
```
- `#frame e` allocates in the per-frame bump arena; reset to zero cost each frame.
- `#level e` allocates in the level arena; freed on level unload.
- `#perm e` allocates in a permanent arena (effectively static).

**Region rule (escape analysis):** a value allocated in region `R` may be stored
into a location of region `R'` only if `R' ⊑ R` (the destination does not outlive
the source). Violations are compile errors, e.g. storing a `#frame` pointer into a
`#perm` field. Region tags are inferred and flow through assignments/returns; the
programmer annotates only at function boundaries when a function is region-generic:
```
fn make_label<'r>(arena: #'r, text: str) -> &'r Label { ... }
```
Most code never writes `'r` — gameplay data is `#frame` or component storage, both
of which the checker handles implicitly.

### 8.3 Borrow rules (scoped to owned/region memory only)
For `&T` / `&mut T` into `~T` or region memory, the classic discipline applies:
- Any number of `&T`, **or** exactly one `&mut T`, never both, over an overlapping
  lifetime.
- Borrows may not outlive their referent's region (enforced by §8.2).

Crucially, these rules **do not apply** to Copy value types (you pass copies) nor
to component access (governed by the scheduler's access-set disjointness, §6.2).
The result: the borrow checker is invisible in ~90% of gameplay code and only
appears when you deliberately reach for `~`/`rc`/raw references.

### 8.4 `unsafe`
`unsafe { ... }` permits: raw pointer deref, `extern` FFI calls, transmute,
manual region/lifetime assertions. The compiler trusts the programmer inside;
everything outside remains checked. C ABI: `extern "C" fn`.

---

## 9. Worked example (parses & type-checks under the above)

```aurora
use engine::{App, Camera, Transform, Mesh, load}

component Spinner { speed: f32 = 1.0 }
component Velocity { v: Vec3 }

system spin(dt: Time) stage(Update) {
    for (t, s) in query<&mut Transform, &Spinner> {
        t.rotate(Quat.y(s.speed * dt))      // f32 * Time -> f32 (Time coerces)
    }
}

system integrate(dt: Time) stage(Update) after(spin) {
    for (t, vel) in query<&mut Transform, &Velocity> {
        t.pos += vel.v * dt
    }
}

@vertex fn vs(in: VertexIn) -> VertexOut {
    VertexOut { clip: camera.view_proj * vec4(in.pos, 1.0), uv: in.uv }
}
@fragment fn fs(in: VertexOut) -> Color {
    texture(albedo, in.uv)                  // albedo: texture -> inferred binding
}

pipeline standard { vs, fs, depth: .less, cull: .back }

fn main() {
    let app = App.new("demo", 1280, 720)
    let cube: Handle<Mesh> = load("cube.glb")
    app.spawn(Transform.identity(), cube, Spinner { speed: 1.5 })
    app.spawn(Camera.perspective(60.0), Transform.at(0.0, 2.0, 5.0))
    app.run()
}
```

Notes the checker verifies here:
- `spin` (writes Transform) and `integrate` (writes Transform) **conflict** →
  `after(spin)` resolves the ordering; without it, a scheduling error is raised.
- `vs`/`fs` type-check in GPU mode; `camera`/`albedo` lift to bindings of pipeline
  `standard`.
- `cube` is a `Handle<Mesh>` (Copy), freely passed to `spawn`.

---

## 10. Open questions (resolve before freezing the grammar)

1. **Generics over const values** (`[T; N]` with `N` generic) — needed for
   fixed-size GPU arrays. Likely adopt const generics with `const N: u32`.
2. **Effect tracking for `defer`/`Drop` in GPU mode** — currently just banned.
3. **Macro/metaprogramming surface** beyond `comptime` — keep minimal; prefer
   `comptime` + derive-style attributes (`@derive(Serialize)`).
4. **String type story** — `str` is a `#`-region or `~`-owned UTF-8 slice; no GC'd
   string. Confirm interning strategy for asset paths.
