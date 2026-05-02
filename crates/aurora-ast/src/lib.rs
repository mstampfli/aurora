//! Abstract syntax tree for Aurora (grammar spec §3–§8).
//!
//! Nodes carry a [`Span`] for diagnostics. The tree is intentionally close to
//! the surface syntax; desugaring (pipes, struct-update, etc.) happens during
//! lowering to HIR, not here, so error messages can point at real source.

use aurora_lexer::{FloatTy, IntTy};
use aurora_span::Span;

mod monomorphize;
pub use monomorphize::monomorphize;
mod schedule;
pub use schedule::parallel_layers;

// ---------------------------------------------------------------------------
// Shared leaves
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq)]
pub struct Ident {
    pub name: String,
    pub span: Span,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Vis {
    Private,
    Pub,
}

/// A `::`-separated path; each segment may carry generic arguments
/// (`engine::load`, `rc<Texture>`, `Handle<Mesh>`).
#[derive(Clone, Debug, PartialEq)]
pub struct Path {
    pub segments: Vec<PathSeg>,
    pub span: Span,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PathSeg {
    pub ident: Ident,
    pub args: Vec<Type>,
}

impl Path {
    pub fn is_single(&self) -> bool {
        self.segments.len() == 1 && self.segments[0].args.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Attributes
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq)]
pub struct Attr {
    pub name: Ident,
    pub args: Vec<AttrArg>,
    pub span: Span,
}

#[derive(Clone, Debug, PartialEq)]
pub enum AttrArg {
    /// `key = expr`
    Named(Ident, Expr),
    /// a bare expression, e.g. `0.01` in `@quantize(0.01)`
    Positional(Expr),
}

// ---------------------------------------------------------------------------
// Module & items
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq)]
pub struct Module {
    pub items: Vec<Item>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Item {
    pub attrs: Vec<Attr>,
    pub vis: Vis,
    pub kind: ItemKind,
    pub span: Span,
}

#[derive(Clone, Debug, PartialEq)]
pub enum ItemKind {
    Use(UseDecl),
    Mod(Ident, Option<Vec<Item>>),
    Fn(FnDecl),
    Struct(StructDecl),
    Enum(EnumDecl),
    Component(StructDecl),
    System(SystemDecl),
    Trait(TraitDecl),
    Impl(ImplDecl),
    Pipeline(PipelineDecl),
    Const(ConstDecl),
    Comptime(Block),
    /// Recovery placeholder for an item that failed to parse.
    Error,
}

#[derive(Clone, Debug, PartialEq)]
pub struct UseDecl {
    pub path: Path,
    pub kind: UseKind,
}

#[derive(Clone, Debug, PartialEq)]
pub enum UseKind {
    /// `use a::b;` or `use a::b as c;`
    Single(Option<Ident>),
    /// `use a::{x, y, z}`
    Group(Vec<Ident>),
}

#[derive(Clone, Debug, PartialEq)]
pub struct FnDecl {
    pub name: Ident,
    pub generics: Vec<GenericParam>,
    pub params: Vec<Param>,
    pub ret: Option<Type>,
    pub where_clause: Vec<WherePred>,
    /// `None` for a trait method signature with no default body.
    pub body: Option<Block>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Param {
    /// `self`, `&self`, `&mut self`
    SelfParam { by_ref: bool, mutable: bool },
    Normal { mutable: bool, name: Ident, ty: Type },
}

#[derive(Clone, Debug, PartialEq)]
pub struct GenericParam {
    pub name: Ident,
    pub bounds: Vec<Path>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct WherePred {
    pub name: Ident,
    pub bounds: Vec<Path>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct StructDecl {
    pub name: Ident,
    pub generics: Vec<GenericParam>,
    pub body: StructBody,
}

#[derive(Clone, Debug, PartialEq)]
pub enum StructBody {
    Named(Vec<Field>),
    Tuple(Vec<Type>),
    Unit,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Field {
    pub attrs: Vec<Attr>,
    pub vis: Vis,
    pub name: Ident,
    pub ty: Type,
    pub default: Option<Expr>,
    pub span: Span,
}

#[derive(Clone, Debug, PartialEq)]
pub struct EnumDecl {
    pub name: Ident,
    pub generics: Vec<GenericParam>,
    pub variants: Vec<Variant>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Variant {
    pub name: Ident,
    pub data: VariantData,
    pub discriminant: Option<Expr>,
    pub span: Span,
}

#[derive(Clone, Debug, PartialEq)]
pub enum VariantData {
    Unit,
    Tuple(Vec<Type>),
    Struct(Vec<Field>),
}

#[derive(Clone, Debug, PartialEq)]
pub struct SystemDecl {
    pub name: Ident,
    pub params: Vec<SysParam>,
    pub schedule: Vec<SysSched>,
    pub body: Block,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SysParam {
    pub name: Ident,
    pub ty: Type,
}

#[derive(Clone, Debug, PartialEq)]
pub enum SysSched {
    After(Vec<Path>),
    Before(Vec<Path>),
    Stage(Ident),
}

#[derive(Clone, Debug, PartialEq)]
pub struct TraitDecl {
    pub name: Ident,
    pub generics: Vec<GenericParam>,
    pub supertraits: Vec<Path>,
    pub items: Vec<AssocItem>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ImplDecl {
    pub generics: Vec<GenericParam>,
    pub trait_: Option<Path>,
    pub self_ty: Type,
    pub where_clause: Vec<WherePred>,
    pub items: Vec<AssocItem>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum AssocItem {
    Fn(FnDecl),
    Const(ConstDecl),
}

#[derive(Clone, Debug, PartialEq)]
pub struct PipelineDecl {
    pub name: Ident,
    pub fields: Vec<PipelineField>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PipelineField {
    pub key: Ident,
    pub value: Expr,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ConstDecl {
    pub name: Ident,
    pub ty: Option<Type>,
    pub value: Expr,
}

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq)]
pub struct Type {
    pub kind: TypeKind,
    pub span: Span,
}

#[derive(Clone, Debug, PartialEq)]
pub enum TypeKind {
    /// `Vec3`, `engine::Mesh`, `rc<Texture>`, `Option<T>`
    Path(Path),
    /// `~T`
    Owned(Box<Type>),
    /// `&T` / `&mut T`
    Ref { mutable: bool, inner: Box<Type> },
    /// `[T]` (slice) or `[T; N]` (array)
    Array { elem: Box<Type>, len: Option<Box<Expr>> },
    Tuple(Vec<Type>),
    /// `dyn Trait`
    Dyn(Path),
    /// `fn(A, B) -> C`
    Fn { params: Vec<Type>, ret: Box<Type> },
    /// A region-annotated type — `#frame T` / `#level T` / `#perm T` — used on
    /// function parameters and returns to declare a region contract at a
    /// boundary (trait signatures, `@extern`), where the body can't be inferred.
    /// The region is checking-only; the value's representation is `inner`'s.
    Region(RegionKind, Box<Type>),
    /// `_`
    Infer,
    Error,
}

// ---------------------------------------------------------------------------
// Statements & blocks
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq)]
pub struct Block {
    pub stmts: Vec<Stmt>,
    /// Trailing expression with no `;` — the block's value.
    pub tail: Option<Box<Expr>>,
    pub span: Span,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Stmt {
    Let(LetStmt),
    Defer(Expr),
    /// An expression used as a statement (terminated by `;` or a block form).
    Expr(Expr),
}

#[derive(Clone, Debug, PartialEq)]
pub struct LetStmt {
    pub mutable: bool,
    pub pat: Pat,
    pub ty: Option<Type>,
    pub init: Option<Expr>,
}

// ---------------------------------------------------------------------------
// Expressions
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq)]
pub struct Expr {
    pub kind: ExprKind,
    pub span: Span,
}

#[derive(Clone, Debug, PartialEq)]
pub enum ExprKind {
    Int(u128, Option<IntTy>),
    Float(f64, Option<FloatTy>),
    Str(String),
    Char(char),
    Bool(bool),
    /// A name path, possibly with turbofish args on segments.
    Path(Path),
    /// `self`
    SelfExpr,
    /// `.variant` shorthand (enum/option/pipeline-field sugar, e.g. `.less`)
    Dot(Ident),

    Unary(UnOp, Box<Expr>),
    Binary(BinOp, Box<Expr>, Box<Expr>),
    /// `lhs = rhs` (op = None) or compound `lhs += rhs` (op = Some).
    Assign(Option<BinOp>, Box<Expr>, Box<Expr>),
    Cast(Box<Expr>, Type),

    Call { callee: Box<Expr>, type_args: Vec<Type>, args: Vec<Arg> },
    Index { base: Box<Expr>, index: Box<Expr> },
    Field { base: Box<Expr>, field: FieldAccess },

    Range { start: Option<Box<Expr>>, end: Option<Box<Expr>>, inclusive: bool },
    /// `x |> f(a)`  (kept explicit; desugared during lowering)
    Pipe { value: Box<Expr>, func: Box<Expr> },

    Struct { path: Path, fields: Vec<FieldInit>, base: Option<Box<Expr>> },
    Array(Vec<Expr>),
    ArrayRepeat { value: Box<Expr>, count: Box<Expr> },
    Tuple(Vec<Expr>),
    Paren(Box<Expr>),

    If(IfExpr),
    Match { scrutinee: Box<Expr>, arms: Vec<MatchArm> },
    For { pat: Pat, iter: Box<Expr>, body: Block },
    While { cond: Box<Expr>, body: Block },
    Loop(Block),
    Block(Block),
    Unsafe(Block),
    Closure { params: Vec<Param>, body: Box<Expr> },

    Query(QueryExpr),
    Spawn(Vec<Arg>),
    Despawn(Box<Expr>),
    /// `#frame e`, `#level e`, `#perm e`
    Region { region: RegionKind, value: Box<Expr> },

    Return(Option<Box<Expr>>),
    Break(Option<Box<Expr>>),
    Continue,
    /// `expr?` — error propagation: yields the success payload, or early-returns
    /// the error variant from the enclosing function.
    Try(Box<Expr>),

    Error,
}

#[derive(Clone, Debug, PartialEq)]
pub enum FieldAccess {
    Named(Ident),
    /// `.0`, `.1` tuple access
    Index(u32),
}

#[derive(Clone, Debug, PartialEq)]
pub struct Arg {
    pub name: Option<Ident>,
    pub value: Expr,
}

#[derive(Clone, Debug, PartialEq)]
pub struct FieldInit {
    pub name: Ident,
    /// `None` => shorthand `{ x }` meaning `{ x: x }`.
    pub value: Option<Expr>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct IfExpr {
    pub cond: Box<Expr>,
    pub then_branch: Block,
    pub else_branch: Option<Box<Expr>>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MatchArm {
    pub pat: Pat,
    pub guard: Option<Expr>,
    pub body: Expr,
}

#[derive(Clone, Debug, PartialEq)]
pub struct QueryExpr {
    pub terms: Vec<QTerm>,
    pub filter: Option<Box<Expr>>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum QTerm {
    Read(Path),
    Write(Path),
    Entity,
    OptRead(Path),
    OptWrite(Path),
    /// `!T` — filter: must NOT have
    Without(Path),
    /// `+T` — filter: must have (no binding)
    With(Path),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RegionKind {
    Frame,
    Level,
    Perm,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UnOp {
    Neg,
    Not,
    Deref,
    RefShared,
    RefMut,
    Own,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Rem,
    Eq,
    Ne,
    Lt,
    Gt,
    Le,
    Ge,
    And,
    Or,
}

// ---------------------------------------------------------------------------
// Patterns
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq)]
pub struct Pat {
    pub kind: PatKind,
    pub span: Span,
}

#[derive(Clone, Debug, PartialEq)]
pub enum PatKind {
    Wild,
    Lit(Box<Expr>),
    /// `name` or `name @ subpat`
    Binding { name: Ident, sub: Option<Box<Pat>> },
    /// A path to a unit variant / constant.
    Path(Path),
    TupleStruct { path: Path, elems: Vec<Pat> },
    Struct { path: Path, fields: Vec<FieldPat>, rest: bool },
    Tuple(Vec<Pat>),
    /// `..`
    Rest,
    Error,
}

#[derive(Clone, Debug, PartialEq)]
pub struct FieldPat {
    pub name: Ident,
    /// `None` => shorthand `{ x }`.
    pub pat: Option<Pat>,
}
