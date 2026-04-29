//! Token kinds for Aurora, per grammar spec §2.

use aurora_span::Span;

#[derive(Clone, Debug, PartialEq)]
pub struct Token {
    pub kind: TokenKind,
    pub span: Span,
    /// True if at least one newline appeared in the trivia before this token.
    /// Drives newline-aware statement termination (ASI) in the parser.
    pub nl_before: bool,
}

impl Token {
    pub fn new(kind: TokenKind, span: Span) -> Token {
        Token { kind, span, nl_before: false }
    }
}

/// Integer literal suffixes (`1i32`, `4u8`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IntTy {
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
}

/// Float literal suffixes (`1.0f32`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FloatTy {
    F32,
    F64,
}

/// Reserved keywords (grammar spec §2.2).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Keyword {
    Let,
    Mut,
    Fn,
    Return,
    If,
    Else,
    Match,
    For,
    While,
    Loop,
    Break,
    Continue,
    Struct,
    Enum,
    Component,
    System,
    Trait,
    Impl,
    Pipeline,
    Use,
    Mod,
    Pub,
    True,
    False,
    Comptime,
    Const,
    Unsafe,
    /// the value `self`
    LowerSelf,
    /// the type `Self`
    UpperSelf,
    As,
    Where,
    Defer,
    Query,
    And,
    Or,
    Not,
}

impl Keyword {
    pub fn from_str(s: &str) -> Option<Keyword> {
        use Keyword::*;
        Some(match s {
            "let" => Let,
            "mut" => Mut,
            "fn" => Fn,
            "return" => Return,
            "if" => If,
            "else" => Else,
            "match" => Match,
            "for" => For,
            "while" => While,
            "loop" => Loop,
            "break" => Break,
            "continue" => Continue,
            "struct" => Struct,
            "enum" => Enum,
            "component" => Component,
            "system" => System,
            "trait" => Trait,
            "impl" => Impl,
            "pipeline" => Pipeline,
            "use" => Use,
            "mod" => Mod,
            "pub" => Pub,
            "true" => True,
            "false" => False,
            "comptime" => Comptime,
            "const" => Const,
            "unsafe" => Unsafe,
            "self" => LowerSelf,
            "Self" => UpperSelf,
            "as" => As,
            "where" => Where,
            "defer" => Defer,
            "query" => Query,
            "and" => And,
            "or" => Or,
            "not" => Not,
            _ => return None,
        })
    }

    pub fn as_str(self) -> &'static str {
        use Keyword::*;
        match self {
            Let => "let",
            Mut => "mut",
            Fn => "fn",
            Return => "return",
            If => "if",
            Else => "else",
            Match => "match",
            For => "for",
            While => "while",
            Loop => "loop",
            Break => "break",
            Continue => "continue",
            Struct => "struct",
            Enum => "enum",
            Component => "component",
            System => "system",
            Trait => "trait",
            Impl => "impl",
            Pipeline => "pipeline",
            Use => "use",
            Mod => "mod",
            Pub => "pub",
            True => "true",
            False => "false",
            Comptime => "comptime",
            Const => "const",
            Unsafe => "unsafe",
            LowerSelf => "self",
            UpperSelf => "Self",
            As => "as",
            Where => "where",
            Defer => "defer",
            Query => "query",
            And => "and",
            Or => "or",
            Not => "not",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum TokenKind {
    // Literals
    Int { value: u128, suffix: Option<IntTy> },
    Float { value: f64, suffix: Option<FloatTy> },
    Str(String),
    Char(char),

    // Names
    Ident(String),
    Kw(Keyword),

    // Delimiters
    LParen,
    RParen,
    LBrace,
    RBrace,
    LBracket,
    RBracket,

    // Punctuation
    Comma,
    Semi,
    Colon,
    ColonColon,
    Dot,
    DotDot,
    DotDotEq,
    Arrow,   // ->
    FatArrow, // =>
    At,       // @
    Hash,     // #
    Tilde,    // ~
    Question, // ?

    // Operators
    Amp,     // &
    Pipe,    // |
    Caret,   // ^
    Bang,    // !
    Plus,
    Minus,
    Star,
    Slash,
    Percent,
    Eq,
    EqEq,
    BangEq,
    Lt,
    Gt,
    Le,
    Ge,
    PlusEq,
    MinusEq,
    StarEq,
    SlashEq,
    PercentEq,
    AmpEq,
    PipeEq,
    CaretEq,
    PipeGt, // |>

    /// End of input.
    Eof,
    /// A lexing error already reported via a diagnostic; lets the parser keep going.
    Error,
}

impl TokenKind {
    /// A short human-facing name used in parser error messages.
    pub fn describe(&self) -> String {
        use TokenKind::*;
        match self {
            Int { .. } => "integer literal".into(),
            Float { .. } => "float literal".into(),
            Str(_) => "string literal".into(),
            Char(_) => "character literal".into(),
            Ident(name) => format!("identifier `{name}`"),
            Kw(kw) => format!("keyword `{}`", kw.as_str()),
            Eof => "end of file".into(),
            Error => "invalid token".into(),
            other => format!("`{}`", other.symbol()),
        }
    }

    /// The literal symbol for punctuation/operator tokens (`""` otherwise).
    pub fn symbol(&self) -> &'static str {
        use TokenKind::*;
        match self {
            LParen => "(",
            RParen => ")",
            LBrace => "{",
            RBrace => "}",
            LBracket => "[",
            RBracket => "]",
            Comma => ",",
            Semi => ";",
            Colon => ":",
            ColonColon => "::",
            Dot => ".",
            DotDot => "..",
            DotDotEq => "..=",
            Arrow => "->",
            FatArrow => "=>",
            At => "@",
            Hash => "#",
            Tilde => "~",
            Question => "?",
            Amp => "&",
            Pipe => "|",
            Caret => "^",
            Bang => "!",
            Plus => "+",
            Minus => "-",
            Star => "*",
            Slash => "/",
            Percent => "%",
            Eq => "=",
            EqEq => "==",
            BangEq => "!=",
            Lt => "<",
            Gt => ">",
            Le => "<=",
            Ge => ">=",
            PlusEq => "+=",
            MinusEq => "-=",
            StarEq => "*=",
            SlashEq => "/=",
            PercentEq => "%=",
            AmpEq => "&=",
            PipeEq => "|=",
            CaretEq => "^=",
            PipeGt => "|>",
            _ => "",
        }
    }
}
