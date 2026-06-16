//! Recursive-descent + Pratt parser for Aurora (grammar spec §3–§8).
//!
//! Items and statements are parsed by recursive descent; expressions use
//! precedence climbing (see `expr.rs`). The parser never panics on bad input:
//! it emits a diagnostic, inserts an `Error` node, and synchronizes to a
//! recovery point so a single mistake does not cascade.

mod expr;
mod flatten;
mod item;
mod pat;
mod stmt;
mod ty;

use aurora_ast::Module;
use aurora_diag::Diagnostic;
use aurora_lexer::{lex, Keyword, Token, TokenKind};
use aurora_span::Span;

pub use aurora_ast as ast;

/// Lex and parse `src` into a [`Module`] plus all diagnostics from both phases.
pub fn parse_str(src: &str) -> (Module, Vec<Diagnostic>) {
    let lexed = lex(src);
    let mut parser = Parser::new(lexed.tokens);
    let mut module = parser.parse_module();
    // Lower `mod` blocks to mangled top-level items so modules namespace cleanly.
    module.items = flatten::flatten_modules(module.items);
    let mut diags = lexed.diagnostics;
    diags.append(&mut parser.diags);
    (module, diags)
}

/// Parse an already-lexed token stream.
pub fn parse(tokens: Vec<Token>) -> (Module, Vec<Diagnostic>) {
    let mut parser = Parser::new(tokens);
    let module = parser.parse_module();
    (module, parser.diags)
}

/// Max expression/type nesting before the parser gives up. Bounds recursion so
/// pathological/generated input produces a diagnostic instead of a stack-overflow
/// abort (which is uncatchable and crashes the whole compiler). Kept well under
/// what would exhaust the stack: one expression nesting level is ~7 deep parser
/// frames, and the resulting AST is walked recursively by later passes too, so a
/// modest cap protects the whole pipeline. Real code never nests this deep.
pub(crate) const MAX_PARSE_DEPTH: u32 = 100;

pub(crate) struct Parser {
    tokens: Vec<Token>,
    pos: usize,
    pub(crate) diags: Vec<Diagnostic>,
    /// When set, a `{` following a path does not begin a struct literal (used
    /// while parsing `if`/`while`/`for`/`match` heads). Grammar spec §5.3 note.
    restrict_struct: bool,
    /// Current expression/type recursion depth (see `MAX_PARSE_DEPTH`).
    depth: u32,
}

impl Parser {
    pub(crate) fn new(mut tokens: Vec<Token>) -> Parser {
        // Guarantee a trailing Eof so cursor reads never go out of bounds.
        if !matches!(tokens.last().map(|t| &t.kind), Some(TokenKind::Eof)) {
            let at = tokens.last().map(|t| t.span.hi).unwrap_or(0);
            tokens.push(Token::new(TokenKind::Eof, Span::new(at, at)));
        }
        Parser { tokens, pos: 0, diags: Vec::new(), restrict_struct: false, depth: 0 }
    }

    // --- cursor --------------------------------------------------------------

    fn cur(&self) -> &Token {
        // pos is always in range because the stream ends in Eof and we never
        // advance past it.
        &self.tokens[self.pos.min(self.tokens.len() - 1)]
    }

    fn kind(&self) -> &TokenKind {
        &self.cur().kind
    }

    fn nth_kind(&self, n: usize) -> &TokenKind {
        let i = (self.pos + n).min(self.tokens.len() - 1);
        &self.tokens[i].kind
    }

    fn prev(&self) -> &Token {
        &self.tokens[self.pos.saturating_sub(1)]
    }

    fn cur_span(&self) -> Span {
        self.cur().span
    }

    /// Span from a recorded start through the previously consumed token.
    fn finish(&self, start: Span) -> Span {
        start.to(self.prev().span)
    }

    fn is_eof(&self) -> bool {
        matches!(self.kind(), TokenKind::Eof)
    }

    fn bump(&mut self) -> Token {
        let tok = self.cur().clone();
        if !self.is_eof() {
            self.pos += 1;
        }
        tok
    }

    // --- predicates ----------------------------------------------------------

    /// True if the current token is the given dataless punctuation/operator kind.
    fn at(&self, k: &TokenKind) -> bool {
        std::mem::discriminant(self.kind()) == std::mem::discriminant(k)
    }

    fn at_kw(&self, kw: Keyword) -> bool {
        matches!(self.kind(), TokenKind::Kw(k) if *k == kw)
    }

    fn at_ident(&self) -> bool {
        matches!(self.kind(), TokenKind::Ident(_))
    }

    /// True if the current token is the contextual identifier `name`.
    fn at_ctx(&self, name: &str) -> bool {
        matches!(self.kind(), TokenKind::Ident(s) if s == name)
    }

    fn eat(&mut self, k: &TokenKind) -> bool {
        if self.at(k) {
            self.bump();
            true
        } else {
            false
        }
    }

    fn eat_kw(&mut self, kw: Keyword) -> bool {
        if self.at_kw(kw) {
            self.bump();
            true
        } else {
            false
        }
    }

    /// Consume the contextual identifier `name` if present.
    fn eat_ctx(&mut self, name: &str) -> bool {
        if self.at_ctx(name) {
            self.bump();
            true
        } else {
            false
        }
    }

    // --- consuming with error reporting --------------------------------------

    /// Expect a dataless token; on mismatch emit an error and do not advance.
    fn expect(&mut self, k: &TokenKind) -> bool {
        if self.eat(k) {
            true
        } else {
            self.err_expected(&k.describe());
            false
        }
    }

    fn ident(&mut self) -> Ident {
        match self.kind().clone() {
            TokenKind::Ident(name) => {
                let span = self.cur_span();
                self.bump();
                Ident { name, span }
            }
            // Allow the contextual `self`/`Self` keywords to be requested by name
            // only where callers explicitly handle them; here we just error.
            _ => {
                self.err_expected("an identifier");
                let span = self.cur_span();
                Ident { name: "<error>".into(), span }
            }
        }
    }

    // --- diagnostics & recovery ----------------------------------------------

    fn error(&mut self, span: Span, msg: impl Into<String>, label: impl Into<String>) {
        self.diags
            .push(Diagnostic::error(msg).with_code("E0100").primary(span, label));
    }

    fn err_expected(&mut self, what: &str) {
        let found = self.kind().describe();
        let span = self.cur_span();
        self.error(span, format!("expected {what}, found {found}"), format!("expected {what}"));
    }

    /// Skip tokens until a likely synchronization point: a top-level item
    /// keyword, a closing brace, or EOF. Used after a failed item/stmt.
    fn recover_to_item(&mut self) {
        while !self.is_eof() {
            if self.at(&TokenKind::RBrace) || self.starts_item() {
                break;
            }
            // A `;` ends the broken construct; consume it and stop.
            if self.eat(&TokenKind::Semi) {
                break;
            }
            self.bump();
        }
    }

    fn starts_item(&self) -> bool {
        use Keyword::*;
        matches!(
            self.kind(),
            TokenKind::Kw(Fn | Struct | Enum | Component | System | Trait | Impl | Pipeline
                | Use | Mod | Pub | Comptime)
        ) || matches!(self.kind(), TokenKind::At)
    }

    // --- entry ---------------------------------------------------------------

    fn parse_module(&mut self) -> Module {
        let mut items = Vec::new();
        while !self.is_eof() {
            let before = self.pos;
            if let Some(item) = self.parse_item() {
                items.push(item);
            }
            // Guarantee forward progress even if a sub-parser consumed nothing.
            if self.pos == before {
                self.bump();
            }
        }
        Module { items }
    }
}

// Re-export AST leaf used pervasively in parser modules.
use aurora_ast::Ident;
