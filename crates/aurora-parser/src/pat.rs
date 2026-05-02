//! Pattern parsing (grammar spec §5.5).

use aurora_ast::{Expr, ExprKind, FieldPat, Pat, PatKind, UnOp};
use aurora_lexer::{Keyword, TokenKind};

use crate::Parser;

impl Parser {
    pub(crate) fn parse_pattern(&mut self) -> Pat {
        let start = self.cur_span();
        let kind = match self.kind().clone() {
            TokenKind::Ident(s) if s == "_" => {
                self.bump();
                PatKind::Wild
            }
            TokenKind::DotDot => {
                self.bump();
                PatKind::Rest
            }
            TokenKind::LParen => {
                self.bump();
                let mut elems = Vec::new();
                while !self.at(&TokenKind::RParen) && !self.is_eof() {
                    elems.push(self.parse_pattern());
                    if !self.eat(&TokenKind::Comma) {
                        break;
                    }
                }
                self.expect(&TokenKind::RParen);
                PatKind::Tuple(elems)
            }
            // Literal patterns.
            TokenKind::Int { .. }
            | TokenKind::Float { .. }
            | TokenKind::Str(_)
            | TokenKind::Char(_)
            | TokenKind::Kw(Keyword::True)
            | TokenKind::Kw(Keyword::False)
            | TokenKind::Minus => PatKind::Lit(Box::new(self.parse_literal_pattern())),
            // Path-led: binding, path, tuple-struct, or struct pattern.
            TokenKind::Ident(_) | TokenKind::Kw(Keyword::UpperSelf) => {
                let path = self.parse_path();
                if self.at(&TokenKind::LParen) {
                    self.bump();
                    let mut elems = Vec::new();
                    while !self.at(&TokenKind::RParen) && !self.is_eof() {
                        elems.push(self.parse_pattern());
                        if !self.eat(&TokenKind::Comma) {
                            break;
                        }
                    }
                    self.expect(&TokenKind::RParen);
                    PatKind::TupleStruct { path, elems }
                } else if self.at(&TokenKind::LBrace) && !self.restrict_struct {
                    self.bump();
                    let mut fields = Vec::new();
                    let mut rest = false;
                    while !self.at(&TokenKind::RBrace) && !self.is_eof() {
                        if self.eat(&TokenKind::DotDot) {
                            rest = true;
                            break;
                        }
                        let name = self.ident();
                        let pat = if self.eat(&TokenKind::Colon) {
                            Some(self.parse_pattern())
                        } else {
                            None
                        };
                        fields.push(FieldPat { name, pat });
                        if !self.eat(&TokenKind::Comma) {
                            break;
                        }
                    }
                    self.expect(&TokenKind::RBrace);
                    PatKind::Struct { path, fields, rest }
                } else if path.is_single() {
                    // A bare lowercase-or-any single name is a binding; it may
                    // carry a subpattern via `@`. (Distinguishing unit variants
                    // from bindings is deferred to name resolution.)
                    let name = path.segments.into_iter().next().unwrap().ident;
                    let sub = if self.eat(&TokenKind::At) {
                        Some(Box::new(self.parse_pattern()))
                    } else {
                        None
                    };
                    PatKind::Binding { name, sub }
                } else {
                    PatKind::Path(path)
                }
            }
            _ => {
                self.err_expected("a pattern");
                PatKind::Error
            }
        };
        Pat { kind, span: self.finish(start) }
    }

    /// Parse a literal (optionally negated) used in pattern position.
    fn parse_literal_pattern(&mut self) -> Expr {
        let start = self.cur_span();
        if self.eat(&TokenKind::Minus) {
            let inner = self.parse_literal_pattern();
            return Expr {
                kind: ExprKind::Unary(UnOp::Neg, Box::new(inner)),
                span: self.finish(start),
            };
        }
        let kind = match self.kind().clone() {
            TokenKind::Int { value, suffix } => ExprKind::Int(value, suffix),
            TokenKind::Float { value, suffix } => ExprKind::Float(value, suffix),
            TokenKind::Str(s) => ExprKind::Str(s),
            TokenKind::Char(c) => ExprKind::Char(c),
            TokenKind::Kw(Keyword::True) => ExprKind::Bool(true),
            TokenKind::Kw(Keyword::False) => ExprKind::Bool(false),
            _ => ExprKind::Error,
        };
        self.bump();
        Expr { kind, span: self.finish(start) }
    }
}
