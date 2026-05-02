//! Type, path, and generic-parameter parsing (grammar spec §7.1, §3.2).

use aurora_ast::{GenericParam, Ident, Path, PathSeg, RegionKind, Type, TypeKind, WherePred};
use aurora_lexer::{Keyword, TokenKind};

use crate::Parser;

impl Parser {
    /// Parse an identifier, or the contextual `self`/`Self` keywords used in
    /// path positions.
    pub(crate) fn ident_or_self(&mut self) -> Ident {
        match self.kind().clone() {
            TokenKind::Ident(name) => {
                let span = self.cur_span();
                self.bump();
                Ident { name, span }
            }
            TokenKind::Kw(Keyword::UpperSelf) => {
                let span = self.cur_span();
                self.bump();
                Ident { name: "Self".into(), span }
            }
            TokenKind::Kw(Keyword::LowerSelf) => {
                let span = self.cur_span();
                self.bump();
                Ident { name: "self".into(), span }
            }
            _ => self.ident(),
        }
    }

    pub(crate) fn parse_type(&mut self) -> Type {
        let start = self.cur_span();
        let kind = match self.kind() {
            TokenKind::Tilde => {
                self.bump();
                TypeKind::Owned(Box::new(self.parse_type()))
            }
            // `#frame T` / `#level T` / `#perm T` — a region contract on a
            // parameter or return type (checking-only; `T` is the representation).
            TokenKind::Hash => {
                self.bump();
                let region = match self.kind() {
                    TokenKind::Ident(s) if s == "frame" => RegionKind::Frame,
                    TokenKind::Ident(s) if s == "level" => RegionKind::Level,
                    TokenKind::Ident(s) if s == "perm" => RegionKind::Perm,
                    _ => {
                        self.err_expected("a region: `frame`, `level`, or `perm`");
                        RegionKind::Frame
                    }
                };
                if matches!(self.kind(), TokenKind::Ident(_)) {
                    self.bump();
                }
                TypeKind::Region(region, Box::new(self.parse_type()))
            }
            TokenKind::Amp => {
                self.bump();
                let mutable = self.eat_kw(Keyword::Mut);
                TypeKind::Ref { mutable, inner: Box::new(self.parse_type()) }
            }
            TokenKind::LBracket => {
                self.bump();
                let elem = Box::new(self.parse_type());
                let len = if self.eat(&TokenKind::Semi) {
                    Some(Box::new(self.parse_expr()))
                } else {
                    None
                };
                self.expect(&TokenKind::RBracket);
                TypeKind::Array { elem, len }
            }
            TokenKind::LParen => {
                self.bump();
                let mut elems = Vec::new();
                let mut trailing_comma = false;
                while !self.at(&TokenKind::RParen) && !self.is_eof() {
                    elems.push(self.parse_type());
                    if self.eat(&TokenKind::Comma) {
                        trailing_comma = true;
                    } else {
                        trailing_comma = false;
                        break;
                    }
                }
                self.expect(&TokenKind::RParen);
                // `(T)` with no comma is just a parenthesized type.
                if elems.len() == 1 && !trailing_comma {
                    return elems.into_iter().next().unwrap();
                }
                TypeKind::Tuple(elems)
            }
            TokenKind::Kw(Keyword::Fn) => {
                self.bump();
                self.expect(&TokenKind::LParen);
                let mut params = Vec::new();
                while !self.at(&TokenKind::RParen) && !self.is_eof() {
                    params.push(self.parse_type());
                    if !self.eat(&TokenKind::Comma) {
                        break;
                    }
                }
                self.expect(&TokenKind::RParen);
                self.expect(&TokenKind::Arrow);
                let ret = Box::new(self.parse_type());
                TypeKind::Fn { params, ret }
            }
            TokenKind::Ident(s) if s == "_" => {
                self.bump();
                TypeKind::Infer
            }
            TokenKind::Ident(s) if s == "dyn" => {
                self.bump();
                TypeKind::Dyn(self.parse_type_path())
            }
            TokenKind::Ident(_) | TokenKind::Kw(Keyword::UpperSelf) => {
                TypeKind::Path(self.parse_type_path())
            }
            _ => {
                self.err_expected("a type");
                TypeKind::Error
            }
        };
        Type { kind, span: self.finish(start) }
    }

    /// Parse a `::`-separated path in **expression/value position**: a bare `<`
    /// is the comparison operator, never generics (expression generics use the
    /// turbofish `::<>`, handled in the postfix parser).
    pub(crate) fn parse_path(&mut self) -> Path {
        self.parse_path_inner(false)
    }

    /// Parse a path in **type position**, where `<...>` are generic arguments
    /// (`Vec<T>`, `Handle<Mesh>`, `rc<Texture>`).
    pub(crate) fn parse_type_path(&mut self) -> Path {
        self.parse_path_inner(true)
    }

    fn parse_path_inner(&mut self, allow_generics: bool) -> Path {
        let start = self.cur_span();
        let mut segments = vec![self.parse_path_seg(allow_generics)];
        while self.at(&TokenKind::ColonColon) {
            // A turbofish (`::<`) belongs to a call expression, not a path
            // segment, so stop here and let the caller handle it.
            if matches!(self.nth_kind(1), TokenKind::Lt) {
                break;
            }
            self.bump(); // ::
            segments.push(self.parse_path_seg(allow_generics));
        }
        Path { segments, span: self.finish(start) }
    }

    fn parse_path_seg(&mut self, allow_generics: bool) -> PathSeg {
        let ident = self.ident_or_self();
        let args = if allow_generics && self.at(&TokenKind::Lt) {
            self.parse_generic_args()
        } else {
            Vec::new()
        };
        PathSeg { ident, args }
    }

    /// `< Type, Type, ... >`
    pub(crate) fn parse_generic_args(&mut self) -> Vec<Type> {
        self.bump(); // <
        let mut args = Vec::new();
        while !self.at(&TokenKind::Gt) && !self.is_eof() {
            args.push(self.parse_type());
            if !self.eat(&TokenKind::Comma) {
                break;
            }
        }
        self.expect(&TokenKind::Gt);
        args
    }

    /// `< T: Bound + Bound, U >` for declarations. Returns empty if absent.
    pub(crate) fn parse_generics(&mut self) -> Vec<GenericParam> {
        if !self.eat(&TokenKind::Lt) {
            return Vec::new();
        }
        let mut params = Vec::new();
        while !self.at(&TokenKind::Gt) && !self.is_eof() {
            let name = self.ident();
            let bounds = if self.eat(&TokenKind::Colon) {
                self.parse_bounds()
            } else {
                Vec::new()
            };
            params.push(GenericParam { name, bounds });
            if !self.eat(&TokenKind::Comma) {
                break;
            }
        }
        self.expect(&TokenKind::Gt);
        params
    }

    /// `Path + Path + ...`
    pub(crate) fn parse_bounds(&mut self) -> Vec<Path> {
        let mut bounds = vec![self.parse_path()];
        while self.eat(&TokenKind::Plus) {
            bounds.push(self.parse_path());
        }
        bounds
    }

    /// `where T: Bound, U: Bound` — returns empty if no `where`.
    pub(crate) fn parse_where(&mut self) -> Vec<WherePred> {
        if !self.eat_kw(Keyword::Where) {
            return Vec::new();
        }
        let mut preds = Vec::new();
        loop {
            let name = self.ident();
            self.expect(&TokenKind::Colon);
            let bounds = self.parse_bounds();
            preds.push(WherePred { name, bounds });
            if !self.eat(&TokenKind::Comma) {
                break;
            }
            // Allow a where-clause to be followed by `{` (fn/impl body) or `;`.
            if self.at(&TokenKind::LBrace) || self.at(&TokenKind::Semi) {
                break;
            }
        }
        preds
    }
}
