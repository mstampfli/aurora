//! Item parsing: functions, structs, enums, components, systems, traits,
//! impls, pipelines, consts, modules, and `use` (grammar spec §3, §4).

use aurora_ast::{
    AssocItem, Attr, AttrArg, ConstDecl, EnumDecl, ExprKind, Field, FnDecl, Ident, Item, ItemKind,
    Param, Path, PathSeg, PipelineDecl, PipelineField, StructBody, StructDecl, SysParam, SysSched,
    SystemDecl, TraitDecl, TypeKind, UseDecl, UseKind, Variant, VariantData, Vis,
};
use aurora_lexer::{Keyword, TokenKind};

use crate::Parser;

impl Parser {
    /// Parse one top-level item. Returns `None` only when nothing could be
    /// recognized (after emitting a diagnostic and recovering).
    pub(crate) fn parse_item(&mut self) -> Option<Item> {
        let start = self.cur_span();
        let attrs = self.parse_attrs();
        let vis = if self.eat_kw(Keyword::Pub) { Vis::Pub } else { Vis::Private };

        let kind = match self.kind() {
            TokenKind::Kw(Keyword::Use) => ItemKind::Use(self.parse_use()),
            TokenKind::Kw(Keyword::Mod) => self.parse_mod(),
            TokenKind::Kw(Keyword::Fn) => ItemKind::Fn(self.parse_fn()),
            TokenKind::Kw(Keyword::Struct) => ItemKind::Struct(self.parse_struct_decl()),
            TokenKind::Kw(Keyword::Enum) => ItemKind::Enum(self.parse_enum()),
            TokenKind::Kw(Keyword::Component) => {
                self.bump();
                ItemKind::Component(self.parse_struct_after_kw())
            }
            TokenKind::Kw(Keyword::System) => ItemKind::System(self.parse_system()),
            TokenKind::Kw(Keyword::Trait) => ItemKind::Trait(self.parse_trait()),
            TokenKind::Kw(Keyword::Impl) => ItemKind::Impl(self.parse_impl()),
            TokenKind::Kw(Keyword::Pipeline) => ItemKind::Pipeline(self.parse_pipeline()),
            TokenKind::Kw(Keyword::Const) => ItemKind::Const(self.parse_const()),
            TokenKind::Kw(Keyword::Comptime) => {
                self.bump();
                ItemKind::Comptime(self.parse_block())
            }
            _ => {
                self.err_expected("an item (fn, struct, component, system, ...)");
                self.recover_to_item();
                if attrs.is_empty() && vis == Vis::Private {
                    return None;
                }
                ItemKind::Error
            }
        };

        Some(Item { attrs, vis, kind, span: self.finish(start) })
    }

    // --- attributes ----------------------------------------------------------

    fn parse_attrs(&mut self) -> Vec<Attr> {
        let mut attrs = Vec::new();
        while self.at(&TokenKind::At) {
            let start = self.cur_span();
            self.bump(); // @
            let name = self.ident();
            let args = if self.at(&TokenKind::LParen) {
                self.parse_attr_args()
            } else {
                Vec::new()
            };
            attrs.push(Attr { name, args, span: self.finish(start) });
        }
        attrs
    }

    fn parse_attr_args(&mut self) -> Vec<AttrArg> {
        self.bump(); // (
        let mut args = Vec::new();
        while !self.at(&TokenKind::RParen) && !self.is_eof() {
            // Named arg uses either `key: value` or `key = value`.
            let named = self.at_ident()
                && matches!(self.nth_kind(1), TokenKind::Colon | TokenKind::Eq);
            if named {
                let name = self.ident();
                self.bump(); // : or =
                let value = self.parse_expr();
                args.push(AttrArg::Named(name, value));
            } else {
                args.push(AttrArg::Positional(self.parse_expr()));
            }
            if !self.eat(&TokenKind::Comma) {
                break;
            }
        }
        self.expect(&TokenKind::RParen);
        args
    }

    // --- use / mod -----------------------------------------------------------

    fn parse_use(&mut self) -> UseDecl {
        let start = self.cur_span();
        self.bump(); // use
        let mut segs = vec![self.ident_or_self()];
        loop {
            if self.at(&TokenKind::ColonColon) {
                if matches!(self.nth_kind(1), TokenKind::LBrace) {
                    self.bump(); // ::
                    self.bump(); // {
                    let mut names = Vec::new();
                    while !self.at(&TokenKind::RBrace) && !self.is_eof() {
                        names.push(self.ident());
                        if !self.eat(&TokenKind::Comma) {
                            break;
                        }
                    }
                    self.expect(&TokenKind::RBrace);
                    self.eat(&TokenKind::Semi); // optional terminator
                    let path = path_from_idents(segs, start, self.prev().span);
                    return UseDecl { path, kind: UseKind::Group(names) };
                }
                self.bump(); // ::
                segs.push(self.ident());
            } else {
                break;
            }
        }
        let alias = if self.eat_kw(Keyword::As) { Some(self.ident()) } else { None };
        self.eat(&TokenKind::Semi); // optional terminator
        let path = path_from_idents(segs, start, self.prev().span);
        UseDecl { path, kind: UseKind::Single(alias) }
    }

    fn parse_mod(&mut self) -> ItemKind {
        self.bump(); // mod
        let name = self.ident();
        if self.eat(&TokenKind::LBrace) {
            let mut items = Vec::new();
            while !self.at(&TokenKind::RBrace) && !self.is_eof() {
                let before = self.pos;
                if let Some(item) = self.parse_item() {
                    items.push(item);
                }
                if self.pos == before {
                    self.bump();
                }
            }
            self.expect(&TokenKind::RBrace);
            ItemKind::Mod(name, Some(items))
        } else {
            self.eat(&TokenKind::Semi);
            ItemKind::Mod(name, None)
        }
    }

    // --- functions -----------------------------------------------------------

    fn parse_fn(&mut self) -> FnDecl {
        self.bump(); // fn
        let name = self.ident();
        let generics = self.parse_generics();
        self.expect(&TokenKind::LParen);
        let params = self.parse_params();
        self.expect(&TokenKind::RParen);
        let ret = if self.eat(&TokenKind::Arrow) {
            Some(self.parse_type())
        } else {
            None
        };
        let where_clause = self.parse_where();
        let body = if self.at(&TokenKind::LBrace) {
            Some(self.parse_block())
        } else {
            self.eat(&TokenKind::Semi); // signature-only (trait method)
            None
        };
        FnDecl { name, generics, params, ret, where_clause, body }
    }

    fn parse_params(&mut self) -> Vec<Param> {
        let mut params = Vec::new();
        while !self.at(&TokenKind::RParen) && !self.is_eof() {
            if self.at_kw(Keyword::LowerSelf) {
                self.bump();
                params.push(Param::SelfParam { by_ref: false, mutable: false });
            } else if self.at(&TokenKind::Amp)
                && matches!(self.nth_kind(1), TokenKind::Kw(Keyword::LowerSelf))
            {
                self.bump(); // &
                self.bump(); // self
                params.push(Param::SelfParam { by_ref: true, mutable: false });
            } else if self.at(&TokenKind::Amp)
                && matches!(self.nth_kind(1), TokenKind::Kw(Keyword::Mut))
                && matches!(self.nth_kind(2), TokenKind::Kw(Keyword::LowerSelf))
            {
                self.bump(); // &
                self.bump(); // mut
                self.bump(); // self
                params.push(Param::SelfParam { by_ref: true, mutable: true });
            } else {
                let mutable = self.eat_kw(Keyword::Mut);
                let name = self.ident();
                self.expect(&TokenKind::Colon);
                let ty = self.parse_type();
                params.push(Param::Normal { mutable, name, ty });
            }
            if !self.eat(&TokenKind::Comma) {
                break;
            }
        }
        params
    }

    // --- structs / components ------------------------------------------------

    fn parse_struct_decl(&mut self) -> StructDecl {
        self.bump(); // struct
        self.parse_struct_after_kw()
    }

    /// Parse the body of a struct/component after its introducer keyword.
    fn parse_struct_after_kw(&mut self) -> StructDecl {
        let name = self.ident();
        let generics = self.parse_generics();
        let body = if self.at(&TokenKind::LBrace) {
            StructBody::Named(self.parse_named_fields())
        } else if self.at(&TokenKind::LParen) {
            self.bump();
            let mut tys = Vec::new();
            while !self.at(&TokenKind::RParen) && !self.is_eof() {
                tys.push(self.parse_type());
                if !self.eat(&TokenKind::Comma) {
                    break;
                }
            }
            self.expect(&TokenKind::RParen);
            self.eat(&TokenKind::Semi);
            StructBody::Tuple(tys)
        } else {
            self.eat(&TokenKind::Semi);
            StructBody::Unit
        };
        StructDecl { name, generics, body }
    }

    fn parse_named_fields(&mut self) -> Vec<Field> {
        self.bump(); // {
        let mut fields = Vec::new();
        // Fields are separated by a comma OR a newline (ASI), so we loop to the
        // closing brace rather than breaking on a missing comma.
        while !self.at(&TokenKind::RBrace) && !self.is_eof() {
            let before = self.pos;
            let fstart = self.cur_span();
            let attrs = self.parse_attrs();
            let vis = if self.eat_kw(Keyword::Pub) { Vis::Pub } else { Vis::Private };
            let name = self.ident();
            self.expect(&TokenKind::Colon);
            let ty = self.parse_type();
            let default = if self.eat(&TokenKind::Eq) {
                Some(self.parse_expr())
            } else {
                None
            };
            let span = self.finish(fstart);
            fields.push(Field { attrs, vis, name, ty, default, span });
            self.eat(&TokenKind::Comma); // optional separator
            if self.pos == before {
                self.bump(); // progress guard
            }
        }
        self.expect(&TokenKind::RBrace);
        fields
    }

    // --- enums ---------------------------------------------------------------

    fn parse_enum(&mut self) -> EnumDecl {
        self.bump(); // enum
        let name = self.ident();
        let generics = self.parse_generics();
        self.expect(&TokenKind::LBrace);
        let mut variants = Vec::new();
        while !self.at(&TokenKind::RBrace) && !self.is_eof() {
            let before = self.pos;
            let vstart = self.cur_span();
            let vname = self.ident();
            let data = if self.at(&TokenKind::LParen) {
                self.bump();
                let mut tys = Vec::new();
                while !self.at(&TokenKind::RParen) && !self.is_eof() {
                    tys.push(self.parse_type());
                    if !self.eat(&TokenKind::Comma) {
                        break;
                    }
                }
                self.expect(&TokenKind::RParen);
                VariantData::Tuple(tys)
            } else if self.at(&TokenKind::LBrace) {
                VariantData::Struct(self.parse_named_fields())
            } else {
                VariantData::Unit
            };
            let discriminant = if self.eat(&TokenKind::Eq) {
                Some(self.parse_expr())
            } else {
                None
            };
            variants.push(Variant { name: vname, data, discriminant, span: self.finish(vstart) });
            self.eat(&TokenKind::Comma); // optional separator (comma or newline)
            if self.pos == before {
                self.bump(); // progress guard
            }
        }
        self.expect(&TokenKind::RBrace);
        EnumDecl { name, generics, variants }
    }

    // --- systems -------------------------------------------------------------

    fn parse_system(&mut self) -> SystemDecl {
        self.bump(); // system
        let name = self.ident();
        self.expect(&TokenKind::LParen);
        let mut params = Vec::new();
        while !self.at(&TokenKind::RParen) && !self.is_eof() {
            let pname = self.ident();
            self.expect(&TokenKind::Colon);
            let ty = self.parse_type();
            params.push(SysParam { name: pname, ty });
            if !self.eat(&TokenKind::Comma) {
                break;
            }
        }
        self.expect(&TokenKind::RParen);

        let mut schedule = Vec::new();
        loop {
            if self.at_ctx("after") {
                self.bump();
                schedule.push(SysSched::After(self.parse_paren_paths()));
            } else if self.at_ctx("before") {
                self.bump();
                schedule.push(SysSched::Before(self.parse_paren_paths()));
            } else if self.at_ctx("stage") {
                self.bump();
                self.expect(&TokenKind::LParen);
                let id = self.ident();
                self.expect(&TokenKind::RParen);
                schedule.push(SysSched::Stage(id));
            } else {
                break;
            }
        }

        let body = self.parse_block();
        SystemDecl { name, params, schedule, body }
    }

    fn parse_paren_paths(&mut self) -> Vec<Path> {
        self.expect(&TokenKind::LParen);
        let mut paths = Vec::new();
        while !self.at(&TokenKind::RParen) && !self.is_eof() {
            paths.push(self.parse_path());
            if !self.eat(&TokenKind::Comma) {
                break;
            }
        }
        self.expect(&TokenKind::RParen);
        paths
    }

    // --- traits / impls ------------------------------------------------------

    fn parse_trait(&mut self) -> TraitDecl {
        self.bump(); // trait
        let name = self.ident();
        let generics = self.parse_generics();
        let supertraits = if self.eat(&TokenKind::Colon) {
            self.parse_bounds()
        } else {
            Vec::new()
        };
        self.expect(&TokenKind::LBrace);
        let items = self.parse_assoc_items();
        TraitDecl { name, generics, supertraits, items }
    }

    fn parse_impl(&mut self) -> aurora_ast::ImplDecl {
        self.bump(); // impl
        let generics = self.parse_generics();
        let first = self.parse_type();
        let (trait_, self_ty) = if self.eat_kw(Keyword::For) {
            let trait_path = match first.kind {
                TypeKind::Path(p) => Some(p),
                _ => {
                    self.error(first.span, "expected a trait path before `for`", "not a trait");
                    None
                }
            };
            (trait_path, self.parse_type())
        } else {
            (None, first)
        };
        let where_clause = self.parse_where();
        self.expect(&TokenKind::LBrace);
        let items = self.parse_assoc_items();
        aurora_ast::ImplDecl { generics, trait_, self_ty, where_clause, items }
    }

    fn parse_assoc_items(&mut self) -> Vec<AssocItem> {
        let mut items = Vec::new();
        while !self.at(&TokenKind::RBrace) && !self.is_eof() {
            let before = self.pos;
            // Attributes/visibility on associated items are accepted but not yet
            // threaded onto FnDecl/ConstDecl (tracked for a later phase).
            let _attrs = self.parse_attrs();
            let _vis = self.eat_kw(Keyword::Pub);
            if self.at_kw(Keyword::Fn) {
                items.push(AssocItem::Fn(self.parse_fn()));
            } else if self.at_kw(Keyword::Const) {
                items.push(AssocItem::Const(self.parse_const()));
            } else {
                self.err_expected("an associated `fn` or `const`");
            }
            if self.pos == before {
                self.bump();
            }
        }
        self.expect(&TokenKind::RBrace);
        items
    }

    // --- pipelines / consts --------------------------------------------------

    fn parse_pipeline(&mut self) -> PipelineDecl {
        self.bump(); // pipeline
        let name = self.ident();
        self.expect(&TokenKind::LBrace);
        let mut fields = Vec::new();
        while !self.at(&TokenKind::RBrace) && !self.is_eof() {
            let before = self.pos;
            let key = self.ident();
            let value = if self.eat(&TokenKind::Colon) {
                self.parse_expr()
            } else {
                // Shorthand `vs` => value is the path `vs`.
                let span = key.span;
                aurora_ast::Expr {
                    kind: ExprKind::Path(Path {
                        segments: vec![PathSeg { ident: key.clone(), args: Vec::new() }],
                        span,
                    }),
                    span,
                }
            };
            fields.push(PipelineField { key, value });
            self.eat(&TokenKind::Comma); // optional separator (comma or newline)
            if self.pos == before {
                self.bump(); // progress guard
            }
        }
        self.expect(&TokenKind::RBrace);
        PipelineDecl { name, fields }
    }

    fn parse_const(&mut self) -> ConstDecl {
        self.bump(); // const
        let name = self.ident();
        let ty = if self.eat(&TokenKind::Colon) {
            Some(self.parse_type())
        } else {
            None
        };
        self.expect(&TokenKind::Eq);
        let value = self.parse_expr();
        self.eat(&TokenKind::Semi);
        ConstDecl { name, ty, value }
    }
}

fn path_from_idents(idents: Vec<Ident>, start: aurora_span::Span, end: aurora_span::Span) -> Path {
    let segments = idents
        .into_iter()
        .map(|ident| PathSeg { ident, args: Vec::new() })
        .collect();
    Path { segments, span: start.to(end) }
}
