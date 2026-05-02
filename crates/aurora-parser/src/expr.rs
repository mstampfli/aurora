//! Expression parsing: precedence-climbing (Pratt) over the operator table in
//! grammar spec §5.2, plus primaries, postfix chains, and `query<...>`.

use aurora_ast::{
    Arg, BinOp, Expr, ExprKind, FieldAccess, FieldInit, IfExpr, MatchArm, Param, QTerm, QueryExpr,
    RegionKind, Type, TypeKind, UnOp,
};
use aurora_lexer::{Keyword, TokenKind};

use crate::Parser;

/// Binding powers (higher binds tighter). Assignment is right-associative.
mod bp {
    pub const ASSIGN_L: u8 = 2;
    pub const ASSIGN_R: u8 = 1;
    pub const PIPE: u8 = 4;
    pub const RANGE: u8 = 6;
    pub const OR: u8 = 8;
    pub const AND: u8 = 10;
    pub const CMP: u8 = 12;
    pub const ADD: u8 = 14;
    pub const MUL: u8 = 16;
    pub const CAST: u8 = 18;
}

impl Parser {
    pub(crate) fn parse_expr(&mut self) -> Expr {
        self.parse_expr_bp(0)
    }

    fn parse_expr_bp(&mut self, min_bp: u8) -> Expr {
        let mut lhs = self.parse_unary();

        loop {
            // Assignment / compound assignment (right-associative, lowest).
            if let Some(op) = self.assign_op() {
                if bp::ASSIGN_L < min_bp {
                    break;
                }
                self.bump();
                let rhs = self.parse_expr_bp(bp::ASSIGN_R);
                let span = lhs.span.to(rhs.span);
                lhs = Expr {
                    kind: ExprKind::Assign(op, Box::new(lhs), Box::new(rhs)),
                    span,
                };
                continue;
            }

            // `expr as Type`
            if self.at_kw(Keyword::As) {
                if bp::CAST < min_bp {
                    break;
                }
                self.bump();
                let ty = self.parse_type();
                let span = lhs.span.to(ty.span);
                lhs = Expr { kind: ExprKind::Cast(Box::new(lhs), ty), span };
                continue;
            }

            // Range.
            if self.at(&TokenKind::DotDot) || self.at(&TokenKind::DotDotEq) {
                if bp::RANGE < min_bp {
                    break;
                }
                let inclusive = self.at(&TokenKind::DotDotEq);
                self.bump();
                let end = if self.starts_expr() {
                    Some(Box::new(self.parse_expr_bp(bp::RANGE + 1)))
                } else {
                    None
                };
                let hi = end.as_ref().map(|e| e.span).unwrap_or(lhs.span);
                let span = lhs.span.to(hi);
                lhs = Expr {
                    kind: ExprKind::Range { start: Some(Box::new(lhs)), end, inclusive },
                    span,
                };
                continue;
            }

            // Pipe `|>`.
            if self.at(&TokenKind::PipeGt) {
                if bp::PIPE < min_bp {
                    break;
                }
                self.bump();
                let func = self.parse_expr_bp(bp::PIPE + 1);
                let span = lhs.span.to(func.span);
                lhs = Expr {
                    kind: ExprKind::Pipe { value: Box::new(lhs), func: Box::new(func) },
                    span,
                };
                continue;
            }

            // Plain binary operators.
            if let Some((op, lbp, rbp)) = self.binop() {
                if lbp < min_bp {
                    break;
                }
                self.bump();
                let rhs = self.parse_expr_bp(rbp);
                let span = lhs.span.to(rhs.span);
                lhs = Expr {
                    kind: ExprKind::Binary(op, Box::new(lhs), Box::new(rhs)),
                    span,
                };
                continue;
            }

            break;
        }

        lhs
    }

    /// Map the current token to a compound-assignment operator, if any.
    /// Returns `Some(None)` for plain `=`, `Some(Some(op))` for `+=` etc.
    fn assign_op(&self) -> Option<Option<BinOp>> {
        Some(match self.kind() {
            TokenKind::Eq => None,
            TokenKind::PlusEq => Some(BinOp::Add),
            TokenKind::MinusEq => Some(BinOp::Sub),
            TokenKind::StarEq => Some(BinOp::Mul),
            TokenKind::SlashEq => Some(BinOp::Div),
            TokenKind::PercentEq => Some(BinOp::Rem),
            _ => return None,
        })
    }

    /// Map the current token to a binary operator and its binding powers.
    fn binop(&self) -> Option<(BinOp, u8, u8)> {
        let (op, l) = match self.kind() {
            TokenKind::Kw(Keyword::Or) => (BinOp::Or, bp::OR),
            TokenKind::Kw(Keyword::And) => (BinOp::And, bp::AND),
            TokenKind::EqEq => (BinOp::Eq, bp::CMP),
            TokenKind::BangEq => (BinOp::Ne, bp::CMP),
            TokenKind::Lt => (BinOp::Lt, bp::CMP),
            TokenKind::Gt => (BinOp::Gt, bp::CMP),
            TokenKind::Le => (BinOp::Le, bp::CMP),
            TokenKind::Ge => (BinOp::Ge, bp::CMP),
            TokenKind::Plus => (BinOp::Add, bp::ADD),
            TokenKind::Minus => (BinOp::Sub, bp::ADD),
            TokenKind::Star => (BinOp::Mul, bp::MUL),
            TokenKind::Slash => (BinOp::Div, bp::MUL),
            TokenKind::Percent => (BinOp::Rem, bp::MUL),
            _ => return None,
        };
        Some((op, l, l + 1))
    }

    // --- unary & postfix -----------------------------------------------------

    fn parse_unary(&mut self) -> Expr {
        let start = self.cur_span();
        let op = match self.kind() {
            TokenKind::Minus => Some(UnOp::Neg),
            TokenKind::Kw(Keyword::Not) => Some(UnOp::Not),
            TokenKind::Star => Some(UnOp::Deref),
            TokenKind::Tilde => Some(UnOp::Own),
            TokenKind::Amp => {
                self.bump();
                let mutable = self.eat_kw(Keyword::Mut);
                let operand = self.parse_unary();
                let span = self.finish(start);
                let op = if mutable { UnOp::RefMut } else { UnOp::RefShared };
                return Expr { kind: ExprKind::Unary(op, Box::new(operand)), span };
            }
            _ => None,
        };
        if let Some(op) = op {
            self.bump();
            let operand = self.parse_unary();
            let span = self.finish(start);
            return Expr { kind: ExprKind::Unary(op, Box::new(operand)), span };
        }
        self.parse_postfix()
    }

    fn parse_postfix(&mut self) -> Expr {
        let start = self.cur_span();
        let mut expr = self.parse_primary();
        loop {
            // A `(` or `[` at the start of a new line begins a new statement, not
            // a call/index on the previous expression (newline-aware ASI).
            let nl = self.cur().nl_before;
            match self.kind() {
                TokenKind::LParen if !nl => {
                    let args = self.parse_args();
                    let span = self.finish(start);
                    expr = Expr {
                        kind: ExprKind::Call {
                            callee: Box::new(expr),
                            type_args: Vec::new(),
                            args,
                        },
                        span,
                    };
                }
                // Turbofish call: `expr::<T>(...)`
                TokenKind::ColonColon if matches!(self.nth_kind(1), TokenKind::Lt) => {
                    self.bump(); // ::
                    let type_args = self.parse_generic_args();
                    let args = if self.at(&TokenKind::LParen) {
                        self.parse_args()
                    } else {
                        self.err_expected("`(` after turbofish arguments");
                        Vec::new()
                    };
                    let span = self.finish(start);
                    expr = Expr {
                        kind: ExprKind::Call { callee: Box::new(expr), type_args, args },
                        span,
                    };
                }
                TokenKind::Dot => {
                    self.bump();
                    let field = match self.kind().clone() {
                        TokenKind::Int { value, .. } => {
                            self.bump();
                            FieldAccess::Index(value as u32)
                        }
                        TokenKind::Ident(_) => FieldAccess::Named(self.ident()),
                        _ => {
                            self.err_expected("a field name or tuple index");
                            FieldAccess::Index(0)
                        }
                    };
                    let span = self.finish(start);
                    expr = Expr { kind: ExprKind::Field { base: Box::new(expr), field }, span };
                }
                TokenKind::LBracket if !nl => {
                    self.bump();
                    let index = self.parse_expr();
                    self.expect(&TokenKind::RBracket);
                    let span = self.finish(start);
                    expr = Expr {
                        kind: ExprKind::Index { base: Box::new(expr), index: Box::new(index) },
                        span,
                    };
                }
                // Postfix `?` — error propagation.
                TokenKind::Question => {
                    self.bump();
                    let span = self.finish(start);
                    expr = Expr { kind: ExprKind::Try(Box::new(expr)), span };
                }
                _ => break,
            }
        }
        expr
    }

    /// `( Arg, Arg, ... )` where each Arg is `[name :] expr`.
    fn parse_args(&mut self) -> Vec<Arg> {
        self.bump(); // (
        let saved = self.restrict_struct;
        self.restrict_struct = false; // struct literals are fine inside parens
        let mut args = Vec::new();
        while !self.at(&TokenKind::RParen) && !self.is_eof() {
            let name = if self.at_ident() && matches!(self.nth_kind(1), TokenKind::Colon) {
                let id = self.ident();
                self.bump(); // :
                Some(id)
            } else {
                None
            };
            let value = self.parse_expr();
            args.push(Arg { name, value });
            if !self.eat(&TokenKind::Comma) {
                break;
            }
        }
        self.restrict_struct = saved;
        self.expect(&TokenKind::RParen);
        args
    }

    // --- primaries -----------------------------------------------------------

    fn parse_primary(&mut self) -> Expr {
        let start = self.cur_span();
        let kind = match self.kind().clone() {
            TokenKind::Int { value, suffix } => {
                self.bump();
                ExprKind::Int(value, suffix)
            }
            TokenKind::Float { value, suffix } => {
                self.bump();
                ExprKind::Float(value, suffix)
            }
            TokenKind::Str(s) => {
                self.bump();
                ExprKind::Str(s)
            }
            TokenKind::Char(c) => {
                self.bump();
                ExprKind::Char(c)
            }
            TokenKind::Kw(Keyword::True) => {
                self.bump();
                ExprKind::Bool(true)
            }
            TokenKind::Kw(Keyword::False) => {
                self.bump();
                ExprKind::Bool(false)
            }
            TokenKind::Kw(Keyword::LowerSelf) => {
                self.bump();
                ExprKind::SelfExpr
            }
            TokenKind::Dot => {
                // `.variant` shorthand (e.g. `depth: .less`)
                self.bump();
                ExprKind::Dot(self.ident())
            }
            TokenKind::LParen => return self.parse_paren_or_tuple(),
            TokenKind::LBracket => return self.parse_array(),
            TokenKind::LBrace => ExprKind::Block(self.parse_block()),
            TokenKind::Hash => return self.parse_region(),
            TokenKind::Pipe => return self.parse_closure(),
            TokenKind::Kw(Keyword::If) => return self.parse_if(),
            TokenKind::Kw(Keyword::Match) => return self.parse_match(),
            TokenKind::Kw(Keyword::For) => return self.parse_for(),
            TokenKind::Kw(Keyword::While) => return self.parse_while(),
            TokenKind::Kw(Keyword::Loop) => {
                self.bump();
                ExprKind::Loop(self.parse_block())
            }
            TokenKind::Kw(Keyword::Unsafe) => {
                self.bump();
                ExprKind::Unsafe(self.parse_block())
            }
            TokenKind::Kw(Keyword::Comptime) => {
                // `comptime { .. }` as an expression: parse the block (the
                // comptime marker is recovered during lowering).
                self.bump();
                ExprKind::Block(self.parse_block())
            }
            TokenKind::Kw(Keyword::Query) => return self.parse_query(),
            TokenKind::Kw(Keyword::Return) => {
                self.bump();
                let val = if self.starts_expr() {
                    Some(Box::new(self.parse_expr()))
                } else {
                    None
                };
                ExprKind::Return(val)
            }
            TokenKind::Kw(Keyword::Break) => {
                self.bump();
                let val = if self.starts_expr() {
                    Some(Box::new(self.parse_expr()))
                } else {
                    None
                };
                ExprKind::Break(val)
            }
            TokenKind::Kw(Keyword::Continue) => {
                self.bump();
                ExprKind::Continue
            }
            TokenKind::Ident(_) | TokenKind::Kw(Keyword::UpperSelf) => {
                let path = self.parse_path();
                if self.at(&TokenKind::LBrace) && !self.restrict_struct {
                    return self.parse_struct_literal(path, start);
                }
                ExprKind::Path(path)
            }
            _ => {
                self.err_expected("an expression");
                self.bump(); // ensure progress
                ExprKind::Error
            }
        };
        Expr { kind, span: self.finish(start) }
    }

    fn parse_paren_or_tuple(&mut self) -> Expr {
        let start = self.cur_span();
        self.bump(); // (
        let saved = self.restrict_struct;
        self.restrict_struct = false;
        if self.eat(&TokenKind::RParen) {
            self.restrict_struct = saved;
            return Expr { kind: ExprKind::Tuple(Vec::new()), span: self.finish(start) };
        }
        let first = self.parse_expr();
        let kind = if self.at(&TokenKind::Comma) {
            let mut elems = vec![first];
            while self.eat(&TokenKind::Comma) {
                if self.at(&TokenKind::RParen) {
                    break;
                }
                elems.push(self.parse_expr());
            }
            ExprKind::Tuple(elems)
        } else {
            ExprKind::Paren(Box::new(first))
        };
        self.restrict_struct = saved;
        self.expect(&TokenKind::RParen);
        Expr { kind, span: self.finish(start) }
    }

    fn parse_array(&mut self) -> Expr {
        let start = self.cur_span();
        self.bump(); // [
        let saved = self.restrict_struct;
        self.restrict_struct = false;
        let kind = if self.eat(&TokenKind::RBracket) {
            ExprKind::Array(Vec::new())
        } else {
            let first = self.parse_expr();
            if self.eat(&TokenKind::Semi) {
                let count = self.parse_expr();
                self.expect(&TokenKind::RBracket);
                ExprKind::ArrayRepeat { value: Box::new(first), count: Box::new(count) }
            } else {
                let mut elems = vec![first];
                while self.eat(&TokenKind::Comma) {
                    if self.at(&TokenKind::RBracket) {
                        break;
                    }
                    elems.push(self.parse_expr());
                }
                self.expect(&TokenKind::RBracket);
                ExprKind::Array(elems)
            }
        };
        self.restrict_struct = saved;
        Expr { kind, span: self.finish(start) }
    }

    fn parse_struct_literal(&mut self, path: aurora_ast::Path, start: aurora_span::Span) -> Expr {
        self.bump(); // {
        let saved = self.restrict_struct;
        self.restrict_struct = false;
        let mut fields = Vec::new();
        let mut base = None;
        while !self.at(&TokenKind::RBrace) && !self.is_eof() {
            if self.eat(&TokenKind::DotDot) {
                base = Some(Box::new(self.parse_expr()));
                break;
            }
            let name = self.ident();
            let value = if self.eat(&TokenKind::Colon) {
                Some(self.parse_expr())
            } else {
                None
            };
            fields.push(FieldInit { name, value });
            if !self.eat(&TokenKind::Comma) {
                break;
            }
        }
        self.restrict_struct = saved;
        self.expect(&TokenKind::RBrace);
        Expr { kind: ExprKind::Struct { path, fields, base }, span: self.finish(start) }
    }

    fn parse_region(&mut self) -> Expr {
        let start = self.cur_span();
        self.bump(); // #
        let region = match self.kind() {
            TokenKind::Ident(s) if s == "frame" => RegionKind::Frame,
            TokenKind::Ident(s) if s == "level" => RegionKind::Level,
            TokenKind::Ident(s) if s == "perm" => RegionKind::Perm,
            _ => {
                self.err_expected("a region: `frame`, `level`, or `perm`");
                RegionKind::Frame
            }
        };
        if !self.is_eof() && matches!(self.kind(), TokenKind::Ident(_)) {
            self.bump();
        }
        let value = self.parse_unary();
        Expr {
            kind: ExprKind::Region { region, value: Box::new(value) },
            span: self.finish(start),
        }
    }

    fn parse_closure(&mut self) -> Expr {
        let start = self.cur_span();
        self.bump(); // |
        let mut params = Vec::new();
        while !self.at(&TokenKind::Pipe) && !self.is_eof() {
            let mutable = self.eat_kw(Keyword::Mut);
            let name = self.ident();
            let ty = if self.eat(&TokenKind::Colon) {
                self.parse_type()
            } else {
                Type { kind: TypeKind::Infer, span: name.span }
            };
            params.push(Param::Normal { mutable, name, ty });
            if !self.eat(&TokenKind::Comma) {
                break;
            }
        }
        self.expect(&TokenKind::Pipe);
        let body = self.parse_expr();
        Expr {
            kind: ExprKind::Closure { params, body: Box::new(body) },
            span: self.finish(start),
        }
    }

    fn parse_if(&mut self) -> Expr {
        let start = self.cur_span();
        self.bump(); // if
        let cond = self.parse_cond();
        let then_branch = self.parse_block();
        let else_branch = if self.eat_kw(Keyword::Else) {
            if self.at_kw(Keyword::If) {
                Some(Box::new(self.parse_if()))
            } else {
                let bstart = self.cur_span();
                let block = self.parse_block();
                Some(Box::new(Expr { kind: ExprKind::Block(block), span: self.finish(bstart) }))
            }
        } else {
            None
        };
        Expr {
            kind: ExprKind::If(IfExpr {
                cond: Box::new(cond),
                then_branch,
                else_branch,
            }),
            span: self.finish(start),
        }
    }

    fn parse_while(&mut self) -> Expr {
        let start = self.cur_span();
        self.bump(); // while
        let cond = self.parse_cond();
        let body = self.parse_block();
        Expr {
            kind: ExprKind::While { cond: Box::new(cond), body },
            span: self.finish(start),
        }
    }

    fn parse_for(&mut self) -> Expr {
        let start = self.cur_span();
        self.bump(); // for
        let saved = self.restrict_struct;
        self.restrict_struct = true;
        let pat = self.parse_pattern();
        if !self.eat_ctx("in") {
            self.err_expected("`in`");
        }
        let iter = self.parse_expr();
        self.restrict_struct = saved;
        let body = self.parse_block();
        Expr {
            kind: ExprKind::For { pat, iter: Box::new(iter), body },
            span: self.finish(start),
        }
    }

    fn parse_match(&mut self) -> Expr {
        let start = self.cur_span();
        self.bump(); // match
        let scrutinee = self.parse_cond();
        self.expect(&TokenKind::LBrace);
        let mut arms = Vec::new();
        while !self.at(&TokenKind::RBrace) && !self.is_eof() {
            let pat = self.parse_pattern();
            let guard = if self.eat_kw(Keyword::If) {
                Some(self.parse_expr())
            } else {
                None
            };
            self.expect(&TokenKind::FatArrow);
            let body = self.parse_expr();
            arms.push(MatchArm { pat, guard, body });
            self.eat(&TokenKind::Comma); // optional separator
        }
        self.expect(&TokenKind::RBrace);
        Expr {
            kind: ExprKind::Match { scrutinee: Box::new(scrutinee), arms },
            span: self.finish(start),
        }
    }

    /// Parse a condition/scrutinee with struct literals disabled so the trailing
    /// `{` is read as the block, not a struct literal (grammar spec §5.3).
    fn parse_cond(&mut self) -> Expr {
        let saved = self.restrict_struct;
        self.restrict_struct = true;
        let e = self.parse_expr();
        self.restrict_struct = saved;
        e
    }

    // --- query ---------------------------------------------------------------

    fn parse_query(&mut self) -> Expr {
        let start = self.cur_span();
        self.bump(); // query
        self.expect(&TokenKind::Lt);
        let mut terms = Vec::new();
        while !self.at(&TokenKind::Gt) && !self.is_eof() {
            terms.push(self.parse_qterm());
            if !self.eat(&TokenKind::Comma) {
                break;
            }
        }
        self.expect(&TokenKind::Gt);
        let filter = if self.eat_kw(Keyword::Where) {
            Some(Box::new(self.parse_cond()))
        } else {
            None
        };
        Expr {
            kind: ExprKind::Query(QueryExpr { terms, filter }),
            span: self.finish(start),
        }
    }

    fn parse_qterm(&mut self) -> QTerm {
        match self.kind() {
            TokenKind::Amp => {
                self.bump();
                if self.eat_kw(Keyword::Mut) {
                    QTerm::Write(self.parse_path())
                } else {
                    QTerm::Read(self.parse_path())
                }
            }
            TokenKind::Question => {
                self.bump();
                self.expect(&TokenKind::Amp);
                if self.eat_kw(Keyword::Mut) {
                    QTerm::OptWrite(self.parse_path())
                } else {
                    QTerm::OptRead(self.parse_path())
                }
            }
            TokenKind::Bang => {
                self.bump();
                QTerm::Without(self.parse_path())
            }
            TokenKind::Plus => {
                self.bump();
                QTerm::With(self.parse_path())
            }
            TokenKind::Ident(s) if s == "Entity" => {
                self.bump();
                QTerm::Entity
            }
            _ => {
                self.err_expected("a query term (`&T`, `&mut T`, `?&T`, `!T`, `+T`, or `Entity`)");
                // Recover by consuming a token to guarantee progress.
                if !self.is_eof() {
                    self.bump();
                }
                QTerm::Entity
            }
        }
    }

    // --- shared predicate ----------------------------------------------------

    /// Whether the current token can begin an expression (used for optional
    /// operands: range ends, `return`/`break` values).
    pub(crate) fn starts_expr(&self) -> bool {
        matches!(
            self.kind(),
            TokenKind::Int { .. }
                | TokenKind::Float { .. }
                | TokenKind::Str(_)
                | TokenKind::Char(_)
                | TokenKind::Ident(_)
                | TokenKind::LParen
                | TokenKind::LBracket
                | TokenKind::LBrace
                | TokenKind::Minus
                | TokenKind::Star
                | TokenKind::Amp
                | TokenKind::Tilde
                | TokenKind::Hash
                | TokenKind::Pipe
                | TokenKind::Dot
        ) || matches!(
            self.kind(),
            TokenKind::Kw(
                Keyword::True
                    | Keyword::False
                    | Keyword::LowerSelf
                    | Keyword::UpperSelf
                    | Keyword::If
                    | Keyword::Match
                    | Keyword::For
                    | Keyword::While
                    | Keyword::Loop
                    | Keyword::Unsafe
                    | Keyword::Query
                    | Keyword::Not
                    | Keyword::Comptime
            )
        )
    }
}
