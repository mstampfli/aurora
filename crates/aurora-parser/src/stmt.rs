//! Block and statement parsing (grammar spec §5.1, §3.3 ASI rule).

use aurora_ast::{Block, Expr, ExprKind, LetStmt, Stmt};
use aurora_lexer::{Keyword, TokenKind};

use crate::Parser;

impl Parser {
    pub(crate) fn parse_block(&mut self) -> Block {
        let start = self.cur_span();
        if !self.eat(&TokenKind::LBrace) {
            self.err_expected("`{`");
            return Block { stmts: Vec::new(), tail: None, span: self.finish(start) };
        }
        // Struct literals are allowed again inside a fresh block scope.
        let saved = self.restrict_struct;
        self.restrict_struct = false;

        let mut stmts = Vec::new();
        let mut tail = None;

        while !self.at(&TokenKind::RBrace) && !self.is_eof() {
            let before = self.pos;

            // Block-form statements (`if`/`while`/`for`/`loop`/`match`/`{…}`)
            // are delimited by their own `}` and need no trailing separator
            // (e.g. `if c { … } expr`); all other statements end at `;`, a
            // newline, or `}`.
            let needs_sep;
            if self.at_kw(Keyword::Let) {
                stmts.push(self.parse_let());
                needs_sep = true;
            } else if self.eat_kw(Keyword::Defer) {
                let e = self.parse_expr();
                needs_sep = !is_block_expr(&e);
                stmts.push(Stmt::Defer(e));
            } else {
                let e = self.parse_expr();
                if self.at(&TokenKind::RBrace) && !self.at(&TokenKind::Semi) {
                    // Final expression with no `;` is the block's value.
                    tail = Some(Box::new(e));
                    break;
                }
                needs_sep = !is_block_expr(&e);
                stmts.push(Stmt::Expr(e));
            }

            // A statement separator (grammar spec §3.3 ASI rule). A non-block
            // statement followed on the SAME line by another token with no `;`
            // is a syntax error (`let x = 5 6`), reported rather than silently
            // split into two statements.
            if needs_sep {
                self.expect_stmt_sep();
            } else {
                self.eat(&TokenKind::Semi); // optional `;` after a block
            }

            // Forward-progress guard against a stuck sub-parser.
            if self.pos == before {
                self.bump();
            }
        }

        self.restrict_struct = saved;
        self.expect(&TokenKind::RBrace);
        Block { stmts, tail, span: self.finish(start) }
    }

    fn parse_let(&mut self) -> Stmt {
        self.bump(); // let
        let mutable = self.eat_kw(Keyword::Mut);
        let pat = self.parse_pattern();
        let ty = if self.eat(&TokenKind::Colon) {
            Some(self.parse_type())
        } else {
            None
        };
        let init = if self.eat(&TokenKind::Eq) {
            Some(self.parse_expr())
        } else {
            None
        };
        Stmt::Let(LetStmt { mutable, pat, ty, init })
    }

    /// Consume a statement separator: an explicit `;`, or — per the ASI rule —
    /// a newline before the next token / the end of the block. If the next
    /// token sits on the *same* line with no `;`, the previous statement never
    /// ended: report it instead of silently starting a second statement.
    fn expect_stmt_sep(&mut self) {
        if self.eat(&TokenKind::Semi) {
            return;
        }
        if self.at(&TokenKind::RBrace) || self.is_eof() {
            return;
        }
        if !self.cur().nl_before {
            self.err_expected("`;` or a newline between statements");
            // Recover: skip the stray token so the block keeps parsing.
            self.bump();
        }
    }
}

/// Does this expression end in a `}` block, making it self-delimiting as a
/// statement (so no `;`/newline separator is required after it)?
fn is_block_expr(e: &Expr) -> bool {
    matches!(
        e.kind,
        ExprKind::If(_)
            | ExprKind::While { .. }
            | ExprKind::For { .. }
            | ExprKind::Loop(_)
            | ExprKind::Block(_)
            | ExprKind::Unsafe(_)
            | ExprKind::Match { .. }
    )
}
