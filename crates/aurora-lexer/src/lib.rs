//! Hand-rolled lexer for Aurora (grammar spec §2).
//!
//! Why hand-rolled rather than `logos`: Aurora's block comments are *nestable*
//! (`/* /* */ */`), which a regex tokenizer cannot match, and we want precise
//! control over spans, numeric suffixes, and error recovery. On an unexpected
//! character the lexer reports a diagnostic, emits a [`TokenKind::Error`] token,
//! and keeps going so later phases still see a full token stream.

mod token;
pub use token::*;

use aurora_diag::Diagnostic;
use aurora_span::Span;

/// The result of lexing: the token stream (always ending in [`TokenKind::Eof`])
/// plus any diagnostics produced along the way.
pub struct LexResult {
    pub tokens: Vec<Token>,
    pub diagnostics: Vec<Diagnostic>,
}

/// Tokenize `src`.
pub fn lex(src: &str) -> LexResult {
    Lexer::new(src).run()
}

struct Lexer<'a> {
    src: &'a str,
    /// Byte offset of the next unconsumed char.
    pos: usize,
    tokens: Vec<Token>,
    diags: Vec<Diagnostic>,
}

impl<'a> Lexer<'a> {
    fn new(src: &'a str) -> Lexer<'a> {
        Lexer { src, pos: 0, tokens: Vec::new(), diags: Vec::new() }
    }

    // --- cursor primitives ---------------------------------------------------

    #[inline]
    fn peek(&self) -> Option<char> {
        self.src[self.pos..].chars().next()
    }

    #[inline]
    fn peek_nth(&self, n: usize) -> Option<char> {
        self.src[self.pos..].chars().nth(n)
    }

    #[inline]
    fn bump(&mut self) -> Option<char> {
        let c = self.peek()?;
        self.pos += c.len_utf8();
        Some(c)
    }

    /// Advance while `pred` holds; return the consumed slice.
    fn eat_while(&mut self, mut pred: impl FnMut(char) -> bool) -> &'a str {
        let start = self.pos;
        while let Some(c) = self.peek() {
            if pred(c) {
                self.pos += c.len_utf8();
            } else {
                break;
            }
        }
        &self.src[start..self.pos]
    }

    fn span_from(&self, lo: usize) -> Span {
        Span::new(lo as u32, self.pos as u32)
    }

    fn error(&mut self, span: Span, msg: impl Into<String>, label: impl Into<String>) {
        self.diags
            .push(Diagnostic::error(msg).with_code("E0001").primary(span, label));
    }

    // --- driver --------------------------------------------------------------

    fn run(mut self) -> LexResult {
        loop {
            let nl_before = self.skip_trivia();
            let start = self.pos;
            let Some(c) = self.peek() else {
                self.tokens.push(Token {
                    kind: TokenKind::Eof,
                    span: self.span_from(start),
                    nl_before,
                });
                break;
            };

            let kind = if c == 'r' && self.peek_nth(1) == Some('"') {
                self.lex_raw_string()
            } else if is_ident_start(c) {
                self.lex_ident()
            } else if c.is_ascii_digit() {
                self.lex_number()
            } else if c == '"' {
                self.lex_string()
            } else if c == '\'' {
                self.lex_char()
            } else {
                self.lex_symbol(c)
            };

            self.tokens.push(Token { kind, span: self.span_from(start), nl_before });
        }

        LexResult { tokens: self.tokens, diagnostics: self.diags }
    }

    /// Skip whitespace and comments (line `//`, `///`, and nestable block `/* */`).
    /// Returns whether a newline was crossed (for ASI in the parser).
    fn skip_trivia(&mut self) -> bool {
        let mut nl = false;
        loop {
            match self.peek() {
                // Whitespace, plus a UTF-8 BOM (`\u{FEFF}`) which editors may
                // prepend; treated as insignificant rather than an error.
                Some(c) if c.is_whitespace() || c == '\u{FEFF}' => {
                    if c == '\n' {
                        nl = true;
                    }
                    self.bump();
                }
                Some('/') if self.peek_nth(1) == Some('/') => {
                    self.eat_while(|c| c != '\n');
                }
                Some('/') if self.peek_nth(1) == Some('*') => {
                    self.skip_block_comment();
                }
                _ => break,
            }
        }
        nl
    }

    fn skip_block_comment(&mut self) {
        let start = self.pos;
        self.bump(); // '/'
        self.bump(); // '*'
        let mut depth = 1usize;
        while depth > 0 {
            match self.bump() {
                Some('/') if self.peek() == Some('*') => {
                    self.bump();
                    depth += 1;
                }
                Some('*') if self.peek() == Some('/') => {
                    self.bump();
                    depth -= 1;
                }
                Some(_) => {}
                None => {
                    let span = self.span_from(start);
                    self.error(span, "unterminated block comment", "comment starts here");
                    break;
                }
            }
        }
    }

    // --- identifiers & keywords ----------------------------------------------

    fn lex_ident(&mut self) -> TokenKind {
        let text = self.eat_while(is_ident_continue);
        match Keyword::from_str(text) {
            Some(kw) => TokenKind::Kw(kw),
            None => TokenKind::Ident(text.to_string()),
        }
    }

    // --- numbers -------------------------------------------------------------

    fn lex_number(&mut self) -> TokenKind {
        // Hex / binary prefixes.
        if self.peek() == Some('0') {
            match self.peek_nth(1) {
                Some('x') | Some('X') => {
                    let start = self.pos;
                    self.bump();
                    self.bump();
                    let digits = self.eat_while(|c| c.is_ascii_hexdigit() || c == '_');
                    return self.finish_int(&digits, 16, start);
                }
                Some('b') | Some('B') => {
                    let start = self.pos;
                    self.bump();
                    self.bump();
                    let digits = self.eat_while(|c| c == '0' || c == '1' || c == '_');
                    return self.finish_int(&digits, 2, start);
                }
                _ => {}
            }
        }

        let num_start = self.pos;
        let mut text = self.eat_while(|c| c.is_ascii_digit() || c == '_').to_string();
        let mut is_float = false;

        // Fractional part: only if `.` is followed by a digit (so `1..2` and
        // `1.method()` are not consumed as floats).
        if self.peek() == Some('.') && self.peek_nth(1).is_some_and(|c| c.is_ascii_digit()) {
            is_float = true;
            self.bump();
            text.push('.');
            text.push_str(self.eat_while(|c| c.is_ascii_digit() || c == '_'));
        }

        // Exponent.
        if matches!(self.peek(), Some('e') | Some('E')) {
            is_float = true;
            text.push('e');
            self.bump();
            if matches!(self.peek(), Some('+') | Some('-')) {
                text.push(self.bump().unwrap());
            }
            text.push_str(self.eat_while(|c| c.is_ascii_digit() || c == '_'));
        }

        let suffix = self.eat_suffix();
        let clean: String = text.chars().filter(|&c| c != '_').collect();

        match suffix {
            Some(Suffix::Float(ft)) => self.make_float(&clean, Some(ft), num_start),
            Some(Suffix::Int(_)) if is_float => {
                let span = self.span_from(num_start);
                self.error(
                    span,
                    "integer suffix on a float literal",
                    format!("`{}` cannot have an integer suffix", &clean),
                );
                self.make_float(&clean, None, num_start)
            }
            Some(Suffix::Int(it)) => self.make_int(&clean, 10, Some(it), num_start),
            None if is_float => self.make_float(&clean, None, num_start),
            None => self.make_int(&clean, 10, None, num_start),
        }
    }

    fn finish_int(&mut self, digits: &str, radix: u32, start: usize) -> TokenKind {
        let clean: String = digits.chars().filter(|&c| c != '_').collect();
        if clean.is_empty() {
            let span = self.span_from(start);
            self.error(span, "missing digits in numeric literal", "expected one or more digits");
        }
        let suffix = match self.eat_suffix() {
            Some(Suffix::Int(it)) => Some(it),
            Some(Suffix::Float(_)) => {
                let span = self.span_from(start);
                self.error(span, "float suffix on an integer literal", "use an integer suffix");
                None
            }
            None => None,
        };
        self.make_int(&clean, radix, suffix, start)
    }

    fn make_int(&mut self, clean: &str, radix: u32, suffix: Option<IntTy>, start: usize) -> TokenKind {
        let value = u128::from_str_radix(if clean.is_empty() { "0" } else { clean }, radix)
            .unwrap_or_else(|_| {
                let span = self.span_from(start);
                self.error(span, "integer literal out of range", "does not fit in u128");
                0
            });
        TokenKind::Int { value, suffix }
    }

    fn make_float(&mut self, clean: &str, suffix: Option<FloatTy>, start: usize) -> TokenKind {
        let value = clean.parse::<f64>().unwrap_or_else(|_| {
            let span = self.span_from(start);
            self.error(span, "invalid float literal", "could not parse as a number");
            0.0
        });
        TokenKind::Float { value, suffix }
    }

    /// Consume a numeric suffix if (and only if) the trailing identifier run is a
    /// known suffix; otherwise leave it for the next token.
    fn eat_suffix(&mut self) -> Option<Suffix> {
        let save = self.pos;
        if !self.peek().is_some_and(is_ident_start) {
            return None;
        }
        let run = self.eat_while(is_ident_continue);
        let suffix = match run {
            "i8" => Suffix::Int(IntTy::I8),
            "i16" => Suffix::Int(IntTy::I16),
            "i32" => Suffix::Int(IntTy::I32),
            "i64" => Suffix::Int(IntTy::I64),
            "u8" => Suffix::Int(IntTy::U8),
            "u16" => Suffix::Int(IntTy::U16),
            "u32" => Suffix::Int(IntTy::U32),
            "u64" => Suffix::Int(IntTy::U64),
            "f32" => Suffix::Float(FloatTy::F32),
            "f64" => Suffix::Float(FloatTy::F64),
            _ => {
                self.pos = save; // not a suffix; un-consume
                return None;
            }
        };
        Some(suffix)
    }

    // --- strings & chars -----------------------------------------------------

    fn lex_string(&mut self) -> TokenKind {
        let start = self.pos;
        self.bump(); // opening quote
        let mut value = String::new();
        loop {
            match self.bump() {
                Some('"') => break,
                Some('\\') => {
                    if let Some(c) = self.lex_escape(start) {
                        value.push(c);
                    }
                }
                Some('\n') | None => {
                    let span = self.span_from(start);
                    self.error(span, "unterminated string literal", "string starts here");
                    break;
                }
                Some(c) => value.push(c),
            }
        }
        TokenKind::Str(value)
    }

    fn lex_raw_string(&mut self) -> TokenKind {
        let start = self.pos;
        self.bump(); // 'r'
        self.bump(); // '"'
        let mut value = String::new();
        loop {
            match self.bump() {
                Some('"') => break,
                None => {
                    let span = self.span_from(start);
                    self.error(span, "unterminated raw string literal", "raw string starts here");
                    break;
                }
                Some(c) => value.push(c),
            }
        }
        TokenKind::Str(value)
    }

    fn lex_char(&mut self) -> TokenKind {
        let start = self.pos;
        self.bump(); // opening '
        let value = match self.bump() {
            Some('\\') => self.lex_escape(start).unwrap_or('\u{FFFD}'),
            Some('\'') => {
                let span = self.span_from(start);
                self.error(span, "empty character literal", "expected a character");
                return TokenKind::Char('\u{FFFD}');
            }
            Some(c) => c,
            None => {
                let span = self.span_from(start);
                self.error(span, "unterminated character literal", "expected a character");
                return TokenKind::Char('\u{FFFD}');
            }
        };
        if self.peek() == Some('\'') {
            self.bump();
        } else {
            let span = self.span_from(start);
            self.error(span, "unterminated character literal", "expected closing `'`");
        }
        TokenKind::Char(value)
    }

    /// Lex the body of an escape (the backslash is already consumed).
    fn lex_escape(&mut self, lit_start: usize) -> Option<char> {
        match self.bump() {
            Some('n') => Some('\n'),
            Some('r') => Some('\r'),
            Some('t') => Some('\t'),
            Some('0') => Some('\0'),
            Some('\\') => Some('\\'),
            Some('"') => Some('"'),
            Some('\'') => Some('\''),
            Some('x') => {
                let hi = self.bump();
                let lo = self.bump();
                match (hi, lo) {
                    (Some(h), Some(l)) => {
                        let code = u8::from_str_radix(&format!("{h}{l}"), 16).ok();
                        match code {
                            Some(byte) if byte <= 0x7F => Some(byte as char),
                            _ => {
                                let span = self.span_from(lit_start);
                                self.error(span, "invalid `\\x` escape", "expected two hex digits 00-7F");
                                None
                            }
                        }
                    }
                    _ => {
                        let span = self.span_from(lit_start);
                        self.error(span, "invalid `\\x` escape", "expected two hex digits");
                        None
                    }
                }
            }
            Some('u') => {
                if self.peek() != Some('{') {
                    let span = self.span_from(lit_start);
                    self.error(span, "invalid `\\u` escape", "expected `{`");
                    return None;
                }
                self.bump(); // {
                let hex = self.eat_while(|c| c != '}');
                let ok = self.peek() == Some('}');
                if ok {
                    self.bump();
                }
                let parsed = u32::from_str_radix(hex.trim(), 16).ok().and_then(char::from_u32);
                match (ok, parsed) {
                    (true, Some(c)) => Some(c),
                    _ => {
                        let span = self.span_from(lit_start);
                        self.error(span, "invalid unicode escape", "expected `\\u{NNNN}`");
                        None
                    }
                }
            }
            other => {
                let span = self.span_from(lit_start);
                let shown = other.map(|c| c.to_string()).unwrap_or_default();
                self.error(span, format!("unknown escape `\\{shown}`"), "not a valid escape");
                other
            }
        }
    }

    // --- punctuation & operators ---------------------------------------------

    fn lex_symbol(&mut self, c: char) -> TokenKind {
        use TokenKind::*;
        // Helper: consume `c` then optionally a second char.
        match c {
            '(' => self.one(LParen),
            ')' => self.one(RParen),
            '{' => self.one(LBrace),
            '}' => self.one(RBrace),
            '[' => self.one(LBracket),
            ']' => self.one(RBracket),
            ',' => self.one(Comma),
            ';' => self.one(Semi),
            '@' => self.one(At),
            '#' => self.one(Hash),
            '~' => self.one(Tilde),
            '?' => self.one(Question),
            ':' => {
                self.bump();
                if self.peek() == Some(':') {
                    self.bump();
                    ColonColon
                } else {
                    Colon
                }
            }
            '.' => {
                self.bump();
                if self.peek() == Some('.') {
                    self.bump();
                    if self.peek() == Some('=') {
                        self.bump();
                        DotDotEq
                    } else {
                        DotDot
                    }
                } else {
                    Dot
                }
            }
            '-' => {
                self.bump();
                match self.peek() {
                    Some('>') => {
                        self.bump();
                        Arrow
                    }
                    Some('=') => {
                        self.bump();
                        MinusEq
                    }
                    _ => Minus,
                }
            }
            '=' => {
                self.bump();
                match self.peek() {
                    Some('=') => {
                        self.bump();
                        EqEq
                    }
                    Some('>') => {
                        self.bump();
                        FatArrow
                    }
                    _ => Eq,
                }
            }
            '!' => {
                self.bump();
                if self.peek() == Some('=') {
                    self.bump();
                    BangEq
                } else {
                    Bang
                }
            }
            '<' => self.with_eq(Lt, Le),
            '>' => self.with_eq(Gt, Ge),
            '+' => self.with_eq(Plus, PlusEq),
            '*' => self.with_eq(Star, StarEq),
            '/' => self.with_eq(Slash, SlashEq),
            '%' => self.with_eq(Percent, PercentEq),
            '^' => self.with_eq(Caret, CaretEq),
            '&' => self.with_eq(Amp, AmpEq),
            '|' => {
                self.bump();
                match self.peek() {
                    Some('>') => {
                        self.bump();
                        PipeGt
                    }
                    Some('=') => {
                        self.bump();
                        PipeEq
                    }
                    _ => Pipe,
                }
            }
            other => {
                self.bump();
                let span = self.span_from(self.pos - other.len_utf8());
                self.error(
                    span,
                    format!("unexpected character `{other}`"),
                    "not valid Aurora syntax",
                );
                Error
            }
        }
    }

    /// Consume one char and return `kind`.
    fn one(&mut self, kind: TokenKind) -> TokenKind {
        self.bump();
        kind
    }

    /// Consume the current char; return `with` if followed by `=`, else `base`.
    fn with_eq(&mut self, base: TokenKind, with: TokenKind) -> TokenKind {
        self.bump();
        if self.peek() == Some('=') {
            self.bump();
            with
        } else {
            base
        }
    }
}

enum Suffix {
    Int(IntTy),
    Float(FloatTy),
}

fn is_ident_start(c: char) -> bool {
    c.is_ascii_alphabetic() || c == '_'
}

fn is_ident_continue(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '_'
}
