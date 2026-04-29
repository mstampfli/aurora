//! Behavioural tests for the lexer. These assert on concrete token kinds so
//! they pass deterministically without a snapshot review step; richer snapshot
//! tests arrive once the parser produces an AST worth snapshotting.

use aurora_lexer::{lex, FloatTy, IntTy, Keyword, TokenKind};

/// Lex `src` and return the token kinds, dropping the trailing `Eof`.
fn kinds(src: &str) -> Vec<TokenKind> {
    let mut toks: Vec<TokenKind> = lex(src).tokens.into_iter().map(|t| t.kind).collect();
    assert_eq!(toks.last(), Some(&TokenKind::Eof), "stream must end in Eof");
    toks.pop();
    toks
}

fn no_errors(src: &str) {
    let r = lex(src);
    assert!(
        r.diagnostics.is_empty(),
        "unexpected diagnostics for {src:?}: {:?}",
        r.diagnostics.iter().map(|d| &d.message).collect::<Vec<_>>()
    );
}

#[test]
fn keywords_vs_identifiers() {
    use Keyword::*;
    use TokenKind::*;
    assert_eq!(
        kinds("let mut foo Component component"),
        vec![
            Kw(Let),
            Kw(Mut),
            Ident("foo".into()),
            Ident("Component".into()), // capitalized => not the keyword
            Kw(Component),
        ]
    );
    no_errors("let mut foo Component component");
}

#[test]
fn multi_char_operators() {
    use TokenKind::*;
    assert_eq!(
        kinds(":: -> => .. ..= |> += == != <= >= &"),
        vec![
            ColonColon, Arrow, FatArrow, DotDot, DotDotEq, PipeGt, PlusEq, EqEq, BangEq, Le, Ge,
            Amp,
        ]
    );
    no_errors(":: -> => .. ..= |> += == != <= >= &");
}

#[test]
fn region_and_attr_sigils() {
    use Keyword::*;
    use TokenKind::*;
    // `#frame`, `@vertex`, `~Mesh`, `&mut`
    assert_eq!(
        kinds("#frame @vertex ~Mesh &mut"),
        vec![
            Hash,
            Ident("frame".into()),
            At,
            Ident("vertex".into()),
            Tilde,
            Ident("Mesh".into()),
            Amp,
            Kw(Mut),
        ]
    );
}

#[test]
fn integer_literals_and_suffixes() {
    use TokenKind::*;
    assert_eq!(
        kinds("0 42 1_000 0xFF 0b1010 7u8 64i32"),
        vec![
            Int { value: 0, suffix: None },
            Int { value: 42, suffix: None },
            Int { value: 1000, suffix: None },
            Int { value: 255, suffix: None },
            Int { value: 10, suffix: None },
            Int { value: 7, suffix: Some(IntTy::U8) },
            Int { value: 64, suffix: Some(IntTy::I32) },
        ]
    );
    no_errors("0 42 1_000 0xFF 0b1010 7u8 64i32");
}

#[test]
fn float_literals() {
    use TokenKind::*;
    assert_eq!(
        kinds("1.0 3.14 2.0e3 1.5f32 1e10"),
        vec![
            Float { value: 1.0, suffix: None },
            Float { value: 3.14, suffix: None },
            Float { value: 2000.0, suffix: None },
            Float { value: 1.5, suffix: Some(FloatTy::F32) },
            Float { value: 1e10, suffix: None },
        ]
    );
    no_errors("1.0 3.14 2.0e3 1.5f32 1e10");
}

#[test]
fn range_is_not_a_float() {
    // `1..2` must lex as Int DotDot Int, not as a malformed float.
    use TokenKind::*;
    assert_eq!(
        kinds("1..2"),
        vec![Int { value: 1, suffix: None }, DotDot, Int { value: 2, suffix: None }]
    );
    // `1..=2` likewise.
    assert_eq!(
        kinds("0..=9"),
        vec![Int { value: 0, suffix: None }, DotDotEq, Int { value: 9, suffix: None }]
    );
}

#[test]
fn method_call_on_int_is_not_a_float() {
    use TokenKind::*;
    assert_eq!(
        kinds("1.foo"),
        vec![Int { value: 1, suffix: None }, Dot, Ident("foo".into())]
    );
}

#[test]
fn strings_with_escapes() {
    use TokenKind::*;
    assert_eq!(kinds(r#""hello\nworld""#), vec![Str("hello\nworld".into())]);
    assert_eq!(kinds(r#""tab\there""#), vec![Str("tab\there".into())]);
    assert_eq!(kinds(r#""quote\"end""#), vec![Str("quote\"end".into())]);
    assert_eq!(kinds(r#""\u{41}""#), vec![Str("A".into())]);
    no_errors(r#""hello\nworld" "\u{41}""#);
}

#[test]
fn char_literals() {
    use TokenKind::*;
    assert_eq!(kinds("'a' '\\n' '\\''"), vec![Char('a'), Char('\n'), Char('\'')]);
}

#[test]
fn nested_block_comments() {
    use TokenKind::*;
    // The nested `/* ... */` must not close the outer comment early.
    assert_eq!(
        kinds("let /* outer /* inner */ still */ x"),
        vec![Kw(Keyword::Let), Ident("x".into())]
    );
    no_errors("let /* outer /* inner */ still */ x");
}

#[test]
fn line_and_doc_comments_are_skipped() {
    use TokenKind::*;
    assert_eq!(
        kinds("a // comment\n/// doc\nb"),
        vec![Ident("a".into()), Ident("b".into())]
    );
}

#[test]
fn unexpected_char_recovers() {
    let r = lex("let `x");
    assert_eq!(r.diagnostics.len(), 1);
    assert!(r.diagnostics[0].message.contains("unexpected character"));
    // Lexing continues past the bad char: `let`, Error, `x`, Eof.
    let names: Vec<_> = r.tokens.iter().map(|t| &t.kind).collect();
    assert!(names.contains(&&TokenKind::Error));
    assert!(names.contains(&&TokenKind::Ident("x".into())));
}

#[test]
fn unterminated_string_reports_once() {
    let r = lex("\"oops");
    assert_eq!(r.diagnostics.len(), 1);
    assert!(r.diagnostics[0].message.contains("unterminated string"));
}

#[test]
fn spans_are_byte_accurate() {
    let toks = lex("let x").tokens;
    assert_eq!(toks[0].span.lo, 0);
    assert_eq!(toks[0].span.hi, 3); // "let"
    assert_eq!(toks[1].span.lo, 4);
    assert_eq!(toks[1].span.hi, 5); // "x"
}
