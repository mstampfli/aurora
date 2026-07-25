//! File-based modules: resolve a bodiless `mod NAME;` declaration by loading
//! `NAME.aur` and appending it to the source as an inline `mod NAME { .. }`
//! block, which the flattener (`flatten.rs`) then lowers to `NAME::`-prefixed
//! top-level items. Reusing the inline-module path is deliberate: typecheck, the
//! JIT and the AOT backend all see file-module items through the one mechanism
//! they already understand, so there is no second path that can drift.
//!
//! Resolution rule (spec: `docs/01-grammar-and-types.md` §3.1):
//!
//! * `mod NAME;` loads `NAME.aur` from the directory of the file that declares
//!   it. A loaded file may declare its own file modules, which resolve against
//!   that file's own directory.
//! * One file is one module: file modules form a **flat** namespace and each is
//!   loaded **at most once**, under the single-segment prefix `NAME::`. Declaring
//!   an already-loaded module again (a diamond, or two files that declare each
//!   other) is a no-op, so a cyclic module graph terminates instead of hanging.
//!   Because a name resolves against the declaring file's directory and a name
//!   holds no separators, every module of one program comes from that program's
//!   own directory: name and file are one-to-one, so loading by name is exact.
//! * The entry file is the root module, so it is never re-loaded under a prefix.
//!   It does not need to be: a module body resolves a name it does not define
//!   itself against the top level, so file modules can call the root file's items
//!   (and the standard prelude) unqualified.
//! * A missing file is a hard error naming the path that was looked for.
//! * `mod a::b;` is rejected by the parser (`item.rs`); directory modules
//!   (`NAME/mod.aur`) are not supported.
//!
//! Expansion only ever **appends**, so every byte offset in the declaring file is
//! unchanged and its spans stay valid. The offset each loaded file lands at is
//! tracked, so a diagnostic reported against a loaded file also carries a span
//! into the expanded text.

use std::collections::HashSet;
use std::ops::Range;
use std::path::{Path, PathBuf};

use aurora_diag::Diagnostic;
use aurora_lexer::{lex, Keyword, TokenKind};
use aurora_span::Span;

/// Extension of an Aurora source file.
const EXT: &str = "aur";

/// A file whose text is already in the output buffer but whose own `mod`
/// declarations still have to be resolved.
struct Pending {
    /// Byte range of the file's text within the output buffer.
    range: Range<usize>,
    /// Directory that this file's `mod NAME;` declarations resolve against.
    dir: PathBuf,
}

/// One bodiless `mod NAME;` declaration and its span within its own file.
struct ModDecl {
    name: String,
    span: Span,
}

/// Expand every bodiless `mod NAME;` in `src` (and, recursively, in the files it
/// pulls in) into an appended inline `mod NAME { .. }` block.
///
/// `path` is the file `src` came from: it supplies the directory to resolve
/// against and the root module's identity, and is not read again. Returns the
/// expanded source plus any resolution errors, whose spans are byte offsets into
/// that expanded source.
pub fn load_file_modules(src: &str, path: &Path) -> (String, Vec<Diagnostic>) {
    let mut out = String::from(src);
    let mut diags = Vec::new();
    // Names already pulled in. This is the load-once guard: a second declaration
    // of the same module is a no-op, which is what makes diamonds and cyclic
    // declarations terminate instead of recursing forever.
    let mut loaded: HashSet<String> = HashSet::new();
    // The entry file is the program's root module; loading it again under its own
    // name would give every item it defines a second, distinct name.
    if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
        loaded.insert(stem.to_string());
    }
    let mut queue = vec![Pending { range: 0..src.len(), dir: parent_dir(path) }];

    while let Some(p) = queue.pop() {
        // Collected up front: the loop body appends to `out`.
        let decls = bodiless_mods(&out[p.range.clone()]);
        for decl in decls {
            if !loaded.insert(decl.name.clone()) {
                continue; // loaded once already: diamonds and cycles stop here
            }
            // Lift the declaration's span into the expanded buffer.
            let base = p.range.start as u32;
            let span = Span::new(decl.span.lo + base, decl.span.hi + base);
            // A module name is an identifier, so it can never contain a path
            // separator or `..`: the join stays inside `p.dir` by construction.
            let target = p.dir.join(format!("{}.{EXT}", decl.name));

            let text = match std::fs::read_to_string(&target) {
                Ok(t) => t,
                Err(e) => {
                    let mut d = Diagnostic::error(format!(
                        "cannot find file for module `{}`",
                        decl.name
                    ))
                    .with_code("E0110")
                    .primary(span, "no file for this module")
                    .note(format!("looked for `{}`: {e}", target.display()));
                    if p.dir.join(&decl.name).is_dir() {
                        d = d.note(format!(
                            "a directory `{}` exists, but directory modules are not \
                             supported: put the items in `{}.{EXT}`",
                            decl.name, decl.name
                        ));
                    }
                    diags.push(d);
                    continue;
                }
            };

            // Append the loaded file as an inline module. Appending (never
            // splicing) leaves every offset before it untouched.
            out.push_str("\n// file module `");
            out.push_str(&decl.name);
            out.push_str("`\nmod ");
            out.push_str(&decl.name);
            out.push_str(" {\n");
            let start = out.len();
            out.push_str(&text);
            let end = out.len();
            out.push_str("\n}\n");
            queue.push(Pending { range: start..end, dir: parent_dir(&target) });
        }
    }
    (out, diags)
}

/// Directory a file's `mod` declarations resolve against (`""` means the CWD,
/// which is what a bare `main.aur` was read relative to).
fn parent_dir(file: &Path) -> PathBuf {
    file.parent().unwrap_or(Path::new("")).to_path_buf()
}

/// Find every bodiless `mod NAME;` item declaration in `src`.
///
/// Token-based, so `mod` inside a string or a comment is never mistaken for a
/// declaration. `mod` is a reserved keyword, so every `mod` token starts a module
/// declaration: the ones with a `{` body are already inline (the flattener owns
/// them) and the ones with a `::` path are rejected by the parser, so both are
/// skipped here. Scanning does descend into inline module bodies, so a bodiless
/// declaration nested in one is still resolved.
fn bodiless_mods(src: &str) -> Vec<ModDecl> {
    let toks = lex(src).tokens;
    let mut out = Vec::new();
    let mut i = 0;
    while i < toks.len() {
        if !matches!(&toks[i].kind, TokenKind::Kw(Keyword::Mod)) {
            i += 1;
            continue;
        }
        let start = toks[i].span;
        let Some(TokenKind::Ident(name)) = toks.get(i + 1).map(|t| &t.kind) else {
            i += 1;
            continue; // malformed `mod`: the parser reports it
        };
        let name = name.clone();
        let mut end = toks[i + 1].span;
        i += 2;
        match toks.get(i).map(|t| &t.kind) {
            // Inline module: the flattener already has the items.
            Some(TokenKind::LBrace) => continue,
            // `mod a::b;`: unsupported, reported by the parser.
            Some(TokenKind::ColonColon) => continue,
            Some(TokenKind::Semi) => {
                end = toks[i].span;
                i += 1;
            }
            // No terminator (ASI): the name ends the declaration.
            _ => {}
        }
        out.push(ModDecl { name, span: start.to(end) });
    }
    out
}
