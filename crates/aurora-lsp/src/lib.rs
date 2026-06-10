//! The Aurora Language Server core: pure, testable LSP request dispatch.
//!
//! [`Lsp::handle`] takes one decoded JSON-RPC message and returns the messages
//! to send back (responses and `publishDiagnostics` notifications). It runs the
//! compiler's own `parse → check → typeck` pipeline so editors see exactly the
//! diagnostics `aurorac check` would. The stdio framing lives in the binary.

use std::collections::HashMap;

use aurora_diag::{Diagnostic, Severity};
use aurora_span::SourceFile;
use serde_json::{json, Value};

/// Language-server state: the text of every open document, keyed by URI.
#[derive(Default)]
pub struct Lsp {
    docs: HashMap<String, String>,
    pub shutdown: bool,
}

impl Lsp {
    pub fn new() -> Lsp {
        Lsp::default()
    }

    /// Handle one incoming JSON-RPC message; return outgoing messages.
    pub fn handle(&mut self, msg: &Value) -> Vec<Value> {
        let method = msg.get("method").and_then(Value::as_str).unwrap_or("");
        let id = msg.get("id").cloned();
        match method {
            "initialize" => vec![response(
                id,
                json!({
                    "capabilities": {
                        // Full-text sync: the client sends the whole document on change.
                        "textDocumentSync": 1,
                        "completionProvider": { "triggerCharacters": [".", ":"] },
                    },
                    "serverInfo": { "name": "aurora-lsp", "version": env!("CARGO_PKG_VERSION") },
                }),
            )],
            "textDocument/didOpen" => {
                let uri = msg.pointer("/params/textDocument/uri").and_then(Value::as_str);
                let text = msg.pointer("/params/textDocument/text").and_then(Value::as_str);
                if let (Some(uri), Some(text)) = (uri, text) {
                    self.docs.insert(uri.to_string(), text.to_string());
                    return vec![self.diagnostics_notification(uri)];
                }
                vec![]
            }
            "textDocument/didChange" => {
                let uri = msg.pointer("/params/textDocument/uri").and_then(Value::as_str);
                // Full sync: take the last content change's full text.
                let text = msg
                    .pointer("/params/contentChanges")
                    .and_then(Value::as_array)
                    .and_then(|a| a.last())
                    .and_then(|c| c.get("text"))
                    .and_then(Value::as_str);
                if let (Some(uri), Some(text)) = (uri, text) {
                    self.docs.insert(uri.to_string(), text.to_string());
                    return vec![self.diagnostics_notification(uri)];
                }
                vec![]
            }
            "textDocument/didClose" => {
                if let Some(uri) = msg.pointer("/params/textDocument/uri").and_then(Value::as_str) {
                    self.docs.remove(uri);
                    // Clear diagnostics for the closed file.
                    return vec![publish(uri, vec![])];
                }
                vec![]
            }
            "textDocument/completion" => {
                let uri = msg.pointer("/params/textDocument/uri").and_then(Value::as_str);
                let src = uri.and_then(|u| self.docs.get(u)).cloned().unwrap_or_default();
                vec![response(id, json!({ "isIncomplete": false, "items": completions(&src) }))]
            }
            "shutdown" => {
                self.shutdown = true;
                vec![response(id, Value::Null)]
            }
            "initialized" | "exit" => vec![],
            _ => {
                // Respond to unknown *requests* (those with an id) so clients
                // don't hang; ignore unknown notifications.
                if id.is_some() {
                    vec![response(id, Value::Null)]
                } else {
                    vec![]
                }
            }
        }
    }

    /// Compute and wrap diagnostics for an open document as a notification.
    fn diagnostics_notification(&self, uri: &str) -> Value {
        let src = self.docs.get(uri).cloned().unwrap_or_default();
        publish(uri, compute_diagnostics(&src))
    }
}

/// Run the compiler pipeline over `src` and convert each diagnostic to LSP form.
pub fn compute_diagnostics(src: &str) -> Vec<Value> {
    let file = SourceFile::new("<lsp>", src.to_string());
    let (module, mut diags) = aurora_parser::parse_str(&file.src);
    diags.extend(aurora_check::check(&module));
    diags.extend(aurora_typeck::check_types(&module));
    diags.iter().map(|d| to_lsp(&file, d)).collect()
}

/// Completion items for a document: language keywords, builtins, and the
/// fn/struct/enum/component names defined in the current source.
pub fn completions(src: &str) -> Vec<Value> {
    // LSP CompletionItemKind: Function=3, Keyword=14, Struct=22, Enum=13.
    let mut items = Vec::new();
    let mut push = |label: &str, kind: i64| {
        items.push(json!({ "label": label, "kind": kind }));
    };
    for kw in [
        "fn", "let", "mut", "struct", "enum", "trait", "impl", "component", "system", "match",
        "if", "else", "while", "for", "in", "return", "break", "continue", "true", "false", "mod",
    ] {
        push(kw, 14);
    }
    for b in [
        "println", "print", "sqrt", "sin", "cos", "abs", "min", "max", "clamp", "len", "str",
        "spawn", "despawn", "run_systems", "entity_count", "framebuffer", "clear", "pixel",
        "triangle", "save_ppm", "play_note", "play_sound", "window_open", "window_present",
        "key_down", "mouse_x", "mouse_y", "mouse_down", "gpu_render", "gpu_compute", "net_bind",
        "net_send", "net_recv", "load_ppm", "scene_save", "scene_load", "frame_reset",
    ] {
        push(b, 3);
    }
    // User-defined symbols from the document.
    let (module, _) = aurora_parser::parse_str(src);
    for item in &module.items {
        match &item.kind {
            aurora_parser::ast::ItemKind::Fn(f) => push(&f.name.name, 3),
            aurora_parser::ast::ItemKind::Struct(s)
            | aurora_parser::ast::ItemKind::Component(s) => push(&s.name.name, 22),
            aurora_parser::ast::ItemKind::Enum(e) => push(&e.name.name, 13),
            _ => {}
        }
    }
    items
}

fn to_lsp(file: &SourceFile, d: &Diagnostic) -> Value {
    // Prefer the primary label's span; fall back to the first label or origin.
    let span = d
        .labels
        .iter()
        .find(|l| l.primary)
        .or_else(|| d.labels.first())
        .map(|l| l.span);
    let (start, end) = match span {
        Some(s) => {
            let lo = file.line_col(s.lo);
            let hi = file.line_col(s.hi);
            (pos(lo.line, lo.col), pos(hi.line, hi.col))
        }
        None => (pos(1, 1), pos(1, 1)),
    };
    let severity = match d.severity {
        Severity::Error => 1,
        Severity::Warning => 2,
        Severity::Note => 3,
        Severity::Help => 4,
    };
    let mut obj = json!({
        "range": { "start": start, "end": end },
        "severity": severity,
        "message": d.message,
        "source": "aurora",
    });
    if let Some(code) = &d.code {
        obj["code"] = json!(code);
    }
    obj
}

/// Convert a 1-based line/col (compiler convention) to a 0-based LSP position.
fn pos(line: u32, col: u32) -> Value {
    json!({ "line": line.saturating_sub(1), "character": col.saturating_sub(1) })
}

fn response(id: Option<Value>, result: Value) -> Value {
    json!({ "jsonrpc": "2.0", "id": id.unwrap_or(Value::Null), "result": result })
}

fn publish(uri: &str, diagnostics: Vec<Value>) -> Value {
    json!({
        "jsonrpc": "2.0",
        "method": "textDocument/publishDiagnostics",
        "params": { "uri": uri, "diagnostics": diagnostics },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clean_source_has_no_diagnostics() {
        let d = compute_diagnostics("fn add(a: i32, b: i32) -> i32 { a + b }");
        assert!(d.is_empty(), "clean source should produce no diagnostics, got {d:?}");
    }

    #[test]
    fn type_error_becomes_an_lsp_diagnostic_with_range() {
        let d = compute_diagnostics("fn f() -> bool { 1 }");
        assert_eq!(d.len(), 1, "expected one diagnostic, got {d:?}");
        assert_eq!(d[0]["severity"], json!(1), "type errors are LSP severity 1 (Error)");
        assert!(d[0]["message"].as_str().unwrap().contains("return value"));
        // The range is well-formed and 0-based.
        assert!(d[0]["range"]["start"]["line"].is_number());
        assert!(d[0]["range"]["end"]["character"].is_number());
    }

    #[test]
    fn initialize_advertises_full_sync() {
        let mut lsp = Lsp::new();
        let out = lsp.handle(&json!({"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}));
        assert_eq!(out.len(), 1);
        assert_eq!(out[0]["result"]["capabilities"]["textDocumentSync"], json!(1));
        assert_eq!(out[0]["id"], json!(1));
    }

    #[test]
    fn did_open_publishes_diagnostics_for_the_doc() {
        let mut lsp = Lsp::new();
        let out = lsp.handle(&json!({
            "jsonrpc": "2.0",
            "method": "textDocument/didOpen",
            "params": { "textDocument": { "uri": "file:///a.aur", "text": "fn f() -> bool { 1 }" } }
        }));
        assert_eq!(out.len(), 1);
        assert_eq!(out[0]["method"], json!("textDocument/publishDiagnostics"));
        assert_eq!(out[0]["params"]["uri"], json!("file:///a.aur"));
        let diags = out[0]["params"]["diagnostics"].as_array().unwrap();
        assert_eq!(diags.len(), 1, "the type error should be reported");
    }

    #[test]
    fn did_change_reflects_new_text() {
        let mut lsp = Lsp::new();
        lsp.handle(&json!({
            "jsonrpc": "2.0", "method": "textDocument/didOpen",
            "params": { "textDocument": { "uri": "file:///a.aur", "text": "fn f() -> bool { 1 }" } }
        }));
        // Fix the error via didChange; diagnostics should clear.
        let out = lsp.handle(&json!({
            "jsonrpc": "2.0", "method": "textDocument/didChange",
            "params": {
                "textDocument": { "uri": "file:///a.aur" },
                "contentChanges": [ { "text": "fn f() -> bool { true }" } ]
            }
        }));
        let diags = out[0]["params"]["diagnostics"].as_array().unwrap();
        assert!(diags.is_empty(), "corrected source should clear diagnostics, got {diags:?}");
    }

    #[test]
    fn completion_offers_keywords_builtins_and_user_symbols() {
        let items = completions("struct Player { hp: i64 }\nfn heal(p: Player) -> i64 { p.hp }");
        let labels: Vec<&str> =
            items.iter().filter_map(|i| i["label"].as_str()).collect();
        assert!(labels.contains(&"fn"), "keywords: {labels:?}");
        assert!(labels.contains(&"println"), "builtins: {labels:?}");
        assert!(labels.contains(&"Player"), "user struct: {labels:?}");
        assert!(labels.contains(&"heal"), "user fn: {labels:?}");
    }

    #[test]
    fn completion_request_returns_items() {
        let mut lsp = Lsp::new();
        lsp.handle(&json!({
            "jsonrpc": "2.0", "method": "textDocument/didOpen",
            "params": { "textDocument": { "uri": "file:///a.aur", "text": "fn go() -> i64 { 1 }" } }
        }));
        let out = lsp.handle(&json!({
            "jsonrpc": "2.0", "id": 5, "method": "textDocument/completion",
            "params": { "textDocument": { "uri": "file:///a.aur" } }
        }));
        let items = out[0]["result"]["items"].as_array().unwrap();
        assert!(items.iter().any(|i| i["label"] == "go"), "should offer the user fn `go`");
    }

    #[test]
    fn shutdown_sets_flag_and_responds() {
        let mut lsp = Lsp::new();
        let out = lsp.handle(&json!({"jsonrpc":"2.0","id":9,"method":"shutdown"}));
        assert!(lsp.shutdown);
        assert_eq!(out[0]["id"], json!(9));
    }
}
