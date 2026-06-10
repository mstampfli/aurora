//! Aurora Language Server — stdio JSON-RPC transport.
//!
//! Reads LSP messages (`Content-Length`-framed JSON) from stdin, dispatches them
//! through [`aurora_lsp::Lsp`], and writes framed responses/notifications to
//! stdout. Point an editor's generic LSP client at this binary for `.aur` files
//! to get live diagnostics.

use std::io::{self, BufReader, Read, Write};

use aurora_lsp::Lsp;
use serde_json::Value;

fn main() {
    let mut reader = BufReader::new(io::stdin());
    let mut stdout = io::stdout();
    let mut lsp = Lsp::new();

    while let Some(msg) = read_message(&mut reader) {
        for out in lsp.handle(&msg) {
            write_message(&mut stdout, &out);
        }
        // `exit` after `shutdown` terminates the server.
        if lsp.shutdown && msg.get("method").and_then(Value::as_str) == Some("exit") {
            break;
        }
    }
}

/// Read one `Content-Length`-framed JSON-RPC message. Returns `None` at EOF.
fn read_message(reader: &mut impl Read) -> Option<Value> {
    // Parse headers up to the blank line.
    let mut len: Option<usize> = None;
    let mut line = Vec::new();
    loop {
        line.clear();
        // Read a single header line (terminated by \r\n).
        let mut byte = [0u8; 1];
        loop {
            if reader.read(&mut byte).ok()? == 0 {
                return None; // EOF
            }
            if byte[0] == b'\n' {
                break;
            }
            if byte[0] != b'\r' {
                line.push(byte[0]);
            }
        }
        if line.is_empty() {
            break; // blank line ends the header block
        }
        if let Ok(s) = std::str::from_utf8(&line) {
            if let Some(v) = s.strip_prefix("Content-Length:") {
                len = v.trim().parse().ok();
            }
        }
    }
    let len = len?;
    let mut body = vec![0u8; len];
    reader.read_exact(&mut body).ok()?;
    serde_json::from_slice(&body).ok()
}

fn write_message(out: &mut impl Write, msg: &Value) {
    let body = serde_json::to_vec(msg).unwrap_or_default();
    let _ = write!(out, "Content-Length: {}\r\n\r\n", body.len());
    let _ = out.write_all(&body);
    let _ = out.flush();
}
