//! Diagnostics: the user-facing errors, warnings, and notes the toolchain emits.
//!
//! Diagnostics are a first-class output of every compiler phase. A
//! [`Diagnostic`] is built with a fluent API and rendered against a
//! [`SourceFile`] into an `rustc`/Elm-style snippet with a caret underline.
//! The renderer is deterministic so it can back snapshot tests.

use std::fmt::Write as _;

use aurora_span::{SourceFile, Span};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Severity {
    Error,
    Warning,
    Note,
    Help,
}

impl Severity {
    fn header(self) -> &'static str {
        match self {
            Severity::Error => "error",
            Severity::Warning => "warning",
            Severity::Note => "note",
            Severity::Help => "help",
        }
    }
}

/// A span annotation within a diagnostic. The `primary` label points at the
/// root cause; `secondary` labels add supporting context.
#[derive(Clone, Debug)]
pub struct Label {
    pub span: Span,
    pub message: String,
    pub primary: bool,
}

#[derive(Clone, Debug)]
pub struct Diagnostic {
    pub severity: Severity,
    /// Stable diagnostic code, e.g. `"E0001"`.
    pub code: Option<String>,
    pub message: String,
    pub labels: Vec<Label>,
    pub notes: Vec<String>,
}

impl Diagnostic {
    pub fn new(severity: Severity, message: impl Into<String>) -> Diagnostic {
        Diagnostic {
            severity,
            code: None,
            message: message.into(),
            labels: Vec::new(),
            notes: Vec::new(),
        }
    }

    pub fn error(message: impl Into<String>) -> Diagnostic {
        Diagnostic::new(Severity::Error, message)
    }

    pub fn warning(message: impl Into<String>) -> Diagnostic {
        Diagnostic::new(Severity::Warning, message)
    }

    pub fn with_code(mut self, code: impl Into<String>) -> Diagnostic {
        self.code = Some(code.into());
        self
    }

    /// Attach the primary label (the caret points here).
    pub fn primary(mut self, span: Span, message: impl Into<String>) -> Diagnostic {
        self.labels.push(Label { span, message: message.into(), primary: true });
        self
    }

    /// Attach a supporting label.
    pub fn secondary(mut self, span: Span, message: impl Into<String>) -> Diagnostic {
        self.labels.push(Label { span, message: message.into(), primary: false });
        self
    }

    pub fn note(mut self, note: impl Into<String>) -> Diagnostic {
        self.notes.push(note.into());
        self
    }

    pub fn is_error(&self) -> bool {
        self.severity == Severity::Error
    }

    /// Render this diagnostic against `file` into a multi-line string.
    pub fn render(&self, file: &SourceFile) -> String {
        let mut out = String::new();

        // Header: `error[E0001]: message`
        match &self.code {
            Some(code) => write!(out, "{}[{}]", self.severity.header(), code).unwrap(),
            None => write!(out, "{}", self.severity.header()).unwrap(),
        }
        writeln!(out, ": {}", self.message).unwrap();

        // Determine gutter width from the largest line number we'll print.
        let max_line = self
            .labels
            .iter()
            .map(|l| file.line_col(l.span.lo).line)
            .max()
            .unwrap_or(1);
        let gutter = max_line.to_string().len();

        // `--> name:line:col` for the primary label (or first label).
        let anchor = self
            .labels
            .iter()
            .find(|l| l.primary)
            .or_else(|| self.labels.first());
        if let Some(l) = anchor {
            let lc = file.line_col(l.span.lo);
            writeln!(
                out,
                "{:width$}--> {}:{}:{}",
                "",
                file.name,
                lc.line,
                lc.col,
                width = gutter + 1
            )
            .unwrap();
        }

        // One snippet per label, in source order.
        let mut labels: Vec<&Label> = self.labels.iter().collect();
        labels.sort_by_key(|l| (l.span.lo, l.span.hi));
        for l in labels {
            render_snippet(&mut out, file, l, gutter);
        }

        for note in &self.notes {
            writeln!(out, "{:width$}= note: {}", "", note, width = gutter + 1).unwrap();
        }

        out
    }
}

fn render_snippet(out: &mut String, file: &SourceFile, label: &Label, gutter: usize) {
    let start = file.line_col(label.span.lo);
    let line_text = file.line_text(start.line);

    // ` | `
    writeln!(out, "{:width$} |", "", width = gutter).unwrap();
    // `LL | source text`
    writeln!(out, "{:>width$} | {}", start.line, line_text, width = gutter).unwrap();

    // Caret line. Column is 1-based; underline length is the span length on this
    // line (clamped so a multi-line span just underlines to end of line).
    let caret_col = start.col.saturating_sub(1) as usize;
    let line_chars = line_text.chars().count();
    let span_chars = file
        .slice(label.span)
        .chars()
        .take_while(|&c| c != '\n')
        .count()
        .max(1);
    let underline_len = span_chars.min(line_chars.saturating_sub(caret_col)).max(1);
    let caret = if label.primary { '^' } else { '-' };

    let mut caret_line = String::new();
    for _ in 0..caret_col {
        caret_line.push(' ');
    }
    for _ in 0..underline_len {
        caret_line.push(caret);
    }
    if label.message.is_empty() {
        writeln!(out, "{:width$} | {}", "", caret_line, width = gutter).unwrap();
    } else {
        writeln!(out, "{:width$} | {} {}", "", caret_line, label.message, width = gutter).unwrap();
    }
}

/// A small accumulator phases push diagnostics into.
#[derive(Default, Debug)]
pub struct Diagnostics {
    pub items: Vec<Diagnostic>,
}

impl Diagnostics {
    pub fn new() -> Diagnostics {
        Diagnostics::default()
    }

    pub fn push(&mut self, d: Diagnostic) {
        self.items.push(d);
    }

    pub fn has_errors(&self) -> bool {
        self.items.iter().any(Diagnostic::is_error)
    }

    pub fn error_count(&self) -> usize {
        self.items.iter().filter(|d| d.is_error()).count()
    }

    pub fn render_all(&self, file: &SourceFile) -> String {
        self.items
            .iter()
            .map(|d| d.render(file))
            .collect::<Vec<_>>()
            .join("\n")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn renders_caret_under_span() {
        let file = SourceFile::new("t.aur", "let x = @\n");
        let at = file.src.find('@').unwrap() as u32;
        let d = Diagnostic::error("unexpected character `@`")
            .with_code("E0001")
            .primary(Span::new(at, at + 1), "not valid here")
            .note("attributes attach to items, not expressions");
        let rendered = d.render(&file);

        assert!(rendered.contains("error[E0001]: unexpected character `@`"));
        assert!(rendered.contains("--> t.aur:1:9"));
        assert!(rendered.contains("1 | let x = @"));
        assert!(rendered.contains("^ not valid here"));
        assert!(rendered.contains("= note: attributes attach to items"));
    }
}
