//! Source positions for the Aurora toolchain.
//!
//! A [`Span`] is a half-open byte range `[lo, hi)` into a single source file.
//! [`SourceFile`] owns the text and answers line/column queries used by the
//! diagnostics renderer. Multi-file support arrives via [`SourceMap`] once the
//! module/`use` graph (resolver) needs it; until then most callers work with a
//! single [`SourceFile`].

use std::fmt;
use std::ops::Range;

/// Identifies one source file within a [`SourceMap`].
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct SourceId(pub u32);

/// A half-open byte range `[lo, hi)` into a source file.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Span {
    pub lo: u32,
    pub hi: u32,
}

impl Span {
    /// A placeholder span for compiler-synthesized nodes with no source text.
    pub const DUMMY: Span = Span { lo: 0, hi: 0 };

    #[inline]
    pub fn new(lo: u32, hi: u32) -> Span {
        debug_assert!(lo <= hi, "span lo ({lo}) must be <= hi ({hi})");
        Span { lo, hi }
    }

    /// The smallest span covering both `self` and `other`.
    #[inline]
    pub fn to(self, other: Span) -> Span {
        Span { lo: self.lo.min(other.lo), hi: self.hi.max(other.hi) }
    }

    #[inline]
    pub fn len(self) -> u32 {
        self.hi - self.lo
    }

    #[inline]
    pub fn is_empty(self) -> bool {
        self.lo == self.hi
    }

    #[inline]
    pub fn range(self) -> Range<usize> {
        self.lo as usize..self.hi as usize
    }
}

impl fmt::Debug for Span {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}..{}", self.lo, self.hi)
    }
}

/// A 1-based line/column location. Columns count Unicode scalar values, not
/// bytes, so multi-byte characters do not skew caret alignment.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct LineCol {
    pub line: u32,
    pub col: u32,
}

/// One source file: its name, full text, and a cached index of line starts.
#[derive(Clone, Debug)]
pub struct SourceFile {
    pub name: String,
    pub src: String,
    /// Byte offset of the start of each line. `line_starts[0] == 0`.
    line_starts: Vec<u32>,
}

impl SourceFile {
    pub fn new(name: impl Into<String>, src: impl Into<String>) -> SourceFile {
        let src = src.into();
        let mut line_starts = vec![0u32];
        for (i, b) in src.bytes().enumerate() {
            if b == b'\n' {
                line_starts.push((i + 1) as u32);
            }
        }
        SourceFile { name: name.into(), src, line_starts }
    }

    /// The text covered by `span`.
    pub fn slice(&self, span: Span) -> &str {
        &self.src[span.range()]
    }

    /// Convert a byte offset to a 1-based line/column.
    pub fn line_col(&self, offset: u32) -> LineCol {
        // Largest line whose start is <= offset.
        let line_idx = match self.line_starts.binary_search(&offset) {
            Ok(i) => i,
            Err(i) => i - 1,
        };
        let line_start = self.line_starts[line_idx] as usize;
        let off = (offset as usize).min(self.src.len());
        let col = self.src[line_start..off].chars().count() as u32 + 1;
        LineCol { line: line_idx as u32 + 1, col }
    }

    /// The text of a 1-based line, without its trailing newline.
    pub fn line_text(&self, line: u32) -> &str {
        let idx = (line as usize).saturating_sub(1);
        let start = self.line_starts.get(idx).copied().unwrap_or(0) as usize;
        let end = self
            .line_starts
            .get(idx + 1)
            .map(|&e| e as usize)
            .unwrap_or(self.src.len());
        self.src[start..end].trim_end_matches(['\n', '\r'])
    }

    pub fn line_count(&self) -> u32 {
        self.line_starts.len() as u32
    }
}

/// A collection of source files addressed by [`SourceId`].
#[derive(Default, Debug)]
pub struct SourceMap {
    files: Vec<SourceFile>,
}

impl SourceMap {
    pub fn new() -> SourceMap {
        SourceMap::default()
    }

    pub fn add(&mut self, file: SourceFile) -> SourceId {
        let id = SourceId(self.files.len() as u32);
        self.files.push(file);
        id
    }

    pub fn get(&self, id: SourceId) -> &SourceFile {
        &self.files[id.0 as usize]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn span_join_and_len() {
        let a = Span::new(2, 5);
        let b = Span::new(10, 12);
        assert_eq!(a.to(b), Span::new(2, 12));
        assert_eq!(a.len(), 3);
        assert!(!a.is_empty());
        assert!(Span::DUMMY.is_empty());
    }

    #[test]
    fn line_col_basic() {
        let f = SourceFile::new("t.aur", "let x = 1\nlet y = 2\n");
        assert_eq!(f.line_col(0), LineCol { line: 1, col: 1 });
        assert_eq!(f.line_col(4), LineCol { line: 1, col: 5 });
        // first byte of line 2 ("let y") is right after the '\n' at offset 9
        assert_eq!(f.line_col(10), LineCol { line: 2, col: 1 });
        assert_eq!(f.line_text(2), "let y = 2");
    }

    #[test]
    fn line_col_counts_chars_not_bytes() {
        // 'é' is two bytes; the 'x' after it should be column 3, not 4.
        let f = SourceFile::new("t.aur", "éx");
        let x_off = "é".len() as u32;
        assert_eq!(f.line_col(x_off), LineCol { line: 1, col: 2 });
    }
}
