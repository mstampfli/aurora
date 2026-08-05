//! The built-in font, checked against the spec it was generated from.
//!
//! Glyph data is the kind of thing that is easy to regenerate wrongly and hard
//! to notice: a table shifted by one is a font where every letter is the next
//! one along, which looks like garbage but reads as "the text renderer is
//! broken" rather than "the table is off by one".

use super::*;

/// Render one glyph into a grid of `#` and space, the way the spec is written.
fn art(c: u8) -> Vec<String> {
    let cols = glyph(c).expect("printable");
    (0..GLYPH_H)
        .map(|row| {
            (0..GLYPH_W)
                .map(|x| {
                    if cols[x as usize] & (1 << row) != 0 {
                        '#'
                    } else {
                        ' '
                    }
                })
                .collect()
        })
        .collect()
}

/// Two glyphs, spelled out. If the table is shifted, rotated or transposed,
/// these are what say so.
#[test]
fn a_glyph_matches_its_spec() {
    assert_eq!(
        art(b'A'),
        vec!["  #  ", " # # ", "#   #", "#   #", "#####", "#   #", "#   #",]
    );
    // Asymmetric top to bottom AND left to right, so a flip in either axis
    // fails here even though 'A' would survive a horizontal one.
    assert_eq!(
        art(b'F'),
        vec!["#####", "#    ", "#    ", "#### ", "#    ", "#    ", "#    ",]
    );
}

#[test]
fn the_table_covers_printable_ascii_and_nothing_else() {
    assert_eq!(GLYPHS.len(), 95);
    assert!(glyph(b' ').is_some(), "space is the first glyph");
    assert!(glyph(b'~').is_some(), "tilde is the last");
    assert!(glyph(31).is_none(), "a control code has no glyph");
    assert!(glyph(127).is_none(), "delete has no glyph");
    // Space is blank, and it is the only blank glyph - a table with a hole in it
    // would otherwise pass every other test here.
    let blank: Vec<u8> = GLYPHS
        .iter()
        .enumerate()
        .filter(|(_, g)| g.iter().all(|c| *c == 0))
        .map(|(i, _)| FIRST + i as u8)
        .collect();
    assert_eq!(blank, vec![b' '], "exactly one glyph is blank");
}

#[test]
fn width_counts_gaps_between_glyphs_but_not_after_the_last() {
    assert_eq!(width("", 1), 0);
    assert_eq!(width("A", 1), 5, "one glyph is its own width");
    assert_eq!(width("AB", 1), 11, "two glyphs and one gap");
    assert_eq!(width("AB", 2), 22, "scale multiplies everything");
}

#[test]
fn scale_never_rounds_text_away() {
    assert_eq!(scale_for(14), 2);
    assert_eq!(scale_for(7), 1);
    // Asked for smaller than the font: drawn at 7px rather than not at all.
    assert_eq!(scale_for(3), 1);
    assert_eq!(scale_for(0), 1);
}

#[test]
fn blit_lights_the_pixels_the_glyph_says() {
    let mut lit: Vec<(i64, i64)> = Vec::new();
    let end = blit(0, 0, "A", 1, |x, y| lit.push((x, y)));
    assert_eq!(end, 6, "advance is the glyph plus its gap");

    // The apex of 'A' is the only pixel in its top row, at column 2.
    let top: Vec<i64> = lit
        .iter()
        .filter(|(_, y)| *y == 0)
        .map(|(x, _)| *x)
        .collect();
    assert_eq!(top, vec![2]);
    // The crossbar is the full five.
    let mut bar: Vec<i64> = lit
        .iter()
        .filter(|(_, y)| *y == 4)
        .map(|(x, _)| *x)
        .collect();
    bar.sort();
    assert_eq!(bar, vec![0, 1, 2, 3, 4]);
}

#[test]
fn blit_scales_by_whole_pixels() {
    let mut lit: Vec<(i64, i64)> = Vec::new();
    blit(0, 0, "A", 2, |x, y| lit.push((x, y)));
    // The apex becomes a 2x2 block at (4,0), not a blurred one.
    let top: Vec<(i64, i64)> = lit.iter().filter(|(_, y)| *y < 2).copied().collect();
    assert_eq!(top.len(), 4);
    for (x, y) in top {
        assert!(
            (4..6).contains(&x) && (0..2).contains(&y),
            "stray pixel at {x},{y}"
        );
    }
}

#[test]
fn a_non_ascii_character_draws_a_question_mark_rather_than_nothing() {
    let mut lit = 0;
    blit(0, 0, "\u{00e9}", 1, |_, _| lit += 1);
    let mut want = 0;
    blit(0, 0, "?", 1, |_, _| want += 1);
    assert_eq!(
        lit, want,
        "an unrepresentable character is visible, not silent"
    );
}
