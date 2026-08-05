//! Reading an image back in order to MEASURE it.
//!
//! `load_image` is the older way to get a PNG's pixels, and it works by replacing
//! the framebuffer - which is also the HUD layer `r3d_capture` composites over
//! the 3D frame. Measuring a capture that way therefore paints the measured image
//! over every capture that follows, full screen, while `r3d_capture` goes on
//! answering success. A game's own check hit exactly that and concluded from two
//! byte-identical files that its renderer was drawing nothing.
//!
//! `image_open` and friends hold the pixels off to one side and touch no drawing
//! state. These tests pin the part that makes them usable in a check: the reject
//! branches. A measurement API that answers a plausible number for a dead handle
//! or an out-of-range rectangle is worse than none, because the check keeps
//! printing a number and nobody looks again.
//!
//! Through the compiled path, because that is the one games run.

use aurora_parser::parse_str;

fn run(src: &str) -> i64 {
    let src = src.to_string();
    std::thread::spawn(move || {
        let (module, diags) = parse_str(&src);
        assert!(
            !diags.iter().any(|d| d.is_error()),
            "source failed to parse: {:?}",
            diags.iter().map(|d| &d.message).collect::<Vec<_>>()
        );
        let jit = aurora_codegen::build(&module).expect("must compile natively");
        jit.call_i64("run", &[]).expect("run")
    })
    .join()
    .expect("worker panicked")
}

/// Two 8x4 PNGs, one black and one white, written through the framebuffer.
fn make_pair(dir: &str) -> String {
    format!(
        "framebuffer(8, 4)
         clear(0, 0, 0)
         save_png(\"{dir}/black.png\")
         clear(255, 255, 255)
         save_png(\"{dir}/white.png\")
        "
    )
}

fn scratch(name: &str) -> String {
    let dir = std::env::temp_dir().join(format!("aurora-image-{name}"));
    let _ = std::fs::create_dir_all(&dir);
    dir.to_string_lossy().replace('\\', "/")
}

#[test]
fn an_image_reports_its_own_size_and_pixels() {
    let dir = scratch("size");
    let setup = make_pair(&dir);
    let n = run(&format!(
        "fn run() -> i64 {{
             {setup}
             let a = image_open(\"{dir}/black.png\")
             let b = image_open(\"{dir}/white.png\")
             if a < 0 {{ return 0 - 1 }}
             if b < 0 {{ return 0 - 2 }}
             if image_width(a) != 8 {{ return 0 - 3 }}
             if image_height(a) != 4 {{ return 0 - 4 }}
             if image_pixel(a, 0, 0) != 0 {{ return 0 - 5 }}
             if image_pixel(b, 7, 3) != 16777215 {{ return 0 - 6 }}
             1
         }}"
    ));
    assert_eq!(
        n, 1,
        "an opened image must answer for its own size and pixels"
    );
}

#[test]
fn brightness_and_difference_are_measured_over_a_region() {
    let dir = scratch("measure");
    let setup = make_pair(&dir);
    let n = run(&format!(
        "fn run() -> i64 {{
             {setup}
             let a = image_open(\"{dir}/black.png\")
             let b = image_open(\"{dir}/white.png\")
             // Black is 0 and white is 255 under any sane weighting, so these
             // pin the scale without pinning the coefficients.
             if image_mean_luma(a, 0, 0, 8, 4) > 0.001 {{ return 0 - 1 }}
             if image_mean_luma(b, 0, 0, 8, 4) < 254.999 {{ return 0 - 2 }}
             // A part of an image, not just the whole of it.
             if image_mean_luma(b, 2, 1, 3, 2) < 254.999 {{ return 0 - 3 }}
             // Two pictures that differ completely, and one compared to itself.
             if image_diff(a, b, 0, 0, 8, 4) < 254.999 {{ return 0 - 4 }}
             if image_diff(a, a, 0, 0, 8, 4) > 0.001 {{ return 0 - 5 }}
             1
         }}"
    ));
    assert_eq!(
        n, 1,
        "brightness and difference must be measurable over a region"
    );
}

#[test]
fn a_bad_handle_or_region_answers_negative_rather_than_plausible() {
    let dir = scratch("reject");
    let setup = make_pair(&dir);
    let n = run(&format!(
        "fn run() -> i64 {{
             {setup}
             let a = image_open(\"{dir}/black.png\")
             let b = image_open(\"{dir}/white.png\")
             // A file that is not there is NOT an empty image.
             if image_open(\"{dir}/nosuchfile.png\") != 0 - 1 {{ return 0 - 1 }}
             // A handle nobody handed out.
             if image_width(0 - 1) != 0 - 1 {{ return 0 - 2 }}
             if image_height(9999) != 0 - 1 {{ return 0 - 3 }}
             if image_pixel(9999, 0, 0) != 0 - 1 {{ return 0 - 4 }}
             // Outside the image is not black - black is a colour something has.
             if image_pixel(a, 8, 0) != 0 - 1 {{ return 0 - 5 }}
             if image_pixel(a, 0, 0 - 1) != 0 - 1 {{ return 0 - 6 }}
             // A region that does not fit is refused, not clamped to one that
             // does: a check that asked for the wrong rectangle must fail rather
             // than quietly measure a different one.
             if image_mean_luma(a, 0, 0, 9, 4) > 0.0 - 0.999 {{ return 0 - 7 }}
             if image_mean_luma(a, 0, 0, 0, 4) > 0.0 - 0.999 {{ return 0 - 8 }}
             if image_mean_luma(a, 0 - 1, 0, 4, 4) > 0.0 - 0.999 {{ return 0 - 9 }}
             if image_diff(a, b, 0, 0, 99, 99) > 0.0 - 0.999 {{ return 0 - 10 }}
             if image_diff(a, 9999, 0, 0, 8, 4) > 0.0 - 0.999 {{ return 0 - 11 }}
             1
         }}"
    ));
    assert_eq!(
        n, 1,
        "a dead handle or an ill-fitting region must answer negative, not a number a check would believe"
    );
}

#[test]
fn a_freed_image_stays_dead_and_its_handle_is_not_reused() {
    let dir = scratch("free");
    let setup = make_pair(&dir);
    let n = run(&format!(
        "fn run() -> i64 {{
             {setup}
             let a = image_open(\"{dir}/black.png\")
             if image_free(a) != 1 {{ return 0 - 1 }}
             // Freeing twice is not an error, but it is not a success either.
             if image_free(a) != 0 {{ return 0 - 2 }}
             if image_width(a) != 0 - 1 {{ return 0 - 3 }}
             if image_pixel(a, 0, 0) != 0 - 1 {{ return 0 - 4 }}
             // The next image must not inherit the dead handle, or a stale one
             // would silently start naming somebody else's picture.
             let b = image_open(\"{dir}/white.png\")
             if b == a {{ return 0 - 5 }}
             if image_pixel(b, 0, 0) != 16777215 {{ return 0 - 6 }}
             1
         }}"
    ));
    assert_eq!(
        n, 1,
        "a freed handle must stay dead and must not be handed out again"
    );
}
