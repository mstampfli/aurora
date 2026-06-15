//! Audio for Aurora: a tested synthesis/mixing core ([`synth`]) plus real
//! playback through the system's default output device via [`cpal`].
//!
//! The synthesis core renders mono `f32` PCM in `[-1, 1]`; [`play`] streams such
//! a buffer to the speakers (upmixed to the device's channel count) and blocks
//! until it finishes. Device acquisition is fallible, so `play` returns a
//! `Result` and the synthesis tests never touch hardware.

mod engine;
mod synth;
pub use engine::{
    active_voices, play as play_mixed, play_spatial as play_mixed_spatial, set_volume,
    start as start_engine, stop_all,
};
pub use synth::{mix, pitch, render_sequence, Adsr, Note, Wave};

use std::sync::mpsc;
use std::sync::{Arc, Mutex};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

/// Play a mono `f32` PCM buffer (sampled at `sample_rate`) on the default output
/// device, blocking until playback completes. Resamples by nearest-neighbor if
/// the device runs at a different rate, and upmixes mono to all channels.
pub fn play(samples: &[f32], sample_rate: u32) -> Result<(), String> {
    if samples.is_empty() {
        return Ok(());
    }
    let host = cpal::default_host();
    let device = host.default_output_device().ok_or("no audio output device")?;
    let config = device
        .default_output_config()
        .map_err(|e| format!("default output config: {e}"))?;
    let channels = config.channels() as usize;
    let device_rate = config.sample_rate().0;

    // Resample (nearest-neighbor) into the device's sample rate, mono.
    let mono: Vec<f32> = if device_rate == sample_rate {
        samples.to_vec()
    } else {
        let ratio = sample_rate as f64 / device_rate as f64;
        let out_len = (samples.len() as f64 / ratio) as usize;
        (0..out_len)
            .map(|i| {
                let src = (i as f64 * ratio) as usize;
                samples.get(src).copied().unwrap_or(0.0)
            })
            .collect()
    };

    let cursor = Arc::new(Mutex::new(0usize));
    let total = mono.len();
    let data = Arc::new(mono);
    let (done_tx, done_rx) = mpsc::channel::<()>();

    let stream = build_stream(&device, &config, channels, data, cursor, total, done_tx)?;
    stream.play().map_err(|e| format!("play: {e}"))?;
    // Block until the feeder signals it has emitted every sample.
    let _ = done_rx.recv();
    Ok(())
}

fn build_stream(
    device: &cpal::Device,
    config: &cpal::SupportedStreamConfig,
    channels: usize,
    data: Arc<Vec<f32>>,
    cursor: Arc<Mutex<usize>>,
    total: usize,
    done_tx: mpsc::Sender<()>,
) -> Result<cpal::Stream, String> {
    let err_fn = |e| eprintln!("audio stream error: {e}");
    let cfg: cpal::StreamConfig = config.config();
    let signalled = Arc::new(Mutex::new(false));

    macro_rules! make {
        ($t:ty, $convert:expr) => {{
            let data = data.clone();
            let cursor = cursor.clone();
            let done_tx = done_tx.clone();
            let signalled = signalled.clone();
            device
                .build_output_stream(
                    &cfg,
                    move |out: &mut [$t], _| {
                        let mut idx = cursor.lock().unwrap();
                        for frame in out.chunks_mut(channels) {
                            let s = data.get(*idx).copied().unwrap_or(0.0);
                            if *idx < total {
                                *idx += 1;
                            }
                            let v = $convert(s);
                            for ch in frame.iter_mut() {
                                *ch = v;
                            }
                        }
                        if *idx >= total {
                            let mut done = signalled.lock().unwrap();
                            if !*done {
                                *done = true;
                                let _ = done_tx.send(());
                            }
                        }
                    },
                    err_fn,
                    None,
                )
                .map_err(|e| format!("build stream: {e}"))
        }};
    }

    match config.sample_format() {
        cpal::SampleFormat::F32 => make!(f32, |s: f32| s),
        cpal::SampleFormat::I16 => make!(i16, |s: f32| (s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16),
        cpal::SampleFormat::U16 => {
            make!(u16, |s: f32| (((s.clamp(-1.0, 1.0) + 1.0) * 0.5) * u16::MAX as f32) as u16)
        }
        other => Err(format!("unsupported sample format: {other:?}")),
    }
}

/// A short built-in melody (an ascending major arpeggio), for `aurorac sound`.
pub fn demo_melody(sample_rate: u32) -> Vec<f32> {
    // C major arpeggio: C4 E4 G4 C5 (semitones from A4).
    let steps = [-9, -5, -2, 3];
    let notes: Vec<Note> =
        steps.iter().map(|&s| Note::new(pitch(s), 0.22).wave(Wave::Triangle).gain(0.6)).collect();
    render_sequence(&notes, sample_rate)
}
