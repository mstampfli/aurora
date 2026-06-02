//! A persistent audio engine: a single output stream whose callback mixes any
//! number of concurrently-playing voices, with a master volume. This makes
//! sound *non-blocking* (fire-and-forget) and lets effects and music overlap —
//! unlike the one-shot blocking [`crate::play`].
//!
//! The mixer state is shared with the audio callback thread via `Arc<Mutex>`;
//! the stream object lives on the thread that started it (the game thread).

use std::sync::{Arc, Mutex, OnceLock};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

struct Voice {
    samples: Arc<Vec<f32>>,
    pos: usize,
    looped: bool,
}

struct Mixer {
    voices: Vec<Voice>,
    volume: f32,
    device_rate: u32,
}

impl Mixer {
    /// Pull the next mixed mono sample, advancing/cleaning up voices.
    fn next_sample(&mut self) -> f32 {
        let mut acc = 0.0f32;
        self.voices.retain_mut(|v| {
            if v.pos >= v.samples.len() {
                if v.looped && !v.samples.is_empty() {
                    v.pos = 0;
                } else {
                    return false;
                }
            }
            acc += v.samples[v.pos];
            v.pos += 1;
            true
        });
        (acc * self.volume).clamp(-1.0, 1.0)
    }
}

static MIXER: OnceLock<Arc<Mutex<Mixer>>> = OnceLock::new();

thread_local! {
    static STREAM: std::cell::RefCell<Option<cpal::Stream>> =
        const { std::cell::RefCell::new(None) };
}

fn mixer() -> &'static Arc<Mutex<Mixer>> {
    MIXER.get_or_init(|| Arc::new(Mutex::new(Mixer { voices: Vec::new(), volume: 1.0, device_rate: 44_100 })))
}

/// Start the audio engine's output stream if it isn't running. Idempotent.
pub fn start() -> Result<(), String> {
    let already = STREAM.with(|s| s.borrow().is_some());
    if already {
        return Ok(());
    }
    let host = cpal::default_host();
    let device = host.default_output_device().ok_or("no audio output device")?;
    let config = device.default_output_config().map_err(|e| format!("config: {e}"))?;
    let channels = config.channels() as usize;
    mixer().lock().unwrap().device_rate = config.sample_rate().0;

    let mix = mixer().clone();
    let cfg: cpal::StreamConfig = config.config();
    let err_fn = |e| eprintln!("audio engine error: {e}");

    macro_rules! make {
        ($t:ty, $conv:expr) => {{
            let mix = mix.clone();
            device.build_output_stream(
                &cfg,
                move |out: &mut [$t], _| {
                    let mut m = mix.lock().unwrap();
                    for frame in out.chunks_mut(channels) {
                        let s = m.next_sample();
                        let v = $conv(s);
                        for ch in frame.iter_mut() {
                            *ch = v;
                        }
                    }
                },
                err_fn,
                None,
            )
        }};
    }

    let stream = match config.sample_format() {
        cpal::SampleFormat::F32 => make!(f32, |s: f32| s),
        cpal::SampleFormat::I16 => make!(i16, |s: f32| (s * i16::MAX as f32) as i16),
        cpal::SampleFormat::U16 => make!(u16, |s: f32| (((s + 1.0) * 0.5) * u16::MAX as f32) as u16),
        other => return Err(format!("unsupported sample format: {other:?}")),
    }
    .map_err(|e| format!("build stream: {e}"))?;
    stream.play().map_err(|e| format!("play: {e}"))?;
    STREAM.with(|s| *s.borrow_mut() = Some(stream));
    Ok(())
}

/// Queue a mono buffer (sampled at `src_rate`) to play, mixed with whatever is
/// already sounding. Non-blocking. Resamples to the device rate if needed.
pub fn play(samples: &[f32], src_rate: u32, looped: bool) {
    if start().is_err() || samples.is_empty() {
        return;
    }
    let device_rate = mixer().lock().unwrap().device_rate;
    let buf = if device_rate == src_rate {
        samples.to_vec()
    } else {
        let ratio = src_rate as f64 / device_rate as f64;
        let n = (samples.len() as f64 / ratio) as usize;
        (0..n).map(|i| samples.get((i as f64 * ratio) as usize).copied().unwrap_or(0.0)).collect()
    };
    mixer().lock().unwrap().voices.push(Voice { samples: Arc::new(buf), pos: 0, looped });
}

/// Set the master volume (0.0..=~1.0+).
pub fn set_volume(v: f32) {
    mixer().lock().unwrap().volume = v.max(0.0);
}

/// Stop all currently-playing voices.
pub fn stop_all() {
    mixer().lock().unwrap().voices.clear();
}

/// Number of voices currently sounding (for tests/introspection).
pub fn active_voices() -> usize {
    mixer().lock().unwrap().voices.len()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mixer_sums_and_advances_voices() {
        // Drive the mixer directly (no device): two constant "voices" sum and
        // clamp, and finish after their samples are consumed.
        let mut m = Mixer { voices: Vec::new(), volume: 1.0, device_rate: 44_100 };
        m.voices.push(Voice { samples: Arc::new(vec![0.6, 0.6]), pos: 0, looped: false });
        m.voices.push(Voice { samples: Arc::new(vec![0.6, 0.6]), pos: 0, looped: false });
        assert!((m.next_sample() - 1.0).abs() < 1e-6, "0.6+0.6 clamps to 1.0");
        assert_eq!(m.voices.len(), 2, "still playing after one sample");
        let _ = m.next_sample(); // consume the 2nd sample of each
        let _ = m.next_sample(); // now exhausted
        assert_eq!(m.voices.len(), 0, "finished voices are dropped");
    }

    #[test]
    fn volume_scales_the_mix() {
        let mut m = Mixer { voices: Vec::new(), volume: 0.5, device_rate: 44_100 };
        m.voices.push(Voice { samples: Arc::new(vec![1.0]), pos: 0, looped: false });
        assert!((m.next_sample() - 0.5).abs() < 1e-6, "volume 0.5 halves the sample");
    }

    #[test]
    fn looped_voice_wraps() {
        let mut m = Mixer { voices: Vec::new(), volume: 1.0, device_rate: 44_100 };
        m.voices.push(Voice { samples: Arc::new(vec![0.2, 0.4]), pos: 0, looped: true });
        let a = m.next_sample();
        let b = m.next_sample();
        let c = m.next_sample(); // wraps back to sample 0
        assert!((a - 0.2).abs() < 1e-6 && (b - 0.4).abs() < 1e-6 && (c - 0.2).abs() < 1e-6);
        assert_eq!(m.voices.len(), 1, "looped voice never finishes");
    }
}
