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
    gain: f32,
    /// Stereo pan in [-1, 1]: -1 full left, 0 center, +1 full right.
    pan: f32,
}

struct Mixer {
    voices: Vec<Voice>,
    volume: f32,
    device_rate: u32,
}

impl Mixer {
    /// Pull the next mixed stereo frame, advancing/cleaning up voices. A centered
    /// voice contributes 0.5 to each channel (so left+right == the mono signal).
    fn next_frame(&mut self) -> (f32, f32) {
        let mut l = 0.0f32;
        let mut r = 0.0f32;
        self.voices.retain_mut(|v| {
            if v.pos >= v.samples.len() {
                if v.looped && !v.samples.is_empty() {
                    v.pos = 0;
                } else {
                    return false;
                }
            }
            let s = v.samples[v.pos] * v.gain;
            let lp = (1.0 - v.pan) * 0.5;
            let rp = (1.0 + v.pan) * 0.5;
            l += s * lp;
            r += s * rp;
            v.pos += 1;
            true
        });
        (soft_limit(l * self.volume), soft_limit(r * self.volume))
    }

    /// Mono mix (sum of both channels), for single-channel devices and tests.
    fn next_sample(&mut self) -> f32 {
        let (l, r) = self.next_frame();
        soft_limit(l + r)
    }
}

/// Soft limiter: transparent below ~0.7, then smoothly saturates toward +-1 instead of HARD
/// clipping. Stacked SFX (e.g. rapid overlapping gunfire) summing past 1.0 no longer buzz/cut.
fn soft_limit(x: f32) -> f32 {
    let a = x.abs();
    if a <= 0.7 {
        x
    } else {
        x.signum() * (1.0 - 0.3 * (-(a - 0.7) / 0.3).exp())
    }
}

static MIXER: OnceLock<Arc<Mutex<Mixer>>> = OnceLock::new();

thread_local! {
    static STREAM: std::cell::RefCell<Option<cpal::Stream>> =
        const { std::cell::RefCell::new(None) };
}

/// Leak the output stream instead of dropping it. Call right before the process exits:
/// cpal's `Stream` panics if torn down in a thread-local destructor at process exit
/// ("thread local panicked on drop"). Leaking it makes shutdown graceful.
pub fn leak() {
    STREAM.with(|s| {
        if let Some(stream) = s.borrow_mut().take() {
            std::mem::forget(stream);
        }
    });
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
                        if channels >= 2 {
                            let (l, r) = m.next_frame();
                            frame[0] = $conv(l);
                            frame[1] = $conv(r);
                            // Mirror to any extra channels.
                            let mono = $conv((l + r).clamp(-1.0, 1.0));
                            for ch in frame.iter_mut().skip(2) {
                                *ch = mono;
                            }
                        } else {
                            let v = $conv(m.next_sample());
                            for ch in frame.iter_mut() {
                                *ch = v;
                            }
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
    play_spatial(samples, src_rate, looped, 1.0, 0.0);
}

/// Like [`play`] but with a per-voice `gain` (0..) and stereo `pan` (-1..1). The
/// runtime uses this for positional 3D audio (distance attenuation + panning).
pub fn play_spatial(samples: &[f32], src_rate: u32, looped: bool, gain: f32, pan: f32) {
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
    mixer().lock().unwrap().voices.push(Voice {
        samples: Arc::new(buf),
        pos: 0,
        looped,
        gain: gain.max(0.0),
        pan: pan.clamp(-1.0, 1.0),
    });
}

/// Like [`play_spatial`] but takes an already-decoded buffer AT THE DEVICE RATE, sharing it by
/// `Arc` so repeated plays (e.g. a gun firing fast) never re-copy or re-decode the samples - which
/// is what caused the periodic hitch on sustained fire.
pub fn play_spatial_arc(samples: Arc<Vec<f32>>, looped: bool, gain: f32, pan: f32) {
    if start().is_err() || samples.is_empty() {
        return;
    }
    mixer().lock().unwrap().voices.push(Voice {
        samples,
        pos: 0,
        looped,
        gain: gain.max(0.0),
        pan: pan.clamp(-1.0, 1.0),
    });
}

/// The audio device's sample rate (starting the device if needed). Lets a caller pre-resample a
/// cached sound ONCE to match the device and then replay it via [`play_spatial_arc`] with no copy.
pub fn device_rate() -> u32 {
    let _ = start();
    mixer().lock().unwrap().device_rate
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
        // soft-limit, and finish after their samples are consumed.
        let mut m = Mixer { voices: Vec::new(), volume: 1.0, device_rate: 44_100 };
        m.voices.push(Voice { samples: Arc::new(vec![0.6, 0.6]), pos: 0, looped: false, gain: 1.0, pan: 0.0 });
        m.voices.push(Voice { samples: Arc::new(vec![0.6, 0.6]), pos: 0, looped: false, gain: 1.0, pan: 0.0 });
        let s = m.next_sample();
        assert!(s > 0.85 && s < 1.0, "0.6+0.6 (sum 1.2) soft-limits just below 1.0, got {s}");
        assert_eq!(m.voices.len(), 2, "still playing after one sample");
        let _ = m.next_sample(); // consume the 2nd sample of each
        let _ = m.next_sample(); // now exhausted
        assert_eq!(m.voices.len(), 0, "finished voices are dropped");
    }

    #[test]
    fn volume_scales_the_mix() {
        let mut m = Mixer { voices: Vec::new(), volume: 0.5, device_rate: 44_100 };
        m.voices.push(Voice { samples: Arc::new(vec![1.0]), pos: 0, looped: false, gain: 1.0, pan: 0.0 });
        assert!((m.next_sample() - 0.5).abs() < 1e-6, "volume 0.5 halves the sample");
    }

    #[test]
    fn pan_splits_into_stereo_channels() {
        let mut m = Mixer { voices: Vec::new(), volume: 1.0, device_rate: 44_100 };
        m.voices.push(Voice { samples: Arc::new(vec![1.0]), pos: 0, looped: false, gain: 1.0, pan: -1.0 });
        let (l, r) = m.next_frame();
        assert!(l > 0.85 && r < 0.1, "pan -1 should be full-left, got l={l} r={r}");
    }

    #[test]
    fn gain_attenuates_a_voice() {
        let mut m = Mixer { voices: Vec::new(), volume: 1.0, device_rate: 44_100 };
        m.voices.push(Voice { samples: Arc::new(vec![1.0]), pos: 0, looped: false, gain: 0.25, pan: 0.0 });
        assert!((m.next_sample() - 0.25).abs() < 1e-6, "gain 0.25 attenuates the voice");
    }

    #[test]
    fn looped_voice_wraps() {
        let mut m = Mixer { voices: Vec::new(), volume: 1.0, device_rate: 44_100 };
        m.voices.push(Voice { samples: Arc::new(vec![0.2, 0.4]), pos: 0, looped: true, gain: 1.0, pan: 0.0 });
        let a = m.next_sample();
        let b = m.next_sample();
        let c = m.next_sample(); // wraps back to sample 0
        assert!((a - 0.2).abs() < 1e-6 && (b - 0.4).abs() < 1e-6 && (c - 0.2).abs() < 1e-6);
        assert_eq!(m.voices.len(), 1, "looped voice never finishes");
    }
}
