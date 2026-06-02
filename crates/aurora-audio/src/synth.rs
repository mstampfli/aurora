//! Deterministic audio synthesis + mixing — the part that needs no device, so
//! it is fully unit-testable. Oscillators, ADSR envelopes, note pitches, and a
//! mixdown that produces an interleaved-free mono `f32` PCM buffer in `[-1, 1]`.

use std::f32::consts::TAU;

/// Oscillator waveforms.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Wave {
    Sine,
    Square,
    Saw,
    Triangle,
}

impl Wave {
    /// Sample the waveform at normalized `phase` in `[0, 1)`, output in `[-1, 1]`.
    pub fn sample(self, phase: f32) -> f32 {
        let p = phase.rem_euclid(1.0);
        match self {
            Wave::Sine => (TAU * p).sin(),
            Wave::Square => {
                if p < 0.5 {
                    1.0
                } else {
                    -1.0
                }
            }
            Wave::Saw => 2.0 * p - 1.0,
            Wave::Triangle => 4.0 * (p - 0.5).abs() - 1.0,
        }
    }
}

/// An ADSR amplitude envelope (seconds for A/D/R; sustain is a 0..1 level).
#[derive(Clone, Copy, Debug)]
pub struct Adsr {
    pub attack: f32,
    pub decay: f32,
    pub sustain: f32,
    pub release: f32,
}

impl Default for Adsr {
    fn default() -> Adsr {
        Adsr { attack: 0.01, decay: 0.05, sustain: 0.7, release: 0.1 }
    }
}

impl Adsr {
    /// Amplitude at time `t` for a note held `dur` seconds (the release follows
    /// the held portion, so total length is `dur + release`).
    pub fn amplitude(&self, t: f32, dur: f32) -> f32 {
        if t < 0.0 {
            return 0.0;
        }
        if t < self.attack {
            return t / self.attack.max(1e-6);
        }
        if t < self.attack + self.decay {
            let x = (t - self.attack) / self.decay.max(1e-6);
            return 1.0 - x * (1.0 - self.sustain);
        }
        if t < dur {
            return self.sustain;
        }
        // Release phase after the note is let go.
        let r = (t - dur) / self.release.max(1e-6);
        (self.sustain * (1.0 - r)).max(0.0)
    }

    fn total(&self, dur: f32) -> f32 {
        dur + self.release
    }
}

/// Pitch in Hz for a number of semitones relative to A4 (440 Hz). E.g. `0` → A4,
/// `12` → A5, `3` → C5, `-9` → C4 (middle C).
pub fn pitch(semitones_from_a4: i32) -> f32 {
    440.0 * 2f32.powf(semitones_from_a4 as f32 / 12.0)
}

/// One note to render.
#[derive(Clone, Copy, Debug)]
pub struct Note {
    pub freq: f32,
    pub dur: f32,
    pub wave: Wave,
    pub gain: f32,
    pub adsr: Adsr,
}

impl Note {
    pub fn new(freq: f32, dur: f32) -> Note {
        Note { freq, dur, wave: Wave::Sine, gain: 0.5, adsr: Adsr::default() }
    }

    pub fn wave(mut self, w: Wave) -> Note {
        self.wave = w;
        self
    }

    pub fn gain(mut self, g: f32) -> Note {
        self.gain = g;
        self
    }

    /// Render this note to a mono PCM buffer at `sample_rate` Hz.
    pub fn render(&self, sample_rate: u32) -> Vec<f32> {
        let total = self.adsr.total(self.dur);
        let n = (total * sample_rate as f32).ceil() as usize;
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let t = i as f32 / sample_rate as f32;
            let phase = self.freq * t;
            let amp = self.adsr.amplitude(t, self.dur);
            out.push(self.wave.sample(phase) * amp * self.gain);
        }
        out
    }
}

/// Render a sequence of notes played one after another (a melody) into a single
/// buffer at `sample_rate` Hz.
pub fn render_sequence(notes: &[Note], sample_rate: u32) -> Vec<f32> {
    let mut out = Vec::new();
    for note in notes {
        out.extend(note.render(sample_rate));
    }
    out
}

/// Mix several buffers sample-wise (summed, then clamped to `[-1, 1]`). The
/// result is as long as the longest input.
pub fn mix(tracks: &[Vec<f32>]) -> Vec<f32> {
    let len = tracks.iter().map(Vec::len).max().unwrap_or(0);
    let mut out = vec![0.0f32; len];
    for t in tracks {
        for (o, s) in out.iter_mut().zip(t) {
            *o += *s;
        }
    }
    for s in &mut out {
        *s = s.clamp(-1.0, 1.0);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn waveforms_stay_in_range() {
        for w in [Wave::Sine, Wave::Square, Wave::Saw, Wave::Triangle] {
            for i in 0..1000 {
                let s = w.sample(i as f32 / 999.0);
                assert!((-1.0..=1.0).contains(&s), "{w:?} out of range: {s}");
            }
        }
    }

    #[test]
    fn pitch_octave_doubles_frequency() {
        assert!((pitch(0) - 440.0).abs() < 1e-3, "A4 = 440");
        assert!((pitch(12) - 880.0).abs() < 1e-3, "A5 = 880");
        assert!((pitch(-12) - 220.0).abs() < 1e-3, "A3 = 220");
        // Middle C (C4) is 9 semitones below A4 ≈ 261.63 Hz.
        assert!((pitch(-9) - 261.63).abs() < 0.1, "C4 ≈ 261.63, got {}", pitch(-9));
    }

    #[test]
    fn sine_note_has_expected_length_and_frequency() {
        let sr = 44_100;
        let note = Note::new(440.0, 1.0).wave(Wave::Sine).gain(1.0);
        let buf = note.render(sr);
        // Length = (dur + release) seconds of samples.
        let expected = ((1.0 + note.adsr.release) * sr as f32).ceil() as usize;
        assert_eq!(buf.len(), expected);
        // Count zero crossings in the sustained middle second — ~2 per cycle.
        let mid = &buf[(sr as usize) / 4..(sr as usize) * 3 / 4]; // 0.25s..0.75s
        let crossings = mid.windows(2).filter(|w| w[0].signum() != w[1].signum()).count();
        // 440 Hz over 0.5 s ≈ 220 cycles ≈ 440 crossings (allow tolerance).
        assert!((420..=460).contains(&crossings), "≈440 crossings expected, got {crossings}");
    }

    #[test]
    fn adsr_starts_silent_and_attacks() {
        let env = Adsr { attack: 0.1, decay: 0.1, sustain: 0.5, release: 0.1 };
        assert!(env.amplitude(0.0, 1.0).abs() < 1e-6, "silent at t=0");
        assert!(env.amplitude(0.05, 1.0) > 0.3, "rising during attack");
        assert!((env.amplitude(0.1, 1.0) - 1.0).abs() < 0.05, "peak at end of attack");
        assert!((env.amplitude(0.5, 1.0) - 0.5).abs() < 0.05, "sustain level");
        assert!(env.amplitude(1.2, 1.0).abs() < 1e-6, "silent after release");
    }

    #[test]
    fn mixing_is_clamped() {
        let a = vec![0.8f32; 10];
        let b = vec![0.8f32; 10];
        let m = mix(&[a, b]);
        assert_eq!(m.len(), 10);
        assert!(m.iter().all(|&s| s <= 1.0), "mix must be clamped to 1.0");
        assert!((m[0] - 1.0).abs() < 1e-6, "0.8 + 0.8 clamps to 1.0");
    }

    #[test]
    fn sequence_concatenates_notes() {
        let sr = 8_000;
        let a = Note::new(440.0, 0.1);
        let b = Note::new(550.0, 0.1);
        let seq = render_sequence(&[a, b], sr);
        assert_eq!(seq.len(), a.render(sr).len() + b.render(sr).len());
    }
}
