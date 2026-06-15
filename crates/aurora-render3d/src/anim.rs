//! Skeletal animation: sample a clip's TRS channels at a time, pose the
//! skeleton, and produce per-joint skinning matrices for the vertex shader.

use glam::{Mat4, Quat, Vec3};

use crate::model::{Channel, Interp, Model, Path, Skeleton};

/// Tracks playback of a clip on a model, with crossfade blending from the
/// previously-playing clip.
#[derive(Clone, Copy)]
pub struct AnimPlayer {
    pub clip: usize,
    pub time: f32,
    pub speed: f32,
    pub looping: bool,
    // Crossfade source (the clip we're blending out of).
    prev_clip: usize,
    prev_time: f32,
    blend: f32,       // 0 = fully prev, 1 = fully current
    blend_rate: f32,  // blend units per second (1/fade_seconds)
}

impl Default for AnimPlayer {
    fn default() -> Self {
        AnimPlayer {
            clip: 0,
            time: 0.0,
            speed: 1.0,
            looping: true,
            prev_clip: 0,
            prev_time: 0.0,
            blend: 1.0,
            blend_rate: 0.0,
        }
    }
}

impl AnimPlayer {
    pub fn new() -> AnimPlayer {
        AnimPlayer::default()
    }

    /// Switch to `clip`, crossfading from the current pose over `fade` seconds
    /// (0 = instant). Restarts the clip at time 0.
    pub fn play(&mut self, clip: usize, looping: bool, speed: f32, fade: f32) {
        if fade > 0.0001 {
            self.prev_clip = self.clip;
            self.prev_time = self.time;
            self.blend = 0.0;
            self.blend_rate = 1.0 / fade;
        } else {
            self.blend = 1.0;
            self.blend_rate = 0.0;
        }
        self.clip = clip;
        self.time = 0.0;
        self.looping = looping;
        self.speed = speed;
    }

    /// Advance playback (and any crossfade) by `dt` seconds.
    pub fn advance(&mut self, model: &Model, dt: f32) {
        advance_time(&mut self.time, model.clips.get(self.clip), dt * self.speed, self.looping);
        if self.blend < 1.0 {
            // Keep the outgoing clip moving for a smooth blend.
            advance_time(&mut self.prev_time, model.clips.get(self.prev_clip), dt * self.speed, true);
            self.blend = (self.blend + self.blend_rate * dt).min(1.0);
        }
    }

    /// The skinning matrices for the current (possibly blended) pose. Empty if
    /// the model has no skeleton.
    pub fn matrices(&self, model: &Model) -> Vec<Mat4> {
        let Some(skel) = &model.skeleton else { return Vec::new() };
        if self.blend >= 1.0 {
            skin_matrices(skel, model.clips.get(self.clip), self.time)
        } else {
            skin_matrices_blended(
                skel,
                model.clips.get(self.prev_clip),
                self.prev_time,
                model.clips.get(self.clip),
                self.time,
                self.blend,
            )
        }
    }
}

fn advance_time(time: &mut f32, clip: Option<&crate::model::Clip>, dt: f32, looping: bool) {
    *time += dt;
    if let Some(c) = clip {
        if c.duration > 0.0 {
            if looping {
                *time = time.rem_euclid(c.duration);
            } else if *time > c.duration {
                *time = c.duration;
            }
        }
    }
}

/// Sample a clip at `time` into per-joint local TRS, starting from the
/// skeleton's defaults.
fn sample_locals(
    skel: &Skeleton,
    clip: Option<&crate::model::Clip>,
    time: f32,
) -> (Vec<Vec3>, Vec<Quat>, Vec<Vec3>) {
    let n = skel.joints.len();
    let mut t: Vec<Vec3> = skel.joints.iter().map(|j| j.t).collect();
    let mut r: Vec<Quat> = skel.joints.iter().map(|j| j.r).collect();
    let mut s: Vec<Vec3> = skel.joints.iter().map(|j| j.s).collect();
    if let Some(clip) = clip {
        for ch in &clip.channels {
            if ch.joint >= n {
                continue;
            }
            match ch.path {
                Path::Translation => t[ch.joint] = sample_vec3(ch, time),
                Path::Scale => s[ch.joint] = sample_vec3(ch, time),
                Path::Rotation => r[ch.joint] = sample_quat(ch, time),
            }
        }
    }
    (t, r, s)
}

/// Turn per-joint local TRS into skinning matrices (`global * inverse_bind`).
fn locals_to_skin(skel: &Skeleton, t: &[Vec3], r: &[Quat], s: &[Vec3]) -> Vec<Mat4> {
    let n = skel.joints.len();
    let local: Vec<Mat4> =
        (0..n).map(|i| Mat4::from_scale_rotation_translation(s[i], r[i], t[i])).collect();
    let mut global: Vec<Option<Mat4>> = vec![None; n];
    for i in 0..n {
        resolve_global(skel, &local, i, &mut global);
    }
    (0..n).map(|i| global[i].unwrap_or(Mat4::IDENTITY) * skel.joints[i].inverse_bind).collect()
}

/// Pose `skel` from `clip` at `time` and return per-joint skinning matrices.
pub fn skin_matrices(skel: &Skeleton, clip: Option<&crate::model::Clip>, time: f32) -> Vec<Mat4> {
    let (t, r, s) = sample_locals(skel, clip, time);
    locals_to_skin(skel, &t, &r, &s)
}

/// Blend two clips' poses by weight `w` (0 = clip a, 1 = clip b) and return the
/// skinning matrices. Blends in local TRS space (correct), not matrix space.
pub fn skin_matrices_blended(
    skel: &Skeleton,
    a: Option<&crate::model::Clip>,
    ta: f32,
    b: Option<&crate::model::Clip>,
    tb: f32,
    w: f32,
) -> Vec<Mat4> {
    let (at, ar, asc) = sample_locals(skel, a, ta);
    let (bt, br, bsc) = sample_locals(skel, b, tb);
    let w = w.clamp(0.0, 1.0);
    let n = skel.joints.len();
    let t: Vec<Vec3> = (0..n).map(|i| at[i].lerp(bt[i], w)).collect();
    let r: Vec<Quat> = (0..n).map(|i| ar[i].slerp(br[i], w)).collect();
    let s: Vec<Vec3> = (0..n).map(|i| asc[i].lerp(bsc[i], w)).collect();
    locals_to_skin(skel, &t, &r, &s)
}

fn resolve_global(skel: &Skeleton, local: &[Mat4], i: usize, global: &mut Vec<Option<Mat4>>) -> Mat4 {
    if let Some(g) = global[i] {
        return g;
    }
    let g = match skel.joints[i].parent {
        Some(p) if p != i => resolve_global(skel, local, p, global) * local[i],
        _ => local[i],
    };
    global[i] = Some(g);
    g
}

/// Find the key interval `[i, i+1]` containing `time` and the fraction within.
fn locate(times: &[f32], time: f32) -> (usize, usize, f32) {
    if times.is_empty() {
        return (0, 0, 0.0);
    }
    if time <= times[0] {
        return (0, 0, 0.0);
    }
    let last = times.len() - 1;
    if time >= times[last] {
        return (last, last, 0.0);
    }
    let mut i = 0;
    while i + 1 < times.len() && times[i + 1] < time {
        i += 1;
    }
    let (a, b) = (times[i], times[i + 1]);
    let f = if b > a { (time - a) / (b - a) } else { 0.0 };
    (i, i + 1, f)
}

fn sample_vec3(ch: &Channel, time: f32) -> Vec3 {
    let (i0, i1, f) = locate(&ch.times, time);
    let get = |k: usize| Vec3::new(ch.values[k * 3], ch.values[k * 3 + 1], ch.values[k * 3 + 2]);
    if ch.interp == Interp::Step || i0 == i1 {
        get(i0)
    } else {
        get(i0).lerp(get(i1), f)
    }
}

fn sample_quat(ch: &Channel, time: f32) -> Quat {
    let (i0, i1, f) = locate(&ch.times, time);
    let get = |k: usize| {
        Quat::from_xyzw(ch.values[k * 4], ch.values[k * 4 + 1], ch.values[k * 4 + 2], ch.values[k * 4 + 3])
            .normalize()
    };
    if ch.interp == Interp::Step || i0 == i1 {
        get(i0)
    } else {
        get(i0).slerp(get(i1), f)
    }
}
