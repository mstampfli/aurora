//! Skeletal animation: sample a clip's TRS channels at a time, pose the
//! skeleton, and produce per-joint skinning matrices for the vertex shader.

use glam::{Mat4, Quat, Vec3};

use crate::model::{Channel, Interp, Model, Path, Skeleton};

/// Tracks playback of one clip on a model: current clip, time, speed, looping.
#[derive(Clone, Copy)]
pub struct AnimPlayer {
    pub clip: usize,
    pub time: f32,
    pub speed: f32,
    pub looping: bool,
}

impl Default for AnimPlayer {
    fn default() -> Self {
        AnimPlayer { clip: 0, time: 0.0, speed: 1.0, looping: true }
    }
}

impl AnimPlayer {
    pub fn new() -> AnimPlayer {
        AnimPlayer::default()
    }

    /// Advance playback by `dt` seconds, wrapping (looping) or clamping at the
    /// clip's end.
    pub fn advance(&mut self, model: &Model, dt: f32) {
        if let Some(c) = model.clips.get(self.clip) {
            self.time += dt * self.speed;
            if c.duration > 0.0 {
                if self.looping {
                    self.time = self.time.rem_euclid(c.duration);
                } else if self.time > c.duration {
                    self.time = c.duration;
                }
            }
        }
    }

    /// The skinning matrices (`global_joint * inverse_bind`) for the current
    /// pose, indexed by joint. Empty if the model has no skeleton.
    pub fn matrices(&self, model: &Model) -> Vec<Mat4> {
        match &model.skeleton {
            Some(skel) => skin_matrices(skel, model.clips.get(self.clip), self.time),
            None => Vec::new(),
        }
    }
}

/// Pose `skel` from `clip` at `time` and return per-joint skinning matrices.
pub fn skin_matrices(skel: &Skeleton, clip: Option<&crate::model::Clip>, time: f32) -> Vec<Mat4> {
    let n = skel.joints.len();
    // Start from the skeleton's default local TRS, then apply channel overrides.
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

    // Local then global transforms. glTF lists joints with parents before
    // children, but we resolve parents explicitly to be robust to any order.
    let local: Vec<Mat4> =
        (0..n).map(|i| Mat4::from_scale_rotation_translation(s[i], r[i], t[i])).collect();
    let mut global: Vec<Option<Mat4>> = vec![None; n];
    for i in 0..n {
        resolve_global(skel, &local, i, &mut global);
    }

    (0..n)
        .map(|i| global[i].unwrap_or(Mat4::IDENTITY) * skel.joints[i].inverse_bind)
        .collect()
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
