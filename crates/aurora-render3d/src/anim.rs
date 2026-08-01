//! Skeletal animation: sample a clip's TRS channels at a time, pose the
//! skeleton, and produce per-joint skinning matrices for the vertex shader.

use glam::{Mat4, Quat, Vec3, Vec4};

use crate::model::{Model, Skeleton};

/// Tracks playback of a clip on a model, with crossfade blending from the
/// previously-playing clip.
#[derive(Clone, Copy)]
pub struct AnimPlayer {
    pub clip: usize,
    pub time: f32,
    pub speed: f32,
    pub looping: bool,
    // Crossfade source (the clip we're blending out of). A WHOLE playback state - clip, clock,
    // loop flag and rate - because those four decide together where the outgoing clip stands
    // while it fades, and three of them cannot answer it. Captured with `prev_clip`, never apart.
    prev_clip: usize,
    prev_time: f32,
    prev_looping: bool,
    prev_speed: f32,
    blend: f32,      // 0 = fully prev, 1 = fully current
    blend_rate: f32, // blend units per second (1/fade_seconds)
    // Sustained two-clip BASE blend (e.g. idle <-> run by movement speed): when `bblend_on`, the
    // full-body base pose is lerp(clip, bclip2, bblend). `btime2` advances bclip2 independently, so
    // both loops play at their own cadence. Call blend() every frame to track a continuous value.
    bclip2: usize,
    btime2: f32,
    bblend: f32,
    bblend_on: bool,
    // Optional upper-body overlay: a second clip applied only to `umask_root` and its
    // descendants (e.g. shoot/reload on the arms while the legs keep running). `uweight`
    // fades the overlay in/out so it never pops.
    pub upper: bool,
    pub uclip: usize,
    pub utime: f32,
    uspeed: f32,
    pub ulooping: bool,
    umask_root: usize,
    uweight: f32,
    uweight_target: f32,
    uweight_rate: f32,
    // Optional AIM BLEND: when `ublend_on`, the upper overlay is a weighted blend of `uclip` and
    // `uclip2` (ublend 0 = uclip, 1 = uclip2) BEFORE it is masked in - e.g. lerp a look-down aim
    // pose into a look-up one by the player's pitch. Both clips share `utime`.
    uclip2: usize,
    ublend: f32,
    ublend_on: bool,
    // Upper-overlay CROSSFADE: when the overlay clip changes (e.g. aim -> katana swing -> aim), the
    // old overlay pose (uprev_clip @ uprev_time) is crossfaded into the new one over `ufade` 0->1, so
    // overlay-to-overlay transitions blend instead of popping. Separate from `uweight` (overlay-vs-base).
    // The same whole state as `prev_*`, for the overlay's own crossfade source.
    uprev_clip: usize,
    uprev_time: f32,
    uprev_looping: bool,
    uprev_speed: f32,
    ufade: f32,
    ufade_rate: f32,
    // Per-bone POSE overrides: an extra local rotation pre-multiplied onto a joint after the clip
    // pose is sampled (and after the upper overlay). Lets game code author a pose the clips don't
    // have - e.g. bend the thighs forward into a slide while the spine keeps its upright clip pose.
    // Set each frame by the caller; cleared with clear_pose(). Fixed array so AnimPlayer stays Copy.
    pose: [(u32, Quat); 8],
    pose_n: usize,
    // ROOT MOTION: the ground the base layer covered during the LAST advance, in the model's own
    // space. A per-update delta and never a running total, because the caller integrates it into a
    // position it owns - a total would have to be reset by hand and the first caller to forget
    // would teleport.
    root_delta: Vec3,
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
            prev_looping: true,
            prev_speed: 1.0,
            blend: 1.0,
            blend_rate: 0.0,
            bclip2: 0,
            btime2: 0.0,
            bblend: 0.0,
            bblend_on: false,
            upper: false,
            uclip: 0,
            utime: 0.0,
            uspeed: 1.0,
            ulooping: true,
            umask_root: 0,
            uweight: 0.0,
            uweight_target: 0.0,
            uweight_rate: 0.0,
            uclip2: 0,
            ublend: 0.0,
            ublend_on: false,
            uprev_clip: 0,
            uprev_time: 0.0,
            uprev_looping: true,
            uprev_speed: 1.0,
            ufade: 1.0,
            ufade_rate: 0.0,
            pose: [(0u32, Quat::IDENTITY); 8],
            pose_n: 0,
            root_delta: Vec3::ZERO,
        }
    }
}

impl AnimPlayer {
    pub fn new() -> AnimPlayer {
        AnimPlayer::default()
    }

    /// Switch to `clip`, crossfading from the current pose over `fade` seconds
    /// (0 = instant). Restarts the clip at time 0.
    /// Ensure `clip` is the clip playing. Idempotent: calling this every frame
    /// with what is already running adjusts the speed and otherwise does
    /// NOTHING.
    ///
    /// This used to restart unconditionally - `time = 0.0` on every call - which
    /// made it unusable from the place games actually drive animation: a frame
    /// loop that says what state a character is in. Every caller had to
    /// remember to guard it, ten call sites hand-rolled three different guards
    /// (`anim_clip` comparisons, a `fell` flag on a component, nothing at all),
    /// and the ones that forgot re-seeded time every frame. That is a character
    /// frozen on frame 0, jittering, with no transition ever completing - which
    /// is exactly how it looked.
    ///
    /// "Play" is a statement about what SHOULD be on screen. Use
    /// [`Player::restart`] for the rare case that genuinely means "again from
    /// the top", such as the second swing of a combo reusing one clip.
    pub fn play(&mut self, clip: usize, looping: bool, speed: f32, fade: f32) {
        if self.clip == clip && self.looping == looping && !self.bblend_on {
            // Already what is asked for. Speed may still be tuned live (a
            // windup stretched to its frame data), so take that and leave the
            // clock alone.
            self.speed = speed;
            return;
        }
        self.restart(clip, looping, speed, fade);
    }

    /// Start `clip` from the top, even if it is already the clip playing.
    pub fn restart(&mut self, clip: usize, looping: bool, speed: f32, fade: f32) {
        if fade > 0.0001 {
            // Captured before `looping`/`speed` below are overwritten with the INCOMING clip's,
            // and in one group so the four cannot be captured apart.
            self.prev_clip = self.clip;
            self.prev_time = self.time;
            self.prev_looping = self.looping;
            self.prev_speed = self.base_speed();
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
        self.bblend_on = false; // a single base clip again, not a sustained blend
    }

    /// Drive the FULL-BODY base as a sustained weighted blend of two clips (`clip_a` at weight 0,
    /// `clip_b` at weight 1) - e.g. idle <-> run by movement speed, so the legs ease smoothly into
    /// standing still instead of snapping. Call every frame to update the weight; the first call
    /// that enters blend mode crossfades in over `fade` (so a jump->land transition is smooth too).
    pub fn blend(&mut self, clip_a: usize, clip_b: usize, weight: f32, speed: f32, fade: f32) {
        if !self.bblend_on {
            if fade > 0.0001 {
                self.prev_clip = self.clip;
                self.prev_time = self.time;
                self.prev_looping = self.looping;
                self.prev_speed = self.base_speed();
                self.blend = 0.0;
                self.blend_rate = 1.0 / fade;
            } else {
                self.blend = 1.0;
            }
            self.btime2 = 0.0;
        }
        self.bblend_on = true;
        self.clip = clip_a;
        self.bclip2 = clip_b;
        self.bblend = weight.clamp(0.0, 1.0);
        self.looping = true;
        self.speed = speed;
    }

    /// Start (or swap) an upper-body overlay clip, masked to `mask_root` + its descendants,
    /// fading the overlay weight in over `fade` seconds. The lower body keeps the base clip.
    pub fn play_upper(
        &mut self,
        clip: usize,
        looping: bool,
        speed: f32,
        fade: f32,
        mask_root: usize,
    ) {
        if self.upper {
            // Already overlaying: crossfade FROM the current overlay pose into the new clip.
            let (dc, dtime) = self.upper_dominant();
            self.uprev_clip = dc;
            self.uprev_time = dtime;
            self.uprev_looping = self.ulooping;
            self.uprev_speed = self.uspeed;
            self.ufade = 0.0;
            self.ufade_rate = if fade > 0.0001 {
                1.0 / fade
            } else {
                1_000_000.0
            };
        } else {
            self.uweight = 0.0;
            self.ufade = 1.0; // coming from the base; the uweight fade-in covers it, no clip crossfade
        }
        self.upper = true;
        self.ublend_on = false; // a plain single-clip overlay (reload/recoil/swing), not an aim blend
        self.uclip = clip;
        self.utime = 0.0;
        self.ulooping = looping;
        self.uspeed = speed;
        self.umask_root = mask_root;
        self.uweight_target = 1.0;
        self.uweight_rate = if fade > 0.0001 {
            1.0 / fade
        } else {
            1_000_000.0
        };
    }

    /// Drive the upper-body overlay as a weighted BLEND of two clips (`clip_a` at weight 0,
    /// `clip_b` at weight 1), masked to `mask_root`. Built to be called EVERY frame to track a
    /// continuous value (e.g. aim pitch): only the first call that enters blend mode fades the
    /// overlay in, so updating the weight/clips per frame stays smooth and never re-pops.
    pub fn aim_upper(
        &mut self,
        clip_a: usize,
        clip_b: usize,
        weight: f32,
        speed: f32,
        fade: f32,
        mask_root: usize,
    ) {
        let was_blend = self.ublend_on;
        if !self.upper {
            self.uweight = 0.0;
            self.utime = 0.0;
            self.ufade = 1.0;
        } else if !was_blend {
            // Transitioning INTO the aim blend from a single-clip overlay (katana/reload/recoil): crossfade.
            self.uprev_clip = self.uclip;
            self.uprev_time = self.utime;
            self.uprev_looping = self.ulooping;
            self.uprev_speed = self.uspeed;
            self.ufade = 0.0;
            self.ufade_rate = if fade > 0.0001 {
                1.0 / fade
            } else {
                1_000_000.0
            };
        }
        self.upper = true;
        self.ublend_on = true;
        self.uclip = clip_a;
        self.uclip2 = clip_b;
        self.ublend = weight.clamp(0.0, 1.0);
        self.ulooping = true;
        self.uspeed = speed;
        self.umask_root = mask_root;
        self.uweight_target = 1.0;
        if !was_blend {
            self.uweight_rate = if fade > 0.0001 {
                1.0 / fade
            } else {
                1_000_000.0
            };
        }
    }

    /// Fade the upper-body overlay back out over `fade` seconds (arms return to the base clip).
    pub fn stop_upper(&mut self, fade: f32) {
        self.uweight_target = 0.0;
        self.uweight_rate = if fade > 0.0001 {
            1.0 / fade
        } else {
            1_000_000.0
        };
    }

    /// The single clip + time currently dominating the upper overlay (the crossfade source): the
    /// higher-weighted clip of an aim blend, else the single overlay clip.
    fn upper_dominant(&self) -> (usize, f32) {
        if self.ublend_on && self.ublend >= 0.5 {
            (self.uclip2, self.utime)
        } else {
            (self.uclip, self.utime)
        }
    }

    /// Jump the upper-overlay playback to `t` seconds (e.g. skip a clip's wind-up). Clamped >= 0.
    pub fn seek_upper(&mut self, t: f32) {
        self.utime = t.max(0.0);
    }

    /// Jump the BASE clip to `t` seconds, cancelling any crossfade in progress.
    ///
    /// For state that is already true when you first see it: a player who went down ten seconds
    /// ago should be lying on the floor, not starting to fall over again, and a replicated body
    /// arrives mid-animation by definition. Playing from zero is a lie about when it happened.
    ///
    /// The crossfade is cancelled because a fade blends from where the PREVIOUS clip was, and
    /// after a deliberate jump that is a pose the game explicitly did not ask for.
    pub fn seek(&mut self, t: f32) {
        self.time = t.max(0.0);
        self.blend = 1.0;
    }

    /// Advance playback (and any crossfade) by `dt` seconds.
    /// Set a per-bone pose override: an extra local rotation pre-multiplied onto `joint` after the
    /// clip pose. Replaces any existing override for that joint. Call each frame; clear_pose() resets.
    pub fn set_pose(&mut self, joint: usize, q: Quat) {
        let j = joint as u32;
        for k in 0..self.pose_n {
            if self.pose[k].0 == j {
                self.pose[k].1 = q;
                return;
            }
        }
        if self.pose_n < self.pose.len() {
            self.pose[self.pose_n] = (j, q);
            self.pose_n += 1;
        }
    }

    /// Drop all per-bone pose overrides (back to the pure clip pose).
    pub fn clear_pose(&mut self) {
        self.pose_n = 0;
    }

    /// The ground the base layer covered during the last [`AnimPlayer::advance`],
    /// in the model's own space. Zero before the first one.
    pub fn root_delta(&self) -> Vec3 {
        self.root_delta
    }

    /// The rate the BASE clip's own clock runs at.
    ///
    /// In a sustained base blend, clip A (e.g. idle) keeps its natural cadence while clip B (e.g.
    /// run) is SPEED-WARPED by `speed`, so its footfalls track ground speed and don't slide. Plain
    /// single-clip playback uses `speed` directly. One function because [`AnimPlayer::advance`]
    /// and the crossfade snapshot must mean the same thing by it: the snapshot records the rate
    /// the outgoing clip WAS running at, and a second copy of this rule is how that would drift.
    fn base_speed(&self) -> f32 {
        if self.bblend_on {
            1.0
        } else {
            self.speed
        }
    }

    pub fn advance(&mut self, model: &Model, dt: f32) {
        let base_spd = self.base_speed();
        // Root motion is taken per clip over exactly the step each clip took, then blended by the
        // same weights the POSE is blended by. Measuring the blended root position instead would
        // fold the weight's own movement into the travel, and a crossfade between two clips whose
        // travel stands in different places would fling the character across the room in one frame.
        let step = advance_time(
            &mut self.time,
            model.clips.get(self.clip),
            dt * base_spd,
            self.looping,
        );
        let mut moved = travel(model, self.clip, step, self.time);
        // Weighted the same way, and in the same ORDER, as `sample_base_blend` composes the pose:
        // the sustained two-clip blend first, then the crossfade in from whatever came before it.
        if self.bblend_on {
            let step = advance_time(
                &mut self.btime2,
                model.clips.get(self.bclip2),
                dt * self.speed,
                true,
            );
            let b = travel(model, self.bclip2, step, self.btime2);
            moved = moved.lerp(b, self.bblend.clamp(0.0, 1.0));
        }
        if self.blend < 1.0 {
            // Keep the outgoing clip moving for a smooth blend - on ITS OWN terms. Hard-coding the
            // loop flag wrapped a played-out one-shot back to frame 0 (`d.rem_euclid(d)` is 0), so
            // the finished clip re-covered its opening ground every fade and the mesh visibly
            // restarted mid-blend. Held non-looping it clamps at its end and travels nothing.
            let step = advance_time(
                &mut self.prev_time,
                model.clips.get(self.prev_clip),
                dt * self.prev_speed,
                self.prev_looping,
            );
            self.blend = (self.blend + self.blend_rate * dt).min(1.0);
            let out = travel(model, self.prev_clip, step, self.prev_time);
            moved = out.lerp(moved, self.blend);
        }
        self.root_delta = moved;
        // The upper-body overlay takes no part in it: it is masked to the upper body by definition,
        // so a clip that cannot move the legs must not move the character either.
        if self.upper {
            advance_time(
                &mut self.utime,
                model.clips.get(self.uclip),
                dt * self.uspeed,
                self.ulooping,
            );
            if self.ufade < 1.0 {
                // Keep the outgoing overlay clip moving while it crossfades out, on its own loop
                // flag and rate for the same reason as the base layer above.
                advance_time(
                    &mut self.uprev_time,
                    model.clips.get(self.uprev_clip),
                    dt * self.uprev_speed,
                    self.uprev_looping,
                );
                self.ufade = (self.ufade + self.ufade_rate * dt).min(1.0);
            }
            if self.uweight < self.uweight_target {
                self.uweight = (self.uweight + self.uweight_rate * dt).min(self.uweight_target);
            } else if self.uweight > self.uweight_target {
                self.uweight = (self.uweight - self.uweight_rate * dt).max(self.uweight_target);
            }
            if self.uweight_target <= 0.0 && self.uweight <= 0.0 {
                self.upper = false;
            }
        }
    }

    /// Sample the sustained two-clip base blend (e.g. idle<->run by speed), additionally crossfading
    /// IN from the previous single clip while `blend` < 1 so entering the blend (e.g. on landing from
    /// a jump) eases in instead of popping.
    fn sample_base_blend(
        &self,
        skel: &Skeleton,
        model: &Model,
    ) -> (Vec<Vec3>, Vec<Quat>, Vec<Vec3>) {
        let (ta, ra, sa) = sample_locals(skel, model.clips.get(self.clip), self.time);
        let (tb, rb, sb) = sample_locals(skel, model.clips.get(self.bclip2), self.btime2);
        let w = self.bblend.clamp(0.0, 1.0);
        let mut t = ta;
        let mut r = ra;
        let mut s = sa;
        for i in 0..r.len() {
            t[i] = t[i].lerp(tb[i], w);
            r[i] = r[i].slerp(rb[i], w);
            s[i] = s[i].lerp(sb[i], w);
        }
        if self.blend < 1.0 {
            let (pt, pr, ps) = sample_locals(skel, model.clips.get(self.prev_clip), self.prev_time);
            let b = self.blend;
            for i in 0..r.len() {
                t[i] = pt[i].lerp(t[i], b);
                r[i] = pr[i].slerp(r[i], b);
                s[i] = ps[i].lerp(s[i], b);
            }
        }
        (t, r, s)
    }

    /// The skinning matrices for the current (possibly blended) pose. Empty if
    /// the model has no skeleton.
    pub fn matrices(&self, model: &Model, hidden: u64) -> Vec<Mat4> {
        let Some(skel) = &model.skeleton else {
            return Vec::new();
        };
        // Base (full-body) local pose, crossfaded if mid-transition.
        let (mut t, mut r, mut s) = if self.bblend_on {
            self.sample_base_blend(skel, model)
        } else if self.blend >= 1.0 {
            sample_locals(skel, model.clips.get(self.clip), self.time)
        } else {
            blended_locals(
                skel,
                model.clips.get(self.prev_clip),
                self.prev_time,
                model.clips.get(self.clip),
                self.time,
                self.blend,
            )
        };
        // Upper-body overlay: replace the masked joints' local TRS with the overlay clip's,
        // weighted by the fade. Lower body is untouched, so the legs keep the base locomotion.
        if self.upper && self.uweight > 0.001 {
            let (mut ut, mut ur, mut us) =
                sample_locals(skel, model.clips.get(self.uclip), self.utime);
            if self.ublend_on {
                // Blend a SECOND upper clip in (aim look up/down) before masking onto the body.
                let (ut2, ur2, us2) = sample_locals(skel, model.clips.get(self.uclip2), self.utime);
                let b = self.ublend.clamp(0.0, 1.0);
                for i in 0..ur.len() {
                    ut[i] = ut[i].lerp(ut2[i], b);
                    ur[i] = ur[i].slerp(ur2[i], b);
                    us[i] = us[i].lerp(us2[i], b);
                }
            }
            if self.ufade < 1.0 {
                // Crossfade FROM the previous overlay clip into this one (smooth katana<->aim<->reload).
                let (pt, pr, ps) =
                    sample_locals(skel, model.clips.get(self.uprev_clip), self.uprev_time);
                let f = self.ufade.clamp(0.0, 1.0);
                for i in 0..ur.len() {
                    ut[i] = pt[i].lerp(ut[i], f);
                    ur[i] = pr[i].slerp(ur[i], f);
                    us[i] = ps[i].lerp(us[i], f);
                }
            }
            let mask = upper_mask(skel, self.umask_root);
            let w = self.uweight.clamp(0.0, 1.0);
            for i in 0..skel.joints.len() {
                if mask[i] {
                    t[i] = t[i].lerp(ut[i], w);
                    r[i] = r[i].slerp(ur[i], w);
                    s[i] = s[i].lerp(us[i], w);
                }
            }
        }
        // Per-bone pose overrides (e.g. a slide): rotate each named joint further in its parent frame.
        for k in 0..self.pose_n {
            let (j, q) = self.pose[k];
            let j = j as usize;
            if j < r.len() {
                r[j] = q * r[j];
            }
        }
        locals_to_skin(skel, &t, &r, &s, hidden)
    }

    /// Model-space global transform of one joint in the CURRENT pose (NOT skinned - no
    /// inverse-bind). For attaching a prop (a weapon) to a bone: world = draw * this.
    pub fn joint_global(&self, model: &Model, joint: usize) -> Option<Mat4> {
        let skel = model.skeleton.as_ref()?;
        if joint >= skel.joints.len() {
            return None;
        }
        let (mut t, mut r, mut s) = if self.bblend_on {
            self.sample_base_blend(skel, model)
        } else if self.blend >= 1.0 {
            sample_locals(skel, model.clips.get(self.clip), self.time)
        } else {
            blended_locals(
                skel,
                model.clips.get(self.prev_clip),
                self.prev_time,
                model.clips.get(self.clip),
                self.time,
                self.blend,
            )
        };
        if self.upper && self.uweight > 0.001 {
            let (mut ut, mut ur, mut us) =
                sample_locals(skel, model.clips.get(self.uclip), self.utime);
            if self.ublend_on {
                // Blend a SECOND upper clip in (aim look up/down) before masking onto the body.
                let (ut2, ur2, us2) = sample_locals(skel, model.clips.get(self.uclip2), self.utime);
                let b = self.ublend.clamp(0.0, 1.0);
                for i in 0..ur.len() {
                    ut[i] = ut[i].lerp(ut2[i], b);
                    ur[i] = ur[i].slerp(ur2[i], b);
                    us[i] = us[i].lerp(us2[i], b);
                }
            }
            if self.ufade < 1.0 {
                // Crossfade FROM the previous overlay clip into this one (smooth katana<->aim<->reload).
                let (pt, pr, ps) =
                    sample_locals(skel, model.clips.get(self.uprev_clip), self.uprev_time);
                let f = self.ufade.clamp(0.0, 1.0);
                for i in 0..ur.len() {
                    ut[i] = pt[i].lerp(ut[i], f);
                    ur[i] = pr[i].slerp(ur[i], f);
                    us[i] = ps[i].lerp(us[i], f);
                }
            }
            let mask = upper_mask(skel, self.umask_root);
            let w = self.uweight.clamp(0.0, 1.0);
            for i in 0..skel.joints.len() {
                if mask[i] {
                    t[i] = t[i].lerp(ut[i], w);
                    r[i] = r[i].slerp(ur[i], w);
                    s[i] = s[i].lerp(us[i], w);
                }
            }
        }
        for k in 0..self.pose_n {
            let (j, q) = self.pose[k];
            let j = j as usize;
            if j < r.len() {
                r[j] = q * r[j];
            }
        }
        let n = skel.joints.len();
        let local: Vec<Mat4> = (0..n)
            .map(|i| Mat4::from_scale_rotation_translation(s[i], r[i], t[i]))
            .collect();
        let mut global: Vec<Option<Mat4>> = vec![None; n];
        resolve_global(skel, &local, joint, &mut global);
        global[joint]
    }
}

/// Advance `time` by `dt`, wrapping a looping clip.
///
/// Returns the step it took: `(from, laps)`, where `from` is the point on the clip the step STARTED
/// at and `laps` is how many times the clip's end was crossed (negative when played backwards).
///
/// Both are here rather than recomputed by the caller because this is the function that decides
/// where the clock lands, and root motion has to measure the same step the pose took. A wrap makes
/// the time jump backwards; read off the two clock readings alone that is a character yanked back to
/// the start of the clip, and the lap count is what turns it into the extra pass of travel it really
/// is. `from` is normalised for the same reason: a seek can leave the clock a hundred laps past the
/// end of a loop, and a step measured from there is a teleport.
fn advance_time(
    time: &mut f32,
    clip: Option<&crate::model::Clip>,
    dt: f32,
    looping: bool,
) -> (f32, f32) {
    let mut from = *time;
    let mut laps = 0.0;
    match clip.map(|c| c.duration) {
        Some(d) if d > 0.0 && looping => {
            from = from.rem_euclid(d);
            *time = from + dt;
            laps = (*time / d).floor();
            *time = time.rem_euclid(d);
        }
        Some(d) if d > 0.0 => {
            *time = (*time + dt).min(d);
        }
        _ => *time += dt,
    }
    (from, laps)
}

/// Whole passes of a clip that one update may pay out as travel.
///
/// `dt` is a delta in the caller's hands - a stalled frame, a script fast-forwarding a clip to pose
/// it - so the lap count is unbounded input, and a looping clip multiplies it by a whole stride. A
/// hitch that lands a character slightly short of where the clock says is recoverable; a character
/// flung through a wall and out of the level is not. Four passes is far past any real frame time at
/// any sane playback rate, so nothing that is merely slow is ever clipped by it.
const MAX_PASSES_PER_UPDATE: f32 = 4.0;

/// The root motion `clip` covered over the `step` [`advance_time`] just took, ending at `to`.
///
/// Zero for a clip that authors no travel, which is most of them.
///
/// The bound is applied HERE and not in [`advance_time`]: the clock stays truthful about where the
/// animation really is, only the distance paid out is capped. The partial-lap term is untouched, so
/// every normal step is exactly what it always was.
fn travel(model: &Model, clip: usize, step: (f32, f32), to: f32) -> Vec3 {
    let (from, laps) = step;
    let Some(c) = model.clips.get(clip) else {
        return Vec3::ZERO;
    };
    if c.root.is_none() {
        return Vec3::ZERO;
    }
    let laps = laps.clamp(-MAX_PASSES_PER_UPDATE, MAX_PASSES_PER_UPDATE);
    c.root_pos(to) - c.root_pos(from) + c.root_pass() * laps
}

/// Sample a clip at `time` into per-joint local TRS, starting from the
/// skeleton's defaults.
fn sample_locals(
    skel: &Skeleton,
    clip: Option<&crate::model::Clip>,
    time: f32,
) -> (Vec<Vec3>, Vec<Quat>, Vec<Vec3>) {
    skel.sample(clip, time)
}

/// Turn per-joint local TRS into skinning matrices (`global * inverse_bind`). Joints whose bit is
/// set in `hidden` are COLLAPSED: their matrix maps every bound vertex to the joint's own world
/// position, so geometry exclusive to them shrinks to a point AT THE BONE (a nearby spot in the
/// body), and seam vertices shared with a visible joint only pull a little toward that bone -
/// instead of streaking to the far model origin. Used for first-person arms (hide torso/head/legs).
fn locals_to_skin(skel: &Skeleton, t: &[Vec3], r: &[Quat], s: &[Vec3], hidden: u64) -> Vec<Mat4> {
    let n = skel.joints.len();
    let local: Vec<Mat4> = (0..n)
        .map(|i| Mat4::from_scale_rotation_translation(s[i], r[i], t[i]))
        .collect();
    let mut global: Vec<Option<Mat4>> = vec![None; n];
    for i in 0..n {
        resolve_global(skel, &local, i, &mut global);
    }
    (0..n)
        .map(|i| {
            let g = global[i].unwrap_or(Mat4::IDENTITY);
            if i < 64 && (hidden >> i) & 1 == 1 {
                // Collapse to the bone position: zero linear part, translation = bone world pos.
                Mat4::from_cols(Vec4::ZERO, Vec4::ZERO, Vec4::ZERO, g.w_axis)
            } else {
                g * skel.joints[i].inverse_bind
            }
        })
        .collect()
}

/// Pose `skel` from `clip` at `time` and return per-joint skinning matrices.
pub fn skin_matrices(skel: &Skeleton, clip: Option<&crate::model::Clip>, time: f32) -> Vec<Mat4> {
    let (t, r, s) = sample_locals(skel, clip, time);
    locals_to_skin(skel, &t, &r, &s, 0)
}

/// Blend two clips' poses by weight `w` (0 = clip a, 1 = clip b) and return the
/// skinning matrices. Blends in local TRS space (correct), not matrix space.
/// Blend two clips' poses into per-joint local TRS by weight `w` (0 = a, 1 = b).
fn blended_locals(
    skel: &Skeleton,
    a: Option<&crate::model::Clip>,
    ta: f32,
    b: Option<&crate::model::Clip>,
    tb: f32,
    w: f32,
) -> (Vec<Vec3>, Vec<Quat>, Vec<Vec3>) {
    let (at, ar, asc) = sample_locals(skel, a, ta);
    let (bt, br, bsc) = sample_locals(skel, b, tb);
    let w = w.clamp(0.0, 1.0);
    let n = skel.joints.len();
    let t: Vec<Vec3> = (0..n).map(|i| at[i].lerp(bt[i], w)).collect();
    let r: Vec<Quat> = (0..n).map(|i| ar[i].slerp(br[i], w)).collect();
    let s: Vec<Vec3> = (0..n).map(|i| asc[i].lerp(bsc[i], w)).collect();
    (t, r, s)
}

/// Blend two clips' poses by weight `w` (0 = clip a, 1 = clip b) and return the skinning
/// matrices. Blends in local TRS space (correct), not matrix space.
pub fn skin_matrices_blended(
    skel: &Skeleton,
    a: Option<&crate::model::Clip>,
    ta: f32,
    b: Option<&crate::model::Clip>,
    tb: f32,
    w: f32,
) -> Vec<Mat4> {
    let (t, r, s) = blended_locals(skel, a, ta, b, tb, w);
    locals_to_skin(skel, &t, &r, &s, 0)
}

/// Mask of joints that are `root` or descend from it (the upper-body overlay set).
fn upper_mask(skel: &Skeleton, root: usize) -> Vec<bool> {
    let n = skel.joints.len();
    let mut mask = vec![false; n];
    for (i, slot) in mask.iter_mut().enumerate() {
        let mut j = i;
        loop {
            if j == root {
                *slot = true;
                break;
            }
            match skel.joints[j].parent {
                Some(p) if p != j => j = p,
                _ => break,
            }
        }
    }
    mask
}

fn resolve_global(
    skel: &Skeleton,
    local: &[Mat4],
    i: usize,
    global: &mut Vec<Option<Mat4>>,
) -> Mat4 {
    if let Some(g) = global[i] {
        return g;
    }
    let g = match skel.joints[i].parent {
        Some(p) if p != i => resolve_global(skel, local, p, global) * local[i],
        // A root joint's parent is the node the skeleton hangs off, NOT the world:
        // glTF resolves a joint's global transform through the whole node tree.
        // A parentless joint's local already includes every non-joint ancestor above
        // it (the armature and any helper nodes), folded in at load.
        _ => local[i],
    };
    global[i] = Some(g);
    g
}

// Clip sampling and key interpolation live in `aurora-asset`, beside the clip
// format itself: see `Skeleton::sample`. Keeping a second copy here is how a
// renderer and an importer quietly start disagreeing about what a clip means.

#[cfg(test)]
mod play_is_idempotent {
    use super::*;

    // The bug this pins: `play` restarted unconditionally, so a frame loop that
    // said "the character is idle" every frame re-seeded the clock every frame.
    // The character stood frozen on frame 0, jittering, and no crossfade ever
    // finished. Every symptom of a broken animation layer from one line.
    #[test]
    fn replaying_the_current_clip_does_not_rewind_it() {
        let mut p = AnimPlayer::default();
        p.play(3, true, 1.0, 0.0);
        p.time = 0.4;
        p.play(3, true, 1.0, 0.0);
        assert_eq!(p.time, 0.4, "a frame loop restating its state must not rewind");
    }

    #[test]
    fn a_live_speed_change_still_takes_effect() {
        let mut p = AnimPlayer::default();
        p.play(3, true, 1.0, 0.0);
        p.time = 0.4;
        p.play(3, true, 2.0, 0.0);
        assert_eq!(p.speed, 2.0, "a windup stretched to its frame data must retune");
        assert_eq!(p.time, 0.4);
    }

    #[test]
    fn a_different_clip_does_start_from_the_top() {
        let mut p = AnimPlayer::default();
        p.play(3, true, 1.0, 0.0);
        p.time = 0.4;
        p.play(5, true, 1.0, 0.0);
        assert_eq!(p.clip, 5);
        assert_eq!(p.time, 0.0);
    }

    // Changing looping is a different state, not the same one: a one-shot death
    // and a looping idle on one clip must not be confused.
    #[test]
    fn changing_looping_restarts() {
        let mut p = AnimPlayer::default();
        p.play(3, true, 1.0, 0.0);
        p.time = 0.4;
        p.play(3, false, 1.0, 0.0);
        assert_eq!(p.time, 0.0);
    }

    // And the deliberate replay still works - the combo case.
    #[test]
    fn restart_replays_the_same_clip_from_the_top() {
        let mut p = AnimPlayer::default();
        p.play(3, false, 1.0, 0.0);
        p.time = 0.4;
        p.restart(3, false, 1.0, 0.0);
        assert_eq!(p.time, 0.0, "the second swing of a combo must actually replay");
    }
}

#[cfg(test)]
mod root_motion {
    use super::*;
    use crate::model::{Interp, Joint, Model, RootMotion};

    /// A model with one clip per entry: `(duration, travel over the clip)`.
    /// `None` is a clip authored in place, like every locomotion loop.
    fn model(clips: &[(f32, Option<Vec3>)]) -> Model {
        Model {
            primitives: Vec::new(),
            skeleton: Some(Skeleton {
                joints: vec![Joint {
                    parent: None,
                    inverse_bind: Mat4::IDENTITY,
                    t: Vec3::ZERO,
                    r: Quat::IDENTITY,
                    s: Vec3::ONE,
                    name: "Root".into(),
                }],
            }),
            clips: clips
                .iter()
                .enumerate()
                .map(|(i, (duration, travel))| crate::model::Clip {
                    name: format!("clip{i}"),
                    duration: *duration,
                    channels: Vec::new(),
                    // A straight line from the origin over the clip's length, so
                    // the travel expected of any interval is exactly its share.
                    root: travel.map(|v| RootMotion {
                        interp: Interp::Linear,
                        times: vec![0.0, *duration],
                        values: vec![0.0, 0.0, 0.0, v.x, v.y, v.z],
                    }),
                })
                .collect(),
        }
    }

    fn close(a: Vec3, b: Vec3) -> bool {
        (a - b).length() < 1e-4
    }

    #[test]
    fn nothing_has_moved_before_the_first_update() {
        let mut p = AnimPlayer::default();
        assert_eq!(p.root_delta(), Vec3::ZERO, "a fresh player has not moved");
        // Nor does asking for a travelling clip move anything on its own: the
        // delta is what an UPDATE covered, so it stays zero until one happens.
        p.play(0, false, 1.0, 0.0);
        assert_eq!(p.root_delta(), Vec3::ZERO, "playing a clip is not advancing it");
    }

    // The number the game acts on: over a known slice of the clip, the delta is
    // the ground the animator laid down over that slice. Not a fraction of it,
    // not a speed guessed to match.
    #[test]
    fn the_delta_is_the_ground_the_clip_covers() {
        let m = model(&[(2.0, Some(Vec3::new(0.0, 0.0, 3.0)))]);
        let mut p = AnimPlayer::default();
        p.play(0, false, 1.0, 0.0);
        p.advance(&m, 0.5);
        assert!(
            close(p.root_delta(), Vec3::new(0.0, 0.0, 0.75)),
            "a quarter of a 3 m clip is 0.75 m, got {:?}",
            p.root_delta()
        );
        p.advance(&m, 1.5);
        assert!(
            close(p.root_delta(), Vec3::new(0.0, 0.0, 2.25)),
            "the rest of it is 2.25 m, got {:?}",
            p.root_delta()
        );
    }

    // The whole clip, integrated one frame at a time, is the whole distance -
    // and a clip played at double speed covers it in half the time.
    #[test]
    fn the_steps_sum_to_the_authored_distance() {
        let m = model(&[(1.0, Some(Vec3::new(1.0, 0.0, 4.0)))]);
        for speed in [0.5, 1.0, 2.0] {
            let mut p = AnimPlayer::default();
            p.play(0, false, speed, 0.0);
            let mut total = Vec3::ZERO;
            for _ in 0..200 {
                p.advance(&m, 1.0 / 60.0);
                total += p.root_delta();
            }
            assert!(
                close(total, Vec3::new(1.0, 0.0, 4.0)),
                "at speed {speed} the clip should still cover its 4 m, got {total:?}"
            );
        }
    }

    // A DELTA, not a total. It is added to a position the caller owns, so a
    // running total would double every distance from the second frame on.
    #[test]
    fn each_update_reports_only_its_own_step() {
        let m = model(&[(4.0, Some(Vec3::new(0.0, 0.0, 4.0)))]);
        let mut p = AnimPlayer::default();
        p.play(0, false, 1.0, 0.0);
        for step in 1..=3 {
            p.advance(&m, 1.0);
            assert!(
                close(p.root_delta(), Vec3::new(0.0, 0.0, 1.0)),
                "step {step} covered one metre, got {:?}",
                p.root_delta()
            );
        }
    }

    // An update that moves nothing reports nothing, rather than repeating the
    // last step for ever.
    #[test]
    fn a_clip_that_has_played_out_stops_reporting_travel() {
        let m = model(&[(1.0, Some(Vec3::new(0.0, 0.0, 1.0)))]);
        let mut p = AnimPlayer::default();
        p.play(0, false, 1.0, 0.0);
        p.advance(&m, 1.0);
        p.advance(&m, 1.0);
        assert_eq!(p.root_delta(), Vec3::ZERO, "a finished one-shot travels no further");
    }

    #[test]
    fn a_clip_authored_in_place_never_moves_the_character() {
        let m = model(&[(1.0, None)]);
        let mut p = AnimPlayer::default();
        p.play(0, true, 1.0, 0.0);
        for _ in 0..120 {
            p.advance(&m, 1.0 / 60.0);
            assert_eq!(p.root_delta(), Vec3::ZERO, "a walk cycle's ground speed is the game's");
        }
    }

    // The bug a naive difference of two times would produce: at the wrap the
    // clock jumps back to zero, and the character would be yanked a whole clip's
    // travel BACKWARDS in a single frame.
    #[test]
    fn a_loop_carries_travel_forward_across_the_wrap() {
        let m = model(&[(1.0, Some(Vec3::new(0.0, 0.0, 2.0)))]);
        let mut p = AnimPlayer::default();
        p.play(0, true, 1.0, 0.0);
        p.advance(&m, 0.75);
        p.advance(&m, 0.5); // 0.75 -> 1.25, wrapping to 0.25
        assert!(
            close(p.root_delta(), Vec3::new(0.0, 0.0, 1.0)),
            "half a lap is a metre forward however the clock wrapped, got {:?}",
            p.root_delta()
        );
        // Several laps in one step count every one of them.
        let mut q = AnimPlayer::default();
        q.play(0, true, 1.0, 0.0);
        q.advance(&m, 2.5);
        assert!(
            close(q.root_delta(), Vec3::new(0.0, 0.0, 5.0)),
            "two and a half laps of a 2 m loop is 5 m, got {:?}",
            q.root_delta()
        );
    }

    // Played backwards, the character backs up: the same wrap arithmetic with
    // the sign the clock actually took.
    #[test]
    fn a_clip_played_backwards_travels_backwards() {
        let m = model(&[(1.0, Some(Vec3::new(0.0, 0.0, 2.0)))]);
        let mut p = AnimPlayer::default();
        p.play(0, true, -1.0, 0.0);
        p.advance(&m, 0.25); // 0.0 -> -0.25, wrapping to 0.75
        assert!(
            close(p.root_delta(), Vec3::new(0.0, 0.0, -0.5)),
            "a quarter lap backwards is half a metre back, got {:?}",
            p.root_delta()
        );
    }

    // A crossfade weights the travel the way it weights the pose. Measuring the
    // blended root POSITION instead would read the gap between two clips' root
    // tracks as one frame of movement, and fling the character across the room.
    #[test]
    fn a_crossfade_weights_travel_like_it_weights_the_pose() {
        let m = model(&[
            (1.0, Some(Vec3::new(0.0, 0.0, 10.0))),
            (1.0, Some(Vec3::new(0.0, 0.0, 0.0))),
        ]);
        let mut p = AnimPlayer::default();
        p.play(0, true, 1.0, 0.0);
        p.advance(&m, 0.5); // stands half a lap along clip 0's track
        p.play(1, true, 1.0, 1.0); // fade to a clip whose track sits at the origin
        p.advance(&m, 0.5);
        // The authored answer, not a band: the outgoing loop covers its second half-lap (5 m)
        // and the fade is half done, so half of it lands. A band wide enough to be safe is
        // wide enough to pass with the travel fabricated, which is exactly what happened.
        assert!(
            close(p.root_delta(), Vec3::new(0.0, 0.0, 2.5)),
            "half of the outgoing loop's 5 m half-lap is 2.5 m, got {:?}",
            p.root_delta()
        );
    }

    // A one-shot that has PLAYED OUT is standing still, and standing still is worth no ground -
    // during its fade least of all.
    //
    // The bug: the crossfade advanced the outgoing clip with `looping` hard-coded true, because
    // the player recorded the incoming clip's flag and nothing else. A finished one-shot sits at
    // `time == duration`, and the looping arm normalises that to 0.0, so a played-out roll spent
    // the whole fade re-covering its own opening ground - weighted in, compounding once per
    // action. Before root motion this only mis-posed a fading mesh; it moves the character now.
    #[test]
    fn a_played_out_one_shot_pays_no_travel_while_it_fades() {
        let m = model(&[(0.8, Some(Vec3::new(0.0, 0.0, 3.0))), (1.0, None)]);
        let mut p = AnimPlayer::default();
        p.play(0, false, 1.0, 0.0);
        let mut total = Vec3::ZERO;
        for _ in 0..60 {
            p.advance(&m, 1.0 / 60.0);
            total += p.root_delta();
        }
        assert!(close(total, Vec3::new(0.0, 0.0, 3.0)), "the roll arrives: {total:?}");

        // Now fade it into a clip authored in place. Every frame of that fade must be zero -
        // exactly zero, because the idle covers no ground and the roll has none left to give.
        p.play(1, true, 1.0, 0.2);
        for frame in 0..24 {
            p.advance(&m, 1.0 / 60.0);
            total += p.root_delta();
            assert_eq!(
                p.root_delta(),
                Vec3::ZERO,
                "frame {frame} of the fade out of a finished roll fabricated travel"
            );
            // The same fact the POSE is sampled from: the outgoing clip holds its last frame
            // instead of restarting at frame 0 halfway through the blend.
            assert_eq!(p.prev_time, 0.8, "the fading clip must hold its end");
        }
        assert!(
            close(total, Vec3::new(0.0, 0.0, 3.0)),
            "one roll is one roll's distance however it was faded out, got {total:?}"
        );
    }

    // The rest of the same defect, which travel alone cannot see. A crossfade source is a whole
    // playback state, and two of its four facts show only in where the fading MESH stands: the
    // outgoing clip runs at ITS OWN speed rather than the incoming clip's, and the overlay layer
    // has the identical crossfade with the identical hole in it.
    #[test]
    fn the_crossfade_source_keeps_its_own_speed_and_loop_flag() {
        let m = model(&[(1.0, None), (1.0, None)]);
        let mut p = AnimPlayer::default();
        p.play(0, true, 2.0, 0.0);
        p.advance(&m, 0.1); // 0.2 s in, because it is running at double speed
        assert!((p.time - 0.2).abs() < 1e-6, "got {}", p.time);
        p.play(1, true, 0.5, 1.0); // fade into a clip playing at half speed
        p.advance(&m, 0.1);
        assert!(
            (p.prev_time - 0.4).abs() < 1e-6,
            "the outgoing clip must keep its own 2x while it fades, got {}",
            p.prev_time
        );

        // The overlay: a played-out one-shot (a swing) crossfading back to a looping aim must
        // hold its last frame, not snap back to frame 0 halfway through the blend.
        let mut q = AnimPlayer::default();
        q.play(0, true, 1.0, 0.0);
        q.play_upper(1, false, 1.0, 0.0, 0);
        q.advance(&m, 1.5);
        assert_eq!(q.utime, 1.0, "the overlay one-shot has played out");
        q.play_upper(0, true, 1.0, 0.2, 0);
        q.advance(&m, 1.0 / 60.0);
        assert_eq!(
            q.uprev_time, 1.0,
            "the fading overlay must hold its end, not restart at frame 0"
        );
    }

    // The other half of that: an INTERRUPTED one-shot has ground left, and must still pay it,
    // weighted by the fade. "Outgoing clips contribute nothing" would be a different bug.
    #[test]
    fn an_interrupted_one_shot_still_pays_its_remaining_travel() {
        let m = model(&[(1.0, Some(Vec3::new(0.0, 0.0, 4.0))), (1.0, None)]);
        let mut p = AnimPlayer::default();
        p.play(0, false, 1.0, 0.0);
        p.advance(&m, 0.25); // a quarter in, 1 m covered
        p.play(1, true, 1.0, 1.0); // cancelled into an in-place clip over a 1 s fade
        p.advance(&m, 0.5);
        // The roll covers 0.25 -> 0.75 of its track (2 m) and the fade is half done: 1 m.
        assert!(
            close(p.root_delta(), Vec3::new(0.0, 0.0, 1.0)),
            "half of the cancelled roll's remaining 2 m step is 1 m, got {:?}",
            p.root_delta()
        );
    }

    // One long dt must not teleport the character. `dt` belongs to the caller - a stalled frame,
    // a script fast-forwarding a clip - so the lap count is unbounded input, and a looping clip
    // multiplies it by a whole stride.
    #[test]
    fn a_hitch_cannot_pay_out_more_than_a_few_whole_passes() {
        let m = model(&[(1.0, Some(Vec3::new(0.0, 0.0, 2.0)))]);
        let mut p = AnimPlayer::default();
        p.play(0, true, 1.0, 0.0);
        p.advance(&m, 100.0); // a hundred laps of a 2 m loop
        assert!(
            close(p.root_delta(), Vec3::new(0.0, 0.0, 2.0 * MAX_PASSES_PER_UPDATE)),
            "a hitch pays at most {MAX_PASSES_PER_UPDATE} passes, got {:?}",
            p.root_delta()
        );
        // The CLOCK is untouched by the bound: the animation is where the time says it is,
        // only the distance is capped.
        assert_eq!(p.time, 0.0, "a hundred whole laps land back at the start");
        // And backwards, for the same reason and with the sign it took.
        let mut q = AnimPlayer::default();
        q.play(0, true, -1.0, 0.0);
        q.advance(&m, 100.0);
        assert!(
            close(q.root_delta(), Vec3::new(0.0, 0.0, -2.0 * MAX_PASSES_PER_UPDATE)),
            "backwards too, got {:?}",
            q.root_delta()
        );
    }

    // The same for a sustained idle<->run blend: at weight 1 the character
    // covers the running clip's ground, at 0 it covers none.
    #[test]
    fn a_sustained_blend_weights_travel_by_its_own_weight() {
        let m = model(&[(1.0, None), (1.0, Some(Vec3::new(0.0, 0.0, 4.0)))]);
        let step = 0.25;
        for (weight, want) in [(0.0, 0.0), (0.5, 0.5), (1.0, 1.0)] {
            let mut p = AnimPlayer::default();
            p.blend(0, 1, weight, 1.0, 0.0);
            p.advance(&m, step);
            assert!(
                close(p.root_delta(), Vec3::new(0.0, 0.0, 4.0 * step * want)),
                "at weight {weight} the step should be {} m, got {:?}",
                4.0 * step * want,
                p.root_delta()
            );
        }
    }

    // An upper-body overlay is masked to the upper body: it cannot move the legs,
    // so it must not move the character either.
    #[test]
    fn an_upper_body_overlay_contributes_no_travel() {
        let m = model(&[(1.0, None), (1.0, Some(Vec3::new(0.0, 0.0, 6.0)))]);
        let mut p = AnimPlayer::default();
        p.play(0, true, 1.0, 0.0);
        p.play_upper(1, true, 1.0, 0.0, 0);
        p.advance(&m, 0.5);
        assert_eq!(p.root_delta(), Vec3::ZERO, "the arms do not carry the body");
    }

    // A seek can leave the clock far past the end of a loop. The step taken from
    // there is one frame of a clip, not the hundred laps the raw clock readings
    // would suggest - which as root motion would be a teleport across the level.
    #[test]
    fn a_seek_past_the_end_of_a_loop_still_steps_once() {
        let m = model(&[(1.0, Some(Vec3::new(0.0, 0.0, 2.0)))]);
        let mut p = AnimPlayer::default();
        p.play(0, true, 1.0, 0.0);
        p.seek(100.0);
        p.advance(&m, 0.25);
        assert!(
            close(p.root_delta(), Vec3::new(0.0, 0.0, 0.5)),
            "a quarter lap is half a metre wherever the clock was, got {:?}",
            p.root_delta()
        );
    }

    // A handle whose clip index is stale must not panic or invent movement.
    #[test]
    fn a_clip_that_is_not_there_reports_no_travel() {
        let m = model(&[(1.0, Some(Vec3::new(0.0, 0.0, 1.0)))]);
        let mut p = AnimPlayer::default();
        p.play(7, false, 1.0, 0.0);
        p.advance(&m, 0.5);
        assert_eq!(p.root_delta(), Vec3::ZERO);
    }
}
