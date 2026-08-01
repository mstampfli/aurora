//! Server-side lag compensation (netcode spec §7).
//!
//! The server keeps a short history of every entity's collider position per
//! tick. When a client fires, it sends the tick it was *seeing*; the server
//! rewinds colliders to that tick and tests the shot against where targets
//! actually were on the firer's screen — not where they are "now" after network
//! latency. This is the fair-hitreg primitive a milsim shooter needs.
//!
//! Math is plain `[f32; 3]`; the rewound test is a ray–sphere intersection
//! against each entity's collider at the requested tick.

use std::collections::HashMap;

pub type V3 = [f32; 3];

fn sub(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn dot(a: V3, b: V3) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[derive(Clone, Copy, Debug)]
struct Snapshot {
    tick: u64,
    pos: V3,
    radius: f32,
    /// Cylinder half-height of the collider's capsule (0 = a plain sphere, used
    /// for crates). A character is a vertical capsule so its rewound hitbox
    /// matches the capsule the client actually raycasts - no "server hit, client
    /// missed" on a side graze that a fat sphere would have caught.
    half_h: f32,
}

/// A hit produced by a rewound raycast.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Hit {
    pub entity: u64,
    /// Distance along the ray direction to the entry point.
    pub distance: f32,
}

/// Per-entity ring of recent collider snapshots.
pub struct LagComp {
    /// How many ticks of history to retain.
    window: u64,
    latest_tick: u64,
    hist: HashMap<u64, Vec<Snapshot>>,
}

impl LagComp {
    pub fn new(window: u64) -> LagComp {
        LagComp {
            window,
            latest_tick: 0,
            hist: HashMap::new(),
        }
    }

    /// Record an entity's collider position for `tick`. `half_h` is the capsule
    /// cylinder half-height (0 = sphere).
    pub fn record(&mut self, tick: u64, entity: u64, pos: V3, radius: f32, half_h: f32) {
        self.latest_tick = self.latest_tick.max(tick);
        let ring = self.hist.entry(entity).or_default();
        ring.push(Snapshot {
            tick,
            pos,
            radius,
            half_h,
        });
        // Evict snapshots older than the retention window.
        let cutoff = self.latest_tick.saturating_sub(self.window);
        ring.retain(|s| s.tick >= cutoff);
    }

    /// Forget an entity completely, dropping its snapshot ring.
    ///
    /// `record` only ever evicts inside the ring it is writing, so an entity that
    /// stops being recorded keeps its whole `Vec` for the life of the process. On a
    /// server that stays up while players rotate through it that is unbounded growth,
    /// and it is a correctness bug too: `raycast_at_tick` scans every entry in `hist`,
    /// so a departed player's last recorded capsule lingers as an invisible hitbox
    /// that rewound shots keep hitting. Call this from the actual departure event -
    /// a timeout, a leave, a retired slot - never from a timer that guesses.
    pub fn remove(&mut self, entity: u64) {
        self.hist.remove(&entity);
    }

    /// How many entities currently have a history ring. The bound this structure is
    /// supposed to hold: it must track live entities, not every entity ever seen.
    pub fn tracked_entities(&self) -> usize {
        self.hist.len()
    }

    /// The collider position of `entity` as of `tick` (the most recent snapshot
    /// at or before `tick`).
    pub fn position_at_tick(&self, entity: u64, tick: u64) -> Option<V3> {
        self.snapshot_at(entity, tick).map(|s| s.pos)
    }

    fn snapshot_at(&self, entity: u64, tick: u64) -> Option<Snapshot> {
        let ring = self.hist.get(&entity)?;
        ring.iter()
            .filter(|s| s.tick <= tick)
            .max_by_key(|s| s.tick)
            .copied()
    }

    /// Cast a ray from `origin` along (normalized) `dir`, rewinding all colliders
    /// to `tick`. Returns the nearest hit, if any. `ignore` skips the shooter.
    pub fn raycast_at_tick(&self, origin: V3, dir: V3, tick: u64, ignore: u64) -> Option<Hit> {
        let mut best: Option<Hit> = None;
        for (&entity, ring) in &self.hist {
            if entity == ignore {
                continue;
            }
            // Newest snapshot at or before the firer's (rewound) view tick. If the view PREDATES this
            // entity's whole history - it just spawned and the firer's view is RTT-old, or its oldest
            // snapshots were evicted - clamp to the EARLIEST known position (where it was when it
            // appeared = what the firer is seeing) instead of missing. Fixes "first hits don't go
            // through" on a fresh spawn. (position_at_tick stays strict; only the shot ray clamps.)
            let snap = match self.snapshot_at(entity, tick) {
                Some(s) => s,
                None => match ring.iter().min_by_key(|s| s.tick) {
                    Some(s) => *s,
                    None => continue,
                },
            };
            if let Some(distance) = ray_capsule(origin, dir, snap.pos, snap.radius, snap.half_h) {
                if best.is_none_or(|b| distance < b.distance) {
                    best = Some(Hit { entity, distance });
                }
            }
        }
        best
    }

    /// A melee swing, rewound: everything within `reach` of `origin` and inside
    /// an arc of `arc_degrees` about `facing`, as of `tick`.
    ///
    /// A sword is not a ray. Validating a swing with `raycast_at_tick` asks
    /// whether an infinitely thin line passed through a target, which misses the
    /// case melee is mostly made of - a wide blade sweeping past a body slightly
    /// off the centre line - and cannot express a cleave through two of them at
    /// once. Reach and arc are what the frame data actually specifies, so they
    /// are what the server checks.
    ///
    /// The arc is measured in the horizontal plane. A swing that connects or
    /// misses depending on whether the target stood on a step would be a rule no
    /// player could see; height is handled by the capsule test, which is what
    /// height is for.
    ///
    /// Returns every hit, nearest first. Melee cleaves - refusing to report the
    /// second target would make "how many can one swing hit" a property of this
    /// function rather than of the weapon.
    pub fn melee_at_tick(
        &self,
        origin: V3,
        facing: V3,
        reach: f32,
        arc_degrees: f32,
        tick: u64,
        ignore: u64,
    ) -> Vec<Hit> {
        let mut hits: Vec<Hit> = Vec::new();
        if reach <= 0.0 {
            return hits;
        }
        // A swing with no width still has its centre line, and one at or past a
        // full circle covers everything: clamping here keeps both ends sane
        // rather than letting a negative cosine invert the test.
        let half = (arc_degrees.max(0.0) * 0.5).min(180.0).to_radians();
        let min_cos = half.cos();

        // Flattened once. A facing of straight up has no horizontal direction to
        // compare against, so every target would be "in front"; that is a caller
        // error, and answering nothing is the safe reading of it.
        let (fx, fz) = match normalize_xz(facing[0], facing[2]) {
            Some(v) => v,
            None => return hits,
        };

        for (&entity, ring) in &self.hist {
            if entity == ignore {
                continue;
            }
            let snap = match self.snapshot_at(entity, tick) {
                Some(s) => s,
                None => match ring.iter().min_by_key(|s| s.tick) {
                    Some(s) => *s,
                    None => continue,
                },
            };

            // Distance to the capsule's surface, so reach means reach from the
            // body rather than from a point inside it. A big target is easier to
            // touch, which is what being big should mean.
            let distance = point_capsule_distance(origin, snap.pos, snap.radius, snap.half_h);
            if distance > reach {
                continue;
            }

            // Inside the arc, measured to the target's centre. A target close
            // enough to be overlapping the swing's origin has no meaningful
            // direction, and refusing it there would make the swing miss the one
            // thing it is definitely touching.
            match normalize_xz(snap.pos[0] - origin[0], snap.pos[2] - origin[2]) {
                Some((dx, dz)) => {
                    if dx * fx + dz * fz < min_cos {
                        continue;
                    }
                }
                None => {}
            }

            hits.push(Hit { entity, distance });
        }

        // Nearest first, then by entity so two targets at equal distance always
        // come back in the same order - a swing that resolved differently between
        // two servers on a tie would desync the fight.
        hits.sort_by(|a, b| {
            a.distance
                .partial_cmp(&b.distance)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.entity.cmp(&b.entity))
        });
        hits
    }
}

/// A horizontal direction, or `None` when there is not one.
fn normalize_xz(x: f32, z: f32) -> Option<(f32, f32)> {
    let len = (x * x + z * z).sqrt();
    if len <= 1e-6 {
        return None;
    }
    Some((x / len, z / len))
}

/// Shortest distance from a point to a vertical capsule's surface, clamped at 0
/// when the point is inside it.
fn point_capsule_distance(point: V3, center: V3, radius: f32, half_h: f32) -> f32 {
    // Nearest point on the capsule's spine, then treat the capsule as that point
    // inflated by its radius.
    let y = (point[1]).clamp(center[1] - half_h, center[1] + half_h);
    let dx = point[0] - center[0];
    let dy = point[1] - y;
    let dz = point[2] - center[2];
    let to_axis = (dx * dx + dy * dy + dz * dz).sqrt();
    (to_axis - radius).max(0.0)
}

/// Ray vs a vertical (Y-axis) capsule: a cylinder of `radius`/`half_h` capped by
/// two hemispheres. `half_h == 0` degenerates to a sphere (crates). Returns the
/// nearest non-negative entry distance. This is the shape a character collider
/// actually is, so the rewound server test agrees with the client's own raycast.
fn ray_capsule(origin: V3, dir: V3, center: V3, radius: f32, half_h: f32) -> Option<f32> {
    if half_h <= 0.0 {
        return ray_sphere(origin, dir, center, radius);
    }
    let mut best: Option<f32> = None;
    // Infinite cylinder about the vertical axis through `center` (XZ-plane circle),
    // accepting only the root whose hit height lands within the cylinder band.
    let ox = origin[0] - center[0];
    let oz = origin[2] - center[2];
    let a = dir[0] * dir[0] + dir[2] * dir[2];
    if a > 1e-12 {
        let b = 2.0 * (ox * dir[0] + oz * dir[2]);
        let c = ox * ox + oz * oz - radius * radius;
        let disc = b * b - 4.0 * a * c;
        if disc >= 0.0 {
            let sd = disc.sqrt();
            for t in [(-b - sd) / (2.0 * a), (-b + sd) / (2.0 * a)] {
                if t >= 0.0 && (origin[1] + dir[1] * t - center[1]).abs() <= half_h {
                    best = Some(t);
                    break; // the first (nearest) valid root wins
                }
            }
        }
    }
    // Hemisphere caps: full spheres at the band ends. Their inner halves lie
    // inside the cylinder band, so the union is exactly the capsule.
    for cap_y in [center[1] + half_h, center[1] - half_h] {
        if let Some(t) = ray_sphere(origin, dir, [center[0], cap_y, center[2]], radius) {
            if best.is_none_or(|b| t < b) {
                best = Some(t);
            }
        }
    }
    best
}

/// Ray–sphere intersection; returns the nearest non-negative hit distance.
/// `dir` need not be unit length.
fn ray_sphere(origin: V3, dir: V3, center: V3, radius: f32) -> Option<f32> {
    let oc = sub(origin, center);
    let a = dot(dir, dir);
    if a == 0.0 {
        return None;
    }
    let b = 2.0 * dot(oc, dir);
    let c = dot(oc, oc) - radius * radius;
    let disc = b * b - 4.0 * a * c;
    if disc < 0.0 {
        return None;
    }
    let sqrt_d = disc.sqrt();
    // Nearest root; if the closer one is behind the origin, try the farther.
    let t0 = (-b - sqrt_d) / (2.0 * a);
    let t1 = (-b + sqrt_d) / (2.0 * a);
    let t = if t0 >= 0.0 { t0 } else { t1 };
    (t >= 0.0).then_some(t)
}

#[cfg(test)]
mod lag_tests {
    use super::*;

    #[test]
    fn position_rewinds_to_the_requested_tick() {
        let mut lag = LagComp::new(64);
        for tick in 0..5 {
            lag.record(tick, 1, [tick as f32, 0.0, 0.0], 0.5, 0.0);
        }
        assert_eq!(lag.position_at_tick(1, 2), Some([2.0, 0.0, 0.0]));
        // Between recorded ticks, uses the most recent at-or-before.
        assert_eq!(lag.position_at_tick(1, 100), Some([4.0, 0.0, 0.0]));
        assert_eq!(lag.position_at_tick(1, 0), Some([0.0, 0.0, 0.0]));
    }

    #[test]
    fn rewound_shot_hits_where_target_used_to_be() {
        // Target sits at x=0 on tick 0, then moves far away by tick 10.
        let mut lag = LagComp::new(64);
        lag.record(0, 7, [0.0, 0.0, 10.0], 1.0, 0.0);
        lag.record(10, 7, [100.0, 0.0, 10.0], 1.0, 0.0);

        // Shooter at origin fires straight down +Z, aiming where the target was
        // at tick 0.
        let origin = [0.0, 0.0, 0.0];
        let dir = [0.0, 0.0, 1.0];

        // Rewound to tick 0: hit.
        let hit = lag.raycast_at_tick(origin, dir, 0, /*ignore*/ 99);
        assert!(hit.is_some(), "should hit the target's tick-0 position");
        assert_eq!(hit.unwrap().entity, 7);
        assert!((hit.unwrap().distance - 9.0).abs() < 0.001); // enters sphere at z=9

        // At the present tick the target has moved; the same ray misses.
        let now = lag.raycast_at_tick(origin, dir, 10, 99);
        assert!(now.is_none(), "without rewind the shot would miss");
    }

    #[test]
    fn nearest_target_is_returned() {
        let mut lag = LagComp::new(64);
        lag.record(0, 1, [0.0, 0.0, 5.0], 1.0, 0.0);
        lag.record(0, 2, [0.0, 0.0, 20.0], 1.0, 0.0);
        let hit = lag
            .raycast_at_tick([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 0, 99)
            .unwrap();
        assert_eq!(hit.entity, 1); // the closer one
    }

    #[test]
    fn shooter_is_ignored() {
        let mut lag = LagComp::new(64);
        lag.record(0, 1, [0.0, 0.0, 5.0], 1.0, 0.0);
        let hit = lag.raycast_at_tick([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 0, /*ignore*/ 1);
        assert!(hit.is_none(), "the shooter's own collider must be ignored");
    }

    #[test]
    fn capsule_hitbox_matches_a_character_not_a_fat_sphere() {
        // Character capsule: radius 0.6, cylinder half-height 0.3 (so it spans y +/-0.9),
        // centred at (0, 0.9, 0). A shot grazing 0.8 m to the side would have hit the old
        // 1.0-radius sphere but must MISS this capsule (that was the "server hit, client
        // missed" desync). A centre-mass shot still hits, and a high shot hits the head cap.
        let mut lag = LagComp::new(64);
        lag.record(0, 1, [0.0, 0.9, 0.0], 0.6, 0.3);

        // Side graze at x=0.8, flying +Z past the target: misses the 0.6-wide capsule.
        let graze = lag.raycast_at_tick([0.8, 0.9, -5.0], [0.0, 0.0, 1.0], 0, 99);
        assert!(
            graze.is_none(),
            "a 0.8 m side graze must miss the 0.6 capsule"
        );

        // Dead-centre torso shot: hits.
        let torso = lag.raycast_at_tick([0.0, 0.9, -5.0], [0.0, 0.0, 1.0], 0, 99);
        assert!(torso.is_some(), "centre-mass must hit");

        // High shot through the head cap (y=1.7, within the +0.9 top hemisphere): hits.
        let head = lag.raycast_at_tick([0.0, 1.7, -5.0], [0.0, 0.0, 1.0], 0, 99);
        assert!(head.is_some(), "a head-height shot must hit the top cap");

        // Way over the head (y=2.2, above cap top 1.8): misses.
        let over = lag.raycast_at_tick([0.0, 2.2, -5.0], [0.0, 0.0, 1.0], 0, 99);
        assert!(over.is_none(), "a shot above the capsule must miss");
    }

    // A departed entity must take its whole ring with it. `record` only evicts inside the
    // ring it is writing, so an entity that stops being recorded is never touched again:
    // its Vec is retained forever (unbounded on a server that stays up while players
    // rotate) AND its last capsule keeps blocking rewound shots as an invisible hitbox.
    #[test]
    fn a_departed_entity_leaves_no_ring_and_no_phantom_hitbox() {
        const PLAYERS: u64 = 200;
        const WINDOW: u64 = 64;
        let mut lag = LagComp::new(WINDOW);
        // A resident that never leaves, plus players cycling through one at a time.
        let mut ring_bytes = 0usize;
        for p in 0..PLAYERS {
            let id = 10 + p;
            for t in 0..8 {
                let tick = p * 8 + t;
                lag.record(tick, 1, [0.0, 0.0, 0.0], 0.6, 0.3); // the resident
                lag.record(tick, id, [50.0, 0.0, 0.0], 0.6, 0.3); // the visitor
            }
            lag.remove(id); // ...who then leaves
            ring_bytes = ring_bytes.max(lag.tracked_entities());
        }
        eprintln!(
            "lag-comp rings after {PLAYERS} players cycled through: {} (peak {ring_bytes}); \
             without removal it would be {}",
            lag.tracked_entities(),
            PLAYERS + 1
        );
        assert_eq!(
            lag.tracked_entities(),
            1,
            "only the resident should still have a ring"
        );

        // The correctness half: a departed player must not keep absorbing shots. Fire down
        // +X at the spot the last visitor stood; only the resident (at the origin, which
        // this ray starts inside of... so aim from behind it) may be hit.
        let ghost = lag.raycast_at_tick([40.0, 0.0, 0.0], [1.0, 0.0, 0.0], u64::MAX, 1);
        assert!(
            ghost.is_none(),
            "a departed player left a phantom hitbox at {:?}",
            ghost
        );
    }

    #[test]
    fn old_snapshots_are_evicted() {
        let mut lag = LagComp::new(4);
        for tick in 0..10 {
            lag.record(tick, 1, [tick as f32, 0.0, 0.0], 0.5, 0.0);
        }
        // Tick 0 is well outside the 4-tick window; only recent ticks remain.
        assert_eq!(lag.position_at_tick(1, 0), None);
        assert_eq!(lag.position_at_tick(1, 9), Some([9.0, 0.0, 0.0]));
    }
}
