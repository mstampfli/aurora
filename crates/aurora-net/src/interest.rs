//! Interest management (netcode spec §10).
//!
//! The server doesn't replicate every entity to every client. A spatial hash
//! grid answers "which entities are within a client's relevance radius?", and
//! [`interest_delta`] computes which entities *entered* or *left* a client's set
//! between snapshots — those become reliable spawn/despawn events so clients
//! never miss a create/destroy. This is what keeps bandwidth bounded as player
//! counts grow.

use std::collections::{HashMap, HashSet};

use crate::lagcomp::V3;

type Cell = (i32, i32, i32);

/// A uniform spatial hash grid over entity positions.
pub struct InterestGrid {
    cell_size: f32,
    cells: HashMap<Cell, Vec<u64>>,
    pos: HashMap<u64, V3>,
}

impl InterestGrid {
    pub fn new(cell_size: f32) -> InterestGrid {
        assert!(cell_size > 0.0, "cell size must be positive");
        InterestGrid { cell_size, cells: HashMap::new(), pos: HashMap::new() }
    }

    fn cell_of(&self, p: V3) -> Cell {
        (
            (p[0] / self.cell_size).floor() as i32,
            (p[1] / self.cell_size).floor() as i32,
            (p[2] / self.cell_size).floor() as i32,
        )
    }

    /// Insert or move an entity to `pos`.
    pub fn insert(&mut self, entity: u64, pos: V3) {
        self.remove(entity);
        let cell = self.cell_of(pos);
        self.cells.entry(cell).or_default().push(entity);
        self.pos.insert(entity, pos);
    }

    pub fn remove(&mut self, entity: u64) {
        if let Some(old) = self.pos.remove(&entity) {
            let cell = self.cell_of(old);
            if let Some(v) = self.cells.get_mut(&cell) {
                v.retain(|&e| e != entity);
                if v.is_empty() {
                    self.cells.remove(&cell);
                }
            }
        }
    }

    pub fn len(&self) -> usize {
        self.pos.len()
    }

    pub fn is_empty(&self) -> bool {
        self.pos.is_empty()
    }

    /// All entities within `radius` of `center`. Only the cells overlapping the
    /// query AABB are scanned, then filtered by true distance.
    pub fn query(&self, center: V3, radius: f32) -> Vec<u64> {
        let r = radius.max(0.0);
        let lo = self.cell_of([center[0] - r, center[1] - r, center[2] - r]);
        let hi = self.cell_of([center[0] + r, center[1] + r, center[2] + r]);
        let r2 = r * r;

        let mut out = Vec::new();
        for cx in lo.0..=hi.0 {
            for cy in lo.1..=hi.1 {
                for cz in lo.2..=hi.2 {
                    let Some(entities) = self.cells.get(&(cx, cy, cz)) else { continue };
                    for &e in entities {
                        let p = self.pos[&e];
                        let d = [p[0] - center[0], p[1] - center[1], p[2] - center[2]];
                        if d[0] * d[0] + d[1] * d[1] + d[2] * d[2] <= r2 {
                            out.push(e);
                        }
                    }
                }
            }
        }
        out
    }

    /// Convenience: the interest set (for delta diffing) as a `HashSet`.
    pub fn query_set(&self, center: V3, radius: f32) -> HashSet<u64> {
        self.query(center, radius).into_iter().collect()
    }
}

/// Entities that entered (in `new`, not `old`) and left (in `old`, not `new`).
/// Drives reliable spawn/despawn events to a client.
pub fn interest_delta(old: &HashSet<u64>, new: &HashSet<u64>) -> (Vec<u64>, Vec<u64>) {
    let mut entered: Vec<u64> = new.difference(old).copied().collect();
    let mut left: Vec<u64> = old.difference(new).copied().collect();
    entered.sort_unstable();
    left.sort_unstable();
    (entered, left)
}

#[cfg(test)]
mod interest_tests {
    use super::*;

    #[test]
    fn query_returns_only_entities_in_radius() {
        let mut grid = InterestGrid::new(10.0);
        grid.insert(1, [0.0, 0.0, 0.0]);
        grid.insert(2, [5.0, 0.0, 0.0]); // within 10
        grid.insert(3, [50.0, 0.0, 0.0]); // far away

        let mut hits = grid.query([0.0, 0.0, 0.0], 10.0);
        hits.sort_unstable();
        assert_eq!(hits, vec![1, 2]);
    }

    #[test]
    fn radius_boundary_uses_true_distance_not_cells() {
        let mut grid = InterestGrid::new(1.0);
        grid.insert(1, [3.0, 4.0, 0.0]); // distance 5 exactly
        assert_eq!(grid.query([0.0, 0.0, 0.0], 5.0), vec![1]);
        assert!(grid.query([0.0, 0.0, 0.0], 4.9).is_empty());
    }

    #[test]
    fn moving_an_entity_updates_its_cell() {
        let mut grid = InterestGrid::new(10.0);
        grid.insert(1, [0.0, 0.0, 0.0]);
        assert_eq!(grid.query([0.0, 0.0, 0.0], 5.0), vec![1]);
        // Move it far; it should no longer be near the origin, and exactly once
        // near its new location (no duplicate from the old cell).
        grid.insert(1, [100.0, 0.0, 0.0]);
        assert!(grid.query([0.0, 0.0, 0.0], 5.0).is_empty());
        assert_eq!(grid.query([100.0, 0.0, 0.0], 5.0), vec![1]);
        assert_eq!(grid.len(), 1);
    }

    #[test]
    fn removed_entities_disappear() {
        let mut grid = InterestGrid::new(10.0);
        grid.insert(1, [0.0, 0.0, 0.0]);
        grid.remove(1);
        assert!(grid.query([0.0, 0.0, 0.0], 100.0).is_empty());
        assert!(grid.is_empty());
    }

    #[test]
    fn delta_reports_entered_and_left() {
        let old: HashSet<u64> = [1, 2, 3].into_iter().collect();
        let new: HashSet<u64> = [2, 3, 4].into_iter().collect();
        let (entered, left) = interest_delta(&old, &new);
        assert_eq!(entered, vec![4]); // newly relevant -> spawn event
        assert_eq!(left, vec![1]); // no longer relevant -> despawn event
    }
}
