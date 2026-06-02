//! Frame-to-frame input state — split from the window so it is unit-testable
//! without opening one. Tracks which keys are held, which were pressed *this*
//! frame (edge), mouse position, and the close request.

use std::collections::HashSet;

/// A keyboard key Aurora games can query. A useful subset of physical keys.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Key {
    W, A, S, D,
    Up, Down, Left, Right,
    Space, Enter, Escape,
    Q, E, R, F,
}

/// Snapshot of input for the current frame.
#[derive(Default)]
pub struct Input {
    down: HashSet<Key>,
    pressed: HashSet<Key>, // went down this frame (edge)
    released: HashSet<Key>,
    pub mouse: (f32, f32),
    pub mouse_down: bool,
    pub close: bool,
    pub frame: u64,
}

impl Input {
    pub fn new() -> Input {
        Input::default()
    }

    /// Is `key` currently held down?
    pub fn is_down(&self, key: Key) -> bool {
        self.down.contains(&key)
    }

    /// Did `key` transition to down on this frame (a fresh press)?
    pub fn just_pressed(&self, key: Key) -> bool {
        self.pressed.contains(&key)
    }

    /// Did `key` transition to up on this frame?
    pub fn just_released(&self, key: Key) -> bool {
        self.released.contains(&key)
    }

    /// Apply a key state change (called by the event loop).
    pub fn set_key(&mut self, key: Key, down: bool) {
        if down {
            if self.down.insert(key) {
                self.pressed.insert(key); // edge: was up, now down
            }
        } else if self.down.remove(&key) {
            self.released.insert(key);
        }
    }

    /// Clear per-frame edges and advance the frame counter. Called once after
    /// each frame's update so `just_pressed`/`just_released` are single-frame.
    pub fn end_frame(&mut self) {
        self.pressed.clear();
        self.released.clear();
        self.frame += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn press_is_an_edge_then_held() {
        let mut i = Input::new();
        i.set_key(Key::W, true);
        assert!(i.is_down(Key::W));
        assert!(i.just_pressed(Key::W), "first frame is an edge");
        i.end_frame();
        // Still held, but no longer a fresh press.
        assert!(i.is_down(Key::W));
        assert!(!i.just_pressed(Key::W));
    }

    #[test]
    fn release_is_a_single_frame_edge() {
        let mut i = Input::new();
        i.set_key(Key::Space, true);
        i.end_frame();
        i.set_key(Key::Space, false);
        assert!(!i.is_down(Key::Space));
        assert!(i.just_released(Key::Space));
        i.end_frame();
        assert!(!i.just_released(Key::Space));
    }

    #[test]
    fn repeated_down_events_do_not_re_edge() {
        let mut i = Input::new();
        i.set_key(Key::D, true);
        i.end_frame();
        i.set_key(Key::D, true); // OS key-repeat
        assert!(!i.just_pressed(Key::D), "repeat must not count as a fresh press");
    }

    #[test]
    fn frame_counter_advances() {
        let mut i = Input::new();
        assert_eq!(i.frame, 0);
        i.end_frame();
        i.end_frame();
        assert_eq!(i.frame, 2);
    }
}
