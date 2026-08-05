//! Gamepads.
//!
//! The state of up to `PAD_MAX` controllers, polled once per frame at the same
//! boundary the keyboard's edge snapshot is taken, and readable as either raw
//! per-pad values or - the way a game should - as named INPUT CODES that the
//! action layer binds exactly like keys.
//!
//! # Why an input has a VALUE and not just a state
//!
//! A key is down or it is not. A stick is 0.37 of the way forward, and a game
//! that reads it as a boolean has thrown away the only thing a stick is for.
//! So the unit here is not "is this held" but "how much is this input giving",
//! in 0..1 - a key or a button answers 1.0 or 0.0, a trigger answers its pull,
//! a stick direction answers its lean.
//!
//! That is what makes analog movement fall out of the existing action layer
//! rather than being a second mechanism beside it: `input_axis(back, forward)`
//! subtracts two VALUES, so binding "PadLeftStickUp" as an alternate for
//! forward makes the same call analog with no change at the call site and no
//! branch anywhere asking "is this a controller".
//!
//! # Half-axes
//!
//! A stick axis is bipolar and an action is not, so a stick appears as four
//! named directions rather than two signed axes: `PadLeftStickUp` is the
//! positive half of the Y axis and gives 0 when the stick is pulled back. This
//! is also what a rebind screen wants - a player binds "push the stick up", not
//! "the Y axis".
//!
//! # Deadzones belong here
//!
//! Every stick rests slightly off centre, so raw values drift a character
//! across the room while nobody is touching it. The correction is a RADIAL
//! deadzone with the remainder rescaled to the full range, applied once, in the
//! engine - not in each game, where it is written differently every time and
//! usually per-axis, which is the version that lets a diagonal through the
//! corner of a square deadzone.

use std::cell::RefCell;

/// How many controllers are tracked. Four is what the console conventions and
/// every "press start" screen assume, and it is what local co-op will want.
pub const PAD_MAX: usize = 4;

/// The raw axes, in the order `pad_axis` takes.
pub const AXIS_LEFT_X: usize = 0;
pub const AXIS_LEFT_Y: usize = 1;
pub const AXIS_RIGHT_X: usize = 2;
pub const AXIS_RIGHT_Y: usize = 3;
pub const AXIS_LEFT_TRIGGER: usize = 4;
pub const AXIS_RIGHT_TRIGGER: usize = 5;
pub const PAD_AXES: usize = 6;

/// The digital buttons, in the order `pad_button` takes.
///
/// Named by POSITION, not by letter: the face buttons are south/east/west/north
/// because the same physical button is A on an Xbox pad and B on a Nintendo
/// one, and a game that binds "A" has bound different buttons on the two. The
/// input-code names below are the Xbox letters, because that is what a player
/// reads on the pad in their hands and what a prompt has to say - but the thing
/// underneath is a position.
pub const BTN_SOUTH: usize = 0;
pub const BTN_EAST: usize = 1;
pub const BTN_WEST: usize = 2;
pub const BTN_NORTH: usize = 3;
pub const BTN_LEFT_BUMPER: usize = 4;
pub const BTN_RIGHT_BUMPER: usize = 5;
pub const BTN_SELECT: usize = 6;
pub const BTN_START: usize = 7;
pub const BTN_LEFT_STICK: usize = 8;
pub const BTN_RIGHT_STICK: usize = 9;
pub const BTN_DPAD_UP: usize = 10;
pub const BTN_DPAD_DOWN: usize = 11;
pub const BTN_DPAD_LEFT: usize = 12;
pub const BTN_DPAD_RIGHT: usize = 13;
pub const PAD_BUTTONS: usize = 14;

/// How far a stick must lean before it counts as leaning at all, and how far a
/// trigger must be pulled before it counts as pressed.
///
/// The stick figure is XInput's own recommendation (7849/32767) rounded, and it
/// is applied RADIALLY - the magnitude of the whole stick, not each axis - so a
/// diagonal cannot slip through the corner of a square deadzone. Past it the
/// remainder is rescaled to 0..1, so the first movement past the deadzone is a
/// crawl rather than a jump to a quarter speed.
pub const STICK_DEADZONE: f64 = 0.24;
/// A trigger's rest position is cleaner than a stick's and its useful travel is
/// shorter, so it gets its own, smaller number rather than sharing one.
pub const TRIGGER_DEADZONE: f64 = 0.12;
/// How far an analog input must be giving before a DIGITAL read of it says
/// "held". Well past the deadzone: a light rest of the thumb should not fire an
/// action bound to the stick, and a trigger bound to attack should need a
/// deliberate pull.
pub const DIGITAL_THRESHOLD: f64 = 0.5;

/// One controller's state, as of the last poll.
#[derive(Clone, Copy)]
pub struct Pad {
    pub connected: bool,
    /// One bit per `BTN_*`.
    pub buttons: u32,
    /// Sticks in -1..1 with the deadzone already applied, triggers in 0..1.
    pub axes: [f64; PAD_AXES],
}

impl Pad {
    const fn new() -> Self {
        Pad {
            connected: false,
            buttons: 0,
            axes: [0.0; PAD_AXES],
        }
    }
}

thread_local! {
    static PADS: RefCell<[Pad; PAD_MAX]> = const { RefCell::new([Pad::new(); PAD_MAX]) };
    // NEVER DROPPED, and that is not laziness.
    //
    // gilrs runs a force-feedback server thread, and dropping the `Gilrs` blocks
    // waiting for it. In a thread_local that destructor runs at THREAD EXIT, so
    // the process hangs on the way out with every piece of work already done -
    // measured exactly that way: the pad test printed its last line and then
    // never returned, and the whole suite sat there.
    //
    // A driver handle legitimately lives for the process, so it is leaked
    // rather than dropped. `ManuallyDrop` says that in the type instead of in a
    // comment somebody has to find.
    static BACKEND: RefCell<Option<std::mem::ManuallyDrop<Backend>>> =
        const { RefCell::new(None) };
}

/// Apply a radial deadzone to a stick and rescale the remainder to 0..1.
///
/// Both axes together, which is the whole point: `x` alone past the deadzone
/// while `y` is inside it must not zero `y`, or a diagonal push becomes an
/// axis-aligned one and a character cannot walk north-east slowly.
pub fn stick_deadzone(x: f64, y: f64) -> (f64, f64) {
    let m = (x * x + y * y).sqrt();
    if m <= STICK_DEADZONE {
        return (0.0, 0.0);
    }
    // Rescaled from the deadzone edge, and clamped: a stick that reads slightly
    // past 1.0 at full lean - most of them do - must not hand the game a speed
    // multiplier above its maximum.
    let scaled = ((m - STICK_DEADZONE) / (1.0 - STICK_DEADZONE)).min(1.0);
    (x / m * scaled, y / m * scaled)
}

/// The same for a unipolar trigger.
pub fn trigger_deadzone(v: f64) -> f64 {
    if v <= TRIGGER_DEADZONE {
        return 0.0;
    }
    ((v - TRIGGER_DEADZONE) / (1.0 - TRIGGER_DEADZONE)).min(1.0)
}

/// Is pad `i` connected?
pub fn connected(i: usize) -> bool {
    PADS.with(|p| p.borrow().get(i).map(|p| p.connected).unwrap_or(false))
}

/// How many pads are connected.
pub fn count() -> usize {
    PADS.with(|p| p.borrow().iter().filter(|p| p.connected).count())
}

/// Is button `b` held on pad `i`?
pub fn button(i: usize, b: usize) -> bool {
    if b >= PAD_BUTTONS {
        return false;
    }
    PADS.with(|p| {
        p.borrow()
            .get(i)
            .map(|p| p.buttons >> b & 1 == 1)
            .unwrap_or(false)
    })
}

/// Axis `a` of pad `i`, deadzoned.
pub fn axis(i: usize, a: usize) -> f64 {
    if a >= PAD_AXES {
        return 0.0;
    }
    PADS.with(|p| p.borrow().get(i).map(|p| p.axes[a]).unwrap_or(0.0))
}

/// Is button `b` held on ANY connected pad?
///
/// What the action layer asks. A game with one player does not care which pad a
/// press came from, and asking "any" is what makes plugging a second controller
/// in mid-session simply work. Per-pad reads stay available for the day local
/// co-op needs to tell them apart.
pub fn any_button(b: usize) -> bool {
    (0..PAD_MAX).any(|i| connected(i) && button(i, b))
}

/// The largest magnitude any connected pad is giving on axis `a`, keeping its
/// sign. Sticks disagree when two pads are plugged in; the one being pushed
/// hardest is the one somebody is holding.
pub fn any_axis(a: usize) -> f64 {
    let mut best = 0.0f64;
    for i in 0..PAD_MAX {
        if !connected(i) {
            continue;
        }
        let v = axis(i, a);
        if v.abs() > best.abs() {
            best = v;
        }
    }
    best
}

/// Set pad `i`'s state directly, marking it connected.
///
/// For headless tests, and it writes the SAME state a real poll writes - the
/// rule `inject_key` follows, because a test that drives a private shadow of
/// the input state is testing the shadow. `poll` leaves an index alone when no
/// physical device is at it, so an injected pad is not clobbered by a machine
/// with nothing plugged in.
pub fn inject_button(i: usize, b: usize, down: bool) {
    if b >= PAD_BUTTONS {
        return;
    }
    PADS.with(|p| {
        if let Some(pad) = p.borrow_mut().get_mut(i) {
            pad.connected = true;
            if down {
                pad.buttons |= 1 << b;
            } else {
                pad.buttons &= !(1 << b);
            }
        }
    });
}

/// Set an axis directly. The value is taken as ALREADY deadzoned, because a
/// test asking for "half forward" means half, not "half, less whatever the
/// deadzone takes off".
pub fn inject_axis(i: usize, a: usize, v: f64) {
    if a >= PAD_AXES {
        return;
    }
    PADS.with(|p| {
        if let Some(pad) = p.borrow_mut().get_mut(i) {
            pad.connected = true;
            pad.axes[a] = v.clamp(-1.0, 1.0);
        }
    });
}

/// Disconnect a pad and forget its state. For a test that wants "no controller".
pub fn inject_disconnect(i: usize) {
    PADS.with(|p| {
        if let Some(pad) = p.borrow_mut().get_mut(i) {
            *pad = Pad::new();
        }
    });
}

/// Shake pad `i` for `seconds`: `strong` is the heavy low-frequency motor,
/// `weak` the light high-frequency one, both 0..1. Answers whether it shook.
///
/// The DURATION goes to the driver, not to a countdown here. A motor set
/// spinning keeps spinning until something sets it to zero, and every design
/// where that something is the game remembering to is a design where one
/// dropped frame leaves the pad buzzing until it is unplugged. gilrs runs a
/// scheduler for exactly this, so the effect is built with `Repeat::For` and
/// ends on its own.
///
/// The effect is KEPT, one per pad: dropping an `Effect` stops it, so a handle
/// that goes out of scope at the end of this function is a rumble that never
/// happens. Replacing the previous one is also what makes a second hit during
/// the first shake restart it rather than queue behind it.
pub fn rumble(i: usize, strong: f64, weak: f64, seconds: f64) -> bool {
    if seconds <= 0.0 {
        stop_rumble(i);
        return false;
    }
    let mag = |v: f64| (v.clamp(0.0, 1.0) * u16::MAX as f64) as u16;
    with_backend(|b| {
        let Some(id) = b.slots.get(i).copied().flatten() else {
            return false;
        };
        if !b.gilrs.gamepad(id).is_ff_supported() {
            return false;
        }
        let play_for = gilrs::ff::Ticks::from_ms((seconds * 1000.0) as u32);
        let replay = gilrs::ff::Replay {
            play_for,
            ..Default::default()
        };
        let built = gilrs::ff::EffectBuilder::new()
            .add_effect(gilrs::ff::BaseEffect {
                kind: gilrs::ff::BaseEffectType::Strong {
                    magnitude: mag(strong),
                },
                scheduling: replay,
                ..Default::default()
            })
            .add_effect(gilrs::ff::BaseEffect {
                kind: gilrs::ff::BaseEffectType::Weak {
                    magnitude: mag(weak),
                },
                scheduling: replay,
                ..Default::default()
            })
            // Not `Infinitely`, which is the builder's default and would leave
            // the pad shaking for the rest of the session.
            .repeat(gilrs::ff::Repeat::For(play_for))
            .add_gamepad(&b.gilrs.gamepad(id))
            .finish(&mut b.gilrs);
        match built {
            Ok(e) => {
                if e.play().is_err() {
                    return false;
                }
                b.effects[i] = Some(e);
                true
            }
            Err(_) => false,
        }
    })
    .unwrap_or(false)
}

/// Stop pad `i` shaking now. Dropping the effect is what stops it.
pub fn stop_rumble(i: usize) {
    with_backend(|b| {
        if let Some(e) = b.effects[i].take() {
            let _ = e.stop();
        }
    });
}

// --- the backend ------------------------------------------------------------

struct Backend {
    gilrs: gilrs::Gilrs,
    /// The effect currently shaking each pad, held because dropping one stops
    /// it. See `rumble`.
    effects: [Option<gilrs::ff::Effect>; PAD_MAX],
    /// Which gilrs id sits at each of our slots, so a pad keeps its index for
    /// as long as it is plugged in. gilrs ids are stable but not small and not
    /// ordered; a game says "player 2" and means a slot.
    slots: [Option<gilrs::GamepadId>; PAD_MAX],
}

fn with_backend<R>(f: impl FnOnce(&mut Backend) -> R) -> Option<R> {
    BACKEND.with(|b| {
        let mut b = b.borrow_mut();
        if b.is_none() {
            // A machine with no gamepad support at all is not an error: it is a
            // machine with no gamepads, which every read below already answers
            // correctly. Silent here and loud nowhere, because "no controller
            // plugged in" is the ordinary case.
            match gilrs::Gilrs::new() {
                Ok(g) => {
                    *b = Some(std::mem::ManuallyDrop::new(Backend {
                        gilrs: g,
                        effects: [const { None }; PAD_MAX],
                        slots: [None; PAD_MAX],
                    }))
                }
                Err(_) => return None,
            }
        }
        b.as_mut().map(|b| f(b))
    })
}

/// Read every connected pad into `PADS`.
///
/// Called from the frame boundary - the same one that advances the keyboard
/// edge snapshot - so a pad button gets `input_pressed` for free and cannot
/// change state halfway through a frame's logic.
/// Forget every PHYSICAL pad, keeping injected ones.
///
/// Called when this process should not be receiving hardware input - it is
/// headless, or its window does not have focus. Clearing rather than freezing,
/// because a button held at the moment focus was lost would otherwise stay held
/// forever: alt-tab out mid-roll and come back to a character still rolling.
///
/// Injected pads are deliberately untouched. They are not hardware, nothing
/// outside the process can be pressing them, and they are the only pads a
/// headless test has.
pub fn release_hardware() {
    let slots = with_backend(|b| {
        let mut had = [false; PAD_MAX];
        for (i, s) in b.slots.iter_mut().enumerate() {
            had[i] = s.is_some();
            *s = None;
        }
        had
    });
    let Some(slots) = slots else { return };
    PADS.with(|p| {
        let mut pads = p.borrow_mut();
        for (i, had) in slots.iter().enumerate() {
            if *had {
                pads[i] = Pad::new();
            }
        }
    });
}

pub fn poll() {
    let seen = with_backend(|b| {
        // gilrs caches state only as its event queue is drained, so this is not
        // optional bookkeeping: without it every read below answers whatever
        // was true when the process started.
        while b.gilrs.next_event().is_some() {}

        // Assign a slot to anything new, and let go of anything unplugged.
        let live: Vec<gilrs::GamepadId> = b
            .gilrs
            .gamepads()
            .filter(|(_, g)| g.is_connected())
            .map(|(id, _)| id)
            .collect();
        for s in b.slots.iter_mut() {
            if let Some(id) = *s {
                if !live.contains(&id) {
                    *s = None;
                }
            }
        }
        for id in &live {
            if b.slots.contains(&Some(*id)) {
                continue;
            }
            if let Some(free) = b.slots.iter_mut().find(|s| s.is_none()) {
                *free = Some(*id);
            }
        }

        let mut out = [None; PAD_MAX];
        for (i, slot) in b.slots.iter().enumerate() {
            let Some(id) = slot else { continue };
            let g = b.gilrs.gamepad(*id);
            let mut pad = Pad::new();
            pad.connected = true;
            let mut set = |bit: usize, on: bool| {
                if on {
                    pad.buttons |= 1 << bit;
                }
            };
            use gilrs::Button;
            set(BTN_SOUTH, g.is_pressed(Button::South));
            set(BTN_EAST, g.is_pressed(Button::East));
            set(BTN_WEST, g.is_pressed(Button::West));
            set(BTN_NORTH, g.is_pressed(Button::North));
            set(BTN_LEFT_BUMPER, g.is_pressed(Button::LeftTrigger));
            set(BTN_RIGHT_BUMPER, g.is_pressed(Button::RightTrigger));
            set(BTN_SELECT, g.is_pressed(Button::Select));
            set(BTN_START, g.is_pressed(Button::Start));
            set(BTN_LEFT_STICK, g.is_pressed(Button::LeftThumb));
            set(BTN_RIGHT_STICK, g.is_pressed(Button::RightThumb));
            set(BTN_DPAD_UP, g.is_pressed(Button::DPadUp));
            set(BTN_DPAD_DOWN, g.is_pressed(Button::DPadDown));
            set(BTN_DPAD_LEFT, g.is_pressed(Button::DPadLeft));
            set(BTN_DPAD_RIGHT, g.is_pressed(Button::DPadRight));

            use gilrs::Axis;
            let (lx, ly) = stick_deadzone(
                g.value(Axis::LeftStickX) as f64,
                g.value(Axis::LeftStickY) as f64,
            );
            let (rx, ry) = stick_deadzone(
                g.value(Axis::RightStickX) as f64,
                g.value(Axis::RightStickY) as f64,
            );
            pad.axes[AXIS_LEFT_X] = lx;
            pad.axes[AXIS_LEFT_Y] = ly;
            pad.axes[AXIS_RIGHT_X] = rx;
            pad.axes[AXIS_RIGHT_Y] = ry;
            // gilrs reports the triggers as buttons with an analog value, which
            // is exactly what they are - `LeftTrigger2` is the pull, and the
            // bumper above is `LeftTrigger`. Reading the wrong one of that pair
            // gives a trigger that is only ever 0 or 1.
            pad.axes[AXIS_LEFT_TRIGGER] = trigger_deadzone(
                g.button_data(Button::LeftTrigger2)
                    .map_or(0.0, |d| d.value()) as f64,
            );
            pad.axes[AXIS_RIGHT_TRIGGER] = trigger_deadzone(
                g.button_data(Button::RightTrigger2)
                    .map_or(0.0, |d| d.value()) as f64,
            );
            out[i] = Some(pad);
        }
        out
    });

    let Some(seen) = seen else { return };
    PADS.with(|p| {
        let mut pads = p.borrow_mut();
        for (i, fresh) in seen.iter().enumerate() {
            let Some(fresh) = fresh else {
                // No physical device in this slot. The state is left ALONE
                // rather than cleared, so an injected pad survives on a machine
                // with nothing plugged in - which is every headless test.
                continue;
            };
            pads[i] = *fresh;
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    // A stick at rest gives NOTHING, and that is the whole reason a deadzone
    // exists: 0.15 of drift on both axes is a character walking across the room
    // while the pad sits on the table.
    #[test]
    fn a_resting_stick_asks_for_no_movement() {
        let (x, y) = stick_deadzone(0.15, 0.15);
        assert_eq!((x, y), (0.0, 0.0));
    }

    // And the deadzone is RADIAL, not per-axis. Each of these is inside the
    // deadzone on its own axis while the stick as a whole is well past it - a
    // square deadzone answers zero and eats every slow diagonal.
    #[test]
    fn a_slow_diagonal_survives_the_deadzone() {
        let (x, y) = stick_deadzone(0.22, 0.22);
        assert!(
            x > 0.0 && y > 0.0,
            "a diagonal push at 0.31 magnitude was eaten"
        );
        assert!((x - y).abs() < 1e-9, "the diagonal was bent off 45 degrees");
    }

    // Past the deadzone the value starts from ZERO, not from the deadzone.
    // Without the rescale the slowest walk a player can ask for is a quarter of
    // full speed, and a stealth pace is unreachable.
    #[test]
    fn the_first_movement_past_the_deadzone_is_a_crawl() {
        let (_, y) = stick_deadzone(0.0, STICK_DEADZONE + 0.001);
        assert!(y > 0.0 && y < 0.01, "expected a crawl, got {y}");
    }

    // Full lean is exactly 1.0, and an over-travelling stick does not hand the
    // game a speed multiplier above its maximum.
    #[test]
    fn full_lean_is_one_and_never_more() {
        let (_, y) = stick_deadzone(0.0, 1.0);
        assert!((y - 1.0).abs() < 1e-9, "expected 1.0, got {y}");
        let (_, over) = stick_deadzone(0.0, 1.4);
        assert!((over - 1.0).abs() < 1e-9, "a stick past 1.0 gave {over}");
    }

    #[test]
    fn a_resting_trigger_is_not_pulled() {
        assert_eq!(trigger_deadzone(0.05), 0.0);
        assert!(trigger_deadzone(1.0) > 0.999);
    }

    // An injected pad is a connected pad, and reads back what was injected.
    // This is the whole basis of a headless controller test.
    #[test]
    fn an_injected_pad_reads_back() {
        inject_disconnect(0);
        assert!(!connected(0), "a fresh slot claims a controller");
        inject_button(0, BTN_SOUTH, true);
        assert!(connected(0));
        assert!(button(0, BTN_SOUTH));
        assert!(!button(0, BTN_NORTH));
        inject_axis(0, AXIS_LEFT_Y, 0.5);
        assert!((axis(0, AXIS_LEFT_Y) - 0.5).abs() < 1e-9);
        inject_button(0, BTN_SOUTH, false);
        assert!(!button(0, BTN_SOUTH));
        inject_disconnect(0);
        assert!(!connected(0));
        assert!(
            !button(0, BTN_SOUTH),
            "a disconnected pad still holds a button"
        );
    }

    // Out-of-range indexes answer "nothing" rather than panicking: an input
    // layer that can be crashed by a number is worse than one that says no.
    #[test]
    fn a_button_that_does_not_exist_is_not_held() {
        assert!(!button(0, PAD_BUTTONS));
        assert!(!button(PAD_MAX, BTN_SOUTH));
        assert_eq!(axis(0, PAD_AXES), 0.0);
        assert_eq!(axis(PAD_MAX, AXIS_LEFT_X), 0.0);
        // And injecting into one is a no-op, not a panic.
        inject_button(PAD_MAX, BTN_SOUTH, true);
        inject_axis(0, PAD_AXES, 1.0);
    }

    // `any_*` is what the action layer reads, and it must ignore a pad that is
    // not plugged in even if stale state is sitting in its slot.
    #[test]
    fn any_reads_only_connected_pads() {
        for i in 0..PAD_MAX {
            inject_disconnect(i);
        }
        assert!(!any_button(BTN_SOUTH));
        assert_eq!(count(), 0);
        inject_button(1, BTN_SOUTH, true);
        assert!(any_button(BTN_SOUTH), "a pad in slot 1 was not read");
        assert_eq!(count(), 1);
        inject_axis(1, AXIS_LEFT_X, 0.4);
        inject_axis(2, AXIS_LEFT_X, -0.9);
        // The pad being pushed hardest wins, sign kept.
        assert!((any_axis(AXIS_LEFT_X) + 0.9).abs() < 1e-9);
        inject_disconnect(2);
        assert!((any_axis(AXIS_LEFT_X) - 0.4).abs() < 1e-9);
        for i in 0..PAD_MAX {
            inject_disconnect(i);
        }
    }
}

#[cfg(test)]
mod focus_tests {
    use super::*;

    // A pad this process should not be reading is RELEASED, not frozen.
    //
    // The bug: gilrs reads the device, not the window, so a headless test - or a
    // game sitting behind another one - sees whatever the player is doing on the
    // pad in whatever they are ACTUALLY playing. It was found with a DualSense
    // being used for a different game on the same machine while the suite ran,
    // and it made six scripts fail intermittently with messages that all looked
    // like gameplay regressions.
    //
    // Released rather than frozen, because a button held at the moment focus was
    // lost would otherwise stay held forever: alt-tab out mid-roll and come back
    // to a character still rolling.
    #[test]
    fn releasing_the_hardware_keeps_injected_pads() {
        // An injected pad is not hardware. Nothing outside this process can be
        // pressing it, and in a headless run it is the only pad there is - so it
        // has to survive, or every scripted controller test goes dark.
        inject_button(0, BTN_SOUTH, true);
        inject_axis(0, AXIS_LEFT_X, 0.75);
        assert!(connected(0), "an injected pad reports connected");
        assert!(button(0, BTN_SOUTH));

        release_hardware();

        assert!(
            connected(0),
            "an injected pad is not hardware and must survive losing focus"
        );
        assert!(button(0, BTN_SOUTH), "and keeps the state it was given");
        assert!((axis(0, AXIS_LEFT_X) - 0.75).abs() < 1e-9);

        inject_disconnect(0);
        assert!(!connected(0));
        assert!(!button(0, BTN_SOUTH), "unplugging clears the state too");
    }
}
