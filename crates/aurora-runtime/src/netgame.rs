//! Generic multiplayer framework for games built in Aurora. The engine owns the
//! reusable machinery - UDP transport, an authoritative server, client-side
//! prediction + reconciliation, snapshot interpolation, lag compensation,
//! interest management, and delta compression - but it does NOT contain any
//! gameplay. Each tick it runs the GAME's own simulation step, registered from
//! Aurora with [`aurora_net_sim`], over an opaque per-player state blob.
//!
//! Contract: a player's state is `f32` floats; the engine reads only `state[0..3]`
//! = x,y,z and `state[3]` = yaw (for transforms, interpolation, lag-comp). Every
//! other float is game-defined (velocity, flags, timers, ...). The same sim
//! function is run by client prediction (and its rollback replay) and by the
//! authoritative server, which is what structurally prevents drift.

use std::cell::RefCell;
use std::collections::VecDeque;
use std::net::{SocketAddr, ToSocketAddrs, UdpSocket};

use aurora_net::{InterpBuffer, LagComp};

const TAG_INPUT: u8 = 1;
const TAG_SNAPSHOT: u8 = 2;
const TAG_FIRE: u8 = 3;
const TAG_HIT: u8 = 4;
const TAG_REJECT: u8 = 5; // host -> a joiner it can't fit: the lobby is full
const TAG_OBJECTS: u8 = 6; // host -> clients: authoritative world-object (crate) positions
const TAG_KILL: u8 = 7; // host -> clients: an authoritative kill event (killer, victim) net ids
const TAG_PROJECTILE: u8 = 8; // client -> host: I LAUNCHED a rocket/grenade (intent only; the
                              // host simulates it and decides where + how hard it detonates)
const TAG_FX: u8 = 9; // host -> clients: transient visuals (loot drops + in-flight projectiles)
const TAG_SHOTFX: u8 = 10; // host -> clients: a shot was fired (shooter, origin, endpoint, weapon)
                           // so every machine can draw the tracer + play the fire sound.
const TAG_LEAVE: u8 = 11;  // client -> host: I'm leaving the lobby (remove me now, don't wait for timeout)
const TAG_BOOM: u8 = 12;   // host -> clients: an explosion detonated (source, point, intensity) so every
                           // machine renders the blast flash + sparks + boom sound. The machine that
                           // CAUSED it (source == my id) skips its own (it predicted the blast locally).

/// Reserved id range for host-controlled bots. They ride the SAME player
/// replication channel as humans (state + meta + name), so a guest receives and
/// renders a bot exactly like a remote player - the guest never runs the AI and
/// needs no "bot" concept. Ids `BOT_ID_BASE + i` stay clear of client ids (1..).
const BOT_ID_BASE: u32 = 1000;
/// Reserved lag-comp id range for world objects (crates). Recorded each tick so
/// the host rewinds them for shot validation; never collides with players/bots.
const OBJ_ID_BASE: u64 = 2000;
/// Lag-comp sphere radius approximating a crate (half-extent ~0.4).
const OBJ_RADIUS: f32 = 0.45;

const STATE_MAX: usize = 32; // max floats in a player state blob
const INPUT_MAX: usize = 24; // max floats in an input blob
const META_LEN: usize = 18; // per-player metadata floats (hp/shield/oc/respawn/cells/heal/kills/
                            // deaths in 0..7; 8 = melee-swing flag; 9 = respawn-ack; 10/11 = round
                            // timer/over; 12..14 = grapple anchor xyz; 15 = grapple active; 16 =
                            // shield-up channel active (holding the blue cube); 17 = spare) -
                           // replicated SEPARATELY from the sim state, so never touch reconciliation.
const NAME_MAX: usize = 20; // per-player display name: a fixed byte field (NOT chars packed into
                            // floats), re-sent on the input/snapshot stream so UDP loss self-heals.

/// The Aurora sim closure's native ABI: `(env, state_ptr_bits, input_ptr_bits)`,
/// matching how compiled Aurora closures are called (see `aurora_par_for`).
type SimFn = extern "C" fn(i64, i64, i64) -> i64;

/// One player's opaque state. `s[0..3]` = position, `s[3]` = yaw; rest is the
/// game's (set/read entirely by the Aurora sim).
#[derive(Clone, Copy)]
struct Player {
    s: [f32; STATE_MAX],
    /// Game-owned metadata (hp/shield/etc.), replicated separately from `s` and NOT run
    /// through the sim, so it can never clobber local-only state slots on reconciliation.
    meta: [f32; META_LEN],
    /// Display name, UTF-8 bytes, null-padded (len = bytes up to the first 0).
    name: [u8; NAME_MAX],
}
impl Player {
    fn spawn() -> Player {
        Player { s: [0.0; STATE_MAX], meta: [0.0; META_LEN], name: [0u8; NAME_MAX] }
    }
}

type InputBlob = [f32; INPUT_MAX];

struct SClient {
    addr: SocketAddr,
    id: u32,
    state: Player,
    inbox: VecDeque<(u32, InputBlob)>,
    acked_seq: u32,
    /// server_tick when we last heard from this client - used to drop it gracefully when it
    /// leaves/times out (so a flaky or reconnecting player doesn't leave a ghost that lingers).
    last_seen: u64,
    last_sent: std::collections::HashMap<u32, Player>,
    /// Per-meta-slot: true once the host has taken authority over it (hp/shield), so the
    /// client's self-reported value is no longer relayed into it - the host's value wins.
    meta_owned: [bool; META_LEN],
}

struct Remote {
    interp: InterpBuffer,
    last: Player,
    last_seen: u32,
}

/// A client's hitscan shot the host has VALIDATED with lag compensation (rewound
/// to the shooter's view tick). The host's game drains these each frame and applies
/// the damage authoritatively - the client only predicted the hitmarker.
#[derive(Clone, Copy)]
struct ServerHit {
    shooter: u32,
    victim: i64,
    point: [f32; 3],
    weapon: u8,
}

/// A client's projectile LAUNCH (intent only). The host simulates the flight in its own
/// world to decide where it detonates - the client never states the explosion result, so a
/// cheat can't blow up arbitrary points. `kind`: 0 = rocket (straight), 1 = grenade (arc).
#[derive(Clone, Copy)]
struct ServerProjectile {
    shooter: u32,
    kind: u8,
    origin: [f32; 3],
    vel: [f32; 3],
}

/// A SHOT-effect event: the host announces every shot (its own, its bots', and relayed client
/// shots) so all machines draw the tracer + play the fire sound. Purely cosmetic - rare UDP loss
/// just drops a tracer. `shooter` is the net id (so each machine skips its own predicted shot).
#[derive(Clone, Copy)]
struct ShotFx {
    shooter: u32,
    o: [f32; 3],
    e: [f32; 3],
    weapon: u8,
}

/// An EXPLOSION event: the host announces every detonation (its own + each client's re-simulated
/// rocket/grenade) so all machines render the blast + play the boom sound. `source` is the net id
/// that caused it, so the machine that fired skips its own (it already predicted the blast).
#[derive(Clone, Copy)]
struct Boom {
    source: u32,
    p: [f32; 3],
    intensity: f32,
}

/// One networking session (server or client).
pub struct Session {
    sock: UdpSocket,
    is_server: bool,
    server_addr: Option<SocketAddr>,
    my_id: u32,
    tick: f32,
    buf: Vec<u8>,
    ids: Vec<u32>,
    interest: f32,
    hit_radius: f32,
    // The game's simulation step (registered from Aurora).
    sim_fn: usize,
    sim_env: usize,
    state_len: usize, // floats replicated per player
    input_len: usize, // floats per input blob
    // Server.
    clients: Vec<SClient>,
    host: Player,
    /// Host-controlled bots, replicated to clients as ordinary players (ids
    /// BOT_ID_BASE+i). The host's game writes these each frame from its local
    /// AI; clients receive them as remotes and never run the AI themselves.
    bots: Vec<Player>,
    /// World objects (crate position + orientation: x,y,z, qx,qy,qz,qw). Host: authoritative,
    /// written each frame + replicated + recorded in lag-comp. Client: last received host pose.
    objects: Vec<[f32; 7]>,
    /// Host change-detection: objects are static until shot/bumped, so we only resend when moved.
    last_sent_objects: Vec<[f32; 7]>,
    /// Transient visuals (loot drops + in-flight projectiles): host fills + replicates each
    /// frame; clients render. Each entry is [x, y, z, kind] (kind 0-2 drops, 3 rocket, 4 grenade).
    fx: Vec<[f32; 4]>,
    /// Was the last fx broadcast empty? (so we send exactly one empty list to clear, then stop).
    last_fx_empty: bool,
    /// Kill events (killer, victim net ids). Host: pushed by its game when the death sweep
    /// credits a kill, broadcast to clients. Client: received kills, drained by the game to
    /// drive the red kill-marker/sound/feed (NOT the predicted per-hit hitmarker).
    kill_out: Vec<(u32, u32)>,
    kill_in: Vec<(u32, u32)>,
    /// Shot effects. Host: filled each frame (own + bot + relayed client shots), broadcast, then
    /// cleared. Client: received shots, drained by the game each frame to spawn tracers + sounds.
    shots_out: Vec<ShotFx>,
    shots_in: Vec<ShotFx>,
    /// Explosion events. Host: pushed each frame (own + each client's re-simmed detonation),
    /// broadcast, then cleared. Client: received booms, drained by the game to render others' blasts.
    booms_out: Vec<Boom>,
    booms_in: Vec<Boom>,
    next_id: u32,
    lag: LagComp,
    server_tick: u64,
    /// Validated client shots awaiting the host game's authoritative damage (drained per frame).
    server_hits: Vec<ServerHit>,
    /// Client projectile launches awaiting the host game's simulation + authoritative damage.
    server_projectiles: Vec<ServerProjectile>,
    // Client.
    pred: Player,
    pending: VecDeque<(u32, InputBlob)>,
    next_seq: u32,
    last_server_tick: u32,
    /// Client: whether we've received any snapshot yet, and our local clock when the last one
    /// arrived - for net_connected() (join handshake "spawn only once initialised" + clean
    /// "host disconnected" detection).
    got_snap: bool,
    last_snap_tick: f32,
    last_snap_players: usize,
    remotes: Vec<(u32, Remote)>,
    last_hit: (i64, [f32; 3]),
    /// Spawn point new players start at (set via net_spawn_at); used so the
    /// server places joining clients here instead of the origin.
    spawn: [f32; 3],
    /// Base index of the 3 input slots (x,y,z) the game uses as its respawn point (set via
    /// net_spawn_input_slot). -1 = unset. When set, the SERVER overwrites those slots in every
    /// client's input before re-simulating, so the respawn POSITION is server-authoritative (a
    /// client can't choose where it teleports to on respawn) and matches the host's spawn config.
    spawn_in: i32,
    /// The local player's outgoing metadata (set via net_set_meta), broadcast each frame.
    local_meta: [f32; META_LEN],
    /// The local player's outgoing display name (set via net_set_name).
    local_name: [u8; NAME_MAX],
    /// Max simultaneously-connected clients the host accepts (set via net_max_clients).
    max_clients: usize,
    /// Client-side: the host rejected our join (lobby full).
    rejected: bool,
    /// Dedicated server: there is no local "host player" - the machine running this Session does
    /// not play, it only simulates + broadcasts. When set, the phantom id-0 host slot is omitted
    /// from the player list, lag-comp and the id set, so every participant (including whoever runs
    /// the host) is a plain client. This is the foundation of the host-as-pure-client architecture.
    dedicated: bool,
}

/// Run the registered Aurora sim on `state` with `input` (mutating `state`).
fn run_sim(sim_fn: usize, sim_env: usize, state: &mut [f32; STATE_MAX], input: &InputBlob) {
    if sim_fn == 0 {
        return;
    }
    // SAFETY: `sim_fn` is finalized JIT/AOT Aurora code; we pass pointers to our
    // own buffers, which the sim reads/writes in place.
    let f: SimFn = unsafe { std::mem::transmute(sim_fn) };
    f(sim_env as i64, state.as_mut_ptr() as i64, input.as_ptr() as i64);
}

impl Session {
    pub fn host(port: u16) -> std::io::Result<Session> {
        // Bind ALL interfaces so real LAN / Internet clients can reach the host (not just
        // loopback). Same-machine clients still connect via 127.0.0.1:port. NOTE: this trips the
        // OS firewall prompt the first time you host - allow it (UDP) so joins get through.
        let sock = UdpSocket::bind(("0.0.0.0", port))?;
        sock.set_nonblocking(true)?;
        Ok(Session::base(sock, true, None))
    }
    pub fn join(addr: SocketAddr) -> std::io::Result<Session> {
        // Bind the wildcard 0.0.0.0 (not 127.0.0.1) so the OS routes our packets OUT the real
        // network interface to a remote host - a loopback-bound socket can only reach itself, so a
        // 127.0.0.1 bind made joining a non-local host impossible. Sending to 127.0.0.1 still works
        // from a wildcard socket, so same-machine play is unaffected.
        let sock = UdpSocket::bind(("0.0.0.0", 0))?;
        sock.set_nonblocking(true)?;
        Ok(Session::base(sock, false, Some(addr)))
    }
    fn base(sock: UdpSocket, is_server: bool, server_addr: Option<SocketAddr>) -> Session {
        Session {
            sock,
            is_server,
            server_addr,
            my_id: 0,
            tick: 0.0,
            buf: vec![0u8; 2048],
            ids: vec![0],
            interest: 80.0,
            hit_radius: 1.0,
            sim_fn: 0,
            sim_env: 0,
            state_len: 4,
            input_len: 8,
            clients: Vec::new(),
            host: Player::spawn(),
            bots: Vec::new(),
            objects: Vec::new(),
            last_sent_objects: Vec::new(),
            fx: Vec::new(),
            last_fx_empty: true,
            kill_out: Vec::new(),
            kill_in: Vec::new(),
            shots_out: Vec::new(),
            shots_in: Vec::new(),
            booms_out: Vec::new(),
            booms_in: Vec::new(),
            next_id: 1,
            lag: LagComp::new(64),
            server_tick: 0,
            server_hits: Vec::new(),
            server_projectiles: Vec::new(),
            pred: Player::spawn(),
            pending: VecDeque::new(),
            next_seq: 1,
            last_server_tick: 0,
            got_snap: false,
            last_snap_tick: 0.0,
            last_snap_players: 0,
            remotes: Vec::new(),
            last_hit: (-1, [0.0; 3]),
            spawn: [0.0, 0.0, 0.0],
            spawn_in: -1,
            local_meta: [0.0; META_LEN],
            local_name: [0u8; NAME_MAX],
            max_clients: 8,
            rejected: false,
            dedicated: false,
        }
    }

    pub fn local_addr(&self) -> SocketAddr {
        let a = self.sock.local_addr().unwrap();
        // The host binds the wildcard 0.0.0.0 (so LAN/Internet clients can reach it). Report a
        // CONNECTABLE loopback address in that case so same-machine clients (and the tests) can
        // actually send to it - you can't send a datagram to 0.0.0.0.
        if a.ip().is_unspecified() {
            SocketAddr::new(std::net::IpAddr::from([127, 0, 0, 1]), a.port())
        } else {
            a
        }
    }

    /// Register the game's simulation step and the replicated/input float counts.
    pub fn set_sim(&mut self, sim_fn: usize, sim_env: usize, state_len: usize, input_len: usize) {
        self.sim_fn = sim_fn;
        self.sim_env = sim_env;
        self.state_len = state_len.clamp(4, STATE_MAX);
        self.input_len = input_len.clamp(1, INPUT_MAX);
    }
    pub fn set_interest(&mut self, radius: f32) {
        self.interest = radius.max(0.0);
    }
    pub fn set_hit_radius(&mut self, r: f32) {
        self.hit_radius = r.max(0.01);
    }
    /// Max connected clients the host will admit (joins past this get a clear rejection).
    pub fn set_max_clients(&mut self, n: usize) {
        self.max_clients = n.max(1);
    }
    /// Mark this server as dedicated: no local host player. Call right after net_host on a server
    /// thread so the machine that hosts joins back as an ordinary client (host == remote).
    pub fn set_dedicated(&mut self) {
        self.dedicated = true;
    }
    /// Client-side: did the host reject our join because the lobby was full?
    pub fn rejected(&self) -> bool {
        self.rejected
    }
    /// Are we still in contact? The host is always connected; a client is connected once it has
    /// received a snapshot AND heard from the host within the last 5s. The game uses this to (a)
    /// hold a guest in "joining..." until its first snapshot, and (b) show "disconnected" + bail to
    /// the menu if the host vanishes.
    pub fn connected(&self) -> bool {
        self.is_server || (self.got_snap && (self.tick - self.last_snap_tick) < 5.0)
    }
    pub fn set_spawn(&mut self, x: f32, y: f32, z: f32) {
        // Set the local player's starting position, and remember it as the spawn
        // for any clients that join (so the server doesn't place them at origin).
        self.spawn = [x, y, z];
        let p = if self.is_server { &mut self.host } else { &mut self.pred };
        p.s[0] = x;
        p.s[1] = y;
        p.s[2] = z;
    }

    /// Submit this frame's input blob (`input[0..input_len]`). Predicts locally on
    /// a client (and sends it); authoritative on the host. Returns the input seq.
    pub fn send_input(&mut self, input: &[f32]) -> u32 {
        let mut blob = [0.0f32; INPUT_MAX];
        for (i, v) in input.iter().take(self.input_len).enumerate() {
            blob[i] = *v;
        }
        if self.is_server {
            run_sim(self.sim_fn, self.sim_env, &mut self.host.s, &blob);
            self.host.meta = self.local_meta; // host owns its own metadata + name
            self.host.name = self.local_name;
            0
        } else {
            let seq = self.next_seq;
            self.next_seq += 1;
            run_sim(self.sim_fn, self.sim_env, &mut self.pred.s, &blob);
            // NOTE: do NOT copy local_meta into pred.meta here. The client's own metadata
            // (hp/shield/kills/deaths/respawn-ack/...) is HOST-AUTHORITATIVE and arrives via the
            // snapshot reconcile. Overwriting it each frame with our outgoing local_meta (whose
            // host-owned slots are stale 0s) made any direct read of it FLICKER between 0 and the
            // real value (visible as the scoreboard showing 0/0 for yourself). local_meta is still
            // sent on the wire below, so our intent signals (respawn/heal/melee/grapple) reach the host.
            self.pred.name = self.local_name;
            self.pending.push_back((seq, blob));
            if let Some(addr) = self.server_addr {
                let _ = self.sock.send_to(
                    &encode_input(seq, &blob, self.input_len, &self.local_meta, &self.local_name, &self.pred.s, self.state_len),
                    addr,
                );
            }
            seq
        }
    }

    /// Fire a hitscan ray. On the host it resolves now; on a client it sends the
    /// shot with the view tick so the server can lag-compensate.
    pub fn fire(&mut self, ox: f32, oy: f32, oz: f32, dx: f32, dy: f32, dz: f32, weapon: u8) {
        let o = [ox, oy, oz];
        let d = [dx, dy, dz];
        if self.is_server {
            // The host's own shot resolves immediately against the live world (it IS the
            // authority); its game applies that damage locally, so we don't enqueue it.
            self.last_hit = match self.lag.raycast_at_tick(o, d, self.server_tick, 0) {
                Some(h) => (h.entity as i64, [o[0] + d[0] * h.distance, o[1] + d[1] * h.distance, o[2] + d[2] * h.distance]),
                None => (-1, [0.0; 3]),
            };
        } else if let Some(addr) = self.server_addr {
            let vt = self.last_server_tick.saturating_sub(2);
            let _ = self.sock.send_to(&encode_fire(vt, o, d, weapon), addr);
        }
    }
    /// Client: tell the host we're leaving the lobby so it drops us immediately (no ghost). Sent a
    /// few times since UDP can lose a single packet; harmless on the host (no server_addr).
    pub fn leave(&mut self) {
        if let Some(addr) = self.server_addr {
            for _ in 0..3 {
                let _ = self.sock.send_to(&[TAG_LEAVE], addr);
            }
        }
    }
    /// A client LAUNCHED a projectile (kind 0 rocket / 1 grenade) from `o` with velocity `v`.
    /// INTENT only - sent to the host, which simulates the flight and decides the detonation.
    /// The host's own projectiles run locally (it is the authority), so they aren't sent.
    pub fn projectile_intent(&mut self, kind: u8, o: [f32; 3], v: [f32; 3]) {
        if self.is_server {
            return;
        }
        if let Some(addr) = self.server_addr {
            let _ = self.sock.send_to(&encode_projectile(kind, o, v), addr);
        }
    }
    pub fn hit_player(&self) -> i64 {
        self.last_hit.0
    }
    pub fn hit_point(&self) -> [f32; 3] {
        self.last_hit.1
    }

    pub fn update(&mut self, dt: f32) {
        loop {
            match self.sock.recv_from(&mut self.buf) {
                Ok((n, from)) => {
                    let pkt = self.buf[..n].to_vec();
                    if self.is_server {
                        self.on_server_packet(&pkt, from);
                    } else {
                        self.on_client_packet(&pkt);
                    }
                }
                Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => break,
                Err(_) => break,
            }
        }

        if self.is_server {
            // HOST-AUTHORITATIVE movement: re-simulate each client's queued inputs in order on the
            // host's own state (c.state.s), so the host - not the client - decides where everyone is.
            // Clients still predict locally and reconcile against this.
            let (sim_fn, sim_env) = (self.sim_fn, self.sim_env);
            let sp = self.spawn;
            let sp_in = self.spawn_in;
            for c in &mut self.clients {
                while let Some((seq, mut inp)) = c.inbox.pop_front() {
                    // SERVER-AUTHORITATIVE SPAWN: overwrite the game's respawn-point input slots with
                    // the host's own spawn (id-offset so players don't stack). The client only PREDICTS
                    // its respawn position; the host decides it, so a forged input can't teleport anyone.
                    if sp_in >= 0 {
                        let b = sp_in as usize;
                        if b + 2 < INPUT_MAX {
                            inp[b] = sp[0] + c.id as f32 * 2.0;
                            inp[b + 1] = sp[1];
                            inp[b + 2] = sp[2];
                        }
                    }
                    run_sim(sim_fn, sim_env, &mut c.state.s, &inp);
                    c.acked_seq = seq;
                }
            }
            self.server_tick += 1;
            // GRACEFUL LEAVE: drop clients we haven't heard from in ~3s (quit / timed out). Without
            // this they linger forever as ghosts the host keeps re-simulating (they fall + respawn-
            // loop = "remote players keep dying"), and their slot never frees for a bot to return.
            let cutoff = self.server_tick.saturating_sub(180);
            self.clients.retain(|c| c.last_seen >= cutoff);
            let st = self.server_tick;
            let r = self.hit_radius;
            if !self.dedicated {
                self.lag.record(st, 0, [self.host.s[0], self.host.s[1], self.host.s[2]], r);
            }
            for c in &self.clients {
                self.lag.record(st, c.id as u64, [c.state.s[0], c.state.s[1], c.state.s[2]], r);
            }
            // Record bots too so the host can lag-comp validate hits on them.
            for (i, b) in self.bots.iter().enumerate() {
                self.lag.record(st, (BOT_ID_BASE + i as u32) as u64, [b.s[0], b.s[1], b.s[2]], r);
            }
            // Record world objects (crates) so a rewound shot is blocked by where a box WAS.
            for (i, o) in self.objects.iter().enumerate() {
                self.lag.record(st, OBJ_ID_BASE + i as u64, [o[0], o[1], o[2]], OBJ_RADIUS);
            }
            self.tick += dt;
            self.broadcast();
            self.ids = if self.dedicated {
                self.clients.iter().map(|c| c.id).collect()
            } else {
                std::iter::once(0u32).chain(self.clients.iter().map(|c| c.id)).collect()
            };
        } else {
            self.tick += dt;
            let now = self.last_server_tick;
            self.remotes.retain(|(_, r)| now.saturating_sub(r.last_seen) <= 90);
            self.ids = std::iter::once(self.my_id).chain(self.remotes.iter().map(|(id, _)| *id)).collect();
        }
    }

    fn broadcast(&mut self) {
        let mut all: Vec<(u32, Player)> = Vec::with_capacity(self.clients.len() + 1 + self.bots.len());
        if !self.dedicated {
            all.push((0, self.host));
        }
        for c in &self.clients {
            all.push((c.id, c.state));
        }
        // Bots ride the player channel: a guest receives them as ordinary remotes.
        for (i, b) in self.bots.iter().enumerate() {
            all.push((BOT_ID_BASE + i as u32, *b));
        }
        let interest2 = self.interest * self.interest;
        let keyframe = self.server_tick % 30 == 0;
        let (tick, stick, slen) = (self.tick, self.server_tick as u32, self.state_len);

        for ci in 0..self.clients.len() {
            let (cid, cpos, acked) = {
                let c = &self.clients[ci];
                (c.id, [c.state.s[0], c.state.s[1], c.state.s[2]], c.acked_seq)
            };
            let mut included: Vec<(u32, Player)> = Vec::new();
            for (id, st) in &all {
                if *id != cid {
                    let d = [st.s[0] - cpos[0], st.s[1] - cpos[1], st.s[2] - cpos[2]];
                    if d[0] * d[0] + d[1] * d[1] + d[2] * d[2] > interest2 {
                        continue;
                    }
                }
                let changed = self.clients[ci]
                    .last_sent
                    .get(id)
                    .map(|p| state_differs(&p.s, &st.s, slen) || meta_differs(&p.meta, &st.meta) || p.name != st.name)
                    .unwrap_or(true);
                if changed || keyframe {
                    included.push((*id, *st));
                    self.clients[ci].last_sent.insert(*id, *st);
                }
            }
            let c = &self.clients[ci];
            let pkt = encode_snapshot(cid, acked, tick, stick, slen, &included);
            let _ = self.sock.send_to(&pkt, c.addr);
        }
        // Replicate world-object (crate) positions to every client when they move (boxes are
        // static until shot, so change-detection keeps this near-zero traffic) or on a keyframe.
        if objects_differ(&self.objects, &self.last_sent_objects) || keyframe {
            let opkt = encode_objects(&self.objects);
            for c in &self.clients {
                let _ = self.sock.send_to(&opkt, c.addr);
            }
            self.last_sent_objects = self.objects.clone();
        }
        // Transient visuals (drops + in-flight projectiles) move every frame, so send them each
        // frame when present (small) and an empty list once to clear, so they don't linger.
        if !self.fx.is_empty() || !self.last_fx_empty {
            let fpkt = encode_fx(&self.fx);
            for c in &self.clients {
                let _ = self.sock.send_to(&fpkt, c.addr);
            }
            self.last_fx_empty = self.fx.is_empty();
        }
        // Broadcast authoritative kill events to every client (drives their kill feed + the
        // killer's red marker/sound). Sent once each; transient, so rare UDP loss is acceptable.
        if !self.kill_out.is_empty() {
            for (killer, victim) in self.kill_out.drain(..) {
                let kpkt = encode_kill(killer, victim);
                for c in &self.clients {
                    let _ = self.sock.send_to(&kpkt, c.addr);
                }
            }
        }
        // Shot effects are flushed separately at END of frame (flush_shots), after the host's
        // combat has pushed its own + bots' + relayed client shots for this frame.
    }

    /// Find (or admit) a client by address. Returns None if it is NEW and the lobby is
    /// already full (caller then rejects it) - so presized game arrays can never overflow.
    fn ensure_client(&mut self, from: SocketAddr) -> Option<usize> {
        if let Some(i) = self.clients.iter().position(|c| c.addr == from) {
            return Some(i);
        }
        if self.clients.len() >= self.max_clients {
            return None;
        }
        let id = self.next_id;
        self.next_id += 1;
        let mut state = Player::spawn();
        // Offset each client along x so players don't spawn stacked.
        state.s[0] = self.spawn[0] + id as f32 * 2.0;
        state.s[1] = self.spawn[1];
        state.s[2] = self.spawn[2];
        self.clients.push(SClient {
            addr: from,
            id,
            state,
            inbox: VecDeque::new(),
            acked_seq: 0,
            last_sent: std::collections::HashMap::new(),
            meta_owned: [false; META_LEN],
            last_seen: self.server_tick,
        });
        Some(self.clients.len() - 1)
    }

    fn on_server_packet(&mut self, pkt: &[u8], from: SocketAddr) {
        match pkt.first().copied() {
            Some(TAG_INPUT) => {
                let sl = self.state_len;
                if let Some((seq, blob, meta, name, _cstate)) = decode_input(pkt, self.input_len, sl) {
                    let Some(idx) = self.ensure_client(from) else {
                        // Lobby full: tell the joiner clearly instead of silently dropping it.
                        let _ = self.sock.send_to(&[TAG_REJECT], from);
                        return;
                    };
                    self.clients[idx].last_seen = self.server_tick; // alive: reset the leave timer
                    // HOST-AUTHORITATIVE movement: queue the input to be re-simulated on the host's
                    // own copy of this client's state (in update()). The client's reported movement
                    // state (_cstate) is NOT trusted - only its inputs are.
                    if seq > self.clients[idx].acked_seq {
                        self.clients[idx].inbox.push_back((seq, blob));
                    }
                    // Relay the client's self-reported metadata, EXCEPT slots the host has taken
                    // authority over (hp/shield) - those keep the host's authoritative value.
                    for s in 0..META_LEN {
                        if !self.clients[idx].meta_owned[s] {
                            self.clients[idx].state.meta[s] = meta[s];
                        }
                    }
                    self.clients[idx].state.name = name;
                }
            }
            Some(TAG_FIRE) => {
                let Some((vt, o, d, weapon)) = decode_fire(pkt) else { return };
                let Some(shooter) = self.clients.iter().find(|c| c.addr == from).map(|c| c.id) else {
                    return;
                };
                let tick = (vt as u64).min(self.server_tick);
                let (id, point) = match self.lag.raycast_at_tick(o, d, tick, shooter as u64) {
                    Some(h) => (h.entity as i64, [o[0] + d[0] * h.distance, o[1] + d[1] * h.distance, o[2] + d[2] * h.distance]),
                    None => (-1, [0.0; 3]),
                };
                // Echo the hit back so the shooter confirms its predicted hitmarker, AND queue
                // it for the host's game to apply authoritative damage to the victim.
                let _ = self.sock.send_to(&encode_hit(id, point), from);
                if id >= 0 {
                    self.server_hits.push(ServerHit { shooter, victim: id, point, weapon });
                }
                // Announce the SHOT effect (tracer + fire sound) to every machine - hit OR miss,
                // so a remote player's shots are always seen/heard. Endpoint = the hit point, else
                // far along the aim. (The shooter skips its own when rendering - it predicted it.)
                let end = if id >= 0 {
                    point
                } else {
                    [o[0] + d[0] * 120.0, o[1] + d[1] * 120.0, o[2] + d[2] * 120.0]
                };
                self.shots_out.push(ShotFx { shooter, o, e: end, weapon });
            }
            Some(TAG_PROJECTILE) => {
                let Some((kind, origin, vel)) = decode_projectile(pkt) else { return };
                let Some(shooter) = self.clients.iter().find(|c| c.addr == from).map(|c| c.id) else {
                    return;
                };
                self.server_projectiles.push(ServerProjectile { shooter, kind, origin, vel });
            }
            Some(TAG_LEAVE) => {
                // The client quit: remove it immediately so its slot frees (a bot returns) and it
                // doesn't linger until the timeout.
                self.clients.retain(|c| c.addr != from);
            }
            _ => {}
        }
    }

    fn on_client_packet(&mut self, pkt: &[u8]) {
        if pkt.first().copied() == Some(TAG_REJECT) {
            self.rejected = true; // the host's lobby is full
            return;
        }
        if pkt.first().copied() == Some(TAG_HIT) {
            if let Some((id, p)) = decode_hit(pkt) {
                self.last_hit = (id, p);
            }
            return;
        }
        if pkt.first().copied() == Some(TAG_OBJECTS) {
            if let Some(objs) = decode_objects(pkt) {
                self.objects = objs; // authoritative crate positions from the host
            }
            return;
        }
        if pkt.first().copied() == Some(TAG_FX) {
            if let Some(fx) = decode_fx(pkt) {
                self.fx = fx; // authoritative drops + in-flight projectiles from the host
            }
            return;
        }
        if pkt.first().copied() == Some(TAG_KILL) {
            if let Some((killer, victim)) = decode_kill(pkt) {
                self.kill_in.push((killer, victim));
            }
            return;
        }
        if pkt.first().copied() == Some(TAG_SHOTFX) {
            if let Some(shots) = decode_shots(pkt) {
                self.shots_in.extend(shots); // accumulated; the game drains them each frame
            }
            return;
        }
        if pkt.first().copied() == Some(TAG_BOOM) {
            if let Some(booms) = decode_booms(pkt) {
                self.booms_in.extend(booms); // accumulated; the game drains them each frame
            }
            return;
        }
        let Some((your_id, acked, tick, stick, players)) = decode_snapshot(pkt) else { return };
        self.got_snap = true;
        self.last_snap_tick = self.tick; // freshness for net_connected()
        self.my_id = your_id;
        self.tick = tick;
        self.last_server_tick = stick;
        self.last_snap_players = players.len();
        for (id, st) in players {
            if id == your_id {
                // RECONCILE against the host's authoritative state. Snap the REPLICATED movement
                // slots (0..state_len) to authoritative, but PRESERVE the local-only working slots
                // (state_len..STATE_MAX) - slot 21 there holds the client's OWN physics-body handle,
                // and a full snap would zero it (the snapshot only carries state_len slots), forcing
                // sim_step to re-create the body every reconcile (a body LEAK + perpetually-fresh
                // body). meta/name are authoritative too.
                for i in 0..self.state_len {
                    self.pred.s[i] = st.s[i];
                }
                self.pred.meta = st.meta;
                self.pred.name = st.name;
                // Drop acked inputs, then REPLAY the still-unacked ones on top of the authoritative
                // base so local prediction stays ahead of the last server snapshot.
                while self.pending.front().map(|(s, _)| *s <= acked).unwrap_or(false) {
                    self.pending.pop_front();
                }
                let pend: Vec<(u32, InputBlob)> = self.pending.iter().copied().collect();
                for (_, inp) in pend {
                    run_sim(self.sim_fn, self.sim_env, &mut self.pred.s, &inp);
                }
            } else {
                let slot = match self.remotes.iter_mut().find(|(rid, _)| *rid == id) {
                    Some((_, r)) => r,
                    None => {
                        self.remotes.push((id, Remote { interp: InterpBuffer::new(0.1), last: st, last_seen: stick }));
                        &mut self.remotes.last_mut().unwrap().1
                    }
                };
                slot.interp.push(tick, [st.s[0], st.s[1], st.s[2]]);
                slot.last = st;
                slot.last_seen = stick;
            }
        }
    }

    // --- accessors ---
    pub fn my_id(&self) -> u32 {
        self.my_id
    }
    pub fn is_server(&self) -> bool {
        self.is_server
    }
    pub fn player_count(&self) -> usize {
        self.ids.len()
    }
    pub fn player_id_at(&self, i: usize) -> i64 {
        self.ids.get(i).map(|&id| id as i64).unwrap_or(-1)
    }
    fn player_blob(&self, id: u32) -> (Player, Option<[f32; 3]>) {
        if self.is_server {
            if id == 0 {
                (self.host, None)
            } else if let Some(c) = self.clients.iter().find(|c| c.id == id) {
                (c.state, None)
            } else {
                (Player::spawn(), None)
            }
        } else if id == self.my_id {
            (self.pred, None)
        } else if let Some((_, r)) = self.remotes.iter().find(|(rid, _)| *rid == id) {
            // Sample at the client's render clock `self.tick` - the SAME time base the samples
            // were pushed with (the snapshot's `tick` seconds). Sampling at `last_server_tick`
            // (a tick COUNT, different scale) put render time past every sample, so it clamped to
            // the latest one => remotes teleported between snapshots + froze (slow fall) between.
            (r.last, r.interp.sample(self.tick))
        } else {
            (Player::spawn(), None)
        }
    }
    pub fn px(&self, id: u32) -> f64 {
        let (p, i) = self.player_blob(id);
        i.map(|q| q[0]).unwrap_or(p.s[0]) as f64
    }
    pub fn py(&self, id: u32) -> f64 {
        let (p, i) = self.player_blob(id);
        i.map(|q| q[1]).unwrap_or(p.s[1]) as f64
    }
    pub fn pz(&self, id: u32) -> f64 {
        let (p, i) = self.player_blob(id);
        i.map(|q| q[2]).unwrap_or(p.s[2]) as f64
    }
    pub fn pyaw(&self, id: u32) -> f64 {
        self.player_blob(id).0.s[3] as f64
    }
    /// Read any state float of a player (game-defined fields beyond the transform).
    pub fn state(&self, id: u32, i: usize) -> f64 {
        if i >= STATE_MAX {
            return 0.0;
        }
        self.player_blob(id).0.s[i] as f64
    }
    /// Set the LOCAL player's metadata slot (broadcast next frame).
    pub fn set_meta(&mut self, slot: usize, v: f64) {
        if slot < META_LEN {
            self.local_meta[slot] = v as f32;
        }
    }
    /// Read a player's replicated metadata slot (hp/shield/etc.).
    pub fn meta(&self, id: u32, slot: usize) -> f64 {
        if slot >= META_LEN {
            return 0.0;
        }
        self.player_blob(id).0.meta[slot] as f64
    }
    /// HOST: override ANY player's replicated metadata slot (e.g. authoritative hp/shield the
    /// host owns for a client). Re-applied each frame AFTER the client's self-report is relayed,
    /// so the host's value wins. The client reads it back as its own meta on the next snapshot.
    pub fn set_player_meta(&mut self, id: u32, slot: usize, v: f64) {
        if slot >= META_LEN || !self.is_server {
            return;
        }
        if id == 0 {
            self.host.meta[slot] = v as f32;
        } else if let Some(c) = self.clients.iter_mut().find(|c| c.id == id) {
            c.state.meta[slot] = v as f32;
            c.meta_owned[slot] = true; // from now on the host owns this slot for this client
        }
    }
    /// Set the LOCAL player's display name (UTF-8 bytes, truncated to NAME_MAX).
    pub fn set_name(&mut self, bytes: &[u8]) {
        self.local_name = [0u8; NAME_MAX];
        let n = bytes.len().min(NAME_MAX);
        self.local_name[..n].copy_from_slice(&bytes[..n]);
    }
    /// Length (bytes) of a player's replicated name.
    pub fn name_len(&self, id: u32) -> i64 {
        let p = self.player_blob(id).0;
        p.name.iter().position(|&b| b == 0).unwrap_or(NAME_MAX) as i64
    }
    /// The `i`-th byte of a player's replicated name (char code), or 0.
    pub fn name_char(&self, id: u32, i: usize) -> i64 {
        if i >= NAME_MAX {
            return 0;
        }
        self.player_blob(id).0.name[i] as i64
    }
    pub fn local_state(&self, i: usize) -> f64 {
        self.state(self.my_id, i)
    }

    // --- host-controlled bots (replicated as players ids BOT_ID_BASE+i) ---
    /// Declare how many bots the host owns this frame (grows/shrinks the set).
    pub fn set_bot_count(&mut self, n: usize) {
        if n > self.bots.len() {
            self.bots.resize(n, Player::spawn());
        } else {
            self.bots.truncate(n);
        }
    }
    pub fn bot_count(&self) -> usize {
        self.bots.len()
    }
    /// Set bot `i`'s transform (position + yaw) - the renderable state a guest reads.
    pub fn set_bot(&mut self, i: usize, x: f64, y: f64, z: f64, yaw: f64) {
        if let Some(b) = self.bots.get_mut(i) {
            b.s[0] = x as f32;
            b.s[1] = y as f32;
            b.s[2] = z as f32;
            b.s[3] = yaw as f32;
        }
    }
    /// Set bot `i`'s replicated metadata slot (hp/shield/oc), same channel as humans.
    pub fn set_bot_meta(&mut self, i: usize, slot: usize, v: f64) {
        if let Some(b) = self.bots.get_mut(i) {
            if slot < META_LEN {
                b.meta[slot] = v as f32;
            }
        }
    }
    /// Set bot `i`'s display name (UTF-8 bytes, truncated to NAME_MAX).
    pub fn set_bot_name(&mut self, i: usize, bytes: &[u8]) {
        if let Some(b) = self.bots.get_mut(i) {
            b.name = [0u8; NAME_MAX];
            let n = bytes.len().min(NAME_MAX);
            b.name[..n].copy_from_slice(&bytes[..n]);
        }
    }

    // --- world objects (crates): host writes + replicates; clients read ---
    pub fn set_object_count(&mut self, n: usize) {
        if n > self.objects.len() {
            // default pose: identity quaternion (qw = 1) so an object that never sets a rotation
            // still renders upright rather than collapsed to a zero quaternion.
            self.objects.resize(n, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]);
        } else {
            self.objects.truncate(n);
        }
    }
    pub fn set_object(&mut self, i: usize, x: f64, y: f64, z: f64) {
        // position only - leaves the orientation (set separately via set_object_rot) intact
        if let Some(o) = self.objects.get_mut(i) {
            o[0] = x as f32;
            o[1] = y as f32;
            o[2] = z as f32;
        }
    }
    pub fn set_object_rot(&mut self, i: usize, qx: f64, qy: f64, qz: f64, qw: f64) {
        if let Some(o) = self.objects.get_mut(i) {
            o[3] = qx as f32;
            o[4] = qy as f32;
            o[5] = qz as f32;
            o[6] = qw as f32;
        }
    }
    pub fn object_count(&self) -> usize {
        self.objects.len()
    }
    pub fn object_pos(&self, i: usize, axis: usize) -> f64 {
        self.objects.get(i).map(|o| o[axis.min(2)] as f64).unwrap_or(0.0)
    }
    /// Orientation component: comp 0..3 = qx,qy,qz,qw. Defaults to identity (qw = 1) if absent.
    pub fn object_rot(&self, i: usize, comp: usize) -> f64 {
        let c = comp.min(3);
        self.objects
            .get(i)
            .map(|o| o[3 + c] as f64)
            .unwrap_or(if c == 3 { 1.0 } else { 0.0 })
    }

    // --- transient visuals (drops + projectiles): host writes + replicates; clients read ---
    pub fn set_fx_count(&mut self, n: usize) {
        if n > self.fx.len() {
            self.fx.resize(n, [0.0; 4]);
        } else {
            self.fx.truncate(n);
        }
    }
    pub fn set_fx(&mut self, i: usize, x: f64, y: f64, z: f64, kind: f64) {
        if let Some(f) = self.fx.get_mut(i) {
            *f = [x as f32, y as f32, z as f32, kind as f32];
        }
    }
    pub fn fx_count(&self) -> usize {
        self.fx.len()
    }
    pub fn fx_field(&self, i: usize, field: usize) -> f64 {
        self.fx.get(i).map(|f| f[field.min(3)] as f64).unwrap_or(0.0)
    }

    // --- host: validated client shots awaiting authoritative damage (drained per frame) ---
    pub fn server_hit_count(&self) -> usize {
        self.server_hits.len()
    }
    pub fn server_hit_shooter(&self, i: usize) -> i64 {
        self.server_hits.get(i).map(|h| h.shooter as i64).unwrap_or(-1)
    }
    pub fn server_hit_victim(&self, i: usize) -> i64 {
        self.server_hits.get(i).map(|h| h.victim).unwrap_or(-1)
    }
    pub fn server_hit_weapon(&self, i: usize) -> i64 {
        self.server_hits.get(i).map(|h| h.weapon as i64).unwrap_or(0)
    }
    pub fn server_hit_x(&self, i: usize) -> f64 {
        self.server_hits.get(i).map(|h| h.point[0] as f64).unwrap_or(0.0)
    }
    pub fn server_hit_y(&self, i: usize) -> f64 {
        self.server_hits.get(i).map(|h| h.point[1] as f64).unwrap_or(0.0)
    }
    pub fn server_hit_z(&self, i: usize) -> f64 {
        self.server_hits.get(i).map(|h| h.point[2] as f64).unwrap_or(0.0)
    }
    /// The host game calls this after draining the queue each frame.
    pub fn clear_server_hits(&mut self) {
        self.server_hits.clear();
    }

    // --- host: client projectile launches awaiting simulation + authoritative damage ---
    pub fn server_projectile_count(&self) -> usize {
        self.server_projectiles.len()
    }
    pub fn server_projectile_shooter(&self, i: usize) -> i64 {
        self.server_projectiles.get(i).map(|p| p.shooter as i64).unwrap_or(-1)
    }
    pub fn server_projectile_kind(&self, i: usize) -> i64 {
        self.server_projectiles.get(i).map(|p| p.kind as i64).unwrap_or(0)
    }
    pub fn server_projectile_origin(&self, i: usize, axis: usize) -> f64 {
        self.server_projectiles.get(i).map(|p| p.origin[axis.min(2)] as f64).unwrap_or(0.0)
    }
    pub fn server_projectile_vel(&self, i: usize, axis: usize) -> f64 {
        self.server_projectiles.get(i).map(|p| p.vel[axis.min(2)] as f64).unwrap_or(0.0)
    }
    pub fn clear_server_projectiles(&mut self) {
        self.server_projectiles.clear();
    }

    // --- kill events: host announces, clients consume (kill confirm only, never hits) ---
    /// Host: announce an authoritative kill (killer + victim net ids), broadcast next frame.
    pub fn push_kill(&mut self, killer: u32, victim: u32) {
        self.kill_out.push((killer, victim));
    }
    /// Client: how many kill events arrived since the last drain.
    pub fn kill_count(&self) -> usize {
        self.kill_in.len()
    }
    pub fn kill_killer(&self, i: usize) -> i64 {
        self.kill_in.get(i).map(|k| k.0 as i64).unwrap_or(-1)
    }
    pub fn kill_victim(&self, i: usize) -> i64 {
        self.kill_in.get(i).map(|k| k.1 as i64).unwrap_or(-1)
    }
    /// Client: the game calls this after rendering the kill events this frame.
    pub fn clear_kills(&mut self) {
        self.kill_in.clear();
    }

    // --- shot effects: host announces every shot; clients draw the tracer + play the sound ---
    /// Host: announce a shot (net-id shooter, origin, endpoint, weapon), broadcast this frame.
    pub fn push_shot(&mut self, shooter: u32, ox: f64, oy: f64, oz: f64, ex: f64, ey: f64, ez: f64, weapon: i64) {
        self.shots_out.push(ShotFx {
            shooter,
            o: [ox as f32, oy as f32, oz as f32],
            e: [ex as f32, ey as f32, ez as f32],
            weapon: weapon as u8,
        });
    }
    // Read accessors source from shots_out on the HOST (it fills + renders + broadcasts its own
    // list) and shots_in on a CLIENT (it renders the received list) - so the game render code is
    // identical on both.
    fn shots_view(&self) -> &[ShotFx] {
        if self.is_server { &self.shots_out } else { &self.shots_in }
    }
    /// How many shot effects to render this frame.
    pub fn shot_count(&self) -> usize {
        self.shots_view().len()
    }
    pub fn shot_shooter(&self, i: usize) -> i64 {
        self.shots_view().get(i).map(|s| s.shooter as i64).unwrap_or(-1)
    }
    /// field 0-2 = origin x/y/z, 3-5 = endpoint x/y/z.
    pub fn shot_field(&self, i: usize, field: usize) -> f64 {
        self.shots_view()
            .get(i)
            .map(|s| if field < 3 { s.o[field] } else { s.e[(field - 3).min(2)] } as f64)
            .unwrap_or(0.0)
    }
    pub fn shot_weapon(&self, i: usize) -> i64 {
        self.shots_view().get(i).map(|s| s.weapon as i64).unwrap_or(0)
    }
    /// End of frame: the HOST broadcasts this frame's shots to clients then clears; a CLIENT just
    /// clears the rendered batch. Called once per frame after the game has drawn the shot effects.
    pub fn flush_shots(&mut self) {
        if self.is_server {
            if !self.shots_out.is_empty() {
                let spkt = encode_shots(&self.shots_out);
                for c in &self.clients {
                    let _ = self.sock.send_to(&spkt, c.addr);
                }
                self.shots_out.clear();
            }
        } else {
            self.shots_in.clear();
        }
    }

    // --- explosion events: host announces every detonation; all machines render others' blasts ---
    /// Host: announce an explosion (the net id that caused it, world point, intensity).
    pub fn push_boom(&mut self, source: u32, x: f64, y: f64, z: f64, intensity: f64) {
        self.booms_out.push(Boom {
            source,
            p: [x as f32, y as f32, z as f32],
            intensity: intensity as f32,
        });
    }
    fn booms_view(&self) -> &[Boom] {
        if self.is_server { &self.booms_out } else { &self.booms_in }
    }
    pub fn boom_count(&self) -> usize {
        self.booms_view().len()
    }
    pub fn boom_source(&self, i: usize) -> i64 {
        self.booms_view().get(i).map(|b| b.source as i64).unwrap_or(-1)
    }
    /// field 0-2 = point x/y/z, 3 = intensity.
    pub fn boom_field(&self, i: usize, field: usize) -> f64 {
        self.booms_view()
            .get(i)
            .map(|b| if field < 3 { b.p[field] } else { b.intensity } as f64)
            .unwrap_or(0.0)
    }
    /// End of frame: the HOST broadcasts this frame's booms then clears; a CLIENT clears its batch.
    pub fn flush_booms(&mut self) {
        if self.is_server {
            if !self.booms_out.is_empty() {
                let bpkt = encode_booms(&self.booms_out);
                for c in &self.clients {
                    let _ = self.sock.send_to(&bpkt, c.addr);
                }
                self.booms_out.clear();
            }
        } else {
            self.booms_in.clear();
        }
    }
}

// --- wire format (big-endian) ---

fn put_u32(b: &mut Vec<u8>, v: u32) {
    b.extend_from_slice(&v.to_be_bytes());
}
fn put_f32(b: &mut Vec<u8>, v: f32) {
    b.extend_from_slice(&v.to_be_bytes());
}
fn rd_u32(b: &[u8], o: usize) -> u32 {
    u32::from_be_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]])
}
fn rd_f32(b: &[u8], o: usize) -> f32 {
    f32::from_be_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]])
}

fn state_differs(a: &[f32; STATE_MAX], b: &[f32; STATE_MAX], len: usize) -> bool {
    (0..len).any(|i| (a[i] - b[i]).abs() > 1e-3)
}
fn meta_differs(a: &[f32; META_LEN], b: &[f32; META_LEN]) -> bool {
    (0..META_LEN).any(|i| (a[i] - b[i]).abs() > 1e-3)
}

fn encode_input(seq: u32, blob: &InputBlob, len: usize, meta: &[f32; META_LEN], name: &[u8; NAME_MAX], state: &[f32; STATE_MAX], slen: usize) -> Vec<u8> {
    let mut b = Vec::with_capacity(5 + len * 4 + META_LEN * 4 + NAME_MAX + slen * 4);
    b.push(TAG_INPUT);
    put_u32(&mut b, seq);
    for v in blob.iter().take(len) {
        put_f32(&mut b, *v);
    }
    for v in meta.iter() {
        put_f32(&mut b, *v);
    }
    b.extend_from_slice(name);
    // The client's PREDICTED movement state (the replicated slots). Movement is client-predicted
    // and the host trusts this (it can't re-simulate a remote body in its own physics world), so
    // the guest moves exactly like the local player. Combat (hits/hp) stays host-authoritative.
    for v in state.iter().take(slen) {
        put_f32(&mut b, *v);
    }
    b
}
fn decode_input(b: &[u8], len: usize, slen: usize) -> Option<(u32, InputBlob, [f32; META_LEN], [u8; NAME_MAX], [f32; STATE_MAX])> {
    if b.len() < 5 + len * 4 + META_LEN * 4 + NAME_MAX + slen * 4 || b[0] != TAG_INPUT {
        return None;
    }
    let seq = rd_u32(b, 1);
    let mut blob = [0.0f32; INPUT_MAX];
    for i in 0..len {
        blob[i] = rd_f32(b, 5 + i * 4);
    }
    let mut meta = [0.0f32; META_LEN];
    for i in 0..META_LEN {
        meta[i] = rd_f32(b, 5 + len * 4 + i * 4);
    }
    let no = 5 + len * 4 + META_LEN * 4;
    let mut name = [0u8; NAME_MAX];
    name.copy_from_slice(&b[no..no + NAME_MAX]);
    let so = no + NAME_MAX;
    let mut state = [0.0f32; STATE_MAX];
    for i in 0..slen {
        state[i] = rd_f32(b, so + i * 4);
    }
    Some((seq, blob, meta, name, state))
}

fn encode_snapshot(your_id: u32, acked: u32, tick: f32, stick: u32, slen: usize, players: &[(u32, Player)]) -> Vec<u8> {
    let mut b = Vec::with_capacity(19 + players.len() * (4 + (slen + META_LEN) * 4 + NAME_MAX));
    b.push(TAG_SNAPSHOT);
    put_u32(&mut b, your_id);
    put_u32(&mut b, acked);
    put_f32(&mut b, tick);
    put_u32(&mut b, stick);
    b.extend_from_slice(&(players.len() as u16).to_be_bytes());
    b.push(slen as u8);
    for (id, p) in players {
        put_u32(&mut b, *id);
        for i in 0..slen {
            put_f32(&mut b, p.s[i]);
        }
        for i in 0..META_LEN {
            put_f32(&mut b, p.meta[i]);
        }
        b.extend_from_slice(&p.name);
    }
    b
}
fn decode_snapshot(b: &[u8]) -> Option<(u32, u32, f32, u32, Vec<(u32, Player)>)> {
    if b.len() < 20 || b[0] != TAG_SNAPSHOT {
        return None;
    }
    let your_id = rd_u32(b, 1);
    let acked = rd_u32(b, 5);
    let tick = rd_f32(b, 9);
    let stick = rd_u32(b, 13);
    let count = u16::from_be_bytes([b[17], b[18]]) as usize;
    let slen = (b[19] as usize).min(STATE_MAX);
    let stride = 4 + (slen + META_LEN) * 4 + NAME_MAX;
    let mut players = Vec::with_capacity(count);
    let mut o = 20;
    for _ in 0..count {
        if o + stride > b.len() {
            break;
        }
        let id = rd_u32(b, o);
        let mut p = Player::spawn();
        for i in 0..slen {
            p.s[i] = rd_f32(b, o + 4 + i * 4);
        }
        for i in 0..META_LEN {
            p.meta[i] = rd_f32(b, o + 4 + slen * 4 + i * 4);
        }
        let no = o + 4 + (slen + META_LEN) * 4;
        p.name.copy_from_slice(&b[no..no + NAME_MAX]);
        players.push((id, p));
        o += stride;
    }
    Some((your_id, acked, tick, stick, players))
}

fn encode_fx(fx: &[[f32; 4]]) -> Vec<u8> {
    let mut b = Vec::with_capacity(3 + fx.len() * 16);
    b.push(TAG_FX);
    b.extend_from_slice(&(fx.len() as u16).to_be_bytes());
    for f in fx {
        for v in f {
            put_f32(&mut b, *v);
        }
    }
    b
}
fn decode_fx(b: &[u8]) -> Option<Vec<[f32; 4]>> {
    if b.len() < 3 || b[0] != TAG_FX {
        return None;
    }
    let count = u16::from_be_bytes([b[1], b[2]]) as usize;
    let mut fx = Vec::with_capacity(count);
    let mut o = 3;
    for _ in 0..count {
        if o + 16 > b.len() {
            break;
        }
        fx.push([rd_f32(b, o), rd_f32(b, o + 4), rd_f32(b, o + 8), rd_f32(b, o + 12)]);
        o += 16;
    }
    Some(fx)
}
fn encode_objects(objs: &[[f32; 7]]) -> Vec<u8> {
    let mut b = Vec::with_capacity(3 + objs.len() * 28);
    b.push(TAG_OBJECTS);
    b.extend_from_slice(&(objs.len() as u16).to_be_bytes());
    for o in objs {
        for v in o {
            put_f32(&mut b, *v);
        }
    }
    b
}
fn decode_objects(b: &[u8]) -> Option<Vec<[f32; 7]>> {
    if b.len() < 3 || b[0] != TAG_OBJECTS {
        return None;
    }
    let count = u16::from_be_bytes([b[1], b[2]]) as usize;
    let mut objs = Vec::with_capacity(count);
    let mut o = 3;
    for _ in 0..count {
        if o + 28 > b.len() {
            break;
        }
        objs.push([
            rd_f32(b, o),
            rd_f32(b, o + 4),
            rd_f32(b, o + 8),
            rd_f32(b, o + 12),
            rd_f32(b, o + 16),
            rd_f32(b, o + 20),
            rd_f32(b, o + 24),
        ]);
        o += 28;
    }
    Some(objs)
}
fn encode_projectile(kind: u8, o: [f32; 3], v: [f32; 3]) -> Vec<u8> {
    let mut b = Vec::with_capacity(26);
    b.push(TAG_PROJECTILE);
    b.push(kind);
    for x in o.iter().chain(v.iter()) {
        put_f32(&mut b, *x);
    }
    b
}
fn decode_projectile(b: &[u8]) -> Option<(u8, [f32; 3], [f32; 3])> {
    if b.len() < 26 || b[0] != TAG_PROJECTILE {
        return None;
    }
    let kind = b[1];
    let o = [rd_f32(b, 2), rd_f32(b, 6), rd_f32(b, 10)];
    let v = [rd_f32(b, 14), rd_f32(b, 18), rd_f32(b, 22)];
    Some((kind, o, v))
}
fn encode_kill(killer: u32, victim: u32) -> Vec<u8> {
    let mut b = Vec::with_capacity(9);
    b.push(TAG_KILL);
    put_u32(&mut b, killer);
    put_u32(&mut b, victim);
    b
}
fn decode_kill(b: &[u8]) -> Option<(u32, u32)> {
    if b.len() < 9 || b[0] != TAG_KILL {
        return None;
    }
    Some((rd_u32(b, 1), rd_u32(b, 5)))
}
fn encode_shots(shots: &[ShotFx]) -> Vec<u8> {
    // tag + u16 count, then per shot: u32 shooter, 6*f32 (o,e), u8 weapon = 29 bytes.
    let mut b = Vec::with_capacity(3 + shots.len() * 29);
    b.push(TAG_SHOTFX);
    b.extend_from_slice(&(shots.len() as u16).to_be_bytes());
    for s in shots {
        put_u32(&mut b, s.shooter);
        for v in s.o.iter().chain(s.e.iter()) {
            put_f32(&mut b, *v);
        }
        b.push(s.weapon);
    }
    b
}
fn decode_shots(b: &[u8]) -> Option<Vec<ShotFx>> {
    if b.len() < 3 || b[0] != TAG_SHOTFX {
        return None;
    }
    let count = u16::from_be_bytes([b[1], b[2]]) as usize;
    let mut shots = Vec::with_capacity(count);
    let mut o = 3;
    for _ in 0..count {
        if o + 29 > b.len() {
            break;
        }
        shots.push(ShotFx {
            shooter: rd_u32(b, o),
            o: [rd_f32(b, o + 4), rd_f32(b, o + 8), rd_f32(b, o + 12)],
            e: [rd_f32(b, o + 16), rd_f32(b, o + 20), rd_f32(b, o + 24)],
            weapon: b[o + 28],
        });
        o += 29;
    }
    Some(shots)
}
fn encode_booms(booms: &[Boom]) -> Vec<u8> {
    // tag + u16 count, then per boom: u32 source, 4*f32 (point xyz + intensity) = 20 bytes.
    let mut b = Vec::with_capacity(3 + booms.len() * 20);
    b.push(TAG_BOOM);
    b.extend_from_slice(&(booms.len() as u16).to_be_bytes());
    for bm in booms {
        put_u32(&mut b, bm.source);
        for v in bm.p.iter() {
            put_f32(&mut b, *v);
        }
        put_f32(&mut b, bm.intensity);
    }
    b
}
fn decode_booms(b: &[u8]) -> Option<Vec<Boom>> {
    if b.len() < 3 || b[0] != TAG_BOOM {
        return None;
    }
    let count = u16::from_be_bytes([b[1], b[2]]) as usize;
    let mut booms = Vec::with_capacity(count);
    let mut o = 3;
    for _ in 0..count {
        if o + 20 > b.len() {
            break;
        }
        booms.push(Boom {
            source: rd_u32(b, o),
            p: [rd_f32(b, o + 4), rd_f32(b, o + 8), rd_f32(b, o + 12)],
            intensity: rd_f32(b, o + 16),
        });
        o += 20;
    }
    Some(booms)
}
fn objects_differ(a: &[[f32; 7]], b: &[[f32; 7]]) -> bool {
    if a.len() != b.len() {
        return true;
    }
    // compare orientation too (a tumbling crate rotates even when its position barely moves)
    a.iter().zip(b.iter()).any(|(x, y)| (0..7).any(|i| (x[i] - y[i]).abs() > 1e-3))
}

fn encode_fire(view_tick: u32, o: [f32; 3], d: [f32; 3], weapon: u8) -> Vec<u8> {
    let mut b = Vec::with_capacity(30);
    b.push(TAG_FIRE);
    put_u32(&mut b, view_tick);
    for v in o.iter().chain(d.iter()) {
        put_f32(&mut b, *v);
    }
    b.push(weapon); // the shooter's weapon, so the host computes damage game-side
    b
}
fn decode_fire(b: &[u8]) -> Option<(u32, [f32; 3], [f32; 3], u8)> {
    if b.len() < 30 || b[0] != TAG_FIRE {
        return None;
    }
    let vt = rd_u32(b, 1);
    let o = [rd_f32(b, 5), rd_f32(b, 9), rd_f32(b, 13)];
    let d = [rd_f32(b, 17), rd_f32(b, 21), rd_f32(b, 25)];
    let weapon = b[29];
    Some((vt, o, d, weapon))
}
fn encode_hit(id: i64, p: [f32; 3]) -> Vec<u8> {
    let mut b = Vec::with_capacity(17);
    b.push(TAG_HIT);
    b.extend_from_slice(&(id as i32).to_be_bytes());
    for v in p {
        put_f32(&mut b, v);
    }
    b
}
fn decode_hit(b: &[u8]) -> Option<(i64, [f32; 3])> {
    if b.len() < 17 || b[0] != TAG_HIT {
        return None;
    }
    let id = i32::from_be_bytes([b[1], b[2], b[3], b[4]]) as i64;
    Some((id, [rd_f32(b, 5), rd_f32(b, 9), rd_f32(b, 13)]))
}

// --- thread-local session + C-ABI builtins ---

thread_local! {
    static NET: RefCell<Option<Session>> = const { RefCell::new(None) };
}

fn with<R>(default: R, f: impl FnOnce(&mut Session) -> R) -> R {
    NET.with(|n| n.borrow_mut().as_mut().map(f).unwrap_or(default))
}
fn read<R>(default: R, f: impl FnOnce(&Session) -> R) -> R {
    NET.with(|n| n.borrow().as_ref().map(f).unwrap_or(default))
}

#[no_mangle]
pub extern "C" fn aurora_net_host(port: i64) -> i64 {
    match Session::host(port.clamp(0, 65535) as u16) {
        Ok(s) => {
            NET.with(|n| *n.borrow_mut() = Some(s));
            1
        }
        Err(_) => 0,
    }
}

#[no_mangle]
pub extern "C" fn aurora_net_join(ptr: *const u8, len: i64, port: i64) -> i64 {
    let host = {
        let s = unsafe { std::slice::from_raw_parts(ptr, len.max(0) as usize) };
        String::from_utf8_lossy(s).into_owned()
    };
    let addr = match (host.as_str(), port.clamp(0, 65535) as u16).to_socket_addrs() {
        Ok(mut it) => match it.next() {
            Some(a) => a,
            None => return 0,
        },
        Err(_) => return 0,
    };
    match Session::join(addr) {
        Ok(s) => {
            NET.with(|n| *n.borrow_mut() = Some(s));
            1
        }
        Err(_) => 0,
    }
}

/// Register the game's Aurora simulation step (a closure `|state, input|`) plus
/// how many state floats to replicate and how many input floats per blob.
#[no_mangle]
pub extern "C" fn aurora_net_sim(sim_fn: *const u8, sim_env: *const u8, state_len: i64, input_len: i64) {
    with((), |s| s.set_sim(sim_fn as usize, sim_env as usize, state_len.max(4) as usize, input_len.max(1) as usize));
}

/// Run the authoritative SERVER loop on a dedicated thread. The thread gets its OWN thread-local
/// physics world + netcode (so it's a real, isolated server), and the closure runs the headless
/// server forever. The main thread then runs the rendered CLIENT (which joins 127.0.0.1), so the
/// host receives the exact same stream as any remote.
///
/// SAFETY/CONTRACT for the closure (`server_main`): it must (1) capture nothing - it runs on
/// another thread, so it takes only by-value args, never closed-over stack state; (2) never call
/// render/window builtins (the GFX context lives on the main thread); (3) set up its own world via
/// phys3d_init + net_host inside itself. The fn-pointer + env are just an immutable code address.
#[no_mangle]
pub extern "C" fn aurora_net_serve(fn_ptr: *const u8, env_ptr: *const u8) {
    let f = fn_ptr as usize;
    let e = env_ptr as usize;
    let _ = std::thread::Builder::new()
        .name("aurora-server".into())
        .stack_size(32 * 1024 * 1024)
        .spawn(move || {
            // The closure ABI is f(env_ptr) -> i64 for a zero-parameter Aurora closure.
            let server_fn: extern "C" fn(i64) -> i64 = unsafe { std::mem::transmute(f) };
            server_fn(e as i64);
        });
}

/// Submit this frame's input blob from an Aurora `[f64; len]` array; returns the
/// input seq. Floats are narrowed to `f32` for the wire / sim blob.
#[no_mangle]
pub extern "C" fn aurora_net_send_input(input: *const f64, len: i64) -> i64 {
    if input.is_null() || len <= 0 {
        return 0;
    }
    let n = len.min(INPUT_MAX as i64) as usize;
    let src = unsafe { std::slice::from_raw_parts(input, n) };
    let mut blob = [0.0f32; INPUT_MAX];
    for (i, v) in src.iter().enumerate() {
        blob[i] = *v as f32;
    }
    with(0, |s| s.send_input(&blob[..n]) as i64)
}

#[no_mangle]
pub extern "C" fn aurora_net_update(dt: f64) {
    with((), |s| s.update(dt as f32));
}
#[no_mangle]
pub extern "C" fn aurora_net_leave() {
    with((), |s| s.leave());
}
#[no_mangle]
pub extern "C" fn aurora_net_interest(radius: f64) {
    with((), |s| s.set_interest(radius as f32));
}
#[no_mangle]
pub extern "C" fn aurora_net_hit_radius(r: f64) {
    with((), |s| s.set_hit_radius(r as f32));
}
/// Host: max clients to admit (joins past this are rejected with a clear signal).
#[no_mangle]
pub extern "C" fn aurora_net_max_clients(n: i64) {
    with((), |s| s.set_max_clients(n.max(1) as usize));
}
/// Server thread: mark this server dedicated (no local host player; host joins back as a client).
#[no_mangle]
pub extern "C" fn aurora_net_dedicated() {
    with((), |s| s.set_dedicated());
}

/// Cross-thread server config (the host -> server control channel). The host (main thread) writes
/// lobby settings here BEFORE launching the server thread via net_serve, and may keep updating them;
/// server_main reads them. A plain global (NOT thread_local), so it crosses the thread boundary that
/// net_serve creates. Slot meaning is game-defined (e.g. 0 bot_count, 1 kill_target, 2 round_len).
static SERVER_CFG: std::sync::Mutex<[f64; 16]> = std::sync::Mutex::new([0.0; 16]);
/// Host: set a server-config slot (read by the dedicated server thread).
#[no_mangle]
pub extern "C" fn aurora_net_cfg_set(i: i64, v: f64) {
    if let Ok(mut c) = SERVER_CFG.lock() {
        let idx = i.max(0) as usize;
        if idx < 16 {
            c[idx] = v;
        }
    }
}
/// Read a server-config slot (the server thread, or anyone).
#[no_mangle]
pub extern "C" fn aurora_net_cfg_get(i: i64) -> f64 {
    SERVER_CFG
        .lock()
        .map(|c| {
            let idx = i.max(0) as usize;
            if idx < 16 {
                c[idx]
            } else {
                0.0
            }
        })
        .unwrap_or(0.0)
}
/// Client: 1 if the host rejected our join (lobby full), else 0.
#[no_mangle]
pub extern "C" fn aurora_net_rejected() -> i64 {
    read(0, |s| s.rejected() as i64)
}
/// 1 if still in contact with the host (or we ARE the host), 0 if not yet joined / disconnected.
#[no_mangle]
pub extern "C" fn aurora_net_connected() -> i64 {
    read(0, |s| s.connected() as i64)
}
#[no_mangle]
pub extern "C" fn aurora_net_spawn_at(x: f64, y: f64, z: f64) {
    with((), |s| s.set_spawn(x as f32, y as f32, z as f32));
}
#[no_mangle]
pub extern "C" fn aurora_net_spawn_input_slot(base: i64) {
    with((), |s| s.spawn_in = base as i32);
}

#[no_mangle]
pub extern "C" fn aurora_net_my_id() -> i64 {
    read(0, |s| s.my_id() as i64)
}
#[no_mangle]
pub extern "C" fn aurora_net_is_server() -> i64 {
    read(0, |s| s.is_server() as i64)
}
#[no_mangle]
pub extern "C" fn aurora_net_player_count() -> i64 {
    read(0, |s| s.player_count() as i64)
}
#[no_mangle]
pub extern "C" fn aurora_net_player_id_at(i: i64) -> i64 {
    read(-1, |s| s.player_id_at(i.max(0) as usize))
}
#[no_mangle]
pub extern "C" fn aurora_net_player_x(id: i64) -> f64 {
    read(0.0, |s| s.px(id.max(0) as u32))
}
#[no_mangle]
pub extern "C" fn aurora_net_player_y(id: i64) -> f64 {
    read(0.0, |s| s.py(id.max(0) as u32))
}
#[no_mangle]
pub extern "C" fn aurora_net_player_z(id: i64) -> f64 {
    read(0.0, |s| s.pz(id.max(0) as u32))
}
#[no_mangle]
pub extern "C" fn aurora_net_player_yaw(id: i64) -> f64 {
    read(0.0, |s| s.pyaw(id.max(0) as u32))
}
/// Read any replicated state float of a player (game-defined fields beyond the
/// transform, e.g. hp/shield a client writes into free state slots). Safe + additive.
#[no_mangle]
pub extern "C" fn aurora_net_player_state(id: i64, slot: i64) -> f64 {
    read(0.0, |s| s.state(id.max(0) as u32, slot.max(0) as usize))
}
/// Set the local player's metadata slot (hp/shield/etc.), replicated to everyone next frame.
#[no_mangle]
pub extern "C" fn aurora_net_set_meta(slot: i64, v: f64) {
    with((), |s| s.set_meta(slot.max(0) as usize, v))
}
/// HOST: override a specific player's metadata slot (authoritative hp/shield the host owns).
#[no_mangle]
pub extern "C" fn aurora_net_set_player_meta(id: i64, slot: i64, v: f64) {
    with((), |s| s.set_player_meta(id.max(0) as u32, slot.max(0) as usize, v))
}
/// Read a player's replicated metadata slot (works on host AND clients).
#[no_mangle]
pub extern "C" fn aurora_net_player_meta(id: i64, slot: i64) -> f64 {
    read(0.0, |s| s.meta(id.max(0) as u32, slot.max(0) as usize))
}
/// Set the local player's display name (broadcast + replicated to everyone).
#[no_mangle]
pub extern "C" fn aurora_net_set_name(ptr: *const u8, len: i64) {
    let bytes = unsafe { std::slice::from_raw_parts(ptr, len.max(0) as usize) };
    with((), |s| s.set_name(bytes))
}
/// Byte length of a player's replicated name (read char-by-char with net_player_name_char).
#[no_mangle]
pub extern "C" fn aurora_net_player_name_len(id: i64) -> i64 {
    read(0, |s| s.name_len(id.max(0) as u32))
}
/// The `i`-th byte (char code) of a player's replicated name.
#[no_mangle]
pub extern "C" fn aurora_net_player_name_char(id: i64, i: i64) -> i64 {
    read(0, |s| s.name_char(id.max(0) as u32, i.max(0) as usize))
}
// --- host-controlled bots: the host writes these each frame from its local AI;
// they replicate to clients as ordinary players (the guest renders them as remotes). ---
/// Host: declare how many bots exist this frame.
#[no_mangle]
pub extern "C" fn aurora_net_set_bot_count(n: i64) {
    with((), |s| s.set_bot_count(n.max(0) as usize))
}
/// Host: set bot `i`'s transform (x,y,z,yaw) - what a guest renders.
#[no_mangle]
pub extern "C" fn aurora_net_set_bot(i: i64, x: f64, y: f64, z: f64, yaw: f64) {
    with((), |s| s.set_bot(i.max(0) as usize, x, y, z, yaw))
}
/// Host: set bot `i`'s metadata slot (hp/shield/oc), same channel humans use.
#[no_mangle]
pub extern "C" fn aurora_net_set_bot_meta(i: i64, slot: i64, v: f64) {
    with((), |s| s.set_bot_meta(i.max(0) as usize, slot.max(0) as usize, v))
}
/// Host: set bot `i`'s display name.
#[no_mangle]
pub extern "C" fn aurora_net_set_bot_name(i: i64, ptr: *const u8, len: i64) {
    let bytes = unsafe { std::slice::from_raw_parts(ptr, len.max(0) as usize) };
    with((), |s| s.set_bot_name(i.max(0) as usize, bytes))
}
/// Number of bots the host currently owns (0 on a pure client).
#[no_mangle]
pub extern "C" fn aurora_net_bot_count() -> i64 {
    read(0, |s| s.bot_count() as i64)
}
// --- world objects (crates): host writes its authoritative positions; clients read them ---
#[no_mangle]
pub extern "C" fn aurora_net_set_object_count(n: i64) {
    with((), |s| s.set_object_count(n.max(0) as usize))
}
#[no_mangle]
pub extern "C" fn aurora_net_set_object(i: i64, x: f64, y: f64, z: f64) {
    with((), |s| s.set_object(i.max(0) as usize, x, y, z))
}
#[no_mangle]
pub extern "C" fn aurora_net_set_object_rot(i: i64, qx: f64, qy: f64, qz: f64, qw: f64) {
    with((), |s| s.set_object_rot(i.max(0) as usize, qx, qy, qz, qw))
}
#[no_mangle]
pub extern "C" fn aurora_net_object_count() -> i64 {
    read(0, |s| s.object_count() as i64)
}
#[no_mangle]
pub extern "C" fn aurora_net_object_x(i: i64) -> f64 {
    read(0.0, |s| s.object_pos(i.max(0) as usize, 0))
}
#[no_mangle]
pub extern "C" fn aurora_net_object_y(i: i64) -> f64 {
    read(0.0, |s| s.object_pos(i.max(0) as usize, 1))
}
#[no_mangle]
pub extern "C" fn aurora_net_object_z(i: i64) -> f64 {
    read(0.0, |s| s.object_pos(i.max(0) as usize, 2))
}
#[no_mangle]
pub extern "C" fn aurora_net_object_qx(i: i64) -> f64 {
    read(0.0, |s| s.object_rot(i.max(0) as usize, 0))
}
#[no_mangle]
pub extern "C" fn aurora_net_object_qy(i: i64) -> f64 {
    read(0.0, |s| s.object_rot(i.max(0) as usize, 1))
}
#[no_mangle]
pub extern "C" fn aurora_net_object_qz(i: i64) -> f64 {
    read(0.0, |s| s.object_rot(i.max(0) as usize, 2))
}
#[no_mangle]
pub extern "C" fn aurora_net_object_qw(i: i64) -> f64 {
    read(1.0, |s| s.object_rot(i.max(0) as usize, 3))
}
// --- transient visuals (host writes drops + projectiles; clients render) ---
#[no_mangle]
pub extern "C" fn aurora_net_set_fx_count(n: i64) {
    with((), |s| s.set_fx_count(n.max(0) as usize))
}
#[no_mangle]
pub extern "C" fn aurora_net_set_fx(i: i64, x: f64, y: f64, z: f64, kind: f64) {
    with((), |s| s.set_fx(i.max(0) as usize, x, y, z, kind))
}
#[no_mangle]
pub extern "C" fn aurora_net_fx_count() -> i64 {
    read(0, |s| s.fx_count() as i64)
}
#[no_mangle]
pub extern "C" fn aurora_net_fx_x(i: i64) -> f64 {
    read(0.0, |s| s.fx_field(i.max(0) as usize, 0))
}
#[no_mangle]
pub extern "C" fn aurora_net_fx_y(i: i64) -> f64 {
    read(0.0, |s| s.fx_field(i.max(0) as usize, 1))
}
#[no_mangle]
pub extern "C" fn aurora_net_fx_z(i: i64) -> f64 {
    read(0.0, |s| s.fx_field(i.max(0) as usize, 2))
}
#[no_mangle]
pub extern "C" fn aurora_net_fx_kind(i: i64) -> f64 {
    read(0.0, |s| s.fx_field(i.max(0) as usize, 3))
}
#[no_mangle]
pub extern "C" fn aurora_net_local_x() -> f64 {
    read(0.0, |s| s.px(s.my_id()))
}
#[no_mangle]
pub extern "C" fn aurora_net_local_y() -> f64 {
    read(0.0, |s| s.py(s.my_id()))
}
#[no_mangle]
pub extern "C" fn aurora_net_local_z() -> f64 {
    read(0.0, |s| s.pz(s.my_id()))
}
#[no_mangle]
pub extern "C" fn aurora_net_local_yaw() -> f64 {
    read(0.0, |s| s.pyaw(s.my_id()))
}
#[no_mangle]
pub extern "C" fn aurora_net_state(id: i64, i: i64) -> f64 {
    read(0.0, |s| s.state(id.max(0) as u32, i.max(0) as usize))
}
#[no_mangle]
pub extern "C" fn aurora_net_local_state(i: i64) -> f64 {
    read(0.0, |s| s.local_state(i.max(0) as usize))
}

#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub extern "C" fn aurora_net_fire(ox: f64, oy: f64, oz: f64, dx: f64, dy: f64, dz: f64, weapon: i64) {
    with((), |s| s.fire(ox as f32, oy as f32, oz as f32, dx as f32, dy as f32, dz as f32, weapon.max(0) as u8));
}
// --- host: drain the validated-shot queue and apply authoritative damage game-side ---
#[no_mangle]
pub extern "C" fn aurora_net_server_hit_count() -> i64 {
    read(0, |s| s.server_hit_count() as i64)
}
#[no_mangle]
pub extern "C" fn aurora_net_server_hit_shooter(i: i64) -> i64 {
    read(-1, |s| s.server_hit_shooter(i.max(0) as usize))
}
#[no_mangle]
pub extern "C" fn aurora_net_server_hit_victim(i: i64) -> i64 {
    read(-1, |s| s.server_hit_victim(i.max(0) as usize))
}
#[no_mangle]
pub extern "C" fn aurora_net_server_hit_weapon(i: i64) -> i64 {
    read(0, |s| s.server_hit_weapon(i.max(0) as usize))
}
#[no_mangle]
pub extern "C" fn aurora_net_server_hit_x(i: i64) -> f64 {
    read(0.0, |s| s.server_hit_x(i.max(0) as usize))
}
#[no_mangle]
pub extern "C" fn aurora_net_server_hit_y(i: i64) -> f64 {
    read(0.0, |s| s.server_hit_y(i.max(0) as usize))
}
#[no_mangle]
pub extern "C" fn aurora_net_server_hit_z(i: i64) -> f64 {
    read(0.0, |s| s.server_hit_z(i.max(0) as usize))
}
#[no_mangle]
pub extern "C" fn aurora_net_server_hits_clear() {
    with((), |s| s.clear_server_hits());
}
// --- projectiles: client announces a LAUNCH (intent); host drains, simulates, applies damage ---
#[no_mangle]
pub extern "C" fn aurora_net_projectile_intent(kind: i64, ox: f64, oy: f64, oz: f64, vx: f64, vy: f64, vz: f64) {
    with((), |s| s.projectile_intent(kind.max(0) as u8, [ox as f32, oy as f32, oz as f32], [vx as f32, vy as f32, vz as f32]));
}
#[no_mangle]
pub extern "C" fn aurora_net_server_projectile_count() -> i64 {
    read(0, |s| s.server_projectile_count() as i64)
}
#[no_mangle]
pub extern "C" fn aurora_net_server_projectile_shooter(i: i64) -> i64 {
    read(-1, |s| s.server_projectile_shooter(i.max(0) as usize))
}
#[no_mangle]
pub extern "C" fn aurora_net_server_projectile_kind(i: i64) -> i64 {
    read(0, |s| s.server_projectile_kind(i.max(0) as usize))
}
#[no_mangle]
pub extern "C" fn aurora_net_server_projectile_ox(i: i64) -> f64 {
    read(0.0, |s| s.server_projectile_origin(i.max(0) as usize, 0))
}
#[no_mangle]
pub extern "C" fn aurora_net_server_projectile_oy(i: i64) -> f64 {
    read(0.0, |s| s.server_projectile_origin(i.max(0) as usize, 1))
}
#[no_mangle]
pub extern "C" fn aurora_net_server_projectile_oz(i: i64) -> f64 {
    read(0.0, |s| s.server_projectile_origin(i.max(0) as usize, 2))
}
#[no_mangle]
pub extern "C" fn aurora_net_server_projectile_vx(i: i64) -> f64 {
    read(0.0, |s| s.server_projectile_vel(i.max(0) as usize, 0))
}
#[no_mangle]
pub extern "C" fn aurora_net_server_projectile_vy(i: i64) -> f64 {
    read(0.0, |s| s.server_projectile_vel(i.max(0) as usize, 1))
}
#[no_mangle]
pub extern "C" fn aurora_net_server_projectile_vz(i: i64) -> f64 {
    read(0.0, |s| s.server_projectile_vel(i.max(0) as usize, 2))
}
#[no_mangle]
pub extern "C" fn aurora_net_server_projectiles_clear() {
    with((), |s| s.clear_server_projectiles());
}
// --- kill events: host announces (push), clients consume (count/killer/victim/clear) ---
#[no_mangle]
pub extern "C" fn aurora_net_push_kill(killer: i64, victim: i64) {
    with((), |s| s.push_kill(killer.max(0) as u32, victim.max(0) as u32));
}
#[no_mangle]
pub extern "C" fn aurora_net_kill_count() -> i64 {
    read(0, |s| s.kill_count() as i64)
}
#[no_mangle]
pub extern "C" fn aurora_net_kill_killer(i: i64) -> i64 {
    read(-1, |s| s.kill_killer(i.max(0) as usize))
}
#[no_mangle]
pub extern "C" fn aurora_net_kill_victim(i: i64) -> i64 {
    read(-1, |s| s.kill_victim(i.max(0) as usize))
}
#[no_mangle]
pub extern "C" fn aurora_net_kills_clear() {
    with((), |s| s.clear_kills());
}
#[no_mangle]
pub extern "C" fn aurora_net_push_shot(
    shooter: i64, ox: f64, oy: f64, oz: f64, ex: f64, ey: f64, ez: f64, weapon: i64,
) {
    with((), |s| s.push_shot(shooter.max(0) as u32, ox, oy, oz, ex, ey, ez, weapon));
}
#[no_mangle]
pub extern "C" fn aurora_net_shot_count() -> i64 {
    read(0, |s| s.shot_count() as i64)
}
#[no_mangle]
pub extern "C" fn aurora_net_shot_shooter(i: i64) -> i64 {
    read(-1, |s| s.shot_shooter(i.max(0) as usize))
}
#[no_mangle]
pub extern "C" fn aurora_net_shot_field(i: i64, field: i64) -> f64 {
    read(0.0, |s| s.shot_field(i.max(0) as usize, field.max(0) as usize))
}
#[no_mangle]
pub extern "C" fn aurora_net_shot_weapon(i: i64) -> i64 {
    read(0, |s| s.shot_weapon(i.max(0) as usize))
}
#[no_mangle]
pub extern "C" fn aurora_net_shots_clear() {
    // End-of-frame: host broadcasts this frame's shots then clears; client clears the rendered batch.
    with((), |s| s.flush_shots());
}
// --- explosion events: host announces every detonation; all machines render others' blasts ---
#[no_mangle]
pub extern "C" fn aurora_net_push_boom(source: i64, x: f64, y: f64, z: f64, intensity: f64) {
    with((), |s| s.push_boom(source.max(0) as u32, x, y, z, intensity));
}
#[no_mangle]
pub extern "C" fn aurora_net_boom_count() -> i64 {
    read(0, |s| s.boom_count() as i64)
}
#[no_mangle]
pub extern "C" fn aurora_net_boom_source(i: i64) -> i64 {
    read(-1, |s| s.boom_source(i.max(0) as usize))
}
#[no_mangle]
pub extern "C" fn aurora_net_boom_field(i: i64, field: i64) -> f64 {
    read(0.0, |s| s.boom_field(i.max(0) as usize, field.max(0) as usize))
}
#[no_mangle]
pub extern "C" fn aurora_net_booms_clear() {
    // End-of-frame: host broadcasts this frame's booms then clears; client clears the rendered batch.
    with((), |s| s.flush_booms());
}
#[no_mangle]
pub extern "C" fn aurora_net_hit_player() -> i64 {
    read(-1, |s| s.hit_player())
}
#[no_mangle]
pub extern "C" fn aurora_net_hit_x() -> f64 {
    read(0.0, |s| s.hit_point()[0] as f64)
}
#[no_mangle]
pub extern "C" fn aurora_net_hit_y() -> f64 {
    read(0.0, |s| s.hit_point()[1] as f64)
}
#[no_mangle]
pub extern "C" fn aurora_net_hit_z() -> f64 {
    read(0.0, |s| s.hit_point()[2] as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    // A Rust stand-in for an Aurora sim: integrate position by an input velocity.
    // State layout: [x,y,z,yaw, vx,vy,vz]. Input: [vx,vz, dt].
    extern "C" fn test_sim(_env: i64, state_bits: i64, input_bits: i64) -> i64 {
        let s = state_bits as *mut f32;
        let inp = input_bits as *const f32;
        unsafe {
            let (vx, vz, dt) = (*inp, *inp.add(1), *inp.add(2));
            *s += vx * dt; // x
            *s.add(2) += vz * dt; // z
            *s.add(4) = vx;
            *s.add(6) = vz;
        }
        0
    }
    fn install(s: &mut Session) {
        s.set_sim(test_sim as usize, 0, 7, 3);
    }

    #[test]
    fn prediction_reconciles_with_a_registered_sim() {
        let mut server = Session::host(0).unwrap();
        install(&mut server);
        let saddr = server.local_addr();
        let mut client = Session::join(saddr).unwrap();
        install(&mut client);
        let dt = 1.0 / 60.0;
        for _ in 0..120 {
            client.send_input(&[3.0, 0.0, dt]); // move +x
            client.update(dt);
            server.update(dt);
            std::thread::sleep(std::time::Duration::from_micros(300));
            client.update(dt);
        }
        let cid = client.my_id();
        assert!(cid >= 1);
        let sx = server.px(cid);
        assert!(sx > 1.0, "server moved the client, got {sx}");
        assert!((client.px(cid) - sx).abs() < 0.5, "client {} converges to server {sx}", client.px(cid));
    }

    #[test]
    fn lag_compensated_shot_hits_with_a_sim() {
        let mut server = Session::host(0).unwrap();
        install(&mut server);
        let saddr = server.local_addr();
        let mut a = Session::join(saddr).unwrap();
        let mut b = Session::join(saddr).unwrap();
        install(&mut a);
        install(&mut b);
        let dt = 1.0 / 60.0;
        for _ in 0..90 {
            server.send_input(&[-4.0, 0.0, dt]); // host moves -x out of the way
            a.send_input(&[0.0, 0.0, dt]);
            b.send_input(&[4.0, 0.0, dt]); // B moves +x
            a.update(dt);
            b.update(dt);
            server.update(dt);
            std::thread::sleep(std::time::Duration::from_micros(300));
            a.update(dt);
            b.update(dt);
        }
        let bid = b.my_id();
        a.fire(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0);
        for _ in 0..10 {
            a.update(dt);
            server.update(dt);
            std::thread::sleep(std::time::Duration::from_micros(300));
            a.update(dt);
        }
        assert_eq!(a.hit_player(), bid as i64, "rewound shot should hit B");
    }

    #[test]
    fn interest_culls_distant_players() {
        let mut server = Session::host(0).unwrap();
        install(&mut server);
        server.set_interest(5.0);
        let saddr = server.local_addr();
        let mut a = Session::join(saddr).unwrap();
        let mut b = Session::join(saddr).unwrap();
        install(&mut a);
        install(&mut b);
        let dt = 1.0 / 60.0;
        for _ in 0..150 {
            server.send_input(&[-14.0, 0.0, dt]); // host leaves A's radius too
            a.send_input(&[0.0, 0.0, dt]);
            b.send_input(&[14.0, 0.0, dt]); // B leaves A's interest radius decisively
            a.update(dt);
            b.update(dt);
            server.update(dt);
            std::thread::sleep(std::time::Duration::from_micros(300));
            a.update(dt);
            b.update(dt);
        }
        assert_eq!(a.player_count(), 1, "A should cull the distant B");
    }
}

#[cfg(test)]
mod meta_replication_test {
    use super::*;
    // Two real Sessions (host + client) over the loopback UDP socket - a HEADLESS 2-player
    // test that the metadata channel replicates BOTH ways.
    #[test]
    fn metadata_replicates_host_and_client() {
        let mut host = Session::host(0).expect("host bind");
        let host_addr = host.local_addr();
        let mut client = Session::join(host_addr).expect("client bind");
        host.set_meta(0, 75.0); // host hp = 75
        client.set_meta(0, 42.0); // client hp = 42
        client.set_meta(16, 1.0); // client raises its shield-up flag (new slot 16, needs META_LEN >= 17)
        host.set_name(b"REAPER");
        client.set_name(b"NOVA");
        let input = [0.0f32; 4];
        for _ in 0..40 {
            client.send_input(&input); // client joins + sends its meta
            host.send_input(&input); // host steps + sets host.meta
            client.update(0.016);
            host.update(0.016); // host receives client input + sends a snapshot
            client.update(0.016); // client receives the snapshot (host + client meta)
        }
        let host_hp_seen_by_client = client.meta(0, 0);
        assert!(
            (host_hp_seen_by_client - 75.0).abs() < 0.01,
            "client saw host hp = {host_hp_seen_by_client}, expected 75"
        );
        let client_id = client.my_id();
        let client_hp_seen_by_host = host.meta(client_id, 0);
        assert!(
            (client_hp_seen_by_host - 42.0).abs() < 0.01,
            "host saw client hp = {client_hp_seen_by_host}, expected 42"
        );
        // the new shield-up flag (slot 16) replicates client -> host like any other meta
        let shield_seen_by_host = host.meta(client_id, 16);
        assert!(
            (shield_seen_by_host - 1.0).abs() < 0.01,
            "host saw client shield-up flag = {shield_seen_by_host}, expected 1"
        );
        // NAMES replicate both ways too (read char-by-char from the replicated byte field).
        let read_name = |s: &Session, id: u32| -> String {
            (0..s.name_len(id)).map(|i| s.name_char(id, i as usize) as u8 as char).collect()
        };
        assert_eq!(read_name(&client, 0), "REAPER", "client should see the host's name");
        assert_eq!(read_name(&host, client_id), "NOVA", "host should see the client's name");
    }

    // RECONCILE must not inflate the client's position. A trivial deterministic sim (fall 1
    // unit per step, no physics) makes any "flying / infinite jump" purely a reconcile artifact
    // (the user's clue: when the host left, the client fell correctly -> reconcile was lifting it).
    #[test]
    fn reconcile_does_not_inflate_position() {
        extern "C" fn fall_sim(_env: i64, state_ptr: i64, _input_ptr: i64) {
            let s = unsafe { &mut *(state_ptr as *mut [f32; STATE_MAX]) };
            s[1] -= 1.0; // "gravity": fall one unit per sim step
        }
        let mut host = Session::host(0).unwrap();
        let host_addr = host.local_addr();
        let mut client = Session::join(host_addr).unwrap();
        host.set_sim(fall_sim as usize, 0, 21, 17);
        client.set_sim(fall_sim as usize, 0, 21, 17);
        let input = [0.0f32; INPUT_MAX];
        let mut sends = 0;
        for f in 0..25 {
            client.send_input(&input);
            sends += 1;
            host.update(0.016);
            client.update(0.016);
            let cid = client.my_id();
            eprintln!(
                "frame {f}: client_sends={sends} client_pred_y={:.1} host_auth_y={:.1} acked seen",
                client.py(cid),
                host.py(cid),
            );
        }
        let cy = client.py(client.my_id());
        assert!(cy < -10.0, "client should have fallen well below 0 after 25 steps, got {cy}");
    }

    // FAITHFUL repro: the reconcile REPLAYS each pending input through the REAL sim_step, which
    // teleports the body (set_pos), applies manual gravity, moves, and steps phys3d + does
    // edge-triggered jump logic. This mirrors sim_step's core so we can see if replay makes the
    // client fly even though the linear test above is clean. (host + client share one phys3d world
    // here, as two instances on one machine effectively would for the shared static arena.)
    #[test]
    fn reconcile_with_physics_does_not_fly() {
        use crate::phys3d::*;
        extern "C" fn phys_fall_sim(_env: i64, state_ptr: i64, input_ptr: i64) {
            let s = unsafe { &mut *(state_ptr as *mut [f32; STATE_MAX]) };
            let inp = unsafe { &*(input_ptr as *const [f32; INPUT_MAX]) };
            let dt = 0.016f64;
            let mut player = s[21] as i64 - 1;
            if player < 0 {
                player = aurora_phys3d_add_character(s[0] as f64, s[1] as f64, s[2] as f64, 0.6, 0.3);
                s[21] = (player + 1) as f32;
            }
            let (px, py, pz) = (s[0] as f64, s[1] as f64, s[2] as f64);
            aurora_phys3d_set_pos(player, px, py, pz);
            let grounded = aurora_phys3d_raycast_world(player, px, py, pz, 0.0, -1.0, 0.0, 1.2) >= 0;
            let mut vy = s[5];
            let jump = inp[3] > 0.5;
            let last_jump = s[10] > 0.5;
            if jump && !last_jump && grounded {
                vy = 8.0;
            }
            s[10] = inp[3];
            let mut g = 16.0f32;
            if vy < 0.0 {
                g = 16.0 * 1.6;
            }
            vy -= g * dt as f32;
            aurora_phys3d_move_character(player, 0.0, (vy as f64) * dt, 0.0, dt);
            aurora_phys3d_step(dt);
            s[0] = aurora_phys3d_x(player) as f32;
            s[1] = aurora_phys3d_y(player) as f32;
            s[2] = aurora_phys3d_z(player) as f32;
            let landed = aurora_phys3d_grounded(player) == 1;
            if (landed || grounded) && vy < 0.0 {
                vy = 0.0;
            }
            s[5] = vy;
        }
        aurora_phys3d_init(0.0, -9.81, 0.0);
        aurora_phys3d_add_box(0.0, -0.5, 0.0, 100.0, 0.5, 100.0, 0); // ground, top y=0
        let mut host = Session::host(0).unwrap();
        let host_addr = host.local_addr();
        let mut client = Session::join(host_addr).unwrap();
        host.set_sim(phys_fall_sim as usize, 0, 21, 17);
        client.set_sim(phys_fall_sim as usize, 0, 21, 17);
        host.set_spawn(0.0, 8.0, 0.0); // host grounds quickly
        client.set_spawn(20.0, 8.0, 0.0); // client falls from y=8
        let no_jump = [0.0f32; INPUT_MAX];
        for f in 0..120 {
            client.send_input(&no_jump);
            host.update(0.016);
            client.update(0.016);
            if f % 15 == 0 {
                let cid = client.my_id();
                eprintln!("frame {f}: client_pred_y={:.2} host_auth_y={:.2}", client.py(cid), host.py(cid));
            }
        }
        let cy = client.py(client.my_id());
        assert!(cy < 1.5 && cy > 0.4, "client should rest on the floor (~0.9), got {cy}");
    }

    // A join past the host's cap is REJECTED with a clear signal (presized game arrays
    // can never overflow), not silently dropped.
    #[test]
    fn lobby_full_rejects_extra_client() {
        let mut host = Session::host(0).expect("host bind");
        host.set_max_clients(1); // admit exactly ONE client
        let addr = host.local_addr();
        let mut c1 = Session::join(addr).expect("c1 bind");
        let mut c2 = Session::join(addr).expect("c2 bind");
        let input = [0.0f32; 4];
        // Admit c1 first.
        for _ in 0..8 {
            c1.send_input(&input);
            host.send_input(&input);
            c1.update(0.016);
            host.update(0.016);
            c1.update(0.016);
        }
        // Now c2 tries to join - the lobby is full.
        for _ in 0..12 {
            c2.send_input(&input);
            c2.update(0.016);
            host.update(0.016);
            c2.update(0.016);
        }
        assert!(!c1.rejected(), "the first client should be admitted");
        assert!(c2.rejected(), "the second client should be rejected (lobby full)");
    }

    // Host-controlled bots replicate to a client as ORDINARY players: the client reads
    // their position / hp / name through the same net_player_* path it uses for humans,
    // with no "bot" concept of its own. This is what lets the AI brain live only on the host.
    #[test]
    fn bots_replicate_as_players() {
        let mut host = Session::host(0).expect("host bind");
        let host_addr = host.local_addr();
        let mut client = Session::join(host_addr).expect("client bind");
        let input = [0.0f32; 4];
        for _ in 0..40 {
            // Host writes its bots each frame from its (here, fake) AI.
            host.set_bot_count(2);
            host.set_bot(0, 33.0, 1.0, 5.0, 0.5);
            host.set_bot_meta(0, 0, 88.0); // bot 0 hp
            host.set_bot_name(0, b"BOT-A");
            host.set_bot(1, 44.0, 1.0, -5.0, 1.2);
            host.set_bot_meta(1, 0, 70.0); // bot 1 hp
            host.set_bot_name(1, b"BOT-B");
            client.send_input(&input);
            host.send_input(&input);
            client.update(0.016);
            host.update(0.016);
            client.update(0.016);
        }
        // The client sees the two bots as remote players at ids BOT_ID_BASE + i.
        let b0 = BOT_ID_BASE;
        let b1 = BOT_ID_BASE + 1;
        assert!((client.px(b0) - 33.0).abs() < 0.5, "client saw bot0 x = {}", client.px(b0));
        assert!((client.px(b1) - 44.0).abs() < 0.5, "client saw bot1 x = {}", client.px(b1));
        assert!((client.meta(b0, 0) - 88.0).abs() < 0.01, "client saw bot0 hp = {}", client.meta(b0, 0));
        assert!((client.meta(b1, 0) - 70.0).abs() < 0.01, "client saw bot1 hp = {}", client.meta(b1, 0));
        let read_name = |s: &Session, id: u32| -> String {
            (0..s.name_len(id)).map(|i| s.name_char(id, i as usize) as u8 as char).collect()
        };
        assert_eq!(read_name(&client, b0), "BOT-A", "client should read bot0's name");
        // And the client lists them among its players (self + host + 2 bots).
        let ids: Vec<i64> = (0..client.player_count()).map(|i| client.player_id_at(i)).collect();
        assert!(ids.contains(&(b0 as i64)) && ids.contains(&(b1 as i64)), "bots in player list: {ids:?}");
    }

    // A client's hitscan shot is VALIDATED by the host's lag-compensated raycast and queued
    // for the host's game to apply authoritative damage - the client can only predict, not
    // assert, a hit. Here the client fires straight at a host bot and the host queues that bot
    // as the victim, with the shooter id and weapon intact.
    #[test]
    fn host_applies_validated_client_shot() {
        let mut host = Session::host(0).expect("host bind");
        let host_addr = host.local_addr();
        let mut client = Session::join(host_addr).expect("client bind");
        let input = [0.0f32; 4];
        // Warm up: connect, and record the bot at (5,5,0) across many ticks so lag-comp has it.
        for _ in 0..24 {
            host.set_bot_count(1);
            host.set_bot(0, 5.0, 5.0, 0.0, 0.0);
            host.set_bot_meta(0, 0, 100.0);
            client.send_input(&input);
            host.send_input(&input);
            client.update(0.016);
            host.update(0.016);
            client.update(0.016);
        }
        host.clear_server_hits();
        // Client fires straight at the bot (weapon 2). Origin y=5 clears the host/client bodies.
        client.fire(0.0, 5.0, 0.0, 1.0, 0.0, 0.0, 2);
        for _ in 0..6 {
            host.set_bot_count(1);
            host.set_bot(0, 5.0, 5.0, 0.0, 0.0);
            client.update(0.016);
            host.update(0.016);
            client.update(0.016);
        }
        assert!(host.server_hit_count() >= 1, "host queued no validated hit");
        assert_eq!(host.server_hit_victim(0), BOT_ID_BASE as i64, "victim should be the bot");
        let cid = client.my_id();
        assert_eq!(host.server_hit_shooter(0), cid as i64, "shooter should be the client");
        assert_eq!(host.server_hit_weapon(0), 2, "weapon should round-trip");
        // The shooter also got its predicted hit echoed back (same bot).
        assert_eq!(client.hit_player(), BOT_ID_BASE as i64, "client should confirm its hitmarker");
    }

    // World objects (crates) replicate host -> client: the host owns the authoritative
    // positions, and a moved crate (as if shot) updates on the client via change-detection.
    #[test]
    fn objects_replicate_to_client() {
        let mut host = Session::host(0).expect("host bind");
        let host_addr = host.local_addr();
        let mut client = Session::join(host_addr).expect("client bind");
        let input = [0.0f32; 4];
        for _ in 0..30 {
            host.set_object_count(3);
            host.set_object(0, 10.0, 0.5, -4.0);
            host.set_object(1, 11.0, 0.5, -4.0);
            host.set_object(2, 12.0, 0.5, -4.0);
            client.send_input(&input);
            host.send_input(&input);
            client.update(0.016);
            host.update(0.016);
            client.update(0.016);
        }
        assert_eq!(client.object_count(), 3, "client should see 3 crates");
        assert!((client.object_pos(1, 0) - 11.0).abs() < 0.01, "crate 1 x = {}", client.object_pos(1, 0));
        // an unrotated crate reports identity (qw = 1) so it renders upright by default
        assert!((client.object_rot(1, 3) - 1.0).abs() < 0.01, "default crate qw = {}", client.object_rot(1, 3));
        // Move + TUMBLE crate 2 (as if shot) and confirm both position and orientation replicate.
        for _ in 0..20 {
            host.set_object_count(3);
            host.set_object(0, 10.0, 0.5, -4.0);
            host.set_object(1, 11.0, 0.5, -4.0);
            host.set_object(2, 18.0, 1.5, -2.0);
            host.set_object_rot(2, 0.0, 0.70710677, 0.0, 0.70710677); // 90deg about Y
            client.send_input(&input);
            host.send_input(&input);
            client.update(0.016);
            host.update(0.016);
            client.update(0.016);
        }
        assert!((client.object_pos(2, 0) - 18.0).abs() < 0.01, "moved crate 2 x = {}", client.object_pos(2, 0));
        assert!((client.object_pos(2, 1) - 1.5).abs() < 0.01, "moved crate 2 y = {}", client.object_pos(2, 1));
        assert!((client.object_rot(2, 1) - 0.70710677).abs() < 0.01, "tumbled crate 2 qy = {}", client.object_rot(2, 1));
        assert!((client.object_rot(2, 3) - 0.70710677).abs() < 0.01, "tumbled crate 2 qw = {}", client.object_rot(2, 3));
    }

    // Kill events announced by the host reach the client (which drives its kill feed + the
    // killer's red marker/sound) - distinct from the predicted per-hit hitmarker.
    #[test]
    fn kill_events_reach_client() {
        let mut host = Session::host(0).expect("host bind");
        let host_addr = host.local_addr();
        let mut client = Session::join(host_addr).expect("client bind");
        let input = [0.0f32; 4];
        // Connect.
        for _ in 0..10 {
            client.send_input(&input);
            host.send_input(&input);
            client.update(0.016);
            host.update(0.016);
            client.update(0.016);
        }
        let cid = client.my_id();
        // Host announces: this client killed bot 0 (net id BOT_ID_BASE).
        host.push_kill(cid, BOT_ID_BASE);
        let mut total = 0;
        for _ in 0..6 {
            host.send_input(&input);
            host.update(0.016);
            client.update(0.016);
            total += client.kill_count();
            client.clear_kills();
        }
        assert!(total >= 1, "client received no kill event");
    }

    // player_count() on the host = itself + connected CLIENTS only (bots ride a separate list),
    // so the game's "active_bots = bot_count - (player_count-1)" correctly drops a bot per joiner.
    #[test]
    fn bots_do_not_inflate_player_count() {
        let mut host = Session::host(0).expect("host bind");
        let host_addr = host.local_addr();
        let mut client = Session::join(host_addr).expect("client bind");
        host.set_bot_count(3); // host runs 3 bots
        let input = [0.0f32; 4];
        for _ in 0..15 {
            client.send_input(&input);
            host.send_input(&input);
            client.update(0.016);
            host.update(0.016);
            client.update(0.016);
        }
        assert_eq!(host.player_count(), 2, "host should count itself + 1 client, NOT the 3 bots");
    }

    // The host announces a SHOT effect (shooter, origin, endpoint, weapon); the client receives it
    // so it can draw the tracer + play the fire sound for a remote player's shot.
    #[test]
    fn shots_reach_client() {
        let mut host = Session::host(0).expect("host bind");
        let host_addr = host.local_addr();
        let mut client = Session::join(host_addr).expect("client bind");
        let input = [0.0f32; 4];
        for _ in 0..10 {
            client.send_input(&input);
            host.send_input(&input);
            client.update(0.016);
            host.update(0.016);
            client.update(0.016);
        }
        let mut got = 0;
        for _ in 0..6 {
            host.push_shot(BOT_ID_BASE as u32, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0);
            host.flush_shots(); // broadcast this frame's shots, then clears the host's list
            client.update(0.016);
            if client.shot_count() > 0 {
                got += 1;
                assert_eq!(client.shot_shooter(0), BOT_ID_BASE as i64, "shooter id");
                assert!((client.shot_field(0, 0) - 1.0).abs() < 0.01, "origin x");
                assert!((client.shot_field(0, 5) - 6.0).abs() < 0.01, "endpoint z");
            }
            client.flush_shots();
        }
        assert!(got >= 1, "client received no shot effect");
        // The HOST reads its own pushed shots (shots_out) so it can render them locally too.
        host.push_shot(7, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0);
        assert_eq!(host.shot_count(), 1, "host should see its own queued shot for local render");
        host.flush_shots();
        assert_eq!(host.shot_count(), 0, "flush clears the host list");
    }

    // A client's projectile LAUNCH (intent: kind + origin + velocity) is queued on the host,
    // which will simulate it to decide the detonation. The client never states the result.
    #[test]
    fn projectile_intent_queues_on_host() {
        let mut host = Session::host(0).expect("host bind");
        let host_addr = host.local_addr();
        let mut client = Session::join(host_addr).expect("client bind");
        let input = [0.0f32; 4];
        for _ in 0..10 {
            client.send_input(&input);
            host.send_input(&input);
            client.update(0.016);
            host.update(0.016);
            client.update(0.016);
        }
        host.clear_server_projectiles();
        client.projectile_intent(0, [2.0, 1.5, 0.0], [1.0, 0.0, 0.0]); // rocket east
        for _ in 0..6 {
            client.update(0.016);
            host.update(0.016);
        }
        assert!(host.server_projectile_count() >= 1, "host queued no projectile");
        let cid = client.my_id();
        assert_eq!(host.server_projectile_shooter(0), cid as i64, "shooter id");
        assert_eq!(host.server_projectile_kind(0), 0, "kind = rocket");
        assert!((host.server_projectile_origin(0, 0) - 2.0).abs() < 0.01, "origin x");
        assert!((host.server_projectile_vel(0, 0) - 1.0).abs() < 0.01, "vel x");
    }

    // The host can OWN a client's hp: it overrides the client's replicated meta, and the
    // client reads that authoritative value back as its own (host-authoritative health).
    #[test]
    fn host_owns_client_hp() {
        let mut host = Session::host(0).expect("host bind");
        let host_addr = host.local_addr();
        let mut client = Session::join(host_addr).expect("client bind");
        client.set_meta(0, 100.0); // the client THINKS it has 100
        let input = [0.0f32; 4];
        for _ in 0..30 {
            client.send_input(&input);
            host.send_input(&input);
            client.update(0.016);
            // Host overrides this client's hp to 37 AFTER the relay each frame (authoritative).
            let cid = client.my_id();
            host.set_player_meta(cid, 0, 37.0);
            host.update(0.016);
            client.update(0.016);
        }
        // The client reads its OWN hp as the host's authoritative value, not its self-report.
        assert!((client.meta(client.my_id(), 0) - 37.0).abs() < 0.01, "client hp = {}", client.meta(client.my_id(), 0));
    }

    // Transient visuals (loot drops + in-flight projectiles) replicate host -> client so a guest
    // sees others' rockets fly + the loot on the ground.
    #[test]
    fn fx_replicate_to_client() {
        let mut host = Session::host(0).expect("host bind");
        let host_addr = host.local_addr();
        let mut client = Session::join(host_addr).expect("client bind");
        let input = [0.0f32; 4];
        for _ in 0..20 {
            host.set_fx_count(2);
            host.set_fx(0, 4.0, 0.6, -2.0, 1.0); // a shield-cell drop
            host.set_fx(1, 9.0, 1.5, 0.0, 3.0); // a rocket in flight
            client.send_input(&input);
            host.send_input(&input);
            client.update(0.016);
            host.update(0.016);
            client.update(0.016);
        }
        assert_eq!(client.fx_count(), 2, "client should see 2 fx");
        assert!((client.fx_field(0, 0) - 4.0).abs() < 0.01, "drop x");
        assert!((client.fx_field(0, 3) - 1.0).abs() < 0.01, "drop kind");
        assert!((client.fx_field(1, 0) - 9.0).abs() < 0.01, "rocket x");
        assert!((client.fx_field(1, 3) - 3.0).abs() < 0.01, "rocket kind");
    }
}
