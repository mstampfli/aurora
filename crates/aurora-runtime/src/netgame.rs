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
const META_LEN: usize = 8; // per-player metadata floats (hp/shield/cells/oc) replicated
                           // SEPARATELY from the sim state, so they never touch reconciliation.
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
    last_sent: std::collections::HashMap<u32, Player>,
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
    /// World objects (crate positions). Host: authoritative, written each frame +
    /// replicated + recorded in lag-comp. Client: the last received host positions.
    objects: Vec<[f32; 3]>,
    /// Host change-detection: objects are static until shot, so we only resend when moved.
    last_sent_objects: Vec<[f32; 3]>,
    next_id: u32,
    lag: LagComp,
    server_tick: u64,
    /// Validated client shots awaiting the host game's authoritative damage (drained per frame).
    server_hits: Vec<ServerHit>,
    // Client.
    pred: Player,
    pending: VecDeque<(u32, InputBlob)>,
    next_seq: u32,
    last_server_tick: u32,
    last_snap_players: usize,
    remotes: Vec<(u32, Remote)>,
    last_hit: (i64, [f32; 3]),
    /// Spawn point new players start at (set via net_spawn_at); used so the
    /// server places joining clients here instead of the origin.
    spawn: [f32; 3],
    /// The local player's outgoing metadata (set via net_set_meta), broadcast each frame.
    local_meta: [f32; META_LEN],
    /// The local player's outgoing display name (set via net_set_name).
    local_name: [u8; NAME_MAX],
    /// Max simultaneously-connected clients the host accepts (set via net_max_clients).
    max_clients: usize,
    /// Client-side: the host rejected our join (lobby full).
    rejected: bool,
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
        // Bind loopback: works for solo play and local multi-client testing, and
        // does NOT trip the OS firewall prompt that binding all interfaces does.
        // (Switch to 0.0.0.0 when real LAN/Internet play is actually wired up.)
        let sock = UdpSocket::bind(("127.0.0.1", port))?;
        sock.set_nonblocking(true)?;
        Ok(Session::base(sock, true, None))
    }
    pub fn join(addr: SocketAddr) -> std::io::Result<Session> {
        let sock = UdpSocket::bind(("127.0.0.1", 0))?;
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
            next_id: 1,
            lag: LagComp::new(64),
            server_tick: 0,
            server_hits: Vec::new(),
            pred: Player::spawn(),
            pending: VecDeque::new(),
            next_seq: 1,
            last_server_tick: 0,
            last_snap_players: 0,
            remotes: Vec::new(),
            last_hit: (-1, [0.0; 3]),
            spawn: [0.0, 0.0, 0.0],
            local_meta: [0.0; META_LEN],
            local_name: [0u8; NAME_MAX],
            max_clients: 8,
            rejected: false,
        }
    }

    pub fn local_addr(&self) -> SocketAddr {
        self.sock.local_addr().unwrap()
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
    /// Client-side: did the host reject our join because the lobby was full?
    pub fn rejected(&self) -> bool {
        self.rejected
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
            self.pred.meta = self.local_meta;
            self.pred.name = self.local_name;
            self.pending.push_back((seq, blob));
            if let Some(addr) = self.server_addr {
                let _ = self.sock.send_to(
                    &encode_input(seq, &blob, self.input_len, &self.local_meta, &self.local_name),
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
            let (sim_fn, sim_env) = (self.sim_fn, self.sim_env);
            for c in &mut self.clients {
                while let Some((seq, inp)) = c.inbox.pop_front() {
                    run_sim(sim_fn, sim_env, &mut c.state.s, &inp);
                    c.acked_seq = seq;
                }
            }
            self.server_tick += 1;
            let st = self.server_tick;
            let r = self.hit_radius;
            self.lag.record(st, 0, [self.host.s[0], self.host.s[1], self.host.s[2]], r);
            for c in &self.clients {
                self.lag.record(st, c.id as u64, [c.state.s[0], c.state.s[1], c.state.s[2]], r);
            }
            // Record bots too so the host can lag-comp validate hits on them.
            for (i, b) in self.bots.iter().enumerate() {
                self.lag.record(st, (BOT_ID_BASE + i as u32) as u64, [b.s[0], b.s[1], b.s[2]], r);
            }
            // Record world objects (crates) so a rewound shot is blocked by where a box WAS.
            for (i, o) in self.objects.iter().enumerate() {
                self.lag.record(st, OBJ_ID_BASE + i as u64, *o, OBJ_RADIUS);
            }
            self.tick += dt;
            self.broadcast();
            self.ids = std::iter::once(0u32).chain(self.clients.iter().map(|c| c.id)).collect();
        } else {
            self.tick += dt;
            let now = self.last_server_tick;
            self.remotes.retain(|(_, r)| now.saturating_sub(r.last_seen) <= 90);
            self.ids = std::iter::once(self.my_id).chain(self.remotes.iter().map(|(id, _)| *id)).collect();
        }
    }

    fn broadcast(&mut self) {
        let mut all: Vec<(u32, Player)> = Vec::with_capacity(self.clients.len() + 1 + self.bots.len());
        all.push((0, self.host));
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
        });
        Some(self.clients.len() - 1)
    }

    fn on_server_packet(&mut self, pkt: &[u8], from: SocketAddr) {
        match pkt.first().copied() {
            Some(TAG_INPUT) => {
                if let Some((seq, blob, meta, name)) = decode_input(pkt, self.input_len) {
                    let Some(idx) = self.ensure_client(from) else {
                        // Lobby full: tell the joiner clearly instead of silently dropping it.
                        let _ = self.sock.send_to(&[TAG_REJECT], from);
                        return;
                    };
                    self.clients[idx].inbox.push_back((seq, blob));
                    self.clients[idx].state.meta = meta; // relay this client's self-reported metadata
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
        let Some((your_id, acked, tick, stick, players)) = decode_snapshot(pkt) else { return };
        self.my_id = your_id;
        self.tick = tick;
        self.last_server_tick = stick;
        self.last_snap_players = players.len();
        let (sim_fn, sim_env) = (self.sim_fn, self.sim_env);
        for (id, st) in players {
            if id == your_id {
                self.pred = st; // snap to authoritative, then replay unacked inputs
                while self.pending.front().map(|(s, _)| *s <= acked).unwrap_or(false) {
                    self.pending.pop_front();
                }
                let pend: Vec<(u32, InputBlob)> = self.pending.iter().copied().collect();
                for (_, inp) in pend {
                    run_sim(sim_fn, sim_env, &mut self.pred.s, &inp);
                }
            } else {
                let slot = match self.remotes.iter_mut().find(|(rid, _)| *rid == id) {
                    Some((_, r)) => r,
                    None => {
                        self.remotes.push((id, Remote { interp: InterpBuffer::new(0.06), last: st, last_seen: stick }));
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
            (r.last, r.interp.sample(self.last_server_tick as f32))
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
            self.objects.resize(n, [0.0; 3]);
        } else {
            self.objects.truncate(n);
        }
    }
    pub fn set_object(&mut self, i: usize, x: f64, y: f64, z: f64) {
        if let Some(o) = self.objects.get_mut(i) {
            *o = [x as f32, y as f32, z as f32];
        }
    }
    pub fn object_count(&self) -> usize {
        self.objects.len()
    }
    pub fn object_pos(&self, i: usize, axis: usize) -> f64 {
        self.objects.get(i).map(|o| o[axis.min(2)] as f64).unwrap_or(0.0)
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

fn encode_input(seq: u32, blob: &InputBlob, len: usize, meta: &[f32; META_LEN], name: &[u8; NAME_MAX]) -> Vec<u8> {
    let mut b = Vec::with_capacity(5 + len * 4 + META_LEN * 4 + NAME_MAX);
    b.push(TAG_INPUT);
    put_u32(&mut b, seq);
    for v in blob.iter().take(len) {
        put_f32(&mut b, *v);
    }
    for v in meta.iter() {
        put_f32(&mut b, *v);
    }
    b.extend_from_slice(name);
    b
}
fn decode_input(b: &[u8], len: usize) -> Option<(u32, InputBlob, [f32; META_LEN], [u8; NAME_MAX])> {
    if b.len() < 5 + len * 4 + META_LEN * 4 + NAME_MAX || b[0] != TAG_INPUT {
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
    Some((seq, blob, meta, name))
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

fn encode_objects(objs: &[[f32; 3]]) -> Vec<u8> {
    let mut b = Vec::with_capacity(3 + objs.len() * 12);
    b.push(TAG_OBJECTS);
    b.extend_from_slice(&(objs.len() as u16).to_be_bytes());
    for o in objs {
        for v in o {
            put_f32(&mut b, *v);
        }
    }
    b
}
fn decode_objects(b: &[u8]) -> Option<Vec<[f32; 3]>> {
    if b.len() < 3 || b[0] != TAG_OBJECTS {
        return None;
    }
    let count = u16::from_be_bytes([b[1], b[2]]) as usize;
    let mut objs = Vec::with_capacity(count);
    let mut o = 3;
    for _ in 0..count {
        if o + 12 > b.len() {
            break;
        }
        objs.push([rd_f32(b, o), rd_f32(b, o + 4), rd_f32(b, o + 8)]);
        o += 12;
    }
    Some(objs)
}
fn objects_differ(a: &[[f32; 3]], b: &[[f32; 3]]) -> bool {
    if a.len() != b.len() {
        return true;
    }
    a.iter().zip(b.iter()).any(|(x, y)| (0..3).any(|i| (x[i] - y[i]).abs() > 1e-3))
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
/// Client: 1 if the host rejected our join (lobby full), else 0.
#[no_mangle]
pub extern "C" fn aurora_net_rejected() -> i64 {
    read(0, |s| s.rejected() as i64)
}
#[no_mangle]
pub extern "C" fn aurora_net_spawn_at(x: f64, y: f64, z: f64) {
    with((), |s| s.set_spawn(x as f32, y as f32, z as f32));
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
        // NAMES replicate both ways too (read char-by-char from the replicated byte field).
        let read_name = |s: &Session, id: u32| -> String {
            (0..s.name_len(id)).map(|i| s.name_char(id, i as usize) as u8 as char).collect()
        };
        assert_eq!(read_name(&client, 0), "REAPER", "client should see the host's name");
        assert_eq!(read_name(&host, client_id), "NOVA", "host should see the client's name");
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
        // Move crate 2 (as if shot) and confirm the change replicates.
        for _ in 0..20 {
            host.set_object_count(3);
            host.set_object(0, 10.0, 0.5, -4.0);
            host.set_object(1, 11.0, 0.5, -4.0);
            host.set_object(2, 18.0, 1.5, -2.0);
            client.send_input(&input);
            host.send_input(&input);
            client.update(0.016);
            host.update(0.016);
            client.update(0.016);
        }
        assert!((client.object_pos(2, 0) - 18.0).abs() < 0.01, "moved crate 2 x = {}", client.object_pos(2, 0));
        assert!((client.object_pos(2, 1) - 1.5).abs() < 0.01, "moved crate 2 y = {}", client.object_pos(2, 1));
    }
}
