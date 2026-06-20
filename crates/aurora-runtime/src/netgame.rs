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

const STATE_MAX: usize = 32; // max floats in a player state blob
const INPUT_MAX: usize = 24; // max floats in an input blob
const META_LEN: usize = 8; // per-player metadata floats (hp/shield/cells/oc/name...) replicated
                           // SEPARATELY from the sim state, so they never touch reconciliation.

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
}
impl Player {
    fn spawn() -> Player {
        Player { s: [0.0; STATE_MAX], meta: [0.0; META_LEN] }
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
    next_id: u32,
    lag: LagComp,
    server_tick: u64,
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
            next_id: 1,
            lag: LagComp::new(64),
            server_tick: 0,
            pred: Player::spawn(),
            pending: VecDeque::new(),
            next_seq: 1,
            last_server_tick: 0,
            last_snap_players: 0,
            remotes: Vec::new(),
            last_hit: (-1, [0.0; 3]),
            spawn: [0.0, 0.0, 0.0],
            local_meta: [0.0; META_LEN],
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
            self.host.meta = self.local_meta; // host owns its own metadata
            0
        } else {
            let seq = self.next_seq;
            self.next_seq += 1;
            run_sim(self.sim_fn, self.sim_env, &mut self.pred.s, &blob);
            self.pred.meta = self.local_meta;
            self.pending.push_back((seq, blob));
            if let Some(addr) = self.server_addr {
                let _ = self.sock.send_to(&encode_input(seq, &blob, self.input_len, &self.local_meta), addr);
            }
            seq
        }
    }

    /// Fire a hitscan ray. On the host it resolves now; on a client it sends the
    /// shot with the view tick so the server can lag-compensate.
    pub fn fire(&mut self, ox: f32, oy: f32, oz: f32, dx: f32, dy: f32, dz: f32) {
        let o = [ox, oy, oz];
        let d = [dx, dy, dz];
        if self.is_server {
            self.last_hit = match self.lag.raycast_at_tick(o, d, self.server_tick, 0) {
                Some(h) => (h.entity as i64, [o[0] + d[0] * h.distance, o[1] + d[1] * h.distance, o[2] + d[2] * h.distance]),
                None => (-1, [0.0; 3]),
            };
        } else if let Some(addr) = self.server_addr {
            let vt = self.last_server_tick.saturating_sub(2);
            let _ = self.sock.send_to(&encode_fire(vt, o, d), addr);
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
        let mut all: Vec<(u32, Player)> = Vec::with_capacity(self.clients.len() + 1);
        all.push((0, self.host));
        for c in &self.clients {
            all.push((c.id, c.state));
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
                    .map(|p| state_differs(&p.s, &st.s, slen) || meta_differs(&p.meta, &st.meta))
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
    }

    fn ensure_client(&mut self, from: SocketAddr) -> usize {
        match self.clients.iter().position(|c| c.addr == from) {
            Some(i) => i,
            None => {
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
                self.clients.len() - 1
            }
        }
    }

    fn on_server_packet(&mut self, pkt: &[u8], from: SocketAddr) {
        match pkt.first().copied() {
            Some(TAG_INPUT) => {
                if let Some((seq, blob, meta)) = decode_input(pkt, self.input_len) {
                    let idx = self.ensure_client(from);
                    self.clients[idx].inbox.push_back((seq, blob));
                    self.clients[idx].state.meta = meta; // relay this client's self-reported metadata
                }
            }
            Some(TAG_FIRE) => {
                let Some((vt, o, d)) = decode_fire(pkt) else { return };
                let Some(shooter) = self.clients.iter().find(|c| c.addr == from).map(|c| c.id) else {
                    return;
                };
                let tick = (vt as u64).min(self.server_tick);
                let (id, point) = match self.lag.raycast_at_tick(o, d, tick, shooter as u64) {
                    Some(h) => (h.entity as i64, [o[0] + d[0] * h.distance, o[1] + d[1] * h.distance, o[2] + d[2] * h.distance]),
                    None => (-1, [0.0; 3]),
                };
                let _ = self.sock.send_to(&encode_hit(id, point), from);
            }
            _ => {}
        }
    }

    fn on_client_packet(&mut self, pkt: &[u8]) {
        if pkt.first().copied() == Some(TAG_HIT) {
            if let Some((id, p)) = decode_hit(pkt) {
                self.last_hit = (id, p);
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
    pub fn local_state(&self, i: usize) -> f64 {
        self.state(self.my_id, i)
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

fn encode_input(seq: u32, blob: &InputBlob, len: usize, meta: &[f32; META_LEN]) -> Vec<u8> {
    let mut b = Vec::with_capacity(5 + len * 4 + META_LEN * 4);
    b.push(TAG_INPUT);
    put_u32(&mut b, seq);
    for v in blob.iter().take(len) {
        put_f32(&mut b, *v);
    }
    for v in meta.iter() {
        put_f32(&mut b, *v);
    }
    b
}
fn decode_input(b: &[u8], len: usize) -> Option<(u32, InputBlob, [f32; META_LEN])> {
    if b.len() < 5 + len * 4 + META_LEN * 4 || b[0] != TAG_INPUT {
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
    Some((seq, blob, meta))
}

fn encode_snapshot(your_id: u32, acked: u32, tick: f32, stick: u32, slen: usize, players: &[(u32, Player)]) -> Vec<u8> {
    let mut b = Vec::with_capacity(19 + players.len() * (4 + (slen + META_LEN) * 4));
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
    let stride = 4 + (slen + META_LEN) * 4;
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
        players.push((id, p));
        o += stride;
    }
    Some((your_id, acked, tick, stick, players))
}

fn encode_fire(view_tick: u32, o: [f32; 3], d: [f32; 3]) -> Vec<u8> {
    let mut b = Vec::with_capacity(29);
    b.push(TAG_FIRE);
    put_u32(&mut b, view_tick);
    for v in o.iter().chain(d.iter()) {
        put_f32(&mut b, *v);
    }
    b
}
fn decode_fire(b: &[u8]) -> Option<(u32, [f32; 3], [f32; 3])> {
    if b.len() < 29 || b[0] != TAG_FIRE {
        return None;
    }
    let vt = rd_u32(b, 1);
    let o = [rd_f32(b, 5), rd_f32(b, 9), rd_f32(b, 13)];
    let d = [rd_f32(b, 17), rd_f32(b, 21), rd_f32(b, 25)];
    Some((vt, o, d))
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
pub extern "C" fn aurora_net_fire(ox: f64, oy: f64, oz: f64, dx: f64, dy: f64, dz: f64) {
    with((), |s| s.fire(ox as f32, oy as f32, oz as f32, dx as f32, dy as f32, dz as f32));
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
        a.fire(0.0, 0.0, 0.0, 1.0, 0.0, 0.0);
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
    }
}
