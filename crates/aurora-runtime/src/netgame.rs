//! Game-ready multiplayer for the 3D movement shooter: an authoritative UDP
//! server with N clients, client-side prediction + server reconciliation for the
//! local player, and snapshot interpolation for remote players. Exposed to
//! Aurora as `net_host`/`net_join`/`net_send_input`/`net_update` plus player
//! transform accessors that the 3D loop reads to draw everyone.
//!
//! The single movement function ([`apply_input`]) is run by BOTH the client
//! (predict + replay) and the server (authoritative), which is what structurally
//! prevents client/server drift (netcode spec §6.3).

use std::cell::RefCell;
use std::collections::VecDeque;
use std::net::{SocketAddr, ToSocketAddrs, UdpSocket};

use aurora_net::{InterpBuffer, LagComp};

const TAG_INPUT: u8 = 1;
const TAG_SNAPSHOT: u8 = 2;
const TAG_FIRE: u8 = 3;
const TAG_HIT: u8 = 4;

#[derive(Clone, Copy)]
struct Cfg {
    speed: f32,
    gravity: f32,
    jump: f32,
    ground: f32,
    /// Player collision box half-extents: radius (x/z) and half-height (y).
    pr: f32,
    ph: f32,
    /// Interest radius: a client is only told about players within this distance.
    interest: f32,
}
impl Default for Cfg {
    fn default() -> Cfg {
        Cfg { speed: 8.0, gravity: 22.0, jump: 9.0, ground: 0.0, pr: 0.4, ph: 0.9, interest: 80.0 }
    }
}

/// A static axis-aligned box the players collide against.
#[derive(Clone, Copy)]
struct Aabb {
    cx: f32,
    cy: f32,
    cz: f32,
    hx: f32,
    hy: f32,
    hz: f32,
}

#[derive(Clone, Copy)]
struct PlayerState {
    x: f32,
    y: f32,
    z: f32,
    yaw: f32,
    vy: f32,
    grounded: bool,
}
impl PlayerState {
    fn spawn() -> PlayerState {
        PlayerState { x: 0.0, y: 0.0, z: 0.0, yaw: 0.0, vy: 0.0, grounded: true }
    }
}

#[derive(Clone, Copy)]
struct Input {
    seq: u32,
    fwd: f32,
    strafe: f32,
    yaw: f32,
    jump: bool,
    dt: f32,
}

/// The shared, drift-proof movement step: run identically by client prediction
/// and the authoritative server. Horizontal move relative to yaw, plus gravity
/// and jump against a flat ground plane.
fn apply_input(p: &mut PlayerState, inp: &Input, cfg: &Cfg, walls: &[Aabb]) {
    let (s, c) = (inp.yaw.sin(), inp.yaw.cos());
    // Forward is -z at yaw 0; right is +x.
    let (fx, fz) = (s, -c);
    let (rx, rz) = (c, s);
    p.x += (fx * inp.fwd + rx * inp.strafe) * cfg.speed * inp.dt;
    p.z += (fz * inp.fwd + rz * inp.strafe) * cfg.speed * inp.dt;
    if p.grounded && inp.jump {
        p.vy = cfg.jump;
    }
    p.vy -= cfg.gravity * inp.dt;
    p.y += p.vy * inp.dt;
    if p.y <= cfg.ground {
        p.y = cfg.ground;
        p.vy = 0.0;
        p.grounded = true;
    } else {
        p.grounded = false;
    }
    resolve_walls(p, cfg, walls);
    p.yaw = inp.yaw;
}

/// Collide-and-slide the player box against static walls: push out along the
/// axis of least penetration each iteration (so motion slides along surfaces),
/// landing on tops and stopping under ceilings. Deterministic, so client
/// prediction replay and the server agree.
fn resolve_walls(p: &mut PlayerState, cfg: &Cfg, walls: &[Aabb]) {
    for _ in 0..4 {
        let mut any = false;
        for w in walls {
            let (cx, cy, cz) = (p.x, p.y + cfg.ph, p.z);
            let (dx, dy, dz) = (cx - w.cx, cy - w.cy, cz - w.cz);
            let px = (cfg.pr + w.hx) - dx.abs();
            let py = (cfg.ph + w.hy) - dy.abs();
            let pz = (cfg.pr + w.hz) - dz.abs();
            if px <= 0.0 || py <= 0.0 || pz <= 0.0 {
                continue;
            }
            any = true;
            if px <= py && px <= pz {
                p.x += if dx < 0.0 { -px } else { px };
            } else if pz <= px && pz <= py {
                p.z += if dz < 0.0 { -pz } else { pz };
            } else if dy >= 0.0 {
                p.y += py; // landed on top
                p.vy = 0.0;
                p.grounded = true;
            } else {
                p.y -= py; // bumped a ceiling
                if p.vy > 0.0 {
                    p.vy = 0.0;
                }
            }
        }
        if !any {
            break;
        }
    }
}

struct SClient {
    addr: SocketAddr,
    id: u32,
    state: PlayerState,
    inbox: VecDeque<Input>,
    acked_seq: u32,
    /// What we last told this client about each player (for delta compression).
    last_sent: std::collections::HashMap<u32, PlayerState>,
}

struct Remote {
    interp: InterpBuffer,
    yaw: f32,
    last: PlayerState,
    last_seen: u32,
}

/// Whether two player states differ enough to be worth re-sending (delta).
fn state_differs(a: &PlayerState, b: &PlayerState) -> bool {
    (a.x - b.x).abs() > 1e-3
        || (a.y - b.y).abs() > 1e-3
        || (a.z - b.z).abs() > 1e-3
        || (a.yaw - b.yaw).abs() > 1e-3
        || (a.vy - b.vy).abs() > 1e-3
}

/// One networking session (server or client).
pub struct Session {
    sock: UdpSocket,
    is_server: bool,
    server_addr: Option<SocketAddr>,
    my_id: u32,
    cfg: Cfg,
    tick: f32,
    buf: Vec<u8>,
    ids: Vec<u32>,
    // Server state.
    clients: Vec<SClient>,
    host: PlayerState,
    next_id: u32,
    // Client state.
    pred: PlayerState,
    pending: VecDeque<Input>,
    next_seq: u32,
    last_snap_tick: f32,
    remotes: Vec<(u32, Remote)>,
    // Static world collision (shared by prediction + server).
    walls: Vec<Aabb>,
    // Server-side lag compensation: a rewindable history of player colliders.
    lag: LagComp,
    server_tick: u64,
    last_server_tick: u32,
    last_snap_players: usize,
    // Last hitscan result (server: own shots; client: from a HIT packet).
    last_hit: (i64, [f32; 3]),
}

impl Session {
    pub fn host(port: u16) -> std::io::Result<Session> {
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
            cfg: Cfg::default(),
            tick: 0.0,
            buf: vec![0u8; 2048],
            ids: if is_server { vec![0] } else { vec![0] },
            clients: Vec::new(),
            host: PlayerState::spawn(),
            next_id: 1,
            pred: PlayerState::spawn(),
            pending: VecDeque::new(),
            next_seq: 1,
            last_snap_tick: 0.0,
            remotes: Vec::new(),
            walls: Vec::new(),
            lag: LagComp::new(64),
            server_tick: 0,
            last_server_tick: 0,
            last_snap_players: 0,
            last_hit: (-1, [0.0; 3]),
        }
    }

    pub fn add_wall(&mut self, x: f32, y: f32, z: f32, hx: f32, hy: f32, hz: f32) {
        self.walls.push(Aabb { cx: x, cy: y, cz: z, hx, hy, hz });
    }
    pub fn set_player_size(&mut self, radius: f32, half_height: f32) {
        self.cfg.pr = radius.max(0.01);
        self.cfg.ph = half_height.max(0.01);
    }

    pub fn local_addr(&self) -> SocketAddr {
        self.sock.local_addr().unwrap()
    }

    /// Apply this frame's input. On the server (host player) it is authoritative;
    /// on a client it predicts locally and sends the input to the server.
    pub fn send_input(&mut self, fwd: f32, strafe: f32, yaw: f32, jump: bool, dt: f32) -> u32 {
        let cfg = self.cfg;
        let walls = self.walls.clone();
        if self.is_server {
            let inp = Input { seq: 0, fwd, strafe, yaw, jump, dt };
            apply_input(&mut self.host, &inp, &cfg, &walls);
            0
        } else {
            let seq = self.next_seq;
            self.next_seq += 1;
            let inp = Input { seq, fwd, strafe, yaw, jump, dt };
            // Predict immediately for a responsive local player.
            apply_input(&mut self.pred, &inp, &cfg, &walls);
            self.pending.push_back(inp);
            if let Some(addr) = self.server_addr {
                let pkt = encode_input(&inp);
                let _ = self.sock.send_to(&pkt, addr);
            }
            seq
        }
    }

    /// Fire a hitscan ray from `(o*)` along `(d*)`. On the host it resolves
    /// authoritatively now; on a client it sends the shot with the view tick so
    /// the server can lag-compensate, and the result arrives via `net_update`.
    pub fn fire(&mut self, ox: f32, oy: f32, oz: f32, dx: f32, dy: f32, dz: f32) {
        let o = [ox, oy, oz];
        let d = [dx, dy, dz];
        if self.is_server {
            self.last_hit = match self.lag.raycast_at_tick(o, d, self.server_tick, 0) {
                Some(h) => (
                    h.entity as i64,
                    [o[0] + d[0] * h.distance, o[1] + d[1] * h.distance, o[2] + d[2] * h.distance],
                ),
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
        // Drain incoming packets.
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
            // Apply each client's queued inputs authoritatively, then broadcast.
            let cfg = self.cfg;
            let walls = self.walls.clone();
            for c in &mut self.clients {
                while let Some(inp) = c.inbox.pop_front() {
                    apply_input(&mut c.state, &inp, &cfg, &walls);
                    c.acked_seq = inp.seq;
                }
            }
            // Record collider history for lag-compensated hit validation.
            self.server_tick += 1;
            let st = self.server_tick;
            let r = (cfg.pr * cfg.pr + cfg.ph * cfg.ph).sqrt();
            self.lag.record(st, 0, [self.host.x, self.host.y + cfg.ph, self.host.z], r);
            for c in &self.clients {
                self.lag.record(st, c.id as u64, [c.state.x, c.state.y + cfg.ph, c.state.z], r);
            }
            self.tick += dt;
            self.broadcast();
            self.ids = std::iter::once(0u32).chain(self.clients.iter().map(|c| c.id)).collect();
        } else {
            self.tick += dt;
            // Drop remotes we haven't heard about for a while (left interest /
            // disconnected), so stale ghosts don't linger.
            let now = self.last_server_tick;
            self.remotes.retain(|(_, r)| now.saturating_sub(r.last_seen) <= 90);
            self.ids = std::iter::once(self.my_id).chain(self.remotes.iter().map(|(id, _)| *id)).collect();
        }
    }

    pub fn last_snapshot_players(&self) -> usize {
        self.last_snap_players
    }

    fn broadcast(&mut self) {
        // All players (host id 0 + each client).
        let mut all: Vec<(u32, PlayerState)> = Vec::with_capacity(self.clients.len() + 1);
        all.push((0, self.host));
        for c in &self.clients {
            all.push((c.id, c.state));
        }
        let interest2 = self.cfg.interest * self.cfg.interest;
        // A periodic keyframe re-syncs everything regardless of the delta state.
        let keyframe = self.server_tick % 30 == 0;
        let tick = self.tick;
        let stick = self.server_tick as u32;

        for ci in 0..self.clients.len() {
            let (cid, cpos, acked) = {
                let c = &self.clients[ci];
                (c.id, [c.state.x, c.state.y, c.state.z], c.acked_seq)
            };
            let mut included: Vec<(u32, PlayerState)> = Vec::new();
            for (id, st) in &all {
                // Interest management: skip players outside the client's radius
                // (the client itself is always relevant).
                if *id != cid {
                    let d = [st.x - cpos[0], st.y - cpos[1], st.z - cpos[2]];
                    if d[0] * d[0] + d[1] * d[1] + d[2] * d[2] > interest2 {
                        continue;
                    }
                }
                // Delta compression: only send players whose state changed since
                // we last told this client (plus periodic keyframes).
                let changed = self.clients[ci]
                    .last_sent
                    .get(id)
                    .map(|p| state_differs(p, st))
                    .unwrap_or(true);
                if changed || keyframe {
                    included.push((*id, *st));
                    self.clients[ci].last_sent.insert(*id, *st);
                }
            }
            let c = &self.clients[ci];
            let pkt = encode_snapshot(cid, acked, tick, stick, &included);
            let _ = self.sock.send_to(&pkt, c.addr);
        }
    }

    fn ensure_client(&mut self, from: SocketAddr) -> usize {
        match self.clients.iter().position(|c| c.addr == from) {
            Some(i) => i,
            None => {
                let id = self.next_id;
                self.next_id += 1;
                self.clients.push(SClient {
                    addr: from,
                    id,
                    state: PlayerState::spawn(),
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
                let Some(inp) = decode_input(pkt) else { return };
                let idx = self.ensure_client(from);
                self.clients[idx].inbox.push_back(inp);
            }
            Some(TAG_FIRE) => {
                let Some((vt, o, d)) = decode_fire(pkt) else { return };
                let Some(shooter) = self.clients.iter().find(|c| c.addr == from).map(|c| c.id) else {
                    return;
                };
                // Rewind colliders to the tick the shooter was seeing.
                let tick = (vt as u64).min(self.server_tick);
                let (id, point) = match self.lag.raycast_at_tick(o, d, tick, shooter as u64) {
                    Some(h) => (
                        h.entity as i64,
                        [o[0] + d[0] * h.distance, o[1] + d[1] * h.distance, o[2] + d[2] * h.distance],
                    ),
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
        self.last_snap_tick = tick;
        self.last_server_tick = stick;
        self.last_snap_players = players.len();
        for (id, st) in players {
            if id == your_id {
                // Reconcile: snap to authoritative, then replay unacked inputs.
                self.pred = st;
                while self.pending.front().map(|i| i.seq <= acked).unwrap_or(false) {
                    self.pending.pop_front();
                }
                let cfg = self.cfg;
                let walls = self.walls.clone();
                let pending: Vec<Input> = self.pending.iter().copied().collect();
                for inp in pending {
                    apply_input(&mut self.pred, &inp, &cfg, &walls);
                }
            } else {
                let slot = match self.remotes.iter_mut().find(|(rid, _)| *rid == id) {
                    Some((_, r)) => r,
                    None => {
                        self.remotes.push((
                            id,
                            Remote { interp: InterpBuffer::new(0.06), yaw: st.yaw, last: st, last_seen: stick },
                        ));
                        &mut self.remotes.last_mut().unwrap().1
                    }
                };
                slot.interp.push(tick, [st.x, st.y, st.z]);
                slot.yaw = st.yaw;
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
    fn player_state(&self, id: u32) -> (PlayerState, Option<[f32; 3]>) {
        if self.is_server {
            if id == 0 {
                (self.host, None)
            } else if let Some(c) = self.clients.iter().find(|c| c.id == id) {
                (c.state, None)
            } else {
                (PlayerState::spawn(), None)
            }
        } else if id == self.my_id {
            (self.pred, None)
        } else if let Some((_, r)) = self.remotes.iter().find(|(rid, _)| *rid == id) {
            (PlayerState { yaw: r.yaw, ..r.last }, r.interp.sample(self.last_snap_tick))
        } else {
            (PlayerState::spawn(), None)
        }
    }
    pub fn px(&self, id: u32) -> f64 {
        let (s, interp) = self.player_state(id);
        interp.map(|p| p[0]).unwrap_or(s.x) as f64
    }
    pub fn py(&self, id: u32) -> f64 {
        let (s, interp) = self.player_state(id);
        interp.map(|p| p[1]).unwrap_or(s.y) as f64
    }
    pub fn pz(&self, id: u32) -> f64 {
        let (s, interp) = self.player_state(id);
        interp.map(|p| p[2]).unwrap_or(s.z) as f64
    }
    pub fn pyaw(&self, id: u32) -> f64 {
        self.player_state(id).0.yaw as f64
    }
    pub fn set_cfg(&mut self, speed: f32, gravity: f32, jump: f32, ground: f32) {
        self.cfg = Cfg {
            speed,
            gravity,
            jump,
            ground,
            pr: self.cfg.pr,
            ph: self.cfg.ph,
            interest: self.cfg.interest,
        };
    }
    pub fn set_interest(&mut self, radius: f32) {
        self.cfg.interest = radius.max(0.0);
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

fn encode_input(i: &Input) -> Vec<u8> {
    let mut b = Vec::with_capacity(22);
    b.push(TAG_INPUT);
    put_u32(&mut b, i.seq);
    put_f32(&mut b, i.fwd);
    put_f32(&mut b, i.strafe);
    put_f32(&mut b, i.yaw);
    b.push(i.jump as u8);
    put_f32(&mut b, i.dt);
    b
}
fn decode_input(b: &[u8]) -> Option<Input> {
    if b.len() < 22 || b[0] != TAG_INPUT {
        return None;
    }
    Some(Input {
        seq: rd_u32(b, 1),
        fwd: rd_f32(b, 5),
        strafe: rd_f32(b, 9),
        yaw: rd_f32(b, 13),
        jump: b[17] != 0,
        dt: rd_f32(b, 18),
    })
}

fn encode_snapshot(your_id: u32, acked: u32, tick: f32, stick: u32, players: &[(u32, PlayerState)]) -> Vec<u8> {
    let mut b = Vec::with_capacity(19 + players.len() * 24);
    b.push(TAG_SNAPSHOT);
    put_u32(&mut b, your_id);
    put_u32(&mut b, acked);
    put_f32(&mut b, tick);
    put_u32(&mut b, stick);
    b.extend_from_slice(&(players.len() as u16).to_be_bytes());
    for (id, s) in players {
        put_u32(&mut b, *id);
        put_f32(&mut b, s.x);
        put_f32(&mut b, s.y);
        put_f32(&mut b, s.z);
        put_f32(&mut b, s.yaw);
        put_f32(&mut b, s.vy);
    }
    b
}
fn decode_snapshot(b: &[u8]) -> Option<(u32, u32, f32, u32, Vec<(u32, PlayerState)>)> {
    if b.len() < 19 || b[0] != TAG_SNAPSHOT {
        return None;
    }
    let your_id = rd_u32(b, 1);
    let acked = rd_u32(b, 5);
    let tick = rd_f32(b, 9);
    let stick = rd_u32(b, 13);
    let count = u16::from_be_bytes([b[17], b[18]]) as usize;
    let mut players = Vec::with_capacity(count);
    let mut o = 19;
    for _ in 0..count {
        if o + 24 > b.len() {
            break;
        }
        let id = rd_u32(b, o);
        let st = PlayerState {
            x: rd_f32(b, o + 4),
            y: rd_f32(b, o + 8),
            z: rd_f32(b, o + 12),
            yaw: rd_f32(b, o + 16),
            vy: rd_f32(b, o + 20),
            grounded: false,
        };
        players.push((id, st));
        o += 24;
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

/// Start an authoritative server on `port` (the host is also player 0).
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

/// Join a server at `host:port` as a predicting client.
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

/// Tune the shared movement model (speed, gravity, jump impulse, ground height).
#[no_mangle]
pub extern "C" fn aurora_net_config(speed: f64, gravity: f64, jump: f64, ground: f64) {
    NET.with(|n| {
        if let Some(s) = n.borrow_mut().as_mut() {
            s.set_cfg(speed as f32, gravity as f32, jump as f32, ground as f32);
        }
    });
}

/// Register a static collision box (center + half-extents) for net movement.
#[no_mangle]
pub extern "C" fn aurora_net_add_wall(x: f64, y: f64, z: f64, hx: f64, hy: f64, hz: f64) {
    NET.with(|n| {
        if let Some(s) = n.borrow_mut().as_mut() {
            s.add_wall(x as f32, y as f32, z as f32, hx as f32, hy as f32, hz as f32);
        }
    });
}

/// Set the player collision capsule-ish box: radius (x/z) and half-height (y).
#[no_mangle]
pub extern "C" fn aurora_net_player_size(radius: f64, half_height: f64) {
    NET.with(|n| {
        if let Some(s) = n.borrow_mut().as_mut() {
            s.set_player_size(radius as f32, half_height as f32);
        }
    });
}

/// Set the interest radius: clients are only told about players within it.
#[no_mangle]
pub extern "C" fn aurora_net_interest(radius: f64) {
    NET.with(|n| {
        if let Some(s) = n.borrow_mut().as_mut() {
            s.set_interest(radius as f32);
        }
    });
}

/// Submit this frame's input; returns the input sequence number (0 on the host).
#[no_mangle]
pub extern "C" fn aurora_net_send_input(fwd: f64, strafe: f64, yaw: f64, jump: i64, dt: f64) -> i64 {
    NET.with(|n| {
        n.borrow_mut()
            .as_mut()
            .map(|s| s.send_input(fwd as f32, strafe as f32, yaw as f32, jump != 0, dt as f32) as i64)
            .unwrap_or(0)
    })
}

/// Pump the network: receive, simulate (server), reconcile + interpolate (client).
#[no_mangle]
pub extern "C" fn aurora_net_update(dt: f64) {
    NET.with(|n| {
        if let Some(s) = n.borrow_mut().as_mut() {
            s.update(dt as f32);
        }
    });
}

/// Fire a hitscan shot from origin `(ox,oy,oz)` along direction `(dx,dy,dz)`.
#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub extern "C" fn aurora_net_fire(ox: f64, oy: f64, oz: f64, dx: f64, dy: f64, dz: f64) {
    NET.with(|n| {
        if let Some(s) = n.borrow_mut().as_mut() {
            s.fire(ox as f32, oy as f32, oz as f32, dx as f32, dy as f32, dz as f32);
        }
    });
}
/// The player id hit by the last shot (server-validated), or -1.
#[no_mangle]
pub extern "C" fn aurora_net_hit_player() -> i64 {
    NET.with(|n| n.borrow().as_ref().map(|s| s.hit_player()).unwrap_or(-1))
}
fn hit_axis(i: usize) -> f64 {
    NET.with(|n| n.borrow().as_ref().map(|s| s.hit_point()[i] as f64).unwrap_or(0.0))
}
#[no_mangle]
pub extern "C" fn aurora_net_hit_x() -> f64 { hit_axis(0) }
#[no_mangle]
pub extern "C" fn aurora_net_hit_y() -> f64 { hit_axis(1) }
#[no_mangle]
pub extern "C" fn aurora_net_hit_z() -> f64 { hit_axis(2) }

#[no_mangle]
pub extern "C" fn aurora_net_my_id() -> i64 {
    NET.with(|n| n.borrow().as_ref().map(|s| s.my_id() as i64).unwrap_or(0))
}
#[no_mangle]
pub extern "C" fn aurora_net_is_server() -> i64 {
    NET.with(|n| n.borrow().as_ref().map(|s| s.is_server() as i64).unwrap_or(0))
}
#[no_mangle]
pub extern "C" fn aurora_net_player_count() -> i64 {
    NET.with(|n| n.borrow().as_ref().map(|s| s.player_count() as i64).unwrap_or(0))
}
#[no_mangle]
pub extern "C" fn aurora_net_player_id_at(i: i64) -> i64 {
    NET.with(|n| n.borrow().as_ref().map(|s| s.player_id_at(i.max(0) as usize)).unwrap_or(-1))
}
fn paxis(id: i64, axis: u8) -> f64 {
    NET.with(|n| {
        n.borrow()
            .as_ref()
            .map(|s| {
                let id = id.max(0) as u32;
                match axis {
                    0 => s.px(id),
                    1 => s.py(id),
                    2 => s.pz(id),
                    _ => s.pyaw(id),
                }
            })
            .unwrap_or(0.0)
    })
}
#[no_mangle]
pub extern "C" fn aurora_net_player_x(id: i64) -> f64 { paxis(id, 0) }
#[no_mangle]
pub extern "C" fn aurora_net_player_y(id: i64) -> f64 { paxis(id, 1) }
#[no_mangle]
pub extern "C" fn aurora_net_player_z(id: i64) -> f64 { paxis(id, 2) }
#[no_mangle]
pub extern "C" fn aurora_net_player_yaw(id: i64) -> f64 { paxis(id, 3) }

fn local_axis(axis: u8) -> f64 {
    NET.with(|n| {
        n.borrow()
            .as_ref()
            .map(|s| {
                let id = s.my_id() as i64;
                match axis {
                    0 => s.px(id as u32),
                    1 => s.py(id as u32),
                    2 => s.pz(id as u32),
                    _ => s.pyaw(id as u32),
                }
            })
            .unwrap_or(0.0)
    })
}
#[no_mangle]
pub extern "C" fn aurora_net_local_x() -> f64 { local_axis(0) }
#[no_mangle]
pub extern "C" fn aurora_net_local_y() -> f64 { local_axis(1) }
#[no_mangle]
pub extern "C" fn aurora_net_local_z() -> f64 { local_axis(2) }
#[no_mangle]
pub extern "C" fn aurora_net_local_yaw() -> f64 { local_axis(3) }

#[cfg(test)]
mod tests {
    use super::*;

    /// Server + client over real loopback UDP: the client predicts immediately,
    /// the server authoritatively simulates, and the client's reconciled position
    /// converges to the server's with no drift.
    #[test]
    fn client_prediction_reconciles_to_authoritative_server() {
        let mut server = Session::host(0).unwrap();
        let saddr = server.local_addr();
        let mut client = Session::join(saddr).unwrap();
        let dt = 1.0 / 60.0;

        // Drive ~2 seconds: the client walks forward (-z) the whole time.
        for _ in 0..120 {
            client.send_input(1.0, 0.0, 0.0, false, dt);
            client.update(dt);
            server.update(dt);
            std::thread::sleep(std::time::Duration::from_micros(300));
            client.update(dt);
        }

        // The client adopted a non-zero id and has a remote-free local prediction.
        assert!(client.my_id() >= 1, "client should be assigned an id, got {}", client.my_id());
        let cid = client.my_id();
        // Server registered the client and simulated it forward.
        let server_z = server.pz(cid);
        assert!(server_z < -1.0, "server should have moved the client forward (-z), got {server_z}");
        // Client's predicted/reconciled z matches the server's (no drift).
        let client_z = client.pz(cid);
        assert!((client_z - server_z).abs() < 0.5, "client {client_z} should converge to server {server_z}");
    }

    #[test]
    fn net_movement_collides_with_a_wall() {
        // A wall at z = -3 blocks a player walking forward (-z); the shared model
        // resolves it identically for the host (authoritative) here.
        let mut server = Session::host(0).unwrap();
        server.add_wall(0.0, 1.0, -3.0, 5.0, 1.0, 0.5); // spans z in [-3.5, -2.5]
        let dt = 1.0 / 60.0;
        for _ in 0..180 {
            server.send_input(1.0, 0.0, 0.0, false, dt); // walk forward forever
        }
        let z = server.pz(0);
        // Without the wall the host would reach z ~ -24; the wall stops it near
        // its front face (-2.5) minus the player radius.
        assert!(z > -3.0, "wall should stop the player, got z={z}");
        assert!(z < -1.5, "player should reach the wall, got z={z}");
    }

    #[test]
    fn lag_compensated_shot_hits_a_target() {
        use std::f32::consts::PI;
        let mut server = Session::host(0).unwrap();
        let saddr = server.local_addr();
        let mut a = Session::join(saddr).unwrap(); // shooter, stays at origin
        let mut b = Session::join(saddr).unwrap(); // target, walks +x
        let dt = 1.0 / 60.0;

        for _ in 0..90 {
            server.send_input(1.0, 0.0, -PI / 2.0, false, dt); // host walks -x, out of the way
            a.send_input(0.0, 0.0, 0.0, false, dt);
            b.send_input(1.0, 0.0, PI / 2.0, false, dt); // B walks +x
            a.update(dt);
            b.update(dt);
            server.update(dt);
            std::thread::sleep(std::time::Duration::from_micros(300));
            a.update(dt);
            b.update(dt);
        }
        let bid = b.my_id();
        assert!(bid >= 1);

        // A fires from its eye straight along +x, where B is.
        a.fire(0.0, 0.9, 0.0, 1.0, 0.0, 0.0);
        for _ in 0..10 {
            a.update(dt);
            server.update(dt);
            std::thread::sleep(std::time::Duration::from_micros(300));
            a.update(dt);
        }
        assert_eq!(a.hit_player(), bid as i64, "the rewound shot should hit B");
        assert!(a.hit_point()[0] > 0.5, "hit point should be out along +x, got {:?}", a.hit_point());
    }

    #[test]
    fn interest_management_culls_distant_players() {
        use std::f32::consts::PI;
        let mut server = Session::host(0).unwrap();
        server.set_interest(5.0); // tiny radius
        let saddr = server.local_addr();
        let mut a = Session::join(saddr).unwrap();
        let mut b = Session::join(saddr).unwrap();
        let dt = 1.0 / 60.0;
        // B walks far away (+x) past the interest radius; A stays put.
        for _ in 0..150 {
            server.send_input(1.0, 0.0, -PI / 2.0, false, dt); // host away (-x)
            a.send_input(0.0, 0.0, 0.0, false, dt);
            b.send_input(1.0, 0.0, PI / 2.0, false, dt);
            a.update(dt);
            b.update(dt);
            server.update(dt);
            std::thread::sleep(std::time::Duration::from_micros(300));
            a.update(dt);
            b.update(dt);
        }
        // A should no longer see B (out of interest) -> only itself.
        assert_eq!(a.player_count(), 1, "A should have culled the distant B");
    }

    #[test]
    fn delta_compression_shrinks_idle_snapshots() {
        let mut server = Session::host(0).unwrap();
        let saddr = server.local_addr();
        let mut a = Session::join(saddr).unwrap();
        let dt = 1.0 / 60.0;
        // Register, then go idle and let several non-keyframe ticks pass.
        for _ in 0..40 {
            a.send_input(0.0, 0.0, 0.0, false, dt); // no movement
            a.update(dt);
            server.update(dt);
            std::thread::sleep(std::time::Duration::from_micros(300));
            a.update(dt);
        }
        // An idle player produces empty (delta) snapshots between keyframes; over
        // many ticks the client must have received at least one zero-player one.
        // (We can't observe every packet, but the latest should reflect delta.)
        assert!(a.last_snapshot_players() <= 1, "idle snapshots should be delta-compressed, got {}", a.last_snapshot_players());
    }

    #[test]
    fn remote_player_is_interpolated_on_other_clients() {
        let mut server = Session::host(0).unwrap();
        let saddr = server.local_addr();
        let mut a = Session::join(saddr).unwrap();
        let mut b = Session::join(saddr).unwrap();
        let dt = 1.0 / 60.0;

        for _ in 0..120 {
            a.send_input(1.0, 0.0, 0.0, false, dt); // A strafes/walks
            b.send_input(0.0, 0.0, 0.0, false, dt); // B stands
            a.update(dt);
            b.update(dt);
            server.update(dt);
            std::thread::sleep(std::time::Duration::from_micros(300));
            a.update(dt);
            b.update(dt);
        }
        // B should see A as a remote player that has moved (-z), via interpolation.
        let aid = a.my_id();
        assert!(b.player_count() >= 2, "B should see at least 2 players");
        let a_seen_by_b = b.pz(aid);
        assert!(a_seen_by_b < -0.5, "B should see A moved forward via interp, got {a_seen_by_b}");
    }
}
