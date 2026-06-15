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

use aurora_net::InterpBuffer;

const TAG_INPUT: u8 = 1;
const TAG_SNAPSHOT: u8 = 2;

#[derive(Clone, Copy)]
struct Cfg {
    speed: f32,
    gravity: f32,
    jump: f32,
    ground: f32,
}
impl Default for Cfg {
    fn default() -> Cfg {
        Cfg { speed: 8.0, gravity: 22.0, jump: 9.0, ground: 0.0 }
    }
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
fn apply_input(p: &mut PlayerState, inp: &Input, cfg: &Cfg) {
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
    p.yaw = inp.yaw;
}

struct SClient {
    addr: SocketAddr,
    id: u32,
    state: PlayerState,
    inbox: VecDeque<Input>,
    acked_seq: u32,
}

struct Remote {
    interp: InterpBuffer,
    yaw: f32,
    last: PlayerState,
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
        }
    }

    pub fn local_addr(&self) -> SocketAddr {
        self.sock.local_addr().unwrap()
    }

    /// Apply this frame's input. On the server (host player) it is authoritative;
    /// on a client it predicts locally and sends the input to the server.
    pub fn send_input(&mut self, fwd: f32, strafe: f32, yaw: f32, jump: bool, dt: f32) -> u32 {
        if self.is_server {
            let inp = Input { seq: 0, fwd, strafe, yaw, jump, dt };
            let cfg = self.cfg;
            apply_input(&mut self.host, &inp, &cfg);
            0
        } else {
            let seq = self.next_seq;
            self.next_seq += 1;
            let inp = Input { seq, fwd, strafe, yaw, jump, dt };
            // Predict immediately for a responsive local player.
            let cfg = self.cfg;
            apply_input(&mut self.pred, &inp, &cfg);
            self.pending.push_back(inp);
            if let Some(addr) = self.server_addr {
                let pkt = encode_input(&inp);
                let _ = self.sock.send_to(&pkt, addr);
            }
            seq
        }
    }

    pub fn update(&mut self, dt: f32) {
        // Drain incoming packets.
        loop {
            match self.sock.recv_from(&mut self.buf) {
                Ok((n, from)) => {
                    let pkt = self.buf[..n].to_vec();
                    if self.is_server {
                        self.on_input_packet(&pkt, from);
                    } else {
                        self.on_snapshot_packet(&pkt);
                    }
                }
                Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => break,
                Err(_) => break,
            }
        }

        if self.is_server {
            // Apply each client's queued inputs authoritatively, then broadcast.
            let cfg = self.cfg;
            for c in &mut self.clients {
                while let Some(inp) = c.inbox.pop_front() {
                    apply_input(&mut c.state, &inp, &cfg);
                    c.acked_seq = inp.seq;
                }
            }
            self.tick += dt;
            self.broadcast();
            self.ids = std::iter::once(0u32).chain(self.clients.iter().map(|c| c.id)).collect();
        } else {
            self.tick += dt;
            self.ids = std::iter::once(self.my_id).chain(self.remotes.iter().map(|(id, _)| *id)).collect();
        }
    }

    fn broadcast(&self) {
        // Snapshot of every player (host id 0 + each client).
        let mut players: Vec<(u32, PlayerState)> = Vec::with_capacity(self.clients.len() + 1);
        players.push((0, self.host));
        for c in &self.clients {
            players.push((c.id, c.state));
        }
        for c in &self.clients {
            let pkt = encode_snapshot(c.id, c.acked_seq, self.tick, &players);
            let _ = self.sock.send_to(&pkt, c.addr);
        }
    }

    fn on_input_packet(&mut self, pkt: &[u8], from: SocketAddr) {
        let Some(inp) = decode_input(pkt) else { return };
        let idx = match self.clients.iter().position(|c| c.addr == from) {
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
                });
                self.clients.len() - 1
            }
        };
        self.clients[idx].inbox.push_back(inp);
    }

    fn on_snapshot_packet(&mut self, pkt: &[u8]) {
        let Some((your_id, acked, tick, players)) = decode_snapshot(pkt) else { return };
        self.my_id = your_id;
        self.last_snap_tick = tick;
        for (id, st) in players {
            if id == your_id {
                // Reconcile: snap to authoritative, then replay unacked inputs.
                self.pred = st;
                while self.pending.front().map(|i| i.seq <= acked).unwrap_or(false) {
                    self.pending.pop_front();
                }
                let cfg = self.cfg;
                let pending: Vec<Input> = self.pending.iter().copied().collect();
                for inp in pending {
                    apply_input(&mut self.pred, &inp, &cfg);
                }
            } else {
                let slot = match self.remotes.iter_mut().find(|(rid, _)| *rid == id) {
                    Some((_, r)) => r,
                    None => {
                        self.remotes.push((id, Remote { interp: InterpBuffer::new(0.06), yaw: st.yaw, last: st }));
                        &mut self.remotes.last_mut().unwrap().1
                    }
                };
                slot.interp.push(tick, [st.x, st.y, st.z]);
                slot.yaw = st.yaw;
                slot.last = st;
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
        self.cfg = Cfg { speed, gravity, jump, ground };
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

fn encode_snapshot(your_id: u32, acked: u32, tick: f32, players: &[(u32, PlayerState)]) -> Vec<u8> {
    let mut b = Vec::with_capacity(15 + players.len() * 24);
    b.push(TAG_SNAPSHOT);
    put_u32(&mut b, your_id);
    put_u32(&mut b, acked);
    put_f32(&mut b, tick);
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
fn decode_snapshot(b: &[u8]) -> Option<(u32, u32, f32, Vec<(u32, PlayerState)>)> {
    if b.len() < 15 || b[0] != TAG_SNAPSHOT {
        return None;
    }
    let your_id = rd_u32(b, 1);
    let acked = rd_u32(b, 5);
    let tick = rd_f32(b, 9);
    let count = u16::from_be_bytes([b[13], b[14]]) as usize;
    let mut players = Vec::with_capacity(count);
    let mut o = 15;
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
    Some((your_id, acked, tick, players))
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
