//! Real UDP transport (netcode spec §3.1, the socket glue).
//!
//! [`Reliable`] is the reliability *logic*; this wraps it around an actual
//! non-blocking [`UdpSocket`] so two processes (or two endpoints on localhost)
//! exchange reliable-ordered messages over the wire. Each datagram piggybacks
//! the sender's cumulative ack, so acknowledgements flow back without a separate
//! channel.
//!
//! Wire format (big-endian):
//! ```text
//!   ack: u32 | flag: u8 | [ seq: u32 | payload: bytes ]   (flag = 1 if data)
//! ```

use std::io;
use std::net::{SocketAddr, ToSocketAddrs, UdpSocket};

use crate::channel::Reliable;

/// A reliable-ordered endpoint over a connected UDP socket.
pub struct UdpEndpoint {
    sock: UdpSocket,
    chan: Reliable,
    buf: Vec<u8>,
}

impl UdpEndpoint {
    /// Bind to `local` (use port 0 for an ephemeral port). The peer is set later
    /// with [`connect`](Self::connect).
    pub fn bind(local: impl ToSocketAddrs) -> io::Result<UdpEndpoint> {
        let sock = UdpSocket::bind(local)?;
        sock.set_nonblocking(true)?;
        Ok(UdpEndpoint { sock, chan: Reliable::new(), buf: vec![0u8; 2048] })
    }

    /// Point this endpoint at `peer`; afterwards send/recv talk only to it.
    pub fn connect(&mut self, peer: impl ToSocketAddrs) -> io::Result<()> {
        self.sock.connect(peer)
    }

    pub fn local_addr(&self) -> io::Result<SocketAddr> {
        self.sock.local_addr()
    }

    /// Queue `msg` for reliable, ordered delivery. Returns its sequence number.
    pub fn queue(&mut self, msg: Vec<u8>) -> u32 {
        self.chan.send(msg)
    }

    pub fn fully_acked(&self) -> bool {
        self.chan.fully_acked()
    }

    /// (Re)transmit every unacknowledged message and a trailing ack-only packet,
    /// so the peer always learns our latest cumulative ack. Returns the number of
    /// datagrams sent.
    pub fn flush(&mut self) -> io::Result<usize> {
        let ack = self.chan.ack_num();
        let mut sent = 0;
        for (seq, payload) in self.chan.outgoing() {
            let pkt = encode_data(ack, seq, &payload);
            self.sock.send(&pkt)?;
            sent += 1;
        }
        self.sock.send(&encode_ack(ack))?;
        Ok(sent + 1)
    }

    /// Drain all pending datagrams: apply piggybacked acks and return any
    /// messages now deliverable, in order. Non-blocking — returns what's ready.
    pub fn poll(&mut self) -> Vec<Vec<u8>> {
        let mut delivered = Vec::new();
        loop {
            match self.sock.recv(&mut self.buf) {
                Ok(n) => {
                    if let Some((ack, data)) = decode(&self.buf[..n]) {
                        self.chan.on_ack(ack);
                        if let Some((seq, payload)) = data {
                            delivered.extend(self.chan.on_recv(seq, payload));
                        }
                    }
                }
                Err(ref e) if e.kind() == io::ErrorKind::WouldBlock => break,
                Err(_) => break,
            }
        }
        delivered
    }
}

fn encode_data(ack: u32, seq: u32, payload: &[u8]) -> Vec<u8> {
    let mut b = Vec::with_capacity(9 + payload.len());
    b.extend_from_slice(&ack.to_be_bytes());
    b.push(1);
    b.extend_from_slice(&seq.to_be_bytes());
    b.extend_from_slice(payload);
    b
}

fn encode_ack(ack: u32) -> Vec<u8> {
    let mut b = Vec::with_capacity(5);
    b.extend_from_slice(&ack.to_be_bytes());
    b.push(0);
    b
}

/// Returns `(cumulative_ack, optional (seq, payload))`, or `None` if malformed.
fn decode(pkt: &[u8]) -> Option<(u32, Option<(u32, Vec<u8>)>)> {
    if pkt.len() < 5 {
        return None;
    }
    let ack = u32::from_be_bytes([pkt[0], pkt[1], pkt[2], pkt[3]]);
    match pkt[4] {
        0 => Some((ack, None)),
        1 if pkt.len() >= 9 => {
            let seq = u32::from_be_bytes([pkt[5], pkt[6], pkt[7], pkt[8]]);
            Some((ack, Some((seq, pkt[9..].to_vec()))))
        }
        _ => None,
    }
}

#[cfg(test)]
mod transport_tests {
    use super::*;

    /// Two endpoints on localhost exchange messages over **real UDP sockets**,
    /// driving retransmission until every message is acknowledged and delivered
    /// in order.
    #[test]
    fn two_endpoints_exchange_over_real_udp() {
        let mut a = UdpEndpoint::bind("127.0.0.1:0").unwrap();
        let mut b = UdpEndpoint::bind("127.0.0.1:0").unwrap();
        let a_addr = a.local_addr().unwrap();
        let b_addr = b.local_addr().unwrap();
        a.connect(b_addr).unwrap();
        b.connect(a_addr).unwrap();

        let msgs: Vec<Vec<u8>> = (0..16u8).map(|i| vec![i, i.wrapping_mul(3)]).collect();
        for m in &msgs {
            a.queue(m.clone());
        }

        let mut got: Vec<Vec<u8>> = Vec::new();
        let mut round = 0;
        while !a.fully_acked() && round < 200 {
            round += 1;
            a.flush().unwrap();
            // Give the loopback stack a moment to deliver, then drain.
            std::thread::sleep(std::time::Duration::from_millis(1));
            got.extend(b.poll());
            b.flush().unwrap(); // sends acks back
            std::thread::sleep(std::time::Duration::from_millis(1));
            a.poll();
        }

        assert!(a.fully_acked(), "all messages should be acked within budget");
        assert_eq!(got, msgs, "received every message exactly once, in order");
    }

    #[test]
    fn decode_rejects_malformed() {
        assert!(decode(&[]).is_none());
        assert!(decode(&[0, 0, 0]).is_none());
        assert!(decode(&[0, 0, 0, 1, 1]).is_none()); // flag=data but no seq
        assert_eq!(decode(&[0, 0, 0, 7, 0]), Some((7, None))); // ack-only
    }
}
