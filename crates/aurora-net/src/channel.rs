//! Reliable-ordered channel logic (netcode spec §3.1).
//!
//! The UDP transport offers per-channel reliability. This is the reliability
//! *core*: a sender that sequences messages and retransmits until acknowledged,
//! and a receiver that delivers in order (buffering out-of-order packets) and
//! produces cumulative acks. It's pure logic over byte payloads, so it's tested
//! deterministically against a simulated lossy/reordering link — the only part
//! that needs a real socket is the actual send/recv glue.

use std::collections::BTreeMap;

/// One endpoint of a reliable-ordered channel.
#[derive(Default)]
pub struct Reliable {
    // Sender state.
    send_seq: u32,
    unacked: BTreeMap<u32, Vec<u8>>,
    // Receiver state.
    recv_next: u32,
    recv_buf: BTreeMap<u32, Vec<u8>>,
}

impl Reliable {
    pub fn new() -> Reliable {
        Reliable::default()
    }

    /// Queue `payload` for reliable delivery; returns its sequence number.
    pub fn send(&mut self, payload: Vec<u8>) -> u32 {
        let seq = self.send_seq;
        self.send_seq += 1;
        self.unacked.insert(seq, payload);
        seq
    }

    /// All unacknowledged `(seq, payload)` to (re)transmit this tick.
    pub fn outgoing(&self) -> Vec<(u32, Vec<u8>)> {
        self.unacked.iter().map(|(&s, p)| (s, p.clone())).collect()
    }

    pub fn unacked_len(&self) -> usize {
        self.unacked.len()
    }

    pub fn fully_acked(&self) -> bool {
        self.unacked.is_empty()
    }

    /// Apply a peer's cumulative ack: everything with `seq < cumulative` has been
    /// received, so drop it from the retransmit buffer.
    pub fn on_ack(&mut self, cumulative: u32) {
        self.unacked.retain(|&seq, _| seq >= cumulative);
    }

    /// Receive a packet. Returns any messages now deliverable, in order.
    /// Duplicates and already-delivered sequences are ignored.
    pub fn on_recv(&mut self, seq: u32, payload: Vec<u8>) -> Vec<Vec<u8>> {
        if seq < self.recv_next {
            return Vec::new(); // duplicate / already delivered
        }
        self.recv_buf.entry(seq).or_insert(payload);

        let mut delivered = Vec::new();
        while let Some(p) = self.recv_buf.remove(&self.recv_next) {
            delivered.push(p);
            self.recv_next += 1;
        }
        delivered
    }

    /// The cumulative ack to send back: we have delivered everything below this.
    pub fn ack_num(&self) -> u32 {
        self.recv_next
    }
}

#[cfg(test)]
mod channel_tests {
    use super::*;

    #[test]
    fn in_order_delivery_no_loss() {
        let mut tx = Reliable::new();
        let mut rx = Reliable::new();
        for i in 0..5u8 {
            tx.send(vec![i]);
        }
        let mut got = Vec::new();
        for (seq, p) in tx.outgoing() {
            got.extend(rx.on_recv(seq, p));
        }
        tx.on_ack(rx.ack_num());
        assert_eq!(got, vec![vec![0], vec![1], vec![2], vec![3], vec![4]]);
        assert!(tx.fully_acked());
    }

    #[test]
    fn out_of_order_packets_are_reordered() {
        let mut rx = Reliable::new();
        // Deliver 2, then 0, then 1 — receiver must emit 0,1,2 in order.
        assert!(rx.on_recv(2, vec![2]).is_empty()); // buffered, nothing ready
        assert!(rx.on_recv(0, vec![0]) == vec![vec![0]]); // 0 ready; 1 still missing
        assert_eq!(rx.on_recv(1, vec![1]), vec![vec![1], vec![2]]); // 1 then buffered 2
        assert_eq!(rx.ack_num(), 3);
    }

    #[test]
    fn duplicates_are_ignored() {
        let mut rx = Reliable::new();
        assert_eq!(rx.on_recv(0, vec![0]), vec![vec![0]]);
        assert!(rx.on_recv(0, vec![0]).is_empty()); // duplicate
        assert_eq!(rx.ack_num(), 1);
    }

    #[test]
    fn reliable_over_a_lossy_link_eventually_delivers_in_order() {
        // Simulate A -> B over a link that deterministically drops packets, with
        // retransmission until everything is acknowledged.
        let mut a = Reliable::new();
        let mut b = Reliable::new();
        let msgs: Vec<Vec<u8>> = (0..20u8).map(|i| vec![i]).collect();
        for m in &msgs {
            a.send(m.clone());
        }

        let mut delivered: Vec<Vec<u8>> = Vec::new();
        let mut round: u32 = 0;
        while !a.fully_acked() && round < 1000 {
            round += 1;
            for (seq, payload) in a.outgoing() {
                // Deterministic, seq-dependent loss; clears on later rounds.
                let drop = (round + seq) % 3 == 0;
                if !drop {
                    delivered.extend(b.on_recv(seq, payload));
                }
            }
            // Cumulative ack flows back (assume the ack itself gets through).
            a.on_ack(b.ack_num());
        }

        assert!(a.fully_acked(), "should converge within the round budget");
        assert_eq!(delivered, msgs, "every message delivered exactly once, in order");
    }
}
