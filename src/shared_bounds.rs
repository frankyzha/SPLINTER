use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicU64, Ordering};

struct Slot {
    fingerprint: AtomicU64, // 0 = empty, else 64-bit hash of signature
    lb_bits: AtomicU64,     // f64 bits, max-CAS (init: 0.0)
    ub_bits: AtomicU64,     // f64 bits, min-CAS (init: INFINITY)
}

pub struct SharedBoundsTable {
    slots: Box<[Slot]>,
    mask: usize, // capacity - 1 (power of 2)
}

fn hash_signature(n_samples: usize, class_counts: &[i64], depth: i32) -> u64 {
    let mut h = DefaultHasher::new();
    n_samples.hash(&mut h);
    class_counts.hash(&mut h);
    depth.hash(&mut h);
    let fp = h.finish();
    // Reserve 0 for empty sentinel
    if fp == 0 {
        1
    } else {
        fp
    }
}

impl SharedBoundsTable {
    pub fn new(capacity_log2: u32) -> Self {
        let capacity = 1usize << capacity_log2;
        let mut slots = Vec::with_capacity(capacity);
        for _ in 0..capacity {
            slots.push(Slot {
                fingerprint: AtomicU64::new(0),
                lb_bits: AtomicU64::new(0.0_f64.to_bits()),
                ub_bits: AtomicU64::new(f64::INFINITY.to_bits()),
            });
        }
        SharedBoundsTable {
            slots: slots.into_boxed_slice(),
            mask: capacity - 1,
        }
    }

    /// Linear probe with max 64 steps.
    /// Returns (slot_index, found_existing) or None if probe exhausted.
    fn find_slot(&self, fingerprint: u64, start: usize) -> Option<(usize, bool)> {
        for step in 0..64 {
            let idx = (start + step) & self.mask;
            let slot = &self.slots[idx];
            let existing = slot.fingerprint.load(Ordering::Relaxed);
            if existing == fingerprint {
                return Some((idx, true));
            }
            if existing == 0 {
                return Some((idx, false));
            }
            // Collision with different key — keep probing
        }
        None
    }

    pub fn lookup(
        &self,
        n_samples: usize,
        class_counts: &[i64],
        depth: i32,
    ) -> (Option<f64>, Option<f64>) {
        let fingerprint = hash_signature(n_samples, class_counts, depth);
        let start = fingerprint as usize & self.mask;

        match self.find_slot(fingerprint, start) {
            Some((idx, true)) => {
                let slot = &self.slots[idx];
                let lb = f64::from_bits(slot.lb_bits.load(Ordering::Relaxed));
                let ub = f64::from_bits(slot.ub_bits.load(Ordering::Relaxed));
                let lb_opt = if lb > 0.0 { Some(lb) } else { None };
                let ub_opt = if ub < f64::INFINITY { Some(ub) } else { None };
                (lb_opt, ub_opt)
            }
            _ => (None, None),
        }
    }

    pub fn update_lb(&self, n_samples: usize, class_counts: &[i64], depth: i32, lb: f64) {
        if n_samples == 0 {
            return;
        }
        let fingerprint = hash_signature(n_samples, class_counts, depth);
        let start = fingerprint as usize & self.mask;

        let idx = match self.find_slot(fingerprint, start) {
            Some((idx, true)) => idx,
            Some((idx, false)) => {
                // Try to claim empty slot
                let slot = &self.slots[idx];
                match slot.fingerprint.compare_exchange(
                    0,
                    fingerprint,
                    Ordering::Relaxed,
                    Ordering::Relaxed,
                ) {
                    Ok(_) => idx,
                    Err(actual) => {
                        if actual == fingerprint {
                            idx // Same key won the race
                        } else {
                            // Different key claimed it; re-probe from next position
                            match self.find_slot(fingerprint, (idx + 1) & self.mask) {
                                Some((idx2, true)) => idx2,
                                Some((idx2, false)) => {
                                    match self.slots[idx2].fingerprint.compare_exchange(
                                        0,
                                        fingerprint,
                                        Ordering::Relaxed,
                                        Ordering::Relaxed,
                                    ) {
                                        Ok(_) => idx2,
                                        Err(a2) if a2 == fingerprint => idx2,
                                        _ => return, // Give up
                                    }
                                }
                                None => return,
                            }
                        }
                    }
                }
            }
            None => return,
        };

        // Max-CAS on lb_bits
        let slot = &self.slots[idx];
        let new_bits = lb.to_bits();
        loop {
            let current = slot.lb_bits.load(Ordering::Relaxed);
            let current_val = f64::from_bits(current);
            if lb <= current_val {
                break;
            }
            match slot.lb_bits.compare_exchange_weak(
                current,
                new_bits,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(_) => continue,
            }
        }
    }
}

// Safety: All fields are atomics, no mutable references needed
unsafe impl Sync for SharedBoundsTable {}
unsafe impl Send for SharedBoundsTable {}
