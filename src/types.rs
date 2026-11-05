use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use dashmap::DashMap;
use smallvec::SmallVec;

/// Hard cap on the number of classes. Class-count buffers (per-node /
/// per-bin / per-leaf) are stack-allocated `[T; MAX_CLASSES]` so the hot
/// loops avoid heap traffic. n_classes==2 (binary) only touches indices
/// 0..2; the unused tail is zero-init but never read. Bumping this widens
/// the supported class range at the cost of (MAX_CLASSES - 8) * 8B per
/// stack array — fine at 16, watch the per-call cost above ~32.
pub const MAX_CLASSES: usize = 16;

// ---------------------------------------------------------------------------
// Threshold interning
// ---------------------------------------------------------------------------

/// Concurrent interner for threshold vectors. Deduplicates identical
/// threshold sets so that many SplitRecords / SplitSpecs sharing the same
/// (feature, thresholds) split point to a single heap allocation.
///
/// Lossless: uses exact bit-level equality (f64::to_bits) for hashing.
pub struct ThresholdInterner {
    map: DashMap<u64, Arc<[f64]>>,
}

/// Content-hash-keyed dedup pool for cost-entry schedules
/// (Arc-sharing): identical `Vec<CostEntry>` schedules across
/// FactoredSplit children share a single heap allocation via `Arc::clone`.
/// This matters when many splits' children point to
/// the same `(sub_ref, scope)` and thus produce the same schedule.
pub struct ScheduleInterner {
    map: DashMap<u64, Arc<Vec<CostEntry>>>,
    pub intern_attempts: std::sync::atomic::AtomicU64,
    pub intern_hits: std::sync::atomic::AtomicU64,
}

impl ScheduleInterner {
    pub fn new() -> Self {
        Self {
            map: DashMap::new(),
            intern_attempts: std::sync::atomic::AtomicU64::new(0),
            intern_hits: std::sync::atomic::AtomicU64::new(0),
        }
    }

    /// Intern a schedule by content hash. Returns a shared `Arc` if the
    /// content matches an existing entry; otherwise wraps and stores.
    pub fn intern(&self, schedule: Vec<CostEntry>) -> Arc<Vec<CostEntry>> {
        self.intern_attempts
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let h = Self::hash_schedule(&schedule);
        // Lookup-then-insert pattern. If multiple threads insert the same
        // hash concurrently the loser's allocation is simply dropped on the
        // returned Arc going out of scope — safe and rare (build paths are
        // single-threaded per root in practice).
        if let Some(existing) = self.map.get(&h) {
            // Probabilistic: if hash collides on different content, we'd
            // alias — but with FxHash-of-bits the collision rate at our
            // scale (10s of millions of schedules) is astronomically low
            // and the cost is at most a slight numeric drift in extraction
            // that the soundness checks would surface. We accept it.
            self.intern_hits
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            return existing.clone();
        }
        let arc = Arc::new(schedule);
        self.map.insert(h, arc.clone());
        arc
    }

    fn hash_schedule(schedule: &[CostEntry]) -> u64 {
        let mut h = DefaultHasher::new();
        schedule.len().hash(&mut h);
        for ce in schedule {
            ce.cost.to_bits().hash(&mut h);
            ce.tree_count().hash(&mut h);
        }
        h.finish()
    }

    pub fn unique_count(&self) -> usize {
        self.map.len()
    }

    pub fn approximate_bytes(&self) -> u64 {
        let entry_bytes = std::mem::size_of::<CostEntry>() as u64;
        self.map
            .iter()
            .map(|e| {
                let v = e.value();
                (v.len() as u64) * entry_bytes
            })
            .sum()
    }
}

impl Default for ScheduleInterner {
    fn default() -> Self {
        Self::new()
    }
}

thread_local! {
    /// Thread-local schedule interner for use during DAG construction in
    /// contexts where threading the interner explicitly would require
    /// invasive signature changes (e.g. `rashomon_dag::build_factored_recursive`).
    /// Cleared at the start of `build_factored_dag` and stats are read at
    /// the end before clearing.
    pub static SCHEDULE_INTERNER_TL: std::cell::RefCell<ScheduleInterner> =
        std::cell::RefCell::new(ScheduleInterner::new());
}

/// Helper to intern a schedule via the thread-local interner. Used by
/// `rashomon_dag::build_factored_recursive` to share Arcs across splits.
pub fn intern_schedule_tl(schedule: Vec<CostEntry>) -> Arc<Vec<CostEntry>> {
    SCHEDULE_INTERNER_TL.with(|cell| cell.borrow().intern(schedule))
}

impl ThresholdInterner {
    pub fn new() -> Self {
        Self {
            map: DashMap::new(),
        }
    }

    /// Intern a threshold vector: return a shared `Arc<[f64]>` that may be
    /// reused across many split records. First call for a given bit-exact
    /// threshold set allocates; subsequent calls return the same Arc.
    pub fn intern(&self, thresholds: Vec<f64>) -> Arc<[f64]> {
        self.intern_slice(&thresholds)
    }

    /// Intern from a slice (avoids allocation when the data already exists).
    /// The map is keyed by a 64-bit content hash; a hit is returned only
    /// when its contents are bitwise equal, a colliding set gets its own
    /// (uninterned) allocation.
    pub fn intern_slice(&self, thresholds: &[f64]) -> Arc<[f64]> {
        let key = Self::hash_thresholds(thresholds);
        let arc = self
            .map
            .entry(key)
            .or_insert_with(|| Arc::from(thresholds))
            .clone();
        let same = arc.len() == thresholds.len()
            && arc
                .iter()
                .zip(thresholds)
                .all(|(a, b)| a.to_bits() == b.to_bits());
        if same {
            arc
        } else {
            Arc::from(thresholds)
        }
    }

    fn hash_thresholds(thresholds: &[f64]) -> u64 {
        let mut hasher = DefaultHasher::new();
        thresholds.len().hash(&mut hasher);
        for &t in thresholds {
            t.to_bits().hash(&mut hasher);
        }
        hasher.finish()
    }

    pub fn len(&self) -> usize {
        self.map.len()
    }
}

/// One step in a split path: identifies which bin of a k-way split this
/// subproblem came from. A sequence of SplitSteps from root to here
/// uniquely identifies the subproblem and allows bitmask reconstruction.
#[derive(Clone, Debug, Hash, Eq, PartialEq)]
pub struct SplitStep {
    pub feature_index: u16,
    /// Index into BitmaskContext::feature_thresholds[feature_index] where
    /// this split's k-1 thresholds start.
    pub threshold_base: u16,
    /// Which bin of the k-way split (0..k-1).
    pub bin: u8,
    /// Arity of the split.
    pub k: u8,
}

#[derive(Clone, Debug)]
pub struct SupportMask {
    pub words: Arc<[u64]>,
    pub n_samples: usize,
    pub fingerprint: u64,
}

impl SupportMask {
    pub fn from_words(words: Vec<u64>, n_samples: usize) -> Self {
        let words: Arc<[u64]> = Arc::from(words);
        let fingerprint = Self::fingerprint(words.as_ref(), n_samples);
        SupportMask {
            words,
            n_samples,
            fingerprint,
        }
    }

    pub fn from_arc(words: Arc<[u64]>, n_samples: usize) -> Self {
        let fingerprint = Self::fingerprint(words.as_ref(), n_samples);
        SupportMask {
            words,
            n_samples,
            fingerprint,
        }
    }

    pub fn root(n_samples: usize) -> Self {
        let words_per_mask = n_samples.div_ceil(64);
        let mut words = vec![!0u64; words_per_mask];
        let remainder = n_samples % 64;
        if remainder > 0 && !words.is_empty() {
            let last = words.len() - 1;
            words[last] &= (1u64 << remainder) - 1;
        }
        Self::from_words(words, n_samples)
    }

    pub fn fingerprint(words: &[u64], n_samples: usize) -> u64 {
        // SplitMix64-style avalanche mix. Replaces DefaultHasher (SipHash13,
        // ~1ns/byte) with a per-word ~3ns mix for SupportMask fingerprinting,
        // which is on the hot path of `partition_support` (constructed
        // very many times per fit).
        #[inline(always)]
        fn mix_u64(mut x: u64) -> u64 {
            x ^= x >> 30;
            x = x.wrapping_mul(0xbf58476d1ce4e5b9);
            x ^= x >> 27;
            x = x.wrapping_mul(0x94d049bb133111eb);
            x ^= x >> 31;
            x
        }
        let mut h = mix_u64(n_samples as u64);
        for &w in words {
            h ^= mix_u64(w);
            h = h.wrapping_mul(0x9e3779b97f4a7c15);
        }
        h
    }
}

/// Read-only borrow of a SupportMask-equivalent (words, n_samples,
/// fingerprint) backed by caller-owned storage (e.g. a scratch `Vec<u64>` on
/// the driver). Used in transient candidate-evaluation paths where we don't
/// want to pay the `Arc<[u64]>` allocation that owning a `SupportMask`
/// requires. Helpers that previously took `&SupportMask` can be given a
/// `SupportView` for the inline (non-recursive) consumption sites; recursive
/// `solve_node` calls still build an owned `SupportMask` from the scratch.
#[derive(Copy, Clone)]
pub struct SupportView<'a> {
    pub words: &'a [u64],
    pub n_samples: usize,
    pub fingerprint: u64,
}

impl<'a> SupportView<'a> {
    #[inline]
    pub fn new(words: &'a [u64], n_samples: usize, fingerprint: u64) -> Self {
        Self {
            words,
            n_samples,
            fingerprint,
        }
    }
}

impl PartialEq for SupportMask {
    fn eq(&self, other: &Self) -> bool {
        self.n_samples == other.n_samples && self.words.as_ref() == other.words.as_ref()
    }
}

impl Eq for SupportMask {}

impl Hash for SupportMask {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.n_samples.hash(state);
        self.fingerprint.hash(state);
    }
}

/// Canonical node identity: support + available-feature state + remaining depth.
///
/// The memo also discriminates on `mode_flags` (bit-packed boolean
/// toggles) and `teacher_alpha_bits` (the raw
/// f64 bit pattern of `MsdtPortConfig::teacher_alpha`, canonicalized to
/// 1.0-bits when the blend is effectively off so plain (no reference ensemble) runs keep
/// sharing memo entries with reference-ensemble-enabled runs at α=1.0).
#[derive(Clone, Debug)]
pub struct MemoKey {
    pub support: SupportMask,
    pub feature_mask: FeatureMask,
    pub depth: i32,
    pub mode_flags: u16,
    pub teacher_alpha_bits: u64,
}

pub type NodeKey = MemoKey;

/// Shared arena for `MemoKey` interning. Replaces
/// `SplitRecord.child_keys: Vec<MemoKey>` (168 B/record at k=2,
/// measured) with `Vec<u32>` indices into this arena (32 B/record).
///
/// **Concurrency**: lock-free reads via `boxcar::Vec` (stable indices,
/// `get(idx)` is a bucket-table lookup with no locks); concurrent writes
/// arbitrated by a `DashMap<u64, Vec<u32>>` dedup index. Lives behind
/// `Arc<MemoKeyArena>` on `SearchState` so solver workers share a single
/// arena, which avoids the index aliasing that per-worker arenas would
/// cause.
///
/// Hash collisions on the 64-bit `MemoKey` hash are resolved by checking
/// `PartialEq` against stored keys before reusing an index. A genuine
/// 64-bit collision between distinct MemoKeys gets distinct arena slots.
pub struct MemoKeyArena {
    storage: boxcar::Vec<MemoKey>,
    by_key_hash: DashMap<u64, Vec<u32>>,
    pub total_pushed: std::sync::atomic::AtomicU64,
    pub dedup_hits: std::sync::atomic::AtomicU64,
}

impl MemoKeyArena {
    pub fn new() -> Self {
        Self {
            storage: boxcar::Vec::new(),
            by_key_hash: DashMap::new(),
            total_pushed: std::sync::atomic::AtomicU64::new(0),
            dedup_hits: std::sync::atomic::AtomicU64::new(0),
        }
    }

    /// Insert a MemoKey; return its arena index. Hash-cons dedup with
    /// PartialEq fallback. Multi-thread safe: concurrent inserts of the
    /// same MemoKey may both push (rare race window between the hash-map
    /// check and the boxcar push); the resulting duplicate slots have
    /// equivalent contents so reads via either index return identical
    /// data. The dedup map then stores both indices under the same hash
    /// bucket — subsequent inserts dedup against the first to commit.
    pub fn intern(&self, key: MemoKey) -> u32 {
        use std::sync::atomic::Ordering;
        self.total_pushed.fetch_add(1, Ordering::Relaxed);
        let h = Self::hash_memo_key(&key);
        if let Some(candidates) = self.by_key_hash.get(&h) {
            for &idx in candidates.iter() {
                if let Some(stored) = self.storage.get(idx as usize) {
                    if *stored == key {
                        self.dedup_hits.fetch_add(1, Ordering::Relaxed);
                        return idx;
                    }
                }
            }
        }
        let idx = self.storage.push(key) as u32;
        self.by_key_hash.entry(h).or_insert_with(Vec::new).push(idx);
        idx
    }

    /// Deref an index. Panics on out-of-bounds (which would be an arena
    /// bug — every index in `SplitRecord.child_keys` was produced by
    /// `intern()` on this same arena).
    #[inline]
    pub fn get(&self, idx: u32) -> &MemoKey {
        self.storage
            .get(idx as usize)
            .expect("MemoKeyArena.get: out-of-bounds index")
    }

    pub fn len(&self) -> usize {
        self.storage.count()
    }

    /// Canonical index of the key at `idx`: the first committed slot
    /// holding an equal key (an `intern` race can give one key two slots).
    pub fn canonical(&self, idx: u32) -> u32 {
        let key = self.get(idx);
        let h = Self::hash_memo_key(key);
        self.by_key_hash
            .get(&h)
            .and_then(|c| {
                c.iter()
                    .copied()
                    .find(|&i| self.storage.get(i as usize).map_or(false, |s| s == key))
            })
            .unwrap_or(idx)
    }

    pub fn approximate_bytes(&self) -> u64 {
        let per_entry = std::mem::size_of::<MemoKey>() as u64;
        // boxcar internally allocates buckets; count() is a close
        // upper bound proxy. FeatureMask.words / SupportMask.words.heap
        // not counted (Arc-shared with live memo entries — counting them
        // here would double-attribute).
        (self.storage.count() as u64) * per_entry
    }

    pub fn pushed(&self) -> u64 {
        self.total_pushed.load(std::sync::atomic::Ordering::Relaxed)
    }

    pub fn hits(&self) -> u64 {
        self.dedup_hits.load(std::sync::atomic::Ordering::Relaxed)
    }

    fn hash_memo_key(key: &MemoKey) -> u64 {
        let mut h = DefaultHasher::new();
        key.hash(&mut h);
        h.finish()
    }
}

impl Default for MemoKeyArena {
    fn default() -> Self {
        Self::new()
    }
}

impl MemoKey {
    pub fn new(
        support: SupportMask,
        feature_mask: FeatureMask,
        depth: i32,
        mode_flags: u16,
        teacher_alpha_bits: u64,
    ) -> Self {
        MemoKey {
            support,
            feature_mask,
            depth,
            mode_flags,
            teacher_alpha_bits,
        }
    }
}

impl PartialEq for MemoKey {
    fn eq(&self, other: &Self) -> bool {
        self.depth == other.depth
            && self.mode_flags == other.mode_flags
            && self.teacher_alpha_bits == other.teacher_alpha_bits
            && self.feature_mask == other.feature_mask
            && self.support == other.support
    }
}

impl Eq for MemoKey {}

impl Hash for MemoKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.depth.hash(state);
        self.mode_flags.hash(state);
        self.teacher_alpha_bits.hash(state);
        self.feature_mask.hash(state);
        self.support.n_samples.hash(state);
        self.support.fingerprint.hash(state);
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Hash)]
pub struct FeatureMask {
    pub words: Vec<u64>,
}

impl FeatureMask {
    fn words_for(n_features: usize) -> usize {
        n_features.div_ceil(64)
    }

    pub fn all(n_features: usize) -> Self {
        let words_len = Self::words_for(n_features);
        let mut words = vec![!0u64; words_len];
        let remainder = n_features % 64;
        if remainder > 0 && !words.is_empty() {
            let last = words.len() - 1;
            words[last] &= (1u64 << remainder) - 1;
        }
        FeatureMask { words }
    }

    pub fn from_indices(n_features: usize, indices: &[usize]) -> Self {
        let mut words = vec![0u64; Self::words_for(n_features)];
        for &feature_idx in indices {
            let word = feature_idx / 64;
            let bit = feature_idx % 64;
            if word < words.len() {
                words[word] |= 1u64 << bit;
            }
        }
        FeatureMask { words }
    }

    #[inline]
    pub fn contains(&self, feature_idx: usize) -> bool {
        let word = feature_idx / 64;
        let bit = feature_idx % 64;
        self.words
            .get(word)
            .map(|w| (w & (1u64 << bit)) != 0)
            .unwrap_or(false)
    }

    pub fn with_added(&self, feature_idx: usize) -> Self {
        let mut next = self.clone();
        let word = feature_idx / 64;
        let bit = feature_idx % 64;
        if word >= next.words.len() {
            next.words.resize(word + 1, 0);
        }
        next.words[word] |= 1u64 << bit;
        next
    }

    pub fn without(&self, feature_idx: usize) -> Self {
        let mut next = self.clone();
        let word = feature_idx / 64;
        let bit = feature_idx % 64;
        if let Some(existing) = next.words.get_mut(word) {
            *existing &= !(1u64 << bit);
        }
        next
    }

    pub fn without_indices(&self, feature_indices: &[usize]) -> Self {
        let mut next = self.clone();
        for &feature_idx in feature_indices {
            let word = feature_idx / 64;
            let bit = feature_idx % 64;
            if let Some(existing) = next.words.get_mut(word) {
                *existing &= !(1u64 << bit);
            }
        }
        next
    }

    pub fn count_set(&self) -> usize {
        self.words.iter().map(|w| w.count_ones() as usize).sum()
    }

    pub fn is_full(&self, n_features: usize) -> bool {
        self.count_set() == n_features
    }

    pub fn is_empty(&self) -> bool {
        self.words.iter().all(|w| *w == 0)
    }
}

#[cfg(test)]
mod tests {
    use super::FeatureMask;
    use std::collections::HashSet;

    #[test]
    fn feature_mask_matches_set_semantics() {
        let n_features = 130;
        let seed = [0usize, 2, 63, 64, 129];
        let expected: HashSet<usize> = seed.into_iter().collect();
        let mask = FeatureMask::from_indices(n_features, &seed);

        for feature_idx in 0..n_features {
            assert_eq!(mask.contains(feature_idx), expected.contains(&feature_idx));
        }
        assert_eq!(mask.count_set(), expected.len());
        assert!(!mask.is_full(n_features));

        let grown = mask.with_added(5).with_added(127);
        assert!(grown.contains(5));
        assert!(grown.contains(127));
        assert_eq!(grown.count_set(), expected.len() + 2);

        let full = FeatureMask::from_indices(5, &[0, 1, 2, 3, 4]);
        assert!(full.is_full(5));
    }
}

// ---------------------------------------------------------------------------
// Sparse global bound table (follows the TreeFARMS bound list)
// ---------------------------------------------------------------------------
//
// Per-split search state lives in a global sparse table rather than on
// each SubProblem:
// - the materialized splits of a node are the bound_table entries with its
//   MemoKey and the MATERIALIZED flag;
// - each entry carries its contextual lower bound and a contextually-dead
//   flag bit;
// - Rashomon admission is a flag bit on BoundEntry;
// - the best split of a node is `best_split_table: DashMap<MemoKey, u64>`,
//   storing the split_id; the SplitRecord itself lives once in SplitRecordPool.
//
// Reference design: the bound list of TreeFARMS (Xin et al., 2022), a
//   concurrent vector of (index, lower bound, upper bound) tuples.
//   We extend the tuple to carry contextual state + pool_idx because our
//   solver maintains contextual bounds per (parent context, split) and dedups
//   SplitRecords through a shared pool.
//
// Atomicity: BoundEntry's `(lower_bound, upper_bound, pool_idx)` MUST update
// together via `DashMap::alter` whenever bounds refresh — pool dedup keys on
// `(split_id, split_upper_bits)`, so a refreshed `split_upper` produces a
// different pool_idx that must replace the old one in lockstep.

/// Bit flags packed into `BoundEntry.flags`.
///
/// `RASHOMON` — this (subproblem, feature) split is admitted into the
///   Rashomon set (i.e. cost ≤ rashomon_bound at insertion time).
/// `CONTEXTUALLY_DEAD` — this split has been proved unable to produce a
///   feasible completion within the BFS scope under which the parent is
///   currently being explored. Per-context flag, NOT a global lower-bound
///   certificate.
/// `MATERIALIZED` — this entry was admitted by `upsert_materialized_split`
///   (or its inline equivalents in `process_bound_update` /
///   `explore_node` finalization). Distinguishes the materialized set
///   (BnB UB propagation source) from rashomon-only entries.
pub const BOUND_FLAG_RASHOMON: u8 = 0b0000_0001;
pub const BOUND_FLAG_CONTEXTUALLY_DEAD: u8 = 0b0000_0010;
pub const BOUND_FLAG_MATERIALIZED: u8 = 0b0000_0100;

/// One row in the global sparse bound table, keyed by (MemoKey, feature_id).
///
/// Layout: 32 B with default repr (8-B alignment).
#[derive(Clone, Copy, Debug)]
pub struct BoundEntry {
    /// Feature index this bound row corresponds to. Redundant with the table
    /// key's u16 component but kept inline for cheap iteration when scanning
    /// all bounds at a node.
    pub feature_id: u16,
    /// Bit-flags: see `BOUND_FLAG_*`.
    pub flags: u8,
    pub _pad0: u8,
    /// Index into the per-solver `SplitRecordPool` for the SplitRecord that
    /// represents this split's full identity (thresholds, child_keys,
    /// group_map). Updated atomically with `lower_bound`/`upper_bound`
    /// whenever `split_upper` refreshes (pool dedup re-keys on the new
    /// `split_upper_bits`).
    pub pool_idx: u32,
    /// Split lower bound (in our cost units, NOT the f32-packed TreeFARMS
    /// representation). Refreshed via `process_bound_update`.
    pub lower_bound: f64,
    /// Split upper bound; same provenance.
    pub upper_bound: f64,
    /// Contextual lower bound learned from BFS scope/root pruning events.
    /// Sidecar of the SplitRecord; valid only for the parent context it was
    /// learned in (not a global lower bound).
    pub contextual_lowerbound: f64,
}

impl BoundEntry {
    pub fn new(feature_id: u16, pool_idx: u32, lb: f64, ub: f64) -> Self {
        Self {
            feature_id,
            flags: 0,
            _pad0: 0,
            pool_idx,
            lower_bound: lb,
            upper_bound: ub,
            contextual_lowerbound: f64::NEG_INFINITY,
        }
    }

    #[inline]
    pub fn is_contextually_dead(&self) -> bool {
        self.flags & BOUND_FLAG_CONTEXTUALLY_DEAD != 0
    }

    #[inline]
    pub fn set_rashomon(&mut self, on: bool) {
        if on {
            self.flags |= BOUND_FLAG_RASHOMON;
        } else {
            self.flags &= !BOUND_FLAG_RASHOMON;
        }
    }

    #[inline]
    pub fn set_contextually_dead(&mut self, on: bool) {
        if on {
            self.flags |= BOUND_FLAG_CONTEXTUALLY_DEAD;
        } else {
            self.flags &= !BOUND_FLAG_CONTEXTUALLY_DEAD;
        }
    }

    #[inline]
    pub fn is_materialized(&self) -> bool {
        self.flags & BOUND_FLAG_MATERIALIZED != 0
    }

    #[inline]
    pub fn set_materialized(&mut self, on: bool) {
        if on {
            self.flags |= BOUND_FLAG_MATERIALIZED;
        } else {
            self.flags &= !BOUND_FLAG_MATERIALIZED;
        }
    }
}

#[derive(Clone)]
pub struct SubProblem {
    pub support: SupportMask,
    /// Split path from root to this subproblem. Used to reconstruct bitmask
    /// on demand. Empty for the root node.
    pub split_path: Vec<SplitStep>,
    pub feature_mask: FeatureMask,
    pub n_samples: usize,
    pub class_counts: [i64; MAX_CLASSES],
    pub n_classes: u8,
    pub max_loss: f64,
    pub base_objective: f64,
    pub lowerbound: f64,
    pub upperbound: f64,
    /// True when upperbound comes from greedy boundary guidance rather than
    /// an exact witness representable by this shallow memo subtree.
    pub upperbound_is_heuristic: bool,
    // The best split of a node lives in `SearchState.best_split_table`
    // as a `(MemoKey -> u64 split_id)` mapping. Resolve the SplitRecord
    // by iterating bound_table[key] for the entry whose pool_idx points
    // at a SplitRecord with matching split_id. Helper API:
    // `BranchAndBoundSolver::resolve_best_split(key) -> Option<SplitRecord>`.
    //
    // Materialized splits live in `SearchState.bound_table[key]`,
    // identified by `BoundEntry.is_materialized()`. Iterate via
    // `search_state.bound_features_at(key).into_iter().filter(|e| e.is_materialized())`.
    // Reconstruct a SplitRecord with overlaid contextual sidecars by
    // cloning the pool entry at `entry.pool_idx` and applying
    // `entry.contextual_lowerbound` + `entry.is_contextually_dead()`.
    /// Monotone lower-bound floor from the most recent full Phase A screen.
    pub phase_a_lb_floor: f64,
    /// Number of Phase A survivors seen on the last BFS exploration pass.
    pub last_survivor_count: usize,
    /// True when the last BFS pass materialized every surviving candidate.
    pub all_survivors_materialized: bool,
    pub is_terminal: bool,
    pub is_solved: bool,
    /// branch-bound-only contextual floor: the widest recursive scope for which this
    /// node has been proved unable to produce a feasible completion. This is
    /// not a global lower bound and must not be used for certification.
    pub branch_bound_scope_floor: f64,
    /// Widest BFS requested_scope seen for this node. Scheduling hint only:
    /// never treat this as part of the node's semantic state.
    pub scope: f64,
    pub explored: bool,
    pub depth: i32,
    /// True when a BoundUpdate work item is already in the queue for this node.
    pub bound_update_pending: bool,
    /// Rashomon mode: ALL splits with materialized_ub ≤ rashomon_bound.
    /// Empty when rashomon_mode=false. The best split is maintained
    /// separately for BnB bound propagation (tightest UB).
    ///
    /// Stored as `Vec<u32>` indices into the per-solver `SplitRecordPool`.
    ///
    /// May hold repeats (the same pool_idx re-pushed, or one split identity
    /// under several pool indices with different `split_upper`). They are
    /// not tree-count semantics: `build_factored_recursive` keeps one
    /// FactoredSplit per split identity.
    pub rashomon_indices: Vec<u32>,
}

/// Pool of `SplitRecord`s shared across a single Phase-2 BFS solve.
///
/// One pool lives per `BranchAndBoundSolver` (i.e., per root in
/// `solve_parallel_root`'s rayon dispatch). `SubProblem.rashomon_indices`
/// and the bound table reference into this pool via `u32` indices, so each
/// SplitRecord is stored once.
///
/// The pool is single-mutator per BFS (Phase-2 BFS is single-threaded
/// per root; rayon parallelism is across roots), so no locks are needed.
/// `&self` reads via `get(idx)` are safe given the borrow checker
/// guarantees the pool isn't mutated concurrently.
pub struct SplitRecordPool {
    storage: Vec<SplitRecord>,
    /// Hash-cons dedup keyed by `(split_id, split_upper_bits)`. Splits
    /// with the same identity (feature_index, thresholds, child_keys,
    /// group_map per `BranchAndBoundSolver::split_record_id`) AND the
    /// same `split_upper` share a pool entry — identical re-pushes from
    /// the same parent context (e.g., `recompute_materialized_bfs_state`
    /// re-fires) are O(1) idempotent. Different `split_upper` values
    /// (e.g., the same identity at two different parents with different
    /// path costs) get distinct pool entries because `split_upper` is
    /// per-parent-context and aliasing it would corrupt BFS pruning.
    /// Per-context mutable state (`contextual_lowerbound`,
    /// `contextually_dead`) lives on the bound-table entries and is NOT
    /// shared via the pool.
    by_key: HashMap<(u64, u64), u32>,
    /// Telemetry: total `push` calls (including dedup hits).
    pub total_pushed: u64,
    /// Telemetry: `push` calls that returned an existing index.
    pub dedup_hits: u64,
}

impl SplitRecordPool {
    pub fn new() -> Self {
        Self {
            storage: Vec::new(),
            by_key: HashMap::new(),
            total_pushed: 0,
            dedup_hits: 0,
        }
    }

    pub fn with_capacity(cap: usize) -> Self {
        Self {
            storage: Vec::with_capacity(cap),
            by_key: HashMap::with_capacity(cap),
            total_pushed: 0,
            dedup_hits: 0,
        }
    }

    /// Insert a SplitRecord; return its `u32` index. Hash-cons dedup by
    /// `(split_id, split_upper_bits)` to preserve per-parent-context
    /// `split_upper` distinctness.
    #[inline]
    pub fn push(&mut self, sr: SplitRecord) -> u32 {
        self.total_pushed += 1;
        let key = (sr.split_id, sr.split_upper.to_bits());
        if let Some(&idx) = self.by_key.get(&key) {
            self.dedup_hits += 1;
            return idx;
        }
        let idx = self.storage.len() as u32;
        self.by_key.insert(key, idx);
        self.storage.push(sr);
        idx
    }

    /// Read a record by index. Panics if `idx >= len()`.
    #[inline]
    pub fn get(&self, idx: u32) -> &SplitRecord {
        &self.storage[idx as usize]
    }

    /// Mutable read for in-place updates (e.g., `contextual_lowerbound`).
    /// Used by parent-bound-propagation paths.
    #[inline]
    pub fn get_mut(&mut self, idx: u32) -> &mut SplitRecord {
        &mut self.storage[idx as usize]
    }

    pub fn len(&self) -> usize {
        self.storage.len()
    }

    pub fn is_empty(&self) -> bool {
        self.storage.is_empty()
    }

    /// Approximate bytes stored (the `pool_total_bytes` telemetry).
    pub fn approximate_bytes(&self) -> u64 {
        let per_record = std::mem::size_of::<SplitRecord>() as u64;
        let cap_bytes = (self.storage.capacity() as u64) * per_record;
        // Add per-record child_keys / thresholds / group_map heap footprint.
        let mut heap: u64 = 0;
        for sr in &self.storage {
            heap += (sr.child_keys.len() as u64) * std::mem::size_of::<MemoKey>() as u64;
            heap += (sr.thresholds.len() as u64) * 8;
            if let Some(gm) = &sr.group_map {
                heap += (gm.len() as u64) * 8;
            }
        }
        cap_bytes + heap
    }
}

impl Default for SplitRecordPool {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Clone)]
pub struct SplitRecord {
    pub split_id: u64,
    pub feature_index: usize,
    pub thresholds: Arc<[f64]>,
    /// Indices into `SearchState.memo_key_arena`. Replaces
    /// the prior `Vec<MemoKey>` (168 B/record at k=2 measured) with
    /// `Vec<u32>` (32 B/record). Resolve via
    /// `solver.search.memo_key_arena.get(idx)` to obtain `&MemoKey`.
    pub child_keys: Vec<u32>,
    pub split_upper: f64,
    /// Monotone split-local floor learned from BFS scope/root pruning
    /// events in specific parent contexts.
    pub contextual_lowerbound: f64,
    /// True when a contextual closure event proves this split can no longer
    /// beat the parent budget under which it was spawned.
    pub contextually_dead: bool,
    /// Shape mode: maps bin index -> child index.
    pub group_map: Option<Vec<usize>>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct BfsExploreOrigin {
    pub parent_key: MemoKey,
    pub split_id: u64,
    pub sibling_lowerbound: f64,
}

/// Split candidate from candidate generation
#[derive(Clone, Debug)]
pub struct SplitCandidate {
    /// Weighted Gini plus objective penalties and, for reference-order
    /// candidates, the original-axis boundary penalty.
    pub rank_cost: f64,
    /// Objective cost when every child is a leaf; excludes boundary penalties.
    pub split_ub: f64,
    pub k: usize,
    pub feature_index: usize,
    pub thresholds: Arc<[f64]>,
    /// Shape mode: maps bin index -> group index (0..k-1).
    /// When Some, thresholds are ALL L-1 bin boundaries (not K-1 split points).
    pub group_map: Option<Vec<usize>>,
    /// Per-child sample counts from DP collapse (len = k).
    pub child_n: Vec<i64>,
    /// Per-child class distributions from DP collapse (len = k, each inner Vec len = n_classes).
    pub child_counts: Vec<Vec<i64>>,
    /// Candidate-family tag used by anchor preservation in candidate
    /// selection. 0 = primary criterion/geometry, 1 = misclass companion,
    /// 2/3 = alternate geometry used by ShapeSPLINTER's contiguous baseline.
    pub family: u8,
    /// Number of original-axis interval components in the partition.
    /// For axis-mode (`group_map = None`) always equals `k` (each branch
    /// is one contiguous interval). For shape-mode, counts the number of
    /// `[a_r, a_{r+1}]` adjacencies where the assigned branch differs,
    /// plus one. Used as a cost-tie-breaker in `compare_split_candidates`.
    pub components: u32,
}

/// Family tag constants for SplitCandidate.family.
///
/// Two-bit encoding: bit 0 = criterion (0=primary, 1=misclass),
/// bit 1 = mode (0=primary mode == config.shape_mode, 1=alt mode).
/// Anchor preservation in `select_top_k_with_anchors` rescues the best
/// candidate per distinct family value, so up to 4 families can each
/// reserve a seat in the top-√k prefix.
pub const CAND_FAMILY_PRIMARY: u8 = 0;
pub const CAND_FAMILY_MISCLASS: u8 = 1;
pub const CAND_FAMILY_ALT_MODE_PRIMARY: u8 = 2;
pub const CAND_FAMILY_ALT_MODE_MISCLASS: u8 = 3;

/// Priority queue entry for BFS. Carries either an Explore or BoundUpdate task.
pub struct WorkItem {
    pub neg_gap: f64,
    /// true = BoundUpdate (recompute parent bounds), false = Explore (expand node)
    pub is_bound_update: bool,
    pub counter: u64,
    pub key: MemoKey,
    pub depth: i32,
    /// BFS scope hint. Only meaningful for Explore items; NAN for BoundUpdate.
    pub requested_scope: f64,
    /// Optional parent/split context that produced this BFS explore scope.
    pub origin: Option<BfsExploreOrigin>,
}

impl PartialEq for WorkItem {
    fn eq(&self, other: &Self) -> bool {
        self.neg_gap == other.neg_gap
            && self.is_bound_update == other.is_bound_update
            && self.counter == other.counter
    }
}

impl Eq for WorkItem {}

impl PartialOrd for WorkItem {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for WorkItem {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // BinaryHeap is a max-heap. Largest neg_gap (= largest gap) pops first.
        // At equal gap, BoundUpdate (true) before Explore (false) — tighter
        // bounds enable better pruning of subsequent Explores.
        other
            .neg_gap
            .partial_cmp(&self.neg_gap)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(other.is_bound_update.cmp(&self.is_bound_update))
            .then(other.counter.cmp(&self.counter))
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum DpCriterion {
    Gini,
    Misclass,
    Depth2,
    Depth2Misclass,
    Rush,
    Twoing,
    Entropy,
    Hellinger,
    Learned,
    Wasserstein,
}

/// Signature for subproblems with the same (n_samples, class_counts).
/// Used for cross-subtree bound transfer via similar_lb/similar_ub tables.
#[derive(Hash, Eq, PartialEq, Clone, Copy)]
pub struct SupportSig {
    pub n_samples: usize,
    pub counts: [i64; MAX_CLASSES], // zero-padded, supports all MAX_CLASSES classes
    pub len: u8,
}

impl SupportSig {
    #[inline]
    pub fn new(n_samples: usize, class_counts: &[i64]) -> Self {
        let mut counts = [0i64; MAX_CLASSES];
        let len = class_counts.len().min(MAX_CLASSES);
        counts[..len].copy_from_slice(&class_counts[..len]);
        SupportSig {
            n_samples,
            counts,
            len: len as u8,
        }
    }

    #[inline]
    pub fn as_slice(&self) -> &[i64] {
        &self.counts[..self.len as usize]
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FeatureType {
    Num,
    Cat,
}

#[derive(Clone)]
pub struct SolverConfig {
    pub n_classes: usize,
    pub n_features: usize,
    pub reg: f64,
    pub max_depth: i32,
    pub max_k: usize,
    pub min_samples_leaf: usize,
    pub feature_exclusion: bool,
    pub time_limit: Option<f64>,
    pub initial_used_features: FeatureMask,
    pub feature_names: Vec<String>,
    /// Per-feature type: Num (continuous) or Cat (categorical).
    /// Length must equal n_features. Default: all Num.
    pub feature_types: Vec<FeatureType>,
    pub warm_start_ub: Option<f64>,
    /// Override n_total for loss normalization (used by parallel root solver
    /// so child solvers normalize by the global dataset size, not their subset size).
    pub n_total_override: Option<usize>,
    pub thresholds_per_feature: Option<HashMap<usize, Vec<f64>>>,
    /// Optional override threshold set used only by `closure_fast_cand_gen`
    /// (the closure-internal fused candidate kernel). When `Some`, closure
    /// nodes bin/scan against THIS map for k>2 candidates instead of the
    /// main `thresholds_per_feature`. Intended for passing a pruned
    /// LightGBM reference-ensemble threshold map so the closure heuristic runs cheaper
    /// without changing exact-search bounds or B&B candidate generation.
    /// `None` = closure falls back to `thresholds_per_feature`.
    pub closure_thresholds_per_feature: Option<HashMap<usize, Vec<f64>>>,
    /// DP collapse criterion: Gini, Misclass, or Depth2 (sub-split).
    pub dp_criterion: DpCriterion,
    /// Multiplier for the beam schedule in DP collapse.
    /// Base schedule is 1 << (max_k - k); this scales it.
    pub beam_width: usize,
    /// Controls how regularization scales with split arity.
    /// Total reg for a K-way split = reg * (1 + (K-1) * reg_alpha).
    /// reg_alpha=1.0 (default): standard per-leaf (K × reg).
    /// reg_alpha=0.0: per-split (1 × reg).
    /// Values in between give a fractional discount for multiway splits.
    pub reg_alpha: f64,
    /// If true, run dp_threshold_scan on each child partition (Phase 2 only)
    /// to compute locally-relevant thresholds for that child's subtree.
    pub local_thresh_scan: bool,
    /// Beam width for the scanning DP (max_k=2 scan per child feature).
    pub scan_beam: usize,
    /// If true, redistribute the beam budget per feature based on marginal
    /// impurity gains across split arities (k=2..max_k). Features where
    /// k=4 gives much more gain than k=2 get wider beams for higher k.
    pub adaptive_beam: bool,
    /// If true, override the per-feature beam_width to a value large enough
    /// that no candidate is dropped by per-feature DP truncation. This is
    /// the "Rashomon-completeness" knob — when on, every distinct (feature,
    /// threshold, arity) the per-feature DP would consider gets emitted.
    /// Independent of `fill_mode`/search strategy: orthogonal lever.
    /// Set in Phase 2 by the wrapper when the user requests `bypass_beam=True`.
    /// Default: false (the beam applies).
    pub bypass_beam: bool,
    /// Transition point compression delta. Adjacent bins with class proportion
    /// within delta of the running group's proportion are merged before DP.
    /// Set to 0.0 to disable compression. Default: 0.08.
    pub transition_delta: f64,
    /// Number of top candidates whose children are materialized (created as
    /// full SubProblems in the memo table and enqueued for exploration).
    /// Higher values give better LB tightening at the cost of more memory.
    /// 1 = most aggressive (only the best candidate). Default: 1.
    pub materialize_top_n: usize,
    /// If true, allow cross-subproblem lower/upper bound transfer using the
    /// similar_lb/similar_ub tables. This is exploratory-only until the
    /// transfer rule is fully proved sound for the active solver state.
    pub allow_heuristic_bound_transfer: bool,
    /// Shape mode: add reference-ensemble-induced bin orders before interval DP collapse,
    /// producing non-contiguous bin-to-group mappings in the original feature
    /// order. Binary uses the reference-ensemble class-probability order; multiclass uses
    /// pairwise reference-ensemble probability margins for each present class pair.
    pub shape_mode: bool,
    /// Binary reference-ensemble-order smoothing: Bayesian shrinkage of the per-bin
    /// shape-sort key toward parent-node class proportion. 0.0 = raw reference-ensemble
    /// probability mass / bin count. >0 = pseudo-count smoothing of the form
    /// `(c0 + α·parent_p0) / (n + α)`. Reduces ranking noise on small-
    /// support bins. Recommended: 1.0 (light), 2.0 (moderate), 5.0 (heavy).
    pub shape_sort_smoothing: f64,
    /// Path length regularization: penalty per internal node, weighted by n/N.
    /// Each split pays lambda * n_node / n_total. 0.0 = disabled (default).
    pub path_penalty: f64,
    /// Additional greedy depth below the exact BnB frontier. Legacy boundary
    /// guidance knob; production SPLINTER uses `closure_depth`.
    pub greedy_extra_depth: i32,
    // ── Rashomon mode ────────────────────────────────────────────────
    /// If true, explore ALL candidates within rashomon_bound (not just best).
    pub rashomon_mode: bool,
    /// Absolute objective bound: all trees with obj ≤ this are in the Rashomon set.
    /// Set by Phase 2 after Phase 1 finds opt.
    pub rashomon_bound: f64,
    /// Depth still to be completed below this solve's depth-0 nodes by
    /// Step B (the RESPLINTER algorithm): `max_depth_total −
    /// solve depth` for the Phase-2 prefix solve, 0 otherwise. When > 0
    /// and in rashomon mode, depth-0 placeholders stay terminal but keep
    /// their admissible completion lower bound instead of the leaf cost.
    pub step_b_remaining: i32,
    // ── Weighted samples (boosting) ─────────────────────────────────
    /// Sum of sample weights for loss normalization. When no weights are
    /// provided, this equals n_data as f64. When weights are provided,
    /// this equals Σ weights[i] × n_data (integer-scaled).
    pub weight_total: f64,
    /// Upper-bound pruning margin: skip candidates whose local ordering key is
    /// more than this much worse than the best. f64::INFINITY disables it.
    /// Larger finite values explore more candidates at the cost of speed.
    pub ub_margin: f64,
    /// Number of worker threads reserved for older internal solve paths.
    /// Production SPLINTER uses the branch-bound engine.
    pub n_workers: usize,
    /// If true, run candidate generation sequentially (no inner rayon par_iter).
    /// Set automatically when n_workers > 1 to avoid oversubscription.
    pub sequential_candidate_gen: bool,
    /// Use cheap CART-style greedy (binary, no DP) instead of full multiway
    /// greedy for boundary guidance. Much faster but binary-only splits.
    pub cheap_boundary_greedy: bool,
    /// If Some(d), nodes at depth <= d are solved by the configured closure
    /// policy and materialized into the memo with proper SplitRecords.
    /// Replaces heuristic greedy boundary guidance for these nodes.
    /// None = disabled (default).
    pub closure_depth: Option<i32>,
    /// Number of top closure candidates to evaluate recursively. 1 keeps a
    /// single local candidate; higher values keep more closure alternatives.
    /// Default 1.
    pub closure_beam: usize,
    /// Parallelize the per-candidate sibling expansion loop inside
    /// closure_build. Each candidate in `to_expand` runs in its own
    /// scoped thread with a fresh WorkerState; all share the same
    /// Arc<SearchState>/memo. Ignored when n_workers > 1 (avoids
    /// oversubscription with concurrent BFS workers).
    pub closure_parallel: bool,
    /// User-visible flag: when true AND `local_thresh_scan=false`,
    /// solve_parallel_root allocates a shared closure memo and hands an
    /// Arc to every child solver so identical (support, feature_mask,
    /// depth) subproblems are solved once across the root workers.
    pub closure_share_memo: bool,
    /// Actual shared closure memo handle, set by solve_parallel_root when
    /// closure_share_memo is safe to use. None everywhere else.
    /// Keyed by MemoKey; values are fully-solved SubProblems from
    /// closure_build publication points.
    pub closure_shared_memo: Option<Arc<DashMap<MemoKey, SubProblem>>>,
    /// Lazy depth-1 exploration: only recurse into depth-1 children if the
    /// parent candidate's UB is within this delta of the best candidate's UB.
    /// 0.0 = only rank-0 children. f64::INFINITY = explore all (current behavior).
    pub lazy_explore_delta: f64,
    /// Override dp_criterion inside closure_build's fused kernel.
    /// None = use the main dp_criterion. Allows e.g. Gini closure with
    /// Rush main search.
    pub closure_criterion: Option<DpCriterion>,
    /// Adaptive `split_ub`-gap beam for closure's `to_expand`.
    /// In SinglePath/Full/Cart modes, after best-per-k selection, drop
    /// candidates whose `split_ub` exceeds the rank-0 candidate's
    /// `split_ub` by more than this gap. f64::INFINITY = keep all
    /// (current behavior). 0.0 = strict single-path closure. Recommended
    /// values: ~`reg` to ~`5*reg` depending on the dataset's tolerance.
    /// Quality risk: rank-0 by inline UB isn't always rank-0 by actual
    /// recursive cost; tighten only after measuring obj impact.
    pub closure_split_ub_gap: f64,
    /// Top-K feature filter for `closure_fast_cand_gen`.
    /// When `Some(k)`, closure scans only the top-k features by
    /// LGBM-derived importance instead of all `n_features`. None = all.
    /// Lossy: closure may pick a worse feature when the optimal one is
    /// outside the top-k. Recommended: k = max(F/2, 8) as a starting
    /// point. Tunable.
    pub closure_feature_top_k: Option<usize>,
    /// Pre-ranked feature indices by LGBM importance, length
    /// `n_features`. Set by Python wrapper from `lgbm_thresh` (count of
    /// thresholds per feature is a first-cut proxy for gain). When
    /// `closure_feature_top_k` is `Some(k)`, the closure cand-gen iterates
    /// the first `k` indices of this Vec. None = use natural order 0..F.
    pub closure_feature_ranking: Option<Arc<Vec<usize>>>,
    /// Small-support gate. When closure entry's `n_support <
    /// threshold`, force `max_k=2` for that call (skip Rush-DP for k>2).
    /// None or 0 = always run full multiway. Recommended: `8 *
    /// min_samples_leaf` as a starting point. Lossy at small supports
    /// where k≥3 splits could marginally beat k=2.
    pub closure_min_n_for_multiway: Option<usize>,
    /// Depth threshold for the per-feature `irr_j + min_split_overhead`
    /// filter inside `closure_fast_cand_gen`. Mirrors the upstream
    /// `unsafe_filter_depth_threshold`.
    ///
    /// - `0` (default): filter applies ONLY at `closure_depth_remaining == 1`
    ///   Strictly admissible at depth==1 because children are
    ///   forced to be leaves.
    /// - `1` (or any equal value to closure_depth_remaining at the call):
    ///   same as `0` for the single-depth case.
    /// - `>= 2`: filter ALSO applies at deeper closure levels where it is
    ///   NON-ADMISSIBLE (a deeper subtree using a different feature can drop
    ///   per-bin misclass below `min(c0, c1)`). Lossy heuristic gated on
    ///   `admissibility_strict=False`.
    /// - When set to `i32::MAX` or `closure_depth`: full lossy filter at
    ///   all closure depths — the historical "unsafe filter" behavior the
    ///   upstream solver had before the LB-admissibility bug fix.
    pub closure_unsafe_filter_depth_threshold: i32,
    /// Phase 0 greedy-CART pre-ranker depth (parallel-root only). When > 0,
    /// every root candidate is scored by recursive greedy expansion to this
    /// depth before Phase 1 deep screening. Top `2 * top_n` candidates by
    /// pre-rank score are forwarded to Phase 1; the rest are dropped.
    /// Lossless ordering heuristic: changes which candidates Phase 1 sees,
    /// not whether the final answer is correct (top_n is a heuristic anyway).
    /// 0 = disabled (skip Phase 0, screen all candidates in Phase 1).
    pub phase0_greedy_prerank_depth: i32,
    /// Plumb equivalent-points (per-support EC minority) LB into the inline
    /// bounds path used during candidate generation. Lossless: tightens the
    /// LB returned by `compute_bounds_inline` / `compute_bounds_from_counts`,
    /// strictly increases pruning, never changes correctness. Binary-only.
    /// Default true; set false to bypass when profiling.
    pub eqp_inline_bounds: bool,
    /// Shared threshold interner for deduplicating threshold vectors.
    pub threshold_interner: Arc<ThresholdInterner>,
    /// Candidate-generation knobs for atomization, top-k retention, and
    /// ShapeSPLINTER-specific local ordering.
    /// Grouped so existing construction sites can stay compact via
    /// `MsdtPortConfig::default()` when these features are off.
    pub msdt: MsdtPortConfig,
    /// Resource and speed settings of the Rashomon completion-lb search
    /// and Step B (`RashomonTuning`); the enumerated tree set does not
    /// depend on them, except `lb_interval` (a tighter-or-equal lb).
    pub rash_tuning: RashomonTuning,
    /// Per-sample reference-ensemble class probabilities, flat row-major
    /// `[n_samples * n_classes]`. Populated once at PyO3 entry by
    /// softmax-converting the LightGBM `raw_score` output captured by the
    /// Python reference-ensemble cache (binary logit `(n,)` is expanded to `(n, 2)`
    /// on the Python side). `None` when no reference ensemble is available.
    /// ShapeSPLINTER uses this to order atoms and optionally blend reference-ensemble
    /// soft-Gini into the local shape DP.
    pub teacher_prob: Option<Arc<[f64]>>,
    /// Binary-only convenience slice derived from `teacher_prob`:
    /// `teacher_prob_c0[i] = teacher_prob[i * 2 + 0]`. Stored alongside
    /// `teacher_prob` so the per-node binning loop can index it directly.
    /// `None` when `teacher_prob` is `None` or `n_classes != 2`.
    pub teacher_prob_c0: Option<Arc<[f64]>>,
    /// Optional reference-model hard predictions (one class label per
    /// sample). When provided, drives `DatasetStore::reference_misclass`
    /// for the SPLIT-style reference-LB tightener. **NOT strictly
    /// admissible** — see `DatasetStore::reference_misclass` for the
    /// admissibility caveat. Default `None` (feature off).
    pub reference_preds: Option<Arc<[u8]>>,
    /// Gate the
    /// per-feature unsafe `irr_j + min_split_overhead < ub_hint` filter
    /// on subproblem depth. At `depth_remaining == 1` (immediate parents
    /// of leaves), the filter IS provably admissible because all
    /// children must be leaves at that frontier. Setting to 1 enables
    /// it ONLY at depth==1 nodes (admissible). Setting to 2+ relaxes
    /// admissibility — children at depth-1 can still split further, so
    /// this is unsafe past 1. Default 0 (filter never fires; pure
    /// admissible behavior).
    pub unsafe_filter_depth_threshold: i32,
    /// Strict-admissibility gate. When true, the solver refuses to run
    /// with any heuristic (lossy) knob enabled. Validation happens at
    /// solver construction (lib.rs entry points) and produces a
    /// PyValueError listing every offending knob. Use this for
    /// proof-of-optimum claims and as a CI regression guard against
    /// silent loss from knob tightening. Default `false` (legacy
    /// behavior; heuristic knobs are silently honored).
    pub admissibility_strict: bool,
}

/// Merge predicate used by atom construction when `atom_mode` is on.
///
/// - `L1Tolerance`: merge adjacent bins iff the L1 distance of their
///   normalized class vectors is ≤ `atom_merge_tolerance`. Sample-size
///   agnostic; on real discretized data rarely fires unless the
///   tolerance is loose.
/// - `WilsonCI`: merge iff each class's Wilson confidence interval
///   (derived from `atom_wilson_z`) overlaps between the two bins —
///   sample-aware, auto-tightens with n.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AtomMergeRule {
    L1Tolerance,
    WilsonCI,
}

/// Candidate-generation knobs: bin atomization and top-k candidate prefix
/// with anchor preservation.
#[derive(Clone, Debug)]
pub struct MsdtPortConfig {
    /// Boundary-cost DP: when > 0, the shape DP
    /// charges a per-interval tax `μ_node × ½ × signed-edge-count` for
    /// every original-axis adjacency edge an interval crosses. Penalizes
    /// fragmentation in the proportion-permuted axis. A typical value is
    /// μ_node = 1/n. 0.0 = no tax (default).
    pub shape_boundary_tax: f64,
    /// Per-atom reference-ensemble prior in the binary shape sort. When
    /// true AND a reference-ensemble prediction is available, the shape DP's
    /// atom ordering uses the per-atom reference-ensemble class-0 mean as
    /// the smoothing prior with weight 1:
    /// `s_r = (N_+(a_r) + bar_p_r) / (n_r + 1)`. When false (default) the
    /// prior is the node-level
    /// `parent_p0` constant, controlled by `shape_sort_smoothing`.
    pub shape_sort_per_atom_teacher_prior: bool,
    /// Enable atomization: merge adjacent bins (by the `atom_merge_rule`
    /// predicate) before DP collapse. Bin-level information is preserved
    /// via a bin→atom projection so emitted thresholds still reference
    /// original bin boundaries.
    pub atom_mode: bool,
    /// Predicate for the atom merge decision. See `AtomMergeRule`.
    pub atom_merge_rule: AtomMergeRule,
    /// L1 tolerance between normalized class-count vectors for atom
    /// merging under `AtomMergeRule::L1Tolerance`. 0.0 = exact-match only.
    pub atom_merge_tolerance: f64,
    /// Wilson z-score used under `AtomMergeRule::WilsonCI`. Default
    /// 1.959964 = two-tailed 95% CI.
    pub atom_wilson_z: f64,
    /// Cap the number of candidates fed to branch-bound recursion at each node.
    /// Anchors (best per family) always preserved. Disabled in rashomon_mode
    /// regardless of this value.
    pub top_k: Option<usize>,
    /// Floor on the effective beam cap so small subproblems don't get
    /// strangled. Default 8 (matches BEAM_MAX in candidates.rs).
    pub top_k_min_beam: usize,
    /// When true AND `top_k = None`, the auto-√N rule uses
    /// √(candidate_count) instead of √(n_rows). Default `false` preserves
    /// historical behavior (cap scales with dataset size, ineffective at
    /// our scales). True gives a "principled" per-subproblem cap that
    /// scales with the actual emitted candidate set — a more honest
    /// near-greedy heuristic without the ad-hoc cap=8 floor.
    pub top_k_auto_sqrt_candidates: bool,
    /// Post-DP cleanup: after candidate generation, emit additional
    /// (k-1)-arity variants for each candidate where adjacent children
    /// have identical class distributions. Contiguous mode only; merges
    /// are exact (no tolerance). Each cleanup variant saves one leaf_reg
    /// of regularization cost without changing the partition quality.
    pub cleanup_mode: bool,
    /// Block compression after Wilson atomization.
    /// When > 0.0, runs a second merge pass over the atomized sequence
    /// merging adjacent atoms that are both "empirically pure" (the
    /// dominant class's share ≥ this threshold) AND share the same
    /// dominant class. 0.0 (default) disables.
    /// Typical values: 0.95, or 1.0 (strict — merge only atoms with 100%
    /// one class).
    pub atom_block_compression_purity: f64,
    /// Blend weight for reference-ensemble soft Gini in the shape-DP edge cost.
    /// 1.0 = pure hard Gini (current behavior). 0.0 = pure reference-ensemble soft Gini.
    /// The blend takes effect only when:
    ///   * `SolverConfig::shape_mode == true`
    ///   * binary classification
    ///   * `DpCriterion::Gini` (other criteria pass through unchanged)
    ///   * `SolverConfig::teacher_prob.is_some()`
    ///   * `SolverConfig::sample_weights.is_none()` (weighted blending is
    ///     not supported)
    /// Values outside [0,1] are clamped at PyO3 entry. When the blend is
    /// effectively off (α = 1.0, or any α with no reference ensemble), the memo
    /// key canonicalizes α to 1.0 so cache entries remain shareable with
    /// plain runs (no reference ensemble).
    pub teacher_alpha: f64,
}

/// Rashomon-mode resource settings, set from `solve_rashomon` kwargs.
#[derive(Clone, Debug)]
pub struct RashomonTuning {
    /// Threads of the intra-cell parallel completion-lb prefetch
    /// (`solver::rash_par`): before each Step A node's candidate
    /// loop, its child completion-lb searches run on a pool of this many
    /// threads against a frozen snapshot of the lb caches and are merged
    /// back; the loop then only hits the memo, so the output is the same
    /// at any count. 1 = off.
    pub lb_threads: usize,
    /// Each Step-B level's placeholder groups are split into this many
    /// contiguous batches, solved in parallel by one seeded solver each
    /// (`rashomon_dag::stepb_level_batched`); same tree set, the DAG bytes
    /// may differ from the unbatched level. 0 = unbatched.
    pub stepb_batches: usize,
    /// `lb_threads` of each batch solver at a batched level of `n`
    /// batches: None = max(1, lb_threads / n), Some(t) = t.
    pub stepb_batch_lb_threads: Option<usize>,
    /// The completion lb at remaining depth 2 comes from an exact interval
    /// DP over the support's store bins (`solver::interval_lb`) instead of
    /// the lb Select pool; its value is <= the pool path's (admissible),
    /// so the tree set can only grow.
    pub lb_interval: bool,
    /// With `lb_interval`, only roots with max_k >= this use the interval
    /// DP (smaller max_k keep the pool path). 2 = every max_k.
    pub lb_interval_mink: usize,
    /// Remaining-depth-1 entries the completion-lb memo holds before they
    /// are dropped (a pure cache: same values).
    pub lb_memo_max_entries: usize,
    /// Total entries (both generations) of the structural depth-1 memo
    /// (`RashD1Struct`; a pure cache).
    pub lb_d1s_max_entries: usize,
    /// Entries (both generations) of the interval DP's φ cache
    /// (`solver::interval_lb::RashIvCache`); 0 = off.
    pub lb_interval_cache_max: usize,
}

impl Default for RashomonTuning {
    fn default() -> Self {
        RashomonTuning {
            lb_threads: 1,
            stepb_batches: 0,
            stepb_batch_lb_threads: None,
            lb_interval: false,
            lb_interval_mink: 3,
            lb_memo_max_entries: 3_000_000,
            lb_d1s_max_entries: 16_000_000,
            lb_interval_cache_max: 100_000,
        }
    }
}

impl RashomonTuning {
    /// `lb_threads` of each batch solver at a level of `n_batches` batches.
    pub fn batch_lb_threads(&self, n_batches: usize) -> usize {
        match self.stepb_batch_lb_threads {
            None => (self.lb_threads / n_batches.max(1)).max(1),
            Some(t) => t.max(1),
        }
    }
}

impl Default for MsdtPortConfig {
    fn default() -> Self {
        MsdtPortConfig {
            shape_boundary_tax: 0.0,
            shape_sort_per_atom_teacher_prior: false,
            atom_mode: false,
            atom_merge_rule: AtomMergeRule::L1Tolerance,
            atom_merge_tolerance: 0.0,
            atom_wilson_z: 1.959_964_f64,
            top_k: None,
            top_k_min_beam: 8,
            top_k_auto_sqrt_candidates: false,
            cleanup_mode: false,
            atom_block_compression_purity: 0.0,
            teacher_alpha: 1.0,
        }
    }
}

impl MsdtPortConfig {
    /// True iff any of these knobs would affect DP output or candidate
    /// count. Used to contribute atom/dual bits to `MemoKey::mode_flags`
    /// so cached subproblems are not shared across modes.
    pub fn is_active(&self) -> bool {
        self.atom_mode
            || self.cleanup_mode
            || self.top_k.is_some()
            || self.teacher_alpha < 1.0 - 1e-12
            || self.shape_boundary_tax > 0.0   // separate cache namespace
            || self.shape_sort_per_atom_teacher_prior // changes shape DP atom order
    }
}

impl SolverConfig {
    /// Validate strict-admissibility constraints. Returns `Ok(())` when
    /// `admissibility_strict == false` or when no heuristic knob is
    /// engaged. On violation, returns `Err(msg)` listing every offending
    /// knob and its current value. Caller (lib.rs entry) converts this
    /// into a `PyValueError`.
    ///
    /// Knobs gated:
    /// - `ub_margin` must be infinite (any finite margin is heuristic)
    /// - `unsafe_filter_depth_threshold` must be 0
    /// - `lazy_explore_delta` must be infinite (off)
    /// - `closure_depth` must be None
    /// - `path_penalty` must be 0.0
    /// - `greedy_extra_depth` must be 0
    /// - `cheap_boundary_greedy` must be false
    /// - `allow_heuristic_bound_transfer` must be false
    /// - `local_thresh_scan` must be false
    ///
    /// `beam_width` is intentionally NOT gated here — the beam clip is
    /// applied per-arity inside the DP, and a sufficiently large beam
    /// is admissible. Callers wanting a strict beam check should set
    /// `beam_width = usize::MAX` themselves.
    pub fn validate_admissibility(&self) -> Result<(), String> {
        if !self.admissibility_strict {
            return Ok(());
        }
        let mut violations: Vec<String> = Vec::new();
        if self.ub_margin.is_finite() {
            violations.push(format!(
                "ub_margin={} is finite (heuristic upper-bound pruning; \
                 cand_ub is a partial-subtree estimate, so any finite \
                 margin can prune optimal subtrees — set to \
                 f64::INFINITY)",
                self.ub_margin
            ));
        }
        if self.unsafe_filter_depth_threshold != 0 {
            violations.push(format!(
                "unsafe_filter_depth_threshold={} != 0 (B1 unsafe \
                 candidate filter; admissible only at depth==1)",
                self.unsafe_filter_depth_threshold
            ));
        }
        if self.lazy_explore_delta.is_finite() {
            violations.push(format!(
                "lazy_explore_delta={} is finite (lazy depth-1 explore \
                 prune; set to f64::INFINITY)",
                self.lazy_explore_delta
            ));
        }
        if self.closure_depth.is_some() {
            violations.push(format!(
                "closure_depth={:?} is Some (closure of \
                 shallow subtrees)",
                self.closure_depth
            ));
        }
        if self.path_penalty != 0.0 {
            violations.push(format!(
                "path_penalty={} != 0.0 (soft path-length penalty)",
                self.path_penalty
            ));
        }
        if self.greedy_extra_depth != 0 {
            violations.push(format!(
                "greedy_extra_depth={} != 0 (greedy boundary lookahead)",
                self.greedy_extra_depth
            ));
        }
        if self.cheap_boundary_greedy {
            violations.push("cheap_boundary_greedy=true (binary-only greedy guidance)".to_string());
        }
        if self.allow_heuristic_bound_transfer {
            violations.push(
                "allow_heuristic_bound_transfer=true (cross-subproblem \
                 similar-bound transfer is unproved)"
                    .to_string(),
            );
        }
        if self.local_thresh_scan {
            violations.push(
                "local_thresh_scan=true (lossy per-child threshold \
                 locality scan)"
                    .to_string(),
            );
        }
        if violations.is_empty() {
            Ok(())
        } else {
            Err(format!(
                "AdmissibilityPolicy strict mode: {} heuristic knob(s) \
                 engaged. Disable each before retrying:\n  - {}",
                violations.len(),
                violations.join("\n  - ")
            ))
        }
    }
}

/// Bin order used to propose a partition and its original-axis cut count.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CandidateOrder {
    Original,
    /// Counted on occupied bins before empty-bin routing is expanded.
    Reference {
        boundary_edges: usize,
    },
}

/// Result from collapse_bins: one K-way split option.
pub struct CollapseEntry {
    pub cost: f64,
    pub thresholds: Arc<[f64]>,
    pub bin_n: Vec<i64>,
    pub bin_counts: Vec<Vec<i64>>,
    /// Shape mode: maps original bin index -> group index (0..k-1).
    pub group_map: Option<Vec<usize>>,
    pub order: CandidateOrder,
}

// ---------------------------------------------------------------------------
// Profiling
// ---------------------------------------------------------------------------

pub const MAX_PROFILE_DEPTH: usize = 16;

/// Always-on profiling counters and phase timings, accumulated during a solve.
#[derive(Clone)]
pub struct SolverProfile {
    // Tier 1: Candidate funnel
    pub n_candidates_generated: u64,
    pub n_candidates_survived_phase_a: u64,
    pub n_candidates_materialized: u64,
    pub n_features_available: u64,
    pub n_features_filtered_tier1: u64,

    // Tier 1: Cache hit rates
    pub n_similar_lb_hits: u64,
    pub n_similar_lb_misses: u64,
    pub n_similar_ub_hits: u64,
    pub n_similar_ub_misses: u64,
    pub n_inline_bounds_calls: u64,

    // Tier 1: Heuristic effectiveness
    pub n_killer_move_hits: u64,
    pub n_killer_move_misses: u64,

    // Tier 1: Pruning / search quality
    pub n_nodes_pruned: u64,
    pub n_nodes_purified: u64,
    pub n_nodes_forced_leaf: u64,
    pub n_phase_a_early_terminations: u64,
    pub n_propagate_up_calls: u64,
    pub n_propagate_up_changes: u64,
    pub n_salvage_rounds: u64,
    pub n_salvage_reenqueued: u64,
    pub n_branch_bound_candidates_scope_pruned: u64,
    /// Candidates dropped by the top-k candidate beam
    /// (distinct from UB-margin pruning: happens before branch-bound sees the list).
    pub n_candidates_beam_truncated: u64,
    /// Count of `(node, feature)` calls where the DP edge-cost actually
    /// took the reference-ensemble-blended path (α < 1.0 with a reference ensemble present).
    /// Zero at α = 1.0 or when teacher_prob is absent. Diagnostic only.
    pub n_teacher_dp_active_nodes: u64,
    pub n_greedy_boundary_nodes: u64,
    pub n_greedy_score_calls: u64,
    pub n_greedy_cache_hits: u64,
    pub n_closure_nodes: u64,
    pub n_closure_fast_leaf: u64,
    pub n_closure_shared_hits: u64,
    /// Count of sibling candidates skipped by the cheap admissible
    /// lower-bound prune in `closure_build`'s sequential sibling loop.
    pub n_closure_lb_skipped: u64,
    /// Count of per-bin LB lookups that hit the closure-local LB
    /// cache and produced a tighter bound than `trivial_lb`. Diagnostic for
    /// how often the cache contributes to LB pruning.
    pub n_closure_lb_cache_hits: u64,
    /// Count of bins short-circuited as provably-leaf-optimal in
    /// the closure sibling loop, skipping closure recursion + cand_gen +
    /// create_subproblem walks. Triggered when `leaf_cost(bin) <= split_reg
    /// + 2·leaf_reg + bin_path` (no split can beat the leaf).
    pub n_closure_terminal_shortcircuit: u64,
    /// Count of candidates aborted mid-bin-loop by the
    /// intra-candidate alpha-beta prune. Bins are processed in ascending-LB
    /// order; once `running_score + sum_remaining_lbs >= best_split_cost`
    /// the remaining bins are skipped (the candidate provably can't improve
    /// best). Lossless.
    pub n_closure_alpha_beta_aborted: u64,
    /// Count of features filtered out at closure_depth==1 by
    /// the admissible `irr_j + min_split_overhead >= leaf_obj` test. Mirrors
    /// the upstream irreducible-error filter. Lossless at
    /// closure_depth_remaining==1 since children are forced to be leaves.
    pub n_closure_b1_filtered: u64,

    // Tier 1b: Closure sub-phase breakdown
    pub time_closure_create_sp: std::time::Duration,
    pub time_closure_cand_gen: std::time::Duration,
    pub time_closure_shortlist: std::time::Duration,
    pub time_closure_sibling_loop: std::time::Duration,

    // Tier 2: Phase timing
    pub time_candidate_gen: std::time::Duration,
    /// Exact completion lb: Select time of the bounded completion-lb DFS
    /// (`bounded_completion_min`), not included in `time_candidate_gen`.
    pub time_candidate_gen_lb: std::time::Duration,
    /// Completion-lb DFS: Select calls (lb Select cache misses).
    pub n_lb_selects: u64,
    /// Completion-lb DFS: remaining-depth-1 nodes answered by the closed
    /// form (depth-1 closed form), and those it could not answer (fell back to
    /// Select: a feature with more than `RASH_LB_D1_MAX_T` thresholds).
    pub n_lb_d1_closed: u64,
    pub n_lb_d1_fallback: u64,
    /// Peak number of split records held in the completion-lb Select cache
    /// (`merge_from` takes the max).
    pub lb_select_cache_records: u64,
    /// Completion-lb DFS: closed-form depth-1 calls whose prefix counts came
    /// from the row walk (depth-1 row walk), and memo hits at remaining
    /// depth 1 (any entry: closed-form, leaf or Select-path).
    pub n_lb_d1_rowwalk: u64,
    pub n_lb_d1_memo_hits: u64,
    /// Wall time inside `rash_depth1_min_closed` (completion-lb DFS).
    pub time_lb_d1_closed: std::time::Duration,
    /// Store threshold statistics at the completion-lb gate: the largest
    /// per-feature threshold count, and the sum / number of features with
    /// >= 1 threshold (mean = sum / count). `merge_from` takes the max.
    pub n_store_thresholds_max: u64,
    pub n_store_thresholds_sum: u64,
    pub n_store_threshold_features: u64,
    /// Step A Select calls (explore pool computed, not taken from the
    /// Shared Select pool cache).
    pub n_explore_selects: u64,
    /// Shared Select pool: Select pools taken from the shared cache (explore
    /// and completion-lb DFS).
    pub n_select_shared_hits: u64,
    /// Depth-1 sweep: depth-1 closed-form calls (counted in
    /// `n_lb_d1_closed`) whose counts came from the parent's sweep.
    pub n_lb_d1_swept: u64,
    /// Lb memo cap: remaining-depth-1 completion-lb memo entries
    /// dropped by the cap.
    pub n_lb_memo_evictions: u64,
    /// Integer-first depth-1 DP: depth-1 closed-form DP calls answered by the
    /// integer-first DP (`rash_d1_dp_int`), and those that fell back to the
    /// reference DP (class count not 2 / 3, or a guard failed).
    pub n_lb_d1_intdp: u64,
    pub n_lb_d1_intdp_fallback: u64,
    pub time_phase_a: std::time::Duration,
    pub time_phase_b: std::time::Duration,
    pub time_propagate: std::time::Duration,
    pub time_killer_eval: std::time::Duration,
    pub time_partition_precompute: std::time::Duration,
    pub time_greedy_boundary: std::time::Duration,
    pub time_closure: std::time::Duration,

    // Tier 3: Per-depth stats
    /// Per-depth wall-time breakdowns for diagnostic profiling.
    pub time_candgen_per_depth: [std::time::Duration; MAX_PROFILE_DEPTH],
    pub time_phase_a_per_depth: [std::time::Duration; MAX_PROFILE_DEPTH],
    pub time_phase_b_per_depth: [std::time::Duration; MAX_PROFILE_DEPTH],
    pub nodes_at_depth: [u64; MAX_PROFILE_DEPTH],
    pub sum_gap_at_depth: [f64; MAX_PROFILE_DEPTH],
    pub n_solved_at_depth: [u64; MAX_PROFILE_DEPTH],
    pub sum_inline_lb_gap: [f64; MAX_PROFILE_DEPTH],
    pub n_materialized_at_depth: [u64; MAX_PROFILE_DEPTH],

    // Tier 5: Diagnostic — depth-1 explosion analysis
    pub n_reexplorations: u64,
    pub n_depth1_from_rank0: u64,
    pub n_depth1_from_rank_n: u64,
    pub n_depth1_lazy_skipped: u64,

    // Tier 6: Beam rejection analysis (per-arity, index 0 = k=2)
    pub beam_gap_max_per_arity: [f64; 8],
    pub beam_gap_sum_per_arity: [f64; 8],
    pub beam_gap_count_per_arity: [u64; 8],
    pub beam_saturated_cells_total_per_arity: [u64; 8],

    // Tier 4: Memory snapshot (filled once at end)
    pub memo_size: usize,
    pub similar_lb_size: usize,
    pub similar_ub_size: usize,
    pub parent_map_size: usize,
    pub used_features_size: usize,
    pub peak_queue_size: usize,
}

impl Default for SolverProfile {
    fn default() -> Self {
        SolverProfile {
            n_candidates_generated: 0,
            n_candidates_survived_phase_a: 0,
            n_candidates_materialized: 0,
            n_features_available: 0,
            n_features_filtered_tier1: 0,
            n_similar_lb_hits: 0,
            n_similar_lb_misses: 0,
            n_similar_ub_hits: 0,
            n_similar_ub_misses: 0,
            n_inline_bounds_calls: 0,
            n_killer_move_hits: 0,
            n_killer_move_misses: 0,
            n_nodes_pruned: 0,
            n_nodes_purified: 0,
            n_nodes_forced_leaf: 0,
            n_phase_a_early_terminations: 0,
            n_propagate_up_calls: 0,
            n_propagate_up_changes: 0,
            n_salvage_rounds: 0,
            n_salvage_reenqueued: 0,
            n_branch_bound_candidates_scope_pruned: 0,
            n_candidates_beam_truncated: 0,
            n_teacher_dp_active_nodes: 0,
            n_greedy_boundary_nodes: 0,
            n_greedy_score_calls: 0,
            n_greedy_cache_hits: 0,
            n_closure_nodes: 0,
            n_closure_fast_leaf: 0,
            n_closure_shared_hits: 0,
            n_closure_lb_skipped: 0,
            n_closure_lb_cache_hits: 0,
            n_closure_terminal_shortcircuit: 0,
            n_closure_alpha_beta_aborted: 0,
            n_closure_b1_filtered: 0,
            time_closure_create_sp: std::time::Duration::ZERO,
            time_closure_cand_gen: std::time::Duration::ZERO,
            time_closure_shortlist: std::time::Duration::ZERO,
            time_closure_sibling_loop: std::time::Duration::ZERO,
            time_candidate_gen: std::time::Duration::ZERO,
            time_candidate_gen_lb: std::time::Duration::ZERO,
            n_lb_selects: 0,
            n_lb_d1_closed: 0,
            n_lb_d1_fallback: 0,
            lb_select_cache_records: 0,
            n_lb_d1_rowwalk: 0,
            n_lb_d1_memo_hits: 0,
            time_lb_d1_closed: std::time::Duration::ZERO,
            n_store_thresholds_max: 0,
            n_store_thresholds_sum: 0,
            n_store_threshold_features: 0,
            n_explore_selects: 0,
            n_select_shared_hits: 0,
            n_lb_d1_swept: 0,
            n_lb_memo_evictions: 0,
            n_lb_d1_intdp: 0,
            n_lb_d1_intdp_fallback: 0,
            time_phase_a: std::time::Duration::ZERO,
            time_phase_b: std::time::Duration::ZERO,
            time_propagate: std::time::Duration::ZERO,
            time_killer_eval: std::time::Duration::ZERO,
            time_partition_precompute: std::time::Duration::ZERO,
            time_greedy_boundary: std::time::Duration::ZERO,
            time_closure: std::time::Duration::ZERO,
            time_candgen_per_depth: [std::time::Duration::ZERO; MAX_PROFILE_DEPTH],
            time_phase_a_per_depth: [std::time::Duration::ZERO; MAX_PROFILE_DEPTH],
            time_phase_b_per_depth: [std::time::Duration::ZERO; MAX_PROFILE_DEPTH],
            nodes_at_depth: [0; MAX_PROFILE_DEPTH],
            sum_gap_at_depth: [0.0; MAX_PROFILE_DEPTH],
            n_solved_at_depth: [0; MAX_PROFILE_DEPTH],
            sum_inline_lb_gap: [0.0; MAX_PROFILE_DEPTH],
            n_materialized_at_depth: [0; MAX_PROFILE_DEPTH],
            n_reexplorations: 0,
            n_depth1_from_rank0: 0,
            n_depth1_from_rank_n: 0,
            n_depth1_lazy_skipped: 0,

            beam_gap_max_per_arity: [0.0; 8],
            beam_gap_sum_per_arity: [0.0; 8],
            beam_gap_count_per_arity: [0; 8],
            beam_saturated_cells_total_per_arity: [0; 8],

            memo_size: 0,
            similar_lb_size: 0,
            similar_ub_size: 0,
            parent_map_size: 0,
            used_features_size: 0,
            peak_queue_size: 0,
        }
    }
}

/// Length of the completion-lb counter array (`SolverProfile::lb_stats`).
pub const LB_STATS_LEN: usize = 16;

/// Completion-lb DFS counters `[n_lb_selects, n_lb_d1_closed,
/// n_lb_d1_fallback, lb_select_cache_records,
/// n_lb_d1_rowwalk, n_lb_d1_memo_hits, time_lb_d1_closed (ns),
/// n_store_thresholds_max, n_store_thresholds_sum,
/// n_store_threshold_features, n_explore_selects, n_select_shared_hits,
/// n_lb_d1_swept, n_lb_memo_evictions, n_lb_d1_intdp,
/// n_lb_d1_intdp_fallback]`
/// of `b` added to `a` (counts and the time summed; the cache records and
/// the store threshold statistics max).
pub fn lb_stats_merge(a: [u64; LB_STATS_LEN], b: [u64; LB_STATS_LEN]) -> [u64; LB_STATS_LEN] {
    [
        a[0] + b[0],
        a[1] + b[1],
        a[2] + b[2],
        a[3].max(b[3]),
        a[4] + b[4],
        a[5] + b[5],
        a[6] + b[6],
        a[7].max(b[7]),
        a[8].max(b[8]),
        a[9].max(b[9]),
        a[10] + b[10],
        a[11] + b[11],
        a[12] + b[12],
        a[13] + b[13],
        a[14] + b[14],
        a[15] + b[15],
    ]
}

impl SolverProfile {
    /// Completion-lb DFS counters (layout of `lb_stats_merge`).
    pub fn lb_stats(&self) -> [u64; LB_STATS_LEN] {
        [
            self.n_lb_selects,
            self.n_lb_d1_closed,
            self.n_lb_d1_fallback,
            self.lb_select_cache_records,
            self.n_lb_d1_rowwalk,
            self.n_lb_d1_memo_hits,
            self.time_lb_d1_closed.as_nanos() as u64,
            self.n_store_thresholds_max,
            self.n_store_thresholds_sum,
            self.n_store_threshold_features,
            self.n_explore_selects,
            self.n_select_shared_hits,
            self.n_lb_d1_swept,
            self.n_lb_memo_evictions,
            self.n_lb_d1_intdp,
            self.n_lb_d1_intdp_fallback,
        ]
    }

    /// Accumulate counters and timings from a worker-local profile.
    /// Skips memory-snapshot fields (filled once at end by the main solver).
    pub fn merge_from(&mut self, other: &SolverProfile) {
        self.n_candidates_generated += other.n_candidates_generated;
        self.n_candidates_survived_phase_a += other.n_candidates_survived_phase_a;
        self.n_candidates_materialized += other.n_candidates_materialized;
        self.n_features_available += other.n_features_available;
        self.n_features_filtered_tier1 += other.n_features_filtered_tier1;
        self.n_similar_lb_hits += other.n_similar_lb_hits;
        self.n_similar_lb_misses += other.n_similar_lb_misses;
        self.n_similar_ub_hits += other.n_similar_ub_hits;
        self.n_similar_ub_misses += other.n_similar_ub_misses;
        self.n_inline_bounds_calls += other.n_inline_bounds_calls;
        self.n_killer_move_hits += other.n_killer_move_hits;
        self.n_killer_move_misses += other.n_killer_move_misses;
        self.n_nodes_pruned += other.n_nodes_pruned;
        self.n_nodes_purified += other.n_nodes_purified;
        self.n_nodes_forced_leaf += other.n_nodes_forced_leaf;
        self.n_phase_a_early_terminations += other.n_phase_a_early_terminations;
        self.n_propagate_up_calls += other.n_propagate_up_calls;
        self.n_propagate_up_changes += other.n_propagate_up_changes;
        self.n_salvage_rounds += other.n_salvage_rounds;
        self.n_salvage_reenqueued += other.n_salvage_reenqueued;
        self.n_branch_bound_candidates_scope_pruned += other.n_branch_bound_candidates_scope_pruned;
        self.n_candidates_beam_truncated += other.n_candidates_beam_truncated;
        self.n_teacher_dp_active_nodes += other.n_teacher_dp_active_nodes;
        self.n_greedy_boundary_nodes += other.n_greedy_boundary_nodes;
        self.n_greedy_score_calls += other.n_greedy_score_calls;
        self.n_greedy_cache_hits += other.n_greedy_cache_hits;
        self.n_closure_nodes += other.n_closure_nodes;
        self.n_closure_fast_leaf += other.n_closure_fast_leaf;
        self.n_closure_shared_hits += other.n_closure_shared_hits;
        self.n_closure_lb_skipped += other.n_closure_lb_skipped;
        self.n_closure_lb_cache_hits += other.n_closure_lb_cache_hits;
        self.n_closure_terminal_shortcircuit += other.n_closure_terminal_shortcircuit;
        self.n_closure_alpha_beta_aborted += other.n_closure_alpha_beta_aborted;
        self.n_closure_b1_filtered += other.n_closure_b1_filtered;
        self.time_closure_create_sp += other.time_closure_create_sp;
        self.time_closure_cand_gen += other.time_closure_cand_gen;
        self.time_closure_shortlist += other.time_closure_shortlist;
        self.time_closure_sibling_loop += other.time_closure_sibling_loop;
        self.time_candidate_gen += other.time_candidate_gen;
        self.time_candidate_gen_lb += other.time_candidate_gen_lb;
        self.n_lb_selects += other.n_lb_selects;
        self.n_lb_d1_closed += other.n_lb_d1_closed;
        self.n_lb_d1_fallback += other.n_lb_d1_fallback;
        self.lb_select_cache_records = self
            .lb_select_cache_records
            .max(other.lb_select_cache_records);
        self.n_lb_d1_rowwalk += other.n_lb_d1_rowwalk;
        self.n_lb_d1_memo_hits += other.n_lb_d1_memo_hits;
        self.time_lb_d1_closed += other.time_lb_d1_closed;
        self.n_store_thresholds_max = self
            .n_store_thresholds_max
            .max(other.n_store_thresholds_max);
        self.n_store_thresholds_sum = self
            .n_store_thresholds_sum
            .max(other.n_store_thresholds_sum);
        self.n_store_threshold_features = self
            .n_store_threshold_features
            .max(other.n_store_threshold_features);
        self.n_explore_selects += other.n_explore_selects;
        self.n_select_shared_hits += other.n_select_shared_hits;
        self.n_lb_d1_swept += other.n_lb_d1_swept;
        self.n_lb_memo_evictions += other.n_lb_memo_evictions;
        self.n_lb_d1_intdp += other.n_lb_d1_intdp;
        self.n_lb_d1_intdp_fallback += other.n_lb_d1_intdp_fallback;
        self.time_phase_a += other.time_phase_a;
        self.time_phase_b += other.time_phase_b;
        self.time_propagate += other.time_propagate;
        self.time_killer_eval += other.time_killer_eval;
        self.time_partition_precompute += other.time_partition_precompute;
        self.time_greedy_boundary += other.time_greedy_boundary;
        self.time_closure += other.time_closure;
        self.n_reexplorations += other.n_reexplorations;
        self.n_depth1_from_rank0 += other.n_depth1_from_rank0;
        self.n_depth1_from_rank_n += other.n_depth1_from_rank_n;
        self.n_depth1_lazy_skipped += other.n_depth1_lazy_skipped;
        for i in 0..MAX_PROFILE_DEPTH {
            self.nodes_at_depth[i] += other.nodes_at_depth[i];
            self.n_solved_at_depth[i] += other.n_solved_at_depth[i];
            self.n_materialized_at_depth[i] += other.n_materialized_at_depth[i];
            self.sum_gap_at_depth[i] += other.sum_gap_at_depth[i];
            self.sum_inline_lb_gap[i] += other.sum_inline_lb_gap[i];
            self.time_candgen_per_depth[i] += other.time_candgen_per_depth[i];
            self.time_phase_a_per_depth[i] += other.time_phase_a_per_depth[i];
            self.time_phase_b_per_depth[i] += other.time_phase_b_per_depth[i];
        }
        for i in 0..8 {
            if other.beam_gap_max_per_arity[i] > self.beam_gap_max_per_arity[i] {
                self.beam_gap_max_per_arity[i] = other.beam_gap_max_per_arity[i];
            }
            self.beam_gap_sum_per_arity[i] += other.beam_gap_sum_per_arity[i];
            self.beam_gap_count_per_arity[i] += other.beam_gap_count_per_arity[i];
            self.beam_saturated_cells_total_per_arity[i] +=
                other.beam_saturated_cells_total_per_arity[i];
        }
    }
}

/// Side-channel stats from generate_candidates for the candidate-funnel counts.
pub struct CandGenStats {
    pub n_available: u64,
    pub n_filtered_available: u64,
    /// Per-arity beam gap (index 0 = k=2): max gap across features at final DP cell.
    pub beam_gap_per_arity: Vec<f64>,
    /// Per-arity: total number of saturated cells summed across features.
    pub beam_saturated_cells_per_arity: Vec<u64>,
    /// Number of candidates dropped by the top-k candidate
    /// beam (candidates outside the ⌈√N⌉ prefix after anchor preservation).
    /// Incremented in `select_top_k_with_anchors`; zero when that knob is off.
    pub n_candidates_beam_truncated: u64,
    /// Count of `(node, feature)` calls where the DP edge-cost actually
    /// took the reference-ensemble-blended path at this node. See
    /// `SolverProfile::n_teacher_dp_active_nodes`.
    pub n_teacher_dp_active_nodes: u64,
}

// ---------------------------------------------------------------------------
// Rashomon DAG types
// ---------------------------------------------------------------------------

/// Compact split specification — the DAG key for grouping child-tuples.
/// Two splits on the same feature with different thresholds (or different
/// arities) are distinct SplitSpecs. In shape mode, two splits that share
/// (feature, arity, thresholds) but differ in `group_map` route rows to
/// different child supports and therefore must remain distinct — so
/// `group_map` participates in Eq/Hash.
///
/// Per-DAG interner storage: many splits share few unique threshold
/// tuples, so per-split `Arc<[f64]>` / `Option<Arc<[u16]>>` pointers are
/// replaced by u32 indices into per-DAG pools (`FactoredDag.threshold_pool`
/// + `group_map_pool`).
#[derive(Clone, Debug)]
pub struct SplitSpec {
    pub feature_index: u16,
    pub arity: u8,
    pub thresholds: Arc<[f64]>, // K-1 (contig) or L-1 (shape) threshold values
    /// Shape mode: maps bin index (0..L) -> child index (0..arity). None = contig
    /// (identity map; bin i goes to child i).
    pub group_map: Option<Arc<[u16]>>,
}

impl PartialEq for SplitSpec {
    fn eq(&self, other: &Self) -> bool {
        self.feature_index == other.feature_index
            && self.arity == other.arity
            && (Arc::ptr_eq(&self.thresholds, &other.thresholds)
                || (self.thresholds.len() == other.thresholds.len()
                    && self
                        .thresholds
                        .iter()
                        .zip(other.thresholds.iter())
                        .all(|(a, b)| (a - b).abs() < 1e-12)))
            && match (&self.group_map, &other.group_map) {
                (None, None) => true,
                (Some(a), Some(b)) => {
                    Arc::ptr_eq(a, b)
                        || (a.len() == b.len() && a.iter().zip(b.iter()).all(|(x, y)| x == y))
                }
                _ => false,
            }
    }
}
impl Eq for SplitSpec {}

impl Hash for SplitSpec {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.feature_index.hash(state);
        self.arity.hash(state);
        for t in self.thresholds.iter() {
            t.to_bits().hash(state);
        }
        match &self.group_map {
            None => 0u8.hash(state),
            Some(gm) => {
                1u8.hash(state);
                for g in gm.iter() {
                    g.hash(state);
                }
            }
        }
    }
}

/// Leaf info stored in a Rashomon DAG node.
#[derive(Clone, Debug)]
pub struct RLeafInfo {
    pub prediction: u8,
    pub loss: f64,
    pub n_samples: usize,
}

/// Marker for a "prefix leaf": a `FactoredSubproblem` that became terminal
/// because the configured `solve_depth` ran out (lookahead truncation), not
/// because splitting failed to beat the leaf objective.
///
/// Carries the capture-set bitmask + class counts needed for downstream
/// recursive fill to launch a nested RESPLINTER solve at each such leaf.
/// `greedy_loss` is the closure/leaf upper bound at the frontier; in
/// SPLIT-style recursive fill it doubles as the per-leaf `rashomon_bound`.
#[derive(Clone, Debug)]
pub struct PrefixLeafInfo {
    pub support: SupportMask,
    pub class_counts: [i64; MAX_CLASSES],
    pub n_classes: u8,
    pub n_samples: usize,
    pub depth_remaining: u8,
    pub greedy_loss: f64,
}

/// Legacy serialized approximation fields and operational build statistics.
/// Numerical delta fields from older files are historical diagnostics, not
/// certified coverage bounds. New fits leave them at zero and report
/// retained-pool completeness through the explicit enumeration status.
#[derive(Clone, Debug, Default)]
pub struct DeltaBudget {
    /// Gap from beam search (missed partitions beyond beam width).
    pub delta_beam: f64,
    /// Gap from transition point compression (merged bin boundaries).
    pub delta_compress: f64,
    /// Gap from direct Rust no-threshold fallback discretization.
    pub delta_quantize: f64,
    /// Gap from proxy criterion vs true objective (0 if Misclass).
    pub delta_criterion: f64,
    /// Sum of the prefix-layer components above (NOT inclusive of leaf δ).
    pub delta_total: f64,
    /// Worst per-leaf δ_total observed across all nested fills (leaf-norm).
    /// Zero when no recursive fill ran.
    pub delta_leaf_max: f64,
    /// Sample-weighted average per-leaf δ_total: Σᵢ (n_i / N) · δᵢ_total
    /// (leaf-norm). Zero when no recursive fill ran.
    pub delta_leaf_weighted: f64,
    /// Number of prefix leaves where a nested RESPLINTER solve actually ran.
    pub n_leaves_solved: u32,
    /// Number of prefix leaves that short-circuited (greedy already
    /// dominates per the SPLIT recipe; no nested solve, no nested DAG).
    pub n_leaves_short_circuited: u32,
    /// Completion search (Step B): depth-d_l placeholders reachable from the
    /// root of the Step-A DAG (before the graft); of those, grafted with a
    /// non-empty MiniRESPLINTER DAG / emptied (beta_u < lb_u, or no tree
    /// within beta_u) / whose nested solve (or a solve below it) timed out.
    /// Zero outside the completion search.
    pub n_prefix_leaves: u32,
    pub n_leaves_filled: u32,
    pub n_leaves_empty: u32,
    pub n_leaves_timed_out: u32,
    /// Completion search: support groups whose placeholder contexts give unequal
    /// scopes beta_u (several prefix costs to one placeholder, or several
    /// placeholders with one support); the group is solved once at the
    /// widest scope.
    pub n_stepb_groups_unequal_scope: u32,
    /// Completion search: Select time (`SolverProfile::time_candidate_gen`, s)
    /// summed over the MiniRESPLINTER solves, incl. nested Step B below
    /// them. Zero outside the completion search.
    pub time_candidate_gen_stepb_s: f64,
    /// Completion search: Select time of the completion-lb DFS
    /// (`SolverProfile::time_candidate_gen_lb`, s) of the same solves.
    pub time_candidate_gen_lb_stepb_s: f64,
    /// Completion search: completion-lb DFS counters of the same solves
    /// (layout of `lb_stats_merge`: counts and time summed;
    /// `lb_select_cache_records` and the store threshold statistics max).
    pub lb_stats_stepb: [u64; LB_STATS_LEN],
    /// Completion search (Step B) ran as one shared seeded solver per depth level
    /// (shared Step B).
    pub stepb_shared: bool,
    /// Re-anchor (`rashomon_dag::reanchor_at_dag_opt`): the DAG was
    /// Re-anchored at its own optimum; the anchor, bound and tree count
    /// before it. Telemetry only (not serialised); 0 / false otherwise.
    pub reanchored: bool,
    pub reanchor_opt_pre: f64,
    pub reanchor_bound_pre: f64,
    pub reanchor_tree_count_pre: u128,
}

/// The objective prices a tree by its leaves only: no
/// split_reg (`(1 − reg_alpha)·reg = 0`) and no path penalty. The Rashomon
/// DAG carries only leaf costs, so this is the condition under which its
/// costs are the objective (`rashomon_incomplete_reason` `non_leaf_objective`
/// otherwise; exact completion lb and the exact cost lattice require it).
pub fn leaf_only_objective(reg: f64, reg_alpha: f64, path_penalty: f64) -> bool {
    (1.0 - reg_alpha.clamp(0.0, 1.0)) * reg == 0.0 && path_penalty.max(0.0) == 0.0
}

// ---------------------------------------------------------------------------
// Factored DAG types (cross-product-free representation)
// ---------------------------------------------------------------------------

/// Arena index into FactoredDag::subproblems.
pub type FSubRef = u32;

/// High-bit-tagged `FSubRef`
/// encoding. When the high bit is set, the FSubRef is NOT an index into
/// `FactoredDag::subproblems`; it's an inline-leaf id, indexing into
/// `FactoredDag::inline_leaves` / `FactoredDag::inline_leaf_costs`.
///
/// This keeps leaf subproblems out of the pre-quotient arena by storing
/// them as compact tuples `(f64 cost, RLeafInfo)` indexed directly off the
/// FactoredDag.
pub const LEAF_TAG: u32 = 0x8000_0000;
pub const LEAF_MASK: u32 = 0x7FFF_FFFF;

#[inline]
pub fn is_leaf_ref(cref: FSubRef) -> bool {
    (cref & LEAF_TAG) != 0
}

#[inline]
pub fn leaf_id(cref: FSubRef) -> usize {
    (cref & LEAF_MASK) as usize
}

#[inline]
pub fn make_leaf_ref(id: u32) -> FSubRef {
    id | LEAF_TAG
}

/// One achievable cost level at a subproblem.
///
/// `tree_count` is stored as two `u64` halves to keep the struct at
/// 24 bytes (vs 32 with a single `u128` field, which forces 8 bytes
/// of padding before the 16-byte-aligned `u128`), about 25% of the
/// `cost_entries` Vec backing. Use `CostEntry::new` to construct and `tree_count()` /
/// `set_tree_count()` to access the logical u128 value.
#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(C)]
pub struct CostEntry {
    pub cost: f64,
    tc_lo: u64,
    tc_hi: u64,
}

impl CostEntry {
    #[inline]
    pub fn new(cost: f64, tree_count: u128) -> Self {
        Self {
            cost,
            tc_lo: tree_count as u64,
            tc_hi: (tree_count >> 64) as u64,
        }
    }

    #[inline]
    pub fn tree_count(&self) -> u128 {
        ((self.tc_hi as u128) << 64) | (self.tc_lo as u128)
    }

    #[inline]
    pub fn set_tree_count(&mut self, tc: u128) {
        self.tc_lo = tc as u64;
        self.tc_hi = (tc >> 64) as u64;
    }
}

// ---------------------------------------------------------------------------
// Saturating u128 tree-count arithmetic
// ---------------------------------------------------------------------------

/// Number of tree-count additions / multiplications that overflowed u128 and
/// were clamped to `u128::MAX`, process-wide. Monotone (never reset): a fit
/// snapshots it before building and flags its DAG `count_saturated` when it
/// grew. Concurrent fits in one process may see each other's events (a
/// conservative false positive, never a missed saturation).
static COUNT_SATURATION_EVENTS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

#[cold]
#[inline(never)]
fn note_count_saturation() -> u128 {
    COUNT_SATURATION_EVENTS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    u128::MAX
}

/// Current value of the process-wide saturation-event counter.
#[inline]
pub fn count_saturation_events() -> u64 {
    COUNT_SATURATION_EVENTS.load(std::sync::atomic::Ordering::Relaxed)
}

/// `a + b` on tree counts, clamped at `u128::MAX` (records the event).
#[inline]
pub fn count_add(a: u128, b: u128) -> u128 {
    match a.checked_add(b) {
        Some(v) => v,
        None => note_count_saturation(),
    }
}

/// `a * b` on tree counts, clamped at `u128::MAX` (records the event).
#[inline]
pub fn count_mul(a: u128, b: u128) -> u128 {
    match a.checked_mul(b) {
        Some(v) => v,
        None => note_count_saturation(),
    }
}

/// Saturating Σ of tree counts (`count_add` fold).
#[inline]
pub fn count_sum<I: IntoIterator<Item = u128>>(it: I) -> u128 {
    it.into_iter().fold(0u128, count_add)
}

/// Saturating Π of tree counts (`count_mul` fold).
#[inline]
pub fn count_product<I: IntoIterator<Item = u128>>(it: I) -> u128 {
    it.into_iter().fold(1u128, count_mul)
}

/// A factored split: per-child cost lists instead of cross-product tuples.
///
/// **On-demand cost-schedule materialization**: per-child
/// cost schedules are NO LONGER stored on the split. Instead, children
/// is a `Vec<FSubRef>` and schedules are recomputed on demand at
/// extract time via `rashomon_dag::materialize_schedule(dag, sub_ref,
/// scope)`, which caches results in `FactoredDag.schedule_cache`, so a
/// child costs 4 B instead of an `(FSubRef, Arc<...>)` pair.
/// Inline storage for a split's child FSubRefs. The
/// dominant case is k=2 (binary); the ceiling is `max_k` (≤5 in shipped
/// configs). Up to 5 `u32`s are stored inline with the `smallvec` `union`
/// feature: 8 B (capacity/len word) + 24 B (union of `[u32;5]` and
/// `(ptr,len)`, padded to 8-byte alignment) = 32 B, the children field of
/// the 56 B `FactoredSplit` (see its field docs). Without `union` the data
/// is an enum needing an extra discriminant word. Inlining removes the
/// SEPARATE per-split heap allocation, the children
/// backing, and the malloc fragmentation those tiny allocations drove.
/// k>5 spills to the heap automatically, with identical semantics to the
/// old `Vec`, so this is safe for any k.
///
/// `SmallVec` derefs to `&[FSubRef]` / `&mut [FSubRef]`, so `.iter()`,
/// `.len()`, indexing, `for x in &mut c`, etc. are all unchanged at the
/// read sites. Construction sites build a `Vec` and convert via
/// `SmallVec::from_vec(..)` / `.into()`.
pub type InlineChildren = SmallVec<[FSubRef; 5]>;

#[derive(Clone, Debug)]
pub struct FactoredSplit {
    /// Index into the per-DAG `FactoredDag.spec_pool`
    /// (`Vec<SplitSpec>`). Many splits share few UNIQUE SplitSpecs, so a
    /// 4 B canonical id replaces a 40 B inline spec (fat-pointer/Arc
    /// slots) on every split. `tree_count` is a u128
    /// stored as two u64 halves (8-align, no 16-byte padding): 16 B
    /// tree_count + 32 B children `SmallVec<[u32; 5]>` + 4 B spec_id = 52 →
    /// padded to 56 B (checked in
    /// `tests/rashomon_u128_counts.rs::struct_sizes`). The
    /// distinct specs are deduplicated into `spec_pool` during build via a
    /// `SplitSpecPool` interner; readers resolve `spec_id` → `&SplitSpec`
    /// via `dag.spec_of(split)` / `&dag.spec_pool[split.spec_id as usize]`.
    pub spec_id: SpecId,
    /// k subproblem references (inline up to 5; spills to heap beyond).
    /// Schedules per (child, scope) are materialized lazily at extract
    /// time via the dag's `schedule_cache`.
    pub children: InlineChildren,
    /// Precomputed total trees from valid cost combos (Σ child_costs ≤ scope),
    /// a logical u128 (saturating at `u128::MAX`, see `count_add`).
    ///
    /// Stored as two `u64` halves, like `CostEntry`, so the struct keeps
    /// 8-byte alignment (a `u128` field would force 16-byte alignment and
    /// padding). A u64 count overflowed (and panicked) above 1.8e19 trees,
    /// which large multiway Rashomon sets reach. Construct with
    /// `FactoredSplit::new` and access via `tree_count()` /
    /// `set_tree_count()`.
    tc_lo: u64,
    tc_hi: u64,
}

impl FactoredSplit {
    #[inline]
    pub fn new(spec_id: SpecId, children: InlineChildren, tree_count: u128) -> Self {
        Self {
            spec_id,
            children,
            tc_lo: tree_count as u64,
            tc_hi: (tree_count >> 64) as u64,
        }
    }

    #[inline]
    pub fn tree_count(&self) -> u128 {
        ((self.tc_hi as u128) << 64) | (self.tc_lo as u128)
    }

    #[inline]
    pub fn set_tree_count(&mut self, tc: u128) {
        self.tc_lo = tc as u64;
        self.tc_hi = (tc >> 64) as u64;
    }

    /// Feed the count into a structural hash. Counts below 2^64 hash as
    /// the u64 they were stored as before the u128 widening, so hash
    /// values (and any dedup / quotient order derived from them) are
    /// unchanged for every DAG that fit in u64.
    #[inline]
    pub fn hash_tree_count<H: Hasher>(&self, state: &mut H) {
        if self.tc_hi == 0 {
            self.tc_lo.hash(state);
        } else {
            self.tree_count().hash(state);
        }
    }
}

const _: () = assert!(std::mem::size_of::<FactoredSplit>() <= 56);

impl PartialEq for FactoredSplit {
    fn eq(&self, other: &Self) -> bool {
        // `spec_id` is canonical within a DAG (the SplitSpecPool
        // dedups identical logical specs to one id), so comparing ids is
        // equivalent to comparing spec contents AND faster. This keeps the
        // split-pool dedup (`promote_splits_to_pool`) and structural hashing
        // logically identical to the pre-pool inline-spec behaviour.
        self.spec_id == other.spec_id
            && self.tc_lo == other.tc_lo
            && self.tc_hi == other.tc_hi
            && self.children == other.children
    }
}
impl Eq for FactoredSplit {}

impl Hash for FactoredSplit {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // Hash the canonical spec_id (equal logical specs share an
        // id) instead of the inline spec fields. Equivalent for the split
        // pool's content-addressing within a DAG.
        self.spec_id.hash(state);
        for c in &self.children {
            c.hash(state);
        }
        self.hash_tree_count(state);
    }
}

/// Canonical pool ID for an interned `SplitSpec`. Index into
/// `FactoredDag.spec_pool` (`Vec<SplitSpec>`).
pub type SpecId = u32;

/// Build-time interner that deduplicates distinct `SplitSpec`s into a flat
/// `Vec<SplitSpec>`, returning a canonical `SpecId` for each. Identical
/// logical specs (same feature/arity/thresholds/group_map per
/// `SplitSpec`'s Eq) collapse to one id, so a `FactoredSplit` need only
/// store a 4 B id rather than the 40 B inline spec.
///
/// The `lookup` map is keyed by the spec content hash and confirms full
/// equality on hit (defends against hash collisions). After build the
/// caller calls `into_pool()` to extract the resident `Vec<SplitSpec>` and
/// drop the lookup map.
#[derive(Default, Debug, Clone)]
pub struct SplitSpecPool {
    pub specs: Vec<SplitSpec>,
    lookup: rustc_hash::FxHashMap<u64, Vec<SpecId>>,
}

impl SplitSpecPool {
    pub fn new() -> Self {
        Self::default()
    }

    fn spec_hash(s: &SplitSpec) -> u64 {
        use rustc_hash::FxHasher;
        use std::hash::Hasher;
        let mut h = FxHasher::default();
        Hash::hash(s, &mut h);
        h.finish()
    }

    /// Intern a spec (by reference), returning its canonical `SpecId`.
    /// Clones the spec only on first insertion (the clone is cheap: it's
    /// fat-pointer/Arc slot copies + Arc refcount bumps, payloads shared).
    pub fn intern(&mut self, spec: &SplitSpec) -> SpecId {
        let h = Self::spec_hash(spec);
        let bucket = self.lookup.entry(h).or_default();
        for &id in bucket.iter() {
            if &self.specs[id as usize] == spec {
                return id;
            }
        }
        let id = self.specs.len() as SpecId;
        self.specs.push(spec.clone());
        bucket.push(id);
        id
    }

    /// Extract the resident `Vec<SplitSpec>` pool, dropping the lookup map.
    pub fn into_pool(self) -> Vec<SplitSpec> {
        let mut specs = self.specs;
        specs.shrink_to_fit();
        specs
    }

    pub fn unique_count(&self) -> usize {
        self.specs.len()
    }
}

/// Canonical pool ID for an interned `FactoredSplit`. Index
/// into `FactoredDag.split_pool.entries`.
pub type SplitId = u32;

/// Per-DAG, content-addressable pool of canonical `FactoredSplit`s.
///
/// Built post-quotient/post-recompose by `promote_splits_to_pool`. Once
/// promoted, `FactoredSubproblem.splits` is emptied and `split_ids`
/// indexes the pool. The `lookup` map is dropped after construction via
/// `finalize_drop_lookup` so only `entries` survives on the resident DAG.
#[derive(Default, Debug, Clone)]
pub struct FactoredSplitPool {
    pub entries: Vec<FactoredSplit>,
    /// Hash → SplitId. Used only during construction; dropped after.
    pub lookup: rustc_hash::FxHashMap<u64, u32>,
    /// Telemetry: count of intern attempts (== total splits seen across
    /// all subproblems before dedup). `entries.len()` is the unique count.
    pub n_intern_attempts: u64,
}

impl FactoredSplitPool {
    pub fn new() -> Self {
        Self::default()
    }

    fn content_hash(s: &FactoredSplit) -> u64 {
        use rustc_hash::FxHasher;
        use std::hash::Hasher;
        let mut h = FxHasher::default();
        std::hash::Hash::hash(s, &mut h);
        h.finish()
    }

    /// Insert split (move-in), return canonical `SplitId`. If the split
    /// already exists in the pool, the passed-in value is dropped.
    pub fn intern(&mut self, split: FactoredSplit) -> SplitId {
        self.n_intern_attempts += 1;
        let h = Self::content_hash(&split);
        if let Some(&existing) = self.lookup.get(&h) {
            // Defend against hash collisions: confirm full equality.
            if self.entries[existing as usize] == split {
                return existing;
            }
            // Hash collision with different content — fall through and
            // append. The lookup table maps one canonical hash; collisions
            // are rare enough we accept a tiny dedup miss.
        }
        let id = self.entries.len() as u32;
        self.entries.push(split);
        self.lookup.insert(h, id);
        id
    }

    /// Drop the construction-time lookup table; shrink `entries`.
    /// Call once after all interning is complete.
    pub fn finalize_drop_lookup(&mut self) {
        self.lookup = rustc_hash::FxHashMap::default();
        self.entries.shrink_to_fit();
    }

    pub fn unique_count(&self) -> u64 {
        self.entries.len() as u64
    }

    pub fn dedup_hits(&self) -> u64 {
        self.n_intern_attempts
            .saturating_sub(self.entries.len() as u64)
    }

    /// Approximate resident bytes: entries Vec backing + per-FactoredSplit
    /// inline (struct + children Vec heap). Threshold/group_map Arcs are
    /// shared via the dag's per-DAG interners and counted there.
    pub fn approximate_bytes(&self) -> u64 {
        use std::mem::size_of;
        let mut b = (self.entries.capacity() * size_of::<FactoredSplit>()) as u64;
        for s in &self.entries {
            // Inline children carry NO heap backing; only spilled
            // (k>4) SmallVecs allocate. The inline payload is already counted
            // inside `size_of::<FactoredSplit>()` above.
            if s.children.spilled() {
                b += (s.children.capacity() * size_of::<FSubRef>()) as u64;
            }
        }
        b
    }
}

/// One subproblem in the factored DAG. One per unique MemoKey.
#[derive(Clone, Debug)]
pub struct FactoredSubproblem {
    /// Leaf option at this subproblem: (leaf_cost, leaf_info).
    pub leaf: Option<(f64, RLeafInfo)>,
    /// Optional MINORITY-class leaf option (count-all-leaves semantics, when
    /// the minority prediction also fits the budget); no current build path
    /// populates it. Carries `(minority_leaf_cost, minority_leaf_info)`. The
    /// majority option remains in `leaf`. `materialize_schedule` and
    /// `extract_at_cost` surface BOTH as separate cost-schedule entries.
    /// `None` under default semantics.
    ///
    /// Boxed (8 B inline instead of 40 B): the option is `None` on almost
    /// every vertex, so the payload lives in a side allocation only where
    /// present.
    pub leaf_minority: Option<Box<(f64, RLeafInfo)>>,
    /// Prefix-leaf marker, set when this subproblem was forced terminal by the
    /// `solve_depth` cap (i.e. there is unused depth budget below it). When
    /// present, `leaf` carries a placeholder majority-class prediction and
    /// callers may either honor that placeholder (current default) or run a
    /// per-leaf recursive fill to expand the subproblem into a sub-Rashomon set.
    ///
    /// Boxed (8 B inline instead of 192 B): after the completion search no vertex
    /// keeps a prefix-leaf marker, so the payload is a side allocation held
    /// only by the (Step-A / unfilled) vertices that carry one.
    pub prefix_leaf: Option<Box<PrefixLeafInfo>>,
    /// All splits available at this subproblem, in insertion order.
    ///
    /// During build/quotient/recompose this holds
    /// the inline FactoredSplits. After `promote_splits_to_pool` runs,
    /// this is `Vec::new()` and `split_ids` indexes the dag's
    /// `split_pool` for canonical, deduplicated storage. Readers should
    /// dispatch via `FactoredDag::iter_splits(subref)` (see
    /// `rashomon_dag.rs`) which handles both states transparently.
    pub splits: Vec<FactoredSplit>,
    /// Pool-resolved split handles. Populated by
    /// `promote_splits_to_pool`. Empty pre-promote.
    pub split_ids: Vec<SplitId>,
    /// Total tree count: 1 (if leaf present) + Σ split.tree_count.
    pub tree_count: u128,
    /// Minimum cost achievable across this subproblem's
    /// aggregate schedule. Used at extract time to short-circuit
    /// materialization when the requested scope can't accommodate any
    /// tree (`scope < min_obj` → empty schedule). `f64::INFINITY` when
    /// the subproblem has no admissible trees.
    pub min_obj: f64,
    /// The canonical cost schedule for this subproblem,
    /// captured at finalize/build time. Used by extract_at_cost and
    /// materialize_schedule so tree_count and enumeration are CONSISTENT:
    /// re-deriving the schedule at extract time with TIGHTER sibling LBs
    /// (sub.min_obj) than the build path used (l_vec eqp-floor) would make
    /// some counted trees unreachable.
    ///
    /// Arc-shared via schedule_interner: identical schedules across
    /// subproblems share one allocation. None until populated by
    /// finalize_node (lickety path) or build_factored_recursive
    /// (full path).
    pub cost_entries: Option<Arc<Vec<CostEntry>>>,
}

/// The complete factored Rashomon DAG.
///
/// Stores one FactoredSubproblem per unique memo key instead of one
/// RashomonNode per (memo_key, exact_cost). Eliminates the D^k cross-product
/// blowup by storing per-child cost schedules and computing combinations
/// via convolution.
#[derive(Clone, Debug)]
pub struct FactoredDag {
    pub subproblems: Vec<FactoredSubproblem>,
    pub root_ref: FSubRef,
    /// Root-level cost schedule: (cost, tree_count) sorted ascending.
    /// Used for top-level index navigation in get_tree_at_index.
    pub root_cost_schedule: Vec<CostEntry>,
    pub tree_count: u128,
    pub opt: f64,
    pub rashomon_bound: f64,
    pub theta: f64,
    pub reg: f64,
    pub delta_budget: DeltaBudget,
    /// Cost quantization precision used during construction.
    /// Higher = finer granularity. E.g. 1e9, 1e7, 1e5.
    pub cost_precision: f64,
    /// True when the counts were built in integer cost-key space
    /// (every `build_factored_dag`). Materialization and extraction then use
    /// the same keyed comparisons; false (lickety, IO files written before
    /// the flag was persisted) keeps the float comparisons.
    pub keyed_costs: bool,
    /// True when `keyed_costs` keys are exact: every leaf objective lies on
    /// the lattice `Z / cost_precision` (`CostLattice`, leaf-only objective)
    /// and `cost_precision` is that lattice's denominator, so key sums are
    /// exact tree costs and a bound `b` admits key `K` iff
    /// `K <= floor((b + 1e-9) * cost_precision)` (exact rational floor).
    /// False keeps the rounded 1e-9-grid scope key.
    pub exact_cost_keys: bool,
    /// Split-record pool (hash-cons) telemetry, snapshotted at
    /// build time before the SplitRecordPool is freed. Zero when the
    /// pool was empty (e.g., non-Rashomon mode or empty Rashomon set).
    pub pool_total_bytes: u64,
    pub pool_unique_split_count: u64,
    pub pool_dedup_hits: u64,
    pub pool_total_pushed: u64,
    /// Schedule-interner (Arc-sharing of cost schedules) telemetry.
    /// Number of unique schedule contents stored in the build-time
    /// `ScheduleInterner` (thread-local). Identical schedules across
    /// FactoredSplit children share via `Arc::clone`.
    pub schedule_intern_unique: u64,
    pub schedule_intern_attempts: u64,
    pub schedule_intern_hits: u64,
    pub schedule_intern_bytes: u64,
    /// MemoKey arena telemetry. Snapshotted from
    /// `SearchState.memo_key_arena` at DAG-build time.
    pub arena_total_pushed: u64,
    pub arena_unique: u64,
    pub arena_dedup_hits: u64,
    pub arena_bytes: u64,
    /// Lazy-materialized schedule cache.
    /// Key = `(sub_ref, quantize_scope(scope))`; value = the convolved
    /// cost schedule for that subproblem at that scope. Populated on
    /// demand by `rashomon_dag::materialize_schedule` during tree
    /// extraction / sampling. Empty after build.
    ///
    /// Wrapped in `Arc<DashMap<...>>` so cloning a `FactoredDag` (e.g.,
    /// for serialization or test fixtures) shares the cache; the
    /// `Default` impl on Arc<DashMap<...>> gives us a fresh empty cache
    /// per fresh DAG.
    pub schedule_cache: Arc<DashMap<(FSubRef, u64), Arc<Vec<CostEntry>>>>,
    /// Wide schedule cache (keyed exact-key DAGs): one schedule per vertex at
    /// the widest scope key it was asked for; narrower requests read a
    /// prefix (`rashomon_dag::materialize_schedule`). Replaces the
    /// per-(vertex, scope) entries of `schedule_cache` on those DAGs.
    pub wide_schedule_cache: Arc<WideScheduleCache>,
    /// Canonical, content-addressable pool of `FactoredSplit`s.
    /// Populated by `promote_splits_to_pool` after build/quotient/recompose.
    /// When non-empty, readers should index via `subproblem.split_ids[i]`
    /// → `dag.split_pool.entries[id as usize]`. The lookup table is
    /// dropped after construction; only `entries` is resident.
    pub split_pool: FactoredSplitPool,
    /// Per-DAG, deduplicated pool of distinct
    /// `SplitSpec`s. Each `FactoredSplit.spec_id` indexes here. Built via a
    /// `SplitSpecPool` interner during construction; tiny relative to the
    /// split pool. Readers use
    /// `dag.spec_of(split)`.
    pub spec_pool: Vec<SplitSpec>,
    /// Inline storage of leaf
    /// subproblems. Indexed by `leaf_id(cref)` when `is_leaf_ref(cref)`
    /// returns true. Keeps leaf entries out of the `subproblems` arena
    /// with a compact 32 B/leaf representation (8 B cost + 24 B RLeafInfo).
    pub inline_leaves: Vec<RLeafInfo>,
    pub inline_leaf_costs: Vec<f64>,
    /// Count-all-leaves extension (no current build path populates it): the
    /// inline-leaf emission sites can record a SECOND prediction
    /// (the minority class) for the same support at an HIGHER cost. The
    /// majority entry lives in `inline_leaves[id]` / `inline_leaf_costs[id]`
    /// at the same index, and the optional minority entry lives here at the
    /// same `id`. When None, the leaf has only the majority option (default).
    ///
    /// At extract time, `extract_at_cost` dispatches on the requested
    /// target_cost: if it matches `inline_leaf_costs[id]` (majority), emit
    /// `inline_leaves[id]`; otherwise if it matches the minority cost, emit
    /// the minority RLeafInfo. The schedule emitted at the parent split
    /// surfaces both costs as separate entries so the convolution picks up
    /// both options.
    pub inline_leaves_minority: Vec<Option<RLeafInfo>>,
    pub inline_leaf_minority_costs: Vec<Option<f64>>,
    /// Quotient telemetry: vertices whose 64-bit structural hash equalled
    /// that of an existing class but whose structure differed (kept
    /// separate by the equality check in `quotient_factored_dag_impl`).
    /// Summed over every quotient pass on this DAG (nested Step-B DAGs are
    /// added at the graft). Expected 0.
    pub n_quotient_hash_collisions: u64,
    /// True when a count addition or multiplication of the fit hit
    /// `u128::MAX` (`count_add` / `count_mul`) and a clamped count reached
    /// the stored DAG, so `tree_count` and the per-split / per-vertex
    /// counts are lower bounds clamped at `u128::MAX`, not exact. Set by
    /// the fit driver (global saturation-event counter grew, then
    /// `rashomon_dag::dag_has_saturated_count`); IO load sets it from
    /// `dag_has_saturated_count`. Not serialized.
    pub count_saturated: bool,
}

/// Wide schedule cache schedule cache of a keyed exact-key `FactoredDag`.
/// `entries[v] = (scope_k, schedule)`: v's schedule at scope key
/// `scope_k`, sorted ascending by cost key, so the schedule at any scope
/// key `s <= scope_k` is its prefix of entries with key `<= s`. `hints[v]`
/// (when set) is the widest scope key a count pass will ask v for, so v is
/// computed once at that key.
#[derive(Debug, Default)]
pub struct WideScheduleCache {
    pub entries: DashMap<FSubRef, (i64, Arc<Vec<CostEntry>>)>,
    pub hints: Vec<i64>,
}

impl FactoredDag {
    /// Resolve a `FactoredSplit`'s canonical `spec_id` to its
    /// `&SplitSpec` in this DAG's `spec_pool`.
    #[inline]
    pub fn spec_of<'a>(&'a self, split: &FactoredSplit) -> &'a SplitSpec {
        &self.spec_pool[split.spec_id as usize]
    }
}

// ---------------------------------------------------------------------------
// Lazy factored DAG construction
// ---------------------------------------------------------------------------
//
// Pending snapshots own the search inputs. Materialization consumes them,
// allowing the memo and split pool to be released during DAG construction.

/// Configuration for the prefix-leaf nested Rashomon fill (Step B), carried
/// alongside `SearchSnapshot` so the (recursive_fill) path also defers
/// to ensure_built. Fields mirror the inline fill block that previously
/// lived in `lib.rs::solve_rashomon`.
#[derive(Clone)]
pub struct PrefixLeafFillConfig {
    pub store: Arc<crate::dataset_store::DatasetStore>,
    pub leaf_base_config: SolverConfig,
    pub reg: f64,
    pub leaf_reg_cap: f64,
    pub leaf_short_circuit_eps: f64,
    pub n_data: usize,
    pub fill_parallel: bool,
    pub verbose: bool,
    /// Completion search (Step B): end of the Phase-2 time budget (Phase-2 start +
    /// `time_limit`). Each MiniRESPLINTER solve gets the remaining time;
    /// solves not started by then count as timed out. None = no limit.
    pub deadline: Option<std::time::Instant>,
    /// Shared Step B: the Step-A solver's completion-lb caches
    /// (`solver::RashLbCache`), moved (`take`, not cloned) into the first
    /// shared Step-B solver when compatible (same store, Select ub and
    /// normalisation).
    pub rash_lb_cache: Option<Arc<std::sync::Mutex<Option<crate::solver::RashLbCache>>>>,
}

/// Leaf-only objective of a Phase-2 solve: a tree costs
/// `err / n_total + leaf_reg * leaves` with integer `err` (split_reg = 0,
/// path_penalty = 0). With `leaf_reg = p/q` exactly, every cost lies on
/// `Z / (n_total * q)`, which `build_factored_dag` uses for exact keys
/// (integer cost keys) after checking every leaf objective in the memo.
#[derive(Clone, Copy, Debug)]
pub struct CostLattice {
    pub n_total: usize,
    pub leaf_reg: f64,
}

/// The solver's memo DashMap, moved (not re-hashed) into a `SearchSnapshot`.
pub type SnapshotMemo = dashmap::ReadOnlyView<MemoKey, SubProblem, rustc_hash::FxBuildHasher>;

/// Read-only memo access for the DAG builders: a std `HashMap` (public
/// entry points) or a `SnapshotMemo`. Only keyed lookups and an
/// order-independent `all`, so the container's iteration order is unobservable.
pub trait MemoLookup {
    fn get(&self, key: &MemoKey) -> Option<&SubProblem>;
    fn all_values(&self, f: &mut dyn FnMut(&SubProblem) -> bool) -> bool;
}

impl<S: std::hash::BuildHasher> MemoLookup for HashMap<MemoKey, SubProblem, S> {
    fn get(&self, key: &MemoKey) -> Option<&SubProblem> {
        HashMap::get(self, key)
    }
    fn all_values(&self, f: &mut dyn FnMut(&SubProblem) -> bool) -> bool {
        self.values().all(f)
    }
}

impl MemoLookup for SnapshotMemo {
    fn get(&self, key: &MemoKey) -> Option<&SubProblem> {
        dashmap::ReadOnlyView::get(self, key)
    }
    fn all_values(&self, f: &mut dyn FnMut(&SubProblem) -> bool) -> bool {
        self.values().all(f)
    }
}

/// Snapshot of the solver state needed to build a FactoredDag on demand.
/// Owns its inputs so the build can drop the memo before materialization and
/// release the live search state inside the build path.
pub struct SearchSnapshot {
    pub memo: SnapshotMemo,
    pub pool: SplitRecordPool,
    pub arena: Arc<MemoKeyArena>,
    pub root_key: MemoKey,
    pub rashomon_bound: f64,
    pub opt: f64,
    pub theta: f64,
    pub reg: f64,
    pub delta_budget: DeltaBudget,
    pub cost_precision: f64,
    /// Exact cost lattice of the solve's objective (None: rounded keys).
    pub cost_lattice: Option<CostLattice>,
    pub solve_depth: i32,
    pub total_max_depth: i32,
    /// When Some, `build_dag_from_snapshot` runs prefix-leaf nested fill
    /// after build+quotient. None → no prefix-leaf fill; matches
    /// `recursive_fill=false` Python kwarg.
    pub prefix_leaf_fill: Option<PrefixLeafFillConfig>,
}

/// Two-state container for a Rashomon-set DAG.
///
/// `Pending`: snapshot owned, DAG not yet built. First call to `dag()` /
///   `dag_mut()` consumes the snapshot and produces `Built(dag)`.
/// `Built`: FactoredDag is resident; subsequent accesses are O(1).
///
/// The `Pending` variant uses `Option<Box<...>>` so the snapshot can be
/// safely taken inside `&mut self` without a sentinel value.
pub enum RashomonSet {
    Pending {
        snapshot: Option<Box<SearchSnapshot>>,
    },
    Built(Box<FactoredDag>),
}
