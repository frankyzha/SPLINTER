use std::collections::hash_map::DefaultHasher;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use dashmap::DashMap;

use ndarray::Array2;
use rayon::prelude::*;

use crate::candidates::{
    compare_split_candidates, dp_threshold_scan, generate_candidates_for_support,
    generate_candidates_for_support_rash, score_candidate,
};
use crate::dataset_store::DatasetStore;
use crate::scratch::{ScratchBank, SolverScratch};
use crate::shared_bounds::SharedBoundsTable;
use crate::tree::TreeNode;
use crate::types::{
    self, CandidateOrder, DpCriterion, FeatureMask, MemoKey, MemoKeyArena, SolverConfig,
    SolverProfile, SplitCandidate, SplitRecord, SplitStep, SubProblem, SupportMask, SupportSig,
    WorkItem, MAX_CLASSES, MAX_PROFILE_DEPTH,
};

/// Interval lb: interval-DP completion lb at remaining
/// depth 2 (`interval_lb`).
mod interval_lb;
pub use interval_lb::{rash_iv_dyadic_debug, rash_iv_stats};
/// `lb_threads`: intra-cell parallel completion-lb prefetch (`rash_par`).
mod rash_par;
pub use rash_par::{rash_lb_delta_merge, rash_par_stats, RashLevelBase};

/// One depth level of the Step A worklist: (key, widest requested
/// scope) in first-request order.
#[derive(Default)]
struct RashLevel {
    items: Vec<(MemoKey, f64)>,
    pos: HashMap<MemoKey, usize>,
}

impl RashLevel {
    fn push(&mut self, key: MemoKey, scope: f64) {
        match self.pos.get(&key) {
            Some(&i) => {
                if scope > self.items[i].1 {
                    self.items[i].1 = scope;
                }
            }
            None => {
                self.pos.insert(key.clone(), self.items.len());
                self.items.push((key, scope));
            }
        }
    }
}

pub struct SolveResult {
    pub objective: f64,
    pub lowerbound: f64,
    pub is_optimal: bool,
    pub certified_optimal: bool,
    pub search_converged: bool,
    pub certification_mode: String,
    pub certification_reason: String,
    pub unsafe_heuristics_used: bool,
    pub timed_out: bool,
    pub termination_reason: String,
    pub final_gap: f64,
    pub n_subproblems: usize,
    pub n_explored: usize,
    pub elapsed: f64,
}

/// A root-level candidate discovered by the parallel root solver.
/// Used to pre-seed Phase 2's sequential solver with good root splits.
pub struct RootCandidate {
    pub objective: f64,
    pub tree: TreeNode,
}

#[derive(Hash, Eq, PartialEq, Clone)]
pub struct GreedyCacheKey {
    indices_key: MemoKey,
    remaining_depth: i32,
    used_features_hash: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct FlatBins {
    offsets: Vec<usize>,
    flat: Vec<u32>,
}

impl FlatBins {
    fn len(&self) -> usize {
        self.offsets.len().saturating_sub(1)
    }

    fn bin_slice(&self, bin_idx: usize) -> &[u32] {
        &self.flat[self.offsets[bin_idx]..self.offsets[bin_idx + 1]]
    }

    fn bin_len(&self, bin_idx: usize) -> usize {
        self.offsets[bin_idx + 1] - self.offsets[bin_idx]
    }

    fn iter(&self) -> impl Iterator<Item = &[u32]> {
        (0..self.len()).map(|bin_idx| self.bin_slice(bin_idx))
    }

    fn to_nested(&self) -> Vec<Vec<u32>> {
        self.iter().map(|bin| bin.to_vec()).collect()
    }
}

#[derive(Clone, Copy, Debug)]
enum BranchBoundResult {
    Feasible { cost: f64, heuristic: bool },
    Infeasible { floor: f64 },
}

pub struct SearchState {
    // Immutable after init
    pub store: Arc<DatasetStore>,
    pub root_support: SupportMask,
    pub n_data: usize,
    pub n_total: usize,
    pub config: SolverConfig,
    pub shared_bounds: Option<Arc<SharedBoundsTable>>,
    pub weight_total: f64,

    // Shared mutable (graph + queue)
    /// Fx-hashed: `MemoKey::hash` already ends on the avalanched support
    /// fingerprint, and nothing observable depends on iteration order.
    pub memo: DashMap<MemoKey, SubProblem, rustc_hash::FxBuildHasher>,
    /// Sparse global bound table (follows the TreeFARMS bound list, Xin et
    /// al., 2022).
    ///
    /// Keyed by `MemoKey`; value is a per-node `Vec<BoundEntry>`, one entry
    /// per attempted feature: (feature, lb, ub) tuples, extended to carry
    /// contextual state + pool_idx because our solver
    /// maintains contextual bounds per (parent context, split) and dedups
    /// SplitRecords through a shared pool.
    ///
    /// Replaces the per-SubProblem `materialized_indices` /
    /// `materialized_contextual_lbs` / `materialized_contextually_dead` /
    /// `rashomon_indices` vectors and the contextual sidecars previously
    /// living on the SplitRecord pool entry.
    ///
    /// Mutations to a single feature's `(lower_bound, upper_bound, pool_idx)`
    /// MUST be atomic — pool dedup re-keys on `split_upper_bits`, so a
    /// refreshed UB produces a different pool_idx that must replace the old
    /// one in lockstep. Use `DashMap::entry(key).or_insert_with(...).iter_mut()`
    /// inside one access scope so the entry's write lock is held for the
    /// whole find-and-update sequence.
    pub bound_table: DashMap<MemoKey, Vec<crate::types::BoundEntry>>,
    /// Current-best-split lookup keyed by MemoKey.
    ///
    /// Stores only the `split_id` (cf. `BranchAndBoundSolver::split_record_id`);
    /// the full SplitRecord lives once in `WorkerState.split_pool` and is
    /// retrieved via the `bound_table` entry's `pool_idx`. Replaces the
    /// per-SubProblem `best_split: Option<SplitRecord>` field.
    pub best_split_table: DashMap<MemoKey, u64>,
    /// Canonical `memo_key_arena` ids of every registered child key; read
    /// only for its size (`parent_map_size`).
    pub parent_map: dashmap::DashSet<u32, rustc_hash::FxBuildHasher>,
    pub queue: Mutex<BinaryHeap<WorkItem>>,
    pub root_key: Mutex<MemoKey>,
    /// Override for config.materialize_top_n during escalation.
    pub materialize_top_n: AtomicUsize,

    /// Shared MemoKey arena. Arc-wrapped so all parallel
    /// workers share a single index space. Lock-free reads via boxcar;
    /// writes go through the arena's internal DashMap dedup.
    pub memo_key_arena: Arc<MemoKeyArena>,

    // Diagnostic (all interior-mutable for Arc safety)
    pub n_subproblems: AtomicUsize,
    pub n_explored: AtomicUsize,
    pub timed_out: AtomicBool,
    pub counter: AtomicU64,
    pub start_time: Mutex<Option<Instant>>,
    pub certification_issues: Mutex<Vec<String>>,
    pub used_ub_margin_prune: AtomicBool,
    pub used_bfs_partial_materialization: AtomicBool,
    pub used_greedy_boundary_guidance: AtomicBool,
    pub used_closure: AtomicBool,
    /// ub_hint of the Step A Select (f64 bits; NaN = unset, use
    /// `config.rashomon_bound`). Step-B mini solves set the outer β.
    pub rashomon_select_ub: AtomicU64,
}

impl SearchState {
    // -----------------------------------------------------------------------
    // bound_table helpers (TreeFARMS bound list).
    //
    // The bound_table is the global sparse store of per-(node, feature) split
    // bookkeeping. These helpers wrap the DashMap-entry write-lock dance so
    // every refresh of `(lower_bound, upper_bound, pool_idx)` is atomic at
    // the (key, feature_id) granularity — see `BoundEntry` doc-comment for
    // why pool_idx must move with the bounds.
    // -----------------------------------------------------------------------

    /// Insert or refresh a bound entry for `(key, feature_id)`. Atomic under
    /// the DashMap entry write-lock. If the entry exists, `(pool_idx,
    /// lower_bound, upper_bound)` are updated together. Contextual sidecars
    /// (`contextual_lowerbound`, `is_contextually_dead`) and flags are
    /// preserved on update; reset on first insert.
    pub fn bound_upsert(
        &self,
        key: &crate::types::MemoKey,
        feature_id: u16,
        pool_idx: u32,
        lower_bound: f64,
        upper_bound: f64,
    ) {
        let mut bounds = self.bound_table.entry(key.clone()).or_insert_with(Vec::new);
        if let Some(entry) = bounds.iter_mut().find(|e| e.feature_id == feature_id) {
            entry.pool_idx = pool_idx;
            entry.lower_bound = lower_bound;
            entry.upper_bound = upper_bound;
        } else {
            bounds.push(crate::types::BoundEntry::new(
                feature_id,
                pool_idx,
                lower_bound,
                upper_bound,
            ));
        }
    }

    /// Same as `bound_upsert` but also sets the `MATERIALIZED` flag —
    /// equivalent to "this split has been admitted to the per-node
    /// materialized set".
    /// Use from BnB-UB-tracking code paths (`upsert_materialized_split`,
    /// `process_bound_update` rebuild, `explore_node` finalization).
    pub fn bound_upsert_materialized(
        &self,
        key: &crate::types::MemoKey,
        feature_id: u16,
        pool_idx: u32,
        lower_bound: f64,
        upper_bound: f64,
    ) {
        let mut bounds = self.bound_table.entry(key.clone()).or_insert_with(Vec::new);
        if let Some(entry) = bounds.iter_mut().find(|e| e.feature_id == feature_id) {
            entry.pool_idx = pool_idx;
            entry.lower_bound = lower_bound;
            entry.upper_bound = upper_bound;
            entry.set_materialized(true);
        } else {
            let mut e =
                crate::types::BoundEntry::new(feature_id, pool_idx, lower_bound, upper_bound);
            e.set_materialized(true);
            bounds.push(e);
        }
    }

    /// Set or clear the Rashomon-membership flag on an existing bound entry.
    /// Returns false if no entry exists for `(key, feature_id)` — the caller
    /// must call `bound_upsert` first to install the bounds.
    pub fn bound_set_rashomon_flag(
        &self,
        key: &crate::types::MemoKey,
        feature_id: u16,
        on: bool,
    ) -> bool {
        if let Some(mut bounds) = self.bound_table.get_mut(key) {
            if let Some(entry) = bounds.iter_mut().find(|e| e.feature_id == feature_id) {
                entry.set_rashomon(on);
                return true;
            }
        }
        false
    }

    /// Update contextual sidecars on an existing bound entry. Returns false
    /// if no entry exists. Contextual state is per-(parent context, split)
    /// — same per-context semantics as the old
    /// `materialized_contextual_lbs` / `materialized_contextually_dead`
    /// vectors.
    pub fn bound_set_contextual(
        &self,
        key: &crate::types::MemoKey,
        feature_id: u16,
        contextual_lb: f64,
        contextually_dead: bool,
    ) -> bool {
        if let Some(mut bounds) = self.bound_table.get_mut(key) {
            if let Some(entry) = bounds.iter_mut().find(|e| e.feature_id == feature_id) {
                entry.contextual_lowerbound = contextual_lb;
                entry.set_contextually_dead(contextually_dead);
                return true;
            }
        }
        false
    }

    /// Read a single bound entry by `(key, feature_id)`. Returns a copy
    /// (BoundEntry is Copy) so the caller doesn't hold the DashMap lock.
    pub fn bound_get(
        &self,
        key: &crate::types::MemoKey,
        feature_id: u16,
    ) -> Option<crate::types::BoundEntry> {
        self.bound_table
            .get(key)
            .and_then(|bounds| bounds.iter().find(|e| e.feature_id == feature_id).copied())
    }

    /// Snapshot all bound entries at a node. Returns an owned Vec so the
    /// DashMap lock is released before the caller iterates. Use for
    /// read-side scans (DAG build, telemetry) where holding the lock
    /// during downstream work would block writers.
    pub fn bound_features_at(&self, key: &crate::types::MemoKey) -> Vec<crate::types::BoundEntry> {
        self.bound_table
            .get(key)
            .map(|bounds| bounds.clone())
            .unwrap_or_default()
    }

    /// Set the best-split lookup for a node. Stores only the `split_id`
    /// (cf. `BranchAndBoundSolver::split_record_id`); the SplitRecord lives
    /// in `WorkerState.split_pool` and is retrievable via the matching
    /// `bound_table` entry's `pool_idx`.
    pub fn set_best_split_id(&self, key: &crate::types::MemoKey, split_id: u64) {
        self.best_split_table.insert(key.clone(), split_id);
    }

    /// Clear the best-split lookup for a node.
    pub fn clear_best_split_id(&self, key: &crate::types::MemoKey) {
        self.best_split_table.remove(key);
    }
}

pub struct WorkerState {
    pub feature_k_hits: HashMap<(usize, usize), u64>,
    pub killer_moves: HashMap<i32, (usize, usize, Arc<[f64]>)>,
    pub greedy_score_cache: HashMap<GreedyCacheKey, f64>,
    pub scratch_bank: ScratchBank,
    pub solver_scratch: SolverScratch,
    pub similar_lb: HashMap<SupportSig, f64>,
    pub similar_ub: HashMap<SupportSig, f64>,
    pub profile: SolverProfile,
    /// Per-solver pool of SplitRecord storage. SubProblem.materialized_indices
    /// and rashomon_indices reference into this pool: one shared pool plus
    /// Vec<u32> indices instead of a `Vec<SplitRecord>` per SubProblem.
    /// Single-mutator (Phase-2 BFS is single-threaded per root); no locks.
    pub split_pool: types::SplitRecordPool,
    /// Thread-local pool for `Vec<SplitCandidate>`.
    /// Reused across `explore_node` / `closure_build` calls via
    /// `std::mem::take` (rent at call start, restore at end). Eliminates
    /// per-node alloc/free of the candidate output Vec.
    /// Per-solver, single-mutator; no locks needed.
    pub candidate_pool: Vec<crate::types::SplitCandidate>,
    /// Exact completion lb: memo of `bounded_completion_min`, keyed by
    /// (support, remaining completion depth).
    pub rash_lb_memo: RashFxMap<(SupportMask, i32), RashLbEntry>,
    /// Remaining-depth-1 entries of `rash_lb_memo` (lb memo cap).
    pub rash_lb_memo_d1: usize,
    /// Structural depth-1 memo: remaining-depth-1 entries keyed structurally
    /// (`RashD1Key`), with their parent interner.
    pub rash_lb_d1s: RashD1Struct,
    /// Exact completion lb: the lb-solver's Select output per support
    /// (`rashomon_select_at(support, used, None)`, sorted by `split_ub`,
    /// splits with <= 1 non-empty child dropped).
    pub rash_lb_select: RashFxMap<SupportMask, Arc<[RashLbSplit]>>,
    /// Shared Select pool: Select pool per support for the explore and the
    /// completion-lb DFS (`rash_shared_pool`); same lifetime as the caches
    /// above.
    pub rash_select_shared: RashFxMap<SupportMask, RashSharedPool>,
    /// Select ub bits (`rashomon_select_ub`) the two caches above hold
    /// results for; None = empty. A different ub clears them
    /// (`rash_lb_sync_ub`): the pool depends on it.
    pub rash_lb_ub_bits: Option<u64>,
    /// Split records currently held in `rash_lb_select` (for
    /// `SolverProfile::lb_select_cache_records`).
    pub rash_lb_select_records: u64,
    /// Preconditions of the fast completion-lb paths (`rash_lb_gate`),
    /// computed once per solver: (mask partitions equal the row partitions,
    /// the depth-1 closed form applies). None = not computed yet.
    pub rash_lb_gate: Option<(bool, bool)>,
    /// Depth-1 row walk: per-feature row -> bin rank table of the store
    /// (`rash_lb_rank_build`), built with the gate when d1_ok. None = not
    /// built (the closed form then builds one on demand if forced).
    pub rash_lb_rank: Option<Arc<RashLbRank>>,
    /// Interval lb: per-feature φ totals of supports searched above
    /// their bound (`interval_lb::RashIvCache`, capped).
    pub rash_iv_cache: interval_lb::RashIvCache,
    /// `lb_threads` pass-1 workers only: the main solver's frozen
    /// completion-lb caches, read after this worker's own (the delta).
    pub rash_base: Option<Arc<rash_par::RashLbBase>>,
    /// `stepb_batches`: the carried completion-lb caches of a Step-B
    /// level, frozen and shared by its batch solvers (and their pass-1
    /// workers), read after `rash_base`; never written. None = no level
    /// base (`attach_rash_level_base`).
    pub rash_level_base: Option<Arc<rash_par::RashLevelBase>>,
    /// `stepb_batches`: lb threads of this solver's pass 1; 0 = the
    /// config value (`rash_lb_threads_eff`).
    pub rash_lb_threads: usize,
    /// `lb_threads` pass-1 workers only: `check_time` stores a
    /// time-out here instead of `search.timed_out`.
    pub rash_time_sink: Option<Arc<AtomicBool>>,
    /// (start, time_limit) of the solve: `check_time` without the
    /// `start_time` mutex when set (same test).
    pub rash_deadline: Option<(Instant, f64)>,
    /// `lb_threads` pass-1 workers only: the lb Selects' config
    /// (`search.config` with `sequential_candidate_gen`; same output).
    pub rash_select_config: Option<Arc<SolverConfig>>,
    /// `lb_threads`: the pass-1 workers of this (main) solver, built on
    /// first use.
    pub rash_par: Option<Box<rash_par::RashParState>>,
    /// Divisor of the completion-lb memo caps (lb memo cap and the
    /// structural map): 1, or T on a pass-1 worker (T deltas resident).
    pub rash_lb_cap_div: usize,
}

/// HashMap of the completion-lb caches: rustc-hash's Fx hasher over the
/// key's (n_samples, fingerprint[, depth]) (`SupportMask`'s `Hash`; the
/// fingerprint is already a full avalanche mix). No use of these maps
/// depends on their iteration order (only order-free sums over values), so
/// the hasher changes no result.
pub type RashFxMap<K, V> = HashMap<K, V, rustc_hash::FxBuildHasher>;

/// Depth-1 row walk: `rank[f][i]` = number of store thresholds of feature
/// f strictly below row i's value = the bin of row i under the le-mask
/// convention (`x <= t` left; row i lies in le-mask b iff rank <= b).
/// `sum_tw` = Σ_f T_f · words (the mask path's word count per call),
/// `f_active` = features with T_f > 0.
#[derive(Debug)]
pub struct RashLbRank {
    pub rank: Vec<Box<[u16]>>,
    pub sum_tw: usize,
    pub f_active: usize,
}

/// Per-thread scratch of `rash_depth1_min_closed` (no allocation per call).
#[derive(Default)]
struct RashD1Scratch {
    cum: Vec<i64>,
    cum_n: Vec<i64>,
    /// Per feature: offset of its boundaries in `cum_n` (`cum`: × nc).
    off: Vec<usize>,
    fine3: Vec<f64>,
    prev: Vec<f64>,
    next: Vec<f64>,
    idx: Vec<u32>,
    lab: Vec<u8>,
    hist: Vec<i64>,
    int: RashD1IntScratch,
}

/// Integer-first depth-1 DP scratch of `rash_d1_dp_int` (per thread): the bin cost
/// table `tab[m] = m / n_total + leaf_reg` for the key (n_total bits,
/// leaf_reg bits), the integer layer / suffix minima `l` / `suf`, the
/// non-class prefix counts `q`, and the calls answered / fallen back since
/// the last `rash_d1_intdp_counts_take`.
#[derive(Default)]
struct RashD1IntScratch {
    tab: Vec<f64>,
    tab_key: (u64, u64),
    l: Vec<i64>,
    suf: Vec<i64>,
    q: Vec<i64>,
    n_int: u64,
    n_fallback: u64,
}

/// Integer-first depth-1 DP: (calls answered by the integer-first DP, calls that
/// fell back to the reference DP) on this thread since the last take.
pub fn rash_d1_intdp_counts_take() -> (u64, u64) {
    RASH_D1_SCRATCH.with(|cell| {
        let mut g = cell.borrow_mut();
        let r = (g.int.n_int, g.int.n_fallback);
        g.int.n_int = 0;
        g.int.n_fallback = 0;
        r
    })
}

thread_local! {
    /// Test hook of `set_rash_d1_reference_dp`.
    static RASH_D1_REFERENCE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Test hook (an oracle for the integer-first DP): `true` makes this
/// thread's depth-1 closed form take the reference float DP
/// (`rash_d1_dp`) and the generic prefix counting instead of the
/// integer-first DP and the 2-/3-class sweep bounds; the values are the
/// same bitwise.
#[doc(hidden)]
pub fn set_rash_d1_reference_dp(on: bool) {
    RASH_D1_REFERENCE.with(|c| c.set(on));
}

/// The integer-first depth-1 DP is used (always, unless a test forces the
/// reference, `set_rash_d1_reference_dp`).
#[inline]
pub(crate) fn rash_d1_intdp_on() -> bool {
    !RASH_D1_REFERENCE.with(|c| c.get())
}

/// Misclassification `n - max_cl c_cl` of the rows between boundaries a < b
/// (`cum` holds NC class counts per boundary).
#[inline(always)]
fn rash_d1_m<const NC: usize>(cum: &[i64], cum_n: &[i64], a: usize, b: usize) -> usize {
    let n = cum_n[b] - cum_n[a];
    let mut mx = cum[b * NC] - cum[a * NC];
    for cl in 1..NC {
        mx = mx.max(cum[b * NC + cl] - cum[a * NC + cl]);
    }
    (n - mx) as usize
}

/// Depth-1 sweep state of one completion-lb scan at remaining depth 2
/// (`BranchAndBoundSolver::rash_d1_sweep_child`): the parent support, its
/// rows and per-feature class histogram by rank (built on the first swept
/// child), and one group per split feature.
struct RashD1Sweep {
    parent: SupportMask,
    rank: Arc<RashLbRank>,
    rows: Vec<u32>,
    parent_hist: Option<Box<[i32]>>,
    /// Per feature g: offset of its (T_g + 1) · nc histogram cells
    /// (rank-major, class-minor); `hlen` cells in all.
    hoff: Vec<usize>,
    hlen: usize,
    groups: Vec<RashD1SweepGroup>,
}

/// One (parent, split feature) group of a `RashD1Sweep`: the estimated
/// direct counting work of its closed-form misses so far (the state is
/// built once it reaches the build work) and the state.
#[derive(Default)]
struct RashD1SweepGroup {
    /// Estimated counting work of the children counted directly so far.
    work: f64,
    state: Option<RashD1SweepState>,
}

/// The parent's rows bucketed by the split feature's rank (`rows`, bucket
/// c at `start[c]..start[c + 1]`) and the prefix histograms cached at
/// rank boundaries c in 0..=T_f + 1 (rows of rank < c).
struct RashD1SweepState {
    start: Vec<usize>,
    rows: Vec<u32>,
    cache: Vec<Option<Box<[i32]>>>,
}

thread_local! {
    static RASH_D1_SCRATCH: std::cell::RefCell<RashD1Scratch> =
        std::cell::RefCell::new(RashD1Scratch::default());
}

/// Shared Step B: the completion-lb caches of one solver, moved out
/// (`take_rash_lb_cache`) to seed another solver over the same store
/// (`seed_rash_lb_cache`). Valid only for the same Select pool and
/// costs: `key` = (store address, Select ub bits, n_total, reg, reg_alpha,
/// path_penalty, beam_width, max_k, n_classes).
#[derive(Clone, Debug, Default)]
pub struct RashLbCache {
    pub key: Option<(usize, u64, usize, u64, u64, u64, usize, usize, usize)>,
    pub memo: RashFxMap<(SupportMask, i32), RashLbEntry>,
    pub select: RashFxMap<SupportMask, Arc<[RashLbSplit]>>,
    /// Shared Select pool pools (`WorkerState::rash_select_shared`),
    /// installed only when `select_fp` (`rash_select_config_fp`: every
    /// config field Select reads) also matches.
    pub shared: RashFxMap<SupportMask, RashSharedPool>,
    pub select_fp: u64,
    /// Structural depth-1 memo structural depth-1 memo (`WorkerState::rash_lb_d1s`),
    /// installed with `memo`.
    pub d1s: RashD1Struct,
}

/// One split of the lb-solver's Select pool (the fields it needs).
#[derive(Clone, Debug)]
pub struct RashLbSplit {
    pub feature_index: usize,
    pub thresholds: Arc<[f64]>,
    pub group_map: Option<Box<[usize]>>,
    /// Arity of the row partition (`thresholds.len() + 1` bins, or the
    /// number of groups of `group_map`): the number of children.
    pub k: usize,
    /// Mask-based lb path: leaf cost (`leaf_objective_from_counts`) and row
    /// count of every child, from Select's `child_counts` / `child_n`; empty
    /// when the cache was built without it (row scan then).
    pub child_leaf: Box<[f64]>,
    pub child_size: Box<[usize]>,
}

/// Shared Select pool: one record of a shared Select pool: one distinct
/// split of Select's output with >= 2 non-empty children (an exact repeat
/// — same identity, arity, counts and split_ub as an earlier record — is
/// dropped: the explore and the completion-lb view both skip it).
#[derive(Clone, Debug)]
pub struct RashPoolRec {
    pub feature_index: u32,
    /// Select's arity (`SplitCandidate::k`, used by the explore's arity bound).
    pub sel_k: u32,
    /// Explore identity id (feature, threshold bits, group_map).
    pub ident: u32,
    pub thresholds: Arc<[f64]>,
    pub group_map: Option<Box<[usize]>>,
    /// Number of children (`child_n.len()`, equal to the partition arity).
    pub data_children: u32,
    /// Per child, `n_classes + 2` words: leaf cost
    /// (`leaf_objective_from_counts`, f64 bits), `child_n`, `child_counts`.
    pub data: Box<[i64]>,
}

/// Shared Select pool: one support's Select pool, shared by the
/// Step-A pass (Step A and the Step-B solves) and the completion-lb
/// DFS. Select's pool depends only on the support, the solver config and
/// the Select ub (the depth argument only gates the unsafe feature filter,
/// off whenever the cache is used), so one call serves every visit.
#[derive(Clone, Debug)]
pub struct RashSharedPool {
    /// The completion-lb view first (`n_lb` records in its order: stable by
    /// `split_ub`, first per identity), then the other records.
    pub recs: RashPoolStore,
    pub n_lb: usize,
    /// Select's pool order over `recs` (what the explore iterates).
    pub explore: Arc<[u32]>,
    /// Number of distinct identities.
    pub n_ident: u32,
    /// (Select arity, entries) over Select's whole output (repeats and
    /// <= 1-non-empty splits included): the explore's candidate count.
    pub k_hist: Arc<[(usize, u32)]>,
    /// `CandGenStats::n_available` of the Select call.
    pub n_available: u64,
}

/// Pool arena: header of one compact pool record (`RashPoolArena`).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct RashPoolHdr {
    /// Explore identity id (as `RashPoolRec::ident`).
    pub ident: u32,
    /// Offset of the record's `n_children × (1 + nc)` cells in `cells`.
    pub cells: u32,
    /// Index into `thr`.
    pub thr: u32,
    /// Index into `gm`; `u32::MAX` = no group_map.
    pub gm: u32,
    pub feature: u16,
    /// Select's arity (`SplitCandidate::k`).
    pub sel_k: u8,
    pub n_children: u8,
}

/// Pool arena: a shared Select pool's records in one arena: per record
/// a header and, per child, `child_n` then its `nc` class counts as u32
/// (the leaf cost is recomputed from them: `leaf_objective_from_counts` on
/// the same integers, the same bits). Thresholds are the pool's distinct
/// `Arc`s (the very ones Select returned), group maps the distinct maps.
#[derive(Debug)]
pub struct RashPoolArena {
    pub hdr: Box<[RashPoolHdr]>,
    pub cells: Box<[u32]>,
    pub thr: Box<[Arc<[f64]>]>,
    pub gm: Box<[Box<[usize]>]>,
    pub nc: u8,
}

/// A record of a `RashPoolArena`.
#[derive(Clone, Copy)]
pub struct RashPoolRecRef<'a> {
    pub a: &'a RashPoolArena,
    pub i: usize,
}

impl RashPoolArena {
    pub fn rec(&self, i: usize) -> RashPoolRecRef<'_> {
        RashPoolRecRef { a: self, i }
    }
    pub fn len(&self) -> usize {
        self.hdr.len()
    }
    pub fn is_empty(&self) -> bool {
        self.hdr.is_empty()
    }
}

impl<'a> RashPoolRecRef<'a> {
    #[inline]
    fn h(&self) -> &'a RashPoolHdr {
        &self.a.hdr[self.i]
    }
    /// Child `c`'s cells: `child_n`, then the class counts.
    #[inline]
    fn cell(&self, c: usize) -> &'a [u32] {
        let st = 1 + self.a.nc as usize;
        let o = self.h().cells as usize + c * st;
        &self.a.cells[o..o + st]
    }
    #[inline]
    fn n_children(&self) -> usize {
        self.h().n_children as usize
    }
    #[inline]
    fn thr_arc(&self) -> &'a Arc<[f64]> {
        &self.a.thr[self.h().thr as usize]
    }
    #[inline]
    fn gm(&self) -> Option<&'a [usize]> {
        let g = self.h().gm;
        if g == u32::MAX {
            None
        } else {
            Some(&self.a.gm[g as usize])
        }
    }
    /// Child `c`'s class counts widened into `buf[..nc]`.
    #[inline]
    fn counts_into<'b>(&self, c: usize, buf: &'b mut [i64; MAX_CLASSES]) -> &'b [i64] {
        let nc = self.a.nc as usize;
        let cell = self.cell(c);
        for j in 0..nc {
            buf[j] = cell[1 + j] as i64;
        }
        &buf[..nc]
    }
}

/// Shared Select pool records: the plain layout or the compact arena
/// (`RashPoolArena`); the same records in the same order.
#[derive(Clone, Debug)]
pub enum RashPoolStore {
    Legacy(Arc<[RashPoolRec]>),
    Arena(Arc<RashPoolArena>),
}

impl RashPoolStore {
    /// Explore identity id of record `i`.
    #[inline]
    pub fn ident(&self, i: usize) -> u32 {
        match self {
            RashPoolStore::Legacy(r) => r[i].ident,
            RashPoolStore::Arena(a) => a.hdr[i].ident,
        }
    }
    pub fn len(&self) -> usize {
        match self {
            RashPoolStore::Legacy(r) => r.len(),
            RashPoolStore::Arena(a) => a.len(),
        }
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// A candidate of the completion-lb DFS (`rash_lb_scan`): a legacy cache
/// record (`RashLbSplit`) or a shared Select pool pool record.
pub trait RashLbRec {
    fn feature_index(&self) -> usize;
    fn thresholds(&self) -> &[f64];
    fn group_map(&self) -> Option<&[usize]>;
    /// Arity of the row partition (number of children).
    fn k(&self) -> usize;
    /// The cached per-child leaf costs and sizes into the buffers; false
    /// when the record holds none. `leaf_of(counts, n_weighted)` is the
    /// solver's `leaf_objective_from_counts`, for records that keep only
    /// the counts (pool arena).
    fn cached_children(
        &self,
        leaf_of: &dyn Fn(&[i64], f64) -> f64,
        leaf: &mut smallvec::SmallVec<[f64; 8]>,
        size: &mut smallvec::SmallVec<[usize; 8]>,
    ) -> bool;
}

impl RashLbRec for RashLbSplit {
    fn feature_index(&self) -> usize {
        self.feature_index
    }
    fn thresholds(&self) -> &[f64] {
        &self.thresholds
    }
    fn group_map(&self) -> Option<&[usize]> {
        self.group_map.as_deref()
    }
    fn k(&self) -> usize {
        self.k
    }
    fn cached_children(
        &self,
        _leaf_of: &dyn Fn(&[i64], f64) -> f64,
        leaf: &mut smallvec::SmallVec<[f64; 8]>,
        size: &mut smallvec::SmallVec<[usize; 8]>,
    ) -> bool {
        if self.child_leaf.len() != self.k {
            return false;
        }
        leaf.extend_from_slice(&self.child_leaf);
        size.extend_from_slice(&self.child_size);
        true
    }
}

impl RashPoolRec {
    #[inline]
    fn n_children(&self) -> usize {
        self.data_children as usize
    }
    #[inline]
    fn stride(&self) -> usize {
        self.data.len() / self.n_children().max(1)
    }
    #[inline]
    fn child_n(&self, i: usize) -> i64 {
        self.data[i * self.stride() + 1]
    }
    #[inline]
    fn child_counts(&self, i: usize) -> &[i64] {
        let st = self.stride();
        &self.data[i * st + 2..(i + 1) * st]
    }
}

impl RashLbRec for RashPoolRec {
    fn feature_index(&self) -> usize {
        self.feature_index as usize
    }
    fn thresholds(&self) -> &[f64] {
        &self.thresholds
    }
    fn group_map(&self) -> Option<&[usize]> {
        self.group_map.as_deref()
    }
    fn k(&self) -> usize {
        self.n_children()
    }
    fn cached_children(
        &self,
        _leaf_of: &dyn Fn(&[i64], f64) -> f64,
        leaf: &mut smallvec::SmallVec<[f64; 8]>,
        size: &mut smallvec::SmallVec<[usize; 8]>,
    ) -> bool {
        let st = self.stride();
        for i in 0..self.n_children() {
            leaf.push(f64::from_bits(self.data[i * st] as u64));
            size.push(self.data[i * st + 1].max(0) as usize);
        }
        true
    }
}

impl RashLbRec for RashPoolRecRef<'_> {
    fn feature_index(&self) -> usize {
        self.h().feature as usize
    }
    fn thresholds(&self) -> &[f64] {
        self.thr_arc()
    }
    fn group_map(&self) -> Option<&[usize]> {
        self.gm()
    }
    fn k(&self) -> usize {
        self.n_children()
    }
    fn cached_children(
        &self,
        leaf_of: &dyn Fn(&[i64], f64) -> f64,
        leaf: &mut smallvec::SmallVec<[f64; 8]>,
        size: &mut smallvec::SmallVec<[usize; 8]>,
    ) -> bool {
        let nc = self.a.nc as usize;
        let mut buf = [0i64; MAX_CLASSES];
        for c in 0..self.n_children() {
            let cc = self.counts_into(c, &mut buf);
            // `rash_shared_build`'s expression on the same integers.
            let nw = cc[..nc].iter().sum::<i64>() as f64;
            leaf.push(leaf_of(cc, nw));
            size.push(self.cell(c)[0] as usize);
        }
        true
    }
}

/// The completion-lb DFS's pool at a support (`rash_lb_pool`).
enum LbPool {
    Legacy(Arc<[RashLbSplit]>),
    Shared(RashSharedPool),
}

/// One explore candidate (`rash_explore_node`): a Select candidate or a
/// Shared Select pool record.
trait ExploreCand {
    fn x_feature(&self) -> usize;
    fn x_thresholds(&self) -> &Arc<[f64]>;
    fn x_group_map(&self) -> Option<&[usize]>;
    fn x_k(&self) -> usize;
    fn x_n_children(&self) -> usize;
    fn x_child_n(&self, i: usize) -> i64;
    /// Child `i`'s class counts (a compact record widens them into `buf`).
    fn x_child_counts<'b>(&'b self, i: usize, buf: &'b mut [i64; MAX_CLASSES]) -> &'b [i64];
}

impl ExploreCand for SplitCandidate {
    fn x_feature(&self) -> usize {
        self.feature_index
    }
    fn x_thresholds(&self) -> &Arc<[f64]> {
        &self.thresholds
    }
    fn x_group_map(&self) -> Option<&[usize]> {
        self.group_map.as_deref()
    }
    fn x_k(&self) -> usize {
        self.k
    }
    fn x_n_children(&self) -> usize {
        self.child_n.len()
    }
    fn x_child_n(&self, i: usize) -> i64 {
        self.child_n[i]
    }
    fn x_child_counts<'b>(&'b self, i: usize, _buf: &'b mut [i64; MAX_CLASSES]) -> &'b [i64] {
        &self.child_counts[i]
    }
}

impl ExploreCand for RashPoolRec {
    fn x_feature(&self) -> usize {
        self.feature_index as usize
    }
    fn x_thresholds(&self) -> &Arc<[f64]> {
        &self.thresholds
    }
    fn x_group_map(&self) -> Option<&[usize]> {
        self.group_map.as_deref()
    }
    fn x_k(&self) -> usize {
        self.sel_k as usize
    }
    fn x_n_children(&self) -> usize {
        self.n_children()
    }
    fn x_child_n(&self, i: usize) -> i64 {
        self.child_n(i)
    }
    fn x_child_counts<'b>(&'b self, i: usize, _buf: &'b mut [i64; MAX_CLASSES]) -> &'b [i64] {
        self.child_counts(i)
    }
}

impl ExploreCand for RashPoolRecRef<'_> {
    fn x_feature(&self) -> usize {
        self.h().feature as usize
    }
    fn x_thresholds(&self) -> &Arc<[f64]> {
        self.thr_arc()
    }
    fn x_group_map(&self) -> Option<&[usize]> {
        self.gm()
    }
    fn x_k(&self) -> usize {
        self.h().sel_k as usize
    }
    fn x_n_children(&self) -> usize {
        self.n_children()
    }
    fn x_child_n(&self, i: usize) -> i64 {
        self.cell(i)[0] as i64
    }
    fn x_child_counts<'b>(&'b self, i: usize, buf: &'b mut [i64; MAX_CLASSES]) -> &'b [i64] {
        self.counts_into(i, buf)
    }
}

/// Bitwise split identity (feature, thresholds, group_map) of two explore
/// candidates.
fn same_identity(a: &dyn ExploreCand, b: &dyn ExploreCand) -> bool {
    a.x_feature() == b.x_feature()
        && a.x_thresholds().len() == b.x_thresholds().len()
        && a.x_thresholds()
            .iter()
            .zip(b.x_thresholds().iter())
            .all(|(x, y)| x.to_bits() == y.to_bits())
        && a.x_group_map() == b.x_group_map()
}

fn identity_hash(c: &dyn ExploreCand) -> u64 {
    let mut h = DefaultHasher::new();
    c.x_feature().hash(&mut h);
    for t in c.x_thresholds().iter() {
        t.to_bits().hash(&mut h);
    }
    c.x_group_map().hash(&mut h);
    h.finish()
}

/// Identity ids of explore candidates (`u32::MAX`: <= 1 non-empty child,
/// never explored) and their number: the explore takes an identity at its
/// first Row-G survivor and skips its repeats.
fn explore_identities(cands: &[&dyn ExploreCand]) -> (Vec<u32>, u32) {
    let mut first_of: HashMap<u64, Vec<u32>> = HashMap::with_capacity(cands.len());
    let mut ident = vec![u32::MAX; cands.len()];
    let mut n_ident = 0u32;
    for (i, &c) in cands.iter().enumerate() {
        if (0..c.x_n_children())
            .filter(|&j| c.x_child_n(j) > 0)
            .count()
            <= 1
        {
            continue;
        }
        let bucket = first_of.entry(identity_hash(c)).or_default();
        match bucket
            .iter()
            .find(|&&j| same_identity(cands[j as usize], c))
        {
            Some(&j) => ident[i] = ident[j as usize],
            None => {
                bucket.push(i as u32);
                ident[i] = n_ident;
                n_ident += 1;
            }
        }
    }
    (ident, n_ident)
}
/// Memo entry of `BranchAndBoundSolver::bounded_completion_min`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum RashLbEntry {
    /// Minimum completion cost over the lb-solver's option set (exact unless
    /// the solve's deadline cut the search, then an admissible lb).
    Min(f64),
    /// Every completion costs more than this bound.
    Above(f64),
}

/// Structural depth-1 memo: structural key of a remaining-depth-1 child: the rows of
/// the interned parent support `pid` (`RashD1Struct::pids`) with f-rank in
/// [lo, hi) (`rash_d1_sweep_cuts` boundaries). Under `rash_lb_gate`
/// mask_exact every child builder gives exactly these rows for a split of
/// the parent on store thresholds of f, so equal keys mean equal supports.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct RashD1Key {
    pub pid: u32,
    pub f: u16,
    pub lo: u16,
    pub hi: u16,
}

/// Two-generation cache: inserts go to `cur`; when `cur` holds more than
/// `half` entries it becomes `old` (the previous `old` is dropped, counted
/// in `evicted`). Lookups try `cur`, then `old`.
#[derive(Clone, Debug)]
pub struct RashGenMap<K, V> {
    cur: RashFxMap<K, V>,
    old: RashFxMap<K, V>,
    half: usize,
    pub evicted: u64,
}

impl<K, V> Default for RashGenMap<K, V> {
    fn default() -> Self {
        RashGenMap {
            cur: RashFxMap::default(),
            old: RashFxMap::default(),
            half: usize::MAX / 2,
            evicted: 0,
        }
    }
}

impl<K: Eq + std::hash::Hash, V: Copy> RashGenMap<K, V> {
    /// Set the total capacity (both generations); takes effect on the next insert.
    pub fn set_cap(&mut self, cap: usize) {
        self.half = (cap / 2).max(1);
    }
    pub fn get(&self, k: &K) -> Option<V> {
        self.cur.get(k).or_else(|| self.old.get(k)).copied()
    }
    pub fn insert(&mut self, k: K, v: V) {
        self.cur.insert(k, v);
        if self.cur.len() > self.half {
            self.evicted += self.old.len() as u64;
            self.old = std::mem::take(&mut self.cur);
        }
    }
    pub fn len(&self) -> usize {
        self.cur.len() + self.old.len()
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    pub fn clear(&mut self) {
        self.cur = RashFxMap::default();
        self.old = RashFxMap::default();
    }
    /// Every entry, older generation first (`lb_threads` merge).
    pub fn into_entries(self) -> impl Iterator<Item = (K, V)> {
        self.old.into_iter().chain(self.cur)
    }
}

/// Structural depth-1 memo: the structural depth-1 completion-lb memo and the
/// parent-support interner its keys refer to (they travel together: ids
/// mean nothing without `pids`).
#[derive(Clone, Debug, Default)]
pub struct RashD1Struct {
    pub pids: RashFxMap<SupportMask, u32>,
    pub map: RashGenMap<RashD1Key, RashLbEntry>,
}

impl RashD1Struct {
    pub fn clear(&mut self) {
        self.pids = RashFxMap::default();
        self.map.clear();
    }
}

pub struct BranchAndBoundSolver {
    pub search: Arc<SearchState>,
    pub worker: WorkerState,
}

impl BranchAndBoundSolver {
    /// Resolve the current best-split SplitRecord for a node.
    /// Chains best_split_table -> bound_table -> SplitRecordPool. Returns
    /// `None` if no best is registered or the pool entry has been evicted.
    /// The returned SplitRecord has contextual sidecars overlaid from the
    /// matching bound_table entry — same shape as the legacy
    /// `sp.best_split.clone()` read.
    fn resolve_best_split(&self, key: &MemoKey) -> Option<SplitRecord> {
        let split_id = self.search.best_split_table.get(key).map(|v| *v)?;
        if let Some(bounds) = self.search.bound_table.get(key) {
            for entry in bounds.iter() {
                let rec = self.worker.split_pool.get(entry.pool_idx);
                if rec.split_id == split_id {
                    let mut out = rec.clone();
                    out.contextual_lowerbound = entry.contextual_lowerbound;
                    out.contextually_dead = entry.is_contextually_dead();
                    return Some(out);
                }
            }
        }
        None
    }

    /// Install a SplitRecord as the current best split for `key`.
    /// Pushes to the pool (or hits dedup), upserts the materialized bound
    /// entry, and records the split_id in best_split_table — three writes
    /// that must travel together so `resolve_best_split` can chain back to
    /// the SplitRecord.
    fn install_best_split(&mut self, key: &MemoKey, record: SplitRecord) {
        let split_id = record.split_id;
        let feat = record.feature_index as u16;
        let su = record.split_upper;
        let pool_idx = self.worker.split_pool.push(record);
        self.search
            .bound_upsert_materialized(key, feat, pool_idx, f64::NEG_INFINITY, su);
        self.search.set_best_split_id(key, split_id);
    }

    fn split_record_id(
        feature_index: usize,
        thresholds: &[f64],
        child_keys: &[MemoKey],
        group_map: Option<&[usize]>,
    ) -> u64 {
        let mut h = DefaultHasher::new();
        feature_index.hash(&mut h);
        thresholds.len().hash(&mut h);
        for threshold in thresholds {
            threshold.to_bits().hash(&mut h);
        }
        child_keys.hash(&mut h);
        if let Some(group_map) = group_map {
            group_map.hash(&mut h);
        }
        h.finish()
    }

    /// Helpers: bridge between `Vec<MemoKey>` (used at
    /// construction sites where child_keys are first computed) and
    /// `Vec<u32>` arena indices (the in-memory representation used by
    /// SplitRecord). These run after `split_record_id` is computed so
    /// the hash determinism is preserved (split_id hashes content, not
    /// indices).
    #[inline]
    fn intern_keys(&self, keys: Vec<MemoKey>) -> Vec<u32> {
        keys.into_iter()
            .map(|k| self.search.memo_key_arena.intern(k))
            .collect()
    }

    /// Convenience accessor — `arena.get(idx)` shorthand.
    #[inline]
    fn ck(&self, idx: u32) -> &MemoKey {
        self.search.memo_key_arena.get(idx)
    }

    fn mode_flags(config: &SolverConfig) -> u16 {
        // Block compression affects the atomized representation the
        // DP operates on, so memo must isolate when it's active.
        let a1_active = config.msdt.atom_mode && config.msdt.atom_block_compression_purity > 0.0;
        (config.rashomon_mode as u16)
            | ((config.shape_mode as u16) << 1)
            | ((config.msdt.atom_mode as u16) << 2)
            | ((config.msdt.cleanup_mode as u16) << 5)
            | ((a1_active as u16) << 10)
    }

    /// Returns the `teacher_alpha` bit pattern to embed into `MemoKey`
    /// for cache discrimination. Canonicalizes to f64::to_bits(1.0) when
    /// the blend is effectively off (shape_mode=false, no reference ensemble, or
    /// α≈1.0), so runs with "teacher_alpha set but ineffective" share
    /// memo entries with plain runs without reference-ensemble logits.
    fn teacher_alpha_memo_bits(config: &SolverConfig) -> u64 {
        let alpha_active = config.shape_mode
            && config.teacher_prob.is_some()
            && config.msdt.teacher_alpha < 1.0 - 1e-12;
        if alpha_active {
            config.msdt.teacher_alpha.to_bits()
        } else {
            1.0_f64.to_bits()
        }
    }

    pub fn new(
        store: Arc<DatasetStore>,
        config: SolverConfig,
        shared_bounds: Option<Arc<SharedBoundsTable>>,
        root_support: Option<SupportMask>,
    ) -> Self {
        let n_data = store.n_samples();
        let n_total = config.n_total_override.unwrap_or(n_data);
        let weight_total = config.weight_total;
        let rs = root_support.unwrap_or_else(|| store.root_support());
        let initial_materialize_top_n = config.materialize_top_n;

        BranchAndBoundSolver {
            search: Arc::new(SearchState {
                store,
                root_support: rs,
                n_data,
                n_total,
                config,
                shared_bounds,
                weight_total,
                memo: DashMap::with_hasher(Default::default()),
                bound_table: DashMap::new(),
                best_split_table: DashMap::new(),
                parent_map: dashmap::DashSet::default(),
                queue: Mutex::new(BinaryHeap::new()),
                root_key: Mutex::new(MemoKey::new(
                    SupportMask::from_words(vec![], 0),
                    FeatureMask::default(),
                    0,
                    0,
                    1.0_f64.to_bits(),
                )),
                materialize_top_n: AtomicUsize::new(initial_materialize_top_n),
                memo_key_arena: Arc::new(MemoKeyArena::new()),
                n_subproblems: AtomicUsize::new(0),
                n_explored: AtomicUsize::new(0),
                timed_out: AtomicBool::new(false),
                counter: AtomicU64::new(0),
                start_time: Mutex::new(None),
                certification_issues: Mutex::new(Vec::new()),
                used_ub_margin_prune: AtomicBool::new(false),
                used_bfs_partial_materialization: AtomicBool::new(false),
                used_greedy_boundary_guidance: AtomicBool::new(false),
                used_closure: AtomicBool::new(false),
                rashomon_select_ub: AtomicU64::new(f64::NAN.to_bits()),
            }),
            worker: WorkerState {
                feature_k_hits: HashMap::new(),
                killer_moves: HashMap::new(),
                greedy_score_cache: HashMap::new(),
                scratch_bank: ScratchBank::new(),
                solver_scratch: SolverScratch::default(),
                similar_lb: HashMap::new(),
                similar_ub: HashMap::new(),
                profile: SolverProfile::default(),
                split_pool: types::SplitRecordPool::new(),
                candidate_pool: Vec::new(),
                rash_lb_memo: RashFxMap::default(),
                rash_lb_memo_d1: 0,
                rash_lb_d1s: RashD1Struct::default(),
                rash_lb_select: RashFxMap::default(),
                rash_select_shared: RashFxMap::default(),
                rash_lb_ub_bits: None,
                rash_lb_select_records: 0,
                rash_lb_gate: None,
                rash_lb_rank: None,
                rash_iv_cache: Default::default(),
                rash_base: None,
                rash_level_base: None,
                rash_lb_threads: 0,
                rash_time_sink: None,
                rash_deadline: None,
                rash_select_config: None,
                rash_par: None,
                rash_lb_cap_div: 1,
            },
        }
    }

    pub fn from_raw(
        x: Array2<f64>,
        y: Vec<u8>,
        config: SolverConfig,
        shared_bounds: Option<Arc<SharedBoundsTable>>,
        weights: Option<Vec<f64>>,
    ) -> Self {
        let reference_preds_vec = config.reference_preds.as_ref().map(|arc| arc.to_vec());
        let store = Arc::new(DatasetStore::new(
            x,
            y,
            weights,
            config.n_classes,
            config.thresholds_per_feature.as_ref(),
            reference_preds_vec,
        ));
        Self::new(store, config, shared_bounds, None)
    }

    pub fn solve(&mut self) -> SolveResult {
        let start = Instant::now();
        *self.search.start_time.lock().unwrap() = Some(start);
        self.rash_deadline_set(start);
        if self.search.config.closure_depth.is_some() && !self.search.config.rashomon_mode {
            return self.topk_closure_solve();
        }
        if self.search.config.rashomon_mode {
            self.rashomon_enumerate();
        } else {
            self.branch_bound_solve();
        }
        self.finish_solve()
    }

    /// Shared Step B: `solve` with the Step-A pass rooted at every
    /// seed (`rashomon_enumerate_seeded`). Returns the result (for the first
    /// seed as root) and the seed keys in order.
    pub fn solve_seeded(&mut self, seeds: &[(SupportMask, f64)]) -> (SolveResult, Vec<MemoKey>) {
        let start = Instant::now();
        *self.search.start_time.lock().unwrap() = Some(start);
        self.rash_deadline_set(start);
        let keys = self.rashomon_enumerate_seeded(seeds);
        (self.finish_solve(), keys)
    }

    /// Epilogue of `solve` / `solve_seeded`: profile memory snapshot and the
    /// `SolveResult` of the root (`search.root_key`).
    fn finish_solve(&mut self) -> SolveResult {
        // Memory snapshot for profile
        self.worker.profile.memo_size = self.search.memo.len();
        self.worker.profile.similar_lb_size = self.worker.similar_lb.len();
        self.worker.profile.similar_ub_size = self.worker.similar_ub.len();
        self.worker.profile.parent_map_size = self.search.parent_map.len();
        self.worker.profile.used_features_size = 0;

        let (root_upperbound, root_lowerbound, root_is_solved) = {
            let root_key = self.search.root_key.lock().unwrap().clone();
            let root = self.search.memo.get(&root_key).unwrap();
            (root.upperbound, root.lowerbound, root.is_solved)
        };
        let elapsed = self
            .search
            .start_time
            .lock()
            .unwrap()
            .unwrap()
            .elapsed()
            .as_secs_f64();

        let gap = (root_upperbound - root_lowerbound).max(0.0);
        let search_converged = root_is_solved || gap < 1e-10;
        let (certified_optimal, certification_mode, certification_reason, unsafe_heuristics_used) =
            self.assess_certification(search_converged);
        let termination_reason = if certified_optimal {
            "optimal"
        } else if search_converged {
            "gap_closed_uncertified"
        } else if self.search.timed_out.load(Ordering::Acquire) {
            "timeout"
        } else {
            "queue_exhausted_with_gap"
        };

        SolveResult {
            objective: root_upperbound,
            lowerbound: root_lowerbound,
            is_optimal: certified_optimal,
            certified_optimal,
            search_converged,
            certification_mode,
            certification_reason,
            unsafe_heuristics_used,
            timed_out: self.search.timed_out.load(Ordering::Relaxed),
            termination_reason: termination_reason.to_string(),
            final_gap: gap,
            n_subproblems: self.search.n_subproblems.load(Ordering::Relaxed),
            n_explored: self.search.n_explored.load(Ordering::Relaxed),
            elapsed,
        }
    }

    fn note_certification_issue(&mut self, issue: impl Into<String>) {
        let issue = issue.into();
        let mut issues = self.search.certification_issues.lock().unwrap();
        if !issues.iter().any(|existing| existing == &issue) {
            issues.push(issue);
        }
    }

    /// Shared exploratory solver for the top-k-with-closure regime.
    ///
    /// Above the closure frontier it evaluates the generated candidate set
    /// recursively. At and below the frontier it keeps only the configured
    /// closure beam. The local candidate generator is the normal one: when
    /// `shape_mode` is false it emits contiguous partitions, and when true it
    /// emits nonlinear `group_map` partitions.
    fn topk_closure_solve(&mut self) -> SolveResult {
        let root_support = self.search.root_support.clone();
        let initial_used = self.search.config.initial_used_features.clone();
        let root_key = MemoKey::new(
            root_support.clone(),
            initial_used.clone(),
            self.search.config.max_depth,
            Self::mode_flags(&self.search.config),
            Self::teacher_alpha_memo_bits(&self.search.config),
        );
        *self.search.root_key.lock().unwrap() = root_key;
        if self.search.config.closure_depth.unwrap_or(0) > 0 {
            self.search.used_closure.store(true, Ordering::Relaxed);
        }

        let objective = self.topk_closure_recurse(
            &root_support,
            &initial_used,
            self.search.config.max_depth,
            f64::INFINITY,
        );

        self.worker.profile.memo_size = self.search.memo.len();
        self.worker.profile.similar_lb_size = self.worker.similar_lb.len();
        self.worker.profile.similar_ub_size = self.worker.similar_ub.len();
        self.worker.profile.parent_map_size = self.search.parent_map.len();
        self.worker.profile.used_features_size = 0;

        let elapsed = self
            .search
            .start_time
            .lock()
            .unwrap()
            .unwrap()
            .elapsed()
            .as_secs_f64();

        let root_key = self.search.root_key.lock().unwrap().clone();
        let lowerbound = self
            .search
            .memo
            .get(&root_key)
            .map(|root| root.lowerbound.min(objective))
            .unwrap_or(0.0);
        let timed_out = self.search.timed_out.load(Ordering::Relaxed);
        self.note_certification_issue("topk_closure_direct");
        SolveResult {
            objective,
            lowerbound,
            is_optimal: false,
            certified_optimal: false,
            search_converged: !timed_out,
            certification_mode: "exploratory".to_string(),
            certification_reason: "topk_closure_direct".to_string(),
            unsafe_heuristics_used: true,
            timed_out,
            termination_reason: if timed_out { "timeout" } else { "topk_closure" }.to_string(),
            final_gap: (objective - lowerbound).max(0.0),
            n_subproblems: self.search.n_subproblems.load(Ordering::Relaxed),
            n_explored: self.search.n_explored.load(Ordering::Relaxed),
            elapsed,
        }
    }

    fn topk_closure_recurse(
        &mut self,
        support: &SupportMask,
        feature_mask: &FeatureMask,
        depth: i32,
        cutoff: f64,
    ) -> f64 {
        let key = MemoKey::new(
            support.clone(),
            feature_mask.clone(),
            depth,
            Self::mode_flags(&self.search.config),
            Self::teacher_alpha_memo_bits(&self.search.config),
        );
        if let Some(sp) = self.search.memo.get(&key) {
            if sp.is_solved || sp.is_terminal {
                return sp.upperbound;
            }
        }

        let closure_depth = self.search.config.closure_depth.unwrap_or(0);
        if depth <= closure_depth && depth > 0 {
            let started = Instant::now();
            let (_closure_key, objective) = self.closure_build(support, feature_mask, depth);
            self.worker.profile.time_closure += started.elapsed();
            self.search.used_closure.store(true, Ordering::Relaxed);
            return objective;
        }

        let mut sp = self.create_subproblem(support, feature_mask, depth);
        if depth <= 0 {
            sp.lowerbound = sp.base_objective;
            sp.upperbound = sp.base_objective;
            sp.upperbound_is_heuristic = false;
            sp.is_terminal = true;
            sp.is_solved = true;
            let obj = sp.upperbound;
            self.search.memo.insert(key, sp);
            return obj;
        }

        let leaf_obj = sp.base_objective;
        if sp.is_terminal || sp.is_solved {
            self.search.memo.insert(key, sp);
            return leaf_obj;
        }

        self.search.memo.insert(key.clone(), sp);
        if self.check_time() {
            let mut sp = self.search.memo.get_mut(&key).unwrap();
            sp.upperbound = leaf_obj;
            sp.upperbound_is_heuristic = true;
            sp.is_terminal = true;
            sp.is_solved = true;
            sp.explored = true;
            drop(sp);
            self.search.clear_best_split_id(&key);
            return leaf_obj;
        }
        self.search.n_explored.fetch_add(1, Ordering::Relaxed);
        let d_idx = (depth as usize).min(MAX_PROFILE_DEPTH - 1);
        self.worker.profile.nodes_at_depth[d_idx] += 1;
        let t_candgen = Instant::now();
        let slb_ref = if self.search.config.dp_criterion == DpCriterion::Learned {
            Some(&self.worker.similar_lb)
        } else {
            None
        };
        let (mut candidates, cand_stats) = generate_candidates_for_support(
            self.search.store.as_ref(),
            support,
            &self.search.config,
            feature_mask,
            self.search.n_total,
            Some(leaf_obj),
            slb_ref,
            &self.worker.scratch_bank,
            None,
            None,
            Some(depth),
        );
        let candgen_elapsed = t_candgen.elapsed();
        self.worker.profile.time_candidate_gen += candgen_elapsed;
        self.worker.profile.time_candgen_per_depth[d_idx] += candgen_elapsed;
        if self.check_time() {
            let mut sp = self.search.memo.get_mut(&key).unwrap();
            sp.upperbound = leaf_obj;
            sp.upperbound_is_heuristic = true;
            sp.is_terminal = true;
            sp.is_solved = true;
            sp.explored = true;
            drop(sp);
            self.search.clear_best_split_id(&key);
            return leaf_obj;
        }

        let split_reg = (1.0 - self.search.config.reg_alpha) * self.search.config.reg;
        let leaf_reg = self.search.config.reg_alpha * self.search.config.reg;
        candidates.retain(|c| split_reg + (c.k as f64) * leaf_reg < leaf_obj - 1e-12);

        self.worker.profile.n_candidates_generated += candidates.len() as u64;
        self.worker.profile.n_features_available += cand_stats.n_available;
        self.worker.profile.n_features_filtered_tier1 +=
            cand_stats.n_available - cand_stats.n_filtered_available;
        self.worker.profile.n_candidates_beam_truncated += cand_stats.n_candidates_beam_truncated;
        self.worker.profile.n_teacher_dp_active_nodes += cand_stats.n_teacher_dp_active_nodes;

        if candidates.is_empty() {
            let mut sp = self.search.memo.get_mut(&key).unwrap();
            sp.upperbound = leaf_obj;
            sp.upperbound_is_heuristic = false;
            sp.is_terminal = true;
            sp.is_solved = true;
            sp.explored = true;
            return leaf_obj;
        }

        let indices = self.search.store.support_to_indices(support);
        let n_weighted = {
            let sp = self.search.memo.get(&key).unwrap();
            sp.class_counts[..sp.n_classes as usize].iter().sum::<i64>() as f64
        };
        let path_cost = self.search.config.path_penalty * n_weighted / self.search.n_total as f64;

        let mut best_cost = leaf_obj.min(cutoff);
        let mut best_record: Option<(usize, Arc<[f64]>, Option<Vec<usize>>, Vec<MemoKey>)> = None;
        let mut best_candidate_idx: Option<usize> = None;
        let t_phase_b = Instant::now();

        let n_total_f = self.search.n_total as f64;
        let min_leaf = self.search.config.min_samples_leaf.max(1) as i64;
        let n_classes_for_sig = self.search.config.n_classes;
        let pp = self.search.config.path_penalty;

        for (cand_idx, cand) in candidates.iter().enumerate() {
            if self.check_time() {
                break;
            }

            let non_empty = cand.child_n.iter().filter(|&&n| n > 0).count();
            if non_empty <= 1 || cand.child_n.iter().any(|&n| n < min_leaf) {
                continue;
            }

            let mut child_lbs: Vec<f64> = Vec::with_capacity(cand.child_n.len());
            let mut candidate_lb = split_reg + path_cost;
            for (ci, &n_ci) in cand.child_n.iter().enumerate() {
                if n_ci <= 0 {
                    child_lbs.push(0.0);
                    continue;
                }
                let max_class = *cand.child_counts[ci][..n_classes_for_sig]
                    .iter()
                    .max()
                    .unwrap_or(&0);
                let bin_misclass = (n_ci - max_class) as f64 / n_total_f;
                let bin_path = pp * n_ci as f64 / n_total_f;
                let leaf_cost_bin = bin_misclass + leaf_reg;
                let pure_split_bin = split_reg + 2.0 * leaf_reg + bin_path;
                let lb = leaf_cost_bin.min(pure_split_bin);

                child_lbs.push(lb);
                candidate_lb += lb;
                if candidate_lb >= best_cost - 1e-12 {
                    break;
                }
            }
            if candidate_lb >= best_cost - 1e-12 {
                self.worker.profile.n_closure_lb_skipped += 1;
                continue;
            }

            let bins = self.partition_into_flat(
                &indices,
                self.search.store.feature_col(cand.feature_index),
                &cand.thresholds,
                cand.group_map.as_deref(),
            );
            if bins.len() != child_lbs.len() {
                continue;
            }
            let child_used = if self.search.config.feature_exclusion {
                feature_mask.without(cand.feature_index)
            } else {
                feature_mask.clone()
            };

            let mut process_order: Vec<usize> = (0..bins.len()).collect();
            process_order.sort_by(|&a, &b| {
                child_lbs[a]
                    .partial_cmp(&child_lbs[b])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            let mut remaining_lb: f64 = child_lbs.iter().sum();
            let mut child_key_slots: Vec<Option<MemoKey>> = vec![None; bins.len()];
            let mut cost = split_reg + path_cost;

            for &bin_idx in &process_order {
                remaining_lb -= child_lbs[bin_idx];
                let child_cutoff = best_cost - cost - remaining_lb;
                if child_cutoff <= child_lbs[bin_idx] + 1e-12 {
                    cost = f64::INFINITY;
                    break;
                }
                if self.check_time() {
                    cost = f64::INFINITY;
                    break;
                }
                let child_support = self
                    .search
                    .store
                    .support_from_indices(bins.bin_slice(bin_idx));
                let child_cost =
                    self.topk_closure_recurse(&child_support, &child_used, depth - 1, child_cutoff);
                if !child_cost.is_finite() {
                    cost = f64::INFINITY;
                    break;
                }
                let child_key = MemoKey::new(
                    child_support,
                    child_used.clone(),
                    depth - 1,
                    Self::mode_flags(&self.search.config),
                    Self::teacher_alpha_memo_bits(&self.search.config),
                );
                child_key_slots[bin_idx] = Some(child_key);
                cost += child_cost;
                if cost + remaining_lb >= best_cost - 1e-12 {
                    cost = f64::INFINITY;
                    break;
                }
            }

            let child_keys: Vec<MemoKey> = if cost.is_finite() {
                child_key_slots.into_iter().flatten().collect()
            } else {
                Vec::new()
            };

            let better = cost < best_cost - 1e-12
                || ((cost - best_cost).abs() <= 1e-12
                    && child_keys.len() == bins.len()
                    && best_candidate_idx.is_some_and(|best_idx| {
                        compare_split_candidates(cand, &candidates[best_idx])
                            == std::cmp::Ordering::Less
                    }));
            if better && child_keys.len() == bins.len() {
                best_cost = cost;
                best_candidate_idx = Some(cand_idx);
                best_record = Some((
                    cand.feature_index,
                    cand.thresholds.clone(),
                    cand.group_map.clone(),
                    child_keys,
                ));
            }
        }

        self.worker.profile.time_phase_b += t_phase_b.elapsed();
        self.worker.profile.time_phase_b_per_depth[d_idx] += t_phase_b.elapsed();

        if let Some((feature_index, thresholds, group_map, child_keys)) = best_record {
            let split_id = Self::split_record_id(
                feature_index,
                &thresholds,
                &child_keys,
                group_map.as_deref(),
            );
            let interned = self.intern_keys(child_keys.clone());
            self.register_parent_links(&interned);
            let record = SplitRecord {
                split_id,
                feature_index,
                thresholds,
                child_keys: interned,
                split_upper: best_cost,
                contextual_lowerbound: f64::NEG_INFINITY,
                contextually_dead: false,
                group_map,
            };
            self.upsert_materialized_split(&key, record);
            self.search.set_best_split_id(&key, split_id);

            let mut sp = self.search.memo.get_mut(&key).unwrap();
            sp.upperbound = best_cost;
            sp.upperbound_is_heuristic = true;
            sp.is_solved = true;
            sp.explored = true;
        } else {
            if leaf_obj >= cutoff - 1e-12 {
                return f64::INFINITY;
            }
            let mut sp = self.search.memo.get_mut(&key).unwrap();
            sp.upperbound = leaf_obj;
            sp.upperbound_is_heuristic = false;
            sp.is_terminal = true;
            sp.is_solved = true;
            sp.explored = true;
            best_cost = leaf_obj;
        }
        best_cost
    }

    fn record_branch_bound_scope_floor(&mut self, key: &MemoKey, scope: f64) {
        if !scope.is_finite() {
            return;
        }
        if let Some(mut sp) = self.search.memo.get_mut(key) {
            if scope > sp.branch_bound_scope_floor {
                sp.branch_bound_scope_floor = scope;
            }
            sp.explored = true;
        }
    }

    fn boundary_guidance_enabled(&self) -> bool {
        self.search.config.greedy_extra_depth > 0
    }

    fn hash_used_features(used_feats: &FeatureMask) -> u64 {
        let mut hasher = DefaultHasher::new();
        used_feats.hash(&mut hasher);
        hasher.finish()
    }

    fn class_counts_for_indices(&self, indices: &[u32]) -> ([i64; MAX_CLASSES], f64) {
        let mut class_counts = [0i64; MAX_CLASSES];
        if let Some(w) = self.search.store.weights() {
            let scale = self.search.n_data as f64;
            for &idx in indices {
                class_counts[self.search.store.labels()[idx as usize] as usize] +=
                    (w[idx as usize] * scale).round() as i64;
            }
        } else {
            for &idx in indices {
                class_counts[self.search.store.labels()[idx as usize] as usize] += 1;
            }
        }
        let n_weighted = class_counts[..self.search.config.n_classes]
            .iter()
            .sum::<i64>() as f64;
        (class_counts, n_weighted)
    }

    pub(crate) fn leaf_objective_from_counts(&self, class_counts: &[i64], n_weighted: f64) -> f64 {
        let leaf_reg = self.search.config.reg_alpha * self.search.config.reg;
        if n_weighted <= 0.0 {
            return leaf_reg;
        }
        let max_count = *class_counts[..self.search.config.n_classes]
            .iter()
            .max()
            .unwrap_or(&0);
        let max_loss = (n_weighted - max_count as f64) / self.search.n_total as f64;
        max_loss + leaf_reg
    }

    fn greedy_shortlist_candidates(&self, candidates: Vec<SplitCandidate>) -> Vec<SplitCandidate> {
        if self.search.config.shape_mode {
            return candidates;
        }
        let mut best_by_feature_k: HashMap<(usize, usize), SplitCandidate> = HashMap::new();
        for cand in candidates {
            let key = (cand.feature_index, cand.k);
            match best_by_feature_k.get_mut(&key) {
                Some(existing) => {
                    if cand.rank_cost < existing.rank_cost {
                        *existing = cand;
                    }
                }
                None => {
                    best_by_feature_k.insert(key, cand);
                }
            }
        }

        let mut shortlist: Vec<SplitCandidate> = best_by_feature_k.into_values().collect();
        shortlist.sort_by(|a, b| {
            a.rank_cost
                .partial_cmp(&b.rank_cost)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.k.cmp(&b.k))
                // Deterministic tie-break: (feature_index, k) is the unique map key,
                // so this makes the order independent of HashMap iteration order.
                .then(a.feature_index.cmp(&b.feature_index))
        });
        shortlist
    }

    /// Cheap greedy CART score: best binary split per feature by misclass
    /// reduction, no DP collapse. O(n × F) per node. Used for boundary
    /// guidance when full greedy_multiway_score is too expensive.
    fn greedy_cart_score(&self, indices: &[u32], remaining_depth: i32) -> f64 {
        let n = indices.len();
        let n_classes = self.search.config.n_classes;
        let n_total = self.search.n_total;
        let leaf_reg = self.search.config.reg_alpha * self.search.config.reg;
        let split_reg = (1.0 - self.search.config.reg_alpha) * self.search.config.reg;
        let path_cost = self.search.config.path_penalty * n as f64 / n_total as f64;
        let labels = self.search.store.labels();

        // Leaf cost
        let mut counts = [0i64; MAX_CLASSES];
        for &idx in indices {
            counts[labels[idx as usize] as usize] += 1;
        }
        let max_count = *counts[..n_classes].iter().max().unwrap_or(&0);
        let leaf_cost = (n as f64 - max_count as f64) / n_total as f64 + leaf_reg;

        if remaining_depth <= 0 || n <= 1 || max_count as usize == n {
            return leaf_cost;
        }

        let min_split_overhead = split_reg + 2.0 * leaf_reg + path_cost;
        if min_split_overhead >= leaf_cost {
            return leaf_cost;
        }

        // O(1) membership test
        let mut in_support = vec![false; self.search.store.n_samples()];
        for &idx in indices {
            in_support[idx as usize] = true;
        }

        let n_features = self.search.config.n_features;
        let mut best_leaf_cost = leaf_cost; // best split evaluated with leaf children
        let mut best_feature = 0usize;
        let mut best_split_pos = 0usize; // number of sorted-active indices in left

        for j in 0..n_features {
            let sorted_order = self.search.store.sorted_feature_order(j);
            let col = self.search.store.feature_col(j);

            let mut left_counts = [0i64; MAX_CLASSES];
            let mut left_n = 0usize;
            let mut prev_val = f64::NEG_INFINITY;

            for &idx in sorted_order {
                if !in_support[idx as usize] {
                    continue;
                }
                let val = col[idx as usize];

                if val > prev_val + 1e-15 && left_n > 0 && left_n < n {
                    let left_max = *left_counts[..n_classes].iter().max().unwrap_or(&0);
                    let left_misclass = (left_n as f64 - left_max as f64) / n_total as f64;
                    let mut right_counts = [0i64; MAX_CLASSES];
                    for c in 0..n_classes {
                        right_counts[c] = counts[c] - left_counts[c];
                    }
                    let right_max = *right_counts[..n_classes].iter().max().unwrap_or(&0);
                    let right_misclass = ((n - left_n) as f64 - right_max as f64) / n_total as f64;
                    let cost = left_misclass + right_misclass + min_split_overhead;

                    if cost < best_leaf_cost {
                        best_leaf_cost = cost;
                        best_feature = j;
                        best_split_pos = left_n;
                    }
                }

                left_counts[labels[idx as usize] as usize] += 1;
                left_n += 1;
                prev_val = val;
            }
        }

        if best_leaf_cost >= leaf_cost {
            return leaf_cost;
        }

        // For depth 1, leaf-level cost is exact
        if remaining_depth <= 1 {
            return best_leaf_cost;
        }

        // Rebuild partitions for the winning feature and recurse
        let sorted_order = self.search.store.sorted_feature_order(best_feature);
        let mut left = Vec::with_capacity(best_split_pos);
        let mut right = Vec::with_capacity(n - best_split_pos);
        let mut count = 0usize;
        for &idx in sorted_order {
            if !in_support[idx as usize] {
                continue;
            }
            if count < best_split_pos {
                left.push(idx);
            } else {
                right.push(idx);
            }
            count += 1;
        }

        let left_score = self.greedy_cart_score(&left, remaining_depth - 1);
        let right_score = self.greedy_cart_score(&right, remaining_depth - 1);
        let recursive_cost = left_score + right_score + split_reg + path_cost;

        recursive_cost.min(leaf_cost)
    }

    fn greedy_multiway_score(
        &mut self,
        indices: &[u32],
        remaining_depth: i32,
        used_feats: &FeatureMask,
    ) -> f64 {
        self.worker.profile.n_greedy_score_calls += 1;

        let support = self.search.store.support_from_indices(indices);
        let cache_key = GreedyCacheKey {
            indices_key: MemoKey::new(
                support,
                used_feats.clone(),
                remaining_depth,
                Self::mode_flags(&self.search.config),
                Self::teacher_alpha_memo_bits(&self.search.config),
            ),
            remaining_depth,
            used_features_hash: Self::hash_used_features(used_feats),
        };
        if let Some(&cached) = self.worker.greedy_score_cache.get(&cache_key) {
            self.worker.profile.n_greedy_cache_hits += 1;
            return cached;
        }

        let (class_counts, n_weighted) = self.class_counts_for_indices(indices);
        let leaf_obj = self
            .leaf_objective_from_counts(&class_counts[..self.search.config.n_classes], n_weighted);
        let split_reg = (1.0 - self.search.config.reg_alpha) * self.search.config.reg;
        let leaf_reg = self.search.config.reg_alpha * self.search.config.reg;
        let path_cost = self.search.config.path_penalty * n_weighted / self.search.n_total as f64;
        let max_count = *class_counts[..self.search.config.n_classes]
            .iter()
            .max()
            .unwrap_or(&0);
        let max_loss = if n_weighted > 0.0 {
            (n_weighted - max_count as f64) / self.search.n_total as f64
        } else {
            0.0
        };

        let no_available_features = self.search.config.feature_exclusion && used_feats.is_empty();
        if remaining_depth <= 0
            || indices.len() <= 1
            || max_loss == 0.0
            || max_loss < split_reg + leaf_reg + path_cost
            || no_available_features
        {
            self.worker.greedy_score_cache.insert(cache_key, leaf_obj);
            return leaf_obj;
        }

        let slb_ref = if self.search.config.dp_criterion == DpCriterion::Learned {
            Some(&self.worker.similar_lb)
        } else {
            None
        };
        let (candidates, _) = generate_candidates_for_support(
            self.search.store.as_ref(),
            &cache_key.indices_key.support,
            &self.search.config,
            used_feats,
            self.search.n_total,
            None,
            slb_ref,
            &self.worker.scratch_bank,
            None,
            None,
            None,
        );
        let candidates = self.greedy_shortlist_candidates(candidates);

        let mut best = leaf_obj;
        for cand in candidates.iter() {
            let bins = self.partition_into_flat(
                indices,
                self.search.store.feature_col(cand.feature_index),
                &cand.thresholds,
                cand.group_map.as_deref(),
            );
            let non_empty = (0..bins.len())
                .filter(|&bin_idx| bins.bin_len(bin_idx) > 0)
                .count();
            if non_empty <= 1 {
                continue;
            }

            let child_used = if self.search.config.feature_exclusion {
                used_feats.without(cand.feature_index)
            } else {
                used_feats.clone()
            };

            // This loop recurses into greedy_multiway_score, so it cannot
            // reuse solver_scratch.child_order without clobbering the parent
            // frame's ordering mid-iteration.
            let mut child_order: Vec<usize> = (0..bins.len()).collect();
            child_order.sort_unstable_by(|&a, &b| bins.bin_len(b).cmp(&bins.bin_len(a)));

            let mut score = split_reg + path_cost;
            let mut pruned = false;
            let child_order_len = child_order.len();
            for eval_idx in 0..child_order_len {
                let ci = child_order[eval_idx];
                score += self.greedy_multiway_score(
                    bins.bin_slice(ci),
                    remaining_depth - 1,
                    &child_used,
                );

                let remaining = child_order_len - eval_idx - 1;
                let optimistic = score + remaining as f64 * leaf_reg;
                if optimistic >= best - 1e-12 {
                    pruned = true;
                    break;
                }
            }

            if !pruned && score < best {
                best = score;
            }
        }

        self.worker.greedy_score_cache.insert(cache_key, best);
        best
    }

    fn boundary_candidate_score(
        &mut self,
        bins: &FlatBins,
        split_feature: usize,
        n_node: usize,
        used_feats: &FeatureMask,
        cutoff: f64,
    ) -> f64 {
        let split_reg = (1.0 - self.search.config.reg_alpha) * self.search.config.reg;
        let leaf_reg = self.search.config.reg_alpha * self.search.config.reg;
        let path_cost =
            self.search.config.path_penalty * n_node as f64 / self.search.n_total as f64;
        let child_used = if self.search.config.feature_exclusion {
            used_feats.without(split_feature)
        } else {
            used_feats.clone()
        };

        let mut child_order: Vec<usize> = (0..bins.len()).collect();
        child_order.sort_unstable_by(|&a, &b| bins.bin_len(b).cmp(&bins.bin_len(a)));

        let mut score = split_reg + path_cost;
        for (eval_idx, &ci) in child_order.iter().enumerate() {
            let child_score = if self.search.config.cheap_boundary_greedy {
                self.greedy_cart_score(bins.bin_slice(ci), self.search.config.greedy_extra_depth)
            } else {
                self.greedy_multiway_score(
                    bins.bin_slice(ci),
                    self.search.config.greedy_extra_depth,
                    &child_used,
                )
            };
            score += child_score;
            let remaining = child_order.len() - eval_idx - 1;
            let optimistic = score + remaining as f64 * leaf_reg;
            if optimistic >= cutoff - 1e-12 {
                return f64::INFINITY;
            }
        }
        score
    }

    /// Build a complete closure subtree into the memo table.
    ///
    /// At each node: generate candidates via the configured local solver,
    /// recursively build the selected children, compare split vs leaf cost,
    /// and retain the achievable upper bound. A greedy beam does not certify
    /// the lower bound, even when the leaf wins. Nodes are marked is_solved
    /// only to memoize this deterministic completion policy.
    /// SplitRecords with child_keys are stored so extract_tree() works.
    ///
    /// Returns (MemoKey, objective) for this node.
    pub fn closure_build(
        &mut self,
        support: &SupportMask,
        feature_mask: &FeatureMask,
        depth: i32,
    ) -> (MemoKey, f64) {
        let key = MemoKey::new(
            support.clone(),
            feature_mask.clone(),
            depth,
            Self::mode_flags(&self.search.config),
            Self::teacher_alpha_memo_bits(&self.search.config),
        );

        // Memo dedup: if already solved, reuse.
        if let Some(existing) = self.search.memo.get(&key) {
            if existing.is_solved || existing.is_terminal {
                return (key, existing.upperbound);
            }
        }

        if self.check_time() {
            let mut sp = self.build_closure_leaf(support, feature_mask, depth);
            sp.upperbound_is_heuristic = true;
            let obj = sp.upperbound;
            self.search.memo.insert(key.clone(), sp);
            self.publish_closure_result(&key);
            self.worker.profile.n_closure_nodes += 1;
            return (key, obj);
        }

        // Shared closure memo: another root worker may have already solved
        // this (support, feature_mask, depth). Absorb the finished subtree
        // into the local memo so extract_tree() can walk it later.
        if let Some(shared) = self.search.config.closure_shared_memo.clone() {
            let hit = {
                let entry = shared.get(&key);
                match entry {
                    Some(e) if e.is_solved || e.is_terminal => Some(e.upperbound),
                    _ => None,
                }
            };
            if let Some(obj) = hit {
                self.absorb_closure_subtree(&key, &shared);
                self.worker.profile.n_closure_shared_hits += 1;
                self.worker.profile.n_closure_nodes += 1;
                return (key, obj);
            }
        }

        // Structural fast-leaf: if the support cannot split at all (depth
        // exhausted or too few samples for two min_samples_leaf-sized bins),
        // bail before paying for create_subproblem + candidate generation.
        // Necessary condition only — create_subproblem's feasibility check
        // is strictly tighter, so any node we short-circuit here would also
        // be marked terminal by the full path.
        let min_leaf = self.search.config.min_samples_leaf.max(1);
        if depth <= 0 || support.n_samples < 2 * min_leaf {
            let sp = self.build_closure_leaf(support, feature_mask, depth);
            let obj = sp.upperbound;
            self.search.memo.insert(key.clone(), sp);
            self.publish_closure_result(&key);
            self.worker.profile.n_closure_fast_leaf += 1;
            self.worker.profile.n_closure_nodes += 1;
            return (key, obj);
        }

        // Create SubProblem for initial bounds / terminal detection.
        let t_sp = Instant::now();
        let mut sp = self.create_subproblem(support, feature_mask, depth);

        // Force leaf at depth 0 (same as get_or_create).
        if depth <= 0 && !sp.is_terminal {
            sp.lowerbound = sp.base_objective;
            sp.upperbound = sp.base_objective;
            sp.upperbound_is_heuristic = false;
            sp.is_terminal = true;
            sp.is_solved = true;
        }

        // Already terminal (pure, single sample, depth 0, etc.)
        if sp.is_terminal {
            self.worker.profile.time_closure_create_sp += t_sp.elapsed();
            let obj = sp.upperbound;
            self.search.memo.insert(key.clone(), sp);
            self.publish_closure_result(&key);
            self.worker.profile.n_closure_nodes += 1;
            return (key, obj);
        }

        let leaf_obj = sp.base_objective;
        let n_samples = sp.n_samples;
        let n_classes = sp.n_classes as usize;
        let class_counts_copy = sp.class_counts;

        // Insert into memo before recursing (prevents re-entry from building
        // a duplicate; children calling get_or_create will see this key).
        sp.is_solved = false;
        self.search.memo.insert(key.clone(), sp);
        self.worker.profile.time_closure_create_sp += t_sp.elapsed();

        // Generate candidates.
        let t_cg = Instant::now();
        let indices: Vec<u32> = self.search.store.support_to_indices(support);
        let slb_ref = if self.search.config.dp_criterion == DpCriterion::Learned {
            Some(&self.worker.similar_lb)
        } else {
            None
        };
        let candidates = if self.search.config.shape_mode {
            // Shape closure is a greedy handoff, not the certified lookahead
            // prefix. The full shape DP at arity 4/5 is expensive on dense
            // LightGBM threshold caches and made the closure tail dominate
            // wall time. Keep the exact prefix at the user-requested max_k,
            // but use the existing fast shape reference generator in closure.
            let closure_config;
            let config_ref = if self.search.config.n_classes == 2 && self.search.config.max_k > 3 {
                closure_config = {
                    let mut cfg = self.search.config.clone();
                    cfg.max_k = 3;
                    cfg
                };
                &closure_config
            } else {
                &self.search.config
            };
            let (cands, _) = generate_candidates_for_support(
                self.search.store.as_ref(),
                support,
                config_ref,
                feature_mask,
                self.search.n_total,
                Some(leaf_obj),
                slb_ref,
                &self.worker.scratch_bank,
                None,
                Some(0.0),
                Some(depth),
            );
            cands
        } else {
            // Fused contiguous closure: binary scan + feature pre-filter +
            // Rush/generic DP for k>2.
            let mut b1_filtered_local: u64 = 0;
            let result = self.closure_fast_cand_gen(
                support,
                feature_mask,
                &indices,
                &class_counts_copy,
                leaf_obj,
                depth,
                &mut b1_filtered_local,
            );
            self.worker.profile.n_closure_b1_filtered += b1_filtered_local;
            result
        };
        self.worker.profile.time_closure_cand_gen += t_cg.elapsed();
        // Arity prune on closure-emitted candidates.
        let candidates = {
            let mut cs = candidates;
            let split_reg = (1.0 - self.search.config.reg_alpha) * self.search.config.reg;
            let leaf_reg = self.search.config.reg_alpha * self.search.config.reg;
            cs.retain(|c| split_reg + (c.k as f64) * leaf_reg < leaf_obj - 1e-12);
            cs
        };
        let t_sl = Instant::now();
        let mut candidates = self.greedy_shortlist_candidates(candidates);
        // Integer-objective mode: the `to_expand` beam truncation below assumes
        // candidates are sorted by ascending split_ub (see the LB-prune comment
        // in the sibling loop), but `greedy_shortlist_candidates` returns small
        // lists (<=32 candidates) in feature-emission order. With closure_beam=1
        // (the lickety default) the depth-1 fast path would lock in the FIRST
        // feature's split rather than the cheapest, giving a loose UB. Sort by
        // split_ub so the beam keeps the cheapest splits. Gated on integer
        // mode; the EXACT/DAG and float-mode paths keep their existing ordering.
        if self.search.config.n_total_override == Some(1) {
            candidates.sort_by(|a, b| {
                a.split_ub
                    .partial_cmp(&b.split_ub)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then(a.k.cmp(&b.k))
            });
        }
        let candidates = candidates;
        self.worker.profile.time_closure_shortlist += t_sl.elapsed();

        if candidates.is_empty() {
            // No viable splits — this is a leaf.
            let mut sp = self.search.memo.get_mut(&key).unwrap();
            sp.upperbound = leaf_obj;
            sp.upperbound_is_heuristic = true;
            sp.is_terminal = true;
            sp.is_solved = true;
            drop(sp);
            // Dual-write: best_split cleared.
            self.search.clear_best_split_id(&key);
            self.similar_ub_update(n_samples, &class_counts_copy[..n_classes], leaf_obj);
            self.publish_closure_result(&key);
            self.worker.profile.n_closure_nodes += 1;
            return (key, leaf_obj);
        }

        let split_reg = (1.0 - self.search.config.reg_alpha) * self.search.config.reg;
        let leaf_reg = self.search.config.reg_alpha * self.search.config.reg;
        let n_total = self.search.n_total;
        let n_weighted: f64 = class_counts_copy[..n_classes].iter().sum::<i64>() as f64;
        let path_cost = self.search.config.path_penalty * n_weighted / n_total as f64;

        let mut best_split_cost = f64::INFINITY;
        let mut best_split_record: Option<SplitRecord> = None;
        let t_loop = Instant::now();

        // Select candidates to expand based on mode, then recursively
        // materialize each and keep the best by actual cost.
        let min_leaf_i64 = min_leaf as i64;
        let beam = self.search.config.closure_beam.max(1);
        let to_expand: Vec<&SplitCandidate> = candidates
            .iter()
            .filter(|c| {
                c.child_n.iter().filter(|&&n| n > 0).count() > 1
                    && !c.child_n.iter().any(|&n| n < min_leaf_i64)
            })
            .take(beam)
            .collect();

        // Depth-1 fast path: children are leaves, so the inline split_ub
        // IS the exact cost. Pick the winner by split_ub alone; only
        // materialize that one candidate's children (needed for extract_tree).
        if depth == 1 && !to_expand.is_empty() {
            let best_cand = to_expand
                .iter()
                .min_by(|a, b| {
                    a.split_ub
                        .partial_cmp(&b.split_ub)
                        .unwrap_or(std::cmp::Ordering::Equal)
                        .then_with(|| compare_split_candidates(a, b))
                })
                .unwrap();

            let cand_cost = best_cand.split_ub;
            if cand_cost < leaf_obj - 1e-12 {
                let bins = self.partition_into_flat(
                    &indices,
                    self.search.store.feature_col(best_cand.feature_index),
                    &best_cand.thresholds,
                    best_cand.group_map.as_deref(),
                );
                let child_used = if self.search.config.feature_exclusion {
                    feature_mask.without(best_cand.feature_index)
                } else {
                    feature_mask.clone()
                };
                let mut child_keys: Vec<MemoKey> = Vec::with_capacity(bins.len());
                for ci in 0..bins.len() {
                    let child_support = self.search.store.support_from_indices(bins.bin_slice(ci));
                    let child_key = self.get_or_create(&child_support, &child_used, depth - 1);
                    child_keys.push(child_key);
                }
                let split_id = Self::split_record_id(
                    best_cand.feature_index,
                    &best_cand.thresholds,
                    &child_keys,
                    best_cand.group_map.as_deref(),
                );
                let child_keys = self.intern_keys(child_keys);
                best_split_cost = cand_cost;
                best_split_record = Some(SplitRecord {
                    split_id,
                    feature_index: best_cand.feature_index,
                    thresholds: best_cand.thresholds.clone(),
                    child_keys,
                    split_upper: cand_cost,
                    contextual_lowerbound: f64::NEG_INFINITY,
                    contextually_dead: false,
                    group_map: best_cand.group_map.clone(),
                });
            }
            // Skip the full sibling loop — jump to final decision.
        } else {
            // Each sibling subtree in `to_expand` is independent: it calls
            // get_or_create on a disjoint child support, and only touches shared
            // state through Arc<SearchState>/DashMap. When the flag is on and we
            // aren't already parallelized at the BFS layer, evaluate siblings
            // concurrently in a scoped thread per candidate.
            // Parallelize the top N levels of a closure subtree. Rayon's work-
            // stealing pool keeps the worker count bounded, but per-task overhead
            // (fresh WorkerState + memo contention) compounds with depth, so we
            // cap at 2 levels. Inner calls fall through to the sequential path.
            //
            // If we're already inside a rayon pool (e.g. solve_parallel_root
            // dispatched us from a worker thread), nesting another par_iter adds
            // pure overhead — the outer level already saturates cores. Skip.
            const CLOSURE_PARALLEL_LEVELS: i32 = 2;
            let within_parallel_window = self
                .search
                .config
                .closure_depth
                .map_or(false, |d| depth > d - CLOSURE_PARALLEL_LEVELS && depth <= d);
            let already_in_rayon = rayon::current_thread_index().is_some();
            let use_parallel = self.search.config.closure_parallel
                && self.search.config.n_workers <= 1
                && to_expand.len() >= 2
                && within_parallel_window
                && !already_in_rayon;

            if use_parallel {
                let search_arc = Arc::clone(&self.search);
                let indices_ref: &[u32] = &indices;
                let feature_mask_ref: &FeatureMask = feature_mask;

                let per_task: Vec<(f64, Option<SplitRecord>, WorkerState)> = to_expand
                    .par_iter()
                    .map(|cand| {
                        let cand: &SplitCandidate = cand;
                        let mut solver = BranchAndBoundSolver {
                            search: Arc::clone(&search_arc),
                            worker: WorkerState {
                                feature_k_hits: HashMap::new(),
                                killer_moves: HashMap::new(),
                                greedy_score_cache: HashMap::new(),
                                scratch_bank: ScratchBank::new(),
                                solver_scratch: SolverScratch::default(),
                                similar_lb: HashMap::new(),
                                similar_ub: HashMap::new(),
                                profile: SolverProfile::default(),
                                split_pool: types::SplitRecordPool::new(),
                                candidate_pool: Vec::new(),
                                rash_lb_memo: RashFxMap::default(),
                                rash_lb_memo_d1: 0,
                                rash_lb_d1s: RashD1Struct::default(),
                                rash_lb_select: RashFxMap::default(),
                                rash_select_shared: RashFxMap::default(),
                                rash_lb_ub_bits: None,
                                rash_lb_select_records: 0,
                                rash_lb_gate: None,
                                rash_lb_rank: None,
                                rash_iv_cache: Default::default(),
                                rash_base: None,
                                rash_level_base: None,
                                rash_lb_threads: 0,
                                rash_time_sink: None,
                                rash_deadline: None,
                                rash_select_config: None,
                                rash_par: None,
                                rash_lb_cap_div: 1,
                            },
                        };

                        let bins = solver.partition_into_flat(
                            indices_ref,
                            solver.search.store.feature_col(cand.feature_index),
                            &cand.thresholds,
                            cand.group_map.as_deref(),
                        );
                        if bins.len() != cand.child_n.len() {
                            return (f64::INFINITY, None, solver.worker);
                        }
                        let child_used = if solver.search.config.feature_exclusion {
                            feature_mask_ref.without(cand.feature_index)
                        } else {
                            feature_mask_ref.clone()
                        };

                        let mut child_keys: Vec<MemoKey> = Vec::with_capacity(bins.len());
                        let mut score = split_reg + path_cost;
                        for ci in 0..bins.len() {
                            let child_support =
                                solver.search.store.support_from_indices(bins.bin_slice(ci));
                            let child_key =
                                solver.get_or_create(&child_support, &child_used, depth - 1);
                            let child_obj = solver.search.memo.get(&child_key).unwrap().upperbound;
                            child_keys.push(child_key);
                            score += child_obj;
                        }

                        let split_id = Self::split_record_id(
                            cand.feature_index,
                            &cand.thresholds,
                            &child_keys,
                            cand.group_map.as_deref(),
                        );
                        let child_keys = solver.intern_keys(child_keys);
                        let sr = SplitRecord {
                            split_id,
                            feature_index: cand.feature_index,
                            thresholds: cand.thresholds.clone(),
                            child_keys,
                            split_upper: score,
                            contextual_lowerbound: f64::NEG_INFINITY,
                            contextually_dead: false,
                            group_map: cand.group_map.clone(),
                        };
                        (score, Some(sr), solver.worker)
                    })
                    .collect();

                for (score, sr_opt, worker) in per_task {
                    if score < best_split_cost {
                        best_split_cost = score;
                        best_split_record = sr_opt;
                    }
                    self.worker.profile.merge_from(&worker.profile);
                }
            } else {
                // Closure-internal LB pruning of losing siblings.
                // `to_expand` is sorted by ascending split_ub, so
                // after the first candidate is recursed, best_split_cost is
                // tight. For each subsequent candidate, compute a cheap
                // admissible lower bound on its subtree cost from the per-bin
                // class counts already cached on the SplitCandidate
                // (child_n / child_counts from DP collapse). For each bin:
                //   trivial_lb_bin = min(leaf_cost_bin, split_reg + 2*leaf_reg + bin_path)
                // — matches `trivial_lb` in create_subproblem.
                // Sum + parent overhead is a strict LB on the candidate's
                // actual cost, so skipping when LB >= best is lossless.
                let n_total_f = n_total as f64;
                let pp = self.search.config.path_penalty;
                let n_classes_for_sig = self.search.config.n_classes;
                for cand in &to_expand {
                    if self.check_time() {
                        break;
                    }
                    // Cheap LB precheck. Skip the partition + recursion if
                    // the candidate provably can't improve best_split_cost.
                    // Use only the count-derived universal bound. A solved
                    // cost from another support with the same class counts is
                    // not a lower bound for this child.
                    if best_split_cost.is_finite() {
                        let mut sum_lb = split_reg + path_cost;
                        for ci in 0..cand.child_n.len() {
                            let n_ci = cand.child_n[ci];
                            if n_ci <= 0 {
                                continue;
                            }
                            let c0_ci = cand.child_counts[ci][0];
                            let c1_ci = n_ci - c0_ci;
                            let bin_min = c0_ci.min(c1_ci) as f64;
                            let bin_path = pp * n_ci as f64 / n_total_f;
                            let leaf_cost_bin = bin_min / n_total_f + leaf_reg;
                            let pure_split_bin = split_reg + 2.0 * leaf_reg + bin_path;
                            let bin_lb = leaf_cost_bin.min(pure_split_bin);
                            sum_lb += bin_lb;
                            if sum_lb >= best_split_cost - 1e-12 {
                                break;
                            }
                        }
                        if sum_lb >= best_split_cost - 1e-12 {
                            self.worker.profile.n_closure_lb_skipped += 1;
                            continue;
                        }
                    }

                    let bins = self.partition_into_flat(
                        &indices,
                        self.search.store.feature_col(cand.feature_index),
                        &cand.thresholds,
                        cand.group_map.as_deref(),
                    );
                    if bins.len() != cand.child_n.len() {
                        continue;
                    }
                    let child_used = if self.search.config.feature_exclusion {
                        feature_mask.without(cand.feature_index)
                    } else {
                        feature_mask.clone()
                    };

                    // Intra-candidate alpha-beta on bin
                    // order. Pre-compute each bin's admissible LB (from cand-
                    // cached child_n / child_counts), sort bins by ascending
                    // LB, then resolve in that order tracking running_score +
                    // sum_remaining_lbs. Break early when the partial sum can
                    // already not beat best_split_cost. Lossless — same LB
                    // used by the sibling pruning above. Composes with the terminal short-
                    // circuit (cheap bins resolve to exact leaf cost first,
                    // tightening the running score sooner).
                    //
                    // child_keys is kept in PARTITION order (split_record_id
                    // hashes them by index), so resolved keys are slotted
                    // into a pre-sized Vec<Option<MemoKey>>.
                    let nbins = bins.len();
                    let mut bin_lbs: Vec<f64> = Vec::with_capacity(nbins);
                    let mut bin_class_counts_per_bin: Vec<[i64; MAX_CLASSES]> =
                        Vec::with_capacity(nbins);
                    let mut bin_provably_leaf: Vec<bool> = Vec::with_capacity(nbins);
                    for ci in 0..nbins {
                        let n_ci = cand.child_n[ci];
                        let mut bcc = [0i64; MAX_CLASSES];
                        for c in 0..n_classes_for_sig {
                            bcc[c] = cand.child_counts[ci][c];
                        }
                        let max_class = *bcc[..n_classes_for_sig].iter().max().unwrap_or(&0);
                        let bin_misclass = (n_ci - max_class) as f64 / n_total_f;
                        let bin_path = pp * n_ci as f64 / n_total_f;
                        let leaf_cost_bin = bin_misclass + leaf_reg;
                        let pure_split_bin = split_reg + 2.0 * leaf_reg + bin_path;
                        let provably_leaf = n_ci > 0 && leaf_cost_bin <= pure_split_bin + 1e-15;
                        // Per-bin admissible LB (same formula as the sibling pruning).
                        let lb = leaf_cost_bin.min(pure_split_bin);
                        bin_lbs.push(lb);
                        bin_class_counts_per_bin.push(bcc);
                        bin_provably_leaf.push(provably_leaf);
                    }
                    // Sort bin indices by ascending LB → cheap bins first.
                    let mut process_order: Vec<usize> = (0..nbins).collect();
                    process_order.sort_by(|&a, &b| {
                        bin_lbs[a]
                            .partial_cmp(&bin_lbs[b])
                            .unwrap_or(std::cmp::Ordering::Equal)
                    });
                    let mut sum_remaining_lbs: f64 = bin_lbs.iter().sum();

                    let mut child_keys_slots: Vec<Option<MemoKey>> = vec![None; nbins];
                    let mut score = split_reg + path_cost;
                    let mut alpha_beta_aborted = false;

                    for &ci in &process_order {
                        if self.check_time() {
                            alpha_beta_aborted = true;
                            break;
                        }
                        sum_remaining_lbs -= bin_lbs[ci];

                        // Resolve this bin (terminal short-circuit when
                        // provably leaf, otherwise recurse via get_or_create).
                        let child_support =
                            self.search.store.support_from_indices(bins.bin_slice(ci));
                        let bin_cc = bin_class_counts_per_bin[ci];
                        let n_ci = cand.child_n[ci];
                        let provably_leaf = bin_provably_leaf[ci];

                        let child_key = if provably_leaf {
                            let key = MemoKey::new(
                                child_support.clone(),
                                child_used.clone(),
                                depth - 1,
                                Self::mode_flags(&self.search.config),
                                Self::teacher_alpha_memo_bits(&self.search.config),
                            );
                            if !self.search.memo.contains_key(&key) {
                                let leaf_sp = self.build_closure_terminal_with_counts(
                                    &child_support,
                                    &child_used,
                                    depth - 1,
                                    n_ci as usize,
                                    bin_cc,
                                );
                                self.search.memo.insert(key.clone(), leaf_sp);
                                self.worker.profile.n_closure_terminal_shortcircuit += 1;
                            }
                            key
                        } else {
                            self.get_or_create(&child_support, &child_used, depth - 1)
                        };
                        let child_obj = {
                            let csp = self.search.memo.get(&child_key).unwrap();
                            csp.upperbound
                        };
                        score += child_obj;
                        child_keys_slots[ci] = Some(child_key);

                        // Alpha-beta: if running_score + sum_remaining_lbs
                        // already ≥ best_split_cost, this candidate provably
                        // can't improve. Skip remaining bins.
                        //
                        // Also cap the threshold by the parent
                        // B&B's `warm_start_ub` if it's tighter. Closure's
                        // returned UB is only useful to the parent if it's
                        // below warm_start_ub; if running cost has already
                        // exceeded warm_start_ub, this candidate provably
                        // won't be picked by the parent and we can abandon
                        // its remaining bins. Lossless: closure still
                        // returns a valid (looser) UB — leaf_obj as floor.
                        let ws_cap = self.search.config.warm_start_ub.unwrap_or(f64::INFINITY);
                        let prune_threshold = best_split_cost.min(ws_cap);
                        if prune_threshold.is_finite()
                            && score + sum_remaining_lbs >= prune_threshold - 1e-12
                        {
                            alpha_beta_aborted = true;
                            self.worker.profile.n_closure_alpha_beta_aborted += 1;
                            break;
                        }
                    }

                    if alpha_beta_aborted {
                        // Some bins weren't resolved. Don't try to register
                        // this candidate as best_split (we don't have all
                        // child keys). The candidate is provably suboptimal
                        // anyway.
                        continue;
                    }

                    // All bins resolved → assemble child_keys in partition
                    // order from the slots.
                    let child_keys: Vec<MemoKey> = child_keys_slots
                        .into_iter()
                        .map(|opt| opt.expect("all bins resolved"))
                        .collect();

                    if score < best_split_cost {
                        best_split_cost = score;
                        let split_id = Self::split_record_id(
                            cand.feature_index,
                            &cand.thresholds,
                            &child_keys,
                            cand.group_map.as_deref(),
                        );
                        let child_keys = self.intern_keys(child_keys);
                        best_split_record = Some(SplitRecord {
                            split_id,
                            feature_index: cand.feature_index,
                            thresholds: cand.thresholds.clone(),
                            child_keys,
                            split_upper: score,
                            contextual_lowerbound: f64::NEG_INFINITY,
                            contextually_dead: false,
                            group_map: cand.group_map.clone(),
                        });
                    }
                }
            }
        } // close depth==1 fast-path else

        // Choose leaf vs best split.
        let final_obj = leaf_obj.min(best_split_cost);
        let split_is_better = best_split_cost < leaf_obj - 1e-12;
        {
            let mut sp = self.search.memo.get_mut(&key).unwrap();
            sp.upperbound = final_obj;
            sp.is_solved = true;
            sp.explored = true;

            if split_is_better {
                // Closure built a multiway subtree greedily, so final_obj is
                // an admissible UB (>= optimal) but not necessarily optimal.
                // Keep sp.lowerbound at the count-based LB create_subproblem
                // set so the LB <= optimal <= UB invariant holds. Mark UB
                // heuristic so re-entry guards know.
                sp.upperbound_is_heuristic = true;
                sp.is_terminal = false;
                drop(sp);
                // Install or clear the best_split via the helper
                // so resolve_best_split can chain back to the SplitRecord.
                match best_split_record.clone() {
                    Some(sr) => self.install_best_split(&key, sr),
                    None => self.search.clear_best_split_id(&key),
                }
                // Re-acquire sp for any downstream mutation in this branch.
                let mut sp = self.search.memo.get_mut(&key).unwrap();
                let _ = &mut sp; // silence unused-mut
                                 // In Rashomon mode, also populate rashomon_indices so the DAG
                                 // builder can follow the closure subtree. Raw push (no dedup):
                                 // repeats are harmless because build_factored_recursive keeps
                                 // one FactoredSplit per split identity.
                if self.search.config.rashomon_mode {
                    let sr_clone = best_split_record.clone();
                    drop(sp);
                    if let Some(sr) = sr_clone {
                        // Dual-write: capture feature/upper before
                        // moving the SplitRecord into the pool.
                        let feat = sr.feature_index as u16;
                        let su = sr.split_upper;
                        let idx = self.worker.split_pool.push(sr);
                        self.search
                            .bound_upsert(&key, feat, idx, f64::NEG_INFINITY, su);
                        self.search.bound_set_rashomon_flag(&key, feat, true);
                        let mut sp = self.search.memo.get_mut(&key).unwrap();
                        sp.rashomon_indices.push(idx);
                    }
                }
            } else {
                // The leaf wins among the considered greedy proposals. This
                // is an achievable upper bound, not a proof of optimality.
                sp.upperbound_is_heuristic = true;
                // Dual-write: best_split cleared.
                self.search.clear_best_split_id(&key);
                sp.is_terminal = true;
            }
        }

        self.worker.profile.time_closure_sibling_loop += t_loop.elapsed();

        // similar_lb is monotone-increasing and only accepts exact lower
        // bounds. `final_obj` here is min(leaf_cost, best_closure_split_cost) —
        // both are achievable trees, so final_obj is an UB on optimal_at_depth_D,
        // NOT an LB. (Even in the `!split_is_better` branch, closure's greedy
        // beam can miss a better split; "leaf wins among considered candidates"
        // ≠ "leaf is globally optimal".) Pushing it as an LB inflates similar_lb
        // and corrupts cross-hop bound transfer. Leaf-only branches
        // (candidates.is_empty()) DO push exact LBs because no split is feasible.
        // Push to similar_ub only.
        self.similar_ub_update(n_samples, &class_counts_copy[..n_classes], final_obj);
        self.publish_closure_result(&key);
        self.worker.profile.n_closure_nodes += 1;

        (key, final_obj)
    }

    /// Mirror a just-inserted closure SubProblem into the shared closure
    /// memo so other root workers can reuse it. Transitively publishes any
    /// child_keys reachable via best_split because some closure descendants
    /// (e.g. depth=0 fast leaves created via get_or_create's non-closure
    /// path) may not have been published at their own finish sites. No-op
    /// when sharing is off. Publish-if-absent to avoid clobbering races.
    fn publish_closure_result(&self, key: &MemoKey) {
        let Some(shared) = self.search.config.closure_shared_memo.clone() else {
            return;
        };
        let mut stack = vec![key.clone()];
        while let Some(k) = stack.pop() {
            if shared.contains_key(&k) {
                continue;
            }
            let sp = match self.search.memo.get(&k) {
                Some(r) => r.clone(),
                None => continue,
            };
            // best_split lives in best_split_table; resolve via the
            // local solver to walk the subtree.
            if let Some(sr) = self.resolve_best_split(&k) {
                for &ck_idx in &sr.child_keys {
                    stack.push(self.ck(ck_idx).clone());
                }
            }
            shared.insert(k, sp);
        }
    }

    /// After a shared closure memo hit, copy the entire subtree (parent plus
    /// all reachable child_keys via best_split) into the local per-solver
    /// memo. extract_tree() and downstream code walk local memo by key, so
    /// the whole subtree must be resident locally.
    fn absorb_closure_subtree(
        &mut self,
        key: &MemoKey,
        shared: &Arc<DashMap<MemoKey, SubProblem>>,
    ) {
        let mut stack = vec![key.clone()];
        while let Some(k) = stack.pop() {
            if self.search.memo.contains_key(&k) {
                continue;
            }
            let sp = match shared.get(&k) {
                Some(entry) => entry.clone(),
                None => continue,
            };
            // NOTE: closure_share_memo=true would need the
            // shared map to also carry best_split_id, but with `share_memo`
            // defaulting to False, the absorbed subtree's
            // best_split is reconstructed from the LOCAL best_split_table
            // (populated by absorb itself when this same solver later
            // re-explores). Cross-solver share-memo absorb is now broken;
            // re-enable it by extending the shared map's value type to
            // include best_split_id.
            if let Some(sr) = self.resolve_best_split(&k) {
                for &ck_idx in &sr.child_keys {
                    stack.push(self.ck(ck_idx).clone());
                }
            }
            self.search.memo.insert(k, sp);
        }
    }

    fn heuristic_bound_transfer_enabled(&self) -> bool {
        self.search.config.allow_heuristic_bound_transfer
    }

    fn witness_objective(&self, key: &MemoKey) -> f64 {
        // resolve_best_split is the single source of truth — chains
        // best_split_table -> bound_table -> SplitRecordPool. Returns None
        // both when no best is registered AND when the registered split_id
        // can't be located in the current bound_table (e.g., dangling after
        // a refresh that culled the entry). Either case → leaf cost.
        let (base_objective, child_keys_opt, split_reg_contribution) = {
            let sp = self.search.memo.get(key).unwrap();
            let sr = match self.resolve_best_split(key) {
                Some(sr) => sr,
                None => return sp.base_objective,
            };
            if sp.upperbound >= sp.base_objective - 1e-12 {
                return sp.base_objective;
            }
            let split_reg = (1.0 - self.search.config.reg_alpha) * self.search.config.reg;
            let n_weighted = sp.class_counts[..sp.n_classes as usize].iter().sum::<i64>() as f64;
            let path_cost =
                self.search.config.path_penalty * n_weighted / self.search.n_total as f64;
            (
                sp.base_objective,
                Some(sr.child_keys.clone()),
                split_reg + path_cost,
            )
        };
        let child_keys: Vec<u32> = child_keys_opt.unwrap();
        let _ = base_objective;
        split_reg_contribution
            + child_keys
                .iter()
                .map(|&ck_idx| self.witness_objective(self.ck(ck_idx)))
                .sum::<f64>()
    }

    fn assess_certification(&mut self, search_converged: bool) -> (bool, String, String, bool) {
        let root_key = self.search.root_key.lock().unwrap().clone();
        let root_upperbound = match self.search.memo.get(&root_key) {
            Some(root) => root.upperbound,
            None => {
                return (
                    false,
                    "exploratory".to_string(),
                    "root_missing".to_string(),
                    true,
                );
            }
        };

        let nontrivial_search = self.worker.profile.n_candidates_generated > 0
            || self.search.n_explored.load(Ordering::Relaxed) > 0
            || self.search.n_subproblems.load(Ordering::Relaxed) > 1;

        if self.heuristic_bound_transfer_enabled() && nontrivial_search {
            self.note_certification_issue("heuristic_support_bound_transfer_enabled");
        }
        if self.worker.profile.n_candidates_generated > 0 {
            self.note_certification_issue("candidate_generation_uses_top_b_truncation");
        }
        if self.search.used_ub_margin_prune.load(Ordering::Relaxed) {
            self.note_certification_issue("ub_margin_pruned_surviving_candidates");
        }
        if self
            .search
            .used_bfs_partial_materialization
            .load(Ordering::Relaxed)
        {
            self.note_certification_issue("bfs_materialized_only_a_subset_of_survivors");
        }
        if self
            .search
            .used_greedy_boundary_guidance
            .load(Ordering::Relaxed)
        {
            self.note_certification_issue("greedy_boundary_guidance_enabled");
        }
        if self.search.used_closure.load(Ordering::Relaxed) {
            self.note_certification_issue(format!(
                "closure_at_depth_{}",
                self.search.config.closure_depth.unwrap_or(0)
            ));
        }

        let witness_obj = self.witness_objective(&root_key);
        if (witness_obj - root_upperbound).abs() > 1e-8 {
            self.note_certification_issue(format!(
                "reported_upperbound_has_no_matching_tree_witness(root_ub={:.12}, witness={:.12})",
                root_upperbound, witness_obj
            ));
        }

        let issues = self.search.certification_issues.lock().unwrap();
        let certified_optimal =
            search_converged && !self.search.timed_out.load(Ordering::Acquire) && issues.is_empty();
        let certification_mode = if issues.is_empty() {
            "exact"
        } else {
            "exploratory"
        };
        let certification_reason = if certified_optimal {
            "certified".to_string()
        } else if issues.is_empty() {
            if self.search.timed_out.load(Ordering::Acquire) {
                "timed_out_before_certificate".to_string()
            } else if search_converged {
                "search_converged_without_certificate".to_string()
            } else {
                "search_not_converged".to_string()
            }
        } else {
            issues.join("; ")
        };
        let has_issues = !issues.is_empty();
        drop(issues);

        (
            certified_optimal,
            certification_mode.to_string(),
            certification_reason,
            has_issues,
        )
    }

    // ------------------------------------------------------------------
    // Time check
    // ------------------------------------------------------------------

    fn check_time(&mut self) -> bool {
        let over = match self.worker.rash_deadline {
            Some((start, limit)) => start.elapsed().as_secs_f64() > limit,
            None => match self.search.config.time_limit {
                Some(limit) => match *self.search.start_time.lock().unwrap() {
                    Some(start) => start.elapsed().as_secs_f64() > limit,
                    None => false,
                },
                None => false,
            },
        };
        if over {
            // `lb_threads` pass-1 workers never set `search.timed_out`:
            // the main thread alone decides the solve's time-out.
            match &self.worker.rash_time_sink {
                Some(sink) => sink.store(true, Ordering::Release),
                None => self.search.timed_out.store(true, Ordering::Release),
            }
        }
        over
    }

    /// `search.timed_out`, or the pass-1 abort flag on a `lb_threads`
    /// worker.
    #[inline]
    fn rash_timed_out(&self) -> bool {
        self.search.timed_out.load(Ordering::Acquire)
            || self
                .worker
                .rash_time_sink
                .as_ref()
                .map_or(false, |s| s.load(Ordering::Acquire))
    }

    /// `rash_deadline` of a solve started now (`start_time` set).
    fn rash_deadline_set(&mut self, start: Instant) {
        self.worker.rash_deadline = self.search.config.time_limit.map(|limit| (start, limit));
    }

    // ------------------------------------------------------------------
    // Priority queue
    // ------------------------------------------------------------------

    fn enqueue_explore(&mut self, key: MemoKey, gap: f64, depth: i32, requested_scope: f64) {
        if gap < 1e-10 {
            return;
        }
        if let Some(mut sp) = self.search.memo.get_mut(&key) {
            if requested_scope.is_finite() && requested_scope > sp.scope {
                sp.scope = requested_scope;
            }
        }
        // Guard dropped — safe to access queue
        let mut q = self.search.queue.lock().unwrap();
        q.push(WorkItem {
            neg_gap: -gap,
            is_bound_update: false,
            counter: self.search.counter.fetch_add(1, Ordering::Relaxed),
            key,
            depth,
            requested_scope,
            origin: None,
        });
        let qs = q.len();
        if qs > self.worker.profile.peak_queue_size {
            self.worker.profile.peak_queue_size = qs;
        }
    }

    // ------------------------------------------------------------------
    // Similar support bound
    // ------------------------------------------------------------------

    fn support_sig(sp: &SubProblem) -> SupportSig {
        SupportSig::new(sp.n_samples, &sp.class_counts[..sp.n_classes as usize])
    }

    fn similar_lb_lookup(&mut self, sp: &SubProblem) -> Option<f64> {
        let has_local = self.heuristic_bound_transfer_enabled();
        let has_shared = has_local && self.search.shared_bounds.is_some();
        if !has_local && !has_shared {
            return None;
        }
        let sig = Self::support_sig(sp);
        if has_local {
            if let Some(&val) = self.worker.similar_lb.get(&sig) {
                return Some(val);
            }
        }
        if has_shared {
            let shared = self.search.shared_bounds.as_ref().unwrap();
            // Depth-keyed lookup: an LB stored at depth D is only admissible
            // for queries at depth ≤ D (deeper allows lower optimum cost).
            // Key by sp.depth_remaining; cross-hop transfer is sound only
            // when both hops use the same effective depth at this support.
            let (lb_opt, _) = shared.lookup(sig.n_samples, sig.as_slice(), sp.depth);
            if let Some(raw_lb) = lb_opt {
                // Cross-call sharing: shared table stores values in
                // call-independent units (obj_local * n_total_call).
                // Convert back to current-call scale on read.
                let n_total = self.search.n_total as f64;
                let lb = if n_total > 0.0 {
                    raw_lb / n_total
                } else {
                    raw_lb
                };
                if has_local {
                    self.worker.similar_lb.insert(sig, lb);
                }
                return Some(lb);
            }
        }
        None
    }

    fn similar_lb_update(
        &mut self,
        n_samples: usize,
        class_counts: &[i64],
        depth_remaining: i32,
        lowerbound: f64,
    ) {
        let has_local = self.heuristic_bound_transfer_enabled();
        let has_shared = has_local && self.search.shared_bounds.is_some();
        if !has_local && !has_shared {
            return;
        }
        if n_samples == 0 {
            return;
        }
        if has_local {
            let sig = SupportSig::new(n_samples, class_counts);
            let current = self.worker.similar_lb.get(&sig).copied().unwrap_or(0.0);
            if lowerbound > current {
                self.worker.similar_lb.insert(sig, lowerbound);
            }
        }
        if has_shared {
            let shared = self.search.shared_bounds.as_ref().unwrap();
            // Cross-call sharing: convert to call-independent units before push.
            // Depth-keyed: bound is only admissible at queries with depth ≤ depth_remaining.
            let n_total = self.search.n_total as f64;
            let raw_lb = lowerbound * n_total;
            shared.update_lb(n_samples, class_counts, depth_remaining, raw_lb);
        }
    }

    fn similar_ub_lookup(&mut self, sp: &SubProblem) -> Option<f64> {
        // UB transfer via shared table is UNSOUND — a UB for one partition
        // doesn't apply to a different partition with the same SupportSig.
        // Only use local (per-solver) UB transfer.
        if !self.heuristic_bound_transfer_enabled() {
            return None;
        }
        let sig = Self::support_sig(sp);
        if let Some(&val) = self.worker.similar_ub.get(&sig) {
            return Some(val);
        }
        None
    }

    fn similar_ub_update(&mut self, n_samples: usize, class_counts: &[i64], upperbound: f64) {
        // UB transfer via shared table is UNSOUND — only use local.
        if !self.heuristic_bound_transfer_enabled() {
            return;
        }
        if n_samples == 0 {
            return;
        }
        let sig = SupportSig::new(n_samples, class_counts);
        let current = self
            .worker
            .similar_ub
            .get(&sig)
            .copied()
            .unwrap_or(f64::INFINITY);
        if upperbound < current {
            self.worker.similar_ub.insert(sig, upperbound);
        }
        // Note: shared UB transfer disabled — unsound across different partitions.
        // Any histogram-only transfer is explicitly heuristic: supports with
        // equal label totals can have different feature separability.
    }

    // ------------------------------------------------------------------
    // Fused closure candidate generation kernel
    // ------------------------------------------------------------------

    /// Fast candidate generation for closure's Greedy/DpBeam path.
    /// Fuses binary scan + feature pre-filter + optional Rush DP for k>2.
    /// Avoids: Vec clones from scratch, threshold interning, redundant
    /// binning of features that can't beat the leaf.
    fn closure_fast_cand_gen(
        &self,
        support: &SupportMask,
        feature_mask: &FeatureMask,
        indices: &[u32],
        class_counts: &[i64; MAX_CLASSES],
        leaf_obj: f64,
        closure_depth_remaining: i32,
        b1_filtered_out: &mut u64,
    ) -> Vec<SplitCandidate> {
        let store = self.search.store.as_ref();
        let config = &self.search.config;
        let n_classes = config.n_classes;
        let n_total = self.search.n_total;
        let n_total_f = n_total as f64;
        let alpha = config.reg_alpha;
        let reg = config.reg;
        let split_reg = (1.0 - alpha) * reg;
        let leaf_reg = alpha * reg;
        let n_weighted = class_counts[..n_classes].iter().sum::<i64>() as f64;
        let path_cost = config.path_penalty * n_weighted / n_total_f;
        let min_leaf = config.min_samples_leaf.max(1) as i64;
        let labels = store.labels();
        let max_k = config.max_k;
        let n_support = indices.len();
        let use_local_sort = n_support < n_total / 2;

        let total_n = class_counts[..n_classes].iter().sum::<i64>();
        let total_c0 = class_counts[0];
        let closure_crit = config.closure_criterion.unwrap_or(config.dp_criterion);
        // Integer-objective mode (n_total_override=Some(1))
        // requires the closure's emitted `split_ub` to be the TRUE
        // misclassification objective, integer-valued and directly comparable
        // to `leaf_obj` (also misclass). Gini scoring produces fractional,
        // impurity-valued split costs that systematically exceed the misclass
        // leaf at near-balanced nodes, so the
        // `binary_dominated = best_binary_cost >= leaf_obj` gate rejected EVERY
        // split and the depth-1 fast path (which trusts split_ub as the exact
        // cost) returned the leaf. Force misclass scoring under integer mode so
        // the objective stays integer and the leaf-vs-split comparison is sound.
        // EXACT/DAG and float-mode (n_total_override != Some(1)) paths keep
        // their Gini ranking unchanged.
        let int_obj_mode = config.n_total_override == Some(1);
        let use_gini = matches!(closure_crit, DpCriterion::Gini) && !int_obj_mode;
        let score = |child_n: &[i64], child_counts: &[Vec<i64>]| {
            let (rank_cost, split_ub) = score_candidate(
                child_n,
                child_counts,
                config,
                n_total,
                path_cost,
                CandidateOrder::Original,
            );
            (if int_obj_mode { split_ub } else { rank_cost }, split_ub)
        };

        // Small-support k=2 gate. When n_support is below the
        // threshold, force effective_max_k=2 (skip Rush-DP for k>2). Closure
        // is heuristic anyway and small supports rarely benefit from k≥3.
        let effective_max_k = match config.closure_min_n_for_multiway {
            Some(threshold) if threshold > 0 && n_support < threshold => 2,
            _ => max_k,
        };

        // Top-K feature filter. Build the iteration order from
        // the pre-ranked feature list when set; cap at top-K. Default falls
        // back to natural order 0..n_features.
        //
        // Depth-aware gating: when the closure has
        // 3+ levels remaining (i.e., it's effectively building most of the
        // tree), skip the K filter so closure-dominant searches get full
        // feature access; at depth ≥ 3 we keep all features. The filter
        // still fires at depth 1, 2 where closure is the tail of B&B.
        let n_features_full = config.n_features;
        let depth_gates_top_k = closure_depth_remaining < 3;
        let effective_top_k = if depth_gates_top_k {
            config.closure_feature_top_k
        } else {
            None
        };
        // Avoid per-call Vec alloc in the dominant (None, None) and
        // (Some, None) paths: borrow the ranking slice directly when
        // possible, allocate only when (None, Some) requires materializing
        // (0..k), which saves one Vec allocation per closure node in the
        // common case.
        let _feature_iter_owned: Vec<usize>;
        let feature_iter: &[usize] =
            match (config.closure_feature_ranking.as_ref(), effective_top_k) {
                (Some(ranking), Some(k)) => {
                    let k = k.min(n_features_full).max(1);
                    &ranking[..k.min(ranking.len())]
                }
                (Some(ranking), None) => &ranking[..],
                (None, Some(k)) => {
                    let k = k.min(n_features_full).max(1);
                    _feature_iter_owned = (0..k).collect();
                    &_feature_iter_owned
                }
                (None, None) => {
                    _feature_iter_owned = (0..n_features_full).collect();
                    &_feature_iter_owned
                }
            };

        // Inline cost: misclass or Gini, depending on criterion.
        #[inline(always)]
        fn bin_cost_misclass(c0: i64, n: i64, n_total_f: f64) -> f64 {
            let max_c = c0.max(n - c0);
            (n - max_c) as f64 / n_total_f
        }
        #[inline(always)]
        fn bin_cost_gini(c0: i64, n: i64, n_total_f: f64) -> f64 {
            if n == 0 {
                return 0.0;
            }
            let c1 = n - c0;
            let ssq = c0 * c0 + c1 * c1;
            (n as f64 - ssq as f64 / n as f64) / n_total_f
        }

        let mut results: Vec<SplitCandidate> = Vec::new();
        let mut sorted_buf: Vec<u32> = if use_local_sort {
            indices.to_vec()
        } else {
            Vec::new()
        };

        // Per-feature `irr_j + min_split_overhead` filter inside closure's
        // fused kernel. Mirrors the irreducible-error filter in
        // `generate_candidates_for_support_opts`.
        //
        // ADMISSIBLE at `closure_depth_remaining == 1` (children forced to
        // be leaves; irr_j bounds sum_leaf_loss exactly).
        //
        // LOSSY at `closure_depth_remaining > 1` — a deeper subtree using
        // a different feature can drop per-bin misclass below min(c0, c1),
        // so irr_j overcounts. Gated on
        // `closure_unsafe_filter_depth_threshold`:
        //   threshold = 0 (default): filter only at depth_remaining==1
        //   threshold >= depth_remaining: filter active (lossy past depth=1)
        let min_split_overhead = split_reg + 2.0 * leaf_reg + path_cost;
        let b1_threshold = config.closure_unsafe_filter_depth_threshold.max(1); // always allow at depth_remaining==1 (admissible)
        let b1_active = closure_depth_remaining <= b1_threshold;

        // At depth-1 closure nodes, scanning the full call-level sorted order
        // with a membership test costs O(n_total) per feature for the
        // irreducible-error filter. For small supports walk the node's own rows in the SAME global
        // sorted order instead (`sorted_support_indices` = the subsequence of
        // `sorted_order[j]` restricted to `indices`, ties and NaN included), so the
        // filter sees the identical (value, label) sequence and returns the
        // identical float. Large supports keep the full scan (cheaper there).
        let irr_support_sort = b1_active && use_local_sort;
        let mut irr_buf: Vec<u32> = if irr_support_sort {
            Vec::with_capacity(n_support)
        } else {
            Vec::new()
        };

        for &j in feature_iter {
            if j >= n_features_full {
                continue;
            }
            if config.feature_exclusion && !feature_mask.contains(j) {
                continue;
            }

            // Admissible filter: skip feature if its irreducible
            // error + min split overhead already meets/exceeds leaf_obj.
            // n_classes==2 uses the binary helper; n_classes>2 dispatches
            // to the multiclass twin (per-run `n - max_c`).
            if b1_active {
                let sorted_j: &[u32] = if irr_support_sort {
                    store.sorted_support_indices(j, indices, &mut irr_buf);
                    &irr_buf
                } else {
                    store.sorted_feature_order(j)
                };
                let irr_j = if n_classes == 2 {
                    crate::candidates::compute_feature_irreducible_error_binary_mask_pub(
                        sorted_j,
                        store.feature_col(j),
                        labels,
                        support,
                        n_total,
                    )
                } else {
                    crate::candidates::compute_feature_irreducible_error_multiclass_mask_pub(
                        sorted_j,
                        store.feature_col(j),
                        labels,
                        support,
                        n_classes,
                        n_total,
                    )
                };
                if irr_j + min_split_overhead >= leaf_obj - 1e-12 {
                    *b1_filtered_out += 1;
                    continue;
                }
            }

            let col = store.feature_col(j);
            let custom_thresholds = config
                .closure_thresholds_per_feature
                .as_ref()
                .or(config.thresholds_per_feature.as_ref())
                .and_then(|thresholds| thresholds.get(&j));
            if custom_thresholds.is_some_and(|thresholds| thresholds.is_empty()) {
                continue;
            }
            // The binning convention is x < threshold on the left. Only a
            // threshold in (previous_value, next_value] can separate this run.
            // An explicit reference-ensemble grid must never acquire raw midpoints here.
            let boundary_between = |previous: f64, next: f64| -> Option<f64> {
                match custom_thresholds {
                    Some(thresholds) => thresholds
                        .get(thresholds.partition_point(|&threshold| threshold <= previous))
                        .copied()
                        .filter(|&threshold| threshold <= next),
                    None => Some(previous * 0.5 + next * 0.5),
                }
            };

            // Sort support indices by this feature's values (O(n_support log n_support)).
            // For large supports, fall back to the global sorted order with bitmask check.
            if use_local_sort {
                sorted_buf.clear();
                sorted_buf.extend_from_slice(indices);
                sorted_buf.sort_unstable_by(|&a, &b| {
                    col[a as usize]
                        .partial_cmp(&col[b as usize])
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
            }

            let mut left_n: i64 = 0;
            let mut left_c0: i64 = 0;
            let mut prev_val = f64::NEG_INFINITY;
            let mut best_binary_cost = f64::INFINITY;
            let mut best_binary_thresh = 0.0f64;
            let mut best_left_n: i64 = 0;
            // Multiclass: track full per-class left-side counts at the best
            // split. `left_c0` is enough for n_classes==2 (right_c0 = total_c0 -
            // left_c0); for n_classes > 2 we need the full vector to preserve
            // child class distributions when emitting the candidate (otherwise
            // downstream code that indexes child_counts[ci][c] for c >= 1 OOBs).
            let mut best_left_counts: Vec<i64> = vec![0; n_classes];

            // Scan macro: iterate either local-sorted indices or global sorted order.
            macro_rules! scan_body {
                ($idx:expr, $val:expr) => {
                    if left_n > 0 && $val > prev_val + 1e-15 {
                        if let Some(boundary) = boundary_between(prev_val, $val) {
                            let right_n = total_n - left_n;
                            if left_n >= min_leaf && right_n >= min_leaf {
                                let right_c0_val = total_c0 - left_c0;
                                let cost = if use_gini {
                                    split_reg
                                        + path_cost
                                        + bin_cost_gini(left_c0, left_n, n_total_f)
                                        + leaf_reg
                                        + bin_cost_gini(right_c0_val, right_n, n_total_f)
                                        + leaf_reg
                                } else {
                                    split_reg
                                        + path_cost
                                        + bin_cost_misclass(left_c0, left_n, n_total_f)
                                        + leaf_reg
                                        + bin_cost_misclass(right_c0_val, right_n, n_total_f)
                                        + leaf_reg
                                };
                                if cost < best_binary_cost {
                                    best_binary_cost = cost;
                                    best_binary_thresh = boundary;
                                    best_left_n = left_n;
                                    // Binary path: best_left_counts has length 2.
                                    // Right-side count derived at emit time.
                                    best_left_counts[0] = left_c0;
                                    if n_classes >= 2 {
                                        best_left_counts[1] = left_n - left_c0;
                                    }
                                }
                            }
                        }
                    }
                };
            }

            if n_classes == 2 {
                if use_local_sort {
                    for &idx in &sorted_buf {
                        let val = col[idx as usize];
                        scan_body!(idx, val);
                        let wi = if let Some(w) = store.weights() {
                            (w[idx as usize] * self.search.n_data as f64).round() as i64
                        } else {
                            1
                        };
                        if labels[idx as usize] == 0 {
                            left_c0 += wi;
                        }
                        left_n += wi;
                        prev_val = val;
                    }
                } else {
                    let support_words = support.words.as_ref();
                    for &idx in store.sorted_feature_order(j) {
                        if !crate::candidates::support_contains_pub(support_words, idx) {
                            continue;
                        }
                        let val = col[idx as usize];
                        scan_body!(idx, val);
                        let wi = if let Some(w) = store.weights() {
                            (w[idx as usize] * self.search.n_data as f64).round() as i64
                        } else {
                            1
                        };
                        if labels[idx as usize] == 0 {
                            left_c0 += wi;
                        }
                        left_n += wi;
                        prev_val = val;
                    }
                }
            } else {
                let mut left_counts = [0i64; MAX_CLASSES];
                let scan_iter: Box<dyn Iterator<Item = u32>> =
                    if use_local_sort {
                        Box::new(sorted_buf.iter().copied())
                    } else {
                        let support_words = support.words.as_ref();
                        Box::new(store.sorted_feature_order(j).iter().copied().filter(
                            move |&idx| crate::candidates::support_contains_pub(support_words, idx),
                        ))
                    };
                for idx in scan_iter {
                    let val = col[idx as usize];
                    if left_n > 0 && val > prev_val + 1e-15 {
                        if let Some(boundary) = boundary_between(prev_val, val) {
                            let right_n = total_n - left_n;
                            if left_n >= min_leaf && right_n >= min_leaf {
                                let left_max = *left_counts[..n_classes].iter().max().unwrap();
                                let left_loss = if use_gini {
                                    let squares = left_counts[..n_classes]
                                        .iter()
                                        .map(|&count| (count as f64) * (count as f64))
                                        .sum::<f64>();
                                    (left_n as f64 - squares / left_n as f64) / n_total_f
                                } else {
                                    (left_n - left_max) as f64 / n_total_f
                                };
                                let mut right_counts = [0i64; MAX_CLASSES];
                                for c in 0..n_classes {
                                    right_counts[c] = class_counts[c] - left_counts[c];
                                }
                                let right_max = *right_counts[..n_classes].iter().max().unwrap();
                                let right_loss = if use_gini {
                                    let squares = right_counts[..n_classes]
                                        .iter()
                                        .map(|&count| (count as f64) * (count as f64))
                                        .sum::<f64>();
                                    (right_n as f64 - squares / right_n as f64) / n_total_f
                                } else {
                                    (right_n - right_max) as f64 / n_total_f
                                };
                                let cost = split_reg
                                    + path_cost
                                    + left_loss
                                    + leaf_reg
                                    + right_loss
                                    + leaf_reg;
                                if cost < best_binary_cost {
                                    best_binary_cost = cost;
                                    best_binary_thresh = boundary;
                                    best_left_n = left_n;
                                    for c in 0..n_classes {
                                        best_left_counts[c] = left_counts[c];
                                    }
                                }
                            }
                        }
                    }
                    let wi = store.weights().map_or(1, |weights| {
                        (weights[idx as usize] * store.n_samples() as f64).round() as i64
                    });
                    left_counts[labels[idx as usize] as usize] += wi;
                    left_n += wi;
                    prev_val = val;
                }
            }

            // Ranking may use Gini, but upper bounds and leaf comparisons
            // always use the misclassification objective.
            let left_max = best_left_counts.iter().copied().max().unwrap_or(0);
            let right_max = (0..n_classes)
                .map(|c| class_counts[c] - best_left_counts[c])
                .max()
                .unwrap_or(0);
            let binary_objective = split_reg
                + path_cost
                + 2.0 * leaf_reg
                + (total_n - left_max - right_max) as f64 / n_total_f;
            let binary_dominated = binary_objective >= leaf_obj;
            if best_binary_cost.is_finite() && (closure_depth_remaining > 1 || !binary_dominated) {
                let right_n = total_n - best_left_n;
                // Build full length-n_classes child_counts. For n_classes==2
                // this matches the legacy {[c0, n-c0], [right_c0, right_n-right_c0]}
                // emission; for n_classes>2 it preserves the per-class
                // distribution captured at the best split point so downstream
                // indexers (Patch-4 LB cache, top-k, child node init) don't
                // OOB.
                let _ = total_c0; // silence unused-warning when only used in n_classes==2 path historically
                let mut left_child_counts = vec![0i64; n_classes];
                let mut right_child_counts = vec![0i64; n_classes];
                for c in 0..n_classes {
                    left_child_counts[c] = best_left_counts[c];
                    right_child_counts[c] = class_counts[c] - best_left_counts[c];
                }
                let child_n = vec![best_left_n, right_n];
                let child_counts = vec![left_child_counts, right_child_counts];
                let (rank_cost, split_ub) = score(&child_n, &child_counts);
                results.push(SplitCandidate {
                    rank_cost,
                    split_ub,
                    k: 2,
                    feature_index: j,
                    thresholds: Arc::from(&[best_binary_thresh][..]),
                    child_n,
                    child_counts,
                    group_map: None,
                    family: crate::types::CAND_FAMILY_PRIMARY,
                    components: 2,
                });
            }

            // Skip k>2 only when max_k<=2 or binary dominates AND
            // n_support is small enough that multiway can't help
            // (need at least 3*min_leaf samples for a 3-way split).
            if max_k <= 2 || (binary_dominated && total_n < 3 * min_leaf) {
                continue;
            }

            // For k>2, bin and run Rush DP.
            // Gated by `effective_max_k` so small-support nodes
            // skip Rush-DP entirely when `closure_min_n_for_multiway` fires.
            if effective_max_k > 2 {
                self.worker.scratch_bank.with_worker(|scratch| {
                    scratch.clear_hot_buffers();
                    // Closure-internal candidate gen prefers
                    // `closure_thresholds_per_feature` (typically the
                    // LGBM-only subset) over the main `thresholds_per_feature`
                    // when set. This lets the heuristic closure path bin
                    // against fewer thresholds without affecting exact B&B.
                    let closure_thresh_map = config
                        .closure_thresholds_per_feature
                        .as_ref()
                        .or(config.thresholds_per_feature.as_ref());
                    let has_custom = closure_thresh_map.map_or(false, |m| m.contains_key(&j));

                    // Multiclass branch. Builds per-class bin counts and
                    // runs the multiclass `collapse_bins_topb` DP.
                    // `rush_dp` has no multiclass twin, so misclass is also
                    // routed through `collapse_bins_topb` (via DpCriterion::
                    // Misclass) — the multiclass `multiclass_cost(use_misclass=true)`
                    // formula is the hard-label score (per-bin
                    // `(n - max_c) / n_total`).
                    if n_classes > 2 {
                        let bin_counts: Vec<Vec<i64>> = if has_custom {
                            let custom_thresh = closure_thresh_map.and_then(|m| m.get(&j)).unwrap();
                            if custom_thresh.is_empty() {
                                return;
                            }
                            scratch.thresholds.clear();
                            scratch.thresholds.extend_from_slice(custom_thresh);
                            if use_local_sort {
                                scratch.bin_n.clear();
                                scratch.bin_n.resize(custom_thresh.len() + 1, 0);
                                let mut bc = vec![vec![0i64; n_classes]; custom_thresh.len() + 1];
                                let mut bin = 0usize;
                                for &idx in &sorted_buf {
                                    let val = col[idx as usize];
                                    while bin < custom_thresh.len() && custom_thresh[bin] <= val {
                                        bin += 1;
                                    }
                                    let wi = if let Some(w) = store.weights() {
                                        (w[idx as usize] * store.n_samples() as f64).round() as i64
                                    } else {
                                        1
                                    };
                                    scratch.bin_n[bin] += wi;
                                    bc[bin][labels[idx as usize] as usize] += wi;
                                }
                                bc
                            } else {
                                let bc =
                                    crate::candidates::fill_bin_samples_sorted_multiclass_mask_pub(
                                        store.sorted_feature_order(j),
                                        col,
                                        labels,
                                        custom_thresh,
                                        n_classes,
                                        support,
                                        store.weights(),
                                        store.n_samples() as f64,
                                        scratch,
                                    );
                                scratch.thresholds.clear();
                                scratch.thresholds.extend_from_slice(custom_thresh);
                                bc
                            }
                        } else if use_local_sort {
                            let mut cur_val = f64::NAN;
                            let mut cur_n = 0i64;
                            let mut cur_counts = vec![0i64; n_classes];
                            let mut bc: Vec<Vec<i64>> = Vec::new();
                            for &idx in &sorted_buf {
                                let val = col[idx as usize];
                                if cur_n > 0 && (val - cur_val).abs() > 1e-15 {
                                    scratch.bin_n.push(cur_n);
                                    bc.push(cur_counts.clone());
                                    scratch.thresholds.push((cur_val + val) / 2.0);
                                    cur_n = 0;
                                    cur_counts.iter_mut().for_each(|x| *x = 0);
                                }
                                cur_val = val;
                                let wi = if let Some(w) = store.weights() {
                                    (w[idx as usize] * store.n_samples() as f64).round() as i64
                                } else {
                                    1
                                };
                                cur_n += wi;
                                cur_counts[labels[idx as usize] as usize] += wi;
                            }
                            if cur_n > 0 {
                                scratch.bin_n.push(cur_n);
                                bc.push(cur_counts);
                            }
                            if scratch.thresholds.is_empty() {
                                return;
                            }
                            bc
                        } else {
                            match crate::candidates::fill_bins_from_sorted_multiclass_mask_pub(
                                store.sorted_feature_order(j),
                                col,
                                labels,
                                support,
                                n_classes,
                                10_000,
                                store.weights(),
                                store.n_samples() as f64,
                                scratch,
                            ) {
                                Some(bc) if !scratch.thresholds.is_empty() => bc,
                                _ => return,
                            }
                        };

                        let beam_schedule: Vec<usize> = (2..=max_k)
                            .map(|k| (1usize << (max_k - k)) * config.beam_width)
                            .collect();

                        // Atomize. Mirrors binary path (block compression
                        // over thresholds with identical class-distribution
                        // behavior under the chosen criterion).
                        let (dp_thresholds, dp_bin_n, dp_bin_counts): (
                            Vec<f64>,
                            Vec<i64>,
                            Vec<Vec<i64>>,
                        );
                        let (dp_th_ref, dp_bn_ref, dp_bc_ref): (&[f64], &[i64], &[Vec<i64>]) =
                            if config.msdt.atom_mode {
                                let (t, n, c, _proj) = crate::candidates::maybe_atomize_multiclass(
                                    config,
                                    scratch.thresholds.clone(),
                                    scratch.bin_n.clone(),
                                    bin_counts,
                                );
                                dp_thresholds = t;
                                dp_bin_n = n;
                                dp_bin_counts = c;
                                (&dp_thresholds, &dp_bin_n, &dp_bin_counts)
                            } else {
                                dp_thresholds = scratch.thresholds.clone();
                                dp_bin_n = scratch.bin_n.clone();
                                dp_bin_counts = bin_counts;
                                (&dp_thresholds, &dp_bin_n, &dp_bin_counts)
                            };

                        // Multiclass DP. `closure_crit` is forced to gini
                        // by the wrapper for n_classes>2 (rush has no twin),
                        // but we pass it through so misclass also lands here
                        // correctly.
                        let (ladder, _) = crate::candidates::collapse_bins_topb_min_child(
                            dp_th_ref,
                            dp_bn_ref,
                            dp_bc_ref,
                            max_k,
                            self.search.n_total,
                            &beam_schedule,
                            closure_crit,
                            None,
                            0.0, // leaf_reg not in the DP; cost added below
                            config.min_samples_leaf,
                        );
                        for (k, entry) in ladder {
                            if k <= 2 {
                                continue;
                            }
                            let (rank_cost, split_ub) = score(&entry.bin_n, &entry.bin_counts);
                            let components =
                                crate::candidates::components_count(k, entry.group_map.as_deref());
                            results.push(SplitCandidate {
                                rank_cost,
                                split_ub,
                                k,
                                feature_index: j,
                                thresholds: Arc::from(entry.thresholds.as_ref()),
                                child_n: entry.bin_n,
                                child_counts: entry.bin_counts,
                                group_map: entry.group_map,
                                family: crate::types::CAND_FAMILY_PRIMARY,
                                components,
                            });
                        }
                        return;
                    }

                    if has_custom {
                        let custom_thresh = closure_thresh_map.and_then(|m| m.get(&j)).unwrap();
                        if custom_thresh.is_empty() {
                            return;
                        }
                        scratch.thresholds.clear();
                        scratch.thresholds.extend_from_slice(custom_thresh);
                        scratch.bin_n.clear();
                        scratch.bin_c0.clear();
                        scratch.bin_n.resize(custom_thresh.len() + 1, 0);
                        scratch.bin_c0.resize(custom_thresh.len() + 1, 0);
                        if use_local_sort {
                            let mut bin = 0usize;
                            for &idx in &sorted_buf {
                                let val = col[idx as usize];
                                while bin < custom_thresh.len() && custom_thresh[bin] <= val {
                                    bin += 1;
                                }
                                let wi = if let Some(w) = store.weights() {
                                    (w[idx as usize] * store.n_samples() as f64).round() as i64
                                } else {
                                    1
                                };
                                scratch.bin_n[bin] += wi;
                                if labels[idx as usize] == 0 {
                                    scratch.bin_c0[bin] += wi;
                                }
                            }
                        } else {
                            crate::candidates::fill_bin_samples_sorted_binary_mask_pub(
                                store.sorted_feature_order(j),
                                col,
                                labels,
                                custom_thresh,
                                support,
                                store.weights(),
                                store.n_samples() as f64,
                                scratch,
                            );
                            scratch.thresholds.clear();
                            scratch.thresholds.extend_from_slice(custom_thresh);
                        }
                    } else if use_local_sort {
                        let mut cur_val = f64::NAN;
                        let mut cur_n = 0i64;
                        let mut cur_c0 = 0i64;
                        for &idx in &sorted_buf {
                            let val = col[idx as usize];
                            if cur_n > 0 && (val - cur_val).abs() > 1e-15 {
                                scratch.bin_n.push(cur_n);
                                scratch.bin_c0.push(cur_c0);
                                scratch.thresholds.push((cur_val + val) / 2.0);
                                cur_n = 0;
                                cur_c0 = 0;
                            }
                            cur_val = val;
                            let wi = if let Some(w) = store.weights() {
                                (w[idx as usize] * store.n_samples() as f64).round() as i64
                            } else {
                                1
                            };
                            cur_n += wi;
                            if labels[idx as usize] == 0 {
                                cur_c0 += wi;
                            }
                        }
                        if cur_n > 0 {
                            scratch.bin_n.push(cur_n);
                            scratch.bin_c0.push(cur_c0);
                        }
                        if scratch.thresholds.is_empty() {
                            return;
                        }
                    } else {
                        let ok = crate::candidates::fill_bins_from_sorted_binary_mask_pub(
                            store.sorted_feature_order(j),
                            col,
                            labels,
                            support,
                            10_000,
                            store.weights(),
                            store.n_samples() as f64,
                            scratch,
                        );
                        if ok.is_none() || scratch.thresholds.is_empty() {
                            return;
                        }
                    }
                    let beam_schedule: Vec<usize> = (2..=max_k)
                        .map(|k| (1usize << (max_k - k)) * config.beam_width)
                        .collect();

                    // Closure-path atomization (parallels `maybe_atomize_binary`
                    // in the top-level candidate generator).
                    // When config.msdt.atom_mode is on, merge adjacent bins
                    // into atoms before feeding the DP, which shrinks the bin
                    // count m on continuous features. Mirrors top-level B&B's
                    // behavior.
                    let (dp_thresholds, dp_bin_n, dp_bin_c0): (Vec<f64>, Vec<i64>, Vec<i64>);
                    let (dp_th_ref, dp_bn_ref, dp_bc0_ref): (&[f64], &[i64], &[i64]) =
                        if config.msdt.atom_mode {
                            let (t, n, c0, _proj) = crate::candidates::maybe_atomize_binary(
                                config,
                                scratch.thresholds.clone(),
                                scratch.bin_n.clone(),
                                scratch.bin_c0.clone(),
                            );
                            dp_thresholds = t;
                            dp_bin_n = n;
                            dp_bin_c0 = c0;
                            (&dp_thresholds, &dp_bin_n, &dp_bin_c0)
                        } else {
                            (&scratch.thresholds, &scratch.bin_n, &scratch.bin_c0)
                        };

                    let ladder = if use_gini {
                        crate::candidates::collapse_bins_heuristic_gini_binary(
                            dp_th_ref,
                            dp_bn_ref,
                            dp_bc0_ref,
                            max_k,
                            self.search.n_total,
                            leaf_reg,
                            config.min_samples_leaf,
                        )
                    } else {
                        let (l, _) = crate::candidates::rush_dp_binary_pub(
                            dp_th_ref,
                            dp_bn_ref,
                            dp_bc0_ref,
                            max_k,
                            self.search.n_total,
                            &beam_schedule,
                            config.min_samples_leaf,
                            leaf_obj,
                            reg,
                            alpha,
                            Some(&mut scratch.rush_dp),
                        );
                        l
                    };
                    for (k, entry) in ladder {
                        if k <= 2 {
                            continue; // already have binary from fused scan
                        }
                        let (rank_cost, split_ub) = score(&entry.bin_n, &entry.bin_counts);
                        let components =
                            crate::candidates::components_count(k, entry.group_map.as_deref());
                        results.push(SplitCandidate {
                            rank_cost,
                            split_ub,
                            k,
                            feature_index: j,
                            thresholds: Arc::from(entry.thresholds.as_ref()),
                            child_n: entry.bin_n,
                            child_counts: entry.bin_counts,
                            group_map: entry.group_map,
                            family: crate::types::CAND_FAMILY_PRIMARY,
                            components,
                        });
                    }
                });
            }
        }

        results
    }

    // ------------------------------------------------------------------
    // Greedy CART UB: cheap binary greedy for SubProblem UB tightening
    // ------------------------------------------------------------------

    fn greedy_cart_leaf_cost(&self, indices: &[u32]) -> f64 {
        let n = indices.len();
        if n == 0 {
            return self.search.config.reg_alpha * self.search.config.reg;
        }
        let labels = self.search.store.labels();
        let n_classes = self.search.config.n_classes;
        let n_total = self.search.n_total as f64;
        let leaf_reg = self.search.config.reg_alpha * self.search.config.reg;

        if let Some(w) = self.search.store.weights() {
            let scale = self.search.n_data as f64;
            let mut counts = vec![0.0f64; n_classes];
            for &idx in indices {
                counts[labels[idx as usize] as usize] += w[idx as usize] * scale;
            }
            let n_weighted: f64 = counts.iter().sum();
            let max_count = counts.iter().cloned().fold(0.0f64, f64::max);
            (n_weighted - max_count) / n_total + leaf_reg
        } else {
            let mut counts = [0i64; MAX_CLASSES];
            for &idx in indices {
                counts[labels[idx as usize] as usize] += 1;
            }
            let n_weighted = counts[..n_classes].iter().sum::<i64>() as f64;
            let max_count = *counts[..n_classes].iter().max().unwrap() as f64;
            (n_weighted - max_count) / n_total + leaf_reg
        }
    }

    /// Greedy binary CART expansion returning a valid UB for the subtree.
    /// Uses incremental class counting on the pre-sorted feature order to
    /// evaluate ALL midpoint thresholds in O(n) per feature. At depth > 1,
    /// picks the best feature+threshold, partitions once, and recurses.
    fn greedy_cart_ub(&self, indices: &[u32], depth: i32) -> f64 {
        let n = indices.len();
        if n <= 1 || depth <= 0 {
            return self.greedy_cart_leaf_cost(indices);
        }

        let leaf_cost = self.greedy_cart_leaf_cost(indices);
        let leaf_reg = self.search.config.reg_alpha * self.search.config.reg;
        if leaf_cost <= leaf_reg + 1e-14 {
            return leaf_cost;
        }

        let labels = self.search.store.labels();
        let n_classes = self.search.config.n_classes;
        let n_total = self.search.n_total as f64;
        let split_reg = (1.0 - self.search.config.reg_alpha) * self.search.config.reg;
        let n_weighted_f = if let Some(w) = self.search.store.weights() {
            let scale = self.search.n_data as f64;
            indices.iter().map(|&i| w[i as usize] * scale).sum::<f64>()
        } else {
            n as f64
        };
        let path_cost = self.search.config.path_penalty * n_weighted_f / n_total;

        let in_support = {
            let mut bits = vec![false; self.search.store.n_samples()];
            for &idx in indices {
                bits[idx as usize] = true;
            }
            bits
        };

        let mut total_counts = [0i64; MAX_CLASSES];
        for &idx in indices {
            total_counts[labels[idx as usize] as usize] += 1;
        }

        let n_features = self.search.config.n_features;
        let mut best_cost = leaf_cost;
        let mut best_feature = 0usize;
        let mut best_threshold = 0.0f64;

        for j in 0..n_features {
            let sorted_order = self.search.store.sorted_feature_order(j);
            let col = self.search.store.feature_col(j);

            let mut left_counts = [0i64; MAX_CLASSES];
            let mut left_n: usize = 0;
            let mut prev_val = f64::NEG_INFINITY;

            for &idx in sorted_order {
                if !in_support[idx as usize] {
                    continue;
                }
                let val = col[idx as usize];

                if left_n > 0 && val > prev_val + 1e-14 {
                    let right_n = n - left_n;
                    if right_n >= 1 {
                        let left_max = *left_counts[..n_classes].iter().max().unwrap() as f64;
                        let left_total = left_counts[..n_classes].iter().sum::<i64>() as f64;
                        let left_leaf = (left_total - left_max) / n_total + leaf_reg;

                        let mut right_counts = [0i64; MAX_CLASSES];
                        for c in 0..n_classes {
                            right_counts[c] = total_counts[c] - left_counts[c];
                        }
                        let right_max = *right_counts[..n_classes].iter().max().unwrap() as f64;
                        let right_total = right_counts[..n_classes].iter().sum::<i64>() as f64;
                        let right_leaf = (right_total - right_max) / n_total + leaf_reg;

                        let cost = split_reg + path_cost + left_leaf + right_leaf;
                        if cost < best_cost {
                            best_cost = cost;
                            best_feature = j;
                            best_threshold = (prev_val + val) / 2.0;
                        }
                    }
                }

                left_counts[labels[idx as usize] as usize] += 1;
                left_n += 1;
                prev_val = val;
            }
        }

        if best_cost >= leaf_cost - 1e-14 {
            return leaf_cost;
        }

        if depth <= 1 {
            return best_cost;
        }

        let col = self.search.store.feature_col(best_feature);
        let (left, right): (Vec<u32>, Vec<u32>) = indices
            .iter()
            .partition(|&&idx| col[idx as usize] <= best_threshold);

        if left.is_empty() || right.is_empty() {
            return leaf_cost;
        }

        let left_cost = self.greedy_cart_ub(&left, depth - 1);
        let right_cost = self.greedy_cart_ub(&right, depth - 1);
        let recursive_cost = split_reg + path_cost + left_cost + right_cost;

        recursive_cost.min(leaf_cost)
    }

    // ------------------------------------------------------------------
    // SubProblem creation
    // ------------------------------------------------------------------

    /// Minimal terminal SubProblem for closure_build's structural
    /// fast-leaf path. Matches the terminal branch of create_subproblem
    /// but skips similar_lb/ub updates (perf: these run per-node across
    /// 1M+ closure invocations and are caches, not correctness-critical).
    fn build_closure_leaf(
        &self,
        support: &SupportMask,
        feature_mask: &FeatureMask,
        depth: i32,
    ) -> SubProblem {
        self.search.n_subproblems.fetch_add(1, Ordering::Relaxed);
        let n = support.n_samples;
        let n_classes = self.search.config.n_classes;
        let reg = self.search.config.reg;
        let alpha = self.search.config.reg_alpha;
        let leaf_reg = alpha * reg;

        if n == 0 {
            return SubProblem {
                support: support.clone(),
                split_path: Vec::new(),
                feature_mask: feature_mask.clone(),
                n_samples: 0,
                class_counts: [0i64; MAX_CLASSES],
                n_classes: n_classes as u8,
                max_loss: 0.0,
                base_objective: leaf_reg,
                lowerbound: leaf_reg,
                upperbound: leaf_reg,
                upperbound_is_heuristic: false,
                phase_a_lb_floor: leaf_reg,
                last_survivor_count: 0,
                all_survivors_materialized: true,
                is_terminal: true,
                is_solved: true,
                branch_bound_scope_floor: f64::NEG_INFINITY,
                scope: f64::NEG_INFINITY,
                explored: false,
                depth,
                bound_update_pending: false,
                rashomon_indices: Vec::new(),
            };
        }

        let mut class_counts = [0i64; MAX_CLASSES];
        if let Some(w) = self.search.store.weights() {
            let scale = self.search.n_data as f64;
            for idx in self.search.store.support_to_indices(support) {
                class_counts[self.search.store.labels()[idx as usize] as usize] +=
                    (w[idx as usize] * scale).round() as i64;
            }
        } else {
            class_counts = self.search.store.count_classes(support).0;
        }
        let max_count = *class_counts[..n_classes].iter().max().unwrap();
        let n_weighted: f64 = class_counts[..n_classes].iter().sum::<i64>() as f64;
        let n_total = self.search.n_total;
        let max_loss = (n_weighted - max_count as f64) / n_total as f64;
        let base_obj = max_loss + leaf_reg;

        SubProblem {
            support: support.clone(),
            split_path: Vec::new(),
            feature_mask: feature_mask.clone(),
            n_samples: n,
            class_counts,
            n_classes: n_classes as u8,
            max_loss,
            base_objective: base_obj,
            lowerbound: base_obj,
            upperbound: base_obj,
            upperbound_is_heuristic: false,
            phase_a_lb_floor: base_obj,
            last_survivor_count: 0,
            all_survivors_materialized: true,
            is_terminal: true,
            is_solved: true,
            branch_bound_scope_floor: f64::NEG_INFINITY,
            scope: f64::NEG_INFINITY,
            explored: false,
            depth,
            bound_update_pending: false,
            rashomon_indices: Vec::new(),
        }
    }

    /// Build a terminal-leaf SubProblem from pre-computed class
    /// counts (from the parent's DP-collapse `cand.child_counts`), skipping
    /// the `support_to_indices` walk that `build_closure_leaf` does.
    /// Caller must guarantee `class_counts` matches the actual partition; the
    /// short-circuit gate (`leaf_cost <= split_reg + 2·leaf_reg + bin_path`)
    /// also guarantees the resulting leaf is provably optimal — no split can
    /// beat it. Leaves do not incur an internal-node path penalty.
    fn build_closure_terminal_with_counts(
        &self,
        support: &SupportMask,
        feature_mask: &FeatureMask,
        depth: i32,
        n_samples: usize,
        class_counts: [i64; MAX_CLASSES],
    ) -> SubProblem {
        self.search.n_subproblems.fetch_add(1, Ordering::Relaxed);
        let n_classes = self.search.config.n_classes;
        let leaf_reg = self.search.config.reg_alpha * self.search.config.reg;
        let max_count = *class_counts[..n_classes].iter().max().unwrap_or(&0);
        let n_weighted: f64 = class_counts[..n_classes].iter().sum::<i64>() as f64;
        let n_total = self.search.n_total;
        let max_loss = (n_weighted - max_count as f64) / n_total as f64;
        let base_obj = max_loss + leaf_reg;

        SubProblem {
            support: support.clone(),
            split_path: Vec::new(),
            feature_mask: feature_mask.clone(),
            n_samples,
            class_counts,
            n_classes: n_classes as u8,
            max_loss,
            base_objective: base_obj,
            lowerbound: base_obj,
            upperbound: base_obj,
            upperbound_is_heuristic: false,
            phase_a_lb_floor: base_obj,
            last_survivor_count: 0,
            all_survivors_materialized: true,
            is_terminal: true,
            is_solved: true,
            branch_bound_scope_floor: f64::NEG_INFINITY,
            scope: f64::NEG_INFINITY,
            explored: false,
            depth,
            bound_update_pending: false,
            rashomon_indices: Vec::new(),
        }
    }

    fn create_subproblem(
        &mut self,
        support: &SupportMask,
        feature_mask: &FeatureMask,
        depth: i32,
    ) -> SubProblem {
        self.search.n_subproblems.fetch_add(1, Ordering::Relaxed);
        let n = support.n_samples;
        let n_classes = self.search.config.n_classes;
        let reg = self.search.config.reg;

        // reg_alpha controls how reg scales with arity:
        //   leaf_reg = alpha * reg   (each leaf pays this)
        //   split_reg = (1-alpha) * reg  (each split node pays this)
        //   Total for K-way = K * leaf_reg + split_reg = reg * (1 + (K-1)*alpha)
        let alpha = self.search.config.reg_alpha;
        let leaf_reg = alpha * reg;

        if n == 0 {
            return SubProblem {
                support: support.clone(),
                split_path: Vec::new(),
                feature_mask: feature_mask.clone(),
                n_samples: 0,
                class_counts: [0i64; MAX_CLASSES],
                n_classes: n_classes as u8,
                max_loss: 0.0,
                base_objective: leaf_reg,
                lowerbound: leaf_reg,
                upperbound: leaf_reg,
                upperbound_is_heuristic: false,
                phase_a_lb_floor: leaf_reg,
                last_survivor_count: 0,
                all_survivors_materialized: true,
                is_terminal: true,
                is_solved: true,
                branch_bound_scope_floor: f64::NEG_INFINITY,
                scope: f64::NEG_INFINITY,
                explored: false,
                depth,
                bound_update_pending: false,
                rashomon_indices: Vec::new(),
            };
        }

        let mut class_counts = [0i64; MAX_CLASSES];
        if let Some(w) = self.search.store.weights() {
            let scale = self.search.n_data as f64;
            for idx in self.search.store.support_to_indices(support) {
                class_counts[self.search.store.labels()[idx as usize] as usize] +=
                    (w[idx as usize] * scale).round() as i64;
            }
        } else {
            class_counts = self.search.store.count_classes(support).0;
        }
        let max_count = *class_counts[..n_classes].iter().max().unwrap();
        let n_weighted: f64 = class_counts[..n_classes].iter().sum::<i64>() as f64;
        let n_total = self.search.n_total;
        let max_loss = (n_weighted - max_count as f64) / n_total as f64;

        let base_obj = max_loss + leaf_reg;
        let split_reg = (1.0 - alpha) * reg;
        let path_cost = self.search.config.path_penalty * n_weighted / n_total as f64;
        let trivial_lb = base_obj.min(split_reg + 2.0 * leaf_reg + path_cost);
        // Equivalent-points LB: admissible depth-free misclass floor from
        // rows with identical bucketized feature vectors. See
        // DatasetStore::construct_majority_bitmask for construction details.
        let eqp = if self.eqp_lb_enabled() {
            self.search.store.eqp_misclass(support, n_total)
        } else {
            0.0
        };
        let eqp_lb = base_obj.min(eqp + 2.0 * leaf_reg + path_cost);
        // Reference-model LB: additional tightener (0.0 when
        // reference_preds wasn't provided). NOT strictly admissible — user
        // is responsible for ensuring the reference model's misclass on
        // every subset ≤ optimum tree. See reference_misclass docstring.
        let ref_miss = self.search.store.reference_misclass(support, n_total);
        let ref_lb = base_obj.min(ref_miss + 2.0 * leaf_reg + path_cost);
        let mut lowerbound = trivial_lb.max(eqp_lb).max(ref_lb);

        let mut is_terminal = false;
        if max_loss == 0.0 {
            is_terminal = true;
            lowerbound = leaf_reg;
        } else if max_loss < split_reg + leaf_reg + path_cost {
            // Splitting can't improve: best split gives split_reg + 2*leaf_reg + path_cost
            // which exceeds leaf cost of max_loss + leaf_reg.
            is_terminal = true;
        } else if n <= 1 {
            is_terminal = true;
        }

        if is_terminal {
            lowerbound = base_obj;
            let sp = SubProblem {
                support: support.clone(),
                split_path: Vec::new(),
                feature_mask: feature_mask.clone(),
                n_samples: n,
                class_counts,
                n_classes: n_classes as u8,
                max_loss,
                base_objective: base_obj,
                lowerbound,
                upperbound: base_obj,
                upperbound_is_heuristic: false,
                phase_a_lb_floor: lowerbound,
                last_survivor_count: 0,
                all_survivors_materialized: true,
                is_terminal: true,
                is_solved: true,
                branch_bound_scope_floor: f64::NEG_INFINITY,
                scope: f64::NEG_INFINITY,
                explored: false,
                depth,
                bound_update_pending: false,
                rashomon_indices: Vec::new(),
            };
            self.similar_lb_update(n, &class_counts[..n_classes], depth, lowerbound);
            self.similar_ub_update(n, &class_counts[..n_classes], base_obj);
            return sp;
        }

        let mut sp = SubProblem {
            support: support.clone(),
            split_path: Vec::new(),
            feature_mask: feature_mask.clone(),
            n_samples: n,
            class_counts,
            n_classes: n_classes as u8,
            max_loss,
            base_objective: base_obj,
            lowerbound,
            upperbound: base_obj,
            upperbound_is_heuristic: false,
            phase_a_lb_floor: lowerbound,
            last_survivor_count: 0,
            all_survivors_materialized: false,
            is_terminal: false,
            is_solved: false,
            branch_bound_scope_floor: f64::NEG_INFINITY,
            scope: f64::NEG_INFINITY,
            explored: false,
            depth,
            bound_update_pending: false,
            rashomon_indices: Vec::new(),
        };

        if let Some(sim_lb) = self.similar_lb_lookup(&sp) {
            if sim_lb > sp.lowerbound {
                sp.lowerbound = sim_lb.min(sp.base_objective);
                if sp.lowerbound >= sp.upperbound - 1e-10 {
                    sp.is_terminal = true;
                    sp.is_solved = true;
                    sp.lowerbound = sp.upperbound;
                }
            }
        }

        // Also check similar UB (from previously solved identical subproblems)
        if let Some(sim_ub) = self.similar_ub_lookup(&sp) {
            if sim_ub < sp.upperbound {
                sp.upperbound = sim_ub;
                sp.upperbound_is_heuristic = false;
            }
            if sp.upperbound - sp.lowerbound < 1e-10 {
                sp.is_solved = true;
            }
        }

        sp
    }

    fn get_or_create(
        &mut self,
        support: &SupportMask,
        feature_mask: &FeatureMask,
        depth: i32,
    ) -> MemoKey {
        let key = MemoKey::new(
            support.clone(),
            feature_mask.clone(),
            depth,
            Self::mode_flags(&self.search.config),
            Self::teacher_alpha_memo_bits(&self.search.config),
        );
        if self.search.memo.contains_key(&key) {
            return key;
        }

        // Closure: below the configured depth, build the full local subtree.
        if let Some(closure_depth) = self.search.config.closure_depth {
            if depth <= closure_depth && depth > 0 {
                let is_entry = depth == closure_depth;
                let t_cc = if is_entry { Some(Instant::now()) } else { None };
                let (k, _obj) = self.closure_build(support, feature_mask, depth);
                if let Some(t) = t_cc {
                    self.worker.profile.time_closure += t.elapsed();
                }
                self.search.used_closure.store(true, Ordering::Relaxed);
                return k;
            }
        }

        let mut sp = self.create_subproblem(support, feature_mask, depth);
        if self.search.config.rashomon_mode {
            self.rash_explore_node_state(&mut sp, depth);
        } else if depth == 0 && !sp.is_terminal {
            sp.lowerbound = sp.base_objective;
            sp.upperbound = sp.base_objective;
            sp.upperbound_is_heuristic = false;
            sp.is_terminal = true;
            sp.is_solved = true;
        }
        self.search.memo.insert(key.clone(), sp);
        key
    }

    /// Create a child SubProblem with a split_path attached.
    /// The split_path is parent_path + new_step.
    fn get_or_create_with_path(
        &mut self,
        support: &SupportMask,
        feature_mask: &FeatureMask,
        depth: i32,
        parent_path: &[SplitStep],
        step: SplitStep,
    ) -> MemoKey {
        let key = MemoKey::new(
            support.clone(),
            feature_mask.clone(),
            depth,
            Self::mode_flags(&self.search.config),
            Self::teacher_alpha_memo_bits(&self.search.config),
        );
        if self.search.memo.contains_key(&key) {
            return key;
        }

        // Closure: dispatch to closure_build (split_path not needed for
        // closed subtrees since they're immediately solved).
        if let Some(closure_depth) = self.search.config.closure_depth {
            if depth <= closure_depth && depth > 0 {
                let is_entry = depth == closure_depth;
                let t_cc = if is_entry { Some(Instant::now()) } else { None };
                let (k, _obj) = self.closure_build(support, feature_mask, depth);
                if let Some(t) = t_cc {
                    self.worker.profile.time_closure += t.elapsed();
                }
                self.search.used_closure.store(true, Ordering::Relaxed);
                return k;
            }
        }

        let mut sp = self.create_subproblem(support, feature_mask, depth);
        let mut path = parent_path.to_vec();
        path.push(step);
        sp.split_path = path;
        if self.search.config.rashomon_mode {
            self.rash_explore_node_state(&mut sp, depth);
        } else if depth == 0 && !sp.is_terminal {
            sp.lowerbound = sp.base_objective;
            sp.upperbound = sp.base_objective;
            sp.upperbound_is_heuristic = false;
            sp.is_terminal = true;
            sp.is_solved = true;
        }
        self.search.memo.insert(key.clone(), sp);
        key
    }

    // ------------------------------------------------------------------
    // Lightweight inline bounds (no memo insertion, no heap alloc for indices)
    // ------------------------------------------------------------------

    /// Compute (lowerbound, upperbound) for a child partition WITHOUT creating
    /// a SubProblem or inserting into the memo table. Used during candidate
    /// evaluation to avoid materializing indices for non-winning candidates.
    ///
    /// Returns (lowerbound, upperbound). Both are valid bounds using the same
    /// logic as create_subproblem, including similar_lb/similar_ub lookups.
    fn compute_bounds_inline(&mut self, indices: &[u32], depth: i32) -> (f64, f64) {
        self.worker.profile.n_inline_bounds_calls += 1;
        let n = indices.len();
        let n_classes = self.search.config.n_classes;
        let reg = self.search.config.reg;
        let alpha = self.search.config.reg_alpha;
        let leaf_reg = alpha * reg;
        let split_reg = (1.0 - alpha) * reg;
        let n_total_f = self.search.n_total as f64;

        if n == 0 {
            return (leaf_reg, leaf_reg);
        }

        let mut class_counts_for_sig = [0i64; MAX_CLASSES];
        if let Some(w) = self.search.store.weights() {
            let scale = self.search.n_data as f64;
            for &idx in indices {
                class_counts_for_sig[self.search.store.labels()[idx as usize] as usize] +=
                    (w[idx as usize] * scale).round() as i64;
            }
        } else {
            for &idx in indices {
                class_counts_for_sig[self.search.store.labels()[idx as usize] as usize] += 1;
            }
        }
        let max_count = *class_counts_for_sig[..n_classes].iter().max().unwrap();
        let n_weighted: f64 = class_counts_for_sig[..n_classes].iter().sum::<i64>() as f64;

        let max_loss = (n_weighted - max_count as f64) / n_total_f;
        let base_obj = max_loss + leaf_reg;
        let path_cost = self.search.config.path_penalty * n_weighted / n_total_f;

        // Terminal checks (same logic as create_subproblem)
        if max_loss == 0.0 || max_loss < split_reg + leaf_reg + path_cost || n <= 1 || depth <= 0 {
            return (base_obj, base_obj);
        }

        let mut lowerbound = base_obj.min(split_reg + 2.0 * leaf_reg + path_cost);
        let mut upperbound = base_obj;

        // Equivalent-points (per-support EC unavoidable-misclass) LB.
        // Lossless tightener: every tree must misclassify at least
        // (n_in_EC - max_class_count_in_EC) per equivalence class for any
        // support, regardless of depth or split structure. Multiclass-correct.
        if self.search.config.eqp_inline_bounds && self.eqp_lb_enabled() {
            let eqp = self
                .search
                .store
                .eqp_misclass_per_support_indices(indices, self.search.n_total);
            let eqp_lb = base_obj.min(eqp + 2.0 * leaf_reg + path_cost);
            if eqp_lb > lowerbound {
                lowerbound = eqp_lb;
                if lowerbound >= upperbound - 1e-10 {
                    return (upperbound, upperbound);
                }
            }
        }

        // Similar support lookups — skip when weighted (continuous counts
        // make SupportSig collisions meaningless)
        if self.heuristic_bound_transfer_enabled() {
            let sig = SupportSig::new(
                n_weighted.round() as usize,
                &class_counts_for_sig[..n_classes],
            );
            match self.worker.similar_lb.get(&sig) {
                Some(&val) if val > lowerbound => {
                    self.worker.profile.n_similar_lb_hits += 1;
                    lowerbound = val.min(base_obj);
                    if lowerbound >= upperbound - 1e-10 {
                        return (upperbound, upperbound);
                    }
                }
                Some(_) => {
                    self.worker.profile.n_similar_lb_misses += 1;
                }
                None => {
                    self.worker.profile.n_similar_lb_misses += 1;
                }
            }
            match self.worker.similar_ub.get(&sig) {
                Some(&val) if val < upperbound => {
                    self.worker.profile.n_similar_ub_hits += 1;
                    upperbound = val;
                    if upperbound - lowerbound < 1e-10 {
                        return (lowerbound, upperbound);
                    }
                }
                Some(_) => {
                    self.worker.profile.n_similar_ub_misses += 1;
                }
                None => {
                    self.worker.profile.n_similar_ub_misses += 1;
                }
            }

            if let Some(ref shared) = self.search.shared_bounds {
                let (lb_opt, ub_opt) = shared.lookup(n, &class_counts_for_sig[..n_classes], depth);
                let n_total_f = self.search.n_total as f64;
                if let Some(raw_lb) = lb_opt {
                    let lb = if n_total_f > 0.0 {
                        raw_lb / n_total_f
                    } else {
                        raw_lb
                    };
                    if lb > lowerbound {
                        lowerbound = lb.min(base_obj);
                    }
                }
                if let Some(raw_ub) = ub_opt {
                    let ub = if n_total_f > 0.0 {
                        raw_ub / n_total_f
                    } else {
                        raw_ub
                    };
                    if ub < upperbound {
                        upperbound = ub;
                    }
                }
            }
        }

        (lowerbound, upperbound)
    }

    /// Compute (lowerbound, upperbound) from pre-computed class counts.
    /// Same logic as compute_bounds_inline but skips the O(n) counting loop.
    fn compute_bounds_from_counts(
        &mut self,
        n: usize,
        class_counts: &[i64],
        depth: i32,
    ) -> (f64, f64) {
        self.worker.profile.n_inline_bounds_calls += 1;
        let n_classes = self.search.config.n_classes;
        let reg = self.search.config.reg;
        let n_total_f = self.search.n_total as f64;
        let alpha = self.search.config.reg_alpha;
        let leaf_reg = alpha * reg;
        let split_reg = (1.0 - alpha) * reg;
        let path_cost = self.search.config.path_penalty * n as f64 / n_total_f;

        if n == 0 {
            return (leaf_reg, leaf_reg);
        }

        let max_count = *class_counts[..n_classes].iter().max().unwrap();
        let max_loss = (n as f64 - max_count as f64) / n_total_f;
        let base_obj = max_loss + leaf_reg;

        if max_loss == 0.0 || max_loss < split_reg + leaf_reg + path_cost || n <= 1 {
            return (base_obj, base_obj);
        }
        if depth <= 0 {
            // A depth-0 placeholder that Step B will still complete
            // is priced at its admissible completion lb, not its leaf cost.
            if self.search.config.rashomon_mode && self.search.config.step_b_remaining > 0 {
                return (
                    base_obj.min(split_reg + 2.0 * leaf_reg + path_cost),
                    base_obj,
                );
            }
            return (base_obj, base_obj);
        }

        let mut lowerbound = base_obj.min(split_reg + 2.0 * leaf_reg + path_cost);
        let mut upperbound = base_obj;

        if self.heuristic_bound_transfer_enabled() {
            let sig = SupportSig::new(n, &class_counts[..n_classes]);
            match self.worker.similar_lb.get(&sig) {
                Some(&val) if val > lowerbound => {
                    self.worker.profile.n_similar_lb_hits += 1;
                    lowerbound = val.min(base_obj);
                    if lowerbound >= upperbound - 1e-10 {
                        return (upperbound, upperbound);
                    }
                }
                Some(_) => {
                    self.worker.profile.n_similar_lb_misses += 1;
                }
                None => {
                    self.worker.profile.n_similar_lb_misses += 1;
                }
            }
            match self.worker.similar_ub.get(&sig) {
                Some(&val) if val < upperbound => {
                    self.worker.profile.n_similar_ub_hits += 1;
                    upperbound = val;
                    if upperbound - lowerbound < 1e-10 {
                        return (lowerbound, upperbound);
                    }
                }
                Some(_) => {
                    self.worker.profile.n_similar_ub_misses += 1;
                }
                None => {
                    self.worker.profile.n_similar_ub_misses += 1;
                }
            }

            if let Some(ref shared) = self.search.shared_bounds {
                let (lb_opt, ub_opt) = shared.lookup(n, &class_counts[..n_classes], depth);
                if let Some(raw_lb) = lb_opt {
                    let lb = if n_total_f > 0.0 {
                        raw_lb / n_total_f
                    } else {
                        raw_lb
                    };
                    if lb > lowerbound {
                        lowerbound = lb.min(base_obj);
                    }
                }
                if let Some(raw_ub) = ub_opt {
                    let ub = if n_total_f > 0.0 {
                        raw_ub / n_total_f
                    } else {
                        raw_ub
                    };
                    if ub < upperbound {
                        upperbound = ub;
                    }
                }
            }
        }

        (lowerbound, upperbound)
    }

    // ------------------------------------------------------------------
    // Partitioning
    // ------------------------------------------------------------------

    #[inline]
    fn bin_index_right(thresholds: &[f64], val: f64) -> usize {
        thresholds.partition_point(|t| *t <= val)
    }

    fn partition_into_flat(
        &self,
        indices: &[u32],
        feature_col: &[f64],
        thresholds: &[f64],
        group_map: Option<&[usize]>,
    ) -> FlatBins {
        let n_bins = thresholds.len() + 1;
        let k = group_map.map_or(n_bins, |gm| gm.iter().copied().max().unwrap_or(0) + 1);
        let mut counts = vec![0usize; k];

        for &idx in indices {
            let bin = Self::bin_index_right(thresholds, feature_col[idx as usize]);
            let group = group_map.map_or(bin, |gm| gm[bin]);
            counts[group] += 1;
        }

        let mut offsets = Vec::with_capacity(k + 1);
        offsets.push(0);
        for &count in &counts {
            offsets.push(offsets.last().copied().unwrap_or(0) + count);
        }

        let mut next = offsets[..k].to_vec();
        let mut flat = vec![0u32; indices.len()];
        for &idx in indices {
            let bin = Self::bin_index_right(thresholds, feature_col[idx as usize]);
            let group = group_map.map_or(bin, |gm| gm[bin]);
            let pos = next[group];
            flat[pos] = idx;
            next[group] += 1;
        }

        FlatBins { offsets, flat }
    }

    fn partition(
        &self,
        indices: &[u32],
        feature_index: usize,
        thresholds: &[f64],
        group_map: Option<&[usize]>,
    ) -> Vec<Vec<u32>> {
        self.partition_into_flat(
            indices,
            self.search.store.feature_col(feature_index),
            thresholds,
            group_map,
        )
        .to_nested()
    }

    fn upsert_rashomon_split(&mut self, key: &MemoKey, record: SplitRecord) {
        // Original semantics: 1 entry per split_id per SubProblem; on
        // collision, skip (first-seen wins). Pool may share identity-equal
        // SplitRecords across SubProblems; the per-SubProblem dedup is by
        // deref-checking `split_id`.
        let split_id = record.split_id;
        let exists = {
            let sp = self.search.memo.get(key).unwrap();
            sp.rashomon_indices
                .iter()
                .any(|&i| self.worker.split_pool.get(i).split_id == split_id)
        };
        if exists {
            return;
        }
        // Dual-write: capture feature/upper before moving record.
        let feat = record.feature_index as u16;
        let su = record.split_upper;
        let pool_idx = self.worker.split_pool.push(record);
        self.search
            .bound_upsert(key, feat, pool_idx, f64::NEG_INFINITY, su);
        self.search.bound_set_rashomon_flag(key, feat, true);
        let mut sp = self.search.memo.get_mut(key).unwrap();
        sp.rashomon_indices.push(pool_idx);
    }

    /// `upsert_rashomon_split` for a split id known to be absent at `key`.
    fn push_rashomon_split(&mut self, key: &MemoKey, record: SplitRecord) {
        let feat = record.feature_index as u16;
        let su = record.split_upper;
        let pool_idx = self.worker.split_pool.push(record);
        self.search
            .bound_upsert(key, feat, pool_idx, f64::NEG_INFINITY, su);
        self.search.bound_set_rashomon_flag(key, feat, true);
        let mut sp = self.search.memo.get_mut(key).unwrap();
        sp.rashomon_indices.push(pool_idx);
    }

    fn upsert_materialized_split(&mut self, key: &MemoKey, record: SplitRecord) {
        // Original semantics: 1 entry per split_id per SubProblem; on
        // collision, *update* split_upper of the existing entry. With pool
        // (and the (split_id, split_upper_bits) key), an updated
        // split_upper means a fresh pool entry — we replace the
        // SubProblem's slot to point at the new pool idx.
        // Per-SubProblem mutable state (contextual_lowerbound /
        // contextually_dead) lives in sidecars and is NOT mutated on
        // re-upsert (matching the original which only updated split_upper).
        let cl = record.contextual_lowerbound;
        let dead = record.contextually_dead;
        let feature_index_u16 = record.feature_index as u16;
        let split_upper = record.split_upper;
        // Dedup is by (key, feature_id) — matches the legacy
        // "1 entry per split_id" semantic because at each node, distinct
        // split_ids materialized via this path always have distinct
        // feature_indices (each upsert_materialized_split call comes from
        // explore_node's per-feature loop). On collision (same feature_id
        // already present), `bound_upsert_materialized` updates pool_idx
        // and split_upper in lockstep, preserving contextual sidecars and
        // the rashomon flag. First-insert seeds contextual state from the
        // SplitRecord; updates preserve any prior per-context state.
        let pre_existed = self
            .search
            .bound_get(key, feature_index_u16)
            .map(|e| e.is_materialized())
            .unwrap_or(false);
        let pool_idx = self.worker.split_pool.push(record);
        self.search.bound_upsert_materialized(
            key,
            feature_index_u16,
            pool_idx,
            f64::NEG_INFINITY,
            split_upper,
        );
        if !pre_existed {
            self.search
                .bound_set_contextual(key, feature_index_u16, cl, dead);
        }
    }

    /// Register interned child keys in `parent_map` (read only for its size).
    fn register_parent_links(&self, child_ids: &[u32]) {
        let arena = &self.search.memo_key_arena;
        for &id in child_ids {
            self.search.parent_map.insert(arena.canonical(id));
        }
    }

    fn import_phase1_tree_at_key(
        &mut self,
        key: MemoKey,
        indices: &[u32],
        depth: i32,
        tree: &TreeNode,
        used_feats: &FeatureMask,
        rashomon_bound: f64,
        current_path: Option<&[SplitStep]>,
    ) -> f64 {
        match tree {
            TreeNode::Leaf { .. } => {
                let base_obj = self.search.memo.get(&key).unwrap().base_objective;
                let mut sp = self.search.memo.get_mut(&key).unwrap();
                if base_obj < sp.upperbound {
                    sp.upperbound = base_obj;
                    sp.upperbound_is_heuristic = false;
                }
                base_obj
            }
            TreeNode::Internal {
                feature_index,
                thresholds,
                children,
                group_map,
            } => {
                if depth <= 0 {
                    let base_obj = self.search.memo.get(&key).unwrap().base_objective;
                    let mut sp = self.search.memo.get_mut(&key).unwrap();
                    if base_obj < sp.upperbound {
                        sp.upperbound = base_obj;
                        sp.upperbound_is_heuristic = false;
                    }
                    return base_obj;
                }

                let bins =
                    self.partition(indices, *feature_index, thresholds, group_map.as_deref());
                let child_used = if self.search.config.feature_exclusion {
                    used_feats.without(*feature_index)
                } else {
                    used_feats.clone()
                };
                let thresh_base_opt = self
                    .search
                    .store
                    .ctx()
                    .find_threshold_base(*feature_index, thresholds);

                let mut child_keys = Vec::with_capacity(children.len());
                let mut child_cost_sum = 0.0;
                for (bin_idx, (child_tree, bin_indices)) in
                    children.iter().zip(bins.iter()).enumerate()
                {
                    let (child_key, child_path_buf) =
                        if let (Some(parent_path), Some(tb)) = (current_path, thresh_base_opt) {
                            let step = SplitStep {
                                feature_index: *feature_index as u16,
                                threshold_base: tb as u16,
                                bin: bin_idx as u8,
                                k: children.len() as u8,
                            };
                            let ck = self.get_or_create_with_path(
                                &self.search.store.support_from_indices(bin_indices),
                                &child_used,
                                depth - 1,
                                parent_path,
                                step.clone(),
                            );
                            let mut path = parent_path.to_vec();
                            path.push(step);
                            (ck, Some(path))
                        } else {
                            (
                                self.get_or_create(
                                    &self.search.store.support_from_indices(bin_indices),
                                    &child_used,
                                    depth - 1,
                                ),
                                None,
                            )
                        };

                    let child_cost = self.import_phase1_tree_at_key(
                        child_key.clone(),
                        bin_indices,
                        depth - 1,
                        child_tree,
                        &child_used,
                        rashomon_bound,
                        child_path_buf.as_deref(),
                    );
                    child_keys.push(child_key.clone());
                    child_cost_sum += child_cost;
                }

                let split_reg = (1.0 - self.search.config.reg_alpha) * self.search.config.reg;
                let path_cost = self.search.config.path_penalty * indices.len() as f64
                    / self.search.n_total as f64;
                let exact_cost = split_reg + path_cost + child_cost_sum;

                let split_id = Self::split_record_id(
                    *feature_index,
                    thresholds,
                    &child_keys,
                    group_map.as_deref(),
                );
                let interned = self.intern_keys(child_keys.clone());
                self.register_parent_links(&interned);
                let record = SplitRecord {
                    split_id,
                    feature_index: *feature_index,
                    thresholds: thresholds.clone(),
                    child_keys: interned,
                    split_upper: exact_cost,
                    contextual_lowerbound: f64::NEG_INFINITY,
                    contextually_dead: false,
                    group_map: group_map.clone(),
                };

                {
                    let mut sp = self.search.memo.get_mut(&key).unwrap();
                    if exact_cost < sp.upperbound {
                        sp.upperbound = exact_cost;
                        sp.upperbound_is_heuristic = false;
                    }
                    drop(sp);
                    // best_split lookup via best_split_table -> bound_table -> pool.
                    let should_update = match self.resolve_best_split(&key) {
                        None => true,
                        Some(existing) => exact_cost <= existing.split_upper,
                    };
                    if should_update {
                        // Install the new incumbent: push record + bound_upsert + table.
                        self.install_best_split(&key, record.clone());
                    }
                }

                // Step A: record an imported split only when it is in
                // Select_B(support) (the pass records it there if it belongs).
                if exact_cost <= rashomon_bound + 1e-10 {
                    let in_pool = !self.search.config.rashomon_mode || {
                        let (support, mask) = {
                            let sp = self.search.memo.get(&key).unwrap();
                            (sp.support.clone(), sp.feature_mask.clone())
                        };
                        self.rash_select_contains(
                            &support,
                            &mask,
                            depth,
                            *feature_index,
                            thresholds,
                            group_map.as_deref(),
                        )
                    };
                    if in_pool {
                        self.upsert_rashomon_split(&key, record);
                    }
                }

                let gap = {
                    let sp = self.search.memo.get(&key).unwrap();
                    sp.upperbound - sp.lowerbound
                };
                // Step A: imported nodes are not enqueued; the Step-A
                // pass reaches each one at its context scope S(v).
                if gap > 1e-10 && !self.search.config.rashomon_mode {
                    let requested_scope = self.search.memo.get(&key).unwrap().upperbound;
                    self.enqueue_explore(key, gap, depth, requested_scope);
                }

                exact_cost
            }
        }
    }

    // ------------------------------------------------------------------
    // ------------------------------------------------------------------
    // BFS: single-level node expansion
    // ------------------------------------------------------------------

    // ------------------------------------------------------------------
    // Step A: Step A of the RESPLINTER algorithm as its own pass
    // ------------------------------------------------------------------

    /// Phase-2 search (Step A: every rashomon-mode solve).
    /// A node v is explored at its context scope S(v): S(root) = β and
    /// S(c_i) = S(v) − split_reg − path_cost − Σ_{j≠i} lb(c_j) (the child
    /// budget β_i). Every split with admissible lb-sum ≤ S(v) is recorded, every
    /// survivor is materialised, and a node is re-explored when a wider
    /// scope is requested (`sp.scope` = widest scope explored so far).
    /// Any tree of cost ≤ β then has lb-sum ≤ S at each of its vertices,
    /// so all its splits are recorded; the DAG builder filters exactly.
    /// Admission tolerance of every scope test in the Step-A pass:
    /// at least the DAG builder's slack (`SCOPE_EPS` = 1e-9 on `β`), so the
    /// pass never drops a split the builder would admit (over-recording is
    /// harmless: the builder filters exactly).
    const RASH_TOL: f64 = 2e-9;

    /// Equivalent-points lb in the Step-A pass: admissible only when
    /// every feature is keyed in `thresholds_per_feature` (the eqp buckets
    /// ignore an unkeyed feature, so its rows could be merged and the lb
    /// over-count). Outside the pass: always on (unchanged).
    fn eqp_lb_enabled(&self) -> bool {
        if !self.search.config.rashomon_mode {
            return true;
        }
        let c = &self.search.config;
        match &c.thresholds_per_feature {
            Some(m) => (0..c.n_features).all(|f| m.contains_key(&f)),
            None => false,
        }
    }

    /// Whether Select_B(`support`) (`rashomon_select`) holds the split
    /// identity (feature, threshold bits, group_map).
    fn rash_select_contains(
        &mut self,
        support: &SupportMask,
        feature_mask: &FeatureMask,
        depth: i32,
        feature_index: usize,
        thresholds: &[f64],
        group_map: Option<&[usize]>,
    ) -> bool {
        self.rashomon_select_at(support, feature_mask, Some(depth))
            .iter()
            .any(|c| {
                c.feature_index == feature_index
                    && c.thresholds.len() == thresholds.len()
                    && c.thresholds
                        .iter()
                        .zip(thresholds)
                        .all(|(a, b)| a.to_bits() == b.to_bits())
                    && c.group_map.as_deref() == group_map
            })
    }

    /// Nodes built by `closure_build` (depth in 1..=closure depth): closure
    /// owns them, the Step-A pass never expands them.
    #[inline]
    fn closure_owned(&self, depth: i32) -> bool {
        matches!(self.search.config.closure_depth, Some(cd) if depth > 0 && depth <= cd)
    }

    /// Step A node state after `create_subproblem`: pure / near-pure
    /// nodes are not terminal (their splits are separate trees); only
    /// n ≤ 1 and depth 0 stay terminal. Lower bounds are kept (base_obj is
    /// the exact minimum there). Depth-0 nodes with Step B still to run
    /// keep their admissible completion lb.
    fn rash_explore_node_state(&self, sp: &mut SubProblem, depth: i32) {
        if depth > 0 {
            if sp.is_terminal && sp.n_samples > 1 {
                sp.is_terminal = false;
                sp.is_solved = false;
                sp.all_survivors_materialized = false;
            }
        } else if !sp.is_terminal {
            if self.search.config.step_b_remaining <= 0 {
                sp.lowerbound = sp.base_objective;
            }
            sp.upperbound = sp.base_objective;
            sp.upperbound_is_heuristic = false;
            sp.is_terminal = true;
            sp.is_solved = true;
        }
    }

    /// ub_hint of the Step A Select at every vertex: the value set by
    /// `set_rashomon_select_ub`, else `config.rashomon_bound`.
    fn rashomon_select_ub(&self) -> f64 {
        let v = f64::from_bits(self.search.rashomon_select_ub.load(Ordering::Relaxed));
        if v.is_nan() {
            self.search.config.rashomon_bound
        } else {
            v
        }
    }

    /// Fix the Select ub_hint of the Step-A pass (Step-B mini solves
    /// pass the outer β so their pool matches the full-depth search).
    pub fn set_rashomon_select_ub(&self, ub: f64) {
        self.search
            .rashomon_select_ub
            .store(ub.to_bits(), Ordering::Relaxed);
    }

    /// Step A gate: explore `key` at scope `s` unless it is terminal,
    /// closure-owned, cannot fit `s`, or was explored at a scope ≥ `s`.
    fn rash_wants(&self, key: &MemoKey, s: f64) -> bool {
        match self.search.memo.get(key) {
            Some(sp) => {
                !(sp.is_terminal
                    || self.closure_owned(sp.depth)
                    || !(sp.lowerbound <= s + Self::RASH_TOL)
                    || (sp.explored && s <= sp.scope + 1e-12))
            }
            None => false,
        }
    }

    /// Step A worklist: one level per remaining depth. Every child sits one
    /// level below its parent, so draining levels top-down explores each
    /// node once per pass, at the widest scope any parent requested
    /// (insertion order kept for determinism; the max is taken on push).
    fn rashomon_enumerate(&mut self) {
        self.rash_lb_sync_ub();
        let root_support = self.search.root_support.clone();
        let initial_feats = self.search.config.initial_used_features.clone();
        let max_depth = self.search.config.max_depth;
        *self.search.root_key.lock().unwrap() = MemoKey::new(
            root_support.clone(),
            initial_feats.clone(),
            max_depth,
            Self::mode_flags(&self.search.config),
            Self::teacher_alpha_memo_bits(&self.search.config),
        );
        let root_key = self.get_or_create(&root_support, &initial_feats, max_depth);
        if max_depth <= 0 {
            return;
        }
        let beta = self.search.config.rashomon_bound;
        let mut levels: Vec<RashLevel> = (0..=max_depth as usize)
            .map(|_| RashLevel::default())
            .collect();
        levels[max_depth as usize].push(root_key, beta);
        self.rash_drain_levels(max_depth, &mut levels);
    }

    /// Shared Step B: Step A from several roots in one pass. Every seed
    /// (support, scope) becomes a root at `max_depth` (all features), pushed
    /// in the given order (`RashLevel::push` keeps the widest scope of a
    /// repeated key), then the levels drain exactly as `rashomon_enumerate`:
    /// a node reached from several roots is explored once per level at the
    /// widest scope requested, so it records a superset of what each
    /// single-root pass records there. `search.root_key` = the first seed.
    /// Returns the seed keys in order.
    pub fn rashomon_enumerate_seeded(&mut self, seeds: &[(SupportMask, f64)]) -> Vec<MemoKey> {
        self.rash_lb_sync_ub();
        let initial_feats = self.search.config.initial_used_features.clone();
        let max_depth = self.search.config.max_depth;
        let mut levels: Vec<RashLevel> = (0..=max_depth.max(0) as usize)
            .map(|_| RashLevel::default())
            .collect();
        let mut keys: Vec<MemoKey> = Vec::with_capacity(seeds.len());
        for (support, scope) in seeds {
            let key = self.get_or_create(support, &initial_feats, max_depth);
            if max_depth > 0 {
                levels[max_depth as usize].push(key.clone(), *scope);
            }
            keys.push(key);
        }
        if let Some(k) = keys.first() {
            *self.search.root_key.lock().unwrap() = k.clone();
        }
        if max_depth > 0 {
            self.rash_drain_levels(max_depth, &mut levels);
        }
        keys
    }

    /// Drain the Step-A worklist top-down (`rashomon_enumerate`).
    fn rash_drain_levels(&mut self, max_depth: i32, levels: &mut [RashLevel]) {
        for d in (1..=max_depth).rev() {
            let items = std::mem::take(&mut levels[d as usize].items);
            for (key, s) in items {
                if self.check_time() {
                    return;
                }
                if !self.rash_wants(&key, s) {
                    continue;
                }
                {
                    let mut sp = self.search.memo.get_mut(&key).unwrap();
                    if sp.explored {
                        self.worker.profile.n_reexplorations += 1;
                    }
                    sp.scope = s;
                    sp.explored = true;
                }
                self.search.n_explored.fetch_add(1, Ordering::Relaxed);
                self.rash_explore_node(&key, d, s, levels);
            }
        }
    }

    /// Explore one node at context scope `s` (Step A): Select with
    /// ub_hint = `rashomon_select_ub`, no killer move, admissible arity and
    /// Phase-A prunes against `s`, every survivor materialised, splits with
    /// lb-sum ≤ `s` recorded, children pushed at their context scopes.
    /// Never touches similar_lb / similar_ub, upper bounds or `is_solved`.
    fn rash_explore_node(&mut self, key: &MemoKey, depth: i32, s: f64, levels: &mut [RashLevel]) {
        let tol = Self::RASH_TOL;
        let (support, used_feats, parent_path, sp_lb, sp_base_obj) = {
            let sp = self.search.memo.get(key).unwrap();
            (
                sp.support.clone(),
                sp.feature_mask.clone(),
                sp.split_path.clone(),
                sp.lowerbound,
                sp.base_objective,
            )
        };
        debug_assert!(
            sp_lb <= sp_base_obj + 1e-12,
            "lb {sp_lb} > leaf {sp_base_obj}"
        );
        let d_idx = (depth as usize).min(MAX_PROFILE_DEPTH - 1);
        self.worker.profile.nodes_at_depth[d_idx] += 1;

        let exact_lb = self.rash_exact_lb_active();
        // Child supports by le-masks (`rash_lb_gate`
        // mask_exact: the same partitions as the row walk), so the row list
        // is built only if a split falls back to the row partition.
        let mask_children = exact_lb && self.rash_lb_gate().0;
        let mut indices: Vec<u32> = if mask_children {
            Vec::new()
        } else {
            self.search.store.support_to_indices(&support)
        };
        let n_node = if mask_children {
            support.n_samples
        } else {
            indices.len()
        };
        let path_cost =
            self.search.config.path_penalty * n_node as f64 / self.search.n_total as f64;
        let split_reg = (1.0 - self.search.config.reg_alpha) * self.search.config.reg;
        let leaf_reg = self.search.config.reg_alpha * self.search.config.reg;

        // Purification vs S(v): no split fits this scope. Nothing is
        // persisted (a reused solver with a larger β re-explores the node).
        let perfect_split_cost = split_reg + 2.0 * leaf_reg + path_cost;
        if perfect_split_cost > s + tol {
            self.worker.profile.n_nodes_purified += 1;
            return;
        }

        // Select: the full post-dedup pool at ub_hint = β (no killer move).
        // Shared Select pool: the completion-lb DFS's pool at this support
        // when it has one (the lb DFS reaches most explored supports first;
        // pools the explore computes are not stored: measured, they are
        // rarely asked for again and cost memory).
        let mut shared_pool: Option<RashSharedPool> = None;
        let mut owned: Vec<SplitCandidate> = Vec::new();
        let n_available: u64;
        let cached = if self.rash_shared_on() {
            self.rash_shared_get(&support)
        } else {
            None
        };
        if let Some(p) = cached {
            self.worker.profile.n_select_shared_hits += 1;
            n_available = p.n_available;
            shared_pool = Some(p);
        } else {
            let t_candgen = Instant::now();
            let slb_ref = if self.search.config.dp_criterion == DpCriterion::Learned {
                Some(&self.worker.similar_lb)
            } else {
                None
            };
            let (candidates, cand_stats) = generate_candidates_for_support_rash(
                self.search.store.as_ref(),
                &support,
                &self.search.config,
                &used_feats,
                self.search.n_total,
                Some(self.rashomon_select_ub()),
                slb_ref,
                &self.worker.scratch_bank,
                Some(depth),
            );
            self.worker.profile.time_candidate_gen += t_candgen.elapsed();
            self.worker.profile.n_explore_selects += 1;
            n_available = cand_stats.n_available;
            owned = candidates;
        }
        // Arity bound: a k-way split costs at least split_reg + k·leaf_reg; pruned
        // against the scope, not the leaf (pure-node splits are trees too).
        let row_g = |k: usize| split_reg + (k as f64) * leaf_reg <= s + tol;
        // The pool can name one split identity (feature, thresholds,
        // group_map) more than once; a repeat materialises the same children
        // and is merged by `upsert_rashomon_split`, so skip it up front
        // (speed only), by identity ids.
        let views: Vec<RashPoolRecRef>;
        let items: Vec<&dyn ExploreCand>;
        let idents: Vec<u32>;
        let n_ident: u32;
        let n_generated: u64;
        match &shared_pool {
            Some(p) => {
                match &p.recs {
                    RashPoolStore::Legacy(r) => {
                        items = p
                            .explore
                            .iter()
                            .map(|&i| &r[i as usize] as &dyn ExploreCand)
                            .collect();
                    }
                    RashPoolStore::Arena(a) => {
                        views = p.explore.iter().map(|&i| a.rec(i as usize)).collect();
                        items = views.iter().map(|v| v as &dyn ExploreCand).collect();
                    }
                }
                idents = p
                    .explore
                    .iter()
                    .map(|&r| p.recs.ident(r as usize))
                    .collect();
                n_ident = p.n_ident;
                n_generated = p
                    .k_hist
                    .iter()
                    .filter(|e| row_g(e.0))
                    .map(|e| e.1 as u64)
                    .sum();
            }
            None => {
                items = owned.iter().map(|c| c as &dyn ExploreCand).collect();
                (idents, n_ident) = explore_identities(&items);
                n_generated = owned.iter().filter(|c| row_g(c.k)).count() as u64;
            }
        }
        self.worker.profile.n_candidates_generated += n_generated;
        self.worker.profile.n_features_available += n_available;
        let use_ident = idents.len() == items.len();
        let mut ident_seen: Vec<bool> = vec![false; if use_ident { n_ident as usize } else { 0 }];
        let mut seen_ident: std::collections::HashSet<(usize, Vec<u64>, Option<Vec<usize>>)> =
            std::collections::HashSet::new();
        // On a node's first exploration the recorded split ids are exactly
        // the ones this call adds (hashed set instead of
        // `upsert_rashomon_split`'s scan).
        let mut new_ids: Option<HashSet<u64>> = if self
            .search
            .memo
            .get(key)
            .map_or(false, |sp| sp.rashomon_indices.is_empty())
        {
            Some(HashSet::new())
        } else {
            None
        };
        // `lb_threads` (pass 1): this node's child completion-lb
        // searches on the lb pool, merged into the lb caches; the loop below
        // (pass 2) is unchanged and finds them in the memo.
        if exact_lb
            && depth - 1 + self.search.config.step_b_remaining.max(0) > 0
            && self.rash_lb_threads_eff() > 1
            && self.worker.rash_base.is_none()
        {
            self.rash_prefetch_lbs(
                &support,
                &items,
                &idents,
                use_ident,
                depth,
                s,
                split_reg,
                path_cost,
                leaf_reg,
                &used_feats,
                mask_children,
                &mut indices,
            );
        }
        for (ci, &cand) in items.iter().enumerate() {
            if !row_g(cand.x_k()) {
                continue;
            }
            if use_ident {
                let g = idents[ci];
                if g == u32::MAX || std::mem::replace(&mut ident_seen[g as usize], true) {
                    continue;
                }
            } else {
                let non_empty = (0..cand.x_n_children())
                    .filter(|&i| cand.x_child_n(i) > 0)
                    .count();
                if non_empty <= 1 {
                    continue;
                }
                let ident = (
                    cand.x_feature(),
                    cand.x_thresholds()
                        .iter()
                        .map(|t| t.to_bits())
                        .collect::<Vec<u64>>(),
                    cand.x_group_map().map(|g| g.to_vec()),
                );
                if !seen_ident.insert(ident) {
                    continue;
                }
            }
            // Phase A (count-based, admissible): lb-sum against S(v).
            let mut child_lb_sum = split_reg + path_cost;
            let mut pruned = false;
            let n_children = cand.x_n_children();
            let mut cbuf = [0i64; MAX_CLASSES];
            for ci in 0..n_children {
                let (lb, _ub) = self.compute_bounds_from_counts(
                    cand.x_child_n(ci) as usize,
                    cand.x_child_counts(ci, &mut cbuf),
                    depth - 1,
                );
                child_lb_sum += lb;
                let remaining = n_children - ci - 1;
                if child_lb_sum + remaining as f64 * leaf_reg > s + tol {
                    pruned = true;
                    break;
                }
            }
            if pruned || child_lb_sum > s + tol {
                continue;
            }
            self.worker.profile.n_candidates_survived_phase_a += 1;

            // Materialise every survivor (under mask_exact: child supports
            // by le-masks, same supports).
            let mask_supports: Option<Vec<SupportMask>> = if mask_children {
                self.search.store.partition_support_le(
                    &support,
                    cand.x_feature(),
                    cand.x_thresholds(),
                    cand.x_group_map(),
                )
            } else {
                None
            };
            if mask_supports.is_none() && indices.is_empty() && support.n_samples > 0 {
                indices = self.search.store.support_to_indices(&support);
            }
            let bins = if mask_supports.is_some() {
                None
            } else {
                Some(self.partition_into_flat(
                    &indices,
                    self.search.store.feature_col(cand.x_feature()),
                    cand.x_thresholds(),
                    cand.x_group_map(),
                ))
            };
            let child_used = if self.search.config.feature_exclusion {
                used_feats.without(cand.x_feature())
            } else {
                used_feats.clone()
            };
            let thresh_base = self
                .search
                .store
                .ctx()
                .find_threshold_base(cand.x_feature(), cand.x_thresholds());
            let k = match (&mask_supports, &bins) {
                (Some(cs), _) => cs.len(),
                (None, Some(b)) => b.len(),
                (None, None) => unreachable!(),
            };
            let step_of = |bin_idx: usize| {
                thresh_base.map(|tb| SplitStep {
                    feature_index: cand.x_feature() as u16,
                    threshold_base: tb as u16,
                    bin: bin_idx as u8,
                    k: k as u8,
                })
            };
            let (child_keys, child_bounds) = if exact_lb {
                // Exact completion lb: exact completion lbs, children
                // inserted only when the split passes the lb-sum test.
                let steps: Vec<Option<SplitStep>> = (0..k).map(step_of).collect();
                let child_supports: Vec<SupportMask> = match mask_supports {
                    Some(cs) => cs,
                    None => {
                        let b = bins.as_ref().unwrap();
                        (0..k)
                            .map(|i| self.search.store.support_from_indices(b.bin_slice(i)))
                            .collect()
                    }
                };
                match self.rash_exact_lb_children(
                    child_supports,
                    &steps,
                    &child_used,
                    depth,
                    &parent_path,
                    s,
                    split_reg + path_cost,
                    &support,
                    if cand.x_group_map().is_none() {
                        Some((cand.x_feature(), cand.x_thresholds()))
                    } else {
                        None
                    },
                ) {
                    Some(kb) => kb,
                    None => continue,
                }
            } else {
                let bins = bins.as_ref().unwrap();
                let mut child_keys: Vec<MemoKey> = Vec::with_capacity(k);
                for bin_idx in 0..k {
                    let child_support = self
                        .search
                        .store
                        .support_from_indices(bins.bin_slice(bin_idx));
                    let ck = if let Some(step) = step_of(bin_idx) {
                        self.get_or_create_with_path(
                            &child_support,
                            &child_used,
                            depth - 1,
                            &parent_path,
                            step,
                        )
                    } else {
                        self.get_or_create(&child_support, &child_used, depth - 1)
                    };
                    child_keys.push(ck);
                }
                let child_bounds: Vec<(f64, f64)> = child_keys
                    .iter()
                    .map(|ck| {
                        let c = self.search.memo.get(ck).unwrap();
                        (c.lowerbound, c.upperbound)
                    })
                    .collect();
                (child_keys, child_bounds)
            };
            let materialized_lb =
                split_reg + path_cost + child_bounds.iter().map(|b| b.0).sum::<f64>();
            // lb-sum test: record iff the admissible lb-sum fits S(v) (one entry
            // per split identity).
            if materialized_lb > s + tol {
                continue;
            }
            let materialized_ub =
                split_reg + path_cost + child_bounds.iter().map(|b| b.1).sum::<f64>();
            let split_id = Self::split_record_id(
                cand.x_feature(),
                cand.x_thresholds(),
                &child_keys,
                cand.x_group_map(),
            );
            let interned = self.intern_keys(child_keys.clone());
            self.register_parent_links(&interned);
            let record = SplitRecord {
                split_id,
                feature_index: cand.x_feature(),
                thresholds: cand.x_thresholds().clone(),
                child_keys: interned,
                split_upper: materialized_ub,
                contextual_lowerbound: f64::NEG_INFINITY,
                contextually_dead: false,
                group_map: cand.x_group_map().map(|g| g.to_vec()),
            };
            match new_ids.as_mut() {
                Some(ids) => {
                    if ids.insert(split_id) {
                        self.push_rashomon_split(key, record);
                    }
                }
                None => self.upsert_rashomon_split(key, record),
            }
            // Child scope: S(c_i) = S(v) − split_reg − path − Σ_{j≠i} lb_j.
            if depth - 1 > 0 {
                for (ck, &(child_lb, _)) in child_keys.into_iter().zip(&child_bounds) {
                    let child_scope = s - (materialized_lb - child_lb);
                    if self.rash_wants(&ck, child_scope) {
                        levels[(depth - 1) as usize].push(ck, child_scope);
                    }
                }
            }
        }
    }

    /// The exact completion lb is used by this pass: the objective is
    /// leaf-only (`types::leaf_only_objective`, the condition
    /// under which the result is not flagged `non_leaf_objective`: with
    /// split_reg / path costs the DAG prices trees by leaf costs only, so
    /// a tighter lb changes which over-beta trees it returns), and the
    /// pass's pool does not depend on state the lb-solver cannot mirror
    /// (no closure below the prefix, no feature exclusion — the lb memo is
    /// keyed without the feature mask — and no Learned criterion, whose
    /// Select reads `similar_lb`).
    fn rash_exact_lb_active(&self) -> bool {
        let c = &self.search.config;
        types::leaf_only_objective(c.reg, c.reg_alpha, c.path_penalty)
            && c.closure_depth.is_none()
            && !c.feature_exclusion
            && c.dp_criterion != DpCriterion::Learned
    }

    /// Clear the completion-lb caches when the Select ub changed since they
    /// were filled (a reused solver at a new beta, `set_rashomon_select_ub`):
    /// their pool would be stale. Called at the start of every pass. A level
    /// base (`stepb_batches`) holds the ub bits `rash_lb_ub_bits` had at
    /// `attach_rash_level_base`: it is dropped with them.
    fn rash_lb_sync_ub(&mut self) {
        let bits = self.rashomon_select_ub().to_bits();
        if self.worker.rash_lb_ub_bits != Some(bits) {
            self.worker.rash_level_base = None;
            self.worker.rash_lb_memo.clear();
            self.worker.rash_lb_memo_d1 = 0;
            self.worker.rash_lb_d1s.clear();
            self.worker.rash_lb_select.clear();
            self.worker.rash_select_shared.clear();
            self.worker.rash_lb_select_records = 0;
            self.worker.rash_lb_ub_bits = Some(bits);
        }
    }

    /// Compatibility key of this solver's completion-lb caches
    /// (`RashLbCache::key`).
    fn rash_lb_cache_key(&self) -> (usize, u64, usize, u64, u64, u64, usize, usize, usize) {
        let c = &self.search.config;
        (
            Arc::as_ptr(&self.search.store) as *const u8 as usize,
            self.rashomon_select_ub().to_bits(),
            self.search.n_total,
            c.reg.to_bits(),
            c.reg_alpha.to_bits(),
            c.path_penalty.to_bits(),
            c.beam_width,
            c.max_k,
            c.n_classes,
        )
    }

    /// Move the completion-lb caches out (shared Step B: Step A hands
    /// them to the shared Step-B solver). Empty when the pass did not use
    /// them.
    pub fn take_rash_lb_cache(&mut self) -> RashLbCache {
        if self.worker.rash_lb_ub_bits.is_none() || !self.rash_exact_lb_active() {
            return RashLbCache::default();
        }
        let key = Some(self.rash_lb_cache_key());
        self.worker.rash_lb_ub_bits = None;
        self.worker.rash_lb_select_records = 0;
        self.worker.rash_lb_memo_d1 = 0;
        RashLbCache {
            key,
            memo: std::mem::take(&mut self.worker.rash_lb_memo),
            select: std::mem::take(&mut self.worker.rash_lb_select),
            shared: std::mem::take(&mut self.worker.rash_select_shared),
            select_fp: self.rash_select_config_fp(),
            d1s: std::mem::take(&mut self.worker.rash_lb_d1s),
        }
    }

    /// Hash of every config field Select reads (with the store, Select ub
    /// and n_total of `rash_lb_cache_key`, the whole input of a Select call
    /// at a support but the scratch): a shared Select pool pool is reused
    /// by another solver only when it matches.
    pub fn rash_select_config_fp(&self) -> u64 {
        let c = &self.search.config;
        let mut h = DefaultHasher::new();
        (
            c.n_features,
            c.n_classes,
            c.max_k,
            c.beam_width,
            c.bypass_beam,
        )
            .hash(&mut h);
        std::mem::discriminant(&c.dp_criterion).hash(&mut h);
        (
            c.transition_delta.to_bits(),
            c.min_samples_leaf,
            c.shape_mode,
            c.adaptive_beam,
        )
            .hash(&mut h);
        (
            c.feature_exclusion,
            c.unsafe_filter_depth_threshold,
            c.rashomon_mode,
        )
            .hash(&mut h);
        (
            c.reg.to_bits(),
            c.reg_alpha.to_bits(),
            c.path_penalty.to_bits(),
        )
            .hash(&mut h);
        c.shape_sort_smoothing.to_bits().hash(&mut h);
        for t in &c.feature_types {
            std::mem::discriminant(t).hash(&mut h);
        }
        match &c.thresholds_per_feature {
            None => 0u8.hash(&mut h),
            Some(m) => {
                let mut keys: Vec<&usize> = m.keys().collect();
                keys.sort();
                for k in keys {
                    k.hash(&mut h);
                    for t in &m[k] {
                        t.to_bits().hash(&mut h);
                    }
                    u64::MAX.hash(&mut h);
                }
            }
        }
        for tp in [c.teacher_prob_c0.as_deref(), c.teacher_prob.as_deref()] {
            match tp {
                None => 0u8.hash(&mut h),
                Some(v) => {
                    v.len().hash(&mut h);
                    v.iter().for_each(|x| x.to_bits().hash(&mut h));
                }
            }
        }
        format!("{:?}", c.msdt).hash(&mut h);
        h.finish()
    }

    /// Install `cache` as this solver's completion-lb caches when its key
    /// matches (same store, Select ub, costs and Select knobs): entries are
    /// facts about (support, depth) over that pool, so they hold for any
    /// solver with the same key. Call after `set_rashomon_select_ub`.
    /// Returns whether it was installed (a mismatched cache is dropped).
    pub fn seed_rash_lb_cache(&mut self, cache: RashLbCache) -> bool {
        if cache.key.is_none() || cache.key != Some(self.rash_lb_cache_key()) {
            return false;
        }
        self.worker.rash_lb_memo = cache.memo;
        self.worker.rash_lb_memo_d1 = self.worker.rash_lb_memo.keys().filter(|k| k.1 == 1).count();
        self.worker.rash_lb_d1s = cache.d1s;
        if cache.select_fp == self.rash_select_config_fp() {
            self.worker.rash_select_shared = cache.shared;
        }
        self.worker.rash_lb_select_records =
            cache.select.values().map(|c| c.len() as u64).sum::<u64>()
                + self
                    .worker
                    .rash_select_shared
                    .values()
                    .map(|p| p.n_lb as u64)
                    .sum::<u64>();
        self.worker.rash_lb_select = cache.select;
        let p = &mut self.worker.profile;
        p.lb_select_cache_records = p
            .lb_select_cache_records
            .max(self.worker.rash_lb_select_records);
        self.worker.rash_lb_ub_bits = Some(self.rashomon_select_ub().to_bits());
        true
    }

    /// Children of one candidate split in the exact completion lb pass
    /// (`rash_explore_node`), `None` when the split is rejected (nothing is
    /// inserted then):
    ///  a/b. per bin, the memo bounds of the child key if present, else a
    ///       fresh `create_subproblem` + `rash_explore_node_state` (not
    ///       inserted): the same lbs the legacy path materialises;
    ///  c.   the lb-sum test with these lbs;
    ///  d.   per child, lb = max(lb, `bounded_completion_min`) at its full
    ///       completion depth (depth − 1 + step_b_remaining) and budget
    ///       S(v) − overhead − Σ_{j≠i} lb_j; no completion within the budget
    ///       or an lb-sum above S(v) rejects the split;
    ///  e.   insert the new children (with split path, as
    ///       `get_or_create_with_path` / `get_or_create`) and raise their
    ///       memo lbs to the tightened values.
    /// Returns the child keys and their (lb, ub).
    #[allow(clippy::too_many_arguments)]
    fn rash_exact_lb_children(
        &mut self,
        child_supports: Vec<SupportMask>,
        steps: &[Option<SplitStep>],
        child_used: &FeatureMask,
        depth: i32,
        parent_path: &[SplitStep],
        s: f64,
        overhead: f64,
        parent: &SupportMask,
        iv: Option<(usize, &Arc<[f64]>)>,
    ) -> Option<(Vec<MemoKey>, Vec<(f64, f64)>)> {
        let tol = Self::RASH_TOL;
        let k = child_supports.len();
        let mut supports: Vec<SupportMask> = Vec::with_capacity(k);
        let mut keys: Vec<MemoKey> = Vec::with_capacity(k);
        let mut pending: Vec<Option<SubProblem>> = Vec::with_capacity(k);
        let mut bounds: Vec<(f64, f64)> = Vec::with_capacity(k);
        for support in child_supports {
            let key = MemoKey::new(
                support.clone(),
                child_used.clone(),
                depth - 1,
                Self::mode_flags(&self.search.config),
                Self::teacher_alpha_memo_bits(&self.search.config),
            );
            let existing = self
                .search
                .memo
                .get(&key)
                .map(|c| (c.lowerbound, c.upperbound));
            match existing {
                Some(b) => {
                    bounds.push(b);
                    pending.push(None);
                }
                None => {
                    let mut sp = self.create_subproblem(&support, child_used, depth - 1);
                    self.rash_explore_node_state(&mut sp, depth - 1);
                    bounds.push((sp.lowerbound, sp.upperbound));
                    pending.push(Some(sp));
                }
            }
            supports.push(support);
            keys.push(key);
        }
        let lb_sum = |b: &[(f64, f64)]| overhead + b.iter().map(|x| x.0).sum::<f64>();
        if lb_sum(&bounds) > s + tol {
            return None;
        }
        let completion_depth = (depth - 1) + self.search.config.step_b_remaining.max(0);
        if completion_depth > 0 {
            // Structural depth-1 memo: structural keys of depth-1 completions of a
            // split on store thresholds (`iv` = feature, thresholds; no
            // group_map).
            let skeys = match iv {
                Some((f, thr)) if completion_depth == 1 => {
                    self.rash_d1_child_keys(parent, f, thr, k)
                }
                _ => None,
            };
            for i in 0..k {
                let others: f64 = bounds
                    .iter()
                    .enumerate()
                    .filter(|&(j, _)| j != i)
                    .map(|(_, b)| b.0)
                    .sum();
                let budget = s - overhead - others;
                let skey = skeys.as_ref().map(|v| v[i]);
                match self.bounded_completion_min_keyed(
                    &supports[i],
                    child_used,
                    completion_depth,
                    budget,
                    skey,
                ) {
                    Some(m) => {
                        if m > bounds[i].0 {
                            bounds[i].0 = m;
                        }
                    }
                    None => return None,
                }
                if lb_sum(&bounds) > s + tol {
                    return None;
                }
            }
        }
        for i in 0..k {
            if let Some(mut sp) = pending[i].take() {
                if !self.search.memo.contains_key(&keys[i]) {
                    if let Some(step) = &steps[i] {
                        let mut path = parent_path.to_vec();
                        path.push(step.clone());
                        sp.split_path = path;
                    }
                    self.search.memo.insert(keys[i].clone(), sp);
                }
            }
            let mut c = self.search.memo.get_mut(&keys[i]).unwrap();
            if bounds[i].0 > c.lowerbound {
                c.lowerbound = bounds[i].0;
            }
            debug_assert!(c.lowerbound <= c.base_objective + 1e-12);
            bounds[i].0 = c.lowerbound;
        }
        Some((keys, bounds))
    }

    /// Bounded exact completion minimum (exact completion lb): the minimum
    /// cost of a subtree of depth <= `depth` at `support` over the
    /// lb-solver's option set, searched only below `budget + RASH_TOL`.
    /// `Some(m)`: the minimum m <= budget + RASH_TOL. `None`: certified that
    /// every completion costs more than budget + RASH_TOL. On the solve's
    /// deadline: `Some(trivial lb)` (admissible, never `None`).
    ///
    /// Admissibility: at every node the option set is a SUPERSET of what the
    /// Step-A pass (and the Step-B minis below it) could record there:
    /// the leaf (always), and every split of `rashomon_select_at(support,
    /// used, None)` at ub_hint `rashomon_select_ub()` — the explore's Select
    /// call with the same pool knobs; depth `None` drops only the
    /// depth-gated unsafe feature filter, so the pool can only grow. Only
    /// splits the pass never records are skipped (<= 1 non-empty child). No
    /// near-pure terminal, killer move, top-N, eqp or bound transfer is
    /// used. Costs are the ones the pass charges: a leaf costs its
    /// `base_objective` (same counts as `create_subproblem`), a split
    /// split_reg + path_cost(n) + Σ children (the lb-sum test). Hence m <= the cost
    /// of every tree the pass can build at `support`, and a `None` holds for
    /// all of them. Pruning uses only admissible bounds: a child costs at
    /// least min(leaf, split_reg + 2·leaf_reg + path) (each leaf costs >=
    /// leaf_reg), a k-way split at least split_reg + path + k·leaf_reg.
    ///
    /// Memo (`WorkerState::rash_lb_memo`, per solver): `Min(m)` answers
    /// every budget; `Above(b)` answers budgets with budget + RASH_TOL <= b
    /// with `None`, a wider budget re-searches.
    pub fn bounded_completion_min(
        &mut self,
        support: &SupportMask,
        used: &FeatureMask,
        depth: i32,
        budget: f64,
    ) -> Option<f64> {
        self.rash_completion_min(support, used, depth, budget + Self::RASH_TOL, None, None)
    }

    /// `bounded_completion_min` with the structural depth-1 memo structural key of
    /// `support` (remaining depth 1 only; `None` = the full-key memo).
    pub fn bounded_completion_min_keyed(
        &mut self,
        support: &SupportMask,
        used: &FeatureMask,
        depth: i32,
        budget: f64,
        skey: Option<RashD1Key>,
    ) -> Option<f64> {
        self.rash_completion_min(support, used, depth, budget + Self::RASH_TOL, None, skey)
    }

    /// Structural depth-1 memo: interned id of a parent support (`RashD1Key::pid`);
    /// equal supports (words compared) share an id.
    fn rash_lb_pid(&mut self, support: &SupportMask) -> u32 {
        // `stepb_batches` level base, then the `lb_threads` pass-1
        // base: their ids first; new supports get ids from the sum of their
        // lengths up (remapped at the merges).
        let mut base_len = 0u32;
        if let Some(l) = &self.worker.rash_level_base {
            if let Some(&id) = l.base.d1s.pids.get(support) {
                return id;
            }
            base_len = l.base.d1s.pids.len() as u32;
        }
        if let Some(b) = &self.worker.rash_base {
            if let Some(&id) = b.d1s.pids.get(support) {
                return id;
            }
            base_len += b.d1s.pids.len() as u32;
        }
        let pids = &mut self.worker.rash_lb_d1s.pids;
        let n = base_len + pids.len() as u32;
        *pids.entry(support.clone()).or_insert(n)
    }

    /// Structural depth-1 memo: structural keys of the children of a split of
    /// `parent` on feature `f` at `thresholds` (child b = f-ranks
    /// [cuts[b], cuts[b+1])), or None when a key does not apply: the switch
    /// is off, no mask_exact (`rash_lb_gate`), a threshold is not a store
    /// threshold, the arity is not `k`, or a field overflows.
    fn rash_d1_child_keys(
        &mut self,
        parent: &SupportMask,
        f: usize,
        thresholds: &[f64],
        k: usize,
    ) -> Option<smallvec::SmallVec<[RashD1Key; 8]>> {
        if !self.rash_lb_gate().0 || f > u16::MAX as usize {
            return None;
        }
        let cuts = self.rash_d1_sweep_cuts(f, thresholds)?;
        if cuts.len() != k + 1 || cuts[k] > u16::MAX as usize {
            return None;
        }
        let pid = self.rash_lb_pid(parent);
        Some(
            (0..k)
                .map(|b| RashD1Key {
                    pid,
                    f: f as u16,
                    lo: cuts[b] as u16,
                    hi: cuts[b + 1] as u16,
                })
                .collect(),
        )
    }

    /// Insert into the completion-lb memo: under the structural key `skey`
    /// when given (structural depth-1 memo; the two-generation map evicts its older
    /// half, counted in `n_lb_memo_evictions`), else `rash_lb_memo_put`.
    fn rash_lb_memo_put_any(
        &mut self,
        key: (SupportMask, i32),
        skey: Option<RashD1Key>,
        e: RashLbEntry,
    ) {
        match skey {
            Some(sk) => self.rash_d1s_put(sk, e),
            None => self.rash_lb_memo_put(key, e),
        }
    }

    /// Insert into the structural depth-1 memo (structural depth-1 memo; the
    /// two-generation map evicts its older half, `n_lb_memo_evictions`).
    fn rash_d1s_put(&mut self, sk: RashD1Key, e: RashLbEntry) {
        let cap =
            self.search.config.rash_tuning.lb_d1s_max_entries / self.worker.rash_lb_cap_div.max(1);
        let m = &mut self.worker.rash_lb_d1s.map;
        let before = m.evicted;
        m.set_cap(cap);
        m.insert(sk, e);
        self.worker.profile.n_lb_memo_evictions += m.evicted - before;
    }

    /// Insert into the completion-lb memo. lb memo cap: when the
    /// remaining-depth-1 entries exceed `rash_lb_memo_max_entries`, all of
    /// them are dropped (`n_lb_memo_evictions`). The memo is a pure cache
    /// (every entry is a fact about (support, depth) recomputed identically
    /// on a miss), so the values and the tree set do not change.
    fn rash_lb_memo_put(&mut self, key: (SupportMask, i32), e: RashLbEntry) {
        let d1 = key.1 == 1;
        if self.worker.rash_lb_memo.insert(key, e).is_some() || !d1 {
            return;
        }
        self.worker.rash_lb_memo_d1 += 1;
        if self.worker.rash_lb_memo_d1
            > (self.search.config.rash_tuning.lb_memo_max_entries
                / self.worker.rash_lb_cap_div.max(1))
            .max(1)
        {
            let before = self.worker.rash_lb_memo.len();
            self.worker.rash_lb_memo.retain(|k, _| k.1 != 1);
            // `retain` keeps the table's capacity: hand it back.
            self.worker.rash_lb_memo.shrink_to_fit();
            self.worker.profile.n_lb_memo_evictions +=
                (before - self.worker.rash_lb_memo.len()) as u64;
            self.worker.rash_lb_memo_d1 = 0;
        }
    }

    /// Leaf objective of a node with these rows (`create_subproblem`'s
    /// `base_objective`).
    pub fn rash_leaf_objective(&self, indices: &[u32]) -> f64 {
        let (counts, n_weighted) = self.class_counts_for_indices(indices);
        self.leaf_objective_from_counts(&counts, n_weighted)
    }

    /// Leaf cost (`rash_leaf_objective`) and row count of every child of
    /// `cand` at the rows `indices`, from one pass with the partition rule
    /// of `partition_into_flat` (no bins allocated).
    fn rash_child_leaf_costs(
        &self,
        indices: &[u32],
        feature_index: usize,
        thresholds: &[f64],
        gm: Option<&[usize]>,
    ) -> (Vec<f64>, Vec<usize>) {
        let n_bins = thresholds.len() + 1;
        let k = gm.map_or(n_bins, |g| g.iter().copied().max().unwrap_or(0) + 1);
        let col = self.search.store.feature_col(feature_index);
        let labels = self.search.store.labels();
        let weights = self.search.store.weights();
        let scale = self.search.n_data as f64;
        let mut counts = vec![[0i64; MAX_CLASSES]; k];
        let mut sizes = vec![0usize; k];
        for &idx in indices {
            let bin = Self::bin_index_right(thresholds, col[idx as usize]);
            let g = gm.map_or(bin, |m| m[bin]);
            sizes[g] += 1;
            let add = match weights {
                Some(w) => (w[idx as usize] * scale).round() as i64,
                None => 1,
            };
            counts[g][labels[idx as usize] as usize] += add;
        }
        let n_classes = self.search.config.n_classes;
        let costs = counts
            .iter()
            .map(|c| {
                let n_weighted = c[..n_classes].iter().sum::<i64>() as f64;
                self.leaf_objective_from_counts(c, n_weighted)
            })
            .collect();
        (costs, sizes)
    }

    /// Largest per-feature threshold count `rash_depth1_min_closed` handles;
    /// a node of a store with a larger feature falls back to Select.
    pub const RASH_LB_D1_MAX_T: usize = 512;

    /// Preconditions of the fast completion-lb paths, from the store and
    /// config (the solver caches them in `WorkerState::rash_lb_gate`):
    ///  0. mask_exact: no weights; Select's thresholds are store thresholds
    ///     (a keyed per-feature table `thresholds_per_feature` whose every
    ///     value is bitwise one of the store's thresholds of that feature,
    ///     and every feature missing from it constant, so Select's
    ///     node-local midpoints are empty); the store's thresholds of a
    ///     feature are >= 1e-12 apart (`find_threshold_base` matches within
    ///     1e-12, so it then finds the bitwise-equal one); and no row value
    ///     equals (or is NaN next to) a store threshold. Then the le-mask
    ///     partition (`x <= t` left, `DatasetStore::partition_support`)
    ///     equals the row partition of `partition_into_flat` / Select
    ///     (`x < t` left) for every pool split, so mask-based lb path's child
    ///     supports and leaf costs are the legacy path's.
    ///  1. d1_ok: mask_exact, no shape mode, no `FeatureType::Cat` feature
    ///     (group_map candidates merge non-adjacent bins), one label mask
    ///     per class, and every feature's store thresholds strictly
    ///     ascending (nested le-masks). Every Select split at depth 1 is then a
    ///     contiguous partition of the store thresholds with the same rows
    ///     per bin, which `rash_depth1_min_closed` minimises over.
    fn rash_lb_gate_compute(&self) -> (bool, bool) {
        let store = self.search.store.as_ref();
        let c = &self.search.config;
        let ths = store.feature_thresholds();
        if store.weights().is_some() || c.n_features > ths.len() {
            return (false, false);
        }
        let source_ok = match &c.thresholds_per_feature {
            None => false,
            Some(map) => (0..c.n_features).all(|f| match map.get(&f) {
                // Ascending (Select's `partition_point` assumes it) and
                // bitwise store thresholds.
                Some(v) => {
                    v.windows(2).all(|w| w[0] < w[1])
                        && v.iter()
                            .all(|t| ths[f].iter().any(|s| s.to_bits() == t.to_bits()))
                }
                None => {
                    let col = store.feature_col(f);
                    col.iter().all(|x| x.to_bits() == col[0].to_bits())
                }
            }),
        };
        let mut mask_exact = source_ok;
        'feat: for (f, t) in ths.iter().enumerate() {
            if !mask_exact {
                break;
            }
            if t.is_empty() {
                continue;
            }
            let mut sorted: Vec<f64> = t.to_vec();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            if sorted.iter().any(|x| x.is_nan()) || sorted.windows(2).any(|w| w[1] - w[0] < 1e-12) {
                mask_exact = false;
                break;
            }
            for &v in store.feature_col(f) {
                if v.is_nan()
                    || sorted
                        .binary_search_by(|x| x.partial_cmp(&v).unwrap())
                        .is_ok()
                {
                    mask_exact = false;
                    break 'feat;
                }
            }
        }
        let d1_ok = mask_exact
            && !c.shape_mode
            && !c
                .feature_types
                .iter()
                .any(|t| *t == types::FeatureType::Cat)
            && store.label_masks().len() == c.n_classes
            && ths.iter().all(|t| t.windows(2).all(|w| w[0] < w[1]))
            && store.feature_threshold_masks().len() == ths.len();
        (mask_exact, d1_ok)
    }

    /// `rash_lb_gate_compute`, cached per solver. The first call also
    /// records the store threshold statistics in the profile and, under
    /// d1_ok and depth-1 row walk, builds the row -> bin rank table
    /// (`rash_lb_rank_build`) for the closed form's row walk.
    pub fn rash_lb_gate(&mut self) -> (bool, bool) {
        if let Some(g) = self.worker.rash_lb_gate {
            return g;
        }
        let g = self.rash_lb_gate_compute();
        self.worker.rash_lb_gate = Some(g);
        {
            let ths = self.search.store.feature_thresholds();
            let p = &mut self.worker.profile;
            p.n_store_thresholds_max = ths.iter().map(|t| t.len() as u64).max().unwrap_or(0);
            p.n_store_thresholds_sum = ths.iter().map(|t| t.len() as u64).sum();
            p.n_store_threshold_features = ths.iter().filter(|t| !t.is_empty()).count() as u64;
        }
        if g.1
            && self
                .search
                .store
                .feature_threshold_masks()
                .iter()
                .all(|m| m.len() <= Self::RASH_LB_D1_MAX_T)
        {
            self.worker.rash_lb_rank = Some(Arc::new(self.rash_lb_rank_build()));
        }
        g
    }

    /// Depth-1 row walk: the store's row -> bin rank table (`RashLbRank`):
    /// rank = #{t : t < x} over the feature's ascending store thresholds,
    /// i.e. the number of le-masks (`x <= t`) NOT containing the row (no NaN
    /// and no row value on a threshold under d1_ok). Only meaningful under
    /// d1_ok with every T_f <= `RASH_LB_D1_MAX_T` (fits u16).
    pub fn rash_lb_rank_build(&self) -> RashLbRank {
        let store = self.search.store.as_ref();
        let ths = store.feature_thresholds();
        let wn = store.root_support().words.len();
        let (mut sum_tw, mut f_active) = (0usize, 0usize);
        let rank = ths
            .iter()
            .enumerate()
            .map(|(f, t)| {
                if t.is_empty() {
                    return Box::default();
                }
                sum_tw += t.len() * wn;
                f_active += 1;
                store
                    .feature_col(f)
                    .iter()
                    .map(|&x| t.partition_point(|&s| s < x) as u16)
                    .collect()
            })
            .collect();
        RashLbRank {
            rank,
            sum_tw,
            f_active,
        }
    }

    /// Depth-1 row walk crossover: a support of n_s rows is counted by
    /// walking its rows when `n_s · f_active < C · Σ_f T_f · words` (row
    /// walk: n_s rank lookups per feature; masks: T_f · words AND+popcounts
    /// per feature). A row step costs about twice a mask word, so the
    /// paths cross near n_s·F / (Σ T·W) ≈ 0.4–0.5; C = 0.4 takes the row
    /// walk only where it is the faster path.
    pub const RASH_LB_D1_ROWWALK_C: f64 = 0.4;

    /// Depth-1 closed form: exact minimum of `overhead + Σ_bins leaf(bin)` over
    /// ALL contiguous partitions of `support` into 2..=max_k bins of every
    /// feature's store thresholds (children as leaves), from the support's
    /// per-bin class counts. `None` when a precondition fails
    /// (`rash_lb_gate_compute` d1_ok) or a feature has more than
    /// `RASH_LB_D1_MAX_T` thresholds: the caller falls back to Select.
    /// Speed paths from two-pass / depth-1 row walk
    /// (`rash_depth1_min_closed_ext`).
    ///
    /// Exactness: the completion-lb value is only an admissible lower bound
    /// (the tree set does not depend on its tightness). At remaining depth 1
    /// the exact minimum over the lb option set is min(leaf, best single
    /// split of the Select pool with leaf children). Under d1_ok every pool
    /// split is a contiguous partition of the store thresholds with the
    /// same rows per bin, so this minimum over all of them (a superset of
    /// the pool) is <= the pool minimum: admissible. Bin costs are
    /// `leaf_objective_from_counts` of the bin's counts (an empty bin costs
    /// leaf_reg) and are summed left to right after `overhead` as the lb-sum
    /// test does, so every candidate value is bitwise the cost that test
    /// gives that partition.
    ///
    /// Per feature with T thresholds: prefix counts at the T + 2 cut
    /// boundaries, then an interval DP over cut positions (layer j = best
    /// left-to-right sum of j bins ending at a cut), O(max_k · T²). The
    /// prefix counts are popcounts of support ∧ le-mask (∧ label mask; the
    /// last class by difference), or, for a small support, a histogram of
    /// the support's rows by rank (`RashLbRank`), prefix-summed: the same
    /// integers. Prunes (admissible: skipped values are >= the best so far,
    /// by monotone rounding of sums of terms >= leaf_reg): extending a layer
    /// entry whose sum + two more bins already reaches the best (the
    /// first-bin prune at j = 1); and every j >= 3 of a feature when
    /// overhead + fine + 3·leaf_reg reaches the best + 1e-9 (slack for
    /// rounding), with fine = the misclassification of the T + 1 threshold
    /// cells (merging cells never lowers it). The result is the exact
    /// minimum of the candidate values whatever the prune order, so every
    /// speed path returns it bitwise.
    pub fn rash_depth1_min_closed(
        &self,
        support: &SupportMask,
        n_node: usize,
        overhead: f64,
    ) -> Option<f64> {
        self.rash_depth1_min_closed_ext(support, n_node, overhead, true, None)
            .map(|r| r.0)
    }

    /// `rash_depth1_min_closed` with its speed paths chosen explicitly
    /// (tests): `twopass` = two-pass (2-bin minimum of
    /// every feature first, then the >= 3-bin layers only for features
    /// whose floor `overhead + fine + 3·leaf_reg` is below that global best
    /// + 1e-9); `rowwalk` = None for the adaptive crossover
    /// (`RASH_LB_D1_ROWWALK_C`, needs the gate's rank table), Some(true) /
    /// Some(false) to always / never walk the rows. Returns (value, rows
    /// walked). Every choice returns the same value bitwise.
    pub fn rash_depth1_min_closed_ext(
        &self,
        support: &SupportMask,
        n_node: usize,
        overhead: f64,
        twopass: bool,
        rowwalk: Option<bool>,
    ) -> Option<(f64, bool)> {
        let d1_ok = match self.worker.rash_lb_gate {
            Some(g) => g.1,
            None => self.rash_lb_gate_compute().1,
        };
        if !d1_ok {
            return None;
        }
        let store = self.search.store.as_ref();
        let masks = store.feature_threshold_masks();
        if masks.iter().any(|m| m.len() > Self::RASH_LB_D1_MAX_T) {
            return None;
        }
        let nc = self.search.config.n_classes;
        let labels = store.label_masks();
        let words = support.words.as_ref();
        let (tot_counts, tot_n) = store.count_classes(support);
        debug_assert_eq!(tot_n, n_node);
        let wn = words.len();
        let lab_masks: smallvec::SmallVec<[&[u64]; MAX_CLASSES]> =
            labels[..nc - 1].iter().map(|m| &m[..wn]).collect();
        let built_rank;
        let rank: Option<&RashLbRank> = match rowwalk {
            Some(false) => None,
            Some(true) => match self.worker.rash_lb_rank.as_deref() {
                Some(r) => Some(r),
                None => {
                    built_rank = self.rash_lb_rank_build();
                    Some(&built_rank)
                }
            },
            None => self.worker.rash_lb_rank.as_deref().filter(|r| {
                ((n_node * r.f_active) as f64) < Self::RASH_LB_D1_ROWWALK_C * r.sum_tw as f64
            }),
        };
        let best = RASH_D1_SCRATCH.with(|cell| {
            let mut guard = cell.borrow_mut();
            let RashD1Scratch {
                cum,
                cum_n,
                off,
                fine3,
                prev,
                next,
                idx,
                lab,
                hist,
                int,
            } = &mut *guard;
            cum.clear();
            cum_n.clear();
            off.clear();
            if rank.is_some() {
                // The support's rows and their labels, once per call.
                let row_labels = store.labels();
                idx.clear();
                lab.clear();
                for (wi, &w0) in words.iter().enumerate() {
                    let mut w = w0;
                    while w != 0 {
                        let i = wi * 64 + w.trailing_zeros() as usize;
                        idx.push(i as u32);
                        lab.push(row_labels[i]);
                        w &= w - 1;
                    }
                }
                debug_assert_eq!(idx.len(), n_node);
            }
            // Counting phase: every feature's boundaries.
            for (f, le) in masks.iter().enumerate() {
                let t_n = le.len();
                off.push(cum_n.len());
                if t_n == 0 {
                    continue;
                }
                // Boundaries: 0 = start, b in 1..=T = after threshold b-1
                // (rows x <= t), T + 1 = end.
                cum.extend(std::iter::repeat(0).take(nc));
                cum_n.push(0);
                match rank {
                    Some(r) => {
                        Self::rash_d1_row_counts(&r.rank[f], idx, lab, nc, t_n, hist, cum, cum_n);
                    }
                    None => Self::rash_d1_mask_counts(words, le, &lab_masks, nc, cum, cum_n),
                }
                cum.extend_from_slice(&tot_counts[..nc]);
                cum_n.push(tot_n as i64);
            }
            self.rash_d1_dp_switched(cum, cum_n, off, fine3, prev, next, int, overhead, twopass)
        });
        Some((best, rank.is_some()))
    }

    /// DP phase of `rash_depth1_min_closed_ext` from the counting phase's
    /// boundaries (`cum` / `cum_n`, per feature f with T_f > 0 its T_f + 2
    /// boundaries at `off[f]`): per feature in order, the finest-partition
    /// floor and the layers (2-bin only under `twopass`), then under
    /// `twopass` the >= 3-bin layers of the features whose floor is below
    /// the best + 1e-9. Each feature's DP reads only its own boundaries and
    /// `best`, so the result does not depend on how the counts were made.
    #[allow(clippy::too_many_arguments)]
    fn rash_d1_dp(
        &self,
        cum: &[i64],
        cum_n: &[i64],
        off: &[usize],
        fine3: &mut Vec<f64>,
        prev: &mut Vec<f64>,
        next: &mut Vec<f64>,
        overhead: f64,
        twopass: bool,
    ) -> f64 {
        let masks = self.search.store.feature_threshold_masks();
        let nc = self.search.config.n_classes;
        let n_total = self.search.n_total as f64;
        let leaf_reg = self.search.config.reg_alpha * self.search.config.reg;
        fine3.clear();
        let mut best = f64::INFINITY;
        for (f, le) in masks.iter().enumerate() {
            let t_n = le.len();
            if t_n == 0 {
                fine3.push(f64::INFINITY);
                continue;
            }
            let base = off[f];
            let fc = &cum[base * nc..];
            let fn_ = &cum_n[base..];
            // Misclassification of the finest partition (the T + 1 cells).
            let mut fine_m: i64 = 0;
            for b in 1..=t_n + 1 {
                let n = fn_[b] - fn_[b - 1];
                let mx = (0..nc)
                    .map(|cl| fc[b * nc + cl] - fc[(b - 1) * nc + cl])
                    .max()
                    .unwrap_or(0);
                fine_m += n - mx;
            }
            let fine_floor3 = overhead + fine_m as f64 / n_total + 3.0 * leaf_reg;
            fine3.push(fine_floor3);
            // Two-pass: only the 2-bin partitions now.
            let j_hi = if twopass { 2 } else { usize::MAX };
            self.rash_d1_layers(
                fc,
                fn_,
                t_n,
                overhead,
                fine_floor3,
                2,
                j_hi,
                &mut best,
                prev,
                next,
            );
        }
        if twopass && self.search.config.max_k >= 3 {
            for (f, le) in masks.iter().enumerate() {
                let t_n = le.len();
                if t_n < 2 || !(fine3[f] < best + 1e-9) {
                    continue;
                }
                let base = off[f];
                let (fc, fn_) = (&cum[base * nc..], &cum_n[base..]);
                self.rash_d1_layers(
                    fc,
                    fn_,
                    t_n,
                    overhead,
                    fine3[f],
                    3,
                    usize::MAX,
                    &mut best,
                    prev,
                    next,
                );
            }
        }
        best
    }

    /// `rash_d1_dp` with the integer-first DP (`rash_d1_dp_int`, binary /
    /// 3-class) when its guards hold, else the reference DP (`rash_d1_dp`);
    /// counted in `int` (answered / fallen back).
    #[allow(clippy::too_many_arguments)]
    fn rash_d1_dp_switched(
        &self,
        cum: &[i64],
        cum_n: &[i64],
        off: &[usize],
        fine3: &mut Vec<f64>,
        prev: &mut Vec<f64>,
        next: &mut Vec<f64>,
        int: &mut RashD1IntScratch,
        overhead: f64,
        twopass: bool,
    ) -> f64 {
        if rash_d1_intdp_on() {
            let r = match self.search.config.n_classes {
                2 => self.rash_d1_dp_int::<2>(cum, cum_n, off, prev, next, int, overhead),
                3 => self.rash_d1_dp_int::<3>(cum, cum_n, off, prev, next, int, overhead),
                _ => None,
            };
            match r {
                Some(b) => {
                    int.n_int += 1;
                    return b;
                }
                None => int.n_fallback += 1,
            }
        }
        self.rash_d1_dp(cum, cum_n, off, fine3, prev, next, overhead, twopass)
    }

    /// Integer-first depth-1 DP: `rash_d1_dp`'s value bitwise, integer first. With
    /// m(a, b) = n - max_cl c_cl the misclassification of the bin between
    /// boundaries a < b, a partition's value is overhead + Σ_bins tab[m]
    /// (left to right, `tab[m] = m / n_total + leaf_reg` = the bin's
    /// `leaf_objective_from_counts`: n - max is exact in f64, an empty bin
    /// costs tab[0] = leaf_reg). m(a, b) = min_cl (Q_cl[b] - Q_cl[a]), Q_cl =
    /// prefix count of the rows not of class cl, so the integer layer minima
    /// L_{i+1}[b] = min_cl (Q_cl[b] + min_{a<b} (L_i[a] - Q_cl[a])) take
    /// O(T·NC) per layer (running minima), likewise the suffix minima
    /// Suf_q[a] (boundaries a..end into q bins). Within one bin count j a
    /// partition with a larger integer total has a strictly larger float
    /// value (the step 1/n_total dwarfs the rounding: guards below), so the
    /// float minimum over j-bin partitions is attained on integer-optimal
    /// ones, and every prefix of one is integer-optimal at its cut: the
    /// float DP (the reference's left-to-right sums, prefix minima per cut)
    /// runs only over cuts a with L_i[a] + Suf_{j-i}[a] = s*_j and
    /// transitions with L_i[a] + m(a, b) = L_{i+1}[b] (all float ties among
    /// them evaluated). The reference returns the exact minimum over all
    /// (feature, j, partition) values whatever its prunes; so does this one
    /// (prunes skip only (feature, j) whose real floor overhead +
    /// s/n_total + j·leaf_reg reaches the best + 1e-9), so the two agree
    /// bitwise. None (the caller runs the reference) unless: leaf_reg >= 0
    /// (the reference's prunes assume bins >= leaf_reg >= 0); V =
    /// |overhead| + max_k·(1 + leaf_reg) < 1e3 (rounding << the 1e-9
    /// slack) and (max_k + 2)·V·n_total·ε < 0.5 (rounding << the step
    /// 1/n_total); n_total <= 2^32; tab[0] is bitwise leaf_reg (not -0.0);
    /// max_k >= 2.
    #[allow(clippy::too_many_arguments)]
    fn rash_d1_dp_int<const NC: usize>(
        &self,
        cum: &[i64],
        cum_n: &[i64],
        off: &[usize],
        prev: &mut Vec<f64>,
        next: &mut Vec<f64>,
        int: &mut RashD1IntScratch,
        overhead: f64,
    ) -> Option<f64> {
        let masks = self.search.store.feature_threshold_masks();
        let n_total = self.search.n_total as f64;
        let leaf_reg = self.search.config.reg_alpha * self.search.config.reg;
        let max_k = self.search.config.max_k;
        let vmag = overhead.abs() + (max_k as f64) * (1.0 + leaf_reg.abs());
        if !(leaf_reg >= 0.0)
            || !(vmag < 1e3)
            || !(((max_k + 2) as f64) * vmag * n_total * f64::EPSILON < 0.5)
            || self.search.n_total > (1usize << 32)
            || max_k < 2
        {
            return None;
        }
        let RashD1IntScratch {
            tab,
            tab_key,
            l,
            suf,
            q,
            ..
        } = int;
        // Cost table for m in 0..=max boundary count (cached per key).
        let key = (n_total.to_bits(), leaf_reg.to_bits());
        if *tab_key != key {
            tab.clear();
            *tab_key = key;
        }
        let max_n = cum_n.iter().copied().max().unwrap_or(0) as usize;
        while tab.len() <= max_n {
            let m = tab.len();
            tab.push((m as f64) / n_total + leaf_reg);
        }
        if tab[0].to_bits() != leaf_reg.to_bits() {
            return None;
        }
        let tab: &[f64] = &tab[..];
        let mut best = f64::INFINITY;
        // Pass 1: two bins (float over the integer-optimal cuts), every
        // feature first.
        for (f, le) in masks.iter().enumerate() {
            let t_n = le.len();
            if t_n == 0 {
                continue;
            }
            let base = off[f];
            let e = t_n + 1;
            let fc = &cum[base * NC..(base + e + 1) * NC];
            let fnn = &cum_n[base..base + e + 1];
            let mut s2 = i64::MAX;
            for a in 1..=t_n {
                let s = (rash_d1_m::<NC>(fc, fnn, 0, a) + rash_d1_m::<NC>(fc, fnn, a, e)) as i64;
                s2 = s2.min(s);
            }
            if overhead + (s2 as f64) / n_total + 2.0 * leaf_reg >= best + 1e-9 {
                continue;
            }
            for a in 1..=t_n {
                let (m1, m2) = (
                    rash_d1_m::<NC>(fc, fnn, 0, a),
                    rash_d1_m::<NC>(fc, fnn, a, e),
                );
                if (m1 + m2) as i64 == s2 {
                    let v = overhead + (tab[m1] + tab[m2]);
                    if v < best {
                        best = v;
                    }
                }
            }
        }
        if max_k < 3 {
            return Some(best);
        }
        // Pass 2: j >= 3 bins.
        for (f, le) in masks.iter().enumerate() {
            let t_n = le.len();
            if t_n < 2 {
                continue;
            }
            let base = off[f];
            let e = t_n + 1;
            let fc = &cum[base * NC..(base + e + 1) * NC];
            let fnn = &cum_n[base..base + e + 1];
            let j_max = max_k.min(t_n + 1);
            // Floor: merging cells never lowers the misclassification of
            // the finest partition (the T + 1 cells).
            let mut fine_m: i64 = 0;
            for b in 1..=e {
                fine_m += rash_d1_m::<NC>(fc, fnn, b - 1, b) as i64;
            }
            if overhead + (fine_m as f64) / n_total + 3.0 * leaf_reg >= best + 1e-9 {
                continue;
            }
            let w = e + 1;
            // q[x · NC + cl] = Q_cl[x] = cum_n[x] - cum[x][cl].
            q.clear();
            for x in 0..w {
                for cl in 0..NC {
                    q.push(fnn[x] - fc[x * NC + cl]);
                }
            }
            // l[i · w + b]: boundaries 0..b into i bins, i in 1..j_max, b in
            // i..=t_n.
            l.clear();
            l.resize(j_max * w, i64::MAX);
            for b in 1..=t_n {
                l[w + b] = rash_d1_m::<NC>(fc, fnn, 0, b) as i64;
            }
            for i in 1..j_max - 1 {
                let mut run = [i64::MAX; NC];
                for b in (i + 1)..=t_n {
                    let a = b - 1;
                    let la = l[i * w + a];
                    let mut v = i64::MAX;
                    for cl in 0..NC {
                        run[cl] = run[cl].min(la - q[a * NC + cl]);
                        v = v.min(q[b * NC + cl] + run[cl]);
                    }
                    l[(i + 1) * w + b] = v;
                }
            }
            // suf[qn · w + a]: boundaries a..e into qn bins (cuts in (a, t_n]).
            suf.clear();
            suf.resize(j_max * w, i64::MAX);
            for a in 0..=t_n {
                suf[w + a] = rash_d1_m::<NC>(fc, fnn, a, e) as i64;
            }
            for qn in 2..j_max {
                let mut run = [i64::MAX; NC];
                let a_hi = t_n + 1 - qn;
                for a in (0..=a_hi).rev() {
                    let b = a + 1;
                    let sb = suf[(qn - 1) * w + b];
                    let mut v = i64::MAX;
                    for cl in 0..NC {
                        run[cl] = run[cl].min(q[b * NC + cl] + sb);
                        v = v.min(run[cl] - q[a * NC + cl]);
                    }
                    suf[qn * w + a] = v;
                }
            }
            for j in 3..=j_max {
                // s*_j: j - 1 bins ending at cut a, then the last bin.
                let mut sj = i64::MAX;
                for a in (j - 1)..=t_n {
                    sj = sj.min(l[(j - 1) * w + a] + suf[w + a]);
                }
                if overhead + (sj as f64) / n_total + (j as f64) * leaf_reg >= best + 1e-9 {
                    // Every j' > j is >= the finest floor with j' bins.
                    if overhead + (fine_m as f64) / n_total + ((j + 1) as f64) * leaf_reg
                        >= best + 1e-9
                    {
                        break;
                    }
                    continue;
                }
                // Float DP over the tight cuts. Layer 1.
                prev.clear();
                prev.resize(w, f64::INFINITY);
                let qn1 = j - 1;
                for a in 1..=(t_n + 1 - qn1) {
                    if l[w + a] + suf[qn1 * w + a] == sj {
                        prev[a] = tab[l[w + a] as usize];
                    }
                }
                for i in 1..(j - 1) {
                    // Layer i -> i + 1 (bins ending at cut b).
                    next.clear();
                    next.resize(w, f64::INFINITY);
                    let rem = j - (i + 1);
                    for a in i..=(t_n + 1 - (j - i)) {
                        let pa = prev[a];
                        if !pa.is_finite() {
                            continue;
                        }
                        let la = l[i * w + a];
                        for b in (a + 1)..=(t_n + 1 - rem) {
                            let lb = l[(i + 1) * w + b];
                            if lb + suf[rem * w + b] != sj {
                                continue;
                            }
                            let m = rash_d1_m::<NC>(fc, fnn, a, b);
                            if la + m as i64 != lb {
                                continue;
                            }
                            let v = pa + tab[m];
                            if v < next[b] {
                                next[b] = v;
                            }
                        }
                    }
                    std::mem::swap(prev, next);
                }
                for a in (j - 1)..=t_n {
                    let pa = prev[a];
                    if !pa.is_finite() {
                        continue;
                    }
                    let v = overhead + (pa + tab[suf[w + a] as usize]);
                    if v < best {
                        best = v;
                    }
                }
            }
        }
        Some(best)
    }

    /// Integer-first depth-1 DP: `rash_d1_sweep_child`'s boundaries (the child's
    /// histogram pb - pa prefix-summed per feature), written in place with
    /// a const class count: the same integers as the generic loop.
    /// Returns Ψ = min over the features g with T_g > 0 of the child's
    /// finest g-grid error Σ_r (n_r − max_cl h_r) (`i64::MAX` if none;
    /// Ψ_dy bound, `interval_lb`).
    #[allow(clippy::too_many_arguments)]
    fn rash_d1_sweep_bounds<const NC: usize>(
        masks: &[Arc<[Arc<[u64]>]>],
        hoff: &[usize],
        pa: &[i32],
        pb: &[i32],
        n_node: usize,
        cum: &mut Vec<i64>,
        cum_n: &mut Vec<i64>,
        off: &mut Vec<usize>,
    ) -> i64 {
        let total: usize = masks
            .iter()
            .map(|m| if m.is_empty() { 0 } else { m.len() + 2 })
            .sum();
        cum.resize(total * NC, 0);
        cum_n.resize(total, 0);
        let mut pos = 0usize;
        let mut psi = i64::MAX;
        for (g, m) in masks.iter().enumerate() {
            off.push(pos);
            let t_n = m.len();
            if t_n == 0 {
                continue;
            }
            // Boundary c in 0..=T + 1 counts the child rows of g-rank < c.
            let o = hoff[g];
            let sa = &pa[o..o + (t_n + 1) * NC];
            let sb = &pb[o..o + (t_n + 1) * NC];
            let dc = &mut cum[pos * NC..(pos + t_n + 2) * NC];
            let dn = &mut cum_n[pos..pos + t_n + 2];
            dc[..NC].fill(0);
            dn[0] = 0;
            let mut run = [0i64; NC];
            let mut run_n = 0i64;
            let mut fine = 0i64;
            for r in 0..=t_n {
                let (mut n_r, mut mx) = (0i64, 0i64);
                for cl in 0..NC {
                    let h = (sb[r * NC + cl] - sa[r * NC + cl]) as i64;
                    run[cl] += h;
                    run_n += h;
                    n_r += h;
                    mx = mx.max(h);
                }
                fine += n_r - mx;
                dc[(r + 1) * NC..(r + 2) * NC].copy_from_slice(&run);
                dn[r + 1] = run_n;
            }
            debug_assert_eq!(run_n, n_node as i64);
            psi = psi.min(fine);
            pos += t_n + 2;
        }
        psi
    }

    /// Depth-1 sweep: per-parent state of a completion-lb scan at
    /// remaining depth 2 (`rash_lb_scan`), or None when the sweep does not
    /// apply: the gate's d1_ok, the
    /// row -> bin rank table built (d1_ok implies no weights), every T_f <=
    /// `RASH_LB_D1_MAX_T`.
    fn rash_d1_sweep_new(&self, parent: &SupportMask) -> Option<RashD1Sweep> {
        if !matches!(self.worker.rash_lb_gate, Some((_, true))) {
            return None;
        }
        let rank = Arc::clone(self.worker.rash_lb_rank.as_ref()?);
        let masks = self.search.store.feature_threshold_masks();
        if masks.iter().any(|m| m.len() > Self::RASH_LB_D1_MAX_T) {
            return None;
        }
        let nc = self.search.config.n_classes;
        let mut hoff = Vec::with_capacity(masks.len());
        let mut hlen = 0usize;
        for m in masks.iter() {
            hoff.push(hlen);
            if !m.is_empty() {
                hlen += (m.len() + 1) * nc;
            }
        }
        Some(RashD1Sweep {
            parent: parent.clone(),
            rank,
            rows: Vec::new(),
            parent_hist: None,
            hoff,
            hlen,
            groups: (0..masks.len())
                .map(|_| RashD1SweepGroup::default())
                .collect(),
        })
    }

    /// Depth-1 sweep (tests): the swept closed-form values
    /// (`rash_d1_sweep_child`, state built at once) of every child of every
    /// split of `parent` on feature `f` in `splits` (thresholds, child row
    /// counts), in bin order, all from one sweep (one group); None when the
    /// sweep does not apply (`rash_d1_sweep_new` / `rash_d1_sweep_cuts`).
    pub fn rash_d1_sweep_values(
        &mut self,
        parent: &SupportMask,
        f: usize,
        splits: &[(Vec<f64>, Vec<usize>)],
        overhead: f64,
        twopass: bool,
    ) -> Option<Vec<Vec<f64>>> {
        self.rash_lb_gate();
        let mut sw = self.rash_d1_sweep_new(parent)?;
        sw.groups[f].work = f64::INFINITY;
        let mut out = Vec::with_capacity(splits.len());
        for (ths, sizes) in splits {
            let cuts = self.rash_d1_sweep_cuts(f, ths)?;
            if cuts.len() != sizes.len() + 1 {
                return None;
            }
            out.push(
                (0..sizes.len())
                    .map(|b| {
                        self.rash_d1_sweep_child(
                            &mut sw,
                            f,
                            cuts[b],
                            cuts[b + 1],
                            sizes[b],
                            overhead,
                            twopass,
                        )
                    })
                    .collect::<Option<Vec<f64>>>()?,
            );
        }
        Some(out)
    }

    /// Depth-1 sweep: the child's f-rank interval [a, b) (cut boundaries
    /// of `RashLbRank`: rank = #{store thresholds < x}) for bin `bin` of a
    /// split on feature `f` at `thresholds`, or None when a threshold is not
    /// bitwise a store threshold of f (the child is then counted directly).
    /// Store index i of threshold t: its le-mask holds the rows of rank <=
    /// i (no row value equals a threshold under d1_ok), so the cuts are 0,
    /// i_0 + 1, ..., i_last + 1, T_f + 1.
    fn rash_d1_sweep_cuts(
        &self,
        f: usize,
        thresholds: &[f64],
    ) -> Option<smallvec::SmallVec<[usize; 8]>> {
        let ths = self.search.store.feature_thresholds().get(f)?;
        let mut cuts: smallvec::SmallVec<[usize; 8]> = smallvec::SmallVec::new();
        cuts.push(0);
        for t in thresholds {
            let i = ths.partition_point(|s| *s < *t);
            if i >= ths.len()
                || ths[i].to_bits() != t.to_bits()
                || cuts.last().is_some_and(|&c| c > i)
            {
                return None;
            }
            cuts.push(i + 1);
        }
        cuts.push(ths.len() + 1);
        Some(cuts)
    }

    /// Depth-1 sweep: `rash_depth1_min_closed_ext`'s value for the child
    /// of `sw.parent` holding its rows of f-rank in [a, b) (`n_node` rows),
    /// from the sweep instead of counting the child: the parent's rows are
    /// bucketed by f-rank once per (parent, f); P(c) = per-feature class
    /// histogram (by rank) of the parent rows with f-rank < c, cached at
    /// every boundary used and built from the nearest cached boundary
    /// (walking the smaller side; P(0) = 0, P(T_f + 1) = the parent's);
    /// the child's histogram is P(b) - P(a), prefix-summed into the
    /// boundaries the counting phase gives (same integers), then the same
    /// DP (`rash_d1_dp`). Returns None while the group's directly counted
    /// children have cost less (estimated) than building its sweep
    /// (rent or buy; the caller then counts the child itself).
    #[allow(clippy::too_many_arguments)]
    fn rash_d1_sweep_child(
        &self,
        sw: &mut RashD1Sweep,
        f: usize,
        a: usize,
        b: usize,
        n_node: usize,
        overhead: f64,
        twopass: bool,
    ) -> Option<f64> {
        if sw.groups[f].state.is_none() {
            // Rent or buy: count children directly until their estimated
            // counting work (row walk n·F or the masks' C·Σ T·W, the
            // closed form's own crossover) reaches the sweep's build
            // work (the parent's histogram n_p·F if not built yet, plus
            // bucketing 2·n_p); then build it.
            let r = &sw.rank;
            let direct =
                ((n_node * r.f_active) as f64).min(Self::RASH_LB_D1_ROWWALK_C * r.sum_tw as f64);
            let n_p = sw.parent.n_samples as f64;
            let build = 2.0 * n_p
                + if sw.parent_hist.is_none() {
                    n_p * r.f_active as f64
                } else {
                    0.0
                };
            let g = &mut sw.groups[f];
            g.work += direct;
            if g.work < build {
                return None;
            }
        }
        let store = self.search.store.as_ref();
        let masks = store.feature_threshold_masks();
        let labels = store.labels();
        let nc = self.search.config.n_classes;
        let (rk, hoff, hlen) = (&sw.rank.rank, &sw.hoff, sw.hlen);
        // The parent's rows and histogram, once per parent.
        if sw.parent_hist.is_none() {
            sw.rows = store.support_to_indices(&sw.parent);
            let mut h = vec![0i32; hlen].into_boxed_slice();
            for &r in &sw.rows {
                let (r, l) = (r as usize, labels[r as usize] as usize);
                for (g, m) in masks.iter().enumerate() {
                    if !m.is_empty() {
                        h[hoff[g] + rk[g][r] as usize * nc + l] += 1;
                    }
                }
            }
            sw.parent_hist = Some(h);
        }
        let t_f = masks[f].len();
        let rows = &sw.rows;
        let group = &mut sw.groups[f];
        let st = group.state.get_or_insert_with(|| {
            // Counting sort of the parent's rows by f-rank.
            let mut start = vec![0usize; t_f + 2];
            for &r in rows {
                start[rk[f][r as usize] as usize + 1] += 1;
            }
            for c in 1..start.len() {
                start[c] += start[c - 1];
            }
            let mut pos = start.clone();
            let mut by_rank = vec![0u32; rows.len()];
            for &r in rows {
                let k = rk[f][r as usize] as usize;
                by_rank[pos[k]] = r;
                pos[k] += 1;
            }
            let mut cache: Vec<Option<Box<[i32]>>> = vec![None; t_f + 2];
            cache[0] = Some(vec![0i32; hlen].into_boxed_slice());
            cache[t_f + 1] = sw.parent_hist.clone();
            RashD1SweepState {
                start,
                rows: by_rank,
                cache,
            }
        });
        for c in [a, b] {
            if st.cache[c].is_some() {
                continue;
            }
            let (mut lo, mut hi) = (c, c);
            while st.cache[lo].is_none() {
                lo -= 1;
            }
            while st.cache[hi].is_none() {
                hi += 1;
            }
            let (s_lo, s_c, s_hi) = (st.start[lo], st.start[c], st.start[hi]);
            let (mut h, walk, add) = if s_c - s_lo <= s_hi - s_c {
                (st.cache[lo].clone().unwrap(), &st.rows[s_lo..s_c], true)
            } else {
                (st.cache[hi].clone().unwrap(), &st.rows[s_c..s_hi], false)
            };
            for &r in walk {
                let (r, l) = (r as usize, labels[r as usize] as usize);
                for (g, m) in masks.iter().enumerate() {
                    if !m.is_empty() {
                        let cell = &mut h[hoff[g] + rk[g][r] as usize * nc + l];
                        if add {
                            *cell += 1;
                        } else {
                            *cell -= 1;
                        }
                    }
                }
            }
            st.cache[c] = Some(h);
        }
        let (pa, pb) = (
            st.cache[a].as_deref().unwrap(),
            st.cache[b].as_deref().unwrap(),
        );
        let best = RASH_D1_SCRATCH.with(|cell| {
            let mut guard = cell.borrow_mut();
            let RashD1Scratch {
                cum,
                cum_n,
                off,
                fine3,
                prev,
                next,
                int,
                ..
            } = &mut *guard;
            cum.clear();
            cum_n.clear();
            off.clear();
            let intdp = rash_d1_intdp_on();
            if intdp && nc == 2 {
                let _ =
                    Self::rash_d1_sweep_bounds::<2>(masks, hoff, pa, pb, n_node, cum, cum_n, off);
            } else if intdp && nc == 3 {
                let _ =
                    Self::rash_d1_sweep_bounds::<3>(masks, hoff, pa, pb, n_node, cum, cum_n, off);
            } else {
                for (g, m) in masks.iter().enumerate() {
                    off.push(cum_n.len());
                    let t_n = m.len();
                    if t_n == 0 {
                        continue;
                    }
                    // Boundary c in 0..=T + 1 counts the child rows of g-rank < c.
                    cum.extend(std::iter::repeat(0).take(nc));
                    cum_n.push(0);
                    let o = hoff[g];
                    let mut run = [0i64; MAX_CLASSES];
                    let mut run_n = 0i64;
                    for r in 0..=t_n {
                        for cl in 0..nc {
                            let h = (pb[o + r * nc + cl] - pa[o + r * nc + cl]) as i64;
                            run[cl] += h;
                            run_n += h;
                        }
                        cum.extend_from_slice(&run[..nc]);
                        cum_n.push(run_n);
                    }
                    debug_assert_eq!(run_n, n_node as i64);
                }
            }
            self.rash_d1_dp_switched(cum, cum_n, off, fine3, prev, next, int, overhead, twopass)
        });
        Some(best)
    }

    /// Leaf cost of the rows between boundaries a < b (`cum` holds nc class
    /// counts per boundary).
    #[inline]
    fn rash_d1_bin_cost(&self, cum: &[i64], cum_n: &[i64], a: usize, b: usize) -> f64 {
        let nc = self.search.config.n_classes;
        let mut cc = [0i64; MAX_CLASSES];
        for cl in 0..nc {
            cc[cl] = cum[b * nc + cl] - cum[a * nc + cl];
        }
        self.leaf_objective_from_counts(&cc, (cum_n[b] - cum_n[a]) as f64)
    }

    /// Prefix counts at boundaries 1..=T (appended to `cum` / `cum_n`) from
    /// popcounts of support ∧ le-mask (∧ label mask; the last class by
    /// difference).
    fn rash_d1_mask_counts(
        words: &[u64],
        le: &[Arc<[u64]>],
        lab: &[&[u64]],
        nc: usize,
        cum: &mut Vec<i64>,
        cum_n: &mut Vec<i64>,
    ) {
        let wn = words.len();
        for m in le.iter() {
            let m = &m[..wn];
            let mut cc = [0i64; MAX_CLASSES];
            let n = if nc == 2 {
                let l0 = lab[0];
                let (mut n, mut c0) = (0i64, 0i64);
                for w in 0..wn {
                    let x = words[w] & m[w];
                    n += x.count_ones() as i64;
                    c0 += (x & l0[w]).count_ones() as i64;
                }
                cc[0] = c0;
                n
            } else {
                for cl in 0..nc - 1 {
                    cc[cl] = words
                        .iter()
                        .zip(m)
                        .zip(lab[cl])
                        .map(|((w, t), l)| (w & t & l).count_ones())
                        .sum::<u32>() as i64;
                }
                words
                    .iter()
                    .zip(m)
                    .map(|(w, t)| (w & t).count_ones())
                    .sum::<u32>() as i64
            };
            // Every row has exactly one label: the last class by difference.
            cc[nc - 1] = n - cc[..nc - 1].iter().sum::<i64>();
            cum.extend_from_slice(&cc[..nc]);
            cum_n.push(n);
        }
    }

    /// Prefix counts at boundaries 1..=T (appended to `cum` / `cum_n`) from a
    /// histogram of the support's rows `idx` (labels `lab`) by rank `rk`:
    /// boundary b counts the rows of rank < b (= rows in le-mask b - 1).
    #[allow(clippy::too_many_arguments)]
    fn rash_d1_row_counts(
        rk: &[u16],
        idx: &[u32],
        lab: &[u8],
        nc: usize,
        t_n: usize,
        hist: &mut Vec<i64>,
        cum: &mut Vec<i64>,
        cum_n: &mut Vec<i64>,
    ) {
        hist.clear();
        hist.resize((t_n + 1) * nc, 0);
        for (&i, &l) in idx.iter().zip(lab.iter()) {
            hist[rk[i as usize] as usize * nc + l as usize] += 1;
        }
        let mut run = [0i64; MAX_CLASSES];
        let mut run_n = 0i64;
        for b in 1..=t_n {
            for cl in 0..nc {
                let h = hist[(b - 1) * nc + cl];
                run[cl] += h;
                run_n += h;
            }
            cum.extend_from_slice(&run[..nc]);
            cum_n.push(run_n);
        }
    }

    /// Interval DP of `rash_depth1_min_closed` for one feature (`cum` /
    /// `cum_n` its T + 2 boundaries): updates `best` with every partition
    /// into j bins, j in j_lo..=min(j_hi, max_k, T + 1) (j_lo in {2, 3}),
    /// building layers 1..j-1 as needed, with the prunes documented there.
    #[allow(clippy::too_many_arguments)]
    fn rash_d1_layers(
        &self,
        cum: &[i64],
        cum_n: &[i64],
        t_n: usize,
        overhead: f64,
        fine_floor3: f64,
        j_lo: usize,
        j_hi: usize,
        best: &mut f64,
        prev: &mut Vec<f64>,
        next: &mut Vec<f64>,
    ) {
        let leaf_reg = self.search.config.reg_alpha * self.search.config.reg;
        let end = t_n + 1;
        // Layer 1: one bin from the start to cut b.
        prev.clear();
        prev.resize(end, f64::INFINITY);
        for b in 1..=t_n {
            prev[b] = self.rash_d1_bin_cost(cum, cum_n, 0, b);
        }
        let j_max = self.search.config.max_k.min(t_n + 1).min(j_hi);
        for j in 2..=j_max {
            if j >= 3 && fine_floor3 >= *best + 1e-9 {
                break;
            }
            if j >= j_lo {
                // j bins: j - 1 bins ending at cut a, then the last bin.
                for a in (j - 1)..=t_n {
                    let pa = prev[a];
                    if !pa.is_finite() {
                        continue;
                    }
                    let v = overhead + (pa + self.rash_d1_bin_cost(cum, cum_n, a, end));
                    if v < *best {
                        *best = v;
                    }
                }
            }
            if j == j_max {
                break;
            }
            // Layer j only feeds partitions of >= 3 bins: skip building it
            // when none of them can beat `best`.
            if fine_floor3 >= *best + 1e-9 {
                break;
            }
            // Layer j: j bins ending at cut b. Every completion of `pa`
            // adds >= 2 bins (>= leaf_reg each, summed left to right).
            next.clear();
            next.resize(end, f64::INFINITY);
            for a in (j - 1)..t_n {
                let pa = prev[a];
                if !pa.is_finite() || overhead + ((pa + leaf_reg) + leaf_reg) >= *best {
                    continue;
                }
                for b in (a + 1)..=t_n {
                    let v = pa + self.rash_d1_bin_cost(cum, cum_n, a, b);
                    if v < next[b] {
                        next[b] = v;
                    }
                }
            }
            std::mem::swap(prev, next);
        }
    }

    /// The lb-solver's Select pool at `support` (cached per support in
    /// `rash_lb_select`): `rashomon_select_at(support, used, None)`, splits
    /// with <= 1 non-empty child dropped, sorted by `split_ub`; under
    /// Lb dedup one record per identical (feature, thresholds,
    /// group_map); under `fast` with each child's leaf cost and size.
    fn rash_lb_candidates(
        &mut self,
        support: &SupportMask,
        used: &FeatureMask,
        fast: bool,
    ) -> Arc<[RashLbSplit]> {
        if let Some(c) = self.rash_select_get(support) {
            return c;
        }
        let t_candgen = Instant::now();
        let mut c = self.rashomon_select_at(support, used, None);
        self.worker.profile.time_candidate_gen_lb += t_candgen.elapsed();
        self.worker.profile.n_lb_selects += 1;
        // The pass never records a split with <= 1 non-empty child.
        c.retain(|cand| cand.child_n.iter().filter(|&&n| n > 0).count() > 1);
        // Cheapest first (ordering only: tightens `best` early).
        c.sort_by(|a, b| {
            a.split_ub
                .partial_cmp(&b.split_ub)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        // Same identity => same children => same costs: keep the first.
        let mut seen: HashSet<(usize, Vec<u64>, Option<Vec<usize>>)> = HashSet::new();
        c.retain(|cand| {
            seen.insert((
                cand.feature_index,
                cand.thresholds.iter().map(|t| t.to_bits()).collect(),
                cand.group_map.clone(),
            ))
        });
        let n_classes = self.search.config.n_classes;
        let c: Arc<[RashLbSplit]> = c
            .into_iter()
            .map(|cand| {
                // Arity of the partition `rash_child_leaf_costs` builds.
                let n_bins = cand.thresholds.len() + 1;
                let k = cand
                    .group_map
                    .as_deref()
                    .map_or(n_bins, |g| g.iter().copied().max().unwrap_or(0) + 1);
                let (child_leaf, child_size): (Box<[f64]>, Box<[usize]>) = if !fast {
                    (Vec::new().into_boxed_slice(), Vec::new().into_boxed_slice())
                } else if cand.child_counts.len() == k && cand.child_n.len() == k {
                    (
                        cand.child_counts
                            .iter()
                            .map(|cc| {
                                let nw = cc[..n_classes.min(cc.len())].iter().sum::<i64>() as f64;
                                self.leaf_objective_from_counts(cc, nw)
                            })
                            .collect(),
                        cand.child_n.iter().map(|&n| n.max(0) as usize).collect(),
                    )
                } else {
                    // Arity mismatch: count the child supports directly.
                    let cs = self.search.store.partition_support(
                        support,
                        cand.feature_index,
                        &cand.thresholds,
                        cand.group_map.as_deref(),
                    );
                    let (lc, sz): (Vec<f64>, Vec<usize>) = cs
                        .iter()
                        .map(|s| {
                            let (cc, n) = self.search.store.count_classes(s);
                            (self.leaf_objective_from_counts(&cc, n as f64), n)
                        })
                        .unzip();
                    (lc.into_boxed_slice(), sz.into_boxed_slice())
                };
                RashLbSplit {
                    feature_index: cand.feature_index,
                    thresholds: cand.thresholds,
                    group_map: cand.group_map.map(|g| g.into_boxed_slice()),
                    k,
                    child_leaf,
                    child_size,
                }
            })
            .collect();
        self.worker
            .rash_lb_select
            .insert(support.clone(), Arc::clone(&c));
        self.worker.rash_lb_select_records += c.len() as u64;
        let p = &mut self.worker.profile;
        p.lb_select_cache_records = p
            .lb_select_cache_records
            .max(self.worker.rash_lb_select_records);
        c
    }

    /// The shared Select pool is used by this pass: the completion-lb DFS
    /// runs with its mask-based path (`rash_exact_lb_active`: no feature
    /// exclusion, no Learned criterion, so the pool ignores the feature
    /// mask and `similar_lb`; the gate's mask_exact), the depth-gated unsafe filter is
    /// off (the pool then ignores the depth) and the caches hold the
    /// current Select ub.
    fn rash_shared_on(&mut self) -> bool {
        self.rash_exact_lb_active()
            && self.search.config.unsafe_filter_depth_threshold <= 0
            && self.worker.rash_lb_ub_bits == Some(self.rashomon_select_ub().to_bits())
            && self.rash_lb_gate().0
    }

    /// Shared Select pool: the completion-lb DFS's Select at `support` (one
    /// `generate_candidates_for_support_rash` call at ub_hint
    /// `rashomon_select_ub`, depth `None`), timed and counted as an lb Select.
    fn rash_lb_select_timed(
        &mut self,
        support: &SupportMask,
        used: &FeatureMask,
    ) -> (Vec<SplitCandidate>, u64) {
        let t = Instant::now();
        let (cands, stats) = generate_candidates_for_support_rash(
            self.search.store.as_ref(),
            support,
            self.worker
                .rash_select_config
                .as_deref()
                .unwrap_or(&self.search.config),
            used,
            self.search.n_total,
            Some(self.rashomon_select_ub()),
            None,
            &self.worker.scratch_bank,
            None,
        );
        self.worker.profile.time_candidate_gen_lb += t.elapsed();
        self.worker.profile.n_lb_selects += 1;
        (cands, stats.n_available)
    }

    /// Shared Select pool: the compact pool of a Select output (see
    /// `RashSharedPool`), `None` when a split's child arrays do not match
    /// its partition arity / class count (the support is then not shared).
    ///
    /// Same views as the legacy consumers: the explore skips <= 1-non-empty
    /// splits and repeats of an identity it took (an exact repeat has the
    /// same arity, so the same Row-G verdict: never taken); the lb view is
    /// the legacy derivation (stable sort by `split_ub`, first per
    /// identity), where an exact repeat sorts after its original (same
    /// `split_ub`, later in pool order) and so is never kept.
    fn rash_shared_build(
        &self,
        cands: &[SplitCandidate],
        n_available: u64,
    ) -> Option<RashSharedPool> {
        let nc = self.search.config.n_classes;
        let mut k_hist: Vec<(usize, u32)> = Vec::new();
        for c in cands {
            match k_hist.iter_mut().find(|e| e.0 == c.k) {
                Some(e) => e.1 += 1,
                None => k_hist.push((c.k, 1)),
            }
        }
        let refs: Vec<&dyn ExploreCand> = cands.iter().map(|c| c as &dyn ExploreCand).collect();
        let (ident, n_ident) = explore_identities(&refs);
        // Kept pool entries (pool order): not an exact repeat.
        let mut kept: Vec<usize> = Vec::with_capacity(cands.len());
        let mut by_ident: HashMap<u32, Vec<usize>> = HashMap::new();
        for (i, c) in cands.iter().enumerate() {
            if ident[i] == u32::MAX {
                continue;
            }
            let bucket = by_ident.entry(ident[i]).or_default();
            let repeat = bucket.iter().any(|&j| {
                let e = &cands[j];
                e.k == c.k
                    && e.child_n == c.child_n
                    && e.child_counts == c.child_counts
                    && e.split_ub.to_bits() == c.split_ub.to_bits()
            });
            if !repeat {
                bucket.push(i);
                kept.push(i);
            }
        }
        // lb view order.
        let mut lb: Vec<usize> = kept.clone();
        lb.sort_by(|&a, &b| {
            cands[a]
                .split_ub
                .partial_cmp(&cands[b].split_ub)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut seen = vec![false; n_ident as usize];
        lb.retain(|&i| !std::mem::replace(&mut seen[ident[i] as usize], true));
        let n_lb = lb.len();
        let mut in_lb = vec![false; cands.len()];
        for &i in &lb {
            in_lb[i] = true;
        }
        let order: Vec<usize> = lb
            .iter()
            .copied()
            .chain(kept.iter().copied().filter(|&i| !in_lb[i]))
            .collect();
        let mut rec_of = vec![u32::MAX; cands.len()];
        for (ri, &i) in order.iter().enumerate() {
            rec_of[i] = ri as u32;
        }
        let explore: Vec<u32> = kept.iter().map(|&i| rec_of[i]).collect();
        // Every record's partition must match its counts (else no pool).
        for &i in &order {
            let c = &cands[i];
            let n_bins = c.thresholds.len() + 1;
            let k_part = c
                .group_map
                .as_deref()
                .map_or(n_bins, |g| g.iter().copied().max().unwrap_or(0) + 1);
            if c.child_n.len() != k_part
                || c.child_counts.len() != k_part
                || c.child_counts.iter().any(|cc| cc.len() != nc)
            {
                return None;
            }
        }
        let recs = match Self::rash_pool_arena(cands, &order, &ident, nc) {
            Some(a) => RashPoolStore::Arena(Arc::new(a)),
            None => {
                RashPoolStore::Legacy(Arc::from(self.rash_pool_legacy(cands, &order, &ident, nc)))
            }
        };
        Some(RashSharedPool {
            recs,
            n_lb,
            explore: Arc::from(explore),
            n_ident,
            k_hist: Arc::from(k_hist),
            n_available,
        })
    }

    /// The legacy pool records (`RashPoolRec`) of `rash_shared_build`, in
    /// `order` (partitions already checked).
    fn rash_pool_legacy(
        &self,
        cands: &[SplitCandidate],
        order: &[usize],
        ident: &[u32],
        nc: usize,
    ) -> Vec<RashPoolRec> {
        let mut recs: Vec<RashPoolRec> = Vec::with_capacity(order.len());
        for &i in order {
            let c = &cands[i];
            let k_part = c.child_n.len();
            let mut data: Vec<i64> = Vec::with_capacity(k_part * (nc + 2));
            for (cc, &n) in c.child_counts.iter().zip(&c.child_n) {
                let nw = cc[..nc].iter().sum::<i64>() as f64;
                data.push(self.leaf_objective_from_counts(cc, nw).to_bits() as i64);
                data.push(n);
                data.extend_from_slice(cc);
            }
            recs.push(RashPoolRec {
                feature_index: c.feature_index as u32,
                sel_k: c.k as u32,
                ident: ident[i],
                thresholds: Arc::clone(&c.thresholds),
                group_map: c.group_map.clone().map(|g| g.into_boxed_slice()),
                data_children: k_part as u32,
                data: data.into_boxed_slice(),
            });
        }
        recs
    }

    /// Pool arena: the records of `rash_shared_build` in `order` as a
    /// compact arena (partitions already checked), or None when a field
    /// does not fit (more than MAX_CLASSES classes, a count outside u32, a
    /// feature above u16, an arity above u8): the legacy layout then.
    fn rash_pool_arena(
        cands: &[SplitCandidate],
        order: &[usize],
        ident: &[u32],
        nc: usize,
    ) -> Option<RashPoolArena> {
        if nc > MAX_CLASSES || nc > u8::MAX as usize {
            return None;
        }
        let mut hdr: Vec<RashPoolHdr> = Vec::with_capacity(order.len());
        let mut cells: Vec<u32> = Vec::new();
        let mut thr: Vec<Arc<[f64]>> = Vec::new();
        let mut thr_of: RashFxMap<(usize, usize), u32> = RashFxMap::default();
        let mut gm: Vec<Box<[usize]>> = Vec::new();
        let mut gm_of: HashMap<Vec<usize>, u32> = HashMap::new();
        let fit = |v: i64| -> Option<u32> { u32::try_from(v).ok() };
        for &i in order {
            let c = &cands[i];
            let k_part = c.child_n.len();
            if c.feature_index > u16::MAX as usize
                || c.k > u8::MAX as usize
                || k_part > u8::MAX as usize
            {
                return None;
            }
            let off = u32::try_from(cells.len()).ok()?;
            for (cc, &n) in c.child_counts.iter().zip(&c.child_n) {
                cells.push(fit(n)?);
                for &x in &cc[..nc] {
                    cells.push(fit(x)?);
                }
            }
            // The very Arcs Select returned (pointer identity), interned.
            let key = (
                Arc::as_ptr(&c.thresholds) as *const f64 as usize,
                c.thresholds.len(),
            );
            let t = match thr_of.get(&key) {
                Some(&t) => t,
                None => {
                    let t = u32::try_from(thr.len()).ok()?;
                    thr.push(Arc::clone(&c.thresholds));
                    thr_of.insert(key, t);
                    t
                }
            };
            let g = match c.group_map.as_ref() {
                None => u32::MAX,
                Some(m) => match gm_of.get(m) {
                    Some(&g) => g,
                    None => {
                        let g = u32::try_from(gm.len()).ok().filter(|&g| g != u32::MAX)?;
                        gm.push(m.clone().into_boxed_slice());
                        gm_of.insert(m.clone(), g);
                        g
                    }
                },
            };
            hdr.push(RashPoolHdr {
                ident: ident[i],
                cells: off,
                thr: t,
                gm: g,
                feature: c.feature_index as u16,
                sel_k: c.k as u8,
                n_children: k_part as u8,
            });
        }
        Some(RashPoolArena {
            hdr: hdr.into_boxed_slice(),
            cells: cells.into_boxed_slice(),
            thr: thr.into_boxed_slice(),
            gm: gm.into_boxed_slice(),
            nc: nc as u8,
        })
    }

    /// The completion-lb DFS's pool at `support`: the shared pool's lb view
    /// (shared Select pool), else the legacy cache (`rash_lb_candidates`).
    fn rash_lb_pool(&mut self, support: &SupportMask, used: &FeatureMask, fast: bool) -> LbPool {
        if self.rash_shared_on() {
            if let Some(p) = self.rash_shared_get(support) {
                self.worker.profile.n_select_shared_hits += 1;
                return LbPool::Shared(p);
            }
            if !self.rash_select_has(support) {
                let (cands, n_avail) = self.rash_lb_select_timed(support, used);
                match self.rash_shared_build(&cands, n_avail) {
                    Some(p) => {
                        self.rash_shared_insert(support, p.clone());
                        return LbPool::Shared(p);
                    }
                    None => {
                        let v = self.rash_lb_view_legacy(support, cands, fast);
                        self.rash_lb_select_insert(support, Arc::clone(&v));
                        return LbPool::Legacy(v);
                    }
                }
            }
        }
        LbPool::Legacy(self.rash_lb_candidates(support, used, fast))
    }

    /// Store a shared pool (its lb records count as lb cache records).
    fn rash_shared_insert(&mut self, support: &SupportMask, p: RashSharedPool) {
        self.worker.rash_lb_select_records += p.n_lb as u64;
        let pr = &mut self.worker.profile;
        pr.lb_select_cache_records = pr
            .lb_select_cache_records
            .max(self.worker.rash_lb_select_records);
        self.worker.rash_select_shared.insert(support.clone(), p);
    }

    /// Store a legacy lb list in the lb Select cache.
    fn rash_lb_select_insert(&mut self, support: &SupportMask, c: Arc<[RashLbSplit]>) {
        self.worker.rash_lb_select_records += c.len() as u64;
        self.worker.rash_lb_select.insert(support.clone(), c);
        let p = &mut self.worker.profile;
        p.lb_select_cache_records = p
            .lb_select_cache_records
            .max(self.worker.rash_lb_select_records);
    }

    /// `rash_lb_candidates`'s derivation of an lb list from a Select output.
    fn rash_lb_view_legacy(
        &self,
        support: &SupportMask,
        mut c: Vec<SplitCandidate>,
        fast: bool,
    ) -> Arc<[RashLbSplit]> {
        c.retain(|cand| cand.child_n.iter().filter(|&&n| n > 0).count() > 1);
        c.sort_by(|a, b| {
            a.split_ub
                .partial_cmp(&b.split_ub)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut seen: HashSet<(usize, Vec<u64>, Option<Vec<usize>>)> = HashSet::new();
        c.retain(|cand| {
            seen.insert((
                cand.feature_index,
                cand.thresholds.iter().map(|t| t.to_bits()).collect(),
                cand.group_map.clone(),
            ))
        });
        let n_classes = self.search.config.n_classes;
        c.into_iter()
            .map(|cand| {
                let n_bins = cand.thresholds.len() + 1;
                let k = cand
                    .group_map
                    .as_deref()
                    .map_or(n_bins, |g| g.iter().copied().max().unwrap_or(0) + 1);
                let (child_leaf, child_size): (Box<[f64]>, Box<[usize]>) = if !fast {
                    (Vec::new().into_boxed_slice(), Vec::new().into_boxed_slice())
                } else if cand.child_counts.len() == k && cand.child_n.len() == k {
                    (
                        cand.child_counts
                            .iter()
                            .map(|cc| {
                                let nw = cc[..n_classes.min(cc.len())].iter().sum::<i64>() as f64;
                                self.leaf_objective_from_counts(cc, nw)
                            })
                            .collect(),
                        cand.child_n.iter().map(|&n| n.max(0) as usize).collect(),
                    )
                } else {
                    let cs = self.search.store.partition_support(
                        support,
                        cand.feature_index,
                        &cand.thresholds,
                        cand.group_map.as_deref(),
                    );
                    let (lc, sz): (Vec<f64>, Vec<usize>) = cs
                        .iter()
                        .map(|s| {
                            let (cc, n) = self.search.store.count_classes(s);
                            (self.leaf_objective_from_counts(&cc, n as f64), n)
                        })
                        .unzip();
                    (lc.into_boxed_slice(), sz.into_boxed_slice())
                };
                RashLbSplit {
                    feature_index: cand.feature_index,
                    thresholds: cand.thresholds,
                    group_map: cand.group_map.map(|g| g.into_boxed_slice()),
                    k,
                    child_leaf,
                    child_size,
                }
            })
            .collect()
    }

    /// Child supports of a pool split at `support` for the completion-lb
    /// DFS's fast path (mask_exact): le-mask AND / ANDNOT for any store
    /// thresholds, else `partition_support` (whose
    /// non-consecutive thresholds take the row walk). Same supports.
    fn rash_mask_children(
        &self,
        support: &SupportMask,
        f: usize,
        ths: &[f64],
        gm: Option<&[usize]>,
    ) -> Vec<SupportMask> {
        if let Some(cs) = self.search.store.partition_support_le(support, f, ths, gm) {
            return cs;
        }
        self.search.store.partition_support(support, f, ths, gm)
    }

    /// `bounded_completion_min` at an absolute bound: `None` iff every
    /// completion costs more than `bound`.
    ///
    /// Speed paths (same values, so the same tree set): the depth-1 closed
    /// form answers remaining depth 1 without Select
    /// (`rash_depth1_min_closed`, memoised as `Min`); the mask-based path
    /// prices the node and its children from popcounts and the cached
    /// Select counts and builds child supports by le-mask ANDs (needs
    /// `rash_lb_gate` mask_exact); duplicate pool records are dropped.
    ///
    /// `sweep` (depth-1 sweep, from `rash_lb_scan` at remaining depth 2):
    /// this node is the parent's child of f-rank interval [a, b); its
    /// closed-form counts may come from the parent's sweep (same value).
    fn rash_completion_min(
        &mut self,
        support: &SupportMask,
        used: &FeatureMask,
        depth: i32,
        bound: f64,
        sweep: Option<(&mut RashD1Sweep, usize, usize, usize)>,
        skey: Option<RashD1Key>,
    ) -> Option<f64> {
        // Structural depth-1 memo: `skey` (remaining depth 1 only) is the support's
        // structural key; its entries are the same facts as the full-key
        // memo's, stored without the support words.
        let skey = skey.filter(|_| depth == 1);
        if let Some(sk) = skey {
            let hit = match self.rash_d1s_get(&sk) {
                Some(RashLbEntry::Min(m)) => Some((m <= bound).then_some(m)),
                Some(RashLbEntry::Above(b)) if bound <= b => Some(None),
                _ => None,
            };
            if let Some(r) = hit {
                self.worker.profile.n_lb_d1_memo_hits += 1;
                return r;
            }
        }
        let mkey = (support.clone(), depth);
        match self.rash_memo_get(&mkey) {
            Some(RashLbEntry::Min(m)) => {
                if depth == 1 {
                    self.worker.profile.n_lb_d1_memo_hits += 1;
                }
                return (m <= bound).then_some(m);
            }
            Some(RashLbEntry::Above(b)) if bound <= b => {
                if depth == 1 {
                    self.worker.profile.n_lb_d1_memo_hits += 1;
                }
                return None;
            }
            _ => {}
        }
        let (split_reg, leaf_reg, path_penalty, n_total) = {
            let c = &self.search.config;
            (
                (1.0 - c.reg_alpha) * c.reg,
                c.reg_alpha * c.reg,
                c.path_penalty,
                self.search.n_total as f64,
            )
        };
        let gate = self.rash_lb_gate();
        let fast = gate.0;
        // Row indices: the legacy path's input (lazily under `fast`).
        let mut indices: Option<Vec<u32>> = None;
        let (n_node, leaf) = if fast {
            let (counts, n) = self.search.store.count_classes(support);
            let leaf = self.leaf_objective_from_counts(&counts, n as f64);
            (n, leaf)
        } else {
            let idx = self.search.store.support_to_indices(support);
            let n = idx.len();
            let leaf = self.rash_leaf_objective(&idx);
            indices = Some(idx);
            (n, leaf)
        };
        let overhead = split_reg + path_penalty * n_node as f64 / n_total;
        let split_floor = overhead + 2.0 * leaf_reg;
        // No split can beat the leaf: the leaf is the exact minimum.
        if depth <= 0 || n_node <= 1 || split_floor >= leaf {
            self.rash_lb_memo_put_any(mkey, skey, RashLbEntry::Min(leaf));
            return (leaf <= bound).then_some(leaf);
        }
        if leaf > bound && split_floor > bound {
            self.rash_lb_memo_put_any(mkey, skey, RashLbEntry::Above(bound));
            return None;
        }
        if self.check_time() {
            return Some(leaf.min(split_floor));
        }
        // Remaining depth 1: the closed-form minimum (<= the pool's).
        if depth == 1 && gate.1 {
            let (rowwalk, twopass) = (None, true);
            let t_closed = Instant::now();
            // Drop DP-path counts of calls outside this one (tests, verify).
            rash_d1_intdp_counts_take();
            let swept = sweep.and_then(|(sw, f, a, b)| {
                self.rash_d1_sweep_child(sw, f, a, b, n_node, overhead, twopass)
            });
            let closed = match swept {
                Some(v) => Some((v, false)),
                None => {
                    self.rash_depth1_min_closed_ext(support, n_node, overhead, twopass, rowwalk)
                }
            };
            self.worker.profile.time_lb_d1_closed += t_closed.elapsed();
            let (n_int, n_int_fallback) = rash_d1_intdp_counts_take();
            self.worker.profile.n_lb_d1_intdp += n_int;
            self.worker.profile.n_lb_d1_intdp_fallback += n_int_fallback;
            match closed {
                Some((closed, walked)) => {
                    let v = leaf.min(closed);
                    self.worker.profile.n_lb_d1_closed += 1;
                    if walked {
                        self.worker.profile.n_lb_d1_rowwalk += 1;
                    }
                    if swept.is_some() {
                        self.worker.profile.n_lb_d1_swept += 1;
                    }
                    // `v` is this node's exact value for every bound.
                    self.rash_lb_memo_put_any(mkey, skey, RashLbEntry::Min(v));
                    return (v <= bound).then_some(v);
                }
                None => self.worker.profile.n_lb_d1_fallback += 1,
            }
        }
        // Interval lb: remaining depth 2 from the interval DP over
        // the support's store bins (<= the pool's value; no Select).
        if depth == 2 && gate.1 && self.search.config.rash_tuning.lb_interval {
            if let Some(best) = self.rash_interval_min2(support, leaf, overhead, bound) {
                if self.rash_timed_out() {
                    return Some(best.min(split_floor));
                }
                return if best <= bound {
                    self.rash_lb_memo_put_any(mkey, skey, RashLbEntry::Min(best));
                    Some(best)
                } else {
                    self.rash_lb_memo_put_any(mkey, skey, RashLbEntry::Above(bound));
                    None
                };
            }
        }
        // Select is depth- and bound-free here: one call per support.
        let best = match self.rash_lb_pool(support, used, fast) {
            LbPool::Legacy(c) => self.rash_lb_scan(
                &c[..],
                support,
                used,
                depth,
                bound,
                leaf,
                overhead,
                fast,
                &mut indices,
            ),
            LbPool::Shared(p) => match &p.recs {
                RashPoolStore::Legacy(r) => self.rash_lb_scan(
                    &r[..p.n_lb],
                    support,
                    used,
                    depth,
                    bound,
                    leaf,
                    overhead,
                    fast,
                    &mut indices,
                ),
                RashPoolStore::Arena(a) => {
                    let refs: Vec<RashPoolRecRef> = (0..p.n_lb).map(|i| a.rec(i)).collect();
                    self.rash_lb_scan(
                        &refs,
                        support,
                        used,
                        depth,
                        bound,
                        leaf,
                        overhead,
                        fast,
                        &mut indices,
                    )
                }
            },
        };
        if self.rash_timed_out() {
            return Some(best.min(split_floor));
        }
        if best <= bound {
            self.rash_lb_memo_put_any(mkey, skey, RashLbEntry::Min(best));
            Some(best)
        } else {
            self.rash_lb_memo_put_any(mkey, skey, RashLbEntry::Above(bound));
            None
        }
    }

    /// The candidate loop of `rash_completion_min` over the pool `candidates`
    /// (legacy lb records or a shared pool's lb view): the minimum of the
    /// leaf (`leaf`) and every feasible split's cost below `bound`.
    #[allow(clippy::too_many_arguments)]
    fn rash_lb_scan<R: RashLbRec>(
        &mut self,
        candidates: &[R],
        support: &SupportMask,
        used: &FeatureMask,
        depth: i32,
        bound: f64,
        leaf: f64,
        overhead: f64,
        fast: bool,
        indices: &mut Option<Vec<u32>>,
    ) -> f64 {
        let (split_reg, leaf_reg, path_penalty, n_total) = {
            let c = &self.search.config;
            (
                (1.0 - c.reg_alpha) * c.reg,
                c.reg_alpha * c.reg,
                c.path_penalty,
                self.search.n_total as f64,
            )
        };
        let mut lc_buf: smallvec::SmallVec<[f64; 8]> = smallvec::SmallVec::new();
        let mut sz_buf: smallvec::SmallVec<[usize; 8]> = smallvec::SmallVec::new();
        // The store, held apart from `self` (lazy le-mask child supports
        // borrow it across the recursive calls).
        let store = Arc::clone(&self.search.store);
        let (mut sweep_state, mut sweep_tried): (Option<RashD1Sweep>, bool) = (None, false);
        // Structural depth-1 memo: structural keys for the remaining-depth-1 children
        // (mask_exact child supports; `rash_d1_child_keys`).
        let opta_memo = depth == 2 && fast;
        // Per-candidate buffers, cleared per candidate.
        let mut vals: smallvec::SmallVec<[f64; 8]> = smallvec::SmallVec::new();
        let mut exact: smallvec::SmallVec<[bool; 8]> = smallvec::SmallVec::new();
        let mut order: smallvec::SmallVec<[usize; 8]> = smallvec::SmallVec::new();
        // Similar-support skip (depth 3, children at remaining depth 2): v2(c) =
        // the interval minimum at this support (no bound), computed before
        // the first child call; None = rule off for this scan (deadline,
        // gate, path not applicable). A
        // child g ⊆ c has value >= v2(c) − |c∖g|·(1 + 2·path_penalty) /
        // n_total, so a budget below that is answered None ("> budget")
        // without the call. (With a rash_par prefetch, skip before it.)
        // Only with `lb_interval` on: pool-only runs keep their path.
        let ss_on = depth == 3 && self.search.config.rash_tuning.lb_interval;
        let mut ss_v2: Option<Option<f64>> = None;
        let ss_coef = (1.0 + 2.0 * path_penalty) / n_total;
        let t_d3 = (depth == 3).then(Instant::now);
        let mut best = leaf;
        for cand in candidates.iter() {
            let limit = best.min(bound);
            if overhead + cand.k() as f64 * leaf_reg > limit {
                continue;
            }
            // Per child: (value, exact). The leaf cost from the cached
            // Select counts (fast) or the row partition (no allocation); a
            // trivial admissible lb until solved.
            lc_buf.clear();
            sz_buf.clear();
            let cached = fast
                && cand.cached_children(
                    &|cc: &[i64], nw: f64| self.leaf_objective_from_counts(cc, nw),
                    &mut lc_buf,
                    &mut sz_buf,
                );
            if !cached && indices.is_none() {
                *indices = Some(self.search.store.support_to_indices(support));
            }
            if !cached {
                let (a, b) = self.rash_child_leaf_costs(
                    indices.as_ref().unwrap(),
                    cand.feature_index(),
                    cand.thresholds(),
                    cand.group_map(),
                );
                lc_buf.extend(a);
                sz_buf.extend(b);
            }
            let (leaf_costs, sizes): (&[f64], &[usize]) = (&lc_buf, &sz_buf);
            let k = leaf_costs.len();
            vals.clear();
            exact.clear();
            for b in 0..k {
                let leaf_c = leaf_costs[b];
                let floor_c = split_reg + 2.0 * leaf_reg + path_penalty * sizes[b] as f64 / n_total;
                if depth - 1 <= 0 || sizes[b] <= 1 || floor_c >= leaf_c {
                    vals.push(leaf_c);
                    exact.push(true);
                } else {
                    vals.push(floor_c);
                    exact.push(false);
                }
            }
            if overhead + vals.iter().sum::<f64>() > limit {
                continue;
            }
            if exact.iter().all(|&e| e) {
                let cost = overhead + vals.iter().sum::<f64>();
                if cost < best {
                    best = cost;
                }
                continue;
            }
            // Child supports: le-mask ANDs (fast) or the row partition
            // (built per solved child). Lazy le-mask children (no
            // group_map): each child's support
            // is built only when the loop below recurses into it.
            let lazy_le: Option<smallvec::SmallVec<[&[u64]; 8]>> =
                if fast && cand.group_map().is_none() {
                    store.le_masks_for(cand.feature_index(), cand.thresholds())
                } else {
                    None
                };
            let fast_children: Option<Vec<SupportMask>> = if lazy_le.is_some() {
                None
            } else if fast {
                Some(self.rash_mask_children(
                    support,
                    cand.feature_index(),
                    cand.thresholds(),
                    cand.group_map(),
                ))
            } else {
                None
            };
            let bins = if fast {
                None
            } else {
                Some(self.partition_into_flat(
                    indices.as_ref().unwrap(),
                    self.search.store.feature_col(cand.feature_index()),
                    cand.thresholds(),
                    cand.group_map(),
                ))
            };
            let mut cuts: Option<Option<smallvec::SmallVec<[usize; 8]>>> = None;
            let mut skeys: Option<smallvec::SmallVec<[RashD1Key; 8]>> = None;
            order.clear();
            order.extend((0..k).filter(|&b| !exact[b]));
            order.sort_by(|&a, &b| {
                vals[a]
                    .partial_cmp(&vals[b])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            let mut feasible = true;
            for &b in &order {
                let others: f64 = vals
                    .iter()
                    .enumerate()
                    .filter(|&(j, _)| j != b)
                    .map(|(_, v)| v)
                    .sum();
                // Depth-1 sweep: children at remaining depth 1 of a split
                // on store thresholds take their counts from the parent's
                // sweep (state built lazily, one per scan).
                if depth == 2 && cuts.is_none() {
                    if sweep_state.is_none() && !sweep_tried {
                        sweep_tried = true;
                        sweep_state = self.rash_d1_sweep_new(support);
                    }
                    cuts = Some(if sweep_state.is_some() && cand.group_map().is_none() {
                        self.rash_d1_sweep_cuts(cand.feature_index(), cand.thresholds())
                    } else {
                        None
                    });
                    if opta_memo && cand.group_map().is_none() {
                        skeys = self.rash_d1_child_keys(
                            support,
                            cand.feature_index(),
                            cand.thresholds(),
                            k,
                        );
                    }
                }
                let skey = skeys.as_ref().map(|v| v[b]);
                let child_bound = limit - overhead - others;
                // Structural depth-1 memo: a structural hit answers before the child
                // support is built.
                let hit = match skey {
                    Some(sk) => match self.rash_d1s_get(&sk) {
                        Some(RashLbEntry::Min(m)) => Some((m <= child_bound).then_some(m)),
                        Some(RashLbEntry::Above(bb)) if child_bound <= bb => Some(None),
                        _ => None,
                    },
                    _ => None,
                };
                // Similar-support skip: the similar-support rule (above).
                let ss_pred = ss_on && {
                    if ss_v2.is_none() {
                        ss_v2 = Some(self.rash_simskip_ref(support, leaf, overhead));
                    }
                    ss_v2.flatten().is_some_and(|v2| {
                        v2 - (support.n_samples - sizes[b]) as f64 * ss_coef > child_bound + 1e-9
                    })
                };
                if ss_pred {
                    interval_lb::SS_SKIPS.fetch_add(1, Ordering::Relaxed);
                }
                let r = match hit {
                    _ if ss_pred => None,
                    Some(r) => {
                        self.worker.profile.n_lb_d1_memo_hits += 1;
                        r
                    }
                    None => {
                        let child_support = match (&lazy_le, &fast_children) {
                            (Some(le), _) => DatasetStore::le_bin_support(support, le, b),
                            (None, Some(cs)) => cs[b].clone(),
                            (None, None) => self
                                .search
                                .store
                                .support_from_indices(bins.as_ref().unwrap().bin_slice(b)),
                        };
                        let hint =
                            match (sweep_state.as_mut(), cuts.as_ref().and_then(|c| c.as_ref())) {
                                (Some(sw), Some(c)) if c.len() == k + 1 => {
                                    Some((sw, cand.feature_index(), c[b], c[b + 1]))
                                }
                                _ => None,
                            };
                        self.rash_completion_min(
                            &child_support,
                            used,
                            depth - 1,
                            child_bound,
                            hint,
                            skey,
                        )
                    }
                };
                match r {
                    Some(m) => {
                        if m > vals[b] {
                            vals[b] = m;
                        }
                    }
                    None => {
                        feasible = false;
                        break;
                    }
                }
                if overhead + vals.iter().sum::<f64>() > limit {
                    feasible = false;
                    break;
                }
            }
            if !feasible {
                continue;
            }
            // The lb-sum test's sum: split_reg + path_cost + Σ children (bin order).
            let cost = overhead + vals.iter().sum::<f64>();
            if cost < best {
                best = cost;
            }
        }
        if let Some(t) = t_d3 {
            interval_lb::SCAN_D3.fetch_add(1, Ordering::Relaxed);
            interval_lb::SCAN_D3_NS.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
        }
        best
    }

    /// Similar-support skip: v2(c) at `support` (the interval minimum, no
    /// bound), or None when it is unavailable: the deadline (also when it
    /// passes during the computation), the interval gate off, or the path
    /// not applicable.
    fn rash_simskip_ref(&mut self, support: &SupportMask, leaf: f64, overhead: f64) -> Option<f64> {
        if self.rash_timed_out() || !self.rash_lb_gate().1 {
            return None;
        }
        let t0 = Instant::now();
        let v = self.rash_interval_min2(support, leaf, overhead, f64::INFINITY);
        interval_lb::SS_NS_REF.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
        if self.rash_timed_out() {
            return None;
        }
        if v.is_some() {
            interval_lb::SS_REFS.fetch_add(1, Ordering::Relaxed);
        }
        v
    }

    /// Select_B(u) of the Step-A pass at `support`: the post-dedup
    /// candidate list `rash_explore_node` generates there (all features,
    /// ub_hint = `rashomon_select_ub`, no killer move). Debug / oracle use
    /// (`rashomon_select_debug`); the arity / scope prunes applied after
    /// generation are admissible and not part of the pool.
    pub fn rashomon_select(&mut self, support: &SupportMask) -> Vec<SplitCandidate> {
        let used_feats = self.search.config.initial_used_features.clone();
        self.rashomon_select_at(support, &used_feats, None)
    }

    /// `rashomon_select` at a node's feature mask and remaining depth `depth`
    /// (the arguments the Step-A pass gives Select): the up-front
    /// anchor's completion (`rashomon_anchor::complete_prefix_with_select`).
    pub(crate) fn rashomon_select_depth(
        &mut self,
        support: &SupportMask,
        used_feats: &FeatureMask,
        depth: i32,
    ) -> Vec<SplitCandidate> {
        self.rashomon_select_at(support, used_feats, Some(depth))
    }

    /// `rashomon_select` at a node: its feature mask and remaining depth,
    /// the same arguments the Step-A pass gives Select there.
    fn rashomon_select_at(
        &mut self,
        support: &SupportMask,
        used_feats: &FeatureMask,
        depth: Option<i32>,
    ) -> Vec<SplitCandidate> {
        let slb_ref = if self.search.config.dp_criterion == DpCriterion::Learned {
            Some(&self.worker.similar_lb)
        } else {
            None
        };
        let (candidates, _stats) = generate_candidates_for_support_rash(
            self.search.store.as_ref(),
            support,
            self.worker
                .rash_select_config
                .as_deref()
                .unwrap_or(&self.search.config),
            used_feats,
            self.search.n_total,
            Some(self.rashomon_select_ub()),
            slb_ref,
            &self.worker.scratch_bank,
            depth,
        );
        candidates
    }

    /// Exact cost lattice of this solve's DAG costs (integer cost keys): the
    /// leaf-only objective `err / n_total + leaf_reg * leaves`
    /// needs split_reg = 0 and path_penalty = 0; otherwise None (rounded
    /// keys). The builder still checks every leaf objective lies on it.
    fn cost_lattice(&self) -> Option<crate::types::CostLattice> {
        let c = &self.search.config;
        let split_reg = (1.0 - c.reg_alpha) * c.reg;
        if split_reg != 0.0 || c.path_penalty != 0.0 {
            return None;
        }
        Some(crate::types::CostLattice {
            n_total: self.search.n_total,
            leaf_reg: c.reg_alpha * c.reg,
        })
    }

    fn branch_bound_solve(&mut self) {
        let root_support = self.search.root_support.clone();
        let initial_feats = self.search.config.initial_used_features.clone();
        *self.search.root_key.lock().unwrap() = MemoKey::new(
            root_support.clone(),
            initial_feats.clone(),
            self.search.config.max_depth,
            Self::mode_flags(&self.search.config),
            Self::teacher_alpha_memo_bits(&self.search.config),
        );
        let root_key = self.search.root_key.lock().unwrap().clone();

        let _ = self.get_or_create(&root_support, &initial_feats, self.search.config.max_depth);

        let root_ub = self.search.memo.get(&root_key).unwrap().upperbound;
        let initial_scope = match self.search.config.warm_start_ub {
            Some(ws) if ws < root_ub => ws + 1e-9,
            _ => root_ub,
        };

        self.branch_bound_recurse(
            &root_key,
            self.search.config.max_depth,
            initial_scope,
            &initial_feats,
        );
    }

    fn branch_bound_recurse(
        &mut self,
        key: &MemoKey,
        depth: i32,
        scope: f64,
        used_feats: &FeatureMask,
    ) -> BranchBoundResult {
        // Stage 0: Early exits
        let (
            current_lb,
            mut node_upperbound,
            mut node_upperbound_is_heuristic,
            support,
            sp_base_obj,
            parent_path_branch_bound,
            sig,
        ) = {
            let sp = self.search.memo.get(key).unwrap();
            if sp.is_solved || sp.is_terminal {
                return BranchBoundResult::Feasible {
                    cost: sp.upperbound,
                    heuristic: sp.upperbound_is_heuristic,
                };
            }
            if depth == 1 && sp.explored && sp.upperbound_is_heuristic {
                return BranchBoundResult::Feasible {
                    cost: sp.upperbound,
                    heuristic: true,
                };
            }
            if sp.lowerbound > scope + 1e-12 {
                let floor = sp.lowerbound;
                drop(sp);
                self.record_branch_bound_scope_floor(key, scope);
                return BranchBoundResult::Infeasible { floor };
            }
            if sp.branch_bound_scope_floor.is_finite()
                && sp.branch_bound_scope_floor >= scope - 1e-12
            {
                return BranchBoundResult::Infeasible {
                    floor: sp.branch_bound_scope_floor,
                };
            }
            (
                sp.lowerbound,
                sp.upperbound,
                sp.upperbound_is_heuristic,
                sp.support.clone(),
                sp.base_objective,
                sp.split_path.clone(),
                SupportSig::new(sp.n_samples, &sp.class_counts[..sp.n_classes as usize]),
            )
        };
        if self.check_time() {
            return BranchBoundResult::Feasible {
                cost: node_upperbound,
                heuristic: node_upperbound_is_heuristic,
            };
        }

        // SPLIT closure frontier for branch-bound. `closure_depth` is expressed
        // in remaining-depth units: with max depth D and lookahead d_l,
        // closure starts when `depth == D - d_l`, i.e. at tree depth d_l.
        // The get_or_create paths already enforce this for BFS/materialized
        // children; branch-bound recurses directly, so it must honor the same frontier
        // here before generating exact-search candidates.
        if let Some(closure_depth) = self.search.config.closure_depth {
            if depth <= closure_depth && depth > 0 {
                let is_entry = depth == closure_depth;
                let t_cc = if is_entry { Some(Instant::now()) } else { None };
                let (_closure_key, obj) = self.closure_build(&support, used_feats, depth);
                if let Some(t) = t_cc {
                    self.worker.profile.time_closure += t.elapsed();
                }
                self.search.used_closure.store(true, Ordering::Relaxed);
                return BranchBoundResult::Feasible {
                    cost: obj,
                    heuristic: false,
                };
            }
        }

        self.search.n_explored.fetch_add(1, Ordering::Relaxed);
        let d_idx = (depth as usize).min(MAX_PROFILE_DEPTH - 1);
        self.worker.profile.nodes_at_depth[d_idx] += 1;

        let indices = self.search.store.support_to_indices(&support);
        let n_node = indices.len();
        let path_cost =
            self.search.config.path_penalty * n_node as f64 / self.search.n_total as f64;

        // Purification bound: even a perfect split (zero misclass) costs
        // split_reg + K_min * leaf_reg + path_cost. If that already exceeds scope,
        // no split can help — this node is optimally a leaf.
        {
            let split_reg = (1.0 - self.search.config.reg_alpha) * self.search.config.reg;
            let leaf_reg = self.search.config.reg_alpha * self.search.config.reg;
            let perfect_split_cost = split_reg + 2.0 * leaf_reg + path_cost;
            if perfect_split_cost > scope + 1e-12 {
                if perfect_split_cost >= sp_base_obj - 1e-12 {
                    let (base_obj, leaf_sig) = {
                        let mut sp = self.search.memo.get_mut(key).unwrap();
                        sp.is_terminal = true;
                        sp.is_solved = true;
                        sp.lowerbound = sp.base_objective;
                        sp.upperbound = sp.base_objective;
                        sp.upperbound_is_heuristic = false;
                        (
                            sp.base_objective,
                            SupportSig::new(
                                sp.n_samples,
                                &sp.class_counts[..sp.n_classes as usize],
                            ),
                        )
                    };
                    self.similar_lb_update(
                        leaf_sig.n_samples,
                        leaf_sig.as_slice(),
                        depth,
                        base_obj,
                    );
                    self.similar_ub_update(leaf_sig.n_samples, leaf_sig.as_slice(), base_obj);
                    return BranchBoundResult::Feasible {
                        cost: base_obj,
                        heuristic: false,
                    };
                }
                self.record_branch_bound_scope_floor(key, scope);
                return BranchBoundResult::Infeasible { floor: scope };
            }
        }

        // Stage 1: Killer move warm-start
        let mut ub_hint = node_upperbound.min(scope);

        if let Some((killer_feat, _killer_k, killer_thresh)) =
            self.worker.killer_moves.get(&depth).cloned()
        {
            let feat_available =
                !self.search.config.feature_exclusion || used_feats.contains(killer_feat);

            if feat_available {
                let bins = self.partition_into_flat(
                    &indices,
                    self.search.store.feature_col(killer_feat),
                    &killer_thresh,
                    None,
                );

                let non_empty = (0..bins.len()).filter(|&i| bins.bin_len(i) > 0).count();
                if non_empty > 1 {
                    let split_reg = (1.0 - self.search.config.reg_alpha) * self.search.config.reg;
                    let mut killer_ub = split_reg + path_cost;
                    for bin_idx in 0..bins.len() {
                        let (_lb, ub) =
                            self.compute_bounds_inline(bins.bin_slice(bin_idx), depth - 1);
                        killer_ub += ub;
                    }

                    // With a feasibility gate:
                    // killer_ub is `split_reg + path + Σ base_obj_of_children` =
                    // the cost of "killer split with every child as a leaf."
                    // This is only achievable if EVERY bin has
                    // ≥ min_samples_leaf samples; otherwise no real tree
                    // realizes killer_ub and using it as a UB to feed the
                    // depth==1 admissible filter would prune valid features.
                    //
                    // Feasibility check: verify all bins ≥ min_samples_leaf.
                    // Iff feasible, killer_ub is a true achievable UB and
                    // tightening ub_hint is sound.
                    let min_leaf = self.search.config.min_samples_leaf;
                    let killer_feasible =
                        (0..bins.len()).all(|i| bins.bin_slice(i).len() >= min_leaf);
                    if killer_feasible && killer_ub < ub_hint {
                        ub_hint = killer_ub;
                        if killer_ub < node_upperbound {
                            node_upperbound = killer_ub;
                            node_upperbound_is_heuristic = false;
                            let mut sp = self.search.memo.get_mut(key).unwrap();
                            sp.upperbound = killer_ub;
                            sp.upperbound_is_heuristic = false;
                        }
                    }
                }
            }
        }

        // Stage 2: Generate + screen candidates (Phase A)
        let t_candgen = Instant::now();
        let slb_ref = if self.search.config.dp_criterion == DpCriterion::Learned {
            Some(&self.worker.similar_lb)
        } else {
            None
        };
        // The DP's top-B beam is needed even at d=1 — Gini intermediate
        // states aren't safely top-1-prunable.
        let (mut candidates, cand_stats) = generate_candidates_for_support(
            self.search.store.as_ref(),
            &support,
            &self.search.config,
            used_feats,
            self.search.n_total,
            Some(ub_hint),
            slb_ref,
            &self.worker.scratch_bank,
            None,
            None,
            Some(depth),
        );
        let candgen_elapsed = t_candgen.elapsed();
        self.worker.profile.time_candidate_gen += candgen_elapsed;
        {
            let d_idx = (depth as usize).min(MAX_PROFILE_DEPTH - 1);
            self.worker.profile.time_candgen_per_depth[d_idx] += candgen_elapsed;
        }
        // Drop candidates of
        // arity k whose minimum split cost (split_reg + k * leaf_reg)
        // already meets/exceeds leaf_obj. Admissible — no k-way split
        // can have err < 0, so its total cost is strictly greater than
        // its regularization floor.
        {
            let split_reg = (1.0 - self.search.config.reg_alpha) * self.search.config.reg;
            let leaf_reg = self.search.config.reg_alpha * self.search.config.reg;
            candidates.retain(|c| split_reg + (c.k as f64) * leaf_reg < sp_base_obj - 1e-12);
        }
        self.worker.profile.n_candidates_generated += candidates.len() as u64;
        self.worker.profile.n_features_available += cand_stats.n_available;
        self.worker.profile.n_features_filtered_tier1 +=
            cand_stats.n_available - cand_stats.n_filtered_available;
        self.worker.profile.n_candidates_beam_truncated += cand_stats.n_candidates_beam_truncated;
        self.worker.profile.n_teacher_dp_active_nodes += cand_stats.n_teacher_dp_active_nodes;
        for (i, &gap) in cand_stats.beam_gap_per_arity.iter().enumerate() {
            if i < 8 {
                if gap > 0.0 {
                    self.worker.profile.beam_gap_sum_per_arity[i] += gap;
                    self.worker.profile.beam_gap_count_per_arity[i] += 1;
                    if gap > self.worker.profile.beam_gap_max_per_arity[i] {
                        self.worker.profile.beam_gap_max_per_arity[i] = gap;
                    }
                }
                self.worker.profile.beam_saturated_cells_total_per_arity[i] += cand_stats
                    .beam_saturated_cells_per_arity
                    .get(i)
                    .copied()
                    .unwrap_or(0);
            }
        }

        let mut candidates = candidates;
        if !self.worker.feature_k_hits.is_empty() {
            let hits = &self.worker.feature_k_hits;
            candidates.sort_by(|a, b| {
                let ha = hits.get(&(a.feature_index, a.k)).copied().unwrap_or(0);
                let hb = hits.get(&(b.feature_index, b.k)).copied().unwrap_or(0);
                a.split_ub
                    .partial_cmp(&b.split_ub)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then(hb.cmp(&ha))
            });
        }

        // Phase A: count-based inline evaluation (no partition allocation)
        let t_phase_a = Instant::now();
        let split_reg = (1.0 - self.search.config.reg_alpha) * self.search.config.reg;
        let leaf_reg = self.search.config.reg_alpha * self.search.config.reg;
        let mut best_candidate_lb = f64::INFINITY;
        let mut surviving_candidates: Vec<(f64, f64, f64, usize)> = Vec::new();
        let mut scope_evidence_sound = true;
        let mut saw_nontrivial_candidate = false;

        for (cand_idx, cand) in candidates.iter().enumerate() {
            if self.search.timed_out.load(Ordering::Acquire) {
                scope_evidence_sound = false;
                break;
            }

            let n_children = cand.child_n.len();
            let non_empty = cand.child_n.iter().filter(|&&n| n > 0).count();
            if non_empty <= 1 {
                continue;
            }
            saw_nontrivial_candidate = true;

            let exploration_boundary = node_upperbound.min(scope);

            self.worker.solver_scratch.child_order.clear();
            self.worker.solver_scratch.child_order.extend(0..n_children);
            self.worker
                .solver_scratch
                .child_order
                .sort_unstable_by(|&a, &b| cand.child_n[b].cmp(&cand.child_n[a]));

            let mut child_lb_sum = split_reg + path_cost;
            let mut child_ub_sum = split_reg + path_cost;
            let mut pruned_early = false;
            let child_order_len = self.worker.solver_scratch.child_order.len();
            for eval_count in 0..child_order_len {
                let ci = self.worker.solver_scratch.child_order[eval_count];
                let (lb, ub) = self.compute_bounds_from_counts(
                    cand.child_n[ci] as usize,
                    &cand.child_counts[ci],
                    depth - 1,
                );
                child_lb_sum += lb;
                child_ub_sum += ub;

                let remaining = child_order_len - eval_count - 1;
                let optimistic_lb = child_lb_sum + remaining as f64 * leaf_reg;
                if optimistic_lb > exploration_boundary {
                    pruned_early = true;
                    self.worker.profile.n_phase_a_early_terminations += 1;
                    break;
                }
            }

            if !pruned_early && child_lb_sum < best_candidate_lb {
                best_candidate_lb = child_lb_sum;
            }

            if pruned_early || child_lb_sum > exploration_boundary {
                continue;
            }

            self.worker.profile.n_candidates_survived_phase_a += 1;
            let sort_key = if self.search.config.shape_mode {
                cand.rank_cost
            } else {
                child_ub_sum
            };
            surviving_candidates.push((sort_key, child_ub_sum, child_lb_sum, cand_idx));

            if child_ub_sum < node_upperbound {
                node_upperbound = child_ub_sum;
                node_upperbound_is_heuristic = false;
                // Same reasoning as the killer-move site above.
                // child_ub_sum is a candidate UB (split + each child as leaf)
                // without a materialized best_split. Keep local only.
            }
        }

        surviving_candidates
            .sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        let phase_a_elapsed = t_phase_a.elapsed();
        self.worker.profile.time_phase_a += phase_a_elapsed;
        {
            let d_idx = (depth as usize).min(MAX_PROFILE_DEPTH - 1);
            self.worker.profile.time_phase_a_per_depth[d_idx] += phase_a_elapsed;
        }

        let boundary_mode = depth == 1 && self.boundary_guidance_enabled();
        if boundary_mode {
            scope_evidence_sound = false;
        }
        let mut surviving_candidates: Vec<(f64, f64, f64, usize, FlatBins)> = surviving_candidates
            .into_iter()
            .map(|(sort_key, cand_ub, cand_lb, cand_idx)| {
                let cand = &candidates[cand_idx];
                let bins = self.partition_into_flat(
                    &indices,
                    self.search.store.feature_col(cand.feature_index),
                    &cand.thresholds,
                    cand.group_map.as_deref(),
                );
                (sort_key, cand_ub, cand_lb, cand_idx, bins)
            })
            .collect();
        if boundary_mode && !surviving_candidates.is_empty() {
            self.search
                .used_greedy_boundary_guidance
                .store(true, Ordering::Relaxed);
            self.worker.profile.n_greedy_boundary_nodes += 1;
            let t_greedy = Instant::now();
            let mut best_boundary = f64::INFINITY;
            for entry in &mut surviving_candidates {
                let cand_idx = entry.3;
                let cand = &candidates[cand_idx];
                let score = self.boundary_candidate_score(
                    &entry.4,
                    cand.feature_index,
                    n_node,
                    used_feats,
                    best_boundary,
                );
                entry.0 = score;
                entry.1 = score;
                if score < best_boundary {
                    best_boundary = score;
                }
            }
            self.worker.profile.time_greedy_boundary += t_greedy.elapsed();
            surviving_candidates
                .sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

            if let Some((_, best_ub, _, _, _)) = surviving_candidates.first() {
                if *best_ub < node_upperbound {
                    node_upperbound = *best_ub;
                    node_upperbound_is_heuristic = true;
                    let mut sp = self.search.memo.get_mut(key).unwrap();
                    sp.upperbound = *best_ub;
                    sp.upperbound_is_heuristic = true;
                }
            }
        }

        // Stage 3: Recursive exploration of surviving candidates
        let t_phase_b = Instant::now();
        let mut current_scope = scope.min(node_upperbound);
        let mut any_candidate_explored = false;

        let best_surviving_ub = surviving_candidates
            .first()
            .map(|s| s.0)
            .unwrap_or(f64::INFINITY);

        for (cand_rank, (cand_sort_key, cand_ub, cand_lb_sum, cand_idx, bins)) in
            surviving_candidates.iter().enumerate()
        {
            if self.search.timed_out.load(Ordering::Acquire) {
                scope_evidence_sound = false;
                break;
            }
            // Upper-bound margin early exit: candidates with inline UB more
            // than the configured margin worse than the best are skipped.
            let ub_margin = self.search.config.ub_margin;
            if *cand_sort_key - best_surviving_ub > ub_margin {
                self.search
                    .used_ub_margin_prune
                    .store(true, Ordering::Relaxed);
                scope_evidence_sound = false;
                self.worker.profile.n_branch_bound_candidates_scope_pruned +=
                    surviving_candidates.len() as u64 - any_candidate_explored as u64;
                break;
            }
            // Skip candidates whose LB exceeds scope.
            if *cand_lb_sum > current_scope {
                continue;
            }

            // Lazy depth-1 exploration: skip non-competitive candidates whose
            // children would be at depth 1. These depth-1 evaluations dominate
            // runtime (97%+ of nodes) but rarely improve the parent's best.
            let lazy_delta = self.search.config.lazy_explore_delta;
            if depth == 2 && lazy_delta.is_finite() {
                if *cand_ub - best_surviving_ub > lazy_delta {
                    scope_evidence_sound = false;
                    self.worker.profile.n_depth1_lazy_skipped += bins.len() as u64;
                    continue;
                }
            }

            let cand = &candidates[*cand_idx];

            let child_used = if self.search.config.feature_exclusion {
                used_feats.without(cand.feature_index)
            } else {
                used_feats.clone()
            };

            // Materialize children — with split paths if thresholds are pre-binarized
            let thresh_base_branch_bound = self
                .search
                .store
                .ctx()
                .find_threshold_base(cand.feature_index, &cand.thresholds);

            let mut child_keys: Vec<MemoKey> = Vec::with_capacity(bins.len());
            let k_branch_bound = bins.len();
            if let Some(tb) = thresh_base_branch_bound {
                for bin_idx in 0..bins.len() {
                    let b = bins.bin_slice(bin_idx);
                    let step = SplitStep {
                        feature_index: cand.feature_index as u16,
                        threshold_base: tb as u16,
                        bin: bin_idx as u8,
                        k: k_branch_bound as u8,
                    };
                    let child_support = self.search.store.support_from_indices(b);
                    let ck = self.get_or_create_with_path(
                        &child_support,
                        &child_used,
                        depth - 1,
                        &parent_path_branch_bound,
                        step,
                    );
                    child_keys.push(ck);
                }
            } else {
                for bin_idx in 0..bins.len() {
                    let child_support = self
                        .search
                        .store
                        .support_from_indices(bins.bin_slice(bin_idx));
                    let ck = self.get_or_create(&child_support, &child_used, depth - 1);
                    child_keys.push(ck);
                }
            }

            let materialized_lb: f64 = child_keys
                .iter()
                .map(|ck| self.search.memo.get(ck).unwrap().lowerbound)
                .sum::<f64>()
                + split_reg
                + path_cost;

            let (child_cost, candidate_is_heuristic, feasible) = if boundary_mode {
                (*cand_ub, true, *cand_ub <= current_scope + 1e-12)
            } else {
                // Collect child LBs and actual costs for scope budgeting
                let mut child_lbs: Vec<f64> = child_keys
                    .iter()
                    .map(|ck| self.search.memo.get(ck).unwrap().lowerbound)
                    .collect();
                // Sort children by size descending
                let mut child_order: Vec<usize> = (0..child_keys.len()).collect();
                child_order.sort_unstable_by(|&a, &b| bins.bin_len(b).cmp(&bins.bin_len(a)));

                let mut child_cost = split_reg + path_cost;
                let mut feasible = true;
                let mut candidate_is_heuristic = false;

                for (eval_idx, &ci) in child_order.iter().enumerate() {
                    let ck = child_keys[ci].clone();
                    let (
                        child_is_solved,
                        child_is_terminal,
                        child_upperbound,
                        child_ub_is_heuristic,
                    ) = {
                        let csp = self.search.memo.get(&ck).unwrap();
                        (
                            csp.is_solved,
                            csp.is_terminal,
                            csp.upperbound,
                            csp.upperbound_is_heuristic,
                        )
                    };

                    if child_is_solved || child_is_terminal {
                        let actual = child_upperbound;
                        child_cost += actual;
                        candidate_is_heuristic |= child_ub_is_heuristic;
                        // Update child_lbs with actual for tighter sibling scope
                        child_lbs[ci] = actual;
                    } else {
                        // Scope for this child: current_scope minus contributions of all siblings
                        // For resolved siblings, use actual cost; for unresolved, use LB
                        let other_contributions: f64 = child_lbs
                            .iter()
                            .enumerate()
                            .filter(|&(j, _)| j != ci)
                            .map(|(_, &lb)| lb)
                            .sum::<f64>()
                            + split_reg
                            + path_cost;
                        let child_scope = current_scope - other_contributions;

                        if child_scope < -1e-12 {
                            feasible = false;
                            break;
                        }

                        // Track depth-1 children by parent's candidate rank
                        if depth - 1 == 1 {
                            if cand_rank == 0 {
                                self.worker.profile.n_depth1_from_rank0 += 1;
                            } else {
                                self.worker.profile.n_depth1_from_rank_n += 1;
                            }
                        }
                        match self.branch_bound_recurse(&ck, depth - 1, child_scope, &child_used) {
                            BranchBoundResult::Feasible { cost, heuristic } => {
                                child_cost += cost;
                                candidate_is_heuristic |= heuristic;
                                child_lbs[ci] = cost;
                            }
                            BranchBoundResult::Infeasible { floor } => {
                                child_lbs[ci] = child_lbs[ci].max(floor);
                                feasible = false;
                                break;
                            }
                        }
                    }

                    let remaining_lb: f64 = child_order
                        .iter()
                        .skip(eval_idx + 1)
                        .map(|&j| child_lbs[j])
                        .sum();
                    if child_cost + remaining_lb > current_scope {
                        feasible = false;
                        break;
                    }
                }

                (child_cost, candidate_is_heuristic, feasible)
            };

            // Store only the best (tighten scope after each).
            let within_scope = feasible && child_cost <= current_scope;

            if within_scope {
                any_candidate_explored = true;
                current_scope = child_cost;

                let improve_ub = {
                    let mut sp = self.search.memo.get_mut(key).unwrap();
                    let improve_ub = child_cost < sp.upperbound - 1e-12
                        || ((child_cost - sp.upperbound).abs() < 1e-12
                            && !candidate_is_heuristic
                            && sp.upperbound_is_heuristic);
                    if improve_ub {
                        sp.upperbound = child_cost;
                        sp.upperbound_is_heuristic = candidate_is_heuristic;
                    }
                    improve_ub
                };
                if improve_ub {
                    node_upperbound = child_cost;
                    node_upperbound_is_heuristic = candidate_is_heuristic;
                }

                let k = cand.thresholds.len() + 1;
                *self
                    .worker
                    .feature_k_hits
                    .entry((cand.feature_index, k))
                    .or_insert(0) += 1;
                self.worker
                    .killer_moves
                    .insert(depth, (cand.feature_index, k, cand.thresholds.clone()));

                let should_update = match self.resolve_best_split(key) {
                    None => true,
                    Some(existing) => child_cost <= existing.split_upper,
                };
                if should_update {
                    let split_id = Self::split_record_id(
                        cand.feature_index,
                        &cand.thresholds,
                        &child_keys,
                        cand.group_map.as_deref(),
                    );
                    // Persist the SplitRecord identity in the
                    // pool + bound_table; best_split_table records the
                    // chosen split_id. The full SplitRecord is recoverable
                    // via `resolve_best_split(key)`.
                    let interned = self.intern_keys(child_keys.clone());
                    let feat = cand.feature_index as u16;
                    let split_upper = child_cost;
                    let record = SplitRecord {
                        split_id,
                        feature_index: cand.feature_index,
                        thresholds: cand.thresholds.clone(),
                        child_keys: interned,
                        split_upper,
                        contextual_lowerbound: f64::NEG_INFINITY,
                        contextually_dead: false,
                        group_map: cand.group_map.clone(),
                    };
                    let pool_idx = self.worker.split_pool.push(record);
                    self.search.bound_upsert_materialized(
                        key,
                        feat,
                        pool_idx,
                        f64::NEG_INFINITY,
                        split_upper,
                    );
                    self.search.set_best_split_id(key, split_id);
                }

                if !candidate_is_heuristic {
                    self.similar_ub_update(sig.n_samples, sig.as_slice(), child_cost);
                }
            }

            if materialized_lb < best_candidate_lb {
                best_candidate_lb = materialized_lb;
            }
        }

        let phase_b_elapsed = t_phase_b.elapsed();
        self.worker.profile.time_phase_b += phase_b_elapsed;
        {
            let d_idx = (depth as usize).min(MAX_PROFILE_DEPTH - 1);
            self.worker.profile.time_phase_b_per_depth[d_idx] += phase_b_elapsed;
        }
        self.worker.profile.n_candidates_materialized += surviving_candidates.len() as u64;
        if self.search.timed_out.load(Ordering::Acquire) {
            scope_evidence_sound = false;
        }

        // Stage 4: Finalization
        if !any_candidate_explored && surviving_candidates.is_empty() {
            let scope_exhaustive = scope >= node_upperbound - 1e-12;
            let globally_forced_leaf =
                scope_evidence_sound && (!saw_nontrivial_candidate || scope_exhaustive);
            if globally_forced_leaf {
                let (base_obj, leaf_sig) = {
                    let mut sp = self.search.memo.get_mut(key).unwrap();
                    // This branch claims "no candidate can improve over the
                    // current UB" (Phase A pruned candidates admissibly wrt
                    // node_upperbound). Collapsing unconditionally to a leaf
                    // would worsen sp.upperbound back to sp.base_objective when
                    // a prior call had recorded a real split with cost <
                    // base_objective, and extract_tree would then emit a leaf
                    // instead of the real subtree.
                    let leaf_is_best = sp.upperbound >= sp.base_objective - 1e-12;
                    if !leaf_is_best {
                        // An earlier call established a better tree. Preserve
                        // its UB and best_split; just mark solved at that UB.
                        sp.lowerbound = sp.upperbound;
                        sp.is_solved = true;
                        (
                            sp.upperbound,
                            SupportSig::new(
                                sp.n_samples,
                                &sp.class_counts[..sp.n_classes as usize],
                            ),
                        )
                    } else {
                        sp.lowerbound = sp.base_objective;
                        sp.upperbound = sp.base_objective;
                        sp.upperbound_is_heuristic = false;
                        sp.is_terminal = true;
                        sp.is_solved = true;
                        (
                            sp.base_objective,
                            SupportSig::new(
                                sp.n_samples,
                                &sp.class_counts[..sp.n_classes as usize],
                            ),
                        )
                    }
                };
                self.similar_lb_update(leaf_sig.n_samples, leaf_sig.as_slice(), depth, base_obj);
                self.similar_ub_update(leaf_sig.n_samples, leaf_sig.as_slice(), base_obj);
                return BranchBoundResult::Feasible {
                    cost: base_obj,
                    heuristic: false,
                };
            }

            if scope_evidence_sound {
                self.record_branch_bound_scope_floor(key, scope);
                return BranchBoundResult::Infeasible { floor: scope };
            }

            return BranchBoundResult::Feasible {
                cost: node_upperbound,
                heuristic: node_upperbound_is_heuristic,
            };
        }

        // In branch-bound, if we completed without timeout, this node is solved:
        // - Every surviving candidate was either explored or pruned (LB > scope)
        // - Every non-surviving candidate was pruned in Phase A
        // - The best found cost IS the optimal for this subproblem
        // The only exception: timeout interrupted exploration
        let best_split_child_keys: Option<Vec<u32>> =
            self.resolve_best_split(key).map(|bs| bs.child_keys.clone());
        let all_best_children_solved = match &best_split_child_keys {
            Some(child_keys) => child_keys.iter().all(|&ck_idx| {
                self.search
                    .memo
                    .get(self.ck(ck_idx))
                    .map(|c| c.is_solved || c.is_terminal)
                    .unwrap_or(false)
            }),
            // No split improved over leaf cost — node is optimally a leaf
            None => true,
        };

        let scope_exhaustive = scope >= node_upperbound - 1e-12;
        let is_solved = scope_exhaustive
            && scope_evidence_sound
            && !self.search.timed_out.load(Ordering::Acquire)
            && all_best_children_solved
            && !node_upperbound_is_heuristic;

        let new_lb = if is_solved {
            node_upperbound
        } else {
            // Partial exploration: use best available LB
            let mut lb = current_lb;
            if best_candidate_lb < f64::INFINITY && best_candidate_lb > lb {
                lb = sp_base_obj.min(best_candidate_lb);
            }
            lb.min(node_upperbound)
        };

        let final_ub = {
            let mut sp = self.search.memo.get_mut(key).unwrap();
            sp.lowerbound = new_lb;
            sp.is_solved = is_solved;
            sp.upperbound = node_upperbound;
            sp.upperbound_is_heuristic = node_upperbound_is_heuristic;
            // If solved with no split (leaf optimal), mark as terminal
            if is_solved && !self.search.best_split_table.contains_key(key) {
                sp.is_terminal = true;
                sp.upperbound = sp.base_objective;
                sp.upperbound_is_heuristic = false;
            }
            sp.upperbound
        };
        self.similar_lb_update(sig.n_samples, sig.as_slice(), depth, new_lb);
        if is_solved && !node_upperbound_is_heuristic {
            self.similar_ub_update(sig.n_samples, sig.as_slice(), final_ub);
        }

        if final_ub <= scope + 1e-12 {
            BranchBoundResult::Feasible {
                cost: final_ub,
                heuristic: node_upperbound_is_heuristic,
            }
        } else if scope_evidence_sound {
            self.record_branch_bound_scope_floor(key, scope);
            BranchBoundResult::Infeasible { floor: scope }
        } else {
            BranchBoundResult::Feasible {
                cost: final_ub,
                heuristic: node_upperbound_is_heuristic,
            }
        }
    }

    // ------------------------------------------------------------------
    // Tree extraction
    // ------------------------------------------------------------------

    pub fn extract_tree(&self) -> TreeNode {
        let root_key = self.search.root_key.lock().unwrap().clone();
        self.extract_node(&root_key)
    }

    fn extract_node(&self, key: &MemoKey) -> TreeNode {
        let sp = self.search.memo.get(key).unwrap();
        let make_leaf = |sp: &SubProblem| {
            let prediction = if sp.n_samples > 0 {
                sp.class_counts[..sp.n_classes as usize]
                    .iter()
                    .enumerate()
                    .max_by_key(|(_, &c)| c)
                    .map(|(i, _)| i)
                    .unwrap_or(0)
            } else {
                0
            };
            TreeNode::Leaf {
                prediction,
                n_samples: sp.n_samples,
                loss: sp.max_loss,
            }
        };

        match self.resolve_best_split(key) {
            None => make_leaf(&sp),
            Some(sr) => {
                // best_split may be stored for propagation tracking even when
                // the split doesn't beat the leaf cost.  Emit a leaf if the
                // node's actual objective chose the leaf (UB == base_obj).
                if sp.upperbound >= sp.base_objective - 1e-12 {
                    make_leaf(&sp)
                } else {
                    let children: Vec<TreeNode> = sr
                        .child_keys
                        .iter()
                        .map(|&ck_idx| self.extract_node(self.ck(ck_idx)))
                        .collect();
                    TreeNode::Internal {
                        feature_index: sr.feature_index,
                        thresholds: sr.thresholds.clone(),
                        children,
                        group_map: sr.group_map.clone(),
                    }
                }
            }
        }
    }

    /// Take ownership of the memo table (moves it out of the solver).
    pub fn take_memo(self) -> HashMap<MemoKey, SubProblem> {
        match Arc::try_unwrap(self.search) {
            Ok(state) => state.memo.into_iter().collect(),
            Err(arc) => arc
                .memo
                .iter()
                .map(|e| (e.key().clone(), e.value().clone()))
                .collect(),
        }
    }

    // ------------------------------------------------------------------
    // Pre-seeding root with parallel root candidates
    // ------------------------------------------------------------------

    /// Pre-seed the root SubProblem with root candidates discovered by
    /// Phase 1's parallel root solver. This ensures Phase 2's sequential
    /// BFS knows about good root splits that it might not discover on its own.
    pub fn preseed_root_candidates(&mut self, candidates: &[RootCandidate], rashomon_bound: f64) {
        if candidates.is_empty() {
            return;
        }

        // Create root SubProblem if not already in memo
        let root_support = self.search.root_support.clone();
        let root_used = self.search.config.initial_used_features.clone();
        let root_key = self.get_or_create(&root_support, &root_used, self.search.config.max_depth);
        *self.search.root_key.lock().unwrap() = root_key.clone();
        let root_indices = self.search.store.support_to_indices(&root_support);

        // Import each Phase 1 finalist tree into the Phase 2 memo, so Phase 2
        // starts from real solved subtrees rather than a bare root split shell.
        for cand in candidates {
            if cand.objective > rashomon_bound + 1e-10 {
                continue;
            }
            let _ = self.import_phase1_tree_at_key(
                root_key.clone(),
                &root_indices,
                self.search.config.max_depth,
                &cand.tree,
                &root_used,
                rashomon_bound,
                Some(&[]),
            );
        }

        // Step A: `scope` is the widest explored scope and the Step-A
        // pass starts from the root itself; nothing to enqueue.
        if self.search.config.rashomon_mode {
            return;
        }

        // Enqueue root for exploration
        let (gap, depth, requested_scope) = {
            let mut root = self.search.memo.get_mut(&root_key).unwrap();
            if !root.scope.is_finite() {
                root.scope = root.upperbound;
            }
            (root.upperbound - root.lowerbound, root.depth, root.scope)
        };
        self.enqueue_explore(root_key, gap, depth, requested_scope);
    }

    // ------------------------------------------------------------------
    // Rashomon DAG construction
    // ------------------------------------------------------------------

    /// Build a compressed Rashomon DAG from the memo table after a Phase 2
    /// Build a factored Rashomon DAG (cross-product-free) from the memo table.
    /// `cost_precision`: quantization granularity (e.g. 1e9, 1e7, 1e5).
    /// `total_max_depth`: the user-requested overall tree depth. When greater
    /// than the solver's configured `max_depth` (i.e. lookahead truncation),
    /// the DAG builder marks depth-0 forced leaves with `PrefixLeafInfo` so
    /// downstream code can run recursive fill on them.
    ///
    /// Consumes the solver to move (not clone) the memo table, halving peak
    /// memory during DAG construction.
    pub fn build_factored_rashomon_dag(
        self,
        opt: f64,
        theta: f64,
        delta_budget: crate::types::DeltaBudget,
        cost_precision: f64,
        total_max_depth: i32,
    ) -> crate::types::FactoredDag {
        let root_key = self.search.root_key.lock().unwrap().clone();
        let rashomon_bound = self.search.config.rashomon_bound;
        let reg = self.search.config.reg;
        let solve_depth = self.search.config.max_depth;
        let cost_lattice = self.cost_lattice();
        // Extract pool + arena before take_memo (which consumes self).
        let BranchAndBoundSolver { search, worker } = self;
        let pool = worker.split_pool;
        // Reconstruct a self-like view to call take_memo's logic inline.
        // Snapshot the arena via Arc::clone before consuming search.
        let arena = Arc::clone(&search.memo_key_arena);
        let memo: HashMap<MemoKey, SubProblem> = match Arc::try_unwrap(search) {
            Ok(state) => state.memo.into_iter().collect(),
            Err(arc) => arc
                .memo
                .iter()
                .map(|e| (e.key().clone(), e.value().clone()))
                .collect(),
        };
        crate::rashomon_dag::build_factored_dag(
            &memo,
            &pool,
            &arena,
            &root_key,
            rashomon_bound,
            opt,
            theta,
            reg,
            delta_budget,
            cost_precision,
            cost_lattice,
            solve_depth,
            total_max_depth,
        )
    }

    /// Consume the solver (after the Rashomon exploration pass) into a
    /// `SearchSnapshot`, deferring DAG
    /// construction to a later `RashomonSet::ensure_built` call. Mirrors
    /// `build_factored_rashomon_dag` but without the actual build —
    /// `ensure_built` will run `build_factored_dag` + `quotient_factored_dag`
    /// from the snapshot's owned inputs and drop them at the end so the
    /// FactoredDag is the sole resident state.
    pub fn into_search_snapshot(
        self,
        opt: f64,
        theta: f64,
        delta_budget: crate::types::DeltaBudget,
        cost_precision: f64,
        total_max_depth: i32,
    ) -> crate::types::SearchSnapshot {
        let root_key = self.search.root_key.lock().unwrap().clone();
        let rashomon_bound = self.search.config.rashomon_bound;
        let reg = self.search.config.reg;
        let solve_depth = self.search.config.max_depth;
        let cost_lattice = self.cost_lattice();
        let BranchAndBoundSolver { search, worker } = self;
        let pool = worker.split_pool;
        let arena = Arc::clone(&search.memo_key_arena);
        // Move the memo DashMap as is (no re-hash into a new table) and free
        // the other search tables first.
        let memo: crate::types::SnapshotMemo = match Arc::try_unwrap(search) {
            Ok(state) => {
                let SearchState {
                    memo,
                    parent_map,
                    bound_table,
                    best_split_table,
                    ..
                } = state;
                drop(parent_map);
                drop(bound_table);
                drop(best_split_table);
                memo.into_read_only()
            }
            Err(arc) => arc.memo.clone().into_read_only(),
        };
        crate::types::SearchSnapshot {
            memo,
            pool,
            arena,
            root_key,
            rashomon_bound,
            opt,
            theta,
            reg,
            delta_budget,
            cost_precision,
            cost_lattice,
            solve_depth,
            total_max_depth,
            prefix_leaf_fill: None,
        }
    }
}

// ======================================================================
// Parallel root solve: evaluate top-N root candidates in parallel
// ======================================================================

/// Canonical tiebreak over invariant candidate properties.
///
/// Breaks ties in screen_obj / greedy_score by lex order on
/// `(feature_index, k, family, thresholds_bits, group_map)` — all intrinsic
/// to the candidate, so the resulting rank is independent of par_iter worker
/// finish order AND of the candidate-generation insertion order.
pub(crate) fn candidate_tiebreak_cmp(a: &SplitCandidate, b: &SplitCandidate) -> std::cmp::Ordering {
    use std::cmp::Ordering;
    a.feature_index
        .cmp(&b.feature_index)
        .then_with(|| a.k.cmp(&b.k))
        .then_with(|| a.family.cmp(&b.family))
        .then_with(|| {
            for (ta, tb) in a.thresholds.iter().zip(b.thresholds.iter()) {
                match ta.to_bits().cmp(&tb.to_bits()) {
                    Ordering::Equal => continue,
                    ord => return ord,
                }
            }
            a.thresholds.len().cmp(&b.thresholds.len())
        })
        .then_with(|| match (&a.group_map, &b.group_map) {
            (None, None) => Ordering::Equal,
            (None, Some(_)) => Ordering::Less,
            (Some(_), None) => Ordering::Greater,
            (Some(ga), Some(gb)) => ga.cmp(gb),
        })
}

pub fn solve_parallel_root(
    x: Array2<f64>,
    y: Vec<u8>,
    config: SolverConfig,
    top_n: usize,
    weights: Option<Vec<f64>>,
) -> (SolveResult, TreeNode, Vec<RootCandidate>, SolverProfile) {
    solve_parallel_root_screen(x, y, config, top_n, weights, None)
}

/// `solve_parallel_root` with the final-screen child time limit chosen by
/// the caller: `full_screen_cap = Some(cap)` gives the children of a screen
/// that IS the final solve (`!need_deep`) `min(time_limit, cap)` instead of
/// 1 s (the Rashomon Phase 1, `phase1_screen_cap_s`). None = 1 s, as
/// `solve_parallel_root` (every non-Rashomon caller).
pub fn solve_parallel_root_screen(
    x: Array2<f64>,
    y: Vec<u8>,
    config: SolverConfig,
    top_n: usize,
    weights: Option<Vec<f64>>,
    full_screen_cap: Option<f64>,
) -> (SolveResult, TreeNode, Vec<RootCandidate>, SolverProfile) {
    let start = Instant::now();
    let reference_preds_vec = config.reference_preds.as_ref().map(|arc| arc.to_vec());
    let store = Arc::new(DatasetStore::new(
        x,
        y,
        weights,
        config.n_classes,
        config.thresholds_per_feature.as_ref(),
        reference_preds_vec,
    ));
    let n_data = store.n_samples();
    let n_total = config.n_total_override.unwrap_or(n_data);
    let n_classes = config.n_classes;
    let n_features = config.n_features;
    let reg = config.reg;
    let max_depth = config.max_depth;
    let max_k = config.max_k;
    let min_samples_leaf = config.min_samples_leaf;
    let feature_exclusion = config.feature_exclusion;
    let time_limit = config.time_limit;
    let feature_names = config.feature_names.clone();
    let used_features = config.initial_used_features.clone();

    // Compute root leaf cost (fallback)
    let mut class_counts = [0i64; MAX_CLASSES];
    for &label in store.labels() {
        class_counts[label as usize] += 1;
    }
    let max_count = *class_counts[..n_classes].iter().max().unwrap_or(&0);
    let root_max_loss = if n_data > 0 {
        (n_data as f64 - max_count as f64) / n_total as f64
    } else {
        0.0
    };
    let alpha = config.reg_alpha;
    let leaf_reg = alpha * reg;
    let root_leaf_obj = root_max_loss + leaf_reg;

    // Per-phase scalar UB snapshots (no shared atomic UB). Phase 1 workers share `root_leaf_obj`;
    // Phase 2 workers share `best_screen_obj` (a deterministic function of
    // Phase 1's canonically-ranked results). There are no mid-solve atomic
    // reads/writes, so the result does not depend on rayon scheduling.
    //
    // NOTE: shared_bounds is created per-candidate (inside the par_iter closure below),
    // not here. Sharing bounds across candidates with different feature exclusions is
    // incorrect: a UB found with feature F excluded is not valid when feature G is
    // excluded instead, since the tree that achieved that UB may use feature G.

    // Edge cases: can't split
    if max_depth <= 0 || n_data < 2 || root_max_loss == 0.0 {
        let prediction = class_counts[..n_classes]
            .iter()
            .enumerate()
            .max_by_key(|(_, &c)| c)
            .map(|(i, _)| i)
            .unwrap_or(0);
        let tree = TreeNode::Leaf {
            prediction,
            n_samples: n_data,
            loss: root_max_loss,
        };
        return (
            SolveResult {
                objective: root_leaf_obj,
                lowerbound: root_leaf_obj,
                is_optimal: true,
                certified_optimal: true,
                search_converged: true,
                certification_mode: "exact".to_string(),
                certification_reason: "certified".to_string(),
                unsafe_heuristics_used: false,
                timed_out: false,
                termination_reason: "optimal".to_string(),
                final_gap: 0.0,
                n_subproblems: 1,
                n_explored: 0,
                elapsed: start.elapsed().as_secs_f64(),
            },
            tree,
            vec![], // no root candidates for edge case
            SolverProfile::default(),
        );
    }

    // Generate root candidates (already parallel per feature)
    let all_indices: Vec<u32> = (0..n_data as u32).collect();
    let root_support = store.root_support();
    let scratch_bank = ScratchBank::new();
    let (candidates, _cand_stats) = generate_candidates_for_support(
        store.as_ref(),
        &root_support,
        &config,
        &used_features,
        n_total,
        Some(root_leaf_obj),
        None,
        &scratch_bank,
        None,
        None,
        Some(config.max_depth), // root: pass max_depth so unsafe filter can fire
    );

    if candidates.is_empty() {
        let prediction = class_counts[..n_classes]
            .iter()
            .enumerate()
            .max_by_key(|(_, &c)| c)
            .map(|(i, _)| i)
            .unwrap_or(0);
        let tree = TreeNode::Leaf {
            prediction,
            n_samples: n_data,
            loss: root_max_loss,
        };
        return (
            SolveResult {
                objective: root_leaf_obj,
                lowerbound: root_leaf_obj,
                is_optimal: true,
                certified_optimal: true,
                search_converged: true,
                certification_mode: "exact".to_string(),
                certification_reason: "certified".to_string(),
                unsafe_heuristics_used: false,
                timed_out: false,
                termination_reason: "optimal".to_string(),
                final_gap: 0.0,
                n_subproblems: 1,
                n_explored: 0,
                elapsed: start.elapsed().as_secs_f64(),
            },
            tree,
            vec![], // no root candidates
            SolverProfile::default(),
        );
    }

    // Pre-compute partitions for ALL candidates
    let all_candidates: Vec<SplitCandidate> = candidates;
    let partitions: Vec<FlatBins> = all_candidates
        .iter()
        .map(|cand| {
            let n_bins = cand.thresholds.len() + 1;
            let k = cand
                .group_map
                .as_ref()
                .map_or(n_bins, |gm| gm.iter().copied().max().unwrap_or(0) + 1);
            let mut counts = vec![0usize; k];
            for &idx in &all_indices {
                let val = store.feature_col(cand.feature_index)[idx as usize];
                let bin = BranchAndBoundSolver::bin_index_right(&cand.thresholds, val);
                let group = cand.group_map.as_ref().map_or(bin, |gm| gm[bin]);
                counts[group] += 1;
            }
            let mut offsets = Vec::with_capacity(k + 1);
            offsets.push(0);
            for count in counts {
                let next = offsets.last().copied().unwrap() + count;
                offsets.push(next);
            }
            let mut flat = vec![0u32; all_indices.len()];
            let mut cursors = offsets[..k].to_vec();
            for &idx in &all_indices {
                let val = store.feature_col(cand.feature_index)[idx as usize];
                let bin = BranchAndBoundSolver::bin_index_right(&cand.thresholds, val);
                let group = cand.group_map.as_ref().map_or(bin, |gm| gm[bin]);
                let pos = cursors[group];
                flat[pos] = idx;
                cursors[group] += 1;
            }
            FlatBins { offsets, flat }
        })
        .collect();

    let split_reg = (1.0 - alpha) * reg;
    let path_cost = config.path_penalty * n_data as f64 / n_total as f64;

    // Profile aggregation happens outside the par_iter bodies.
    // Each worker returns its own `SolverProfile`; `solve_candidate` merges
    // its k children in bin_idx order; `solve_parallel_root` merges across
    // candidates in canonical (finalist-index) order after Phase 1 and
    // Phase 2 collect. This makes the final aggregated profile a
    // deterministic function of inputs (f64 sums in `merge_from` do not
    // depend on rayon arrival order).

    // shared_bounds is disabled in the parallel root: sharing bounds across
    // concurrent workers gave worse objectives in testing.
    let shared_bounds: Option<Arc<SharedBoundsTable>> = None;
    let shared_closure: Option<Arc<DashMap<MemoKey, SubProblem>>> = None;

    // Helper: solve one candidate's children and return aggregated result.
    // child_depth and child_time_limit control the solve budget.
    //
    // `warm_start_ub` is a SCALAR SNAPSHOT fixed by the caller
    // before the enclosing par_iter dispatches. Workers never read a shared
    // atomic mid-solve, and this helper no longer writes one. Callers merge
    // results in canonical order post-par_iter to feed the next phase's
    // snapshot deterministically.
    let solve_candidate = |cand: &SplitCandidate,
                           bins: &FlatBins,
                           child_depth: i32,
                           child_time_limit: Option<f64>,
                           warm_start_ub: f64,
                           store: &Arc<DatasetStore>,
                           use_local_thresh: bool,
                           scan_beam: usize|
     -> (
        f64,
        f64,
        bool,
        bool,
        usize,
        usize,
        Vec<TreeNode>,
        SolverProfile,
    ) {
        let child_used = if feature_exclusion {
            used_features.without(cand.feature_index)
        } else {
            used_features.clone()
        };

        let k = bins.len();
        // Return each child's profile along with its result. Rayon's
        // `par_iter().map().collect::<Vec<_>>()` preserves input (bin_idx)
        // order, so the subsequent merge happens in a canonical order.
        let child_results: Vec<(SolveResult, TreeNode, SolverProfile)> = (0..k)
            .into_par_iter()
            .map(|bin_idx| {
                let bin_indices = bins.bin_slice(bin_idx);
                if bin_indices.is_empty() {
                    return (
                        SolveResult {
                            objective: leaf_reg,
                            lowerbound: leaf_reg,
                            is_optimal: true,
                            certified_optimal: true,
                            search_converged: true,
                            certification_mode: "exact".to_string(),
                            certification_reason: "certified".to_string(),
                            unsafe_heuristics_used: false,
                            timed_out: false,
                            termination_reason: "leaf".to_string(),
                            final_gap: 0.0,
                            n_subproblems: 1,
                            n_explored: 0,
                            elapsed: 0.0,
                        },
                        TreeNode::Leaf {
                            prediction: 0,
                            n_samples: 0,
                            loss: 0.0,
                        },
                        SolverProfile::default(),
                    );
                }

                // Pruning budget derived from a scalar snapshot (not a
                // mid-par_iter atomic load), so every worker under a given
                // candidate sees the same UB regardless of sibling finish
                // order. Cross-candidate pruning is replayed in the caller's
                // canonical merge step between phases.
                let child_budget = warm_start_ub - (k - 1) as f64 * leaf_reg;

                // DP² local threshold scan: compute locally-relevant thresholds
                let (local_thresholds, eff_k, eff_beam) = if use_local_thresh {
                    let lt = dp_threshold_scan(
                        store.row_major(),
                        store.labels(),
                        bin_indices,
                        &config,
                        n_total,
                        scan_beam,
                    );
                    if lt.is_empty() {
                        (
                            config.thresholds_per_feature.clone(),
                            max_k,
                            config.beam_width,
                        )
                    } else {
                        let max_local = lt.values().map(|v| v.len()).max().unwrap_or(0);
                        let eff_k = (max_local + 1).max(2).min(max_k);
                        let eff_beam = config.beam_width.max(1usize << max_local.min(8));
                        (Some(lt), eff_k, eff_beam)
                    }
                } else {
                    (
                        config.thresholds_per_feature.clone(),
                        max_k,
                        config.beam_width,
                    )
                };

                let child_config = SolverConfig {
                    n_classes,
                    n_features,
                    reg,
                    max_depth: child_depth,
                    max_k: eff_k,
                    min_samples_leaf,
                    feature_exclusion,
                    time_limit: child_time_limit,
                    initial_used_features: child_used.clone(),
                    feature_names: feature_names.clone(),
                    feature_types: config.feature_types.clone(),
                    warm_start_ub: if child_budget > 0.0 && child_budget < f64::INFINITY {
                        Some(child_budget)
                    } else {
                        None
                    },
                    n_total_override: Some(n_total),
                    thresholds_per_feature: local_thresholds,
                    closure_thresholds_per_feature: config.closure_thresholds_per_feature.clone(),
                    closure_split_ub_gap: config.closure_split_ub_gap,
                    closure_feature_top_k: config.closure_feature_top_k,
                    closure_feature_ranking: config.closure_feature_ranking.clone(),
                    closure_min_n_for_multiway: config.closure_min_n_for_multiway,
                    closure_unsafe_filter_depth_threshold: config
                        .closure_unsafe_filter_depth_threshold,
                    dp_criterion: config.dp_criterion,
                    beam_width: eff_beam,
                    reg_alpha: config.reg_alpha,
                    local_thresh_scan: false, // don't recurse scan into grandchildren
                    scan_beam: config.scan_beam,
                    adaptive_beam: config.adaptive_beam,
                    bypass_beam: config.bypass_beam,

                    transition_delta: config.transition_delta,
                    materialize_top_n: config.materialize_top_n,
                    allow_heuristic_bound_transfer: false,
                    shape_mode: config.shape_mode,
                    shape_sort_smoothing: config.shape_sort_smoothing,
                    path_penalty: config.path_penalty,
                    greedy_extra_depth: config.greedy_extra_depth,
                    rashomon_mode: false,
                    rashomon_bound: f64::INFINITY,
                    step_b_remaining: 0,
                    weight_total: n_total as f64,
                    ub_margin: config.ub_margin,
                    n_workers: 1,
                    sequential_candidate_gen: false,
                    lazy_explore_delta: config.lazy_explore_delta,
                    cheap_boundary_greedy: config.cheap_boundary_greedy,
                    closure_depth: config.closure_depth,
                    closure_beam: config.closure_beam,
                    closure_parallel: config.closure_parallel,
                    closure_share_memo: config.closure_share_memo,
                    closure_shared_memo: shared_closure.clone(),
                    closure_criterion: config.closure_criterion,
                    phase0_greedy_prerank_depth: config.phase0_greedy_prerank_depth,
                    eqp_inline_bounds: config.eqp_inline_bounds,
                    threshold_interner: config.threshold_interner.clone(),
                    rash_tuning: config.rash_tuning.clone(),
                    msdt: config.msdt.clone(),
                    teacher_prob: config.teacher_prob.clone(),
                    teacher_prob_c0: config.teacher_prob_c0.clone(),
                    reference_preds: config.reference_preds.clone(),
                    unsafe_filter_depth_threshold: 0,
                    admissibility_strict: config.admissibility_strict,
                };

                let child_support = store.support_from_indices(bin_indices);
                // Pass `shared_bounds` (always `None`) through.
                let mut solver = BranchAndBoundSolver::new(
                    store.clone(),
                    child_config,
                    shared_bounds.clone(),
                    Some(child_support),
                );
                let result = solver.solve();
                let tree = solver.extract_tree();
                // Return the profile clone instead of merging into a
                // shared Mutex. Caller merges in canonical (bin_idx) order.
                let worker_profile = solver.worker.profile.clone();
                (result, tree, worker_profile)
            })
            .collect();

        // Merge per-child profiles in bin_idx order.
        let mut candidate_profile = SolverProfile::default();
        for (_, _, p) in &child_results {
            candidate_profile.merge_from(p);
        }

        let total_obj: f64 = child_results
            .iter()
            .map(|(r, _, _)| r.objective)
            .sum::<f64>()
            + split_reg
            + path_cost;
        let total_lb: f64 = child_results
            .iter()
            .map(|(r, _, _)| r.lowerbound)
            .sum::<f64>()
            + split_reg
            + path_cost;
        let all_optimal = child_results.iter().all(|(r, _, _)| r.is_optimal);
        let any_timed_out = child_results.iter().any(|(r, _, _)| r.timed_out);
        let total_subproblems: usize = child_results.iter().map(|(r, _, _)| r.n_subproblems).sum();
        let total_explored: usize = child_results.iter().map(|(r, _, _)| r.n_explored).sum();
        let child_trees: Vec<TreeNode> = child_results.into_iter().map(|(_, t, _)| t).collect();

        // No shared-atomic update. The caller observes `total_obj` via
        // the par_iter result vec and merges in canonical order.
        // Return `candidate_profile` (merged in bin_idx order) so the
        // caller can further merge across candidates in finalist-index order.
        (
            total_obj,
            total_lb,
            all_optimal,
            any_timed_out,
            total_subproblems,
            total_explored,
            child_trees,
            candidate_profile,
        )
    };

    // ── Phase 0 (optional): Greedy-CART pre-ranking ───────────────────────
    // When phase0_greedy_prerank_depth > 0, pre-score every root candidate by
    // recursive greedy-CART expansion at the requested depth and forward only
    // the top `2 * top_n` candidates to Phase 1 deep screening. Saves Phase 1
    // solve budget on clearly-inferior root features. Lossless wrt final tree
    // quality given top_n is itself a heuristic; biggest impact at la=3+ on
    // datasets with many features.
    let phase0_depth = config.phase0_greedy_prerank_depth;
    let pre_ranked_indices: Option<Vec<usize>> = if phase0_depth > 0 {
        let greedy_solver = BranchAndBoundSolver::new(
            store.clone(),
            SolverConfig {
                max_depth: max_depth - 1,
                phase0_greedy_prerank_depth: 0,
                eqp_inline_bounds: true, // don't recurse into prerank
                ..config.clone()
            },
            None,
            None,
        );
        let mut greedy_scores: Vec<(usize, f64)> = all_candidates
            .iter()
            .enumerate()
            .zip(partitions.iter())
            .map(|((i, _cand), bins)| {
                let mut child_cost = split_reg + path_cost;
                for bin_idx in 0..bins.len() {
                    let bin_indices = bins.bin_slice(bin_idx);
                    if bin_indices.is_empty() {
                        child_cost += leaf_reg;
                    } else {
                        child_cost += greedy_solver.greedy_cart_ub(bin_indices, phase0_depth);
                    }
                }
                (i, child_cost.min(root_leaf_obj))
            })
            .collect();
        // Canonical tiebreak so pre-rank is deterministic w.r.t. candidates
        // alone (independent of any rayon ordering).
        greedy_scores.sort_by(|a, b| {
            a.1.partial_cmp(&b.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| candidate_tiebreak_cmp(&all_candidates[a.0], &all_candidates[b.0]))
        });
        Some(
            greedy_scores
                .iter()
                .take(top_n * 2)
                .map(|&(i, _)| i)
                .collect(),
        )
    } else {
        None
    };

    // ── Phase 1: Screen candidates with shallow child solves ─────────────
    // Use min(2, max_depth-1) depth and a 1-second time limit per child.
    // At low depth this is exact; at high depth it's a cheap approximation.
    // When `!need_deep` the screen IS the final solve: with
    // `full_screen_cap = Some(cap)` (Rashomon Phase 1)
    // its children get min(time_limit, cap); otherwise the 1 s.
    //
    // Every worker in Phase 1 uses the same scalar UB snapshot
    // (`root_leaf_obj`). No cross-worker pruning within Phase 1; workers
    // still benefit from the root-leaf floor.
    let screen_depth = (max_depth - 1).min(2);
    let full_depth = max_depth - 1;
    let need_deep = screen_depth < full_depth;
    let screen_time = match full_screen_cap {
        Some(cap) if !need_deep => Some(time_limit.unwrap_or(f64::INFINITY).min(cap)),
        _ => Some(1.0_f64),
    };
    let screen_ub_snapshot = root_leaf_obj;

    // If greedy pre-ranked, only screen those; otherwise screen all.
    let screen_indices: Vec<usize> =
        pre_ranked_indices.unwrap_or_else(|| (0..all_candidates.len()).collect());

    type CandResult = (
        f64,
        f64,
        bool,
        bool,
        usize,
        usize,
        Vec<TreeNode>,
        SolverProfile,
    );
    let screen_results: Vec<(usize, CandResult)> = screen_indices
        .par_iter()
        .map(|&ci| {
            let phase1_scan = config.local_thresh_scan && !need_deep;
            let r = solve_candidate(
                &all_candidates[ci],
                &partitions[ci],
                screen_depth,
                screen_time,
                screen_ub_snapshot,
                &store,
                phase1_scan,
                config.scan_beam,
            );
            (ci, r)
        })
        .collect();

    // Rank by screening objective, pick top_n finalists.
    // Canonical tiebreak on screen_obj ties.
    let mut ranked: Vec<(usize, f64)> = screen_results.iter().map(|(ci, r)| (*ci, r.0)).collect();
    ranked.sort_by(|a, b| {
        a.1.partial_cmp(&b.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| candidate_tiebreak_cmp(&all_candidates[a.0], &all_candidates[b.0]))
    });
    let finalist_indices: Vec<usize> = ranked.iter().take(top_n).map(|&(i, _)| i).collect();

    // ── Phase 2: Deep solve finalists with full budget ────────────────────
    // If screen_depth == max_depth-1, screening was already the full solve —
    // just reuse those results.
    //
    // Phase 2 workers all share the same snapshot UB derived
    // deterministically from Phase 1's canonically-ranked results
    // (`best_screen_obj` is already computed from `ranked` after the
    // canonical tiebreak). No cross-worker pruning within Phase 2.
    let best_screen_obj = ranked.first().map(|&(_, obj)| obj).unwrap_or(root_leaf_obj);
    let phase2_ub_snapshot = best_screen_obj.min(root_leaf_obj);

    let results: Vec<(usize, CandResult)> = if need_deep {
        finalist_indices
            .par_iter()
            .map(|&ci| {
                let r = solve_candidate(
                    &all_candidates[ci],
                    &partitions[ci],
                    full_depth,
                    time_limit,
                    phase2_ub_snapshot,
                    &store,
                    config.local_thresh_scan,
                    config.scan_beam,
                );
                (ci, r)
            })
            .collect()
    } else {
        // Screening was already at full depth — reuse results
        finalist_indices
            .iter()
            .filter_map(|&ci| {
                screen_results
                    .iter()
                    .find(|(sci, _)| *sci == ci)
                    .map(|(sci, r)| (*sci, r.clone()))
            })
            .collect()
    };

    // Find best candidate (compare against leaf option too)
    let mut best_obj = root_leaf_obj;
    let mut best_entry: Option<(usize, &CandResult)> = None;

    for (idx, (_ci, r)) in results.iter().enumerate() {
        if r.0 < best_obj {
            best_obj = r.0;
            best_entry = Some((idx, r));
        }
    }

    let elapsed = start.elapsed().as_secs_f64();
    let total_subproblems: usize = screen_results.iter().map(|(_, r)| r.4).sum::<usize>()
        + if need_deep {
            results.iter().map(|(_, r)| r.4).sum::<usize>()
        } else {
            0
        };
    let total_explored: usize = screen_results.iter().map(|(_, r)| r.5).sum::<usize>()
        + if need_deep {
            results.iter().map(|(_, r)| r.5).sum::<usize>()
        } else {
            0
        };

    // Collect all evaluated root candidates for Phase 2 pre-seeding
    let root_candidates: Vec<RootCandidate> = results
        .iter()
        .map(|(ci, r)| {
            let cand = &all_candidates[*ci];
            let tree = TreeNode::Internal {
                feature_index: cand.feature_index,
                thresholds: cand.thresholds.clone(),
                children: r.6.clone(),
                group_map: cand.group_map.clone(),
            };
            RootCandidate {
                objective: r.0,
                tree,
            }
        })
        .collect();

    // Merge profiles in canonical order — Phase 1 results in
    // `screen_indices` order (already canonical after the pre-ranking),
    // Phase 2 results in `finalist_indices` order (canonical-sorted).
    // When Phase 2 reused Phase 1 results (!need_deep), we avoid double-
    // counting by skipping the Phase 1 merge for those finalists.
    let mut final_profile = SolverProfile::default();
    for (_, r) in screen_results.iter() {
        final_profile.merge_from(&r.7);
    }
    if need_deep {
        for (_, r) in results.iter() {
            final_profile.merge_from(&r.7);
        }
    }

    match best_entry {
        None => {
            // Leaf is best
            let prediction = class_counts
                .iter()
                .enumerate()
                .max_by_key(|(_, &c)| c)
                .map(|(i, _)| i)
                .unwrap_or(0);
            let tree = TreeNode::Leaf {
                prediction,
                n_samples: n_data,
                loss: root_max_loss,
            };
            (
                SolveResult {
                    objective: root_leaf_obj,
                    lowerbound: root_leaf_obj,
                    is_optimal: false,
                    certified_optimal: false,
                    search_converged: false,
                    certification_mode: "exploratory".to_string(),
                    certification_reason:
                        "parallel_root_uses_candidate_screening_and_top_n_selection".to_string(),
                    unsafe_heuristics_used: true,
                    timed_out: false,
                    termination_reason: "screened_best_found".to_string(),
                    final_gap: 0.0,
                    n_subproblems: total_subproblems,
                    n_explored: total_explored,
                    elapsed,
                },
                tree,
                root_candidates,
                final_profile,
            )
        }
        Some((idx, r)) => {
            let (
                ref total_obj,
                ref total_lb,
                ref all_optimal,
                ref any_timed_out,
                _,
                _,
                ref child_trees,
                _,
            ) = *r;
            let ci = results[idx].0;
            let cand = &all_candidates[ci];

            let tree = TreeNode::Internal {
                feature_index: cand.feature_index,
                thresholds: cand.thresholds.clone(),
                children: child_trees.clone(),
                group_map: cand.group_map.clone(),
            };

            let termination_reason = if *all_optimal {
                "screened_converged_uncertified"
            } else if *any_timed_out {
                "timeout"
            } else {
                "screened_best_found"
            };

            (
                SolveResult {
                    objective: *total_obj,
                    lowerbound: *total_lb,
                    is_optimal: false,
                    certified_optimal: false,
                    search_converged: false,
                    certification_mode: "exploratory".to_string(),
                    certification_reason:
                        "parallel_root_uses_candidate_screening_and_top_n_selection".to_string(),
                    unsafe_heuristics_used: true,
                    timed_out: *any_timed_out,
                    termination_reason: termination_reason.to_string(),
                    // `total_lb` is summed from
                    // the children's `lowerbound`, each produced by a
                    // time-limited (1s screen), nested-rayon `BranchAndBoundSolver`
                    // child solve. Those child lower bounds are NON-DETERMINISTIC
                    // under multithreading (an uncertified screen that does not
                    // fully converge reports whatever LB it reached when the work
                    // got stolen/timed out), so `total_obj - total_lb` is a raced
                    // quantity. Downstream (`lib.rs::solve_rashomon`) ADDS this
                    // gap to the Rashomon bound `opt*(1+theta) + gap`; a raced,
                    // spuriously-low LB inflates the bound, which combinatorially
                    // inflates `tree_count`. This screen is explicitly UNCERTIFIED
                    // (`certified_optimal: false`), so it makes no certified
                    // optimality-gap claim: report 0.0. The `lowerbound` field
                    // below is retained (informational) but is not used to widen
                    // the Rashomon bound. With gap=0 the bound is the
                    // deterministic `opt*(1+theta)`, matching the single-threaded
                    // ground truth.
                    final_gap: 0.0,
                    n_subproblems: total_subproblems,
                    n_explored: total_explored,
                    elapsed,
                },
                tree,
                root_candidates,
                final_profile,
            )
        }
    }
}
