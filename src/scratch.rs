use std::sync::Mutex;

use crate::candidates::RushTopB;
use crate::types::{CostEntry, MemoKey, SplitCandidate, SupportMask};

/// Scratch arena for `convolve_cost_schedules`.
///
/// Pools the three hot-loop allocations in
/// `rashomon_dag::convolve_cost_schedules_in`:
///   - `partial`: running schedule across the children fold
///   - `next`: per-step accumulator (FxHashMap, replacing the old BTreeMap)
///   - `out`: final `Vec<CostEntry>` sorted ascending by cost
///
/// All buffers are `clear()`ed on reset; their backing capacity persists
/// across `solve_node` calls so the hot loop hits zero allocations in
/// steady state. One instance is used per (single-threaded) driver.
/// The inner aggregator uses `(i32, u128)` instead
/// of `(f64, u128)`. The cost is stored quantized as an i32
/// (cost_precision-scaled) so the running sum is integer-exact. Tree
/// counts are u128 with saturating arithmetic (`count_add` / `count_mul`):
/// they were u64 (a single `mulx`) until large multiway sets showed the
/// u64 fold silently clamping above 1.8e19 trees. The final `out` Vec
/// de-quantizes to f64 via `CostEntry::new` at emission time.
#[derive(Default)]
pub struct ConvolveScratch {
    pub partial: Vec<(i32, u128)>,
    pub next: rustc_hash::FxHashMap<i32, u128>,
    pub out: Vec<CostEntry>,
    /// Pre-quantized cache of the current child schedule's (cost_q, count)
    /// pairs — populated once per fold step so the inner loop avoids the
    /// per-pair `cost_key_with_precision` float→int call.
    pub r_quantized: Vec<(i32, u128)>,
}

impl ConvolveScratch {
    pub fn reset(&mut self) {
        self.partial.clear();
        self.next.clear();
        self.out.clear();
        self.r_quantized.clear();
    }
}

#[derive(Default)]
pub struct WorkerScratch {
    pub feature_values: Vec<f64>,
    pub labels: Vec<u8>,
    pub weights: Vec<f64>,
    pub order: Vec<usize>,
    pub thresholds: Vec<f64>,
    pub bin_n: Vec<i64>,
    pub bin_c0: Vec<i64>,
    pub bin_counts_flat: Vec<i64>,
    /// Per-bin sum of reference-ensemble class-0 probability mass, populated only
    /// when `SolverConfig::teacher_prob_c0` is present. Same length as
    /// `bin_n` / `bin_c0`.
    pub bin_teacher_c0: Vec<f64>,
    pub group_map: Vec<usize>,
    pub selected_thresholds: Vec<f64>,
    pub tmp_candidates: Vec<SplitCandidate>,
    /// Pre-allocated streaming buffer, sized once to
    /// `2 × n_features × max_k` so candidate emission via
    /// `for_each_candidate_for_support`'s callback can land in this
    /// buffer without a per-call Vec allocation. Mirrors TreeFARMS'
    /// `LocalState::neighbourhood`. Callers `clear()` before reuse;
    /// capacity persists across explore_node calls. The full
    /// closure-based migration (eliminating the OUTER Vec allocation
    /// from generate_candidates_for_support's pipeline tail) is not done;
    /// this buffer is the reuse point for it.
    pub neighbourhood: Vec<SplitCandidate>,
    /// Phase A→B handoff buffer. Survivors of Phase A's screen
    /// land here before Phase B's recursive child evaluation. Decoupled
    /// from the candidate input Vec lifetime so
    /// recursive child explore_node calls can rent the same pool.
    pub surviving_records: Vec<(f64, f64, crate::types::SplitRecord)>,
    // Pre-allocated Rush DP buffers — reused across calls to avoid
    // ~75KB allocation per feature × per closure node.
    pub rush_proj_to_orig: Vec<usize>,
    pub rush_proj_n: Vec<i64>,
    pub rush_proj_c0: Vec<i64>,
    pub rush_s: Vec<i64>,
    pub rush_d: Vec<i64>,
    pub rush_epb_suf: Vec<f64>,
    pub rush_dp: Vec<RushTopB>,
}

impl WorkerScratch {
    pub fn clear_hot_buffers(&mut self) {
        self.feature_values.clear();
        self.labels.clear();
        self.weights.clear();
        self.order.clear();
        self.thresholds.clear();
        self.bin_n.clear();
        self.bin_c0.clear();
        self.bin_counts_flat.clear();
        self.bin_teacher_c0.clear();
        self.group_map.clear();
        self.selected_thresholds.clear();
        self.tmp_candidates.clear();
        self.neighbourhood.clear();
        self.surviving_records.clear();
        // rush_* buffers are NOT cleared here — they're resized per-call
        // via rush_dp_binary_reuse, which reuses capacity without reallocating.
    }
}

pub struct ScratchBank {
    per_thread: Vec<Mutex<WorkerScratch>>,
}

impl Default for ScratchBank {
    fn default() -> Self {
        Self::new()
    }
}

impl ScratchBank {
    pub fn new() -> Self {
        let n_threads = rayon::current_num_threads().max(1);
        let mut per_thread = Vec::with_capacity(n_threads);
        for _ in 0..n_threads {
            per_thread.push(Mutex::new(WorkerScratch::default()));
        }
        ScratchBank { per_thread }
    }

    /// A bank of `n` slots (>= 1), independent of the current rayon pool
    /// (`lb_threads` pass-1 workers own one slot each; `with_worker`
    /// clamps the thread index to it).
    pub fn with_threads(n: usize) -> Self {
        let per_thread = (0..n.max(1))
            .map(|_| Mutex::new(WorkerScratch::default()))
            .collect();
        ScratchBank { per_thread }
    }

    pub fn with_worker<R>(&self, f: impl FnOnce(&mut WorkerScratch) -> R) -> R {
        let idx = rayon::current_thread_index()
            .unwrap_or(0)
            .min(self.per_thread.len().saturating_sub(1));
        let mut guard = self.per_thread[idx].lock().unwrap();
        f(&mut guard)
    }
}

#[derive(Default)]
pub struct SolverScratch {
    pub child_order: Vec<usize>,
    pub child_keys: Vec<MemoKey>,
    pub child_lbs: Vec<f64>,
    pub child_supports: Vec<SupportMask>,
}
