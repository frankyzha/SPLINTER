//! `RashomonTuning::lb_threads` (default 1 = off): intra-cell parallel
//! completion-lb search of the Step-A pass, byte-identical at any thread
//! count.
//!
//! Pass 1 (`rash_prefetch_lbs`, before a node's candidate loop): replay the
//! loop's arity bound / identity dedup / Phase A / materialisation read-only and
//! run one `bounded_completion_min_keyed` per (survivor, child) at the
//! untightened budget `s − overhead − Σ_{j≠i} lb_j^(1)` (lb^(1) = the
//! child's memo lb, else its trivial count lb, <= every lb pass 2 uses) on a
//! dedicated pool. Each worker reads a frozen snapshot of the main lb
//! caches (`RashLbBase`) and writes its own delta; the deltas are merged
//! into the main caches in worker order. Pass 2 = the unchanged loop.
//!
//! Why the output is identical: every lb-cache entry is a fact about
//! (store, support[, depth]) (`Min(F)` = the exact minimum, `Above(b)` =
//! every completion costs more than b; never written on a time-out), so a
//! hit answers exactly what the uncached search would, whatever bound
//! created the entry; pass 1 only adds facts (budgets b1 >= pass 2's b2,
//! so its `Above(b1)` answers pass 2's query). Pass 1 never touches
//! `search.memo`, `parent_map`, the levels or `search.timed_out` (its
//! `check_time` stores into the pass's abort flag); the main thread alone
//! decides the solve's time-out. Pools are Select outputs, independent of
//! threading (`sequential_candidate_gen` gives the same output).
//! Schedule-dependent: the lb counters (Selects, memo hits, evictions,
//! interval counters, lb times as CPU sums), never the trees.

use super::*;

/// The main solver's completion-lb caches, frozen for one pass 1 (moved out
/// with `std::mem::take`, moved back before the merge).
pub struct RashLbBase {
    pub memo: RashFxMap<(SupportMask, i32), RashLbEntry>,
    pub d1s: RashD1Struct,
    pub select: RashFxMap<SupportMask, Arc<[RashLbSplit]>>,
    pub shared: RashFxMap<SupportMask, RashSharedPool>,
    pub iv: interval_lb::RashIvCache,
}

/// `stepb_batches`: the carried completion-lb caches of one Step-B
/// level (`RashLbCache`, moved in once), frozen and shared read-only by the
/// level's batch solvers (`attach_rash_level_base`); each solver's own
/// caches are its delta, merged back at the level end
/// (`rash_lb_delta_merge`, batch order). The Interval-DP cache is empty.
pub struct RashLevelBase {
    pub key: Option<(usize, u64, usize, u64, u64, u64, usize, usize, usize)>,
    pub select_fp: u64,
    pub base: RashLbBase,
    /// Split records held by `base.select` and `base.shared`.
    records: u64,
}

impl RashLevelBase {
    /// The level base of `cache`; None when it holds no caches (no key).
    pub fn from_cache(cache: RashLbCache) -> Option<Self> {
        cache.key?;
        let records = cache.select.values().map(|c| c.len() as u64).sum::<u64>()
            + cache.shared.values().map(|p| p.n_lb as u64).sum::<u64>();
        Some(RashLevelBase {
            key: cache.key,
            select_fp: cache.select_fp,
            base: RashLbBase {
                memo: cache.memo,
                d1s: cache.d1s,
                select: cache.select,
                shared: cache.shared,
                iv: Default::default(),
            },
            records,
        })
    }

    /// Back to the carried cache (`from_cache` inverse).
    pub fn into_cache(self) -> RashLbCache {
        RashLbCache {
            key: self.key,
            memo: self.base.memo,
            select: self.base.select,
            shared: self.base.shared,
            select_fp: self.select_fp,
            d1s: self.base.d1s,
        }
    }

    /// Parent ids of the base (a delta's own ids start here).
    pub fn pid_len(&self) -> u32 {
        self.base.d1s.pids.len() as u32
    }
}

/// `stepb_batches`: merge `src` (`take_rash_lb_cache` of a batch
/// solver: its delta over the level base `dst` when `base_len` = the
/// base's parent ids, `RashLevelBase::pid_len`, else 0 = a full cache) into
/// `dst`, by the `rash_par_merge` rules: lb facts joined
/// (`rash_lb_join`; written only when the join changes the entry), the
/// delta's own structural parent ids (>= `base_len`) remapped to `dst`'s
/// by support, inserts capped as `rash_d1s_put` / `rash_lb_memo_put` (cap
/// divisor 1), Select pools first wins (shared pools only with equal
/// Select fingerprints). Both caches must have the same key, else `src` is
/// dropped. A pure cache: the values a solver reads do not change.
pub fn rash_lb_delta_merge(
    dst: &mut RashLbCache,
    src: RashLbCache,
    base_len: u32,
    tuning: &crate::types::RashomonTuning,
) {
    if src.key.is_none() || src.key != dst.key {
        return;
    }
    // 1. Structural depth-1 memo.
    let mut local: Vec<(u32, SupportMask)> =
        src.d1s.pids.into_iter().map(|(s, id)| (id, s)).collect();
    local.sort_unstable_by_key(|e| e.0);
    let mut remap: RashFxMap<u32, u32> = RashFxMap::default();
    for (id, s) in local {
        debug_assert!(
            id >= base_len,
            "stepb_batches: delta parent id below the base"
        );
        let n = dst.d1s.pids.len() as u32;
        remap.insert(id, *dst.d1s.pids.entry(s).or_insert(n));
    }
    dst.d1s.map.set_cap(tuning.lb_d1s_max_entries);
    for (mut k, e) in src.d1s.map.into_entries() {
        if k.pid >= base_len {
            let Some(&p) = remap.get(&k.pid) else {
                continue;
            };
            k.pid = p;
        }
        let cur = dst.d1s.map.get(&k);
        let j = rash_lb_join(cur, Some(e));
        if j != cur {
            dst.d1s.map.insert(k, j.unwrap());
        }
    }
    // 2. Full-key memo (capped as `rash_lb_memo_put`).
    let cap = tuning.lb_memo_max_entries.max(1);
    let mut n_d1 = dst.memo.keys().filter(|k| k.1 == 1).count();
    for (k, e) in src.memo {
        let cur = dst.memo.get(&k).copied();
        let j = rash_lb_join(cur, Some(e));
        if j == cur {
            continue;
        }
        let d1 = k.1 == 1;
        if dst.memo.insert(k, j.unwrap()).is_some() || !d1 {
            continue;
        }
        n_d1 += 1;
        if n_d1 > cap {
            dst.memo.retain(|k, _| k.1 != 1);
            dst.memo.shrink_to_fit();
            n_d1 = 0;
        }
    }
    // 3. Select pools: the first wins.
    for (sup, c) in src.select {
        dst.select.entry(sup).or_insert(c);
    }
    if src.select_fp == dst.select_fp {
        for (sup, p) in src.shared {
            dst.shared.entry(sup).or_insert(p);
        }
    }
}

/// One pass-1 unit of work: a completion-lb search of `support` at
/// `budget` (structural key `skey` at remaining depth 1).
struct RashLbTask {
    support: SupportMask,
    budget: f64,
    skey: Option<RashD1Key>,
    n: usize,
}

/// Pass-1 pool state of a solver (built on first use, dropped with it).
pub struct RashParState {
    /// T long-lived per-thread states (caches empty between nodes).
    pub workers: Vec<Option<WorkerState>>,
    /// Cooperative deadline flag of the pass (the workers' `rash_time_sink`).
    pub abort: Arc<AtomicBool>,
}

static PAR_RUNS: AtomicU64 = AtomicU64::new(0);
static PAR_TASKS: AtomicU64 = AtomicU64::new(0);

/// `lb_threads` counters, cumulative over the process: (pass-1 runs,
/// pass-1 tasks).
pub fn rash_par_stats() -> (u64, u64) {
    (
        PAR_RUNS.load(Ordering::Relaxed),
        PAR_TASKS.load(Ordering::Relaxed),
    )
}

/// The dedicated pass-1 pool of `t` threads (one per thread count, built
/// once; the global rayon pool is not used).
fn rash_par_thread_pool(t: usize) -> Arc<rayon::ThreadPool> {
    static POOLS: Mutex<Vec<(usize, Arc<rayon::ThreadPool>)>> = Mutex::new(Vec::new());
    let mut g = POOLS.lock().unwrap_or_else(|e| e.into_inner());
    if let Some((_, p)) = g.iter().find(|(n, _)| *n == t) {
        return Arc::clone(p);
    }
    let p = Arc::new(
        rayon::ThreadPoolBuilder::new()
            .num_threads(t)
            .thread_name(|i| format!("rash-lb-{i}"))
            .build()
            .expect("lb_threads: pool"),
    );
    g.push((t, Arc::clone(&p)));
    p
}

/// Join of two facts about one key: `Min` beats `Above` (two `Min`s are the
/// same exact minimum), two `Above`s keep the larger bound.
fn rash_lb_join(a: Option<RashLbEntry>, b: Option<RashLbEntry>) -> Option<RashLbEntry> {
    match (a, b) {
        (Some(RashLbEntry::Min(x)), Some(RashLbEntry::Min(y))) => {
            debug_assert_eq!(x.to_bits(), y.to_bits(), "lb_threads: two Min facts differ");
            Some(RashLbEntry::Min(x))
        }
        (Some(RashLbEntry::Min(m)), _) | (_, Some(RashLbEntry::Min(m))) => {
            Some(RashLbEntry::Min(m))
        }
        (Some(RashLbEntry::Above(x)), Some(RashLbEntry::Above(y))) => {
            Some(RashLbEntry::Above(x.max(y)))
        }
        (a, None) => a,
        (None, b) => b,
    }
}

impl BranchAndBoundSolver {
    // ---- lookups: this solver's caches (the delta on a pass-1 worker or a
    // batch solver), then the frozen pass-1 base (pass-1 workers only), then
    // the `stepb_batches` level base (batch solvers and their pass-1
    // workers only). None layers are skipped.

    /// The read-only layers below this solver's own caches, top first.
    #[inline]
    fn rash_bases(&self) -> impl Iterator<Item = &RashLbBase> {
        self.worker
            .rash_base
            .as_deref()
            .into_iter()
            .chain(self.worker.rash_level_base.as_deref().map(|l| &l.base))
    }

    #[inline]
    pub(super) fn rash_memo_get(&self, k: &(SupportMask, i32)) -> Option<RashLbEntry> {
        let l = self.worker.rash_lb_memo.get(k).copied();
        self.rash_bases()
            .fold(l, |a, b| rash_lb_join(a, b.memo.get(k).copied()))
    }

    #[inline]
    pub(super) fn rash_d1s_get(&self, k: &RashD1Key) -> Option<RashLbEntry> {
        let l = self.worker.rash_lb_d1s.map.get(k);
        self.rash_bases()
            .fold(l, |a, b| rash_lb_join(a, b.d1s.map.get(k)))
    }

    #[inline]
    pub(super) fn rash_shared_get(&self, k: &SupportMask) -> Option<RashSharedPool> {
        match self.worker.rash_select_shared.get(k) {
            Some(p) => Some(p.clone()),
            None => self.rash_bases().find_map(|b| b.shared.get(k).cloned()),
        }
    }

    #[inline]
    pub(super) fn rash_select_get(&self, k: &SupportMask) -> Option<Arc<[RashLbSplit]>> {
        match self.worker.rash_lb_select.get(k) {
            Some(c) => Some(Arc::clone(c)),
            None => self
                .rash_bases()
                .find_map(|b| b.select.get(k).map(Arc::clone)),
        }
    }

    #[inline]
    pub(super) fn rash_select_has(&self, k: &SupportMask) -> bool {
        self.worker.rash_lb_select.contains_key(k)
            || self.rash_bases().any(|b| b.select.contains_key(k))
    }

    #[inline]
    pub(super) fn rash_iv_get(&self, k: &SupportMask) -> Option<&[u32]> {
        match self.worker.rash_iv_cache.get(k) {
            Some(v) => Some(v),
            None => self.rash_bases().find_map(|b| b.iv.get(k)),
        }
    }

    /// Lb threads of this solver's pass 1: `RashomonTuning::lb_threads`
    /// unless set per solver (`WorkerState::rash_lb_threads`, `stepb_batches`).
    pub(super) fn rash_lb_threads_eff(&self) -> usize {
        match self.worker.rash_lb_threads {
            0 => self.search.config.rash_tuning.lb_threads.max(1),
            t => t,
        }
    }

    /// `stepb_batches`: install `level` as this solver's level base (read
    /// after its own caches, never written; its own caches become the
    /// delta, `take_rash_lb_cache`). Only when the key matches (same store,
    /// Select ub and costs, as `seed_rash_lb_cache`) and, if the base holds
    /// Shared Select pools, the Select fingerprint too; otherwise nothing is
    /// installed (a pure cache: same values). Call after
    /// `set_rashomon_select_ub`, on a solver with empty lb caches. Returns
    /// whether it was installed.
    pub fn attach_rash_level_base(&mut self, level: &Arc<RashLevelBase>) -> bool {
        if level.key.is_none() || level.key != Some(self.rash_lb_cache_key()) {
            return false;
        }
        if !level.base.shared.is_empty() && level.select_fp != self.rash_select_config_fp() {
            return false;
        }
        self.worker.rash_level_base = Some(Arc::clone(level));
        self.worker.rash_lb_select_records = level.records;
        let p = &mut self.worker.profile;
        p.lb_select_cache_records = p.lb_select_cache_records.max(level.records);
        self.worker.rash_lb_ub_bits = Some(self.rashomon_select_ub().to_bits());
        true
    }

    /// Whether the lb caches already answer `rash_completion_min(support,
    /// depth, bound, skey)` (same lookups, same order).
    fn rash_lb_known(
        &self,
        support: &SupportMask,
        depth: i32,
        bound: f64,
        skey: Option<RashD1Key>,
    ) -> bool {
        let skey = skey.filter(|_| depth == 1);
        if let Some(sk) = skey {
            match self.rash_d1s_get(&sk) {
                Some(RashLbEntry::Min(_)) => return true,
                Some(RashLbEntry::Above(b)) if bound <= b => return true,
                _ => {}
            }
        }
        match self.rash_memo_get(&(support.clone(), depth)) {
            Some(RashLbEntry::Min(_)) => true,
            Some(RashLbEntry::Above(b)) => bound <= b,
            None => false,
        }
    }

    /// `compute_bounds_from_counts`'s lower bound without the heuristic
    /// tighteners and counters: <= the Phase-A lb and <= the lb
    /// `create_subproblem` + `rash_explore_node_state` give a node with
    /// these counts.
    fn rash_trivial_lb(&self, n: usize, class_counts: &[i64], depth: i32) -> f64 {
        let c = &self.search.config;
        let n_total_f = self.search.n_total as f64;
        let leaf_reg = c.reg_alpha * c.reg;
        let split_reg = (1.0 - c.reg_alpha) * c.reg;
        let path_cost = c.path_penalty * n as f64 / n_total_f;
        if n == 0 {
            return leaf_reg;
        }
        let max_count = *class_counts[..c.n_classes].iter().max().unwrap();
        let max_loss = (n as f64 - max_count as f64) / n_total_f;
        let base_obj = max_loss + leaf_reg;
        if max_loss == 0.0 || max_loss < split_reg + leaf_reg + path_cost || n <= 1 {
            return base_obj;
        }
        if depth <= 0 && !(c.rashomon_mode && c.step_b_remaining > 0) {
            return base_obj;
        }
        base_obj.min(split_reg + 2.0 * leaf_reg + path_cost)
    }

    /// `lb_threads` pass 1 of `rash_explore_node` (module doc). Reads
    /// `search.memo`, never writes it; may fill `indices` (the row list
    /// pass 2 would build) and intern parent ids (`rash_d1_child_keys`).
    /// None when fewer than two searches are left to run.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn rash_prefetch_lbs(
        &mut self,
        support: &SupportMask,
        items: &[&dyn ExploreCand],
        idents: &[u32],
        use_ident: bool,
        depth: i32,
        s: f64,
        split_reg: f64,
        path_cost: f64,
        leaf_reg: f64,
        used_feats: &FeatureMask,
        mask_children: bool,
        indices: &mut Vec<u32>,
    ) {
        let tol = Self::RASH_TOL;
        let t = self.rash_lb_threads_eff();
        let completion_depth = (depth - 1) + self.search.config.step_b_remaining.max(0);
        let overhead = split_reg + path_cost;
        let row_g = |k: usize| split_reg + (k as f64) * leaf_reg <= s + tol;
        // No feature exclusion under exact completion lb: child_used = used.
        let child_used = used_feats;
        let (mode, alpha_bits) = (
            Self::mode_flags(&self.search.config),
            Self::teacher_alpha_memo_bits(&self.search.config),
        );
        // Private copies of the loop's dedup state.
        let n_ident = if use_ident {
            idents
                .iter()
                .filter(|&&g| g != u32::MAX)
                .map(|&g| g as usize + 1)
                .max()
                .unwrap_or(0)
        } else {
            0
        };
        let mut ident_seen = vec![false; n_ident];
        let mut seen_ident: HashSet<(usize, Vec<u64>, Option<Vec<usize>>)> = HashSet::new();
        let mut tasks: Vec<RashLbTask> = Vec::new();
        let mut pos: RashFxMap<(SupportMask, Option<RashD1Key>), usize> = RashFxMap::default();
        let mut cbuf = [0i64; MAX_CLASSES];
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
            // Phase A with the trivial count lbs (a superset of survivors).
            let mut child_lb_sum = overhead;
            let mut pruned = false;
            let n_children = cand.x_n_children();
            for ci in 0..n_children {
                child_lb_sum += self.rash_trivial_lb(
                    cand.x_child_n(ci) as usize,
                    cand.x_child_counts(ci, &mut cbuf),
                    depth - 1,
                );
                let remaining = n_children - ci - 1;
                if child_lb_sum + remaining as f64 * leaf_reg > s + tol {
                    pruned = true;
                    break;
                }
            }
            if pruned || child_lb_sum > s + tol {
                continue;
            }
            // Materialisation: the child supports pass 2 builds.
            let mask_supports = if mask_children {
                self.search.store.partition_support_le(
                    support,
                    cand.x_feature(),
                    cand.x_thresholds(),
                    cand.x_group_map(),
                )
            } else {
                None
            };
            let child_supports: Vec<SupportMask> = match mask_supports {
                Some(cs) => cs,
                None => {
                    if indices.is_empty() && support.n_samples > 0 {
                        *indices = self.search.store.support_to_indices(support);
                    }
                    let b = self.partition_into_flat(
                        indices,
                        self.search.store.feature_col(cand.x_feature()),
                        cand.x_thresholds(),
                        cand.x_group_map(),
                    );
                    (0..b.len())
                        .map(|i| self.search.store.support_from_indices(b.bin_slice(i)))
                        .collect()
                }
            };
            let k = child_supports.len();
            // lb^(1): the child's memo lb, else its trivial count lb.
            let lbs: Vec<f64> = child_supports
                .iter()
                .map(|cs| {
                    let key =
                        MemoKey::new(cs.clone(), child_used.clone(), depth - 1, mode, alpha_bits);
                    match self.search.memo.get(&key).map(|c| c.lowerbound) {
                        Some(lb) => lb,
                        None => {
                            let (cc, n) = self.search.store.count_classes(cs);
                            self.rash_trivial_lb(n, &cc, depth - 1)
                        }
                    }
                })
                .collect();
            // lb-sum test (pass 2's lbs are >= these: it rejects at least as much).
            if overhead + lbs.iter().sum::<f64>() > s + tol {
                continue;
            }
            let skeys = match cand.x_group_map() {
                None if completion_depth == 1 => {
                    self.rash_d1_child_keys(support, cand.x_feature(), cand.x_thresholds(), k)
                }
                _ => None,
            };
            for i in 0..k {
                let others: f64 = lbs
                    .iter()
                    .enumerate()
                    .filter(|&(j, _)| j != i)
                    .map(|(_, b)| b)
                    .sum();
                let budget = s - overhead - others;
                let skey = skeys.as_ref().map(|v| v[i]);
                if self.rash_lb_known(&child_supports[i], completion_depth, budget + tol, skey) {
                    continue;
                }
                // One task per (support, key): the widest budget answers the
                // others (`Min` any bound, `Above(b)` every bound <= b).
                match pos.entry((child_supports[i].clone(), skey)) {
                    std::collections::hash_map::Entry::Occupied(e) => {
                        let task = &mut tasks[*e.get()];
                        if budget > task.budget {
                            task.budget = budget;
                        }
                    }
                    std::collections::hash_map::Entry::Vacant(e) => {
                        e.insert(tasks.len());
                        tasks.push(RashLbTask {
                            support: child_supports[i].clone(),
                            budget,
                            skey,
                            n: child_supports[i].n_samples,
                        });
                    }
                }
            }
        }
        if tasks.len() < 2 {
            return;
        }
        // LPT: largest supports first (stable: first-request order).
        tasks.sort_by(|a, b| b.n.cmp(&a.n));
        self.rash_par_run(&tasks, child_used, completion_depth, t);
        PAR_RUNS.fetch_add(1, Ordering::Relaxed);
        PAR_TASKS.fetch_add(tasks.len() as u64, Ordering::Relaxed);
    }

    /// A pass-1 worker state: empty caches, a one-slot scratch bank, the
    /// pass's abort flag as time sink, the sequential-Select config.
    fn rash_par_worker_new(
        t: usize,
        abort: &Arc<AtomicBool>,
        select_config: &Arc<SolverConfig>,
    ) -> WorkerState {
        WorkerState {
            feature_k_hits: HashMap::new(),
            killer_moves: HashMap::new(),
            greedy_score_cache: HashMap::new(),
            scratch_bank: ScratchBank::with_threads(1),
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
            rash_time_sink: Some(Arc::clone(abort)),
            rash_deadline: None,
            rash_select_config: Some(Arc::clone(select_config)),
            rash_par: None,
            rash_lb_cap_div: t,
        }
    }

    /// Run `tasks` on the pass-1 pool (T workers pulling from one counter)
    /// against the frozen caches, then merge the deltas in worker order.
    fn rash_par_run(&mut self, tasks: &[RashLbTask], used: &FeatureMask, depth: i32, t: usize) {
        let gate = self.rash_lb_gate();
        let mut st = match self.worker.rash_par.take() {
            Some(st) if st.workers.len() == t => st,
            _ => {
                let abort = Arc::new(AtomicBool::new(false));
                let mut cfg = self.search.config.clone();
                cfg.sequential_candidate_gen = true;
                let cfg = Arc::new(cfg);
                let workers = (0..t)
                    .map(|_| Some(Self::rash_par_worker_new(t, &abort, &cfg)))
                    .collect();
                Box::new(RashParState { workers, abort })
            }
        };
        st.abort.store(false, Ordering::Release);
        let base = Arc::new(RashLbBase {
            memo: std::mem::take(&mut self.worker.rash_lb_memo),
            d1s: std::mem::take(&mut self.worker.rash_lb_d1s),
            select: std::mem::take(&mut self.worker.rash_lb_select),
            shared: std::mem::take(&mut self.worker.rash_select_shared),
            iv: std::mem::take(&mut self.worker.rash_iv_cache),
        });
        // Pass-1 ids start above the level base's and the frozen base's.
        let level = self.worker.rash_level_base.clone();
        let base_len = level.as_ref().map_or(0, |l| l.pid_len()) + base.d1s.pids.len() as u32;
        let deadline = self.worker.rash_deadline.or_else(|| {
            let limit = self.search.config.time_limit?;
            let start = (*self.search.start_time.lock().unwrap())?;
            Some((start, limit))
        });
        let RashParState { workers, abort } = &mut *st;
        for w in workers.iter_mut().flatten() {
            w.rash_base = Some(Arc::clone(&base));
            w.rash_level_base = level.clone();
            w.rash_lb_gate = Some(gate);
            w.rash_lb_rank = self.worker.rash_lb_rank.clone();
            w.rash_lb_ub_bits = self.worker.rash_lb_ub_bits;
            w.rash_deadline = deadline;
        }
        let pool = rash_par_thread_pool(t);
        let next = AtomicUsize::new(0);
        let search = &self.search;
        let abort: &AtomicBool = &**abort;
        pool.scope(|sc| {
            for slot in workers.iter_mut() {
                let search = Arc::clone(search);
                let next = &next;
                sc.spawn(move |_| {
                    let worker = slot.take().expect("lb_threads: worker state");
                    let mut solver = BranchAndBoundSolver { search, worker };
                    loop {
                        if abort.load(Ordering::Acquire) {
                            break;
                        }
                        let i = next.fetch_add(1, Ordering::Relaxed);
                        let Some(task) = tasks.get(i) else { break };
                        solver.bounded_completion_min_keyed(
                            &task.support,
                            used,
                            depth,
                            task.budget,
                            task.skey,
                        );
                    }
                    *slot = Some(solver.worker);
                });
            }
        });
        for w in workers.iter_mut().flatten() {
            w.rash_base = None;
            w.rash_level_base = None;
        }
        drop(level);
        let base =
            Arc::try_unwrap(base).unwrap_or_else(|_| panic!("lb_threads: base still shared"));
        self.worker.rash_lb_memo = base.memo;
        self.worker.rash_lb_d1s = base.d1s;
        self.worker.rash_lb_select = base.select;
        self.worker.rash_select_shared = base.shared;
        self.worker.rash_iv_cache = base.iv;
        for w in workers.iter_mut() {
            let w = w.as_mut().expect("lb_threads: worker state");
            self.rash_par_merge(w, base_len);
        }
        self.worker.rash_par = Some(st);
    }

    /// Merge one worker's delta into this solver's caches (insert order as
    /// sequential inserts: caps, rotations and record counts as usual) and
    /// empty it for the next node. `base_len` = parent ids of the level
    /// base (`stepb_batches`) plus the frozen base.
    fn rash_par_merge(&mut self, w: &mut WorkerState, base_len: u32) {
        // 1. Structural depth-1 memo: thread-local parent ids -> main ids.
        let d1s = std::mem::take(&mut w.rash_lb_d1s);
        let mut local: Vec<(u32, SupportMask)> =
            d1s.pids.into_iter().map(|(s, id)| (id, s)).collect();
        local.sort_unstable_by_key(|e| e.0);
        let remap: Vec<u32> = local
            .iter()
            .enumerate()
            .map(|(i, (id, s))| {
                debug_assert_eq!(*id, base_len + i as u32);
                self.rash_lb_pid(s)
            })
            .collect();
        for (mut k, e) in d1s.map.into_entries() {
            if k.pid >= base_len {
                k.pid = remap[(k.pid - base_len) as usize];
            }
            let cur = self.worker.rash_lb_d1s.map.get(&k);
            let j = rash_lb_join(cur, Some(e));
            if j != cur {
                self.rash_d1s_put(k, j.unwrap());
            }
        }
        // 2. Full-key memo.
        for (k, e) in std::mem::take(&mut w.rash_lb_memo) {
            let cur = self.worker.rash_lb_memo.get(&k).copied();
            let j = rash_lb_join(cur, Some(e));
            if j != cur {
                self.rash_lb_memo_put(k, j.unwrap());
            }
        }
        w.rash_lb_memo_d1 = 0;
        // 3. Select pools: the first wins (equal pools: same Select).
        for (sup, p) in std::mem::take(&mut w.rash_select_shared) {
            if !self.worker.rash_select_shared.contains_key(&sup) {
                self.rash_shared_insert(&sup, p);
            }
        }
        for (sup, c) in std::mem::take(&mut w.rash_lb_select) {
            if !self.worker.rash_lb_select.contains_key(&sup) {
                self.rash_lb_select_insert(&sup, c);
            }
        }
        w.rash_lb_select_records = 0;
        // 4. Interval-DP φ cache.
        let cache_max = self.search.config.rash_tuning.lb_interval_cache_max;
        self.worker
            .rash_iv_cache
            .merge_from(std::mem::take(&mut w.rash_iv_cache), cache_max);
        // 5. Counters.
        self.worker.profile.merge_from(&w.profile);
        w.profile = SolverProfile::default();
    }
}
