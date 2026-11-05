//! The interval lower bound (`RashomonTuning::lb_interval`, default off):
//! the completion lower bound of
//! `rash_completion_min` at remaining depth 2 from an exact interval DP over
//! the store bins of the support instead of the beam-limited lb Select pool.
//!
//! Value. At a support S with remaining depth 2 the result is the minimum of
//! the leaf and, over every feature f and every contiguous partition of the
//! f-bins present in S into 2..=max_k bins, `overhead + Σ_bins D1(bin)`
//! (bin order, left to right as the lb-sum test sums), where D1(bin) is the value the
//! pool path gives a child at remaining depth 1 (leaf when the child has
//! <= 1 row or its floor reaches its leaf, else max(floor, min(leaf, the
//! Depth-1 closed form))). The pool's splits on store thresholds are such
//! partitions (an empty bin only adds leaf_reg), so the value is <= the
//! pool path's (bitwise: equal partitions give equal sums, and the DP's
//! prefix minima are exact under monotone rounding), hence still an
//! admissible lower bound.
//!
//! Method. Per feature f: the rows of S bucketed by f-rank (the nonempty
//! cells 0..m), prefix histograms P[i] (g-rank × class, every g) of the
//! first i cells, and the depth-1 closed form of a cell interval [a, b)
//! from P[b] - P[a] (`rash_d1_sweep_bounds` + `rash_d1_dp_switched`, the
//! swept closed form's code). A DP over cuts (layer t = best left-to-right
//! sum of t bins ending at a cut) is pruned by admissible lower bounds
//! only: per cell φ = min_g misclassification of the cell's finest g-grid,
//! so a bin costs >= Σφ / n_total + leaf_reg (misclassification is
//! superadditive over row partitions; any depth-1 tree's error on the bin
//! is >= its finest g-grid's for its split feature g), the cheap
//! leaf/floor bound, and monotonicity in the row set (D1 of a sub-interval
//! <= D1 of the interval). A prune skips only partitions whose lower bound
//! exceeds min(best, bound) + 1e-9.
//!
//! Ψ bounds (exact). For a row set X and a
//! feature g let fine_g(X) = Σ over g-cells c of (n_c(X) − max_cl
//! n_{c,cl}(X)), Ψ(a, b) = min_g fine_g(cells [a, b)) and Ψ_dy(a, b) =
//! min_g Σ_{B in the dyadic cover of [a, b)} fine_g(B) (`IvDyadic`, built
//! per root feature from the prefix tables).
//! L1: any depth-1 tree on X splitting on g into bins of consecutive
//! g-cells has error >= fine_g(X) (n − max is superadditive).
//! L2: by superadditivity over disjoint row sets, Ψ >= Ψ_dy >= Σ_{i in
//! [a, b)} min_g fine_g(cell i) = φ[b] − φ[a].
//! L3: every partition the closed form evaluates has >= 2 bins, so its
//! value oc + Σ_j (m_j / n_total + leaf_reg) >= oc + 2·leaf_reg + Ψ_dy /
//! n_total (also with Ψ in place of Ψ_dy).
//! Hence the DP's bin bound takes max(Σ φ, Ψ_dy) as its error term, and D1
//! (resp. the closed form) returns the leaf without the closed form (resp.
//! its DP) when oc + 2·leaf_reg + Ψ_dy (resp. Ψ) / n_total >= leaf + 1e-9.
//! Suffix-layer DP (when Ψ_dy is available): the k-way pass
//! also bounds the bins still to place over a suffix [b, m) by the least
//! Σ of their single-bin bounds (a suffix-layer DP over those bounds;
//! each bin's D1 is >= its bound, so the bound is admissible).
//! Float guard (checked per search, else the Ψ paths are off): split_reg,
//! leaf_reg, path_penalty >= 0, n_node <= n_total (misclassification /
//! n_total <= 1), V = split_reg + path_penalty + max_k·(1 + leaf_reg) <
//! 1e3 (and n_total <= 2^32). With j = min(max_k, T + 1) <= 513 bins
//! (T <= `RASH_LB_D1_MAX_T` = 512), every compared sum is within (j + 6)·
//! (ε/2)·V < 1e-9 of its real value, so a shortcut fires only where the
//! computed closed form exceeds the leaf and `leaf.min(closed)` is the
//! leaf bitwise: D1 entries, DP minima, φ totals, memo entries and the
//! enumerated trees are unchanged.

use super::*;
use std::cell::RefCell;

/// Process-wide counters of the interval path: calls answered, calls not
/// applicable (fell back to the pool), wall ns, D1 evaluations.
static IV_CALLS: AtomicU64 = AtomicU64::new(0);
static IV_FALLBACK: AtomicU64 = AtomicU64::new(0);
static IV_NS: AtomicU64 = AtomicU64::new(0);
static IV_D1: AtomicU64 = AtomicU64::new(0);
static IV_NS_BUILD: AtomicU64 = AtomicU64::new(0);
static IV_FEATS: AtomicU64 = AtomicU64::new(0);
static IV_FEATS_PRUNED: AtomicU64 = AtomicU64::new(0);
static IV_ABOVE: AtomicU64 = AtomicU64::new(0);
static IV_D1_K: AtomicU64 = AtomicU64::new(0);
static IV_NS_P: AtomicU64 = AtomicU64::new(0);
static IV_CACHE_HITS: AtomicU64 = AtomicU64::new(0);
static IV_CACHE_ROT: AtomicU64 = AtomicU64::new(0);
static IV_FEATS_CACHED: AtomicU64 = AtomicU64::new(0);
static IV_FEATS_KWAY_PRUNED: AtomicU64 = AtomicU64::new(0);
static IV_FEATS_P_SKIPPED: AtomicU64 = AtomicU64::new(0);
static IV_D1_SKIPPED: AtomicU64 = AtomicU64::new(0);
/// Ψ_dy bound: D1 evaluations answered by the Ψ_dy leaf shortcut / run
/// in full with Ψ_dy available, closed forms answered by the exact-Ψ shortcut, ns building the dyadic
/// tables (every level; level 0 is also the φ pass when the prefix tables
/// come first, so that time is in `time_lb_interval_build_ns` too).
static IV_PSIDY_SHORT: AtomicU64 = AtomicU64::new(0);
static IV_PSIDY_FULL: AtomicU64 = AtomicU64::new(0);
static IV_PSI_EXACT_SHORT: AtomicU64 = AtomicU64::new(0);
static IV_NS_PSIDY_BUILD: AtomicU64 = AtomicU64::new(0);
/// Suffix-layer DP: k-way passes with the suffix-layer tables, ns building them.
static IV_SDP_BUILDS: AtomicU64 = AtomicU64::new(0);
static IV_NS_SDP: AtomicU64 = AtomicU64::new(0);
/// Similar-support skip (`rash_lb_scan`): child calls skipped (`=2`: predicted
/// and checked), reference computations v2(c) made, ns in them.
pub(super) static SS_SKIPS: AtomicU64 = AtomicU64::new(0);
pub(super) static SS_REFS: AtomicU64 = AtomicU64::new(0);
pub(super) static SS_NS_REF: AtomicU64 = AtomicU64::new(0);
/// `rash_lb_scan` at completion depth 3: scans and their wall ns.
pub(super) static SCAN_D3: AtomicU64 = AtomicU64::new(0);
pub(super) static SCAN_D3_NS: AtomicU64 = AtomicU64::new(0);

/// Interval lb counters (result-dict keys, values), cumulative over
/// the process.
pub fn rash_iv_stats() -> Vec<(&'static str, u64)> {
    vec![
        ("n_lb_interval_calls", IV_CALLS.load(Ordering::Relaxed)),
        (
            "n_lb_interval_fallback",
            IV_FALLBACK.load(Ordering::Relaxed),
        ),
        ("time_lb_interval_ns", IV_NS.load(Ordering::Relaxed)),
        ("n_lb_interval_d1", IV_D1.load(Ordering::Relaxed)),
        (
            "time_lb_interval_build_ns",
            IV_NS_BUILD.load(Ordering::Relaxed),
        ),
        ("n_lb_interval_features", IV_FEATS.load(Ordering::Relaxed)),
        (
            "n_lb_interval_features_pruned",
            IV_FEATS_PRUNED.load(Ordering::Relaxed),
        ),
        ("n_lb_interval_above", IV_ABOVE.load(Ordering::Relaxed)),
        ("n_lb_interval_d1_kway", IV_D1_K.load(Ordering::Relaxed)),
        (
            "time_lb_interval_prefix_ns",
            IV_NS_P.load(Ordering::Relaxed),
        ),
        (
            "n_lb_interval_cache_hits",
            IV_CACHE_HITS.load(Ordering::Relaxed),
        ),
        (
            "n_lb_interval_cache_rotations",
            IV_CACHE_ROT.load(Ordering::Relaxed),
        ),
        (
            "n_lb_interval_features_cached_skip",
            IV_FEATS_CACHED.load(Ordering::Relaxed),
        ),
        (
            "n_lb_interval_features_kway_pruned",
            IV_FEATS_KWAY_PRUNED.load(Ordering::Relaxed),
        ),
        (
            "n_lb_interval_features_prefix_skipped",
            IV_FEATS_P_SKIPPED.load(Ordering::Relaxed),
        ),
        (
            "n_lb_interval_d1_skipped",
            IV_D1_SKIPPED.load(Ordering::Relaxed),
        ),
        (
            "n_lb_interval_psidy_short",
            IV_PSIDY_SHORT.load(Ordering::Relaxed),
        ),
        (
            "n_lb_interval_psidy_full",
            IV_PSIDY_FULL.load(Ordering::Relaxed),
        ),
        (
            "n_lb_interval_psi_exact_short",
            IV_PSI_EXACT_SHORT.load(Ordering::Relaxed),
        ),
        (
            "time_lb_interval_psidy_build_ns",
            IV_NS_PSIDY_BUILD.load(Ordering::Relaxed),
        ),
        (
            "n_lb_interval_sdp_builds",
            IV_SDP_BUILDS.load(Ordering::Relaxed),
        ),
        ("time_lb_interval_sdp_ns", IV_NS_SDP.load(Ordering::Relaxed)),
        ("n_lb_simskip_skipped", SS_SKIPS.load(Ordering::Relaxed)),
        ("n_lb_simskip_refs", SS_REFS.load(Ordering::Relaxed)),
        ("time_lb_simskip_ref_ns", SS_NS_REF.load(Ordering::Relaxed)),
        ("n_lb_scan_d3", SCAN_D3.load(Ordering::Relaxed)),
        ("time_lb_scan_d3_ns", SCAN_D3_NS.load(Ordering::Relaxed)),
    ]
}

/// Per-thread scratch of `rash_interval_min2` (reused across calls; large
/// buffers are shrunk back to `IV_SCRATCH_KEEP` elements after a call).
#[derive(Default)]
struct RashIvScratch {
    rows: Vec<u32>,
    /// Per row position i: its histogram cell `hoff[g] + rank_g · nc +
    /// label` for every active feature g (row-major), and its label.
    codes: Vec<u32>,
    lab: Vec<u8>,
    start: Vec<usize>,
    by_rank: Vec<u32>,
    cells: Vec<usize>,
    hoff: Vec<usize>,
    /// One cell's histogram (kept all-zero between cells) and the
    /// per-(g, rank) visit stamps of the φ pass.
    h: Vec<i32>,
    stamp: Vec<u32>,
    sid: u32,
    /// Working (m + 1) prefix histograms of `hlen` cells (g-rank × class).
    p: Vec<i32>,
    /// Per feature: cells, prefix class / row counts, φ prefix, kept prefix
    /// histograms (empty = rebuild) and whether the k-way pass is pending.
    feats: Vec<IvFeat>,
    /// Per feature φ total of this call (cache value; `IV_PHI_*` codes).
    cphi: Vec<u32>,
    /// D1 of cell interval [a, b) at a·(m + 1) + b; NaN = not computed.
    dtab: Vec<f64>,
    /// Layer minima: t·(m + 1) + b = best sum of t bins over cells [0, b).
    pre: Vec<f64>,
    /// Running fraction of features pruned by their φ bound (exponential
    /// average; None before the first): the φ-order cost model's estimate.
    prune_rate: Option<f64>,
    /// Ψ_dy bound: the current feature's dyadic block table and the
    /// per-cell φ it computes (level 0).
    dy: IvDyadic,
    phi_cell: Vec<i64>,
}

#[derive(Default)]
struct IvFeat {
    cls: Vec<i64>,
    cn: Vec<i64>,
    phi: Vec<i64>,
}

/// Ψ_dy bound: dyadic fine_g block table of one root feature's cells
/// (module doc): `blk[(l · m + i) · fa + gi]` = fine_g of cells [i, i +
/// 2^l) for the gi-th active feature g, i a multiple of 2^l, i + 2^l <= m.
/// `levels == 0`: not built (Ψ_dy unavailable).
#[derive(Default)]
struct IvDyadic {
    blk: Vec<i32>,
    acc: Vec<i32>,
    levels: usize,
    m: usize,
    fa: usize,
    /// Suffix-layer DP scratch of the k-way pass (`rash_iv_dp`): single-bin
    /// bounds of every cell interval and the suffix layers.
    ptab: Vec<f64>,
    stab: Vec<f64>,
    /// Fast-table scratch of the k-way pass: column-DP sums ((m + 1)
    /// · fa), per-a cover-chain / query flags and φ errors (m + 1), and
    /// pst[a·w + b] = Ψ_dy(a, b) of every non-leaf ptab entry (i64::MIN
    /// elsewhere).
    col: Vec<i32>,
    need: Vec<bool>,
    qv: Vec<bool>,
    perr: Vec<f64>,
    pst: Vec<i64>,
}

/// fine_g(P_b − P_a) for every active g (in `active` order) into `out`:
/// Σ over g-cells of (rows − majority count). `fast`: the 2-class kernel
/// when nc == 2 (the generic kernel is the reference of
/// `rash_iv_dyadic_debug`).
#[inline]
#[allow(clippy::too_many_arguments)]
fn iv_fine(
    pa: &[i32],
    pb: &[i32],
    hoff: &[usize],
    tlen: &[usize],
    active: &[usize],
    nc: usize,
    fast: bool,
    out: &mut [i32],
) {
    if fast && nc == 2 {
        iv_fine_nc2(pa, pb, hoff, tlen, active, out);
        return;
    }
    iv_fine_generic(pa, pb, hoff, tlen, active, nc, out);
}

/// `iv_fine` with 2 classes: rows − majority = min(n0, n1) per g-cell
/// (pairs of counts, vectorisable).
#[inline]
fn iv_fine_nc2(
    pa: &[i32],
    pb: &[i32],
    hoff: &[usize],
    tlen: &[usize],
    active: &[usize],
    out: &mut [i32],
) {
    for (gi, &g) in active.iter().enumerate() {
        let (o, e) = (hoff[g], hoff[g] + (tlen[g] + 1) * 2);
        let mut sg = 0i32;
        for (ca, cb) in pa[o..e].chunks_exact(2).zip(pb[o..e].chunks_exact(2)) {
            sg += (cb[0] - ca[0]).min(cb[1] - ca[1]);
        }
        out[gi] = sg;
    }
}

/// `iv_fine` for any class count.
#[inline]
fn iv_fine_generic(
    pa: &[i32],
    pb: &[i32],
    hoff: &[usize],
    tlen: &[usize],
    active: &[usize],
    nc: usize,
    out: &mut [i32],
) {
    for (gi, &g) in active.iter().enumerate() {
        let (o, e) = (hoff[g], hoff[g] + (tlen[g] + 1) * nc);
        let mut sg = 0i32;
        for (ca, cb) in pa[o..e].chunks_exact(nc).zip(pb[o..e].chunks_exact(nc)) {
            let (mut n_r, mut mx) = (0i32, 0i32);
            for cl in 0..nc {
                let v = cb[cl] - ca[cl];
                n_r += v;
                mx = mx.max(v);
            }
            sg += n_r - mx;
        }
        out[gi] = sg;
    }
}

impl IvDyadic {
    fn clear(&mut self) {
        self.levels = 0;
    }

    /// Level 0 from the prefix tables `p` ((m + 1) × hlen): blk[i · fa + gi]
    /// = fine_g(cell i); sizes the table for every level (⌊log2 m⌋ + 1) and
    /// writes φ_i = min_g fine_g(cell i) into `phi_cell[i]`.
    #[allow(clippy::too_many_arguments)]
    fn build_level0(
        &mut self,
        p: &[i32],
        hlen: usize,
        m: usize,
        hoff: &[usize],
        tlen: &[usize],
        active: &[usize],
        nc: usize,
        fast: bool,
        phi_cell: &mut [i64],
    ) {
        let fa = active.len();
        let mut levels = 0usize;
        while (1usize << levels) <= m {
            levels += 1;
        }
        self.m = m;
        self.fa = fa;
        self.levels = levels;
        self.blk.clear();
        self.blk.resize(levels * m * fa, 0);
        self.acc.clear();
        self.acc.resize(fa, 0);
        for i in 0..m {
            let row = &mut self.blk[i * fa..(i + 1) * fa];
            iv_fine(
                &p[i * hlen..(i + 1) * hlen],
                &p[(i + 1) * hlen..(i + 2) * hlen],
                hoff,
                tlen,
                active,
                nc,
                fast,
                row,
            );
            phi_cell[i] = row.iter().copied().min().unwrap_or(0) as i64;
        }
    }

    /// Levels 1.. (after `build_level0` on the same tables): blk[(l · m +
    /// i) · fa + gi] = fine_g(P[i + 2^l] − P[i]). O(m · hlen).
    #[allow(clippy::too_many_arguments)]
    fn build_upper(
        &mut self,
        p: &[i32],
        hlen: usize,
        hoff: &[usize],
        tlen: &[usize],
        active: &[usize],
        nc: usize,
        fast: bool,
    ) {
        let (m, fa) = (self.m, self.fa);
        for l in 1..self.levels {
            let bs = 1usize << l;
            let mut i = 0usize;
            while i + bs <= m {
                let row = &mut self.blk[(l * m + i) * fa..(l * m + i + 1) * fa];
                iv_fine(
                    &p[i * hlen..(i + 1) * hlen],
                    &p[(i + bs) * hlen..(i + bs + 1) * hlen],
                    hoff,
                    tlen,
                    active,
                    nc,
                    fast,
                    row,
                );
                i += bs;
            }
        }
    }

    /// Ψ_dy(a, b) = min_g Σ over the greedy dyadic cover of [a, b) (from a,
    /// the largest aligned block that fits) of fine_g(block). Requires
    /// `levels > 0` and a < b <= m.
    #[inline]
    fn query(&mut self, a: usize, b: usize) -> i64 {
        debug_assert!(self.levels > 0 && a < b && b <= self.m);
        let (m, fa, levels) = (self.m, self.fa, self.levels);
        let acc = &mut self.acc[..];
        acc.iter_mut().for_each(|x| *x = 0);
        let mut x = a;
        while x < b {
            let mut l = if x == 0 {
                levels - 1
            } else {
                (x.trailing_zeros() as usize).min(levels - 1)
            };
            while (1usize << l) > b - x {
                l -= 1;
            }
            let row = &self.blk[(l * m + x) * fa..(l * m + x + 1) * fa];
            for (s, &v) in acc.iter_mut().zip(row) {
                *s += v;
            }
            x += 1usize << l;
        }
        acc.iter().copied().min().unwrap_or(0) as i64
    }
}

/// Debug / tests: Ψ_dy(a, b) at a · (m + 1) + b for every a < b <= m (−1
/// elsewhere) from prefix tables `p` ((m + 1) × hlen, histogram cell
/// `hoff[g] + rank_g · nc + label`; `tlen[g]` = T_g, `active` = the g with
/// T_g > 0).
pub fn rash_iv_dyadic_debug(
    p: &[i32],
    hlen: usize,
    m: usize,
    hoff: &[usize],
    tlen: &[usize],
    active: &[usize],
    nc: usize,
) -> Vec<i64> {
    let w = m + 1;
    let mut out = vec![-1i64; w * w];
    if m == 0 {
        return out;
    }
    let mut dy = IvDyadic::default();
    let mut phi_cell = vec![0i64; m];
    // The reference path (generic kernel, `query` for every entry).
    dy.build_level0(p, hlen, m, hoff, tlen, active, nc, false, &mut phi_cell);
    dy.build_upper(p, hlen, hoff, tlen, active, nc, false);
    for a in 0..m {
        for b in (a + 1)..=m {
            out[a * w + b] = dy.query(a, b);
        }
    }
    out
}

/// Buffers above this many elements are shrunk after a call (the
/// thread-local scratch then holds at most a few × 32 MB).
const IV_SCRATCH_KEEP: usize = 1 << 22;
/// `cphi` codes: φ total not computed / feature has < 2 nonempty cells.
const IV_PHI_UNKNOWN: u32 = u32::MAX;
const IV_PHI_NOSPLIT: u32 = u32::MAX - 1;

fn iv_shrink<T>(v: &mut Vec<T>) {
    if v.capacity() > IV_SCRATCH_KEEP {
        v.clear();
        v.shrink_to(IV_SCRATCH_KEEP);
    }
}

/// Interval lb: per-solver cache of the per-feature φ totals (integer
/// root bounds: every split on f costs >= overhead + Φ_f / n_total + 2 ·
/// leaf_reg) of supports whose interval search ended above its bound, so a
/// re-search at a wider budget skips the features whose bound still
/// exceeds it without counting them. Pure facts of (store, support): the
/// values never change. Two generations of at most cap / 2 entries each
/// (`RashomonTuning::lb_interval_cache_max`; 0 = off); the older
/// generation is dropped when the newer fills.
#[derive(Default)]
pub struct RashIvCache {
    cur: RashFxMap<SupportMask, Box<[u32]>>,
    old: RashFxMap<SupportMask, Box<[u32]>>,
}

impl RashIvCache {
    pub(super) fn get(&self, k: &SupportMask) -> Option<&[u32]> {
        self.cur.get(k).or_else(|| self.old.get(k)).map(|v| &v[..])
    }

    fn insert(&mut self, k: SupportMask, v: Box<[u32]>, cap: usize) {
        if cap == 0 {
            return;
        }
        if self.cur.len() >= (cap / 2).max(1) {
            self.old = std::mem::take(&mut self.cur);
            IV_CACHE_ROT.fetch_add(1, Ordering::Relaxed);
        }
        self.cur.insert(k, v);
    }

    /// `lb_threads` merge: insert every entry of `other` this cache
    /// lacks (older generation first; rotation as for plain inserts).
    pub(super) fn merge_from(&mut self, other: Self, cap: usize) {
        for (k, v) in other.old.into_iter().chain(other.cur) {
            if self.get(&k).is_none() {
                self.insert(k, v, cap);
            }
        }
    }

    /// Entries resident (both generations).
    pub fn len(&self) -> usize {
        self.cur.len() + self.old.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

thread_local! {
    static RASH_IV_SCRATCH: RefCell<RashIvScratch> = RefCell::new(RashIvScratch::default());
}

impl BranchAndBoundSolver {
    /// Interval lb: the remaining-depth-2 completion minimum at
    /// `support` (module doc), pruned below `bound` (the minimum is exact
    /// when it is <= bound; above it only "> bound" is certified). `leaf`
    /// and `overhead` as `rash_completion_min` computes them. None when the
    /// path does not apply (no rank table, a feature above
    /// `RASH_LB_D1_MAX_T`, a negative regulariser, or max_k below
    /// `lb_interval_mink`): the caller uses the pool. On the deadline
    /// the best value so far is returned (the caller then answers
    /// min(best, split floor), as for the pool).
    pub(super) fn rash_interval_min2(
        &mut self,
        support: &SupportMask,
        leaf: f64,
        overhead: f64,
        bound: f64,
    ) -> Option<f64> {
        self.rash_interval_min2_gated(support, leaf, overhead, bound, true)
    }

    /// `rash_interval_min2`; `mink_gate` = whether `lb_interval_mink`
    /// applies (off for the debug entry point, which tests every max_k).
    fn rash_interval_min2_gated(
        &mut self,
        support: &SupportMask,
        leaf: f64,
        overhead: f64,
        bound: f64,
        mink_gate: bool,
    ) -> Option<f64> {
        let (split_reg, leaf_reg, path_penalty, max_k) = {
            let c = &self.search.config;
            (
                (1.0 - c.reg_alpha) * c.reg,
                c.reg_alpha * c.reg,
                c.path_penalty,
                c.max_k,
            )
        };
        let ok = split_reg >= 0.0
            && leaf_reg >= 0.0
            && path_penalty >= 0.0
            && max_k >= 2
            && (!mink_gate || max_k >= self.search.config.rash_tuning.lb_interval_mink);
        let rank = match (ok, self.worker.rash_lb_rank.as_ref()) {
            (true, Some(r)) => Arc::clone(r),
            _ => {
                IV_FALLBACK.fetch_add(1, Ordering::Relaxed);
                return None;
            }
        };
        let store = Arc::clone(&self.search.store);
        if store
            .feature_threshold_masks()
            .iter()
            .any(|m| m.len() > Self::RASH_LB_D1_MAX_T)
        {
            IV_FALLBACK.fetch_add(1, Ordering::Relaxed);
            return None;
        }
        let t0 = Instant::now();
        let best = RASH_IV_SCRATCH.with(|cell| {
            let mut sc = cell.borrow_mut();
            self.rash_iv_run(&mut sc, support, leaf, overhead, bound, &store, &rank)
        });
        IV_CALLS.fetch_add(1, Ordering::Relaxed);
        if best > bound {
            IV_ABOVE.fetch_add(1, Ordering::Relaxed);
        }
        IV_NS.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
        Some(best)
    }

    /// Debug / tests: `rash_interval_min2` at `support` without the flag
    /// (gate and rank table built on demand): the remaining-depth-2
    /// completion minimum when it is <= `bound`, None when certified above
    /// it (same contract as `bounded_completion_min` without RASH_TOL), or
    /// None when the path does not apply.
    pub fn rash_interval_completion_min2_debug(
        &mut self,
        support: &SupportMask,
        bound: f64,
    ) -> Option<f64> {
        let gate = self.rash_lb_gate();
        if !gate.1 {
            return None;
        }
        if self.worker.rash_lb_rank.is_none() {
            self.worker.rash_lb_rank = Some(Arc::new(self.rash_lb_rank_build()));
        }
        let (counts, n) = self.search.store.count_classes(support);
        let leaf = self.leaf_objective_from_counts(&counts, n as f64);
        let c = &self.search.config;
        let overhead =
            (1.0 - c.reg_alpha) * c.reg + c.path_penalty * n as f64 / self.search.n_total as f64;
        let v = self.rash_interval_min2_gated(support, leaf, overhead, bound, false)?;
        (v <= bound).then_some(v)
    }

    /// The interval search (module doc), per feature: skip it by its
    /// cached φ total (`RashIvCache`) when that bound exceeds the limit;
    /// else bucket the rows by f-rank, compute φ per cell (before the
    /// prefix tables when the cells are small next to them, else from
    /// them), skip the feature when its φ bound exceeds the limit, else run
    /// the 2-bin DP and then, when its 3-bin bound (overhead + Φ_f /
    /// n_total + 3 · leaf_reg) is within the limit, the >= 3-bin DP on the
    /// same D1 table. Every skip is admissible, so the result is the same
    /// minimum (bitwise) as without them. φ comes first (no prefix tables)
    /// unless the cost model prefers the tables, so a pruned feature
    /// usually never builds them.
    #[allow(clippy::too_many_arguments)]
    fn rash_iv_run(
        &mut self,
        sc: &mut RashIvScratch,
        support: &SupportMask,
        leaf: f64,
        overhead: f64,
        bound: f64,
        store: &DatasetStore,
        rank: &RashLbRank,
    ) -> f64 {
        const EPS: f64 = 1e-9;
        let (nc, max_k, leaf_reg, n_total) = {
            let c = &self.search.config;
            (
                c.n_classes,
                c.max_k,
                c.reg_alpha * c.reg,
                self.search.n_total as f64,
            )
        };
        let twopass = true;
        let masks = store.feature_threshold_masks();
        let labels = store.labels();
        let rk = &rank.rank;
        let nf = masks.len();
        // Ψ_dy bound float guard (module doc): the shortcuts' 1e-9
        // margin covers the rounding of every compared sum.
        debug_assert!(
            support.n_samples as f64 <= n_total,
            "support larger than n_total"
        );
        let psi_ok = {
            let c = &self.search.config;
            let (split_reg, path_penalty) = ((1.0 - c.reg_alpha) * c.reg, c.path_penalty);
            let v = split_reg + path_penalty + max_k as f64 * (1.0 + leaf_reg);
            split_reg >= 0.0
                && leaf_reg >= 0.0
                && path_penalty >= 0.0
                && v < 1e3
                && support.n_samples as f64 <= n_total
                && n_total <= 4294967296.0
        };
        let tlen: smallvec::SmallVec<[usize; 64]> = masks.iter().map(|m| m.len()).collect();
        let RashIvScratch {
            rows,
            codes,
            lab,
            start,
            by_rank,
            cells,
            hoff,
            h,
            stamp,
            sid,
            p,
            feats,
            cphi,
            dtab,
            pre,
            prune_rate,
            dy,
            phi_cell,
        } = sc;
        hoff.clear();
        let mut hlen = 0usize;
        for m in masks.iter() {
            hoff.push(hlen);
            if !m.is_empty() {
                hlen += (m.len() + 1) * nc;
            }
        }
        // Also on an n_classes change at equal hlen (stamp is per (g, rank)).
        if h.len() != hlen || stamp.len() != hlen / nc.max(1) + 1 {
            h.clear();
            h.resize(hlen, 0);
            stamp.clear();
            stamp.resize(hlen / nc.max(1) + 1, 0);
            *sid = 0;
        }
        cphi.clear();
        let cached = self.rash_iv_get(support);
        match cached {
            Some(v) if v.len() == nf => {
                IV_CACHE_HITS.fetch_add(1, Ordering::Relaxed);
                cphi.extend_from_slice(v);
            }
            _ => cphi.resize(nf, IV_PHI_UNKNOWN),
        }
        if feats.len() < nf {
            feats.resize_with(nf, IvFeat::default);
        }
        let active: smallvec::SmallVec<[usize; 64]> =
            (0..nf).filter(|&g| !masks[g].is_empty()).collect();
        let fa = active.len();
        let mut rows_ready = false;
        let mut best = leaf;
        let mut timed_out = false;
        // Rows of the support and their histogram cells (on first use).
        macro_rules! ensure_rows {
            () => {
                if !rows_ready {
                    rows_ready = true;
                    store.fill_active_indices(support, rows);
                    codes.clear();
                    codes.reserve(rows.len() * fa);
                    lab.clear();
                    for &r in rows.iter() {
                        let (r, l) = (r as usize, labels[r as usize]);
                        for &g in active.iter() {
                            codes.push((hoff[g] + rk[g][r] as usize * nc + l as usize) as u32);
                        }
                        lab.push(l);
                    }
                }
            };
        }
        // Rows bucketed by f-rank (positions into `rows` / `codes`) and the
        // nonempty cells.
        macro_rules! bucket {
            ($f:expr) => {{
                let f = $f;
                let t_f = masks[f].len();
                let rkf = &rk[f];
                start.clear();
                start.resize(t_f + 2, 0);
                for &r in rows.iter() {
                    start[rkf[r as usize] as usize + 1] += 1;
                }
                for q in 1..start.len() {
                    start[q] += start[q - 1];
                }
                by_rank.clear();
                by_rank.resize(rows.len(), 0);
                {
                    let mut pos: smallvec::SmallVec<[usize; 64]> = start.iter().copied().collect();
                    for (i, &r) in rows.iter().enumerate() {
                        let q = rkf[r as usize] as usize;
                        by_rank[pos[q]] = i as u32;
                        pos[q] += 1;
                    }
                }
                cells.clear();
                for q in 0..=t_f {
                    if start[q + 1] > start[q] {
                        cells.push(q);
                    }
                }
                cells.len()
            }};
        }
        // Prefix histograms of the bucketed cells into `$dst`.
        macro_rules! build_p {
            ($dst:expr, $m:expr) => {{
                let (dst, m): (&mut Vec<i32>, usize) = ($dst, $m);
                dst.clear();
                dst.resize((m + 1) * hlen, 0);
                for (i, &q) in cells.iter().enumerate() {
                    let (lo, hi) = dst.split_at_mut((i + 1) * hlen);
                    let row = &mut hi[..hlen];
                    row.copy_from_slice(&lo[i * hlen..]);
                    for &ri in &by_rank[start[q]..start[q + 1]] {
                        let ri = ri as usize;
                        for &cd in &codes[ri * fa..(ri + 1) * fa] {
                            row[cd as usize] += 1;
                        }
                    }
                }
            }};
        }
        for f in 0..nf {
            if masks[f].is_empty() {
                continue;
            }
            if self.check_time() {
                timed_out = true;
                break;
            }
            let lim = best.min(bound) + EPS;
            // No split at all can reach the limit.
            if overhead + 2.0 * leaf_reg > lim {
                break;
            }
            match cphi[f] {
                IV_PHI_NOSPLIT => continue,
                IV_PHI_UNKNOWN => {}
                v => {
                    if overhead + v as f64 / n_total + 2.0 * leaf_reg > lim {
                        IV_FEATS_CACHED.fetch_add(1, Ordering::Relaxed);
                        continue;
                    }
                }
            }
            let t_build = Instant::now();
            ensure_rows!();
            let m = bucket!(f);
            if m < 2 {
                cphi[f] = IV_PHI_NOSPLIT;
                continue;
            }
            // Per-cell class counts and φ from each cell's own histogram
            // (`h`, all-zero between cells; no prefix tables): a small cell
            // visits its touched entries (O(n_i · F)), a large one scans the
            // histogram (O(hlen), contiguous).
            let fd = &mut feats[f];
            fd.cls.clear();
            fd.cls.resize((m + 1) * nc, 0);
            fd.cn.clear();
            fd.cn.resize(m + 1, 0);
            fd.phi.clear();
            fd.phi.resize(m + 1, 0);
            let mut acc: smallvec::SmallVec<[i64; 64]> = smallvec::smallvec![0; fa];
            // φ before any prefix table (a pruned feature then never builds
            // it) unless the expected cost is lower from the tables, built
            // first (large cells: the φ pass re-adds every row, the tables
            // add them again when the feature survives). Cost units: one
            // scattered row-code add = 2, one contiguous entry = 1; p = the
            // running prune rate. φ-first: c_first + (1 - p)·c_build;
            // tables first: c_build + c_scan. Small cells: φ first.
            // Ψ_dy bound: a surviving feature scans the tables anyway
            // (level 0 of its dyadic table, which is also its φ), so
            // φ-first costs c_first + (1 - p)·(c_build + c_scan).
            let small = rows.len() * fa * 8 < m * hlen;
            let phi_first = if psi_ok {
                let mut c_first = 0f64;
                for &q in cells.iter() {
                    let nf_i = ((start[q + 1] - start[q]) * fa) as f64;
                    c_first += if nf_i * 8.0 >= hlen as f64 {
                        2.0 * nf_i + 2.0 * hlen as f64
                    } else {
                        6.0 * nf_i
                    };
                }
                let c_build = (m * hlen) as f64 + 2.0 * (rows.len() * fa) as f64;
                let c_scan = 2.0 * (m * hlen) as f64;
                c_first < prune_rate.unwrap_or(0.5) * (c_build + c_scan)
            } else if small {
                true
            } else {
                let mut c_first = 0f64;
                for &q in cells.iter() {
                    let nf_i = ((start[q + 1] - start[q]) * fa) as f64;
                    c_first += if nf_i * 8.0 >= hlen as f64 {
                        2.0 * nf_i + 2.0 * hlen as f64
                    } else {
                        6.0 * nf_i
                    };
                }
                let c_build = (m * hlen) as f64 + 2.0 * (rows.len() * fa) as f64;
                let c_scan = 2.0 * (m * hlen) as f64;
                c_first < prune_rate.unwrap_or(0.5) * c_build + c_scan
            };
            if !phi_first {
                let t_p = Instant::now();
                build_p!(&mut *p, m);
                IV_NS_P.fetch_add(t_p.elapsed().as_nanos() as u64, Ordering::Relaxed);
                if psi_ok {
                    // Ψ_dy bound: φ per cell = level 0 of the dyadic table.
                    let t_d = Instant::now();
                    phi_cell.clear();
                    phi_cell.resize(m, 0);
                    dy.build_level0(p, hlen, m, hoff, &tlen, &active, nc, true, phi_cell);
                    IV_NS_PSIDY_BUILD.fetch_add(t_d.elapsed().as_nanos() as u64, Ordering::Relaxed);
                }
            }
            let fd = &mut feats[f];
            for (i, &q) in cells.iter().enumerate() {
                let rs = &by_rank[start[q]..start[q + 1]];
                for cl in 0..nc {
                    fd.cls[(i + 1) * nc + cl] = fd.cls[i * nc + cl];
                }
                if !phi_first && psi_ok {
                    for &ri in rs {
                        fd.cls[(i + 1) * nc + lab[ri as usize] as usize] += 1;
                    }
                    fd.cn[i + 1] = fd.cn[i] + rs.len() as i64;
                    fd.phi[i + 1] = fd.phi[i] + phi_cell[i];
                    continue;
                }
                if !phi_first {
                    for &ri in rs {
                        fd.cls[(i + 1) * nc + lab[ri as usize] as usize] += 1;
                    }
                    let (pa, pb) = (
                        &p[i * hlen..(i + 1) * hlen],
                        &p[(i + 1) * hlen..(i + 2) * hlen],
                    );
                    let mut best_g = i64::MAX;
                    for &g in active.iter() {
                        let mut sg = 0i64;
                        let (o, e) = (hoff[g], hoff[g] + (masks[g].len() + 1) * nc);
                        for (ca, cb) in pa[o..e].chunks_exact(nc).zip(pb[o..e].chunks_exact(nc)) {
                            let (mut n_r, mut mx) = (0i32, 0i32);
                            for cl in 0..nc {
                                let v = cb[cl] - ca[cl];
                                n_r += v;
                                mx = mx.max(v);
                            }
                            sg += (n_r - mx) as i64;
                        }
                        best_g = best_g.min(sg);
                    }
                    fd.cn[i + 1] = fd.cn[i] + rs.len() as i64;
                    fd.phi[i + 1] = fd.phi[i] + best_g;
                    continue;
                }
                for &ri in rs {
                    let ri = ri as usize;
                    for &cd in &codes[ri * fa..(ri + 1) * fa] {
                        h[cd as usize] += 1;
                    }
                    fd.cls[(i + 1) * nc + lab[ri] as usize] += 1;
                }
                let n_i = rs.len() as i64;
                if rs.len() * fa * 8 >= hlen {
                    // Large cell: scan its histogram, then clear it.
                    let mut best_g = i64::MAX;
                    for &g in active.iter() {
                        let mut sg = 0i64;
                        let blk = &h[hoff[g]..hoff[g] + (masks[g].len() + 1) * nc];
                        for c in blk.chunks_exact(nc) {
                            let (mut n_r, mut mx) = (0i32, 0i32);
                            for &v in c {
                                n_r += v;
                                mx = mx.max(v);
                            }
                            sg += (n_r - mx) as i64;
                        }
                        best_g = best_g.min(sg);
                    }
                    h.fill(0);
                    fd.cn[i + 1] = fd.cn[i] + n_i;
                    fd.phi[i + 1] = fd.phi[i] + best_g;
                    continue;
                }
                // Small cell: visit only its touched (g, rank) entries.
                if *sid == u32::MAX {
                    stamp.iter_mut().for_each(|s| *s = 0);
                    *sid = 0;
                }
                *sid += 1;
                acc.iter_mut().for_each(|a| *a = 0);
                for &ri in rs {
                    let ri = ri as usize;
                    let l = lab[ri] as usize;
                    for (j, &cd) in codes[ri * fa..(ri + 1) * fa].iter().enumerate() {
                        let base = cd as usize - l;
                        let bi = base / nc;
                        if stamp[bi] != *sid {
                            stamp[bi] = *sid;
                            acc[j] += (0..nc).map(|cl| h[base + cl]).max().unwrap_or(0) as i64;
                        }
                    }
                }
                for &ri in rs {
                    let ri = ri as usize;
                    for &cd in &codes[ri * fa..(ri + 1) * fa] {
                        h[cd as usize] = 0;
                    }
                }
                fd.cn[i + 1] = fd.cn[i] + n_i;
                let best_acc = acc.iter().copied().max().unwrap_or(0);
                fd.phi[i + 1] = fd.phi[i] + (n_i - best_acc);
            }
            let phi_m = fd.phi[m];
            cphi[f] = phi_m.min(IV_PHI_NOSPLIT as i64 - 1) as u32;
            IV_NS_BUILD.fetch_add(t_build.elapsed().as_nanos() as u64, Ordering::Relaxed);
            IV_FEATS.fetch_add(1, Ordering::Relaxed);
            let pruned = overhead + phi_m as f64 / n_total + 2.0 * leaf_reg > lim;
            let x = if pruned { 1.0 } else { 0.0 };
            *prune_rate = Some(prune_rate.map_or(x, |r| r + (x - r) / 32.0));
            if pruned {
                IV_FEATS_PRUNED.fetch_add(1, Ordering::Relaxed);
                if phi_first && !small {
                    // The table-first order would have built the tables.
                    IV_FEATS_P_SKIPPED.fetch_add(1, Ordering::Relaxed);
                }
                continue;
            }
            if phi_first {
                let t_p = Instant::now();
                build_p!(&mut *p, m);
                IV_NS_P.fetch_add(t_p.elapsed().as_nanos() as u64, Ordering::Relaxed);
                if psi_ok {
                    // Ψ_dy bound: level 0 of the dyadic table (per-cell φ
                    // from the tables).
                    let t_d = Instant::now();
                    phi_cell.clear();
                    phi_cell.resize(m, 0);
                    dy.build_level0(p, hlen, m, hoff, &tlen, &active, nc, true, phi_cell);
                    IV_NS_PSIDY_BUILD.fetch_add(t_d.elapsed().as_nanos() as u64, Ordering::Relaxed);
                }
            }
            // Ψ_dy bound: the upper levels of this feature's dyadic table
            // (level 0 built with φ above).
            if psi_ok {
                let t_d = Instant::now();
                dy.build_upper(p, hlen, hoff, &tlen, &active, nc, true);
                IV_NS_PSIDY_BUILD.fetch_add(t_d.elapsed().as_nanos() as u64, Ordering::Relaxed);
            } else {
                dy.clear();
            }
            // 2 bins, then (max_k >= 3) 3.. bins on the same D1 table.
            let fd = &feats[f];
            self.rash_iv_dp(
                p, &fd.cls, &fd.cn, &fd.phi, m, hoff, hlen, dtab, pre, overhead, bound, &mut best,
                false, twopass, false, dy,
            );
            if max_k >= 3 && m >= 3 {
                if overhead + phi_m as f64 / n_total + 3.0 * leaf_reg > best.min(bound) + EPS {
                    IV_FEATS_KWAY_PRUNED.fetch_add(1, Ordering::Relaxed);
                    continue;
                }
                self.rash_iv_dp(
                    p, &fd.cls, &fd.cn, &fd.phi, m, hoff, hlen, dtab, pre, overhead, bound,
                    &mut best, true, twopass, true, dy,
                );
            }
        }
        // A search above its bound may be repeated at a wider one.
        let cache_max = self.search.config.rash_tuning.lb_interval_cache_max;
        if best > bound && !timed_out && cache_max > 0 {
            self.worker.rash_iv_cache.insert(
                support.clone(),
                cphi.clone().into_boxed_slice(),
                cache_max,
            );
        }
        iv_shrink(p);
        iv_shrink(dtab);
        iv_shrink(codes);
        iv_shrink(&mut dy.blk);
        iv_shrink(&mut dy.ptab);
        iv_shrink(&mut dy.stab);
        iv_shrink(&mut dy.col);
        iv_shrink(&mut dy.need);
        iv_shrink(&mut dy.qv);
        iv_shrink(&mut dy.perr);
        iv_shrink(&mut dy.pst);
        best
    }

    /// The cut DP of one feature (`m` cells, tables p / cls / cn / phi).
    /// `kway = false`: 2 bins (layer 1, then close). `kway = true`: 3 ..=
    /// min(max_k, m) bins only. Lowers `best` to every partition value
    /// below it; skips only partitions whose admissible lower bound
    /// exceeds min(best, bound) + 1e-9. When the next bin is the last one
    /// (2-bin layer 1, k-way layer jmax - 1) its bound is also the cheap
    /// bound of that bin, lbbin(b, m) (>= the φ suffix).
    #[allow(clippy::too_many_arguments)]
    fn rash_iv_dp(
        &self,
        p: &[i32],
        cls: &[i64],
        cn: &[i64],
        phi: &[i64],
        m: usize,
        hoff: &[usize],
        hlen: usize,
        dtab: &mut Vec<f64>,
        pre: &mut Vec<f64>,
        overhead: f64,
        bound: f64,
        best: &mut f64,
        kway: bool,
        twopass: bool,
        reuse: bool,
        dy: &mut IvDyadic,
    ) {
        const EPS: f64 = 1e-9;
        let (max_k, leaf_reg, n_total) = {
            let c = &self.search.config;
            (c.max_k, c.reg_alpha * c.reg, self.search.n_total as f64)
        };
        let w = m + 1;
        let jmax = if kway { max_k.min(m) } else { 2 };
        if kway && jmax < 3 {
            return;
        }
        // `reuse`: this feature's D1 table from pass A (same tables).
        if !(reuse && dtab.len() == w * w) {
            dtab.clear();
            dtab.resize(w * w, f64::NAN);
        }
        pre.clear();
        pre.resize(jmax * w, f64::INFINITY);
        let phi_m = phi[m];
        // Suffix-layer DP (k-way, Ψ_dy available): ptab[a·w + b] = the cheap
        // single-bin bound of [a, b) (lbbin's first term, Ψ_dy in its error)
        // for every a < b; stab[j·w + b] = min over partitions of [b, m)
        // into exactly j bins of Σ ptab (j = 1..jmax−1; ∞ when infeasible).
        // Each bin's D1 >= its ptab entry, so j more bins over [b, m) cost
        // >= stab[j·w + b]: an admissible suffix bound.
        let use_sdp = kway && dy.levels > 0;
        // ptab's Ψ_dy by the column DP, kept in dy.pst for D1 (`query` for
        // the entries it lacks).
        let tblfast = use_sdp;
        let mut ptab = std::mem::take(&mut dy.ptab);
        let mut stab = std::mem::take(&mut dy.stab);
        if use_sdp {
            let t_s = Instant::now();
            let c = &self.search.config;
            let nc = c.n_classes;
            let (split_reg, path_penalty) = ((1.0 - c.reg_alpha) * c.reg, c.path_penalty);
            ptab.clear();
            ptab.resize(w * w, f64::NAN);
            if tblfast {
                // Fast tables: for each b, the entries needing Ψ_dy(a, b)
                // and the greedy-cover chains from them (a → a + 2^l(a, b),
                // `query`'s level rule); then S[a] = blk(l(a, b), a) +
                // S[a + 2^l] down the chains (S[b] = 0), min over g.
                let (fa, levels) = (dy.fa, dy.levels);
                let mut col = std::mem::take(&mut dy.col);
                let mut need = std::mem::take(&mut dy.need);
                let mut qv = std::mem::take(&mut dy.qv);
                let mut perrv = std::mem::take(&mut dy.perr);
                let mut pst = std::mem::take(&mut dy.pst);
                col.clear();
                col.resize(w * fa, 0);
                need.clear();
                need.resize(w, false);
                qv.clear();
                qv.resize(w, false);
                perrv.clear();
                perrv.resize(w, 0.0);
                pst.clear();
                pst.resize(w * w, i64::MIN);
                for b in 1..=m {
                    let mut any = false;
                    for a in 0..b {
                        need[a] = false;
                        qv[a] = false;
                        let n = (cn[b] - cn[a]) as usize;
                        let mut cc = [0i64; MAX_CLASSES];
                        for cl in 0..nc {
                            cc[cl] = cls[b * nc + cl] - cls[a * nc + cl];
                        }
                        let leaf_c = self.leaf_objective_from_counts(&cc, n as f64);
                        let floor_c =
                            split_reg + 2.0 * leaf_reg + path_penalty * n as f64 / n_total;
                        // The leaf (no Ψ) when the cheap bound is the leaf.
                        if n <= 1 || floor_c >= leaf_c {
                            ptab[a * w + b] = leaf_c;
                            continue;
                        }
                        need[a] = true;
                        qv[a] = true;
                        perrv[a] = (phi[b] - phi[a]) as f64 / n_total;
                        any = true;
                    }
                    if !any {
                        continue;
                    }
                    let lvl = |a: usize| -> usize {
                        let mut l = if a == 0 {
                            levels - 1
                        } else {
                            (a.trailing_zeros() as usize).min(levels - 1)
                        };
                        while (1usize << l) > b - a {
                            l -= 1;
                        }
                        l
                    };
                    for a in 0..b {
                        if need[a] {
                            let nx = a + (1usize << lvl(a));
                            if nx < b {
                                need[nx] = true;
                            }
                        }
                    }
                    for a in (0..b).rev() {
                        if !need[a] {
                            continue;
                        }
                        let l = lvl(a);
                        let nx = a + (1usize << l);
                        let row = &dy.blk[(l * m + a) * fa..(l * m + a + 1) * fa];
                        let mut mn = i32::MAX;
                        if nx >= b {
                            col[a * fa..(a + 1) * fa].copy_from_slice(row);
                            if qv[a] {
                                mn = row.iter().copied().min().unwrap_or(0);
                            }
                        } else {
                            let (lo, hi) = col.split_at_mut(nx * fa);
                            for ((d, &s1), &r) in
                                lo[a * fa..(a + 1) * fa].iter_mut().zip(&hi[..fa]).zip(row)
                            {
                                *d = s1 + r;
                                mn = mn.min(*d);
                            }
                        }
                        if qv[a] {
                            let psi = if fa == 0 { 0 } else { mn as i64 };
                            let mut perr = perrv[a];
                            let ps = psi as f64 / n_total;
                            if ps > perr {
                                perr = ps;
                            }
                            ptab[a * w + b] = self.rash_iv_cheap(cls, cn, a, b, perr);
                            pst[a * w + b] = psi;
                        }
                    }
                }
                dy.col = col;
                dy.need = need;
                dy.qv = qv;
                dy.perr = perrv;
                dy.pst = pst;
            } else {
                for a in 0..m {
                    for b in (a + 1)..=m {
                        let n = (cn[b] - cn[a]) as usize;
                        let mut cc = [0i64; MAX_CLASSES];
                        for cl in 0..nc {
                            cc[cl] = cls[b * nc + cl] - cls[a * nc + cl];
                        }
                        let leaf_c = self.leaf_objective_from_counts(&cc, n as f64);
                        let floor_c =
                            split_reg + 2.0 * leaf_reg + path_penalty * n as f64 / n_total;
                        // The leaf (no Ψ query) when the cheap bound is the leaf.
                        ptab[a * w + b] = if n <= 1 || floor_c >= leaf_c {
                            leaf_c
                        } else {
                            let mut perr = (phi[b] - phi[a]) as f64 / n_total;
                            let ps = dy.query(a, b) as f64 / n_total;
                            if ps > perr {
                                perr = ps;
                            }
                            self.rash_iv_cheap(cls, cn, a, b, perr)
                        };
                    }
                }
            }
            stab.clear();
            stab.resize(jmax * w, f64::INFINITY);
            for b in 0..m {
                stab[w + b] = ptab[b * w + m];
            }
            for j in 2..jmax {
                for b in 0..m {
                    let mut v = f64::INFINITY;
                    // First bin [b, a), then j − 1 bins over [a, m).
                    for a in (b + 1)..=m.saturating_sub(j - 1) {
                        let x = ptab[b * w + a] + stab[(j - 1) * w + a];
                        if x < v {
                            v = x;
                        }
                    }
                    stab[j * w + b] = v;
                }
            }
            IV_SDP_BUILDS.fetch_add(1, Ordering::Relaxed);
            IV_NS_SDP.fetch_add(t_s.elapsed().as_nanos() as u64, Ordering::Relaxed);
        }
        // Suffix-layer DP: at least `qmin`, at most `jr` more bins over [b, m).
        let lbsdp = |b: usize, qmin: usize, jr: usize| -> f64 {
            if !use_sdp || b >= m {
                return f64::NEG_INFINITY;
            }
            let mut v = f64::INFINITY;
            for j in qmin.max(1)..=jr.min(jmax - 1) {
                let x = stab[j * w + b];
                if x < v {
                    v = x;
                }
            }
            v
        };
        // D1(A) + D1(B) <= D1(A ∪ B) + max_k · leaf_reg + split_reg (restrict
        // the union's optimal depth-<=1 tree to each part; a part's value is
        // <= that tree's cost there): `pen` (+ slack for rounding; NaN
        // terms of unknown D1 compare false).
        let split_reg = (1.0 - self.search.config.reg_alpha) * self.search.config.reg;
        let pen = max_k as f64 * leaf_reg + split_reg + EPS;
        // At least `q` more bins over cells [b, m).
        let lbsuf = |b: usize, q: usize| -> f64 {
            if b == m {
                0.0
            } else {
                (phi_m - phi[b]) as f64 / n_total + q as f64 * leaf_reg
            }
        };
        macro_rules! lbbin {
            ($a:expr, $b:expr) => {{
                let (a, b) = ($a, $b);
                // Ψ_dy bound: Ψ_dy >= Σ φ is also a bin's error bound
                // (suffix-layer DP: tabulated in ptab).
                let mut lb = if use_sdp {
                    ptab[a * w + b]
                } else {
                    let mut perr = (phi[b] - phi[a]) as f64 / n_total;
                    if dy.levels > 0 {
                        let ps = dy.query(a, b) as f64 / n_total;
                        if ps > perr {
                            perr = ps;
                        }
                    }
                    self.rash_iv_cheap(cls, cn, a, b, perr)
                };
                if a + 1 < b {
                    let s1 = dtab[(a + 1) * w + b];
                    if s1 > lb {
                        lb = s1;
                    }
                    let s2 = dtab[a * w + b - 1];
                    if s2 > lb {
                        lb = s2;
                    }
                    // Near-superadditivity: D1(A ∪ B) >= D1(A) + D1(B) - pen.
                    let c1 = dtab[a * w + a + 1] + s1 - pen;
                    if c1 > lb {
                        lb = c1;
                    }
                    let c2 = s2 + dtab[(b - 1) * w + b] - pen;
                    if c2 > lb {
                        lb = c2;
                    }
                }
                lb
            }};
        }
        macro_rules! d1 {
            ($a:expr, $b:expr) => {{
                let (a, b) = ($a, $b);
                let v = dtab[a * w + b];
                if v.is_nan() {
                    let psi = (dy.levels > 0).then(|| {
                        let t = if tblfast { dy.pst[a * w + b] } else { i64::MIN };
                        if t == i64::MIN {
                            dy.query(a, b)
                        } else {
                            t
                        }
                    });
                    let v = self.rash_iv_d1(
                        &p[a * hlen..(a + 1) * hlen],
                        &p[b * hlen..(b + 1) * hlen],
                        cls,
                        cn,
                        hoff,
                        a,
                        b,
                        twopass,
                        psi,
                    );
                    IV_D1.fetch_add(1, Ordering::Relaxed);
                    dtab[a * w + b] = v;
                    v
                } else {
                    v
                }
            }};
        }
        // Layer 1: the first bin [0, b), b < m (1 more bin for 2-bin
        // partitions, 2 more for >= 3).
        let q1 = if kway { 2 } else { 1 };
        // Exactly one bin [b, m) after a bin ending at b: its bound.
        macro_rules! lblast {
            ($b:expr, $q:expr, $last:expr) => {{
                let (b, q) = ($b, $q);
                let s = lbsuf(b, q);
                if $last {
                    let c = lbbin!(b, m);
                    if c > s {
                        c
                    } else {
                        s
                    }
                } else {
                    s
                }
            }};
        }
        for b in 1..m {
            let lb0 = lbbin!(0, b);
            let lim = best.min(bound) + EPS;
            let mut sq1 = lbsuf(b, q1);
            if use_sdp {
                // 2 ..= jmax − 1 more bins.
                let s = lbsdp(b, 2, jmax - 1);
                if s > sq1 {
                    sq1 = s;
                }
            }
            if overhead + lb0 + sq1 > lim {
                continue;
            }
            if overhead + lb0 + lblast!(b, q1, !kway) > lim {
                IV_D1_SKIPPED.fetch_add(1, Ordering::Relaxed);
                continue;
            }
            pre[w + b] = d1!(0, b);
        }
        for t in 1..jmax {
            // Close with the last bin [a, m): t + 1 bins.
            if !kway || t >= 2 {
                for a in (t..m).rev() {
                    let pa = pre[t * w + a];
                    if !pa.is_finite() || overhead + pa + lbbin!(a, m) > best.min(bound) + EPS {
                        continue;
                    }
                    let v = overhead + (pa + d1!(a, m));
                    if v < *best {
                        *best = v;
                    }
                }
            }
            if t + 1 >= jmax {
                break;
            }
            // Layer t + 1: bins ending at cut b < m (short intervals
            // first, so their D1 bound the longer ones).
            let q = if t + 1 >= 2 { 1 } else { 2 };
            // Layer t + 1 only closes (t + 2 = jmax): one last bin [b, m).
            let last = t + 2 >= jmax;
            for b in (t + 1)..m {
                let mut cur = f64::INFINITY;
                let (mut sq, mut sl) = (lbsuf(b, q), lblast!(b, q, last));
                if use_sdp {
                    // 1 ..= jmax − (t + 1) more bins.
                    let s = lbsdp(b, q, jmax - (t + 1));
                    if s > sq {
                        sq = s;
                    }
                    if sq > sl {
                        sl = sq;
                    }
                }
                for a in (t..b).rev() {
                    let pa = pre[t * w + a];
                    if !pa.is_finite() {
                        continue;
                    }
                    let lim = best.min(bound) + EPS;
                    let lab_ = overhead + pa + lbbin!(a, b);
                    if lab_ + sq > lim {
                        continue;
                    }
                    if lab_ + sl > lim {
                        IV_D1_SKIPPED.fetch_add(1, Ordering::Relaxed);
                        continue;
                    }
                    IV_D1_K.fetch_add(1, Ordering::Relaxed);
                    let v = pa + d1!(a, b);
                    if v < cur {
                        cur = v;
                    }
                }
                pre[(t + 1) * w + b] = cur;
            }
        }
        dy.ptab = ptab;
        dy.stab = stab;
    }
    /// Cheap admissible bound on D1 of cell interval [a, b) (`phi_err` =
    /// Σ φ of its cells / n_total): the leaf when the pool path prices the
    /// child as a leaf, else max(floor, min(leaf, split overhead + 2·leaf_reg
    /// + phi_err)) (every depth-1 split's error is >= Σ φ).
    #[inline]
    fn rash_iv_cheap(&self, cls: &[i64], cn: &[i64], a: usize, b: usize, phi_err: f64) -> f64 {
        let c = &self.search.config;
        let nc = c.n_classes;
        let (split_reg, leaf_reg) = ((1.0 - c.reg_alpha) * c.reg, c.reg_alpha * c.reg);
        let n = (cn[b] - cn[a]) as usize;
        let mut cc = [0i64; MAX_CLASSES];
        for cl in 0..nc {
            cc[cl] = cls[b * nc + cl] - cls[a * nc + cl];
        }
        let leaf_c = self.leaf_objective_from_counts(&cc, n as f64);
        let floor_c =
            split_reg + 2.0 * leaf_reg + c.path_penalty * n as f64 / self.search.n_total as f64;
        if n <= 1 || floor_c >= leaf_c {
            return leaf_c;
        }
        let split_lb = split_reg
            + c.path_penalty * n as f64 / self.search.n_total as f64
            + 2.0 * leaf_reg
            + phi_err;
        floor_c.max(leaf_c.min(split_lb))
    }

    /// D1 of cell interval [a, b): the value `rash_lb_scan` gives this child
    /// (rows of P[b] - P[a]) at remaining depth 1 — leaf when it has <= 1
    /// row or its floor reaches its leaf, else max(floor, min(leaf, closed
    /// form)), the closed form skipped when its split floor reaches the
    /// leaf (as `rash_completion_min`).
    #[allow(clippy::too_many_arguments)]
    fn rash_iv_d1(
        &self,
        pa: &[i32],
        pb: &[i32],
        cls: &[i64],
        cn: &[i64],
        hoff: &[usize],
        a: usize,
        b: usize,
        twopass: bool,
        psi: Option<i64>,
    ) -> f64 {
        let c = &self.search.config;
        let nc = c.n_classes;
        let (split_reg, leaf_reg, path_penalty) = (
            (1.0 - c.reg_alpha) * c.reg,
            c.reg_alpha * c.reg,
            c.path_penalty,
        );
        let n_total = self.search.n_total as f64;
        let n = (cn[b] - cn[a]) as usize;
        let mut cc = [0i64; MAX_CLASSES];
        for cl in 0..nc {
            cc[cl] = cls[b * nc + cl] - cls[a * nc + cl];
        }
        let leaf_c = self.leaf_objective_from_counts(&cc, n as f64);
        let floor_c = split_reg + 2.0 * leaf_reg + path_penalty * n as f64 / n_total;
        if n <= 1 || floor_c >= leaf_c {
            return leaf_c;
        }
        let oc = split_reg + path_penalty * n as f64 / n_total;
        let m = if oc + 2.0 * leaf_reg >= leaf_c {
            leaf_c
        } else if psi.is_some_and(|ps| oc + 2.0 * leaf_reg + ps as f64 / n_total >= leaf_c + 1e-9) {
            // Ψ_dy bound: closed >= oc + 2·leaf_reg + Ψ_dy / n_total
            // (L1–L3), so min(leaf, closed) is the leaf bitwise.
            IV_PSIDY_SHORT.fetch_add(1, Ordering::Relaxed);
            leaf_c
        } else {
            if psi.is_some() {
                IV_PSIDY_FULL.fetch_add(1, Ordering::Relaxed);
            }
            let closed = self.rash_iv_closed(
                pa,
                pb,
                hoff,
                n,
                oc,
                twopass,
                leaf_c,
                leaf_reg,
                psi.is_some(),
            );
            leaf_c.min(closed)
        };
        if m > floor_c {
            m
        } else {
            floor_c
        }
    }

    /// The depth-1 closed form of the rows with histogram pb - pa
    /// (`rash_d1_sweep_child`'s counting + DP, same integers and value).
    /// `psi_on` (Ψ_dy bound, the float guard holds): `leaf_c` without
    /// the DP when oc + 2·leaf_reg + Ψ / n_total >= leaf_c + 1e-9 (Ψ =
    /// the exact finest-grid minimum of these rows; module doc L3), which
    /// the caller's leaf_c.min(·) turns into the same value bitwise.
    #[allow(clippy::too_many_arguments)]
    fn rash_iv_closed(
        &self,
        pa: &[i32],
        pb: &[i32],
        hoff: &[usize],
        n_node: usize,
        overhead: f64,
        twopass: bool,
        leaf_c: f64,
        leaf_reg: f64,
        psi_on: bool,
    ) -> f64 {
        let masks = self.search.store.feature_threshold_masks();
        let nc = self.search.config.n_classes;
        let n_total = self.search.n_total as f64;
        RASH_D1_SCRATCH.with(|cell| {
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
            let psi = if intdp && nc == 2 {
                Self::rash_d1_sweep_bounds::<2>(masks, hoff, pa, pb, n_node, cum, cum_n, off)
            } else if intdp && nc == 3 {
                Self::rash_d1_sweep_bounds::<3>(masks, hoff, pa, pb, n_node, cum, cum_n, off)
            } else {
                let mut psi = i64::MAX;
                for (g, m) in masks.iter().enumerate() {
                    off.push(cum_n.len());
                    let t_n = m.len();
                    if t_n == 0 {
                        continue;
                    }
                    cum.extend(std::iter::repeat(0).take(nc));
                    cum_n.push(0);
                    let o = hoff[g];
                    let mut run = [0i64; MAX_CLASSES];
                    let mut run_n = 0i64;
                    let mut fine = 0i64;
                    for r in 0..=t_n {
                        let (mut n_r, mut mx) = (0i64, 0i64);
                        for cl in 0..nc {
                            let h = (pb[o + r * nc + cl] - pa[o + r * nc + cl]) as i64;
                            run[cl] += h;
                            run_n += h;
                            n_r += h;
                            mx = mx.max(h);
                        }
                        fine += n_r - mx;
                        cum.extend_from_slice(&run[..nc]);
                        cum_n.push(run_n);
                    }
                    debug_assert_eq!(run_n, n_node as i64);
                    psi = psi.min(fine);
                }
                psi
            };
            // Ψ_dy bound: exact-Ψ leaf shortcut (module doc).
            if psi_on
                && psi < i64::MAX
                && overhead + 2.0 * leaf_reg + psi as f64 / n_total >= leaf_c + 1e-9
            {
                IV_PSI_EXACT_SHORT.fetch_add(1, Ordering::Relaxed);
                return leaf_c;
            }
            self.rash_d1_dp_switched(cum, cum_n, off, fine3, prev, next, int, overhead, twopass)
        })
    }
}
