use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use ndarray::Array2;
use rayon::prelude::*;

use crate::dataset_store::DatasetStore;
use crate::scratch::{ScratchBank, WorkerScratch};
use crate::types::{
    AtomMergeRule, CandGenStats, CandidateOrder, CollapseEntry, DpCriterion, FeatureMask,
    FeatureType, MsdtPortConfig, SolverConfig, SplitCandidate, SupportMask, SupportSig,
    CAND_FAMILY_ALT_MODE_MISCLASS, CAND_FAMILY_ALT_MODE_PRIMARY, CAND_FAMILY_MISCLASS,
    CAND_FAMILY_PRIMARY, MAX_CLASSES,
};

// ---------------------------------------------------------------------------
// All-midpoints threshold generation
// ---------------------------------------------------------------------------

/// Maximum bins used only by the direct Rust no-threshold fallback.
///
/// The public SPLINTER / ShapeSPLINTER wrappers always pass LightGBM reference-ensemble
/// thresholds. This cap exists so lower-level tests and ad hoc Rust callers
/// remain bounded when they intentionally omit a threshold map.
const FALLBACK_MAX_BINS: usize = 500;

/// Score a completed partition independently of the DP that proposed it.
/// Both values include the configured objective penalties; only ranking
/// includes the reference-order adjacency penalty.
pub(crate) fn score_candidate(
    child_n: &[i64],
    child_counts: &[Vec<i64>],
    config: &SolverConfig,
    n_total: usize,
    path_cost: f64,
    order: CandidateOrder,
) -> (f64, f64) {
    let denominator = n_total.max(1) as f64;
    let objective_penalties = (1.0 - config.reg_alpha) * config.reg
        + path_cost
        + child_n.len() as f64 * config.reg_alpha * config.reg;
    let mut rank_cost = objective_penalties;
    let mut split_ub = objective_penalties;
    for (&n, counts) in child_n.iter().zip(child_counts) {
        if n <= 0 {
            continue;
        }
        let n = n as f64;
        let max_count = counts.iter().copied().max().unwrap_or(0) as f64;
        let sum_sq: f64 = counts.iter().map(|&c| (c as f64) * (c as f64)).sum();
        rank_cost += (n - sum_sq / n) / denominator;
        split_ub += (n - max_count) / denominator;
    }
    if let CandidateOrder::Reference { boundary_edges } = order {
        rank_cost += config.msdt.shape_boundary_tax * boundary_edges as f64 / denominator;
    }
    (rank_cost, split_ub)
}

#[inline]
fn reference_order(group_map: &[usize]) -> CandidateOrder {
    CandidateOrder::Reference {
        boundary_edges: group_map
            .windows(2)
            .filter(|edge| edge[0] != edge[1])
            .count(),
    }
}

/// Compute midpoint thresholds between all consecutive unique values of a
/// feature. This gives the DP collapse the full set of possible split points
/// to work with, rather than the greedy subset found by a CART tree.
///
/// If there are more than `FALLBACK_MAX_BINS` unique values, this fallback
/// helper subsamples evenly-spaced cutpoints to keep DP collapse bounded.
pub fn all_midpoints(feature_vals: &[f64]) -> Vec<f64> {
    if feature_vals.is_empty() {
        return vec![];
    }

    // Sort and deduplicate
    let mut unique: Vec<f64> = feature_vals.to_vec();
    unique.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    unique.dedup_by(|a, b| (*a - *b).abs() < 1e-15);

    if unique.len() <= 1 {
        return vec![];
    }

    // Direct Rust fallback only: cap very dense features with evenly-spaced
    // cutpoints so ad hoc no-threshold calls remain bounded.
    if unique.len() > FALLBACK_MAX_BINS + 1 {
        let n = unique.len();
        let step = (n - 1) as f64 / FALLBACK_MAX_BINS as f64;
        let mut sampled = Vec::with_capacity(FALLBACK_MAX_BINS + 1);
        for i in 0..=FALLBACK_MAX_BINS {
            let idx = (i as f64 * step).round() as usize;
            sampled.push(unique[idx.min(n - 1)]);
        }
        sampled.dedup_by(|a, b| (*a - *b).abs() < 1e-15);
        unique = sampled;
    }

    // Midpoints between consecutive unique values
    let mut midpoints = Vec::with_capacity(unique.len() - 1);
    for i in 0..unique.len() - 1 {
        midpoints.push((unique[i] + unique[i + 1]) / 2.0);
    }

    midpoints
}

// ---------------------------------------------------------------------------
// Bin samples by thresholds
// ---------------------------------------------------------------------------

/// Binary search equivalent to np.searchsorted(thresholds, val, side='right').
/// Returns the index of the first threshold strictly greater than val,
/// or thresholds.len() if val >= all thresholds.
#[inline]
pub(crate) fn searchsorted_right(thresholds: &[f64], val: f64) -> usize {
    let mut lo = 0usize;
    let mut hi = thresholds.len();
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        if thresholds[mid] <= val {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    lo
}

/// Assign samples to bins and compute class counts per bin.
/// Equivalent to np.searchsorted(thresholds, vals, side='right') + bincount.
pub fn bin_samples(
    feature_vals: &[f64],
    labels: &[u8],
    thresholds: &[f64],
    n_classes: usize,
    weights: Option<&[f64]>,
    weight_scale: f64,
) -> (Vec<i64>, Vec<Vec<i64>>) {
    let n_bins = thresholds.len() + 1;
    let mut bin_n = vec![0i64; n_bins];
    let mut bin_counts = vec![vec![0i64; n_classes]; n_bins];

    if let Some(w) = weights {
        for (i, &val) in feature_vals.iter().enumerate() {
            let bin = searchsorted_right(thresholds, val);
            let wi = (w[i] * weight_scale).round() as i64;
            bin_n[bin] += wi;
            bin_counts[bin][labels[i] as usize] += wi;
        }
    } else {
        for (i, &val) in feature_vals.iter().enumerate() {
            let bin = searchsorted_right(thresholds, val);
            bin_n[bin] += 1;
            bin_counts[bin][labels[i] as usize] += 1;
        }
    }

    (bin_n, bin_counts)
}

// ---------------------------------------------------------------------------
// Transition point compression
// ---------------------------------------------------------------------------

/// Merge adjacent bins whose class proportions are similar, keeping only
/// thresholds at "transition points" where the distribution actually changes.
///
/// Two triggers force a new group boundary:
///   (a) majority class flips (class-0-dominant → class-1-dominant or vice versa)
///   (b) cumulative group proportion drifts from the next bin's proportion by > delta
///
/// This reduces the number of bins from L to ~15-30 on typical features,
/// making the O(L²K) DP collapse dramatically cheaper.
///
/// Returns (compressed_thresholds, compressed_bin_n, compressed_bin_c0).
/// The compressed vectors have len = (new_n_bins), with new_n_bins ≤ original.
/// If compression results in ≤ 1 bin, returns empty vectors (feature is useless).
pub fn compress_transition_points(
    thresholds: &[f64],
    bin_n: &[i64],
    bin_c0: &[i64],
    delta: f64,
) -> (Vec<f64>, Vec<i64>, Vec<i64>) {
    let n_bins = bin_n.len();
    if n_bins <= 2 {
        // 0 or 1 thresholds — nothing to compress
        return (thresholds.to_vec(), bin_n.to_vec(), bin_c0.to_vec());
    }

    let mut new_thresholds: Vec<f64> = Vec::with_capacity(n_bins);
    let mut new_bin_n: Vec<i64> = Vec::with_capacity(n_bins);
    let mut new_bin_c0: Vec<i64> = Vec::with_capacity(n_bins);

    // Running group accumulators
    let mut run_n: i64 = 0;
    let mut run_c0: i64 = 0;

    for i in 0..n_bins {
        run_n += bin_n[i];
        run_c0 += bin_c0[i];

        // Last bin: always emit the final group
        if i == n_bins - 1 {
            new_bin_n.push(run_n);
            new_bin_c0.push(run_c0);
            break;
        }

        // Check whether to break here (emit threshold between bin i and bin i+1)
        let should_break = if run_n == 0 || bin_n[i + 1] == 0 {
            // Empty run or next bin is empty — don't break on empties
            false
        } else {
            let run_p = run_c0 as f64 / run_n as f64;
            let next_n = bin_n[i + 1];
            let next_c0 = bin_c0[i + 1];
            let next_p = next_c0 as f64 / next_n as f64;

            // (a) Majority class flip
            let run_majority_is_c0 = run_p <= 0.5;
            let next_majority_is_c0 = next_p <= 0.5;
            if run_majority_is_c0 != next_majority_is_c0 {
                true
            } else {
                // (b) Proportion drift exceeds delta
                (run_p - next_p).abs() > delta
            }
        };

        if should_break {
            // Emit the current group
            new_bin_n.push(run_n);
            new_bin_c0.push(run_c0);
            // The threshold between bin i and bin i+1 is thresholds[i]
            new_thresholds.push(thresholds[i]);
            // Reset running group
            run_n = 0;
            run_c0 = 0;
        }
    }

    // If compression left ≤ 1 bin, the feature has no useful split
    if new_thresholds.is_empty() {
        return (vec![], vec![], vec![]);
    }

    (new_thresholds, new_bin_n, new_bin_c0)
}

/// Multiclass variant: merge adjacent bins whose max-class proportions are similar.
pub fn compress_transition_points_multiclass(
    thresholds: &[f64],
    bin_n: &[i64],
    bin_counts: &[Vec<i64>],
    delta: f64,
) -> (Vec<f64>, Vec<i64>, Vec<Vec<i64>>) {
    let n_bins = bin_n.len();
    if n_bins <= 2 {
        return (thresholds.to_vec(), bin_n.to_vec(), bin_counts.to_vec());
    }

    let n_classes = bin_counts[0].len();
    let mut new_thresholds: Vec<f64> = Vec::with_capacity(n_bins);
    let mut new_bin_n: Vec<i64> = Vec::with_capacity(n_bins);
    let mut new_bin_counts: Vec<Vec<i64>> = Vec::with_capacity(n_bins);

    let mut run_n: i64 = 0;
    let mut run_counts = vec![0i64; n_classes];

    for i in 0..n_bins {
        run_n += bin_n[i];
        for c in 0..n_classes {
            run_counts[c] += bin_counts[i][c];
        }

        if i == n_bins - 1 {
            new_bin_n.push(run_n);
            new_bin_counts.push(run_counts.clone());
            break;
        }

        let should_break = if run_n == 0 || bin_n[i + 1] == 0 {
            false
        } else {
            // Majority class comparison
            let run_majority = run_counts
                .iter()
                .enumerate()
                .max_by_key(|(_, &c)| c)
                .map(|(idx, _)| idx)
                .unwrap_or(0);
            let next_majority = bin_counts[i + 1]
                .iter()
                .enumerate()
                .max_by_key(|(_, &c)| c)
                .map(|(idx, _)| idx)
                .unwrap_or(0);

            if run_majority != next_majority {
                true
            } else {
                // Max-class proportion drift
                let run_p = run_counts[run_majority] as f64 / run_n as f64;
                let next_n = bin_n[i + 1];
                let next_p = bin_counts[i + 1][run_majority] as f64 / next_n as f64;
                (run_p - next_p).abs() > delta
            }
        };

        if should_break {
            new_bin_n.push(run_n);
            new_bin_counts.push(run_counts.clone());
            new_thresholds.push(thresholds[i]);
            run_n = 0;
            for c in run_counts.iter_mut() {
                *c = 0;
            }
        }
    }

    if new_thresholds.is_empty() {
        return (vec![], vec![], vec![]);
    }

    (new_thresholds, new_bin_n, new_bin_counts)
}

// ---------------------------------------------------------------------------
// Atomization: merge adjacent bins whose class distributions are
// indistinguishable by the active `AtomMergeRule` into "atoms". Thresholds
// at atom boundaries are preserved so the DP collapse can still emit
// bin-level split points. Bins are merged greedily from left to right.
// ---------------------------------------------------------------------------

/// Wilson score confidence interval for a binomial proportion.
///
/// Returns `(lo, hi)`, both clamped to `[0, 1]`. For `n == 0` returns the
/// widest possible interval `(0.0, 1.0)` so an empty-on-either-side pair
/// trivially overlaps — matching the existing "empty bins absorb" semantics
/// we had under L1.
///
/// Formula (Wilson, 1927):
///     center    = (p + z²/(2n)) / (1 + z²/n)
///     halfwidth = z · √( p(1-p)/n + z²/(4n²) ) / (1 + z²/n)
#[inline]
pub fn wilson_ci(successes: f64, n: f64, z: f64) -> (f64, f64) {
    if n <= 0.0 {
        return (0.0, 1.0);
    }
    let p = successes / n;
    let z2 = z * z;
    let denom = 1.0 + z2 / n;
    let center = (p + z2 / (2.0 * n)) / denom;
    let half = z * (p * (1.0 - p) / n + z2 / (4.0 * n * n)).sqrt() / denom;
    let lo = (center - half).max(0.0);
    let hi = (center + half).min(1.0);
    (lo, hi)
}

/// Do two closed intervals `[a.0, a.1]` and `[b.0, b.1]` share any point?
#[inline]
fn intervals_overlap(a: (f64, f64), b: (f64, f64)) -> bool {
    a.0.max(b.0) <= a.1.min(b.1)
}

/// Binary-class Wilson overlap predicate: merge iff the Wilson CI on the
/// class-0 proportion overlaps between the two bins.
#[inline]
fn bins_should_merge_wilson_binary(
    cur_n: i64,
    cur_c0: i64,
    next_n: i64,
    next_c0: i64,
    z: f64,
) -> bool {
    // For binary data the two class proportions sum to 1, so overlapping
    // on class 0 implies overlapping on class 1. One CI pair suffices.
    let ci_cur = wilson_ci(cur_c0 as f64, cur_n as f64, z);
    let ci_next = wilson_ci(next_c0 as f64, next_n as f64, z);
    intervals_overlap(ci_cur, ci_next)
}

/// Multiclass Wilson overlap predicate: merge iff the Wilson CI on every
/// class's proportion overlaps. Short-circuits on the first class where
/// overlap fails.
#[inline]
fn bins_should_merge_wilson_multiclass(
    cur_n: i64,
    cur_counts: &[i64],
    next_n: i64,
    next_counts: &[i64],
    z: f64,
) -> bool {
    debug_assert_eq!(cur_counts.len(), next_counts.len());
    let cur_nf = cur_n as f64;
    let next_nf = next_n as f64;
    for c in 0..cur_counts.len() {
        let ci_cur = wilson_ci(cur_counts[c] as f64, cur_nf, z);
        let ci_next = wilson_ci(next_counts[c] as f64, next_nf, z);
        if !intervals_overlap(ci_cur, ci_next) {
            return false;
        }
    }
    true
}

/// Merge adjacent bins into atoms using the predicate selected by
/// `policy.atom_merge_rule`. Preserves threshold-level information via
/// `bin_to_atom` so shape-mode/tree-emit downstream can still index in
/// the original bin space.
///
/// `thresholds.len() == bin_n.len() - 1` (L-1 thresholds for L bins).
/// Empty bins participate passively: they attach to the current atom
/// (no class-distribution information to preserve), which keeps
/// `bin_to_atom` dense regardless of rule.
///
/// Returns `(atom_thresholds, atom_n, atom_counts, bin_to_atom)`.
/// `bin_to_atom.len() == bin_n.len()`, with values in `0..atom_n.len()`.
/// If all bins merge into one, `atom_thresholds` is empty.
pub fn merge_adjacent_bins_into_atoms_multiclass(
    thresholds: &[f64],
    bin_n: &[i64],
    bin_counts: &[Vec<i64>],
    policy: &MsdtPortConfig,
) -> (Vec<f64>, Vec<i64>, Vec<Vec<i64>>, Vec<usize>) {
    let n_bins = bin_n.len();
    debug_assert_eq!(thresholds.len() + 1, n_bins);
    debug_assert_eq!(bin_counts.len(), n_bins);
    if n_bins == 0 {
        return (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    }
    let n_classes = bin_counts[0].len();

    let mut atom_thresholds: Vec<f64> = Vec::with_capacity(n_bins);
    let mut atom_n: Vec<i64> = Vec::with_capacity(n_bins);
    let mut atom_counts: Vec<Vec<i64>> = Vec::with_capacity(n_bins);
    let mut bin_to_atom: Vec<usize> = Vec::with_capacity(n_bins);

    atom_n.push(bin_n[0]);
    atom_counts.push(bin_counts[0].clone());
    bin_to_atom.push(0);

    for i in 1..n_bins {
        let cur_atom = atom_n.last().copied().unwrap_or(0);
        let cur_counts = atom_counts.last().unwrap();
        let next_n = bin_n[i];
        let next_counts = &bin_counts[i];

        let should_merge = if cur_atom == 0 || next_n == 0 {
            true
        } else {
            match policy.atom_merge_rule {
                AtomMergeRule::L1Tolerance => {
                    let mut l1 = 0.0f64;
                    let cur_total = cur_atom as f64;
                    let next_total = next_n as f64;
                    for c in 0..n_classes {
                        let p_cur = cur_counts[c] as f64 / cur_total;
                        let p_next = next_counts[c] as f64 / next_total;
                        l1 += (p_cur - p_next).abs();
                    }
                    l1 <= policy.atom_merge_tolerance
                }
                AtomMergeRule::WilsonCI => bins_should_merge_wilson_multiclass(
                    cur_atom,
                    cur_counts,
                    next_n,
                    next_counts,
                    policy.atom_wilson_z,
                ),
            }
        };

        if should_merge {
            let last = atom_n.len() - 1;
            atom_n[last] += next_n;
            for c in 0..n_classes {
                atom_counts[last][c] += next_counts[c];
            }
            bin_to_atom.push(last);
        } else {
            atom_thresholds.push(thresholds[i - 1]);
            atom_n.push(next_n);
            atom_counts.push(next_counts.clone());
            bin_to_atom.push(atom_n.len() - 1);
        }
    }

    (atom_thresholds, atom_n, atom_counts, bin_to_atom)
}

/// Binary classification variant — operates on `bin_c0` (class-0 counts)
/// directly, avoiding Vec<Vec<i64>> allocation. Dispatches on
/// `policy.atom_merge_rule`:
///   - `L1Tolerance`: `|c0_cur/n_cur - c0_next/n_next| * 2 <= tolerance`.
///   - `WilsonCI`: Wilson CIs on class-0 proportion overlap at confidence
///     level derived from `atom_wilson_z`.
///
/// Returns `(atom_thresholds, atom_n, atom_c0, bin_to_atom)`.
pub fn merge_adjacent_bins_into_atoms_binary(
    thresholds: &[f64],
    bin_n: &[i64],
    bin_c0: &[i64],
    policy: &MsdtPortConfig,
) -> (Vec<f64>, Vec<i64>, Vec<i64>, Vec<usize>) {
    let n_bins = bin_n.len();
    debug_assert_eq!(thresholds.len() + 1, n_bins);
    debug_assert_eq!(bin_c0.len(), n_bins);
    if n_bins == 0 {
        return (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    }

    let mut atom_thresholds: Vec<f64> = Vec::with_capacity(n_bins);
    let mut atom_n: Vec<i64> = Vec::with_capacity(n_bins);
    let mut atom_c0: Vec<i64> = Vec::with_capacity(n_bins);
    let mut bin_to_atom: Vec<usize> = Vec::with_capacity(n_bins);

    atom_n.push(bin_n[0]);
    atom_c0.push(bin_c0[0]);
    bin_to_atom.push(0);

    for i in 1..n_bins {
        let cur_n = *atom_n.last().unwrap();
        let cur_c0 = *atom_c0.last().unwrap();
        let next_n = bin_n[i];
        let next_c0 = bin_c0[i];

        let should_merge = if cur_n == 0 || next_n == 0 {
            true
        } else {
            match policy.atom_merge_rule {
                AtomMergeRule::L1Tolerance => {
                    let p_cur = cur_c0 as f64 / cur_n as f64;
                    let p_next = next_c0 as f64 / next_n as f64;
                    // L1 over a 2-class probability vector = 2 * |p_cur - p_next|
                    (p_cur - p_next).abs() * 2.0 <= policy.atom_merge_tolerance
                }
                AtomMergeRule::WilsonCI => bins_should_merge_wilson_binary(
                    cur_n,
                    cur_c0,
                    next_n,
                    next_c0,
                    policy.atom_wilson_z,
                ),
            }
        };

        if should_merge {
            let last = atom_n.len() - 1;
            atom_n[last] += next_n;
            atom_c0[last] += next_c0;
            bin_to_atom.push(last);
        } else {
            atom_thresholds.push(thresholds[i - 1]);
            atom_n.push(next_n);
            atom_c0.push(next_c0);
            bin_to_atom.push(atom_n.len() - 1);
        }
    }

    (atom_thresholds, atom_n, atom_c0, bin_to_atom)
}

/// Block compression over an already-atomized binary sequence.
/// Merges adjacent atoms iff BOTH are "empirically pure" at the given
/// purity threshold (the dominant class's share ≥ `purity`) AND share
/// the same dominant class.
///
/// Returns `(block_thresholds, block_n, block_c0, atom_to_block)`.
/// `atom_to_block.len() == atom_n.len()`, values in `0..block_n.len()`.
pub fn compress_atoms_to_blocks_binary(
    atom_thresholds: &[f64],
    atom_n: &[i64],
    atom_c0: &[i64],
    purity: f64,
) -> (Vec<f64>, Vec<i64>, Vec<i64>, Vec<usize>) {
    let n_atoms = atom_n.len();
    debug_assert_eq!(atom_thresholds.len() + 1, n_atoms);
    debug_assert_eq!(atom_c0.len(), n_atoms);
    if n_atoms == 0 || purity <= 0.0 {
        return (
            atom_thresholds.to_vec(),
            atom_n.to_vec(),
            atom_c0.to_vec(),
            (0..n_atoms).collect(),
        );
    }

    // Returns (is_pure, dominant_class). Dominant class = 0 if c0 ≥ c1,
    // else 1. Empty atoms are considered non-pure and not merged.
    let classify = |n: i64, c0: i64| -> (bool, u8) {
        if n <= 0 {
            return (false, 0);
        }
        let c0f = c0 as f64;
        let c1f = (n - c0) as f64;
        let nf = n as f64;
        let dom_share = c0f.max(c1f) / nf;
        let dom_class: u8 = if c0f >= c1f { 0 } else { 1 };
        (dom_share >= purity, dom_class)
    };

    let mut block_thresholds: Vec<f64> = Vec::with_capacity(n_atoms);
    let mut block_n: Vec<i64> = Vec::with_capacity(n_atoms);
    let mut block_c0: Vec<i64> = Vec::with_capacity(n_atoms);
    let mut atom_to_block: Vec<usize> = Vec::with_capacity(n_atoms);

    block_n.push(atom_n[0]);
    block_c0.push(atom_c0[0]);
    atom_to_block.push(0);

    for i in 1..n_atoms {
        let cur_idx = block_n.len() - 1;
        let (cur_pure, cur_dom) = classify(block_n[cur_idx], block_c0[cur_idx]);
        let (next_pure, next_dom) = classify(atom_n[i], atom_c0[i]);
        let should_merge = cur_pure && next_pure && cur_dom == next_dom;
        if should_merge {
            block_n[cur_idx] += atom_n[i];
            block_c0[cur_idx] += atom_c0[i];
            atom_to_block.push(cur_idx);
        } else {
            block_thresholds.push(atom_thresholds[i - 1]);
            block_n.push(atom_n[i]);
            block_c0.push(atom_c0[i]);
            atom_to_block.push(block_n.len() - 1);
        }
    }
    (block_thresholds, block_n, block_c0, atom_to_block)
}

/// Convenience wrapper: if `config.msdt.atom_mode`, apply
/// `merge_adjacent_bins_into_atoms_binary`; otherwise pass through.
/// Returns `(thresholds, bin_n, bin_c0, bin_to_atom)`. `bin_to_atom` is
/// `Some` only when atomization actually ran AND `shape_mode` is on
/// (shape-mode group_map expansion needs the projection; non-shape paths
/// emit original-space thresholds directly).
pub fn maybe_atomize_binary(
    config: &SolverConfig,
    thresholds: Vec<f64>,
    bin_n: Vec<i64>,
    bin_c0: Vec<i64>,
) -> (Vec<f64>, Vec<i64>, Vec<i64>, Option<(Vec<usize>, Vec<f64>)>) {
    if !config.msdt.atom_mode {
        return (thresholds, bin_n, bin_c0, None);
    }
    let (atom_th, atom_n, atom_c0, bin_to_atom) =
        merge_adjacent_bins_into_atoms_binary(&thresholds, &bin_n, &bin_c0, &config.msdt);
    // Optional block-compression post-pass. Merges adjacent atoms
    // that are both pure and share a dominant class, producing a
    // coarser "block" sequence for the DP to operate on. bin_to_atom
    // is composed with atom_to_block to yield bin_to_block so shape-mode
    // group_map expansion still lands in bin space at the end.
    let (final_th, final_n, final_c0, final_bin_to) =
        if config.msdt.atom_block_compression_purity > 0.0 {
            let (block_th, block_n, block_c0, atom_to_block) = compress_atoms_to_blocks_binary(
                &atom_th,
                &atom_n,
                &atom_c0,
                config.msdt.atom_block_compression_purity,
            );
            let bin_to_block: Vec<usize> = bin_to_atom.iter().map(|&a| atom_to_block[a]).collect();
            (block_th, block_n, block_c0, bin_to_block)
        } else {
            (atom_th, atom_n, atom_c0, bin_to_atom)
        };
    let projection = if config.shape_mode {
        Some((final_bin_to, thresholds))
    } else {
        None
    };
    (final_th, final_n, final_c0, projection)
}

/// Multiclass twin of `maybe_atomize_binary`. Block compression is
/// supported here too — for multiclass, "pure" means the dominant
/// class has share ≥ purity AND both atoms share the same dominant
/// class index.
pub fn maybe_atomize_multiclass(
    config: &SolverConfig,
    thresholds: Vec<f64>,
    bin_n: Vec<i64>,
    bin_counts: Vec<Vec<i64>>,
) -> (
    Vec<f64>,
    Vec<i64>,
    Vec<Vec<i64>>,
    Option<(Vec<usize>, Vec<f64>)>,
) {
    if !config.msdt.atom_mode {
        return (thresholds, bin_n, bin_counts, None);
    }
    let (atom_th, atom_n, atom_counts, bin_to_atom) =
        merge_adjacent_bins_into_atoms_multiclass(&thresholds, &bin_n, &bin_counts, &config.msdt);
    // Multiclass block compression.
    let (final_th, final_n, final_counts, final_bin_to) =
        if config.msdt.atom_block_compression_purity > 0.0 {
            let purity = config.msdt.atom_block_compression_purity;
            let n_atoms = atom_n.len();
            // Classify each atom: (is_pure, dom_class).
            let classify = |n: i64, counts: &[i64]| -> (bool, usize) {
                if n <= 0 {
                    return (false, 0);
                }
                let (dom_class, dom_count) = counts
                    .iter()
                    .copied()
                    .enumerate()
                    .max_by_key(|&(_, c)| c)
                    .unwrap_or((0, 0));
                let share = dom_count as f64 / n as f64;
                (share >= purity, dom_class)
            };
            let mut block_th: Vec<f64> = Vec::with_capacity(n_atoms);
            let mut block_n: Vec<i64> = Vec::with_capacity(n_atoms);
            let mut block_counts: Vec<Vec<i64>> = Vec::with_capacity(n_atoms);
            let mut atom_to_block: Vec<usize> = Vec::with_capacity(n_atoms);
            if n_atoms > 0 {
                block_n.push(atom_n[0]);
                block_counts.push(atom_counts[0].clone());
                atom_to_block.push(0);
                for i in 1..n_atoms {
                    let cur_idx = block_n.len() - 1;
                    let (cur_pure, cur_dom) = classify(block_n[cur_idx], &block_counts[cur_idx]);
                    let (next_pure, next_dom) = classify(atom_n[i], &atom_counts[i]);
                    if cur_pure && next_pure && cur_dom == next_dom {
                        block_n[cur_idx] += atom_n[i];
                        for c in 0..atom_counts[i].len() {
                            block_counts[cur_idx][c] += atom_counts[i][c];
                        }
                        atom_to_block.push(cur_idx);
                    } else {
                        block_th.push(atom_th[i - 1]);
                        block_n.push(atom_n[i]);
                        block_counts.push(atom_counts[i].clone());
                        atom_to_block.push(block_n.len() - 1);
                    }
                }
            }
            let bin_to_block: Vec<usize> = bin_to_atom.iter().map(|&a| atom_to_block[a]).collect();
            (block_th, block_n, block_counts, bin_to_block)
        } else {
            (atom_th, atom_n, atom_counts, bin_to_atom)
        };
    let projection = if config.shape_mode {
        Some((final_bin_to, thresholds))
    } else {
        None
    };
    (final_th, final_n, final_counts, projection)
}

/// Expand atom-indexed shape-mode CollapseEntries back into bin-indexed
/// form. Replaces each entry's `thresholds` (currently all atom boundaries)
/// with the original pre-atom thresholds and rewrites `group_map` from
/// `len == atoms.len()` to `len == bins.len()` via the projection.
///
/// No-op when `projection` is `None`.
pub fn expand_atom_entries_to_bins(
    entries: Vec<(usize, CollapseEntry)>,
    projection: Option<&(Vec<usize>, Vec<f64>)>,
) -> Vec<(usize, CollapseEntry)> {
    let (bin_to_atom, original_thresholds) = match projection {
        Some(p) => p,
        None => return entries,
    };
    entries
        .into_iter()
        .map(|(k, entry)| {
            let new_group_map = entry.group_map.map(|atom_gm| {
                bin_to_atom
                    .iter()
                    .map(|&a| atom_gm.get(a).copied().unwrap_or(0))
                    .collect::<Vec<usize>>()
            });
            (
                k,
                CollapseEntry {
                    cost: entry.cost,
                    thresholds: Arc::from(original_thresholds.as_slice()),
                    bin_n: entry.bin_n,
                    bin_counts: entry.bin_counts,
                    group_map: new_group_map,
                    order: entry.order,
                },
            )
        })
        .collect()
}

/// True if the primary criterion is already Misclass-based — in which case
/// the companion Misclass pass would duplicate work (same DP, same output).
#[inline]
pub fn primary_is_misclass(dp_criterion: DpCriterion) -> bool {
    matches!(
        dp_criterion,
        DpCriterion::Misclass | DpCriterion::Depth2Misclass
    )
}

#[inline]
fn emit_misclassification_companion(dp_criterion: DpCriterion) -> bool {
    !primary_is_misclass(dp_criterion)
}

#[inline]
fn shape_includes_contiguous_baseline(config: &SolverConfig) -> bool {
    config.shape_mode
}

#[inline]
fn candidate_family_count(config: &SolverConfig) -> usize {
    let criteria = if emit_misclassification_companion(config.dp_criterion) {
        2
    } else {
        1
    };
    let modes = if shape_includes_contiguous_baseline(config) {
        2
    } else {
        1
    };
    criteria * modes
}

fn search_capped_beam(config: &SolverConfig, beam_schedule: &[usize]) -> Vec<usize> {
    if config.rashomon_mode || config.bypass_beam {
        return beam_schedule.to_vec();
    }
    let Some(top_k) = config.msdt.top_k else {
        return beam_schedule.to_vec();
    };
    let cap = top_k.max(1);
    beam_schedule.iter().map(|&b| b.max(1).min(cap)).collect()
}

/// Run one local interval-DP pass under the given partition geometry and
/// criterion for binary classification. SPLINTER emits the primary criterion
/// plus a misclassification companion; ShapeSPLINTER also emits the ordinary
/// contiguous geometry so nonlinear partitions compete with, not replace,
/// contiguous ones.
pub fn run_dp_pass_binary(
    use_shape: bool,
    crit: DpCriterion,
    config: &SolverConfig,
    thresholds: &[f64],
    bin_n: &[i64],
    bin_c0: &[i64],
    _n_node: usize,
    n_total: usize,
    eff_beam: &[usize],
    atom_proj: Option<&(Vec<usize>, Vec<f64>)>,
    similar_lb: Option<&HashMap<SupportSig, f64>>,
    atom_teacher_c0: Option<&[f64]>,
    teacher_dp_active: Option<&mut u64>,
) -> Vec<(usize, CollapseEntry)> {
    // Reference-ensemble DP blend fires only on the shape+Gini path, binary-only,
    // when reference-ensemble logits are provided. The Rust side trusts α
    // as passed (the caller sets `teacher_alpha=1.0` when sample weights
    // are active).
    let dp_teacher_active = use_shape
        && matches!(crit, DpCriterion::Gini)
        && config.msdt.teacher_alpha < 1.0 - 1e-12
        && atom_teacher_c0.is_some();
    let dp_teacher_alpha: f64 = if dp_teacher_active {
        config.msdt.teacher_alpha
    } else {
        1.0
    };

    if use_shape {
        // ShapeSPLINTER charges boundary cuts against the global sample
        // denominator so the same local partition has the same nonlinear
        // penalty scale at every node.
        let mu_node = if config.msdt.shape_boundary_tax > 0.0 {
            config.msdt.shape_boundary_tax / (n_total.max(1) as f64)
        } else {
            0.0
        };
        let entries = shape_collapse_topb_binary_with_teacher_smoothed_taxed(
            thresholds,
            bin_n,
            bin_c0,
            config.max_k,
            n_total,
            eff_beam,
            crit,
            similar_lb,
            atom_teacher_c0,
            dp_teacher_alpha,
            config.reg_alpha * config.reg,                 // leaf_reg
            config.shape_sort_smoothing,                   // smoothing_alpha
            mu_node,                                       // boundary tax
            config.msdt.shape_sort_per_atom_teacher_prior, // per-atom prior
            config.min_samples_leaf,
        );
        if dp_teacher_active {
            if let Some(counter) = teacher_dp_active {
                *counter = counter.saturating_add(1);
            }
        }
        expand_atom_entries_to_bins(entries, atom_proj)
    } else {
        let (l, _) = collapse_bins_topb_binary_min_child(
            thresholds,
            bin_n,
            bin_c0,
            config.max_k,
            n_total,
            eff_beam,
            crit,
            None,
            config.min_samples_leaf,
        );
        l
    }
}

/// Cleanup pass: for each contiguous-mode candidate, emit additional
/// (k-1)-arity variants by merging adjacent children whose class-count
/// vectors are bitwise equal. Each merge drops one threshold and sums
/// the child_n / child_counts of the merged pair; the resulting split
/// has identical partition quality but saves one leaf_reg of regularization
/// cost, which can tip the branch-bound comparison on datasets where the DP
/// emits structurally-redundant splits. Skips shape-mode candidates
/// (group_map.is_some) — merging there requires re-indexing, deferred.
pub fn cleanup_collapse_contiguous(
    candidates: &[SplitCandidate],
    config: &SolverConfig,
    path_cost: f64,
    n_total: usize,
) -> Vec<SplitCandidate> {
    let mut out: Vec<SplitCandidate> = Vec::new();
    for cand in candidates {
        if cand.group_map.is_some() {
            continue; // shape mode: skip (group_map re-indexing deferred)
        }
        let k = cand.child_n.len();
        if k < 2 {
            continue;
        }
        for i in 0..k - 1 {
            if cand.child_counts[i] != cand.child_counts[i + 1] {
                continue;
            }
            // Merge children i and i+1 by dropping threshold[i] and summing
            // their counts. Requires thresholds to have at least k-1 entries
            // indexed 0..k-2 (the split points between adjacent children).
            if cand.thresholds.len() + 1 != k {
                continue;
            }
            let mut new_thresholds: Vec<f64> = cand.thresholds.to_vec();
            new_thresholds.remove(i);
            let mut new_child_n = cand.child_n.clone();
            new_child_n[i] = new_child_n[i] + new_child_n[i + 1];
            new_child_n.remove(i + 1);
            let mut new_child_counts = cand.child_counts.clone();
            {
                let (lo, hi) = new_child_counts.split_at_mut(i + 1);
                for (a, b) in lo[i].iter_mut().zip(hi[0].iter()) {
                    *a += *b;
                }
            }
            new_child_counts.remove(i + 1);

            let new_k = k - 1;
            let (rank_cost, split_ub) = score_candidate(
                &new_child_n,
                &new_child_counts,
                config,
                n_total,
                path_cost,
                CandidateOrder::Original,
            );
            out.push(SplitCandidate {
                rank_cost,
                split_ub,
                k: new_k,
                feature_index: cand.feature_index,
                thresholds: config.threshold_interner.intern_slice(&new_thresholds),
                group_map: None,
                child_n: new_child_n,
                child_counts: new_child_counts,
                family: cand.family,
                components: new_k as u32,
            });
        }
    }
    out
}

/// Multiclass twin of `run_dp_pass_binary`.
pub fn run_dp_pass_multiclass(
    use_shape: bool,
    crit: DpCriterion,
    config: &SolverConfig,
    thresholds: &[f64],
    bin_n: &[i64],
    bin_counts: &[Vec<i64>],
    _n_node: usize,
    n_total: usize,
    beam_schedule: &[usize],
    atom_proj: Option<&(Vec<usize>, Vec<f64>)>,
    similar_lb: Option<&HashMap<SupportSig, f64>>,
    atom_teacher_prob: Option<&[f64]>,
) -> Vec<(usize, CollapseEntry)> {
    if use_shape {
        let mu_node = if config.msdt.shape_boundary_tax > 0.0 {
            config.msdt.shape_boundary_tax / (n_total.max(1) as f64)
        } else {
            0.0
        };
        let entries = shape_collapse_topb_pairwise_multiclass_min_child(
            thresholds,
            bin_n,
            bin_counts,
            atom_teacher_prob,
            config.max_k,
            n_total,
            beam_schedule,
            crit,
            similar_lb,
            config.reg_alpha * config.reg, // leaf_reg
            mu_node,
            config.min_samples_leaf,
        );
        expand_atom_entries_to_bins(entries, atom_proj)
    } else {
        let (l, _) = collapse_bins_topb_min_child(
            thresholds,
            bin_n,
            bin_counts,
            config.max_k,
            n_total,
            beam_schedule,
            crit,
            None,
            config.reg_alpha * config.reg, // leaf_reg
            config.min_samples_leaf,
        );
        l
    }
}

/// Bin samples for binary classification — flat Vec instead of Vec<Vec>.
pub fn bin_samples_binary(
    feature_vals: &[f64],
    labels: &[u8],
    thresholds: &[f64],
    weights: Option<&[f64]>,
    weight_scale: f64,
) -> (Vec<i64>, Vec<i64>) {
    let n_bins = thresholds.len() + 1;
    let mut bin_n = vec![0i64; n_bins];
    let mut bin_c0 = vec![0i64; n_bins];
    if let Some(w) = weights {
        for (i, &val) in feature_vals.iter().enumerate() {
            let bin = searchsorted_right(thresholds, val);
            let wi = (w[i] * weight_scale).round() as i64;
            bin_n[bin] += wi;
            if labels[i] == 0 {
                bin_c0[bin] += wi;
            }
        }
    } else {
        for (i, &val) in feature_vals.iter().enumerate() {
            let bin = searchsorted_right(thresholds, val);
            bin_n[bin] += 1;
            if labels[i] == 0 {
                bin_c0[bin] += 1;
            }
        }
    }
    (bin_n, bin_c0)
}

// ---------------------------------------------------------------------------
// Top-B DP collapse: true top-B diverse K-partitions with loop-swapped DP
// ---------------------------------------------------------------------------

/// Maximum beam width supported without heap allocation.
/// beam_schedule values are capped at this.
/// At 16: TopB struct = 184 bytes; full dp table (L=500, max_k=5) ≈ 460 KB (L2/L3 resident).
/// Set to 128. Per-entry cost: 128 f64 + 128 u16 + 128 u8 + 16 bytes overhead =
/// ~1.4 KB/TopB. For L=500, max_k=5 full dp table ≈ 3.6 MB (L3 resident).
/// `prev_b` remains u8 (values < 128 fit).
pub(crate) const BEAM_MAX: usize = 128;

/// Fixed-size sorted list of the top-B (cost, backpointer) entries.
/// No heap allocation: all arrays are stack/inline in the parent Vec.
///
/// Invariant: costs[0..count] is sorted ascending; count <= BEAM_MAX.
#[derive(Clone)]
struct TopB {
    count: usize,
    costs: [f64; BEAM_MAX],
    prev_i: [u16; BEAM_MAX], // start bin of last group (i <= 500 fits u16)
    prev_b: [u8; BEAM_MAX],  // which entry in dp[k-1][prev_i-1] (b < 16 fits u8)
    best_rejected: f64,      // lowest cost ever rejected/evicted (INFINITY if none)
}

impl TopB {
    #[inline]
    fn new() -> Self {
        TopB {
            count: 0,
            costs: [f64::INFINITY; BEAM_MAX],
            prev_i: [0; BEAM_MAX],
            prev_b: [0; BEAM_MAX],
            best_rejected: f64::INFINITY,
        }
    }

    /// Try to insert (cost, pi, pb) maintaining the sorted top-BEAM_MAX list.
    ///
    /// Returns false if rejected because the lex key (cost, prev_i, prev_b)
    /// is not strictly smaller than the current worst stored entry.
    ///
    /// ## Canonical lex tie-breaking.
    /// Two entries with identical `cost` are ordered by `(prev_i, prev_b)`,
    /// invariant properties of the state itself rather than insertion
    /// order. This makes the surviving top-B set a deterministic function
    /// of the inserted state multiset (independent of insertion order),
    /// which in turn makes `top_B ⊆ top_{B'}` hold as ordered prefixes,
    /// not just as multisets.
    #[inline]
    fn try_insert(&mut self, cost: f64, pi: u16, pb: u8) -> bool {
        // Lex comparator: returns true iff a ≺ b under (cost, pi, pb).
        // NaN costs sort to the back (treat !(< ) as ordered-greater).
        let lt = |a_cost: f64, a_pi: u16, a_pb: u8, b_cost: f64, b_pi: u16, b_pb: u8| -> bool {
            if a_cost < b_cost {
                return true;
            }
            if a_cost > b_cost {
                return false;
            }
            if a_pi < b_pi {
                return true;
            }
            if a_pi > b_pi {
                return false;
            }
            a_pb < b_pb
        };

        // Fast rejection: full and new key is not lex-smaller than worst stored
        if self.count == BEAM_MAX {
            let worst_cost = self.costs[BEAM_MAX - 1];
            let worst_pi = self.prev_i[BEAM_MAX - 1];
            let worst_pb = self.prev_b[BEAM_MAX - 1];
            if !lt(cost, pi, pb, worst_cost, worst_pi, worst_pb) {
                if cost < self.best_rejected {
                    self.best_rejected = cost;
                }
                return false;
            }
        }
        // Insertion sort: find lex position, shift right, insert
        let end = if self.count < BEAM_MAX {
            self.count
        } else {
            // Evicting worst entry — track it
            let evicted = self.costs[BEAM_MAX - 1];
            if evicted < self.best_rejected {
                self.best_rejected = evicted;
            }
            BEAM_MAX - 1
        };
        let mut idx = end;
        while idx > 0
            && lt(
                cost,
                pi,
                pb,
                self.costs[idx - 1],
                self.prev_i[idx - 1],
                self.prev_b[idx - 1],
            )
        {
            self.costs[idx] = self.costs[idx - 1];
            self.prev_i[idx] = self.prev_i[idx - 1];
            self.prev_b[idx] = self.prev_b[idx - 1];
            idx -= 1;
        }
        self.costs[idx] = cost;
        self.prev_i[idx] = pi;
        self.prev_b[idx] = pb;
        if self.count < BEAM_MAX {
            self.count += 1;
        }
        true
    }
}

/// Per-arity beam rejection statistics from a single DP collapse invocation.
pub struct DpBeamStats {
    /// Saturation gap per arity: best_rejected - costs[count-1] (worst retained).
    pub gap_at_final: Vec<f64>,
    /// Number of cells with rejections/evictions at each arity level.
    pub n_saturated_cells: Vec<u64>,
}

impl DpBeamStats {
    fn new(eff_max_k: usize) -> Self {
        let len = eff_max_k.saturating_sub(1);
        DpBeamStats {
            gap_at_final: vec![0.0; len],
            n_saturated_cells: vec![0; len],
        }
    }

    pub fn empty() -> Self {
        DpBeamStats {
            gap_at_final: vec![],
            n_saturated_cells: vec![],
        }
    }
}

/// Reconstruct the boundary list for entry b of dp[k * l + (l-1)].
/// Returns the start indices of groups 2..=k (left-to-right), each >= 1.
fn reconstruct_path(dp: &[TopB], l: usize, k: usize, b: usize) -> Vec<usize> {
    let mut boundaries = Vec::with_capacity(k - 1);
    let mut curr_k = k;
    let mut curr_j = l - 1;
    let mut curr_b = b;
    while curr_k > 1 {
        let e = &dp[curr_k * l + curr_j];
        let pi = e.prev_i[curr_b] as usize;
        let pb = e.prev_b[curr_b] as usize;
        boundaries.push(pi);
        curr_j = pi - 1;
        curr_b = pb;
        curr_k -= 1;
    }
    boundaries.reverse();
    boundaries
}

/// Binary impurity cost for a group with c0 class-0 samples out of n total.
#[inline(always)]
fn binary_cost(c0: i64, n: i64, n_total_f: f64, use_misclass: bool) -> f64 {
    if use_misclass {
        let max_c = c0.max(n - c0);
        (n - max_c) as f64 / n_total_f
    } else {
        let c1 = n - c0;
        let ssq = c0 * c0 + c1 * c1;
        (n as f64 - ssq as f64 / n as f64) / n_total_f
    }
}

#[inline(always)]
fn soft_binary_gini_normalized(c0_soft: f64, n_soft: f64, n_total_f: f64) -> f64 {
    if n_soft <= 0.0 {
        return 0.0;
    }
    let c0 = c0_soft.clamp(0.0, n_soft);
    let c1 = n_soft - c0;
    (n_soft - (c0 * c0 + c1 * c1) / n_soft) / n_total_f
}

/// Multiclass impurity cost accumulated into a small stack array.
#[inline]
fn multiclass_cost(class_acc: &[i64], n: i64, n_total_f: f64, use_misclass: bool) -> f64 {
    if use_misclass {
        let max_c = class_acc.iter().copied().max().unwrap_or(0);
        (n - max_c) as f64 / n_total_f
    } else {
        let ssq: i64 = class_acc.iter().map(|&c| c * c).sum();
        (n as f64 - ssq as f64 / n as f64) / n_total_f
    }
}

/// Additive proxy for Twoing criterion (multiclass).
///
/// cost(g) = (n_g / N) * (1 - TV(p_g, p_bar)^2)
/// where TV = (1/2) * sum_c |p_g(c)/n_g - p_bar(c)| is total variation distance
/// from the node's overall class distribution.
///
/// Groups far from the population average get low cost (rewarded);
/// groups matching the average get high cost (penalized).
/// For C=2 this reduces to Gini.
#[inline]
fn twoing_proxy_cost(class_acc: &[i64], n: i64, n_total_f: f64, p_bar: &[f64]) -> f64 {
    if n == 0 {
        return 0.0;
    }
    let n_f = n as f64;
    let mut tv = 0.0;
    for (i, &cnt) in class_acc.iter().enumerate() {
        tv += (cnt as f64 / n_f - p_bar[i]).abs();
    }
    tv *= 0.5;
    (n_f / n_total_f) * (1.0 - tv * tv)
}

/// Binary entropy cost: H(group) = (n·ln(n) - c0·ln(c0) - c1·ln(c1)) / N.
/// Convention: 0·ln(0) = 0.
#[inline(always)]
fn binary_cost_entropy(c0: i64, n: i64, n_total_f: f64) -> f64 {
    if n == 0 {
        return 0.0;
    }
    let n_f = n as f64;
    let mut cost = n_f * n_f.ln();
    if c0 > 0 {
        cost -= c0 as f64 * (c0 as f64).ln();
    }
    let c1 = n - c0;
    if c1 > 0 {
        cost -= c1 as f64 * (c1 as f64).ln();
    }
    cost / n_total_f
}

/// Binary Hellinger (Bhattacharyya) cost: 2·√(c0·c1) / N.
/// Pure groups → 0. Balanced → n/N. Fast: one multiply + one sqrt.
#[inline(always)]
fn binary_cost_hellinger(c0: i64, n: i64, n_total_f: f64) -> f64 {
    if n == 0 {
        return 0.0;
    }
    let c1 = n - c0;
    2.0 * ((c0 as f64 * c1 as f64).sqrt()) / n_total_f
}

/// Multiclass entropy cost: (n·ln(n) - Σ_c c_c·ln(c_c)) / N.
#[inline]
fn multiclass_cost_entropy(class_acc: &[i64], n: i64, n_total_f: f64) -> f64 {
    if n == 0 {
        return 0.0;
    }
    let n_f = n as f64;
    let mut cost = n_f * n_f.ln();
    for &c in class_acc {
        if c > 0 {
            cost -= c as f64 * (c as f64).ln();
        }
    }
    cost / n_total_f
}

/// Multiclass Hellinger cost: (n/N) · (1 - (Σ_c √(c_c/n))² / C).
#[inline]
fn multiclass_cost_hellinger(class_acc: &[i64], n: i64, n_total_f: f64) -> f64 {
    if n == 0 {
        return 0.0;
    }
    let n_f = n as f64;
    let n_classes = class_acc.len() as f64;
    let sum_sqrt: f64 = class_acc.iter().map(|&c| (c as f64 / n_f).sqrt()).sum();
    (n_f / n_total_f) * (1.0 - sum_sqrt * sum_sqrt / n_classes)
}

/// Learned cost for binary classification: look up similar_lb, fall back to Gini.
#[inline]
fn learned_cost_binary(
    c0: i64,
    n: i64,
    n_total_f: f64,
    similar_lb: &HashMap<SupportSig, f64>,
) -> f64 {
    if n == 0 {
        return 0.0;
    }
    let counts = [c0, n - c0];
    let sig = SupportSig::new(n as usize, &counts);
    if let Some(&lb) = similar_lb.get(&sig) {
        lb
    } else {
        // Fallback to Gini
        let c1 = n - c0;
        let ssq = c0 * c0 + c1 * c1;
        (n as f64 - ssq as f64 / n as f64) / n_total_f
    }
}

/// Learned cost for multiclass: look up similar_lb, fall back to Gini.
#[inline]
fn learned_cost_multiclass(
    class_acc: &[i64],
    n: i64,
    n_total_f: f64,
    similar_lb: &HashMap<SupportSig, f64>,
) -> f64 {
    if n == 0 {
        return 0.0;
    }
    let sig = SupportSig::new(n as usize, class_acc);
    if let Some(&lb) = similar_lb.get(&sig) {
        lb
    } else {
        let ssq: i64 = class_acc.iter().map(|&c| c * c).sum();
        (n as f64 - ssq as f64 / n as f64) / n_total_f
    }
}

/// Binary Wasserstein separation cost (linear analogue of Twoing).
///
/// cost(group) = (n/N) * (1 - |c0/n - node_p0|)
///
/// Groups matching the node's average have cost = n/N (maximum, penalized).
/// Groups deviating from the average have lower cost (rewarded).
/// This is W₁ between Bernoulli(c0/n) and Bernoulli(node_p0) = |p - p0|.
/// Twoing uses TV² = |p - p0|²; this uses TV = |p - p0| (linear, stronger gradient).
#[inline(always)]
fn binary_cost_wasserstein(c0: i64, n: i64, n_total_f: f64, node_p0: f64) -> f64 {
    if n == 0 {
        return 0.0;
    }
    let p = c0 as f64 / n as f64;
    let w1 = (p - node_p0).abs();
    (n as f64 / n_total_f) * (1.0 - w1)
}

/// Multiclass ordinal Wasserstein separation cost.
///
/// cost(group) = (n/N) * (1 - W₁_ordinal(p_group, p_bar) / (C-1))
///
/// W₁ with ordinal ground metric d(i,j) = |i-j| is computed as:
///   W₁ = Σ_{c=0}^{C-2} |CDF_group(c) - CDF_bar(c)|
///
/// Normalization by (C-1) ensures W₁ ∈ [0, 1].
/// Unlike TV-based Twoing, this respects class ordering: adjacent classes
/// contribute less distance than distant classes.
#[inline]
fn multiclass_cost_wasserstein(class_acc: &[i64], n: i64, n_total_f: f64, p_bar: &[f64]) -> f64 {
    if n == 0 {
        return 0.0;
    }
    let n_f = n as f64;
    let c_count = class_acc.len();
    if c_count <= 1 {
        return n_f / n_total_f;
    }
    let mut cdf_g = 0.0;
    let mut cdf_bar = 0.0;
    let mut w1 = 0.0;
    for c in 0..(c_count - 1) {
        cdf_g += class_acc[c] as f64 / n_f;
        cdf_bar += p_bar[c];
        w1 += (cdf_g - cdf_bar).abs();
    }
    let max_w1 = (c_count - 1) as f64;
    (n_f / n_total_f) * (1.0 - w1 / max_w1)
}

/// Precompute subsplit table for binary classification (depth-2 criterion).
///
/// Returns a flat L×L table where `subsplit[i*l + j]` =
///   min over split point t in [i+1..j] of (gini(bins[i..t-1]) + gini(bins[t..j])).
/// Single-bin groups (i==j): fallback to gini of that single bin.
/// Uses prefix sums for O(1) per split-point → O(L³) total.
fn precompute_subsplit_binary(bin_n: &[i64], bin_c0: &[i64], n_total_f: f64) -> Vec<f64> {
    let l = bin_n.len();
    let mut table = vec![f64::INFINITY; l * l];

    // Prefix sums for O(1) range queries
    let mut cum_n = vec![0i64; l + 1];
    let mut cum_c0 = vec![0i64; l + 1];
    for i in 0..l {
        cum_n[i + 1] = cum_n[i] + bin_n[i];
        cum_c0[i + 1] = cum_c0[i] + bin_c0[i];
    }

    // Helper: gini cost for bins[a..=b]
    let gini = |a: usize, b: usize| -> f64 {
        let n = cum_n[b + 1] - cum_n[a];
        if n == 0 {
            return 0.0;
        }
        let c0 = cum_c0[b + 1] - cum_c0[a];
        let c1 = n - c0;
        let ssq = c0 * c0 + c1 * c1;
        (n as f64 - ssq as f64 / n as f64) / n_total_f
    };

    for i in 0..l {
        // Single-bin: fallback to gini of that bin
        table[i * l + i] = gini(i, i);

        // Multi-bin groups: try all split points
        for j in (i + 1)..l {
            let mut best = f64::INFINITY;
            for t in (i + 1)..=j {
                let cost = gini(i, t - 1) + gini(t, j);
                if cost < best {
                    best = cost;
                }
            }
            table[i * l + j] = best;
        }
    }

    table
}

/// Precompute subsplit table for multiclass classification (depth-2 criterion).
///
/// Same structure as binary version but uses Gini over all classes.
fn precompute_subsplit_multiclass(
    bin_n: &[i64],
    bin_counts: &[Vec<i64>],
    n_total_f: f64,
) -> Vec<f64> {
    let l = bin_n.len();
    let n_classes = bin_counts[0].len();
    let mut table = vec![f64::INFINITY; l * l];

    // Prefix sums
    let mut cum_n = vec![0i64; l + 1];
    let mut cum_counts = vec![vec![0i64; n_classes]; l + 1];
    for i in 0..l {
        cum_n[i + 1] = cum_n[i] + bin_n[i];
        for c in 0..n_classes {
            cum_counts[i + 1][c] = cum_counts[i][c] + bin_counts[i][c];
        }
    }

    // Helper: gini cost for bins[a..=b]
    let gini = |a: usize, b: usize| -> f64 {
        let n = cum_n[b + 1] - cum_n[a];
        if n == 0 {
            return 0.0;
        }
        let mut ssq = 0i64;
        for c in 0..n_classes {
            let cnt = cum_counts[b + 1][c] - cum_counts[a][c];
            ssq += cnt * cnt;
        }
        (n as f64 - ssq as f64 / n as f64) / n_total_f
    };

    for i in 0..l {
        table[i * l + i] = gini(i, i);
        for j in (i + 1)..l {
            let mut best = f64::INFINITY;
            for t in (i + 1)..=j {
                let cost = gini(i, t - 1) + gini(t, j);
                if cost < best {
                    best = cost;
                }
            }
            table[i * l + j] = best;
        }
    }

    table
}

/// Precompute subsplit table using misclassification cost (depth-2 misclass criterion).
fn precompute_subsplit_binary_misclass(bin_n: &[i64], bin_c0: &[i64], n_total_f: f64) -> Vec<f64> {
    let l = bin_n.len();
    let mut table = vec![f64::INFINITY; l * l];

    let mut cum_n = vec![0i64; l + 1];
    let mut cum_c0 = vec![0i64; l + 1];
    for i in 0..l {
        cum_n[i + 1] = cum_n[i] + bin_n[i];
        cum_c0[i + 1] = cum_c0[i] + bin_c0[i];
    }

    let misclass = |a: usize, b: usize| -> f64 {
        let n = cum_n[b + 1] - cum_n[a];
        if n == 0 {
            return 0.0;
        }
        let c0 = cum_c0[b + 1] - cum_c0[a];
        let c1 = n - c0;
        (n - c0.max(c1)) as f64 / n_total_f
    };

    for i in 0..l {
        table[i * l + i] = misclass(i, i);
        for j in (i + 1)..l {
            let mut best = f64::INFINITY;
            for t in (i + 1)..=j {
                let cost = misclass(i, t - 1) + misclass(t, j);
                if cost < best {
                    best = cost;
                }
            }
            table[i * l + j] = best;
        }
    }

    table
}

/// Precompute subsplit table using misclassification cost (multiclass).
fn precompute_subsplit_multiclass_misclass(
    bin_n: &[i64],
    bin_counts: &[Vec<i64>],
    n_total_f: f64,
) -> Vec<f64> {
    let l = bin_n.len();
    let n_classes = bin_counts[0].len();
    let mut table = vec![f64::INFINITY; l * l];

    let mut cum_n = vec![0i64; l + 1];
    let mut cum_counts = vec![vec![0i64; n_classes]; l + 1];
    for i in 0..l {
        cum_n[i + 1] = cum_n[i] + bin_n[i];
        for c in 0..n_classes {
            cum_counts[i + 1][c] = cum_counts[i][c] + bin_counts[i][c];
        }
    }

    let misclass = |a: usize, b: usize| -> f64 {
        let n = cum_n[b + 1] - cum_n[a];
        if n == 0 {
            return 0.0;
        }
        let mut max_c = 0i64;
        for c in 0..n_classes {
            let cnt = cum_counts[b + 1][c] - cum_counts[a][c];
            if cnt > max_c {
                max_c = cnt;
            }
        }
        (n - max_c) as f64 / n_total_f
    };

    for i in 0..l {
        table[i * l + i] = misclass(i, i);
        for j in (i + 1)..l {
            let mut best = f64::INFINITY;
            for t in (i + 1)..=j {
                let cost = misclass(i, t - 1) + misclass(t, j);
                if cost < best {
                    best = cost;
                }
            }
            table[i * l + j] = best;
        }
    }

    table
}

#[derive(Clone)]
pub struct RushTopB {
    count: usize,
    costs: [f64; BEAM_MAX],
    prev_t: [u16; BEAM_MAX],
    prev_b: [u8; BEAM_MAX],
    best_rejected: f64,
}

impl RushTopB {
    #[inline]
    fn new() -> Self {
        RushTopB {
            count: 0,
            costs: [f64::INFINITY; BEAM_MAX],
            prev_t: [0; BEAM_MAX],
            prev_b: [0; BEAM_MAX],
            best_rejected: f64::INFINITY,
        }
    }

    /// Admit a sorted predecessor beam in linear time. Existing entries win
    /// cost ties, matching repeated stable `try_insert` calls exactly.
    fn merge_predecessor(&mut self, previous: &Self, support: f64, margin: f64, pt: u16) {
        if previous.count == 0 {
            return;
        }
        let old = self.clone();
        let mut left = 0;
        let mut right = 0;
        self.count = 0;
        while self.count < BEAM_MAX && (left < old.count || right < previous.count) {
            let incoming = if right < previous.count {
                (previous.costs[right] - support) + margin
            } else {
                f64::INFINITY
            };
            let dst = self.count;
            if left < old.count && (right >= previous.count || old.costs[left] <= incoming) {
                self.costs[dst] = old.costs[left];
                self.prev_t[dst] = old.prev_t[left];
                self.prev_b[dst] = old.prev_b[left];
                left += 1;
            } else {
                self.costs[dst] = incoming;
                self.prev_t[dst] = pt;
                self.prev_b[dst] = right as u8;
                right += 1;
            }
            self.count += 1;
        }
        self.best_rejected = old.best_rejected;
        if left < old.count {
            self.best_rejected = self.best_rejected.min(old.costs[left]);
        }
        if right < previous.count {
            self.best_rejected = self
                .best_rejected
                .min((previous.costs[right] - support) + margin);
        }
    }

    #[inline]
    fn try_insert(&mut self, cost: f64, pt: u16, pb: u8) -> bool {
        if self.count == BEAM_MAX && cost >= self.costs[BEAM_MAX - 1] {
            if cost < self.best_rejected {
                self.best_rejected = cost;
            }
            return false;
        }
        let end = if self.count < BEAM_MAX {
            self.count
        } else {
            let evicted = self.costs[BEAM_MAX - 1];
            if evicted < self.best_rejected {
                self.best_rejected = evicted;
            }
            BEAM_MAX - 1
        };
        let mut idx = end;
        while idx > 0 && self.costs[idx - 1] > cost {
            self.costs[idx] = self.costs[idx - 1];
            self.prev_t[idx] = self.prev_t[idx - 1];
            self.prev_b[idx] = self.prev_b[idx - 1];
            idx -= 1;
        }
        self.costs[idx] = cost;
        self.prev_t[idx] = pt;
        self.prev_b[idx] = pb;
        if self.count < BEAM_MAX {
            self.count += 1;
        }
        true
    }
}

/// Merge the two realized sorted channel beams, retaining the historical
/// first-seen order on ties (A entries first, then new B entries). The stamp
/// table deduplicates predecessor identities without a quadratic search.
#[allow(clippy::too_many_arguments)]
fn merge_rush_channels(
    output: &mut RushTopB,
    a: &RushTopB,
    b: &RushTopB,
    support: f64,
    margin: f64,
    cap: usize,
    stamps: &mut [u32],
    ranks: &mut [u16],
    generation: &mut u32,
    n_total: f64,
    suffix_lb: f64,
    regularization_lb: f64,
    upper_bound: f64,
) {
    *generation = generation.wrapping_add(2);
    if *generation < 2 {
        stamps.fill(0);
        *generation = 2;
    }
    let present = *generation;
    let emitted = present + 1;
    let mut identities_t = [0u16; BEAM_MAX * 2];
    let mut identities_b = [0u8; BEAM_MAX * 2];
    let mut realized = [f64::INFINITY; BEAM_MAX * 2];
    for i in 0..a.count {
        let key = a.prev_t[i] as usize * BEAM_MAX + a.prev_b[i] as usize;
        stamps[key] = present;
        ranks[key] = i as u16;
        identities_t[i] = a.prev_t[i];
        identities_b[i] = a.prev_b[i];
        realized[i] = support + a.costs[i] - margin;
    }
    for i in 0..b.count {
        let key = b.prev_t[i] as usize * BEAM_MAX + b.prev_b[i] as usize;
        let cost = support + b.costs[i] + margin;
        if stamps[key] == present {
            let rank = ranks[key] as usize;
            if cost < realized[rank] {
                realized[rank] = cost;
            }
        } else {
            let rank = a.count + i;
            stamps[key] = present;
            ranks[key] = rank as u16;
            identities_t[rank] = b.prev_t[i];
            identities_b[rank] = b.prev_b[i];
            realized[rank] = cost;
        }
    }
    let mut ai = 0;
    let mut bi = 0;
    while ai < a.count || bi < b.count {
        let ac = if ai < a.count {
            support + a.costs[ai] - margin
        } else {
            f64::INFINITY
        };
        let bc = if bi < b.count {
            support + b.costs[bi] + margin
        } else {
            f64::INFINITY
        };
        let cost = ac.min(bc);
        if upper_bound.is_finite()
            && cost / (2.0 * n_total) + suffix_lb + regularization_lb >= upper_bound
        {
            break;
        }
        // B's equal-cost entries need not have A's first-seen order. A small
        // indexed bit set restores that stable order without sorting the run.
        // The summary skips empty words, so work is linear in emitted entries.
        let mut tied = [0u64; (BEAM_MAX * 2 + 63) / 64];
        let mut words = 0u64;
        while ai < a.count && support + a.costs[ai] - margin == cost {
            let key = a.prev_t[ai] as usize * BEAM_MAX + a.prev_b[ai] as usize;
            if stamps[key] != emitted {
                stamps[key] = emitted;
                let rank = ranks[key] as usize;
                tied[rank / 64] |= 1u64 << (rank % 64);
                words |= 1u64 << (rank / 64);
            }
            ai += 1;
        }
        while bi < b.count && support + b.costs[bi] + margin == cost {
            let key = b.prev_t[bi] as usize * BEAM_MAX + b.prev_b[bi] as usize;
            if stamps[key] != emitted {
                stamps[key] = emitted;
                let rank = ranks[key] as usize;
                tied[rank / 64] |= 1u64 << (rank % 64);
                words |= 1u64 << (rank / 64);
            }
            bi += 1;
        }
        while words != 0 {
            let word = words.trailing_zeros() as usize;
            while tied[word] != 0 {
                let bit = tied[word].trailing_zeros() as usize;
                tied[word] &= tied[word] - 1;
                let rank = word * 64 + bit;
                if output.count == cap {
                    output.best_rejected = output.best_rejected.min(realized[rank]);
                    return;
                }
                let dst = output.count;
                output.costs[dst] = realized[rank];
                output.prev_t[dst] = identities_t[rank];
                output.prev_b[dst] = identities_b[rank];
                output.count += 1;
            }
            words &= words - 1;
        }
    }
}

fn reconstruct_rush_path(
    dp: &[RushTopB],
    width: usize,
    k: usize,
    b: usize,
    end_t: usize,
) -> Vec<usize> {
    let mut boundaries = Vec::with_capacity(k - 1);
    let mut curr_k = k;
    let mut curr_t = end_t;
    let mut curr_b = b;
    while curr_k > 1 {
        let e = &dp[curr_k * width + curr_t];
        let pt = e.prev_t[curr_b] as usize;
        let pb = e.prev_b[curr_b] as usize;
        boundaries.push(pt);
        curr_t = pt;
        curr_b = pb;
        curr_k -= 1;
    }
    boundaries.reverse();
    boundaries
}

#[inline]
#[cfg(test)]
fn upsert_rush_candidate(
    cand_prev_t: &mut [u16; BEAM_MAX * 2],
    cand_prev_b: &mut [u8; BEAM_MAX * 2],
    cand_costs: &mut [f64; BEAM_MAX * 2],
    cand_count: &mut usize,
    pt: u16,
    pb: u8,
    cost: f64,
) {
    for idx in 0..*cand_count {
        if cand_prev_t[idx] == pt && cand_prev_b[idx] == pb {
            if cost < cand_costs[idx] {
                cand_costs[idx] = cost;
            }
            return;
        }
    }
    if *cand_count < BEAM_MAX * 2 {
        let idx = *cand_count;
        cand_prev_t[idx] = pt;
        cand_prev_b[idx] = pb;
        cand_costs[idx] = cost;
        *cand_count += 1;
        return;
    }
    let mut worst_idx = 0usize;
    for idx in 1..(BEAM_MAX * 2) {
        if cand_costs[idx] > cand_costs[worst_idx] {
            worst_idx = idx;
        }
    }
    if cost < cand_costs[worst_idx] {
        cand_prev_t[worst_idx] = pt;
        cand_prev_b[worst_idx] = pb;
        cand_costs[worst_idx] = cost;
    }
}

/// RUSH-DP for binary misclassification with Top-B beam states.
///
/// - Projects to non-empty bins only (exact over projected grid).
/// - Uses running minima recurrence with top-B minima channels A/B.
/// - Returns up to beam_schedule[k-2] candidates per feasible arity k.
/// - Optional UB/suffix pruning is enabled only when `ub_best` is finite.
pub fn rush_dp_binary_pub(
    thresholds: &[f64],
    bin_n: &[i64],
    bin_c0: &[i64],
    max_k: usize,
    n_total: usize,
    beam_schedule: &[usize],
    min_support: usize,
    ub_best: f64,
    reg: f64,
    reg_alpha: f64,
    dp_reuse: Option<&mut Vec<RushTopB>>,
) -> (Vec<(usize, CollapseEntry)>, DpBeamStats) {
    rush_dp_binary(
        thresholds,
        bin_n,
        bin_c0,
        max_k,
        n_total,
        beam_schedule,
        min_support,
        ub_best,
        reg,
        reg_alpha,
        dp_reuse,
    )
}

fn rush_dp_binary(
    thresholds: &[f64],
    bin_n: &[i64],
    bin_c0: &[i64],
    max_k: usize,
    n_total: usize,
    beam_schedule: &[usize],
    min_support: usize,
    ub_best: f64,
    reg: f64,
    reg_alpha: f64,
    dp_reuse: Option<&mut Vec<RushTopB>>,
) -> (Vec<(usize, CollapseEntry)>, DpBeamStats) {
    let l_orig = bin_n.len();
    if l_orig <= 1 {
        return (vec![], DpBeamStats::empty());
    }

    let m = min_support.max(1) as i64;
    let n_total_f = n_total as f64;
    let split_reg = (1.0 - reg_alpha) * reg;
    let leaf_reg = reg_alpha * reg;

    // Project to non-empty bins to avoid degenerate states in the DP grid.
    let mut proj_to_orig = Vec::with_capacity(l_orig);
    let mut proj_n = Vec::with_capacity(l_orig);
    let mut proj_c0 = Vec::with_capacity(l_orig);
    for b in 0..l_orig {
        if bin_n[b] > 0 {
            proj_to_orig.push(b);
            proj_n.push(bin_n[b]);
            proj_c0.push(bin_c0[b]);
        }
    }

    let l = proj_n.len();
    if l <= 1 {
        return (vec![], DpBeamStats::empty());
    }

    // Prefix support/margin on projected bins with t in [0..L].
    let mut s = vec![0i64; l + 1];
    let mut d = vec![0i64; l + 1];
    for t in 0..l {
        s[t + 1] = s[t] + proj_n[t];
        d[t + 1] = d[t] + (2 * proj_c0[t] - proj_n[t]);
    }
    let total_support = s[l];
    if total_support == 0 {
        return (vec![], DpBeamStats::empty());
    }

    // Suffix admissible LB: sum of per-bin misclass on remaining projected bins.
    let mut epb_suf = vec![0.0f64; l + 1];
    for t in (0..l).rev() {
        let gn = proj_n[t];
        let gc0 = proj_c0[t];
        let mis = (gn - gc0.max(gn - gc0)) as f64 / n_total_f;
        epb_suf[t] = epb_suf[t + 1] + mis;
    }

    let mut ub_local = ub_best;
    if ub_local.is_finite() {
        let total_c0: i64 = proj_c0.iter().sum();
        let mis_one = (total_support - total_c0.max(total_support - total_c0)) as f64 / n_total_f;
        ub_local = ub_local.min(mis_one + split_reg + 2.0 * leaf_reg);
    }

    let mut m_max = max_k.min(l);
    let support_cap = (total_support / m) as usize;
    m_max = m_max.min(support_cap);
    if ub_local.is_finite() && leaf_reg > 0.0 {
        let rhs = (ub_local - split_reg - epb_suf[0]) / leaf_reg;
        if rhs.is_finite() {
            let cap = rhs.floor() as isize;
            if cap < 2 {
                return (vec![], DpBeamStats::empty());
            }
            m_max = m_max.min(cap as usize);
        }
    }
    if m_max < 2 {
        return (vec![], DpBeamStats::empty());
    }

    let width = l + 1;
    let needed = (m_max + 1) * width;
    let mut dp_owned;
    let dp: &mut [RushTopB] = if let Some(buf) = dp_reuse {
        if buf.len() < needed {
            buf.resize(needed, RushTopB::new());
        }
        for e in buf[..needed].iter_mut() {
            *e = RushTopB::new();
        }
        &mut buf[..needed]
    } else {
        dp_owned = vec![RushTopB::new(); needed];
        &mut dp_owned
    };

    let mut union_stamps = vec![0u32; width * BEAM_MAX];
    let mut union_ranks = vec![0u16; width * BEAM_MAX];
    let mut union_generation = 0u32;

    // Base i=1: exactly one way for each prefix t.
    for t in 1..=l {
        if s[t] < m {
            continue;
        }
        let dp2 = (s[t] - d[t].abs()) as f64;
        if ub_local.is_finite() {
            let reg_lb = if t < l {
                split_reg + 2.0 * leaf_reg
            } else {
                split_reg + leaf_reg
            };
            let obj_lb = dp2 / (2.0 * n_total_f) + epb_suf[t] + reg_lb;
            if obj_lb >= ub_local {
                continue;
            }
        }
        dp[width + t].try_insert(dp2, 0, 0);
    }

    // i >= 2: running top-B minima channels.
    for i in 2..=m_max {
        let mut a_top = RushTopB::new();
        let mut b_top = RushTopB::new();
        let mut k_ptr = i - 1;

        for t in i..=l {
            if s[t] < (i as i64) * m {
                continue;
            }
            let tau = s[t] - m;
            while k_ptr < t && s[k_ptr] <= tau {
                let prev = &dp[(i - 1) * width + k_ptr];
                a_top.merge_predecessor(prev, s[k_ptr] as f64, d[k_ptr] as f64, k_ptr as u16);
                b_top.merge_predecessor(prev, s[k_ptr] as f64, -(d[k_ptr] as f64), k_ptr as u16);
                k_ptr += 1;
            }

            if a_top.count == 0 && b_top.count == 0 {
                continue;
            }

            let reg_lb = if t < l {
                split_reg + ((i + 1) as f64) * leaf_reg
            } else {
                split_reg + (i as f64) * leaf_reg
            };
            merge_rush_channels(
                &mut dp[i * width + t],
                &a_top,
                &b_top,
                s[t] as f64,
                d[t] as f64,
                BEAM_MAX,
                &mut union_stamps,
                &mut union_ranks,
                &mut union_generation,
                n_total_f,
                epb_suf[t],
                reg_lb,
                ub_local,
            );
        }
    }

    // --- Beam rejection stats extraction ---
    let mut beam_stats = DpBeamStats::new(m_max);
    for k in 2..=m_max {
        let idx = k - 2;
        let fc = &dp[k * width + l];
        if fc.best_rejected < f64::INFINITY && fc.count > 0 {
            beam_stats.gap_at_final[idx] = fc.best_rejected - fc.costs[fc.count - 1];
        }
        for j in 0..=l {
            if dp[k * width + j].best_rejected < f64::INFINITY {
                beam_stats.n_saturated_cells[idx] += 1;
            }
        }
    }

    // Prefix sums over original bins for group-stat extraction.
    let mut cum_n = vec![0i64; l_orig + 1];
    let mut cum_c0 = vec![0i64; l_orig + 1];
    for i in 0..l_orig {
        cum_n[i + 1] = cum_n[i] + bin_n[i];
        cum_c0[i + 1] = cum_c0[i] + bin_c0[i];
    }

    let mut ladder = Vec::new();
    for k in 2..=m_max {
        let beam = beam_schedule
            .get(k - 2)
            .copied()
            .unwrap_or(1)
            .max(1)
            .min(BEAM_MAX);
        let final_entry = &dp[k * width + l];
        for b in 0..final_entry.count.min(beam) {
            let proj_boundaries = reconstruct_rush_path(&dp, width, k, b, l);
            if proj_boundaries.len() != k - 1 {
                continue;
            }

            let mut orig_starts = Vec::with_capacity(k - 1);
            let mut split_thresh = Vec::with_capacity(k - 1);
            let mut valid = true;
            for &pb in &proj_boundaries {
                if pb >= l {
                    valid = false;
                    break;
                }
                let orig_start = proj_to_orig[pb];
                if orig_start == 0 {
                    valid = false;
                    break;
                }
                orig_starts.push(orig_start);
                split_thresh.push(thresholds[orig_start - 1]);
            }
            if !valid {
                continue;
            }

            let starts = std::iter::once(0usize).chain(orig_starts.iter().copied());
            let ends = orig_starts.iter().copied().chain(std::iter::once(l_orig));

            let mut group_n = Vec::with_capacity(k);
            let mut group_counts = Vec::with_capacity(k);
            for (gs, ge) in starts.zip(ends) {
                let gn = cum_n[ge] - cum_n[gs];
                if gn < m {
                    valid = false;
                    break;
                }
                let gc0 = cum_c0[ge] - cum_c0[gs];
                group_n.push(gn);
                group_counts.push(vec![gc0, gn - gc0]);
            }
            if !valid {
                continue;
            }

            // Expose Rush's output on the regularized-objective
            // scale `Σ φ(groups) + k·leaf_reg`, matching the main DP's
            // convention. `split_reg` is applied by the caller alongside
            // `path_cost`; per-group `leaf_reg` is paid here so that top-B
            // candidate ordering is consistent with the final regularized
            // tree cost. Without this, Rush candidate ranking is tilted
            // toward larger k (more groups = lower raw misclass) with no
            // offsetting regularization penalty at rank time.
            let reg_cost = final_entry.costs[b] / (2.0 * n_total_f) + (k as f64) * leaf_reg;
            ladder.push((
                k,
                CollapseEntry {
                    cost: reg_cost,
                    thresholds: Arc::from(split_thresh.as_slice()),
                    bin_n: group_n,
                    bin_counts: group_counts,
                    group_map: None,
                    order: CandidateOrder::Original,
                },
            ));
        }
    }

    (ladder, beam_stats)
}

// ---------------------------------------------------------------------------
// Exact speed paths of the binary Select DPs (per-thread tables, the Rush
// slot table, and the stats-free `topcell` / `cap`). Same returned ladder
// as `rush_dp_binary` / `collapse_bins_topb_binary_min_child` (bitwise,
// same order); only the beam statistics (`DpBeamStats`) differ under
// `topcell` / `cap`. The contiguous Gini ladder and Misclass companion use
// them for every caller, so there `DpBeamStats` (`beam_rejection_stats`)
// is measured against the capped cell size b_k, not BEAM_MAX; the other
// DPs use them only for stats-free callers.
// ---------------------------------------------------------------------------

/// Stats-free speed paths of a binary Select DP (the fast DPs always use
/// per-thread tables with counts reset only, no k = 0 row, borrowed
/// predecessor rows, and the Rush (prev_t, prev_b) slot table).
#[derive(Clone, Copy, Debug, Default)]
pub struct DpOpts {
    /// Top arity layer: only the final cell (the one extraction reads).
    pub topcell: bool,
    /// Cells hold only the entries a later extraction can read: final cell
    /// of arity k at b_k; collapse interior cells of arity k at
    /// max_{k' > k} b_k'.
    pub cap: bool,
}

impl RushTopB {
    #[inline]
    fn reset(&mut self) {
        self.count = 0;
        self.best_rejected = f64::INFINITY;
    }
}

impl TopB {
    /// `try_insert` holding at most `cap` (1..=BEAM_MAX) entries: the
    /// lex-smallest `cap` of the inserted (cost, prev_i, prev_b) keys, i.e.
    /// the first `cap` entries of the BEAM_MAX list.
    #[inline]
    fn try_insert_cap(&mut self, cost: f64, pi: u16, pb: u8, cap: usize) -> bool {
        let lt = |a_cost: f64, a_pi: u16, a_pb: u8, b_cost: f64, b_pi: u16, b_pb: u8| -> bool {
            if a_cost < b_cost {
                return true;
            }
            if a_cost > b_cost {
                return false;
            }
            if a_pi < b_pi {
                return true;
            }
            if a_pi > b_pi {
                return false;
            }
            a_pb < b_pb
        };
        if self.count == cap {
            let w = cap - 1;
            if !lt(cost, pi, pb, self.costs[w], self.prev_i[w], self.prev_b[w]) {
                if cost < self.best_rejected {
                    self.best_rejected = cost;
                }
                return false;
            }
        }
        let end = if self.count < cap {
            self.count
        } else {
            let evicted = self.costs[cap - 1];
            if evicted < self.best_rejected {
                self.best_rejected = evicted;
            }
            cap - 1
        };
        let mut idx = end;
        while idx > 0
            && lt(
                cost,
                pi,
                pb,
                self.costs[idx - 1],
                self.prev_i[idx - 1],
                self.prev_b[idx - 1],
            )
        {
            self.costs[idx] = self.costs[idx - 1];
            self.prev_i[idx] = self.prev_i[idx - 1];
            self.prev_b[idx] = self.prev_b[idx - 1];
            idx -= 1;
        }
        self.costs[idx] = cost;
        self.prev_i[idx] = pi;
        self.prev_b[idx] = pb;
        if self.count < cap {
            self.count += 1;
        }
        true
    }

    #[inline]
    fn reset(&mut self) {
        self.count = 0;
        self.best_rejected = f64::INFINITY;
    }
}

/// Per-thread buffers of the DP speed paths.
#[derive(Default)]
struct DpScratch {
    rush: Vec<RushTopB>,
    collapse: Vec<TopB>,
    /// Rush slot table over (prev_t, prev_b): generation stamp
    /// and candidate index.
    stamp_gen: Vec<u32>,
    stamp_slot: Vec<u16>,
    gen: u32,
    /// The per-call buffers of `rush_dp_binary_fast_in`.
    rush_bufs: RushBufs,
}

/// Per-call buffers of `rush_dp_binary_fast_in` (every use re-initialises
/// them to the values a fresh `Vec` would hold).
#[derive(Default)]
struct RushBufs {
    proj_to_orig: Vec<usize>,
    proj_n: Vec<i64>,
    proj_c0: Vec<i64>,
    s: Vec<i64>,
    d: Vec<i64>,
    epb_suf: Vec<f64>,
    cum_n: Vec<i64>,
    cum_c0: Vec<i64>,
}

thread_local! {
    /// `exact_contiguous_reference_binary_candidates`'s
    /// interval loss / impurity tables.
    static GINI_REF_TABLES: std::cell::RefCell<(Vec<f64>, Vec<f64>)> =
        std::cell::RefCell::new((Vec::new(), Vec::new()));
}

thread_local! {
    static DP_SCRATCH: std::cell::RefCell<DpScratch> = std::cell::RefCell::new(DpScratch::default());
}

/// Run `f` with this thread's DP scratch (a fresh one if it is in use).
fn with_dp_scratch<R>(f: impl FnOnce(&mut DpScratch) -> R) -> R {
    DP_SCRATCH.with(|cell| match cell.try_borrow_mut() {
        Ok(mut g) => f(&mut g),
        Err(_) => f(&mut DpScratch::default()),
    })
}

/// Beam of arity `k` read by the ladder extraction (`rush_dp_binary` /
/// `collapse_bins_topb_binary_with_teacher_taxed_min_child`).
#[inline]
fn extraction_beam(beam_schedule: &[usize], k: usize) -> usize {
    beam_schedule
        .get(k - 2)
        .copied()
        .unwrap_or(1)
        .max(1)
        .min(BEAM_MAX)
}

/// `rush_dp_binary` (dp_reuse None) through the `opts` speed paths: the
/// same ladder bitwise and in order.
///
/// - topcell: no cell of layer m_max other than t = L is read (layer
///   m_max + 1 does not exist; reconstruction starts at the final cells and
///   steps to layer k − 1 at t' < t), and the minima channels advance in
///   the same k_ptr order whether or not the intermediate cells are filled.
/// - cap: the final cell (k, L) is read only by extraction (layer k + 1
///   reads cells t' < t <= L), which takes its first b_k entries; a stable
///   top-b_k list is the b_k-prefix of the stable top-BEAM_MAX list, and
///   the insert loop's `break` on rejection drops only entries that sort
///   after it. Interior cells keep BEAM_MAX (the channel merge is not
///   prefix-monotone).
/// - stamp: the scan in `upsert_rush_candidate` finds the (pt, pb) slot of
///   the current t; a slot table gives the same slot, the same first-seen
///   order and the same min update (at most 2·BEAM_MAX distinct inserts per
///   t, so the eviction branch never runs).
/// - reuse: only `count` / `best_rejected` are read before written, so a
///   reset table equals a fresh one; row k is stored at k − 1.
fn rush_dp_binary_fast(
    thresholds: &[f64],
    bin_n: &[i64],
    bin_c0: &[i64],
    max_k: usize,
    n_total: usize,
    beam_schedule: &[usize],
    min_support: usize,
    ub_best: f64,
    reg: f64,
    reg_alpha: f64,
    opts: DpOpts,
) -> (Vec<(usize, CollapseEntry)>, DpBeamStats) {
    with_dp_scratch(|sc| {
        rush_dp_binary_fast_in(
            thresholds,
            bin_n,
            bin_c0,
            max_k,
            n_total,
            beam_schedule,
            min_support,
            ub_best,
            reg,
            reg_alpha,
            opts,
            sc,
        )
    })
}

#[allow(clippy::too_many_arguments)]
fn rush_dp_binary_fast_in(
    thresholds: &[f64],
    bin_n: &[i64],
    bin_c0: &[i64],
    max_k: usize,
    n_total: usize,
    beam_schedule: &[usize],
    min_support: usize,
    ub_best: f64,
    reg: f64,
    reg_alpha: f64,
    opts: DpOpts,
    sc: &mut DpScratch,
) -> (Vec<(usize, CollapseEntry)>, DpBeamStats) {
    // The per-call buffers from the thread's scratch (taken out for the
    // call: `sc` is borrowed by the body too).
    let mut bufs = std::mem::take(&mut sc.rush_bufs);
    let out = rush_dp_binary_fast_bufs(
        thresholds,
        bin_n,
        bin_c0,
        max_k,
        n_total,
        beam_schedule,
        min_support,
        ub_best,
        reg,
        reg_alpha,
        opts,
        sc,
        &mut bufs,
    );
    sc.rush_bufs = bufs;
    out
}

/// `rush_dp_binary_fast_in`'s body over the buffers `bufs`.
#[allow(clippy::too_many_arguments)]
fn rush_dp_binary_fast_bufs(
    thresholds: &[f64],
    bin_n: &[i64],
    bin_c0: &[i64],
    max_k: usize,
    n_total: usize,
    beam_schedule: &[usize],
    min_support: usize,
    ub_best: f64,
    reg: f64,
    reg_alpha: f64,
    opts: DpOpts,
    sc: &mut DpScratch,
    bufs: &mut RushBufs,
) -> (Vec<(usize, CollapseEntry)>, DpBeamStats) {
    let l_orig = bin_n.len();
    if l_orig <= 1 {
        return (vec![], DpBeamStats::empty());
    }

    let m = min_support.max(1) as i64;
    let n_total_f = n_total as f64;
    let split_reg = (1.0 - reg_alpha) * reg;
    let leaf_reg = reg_alpha * reg;

    let RushBufs {
        proj_to_orig,
        proj_n,
        proj_c0,
        s,
        d,
        epb_suf,
        cum_n,
        cum_c0,
    } = bufs;
    proj_to_orig.clear();
    proj_n.clear();
    proj_c0.clear();
    for b in 0..l_orig {
        if bin_n[b] > 0 {
            proj_to_orig.push(b);
            proj_n.push(bin_n[b]);
            proj_c0.push(bin_c0[b]);
        }
    }

    let l = proj_n.len();
    if l <= 1 {
        return (vec![], DpBeamStats::empty());
    }

    s.clear();
    s.resize(l + 1, 0);
    d.clear();
    d.resize(l + 1, 0);
    for t in 0..l {
        s[t + 1] = s[t] + proj_n[t];
        d[t + 1] = d[t] + (2 * proj_c0[t] - proj_n[t]);
    }
    let total_support = s[l];
    if total_support == 0 {
        return (vec![], DpBeamStats::empty());
    }

    epb_suf.clear();
    epb_suf.resize(l + 1, 0.0);
    for t in (0..l).rev() {
        let gn = proj_n[t];
        let gc0 = proj_c0[t];
        let mis = (gn - gc0.max(gn - gc0)) as f64 / n_total_f;
        epb_suf[t] = epb_suf[t + 1] + mis;
    }

    let mut ub_local = ub_best;
    if ub_local.is_finite() {
        let total_c0: i64 = proj_c0.iter().sum();
        let mis_one = (total_support - total_c0.max(total_support - total_c0)) as f64 / n_total_f;
        ub_local = ub_local.min(mis_one + split_reg + 2.0 * leaf_reg);
    }

    let mut m_max = max_k.min(l);
    let support_cap = (total_support / m) as usize;
    m_max = m_max.min(support_cap);
    if ub_local.is_finite() && leaf_reg > 0.0 {
        let rhs = (ub_local - split_reg - epb_suf[0]) / leaf_reg;
        if rhs.is_finite() {
            let cap = rhs.floor() as isize;
            if cap < 2 {
                return (vec![], DpBeamStats::empty());
            }
            m_max = m_max.min(cap as usize);
        }
    }
    if m_max < 2 {
        return (vec![], DpBeamStats::empty());
    }

    let width = l + 1;
    // Row of layer k (1..=m_max) in the table: k - 1.
    let ix = |k: usize, t: usize| (k - 1) * width + t;
    let needed = m_max * width;
    let mut dp_tl = std::mem::take(&mut sc.rush);
    if dp_tl.len() < needed {
        dp_tl.resize(needed, RushTopB::new());
    }
    for e in dp_tl[..needed].iter_mut() {
        e.reset();
    }
    let dp: &mut [RushTopB] = &mut dp_tl[..needed];
    let slots = width * BEAM_MAX;
    if sc.stamp_gen.len() < slots {
        sc.stamp_gen.resize(slots, 0);
        sc.stamp_slot.resize(slots, 0);
    }

    for t in 1..=l {
        if s[t] < m {
            continue;
        }
        let dp2 = (s[t] - d[t].abs()) as f64;
        if ub_local.is_finite() {
            let reg_lb = if t < l {
                split_reg + 2.0 * leaf_reg
            } else {
                split_reg + leaf_reg
            };
            let obj_lb = dp2 / (2.0 * n_total_f) + epb_suf[t] + reg_lb;
            if obj_lb >= ub_local {
                continue;
            }
        }
        dp[ix(1, t)].try_insert(dp2, 0, 0);
    }

    for i in 2..=m_max {
        let mut a_top = RushTopB::new();
        let mut b_top = RushTopB::new();
        let mut k_ptr = i - 1;
        let top_layer = opts.topcell && i == m_max;
        let final_cap = if opts.cap {
            extraction_beam(beam_schedule, i)
        } else {
            BEAM_MAX
        };

        for t in i..=l {
            if s[t] < (i as i64) * m {
                continue;
            }
            let tau = s[t] - m;
            while k_ptr < t && s[k_ptr] <= tau {
                let prev = &dp[ix(i - 1, k_ptr)];
                a_top.merge_predecessor(prev, s[k_ptr] as f64, d[k_ptr] as f64, k_ptr as u16);
                b_top.merge_predecessor(prev, s[k_ptr] as f64, -(d[k_ptr] as f64), k_ptr as u16);
                k_ptr += 1;
            }
            if top_layer && t != l {
                continue;
            }

            if a_top.count == 0 && b_top.count == 0 {
                continue;
            }

            let st = s[t] as f64;
            let dt = d[t] as f64;
            let reg_lb = if t < l {
                split_reg + ((i + 1) as f64) * leaf_reg
            } else {
                split_reg + (i as f64) * leaf_reg
            };
            let cap = if t == l { final_cap } else { BEAM_MAX };
            merge_rush_channels(
                &mut dp[ix(i, t)],
                &a_top,
                &b_top,
                st,
                dt,
                cap,
                &mut sc.stamp_gen,
                &mut sc.stamp_slot,
                &mut sc.gen,
                n_total_f,
                epb_suf[t],
                reg_lb,
                ub_local,
            );
        }
    }
    // Beam statistics (exact unless topcell / cap; callers of those discard
    // them).
    let mut beam_stats = DpBeamStats::new(m_max);
    for k in 2..=m_max {
        let idx = k - 2;
        let fc = &dp[ix(k, l)];
        if fc.best_rejected < f64::INFINITY && fc.count > 0 {
            beam_stats.gap_at_final[idx] = fc.best_rejected - fc.costs[fc.count - 1];
        }
        for j in 0..=l {
            if dp[ix(k, j)].best_rejected < f64::INFINITY {
                beam_stats.n_saturated_cells[idx] += 1;
            }
        }
    }

    cum_n.clear();
    cum_n.resize(l_orig + 1, 0);
    cum_c0.clear();
    cum_c0.resize(l_orig + 1, 0);
    for i in 0..l_orig {
        cum_n[i + 1] = cum_n[i] + bin_n[i];
        cum_c0[i + 1] = cum_c0[i] + bin_c0[i];
    }

    let mut ladder = Vec::new();
    for k in 2..=m_max {
        let beam = extraction_beam(beam_schedule, k);
        let final_entry = &dp[ix(k, l)];
        for b in 0..final_entry.count.min(beam) {
            // `reconstruct_rush_path` with the row offset.
            let mut proj_boundaries = Vec::with_capacity(k - 1);
            {
                let (mut curr_k, mut curr_t, mut curr_b) = (k, l, b);
                while curr_k > 1 {
                    let e = &dp[ix(curr_k, curr_t)];
                    let pt = e.prev_t[curr_b] as usize;
                    let pb = e.prev_b[curr_b] as usize;
                    proj_boundaries.push(pt);
                    curr_t = pt;
                    curr_b = pb;
                    curr_k -= 1;
                }
                proj_boundaries.reverse();
            }
            if proj_boundaries.len() != k - 1 {
                continue;
            }

            let mut orig_starts = Vec::with_capacity(k - 1);
            let mut split_thresh = Vec::with_capacity(k - 1);
            let mut valid = true;
            for &pb in &proj_boundaries {
                if pb >= l {
                    valid = false;
                    break;
                }
                let orig_start = proj_to_orig[pb];
                if orig_start == 0 {
                    valid = false;
                    break;
                }
                orig_starts.push(orig_start);
                split_thresh.push(thresholds[orig_start - 1]);
            }
            if !valid {
                continue;
            }

            let starts = std::iter::once(0usize).chain(orig_starts.iter().copied());
            let ends = orig_starts.iter().copied().chain(std::iter::once(l_orig));

            let mut group_n = Vec::with_capacity(k);
            let mut group_counts = Vec::with_capacity(k);
            for (gs, ge) in starts.zip(ends) {
                let gn = cum_n[ge] - cum_n[gs];
                if gn < m {
                    valid = false;
                    break;
                }
                let gc0 = cum_c0[ge] - cum_c0[gs];
                group_n.push(gn);
                group_counts.push(vec![gc0, gn - gc0]);
            }
            if !valid {
                continue;
            }
            let reg_cost = final_entry.costs[b] / (2.0 * n_total_f) + (k as f64) * leaf_reg;
            ladder.push((
                k,
                CollapseEntry {
                    cost: reg_cost,
                    thresholds: Arc::from(split_thresh.as_slice()),
                    bin_n: group_n,
                    bin_counts: group_counts,
                    group_map: None,
                    order: CandidateOrder::Original,
                },
            ));
        }
    }
    sc.rush = dp_tl;
    (ladder, beam_stats)
}

/// `collapse_bins_topb_binary_min_child(.., criterion, None, min_child)`
/// (no reference ensemble, no tax, leaf_reg 0) through the `opts` speed paths, for
/// the criteria whose edge cost is `binary_cost` (Misclass / Gini): the
/// same ladder bitwise and in order.
///
/// TopB keeps the lex-smallest (cost, prev_i, prev_b) keys whatever the
/// insertion order, and for one start bin i the keys prev[b] + edge are
/// lex-increasing in b (fl-add is monotone, b breaks ties). So the top-c of
/// a cell uses only the top-c of each predecessor, and a list capped at c
/// is the c-prefix of the BEAM_MAX list (same entries, same backpointers).
/// - topcell: layer eff_max_k has no reader but its final cell.
/// - cap: the final cell (k, L−1) is read only by extraction (top b_k;
///   layer k + 1 reads cells j <= L − 2); interior cells of layer k feed
///   layers k' > k, which read at most max_{k'>k} b_k' from them.
/// - reuse: reset counts only; predecessor row borrowed (split_at_mut), not
///   copied; row k at k − 1.
fn collapse_binary_fast(
    thresholds: &[f64],
    bin_n: &[i64],
    bin_c0: &[i64],
    max_k: usize,
    n_total: usize,
    beam_schedule: &[usize],
    criterion: DpCriterion,
    min_child_size: usize,
    opts: DpOpts,
) -> (Vec<(usize, CollapseEntry)>, DpBeamStats) {
    debug_assert!(matches!(
        criterion,
        DpCriterion::Misclass | DpCriterion::Gini
    ));
    let l = bin_n.len();
    if l <= 1 {
        return (vec![], DpBeamStats::empty());
    }
    // Empty-bin projection, as the legacy function (which recurses).
    if bin_n.iter().any(|&n| n == 0) {
        let mut proj_thresholds = Vec::new();
        let mut proj_n = Vec::with_capacity(l);
        let mut proj_c0 = Vec::with_capacity(l);
        let mut seen = 0usize;
        for (orig_idx, &n) in bin_n.iter().enumerate() {
            if n <= 0 {
                continue;
            }
            if seen > 0 {
                let boundary = orig_idx.saturating_sub(1);
                if let Some(&thr) = thresholds.get(boundary) {
                    proj_thresholds.push(thr);
                } else {
                    return (vec![], DpBeamStats::empty());
                }
            }
            proj_n.push(n);
            proj_c0.push(bin_c0[orig_idx]);
            seen += 1;
        }
        if proj_n.len() <= 1 {
            return (vec![], DpBeamStats::empty());
        }
        return collapse_binary_fast(
            &proj_thresholds,
            &proj_n,
            &proj_c0,
            max_k,
            n_total,
            beam_schedule,
            criterion,
            min_child_size,
            opts,
        );
    }
    with_dp_scratch(|sc| {
        collapse_binary_fast_in(
            thresholds,
            bin_n,
            bin_c0,
            max_k,
            n_total,
            beam_schedule,
            criterion,
            min_child_size,
            opts,
            sc,
        )
    })
}

#[allow(clippy::too_many_arguments)]
fn collapse_binary_fast_in(
    thresholds: &[f64],
    bin_n: &[i64],
    bin_c0: &[i64],
    max_k: usize,
    n_total: usize,
    beam_schedule: &[usize],
    criterion: DpCriterion,
    min_child_size: usize,
    opts: DpOpts,
    sc: &mut DpScratch,
) -> (Vec<(usize, CollapseEntry)>, DpBeamStats) {
    let l = bin_n.len();
    let n_total_f = n_total as f64;
    let min_child = min_child_size.max(1) as i64;
    let total_support: i64 = bin_n.iter().sum();
    let support_cap = (total_support / min_child).max(0) as usize;
    let eff_max_k = max_k.min(l).min(support_cap);
    if eff_max_k < 2 {
        return (vec![], DpBeamStats::empty());
    }
    let misclass = matches!(criterion, DpCriterion::Misclass);
    let leaf_reg = 0.0f64;
    let edge_cost = |c0: i64, n: i64| -> f64 { binary_cost(c0, n, n_total_f, misclass) };

    // Caps: final cell of layer k at b_k, interior cells at max_{k'>k} b_k'.
    let mut cap_final = vec![BEAM_MAX; eff_max_k + 1];
    let mut cap_int = vec![BEAM_MAX; eff_max_k + 1];
    if opts.cap {
        let mut tail = 1usize;
        for k in (1..=eff_max_k).rev() {
            cap_int[k] = tail;
            cap_final[k] = if k >= 2 {
                extraction_beam(beam_schedule, k)
            } else {
                tail
            };
            if k >= 2 {
                tail = tail.max(cap_final[k]);
            }
        }
    }

    let needed = eff_max_k * l;
    let mut dp_tl = std::mem::take(&mut sc.collapse);
    if dp_tl.len() < needed {
        dp_tl.resize(needed, TopB::new());
    }
    for e in dp_tl[..needed].iter_mut() {
        e.reset();
    }
    let dp: &mut [TopB] = &mut dp_tl[..needed];
    let ix = |k: usize, j: usize| (k - 1) * l + j;

    {
        let (mut c0, mut n) = (0i64, 0i64);
        for j in 0..l {
            c0 += bin_c0[j];
            n += bin_n[j];
            if n >= min_child {
                let cost = edge_cost(c0, n) + leaf_reg;
                let cap = if j == l - 1 { cap_final[1] } else { cap_int[1] };
                dp[ix(1, j)].try_insert_cap(cost, 0, 0, cap);
            }
        }
    }

    for k in 2..=eff_max_k {
        let top_layer = opts.topcell && k == eff_max_k;
        let row_start = ix(k, 0);
        for i in (k - 1)..l {
            let (lo, hi) = dp.split_at_mut(row_start);
            let prev = &lo[ix(k - 1, i - 1)];
            let prev_count = prev.count;
            if prev_count == 0 {
                continue;
            }
            let prev_costs: &[f64] = &prev.costs[..prev_count];
            let (mut c0, mut n) = (0i64, 0i64);
            for j in i..l {
                c0 += bin_c0[j];
                n += bin_n[j];
                if n < min_child {
                    continue;
                }
                if top_layer && j != l - 1 {
                    continue;
                }
                let edge = edge_cost(c0, n) + leaf_reg;
                let cap = if j == l - 1 { cap_final[k] } else { cap_int[k] };
                let cur = &mut hi[j];
                for b in 0..prev_count {
                    if !cur.try_insert_cap(prev_costs[b] + edge, i as u16, b as u8, cap) {
                        break;
                    }
                }
            }
        }
    }

    let mut beam_stats = DpBeamStats::new(eff_max_k);
    for k in 2..=eff_max_k {
        let idx = k - 2;
        let fc = &dp[ix(k, l - 1)];
        if fc.best_rejected < f64::INFINITY && fc.count > 0 {
            beam_stats.gap_at_final[idx] = fc.best_rejected - fc.costs[fc.count - 1];
        }
        for j in 0..l {
            if dp[ix(k, j)].best_rejected < f64::INFINITY {
                beam_stats.n_saturated_cells[idx] += 1;
            }
        }
    }

    let mut cum_n = vec![0i64; l + 1];
    let mut cum_c0 = vec![0i64; l + 1];
    for i in 0..l {
        cum_n[i + 1] = cum_n[i] + bin_n[i];
        cum_c0[i + 1] = cum_c0[i] + bin_c0[i];
    }

    let mut ladder = Vec::new();
    for k in 2..=eff_max_k {
        let beam = extraction_beam(beam_schedule, k);
        let final_entry = &dp[ix(k, l - 1)];
        for b in 0..final_entry.count.min(beam) {
            // `reconstruct_path` with the row offset.
            let mut boundaries = Vec::with_capacity(k - 1);
            {
                let (mut curr_k, mut curr_j, mut curr_b) = (k, l - 1, b);
                while curr_k > 1 {
                    let e = &dp[ix(curr_k, curr_j)];
                    let pi = e.prev_i[curr_b] as usize;
                    let pb = e.prev_b[curr_b] as usize;
                    boundaries.push(pi);
                    curr_j = pi - 1;
                    curr_b = pb;
                    curr_k -= 1;
                }
                boundaries.reverse();
            }
            let split_thresh: Vec<f64> = boundaries.iter().map(|&bi| thresholds[bi - 1]).collect();
            let starts = std::iter::once(0).chain(boundaries.iter().copied());
            let ends = boundaries.iter().copied().chain(std::iter::once(l));
            let mut group_n = Vec::with_capacity(k);
            let mut group_counts = Vec::with_capacity(k);
            for (s, e) in starts.zip(ends) {
                let gn = cum_n[e] - cum_n[s];
                let gc0 = cum_c0[e] - cum_c0[s];
                group_n.push(gn);
                group_counts.push(vec![gc0, gn - gc0]);
            }
            ladder.push((
                k,
                CollapseEntry {
                    cost: final_entry.costs[b],
                    thresholds: Arc::from(split_thresh.as_slice()),
                    bin_n: group_n,
                    bin_counts: group_counts,
                    group_map: None,
                    order: CandidateOrder::Original,
                },
            ));
        }
    }
    sc.collapse = dp_tl;
    (ladder, beam_stats)
}

const CANDIDATE_DEDUP_EPS: f64 = 1e-12;

/// Fast heuristic Gini collapse for contiguous closure.
///
/// This emits a feasible partition and the caller recomputes its true cost, so
/// it is safe as an upper-bound closure candidate. It is not an exact interval
/// DP and must not be used for certified prefix search.
pub fn collapse_bins_heuristic_gini_binary(
    thresholds: &[f64],
    bin_n: &[i64],
    bin_c0: &[i64],
    max_k: usize,
    n_total: usize,
    leaf_reg: f64,
    min_child_size: usize,
) -> Vec<(usize, CollapseEntry)> {
    let l_orig = bin_n.len();
    if l_orig <= 1 {
        return Vec::new();
    }

    let mut proj_to_orig = Vec::with_capacity(l_orig);
    let mut pn = Vec::with_capacity(l_orig);
    let mut pc0 = Vec::with_capacity(l_orig);
    for b in 0..l_orig {
        if bin_n[b] > 0 {
            proj_to_orig.push(b);
            pn.push(bin_n[b]);
            pc0.push(bin_c0[b]);
        }
    }
    let l = pn.len();
    if l <= 1 {
        return Vec::new();
    }

    let min_child = min_child_size.max(1) as i64;
    let n_total_f = n_total as f64;
    let mut cum_n = vec![0i64; l + 1];
    let mut cum_c0 = vec![0i64; l + 1];
    for i in 0..l {
        cum_n[i + 1] = cum_n[i] + pn[i];
        cum_c0[i + 1] = cum_c0[i] + pc0[i];
    }
    let range_cost = |i: usize, j: usize| -> f64 {
        let n = cum_n[j + 1] - cum_n[i];
        let c0 = cum_c0[j + 1] - cum_c0[i];
        binary_cost(c0, n, n_total_f, false)
    };

    let total_support: i64 = pn.iter().sum();
    let eff_max_k = max_k
        .min(l)
        .min((total_support / min_child).max(0) as usize);
    if eff_max_k < 2 {
        return Vec::new();
    }

    let mut backptr = vec![vec![0usize; l]; eff_max_k + 1];
    let mut dp = vec![vec![f64::INFINITY; l]; eff_max_k + 1];
    for j in 0..l {
        if cum_n[j + 1] >= min_child {
            dp[1][j] = range_cost(0, j) + leaf_reg;
        }
    }

    for k in 2..=eff_max_k {
        let dp_prev = dp[k - 1].clone();
        let mut prev_opt = k - 1;
        for j in (k - 1)..l {
            let mut best_i = prev_opt;
            let mut best_cost = f64::INFINITY;
            for i in prev_opt..=j {
                if cum_n[j + 1] - cum_n[i] < min_child {
                    continue;
                }
                let prefix = dp_prev[i - 1];
                if prefix >= f64::INFINITY * 0.5 {
                    continue;
                }
                let cost = prefix + range_cost(i, j) + leaf_reg;
                if cost < best_cost {
                    best_cost = cost;
                    best_i = i;
                }
            }
            dp[k][j] = best_cost;
            backptr[k][j] = best_i;
            prev_opt = best_i;
        }
    }

    let mut cum_n_orig = vec![0i64; l_orig + 1];
    let mut cum_c0_orig = vec![0i64; l_orig + 1];
    for i in 0..l_orig {
        cum_n_orig[i + 1] = cum_n_orig[i] + bin_n[i];
        cum_c0_orig[i + 1] = cum_c0_orig[i] + bin_c0[i];
    }

    let mut ladder = Vec::new();
    for k in 2..=eff_max_k {
        if dp[k][l - 1] >= f64::INFINITY * 0.5 {
            continue;
        }
        let mut proj_boundaries = Vec::with_capacity(k - 1);
        let mut j = l - 1;
        for kk in (2..=k).rev() {
            let i = backptr[kk][j];
            proj_boundaries.push(i);
            j = i - 1;
        }
        proj_boundaries.reverse();

        let mut orig_starts = Vec::with_capacity(k - 1);
        let mut split_thresh = Vec::with_capacity(k - 1);
        let mut valid = true;
        for &pb in &proj_boundaries {
            if pb >= l {
                valid = false;
                break;
            }
            let orig_start = proj_to_orig[pb];
            if orig_start == 0 {
                valid = false;
                break;
            }
            orig_starts.push(orig_start);
            split_thresh.push(thresholds[orig_start - 1]);
        }
        if !valid {
            continue;
        }

        let starts = std::iter::once(0usize).chain(orig_starts.iter().copied());
        let ends = orig_starts.iter().copied().chain(std::iter::once(l_orig));
        let mut group_n = Vec::with_capacity(k);
        let mut group_counts = Vec::with_capacity(k);
        let mut groups_valid = true;
        for (gs, ge) in starts.zip(ends) {
            let gn = cum_n_orig[ge] - cum_n_orig[gs];
            if gn < min_child {
                groups_valid = false;
                break;
            }
            let gc0 = cum_c0_orig[ge] - cum_c0_orig[gs];
            group_n.push(gn);
            group_counts.push(vec![gc0, gn - gc0]);
        }
        if groups_valid {
            ladder.push((
                k,
                CollapseEntry {
                    cost: dp[k][l - 1],
                    thresholds: Arc::from(split_thresh.as_slice()),
                    bin_n: group_n,
                    bin_counts: group_counts,
                    group_map: None,
                    order: CandidateOrder::Original,
                },
            ));
        }
    }
    ladder
}

#[derive(Hash, Eq, PartialEq)]
struct CandidateDedupKey {
    feature_index: usize,
    k: usize,
    criterion: u8,
    threshold_bits: Vec<u64>,
    group_map: Option<Vec<usize>>,
}

fn quantize_threshold_bits(t: f64) -> u64 {
    if !t.is_finite() || t.abs() > 1e290 {
        return t.to_bits();
    }
    let q = ((t / CANDIDATE_DEDUP_EPS).round()) * CANDIDATE_DEDUP_EPS;
    if q == 0.0 {
        0.0f64.to_bits()
    } else {
        q.to_bits()
    }
}

fn dedup_split_candidates(cands: Vec<SplitCandidate>) -> Vec<SplitCandidate> {
    let mut best_by_key: HashMap<CandidateDedupKey, SplitCandidate> =
        HashMap::with_capacity(cands.len());
    for cand in cands {
        let key = CandidateDedupKey {
            feature_index: cand.feature_index,
            k: cand.k,
            criterion: cand.family & 1,
            threshold_bits: cand
                .thresholds
                .iter()
                .map(|&t| quantize_threshold_bits(t))
                .collect(),
            group_map: cand.group_map.as_ref().map(|gm| gm.to_vec()),
        };
        match best_by_key.get_mut(&key) {
            // Preserve insertion order for exact ShapeSPLINTER duplicates:
            // the standalone appends axis nominees before shape nominees and
            // drops later identical source variants.
            Some(_existing) => {}
            None => {
                best_by_key.insert(key, cand);
            }
        }
    }
    best_by_key.into_values().collect()
}

fn compare_f64_slices(lhs: &[f64], rhs: &[f64]) -> std::cmp::Ordering {
    for (a, b) in lhs.iter().zip(rhs.iter()) {
        let ord = a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal);
        if ord != std::cmp::Ordering::Equal {
            return ord;
        }
    }
    lhs.len().cmp(&rhs.len())
}

fn compare_usize_slices(lhs: &[usize], rhs: &[usize]) -> std::cmp::Ordering {
    for (a, b) in lhs.iter().zip(rhs.iter()) {
        let ord = a.cmp(b);
        if ord != std::cmp::Ordering::Equal {
            return ord;
        }
    }
    lhs.len().cmp(&rhs.len())
}

fn compare_i64_slices(lhs: &[i64], rhs: &[i64]) -> std::cmp::Ordering {
    for (a, b) in lhs.iter().zip(rhs.iter()) {
        let ord = a.cmp(b);
        if ord != std::cmp::Ordering::Equal {
            return ord;
        }
    }
    lhs.len().cmp(&rhs.len())
}

fn compare_i64_matrix(lhs: &[Vec<i64>], rhs: &[Vec<i64>]) -> std::cmp::Ordering {
    for (row_l, row_r) in lhs.iter().zip(rhs.iter()) {
        let ord = compare_i64_slices(row_l, row_r);
        if ord != std::cmp::Ordering::Equal {
            return ord;
        }
    }
    lhs.len().cmp(&rhs.len())
}

pub(crate) fn compare_split_candidates(
    lhs: &SplitCandidate,
    rhs: &SplitCandidate,
) -> std::cmp::Ordering {
    let shape_parity_order = lhs.group_map.is_some()
        || rhs.group_map.is_some()
        || lhs.family != CAND_FAMILY_PRIMARY
        || rhs.family != CAND_FAMILY_PRIMARY;
    if shape_parity_order {
        let lhs_source = (lhs.family >> 1) as usize;
        let rhs_source = (rhs.family >> 1) as usize;
        let lhs_criterion = (lhs.family & 1) as usize;
        let rhs_criterion = (rhs.family & 1) as usize;
        return lhs
            .rank_cost
            .partial_cmp(&rhs.rank_cost)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(lhs.components.cmp(&rhs.components))
            .then(lhs_source.cmp(&rhs_source))
            .then(lhs.feature_index.cmp(&rhs.feature_index))
            .then(lhs.k.cmp(&rhs.k))
            .then(lhs_criterion.cmp(&rhs_criterion))
            .then(compare_f64_slices(&lhs.thresholds, &rhs.thresholds))
            .then(compare_usize_slices(
                lhs.group_map.as_deref().unwrap_or(&[]),
                rhs.group_map.as_deref().unwrap_or(&[]),
            ))
            .then(compare_i64_slices(&lhs.child_n, &rhs.child_n))
            .then(compare_i64_matrix(&lhs.child_counts, &rhs.child_counts));
    }
    lhs.rank_cost
        .partial_cmp(&rhs.rank_cost)
        .unwrap_or(std::cmp::Ordering::Equal)
        // On cost-tie, prefer the
        // partition with FEWER original-axis components (more contiguous
        // in axis space). Axis-mode partitions have `components = k`;
        // shape-mode partitions have `components ≥ k`. This nudges the
        // pool toward less-fragmented choices when costs are equal.
        .then(lhs.components.cmp(&rhs.components))
        .then(lhs.k.cmp(&rhs.k))
        .then(lhs.feature_index.cmp(&rhs.feature_index))
        .then(compare_f64_slices(&lhs.thresholds, &rhs.thresholds))
        .then(compare_usize_slices(
            lhs.group_map.as_deref().unwrap_or(&[]),
            rhs.group_map.as_deref().unwrap_or(&[]),
        ))
        .then(compare_i64_slices(&lhs.child_n, &rhs.child_n))
        .then(compare_i64_matrix(&lhs.child_counts, &rhs.child_counts))
}

/// Compute the number of original-axis interval components in a
/// partition. For axis-mode (`group_map = None`), all atoms are in
/// original-bin order so the partition is `k` contiguous intervals →
/// `components = k`. For shape-mode, the `group_map[i]` entries are
/// indexed by ORIGINAL bin index; count branch-transitions along that
/// order to get the component count.
#[inline]
pub fn components_count(k: usize, group_map: Option<&[usize]>) -> u32 {
    match group_map {
        None => k as u32,
        Some(gm) if gm.is_empty() => k as u32,
        Some(gm) => {
            let mut count: u32 = 1;
            for w in gm.windows(2) {
                if w[0] != w[1] {
                    count += 1;
                }
            }
            count
        }
    }
}

#[derive(Clone)]
struct ExactBinaryNominee {
    k: usize,
    family: u8,
    thresholds: Arc<[f64]>,
    group_map: Option<Vec<usize>>,
    child_n: Vec<i64>,
    child_counts: Vec<Vec<i64>>,
    order: CandidateOrder,
    components: u32,
}

fn canonicalize_group_map(group_map: &mut [usize], k: usize) -> Option<Vec<usize>> {
    let mut remap = vec![usize::MAX; k];
    let mut next = 0usize;
    for group in group_map.iter_mut() {
        if *group >= k {
            return None;
        }
        if remap[*group] == usize::MAX {
            remap[*group] = next;
            next += 1;
        }
        *group = remap[*group];
    }
    if next == k {
        Some(remap)
    } else {
        None
    }
}

fn exact_interval_binary_candidate(
    thresholds: &[f64],
    bin_n: &[i64],
    bin_c0: &[i64],
    axis_bins: &[usize],
    full_bin_count: usize,
    order: &[usize],
    split_points: &[usize],
    family: u8,
    min_child_size: usize,
    emit_group_map: bool,
) -> Option<ExactBinaryNominee> {
    let l = bin_n.len();
    let k = split_points.len() + 1;
    let mut occupied_group_map = vec![usize::MAX; l];
    let mut child_n = vec![0i64; k];
    let mut child_counts = vec![vec![0i64; 2]; k];
    let mut starts = Vec::with_capacity(k);
    starts.push(0usize);
    starts.extend_from_slice(split_points);
    let ends = split_points.iter().copied().chain(std::iter::once(l));
    for (group, (start, end)) in starts.into_iter().zip(ends).enumerate() {
        if start >= end {
            return None;
        }
        for pos in start..end {
            let orig = order[pos];
            occupied_group_map[orig] = group;
            child_n[group] += bin_n[orig];
            child_counts[group][0] += bin_c0[orig];
            child_counts[group][1] += bin_n[orig] - bin_c0[orig];
        }
        if child_n[group] < min_child_size as i64 {
            return None;
        }
    }

    if emit_group_map {
        let remap = canonicalize_group_map(&mut occupied_group_map, k)?;
        if remap.iter().enumerate().any(|(old, &new)| old != new) {
            let mut new_n = vec![0i64; k];
            let mut new_counts = vec![vec![0i64; 2]; k];
            for old in 0..k {
                let new = remap[old];
                new_n[new] += child_n[old];
                new_counts[new][0] += child_counts[old][0];
                new_counts[new][1] += child_counts[old][1];
            }
            child_n = new_n;
            child_counts = new_counts;
        }
    }

    if child_n.iter().any(|&n| n <= 0) {
        return None;
    }
    let candidate_order = if emit_group_map {
        reference_order(&occupied_group_map)
    } else {
        CandidateOrder::Original
    };

    let mut components = 0u32;
    for group in 0..k {
        let mut values: Vec<usize> = axis_bins
            .iter()
            .enumerate()
            .filter_map(|(pos, &bin)| (occupied_group_map[pos] == group).then_some(bin))
            .collect();
        values.sort_unstable();
        if values.is_empty() {
            return None;
        }
        components += 1;
        for w in values.windows(2) {
            if w[1] != w[0] + 1 {
                components += 1;
            }
        }
    }

    let (thresholds, group_map, components) = if emit_group_map {
        let mut largest_child = 0usize;
        for child in 1..k {
            if child_n[child] > child_n[largest_child] {
                largest_child = child;
            }
        }
        let mut group_map = vec![largest_child; full_bin_count];
        for (pos, &bin) in axis_bins.iter().enumerate() {
            if bin < full_bin_count {
                group_map[bin] = occupied_group_map[pos];
            }
        }
        (Arc::from(thresholds), Some(group_map), components)
    } else {
        let mut split_thresholds = Vec::with_capacity(split_points.len());
        for &p in split_points {
            let left_orig = order[p - 1];
            let left_bin = axis_bins[left_orig];
            if left_bin >= thresholds.len() {
                return None;
            }
            // Match `exact_contiguous_reference_binary_candidates`: when a
            // split falls across empty training bins, choose the earliest
            // equivalent full-bin threshold, i.e. immediately after the last
            // occupied bin on the left. Training counts are identical across
            // the gap, but test-time routing is not.
            split_thresholds.push(thresholds[left_bin]);
        }
        (Arc::from(split_thresholds), None, k as u32)
    };

    Some(ExactBinaryNominee {
        k,
        family,
        thresholds,
        group_map,
        child_n,
        child_counts,
        order: candidate_order,
        components,
    })
}

fn exact_binary_order_nominees(
    thresholds: &[f64],
    bin_n: &[i64],
    bin_c0: &[i64],
    axis_bins: &[usize],
    full_bin_count: usize,
    order: &[usize],
    max_k: usize,
    beam_schedule: &[usize],
    min_child_size: usize,
    n_total: usize,
    mu_node: f64,
    shape_source: bool,
) -> Vec<ExactBinaryNominee> {
    let l = bin_n.len();
    let mut out = Vec::new();
    if l < 2 || max_k < 2 {
        return out;
    }
    let width = l + 1;
    let cell = |p: usize, t: usize| -> usize { p * width + t };
    let mut interval_loss = vec![f64::INFINITY; width * width];
    let mut interval_impurity = vec![f64::INFINITY; width * width];
    let mut neighbors = vec![Vec::<usize>::new(); l];
    if shape_source && mu_node > 0.0 {
        let mut pos_by_axis = vec![0usize; l];
        for (pos, &axis) in order.iter().enumerate() {
            pos_by_axis[axis] = pos;
        }
        for axis in 0..l.saturating_sub(1) {
            let a = pos_by_axis[axis];
            let b = pos_by_axis[axis + 1];
            neighbors[a].push(b);
            neighbors[b].push(a);
        }
    }
    let n_total_f = n_total as f64;
    let mut active = vec![false; l];
    for t in 1..=l {
        let mut n = 0i64;
        let mut c0 = 0i64;
        let mut boundary_edge_weight = 0.0;
        if shape_source && mu_node > 0.0 {
            active.fill(false);
        }
        for p in (0..t).rev() {
            let orig = order[p];
            n += bin_n[orig];
            c0 += bin_c0[orig];
            if shape_source && mu_node > 0.0 {
                active[p] = true;
                for &nb in &neighbors[p] {
                    if active[nb] {
                        boundary_edge_weight -= mu_node;
                    } else {
                        boundary_edge_weight += mu_node;
                    }
                }
            }
            if n < min_child_size as i64 {
                continue;
            }
            let c1 = n - c0;
            let n_f = n as f64;
            let boundary = if shape_source {
                0.5 * boundary_edge_weight
            } else {
                0.0
            };
            let idx = cell(p, t);
            interval_loss[idx] = (c0.min(c1) as f64) / n_total_f + boundary;
            interval_impurity[idx] =
                (n_f - ((c0 * c0 + c1 * c1) as f64) / n_f) / n_total_f + boundary;
        }
    }
    let families = if shape_source {
        (CAND_FAMILY_PRIMARY, CAND_FAMILY_MISCLASS)
    } else {
        (CAND_FAMILY_ALT_MODE_PRIMARY, CAND_FAMILY_ALT_MODE_MISCLASS)
    };
    let beam_for = |k: usize| -> usize {
        beam_schedule
            .get(k.saturating_sub(2))
            .copied()
            .unwrap_or(1)
            .max(1)
    };
    fn push_top2(v: &mut Vec<(f64, usize)>, item: (f64, usize), beam: usize) {
        let better = |a: &(f64, usize), b: &(f64, usize)| {
            a.0 < b.0 - 1e-12 || ((a.0 - b.0).abs() <= 1e-12 && a.1 < b.1)
        };
        if v.len() >= beam && !better(&item, v.last().unwrap()) {
            return;
        }
        let pos = v.iter().position(|x| better(&item, x)).unwrap_or(v.len());
        if pos < beam {
            v.insert(pos, item);
            if v.len() > beam {
                v.pop();
            }
        } else if v.len() < beam {
            v.push(item);
        }
    }
    fn push_top3(v: &mut Vec<(f64, usize, usize)>, item: (f64, usize, usize), beam: usize) {
        let better = |a: &(f64, usize, usize), b: &(f64, usize, usize)| {
            a.0 < b.0 - 1e-12 || ((a.0 - b.0).abs() <= 1e-12 && (a.2, a.1) < (b.2, b.1))
        };
        if v.len() >= beam && !better(&item, v.last().unwrap()) {
            return;
        }
        let pos = v.iter().position(|x| better(&item, x)).unwrap_or(v.len());
        if pos < beam {
            v.insert(pos, item);
            if v.len() > beam {
                v.pop();
            }
        } else if v.len() < beam {
            v.push(item);
        }
    }
    let beam2 = beam_for(2);
    let beam3 = beam_for(3);
    let mut k2_impurity: Vec<(f64, usize)> = Vec::new();
    let mut k2_loss: Vec<(f64, usize)> = Vec::new();
    for p in 1..l {
        let imp = interval_impurity[cell(0, p)] + interval_impurity[cell(p, l)];
        if imp.is_finite() {
            push_top2(&mut k2_impurity, (imp, p), beam2);
        }
        let loss = interval_loss[cell(0, p)] + interval_loss[cell(p, l)];
        if loss.is_finite() {
            push_top2(&mut k2_loss, (loss, p), beam2);
        }
    }
    for &(_, p) in &k2_impurity {
        if let Some(cand) = exact_interval_binary_candidate(
            thresholds,
            bin_n,
            bin_c0,
            axis_bins,
            full_bin_count,
            order,
            &[p],
            families.0,
            min_child_size,
            shape_source,
        ) {
            out.push(cand);
        }
    }
    for &(_, p) in &k2_loss {
        if let Some(cand) = exact_interval_binary_candidate(
            thresholds,
            bin_n,
            bin_c0,
            axis_bins,
            full_bin_count,
            order,
            &[p],
            families.1,
            min_child_size,
            shape_source,
        ) {
            out.push(cand);
        }
    }

    if max_k >= 3 && l >= 3 {
        let mut k3_impurity: Vec<(f64, usize, usize)> = Vec::new();
        let mut k3_loss: Vec<(f64, usize, usize)> = Vec::new();
        for p1 in 1..l {
            for p2 in p1 + 1..l {
                let imp = interval_impurity[cell(0, p1)]
                    + interval_impurity[cell(p1, p2)]
                    + interval_impurity[cell(p2, l)];
                if imp.is_finite() {
                    push_top3(&mut k3_impurity, (imp, p1, p2), beam3);
                }
                let loss = interval_loss[cell(0, p1)]
                    + interval_loss[cell(p1, p2)]
                    + interval_loss[cell(p2, l)];
                if loss.is_finite() {
                    push_top3(&mut k3_loss, (loss, p1, p2), beam3);
                }
            }
        }
        for &(_, p1, p2) in &k3_impurity {
            if let Some(cand) = exact_interval_binary_candidate(
                thresholds,
                bin_n,
                bin_c0,
                axis_bins,
                full_bin_count,
                order,
                &[p1, p2],
                families.0,
                min_child_size,
                shape_source,
            ) {
                out.push(cand);
            }
        }
        for &(_, p1, p2) in &k3_loss {
            if let Some(cand) = exact_interval_binary_candidate(
                thresholds,
                bin_n,
                bin_c0,
                axis_bins,
                full_bin_count,
                order,
                &[p1, p2],
                families.1,
                min_child_size,
                shape_source,
            ) {
                out.push(cand);
            }
        }
    }
    out
}

fn exact_contiguous_reference_binary_candidates(
    thresholds: &[f64],
    bin_n: &[i64],
    bin_c0: &[i64],
    feature_index: usize,
    config: &SolverConfig,
    beam_schedule: &[usize],
    n_total: usize,
    path_cost: f64,
    leaf_objective: Option<f64>,
) -> Vec<SplitCandidate> {
    let l = bin_n.len();
    if l < 2 || thresholds.len() + 1 != l || config.max_k < 2 {
        return Vec::new();
    }

    let width = l + 1;
    let cell = |p: usize, t: usize| -> usize { p * width + t };
    // The two tables from per-thread scratch, refilled with the same
    // values a fresh table holds.
    let (mut interval_loss, mut interval_impurity) =
        GINI_REF_TABLES.with(|c| std::mem::take(&mut *c.borrow_mut()));
    interval_loss.clear();
    interval_loss.resize(width * width, f64::INFINITY);
    interval_impurity.clear();
    interval_impurity.resize(width * width, f64::INFINITY);
    let n_total_f = n_total as f64;
    let min_child_size = config.min_samples_leaf.max(1) as i64;

    for t in 1..=l {
        let mut n = 0i64;
        let mut c0 = 0i64;
        for p in (0..t).rev() {
            n += bin_n[p];
            c0 += bin_c0[p];
            if n < min_child_size {
                continue;
            }
            let c1 = n - c0;
            let n_f = n as f64;
            let idx = cell(p, t);
            interval_loss[idx] = (c0.min(c1) as f64) / n_total_f;
            interval_impurity[idx] = (n_f - ((c0 * c0 + c1 * c1) as f64) / n_f) / n_total_f;
        }
    }

    fn push_top2(v: &mut Vec<(f64, usize)>, item: (f64, usize), beam: usize) {
        let better = |a: &(f64, usize), b: &(f64, usize)| {
            a.0 < b.0 - 1e-12 || ((a.0 - b.0).abs() <= 1e-12 && a.1 < b.1)
        };
        if v.len() >= beam && !better(&item, v.last().unwrap()) {
            return;
        }
        let pos = v.iter().position(|x| better(&item, x)).unwrap_or(v.len());
        if pos < beam {
            v.insert(pos, item);
            if v.len() > beam {
                v.pop();
            }
        } else if v.len() < beam {
            v.push(item);
        }
    }
    fn push_top3(v: &mut Vec<(f64, usize, usize)>, item: (f64, usize, usize), beam: usize) {
        let better = |a: &(f64, usize, usize), b: &(f64, usize, usize)| {
            a.0 < b.0 - 1e-12 || ((a.0 - b.0).abs() <= 1e-12 && (a.2, a.1) < (b.2, b.1))
        };
        if v.len() >= beam && !better(&item, v.last().unwrap()) {
            return;
        }
        let pos = v.iter().position(|x| better(&item, x)).unwrap_or(v.len());
        if pos < beam {
            v.insert(pos, item);
            if v.len() > beam {
                v.pop();
            }
        } else if v.len() < beam {
            v.push(item);
        }
    }

    let beam_for = |k: usize| -> usize {
        beam_schedule
            .get(k.saturating_sub(2))
            .copied()
            .unwrap_or(1)
            .max(1)
    };
    let split_reg = (1.0 - config.reg_alpha) * config.reg;
    let leaf_reg = config.reg_alpha * config.reg;
    let max_k = config.max_k.min(l).min(3);

    let mut out = Vec::new();
    let mut emit = |split_points: &[usize], family: u8| {
        let k = split_points.len() + 1;
        let min_split_overhead = split_reg + path_cost + (k as f64) * leaf_reg;
        if leaf_objective.is_some_and(|leaf| min_split_overhead + 1e-12 >= leaf) {
            return;
        }

        let mut starts = Vec::with_capacity(k);
        starts.push(0usize);
        starts.extend_from_slice(split_points);
        let ends = split_points.iter().copied().chain(std::iter::once(l));
        let mut child_n = Vec::with_capacity(k);
        let mut child_counts = Vec::with_capacity(k);

        for (start, end) in starts.into_iter().zip(ends) {
            let mut n = 0i64;
            let mut c0 = 0i64;
            for b in start..end {
                n += bin_n[b];
                c0 += bin_c0[b];
            }
            if n < min_child_size {
                return;
            }
            let c1 = n - c0;
            child_n.push(n);
            child_counts.push(vec![c0, c1]);
        }

        let split_thresholds: Vec<f64> = split_points.iter().map(|&p| thresholds[p - 1]).collect();
        let (rank_cost, split_ub) = score_candidate(
            &child_n,
            &child_counts,
            config,
            n_total,
            path_cost,
            CandidateOrder::Original,
        );
        out.push(SplitCandidate {
            rank_cost,
            split_ub,
            k,
            feature_index,
            thresholds: config.threshold_interner.intern_slice(&split_thresholds),
            group_map: None,
            child_n,
            child_counts,
            family,
            components: k as u32,
        });
    };

    let beam2 = beam_for(2);
    let mut k2_impurity: Vec<(f64, usize)> = Vec::new();
    let mut k2_loss: Vec<(f64, usize)> = Vec::new();
    for p in 1..l {
        let imp = interval_impurity[cell(0, p)] + interval_impurity[cell(p, l)];
        if imp.is_finite() {
            push_top2(&mut k2_impurity, (imp, p), beam2);
        }
        let loss = interval_loss[cell(0, p)] + interval_loss[cell(p, l)];
        if loss.is_finite() {
            push_top2(&mut k2_loss, (loss, p), beam2);
        }
    }
    for &(_, p) in &k2_impurity {
        emit(&[p], CAND_FAMILY_PRIMARY);
    }
    for &(_, p) in &k2_loss {
        emit(&[p], CAND_FAMILY_MISCLASS);
    }

    if max_k >= 3 {
        let beam3 = beam_for(3);
        let mut k3_impurity: Vec<(f64, usize, usize)> = Vec::new();
        let mut k3_loss: Vec<(f64, usize, usize)> = Vec::new();
        for p1 in 1..l {
            for p2 in p1 + 1..l {
                let imp = interval_impurity[cell(0, p1)]
                    + interval_impurity[cell(p1, p2)]
                    + interval_impurity[cell(p2, l)];
                if imp.is_finite() {
                    push_top3(&mut k3_impurity, (imp, p1, p2), beam3);
                }
                let loss = interval_loss[cell(0, p1)]
                    + interval_loss[cell(p1, p2)]
                    + interval_loss[cell(p2, l)];
                if loss.is_finite() {
                    push_top3(&mut k3_loss, (loss, p1, p2), beam3);
                }
            }
        }
        for &(_, p1, p2) in &k3_impurity {
            emit(&[p1, p2], CAND_FAMILY_PRIMARY);
        }
        for &(_, p1, p2) in &k3_loss {
            emit(&[p1, p2], CAND_FAMILY_MISCLASS);
        }
    }

    GINI_REF_TABLES.with(|c| *c.borrow_mut() = (interval_loss, interval_impurity));
    out
}

fn exact_shape_reference_binary_candidates(
    thresholds: &[f64],
    bin_n: &[i64],
    bin_c0: &[i64],
    teacher_c0: Option<&[f64]>,
    feature_index: usize,
    config: &SolverConfig,
    beam_schedule: &[usize],
    n_total: usize,
    path_cost: f64,
    leaf_objective: Option<f64>,
) -> Vec<SplitCandidate> {
    let l = bin_n.len();
    if l < 2 || thresholds.len() + 1 != l {
        return Vec::new();
    }
    let occupied: Vec<usize> = (0..l).filter(|&i| bin_n[i] > 0).collect();
    if occupied.len() < 2 {
        return Vec::new();
    }
    let occ_n: Vec<i64> = occupied.iter().map(|&i| bin_n[i]).collect();
    let occ_c0: Vec<i64> = occupied.iter().map(|&i| bin_c0[i]).collect();
    let occ_teacher: Option<Vec<f64>> = teacher_c0
        .filter(|t| t.len() == l)
        .map(|t| occupied.iter().map(|&i| t[i]).collect());
    let max_k = config.max_k.min(occupied.len()).min(3);
    // The boundary tax uses the global uniform sample weight at every
    // node, not 1 / node_size.
    let mu_node = if config.msdt.shape_boundary_tax > 0.0 {
        config.msdt.shape_boundary_tax / (n_total.max(1) as f64)
    } else {
        0.0
    };
    let mut nominees = Vec::new();
    let axis_order: Vec<usize> = (0..occupied.len()).collect();
    if shape_includes_contiguous_baseline(config) {
        nominees.extend(exact_binary_order_nominees(
            thresholds,
            &occ_n,
            &occ_c0,
            &occupied,
            l,
            &axis_order,
            max_k,
            beam_schedule,
            config.min_samples_leaf,
            n_total,
            0.0,
            false,
        ));
    }
    let mut shape_order: Vec<usize> = (0..occupied.len()).collect();
    if let Some(t) = occ_teacher.as_deref() {
        shape_order.sort_by(|&a, &b| {
            let pa = t[a] / occ_n[a] as f64;
            let pb = t[b] / occ_n[b] as f64;
            let ka = (((2.0 * pa - 1.0) / 1.0e-12).round()) as i64;
            let kb = (((2.0 * pb - 1.0) / 1.0e-12).round()) as i64;
            ka.cmp(&kb)
                .then(occupied[a].cmp(&occupied[b]))
                .then(a.cmp(&b))
        });
    } else {
        shape_order.sort_by(|&a, &b| {
            let ka = occ_c0[a] as f64 / occ_n[a] as f64;
            let kb = occ_c0[b] as f64 / occ_n[b] as f64;
            ka.partial_cmp(&kb)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(occupied[a].cmp(&occupied[b]))
                .then(a.cmp(&b))
        });
    }
    nominees.extend(exact_binary_order_nominees(
        thresholds,
        &occ_n,
        &occ_c0,
        &occupied,
        l,
        &shape_order,
        max_k,
        beam_schedule,
        config.min_samples_leaf,
        n_total,
        mu_node,
        true,
    ));

    let mut out = Vec::new();
    for n in nominees {
        if let Some(leaf) = leaf_objective {
            if config.reg * n.k as f64 + 1e-12 >= leaf {
                continue;
            }
        }
        let (rank_cost, split_ub) = score_candidate(
            &n.child_n,
            &n.child_counts,
            config,
            n_total,
            path_cost,
            n.order,
        );
        out.push(SplitCandidate {
            rank_cost,
            split_ub,
            k: n.k,
            feature_index,
            thresholds: n.thresholds,
            group_map: n.group_map,
            child_n: n.child_n,
            child_counts: n.child_counts,
            family: n.family,
            components: n.components,
        });
    }
    out
}

/// Top-k candidate beam with anchor preservation.
///
/// Assumes `all` is sorted by `compare_split_candidates` (ascending `rank_cost`).
/// Takes the `cap`-prefix then, for each family tag not yet present in the
/// prefix, splices in that family's best-ranked candidate (anchor rule).
///
/// Disabled when `config.rashomon_mode` is true, because Rashomon
/// enumeration needs the full candidate pool.
///
/// When no explicit cap is set, applies a ⌈√N⌉ default cap iff any
/// `MsdtPortConfig` knob is active. Respects
/// `top_k_min_beam` as a floor so small subproblems aren't strangled.
///
/// Returns `(kept, n_truncated)` where `n_truncated` is the number of
/// candidates dropped (for diagnostic accounting).
pub fn select_top_k_with_anchors(
    all: Vec<SplitCandidate>,
    config: &SolverConfig,
    n_total: usize,
) -> (Vec<SplitCandidate>, u64) {
    if config.rashomon_mode {
        return (all, 0);
    }
    let explicit_cap = config.msdt.top_k;
    let auto_sqrt = explicit_cap.is_none() && config.msdt.is_active();
    if explicit_cap.is_none() && !auto_sqrt {
        return (all, 0);
    }
    // Scale the auto-√N cap by the number of candidate families so each
    // family (mode × criterion) gets roughly √N slots in the prefix.
    // Explicit caps are honored as-is — user intent overrides heuristic.
    //
    // Auto basis selection:
    //   default (auto_cap_use_sqrt_candidates=false): √(n_rows), e.g. ~100
    //     for 10k rows (little truncation).
    //   true: √(candidate_count) — caps per-subproblem pool at the square
    //     root of the actual emitted candidate set, e.g. ~14 for a 184-pool.
    //     A genuinely tighter "principled" cap that scales with the search
    //     breadth at the current node, not with dataset size.
    let n_families = candidate_family_count(config);
    let auto_basis: f64 = if config.msdt.top_k_auto_sqrt_candidates {
        all.len() as f64
    } else {
        n_total as f64
    };
    let raw_cap =
        explicit_cap.unwrap_or_else(|| (n_families as f64 * auto_basis.sqrt()).ceil() as usize);
    let cap = raw_cap.max(config.msdt.top_k_min_beam);
    if all.len() <= cap {
        return (all, 0);
    }

    let mut anchor_indices = Vec::new();
    let mut seen_families: u64 = 0;
    for (i, c) in all.iter().enumerate() {
        let fam_bit = 1u64 << (c.family as u32);
        if (seen_families & fam_bit) == 0 {
            seen_families |= fam_bit;
            anchor_indices.push(i);
        }
    }
    let budget = cap.min(all.len());
    let mut keep_indices: Vec<usize> = (0..budget).collect();
    for &anchor in &anchor_indices {
        if keep_indices.contains(&anchor) {
            continue;
        }
        let mut replace = keep_indices.len().saturating_sub(1);
        while replace > 0 && anchor_indices.contains(&keep_indices[replace]) {
            replace -= 1;
        }
        if anchor_indices.contains(&keep_indices[replace]) {
            continue;
        }
        keep_indices[replace] = anchor;
    }
    keep_indices.sort_unstable();
    keep_indices.dedup();
    let n_truncated = (all.len() - keep_indices.len()) as u64;
    let mut next_keep = keep_indices.into_iter().peekable();
    let kept: Vec<SplitCandidate> = all
        .into_iter()
        .enumerate()
        .filter_map(|(i, c)| {
            if next_keep.peek().copied() == Some(i) {
                next_keep.next();
                Some(c)
            } else {
                None
            }
        })
        .collect();
    (kept, n_truncated)
}

/// Top-B DP for binary classification (n_classes == 2).
///
/// Uses loop-swapped DP (i-outer, j-inner) to accumulate c0/n in CPU registers
/// instead of prefix-sum array lookups, eliminating cache misses in the inner loop.
/// The early-break on the b-loop makes it amortized O(1) per (i,j) pair once
/// dp[k][j] fills up.
///
/// Time: O(max_k * L^2) amortized.  Space: O(max_k * L * BEAM_MAX).
/// Backward-compatible wrapper that preserves the pre-blend
/// signature. Existing call sites (contig paths, secondary shape paths,
/// and any criteria where the reference-ensemble blend is out of scope) stay
/// untouched. The primary shape-mode site calls
/// `collapse_bins_topb_binary_with_teacher` directly.
#[inline]
pub fn collapse_bins_topb_binary(
    thresholds: &[f64],
    bin_n: &[i64],
    bin_c0: &[i64],
    max_k: usize,
    n_total: usize,
    beam_schedule: &[usize],
    criterion: DpCriterion,
    similar_lb: Option<&HashMap<SupportSig, f64>>,
) -> (Vec<(usize, CollapseEntry)>, DpBeamStats) {
    // leaf_reg = 0.0 (set by `collapse_bins_topb_binary_min_child`).
    collapse_bins_topb_binary_min_child(
        thresholds,
        bin_n,
        bin_c0,
        max_k,
        n_total,
        beam_schedule,
        criterion,
        similar_lb,
        1,
    )
}

pub fn collapse_bins_topb_binary_min_child(
    thresholds: &[f64],
    bin_n: &[i64],
    bin_c0: &[i64],
    max_k: usize,
    n_total: usize,
    beam_schedule: &[usize],
    criterion: DpCriterion,
    similar_lb: Option<&HashMap<SupportSig, f64>>,
    min_child_size: usize,
) -> (Vec<(usize, CollapseEntry)>, DpBeamStats) {
    collapse_bins_topb_binary_with_teacher_min_child(
        thresholds,
        bin_n,
        bin_c0,
        max_k,
        n_total,
        beam_schedule,
        criterion,
        similar_lb,
        None,
        1.0,
        0.0,
        min_child_size,
    )
}

/// Reference-ensemble-aware binary top-B DP.
///
/// When `teacher_c0.is_some() && teacher_alpha < 1.0` and
/// `criterion == DpCriterion::Gini`, the per-edge cost blends hard
/// Gini with a reference-ensemble-weighted soft Gini:
/// `α·hard + (1-α)·soft`. For any other combination of inputs, the
/// output is bit-identical to the pre-blend behavior.
pub fn collapse_bins_topb_binary_with_teacher(
    thresholds: &[f64],
    bin_n: &[i64],
    bin_c0: &[i64],
    max_k: usize,
    n_total: usize,
    beam_schedule: &[usize],
    criterion: DpCriterion,
    similar_lb: Option<&HashMap<SupportSig, f64>>,
    teacher_c0: Option<&[f64]>,
    teacher_alpha: f64,
    // Per-leaf regularization included in the DP
    // objective. Each group's edge cost becomes `φ(group) + leaf_reg`,
    // so a k-way partition's DP cost is `Σ φ(groups) + k·leaf_reg`,
    // matching the BnB's regularized objective exactly. Pass 0.0 to
    // leave regularization out of the DP (caller adds reg post-DP). When
    // non-zero, the DP's top-B ordering is consistent with BnB's
    // candidate ordering across different k.
    leaf_reg: f64,
) -> (Vec<(usize, CollapseEntry)>, DpBeamStats) {
    collapse_bins_topb_binary_with_teacher_min_child(
        thresholds,
        bin_n,
        bin_c0,
        max_k,
        n_total,
        beam_schedule,
        criterion,
        similar_lb,
        teacher_c0,
        teacher_alpha,
        leaf_reg,
        1,
    )
}

pub fn collapse_bins_topb_binary_with_teacher_min_child(
    thresholds: &[f64],
    bin_n: &[i64],
    bin_c0: &[i64],
    max_k: usize,
    n_total: usize,
    beam_schedule: &[usize],
    criterion: DpCriterion,
    similar_lb: Option<&HashMap<SupportSig, f64>>,
    teacher_c0: Option<&[f64]>,
    teacher_alpha: f64,
    leaf_reg: f64,
    min_child_size: usize,
) -> (Vec<(usize, CollapseEntry)>, DpBeamStats) {
    collapse_bins_topb_binary_with_teacher_taxed_min_child(
        thresholds,
        bin_n,
        bin_c0,
        max_k,
        n_total,
        beam_schedule,
        criterion,
        similar_lb,
        teacher_c0,
        teacher_alpha,
        leaf_reg,
        None,
        min_child_size,
    )
}

pub fn collapse_bins_topb_binary_with_teacher_taxed_min_child(
    thresholds: &[f64],
    bin_n: &[i64],
    bin_c0: &[i64],
    max_k: usize,
    n_total: usize,
    beam_schedule: &[usize],
    criterion: DpCriterion,
    similar_lb: Option<&HashMap<SupportSig, f64>>,
    teacher_c0: Option<&[f64]>,
    teacher_alpha: f64,
    leaf_reg: f64,
    interval_tax: Option<&[f64]>,
    min_child_size: usize,
) -> (Vec<(usize, CollapseEntry)>, DpBeamStats) {
    let l = bin_n.len();
    if l <= 1 {
        return (vec![], DpBeamStats::empty());
    }

    // LightGBM threshold caches are global, while recursive nodes often cover
    // only a small subset of the feature range. Empty bins do not affect any
    // feasible sample partition, but leaving them in the DP makes the generic
    // k>3 path quadratic in the global threshold count. Project to non-empty
    // bins and keep the threshold immediately before each projected bin as the
    // split boundary. This is exact for contiguous sample partitions.
    if interval_tax.is_none() && bin_n.iter().any(|&n| n == 0) {
        let mut proj_thresholds = Vec::new();
        let mut proj_n = Vec::with_capacity(l);
        let mut proj_c0 = Vec::with_capacity(l);
        let mut proj_teacher = if teacher_c0.is_some() {
            Some(Vec::with_capacity(l))
        } else {
            None
        };
        let mut seen = 0usize;
        for (orig_idx, &n) in bin_n.iter().enumerate() {
            if n <= 0 {
                continue;
            }
            if seen > 0 {
                let boundary = orig_idx.saturating_sub(1);
                if let Some(&thr) = thresholds.get(boundary) {
                    proj_thresholds.push(thr);
                } else {
                    return (vec![], DpBeamStats::empty());
                }
            }
            proj_n.push(n);
            proj_c0.push(bin_c0[orig_idx]);
            if let (Some(src), Some(dst)) = (teacher_c0, proj_teacher.as_mut()) {
                if let Some(&v) = src.get(orig_idx) {
                    dst.push(v);
                } else {
                    return (vec![], DpBeamStats::empty());
                }
            }
            seen += 1;
        }
        if proj_n.len() <= 1 {
            return (vec![], DpBeamStats::empty());
        }
        return collapse_bins_topb_binary_with_teacher_taxed_min_child(
            &proj_thresholds,
            &proj_n,
            &proj_c0,
            max_k,
            n_total,
            beam_schedule,
            criterion,
            similar_lb,
            proj_teacher.as_deref(),
            teacher_alpha,
            leaf_reg,
            None,
            min_child_size,
        );
    }

    let n_total_f = n_total as f64;
    let min_child = min_child_size.max(1) as i64;
    let total_support: i64 = bin_n.iter().sum();
    let support_cap = (total_support / min_child).max(0) as usize;
    let eff_max_k = max_k.min(l).min(support_cap);
    if eff_max_k < 2 {
        return (vec![], DpBeamStats::empty());
    }
    let mut prefix_n = vec![0i64; l + 1];
    for i in 0..l {
        prefix_n[i + 1] = prefix_n[i] + bin_n[i];
    }

    // Depth2: precompute subsplit table
    let subsplit_table = match criterion {
        DpCriterion::Depth2 => Some(precompute_subsplit_binary(bin_n, bin_c0, n_total_f)),
        DpCriterion::Depth2Misclass => Some(precompute_subsplit_binary_misclass(
            bin_n, bin_c0, n_total_f,
        )),
        _ => None,
    };

    // Wasserstein: precompute node class-0 proportion
    let node_p0: f64 = if criterion == DpCriterion::Wasserstein {
        let total_c0: i64 = bin_c0.iter().sum();
        let total_n: i64 = bin_n.iter().sum();
        if total_n > 0 {
            total_c0 as f64 / total_n as f64
        } else {
            0.5
        }
    } else {
        0.0
    };

    // Reference-ensemble blend gate. Blend is active only when:
    //   * teacher_c0 is provided with the right length,
    //   * alpha < 1.0 (at 1.0, hard-only is exact — skip the arithmetic),
    //   * and the outer logic is using the Gini criterion (other criteria
    //     have their own geometric assumptions; v1 scopes to Gini).
    //
    // The final criterion gate is evaluated per-edge inside the closure
    // below so that non-Gini callers keep identical numeric output even
    // when a reference-ensemble slice is passed in.
    let teacher_len_ok = teacher_c0.map_or(false, |s| s.len() == l);
    let use_teacher = teacher_len_ok && teacher_alpha < 1.0 - 1e-12;
    // Empty slice sentinel for the plain (no reference ensemble) path — never indexed.
    let teacher_slice: &[f64] = teacher_c0.filter(|_| use_teacher).unwrap_or(&[]);

    // Edge cost dispatch closure. `t_c0` is the accumulated reference-ensemble class-0
    // mass over the same bin range as `(c0, n)`; ignored when
    // `use_teacher == false` or criterion != Gini.
    let edge_cost = |c0: i64, n: i64, t_c0: f64| -> f64 {
        let hard = match criterion {
            DpCriterion::Misclass => binary_cost(c0, n, n_total_f, true),
            DpCriterion::Entropy => binary_cost_entropy(c0, n, n_total_f),
            DpCriterion::Hellinger => binary_cost_hellinger(c0, n, n_total_f),
            DpCriterion::Wasserstein => binary_cost_wasserstein(c0, n, n_total_f, node_p0),
            DpCriterion::Learned => {
                if let Some(slb) = similar_lb {
                    learned_cost_binary(c0, n, n_total_f, slb)
                } else {
                    binary_cost(c0, n, n_total_f, false) // fallback Gini
                }
            }
            _ => binary_cost(c0, n, n_total_f, false), // Gini default
        };
        if !use_teacher || !matches!(criterion, DpCriterion::Gini) {
            return hard;
        }
        // Soft Gini on reference-ensemble-weighted class mass. `teacher_total = n`
        // because `teacher_prob_c0 + teacher_prob_c1 = 1` per row
        // (softmax output).
        let soft = soft_binary_gini_normalized(t_c0, n as f64, n_total_f);
        teacher_alpha * hard + (1.0 - teacher_alpha) * soft
    };

    // Flat dp table: dp[k * l + j] = top-B for k groups covering bins[0..=j]
    let mut dp = vec![TopB::new(); (eff_max_k + 1) * l];

    // Per-interval tax for group `bins[start..=end]`.
    // Returns 0.0 when interval_tax is None (sound default).
    let tax_at = |start: usize, end: usize| -> f64 {
        match interval_tax {
            Some(t) => t[start * l + end],
            None => 0.0,
        }
    };

    // Base case k=1: cost of grouping bins[0..=j].
    // Add leaf_reg for the single group (k=1 ⇒ 1 × leaf_reg).
    // Add interval_tax[0][j] when the boundary tax is in the DP.
    if let Some(ref table) = subsplit_table {
        for j in 0..l {
            if prefix_n[j + 1] < min_child {
                continue;
            }
            let cost = table[j]; // subsplit[0][j]
            if cost < f64::INFINITY {
                dp[l + j].try_insert(cost + leaf_reg + tax_at(0, j), 0, 0);
            }
        }
    } else {
        let (mut c0, mut n, mut t_c0) = (0i64, 0i64, 0.0_f64);
        for j in 0..l {
            c0 += bin_c0[j];
            n += bin_n[j];
            if use_teacher {
                t_c0 += teacher_slice[j];
            }
            if n >= min_child {
                let cost = edge_cost(c0, n, t_c0) + leaf_reg + tax_at(0, j);
                dp[l + j].try_insert(cost, 0, 0);
            }
        }
    }

    // k >= 2: i-outer loop swap
    for k in 2..=eff_max_k {
        for i in (k - 1)..l {
            // Copy prev entry's data to stack before taking any mutable borrow
            let (prev_count, prev_costs) = {
                let prev = &dp[(k - 1) * l + (i - 1)];
                (prev.count, prev.costs)
            };
            if prev_count == 0 {
                continue;
            }

            if let Some(ref table) = subsplit_table {
                // Depth2: use precomputed table for edge cost.
                // Add leaf_reg for the k-th group.
                // Add interval_tax[i][j] for the new group bins[i..=j].
                for j in i..l {
                    if prefix_n[j + 1] - prefix_n[i] < min_child {
                        continue;
                    }
                    let edge = table[i * l + j];
                    if edge == f64::INFINITY {
                        continue;
                    }
                    let edge_plus = edge + leaf_reg + tax_at(i, j);
                    let cur = &mut dp[k * l + j];
                    for b in 0..prev_count {
                        if !cur.try_insert(prev_costs[b] + edge_plus, i as u16, b as u8) {
                            break;
                        }
                    }
                }
            } else {
                // Incremental accumulation. Add leaf_reg per group.
                // Add interval_tax[i][j] for the new group bins[i..=j].
                let (mut c0, mut n, mut t_c0) = (0i64, 0i64, 0.0_f64);
                for j in i..l {
                    c0 += bin_c0[j];
                    n += bin_n[j];
                    if use_teacher {
                        t_c0 += teacher_slice[j];
                    }
                    if n < min_child {
                        continue;
                    }
                    let edge = edge_cost(c0, n, t_c0) + leaf_reg + tax_at(i, j);

                    let cur = &mut dp[k * l + j];
                    for b in 0..prev_count {
                        if !cur.try_insert(prev_costs[b] + edge, i as u16, b as u8) {
                            break;
                        }
                    }
                }
            }
        }
    }

    // --- Beam rejection stats extraction ---
    let mut beam_stats = DpBeamStats::new(eff_max_k);
    for k in 2..=eff_max_k {
        let idx = k - 2;
        let fc = &dp[k * l + (l - 1)];
        if fc.best_rejected < f64::INFINITY && fc.count > 0 {
            beam_stats.gap_at_final[idx] = fc.best_rejected - fc.costs[fc.count - 1];
        }
        for j in 0..l {
            if dp[k * l + j].best_rejected < f64::INFINITY {
                beam_stats.n_saturated_cells[idx] += 1;
            }
        }
    }

    // Prefix sums for group-stat extraction
    let mut cum_n = vec![0i64; l + 1];
    let mut cum_c0 = vec![0i64; l + 1];
    for i in 0..l {
        cum_n[i + 1] = cum_n[i] + bin_n[i];
        cum_c0[i + 1] = cum_c0[i] + bin_c0[i];
    }

    // Extract results for each k
    let mut ladder = Vec::new();
    for k in 2..=eff_max_k {
        let beam = beam_schedule
            .get(k - 2)
            .copied()
            .unwrap_or(1)
            .max(1)
            .min(BEAM_MAX);
        let final_entry = &dp[k * l + (l - 1)];

        for b in 0..final_entry.count.min(beam) {
            let boundaries = reconstruct_path(&dp, l, k, b);

            let split_thresh: Vec<f64> = boundaries.iter().map(|&bi| thresholds[bi - 1]).collect();

            // Group spans: [0, b0), [b0, b1), ..., [b_{k-2}, l)
            let starts = std::iter::once(0).chain(boundaries.iter().copied());
            let ends = boundaries.iter().copied().chain(std::iter::once(l));

            let mut group_n = Vec::with_capacity(k);
            let mut group_counts = Vec::with_capacity(k);
            for (s, e) in starts.zip(ends) {
                let gn = cum_n[e] - cum_n[s];
                let gc0 = cum_c0[e] - cum_c0[s];
                group_n.push(gn);
                group_counts.push(vec![gc0, gn - gc0]);
            }

            ladder.push((
                k,
                CollapseEntry {
                    cost: final_entry.costs[b],
                    thresholds: Arc::from(split_thresh.as_slice()),
                    bin_n: group_n,
                    bin_counts: group_counts,
                    group_map: None,
                    order: CandidateOrder::Original,
                },
            ));
        }
    }

    (ladder, beam_stats)
}

/// Top-B DP for multiclass classification (n_classes > 2).
///
/// Same algorithm as the binary variant but accumulates a small class-count
/// array per inner iteration instead of a single scalar.
pub fn collapse_bins_topb(
    thresholds: &[f64],
    bin_n: &[i64],
    bin_counts: &[Vec<i64>],
    max_k: usize,
    n_total: usize,
    beam_schedule: &[usize],
    criterion: DpCriterion,
    similar_lb: Option<&HashMap<SupportSig, f64>>,
    // Mirrors the binary path. Adds `leaf_reg` per
    // group to the DP edge cost so the final multiclass DP cost is on the
    // same regularized scale as the binary path. Pass 0.0 to leave
    // regularization out of the DP (caller adds reg post-DP).
    leaf_reg: f64,
) -> (Vec<(usize, CollapseEntry)>, DpBeamStats) {
    collapse_bins_topb_min_child(
        thresholds,
        bin_n,
        bin_counts,
        max_k,
        n_total,
        beam_schedule,
        criterion,
        similar_lb,
        leaf_reg,
        1,
    )
}

pub fn collapse_bins_topb_min_child(
    thresholds: &[f64],
    bin_n: &[i64],
    bin_counts: &[Vec<i64>],
    max_k: usize,
    n_total: usize,
    beam_schedule: &[usize],
    criterion: DpCriterion,
    similar_lb: Option<&HashMap<SupportSig, f64>>,
    leaf_reg: f64,
    min_child_size: usize,
) -> (Vec<(usize, CollapseEntry)>, DpBeamStats) {
    collapse_bins_topb_taxed_min_child(
        thresholds,
        bin_n,
        bin_counts,
        max_k,
        n_total,
        beam_schedule,
        criterion,
        similar_lb,
        leaf_reg,
        None,
        min_child_size,
    )
}

pub fn collapse_bins_topb_taxed_min_child(
    thresholds: &[f64],
    bin_n: &[i64],
    bin_counts: &[Vec<i64>],
    max_k: usize,
    n_total: usize,
    beam_schedule: &[usize],
    criterion: DpCriterion,
    similar_lb: Option<&HashMap<SupportSig, f64>>,
    leaf_reg: f64,
    interval_tax: Option<&[f64]>,
    min_child_size: usize,
) -> (Vec<(usize, CollapseEntry)>, DpBeamStats) {
    let l = bin_n.len();
    if l <= 1 {
        return (vec![], DpBeamStats::empty());
    }

    if interval_tax.is_none() && bin_n.iter().any(|&n| n == 0) {
        let mut proj_thresholds = Vec::new();
        let mut proj_n = Vec::with_capacity(l);
        let mut proj_counts = Vec::with_capacity(l);
        let mut seen = 0usize;
        for (orig_idx, &n) in bin_n.iter().enumerate() {
            if n <= 0 {
                continue;
            }
            if seen > 0 {
                let boundary = orig_idx.saturating_sub(1);
                if let Some(&thr) = thresholds.get(boundary) {
                    proj_thresholds.push(thr);
                } else {
                    return (vec![], DpBeamStats::empty());
                }
            }
            proj_n.push(n);
            proj_counts.push(bin_counts[orig_idx].clone());
            seen += 1;
        }
        if proj_n.len() <= 1 {
            return (vec![], DpBeamStats::empty());
        }
        return collapse_bins_topb_taxed_min_child(
            &proj_thresholds,
            &proj_n,
            &proj_counts,
            max_k,
            n_total,
            beam_schedule,
            criterion,
            similar_lb,
            leaf_reg,
            None,
            min_child_size,
        );
    }

    let n_classes = bin_counts[0].len();
    let n_total_f = n_total as f64;
    let min_child = min_child_size.max(1) as i64;
    let total_support: i64 = bin_n.iter().sum();
    let support_cap = (total_support / min_child).max(0) as usize;
    let eff_max_k = max_k.min(l).min(support_cap);
    if eff_max_k < 2 {
        return (vec![], DpBeamStats::empty());
    }
    let mut prefix_n = vec![0i64; l + 1];
    for i in 0..l {
        prefix_n[i + 1] = prefix_n[i] + bin_n[i];
    }

    // Twoing: precompute population class distribution p_bar
    let p_bar: Vec<f64> =
        if criterion == DpCriterion::Twoing || criterion == DpCriterion::Wasserstein {
            let total_n: f64 = bin_n.iter().sum::<i64>() as f64;
            let mut counts = vec![0i64; n_classes];
            for bc in bin_counts {
                for c in 0..n_classes {
                    counts[c] += bc[c];
                }
            }
            counts.iter().map(|&cnt| cnt as f64 / total_n).collect()
        } else {
            vec![]
        };

    // Depth2: precompute subsplit table
    let subsplit_table = match criterion {
        DpCriterion::Depth2 => Some(precompute_subsplit_multiclass(bin_n, bin_counts, n_total_f)),
        DpCriterion::Depth2Misclass => Some(precompute_subsplit_multiclass_misclass(
            bin_n, bin_counts, n_total_f,
        )),
        _ => None,
    };

    let mut dp = vec![TopB::new(); (eff_max_k + 1) * l];
    let tax_at = |start: usize, end: usize| -> f64 {
        match interval_tax {
            Some(t) => t[start * l + end],
            None => 0.0,
        }
    };

    // Base case k=1. Add leaf_reg for the single group.
    if let Some(ref table) = subsplit_table {
        for j in 0..l {
            if prefix_n[j + 1] < min_child {
                continue;
            }
            let cost = table[j];
            if cost < f64::INFINITY {
                dp[l + j].try_insert(cost + leaf_reg + tax_at(0, j), 0, 0);
            }
        }
    } else {
        let mut class_acc = vec![0i64; n_classes];
        let mut n = 0i64;
        for j in 0..l {
            for c in 0..n_classes {
                class_acc[c] += bin_counts[j][c];
            }
            n += bin_n[j];
            if n >= min_child {
                let cost = match criterion {
                    DpCriterion::Twoing => twoing_proxy_cost(&class_acc, n, n_total_f, &p_bar),
                    DpCriterion::Wasserstein => {
                        multiclass_cost_wasserstein(&class_acc, n, n_total_f, &p_bar)
                    }
                    DpCriterion::Misclass => multiclass_cost(&class_acc, n, n_total_f, true),
                    DpCriterion::Entropy => multiclass_cost_entropy(&class_acc, n, n_total_f),
                    DpCriterion::Hellinger => multiclass_cost_hellinger(&class_acc, n, n_total_f),
                    DpCriterion::Learned => {
                        if let Some(slb) = similar_lb {
                            learned_cost_multiclass(&class_acc, n, n_total_f, slb)
                        } else {
                            multiclass_cost(&class_acc, n, n_total_f, false)
                        }
                    }
                    _ => multiclass_cost(&class_acc, n, n_total_f, false),
                };
                dp[l + j].try_insert(cost + leaf_reg + tax_at(0, j), 0, 0);
            }
        }
    }

    // k >= 2: i-outer loop swap
    for k in 2..=eff_max_k {
        for i in (k - 1)..l {
            let (prev_count, prev_costs) = {
                let prev = &dp[(k - 1) * l + (i - 1)];
                (prev.count, prev.costs)
            };
            if prev_count == 0 {
                continue;
            }

            if let Some(ref table) = subsplit_table {
                for j in i..l {
                    if prefix_n[j + 1] - prefix_n[i] < min_child {
                        continue;
                    }
                    let edge = table[i * l + j];
                    if edge == f64::INFINITY {
                        continue;
                    }
                    // Add leaf_reg for the k-th group.
                    let edge_plus = edge + leaf_reg + tax_at(i, j);
                    let cur = &mut dp[k * l + j];
                    for b in 0..prev_count {
                        if !cur.try_insert(prev_costs[b] + edge_plus, i as u16, b as u8) {
                            break;
                        }
                    }
                }
            } else {
                let mut class_acc = vec![0i64; n_classes];
                let mut n = 0i64;
                for j in i..l {
                    for c in 0..n_classes {
                        class_acc[c] += bin_counts[j][c];
                    }
                    n += bin_n[j];
                    if n < min_child {
                        continue;
                    }
                    let edge = match criterion {
                        DpCriterion::Twoing => twoing_proxy_cost(&class_acc, n, n_total_f, &p_bar),
                        DpCriterion::Wasserstein => {
                            multiclass_cost_wasserstein(&class_acc, n, n_total_f, &p_bar)
                        }
                        DpCriterion::Misclass => multiclass_cost(&class_acc, n, n_total_f, true),
                        DpCriterion::Entropy => multiclass_cost_entropy(&class_acc, n, n_total_f),
                        DpCriterion::Hellinger => {
                            multiclass_cost_hellinger(&class_acc, n, n_total_f)
                        }
                        DpCriterion::Learned => {
                            if let Some(slb) = similar_lb {
                                learned_cost_multiclass(&class_acc, n, n_total_f, slb)
                            } else {
                                multiclass_cost(&class_acc, n, n_total_f, false)
                            }
                        }
                        _ => multiclass_cost(&class_acc, n, n_total_f, false),
                    };
                    // Add leaf_reg for the k-th group.
                    let edge_plus = edge + leaf_reg + tax_at(i, j);
                    let cur = &mut dp[k * l + j];
                    for b in 0..prev_count {
                        if !cur.try_insert(prev_costs[b] + edge_plus, i as u16, b as u8) {
                            break;
                        }
                    }
                }
            }
        }
    }

    // --- Beam rejection stats extraction ---
    let mut beam_stats = DpBeamStats::new(eff_max_k);
    for k in 2..=eff_max_k {
        let idx = k - 2;
        let fc = &dp[k * l + (l - 1)];
        if fc.best_rejected < f64::INFINITY && fc.count > 0 {
            beam_stats.gap_at_final[idx] = fc.best_rejected - fc.costs[fc.count - 1];
        }
        for j in 0..l {
            if dp[k * l + j].best_rejected < f64::INFINITY {
                beam_stats.n_saturated_cells[idx] += 1;
            }
        }
    }

    // Prefix sums for extraction
    let mut cum_n = vec![0i64; l + 1];
    let mut cum_counts = vec![vec![0i64; n_classes]; l + 1];
    for i in 0..l {
        cum_n[i + 1] = cum_n[i] + bin_n[i];
        for c in 0..n_classes {
            cum_counts[i + 1][c] = cum_counts[i][c] + bin_counts[i][c];
        }
    }

    let mut ladder = Vec::new();
    for k in 2..=eff_max_k {
        let beam = beam_schedule
            .get(k - 2)
            .copied()
            .unwrap_or(1)
            .max(1)
            .min(BEAM_MAX);
        let final_entry = &dp[k * l + (l - 1)];

        for b in 0..final_entry.count.min(beam) {
            let boundaries = reconstruct_path(&dp, l, k, b);

            let split_thresh: Vec<f64> = boundaries.iter().map(|&bi| thresholds[bi - 1]).collect();

            let starts = std::iter::once(0).chain(boundaries.iter().copied());
            let ends = boundaries.iter().copied().chain(std::iter::once(l));

            let mut group_n = Vec::with_capacity(k);
            let mut group_counts = Vec::with_capacity(k);
            for (s, e) in starts.zip(ends) {
                let gn = cum_n[e] - cum_n[s];
                let mut gc = vec![0i64; n_classes];
                for c in 0..n_classes {
                    gc[c] = cum_counts[e][c] - cum_counts[s][c];
                }
                group_n.push(gn);
                group_counts.push(gc);
            }

            ladder.push((
                k,
                CollapseEntry {
                    cost: final_entry.costs[b],
                    thresholds: Arc::from(split_thresh.as_slice()),
                    bin_n: group_n,
                    bin_counts: group_counts,
                    group_map: None,
                    order: CandidateOrder::Original,
                },
            ));
        }
    }

    (ladder, beam_stats)
}

// ---------------------------------------------------------------------------
// DP² local threshold scan
// ---------------------------------------------------------------------------

/// Run a cheap binary DP scan (max_k=2) on each feature of the given subset
/// to find locally-relevant threshold candidates for that child partition.
///
/// Returns a map from feature index → sorted Vec<f64> of selected thresholds.
/// Only features with at least one valid binary split are included.
/// Returns an empty map if the subset is too small or pure.
pub fn dp_threshold_scan(
    x: &Array2<f64>,
    y: &[u8],
    indices: &[u32],
    config: &SolverConfig,
    n_total: usize,
    scan_beam: usize,
) -> HashMap<usize, Vec<f64>> {
    let n_sub = indices.len();
    if n_sub < 4 {
        return HashMap::new();
    }

    // Check purity
    let labels: Vec<u8> = indices.iter().map(|&i| y[i as usize]).collect();
    let c0: usize = labels.iter().filter(|&&l| l == 0).count();
    if c0 == 0 || c0 == n_sub {
        return HashMap::new();
    }

    let beam_schedule = [scan_beam.max(1).min(16)];

    let results: Vec<Option<(usize, Vec<f64>)>> = (0..config.n_features)
        .into_par_iter()
        .map(|j| {
            let feature_vals: Vec<f64> = indices.iter().map(|&i| x[[i as usize, j]]).collect();

            let thresholds = match &config.thresholds_per_feature {
                Some(map) if map.contains_key(&j) => map[&j].clone(),
                _ => all_midpoints(&feature_vals),
            };
            if thresholds.is_empty() {
                return None;
            }

            // Binary-only scan: use binary path for speed
            let (bin_n, bin_c0) =
                bin_samples_binary(&feature_vals, &labels, &thresholds, None, 1.0);

            // Need at least 2 occupied bins
            let occupied = bin_n.iter().filter(|&&n| n > 0).count();
            if occupied < 2 {
                return None;
            }

            // Use Gini criterion for scanning (fast, criterion-agnostic)
            let (ladder, _) = collapse_bins_topb_binary(
                &thresholds,
                &bin_n,
                &bin_c0,
                2, // max_k=2: binary scan
                n_total,
                &beam_schedule,
                DpCriterion::Gini,
                None,
            );

            if ladder.is_empty() {
                return None;
            }

            // Union of all threshold sets across beam entries
            let mut selected: Vec<f64> = ladder
                .into_iter()
                .flat_map(|(_, entry)| entry.thresholds.to_vec())
                .collect();
            selected.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            selected.dedup_by(|a, b| (*a - *b).abs() < 1e-15);

            if selected.is_empty() {
                None
            } else {
                Some((j, selected))
            }
        })
        .collect();

    results.into_iter().flatten().collect()
}

// ---------------------------------------------------------------------------
// Parallel candidate generation
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Shape-function collapse: sort bins by reference-ensemble/label score, then standard DP
// ---------------------------------------------------------------------------

fn nonempty_bin_projection(
    thresholds: &[f64],
    bin_n: &[i64],
) -> Option<(Vec<usize>, Vec<usize>, Vec<f64>)> {
    let occupied: Vec<usize> = bin_n
        .iter()
        .enumerate()
        .filter_map(|(idx, &n)| if n > 0 { Some(idx) } else { None })
        .collect();
    if occupied.len() == bin_n.len() {
        return None;
    }
    if occupied.is_empty() {
        return Some((occupied, vec![0; bin_n.len()], Vec::new()));
    }

    let mut orig_to_projected = vec![0usize; bin_n.len()];
    let mut occ_pos = 0usize;
    for orig in 0..bin_n.len() {
        while occ_pos + 1 < occupied.len() && occupied[occ_pos + 1] <= orig {
            occ_pos += 1;
        }
        orig_to_projected[orig] = if orig < occupied[0] { 0 } else { occ_pos };
    }

    let mut projected_thresholds = Vec::with_capacity(occupied.len().saturating_sub(1));
    for &orig_idx in occupied.iter().skip(1) {
        if let Some(&thr) = thresholds.get(orig_idx.saturating_sub(1)) {
            projected_thresholds.push(thr);
        }
    }
    Some((occupied, orig_to_projected, projected_thresholds))
}

pub fn shape_collapse_topb_binary_min_child(
    thresholds: &[f64],
    bin_n: &[i64],
    bin_c0: &[i64],
    max_k: usize,
    n_total: usize,
    beam_schedule: &[usize],
    criterion: DpCriterion,
    similar_lb: Option<&HashMap<SupportSig, f64>>,
    leaf_reg: f64,
    min_child_size: usize,
) -> Vec<(usize, CollapseEntry)> {
    shape_collapse_topb_binary_with_teacher(
        thresholds,
        bin_n,
        bin_c0,
        max_k,
        n_total,
        beam_schedule,
        criterion,
        similar_lb,
        None,
        1.0,
        leaf_reg,
        min_child_size,
    )
}

/// Wrap `collapse_bins_topb_binary_with_teacher` with proportion-based sorting.
/// Returns CollapseEntry items whose `thresholds` are ALL L-1 bin boundaries
/// and whose `group_map` maps each original bin to a group.
pub fn shape_collapse_topb_binary_with_teacher(
    thresholds: &[f64],
    bin_n: &[i64],
    bin_c0: &[i64],
    max_k: usize,
    n_total: usize,
    beam_schedule: &[usize],
    criterion: DpCriterion,
    similar_lb: Option<&HashMap<SupportSig, f64>>,
    teacher_c0: Option<&[f64]>,
    teacher_alpha: f64,
    // Threaded through to the underlying
    // `collapse_bins_topb_binary_with_teacher` so shape-DP candidate
    // ordering matches main-DP ordering when reg > 0. Callers pass
    // `reg_alpha * reg`.
    leaf_reg: f64,
    min_child_size: usize,
) -> Vec<(usize, CollapseEntry)> {
    shape_collapse_topb_binary_with_teacher_smoothed(
        thresholds,
        bin_n,
        bin_c0,
        max_k,
        n_total,
        beam_schedule,
        criterion,
        similar_lb,
        teacher_c0,
        teacher_alpha,
        leaf_reg,
        0.0,
        min_child_size,
    )
}

/// Shape collapse with Bayesian shrinkage of the
/// per-bin sort key toward the parent (root-of-this-call) class-0 fraction.
///
/// Original sort key: `c0 / n`. With smoothing `α > 0`:
/// `(c0 + α·parent_p0) / (n + α)` — a pseudo-count smoothing where the
/// parent's class-0 fraction acts as a prior. Reduces ranking noise on
/// small-support bins where raw `c0/n` has high variance.
///
/// `smoothing_alpha = 0.0` recovers the unsmoothed sort. Recommended
/// values: 1.0 (light), 2.0 (moderate), 5.0 (heavy).
///
/// `shape_boundary_tax`: when > 0, adds a per-candidate boundary
/// cost = `μ_node × Σ_groups max(0, boundary_loss(group))` where
/// `boundary_loss` counts original-axis adjacency edges crossed minus
/// absorbed (× ½) by the proportion-permuted segment. 0.0 = no tax.
/// This wrapper always passes a zero tax; the `_taxed` variant adds the
/// tax inline into the DP edge cost.
pub fn shape_collapse_topb_binary_with_teacher_smoothed(
    thresholds: &[f64],
    bin_n: &[i64],
    bin_c0: &[i64],
    max_k: usize,
    n_total: usize,
    beam_schedule: &[usize],
    criterion: DpCriterion,
    similar_lb: Option<&HashMap<SupportSig, f64>>,
    teacher_c0: Option<&[f64]>,
    teacher_alpha: f64,
    leaf_reg: f64,
    smoothing_alpha: f64,
    min_child_size: usize,
) -> Vec<(usize, CollapseEntry)> {
    shape_collapse_topb_binary_with_teacher_smoothed_taxed(
        thresholds,
        bin_n,
        bin_c0,
        max_k,
        n_total,
        beam_schedule,
        criterion,
        similar_lb,
        teacher_c0,
        teacher_alpha,
        leaf_reg,
        smoothing_alpha,
        0.0,
        false,
        min_child_size,
    )
}

/// Boundary-tax entry point. ShapeSPLINTER callers pass a global-`N` scaled
/// boundary weight. When `mu_node == 0.0`, identical to
/// the non-taxed variant (bit-identical).
///
/// `per_atom_teacher_prior`: when true AND `teacher_c0` is provided, the
/// shape order is the per-atom mean reference-ensemble class-0 probability
/// (binary shape atoms sorted by `p(class 0) - p(class 1)`). When false, the legacy hard-label
/// proportion / parent smoothing order is used.
pub fn shape_collapse_topb_binary_with_teacher_smoothed_taxed(
    thresholds: &[f64],
    bin_n: &[i64],
    bin_c0: &[i64],
    max_k: usize,
    n_total: usize,
    beam_schedule: &[usize],
    criterion: DpCriterion,
    similar_lb: Option<&HashMap<SupportSig, f64>>,
    teacher_c0: Option<&[f64]>,
    teacher_alpha: f64,
    leaf_reg: f64,
    smoothing_alpha: f64,
    mu_node: f64,
    per_atom_teacher_prior: bool,
    min_child_size: usize,
) -> Vec<(usize, CollapseEntry)> {
    let l = bin_n.len();
    if l <= 1 {
        return vec![];
    }
    if let Some((occupied, orig_to_projected, projected_thresholds)) =
        nonempty_bin_projection(thresholds, bin_n)
    {
        if occupied.len() <= 1 {
            return vec![];
        }
        let projected_n: Vec<i64> = occupied.iter().map(|&idx| bin_n[idx]).collect();
        let projected_c0: Vec<i64> = occupied.iter().map(|&idx| bin_c0[idx]).collect();
        let projected_teacher: Option<Vec<f64>> = teacher_c0
            .filter(|t| t.len() == l)
            .map(|t| occupied.iter().map(|&idx| t[idx]).collect());
        let projected_entries = shape_collapse_topb_binary_with_teacher_smoothed_taxed(
            &projected_thresholds,
            &projected_n,
            &projected_c0,
            max_k,
            n_total,
            beam_schedule,
            criterion,
            similar_lb,
            projected_teacher.as_deref(),
            teacher_alpha,
            leaf_reg,
            smoothing_alpha,
            mu_node,
            per_atom_teacher_prior,
            min_child_size,
        );
        return projected_entries
            .into_iter()
            .map(|(k, mut entry)| {
                if let Some(projected_map) = entry.group_map.take() {
                    entry.group_map = Some(
                        orig_to_projected
                            .iter()
                            .map(|&p| projected_map.get(p).copied().unwrap_or(0))
                            .collect(),
                    );
                }
                entry.thresholds = Arc::from(thresholds);
                (k, entry)
            })
            .collect();
    }

    // 1. Sort bins by class-0 proportion (ascending) with optional smoothing.
    //    Three modes, in priority order:
    //    (a) reference-ensemble order: `mean_teacher_p0[a]`.
    //    (b) parent-p0 Bayesian shrinkage:
    //        `(c0[a] + α · parent_p0) / (n[a] + α)` when smoothing_alpha > 0.
    //    (c) raw `c0/n` otherwise.
    let mut order: Vec<usize> = (0..l).collect();
    let use_per_atom_prior =
        per_atom_teacher_prior && teacher_c0.is_some() && teacher_c0.unwrap().len() == l;
    let parent_p0 = if !use_per_atom_prior && smoothing_alpha > 0.0 {
        let total_c0: i64 = bin_c0.iter().sum();
        let total_n: i64 = bin_n.iter().sum();
        if total_n > 0 {
            total_c0 as f64 / total_n as f64
        } else {
            0.5
        }
    } else {
        0.5 // unused
    };
    let teacher_c0_slice = if use_per_atom_prior { teacher_c0 } else { None };
    let key_for = |i: usize| -> f64 {
        if let Some(t) = teacher_c0_slice {
            // Reference ShapeSPLINTER orders shape atoms by the mean reference-ensemble
            // probability vector, not by hard labels smoothed with that mean.
            if bin_n[i] > 0 {
                t[i] / bin_n[i] as f64
            } else {
                0.5
            }
        } else if smoothing_alpha > 0.0 {
            (bin_c0[i] as f64 + smoothing_alpha * parent_p0) / (bin_n[i] as f64 + smoothing_alpha)
        } else if bin_n[i] > 0 {
            bin_c0[i] as f64 / bin_n[i] as f64
        } else {
            0.5
        }
    };
    order.sort_by(|&a, &b| {
        key_for(a)
            .partial_cmp(&key_for(b))
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let sorted_n: Vec<i64> = order.iter().map(|&i| bin_n[i]).collect();
    let sorted_c0: Vec<i64> = order.iter().map(|&i| bin_c0[i]).collect();

    // Reorder teacher_c0 into the proportion-sorted bin order so the DP
    // sees reference-ensemble masses aligned with `sorted_n`/`sorted_c0`. Only done
    // when blend is actually active to avoid an O(L) copy on the common
    // α = 1.0 path.
    let teacher_active = teacher_c0.is_some() && teacher_alpha < 1.0 - 1e-12;
    let sorted_teacher: Option<Vec<f64>> = if teacher_active {
        let src = teacher_c0.unwrap();
        if src.len() == l {
            Some(order.iter().map(|&i| src[i]).collect())
        } else {
            None // mismatch → fall back to hard-only in the inner DP
        }
    } else {
        None
    };

    // 2. Dummy thresholds for the DP (it only uses them to label output)
    let dummy: Vec<f64> = (0..l.saturating_sub(1)).map(|i| i as f64).collect();

    // 3a. Precompute per-interval boundary tax in shape-position
    // coordinates. The helper uses incremental neighbor accounting, so this is
    // O(L^2) rather than scanning every original-axis edge for every interval.
    let interval_tax_vec = shape_interval_tax(&order, mu_node);

    // 3. Run DP on proportion-sorted bins. When interval_tax_vec is Some,
    // the DP transition adds the per-interval boundary tax inline.
    // When None, behavior is bit-identical to the legacy un-taxed call.
    let (raw, _) = collapse_bins_topb_binary_with_teacher_taxed_min_child(
        &dummy,
        &sorted_n,
        &sorted_c0,
        max_k,
        n_total,
        beam_schedule,
        criterion,
        similar_lb,
        sorted_teacher.as_deref(),
        teacher_alpha,
        leaf_reg,
        interval_tax_vec.as_deref(),
        min_child_size,
    );

    raw.into_iter()
        .map(|(k, entry)| {
            // Recover boundary positions from dummy thresholds
            let boundaries: Vec<usize> = entry.thresholds.iter().map(|&t| t as usize + 1).collect();

            let mut group_map = vec![0usize; l];
            let starts: Vec<usize> = std::iter::once(0)
                .chain(boundaries.iter().copied())
                .collect();
            let ends: Vec<usize> = boundaries
                .iter()
                .copied()
                .chain(std::iter::once(l))
                .collect();
            for (g, (&s, &e)) in starts.iter().zip(ends.iter()).enumerate() {
                for pos in s..e {
                    group_map[order[pos]] = g;
                }
            }

            // The DP proposal cost already includes the boundary tax.
            // Final ranking and objective costs are computed separately
            // from the child counts by `score_candidate`.
            let adjusted_cost = entry.cost;

            (
                k,
                CollapseEntry {
                    cost: adjusted_cost,
                    thresholds: Arc::from(thresholds), // ALL original thresholds
                    bin_n: entry.bin_n,
                    bin_counts: entry.bin_counts,
                    order: reference_order(&group_map),
                    group_map: Some(group_map),
                },
            )
        })
        .collect()
}

fn order_equivalence_key(order: &[usize]) -> Vec<usize> {
    let reversed_is_smaller = order.iter().rev().cmp(order.iter()) == std::cmp::Ordering::Less;
    let mut key = Vec::with_capacity(order.len() + 1);
    key.push(order.len());
    if reversed_is_smaller {
        key.extend(order.iter().rev().copied());
    } else {
        key.extend(order.iter().copied());
    }
    key
}

fn shape_interval_tax(order: &[usize], mu_node: f64) -> Option<Vec<f64>> {
    if mu_node <= 0.0 {
        return None;
    }
    let l = order.len();
    let mut shape_pos_of = vec![0usize; l];
    for (shape_pos, &orig) in order.iter().enumerate() {
        shape_pos_of[orig] = shape_pos;
    }
    let mut neighbor0 = vec![usize::MAX; l];
    let mut neighbor1 = vec![usize::MAX; l];
    let add_neighbor =
        |slot: usize, nb: usize, neighbor0: &mut [usize], neighbor1: &mut [usize]| {
            if neighbor0[slot] == usize::MAX {
                neighbor0[slot] = nb;
            } else {
                neighbor1[slot] = nb;
            }
        };
    for orig in 0..l.saturating_sub(1) {
        let a = shape_pos_of[orig];
        let b = shape_pos_of[orig + 1];
        add_neighbor(a, b, &mut neighbor0, &mut neighbor1);
        add_neighbor(b, a, &mut neighbor0, &mut neighbor1);
    }

    let mut tax = vec![0.0_f64; l * l];
    let mut active_stamp = vec![0u32; l];
    let mut stamp = 1u32;
    for s in 0..l {
        stamp = stamp.wrapping_add(1);
        if stamp == 0 {
            active_stamp.fill(0);
            stamp = 1;
        }
        let mut crossed: i64 = 0;
        for e in s..l {
            active_stamp[e] = stamp;
            let first = neighbor0[e];
            if first != usize::MAX {
                if active_stamp[first] == stamp {
                    crossed -= 1;
                } else {
                    crossed += 1;
                }
            }
            let second = neighbor1[e];
            if second != usize::MAX {
                if active_stamp[second] == stamp {
                    crossed -= 1;
                } else {
                    crossed += 1;
                }
            }
            tax[s * l + e] = 0.5 * mu_node * crossed as f64;
        }
    }
    Some(tax)
}

fn shape_collapse_topb_multiclass_order_min_child(
    thresholds: &[f64],
    bin_n: &[i64],
    bin_counts: &[Vec<i64>],
    order: &[usize],
    max_k: usize,
    n_total: usize,
    beam_schedule: &[usize],
    criterion: DpCriterion,
    similar_lb: Option<&HashMap<SupportSig, f64>>,
    leaf_reg: f64,
    mu_node: f64,
    min_child_size: usize,
) -> Vec<(usize, CollapseEntry)> {
    let l = bin_n.len();
    let sorted_n: Vec<i64> = order.iter().map(|&i| bin_n[i]).collect();
    let sorted_counts: Vec<Vec<i64>> = order.iter().map(|&i| bin_counts[i].clone()).collect();
    let dummy: Vec<f64> = (0..l.saturating_sub(1)).map(|i| i as f64).collect();
    let interval_tax = shape_interval_tax(order, mu_node);

    let (raw, _) = collapse_bins_topb_taxed_min_child(
        &dummy,
        &sorted_n,
        &sorted_counts,
        max_k,
        n_total,
        beam_schedule,
        criterion,
        similar_lb,
        leaf_reg,
        interval_tax.as_deref(),
        min_child_size,
    );

    raw.into_iter()
        .map(|(k, entry)| {
            let boundaries: Vec<usize> = entry.thresholds.iter().map(|&t| t as usize + 1).collect();
            let mut group_map = vec![0usize; l];
            let starts = std::iter::once(0).chain(boundaries.iter().copied());
            let ends = boundaries.iter().copied().chain(std::iter::once(l));
            for (g, (s, e)) in starts.zip(ends).enumerate() {
                for pos in s..e {
                    group_map[order[pos]] = g;
                }
            }
            (
                k,
                CollapseEntry {
                    cost: entry.cost,
                    thresholds: Arc::from(thresholds),
                    bin_n: entry.bin_n,
                    bin_counts: entry.bin_counts,
                    order: reference_order(&group_map),
                    group_map: Some(group_map),
                },
            )
        })
        .collect()
}

/// Multiclass ShapeSPLINTER candidate generator using the ground-truth
/// LightGBM pairwise-margin orders. For every pair of classes present at the
/// node, bins are sorted by mean reference-ensemble probability difference
/// `p_teacher(left) - p_teacher(right)`. Reverse-equivalent orders are
/// deduplicated because they induce the same interval partitions.
pub fn shape_collapse_topb_pairwise_multiclass_min_child(
    thresholds: &[f64],
    bin_n: &[i64],
    bin_counts: &[Vec<i64>],
    teacher_prob_mass: Option<&[f64]>,
    max_k: usize,
    n_total: usize,
    beam_schedule: &[usize],
    criterion: DpCriterion,
    similar_lb: Option<&HashMap<SupportSig, f64>>,
    leaf_reg: f64,
    mu_node: f64,
    min_child_size: usize,
) -> Vec<(usize, CollapseEntry)> {
    let l = bin_n.len();
    if l <= 1 {
        return Vec::new();
    }
    if let Some((occupied, orig_to_projected, projected_thresholds)) =
        nonempty_bin_projection(thresholds, bin_n)
    {
        if occupied.len() <= 1 {
            return Vec::new();
        }
        let n_classes = bin_counts.first().map(|v| v.len()).unwrap_or(0);
        if n_classes < 2 {
            return Vec::new();
        }
        let projected_n: Vec<i64> = occupied.iter().map(|&idx| bin_n[idx]).collect();
        let projected_counts: Vec<Vec<i64>> = occupied
            .iter()
            .map(|&idx| bin_counts[idx].clone())
            .collect();
        let projected_teacher: Option<Vec<f64>> = teacher_prob_mass
            .filter(|t| t.len() == l * n_classes)
            .map(|t| {
                let mut out = Vec::with_capacity(occupied.len() * n_classes);
                for &idx in &occupied {
                    let base = idx * n_classes;
                    out.extend_from_slice(&t[base..base + n_classes]);
                }
                out
            });
        let projected_entries = shape_collapse_topb_pairwise_multiclass_min_child(
            &projected_thresholds,
            &projected_n,
            &projected_counts,
            projected_teacher.as_deref(),
            max_k,
            n_total,
            beam_schedule,
            criterion,
            similar_lb,
            leaf_reg,
            mu_node,
            min_child_size,
        );
        return projected_entries
            .into_iter()
            .map(|(k, mut entry)| {
                if let Some(projected_map) = entry.group_map.take() {
                    entry.group_map = Some(
                        orig_to_projected
                            .iter()
                            .map(|&p| projected_map.get(p).copied().unwrap_or(0))
                            .collect(),
                    );
                }
                entry.thresholds = Arc::from(thresholds);
                (k, entry)
            })
            .collect();
    }
    let n_classes = bin_counts.first().map(|v| v.len()).unwrap_or(0);
    let Some(teacher) = teacher_prob_mass else {
        return Vec::new();
    };
    if n_classes < 2 || teacher.len() != l * n_classes {
        return Vec::new();
    }

    let mut present = Vec::with_capacity(n_classes);
    for cls in 0..n_classes {
        let weight: i64 = bin_counts.iter().map(|row| row[cls]).sum();
        if weight > 0 {
            present.push(cls);
        }
    }
    if present.len() < 2 {
        return Vec::new();
    }

    let mut seen_orders = HashSet::<Vec<usize>>::new();
    let mut out = Vec::new();
    for li in 0..present.len() {
        for ri in li + 1..present.len() {
            let left = present[li];
            let right = present[ri];
            let mut keyed_order: Vec<(i64, i32, usize)> = Vec::with_capacity(l);
            for axis in 0..l {
                let denom = (bin_n[axis] as f64).max(1.0);
                let base = axis * n_classes;
                let score = teacher[base + left] / denom - teacher[base + right] / denom;
                keyed_order.push(((score / 1.0e-12).round() as i64, axis as i32, axis));
            }
            keyed_order.sort_unstable_by(|a, b| {
                a.0.cmp(&b.0)
                    .then_with(|| a.1.cmp(&b.1))
                    .then_with(|| a.2.cmp(&b.2))
            });
            let order: Vec<usize> = keyed_order.into_iter().map(|(_, _, axis)| axis).collect();
            if !seen_orders.insert(order_equivalence_key(&order)) {
                continue;
            }
            out.extend(shape_collapse_topb_multiclass_order_min_child(
                thresholds,
                bin_n,
                bin_counts,
                &order,
                max_k,
                n_total,
                beam_schedule,
                criterion,
                similar_lb,
                leaf_reg,
                mu_node,
                min_child_size,
            ));
        }
    }

    out
}

// ---------------------------------------------------------------------------
// Support-mask helpers
// ---------------------------------------------------------------------------

#[inline]
pub fn support_contains_pub(words: &[u64], idx: u32) -> bool {
    support_contains(words, idx)
}

fn support_contains(words: &[u64], idx: u32) -> bool {
    let idx = idx as usize;
    let word = idx / 64;
    let bit = idx % 64;
    words
        .get(word)
        .map(|w| (w & (1u64 << bit)) != 0)
        .unwrap_or(false)
}

pub fn compute_feature_irreducible_error_binary_mask_pub(
    sorted_order: &[u32],
    feature_col: &[f64],
    labels: &[u8],
    support: &SupportMask,
    n_total: usize,
) -> f64 {
    compute_feature_irreducible_error_binary_mask(
        sorted_order,
        feature_col,
        labels,
        support,
        n_total,
    )
}

fn compute_feature_irreducible_error_binary_mask(
    sorted_order: &[u32],
    feature_col: &[f64],
    labels: &[u8],
    support: &SupportMask,
    n_total: usize,
) -> f64 {
    if support.n_samples == 0 {
        return 0.0;
    }

    let mut total_irreducible: i64 = 0;
    let mut run_c0: i64 = 0;
    let mut run_c1: i64 = 0;
    let mut last_val = 0.0;
    let mut seen_any = false;

    for &idx in sorted_order {
        if !support_contains(support.words.as_ref(), idx) {
            continue;
        }
        let val = feature_col[idx as usize];
        if seen_any && (val - last_val).abs() > 1e-15 {
            total_irreducible += run_c0.min(run_c1);
            run_c0 = 0;
            run_c1 = 0;
        }
        seen_any = true;
        last_val = val;
        if labels[idx as usize] == 0 {
            run_c0 += 1;
        } else {
            run_c1 += 1;
        }
    }

    if seen_any {
        total_irreducible += run_c0.min(run_c1);
    }

    total_irreducible as f64 / n_total as f64
}

/// Public wrapper for the multiclass per-feature irreducible LB. Same
/// signature as the binary `_pub` wrapper plus an `n_classes` parameter.
pub fn compute_feature_irreducible_error_multiclass_mask_pub(
    sorted_order: &[u32],
    feature_col: &[f64],
    labels: &[u8],
    support: &SupportMask,
    n_classes: usize,
    n_total: usize,
) -> f64 {
    compute_feature_irreducible_error_multiclass_mask(
        sorted_order,
        feature_col,
        labels,
        support,
        n_classes,
        n_total,
    )
}

/// Multiclass twin. Per-unique-feature-value run, the unavoidable
/// misclassification is `n_run - max_class_count_in_run`. Summed across
/// runs and normalized to `n_total`. Admissible LB on the per-feature
/// leaf-equivalent loss; same role as the binary version.
fn compute_feature_irreducible_error_multiclass_mask(
    sorted_order: &[u32],
    feature_col: &[f64],
    labels: &[u8],
    support: &SupportMask,
    n_classes: usize,
    n_total: usize,
) -> f64 {
    if support.n_samples == 0 {
        return 0.0;
    }

    let mut total_irreducible: i64 = 0;
    let mut run_counts = [0i64; MAX_CLASSES];
    let mut last_val = 0.0;
    let mut seen_any = false;

    for &idx in sorted_order {
        if !support_contains(support.words.as_ref(), idx) {
            continue;
        }
        let val = feature_col[idx as usize];
        if seen_any && (val - last_val).abs() > 1e-15 {
            let n: i64 = run_counts[..n_classes].iter().sum();
            let max: i64 = *run_counts[..n_classes].iter().max().unwrap_or(&0);
            total_irreducible += n - max;
            for c in 0..n_classes {
                run_counts[c] = 0;
            }
        }
        seen_any = true;
        last_val = val;
        run_counts[labels[idx as usize] as usize] += 1;
    }

    if seen_any {
        let n: i64 = run_counts[..n_classes].iter().sum();
        let max: i64 = *run_counts[..n_classes].iter().max().unwrap_or(&0);
        total_irreducible += n - max;
    }

    total_irreducible as f64 / n_total as f64
}

pub fn fill_bins_from_sorted_binary_mask_pub(
    sorted_order: &[u32],
    feature_col: &[f64],
    labels: &[u8],
    support: &SupportMask,
    max_bins: usize,
    weights: Option<&[f64]>,
    weight_scale: f64,
    scratch: &mut WorkerScratch,
) -> Option<()> {
    fill_bins_from_sorted_binary_mask(
        sorted_order,
        feature_col,
        labels,
        support,
        max_bins,
        weights,
        weight_scale,
        scratch,
    )
}

fn fill_bins_from_sorted_binary_mask(
    sorted_order: &[u32],
    feature_col: &[f64],
    labels: &[u8],
    support: &SupportMask,
    max_bins: usize,
    weights: Option<&[f64]>,
    weight_scale: f64,
    scratch: &mut WorkerScratch,
) -> Option<()> {
    scratch.thresholds.clear();
    scratch.bin_n.clear();
    scratch.bin_c0.clear();

    let mut cur_val = f64::NAN;
    let mut cur_n = 0i64;
    let mut cur_c0 = 0i64;
    let mut n_unique = 0usize;

    for &idx in sorted_order {
        if !support_contains(support.words.as_ref(), idx) {
            continue;
        }

        let val = feature_col[idx as usize];
        if n_unique == 0 || (val - cur_val).abs() > 1e-15 {
            if n_unique > 0 {
                scratch.bin_n.push(cur_n);
                scratch.bin_c0.push(cur_c0);
                scratch.thresholds.push((cur_val + val) / 2.0);
                cur_n = 0;
                cur_c0 = 0;
            }
            n_unique += 1;
            cur_val = val;
            if n_unique > max_bins + 1 {
                scratch.thresholds.clear();
                scratch.bin_n.clear();
                scratch.bin_c0.clear();
                return None;
            }
        }

        let wi = if let Some(w) = weights {
            (w[idx as usize] * weight_scale).round() as i64
        } else {
            1
        };
        cur_n += wi;
        if labels[idx as usize] == 0 {
            cur_c0 += wi;
        }
    }

    if n_unique > 0 {
        scratch.bin_n.push(cur_n);
        scratch.bin_c0.push(cur_c0);
    }

    if scratch.thresholds.is_empty() {
        scratch.bin_n.clear();
        scratch.bin_c0.clear();
    }

    Some(())
}

/// Public wrapper. Multiclass twin of `fill_bins_from_sorted_binary_mask_pub`.
/// On Some(bin_counts) the scratch's `thresholds` and `bin_n` are populated
/// in lock-step; the per-class counts are returned as a fresh
/// `Vec<Vec<i64>>` so the closure DP can pass it straight to
/// `collapse_bins_topb`. Returns None when `max_bins` would be exceeded.
pub fn fill_bins_from_sorted_multiclass_mask_pub(
    sorted_order: &[u32],
    feature_col: &[f64],
    labels: &[u8],
    support: &SupportMask,
    n_classes: usize,
    max_bins: usize,
    weights: Option<&[f64]>,
    weight_scale: f64,
    scratch: &mut WorkerScratch,
) -> Option<Vec<Vec<i64>>> {
    fill_bins_from_sorted_multiclass_mask(
        sorted_order,
        feature_col,
        labels,
        support,
        n_classes,
        max_bins,
        weights,
        weight_scale,
        scratch,
    )
}

fn fill_bins_from_sorted_multiclass_mask(
    sorted_order: &[u32],
    feature_col: &[f64],
    labels: &[u8],
    support: &SupportMask,
    n_classes: usize,
    max_bins: usize,
    weights: Option<&[f64]>,
    weight_scale: f64,
    scratch: &mut WorkerScratch,
) -> Option<Vec<Vec<i64>>> {
    scratch.thresholds.clear();
    scratch.bin_n.clear();

    let mut bin_counts: Vec<Vec<i64>> = Vec::new();
    let mut cur_val = f64::NAN;
    let mut cur_n = 0i64;
    let mut cur_counts = vec![0i64; n_classes];
    let mut n_unique = 0usize;

    for &idx in sorted_order {
        if !support_contains(support.words.as_ref(), idx) {
            continue;
        }

        let val = feature_col[idx as usize];
        if n_unique == 0 || (val - cur_val).abs() > 1e-15 {
            if n_unique > 0 {
                scratch.bin_n.push(cur_n);
                bin_counts.push(cur_counts.clone());
                scratch.thresholds.push((cur_val + val) / 2.0);
                cur_n = 0;
                cur_counts.fill(0);
            }
            n_unique += 1;
            cur_val = val;
            if n_unique > max_bins + 1 {
                scratch.thresholds.clear();
                scratch.bin_n.clear();
                return None;
            }
        }

        let wi = if let Some(w) = weights {
            (w[idx as usize] * weight_scale).round() as i64
        } else {
            1
        };
        cur_n += wi;
        cur_counts[labels[idx as usize] as usize] += wi;
    }

    if n_unique > 0 {
        scratch.bin_n.push(cur_n);
        bin_counts.push(cur_counts);
    }

    if scratch.thresholds.is_empty() {
        scratch.bin_n.clear();
        return Some(Vec::new());
    }

    Some(bin_counts)
}

pub fn fill_bin_samples_sorted_binary_mask_pub(
    sorted_order: &[u32],
    feature_col: &[f64],
    labels: &[u8],
    thresholds: &[f64],
    support: &SupportMask,
    weights: Option<&[f64]>,
    weight_scale: f64,
    scratch: &mut WorkerScratch,
) {
    fill_bin_samples_sorted_binary_mask(
        sorted_order,
        feature_col,
        labels,
        thresholds,
        support,
        weights,
        weight_scale,
        None,
        scratch,
    )
}

/// Compute per-atom reference-ensemble class-0 probability mass over a node's
/// support, given the final (post-atomization) thresholds. One linear
/// pass over the feature's sorted sample order; O(n_support).
///
/// Used by the binary reference-ensemble-order path when `SolverConfig::teacher_prob_c0`
/// is available.
fn compute_atom_teacher_c0_binary(
    sorted_order: &[u32],
    feature_col: &[f64],
    atom_thresholds: &[f64],
    support: &SupportMask,
    teacher_prob_c0: &[f64],
    n_atoms: usize,
) -> Vec<f64> {
    let mut out = vec![0.0f64; n_atoms.max(1)];
    if n_atoms == 0 {
        return out;
    }
    let mut bin = 0usize;
    for &idx in sorted_order {
        if !support_contains(support.words.as_ref(), idx) {
            continue;
        }
        let val = feature_col[idx as usize];
        while bin < atom_thresholds.len() && atom_thresholds[bin] <= val {
            bin += 1;
        }
        out[bin] += teacher_prob_c0[idx as usize];
    }
    out
}

fn compute_bin_teacher_prob_multiclass(
    sorted_order: &[u32],
    feature_col: &[f64],
    thresholds: &[f64],
    support: &SupportMask,
    weights: Option<&[f64]>,
    weight_scale: f64,
    teacher_prob: &[f64],
    n_classes: usize,
) -> Option<Vec<f64>> {
    if n_classes < 2 || teacher_prob.len() % n_classes != 0 {
        return None;
    }
    let n_bins = thresholds.len() + 1;
    let mut out = vec![0.0_f64; n_bins * n_classes];
    let mut bin = 0usize;
    for &idx in sorted_order {
        if !support_contains(support.words.as_ref(), idx) {
            continue;
        }
        let idx_usize = idx as usize;
        let src = idx_usize.checked_mul(n_classes)?;
        if src + n_classes > teacher_prob.len() {
            return None;
        }
        let val = feature_col[idx_usize];
        while bin < thresholds.len() && thresholds[bin] <= val {
            bin += 1;
        }
        let wi = if let Some(w) = weights {
            (w[idx_usize] * weight_scale).round()
        } else {
            1.0
        };
        let dst = bin * n_classes;
        for cls in 0..n_classes {
            out[dst + cls] += teacher_prob[src + cls] * wi;
        }
    }
    Some(out)
}

fn fill_bin_samples_sorted_binary_mask(
    sorted_order: &[u32],
    feature_col: &[f64],
    labels: &[u8],
    thresholds: &[f64],
    support: &SupportMask,
    weights: Option<&[f64]>,
    weight_scale: f64,
    teacher_prob_c0: Option<&[f64]>,
    scratch: &mut WorkerScratch,
) {
    scratch.bin_n.clear();
    scratch.bin_c0.clear();
    scratch.bin_n.resize(thresholds.len() + 1, 0);
    scratch.bin_c0.resize(thresholds.len() + 1, 0);
    scratch.bin_teacher_c0.clear();
    if teacher_prob_c0.is_some() {
        scratch.bin_teacher_c0.resize(thresholds.len() + 1, 0.0);
    }

    let mut bin = 0usize;
    for &idx in sorted_order {
        if !support_contains(support.words.as_ref(), idx) {
            continue;
        }
        let val = feature_col[idx as usize];
        while bin < thresholds.len() && thresholds[bin] <= val {
            bin += 1;
        }
        let wi = if let Some(w) = weights {
            (w[idx as usize] * weight_scale).round() as i64
        } else {
            1
        };
        scratch.bin_n[bin] += wi;
        if labels[idx as usize] == 0 {
            scratch.bin_c0[bin] += wi;
        }
        if let Some(tp) = teacher_prob_c0 {
            // Reference-ensemble contribution is prob × weight (float). When
            // weights are unit, wi == 1 and this reduces to raw reference-ensemble
            // prob mass. Weighted runs (boosting) scale reference-ensemble
            // contributions by the same factor as the hard counts.
            scratch.bin_teacher_c0[bin] += tp[idx as usize] * wi as f64;
        }
    }
}

/// Public wrapper. Multiclass twin of `fill_bin_samples_sorted_binary_mask_pub`.
/// Used when the caller already has a fixed `thresholds` slice (e.g.,
/// `closure_thresholds_per_feature`). Populates `bin_n` and returns
/// per-class bin counts as a `Vec<Vec<i64>>`. Counts are also left in
/// scratch.bin_counts_flat for callers that prefer the flat layout.
pub fn fill_bin_samples_sorted_multiclass_mask_pub(
    sorted_order: &[u32],
    feature_col: &[f64],
    labels: &[u8],
    thresholds: &[f64],
    n_classes: usize,
    support: &SupportMask,
    weights: Option<&[f64]>,
    weight_scale: f64,
    scratch: &mut WorkerScratch,
) -> Vec<Vec<i64>> {
    fill_bin_samples_sorted_multiclass_mask(
        sorted_order,
        feature_col,
        labels,
        thresholds,
        n_classes,
        support,
        weights,
        weight_scale,
        scratch,
    )
}

fn fill_bin_samples_sorted_multiclass_mask(
    sorted_order: &[u32],
    feature_col: &[f64],
    labels: &[u8],
    thresholds: &[f64],
    n_classes: usize,
    support: &SupportMask,
    weights: Option<&[f64]>,
    weight_scale: f64,
    scratch: &mut WorkerScratch,
) -> Vec<Vec<i64>> {
    scratch.bin_n.clear();
    scratch.bin_n.resize(thresholds.len() + 1, 0);
    scratch.bin_counts_flat.clear();
    scratch
        .bin_counts_flat
        .resize((thresholds.len() + 1) * n_classes, 0);

    let mut bin = 0usize;
    for &idx in sorted_order {
        if !support_contains(support.words.as_ref(), idx) {
            continue;
        }
        let val = feature_col[idx as usize];
        while bin < thresholds.len() && thresholds[bin] <= val {
            bin += 1;
        }
        let wi = if let Some(w) = weights {
            (w[idx as usize] * weight_scale).round() as i64
        } else {
            1
        };
        scratch.bin_n[bin] += wi;
        let offset = bin * n_classes + labels[idx as usize] as usize;
        scratch.bin_counts_flat[offset] += wi;
    }

    (0..thresholds.len() + 1)
        .map(|bin_idx| {
            let start = bin_idx * n_classes;
            scratch.bin_counts_flat[start..start + n_classes].to_vec()
        })
        .collect()
}

/// Support-sorted Select scan crossover: Select scans the support's own rows
/// (sorted by rank, O(n log n)) instead of the feature's full sorted order
/// (O(N)) when `SUPPORT_SORT_RATIO · n · ⌈log2 n⌉ < N`.
const SUPPORT_SORT_RATIO: usize = 2;

/// Row order of feature `j` that Select's sorted scans walk: the support's
/// rows in value order (support-sorted Select scan) or the feature's full
/// sorted order (each scan skips rows outside the support). Both give the
/// same subsequence of in-support rows, so every scan result is identical.
///
/// A feature whose support order was not built (empty list: Select builds
/// only the ones it reads; a non-empty support has a non-empty list)
/// falls back to the full order, which gives the same scan.
#[inline]
fn order_for<'a>(store: &'a DatasetStore, sorted: Option<&'a [Vec<u32>]>, j: usize) -> &'a [u32] {
    match sorted {
        Some(v) if !v[j].is_empty() => v[j].as_slice(),
        _ => store.sorted_feature_order(j),
    }
}

pub fn generate_candidates_for_support(
    store: &DatasetStore,
    support: &SupportMask,
    config: &SolverConfig,
    available_features: &FeatureMask,
    n_total: usize,
    ub_hint: Option<f64>,
    similar_lb: Option<&HashMap<SupportSig, f64>>,
    scratch_bank: &ScratchBank,
    beam_width_override: Option<usize>,
    transition_delta_override: Option<f64>,
    depth_remaining: Option<i32>,
) -> (Vec<SplitCandidate>, CandGenStats) {
    let n = support.n_samples;
    let log2n = (usize::BITS - n.leading_zeros()) as usize;
    let support_sort =
        n.saturating_mul(log2n).saturating_mul(SUPPORT_SORT_RATIO) < store.n_samples();
    generate_candidates_for_support_scan(
        store,
        support,
        config,
        available_features,
        n_total,
        ub_hint,
        similar_lb,
        scratch_bank,
        beam_width_override,
        transition_delta_override,
        depth_remaining,
        support_sort,
    )
}

/// `generate_candidates_for_support` for a caller that discards the beam
/// statistics of `CandGenStats` (the Step-A pass and the
/// completion-lb DFS): the same pool, with the stats-free DP paths
/// (`topcell`, `cap`) allowed in every DP (the
/// contiguous Gini / Misclass ladder takes them for all callers).
#[allow(clippy::too_many_arguments)]
pub fn generate_candidates_for_support_rash(
    store: &DatasetStore,
    support: &SupportMask,
    config: &SolverConfig,
    available_features: &FeatureMask,
    n_total: usize,
    ub_hint: Option<f64>,
    similar_lb: Option<&HashMap<SupportSig, f64>>,
    scratch_bank: &ScratchBank,
    depth_remaining: Option<i32>,
) -> (Vec<SplitCandidate>, CandGenStats) {
    let n = support.n_samples;
    let log2n = (usize::BITS - n.leading_zeros()) as usize;
    let support_sort =
        n.saturating_mul(log2n).saturating_mul(SUPPORT_SORT_RATIO) < store.n_samples();
    generate_candidates_for_support_opts(
        store,
        support,
        config,
        available_features,
        n_total,
        ub_hint,
        similar_lb,
        scratch_bank,
        None,
        None,
        depth_remaining,
        SelectOpts {
            support_sort,
            stats_free: true,
        },
    )
}

/// Select knobs fixed by the caller (`generate_candidates_for_support_opts`).
/// None of them changes the returned pool (bitwise, in order).
#[derive(Clone, Copy, Debug)]
pub struct SelectOpts {
    /// Walk each feature's support rows sorted by rank
    /// (support-sorted Select scan) instead of the full sorted order.
    pub support_sort: bool,
    /// The caller discards the beam statistics of `CandGenStats`: the DPs
    /// may skip the cells only they read (`topcell` /
    /// `cap`). The contiguous Gini ladder and Misclass
    /// companion skip them regardless (`ladder_opts`).
    pub stats_free: bool,
}

/// `generate_candidates_for_support` with the scan order fixed:
/// `support_sort` = walk each feature's support rows sorted by rank
/// (`DatasetStore::sorted_support_indices`) instead of the full sorted
/// order. The output is identical either way (tests call both).
#[allow(clippy::too_many_arguments)]
pub fn generate_candidates_for_support_scan(
    store: &DatasetStore,
    support: &SupportMask,
    config: &SolverConfig,
    available_features: &FeatureMask,
    n_total: usize,
    ub_hint: Option<f64>,
    similar_lb: Option<&HashMap<SupportSig, f64>>,
    scratch_bank: &ScratchBank,
    beam_width_override: Option<usize>,
    transition_delta_override: Option<f64>,
    depth_remaining: Option<i32>,
    support_sort: bool,
) -> (Vec<SplitCandidate>, CandGenStats) {
    generate_candidates_for_support_opts(
        store,
        support,
        config,
        available_features,
        n_total,
        ub_hint,
        similar_lb,
        scratch_bank,
        beam_width_override,
        transition_delta_override,
        depth_remaining,
        SelectOpts {
            support_sort,
            stats_free: false,
        },
    )
}

/// Select (`generate_candidates_for_support`) with explicit `SelectOpts`.
#[allow(clippy::too_many_arguments)]
pub fn generate_candidates_for_support_opts(
    store: &DatasetStore,
    support: &SupportMask,
    config: &SolverConfig,
    available_features: &FeatureMask,
    n_total: usize,
    ub_hint: Option<f64>,
    similar_lb: Option<&HashMap<SupportSig, f64>>,
    scratch_bank: &ScratchBank,
    beam_width_override: Option<usize>,
    transition_delta_override: Option<f64>,
    depth_remaining: Option<i32>,
    opts: SelectOpts,
) -> (Vec<SplitCandidate>, CandGenStats) {
    let support_sort = opts.support_sort;
    // DP speed paths; the stats-free ones only for callers that discard the
    // beam statistics.
    let dp_opts = DpOpts {
        topcell: opts.stats_free,
        cap: opts.stats_free,
    };
    // The contiguous Gini ladder and Misclass companion: topcell / cap for
    // every caller (only `DpBeamStats` differ, see `collapse_binary_fast`).
    let ladder_opts = DpOpts {
        topcell: true,
        cap: true,
    };
    // `config.bypass_beam` is the user-facing Rashomon-completeness lever:
    // when on, force the per-feature DP beam to a value large enough that
    // no candidate is ever truncated. Takes precedence over an explicit
    // `beam_width_override` from the caller (this makes the effect
    // available to Phase-2 BFS and any other caller via a single config
    // flag).
    let beam_width = if config.bypass_beam {
        1usize << 30
    } else {
        beam_width_override.unwrap_or(config.beam_width)
    };
    let transition_delta = transition_delta_override.unwrap_or(config.transition_delta);
    let n_node = support.n_samples;
    let available: Vec<usize> = (0..config.n_features)
        .filter(|j| !config.feature_exclusion || available_features.contains(*j))
        .collect();
    let n_available = available.len();
    let has_custom_of = |j: usize| {
        config
            .thresholds_per_feature
            .as_ref()
            .map_or(false, |m| m.contains_key(&j))
    };
    // The irreducible-error pass below only orders the
    // features (the pool is order-free: dedup keys include the feature and
    // the final sort is total on distinct keys), so it runs only when the
    // depth-gated unsafe filter can drop features.
    let unsafe_filter_active = match depth_remaining {
        Some(d) => {
            config.unsafe_filter_depth_threshold > 0 && d <= config.unsafe_filter_depth_threshold
        }
        None => false,
    };
    let irr_pass = unsafe_filter_active;
    // Support sort: one value-ordered row list per available feature,
    // shared by the irreducible-error pass and the bins pass (only for the
    // features whose scans read one; the binary custom-threshold bins pass
    // walks the support's row list).
    let reads_sorted =
        |j: usize| irr_pass || config.n_classes != 2 || !has_custom_of(j) || config.shape_mode;
    let support_sorted_vecs: Option<Vec<Vec<u32>>> =
        if support_sort && available.iter().any(|&j| reads_sorted(j)) {
            let indices = store.support_to_indices(support);
            let mut v: Vec<Vec<u32>> = vec![Vec::new(); config.n_features];
            for &j in &available {
                if reads_sorted(j) {
                    store.sorted_support_indices(j, &indices, &mut v[j]);
                }
            }
            Some(v)
        } else {
            None
        };
    let support_sorted: Option<&[Vec<u32>]> = support_sorted_vecs.as_deref();
    // The support's row list, built once for every binary
    // custom-threshold feature.
    let support_indices: Option<Vec<u32>> =
        if config.n_classes == 2 && available.iter().any(|&j| has_custom_of(j)) {
            Some(store.support_to_indices(support))
        } else {
            None
        };
    // Reference-ensemble sums are read by the shape-mode DPs only.
    let teacher_c0_used = config.shape_mode;

    let alpha = config.reg_alpha;
    let split_reg = (1.0 - alpha) * config.reg;
    let leaf_reg = alpha * config.reg;
    let path_cost = config.path_penalty * n_node as f64 / n_total as f64;
    let min_split_overhead = split_reg + 2.0 * leaf_reg + path_cost;

    let filtered_available: Vec<usize> = if let (Some(ub), false) = (ub_hint, irr_pass) {
        // Same survivors as the pass below (only `min_split_overhead < ub`
        // can fail with the unsafe filter off), in feature order.
        if min_split_overhead >= ub {
            Vec::new()
        } else {
            available
        }
    } else if let Some(ub) = ub_hint {
        let irr_map = |&j: &usize| {
            let irr = if config.n_classes == 2 {
                compute_feature_irreducible_error_binary_mask(
                    order_for(store, support_sorted, j),
                    store.feature_col(j),
                    store.labels(),
                    support,
                    n_total,
                )
            } else {
                compute_feature_irreducible_error_multiclass_mask(
                    order_for(store, support_sorted, j),
                    store.feature_col(j),
                    store.labels(),
                    support,
                    config.n_classes,
                    n_total,
                )
            };
            (j, irr + min_split_overhead)
        };
        // `irr + min_split_overhead` is NOT an admissible LB on
        // any tree-with-j-at-root cost. `irr` (per-unique-value Bayes floor)
        // bounds sum_leaf_loss ONLY when j's children are leaves. Deeper
        // subtrees using OTHER features can drop per-bin misclass below
        // min(c0, c1) (e.g., feature m perfectly separates a bin that j sees
        // as half-and-half), so sum_leaf_loss can go arbitrarily low. The
        // admissible structure-free LB is just `min_split_overhead` (assumes
        // k≥2 leaves and misclass≥0). Tightening back up needs a per-
        // subproblem bound like equivalent-points (see DatasetStore
        // eqp_misclass) — not a per-feature filter.
        // This fix applies unconditionally (max_k=2 has the same bug: any
        // deeper split using another feature beats the per-bin floor).
        //
        // We still
        // *compute* irr_j and use it as a candidate-ordering heuristic via
        // the `surviving.sort_by` below. Lowest-irr features go through DP
        // first; their candidates are typically the cheapest by split_ub
        // and tighten Phase A's `node_upperbound` cascade earliest. No
        // admissibility risk: every feature still passes the admissible
        // `min_split_overhead < ub` filter; the irr value is informational.
        // When
        // `depth_remaining <= unsafe_filter_depth_threshold`, also filter
        // on the unsafe `irr + min_split_overhead < ub` check. At depth=1
        // this IS admissible (children must be leaves). At deeper depths
        // it is not admissible (a heuristic setting).
        let pass_filter = |lb: f64| -> bool {
            if min_split_overhead >= ub {
                return false;
            }
            if unsafe_filter_active && lb >= ub {
                return false;
            }
            true
        };
        let mut surviving: Vec<(usize, f64)> = if config.sequential_candidate_gen {
            available
                .iter()
                .map(irr_map)
                .filter(|&(_, lb)| pass_filter(lb))
                .collect()
        } else {
            available
                .par_iter()
                .map(irr_map)
                .filter(|&(_, lb)| pass_filter(lb))
                .collect()
        };
        surviving.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        surviving.into_iter().map(|(j, _)| j).collect()
    } else {
        available
    };

    let n_arities = config.max_k.saturating_sub(1);
    let mut cand_stats = CandGenStats {
        n_available: n_available as u64,
        n_filtered_available: filtered_available.len() as u64,
        beam_gap_per_arity: vec![0.0; n_arities],
        beam_saturated_cells_per_arity: vec![0; n_arities],
        n_candidates_beam_truncated: 0,
        n_teacher_dp_active_nodes: 0,
    };
    if filtered_available.is_empty() {
        return (vec![], cand_stats);
    }

    let weights = store.weights();
    let weight_scale = store.n_samples() as f64;
    let beam_schedule: Vec<usize> = (2..=config.max_k)
        .map(|k| (1 << (config.max_k - k)) * beam_width)
        .collect();
    let search_beam_schedule = search_capped_beam(config, &beam_schedule);

    let candidate_gen_closure = |&j: &usize| -> (Vec<SplitCandidate>, Option<DpBeamStats>, u64) {
        let mut teacher_dp_active_local: u64 = 0;
        let inner = scratch_bank.with_worker(|scratch| {
            scratch.clear_hot_buffers();

            // ── Categorical feature fast-path ────────────────────────
            // Each unique category value becomes its own bin. Skip
            // transition compression and force shape-collapse DP
            // (provably optimal for binary via Fisher's theorem).
            if config.feature_types[j] == FeatureType::Cat {
                let has_custom = config
                    .thresholds_per_feature
                    .as_ref()
                    .map_or(false, |m| m.contains_key(&j));
                if !has_custom {
                    // Use fill_bins with a large cap so every category gets its own bin
                    if config.n_classes == 2 {
                        if fill_bins_from_sorted_binary_mask(
                            order_for(store, support_sorted, j),
                            store.feature_col(j),
                            store.labels(),
                            support,
                            1_000_000,
                            weights,
                            weight_scale,
                            scratch,
                        )
                        .is_none()
                        {
                            return (Vec::new(), None);
                        }
                        let thresholds = scratch.thresholds.clone();
                        let bin_n = scratch.bin_n.clone();
                        let bin_c0 = scratch.bin_c0.clone();
                        if thresholds.is_empty() {
                            return (Vec::new(), None);
                        }
                        // Force shape collapse — no transition compression
                        let crit = match config.dp_criterion {
                            DpCriterion::Twoing => DpCriterion::Gini,
                            DpCriterion::Rush => DpCriterion::Gini,
                            DpCriterion::Depth2 => DpCriterion::Gini,
                            DpCriterion::Depth2Misclass => DpCriterion::Misclass,
                            other => other,
                        };
                        let ladder = shape_collapse_topb_binary_min_child(
                            &thresholds,
                            &bin_n,
                            &bin_c0,
                            config.max_k,
                            n_total,
                            &search_beam_schedule,
                            crit,
                            similar_lb,
                            config.reg_alpha * config.reg, // leaf_reg
                            config.min_samples_leaf,
                        );
                        let candidates: Vec<SplitCandidate> = ladder
                            .into_iter()
                            .map(|(k, info)| {
                                let (rank_cost, split_ub) = score_candidate(
                                    &info.bin_n,
                                    &info.bin_counts,
                                    config,
                                    n_total,
                                    path_cost,
                                    info.order,
                                );
                                let components = components_count(k, info.group_map.as_deref());
                                SplitCandidate {
                                    rank_cost,
                                    split_ub,
                                    k,
                                    feature_index: j,
                                    thresholds: config
                                        .threshold_interner
                                        .intern_slice(&info.thresholds),
                                    group_map: info.group_map,
                                    child_n: info.bin_n,
                                    child_counts: info.bin_counts,
                                    family: crate::types::CAND_FAMILY_PRIMARY,
                                    components,
                                }
                            })
                            .collect();
                        return (candidates, None);
                    } else {
                        // Multiclass categorical
                        let Some(bin_counts) = fill_bins_from_sorted_multiclass_mask(
                            order_for(store, support_sorted, j),
                            store.feature_col(j),
                            store.labels(),
                            support,
                            config.n_classes,
                            1_000_000,
                            weights,
                            weight_scale,
                            scratch,
                        ) else {
                            return (Vec::new(), None);
                        };
                        let thresholds = scratch.thresholds.clone();
                        let bin_n = scratch.bin_n.clone();
                        if thresholds.is_empty() {
                            return (Vec::new(), None);
                        }
                        let base_crit = match config.dp_criterion {
                            DpCriterion::Depth2 | DpCriterion::Rush => DpCriterion::Gini,
                            DpCriterion::Depth2Misclass => DpCriterion::Misclass,
                            _ => config.dp_criterion,
                        };
                        let atom_teacher_prob = config.teacher_prob.as_deref().and_then(|tp| {
                            compute_bin_teacher_prob_multiclass(
                                order_for(store, support_sorted, j),
                                store.feature_col(j),
                                &thresholds,
                                support,
                                weights,
                                weight_scale,
                                tp,
                                config.n_classes,
                            )
                        });
                        let mu_node = if config.msdt.shape_boundary_tax > 0.0 {
                            config.msdt.shape_boundary_tax / (n_total.max(1) as f64)
                        } else {
                            0.0
                        };
                        let ladder = shape_collapse_topb_pairwise_multiclass_min_child(
                            &thresholds,
                            &bin_n,
                            &bin_counts,
                            atom_teacher_prob.as_deref(),
                            config.max_k,
                            n_total,
                            &search_beam_schedule,
                            base_crit,
                            similar_lb,
                            config.reg_alpha * config.reg, // leaf_reg
                            mu_node,
                            config.min_samples_leaf,
                        );
                        let candidates: Vec<SplitCandidate> = ladder
                            .into_iter()
                            .map(|(k, info)| {
                                let (rank_cost, split_ub) = score_candidate(
                                    &info.bin_n,
                                    &info.bin_counts,
                                    config,
                                    n_total,
                                    path_cost,
                                    info.order,
                                );
                                let components = components_count(k, info.group_map.as_deref());
                                SplitCandidate {
                                    rank_cost,
                                    split_ub,
                                    k,
                                    feature_index: j,
                                    thresholds: config
                                        .threshold_interner
                                        .intern_slice(&info.thresholds),
                                    group_map: info.group_map,
                                    child_n: info.bin_n,
                                    child_counts: info.bin_counts,
                                    family: crate::types::CAND_FAMILY_PRIMARY,
                                    components,
                                }
                            })
                            .collect();
                        return (candidates, None);
                    }
                }
                // If has_custom, fall through to the normal path below
            }

            let has_custom_thresholds = config
                .thresholds_per_feature
                .as_ref()
                .map_or(false, |m| m.contains_key(&j));

            let thresholds: Vec<f64>;
            let bin_n_raw: Vec<i64>;
            let bin_c0_raw: Vec<i64>;
            let bin_teacher_c0_raw: Option<Vec<f64>>;
            let bin_counts_raw: Vec<Vec<i64>>;

            if config.n_classes == 2 {
                if has_custom_thresholds {
                    thresholds = config
                        .thresholds_per_feature
                        .as_ref()
                        .and_then(|m| m.get(&j))
                        .cloned()
                        .unwrap_or_default();
                    if thresholds.is_empty() {
                        return (Vec::new(), None);
                    }
                    scratch.bin_n.clear();
                    scratch.bin_c0.clear();
                    scratch.bin_n.resize(thresholds.len() + 1, 0);
                    scratch.bin_c0.resize(thresholds.len() + 1, 0);
                    scratch.bin_teacher_c0.clear();
                    let teacher_c0 = if teacher_c0_used {
                        config.teacher_prob_c0.as_deref()
                    } else {
                        None
                    };
                    if teacher_c0.is_some() {
                        scratch.bin_teacher_c0.resize(thresholds.len() + 1, 0.0);
                    }
                    let feature_col = store.feature_col(j);
                    let owned_indices;
                    let rows: &[u32] = match support_indices.as_deref() {
                        Some(r) => r,
                        None => {
                            owned_indices = store.support_to_indices(support);
                            &owned_indices
                        }
                    };
                    // Select rank table: the bin of every row from the
                    // store's table of the same `searchsorted_right` on
                    // this exact list (no weights / reference-ensemble sums / shape
                    // mode: plain counts).
                    let rank_tab =
                        if weights.is_none() && teacher_c0.is_none() && !config.shape_mode {
                            store.select_rank(j, &thresholds)
                        } else {
                            None
                        };
                    if let Some(rank) = rank_tab {
                        let labels = store.labels();
                        for &idx in rows {
                            let r = idx as usize;
                            let bin = rank[r] as usize;
                            scratch.bin_n[bin] += 1;
                            scratch.bin_c0[bin] += (labels[r] == 0) as i64;
                        }
                    } else {
                        for &idx in rows {
                            let idx_usize = idx as usize;
                            let bin = searchsorted_right(&thresholds, feature_col[idx_usize]);
                            let wi = if let Some(w) = weights {
                                (w[idx_usize] * weight_scale).round() as i64
                            } else {
                                1
                            };
                            scratch.bin_n[bin] += wi;
                            if store.labels()[idx_usize] == 0 {
                                scratch.bin_c0[bin] += wi;
                            }
                            if let Some(tp) = teacher_c0 {
                                scratch.bin_teacher_c0[bin] += tp[idx_usize] * wi as f64;
                            }
                        }
                    }
                    bin_n_raw = scratch.bin_n.clone();
                    bin_c0_raw = scratch.bin_c0.clone();
                    bin_teacher_c0_raw = teacher_c0.map(|_| scratch.bin_teacher_c0.clone());
                } else {
                    if fill_bins_from_sorted_binary_mask(
                        order_for(store, support_sorted, j),
                        store.feature_col(j),
                        store.labels(),
                        support,
                        FALLBACK_MAX_BINS,
                        weights,
                        weight_scale,
                        scratch,
                    )
                    .is_none()
                    {
                        return (Vec::new(), None);
                    }
                    thresholds = scratch.thresholds.clone();
                    bin_n_raw = scratch.bin_n.clone();
                    bin_c0_raw = scratch.bin_c0.clone();
                    bin_teacher_c0_raw = None;
                }
                bin_counts_raw = Vec::new();
            } else {
                bin_teacher_c0_raw = None;
                if has_custom_thresholds {
                    thresholds = config
                        .thresholds_per_feature
                        .as_ref()
                        .and_then(|m| m.get(&j))
                        .cloned()
                        .unwrap_or_default();
                    if thresholds.is_empty() {
                        return (Vec::new(), None);
                    }
                    bin_counts_raw = fill_bin_samples_sorted_multiclass_mask(
                        order_for(store, support_sorted, j),
                        store.feature_col(j),
                        store.labels(),
                        &thresholds,
                        config.n_classes,
                        support,
                        weights,
                        weight_scale,
                        scratch,
                    );
                    bin_n_raw = scratch.bin_n.clone();
                } else {
                    let Some(bin_counts) = fill_bins_from_sorted_multiclass_mask(
                        order_for(store, support_sorted, j),
                        store.feature_col(j),
                        store.labels(),
                        support,
                        config.n_classes,
                        FALLBACK_MAX_BINS,
                        weights,
                        weight_scale,
                        scratch,
                    ) else {
                        return (Vec::new(), None);
                    };
                    thresholds = scratch.thresholds.clone();
                    bin_n_raw = scratch.bin_n.clone();
                    bin_counts_raw = bin_counts;
                }
                bin_c0_raw = Vec::new();
            }

            if thresholds.is_empty() {
                return (Vec::new(), None);
            }

            let (ladder, family_tags, dp_beam_stats): (
                Vec<(usize, CollapseEntry)>,
                Vec<u8>,
                Option<DpBeamStats>,
            ) = if config.n_classes == 2 {
                let (thresholds, bin_n, bin_c0) = if transition_delta > 0.0 {
                    let (ct, cn, cc) = compress_transition_points(
                        &thresholds,
                        &bin_n_raw,
                        &bin_c0_raw,
                        transition_delta,
                    );
                    if ct.is_empty() {
                        return (Vec::new(), None);
                    }
                    (ct, cn, cc)
                } else {
                    (thresholds, bin_n_raw, bin_c0_raw)
                };

                let (thresholds, bin_n, bin_c0, atom_proj) =
                    maybe_atomize_binary(config, thresholds, bin_n, bin_c0);

                // Compute atom-level reference-ensemble class-0 mass for the shape
                // ordering. Only runs when the caller provided a reference ensemble.
                // (`config.teacher_prob_c0`). Second scan over the
                // feature's sorted order — cheap (O(n_support)).
                let atom_teacher_c0: Option<Vec<f64>> = if !teacher_c0_used {
                    None
                } else if config.teacher_prob_c0.is_some()
                    && !config.msdt.atom_mode
                    && bin_teacher_c0_raw
                        .as_ref()
                        .is_some_and(|v| v.len() == bin_n.len())
                {
                    bin_teacher_c0_raw
                } else {
                    config.teacher_prob_c0.as_deref().map(|tp| {
                        compute_atom_teacher_c0_binary(
                            order_for(store, support_sorted, j),
                            store.feature_col(j),
                            &thresholds,
                            support,
                            tp,
                            bin_n.len(),
                        )
                    })
                };

                if config.shape_mode
                    && config.max_k <= 3
                    && !config.msdt.atom_mode
                    && !config.msdt.cleanup_mode
                {
                    let base_shape_local_beam =
                        vec![config.beam_width.max(1); config.max_k.saturating_sub(1)];
                    let shape_local_beam = search_capped_beam(config, &base_shape_local_beam);
                    let candidates = exact_shape_reference_binary_candidates(
                        &thresholds,
                        &bin_n,
                        &bin_c0,
                        atom_teacher_c0.as_deref(),
                        j,
                        config,
                        &shape_local_beam,
                        n_total,
                        path_cost,
                        ub_hint,
                    );
                    return (candidates, None);
                }
                if !config.shape_mode
                    && config.max_k <= 3
                    && matches!(config.dp_criterion, DpCriterion::Gini)
                    && !config.adaptive_beam
                    && !config.msdt.atom_mode
                    && !shape_includes_contiguous_baseline(config)
                    && !config.msdt.cleanup_mode
                {
                    let base_local_beam =
                        vec![config.beam_width.max(1); config.max_k.saturating_sub(1)];
                    let local_beam = search_capped_beam(config, &base_local_beam);
                    let candidates = exact_contiguous_reference_binary_candidates(
                        &thresholds,
                        &bin_n,
                        &bin_c0,
                        j,
                        config,
                        &local_beam,
                        n_total,
                        path_cost,
                        ub_hint,
                    );
                    return (candidates, None);
                }

                let raw_eff_beam: Vec<usize> = if config.adaptive_beam
                    && config.max_k > 2
                    && !matches!(config.dp_criterion, DpCriterion::Rush)
                {
                    let narrow_sched = vec![1usize; config.max_k - 1];
                    let (narrow, _) = collapse_bins_topb_binary_min_child(
                        &thresholds,
                        &bin_n,
                        &bin_c0,
                        config.max_k,
                        n_total,
                        &narrow_sched,
                        DpCriterion::Gini,
                        None,
                        config.min_samples_leaf,
                    );
                    let mut best_k = vec![f64::INFINITY; config.max_k + 1];
                    for (k, entry) in &narrow {
                        if entry.cost < best_k[*k] {
                            best_k[*k] = entry.cost;
                        }
                    }
                    let total_n: i64 = bin_n.iter().sum();
                    let total_c0: i64 = bin_c0.iter().sum();
                    best_k[1] = if total_n > 0 {
                        let c1 = total_n - total_c0;
                        (total_n as f64 - (total_c0 * total_c0 + c1 * c1) as f64 / total_n as f64)
                            / n_total as f64
                    } else {
                        0.0
                    };

                    (2..=config.max_k)
                        .map(|k| {
                            let base = beam_schedule[k - 2];
                            let gain = (best_k[k - 1] - best_k[k]).max(0.0);
                            let bonus = (gain / config.reg).floor() as usize;
                            (base + bonus).clamp(base, BEAM_MAX)
                        })
                        .collect()
                } else {
                    beam_schedule.clone()
                };
                let eff_beam = search_capped_beam(config, &raw_eff_beam);

                let (primary_ladder, dp_beam_stats): (
                    Vec<(usize, CollapseEntry)>,
                    Option<DpBeamStats>,
                ) = if config.shape_mode {
                    let crit = match config.dp_criterion {
                        DpCriterion::Twoing => DpCriterion::Gini,
                        DpCriterion::Rush => DpCriterion::Gini,
                        DpCriterion::Depth2 => DpCriterion::Gini,
                        DpCriterion::Depth2Misclass => DpCriterion::Misclass,
                        other => other,
                    };
                    // Reference-ensemble DP blend fires at the primary shape site
                    // when α < 1.0 with a reference ensemble available and Gini crit.
                    // (Binary shape + Gini only.)
                    let dp_teacher_active = matches!(crit, DpCriterion::Gini)
                        && config.msdt.teacher_alpha < 1.0 - 1e-12
                        && atom_teacher_c0.is_some();
                    let dp_teacher_alpha = if dp_teacher_active {
                        config.msdt.teacher_alpha
                    } else {
                        1.0
                    };
                    if dp_teacher_active {
                        teacher_dp_active_local = teacher_dp_active_local.saturating_add(1);
                    }
                    // Route through
                    // the _taxed entry point so mu_node/smoothing_alpha/
                    // per_atom_teacher_prior actually flow into the DP.
                    let mu_node_primary = if config.msdt.shape_boundary_tax > 0.0 {
                        config.msdt.shape_boundary_tax / (n_total.max(1) as f64)
                    } else {
                        0.0
                    };
                    let shape_entries = shape_collapse_topb_binary_with_teacher_smoothed_taxed(
                        &thresholds,
                        &bin_n,
                        &bin_c0,
                        config.max_k,
                        n_total,
                        &eff_beam,
                        crit,
                        similar_lb,
                        atom_teacher_c0.as_deref(),
                        dp_teacher_alpha,
                        config.reg_alpha * config.reg, // leaf_reg
                        config.shape_sort_smoothing,   // smoothing_alpha
                        mu_node_primary,               // boundary tax
                        config.msdt.shape_sort_per_atom_teacher_prior, // per-atom prior
                        config.min_samples_leaf,
                    );
                    (
                        expand_atom_entries_to_bins(shape_entries, atom_proj.as_ref()),
                        None,
                    )
                } else {
                    match config.dp_criterion {
                        DpCriterion::Rush => {
                            // No node budget: the DP's suffix-bound
                            // prune gates stay inert (only BEAM_MAX limits it).
                            let ub_best = f64::INFINITY;
                            let (l, s) = rush_dp_binary_fast(
                                &thresholds,
                                &bin_n,
                                &bin_c0,
                                config.max_k,
                                n_total,
                                &search_beam_schedule,
                                1,
                                ub_best,
                                0.0,
                                1.0,
                                dp_opts,
                            );
                            (l, Some(s))
                        }
                        DpCriterion::Twoing => {
                            let (l, s) = collapse_bins_topb_binary_min_child(
                                &thresholds,
                                &bin_n,
                                &bin_c0,
                                config.max_k,
                                n_total,
                                &eff_beam,
                                DpCriterion::Gini,
                                None,
                                config.min_samples_leaf,
                            );
                            (l, Some(s))
                        }
                        DpCriterion::Depth2 => {
                            let (mut l, s) = collapse_bins_topb_binary_min_child(
                                &thresholds,
                                &bin_n,
                                &bin_c0,
                                config.max_k,
                                n_total,
                                &eff_beam,
                                DpCriterion::Gini,
                                None,
                                config.min_samples_leaf,
                            );
                            let (l2, _) = collapse_bins_topb_binary_min_child(
                                &thresholds,
                                &bin_n,
                                &bin_c0,
                                config.max_k,
                                n_total,
                                &eff_beam,
                                DpCriterion::Depth2,
                                None,
                                config.min_samples_leaf,
                            );
                            l.extend(l2);
                            (l, Some(s))
                        }
                        DpCriterion::Depth2Misclass => {
                            let (mut l, s) = collapse_bins_topb_binary_min_child(
                                &thresholds,
                                &bin_n,
                                &bin_c0,
                                config.max_k,
                                n_total,
                                &eff_beam,
                                DpCriterion::Misclass,
                                None,
                                config.min_samples_leaf,
                            );
                            let (l2, _) = collapse_bins_topb_binary_min_child(
                                &thresholds,
                                &bin_n,
                                &bin_c0,
                                config.max_k,
                                n_total,
                                &eff_beam,
                                DpCriterion::Depth2Misclass,
                                None,
                                config.min_samples_leaf,
                            );
                            l.extend(l2);
                            (l, Some(s))
                        }
                        // Gini: the capped speed path (`ladder_opts`),
                        // same ladder bitwise: a lex top-c cell is the
                        // c-prefix of the BEAM_MAX cell and extraction
                        // reads only the top eff_beam[k] entries (see
                        // `collapse_binary_fast`). Only `DpBeamStats`
                        // (diagnostics) differ.
                        crit if matches!(crit, DpCriterion::Gini) => {
                            let (l, s) = collapse_binary_fast(
                                &thresholds,
                                &bin_n,
                                &bin_c0,
                                config.max_k,
                                n_total,
                                &eff_beam,
                                crit,
                                config.min_samples_leaf,
                                ladder_opts,
                            );
                            (l, Some(s))
                        }
                        crit => {
                            let (l, s) = collapse_bins_topb_binary_min_child(
                                &thresholds,
                                &bin_n,
                                &bin_c0,
                                config.max_k,
                                n_total,
                                &eff_beam,
                                crit,
                                similar_lb,
                                config.min_samples_leaf,
                            );
                            (l, Some(s))
                        }
                    }
                };
                let mut ladder = primary_ladder;
                let mut family_tags: Vec<u8> = vec![CAND_FAMILY_PRIMARY; ladder.len()];

                // Companion misclassification pass in the primary geometry.
                // (Contiguous geometry with the DP speed paths:
                // `run_dp_pass_binary` there is
                // `collapse_bins_topb_binary_min_child(.., Misclass, None, ..)`.)
                // Its stats are discarded, so `ladder_opts` (cap +
                // topcell; prefix argument at `collapse_binary_fast`)
                // is always safe.
                if emit_misclassification_companion(config.dp_criterion) && !config.shape_mode {
                    let (ext, _) = collapse_binary_fast(
                        &thresholds,
                        &bin_n,
                        &bin_c0,
                        config.max_k,
                        n_total,
                        &eff_beam,
                        DpCriterion::Misclass,
                        config.min_samples_leaf,
                        ladder_opts,
                    );
                    family_tags.extend(std::iter::repeat(CAND_FAMILY_MISCLASS).take(ext.len()));
                    ladder.extend(ext);
                } else if emit_misclassification_companion(config.dp_criterion) {
                    let ext = run_dp_pass_binary(
                        config.shape_mode,
                        DpCriterion::Misclass,
                        config,
                        &thresholds,
                        &bin_n,
                        &bin_c0,
                        n_node,
                        n_total,
                        &eff_beam,
                        atom_proj.as_ref(),
                        similar_lb,
                        atom_teacher_c0.as_deref(),
                        Some(&mut teacher_dp_active_local),
                    );
                    family_tags.extend(std::iter::repeat(CAND_FAMILY_MISCLASS).take(ext.len()));
                    ladder.extend(ext);
                }

                // ShapeSPLINTER also includes ordinary contiguous candidates.
                if shape_includes_contiguous_baseline(config) {
                    let alt_primary_crit = if primary_is_misclass(config.dp_criterion) {
                        DpCriterion::Misclass
                    } else {
                        DpCriterion::Gini
                    };
                    let ext = run_dp_pass_binary(
                        !config.shape_mode,
                        alt_primary_crit,
                        config,
                        &thresholds,
                        &bin_n,
                        &bin_c0,
                        n_node,
                        n_total,
                        &eff_beam,
                        atom_proj.as_ref(),
                        similar_lb,
                        atom_teacher_c0.as_deref(),
                        Some(&mut teacher_dp_active_local),
                    );
                    family_tags
                        .extend(std::iter::repeat(CAND_FAMILY_ALT_MODE_PRIMARY).take(ext.len()));
                    ladder.extend(ext);

                    if emit_misclassification_companion(config.dp_criterion) {
                        let ext = run_dp_pass_binary(
                            !config.shape_mode,
                            DpCriterion::Misclass,
                            config,
                            &thresholds,
                            &bin_n,
                            &bin_c0,
                            n_node,
                            n_total,
                            &eff_beam,
                            atom_proj.as_ref(),
                            similar_lb,
                            atom_teacher_c0.as_deref(),
                            Some(&mut teacher_dp_active_local),
                        );
                        family_tags.extend(
                            std::iter::repeat(CAND_FAMILY_ALT_MODE_MISCLASS).take(ext.len()),
                        );
                        ladder.extend(ext);
                    }
                }

                (ladder, family_tags, dp_beam_stats)
            } else {
                let (thresholds, bin_n, bin_counts) = if transition_delta > 0.0 {
                    let (ct, cn, cc) = compress_transition_points_multiclass(
                        &thresholds,
                        &bin_n_raw,
                        &bin_counts_raw,
                        transition_delta,
                    );
                    if ct.is_empty() {
                        return (Vec::new(), None);
                    }
                    (ct, cn, cc)
                } else {
                    (thresholds, bin_n_raw, bin_counts_raw)
                };

                let (thresholds, bin_n, bin_counts, atom_proj) =
                    maybe_atomize_multiclass(config, thresholds, bin_n, bin_counts);
                let atom_teacher_prob: Option<Vec<f64>> = if config.shape_mode {
                    config.teacher_prob.as_deref().and_then(|tp| {
                        compute_bin_teacher_prob_multiclass(
                            order_for(store, support_sorted, j),
                            store.feature_col(j),
                            &thresholds,
                            support,
                            weights,
                            weight_scale,
                            tp,
                            config.n_classes,
                        )
                    })
                } else {
                    None
                };

                let base_crit = match config.dp_criterion {
                    DpCriterion::Depth2 | DpCriterion::Rush => DpCriterion::Gini,
                    DpCriterion::Depth2Misclass => DpCriterion::Misclass,
                    _ => config.dp_criterion,
                };
                let (primary_ladder, dp_beam_stats): (
                    Vec<(usize, CollapseEntry)>,
                    Option<DpBeamStats>,
                ) = if config.shape_mode {
                    let mu_node = if config.msdt.shape_boundary_tax > 0.0 {
                        config.msdt.shape_boundary_tax / (n_total.max(1) as f64)
                    } else {
                        0.0
                    };
                    let shape_entries = shape_collapse_topb_pairwise_multiclass_min_child(
                        &thresholds,
                        &bin_n,
                        &bin_counts,
                        atom_teacher_prob.as_deref(),
                        config.max_k,
                        n_total,
                        &search_beam_schedule,
                        base_crit,
                        similar_lb,
                        config.reg_alpha * config.reg, // leaf_reg
                        mu_node,
                        config.min_samples_leaf,
                    );
                    (
                        expand_atom_entries_to_bins(shape_entries, atom_proj.as_ref()),
                        None,
                    )
                } else {
                    let (mut l, s) = collapse_bins_topb_min_child(
                        &thresholds,
                        &bin_n,
                        &bin_counts,
                        config.max_k,
                        n_total,
                        &search_beam_schedule,
                        base_crit,
                        similar_lb,
                        config.reg_alpha * config.reg, // leaf_reg
                        config.min_samples_leaf,
                    );
                    if config.dp_criterion == DpCriterion::Depth2 {
                        let (l2, _) = collapse_bins_topb_min_child(
                            &thresholds,
                            &bin_n,
                            &bin_counts,
                            config.max_k,
                            n_total,
                            &search_beam_schedule,
                            DpCriterion::Depth2,
                            None,
                            config.reg_alpha * config.reg,
                            config.min_samples_leaf,
                        );
                        l.extend(l2);
                    }
                    if config.dp_criterion == DpCriterion::Depth2Misclass {
                        let (l2, _) = collapse_bins_topb_min_child(
                            &thresholds,
                            &bin_n,
                            &bin_counts,
                            config.max_k,
                            n_total,
                            &search_beam_schedule,
                            DpCriterion::Depth2Misclass,
                            None,
                            config.reg_alpha * config.reg,
                            config.min_samples_leaf,
                        );
                        l.extend(l2);
                    }
                    (l, Some(s))
                };
                let mut ladder = primary_ladder;
                let mut family_tags: Vec<u8> = vec![CAND_FAMILY_PRIMARY; ladder.len()];

                if emit_misclassification_companion(config.dp_criterion) {
                    let ext = run_dp_pass_multiclass(
                        config.shape_mode,
                        DpCriterion::Misclass,
                        config,
                        &thresholds,
                        &bin_n,
                        &bin_counts,
                        n_node,
                        n_total,
                        &search_beam_schedule,
                        atom_proj.as_ref(),
                        similar_lb,
                        atom_teacher_prob.as_deref(),
                    );
                    family_tags.extend(std::iter::repeat(CAND_FAMILY_MISCLASS).take(ext.len()));
                    ladder.extend(ext);
                }

                if shape_includes_contiguous_baseline(config) {
                    let alt_primary_crit = if primary_is_misclass(config.dp_criterion) {
                        DpCriterion::Misclass
                    } else {
                        DpCriterion::Gini
                    };
                    let ext = run_dp_pass_multiclass(
                        !config.shape_mode,
                        alt_primary_crit,
                        config,
                        &thresholds,
                        &bin_n,
                        &bin_counts,
                        n_node,
                        n_total,
                        &search_beam_schedule,
                        atom_proj.as_ref(),
                        similar_lb,
                        atom_teacher_prob.as_deref(),
                    );
                    family_tags
                        .extend(std::iter::repeat(CAND_FAMILY_ALT_MODE_PRIMARY).take(ext.len()));
                    ladder.extend(ext);

                    if emit_misclassification_companion(config.dp_criterion) {
                        let ext = run_dp_pass_multiclass(
                            !config.shape_mode,
                            DpCriterion::Misclass,
                            config,
                            &thresholds,
                            &bin_n,
                            &bin_counts,
                            n_node,
                            n_total,
                            &search_beam_schedule,
                            atom_proj.as_ref(),
                            similar_lb,
                            atom_teacher_prob.as_deref(),
                        );
                        family_tags.extend(
                            std::iter::repeat(CAND_FAMILY_ALT_MODE_MISCLASS).take(ext.len()),
                        );
                        ladder.extend(ext);
                    }
                }

                (ladder, family_tags, dp_beam_stats)
            };

            let candidates: Vec<SplitCandidate> = ladder
                .into_iter()
                .zip(family_tags.into_iter())
                .map(|((k, info), family)| {
                    let (rank_cost, split_ub) = score_candidate(
                        &info.bin_n,
                        &info.bin_counts,
                        config,
                        n_total,
                        path_cost,
                        info.order,
                    );
                    let components = components_count(k, info.group_map.as_deref());
                    SplitCandidate {
                        rank_cost,
                        split_ub,
                        k,
                        feature_index: j,
                        thresholds: config.threshold_interner.intern_slice(&info.thresholds),
                        group_map: info.group_map,
                        child_n: info.bin_n,
                        child_counts: info.bin_counts,
                        family,
                        components,
                    }
                })
                .collect::<Vec<_>>();
            (candidates, dp_beam_stats)
        });
        let (candidates, dp_beam_stats) = inner;
        (candidates, dp_beam_stats, teacher_dp_active_local)
    };

    let per_feature: Vec<(Vec<SplitCandidate>, Option<DpBeamStats>, u64)> =
        if config.sequential_candidate_gen {
            filtered_available
                .iter()
                .map(candidate_gen_closure)
                .collect()
        } else {
            filtered_available
                .par_iter()
                .map(candidate_gen_closure)
                .collect()
        };

    // Aggregate beam stats and reference-ensemble DP counts across features.
    for (_, maybe_stats, teacher_active) in &per_feature {
        if let Some(stats) = maybe_stats {
            for (i, &g) in stats.gap_at_final.iter().enumerate() {
                if i < n_arities {
                    if g > cand_stats.beam_gap_per_arity[i] {
                        cand_stats.beam_gap_per_arity[i] = g;
                    }
                }
            }
            for (i, &r) in stats.n_saturated_cells.iter().enumerate() {
                if i < n_arities {
                    cand_stats.beam_saturated_cells_per_arity[i] += r;
                }
            }
        }
        cand_stats.n_teacher_dp_active_nodes = cand_stats
            .n_teacher_dp_active_nodes
            .saturating_add(*teacher_active);
    }

    let mut all: Vec<SplitCandidate> = dedup_split_candidates(
        per_feature
            .into_iter()
            .flat_map(|(cands, _, _)| cands)
            .collect(),
    );
    if config.msdt.cleanup_mode {
        let cleanup_variants = cleanup_collapse_contiguous(&all, config, path_cost, n_total);
        all.extend(cleanup_variants);
    }
    all.sort_by(compare_split_candidates);
    let (all, n_truncated) = select_top_k_with_anchors(all, config, n_total);
    cand_stats.n_candidates_beam_truncated += n_truncated;

    (all, cand_stats)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_reference_edges_ignore_empty_bin_routing() {
        let candidate = exact_interval_binary_candidate(
            &[0.5, 1.5, 2.5, 3.5, 4.5, 5.5],
            &[10, 2, 2, 10],
            &[10, 0, 0, 10],
            &[0, 2, 4, 6],
            7,
            &[0, 3, 1, 2],
            &[2],
            CAND_FAMILY_PRIMARY,
            1,
            true,
        )
        .unwrap();
        assert_eq!(
            candidate.order,
            CandidateOrder::Reference { boundary_edges: 2 }
        );
        // Empty bins route to the largest child, which creates extra full-grid
        // transitions. Those transitions have no training mass and no charge.
        let full_map = candidate.group_map.unwrap();
        assert_eq!(
            reference_order(&full_map),
            CandidateOrder::Reference { boundary_edges: 4 }
        );
    }

    #[test]
    fn reference_dp_preserves_occupied_edges_through_projection() {
        let thresholds = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5];
        let bin_n = [10, 0, 2, 0, 2, 0, 10];
        let bin_c0 = [10, 0, 0, 0, 0, 0, 10];
        for criterion in [DpCriterion::Gini, DpCriterion::Misclass] {
            let entries = shape_collapse_topb_binary_with_teacher_smoothed_taxed(
                &thresholds,
                &bin_n,
                &bin_c0,
                3,
                24,
                &[8, 8],
                criterion,
                None,
                None,
                1.0,
                0.01,
                0.0,
                1.0 / 24.0,
                false,
                1,
            );
            assert!(!entries.is_empty());
            for (_, entry) in entries {
                let occupied_map: Vec<usize> = entry
                    .group_map
                    .as_ref()
                    .unwrap()
                    .iter()
                    .zip(bin_n)
                    .filter_map(|(&group, n)| (n > 0).then_some(group))
                    .collect();
                assert_eq!(entry.order, reference_order(&occupied_map));
                let expected_order = entry.order;
                let projection = (
                    vec![0, 0, 1, 2, 3, 4, 5, 6, 6],
                    (0..8).map(|i| i as f64).collect(),
                );
                let expanded = expand_atom_entries_to_bins(
                    vec![(entry.bin_n.len(), entry)],
                    Some(&projection),
                );
                assert_eq!(expanded[0].1.order, expected_order);
            }
        }
    }

    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            self.0 >> 33
        }
        fn below(&mut self, n: u64) -> u64 {
            self.next() % n
        }
    }

    fn rush_beam_bits(beam: &RushTopB) -> Vec<(u64, u16, u8)> {
        (0..beam.count)
            .map(|i| (beam.costs[i].to_bits(), beam.prev_t[i], beam.prev_b[i]))
            .collect()
    }

    #[test]
    fn rush_sorted_merges_preserve_legacy_ties_and_costs() {
        let mut rng = Lcg(0x4355_5253);
        let mut stamps = vec![0; 24 * BEAM_MAX];
        let mut ranks = vec![0; 24 * BEAM_MAX];
        let mut generation = 0;
        for case in 0..256 {
            let mut a = RushTopB::new();
            let mut b = RushTopB::new();
            let mut legacy_a = RushTopB::new();
            let mut legacy_b = RushTopB::new();
            for predecessor in 1..=20 {
                let mut prev = RushTopB::new();
                let count = 1 + rng.below(BEAM_MAX as u64) as usize;
                for rank in 0..count {
                    // Integer doubled costs deliberately create many ties.
                    prev.try_insert(rng.below(100) as f64, 0, rank as u8);
                }
                let support = rng.below(200) as f64;
                let margin = rng.below(100) as f64 - 50.0;
                for rank in 0..prev.count {
                    let g = prev.costs[rank] - support;
                    legacy_a.try_insert(g + margin, predecessor, rank as u8);
                    legacy_b.try_insert(g - margin, predecessor, rank as u8);
                }
                a.merge_predecessor(&prev, support, margin, predecessor);
                b.merge_predecessor(&prev, support, -margin, predecessor);
                assert_eq!(
                    rush_beam_bits(&a),
                    rush_beam_bits(&legacy_a),
                    "A case={case}"
                );
                assert_eq!(
                    rush_beam_bits(&b),
                    rush_beam_bits(&legacy_b),
                    "B case={case}"
                );
                assert_eq!(a.best_rejected, legacy_a.best_rejected);
                assert_eq!(b.best_rejected, legacy_b.best_rejected);
            }
            let support = 100.0;
            let margin = rng.below(100) as f64 - 50.0;
            let mut ts = [0; BEAM_MAX * 2];
            let mut bs = [0; BEAM_MAX * 2];
            let mut cs = [f64::INFINITY; BEAM_MAX * 2];
            let mut count = 0;
            for i in 0..a.count {
                upsert_rush_candidate(
                    &mut ts,
                    &mut bs,
                    &mut cs,
                    &mut count,
                    a.prev_t[i],
                    a.prev_b[i],
                    support + a.costs[i] - margin,
                );
            }
            for i in 0..b.count {
                upsert_rush_candidate(
                    &mut ts,
                    &mut bs,
                    &mut cs,
                    &mut count,
                    b.prev_t[i],
                    b.prev_b[i],
                    support + b.costs[i] + margin,
                );
            }
            let mut order: Vec<usize> = (0..count).collect();
            order.sort_by(|&x, &y| cs[x].partial_cmp(&cs[y]).unwrap());
            for cap in [1, 3, 32, BEAM_MAX] {
                let mut merged = RushTopB::new();
                merge_rush_channels(
                    &mut merged,
                    &a,
                    &b,
                    support,
                    margin,
                    cap,
                    &mut stamps,
                    &mut ranks,
                    &mut generation,
                    100.0,
                    0.0,
                    0.0,
                    f64::INFINITY,
                );
                let expected: Vec<_> = order
                    .iter()
                    .take(cap)
                    .map(|&i| (cs[i].to_bits(), ts[i], bs[i]))
                    .collect();
                assert_eq!(
                    rush_beam_bits(&merged),
                    expected,
                    "union case={case} cap={cap}"
                );
            }
        }
    }

    /// Bitwise key of a ladder: (k, cost bits, threshold bits, group stats).
    fn ladder_bits(
        l: &[(usize, CollapseEntry)],
    ) -> Vec<(usize, u64, Vec<u64>, Vec<i64>, Vec<Vec<i64>>)> {
        l.iter()
            .map(|(k, e)| {
                assert!(e.group_map.is_none());
                (
                    *k,
                    e.cost.to_bits(),
                    e.thresholds.iter().map(|t| t.to_bits()).collect(),
                    e.bin_n.clone(),
                    e.bin_counts.clone(),
                )
            })
            .collect()
    }

    /// Per-arity extraction beams as the contiguous Select builds them:
    /// raw B_k = min(B·2^(kmax−k), BEAM_MAX), then the top_k cap
    /// (`search_capped_beam`), plus hand-picked non-monotone schedules.
    fn schedules(max_k: usize) -> Vec<Vec<usize>> {
        let mut out = Vec::new();
        for &b in &[1usize, 2, 3, 8, 32, 128] {
            let raw: Vec<usize> = (2..=max_k)
                .map(|k| (b << (max_k - k)).min(BEAM_MAX))
                .collect();
            for top_k in [None, Some(1usize), Some(3), Some(10), Some(64)] {
                out.push(match top_k {
                    None => raw.clone(),
                    Some(t) => raw.iter().map(|&x| x.max(1).min(t.max(1))).collect(),
                });
            }
        }
        let odd: [&[usize]; 4] = [&[1, 128, 2, 7], &[128, 1, 1, 128], &[5, 2, 9, 1], &[2]];
        for s in odd {
            out.push(s.iter().copied().take(max_k - 1).collect());
        }
        out
    }

    /// The fast Rush DP (`rush_dp_binary_fast`, the slot-table channel
    /// merge on per-thread tables, with and without cap + topcell) returns
    /// the reference Rush DP's ladder (`rush_dp_binary`) bitwise and in
    /// order on random bins with many cost ties.
    #[test]
    fn fast_rush_dp_matches_reference_ladder() {
        let capped = DpOpts {
            topcell: true,
            cap: true,
        };
        let uncapped = DpOpts {
            topcell: false,
            cap: false,
        };
        let mut rng = Lcg(0x7255_5348);
        let (mut n_cases, mut n_entries) = (0usize, 0usize);
        for &l in &[2usize, 3, 5, 9, 17, 30, 44] {
            for trial in 0..4 {
                let max_n = if trial % 2 == 0 { 3 } else { 40 };
                let mut bin_n = Vec::with_capacity(l);
                let mut bin_c0 = Vec::with_capacity(l);
                for _ in 0..l {
                    let n = if rng.below(8) == 0 {
                        0
                    } else {
                        1 + rng.below(max_n) as i64
                    };
                    bin_n.push(n);
                    bin_c0.push(if n == 0 {
                        0
                    } else {
                        rng.below(n as u64 + 1) as i64
                    });
                }
                let thresholds: Vec<f64> =
                    (0..l.saturating_sub(1)).map(|i| i as f64 + 0.5).collect();
                let n_total = bin_n.iter().sum::<i64>().max(1) as usize + rng.below(50) as usize;
                for max_k in 2..=5usize {
                    for sched in schedules(max_k) {
                        let (want, _) = rush_dp_binary(
                            &thresholds,
                            &bin_n,
                            &bin_c0,
                            max_k,
                            n_total,
                            &sched,
                            1,
                            f64::INFINITY,
                            0.0,
                            1.0,
                            None,
                        );
                        for opts in [capped, uncapped] {
                            let (got, _) = rush_dp_binary_fast(
                                &thresholds,
                                &bin_n,
                                &bin_c0,
                                max_k,
                                n_total,
                                &sched,
                                1,
                                f64::INFINITY,
                                0.0,
                                1.0,
                                opts,
                            );
                            assert_eq!(
                                ladder_bits(&want),
                                ladder_bits(&got),
                                "Rush {opts:?} L={l} max_k={max_k} sched={sched:?} \
                                 bin_n={bin_n:?} bin_c0={bin_c0:?}"
                            );
                        }
                        n_cases += 1;
                        n_entries += want.len();
                    }
                }
            }
        }
        assert!(
            n_cases > 500 && n_entries > 10_000,
            "cases {n_cases} entries {n_entries}"
        );
    }

    /// The capped speed path (cap + topcell) returns the same extracted
    /// ladder, bitwise and in order, as the legacy Gini DP and as the
    /// uncapped Misclass speed path, on random bins with many cost ties.
    #[test]
    fn capped_contiguous_dp_matches_legacy_ladder() {
        let capped = DpOpts {
            topcell: true,
            cap: true,
        };
        let uncapped = DpOpts {
            topcell: false,
            cap: false,
        };
        let mut rng = Lcg(0x5eed_e4);
        let mut n_cases = 0usize;
        let mut n_entries = 0usize;
        for &l in &[2usize, 3, 5, 9, 17, 30, 44] {
            for trial in 0..4 {
                // Small counts -> many exact cost ties; some empty bins.
                let max_n = if trial % 2 == 0 { 3 } else { 40 };
                let mut bin_n = Vec::with_capacity(l);
                let mut bin_c0 = Vec::with_capacity(l);
                for _ in 0..l {
                    let n = if rng.below(8) == 0 {
                        0
                    } else {
                        1 + rng.below(max_n) as i64
                    };
                    bin_n.push(n);
                    bin_c0.push(if n == 0 {
                        0
                    } else {
                        rng.below(n as u64 + 1) as i64
                    });
                }
                let thresholds: Vec<f64> =
                    (0..l.saturating_sub(1)).map(|i| i as f64 + 0.5).collect();
                let n_total = bin_n.iter().sum::<i64>().max(1) as usize + rng.below(50) as usize;
                for max_k in 2..=5usize {
                    for sched in schedules(max_k) {
                        for &min_child in &[1usize, 2, 5] {
                            let (want_g, _) = collapse_bins_topb_binary_min_child(
                                &thresholds,
                                &bin_n,
                                &bin_c0,
                                max_k,
                                n_total,
                                &sched,
                                DpCriterion::Gini,
                                None,
                                min_child,
                            );
                            let (got_g, _) = collapse_binary_fast(
                                &thresholds,
                                &bin_n,
                                &bin_c0,
                                max_k,
                                n_total,
                                &sched,
                                DpCriterion::Gini,
                                min_child,
                                capped,
                            );
                            assert_eq!(
                                ladder_bits(&want_g),
                                ladder_bits(&got_g),
                                "Gini L={l} max_k={max_k} sched={sched:?} min_child={min_child} \
                                 bin_n={bin_n:?} bin_c0={bin_c0:?}"
                            );
                            let (got_k, _) = collapse_binary_fast(
                                &thresholds,
                                &bin_n,
                                &bin_c0,
                                max_k,
                                n_total,
                                &sched,
                                DpCriterion::Gini,
                                min_child,
                                uncapped,
                            );
                            assert_eq!(
                                ladder_bits(&want_g),
                                ladder_bits(&got_k),
                                "Gini uncapped L={l} max_k={max_k} sched={sched:?} \
                                 min_child={min_child} bin_n={bin_n:?} bin_c0={bin_c0:?}"
                            );
                            let (want_m, _) = collapse_binary_fast(
                                &thresholds,
                                &bin_n,
                                &bin_c0,
                                max_k,
                                n_total,
                                &sched,
                                DpCriterion::Misclass,
                                min_child,
                                uncapped,
                            );
                            let (legacy_m, _) = collapse_bins_topb_binary_min_child(
                                &thresholds,
                                &bin_n,
                                &bin_c0,
                                max_k,
                                n_total,
                                &sched,
                                DpCriterion::Misclass,
                                None,
                                min_child,
                            );
                            let (got_m, _) = collapse_binary_fast(
                                &thresholds,
                                &bin_n,
                                &bin_c0,
                                max_k,
                                n_total,
                                &sched,
                                DpCriterion::Misclass,
                                min_child,
                                capped,
                            );
                            assert_eq!(
                                ladder_bits(&want_m),
                                ladder_bits(&got_m),
                                "Misclass L={l} max_k={max_k} sched={sched:?} min_child={min_child} \
                                 bin_n={bin_n:?} bin_c0={bin_c0:?}"
                            );
                            assert_eq!(
                                ladder_bits(&legacy_m),
                                ladder_bits(&got_m),
                                "Misclass legacy L={l} max_k={max_k} sched={sched:?} \
                                 min_child={min_child} bin_n={bin_n:?} bin_c0={bin_c0:?}"
                            );
                            n_cases += 1;
                            n_entries += want_g.len() + want_m.len();
                        }
                    }
                }
            }
        }
        assert!(
            n_cases > 1000 && n_entries > 100_000,
            "cases {n_cases} entries {n_entries}"
        );
    }
}
