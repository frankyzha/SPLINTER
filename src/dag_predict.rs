//! Exact per-sample prediction distributions from the Rashomon DAG.
//!
//! Computes P(class=1 | x_i, Rashomon set) for each sample by walking the
//! FactoredDag with a memoized DP. No tree sampling or extraction needed.
//!
//! Uses the "cost-blind" approximation: each split is weighted by its total
//! tree_count (from cost-constrained convolution) but the child's prediction
//! distribution is not conditioned on cost level. This is exact when
//! prediction class doesn't correlate with cost within the Rashomon bound.

use crate::types::{FSubRef, FactoredDag};
use rayon::prelude::*;

/// Determine which child bin a sample falls into for a k-way split.
///
/// For thresholds [t0, t1, ..., t_{k-2}]:
///   child 0:   val <= t0
///   child i:   t_{i-1} < val <= t_i
///   child k-1: val > t_{k-2}
///
/// Equivalent to `np.searchsorted(thresholds, val, side='right')`.
#[inline]
fn bin_index_right(thresholds: &[f64], val: f64) -> usize {
    thresholds.partition_point(|t| *t <= val)
}

/// Route a sample value to a child index on a split, honoring shape-mode
/// `group_map` when present. For contig splits (group_map=None) the bin
/// index is the child index directly.
#[inline]
pub(crate) fn child_index_for(split_spec: &crate::types::SplitSpec, val: f64) -> usize {
    let bin = bin_index_right(&split_spec.thresholds, val);
    match &split_spec.group_map {
        Some(gm) => gm[bin] as usize,
        None => bin,
    }
}

/// Compute P(class=1) at a subproblem for a single sample.
///
/// Memoized: `memo[sub_ref]` caches the result. NaN = not yet computed.
fn predict_prob_class1(dag: &FactoredDag, sub_ref: FSubRef, x: &[f64], memo: &mut [f64]) -> f64 {
    // Phase A — inline-leaf child: P(class=1) is determined entirely by
    // the inline leaf's prediction; no recursion needed and no memo entry
    // (the `memo` array indexes only the arena, which doesn't hold
    // inline leaves).
    if crate::types::is_leaf_ref(sub_ref) {
        let id = crate::types::leaf_id(sub_ref);
        let li = &dag.inline_leaves[id];
        let p_maj = if li.prediction == 1 { 1.0 } else { 0.0 };
        // Paired-leaf case: average the two prediction options
        // (each contributes 1 tree, so marginalization weight 1/2 each).
        if let Some(ml) = dag.inline_leaves_minority.get(id).and_then(|o| o.as_ref()) {
            let p_min = if ml.prediction == 1 { 1.0 } else { 0.0 };
            return 0.5 * p_maj + 0.5 * p_min;
        }
        return p_maj;
    }
    let idx = sub_ref as usize;
    if !memo[idx].is_nan() {
        return memo[idx];
    }

    let sub = &dag.subproblems[idx];

    let total = sub.tree_count as f64;
    if total == 0.0 {
        memo[idx] = 0.0;
        return 0.0;
    }

    let mut weighted_p1: f64 = 0.0;

    // Leaf contributes 1 tree
    if let Some((_, ref leaf)) = sub.leaf {
        if leaf.prediction == 1 {
            weighted_p1 += 1.0 / total;
        }
    }
    // The minority leaf option contributes 1 additional tree.
    if let Some((_, ref leaf)) = sub.leaf_minority.as_deref() {
        if leaf.prediction == 1 {
            weighted_p1 += 1.0 / total;
        }
    }

    // Each split routes this sample to exactly one child
    // Dispatch via the split pool when promoted, fall back to inline.
    let split_iter = crate::rashomon_dag::iter_splits(dag, sub_ref);
    for split in split_iter {
        let spec = dag.spec_of(split);
        let child_idx = child_index_for(spec, x[spec.feature_index as usize]);
        let child_ref = split.children[child_idx];
        let child_p1 = predict_prob_class1(dag, child_ref, x, memo);
        let split_weight = split.tree_count() as f64 / total;
        weighted_p1 += split_weight * child_p1;
    }

    memo[idx] = weighted_p1;
    weighted_p1
}

/// Compute P(class=1) for each sample in X, parallelized across samples.
///
/// # Arguments
/// * `dag` - The FactoredDag (Rashomon set)
/// * `x_flat` - Row-major flat array: x_flat[i * n_features + j] = feature j of sample i
/// * `n_samples` - Number of samples
/// * `n_features` - Number of features per sample
///
/// # Returns
/// Vec of length n_samples, each in [0.0, 1.0].
pub fn batch_predict_probs(
    dag: &FactoredDag,
    x_flat: &[f64],
    n_samples: usize,
    n_features: usize,
) -> Vec<f64> {
    let n_subproblems = dag.subproblems.len();

    (0..n_samples)
        .into_par_iter()
        .map(|i| {
            let x_row = &x_flat[i * n_features..(i + 1) * n_features];
            let mut memo = vec![f64::NAN; n_subproblems];
            predict_prob_class1(dag, dag.root_ref, x_row, &mut memo)
        })
        .collect()
}
