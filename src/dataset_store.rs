use std::collections::HashMap;
use std::sync::Arc;

use ndarray::Array2;

use crate::bitmask_ctx::{bin_index, thresholds_strictly_increasing, BitmaskContext};
use crate::types::{SupportMask, MAX_CLASSES};

pub struct DatasetStore {
    x: Arc<Array2<f64>>,
    labels: Arc<[u8]>,
    weights: Option<Arc<[f64]>>,
    feature_cols: Arc<[Arc<[f64]>]>,
    sorted_order: Arc<[Arc<[u32]>]>,
    /// `value_rank[j][i]` = position of row `i` in `sorted_order[j]`
    /// (support-sorted Select scan, `sorted_support_indices`).
    value_rank: Arc<[Arc<[u32]>]>,
    feature_thresholds: Arc<[Arc<[f64]>]>,
    feature_threshold_masks: Arc<[Arc<[Arc<[u64]>]>]>,
    label_masks: Arc<[Arc<[u64]>]>,
    // Equivalent-points (GOSDT/SPLIT-style) majority bitmask: bit `i` = 1 iff
    // sample `i` carries its equivalence class's most-common label, where the
    // equivalence class is "rows with identical bucketized-by-threshold feature
    // vectors." Used as a depth-free, structure-free misclass lower bound —
    // popcount(support & !majority_bitmask) is admissible for any tree.
    majority_bitmask: Arc<[u64]>,
    // Total minority count across all rows (independent of support). Stored
    // for potential future diagnostic use; not consulted on the hot path.
    _global_minority_count: u64,
    // Equivalence-class id per sample. Two samples share the same EC iff
    // they have the same bucketized feature vector. Used by
    // `eqp_misclass_per_support` to compute the per-support equivalent-points
    // misclass LB (a true LB at any depth, unlike the global-majority-based
    // `eqp_misclass` which is only admissible at the root).
    ec_id_per_sample: Arc<[u32]>,
    // Optional reference-model correctness bitmask (SPLIT's reference_LB).
    // When provided at dataset construction: bit `i` = 1 iff the supplied
    // reference model predicted sample `i` correctly (pred == label).
    // `reference_misclass(support)` returns popcount(support & !bitmask) /
    // n_total — the count of samples in the support where the reference
    // model was wrong.
    //
    // ADMISSIBILITY: this LB is admissible ONLY under the assumption that
    // the reference model's misclass on EVERY subset of the training data
    // is ≤ the optimum achievable by any tree the solver will search. The
    // harness does NOT verify this assumption. If violated, the solver may
    // report "proved optimum" trees that are not actually optimal. Users
    // opt in by passing `reference_preds`; default None leaves this None.
    reference_correct_bitmask: Option<Arc<[u64]>>,
    /// Select rank table: per feature, the threshold list (f64 bits) of
    /// the first `select_rank` call and its row -> bin table
    /// (`searchsorted_right` of every row's value).
    select_rank: Box<[std::sync::OnceLock<(Box<[u64]>, Box<[u16]>)>]>,
    ctx: Arc<BitmaskContext>,
    n_samples: usize,
    n_features: usize,
}

impl DatasetStore {
    pub fn new(
        x: Array2<f64>,
        labels: Vec<u8>,
        weights: Option<Vec<f64>>,
        n_classes: usize,
        thresholds_per_feature: Option<&HashMap<usize, Vec<f64>>>,
        reference_preds: Option<Vec<u8>>,
    ) -> Self {
        let n_samples = labels.len();
        if let Some(ref p) = reference_preds {
            assert_eq!(
                p.len(),
                n_samples,
                "reference_preds length ({}) must match labels length ({})",
                p.len(),
                n_samples
            );
        }
        let n_features = x.ncols();

        let mut feature_cols = Vec::with_capacity(n_features);
        let mut sorted_order = Vec::with_capacity(n_features);
        let mut value_rank = Vec::with_capacity(n_features);
        for j in 0..n_features {
            let col: Arc<[f64]> = Arc::from((0..n_samples).map(|i| x[[i, j]]).collect::<Vec<_>>());
            let mut order: Vec<u32> = (0..n_samples as u32).collect();
            order.sort_by(|&a, &b| {
                col[a as usize]
                    .partial_cmp(&col[b as usize])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            let mut rank = vec![0u32; n_samples];
            for (pos, &i) in order.iter().enumerate() {
                rank[i as usize] = pos as u32;
            }
            feature_cols.push(col);
            sorted_order.push(Arc::<[u32]>::from(order));
            value_rank.push(Arc::<[u32]>::from(rank));
        }

        let threshold_pairs: Vec<(usize, Vec<f64>)> = thresholds_per_feature
            .map(|m| m.iter().map(|(k, v)| (*k, v.clone())).collect())
            .unwrap_or_default();
        let ctx = if threshold_pairs.is_empty() {
            BitmaskContext::new_from_raw(&x, &labels, n_classes)
        } else {
            BitmaskContext::new(&x, &labels, n_classes, &threshold_pairs)
        };

        let feature_thresholds: Arc<[Arc<[f64]>]> = Arc::from(
            ctx.feature_thresholds
                .iter()
                .map(|v| Arc::<[f64]>::from(v.clone()))
                .collect::<Vec<_>>(),
        );
        let feature_threshold_masks: Arc<[Arc<[Arc<[u64]>]>]> = Arc::from(
            ctx.feature_le_masks
                .iter()
                .map(|feature_masks| {
                    Arc::<[Arc<[u64]>]>::from(
                        feature_masks
                            .iter()
                            .map(|mask| Arc::<[u64]>::from(mask.clone()))
                            .collect::<Vec<_>>(),
                    )
                })
                .collect::<Vec<_>>(),
        );
        let label_masks: Arc<[Arc<[u64]>]> = Arc::from(
            ctx.label_masks
                .iter()
                .map(|mask| Arc::<[u64]>::from(mask.clone()))
                .collect::<Vec<_>>(),
        );

        let (majority_bitmask, global_minority_count, ec_id_per_sample) =
            Self::construct_majority_bitmask(&ctx, &x, &labels, n_classes, n_samples);

        // Optional reference-model correctness bitmask. When provided, bit i
        // = 1 iff `reference_preds[i] == labels[i]`. Length matches
        // `ctx.words_per_mask` so it ANDs directly with any SupportMask.
        let reference_correct_bitmask: Option<Arc<[u64]>> = reference_preds.map(|preds| {
            let mut bm = vec![0u64; ctx.words_per_mask];
            for i in 0..n_samples {
                if preds[i] == labels[i] {
                    bm[i / 64] |= 1u64 << (i % 64);
                }
            }
            Arc::from(bm)
        });

        DatasetStore {
            x: Arc::new(x),
            labels: Arc::from(labels),
            weights: weights.map(Arc::from),
            feature_cols: Arc::from(feature_cols),
            sorted_order: Arc::from(sorted_order),
            value_rank: Arc::from(value_rank),
            feature_thresholds,
            feature_threshold_masks,
            label_masks,
            majority_bitmask,
            _global_minority_count: global_minority_count,
            ec_id_per_sample,
            reference_correct_bitmask,
            select_rank: (0..n_features)
                .map(|_| std::sync::OnceLock::new())
                .collect(),
            ctx: Arc::new(ctx),
            n_samples,
            n_features,
        }
    }

    /// Precomputes the equivalent-points majority bitmask. Bit `i` is set iff
    /// sample `i`'s label equals the most-common label among all samples
    /// sharing the same bucketized feature vector (as defined by the solver's
    /// fixed threshold set — same bits any tree can ever route on).
    ///
    /// Returns (bitmask, total minority count across all rows, per-sample EC
    /// id). The bitmask length matches `ctx.words_per_mask` so it can be ANDed
    /// directly with a SupportMask for any subproblem.
    ///
    /// The EC partition is computed on a fast path when every masked
    /// feature's threshold list is strictly increasing and NaN-free: rows are
    /// grouped by their per-feature bin-index vector (`ec_ids_from_bins`),
    /// which is a bijective image of the le-bit fingerprint. Otherwise the
    /// fingerprint scatter is used (`ec_ids_from_fingerprints`). Both assign
    /// EC ids in row order of first appearance, and the majority / minority /
    /// bitmask tail is shared, so the returned triple is identical either way.
    fn construct_majority_bitmask(
        ctx: &BitmaskContext,
        x: &Array2<f64>,
        labels: &[u8],
        n_classes: usize,
        n_samples: usize,
    ) -> (Arc<[u64]>, u64, Arc<[u32]>) {
        // Per-sample fingerprint: concatenate `feature_le_masks[j][t][i]` bits
        // across all (feature j, threshold t) pairs. Samples with identical
        // bit patterns are indistinguishable to any tree the solver can produce.
        let total_bits: usize = ctx.feature_le_masks.iter().map(|m| m.len()).sum();

        // Short-circuit: if there are no thresholds (nothing to split on), every
        // sample is its own equivalence class with no discriminating info — the
        // majority bitmask then just marks rows whose label matches the GLOBAL
        // majority. But in practice the solver has thresholds so this is rare.
        if total_bits == 0 {
            let mut counts = [0usize; MAX_CLASSES];
            for &l in labels {
                counts[l as usize] += 1;
            }
            let global_majority = (0..n_classes).max_by_key(|&c| counts[c]).unwrap_or(0) as u8;
            let mut bitmask = vec![0u64; ctx.words_per_mask];
            let mut minority = 0u64;
            for (i, &l) in labels.iter().enumerate() {
                if l == global_majority {
                    bitmask[i / 64] |= 1u64 << (i % 64);
                } else {
                    minority += 1;
                }
            }
            // No splitting features → every sample is its own EC for
            // per-support eqp purposes (no within-EC class confusion to
            // bound).
            let ec_ids: Vec<u32> = (0..n_samples as u32).collect();
            return (Arc::from(bitmask), minority, Arc::from(ec_ids));
        }

        let masked: Vec<usize> = (0..ctx.feature_le_masks.len())
            .filter(|&j| !ctx.feature_le_masks[j].is_empty())
            .collect();
        let fast_ok = masked.iter().all(|&j| {
            let thr = &ctx.feature_thresholds[j];
            thr.len() == ctx.feature_le_masks[j].len() && thresholds_strictly_increasing(thr)
        });
        let ec_id_per_sample: Vec<u32> = if fast_ok {
            Self::ec_ids_from_bins(ctx, x, &masked, n_samples)
        } else {
            Self::ec_ids_from_fingerprints(ctx, n_samples)
        };
        Self::majority_from_ec_ids(ctx, labels, n_classes, n_samples, ec_id_per_sample)
    }

    /// Reference EC partition: per-sample fingerprints concatenating
    /// `feature_le_masks[j][t][i]` bits across all (feature j, threshold t)
    /// pairs, grouped by a HashMap. EC id = order of first appearance in row
    /// order. Fallback when some threshold list is not strictly increasing
    /// (or has NaN), and the oracle for `ec_ids_from_bins` in the unit tests.
    pub(crate) fn ec_ids_from_fingerprints(ctx: &BitmaskContext, n_samples: usize) -> Vec<u32> {
        let total_bits: usize = ctx.feature_le_masks.iter().map(|m| m.len()).sum();
        let words_per_fp = (total_bits + 63) / 64;
        let mut fingerprints: Vec<Vec<u64>> = vec![vec![0u64; words_per_fp]; n_samples];
        let mut bit_idx = 0usize;
        for feature_masks in &ctx.feature_le_masks {
            for word_vec in feature_masks {
                // word_vec is the bitmask for this (feature, threshold) over all
                // samples. bit `i` set means sample `i` satisfies feat ≤ thresh.
                let dst_word = bit_idx / 64;
                let dst_shift = bit_idx % 64;
                for i in 0..n_samples {
                    if (word_vec[i / 64] >> (i % 64)) & 1 != 0 {
                        fingerprints[i][dst_word] |= 1u64 << dst_shift;
                    }
                }
                bit_idx += 1;
            }
        }

        // Group by fingerprint; assign each sample an EC id for the
        // per-support eqp routine.
        let mut groups: HashMap<Vec<u64>, u32> = HashMap::new();
        let mut next_ec_id: u32 = 0;
        let mut ec_id_per_sample = vec![0u32; n_samples];
        for (i, fingerprint) in fingerprints.into_iter().enumerate() {
            let id = *groups.entry(fingerprint).or_insert_with(|| {
                let id = next_ec_id;
                next_ec_id += 1;
                id
            });
            ec_id_per_sample[i] = id;
        }
        ec_id_per_sample
    }

    /// Fast EC partition (strictly increasing thresholds on every masked
    /// feature, `masked` in ascending feature order).
    fn ec_ids_from_bins(
        ctx: &BitmaskContext,
        x: &Array2<f64>,
        masked: &[usize],
        n_samples: usize,
    ) -> Vec<u32> {
        // Row i's fingerprint is the concatenation over masked features j of
        // its T_j-bit le-pattern; by the bijection in `bin_index` that pattern
        // is 0^b 1^(T_j-b) with b = bin_index(thr_j, x_ij), so the fingerprint
        // is a bijective image of the bin-index vector (b_j(i))_j. Grouping on
        // the bin-index vector therefore yields exactly the same partition as
        // grouping on the fingerprint. EC ids are assigned in row order of
        // first appearance, exactly as in `ec_ids_from_fingerprints`.
        let mf = masked.len();
        let mut keys = vec![0u32; n_samples * mf];
        for (kj, &j) in masked.iter().enumerate() {
            let thr = &ctx.feature_thresholds[j];
            for i in 0..n_samples {
                keys[i * mf + kj] = bin_index(thr, x[[i, j]]) as u32;
            }
        }
        let mut groups: HashMap<&[u32], u32> = HashMap::with_capacity(n_samples);
        let mut next_ec_id = 0u32;
        let mut ec = vec![0u32; n_samples];
        for i in 0..n_samples {
            let id = *groups
                .entry(&keys[i * mf..(i + 1) * mf])
                .or_insert_with(|| {
                    let id = next_ec_id;
                    next_ec_id += 1;
                    id
                });
            ec[i] = id;
        }
        ec
    }

    /// Shared tail of `construct_majority_bitmask`: per-EC majority class,
    /// majority bitmask and minority total from the EC ids. Bit i depends
    /// only on (label_i, majority of i's EC) and the minority total is a sum
    /// of per-row indicators, so the result does not depend on EC iteration
    /// order.
    pub(crate) fn majority_from_ec_ids(
        ctx: &BitmaskContext,
        labels: &[u8],
        n_classes: usize,
        n_samples: usize,
        ec_id_per_sample: Vec<u32>,
    ) -> (Arc<[u64]>, u64, Arc<[u32]>) {
        let n_ec = ec_id_per_sample
            .iter()
            .copied()
            .max()
            .map_or(0, |m| m as usize + 1);
        let mut counts = vec![[0usize; MAX_CLASSES]; n_ec];
        for i in 0..n_samples {
            counts[ec_id_per_sample[i] as usize][labels[i] as usize] += 1;
        }
        let mut majority = vec![0u8; n_ec];
        for e in 0..n_ec {
            // Tie-break: smallest class index among the tied maxima wins
            // (strict `>` from class 0 upward) — same rule as before.
            let (mut best, mut mc) = (0usize, 0u8);
            for c in 0..n_classes {
                if counts[e][c] > best {
                    best = counts[e][c];
                    mc = c as u8;
                }
            }
            majority[e] = mc;
        }
        let mut bitmask = vec![0u64; ctx.words_per_mask];
        let mut minority_total = 0u64;
        for i in 0..n_samples {
            if labels[i] == majority[ec_id_per_sample[i] as usize] {
                bitmask[i / 64] |= 1u64 << (i % 64);
            } else {
                minority_total += 1;
            }
        }
        (
            Arc::from(bitmask),
            minority_total,
            Arc::from(ec_id_per_sample),
        )
    }

    /// Per-support equivalent-points misclass LB. For each EC, counts
    /// samples in `support` per class, sums `min(c0, c1)`, and divides by
    /// `n_total`. This is a TRUE LB on any tree's misclass on the support
    /// at any depth (since equivalent-feature-tuple samples must receive
    /// the same prediction, the minimum mistakes per EC is the within-EC
    /// minority count).
    ///
    /// Distinct from `eqp_misclass`, which uses the global majority
    /// bitmask and is admissible only at the root. Binary-only.
    /// O(n_support) per call; caller should cache by support fingerprint
    /// when invoked in a hot loop.
    pub fn eqp_misclass_per_support(&self, support: &SupportMask, n_total: usize) -> f64 {
        if support.n_samples == 0 || n_total == 0 {
            return 0.0;
        }
        self.eqp_misclass_per_support_words(support.words.as_ref(), n_total)
    }

    /// Word-slice variant of `eqp_misclass_per_support`. Caller
    /// must ensure the words slice is non-empty when n_samples > 0.
    pub fn eqp_misclass_per_support_words(&self, words: &[u64], n_total: usize) -> f64 {
        if n_total == 0 {
            return 0.0;
        }
        use rustc_hash::FxHashMap;
        let mut per_ec: FxHashMap<u32, [u32; MAX_CLASSES]> = FxHashMap::default();
        let labels = self.labels.as_ref();
        let ec_ids = self.ec_id_per_sample.as_ref();
        let mut visit = |idx: u32| {
            let i = idx as usize;
            let label = labels[i] as usize;
            let entry = per_ec.entry(ec_ids[i]).or_insert([0u32; MAX_CLASSES]);
            entry[label] = entry[label].saturating_add(1);
        };
        self.for_each_active_index_words(words, &mut visit);
        let total_unavoidable: u32 = per_ec
            .values()
            .map(|c| {
                let n: u32 = c.iter().sum();
                let max: u32 = *c.iter().max().unwrap_or(&0);
                n - max
            })
            .sum();
        total_unavoidable as f64 / n_total as f64
    }

    /// Indices-based variant of `eqp_misclass_per_support` for the inline
    /// bounds path (compute_bounds_inline) where no SupportMask is available.
    /// Same admissibility guarantees: per-equivalence-class minority count is
    /// a true LB on any tree's misclass at any depth. Binary-only; >2 classes
    /// returns 0.0 as a safe LB. O(|indices|) time + small FxHashMap alloc.
    pub fn eqp_misclass_per_support_indices(&self, indices: &[u32], n_total: usize) -> f64 {
        if indices.is_empty() || n_total == 0 {
            return 0.0;
        }
        use rustc_hash::FxHashMap;
        let mut per_ec: FxHashMap<u32, [u32; MAX_CLASSES]> = FxHashMap::default();
        let labels = self.labels.as_ref();
        let ec_ids = self.ec_id_per_sample.as_ref();
        for &idx in indices {
            let i = idx as usize;
            let label = labels[i] as usize;
            let entry = per_ec.entry(ec_ids[i]).or_insert([0u32; MAX_CLASSES]);
            entry[label] = entry[label].saturating_add(1);
        }
        let total_unavoidable: u32 = per_ec
            .values()
            .map(|c| {
                let n: u32 = c.iter().sum();
                let max: u32 = *c.iter().max().unwrap_or(&0);
                n - max
            })
            .sum();
        total_unavoidable as f64 / n_total as f64
    }

    /// Equivalent-points misclass lower bound for a subproblem support.
    /// Computes `popcount(support & !majority_bitmask) / n_total` — the
    /// minority-class sample count within the subproblem, normalized to
    /// global objective units. This is admissible as a LB on any tree's
    /// total misclass contribution from this subproblem at any depth.
    ///
    /// O(n_words) bitmask operations. When `weights` are present, falls back
    /// to the iterating path (slower) to apply per-sample weights.
    pub fn eqp_misclass(&self, support: &SupportMask, n_total: usize) -> f64 {
        if let Some(weights) = self.weights() {
            let scale = self.n_samples as f64;
            let mut weighted_minority: f64 = 0.0;
            self.for_each_active_index(support, |idx| {
                let i = idx as usize;
                let is_majority = (self.majority_bitmask[i / 64] >> (i % 64)) & 1 != 0;
                if !is_majority {
                    weighted_minority += weights[i] * scale;
                }
            });
            return weighted_minority.round() / n_total as f64;
        }
        let mut minority: u64 = 0;
        for (s, m) in support.words.iter().zip(self.majority_bitmask.iter()) {
            minority += (s & !m).count_ones() as u64;
        }
        minority as f64 / n_total as f64
    }

    /// Reference-model lower bound on misclass for a subproblem support.
    /// Returns `popcount(support & !reference_correct_bitmask) / n_total` —
    /// the count of samples in `support` where the supplied reference model
    /// was wrong. Returns 0.0 when no reference model was provided at
    /// dataset construction (feature disabled).
    ///
    /// **NOT strictly admissible.** Only admissible under the assumption
    /// that the reference model's misclass on every possible subset is ≤
    /// the optimum achievable by any tree the solver will search. If
    /// violated, the solver may prune subtrees containing the true
    /// optimum and report "proved optimum" trees that are not actually
    /// optimal. See SPLIT's `reference_LB` for the matching semantics.
    pub fn reference_misclass(&self, support: &SupportMask, n_total: usize) -> f64 {
        let ref_bm = match &self.reference_correct_bitmask {
            Some(bm) => bm,
            None => return 0.0,
        };
        if let Some(weights) = self.weights() {
            let scale = self.n_samples as f64;
            let mut weighted_wrong: f64 = 0.0;
            self.for_each_active_index(support, |idx| {
                let i = idx as usize;
                let is_correct = (ref_bm[i / 64] >> (i % 64)) & 1 != 0;
                if !is_correct {
                    weighted_wrong += weights[i] * scale;
                }
            });
            return weighted_wrong.round() / n_total as f64;
        }
        let mut wrong: u64 = 0;
        for (s, r) in support.words.iter().zip(ref_bm.iter()) {
            wrong += (s & !r).count_ones() as u64;
        }
        wrong as f64 / n_total as f64
    }

    pub fn row_major(&self) -> &Array2<f64> {
        self.x.as_ref()
    }

    pub fn labels(&self) -> &[u8] {
        self.labels.as_ref()
    }

    pub fn weights(&self) -> Option<&[f64]> {
        self.weights.as_deref()
    }

    pub fn feature_cols(&self) -> &[Arc<[f64]>] {
        self.feature_cols.as_ref()
    }

    pub fn sorted_order(&self) -> &[Arc<[u32]>] {
        self.sorted_order.as_ref()
    }

    pub fn feature_col(&self, feature_index: usize) -> &[f64] {
        self.feature_cols[feature_index].as_ref()
    }

    pub fn sorted_feature_order(&self, feature_index: usize) -> &[u32] {
        self.sorted_order[feature_index].as_ref()
    }

    /// The rows `indices` in `sorted_feature_order(j)` order: `out` = the
    /// subsequence of `sorted_order[j]` restricted to `indices` (ranks are
    /// unique, so ties keep the stable sort's ascending-row order). O(n log
    /// n) in the support size instead of a full-N scan.
    pub fn sorted_support_indices(&self, j: usize, indices: &[u32], out: &mut Vec<u32>) {
        let rank = self.value_rank[j].as_ref();
        out.clear();
        out.extend_from_slice(indices);
        out.sort_unstable_by_key(|&i| rank[i as usize]);
    }

    pub fn ctx(&self) -> &BitmaskContext {
        self.ctx.as_ref()
    }

    pub fn n_samples(&self) -> usize {
        self.n_samples
    }

    /// Words per mask (n_samples.div_ceil(64)).
    pub fn words_per_mask(&self) -> usize {
        self.ctx.words_per_mask
    }

    pub fn n_features(&self) -> usize {
        self.n_features
    }

    pub fn feature_thresholds(&self) -> &[Arc<[f64]>] {
        self.feature_thresholds.as_ref()
    }

    pub fn feature_threshold_masks(&self) -> &[Arc<[Arc<[u64]>]>] {
        self.feature_threshold_masks.as_ref()
    }

    pub fn label_masks(&self) -> &[Arc<[u64]>] {
        self.label_masks.as_ref()
    }

    pub fn root_support(&self) -> SupportMask {
        SupportMask::from_words(self.ctx.root_mask(), self.n_samples)
    }

    pub fn support_from_indices(&self, indices: &[u32]) -> SupportMask {
        let mut words = vec![0u64; self.ctx.words_per_mask];
        for &idx in indices {
            let i = idx as usize;
            words[i / 64] |= 1u64 << (i % 64);
        }
        SupportMask::from_words(words, indices.len())
    }

    pub fn support_to_indices(&self, support: &SupportMask) -> Vec<u32> {
        BitmaskContext::bitmask_to_indices(support.words.as_ref(), self.n_samples)
    }

    pub fn fill_active_indices(&self, support: &SupportMask, out: &mut Vec<u32>) {
        out.clear();
        for (word_idx, &word) in support.words.iter().enumerate() {
            let mut w = word;
            while w != 0 {
                let bit = w.trailing_zeros() as usize;
                let idx = word_idx * 64 + bit;
                if idx < self.n_samples {
                    out.push(idx as u32);
                }
                w &= w - 1;
            }
        }
    }

    pub fn for_each_active_index(&self, support: &SupportMask, mut f: impl FnMut(u32)) {
        self.for_each_active_index_words(support.words.as_ref(), &mut f)
    }

    /// Word-slice variant for SupportView paths.
    #[inline]
    pub fn for_each_active_index_words(&self, words: &[u64], f: &mut impl FnMut(u32)) {
        for (word_idx, &word) in words.iter().enumerate() {
            let mut w = word;
            while w != 0 {
                let bit = w.trailing_zeros() as usize;
                let idx = word_idx * 64 + bit;
                if idx < self.n_samples {
                    f(idx as u32);
                }
                w &= w - 1;
            }
        }
    }

    #[inline]
    pub fn support_contains(&self, support: &SupportMask, idx: u32) -> bool {
        let idx = idx as usize;
        let word = idx / 64;
        let bit = idx % 64;
        support
            .words
            .get(word)
            .map(|w| (w & (1u64 << bit)) != 0)
            .unwrap_or(false)
    }

    pub fn count_classes(&self, support: &SupportMask) -> ([i64; MAX_CLASSES], usize) {
        self.ctx.count_classes(support.words.as_ref())
    }

    pub fn partition_support(
        &self,
        support: &SupportMask,
        feature_index: usize,
        thresholds: &[f64],
        group_map: Option<&[usize]>,
    ) -> Vec<SupportMask> {
        let base_partition =
            if let Some(thresh_base) = self.ctx.find_threshold_base(feature_index, thresholds) {
                self.ctx.partition_bitmask(
                    support.words.as_ref(),
                    feature_index,
                    thresh_base,
                    thresholds.len() + 1,
                )
            } else {
                self.ctx.partition_bitmask_raw(
                    support.words.as_ref(),
                    self.row_major(),
                    feature_index,
                    thresholds,
                )
            };

        if let Some(group_map) = group_map {
            let n_groups = group_map.iter().copied().max().unwrap_or(0) + 1;
            let mut grouped = vec![vec![0u64; self.ctx.words_per_mask]; n_groups];
            for (bin_idx, (mask, _, _)) in base_partition.into_iter().enumerate() {
                let group = group_map[bin_idx];
                for (dst, src) in grouped[group].iter_mut().zip(mask.iter()) {
                    *dst |= *src;
                }
            }
            grouped
                .into_iter()
                .map(|mask| {
                    let (_, n_samples) = self.ctx.count_classes(&mask);
                    SupportMask::from_words(mask, n_samples)
                })
                .collect()
        } else {
            base_partition
                .into_iter()
                .map(|(mask, _, n_samples)| SupportMask::from_words(mask, n_samples))
                .collect()
        }
    }

    /// Select rank table: `rank[i] = searchsorted_right(thresholds,
    /// x_ij)` for every row i of feature j (Select's binary custom-threshold
    /// bin of row i). Built on the first call per feature from that call's
    /// `thresholds`; `None` when a later call passes a different list
    /// (bitwise) or the list has >= u16::MAX entries (the caller then bins
    /// row by row). The same function on the same list: the same bins.
    pub fn select_rank(&self, j: usize, thresholds: &[f64]) -> Option<&[u16]> {
        if thresholds.len() >= u16::MAX as usize {
            return None;
        }
        let cell = self.select_rank.get(j)?;
        let (bits, rank) = cell.get_or_init(|| {
            let col = self.feature_col(j);
            (
                thresholds.iter().map(|t| t.to_bits()).collect(),
                col.iter()
                    .map(|&x| crate::candidates::searchsorted_right(thresholds, x) as u16)
                    .collect(),
            )
        });
        (bits.len() == thresholds.len()
            && bits.iter().zip(thresholds).all(|(b, t)| *b == t.to_bits()))
        .then_some(&rank[..])
    }

    /// The le-masks (`x <= t`) of `thresholds` among feature f's store
    /// thresholds; `None` unless the thresholds are strictly ascending and
    /// each is bitwise one of the store thresholds.
    pub fn le_masks_for(
        &self,
        feature_index: usize,
        thresholds: &[f64],
    ) -> Option<smallvec::SmallVec<[&[u64]; 8]>> {
        if thresholds.is_empty() || !thresholds.windows(2).all(|w| w[0] < w[1]) {
            return None;
        }
        let store_ths = self.feature_thresholds.get(feature_index)?;
        let masks = self.feature_threshold_masks.get(feature_index)?;
        let mut le: smallvec::SmallVec<[&[u64]; 8]> = smallvec::SmallVec::new();
        for t in thresholds {
            let i = store_ths.iter().position(|s| s.to_bits() == t.to_bits())?;
            le.push(masks.get(i)?.as_ref());
        }
        Some(le)
    }

    /// Fast child masks: bin `b` of the le-mask partition of `support`
    /// by the nested le-masks `le` (`le_masks_for`): S & le[0], S & le[b] &
    /// !le[b-1], S & !le[last]. Written straight into the `Arc<[u64]>` with
    /// the popcount fused; the same words, count and fingerprint as
    /// `partition_support_le`'s bin b.
    pub fn le_bin_support(support: &SupportMask, le: &[&[u64]], b: usize) -> SupportMask {
        let w = support.words.as_ref();
        let n_bins = le.len() + 1;
        let hi: Option<&[u64]> = (b < n_bins - 1).then(|| le[b]);
        let lo: Option<&[u64]> = (b > 0).then(|| le[b - 1]);
        let mut n = 0usize;
        let words: Arc<[u64]> = match (lo, hi) {
            (None, Some(h)) => (0..w.len())
                .map(|i| {
                    let v = w[i] & h[i];
                    n += v.count_ones() as usize;
                    v
                })
                .collect(),
            (Some(l), Some(h)) => (0..w.len())
                .map(|i| {
                    let v = w[i] & h[i] & !l[i];
                    n += v.count_ones() as usize;
                    v
                })
                .collect(),
            (Some(l), None) => (0..w.len())
                .map(|i| {
                    let v = w[i] & !l[i];
                    n += v.count_ones() as usize;
                    v
                })
                .collect(),
            (None, None) => {
                n = w.iter().map(|x| x.count_ones() as usize).sum();
                Arc::clone(&support.words)
            }
        };
        SupportMask::from_arc(words, n)
    }

    /// Child supports of a split by le-mask AND / ANDNOT (`x <= t` left):
    /// bin 0 = S & le(t_0), bin b = S & le(t_b) & !le(t_{b-1}), last =
    /// S & !le(t_last); group_map ORs bins into groups as
    /// `partition_support`. `None` unless the thresholds are strictly
    /// ascending and each is bitwise one of the feature's store thresholds
    /// (whose le-masks are then nested). Equals the row partition of
    /// `x < t` left whenever no row value equals a threshold (the caller's
    /// precondition, e.g. the completion-lb gate's mask_exact).
    pub fn partition_support_le(
        &self,
        support: &SupportMask,
        feature_index: usize,
        thresholds: &[f64],
        group_map: Option<&[usize]>,
    ) -> Option<Vec<SupportMask>> {
        if thresholds.is_empty() || !thresholds.windows(2).all(|w| w[0] < w[1]) {
            return None;
        }
        let store_ths = self.feature_thresholds.get(feature_index)?;
        let masks = self.feature_threshold_masks.get(feature_index)?;
        let mut le: smallvec::SmallVec<[&[u64]; 8]> = smallvec::SmallVec::new();
        for t in thresholds {
            let i = store_ths.iter().position(|s| s.to_bits() == t.to_bits())?;
            le.push(masks.get(i)?.as_ref());
        }
        let n_bins = thresholds.len() + 1;
        let gm = match group_map {
            None => {
                return Some(
                    (0..n_bins)
                        .map(|b| Self::le_bin_support(support, &le, b))
                        .collect(),
                )
            }
            Some(gm) => gm,
        };
        let w = support.words.as_ref();
        let nw = w.len();
        let bin_words = |b: usize, i: usize| -> u64 {
            let mut v = w[i];
            if b < n_bins - 1 {
                v &= le[b][i];
            }
            if b > 0 {
                v &= !le[b - 1][i];
            }
            v
        };
        let to_mask = |words: Vec<u64>| {
            let n = words.iter().map(|x| x.count_ones() as usize).sum();
            SupportMask::from_words(words, n)
        };
        if gm.len() != n_bins {
            return None;
        }
        let n_groups = gm.iter().copied().max().unwrap_or(0) + 1;
        let mut grouped = vec![vec![0u64; nw]; n_groups];
        for b in 0..n_bins {
            let dst = &mut grouped[gm[b]];
            for i in 0..nw {
                dst[i] |= bin_words(b, i);
            }
        }
        Some(grouped.into_iter().map(to_mask).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

    /// Slow-path oracle for the store's (majority_bitmask, minority, ec ids).
    fn slow_triple(
        ctx: &BitmaskContext,
        labels: &[u8],
        n_classes: usize,
    ) -> (Arc<[u64]>, u64, Arc<[u32]>) {
        let n = labels.len();
        let total_bits: usize = ctx.feature_le_masks.iter().map(|m| m.len()).sum();
        assert!(total_bits > 0);
        let ec = DatasetStore::ec_ids_from_fingerprints(ctx, n);
        DatasetStore::majority_from_ec_ids(ctx, labels, n_classes, n, ec)
    }

    fn check_case(x: &Array2<f64>, y: &[u8], n_classes: usize, ths: &HashMap<usize, Vec<f64>>) {
        let n = y.len();
        let wpm = (n + 63) / 64;
        let pairs: Vec<(usize, Vec<f64>)> = ths.iter().map(|(k, v)| (*k, v.clone())).collect();
        let ctx = BitmaskContext::new(x, y, n_classes, &pairs);
        for j in 0..x.ncols() {
            let want = match ths.get(&j) {
                Some(t) if !t.is_empty() => BitmaskContext::le_masks_slow(x, j, t, n, wpm),
                _ => Vec::new(),
            };
            assert_eq!(ctx.feature_le_masks[j], want, "feature {j} masks");
        }
        let total_bits: usize = ctx.feature_le_masks.iter().map(|m| m.len()).sum();
        // NaN in `x` makes the `sort_by(partial_cmp ... Equal)` comparator in
        // `DatasetStore::new` non-total, which the std sort may detect and
        // panic on (pre-existing, outside this test's scope). So the NaN data
        // goes straight into `construct_majority_bitmask` (the dispatcher
        // under test), and only NaN-free data goes through `DatasetStore::new`.
        let got = DatasetStore::construct_majority_bitmask(&ctx, x, y, n_classes, n);
        if total_bits == 0 {
            // Every feature unmasked: short-circuit path, ec ids = 0..n.
            let ids: Vec<u32> = (0..n as u32).collect();
            assert_eq!(&got.2[..], &ids[..]);
        } else {
            assert_eq!(got, slow_triple(&ctx, y, n_classes));
        }
        if x.iter().all(|v| !v.is_nan()) {
            let store = DatasetStore::new(x.clone(), y.to_vec(), None, n_classes, Some(ths), None);
            let from_store = (
                store.majority_bitmask.clone(),
                store._global_minority_count,
                store.ec_id_per_sample.clone(),
            );
            assert_eq!(from_store, got);
        }
    }

    /// `x` with NaN replaced by `fill` (for the `DatasetStore::new` check).
    fn without_nan(x: &Array2<f64>, fill: f64) -> Array2<f64> {
        x.mapv(|v| if v.is_nan() { fill } else { v })
    }

    #[test]
    fn bin_index_paths_match_scalar_paths() {
        let mut rng = Lcg(0x5eed_1234);
        let ns = [1usize, 7, 64, 65, 130, 300];
        for case in 0..12usize {
            let n = ns[case % ns.len()];
            let n_feat = 1 + (case % 5);
            let n_classes = 2 + (case % 2);
            let vmax = 3 + rng.below(4); // values in 0..vmax: dense ties
            let mut x = Array2::<f64>::zeros((n, n_feat));
            for i in 0..n {
                for j in 0..n_feat {
                    x[[i, j]] = if rng.below(20) == 0 {
                        f64::NAN
                    } else {
                        rng.below(vmax) as f64
                    };
                }
            }
            let y: Vec<u8> = (0..n).map(|_| rng.below(n_classes as u64) as u8).collect();
            let mut ths: HashMap<usize, Vec<f64>> = HashMap::new();
            let all_unmasked = case == 11;
            for j in 0..n_feat {
                if all_unmasked {
                    ths.insert(j, Vec::new());
                    continue;
                }
                if j > 0 && rng.below(4) == 0 {
                    continue; // feature with no thresholds
                }
                // Random non-empty subset of midpoints plus some thresholds
                // equal to row values (tests the `<=` boundary).
                let mut t: Vec<f64> = Vec::new();
                for v in 0..vmax {
                    if rng.below(2) == 0 {
                        t.push(v as f64 + 0.5);
                    }
                    if rng.below(3) == 0 {
                        t.push(v as f64);
                    }
                }
                if t.is_empty() {
                    t.push(0.5);
                }
                t.sort_by(|a, b| a.partial_cmp(b).unwrap());
                assert!(crate::bitmask_ctx::thresholds_strictly_increasing(&t));
                ths.insert(j, t);
            }
            check_case(&x, &y, n_classes, &ths);
            check_case(&without_nan(&x, vmax as f64), &y, n_classes, &ths);
        }

        // Fallback: thresholds not strictly increasing or with NaN take the
        // slow path; output must still equal the slow oracle.
        let fallbacks: [Vec<f64>; 3] = [
            vec![0.5, 0.2, 0.9],
            vec![0.2, 0.2],
            vec![0.1, f64::NAN, 0.7],
        ];
        for (c, bad) in fallbacks.iter().enumerate() {
            assert!(!crate::bitmask_ctx::thresholds_strictly_increasing(bad));
            let n = 70 + c;
            let mut x = Array2::<f64>::zeros((n, 2));
            for i in 0..n {
                x[[i, 0]] = if rng.below(15) == 0 {
                    f64::NAN
                } else {
                    rng.below(10) as f64 / 10.0
                };
                x[[i, 1]] = rng.below(4) as f64;
            }
            let y: Vec<u8> = (0..n).map(|_| rng.below(2) as u8).collect();
            let mut ths: HashMap<usize, Vec<f64>> = HashMap::new();
            ths.insert(0, bad.clone());
            ths.insert(1, vec![0.5, 1.0, 2.5]);
            check_case(&x, &y, 2, &ths);
            check_case(&without_nan(&x, 0.3), &y, 2, &ths);
        }
    }
}
