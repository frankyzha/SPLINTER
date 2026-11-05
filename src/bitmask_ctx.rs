use ndarray::Array2;

use crate::types::MAX_CLASSES;

/// True iff `thr` has no NaN and is strictly increasing. This is the
/// precondition for the bin-index fast paths (`le_masks_fast`,
/// `DatasetStore::construct_majority_bitmask` fast path): under it
/// `partition_point(|t| t < v)` is well defined (the predicate is
/// monotone) and the le-bit pattern of a value is a bijective function
/// of its bin index (see `bin_index`).
pub(crate) fn thresholds_strictly_increasing(thr: &[f64]) -> bool {
    thr.iter().all(|t| !t.is_nan()) && thr.windows(2).all(|w| w[0] < w[1])
}

/// Bin index of value `v` against strictly increasing `thr` (len T):
/// b = #{t : t < v} = partition_point(|t| t < v), with NaN -> T.
///
/// Bijection argument. For every k in 0..T:
///     v <= thr[k]  <=>  !(thr[k] < v)  <=>  b <= k,
/// because the thresholds strictly increase, so {t : t < v} is exactly
/// the prefix thr[..b]. Hence the T-bit le-pattern of v is 0^b 1^(T-b):
/// bit k is set iff b <= k. The map b -> pattern is injective on 0..=T
/// (T+1 distinct patterns), so two rows have the same le-pattern on a
/// feature iff they have the same bin index. For NaN, `v <= thr[k]` is
/// false for all k (all-zero pattern) which is exactly the b = T pattern.
#[inline]
pub(crate) fn bin_index(thr: &[f64], v: f64) -> usize {
    if v.is_nan() {
        thr.len()
    } else {
        thr.partition_point(|&t| t < v)
    }
}

/// Global bitmask context: pre-binarized feature columns and label masks.
///
/// Allocated once at solver initialization. All bitmasks use the same
/// `words_per_mask` layout so bitwise operations are uniform.
pub struct BitmaskContext {
    pub n_samples: usize,
    pub n_classes: usize,
    pub words_per_mask: usize,

    /// Pre-binarized feature columns for compressed thresholds.
    /// `feature_le_masks[feat][thresh_idx]` = bitmask where bit i = 1 iff
    /// X[i, feat] <= feature_thresholds[feat][thresh_idx].
    pub feature_le_masks: Vec<Vec<Vec<u64>>>,

    /// Sorted compressed thresholds per feature (matching feature_le_masks).
    pub feature_thresholds: Vec<Vec<f64>>,

    /// Label bitmasks: `label_masks[class][word]` has bit i = 1 iff y[i] == class.
    pub label_masks: Vec<Vec<u64>>,
}

impl BitmaskContext {
    /// Build the context from data and compressed thresholds.
    ///
    /// `thresholds_per_feature`: for each feature index, the sorted compressed
    /// thresholds to pre-binarize. Features not in this map get empty masks
    /// (lazy binarization will be used for them).
    pub fn new(
        x: &Array2<f64>,
        y: &[u8],
        n_classes: usize,
        thresholds_per_feature: &[(usize, Vec<f64>)],
    ) -> Self {
        let n_samples = y.len();
        let n_features = x.ncols();
        let words_per_mask = (n_samples + 63) / 64;

        // Build label masks
        let mut label_masks = vec![vec![0u64; words_per_mask]; n_classes];
        for (i, &label) in y.iter().enumerate() {
            let word = i / 64;
            let bit = i % 64;
            label_masks[label as usize][word] |= 1u64 << bit;
        }

        // Build feature <= threshold masks (compressed thresholds only)
        let mut feature_le_masks = vec![Vec::new(); n_features];
        let mut feature_thresholds = vec![Vec::new(); n_features];

        for &(feat_idx, ref thresholds) in thresholds_per_feature {
            if feat_idx >= n_features || thresholds.is_empty() {
                continue;
            }
            let masks = if thresholds_strictly_increasing(thresholds) {
                Self::le_masks_fast(x, feat_idx, thresholds, n_samples, words_per_mask)
            } else {
                Self::le_masks_slow(x, feat_idx, thresholds, n_samples, words_per_mask)
            };
            feature_le_masks[feat_idx] = masks;
            feature_thresholds[feat_idx] = thresholds.clone();
        }

        BitmaskContext {
            n_samples,
            n_classes,
            words_per_mask,
            feature_le_masks,
            feature_thresholds,
            label_masks,
        }
    }

    /// Scalar reference construction of the `<=` masks of one feature:
    /// bit i of mask t set iff `x[i, feat_idx] <= thresholds[t]`, O(n * T).
    /// Fallback for threshold lists that are not strictly increasing or
    /// contain NaN, and the oracle for the fast path in the unit tests.
    pub(crate) fn le_masks_slow(
        x: &Array2<f64>,
        feat_idx: usize,
        thresholds: &[f64],
        n_samples: usize,
        words_per_mask: usize,
    ) -> Vec<Vec<u64>> {
        let mut masks = Vec::with_capacity(thresholds.len());
        for &thresh in thresholds {
            let mut mask = vec![0u64; words_per_mask];
            for i in 0..n_samples {
                if x[[i, feat_idx]] <= thresh {
                    mask[i / 64] |= 1u64 << (i % 64);
                }
            }
            masks.push(mask);
        }
        masks
    }

    /// Bin-index construction of the `<=` masks of one feature, O(n + T * words).
    /// Requires strictly increasing, NaN-free `thresholds`. Row i is scattered
    /// into bucket b_i = `bin_index(thresholds, x_i)`, then a prefix OR over
    /// buckets gives mask_k = rows with b_i <= k. Bit i of mask_k is set iff
    /// b_i <= k iff x_i <= thr[k] (strictly increasing thresholds, see
    /// `bin_index`); NaN rows (b_i = T) never set a bit. Identical words to
    /// `le_masks_slow`.
    fn le_masks_fast(
        x: &Array2<f64>,
        feat_idx: usize,
        thresholds: &[f64],
        n_samples: usize,
        words_per_mask: usize,
    ) -> Vec<Vec<u64>> {
        let t_len = thresholds.len();
        let mut masks = vec![vec![0u64; words_per_mask]; t_len];
        // Scatter: row i lands in bucket b_i; rows with b_i == T (v > all
        // thresholds, or NaN) belong to no mask.
        for i in 0..n_samples {
            let b = bin_index(thresholds, x[[i, feat_idx]]);
            if b < t_len {
                masks[b][i / 64] |= 1u64 << (i % 64);
            }
        }
        // Cumulative OR: mask_k = rows with b_i <= k  (= x_i <= thr[k], see bin_index).
        for k in 1..t_len {
            let (prev, cur) = masks.split_at_mut(k);
            for (c, p) in cur[0].iter_mut().zip(prev[k - 1].iter()) {
                *c |= *p;
            }
        }
        masks
    }

    /// Build a context with ALL midpoint thresholds pre-binarized (no compression).
    /// Used when compressed thresholds aren't known yet.
    pub fn new_from_raw(x: &Array2<f64>, y: &[u8], n_classes: usize) -> Self {
        // Just build label masks — no feature masks pre-binarized.
        // Features will use lazy binarization.
        let n_samples = y.len();
        let words_per_mask = (n_samples + 63) / 64;
        let n_features = x.ncols();

        let mut label_masks = vec![vec![0u64; words_per_mask]; n_classes];
        for (i, &label) in y.iter().enumerate() {
            label_masks[label as usize][i / 64] |= 1u64 << (i % 64);
        }

        BitmaskContext {
            n_samples,
            n_classes,
            words_per_mask,
            feature_le_masks: vec![Vec::new(); n_features],
            feature_thresholds: vec![Vec::new(); n_features],
            label_masks,
        }
    }

    /// Create the root bitmask (all samples active).
    pub fn root_mask(&self) -> Vec<u64> {
        let mut mask = vec![!0u64; self.words_per_mask];
        Self::clean_last_word(&mut mask, self.n_samples);
        mask
    }

    /// Zero out unused bits in the last word.
    #[inline]
    pub fn clean_last_word(mask: &mut [u64], n_samples: usize) {
        let remainder = n_samples % 64;
        if remainder > 0 && !mask.is_empty() {
            let last = mask.len() - 1;
            mask[last] &= (1u64 << remainder) - 1;
        }
    }

    /// Popcount of a bitmask.
    #[inline]
    pub fn popcount(mask: &[u64]) -> usize {
        mask.iter().map(|w| w.count_ones() as usize).sum()
    }

    /// Count samples per class within a bitmask.
    pub fn count_classes(&self, mask: &[u64]) -> ([i64; MAX_CLASSES], usize) {
        let mut counts = [0i64; MAX_CLASSES];
        let mut total = 0usize;
        for c in 0..self.n_classes {
            let mut count = 0u64;
            for w in 0..self.words_per_mask {
                count += (mask[w] & self.label_masks[c][w]).count_ones() as u64;
            }
            counts[c] = count as i64;
            total += count as usize;
        }
        (counts, total)
    }

    /// Partition a parent bitmask by a k-way split on a feature.
    ///
    /// Returns k child bitmasks, one per bin, along with class counts and sample counts.
    /// The thresholds are k-1 values from feature_thresholds[feature], starting at thresh_base.
    pub fn partition_bitmask(
        &self,
        parent_mask: &[u64],
        feature_index: usize,
        thresh_base: usize,
        k: usize,
    ) -> Vec<(Vec<u64>, [i64; MAX_CLASSES], usize)> {
        let le_masks = &self.feature_le_masks[feature_index];
        let mut result = Vec::with_capacity(k);

        for bin in 0..k {
            let mut child_mask = vec![0u64; self.words_per_mask];
            if bin == 0 {
                let le = &le_masks[thresh_base];
                for w in 0..self.words_per_mask {
                    child_mask[w] = parent_mask[w] & le[w];
                }
            } else if bin == k - 1 {
                let le = &le_masks[thresh_base + k - 2];
                for w in 0..self.words_per_mask {
                    child_mask[w] = parent_mask[w] & !le[w];
                }
                Self::clean_last_word(&mut child_mask, self.n_samples);
            } else {
                let le_upper = &le_masks[thresh_base + bin];
                let le_lower = &le_masks[thresh_base + bin - 1];
                for w in 0..self.words_per_mask {
                    child_mask[w] = parent_mask[w] & le_upper[w] & !le_lower[w];
                }
            }

            let (counts, n) = self.count_classes(&child_mask);
            result.push((child_mask, counts, n));
        }

        result
    }

    /// Partition using raw threshold values (not pre-binarized).
    /// Falls back to scanning X values. Used when feature_le_masks[feat] is empty.
    pub fn partition_bitmask_raw(
        &self,
        parent_mask: &[u64],
        x: &Array2<f64>,
        feature_index: usize,
        thresholds: &[f64],
    ) -> Vec<(Vec<u64>, [i64; MAX_CLASSES], usize)> {
        let k = thresholds.len() + 1;
        let mut bins: Vec<Vec<u64>> = vec![vec![0u64; self.words_per_mask]; k];

        // Scan parent mask bits, assign each active sample to a bin
        for word_idx in 0..self.words_per_mask {
            let mut word = parent_mask[word_idx];
            while word != 0 {
                let bit = word.trailing_zeros() as usize;
                let sample_idx = word_idx * 64 + bit;
                if sample_idx < self.n_samples {
                    let val = x[[sample_idx, feature_index]];
                    let bin = match thresholds.iter().position(|&t| t > val) {
                        Some(pos) => pos,
                        None => k - 1,
                    };
                    bins[bin][word_idx] |= 1u64 << bit;
                }
                word &= word - 1; // clear lowest set bit
            }
        }

        bins.into_iter()
            .map(|mask| {
                let (counts, n) = self.count_classes(&mask);
                (mask, counts, n)
            })
            .collect()
    }

    /// Extract sample indices from a bitmask (escape hatch for code that still
    /// needs Vec<u32>).
    pub fn bitmask_to_indices(mask: &[u64], n_samples: usize) -> Vec<u32> {
        let mut indices = Vec::new();
        for (word_idx, &word) in mask.iter().enumerate() {
            let mut w = word;
            while w != 0 {
                let bit = w.trailing_zeros() as usize;
                let idx = word_idx * 64 + bit;
                if idx < n_samples {
                    indices.push(idx as u32);
                }
                w &= w - 1;
            }
        }
        indices
    }

    /// Find the threshold_base index in feature_thresholds[feat] that corresponds
    /// to the given threshold values. Returns None if not found (threshold isn't
    /// pre-binarized).
    pub fn find_threshold_base(&self, feature_index: usize, thresholds: &[f64]) -> Option<usize> {
        let ft = &self.feature_thresholds[feature_index];
        if ft.is_empty() || thresholds.is_empty() {
            return None;
        }
        // Find the first threshold in the feature's threshold list
        let first = thresholds[0];
        for (i, &t) in ft.iter().enumerate() {
            if (t - first).abs() < 1e-12 {
                // Verify all k-1 thresholds match consecutively
                if i + thresholds.len() <= ft.len() {
                    let all_match = thresholds
                        .iter()
                        .enumerate()
                        .all(|(j, &tj)| (ft[i + j] - tj).abs() < 1e-12);
                    if all_match {
                        return Some(i);
                    }
                }
            }
        }
        None
    }
}
