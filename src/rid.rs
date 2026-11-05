//! Exact Rashomon Importance Distribution (Donnelly et al., NeurIPS 2023)
//! over a factored Rashomon DAG.
//!
//! The importance of feature j for a tree t on evaluation rows (X, y) of
//! size n is the subtractive model reliance with every donor row
//! (Fisher et al.): the expected increase of the 0-1 loss when row i's
//! value of j is replaced by the value of a uniformly random row r
//! (r = i included). Per leaf ℓ with prediction c_ℓ:
//!
//!   T_ℓ = rows whose non-j path conditions to ℓ hold,
//!   D_ℓ = rows whose j path conditions hold (all rows if j is not on the path),
//!   S_ℓ = T_ℓ ∧ D_ℓ (the rows that reach ℓ),
//!   wrong(A, c) = |A| − |A ∧ Y_c|,
//!   a_ℓ = wrong(T_ℓ, c_ℓ)·|D_ℓ| − n·wrong(S_ℓ, c_ℓ),
//!
//! and N_j(t) = Σ_ℓ a_ℓ = Σ_i Σ_r [t(x_i with x_ij ← x_rj) ≠ y_i] − n·Σ_i [t(x_i) ≠ y_i],
//! an integer; φ_j = N_j / n². N_j = 0 exactly when t does not use j.
//!
//! A split's child-i rows depend only on the split spec (bins are
//! partitioned independently of the support, `DatasetStore::partition_support`),
//! so each spec's child masks G[spec][i] are computed once on the full rows
//! and a path context is refined by AND.
//!
//! `rid_exact_histograms` computes, per feature, the exact histogram
//! {N_j(t) → number of trees} over every tree of the DAG within a bound by
//! a memoized walk that mirrors `extract_at_cost` case for case (inline
//! leaves, nested prefix-leaf DAGs, vertex leaf options, splits with the
//! same child scopes and cost combos). Histograms of a split's children
//! combine by convolution (values add, counts multiply).

use std::collections::HashMap;
use std::rc::Rc;
use std::time::Instant;

use crate::dataset_store::DatasetStore;
use crate::rashomon_dag::{
    cost_key_with_precision, iter_splits, materialize_schedule, root_entry_within_bound,
    visit_combos_at_target,
};
use crate::tree::TreeNode;
use crate::types::{
    count_add, count_mul, count_product, is_leaf_ref, leaf_id, CostEntry, FSubRef, FactoredDag,
};

/// Options of `rid_exact_histograms`.
#[derive(Clone, Debug)]
pub struct RidOpts {
    /// Root cost entries within this bound (`root_entry_within_bound`);
    /// None = every root entry (all `tree_count` trees).
    pub bound: Option<f64>,
    /// Features to compute; None = all. Others get an empty histogram.
    pub features: Option<Vec<usize>>,
    /// Abort (`complete = false`) once more memo states than this are built.
    pub max_states: usize,
    /// Abort (`complete = false`) once this instant has passed.
    pub deadline: Option<Instant>,
    /// Abort (`complete = false`) once the interned row masks of one
    /// feature take more than this many bytes.
    pub max_mask_bytes: usize,
}

/// Default `RidOpts::max_mask_bytes` (2 GiB).
pub const DEFAULT_MAX_MASK_BYTES: usize = 2 << 30;

/// Exact per-feature histograms of N_j over the trees of the DAG.
#[derive(Clone, Debug, Default)]
pub struct RidHist {
    /// `per_feature[j]` = sorted (N_j value, tree count), equal values merged,
    /// zero counts dropped; length = number of features.
    pub per_feature: Vec<Vec<(i64, u128)>>,
    /// Σ of the DAG's root-entry tree counts within the bound.
    pub n_trees: u128,
    /// Memo states built (summed over features).
    pub n_states: u64,
    /// (vertex, cost) entries whose histogram count differs from the DAG's
    /// tree count for that entry (0 when the walk enumerates exactly the
    /// trees `get_tree` decodes).
    pub count_mismatch: u64,
    /// Minority-class leaf options met while building states.
    pub n_minority_leaf_options: u64,
    /// Peak bytes of the interned row masks (over features; each feature
    /// starts from an empty table).
    pub mask_bytes_peak: u64,
    /// False when the walk stopped early (`max_states` or `deadline`).
    pub complete: bool,
}

type Hist = Vec<(i64, u128)>;

#[inline]
fn popcount(w: &[u64]) -> i64 {
    w.iter().map(|x| x.count_ones() as i64).sum()
}

#[inline]
fn and_popcount(a: &[u64], b: &[u64]) -> i64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x & y).count_ones() as i64)
        .sum()
}

#[inline]
fn and3_popcount(a: &[u64], b: &[u64], c: &[u64]) -> i64 {
    a.iter()
        .zip(b)
        .zip(c)
        .map(|((x, y), z)| (x & y & z).count_ones() as i64)
        .sum()
}

/// a_ℓ = wrong(T, c)·|D| − n·wrong(T ∧ D, c) for a leaf predicting `pred`.
fn leaf_term(labels: &[std::sync::Arc<[u64]>], n: i64, t: &[u64], d: &[u64], pred: usize) -> i64 {
    let nt = popcount(t);
    let nd = popcount(d);
    let ns = and_popcount(t, d);
    let (nty, nsy) = match labels.get(pred) {
        Some(yc) => (and_popcount(t, yc), and3_popcount(t, d, yc)),
        None => (0, 0),
    };
    (nt - nty) * nd - n * (ns - nsy)
}

/// Child masks of a split spec on the full rows of `store` (bins of one
/// child OR-ed together for a group/shape split).
fn spec_child_masks(
    store: &DatasetStore,
    feature_index: usize,
    thresholds: &[f64],
    group_map: Option<&[usize]>,
) -> Vec<Box<[u64]>> {
    let root = store.root_support();
    store
        .partition_support(&root, feature_index, thresholds, group_map)
        .into_iter()
        .map(|m| Box::<[u64]>::from(m.words.as_ref()))
        .collect()
}

/// Sort by value, merge equal values, drop zero counts.
fn merge_hist(mut v: Hist) -> Hist {
    v.sort_unstable_by_key(|e| e.0);
    let mut out: Hist = Vec::with_capacity(v.len());
    for (val, c) in v {
        if c == 0 {
            continue;
        }
        match out.last_mut() {
            Some(last) if last.0 == val => last.1 = count_add(last.1, c),
            _ => out.push((val, c)),
        }
    }
    out
}

/// Pairwise sums of values, products of counts.
fn convolve(a: &[(i64, u128)], b: &[(i64, u128)]) -> Hist {
    if a.len() == 1 && a[0] == (0, 1) {
        return b.to_vec();
    }
    let mut out: Hist = Vec::with_capacity(a.len() * b.len());
    for &(va, ca) in a {
        for &(vb, cb) in b {
            out.push((va + vb, count_mul(ca, cb)));
        }
    }
    merge_hist(out)
}

#[inline]
fn hist_total(h: &[(i64, u128)]) -> u128 {
    h.iter().fold(0u128, |s, e| count_add(s, e.1))
}

// ---------------------------------------------------------------------------
// Per-tree numerators
// ---------------------------------------------------------------------------

/// N_j(t) for every feature j (length = `store.n_features()`), evaluated on
/// the rows of `store`. Features the tree does not split on get 0.
pub fn tree_phi_numerators(tree: &TreeNode, store: &DatasetStore) -> Vec<i64> {
    type SpecKey = (usize, Vec<u64>, Option<Vec<usize>>);
    fn used(node: &TreeNode, out: &mut Vec<bool>) {
        if let TreeNode::Internal {
            feature_index,
            children,
            ..
        } = node
        {
            if *feature_index < out.len() {
                out[*feature_index] = true;
            }
            for c in children {
                used(c, out);
            }
        }
    }
    #[allow(clippy::too_many_arguments)]
    fn walk(
        node: &TreeNode,
        j: usize,
        t: &[u64],
        d: &[u64],
        store: &DatasetStore,
        n: i64,
        cache: &mut HashMap<SpecKey, Rc<Vec<Box<[u64]>>>>,
    ) -> i64 {
        match node {
            TreeNode::Leaf { prediction, .. } => {
                leaf_term(store.label_masks(), n, t, d, *prediction)
            }
            TreeNode::Internal {
                feature_index,
                thresholds,
                children,
                group_map,
            } => {
                let key: SpecKey = (
                    *feature_index,
                    thresholds.iter().map(|x| x.to_bits()).collect(),
                    group_map.clone(),
                );
                let masks = cache
                    .entry(key)
                    .or_insert_with(|| {
                        Rc::new(spec_child_masks(
                            store,
                            *feature_index,
                            thresholds,
                            group_map.as_deref(),
                        ))
                    })
                    .clone();
                assert_eq!(
                    masks.len(),
                    children.len(),
                    "tree_phi_numerators: split on feature {} routes to {} parts for {} children",
                    feature_index,
                    masks.len(),
                    children.len()
                );
                let mut acc = 0i64;
                for (child, g) in children.iter().zip(masks.iter()) {
                    let refined: Vec<u64> = if *feature_index == j {
                        d.iter().zip(g.iter()).map(|(a, b)| a & b).collect()
                    } else {
                        t.iter().zip(g.iter()).map(|(a, b)| a & b).collect()
                    };
                    acc += if *feature_index == j {
                        walk(child, j, t, &refined, store, n, cache)
                    } else {
                        walk(child, j, &refined, d, store, n, cache)
                    };
                }
                acc
            }
        }
    }

    let p = store.n_features();
    let n = store.n_samples() as i64;
    let mut is_used = vec![false; p];
    used(tree, &mut is_used);
    let all: Vec<u64> = store.root_support().words.to_vec();
    let mut cache: HashMap<SpecKey, Rc<Vec<Box<[u64]>>>> = HashMap::new();
    (0..p)
        .map(|j| {
            if is_used[j] {
                walk(tree, j, &all, &all, store, n, &mut cache)
            } else {
                0
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Exact histograms over the DAG
// ---------------------------------------------------------------------------

#[inline]
fn dag_id(dag: &FactoredDag) -> usize {
    dag as *const FactoredDag as usize
}

/// Memo key: (dag id, vertex ref, cost key, T mask id, D mask id).
type StateKey = (usize, FSubRef, u64, u32, u32);

struct Walker<'a> {
    store: &'a DatasetStore,
    n: i64,
    /// Feature-bitset words.
    fwords: usize,
    /// Interned row masks: one copy per mask, shared by the lookup map and
    /// the id table. Cleared per feature.
    mask_ids: HashMap<Rc<[u64]>, u32>,
    masks: Vec<Rc<[u64]>>,
    mask_bytes: usize,
    mask_bytes_peak: usize,
    max_mask_bytes: usize,
    all_id: u32,
    /// (dag id, spec id) → index into `gtab`.
    gidx: HashMap<(usize, u32), usize>,
    gtab: Vec<Vec<Box<[u64]>>>,
    /// (mask id, gtab index, child) → refined mask id. Cleared per feature.
    refine_cache: HashMap<(u32, usize, u32), u32>,
    /// Per DAG: features split on at or below each vertex (None = not built).
    used_below: HashMap<usize, Vec<Option<Rc<[u64]>>>>,
    /// Per feature: state → (histogram, its total count).
    memo: HashMap<StateKey, (Rc<Hist>, u128)>,
    j: usize,
    max_states: usize,
    deadline: Option<Instant>,
    aborted: bool,
    n_states: u64,
    count_mismatch: u64,
    n_minority: u64,
}

impl<'a> Walker<'a> {
    fn intern(&mut self, w: Vec<u64>) -> u32 {
        if let Some(&id) = self.mask_ids.get(w.as_slice()) {
            return id;
        }
        let id = self.masks.len() as u32;
        self.mask_bytes += w.len() * std::mem::size_of::<u64>();
        self.mask_bytes_peak = self.mask_bytes_peak.max(self.mask_bytes);
        if self.mask_bytes > self.max_mask_bytes {
            self.aborted = true;
        }
        let rc: Rc<[u64]> = Rc::from(w);
        self.masks.push(rc.clone());
        self.mask_ids.insert(rc, id);
        id
    }

    /// Drop the interned masks, the refine cache and the memo (all keyed by
    /// mask ids) and re-intern the all-rows mask.
    fn reset_masks(&mut self, root_all: &[u64]) {
        self.memo.clear();
        self.refine_cache.clear();
        self.mask_ids.clear();
        self.masks.clear();
        self.mask_bytes = 0;
        self.all_id = self.intern(root_all.to_vec());
    }

    fn spec_table(&mut self, dag: &FactoredDag, spec_id: u32, k: usize) -> usize {
        let key = (dag_id(dag), spec_id);
        if let Some(&g) = self.gidx.get(&key) {
            return g;
        }
        let spec = &dag.spec_pool[spec_id as usize];
        let gm: Option<Vec<usize>> = spec
            .group_map
            .as_ref()
            .map(|gm| gm.iter().map(|&g| g as usize).collect());
        let masks = spec_child_masks(
            self.store,
            spec.feature_index as usize,
            &spec.thresholds,
            gm.as_deref(),
        );
        assert_eq!(
            masks.len(),
            k,
            "rid_exact_histograms: split on feature {} routes to {} parts for {} children",
            spec.feature_index,
            masks.len(),
            k
        );
        let g = self.gtab.len();
        self.gtab.push(masks);
        self.gidx.insert(key, g);
        g
    }

    fn refine(&mut self, mask: u32, g: usize, child: usize) -> u32 {
        let key = (mask, g, child as u32);
        if let Some(&id) = self.refine_cache.get(&key) {
            return id;
        }
        let w: Vec<u64> = self.masks[mask as usize]
            .iter()
            .zip(self.gtab[g][child].iter())
            .map(|(a, b)| a & b)
            .collect();
        let id = self.intern(w);
        self.refine_cache.insert(key, id);
        id
    }

    /// Features split on at or below `v` in `dag`. Inline leaves use none.
    fn used_bits(&mut self, dag: &FactoredDag, v: FSubRef) -> Option<Rc<[u64]>> {
        if is_leaf_ref(v) || (v as usize) >= dag.subproblems.len() {
            return None;
        }
        let id = dag_id(dag);
        if let Some(Some(b)) = self.used_below.get(&id).and_then(|t| t.get(v as usize)) {
            return Some(b.clone());
        }
        let mut bits = vec![0u64; self.fwords];
        for split in iter_splits(dag, v) {
            let f = dag.spec_of(split).feature_index as usize;
            if f / 64 < bits.len() {
                bits[f / 64] |= 1u64 << (f % 64);
            }
            for &c in split.children.iter() {
                if let Some(cb) = self.used_bits(dag, c) {
                    for (a, b) in bits.iter_mut().zip(cb.iter()) {
                        *a |= *b;
                    }
                }
            }
        }
        let rc: Rc<[u64]> = Rc::from(bits);
        let n_sub = dag.subproblems.len();
        let table = self
            .used_below
            .entry(id)
            .or_insert_with(|| vec![None; n_sub]);
        table[v as usize] = Some(rc.clone());
        Some(rc)
    }

    fn uses_j(&mut self, dag: &FactoredDag, v: FSubRef) -> bool {
        let j = self.j;
        match self.used_bits(dag, v) {
            Some(b) => (b[j / 64] >> (j % 64)) & 1 == 1,
            None => false,
        }
    }

    fn leaf(&self, t: u32, d: u32, pred: usize) -> i64 {
        leaf_term(
            self.store.label_masks(),
            self.n,
            &self.masks[t as usize],
            &self.masks[d as usize],
            pred,
        )
    }

    /// Histogram of N_j over the trees `extract_at_cost(dag, v, cost, ·)`
    /// decodes, under path context (T = `t`, D = `d`). `expected` is the
    /// DAG's tree count for this (v, cost) entry.
    fn h(
        &mut self,
        dag: &FactoredDag,
        v: FSubRef,
        t: u32,
        d: u32,
        cost: f64,
        expected: u128,
    ) -> Rc<Hist> {
        if self.aborted {
            return Rc::new(Vec::new());
        }
        // Shortcut: no j condition on the path and none below → N_j = 0.
        if d == self.all_id && !self.uses_j(dag, v) {
            return Rc::new(if expected > 0 {
                vec![(0, expected)]
            } else {
                Vec::new()
            });
        }
        let prec = dag.cost_precision;
        let ck_ = |c: f64| -> u64 { cost_key_with_precision(c, prec) };
        let target_key = ck_(cost);
        let key: StateKey = (dag_id(dag), v, target_key, t, d);
        if let Some((hist, total)) = self.memo.get(&key) {
            if *total != expected {
                self.count_mismatch += 1;
            }
            return hist.clone();
        }
        self.n_states += 1;
        if self.n_states as usize > self.max_states
            || (self.n_states % 256 == 0
                && self.deadline.map(|dl| Instant::now() > dl).unwrap_or(false))
        {
            self.aborted = true;
            return Rc::new(Vec::new());
        }

        let mut parts: Hist = Vec::new();
        if is_leaf_ref(v) {
            // extract_at_cost Phase A: inline leaf, majority and/or minority.
            let id = leaf_id(v);
            let maj_matches = ck_(dag.inline_leaf_costs[id]) == target_key;
            let min_matches = dag
                .inline_leaf_minority_costs
                .get(id)
                .copied()
                .unwrap_or(None)
                .map(|mc| ck_(mc) == target_key)
                .unwrap_or(false);
            if maj_matches {
                let pred = dag.inline_leaves[id].prediction as usize;
                parts.push((self.leaf(t, d, pred), 1));
            }
            if min_matches {
                if let Some(leaf) = dag.inline_leaves_minority.get(id).and_then(|o| o.as_ref()) {
                    self.n_minority += 1;
                    parts.push((self.leaf(t, d, leaf.prediction as usize), 1));
                }
            }
        } else {
            let sub = &dag.subproblems[v as usize];
            // 1. Majority leaf option, 1b. minority leaf option.
            if let Some((leaf_cost, ref info)) = sub.leaf {
                if ck_(leaf_cost) == target_key {
                    parts.push((self.leaf(t, d, info.prediction as usize), 1));
                }
            }
            if let Some(&(leaf_cost, ref info)) = sub.leaf_minority.as_deref() {
                if ck_(leaf_cost) == target_key {
                    self.n_minority += 1;
                    parts.push((self.leaf(t, d, info.prediction as usize), 1));
                }
            }
            // 2. Splits: children at the flat bound and the same cost
            // combos as extract.
            for split in iter_splits(dag, v) {
                let mut child_schedules = Vec::with_capacity(split.children.len());
                let mut any_empty = false;
                for &cref in split.children.iter() {
                    let arc = materialize_schedule(dag, cref, dag.rashomon_bound);
                    if arc.is_empty() {
                        any_empty = true;
                        break;
                    }
                    child_schedules.push(arc);
                }
                if any_empty {
                    continue;
                }
                let spec = dag.spec_of(split);
                let on_j = spec.feature_index as usize == self.j;
                let k = split.children.len();
                let g = self.spec_table(dag, split.spec_id, k);
                let ctxs: Vec<(u32, u32)> = (0..k)
                    .map(|i| {
                        if on_j {
                            (t, self.refine(d, g, i))
                        } else {
                            (self.refine(t, g, i), d)
                        }
                    })
                    .collect();
                visit_combos_at_target(&child_schedules, cost, prec, dag.keyed_costs, |combo| {
                    let combo_count: u128 = count_product(combo.iter().map(|e| e.tree_count()));
                    if combo_count == 0 {
                        return true;
                    }
                    let mut acc: Hist = vec![(0, 1)];
                    for (i, e) in combo.iter().enumerate() {
                        let (ct, cd) = ctxs[i];
                        let hc = self.h(dag, split.children[i], ct, cd, e.cost, e.tree_count());
                        acc = convolve(&acc, &hc);
                        if acc.is_empty() {
                            break;
                        }
                    }
                    parts.extend(acc);
                    !self.aborted
                });
            }
        }
        if self.aborted {
            return Rc::new(Vec::new());
        }
        let hist = merge_hist(parts);
        let total = hist_total(&hist);
        if total != expected {
            self.count_mismatch += 1;
        }
        let rc = Rc::new(hist);
        self.memo.insert(key, (rc.clone(), total));
        rc
    }
}

/// Exact per-feature histograms of N_j (see the module doc) over the trees
/// of `dag` whose root cost entry is within `opts.bound`, evaluated on the
/// rows of `store` (which must bin like the DAG's training store).
pub fn rid_exact_histograms(dag: &FactoredDag, store: &DatasetStore, opts: &RidOpts) -> RidHist {
    let p = store.n_features();
    let root_all = store.root_support().words.to_vec();
    let mut w = Walker {
        store,
        n: store.n_samples() as i64,
        fwords: (p + 63) / 64,
        mask_ids: HashMap::new(),
        masks: Vec::new(),
        mask_bytes: 0,
        mask_bytes_peak: 0,
        max_mask_bytes: opts.max_mask_bytes,
        all_id: 0,
        gidx: HashMap::new(),
        gtab: Vec::new(),
        refine_cache: HashMap::new(),
        used_below: HashMap::new(),
        memo: HashMap::new(),
        j: 0,
        max_states: opts.max_states,
        deadline: opts.deadline,
        aborted: false,
        n_states: 0,
        count_mismatch: 0,
        n_minority: 0,
    };

    let entries: Vec<CostEntry> = dag
        .root_cost_schedule
        .iter()
        .filter(|e| {
            opts.bound
                .map(|b| root_entry_within_bound(dag, e.cost, b))
                .unwrap_or(true)
        })
        .cloned()
        .collect();
    let n_trees = entries
        .iter()
        .fold(0u128, |s, e| count_add(s, e.tree_count()));

    let features: Vec<usize> = match &opts.features {
        Some(fs) => fs.iter().copied().filter(|&j| j < p).collect(),
        None => (0..p).collect(),
    };
    let mut per_feature: Vec<Hist> = vec![Vec::new(); p];
    for j in features {
        w.j = j;
        w.reset_masks(&root_all);
        let mut parts: Hist = Vec::new();
        for e in &entries {
            let all = w.all_id;
            let h = w.h(dag, dag.root_ref, all, all, e.cost, e.tree_count());
            parts.extend_from_slice(&h);
        }
        if w.aborted {
            break;
        }
        per_feature[j] = merge_hist(parts);
    }

    RidHist {
        per_feature,
        n_trees,
        n_states: w.n_states,
        count_mismatch: w.count_mismatch,
        n_minority_leaf_options: w.n_minority,
        mask_bytes_peak: w.mask_bytes_peak as u64,
        complete: !w.aborted,
    }
}
