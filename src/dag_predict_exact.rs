//! Exact per-row class counts over the Rashomon set (`class_counts`).
//!
//! For a row x and a vertex v, `S_v^x : cost key K -> (c0, c1)` holds the
//! numbers of trees of v's schedule entry K that predict class 0 / class 1
//! on x. It mirrors the count pass (`wide_vertex_schedule` /
//! `split_schedule_wide` / `convolve_cost_schedules_keyed` in
//! `rashomon_dag.rs`) entry for entry, so `c0 + c1` equals the plain
//! schedule count bitwise:
//!   - a leaf option of cost c fits the scope key s iff ck(c) <= s and adds
//!     (ck(c), [pred == 0], [pred == 1]);
//!   - a split routes x to exactly one child i; its contribution is the
//!     keyed convolution of `S_{c_i}^x` (at child scope i) with the PLAIN
//!     schedules of the other children (at theirs), capped at s:
//!     `c_cls(K) = Σ_{Σ K_j = K} c_cls(c_i, K_i) · Π_{j≠i} tc_j(K_j)`.
//! At the root (scope beta = `rashomon_bound`), the class-1 count of x over
//! the trees within `bound` is `Σ_{K within bound} c1(K)`: exact, unlike the
//! cost-blind `dag_predict::batch_predict_probs`.
//!
//! One schedule per visited vertex and row, computed at the widest scope
//! key the count pass asks (`WideScheduleCache::hints`); narrower requests
//! read its prefix (the argument of `wide_schedule_prefix`). Rows are
//! reduced to per-feature bins of the union of the DAG's thresholds on
//! that feature (every split spec routes rows of one bin alike); rows with
//! the same bins share one DP. `verify` compares
//! every vertex's `c0 + c1` with `materialize_schedule` (and the root with
//! `root_cost_schedule`); any difference is counted in `count_mismatch`.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use rayon::prelude::*;
use rustc_hash::FxHashMap;

use crate::dag_predict::child_index_for;
use crate::rashomon_dag::{
    child_min_obj, cost_key_with_precision, iter_splits, materialize_schedule,
    materialize_schedule_wide, sched_wide_active, scope_key_with_precision, tree_count_at_bound,
    SCOPE_EPS,
};
use crate::types::{
    count_add, count_mul, is_leaf_ref, leaf_id, CostEntry, FSubRef, FactoredDag, FactoredSplit,
};

/// One entry of a class schedule: cost key and per-class tree counts.
#[derive(Clone, Copy, Debug)]
pub struct ClassEntry {
    pub key: i64,
    pub c: [u128; 2],
}

pub struct ExactPredictOpts {
    /// Root bound of the counted set (None = `rashomon_bound`, all trees).
    pub bound: Option<f64>,
    pub deadline: Option<Instant>,
    /// Check every vertex's class totals against the plain schedule.
    pub verify: bool,
    /// Cap on vertex computations over all rows (then `complete = false`).
    pub max_states: usize,
}

pub struct ExactPredictResult {
    pub counts1: Vec<u128>,
    pub counts0: Vec<u128>,
    pub n_trees: u128,
    pub n_states: u64,
    pub n_unique_rows: usize,
    pub count_mismatch: u64,
    pub complete: bool,
}

/// Shared across the rows of one call.
struct Shared<'a> {
    opts: &'a ExactPredictOpts,
    n_states: AtomicU64,
    mismatch: AtomicU64,
    aborted: AtomicBool,
    multiclass: AtomicBool,
}

/// Plain schedule of `v` at `scope` as `(arc, len)` (answer `arc[..len]`),
/// without the prefix copy `materialize_schedule` makes on wide DAGs.
fn plain_prefix(dag: &FactoredDag, v: FSubRef, scope: f64) -> (Arc<Vec<CostEntry>>, usize) {
    if sched_wide_active(dag) {
        return materialize_schedule_wide(dag, v, scope);
    }
    let arc = materialize_schedule(dag, v, scope);
    let n = arc.len();
    (arc, n)
}

/// Class schedule memo of one row: per (DAG arena, vertex), the schedule
/// at the scope key it was computed at.
type Memo = FxHashMap<(usize, FSubRef), (i64, Arc<Vec<ClassEntry>>)>;

struct RowWalker<'a, 'b> {
    x: &'b [f64],
    sh: &'b Shared<'a>,
    memo: Memo,
    local_states: u64,
}

fn dag_id(dag: &FactoredDag) -> usize {
    dag as *const FactoredDag as usize
}

impl<'a, 'b> RowWalker<'a, 'b> {
    fn abort(&self) -> bool {
        self.sh.aborted.load(Ordering::Relaxed)
    }

    fn class_of(&self, pred: u8) -> usize {
        if pred > 1 {
            self.sh.multiclass.store(true, Ordering::Relaxed);
            self.sh.aborted.store(true, Ordering::Relaxed);
            return 0;
        }
        pred as usize
    }

    /// Class schedule of an inline-leaf ref at the f64 `scope` (mirrors
    /// `inline_leaf_schedule`, entry order included).
    fn inline_leaf(&self, dag: &FactoredDag, v: FSubRef, scope: f64) -> Vec<ClassEntry> {
        let prec = dag.cost_precision;
        let ck = |c: f64| -> i64 { cost_key_with_precision(c, prec) as i64 };
        let id = leaf_id(v);
        let maj_cost = dag.inline_leaf_costs[id];
        let maj = self.class_of(dag.inline_leaves[id].prediction);
        let one = |cls: usize| -> [u128; 2] {
            let mut c = [0u128; 2];
            c[cls] = 1;
            c
        };
        let min_cost_opt = dag
            .inline_leaf_minority_costs
            .get(id)
            .copied()
            .unwrap_or(None);
        match min_cost_opt {
            Some(min_cost) => {
                // The plain schedule only reads the minority cost; its class
                // comes from `inline_leaves_minority` (absent: count it as
                // the majority class, the tree is not decodable anyway).
                let min_cls = dag
                    .inline_leaves_minority
                    .get(id)
                    .and_then(|o| o.as_ref())
                    .map(|l| self.class_of(l.prediction))
                    .unwrap_or(maj);
                let maj_fits = maj_cost <= scope + SCOPE_EPS;
                let min_fits = min_cost <= scope + SCOPE_EPS;
                match (maj_fits, min_fits) {
                    (false, false) => Vec::new(),
                    (true, true) => {
                        if ck(maj_cost) == ck(min_cost) {
                            let mut c = one(maj);
                            c[min_cls] += 1;
                            vec![ClassEntry {
                                key: ck(maj_cost),
                                c,
                            }]
                        } else {
                            vec![
                                ClassEntry {
                                    key: ck(maj_cost),
                                    c: one(maj),
                                },
                                ClassEntry {
                                    key: ck(min_cost),
                                    c: one(min_cls),
                                },
                            ]
                        }
                    }
                    (true, false) => vec![ClassEntry {
                        key: ck(maj_cost),
                        c: one(maj),
                    }],
                    (false, true) => vec![ClassEntry {
                        key: ck(min_cost),
                        c: one(min_cls),
                    }],
                }
            }
            None => {
                if maj_cost <= scope + SCOPE_EPS {
                    vec![ClassEntry {
                        key: ck(maj_cost),
                        c: one(maj),
                    }]
                } else {
                    Vec::new()
                }
            }
        }
    }

    /// Class schedule of `v` at the f64 `scope` as `(rc, len)` (answer
    /// `rc[..len]`), like `wide_schedule_prefix`.
    fn schedule(
        &mut self,
        dag: &FactoredDag,
        v: FSubRef,
        scope: f64,
    ) -> (Arc<Vec<ClassEntry>>, usize) {
        if is_leaf_ref(v) {
            let s = self.inline_leaf(dag, v, scope);
            let n = s.len();
            return (Arc::new(s), n);
        }
        let empty = || (Arc::new(Vec::new()), 0usize);
        let prec = dag.cost_precision;
        let Some(sub) = dag.subproblems.get(v as usize) else {
            return empty();
        };
        let scope_k = scope_key_with_precision(scope, prec, dag.exact_cost_keys);
        if cost_key_with_precision(sub.min_obj, prec) as i64 > scope_k {
            return empty();
        }
        let mkey = (dag_id(dag), v);
        if let Some((k_max, rc)) = self.memo.get(&mkey) {
            if scope_k <= *k_max {
                let n = rc.partition_point(|e| e.key <= scope_k);
                return (rc.clone(), n);
            }
        }
        if self.abort() {
            return empty();
        }
        self.local_states += 1;
        let total = self.sh.n_states.fetch_add(1, Ordering::Relaxed) + 1;
        if total as usize > self.sh.opts.max_states
            || (self.local_states % 256 == 0
                && self
                    .sh
                    .opts
                    .deadline
                    .map(|dl| Instant::now() > dl)
                    .unwrap_or(false))
        {
            self.sh.aborted.store(true, Ordering::Relaxed);
            return empty();
        }
        let hint = if sched_wide_active(dag) {
            dag.wide_schedule_cache
                .hints
                .get(v as usize)
                .copied()
                .unwrap_or(i64::MIN)
        } else {
            i64::MIN
        };
        let target_k = scope_k.max(hint);
        let sched = self.vertex(dag, v, target_k);
        if self.abort() {
            return empty();
        }
        if self.sh.opts.verify {
            let (plain, len) = plain_prefix(dag, v, target_k as f64 / prec);
            let same = len == sched.len()
                && plain[..len].iter().zip(sched.iter()).all(|(p, e)| {
                    cost_key_with_precision(p.cost, prec) as i64 == e.key
                        && p.tree_count() == count_add(e.c[0], e.c[1])
                });
            if !same {
                self.sh.mismatch.fetch_add(1, Ordering::Relaxed);
            }
        }
        let rc = Arc::new(sched);
        self.memo.insert(mkey, (target_k, rc.clone()));
        let n = rc.partition_point(|e| e.key <= scope_k);
        (rc, n)
    }

    /// Class schedule of vertex `v` at scope key `scope_k` (mirrors
    /// `wide_vertex_schedule`).
    fn vertex(&mut self, dag: &FactoredDag, v: FSubRef, scope_k: i64) -> Vec<ClassEntry> {
        let prec = dag.cost_precision;
        let ck = |c: f64| -> i64 { cost_key_with_precision(c, prec) as i64 };
        let sub = &dag.subproblems[v as usize];
        let mut acc: BTreeMap<i64, [u128; 2]> = BTreeMap::new();
        let add = |acc: &mut BTreeMap<i64, [u128; 2]>, key: i64, c: [u128; 2]| {
            let a = acc.entry(key).or_insert([0, 0]);
            a[0] = count_add(a[0], c[0]);
            a[1] = count_add(a[1], c[1]);
        };
        if let Some((leaf_cost, ref info)) = sub.leaf {
            if ck(leaf_cost) <= scope_k {
                let mut c = [0u128; 2];
                c[self.class_of(info.prediction)] = 1;
                add(&mut acc, ck(leaf_cost), c);
            }
        }
        if let Some(&(leaf_cost, ref info)) = sub.leaf_minority.as_deref() {
            if ck(leaf_cost) <= scope_k {
                let mut c = [0u128; 2];
                c[self.class_of(info.prediction)] = 1;
                add(&mut acc, ck(leaf_cost), c);
            }
        }
        for split in iter_splits(dag, v) {
            for e in self.split(dag, split, scope_k) {
                add(&mut acc, e.key, e.c);
            }
            if self.abort() {
                return Vec::new();
            }
        }
        acc.into_iter()
            .map(|(key, c)| ClassEntry { key, c })
            .collect()
    }

    /// Class schedule of one split at scope key `scope_k` (mirrors
    /// `split_schedule_wide`: same lb keys, child scopes and keyed fold).
    fn split(&mut self, dag: &FactoredDag, split: &FactoredSplit, scope_k: i64) -> Vec<ClassEntry> {
        let prec = dag.cost_precision;
        let k = split.children.len();
        if k == 0 {
            return Vec::new();
        }
        let lb_keys: Vec<i64> = split
            .children
            .iter()
            .map(|&c| {
                let lb = child_min_obj(dag, c);
                if lb.is_finite() {
                    cost_key_with_precision(lb, prec) as i64
                } else {
                    1i64 << 52
                }
            })
            .collect();
        let total_lb_key: i64 = lb_keys.iter().sum();
        if total_lb_key > scope_k {
            return Vec::new();
        }
        let spec = dag.spec_of(split);
        let routed = child_index_for(spec, self.x[spec.feature_index as usize]);
        if routed >= k {
            self.sh.mismatch.fetch_add(1, Ordering::Relaxed);
            return Vec::new();
        }
        let child_scope = |i: usize| (scope_k - (total_lb_key - lb_keys[i])) as f64 / prec;
        let mut siblings: Vec<Option<(Arc<Vec<CostEntry>>, usize)>> = Vec::with_capacity(k);
        for (i, &cref) in split.children.iter().enumerate() {
            if i == routed {
                siblings.push(None);
                continue;
            }
            let (arc, n) = plain_prefix(dag, cref, child_scope(i));
            if n == 0 {
                return Vec::new();
            }
            siblings.push(Some((arc, n)));
        }
        let (rc, n) = self.schedule(dag, split.children[routed], child_scope(routed));
        if n == 0 {
            return Vec::new();
        }
        convolve_class_keyed(&siblings, routed, &rc[..n], scope_k, prec)
    }

    /// Class counts of the trees of `dag` whose root entry is within
    /// `bound` (root scope = `rashomon_bound`, as `root_schedule_at_bound`).
    fn root_counts(&mut self, dag: &FactoredDag, bound: f64) -> [u128; 2] {
        let prec = dag.cost_precision;
        let (rc, n) = self.schedule(dag, dag.root_ref, dag.rashomon_bound);
        if self.abort() {
            return [0, 0];
        }
        let beta_k = scope_key_with_precision(dag.rashomon_bound, prec, dag.exact_cost_keys);
        if self.sh.opts.verify {
            let root: Vec<&ClassEntry> = rc[..n].iter().filter(|e| e.key <= beta_k).collect();
            let same = root.len() == dag.root_cost_schedule.len()
                && dag
                    .root_cost_schedule
                    .iter()
                    .zip(root.iter())
                    .all(|(p, e)| {
                        cost_key_with_precision(p.cost, prec) as i64 == e.key
                            && p.tree_count() == count_add(e.c[0], e.c[1])
                    });
            if !same {
                self.sh.mismatch.fetch_add(1, Ordering::Relaxed);
            }
        }
        let bound_k = scope_key_with_precision(bound, prec, dag.exact_cost_keys);
        let mut c = [0u128; 2];
        for e in rc[..n]
            .iter()
            .filter(|e| e.key <= beta_k && e.key <= bound_k)
        {
            c[0] = count_add(c[0], e.c[0]);
            c[1] = count_add(c[1], e.c[1]);
        }
        c
    }
}

/// `convolve_cost_schedules_keyed` with the routed child's per-class
/// counts: the children are folded in order, the plain ones (`siblings[i]`,
/// `None` at `routed`) as counts, the routed one as class pairs. Entries
/// are sorted ascending, so a key sum above `scope_key` ends a child.
fn convolve_class_keyed(
    siblings: &[Option<(Arc<Vec<CostEntry>>, usize)>],
    routed: usize,
    routed_sched: &[ClassEntry],
    scope_key: i64,
    prec: f64,
) -> Vec<ClassEntry> {
    let key = |c: f64| -> i64 { cost_key_with_precision(c, prec) as i64 };
    let mut suffix_min = vec![0i64; siblings.len() + 1];
    for i in (0..siblings.len()).rev() {
        let min = if i == routed {
            let Some(first) = routed_sched.first() else {
                return Vec::new();
            };
            first.key
        } else {
            let (arc, n) = siblings[i].as_ref().expect("plain sibling");
            if *n == 0 {
                return Vec::new();
            }
            key(arc[0].cost)
        };
        suffix_min[i] = suffix_min[i + 1].saturating_add(min);
    }
    // Before the routed child: plain counts.
    let mut plain: BTreeMap<i64, u128> = BTreeMap::new();
    plain.insert(0, 1);
    for (i, sib) in siblings[..routed].iter().enumerate() {
        let cap = scope_key.saturating_sub(suffix_min[i + 1]);
        let (arc, n) = sib.as_ref().expect("plain sibling");
        let mut next: BTreeMap<i64, u128> = BTreeMap::new();
        for (&pk, &pc) in &plain {
            for e in arc[..*n].iter() {
                let q = pk.saturating_add(key(e.cost));
                if q > cap {
                    break;
                }
                let agg = next.entry(q).or_insert(0);
                *agg = count_add(*agg, count_mul(pc, e.tree_count()));
            }
        }
        if next.is_empty() {
            return Vec::new();
        }
        plain = next;
    }
    // The routed child: class pairs.
    let cap = scope_key.saturating_sub(suffix_min[routed + 1]);
    let mut partial: BTreeMap<i64, [u128; 2]> = BTreeMap::new();
    for (&pk, &pc) in &plain {
        for e in routed_sched {
            let q = pk.saturating_add(e.key);
            if q > cap {
                break;
            }
            let agg = partial.entry(q).or_insert([0, 0]);
            agg[0] = count_add(agg[0], count_mul(pc, e.c[0]));
            agg[1] = count_add(agg[1], count_mul(pc, e.c[1]));
        }
    }
    if partial.is_empty() {
        return Vec::new();
    }
    // After the routed child: plain siblings scale both classes.
    for (offset, sib) in siblings[routed + 1..].iter().enumerate() {
        let cap = scope_key.saturating_sub(suffix_min[routed + offset + 2]);
        let (arc, n) = sib.as_ref().expect("plain sibling");
        let mut next: BTreeMap<i64, [u128; 2]> = BTreeMap::new();
        for (&pk, pc) in &partial {
            for e in arc[..*n].iter() {
                let q = pk.saturating_add(key(e.cost));
                if q > cap {
                    break;
                }
                let tc = e.tree_count();
                let agg = next.entry(q).or_insert([0, 0]);
                agg[0] = count_add(agg[0], count_mul(pc[0], tc));
                agg[1] = count_add(agg[1], count_mul(pc[1], tc));
            }
        }
        if next.is_empty() {
            return Vec::new();
        }
        partial = next;
    }
    partial
        .into_iter()
        .map(|(key, c)| ClassEntry { key, c })
        .collect()
}

/// Every split spec of `dag`, for row signatures.
fn collect_specs<'d>(dag: &'d FactoredDag, out: &mut Vec<&'d crate::types::SplitSpec>) {
    out.extend(dag.spec_pool.iter());
}

/// Per row of `x_flat` (row-major, `n_samples` × `n_features`), the exact
/// numbers of trees within `opts.bound` predicting class 1 / class 0.
/// Requires a keyed-cost DAG with unsaturated counts and binary leaves.
/// `complete = false` when the deadline or `max_states` stopped the walk
/// (counts are then unusable); `count_mismatch > 0` when some class total
/// differs from the plain schedule (then the counts are not trusted).
pub fn class_counts(
    dag: &FactoredDag,
    x_flat: &[f64],
    n_samples: usize,
    n_features: usize,
    opts: &ExactPredictOpts,
) -> Result<ExactPredictResult, String> {
    if !dag.keyed_costs {
        return Err("class_counts needs a keyed-cost DAG (keyed_costs)".into());
    }
    if dag.count_saturated {
        return Err("class_counts: tree count saturated at u128::MAX".into());
    }
    if x_flat.len() != n_samples * n_features {
        return Err(format!(
            "class_counts: x has {} values for {} x {} rows",
            x_flat.len(),
            n_samples,
            n_features
        ));
    }
    let bound = opts.bound.unwrap_or(dag.rashomon_bound);
    let n_trees = tree_count_at_bound(dag, bound);
    let mut res = ExactPredictResult {
        counts1: vec![0; n_samples],
        counts0: vec![0; n_samples],
        n_trees,
        n_states: 0,
        n_unique_rows: 0,
        count_mismatch: 0,
        complete: true,
    };
    if dag.tree_count == 0 || n_samples == 0 {
        return Ok(res);
    }
    let mut specs = Vec::new();
    collect_specs(dag, &mut specs);
    if let Some(s) = specs
        .iter()
        .find(|s| s.feature_index as usize >= n_features)
    {
        return Err(format!(
            "class_counts: a split uses feature {} but x has {} columns",
            s.feature_index, n_features
        ));
    }
    // Per feature, the union of the thresholds of every spec on it: a
    // spec's child index is a function of the row's bin in that union.
    let mut cuts: Vec<Vec<f64>> = vec![Vec::new(); n_features];
    for s in &specs {
        cuts[s.feature_index as usize].extend(s.thresholds.iter().copied());
    }
    for c in cuts.iter_mut() {
        c.sort_by(|a, b| a.total_cmp(b));
        c.dedup_by(|a, b| a.to_bits() == b.to_bits());
    }
    // Rows with the same bins route identically through every spec and
    // share one DP.
    let mut uid_of: FxHashMap<Vec<u32>, usize> = FxHashMap::default();
    let mut rep_row: Vec<usize> = Vec::new();
    let mut row_uid: Vec<usize> = Vec::with_capacity(n_samples);
    for r in 0..n_samples {
        let x = &x_flat[r * n_features..(r + 1) * n_features];
        let bins: Vec<u32> = cuts
            .iter()
            .zip(x.iter())
            .map(|(c, &v)| c.partition_point(|t| *t <= v) as u32)
            .collect();
        let next = rep_row.len();
        let u = *uid_of.entry(bins).or_insert(next);
        if u == next {
            rep_row.push(r);
        }
        row_uid.push(u);
    }
    res.n_unique_rows = rep_row.len();
    let sh = Shared {
        opts,
        n_states: AtomicU64::new(0),
        mismatch: AtomicU64::new(0),
        aborted: AtomicBool::new(false),
        multiclass: AtomicBool::new(false),
    };
    let per_uid: Vec<[u128; 2]> = (0..rep_row.len())
        .into_par_iter()
        .map(|u| {
            let r = rep_row[u];
            let x = &x_flat[r * n_features..(r + 1) * n_features];
            let mut w = RowWalker {
                x,
                sh: &sh,
                memo: Memo::default(),
                local_states: 0,
            };
            w.root_counts(dag, bound)
        })
        .collect();
    if sh.multiclass.load(Ordering::Relaxed) {
        return Err("class_counts supports binary classes only (a leaf predicts class > 1)".into());
    }
    res.n_states = sh.n_states.load(Ordering::Relaxed);
    res.count_mismatch = sh.mismatch.load(Ordering::Relaxed);
    res.complete = !sh.aborted.load(Ordering::Relaxed);
    for (r, &u) in row_uid.iter().enumerate() {
        res.counts0[r] = per_uid[u][0];
        res.counts1[r] = per_uid[u][1];
    }
    if res.complete {
        // Every row's class counts add up to the counted set.
        res.count_mismatch += per_uid
            .iter()
            .filter(|c| count_add(c[0], c[1]) != n_trees)
            .count() as u64;
    }
    Ok(res)
}
