use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use crate::tree::TreeNode;
use crate::types::{
    count_add, count_mul, count_product, count_sum, CostEntry, DeltaBudget, FSubRef, FactoredDag,
    FactoredSplit, FactoredSplitPool, FactoredSubproblem, MemoKey, MemoKeyArena, MemoLookup,
    PrefixLeafFillConfig, PrefixLeafInfo, RLeafInfo, RashomonSet, SearchSnapshot, SplitId,
    SplitRecordPool, SplitSpec,
};

// ---------------------------------------------------------------------------
// RashomonSet ensure_built (lazy DAG materialization)
// ---------------------------------------------------------------------------

impl RashomonSet {
    /// Borrow the FactoredDag, materializing on first access.
    /// Consumes pending search inputs once, allowing search storage to be
    /// released during construction rather than retained alongside the DAG.
    pub fn dag_mut(&mut self) -> &mut FactoredDag {
        self.ensure_built();
        match self {
            RashomonSet::Built(d) => d,
            RashomonSet::Pending { .. } => unreachable!("ensure_built() should have transitioned"),
        }
    }

    pub fn dag(&self) -> &FactoredDag {
        match self {
            RashomonSet::Built(d) => d,
            RashomonSet::Pending { .. } => {
                panic!("RashomonSet::dag() called on Pending; call dag_mut() to materialize")
            }
        }
    }

    /// Materialize the DAG from snapshot if still Pending. Idempotent.
    pub fn ensure_built(&mut self) {
        if let RashomonSet::Built(_) = self {
            return;
        }
        let snap = match self {
            RashomonSet::Pending { snapshot } => snapshot
                .take()
                .expect("RashomonSet::ensure_built called with consumed snapshot"),
            RashomonSet::Built(_) => return,
        };
        let dag = build_dag_from_snapshot(*snap);
        *self = RashomonSet::Built(Box::new(dag));
    }

    /// Wrap an already-materialized factored DAG.
    pub fn from_built(dag: FactoredDag) -> Self {
        RashomonSet::Built(Box::new(dag))
    }
}

/// Build a FactoredDag from a SearchSnapshot, taking ownership of
/// the snapshot's inputs. **Drops memo+pool+arena immediately after the
/// recursive build completes**, before `quotient_factored_dag` runs — so
/// the during-quotient peak is just (DAG + arena Arc refcount), not
/// (memo + pool + arena + DAG). The FactoredDag's `FactoredSplit::children`
/// is `Vec<FSubRef>` — indices into the DAG's own
/// subproblems Vec, not into the original arena/pool — so the build
/// outputs are self-contained and the snapshot is fully releasable.
fn build_dag_from_snapshot(snap: SearchSnapshot) -> FactoredDag {
    let mut dag = build_dag_core(snap);
    // Re-anchor at the DAG optimum on inline splits (before the promotion
    // below drains them).
    reanchor_at_dag_opt(&mut dag);
    // Content-address `FactoredSplit`s into the per-DAG pool
    // after all build/mutation phases are complete (once, at the outer
    // DAG: nested Step-B DAGs are grafted with inline splits).
    promote_splits_to_pool(&mut dag);
    dag
}

/// `build_dag_from_snapshot` without the split-pool promotion: the build,
/// then the completion search (`stepb_fill`), the finalization of keyed
/// full-build DAGs (`finalize_at_bound`), or the quotient alone.
/// MiniRESPLINTER solves call this directly so their DAGs keep inline
/// splits for the graft.
fn build_dag_core(snap: SearchSnapshot) -> FactoredDag {
    let SearchSnapshot {
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
        prefix_leaf_fill,
    } = snap;
    // Completion search (Step B): placeholders at their lower bounds in Step A, then
    // `stepb_fill` (graft + recompute, compaction and quotient last).
    let stepb = prefix_leaf_fill.is_some() && total_max_depth > solve_depth;
    let mut dag = build_factored_dag_impl(
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
        stepb,
    );
    // Drop memo+pool+arena BEFORE quotient. quotient_factored_dag
    // operates on dag.subproblems in-place via `std::mem::take`; it doesn't
    // reference memo/pool/arena. Dropping here lowers the during-quotient
    // peak by the size of memo + pool + arena.
    drop(memo);
    drop(pool);
    drop(arena);
    if stepb {
        let fill_config = prefix_leaf_fill
            .as_ref()
            .expect("stepb implies a fill config");
        stepb_fill(
            &mut dag,
            fill_config,
            solve_depth,
            total_max_depth - solve_depth,
            theta,
            cost_precision,
        );
        return dag;
    }
    // Keyed full-build DAGs without prefix leaves (closure mode, full
    // depth, nested Step-B solves at their last level): lossless trim at
    // beta, then compaction, quotient and children-first order. The tree
    // set is unchanged (`trim_to_bound`).
    if dag.keyed_costs && !dag.subproblems.iter().any(|sp| sp.prefix_leaf.is_some()) {
        let tc_pre = dag.tree_count;
        finalize_at_bound(&mut dag);
        if dag.tree_count != tc_pre {
            eprintln!(
                "  [rashomon] WARNING: trim changed tree_count {} -> {}",
                tc_pre, dag.tree_count
            );
        }
        return dag;
    }
    // Prefix leaves without a completion config (no recursive fill) stay
    // unfilled; with one, `stepb` above holds whenever prefix leaves exist
    // (both need total_max_depth > solve_depth).
    quotient_factored_dag_impl(&mut dag);
    dag
}

// ---------------------------------------------------------------------------
// Completion search (Step B)
// ---------------------------------------------------------------------------

/// Vertices reachable from `root` through split children, parents before
/// children (reverse post-order). After a graft the arena order is not
/// topological, so every Step-B pass walks this order.
fn topo_parents_first(subproblems: &[FactoredSubproblem], root: FSubRef) -> Vec<usize> {
    let n = subproblems.len();
    let mut post: Vec<usize> = Vec::with_capacity(n);
    if (root as usize) >= n {
        return post;
    }
    let mut state = vec![0u8; n];
    let mut stack: Vec<(usize, bool)> = vec![(root as usize, false)];
    while let Some((i, children_done)) = stack.pop() {
        if children_done {
            post.push(i);
            continue;
        }
        if state[i] != 0 {
            continue;
        }
        state[i] = 1;
        stack.push((i, true));
        for split in &subproblems[i].splits {
            for &c in &split.children {
                let c = c as usize;
                if c < n && state[c] == 0 {
                    stack.push((c, false));
                }
            }
        }
    }
    post.reverse();
    post
}

/// Admissible prefix cost of every vertex of a Step-A DAG (placeholders
/// at their lower bounds = their `min_obj`), minus the vertex's own subtree:
///   pc(root) = 0,
///   pc(c_i)  = min over parents v and splits of
///              pc(v) + overhead(v) + Σ_{j≠i} min_obj(c_j).
/// `overhead(v)` is the per-split cost the DAG does not carry (split_reg +
/// path cost; 0 on the leaf-only objective). `INFINITY` for vertices not
/// reachable through splits whose children all have finite `min_obj`.
/// Any tree T through u then has cost(T) - cost(T_u) >= pc(u), so
/// beta_u = beta - pc(u) (= beta_θ - (prefix_cost(u) - lb_u))
/// admits every T_u of a tree with cost(T) <= beta.
pub fn stepb_prefix_costs(dag: &FactoredDag, overhead: &dyn Fn(usize) -> f64) -> Vec<f64> {
    let n = dag.subproblems.len();
    let mut pc = vec![f64::INFINITY; n];
    if (dag.root_ref as usize) >= n {
        return pc;
    }
    pc[dag.root_ref as usize] = 0.0;
    for v in topo_parents_first(&dag.subproblems, dag.root_ref) {
        let pv = pc[v];
        if !pv.is_finite() {
            continue;
        }
        let ov = overhead(v);
        for split in &dag.subproblems[v].splits {
            let mins: Vec<f64> = split
                .children
                .iter()
                .map(|&c| dag.subproblems[c as usize].min_obj)
                .collect();
            if mins.iter().any(|m| !m.is_finite()) {
                continue;
            }
            for (i, &c) in split.children.iter().enumerate() {
                let others: f64 = mins
                    .iter()
                    .enumerate()
                    .filter(|(j, _)| *j != i)
                    .map(|(_, &m)| m)
                    .sum();
                let cand = pv + ov + others;
                if cand < pc[c as usize] {
                    pc[c as usize] = cand;
                }
            }
        }
    }
    pc
}

/// Per-placeholder Step-B scope beta_u = beta - pc(u) (key space on keyed
/// DAGs: scope_key(beta) - key(pc(u))). None when u is unreachable or
/// beta_u < lb_u (= `min_obj(u)`, the placeholder's provisional cost): no
/// tree with cost <= beta goes through u.
pub fn stepb_scope(dag: &FactoredDag, pc: &[f64], u: usize) -> Option<f64> {
    let pu = pc[u];
    let lb = dag.subproblems[u].min_obj;
    if !pu.is_finite() || !lb.is_finite() {
        return None;
    }
    let prec = dag.cost_precision;
    if dag.keyed_costs {
        let bk = scope_key_with_precision(dag.rashomon_bound, prec, dag.exact_cost_keys)
            - cost_key_with_precision(pu, prec) as i64;
        if (cost_key_with_precision(lb, prec) as i64) > bk {
            return None;
        }
        Some(bk as f64 / prec)
    } else {
        let b = dag.rashomon_bound - pu;
        if lb > b + SCOPE_EPS {
            return None;
        }
        Some(b)
    }
}

fn empty_subproblem() -> FactoredSubproblem {
    FactoredSubproblem {
        leaf: None,
        leaf_minority: None,
        prefix_leaf: None,
        splits: Vec::new(),
        split_ids: Vec::new(),
        tree_count: 0,
        min_obj: f64::INFINITY,
        cost_entries: None,
    }
}

/// Build-time `SplitSpecPool` over `dag.spec_pool` for a graft: every outer
/// spec is interned (outer splits' `spec_id`s are remapped should the pool
/// hold a duplicate), so nested specs can be re-interned into the same
/// pool. Install the result with `dag.spec_pool = pool.into_pool()`.
fn stepb_spec_interner(dag: &mut FactoredDag) -> crate::types::SplitSpecPool {
    let mut pool = crate::types::SplitSpecPool::new();
    let remap: Vec<crate::types::SpecId> =
        dag.spec_pool.iter().map(|spec| pool.intern(spec)).collect();
    if remap.iter().enumerate().any(|(i, &id)| id as usize != i) {
        for sp in dag.subproblems.iter_mut() {
            for split in &mut sp.splits {
                split.spec_id = remap[split.spec_id as usize];
            }
        }
    }
    pool
}

/// AttachRecursive: append each nested arena to
/// `dag.subproblems` with its child refs offset and point every split
/// child that referenced one of the listed placeholders at the appended
/// nested root (one shared copy per group, no per-placeholder clone; the
/// placeholder slots are emptied and left unreachable). A nested DAG with
/// no tree, and every vertex in `empty`, becomes an empty vertex (no leaf,
/// no split, min_obj = ∞). Any other remaining `prefix_leaf` vertex
/// (unreachable placeholder) is emptied too. Each nested DAG's `spec_pool`
/// is re-interned into the outer pool (its splits' `spec_id`s remapped).
/// Nested DAGs must not be promoted (inline `splits`, as `build_dag_core`
/// returns them). Counts are not updated: call `stepb_recompute` next.
/// Returns the number of placeholders grafted with a non-empty DAG.
pub fn stepb_graft(
    dag: &mut FactoredDag,
    grafts: Vec<(Vec<usize>, FactoredDag)>,
    empty: &[usize],
) -> usize {
    let mut specs = stepb_spec_interner(dag);
    let mut redirect: HashMap<FSubRef, FSubRef> = HashMap::new();
    let mut n_filled = 0usize;
    for (vs, nested) in grafts {
        if stepb_graft_one(dag, &mut specs, &vs, nested, &mut redirect) {
            n_filled += vs.len();
        }
    }
    dag.spec_pool = specs.into_pool();
    stepb_graft_finish(dag, &redirect, empty);
    n_filled
}

/// One group of `stepb_graft`: append `nested` (moved, not cloned, specs
/// re-interned into `specs`) and record `placeholder -> nested root` in
/// `redirect`, or empty the placeholders when `nested` has no tree.
/// Returns whether it was grafted.
fn stepb_graft_one(
    dag: &mut FactoredDag,
    specs: &mut crate::types::SplitSpecPool,
    vs: &[usize],
    nested: FactoredDag,
    redirect: &mut HashMap<FSubRef, FSubRef>,
) -> bool {
    dag.n_quotient_hash_collisions += nested.n_quotient_hash_collisions;
    if nested.tree_count == 0 || (nested.root_ref as usize) >= nested.subproblems.len() {
        for &v in vs {
            dag.subproblems[v] = empty_subproblem();
        }
        return false;
    }
    debug_assert!(
        nested.split_pool.entries.is_empty() && nested.inline_leaves.is_empty(),
        "stepb_graft: nested DAG must be unpromoted, without inline leaves"
    );
    let spec_map: Vec<crate::types::SpecId> = nested
        .spec_pool
        .iter()
        .map(|spec| specs.intern(spec))
        .collect();
    let offset = dag.subproblems.len() as FSubRef;
    let nested_root = offset + nested.root_ref;
    dag.subproblems.reserve(nested.subproblems.len());
    for mut sp in nested.subproblems {
        for split in &mut sp.splits {
            split.spec_id = spec_map[split.spec_id as usize];
            for c in split.children.iter_mut() {
                *c += offset;
            }
        }
        dag.subproblems.push(sp);
    }
    dag.subproblems[nested_root as usize].prefix_leaf = None;
    for &v in vs {
        dag.subproblems[v] = empty_subproblem();
        redirect.insert(v as FSubRef, nested_root);
    }
    true
}

/// Last step of `stepb_graft`: empty `empty` and every remaining
/// `prefix_leaf` vertex, then apply `redirect` to every split child (and
/// the root).
fn stepb_graft_finish(
    dag: &mut FactoredDag,
    redirect: &HashMap<FSubRef, FSubRef>,
    empty: &[usize],
) {
    for &v in empty {
        dag.subproblems[v] = empty_subproblem();
    }
    for sp in dag.subproblems.iter_mut() {
        if sp.prefix_leaf.is_some() {
            *sp = empty_subproblem();
        }
        if redirect.is_empty() {
            continue;
        }
        for split in &mut sp.splits {
            for c in split.children.iter_mut() {
                if let Some(&r) = redirect.get(c) {
                    *c = r;
                }
            }
        }
    }
    if let Some(&r) = redirect.get(&dag.root_ref) {
        dag.root_ref = r;
    }
}

/// Recompute a grafted DAG at `dag.rashomon_bound` (beta):
///   1. bottom-up in topological order: drop splits with a child of
///      min_obj = ∞ or Σ min_obj(children) > beta (no tree within beta),
///      min_obj = min(leaf cost, Σ min_obj(children) of a kept split);
///   1b. lossless trim to the vertices, splits and leaves on
///      some tree within beta (`trim_to_bound`);
///   2. split / vertex tree_count at beta (keyed/exact convolve, child i at
///      beta - Σ_{j≠i} min_obj(c_j), `materialize_split_schedule`);
///   3. compaction, then the quotient (QuotientByIsomorphism last; the
///      hash runs children-first because the grafted arena is not), then
///      the arena is renumbered
///      children-first (`renumber_children_first`) so consumers that assume
///      arena order (save/load) stay valid;
///   4. fresh schedule cache, root schedule = materialize_schedule(root,
///      beta) filtered at beta, tree_count. Extraction materializes children
///      at the same key (`dag.rashomon_bound`), so counts and get_tree agree.
pub fn stepb_recompute(dag: &mut FactoredDag) {
    finalize_at_bound(dag);
}

/// Re-anchor: when the DAG optimum opt_dag (min cost of the root
/// schedule) is strictly below the anchor `dag.opt` (in the DAG's own
/// metric: cost keys on keyed DAGs, else float with SCOPE_EPS), set
/// `dag.opt = opt_dag`, `dag.rashomon_bound = opt_dag·(1+θ)` and recompute
/// the DAG there (`finalize_at_bound`: trim, counts, compaction, quotient,
/// children-first order, root schedule). Sound: a smaller beta only removes
/// trees. Complete: beta_new ≤ beta_enum, and Step A/B enumerated every pool
/// tree within beta_enum, so the set at beta_new is exactly the pool trees
/// with cost ≤ (1+θ)·opt_dag (a lower bound after a time-out, as before).
/// Skipped (false) when a prefix-leaf vertex remains (no Step B ran: no
/// recursive fill). Records the pre-anchor opt, bound and count
/// in `delta_budget`. Returns whether it re-anchored.
pub fn reanchor_at_dag_opt(dag: &mut FactoredDag) -> bool {
    let Some(opt_dag) = dag
        .root_cost_schedule
        .iter()
        .map(|e| e.cost)
        .reduce(f64::min)
    else {
        return false;
    };
    if dag.subproblems.iter().any(|sp| sp.prefix_leaf.is_some()) {
        return false;
    }
    let prec = dag.cost_precision;
    let cheaper = if dag.keyed_costs {
        cost_key_with_precision(opt_dag, prec) < cost_key_with_precision(dag.opt, prec)
    } else {
        opt_dag < dag.opt - SCOPE_EPS
    };
    if !cheaper {
        return false;
    }
    let bound_pre = dag.rashomon_bound;
    dag.delta_budget.reanchored = true;
    dag.delta_budget.reanchor_opt_pre = dag.opt;
    dag.delta_budget.reanchor_bound_pre = bound_pre;
    dag.delta_budget.reanchor_tree_count_pre = dag.tree_count;
    dag.opt = opt_dag;
    dag.rashomon_bound = opt_dag * (1.0 + dag.theta);
    debug_assert!(dag.rashomon_bound <= bound_pre + SCOPE_EPS);
    let expect = tree_count_at_bound(dag, dag.rashomon_bound);
    finalize_at_bound(dag);
    debug_assert_eq!(dag.tree_count, expect);
    true
}

/// min_obj of a split from its children's min_obj values. Keyed DAGs: the
/// de-keyed integer sum (Σ cost keys) / P, the arithmetic of the keyed
/// convolve and trim, so `within` / `fits(min_obj)` agree with the key
/// tests on rounded-key DAGs; else the float sum. Any non-finite child
/// (or no child) gives +inf.
fn children_min_sum(mins: &[f64], keyed: bool, prec: f64) -> f64 {
    if mins.is_empty() || mins.iter().any(|m| !m.is_finite()) {
        return f64::INFINITY;
    }
    if keyed {
        let k: u64 = mins.iter().map(|&m| cost_key_with_precision(m, prec)).sum();
        k as f64 / prec
    } else {
        mins.iter().sum()
    }
}

/// Step 1 of `finalize_at_bound`: bottom-up in topological order, drop
/// splits with a child of min_obj = ∞ or Σ min_obj(children) > beta (no
/// tree within beta) and set min_obj = min(leaf cost, Σ min_obj(children)
/// of a kept split). `keep_placeholders`: prefix-leaf vertices keep their
/// min_obj (their lb; shared Step B between levels). Returns the order.
fn min_obj_bottom_up(dag: &mut FactoredDag, keep_placeholders: bool) -> Vec<usize> {
    let beta = dag.rashomon_bound;
    let prec = dag.cost_precision;
    let keyed = dag.keyed_costs;
    let scope_k = scope_key_with_precision(beta, prec, dag.exact_cost_keys);
    let within = |mins: &[f64]| -> bool {
        if keyed {
            let s: i64 = mins
                .iter()
                .map(|&m| cost_key_with_precision(m, prec) as i64)
                .sum();
            s <= scope_k
        } else {
            mins.iter().sum::<f64>() <= beta + SCOPE_EPS
        }
    };
    let order = topo_parents_first(&dag.subproblems, dag.root_ref);
    for &v in order.iter().rev() {
        if keep_placeholders && dag.subproblems[v].prefix_leaf.is_some() {
            continue;
        }
        let keep: Vec<(bool, f64)> = dag.subproblems[v]
            .splits
            .iter()
            .map(|split| {
                let mins: Vec<f64> = split
                    .children
                    .iter()
                    .map(|&c| dag.subproblems[c as usize].min_obj)
                    .collect();
                if mins.is_empty() || mins.iter().any(|m| !m.is_finite()) || !within(&mins) {
                    (false, f64::INFINITY)
                } else {
                    (true, children_min_sum(&mins, keyed, prec))
                }
            })
            .collect();
        let sp = &mut dag.subproblems[v];
        let mut m = sp.leaf.as_ref().map(|(c, _)| *c).unwrap_or(f64::INFINITY);
        let mut it = keep.iter();
        sp.splits
            .retain(|_| it.next().map(|k| k.0).unwrap_or(false));
        for &(kept, s) in &keep {
            if kept && s < m {
                m = s;
            }
        }
        sp.min_obj = m;
    }
    order
}

/// Recompute at beta, lossless trim, compaction, quotient and
/// children-first order (`stepb_recompute`; also the finalization of
/// keyed full-build DAGs).
fn finalize_at_bound(dag: &mut FactoredDag) {
    // Stored build-time schedules (`cost_entries`, placeholder [(lb_u, 1)]
    // entries included) describe the pre-graft / pre-trim vertex: drop them
    // so every schedule is re-derived from the splits (as on IO load).
    for sp in dag.subproblems.iter_mut() {
        sp.cost_entries = None;
    }

    // 1. min_obj bottom-up.
    let order = min_obj_bottom_up(dag, false);

    // 1b. Lossless trim.
    let order = if trim_to_bound(dag) {
        topo_parents_first(&dag.subproblems, dag.root_ref)
    } else {
        order
    };

    // 2. tree counts at beta.
    counts_at_bound(dag, &order);

    // 3. Compaction, quotient last, children-first order.
    dag.root_ref = compact_reachable(&mut dag.subproblems, dag.root_ref);
    quotient_factored_dag_impl(dag);
    dag.root_ref = renumber_children_first(&mut dag.subproblems, dag.root_ref);

    // 4. Root schedule at beta from a fresh cache (refs were remapped).
    root_schedule_at_bound(dag);
}

/// Step 2 of `finalize_at_bound`: every split's and vertex's tree_count at
/// beta (`materialize_split_schedule` from a fresh cache; vertices of
/// `order`, inline `splits`).
fn counts_at_bound(dag: &mut FactoredDag, order: &[usize]) {
    let beta = dag.rashomon_bound;
    dag.schedule_cache = Arc::new(dashmap::DashMap::new());
    dag.wide_schedule_cache = fresh_wide_cache(dag, order, true);
    let counts: Vec<(usize, Vec<u128>, u128)> = order
        .iter()
        .map(|&v| {
            let sp = &dag.subproblems[v];
            let split_tc: Vec<u128> = sp
                .splits
                .iter()
                .map(|split| {
                    let sched = materialize_split_schedule(dag, split, beta);
                    count_sum(sched.iter().map(|e| e.tree_count()))
                })
                .collect();
            let leaf_tc: u128 = match sp.leaf {
                Some((c, _)) if root_entry_within_bound(dag, c, beta) => 1,
                _ => 0,
            };
            let total = split_tc.iter().fold(leaf_tc, |a, &b| count_add(a, b));
            (v, split_tc, total)
        })
        .collect();
    for (v, split_tc, total) in counts {
        let sp = &mut dag.subproblems[v];
        for (split, tc) in sp.splits.iter_mut().zip(split_tc) {
            split.set_tree_count(tc);
        }
        sp.tree_count = total;
    }
}

/// Step 4 of `finalize_at_bound`: root schedule = materialize_schedule(
/// root, beta) from a fresh cache, filtered at beta, and tree_count. The
/// cache is then dropped (exact-key DAGs only: their
/// cache keys are exact scope keys, so a lazy refill at extraction gives
/// the same schedules; rounded-key DAGs share a 1e-6 scope bucket with the
/// first scope that filled it, so their cache is kept).
fn root_schedule_at_bound(dag: &mut FactoredDag) {
    let beta = dag.rashomon_bound;
    dag.schedule_cache = Arc::new(dashmap::DashMap::new());
    dag.wide_schedule_cache = if sched_wide_active(dag) {
        let order = topo_parents_first(&dag.subproblems, dag.root_ref);
        fresh_wide_cache(dag, &order, false)
    } else {
        Arc::new(crate::types::WideScheduleCache::default())
    };
    let root_schedule = materialize_schedule(dag, dag.root_ref, beta);
    let root: Vec<CostEntry> = root_schedule
        .iter()
        .filter(|e| root_entry_within_bound(dag, e.cost, beta))
        .cloned()
        .collect();
    dag.tree_count = count_sum(root.iter().map(|e| e.tree_count()));
    dag.root_cost_schedule = root;
    if dag.exact_cost_keys {
        dag.schedule_cache = Arc::new(dashmap::DashMap::new());
        // Wide schedule cache: drop the schedules, keep the root pass's hints
        // for the lazy refill at extraction.
        let hints = dag.wide_schedule_cache.hints.clone();
        dag.wide_schedule_cache = Arc::new(crate::types::WideScheduleCache {
            hints,
            ..Default::default()
        });
    }
}

/// Counts of a DAG loaded without them (RSMY v5, keyed DAGs): the count
/// steps of `finalize_at_bound` on the stored (already trimmed, quotiented,
/// children-first) structure at the stored beta and precision: min_obj
/// bottom-up, split / vertex tree_count, root schedule and tree_count.
pub fn recount_at_bound(dag: &mut FactoredDag) {
    let order = min_obj_bottom_up(dag, false);
    counts_at_bound(dag, &order);
    root_schedule_at_bound(dag);
}

/// Test fixture for the u128 count path (Rust and Python tests): a chain
/// DAG whose vertex `v_i` (i = 0..=levels) has a leaf option of cost
/// `leaf_cost` and, for i ≥ 1, one binary split on feature 0 with both
/// children `v_{i-1}`. Every tree is within the bound `1.0`, so the count
/// is c_0 = 1, c_i = 1 + c_{i-1}² (c_7 ≈ 4.4e22 > 2^64; c_8 > 2^128
/// saturates). Counts and the root schedule come from `recount_at_bound`;
/// `count_saturated` from `dag_has_saturated_count`. Keep
/// `levels * leaf_cost * 2^levels` well under 1.0.
#[doc(hidden)]
pub fn count_chain_fixture(levels: usize, leaf_cost: f64) -> FactoredDag {
    let mut spec_pool = crate::types::SplitSpecPool::new();
    let spec_id = spec_pool.intern(&SplitSpec {
        feature_index: 0,
        arity: 2,
        thresholds: Arc::from(vec![0.5]),
        group_map: None,
    });
    let subproblems: Vec<FactoredSubproblem> = (0..=levels)
        .map(|i| FactoredSubproblem {
            leaf: Some((
                leaf_cost,
                RLeafInfo {
                    prediction: (i % 2) as u8,
                    loss: 0.0,
                    n_samples: 1,
                },
            )),
            leaf_minority: None,
            prefix_leaf: None,
            splits: if i == 0 {
                Vec::new()
            } else {
                let c = (i - 1) as FSubRef;
                vec![FactoredSplit::new(spec_id, [c, c].into_iter().collect(), 0)]
            },
            split_ids: Vec::new(),
            tree_count: 0,
            min_obj: f64::INFINITY,
            cost_entries: None,
        })
        .collect();
    let mut dag = FactoredDag {
        subproblems,
        root_ref: levels as FSubRef,
        root_cost_schedule: Vec::new(),
        tree_count: 0,
        opt: leaf_cost,
        rashomon_bound: 1.0,
        theta: 0.0,
        reg: 0.0,
        delta_budget: DeltaBudget::default(),
        cost_precision: 1e9,
        keyed_costs: false,
        exact_cost_keys: false,
        pool_total_bytes: 0,
        pool_unique_split_count: 0,
        pool_dedup_hits: 0,
        pool_total_pushed: 0,
        schedule_intern_unique: 0,
        schedule_intern_attempts: 0,
        schedule_intern_hits: 0,
        schedule_intern_bytes: 0,
        arena_total_pushed: 0,
        arena_unique: 0,
        arena_dedup_hits: 0,
        arena_bytes: 0,
        schedule_cache: Arc::new(dashmap::DashMap::new()),
        wide_schedule_cache: Default::default(),
        split_pool: FactoredSplitPool::new(),
        spec_pool: spec_pool.into_pool(),
        inline_leaves: Vec::new(),
        inline_leaf_costs: Vec::new(),
        inline_leaves_minority: Vec::new(),
        inline_leaf_minority_costs: Vec::new(),
        n_quotient_hash_collisions: 0,
        count_saturated: false,
    };
    recount_at_bound(&mut dag);
    dag.count_saturated = dag_has_saturated_count(&dag);
    dag
}

/// True when a stored count of `dag` (root, root schedule, vertex or split)
/// sits at the `u128::MAX` clamp of `count_add` /
/// `count_mul`. A clamp propagates upward (Σ / Π with u128::MAX stay at
/// u128::MAX), so a saturated count that reaches the stored DAG shows here;
/// one that was filtered out by a parent's scope left the stored counts
/// exact. `count_saturated` on load (the flag is not serialized), and the
/// fit's flag when the global saturation-event counter grew.
pub fn dag_has_saturated_count(dag: &FactoredDag) -> bool {
    const M: u128 = u128::MAX;
    dag.tree_count == M
        || dag.root_cost_schedule.iter().any(|e| e.tree_count() == M)
        || dag.split_pool.entries.iter().any(|s| s.tree_count() == M)
        || dag
            .subproblems
            .iter()
            .any(|sp| sp.tree_count == M || sp.splits.iter().any(|s| s.tree_count() == M))
}

/// Lossless trim of a keyed DAG at beta (compaction): keep exactly the
/// vertices, splits and leaf options that lie on some root tree with cost
/// <= beta. In integer key space (a tree's key is the sum of its leaf
/// keys; the counted set is {T : key(T) <= scope_key(beta)}):
///   mk(v) = min key of a tree at v (bottom-up),
///   pk(v) = min over contexts of the key of the rest of a root tree
///           through v (pk(root) = 0, pk(c_i) = min pk(v) + Σ_{j≠i} mk(c_j)),
/// and v is kept iff pk(v) + mk(v) <= scope_key, a split iff
/// pk(v) + Σ mk(children) <= scope_key, a leaf iff pk(v) + key(leaf) <=
/// scope_key. Every tree within beta passes these tests at each of its
/// vertices, so no counted tree (and no tree at a vertex at a cost some
/// counted tree uses) is removed; every kept item lies on its cheapest
/// completion, which is within beta. Dropped vertices are emptied (and
/// become unreachable: a kept split's children are kept); min_obj is
/// recomputed over what is left. Float-cost DAGs are left unchanged (no
/// exact characterization of the counted set). The "materialized schedule
/// has an entry within the context scope" test is implied: with exact keys
/// a kept split's cheapest completion fits its widest context. Returns
/// whether anything was dropped.
fn trim_to_bound(dag: &mut FactoredDag) -> bool {
    if !dag.keyed_costs {
        return false;
    }
    const INF: i64 = i64::MAX / 4;
    let prec = dag.cost_precision;
    let key = |c: f64| -> i64 { cost_key_with_precision(c, prec) as i64 };
    let scope_k = scope_key_with_precision(dag.rashomon_bound, prec, dag.exact_cost_keys);
    let n = dag.subproblems.len();
    let order = topo_parents_first(&dag.subproblems, dag.root_ref);
    let split_mk = |mk: &[i64], split: &FactoredSplit| -> i64 {
        let mut s = 0i64;
        for &c in &split.children {
            let m = mk[c as usize];
            if m >= INF {
                return INF;
            }
            s += m;
        }
        if split.children.is_empty() {
            INF
        } else {
            s
        }
    };
    let mut mk = vec![INF; n];
    for &v in order.iter().rev() {
        let sp = &dag.subproblems[v];
        let mut m = sp.leaf.as_ref().map(|(c, _)| key(*c)).unwrap_or(INF);
        for split in &sp.splits {
            m = m.min(split_mk(&mk, split));
        }
        mk[v] = m;
    }
    let mut pk = vec![INF; n];
    if (dag.root_ref as usize) < n {
        pk[dag.root_ref as usize] = 0;
    }
    for &v in &order {
        let pv = pk[v];
        if pv >= INF {
            continue;
        }
        for split in &dag.subproblems[v].splits {
            let total = split_mk(&mk, split);
            if total >= INF {
                continue;
            }
            for &c in &split.children {
                let cand = pv + total - mk[c as usize];
                if cand < pk[c as usize] {
                    pk[c as usize] = cand;
                }
            }
        }
    }
    let mut changed = false;
    for v in 0..n {
        let pv = pk[v];
        if pv >= INF || mk[v] >= INF || pv + mk[v] > scope_k {
            let sp = &dag.subproblems[v];
            if sp.leaf.is_some() || !sp.splits.is_empty() || sp.prefix_leaf.is_some() {
                dag.subproblems[v] = empty_subproblem();
                changed = true;
            }
            continue;
        }
        let keep: Vec<bool> = dag.subproblems[v]
            .splits
            .iter()
            .map(|split| {
                let t = split_mk(&mk, split);
                t < INF && pv + t <= scope_k
            })
            .collect();
        let sp = &mut dag.subproblems[v];
        if keep.iter().any(|k| !k) {
            let mut it = keep.iter();
            sp.splits.retain(|_| *it.next().unwrap());
            changed = true;
        }
        if let Some((c, _)) = sp.leaf {
            if pv + key(c) > scope_k {
                sp.leaf = None;
                changed = true;
            }
        }
    }
    if changed {
        // min_obj over the kept options (children before parents).
        let order = topo_parents_first(&dag.subproblems, dag.root_ref);
        for &v in order.iter().rev() {
            let mins: Vec<f64> = dag.subproblems[v]
                .splits
                .iter()
                .map(|split| {
                    let ms: Vec<f64> = split
                        .children
                        .iter()
                        .map(|&c| dag.subproblems[c as usize].min_obj)
                        .collect();
                    // Keyed DAG (checked above): de-keyed integer sum.
                    children_min_sum(&ms, true, prec)
                })
                .collect();
            let sp = &mut dag.subproblems[v];
            let mut m = sp.leaf.as_ref().map(|(c, _)| *c).unwrap_or(f64::INFINITY);
            for s in mins {
                if s < m {
                    m = s;
                }
            }
            sp.min_obj = m;
        }
    }
    changed
}

/// Renumber the arena children-first: post-order from the root, then from
/// every other vertex in index order (unreachable ones kept when
/// Compaction is off). Child refs are remapped; returns the new root.
fn renumber_children_first(subproblems: &mut Vec<FactoredSubproblem>, root: FSubRef) -> FSubRef {
    let n = subproblems.len();
    if n == 0 {
        return root;
    }
    let mut post: Vec<usize> = Vec::with_capacity(n);
    let mut state = vec![0u8; n];
    let starts = std::iter::once(root as usize)
        .filter(|&r| r < n)
        .chain(0..n);
    for start in starts {
        if state[start] != 0 {
            continue;
        }
        let mut stack: Vec<(usize, bool)> = vec![(start, false)];
        while let Some((i, children_done)) = stack.pop() {
            if children_done {
                post.push(i);
                continue;
            }
            if state[i] != 0 {
                continue;
            }
            state[i] = 1;
            stack.push((i, true));
            for split in subproblems[i].splits.iter().rev() {
                for &c in split.children.iter().rev() {
                    let c = c as usize;
                    if c < n && state[c] == 0 {
                        stack.push((c, false));
                    }
                }
            }
        }
    }
    if post.iter().enumerate().all(|(new, &old)| new == old) {
        return root;
    }
    let mut remap: Vec<FSubRef> = vec![FSubRef::MAX; n];
    for (new, &old) in post.iter().enumerate() {
        remap[old] = new as FSubRef;
    }
    let mut old: Vec<Option<FactoredSubproblem>> =
        std::mem::take(subproblems).into_iter().map(Some).collect();
    subproblems.reserve_exact(n);
    for &o in &post {
        let mut sp = old[o].take().expect("renumber: vertex visited twice");
        for split in &mut sp.splits {
            for c in split.children.iter_mut() {
                *c = remap[*c as usize];
            }
        }
        subproblems.push(sp);
    }
    if (root as usize) < n {
        remap[root as usize]
    } else {
        root
    }
}

/// Largest prefix cost over the contexts (parent, split, position) of each
/// vertex whose cheapest completion fits beta: pc(v) + overhead(v) +
/// Σ_{j≠i} min_obj(c_j) with pc(c_i) + min_obj(c_i) <= beta. A placeholder
/// with max > pc has contexts with unequal scopes. NEG_INFINITY when none.
fn stepb_context_pc_max(
    dag: &FactoredDag,
    pc: &[f64],
    overhead: &dyn Fn(usize) -> f64,
) -> Vec<f64> {
    let n = dag.subproblems.len();
    let mut pmax = vec![f64::NEG_INFINITY; n];
    let beta = dag.rashomon_bound;
    for v in topo_parents_first(&dag.subproblems, dag.root_ref) {
        let pv = pc[v];
        if !pv.is_finite() {
            continue;
        }
        let ov = overhead(v);
        for split in &dag.subproblems[v].splits {
            let mins: Vec<f64> = split
                .children
                .iter()
                .map(|&c| dag.subproblems[c as usize].min_obj)
                .collect();
            if mins.iter().any(|m| !m.is_finite()) {
                continue;
            }
            let total: f64 = mins.iter().sum();
            for (i, &c) in split.children.iter().enumerate() {
                let cand = pv + ov + total - mins[i];
                if cand + mins[i] <= beta + SCOPE_EPS && cand > pmax[c as usize] {
                    pmax[c as usize] = cand;
                }
            }
        }
    }
    pmax
}

/// One Step-B group: the reachable placeholders `vs` (ascending arena
/// index) with one support, solved once at the widest scope `beta_u` of
/// their contexts; `pc_lo` / `pc_hi` = smallest / largest prefix cost.
struct StepbGroup {
    support: crate::types::SupportMask,
    beta_u: f64,
    vs: Vec<usize>,
    pc_lo: f64,
    pc_hi: f64,
}

/// The placeholders of one Step-B level of `dag` (prefix-cost DP, beta_u,
/// grouping by full SupportMask equality, cheapest-first order).
struct StepbLevel {
    groups: Vec<StepbGroup>,
    /// Reachable placeholders with no tree within beta (beta_u < lb_u).
    empty: Vec<usize>,
    /// Reachable placeholders (grouped + empty).
    n_prefix: usize,
    /// Groups whose contexts give unequal scopes.
    n_unequal: usize,
    time_pc: std::time::Duration,
}

/// Prefix-cost DP → beta_u per reachable placeholder → groups by support
/// (max beta_u), cheapest first
/// (`stepb_order_groups`).
fn stepb_level(dag: &FactoredDag, config: &PrefixLeafFillConfig) -> StepbLevel {
    use crate::types::SupportMask;

    let base = &config.leaf_base_config;
    let split_reg = (1.0 - base.reg_alpha) * base.reg;
    let path_penalty = base.path_penalty;
    let n_total = config.n_data.max(1) as f64;

    // Vertex sample counts, only needed for the path cost.
    let n = dag.subproblems.len();
    let mut vertex_n = vec![0.0f64; n];
    if path_penalty != 0.0 {
        for &v in topo_parents_first(&dag.subproblems, dag.root_ref)
            .iter()
            .rev()
        {
            let sp = &dag.subproblems[v];
            vertex_n[v] = if let Some((_, li)) = &sp.leaf {
                li.n_samples as f64
            } else if let Some(pl) = &sp.prefix_leaf {
                pl.n_samples as f64
            } else if let Some(s) = sp.splits.first() {
                s.children.iter().map(|&c| vertex_n[c as usize]).sum()
            } else {
                0.0
            };
        }
    }
    let overhead = |v: usize| split_reg + path_penalty * vertex_n[v] / n_total;
    let t_pc = std::time::Instant::now();
    let pc = stepb_prefix_costs(dag, &overhead);
    let pc_max = stepb_context_pc_max(dag, &pc, &overhead);
    let time_pc = t_pc.elapsed();

    // Placeholders reachable from the root, grouped by support.
    let mut n_prefix = 0usize;
    let mut empty: Vec<usize> = Vec::new();
    let mut group_of: HashMap<SupportMask, usize> = HashMap::new();
    let mut groups: Vec<StepbGroup> = Vec::new();
    for v in 0..n {
        let support = match &dag.subproblems[v].prefix_leaf {
            Some(pl) => pl.support.clone(),
            None => continue,
        };
        if !pc[v].is_finite() {
            continue; // not reachable through recorded splits
        }
        n_prefix += 1;
        match stepb_scope(dag, &pc, v) {
            None => empty.push(v),
            Some(beta_u) => {
                let g = *group_of.entry(support.clone()).or_insert_with(|| {
                    groups.push(StepbGroup {
                        support,
                        beta_u,
                        vs: Vec::new(),
                        pc_lo: f64::INFINITY,
                        pc_hi: f64::NEG_INFINITY,
                    });
                    groups.len() - 1
                });
                let gr = &mut groups[g];
                if beta_u > gr.beta_u {
                    gr.beta_u = beta_u;
                }
                gr.vs.push(v);
                gr.pc_lo = gr.pc_lo.min(pc[v]);
                gr.pc_hi = gr.pc_hi.max(pc[v]).max(pc_max[v]);
            }
        }
    }
    let n_unequal = groups
        .iter()
        .filter(|g| g.pc_hi > g.pc_lo + SCOPE_EPS)
        .count();
    stepb_order_groups(dag, &mut groups);
    StepbLevel {
        groups,
        empty,
        n_prefix,
        n_unequal,
        time_pc,
    }
}

/// Step-B group order: cheapest first — ascending pc_lo + lb (`min_obj`) of
/// the group's first placeholder, ties by that placeholder's arena index
/// (deterministic). Only which groups a time-out leaves unsolved depends
/// on the order.
fn stepb_order_groups(dag: &FactoredDag, groups: &mut [StepbGroup]) {
    let key = |g: &StepbGroup| g.pc_lo + dag.subproblems[g.vs[0]].min_obj;
    groups.sort_by(|a, b| {
        key(a)
            .partial_cmp(&key(b))
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.vs[0].cmp(&b.vs[0]))
    });
}

/// Completion search (Step B) on a Step-A DAG (`build_factored_dag_impl(.., stepb=true)`,
/// placeholders at their lower bounds, not yet compacted or quotiented),
/// one shared solver per depth level:
///   1. `stepb_level` over the current DAG (pc DP, beta_u, groups, order);
///   2. one solver with the MiniRESPLINTER config (`stepb_mini_resplinter`;
///      `rashomon_bound` = max beta_u, Select ub = outer beta, seeded with
///      the Step-A / previous level's completion-lb caches) explores every
///      group's support as a root at its beta_u (`solve_seeded`): a node
///      shared by several groups is explored once at the widest scope any of
///      them requests, so it records a superset of each group's splits;
///   3. one multi-root build (`build_dag_multi`, each root at its beta_u,
///      placeholders when depth remains) and one graft (`stepb_graft_multi`);
///   4. while depth remains: refresh min_obj (`stepb_refresh_min_obj`) and
///      repeat on the merged DAG's new placeholders. The merged pc gives
///      beta − pc(w) = max over the groups' nested scopes, which the
///      per-group recursion takes via max-beta_u grouping;
///   5. `stepb_recompute` at beta (exact filter, compaction, quotient).
/// Afterwards the DAG holds exactly the trees with cost <= beta over the
/// retained pool (a lower bound when any placeholder timed out), with no
/// nested DAGs or prefix leaves left. Counts are those of the first level;
/// `n_leaves_timed_out` = every first-level grouped placeholder when any
/// level's solve timed out or did not start.
fn stepb_fill(
    dag: &mut FactoredDag,
    config: &PrefixLeafFillConfig,
    d_l: i32,
    remaining: i32,
    theta: f64,
    cost_precision: f64,
) {
    let beta = dag.rashomon_bound;
    let mut level_remaining = remaining;
    let mut first = true;
    let (mut n_prefix, mut n_unequal, mut n_grouped_top) = (0usize, 0usize, 0usize);
    let (mut n_filled, mut n_empty) = (0usize, 0usize);
    let mut any_timeout = false;
    let mut carry: Option<crate::solver::RashLbCache> = None;
    let mut profile = crate::types::SolverProfile::default();
    let mut n_explored = 0usize;
    let (mut t_solve, mut t_build, mut t_graft) = (
        std::time::Duration::ZERO,
        std::time::Duration::ZERO,
        std::time::Duration::ZERO,
    );
    let mut time_pc = std::time::Duration::ZERO;
    let mut n_groups_total = 0usize;
    while level_remaining > 0 {
        let lvl = stepb_level(dag, config);
        time_pc += lvl.time_pc;
        let grouped: usize = lvl.groups.iter().map(|g| g.vs.len()).sum();
        if first {
            n_prefix = lvl.n_prefix;
            n_unequal = lvl.n_unequal;
            n_grouped_top = grouped;
            n_empty = lvl.empty.len();
        }
        n_groups_total += lvl.groups.len();
        if config.verbose {
            eprintln!(
                "  Step B (shared): {} placeholders ({} beyond scope), {} unique supports \
                 ({} with unequal scopes), depth {} (d_l={})",
                lvl.n_prefix,
                lvl.empty.len(),
                lvl.groups.len(),
                lvl.n_unequal,
                level_remaining,
                d_l,
            );
        }
        if lvl.groups.is_empty() {
            stepb_graft_finish(dag, &HashMap::new(), &lvl.empty);
            break;
        }
        let time_limit = match config.deadline {
            Some(d) => {
                let left = d.saturating_duration_since(std::time::Instant::now());
                if left.is_zero() {
                    // Not started: every group is emptied and timed out.
                    let mut e = lvl.empty.clone();
                    for g in &lvl.groups {
                        e.extend_from_slice(&g.vs);
                    }
                    stepb_graft_finish(dag, &HashMap::new(), &e);
                    any_timeout = true;
                    break;
                }
                Some(left.as_secs_f64())
            }
            None => None,
        };
        let solve_depth = level_remaining.min(d_l).max(1);
        let n_batches = config
            .leaf_base_config
            .rash_tuning
            .stepb_batches
            .min(lvl.groups.len());
        if n_batches > 0 {
            let out = stepb_level_batched(
                dag,
                config,
                &lvl,
                n_batches,
                level_remaining,
                solve_depth,
                beta,
                theta,
                cost_precision,
                &mut carry,
            );
            profile.merge_from(&out.profile);
            n_explored += out.n_explored;
            any_timeout |= out.timed_out;
            t_solve += out.t_solve;
            t_build += out.t_build;
            t_graft += out.t_graft;
            if first {
                n_empty += out.n_empty;
                n_filled = out.n_filled;
            }
            first = false;
            level_remaining -= solve_depth;
            if level_remaining > 0 {
                stepb_refresh_min_obj(dag);
            }
            continue;
        }
        let mut solver = stepb_shared_solver(
            config,
            &lvl.groups,
            level_remaining,
            solve_depth,
            time_limit,
        );
        // The previous level's caches, else Step A's (moved, once).
        let cache = carry.take().or_else(|| {
            config
                .rash_lb_cache
                .as_ref()
                .and_then(|m| m.lock().ok().and_then(|mut g| g.take()))
        });
        if let Some(c) = cache {
            solver.seed_rash_lb_cache(c);
        }
        let seeds: Vec<(crate::types::SupportMask, f64)> = lvl
            .groups
            .iter()
            .map(|g| (g.support.clone(), g.beta_u))
            .collect();
        let t0 = std::time::Instant::now();
        let (result, keys) = solver.solve_seeded(&seeds);
        t_solve += t0.elapsed();
        drop(seeds);
        profile.merge_from(&solver.worker.profile);
        n_explored += result.n_explored;
        if result.timed_out {
            any_timeout = true;
        }
        // No level reads the caches after the last one, so free them
        // before the snapshot / DAG build / graft / recompute.
        let last = level_remaining - solve_depth <= 0;
        if last {
            drop(solver.take_rash_lb_cache());
            carry = None;
        } else {
            carry = Some(solver.take_rash_lb_cache());
        }
        let mut snap = solver.into_search_snapshot(
            result.objective,
            theta,
            DeltaBudget::default(),
            cost_precision,
            level_remaining,
        );
        snap.rashomon_bound = beta;
        let roots: Vec<(MemoKey, f64)> = keys
            .into_iter()
            .zip(lvl.groups.iter().map(|g| g.beta_u))
            .collect();
        let t0 = std::time::Instant::now();
        let (nested, root_counts) = build_dag_multi(snap, &roots);
        t_build += t0.elapsed();
        drop(roots);
        let grafts: Vec<(&[usize], FSubRef, u128)> = lvl
            .groups
            .iter()
            .zip(root_counts)
            .map(|(g, (r, c))| (g.vs.as_slice(), r, c))
            .collect();
        if first {
            n_empty += grafts
                .iter()
                .filter(|g| g.2 == 0)
                .map(|g| g.0.len())
                .sum::<usize>();
        }
        let t0 = std::time::Instant::now();
        let nf = stepb_graft_multi(dag, nested, &grafts, &lvl.empty);
        t_graft += t0.elapsed();
        if first {
            n_filled = nf;
        }
        first = false;
        level_remaining -= solve_depth;
        if level_remaining > 0 {
            stepb_refresh_min_obj(dag);
        }
    }
    let tc_pre = dag.tree_count;
    let t0 = std::time::Instant::now();
    stepb_recompute(dag);
    let time_recompute = t0.elapsed();
    let n_timed_out = if any_timeout { n_grouped_top } else { 0 };

    dag.delta_budget.n_prefix_leaves = n_prefix as u32;
    dag.delta_budget.n_leaves_filled = n_filled as u32;
    dag.delta_budget.n_leaves_empty = n_empty as u32;
    dag.delta_budget.n_leaves_timed_out = n_timed_out as u32;
    dag.delta_budget.n_stepb_groups_unequal_scope = n_unequal as u32;
    dag.delta_budget.time_candidate_gen_stepb_s = profile.time_candidate_gen.as_secs_f64();
    dag.delta_budget.time_candidate_gen_lb_stepb_s = profile.time_candidate_gen_lb.as_secs_f64();
    dag.delta_budget.lb_stats_stepb = profile.lb_stats();
    dag.delta_budget.stepb_shared = true;
    if config.verbose {
        eprintln!(
            "  Step B done (shared): filled={} empty={} timed_out={}; tree_count {} (prefix, lb-priced) -> {}",
            n_filled, n_empty, n_timed_out, tc_pre, dag.tree_count,
        );
        eprintln!(
            "  [stepb-prof] shared depth={} pc_dp={:.3}s groups={} solve={:.3}s build={:.3}s \
             graft={:.3}s recompute={:.3}s n_explored={} time_candidate_gen={:.3}s \
             time_candidate_gen_lb={:.3}s",
            remaining,
            time_pc.as_secs_f64(),
            n_groups_total,
            t_solve.as_secs_f64(),
            t_build.as_secs_f64(),
            t_graft.as_secs_f64(),
            time_recompute.as_secs_f64(),
            n_explored,
            profile.time_candidate_gen.as_secs_f64(),
            profile.time_candidate_gen_lb.as_secs_f64(),
        );
    }
}

/// One shared Step-B solver (`stepb_fill_shared`) for `groups` at this
/// level: the MiniRESPLINTER config at depth `solve_depth`,
/// `rashomon_bound` = max beta_u over `groups`, rooted at the first
/// group's support, Select_B(u) at the outer beta. Not yet seeded.
fn stepb_shared_solver(
    config: &PrefixLeafFillConfig,
    groups: &[StepbGroup],
    level_remaining: i32,
    solve_depth: i32,
    time_limit: Option<f64>,
) -> crate::solver::BranchAndBoundSolver {
    let beta_max = groups
        .iter()
        .map(|g| g.beta_u)
        .fold(f64::NEG_INFINITY, f64::max);
    let mut cfg = config.leaf_base_config.clone();
    cfg.max_depth = solve_depth;
    cfg.rashomon_mode = true;
    cfg.rashomon_bound = beta_max;
    cfg.step_b_remaining = level_remaining - solve_depth;
    cfg.warm_start_ub = None;
    cfg.n_total_override = Some(config.n_data);
    cfg.weight_total = config.n_data as f64;
    cfg.n_workers = 1;
    cfg.closure_depth = None;
    cfg.closure_shared_memo = None;
    cfg.closure_parallel = false;
    cfg.initial_used_features = crate::types::FeatureMask::all(cfg.n_features);
    cfg.time_limit = time_limit;
    let solver = crate::solver::BranchAndBoundSolver::new(
        config.store.clone(),
        cfg,
        None,
        Some(groups[0].support.clone()),
    );
    // Select_B(u) at the outer beta, as `stepb_mini_resplinter`.
    solver.set_rashomon_select_ub(config.leaf_base_config.rashomon_bound);
    solver
}

/// Totals of one batched shared Step-B level (`stepb_level_batched`):
/// solve / build times summed over batches, `n_empty` = placeholders of
/// groups with no tree, `n_filled` = placeholders grafted.
#[derive(Default)]
struct StepbBatchedOut {
    profile: crate::types::SolverProfile,
    n_explored: usize,
    timed_out: bool,
    t_solve: std::time::Duration,
    t_build: std::time::Duration,
    t_graft: std::time::Duration,
    n_empty: usize,
    n_filled: usize,
}

/// `stepb_batches`: one level of `stepb_fill_shared` with `lvl.groups`
/// split into `n_batches` contiguous batches (batch i =
/// groups[i·G/n .. (i+1)·G/n], independent of the thread count), each
/// solved in parallel as the unbatched level is (own solver, `solve_seeded`,
/// snapshot, `build_dag_multi`, the snapshot dropped once built), then one
/// graft of every batch's nested DAG in batch order (`stepb_graft_parts`).
/// The carried lb caches are not copied: they become one read-only level
/// base shared by every batch solver (`RashLevelBase`,
/// `attach_rash_level_base`), each solver writing only its own delta; at
/// the level end the deltas are merged into the base in batch order
/// (`rash_lb_delta_merge`) giving `carry`, unless this is the last level,
/// which drops the caches (base and deltas dropped, no merge). Same tree set: each root is built at its own beta_u and the lb
/// caches hold exact facts (Min / Above), so they do not depend on which
/// solver explored a node. Each batch takes its time limit from
/// `config.deadline` when it starts; a batch starting past it is not
/// solved (its groups are emptied, the level timed out).
#[allow(clippy::too_many_arguments)]
fn stepb_level_batched(
    dag: &mut FactoredDag,
    config: &PrefixLeafFillConfig,
    lvl: &StepbLevel,
    n_batches: usize,
    level_remaining: i32,
    solve_depth: i32,
    beta: f64,
    theta: f64,
    cost_precision: f64,
    carry: &mut Option<crate::solver::RashLbCache>,
) -> StepbBatchedOut {
    use rayon::prelude::*;
    let n_groups = lvl.groups.len();
    let batches: Vec<&[StepbGroup]> = (0..n_batches)
        .map(|i| &lvl.groups[i * n_groups / n_batches..(i + 1) * n_groups / n_batches])
        .filter(|b| !b.is_empty())
        .collect();
    // The previous level's caches, else Step A's (moved, once), frozen.
    let level = carry
        .take()
        .or_else(|| {
            config
                .rash_lb_cache
                .as_ref()
                .and_then(|m| m.lock().ok().and_then(|mut g| g.take()))
        })
        .and_then(crate::solver::RashLevelBase::from_cache)
        .map(Arc::new);
    let base_len = level.as_ref().map_or(0, |l| l.pid_len());
    let keep_cache = level_remaining - solve_depth > 0;
    let lb_threads = config
        .leaf_base_config
        .rash_tuning
        .batch_lb_threads(batches.len());
    STEPB_BATCH_LEVELS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    STEPB_BATCH_SOLVES.fetch_add(batches.len() as u64, std::sync::atomic::Ordering::Relaxed);
    let t_wall = std::time::Instant::now();
    let solved: Vec<Option<_>> = batches
        .par_iter()
        .map(|groups| {
            // The time left at this batch's start (None: past the deadline).
            let time_limit = match config.deadline {
                Some(d) => {
                    let left = d.saturating_duration_since(std::time::Instant::now());
                    if left.is_zero() {
                        return None;
                    }
                    Some(left.as_secs_f64())
                }
                None => None,
            };
            let mut solver =
                stepb_shared_solver(config, groups, level_remaining, solve_depth, time_limit);
            solver.worker.rash_lb_threads = lb_threads;
            let attached = level
                .as_ref()
                .map_or(false, |l| solver.attach_rash_level_base(l));
            let seeds: Vec<(crate::types::SupportMask, f64)> = groups
                .iter()
                .map(|g| (g.support.clone(), g.beta_u))
                .collect();
            let t0 = std::time::Instant::now();
            let (result, keys) = solver.solve_seeded(&seeds);
            let t_solve = t0.elapsed();
            drop(seeds);
            let profile = solver.worker.profile.clone();
            // The delta (own ids from the base's length up when attached).
            // Offset from whether the base is still installed (a ub change drops it).
            let still_attached = attached && solver.worker.rash_level_base.is_some();
            let cache = if keep_cache {
                Some((
                    solver.take_rash_lb_cache(),
                    if still_attached { base_len } else { 0 },
                ))
            } else {
                drop(solver.take_rash_lb_cache());
                None
            };
            solver.worker.rash_level_base = None;
            let mut snap = solver.into_search_snapshot(
                result.objective,
                theta,
                DeltaBudget::default(),
                cost_precision,
                level_remaining,
            );
            snap.rashomon_bound = beta;
            let roots: Vec<(MemoKey, f64)> = keys
                .into_iter()
                .zip(groups.iter().map(|g| g.beta_u))
                .collect();
            let t0 = std::time::Instant::now();
            let (nested, root_counts) = build_dag_multi(snap, &roots);
            let t_build = t0.elapsed();
            Some((
                nested,
                root_counts,
                cache,
                profile,
                result.n_explored,
                result.timed_out,
                t_solve,
                t_build,
            ))
        })
        .collect();
    let wall = t_wall.elapsed();
    // Every batch solver is dropped: the base is this function's alone.
    let mut merged: Option<crate::solver::RashLbCache> = match level {
        Some(l) if keep_cache => Some(
            Arc::try_unwrap(l)
                .unwrap_or_else(|_| panic!("stepb_batches: level base still shared"))
                .into_cache(),
        ),
        _ => None,
    };
    let mut out = StepbBatchedOut::default();
    let mut parts = Vec::with_capacity(solved.len());
    let mut emptied: Vec<usize> = lvl.empty.clone();
    for (groups, s) in batches.iter().zip(solved) {
        let Some((nested, root_counts, cache, profile, n_explored, timed_out, ts, tb)) = s else {
            // Not started (past the deadline): every group emptied.
            out.timed_out = true;
            for g in groups.iter() {
                emptied.extend_from_slice(&g.vs);
                out.n_empty += g.vs.len();
            }
            continue;
        };
        out.profile.merge_from(&profile);
        out.n_explored += n_explored;
        out.timed_out |= timed_out;
        out.t_solve += ts;
        out.t_build += tb;
        if let Some((c, bl)) = cache {
            match merged.as_mut() {
                None if bl == 0 => merged = Some(c),
                None => {}
                Some(m) => crate::solver::rash_lb_delta_merge(
                    m,
                    c,
                    bl,
                    &config.leaf_base_config.rash_tuning,
                ),
            }
        }
        let grafts: Vec<(&[usize], FSubRef, u128)> = groups
            .iter()
            .zip(root_counts)
            .map(|(g, (r, c))| (g.vs.as_slice(), r, c))
            .collect();
        out.n_empty += grafts
            .iter()
            .filter(|g| g.2 == 0)
            .map(|g| g.0.len())
            .sum::<usize>();
        parts.push((nested, grafts));
    }
    *carry = merged.filter(|c| c.key.is_some());
    let t0 = std::time::Instant::now();
    out.n_filled = stepb_graft_parts(dag, parts, &emptied);
    out.t_graft = t0.elapsed();
    if config.verbose {
        eprintln!(
            "  [stepb-prof] batched depth={} batches={} groups={} solve+build wall={:.3}s lb_threads={}",
            level_remaining,
            batches.len(),
            n_groups,
            wall.as_secs_f64(),
            lb_threads,
        );
    }
    out
}

/// AttachRecursive for one shared Step-B level (shared Step B): the
/// placeholders of every group in `grafts` (placeholders, nested root,
/// root tree count within its beta_u) point at `offset + root` in ONE
/// appended copy of `nested` (specs re-interned once); groups with no tree,
/// `empty`, and every other placeholder of the current arena are emptied
/// (`stepb_graft_finish`, run before the append so the nested DAG's own
/// placeholders — the next level — are kept). Counts are not updated.
/// Returns the number of placeholders grafted.
fn stepb_graft_multi(
    dag: &mut FactoredDag,
    nested: FactoredDag,
    grafts: &[(&[usize], FSubRef, u128)],
    empty: &[usize],
) -> usize {
    stepb_graft_parts(dag, vec![(nested, grafts.to_vec())], empty)
}

/// `stepb_graft_multi` over several nested DAGs (`stepb_batches`: one
/// per batch): one `stepb_graft_finish` with every group redirected to
/// its root in its part, the parts appended in order at cumulative
/// offsets. One part is `stepb_graft_multi`.
fn stepb_graft_parts(
    dag: &mut FactoredDag,
    parts: Vec<(FactoredDag, Vec<(&[usize], FSubRef, u128)>)>,
    empty: &[usize],
) -> usize {
    let mut offset = dag.subproblems.len() as FSubRef;
    let mut offsets: Vec<FSubRef> = Vec::with_capacity(parts.len());
    let mut redirect: HashMap<FSubRef, FSubRef> = HashMap::new();
    let mut emptied: Vec<usize> = empty.to_vec();
    let mut n_filled = 0usize;
    for (nested, grafts) in &parts {
        debug_assert!(
            nested.split_pool.entries.is_empty() && nested.inline_leaves.is_empty(),
            "stepb_graft_multi: nested DAG must be unpromoted, without inline leaves"
        );
        dag.n_quotient_hash_collisions += nested.n_quotient_hash_collisions;
        let n_nested = nested.subproblems.len() as FSubRef;
        for &(vs, root, count) in grafts {
            if count == 0 || root >= n_nested {
                emptied.extend_from_slice(vs);
                continue;
            }
            for &v in vs {
                redirect.insert(v as FSubRef, offset + root);
            }
            n_filled += vs.len();
        }
        offsets.push(offset);
        offset += n_nested;
    }
    stepb_graft_finish(dag, &redirect, &emptied);
    let mut specs = stepb_spec_interner(dag);
    for ((nested, grafts), offset) in parts.into_iter().zip(offsets) {
        let n_nested = nested.subproblems.len() as FSubRef;
        let spec_map: Vec<crate::types::SpecId> = nested
            .spec_pool
            .iter()
            .map(|spec| specs.intern(spec))
            .collect();
        dag.subproblems.reserve(nested.subproblems.len());
        for mut sp in nested.subproblems {
            for split in &mut sp.splits {
                split.spec_id = spec_map[split.spec_id as usize];
                for c in split.children.iter_mut() {
                    *c += offset;
                }
            }
            dag.subproblems.push(sp);
        }
        for &(_, root, count) in &grafts {
            if count > 0 && root < n_nested {
                dag.subproblems[(offset + root) as usize].prefix_leaf = None;
            }
        }
    }
    dag.spec_pool = specs.into_pool();
    n_filled
}

/// Between shared Step-B levels: min_obj bottom-up over the merged DAG
/// (`min_obj_bottom_up`) with every placeholder kept at its lb, so the
/// next level's pc uses the grafted subtrees' minima.
fn stepb_refresh_min_obj(dag: &mut FactoredDag) {
    min_obj_bottom_up(dag, true);
}

/// Multi-root build of one shared Step-B solve (shared Step B): every
/// root (seed key, beta_u) through one `build_factored_recursive` cache,
/// depth-0 vertices as placeholders when depth remains below the solve
/// (`total_max_depth > solve_depth`). No compaction or `finalize_at_bound`
/// (the outer `stepb_recompute` trims at beta). Returns the DAG
/// (unpromoted, `root_ref` = first root) and per root its ref and tree
/// count within its scope.
fn build_dag_multi(
    snap: SearchSnapshot,
    roots: &[(MemoKey, f64)],
) -> (FactoredDag, Vec<(FSubRef, u128)>) {
    let SearchSnapshot {
        memo,
        pool,
        arena,
        root_key: _,
        rashomon_bound,
        opt,
        theta,
        reg,
        delta_budget,
        cost_precision,
        cost_lattice,
        solve_depth,
        total_max_depth,
        prefix_leaf_fill: _,
    } = snap;
    let stepb = total_max_depth > solve_depth;
    let out = build_factored_dag_roots(
        &memo,
        &pool,
        &arena,
        roots,
        rashomon_bound,
        opt,
        theta,
        reg,
        delta_budget,
        cost_precision,
        cost_lattice,
        solve_depth,
        total_max_depth,
        stepb,
        false,
    );
    drop(memo);
    drop(pool);
    drop(arena);
    out
}

// ---------------------------------------------------------------------------
// Factored DAG scope and compatibility helpers
// ---------------------------------------------------------------------------

pub(crate) const SCOPE_EPS: f64 = 1e-9;

// Scope keys for legacy floating-point DAG readers.
fn quantize_scope(scope: f64) -> u64 {
    ((scope * 1e6).round() as i64).max(0) as u64
}

/// Wide schedule cache applies to `dag` (keyed exact-key DAGs only).
pub(crate) fn sched_wide_active(dag: &FactoredDag) -> bool {
    dag.keyed_costs && dag.exact_cost_keys
}
static STEPB_BATCH_LEVELS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static STEPB_BATCH_SOLVES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// `stepb_batches` counters, cumulative over the process: (batched
/// Step-B levels, batch solvers started or skipped).
pub fn rash_stepb_batch_stats() -> (u64, u64) {
    (
        STEPB_BATCH_LEVELS.load(std::sync::atomic::Ordering::Relaxed),
        STEPB_BATCH_SOLVES.load(std::sync::atomic::Ordering::Relaxed),
    )
}

/// Integer key of a scope ceiling: an entry fits `scope` iff
/// `cost_key(cost) <= scope_key(scope)`. Rounded like `cost_key` (round on
/// both sides), so the only tolerance is the absolute SCOPE_EPS: at
/// precision 1 a bound of 315 admits 315 but not 316 (`ceil` admitted 316).
///
/// `exact` (keys on the cost lattice, `FactoredDag::exact_cost_keys`):
/// key `K` stands for the cost `K / precision` exactly, so the ceiling is
/// `floor((scope + SCOPE_EPS) * precision)` taken on the exact rational
/// value of the f64 `scope + SCOPE_EPS` — `K` fits iff
/// `K / precision <= scope + SCOPE_EPS`, with no rounding slack.
pub(crate) fn scope_key_with_precision(scope: f64, precision: f64, exact: bool) -> i64 {
    if exact {
        floor_scaled_exact(scope + SCOPE_EPS, precision as u64)
    } else {
        cost_key_with_precision(scope + SCOPE_EPS, precision) as i64
    }
}

/// `floor(x * p)` on the exact value of `x` (`x = m * 2^e`, i128 shift),
/// saturating to ±2^62 for huge / non-finite `x`.
fn floor_scaled_exact(x: f64, p: u64) -> i64 {
    const CAP: i128 = 1 << 62;
    if x.is_nan() {
        return -(CAP as i64);
    }
    if x.is_infinite() || x.abs() >= 1e15 {
        return if x > 0.0 { CAP as i64 } else { -(CAP as i64) };
    }
    if x == 0.0 {
        return 0;
    }
    let bits = x.to_bits();
    let neg = (bits >> 63) != 0;
    let exp_bits = ((bits >> 52) & 0x7ff) as i32;
    let frac = (bits & ((1u64 << 52) - 1)) as i128;
    let (mant, exp) = if exp_bits == 0 {
        (frac, -1074)
    } else {
        (frac | (1i128 << 52), exp_bits - 1075)
    };
    let m = (if neg { -mant } else { mant }) * p as i128;
    let v = if exp >= 0 {
        m << exp
    } else if exp > -127 {
        m >> (-exp) // arithmetic shift = floor, also for negatives
    } else if m < 0 {
        -1
    } else {
        0
    };
    v.clamp(-CAP, CAP) as i64
}

/// Largest lattice denominator used for exact keys: keeps
/// `SCOPE_EPS * precision < 1` (the 1e-9 slack stays below one lattice
/// step, so re-keying a child scope `K / precision` returns `K`) and
/// every key sum far below 2^53.
const EXACT_PRECISION_MAX: f64 = 1e8;

/// Exact-key precision of the integer cost keys on the leaf-only objective
/// (`CostLattice`): `P = n_total * q` where `leaf_reg = p / q` exactly in
/// f64 (smallest such `q`, i.e. the reduced fraction; it equals
/// `Fraction(leaf_reg).limit_denominator(1e6)` since distinct fractions
/// with denominators <= 1e6 are > 1e-12 apart). Every tree cost
/// `err / n_total + (p/q) * leaves` is then `(err*q + p*n_total*leaves) / P`,
/// an integer key, so key sums are exact. Returns None (rounded keys)
/// unless every finite leaf objective in the memo lies on the lattice to
/// float noise (|c*P - round(c*P)| <= 1e-6) and `P <= EXACT_PRECISION_MAX`.
fn exact_lattice_precision(
    memo: &dyn MemoLookup,
    lattice: crate::types::CostLattice,
) -> Option<f64> {
    let n = lattice.n_total as f64;
    let reg = lattice.leaf_reg;
    if lattice.n_total == 0 || !reg.is_finite() || reg < 0.0 {
        return None;
    }
    let q_max = (EXACT_PRECISION_MAX / n).floor() as u64;
    let q = (1..=q_max.min(1_000_000)).find(|&q| {
        let qf = q as f64;
        ((reg * qf).round() / qf) == reg
    })?;
    let precision = n * q as f64;
    let on_lattice = memo.all_values(&mut |sp| {
        let c = sp.base_objective;
        if !c.is_finite() {
            return true;
        }
        let k = c * precision;
        (k - k.round()).abs() <= 1e-6
    });
    if on_lattice {
        Some(precision)
    } else {
        None
    }
}

/// Number of root trees with cost ≤ `bound` (in key space on a keyed DAG).
/// Used for `tree_count_at_target` when the enumeration bound was widened
/// by a phase-1 gap.
pub fn tree_count_at_bound(dag: &FactoredDag, bound: f64) -> u128 {
    count_sum(
        dag.root_cost_schedule
            .iter()
            .filter(|e| root_entry_within_bound(dag, e.cost, bound))
            .map(|e| e.tree_count()),
    )
}

pub struct DagSummary {
    pub n_nodes: usize,
    pub n_internal: usize,
    pub n_leaf_only: usize,
    pub n_both: usize,
    pub total_split_entries: usize,
    pub tree_count: u128,
    pub n_cost_groups: usize,
    pub opt: f64,
    pub rashomon_bound: f64,
    pub theta: f64,
    pub delta_budget: DeltaBudget,
}

// ===========================================================================
// Factored DAG: cross-product-free representation
// ===========================================================================

/// Quantize a cost to u64 for BTreeMap keying.
/// `precision` is the multiplier (e.g. 1e9 for 1e-9 resolution, 1e5 for 1e-5).
pub fn cost_key_with_precision(c: f64, precision: f64) -> u64 {
    ((c * precision).round() as i64).max(0) as u64
}

/// Build a factored DAG from the BnB memo table.
///
/// Instead of materializing D^k cross-product nodes per split, stores each
/// child's cost schedule separately and computes tree counts via convolution.
///
/// `cost_precision`: controls cost quantization granularity.
///   - 1e9 = finest (1e-9 resolution, ~no merging, exact)
///   - 1e7 = moderate (1e-7, slight merging)
///   - 1e6 = aggressive (1e-6, significant merging, fewer cost levels)
///   Higher values = more precision, more memory. Lower = less memory, slight overcounting.
///   Minimum safe value is 1e6 (below this, floating-point rounding can cause
///   extraction failures where convolved costs can't be reconstructed from children).
pub fn build_factored_dag(
    memo: &dyn MemoLookup,
    pool: &SplitRecordPool,
    arena: &MemoKeyArena,
    root_key: &MemoKey,
    rashomon_bound: f64,
    opt: f64,
    theta: f64,
    reg: f64,
    delta_budget: DeltaBudget,
    cost_precision: f64,
    cost_lattice: Option<crate::types::CostLattice>,
    solve_depth: i32,
    total_max_depth: i32,
) -> FactoredDag {
    build_factored_dag_impl(
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
        false,
    )
}

/// `stepb` (Step A ahead of the completion search, Step B): every depth-0 vertex
/// with depth still to complete is a placeholder priced at its admissible
/// lower bound — schedule [(lb_u, 1)], kept even when its leaf cost exceeds
/// the scope (the leaf option itself is kept only when it fits) — and
/// Compaction is deferred to `stepb_fill` (compaction and quotient last).
#[allow(clippy::too_many_arguments)]
fn build_factored_dag_impl(
    memo: &dyn MemoLookup,
    pool: &SplitRecordPool,
    arena: &MemoKeyArena,
    root_key: &MemoKey,
    rashomon_bound: f64,
    opt: f64,
    theta: f64,
    reg: f64,
    delta_budget: DeltaBudget,
    cost_precision: f64,
    cost_lattice: Option<crate::types::CostLattice>,
    solve_depth: i32,
    total_max_depth: i32,
    stepb: bool,
) -> FactoredDag {
    build_factored_dag_roots(
        memo,
        pool,
        arena,
        &[(root_key.clone(), rashomon_bound)],
        rashomon_bound,
        opt,
        theta,
        reg,
        delta_budget,
        cost_precision,
        cost_lattice,
        solve_depth,
        total_max_depth,
        stepb,
        true,
    )
    .0
}

/// `build_factored_dag_impl` over several roots (key, scope) sharing one
/// builder cache. `single` (one root, at `rashomon_bound`): the original
/// single-root build (compaction compaction outside Step B, root schedule
/// filtered at beta). Otherwise (shared Step B, `build_dag_multi`): no
/// Compaction, `root_ref` = first root, `tree_count` = Σ root counts.
/// Returns the DAG and per root (ref, tree count within its scope).
#[allow(clippy::too_many_arguments)]
fn build_factored_dag_roots(
    memo: &dyn MemoLookup,
    pool: &SplitRecordPool,
    arena: &MemoKeyArena,
    roots: &[(MemoKey, f64)],
    rashomon_bound: f64,
    opt: f64,
    theta: f64,
    reg: f64,
    delta_budget: DeltaBudget,
    cost_precision: f64,
    cost_lattice: Option<crate::types::CostLattice>,
    solve_depth: i32,
    total_max_depth: i32,
    stepb: bool,
    single: bool,
) -> (FactoredDag, Vec<(FSubRef, u128)>) {
    let mut subproblems: Vec<FactoredSubproblem> = Vec::new();
    // Integer cost keys. On the leaf-only objective: exact lattice keys
    // (the DAG's cost_precision becomes the lattice denominator).
    // Otherwise (split_reg / path_penalty != 0, off-lattice leaf
    // objectives) the given precision with rounded keys.
    let exact_precision = cost_lattice.and_then(|l| exact_lattice_precision(memo, l));
    let exact = exact_precision.is_some();
    let cost_precision = exact_precision.unwrap_or(cost_precision);
    let mut cache: HashMap<MemoKey, (FSubRef, Vec<CostEntry>, f64)> = HashMap::new();
    // Build-time SplitSpec interner (full-mode path).
    let mut spec_pool = crate::types::SplitSpecPool::new();

    // depth_remaining > 0 means the user wanted more tree depth than the solver
    // ran for (lookahead/closure truncation). Depth-0 forced leaves get marked
    // as prefix leaves; downstream callers can opt into recursive fill.
    let prefix_remaining = (total_max_depth - solve_depth).max(0);

    let mut built: Vec<(FSubRef, Vec<CostEntry>)> = Vec::with_capacity(roots.len());
    for (root_key, scope) in roots {
        built.push(build_factored_recursive(
            memo,
            pool,
            arena,
            root_key,
            *scope,
            &mut subproblems,
            &mut cache,
            &mut spec_pool,
            cost_precision,
            exact,
            prefix_remaining,
            stepb,
        ));
    }
    let fits_at = |c: f64, scope: f64| -> bool {
        cost_key_with_precision(c, cost_precision) as i64
            <= scope_key_with_precision(scope, cost_precision, exact)
    };
    let root_counts: Vec<(FSubRef, u128)> = if single {
        Vec::new()
    } else {
        built
            .iter()
            .zip(roots)
            .map(|((r, sched), (_, scope))| {
                let c = sched
                    .iter()
                    .filter(|e| fits_at(e.cost, *scope))
                    .map(|e| e.tree_count())
                    .fold(0u128, count_add);
                (*r, c)
            })
            .collect()
    };
    let (root_ref, root_schedule) = built.swap_remove(0);
    drop(built);

    // Keep only vertices reachable from the root through recorded splits
    // (every recorded split has non-empty child schedules), so n_dag_nodes
    // / DAG bytes count no slots reserved for splits whose schedules
    // turned out empty.
    let mut root_ref = root_ref;
    if single && !stepb {
        root_ref = compact_reachable(&mut subproblems, root_ref);
    }

    // Filter root schedule to rashomon_bound
    let root_scope = roots[0].1;
    let root_cost_schedule: Vec<CostEntry> = root_schedule
        .into_iter()
        .filter(|e| fits_at(e.cost, root_scope))
        .collect();

    let tree_count: u128 = if single {
        count_sum(root_cost_schedule.iter().map(|e| e.tree_count()))
    } else {
        root_counts.iter().fold(0u128, |a, r| count_add(a, r.1))
    };

    let dag = FactoredDag {
        subproblems,
        root_ref,
        root_cost_schedule,
        tree_count,
        opt,
        rashomon_bound,
        theta,
        reg,
        delta_budget,
        cost_precision,
        keyed_costs: true,
        exact_cost_keys: exact,
        // Split-record pool telemetry: snapshot pool stats before drop.
        pool_total_bytes: pool.approximate_bytes(),
        pool_unique_split_count: pool.len() as u64,
        pool_dedup_hits: pool.dedup_hits,
        pool_total_pushed: pool.total_pushed,
        // Schedule-interner telemetry: snapshot the thread-local schedule
        // interner populated during `build_factored_recursive` (full mode
        // path).
        schedule_intern_unique: crate::types::SCHEDULE_INTERNER_TL
            .with(|c| c.borrow().unique_count() as u64),
        schedule_intern_attempts: crate::types::SCHEDULE_INTERNER_TL.with(|c| {
            c.borrow()
                .intern_attempts
                .load(std::sync::atomic::Ordering::Relaxed)
        }),
        schedule_intern_hits: crate::types::SCHEDULE_INTERNER_TL.with(|c| {
            c.borrow()
                .intern_hits
                .load(std::sync::atomic::Ordering::Relaxed)
        }),
        schedule_intern_bytes: crate::types::SCHEDULE_INTERNER_TL
            .with(|c| c.borrow().approximate_bytes()),
        // MemoKey arena telemetry (full mode path).
        arena_total_pushed: arena.pushed(),
        arena_unique: arena.len() as u64,
        arena_dedup_hits: arena.hits(),
        arena_bytes: arena.approximate_bytes(),
        // Empty schedule cache; populated lazily at extract time.
        schedule_cache: Arc::new(dashmap::DashMap::new()),
        wide_schedule_cache: Default::default(),
        // Pool populated post-quotient by `promote_splits_to_pool`.
        split_pool: crate::types::FactoredSplitPool::new(),
        // Transfer the deduplicated SplitSpec pool.
        spec_pool: spec_pool.into_pool(),
        // The full-mode builder does not emit inline leaves; the
        // lickety driver does. Loaded DAGs / full-mode DAGs leave these
        // empty.
        inline_leaves: Vec::new(),
        inline_leaf_costs: Vec::new(),
        inline_leaves_minority: Vec::new(),
        inline_leaf_minority_costs: Vec::new(),
        n_quotient_hash_collisions: 0,
        count_saturated: false,
    };
    (dag, root_counts)
}

/// Recursively build a factored subproblem and return its cost schedule.
///
/// The cost schedule lists all achievable subtree costs at this subproblem,
/// each with the total number of trees at that exact cost.
///
/// The cache stores each node's schedule with the scope it was built at.
/// A hit is served only when that scope covers the
/// request; a wider request rebuilds the node into the SAME arena slot, so
/// every parent sees the widest schedule (counts at a cost ≤ scope do not
/// depend on the scope, so earlier parents stay consistent).
fn build_factored_recursive(
    memo: &dyn MemoLookup,
    pool: &SplitRecordPool,
    arena: &MemoKeyArena,
    key: &MemoKey,
    scope: f64,
    subproblems: &mut Vec<FactoredSubproblem>,
    cache: &mut HashMap<MemoKey, (FSubRef, Vec<CostEntry>, f64)>,
    spec_pool: &mut crate::types::SplitSpecPool,
    cost_precision: f64,
    exact: bool,
    prefix_remaining: i32,
    stepb: bool,
) -> (FSubRef, Vec<CostEntry>) {
    let ck_ = |c: f64| -> u64 { cost_key_with_precision(c, cost_precision) };
    // `fits(cost, scope)`: the one admission test used for every scope
    // filter in this builder (integer cost keys).
    let fits = |c: f64, s: f64| -> bool {
        ck_(c) as i64 <= scope_key_with_precision(s, cost_precision, exact)
    };

    // Cache hit: return existing subproblem ref and schedule
    let mut reuse_slot: Option<FSubRef> = None;
    if let Some((ref_idx, schedule, built_scope)) = cache.get(key) {
        if *built_scope + SCOPE_EPS >= scope {
            // Filter to current scope (cache was built at a wider scope)
            let filtered: Vec<CostEntry> = schedule
                .iter()
                .filter(|e| fits(e.cost, scope))
                .cloned()
                .collect();
            return (*ref_idx, filtered);
        }
        reuse_slot = Some(*ref_idx);
    }

    let sp = match memo.get(key) {
        Some(sp) => sp,
        None => {
            // Unknown subproblem — create empty placeholder
            let ref_idx = subproblems.len() as FSubRef;
            subproblems.push(FactoredSubproblem {
                leaf: None,
                leaf_minority: None,
                prefix_leaf: None,
                splits: Vec::new(),
                split_ids: Vec::new(),
                tree_count: 0,
                min_obj: f64::INFINITY,
                cost_entries: None,
            });
            cache.insert(key.clone(), (ref_idx, Vec::new(), f64::INFINITY));
            return (ref_idx, Vec::new());
        }
    };

    // Reserve slot in arena (needed before recursing to handle DAG structure);
    // a widening rebuild overwrites its existing slot.
    let ref_idx = match reuse_slot {
        Some(r) => r,
        None => {
            let r = subproblems.len() as FSubRef;
            subproblems.push(FactoredSubproblem {
                leaf: None,
                leaf_minority: None,
                prefix_leaf: None,
                splits: Vec::new(),
                split_ids: Vec::new(),
                tree_count: 0,
                min_obj: f64::INFINITY,
                cost_entries: None,
                // Reserved slot, overwritten by full-build finalize below.
            });
            r
        }
    };

    // Aggregate cost schedule across leaf + all splits
    let mut all_costs: BTreeMap<u64, (f64, u128)> = BTreeMap::new();

    // 1. Leaf option
    let leaf = if fits(sp.base_objective, scope) {
        let n_classes = sp.n_classes as usize;
        let prediction = sp.class_counts[..n_classes]
            .iter()
            .enumerate()
            .max_by_key(|(_, &c)| c)
            .map(|(i, _)| i as u8)
            .unwrap_or(0);

        let leaf_info = RLeafInfo {
            prediction,
            loss: sp.max_loss,
            n_samples: sp.n_samples,
        };
        let ck = ck_(sp.base_objective);
        let entry = all_costs.entry(ck).or_insert((sp.base_objective, 0));
        entry.1 += 1;
        Some((sp.base_objective, leaf_info))
    } else {
        None
    };

    // 1b. Prefix-leaf marker for depth-0 forced leaves when the user asked for
    //     deeper trees than the solver ran for. The leaf above stays in place
    //     as a majority-class placeholder so existing tree extraction works;
    //     the prefix_leaf field is purely additive metadata for recursive fill.
    //     Completion search (Step B, `stepb`): every such vertex is a placeholder, even
    //     when its leaf cost exceeds the scope.
    let placeholder = stepb && prefix_remaining > 0 && sp.depth == 0;
    let prefix_leaf = if prefix_remaining > 0 && sp.depth == 0 && (leaf.is_some() || placeholder) {
        Some(PrefixLeafInfo {
            support: sp.support.clone(),
            class_counts: sp.class_counts,
            n_classes: sp.n_classes,
            n_samples: sp.n_samples,
            depth_remaining: prefix_remaining.min(u8::MAX as i32) as u8,
            // sp.upperbound is the tightest bound the solver computed at this
            // node; for a forced-leaf depth-0 subproblem it equals the leaf
            // objective. For closure-informed runs it carries the closure UB
            // for free, which is the SPLIT-style beat-greedy bound we want.
            greedy_loss: sp.upperbound,
        })
    } else {
        None
    };
    // 1c. Step-B placeholder schedule: one provisional entry at the
    //     admissible completion lb, not the leaf cost.
    if placeholder {
        all_costs.clear();
        let lb = sp.lowerbound.min(sp.base_objective);
        if fits(lb, scope) {
            all_costs.insert(ck_(lb), (lb, 1));
        }
    }

    // 2. Splits
    let mut splits: Vec<FactoredSplit> = Vec::new();

    // `sp.rashomon_indices` may name the same split several
    // times (raw re-pushes, and the pool keys on (split_id, split_upper) so
    // one split can own several indices). Each copy would become an
    // identical FactoredSplit and multiply tree_count. Keep the first copy
    // per split identity (feature, threshold bits, group_map, child keys);
    // never dedup on predictions or objective, so splits whose children
    // differ (incl. threshold-granularity variants) all stay.
    let mut seen_idx: std::collections::HashSet<u32> = std::collections::HashSet::new();
    let mut seen_id: std::collections::HashSet<(usize, Vec<u64>, Option<Vec<usize>>, Vec<u32>)> =
        std::collections::HashSet::new();

    let split_indices: &[u32] = if placeholder {
        &[]
    } else {
        &sp.rashomon_indices
    };
    for &sr_idx in split_indices {
        if !seen_idx.insert(sr_idx) {
            continue;
        }
        let sr = pool.get(sr_idx);
        let k = sr.child_keys.len();
        if k < 2 {
            continue;
        }
        let identity = (
            sr.feature_index,
            sr.thresholds
                .iter()
                .map(|t| t.to_bits())
                .collect::<Vec<u64>>(),
            sr.group_map.clone(),
            sr.child_keys.clone(),
        );
        if !seen_id.insert(identity) {
            continue;
        }

        // Get child lowerbounds
        let child_lbs: Vec<f64> = sr
            .child_keys
            .iter()
            .map(|&ck_idx| {
                memo.get(arena.get(ck_idx))
                    .map(|csp| csp.lowerbound)
                    .unwrap_or(f64::INFINITY)
            })
            .collect();

        // Per-child budgets in key space,
        // child_key_i = scope_key − Σ_{j≠i} ck(lb_j). Every counted combo
        // (Σ ck(c_j) ≤ scope_key, ck(c_j) ≥ ck(lb_j)) then passes each
        // child filter, so extraction's flat-scope combo walk sees the
        // same combo set.
        let scope_k = scope_key_with_precision(scope, cost_precision, exact);
        let lb_keys: Vec<i64> = child_lbs
            .iter()
            .map(|&lb| {
                if lb.is_finite() {
                    ck_(lb) as i64
                } else {
                    1i64 << 52
                }
            })
            .collect();
        let total_lb_key: i64 = lb_keys.iter().sum();
        if total_lb_key > scope_k {
            continue;
        }

        // Recurse into each child and collect their cost schedules
        let mut children: Vec<FSubRef> = Vec::with_capacity(k);
        let mut child_schedules: Vec<Vec<CostEntry>> = Vec::with_capacity(k);
        let mut any_empty = false;

        for (i, &ck_idx) in sr.child_keys.iter().enumerate() {
            let ck = arena.get(ck_idx);
            let child_key = scope_k - (total_lb_key - lb_keys[i]);
            // Build the child at a float scope whose key covers child_key.
            let child_scope = child_key as f64 / cost_precision;

            let (child_ref, child_schedule) = build_factored_recursive(
                memo,
                pool,
                arena,
                ck,
                child_scope,
                subproblems,
                cache,
                spec_pool,
                cost_precision,
                exact,
                prefix_remaining,
                stepb,
            );

            // Filter to child_scope
            let filtered: Vec<CostEntry> = child_schedule
                .into_iter()
                .filter(|e| ck_(e.cost) as i64 <= child_key)
                .collect();

            if filtered.is_empty() {
                any_empty = true;
                break;
            }

            child_schedules.push(filtered);
            children.push(child_ref);
        }

        if any_empty {
            continue;
        }

        // Convolve children's cost schedules to get this split's contribution
        let child_schedule_refs: Vec<&[CostEntry]> =
            child_schedules.iter().map(|v| v.as_slice()).collect();
        let split_schedule =
            convolve_cost_schedules_keyed(&child_schedule_refs, scope_k, cost_precision);
        let split_tree_count: u128 = count_sum(split_schedule.iter().map(|e| e.tree_count()));

        if split_tree_count == 0 {
            continue;
        }

        // Add to aggregate cost schedule
        for entry in &split_schedule {
            let ck = ck_(entry.cost);
            let agg = all_costs.entry(ck).or_insert((entry.cost, 0));
            agg.1 = count_add(agg.1, entry.tree_count());
        }

        let spec = SplitSpec {
            feature_index: sr.feature_index as u16,
            arity: k as u8,
            thresholds: sr.thresholds.clone(),
            group_map: sr
                .group_map
                .as_ref()
                .map(|gm| gm.iter().map(|&g| g as u16).collect::<Vec<_>>().into()),
        };
        // Intern spec → canonical spec_id (4 B on the split).
        let spec_id = spec_pool.intern(&spec);

        // Shrink the per-split children Vec before push.
        // Each split's children is exactly k entries known in advance; the
        // builder above grows incrementally via push so cap may be ~2× len.
        // Children move into the inline SmallVec; k≤5 stays inline
        // (no heap, no shrink needed). `from_vec` reuses the existing Vec
        // allocation only when it spills (k>4).
        let children: crate::types::InlineChildren = smallvec::SmallVec::from_vec(children);
        splits.push(FactoredSplit::new(spec_id, children, split_tree_count));
    }

    // Compute total tree count
    let total_tree_count: u128 = count_sum(all_costs.values().map(|&(_, tc)| tc));

    // Build cost schedule
    let cost_schedule: Vec<CostEntry> = all_costs
        .into_values()
        .map(|(cost, tree_count)| CostEntry::new(cost, tree_count))
        .collect();

    // Capture min_obj from the aggregate schedule first entry.
    let min_obj = cost_schedule
        .first()
        .map(|e| e.cost)
        .unwrap_or(f64::INFINITY);

    // Shrink per-node splits Vec — each FactoredSubproblem
    // pushes splits incrementally so cap may exceed len by ~40%.
    let mut splits = splits;
    splits.shrink_to_fit();

    // Persist the canonical schedule on subproblem via the
    // thread-local schedule_interner so extract_at_cost / materialize_schedule
    // can use the SAME entries that produced tree_count.
    let interned_schedule = crate::types::intern_schedule_tl(cost_schedule.clone());

    // Fill in the reserved slot
    subproblems[ref_idx as usize] = FactoredSubproblem {
        leaf,
        leaf_minority: None,
        prefix_leaf: prefix_leaf.map(Box::new),
        splits,
        split_ids: Vec::new(),
        tree_count: total_tree_count,
        min_obj,
        cost_entries: Some(interned_schedule),
    };

    cache.insert(key.clone(), (ref_idx, cost_schedule.clone(), scope));
    (ref_idx, cost_schedule)
}

/// Convolve multiple children's cost schedules, returning the combined
/// cost schedule subject to Σ child_costs ≤ scope.
///
/// Folds children left-to-right. At each step, the partial schedule
/// contains all valid partial sums seen so far, with aggregated tree counts.
///
/// Takes child schedules as borrowed slices so callers don't have to
/// clone the child `cost_entries` Vec into a temporary `Vec<Vec<CostEntry>>`
/// (the lickety driver invokes this in a hot loop).
pub fn convolve_cost_schedules(
    children: &[&[CostEntry]],
    scope: f64,
    cost_precision: f64,
) -> Vec<CostEntry> {
    let mut scratch = crate::scratch::ConvolveScratch::default();
    convolve_cost_schedules_in(children, scope, cost_precision, &mut scratch);
    std::mem::take(&mut scratch.out)
}

/// Integer cost keys convolve for the full-build path: every child entry is
/// keyed once (`ck(cost)`), partial sums are integer key sums, and the
/// only admission test is `Σ key ≤ scope_key`. Output costs are de-keyed
/// (`key / precision`) so `ck(out.cost) == key` round-trips exactly, which
/// is what `enumerate_combos_at_target` matches against. Counts stay u128.
fn convolve_cost_schedules_keyed(
    children: &[&[CostEntry]],
    scope_key: i64,
    cost_precision: f64,
) -> Vec<CostEntry> {
    let key = |c: f64| -> i64 { cost_key_with_precision(c, cost_precision) as i64 };
    let mut suffix_min = vec![0i64; children.len() + 1];
    for i in (0..children.len()).rev() {
        let Some(first) = children[i].first() else {
            return Vec::new();
        };
        suffix_min[i] = suffix_min[i + 1].saturating_add(key(first.cost));
    }
    let mut partial: BTreeMap<i64, u128> = BTreeMap::new();
    partial.insert(0, 1);
    for (i, child) in children.iter().enumerate() {
        // A partial sum must leave enough budget for every remaining child.
        let cap = scope_key.saturating_sub(suffix_min[i + 1]);
        let mut next: BTreeMap<i64, u128> = BTreeMap::new();
        for (&pk, &pc) in &partial {
            for e in child.iter() {
                let q = pk.saturating_add(key(e.cost));
                if q > cap {
                    break; // child schedules are sorted ascending
                }
                let agg = next.entry(q).or_insert(0);
                *agg = count_add(*agg, count_mul(pc, e.tree_count()));
            }
        }
        if next.is_empty() {
            return Vec::new();
        }
        partial = next;
    }
    partial
        .into_iter()
        .map(|(q, tc)| CostEntry::new(q as f64 / cost_precision, tc))
        .collect()
}

/// Scratch-arena variant of `convolve_cost_schedules`. Uses the
/// caller-provided `ConvolveScratch` for the running `partial`, the
/// per-step accumulator (now an `FxHashMap` rather than a fresh
/// `BTreeMap`), and the final `out` `Vec<CostEntry>`. All three buffers
/// are `clear()`ed on entry; their capacity persists across calls so the
/// hot loop in `solve_node` / `solve_depth1_inline` hits zero heap
/// allocations in steady state.
///
/// On return the result is in `scratch.out`, sorted ascending by cost.
/// On empty children or empty fold result, `scratch.out` is left empty.
pub fn convolve_cost_schedules_in(
    children: &[&[CostEntry]],
    scope: f64,
    cost_precision: f64,
    scratch: &mut crate::scratch::ConvolveScratch,
) {
    scratch.reset();
    if children.is_empty() {
        return;
    }

    // Quantize cost to i32 once (cost * precision rounded). With
    // the default precision=1e9 and cost ∈ [0, 1], q ∈ [0, 1e9] which
    // fits inside i32::MAX (≈2.147e9) with margin. Anything wider would
    // need to be caught here, but in practice the rashomon_bound is < 1.
    // The scope ceiling is pre-quantized so the inner break-check is a
    // single i32 comparison rather than a float add+compare.
    let quant = |c: f64| -> i32 { (c * cost_precision).round().max(0.0) as i32 };
    // Use ceil so we don't accidentally exclude entries exactly at the
    // boundary that the float comparison `combined <= scope + SCOPE_EPS`
    // would have included.
    let scope_q: i32 = ((scope + SCOPE_EPS) * cost_precision).ceil().max(0.0) as i32;

    // Start with first child's schedule, filtered to scope. Tree counts
    // in the inner aggregator are u128 with saturating `count_add` /
    // `count_mul` (exact below u128::MAX; a clamp records the event).
    scratch.partial.extend(
        children[0]
            .iter()
            .filter(|e| e.cost <= scope + SCOPE_EPS)
            .map(|e| (quant(e.cost), e.tree_count())),
    );

    // Fold in each subsequent child.
    for child_schedule in &children[1..] {
        scratch.next.clear();

        // Pre-quantize this child once so the inner loop is integer-only.
        // The same read-ahead as in PRAXIS (Heile et al.).
        scratch.r_quantized.clear();
        scratch.r_quantized.extend(
            child_schedule
                .iter()
                .map(|e| (quant(e.cost), e.tree_count())),
        );

        for &(p_q, p_count) in &scratch.partial {
            for &(r_q, r_count) in &scratch.r_quantized {
                let combined_q = p_q + r_q;
                if combined_q > scope_q {
                    break; // child_schedule is sorted ascending → early exit
                }
                let agg = scratch.next.entry(combined_q).or_insert(0);
                *agg = count_add(*agg, count_mul(p_count, r_count));
            }
        }

        // Rebuild partial from next; sort ascending by quantized cost so
        // the early-exit in the next inner loop still works. Integer
        // sort_unstable is total-ordered and faster than f64 partial_cmp.
        scratch.partial.clear();
        scratch
            .partial
            .extend(scratch.next.iter().map(|(&k, &v)| (k, v)));
        if scratch.partial.is_empty() {
            return;
        }
        scratch.partial.sort_unstable_by_key(|&(k, _)| k);
    }

    // Emit final `out`, sorted ascending. De-quantize back to f64.
    let inv = 1.0 / cost_precision;
    scratch.out.extend(
        scratch
            .partial
            .iter()
            .map(|&(q, tc)| CostEntry::new(q as f64 * inv, tc)),
    );
    if children.len() == 1 {
        scratch.out.sort_unstable_by(|a, b| {
            a.cost
                .partial_cmp(&b.cost)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }
}

/// Compaction: drop the vertices of `subproblems` not reachable from
/// `root` through split children (slots reserved for splits whose child
/// schedules turned out empty, placeholders of dropped splits). Survivors
/// keep their relative order and child refs are remapped; returns the new
/// root ref. Runs before the quotient and before `promote_splits_to_pool`
/// (`split_ids` still empty), so only `splits[*].children` hold refs.
fn compact_reachable(subproblems: &mut Vec<FactoredSubproblem>, root: FSubRef) -> FSubRef {
    let n = subproblems.len();
    if n == 0 || (root as usize) >= n {
        return root;
    }
    let mut reach = vec![false; n];
    let mut stack = vec![root as usize];
    reach[root as usize] = true;
    while let Some(i) = stack.pop() {
        for split in &subproblems[i].splits {
            for &c in split.children.iter() {
                let c = c as usize;
                if c < n && !reach[c] {
                    reach[c] = true;
                    stack.push(c);
                }
            }
        }
    }
    if reach.iter().all(|&r| r) {
        return root;
    }
    let mut remap: Vec<FSubRef> = vec![FSubRef::MAX; n];
    let mut next: FSubRef = 0;
    for i in 0..n {
        if reach[i] {
            remap[i] = next;
            next += 1;
        }
    }
    let old = std::mem::take(subproblems);
    subproblems.reserve_exact(next as usize);
    for (i, mut sp) in old.into_iter().enumerate() {
        if !reach[i] {
            continue;
        }
        for split in &mut sp.splits {
            for c in split.children.iter_mut() {
                *c = remap[*c as usize];
            }
        }
        subproblems.push(sp);
    }
    remap[root as usize]
}

/// Per-(vertex, scope key) schedule cache (`FactoredDag.schedule_cache`).
type ScheduleCacheMap = dashmap::DashMap<(FSubRef, u64), Arc<Vec<CostEntry>>>;

/// `schedule_cache` key of (`sub_ref`, `scope`): the scope quantized to
/// 1e-6, or on exact-key DAGs the exact scope key (distinct lattice scopes
/// never share a cached schedule).
fn schedule_cache_key(exact: bool, precision: f64, sub_ref: FSubRef, scope: f64) -> (FSubRef, u64) {
    if exact {
        let k = scope_key_with_precision(scope, precision, true);
        (sub_ref, k.max(0) as u64)
    } else {
        (sub_ref, quantize_scope(scope))
    }
}

/// Lazily materialize the cost schedule for a subproblem under
/// the given scope. Hits the per-DAG `schedule_cache` when possible;
/// otherwise recursively descends, convolves children, and inserts the
/// resulting Arc'd schedule into the cache. Returns an
/// `Arc<Vec<CostEntry>>` callers can borrow as `&schedule[..]`.
///
/// Schedules under `scope < sub.min_obj - SCOPE_EPS` short-circuit to an
/// empty Arc — no admissible trees fit. Subproblem `min_obj` is set at
/// build time (full mode) or finalize time (lickety mode); for IO-loaded
/// DAGs it is computed bottom-up post-load via `recompute_min_obj_arena`.
pub fn materialize_schedule(
    dag: &FactoredDag,
    sub_ref: FSubRef,
    scope: f64,
) -> Arc<Vec<CostEntry>> {
    if sched_wide_active(dag) {
        let (arc, len) = materialize_schedule_wide(dag, sub_ref, scope);
        return if len == arc.len() {
            arc
        } else {
            Arc::new(arc[..len].to_vec())
        };
    }
    materialize_schedule_legacy(dag, sub_ref, scope, &dag.schedule_cache)
}

/// `materialize_schedule` with the per-(vertex, scope key) cache `cache`
/// (`dag.schedule_cache`).
fn materialize_schedule_legacy(
    dag: &FactoredDag,
    sub_ref: FSubRef,
    scope: f64,
    cache: &ScheduleCacheMap,
) -> Arc<Vec<CostEntry>> {
    if crate::types::is_leaf_ref(sub_ref) {
        return inline_leaf_schedule(dag, sub_ref, scope);
    }

    let key = schedule_cache_key(dag.exact_cost_keys, dag.cost_precision, sub_ref, scope);
    if let Some(cached) = cache.get(&key) {
        return cached.clone();
    }

    let sub = match dag.subproblems.get(sub_ref as usize) {
        Some(s) => s,
        None => {
            let empty: Arc<Vec<CostEntry>> = Arc::new(Vec::new());
            cache.insert(key, empty.clone());
            return empty;
        }
    };

    // Integer keys only on full-build DAGs (`keyed_costs`); lickety and
    // pre-flag IO-loaded DAGs keep the legacy float comparisons.
    let cost_precision = dag.cost_precision;
    let ks = dag.keyed_costs;
    let scope_k = scope_key_with_precision(scope, cost_precision, dag.exact_cost_keys);
    let fits = |c: f64| -> bool {
        if ks {
            cost_key_with_precision(c, cost_precision) as i64 <= scope_k
        } else {
            c <= scope + SCOPE_EPS
        }
    };

    // Short-circuit: scope too tight to admit any tree from this subproblem.
    if !fits(sub.min_obj) {
        let empty: Arc<Vec<CostEntry>> = Arc::new(Vec::new());
        cache.insert(key, empty.clone());
        return empty;
    }

    // If cost_entries was persisted at build time, use it
    // directly. This is the SAME schedule that produced `sub.tree_count`,
    // so extraction will be index-consistent. We re-filter by scope to
    // handle narrower extraction queries (the stored schedule is at the
    // BUILD-time widest budget; callers may request narrower scopes).
    if let Some(stored) = &sub.cost_entries {
        // Fast path: if scope is at least the largest entry's cost, the
        // stored schedule is exactly what we want — share the Arc.
        let max_cost = stored.last().map(|e| e.cost).unwrap_or(f64::NEG_INFINITY);
        if fits(max_cost) {
            cache.insert(key, stored.clone());
            return stored.clone();
        }
        // Narrower scope: filter.
        let filtered: Vec<CostEntry> = stored.iter().filter(|e| fits(e.cost)).cloned().collect();
        let arc = Arc::new(filtered);
        cache.insert(key, arc.clone());
        return arc;
    }

    let ck_ = |c: f64| -> u64 { cost_key_with_precision(c, cost_precision) };
    let mut all_costs: BTreeMap<u64, (f64, u128)> = BTreeMap::new();

    // 1. Leaf option (if its cost fits)
    if let Some((leaf_cost, _)) = sub.leaf {
        if fits(leaf_cost) {
            let k = ck_(leaf_cost);
            let agg = all_costs.entry(k).or_insert((leaf_cost, 0));
            agg.1 = count_add(agg.1, 1);
        }
    }
    // 1b. Minority leaf option. Aggregated AFTER the majority
    // option so the extract_at_cost walk order matches the count order.
    if let Some(&(leaf_cost, _)) = sub.leaf_minority.as_deref() {
        if fits(leaf_cost) {
            let k = ck_(leaf_cost);
            let agg = all_costs.entry(k).or_insert((leaf_cost, 0));
            agg.1 = count_add(agg.1, 1);
        }
    }

    // 2. Walk splits: per-child scope = scope - sum(other children min_obj),
    //    materialize each child's schedule, convolve, and aggregate.
    // Dispatch through the pool when promoted.
    for split in iter_splits(dag, sub_ref) {
        for entry in &materialize_split_schedule_legacy(dag, split, scope, cache) {
            let k = ck_(entry.cost);
            let agg = all_costs.entry(k).or_insert((entry.cost, 0));
            agg.1 = count_add(agg.1, entry.tree_count());
        }
    }

    let schedule: Vec<CostEntry> = all_costs
        .into_values()
        .map(|(cost, tree_count)| CostEntry::new(cost, tree_count))
        .collect();
    let arc = Arc::new(schedule);
    cache.insert(key, arc.clone());
    arc
}

/// Schedule of an inline-leaf child ref (`is_leaf_ref`) under `scope`.
fn inline_leaf_schedule(dag: &FactoredDag, sub_ref: FSubRef, scope: f64) -> Arc<Vec<CostEntry>> {
    // Inline-leaf child: short-circuit to a 1- or 2-entry
    // schedule. When the inline-leaf carries a minority-class
    // option (Some at `inline_leaf_minority_costs[id]`), emit BOTH
    // (cost_majority, 1) and (cost_minority, 1) when each fits `scope`.
    // Tied case (c_maj == c_min in quantized cost) collapses to a single
    // entry with tree_count=2 — handled by the construction site, which
    // never registers a minority slot when costs tie (it emits one paired
    // slot with cost_min == None and bumps the majority schedule's count
    // to 2 via the pre-quantized check). So here we just dispatch on the
    // presence of the minority cost field.
    // Do NOT insert into `schedule_cache` (tagged refs must never appear
    // as cache keys; the inline-leaf branch is always cheap to recompute).
    let id = crate::types::leaf_id(sub_ref);
    let leaf_cost = dag.inline_leaf_costs[id];
    let min_cost_opt = dag
        .inline_leaf_minority_costs
        .get(id)
        .copied()
        .unwrap_or(None);
    match min_cost_opt {
        Some(min_cost) => {
            let prec = dag.cost_precision;
            let ck_ = |c: f64| -> u64 { cost_key_with_precision(c, prec) };
            let maj_fits = leaf_cost <= scope + SCOPE_EPS;
            let min_fits = min_cost <= scope + SCOPE_EPS;
            if !maj_fits && !min_fits {
                return Arc::new(Vec::new());
            }
            if maj_fits && min_fits {
                if ck_(leaf_cost) == ck_(min_cost) {
                    // Tied: collapse to a single bin with count=2.
                    return Arc::new(vec![CostEntry::new(leaf_cost, 2)]);
                }
                return Arc::new(vec![
                    CostEntry::new(leaf_cost, 1),
                    CostEntry::new(min_cost, 1),
                ]);
            }
            if maj_fits {
                return Arc::new(vec![CostEntry::new(leaf_cost, 1)]);
            }
            // Only minority fits — by Bayes optimality of majority,
            // this should not occur (c_maj <= c_min always). Defensive:
            // emit minority only.
            return Arc::new(vec![CostEntry::new(min_cost, 1)]);
        }
        None => {
            if leaf_cost <= scope + SCOPE_EPS {
                return Arc::new(vec![CostEntry::new(leaf_cost, 1)]);
            } else {
                return Arc::new(Vec::new());
            }
        }
    }
}

/// Cost schedule of one split of a vertex under `scope` (the split part of
/// `materialize_schedule`): child i is materialized at
/// `scope - Σ_{j≠i} min_obj(c_j)` and the children are convolved under
/// `scope`. Empty when the split admits no tree within `scope`.
fn materialize_split_schedule(
    dag: &FactoredDag,
    split: &FactoredSplit,
    scope: f64,
) -> Vec<CostEntry> {
    if sched_wide_active(dag) {
        let scope_k = scope_key_with_precision(scope, dag.cost_precision, true);
        return split_schedule_wide(dag, split, scope_k);
    }
    materialize_split_schedule_legacy(dag, split, scope, &dag.schedule_cache)
}

/// `materialize_split_schedule` with the per-scope cache `cache`.
fn materialize_split_schedule_legacy(
    dag: &FactoredDag,
    split: &FactoredSplit,
    scope: f64,
    cache: &ScheduleCacheMap,
) -> Vec<CostEntry> {
    let cost_precision = dag.cost_precision;
    let ks = dag.keyed_costs;
    let scope_k = scope_key_with_precision(scope, cost_precision, dag.exact_cost_keys);
    let ck_ = |c: f64| -> u64 { cost_key_with_precision(c, cost_precision) };
    let k = split.children.len();
    if k == 0 {
        return Vec::new();
    }
    // Child lowerbounds (their min_obj). Inline-leaf children
    // expose their leaf cost as their min_obj.
    let child_lbs: Vec<f64> = split
        .children
        .iter()
        .map(|&cref| {
            if crate::types::is_leaf_ref(cref) {
                dag.inline_leaf_costs[crate::types::leaf_id(cref)]
            } else {
                dag.subproblems
                    .get(cref as usize)
                    .map(|c| c.min_obj)
                    .unwrap_or(f64::INFINITY)
            }
        })
        .collect();
    let total_child_lb: f64 = child_lbs.iter().sum();
    let lb_keys: Vec<i64> = child_lbs
        .iter()
        .map(|&lb| {
            if lb.is_finite() {
                ck_(lb) as i64
            } else {
                1i64 << 52
            }
        })
        .collect();
    let total_lb_key: i64 = lb_keys.iter().sum();
    if ks {
        if total_lb_key > scope_k {
            return Vec::new();
        }
    } else if total_child_lb > scope + SCOPE_EPS {
        return Vec::new();
    }

    // Materialize each child's schedule under per-child scope.
    let mut child_arcs: Vec<Arc<Vec<CostEntry>>> = Vec::with_capacity(k);
    for (i, &cref) in split.children.iter().enumerate() {
        let other_lb: f64 = child_lbs
            .iter()
            .enumerate()
            .filter(|(j, _)| *j != i)
            .map(|(_, &lb)| lb)
            .sum();
        let child_scope = if ks {
            (scope_k - (total_lb_key - lb_keys[i])) as f64 / cost_precision
        } else {
            scope - other_lb
        };
        let arc = materialize_schedule_legacy(dag, cref, child_scope, cache);
        if arc.is_empty() {
            return Vec::new();
        }
        child_arcs.push(arc);
    }

    // Convolve children's schedules under this scope.
    let child_refs: Vec<&[CostEntry]> = child_arcs.iter().map(|a| a.as_slice()).collect();
    if ks {
        convolve_cost_schedules_keyed(&child_refs, scope_k, cost_precision)
    } else {
        convolve_cost_schedules(&child_refs, scope, cost_precision)
    }
}

/// Number of leading entries of `sched` (sorted ascending by cost key)
/// whose cost key is `<= scope_k`.
fn wide_prefix_len(sched: &[CostEntry], scope_k: i64, precision: f64) -> usize {
    sched.partition_point(|e| cost_key_with_precision(e.cost, precision) as i64 <= scope_k)
}

/// Wide schedule cache `materialize_schedule` (keyed exact-key DAGs): the
/// schedule of `sub_ref` at `scope` as `(arc, len)`, the answer being
/// `arc[..len]`.
pub(crate) fn materialize_schedule_wide(
    dag: &FactoredDag,
    sub_ref: FSubRef,
    scope: f64,
) -> (Arc<Vec<CostEntry>>, usize) {
    wide_schedule_prefix(dag, sub_ref, scope)
}

/// The wide schedule cache lookup. A vertex's cached schedule at scope key
/// `k_max` answers any scope key `s <= k_max` with its prefix of keys
/// `<= s`: the per-scope schedule at `s` is exactly that prefix (same keys,
/// counts and first-inserted cost representatives; child scopes, leaf tests
/// and the keyed convolve depend on the scope only through `<= s` key
/// tests, and a combo within `s` has every child within its narrower child
/// scope). A wider request recomputes the vertex at max(request, hint) and
/// replaces the entry.
fn wide_schedule_prefix(
    dag: &FactoredDag,
    sub_ref: FSubRef,
    scope: f64,
) -> (Arc<Vec<CostEntry>>, usize) {
    if crate::types::is_leaf_ref(sub_ref) {
        let arc = inline_leaf_schedule(dag, sub_ref, scope);
        let n = arc.len();
        return (arc, n);
    }
    let prec = dag.cost_precision;
    // Schedules already in `dag.schedule_cache` win, as in
    // the per-scope cache.
    let key = schedule_cache_key(true, prec, sub_ref, scope);
    if let Some(cached) = dag.schedule_cache.get(&key) {
        let arc = cached.clone();
        let n = arc.len();
        return (arc, n);
    }
    let Some(sub) = dag.subproblems.get(sub_ref as usize) else {
        return (Arc::new(Vec::new()), 0);
    };
    let scope_k = scope_key_with_precision(scope, prec, true);
    if cost_key_with_precision(sub.min_obj, prec) as i64 > scope_k {
        return (Arc::new(Vec::new()), 0);
    }
    if sub.cost_entries.is_some() {
        // Stored build-time schedule: the per-scope path.
        let arc = materialize_schedule_legacy(dag, sub_ref, scope, &dag.schedule_cache);
        let n = arc.len();
        return (arc, n);
    }
    let wide = &dag.wide_schedule_cache;
    let cached = wide.entries.get(&sub_ref).map(|e| e.value().clone());
    if let Some((k_max, arc)) = cached {
        if scope_k <= k_max {
            let n = wide_prefix_len(&arc, scope_k, prec);
            return (arc, n);
        }
    }
    let hint = wide
        .hints
        .get(sub_ref as usize)
        .copied()
        .unwrap_or(i64::MIN);
    let target_k = scope_k.max(hint);
    let fresh = Arc::new(wide_vertex_schedule(dag, sub_ref, sub, target_k));
    // Keep the wider entry if another thread filled one meanwhile.
    let arc = match wide.entries.entry(sub_ref) {
        dashmap::mapref::entry::Entry::Occupied(mut o) => {
            if o.get().0 >= target_k {
                o.get().1.clone()
            } else {
                o.insert((target_k, fresh.clone()));
                fresh
            }
        }
        dashmap::mapref::entry::Entry::Vacant(v) => {
            v.insert((target_k, fresh.clone()));
            fresh
        }
    };
    let n = wide_prefix_len(&arc, scope_k, prec);
    (arc, n)
}

/// Schedule of vertex `sub` at scope key `scope_k` (the uncached part of
/// `materialize_schedule_legacy` on keyed exact-key DAGs): leaf options,
/// then every split's schedule, aggregated per cost key with the
/// first-inserted cost as representative. The caller checked `min_obj`.
fn wide_vertex_schedule(
    dag: &FactoredDag,
    sub_ref: FSubRef,
    sub: &FactoredSubproblem,
    scope_k: i64,
) -> Vec<CostEntry> {
    let prec = dag.cost_precision;
    let ck_ = |c: f64| -> u64 { cost_key_with_precision(c, prec) };
    let fits = |c: f64| -> bool { ck_(c) as i64 <= scope_k };
    let mut all_costs: BTreeMap<u64, (f64, u128)> = BTreeMap::new();
    if let Some((leaf_cost, _)) = sub.leaf {
        if fits(leaf_cost) {
            let agg = all_costs.entry(ck_(leaf_cost)).or_insert((leaf_cost, 0));
            agg.1 = count_add(agg.1, 1);
        }
    }
    if let Some(&(leaf_cost, _)) = sub.leaf_minority.as_deref() {
        if fits(leaf_cost) {
            let agg = all_costs.entry(ck_(leaf_cost)).or_insert((leaf_cost, 0));
            agg.1 = count_add(agg.1, 1);
        }
    }
    for split in iter_splits(dag, sub_ref) {
        for entry in &split_schedule_wide(dag, split, scope_k) {
            let agg = all_costs.entry(ck_(entry.cost)).or_insert((entry.cost, 0));
            agg.1 = count_add(agg.1, entry.tree_count());
        }
    }
    all_costs
        .into_values()
        .map(|(cost, tree_count)| CostEntry::new(cost, tree_count))
        .collect()
}

/// `materialize_split_schedule_legacy` on a keyed exact-key DAG at scope
/// key `scope_k`, the children read as wide schedule cache prefixes (no copy).
fn split_schedule_wide(dag: &FactoredDag, split: &FactoredSplit, scope_k: i64) -> Vec<CostEntry> {
    let prec = dag.cost_precision;
    if split.children.is_empty() {
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
    let mut children: Vec<(Arc<Vec<CostEntry>>, usize)> = Vec::with_capacity(split.children.len());
    for (i, &cref) in split.children.iter().enumerate() {
        let child_scope = (scope_k - (total_lb_key - lb_keys[i])) as f64 / prec;
        let (arc, n) = materialize_schedule_wide(dag, cref, child_scope);
        if n == 0 {
            return Vec::new();
        }
        children.push((arc, n));
    }
    let child_refs: Vec<&[CostEntry]> = children.iter().map(|(a, n)| &a[..*n]).collect();
    convolve_cost_schedules_keyed(&child_refs, scope_k, prec)
}

/// Wide schedule cache widening hints: per vertex, the widest scope key a pass
/// asks it for (i64::MIN = never), from the parents' scopes minus the
/// other children's lb keys exactly as `split_schedule_wide` derives child
/// scopes (through the same f64 round trip). `all_at_beta`: every vertex's
/// splits are evaluated at beta (`counts_at_bound`); else only the root is,
/// and vertices are visited parents-first along `order`
/// (`root_schedule_at_bound`), skipping those the lookup answers without
/// recursing (min_obj above the scope, stored schedules). Hints only widen
/// a computation, so an overestimate costs memory, never correctness.
fn wide_scope_hints(dag: &FactoredDag, order: &[usize], all_at_beta: bool) -> Vec<i64> {
    let prec = dag.cost_precision;
    let beta_k = scope_key_with_precision(dag.rashomon_bound, prec, true);
    let mut hint = vec![i64::MIN; dag.subproblems.len()];
    if let Some(h) = hint.get_mut(dag.root_ref as usize) {
        *h = beta_k;
    }
    for &v in order {
        let Some(sp) = dag.subproblems.get(v) else {
            continue;
        };
        let s = if all_at_beta { beta_k } else { hint[v] };
        if s == i64::MIN {
            continue;
        }
        if !all_at_beta
            && (cost_key_with_precision(sp.min_obj, prec) as i64 > s || sp.cost_entries.is_some())
        {
            continue;
        }
        for split in iter_splits(dag, v as FSubRef) {
            if split.children.is_empty() {
                continue;
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
            if total_lb_key > s {
                continue;
            }
            for (i, &c) in split.children.iter().enumerate() {
                if crate::types::is_leaf_ref(c) {
                    continue;
                }
                let child_scope = (s - (total_lb_key - lb_keys[i])) as f64 / prec;
                let child_k = scope_key_with_precision(child_scope, prec, true);
                if let Some(h) = hint.get_mut(c as usize) {
                    if child_k > *h {
                        *h = child_k;
                    }
                }
            }
        }
    }
    hint
}

/// Fresh wide schedule cache cache for `dag` (hints from `wide_scope_hints`
/// when the scheme applies, else empty).
fn fresh_wide_cache(
    dag: &FactoredDag,
    order: &[usize],
    all_at_beta: bool,
) -> Arc<crate::types::WideScheduleCache> {
    let mut c = crate::types::WideScheduleCache::default();
    if sched_wide_active(dag) {
        c.hints = wide_scope_hints(dag, order, all_at_beta);
    }
    Arc::new(c)
}

// ---------------------------------------------------------------------------
// Factored DAG: tree extraction
// ---------------------------------------------------------------------------

/// Extract the tree at position `idx` from a factored DAG.
pub fn get_tree_at_index_factored(
    dag: &FactoredDag,
    idx: u128,
    feature_names: &[String],
) -> TreeNode {
    let mut remaining = idx;

    for entry in &dag.root_cost_schedule {
        if remaining < entry.tree_count() {
            return extract_at_cost(dag, dag.root_ref, entry.cost, remaining, feature_names);
        }
        remaining -= entry.tree_count();
    }

    panic!(
        "index {} out of bounds for factored DAG with tree_count={}",
        idx, dag.tree_count
    );
}

/// Set every leaf's `n_samples` of an extracted tree to the number of
/// training rows of `store` that reach it (the root support routed through
/// the tree's splits with `DatasetStore::partition_support`, the solver's
/// child-support routine). A DAG vertex's stored `n_samples` belongs to one
/// support of its quotient class, and the quotient deliberately merges
/// vertices with equal (cost, prediction, loss) but different supports, so
/// the stored value can differ from the tree's own leaf support.
pub fn route_leaf_n_samples(store: &crate::dataset_store::DatasetStore, tree: &mut TreeNode) {
    fn walk(
        store: &crate::dataset_store::DatasetStore,
        node: &mut TreeNode,
        support: &crate::types::SupportMask,
    ) {
        match node {
            TreeNode::Leaf { n_samples, .. } => *n_samples = support.n_samples,
            TreeNode::Internal {
                feature_index,
                thresholds,
                children,
                group_map,
            } => {
                let parts = store.partition_support(
                    support,
                    *feature_index,
                    thresholds,
                    group_map.as_deref(),
                );
                assert_eq!(
                    parts.len(),
                    children.len(),
                    "route_leaf_n_samples: split on feature {} routes to {} parts for {} children",
                    feature_index,
                    parts.len(),
                    children.len()
                );
                for (child, part) in children.iter_mut().zip(parts.iter()) {
                    walk(store, child, part);
                }
            }
        }
    }
    let root = store.root_support();
    walk(store, tree, &root);
}

/// `min_obj` of a split child (an inline leaf's cost for a leaf ref; +inf
/// for a dangling ref), the lower bound the per-child scopes subtract.
pub(crate) fn child_min_obj(dag: &FactoredDag, cref: FSubRef) -> f64 {
    if crate::types::is_leaf_ref(cref) {
        dag.inline_leaf_costs[crate::types::leaf_id(cref)]
    } else {
        dag.subproblems
            .get(cref as usize)
            .map(|c| c.min_obj)
            .unwrap_or(f64::INFINITY)
    }
}

/// Extract the idx-th tree from a subproblem at a specific target cost.
fn extract_at_cost(
    dag: &FactoredDag,
    sub_ref: FSubRef,
    target_cost: f64,
    idx: u128,
    feature_names: &[String],
) -> TreeNode {
    let prec = dag.cost_precision;
    let ck_ = |c: f64| -> u64 { cost_key_with_precision(c, prec) };

    // Inline-leaf child: synthesize TreeNode::Leaf directly.
    // Dispatch on target_cost when the inline-leaf carries a
    // minority-class option.
    //   - target_cost matches majority cost  → emit majority RLeafInfo
    //   - target_cost matches minority cost  → emit minority RLeafInfo
    //   - tied case (c_maj == c_min, two trees in one bin): idx 0 = maj,
    //     idx 1 = min — single bin, two trees.
    if crate::types::is_leaf_ref(sub_ref) {
        let id = crate::types::leaf_id(sub_ref);
        let leaf_cost = dag.inline_leaf_costs[id];
        let min_cost_opt = dag
            .inline_leaf_minority_costs
            .get(id)
            .copied()
            .unwrap_or(None);
        let target_key = ck_(target_cost);
        let maj_matches = ck_(leaf_cost) == target_key;
        let min_matches = min_cost_opt
            .map(|mc| ck_(mc) == target_key)
            .unwrap_or(false);
        if maj_matches && min_matches {
            // Tied case (c_maj == c_min): two distinct trees in the same
            // bin. idx selects: 0 → majority, 1 → minority.
            debug_assert!(idx < 2, "tied paired-leaf: idx must be 0 or 1");
            if idx == 0 {
                let leaf = &dag.inline_leaves[id];
                return TreeNode::Leaf {
                    prediction: leaf.prediction as usize,
                    n_samples: leaf.n_samples,
                    loss: leaf.loss,
                };
            } else {
                let leaf = dag.inline_leaves_minority[id]
                    .as_ref()
                    .expect("min_matches implies minority slot populated");
                return TreeNode::Leaf {
                    prediction: leaf.prediction as usize,
                    n_samples: leaf.n_samples,
                    loss: leaf.loss,
                };
            }
        }
        if maj_matches {
            debug_assert_eq!(
                idx, 0,
                "extract_at_cost inline-leaf (majority): idx must be 0"
            );
            let leaf = &dag.inline_leaves[id];
            return TreeNode::Leaf {
                prediction: leaf.prediction as usize,
                n_samples: leaf.n_samples,
                loss: leaf.loss,
            };
        }
        if min_matches {
            debug_assert_eq!(
                idx, 0,
                "extract_at_cost inline-leaf (minority): idx must be 0"
            );
            let leaf = dag.inline_leaves_minority[id]
                .as_ref()
                .expect("min_matches implies minority slot populated");
            return TreeNode::Leaf {
                prediction: leaf.prediction as usize,
                n_samples: leaf.n_samples,
                loss: leaf.loss,
            };
        }
        // Neither matched — surface a clean panic with diagnostics rather
        // than the previous debug_assert.
        panic!(
            "extract_at_cost inline-leaf: target_cost {} matches neither \
             majority {} nor minority {:?} (leaf_id={})",
            target_cost, leaf_cost, min_cost_opt, id,
        );
    }
    let sub = &dag.subproblems[sub_ref as usize];
    let mut remaining = idx;
    let target_key = ck_(target_cost);

    // 1. Check leaf (majority option)
    if let Some((leaf_cost, ref leaf_info)) = sub.leaf {
        if ck_(leaf_cost) == target_key {
            if remaining == 0 {
                return TreeNode::Leaf {
                    prediction: leaf_info.prediction as usize,
                    n_samples: leaf_info.n_samples,
                    loss: leaf_info.loss,
                };
            }
            remaining -= 1;
        }
    }

    // 1b. Minority leaf option. Counted AFTER the majority option
    // in the same cost bin (when costs tie) or in its own bin (distinct
    // cost). The aggregation order in `materialize_schedule` matches:
    // majority leaf, then minority leaf, then splits.
    if let Some(&(leaf_cost, ref leaf_info)) = sub.leaf_minority.as_deref() {
        if ck_(leaf_cost) == target_key {
            if remaining == 0 {
                return TreeNode::Leaf {
                    prediction: leaf_info.prediction as usize,
                    n_samples: leaf_info.n_samples,
                    loss: leaf_info.loss,
                };
            }
            remaining -= 1;
        }
    }

    // 2. Walk splits in order
    // Dispatch through the pool when promoted.
    for split in iter_splits(dag, sub_ref) {
        // Children are materialized at the flat bound, as the count pass
        // convolved them.
        let mut child_schedules: Vec<Arc<Vec<CostEntry>>> =
            Vec::with_capacity(split.children.len());
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

        // Visit combinations in the same canonical order as counting and
        // stop at the requested index, retaining only the selected tuple.
        let mut selected = None;
        visit_combos_at_target(
            &child_schedules,
            target_cost,
            prec,
            dag.keyed_costs,
            |combo| {
                let count = count_product(combo.iter().map(|e| e.tree_count()));
                if count > 0 && remaining < count {
                    selected = Some(combo.to_vec());
                    false
                } else {
                    remaining -= count;
                    true
                }
            },
        );
        if let Some(combo) = selected {
            let k = combo.len();
            let counts: Vec<u128> = combo.iter().map(|e| e.tree_count()).collect();
            let mut children_nodes = Vec::with_capacity(k);
            let mut r = remaining;
            for i in (0..k).rev() {
                let child_idx = r % counts[i];
                r /= counts[i];
                children_nodes.push(extract_at_cost(
                    dag,
                    split.children[i],
                    combo[i].cost,
                    child_idx,
                    feature_names,
                ));
            }
            children_nodes.reverse();
            let spec = dag.spec_of(split);
            return TreeNode::Internal {
                feature_index: spec.feature_index as usize,
                thresholds: spec.thresholds.clone(),
                children: children_nodes,
                group_map: spec
                    .group_map
                    .as_ref()
                    .map(|gm| gm.iter().map(|&g| g as usize).collect()),
            };
        }
    }

    let has_ce = sub.cost_entries.is_some();
    let ce_len = sub.cost_entries.as_ref().map(|s| s.len()).unwrap_or(0);
    let ce_at_tc = sub.cost_entries.as_ref().and_then(|s| {
        s.iter()
            .find(|e| ck_(e.cost) == target_key)
            .map(|e| e.tree_count())
    });
    let ce_max = sub
        .cost_entries
        .as_ref()
        .and_then(|s| s.last().map(|e| e.cost))
        .unwrap_or(f64::NAN);
    panic!(
        "extract_at_cost: idx {} out of bounds at sub_ref={} target_cost={:.9} has_ce={} ce_len={} tc_at_target={:?} sub.tree_count={} ce_max={:.9} n_splits={}",
        idx, sub_ref, target_cost, has_ce, ce_len, ce_at_tc, sub.tree_count,
        ce_max, sub.splits.len()
    );
}

/// Visit child-cost combinations in canonical order without retaining them.
/// Returning false from the visitor stops enumeration immediately. A combo
/// matches when the sum of its per-child cost keys equals the key of
/// `target`, the same sum the count pass convolves.
pub(crate) fn visit_combos_at_target<F: FnMut(&[CostEntry]) -> bool>(
    child_schedules: &[Arc<Vec<CostEntry>>],
    target: f64,
    cost_precision: f64,
    keyed: bool,
    mut visitor: F,
) {
    if child_schedules.is_empty() {
        if !keyed && cost_key_with_precision(target, cost_precision) == 0 {
            visitor(&[]);
        }
        return;
    }
    let mut partial = Vec::with_capacity(child_schedules.len());
    visit_combos_recurse(
        child_schedules,
        0,
        0,
        cost_key_with_precision(target, cost_precision),
        &mut partial,
        cost_precision,
        keyed,
        &mut visitor,
    );
}

fn visit_combos_recurse<F: FnMut(&[CostEntry]) -> bool>(
    schedules: &[Arc<Vec<CostEntry>>],
    depth: usize,
    partial_key: u64,
    target_key: u64,
    partial: &mut Vec<CostEntry>,
    precision: f64,
    keyed: bool,
    visitor: &mut F,
) -> bool {
    let last = depth + 1 == schedules.len();
    for entry in schedules[depth].iter() {
        // Both modes add per-child keys, as the count pass does
        // (`convolve_cost_schedules_in` / `convolve_cost_schedules_keyed`);
        // the key of the float sum of child costs can differ from it.
        let q = partial_key.saturating_add(cost_key_with_precision(entry.cost, precision));
        if q > target_key {
            break;
        }
        if last {
            if q == target_key {
                partial.push(entry.clone());
                let keep_going = visitor(partial);
                partial.pop();
                if !keep_going {
                    return false;
                }
                // Legacy keyed enumeration accepted the first matching last
                // child; keyed histograms have a single entry per cost key.
                if keyed {
                    break;
                }
            }
        } else {
            partial.push(entry.clone());
            let keep_going = visit_combos_recurse(
                schedules,
                depth + 1,
                q,
                target_key,
                partial,
                precision,
                keyed,
                visitor,
            );
            partial.pop();
            if !keep_going {
                return false;
            }
        }
    }
    true
}

// ---------------------------------------------------------------------------
// Factored DAG: sampling
// ---------------------------------------------------------------------------

/// Sample n trees uniformly at random (with replacement) from a factored DAG.
///
/// `seed = None` seeds from the system clock (non-reproducible); `Some(s)`
/// makes the draws a deterministic function of `s`. `bound = None` draws from
/// all `tree_count` trees; `Some(b)` draws only among trees whose root cost
/// entry is within `b` (the set counted by `tree_count_at_bound(dag, b)`).
pub fn sample_trees_factored(
    dag: &FactoredDag,
    n: usize,
    feature_names: &[String],
    seed: Option<u64>,
    bound: Option<f64>,
) -> Vec<TreeNode> {
    sample_indices_factored(dag, n, seed, bound)
        .into_iter()
        .map(|idx| get_tree_at_index_factored(dag, idx, feature_names))
        .collect()
}

/// Root-entry filter shared with `tree_count_at_bound`.
pub(crate) fn root_entry_within_bound(dag: &FactoredDag, cost: f64, bound: f64) -> bool {
    let prec = dag.cost_precision;
    if dag.keyed_costs {
        cost_key_with_precision(cost, prec) as i64
            <= scope_key_with_precision(bound, prec, dag.exact_cost_keys)
    } else {
        cost <= bound + SCOPE_EPS
    }
}

/// Global index (for `get_tree_at_index_factored`) of the `r`-th tree among
/// the root cost entries within `bound`, in root-schedule order.
fn index_within_bound(dag: &FactoredDag, bound: f64, r: u128) -> u128 {
    let mut remaining = r;
    let mut offset: u128 = 0;
    for entry in &dag.root_cost_schedule {
        let tc = entry.tree_count();
        if root_entry_within_bound(dag, entry.cost, bound) {
            if remaining < tc {
                return offset + remaining;
            }
            remaining -= tc;
        }
        offset += tc;
    }
    panic!(
        "index {} out of bounds for bound {} (tree_count_at_bound={})",
        r,
        bound,
        tree_count_at_bound(dag, bound)
    );
}

/// Sample n tree indices uniformly at random (with replacement); same
/// `seed` / `bound` semantics and RNG stream as `sample_trees_factored`.
pub fn sample_indices_factored(
    dag: &FactoredDag,
    n: usize,
    seed: Option<u64>,
    bound: Option<f64>,
) -> Vec<u128> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    use std::time::SystemTime;

    let universe = match bound {
        Some(b) => tree_count_at_bound(dag, b),
        None => dag.tree_count,
    };
    if universe == 0 || n == 0 {
        return Vec::new();
    }

    let seed = seed.unwrap_or_else(|| {
        SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64
    });

    let mut state = seed;
    let mut indices = Vec::with_capacity(n);

    for _ in 0..n {
        let mut h = DefaultHasher::new();
        state.hash(&mut h);
        let lo = h.finish();
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);

        let mut h2 = DefaultHasher::new();
        state.hash(&mut h2);
        let hi = h2.finish();
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);

        let raw = ((hi as u128) << 64) | (lo as u128);
        let r = raw % universe;
        indices.push(match bound {
            Some(b) => index_within_bound(dag, b, r),
            None => r,
        });
    }

    indices
}

// ---------------------------------------------------------------------------
// Factored DAG: summary
// ---------------------------------------------------------------------------

pub fn dag_summary_factored(dag: &FactoredDag) -> DagSummary {
    // `n_splits` dispatches on inline vs pooled storage.
    let has_splits = |s: &FactoredSubproblem| n_splits(s) > 0;
    let n_leaf_only = dag
        .subproblems
        .iter()
        .filter(|s| s.leaf.is_some() && !has_splits(s))
        .count();
    let n_internal = dag.subproblems.iter().filter(|s| has_splits(s)).count();
    let n_both = dag
        .subproblems
        .iter()
        .filter(|s| s.leaf.is_some() && has_splits(s))
        .count();

    // Per-edge cost schedules are not stored on
    // splits; they are materialized lazily via `materialize_schedule`.
    let total_split_entries: usize = 0;

    DagSummary {
        n_nodes: dag.subproblems.len(),
        n_internal,
        n_leaf_only,
        n_both,
        total_split_entries,
        tree_count: dag.tree_count,
        n_cost_groups: dag.root_cost_schedule.len(),
        opt: dag.opt,
        rashomon_bound: dag.rashomon_bound,
        theta: dag.theta,
        delta_budget: dag.delta_budget.clone(),
    }
}

// ---------------------------------------------------------------------------
// Symmetry Quotient: merge structurally isomorphic subproblems
// ---------------------------------------------------------------------------

/// Merge structurally isomorphic subproblems in the FactoredDag.
///
/// Two subproblems are isomorphic if they have identical:
/// - leaf cost and prediction (if leaf option exists)
/// - set of split options (same features, arities, child structural hashes)
/// - children cost schedules at each split
///
/// This is lossless: tree counting, sampling, and indexing are exact after
/// quotienting. The tree_count and root_cost_schedule are unchanged.
///
/// Returns the number of subproblems before and after quotienting.
///
/// Structural signatures are computed children-first (post-order DFS):
/// cache-shared children (and widening rebuilds) can sit at lower arena
/// indices than later parents, so the arena order is not topological.
fn quotient_factored_dag_impl(dag: &mut FactoredDag) -> (usize, usize) {
    let n = dag.subproblems.len();
    if n == 0 {
        return (0, 0);
    }

    // Precompute per-inline-leaf structural sigs. The
    // structural hash for an arena leaf was `(1u8, cost, prediction,
    // loss, 0u8 prefix_leaf, 0 splits, tree_count=1)`. We compute the
    // same for each inline leaf so parent subproblems can pull a
    // *stable, content-derived* sig for their tagged children. This
    // keeps the quotient merge behavior of arena-stored leaves, where
    // `leaf_dedup` returned canonical arena refs across the recursion
    // (so multiple parents with structurally-identical leaf children
    // saw the same child sig and merged accordingly).
    let inline_leaf_sigs: Vec<u64> = (0..dag.inline_leaves.len())
        .map(|id| {
            use std::collections::hash_map::DefaultHasher;
            use std::hash::{Hash, Hasher};
            let mut h = DefaultHasher::new();
            // Match the arena-leaf branch of `structural_hash` exactly.
            let cost = dag.inline_leaf_costs[id];
            let li = &dag.inline_leaves[id];
            1u8.hash(&mut h);
            cost.to_bits().hash(&mut h);
            li.prediction.hash(&mut h);
            li.loss.to_bits().hash(&mut h);
            0u8.hash(&mut h); // prefix_leaf None
            0usize.hash(&mut h); // split_hashes.len() == 0
                                 // Include minority info so paired leaves with distinct
                                 // minority class don't collide with single-prediction leaves
                                 // of the same majority info. When NO minority is present only
                                 // tree_count = 1 is mixed in, keeping the default (flag-off)
                                 // quotient unchanged.
            let min_cost_opt = dag
                .inline_leaf_minority_costs
                .get(id)
                .copied()
                .unwrap_or(None);
            let min_leaf_opt = dag.inline_leaves_minority.get(id).and_then(|o| o.as_ref());
            if let (Some(mc), Some(ml)) = (min_cost_opt, min_leaf_opt) {
                // Paired leaf: mix in minority info + tree_count = 2.
                mc.to_bits().hash(&mut h);
                ml.prediction.hash(&mut h);
                ml.loss.to_bits().hash(&mut h);
                2u128.hash(&mut h);
            } else {
                1u128.hash(&mut h); // tree_count = 1
            }
            h.finish()
        })
        .collect();

    // Phase 1: Bottom-up structural hashing, children first (each parent
    // reads its children's already-computed signatures). The inline-leaf
    // branch inside `structural_hash` is unaffected (those refs are tagged
    // and never index into `sig`).
    let mut sig: Vec<u64> = vec![0; n];
    // Exact quotient: vertices are grouped by `sig` and then confirmed
    // structurally equal (`QuotientClasses::assign`); a hash tie with
    // unequal structure opens a new class (counted). Classes are assigned
    // in the same order as `sig`, so a child's class is read exactly when
    // its signature is.
    let inline_leaf_class = inline_leaf_classes(dag);
    let mut classes = QuotientClasses::new(n);
    // Iterative post-order DFS: a node is hashed only after all its
    // arena children (0 = unvisited, 1 = expanded, 2 = hashed).
    let mut state: Vec<u8> = vec![0; n];
    let mut stack: Vec<(usize, bool)> = Vec::new();
    for start in (0..n).rev() {
        if state[start] != 0 {
            continue;
        }
        stack.push((start, false));
        while let Some((i, children_done)) = stack.pop() {
            if children_done {
                sig[i] =
                    structural_hash(&dag.subproblems[i], &sig, &inline_leaf_sigs, &dag.spec_pool);
                classes.assign(
                    i,
                    sig[i],
                    &dag.subproblems,
                    &dag.spec_pool,
                    &inline_leaf_class,
                );
                state[i] = 2;
                continue;
            }
            if state[i] != 0 {
                continue;
            }
            state[i] = 1;
            stack.push((i, true));
            for split in &dag.subproblems[i].splits {
                for &c in &split.children {
                    if !crate::types::is_leaf_ref(c) && state[c as usize] == 0 {
                        stack.push((c as usize, false));
                    }
                }
            }
        }
    }
    dag.n_quotient_hash_collisions += classes.n_collisions;

    // Phase 2: Group by (verified) class, pick representatives.
    // Use the lowest index in each group as representative (preserves topo order).
    let n_classes = classes.n_classes();
    let post_count = n_classes;
    if post_count == n {
        // No duplicates found — nothing to do
        return (n, n);
    }
    let mut class_rep: Vec<FSubRef> = vec![FSubRef::MAX; n_classes];
    for i in 0..n {
        let c = classes.class[i] as usize;
        if class_rep[c] == FSubRef::MAX {
            class_rep[c] = i as FSubRef; // lowest index = representative
        }
    }

    // Phase 3: Build remap table (old FSubRef -> new FSubRef).
    // Representatives are sorted by their original index to preserve topo order.
    let mut representatives: Vec<FSubRef> = class_rep.clone();
    representatives.sort();

    let mut remap: Vec<FSubRef> = vec![0; n];
    let mut new_idx_map: HashMap<FSubRef, FSubRef> = HashMap::new();
    for (new_idx, &old_idx) in representatives.iter().enumerate() {
        new_idx_map.insert(old_idx, new_idx as FSubRef);
    }

    // Map every old index to its representative's new index
    for old_idx in 0..n {
        let rep = class_rep[classes.class[old_idx] as usize];
        remap[old_idx] = new_idx_map[&rep];
    }

    // Phase 4: Build compacted subproblems by MOVING reps out of the
    // original arena (not cloning).
    //
    // A clone-based rebuild would transiently double the
    // `dag.subproblems` heap right before returning the DAG. Instead each
    // original entry is wrapped in `Option<>` (~8 B/entry overhead), each
    // rep is taken with `Option::take`, and the non-rep entries drop
    // when `originals` falls out of scope. Net transient is
    // bounded by `Option<>` overhead, not the doubled heap.
    let originals_raw = std::mem::take(&mut dag.subproblems);
    let mut originals: Vec<Option<FactoredSubproblem>> =
        originals_raw.into_iter().map(Some).collect();
    let mut new_subproblems: Vec<FactoredSubproblem> = Vec::with_capacity(post_count);
    for &rep_idx in &representatives {
        let mut sp = originals[rep_idx as usize]
            .take()
            .expect("quotient: rep_idx visited twice");
        for split in &mut sp.splits {
            for child_ref in &mut split.children {
                // Pass inline-leaf refs through untouched; they
                // don't live in the subproblems arena and don't need remapping.
                if !crate::types::is_leaf_ref(*child_ref) {
                    *child_ref = remap[*child_ref as usize];
                }
            }
        }
        new_subproblems.push(sp);
    }
    drop(originals); // explicit: free non-rep entries before installing new arena

    // Phase 5: Update DAG.
    dag.root_ref = remap[dag.root_ref as usize];
    dag.subproblems = new_subproblems;
    // tree_count and root_cost_schedule are unchanged (they depend on the
    // root subproblem's structure which is preserved by the quotient).

    (n, post_count)
}

/// Exact class of every inline leaf (cost, prediction and loss bits, and
/// the minority option), the inline-leaf part of the quotient equality.
fn inline_leaf_classes(dag: &FactoredDag) -> Vec<u32> {
    let mut ids: HashMap<(u64, u8, u64, Option<(u64, u8, u64)>), u32> = HashMap::new();
    (0..dag.inline_leaves.len())
        .map(|id| {
            let li = &dag.inline_leaves[id];
            let min_cost = dag
                .inline_leaf_minority_costs
                .get(id)
                .copied()
                .unwrap_or(None);
            let min_leaf = dag.inline_leaves_minority.get(id).and_then(|o| o.as_ref());
            let minority = match (min_cost, min_leaf) {
                (Some(mc), Some(ml)) => Some((mc.to_bits(), ml.prediction, ml.loss.to_bits())),
                _ => None,
            };
            let key = (
                dag.inline_leaf_costs[id].to_bits(),
                li.prediction,
                li.loss.to_bits(),
                minority,
            );
            let next = ids.len() as u32;
            *ids.entry(key).or_insert(next)
        })
        .collect()
}

/// Class bookkeeping of `quotient_factored_dag_impl`: `class[v]` (u32::MAX
/// until assigned) and, per structural hash, the classes seen with it and
/// their first member.
struct QuotientClasses {
    class: Vec<u32>,
    by_sig: HashMap<u64, Vec<(u32, usize)>>,
    n_next: u32,
    n_collisions: u64,
}

/// A split of a vertex as compared by the quotient: feature, arity,
/// threshold bits, group_map, child classes (inline leaves tagged 1, arena
/// vertices 0, unassigned u32::MAX) and the split tree_count.
type SplitShape = (u16, u8, Vec<u64>, Option<Vec<u16>>, Vec<(u8, u32)>, u128);

impl QuotientClasses {
    fn new(n: usize) -> Self {
        Self {
            class: vec![u32::MAX; n],
            by_sig: HashMap::new(),
            n_next: 0,
            n_collisions: 0,
        }
    }

    fn n_classes(&self) -> usize {
        self.n_next as usize
    }

    /// Give vertex `v` (structural hash `sig`) the class of the first
    /// vertex with the same hash and an equal structure, else a new class;
    /// a hash shared only with unequal structures counts as a collision.
    fn assign(
        &mut self,
        v: usize,
        sig: u64,
        subproblems: &[FactoredSubproblem],
        spec_pool: &[SplitSpec],
        inline_leaf_class: &[u32],
    ) {
        let mut shape_v: Option<Vec<SplitShape>> = None;
        let found = self.by_sig.get(&sig).and_then(|cands| {
            cands.iter().find_map(|&(c, rep)| {
                if !vertex_payload_eq(&subproblems[v], &subproblems[rep]) {
                    return None;
                }
                let sv = shape_v.get_or_insert_with(|| {
                    self.split_shapes(&subproblems[v], spec_pool, inline_leaf_class)
                });
                (*sv == self.split_shapes(&subproblems[rep], spec_pool, inline_leaf_class))
                    .then_some(c)
            })
        });
        let c = match found {
            Some(c) => c,
            None => {
                let c = self.n_next;
                self.n_next += 1;
                let cands = self.by_sig.entry(sig).or_default();
                if !cands.is_empty() {
                    self.n_collisions += 1;
                }
                cands.push((c, v));
                c
            }
        };
        self.class[v] = c;
    }

    /// The vertex's splits as a sorted multiset (the hash is
    /// order-insensitive over splits too).
    fn split_shapes(
        &self,
        sp: &FactoredSubproblem,
        spec_pool: &[SplitSpec],
        inline_leaf_class: &[u32],
    ) -> Vec<SplitShape> {
        let mut out: Vec<SplitShape> = sp
            .splits
            .iter()
            .map(|split| {
                let spec = &spec_pool[split.spec_id as usize];
                let children = split
                    .children
                    .iter()
                    .map(|&c| {
                        if crate::types::is_leaf_ref(c) {
                            (1u8, inline_leaf_class[crate::types::leaf_id(c)])
                        } else {
                            (0u8, self.class.get(c as usize).copied().unwrap_or(u32::MAX))
                        }
                    })
                    .collect();
                (
                    spec.feature_index,
                    spec.arity,
                    spec.thresholds.iter().map(|t| t.to_bits()).collect(),
                    spec.group_map.as_ref().map(|gm| gm.to_vec()),
                    children,
                    split.tree_count(),
                )
            })
            .collect();
        out.sort();
        out
    }
}

/// Non-split part of the quotient equality: leaf and minority-leaf
/// options (cost, prediction, loss bits; not `n_samples`, as in the hash),
/// prefix-leaf marker (support words, size, depth remaining) and the
/// vertex tree_count.
fn vertex_payload_eq(a: &FactoredSubproblem, b: &FactoredSubproblem) -> bool {
    let leaf_eq = |x: Option<&(f64, RLeafInfo)>, y: Option<&(f64, RLeafInfo)>| match (x, y) {
        (None, None) => true,
        (Some((cx, lx)), Some((cy, ly))) => {
            cx.to_bits() == cy.to_bits()
                && lx.prediction == ly.prediction
                && lx.loss.to_bits() == ly.loss.to_bits()
        }
        _ => false,
    };
    let prefix_eq = match (a.prefix_leaf.as_deref(), b.prefix_leaf.as_deref()) {
        (None, None) => true,
        (Some(x), Some(y)) => {
            x.support.n_samples == y.support.n_samples
                && x.depth_remaining == y.depth_remaining
                && x.support.words[..] == y.support.words[..]
        }
        _ => false,
    };
    leaf_eq(a.leaf.as_ref(), b.leaf.as_ref())
        && leaf_eq(a.leaf_minority.as_deref(), b.leaf_minority.as_deref())
        && prefix_eq
        && a.tree_count == b.tree_count
}

/// Content-address each `FactoredSubproblem.splits` into the
/// per-DAG `split_pool`, replacing the inline `Vec<FactoredSplit>` with a
/// `Vec<SplitId>` (4 B/slot). Must be called AFTER all in-place mutations
/// of `splits` (quotient).
///
/// Typical dedup rate: about a third of splits, which shrinks `splits_vec_cap_bytes` (the per-subproblem
/// FactoredSplit Vec backing) accordingly.
pub fn promote_splits_to_pool(dag: &mut FactoredDag) {
    let mut pool = FactoredSplitPool::new();
    // Pre-size the construction-time lookup table to the expected
    // unique count (typically about a third of intern attempts). This avoids
    // repeated rehash/realloc as the FxHashMap grows — each rehash temporarily
    // pins ~2x memory and slows construction. Total attempts are bounded
    // by the sum of `splits` lengths across all subproblems.
    let total_attempts: usize = dag.subproblems.iter().map(|sp| sp.splits.len()).sum();
    // Reserve at ~40% of attempts (slightly above measured unique rate
    // of ~32-35% so we don't trigger a final rehash at the boundary).
    let expected_unique = (total_attempts * 4 / 10).max(64);
    // Pre-size BOTH the lookup map and the
    // `entries` Vec to the expected final unique count. Without this,
    // `entries` grows by doubling and the realloc-copy transient pins BOTH
    // the old and new backing simultaneously while the per-subproblem
    // `Vec<FactoredSplit>` are still live. Pre-reserving lands a single allocation up front so growth never
    // copies.
    pool.lookup.reserve(expected_unique);
    pool.entries.reserve(expected_unique);
    for sp in dag.subproblems.iter_mut() {
        if sp.splits.is_empty() {
            continue;
        }
        let mut ids: Vec<SplitId> = Vec::with_capacity(sp.splits.len());
        // Drain rather than iter so we move FactoredSplits into the pool
        // without cloning. The Vec backing is freed below.
        let drained = std::mem::take(&mut sp.splits);
        for split in drained.into_iter() {
            let id = pool.intern(split);
            ids.push(id);
        }
        ids.shrink_to_fit();
        sp.split_ids = ids;
        // sp.splits is now Vec::new(); explicitly shrink-to-fit equivalent.
    }
    pool.finalize_drop_lookup();
    dag.split_pool = pool;
}

/// Iterator over (split_index, &FactoredSplit) for a subproblem. After
/// promotion, dispatches via `dag.split_pool`. Before promotion (e.g.,
/// mid-build), falls back to the inline `splits` Vec.
#[inline]
pub fn iter_splits<'a>(
    dag: &'a FactoredDag,
    sub_ref: FSubRef,
) -> Box<dyn Iterator<Item = &'a FactoredSplit> + 'a> {
    let sp = &dag.subproblems[sub_ref as usize];
    if !sp.split_ids.is_empty() {
        Box::new(
            sp.split_ids
                .iter()
                .map(move |&id| &dag.split_pool.entries[id as usize]),
        )
    } else {
        Box::new(sp.splits.iter())
    }
}

/// Number of splits at a subproblem (works pre- or post-promote).
#[inline]
pub fn n_splits(sp: &FactoredSubproblem) -> usize {
    if !sp.split_ids.is_empty() {
        sp.split_ids.len()
    } else {
        sp.splits.len()
    }
}

/// Compute a structural hash for a FactoredSubproblem.
///
/// The hash captures the subproblem's "output signature": everything that
/// determines its behavior for counting/sampling. Two subproblems with
/// identical structural hashes can be merged without affecting any operation.
///
/// `inline_leaf_sigs[id]` is the structural sig of inline-leaf
/// `id` (precomputed by `quotient_factored_dag`, derived from the same
/// hash recipe an arena-stored leaf would have produced). Lookups via the
/// tagged FSubRef in `split.children` resolve to a stable, content-derived
/// child sig, so parents with structurally-identical inline-leaf children
/// merge consistently across the recursion.
///
/// Each split's `group_map` is hashed too (shape mode: same thresholds,
/// different bin→child grouping are different trees on extraction).
///
/// Leaf `n_samples` is deliberately NOT hashed: leaves with equal (cost,
/// prediction, loss) but different support sizes (e.g. pure leaves) still
/// merge. Counts and extracted tree structure are unaffected; only the
/// `n_samples` metadata on an extracted leaf may come from the merged
/// representative.
fn structural_hash(
    sp: &FactoredSubproblem,
    child_sigs: &[u64],
    inline_leaf_sigs: &[u64],
    spec_pool: &[SplitSpec],
) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut h = DefaultHasher::new();

    // Hash leaf option
    match &sp.leaf {
        Some((cost, leaf)) => {
            1u8.hash(&mut h);
            cost.to_bits().hash(&mut h);
            leaf.prediction.hash(&mut h);
            leaf.loss.to_bits().hash(&mut h);
        }
        None => {
            0u8.hash(&mut h);
        }
    }

    // Hash the minority leaf option (count_all_leaves). Keeps
    // quotient honest: two subproblems identical except for the minority
    // prediction/cost must not collapse. Only mixed in when a minority
    // option is present, so the default (flag-off) signatures do not
    // change.
    if let Some((cost, leaf)) = sp.leaf_minority.as_deref() {
        3u8.hash(&mut h);
        cost.to_bits().hash(&mut h);
        leaf.prediction.hash(&mut h);
        leaf.loss.to_bits().hash(&mut h);
    }

    // Hash prefix-leaf marker. Prefix leaves with different captures must NOT
    // be quotient-merged: downstream recursive fill uses the support to launch
    // per-leaf nested solves, so merging would silently lose distinct leaves.
    // Hashing the support fingerprint and depth_remaining keeps the quotient
    // honest while still allowing genuinely identical prefix leaves to share.
    match &sp.prefix_leaf {
        Some(pl) => {
            2u8.hash(&mut h);
            pl.support.fingerprint.hash(&mut h);
            pl.support.n_samples.hash(&mut h);
            pl.depth_remaining.hash(&mut h);
        }
        None => {
            0u8.hash(&mut h);
        }
    }

    // Hash splits in a canonical order (sorted by split hash).
    // Two subproblems with the same set of splits (possibly in different order)
    // must hash identically.
    let mut split_hashes: Vec<u64> = sp
        .splits
        .iter()
        .map(|split| {
            let mut sh = DefaultHasher::new();
            // Resolve the canonical spec_id to its SplitSpec.
            // Hash the SAME fields as before (feature/arity/thresholds) so
            // the quotient equivalence classes are byte-identical to the
            // pre-pool inline-spec behaviour.
            let spec = &spec_pool[split.spec_id as usize];
            spec.feature_index.hash(&mut sh);
            spec.arity.hash(&mut sh);
            // Thresholds matter: same feature + arity but different thresholds
            // produce different children partitions
            for &t in spec.thresholds.iter() {
                t.to_bits().hash(&mut sh);
            }
            match &spec.group_map {
                Some(gm) => {
                    1u8.hash(&mut sh);
                    gm.hash(&mut sh);
                }
                None => 0u8.hash(&mut sh),
            }
            // Schedules are lazy. Hash only the child structural
            // signatures (which capture each child's identity) and the
            // split's aggregate tree_count. Two splits with identical
            // (feature, arity, thresholds, child_sigs in order) are
            // isomorphic for sampling/extraction purposes.
            for &child_ref in &split.children {
                if crate::types::is_leaf_ref(child_ref) {
                    // With the reverse-iteration topo
                    // order, child sigs are populated before parents.
                    // Inline leaves don't live in the arena, but their
                    // structural identity (cost, prediction, loss) is
                    // precomputed in `inline_leaf_sigs[leaf_id]`. Hash
                    // that instead of 0u64 so quotient distinguishes
                    // parents that wrap different leaves (a constant
                    // hash would merge them).
                    let id = crate::types::leaf_id(child_ref);
                    inline_leaf_sigs[id].hash(&mut sh);
                } else {
                    child_sigs[child_ref as usize].hash(&mut sh);
                }
            }
            split.hash_tree_count(&mut sh);
            sh.finish()
        })
        .collect();
    split_hashes.sort();

    split_hashes.len().hash(&mut h);
    for sh in &split_hashes {
        sh.hash(&mut h);
    }

    sp.tree_count.hash(&mut h);

    h.finish()
}

#[cfg(test)]
mod schedule_regressions {
    use super::*;

    fn schedule(values: &[(f64, u128)]) -> Arc<Vec<CostEntry>> {
        Arc::new(values.iter().map(|&(c, n)| CostEntry::new(c, n)).collect())
    }

    #[test]
    fn residual_pruning_preserves_exhaustive_cost_histogram() {
        let a = schedule(&[(1.0, 2), (2.0, 3), (3.0, 1)]);
        let b = schedule(&[(1.0, 5), (2.0, 7)]);
        let c = schedule(&[(1.0, 11), (2.0, 13)]);
        let children = [a.as_slice(), b.as_slice(), c.as_slice()];
        let mut expected = BTreeMap::<i64, u128>::new();
        for x in a.iter() {
            for y in b.iter() {
                for z in c.iter() {
                    let sum = (x.cost + y.cost + z.cost) as i64;
                    if sum <= 6 {
                        *expected.entry(sum).or_default() +=
                            x.tree_count() * y.tree_count() * z.tree_count();
                    }
                }
            }
        }
        let actual: BTreeMap<_, _> = convolve_cost_schedules_keyed(&children, 60, 10.0)
            .into_iter()
            .map(|e| ((e.cost as i64), e.tree_count()))
            .collect();
        assert_eq!(actual, expected);
        assert!(convolve_cost_schedules_keyed(&children, 29, 10.0).is_empty());
    }

    #[test]
    fn combo_visitor_matches_convolve_key_sums() {
        let children = [
            schedule(&[(0.16, 1)]),
            schedule(&[(0.16, 1)]),
            schedule(&[(0.16, 1)]),
        ];
        let refs: Vec<&[CostEntry]> = children.iter().map(|a| a.as_slice()).collect();
        let conv = convolve_cost_schedules(&refs, 1.0, 10.0);
        assert_eq!(conv.len(), 1);
        assert_eq!(cost_key_with_precision(conv[0].cost, 10.0), 6);
        for keyed in [false, true] {
            let mut n = 0;
            visit_combos_at_target(&children, conv[0].cost, 10.0, keyed, |_| {
                n += 1;
                true
            });
            assert_eq!(n, 1, "keyed={keyed}");
        }
    }

    #[test]
    fn streamed_tuples_preserve_canonical_order_and_stop_early() {
        let children = [
            schedule(&[(1.0, 2), (2.0, 3), (3.0, 1)]),
            schedule(&[(1.0, 5), (2.0, 7)]),
            schedule(&[(1.0, 11), (2.0, 13)]),
        ];
        let expected = vec![vec![1, 2, 2], vec![2, 1, 2], vec![2, 2, 1], vec![3, 1, 1]];
        for keyed in [false, true] {
            let mut tuples = Vec::new();
            visit_combos_at_target(&children, 5.0, 10.0, keyed, |combo| {
                tuples.push(combo.iter().map(|e| e.cost as i64).collect::<Vec<_>>());
                true
            });
            assert_eq!(tuples, expected);
            let mut visited = 0;
            visit_combos_at_target(&children, 5.0, 10.0, keyed, |_| {
                visited += 1;
                false
            });
            assert_eq!(visited, 1);
        }
    }
}
