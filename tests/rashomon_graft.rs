//! Step B on hand-built DAGs: the prefix-cost DP and per-placeholder scope
//! (incl. a two-placeholder example), and AttachRecursive graft + recompute vs an
//! independently built full-depth DAG.
//! Also: save/load round trips (grafted DAGs are not children-first in
//! construction; a full build reserves parents before children) and the
//! compaction trim (every remaining vertex lies on a tree within beta).
//! Every hand-built DAG interns its split specs in its own `SplitSpecPool`
//! (`Specs`), so the graft's re-interning of nested spec pools into the
//! outer pool is exercised (keys name the feature through the spec).
//!   cargo test --release --no-default-features --test rashomon_graft

use std::collections::BTreeMap;
use std::sync::Arc;

use dashmap::DashMap;
use multiway_solver::factored_dag_io::{
    deserialize_factored_dag, serialize_factored_dag, serialize_factored_dag_version,
};
use multiway_solver::rashomon_dag::{
    get_tree_at_index_factored, materialize_schedule, stepb_graft, stepb_prefix_costs,
    stepb_recompute, stepb_scope,
};
use multiway_solver::tree::TreeNode;
use multiway_solver::types::{
    CostEntry, DeltaBudget, FactoredDag, FactoredSplit, FactoredSplitPool, FactoredSubproblem,
    PrefixLeafInfo, RLeafInfo, SplitSpec, SplitSpecPool, SupportMask, MAX_CLASSES,
};

/// Local spec interner of one hand-built DAG.
type Specs = SplitSpecPool;

fn bare(leaf: Option<(f64, RLeafInfo)>, tree_count: u128, min_obj: f64) -> FactoredSubproblem {
    FactoredSubproblem {
        leaf,
        leaf_minority: None,
        prefix_leaf: None,
        splits: Vec::new(),
        split_ids: Vec::new(),
        tree_count,
        min_obj,
        cost_entries: None,
    }
}

fn leaf(cost: f64, pred: u8) -> FactoredSubproblem {
    bare(
        Some((
            cost,
            RLeafInfo {
                prediction: pred,
                loss: cost,
                n_samples: 1,
            },
        )),
        1,
        cost,
    )
}

fn split(specs: &mut Specs, feature: u16, children: &[u32]) -> FactoredSplit {
    let spec = SplitSpec {
        feature_index: feature,
        arity: children.len() as u8,
        thresholds: Arc::from(vec![0.5 + feature as f64; children.len() - 1]),
        group_map: None,
    };
    FactoredSplit::new(specs.intern(&spec), children.to_vec().into(), 0)
}

fn node(leaf_cost: Option<(f64, u8)>, splits: Vec<FactoredSplit>) -> FactoredSubproblem {
    let mut sp = match leaf_cost {
        Some((c, p)) => leaf(c, p),
        None => bare(None, 0, f64::INFINITY),
    };
    sp.splits = splits;
    sp
}

/// Depth-0 placeholder priced at `lb` (Step A provisional schedule
/// [(lb, 1)]); `word` makes its support distinct.
fn placeholder(lb: f64, word: u64) -> FactoredSubproblem {
    let mut sp = bare(None, 1, lb);
    sp.prefix_leaf = Some(Box::new(PrefixLeafInfo {
        support: SupportMask::from_words(vec![word], word.count_ones() as usize),
        class_counts: [0i64; MAX_CLASSES],
        n_classes: 2,
        n_samples: word.count_ones() as usize,
        depth_remaining: 1,
        greedy_loss: lb,
    }));
    sp
}

fn mk_dag(
    subproblems: Vec<FactoredSubproblem>,
    specs: Specs,
    beta: f64,
    keyed: bool,
    precision: f64,
) -> FactoredDag {
    FactoredDag {
        subproblems,
        root_ref: 0,
        root_cost_schedule: Vec::new(),
        tree_count: 0,
        opt: 0.0,
        rashomon_bound: beta,
        theta: 0.0,
        reg: 0.0,
        delta_budget: DeltaBudget::default(),
        cost_precision: precision,
        keyed_costs: keyed,
        exact_cost_keys: keyed,
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
        schedule_cache: Arc::new(DashMap::new()),
        wide_schedule_cache: Default::default(),
        split_pool: FactoredSplitPool::new(),
        spec_pool: specs.into_pool(),
        inline_leaves: Vec::new(),
        inline_leaf_costs: Vec::new(),
        inline_leaves_minority: Vec::new(),
        inline_leaf_minority_costs: Vec::new(),
        n_quotient_hash_collisions: 0,
        count_saturated: false,
    }
}

/// Independent min_obj (vertices listed children-after-parents is not
/// assumed: fixpoint over the arena).
fn set_min_obj(dag: &mut FactoredDag) {
    loop {
        let mut changed = false;
        for v in 0..dag.subproblems.len() {
            let mut m = dag.subproblems[v]
                .leaf
                .as_ref()
                .map(|l| l.0)
                .unwrap_or(f64::INFINITY);
            for s in &dag.subproblems[v].splits {
                let sum: f64 = s
                    .children
                    .iter()
                    .map(|&c| dag.subproblems[c as usize].min_obj)
                    .sum();
                m = m.min(sum);
            }
            if m != dag.subproblems[v].min_obj {
                dag.subproblems[v].min_obj = m;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
}

fn key(t: &TreeNode) -> String {
    match t {
        TreeNode::Leaf {
            prediction, loss, ..
        } => format!("L{}:{:.4}", prediction, loss),
        TreeNode::Internal {
            feature_index,
            children,
            ..
        } => format!(
            "N{}({})",
            feature_index,
            children.iter().map(key).collect::<Vec<_>>().join(",")
        ),
    }
}

/// Brute force: every tree of the DAG below `v` as (cost, key).
fn brute(dag: &FactoredDag, v: u32) -> Vec<(f64, String)> {
    let sp = &dag.subproblems[v as usize];
    let mut out = Vec::new();
    if let Some((c, l)) = &sp.leaf {
        out.push((*c, format!("L{}:{:.4}", l.prediction, l.loss)));
    }
    for s in &sp.splits {
        let mut partial: Vec<(f64, Vec<String>)> = vec![(0.0, Vec::new())];
        for &c in &s.children {
            let sub = brute(dag, c);
            let mut next = Vec::new();
            for (a, ks) in &partial {
                for (b, k) in &sub {
                    let mut ks2 = ks.clone();
                    ks2.push(k.clone());
                    next.push((a + b, ks2));
                }
            }
            partial = next;
        }
        for (a, ks) in partial {
            let f = dag.spec_of(s).feature_index;
            out.push((a, format!("N{}({})", f, ks.join(","))));
        }
    }
    out
}

fn extracted(dag: &FactoredDag) -> Vec<String> {
    (0..dag.tree_count)
        .map(|i| key(&get_tree_at_index_factored(dag, i, &[])))
        .collect()
}

fn schedule(entries: &[CostEntry], precision: f64) -> BTreeMap<i64, u128> {
    let mut m = BTreeMap::new();
    for e in entries {
        *m.entry((e.cost * precision).round() as i64).or_insert(0) += e.tree_count();
    }
    m
}

/// Two placeholders: a root split into placeholders u, w with lb_u = 0.10,
/// lb_w = 0.05, opt = 0.15, beta = 0.1575. Scoping u at beta minus the
/// whole prefix cost, 0.1575 - 0.15 = 0.0075 < lb_u, would drop the
/// optimum; beta_u = beta - pc(u) = 0.1575 - 0.05 = 0.1075 admits it.
fn two_placeholder_scopes(keyed: bool) {
    let precision = if keyed { 400.0 } else { 1e9 };
    let beta = 0.1575;
    let mut sp = Specs::new();
    let root = node(None, vec![split(&mut sp, 0, &[1, 2])]);
    let mut dag = mk_dag(
        vec![root, placeholder(0.10, 0b11), placeholder(0.05, 0b100)],
        sp,
        beta,
        keyed,
        precision,
    );
    let pc = stepb_prefix_costs(&dag, &|_| 0.0);
    assert!(
        (pc[0] - 0.0).abs() < 1e-12 && (pc[1] - 0.05).abs() < 1e-12 && (pc[2] - 0.10).abs() < 1e-12
    );
    let bu = stepb_scope(&dag, &pc, 1).expect("optimum's T_u admitted");
    let bw = stepb_scope(&dag, &pc, 2).expect("optimum's T_w admitted");
    assert!((bu - 0.1075).abs() < 1e-9, "beta_u = {}", bu);
    assert!((bw - 0.0575).abs() < 1e-9, "beta_w = {}", bw);
    // Complete u and w with their optimal subtrees (a leaf each) and check
    // the optimum (cost 0.15) is in the grafted set.
    let nu = {
        let mut d = mk_dag(vec![leaf(0.10, 0)], Specs::new(), bu, keyed, precision);
        d.tree_count = 1;
        d
    };
    let nw = {
        let mut d = mk_dag(vec![leaf(0.05, 1)], Specs::new(), bw, keyed, precision);
        d.tree_count = 1;
        d
    };
    assert_eq!(
        stepb_graft(&mut dag, vec![(vec![1], nu), (vec![2], nw)], &[]),
        2
    );
    stepb_recompute(&mut dag);
    assert_eq!(dag.tree_count, 1);
    assert!((dag.root_cost_schedule[0].cost - 0.15).abs() < 1e-9);
    assert_eq!(extracted(&dag), vec!["N0(L0:0.1000,L1:0.0500)".to_string()]);
}

#[test]
fn stepb_prefix_cost_two_placeholders_float() {
    two_placeholder_scopes(false);
}

#[test]
fn stepb_prefix_cost_two_placeholders_exact_keys() {
    two_placeholder_scopes(true);
}

/// Full-depth DAG F (vertex ids):
///   0 root: leaf 0.5 | f0[1,2] | f3[2,7] | f4[1,8]
///   1 a: leaf 0.3 | f1[3,4]     2 b: leaf 0.2 | f2[5,6]
///   7 c: leaf 0.25 | f1[3,3]    8 z: leaf 0.4
///   3: 0.05 (L0)  4: 0.10 (L1)  5: 0.05 (L2)  6: 0.05 (L3)
fn full_dag(beta: f64, keyed: bool, precision: f64) -> FactoredDag {
    let mut sp = Specs::new();
    let root_splits = vec![
        split(&mut sp, 0, &[1, 2]),
        split(&mut sp, 3, &[2, 7]),
        split(&mut sp, 4, &[1, 8]),
    ];
    let a = vec![split(&mut sp, 1, &[3, 4])];
    let b = vec![split(&mut sp, 2, &[5, 6])];
    let c = vec![split(&mut sp, 1, &[3, 3])];
    let mut d = mk_dag(
        vec![
            node(Some((0.5, 9)), root_splits),
            node(Some((0.3, 4)), a),
            node(Some((0.2, 5)), b),
            leaf(0.05, 0),
            leaf(0.10, 1),
            leaf(0.05, 2),
            leaf(0.05, 3),
            node(Some((0.25, 6)), c),
            leaf(0.4, 7),
        ],
        sp,
        beta,
        keyed,
        precision,
    );
    set_min_obj(&mut d);
    d
}

/// Same trees through a depth-1 prefix: placeholders (lb 0.1 / 0.4) at the
/// root's children; b appears twice (vertices 2 and 3, same support) and
/// gets one shared nested DAG; vertex 7 is an unreachable placeholder.
fn grafted_dag(beta: f64, keyed: bool, precision: f64) -> FactoredDag {
    // Outer specs interned in a different order from the nested DAGs'.
    let mut sp = Specs::new();
    let root_splits = vec![
        split(&mut sp, 4, &[5, 6]),
        split(&mut sp, 3, &[3, 4]),
        split(&mut sp, 0, &[1, 2]),
    ];
    let mut p = mk_dag(
        vec![
            node(Some((0.5, 9)), root_splits),
            placeholder(0.1, 0b1),      // a
            placeholder(0.1, 0b10),     // b
            placeholder(0.1, 0b10),     // b (same support)
            placeholder(0.1, 0b100),    // c
            placeholder(0.1, 0b1000),   // a again under f4 (pc 0.4: beyond scope)
            placeholder(0.4, 0b10000),  // z (lb 0.4 > beta - 0.1)
            placeholder(0.1, 0b100000), // unreachable
        ],
        sp,
        beta,
        keyed,
        precision,
    );
    let pc = stepb_prefix_costs(&p, &|_| 0.0);
    let scopes: Vec<Option<f64>> = (0..p.subproblems.len())
        .map(|v| stepb_scope(&p, &pc, v))
        .collect();
    assert!(
        scopes[1].is_some() && scopes[2].is_some() && scopes[3].is_some() && scopes[4].is_some()
    );
    assert!(scopes[7].is_none(), "{:?}", scopes);
    // f4 = [a, z] has lb sum 0.5: within scope only for beta >= 0.5.
    assert_eq!(
        scopes[5].is_some() && scopes[6].is_some(),
        beta >= 0.5,
        "{:?}",
        scopes
    );
    // One nested DAG: a root (leaf + one split on `feature`) over `kids`.
    let mk_nested =
        |root_leaf: (f64, u8), feature: u16, ch: &[u32], kids: Vec<FactoredSubproblem>, b: f64| {
            let mut specs = Specs::new();
            let mut subs = vec![node(Some(root_leaf), vec![split(&mut specs, feature, ch)])];
            subs.extend(kids);
            let mut d = mk_dag(subs, specs, b, keyed, precision);
            set_min_obj(&mut d);
            d.tree_count = 1; // non-empty; counts are recomputed after the graft
            d
        };
    let na = mk_nested(
        (0.3, 4),
        1,
        &[1, 2],
        vec![leaf(0.05, 0), leaf(0.10, 1)],
        scopes[1].unwrap(),
    );
    let nb = mk_nested(
        (0.2, 5),
        2,
        &[1, 2],
        vec![leaf(0.05, 2), leaf(0.05, 3)],
        scopes[2].unwrap().max(scopes[3].unwrap()),
    );
    let nc = mk_nested(
        (0.25, 6),
        1,
        &[1, 1],
        vec![leaf(0.05, 0)],
        scopes[4].unwrap(),
    );
    let mut grafts = vec![(vec![1], na), (vec![2, 3], nb), (vec![4], nc)];
    let mut empty = Vec::new();
    for v in [5usize, 6] {
        match scopes[v] {
            None => empty.push(v),
            Some(b) => grafts.push((
                vec![v],
                if v == 5 {
                    mk_nested((0.3, 4), 1, &[1, 2], vec![leaf(0.05, 0), leaf(0.10, 1)], b)
                } else {
                    let mut d = mk_dag(vec![leaf(0.4, 7)], Specs::new(), b, keyed, precision);
                    d.tree_count = 1;
                    d
                },
            )),
        }
    }
    let n_filled = stepb_graft(&mut p, grafts, &empty);
    assert_eq!(n_filled, 6 - empty.len());
    stepb_recompute(&mut p);
    p
}

fn graft_vs_full(keyed: bool) {
    let precision = if keyed { 20.0 } else { 1e9 };
    for &beta in &[0.2, 0.25, 0.3, 0.45, 0.6, 1.0] {
        let full = full_dag(beta, keyed, precision);
        let g = grafted_dag(beta, keyed, precision);
        assert!(g.subproblems.iter().all(|sp| sp.prefix_leaf.is_none()));
        // Root schedule: grafted == full-depth materialization == brute force.
        let full_root = materialize_schedule(&full, 0, beta);
        let mut brute_sched: BTreeMap<i64, u128> = BTreeMap::new();
        let mut brute_keys: Vec<String> = Vec::new();
        for (c, k) in brute(&full, 0) {
            if c <= beta + 1e-9 {
                *brute_sched
                    .entry((c * precision).round() as i64)
                    .or_insert(0) += 1;
                brute_keys.push(k);
            }
        }
        assert_eq!(
            schedule(&g.root_cost_schedule, precision),
            brute_sched,
            "beta={}",
            beta
        );
        assert_eq!(
            schedule(&full_root, precision),
            brute_sched,
            "beta={}",
            beta
        );
        assert_eq!(g.tree_count, brute_keys.len() as u128, "beta={}", beta);
        let mut got = extracted(&g);
        got.sort();
        brute_keys.sort();
        assert_eq!(got, brute_keys, "beta={}", beta);
    }
}

#[test]
fn stepb_graft_recompute_equals_full_depth_float() {
    graft_vs_full(false);
}

#[test]
fn stepb_graft_recompute_equals_full_depth_exact_keys() {
    graft_vs_full(true);
}

/// Every root tree of the DAG as (cost, vertices used).
fn brute_vertices(dag: &FactoredDag, v: u32) -> Vec<(f64, Vec<u32>)> {
    let sp = &dag.subproblems[v as usize];
    let mut out = Vec::new();
    if let Some((c, _)) = &sp.leaf {
        out.push((*c, vec![v]));
    }
    for s in &sp.splits {
        let mut partial: Vec<(f64, Vec<u32>)> = vec![(0.0, vec![v])];
        for &c in &s.children {
            let sub = brute_vertices(dag, c);
            let mut next = Vec::new();
            for (a, vs) in &partial {
                for (b, ws) in &sub {
                    let mut u = vs.clone();
                    u.extend_from_slice(ws);
                    next.push((a + b, u));
                }
            }
            partial = next;
        }
        out.extend(partial);
    }
    out
}

/// Save/load (default v5 bytes, and legacy v4) keeps the tree set: same
/// count, same extracted trees in the same index order.
fn assert_roundtrip(dag: &FactoredDag, label: &str) {
    let v5 = serialize_factored_dag(dag, &[], &[], None);
    let v4 = serialize_factored_dag_version(dag, &[], &[], None, 4).expect("v4");
    for (fmt, bytes) in [("v5", v5), ("v4", v4)] {
        let (loaded, _, _, _) = deserialize_factored_dag(&bytes).expect("deserialize");
        assert_eq!(loaded.tree_count, dag.tree_count, "{} {}", label, fmt);
        assert_eq!(extracted(&loaded), extracted(dag), "{} {}", label, fmt);
    }
}

fn stepb_roundtrip(keyed: bool) {
    let precision = if keyed { 20.0 } else { 1e9 };
    for &beta in &[0.2, 0.25, 0.3, 0.45, 0.6, 1.0] {
        let g = grafted_dag(beta, keyed, precision);
        assert!(g.tree_count > 0);
        // Children-first after stepb_recompute.
        for (i, sp) in g.subproblems.iter().enumerate() {
            for s in &sp.splits {
                assert!(
                    s.children.iter().all(|&c| (c as usize) < i),
                    "beta={} v={}",
                    beta,
                    i
                );
            }
        }
        assert_roundtrip(&g, &format!("grafted beta={}", beta));
        // A parents-first arena (full_dag: root 0, children after it).
        let mut f = full_dag(beta, keyed, precision);
        let root: Vec<CostEntry> = materialize_schedule(&f, 0, beta)
            .iter()
            .filter(|e| e.cost <= beta + 1e-9)
            .cloned()
            .collect();
        f.tree_count = root.iter().map(|e| e.tree_count()).sum();
        f.root_cost_schedule = root;
        f.schedule_cache = Arc::new(DashMap::new());
        assert_roundtrip(&f, &format!("full beta={}", beta));
    }
}

#[test]
fn stepb_dag_save_load_roundtrip_float() {
    stepb_roundtrip(false);
}

#[test]
fn stepb_dag_save_load_roundtrip_exact_keys() {
    stepb_roundtrip(true);
}

/// Compaction trim (keyed DAGs): after stepb_recompute every vertex of
/// the arena lies on some root tree with cost <= beta.
#[test]
fn stepb_trim_keeps_only_vertices_within_beta() {
    let precision = 20.0;
    for &beta in &[0.2, 0.25, 0.3, 0.45, 0.6, 1.0] {
        let g = grafted_dag(beta, true, precision);
        let mut used = vec![false; g.subproblems.len()];
        for (c, vs) in brute_vertices(&g, g.root_ref) {
            if c <= beta + 1e-9 {
                for v in vs {
                    used[v as usize] = true;
                }
            }
        }
        assert!(used.iter().all(|&u| u), "beta={} used={:?}", beta, used);
    }
}
