//! Rashomon count fixes in the full-build factored DAG, on
//! hand-built memos:
//!   cargo test --release --no-default-features --test rashomon_count_fix

use std::collections::HashMap;
use std::sync::Arc;

use multiway_solver::rashomon_dag::{
    build_factored_dag, get_tree_at_index_factored, tree_count_at_bound,
};
use multiway_solver::types::{
    DeltaBudget, FeatureMask, MemoKey, MemoKeyArena, SplitRecord, SplitRecordPool, SubProblem,
    SupportMask, MAX_CLASSES,
};

fn key(bits: u64) -> MemoKey {
    MemoKey {
        support: SupportMask::from_words(vec![bits], bits.count_ones() as usize),
        feature_mask: FeatureMask::all(2),
        depth: 1,
        mode_flags: 0,
        teacher_alpha_bits: 1.0f64.to_bits(),
    }
}

/// Subproblem whose leaf costs `leaf` (use a large value for "no leaf")
/// with lower bound `lb` and the given Rashomon split indices.
fn sub(bits: u64, leaf: f64, lb: f64, rashomon_indices: Vec<u32>) -> SubProblem {
    let mut class_counts = [0i64; MAX_CLASSES];
    class_counts[0] = bits.count_ones() as i64;
    SubProblem {
        support: key(bits).support,
        split_path: Vec::new(),
        feature_mask: FeatureMask::all(2),
        n_samples: bits.count_ones() as usize,
        class_counts,
        n_classes: 2,
        max_loss: 0.0,
        base_objective: leaf,
        lowerbound: lb,
        upperbound: leaf,
        upperbound_is_heuristic: false,
        phase_a_lb_floor: f64::NEG_INFINITY,
        last_survivor_count: 0,
        all_survivors_materialized: true,
        is_terminal: rashomon_indices.is_empty(),
        is_solved: true,
        branch_bound_scope_floor: f64::NEG_INFINITY,
        scope: f64::INFINITY,
        explored: true,
        depth: 1,
        bound_update_pending: false,
        rashomon_indices,
    }
}

fn split(id: u64, feature: usize, th: &[f64], children: Vec<u32>, upper: f64) -> SplitRecord {
    SplitRecord {
        split_id: id,
        feature_index: feature,
        thresholds: Arc::from(th.to_vec().into_boxed_slice()),
        child_keys: children,
        split_upper: upper,
        contextual_lowerbound: f64::NEG_INFINITY,
        contextually_dead: false,
        group_map: None,
    }
}

fn build(
    memo: &HashMap<MemoKey, SubProblem>,
    pool: &SplitRecordPool,
    arena: &MemoKeyArena,
    root: &MemoKey,
    bound: f64,
) -> multiway_solver::types::FactoredDag {
    build_with_precision(memo, pool, arena, root, bound, 1e9)
}

fn build_with_precision(
    memo: &HashMap<MemoKey, SubProblem>,
    pool: &SplitRecordPool,
    arena: &MemoKeyArena,
    root: &MemoKey,
    bound: f64,
    precision: f64,
) -> multiway_solver::types::FactoredDag {
    build_factored_dag(
        memo,
        pool,
        arena,
        root,
        bound,
        0.0,
        0.0,
        0.0,
        DeltaBudget::default(),
        precision,
        None,
        2,
        2,
    )
}

/// Shared child requested first at a narrow scope, then at a wide one.
/// w has trees at 0.2, 0.5, 0.6;
/// r = (w,x) | (w,y) with x = 0.6, y = 0.1. Under bound 1.0 the true
/// set is {0.8, 0.3, 0.6, 0.7}: 4 trees. The legacy cache served w's
/// schedule built at scope 0.4 to the (w,y) request at 0.9 → 2 trees.
#[test]
fn shared_child_cache_scope() {
    let arena = MemoKeyArena::new();
    let mut pool = SplitRecordPool::new();
    let (kp, kq, ks, kw, kx, ky, kr) = (key(1), key(2), key(4), key(8), key(16), key(32), key(64));
    let (ip, iq, is, iw, ix, iy) = (
        arena.intern(kp.clone()),
        arena.intern(kq.clone()),
        arena.intern(ks.clone()),
        arena.intern(kw.clone()),
        arena.intern(kx.clone()),
        arena.intern(ky.clone()),
    );
    let w1 = pool.push(split(1, 0, &[0.5], vec![ip, iq], 0.5));
    let w2 = pool.push(split(2, 0, &[1.5], vec![ip, is], 0.6));
    let r1 = pool.push(split(3, 1, &[0.5], vec![iw, ix], 0.8));
    let r2 = pool.push(split(4, 1, &[1.5], vec![iw, iy], 0.3));
    let mut memo = HashMap::new();
    memo.insert(kp, sub(1, 0.25, 0.25, vec![]));
    memo.insert(kq, sub(2, 0.25, 0.25, vec![]));
    memo.insert(ks, sub(4, 0.35, 0.35, vec![]));
    memo.insert(kw, sub(8, 0.2, 0.2, vec![w1, w2]));
    memo.insert(kx, sub(16, 0.6, 0.6, vec![]));
    memo.insert(ky, sub(32, 0.1, 0.1, vec![]));
    memo.insert(kr.clone(), sub(64, 10.0, 0.3, vec![r1, r2]));

    let dag = build(&memo, &pool, &arena, &kr, 1.0);
    let costs: Vec<(u64, u128)> = dag
        .root_cost_schedule
        .iter()
        .map(|e| ((e.cost * 1e3).round() as u64, e.tree_count()))
        .collect();
    assert_eq!(costs, vec![(300, 1), (600, 1), (700, 1), (800, 1)]);
    assert_eq!(dag.tree_count, 4);
    let names = vec!["f0".to_string(), "f1".to_string()];
    for i in 0..dag.tree_count {
        let _ = get_tree_at_index_factored(&dag, i, &names); // must not panic
    }
    // tree_count_at_target: a reporting bound below the exploration bound
    // keeps only the root entries at or under it (0.3, 0.6).
    assert_eq!(tree_count_at_bound(&dag, 0.65), 2);
    assert_eq!(tree_count_at_bound(&dag, 0.6), 2);
    assert_eq!(tree_count_at_bound(&dag, 1.0), dag.tree_count);
}

/// One split pushed twice (same pool index), once more under another
/// split_upper (new pool index, same identity), plus a granularity variant
/// (different threshold, same children). Dedup keeps exactly two splits:
/// the identity and the threshold variant.
#[test]
fn duplicate_split_indices_become_one_split() {
    let arena = MemoKeyArena::new();
    let mut pool = SplitRecordPool::new();
    let (ka, kb, kr) = (key(1), key(2), key(4));
    let (ia, ib) = (arena.intern(ka.clone()), arena.intern(kb.clone()));
    let s = pool.push(split(7, 0, &[0.5], vec![ia, ib], 0.2));
    let s_upper = pool.push(split(7, 0, &[0.5], vec![ia, ib], 0.25));
    assert_ne!(s, s_upper, "pool keys on (split_id, split_upper)");
    let s_variant = pool.push(split(8, 0, &[1.5], vec![ia, ib], 0.2));
    let mut memo = HashMap::new();
    memo.insert(ka, sub(1, 0.1, 0.1, vec![]));
    memo.insert(kb, sub(2, 0.1, 0.1, vec![]));
    memo.insert(kr.clone(), sub(4, 0.3, 0.2, vec![s, s, s_upper, s_variant]));

    let dag = build(&memo, &pool, &arena, &kr, 1.0);
    let root = &dag.subproblems[dag.root_ref as usize];
    assert_eq!(root.splits.len(), 2);
    // leaf (0.3) + split at 0.5 + split at 1.5 (both cost 0.2)
    assert_eq!(dag.tree_count, 3);
}

/// Scope key at precision 1 (integer costs): bound 315 admits cost 315
/// and nothing above. Root leaf 316 and split (a, c) = 150 + 166 = 316
/// are out; split (a, b) = 150 + 165 = 315 is in. A `ceil` scope key
/// (316) admitted all three.
#[test]
fn scope_key_precision_one_admits_nothing_above_bound() {
    let arena = MemoKeyArena::new();
    let mut pool = SplitRecordPool::new();
    let (ka, kb, kc, kr) = (key(1), key(2), key(4), key(8));
    let (ia, ib, ic) = (
        arena.intern(ka.clone()),
        arena.intern(kb.clone()),
        arena.intern(kc.clone()),
    );
    let s_ab = pool.push(split(1, 0, &[0.5], vec![ia, ib], 315.0));
    let s_ac = pool.push(split(2, 1, &[0.5], vec![ia, ic], 316.0));
    let mut memo = HashMap::new();
    memo.insert(ka, sub(1, 150.0, 150.0, vec![]));
    memo.insert(kb, sub(2, 165.0, 165.0, vec![]));
    memo.insert(kc, sub(4, 166.0, 166.0, vec![]));
    memo.insert(kr.clone(), sub(8, 316.0, 315.0, vec![s_ab, s_ac]));

    let dag = build_with_precision(&memo, &pool, &arena, &kr, 315.0, 1.0);
    let costs: Vec<(f64, u128)> = dag
        .root_cost_schedule
        .iter()
        .map(|e| (e.cost, e.tree_count()))
        .collect();
    assert_eq!(costs, vec![(315.0, 1)]);
    assert_eq!(dag.tree_count, 1);
    let names = vec!["f0".to_string(), "f1".to_string()];
    let _ = get_tree_at_index_factored(&dag, 0, &names); // must not panic
}
