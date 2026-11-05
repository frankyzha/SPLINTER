//! Exact per-row class counts (`dag_predict_exact::class_counts`) vs a brute
//! force over `get_tree_at_index_factored` on keyed exact-key DAGs, at the
//! full bound and a narrower target; and
//! the u128 sum identity / closed form on a 7-level chain (> 2^64 trees).
//!   cargo test --release --no-default-features --test rashomon_predict_exact

use std::sync::Arc;

use multiway_solver::dag_predict_exact::{class_counts, ExactPredictOpts};
use multiway_solver::rashomon_dag::{
    count_chain_fixture, get_tree_at_index_factored, recount_at_bound, tree_count_at_bound,
};
use multiway_solver::tree::TreeNode;
use multiway_solver::types::{FSubRef, FactoredDag, FactoredSplit, SplitSpec};

/// As `tests/rashomon_sched_wide.rs::keyed_dag` (lattice step 0.01, leaf
/// predictions i % 2, a binding bound); with `two_features` the extra splits
/// split on feature 1, so rows route differently per split.
fn keyed_dag(levels: usize, bound: f64, two_features: bool) -> FactoredDag {
    let mut dag = count_chain_fixture(levels, 0.01);
    let spec0 = dag.subproblems[1].splits[0].spec_id;
    let spec1 = if two_features {
        dag.spec_pool.push(SplitSpec {
            feature_index: 1,
            arity: 2,
            thresholds: Arc::from(vec![0.5]),
            group_map: None,
        });
        (dag.spec_pool.len() - 1) as _
    } else {
        spec0
    };
    for i in 0..=levels {
        let sp = &mut dag.subproblems[i];
        if let Some((c, _)) = sp.leaf.as_mut() {
            *c = 0.01 * (1 + i % 3) as f64;
        }
        if i >= 2 {
            let (a, b) = ((i - 1) as FSubRef, (i - 2) as FSubRef);
            sp.splits
                .push(FactoredSplit::new(spec1, [a, b].into_iter().collect(), 0));
            sp.splits
                .push(FactoredSplit::new(spec0, [b, 0].into_iter().collect(), 0));
        }
    }
    dag.keyed_costs = true;
    dag.exact_cost_keys = true;
    dag.cost_precision = 100.0;
    dag.rashomon_bound = bound;
    dag
}

fn route(t: &TreeNode, x: &[f64]) -> usize {
    match t {
        TreeNode::Leaf { prediction, .. } => *prediction,
        TreeNode::Internal {
            feature_index,
            thresholds,
            children,
            group_map,
        } => {
            let bin = thresholds.partition_point(|t| *t <= x[*feature_index]);
            let c = group_map.as_ref().map(|g| g[bin]).unwrap_or(bin);
            route(&children[c], x)
        }
    }
}

const ROWS: [[f64; 2]; 4] = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];

fn opts(bound: Option<f64>) -> ExactPredictOpts {
    ExactPredictOpts {
        bound,
        deadline: None,
        verify: true,
        max_states: usize::MAX,
    }
}

fn check_brute(dag: &FactoredDag, bound: f64, tag: &str) {
    let x: Vec<f64> = ROWS.iter().flatten().copied().collect();
    let res = class_counts(dag, &x, ROWS.len(), 2, &opts(Some(bound))).unwrap();
    let n = tree_count_at_bound(dag, bound);
    assert!(n > 0, "{tag}: empty set");
    assert!(res.complete, "{tag}: incomplete");
    assert_eq!(res.count_mismatch, 0, "{tag}: count mismatch");
    assert_eq!(res.n_trees, n, "{tag}: n_trees");
    let names = vec!["x0".to_string(), "x1".to_string()];
    let mut brute = vec![0u128; ROWS.len()];
    for i in 0..n {
        let t = get_tree_at_index_factored(dag, i, &names);
        for (r, row) in ROWS.iter().enumerate() {
            brute[r] += route(&t, row) as u128;
        }
    }
    assert_eq!(res.counts1, brute, "{tag}: class-1 counts vs brute force");
    for r in 0..ROWS.len() {
        assert_eq!(res.counts0[r] + res.counts1[r], n, "{tag}: row {r} sum");
    }
    eprintln!(
        "{tag}: n {n} counts1 {:?} unique rows {}",
        res.counts1, res.n_unique_rows
    );
}

#[test]
fn class_counts_match_bruteforce() {
    let levels = 5;
    for &two in &[false, true] {
        for &bound in &[0.08, 0.11, 0.14] {
            let mut dag = keyed_dag(levels, bound, two);
            recount_at_bound(&mut dag);
            let tag = format!("two {two} bound {bound}");
            check_brute(&dag, bound, &tag);
            // A narrower target inside the DAG's bound.
            let narrow = bound - 0.03;
            if tree_count_at_bound(&dag, narrow) > 0 {
                assert!(tree_count_at_bound(&dag, narrow) < dag.tree_count);
                check_brute(&dag, narrow, &format!("{tag} narrow {narrow}"));
            }
        }
    }
}

/// 7-level chain (c_7 ≈ 4.4e22 trees > 2^64), every tree within the bound:
/// counts0 + counts1 = tree_count bitwise, and the class-1 count follows
/// ones_i = [i odd] + ones_{i-1} · c_{i-1} (x reaches one child, the other
/// child is free).
#[test]
fn chain_u128_sum_identity() {
    let levels = 7;
    let mut dag = count_chain_fixture(levels, 1e-6);
    dag.keyed_costs = true;
    dag.exact_cost_keys = true;
    dag.cost_precision = 1e6;
    recount_at_bound(&mut dag);
    assert!(dag.tree_count > u64::MAX as u128);
    let x = vec![0.0, 1.0];
    let res = class_counts(&dag, &x, 2, 1, &opts(None)).unwrap();
    assert!(res.complete);
    assert_eq!(res.count_mismatch, 0);
    assert_eq!(res.n_trees, dag.tree_count);
    let (mut c, mut ones) = (1u128, 0u128);
    for i in 1..=levels {
        ones = (i % 2) as u128 + ones * c;
        c = 1 + c * c;
    }
    assert_eq!(c, dag.tree_count);
    for r in 0..2 {
        assert_eq!(res.counts0[r] + res.counts1[r], dag.tree_count);
        assert_eq!(res.counts1[r], ones, "row {r}");
    }
}
