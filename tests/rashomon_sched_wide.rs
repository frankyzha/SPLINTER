//! The wide schedule cache of keyed exact-key DAGs (one schedule per
//! vertex at its widest scope key, narrower scopes read as a prefix): the
//! root schedule sums to the tree count, a second count pass agrees,
//! every index extracts, and `materialize_schedule` answers in any request order equal the
//! answers from a fresh cache.
//!   cargo test --release --no-default-features --test rashomon_sched_wide

use std::sync::Arc;

use dashmap::DashMap;
use multiway_solver::rashomon_dag::{
    count_chain_fixture, get_tree_at_index_factored, materialize_schedule, recount_at_bound,
};
use multiway_solver::tree::TreeNode;
use multiway_solver::types::{FSubRef, FactoredDag, FactoredSplit};

/// Lattice step 0.01 (precision 100). Vertex i: leaf cost 0.01·(1 + i%3),
/// splits (v_{i-1}, v_{i-1}) and, for i ≥ 2, (v_{i-1}, v_{i-2}) and
/// (v_{i-2}, v_0). The bound keeps part of the trees, so child scopes
/// differ per parent and the prefix matters.
fn keyed_dag(levels: usize, bound: f64) -> FactoredDag {
    let mut dag = count_chain_fixture(levels, 0.01);
    let spec = dag.subproblems[1].splits[0].spec_id;
    for i in 0..=levels {
        let sp = &mut dag.subproblems[i];
        if let Some((c, _)) = sp.leaf.as_mut() {
            *c = 0.01 * (1 + i % 3) as f64;
        }
        if i >= 2 {
            let (a, b) = ((i - 1) as FSubRef, (i - 2) as FSubRef);
            sp.splits
                .push(FactoredSplit::new(spec, [a, b].into_iter().collect(), 0));
            sp.splits
                .push(FactoredSplit::new(spec, [b, 0].into_iter().collect(), 0));
        }
    }
    dag.keyed_costs = true;
    dag.exact_cost_keys = true;
    dag.cost_precision = 100.0;
    dag.rashomon_bound = bound;
    dag
}

fn sig(t: &TreeNode) -> String {
    match t {
        TreeNode::Leaf { prediction, .. } => format!("L{}", prediction),
        TreeNode::Internal { children, .. } => {
            let cs: Vec<String> = children.iter().map(sig).collect();
            format!("({})", cs.join(","))
        }
    }
}

/// Everything the count pass and extraction expose, cost bits included.
fn snapshot(dag: &FactoredDag) -> (u128, Vec<(u64, u128)>, Vec<(u128, Vec<u128>)>, Vec<String>) {
    let root: Vec<(u64, u128)> = dag
        .root_cost_schedule
        .iter()
        .map(|e| (e.cost.to_bits(), e.tree_count()))
        .collect();
    let verts: Vec<(u128, Vec<u128>)> = dag
        .subproblems
        .iter()
        .map(|sp| {
            (
                sp.tree_count,
                sp.splits.iter().map(|s| s.tree_count()).collect(),
            )
        })
        .collect();
    let names = vec!["x0".to_string()];
    // Extraction of every tree on small sets only.
    let n_trees = if dag.tree_count <= 20_000 {
        dag.tree_count
    } else {
        0
    };
    let trees: Vec<String> = (0..n_trees)
        .map(|i| sig(&get_tree_at_index_factored(dag, i, &names)))
        .collect();
    (dag.tree_count, root, verts, trees)
}

/// `materialize_schedule` answers for (vertex, scope) requests in the
/// given order, from a fresh cache, as (cost bits, count) lists.
fn answers(dag: &mut FactoredDag, reqs: &[(FSubRef, f64)]) -> Vec<Vec<(u64, u128)>> {
    dag.schedule_cache = Arc::new(DashMap::new());
    dag.wide_schedule_cache = Default::default();
    reqs.iter()
        .map(|&(v, s)| {
            materialize_schedule(dag, v, s)
                .iter()
                .map(|e| (e.cost.to_bits(), e.tree_count()))
                .collect()
        })
        .collect()
}

#[test]
fn sched_wide_prefix_answers_match_fresh_cache() {
    let levels = 5;
    for &bound in &[0.08, 0.15, 0.25] {
        let mut dag = keyed_dag(levels, bound);
        recount_at_bound(&mut dag);
        let snap = snapshot(&dag);
        let wide_entries = dag.wide_schedule_cache.entries.len();
        assert!(snap.0 > 0, "bound {bound}: no trees");
        eprintln!(
            "bound {bound}: {} trees, {} wide entries",
            snap.0, wide_entries
        );
        assert!(
            wide_entries <= levels + 1,
            "one wide entry per vertex at most"
        );
        assert_eq!(
            snap.1.iter().map(|e| e.1).sum::<u128>(),
            snap.0,
            "bound {bound}: root sum"
        );
        if snap.0 <= 20_000 {
            assert_eq!(
                snap.3.len() as u128,
                snap.0,
                "bound {bound}: extracted trees"
            );
        }
        // A second count pass gives the same counts, schedule and trees.
        recount_at_bound(&mut dag);
        assert_eq!(snapshot(&dag), snap, "bound {bound}: recount differs");

        // Requests narrow-to-wide, wide-to-narrow and interleaved (wider
        // requests replace the wide entry, narrower read a prefix).
        let mut reqs: Vec<(FSubRef, f64)> = Vec::new();
        for v in 0..=levels as FSubRef {
            for k in 0..=30 {
                reqs.push((v, k as f64 * 0.01));
            }
            for k in (0..=30).rev() {
                reqs.push((v, k as f64 * 0.01 + 0.004));
            }
        }
        for k in 0..=30 {
            for v in (0..=levels as FSubRef).rev() {
                reqs.push((v, ((k * 17) % 31) as f64 * 0.01));
            }
        }
        let batched = answers(&mut dag, &reqs);
        for (i, &r) in reqs.iter().enumerate() {
            let fresh = answers(&mut dag, &[r]);
            assert_eq!(
                batched[i], fresh[0],
                "bound {bound}: request {r:?} differs from a fresh cache"
            );
        }
    }
}
