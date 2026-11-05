//! u128 tree counts: exact convolution above 2^64, saturation at u128::MAX
//! (with the saturation-event counter behind `FactoredDag.count_saturated`),
//! and tree indexing above 2^64 on a chain DAG (`count_chain_fixture`).

use std::collections::HashSet;

use multiway_solver::rashomon_dag::{
    convolve_cost_schedules, count_chain_fixture, get_tree_at_index_factored, iter_splits,
};
use multiway_solver::scratch::ConvolveScratch;
use multiway_solver::tree::TreeNode;
use multiway_solver::types::{
    count_saturation_events, CostEntry, FactoredSplit, InlineChildren, SpecId,
};

/// c_0 = 1, c_i = 1 + c_{i-1}^2 (None once it overflows u128).
fn chain_count(levels: usize) -> Option<u128> {
    let mut c: u128 = 1;
    for _ in 0..levels {
        c = c.checked_mul(c)?.checked_add(1)?;
    }
    Some(c)
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

#[test]
fn struct_sizes() {
    // The pre-widening layout (u64 count), for the size report.
    #[allow(dead_code)]
    struct FactoredSplitU64 {
        spec_id: SpecId,
        children: InlineChildren,
        tree_count: u64,
    }
    let old = std::mem::size_of::<FactoredSplitU64>();
    let new = std::mem::size_of::<FactoredSplit>();
    eprintln!(
        "size_of FactoredSplit: u64-count {} B -> u128-halves {} B (align {})",
        old,
        new,
        std::mem::align_of::<FactoredSplit>()
    );
    eprintln!(
        "ConvolveScratch element: (i32,u64) {} B -> (i32,u128) {} B",
        std::mem::size_of::<(i32, u64)>(),
        std::mem::size_of::<(i32, u128)>()
    );
    assert_eq!(std::mem::align_of::<FactoredSplit>(), 8);
    assert_eq!(new, old + 8);
    let _ = ConvolveScratch::default();
}

/// (a) A product between 2^64 and 2^128 is exact.
#[test]
fn convolve_exact_above_u64() {
    let ev0 = count_saturation_events();
    let a = vec![CostEntry::new(0.1, 1 << 40), CostEntry::new(0.2, 3)];
    let b = vec![CostEntry::new(0.1, 1 << 40), CostEntry::new(0.3, 5)];
    let c = vec![CostEntry::new(0.0, (1 << 30) + 7)];
    let out = convolve_cost_schedules(&[&a, &b], 10.0, 1e6);
    let got: Vec<(i64, u128)> = out
        .iter()
        .map(|e| ((e.cost * 1e6).round() as i64, e.tree_count()))
        .collect();
    let mut got = got;
    got.sort();
    assert_eq!(
        got,
        vec![
            (200_000, 1u128 << 80),
            (300_000, 3u128 << 40),
            (400_000, 5u128 << 40),
            (500_000, 15)
        ]
    );
    let out3 = convolve_cost_schedules(&[&a, &b, &c], 10.0, 1e6);
    let first = out3
        .iter()
        .find(|e| ((e.cost * 1e6).round() as i64) == 200_000)
        .unwrap();
    assert_eq!(first.tree_count(), (1u128 << 80) * ((1u128 << 30) + 7));
    assert!(first.tree_count() > u64::MAX as u128);
    // No clamp happened in this test (other tests may run concurrently and
    // bump the global counter, so only check exactness above).
    let _ = ev0;
}

/// (b) A product above 2^128 saturates at u128::MAX and records the event.
#[test]
fn convolve_saturates_above_u128() {
    let ev0 = count_saturation_events();
    let a = vec![CostEntry::new(0.1, 1u128 << 100)];
    let b = vec![CostEntry::new(0.1, 1u128 << 30)];
    let out = convolve_cost_schedules(&[&a, &b], 10.0, 1e6);
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].tree_count(), u128::MAX);
    assert!(count_saturation_events() > ev0);
}

/// Chain DAG with c_7 ≈ 4.4e22 > 2^64 trees: exact counts, split counts
/// above u64, distinct trees at indices above 2^64.
#[test]
fn chain_dag_counts_and_indexing_above_u64() {
    let dag = count_chain_fixture(7, 0.001);
    let c7 = chain_count(7).unwrap();
    assert!(c7 > u64::MAX as u128);
    assert_eq!(dag.tree_count, c7);
    assert!(!dag.count_saturated);
    let root_split = iter_splits(&dag, dag.root_ref).next().unwrap();
    assert_eq!(root_split.tree_count(), c7 - 1);
    let names = vec!["x0".to_string()];
    let base = 1u128 << 64;
    let idxs = [0, 1, base, base + 1, base + 12_345, c7 - 2, c7 - 1];
    let sigs: HashSet<String> = idxs
        .iter()
        .map(|&i| sig(&get_tree_at_index_factored(&dag, i, &names)))
        .collect();
    assert_eq!(sigs.len(), idxs.len());
}

/// Small chain: indexing is a bijection onto the c_4 = 677 trees.
#[test]
fn chain_dag_small_bijection() {
    let dag = count_chain_fixture(4, 0.001);
    assert_eq!(dag.tree_count, 677);
    let names = vec!["x0".to_string()];
    let sigs: HashSet<String> = (0..677u128)
        .map(|i| sig(&get_tree_at_index_factored(&dag, i, &names)))
        .collect();
    assert_eq!(sigs.len(), 677);
}
