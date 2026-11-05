//! Regression tests for the single-tree closure's objective and pruning rules.
use multiway_solver::solver::BranchAndBoundSolver;
use multiway_solver::tree::TreeNode;
use multiway_solver::types::{
    DpCriterion, FeatureMask, FeatureType, MsdtPortConfig, RashomonTuning, SolverConfig,
    SupportSig, ThresholdInterner,
};
use ndarray::Array2;
use std::sync::Arc;

fn config(
    n_classes: usize,
    n_features: usize,
    reg: f64,
    max_depth: i32,
    max_k: usize,
) -> SolverConfig {
    SolverConfig {
        n_classes,
        n_features,
        reg,
        max_depth,
        max_k,
        min_samples_leaf: 1,
        feature_exclusion: false,
        time_limit: None,
        initial_used_features: FeatureMask::all(n_features),
        feature_names: (0..n_features).map(|j| format!("x{j}")).collect(),
        feature_types: vec![FeatureType::Num; n_features],
        warm_start_ub: None,
        n_total_override: None,
        thresholds_per_feature: None,
        closure_thresholds_per_feature: None,
        closure_split_ub_gap: f64::INFINITY,
        closure_feature_top_k: None,
        closure_feature_ranking: None,
        closure_min_n_for_multiway: None,
        closure_unsafe_filter_depth_threshold: 0,
        dp_criterion: DpCriterion::Gini,
        beam_width: 16,
        reg_alpha: 1.0,
        local_thresh_scan: false,
        scan_beam: 1,
        adaptive_beam: false,
        bypass_beam: false,

        transition_delta: 0.0,
        materialize_top_n: 100,
        allow_heuristic_bound_transfer: false,
        shape_mode: false,
        shape_sort_smoothing: 0.0,
        path_penalty: 0.0,
        greedy_extra_depth: 0,
        rashomon_mode: false,
        rashomon_bound: f64::INFINITY,
        step_b_remaining: 0,
        weight_total: 10.0,
        ub_margin: f64::INFINITY,
        n_workers: 1,
        sequential_candidate_gen: false,
        lazy_explore_delta: f64::INFINITY,
        cheap_boundary_greedy: false,
        closure_depth: Some(max_depth),
        closure_beam: 1,
        closure_parallel: false,
        closure_share_memo: false,
        closure_shared_memo: None,
        closure_criterion: None,
        phase0_greedy_prerank_depth: 0,
        eqp_inline_bounds: true,
        threshold_interner: Arc::new(ThresholdInterner::new()),
        rash_tuning: RashomonTuning::default(),
        msdt: MsdtPortConfig::default(),
        teacher_prob: None,
        teacher_prob_c0: None,
        reference_preds: None,
        unsafe_filter_depth_threshold: 0,
        admissibility_strict: false,
    }
}

fn objective(node: &TreeNode, reg: f64, path_penalty: f64, total: usize) -> (f64, usize) {
    match node {
        TreeNode::Leaf {
            n_samples, loss, ..
        } => (loss + reg, *n_samples),
        TreeNode::Internal { children, .. } => {
            let (cost, n) = children
                .iter()
                .map(|child| objective(child, reg, path_penalty, total))
                .fold((0.0, 0), |(cost, n), (child_cost, child_n)| {
                    (cost + child_cost, n + child_n)
                });
            (cost + path_penalty * n as f64 / total as f64, n)
        }
    }
}

fn solve(x: Array2<f64>, y: Vec<u8>, cfg: SolverConfig) -> (f64, TreeNode) {
    let reg = cfg.reg;
    let penalty = cfg.path_penalty;
    let total = y.len();
    let mut solver = BranchAndBoundSolver::from_raw(x, y, cfg, None, None);
    let result = solver.solve();
    let tree = solver.extract_tree();
    let reconstructed = objective(&tree, reg, penalty, total).0;
    assert!(
        (result.objective - reconstructed).abs() < 1e-10,
        "reported={}, tree={reconstructed}",
        result.objective
    );
    (result.objective, tree)
}

#[test]
fn gini_proposal_is_evaluated_by_misclassification() {
    let x = Array2::from_shape_fn((100, 1), |(i, _)| if i < 50 { 0.0 } else { 1.0 });
    let y: Vec<u8> = (0..100)
        .map(|i| {
            if i < 35 || (50..65).contains(&i) {
                0
            } else {
                1
            }
        })
        .collect();
    let mut cfg = config(2, 1, 0.1, 1, 2);
    cfg.thresholds_per_feature = Some([(0, vec![0.5])].into_iter().collect());
    let (cost, tree) = solve(x, y, cfg);
    assert!((cost - 0.5).abs() < 1e-10);
    assert!(matches!(tree, TreeNode::Internal { .. }));
}

#[test]
fn terminal_shortcut_does_not_charge_leaf_path_penalty() {
    let x = Array2::from_shape_fn((100, 1), |(i, _)| if i < 40 { 0.0 } else { 1.0 });
    let y = (0..100).map(|i| u8::from(i >= 40)).collect();
    let mut cfg = config(2, 1, 0.01, 3, 2);
    cfg.path_penalty = 0.05;
    cfg.thresholds_per_feature = Some([(0, vec![0.5])].into_iter().collect());
    let (cost, _) = solve(x, y, cfg);
    assert!((cost - 0.07).abs() < 1e-10);
}

#[test]
fn closure_binary_scan_obeys_supplied_threshold_grid() {
    let x = Array2::from_shape_fn((6, 1), |(i, _)| i as f64 / 5.0);
    let mut cfg = config(2, 1, 0.01, 1, 2);
    cfg.thresholds_per_feature = Some([(0, vec![0.3])].into_iter().collect());
    let (_, tree) = solve(x, vec![0, 0, 0, 1, 1, 1], cfg);
    match tree {
        TreeNode::Internal { thresholds, .. } => assert_eq!(&*thresholds, &[0.3]),
        _ => panic!("expected the allowed improving split"),
    }
}

#[test]
fn support_signature_retains_all_sixteen_classes() {
    let mut first = [0; 16];
    first[8] = 3;
    first[15] = 1;
    let mut second = first;
    second[8] = 1;
    second[15] = 3;
    let a = SupportSig::new(4, &first);
    let b = SupportSig::new(4, &second);
    assert!(a != b);
    assert_eq!(a.as_slice(), &first);
}

#[test]
fn depth_zero_leaf_cannot_poison_another_support() {
    // XOR needs an initially uninformative split, followed by pure children.
    let x = Array2::from_shape_vec(
        (8, 2),
        vec![
            0., 0., 0., 0., 0., 1., 0., 1., 1., 0., 1., 0., 1., 1., 1., 1.,
        ],
    )
    .unwrap();
    let y = vec![0, 0, 1, 1, 1, 1, 0, 0];
    let mut cfg = config(2, 2, 0.01, 2, 2);
    cfg.closure_depth = Some(1);
    cfg.msdt.top_k = None;
    cfg.thresholds_per_feature = Some([(0, vec![0.5]), (1, vec![0.5])].into_iter().collect());
    let mut solver = BranchAndBoundSolver::from_raw(x, y, cfg, None, None);
    let support = solver.search.store.support_from_indices(&[0, 1, 2, 3]);
    let (_, leaf_cost) = solver.closure_build(&support, &FeatureMask::all(2), 0);
    assert!((leaf_cost - 0.26).abs() < 1e-10);
    let result = solver.solve();
    assert!(
        (result.objective - 0.04).abs() < 1e-10,
        "cost={}",
        result.objective
    );
    assert!((objective(&solver.extract_tree(), 0.01, 0.0, 8).0 - 0.04).abs() < 1e-10);
}

#[test]
fn closure_objective_supports_sixteen_classes() {
    let x = Array2::from_shape_fn((32, 4), |(i, j)| (((i / 2) >> j) & 1) as f64);
    let y = (0..32).map(|i| (i / 2) as u8).collect();
    let mut cfg = config(16, 4, 0.001, 4, 2);
    cfg.min_samples_leaf = 2;
    cfg.thresholds_per_feature = Some((0..4).map(|j| (j, vec![0.5])).collect());
    let (cost, _) = solve(x, y, cfg);
    assert!((cost - 0.016).abs() < 1e-10, "cost={cost}");
}

#[test]
fn depth_one_evaluates_the_ranked_shortlist_by_objective() {
    // Feature 0 has lower Gini but higher error than feature 1. Both fit in
    // the shortlist, so the completed tree must use feature 1.
    let x = Array2::from_shape_fn((20, 2), |(i, j)| {
        let left = if j == 0 {
            i == 10
        } else {
            i < 2 || (10..14).contains(&i)
        };
        if left {
            0.0
        } else {
            1.0
        }
    });
    let y = (0..20).map(|i| u8::from(i >= 10)).collect();
    let mut cfg = config(2, 2, 0.01, 1, 2);
    cfg.closure_beam = 2;
    cfg.thresholds_per_feature = Some([(0, vec![0.5]), (1, vec![0.5])].into_iter().collect());
    let (cost, tree) = solve(x, y, cfg);
    assert!((cost - 0.42).abs() < 1e-10, "cost={cost}");
    assert!(matches!(
        tree,
        TreeNode::Internal {
            feature_index: 1,
            ..
        }
    ));
}

#[test]
fn completed_candidate_scores_are_uniform_across_generator_paths() {
    use multiway_solver::candidates::generate_candidates_for_support;
    use std::collections::HashMap;

    let x = Array2::from_shape_fn((60, 1), |(i, _)| (2 * (i / 10)) as f64);
    let zeros = [9, 1, 8, 2, 7, 3];
    let y: Vec<u8> = (0..60).map(|i| u8::from(i % 10 >= zeros[i / 10])).collect();
    let mut seen: HashMap<(bool, Vec<usize>), (f64, f64)> = HashMap::new();
    let mut matches = 0;
    let mut reference_candidates = 0;
    for shape_mode in [false, true] {
        for n_classes in [2, 3] {
            // The third class is deliberately unused.
            for max_k in [3, 4] {
                for criterion in [DpCriterion::Gini, DpCriterion::Misclass] {
                    let mut cfg = config(n_classes, 1, 0.007, 2, max_k);
                    cfg.shape_mode = shape_mode;
                    cfg.dp_criterion = criterion;
                    cfg.path_penalty = 0.03;
                    cfg.beam_width = 32;
                    cfg.msdt.top_k = None;
                    cfg.msdt.shape_boundary_tax = 1.0;
                    // Odd thresholds leave empty bins between occupied bins.
                    cfg.thresholds_per_feature = Some(
                        [(0, (0..10).map(|i| i as f64 + 0.5).collect())]
                            .into_iter()
                            .collect(),
                    );
                    let solver =
                        BranchAndBoundSolver::from_raw(x.clone(), y.clone(), cfg, None, None);
                    let support = solver
                        .search
                        .store
                        .support_from_indices(&(0..60).collect::<Vec<u32>>());
                    let (candidates, _) = generate_candidates_for_support(
                        &solver.search.store,
                        &support,
                        &solver.search.config,
                        &FeatureMask::all(1),
                        60,
                        None,
                        None,
                        &solver.worker.scratch_bank,
                        None,
                        None,
                        Some(2),
                    );
                    assert!(!candidates.is_empty());
                    for candidate in candidates {
                        let is_reference = shape_mode && candidate.family < 2;
                        reference_candidates += usize::from(is_reference);
                        let mut groups = Vec::new();
                        let mut canonical = HashMap::new();
                        for i in 0..6 {
                            let bin = candidate
                                .thresholds
                                .partition_point(|&t| t < (2 * i) as f64);
                            let group = candidate.group_map.as_ref().map_or(bin, |map| map[bin]);
                            let next = canonical.len();
                            groups.push(*canonical.entry(group).or_insert(next));
                        }
                        let mut expected_rank = candidate.k as f64 * 0.007 + 0.03;
                        let mut expected_ub = expected_rank;
                        for (&n, counts) in candidate.child_n.iter().zip(&candidate.child_counts) {
                            let n = n as f64;
                            let squares: f64 = counts.iter().map(|&c| (c as f64).powi(2)).sum();
                            expected_rank += (n - squares / n) / 60.0;
                            expected_ub += (n - *counts.iter().max().unwrap() as f64) / 60.0;
                        }
                        if is_reference {
                            expected_rank +=
                                groups.windows(2).filter(|w| w[0] != w[1]).count() as f64 / 60.0;
                        }
                        assert!((candidate.rank_cost - expected_rank).abs() < 1e-12);
                        assert!((candidate.split_ub - expected_ub).abs() < 1e-12);
                        let scores = (candidate.rank_cost, candidate.split_ub);
                        if let Some(previous) = seen.insert((is_reference, groups), scores) {
                            assert!((previous.0 - scores.0).abs() < 1e-12);
                            assert!((previous.1 - scores.1).abs() < 1e-12);
                            matches += 1;
                        }
                    }
                }
            }
        }
    }
    assert!(matches > 0);
    assert!(reference_candidates > 0);
}
