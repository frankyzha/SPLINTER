//! Helpers shared by the integration tests (`mod common;` in each test file).
#![allow(dead_code)] // each test binary uses a different subset

use std::sync::Arc;

use multiway_solver::types::{
    DpCriterion, FeatureMask, FeatureType, MsdtPortConfig, RashomonTuning, SolverConfig,
    SplitCandidate, ThresholdInterner,
};

/// Phase-2-style config (mirrors lib.rs `phase2_solver_config`): rashomon
/// mode, one feature x = 0..9 with every midpoint threshold, reg_alpha = 1.
pub fn phase2_test_config(reg: f64, max_depth: i32, max_k: usize, beta: f64) -> SolverConfig {
    SolverConfig {
        n_classes: 2,
        n_features: 1,
        reg,
        max_depth,
        max_k,
        min_samples_leaf: 1,
        feature_exclusion: false,
        time_limit: None,
        initial_used_features: FeatureMask::all(1),
        feature_names: vec!["x0".to_string()],
        feature_types: vec![FeatureType::Num],
        warm_start_ub: None,
        n_total_override: None,
        thresholds_per_feature: Some(
            [(0usize, (0..9).map(|v| v as f64 + 0.5).collect::<Vec<f64>>())]
                .into_iter()
                .collect(),
        ),
        closure_thresholds_per_feature: None,
        closure_split_ub_gap: f64::INFINITY,
        closure_feature_top_k: None,
        closure_feature_ranking: None,
        closure_min_n_for_multiway: None,
        closure_unsafe_filter_depth_threshold: 0,
        dp_criterion: DpCriterion::Rush,
        beam_width: 16,
        reg_alpha: 1.0,
        local_thresh_scan: false,
        scan_beam: 1,
        adaptive_beam: false,
        bypass_beam: true,

        transition_delta: 0.0,
        materialize_top_n: 100,
        allow_heuristic_bound_transfer: false,
        shape_mode: false,
        shape_sort_smoothing: 0.0,
        path_penalty: 0.0,
        greedy_extra_depth: 0,
        rashomon_mode: true,
        rashomon_bound: beta,
        step_b_remaining: 0,
        weight_total: 10.0,
        ub_margin: f64::INFINITY,
        n_workers: 1,
        sequential_candidate_gen: false,
        lazy_explore_delta: f64::INFINITY,
        cheap_boundary_greedy: false,
        closure_depth: None,
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

/// Deterministic LCG in [0, m).
pub fn lcg(st: &mut u64, m: u32) -> u32 {
    *st = st
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*st >> 33) as u32) % m
}

/// Canonical, bitwise form of a Select pool (every field, in order).
pub fn pool_key(c: &[SplitCandidate]) -> Vec<String> {
    c.iter()
        .map(|c| {
            format!(
                "{}|{}|{:?}|{:?}|{:?}|{:?}|{:x}|{:x}|{}|{}",
                c.feature_index,
                c.k,
                c.thresholds
                    .iter()
                    .map(|t| t.to_bits())
                    .collect::<Vec<u64>>(),
                c.group_map,
                c.child_n,
                c.child_counts,
                c.split_ub.to_bits(),
                c.rank_cost.to_bits(),
                c.family,
                c.components
            )
        })
        .collect()
}

/// A keyed numeric config over `ths` (one entry per feature).
pub fn d1_wide_config(
    n_classes: usize,
    max_k: usize,
    reg_alpha: f64,
    ths: &[Vec<f64>],
) -> SolverConfig {
    let nf = ths.len();
    let mut cfg = phase2_test_config(0.03, 2, max_k, 0.8);
    cfg.n_classes = n_classes;
    cfg.n_features = nf;
    cfg.reg_alpha = reg_alpha;
    cfg.feature_names = (0..nf).map(|f| format!("f{f}")).collect();
    cfg.feature_types = vec![FeatureType::Num; nf];
    cfg.initial_used_features = FeatureMask::all(nf);
    cfg.thresholds_per_feature = Some(ths.iter().cloned().enumerate().collect());
    cfg
}
