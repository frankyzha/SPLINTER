//! The Step-A pass (`BranchAndBoundSolver::rashomon_enumerate`) and the
//! completion lower bounds on tiny solves:
//!   cargo test --release --no-default-features --test rashomon_stepa

mod common;

use std::sync::Arc;

use multiway_solver::solver::{
    rash_iv_dyadic_debug, set_rash_d1_reference_dp, BranchAndBoundSolver,
};
use multiway_solver::types::{FeatureMask, FeatureType, SolverConfig};
use ndarray::Array2;

use common::{d1_wide_config, lcg, phase2_test_config, pool_key};

fn solver(y: &[u8], config: SolverConfig) -> BranchAndBoundSolver {
    let n = y.len();
    let x = Array2::from_shape_fn((n, 1), |(i, _)| i as f64);
    BranchAndBoundSolver::from_raw(x, y.to_vec(), config, None, None)
}

fn root_state(s: &BranchAndBoundSolver) -> (bool, f64, Vec<Vec<f64>>) {
    let root_key = s.search.root_key.lock().unwrap().clone();
    let root = s.search.memo.get(&root_key).unwrap();
    let ths = root
        .rashomon_indices
        .iter()
        .map(|&i| s.worker.split_pool.get(i).thresholds.to_vec())
        .collect();
    (root.is_terminal, root.scope, ths)
}

/// A near-pure node (max_loss = 0.1 < split_reg + leaf_reg = 0.15; legacy
/// marks it terminal) keeps its within-β split x ≤ 0.5 (cost 0.30 vs leaf
/// 0.25, β = 0.35): pure / near-pure splits are separate trees.
#[test]
fn near_pure_node_split_recorded() {
    let y = [1u8, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    let mut s = solver(&y, phase2_test_config(0.15, 1, 2, 0.35));
    s.solve();
    let (terminal, _scope, ths) = root_state(&s);
    assert!(!terminal, "near-pure root must stay expandable");
    assert!(
        ths.iter()
            .any(|t| t.len() == 1 && (t[0] - 0.5).abs() < 1e-12),
        "split x <= 0.5 not recorded: {:?}",
        ths
    );
}

/// Widening the scope re-explores an explored node: after a solve at β1 the
/// root is explored at β1; requesting β2 > β1 re-explores it at β2 and
/// records the splits whose admissible lb-sum lies in (β1, β2].
#[test]
fn scope_widening_reexplores_node() {
    let y = [0u8, 0, 0, 1, 1, 1, 0, 0, 1, 1];
    let (b1, b2) = (0.15, 0.60);
    let mut s = solver(&y, phase2_test_config(0.05, 2, 2, b1));
    s.solve();
    let (_, scope1, ths1) = root_state(&s);
    assert!(
        (scope1 - b1).abs() < 1e-12,
        "root explored at {scope1}, want {b1}"
    );
    let reexp1 = s.worker.profile.n_reexplorations;

    Arc::get_mut(&mut s.search)
        .expect("search state shared")
        .config
        .rashomon_bound = b2;
    s.solve();
    let (_, scope2, ths2) = root_state(&s);
    assert!(
        (scope2 - b2).abs() < 1e-12,
        "root explored at {scope2}, want {b2}"
    );
    assert!(
        s.worker.profile.n_reexplorations > reexp1,
        "no re-exploration"
    );
    assert!(
        ths2.len() > ths1.len(),
        "wider scope recorded no new root split ({} vs {})",
        ths2.len(),
        ths1.len()
    );
    for t in &ths1 {
        assert!(ths2.contains(t), "split {:?} lost on widening", t);
    }
}

// ---------------------------------------------------------------------------
// Exact completion lower bounds of the Step-A pass
// ---------------------------------------------------------------------------

/// Tiny deterministic label vectors (LCG), n in 6..=10.
fn tiny_labels(n_inst: usize) -> Vec<Vec<u8>> {
    let mut st: u64 = 0x9e3779b97f4a7c15;
    let mut next = || {
        st = st
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (st >> 33) as u32
    };
    (0..n_inst)
        .map(|_| {
            let n = 6 + (next() % 5) as usize;
            (0..n).map(|_| (next() % 2) as u8).collect()
        })
        .collect()
}

/// Brute-force minimum tree cost at `rows` (depth <= `depth`) over
/// `rashomon_select`'s pool: leaf, or split_reg + Σ children (path 0).
fn brute_min(s: &mut BranchAndBoundSolver, rows: &[u32], depth: i32, split_reg: f64) -> f64 {
    let leaf = s.rash_leaf_objective(rows);
    if depth <= 0 || rows.len() <= 1 {
        return leaf;
    }
    let support = s.search.store.support_from_indices(rows);
    let x: Vec<f64> = s.search.store.feature_col(0).to_vec();
    let mut best = leaf;
    for cand in s.rashomon_select(&support) {
        assert!(cand.group_map.is_none());
        let k = cand.thresholds.len() + 1;
        let mut bins: Vec<Vec<u32>> = vec![Vec::new(); k];
        for &r in rows {
            bins[cand.thresholds.partition_point(|t| *t <= x[r as usize])].push(r);
        }
        if bins.iter().filter(|b| !b.is_empty()).count() <= 1 {
            continue;
        }
        let children: Vec<f64> = bins
            .iter()
            .map(|b| brute_min(s, b, depth - 1, split_reg))
            .collect();
        let cost = split_reg + children.iter().sum::<f64>();
        best = best.min(cost);
    }
    best
}

/// `bounded_completion_min` equals the brute-force minimum over the
/// lb-solver's pool (`rashomon_select`) when it fits the budget and returns
/// None otherwise; a wide query after a tight infeasible one (same solver,
/// memo holds `Above`) re-searches and returns the exact minimum. Also: a
/// leaf-only completion (depth 0) prices every explored node at the leaf
/// cost the explore assigns (`create_subproblem`'s base_objective).
#[test]
fn bounded_lb_le_bruteforce_min() {
    let mut n_checked = 0usize;
    let mut n_split_min = 0usize;
    for (inst, y) in tiny_labels(12).into_iter().enumerate() {
        let n = y.len();
        let (reg, max_k) = if inst % 2 == 0 { (0.05, 2) } else { (0.08, 3) };
        let beta = if inst % 3 == 0 { 0.45 } else { 2.0 };
        let mut probe = solver(&y, phase2_test_config(reg, 2, max_k, beta));
        probe.solve();
        // Leaf-only completion == the explore's leaf cost.
        let mask = FeatureMask::all(1);
        let entries: Vec<_> = probe
            .search
            .memo
            .iter()
            .map(|e| (e.value().support.clone(), e.value().base_objective))
            .collect();
        for (sup, base) in entries {
            assert_eq!(
                probe.bounded_completion_min(&sup, &mask, 0, 10.0),
                Some(base)
            );
        }
        let all: Vec<u32> = (0..n as u32).collect();
        let half: Vec<u32> = (0..n as u32).filter(|i| i % 2 == 0 || *i < 3).collect();
        for rows in [all, half] {
            for depth in 1..=2 {
                let fresh = || solver(&y, phase2_test_config(reg, 2, max_k, beta));
                let mut s = fresh();
                let support = s.search.store.support_from_indices(&rows);
                let m = brute_min(&mut s, &rows, depth, 0.0);
                if m < s.rash_leaf_objective(&rows) - 1e-12 {
                    n_split_min += 1;
                }
                // Wide budget: the exact minimum.
                let got = s.bounded_completion_min(&support, &mask, depth, m + 0.1);
                assert!(
                    matches!(got, Some(v) if (v - m).abs() < 1e-12),
                    "wide {got:?} vs {m}"
                );
                // Budget exactly m: still fits (RASH_TOL slack).
                let mut s = fresh();
                let got = s.bounded_completion_min(&support, &mask, depth, m);
                assert!(
                    matches!(got, Some(v) if (v - m).abs() < 1e-12),
                    "exact {got:?} vs {m}"
                );
                // Tight then wide on one solver.
                let mut s = fresh();
                let tight = m - 0.02;
                assert_eq!(
                    s.bounded_completion_min(&support, &mask, depth, tight),
                    None
                );
                assert_eq!(
                    s.bounded_completion_min(&support, &mask, depth, tight - 0.01),
                    None
                );
                let got = s.bounded_completion_min(&support, &mask, depth, m + 0.05);
                assert!(
                    matches!(got, Some(v) if (v - m).abs() < 1e-12),
                    "re-query {got:?} vs {m}"
                );
                n_checked += 1;
            }
        }
    }
    assert_eq!(n_checked, 48);
    assert!(
        n_split_min >= 10,
        "only {n_split_min} minima below the leaf"
    );
}

/// Children enter the memo only for recorded
/// splits: after the Step-A pass every memo key is the root or a child of a
/// recorded split (also with Step B pending below, step_b_remaining = 1).
#[test]
fn lazy_insert_no_orphans() {
    let mut n_keys = 0usize;
    for (inst, y) in tiny_labels(8).into_iter().enumerate() {
        for (depth, sbr, beta) in [
            (2, 0, 0.6),
            (3, 0, 0.3),
            (2, 1, 0.3),
            (1, 2, 0.25),
            (2, 1, 0.2),
        ] {
            let mut cfg = phase2_test_config(0.05, depth, 2 + inst % 2, beta);
            cfg.step_b_remaining = sbr;
            let mut s = solver(&y, cfg);
            s.solve();
            let root = s.search.root_key.lock().unwrap().clone();
            let mut recorded: std::collections::HashSet<multiway_solver::types::MemoKey> =
                std::collections::HashSet::new();
            for e in s.search.memo.iter() {
                for &i in &e.value().rashomon_indices {
                    for &ck in &s.worker.split_pool.get(i).child_keys {
                        recorded.insert(s.search.memo_key_arena.get(ck).clone());
                    }
                }
            }
            for e in s.search.memo.iter() {
                assert!(
                    *e.key() == root || recorded.contains(e.key()),
                    "orphan memo entry (n={}, depth={})",
                    e.value().n_samples,
                    e.key().depth
                );
                n_keys += 1;
            }
        }
    }
    assert!(n_keys > 24);
}

// ---------------------------------------------------------------------------
// Support-sorted Select scans / the seeded multi-root pass
// ---------------------------------------------------------------------------

/// Three-feature data with heavy ties (x0 in 0..4 categorical-like, x1 in
/// 0..3, x2 in 0..12), labels in 0..n_classes.
fn tie_data(n: usize, n_classes: u32, seed: u64) -> (Array2<f64>, Vec<u8>) {
    let mut st = seed;
    let mut x = Array2::<f64>::zeros((n, 3));
    let mut y = Vec::with_capacity(n);
    for i in 0..n {
        x[[i, 0]] = lcg(&mut st, 4) as f64;
        x[[i, 1]] = lcg(&mut st, 3) as f64;
        x[[i, 2]] = lcg(&mut st, 12) as f64;
        y.push(lcg(&mut st, n_classes) as u8);
    }
    (x, y)
}

/// Select over the support's rows sorted by rank returns the same pool,
/// bitwise and in order, as the full sorted-order scan: binary and
/// 3-class, numeric (no custom thresholds: sorted bins pass) and
/// categorical features, random supports of every size.
#[test]
fn select_support_sort_identical() {
    use multiway_solver::candidates::generate_candidates_for_support_scan;
    use multiway_solver::dataset_store::DatasetStore;
    use multiway_solver::scratch::ScratchBank;
    let mut n_pools = 0usize;
    let mut n_cands = 0usize;
    for (n_classes, max_k) in [(2u32, 2usize), (2, 3), (3, 2), (3, 3)] {
        let n = 60;
        let (x, y) = tie_data(n, n_classes, 7 + n_classes as u64 * 31 + max_k as u64);
        let store = DatasetStore::new(x, y, None, n_classes as usize, None, None);
        let mut cfg = phase2_test_config(0.02, 2, max_k, 0.6);
        cfg.n_classes = n_classes as usize;
        cfg.n_features = 3;
        cfg.feature_names = vec!["a".into(), "b".into(), "c".into()];
        cfg.feature_types = vec![FeatureType::Cat, FeatureType::Num, FeatureType::Num];
        cfg.thresholds_per_feature = None;
        cfg.initial_used_features = FeatureMask::all(3);
        let bank = ScratchBank::new();
        let mut st = 99u64 + n_classes as u64;
        for trial in 0..24 {
            let keep = 2 + lcg(&mut st, 9);
            let rows: Vec<u32> = (0..n as u32).filter(|_| lcg(&mut st, 10) < keep).collect();
            if rows.len() < 2 {
                continue;
            }
            let support = store.support_from_indices(&rows);
            let ub = if trial % 3 == 0 { None } else { Some(0.6) };
            let run = |sort: bool| {
                generate_candidates_for_support_scan(
                    &store,
                    &support,
                    &cfg,
                    &FeatureMask::all(3),
                    n,
                    ub,
                    None,
                    &bank,
                    None,
                    None,
                    Some(2),
                    sort,
                )
                .0
            };
            let (a, b) = (run(false), run(true));
            assert_eq!(
                pool_key(&a),
                pool_key(&b),
                "pool differs (n={})",
                rows.len()
            );
            n_pools += 1;
            n_cands += a.len();
        }
    }
    assert!(
        n_pools >= 80 && n_cands > 200,
        "{n_pools} pools / {n_cands} candidates"
    );
}

/// Recorded splits per memo key: (feature, threshold bits).
fn recorded_splits(
    s: &BranchAndBoundSolver,
) -> std::collections::HashMap<
    multiway_solver::types::MemoKey,
    std::collections::HashSet<(usize, Vec<u64>)>,
> {
    let mut out = std::collections::HashMap::new();
    for e in s.search.memo.iter() {
        let set: std::collections::HashSet<(usize, Vec<u64>)> = e
            .value()
            .rashomon_indices
            .iter()
            .map(|&i| {
                let r = s.worker.split_pool.get(i);
                (
                    r.feature_index,
                    r.thresholds.iter().map(|t| t.to_bits()).collect(),
                )
            })
            .collect();
        out.insert(e.key().clone(), set);
    }
    out
}

/// A seeded pass over several roots (`solve_seeded`) records at every node
/// a superset of what each single-root pass records there (a node shared
/// by several roots is explored at the widest requested scope); the seed
/// keys come back in order; the lb caches move between solvers over one
/// store only with a matching key and give the same minima.
#[test]
fn seeded_enumerate_superset() {
    use multiway_solver::dataset_store::DatasetStore;
    let mut n_checked = 0usize;
    for (inst, y) in tiny_labels(10).into_iter().enumerate() {
        let n = y.len();
        let x = Array2::from_shape_fn((n, 1), |(i, _)| i as f64);
        let store = Arc::new(DatasetStore::new(
            x,
            y.clone(),
            None,
            2,
            phase2_test_config(0.05, 2, 2, 1.0)
                .thresholds_per_feature
                .as_ref(),
            None,
        ));
        let (depth, max_k) = (1 + (inst % 2) as i32, 2 + inst % 2);
        let mk = |beta: f64, sbr: i32| {
            let mut c = phase2_test_config(0.05, depth, max_k, beta);
            c.step_b_remaining = sbr;
            c
        };
        let sbr = (inst % 3 == 0) as i32;
        let seeds: Vec<(Vec<u32>, f64)> = vec![
            ((0..n as u32).collect(), 0.5),
            ((0..n as u32).filter(|i| i % 2 == 0).collect(), 0.3),
            ((0..n as u32 - 2).collect(), 0.35),
            ((3..n as u32).collect(), 0.25),
        ];
        let seed_masks: Vec<(multiway_solver::types::SupportMask, f64)> = seeds
            .iter()
            .map(|(r, b)| (store.support_from_indices(r), *b))
            .collect();
        let beta_max = seeds.iter().map(|s| s.1).fold(0.0, f64::max);
        let mut shared = BranchAndBoundSolver::new(
            store.clone(),
            mk(beta_max, sbr),
            None,
            Some(seed_masks[0].0.clone()),
        );
        shared.set_rashomon_select_ub(0.6);
        let (_res, keys) = shared.solve_seeded(&seed_masks);
        assert_eq!(keys.len(), seeds.len());
        let got = recorded_splits(&shared);
        for (i, (sup, beta)) in seed_masks.iter().enumerate() {
            assert_eq!(&keys[i].support, sup);
            let mut single =
                BranchAndBoundSolver::new(store.clone(), mk(*beta, sbr), None, Some(sup.clone()));
            single.set_rashomon_select_ub(0.6);
            single.solve();
            for (key, splits) in recorded_splits(&single) {
                let have = got.get(&key).expect("node of a single-root pass missing");
                assert!(splits.is_subset(have), "seeded pass lost a split");
                n_checked += splits.len();
            }
        }
        // lb cache carry: same key installs and answers like a fresh solver.
        {
            let mut a = BranchAndBoundSolver::new(store.clone(), mk(0.5, 1), None, None);
            a.set_rashomon_select_ub(0.6);
            a.solve();
            let cache = a.take_rash_lb_cache();
            let mut c = BranchAndBoundSolver::new(store.clone(), mk(0.3, 0), None, None);
            c.set_rashomon_select_ub(0.7);
            assert!(!c.seed_rash_lb_cache(cache.clone()));
            let mut b = BranchAndBoundSolver::new(store.clone(), mk(0.3, 0), None, None);
            b.set_rashomon_select_ub(0.6);
            let had_key = cache.key.is_some();
            assert_eq!(b.seed_rash_lb_cache(cache), had_key);
            let mut fresh = BranchAndBoundSolver::new(store.clone(), mk(0.3, 0), None, None);
            fresh.set_rashomon_select_ub(0.6);
            let mask = FeatureMask::all(1);
            for (sup, _) in &seed_masks {
                for d in 1..=2 {
                    assert_eq!(
                        b.bounded_completion_min(sup, &mask, d, 0.4),
                        fresh.bounded_completion_min(sup, &mask, d, 0.4)
                    );
                }
            }
        }
    }
    assert!(n_checked > 20, "only {n_checked} splits compared");
}

// ---------------------------------------------------------------------------
// Closed-form depth-1 completion minimum
// ---------------------------------------------------------------------------

/// Two integer features (x0 in 0..5, x1 in 0..8) with every midpoint of
/// the present values as thresholds, labels in 0..n_classes.
fn d1_data(n: usize, n_classes: u32, seed: u64) -> (Array2<f64>, Vec<u8>, Vec<Vec<f64>>) {
    let mut st = seed;
    let mut x = Array2::<f64>::zeros((n, 2));
    let mut y = Vec::with_capacity(n);
    for i in 0..n {
        x[[i, 0]] = lcg(&mut st, 5) as f64;
        x[[i, 1]] = lcg(&mut st, 8) as f64;
        y.push(lcg(&mut st, n_classes) as u8);
    }
    let ths = (0..2)
        .map(|f| {
            let mut v: Vec<f64> = x.column(f).to_vec();
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            v.dedup();
            v.windows(2).map(|w| 0.5 * (w[0] + w[1])).collect()
        })
        .collect();
    (x, y, ths)
}

/// Every set of 1..=max_k-1 cut indices out of 0..t (increasing).
fn cut_sets(t: usize, max_cuts: usize) -> Vec<Vec<usize>> {
    let mut out = Vec::new();
    let mut cur = Vec::new();
    fn rec(
        start: usize,
        t: usize,
        max_cuts: usize,
        cur: &mut Vec<usize>,
        out: &mut Vec<Vec<usize>>,
    ) {
        if !cur.is_empty() {
            out.push(cur.clone());
        }
        if cur.len() == max_cuts {
            return;
        }
        for i in start..t {
            cur.push(i);
            rec(i + 1, t, max_cuts, cur, out);
            cur.pop();
        }
    }
    rec(0, t, max_cuts, &mut cur, &mut out);
    out
}

/// The closed-form depth-1 minimum (`rash_depth1_min_closed`) equals an
/// independent brute force over every contiguous partition into 2..=max_k
/// bins of every feature's thresholds (empty bins allowed, cost
/// overhead + Σ leaf), and min(leaf, closed) <= the brute-force minimum
/// over `rashomon_select`'s pool with leaf children: binary and 3-class,
/// max_k 2..=5, with and without a split cost.
#[test]
fn d1_closed_le_pool_min() {
    let mut n_checked = 0usize;
    let mut n_strict = 0usize;
    for (n_classes, max_k, reg_alpha) in [
        (2u32, 2usize, 1.0),
        (2, 3, 1.0),
        (2, 4, 0.5),
        (2, 5, 1.0),
        (3, 2, 0.5),
        (3, 3, 1.0),
        (3, 4, 1.0),
        (3, 5, 0.5),
    ] {
        for inst in 0..6u64 {
            let n = 12 + (inst as usize * 5) % 19;
            let (x, y, ths) = d1_data(
                n,
                n_classes,
                1 + inst * 977 + max_k as u64 * 31 + n_classes as u64,
            );
            let mut cfg = phase2_test_config(0.03, 2, max_k, 0.8);
            cfg.n_classes = n_classes as usize;
            cfg.n_features = 2;
            cfg.reg_alpha = reg_alpha;
            cfg.feature_names = vec!["a".into(), "b".into()];
            cfg.feature_types = vec![FeatureType::Num, FeatureType::Num];
            cfg.initial_used_features = FeatureMask::all(2);
            cfg.thresholds_per_feature = Some(ths.iter().cloned().enumerate().collect());
            let split_reg = (1.0 - reg_alpha) * 0.03;
            let mut s = BranchAndBoundSolver::from_raw(x.clone(), y.clone(), cfg, None, None);
            let mut st = 5 + inst;
            for trial in 0..4 {
                let rows: Vec<u32> = if trial == 0 {
                    (0..n as u32).collect()
                } else {
                    (0..n as u32).filter(|_| lcg(&mut st, 10) < 7).collect()
                };
                if rows.len() < 2 {
                    continue;
                }
                let support = s.search.store.support_from_indices(&rows);
                let closed = s
                    .rash_depth1_min_closed(&support, rows.len(), split_reg)
                    .expect("closed form must apply (numeric, keyed thresholds)");
                // Independent brute force over all contiguous partitions.
                let mut brute = f64::INFINITY;
                for (f, t) in ths.iter().enumerate() {
                    for cuts in cut_sets(t.len(), max_k - 1) {
                        let cut_vals: Vec<f64> = cuts.iter().map(|&i| t[i]).collect();
                        let mut bins: Vec<Vec<u32>> = vec![Vec::new(); cut_vals.len() + 1];
                        for &r in &rows {
                            bins[cut_vals.partition_point(|c| *c <= x[[r as usize, f]])].push(r);
                        }
                        let sum: f64 = bins.iter().map(|b| s.rash_leaf_objective(b)).sum();
                        brute = brute.min(split_reg + sum);
                    }
                }
                assert!(
                    (closed - brute).abs() < 1e-12,
                    "closed {closed} vs brute {brute}"
                );
                // Select pool with leaf children.
                let leaf = s.rash_leaf_objective(&rows);
                let mut pool = leaf;
                for cand in s.rashomon_select(&support) {
                    let f = cand.feature_index;
                    let mut bins: Vec<Vec<u32>> = vec![Vec::new(); cand.thresholds.len() + 1];
                    for &r in &rows {
                        bins[cand
                            .thresholds
                            .partition_point(|c| *c <= x[[r as usize, f]])]
                        .push(r);
                    }
                    if bins.iter().filter(|b| !b.is_empty()).count() <= 1 {
                        continue;
                    }
                    let sum: f64 = bins.iter().map(|b| s.rash_leaf_objective(b)).sum();
                    pool = pool.min(split_reg + sum);
                }
                let v = leaf.min(closed);
                assert!(v <= pool + 1e-12, "closed {v} > pool min {pool}");
                if v < pool - 1e-12 {
                    n_strict += 1;
                }
                n_checked += 1;
            }
        }
    }
    assert!(
        n_checked >= 150,
        "only {n_checked} supports checked ({n_strict} strictly below)"
    );
}

/// `vals.len()` integer features (feature f in 0..vals[f]) with every
/// midpoint of the present values as thresholds, labels in 0..n_classes;
/// with `skew`, the label follows feature 0 (y = class of x0 / vals[0])
/// for 3 rows out of 4.
fn d1_data_wide(
    n: usize,
    n_classes: u32,
    vals: &[u32],
    seed: u64,
    skew: bool,
) -> (Array2<f64>, Vec<u8>, Vec<Vec<f64>>) {
    let mut st = seed;
    let nf = vals.len();
    let mut x = Array2::<f64>::zeros((n, nf));
    let mut y = Vec::with_capacity(n);
    for i in 0..n {
        for (f, &m) in vals.iter().enumerate() {
            x[[i, f]] = lcg(&mut st, m) as f64;
        }
        let r = lcg(&mut st, n_classes);
        let yl = if skew && lcg(&mut st, 4) > 0 {
            ((x[[i, 0]] as u32 * n_classes) / vals[0]).min(n_classes - 1)
        } else {
            r
        };
        y.push(yl as u8);
    }
    let ths = (0..nf)
        .map(|f| {
            let mut v: Vec<f64> = x.column(f).to_vec();
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            v.dedup();
            v.windows(2).map(|w| 0.5 * (w[0] + w[1])).collect()
        })
        .collect();
    (x, y, ths)
}

/// The closed-form depth-1 minimum is bitwise identical across its speed
/// paths ({two-pass on/off} × {row walk forced on / forced off /
/// adaptive}) and equals an independent brute force over every contiguous
/// partition into 2..=max_k bins: binary and 3-class, max_k 2..=5, features
/// with many thresholds (up to ~40), full, random and tiny (2..=5 row)
/// supports.
#[test]
fn d1_closed_variants_identical() {
    let mut n_checked = 0usize;
    for (n_classes, max_k, reg_alpha) in [
        (2u32, 2usize, 1.0),
        (2, 3, 0.5),
        (2, 4, 1.0),
        (2, 5, 1.0),
        (3, 2, 1.0),
        (3, 3, 0.5),
        (3, 4, 1.0),
        (3, 5, 1.0),
    ] {
        for inst in 0..4u64 {
            let vals: &[u32] = if inst % 2 == 0 {
                &[5, 9, 60]
            } else {
                &[45, 3, 20]
            };
            let n = 36 + inst as usize * 7;
            let seed = 11 + inst * 1013 + max_k as u64 * 37 + n_classes as u64;
            let (x, y, ths) = d1_data_wide(n, n_classes, vals, seed, inst >= 2);
            let cfg = d1_wide_config(n_classes as usize, max_k, reg_alpha, &ths);
            let split_reg = (1.0 - reg_alpha) * 0.03;
            let mut s = BranchAndBoundSolver::from_raw(x.clone(), y.clone(), cfg, None, None);
            assert_eq!(
                s.rash_lb_gate(),
                (true, true),
                "gate must pass (keyed numeric thresholds)"
            );
            let mut st = 7 + inst;
            for trial in 0..7 {
                let rows: Vec<u32> = match trial {
                    0 => (0..n as u32).collect(),
                    1 | 2 => (0..n as u32).filter(|_| lcg(&mut st, 10) < 6).collect(),
                    _ => {
                        let want = 2 + lcg(&mut st, 4) as usize;
                        let mut r: Vec<u32> = Vec::new();
                        while r.len() < want {
                            let i = lcg(&mut st, n as u32);
                            if !r.contains(&i) {
                                r.push(i);
                            }
                        }
                        r.sort();
                        r
                    }
                };
                if rows.len() < 2 {
                    continue;
                }
                let support = s.search.store.support_from_indices(&rows);
                let mut vals_seen: Vec<u64> = Vec::new();
                for twopass in [false, true] {
                    for rw in [Some(false), Some(true), None] {
                        let (v, walked) = s
                            .rash_depth1_min_closed_ext(
                                &support,
                                rows.len(),
                                split_reg,
                                twopass,
                                rw,
                            )
                            .expect("closed form must apply");
                        if let Some(forced) = rw {
                            assert_eq!(walked, forced, "row-walk path not taken as forced");
                        }
                        vals_seen.push(v.to_bits());
                    }
                }
                assert!(
                    vals_seen.iter().all(|&b| b == vals_seen[0]),
                    "closed-form variants differ: {:?}",
                    vals_seen
                        .iter()
                        .map(|&b| f64::from_bits(b))
                        .collect::<Vec<_>>()
                );
                let closed = f64::from_bits(vals_seen[0]);
                assert_eq!(
                    closed.to_bits(),
                    s.rash_depth1_min_closed(&support, rows.len(), split_reg)
                        .unwrap()
                        .to_bits()
                );
                // Independent brute force over all contiguous partitions (at
                // most 4 bins on a feature with > 30 thresholds: T^4 cut sets
                // are too many for max_k 5).
                let mut brute = f64::INFINITY;
                for (f, t) in ths.iter().enumerate() {
                    for cuts in cut_sets(t.len(), (max_k - 1).min(if t.len() > 30 { 3 } else { 4 }))
                    {
                        let cut_vals: Vec<f64> = cuts.iter().map(|&i| t[i]).collect();
                        let mut bins: Vec<Vec<u32>> = vec![Vec::new(); cut_vals.len() + 1];
                        for &r in &rows {
                            bins[cut_vals.partition_point(|c| *c <= x[[r as usize, f]])].push(r);
                        }
                        let sum: f64 = bins.iter().map(|b| s.rash_leaf_objective(b)).sum();
                        brute = brute.min(split_reg + sum);
                    }
                }
                let capped = max_k == 5 && ths.iter().any(|t| t.len() > 30);
                if capped {
                    // Brute force over a subset of the partitions: >= closed.
                    assert!(
                        closed <= brute + 1e-12,
                        "closed {closed} > partial brute {brute}"
                    );
                } else {
                    assert!(
                        (closed - brute).abs() < 1e-12,
                        "closed {closed} vs brute {brute}"
                    );
                }
                n_checked += 1;
            }
        }
    }
    assert!(n_checked >= 200, "only {n_checked} supports checked");
}

// ---------------------------------------------------------------------------
// Children's depth-1 closed form from the parent's sweep
// ---------------------------------------------------------------------------

/// Random splits of feature f: `n` threshold sets of 1..=max_k-1 distinct
/// store thresholds (ascending), with the children's supports
/// (`partition_support_le`).
fn sweep_splits(
    s: &BranchAndBoundSolver,
    parent: &multiway_solver::types::SupportMask,
    f: usize,
    ths: &[f64],
    max_k: usize,
    n: usize,
    st: &mut u64,
) -> Vec<(Vec<f64>, Vec<multiway_solver::types::SupportMask>)> {
    let mut out = Vec::new();
    for _ in 0..n {
        let n_cuts = 1 + lcg(st, (max_k - 1).min(ths.len()) as u32) as usize;
        let mut idx: Vec<usize> = Vec::new();
        while idx.len() < n_cuts {
            let i = lcg(st, ths.len() as u32) as usize;
            if !idx.contains(&i) {
                idx.push(i);
            }
        }
        idx.sort();
        let cut: Vec<f64> = idx.iter().map(|&i| ths[i]).collect();
        let cs = s
            .search
            .store
            .partition_support_le(parent, f, &cut, None)
            .expect("store thresholds");
        out.push((cut, cs));
    }
    out
}

/// The swept depth-1 closed form (`rash_d1_sweep_values`)
/// equals `rash_depth1_min_closed_ext` bitwise for every child of many
/// random splits sharing one sweep: binary and 3-class, max_k 2..=5,
/// features with many thresholds (up to ~60), full, random and tiny
/// parents, two-pass on and off, with and without a split cost.
#[test]
fn d1_sweep_identical() {
    let mut n_checked = 0usize;
    for (n_classes, max_k, reg_alpha) in [
        (2u32, 2usize, 1.0),
        (2, 3, 0.5),
        (2, 4, 1.0),
        (2, 5, 1.0),
        (3, 2, 1.0),
        (3, 3, 0.5),
        (3, 4, 1.0),
        (3, 5, 1.0),
    ] {
        for inst in 0..4u64 {
            let vals: &[u32] = if inst % 2 == 0 {
                &[5, 9, 60]
            } else {
                &[45, 3, 20, 7]
            };
            let n = 40 + inst as usize * 23;
            let seed = 17 + inst * 733 + max_k as u64 * 41 + n_classes as u64;
            let (x, y, ths) = d1_data_wide(n, n_classes, vals, seed, inst >= 2);
            let cfg = d1_wide_config(n_classes as usize, max_k, reg_alpha, &ths);
            let split_reg = (1.0 - reg_alpha) * 0.03;
            let mut s = BranchAndBoundSolver::from_raw(x, y, cfg, None, None);
            assert_eq!(
                s.rash_lb_gate(),
                (true, true),
                "gate must pass (keyed numeric thresholds)"
            );
            let mut st = 3 + inst;
            for trial in 0..5 {
                let rows: Vec<u32> = match trial {
                    0 => (0..n as u32).collect(),
                    1 | 2 => (0..n as u32).filter(|_| lcg(&mut st, 10) < 6).collect(),
                    _ => (0..n as u32).filter(|_| lcg(&mut st, 10) < 1).collect(),
                };
                if rows.len() < 2 {
                    continue;
                }
                let parent = s.search.store.support_from_indices(&rows);
                for (f, t) in ths.iter().enumerate() {
                    if t.is_empty() {
                        continue;
                    }
                    let splits = sweep_splits(&s, &parent, f, t, max_k, 12, &mut st);
                    let sizes: Vec<(Vec<f64>, Vec<usize>)> = splits
                        .iter()
                        .map(|(c, cs)| (c.clone(), cs.iter().map(|m| m.n_samples).collect()))
                        .collect();
                    for twopass in [false, true] {
                        let swept = s
                            .rash_d1_sweep_values(&parent, f, &sizes, split_reg, twopass)
                            .expect("sweep must apply (store thresholds, d1_ok)");
                        for ((_, cs), sv) in splits.iter().zip(&swept) {
                            for (c, &v) in cs.iter().zip(sv) {
                                let (d, _) = s
                                    .rash_depth1_min_closed_ext(
                                        c,
                                        c.n_samples,
                                        split_reg,
                                        twopass,
                                        None,
                                    )
                                    .expect("closed form must apply");
                                assert_eq!(
                                    v.to_bits(),
                                    d.to_bits(),
                                    "swept {v} vs direct {d} (n={})",
                                    c.n_samples
                                );
                                n_checked += 1;
                            }
                        }
                    }
                }
            }
        }
    }
    assert!(n_checked >= 3000, "only {n_checked} children checked");
}

// ---------------------------------------------------------------------------
// Integer-first depth-1 DP
// ---------------------------------------------------------------------------

/// Values of the depth-1 closed form with the integer-first DP and with
/// the reference DP (`set_rash_d1_reference_dp`): every direct speed path
/// ({two-pass on/off} × {mask counts, adaptive}) and, per feature, the
/// swept values of a few random splits. Returns (integer-first, reference)
/// as bit patterns.
fn intdp_on_off(
    s: &mut BranchAndBoundSolver,
    support: &multiway_solver::types::SupportMask,
    n_node: usize,
    overhead: f64,
    sweeps: &[(usize, Vec<(Vec<f64>, Vec<usize>)>)],
    parent: &multiway_solver::types::SupportMask,
) -> (Vec<u64>, Vec<u64>) {
    let mut out = (Vec::new(), Vec::new());
    for on in [true, false] {
        set_rash_d1_reference_dp(!on);
        let mut v = Vec::new();
        for twopass in [false, true] {
            for rw in [Some(false), None] {
                let (d, _) = s
                    .rash_depth1_min_closed_ext(support, n_node, overhead, twopass, rw)
                    .expect("closed form must apply");
                v.push(d.to_bits());
            }
        }
        for (f, sizes) in sweeps {
            for twopass in [false, true] {
                let sw = s
                    .rash_d1_sweep_values(parent, *f, sizes, overhead, twopass)
                    .expect("sweep must apply");
                v.extend(sw.concat().iter().map(|x| x.to_bits()));
            }
        }
        set_rash_d1_reference_dp(false);
        if on {
            out.0 = v;
        } else {
            out.1 = v;
        }
    }
    out
}

/// The integer-first depth-1 DP returns the reference DP's value
/// bitwise, direct and swept: binary and 3-class, max_k 2..=5, T up to 250,
/// full / 60% / 10% / 2–7-row supports, regularisation 0 / 1e-4 /
/// 0.003–0.02, with and without a split cost, skewed and random labels
/// (random labels reach the >= 3-bin pass); plus adversarial cases: many
/// exact integer ties (periodic and near-constant labels), n_total at and
/// just above the 2^32 guard, overhead at and just above the 1e3 guard,
/// leaf_reg = -0.0 (tab[0] guard). The guard cases must fall back.
#[test]
fn d1_intdp_identical() {
    use multiway_solver::solver::rash_d1_intdp_counts_take;
    let (mut n_checked, mut n_int, mut n_fb) = (0usize, 0u64, 0u64);
    rash_d1_intdp_counts_take();
    for (n_classes, max_k, reg_alpha, reg) in [
        (2u32, 2usize, 1.0, 0.01),
        (2, 3, 0.5, 0.01),
        (2, 4, 1.0, 0.003),
        (2, 5, 1.0, 0.02),
        (3, 2, 1.0, 0.01),
        (3, 3, 0.5, 0.005),
        (3, 4, 1.0, 0.01),
        (3, 5, 0.7, 0.01),
        (2, 3, 1.0, 0.0),
        (2, 4, 1.0, 1e-4),
        (3, 3, 1.0, 0.0),
        (3, 4, 1.0, 1e-4),
    ] {
        for inst in 0..4u64 {
            let vals: Vec<u32> = if inst % 2 == 0 {
                vec![251, 9, 60, 3]
            } else {
                vec![45, 2, 120, 7, 251]
            };
            let n = [60usize, 400, 900, 1500][inst as usize];
            let seed = 101 + inst * 7919 + max_k as u64 * 31 + n_classes as u64;
            let (x, y, ths) = d1_data_wide(n, n_classes, &vals, seed, inst >= 2);
            let mut cfg = d1_wide_config(n_classes as usize, max_k, reg_alpha, &ths);
            cfg.reg = reg;
            let split_reg = (1.0 - reg_alpha) * 0.03;
            let mut s = BranchAndBoundSolver::from_raw(x, y, cfg, None, None);
            assert_eq!(s.rash_lb_gate(), (true, true));
            let mut st = 5 + inst * 3;
            for trial in 0..4 {
                let rows: Vec<u32> = match trial {
                    0 => (0..n as u32).collect(),
                    1 => (0..n as u32).filter(|_| lcg(&mut st, 10) < 6).collect(),
                    2 => (0..n as u32).filter(|_| lcg(&mut st, 10) < 1).collect(),
                    _ => {
                        let want = 2 + lcg(&mut st, 6) as usize;
                        let mut r: Vec<u32> = Vec::new();
                        while r.len() < want {
                            let i = lcg(&mut st, n as u32);
                            if !r.contains(&i) {
                                r.push(i);
                            }
                        }
                        r.sort();
                        r
                    }
                };
                if rows.len() < 2 {
                    continue;
                }
                let support = s.search.store.support_from_indices(&rows);
                let sweeps: Vec<(usize, Vec<(Vec<f64>, Vec<usize>)>)> = (0..vals.len())
                    .filter(|&f| !ths[f].is_empty())
                    .map(|f| {
                        let sp = sweep_splits(&s, &support, f, &ths[f], max_k, 3, &mut st);
                        (
                            f,
                            sp.iter()
                                .map(|(c, cs)| {
                                    (c.clone(), cs.iter().map(|m| m.n_samples).collect())
                                })
                                .collect(),
                        )
                    })
                    .collect();
                let (on, off) =
                    intdp_on_off(&mut s, &support, rows.len(), split_reg, &sweeps, &support);
                assert_eq!(
                    on, off,
                    "intdp differs: nc={n_classes} k={max_k} reg={reg} inst={inst} trial={trial}"
                );
                n_checked += on.len();
            }
        }
    }
    let (a, b) = rash_d1_intdp_counts_take();
    n_int += a;
    n_fb += b;
    assert_eq!(n_fb, 0, "the random cases must not fall back");

    // Adversarial cases on one store per class count.
    let mut n_fb_want = 0u64;
    for n_classes in [2u32, 3] {
        for max_k in [3usize, 5] {
            for labels in 0..3 {
                let vals = vec![61u32, 12, 5];
                let n = 700usize;
                let (x, mut y, ths) = d1_data_wide(n, n_classes, &vals, 77 + max_k as u64, false);
                for i in 0..n {
                    y[i] = match labels {
                        // Periodic in x0: many partitions with equal counts.
                        0 => ((x[[i, 0]] as u32 / 2) % n_classes) as u8,
                        // Near-constant: almost every bin has m = 0.
                        1 => u8::from(i % 97 == 0) * (n_classes as u8 - 1),
                        // Periodic in x1 (few thresholds) with noise.
                        _ => ((x[[i, 1]] as u32 + u32::from(i % 13 == 0)) % n_classes) as u8,
                    };
                }
                for case in 0..6 {
                    let mut cfg = d1_wide_config(n_classes as usize, max_k, 1.0, &ths);
                    cfg.reg = if case == 5 {
                        -0.0
                    } else {
                        [0.0, 0.01, 1e-4, 0.01, 0.01][case]
                    };
                    let mut s =
                        BranchAndBoundSolver::from_raw(x.clone(), y.clone(), cfg, None, None);
                    assert_eq!(s.rash_lb_gate(), (true, true));
                    let lr = if case == 5 {
                        -0.0
                    } else {
                        [0.0, 0.01, 1e-4, 0.01, 0.01][case]
                    };
                    let mut overheads = vec![0.0, 0.017];
                    match case {
                        // n_total at the guard (answered) and just above (fallback).
                        3 | 4 => {
                            Arc::get_mut(&mut s.search)
                                .expect("unique search state")
                                .n_total = (1usize << 32) + usize::from(case == 4);
                        }
                        // Overhead just under and just over the 1e3 guard.
                        2 => {
                            let g = 1e3 - max_k as f64 * (1.0 + lr);
                            overheads = vec![g - 1e-6, g + 1e-6];
                        }
                        _ => {}
                    }
                    let mut st = 11 + case as u64;
                    for trial in 0..3 {
                        let rows: Vec<u32> = match trial {
                            0 => (0..n as u32).collect(),
                            1 => (0..n as u32).filter(|_| lcg(&mut st, 10) < 3).collect(),
                            _ => (0..n as u32).filter(|_| lcg(&mut st, 10) < 1).collect(),
                        };
                        let support = s.search.store.support_from_indices(&rows);
                        let sweeps: Vec<(usize, Vec<(Vec<f64>, Vec<usize>)>)> = (0..vals.len())
                            .map(|f| {
                                let sp = sweep_splits(&s, &support, f, &ths[f], max_k, 3, &mut st);
                                (
                                    f,
                                    sp.iter()
                                        .map(|(c, cs)| {
                                            (c.clone(), cs.iter().map(|m| m.n_samples).collect())
                                        })
                                        .collect(),
                                )
                            })
                            .collect();
                        for (oi, &ov) in overheads.iter().enumerate() {
                            rash_d1_intdp_counts_take();
                            let (on, off) =
                                intdp_on_off(&mut s, &support, rows.len(), ov, &sweeps, &support);
                            assert_eq!(on, off, "intdp differs: adversarial nc={n_classes} k={max_k} labels={labels} case={case} trial={trial} overhead={ov}");
                            n_checked += on.len();
                            let (a, b) = rash_d1_intdp_counts_take();
                            let must_fall_back = case == 4 || case == 5 || (case == 2 && oi == 1);
                            if must_fall_back {
                                assert_eq!(a, 0, "guard case {case} must fall back");
                                assert!(b > 0);
                                n_fb_want += b;
                            } else {
                                assert_eq!(b, 0, "case {case} must not fall back");
                                assert!(a > 0);
                            }
                            n_int += a;
                            n_fb += b;
                        }
                    }
                }
            }
        }
    }
    assert_eq!(n_fb, n_fb_want);
    assert!(
        n_checked >= 20_000 && n_int >= 5_000,
        "only {n_checked} values / {n_int} integer-first calls"
    );
    eprintln!(
        "d1_intdp_identical: {n_checked} values, {n_int} integer-first calls, {n_fb} fallbacks"
    );
}

// ---------------------------------------------------------------------------
// Interval-DP completion minimum at remaining depth 2
// ---------------------------------------------------------------------------

/// Brute-force depth-1 minimum at `rows`: the leaf, or split_reg + Σ leaf
/// over every contiguous partition into 2..=max_k bins of every feature's
/// thresholds (empty bins allowed).
fn brute_d1_all(
    s: &BranchAndBoundSolver,
    x: &Array2<f64>,
    ths: &[Vec<f64>],
    rows: &[u32],
    max_k: usize,
    split_reg: f64,
) -> f64 {
    let mut best = s.rash_leaf_objective(rows);
    for (f, t) in ths.iter().enumerate() {
        for cuts in cut_sets(t.len(), max_k - 1) {
            let cut_vals: Vec<f64> = cuts.iter().map(|&i| t[i]).collect();
            let mut bins: Vec<Vec<u32>> = vec![Vec::new(); cut_vals.len() + 1];
            for &r in rows {
                bins[cut_vals.partition_point(|c| *c <= x[[r as usize, f]])].push(r);
            }
            let sum: f64 = bins.iter().map(|b| s.rash_leaf_objective(b)).sum();
            best = best.min(split_reg + sum);
        }
    }
    best
}

/// Brute-force depth-2 minimum at `rows` over ALL contiguous partitions
/// (arity 2..=max_k at both levels, empty bins allowed): the leaf, or
/// split_reg + Σ_children `brute_d1_all`.
fn brute_d2_all(
    s: &BranchAndBoundSolver,
    x: &Array2<f64>,
    ths: &[Vec<f64>],
    rows: &[u32],
    max_k: usize,
    split_reg: f64,
) -> f64 {
    let mut best = s.rash_leaf_objective(rows);
    for (f, t) in ths.iter().enumerate() {
        for cuts in cut_sets(t.len(), max_k - 1) {
            let cut_vals: Vec<f64> = cuts.iter().map(|&i| t[i]).collect();
            let mut bins: Vec<Vec<u32>> = vec![Vec::new(); cut_vals.len() + 1];
            for &r in rows {
                bins[cut_vals.partition_point(|c| *c <= x[[r as usize, f]])].push(r);
            }
            let sum: f64 = bins
                .iter()
                .map(|b| brute_d1_all(s, x, ths, b, max_k, split_reg))
                .sum();
            best = best.min(split_reg + sum);
        }
    }
    best
}

/// The interval-DP completion minimum at remaining depth 2
/// (`rash_interval_completion_min2_debug`) equals an independent brute
/// force over every contiguous partition (depth <= 2, arity <= max_k at
/// both levels), is <= the pool path (`bounded_completion_min` with the
/// interval flag off), and honours its bound (None below the minimum, the
/// minimum at it): binary and 3-class, max_k 2..=4, with and without a
/// split cost, skewed and random labels, full and beam-1 pools.
#[test]
fn interval_d2_equals_bruteforce_le_pool() {
    let (mut n_checked, mut n_strict, mut n_split) = (0usize, 0usize, 0usize);
    for (n_classes, max_k, reg_alpha) in [
        (2u32, 2usize, 1.0),
        (2, 3, 1.0),
        (2, 4, 0.5),
        (3, 2, 0.5),
        (3, 3, 1.0),
        (2, 2, 0.5),
        (3, 4, 1.0),
    ] {
        for inst in 0..5u64 {
            let n = 14 + (inst as usize * 7) % 23;
            let vals: &[u32] = if inst % 2 == 0 { &[5, 7, 4] } else { &[6, 3] };
            let (x, y, ths) = d1_data_wide(
                n,
                n_classes,
                vals,
                3 + inst * 131 + max_k as u64 * 17,
                inst % 3 == 1,
            );
            let split_reg = (1.0 - reg_alpha) * 0.03;
            // Odd instances: a beam-1 Select pool (beam-limited lb).
            let cfg = || {
                let mut c = d1_wide_config(n_classes as usize, max_k, reg_alpha, &ths);
                if inst % 2 == 1 {
                    c.beam_width = 1;
                    c.bypass_beam = false;
                }
                c
            };
            let mut s = BranchAndBoundSolver::from_raw(x.clone(), y.clone(), cfg(), None, None);
            let mask = FeatureMask::all(ths.len());
            let mut st = 11 + inst;
            for trial in 0..4 {
                let rows: Vec<u32> = if trial == 0 {
                    (0..n as u32).collect()
                } else {
                    (0..n as u32).filter(|_| lcg(&mut st, 10) < 6).collect()
                };
                if rows.len() < 3 {
                    continue;
                }
                let support = s.search.store.support_from_indices(&rows);
                let brute = brute_d2_all(&s, &x, &ths, &rows, max_k, split_reg);
                let iv = s
                    .rash_interval_completion_min2_debug(&support, 10.0)
                    .expect("interval path must apply (numeric, keyed thresholds)");
                assert!(
                    (iv - brute).abs() < 1e-12,
                    "interval {iv} vs brute {brute} (k={max_k}, nc={n_classes})"
                );
                if brute < s.rash_leaf_objective(&rows) - 1e-12 {
                    n_split += 1;
                }
                // Bound contract on a fresh solver.
                let mut s2 =
                    BranchAndBoundSolver::from_raw(x.clone(), y.clone(), cfg(), None, None);
                assert_eq!(
                    s2.rash_interval_completion_min2_debug(&support, iv - 1e-6),
                    None
                );
                assert_eq!(
                    s2.rash_interval_completion_min2_debug(&support, iv)
                        .map(f64::to_bits),
                    Some(iv.to_bits())
                );
                // Pool path (interval lower bound off): never below the interval value.
                let mut s3 =
                    BranchAndBoundSolver::from_raw(x.clone(), y.clone(), cfg(), None, None);
                let pool = s3
                    .bounded_completion_min(&support, &mask, 2, 10.0)
                    .expect("wide bound");
                assert!(iv <= pool, "interval {iv} > pool {pool}");
                if iv < pool {
                    n_strict += 1;
                }
                n_checked += 1;
            }
        }
    }
    let stat = |k: &str| {
        multiway_solver::solver::rash_iv_stats()
            .iter()
            .find(|x| x.0 == k)
            .map_or(0, |x| x.1)
    };
    eprintln!(
        "interval_d2: checked {n_checked}, strictly below pool {n_strict}, split minima {n_split}; psidy short {} full {}",
        stat("n_lb_interval_psidy_short"),
        stat("n_lb_interval_psidy_full"),
    );
    assert!(n_checked >= 100, "only {n_checked} supports checked");
    assert!(n_split >= 20, "only {n_split} minima below the leaf");
    assert!(
        n_strict >= 1,
        "the pool never lost a partition: test too weak"
    );
}

/// The dyadic bound Ψ_dy(a, b) (`rash_iv_dyadic_debug`) of
/// a root feature's cell interval [a, b) is <= the finest-grid error of
/// every feature on those rows and <= the error of every depth-1 split
/// (up to 4 bins) on them, >= the per-cell φ sum, and equal to the
/// finest-grid minimum on aligned power-of-two intervals: random binary
/// and 3-class data, 3-4 features of 4-8 values, 30 seeds.
#[test]
fn psidy_dyadic_le_depth1_error() {
    // Σ over g-ranks of (rows − majority) of `rows` (labels `y`).
    fn err_by(rows: &[usize], key: &dyn Fn(usize) -> usize, nb: usize, y: &[u8], nc: usize) -> i64 {
        let mut h = vec![0i64; nb * nc];
        for &r in rows {
            h[key(r) * nc + y[r] as usize] += 1;
        }
        h.chunks_exact(nc)
            .map(|c| c.iter().sum::<i64>() - c.iter().copied().max().unwrap_or(0))
            .sum()
    }
    let (mut n_checked, mut n_aligned) = (0usize, 0usize);
    for seed in 0..30u64 {
        let nc = 2 + (seed % 2) as usize;
        let mut st = 0x9e37 + seed * 7919;
        let nf = 3 + (seed % 2) as usize;
        let vals: Vec<u32> = (0..nf).map(|_| 4 + lcg(&mut st, 5)).collect();
        let n = 20 + ((seed * 13) % 41) as usize;
        let (x, y, ths) = d1_data_wide(n, nc as u32, &vals, 5 + seed * 104_729, seed % 3 == 0);
        // Ranks as `rash_lb_rank_build`; histogram layout as `rash_iv_run`.
        let rank: Vec<Vec<usize>> = (0..nf)
            .map(|g| {
                (0..n)
                    .map(|r| ths[g].partition_point(|&t| t < x[[r, g]]))
                    .collect()
            })
            .collect();
        let tlen: Vec<usize> = ths.iter().map(|t| t.len()).collect();
        let mut hoff = Vec::new();
        let mut hlen = 0usize;
        for &t in &tlen {
            hoff.push(hlen);
            if t > 0 {
                hlen += (t + 1) * nc;
            }
        }
        let active: Vec<usize> = (0..nf).filter(|&g| tlen[g] > 0).collect();
        for &f in &active {
            let cell_rows: Vec<Vec<usize>> = (0..=tlen[f])
                .map(|q| (0..n).filter(|&r| rank[f][r] == q).collect::<Vec<usize>>())
                .filter(|c| !c.is_empty())
                .collect();
            let m = cell_rows.len();
            let mut p = vec![0i32; (m + 1) * hlen];
            for (i, rs) in cell_rows.iter().enumerate() {
                let (lo, hi) = p.split_at_mut((i + 1) * hlen);
                hi[..hlen].copy_from_slice(&lo[i * hlen..]);
                for &r in rs {
                    for &g in &active {
                        hi[hoff[g] + rank[g][r] * nc + y[r] as usize] += 1;
                    }
                }
            }
            let psi = rash_iv_dyadic_debug(&p, hlen, m, &hoff, &tlen, &active, nc);
            let fine_min = |rows: &[usize]| -> i64 {
                active
                    .iter()
                    .map(|&g| err_by(rows, &|r| rank[g][r], tlen[g] + 1, &y, nc))
                    .min()
                    .unwrap()
            };
            let phi_cell: Vec<i64> = cell_rows.iter().map(|c| fine_min(c)).collect();
            for a in 0..m {
                for b in (a + 1)..=m {
                    let rows: Vec<usize> = cell_rows[a..b].iter().flatten().copied().collect();
                    let v = psi[a * (m + 1) + b];
                    let fm = fine_min(&rows);
                    assert!(
                        v <= fm,
                        "Ψ_dy {v} > finest-grid min {fm} (seed {seed}, f {f}, [{a}, {b}))"
                    );
                    for &g in &active {
                        for cuts in cut_sets(tlen[g], 3) {
                            let e = err_by(
                                &rows,
                                &|r| cuts.partition_point(|&c| c < rank[g][r]),
                                cuts.len() + 1,
                                &y,
                                nc,
                            );
                            assert!(v <= e, "Ψ_dy {v} > split error {e} (seed {seed}, f {f}, g {g}, cuts {cuts:?})");
                        }
                    }
                    let lo: i64 = phi_cell[a..b].iter().sum();
                    assert!(
                        v >= lo,
                        "Ψ_dy {v} < φ sum {lo} (seed {seed}, f {f}, [{a}, {b}))"
                    );
                    let len = b - a;
                    if len.is_power_of_two() && a % len == 0 {
                        assert_eq!(v, fm, "aligned block [{a}, {b}) (seed {seed}, f {f})");
                        n_aligned += 1;
                    }
                    n_checked += 1;
                }
            }
            for a in 0..=m {
                for b in 0..=a {
                    assert_eq!(psi[a * (m + 1) + b], -1);
                }
            }
        }
    }
    eprintln!("psidy_dyadic: checked {n_checked} intervals ({n_aligned} aligned)");
    assert!(n_checked >= 1000, "only {n_checked} intervals checked");
}
