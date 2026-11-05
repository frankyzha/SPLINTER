//! Exact speed paths of Select and the Step-A / completion-lb passes: the
//! support-sorted scan and the stats-free DPs give the same Select pool
//! bitwise and in order; a handed-over lb cache, a tiny lb memo cap, the
//! le-mask child supports and the parallel lb prefetch (any
//! `RashomonTuning::lb_threads`) give the same recorded splits / children
//! per node and the same lb values; batched Step B (any
//! `RashomonTuning::stepb_batches`) gives the same tree set.
//!   cargo test --release --no-default-features --test rashomon_speed_paths

mod common;

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use multiway_solver::candidates::{generate_candidates_for_support_opts, SelectOpts};
use multiway_solver::dataset_store::DatasetStore;
use multiway_solver::rashomon_dag::{get_tree_at_index_factored, rash_stepb_batch_stats};
use multiway_solver::scratch::ScratchBank;
use multiway_solver::solver::{rash_par_stats, BranchAndBoundSolver};
use multiway_solver::types::SupportMask;
use multiway_solver::types::{DpCriterion, FeatureMask, FeatureType, RashomonTuning, SolverConfig};
use ndarray::Array2;

use common::{lcg, phase2_test_config, pool_key};

/// The pass-1 / Step-B batch counters (`rash_par_stats`,
/// `rash_stepb_batch_stats`) are process-wide: tests that read them run one
/// at a time.
static STATS_LOCK: Mutex<()> = Mutex::new(());

/// `n_feat` integer features (value ranges 3..=14, heavy ties), labels in
/// 0..n_classes with a feature-0 signal, and custom thresholds per feature:
/// a random non-empty subset of the midpoints of the present values (never
/// equal to a row value, so masks and row partitions agree).
fn custom_data(
    n: usize,
    n_feat: usize,
    n_classes: u32,
    seed: u64,
) -> (Array2<f64>, Vec<u8>, HashMap<usize, Vec<f64>>) {
    let mut st = seed;
    let ranges: Vec<u32> = (0..n_feat).map(|_| 3 + lcg(&mut st, 12)).collect();
    let mut x = Array2::<f64>::zeros((n, n_feat));
    let mut y = Vec::with_capacity(n);
    for i in 0..n {
        for f in 0..n_feat {
            x[[i, f]] = lcg(&mut st, ranges[f]) as f64;
        }
        let noisy = lcg(&mut st, 4) == 0;
        let base = ((x[[i, 0]] as u32) * n_classes / ranges[0]).min(n_classes - 1);
        y.push(if noisy { lcg(&mut st, n_classes) } else { base } as u8);
    }
    let mut ths = HashMap::new();
    for f in 0..n_feat {
        let mut v: Vec<f64> = x.column(f).to_vec();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v.dedup();
        let mids: Vec<f64> = v.windows(2).map(|w| 0.5 * (w[0] + w[1])).collect();
        let keep = 1 + lcg(&mut st, 3);
        let mut sel: Vec<f64> = mids
            .iter()
            .copied()
            .filter(|_| lcg(&mut st, 4) < keep)
            .collect();
        if sel.is_empty() && !mids.is_empty() {
            sel.push(mids[mids.len() / 2]);
        }
        ths.insert(f, sel);
    }
    (x, y, ths)
}

/// Select with the support-sorted scan and / or the stats-free entry
/// (capped DPs) returns the full-order, full-DP pool bitwise and in order:
/// binary and 3-class, Rush / Gini criteria, max_k 2..5, beams 1..64,
/// random supports (tiny ones included), ub_hint none / wide / tight,
/// unsafe filter on/off.
#[test]
fn select_pool_identical_scan_and_stats_paths() {
    let bank = ScratchBank::new();
    let (mut n_pools, mut n_cands, mut n_nonempty) = (0usize, 0usize, 0usize);
    let mut case = 0u64;
    for n_classes in [2u32, 3] {
        for max_k in 2usize..=5 {
            for &beam in &[1usize, 2, 4, 16, 64] {
                for crit in [DpCriterion::Rush, DpCriterion::Gini] {
                    if crit == DpCriterion::Gini && beam > 4 {
                        continue;
                    }
                    case += 1;
                    let n = 80 + (case as usize * 37) % 160;
                    let n_feat = 2 + (case as usize) % 3;
                    let (x, y, ths) = custom_data(n, n_feat, n_classes, 1000 + case * 7919);
                    let store = DatasetStore::new(x, y, None, n_classes as usize, Some(&ths), None);
                    let mut cfg = phase2_test_config(0.01, 3, max_k, 0.5);
                    cfg.n_classes = n_classes as usize;
                    cfg.n_features = n_feat;
                    cfg.feature_names = (0..n_feat).map(|f| format!("x{f}")).collect();
                    cfg.feature_types = vec![FeatureType::Num; n_feat];
                    cfg.thresholds_per_feature = Some(ths);
                    cfg.initial_used_features = FeatureMask::all(n_feat);
                    cfg.bypass_beam = false;
                    cfg.beam_width = beam;
                    cfg.dp_criterion = crit;
                    cfg.sequential_candidate_gen = case % 2 == 0;
                    cfg.unsafe_filter_depth_threshold = if case % 5 == 0 { 2 } else { 0 };
                    let mut st = 77u64 + case;
                    for trial in 0..14 {
                        let rows: Vec<u32> = match trial {
                            0 => (0..n as u32).collect(),
                            1 => (0..n as u32).take(2 + (case as usize % 4)).collect(),
                            _ => {
                                let keep = 1 + lcg(&mut st, 9);
                                (0..n as u32).filter(|_| lcg(&mut st, 10) < keep).collect()
                            }
                        };
                        if rows.len() < 2 {
                            continue;
                        }
                        let support = store.support_from_indices(&rows);
                        let ub = match trial % 4 {
                            0 => None,
                            1 => Some(0.5),
                            2 => Some(10.0),
                            _ => Some(0.004),
                        };
                        let depth = Some(1 + (trial % 3) as i32);
                        let run = |sort: bool, stats_free: bool| {
                            generate_candidates_for_support_opts(
                                &store,
                                &support,
                                &cfg,
                                &FeatureMask::all(n_feat),
                                n,
                                ub,
                                None,
                                &bank,
                                None,
                                None,
                                depth,
                                SelectOpts {
                                    support_sort: sort,
                                    stats_free,
                                },
                            )
                            .0
                        };
                        let want = pool_key(&run(false, false));
                        for sort in [false, true] {
                            for stats_free in [false, true] {
                                let got = pool_key(&run(sort, stats_free));
                                assert_eq!(
                                    got,
                                    want,
                                    "pool differs: sort={sort} stats_free={stats_free} \
                                     classes={n_classes} max_k={max_k} beam={beam} rush={} n={}",
                                    crit == DpCriterion::Rush,
                                    rows.len()
                                );
                            }
                        }
                        n_pools += 1;
                        n_cands += want.len();
                        n_nonempty += (!want.is_empty()) as usize;
                    }
                }
            }
        }
    }
    assert!(
        n_pools > 800 && n_nonempty > 500 && n_cands > 20_000,
        "{n_pools} pools / {n_nonempty} non-empty / {n_cands} candidates"
    );
}

/// Per memo key, in record order: (feature, threshold bits, group_map,
/// child supports' words, split_upper bits), plus the key's (lb, ub) bits
/// and explored scope bits.
type NodeDump = (
    Vec<(usize, Vec<u64>, Option<Vec<usize>>, Vec<Vec<u64>>, u64)>,
    u64,
    u64,
    u64,
);

fn dump_nodes(s: &BranchAndBoundSolver) -> HashMap<(Vec<u64>, usize, i32), NodeDump> {
    let mut out = HashMap::new();
    for e in s.search.memo.iter() {
        let sp = e.value();
        let recs = sp
            .rashomon_indices
            .iter()
            .map(|&i| {
                let r = s.worker.split_pool.get(i);
                (
                    r.feature_index,
                    r.thresholds.iter().map(|t| t.to_bits()).collect(),
                    r.group_map.clone(),
                    r.child_keys
                        .iter()
                        .map(|&c| s.search.memo_key_arena.get(c).support.words.to_vec())
                        .collect(),
                    r.split_upper.to_bits(),
                )
            })
            .collect();
        let k = e.key();
        out.insert(
            (k.support.words.to_vec(), k.support.n_samples, k.depth),
            (
                recs,
                sp.lowerbound.to_bits(),
                sp.upperbound.to_bits(),
                sp.scope.to_bits(),
            ),
        );
    }
    out
}

/// Small solve configs over `custom_data` (store thresholds = Select's).
fn explore_cases() -> Vec<(Arc<DatasetStore>, SolverConfig)> {
    let mut out = Vec::new();
    for case in 0..12u64 {
        let n_classes = if case % 4 == 3 { 3 } else { 2 };
        let n = 60 + (case as usize * 23) % 80;
        let n_feat = 2 + (case as usize) % 2;
        let (x, y, ths) = custom_data(n, n_feat, n_classes, 50 + case * 131);
        let store = Arc::new(DatasetStore::new(
            x,
            y,
            None,
            n_classes as usize,
            Some(&ths),
            None,
        ));
        let max_k = 2 + (case as usize) % 3;
        let depth = 2 + (case % 2) as i32;
        let mut cfg = phase2_test_config(0.02, depth, max_k, 0.32 + 0.02 * (case % 3) as f64);
        cfg.n_classes = n_classes as usize;
        cfg.n_features = n_feat;
        cfg.feature_names = (0..n_feat).map(|f| format!("x{f}")).collect();
        cfg.feature_types = vec![FeatureType::Num; n_feat];
        cfg.thresholds_per_feature = Some(ths);
        cfg.initial_used_features = FeatureMask::all(n_feat);
        cfg.bypass_beam = case % 3 == 0;
        cfg.beam_width = [1usize, 2, 4][case as usize % 3];
        cfg.step_b_remaining = (case % 3) as i32;
        cfg.weight_total = n as f64;
        out.push((store, cfg));
    }
    out
}

/// `cfg` with its `RashomonTuning` changed by `f`.
fn tuned(cfg: &SolverConfig, f: impl FnOnce(&mut RashomonTuning)) -> SolverConfig {
    let mut c = cfg.clone();
    f(&mut c.rash_tuning);
    c
}

/// A Step-B style solver seeded with the Step-A solver's lb caches
/// (`take_rash_lb_cache` / `seed_rash_lb_cache`, a pure cache) records the
/// same splits (same order, children, bounds, scopes) at every node as the
/// same solver without them; the shared-pool, lb-Select, explore-Select
/// and swept depth-1 paths all run.
#[test]
fn lb_cache_handoff_identical() {
    let (mut n_nodes, mut n_recs, mut n_seeded) = (0usize, 0usize, 0usize);
    let mut prof_all = [0u64; 4];
    for (store, cfg) in explore_cases() {
        let n = store.n_samples();
        let mut a = BranchAndBoundSolver::new(store.clone(), cfg.clone(), None, None);
        a.solve();
        let dump_a = dump_nodes(&a);
        let mut cfg_b = cfg.clone();
        cfg_b.rashomon_bound = cfg.rashomon_bound * 0.8;
        cfg_b.step_b_remaining = 0;
        let rows: Vec<u32> = (0..n as u32).filter(|i| i % 3 != 1).collect();
        let sup = store.support_from_indices(&rows);
        let run_b = |cache: Option<multiway_solver::solver::RashLbCache>| {
            let mut b =
                BranchAndBoundSolver::new(store.clone(), cfg_b.clone(), None, Some(sup.clone()));
            b.set_rashomon_select_ub(cfg.rashomon_bound);
            let seeded = cache.map_or(false, |c| b.seed_rash_lb_cache(c));
            b.solve_seeded(&[(sup.clone(), cfg.rashomon_bound * 0.8)]);
            (dump_nodes(&b), seeded, b)
        };
        let (want, _, _) = run_b(None);
        let (got, seeded, b) = run_b(Some(a.take_rash_lb_cache()));
        assert!(
            got == want,
            "seeded Step-B records differ from the unseeded solve"
        );
        n_seeded += seeded as usize;
        for (i, s) in [&a, &b].iter().enumerate() {
            let p = &s.worker.profile;
            prof_all[0] += p.n_select_shared_hits;
            prof_all[1] += p.n_lb_selects;
            prof_all[2] += p.n_explore_selects;
            prof_all[3] += p.n_lb_d1_swept * (i == 0) as u64;
        }
        n_nodes += dump_a.len() + want.len();
        n_recs += dump_a.values().map(|v| v.0.len()).sum::<usize>();
    }
    assert!(
        n_nodes > 300 && n_recs > 300 && n_seeded > 0,
        "{n_nodes} nodes / {n_recs} records / {n_seeded} seeded"
    );
    assert!(
        prof_all.iter().all(|&v| v > 0),
        "speed paths idle: {prof_all:?}"
    );
    eprintln!("lb cache hand-off: {n_nodes} nodes / {n_recs} records / [shared hits, lb selects, explore selects, swept] = {prof_all:?}");
}

/// A tiny lb memo cap (`lb_memo_max_entries` = `lb_d1s_max_entries` = 3,
/// forcing evictions) gives the same explore records and the same
/// `bounded_completion_min` values as the default (here uncapped) memo.
#[test]
fn lb_memo_cap_identical() {
    let mut n_evict = 0u64;
    let mut n_lb = 0usize;
    for (store, cfg) in explore_cases() {
        let n = store.n_samples();
        let n_feat = cfg.n_features;
        let run = |cap: Option<usize>| {
            let cfg = match cap {
                Some(c) => tuned(&cfg, |t| {
                    t.lb_memo_max_entries = c;
                    t.lb_d1s_max_entries = c;
                }),
                None => cfg.clone(),
            };
            let r = {
                let mut a = BranchAndBoundSolver::new(store.clone(), cfg.clone(), None, None);
                a.solve();
                let dump = dump_nodes(&a);
                let mut c = BranchAndBoundSolver::new(store.clone(), cfg.clone(), None, None);
                let mut st = 11u64 + n as u64;
                let mut lbs = Vec::new();
                for t in 0..30 {
                    let keep = 2 + lcg(&mut st, 8);
                    let rows: Vec<u32> =
                        (0..n as u32).filter(|_| lcg(&mut st, 10) < keep).collect();
                    if rows.len() < 2 {
                        continue;
                    }
                    let sup = store.support_from_indices(&rows);
                    let budget = [0.02, 0.08, 0.15, 0.3, 1.0][t % 5];
                    for d in 1..=3 {
                        lbs.push(
                            c.bounded_completion_min(&sup, &FeatureMask::all(n_feat), d, budget)
                                .map(f64::to_bits),
                        );
                    }
                }
                let ev =
                    a.worker.profile.n_lb_memo_evictions + c.worker.profile.n_lb_memo_evictions;
                (dump, lbs, ev)
            };
            r
        };
        let want = run(None);
        assert_eq!(want.2, 0, "evictions at the default cap");
        let got = run(Some(3));
        assert!(
            got.0 == want.0,
            "explore records differ under a tiny memo cap"
        );
        assert_eq!(got.1, want.1, "completion lbs differ under a tiny memo cap");
        n_evict += got.2;
        n_lb += want.1.iter().filter(|v| v.is_some()).count();
    }
    assert!(
        n_evict > 100 && n_lb > 100,
        "{n_evict} evictions / {n_lb} lbs"
    );
}

/// `partition_support_le` children built straight into their Arc (fused
/// popcount) have the same words, row count and fingerprint as the row
/// partition (`partition_support`) and as `SupportMask::from_words`;
/// `le_bin_support` gives the same per bin.
#[test]
fn child_mask_fast_identical() {
    let mut n_checked = 0usize;
    for case in 0..10u64 {
        let n_classes = 2 + (case % 2) as u32;
        let n = 70 + (case as usize * 41) % 300;
        let n_feat = 2 + (case as usize) % 3;
        let (x, y, ths) = custom_data(n, n_feat, n_classes, 300 + case * 17);
        let store = DatasetStore::new(x, y, None, n_classes as usize, Some(&ths), None);
        let mut st = 9u64 + case;
        for trial in 0..8 {
            let keep = 1 + lcg(&mut st, 10);
            let rows: Vec<u32> = (0..n as u32).filter(|_| lcg(&mut st, 10) < keep).collect();
            let sup = store.support_from_indices(&rows);
            for f in 0..n_feat {
                let t = store.feature_thresholds()[f].to_vec();
                if t.is_empty() {
                    continue;
                }
                let k = 1 + (trial % t.len().min(4));
                let cut: Vec<f64> = t
                    .iter()
                    .copied()
                    .step_by((t.len() / k).max(1))
                    .take(k)
                    .collect();
                let fast = store.partition_support_le(&sup, f, &cut, None).unwrap();
                let slow = store.partition_support(&sup, f, &cut, None);
                let le = store.le_masks_for(f, &cut).unwrap();
                assert_eq!(fast.len(), slow.len());
                for (b, (a, c)) in fast.iter().zip(&slow).enumerate() {
                    let r = SupportMask::from_words(c.words.to_vec(), c.n_samples);
                    let one = DatasetStore::le_bin_support(&sup, &le, b);
                    for m in [c, &r, &one] {
                        assert!(a.words[..] == m.words[..], "child words differ (bin {b})");
                        assert_eq!(a.n_samples, m.n_samples, "child count differs (bin {b})");
                        assert_eq!(
                            a.fingerprint, m.fingerprint,
                            "child fingerprint differs (bin {b})"
                        );
                    }
                    assert_eq!(
                        a.n_samples,
                        a.words
                            .iter()
                            .map(|w| w.count_ones() as usize)
                            .sum::<usize>()
                    );
                    n_checked += 1;
                }
            }
        }
    }
    assert!(n_checked > 500, "only {n_checked} children checked");
}

/// Select's rank table: the store's row -> bin table is
/// `searchsorted_right` (count of thresholds <= x) of every row for the
/// first list it is built from, and None for a different list.
#[test]
fn select_rank_table_matches() {
    for case in 0..6u64 {
        let n = 50 + case as usize * 31;
        let (x, y, ths) = custom_data(n, 3, 2, 900 + case);
        let store = DatasetStore::new(x.clone(), y, None, 2, Some(&ths), None);
        for f in 0..3 {
            let t = &ths[&f];
            let rank = store
                .select_rank(f, t)
                .expect("first list builds the table")
                .to_vec();
            for i in 0..n {
                let want = t.iter().filter(|&&s| s <= x[[i, f]]).count();
                assert_eq!(rank[i] as usize, want, "row {i} feature {f}");
            }
            assert_eq!(store.select_rank(f, t).unwrap(), &rank[..]);
            let mut other = t.clone();
            other.push(1e9);
            assert!(
                store.select_rank(f, &other).is_none(),
                "a different list must not reuse the table"
            );
        }
    }
}

/// `lb_threads`: Step A, a seeded Step B and the completion lbs give the
/// same records (splits, children, bounds, scopes) and values at T = 1, 2
/// and 4 threads; the T = 4 run repeated gives the same again. The pass-1
/// pool must have run.
#[test]
fn lb_threads_identical() {
    let _g = STATS_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let (runs0, tasks0) = rash_par_stats();
    let (mut n_nodes, mut lb_selects_par) = (0usize, 0u64);
    for (store, cfg) in explore_cases() {
        let n = store.n_samples();
        let n_feat = cfg.n_features;
        let run = |t: usize| {
            let cfg = tuned(&cfg, |tu| tu.lb_threads = t);
            let r = {
                let mut a = BranchAndBoundSolver::new(store.clone(), cfg.clone(), None, None);
                let ra = a.solve();
                let dump_a = dump_nodes(&a);
                let memo_a = (
                    a.search.memo.len(),
                    ra.n_subproblems,
                    ra.n_explored,
                    ra.timed_out,
                );
                let cache = a.take_rash_lb_cache();
                let mut cfg_b = cfg.clone();
                cfg_b.rashomon_bound = cfg.rashomon_bound * 0.8;
                cfg_b.step_b_remaining = 0;
                let rows: Vec<u32> = (0..n as u32).filter(|i| i % 3 != 1).collect();
                let sup = store.support_from_indices(&rows);
                let mut b =
                    BranchAndBoundSolver::new(store.clone(), cfg_b, None, Some(sup.clone()));
                b.set_rashomon_select_ub(cfg.rashomon_bound);
                b.seed_rash_lb_cache(cache);
                b.solve_seeded(&[(sup, cfg.rashomon_bound * 0.8)]);
                let dump_b = dump_nodes(&b);
                let mut c = BranchAndBoundSolver::new(store.clone(), cfg.clone(), None, None);
                let mut st = 5u64 + n as u64;
                let mut lbs = Vec::new();
                for t in 0..30 {
                    let keep = 2 + lcg(&mut st, 8);
                    let rows: Vec<u32> =
                        (0..n as u32).filter(|_| lcg(&mut st, 10) < keep).collect();
                    if rows.len() < 2 {
                        continue;
                    }
                    let sup = store.support_from_indices(&rows);
                    let budget = [0.02, 0.08, 0.15, 0.3, 1.0][t % 5];
                    for d in 1..=3 {
                        lbs.push(
                            c.bounded_completion_min(&sup, &FeatureMask::all(n_feat), d, budget)
                                .map(f64::to_bits),
                        );
                    }
                }
                let sel = a.worker.profile.n_lb_selects + b.worker.profile.n_lb_selects;
                (dump_a, dump_b, lbs, memo_a, sel)
            };
            r
        };
        let want = run(1);
        for t in [2usize, 4] {
            for _ in 0..if t == 4 { 5 } else { 1 } {
                let got = run(t);
                assert!(got.0 == want.0, "Step-A records differ (T={t})");
                assert!(got.1 == want.1, "seeded Step-B records differ (T={t})");
                assert_eq!(got.2, want.2, "completion lbs differ (T={t})");
                assert_eq!(
                    got.3, want.3,
                    "memo size / subproblems / explored differ (T={t})"
                );
                lb_selects_par += got.4;
            }
        }
        n_nodes += want.0.len() + want.1.len();
    }
    let (runs, tasks) = rash_par_stats();
    assert!(
        runs > runs0 && lb_selects_par > 0,
        "pass 1 never ran ({} runs, {lb_selects_par} lb selects)",
        runs - runs0
    );
    eprintln!(
        "lb threads identity: {n_nodes} nodes, {} pass-1 runs, {} tasks",
        runs - runs0,
        tasks - tasks0
    );
}

/// `lb_threads` under a deadline: the main thread decides the time-out
/// (`timed_out` set; a zero limit gives the T = 1 result), and pass 1 cut
/// by the deadline leaves no wrong fact behind: every full-key lb memo
/// entry of a timed-out T = 4 solve equals a fresh untimed search (`Min`
/// bitwise, `Above(b)` with the minimum above b - 1e-9).
#[test]
fn lb_threads_timeout_decided_by_pass2() {
    let (mut n_checked, mut n_timed) = (0usize, 0usize);
    for (store, cfg) in explore_cases() {
        let n_feat = cfg.n_features;
        let solve_at = |t: usize, limit: f64| {
            let mut c = tuned(&cfg, |tu| tu.lb_threads = t);
            c.time_limit = Some(limit);
            let mut a = BranchAndBoundSolver::new(store.clone(), c, None, None);
            let r = a.solve();
            (a, r)
        };
        let (a1, r1) = solve_at(1, 0.0);
        let (a4, r4) = solve_at(4, 0.0);
        assert!(
            r1.timed_out && r4.timed_out,
            "zero time limit must time out"
        );
        assert!(
            dump_nodes(&a1) == dump_nodes(&a4),
            "zero-limit records differ"
        );
        for limit in [2e-4, 1e-3, 4e-3] {
            let (mut a, r) = solve_at(4, limit);
            n_timed += r.timed_out as usize;
            let memo = a.take_rash_lb_cache().memo;
            let mut fresh = BranchAndBoundSolver::new(store.clone(), cfg.clone(), None, None);
            for ((sup, d), e) in memo {
                let v = fresh.bounded_completion_min(&sup, &FeatureMask::all(n_feat), d, 1e9);
                match e {
                    multiway_solver::solver::RashLbEntry::Min(m) => {
                        assert_eq!(
                            v.map(f64::to_bits),
                            Some(m.to_bits()),
                            "Min fact differs (n={}, d={d})",
                            sup.n_samples
                        )
                    }
                    multiway_solver::solver::RashLbEntry::Above(b) => {
                        // A minimum at the bound itself (to rounding) is
                        // certified above it by the pruned search (as in a
                        // sequential solve): compare at 1e-9.
                        assert!(
                            v.map_or(false, |x| x > b - 1e-9),
                            "Above({b}) fact wrong: min {v:?} (n={}, d={d})",
                            sup.n_samples
                        )
                    }
                }
                n_checked += 1;
            }
        }
    }
    assert!(n_checked > 0, "no lb facts checked");
    eprintln!("lb threads timeout: {n_timed} timed-out solves, {n_checked} facts checked");
}

fn tree_key(t: &multiway_solver::tree::TreeNode) -> String {
    use multiway_solver::tree::TreeNode;
    match t {
        TreeNode::Leaf { prediction, .. } => format!("L{prediction}"),
        TreeNode::Internal {
            feature_index,
            thresholds,
            children,
            group_map,
        } => format!(
            "N{feature_index}{:?}{:?}({})",
            thresholds.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            group_map,
            children.iter().map(tree_key).collect::<Vec<_>>().join(",")
        ),
    }
}

/// Step A at `cfg` (depth `cfg.max_depth`, `cfg.step_b_remaining` below),
/// then Step B through the snapshot build (Step-A lb caches handed over
/// as in `solve_rashomon`), Step B's deadline
/// `deadline` after the Step-A solve. Returns the tree count, the sorted
/// tree keys, the number of first-level placeholders and of timed-out ones.
fn stepb_fit(
    store: &Arc<DatasetStore>,
    cfg: &SolverConfig,
    deadline: Option<std::time::Duration>,
) -> (u128, Vec<String>, u32, u32) {
    use multiway_solver::types::{DeltaBudget, PrefixLeafFillConfig, RashomonSet};
    let mut a = BranchAndBoundSolver::new(store.clone(), cfg.clone(), None, None);
    let ra = a.solve();
    let cache = a.take_rash_lb_cache();
    let fill = PrefixLeafFillConfig {
        store: store.clone(),
        leaf_base_config: cfg.clone(),
        reg: cfg.reg,
        leaf_reg_cap: 0.1,
        leaf_short_circuit_eps: 1e-4,
        n_data: store.n_samples(),
        fill_parallel: true,
        verbose: false,
        deadline: deadline.map(|d| std::time::Instant::now() + d),
        rash_lb_cache: cache
            .key
            .is_some()
            .then(|| Arc::new(Mutex::new(Some(cache)))),
    };
    let total = cfg.max_depth + cfg.step_b_remaining;
    let mut snap = a.into_search_snapshot(ra.objective, 0.1, DeltaBudget::default(), 1e9, total);
    snap.prefix_leaf_fill = Some(fill);
    let mut set = RashomonSet::Pending {
        snapshot: Some(Box::new(snap)),
    };
    let dag = set.dag_mut();
    let mut keys: Vec<String> = (0..dag.tree_count)
        .map(|i| tree_key(&get_tree_at_index_factored(dag, i, &[])))
        .collect();
    keys.sort();
    (
        dag.tree_count,
        keys,
        dag.delta_budget.n_prefix_leaves,
        dag.delta_budget.n_leaves_timed_out,
    )
}

/// Step-B cases: `explore_cases` with Step B, plus Step-A depth 1 with two
/// or three Step-B levels below (2-3 remaining).
fn stepb_cases() -> Vec<(Arc<DatasetStore>, SolverConfig)> {
    let mut cases = explore_cases();
    for (i, (store, cfg)) in explore_cases().into_iter().enumerate().take(6) {
        let mut c = cfg;
        c.max_depth = 1;
        c.step_b_remaining = 2 + (i % 2) as i32;
        cases.push((store, c));
    }
    cases.retain(|(_, c)| c.step_b_remaining > 0);
    cases
}

/// `stepb_batches`: Step B with each level's groups split into 1, 3 or 8
/// parallel batches (one read-only level base, per-batch deltas merged at
/// the level end) gives the same tree set (count and sorted tree keys) as
/// the unbatched level, with one (Step-A depth 2-3) and two or three Step-B
/// levels (Step-A depth 1, 2-3 below); also with `lb_threads` = 8 (per
/// batch 8 / B: pass 1 inside B = 3 batches). Some level must have run > 1
/// batch.
#[test]
fn stepb_batches_same_tree_set() {
    let _g = STATS_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let cases = stepb_cases();
    let (mut n_multi, mut n_trees) = (0usize, 0u128);
    let (mut n_batched_runs, mut n_split_runs, mut n_runs, mut n_par_runs) =
        (0usize, 0usize, 0usize, 0usize);
    // (lb_threads, batch counts).
    let variants: [(usize, &[usize]); 2] = [(1, &[1, 3, 8]), (8, &[3, 8])];
    for (store, cfg) in &cases {
        let want = stepb_fit(store, &tuned(cfg, |t| t.stepb_batches = 0), None);
        for &(t, bs) in &variants {
            for &b in bs {
                let c = tuned(cfg, |tu| {
                    tu.stepb_batches = b;
                    tu.lb_threads = t;
                });
                let (l0, s0) = rash_stepb_batch_stats();
                let p0 = rash_par_stats().0;
                let got = stepb_fit(store, &c, None);
                let (l1, s1) = rash_stepb_batch_stats();
                if t > 1 && s1 - s0 > l1 - l0 {
                    n_par_runs += (rash_par_stats().0 > p0) as usize;
                }
                n_batched_runs += (l1 > l0) as usize;
                if s1 - s0 > l1 - l0 {
                    n_split_runs += 1;
                }
                n_runs += 1;
                let tag = format!(
                    "B={b}, T={t}, depth {} + {}",
                    cfg.max_depth, cfg.step_b_remaining
                );
                assert_eq!(got.0, want.0, "tree count differs ({tag})");
                assert!(got.1 == want.1, "tree set differs ({tag})");
                assert_eq!(got.2, want.2, "placeholders differ ({tag})");
                assert_eq!(got.3, 0, "timed out ({tag})");
            }
        }
        n_trees += want.0;
        if cfg.step_b_remaining > cfg.max_depth && want.2 >= 2 {
            n_multi += 1;
        }
    }
    assert!(
        n_multi > 0,
        "no multi-level Step-B case with >= 2 placeholders"
    );
    assert!(n_batched_runs > 0, "batched path never ran");
    assert!(n_split_runs > 0, "no batched level ran more than one batch");
    assert!(
        n_par_runs > 0,
        "no pass 1 (lb_threads) ran inside a batched level"
    );
    eprintln!(
        "stepb batches: {} cases, {n_multi} multi-level, {n_trees} trees; {n_runs} batched runs, \
         {n_batched_runs} through the batched path, {n_split_runs} with > 1 batch, \
         {n_par_runs} with pass 1 inside the batches",
        cases.len()
    );
}

/// `stepb_batches` with a Step-B deadline: no panic for deadlines from
/// already past (every placeholder timed out) to a few ms (some batches
/// may start past it and are skipped), with `lb_threads` 1 and 4.
#[test]
fn stepb_batches_deadline() {
    let cases = stepb_cases();
    let (mut n_timed, mut n_timed_past) = (0usize, 0usize);
    for (store, cfg) in &cases {
        for t in [1usize, 4] {
            for us in [0u64, 20, 100, 400, 1500, 5000] {
                let c = tuned(cfg, |tu| {
                    tu.stepb_batches = 8;
                    tu.lb_threads = t;
                });
                let got = stepb_fit(store, &c, Some(std::time::Duration::from_micros(us)));
                if got.3 > 0 {
                    n_timed += 1;
                    n_timed_past += (us == 0) as usize;
                }
            }
        }
    }
    assert!(
        n_timed_past > 0,
        "no past-deadline run flagged as timed out"
    );
    eprintln!(
        "stepb batches deadline: {} cases, {n_timed} timed-out runs ({n_timed_past} past at start)",
        cases.len()
    );
}
