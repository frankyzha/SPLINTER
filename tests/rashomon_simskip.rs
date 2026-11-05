//! The similar-support skip of depth-3 completion scans (`rash_lb_scan`,
//! with the interval lower bound on): the interval value's Lipschitz bound
//! in the row set, and end-to-end exactness of the scans that skip
//! (answers equal to exact minima computed without any skip). Run with:
//!   cargo test --release --no-default-features --test rashomon_simskip

mod common;

use multiway_solver::solver::{rash_iv_stats, BranchAndBoundSolver};
use multiway_solver::types::{FeatureMask, SolverConfig};
use ndarray::Array2;

use common::{d1_wide_config, lcg};

/// `vals.len()` integer features with midpoint thresholds; the label is a
/// noisy function of features 0 and 1 (depth-2/3 structure), kept for
/// `skew` rows out of 10.
fn gen_data(
    n: usize,
    nc: u32,
    vals: &[u32],
    seed: u64,
    skew: u32,
) -> (Array2<f64>, Vec<u8>, Vec<Vec<f64>>) {
    let mut st = seed;
    let nf = vals.len();
    let mut x = Array2::<f64>::zeros((n, nf));
    let mut y = Vec::with_capacity(n);
    for i in 0..n {
        for (f, &m) in vals.iter().enumerate() {
            x[[i, f]] = lcg(&mut st, m) as f64;
        }
        // label: noisy function of two features (gives depth-2/3 structure)
        let base = ((x[[i, 0]] as u32 + 2 * x[[i, 1 % nf]] as u32) / 3) % nc;
        let yl = if lcg(&mut st, 10) < skew {
            base
        } else {
            lcg(&mut st, nc)
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

/// `d1_wide_config` at depth 3 with a path penalty, an optional n_total
/// override and an optional beam-1 Select pool.
fn mk(
    nc: usize,
    max_k: usize,
    ra: f64,
    pp: f64,
    ths: &[Vec<f64>],
    ntot: Option<usize>,
    beam1: bool,
) -> SolverConfig {
    mk_iv(nc, max_k, ra, pp, ths, ntot, beam1, false)
}

/// `mk` with the interval lower bound (and so the skip rule) on or off.
#[allow(clippy::too_many_arguments)]
fn mk_iv(
    nc: usize,
    max_k: usize,
    ra: f64,
    pp: f64,
    ths: &[Vec<f64>],
    ntot: Option<usize>,
    beam1: bool,
    interval: bool,
) -> SolverConfig {
    let mut c = d1_wide_config(nc, max_k, ra, ths);
    c.rash_tuning.lb_interval = interval;
    c.path_penalty = pp;
    c.max_depth = 3;
    c.n_total_override = ntot;
    if beam1 {
        c.beam_width = 1;
        c.bypass_beam = false;
    }
    c
}

/// The rule's premise: for g ⊆ c, IV(g) >= IV(c) − |c∖g|·(1 + 2·pp) / N
/// (IV = `rash_interval_completion_min2_debug`), pp ∈ {0, 0.05, 0.3}, N =
/// n (or an n_total override of 3n), split children and random subsets;
/// with pp = 0 also the uncorrected IV(c) − |c∖g| / N.
#[test]
fn simskip_lipschitz_iv() {
    let mut worst0 = f64::INFINITY;
    let mut worst_pp = f64::INFINITY;
    let (mut n_chk, mut v0, mut vpp, mut v0_pp0) = (0usize, 0usize, 0usize, 0usize);
    let mut example: Option<String> = None;
    for &(nc, max_k, ra) in &[
        (2u32, 2usize, 1.0),
        (2, 3, 1.0),
        (2, 4, 0.5),
        (3, 3, 1.0),
        (3, 2, 0.5),
        (2, 5, 1.0),
    ] {
        for &pp in &[0.0, 0.05, 0.3] {
            for &ntot_mul in &[0usize, 3] {
                for inst in 0..6u64 {
                    let n = 18 + ((inst as usize) * 11) % 40;
                    let vals: &[u32] = if inst % 2 == 0 {
                        &[5, 7, 4]
                    } else {
                        &[6, 3, 9, 2]
                    };
                    let (x, y, ths) = gen_data(
                        n,
                        nc,
                        vals,
                        7 + inst * 1013 + max_k as u64 * 31 + nc as u64,
                        5 + (inst % 4) as u32,
                    );
                    let ntot = if ntot_mul == 0 {
                        None
                    } else {
                        Some(n * ntot_mul)
                    };
                    let cfg = mk(nc as usize, max_k, ra, pp, &ths, ntot, false);
                    let mut s =
                        BranchAndBoundSolver::from_raw(x.clone(), y.clone(), cfg, None, None);
                    let nt = s.search.n_total as f64;
                    let mut st = 99 + inst;
                    for trial in 0..6 {
                        let crow: Vec<u32> = if trial == 0 {
                            (0..n as u32).collect()
                        } else {
                            (0..n as u32).filter(|_| lcg(&mut st, 10) < 7).collect()
                        };
                        if crow.len() < 4 {
                            continue;
                        }
                        let csup = s.search.store.support_from_indices(&crow);
                        let ivc = s
                            .rash_interval_completion_min2_debug(&csup, 1e9)
                            .expect("iv applies");
                        // children: split children on random store-threshold cut sets, and random subsets
                        let mut gs: Vec<Vec<u32>> = Vec::new();
                        for f in 0..ths.len() {
                            for _ in 0..3 {
                                let t = &ths[f];
                                let ncut = 1 + lcg(&mut st, (max_k - 1) as u32) as usize;
                                let mut cuts: Vec<f64> = (0..ncut)
                                    .map(|_| t[lcg(&mut st, t.len() as u32) as usize])
                                    .collect();
                                cuts.sort_by(|a, b| a.partial_cmp(b).unwrap());
                                cuts.dedup();
                                let mut bins: Vec<Vec<u32>> = vec![Vec::new(); cuts.len() + 1];
                                for &r in &crow {
                                    bins[cuts.partition_point(|c| *c <= x[[r as usize, f]])]
                                        .push(r);
                                }
                                gs.extend(bins.into_iter().filter(|b| !b.is_empty()));
                            }
                        }
                        for _ in 0..6 {
                            let keep = 3 + lcg(&mut st, 7);
                            let g: Vec<u32> = crow
                                .iter()
                                .copied()
                                .filter(|_| lcg(&mut st, 10) < keep)
                                .collect();
                            if !g.is_empty() {
                                gs.push(g);
                            }
                        }
                        for g in gs {
                            let gsup = s.search.store.support_from_indices(&g);
                            let ivg = s
                                .rash_interval_completion_min2_debug(&gsup, 1e9)
                                .expect("iv applies");
                            let d = (crow.len() - g.len()) as f64;
                            let s0 = ivg - (ivc - d / nt);
                            let spp = ivg - (ivc - d * (1.0 + 2.0 * pp) / nt);
                            n_chk += 1;
                            worst0 = worst0.min(s0);
                            worst_pp = worst_pp.min(spp);
                            if s0 < -1e-12 {
                                v0 += 1;
                                if pp == 0.0 {
                                    v0_pp0 += 1;
                                }
                                if example.is_none() {
                                    example = Some(format!("pp={pp} nc={nc} k={max_k} ra={ra} N={nt} |c|={} |g|={} ivc={ivc} ivg={ivg} ivc-d/N={} slack={s0:e}", crow.len(), g.len(), ivc - d / nt));
                                }
                            }
                            if spp < -1e-12 {
                                vpp += 1;
                            }
                        }
                    }
                }
            }
        }
    }
    eprintln!("simskip_lipschitz: checked={n_chk} viol_uncorrected={v0} (pp=0: {v0_pp0}) viol_corrected={vpp} worst_uncorr={worst0:e} worst_corr={worst_pp:e}");
    eprintln!("simskip_lipschitz: example: {example:?}");
    assert_eq!(v0_pp0, 0);
    assert_eq!(vpp, 0);
}

/// Admission slack `bounded_completion_min` adds to every budget
/// (`BranchAndBoundSolver::RASH_TOL`).
const RASH_TOL: f64 = 2e-9;

/// End to end: depth-3 and depth-2 completion minima on identical shuffled
/// query sequences (bounds around each exact minimum) from one warm solver
/// equal the exact minima of a fresh solver per (support, depth) at an
/// unbinding bound (no child budget is tight there, so nothing is skipped):
/// `Some(v)` iff v <= bound + RASH_TOL, bitwise. Returns (queries,
/// differences).
fn run_exact(interval: bool, pps: &[f64], beam1: bool) -> (usize, usize) {
    let (mut n_q, mut n_diff) = (0usize, 0usize);
    for &(nc, max_k, ra) in &[
        (2u32, 2usize, 1.0),
        (2, 3, 1.0),
        (2, 4, 0.5),
        (3, 3, 1.0),
        (3, 2, 0.5),
    ] {
        for &pp in pps {
            for inst in 0..4u64 {
                let n = 14 + ((inst as usize) * 13) % 50;
                let vals: &[u32] = if inst % 2 == 0 {
                    &[5, 7, 4]
                } else {
                    &[6, 3, 9, 2]
                };
                let (x, y, ths) = gen_data(
                    n,
                    nc,
                    vals,
                    5 + inst * 733 + max_k as u64 * 17 + nc as u64,
                    6 + (inst % 3) as u32,
                );
                let cfg = || mk_iv(nc as usize, max_k, ra, pp, &ths, None, beam1, interval);
                let mut b = BranchAndBoundSolver::from_raw(x.clone(), y.clone(), cfg(), None, None);
                let mask = FeatureMask::all(ths.len());
                let mut st = 3 + inst;
                let mut sups = Vec::new();
                for trial in 0..4 {
                    let rows: Vec<u32> = if trial == 0 {
                        (0..n as u32).collect()
                    } else {
                        (0..n as u32).filter(|_| lcg(&mut st, 10) < 7).collect()
                    };
                    if rows.len() >= 4 {
                        sups.push(b.search.store.support_from_indices(&rows));
                    }
                }
                // Exact values (fresh solver per support and depth).
                let exact = |sp: &_, dep: i32| {
                    let mut r =
                        BranchAndBoundSolver::from_raw(x.clone(), y.clone(), cfg(), None, None);
                    r.bounded_completion_min(sp, &mask, dep, 1e9).unwrap()
                };
                let ex3: Vec<f64> = sups.iter().map(|sp| exact(sp, 3)).collect();
                let ex2: Vec<f64> = sups.iter().map(|sp| exact(sp, 2)).collect();
                let mut qs: Vec<(usize, f64, i32)> = Vec::new();
                for (i, &v) in ex3.iter().enumerate() {
                    for &dl in &[-0.08, -0.03, -0.01, -1e-6, 0.0, 1e-6, 0.01, 0.05] {
                        qs.push((i, v + dl, 3));
                    }
                    qs.push((i, v - 0.02, 2));
                    qs.push((i, ex2[i], 2));
                }
                for j in (1..qs.len()).rev() {
                    let k = lcg(&mut st, (j + 1) as u32) as usize;
                    qs.swap(j, k);
                }
                for &(i, bd, dep) in &qs {
                    let v = if dep == 3 { ex3[i] } else { ex2[i] };
                    let want = (v <= bd + RASH_TOL).then_some(v);
                    let got = b.bounded_completion_min(&sups[i], &mask, dep, bd);
                    n_q += 1;
                    if want.map(f64::to_bits) != got.map(f64::to_bits) {
                        n_diff += 1;
                        if n_diff <= 3 {
                            eprintln!("DIFF interval={interval} pp={pp} nc={nc} k={max_k} ra={ra} dep={dep} bound={bd} want={want:?} got={got:?}");
                        }
                    }
                }
            }
        }
    }
    (n_q, n_diff)
}

fn simskip_count() -> u64 {
    rash_iv_stats()
        .iter()
        .find(|x| x.0 == "n_lb_simskip_skipped")
        .map_or(0, |x| x.1)
}

/// With the interval lower bound on, the skip fires and every answer is the
/// exact minimum; with it off (pool path) it never fires and the answers
/// are exact too. pp ∈ {0, 0.05, 0.3}, full and beam-1 pools.
#[test]
fn simskip_exact() {
    for interval in [true, false] {
        for beam1 in [false, true] {
            let s0 = simskip_count();
            let (n_q, n_diff) = run_exact(interval, &[0.0, 0.05, 0.3], beam1);
            let skips = simskip_count() - s0;
            eprintln!("simskip_exact interval={interval} beam1={beam1}: queries={n_q} diffs={n_diff} skips={skips}");
            assert_eq!(n_diff, 0);
            if interval {
                assert!(skips > 0, "the rule never fired: test too weak");
            } else {
                assert_eq!(skips, 0, "the skip fired without the interval lower bound");
            }
        }
    }
}
