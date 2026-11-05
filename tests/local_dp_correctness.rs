use multiway_solver::candidates::{
    collapse_bins_topb_binary_with_teacher_taxed_min_child, collapse_bins_topb_min_child,
    rush_dp_binary_pub, shape_collapse_topb_binary_with_teacher_smoothed_taxed,
    shape_collapse_topb_pairwise_multiclass_min_child,
};
use multiway_solver::types::DpCriterion;

const EPS: f64 = 1e-10;

fn combinations(n: usize, k: usize) -> Vec<Vec<usize>> {
    fn rec(start: usize, n: usize, k: usize, cur: &mut Vec<usize>, out: &mut Vec<Vec<usize>>) {
        if cur.len() == k {
            out.push(cur.clone());
            return;
        }
        let remaining = k - cur.len();
        for v in start..=n - remaining {
            cur.push(v);
            rec(v + 1, n, k, cur, out);
            cur.pop();
        }
    }

    if k == 0 {
        return vec![vec![]];
    }
    if k > n {
        return vec![];
    }
    let mut out = Vec::new();
    rec(1, n + 1, k, &mut Vec::new(), &mut out);
    out
}

fn binary_group_cost(
    bin_n: &[i64],
    bin_c0: &[i64],
    start: usize,
    end: usize,
    n_total: f64,
    criterion: DpCriterion,
) -> f64 {
    let n: i64 = bin_n[start..end].iter().sum();
    let c0: i64 = bin_c0[start..end].iter().sum();
    let c1 = n - c0;
    match criterion {
        DpCriterion::Misclass => c0.min(c1) as f64 / n_total,
        DpCriterion::Gini => {
            let nf = n as f64;
            (nf - ((c0 * c0 + c1 * c1) as f64 / nf)) / n_total
        }
        _ => panic!("test helper only supports gini/misclass"),
    }
}

fn multiclass_group_cost(
    bin_n: &[i64],
    bin_counts: &[Vec<i64>],
    start: usize,
    end: usize,
    n_total: f64,
    criterion: DpCriterion,
) -> f64 {
    let n: i64 = bin_n[start..end].iter().sum();
    let n_classes = bin_counts[0].len();
    let mut counts = vec![0i64; n_classes];
    for row in &bin_counts[start..end] {
        for (dst, src) in counts.iter_mut().zip(row.iter()) {
            *dst += *src;
        }
    }
    match criterion {
        DpCriterion::Misclass => (n - counts.iter().copied().max().unwrap()) as f64 / n_total,
        DpCriterion::Gini => {
            let nf = n as f64;
            let sq_sum: i64 = counts.iter().map(|c| c * c).sum();
            (nf - (sq_sum as f64 / nf)) / n_total
        }
        _ => panic!("test helper only supports gini/misclass"),
    }
}

fn brute_binary_costs(
    bin_n: &[i64],
    bin_c0: &[i64],
    max_k: usize,
    criterion: DpCriterion,
    leaf_reg: f64,
    min_child: i64,
) -> Vec<(usize, f64)> {
    let l = bin_n.len();
    let n_total = bin_n.iter().sum::<i64>() as f64;
    let mut out = Vec::new();
    for k in 2..=max_k.min(l) {
        for cuts in combinations(l - 1, k - 1) {
            let starts = std::iter::once(0).chain(cuts.iter().copied());
            let ends = cuts.iter().copied().chain(std::iter::once(l));
            let mut cost = 0.0;
            let mut feasible = true;
            for (s, e) in starts.zip(ends) {
                if bin_n[s..e].iter().sum::<i64>() < min_child {
                    feasible = false;
                    break;
                }
                cost += binary_group_cost(bin_n, bin_c0, s, e, n_total, criterion) + leaf_reg;
            }
            if feasible {
                out.push((k, cost));
            }
        }
    }
    out.sort_by(|a, b| {
        a.0.cmp(&b.0)
            .then(a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
    });
    out
}

fn brute_multiclass_costs(
    bin_n: &[i64],
    bin_counts: &[Vec<i64>],
    max_k: usize,
    criterion: DpCriterion,
    leaf_reg: f64,
    min_child: i64,
) -> Vec<(usize, f64)> {
    let l = bin_n.len();
    let n_total = bin_n.iter().sum::<i64>() as f64;
    let mut out = Vec::new();
    for k in 2..=max_k.min(l) {
        for cuts in combinations(l - 1, k - 1) {
            let starts = std::iter::once(0).chain(cuts.iter().copied());
            let ends = cuts.iter().copied().chain(std::iter::once(l));
            let mut cost = 0.0;
            let mut feasible = true;
            for (s, e) in starts.zip(ends) {
                if bin_n[s..e].iter().sum::<i64>() < min_child {
                    feasible = false;
                    break;
                }
                cost +=
                    multiclass_group_cost(bin_n, bin_counts, s, e, n_total, criterion) + leaf_reg;
            }
            if feasible {
                out.push((k, cost));
            }
        }
    }
    out.sort_by(|a, b| {
        a.0.cmp(&b.0)
            .then(a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
    });
    out
}

fn shape_tax(order: &[usize], start: usize, end: usize, mu_node: f64) -> f64 {
    let l = order.len();
    let mut pos_of = vec![0usize; l];
    for (shape_pos, &orig) in order.iter().enumerate() {
        pos_of[orig] = shape_pos;
    }
    let mut crossed = 0u32;
    for b in 0..l.saturating_sub(1) {
        let lhs_in = pos_of[b] >= start && pos_of[b] < end;
        let rhs_in = pos_of[b + 1] >= start && pos_of[b + 1] < end;
        if lhs_in ^ rhs_in {
            crossed += 1;
        }
    }
    0.5 * mu_node * crossed as f64
}

fn brute_shape_binary_costs(
    bin_n: &[i64],
    bin_c0: &[i64],
    teacher_c0_mass: &[f64],
    max_k: usize,
    criterion: DpCriterion,
    leaf_reg: f64,
    mu_node: f64,
    min_child: i64,
) -> Vec<(usize, f64)> {
    let mut order: Vec<usize> = (0..bin_n.len()).collect();
    order.sort_by(|&a, &b| {
        (teacher_c0_mass[a] / bin_n[a] as f64)
            .partial_cmp(&(teacher_c0_mass[b] / bin_n[b] as f64))
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let sorted_n: Vec<i64> = order.iter().map(|&i| bin_n[i]).collect();
    let sorted_c0: Vec<i64> = order.iter().map(|&i| bin_c0[i]).collect();
    let l = sorted_n.len();
    let n_total = sorted_n.iter().sum::<i64>() as f64;
    let mut out = Vec::new();
    for k in 2..=max_k.min(l) {
        for cuts in combinations(l - 1, k - 1) {
            let starts = std::iter::once(0).chain(cuts.iter().copied());
            let ends = cuts.iter().copied().chain(std::iter::once(l));
            let mut cost = 0.0;
            let mut feasible = true;
            for (s, e) in starts.zip(ends) {
                if sorted_n[s..e].iter().sum::<i64>() < min_child {
                    feasible = false;
                    break;
                }
                cost += binary_group_cost(&sorted_n, &sorted_c0, s, e, n_total, criterion)
                    + leaf_reg
                    + shape_tax(&order, s, e, mu_node);
            }
            if feasible {
                out.push((k, cost));
            }
        }
    }
    out.sort_by(|a, b| {
        a.0.cmp(&b.0)
            .then(a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
    });
    out
}

fn order_equivalence_key(order: &[usize]) -> Vec<usize> {
    let reversed_is_smaller = order.iter().rev().cmp(order.iter()) == std::cmp::Ordering::Less;
    let mut key = Vec::with_capacity(order.len() + 1);
    key.push(order.len());
    if reversed_is_smaller {
        key.extend(order.iter().rev().copied());
    } else {
        key.extend(order.iter().copied());
    }
    key
}

fn brute_pairwise_multiclass_shape_costs(
    bin_n: &[i64],
    bin_counts: &[Vec<i64>],
    teacher_mass: &[f64],
    max_k: usize,
    criterion: DpCriterion,
    leaf_reg: f64,
    mu_node: f64,
    min_child: i64,
) -> Vec<(usize, f64)> {
    let n_classes = bin_counts[0].len();
    let mut present = Vec::new();
    for cls in 0..n_classes {
        if bin_counts.iter().map(|row| row[cls]).sum::<i64>() > 0 {
            present.push(cls);
        }
    }
    let mut seen = std::collections::HashSet::new();
    let mut out = Vec::new();
    for li in 0..present.len() {
        for ri in li + 1..present.len() {
            let left = present[li];
            let right = present[ri];
            let mut order: Vec<usize> = (0..bin_n.len()).collect();
            order.sort_by(|&a, &b| {
                let score_a = teacher_mass[a * n_classes + left] / bin_n[a] as f64
                    - teacher_mass[a * n_classes + right] / bin_n[a] as f64;
                let score_b = teacher_mass[b * n_classes + left] / bin_n[b] as f64
                    - teacher_mass[b * n_classes + right] / bin_n[b] as f64;
                let key_a = (score_a / 1.0e-12).round() as i64;
                let key_b = (score_b / 1.0e-12).round() as i64;
                key_a.cmp(&key_b).then(a.cmp(&b))
            });
            if !seen.insert(order_equivalence_key(&order)) {
                continue;
            }
            let sorted_n: Vec<i64> = order.iter().map(|&i| bin_n[i]).collect();
            let sorted_counts: Vec<Vec<i64>> =
                order.iter().map(|&i| bin_counts[i].clone()).collect();
            let l = sorted_n.len();
            let n_total = sorted_n.iter().sum::<i64>() as f64;
            for k in 2..=max_k.min(l) {
                for cuts in combinations(l - 1, k - 1) {
                    let starts = std::iter::once(0).chain(cuts.iter().copied());
                    let ends = cuts.iter().copied().chain(std::iter::once(l));
                    let mut cost = 0.0;
                    let mut feasible = true;
                    for (s, e) in starts.zip(ends) {
                        if sorted_n[s..e].iter().sum::<i64>() < min_child {
                            feasible = false;
                            break;
                        }
                        cost += multiclass_group_cost(
                            &sorted_n,
                            &sorted_counts,
                            s,
                            e,
                            n_total,
                            criterion,
                        ) + leaf_reg
                            + shape_tax(&order, s, e, mu_node);
                    }
                    if feasible {
                        out.push((k, cost));
                    }
                }
            }
        }
    }
    out.sort_by(|a, b| {
        a.0.cmp(&b.0)
            .then(a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
    });
    out
}

fn assert_top_costs_match(actual: &[(usize, f64)], expected: &[(usize, f64)], beam: usize) {
    let max_k = actual
        .iter()
        .chain(expected.iter())
        .map(|(k, _)| *k)
        .max()
        .unwrap_or(0);
    for k in 2..=max_k {
        let a: Vec<f64> = actual
            .iter()
            .filter(|(kk, _)| *kk == k)
            .take(beam)
            .map(|(_, c)| *c)
            .collect();
        let e: Vec<f64> = expected
            .iter()
            .filter(|(kk, _)| *kk == k)
            .take(beam)
            .map(|(_, c)| *c)
            .collect();
        assert_eq!(
            a.len(),
            e.len(),
            "arity {k} emitted a different beam length"
        );
        for (idx, (aa, ee)) in a.iter().zip(e.iter()).enumerate() {
            assert!(
                (aa - ee).abs() < EPS,
                "arity {k}, beam rank {idx}: DP cost {aa} != brute force {ee}"
            );
        }
    }
}

#[test]
fn contiguous_binary_dp_matches_exhaustive_interval_search() {
    let thresholds = [0.5, 1.5, 2.5, 3.5, 4.5];
    let bin_n = [4, 3, 5, 2, 6, 4];
    let bin_c0 = [3, 1, 0, 2, 4, 1];
    let n_total = bin_n.iter().sum::<i64>() as usize;
    let max_k = 4;
    let beam = 16;
    let beam_schedule = [beam, beam, beam];
    let leaf_reg = 0.007;
    let min_child = 3;

    for criterion in [DpCriterion::Gini, DpCriterion::Misclass] {
        let (actual_entries, _) = collapse_bins_topb_binary_with_teacher_taxed_min_child(
            &thresholds,
            &bin_n,
            &bin_c0,
            max_k,
            n_total,
            &beam_schedule,
            criterion,
            None,
            None,
            1.0,
            leaf_reg,
            None,
            min_child as usize,
        );
        let actual: Vec<(usize, f64)> = actual_entries
            .iter()
            .map(|(k, entry)| (*k, entry.cost))
            .collect();
        let expected = brute_binary_costs(&bin_n, &bin_c0, max_k, criterion, leaf_reg, min_child);
        assert_top_costs_match(&actual, &expected, beam);
    }
}

#[test]
fn contiguous_multiclass_dp_matches_exhaustive_interval_search() {
    let thresholds = [0.5, 1.5, 2.5, 3.5, 4.5];
    let bin_counts = vec![
        vec![3, 1, 0],
        vec![0, 2, 1],
        vec![2, 0, 3],
        vec![1, 4, 0],
        vec![0, 1, 5],
        vec![3, 2, 1],
    ];
    let bin_n: Vec<i64> = bin_counts.iter().map(|row| row.iter().sum()).collect();
    let n_total = bin_n.iter().sum::<i64>() as usize;
    let max_k = 4;
    let beam = 16;
    let beam_schedule = [beam, beam, beam];
    let leaf_reg = 0.005;
    let min_child = 4;

    for criterion in [DpCriterion::Gini, DpCriterion::Misclass] {
        let (actual_entries, _) = collapse_bins_topb_min_child(
            &thresholds,
            &bin_n,
            &bin_counts,
            max_k,
            n_total,
            &beam_schedule,
            criterion,
            None,
            leaf_reg,
            min_child as usize,
        );
        let actual: Vec<(usize, f64)> = actual_entries
            .iter()
            .map(|(k, entry)| (*k, entry.cost))
            .collect();
        let expected =
            brute_multiclass_costs(&bin_n, &bin_counts, max_k, criterion, leaf_reg, min_child);
        assert_top_costs_match(&actual, &expected, beam);
    }
}

#[test]
fn binary_shape_dp_matches_exhaustive_search_on_reference_order_with_boundary_tax() {
    let thresholds = [0.5, 1.5, 2.5, 3.5, 4.5];
    let bin_n = [5, 4, 6, 3, 7, 5];
    let bin_c0 = [4, 1, 0, 3, 5, 2];
    let teacher_c0_mass = [4.8, 0.7, 2.4, 0.9, 6.2, 1.4];
    let n_total = bin_n.iter().sum::<i64>() as usize;
    let max_k = 4;
    let beam = 16;
    let beam_schedule = [beam, beam, beam];
    let leaf_reg = 0.004;
    let mu_node = 0.031;
    let min_child = 4;

    let actual_entries = shape_collapse_topb_binary_with_teacher_smoothed_taxed(
        &thresholds,
        &bin_n,
        &bin_c0,
        max_k,
        n_total,
        &beam_schedule,
        DpCriterion::Gini,
        None,
        Some(&teacher_c0_mass),
        1.0,
        leaf_reg,
        0.0,
        mu_node,
        true,
        min_child as usize,
    );
    let actual: Vec<(usize, f64)> = actual_entries
        .iter()
        .map(|(k, entry)| (*k, entry.cost))
        .collect();
    let expected = brute_shape_binary_costs(
        &bin_n,
        &bin_c0,
        &teacher_c0_mass,
        max_k,
        DpCriterion::Gini,
        leaf_reg,
        mu_node,
        min_child,
    );
    assert_top_costs_match(&actual, &expected, beam);
}

#[test]
fn binary_shape_dp_ignores_empty_bins_with_boundary_tax() {
    let thresholds = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5];
    let bin_n = [5, 0, 4, 0, 6, 0, 5];
    let bin_c0 = [4, 0, 1, 0, 5, 0, 2];
    let teacher_c0_mass = [4.8, 0.0, 0.7, 0.0, 6.2, 0.0, 1.4];
    let occupied = [0usize, 2, 4, 6];
    let projected_n: Vec<i64> = occupied.iter().map(|&idx| bin_n[idx]).collect();
    let projected_c0: Vec<i64> = occupied.iter().map(|&idx| bin_c0[idx]).collect();
    let projected_teacher: Vec<f64> = occupied.iter().map(|&idx| teacher_c0_mass[idx]).collect();
    let n_total = bin_n.iter().sum::<i64>() as usize;
    let max_k = 4;
    let beam = 16;
    let beam_schedule = [beam, beam, beam];
    let leaf_reg = 0.004;
    let mu_node = 0.031;
    let min_child = 4;

    let actual_entries = shape_collapse_topb_binary_with_teacher_smoothed_taxed(
        &thresholds,
        &bin_n,
        &bin_c0,
        max_k,
        n_total,
        &beam_schedule,
        DpCriterion::Gini,
        None,
        Some(&teacher_c0_mass),
        1.0,
        leaf_reg,
        0.0,
        mu_node,
        true,
        min_child as usize,
    );
    assert!(
        actual_entries.iter().all(|(_, entry)| entry
            .group_map
            .as_ref()
            .is_some_and(|m| m.len() == bin_n.len())),
        "shape entries should expand group_map back to original bin count"
    );
    let actual: Vec<(usize, f64)> = actual_entries
        .iter()
        .map(|(k, entry)| (*k, entry.cost))
        .collect();
    let expected = brute_shape_binary_costs(
        &projected_n,
        &projected_c0,
        &projected_teacher,
        max_k,
        DpCriterion::Gini,
        leaf_reg,
        mu_node,
        min_child,
    );
    assert_top_costs_match(&actual, &expected, beam);
}

#[test]
fn rush_binary_misclass_dp_matches_exhaustive_interval_search() {
    let thresholds = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5];
    let bin_n = [5, 2, 7, 4, 6, 3, 5];
    let bin_c0 = [4, 0, 3, 1, 5, 2, 0];
    let n_total = bin_n.iter().sum::<i64>() as usize;
    let max_k = 5;
    let beam = 16;
    let beam_schedule = [beam, beam, beam, beam];
    let leaf_reg = 0.006;
    let min_child = 3;

    let (actual_entries, _) = rush_dp_binary_pub(
        &thresholds,
        &bin_n,
        &bin_c0,
        max_k,
        n_total,
        &beam_schedule,
        min_child as usize,
        f64::INFINITY,
        leaf_reg,
        1.0,
        None,
    );
    let actual: Vec<(usize, f64)> = actual_entries
        .iter()
        .map(|(k, entry)| (*k, entry.cost))
        .collect();
    let expected = brute_binary_costs(
        &bin_n,
        &bin_c0,
        max_k,
        DpCriterion::Misclass,
        leaf_reg,
        min_child,
    );
    assert_top_costs_match(&actual, &expected, beam);
}

#[test]
fn multiclass_shape_uses_pairwise_reference_margin_orders() {
    let thresholds = [0.5, 1.5, 2.5, 3.5];
    let bin_counts = vec![
        vec![5, 0, 1],
        vec![0, 4, 1],
        vec![2, 1, 3],
        vec![1, 5, 0],
        vec![0, 1, 6],
    ];
    let bin_n: Vec<i64> = bin_counts.iter().map(|row| row.iter().sum()).collect();
    let n_total = bin_n.iter().sum::<i64>() as usize;
    let teacher_mass = vec![
        5.4, 0.2, 0.4, 0.5, 4.9, 0.6, 2.0, 1.5, 2.5, 0.9, 5.0, 0.1, 0.3, 1.2, 5.5,
    ];
    let max_k = 4;
    let beam = 16;
    let beam_schedule = [beam, beam, beam];
    let leaf_reg = 0.003;
    let mu_node = 0.017;
    let min_child = 4;

    let actual_entries = shape_collapse_topb_pairwise_multiclass_min_child(
        &thresholds,
        &bin_n,
        &bin_counts,
        Some(&teacher_mass),
        max_k,
        n_total,
        &beam_schedule,
        DpCriterion::Gini,
        None,
        leaf_reg,
        mu_node,
        min_child as usize,
    );
    let mut actual: Vec<(usize, f64)> = actual_entries
        .iter()
        .map(|(k, entry)| (*k, entry.cost))
        .collect();
    actual.sort_by(|a, b| {
        a.0.cmp(&b.0)
            .then(a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
    });
    let expected = brute_pairwise_multiclass_shape_costs(
        &bin_n,
        &bin_counts,
        &teacher_mass,
        max_k,
        DpCriterion::Gini,
        leaf_reg,
        mu_node,
        min_child,
    );
    assert_top_costs_match(&actual, &expected, beam);
}
