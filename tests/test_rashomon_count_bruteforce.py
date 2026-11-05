"""Rashomon tree counts of the full-build factored DAG vs brute force.

Ground truth: every multiway threshold tree on a tiny dataset (exact
Fraction costs, misclassified/n + reg per leaf), enumerated over the SAME
explicit thresholds the solver is given (thresholds_per_feature). A split
on feature f is a choice of 1..max_k-1 of those thresholds whose bins are
all non-empty on the node's support; two threshold sets with the same
partition are different trees (granularity variants), matching the
exact-structure key used for `distinct`.

Checks, per instance:
  * tree_count == number of distinct trees returned by get_tree (no dupes),
  * every extracted tree's recomputed objective <= rashomon_bound,
  * tree_count <= brute-force count at rashomon_bound. Instances where the
    brute force finds MORE trees than the solver are coverage gaps of the
    enumeration search (not count bugs) and are allowed.
"""
import random

import numpy as np

from splinter.multiway_solver import solve_rashomon
from splinter.resplinter import dict_to_tree

from _helpers import (EPS, TOY, SelectOracle, brute_schedule, frac_budget, tree_key,
                      tree_objective)


def run_instance(X, y, thresholds, depth, max_k, reg, theta):
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.uint8)
    res = solve_rashomon(X, y, 2, reg, depth, max_k, theta, lookahead_depth=None,
                         min_samples_leaf=1, beam_width=256,
                         thresholds_per_feature={f: list(t) for f, t in thresholds.items()},
                         fill_parallel=False)
    dag, tc, bound = res["dag"], int(res["tree_count"]), float(res["rashomon_bound"])
    keys, over = set(), 0
    for i in range(tc):
        t = dict_to_tree(dag.get_tree(i))
        keys.add(tree_key(t))
        if tree_objective(t, X, y.astype(int), reg) > bound + EPS:
            over += 1
    bs = brute_schedule(X.tolist(), y.tolist(), thresholds, depth, max_k, reg)
    brute = sum(c for cost, c in bs.items() if float(cost) <= bound + EPS)
    bp = brute_schedule(X.tolist(), y.tolist(), thresholds, depth, max_k, reg, by_partition=True)
    brute_part = sum(c for cost, c in bp.items() if float(cost) <= bound + EPS)
    return dict(tree_count=tc, distinct=len(keys), over_bound=over, brute=brute,
                brute_part=brute_part,
                opt=float(res["opt"]), bound=bound, brute_opt=float(min(bs)))


def tiny_random_instances(n_inst=200, seed=0):
    rng = random.Random(seed)
    out = []
    while len(out) < n_inst:
        n, p = rng.randint(5, 8), rng.randint(2, 3)
        X = [[rng.randint(0, 2) for _ in range(p)] for _ in range(n)]
        y = [rng.randint(0, 1) for _ in range(n)]
        if len(set(y)) < 2:
            continue
        out.append(dict(X=X, y=y, thresholds={f: [0.5, 1.5] for f in range(p)},
                        depth=rng.randint(2, 3), max_k=rng.randint(2, 3),
                        reg=0.05, theta=rng.choice([0.1, 0.5, 1.0])))
    return out


def test_toy_no_duplicates_within_bound():
    r = run_instance(**TOY)
    assert r["brute"] == 6
    assert r["tree_count"] == r["distinct"], r
    assert r["over_bound"] == 0, r


def test_toy_count_equals_select_oracle():
    # 6 trees exist; the default Select (transition_delta = 0.02) keeps a
    # pool with 3 of them, and the count equals the brute force over that
    # pool. (With no transition merge, no atoms and a full beam the count is
    # the full 6: test_rashomon_oracle.test_unmerged_pool_toy_is_six.)
    r = run_instance(**TOY)
    oracle = SelectOracle(TOY["X"], TOY["y"], TOY["thresholds"], TOY["depth"], TOY["max_k"],
                          TOY["reg"], r["bound"], beam_width=256, transition_delta=0.02)
    root = frozenset(range(len(TOY["y"])))
    n_oracle = sum(c for cost, c in oracle.sched(root, TOY["depth"]).items()
                   if cost <= frac_budget(r["bound"]))
    assert r["tree_count"] == r["distinct"] == n_oracle == 3, (r, n_oracle)


def run_random(n_inst=200, seed=0):
    rows = []
    for k, inst in enumerate(tiny_random_instances(n_inst, seed)):
        r = run_instance(**inst)
        r.update(inst=k, depth=inst["depth"], max_k=inst["max_k"], theta=inst["theta"])
        rows.append(r)
    return rows


def summarize(rows):
    bad_dup = [r for r in rows if r["tree_count"] != r["distinct"]]
    bad_over = [r for r in rows if r["over_bound"]]
    over_count = [r for r in rows if r["tree_count"] > r["brute"]]
    gaps = [r for r in rows if r["tree_count"] < r["brute"]]
    return dict(n=len(rows), exact=sum(r["tree_count"] == r["brute"] for r in rows),
                count_ne_distinct=len(bad_dup), over_bound=len(bad_over),
                solver_over_brute=len(over_count), coverage_gaps=len(gaps),
                eq_brute_part=sum(r["tree_count"] == r["brute_part"] for r in rows),
                gap_examples=[(r["inst"], r["tree_count"], r["brute"], r["brute_part"], r["opt"],
                               r["brute_opt"]) for r in gaps[:10]],
                defect_examples=[r for r in rows if r["over_bound"]
                                 or r["tree_count"] != r["distinct"]][:5],
                overcount_examples=[(r["inst"], r["tree_count"], r["brute"]) for r in over_count[:10]])


def test_random_tiny_datasets():
    s = summarize(run_random(n_inst=30))
    assert s["count_ne_distinct"] == 0, s
    assert s["over_bound"] == 0, s
    assert s["solver_over_brute"] == 0, s

