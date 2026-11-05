"""Step B of the recursive fill vs a full-depth solve.

With CART closure off and lookahead d_l < d, solve_rashomon(recursive_fill=
True) runs Alg. resplinter: Step A (prefix Rashomon at depth d_l, the
depth-d_l placeholders priced at their lower bounds) and Step B (one
MiniRESPLINTER per placeholder at beta_u = beta - prefix_cost(u), grafted
into the DAG, recomputed at beta, quotient last). Its set must be exactly the
trees of depth <= d with cost <= beta over the retained pool — the set a
full-depth search (lookahead None) returns.

The two solves have different beta (phase 1 of the Step-B solve finds the
depth-d_l prefix optimum), so both are compared at beta_min = min of the two.
Per instance and configuration:
  (a) count of trees with cost <= beta_min equal (Step B vs full depth vs
      the brute-force Select-pool oracle of tests/_helpers.py),
  (b) identical exact-structure key sets at beta_min (vs both),
  (c) count == distinct keys (no duplicate tree in the Step-B DAG),
  (d) every extracted Step-B tree has recomputed objective <= its beta
      (indices up to beta_min + a sample of the rest).
Trees are indexed in ascending cost (root schedule order), so the trees
within beta_min are a prefix of the index range.

Configurations: 30 random tiny datasets (n <= 20) each at (d, d_l) in
{(3, 1), (3, 2), (2, 1)}; nested recursion (d - d_l > d_l) is tested
separately. A solve with no reachable placeholder reports stepb_mode 'none'
(nothing to fill).

Further checks:
  anchored band   Step B vs the oracle at the Step-B run's own beta_a (Select
                  ub = beta_a): the whole Step-B set. Step B is anchored at
                  the pool optimum, so beta_a <= beta_b and the band
                  (beta_b, beta_a] is empty.
  nested prefix   d=5, d_l=2 on 5 small instances (n <= 12): nested Step A
                  with a two-level prefix and a nested Step B below it, vs
                  full depth at min beta and vs the oracle.
  unequal scope   groups whose placeholder contexts give unequal beta_u
                  (n_stepb_groups_unequal_scope > 0) occur and pass.
  parallel fill   fill_parallel=True and False give the same trees in the
                  same order.
  save / load     Step-B DAG to_bytes / from_bytes: same trees, same order;
                  also a full-depth DAG (lookahead None).
"""
import random

import numpy as np
import pytest

from splinter.resplinter import dict_to_tree

from _helpers import (EPS, KEY_CAP, SelectOracle, frac_budget, small_instances, solve_instance,
                      tree_key, tree_objective)

N_SAMPLE = 200  # extra random indices checked for objective <= beta
N_INST = 30   # every random suite: <= 30 instances, n <= 20
N_SMALL = 10  # parallel-fill and save/load suites


def _all_keys(dag, tc, seed=0):
    """Ordered keys of every tree, or of the first 200 + 100 random
    indices when tc > KEY_CAP."""
    if tc <= KEY_CAP:
        idx = range(tc)
    else:
        idx = list(range(200)) + sorted(random.Random(seed).sample(range(200, tc), 100))
    return [tree_key(dict_to_tree(dag.get_tree(i))) for i in idx]


def _extract(res, X, y, reg, bound, seed):
    """Trees with cost <= bound (index prefix), plus a random sample of the
    remaining indices. Returns (keys within bound, n within bound, over_beta)."""
    dag, tc, beta = res["dag"], int(res["tree_count"]), float(res["rashomon_bound"])
    keys, n_in, over = [], 0, 0
    i = 0
    while i < tc:
        t = dict_to_tree(dag.get_tree(i))
        i += 1
        obj = tree_objective(t, X, y, reg)
        if obj > beta + EPS:
            over += 1
        if obj > bound + EPS:
            break
        keys.append(tree_key(t))
        n_in += 1
    rng = random.Random(seed)
    rest = range(i, tc)
    for j in (rng.sample(rest, min(N_SAMPLE, len(rest))) if len(rest) else []):
        t = dict_to_tree(dag.get_tree(j))
        obj = tree_objective(t, X, y, reg)
        if obj > beta + EPS:
            over += 1
        if obj <= bound + EPS:
            over += 1  # index order is ascending cost: must not be within bound
    return keys, n_in, over


def run_instance(inst, depth, lookahead, seed=0):
    X = np.asarray(inst["X"], dtype=np.float64)
    y = np.asarray(inst["y"], dtype=np.uint8)
    ra = solve_instance(inst, depth, lookahead)
    rb = solve_instance(inst, depth, None)
    beta_a, beta_b = float(ra["rashomon_bound"]), float(rb["rashomon_bound"])
    bound = min(beta_a, beta_b)
    ka, na, oa = _extract(ra, X, y.astype(int), inst["reg"], bound, seed)
    kb, nb, ob = _extract(rb, X, y.astype(int), inst["reg"], bound, seed + 1)
    oracle = SelectOracle(X, y, inst["thresholds"], depth, inst["max_k"], inst["reg"], beta_b,
                          beam_width=inst["beam"], transition_delta=0.02, atom_mode=True)
    root = frozenset(range(len(y)))
    okeys = [k for _, k in oracle.trees(root, depth, frac_budget(bound))]
    sa, sb, so = set(ka), set(kb), set(okeys)
    return dict(
        depth=depth, lookahead=lookahead, beta_a=beta_a, beta_b=beta_b,
        n_a=na, n_b=nb, n_oracle=len(okeys), distinct_a=len(sa),
        tc_a=int(ra["tree_count"]), tc_b=int(rb["tree_count"]),
        keys_eq_full=sa == sb, keys_eq_oracle=sa == so,
        missing=len((sb | so) - sa), extra=len(sa - (sb | so)),
        over_beta=oa + ob,
        stepb_mode=ra.get("stepb_mode"), complete=ra.get("rashomon_complete"),
        reason=ra.get("rashomon_incomplete_reason"),
        n_prefix_leaves=ra.get("n_prefix_leaves"), n_filled=ra.get("n_leaves_filled"),
        n_unequal=ra.get("n_stepb_groups_unequal_scope"),
    )


def stepb_label_ok(r):
    return r["stepb_mode"] == "paper" or (r["stepb_mode"] == "none"
                                          and r["n_prefix_leaves"] == 0)


def ok(r):
    return (r["n_a"] == r["n_b"] == r["n_oracle"] == r["distinct_a"]
            and r["keys_eq_full"] and r["keys_eq_oracle"]
            and r["over_beta"] == 0
            and stepb_label_ok(r) and r["complete"] is True)


def run_config(depth, lookahead, n_inst=30, seed=0, instances=None):
    key = (depth, lookahead, n_inst, seed) if instances is None else None
    if key in _ROWS:
        return _ROWS[key]
    rows = []
    for k, inst in enumerate(instances or small_instances(n_inst, seed=seed)):
        r = run_instance(inst, depth, lookahead, seed=k)
        r.update(inst=k)
        rows.append(r)
    if key is not None:
        _ROWS[key] = rows
    return rows


_ROWS = {}


# ------------------------------------------------------------ anchored band
def run_instance_b(inst, depth, lookahead, seed=0):
    """Step B vs the oracle at the Step-B run's own beta_a (Select ub =
    beta_a). n_band: oracle trees with cost in (beta_b, beta_a]."""
    X = np.asarray(inst["X"], dtype=np.float64)
    y = np.asarray(inst["y"], dtype=np.uint8)
    ra = solve_instance(inst, depth, lookahead)
    if int(ra["tree_count"]) > KEY_CAP:
        return None  # skipped: brute-force oracle too large
    beta_a = float(ra["rashomon_bound"])
    beta_b = float(solve_instance(inst, depth, None)["rashomon_bound"])
    ka, na, oa = _extract(ra, X, y.astype(int), inst["reg"], beta_a, seed)
    oracle = SelectOracle(X, y, inst["thresholds"], depth, inst["max_k"], inst["reg"], beta_a,
                          beam_width=inst["beam"], transition_delta=0.02, atom_mode=True)
    root = frozenset(range(len(y)))
    otrees = oracle.trees(root, depth, frac_budget(beta_a))
    so = set(k for _, k in otrees)
    n_band = sum(1 for c, _ in otrees if c > frac_budget(beta_b))
    return dict(depth=depth, lookahead=lookahead, beta_a=beta_a, beta_b=beta_b,
                n_a=na, tc_a=int(ra["tree_count"]), n_oracle=len(otrees),
                distinct_a=len(set(ka)), keys_eq_oracle=set(ka) == so,
                missing=len(so - set(ka)), extra=len(set(ka) - so), n_band=n_band,
                over_beta=oa, stepb_mode=ra.get("stepb_mode"),
                n_prefix_leaves=ra.get("n_prefix_leaves"),
                complete=ra.get("rashomon_complete"),
                reason=ra.get("rashomon_incomplete_reason"))


def ok_b(r):
    return (r["n_a"] == r["tc_a"] == r["n_oracle"] == r["distinct_a"]
            and r["keys_eq_oracle"] and r["over_beta"] == 0
            and stepb_label_ok(r) and r["complete"] is True)


def summarize_b(rows):
    bad = [r for r in rows if not ok_b(r)]
    return dict(n=len(rows), passed=len(rows) - len(bad),
                n_trees_checked=sum(r["n_a"] for r in rows),
                n_band_trees=sum(r["n_band"] for r in rows),
                n_inst_with_band=sum(r["n_band"] > 0 for r in rows),
                failures=[{k: r.get(k) for k in ("inst", "n_a", "tc_a", "n_oracle", "distinct_a",
                                                 "keys_eq_oracle", "missing", "extra",
                                                 "over_beta", "complete", "reason")}
                          for r in bad[:10]])


def run_config_b(depth, lookahead, n_inst=30, seed=0):
    rows = []
    for k, inst in enumerate(small_instances(n_inst, seed=seed)):
        r = run_instance_b(inst, depth, lookahead, seed=k)
        if r is None:
            continue
        r.update(inst=k)
        rows.append(r)
    return rows


# ------------------------------------------------------------ parallel fill / save-load
def run_parallel_check(depth, lookahead, n_inst=30, seed=0):
    """fill_parallel True vs False: identical ordered tree lists."""
    bad = []
    for k, inst in enumerate(small_instances(n_inst, seed=seed)):
        rs = solve_instance(inst, depth, lookahead, fill_parallel=False)
        rp = solve_instance(inst, depth, lookahead, fill_parallel=True)
        ts, tp = int(rs["tree_count"]), int(rp["tree_count"])
        if ts != tp or _all_keys(rs["dag"], ts) != _all_keys(rp["dag"], tp):
            bad.append((k, ts, tp))
    return dict(n=n_inst, passed=n_inst - len(bad), failures=bad[:10])


def run_save_load_check(depth, lookahead, n_inst=30, seed=0):
    """Step-B DAG to_bytes / from_bytes: same count, same ordered trees."""
    bad = []
    for k, inst in enumerate(small_instances(n_inst, seed=seed)):
        r = solve_instance(inst, depth, lookahead)
        dag, tc = r["dag"], int(r["tree_count"])
        loaded = type(dag).from_bytes(dag.to_bytes())
        if int(loaded.tree_count) != tc or _all_keys(loaded, tc) != _all_keys(dag, tc):
            bad.append((k, tc, int(loaded.tree_count)))
    return dict(n=n_inst, passed=n_inst - len(bad), failures=bad[:10])


def summarize(rows):
    bad = [r for r in rows if not ok(r)]
    return dict(n=len(rows), passed=len(rows) - len(bad),
                n_trees_checked=sum(r["n_a"] for r in rows),
                beta_a_gt_b=sum(r["beta_a"] > r["beta_b"] + EPS for r in rows),
                failures=[{k: r.get(k) for k in ("inst", "n_a", "n_b", "n_oracle", "distinct_a",
                                                 "keys_eq_full", "keys_eq_oracle", "missing",
                                                 "extra", "over_beta", "complete",
                                                 "reason")}
                          for r in bad[:10]])


def _check(depth, lookahead, n_inst=N_INST):
    s = summarize(run_config(depth, lookahead, n_inst))
    assert s["passed"] == s["n"], s


@pytest.mark.parametrize("depth,lookahead", [(3, 1), (3, 2), (2, 1)])
def test_stepb_vs_full_depth(depth, lookahead):
    _check(depth, lookahead)


@pytest.mark.parametrize("depth,lookahead", [(3, 1), (3, 2)])
def test_stepb_anchored_band_empty(depth, lookahead):
    """Step B anchored at the pool optimum: its beta is <= the full-depth
    solve's and the band (beta_b, beta_a] is empty."""
    rows = run_config_b(depth, lookahead, N_INST)
    assert len(rows) >= 10, len(rows)
    s = summarize_b(rows)
    assert s["passed"] == s["n"], s
    assert s["n_band_trees"] == 0, s
    assert all(r["beta_a"] <= r["beta_b"] + EPS for r in rows), \
        [(r["inst"], r["beta_a"], r["beta_b"]) for r in rows if r["beta_a"] > r["beta_b"] + EPS]


def test_stepb_nested_prefix_d5_la2():
    # d - d_l = 3 > d_l: MiniRESPLINTER runs its own two-level Step A and a
    # nested Step B below it.
    rows = run_config(5, 2, instances=small_instances(5, n_max=12))
    s = summarize(rows)
    assert s["passed"] == s["n"], s
    assert sum((r["n_filled"] or 0) > 0 for r in rows) > 0, s


def test_stepb_unequal_scope_groups():
    rows = run_config(3, 1, N_INST)
    hit = [r for r in rows if (r["n_unequal"] or 0) > 0]
    assert hit, "no instance with unequal beta_u in a group"
    assert all(ok(r) for r in hit), [r for r in hit if not ok(r)][:5]


def test_stepb_fill_parallel_identical():
    s = run_parallel_check(3, 1, n_inst=N_SMALL)
    assert s["passed"] == s["n"], s


@pytest.mark.parametrize("lookahead", [1, None])
def test_stepb_save_load_roundtrip(lookahead):
    # lookahead None: a full-depth DAG (no placeholders) also reloads.
    s = run_save_load_check(3, lookahead, n_inst=N_SMALL)
    assert s["passed"] == s["n"], s

