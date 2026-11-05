"""Rashomon set of solve_rashomon vs a brute-force oracle over the SAME pool.

Ground truth (the RESPLINTER enumeration, "keep all subproblems with cost <= beta"):

    R_B(sup, d) = {leaf} ∪ ⋃_{c ∈ Select(sup)} ⨂_i R_B(child_i(c), d-1)

with exact Fraction costs (misclassified/n + reg per leaf; reg_alpha = 1 so
split_reg = 0). Select(sup) is taken from the solver itself through the
`rashomon_select_debug` binding (the post-dedup candidate list the Phase-2
search generates at that support, ub_hint = beta), so the oracle is valid
under any rho (transition_delta / atom merge) and beam B. Children are routed
with searchsorted(thresholds, x, side='right') (+ group_map). Mirrors of
explore_node: candidates with <= 1 non-empty child are skipped; n <= 1 and
depth 0 are leaves; an empty child is a leaf of cost reg.

Checks per instance (closure off, lookahead None = full-depth BFS):
  (a) tree_count at beta == brute count at beta,
  (b) exact-structure key set of ALL get_tree(i) == brute key set,
  (c) every extracted tree's recomputed objective <= beta.

Merged pool: the default candidate merge (transition_delta=0.02,
atom_mode=True), N random tiny datasets + the 5-row toy.
Unmerged pool: no merge (transition_delta=0, atom_mode=False), bypass_beam,
count == brute_part (partition-dedup brute force over the full threshold
grid). N = RASH_ORACLE_N (default 30).

Valid configurations: `rashomon_select_debug` hard-wires
feature_exclusion=false, closure off (closure_depth=None) and the default
reference-ensemble settings (shape_boundary_tax=0, teacher_alpha=1, top_k=None,
no teacher_logit / reference_preds), and the oracle's cost model needs split_reg = 0
(reg_alpha = 1) and path_penalty = 0. Solves outside that set (e.g. the
configuration with closure on) are not checked by this oracle.
"""
from math import comb

import numpy as np

from splinter.multiway_solver import solve_rashomon
from splinter.resplinter import dict_to_tree

from _helpers import (EPS, N_ORACLE_INST, TOY, SelectOracle, brute_schedule, frac_budget,
                      random_instances, tree_key, tree_objective)

KEY_CAP = 4000  # enumerate structure keys only for sets up to this size


# ------------------------------------------------------------ one instance
def run_instance(X, y, thresholds, depth, max_k, reg, theta, beam=4,
                 transition_delta=0.02, atom_mode=True, bypass_beam=False, brute_part=False):
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.uint8)
    res = solve_rashomon(X, y, 2, reg, depth, max_k, theta, lookahead_depth=None,
                         min_samples_leaf=1, beam_width=beam, reg_alpha=1.0,
                         transition_delta=transition_delta, atom_mode=atom_mode,
                         bypass_beam=bypass_beam,
                         thresholds_per_feature={f: list(t) for f, t in thresholds.items()},
                         fill_parallel=False)
    dag, tc, beta = res["dag"], int(res["tree_count"]), float(res["rashomon_bound"])
    reg_alpha = 1.0
    assert (1.0 - reg_alpha) * reg == 0.0  # split_reg = 0 in the oracle's cost model
    oracle = SelectOracle(X, y, thresholds, depth, max_k, reg, beta, beam_width=beam,
                          transition_delta=transition_delta, atom_mode=atom_mode,
                          bypass_beam=bypass_beam)
    root = frozenset(range(len(y)))
    budget = frac_budget(beta)
    brute = sum(c for cost, c in oracle.sched(root, depth).items() if cost <= budget)
    out = dict(tree_count=tc, brute=brute, beta=beta, opt=float(res["opt"]),
               complete=res.get("rashomon_complete"), over_bound=0,
               keys_equal=None, n_keys=None)
    if brute_part:
        bp = brute_schedule(X.tolist(), y.tolist(), thresholds, depth, max_k, reg,
                            by_partition=True)
        out["brute_part"] = sum(c for cost, c in bp.items() if float(cost) <= beta + EPS)
    if max(tc, brute) <= KEY_CAP:
        keys = set()
        for i in range(tc):
            t = dict_to_tree(dag.get_tree(i))
            keys.add(tree_key(t))
            if tree_objective(t, X, y.astype(int), reg) > beta + EPS:
                out["over_bound"] += 1
        bkeys = {k for _, k in oracle.trees(root, depth, budget)}
        out["keys_equal"] = keys == bkeys
        out["n_keys"] = len(keys)
        out["missing"] = len(bkeys - keys)
        out["extra"] = len(keys - bkeys)
    return out


def ok(r, key="brute"):
    return (r["tree_count"] == r[key] and r["over_bound"] == 0
            and r["keys_equal"] in (True, None))


def run_merged_pool(n_inst=N_ORACLE_INST, seed=0):
    rows = []
    for k, inst in enumerate(random_instances(n_inst, seed)):
        r = run_instance(**inst)
        r.update(inst=k)
        rows.append(r)
    return rows


def run_unmerged_pool(n_inst=N_ORACLE_INST, seed=1):
    """No candidate merge, full beam (bypass_beam), count == brute_part."""
    rows = []
    for k, inst in enumerate(random_instances(n_inst, seed)):
        inst = dict(inst, beam=128)
        # beam >= #cuts: every (j, k) has at most 128 partitions, so the
        # 128-partition cap of the Rush DP never truncates.
        assert all(comb(len(t), kk - 1) <= 128 for t in inst["thresholds"].values()
                   for kk in range(2, inst["max_k"] + 1))
        r = run_instance(**inst, transition_delta=0.0, atom_mode=False, bypass_beam=True,
                         brute_part=True)
        r.update(inst=k)
        rows.append(r)
    return rows


def summarize(rows, key="brute"):
    bad = [r for r in rows if not ok(r, key)]
    return dict(n=len(rows), passed=len(rows) - len(bad),
                key_checked=sum(r["keys_equal"] is not None for r in rows),
                failures=[{k: r.get(k) for k in ("inst", "tree_count", key, "over_bound",
                                                 "keys_equal", "missing", "extra")}
                          for r in bad[:10]])


def test_toy_select_oracle():
    r = run_instance(**TOY)
    assert ok(r), r


def test_merged_pool_random_select_oracle():
    s = summarize(run_merged_pool())
    assert s["passed"] == s["n"], s


def test_unmerged_pool_toy_is_six():
    r = run_instance(**TOY, transition_delta=0.0, atom_mode=False, bypass_beam=True,
                     beam=128, brute_part=True)
    assert r["brute_part"] == 6, r
    assert r["tree_count"] == 6 and ok(r), r


def test_unmerged_pool_partition_brute():
    s = summarize(run_unmerged_pool(), key="brute_part")
    assert s["passed"] == s["n"], s

