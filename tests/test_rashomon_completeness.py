"""Phase-2 completeness flag (`rashomon_complete`, `rashomon_incomplete_reason`)
and the Step-A pass on configurations the select oracle (test_rashomon_oracle)
does not cover:

  * closure on (atom merge, transition_delta 0.02, beam 1): count ==
    distinct, objectives <= beta, reason "closure". Seeds 5 and 6 are
    regressions for the former class-count-only closure cache: depth-zero
    leaf costs must not prune deeper subproblems with the same class totals.
  * non-leaf objectives are rejected before enumeration because the
    factored representation prices leaf costs only.
  * prefix solve (closure off, lookahead < depth): recursive fill is Step B
    (complete; tests/test_rashomon_stepb.py checks the set); no fill gives
    "unfilled_prefix_leaves".
"""
import random

import numpy as np
import pytest

from splinter.multiway_solver import solve_rashomon
from splinter.resplinter import dict_to_tree
from splinter.tree import MWLeaf

from _helpers import tree_key

EPS = 1e-9
# Rounded keys (1e-9 grid, SCOPE_EPS = 1e-9): a tree is admitted when its
# rounded key sum fits the rounded scope key, so its exact objective can
# exceed beta by the per-leaf rounding (<= 0.5e-9 each) plus SCOPE_EPS.
ROUNDED_TOL = 1e-8


def objective(t, X, y, reg, reg_alpha=1.0, path_penalty=0.0, idx=None):
    """err/n + reg_alpha*reg per leaf + (1-reg_alpha)*reg + path_penalty*n_v/n
    per internal node (the solver's objective)."""
    idx = np.arange(len(y)) if idx is None else idx
    if isinstance(t, MWLeaf):
        return int((y[idx] != t.prediction).sum()) / len(y) + reg_alpha * reg
    b = np.searchsorted(np.asarray(t.thresholds), X[idx, t.feature_index], side="right")
    own = (1.0 - reg_alpha) * reg + path_penalty * len(idx) / len(y)
    return own + sum(objective(c, X, y, reg, reg_alpha, path_penalty, idx[b == j])
                     for j, c in enumerate(t.children))


def extract(res, X, y, reg, tol, **obj_kw):
    """(keys, over_bound) of every get_tree(i)."""
    dag, tc, beta = res["dag"], int(res["tree_count"]), float(res["rashomon_bound"])
    keys, over = [], 0
    for i in range(tc):
        t = dict_to_tree(dag.get_tree(i))
        keys.append(tree_key(t))
        over += objective(t, X, y, reg, **obj_kw) > beta + tol
    return keys, over


def tiny(seed, n=(10, 30), p=(2, 3), vmax=(3, 6)):
    rng = random.Random(seed)
    while True:
        nn, pp, vm = rng.randint(*n), rng.randint(*p), rng.randint(*vmax)
        X = [[rng.randint(0, vm) for _ in range(pp)] for _ in range(nn)]
        y = [rng.randint(0, 1) for _ in range(nn)]
        if len(set(y)) == 2:
            ths = {f: [v + 0.5 for v in range(vm)] for f in range(pp)}
            return np.asarray(X, dtype=np.float64), np.asarray(y, dtype=np.uint8), ths, rng


def reasons(res):
    r = res["rashomon_incomplete_reason"]
    return set(r.split(";")) if r else set()


# ------------------------------------------------------------ closure on
def closure_solve(seed):
    """Closure-on config (atom_mode, transition_delta 0.02, B = 1, closure beam
    3 below lookahead 2 with the wrapper's "greedy" criterion, depth 4) on
    a tiny dataset."""
    X, y, ths, rng = tiny(seed)
    k = rng.choice([2, 3])
    res = solve_rashomon(X, y, 2, 0.01, 4, k, 0.05, lookahead_depth=2, beam_width=1,
                         transition_delta=0.02, atom_mode=True,
                         thresholds_per_feature=ths, closure_depth=2,
                         closure_beam=3, closure_criterion="greedy",
                         recursive_fill=True, fill_parallel=False)
    return X, y, res


CLOSURE_SEEDS = list(range(12))


def test_closure_on_count_objectives_reason():
    for seed in CLOSURE_SEEDS:
        X, y, res = closure_solve(seed)
        keys, over = extract(res, X, y.astype(int), 0.01, EPS)
        tc = int(res["tree_count"])
        assert tc == len(set(keys)) == len(keys), (seed, tc, len(set(keys)))
        assert over == 0, (seed, over)
        assert res["rashomon_complete"] is False
        assert "closure" in reasons(res), res["rashomon_incomplete_reason"]


# ------------------------------------------------------------ supported objective
@pytest.mark.parametrize("reg_alpha,path_penalty", [(0.5, 0.0), (1.0, 0.01)])
def test_nonleaf_objective_is_rejected_instead_of_returning_invalid_trees(reg_alpha, path_penalty):
    for seed in range(15):
        X, y, ths, rng = tiny(100 + seed, n=(6, 20))
        with pytest.raises(ValueError, match="leaf-only objective"):
            solve_rashomon(
                X, y, 2, 0.05, rng.choice([2, 3]), rng.choice([2, 3]), 0.1,
                beam_width=4, reg_alpha=reg_alpha, path_penalty=path_penalty,
                thresholds_per_feature=ths, fill_parallel=False,
            )


# ------------------------------------------------------------ prefix solve
@pytest.mark.parametrize("recursive_fill", [True, False])
def test_prefix_solve_completeness(recursive_fill):
    X, y, ths, _ = tiny(7, n=(20, 30))
    res = solve_rashomon(X, y, 2, 0.02, 3, 2, 0.05, lookahead_depth=1, beam_width=4,
                         thresholds_per_feature=ths, recursive_fill=recursive_fill,
                         fill_parallel=False)
    assert res["solve_depth"] == 1 and res["remaining_depth"] == 2
    if recursive_fill:
        assert res["stepb_mode"] == "paper"
        assert res["rashomon_complete"] is True, res["rashomon_incomplete_reason"]
        return
    assert res["rashomon_complete"] is False
    assert reasons(res) == {"unfilled_prefix_leaves"}, res["rashomon_incomplete_reason"]


def test_full_depth_exact_keys_complete():
    X, y, ths, _ = tiny(3)
    res = solve_rashomon(X, y, 2, 0.05, 2, 2, 0.1, lookahead_depth=None, beam_width=4,
                         thresholds_per_feature=ths, fill_parallel=False)
    assert reasons(res) == set(), res["rashomon_incomplete_reason"]
    assert res["rashomon_complete"] is True, res["rashomon_incomplete_reason"]

