"""Exact predictive multiplicity: `dag_.class1_counts`.

`dag_.class1_counts` computes per-row class-1 counts over the Rashomon set
with a cost-keyed DAG walk. On tiny synthetic instances (0/1/2-valued
grids, evaluated on the train rows plus every grid cell), the counts
(at_target True and False) equal an independent brute force over get_tree.
"""
import numpy as np
import pytest

from splinter.tree import MWLeaf, MWShapeNode
from splinter.multiway_solver import solve_rashomon
from splinter.resplinter import dict_to_tree

TH2 = {0: [0.5, 1.5], 1: [0.5, 1.5]}
TH3 = {0: [0.5, 1.5], 1: [0.5, 1.5], 2: [0.5, 1.5]}
# TOY and three small synthetic instances; s0_1 and s0_6 have a cost-blind
# predict_probs error of 0.156 / 0.136.
INSTANCES = {
    "toy": dict(X=[[2, 2], [0, 1], [0, 0], [1, 1], [1, 2]], y=[1, 1, 0, 0, 1],
                thresholds=TH2, depth=2, max_k=3, reg=0.05, theta=0.5),
    "s0_1": dict(X=[[1, 0, 1], [2, 1, 1], [1, 1, 1], [2, 0, 2], [0, 1, 0], [0, 2, 1],
                    [2, 2, 2], [0, 1, 0], [2, 0, 2], [1, 1, 2], [0, 1, 1], [1, 2, 2]],
                 y=[0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0],
                 thresholds=TH3, depth=3, max_k=2, reg=0.02, theta=0.5),
    "s0_6": dict(X=[[0, 2], [0, 0], [1, 2], [1, 0], [2, 1], [2, 0], [0, 1], [2, 1],
                    [2, 2], [1, 2], [1, 1]],
                 y=[1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1],
                 thresholds=TH2, depth=3, max_k=3, reg=0.05, theta=0.5),
    "s0_8": dict(X=[[1, 0, 0], [0, 2, 0], [2, 2, 2], [0, 0, 0], [2, 2, 0], [1, 1, 2],
                    [0, 0, 1], [1, 2, 0], [0, 1, 1]],
                 y=[0, 1, 0, 1, 0, 0, 1, 0, 0],
                 thresholds=TH3, depth=2, max_k=2, reg=0.05, theta=1.0),
}


class _Model:
    """The two attributes the helpers use: `dag_` and `predict_probs`."""

    def __init__(self, dag):
        self.dag_ = dag

    def predict_probs(self, X):
        return np.asarray(self.dag_.predict_probs(np.ascontiguousarray(X, dtype=np.float64)))


def _fit(inst):
    X = np.asarray(inst["X"], dtype=np.float64)
    y = np.asarray(inst["y"], dtype=np.uint8)
    res = solve_rashomon(X, y, 2, inst["reg"], inst["depth"], inst["max_k"], inst["theta"],
                         lookahead_depth=None, min_samples_leaf=1, beam_width=256,
                         thresholds_per_feature={f: list(t) for f, t in inst["thresholds"].items()},
                         fill_parallel=False)
    p = X.shape[1]
    grid = np.array(np.meshgrid(*[[0, 1, 2]] * p)).reshape(p, -1).T.astype(np.float64)
    return _Model(res["dag"]), np.vstack([X, grid])


def _route(t, X):
    """Independent router (the verifier's), not the helper's."""
    out = np.zeros(len(X), dtype=np.int64)
    stack = [(t, np.arange(len(X)))]
    while stack:
        node, rows = stack.pop()
        if isinstance(node, MWLeaf):
            out[rows] = int(node.prediction)
            continue
        b = np.searchsorted(np.asarray(node.thresholds), X[rows, node.feature_index], side="right")
        if isinstance(node, MWShapeNode):
            b = np.asarray(node.group_map)[b]
        for j, c in enumerate(node.children):
            r = rows[b == j]
            if len(r):
                stack.append((c, r))
    return out


def _brute_counts(model, X, n=None):
    """Class-1 counts over get_tree(0 .. n-1) (n = tree_count_at_target)."""
    n = int(model.dag_.tree_count_at_target) if n is None else n
    S = np.zeros(len(X), dtype=np.int64)
    for i in range(n):
        S += _route(dict_to_tree(model.dag_.get_tree(i)), X)
    return S, n


@pytest.mark.parametrize("name", sorted(INSTANCES))
def test_class1_counts_match_bruteforce(name):
    model, Xe = _fit(INSTANCES[name])
    info = model.dag_.class1_counts(np.ascontiguousarray(Xe))
    S, tct = _brute_counts(model, Xe)
    assert info["complete"] and info["count_mismatch"] == 0
    assert info["n_trees"] == tct > 0
    assert [int(c) for c in info["counts"]] == [int(c) for c in S]
    assert [int(a) + int(b) for a, b in zip(info["counts"], info["counts0"])] == [tct] * len(Xe)
    # at_target=False: over all tree_count trees (the index prefix too).
    tc = int(model.dag_.tree_count)
    info_all = model.dag_.class1_counts(np.ascontiguousarray(Xe), at_target=False)
    S_all, _ = _brute_counts(model, Xe, n=tc)
    assert info_all["complete"] and info_all["count_mismatch"] == 0
    assert info_all["n_trees"] == tc
    assert [int(c) for c in info_all["counts"]] == [int(c) for c in S_all]
