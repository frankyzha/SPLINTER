"""Seeded sampling of the factored DAG.

  * seeded `sample_indices` is deterministic and stays in
    [0, tree_count_at_target); every index is hit (uniformity) with
    20 x count draws;
  * `_RESPLINTER.sample_trees(..., return_indices=True)` returns exactly the
    trees `get_tree(i)` gives for its indices.
"""
import numpy as np

from splinter import RESPLINTER
from splinter.multiway_solver import solve_rashomon

from _helpers import TOY, tiny_data, tree_key


def toy_dag():
    X = np.asarray(TOY["X"], dtype=np.float64)
    y = np.asarray(TOY["y"], dtype=np.uint8)
    res = solve_rashomon(X, y, 2, 0.05, 2, 3, 0.5, lookahead_depth=None,
                         min_samples_leaf=1, beam_width=256,
                         thresholds_per_feature={f: list(t) for f, t in TOY["thresholds"].items()},
                         fill_parallel=False)
    return res["dag"]


def fit_tiny(**kw):
    X, y = tiny_data(n=60)
    params = dict(max_depth=2, max_k=3, reg=0.02, theta=0.3, beam_width=4,
                  lgbm_n_estimators=20)
    params.update(kw)
    return RESPLINTER(**params).fit(X, y), X, y


# ------------------------------------------------------------ sampling
def test_sample_indices_seeded_deterministic_and_in_range():
    dag = toy_dag()
    tc_t = int(dag.tree_count_at_target)
    assert tc_t > 0
    a = [int(i) for i in dag.sample_indices(200, seed=123, at_target=True)]
    b = [int(i) for i in dag.sample_indices(200, seed=123, at_target=True)]
    c = [int(i) for i in dag.sample_indices(200, seed=124, at_target=True)]
    assert a == b
    assert a != c
    assert len(a) == 200
    assert all(0 <= i < tc_t for i in a)


def test_sample_indices_hit_every_index():
    dag = toy_dag()
    for at_target in (False, True):
        count = int(dag.tree_count_at_target) if at_target else int(dag.tree_count)
        draws = [int(i) for i in dag.sample_indices(20 * count, seed=7, at_target=at_target)]
        assert set(draws) == set(range(count)), (at_target, count, sorted(set(draws)))


def test_sample_trees_return_indices_matches_get_tree():
    model, _, _ = fit_tiny()
    assert model.tree_count_ > 1
    trees, idxs = model.sample_trees(50, seed=11, at_target=True, return_indices=True)
    assert len(trees) == len(idxs) == 50
    assert all(0 <= i < model.tree_count_at_target_ for i in idxs)
    for t, i in zip(trees, idxs):
        assert tree_key(t) == tree_key(model.get_tree(i))
    # Same seed -> same indices as the raw sampler and as a repeat call.
    assert idxs == [int(i) for i in model.dag_.sample_indices(50, seed=11, at_target=True)]
    _, idxs2 = model.sample_trees(50, seed=11, at_target=True, return_indices=True)
    assert idxs == idxs2
    # Bulk path (no indices) draws the same trees.
    bulk = model.sample_trees(50, seed=11, at_target=True)
    assert [tree_key(t) for t in bulk] == [tree_key(t) for t in trees]

