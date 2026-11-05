"""Package behaviour on tiny synthetic data:

  * the `threshold_budget` token parser and `_apply_budget`;
  * a DAG loaded with `from_bytes(to_bytes(6))` answers count, sampling,
    class-count and tree queries like the fitted DAG;
  * the committed v4 / v5 fixture files load from Python;
  * v6 is smaller than v5 on a DAG of at least 50 trees;
  * `predict_probs_exact` equals the mean over the enumerated set;
  * LicketySPLINTER runs with `closure_resplit=False`.
"""
import os

import numpy as np
import pytest

from splinter import RESPLINTER, LicketySPLINTER
from splinter.multiway_solver import PyRashomonDag
from splinter.thresholds import _apply_budget, _parse_budget
from splinter.tree import predict_encoded

from _helpers import tiny_data

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


@pytest.fixture(scope="module")
def fitted():
    """A RESPLINTER fit whose set holds 108 trees (theta pinned for >= 50)."""
    X, y = tiny_data()
    model = RESPLINTER(max_depth=2, max_k=3, reg=0.02, theta=0.3, beam_width=4,
                       lgbm_n_estimators=20).fit(X, y)
    return model, X, y


def _strip_n(t):
    if t["type"] == "leaf":
        return {k: v for k, v in t.items() if k != "n_samples"}
    return {**t, "children": [_strip_n(c) for c in t["children"]]}


# ------------------------------------------------------------ threshold budget
RANKED = [(0, 1.5, 9.0), (1, 0.5, 5.0), (0, 2.5, 3.0), (1, 1.5, 1.0)]


def test_threshold_budget_tokens():
    for keep_all in (None, "full", "all", "none", " Full "):
        assert _parse_budget(keep_all)(RANKED) == len(RANKED)
    assert _parse_budget(2)(RANKED) == 2
    assert _parse_budget(10)(RANKED) == len(RANKED)
    assert _parse_budget("3")(RANKED) == 3
    # Gain-mass prefix: 9 / 18 = 0.5 reaches 0.5 with the first edge.
    assert _parse_budget(0.5)(RANKED) == 1
    assert _parse_budget("0.5")(RANKED) == 1
    assert _parse_budget(0.99)(RANKED) == len(RANKED)
    for bad in (True, 0, -1, "0", 0.0, 1.0, 1.5, float("nan"), "1.5", "abc", "0.5x"):
        with pytest.raises(ValueError):
            _parse_budget(bad)


def test_apply_threshold_budget_keeps_top_and_empty_features():
    thresholds = {0: [2.5, 1.5], 1: [0.5, 1.5], 2: [0.25]}
    ranked = RANKED + [(2, 0.25, 0.5)]
    out = _apply_budget(thresholds, ranked, 2)
    assert out == {0: [1.5], 1: [0.5], 2: []}
    assert sum(len(v) for v in out.values()) <= 2
    out = _apply_budget(thresholds, ranked, 1)
    assert out == {0: [1.5], 1: [], 2: []}
    assert _apply_budget(thresholds, ranked, 99) == {0: [1.5, 2.5], 1: [0.5, 1.5],
                                                               2: [0.25]}


# ------------------------------------------------------------ loaded DAG
def test_loaded_dag_matches_fitted(fitted):
    model, X, _ = fitted
    dag = model.dag_
    loaded = type(dag).from_bytes(bytes(dag.to_bytes(6)))
    assert int(loaded.tree_count) == int(dag.tree_count)
    assert int(loaded.tree_count_at_target) == int(dag.tree_count_at_target)
    for at_target in (True, False):
        assert ([int(i) for i in loaded.sample_indices(40, seed=5, at_target=at_target)]
                == [int(i) for i in dag.sample_indices(40, seed=5, at_target=at_target)])
    Xc = np.ascontiguousarray(X)
    want, got = dag.class1_counts(Xc), loaded.class1_counts(Xc)
    for key in ("complete", "count_mismatch", "n_trees"):
        assert got[key] == want[key], key
    for key in ("counts", "counts0"):
        assert [int(c) for c in got[key]] == [int(c) for c in want[key]], key
    for i in range(int(dag.tree_count)):
        assert _strip_n(loaded.get_tree(i)) == _strip_n(dag.get_tree(i)), i


@pytest.mark.parametrize("name", ["dag_v4_fixture.bin", "dag_v5_fixture.bin"])
def test_committed_fixtures_load(name):
    dag = PyRashomonDag.load(os.path.join(DATA, name), None)
    assert int(dag.tree_count) > 0


def test_v6_smaller_than_v5(fitted):
    dag = fitted[0].dag_
    assert int(dag.tree_count) >= 50
    assert len(bytes(dag.to_bytes(6))) < len(bytes(dag.to_bytes(5)))


# ------------------------------------------------------------ exact probabilities
def test_predict_probs_exact_equals_enumeration(fitted):
    model, X, _ = fitted
    n = model.tree_count_at_target_
    assert n > 1
    votes = np.array([predict_encoded(model.get_tree(i), X) for i in range(n)])
    np.testing.assert_array_equal(model.predict_probs_exact(X), votes.mean(axis=0))


# ------------------------------------------------------------ LicketySPLINTER
@pytest.mark.parametrize("options", [dict(closure_resplit=False), dict()])
def test_lickety_closure_resplit(options):
    X, y = tiny_data(n=200)
    model = LicketySPLINTER(max_depth=3, max_k=3, reg=0.01, min_samples_leaf=5,
                            lgbm_n_estimators=20, lgbm_min_samples_leaf=5,
                            **options).fit(X, y)
    assert set(np.unique(model.predict(X))) <= set(model.classes_)
