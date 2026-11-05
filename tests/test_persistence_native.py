"""Native DAG persistence on tiny synthetic data:

  * a DAG loaded with `from_bytes(to_bytes(6))` plus `attach_training_data`
    answers tree (with leaf `n_samples`), RID, class-count and sampling
    queries exactly like the fitted DAG;
  * `attach_training_data` rejects mismatched inputs with ValueError;
  * the fit config (`exactify_top_k` and the extra keys) survives a round trip;
  * `PyRashomonDag` reports the package module name (for pickling).
"""
import numpy as np
import pytest

from splinter import RESPLINTER
from splinter.multiway_solver import PyRashomonDag

from _helpers import tiny_data


@pytest.fixture(scope="module")
def fitted():
    """A RESPLINTER fit whose set holds >= 50 trees, and its fit inputs."""
    X, y = tiny_data()
    model = RESPLINTER(max_depth=2, max_k=3, reg=0.02, theta=0.3, beam_width=4,
                       lgbm_n_estimators=20).fit(X, y)
    y_enc = np.searchsorted(model.classes_, y).astype(np.int64)
    return model, np.ascontiguousarray(X), y_enc


def _reload(model, X, y_enc):
    d2 = PyRashomonDag.from_bytes(bytes(model.dag_.to_bytes(6)))
    d2.attach_training_data(X, y_enc, model.lgbm_thresholds_, 2)
    return d2


def _features(tree):
    if tree["type"] == "leaf":
        return set()
    return {tree["feature_index"]}.union(*(_features(c) for c in tree["children"]))


def test_reload_without_training_data_differs(fitted):
    """Negative control: without `attach_training_data` the reloaded DAG has
    no fit rows, so leaf `n_samples` differ and RID cannot run."""
    model, _, _ = fitted
    dag = model.dag_
    d2 = PyRashomonDag.from_bytes(bytes(dag.to_bytes(6)))
    n = int(dag.tree_count)
    assert any(d2.get_tree(i) != dag.get_tree(i) for i in range(n))
    with pytest.raises(ValueError):
        d2.rid_histograms()


def test_reload_with_training_data_equals_fit(fitted):
    model, X, y_enc = fitted
    dag = model.dag_
    d2 = _reload(model, X, y_enc)
    n = int(dag.tree_count)
    assert n >= 50
    assert d2.n_train == dag.n_train == len(X)
    for i in range(n):
        assert d2.get_tree(i) == dag.get_tree(i), i
    assert d2.rashomon_bound_target == dag.rashomon_bound_target
    assert int(d2.tree_count_at_target) == int(dag.tree_count_at_target)
    for at_target in (True, False):
        assert ([int(i) for i in d2.sample_indices(40, seed=5, at_target=at_target)]
                == [int(i) for i in dag.sample_indices(40, seed=5, at_target=at_target)])
    want, got = dag.class1_counts(X), d2.class1_counts(X)
    for key in ("complete", "count_mismatch", "n_trees"):
        assert got[key] == want[key], key
    for key in ("counts", "counts0"):
        assert [int(c) for c in got[key]] == [int(c) for c in want[key]], key
    want, got = dag.rid_histograms(), d2.rid_histograms()
    for key in ("nums", "counts", "n_eval", "n_trees", "complete", "count_mismatch"):
        assert got[key] == want[key], key
    want = dag.rid_histograms(X, [int(v) for v in y_enc])
    got = d2.rid_histograms(X, [int(v) for v in y_enc])
    assert (got["nums"], got["counts"]) == (want["nums"], want["counts"])
    idx = list(range(int(dag.tree_count_at_target)))
    np.testing.assert_array_equal(d2.tree_phis(idx), dag.tree_phis(idx))
    for g in ("reg", "exact_cost_keys", "keyed_costs"):
        assert getattr(d2, g) == getattr(dag, g), g


def test_attach_validation(fitted):
    model, X, y_enc = fitted
    thr = model.lgbm_thresholds_
    d2 = PyRashomonDag.from_bytes(bytes(model.dag_.to_bytes(6)))
    bad_inputs = [
        (X[:-1], y_enc[:-1], 2),                 # row count != n_train
        (X[:, :-1], y_enc, 2),                   # feature count
        (X, np.where(y_enc == 1, 2, 0), 2),      # label >= n_classes
    ]
    for x, y, n_classes in bad_inputs:
        with pytest.raises(ValueError):
            d2.attach_training_data(np.ascontiguousarray(x), y, thr, n_classes)
        with pytest.raises(Exception):
            d2.attach_training_data(np.ascontiguousarray(x), y, thr, n_classes)


def test_attach_rejects_x_without_split_features(fitted):
    """With no feature metadata, x must still have a column for every
    feature a split uses."""
    model, X, y_enc = fitted
    dag = model.dag_
    max_feature = max(set().union(*(_features(dag.get_tree(i))
                                    for i in range(int(dag.tree_count)))))
    assert max_feature >= 1
    d2 = PyRashomonDag.from_bytes(bytes(dag.to_bytes(6)), [])
    narrow = np.ascontiguousarray(X[:, :max_feature])
    with pytest.raises(ValueError):
        d2.attach_training_data(narrow, y_enc, {}, 2)
    with pytest.raises(Exception):
        d2.attach_training_data(narrow, y_enc, {}, 2)


def test_fit_config_round_trip(fitted):
    model, _, _ = fitted
    assert (model.dag_.fit_config["extras"]["rashomon_bound_target"]
            == model.theta_bound_)
    dag = PyRashomonDag.from_bytes(bytes(model.dag_.to_bytes(6)))
    cfg = dict(model.dag_.fit_config)
    cfg["exactify_top_k"] = 3
    cfg.pop("top_k", None)
    cfg["dual_family"] = False
    cfg["both_modes"] = True
    cfg["search_strategy"] = "bfs"
    dag.set_fit_config(cfg)
    loaded = PyRashomonDag.from_bytes(bytes(dag.to_bytes(6))).fit_config
    assert loaded["top_k"] == 3
    extras = loaded["extras"]
    assert (extras["dual_family"], extras["both_modes"], extras["search_strategy"]) == (
        False, True, "bfs")
    assert extras["rashomon_bound_target"] == model.dag_.rashomon_bound_target
    # Dicts written with the old key still load.
    cfg.pop("exactify_top_k")
    cfg["top_k"] = 4
    dag.set_fit_config(cfg)
    assert PyRashomonDag.from_bytes(bytes(dag.to_bytes(6))).fit_config["top_k"] == 4


def test_module_name():
    assert PyRashomonDag.__module__ == "splinter.multiway_solver"
