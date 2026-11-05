"""Estimator API on small synthetic and toy data: private fitted
logits, `sample_weight` normalization, the `threshold_budget` /
`fixed_thresholds` conflict, and the layout of user-supplied 1-D logits.
"""
import numpy as np
import pytest
from sklearn.datasets import load_breast_cancer

from splinter import SPLINTER, ShapeSPLINTER, LicketySPLINTER, RESPLINTER


def _data():
    X, y = load_breast_cancer(return_X_y=True)
    return X[::2], y[::2]


@pytest.mark.parametrize("kind", [SPLINTER, ShapeSPLINTER, LicketySPLINTER])
def test_fitted_logits_are_private_and_reset(kind):
    X, y = _data()
    model = kind(max_depth=2, lgbm_n_estimators=4).fit(X, y)
    assert not hasattr(model, "teacher_logit")
    assert not hasattr(model, "closure_min_n_for_multiway")
    assert model._logits.shape == (len(X), 2)
    with pytest.raises(ValueError):
        model.fit(X, np.zeros(len(X)))
    assert getattr(model, "_logits", None) is None
    assert not hasattr(model, "_closure_min_n")


def test_lickety_closure_min_n_is_private():
    X, y = _data()
    model = LicketySPLINTER(max_depth=2, lgbm_n_estimators=4, min_samples_leaf=5).fit(X, y)
    assert model._closure_min_n == 40


@pytest.mark.parametrize("kind", [SPLINTER, ShapeSPLINTER])
def test_unit_weights_equal_the_unweighted_fit(kind):
    X, y = _data()
    plain = kind(max_depth=3, lgbm_n_estimators=8).fit(X, y)
    for w in (np.ones(len(y)), np.full(len(y), 7.5)):
        weighted = kind(max_depth=3, lgbm_n_estimators=8).fit(X, y, sample_weight=w)
        assert _strip(weighted.tree_) == _strip(plain.tree_)
        np.testing.assert_array_equal(weighted.predict(X), plain.predict(X))
        np.testing.assert_allclose(weighted.predict_proba(X), plain.predict_proba(X))


def _strip(node):
    """Tree structure and leaf fields without class_counts (ints vs weight totals)."""
    if hasattr(node, "children"):
        return (node.feature_index, tuple(node.thresholds), tuple(getattr(node, "group_map", ())),
                tuple(_strip(c) for c in node.children))
    return (node.prediction, node.n_samples, node.loss)


@pytest.mark.parametrize("bad", [np.full(10, np.nan), -np.ones(10), np.zeros(10), np.ones(9)])
def test_sample_weight_validation(bad):
    X = np.arange(10, dtype=float).reshape(-1, 1)
    y = (X[:, 0] >= 5).astype(int)
    with pytest.raises(ValueError, match="sample_weight"):
        SPLINTER(max_depth=1, min_samples_leaf=1, lgbm_n_estimators=4).fit(
            X, y, sample_weight=bad)


def test_score_passes_sample_weight():
    X, y = _data()
    model = SPLINTER(max_depth=1, lgbm_n_estimators=4).fit(X, y)
    correct = model.predict(X) == y
    w = np.where(correct, 1.0, 3.0)
    assert model.score(X, y) == pytest.approx(correct.mean())
    assert model.score(X, y, sample_weight=w) == pytest.approx(np.average(correct, weights=w))


@pytest.mark.parametrize("kind", [LicketySPLINTER, RESPLINTER])
def test_estimators_without_weights_reject_the_keyword(kind):
    X, y = _data()
    with pytest.raises(TypeError):
        kind().fit(X, y, sample_weight=np.ones(len(y)))


def test_threshold_budget_with_fixed_thresholds_raises():
    X = np.repeat([[0.0], [1.0]], 5, axis=0)
    y = np.repeat([0, 1], 5)
    with pytest.raises(ValueError, match="threshold_budget cannot be combined with "
                                         "fixed_thresholds"):
        RESPLINTER(max_depth=1, max_k=2, fixed_thresholds={0: [0.5]},
                   threshold_budget=1).fit(X, y)


def test_one_dimensional_logits_use_the_internal_layout():
    X, y = _data()
    default = ShapeSPLINTER(max_depth=3, lgbm_n_estimators=16).fit(X, y)
    internal = default._logits
    np.testing.assert_array_equal(internal[:, 0], -internal[:, 1])
    margin = internal[:, 1] - internal[:, 0]  # 2 * raw
    given = ShapeSPLINTER(max_depth=3, lgbm_n_estimators=16,
                          precomputed_logits=margin).fit(X, y)
    np.testing.assert_array_equal(given._logits, internal)
    assert given.tree_ == default.tree_
    two_d = ShapeSPLINTER(max_depth=3, lgbm_n_estimators=16,
                          precomputed_logits=internal * 3.0).fit(X, y)
    np.testing.assert_array_equal(two_d._logits, internal * 3.0)
