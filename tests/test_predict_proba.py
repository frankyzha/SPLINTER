"""`predict_proba` on small synthetic and toy data: its argmax equals
`predict` on every row (including tied and empty leaves), rows sum to 1,
single-tree probabilities equal the routed training-class frequencies, and
`cross_val_score(..., scoring="roc_auc")` runs without NaN.
"""
import numpy as np
import pytest
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.model_selection import cross_val_score

from splinter import SPLINTER, ShapeSPLINTER, LicketySPLINTER, RESPLINTER
from splinter.tree import (MWLeaf, MWNode, annotate_class_counts, predict_encoded,
                           predict_proba_encoded)

SINGLE = [SPLINTER, ShapeSPLINTER, LicketySPLINTER]


def _leaf_of_rows(tree, X):
    """Brute force: the leaf each row reaches, one row at a time."""
    out = []
    for row in X:
        node = tree
        while not isinstance(node, MWLeaf):
            branch = int(np.searchsorted(node.thresholds, row[node.feature_index], side="right"))
            if hasattr(node, "group_map"):
                branch = node.group_map[branch]
            node = node.children[branch]
        out.append(node)
    return out


@pytest.mark.parametrize("kind", SINGLE)
@pytest.mark.parametrize("dataset", ["iris", "breast_cancer"])
def test_single_tree_proba_equals_routed_frequencies(kind, dataset):
    X, y = (load_iris if dataset == "iris" else load_breast_cancer)(return_X_y=True)
    X, y = X[::3], y[::3]
    model = kind(max_depth=2, min_samples_leaf=3, lgbm_n_estimators=8).fit(X, y)
    proba = model.predict_proba(X)
    assert proba.shape == (len(X), len(model.classes_))
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)
    np.testing.assert_array_equal(model.classes_[proba.argmax(axis=1)], model.predict(X))
    leaves = _leaf_of_rows(model.tree_, X)
    encoded = np.searchsorted(model.classes_, y)
    for leaf in set(map(id, leaves)):
        rows = [i for i, l in enumerate(leaves) if id(l) == leaf]
        freq = np.bincount(encoded[rows], minlength=len(model.classes_)) / len(rows)
        np.testing.assert_allclose(proba[rows], np.tile(freq, (len(rows), 1)), atol=1e-11)


def test_tied_and_empty_leaves_follow_prediction():
    tree = MWNode(0, "x", [1.0, 2.0, 3.0], [MWLeaf(1, 4, 0.0), MWLeaf(0, 4, 0.0),
                                          MWLeaf(2, 0, 0.0), MWLeaf(1, 3, 0.0)])
    X_train = np.array([[0.0]] * 4 + [[1.5]] * 4 + [[3.5]] * 3)
    y_train = np.array([0, 0, 1, 1, 0, 0, 1, 1, 1, 2, 0])
    annotate_class_counts(tree, X_train, y_train, 3)
    assert [leaf.class_counts for leaf in tree.children] == [
        (2, 2, 0), (2, 2, 0), (0, 0, 0), (1, 1, 1)]
    X = np.array([[0.0], [1.5], [2.5], [3.5]])
    proba = predict_proba_encoded(tree, X, 3)
    np.testing.assert_array_equal(proba.argmax(axis=1), predict_encoded(tree, X))
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)
    np.testing.assert_array_equal(proba[2], [0.0, 0.0, 1.0])  # empty leaf: one-hot
    np.testing.assert_allclose(proba[0], [0.5, 0.5, 0.0], atol=1e-11)
    assert proba[0, 1] > proba[0, 0]
    assert proba[1, 0] == 0.5 and proba[1, 1] == 0.5  # tie already broken toward prediction


def test_annotation_keeps_prediction_and_loss():
    X, y = load_breast_cancer(return_X_y=True)
    model = SPLINTER(max_depth=2, lgbm_n_estimators=8).fit(X, y)
    leaves = _leaf_of_rows(model.tree_, X)
    for leaf in {id(l): l for l in leaves}.values():
        counts = np.asarray(leaf.class_counts)
        assert counts.sum() == leaf.n_samples
        assert counts[leaf.prediction] == counts.max()


def test_weighted_counts_are_the_solver_copies():
    X, y = load_breast_cancer(return_X_y=True)
    w = np.random.default_rng(0).uniform(0.2, 2.0, len(y))
    model = SPLINTER(max_depth=2, lgbm_n_estimators=8).fit(X, y, sample_weight=w)
    copies = np.floor(w / w.sum() * len(y) + 0.5)
    leaves = _leaf_of_rows(model.tree_, X)
    for leaf in {id(l): l for l in leaves}.values():
        rows = [i for i, l in enumerate(leaves) if l is leaf]
        want = np.bincount(y[rows], weights=copies[rows], minlength=2)
        assert all(isinstance(v, float) for v in leaf.class_counts)
        np.testing.assert_array_equal(leaf.class_counts, want)
    proba = model.predict_proba(X)
    np.testing.assert_array_equal(model.classes_[proba.argmax(axis=1)], model.predict(X))


def test_weights_below_half_the_mean_keep_argmax_equal_to_predict():
    """Class 0 has the larger weight total (22 vs 18), but each class-0 row is
    2 copies and each class-1 row 1 copy (40 vs 30), so the solver predicts 1;
    the counts are the copies, so predict_proba agrees."""
    X = np.zeros((40, 1))
    y = np.array([0] * 10 + [1] * 30)
    w = np.array([2.2] * 10 + [0.6] * 30)
    model = SPLINTER(max_depth=1, lgbm_n_estimators=4).fit(X, y, sample_weight=w)
    proba = model.predict_proba(X)
    np.testing.assert_array_equal(model.classes_[proba.argmax(axis=1)], model.predict(X))
    assert model.tree_.class_counts == (20.0, 30.0)


def test_resplinter_proba_and_exact_ties(monkeypatch):
    rng = np.random.default_rng(0)
    X = rng.integers(0, 4, size=(120, 3)).astype(np.float64)
    y = ((X[:, 0] >= 2) ^ (rng.random(120) < 0.15)).astype(int)
    model = RESPLINTER(max_depth=2, max_k=3, reg=0.02, theta=0.3, beam_width=4,
                       lgbm_n_estimators=20).fit(X, y)
    proba = model.predict_proba(X)
    np.testing.assert_allclose(proba[:, 1], model.predict_probs_exact(X))
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)
    np.testing.assert_array_equal(model.classes_[proba.argmax(axis=1)], model.predict(X))
    monkeypatch.setattr(model, "predict_probs_exact",
                        lambda X, **kw: np.array([0.5, 0.2, 0.8, 0.0, 1.0]))
    tied = model.predict_proba(X[:5])
    np.testing.assert_array_equal(tied[0], [0.5 - 1e-12, 0.5 + 1e-12])
    np.testing.assert_array_equal(model.classes_[tied.argmax(axis=1)], model.predict(X[:5]))
    np.testing.assert_allclose(tied.sum(axis=1), 1.0)


def test_cost_blind_probabilities_are_private():
    assert not hasattr(RESPLINTER, "predict_probs")


@pytest.mark.parametrize("model", [
    SPLINTER(max_depth=2, lgbm_n_estimators=8),
    ShapeSPLINTER(max_depth=2, lgbm_n_estimators=8),
    LicketySPLINTER(max_depth=2, lgbm_n_estimators=8),
    RESPLINTER(max_depth=2, lgbm_n_estimators=8),
])
def test_cross_val_roc_auc(model):
    X, y = load_breast_cancer(return_X_y=True)
    X, y = X[::4], y[::4]
    scores = cross_val_score(model, X, y, cv=3, scoring="roc_auc")
    assert np.isfinite(scores).all()
