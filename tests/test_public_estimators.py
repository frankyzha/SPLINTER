"""Regression tests for the public estimators."""

import pickle

import numpy as np
import pytest
from sklearn.base import clone, is_classifier
from sklearn.exceptions import NotFittedError

from splinter import SPLINTER, ShapeSPLINTER, LicketySPLINTER, RESPLINTER
from splinter.thresholds import (
    ThresholdCache, _coerce_cache, load_threshold_cache,
    save_threshold_cache, _reference_encode_target,
)
from splinter.tree import MWLeaf, MWNode, MWShapeNode, predict_encoded
from splinter.resplinter import _RESPLINTER
from splinter.multiway_solver import CANDIDATE_RANKING_POLICY, PyRashomonDag


def cache(X, encoded):
    classes = int(np.max(encoded)) + 1
    thresholds = {j: ((np.unique(X[:, j])[:-1] + np.unique(X[:, j])[1:]) / 2).tolist()
                  for j in range(X.shape[1])}
    width = max((len(edges) for edges in thresholds.values()), default=0)
    return ThresholdCache(
        thresholds=thresholds, logits=np.eye(classes)[encoded] * 4.0,
        boundary_gain=np.ones((X.shape[1], width)), boundary_cover=None,
        boundary_value_jump=None, bin_representatives=None,
        fill_values=np.zeros(X.shape[1]),
        n_bins_per_feature=np.array([len(v) + 1 for v in thresholds.values()]),
    )


@pytest.mark.parametrize("estimator", [SPLINTER(), ShapeSPLINTER(), LicketySPLINTER(), RESPLINTER()])
def test_estimators_clone_as_classifiers(estimator):
    copied = clone(estimator)
    assert is_classifier(copied)
    assert copied.get_params().keys() == estimator.get_params().keys()


@pytest.mark.parametrize("version", [4, 5, 6])
def test_resplinter_saved_config_records_candidate_pool_settings(tmp_path, version):
    X = np.repeat([[0.0], [1.0]], 6, axis=0)
    y = np.repeat([0, 1], 6)
    model = _RESPLINTER(  # binary_threshold_backfill / fill_parallel: hidden settings
        max_depth=2, lookahead_depth=1, max_k=2, reg=0.01, cart_closure=False,
        lgbm_n_estimators=4, lgbm_num_leaves=31, lgbm_random_state=None,
        min_samples_leaf=1, binary_threshold_backfill=False,
        threshold_budget=np.float64(0.99), fill_parallel=False,
    ).fit(X, y)
    config = model.dag_.fit_config
    assert config["dp_criterion"] == "rush"
    extras = config["extras"]
    assert model.fit_info_["candidate_ranking_policy"] == CANDIDATE_RANKING_POLICY
    assert model.fit_info_["raw_result"]["candidate_ranking_policy"] == CANDIDATE_RANKING_POLICY
    assert extras["candidate_ranking_policy"] == CANDIDATE_RANKING_POLICY
    assert extras["lgbm_num_leaves"] == 31
    assert extras["lgbm_random_state"] is None
    assert extras["binary_threshold_backfill"] is False
    assert extras["threshold_budget"] == 0.99
    assert extras["threshold_source"] == "lightgbm"
    assert extras["fill_mode"] == "recursive"
    assert extras["lgbm_min_samples_leaf"] == 1
    path = tmp_path / f"rashomon_v{version}.bin"
    model.dag_.save(str(path), version)
    loaded = PyRashomonDag.load(str(path))
    assert loaded.fit_config == config
    assert int(loaded.tree_count) == model.tree_count_


@pytest.mark.parametrize("kind", [SPLINTER, ShapeSPLINTER, LicketySPLINTER])
@pytest.mark.parametrize("labels", [np.array(["low", "high"]), np.array([2.0, 9.0])])
def test_original_labels_and_prediction_feature_validation(kind, labels):
    X = np.repeat(np.arange(4, dtype=float), 6).reshape(-1, 1)
    encoded = (X[:, 0] >= 2).astype(np.uint8)
    y = labels[encoded]
    model = kind(max_depth=2, max_k=2, reg=0.001, min_samples_leaf=2,
                 precomputed_thresholds=cache(X, encoded)).fit(X, y)
    assert model.fit_info_["candidate_ranking_policy"] == CANDIDATE_RANKING_POLICY
    np.testing.assert_array_equal(model.predict(X), y)
    assert model.score(X, y) == 1.0
    with pytest.raises(ValueError, match="features"):
        model.predict(np.zeros((2, 2)))


@pytest.mark.parametrize("kind", [SPLINTER, ShapeSPLINTER])
def test_zero_lookahead_uses_native_constraints_and_real_objective(kind):
    X = np.repeat(np.arange(6, dtype=float), 4).reshape(-1, 1)
    y = np.repeat([0, 1, 2, 0, 1, 2], 4).astype(np.uint8)
    model = kind(max_depth=2, lookahead_depth=0, max_k=3, reg=0.001,
                 min_samples_leaf=3, path_penalty=0.02,
                 precomputed_thresholds=cache(X, y)).fit(X, y)
    assert model.n_subproblems_ > 0
    assert not model.certified_optimal_
    pending = [model.tree_]
    while pending:
        node = pending.pop()
        if isinstance(node, MWLeaf):
            assert node.n_samples >= 3
        else:
            pending.extend(node.children)
    expected = np.mean(model.predict(X) != y) + model.reg * model.num_leaves()
    expected += model.path_penalty * model.avg_path_length()
    assert model.objective() == pytest.approx(expected)


def test_repeated_lickety_fit_replaces_reference_ensemble_and_failed_fit_discards_tree():
    X = np.repeat(np.arange(4, dtype=float), 6).reshape(-1, 1)
    y = (X[:, 0] >= 2).astype(np.uint8)
    first = cache(X, y)
    model = LicketySPLINTER(max_depth=1, min_samples_leaf=1, precomputed_thresholds=first)
    model.fit(X, y)
    second = cache(X, 1 - y)
    model.set_params(precomputed_thresholds=second)
    model.fit(X, 1 - y)
    np.testing.assert_array_equal(model._logits, second.logits)
    np.testing.assert_array_equal(model.predict(X), 1 - y)
    with pytest.raises(ValueError):
        model.fit(X, np.zeros(len(X)))
    with pytest.raises(NotFittedError):
        model.predict(X)


def test_fitted_lickety_model_pickles_with_same_predictions():
    X = np.repeat(np.arange(4, dtype=float), 6).reshape(-1, 1)
    y = (X[:, 0] >= 2).astype(np.uint8)
    model = LicketySPLINTER(max_depth=2, min_samples_leaf=1,
                            precomputed_thresholds=cache(X, y)).fit(X, y)
    restored = pickle.loads(pickle.dumps(model))
    np.testing.assert_array_equal(restored.predict(X), model.predict(X))


@pytest.mark.parametrize("kwargs", [{"reg": np.nan}, {"path_penalty": -1}, {"max_k": 1},
                                    {"max_depth": 1.5}, {"min_samples_leaf": 0},
                                    {"time_limit": np.inf}, {"beam_width": 0}])
def test_invalid_parameters_fail_before_reference_ensemble_training(kwargs):
    with pytest.raises(ValueError):
        SPLINTER(**kwargs).fit([[0], [1]], [0, 1])


def test_tree_routing_agrees_at_cuts_and_across_nonlinear_groups():
    tree = MWShapeNode(0, "x", [0.5, 1.5, 2.5], [0, 1, 0, 1],
                       [MWLeaf(0, 2, 0), MWNode(1, "z", [0.0],
                                               [MWLeaf(1, 1, 0), MWLeaf(2, 1, 0)])])
    X = np.array([[0.49, 3], [0.5, -1], [1.5, -1], [2.5, 0]])
    np.testing.assert_array_equal(predict_encoded(tree, X), [0, 1, 0, 2])


def test_multiclass_reference_columns_match_numeric_solver_labels():
    labels = np.array([11, 2, 10, 0, 1, 3, 4, 5, 6, 7, 8, 9])
    encoded, classes = _reference_encode_target(labels)
    np.testing.assert_array_equal(encoded, labels)
    assert classes == list(range(12))


def test_cache_forms_have_identical_validated_arrays_and_atomic_save(tmp_path):
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    source = cache(X, np.array([0, 0, 1, 1]))
    path = tmp_path / "reference.npz"
    save_threshold_cache(source, path, metadata={"split": "test"})
    loaded = load_threshold_cache(path)
    assert loaded.thresholds == source.thresholds
    np.testing.assert_array_equal(loaded.logits, source.logits)
    assert list(tmp_path.iterdir()) == [path]
    a = _coerce_cache(source, 4, 1, 2)
    b = _coerce_cache(path, 4, 1, 2)
    assert a["thresholds"] == b["thresholds"]
    with pytest.raises(ValueError, match="different number of rows"):
        _coerce_cache(path, 3, 1, 2)
    with pytest.raises(ValueError, match="feature"):
        _coerce_cache({"thresholds": {2: [0.5]}, "teacher_logit": None}, 4, 1, 2)


def test_cache_npz_uses_plain_keys_and_reads_legacy_keys(tmp_path):
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    source = cache(X, np.array([0, 0, 1, 1]))
    path = tmp_path / "cache.npz"
    save_threshold_cache(source, path)
    with np.load(path) as data:
        arrays = {name: data[name] for name in data.files}
    assert "logits" in arrays and "boundary_gain" in arrays
    assert not any(name.startswith("teacher") for name in arrays)
    legacy = {("teacher_logit" if name == "logits" else
               "teacher_" + name if name.startswith("boundary_") else name): value
              for name, value in arrays.items()}
    np.savez(tmp_path / "legacy.npz", **legacy)
    loaded = load_threshold_cache(tmp_path / "legacy.npz")
    np.testing.assert_array_equal(loaded.logits, source.logits)
    np.testing.assert_array_equal(loaded.boundary_gain, source.boundary_gain)
    assert loaded.thresholds == source.thresholds


def test_resplinter_guarantee_never_certifies_best_found_anchor_or_delta():
    model = RESPLINTER()
    model.fit_info_ = {"first_stage": {
        "objective": 0.2,
        "lower_bound": 0.2,  # a screened search's nominal zero gap
        "gap": 0.0, "search_converged": True, "certified_optimal": False}}
    model.opt_ = 0.18
    model.rashomon_bound_ = 0.19
    model.theta_bound_ = 0.189
    model.rashomon_complete_ = False
    model.fit_info_["incomplete_reason_codes"] = "stepb_timeout"
    report = model.guarantee()
    assert report["search_converged"] and not report["best_tree_certified_optimal"]
    assert report["lower_bound"] is None and report["optimality_gap"] is None
    assert report["best_objective"] == 0.18 and report["search_objective"] == 0.2
    assert "complete_below_objective" not in report
    assert not report["complete"]
    sentence = ("Filling the levels below lookahead_depth reached its time limit at some "
                "leaves, so trees may be missing.")
    assert report["incomplete_reason"] == sentence
    assert report["completeness_note"] == sentence
    model.rashomon_complete_ = True
    model.fit_info_["incomplete_reason_codes"] = ""
    assert model.guarantee()["complete"]
    assert model.guarantee()["completeness_note"] is None
    assert model.guarantee()["coverage_scope"].startswith("candidate splits kept by the search")
    assert not model.guarantee()["inclusion_exact"]
    model.fit_info_["exact_cost_keys"] = True
    assert model.guarantee()["inclusion_exact"]
    assert model.guarantee()["objective_bound"] == 0.19
    model.fit_info_["exact_cost_keys"] = False
    assert not model.guarantee()["inclusion_exact"]


def _tiny_binary():
    X = np.array([[i % 4, i // 4] for i in range(24)], dtype=float)
    y = np.array([int(x[0] >= 2) for x in X])
    y[[5, 18]] ^= 1
    return X, y


@pytest.mark.parametrize("value", [0, 2, 5])
def test_resplinter_min_samples_leaf_rejected_before_reference_ensemble(monkeypatch, value):
    import splinter.thresholds as threshold_module

    def no_reference_ensemble(*args, **kwargs):
        raise AssertionError("the LightGBM reference ensemble must not be fit")

    monkeypatch.setattr(threshold_module, "fit_thresholds", no_reference_ensemble)
    X, y = _tiny_binary()
    for call in (lambda m: m.fit(X, y), lambda m: m._derive_thresholds(X, y)):
        with pytest.raises(ValueError, match="min_samples_leaf|minimum leaf size"):
            call(_RESPLINTER(min_samples_leaf=value))


def test_resplinter_lgbm_min_samples_leaf_param(monkeypatch):
    import splinter.thresholds as threshold_module
    import splinter.resplinter as resplinter_module
    seen = {"lgbm": [], "rust": []}
    fit_lgbm, solve = threshold_module.fit_thresholds, resplinter_module.solve_rashomon

    def spy_lgbm(*args, **kwargs):
        seen["lgbm"].append(kwargs["min_samples_leaf"])
        return fit_lgbm(*args, **kwargs)

    def spy_solve(*args, **kwargs):
        seen["rust"].append(kwargs["min_samples_leaf"])
        return solve(*args, **kwargs)

    monkeypatch.setattr(threshold_module, "fit_thresholds", spy_lgbm)
    monkeypatch.setattr(resplinter_module, "solve_rashomon", spy_solve)
    X, y = _tiny_binary()
    kw = dict(max_depth=2, lookahead_depth=1, beam_width=2, reg=0.02, theta=0.5,
              lgbm_n_estimators=8, cart_closure=False)
    model = RESPLINTER(**kw, lgbm_min_samples_leaf=5)
    assert model.get_params()["lgbm_min_samples_leaf"] == 5
    assert clone(model).get_params()["lgbm_min_samples_leaf"] == 5
    assert RESPLINTER().get_params()["lgbm_min_samples_leaf"] is None
    model.fit(X, y)
    assert model.lgbm_min_samples_leaf_ == 5 and model.tree_count_ >= 1
    assert model.dag_.fit_config["extras"]["lgbm_min_samples_leaf"] == 5
    default = RESPLINTER(**kw).fit(X, y)
    explicit = RESPLINTER(**kw, lgbm_min_samples_leaf=1).fit(X, y)
    assert default.lgbm_min_samples_leaf_ == explicit.lgbm_min_samples_leaf_ == 1
    assert default.tree_count_ == explicit.tree_count_
    # The reference ensemble gets the requested leaf size; the Rashomon search always gets 1.
    assert seen["lgbm"] == [5, 1, 1] and seen["rust"] == [1, 1, 1]
    with pytest.raises(ValueError, match="lgbm_min_samples_leaf"):
        RESPLINTER(**kw, lgbm_min_samples_leaf=0).fit(X, y)


def test_splinter_lgbm_min_samples_leaf_none_still_rejected():
    with pytest.raises(ValueError, match="lgbm_min_samples_leaf"):
        SPLINTER(lgbm_min_samples_leaf=None).fit([[0], [1]], [0, 1])


def test_rid_accepts_lgbm_min_samples_leaf():
    from splinter.rid import compute_rid
    X, y = _tiny_binary()
    estimator = RESPLINTER(max_depth=2, max_k=3, reg=0.02, theta=0.3, beam_width=4,
                           lgbm_n_estimators=8, lgbm_min_samples_leaf=3)
    result = compute_rid(estimator, X, y, B=2)
    assert result is not None and result.B_ok == 2


@pytest.mark.parametrize("kwargs", [{"path_penalty": 0.01}, {"reg_alpha": 0.5}])
def test_resplinter_rejects_objectives_not_represented_by_factored_dag(kwargs):
    with pytest.raises(ValueError, match="leaf-only objective"):
        _RESPLINTER(**kwargs).fit([[0.0], [1.0]], [0, 1])


def test_native_rashomon_rejects_invalid_class_indices_and_objective():
    from splinter.multiway_solver import solve_rashomon
    kwargs = dict(x=np.array([[0.0], [1.0]]), y=np.array([0, 7], dtype=np.uint8),
                  n_classes=2, reg=0.01, max_depth=1, max_k=2, theta=0.05)
    with pytest.raises(ValueError, match="encoded labels"):
        solve_rashomon(**kwargs)
    kwargs["y"] = np.array([0, 1], dtype=np.uint8)
    with pytest.raises(ValueError, match="leaf-only objective"):
        solve_rashomon(**kwargs, path_penalty=0.01)


def test_resplinter_rejects_multiclass_before_native_call_and_preserves_labels():
    with pytest.raises(ValueError, match="Only binary classification is supported.*got 3 classes"):
        RESPLINTER().fit([[0], [1], [2]], [0, 1, 2])
    X = np.repeat([[0.0], [1.0]], 5, axis=0)
    y = np.repeat(["no", "yes"], 5)
    model = RESPLINTER(max_depth=1, max_k=2, reg=0.01,
                       fixed_thresholds={0: [0.5]}, cart_closure=False).fit(X, y)
    np.testing.assert_array_equal(model.predict(X), y)
    np.testing.assert_array_equal(model.classes_, ["no", "yes"])
    assert "complete_below_objective" not in model.guarantee()


_BASES = {SPLINTER: "contiguous._SPLINTER",
          ShapeSPLINTER: "nonlinear._ShapeSPLINTER",
          LicketySPLINTER: "lickety_splinter._LicketySPLINTER",
          RESPLINTER: "resplinter._RESPLINTER"}


@pytest.mark.parametrize("public", list(_BASES))
def test_hidden_settings_are_class_defaults_not_parameters(public):
    """A public class fixes each base-class parameter missing from its
    signature as a class attribute equal to the base default; it is not a
    parameter (get_params, set_params, clone, repr) or an instance attribute."""
    import importlib
    import inspect
    module, name = _BASES[public].rsplit(".", 1)
    base = getattr(importlib.import_module(f"splinter.{module}"), name)
    assert issubclass(public, base)
    base_params = inspect.signature(base.__init__).parameters
    public_params = [p for p in inspect.signature(public.__init__).parameters if p != "self"]
    assert set(public_params) <= set(base_params)
    for p in public_params:
        assert inspect.signature(public.__init__).parameters[p].default == base_params[p].default
    hidden = [p for p in base_params if p != "self" and p not in public_params]
    model = public()
    assert sorted(model.get_params()) == sorted(public_params)
    assert sorted(vars(model)) == sorted(public_params)
    copied = clone(model)
    for p in hidden:
        default = base_params[p].default
        assert p in vars(public) and type(getattr(public, p)) is type(default), p
        assert getattr(public, p) == default, p
        assert getattr(copied, p) == default, p
        assert f"{p}=" not in repr(model)
        with pytest.raises(ValueError):
            public().set_params(**{p: default})
    with pytest.raises(TypeError):
        public(**{p: base_params[p].default for p in hidden[:1]}) if hidden else public(bogus=1)


def test_single_tree_fit_takes_only_data_names_and_weights():
    import inspect
    for cls in (SPLINTER, ShapeSPLINTER):
        assert list(inspect.signature(cls.fit).parameters) == [
            "self", "X", "y", "feature_names", "sample_weight"]


@pytest.mark.parametrize("loader", ["load_breast_cancer", "load_wine"])
def test_contiguous_lickety_ignores_precomputed_logits(loader):
    import sklearn.datasets
    from splinter.rules import export_text
    X, y = getattr(sklearn.datasets, loader)(return_X_y=True)
    logits = np.random.default_rng(0).normal(size=(len(X), len(np.unique(y)))) * 5
    kw = dict(max_depth=3, min_samples_leaf=3, lgbm_n_estimators=16, shape_mode=False)
    plain = LicketySPLINTER(**kw).fit(X, y)
    given = LicketySPLINTER(**kw, precomputed_logits=logits).fit(X, y)
    assert export_text(given) == export_text(plain)
    np.testing.assert_array_equal(given.predict(X), plain.predict(X))


# The paper's RESPLINTER configuration, spelled out (the hidden settings
# included); the defaults must reproduce it.
_PAPER_SETTINGS = dict(
    max_depth=4, lookahead_depth=2, max_k=4, reg=0.01, theta=0.05, beam_width=4,
    transition_delta=0.02, cart_closure=False, fill_mode="recursive", atom_mode=True,
    binary_threshold_backfill=False, lgbm_num_leaves=31, lgbm_n_estimators=200,
    lgbm_min_samples_leaf=1, interval_lower_bound=True)


def test_resplinter_defaults_are_the_paper_configuration():
    from sklearn.datasets import load_breast_cancer
    X, y = load_breast_cancer(return_X_y=True)
    rows = np.random.default_rng(0).choice(len(X), 200, replace=False)
    # Coarse features keep the default depth-4 fit fast.
    X, y = np.round(X[rows, :6] / X[rows, :6].std(axis=0)), y[rows]
    default = RESPLINTER().fit(X, y)
    assert default.rashomon_complete_ is True
    assert default.fit_info_["fill_mode"] == "recursive"
    assert default.fit_info_["completion"]["mode"] == "recursive"
    assert default.fit_info_["raw_result"]["stepb_mode"] == "paper"
    assert default.dag_.fit_config["extras"]["lookahead_depth"] == 2
    explicit = _RESPLINTER(**_PAPER_SETTINGS).fit(X, y)
    for name in ("tree_count_", "tree_count_at_target_", "opt_", "theta_bound_"):
        assert getattr(default, name) == getattr(explicit, name), name
    assert bytes(default.dag_.to_bytes(6)) == bytes(explicit.dag_.to_bytes(6))
    # lookahead_depth=max_depth enumerates every level in one search.
    full = RESPLINTER(max_depth=2, lookahead_depth=2).fit(X, y)
    assert full.fit_info_["fill_mode"] == "full" and full.rashomon_complete_ is True
    assert full.fit_info_["completion"]["mode"] == "none"
    # The check that the set's best tree is within the first search's tree
    # also runs at full depth (it was skipped there before).
    assert full.fit_info_["full_depth_tree_in_set"] is True
