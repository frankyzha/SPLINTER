"""Estimator persistence on small synthetic and toy data:

  * pickle round trips for all four estimators keep predictions, text,
    trees (with leaf `n_samples`), the Rashomon table and probabilities;
  * `RESPLINTER.save` / `RESPLINTER.load` give the same results, and a
    corrupted or truncated file, or one with another format version, raises
    ValueError;
  * fitted estimators drop the reference-ensemble arrays, so single-tree
    pickles are at least 5x smaller.
"""
import copy
import json
import os
import pickle

import numpy as np
import pytest
from sklearn.datasets import load_breast_cancer

from splinter import SPLINTER, ShapeSPLINTER, LicketySPLINTER, RESPLINTER
from splinter.explore import rashomon_table
from splinter.thresholds import _prepare

from _helpers import tiny_data

LABELS = ("no", "yes")


@pytest.fixture(scope="module")
def rashomon():
    X, y = tiny_data(labels=LABELS)
    model = RESPLINTER(max_depth=2, max_k=3, reg=0.02, theta=0.3, beam_width=4,
                       lgbm_n_estimators=20).fit(X, y)
    assert model.tree_count_at_target_ >= 20
    return model, X, y


def _same_rashomon(a, b, X, y):
    np.testing.assert_array_equal(a.predict(X), b.predict(X))
    np.testing.assert_array_equal(a.predict_proba(X), b.predict_proba(X))
    assert str(a) == str(b)
    assert a.guarantee() == b.guarantee()
    for i in range(a.tree_count_):
        assert a.get_tree(i) == b.get_tree(i), i
    ta, tb = rashomon_table(a, X, y), rashomon_table(b, X, y)
    assert ta.equals(tb)
    np.testing.assert_array_equal(ta.attrs["predictions"].matrix,
                                  tb.attrs["predictions"].matrix)


@pytest.mark.parametrize("kind", [SPLINTER, ShapeSPLINTER, LicketySPLINTER])
def test_single_tree_pickle_round_trip(kind):
    X, y = tiny_data(labels=LABELS)
    model = kind(max_depth=2, min_samples_leaf=5, lgbm_n_estimators=8).fit(X, y)
    restored = pickle.loads(pickle.dumps(model))
    np.testing.assert_array_equal(restored.predict(X), model.predict(X))
    np.testing.assert_array_equal(restored.predict_proba(X), model.predict_proba(X))
    assert str(restored) == str(model)
    assert restored.tree_ == model.tree_


def test_resplinter_pickle_round_trip(rashomon):
    model, X, y = rashomon
    restored = pickle.loads(pickle.dumps(model))
    _same_rashomon(model, restored, X, y)
    # The live estimator keeps its native DAG.
    assert type(model.dag_).__name__ == "PyRashomonDag"
    assert "dag" not in model.fit_info_["raw_result"]


def test_unfitted_resplinter_pickles():
    model = RESPLINTER(max_depth=3)
    assert pickle.loads(pickle.dumps(model)).get_params() == model.get_params()


def test_resplinter_save_load(rashomon, tmp_path):
    model, X, y = rashomon
    path = tmp_path / "model.npz"
    model.save(path)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["model.npz"]
    loaded = RESPLINTER.load(path)
    _same_rashomon(model, loaded, X, y)
    assert loaded.get_params() == model.get_params()
    np.testing.assert_array_equal(loaded.classes_, model.classes_)
    assert loaded.classes_.dtype == model.classes_.dtype
    for name in ("feature_names_", "n_features_in_", "n_classes_", "n_total_", "tree_count_",
                 "tree_count_at_target_", "opt_", "rashomon_bound_", "theta_bound_",
                 "n_dag_nodes_", "fit_time_", "rashomon_complete_",
                 "rashomon_incomplete_reason_", "count_saturated_", "lgbm_thresholds_"):
        assert getattr(loaded, name) == getattr(model, name), name
    assert loaded.fit_info_ == json.loads(json.dumps(model.fit_info_))
    np.testing.assert_array_equal(loaded.dag_.tree_phis([0, 1]), model.dag_.tree_phis([0, 1]))


def test_resplinter_save_load_fixed_thresholds_and_feature_names(tmp_path):
    import pandas as pd
    X, y = tiny_data(seed=3, labels=LABELS)
    frame = pd.DataFrame(X, columns=["a", "b", "c"])
    model = RESPLINTER(max_depth=2, max_k=2, reg=0.02, theta=0.2,
                       fixed_thresholds={0: [1.5, 2.5], 1: [0.5], 2: []}).fit(frame, y)
    model.save(tmp_path / "fixed.npz")
    loaded = RESPLINTER.load(tmp_path / "fixed.npz")
    assert loaded.get_params() == model.get_params()
    np.testing.assert_array_equal(loaded.feature_names_in_, model.feature_names_in_)
    _same_rashomon(model, loaded, frame, y)


def test_resplinter_save_rejects_array_parameters(rashomon, tmp_path):
    model, X, _ = rashomon
    clone_ = copy.copy(model)
    clone_.lgbm_random_state = np.random.RandomState(0)
    with pytest.raises(ValueError, match="lgbm_random_state"):
        clone_.save(tmp_path / "bad.npz")
    assert not list(tmp_path.iterdir())


def test_resplinter_load_rejects_corrupted_or_truncated_files(rashomon, tmp_path):
    model, _, _ = rashomon
    path = tmp_path / "model.npz"
    model.save(path)
    data = path.read_bytes()
    truncated = tmp_path / "truncated.npz"
    truncated.write_bytes(data[: len(data) // 2])
    garbage = tmp_path / "garbage.npz"
    garbage.write_bytes(b"not a model file" * 10)
    for bad in (truncated, garbage):
        with pytest.raises(ValueError):
            RESPLINTER.load(bad)
    # A valid archive whose DAG bytes are damaged.
    with np.load(path, allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    dag = arrays["dag"].copy()
    dag[len(dag) // 2] ^= 0xFF
    arrays["dag"] = dag
    damaged = tmp_path / "damaged.npz"
    np.savez(damaged, **arrays)
    with pytest.raises(ValueError):
        RESPLINTER.load(damaged)


def test_save_dag_and_load_dag_removed():
    assert not hasattr(RESPLINTER, "save_dag")
    assert not hasattr(RESPLINTER, "load_dag")


@pytest.mark.parametrize("kind", [SPLINTER, LicketySPLINTER])
def test_single_tree_pickle_is_slim(kind):
    X, y = load_breast_cancer(return_X_y=True)
    model = kind(max_depth=2, lgbm_n_estimators=16).fit(X, y)
    for name in ("boundary_gain_", "boundary_cover_", "boundary_value_jump_",
                 "bin_representatives_", "bin_fill_values_", "n_bins_per_feature_",
                 "_logits_full"):
        assert not hasattr(model, name), name
    assert model.lgbm_thresholds_
    slim = len(pickle.dumps(model))
    # The same model with the reference-ensemble arrays it used to keep.
    full = copy.deepcopy(model)
    _prepare(full, np.ascontiguousarray(X, dtype=np.float64),
                    np.searchsorted(model.classes_, y).astype(np.uint8))
    assert full.lgbm_thresholds_ == model.lgbm_thresholds_
    before = len(pickle.dumps(full))
    assert before >= 5 * slim, (before, slim)


def test_resplinter_drops_reference_ensemble_arrays(rashomon):
    model, X, _ = rashomon
    for name in ("boundary_gain_", "boundary_cover_", "boundary_value_jump_",
                 "bin_representatives_", "bin_fill_values_", "n_bins_per_feature_",
                 "_logits_full", "_logits"):
        assert not hasattr(model, name), name
    assert model.lgbm_thresholds_
    restored = pickle.loads(pickle.dumps(model))
    np.testing.assert_array_equal(restored.predict(X), model.predict(X))


def test_resplinter_keeps_copies_of_its_training_data(tmp_path):
    X, y = tiny_data(seed=5, labels=LABELS)
    model = RESPLINTER(max_depth=2, max_k=3, reg=0.02, theta=0.3, beam_width=4,
                       lgbm_n_estimators=20).fit(X, y)
    trees = [model.get_tree(i) for i in range(model.tree_count_)]
    X += 1.0
    for edges in model.lgbm_thresholds_.values():
        edges.append(99.0)
    model.save(tmp_path / "model.npz")
    for restored in (pickle.loads(pickle.dumps(model)), RESPLINTER.load(tmp_path / "model.npz")):
        assert [restored.get_tree(i) for i in range(restored.tree_count_)] == trees


def test_resplinter_load_reports_missing_contents(rashomon, tmp_path):
    model, _, _ = rashomon
    path = tmp_path / "model.npz"
    model.save(path)
    with np.load(path, allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    without_dag = dict(arrays)
    del without_dag["dag"]
    np.savez(tmp_path / "no_dag.npz", **without_dag)
    with pytest.raises(ValueError, match="missing arrays"):
        RESPLINTER.load(tmp_path / "no_dag.npz")
    import json
    fitted = json.loads(str(arrays["fitted_json"]))
    del fitted["feature_names_"]
    np.savez(tmp_path / "no_names.npz", **dict(arrays, fitted_json=np.asarray(json.dumps(fitted))))
    with pytest.raises(ValueError, match="missing fitted attributes"):
        RESPLINTER.load(tmp_path / "no_names.npz")


def test_resplinter_load_rejects_other_format_version(rashomon, tmp_path):
    model, _, _ = rashomon
    path = tmp_path / "model.npz"
    model.save(path)
    with np.load(path, allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    np.savez(tmp_path / "v2.npz", **dict(arrays, format_version=np.asarray(2)))
    with pytest.raises(ValueError, match="unsupported format_version 2"):
        RESPLINTER.load(tmp_path / "v2.npz")


def test_resplinter_save_uses_normal_permissions(rashomon, tmp_path):
    model, _, _ = rashomon
    path = tmp_path / "model.npz"
    model.save(path)
    umask = os.umask(0)
    os.umask(umask)
    assert (path.stat().st_mode & 0o777) == (0o666 & ~umask)
