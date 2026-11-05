"""Bad user input raises a Python exception with a plain message, never a
native panic: the native entry points (labels, class count, max_k, feature
count, feature_types, criterion names, threshold / category counts,
sample_weight layout, prediction columns), saved DAG files whose header
claims huge tables, and the Python estimators (save, refit state, the
RESPLINTER cache and time settings, message wording).
"""
import os
import struct
import zlib

import numpy as np
import pytest

from splinter import LicketySPLINTER, RESPLINTER
from splinter import multiway_solver as ms

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
RNG = np.random.default_rng(0)
X = RNG.random((40, 3))
Y = (X[:, 0] > 0.5).astype(np.uint8)
THRESHOLDS = {0: [0.25, 0.5, 0.75], 1: [0.5], 2: [0.5]}


def _bnb(**kw):
    args = dict(x=X, y=Y, n_classes=2, reg=0.01, max_depth=2, max_k=3, min_samples_leaf=1,
                feature_exclusion=False, time_limit=None, initial_used_features=[],
                feature_names=["a", "b", "c"], thresholds_per_feature=THRESHOLDS)
    args.update(kw)
    return ms.solve_branch_and_bound(**args)


def _rashomon(**kw):
    args = dict(x=X, y=Y, n_classes=2, reg=0.01, max_depth=2, max_k=3, theta=0.05,
                thresholds_per_feature=THRESHOLDS, fill_parallel=False,
                feature_names=["a", "b", "c"])
    args.update(kw)
    return ms.solve_rashomon(**args)


def _select_debug(**kw):
    args = dict(x=X, y=Y, n_classes=2, reg=0.01, max_depth=2, max_k=3, rashomon_bound=0.5,
                support_idx=list(range(len(Y))), thresholds_per_feature=THRESHOLDS)
    args.update(kw)
    return ms.rashomon_select_debug(**args)


ENTRY_POINTS = [_bnb, _rashomon, _select_debug]


@pytest.mark.parametrize("solve", ENTRY_POINTS)
def test_labels_and_class_count(solve):
    with pytest.raises(ValueError, match="encoded labels"):
        solve(y=np.where(Y == 1, 5, 0).astype(np.uint8))
    with pytest.raises(ValueError, match="between 2 and 16"):
        solve(n_classes=17, y=(np.arange(len(Y)) % 17).astype(np.uint8))


@pytest.mark.parametrize("solve", ENTRY_POINTS)
def test_max_k_above_255(solve):
    with pytest.raises(ValueError, match="max_k must be at most 255"):
        solve(max_k=256)


def test_more_than_65535_features():
    x = np.zeros((2, 65536))
    with pytest.raises(ValueError, match="at most 65535 features"):
        _rashomon(x=x, y=np.array([0, 1], np.uint8), thresholds_per_feature=None,
                  feature_names=None)


@pytest.mark.parametrize("solve", ENTRY_POINTS)
def test_feature_types(solve):
    with pytest.raises(ValueError, match="feature_types has 2 entries for 3 features"):
        solve(feature_types=["num", "num"])
    with pytest.raises(ValueError, match="unknown feature type 'bogus'|unknown feature type \"bogus\""):
        solve(feature_types=["num", "bogus", "num"])
    solve(feature_types=["num", "numeric", "cat"])


def test_criterion_names():
    with pytest.raises(ValueError, match="unknown dp_criterion"):
        _bnb(dp_criterion="bogus")
    with pytest.raises(ValueError, match="unknown atom_merge_rule"):
        _bnb(atom_merge_rule="bogus")
    for solve in (_bnb, _rashomon):
        with pytest.raises(ValueError, match="unknown closure criterion"):
            solve(closure_criterion="bogus")
    # "greedy" (RESPLINTER's default cart_closure_mode) is Gini.
    assert (_rashomon(closure_criterion="greedy")["tree_count"]
            == _rashomon(closure_criterion="gini")["tree_count"])


def test_too_many_thresholds_or_categories():
    many = {0: list(np.linspace(0, 1, 65535)), 1: [0.5], 2: [0.5]}
    with pytest.raises(ValueError, match="feature 0 has 65535 thresholds"):
        _bnb(thresholds_per_feature=many)
    with pytest.raises(ValueError, match="feature 0 has 65535 thresholds"):
        _bnb(closure_thresholds_per_feature=many)
    n = 65537
    x = np.column_stack([np.arange(n, dtype=np.float64), np.zeros(n)])
    y = (np.arange(n) % 2).astype(np.uint8)
    with pytest.raises(ValueError, match="categorical feature 0 has 65537 categories"):
        _rashomon(x=x, y=y, thresholds_per_feature=None, feature_names=None,
                  feature_types=["cat", "num"])


def test_sample_weights_layout():
    strided = np.ones(2 * len(Y))[::2]
    assert not strided.flags["C_CONTIGUOUS"]
    a = _bnb(sample_weights=strided)
    b = _bnb(sample_weights=np.ones(len(Y)))
    assert a["tree"] == b["tree"]
    with pytest.raises(ValueError, match="sample_weight has 39 entries for 40 rows"):
        _bnb(sample_weights=np.ones(len(Y) - 1))


def test_prediction_columns():
    dag = _rashomon(max_depth=3, reg=0.001, theta=0.5)["dag"]
    for bad in (X[:, :1].copy(), np.hstack([X, X[:, :2]])):
        with pytest.raises(ValueError, match="columns, the DAG was fit on 3"):
            dag.predict_probs(bad)
        with pytest.raises(ValueError, match="columns, the DAG was fit on 3"):
            dag.class1_counts(bad)
    with pytest.raises(IndexError):
        dag.get_tree(int(dag.tree_count))


@pytest.mark.parametrize("name", ["dag_v4_fixture.bin", "dag_v5_fixture.bin"])
def test_saved_file_with_huge_table_counts(name):
    data = bytearray(open(os.path.join(DATA, name), "rb").read())
    body = len(data) - 8
    assert struct.unpack_from("<I", data, body)[0] == zlib.crc32(bytes(data[:body]))
    # n_subproblems and n_thresholds of the header claim 2**32 - 1 entries.
    struct.pack_into("<II", data, 8, 0xFFFFFFFF, 0xFFFFFFFF)
    struct.pack_into("<I", data, body, zlib.crc32(bytes(data[:body])))
    with pytest.raises(ValueError):
        ms.PyRashomonDag.from_bytes(bytes(data))


# ------------------------------------------------------------ Python layer
def _small():
    from sklearn.datasets import make_classification
    return make_classification(120, 4, random_state=0)


def test_save_refuses_unsavable_fitted_attribute(tmp_path):
    X_, y_ = _small()
    model = RESPLINTER(max_depth=2, lgbm_n_estimators=8).fit(X_, y_)
    model.extra_ = np.zeros(3)
    with pytest.raises(ValueError, match="Cannot save fitted attribute extra_"):
        model.save(tmp_path / "m.npz")
    del model.extra_
    model.fit_info_["extra"] = object()
    with pytest.raises(ValueError, match="Cannot save fitted attribute fit_info_"):
        model.save(tmp_path / "m.npz")


def test_refit_clears_lickety_private_state():
    X_, y_ = _small()
    model = LicketySPLINTER(max_depth=2, lgbm_n_estimators=8).fit(X_, y_)
    private = ("_fit_start_time", "_n_solver_calls", "_thresholds_for_rust",
               "_closure_feature_ranking", "_fit_time")
    assert all(hasattr(model, name) for name in private)
    with pytest.raises(ValueError):
        model.fit(X_, np.zeros(len(y_)))
    assert not any(hasattr(model, name) for name in private)


@pytest.mark.parametrize("setting", [
    dict(screen_time_cap=0.0), dict(screen_time_cap=float("inf")),
    dict(screen_time_cap=float("nan")), dict(lower_bound_memo_size=0),
    dict(depth1_memo_size=0), dict(interval_cache_size=-1),
    dict(lower_bound_memo_size=1.5)])
def test_resplinter_cache_and_time_settings(setting):
    X_, y_ = _small()
    model = RESPLINTER(max_depth=2, lgbm_n_estimators=8)
    for name, value in setting.items():  # hidden settings: class attributes
        setattr(model, name, value)
    with pytest.raises(ValueError, match=next(iter(setting))):
        model.fit(X_, y_)


def test_resplinter_rejects_unknown_closure_mode():
    X_, y_ = _small()
    with pytest.raises(ValueError, match="unknown closure criterion"):
        model = RESPLINTER(max_depth=2, lgbm_n_estimators=8)
        model.cart_closure_mode = "bogus"  # hidden setting: a class attribute
        model.fit(X_, y_)


def test_messages_name_no_internals():
    X_, y_ = _small()
    model = RESPLINTER(max_depth=2, lgbm_n_estimators=8).fit(X_, y_)

    class Unfinished:  # a count stopped by max_seconds
        def class1_counts(self, X, **kw):
            return {"complete": False, "count_mismatch": 0}

    model.dag_ = Unfinished()
    with pytest.raises(RuntimeError, match="did not finish within max_seconds") as err:
        model.predict_class1_counts(X_)
    assert "class1_counts" not in str(err.value)
