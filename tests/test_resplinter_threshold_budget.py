"""RESPLINTER `threshold_budget` (HEURISTIC LightGBM-threshold budget, off by
default): None leaves the LightGBM thresholds exactly as before; a budget
keeps fewer thresholds (the same rule as LicketySPLINTER). Small synthetic
fits only. KW sets hidden settings, so the fits use the base class.
"""
import copy

import numpy as np

from splinter.resplinter import _RESPLINTER as RESPLINTER
from splinter.thresholds import _apply_budget, fit_thresholds, _parse_budget, _gain_per_threshold

KW = dict(max_depth=3, lookahead_depth=3, max_k=2, reg=0.02, theta=0.05, beam_width=1,
          shape_mode=False, atom_mode=True, cart_closure=False, cart_closure_beam=3,
          cart_closure_mode="rush", lgbm_num_leaves=31, lgbm_n_estimators=200,
          binary_threshold_backfill=False, fill_mode="full", time_limit=60.0, verbose=False)


def _data(n=300, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4)).round(2)
    y = ((X[:, 0] + 0.5 * X[:, 1] + 0.3 * rng.normal(size=n)) > 0).astype(int)
    return X, y


def test_none_thresholds_identical_and_budget_reduces():
    X, y = _data()
    kw = dict(KW)
    full = RESPLINTER(**kw).fit(X, y)
    again = RESPLINTER(**kw, threshold_budget=None).fit(X, y)
    assert again.lgbm_thresholds_ == full.lgbm_thresholds_
    assert again.tree_count_ == full.tree_count_
    assert full.threshold_budget_selected_ is None
    n_full = sum(len(v) for v in full.lgbm_thresholds_.values())
    cut = RESPLINTER(**kw, threshold_budget=0.8).fit(X, y)
    n_cut = sum(len(v) for v in cut.lgbm_thresholds_.values())
    assert n_cut < n_full
    assert cut.threshold_budget_selected_ == n_cut
    assert cut.threshold_budget_total_ == n_full
    # Kept thresholds are a subset of the full ones, per feature.
    for j, v in cut.lgbm_thresholds_.items():
        assert set(v) <= set(full.lgbm_thresholds_.get(j, []))
    # Same rule as LicketySPLINTER.
    m = RESPLINTER(**kw)
    res = fit_thresholds(X, y, max_bins=int(m.max_bins), n_estimators=int(m.lgbm_n_estimators),
                         min_samples_leaf=cut.lgbm_min_samples_leaf_, random_state=m.lgbm_random_state,
                         num_threads=1, max_depth=-1, num_leaves=m.lgbm_num_leaves or 63)
    ranked = _gain_per_threshold(res.thresholds, res.boundary_gain)
    want = _apply_budget(copy.deepcopy(res.thresholds), ranked,
                         _parse_budget(0.8)(ranked))
    assert cut.lgbm_thresholds_ == want
