"""LightGBM reference-ensemble output options and atomic cache replacement."""

import numpy as np
import pytest

from splinter import thresholds as threshold_module


def training_data():
    X = np.repeat(np.arange(4, dtype=float), 4).reshape(-1, 1)
    y = (X[:, 0] >= 2).astype(np.uint8)
    return X, y


def fit_reference_ensemble(**options):
    X, y = training_data()
    return threshold_module.fit_thresholds(
        X, y, n_estimators=4, min_samples_leaf=1, max_bins=8,
        collect_bin_representatives=False, **options,
    )


def test_boundary_priors_are_independent_of_reference_logits():
    complete = fit_reference_ensemble()
    without_logits = fit_reference_ensemble(collect_logits=False)
    assert without_logits.logits is None
    assert without_logits.thresholds == complete.thresholds
    assert np.any(without_logits.boundary_gain > 0)
    for name in ("boundary_gain", "boundary_cover", "boundary_value_jump"):
        np.testing.assert_array_equal(getattr(without_logits, name), getattr(complete, name))


def test_logits_can_be_collected_without_boundary_priors():
    cache = fit_reference_ensemble(collect_boundary_priors=False)
    assert cache.logits.shape == (16, 2)
    assert cache.boundary_gain is None
    assert cache.boundary_cover is None
    assert cache.boundary_value_jump is None


def test_failed_cache_save_preserves_existing_cache(tmp_path, monkeypatch):
    cache = fit_reference_ensemble()
    path = tmp_path / "reference.npz"
    threshold_module.save_threshold_cache(cache, path, metadata={"original": True})
    original = path.read_bytes()

    def failed_write(stream, **payload):
        stream.write(b"incomplete archive")
        raise OSError("disk full")

    monkeypatch.setattr(threshold_module.np, "savez", failed_write)
    with pytest.raises(OSError, match="disk full"):
        threshold_module.save_threshold_cache(cache, path, metadata={"original": False})
    assert path.read_bytes() == original
    assert sorted(p.name for p in tmp_path.iterdir()) == ["reference.npz"]
