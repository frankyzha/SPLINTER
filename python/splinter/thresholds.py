"""LightGBM reference-ensemble threshold extraction for SPLINTER.

SPLINTER, ShapeSPLINTER and LicketySPLINTER use one threshold source: a
LightGBM classifier (the reference ensemble) trained on the supplied data. The
cache returned by `fit_thresholds` contains per-feature split
thresholds, split-gain boundary scores, raw reference-ensemble logits, and bin
diagnostics. Python estimators reject bare quantile or midpoint threshold
dictionaries so a fit cannot silently switch threshold semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
import tempfile
import time
from numbers import Integral, Real
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Dict, List, Optional

import numpy as np

__all__ = ["ThresholdCache", "fit_thresholds", "save_threshold_cache", "load_threshold_cache"]

if TYPE_CHECKING:
    from lightgbm import LGBMClassifier


# ---------------------------------------------------------------------------
# Threshold budget — post-cache per-edge pruning by per-threshold split gain.
# Operates after threshold extraction using the LightGBM reference ensemble's accumulated
# boundary split gain. A float budget keeps the
# shortest top-ranked prefix whose non-negative gain mass reaches
# `p * sum(max(0, gain_e))`; an integer budget keeps exactly that many
# top-ranked thresholds.
# ---------------------------------------------------------------------------


def _gain_per_threshold(thresholds, boundary_gain):
    """Rank per-feature thresholds by cached LightGBM split gain.

    `boundary_gain[j, r]` is aligned with the r-th sorted threshold for
    feature j in `fit_thresholds` and in the cache it returns.
    """
    if boundary_gain is None:
        return None
    gains = np.asarray(boundary_gain, dtype=np.float64)
    if gains.ndim != 2:
        return None
    flat = []
    for j in sorted(thresholds):
        if j < 0 or j >= gains.shape[0]:
            return None
        vals = thresholds[j]
        if len(vals) > gains.shape[1]:
            return None
        for r, t in enumerate(vals):
            g = gains[j, r]
            flat.append((j, float(t), float(g) if np.isfinite(g) else 0.0))
    flat.sort(key=lambda x: (-x[2], x[0], x[1]))
    return flat


def _gain_mass_budget(ranked_edges, gain_mass_fraction, min_thresholds=1):
    """Return the smallest top-B prefix covering a gain-mass fraction.

    Parameters
    ----------
    ranked_edges : list of (feat, thresh, gain) sorted by gain desc.
    gain_mass_fraction : float
        Target fraction of sum(max(0, gain)).
    min_thresholds : int
        Optional lower bound on the number of thresholds kept.

    Returns
    -------
    int : number of edges to keep (≥ min_thresholds, ≤ len(ranked_edges)).
    """
    if not np.isfinite(gain_mass_fraction) or not (0.0 < gain_mass_fraction < 1.0):
        raise ValueError("float threshold_budget must be in (0, 1).")
    total_edges = len(ranked_edges)
    min_thresholds = max(0, min(int(min_thresholds), total_edges))
    if total_edges <= min_thresholds:
        return total_edges
    pos_gains = [g for _, _, g in ranked_edges if g > 0]
    if not pos_gains:
        # With no positive gain signal there is no meaningful top-p prefix.
        # Keep all thresholds rather than pruning arbitrarily by tie-break.
        return total_edges
    target = gain_mass_fraction * sum(pos_gains)
    cum = 0.0
    for k, (_, _, g) in enumerate(ranked_edges, start=1):
        if g > 0:
            cum += g
        if cum >= target:
            return max(min_thresholds, k)
    return total_edges


def _parse_budget(token):
    """Parse a threshold budget into a callable over ranked edges.

    Accepts:
      - None / 'full' / 'all' / 'none' -> keep all.
      - float in (0, 1), including strings like '0.99' -> gain-mass prefix.
      - integer, including strings like '1024' -> exact global top-K count.
    """
    if token is None:
        return lambda edges: len(edges)
    if isinstance(token, bool):
        raise ValueError("threshold_budget must be None, a float in (0, 1), or an integer count.")
    if isinstance(token, Integral):
        if token < 1:
            raise ValueError("integer threshold_budget must be >= 1.")
        target = int(token)
        return lambda edges: min(target, len(edges))
    if isinstance(token, Real):
        if not np.isfinite(token) or not (0.0 < token < 1.0):
            raise ValueError("float threshold_budget must be in (0, 1).")
        frac = float(token)
        return lambda edges: _gain_mass_budget(
            edges, gain_mass_fraction=frac, min_thresholds=1
        )
    s = str(token).strip().lower()
    if s in ('full', 'all', 'none'):
        return lambda edges: len(edges)
    try:
        target = int(s)
    except ValueError:
        target = None
    if target is not None and str(target) == s:
        if target < 1:
            raise ValueError("integer threshold_budget must be >= 1.")
        return lambda edges: min(target, len(edges))
    try:
        frac = float(s)
    except ValueError as e:
        raise ValueError(f"Invalid threshold_budget token: {token}") from e
    else:
        if not np.isfinite(frac) or not (0.0 < frac < 1.0):
            raise ValueError("float threshold_budget must be in (0, 1).")
        return lambda edges: _gain_mass_budget(
            edges, gain_mass_fraction=frac, min_thresholds=1
        )


def _apply_budget(thresholds, ranked_edges, target_count):
    """Filter thresholds to the top-ranked global budget.

    Parameters
    ----------
    thresholds : dict[int, list[float]]
        Original threshold dict.
    ranked_edges : list of (feat, thresh, gain) sorted by gain desc.
    target_count : int
        Number of edges to keep globally.

    Returns
    -------
    dict[int, list[float]] : pruned thresholds, sorted ascending per feature.
        Features that were pruned to zero thresholds are kept with an
        empty list so Rust does not fall back to midpoint thresholds.
    """
    target_count = max(0, min(int(target_count), len(ranked_edges)))
    if target_count >= len(ranked_edges):
        return {int(j): sorted(set(map(float, edges))) for j, edges in thresholds.items()}
    kept_per_feat = {j: [] for j in thresholds}
    # Take the top `target_count` globally.
    for k, (j, t, _) in enumerate(ranked_edges):
        if k >= target_count:
            break
        if j in kept_per_feat:
            kept_per_feat[j].append(t)
    return {int(j): sorted(set(map(float, edges))) for j, edges in kept_per_feat.items()}


# ---------------------------------------------------------------------------
# LightGBM reference-ensemble cache
#
# Everything below is self-contained. Helper names stay private (leading
# underscore) to keep the public threshold API narrow.
# ---------------------------------------------------------------------------


def _reference_check_array_allow_nan(X) -> np.ndarray:
    """Coerce feature arrays while preserving NaNs for LightGBM."""
    if hasattr(X, "to_numpy"):
        X_arr = np.asarray(X.to_numpy(), dtype=float)
    else:
        X_arr = np.asarray(X, dtype=float)
    if X_arr.ndim != 2:
        raise ValueError(f"Expected a 2D array, got shape {X_arr.shape}")
    return X_arr


def _reference_check_X_y_allow_nan(X, y) -> tuple[np.ndarray, np.ndarray]:
    """Coerce X/y and verify matching row counts."""
    X_arr = _reference_check_array_allow_nan(X)
    y_arr = np.asarray(y)
    if y_arr.ndim != 1:
        y_arr = np.asarray(y_arr).reshape(-1)
    if X_arr.shape[0] != y_arr.shape[0]:
        raise ValueError(
            f"X and y have inconsistent lengths: {X_arr.shape[0]} != {y_arr.shape[0]}"
        )
    return X_arr, np.asarray(y_arr)


def _reference_encode_target(y_arr: np.ndarray) -> tuple[np.ndarray, list]:
    """Encode arbitrary class labels to LightGBM's 0-based class IDs."""
    y = np.asarray(y_arr)
    unique = np.unique(y)
    if unique.size < 2:
        raise ValueError(
            f"fit_thresholds expects at least 2 classes; received {unique.size}"
        )
    # Match the estimators' np.unique encoding. Stringifying encoded labels
    # would put class 10 before class 2 and misalign multiclass LightGBM scores.
    ordered = unique.tolist()
    mapping = {value: idx for idx, value in enumerate(ordered)}
    encoded = np.array([mapping[value] for value in y], dtype=np.int32)
    return encoded, ordered


def _reference_encode_target_with_order(y_arr: np.ndarray, ordered_classes: list) -> np.ndarray:
    """Encode validation labels with the training class order."""
    mapping = {value: idx for idx, value in enumerate(ordered_classes)}
    try:
        return np.array([mapping[value] for value in np.asarray(y_arr)], dtype=np.int32)
    except KeyError as exc:
        raise ValueError(
            "Validation labels must be a subset of the training labels used "
            "to fit LightGBM."
        ) from exc


def _reference_collect_threshold_scores(
    tree_node: Dict, feature_scores: List[Dict[float, float]]
) -> None:
    """Accumulate LightGBM split gain per feature threshold."""
    if "split_feature" not in tree_node:
        return
    feature_idx = int(tree_node["split_feature"])
    threshold = float(tree_node["threshold"])
    split_gain = float(tree_node.get("split_gain", 1.0))
    if np.isfinite(threshold):
        score = max(split_gain, 1e-12)
        feature_scores[feature_idx][threshold] = (
            feature_scores[feature_idx].get(threshold, 0.0) + score
        )
    left = tree_node.get("left_child")
    right = tree_node.get("right_child")
    if isinstance(left, dict):
        _reference_collect_threshold_scores(left, feature_scores)
    if isinstance(right, dict):
        _reference_collect_threshold_scores(right, feature_scores)


def _reference_merge_threshold_score_maps(
    dst: List[Dict[float, float]], src: List[Dict[float, float]]
) -> None:
    """Merge per-run threshold score maps."""
    for j, src_map in enumerate(src):
        dst_map = dst[j]
        for threshold, score in src_map.items():
            dst_map[threshold] = dst_map.get(threshold, 0.0) + float(score)


def _reference_nearest_boundary_index(edges: np.ndarray, threshold: float) -> Optional[int]:
    """Find the cached edge closest to a LightGBM split threshold."""
    if edges.size == 0 or not np.isfinite(threshold):
        return None
    pos = int(np.searchsorted(edges, threshold, side="left"))
    if pos <= 0:
        return 0
    if pos >= edges.size:
        return int(edges.size - 1)
    prev_edge = float(edges[pos - 1])
    next_edge = float(edges[pos])
    if abs(threshold - prev_edge) <= abs(next_edge - threshold):
        return pos - 1
    return pos


def _reference_accumulate_boundary_priors(
    tree_node: Dict,
    bin_edges_per_feature: List[np.ndarray],
    boundary_gain: np.ndarray,
    boundary_cover: np.ndarray,
    boundary_value_jump: np.ndarray,
) -> tuple[float, float]:
    """Accumulate gain, cover, and value-jump diagnostics per boundary."""
    if "split_feature" not in tree_node:
        leaf_value = float(tree_node.get("leaf_value", 0.0))
        leaf_count = float(tree_node.get("leaf_count", 0.0))
        if leaf_count <= 0.0:
            leaf_count = 1.0
        return leaf_value * leaf_count, leaf_count

    left = tree_node.get("left_child")
    right = tree_node.get("right_child")
    left_sum = left_count = 0.0
    right_sum = right_count = 0.0
    if isinstance(left, dict):
        left_sum, left_count = _reference_accumulate_boundary_priors(
            left, bin_edges_per_feature, boundary_gain, boundary_cover, boundary_value_jump
        )
    if isinstance(right, dict):
        right_sum, right_count = _reference_accumulate_boundary_priors(
            right, bin_edges_per_feature, boundary_gain, boundary_cover, boundary_value_jump
        )

    feature_idx = int(tree_node["split_feature"])
    threshold = float(tree_node.get("threshold", np.nan))
    boundary_idx = _reference_nearest_boundary_index(
        bin_edges_per_feature[feature_idx], threshold
    )
    if boundary_idx is not None:
        split_gain = max(float(tree_node.get("split_gain", 0.0)), 0.0)
        split_cover = float(tree_node.get("internal_count", left_count + right_count))
        if split_cover <= 0.0:
            split_cover = left_count + right_count
        left_mean = left_sum / left_count if left_count > 0.0 else 0.0
        right_mean = right_sum / right_count if right_count > 0.0 else 0.0
        boundary_gain[feature_idx, boundary_idx] += split_gain
        boundary_cover[feature_idx, boundary_idx] += split_cover
        boundary_value_jump[feature_idx, boundary_idx] += split_cover * abs(
            left_mean - right_mean
        )

    return left_sum + right_sum, left_count + right_count


def _reference_extract_raw_margin_predictions(
    model: LGBMClassifier, X: np.ndarray
) -> np.ndarray:
    """Return raw LightGBM margins for reference-ensemble shape sorting."""
    booster = getattr(model, "booster_", None)
    if booster is not None:
        raw = np.asarray(booster.predict(X, raw_score=True), dtype=np.float64)
    else:
        raw = np.asarray(model.predict(X, raw_score=True), dtype=np.float64)
    if raw.ndim == 2:
        if raw.shape[1] == 1:
            return np.ascontiguousarray(raw[:, 0], dtype=np.float64)
        return np.ascontiguousarray(raw, dtype=np.float64)
    return np.ascontiguousarray(raw.reshape(-1), dtype=np.float64)


def _reference_compute_bin_representatives(
    X_work: np.ndarray, bin_edges_per_feature: List[np.ndarray]
) -> List[np.ndarray]:
    """Compute representative numeric values for each LightGBM bin."""
    representatives: List[np.ndarray] = []
    n_features = X_work.shape[1]
    for j in range(n_features):
        edges = np.asarray(bin_edges_per_feature[j], dtype=np.float64)
        feature_values = X_work[:, j]
        n_bins = int(edges.size + 1)
        reps = np.zeros(n_bins, dtype=np.float64)
        bins = (
            np.digitize(feature_values, edges, right=False).astype(np.int32)
            if edges.size
            else np.zeros(feature_values.shape[0], dtype=np.int32)
        )
        feature_min = float(np.min(feature_values))
        feature_max = float(np.max(feature_values))
        for b in range(n_bins):
            mask = bins == b
            if np.any(mask):
                reps[b] = float(np.mean(feature_values[mask]))
                continue
            if n_bins == 1:
                reps[b] = feature_min
            elif b == 0:
                reps[b] = 0.5 * (feature_min + float(edges[0]))
            elif b == n_bins - 1:
                reps[b] = 0.5 * (float(edges[-1]) + feature_max)
            else:
                reps[b] = 0.5 * (float(edges[b - 1]) + float(edges[b]))
        representatives.append(reps)
    return representatives


@dataclass
class ThresholdCache:
    """Bundle of outputs from `fit_thresholds`.

    - `thresholds`: feature index -> sorted candidate thresholds (what the
      estimators split on).
    - `logits`: `(n, n_classes)` ndarray or None: the LightGBM raw scores
      of the training rows. ShapeSPLINTER orders bins by them.
    - `boundary_gain`, `boundary_cover`, `boundary_value_jump`:
      `(n_features, max_edges)` tensors; zero-padded where a feature has
      fewer than `max_edges` edges. The search does not use them; they are
      stored on the estimator for inspection, and `boundary_gain` ranks the
      thresholds for `threshold_budget`.
    - `bin_representatives`: per-bin mean values, keyed by feature index.
    - `fill_values`: per-feature values used by the LightGBM imputation.
      The public tree estimators require finite inputs at fit and predict.
    - `n_bins_per_feature`: diagnostic aid (edges+1 per feature).
    """

    thresholds: Dict[int, List[float]]
    logits: Optional[np.ndarray]
    boundary_gain: Optional[np.ndarray]
    boundary_cover: Optional[np.ndarray]
    boundary_value_jump: Optional[np.ndarray]
    bin_representatives: Optional[Dict[int, np.ndarray]]
    fill_values: np.ndarray
    n_bins_per_feature: np.ndarray


def save_threshold_cache(
    cache: ThresholdCache,
    path,
    *,
    metadata: Optional[dict] = None,
) -> None:
    """Save a threshold cache from `fit_thresholds` to one ``.npz`` file.

    `metadata` (a JSON-serializable dict) is stored with it. The single-tree
    estimators accept the saved path as ``precomputed_thresholds``; read it
    back with `load_threshold_cache`.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n_features = int(len(cache.thresholds))

    edge_lengths = np.zeros(n_features, dtype=np.int64)
    edge_parts = []
    for j in range(n_features):
        arr = np.asarray(cache.thresholds.get(j, []), dtype=np.float64).ravel()
        edge_lengths[j] = arr.size
        edge_parts.append(arr)
    edge_flat = (
        np.concatenate(edge_parts).astype(np.float64, copy=False)
        if edge_parts else np.array([], dtype=np.float64)
    )

    payload = {
        "binner_bin_edges_lengths": edge_lengths,
        "binner_bin_edges_flat": edge_flat,
        "binner_fill_values": np.asarray(cache.fill_values, dtype=np.float64),
        "n_bins_per_feature": np.asarray(cache.n_bins_per_feature, dtype=np.int64),
        "metadata_json": np.array(
            json.dumps(metadata or {}, sort_keys=True), dtype=np.str_
        ),
    }
    if cache.logits is not None:
        payload["logits"] = np.asarray(cache.logits, dtype=np.float64)
    if cache.boundary_gain is not None:
        payload["boundary_gain"] = np.asarray(cache.boundary_gain, dtype=np.float64)
    if cache.boundary_cover is not None:
        payload["boundary_cover"] = np.asarray(cache.boundary_cover, dtype=np.float64)
    if cache.boundary_value_jump is not None:
        payload["boundary_value_jump"] = np.asarray(
            cache.boundary_value_jump, dtype=np.float64
        )
    if cache.bin_representatives is not None:
        rep_lengths = np.zeros(n_features, dtype=np.int64)
        rep_parts = []
        for j in range(n_features):
            arr = np.asarray(cache.bin_representatives.get(j, []), dtype=np.float64).ravel()
            rep_lengths[j] = arr.size
            rep_parts.append(arr)
        payload["binner_bin_representatives_lengths"] = rep_lengths
        payload["binner_bin_representatives_flat"] = (
            np.concatenate(rep_parts).astype(np.float64, copy=False)
            if rep_parts else np.array([], dtype=np.float64)
        )
    # A reader must see either the old complete cache or the new complete
    # cache, never a partly written ZIP. Concurrent writers must use their own lock.
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".npz", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            np.savez(stream, **payload)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _legacy_key(data, key):
    """`data[key]`, else the array under its pre-release name (``teacher_`` +
    key; ``teacher_logit`` for ``logits``), else None."""
    if key in data:
        return data[key]
    legacy = "teacher_logit" if key == "logits" else "teacher_" + key
    return data.get(legacy)


def load_threshold_cache(path) -> ThresholdCache:
    """Load a cache written by :func:`save_threshold_cache`. Cache files
    written by older versions also load."""
    with np.load(path, allow_pickle=False) as archive:
        data = {name: archive[name] for name in archive.files}
    lengths = np.asarray(data["binner_bin_edges_lengths"], dtype=np.int64)
    flat = np.asarray(data["binner_bin_edges_flat"], dtype=np.float64)
    if (lengths.ndim != 1 or flat.ndim != 1 or np.any(lengths < 0)
            or int(lengths.sum()) != flat.size):
        raise ValueError("LightGBM cache has inconsistent threshold lengths.")
    thresholds: Dict[int, List[float]] = {}
    offset = 0
    for j, length in enumerate(lengths):
        length = int(length)
        thresholds[j] = flat[offset:offset + length].astype(np.float64).tolist()
        offset += length

    bin_representatives = None
    if ("binner_bin_representatives_lengths" in data
            and "binner_bin_representatives_flat" in data):
        rep_lengths = np.asarray(
            data["binner_bin_representatives_lengths"], dtype=np.int64
        )
        rep_flat = np.asarray(
            data["binner_bin_representatives_flat"], dtype=np.float64
        )
        bin_representatives = {}
        offset = 0
        for j, length in enumerate(rep_lengths):
            length = int(length)
            bin_representatives[j] = rep_flat[offset:offset + length].copy()
            offset += length

    return ThresholdCache(
        thresholds=thresholds,
        logits=_legacy_key(data, "logits"),
        boundary_gain=_legacy_key(data, "boundary_gain"),
        boundary_cover=_legacy_key(data, "boundary_cover"),
        boundary_value_jump=_legacy_key(data, "boundary_value_jump"),
        bin_representatives=bin_representatives,
        fill_values=(
            data["binner_fill_values"]
            if "binner_fill_values" in data
            else np.zeros(len(thresholds), dtype=np.float64)
        ),
        n_bins_per_feature=(
            data["n_bins_per_feature"]
            if "n_bins_per_feature" in data
            else lengths + 1
        ),
    )


def _normalize_thresholds(thresholds, n_features=None):
    """Validate and normalize an explicit feature-to-threshold mapping."""
    if thresholds is None or not hasattr(thresholds, "items"):
        raise ValueError("A LightGBM cache must contain a threshold mapping.")
    out = {}
    for key, values in thresholds.items():
        try:
            feature = int(key)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"Invalid threshold feature index {key!r}.") from exc
        if isinstance(key, (bool, np.bool_)) or str(feature) != str(key):
            raise ValueError(f"Invalid threshold feature index {key!r}.")
        if feature < 0 or (n_features is not None and feature >= n_features):
            raise ValueError("Threshold feature index is outside the feature range.")
        edges = np.asarray(values, dtype=np.float64).ravel()
        if not np.isfinite(edges).all():
            raise ValueError("Thresholds must be finite.")
        out[feature] = np.unique(edges).tolist()
    if n_features is not None:
        for feature in range(n_features):
            out.setdefault(feature, [])
    return out


def _normalize_logits(logits, n_rows, n_classes):
    """Validate reference-ensemble logits as an ``(n_rows, n_classes)`` array.

    A 2-D input is used as is. For binary data a 1-D input ``l`` is the
    class-1 minus class-0 margin; it becomes ``[-l/2, l/2]``, the layout of
    the internal LightGBM logits ``[-raw, raw]`` (so ``l = 2 * raw``).
    """
    if logits is None:
        return None
    logits = np.asarray(logits, dtype=np.float64)
    if logits.ndim == 1 and n_classes == 2:
        logits = np.column_stack([-logits / 2, logits / 2])
    if logits.ndim != 2 or logits.shape != (n_rows, n_classes):
        raise ValueError(f"The LightGBM scores have shape {logits.shape}, but the training "
                         f"data needs ({n_rows}, {n_classes}): the scores or the threshold "
                         "cache were computed for a different number of rows. A 1-D array "
                         "is accepted only for binary classification.")
    if not np.isfinite(logits).all():
        raise ValueError("The LightGBM scores must be finite.")
    return np.ascontiguousarray(logits)


def _coerce_cache(source, n_rows, n_features, n_classes, logits_override=None):
    """Read the supported cache forms once for all single-tree estimators."""
    if source is None:
        return None
    fields = ("thresholds", "logits", "boundary_gain", "boundary_cover",
              "boundary_value_jump", "bin_representatives", "fill_values", "n_bins_per_feature")
    if isinstance(source, (str, os.PathLike)):
        source = load_threshold_cache(source)
    if isinstance(source, dict):
        if "thresholds" not in source or not any(
            key in source for key in ("boundary_gain", "teacher_boundary_gain", "logits",
                                      "teacher_logit", "n_bins_per_feature",
                                      "binner_bin_edges_lengths")
        ):
            raise ValueError("precomputed_thresholds must be a LightGBM reference-ensemble "
                             "cache, not a bare threshold dictionary.")
        # Pre-release cache dictionaries and NPZ files name the logits
        # "teacher_logit" and prefix the boundary arrays with "teacher_".
        result = {field: _legacy_key(source, field) for field in fields}
    elif hasattr(source, "thresholds"):
        result = {field: getattr(source, field, None) for field in fields}
    else:
        raise ValueError("precomputed_thresholds must be a LightGBM reference-ensemble "
                         "cache, cache dictionary, or NPZ path.")
    result["thresholds"] = _normalize_thresholds(result["thresholds"], n_features)
    result["logits"] = _normalize_logits(
        logits_override if logits_override is not None else result["logits"],
        n_rows, n_classes,
    )
    for field in ("fill_values", "n_bins_per_feature"):
        if result[field] is not None:
            value = np.asarray(result[field])
            if value.shape != (n_features,):
                raise ValueError(f"Cached {field} must contain one value per feature.")
            result[field] = value
    return result


def _prepare(estimator, X, y):
    """Prepare the same LightGBM cache and threshold budget for every wrapper."""
    estimator.threshold_budget_selected_ = None
    estimator.threshold_budget_total_ = None
    estimator.threshold_budget_gain_source_ = None
    cached = _coerce_cache(estimator.precomputed_thresholds, len(X), X.shape[1],
                                 estimator.n_classes_, estimator.precomputed_logits)
    if cached is None:
        started = time.perf_counter()
        source = fit_thresholds(
            X, y, max_bins=int(estimator.max_bins),
            n_estimators=int(estimator.lgbm_n_estimators),
            min_samples_leaf=int(estimator.lgbm_min_samples_leaf),
            random_state=estimator.lgbm_random_state, num_threads=1,
        )
        estimator.lgbm_time_ = time.perf_counter() - started
        cached = _coerce_cache(source, len(X), X.shape[1], estimator.n_classes_,
                                      estimator.precomputed_logits)
    else:
        estimator.lgbm_time_ = 0.0
    estimator._logits = cached["logits"]
    for field in ("boundary_gain", "boundary_cover", "boundary_value_jump",
                  "bin_representatives", "n_bins_per_feature"):
        setattr(estimator, f"{field}_", cached[field])
    estimator.bin_fill_values_ = cached["fill_values"]
    thresholds = cached["thresholds"]
    budget = estimator.threshold_budget
    is_full = budget is None or (isinstance(budget, str) and budget.strip().lower() in
                                 ("full", "all", "none"))
    if not is_full:
        ranked = _gain_per_threshold(thresholds, estimator.boundary_gain_)
        if ranked is None:
            raise ValueError("threshold_budget requires LightGBM split gains.")
        thresholds = _apply_budget(thresholds, ranked, _parse_budget(budget)(ranked))
        estimator.threshold_budget_selected_ = sum(len(edges) for edges in thresholds.values())
        estimator.threshold_budget_total_ = len(ranked)
        estimator.threshold_budget_gain_source_ = "lightgbm_split_gain"
    estimator.lgbm_thresholds_ = thresholds
    return thresholds


# Reference-ensemble arrays read only during fit (`_prepare`,
# `_RESPLINTER._derive_thresholds`).
_FIT_ONLY_ATTRIBUTES = ("boundary_gain_", "boundary_cover_", "boundary_value_jump_",
                        "bin_representatives_", "bin_fill_values_", "n_bins_per_feature_",
                        "_logits_full")


def _discard_fit_only_arrays(estimator):
    """Delete the reference-ensemble arrays a fitted estimator no longer
    needs, keeping pickles small. ``lgbm_thresholds_`` stays."""
    for name in _FIT_ONLY_ATTRIBUTES:
        if hasattr(estimator, name):
            delattr(estimator, name)


def fit_thresholds(
    X,
    y,
    *,
    X_val=None,
    y_val=None,
    max_bins: int = 1024,
    n_estimators: int = 512,
    learning_rate: float = 0.05,
    num_leaves: int = 63,
    min_samples_leaf: int = 20,
    min_data_in_bin: int = 1,
    max_depth: int = -1,
    lambda_l2: float = 0.0,
    early_stopping_rounds: int = 100,
    threshold_dedup_eps: float = 1e-9,
    ensemble_runs: int = 1,
    feature_fraction: float = 1.0,
    bagging_fraction: float = 1.0,
    bagging_freq: int = 0,
    ensemble_feature_fraction: float = 0.8,
    ensemble_bagging_fraction: float = 0.8,
    ensemble_bagging_freq: int = 1,
    random_state: Optional[int] = 42,
    collect_logits: bool = True,
    collect_boundary_priors: bool = True,
    collect_bin_representatives: bool = True,
    num_threads: int = 1,
    verbose: int = -1,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> ThresholdCache:
    """Train the LightGBM reference ensemble and return SPLINTER's threshold cache.

    Trains one or more LightGBM boosters with `max_bin=max_bins`
    (default 1024) and extracts per-feature split thresholds ranked by
    accumulated split-gain, capped at `max_bins - 1` edges per feature.
    Features that LightGBM never splits on get an empty threshold list, and
    the estimators never split on them (there is no fallback to quantiles
    or midpoints).

    The returned `ThresholdCache` holds the thresholds, the LightGBM scores
    of the training rows, the per-threshold statistics and the bin
    representatives, so one call provides everything the estimators need.

    Use `precomputed_thresholds=cache` on the estimators to reuse the same
    reference ensemble across a depth or hyperparameter sweep.
    """
    if max_bins < 2:
        raise ValueError("max_bins must be at least 2")
    if min_samples_leaf < 1:
        raise ValueError("min_samples_leaf must be at least 1")
    if ensemble_runs < 1:
        raise ValueError("ensemble_runs must be at least 1")
    if not (0.0 < ensemble_feature_fraction <= 1.0):
        raise ValueError("ensemble_feature_fraction must be in (0, 1]")
    if not (0.0 < ensemble_bagging_fraction <= 1.0):
        raise ValueError("ensemble_bagging_fraction must be in (0, 1]")
    if ensemble_bagging_freq < 0:
        raise ValueError("ensemble_bagging_freq must be >= 0")
    if threshold_dedup_eps < 0:
        raise ValueError("threshold_dedup_eps must be >= 0")
    min_data_in_leaf = int(min_samples_leaf)
    if lambda_l2 < 0:
        raise ValueError("lambda_l2 must be >= 0")
    if early_stopping_rounds < 0:
        raise ValueError("early_stopping_rounds must be >= 0")

    import lightgbm as lgb
    from lightgbm import LGBMClassifier

    X_arr, y_arr = _reference_check_X_y_allow_nan(X, y)
    y_bin, ordered_classes = _reference_encode_target(y_arr)
    n_classes = int(len(ordered_classes))
    n_features = X_arr.shape[1]

    X_val_work: Optional[np.ndarray] = None
    y_val_bin: Optional[np.ndarray] = None
    if X_val is not None or y_val is not None:
        if X_val is None or y_val is None:
            raise ValueError("X_val and y_val must both be provided for early stopping")
        X_val_arr, y_val_arr = _reference_check_X_y_allow_nan(X_val, y_val)
        if X_val_arr.shape[1] != n_features:
            raise ValueError(
                f"validation feature count mismatch: train has {n_features}, "
                f"val has {X_val_arr.shape[1]}"
            )
        X_val_work = X_val_arr.copy()
        y_val_bin = _reference_encode_target_with_order(y_val_arr, ordered_classes)

    X_work = X_arr.copy()
    fill_values = np.zeros(n_features, dtype=float)
    for j in range(n_features):
        col = X_work[:, j]
        if np.isnan(col).any():
            finite = col[~np.isnan(col)]
            fill = float(np.median(finite)) if finite.size > 0 else 0.0
            fill_values[j] = fill
            col[np.isnan(col)] = fill
            X_work[:, j] = col
        else:
            fill_values[j] = 0.0
    if X_val_work is not None:
        for j in range(n_features):
            col = X_val_work[:, j]
            if np.isnan(col).any():
                col[np.isnan(col)] = fill_values[j]
                X_val_work[:, j] = col

    def _progress(message: str) -> None:
        if progress_callback is not None:
            progress_callback(str(message))

    def _fit_one(
        run_seed: Optional[int],
        run_feature_fraction: float,
        run_bagging_fraction: float,
        run_bagging_freq: int,
    ) -> LGBMClassifier:
        subsample_freq = run_bagging_freq if run_bagging_fraction < 0.999 else 0
        params = {
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "max_bin": int(max_bins),
            "num_leaves": num_leaves,
            "max_depth": max_depth,
            "min_child_samples": min_data_in_leaf,
            "min_data_in_bin": min_data_in_bin,
            "reg_lambda": lambda_l2,
            "colsample_bytree": run_feature_fraction,
            "subsample": run_bagging_fraction,
            "subsample_freq": subsample_freq,
            "random_state": run_seed,
            "n_jobs": num_threads,
            "deterministic": True,
            "force_col_wise": True,
            "verbose": verbose,
        }
        if n_classes == 2:
            params["objective"] = "binary"
        else:
            params["objective"] = "multiclass"
            params["num_class"] = int(n_classes)
        model = LGBMClassifier(**params)
        fit_kwargs = {}
        if X_val_work is not None and y_val_bin is not None and early_stopping_rounds > 0:
            fit_kwargs = {
                "eval_set": [(X_val_work, y_val_bin)],
                "eval_metric": "binary_logloss" if n_classes == 2 else "multi_logloss",
                "callbacks": [
                    lgb.early_stopping(
                        stopping_rounds=int(early_stopping_rounds), verbose=False
                    )
                ],
            }
        model.fit(X_work, y_bin, **fit_kwargs)
        return model

    _progress(
        f"lightgbm:start rows={X_work.shape[0]} features={n_features} "
        f"max_bins={int(max_bins)} ensemble_runs={int(ensemble_runs)}"
    )

    feature_scores: List[Dict[float, float]] = [dict() for _ in range(n_features)]
    margin_sum: Optional[np.ndarray]
    if collect_logits:
        if n_classes == 2:
            margin_sum = np.zeros(X_work.shape[0], dtype=np.float64)
        else:
            margin_sum = np.zeros((X_work.shape[0], n_classes), dtype=np.float64)
    else:
        margin_sum = None
    margin_runs = 0
    models: Optional[List[LGBMClassifier]] = (
        [] if collect_boundary_priors else None
    )

    for run_idx in range(int(ensemble_runs)):
        if run_idx == 0:
            run_feature_fraction = float(feature_fraction)
            run_bagging_fraction = float(bagging_fraction)
            run_bagging_freq = int(bagging_freq)
        else:
            run_feature_fraction = min(
                float(feature_fraction), float(ensemble_feature_fraction)
            )
            run_bagging_fraction = min(
                float(bagging_fraction), float(ensemble_bagging_fraction)
            )
            run_bagging_freq = max(int(bagging_freq), int(ensemble_bagging_freq))
        run_seed = None if random_state is None else int(random_state + 1009 * run_idx)
        lgbm = _fit_one(
            run_seed=run_seed,
            run_feature_fraction=run_feature_fraction,
            run_bagging_fraction=run_bagging_fraction,
            run_bagging_freq=run_bagging_freq,
        )

        run_feature_scores: List[Dict[float, float]] = [dict() for _ in range(n_features)]
        dump = lgbm.booster_.dump_model()
        for tree_info in dump.get("tree_info", []):
            tree_structure = tree_info.get("tree_structure")
            if isinstance(tree_structure, dict):
                _reference_collect_threshold_scores(tree_structure, run_feature_scores)
        _reference_merge_threshold_score_maps(feature_scores, run_feature_scores)
        if collect_logits and margin_sum is not None:
            margin_sum += _reference_extract_raw_margin_predictions(lgbm, X_work)
            margin_runs += 1
        if models is not None:
            models.append(lgbm)

    max_edges = max_bins - 1
    bin_edges_per_feature: List[np.ndarray] = []

    for j in range(n_features):
        feature_values = X_work[:, j]
        if np.unique(feature_values).size <= 1:
            edges = np.array([], dtype=float)
        else:
            lo = float(np.min(feature_values))
            hi = float(np.max(feature_values))
            candidates = [
                (thr, score)
                for thr, score in feature_scores[j].items()
                if np.isfinite(thr) and lo < float(thr) < hi
            ]
            candidates.sort(key=lambda item: (-item[1], item[0]))
            chosen: List[float] = []
            for threshold, _ in candidates:
                t = float(threshold)
                if threshold_dedup_eps > 0 and any(
                    abs(prev - t) <= threshold_dedup_eps for prev in chosen
                ):
                    continue
                if threshold_dedup_eps == 0 and chosen and abs(chosen[-1] - t) < 1e-12:
                    continue
                chosen.append(t)
                if len(chosen) >= max_edges:
                    break
            edges = np.array(sorted(chosen), dtype=float)

        bin_edges_per_feature.append(np.sort(edges))

    n_bins_per_feature = np.array(
        [int(edges.size + 1) if edges.size else 1 for edges in bin_edges_per_feature],
        dtype=np.int32,
    )
    _progress(
        f"lightgbm:thresholds_ready "
        f"mean_bins={float(np.mean(n_bins_per_feature)):.2f} "
        f"max_bins_used={int(np.max(n_bins_per_feature)) if n_bins_per_feature.size else 0}"
    )

    boundary_gain: Optional[np.ndarray] = None
    boundary_cover: Optional[np.ndarray] = None
    boundary_value_jump: Optional[np.ndarray] = None
    if collect_boundary_priors and models:
        boundary_gain = np.zeros((n_features, max_edges), dtype=np.float64)
        boundary_cover = np.zeros((n_features, max_edges), dtype=np.float64)
        boundary_value_jump = np.zeros((n_features, max_edges), dtype=np.float64)
        for model in models:
            dump = model.booster_.dump_model()
            for tree_info in dump.get("tree_info", []):
                tree_structure = tree_info.get("tree_structure")
                if isinstance(tree_structure, dict):
                    _reference_accumulate_boundary_priors(
                        tree_structure,
                        bin_edges_per_feature,
                        boundary_gain,
                        boundary_cover,
                        boundary_value_jump,
                    )

    bin_representatives: Optional[Dict[int, np.ndarray]] = None
    if collect_bin_representatives:
        reps_list = _reference_compute_bin_representatives(X_work, bin_edges_per_feature)
        bin_representatives = {j: reps_list[j] for j in range(n_features)}

    # Package every feature, including features with no LightGBM split, so
    # Rust receives an explicit empty threshold list and never falls back to
    # local midpoint generation for a missing key.
    thresholds: Dict[int, List[float]] = {}
    for j, edges in enumerate(bin_edges_per_feature):
        thresholds[int(j)] = [float(v) for v in edges.tolist()]

    # Reference-ensemble logit: average raw margins across runs, reshape binary to (n, 2).
    logits: Optional[np.ndarray] = None
    if (
        collect_logits
        and margin_sum is not None
        and margin_runs > 0
    ):
        raw = margin_sum / float(margin_runs)
        if raw.ndim == 1:
            logits = np.stack([-raw, raw], axis=1)
        else:
            logits = raw
        logits = np.ascontiguousarray(logits, dtype=np.float64)

    _progress("lightgbm:done")

    return ThresholdCache(
        thresholds=thresholds,
        logits=logits,
        boundary_gain=boundary_gain,
        boundary_cover=boundary_cover,
        boundary_value_jump=boundary_value_jump,
        bin_representatives=bin_representatives,
        fill_values=fill_values,
        n_bins_per_feature=n_bins_per_feature,
    )
