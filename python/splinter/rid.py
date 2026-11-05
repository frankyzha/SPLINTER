"""Rashomon Importance Distribution (Donnelly et al., NeurIPS 2023), exact.

Use ``compute_rid(RESPLINTER(...), X, y, B=10)``; it returns a ``RID``.

Recipe:
  1. bootstrap the data B times (`_bootstrap_indices`);
  2. fit a copy of the RESPLINTER on each bootstrap D_b with candidate
     thresholds fixed from the full data (`_RESPLINTER._derive_thresholds`);
  3. per feature j, the exact histogram of the subtractive model reliance
     phi_j over ALL trees of R_b (the trees within (1+theta)*opt_b), scored
     on D_b (`PyRashomonDag.rid_histograms`);
  4. mix the per-bootstrap histograms with weight 1/(B_ok * |R_b|)
     (`_aggregate_rid`).

phi_j is the exact all-donor expectation of the 0-1 loss increase when row
i's value of j is taken from a uniformly random row (ArborEnum's lossless
formula); n^2 * phi_j is an integer (`nums`). When the exact walk cannot
finish (state / time / mask-memory cap), a bootstrap falls back to
`n_fallback` uniform tree draws scored exactly (`PyRashomonDag.tree_phis`),
mode 'sampled'. A walk whose counts disagree with the DAG
(`count_mismatch > 0`), a fit that raised (or panicked in Rust) and an
incomplete reporting set make the bootstrap 'failed'.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import math
import os
import time
from dataclasses import dataclass, field
from fractions import Fraction

import numpy as np
import pandas as pd
from sklearn.base import clone

from .multiway_solver import CANDIDATE_RANKING_POLICY

__all__ = ["compute_rid", "RID"]

# Completeness codes (`fit_info_["incomplete_reason_codes"]`) that fail a
# bootstrap: an enumeration or fill time-out, unfilled frontier leaves, or a
# bound not set from a tree of the set.
_FAIL_REASONS = ("stepb_timeout", "phase2_timeout", "unfilled_prefix_leaves",
                 "upfront_anchor_not_in_dag")
_DEFAULT_CAPS = dict(max_states=5_000_000, max_rid_seconds=None, n_fallback=10_000)


class _ResumeMismatch(ValueError):
    """A saved bootstrap was produced by another configuration (or other
    row indices) than the run that would reuse it."""


def _h(tag: str) -> int:
    """First 4 bytes of sha256(tag) as a (big-endian) int."""
    return int.from_bytes(hashlib.sha256(str(tag).encode()).digest()[:4], "big")


def _bootstrap_indices(n, b, base_seed=0, tag="full") -> np.ndarray:
    """Row indices of bootstrap `b`: n draws with replacement from [0, n)."""
    ss = np.random.SeedSequence([int(base_seed), _h(tag), int(b)])
    return np.random.Generator(np.random.PCG64(ss)).integers(0, int(n), int(n))


def _sha256_array(a) -> str:
    return hashlib.sha256(np.ascontiguousarray(a, dtype=np.int64).tobytes()).hexdigest()


def _thresholds_sha256(thresholds) -> str:
    canon = {str(int(j)): [float(t).hex() for t in sorted(map(float, v))]
             for j, v in sorted(thresholds.items(), key=lambda kv: int(kv[0]))}
    return hashlib.sha256(json.dumps(canon, sort_keys=True).encode()).hexdigest()


def _json_safe(v):
    """JSON-safe, canonical form of a fit kwarg (numpy scalars / arrays to
    Python, dicts with str keys, anything else by repr)."""
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, np.ndarray):
        return [_json_safe(x) for x in v.tolist()]
    if isinstance(v, (list, tuple)):
        return [_json_safe(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _json_safe(x) for k, x in v.items()}
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    return repr(v)


def _config_sha256(fit_kwargs, *, base_seed, tag, n, n_features, max_states,
                  max_rid_seconds, n_fallback, thresholds_sha256) -> str:
    """sha256 of the canonical JSON (sorted keys) of everything a saved
    bootstrap depends on besides its row indices."""
    cfg = {"fit_kwargs": _json_safe(dict(fit_kwargs)), "base_seed": int(base_seed),
           "candidate_ranking_policy": CANDIDATE_RANKING_POLICY,
           "tag": str(tag), "n": int(n), "n_features": int(n_features),
           "max_states": int(max_states),
           "max_rid_seconds": None if max_rid_seconds is None else float(max_rid_seconds),
           "n_fallback": int(n_fallback), "thresholds_sha256": str(thresholds_sha256)}
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()


def _check_resume(meta, *, config_sha, idx, where):
    """Raise _ResumeMismatch unless the saved `meta` has this `config_sha`
    and the sha256 of these row indices `idx`."""
    got = meta.get("config_sha256")
    if got != config_sha:
        raise _ResumeMismatch(f"{where}: the saved bootstrap was computed with different "
                             "settings (estimator parameters, seed, tag, caps or thresholds); "
                             "use another out_dir or delete the saved files")
    want = _sha256_array(idx)
    if meta.get("idx_sha256") != want:
        raise _ResumeMismatch(f"{where}: the saved bootstrap used different rows; "
                             "use another out_dir or delete the saved files")


def _counts_array(counts) -> np.ndarray:
    """int64 array when every count fits, else an object array of ints."""
    counts = [int(c) for c in counts]
    if all(c < 2**63 for c in counts):
        return np.asarray(counts, dtype=np.int64)
    return np.asarray(counts, dtype=object)


@dataclass
class _BootResult:
    b: int
    n: int
    tree_count: int
    mode: str  # 'exact' | 'sampled' | 'failed'
    nums: list
    counts: list
    meta: dict = field(default_factory=dict)


def _default_sample_seed(base_seed, tag, b) -> int:
    ss = np.random.SeedSequence([int(base_seed), _h(tag), int(b), 1])
    return int(ss.generate_state(1, dtype=np.uint64)[0] >> np.uint64(1))


# Resource settings: they change speed and memory, not the set of trees.
_RESOURCE_SETTINGS = frozenset({
    "n_threads", "completion_batches", "completion_batch_threads",
    "lower_bound_memo_size", "depth1_memo_size", "interval_cache_size"})
# Search settings added after the checkpoint format: hashed only when changed
# from their default, so checkpoints written at the defaults stay valid.
_HASHED_IF_CHANGED = frozenset({
    "interval_lower_bound", "interval_min_arity", "screen_time_cap"})


def _fit_settings(estimator):
    """`estimator.get_params()` plus the fixed settings its class hides
    (the base-class parameters missing from its signature): everything the
    set of trees depends on."""
    from .resplinter import _RESPLINTER
    params = estimator.get_params(deep=False)
    signature = inspect.signature(_RESPLINTER.__init__).parameters
    for name, parameter in signature.items():
        if name == "self" or name in params or name in _RESOURCE_SETTINGS:
            continue
        value = getattr(estimator, name)
        if name in _HASHED_IF_CHANGED and value == parameter.default:
            continue
        params[name] = value
    return params


def _rid_one_bootstrap(estimator, X, y, b, thresholds, *, base_seed=0, tag="full",
                       max_states=_DEFAULT_CAPS["max_states"],
                       max_rid_seconds=_DEFAULT_CAPS["max_rid_seconds"],
                       n_fallback=_DEFAULT_CAPS["n_fallback"], sample_seed=None,
                       max_mask_bytes=None, indices=None) -> _BootResult:
    """Fit a clone of `estimator` with fixed_thresholds=thresholds on bootstrap
    `b` of (X, y) (or on the rows `indices` when given; `b` may then be -1)
    and return the exact per-feature N_j histograms over its reporting set
    (or the sampled fallback when the walk hit a cap; mode 'failed' when the
    fit raised or panicked, the walk's counts disagree with the DAG, or the
    set is incomplete: an enumeration or recursive-fill time-out, or
    unfilled frontier leaves). `max_mask_bytes` caps the walk's interned row masks (None =
    the binding's 2 GiB default)."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    n, p = X.shape
    idx = (_bootstrap_indices(n, b, base_seed=base_seed, tag=tag) if indices is None
           else np.asarray(indices, dtype=np.int64))
    thr_sha = _thresholds_sha256(thresholds)
    meta = {"idx_sha256": _sha256_array(idx),
            "candidate_ranking_policy": CANDIDATE_RANKING_POLICY,
            "thresholds_sha256": thr_sha,
            "config_sha256": _config_sha256(
                _fit_settings(estimator), base_seed=base_seed, tag=tag, n=n, n_features=p,
                max_states=max_states, max_rid_seconds=max_rid_seconds,
                n_fallback=n_fallback, thresholds_sha256=thr_sha),
            "max_mask_bytes": None if max_mask_bytes is None else int(max_mask_bytes),
            "opt": None, "bound": None, "tree_count_at_target": None,
            "fit_s": None, "rid_s": None, "n_states": None, "count_mismatch": None,
            "mask_bytes_peak": None, "reasons": "", "reason_text": ""}

    def failed(reason, text=None):
        # `reasons` keeps the raw code (completeness codes, error:/panic:
        # messages); `reason_text` is the plain-sentence version.
        meta["reasons"] = reason
        meta["reason_text"] = reason if text is None else text
        return _BootResult(b=int(b), n=int(n), tree_count=0, mode="failed",
                          nums=[np.zeros(0, dtype=np.int64) for _ in range(p)],
                          counts=[np.zeros(0, dtype=np.int64) for _ in range(p)],
                          meta=meta)

    model = None
    try:
        # `thresholds` already reflect any threshold_budget (applied when
        # they were derived), and RESPLINTER rejects the two together.
        model = clone(estimator).set_params(threshold_budget=None,
                                            fixed_thresholds=thresholds)
        t0 = time.perf_counter()
        model.fit(X[idx], y[idx])
        meta["fit_s"] = time.perf_counter() - t0
        info = model.fit_info_
        reasons = str(info.get("incomplete_reason_codes") or "")
        text = " ".join(model.rashomon_incomplete_reason_ or [])
        meta["reasons"] = reasons
        meta["reason_text"] = text
        meta["opt"] = float(model.opt_)
        meta["bound"] = float(model.theta_bound_)
        tcat = int(model.tree_count_at_target_)
        meta["tree_count_at_target"] = tcat
        if bool(info.get("enumeration_timed_out")):
            return failed(reasons or "phase2_timed_out",
                          text or "The enumeration of the set reached its time limit.")
        hit = [r for r in _FAIL_REASONS if r in reasons]
        if hit:
            return failed(reasons, text)
        if tcat <= 0:
            return failed(reasons or "empty_rashomon_set", text or "The set is empty.")

        dag = model.dag_
        t0 = time.perf_counter()
        res = dag.rid_histograms(max_states=int(max_states), max_seconds=max_rid_seconds,
                                 max_mask_bytes=max_mask_bytes)
        meta["n_states"] = int(res["n_states"])
        meta["count_mismatch"] = int(res["count_mismatch"])
        meta["mask_bytes_peak"] = int(res["mask_bytes_peak"])
        if int(res["count_mismatch"]) > 0:
            meta["rid_s"] = time.perf_counter() - t0
            return failed(f"count_mismatch={int(res['count_mismatch'])}",
                          "The importance walk's counts disagree with the set.")
        if res["complete"]:
            nums = [np.asarray(a, dtype=np.int64) for a in res["nums"]]
            counts = [_counts_array(c) for c in res["counts"]]
            mode = "exact"
        else:
            seed = (_default_sample_seed(base_seed, tag, b) if sample_seed is None
                    else int(sample_seed))
            idxs = dag.sample_indices(int(n_fallback), seed=seed, at_target=True)
            phis = np.asarray(dag.tree_phis(idxs), dtype=np.int64).reshape(len(idxs), p)
            nums, counts = [], []
            for j in range(p):
                v, c = np.unique(phis[:, j], return_counts=True)
                nums.append(v.astype(np.int64))
                counts.append(c.astype(np.int64))
            mode = "sampled"
            meta["sample_seed"] = seed
        meta["rid_s"] = time.perf_counter() - t0
        return _BootResult(b=int(b), n=int(n), tree_count=tcat, mode=mode,
                          nums=nums, counts=counts, meta=meta)
    except BaseException as e:  # noqa: BLE001 — a failed fit is a failed bootstrap
        # pyo3's PanicException (a Rust panic) derives from BaseException and
        # is not importable from Python; recognise it by name.
        if isinstance(e, (KeyboardInterrupt, SystemExit, GeneratorExit)):
            raise
        if type(e).__name__ == "PanicException":
            return failed(f"panic:{e}")
        return failed(f"error:{type(e).__name__}:{e}")
    finally:
        del model


@dataclass
class RID:
    """Rashomon Importance Distribution, returned by ``compute_rid``: for
    each feature, the distribution of its importance over the trees of the
    Rashomon sets of all bootstraps.

    A tree's importance for a feature is its model reliance: the expected
    increase of the tree's training error when each row's value of that
    feature is replaced by the value of a uniformly random row. For feature
    j, `values[j]` holds the distinct importances (sorted) and `weights[j]`
    their probabilities (floats summing to 1); `nums[j]` holds the same
    importances times the squared number of rows, which are exact integers.
    `iweights[j]` are the same weights
    as exact integers over the common denominator `itotal` (so cdf/quantile
    tests are exact). `B` is the number of bootstraps and `B_ok` the number
    that did not fail; `modes[b]` is 'exact', 'sampled' or 'failed'."""
    values: list
    weights: list
    nums: list
    n: int
    B: int
    B_ok: int
    modes: list
    iweights: list = field(default_factory=list)
    itotal: int = 1

    def cdf(self, j, k):
        """Share of the distribution of feature ``j``'s importance that is at
        most ``k`` (the CDF at ``k``).

        Parameters
        ----------
        j : int
            Feature index.
        k : float
            Importance value.

        Returns
        -------
        float
        """
        iw = self.iweights[j]
        s = sum((w for v, w in zip(self.values[j], iw) if v <= k), 0)
        return float(Fraction(s, self.itotal))

    def quantile(self, j, q):
        """The ``q``-quantile of feature ``j``'s importance: the smallest
        value ``v`` with ``cdf(j, v) >= q``.

        Parameters
        ----------
        j : int
            Feature index.
        q : float
            Probability in [0, 1].

        Returns
        -------
        float
            NaN when the feature's distribution is empty.
        """
        if len(self.values[j]) == 0:
            return float("nan")
        need = Fraction(q) * self.itotal
        acc = 0
        for v, w in zip(self.values[j], self.iweights[j]):
            acc += w
            if acc >= need:
                return float(v)
        return float(self.values[j][-1])

    def mean(self, j):
        """Mean importance of feature ``j`` (exact; NaN when its
        distribution is empty).

        Parameters
        ----------
        j : int
            Feature index.

        Returns
        -------
        float
        """
        if len(self.nums[j]) == 0:
            return float("nan")
        s = sum(int(w) * int(v) for v, w in zip(self.nums[j], self.iweights[j]))
        return float(Fraction(s, self.itotal * self.n * self.n))

    def _mass(self, j, pred):
        s = sum((int(w) for v, w in zip(self.nums[j], self.iweights[j]) if pred(int(v))), 0)
        return float(Fraction(s, self.itotal))

    def summary(self) -> pd.DataFrame:
        """Summary table of the importance of every feature.

        One row per feature (``feature_idx``): ``mean``, ``median``, the
        quartiles ``q1`` and ``q3`` and their range ``iqr``, box-plot
        whiskers ``whisker_lo`` / ``whisker_hi`` (q1 - 1.5 iqr and
        q3 + 1.5 iqr), the shares of importance above zero and at zero
        (``p_pos``, ``p_zero``), ``min`` and ``max``, and the bootstrap
        counts ``B``, ``n_bootstraps_used`` (not failed) and
        ``n_bootstraps_sampled`` (scored on a sample of trees).

        Returns
        -------
        pandas.DataFrame
        """
        n_sampled = sum(1 for m in self.modes if m == "sampled")
        rows = []
        for j in range(len(self.values)):
            q1, med, q3 = (self.quantile(j, q) for q in (0.25, 0.5, 0.75))
            iqr = q3 - q1
            empty = len(self.values[j]) == 0
            rows.append({
                "feature_idx": j, "mean": self.mean(j), "median": med,
                "q1": q1, "q3": q3, "iqr": iqr,
                "whisker_lo": q1 - 1.5 * iqr, "whisker_hi": q3 + 1.5 * iqr,
                "p_pos": float("nan") if empty else self._mass(j, lambda v: v > 0),
                "p_zero": float("nan") if empty else self._mass(j, lambda v: v == 0),
                "min": float("nan") if empty else float(self.values[j][0]),
                "max": float("nan") if empty else float(self.values[j][-1]),
                "B": self.B, "n_bootstraps_used": self.B_ok,
                "n_bootstraps_sampled": n_sampled,
            })
        return pd.DataFrame(rows)


def _aggregate_rid(results, n_features) -> RID:
    """Mix bootstrap histograms: an entry of bootstrap b weighs
    count / (sum of b's counts * B_ok), over non-failed bootstraps."""
    results = list(results)
    ok = [r for r in results if r.mode != "failed"]
    B_ok = len(ok)
    ns = {int(r.n) for r in results}
    assert len(ns) <= 1, f"bootstraps disagree on n: {sorted(ns)}"
    n = ns.pop() if ns else 0
    # totals[i][j] = sum of bootstrap i's counts for feature j (= |R_b|, or
    # the number of draws when sampled)
    totals = [[sum(int(x) for x in c) for c in r.counts] for r in ok]
    L = 1
    for row in totals:
        for t in row:
            if t:
                L = L * t // math.gcd(L, t)
    itotal = L * B_ok if B_ok else 1
    values, weights, nums, iweights = [], [], [], []
    for j in range(int(n_features)):
        acc: dict[int, int] = {}
        for r, row in zip(ok, totals):
            if j >= len(r.nums) or not row[j]:
                continue
            scale = L // row[j]
            for v, c in zip(r.nums[j], r.counts[j]):
                acc[int(v)] = acc.get(int(v), 0) + int(c) * scale
        keys = sorted(acc)
        iw = [acc[k] for k in keys]
        nums.append(np.asarray(keys, dtype=np.int64))
        iweights.append(np.asarray(iw, dtype=object))
        values.append(np.asarray(keys, dtype=np.float64) / float(n * n) if n else
                      np.zeros(0, dtype=np.float64))
        weights.append(np.asarray([float(Fraction(w, itotal)) for w in iw], dtype=np.float64))
    return RID(values=values, weights=weights, nums=nums, n=int(n), B=len(results),
               B_ok=B_ok, modes=[r.mode for r in results], iweights=iweights,
               itotal=int(itotal))


# --------------------------------------------------------------- persistence
def _save_boot(out_dir, r: _BootResult):
    base = os.path.join(out_dir, f"boot_{r.b:03d}")
    arrs = {}
    for j, (v, c) in enumerate(zip(r.nums, r.counts)):
        arrs[f"nums_{j}"] = np.asarray(v, dtype=np.int64)
        arrs[f"counts_{j}"] = np.asarray([str(int(x)) for x in c], dtype=str)
    np.savez(base + ".npz", **arrs)
    with open(base + ".json", "w") as f:
        json.dump({"b": r.b, "n": r.n, "tree_count": r.tree_count, "mode": r.mode,
                   "n_features": len(r.nums), "meta": r.meta}, f, indent=1, default=str)


def _load_boot(out_dir, b):
    base = os.path.join(out_dir, f"boot_{b:03d}")
    if not (os.path.exists(base + ".json") and os.path.exists(base + ".npz")):
        return None
    with open(base + ".json") as f:
        js = json.load(f)
    z = np.load(base + ".npz")
    p = int(js["n_features"])
    nums = [z[f"nums_{j}"].astype(np.int64) for j in range(p)]
    counts = [_counts_array(int(x) for x in z[f"counts_{j}"]) for j in range(p)]
    return _BootResult(b=int(js["b"]), n=int(js["n"]), tree_count=int(js["tree_count"]),
                      mode=str(js["mode"]), nums=nums, counts=counts, meta=js["meta"])


def compute_rid(estimator, X, y, *, B=10, random_state=0, out_dir=None):
    """Rashomon Importance Distribution (RID) of each feature.

    Fits a copy of `estimator` on each of `B` bootstrap samples of (X, y),
    all with the same candidate thresholds, and mixes the exact per-tree
    importance distributions of every Rashomon set (Donnelly et al.,
    NeurIPS 2023). The thresholds are the estimator's ``fixed_thresholds``
    when set (``threshold_budget`` must then be None), else derived once
    from LightGBM on the full data.

    Parameters
    ----------
    estimator : RESPLINTER
        An unfitted estimator; its parameters configure every bootstrap fit.
    X : array-like of shape (n_samples, n_features)
    y : array-like of shape (n_samples,)
        Binary class labels (any two values, e.g. strings).
    B : int, default=10
        Number of bootstrap samples.
    random_state : int, default=0
        Seed of the bootstrap samples.
    out_dir : str or path-like, optional
        Directory for per-bootstrap checkpoints (``boot_XXX.npz`` / ``.json``).
        A rerun with the same settings and data reuses them; a checkpoint
        written with other settings raises ValueError.

    Returns
    -------
    RID
        ``summary()`` gives one row per feature.

    Raises
    ------
    ValueError
        For an invalid configuration (e.g. ``fill_mode='recursive'`` without
        ``lookahead_depth < max_depth``, or bad ``fixed_thresholds``),
        before any bootstrap is fit.
    RuntimeError
        When every bootstrap fit raised an error.
    """
    return _compute_rid(estimator, X, y, B=B, base_seed=random_state, out_dir=out_dir)


def _compute_rid(estimator, X, y, *, B=10, base_seed=0, tag="full", out_dir=None,
                 b_range=None, retry_failed=False, **caps):
    """`compute_rid` over bootstraps b in `b_range` (default range(B)). With
    `out_dir`, each bootstrap is written to boot_{b:03d}.npz/.json and
    reused on a rerun (its thresholds hash, `_config_sha256` and row-index
    hash must match, else _ResumeMismatch, a ValueError); `retry_failed`
    re-runs (and overwrites) the saved bootstraps of `b_range` whose mode is
    'failed'. Returns the aggregate when every b in range(B) is available,
    else None. `caps` go to `_rid_one_bootstrap` (max_states,
    max_rid_seconds, n_fallback, sample_seed, max_mask_bytes)."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    n, p = X.shape
    template = clone(estimator)
    template._effective_fill_mode()  # fail fast, before any bootstrap
    if template.fixed_thresholds is not None:
        thresholds = template._checked_fixed_thresholds(p)
    else:
        thresholds = template._derive_thresholds(X, y)
    thr_sha = _thresholds_sha256(thresholds)
    cfg = {k: caps.get(k, v) for k, v in _DEFAULT_CAPS.items()}
    cfg_sha = _config_sha256(_fit_settings(estimator), base_seed=base_seed, tag=tag, n=n,
                             n_features=p, thresholds_sha256=thr_sha, **cfg)

    def load(b):
        r = _load_boot(out_dir, b)
        if r is None:
            return None
        where = f"{out_dir}/boot_{b:03d}"
        if r.meta.get("thresholds_sha256") != thr_sha:
            raise _ResumeMismatch(f"{where}: the saved bootstrap used different thresholds; "
                                 "use another out_dir or delete the saved files")
        _check_resume(r.meta, config_sha=cfg_sha,
                      idx=_bootstrap_indices(n, b, base_seed=base_seed, tag=tag), where=where)
        return r

    todo = list(range(int(B))) if b_range is None else [int(b) for b in b_range]
    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
    have: dict[int, _BootResult] = {}
    for b in todo:
        r = load(b) if out_dir is not None else None
        if r is not None and retry_failed and r.mode == "failed":
            r = None
        if r is None:
            r = _rid_one_bootstrap(estimator, X, y, b, thresholds, base_seed=base_seed,
                                   tag=tag, **caps)
            if out_dir is not None:
                _save_boot(out_dir, r)
        have[b] = r
    for b in range(int(B)):
        if b not in have and out_dir is not None:
            r = load(b)
            if r is not None:
                have[b] = r
    if any(b not in have for b in range(int(B))):
        return None
    results = [have[b] for b in range(int(B))]
    reasons = [str(r.meta.get("reasons", "")) for r in results]
    if results and all(r.mode == "failed" and reason.startswith("error:")
                       for r, reason in zip(results, reasons)):
        raise RuntimeError(f"every RID bootstrap failed; the first: {reasons[0]}")
    return _aggregate_rid(results, X.shape[1])
