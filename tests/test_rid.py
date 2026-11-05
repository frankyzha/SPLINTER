"""Exact RID (Donnelly et al. 2023) on the factored DAG.

Ground truth is the all-donor brute force on tiny instances with explicit
thresholds: for a tree t and feature j,
    N_j(t) = sum_i sum_r [t(x_i with x_ij <- x_rj) != y_i] - n * sum_i [t(x_i) != y_i]
(trees decoded with get_tree, routed with `splinter.tree.predict_encoded`).

  * tree_phis == brute force (integers), every tree at the target;
  * N/n^2 == ((n-1)/n) (e_switch - e_orig) (Fisher et al.), exact Fractions;
  * rid_histograms == Counter(tree_phis) over all trees; sum == count,
    count_mismatch == 0; at_target=False; shape mode; recursive fill;
  * a noise column without thresholds -> {0: |R|};
  * rid_histograms(x, y) on other rows == enumeration on those rows;
  * aggregate_rid vs an explicit replicated sample (LCM weights);
  * bootstrap_indices deterministic, differs across b and tag;
  * compute_rid reproducible (and resumable); fill_parallel invariance;
  * fixed_thresholds = _derive_thresholds(D) reproduces the default fit;
  * every row doubled: same opt / count, nums x4;
  * max_states=1 -> incomplete -> sampled fallback, seeded;
  * resume integrity: changed fit kwarg / tampered idx -> ValueError;
    retry_failed re-runs a saved failed bootstrap;
  * a Rust panic in the fit -> failed bootstrap, compute_rid continues;
  * count_mismatch > 0 -> failed (no sampled fallback); lickety DAGs
    (the only path with minority leaf options) are failed this way;
  * fixed_thresholds must list every feature; logit_weight must be 1.

The fits set hidden settings (fill_parallel, logit_weight), so they use
the base class _RESPLINTER; `run_rid` / `rid_one_bootstrap` below drive the
private `_compute_rid` / `_rid_one_bootstrap`, which take the resume, cap
and bootstrap-range arguments that `compute_rid` does not expose.
"""
import collections
import os
from fractions import Fraction as Fr

import numpy as np
import pytest

import splinter.rid as rid_mod
from splinter import RESPLINTER as PublicRESPLINTER
from splinter.resplinter import _RESPLINTER as RESPLINTER
from splinter.multiway_solver import solve_rashomon
from splinter.resplinter import dict_to_tree
from splinter.rid import (_BootResult as BootResult, _aggregate_rid as aggregate_rid,
                          _bootstrap_indices as bootstrap_indices, compute_rid)
from splinter.tree import predict_encoded


def test_resume_rejects_changed_ranking_policy_without_fitting(monkeypatch):
    settings = dict(base_seed=0, tag="policy-test", n=4, n_features=2,
                    max_states=100, max_rid_seconds=None, n_fallback=10,
                    thresholds_sha256="fixed-thresholds")
    idx = np.array([0, 2, 2, 3], dtype=np.int64)
    fit_kwargs = dict(max_depth=2, max_k=3, reg=0.01)
    current = rid_mod._config_sha256(fit_kwargs, **settings)
    meta = {"config_sha256": current, "idx_sha256": rid_mod._sha256_array(idx)}
    rid_mod._check_resume(meta, config_sha=current, idx=idx, where="fixture")

    monkeypatch.setattr(rid_mod, "CANDIDATE_RANKING_POLICY", "legacy_ranking_policy")
    legacy = rid_mod._config_sha256(fit_kwargs, **settings)
    assert legacy != current
    meta["config_sha256"] = legacy
    with pytest.raises(rid_mod._ResumeMismatch, match="different settings"):
        rid_mod._check_resume(meta, config_sha=current, idx=idx, where="fixture")


# ------------------------------------------------------------ instances
def tiny(seed, n=12, p=3, noise_col=False):
    rng = np.random.default_rng(seed)
    X = rng.integers(0, 4, size=(n, p)).astype(np.float64)
    y = ((X[:, 0] + X[:, 1] >= 3) ^ (rng.random(n) < 0.2)).astype(np.uint8)
    thr = {f: [0.5, 1.5, 2.5] for f in range(p)}
    if noise_col:
        X = np.hstack([X, rng.integers(0, 4, size=(n, 1)).astype(np.float64)])
    return X, y, thr


def solve(X, y, thr, *, depth=2, max_k=3, reg=0.03, theta=1.0, **kw):
    res = solve_rashomon(X, y, 2, reg, depth, max_k, theta, lookahead_depth=kw.pop("lookahead", None),
                         min_samples_leaf=1, beam_width=256,
                         thresholds_per_feature={f: list(t) for f, t in thr.items()},
                         fill_parallel=False, **kw)
    return res["dag"]


INSTANCES = [
    dict(seed=1, max_k=3),
    dict(seed=2, max_k=2),
    dict(seed=5, max_k=3, n=14),
    dict(seed=7, max_k=3, n=10),
]


def brute_nums(tree, X, y):
    """N_j for every j by explicit donor substitution."""
    n, p = X.shape
    y = np.asarray(y).astype(int)
    base = int((predict_encoded(tree, X) != y).sum())
    out = []
    for j in range(p):
        big = np.repeat(X, n, axis=0)           # row (i, r) = x_i ...
        big[:, j] = np.tile(X[:, j], n)         # ... with x_ij <- x_rj
        wrong = int((predict_encoded(tree, big) != np.repeat(y, n)).sum())
        out.append(wrong - n * base)
    return out, base


def repeats_feature_on_path(t, seen=()):
    """True when some root-to-leaf path of tree dict `t` splits on one
    feature twice."""
    if t.get("type") != "node":
        return False
    f = t["feature_index"]
    return f in seen or any(repeats_feature_on_path(c, seen + (f,)) for c in t["children"])


def all_trees(dag, count):
    return [dict_to_tree(dag.get_tree(i)) for i in range(count)]


def hist_dict(h, j):
    return dict(zip(h["nums"][j].tolist(), [int(c) for c in h["counts"][j]]))


# ------------------------------------------------------------ tree phis vs brute force
@pytest.mark.parametrize("inst", INSTANCES)
def test_tree_phis_equal_donor_brute_force(inst):
    X, y, thr = tiny(inst["seed"], n=inst.get("n", 12))
    dag = solve(X, y, thr, max_k=inst["max_k"])
    tc = int(dag.tree_count_at_target)
    assert tc > 1
    phis = np.asarray(dag.tree_phis(list(range(tc))))
    assert phis.shape == (tc, X.shape[1]) and phis.dtype == np.int64
    # the instance exercises a feature re-split on one path (D refined twice)
    assert any(repeats_feature_on_path(dag.get_tree(i)) for i in range(tc))
    n = len(y)
    for i, t in enumerate(all_trees(dag, tc)):
        brute, base = brute_nums(t, X, y)
        assert phis[i].tolist() == brute, (i, phis[i], brute)
        # Fisher et al. identity, exactly
        e_orig = Fr(base, n)
        for j in range(X.shape[1]):
            total_wrong = brute[j] + n * base          # sum over all (i, r)
            e_switch = Fr(total_wrong - base, n * (n - 1))   # r != i (r = i is the original)
            assert Fr(int(phis[i, j]), n * n) == Fr(n - 1, n) * (e_switch - e_orig)


# ------------------------------------------------------------ histograms
def check_hist_vs_enumeration(dag, at_target=True):
    count = int(dag.tree_count_at_target) if at_target else int(dag.tree_count)
    h = dag.rid_histograms(at_target=at_target)
    assert h["complete"] and h["count_mismatch"] == 0
    assert int(h["n_trees"]) == count
    phis = np.asarray(dag.tree_phis(list(range(count)), at_target=at_target))
    for j in range(phis.shape[1]):
        got = hist_dict(h, j)
        assert sum(got.values()) == count
        assert got == dict(collections.Counter(phis[:, j].tolist())), j
        assert list(h["nums"][j]) == sorted(got)
    return h


@pytest.mark.parametrize("inst", INSTANCES)
def test_histograms_equal_enumeration(inst):
    X, y, thr = tiny(inst["seed"], n=inst.get("n", 12))
    dag = solve(X, y, thr, max_k=inst["max_k"])
    check_hist_vs_enumeration(dag, at_target=True)
    check_hist_vs_enumeration(dag, at_target=False)


@pytest.mark.parametrize("seed", [1, 2])
def test_histograms_shape_mode(seed):
    X, y, thr = tiny(seed)
    dag = solve(X, y, thr, max_k=3, shape_mode=True)
    tc = int(dag.tree_count_at_target)
    assert any("group_map" in dag.get_tree(i) for i in range(tc))
    check_hist_vs_enumeration(dag)
    phis = np.asarray(dag.tree_phis(list(range(tc))))
    for i, t in enumerate(all_trees(dag, tc)):
        assert phis[i].tolist() == brute_nums(t, X, y)[0]


def test_histograms_recursive_fill():
    X, y, thr = tiny(4, n=14)
    dag = solve(X, y, thr, depth=3, max_k=2, theta=0.5, lookahead=1, recursive_fill=True)
    assert int(dag.tree_count_at_target) > 1
    check_hist_vs_enumeration(dag)


# ------------------------------------------------------------ noise column
def test_noise_column_is_zero():
    X, y, thr = tiny(1, noise_col=True)
    thr[3] = []                          # column 3: no candidate thresholds
    dag = solve(X, y, thr, max_k=3)
    tc = int(dag.tree_count_at_target)
    h = dag.rid_histograms()
    assert hist_dict(h, 3) == {0: tc}


# ------------------------------------------------------------ other evaluation rows
def test_other_evaluation_rows():
    X, y, thr = tiny(5, n=14)
    dag = solve(X, y, thr, max_k=3)
    rng = np.random.default_rng(99)
    X2 = rng.integers(0, 4, size=(9, 3)).astype(np.float64)
    y2 = rng.integers(0, 2, size=9)
    tc = int(dag.tree_count_at_target)
    h = dag.rid_histograms(X2, y2)
    assert h["n_eval"] == 9 and h["complete"] and h["count_mismatch"] == 0
    phis = np.asarray(dag.tree_phis(list(range(tc)), X2, y2))
    for i, t in enumerate(all_trees(dag, tc)):
        assert phis[i].tolist() == brute_nums(t, X2, y2)[0]
    for j in range(3):
        assert hist_dict(h, j) == dict(collections.Counter(phis[:, j].tolist()))
    with pytest.raises(ValueError):
        dag.rid_histograms(X2, None)


# ------------------------------------------------------------ aggregation
def test_aggregate_matches_replicated_sample():
    n, p = 5, 2
    boots = [
        BootResult(0, n, 3, "exact", [np.array([0, 4]), np.array([-2, 0, 6])],
                   [np.array([2, 1]), np.array([1, 1, 1])]),
        BootResult(1, n, 4, "exact", [np.array([0, 3, 4]), np.array([0])],
                   [np.array([1, 2, 1]), np.array([4])]),
        BootResult(2, n, 6, "sampled", [np.array([-1, 4, 9]), np.array([1, 6])],
                   [np.array([1, 2, 3]), np.array([5, 1])]),
        BootResult(3, n, 0, "failed", [np.zeros(0, int)] * p, [np.zeros(0, int)] * p),
    ]
    rid = aggregate_rid(boots, p)
    assert rid.B == 4 and rid.B_ok == 3 and rid.n == n
    L = 12   # lcm(3, 4, 6)
    for j in range(p):
        sample = []
        for bt in boots[:3]:
            tot = int(sum(bt.counts[j]))
            for v, c in zip(bt.nums[j], bt.counts[j]):
                sample += [v / n**2] * (int(c) * L // tot)
        sample = np.sort(np.asarray(sample))
        assert abs(rid.weights[j].sum() - 1.0) < 1e-12
        for k in np.linspace(-0.2, 0.5, 29):
            assert abs(rid.cdf(j, k) - float((sample <= k).mean())) < 1e-12
        assert abs(rid.mean(j) - sample.mean()) < 1e-12
        qs = {q: float(np.quantile(sample, q, method="inverted_cdf")) for q in (0.25, 0.5, 0.75)}
        for q, v in qs.items():
            assert rid.quantile(j, q) == v
        row = rid.summary().iloc[j]
        iqr = qs[0.75] - qs[0.25]
        assert row["median"] == qs[0.5] and row["q1"] == qs[0.25] and row["q3"] == qs[0.75]
        assert abs(row["whisker_lo"] - (qs[0.25] - 1.5 * iqr)) < 1e-12
        assert abs(row["whisker_hi"] - (qs[0.75] + 1.5 * iqr)) < 1e-12
        assert abs(row["p_pos"] - float((sample > 0).mean())) < 1e-12
        assert abs(row["p_zero"] - float((sample == 0).mean())) < 1e-12
        assert row["min"] == sample.min() and row["max"] == sample.max()
        assert row["n_bootstraps_sampled"] == 1 and row["n_bootstraps_used"] == 3


# ------------------------------------------------------------ bootstrap indices
def test_bootstrap_indices_deterministic():
    a = bootstrap_indices(50, 0)
    assert a.shape == (50,) and a.min() >= 0 and a.max() < 50
    assert np.array_equal(a, bootstrap_indices(50, 0))
    assert not np.array_equal(a, bootstrap_indices(50, 1))
    assert not np.array_equal(a, bootstrap_indices(50, 0, tag="other"))
    assert not np.array_equal(a, bootstrap_indices(50, 0, base_seed=1))


# ------------------------------------------------------------ RESPLINTER-level
def synth(n=60, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.integers(0, 4, size=(n, 3)).astype(np.float64)
    y = ((X[:, 0] >= 2) ^ (rng.random(n) < 0.15)).astype(int)
    return X, y


FIT = dict(max_depth=2, max_k=3, reg=0.02, theta=0.3, beam_width=4,
           lgbm_n_estimators=20, fill_parallel=False)


def run_rid(X, y, fit, **kw):
    return rid_mod._compute_rid(RESPLINTER(**fit), X, y, **kw)


def rid_one_bootstrap(X, y, b, fit, thresholds, **kw):
    return rid_mod._rid_one_bootstrap(RESPLINTER(**fit), X, y, b, thresholds, **kw)


def test_public_compute_rid_signature_and_seed():
    import inspect
    assert list(inspect.signature(compute_rid).parameters) == [
        "estimator", "X", "y", "B", "random_state", "out_dir"]
    assert rid_mod.__all__ == ["compute_rid", "RID"]
    X, y = synth()
    public = {k: v for k, v in FIT.items() if k != "fill_parallel"}
    estimator = PublicRESPLINTER(**public)
    a = compute_rid(estimator, X, y, B=2, random_state=3)
    assert not hasattr(estimator, "lgbm_thresholds_")  # the caller's estimator is not fit
    b = rid_mod._compute_rid(PublicRESPLINTER(**public), X, y, B=2, base_seed=3)
    assert a.summary().equals(b.summary()) and a.B_ok == 2
    assert not a.summary().equals(compute_rid(estimator, X, y, B=2).summary())


def test_compute_rid_reproducible(tmp_path):
    X, y = synth()
    r1 = run_rid(X, y, FIT, B=3)
    r2 = run_rid(X, y, FIT, B=3)
    assert r1 is not None and r1.B_ok == 3
    s1, s2 = r1.summary(), r2.summary()
    assert s1.equals(s2)
    # resumable: a partial run returns None, the rest completes from disk
    out = str(tmp_path / "rid")
    assert run_rid(X, y, FIT, B=3, out_dir=out, b_range=[0]) is None
    r3 = run_rid(X, y, FIT, B=3, out_dir=out)
    assert r3.summary().equals(s1)
    r4 = run_rid(X, y, FIT, B=3, out_dir=out)     # all from disk
    assert r4.summary().equals(s1)


def test_fill_parallel_invariance():
    X, y = synth()
    thr = RESPLINTER(**FIT)._derive_thresholds(X, y)
    kw = dict(FIT, max_depth=3, lookahead_depth=1, max_k=2)
    hs = []
    for fp in (False, True):
        m = RESPLINTER(**dict(kw, fill_parallel=fp), fixed_thresholds=thr).fit(X, y)
        h = m.dag_.rid_histograms()
        assert h["complete"] and h["count_mismatch"] == 0
        hs.append([hist_dict(h, j) for j in range(3)])
    assert hs[0] == hs[1]


def test_fixed_thresholds_reproduce_default_fit():
    X, y = synth()
    default = RESPLINTER(**FIT).fit(X, y)
    thr = RESPLINTER(**FIT)._derive_thresholds(X, y)
    fixed = RESPLINTER(**FIT, fixed_thresholds=thr).fit(X, y)
    assert fixed.threshold_source_ == "fixed" and default.threshold_source_ == "lightgbm"
    assert fixed.lgbm_time_ == 0
    assert fixed.tree_count_ == default.tree_count_
    assert fixed.opt_ == default.opt_
    assert "fixed_thresholds" in RESPLINTER().get_params()


def test_doubled_rows_scale_by_four():
    X, y = synth()
    thr = RESPLINTER(**FIT)._derive_thresholds(X, y)
    a = RESPLINTER(**FIT, fixed_thresholds=thr).fit(X, y)
    b = RESPLINTER(**FIT, fixed_thresholds=thr).fit(np.vstack([X, X]), np.concatenate([y, y]))
    assert a.opt_ == b.opt_
    assert a.tree_count_at_target_ == b.tree_count_at_target_
    ha, hb = a.dag_.rid_histograms(), b.dag_.rid_histograms()
    for j in range(3):
        assert {4 * k: v for k, v in hist_dict(ha, j).items()} == hist_dict(hb, j)


def test_state_cap_falls_back_to_sampling():
    X, y = synth()
    thr = RESPLINTER(**FIT)._derive_thresholds(X, y)
    m = RESPLINTER(**FIT, fixed_thresholds=thr).fit(X, y)
    assert m.dag_.rid_histograms(max_states=1)["complete"] is False
    M = 500
    r1 = rid_one_bootstrap(X, y, 0, FIT, thr, max_states=1, n_fallback=M, sample_seed=7)
    r2 = rid_one_bootstrap(X, y, 0, FIT, thr, max_states=1, n_fallback=M, sample_seed=7)
    assert r1.mode == "sampled"
    for j in range(3):
        assert int(r1.counts[j].sum()) == M
        assert np.array_equal(r1.nums[j], r2.nums[j])
        assert np.array_equal(r1.counts[j], r2.counts[j])
    ex = rid_one_bootstrap(X, y, 0, FIT, thr)
    assert ex.mode == "exact"
    assert all(int(c.sum()) == ex.tree_count for c in ex.counts)


# ------------------------------------------------------------ resume integrity
def _rewrite_meta(path, **kw):
    import json
    with open(path) as f:
        js = json.load(f)
    js.update({k: v for k, v in kw.items() if k != "meta"})
    js["meta"].update(kw.get("meta", {}))
    with open(path, "w") as f:
        json.dump(js, f)


def test_resume_rejects_other_config_and_indices(tmp_path):
    X, y = synth()
    out = str(tmp_path / "rid")
    r = run_rid(X, y, FIT, B=1, out_dir=out)
    assert r is not None and r.B_ok == 1
    with pytest.raises(ValueError, match="different settings"):
        run_rid(X, y, dict(FIT, theta=0.2), B=1, out_dir=out)
    with pytest.raises(ValueError, match="different settings"):
        run_rid(X, y, FIT, B=1, out_dir=out, max_states=1)
    _rewrite_meta(os.path.join(out, "boot_000.json"), meta={"idx_sha256": "0" * 64})
    with pytest.raises(ValueError, match="different rows"):
        run_rid(X, y, FIT, B=1, out_dir=out)


class PanicException(BaseException):
    """Stand-in with pyo3's name (pyo3's PanicException derives from
    BaseException)."""


def test_panic_is_a_failed_bootstrap_and_retry(tmp_path, monkeypatch):
    exc = PanicException
    assert not issubclass(exc, Exception)
    X, y = synth()
    orig = RESPLINTER.fit
    calls = []

    def fit(self, *a, **k):
        calls.append(1)
        if len(calls) == 1:
            raise exc("boom")
        return orig(self, *a, **k)

    monkeypatch.setattr(RESPLINTER, "fit", fit)
    out = str(tmp_path / "rid")
    r = run_rid(X, y, FIT, B=2, out_dir=out)
    assert r is not None and r.B == 2 and r.B_ok == 1 and r.modes == ["failed", "exact"]
    saved = rid_mod._load_boot(out, 0)
    assert saved.mode == "failed" and saved.meta["reasons"] == "panic:boom"
    monkeypatch.setattr(RESPLINTER, "fit", orig)
    # a rerun reuses the failed bootstrap ...
    assert run_rid(X, y, FIT, B=2, out_dir=out).B_ok == 1
    # ... unless retry_failed, which re-runs and overwrites it
    r2 = run_rid(X, y, FIT, B=2, out_dir=out, retry_failed=True)
    assert r2.B_ok == 2 and rid_mod._load_boot(out, 0).mode == "exact"
    assert r2.summary().equals(run_rid(X, y, FIT, B=2).summary())


def test_keyboard_interrupt_propagates(monkeypatch):
    X, y = synth()
    thr = RESPLINTER(**FIT)._derive_thresholds(X, y)

    def fit(self, *a, **k):
        raise KeyboardInterrupt

    monkeypatch.setattr(RESPLINTER, "fit", fit)
    with pytest.raises(KeyboardInterrupt):
        rid_one_bootstrap(X, y, 0, FIT, thr)


def test_other_exception_reason(monkeypatch):
    X, y = synth()
    thr = RESPLINTER(**FIT)._derive_thresholds(X, y)

    def fit(self, *a, **k):
        raise RuntimeError("bad")

    monkeypatch.setattr(RESPLINTER, "fit", fit)
    r = rid_one_bootstrap(X, y, 0, FIT, thr)
    assert r.mode == "failed" and r.meta["reasons"] == "error:RuntimeError:bad"


# ------------------------------------------------------------ count mismatch
class _DagProxy:
    def __init__(self, dag, **override):
        self._dag, self._override, self.sampled = dag, override, False

    def rid_histograms(self, **kw):
        res = dict(self._dag.rid_histograms(**kw))
        res.update(self._override)
        return res

    def sample_indices(self, *a, **k):
        self.sampled = True
        return self._dag.sample_indices(*a, **k)

    def __getattr__(self, name):
        return getattr(self._dag, name)


@pytest.mark.parametrize("complete", [True, False])
def test_count_mismatch_is_failed_not_sampled(monkeypatch, complete):
    X, y = synth()
    thr = RESPLINTER(**FIT)._derive_thresholds(X, y)
    orig = RESPLINTER.fit
    proxies = []

    def fit(self, *a, **k):
        out = orig(self, *a, **k)
        self.dag_ = _DagProxy(self.dag_, count_mismatch=3, complete=complete)
        proxies.append(self.dag_)
        return out

    monkeypatch.setattr(RESPLINTER, "fit", fit)
    r = rid_one_bootstrap(X, y, 0, FIT, thr)
    assert r.mode == "failed" and r.meta["reasons"] == "count_mismatch=3"
    assert r.meta["count_mismatch"] == 3 and not proxies[0].sampled


# ------------------------------------------------------------ fixed_thresholds validation
def test_fixed_thresholds_validation():
    X, y = synth()
    thr = RESPLINTER(**FIT)._derive_thresholds(X, y)
    assert sorted(thr) == [0, 1, 2]
    missing = {j: v for j, v in thr.items() if j != 1}
    with pytest.raises(ValueError, match="missing"):
        RESPLINTER(**FIT, fixed_thresholds=missing).fit(X, y)
    with pytest.raises(ValueError, match="out of range"):
        RESPLINTER(**FIT, fixed_thresholds={**thr, 3: [0.5]}).fit(X, y)
    with pytest.raises(ValueError, match="logit_weight"):
        RESPLINTER(**dict(FIT, logit_weight=0.5), fixed_thresholds=thr).fit(X, y)
    # [] for a feature is allowed (never split on it)
    m = RESPLINTER(**FIT, fixed_thresholds={**thr, 2: []}).fit(X, y)
    assert m.tree_count_ > 0


def test_mask_bytes_cap_and_peak():
    X, y = synth()
    thr = RESPLINTER(**FIT)._derive_thresholds(X, y)
    m = RESPLINTER(**FIT, fixed_thresholds=thr).fit(X, y)
    h = m.dag_.rid_histograms()
    assert h["complete"] and h["mask_bytes_peak"] > 0
    h2 = m.dag_.rid_histograms(max_mask_bytes=int(h["mask_bytes_peak"]))
    assert h2["complete"] and all(np.array_equal(a, b) for a, b in zip(h["nums"], h2["nums"]))
    assert m.dag_.rid_histograms(max_mask_bytes=int(h["mask_bytes_peak"]) - 1)["complete"] is False
    r = rid_one_bootstrap(X, y, 0, FIT, thr, max_mask_bytes=8)
    assert r.mode == "sampled" and r.meta["max_mask_bytes"] == 8


def test_explicit_indices():
    X, y = synth()
    thr = RESPLINTER(**FIT)._derive_thresholds(X, y)
    n = len(y)
    a = rid_one_bootstrap(X, y, -1, FIT, thr, indices=np.arange(n), tag="identity")
    assert a.meta["idx_sha256"] == rid_mod._sha256_array(np.arange(n)) and a.b == -1
    direct = RESPLINTER(**FIT, fixed_thresholds=thr).fit(X, y)
    assert a.mode == "exact" and a.tree_count == direct.tree_count_at_target_


# ------------------------------------------------------------ compute_rid inputs
_PUBLIC_FIT = {k: v for k, v in FIT.items() if k != "fill_parallel"}


def test_compute_rid_uses_fixed_thresholds_in_every_bootstrap(monkeypatch):
    import splinter.thresholds as threshold_module
    X, y = synth()
    user = {0: [1.5], 1: [0.5, 2.5], 2: []}
    seen = []
    orig = RESPLINTER.fit

    def fit(self, *a, **k):
        seen.append(self.fixed_thresholds)
        return orig(self, *a, **k)

    def no_lightgbm(*a, **k):
        raise AssertionError("LightGBM must not be fit when fixed_thresholds is set")

    monkeypatch.setattr(RESPLINTER, "fit", fit)
    monkeypatch.setattr(threshold_module, "fit_thresholds", no_lightgbm)
    r = compute_rid(PublicRESPLINTER(**_PUBLIC_FIT, fixed_thresholds=user), X, y, B=3)
    assert r.B_ok == 3 and len(seen) == 3
    assert all(s == user for s in seen)
    with pytest.raises(ValueError, match="threshold_budget"):
        compute_rid(PublicRESPLINTER(**_PUBLIC_FIT, fixed_thresholds=user, threshold_budget=2),
                    X, y, B=1)
    with pytest.raises(ValueError, match="missing"):
        compute_rid(PublicRESPLINTER(**_PUBLIC_FIT, fixed_thresholds={0: [1.5]}), X, y, B=1)


def test_compute_rid_accepts_string_labels():
    X, y = synth()
    r = compute_rid(PublicRESPLINTER(**_PUBLIC_FIT), X, np.where(y == 1, "yes", "no"), B=2)
    assert r.B_ok == r.B == 2
    assert r.summary().equals(compute_rid(PublicRESPLINTER(**_PUBLIC_FIT), X, y, B=2).summary())


def test_compute_rid_rejects_bad_fill_mode_before_bootstraps(monkeypatch):
    X, y = synth()

    def no_bootstrap(*a, **k):
        raise AssertionError("no bootstrap should run")

    monkeypatch.setattr(rid_mod, "_rid_one_bootstrap", no_bootstrap)
    with pytest.raises(ValueError, match="lookahead_depth"):
        compute_rid(PublicRESPLINTER(**dict(_PUBLIC_FIT, lookahead_depth=2),
                                     fill_mode="recursive"), X, y, B=2)


def test_compute_rid_raises_when_every_bootstrap_errors(monkeypatch):
    X, y = synth()

    def fit(self, *a, **k):
        raise RuntimeError("bad")

    monkeypatch.setattr(RESPLINTER, "fit", fit)
    with pytest.raises(RuntimeError, match="error:RuntimeError:bad"):
        compute_rid(PublicRESPLINTER(**_PUBLIC_FIT), X, y, B=2)
