"""DAG save/load in the RSMY v6 format (default writer), v5 and the legacy
v4 format, and per-tree leaf n_samples.

On small Step-B fits (the tiny random instances of tests/_helpers.py,
(d, d_l) = (3, 1) and a full-depth solve):
  (a) v6 round trip: to_bytes() is v6; from_bytes gives the same tree_count,
      the same root schedule (cost bits, counts) and, for every index i, the
      same get_tree(i) as the fitted DAG (n_samples aside: a loaded DAG has
      no training rows and reports the stored per-vertex value);
  (b) v5 / v4 read: to_bytes(5) / to_bytes(4) are v5 / v4 and load to the
      same trees, and v6-, v5- and v4-loaded DAGs give identical get_tree(i)
      dicts (n_samples included);
  (c) v5 is smaller than v4, and v6 smaller than v5 once the DAG holds at
      least 50 trees (stream framing dominates on tinier DAGs);
  (d) every leaf of every fitted get_tree(i) reports the number of training
      rows that reach it (rows routed through the tree), and its loss;
  (e) the default writer emits v6 with DEFLATE streams; a v6 file
      re-saved after loading is byte-identical (n_train is kept);
  (f) merging identical subtrees joins vertices reached by different rows
      (leaves with equal cost, prediction and loss, e.g. pure leaves of any
      size), so a loaded DAG without training rows reports one merged
      support's count on some leaves; attach_training_data and
      RESPLINTER.load restore the routed per-tree counts.
"""
import numpy as np
import pytest

from splinter import RESPLINTER

from _helpers import small_instances, solve_instance, tiny_data

N_INST = 10


def _strip_n(t):
    if t["type"] == "leaf":
        return {k: v for k, v in t.items() if k != "n_samples"}
    return {**t, "children": [_strip_n(c) for c in t["children"]]}


def _hist(dag):
    return [(float(c).hex(), int(m)) for c, m in dag.root_histogram()]


def _leaf_errors(t, X, y, idx, n):
    """(# leaves whose n_samples / loss differ from the routed rows, # leaves)."""
    if t["type"] == "leaf":
        bad = int(len(idx) != t["n_samples"])
        bad += int(abs(int((y[idx] != t["prediction"]).sum()) / n - t["loss"]) > 1e-9)
        return bad, 1
    b = np.searchsorted(np.asarray(t["thresholds"]), X[idx, t["feature_index"]], side="right")
    if t.get("group_map") is not None:
        b = np.asarray(t["group_map"])[b]
    out = (0, 0)
    for j, c in enumerate(t["children"]):
        e = _leaf_errors(c, X, y, idx[b == j], n)
        out = (out[0] + e[0], out[1] + e[1])
    return out


def _version(b):
    return int.from_bytes(b[4:6], "little")


@pytest.mark.parametrize("lookahead", [1, None])
def test_v6_v5_v4_roundtrip_and_leaf_n_samples(lookahead):
    n_trees = 0
    for inst in small_instances(N_INST, seed=0):
        r = solve_instance(inst, 3, lookahead)
        dag, tc = r["dag"], int(r["tree_count"])
        v6, v5, v4 = bytes(dag.to_bytes()), bytes(dag.to_bytes(5)), bytes(dag.to_bytes(4))
        assert bytes(dag.to_bytes(6)) == v6
        assert _version(v6) == 6 and _version(v5) == 5 and _version(v4) == 4
        assert len(v5) < len(v4)
        if tc >= 50:
            assert len(v6) < len(v5), (tc, len(v6), len(v5))
        l6, l5, l4 = (type(dag).from_bytes(b) for b in (v6, v5, v4))
        assert int(l6.tree_count) == tc and int(l5.tree_count) == tc and int(l4.tree_count) == tc
        assert _hist(l6) == _hist(dag) and _hist(l5) == _hist(dag) and _hist(l4) == _hist(dag)
        # A loaded v6 file re-saves byte-identically (n_train recorded).
        assert bytes(l6.to_bytes()) == v6
        X = np.asarray(inst["X"], dtype=np.float64)
        y = np.asarray(inst["y"]).astype(int)
        for i in range(tc):
            t, t6, t5, t4 = dag.get_tree(i), l6.get_tree(i), l5.get_tree(i), l4.get_tree(i)
            assert t6 == t5 == t4
            assert _strip_n(t6) == _strip_n(t)
            bad, _ = _leaf_errors(t, X, y, np.arange(len(y)), len(y))
            assert bad == 0, (i, t)
        n_trees += tc
    assert n_trees > 0


def test_v4_writer_unknown_version_rejected():
    inst = small_instances(1, seed=0)[0]
    dag = solve_instance(inst, 3, 1)["dag"]
    with pytest.raises(ValueError):
        dag.to_bytes(3)
    with pytest.raises(ValueError):
        dag.to_bytes(7)


def test_default_writer_is_v6_deflate():
    inst = small_instances(1, seed=0)[0]
    dag = solve_instance(inst, 3, 1)["dag"]
    comp = bytes(dag.to_bytes())
    assert _version(comp) == 6 and comp[24] == 1
    assert bytes(dag.to_bytes(6)) == comp


def test_merged_leaf_counts_are_routed_after_attach(tmp_path):
    n_stored_differs = 0
    for inst in small_instances(N_INST, seed=0):
        r = solve_instance(inst, 3, 1)
        dag, tc = r["dag"], int(r["tree_count"])
        X = np.asarray(inst["X"], dtype=np.float64)
        y = np.asarray(inst["y"]).astype(int)
        loaded = type(dag).from_bytes(bytes(dag.to_bytes()))
        for i in range(tc):
            n_stored_differs += _leaf_errors(loaded.get_tree(i), X, y, np.arange(len(y)),
                                             len(y))[0]
        loaded.attach_training_data(X, y.tolist(), {f: list(t) for f, t in
                                                    inst["thresholds"].items()}, 2)
        for i in range(tc):
            assert loaded.get_tree(i) == dag.get_tree(i)
    # Merging joined vertices with different supports on these instances.
    assert n_stored_differs > 0
    # The public estimator re-attaches the rows on load.
    X, y = tiny_data(n=60, seed=1)
    rset = RESPLINTER(max_depth=2, lgbm_n_estimators=8).fit(X, y)
    rset.save(tmp_path / "set.npz")
    back = RESPLINTER.load(tmp_path / "set.npz")
    for i in range(min(int(back.tree_count_), 50)):
        t = back.dag_.get_tree(i)
        assert _leaf_errors(t, X, y, np.arange(len(y)), len(y))[0] == 0, (i, t)
