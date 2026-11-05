"""Tree drawings: edge labels must describe the routing exactly."""

import re
import shutil
import subprocess
import sys
import textwrap
from types import SimpleNamespace

import numpy as np
import pytest
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from splinter import SPLINTER, ShapeSPLINTER
from splinter.rules import export_text
from splinter.tree import (MWLeaf, MWNode, MWShapeNode, count_internal_nodes, count_leaves,
                           leaf_loss, predict_encoded)
from splinter.viz import edge_labels, merge_adjacent_leaves, plot_tree

from _helpers import ZERO_SENTINEL, predicate, random_tree, thresholds_of

CUTS = [ZERO_SENTINEL, 2.9999999999999996, 17.0, 647.0000000000001, 1000.0000000004]
BETWEEN = [2.9999999999999996, 1000.0000000002]


def assert_labels_match_routing(node, labels, x):
    """Each row's routed child, and only that child, has a label containing x.

    The node's children are ``leaves(k)``, so a row's prediction is its child.
    """
    child = predict_encoded(node, x[:, None])
    contains = np.array([predicate(label)(x) for label in labels])
    bad = ~contains[child, np.arange(len(x))] | (contains.sum(axis=0) != 1)
    assert not bad.any(), [(float(v), labels) for v in x[bad][:5]]


def leaves(count):
    return [MWLeaf(i, 1, 0.0) for i in range(count)]


def test_contiguous_labels_match_routing():
    node = MWNode(0, "x", list(CUTS), leaves(6))
    printed = [0.0, 3.0, 17.0, 647.0, 1000.0]
    centers = np.array(CUTS + printed)
    x = np.concatenate([centers, centers - 1, centers + 1, [0.0, -5.0], BETWEEN,
                        np.random.default_rng(0).uniform(-10, 1100, 200)])
    labels = edge_labels(node)
    assert labels[0] == "≤ 0"
    assert labels[2] == "[3, 17)"
    assert labels[3] == "[17, 647]"
    assert_labels_match_routing(node, labels, x[~np.isin(x, BETWEEN)])

    guarded = edge_labels(node, values=x)
    assert "2.9999999999999996" in guarded[1] and "2.9999999999999996" in guarded[2]
    assert "1000.0000000004" in guarded[4] and "1000.0000000004" in guarded[5]
    assert guarded[0] == "≤ 0" and guarded[3] == "[17, 647]"
    assert_labels_match_routing(node, guarded, x)


def test_shape_labels_match_routing():
    node = MWShapeNode(0, "x", [1.0, 2.0, 3.0, 4.0], [0, 1, 0, 2, 1], leaves(3))
    labels = edge_labels(node)
    assert labels[0] == "< 1 ∪ [2, 3)"
    rng = np.random.default_rng(1)
    x = np.concatenate([[1.0, 2.0, 3.0, 4.0], rng.uniform(-1, 6, 300)])
    assert_labels_match_routing(node, labels, x)


def test_category_labels_follow_routing():
    names = ["a", "b", "c", "d", "e"]
    shape = MWShapeNode(0, "x", [0.5, 1.5, 2.5, 3.5], [1, 0, 1, 2, 0], leaves(3))
    assert edge_labels(shape, category_names=names) == ["b, e", "a, c", "d"]
    assert edge_labels(shape, category_names=dict(enumerate(names))) == ["b, e", "a, c", "d"]
    contiguous = MWNode(0, "x", [1.0, 3.0], leaves(3))
    assert edge_labels(contiguous, category_names=names) == ["a", "b, c", "d, e"]
    unreached = MWNode(0, "x", [1.0, 3.0, 10.0], leaves(4))
    labels = edge_labels(unreached, category_names=names)
    assert labels[3] == edge_labels(unreached)[3]
    assert predicate(labels[3])(np.array([9.0, 10.0])).tolist() == [False, True]


def total_samples(tree):
    if isinstance(tree, MWLeaf):
        return tree.n_samples
    return sum(total_samples(child) for child in tree.children)


def test_merge_adjacent_leaves_preserves_predictions():
    rng = np.random.default_rng(7)
    merged_any = False
    for _ in range(60):
        tree = random_tree(rng, 3)
        before = export_text(tree)
        merged = merge_adjacent_leaves(tree)
        cuts = np.array(thresholds_of(tree) + [0.0])
        X = rng.choice(np.r_[cuts, rng.uniform(-6, 6, 40)], size=(600, 3))
        X[:len(cuts)] = cuts[:, None]
        np.testing.assert_array_equal(predict_encoded(merged, X), predict_encoded(tree, X))
        assert count_leaves(merged) <= count_leaves(tree)
        assert leaf_loss(merged) == leaf_loss(tree)
        assert total_samples(merged) == total_samples(tree)
        assert export_text(tree) == before
        assert all(type(t) is float for t in thresholds_of(merged))
        merged_any |= count_leaves(merged) < count_leaves(tree)
    assert merged_any

    a, b = MWLeaf(0, 2, 1.0), MWLeaf(1, 3, 0.0)
    node = merge_adjacent_leaves(MWNode(0, "x", [1.0, 2.0, 3.0], [a, a, b, MWLeaf(0, 4, 2.0)]))
    assert isinstance(node, MWNode) and len(node.children) == 3 and node.thresholds == [2.0, 3.0]
    assert node.children[0] == MWLeaf(0, 4, 2.0)

    shape = MWShapeNode(0, "x", [1.0, 2.0, 3.0], [0, 1, 2, 1], [a, b, MWLeaf(1, 5, 1.0)])
    contiguous = merge_adjacent_leaves(shape)
    assert type(contiguous) is MWNode and contiguous.thresholds == [1.0]
    assert contiguous.children == [a, MWLeaf(1, 8, 1.0)]

    assert merge_adjacent_leaves(MWNode(0, "x", [1.0, 2.0], [a, a, a])) == MWLeaf(0, 6, 3.0)
    assert merge_adjacent_leaves(MWShapeNode(0, "x", [1.0], [1, 0], [a, a])) == MWLeaf(0, 4, 2.0)


@pytest.mark.parametrize("estimator", [SPLINTER, ShapeSPLINTER])
def test_plot_tree_fitted_models(estimator, tmp_path):
    pytest.importorskip("graphviz")
    X, y = load_breast_cancer(return_X_y=True)
    y = np.where(y == 1, "benign", "malignant")
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, stratify=y, random_state=0)
    model = estimator(max_depth=2, lookahead_depth=1, min_samples_leaf=5,
                      lgbm_n_estimators=8).fit(X_tr, y_tr)
    n_nodes = count_leaves(model.tree_) + count_internal_nodes(model.tree_)
    for style in ("simple", "detailed"):
        dot = plot_tree(model, X_te, y_te, style=style)
        body = "".join(dot.body)
        assert len(re.findall(r"^\tn(?:_\d+)* \[", body, re.M)) == n_nodes
        assert body.count(" -> ") == n_nodes - 1
        assert "benign" in body and "malignant" in body
    if shutil.which("dot"):
        path = dot.render(tmp_path / "tree", format="png", cleanup=True)
        assert (tmp_path / "tree.png").exists() and str(path).endswith("tree.png")


def test_plot_tree_bare_record_and_input_errors():
    pytest.importorskip("graphviz")
    record = MWNode(0, "f", [0.5], [MWLeaf(0, 3, 0.0), MWLeaf(1, 2, 0.0)])
    X, classes = np.array([[0.0], [1.0], [1.0]]), np.array(["x", "y"])
    source = plot_tree(record, X, ["x", "y", "y"], classes=classes).source
    assert "x\nn = 1" in source and "y\nn = 2" in source
    with pytest.raises(ValueError):
        plot_tree(record, X, ["x", "z", "y"], classes=classes)
    with pytest.raises(ValueError):
        plot_tree(record, y=["x", "y", "y"], classes=classes)
    with pytest.raises(ValueError):
        plot_tree(record, style="fancy")
    with pytest.raises(ValueError):
        plot_tree(SimpleNamespace(tree_=record, classes_=classes), classes=classes)
    with pytest.raises(TypeError):
        plot_tree({"type": "leaf"})


def test_viz_helpers_without_graphviz():
    code = textwrap.dedent("""
        import sys
        sys.modules["graphviz"] = None
        from splinter.tree import MWLeaf, MWNode
        from splinter.viz import edge_labels, merge_adjacent_leaves, plot_tree
        node = MWNode(0, "x", [1.0], [MWLeaf(0, 1, 0.0), MWLeaf(0, 1, 0.0)])
        assert len(edge_labels(node)) == 2
        assert merge_adjacent_leaves(node) == MWLeaf(0, 2, 0.0)
        try:
            plot_tree(node)
        except ImportError as error:
            assert "graphviz" in str(error), error
        else:
            raise AssertionError("plot_tree drew without graphviz")
        sys.modules["matplotlib"] = None
        del sys.modules["splinter.viz"]
        import splinter.viz
        try:
            splinter.viz.plot_rid(None)
        except ImportError as error:
            assert "matplotlib" in str(error), error
        else:
            raise AssertionError("plot_rid drew without matplotlib")
    """)
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_plot_tree_routes_shape_nodes_and_guards_with_all_rows():
    pytest.importorskip("graphviz")
    shape = MWShapeNode(0, "x", [1.0, 2.0], [0, 1, 0], [MWLeaf(0, 0, 0.0), MWLeaf(1, 0, 0.0)])
    source = plot_tree(shape, np.array([[0.0], [1.5], [2.5], [2.5]])).source
    assert "0\nn = 3" in source and "1\nn = 1" in source
    # The row that triggers the guard on the child split reaches the other branch.
    t = 2.9999999999999996
    child = MWNode(1, "z", [t], [MWLeaf(0, 0, 0.0), MWLeaf(1, 0, 0.0)])
    root = MWNode(0, "x", [0.5], [child, MWLeaf(1, 0, 0.0)])
    X = np.array([[0.0, 1.0], [0.0, 4.0], [1.0, t]])
    assert repr(t) in plot_tree(root, X).source


def test_plot_tree_accepts_unsorted_classes():
    pytest.importorskip("graphviz")
    node = MWNode(0, "f", [0.5], [MWLeaf(0, 1, 0.0), MWLeaf(1, 1, 0.0)])
    source = plot_tree(node, np.array([[0.0], [1.0]]), np.array(["yes", "no"]),
                       classes=np.array(["yes", "no"])).source
    assert "yes\nn = 1" in source and "no\nn = 1" in source


def synthetic_rid():
    """Features a, b, c (n = 2, weights over itotal = 4) and an empty feature d."""
    from splinter.rid import RID
    nums = [np.array([0, 2]), np.array([2, 4]), np.array([0, 1]), np.array([], dtype=np.int64)]
    iweights = [np.array([2, 2], dtype=object), np.array([1, 3], dtype=object),
                np.array([3, 1], dtype=object), np.array([], dtype=object)]
    return RID(values=[v / 4.0 for v in nums], weights=[w.astype(float) / 4 for w in iweights],
               nums=nums, n=2, B=1, B_ok=1, modes=["exact"], iweights=iweights, itotal=4)


def test_rid_plots_from_synthetic_rid():
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    from splinter.viz import plot_rid, plot_rid_cdf
    rid, names = synthetic_rid(), ["a", "b", "c", "d"]
    assert [rid.mean(j) for j in range(3)] == [0.25, 0.875, 0.0625]

    def rows(fig):
        ax = fig.axes[0]
        dots = next(line for line in ax.lines if line.get_marker() == "o")
        labels = {tick: label.get_text() for tick, label in zip(ax.get_yticks(), ax.get_yticklabels())}
        order = np.argsort(-dots.get_ydata())
        return [labels[y] for y in dots.get_ydata()[order]], list(dots.get_xdata()[order])

    assert rows(plot_rid(rid, names)) == (["b", "a", "c"], [0.875, 0.25, 0.0625])
    assert rows(plot_rid(rid, names, top=2)) == (["b", "a"], [0.875, 0.25])
    assert rows(plot_rid(rid))[0] == ["x1", "x0", "x2"]

    fig = plot_rid_cdf(rid, names, features=["c", 1])
    lines = fig.axes[0].lines
    assert [line.get_label() for line in lines] == ["c", "b"]
    assert all(np.isclose(line.get_ydata()[-1], 1.0) for line in lines)
    c_line = lines[0]
    assert c_line.get_drawstyle() == "steps-post"
    np.testing.assert_allclose(c_line.get_xdata(), [0.0, 0.0, 0.25])
    np.testing.assert_allclose(c_line.get_ydata(), [0.0, 0.75, 1.0])
    with pytest.raises(ValueError):
        plot_rid_cdf(rid, features=[])

    from splinter.rid import RID
    # Whiskers are the exact 5/25/75/95% quantiles; 1/20 of the mass sits at 0.
    edge = RID(values=[np.array([0.0, 1.0])], weights=[np.array([0.05, 0.95])],
               nums=[np.array([0, 4], dtype=object)], n=2, B=1, B_ok=1, modes=["exact"],
               iweights=[np.array([1, 19], dtype=object)], itotal=20)
    whisker, box = plot_rid(edge).axes[0].collections[:2]
    np.testing.assert_allclose(np.asarray(whisker.get_segments())[0][:, 0], [0.0, 1.0])
    np.testing.assert_allclose(np.asarray(box.get_segments())[0][:, 0], [1.0, 1.0])
    assert len(plot_rid_cdf(rid, names).axes[0].lines) == 3
    with pytest.raises(ValueError):
        plot_rid_cdf(rid, features=[0, 1, 2, 0, 1, 2, 0])


def test_multiplicity_on_small_resplinter():
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    from splinter import RESPLINTER
    from splinter.viz import multiplicity_summary, plot_multiplicity
    X = np.array([[i % 4, i // 4] for i in range(24)], dtype=float)
    y = np.array([int(x[0] >= 2) for x in X])
    y[[5, 18]] ^= 1  # two flipped labels and a wide theta, so some rows get conflicting votes
    rset = RESPLINTER(max_depth=2, lookahead_depth=1, beam_width=2,
                      reg=0.02, theta=1.0, lgbm_n_estimators=8, cart_closure=False).fit(X, y)
    summary = multiplicity_summary(rset, X)
    assert summary["n_trees"] == rset.tree_count_at_target_ > 1
    assert 0 < summary["share_conflicting"] < 1
    np.testing.assert_array_equal(summary["p"], rset.predict_probs_exact(X))
    assert 0 <= summary["share_conflicting"] <= 1 and 0 <= summary["share_unanimous"] <= 1
    assert summary["share_conflicting"] + summary["share_unanimous"] == 1
    fig = plot_multiplicity(rset, X)
    patches = fig.axes[0].patches
    assert sum(patch.get_height() for patch in patches) == len(X)
    p = summary["p"]
    gray = [patch.get_height() for patch in patches
            if matplotlib.colors.to_hex(patch.get_facecolor()) == "#999999"]
    assert gray == [np.sum(p == 0), np.sum(p == 1)]
    assert sum(patch.get_height() for patch in patches[2:]) == summary["conflicting"].sum()
    with pytest.raises(ValueError):
        plot_multiplicity(RESPLINTER(), X)
