import numpy as np
import pytest

from splinter.explore import (_max_arity, explanation_size, rashomon_table, select,
                              tree_predictions)
from splinter.tree import MWLeaf, MWNode, MWShapeNode, predict_encoded


def _small_rset(labels=None):
    from splinter import RESPLINTER
    X = np.array([[i % 4, i // 4] for i in range(24)], dtype=float)
    y = np.array([int(x[0] >= 2) for x in X])
    y[[5, 18]] ^= 1  # two flipped labels and a wide theta, so the set holds many trees
    if labels is not None:
        y = np.asarray(labels)[y]
    rset = RESPLINTER(max_depth=2, lookahead_depth=1, beam_width=2,
                      reg=0.02, theta=1.0, lgbm_n_estimators=8, cart_closure=False).fit(X, y)
    return rset, X, y


@pytest.fixture(scope="module")
def small():
    rset, X, y = _small_rset()
    return rset, X, y, rashomon_table(rset, X, y)


def test_explanation_size_counts_distinct_features():
    leaf = MWLeaf(0, 1, 0.0)
    again_f0 = MWNode(0, "f0", [1.0], [leaf, leaf])
    then_f1 = MWNode(1, "f1", [0.5], [leaf, leaf])
    tree = MWNode(0, "f0", [2.0], [again_f0, then_f1])
    # f0 twice counts once; f0 then f1 counts two. A row at 2.0 routes right (to f1).
    X = np.array([[0.0, 0.0], [1.0, 0.0], [1.9, 0.5], [2.0, 0.0], [2.0, 0.5], [9.0, 1.0]])
    np.testing.assert_array_equal(explanation_size(tree, X), [1, 1, 1, 2, 2, 2])
    # A shape split sends x1 < 1 and x1 >= 2 to the f0 child, [1, 2) to a leaf.
    shape = MWShapeNode(1, "f1", [1.0, 2.0], [0, 1, 0], [again_f0, leaf])
    X = np.array([[0.0, 0.0], [3.0, 1.5], [0.5, 2.0], [9.0, 0.9]])
    np.testing.assert_array_equal(explanation_size(shape, X), [2, 1, 2, 2])
    assert _max_arity(shape) == 2


def test_max_arity_hand_trees():
    leaf = MWLeaf(0, 1, 0.0)
    assert _max_arity(leaf) == 0
    three = MWNode(1, "f1", [1.0, 2.0], [leaf, leaf, leaf])
    assert _max_arity(MWNode(0, "f0", [2.0], [leaf, three])) == 3
    four = MWNode(0, "f0", [1.0, 2.0, 3.0], [leaf, three, leaf, leaf])
    assert _max_arity(four) == 4


def test_rashomon_table_all(small):
    rset, X, y, table = small
    assert table.attrs["mode"] == "all"
    assert table.attrs["n_trees"] == rset.tree_count_at_target_ == len(table) > 1
    assert table.attrs["n_rows"] == len(X)
    assert list(table.columns) == ["tree", "objective", "acc", "aes", "leaves", "min_leaf",
                                   "depth", "max_arity", "n_features", "root", "features"]
    objective = table["objective"].to_numpy()
    assert np.all(np.diff(objective) >= -1e-12)
    assert abs(objective[0] - rset.optimal_objective()) <= 1e-9
    assert np.all(objective <= rset.theta_bound_ + 1e-9)
    predictions = tree_predictions(table)
    assert predictions.dtype == np.uint8 and predictions.shape == (len(table), len(X))
    np.testing.assert_array_equal(table.attrs["classes"], rset.classes_)

    def arity(node):  # brute force: children of every internal node
        if isinstance(node, MWLeaf):
            return []
        return [len(node.children)] + [a for child in node.children for a in arity(child)]

    def leaf_sizes(node):  # brute force: training n_samples of every leaf
        if isinstance(node, MWLeaf):
            return [node.n_samples]
        return [n for child in node.children for n in leaf_sizes(child)]

    for k, row in enumerate(table.itertuples()):
        tree = rset.get_tree(row.tree)
        assert row.acc == (rset.classes_[predict_encoded(tree, X)] == y).mean()
        assert row.aes == explanation_size(tree, X).mean()
        assert row.n_features == len(row.features)
        assert set(row.features) <= set(rset.feature_names_)
        np.testing.assert_array_equal(predictions[k], predict_encoded(tree, X))
        assert row.max_arity == max(arity(tree), default=0)
        assert row.min_leaf == min(leaf_sizes(tree))
        assert row.root == (None if isinstance(tree, MWLeaf)
                            else rset.feature_names_[tree.feature_index])
        assert row.root in row.features
    assert table.max_arity.nunique() > 1 and table.root.nunique() > 1

    def route(node, row, used):  # brute force: one row's path and the features it tests
        if isinstance(node, MWLeaf):
            return node.prediction, used
        branch = sum(row[node.feature_index] >= t for t in node.thresholds)
        if isinstance(node, MWShapeNode):
            branch = node.group_map[branch]
        return route(node.children[branch], row, used | {node.feature_index})

    for row in table.itertuples():
        paths = [route(rset.get_tree(row.tree), x, frozenset()) for x in X]
        assert row.aes == np.mean([len(used) for _, used in paths])
        assert row.acc == np.mean([rset.classes_[p] == t for (p, _), t in zip(paths, y)])

def test_rashomon_table_sample(small):
    rset, X, y, _ = small
    a = rashomon_table(rset, X, y, max_trees=5, seed=3)
    b = rashomon_table(rset, X, y, max_trees=5, seed=3)
    assert a.attrs["mode"] == "sample" and len(a) == 5
    assert a.equals(b)
    assert np.all(a["tree"] < rset.tree_count_at_target_)
    with pytest.raises(ValueError):
        rashomon_table(rset, X, np.full(len(X), 7))


def test_rashomon_table_string_labels():
    rset, X, y = _small_rset(labels=["no", "yes"])
    assert list(rset.classes_) == ["no", "yes"]
    table = rashomon_table(rset, X, y)
    assert len(table) > 1
    for row in table.itertuples():
        tree = rset.get_tree(row.tree)
        assert row.acc == (rset.classes_[predict_encoded(tree, X)] == y).mean()
    stored = tree_predictions(table)
    i = int(np.flatnonzero(stored.min(0) != stored.max(0))[0])
    yes, no = select(table, predicts={i: "yes"}), select(table, predicts={i: "no"})
    assert 0 < len(yes) < len(table) and len(yes) + len(no) == len(table)
    with pytest.raises(ValueError, match="not a class"):
        select(table, predicts={i: 1})


def test_select_ranges(small):
    rset, X, y, table = small
    closed = select(table, acc=(0.85, 0.95))
    assert closed.equals(table[(table.acc >= 0.85) & (table.acc <= 0.95)])
    low = select(table, aes=(None, 1.7))
    assert low.equals(table[table.aes <= 1.7])
    high = select(table, objective=(0.2, None))
    assert high.equals(table[table.objective >= 0.2])
    exact = select(table, max_arity=3)
    assert exact.equals(table[table.max_arity == 3])
    combined = select(table, acc=(0.9, None), aes=(None, 2), depth=(None, 2),
                      max_arity=(3, None), leaves=[None, 8], n_features=2)
    expected = table[(table.acc >= 0.9) & (table.aes <= 2) & (table.depth <= 2)
                     & (table.max_arity >= 3) & (table.leaves <= 8) & (table.n_features == 2)]
    assert combined.equals(expected)
    for kept in (closed, low, high, exact, combined):
        assert 0 < len(kept) < len(table)
    assert select(table, acc=(None, None)).equals(table)
    assert table.min_leaf.nunique() > 1
    lo = int(table.min_leaf.median())
    sized = select(table, min_leaf=(lo, None))
    assert sized.equals(table[table.min_leaf >= lo])
    assert select(table, min_leaf=(None, lo)).equals(table[table.min_leaf <= lo])
    with pytest.raises(ValueError, match="allowed"):
        select(table, min_acc=0.9)
    with pytest.raises(ValueError, match="allowed"):
        select(table, features=("x0",))
    with pytest.raises(ValueError, match="range"):
        select(table, acc=(0.1, 0.5, 0.9))


def test_select_features_and_root(small):
    rset, X, y, table = small
    names = list(rset.feature_names_)
    feature_sets = table.features.map(set)
    for value in (names[0], [names[0]], names, (names[1],)):
        listed = {value} if isinstance(value, str) else set(value)
        assert select(table, uses=value).equals(table[feature_sets.map(lambda f: listed <= f)])
        assert select(table, uses_any=value).equals(table[feature_sets.map(lambda f: bool(listed & f))])
        assert select(table, avoids=value).equals(table[feature_sets.map(lambda f: not listed & f)])
        assert select(table, root=value).equals(table[table.root.map(lambda r: r in listed)])
    assert 0 < len(select(table, root=names[0])) < len(table)
    assert 0 < len(select(table, avoids=names[1])) < len(table)
    both = select(table, uses=names[0], root=names[1], max_arity=(3, None))
    assert both.equals(table[feature_sets.map(lambda f: names[0] in f) & (table.root == names[1])
                             & (table.max_arity >= 3)])
    for key in ("uses", "uses_any", "avoids", "root"):
        with pytest.raises(ValueError, match="valid names"):
            select(table, **{key: ["not_a_feature"]})


def test_select_predicts(small):
    rset, X, y, table = small
    predictions = tree_predictions(table)
    disputed = np.flatnonzero(predictions.min(0) != predictions.max(0))
    assert len(disputed) >= 2
    i, j = (int(r) for r in disputed[:2])
    yes = select(table, predicts={i: rset.classes_[1]})
    no = select(table, predicts={i: rset.classes_[0]})
    assert 0 < len(yes) < len(table)
    assert set(yes.tree).isdisjoint(no.tree) and set(yes.tree) | set(no.tree) == set(table.tree)
    assert len(yes) + len(no) == len(table)
    both = select(table, predicts={i: 1, j: 0})
    brute = [row.tree for row in table.itertuples()
             if list(predict_encoded(rset.get_tree(row.tree), X)[[i, j]]) == [1, 0]]
    assert list(both.tree) == brute
    for kept in (yes, no, both):
        np.testing.assert_array_equal(tree_predictions(kept), predictions[kept.index])
        assert kept.attrs["n_trees"] == table.attrs["n_trees"]
    chained = select(select(table, depth=2, max_arity=(3, None)), predicts={i: 1})
    direct = table[(table.depth == 2) & (table.max_arity >= 3)
                   & (table.tree.isin(yes.tree))]
    assert chained.equals(direct) and len(chained) > 0
    for row, stored in zip(chained.itertuples(), tree_predictions(chained)):
        np.testing.assert_array_equal(stored, predict_encoded(rset.get_tree(row.tree), X))
    # Plain pandas filtering stays aligned too, and attrs are shared, not copied.
    masked = table[table.depth == 2]
    for row, stored in zip(masked.itertuples(), tree_predictions(masked)):
        np.testing.assert_array_equal(stored, predict_encoded(rset.get_tree(row.tree), X))
    assert masked.attrs["predictions"] is table.attrs["predictions"]
    with pytest.raises(IndexError):
        select(table, predicts={len(X): 1})
    with pytest.raises(IndexError):
        select(table, predicts={-1: 1})
    with pytest.raises(ValueError, match="not a class"):
        select(table, predicts={i: 7})


def test_plot_tradeoff(small):
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    from splinter.viz import plot_tradeoff
    rset, X, y, table = small
    max_count = table.groupby(["aes", "acc"]).size().max()

    def markers(fig):
        filled, ring = [], []
        for collection in fig.axes[0].collections:
            (ring if len(collection.get_facecolors()) == 0 else filled).append(collection)
        return filled, ring

    def counts(collection, max_count):
        return (collection.get_sizes() - 20) / 180 * max_count

    fig = plot_tradeoff(table)
    (blue,), (ring,) = markers(fig)
    assert matplotlib.colors.to_hex(blue.get_facecolors()[0], keep_alpha=False) == "#0072b2"
    assert np.isclose(counts(blue, max_count).sum(), len(table))
    best = table.loc[table.objective.idxmin()]
    np.testing.assert_array_equal(ring.get_offsets()[0], [best.aes, best.acc])
    assert fig.axes[0].get_title() == f"{len(table)} trees"
    part = select(table, acc=(0.9, None))
    assert plot_tradeoff(part).axes[0].get_title() == f"{len(part)} of {len(table)} trees"

    chosen = select(table, aes=(None, 1.0))
    assert 0 < len(chosen) < len(table)
    chosen_flag = table.tree.isin(chosen.tree)
    split_max = table.assign(c=chosen_flag).groupby(["aes", "acc", "c"]).size().max()
    fig = plot_tradeoff(table, highlight=chosen)
    filled, _ = markers(fig)
    by_color = {matplotlib.colors.to_hex(c.get_facecolors()[0], keep_alpha=False): c for c in filled}
    assert np.isclose(counts(by_color["#0072b2"], split_max).sum(), len(chosen))
    assert np.isclose(sum(counts(c, split_max).sum() for c in filled), len(table))


def test_plot_tradeoff_sample_duplicates(small):
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    from splinter.viz import plot_tradeoff
    rset, X, y, table = small
    sample = rashomon_table(rset, X, y, max_trees=len(table) - 1, seed=0)
    assert sample.attrs["mode"] == "sample" and sample.tree.duplicated().any()
    chosen = select(sample, aes=(None, 1.5))
    assert chosen.tree.duplicated().any() and 0 < len(chosen) < len(sample)
    fig = plot_tradeoff(sample, highlight=chosen)
    assert fig.axes[0].get_title() == f"{len(sample)} sampled of {len(table)} trees"
    flag = sample.tree.isin(chosen.tree)
    split_max = sample.assign(c=flag).groupby(["aes", "acc", "c"]).size().max()
    filled = [c for c in fig.axes[0].collections if len(c.get_facecolors())]
    by_color = {matplotlib.colors.to_hex(c.get_facecolors()[0], keep_alpha=False): c for c in filled}
    blue = (by_color["#0072b2"].get_sizes() - 20) / 180 * split_max
    assert np.isclose(blue.sum(), flag.sum()) and flag.sum() == len(chosen)
    np.testing.assert_array_equal(tree_predictions(chosen),
                                  tree_predictions(sample)[flag.to_numpy()])
