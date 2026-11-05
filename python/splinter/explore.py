"""Inspect a fitted RESPLINTER Rashomon set as a table, one row per tree.

Average explanation size (AES): for each row, the number of distinct
features on its root-to-leaf path (a feature split twice counts once),
averaged over the rows passed in.

``min_leaf`` is the smallest number of training rows in any leaf of a tree.
"""

import numpy as np
import pandas as pd

from ._validation import _class_indices, prediction_data
from .tree import MWLeaf, MWShapeNode, count_leaves, leaf_loss, tree_depth

__all__ = ["explanation_size", "rashomon_table", "select", "tree_predictions"]

_COLUMNS = ["tree", "objective", "acc", "aes", "leaves", "min_leaf", "depth", "max_arity",
            "n_features", "root", "features"]
_RANGE_COLUMNS = ("objective", "acc", "aes", "leaves", "min_leaf", "depth", "max_arity",
                  "n_features")


def _route(tree, X):
    """Per row, the predicted class index and the distinct features on its path."""
    predictions = np.empty(len(X), dtype=np.intp)
    sizes = np.empty(len(X), dtype=np.intp)
    pending = [(tree, np.arange(len(X)), frozenset())]
    while pending:
        node, rows, features = pending.pop()
        if isinstance(node, MWLeaf):
            predictions[rows] = node.prediction
            sizes[rows] = len(features)
            continue
        branches = np.searchsorted(node.thresholds, X[rows, node.feature_index], side="right")
        if isinstance(node, MWShapeNode):
            branches = np.asarray(node.group_map, dtype=np.intp)[branches]
        if np.any(branches < 0) or np.any(branches >= len(node.children)):
            raise ValueError("Tree partition contains a branch without a child.")
        features = features | {int(node.feature_index)}
        for branch, child in enumerate(node.children):
            selected = rows[branches == branch]
            if selected.size:
                pending.append((child, selected, features))
    return predictions, sizes


def explanation_size(tree, X):
    """Number of distinct features on each row's root-to-leaf path.

    This is the length of the explanation a tree gives for a row: the
    features a reader must check to follow the row to its leaf.

    Parameters
    ----------
    tree : MWNode, MWShapeNode or MWLeaf
        A tree, e.g. ``model.tree_`` or ``RESPLINTER.get_tree(i)``.
    X : array-like of shape (n_samples, n_features)
        Rows to route.

    Returns
    -------
    sizes : ndarray of shape (n_samples,)
        Distinct features on each row's path (0 for a single-leaf tree).
    """
    return _route(tree, np.asarray(X, dtype=np.float64))[1]


def _features_used(node):
    if isinstance(node, MWLeaf):
        return set()
    return {int(node.feature_index)}.union(*(_features_used(child) for child in node.children))


def _min_leaf(node):
    """The smallest training ``n_samples`` over the tree's leaves."""
    if isinstance(node, MWLeaf):
        return int(node.n_samples)
    return min(_min_leaf(child) for child in node.children)


def _max_arity(node):
    """The largest number of children of any internal node; 0 for a single leaf."""
    if isinstance(node, MWLeaf):
        return 0
    return max(len(node.children), *(_max_arity(child) for child in node.children))


class _Predictions:
    """Predicted class index per (tree, row), keyed by tree index. Shared, never
    copied: pandas deep-copies ``attrs`` on most operations."""

    def __init__(self, index, matrix):
        self.index, self.matrix = index, matrix

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        return self

    def rows(self, trees):
        return self.matrix[[self.index[int(t)] for t in trees]]


def tree_predictions(table):
    """Predicted class index of each table row's tree for every row of the X given to
    ``rashomon_table``: a ``uint8`` array of shape ``(len(table), n_rows)``, aligned
    with ``table`` after any filtering (labels: ``classes[index]``)."""
    return table.attrs["predictions"].rows(table["tree"])


def rashomon_table(rset, X, y, *, max_trees=2000, seed=0):
    """Return a DataFrame with one row per tree of a fitted RESPLINTER's reported set.

    With at most ``max_trees`` reported trees, every one is listed in objective
    order (``attrs["mode"] == "all"``); otherwise ``max_trees`` uniform draws
    (with replacement, ``seed``) are listed (``"sample"``). ``acc`` and
    ``aes`` are measured on ``X, y``; ``min_leaf`` is the smallest training
    row count over a tree's leaves; ``tree`` is the index for ``get_tree``.
    Each distinct tree's predictions on every row of ``X`` are stored once
    (``tree_predictions``; ``len(X)`` bytes per tree), so pass a modest ``X``
    such as the test set; they stay aligned under any filtering.
    """
    X = prediction_data(rset, X, "dag_")
    y = _class_indices(y, rset.classes_)
    if len(y) != len(X):
        raise ValueError(f"X has {len(X)} rows but y has {len(y)}.")
    if len(rset.classes_) > 256:
        raise ValueError(f"Stored predictions are uint8; got {len(rset.classes_)} classes.")
    n_trees = int(rset.tree_count_at_target_)
    names = list(rset.feature_names_)
    if n_trees <= max_trees:
        mode = "all"
        records = [(i, rset.get_tree(i)) for i in range(n_trees)]
    else:
        mode = "sample"
        trees, indices = rset.sample_trees(max_trees, seed=seed, at_target=True,
                                           return_indices=True)
        records = list(zip(indices, trees))
    rows = []
    index, stored = {}, []
    row_cache = {}
    for i, tree in records:
        i = int(i)
        if i in row_cache:
            rows.append(row_cache[i])
            continue
        predictions, sizes = _route(tree, X)
        index[i] = len(stored)
        stored.append(predictions.astype(np.uint8))
        used = _features_used(tree)
        leaves = count_leaves(tree)
        root = None if isinstance(tree, MWLeaf) else names[int(tree.feature_index)]
        row = (i, leaf_loss(tree) + rset.reg * leaves,
               float(np.mean(predictions == y)) if len(y) else float("nan"),
               float(np.mean(sizes)) if len(sizes) else float("nan"),
               leaves, _min_leaf(tree), tree_depth(tree), _max_arity(tree),
               len(used), root, tuple(sorted(names[j] for j in used)))
        row_cache[i] = row
        rows.append(row)
    table = pd.DataFrame(rows, columns=_COLUMNS)
    table.attrs.update(mode=mode, n_trees=n_trees, n_rows=len(X),
                       feature_names=names, classes=np.asarray(rset.classes_),
                       predictions=_Predictions(
                           index, np.array(stored, dtype=np.uint8).reshape(len(stored), len(X))))
    return table


def _names(value):
    return [value] if isinstance(value, str) else list(value)


def select(table, *, uses=None, uses_any=None, avoids=None, root=None, predicts=None,
           **ranges):
    """Return a copy of the rows of ``table`` that meet every given condition.

    Each keyword ``column=(lo, hi)`` keeps values in the inclusive range (an
    end may be None); ``column=value`` keeps that value exactly. Columns:
    objective, acc, aes, leaves, min_leaf, depth, max_arity, n_features. ``uses`` needs
    every listed feature, ``uses_any`` at least one, ``avoids`` none; ``root``
    keeps trees whose root split is on one of the listed features.
    ``predicts={row: label}`` keeps trees that predict ``label`` for each
    ``row`` (a position in the X given to ``rashomon_table``). Examples::

        select(table, acc=(0.9, None), aes=(None, 2), depth=2)
        select(table, min_leaf=(5, None))
        select(table, uses="dose", avoids=["age", "noise1"], root="dose")
        select(table, predicts={7: 1})

    ``attrs`` are kept.
    """
    keep = np.ones(len(table), dtype=bool)
    unknown_keys = sorted(set(ranges) - set(_RANGE_COLUMNS))
    if unknown_keys:
        raise ValueError(f"Unknown select keys {unknown_keys}; allowed: "
                         f"{list(_RANGE_COLUMNS)} and uses, uses_any, avoids, root, predicts")
    for column, condition in ranges.items():
        values = table[column].to_numpy()
        if isinstance(condition, (tuple, list)):
            if len(condition) != 2:
                raise ValueError(f"{column}: a range is (lo, hi); got {condition!r}")
            lo, hi = condition
            if lo is not None:
                keep &= values >= lo
            if hi is not None:
                keep &= values <= hi
        else:
            keep &= values == condition
    named = {key: None if value is None else _names(value)
             for key, value in (("uses", uses), ("uses_any", uses_any), ("avoids", avoids),
                                ("root", root))}
    valid = set(table.attrs.get("feature_names", ())).union(*table["features"])
    unknown = [name for names in named.values() if names for name in names if name not in valid]
    if unknown:
        raise ValueError(f"Unknown feature names {unknown}; valid names: {sorted(valid)}")
    features = [set(f) for f in table["features"]]
    if named["uses"] is not None:
        keep &= np.array([set(named["uses"]) <= f for f in features], dtype=bool)
    if named["uses_any"] is not None:
        keep &= np.array([bool(set(named["uses_any"]) & f) for f in features], dtype=bool)
    if named["avoids"] is not None:
        keep &= np.array([not set(named["avoids"]) & f for f in features], dtype=bool)
    if named["root"] is not None:
        keep &= table["root"].isin(named["root"]).to_numpy()
    if predicts:
        if "predictions" not in table.attrs:
            raise ValueError("predicts needs a table from rashomon_table.")
        stored = tree_predictions(table)
        classes = np.asarray(table.attrs["classes"]).tolist()
        for row, label in predicts.items():
            if not 0 <= row < stored.shape[1]:
                raise IndexError(f"predicts row {row} is outside 0..{stored.shape[1] - 1}.")
            if label not in classes:
                raise ValueError(f"predicts label {label!r} is not a class; classes: {classes}")
            keep &= stored[:, row] == classes.index(label)
    out = table[keep].copy()
    out.attrs = dict(table.attrs)
    return out
