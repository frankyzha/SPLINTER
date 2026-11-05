"""Shared multiway tree records, routing, and statistics.

A value equal to a threshold goes to the child on the right of that
threshold (the rule of NumPy's ``searchsorted(..., side="right")``).
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class MWLeaf:
    prediction: int
    n_samples: int
    loss: float
    # Per-class training totals (ints, or float weight totals for a weighted
    # fit), set by `annotate_class_counts`; None when not recorded.
    class_counts: tuple | None = None


@dataclass
class MWNode:
    feature_index: int
    feature_name: str
    thresholds: list[float]
    children: list


@dataclass
class MWShapeNode:
    feature_index: int
    feature_name: str
    thresholds: list[float]
    group_map: list[int]
    children: list


def dict_to_tree(record):
    """Convert a tree dictionary returned by the Rust solver to tree records."""
    if record["type"] == "leaf":
        return MWLeaf(int(record["prediction"]), int(record["n_samples"]),
                      float(record["loss"]))
    feature = int(record["feature_index"])
    fields = dict(feature_index=feature,
                  feature_name=str(record.get("feature_name", f"x{feature}")),
                  thresholds=[float(t) for t in record["thresholds"]],
                  children=[dict_to_tree(c) for c in record["children"]])
    if record.get("group_map") is not None:
        return MWShapeNode(group_map=[int(g) for g in record["group_map"]], **fields)
    return MWNode(**fields)


def predict_encoded(tree, X):
    """Route rows in batches and return their zero-based class indices."""
    predictions = np.empty(len(X), dtype=np.intp)
    pending = [(tree, np.arange(len(X)))]
    while pending:
        node, rows = pending.pop()
        if isinstance(node, MWLeaf):
            predictions[rows] = node.prediction
            continue
        branches = np.searchsorted(node.thresholds, X[rows, node.feature_index], side="right")
        if isinstance(node, MWShapeNode):
            branches = np.asarray(node.group_map, dtype=np.intp)[branches]
        if np.any(branches < 0) or np.any(branches >= len(node.children)):
            raise ValueError("Tree partition contains a branch without a child.")
        for branch, child in enumerate(node.children):
            selected = rows[branches == branch]
            if selected.size:
                pending.append((child, selected))
    return predictions


def _leaf_rows(tree, X):
    """Yield (leaf, row indices) for each leaf reached by rows of ``X``,
    routing exactly as ``predict_encoded`` does."""
    pending = [(tree, np.arange(len(X)))]
    while pending:
        node, rows = pending.pop()
        if isinstance(node, MWLeaf):
            yield node, rows
            continue
        branches = np.searchsorted(node.thresholds, X[rows, node.feature_index], side="right")
        if isinstance(node, MWShapeNode):
            branches = np.asarray(node.group_map, dtype=np.intp)[branches]
        if np.any(branches < 0) or np.any(branches >= len(node.children)):
            raise ValueError("Tree partition contains a branch without a child.")
        for branch, child in enumerate(node.children):
            selected = rows[branches == branch]
            if selected.size:
                pending.append((child, selected))


def route_paths(tree, X):
    """Route rows as ``predict_encoded`` does and return, per row, the leaf
    reached and its path ``[(node, child index), ...]`` from the root."""
    leaves, paths = [None] * len(X), [None] * len(X)
    pending = [(tree, np.arange(len(X)), [])]
    while pending:
        node, rows, path = pending.pop()
        if isinstance(node, MWLeaf):
            for row in rows.tolist():
                leaves[row], paths[row] = node, path
            continue
        branches = np.searchsorted(node.thresholds, X[rows, node.feature_index], side="right")
        if isinstance(node, MWShapeNode):
            branches = np.asarray(node.group_map, dtype=np.intp)[branches]
        if np.any(branches < 0) or np.any(branches >= len(node.children)):
            raise ValueError("Tree partition contains a branch without a child.")
        for branch, child in enumerate(node.children):
            selected = rows[branches == branch]
            if selected.size:
                pending.append((child, selected, path + [(node, branch)]))
    return leaves, paths


def _leaves(tree):
    if isinstance(tree, MWLeaf):
        return [tree]
    return [leaf for child in tree.children for leaf in _leaves(child)]


def annotate_class_counts(tree, X, y_encoded, n_classes, weights=None):
    """Store each leaf's per-class training totals in ``leaf.class_counts``.

    Rows of ``X`` are routed as in ``predict_encoded``. Totals are ints, or
    float sums of ``weights`` when given; a leaf no row reaches gets zeros.
    ``prediction`` and ``loss`` are left unchanged.
    """
    y_encoded = np.asarray(y_encoded, dtype=np.intp)
    totals = {id(leaf): np.zeros(n_classes, dtype=np.float64 if weights is not None
                                 else np.int64) for leaf in _leaves(tree)}
    for leaf, rows in _leaf_rows(tree, X):
        totals[id(leaf)] += np.bincount(
            y_encoded[rows], weights=None if weights is None else weights[rows],
            minlength=n_classes)[:n_classes].astype(totals[id(leaf)].dtype)
    cast = float if weights is not None else int
    for leaf in _leaves(tree):
        leaf.class_counts = tuple(cast(v) for v in totals[id(leaf)])
    return tree


def predict_proba_encoded(tree, X, n_classes):
    """Class probabilities per row from the routed leaf's ``class_counts``.

    A leaf without counts (no training rows) gives one-hot on its
    ``prediction``. When the counts tie so that their argmax is not
    ``prediction`` (the solver breaks ties toward the higher class index),
    1e-12 is added to the ``prediction`` entry and the row renormalized, so
    the argmax of every row equals ``predict_encoded``.
    """
    probabilities = np.zeros((len(X), n_classes), dtype=np.float64)
    for leaf, rows in _leaf_rows(tree, X):
        counts = (np.asarray(leaf.class_counts, dtype=np.float64)
                  if leaf.class_counts is not None else np.zeros(n_classes))
        if counts.sum() > 0:
            row = counts / counts.sum()
        else:
            row = np.zeros(n_classes)
            row[leaf.prediction] = 1.0
        if int(np.argmax(row)) != leaf.prediction:
            row[leaf.prediction] += 1e-12
            row /= row.sum()
        probabilities[rows] = row
    return probabilities


def leaf_loss(tree):
    if isinstance(tree, MWLeaf):
        return tree.loss
    return sum(leaf_loss(child) for child in tree.children)


def count_leaves(tree):
    if isinstance(tree, MWLeaf):
        return 1
    return sum(count_leaves(child) for child in tree.children)


def count_internal_nodes(tree):
    if isinstance(tree, MWLeaf):
        return 0
    return 1 + sum(count_internal_nodes(child) for child in tree.children)


def tree_depth(tree):
    if isinstance(tree, MWLeaf):
        return 0
    return 1 + max(tree_depth(child) for child in tree.children)


def internal_sample_count(tree):
    """Sum internal-node supports; dividing by N gives mean path length."""
    # Each sample contributes once per ancestor, equivalently its leaf depth.
    total = 0
    pending = [(tree, 0)]
    while pending:
        node, depth = pending.pop()
        if isinstance(node, MWLeaf):
            total += node.n_samples * depth
        else:
            pending.extend((child, depth + 1) for child in node.children)
    return total


def objective(tree, reg, n_total, path_penalty=0.0, reg_alpha=1.0):
    return (leaf_loss(tree) + reg * reg_alpha * count_leaves(tree)
            + reg * (1.0 - reg_alpha) * count_internal_nodes(tree)
            + path_penalty * internal_sample_count(tree) / n_total)
