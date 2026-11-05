"""Fitted multiway trees as text: an indented tree, and the rules behind each prediction.

Branch conditions are the exact labels of ``splinter.viz.edge_labels``: a
value equal to a threshold goes to the right child, so a child covers the
range ``[a, b)``. A RESPLINTER tree is passed as a record:
``export_text(model.get_tree(i), classes=model.classes_)``.
"""

import numpy as np
from sklearn.utils.validation import check_array

from ._labels import _labels
from ._validation import prediction_data
from .tree import MWLeaf, MWNode, MWShapeNode, route_paths

__all__ = ["export_text", "explain"]


def _resolve(model_or_tree, classes, X):
    """(tree, classes or None, X as a float array or None)."""
    if hasattr(model_or_tree, "tree_"):
        if classes is not None:
            raise ValueError("classes= is only for tree records; a fitted model uses its classes_.")
        model = model_or_tree
        return (model.tree_, model.classes_,
                None if X is None else prediction_data(model, X))
    if not isinstance(model_or_tree, (MWLeaf, MWNode, MWShapeNode)):
        hint = (" For RESPLINTER pass model.get_tree(i) and classes=model.classes_."
                if hasattr(model_or_tree, "get_tree") else "")
        raise TypeError("Expected a fitted single-tree estimator or a tree record; "
                        f"got {type(model_or_tree).__name__}.{hint}")
    return (model_or_tree, None if classes is None else np.asarray(classes),
            None if X is None else check_array(X, dtype=np.float64))


class _Namer:
    """Feature names, category names and per-node branch labels."""

    def __init__(self, feature_names, category_names, X):
        self.feature_names = feature_names
        self.category_names = category_names or {}
        self.X = X
        self.cache = {}

    def feature(self, node):
        if self.feature_names is not None:
            return str(self.feature_names[node.feature_index])
        return node.feature_name

    def labels(self, node):
        if id(node) not in self.cache:
            keys = [node.feature_index, node.feature_name]
            if self.feature_names is not None:
                keys.append(self.feature_names[node.feature_index])
            categories = next((self.category_names[key] for key in keys
                               if key in self.category_names), None)
            values = None if self.X is None else self.X[:, node.feature_index]
            self.cache[id(node)] = [label for label, _ in _labels(node, categories, values)]
        return self.cache[id(node)]


def _count_text(value):
    return str(value) if isinstance(value, (int, np.integer)) else f"{value:.6g}"


def export_text(model_or_tree, *, classes=None, class_names=None, feature_names=None,
                category_names=None, X=None, show_counts=True):
    """Return the tree as indented text, one line per branch.

    An internal node prints its feature and arity; each branch prints the
    exact range (or, for nonlinear nodes, the union of ranges) of values
    routed to it; a leaf prints its class, its training row count ``n`` and,
    with ``show_counts``, its per-class training totals when recorded.

    ``model_or_tree`` is a fitted SPLINTER, ShapeSPLINTER or LicketySPLINTER,
    or a tree record (``MWLeaf`` / ``MWNode`` / ``MWShapeNode``, e.g.
    ``rset.get_tree(i)``); ``classes`` maps a record's class indices to
    labels (without it, indices are printed). ``class_names`` overrides the
    printed class labels, by class index. ``feature_names`` overrides the
    node feature names, by feature index. ``category_names`` is ``{feature
    index or name: names by code}``. With ``X``, a cutpoint whose 12-digit
    text would misstate a value of ``X`` is printed at full precision.
    """
    tree, classes, X = _resolve(model_or_tree, classes, X)
    namer = _Namer(feature_names, category_names, X)

    def class_label(index):
        if class_names is not None:
            return str(class_names[index])
        return str(classes[index]) if classes is not None else str(index)

    def describe(node):
        if isinstance(node, MWLeaf):
            text = f"class = {class_label(node.prediction)}  (n = {node.n_samples}"
            if show_counts and node.class_counts is not None:
                text += "; " + ", ".join(f"{class_label(c)}: {_count_text(v)}"
                                         for c, v in enumerate(node.class_counts))
            return text + ")"
        return f"{namer.feature(node)}: {len(node.children)}-way split"

    lines = [describe(tree)]

    def visit(node, prefix):
        if isinstance(node, MWLeaf):
            return
        labels = namer.labels(node)
        for c, (child, label) in enumerate(zip(node.children, labels)):
            last = c == len(node.children) - 1
            lines.append(f"{prefix}{'└─ ' if last else '├─ '}{label} → {describe(child)}")
            visit(child, prefix + ("   " if last else "│  "))

    visit(tree, "")
    return "\n".join(lines)


def explain(model_or_tree, X, *, classes=None, feature_names=None, category_names=None):
    """Explain each row's prediction by the branch conditions on its path.

    Returns one dict per row of ``X``: ``prediction`` (the class label, or
    the class index for a record without ``classes``), ``path`` (a list of
    ``(feature name, condition)`` from the root, where the condition is the
    exact label of the branch the row took, as in ``export_text``) and
    ``leaf_n`` (the training row count of the leaf reached). Arguments are
    as in ``export_text``.
    """
    tree, classes, X = _resolve(model_or_tree, classes, X)
    namer = _Namer(feature_names, category_names, X)
    leaves, paths = route_paths(tree, X)
    out = []
    for leaf, path in zip(leaves, paths):
        prediction = leaf.prediction
        if classes is not None:
            prediction = classes[prediction]
            prediction = prediction.item() if isinstance(prediction, np.generic) else prediction
        out.append({"prediction": prediction,
                    "path": [(namer.feature(node), namer.labels(node)[branch])
                             for node, branch in path],
                    "leaf_n": int(leaf.n_samples)})
    return out
