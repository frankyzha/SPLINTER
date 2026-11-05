"""Exact branch labels of multiway tree nodes, shared by text and drawings.

A value equal to a threshold goes to the child on the right of that
threshold (``splinter.tree``), so a child covers the range ``[a, b)``.
"""

from collections.abc import Mapping

import numpy as np

from .tree import MWShapeNode


def _cut(t, values=None):
    """Return (t, printed value, text) for a cutpoint, exact on ``values``."""
    v = 0.0 if abs(t) < 1e-20 else float(f"{t:.12g}")
    if values is not None and np.any((values >= t) != ((values > v) if t > v else (values >= v))):
        return t, t, repr(float(t))
    return t, v, "0" if v == 0 else f"{v:.12g}"


def _range_label(lo, hi):
    if lo is None and hi is None:
        return "any"
    if lo is None:
        return ("≤ " if hi[0] > hi[1] else "< ") + hi[2]
    if hi is None:
        return ("> " if lo[0] > lo[1] else "≥ ") + lo[2]
    return (("(" if lo[0] > lo[1] else "[") + lo[2] + ", " + hi[2]
            + ("]" if hi[0] > hi[1] else ")"))


def _child_of_bin(node):
    if isinstance(node, MWShapeNode):
        return np.asarray(node.group_map, dtype=np.intp)
    return np.arange(len(node.thresholds) + 1)


def _labels(node, category_names, values):
    """Per child, (label, True when the label lists category names)."""
    values = None if values is None else np.asarray(values, dtype=np.float64).ravel()
    cuts = [_cut(float(t), values) for t in node.thresholds]
    child_of_bin = _child_of_bin(node)
    routed = [[] for _ in node.children]
    if category_names is not None:
        if isinstance(category_names, Mapping):
            codes = sorted(category_names)
            names = [category_names[code] for code in codes]
        else:
            codes, names = range(len(category_names)), list(category_names)
        bins = np.searchsorted(node.thresholds, np.asarray(codes, dtype=np.float64), side="right")
        for child, name in zip(child_of_bin[bins], names):
            routed[child].append(str(name))
    labels = []
    for c in range(len(node.children)):
        if routed[c]:
            labels.append((", ".join(routed[c]), True))
            continue
        bins = np.flatnonzero(child_of_bin == c)
        runs = np.split(bins, np.flatnonzero(np.diff(bins) > 1) + 1) if bins.size else []
        parts = [_range_label(cuts[run[0] - 1] if run[0] > 0 else None,
                              cuts[run[-1]] if run[-1] < len(cuts) else None) for run in runs]
        labels.append((" ∪ ".join(parts) or "(empty)", False))
    return labels
