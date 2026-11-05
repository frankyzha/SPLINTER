"""Graphviz drawings of fitted SPLINTER trees, and matplotlib plots of Rashomon sets.

Edge labels follow the routing rule of ``splinter.tree``: a value equal to a
threshold goes to the right child, so a child covers the range ``[a, b)``.
"""

import copy
import re
import textwrap
from fractions import Fraction

import numpy as np
from sklearn.utils.validation import check_array

from ._labels import _child_of_bin, _labels
from ._validation import _class_indices, prediction_data
from .tree import MWLeaf, MWNode, MWShapeNode, _leaves

__all__ = ["plot_tree", "edge_labels", "merge_adjacent_leaves", "plot_rid", "plot_rid_cdf",
           "plot_multiplicity", "multiplicity_summary", "plot_tradeoff"]

_PAD = " "
_LEAF_STYLE = dict(shape="box", style="filled,rounded", color="black",
                   penwidth="1.8", width="0.9", height="0.5")
_NODE_STYLE = dict(shape="box", style="dashed,rounded", fillcolor="white",
                   color="gray55", penwidth="3.0", width="1.1", height="0.5")
_BINARY = [(67, 147, 195), (214, 96, 77)]
_OKABE_ITO = ["#0072B2", "#D55E00", "#009E73", "#E69F00", "#CC79A7", "#56B4E9", "#F0E442"]


def edge_labels(node, *, category_names=None, values=None):
    """Return one label per child of ``node``, in child order.

    ``category_names`` (names by code, or ``{code: name}``) lists the codes
    routed to each child; ``values`` (the feature's data) prints a cutpoint at
    full precision whenever the 12-digit text would misstate a value.
    """
    return [label for label, _ in _labels(node, category_names, values)]


def _join(a, b):
    return MWLeaf(a.prediction, a.n_samples + b.n_samples, a.loss + b.loss)


def _merge(node):
    if isinstance(node, MWLeaf):
        return node
    children = [_merge(child) for child in node.children]
    if isinstance(node, MWShapeNode):
        return _merge_shape(node, children)
    kept, thresholds = [children[0]], []
    for t, child in zip(node.thresholds, children[1:]):
        last = kept[-1]
        if isinstance(last, MWLeaf) and isinstance(child, MWLeaf) and last.prediction == child.prediction:
            kept[-1] = _join(last, child)
        else:
            kept.append(child)
            thresholds.append(float(t))
    if len(kept) == 1:
        return kept[0]
    return MWNode(node.feature_index, node.feature_name, thresholds, kept)


def _merge_shape(node, children):
    kept, slot_of, new_slot = [], {}, []
    for child in children:
        if isinstance(child, MWLeaf) and child.prediction in slot_of:
            slot = slot_of[child.prediction]
            kept[slot] = _join(kept[slot], child)
        else:
            slot = len(kept)
            kept.append(child)
            if isinstance(child, MWLeaf):
                slot_of[child.prediction] = slot
        new_slot.append(slot)
    if len(kept) == 1:
        return kept[0]
    group_map = [new_slot[int(g)] for g in node.group_map]
    thresholds, groups = [], group_map[:1]
    for t, group in zip(node.thresholds, group_map[1:]):
        if group != groups[-1]:
            thresholds.append(float(t))
            groups.append(group)
    if groups == list(range(len(kept))):
        return MWNode(node.feature_index, node.feature_name, thresholds, kept)
    return MWShapeNode(node.feature_index, node.feature_name, thresholds, groups, kept)


def merge_adjacent_leaves(tree):
    """Return a copy with same-class sibling leaves merged; predictions are unchanged.

    Contiguous nodes merge runs of neighbouring leaves; nonlinear nodes merge all
    leaves of a class and drop cutpoints between bins that now share a child.
    """
    return _merge(copy.deepcopy(tree))


def _wrap(text, wrap, **kwargs):
    return "\n".join(textwrap.wrap(text, wrap, break_long_words=False, **kwargs)) or text


def _wrap_categories(text, wrap):
    """Wrap a category list only after its ", " separators."""
    protected = re.sub(r"(?<!,) ", "\0", text)
    return _wrap(protected, wrap, break_on_hyphens=False).replace("\0", " ")


def _fill(prediction, n_classes, counts, n):
    if n_classes <= 2:
        base = _BINARY[prediction]
    else:
        base = tuple(bytes.fromhex((_OKABE_ITO[prediction] if prediction < len(_OKABE_ITO)
                                    else "#999999")[1:]))
    strength = 0.6
    if counts is not None:
        chance = 1.0 / max(n_classes, 2)
        strength = float(np.clip((counts[prediction] / n - chance) / (1.0 - chance), 0.0, 1.0))
    return "#" + "".join(f"{int(255 - strength * (255 - channel)):02x}" for channel in base)


def plot_tree(tree, X=None, y=None, *, feature_names=None, class_names=None,
              classes=None, category_names=None, style="simple", title=None,
              merge_leaves=False, wrap=14):
    """Draw a fitted estimator or a tree record and return a ``graphviz.Digraph``.

    With ``X`` (and ``y``), nodes show the routed row counts (and leaf colours
    show class purity); without it, the build-time leaf counts. ``classes``
    maps a bare record's class indices to labels. ``category_names`` is
    ``{feature index or name: names by code}``.
    """
    try:
        import graphviz
    except ImportError as error:
        raise ImportError("plot_tree needs the graphviz package: pip install "
                          "'splinter[visualization]' (and the system Graphviz 'dot' binary)") from error
    if style not in ("simple", "detailed"):
        raise ValueError(f"style must be 'simple' or 'detailed'; got {style!r}.")
    if hasattr(tree, "tree_"):
        if classes is not None:
            raise ValueError("classes= is only for tree records; a fitted model uses its classes_.")
        model, tree, classes = tree, tree.tree_, tree.classes_
        if X is not None:
            X = prediction_data(model, X)
    elif not isinstance(tree, (MWLeaf, MWNode, MWShapeNode)):
        raise TypeError(f"plot_tree needs a fitted SPLINTER estimator or a tree record; "
                        f"got {type(tree).__name__}.")
    elif X is not None:
        X = check_array(X, dtype=np.float64)
    if classes is not None:
        classes = np.asarray(classes)
    if y is not None:
        if X is None:
            raise ValueError("y needs X: pass the rows that y labels.")
        y = _class_indices(y, classes)
        if len(y) != len(X):
            raise ValueError(f"X has {len(X)} rows but y has {len(y)}.")
    if merge_leaves:
        tree = merge_adjacent_leaves(tree)
    if classes is not None:
        n_classes = len(classes)
    else:
        n_classes = int(max([leaf.prediction for leaf in _leaves(tree)]
                            + list(y if y is not None else []))) + 1
    if class_names is None:
        class_names = [str(c) for c in classes] if classes is not None else [str(i) for i in range(n_classes)]
    category_names = category_names or {}

    def display_name(node):
        name = feature_names[node.feature_index] if feature_names is not None else node.feature_name
        return _wrap(str(name).replace("_", " "), wrap)

    def categories(node):
        keys = [node.feature_index, node.feature_name]
        if feature_names is not None:
            keys.append(feature_names[node.feature_index])
        return next((category_names[key] for key in keys if key in category_names), None)

    graph = {"rankdir": "TB", "pad": "0.02", "nodesep": "0.08", "ranksep": "0.45"}
    if title is not None:
        graph.update(label=title, labelloc="t", fontname="Helvetica Bold", fontsize="16")
    dot = graphviz.Digraph(
        graph_attr=graph,
        node_attr={"margin": "0.12,0.05", "fontname": "Helvetica", "fontsize": "14"},
        edge_attr={"fontname": "Helvetica", "fontsize": "13"})

    def add(node, node_id, rows, parent_id=None, edge_label=None):
        if rows is not None:
            n = len(rows)
        else:
            n = sum(leaf.n_samples for leaf in _leaves(node))
        if isinstance(node, MWLeaf):
            p = int(node.prediction)
            counts = np.bincount(y[rows], minlength=n_classes) if y is not None else None
            label = f"{class_names[p]}\nn = {n}"
            if style == "detailed" and counts is not None:
                label += "\n[" + ", ".join(str(c) for c in counts) + "]"
            fill = "#f0f0f0" if rows is not None and n == 0 else _fill(p, n_classes, counts, n)
            dot.node(node_id, label=label, fillcolor=fill, **_LEAF_STYLE)
        else:
            label = display_name(node)
            if style == "detailed":
                label += f"\n{len(node.children)}-way"
                label += " · nonlinear" if isinstance(node, MWShapeNode) else ""
                label += f"\nn = {n}"
            dot.node(node_id, label=label, **_NODE_STYLE)
        if parent_id is not None:
            dot.edge(parent_id, node_id, label=edge_label)
        if isinstance(node, MWLeaf):
            return
        values = X[:, node.feature_index] if X is not None else None
        labels = _labels(node, categories(node), values)
        if rows is not None:
            branches = _child_of_bin(node)[np.searchsorted(
                node.thresholds, X[rows, node.feature_index], side="right")]
        for c, (child, (text, is_category)) in enumerate(zip(node.children, labels)):
            text = _wrap_categories(text, wrap) if is_category else text
            add(child, f"{node_id}_{c}", rows[branches == c] if rows is not None else None,
                node_id, _PAD + text + _PAD)

    add(tree, "n", np.arange(len(X)) if X is not None else None)
    return dot


# ----------------------------------------------------------- Rashomon plots
_ACCENT, _GRAY = "#0072B2", "#999999"
_RC = {"font.size": 9, "axes.labelsize": 9, "axes.titlesize": 10, "xtick.labelsize": 8,
       "ytick.labelsize": 8, "legend.fontsize": 8, "axes.linewidth": 0.8,
       "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False}


def _matplotlib(name):
    try:
        import matplotlib
        from matplotlib.figure import Figure
    except ImportError as error:
        raise ImportError(f"{name} needs matplotlib: pip install 'splinter[visualization]'") from error
    return matplotlib, Figure


def _new_axes(Figure, ax, size):
    """``ax`` and its figure; a new (non-pyplot) figure of ``size`` when ``ax`` is None."""
    if ax is None:
        ax = Figure(figsize=size, layout="constrained").add_subplot()
    ax.spines[["top", "right"]].set_visible(False)
    return ax.figure, ax


def _rid_names(rid, feature_names):
    names = [f"x{j}" for j in range(len(rid.values))] if feature_names is None else list(feature_names)
    if len(names) != len(rid.values):
        raise ValueError(f"feature_names has {len(names)} names; the RID has {len(rid.values)} features.")
    return names


def _by_mean(rid):
    """Indices of features with a non-empty distribution, largest mean first."""
    kept = [j for j in range(len(rid.values)) if len(rid.values[j])]
    return sorted(kept, key=lambda j: -rid.mean(j))


def plot_rid(rid, feature_names=None, *, top=None, ax=None):
    """Plot the variable importances in a ``splinter.rid.RID``: for each feature,
    the mean (dot), the interquartile range (thick line) and the 5-95% range
    (thin line).

    Rows are sorted by mean, largest on top; ``top=k`` keeps the k largest.
    Features with an empty distribution are skipped. Returns the Figure.
    """
    matplotlib, Figure = _matplotlib("plot_rid")
    names = _rid_names(rid, feature_names)
    rows = _by_mean(rid)[:top]
    ys = np.arange(len(rows))[::-1]
    with matplotlib.rc_context(_RC):
        fig, ax = _new_axes(Figure, ax, (4.5, 0.6 + 0.28 * len(rows)))
        # Exact fractions: Fraction(0.05) is slightly above 1/20.
        quantile = lambda a, b: [rid.quantile(j, Fraction(a, b)) for j in rows]
        ax.hlines(ys, quantile(1, 20), quantile(19, 20), color=_GRAY, lw=0.8)
        ax.hlines(ys, quantile(1, 4), quantile(3, 4), color=_ACCENT, lw=3.5, alpha=0.45)
        ax.plot([rid.mean(j) for j in rows], ys, "o", color=_ACCENT, ms=5, zorder=3)
        ax.set_yticks(ys, [str(names[j]).replace("_", " ") for j in rows])
        ax.set_ylim(-0.6, len(rows) - 0.4)
        ax.tick_params(axis="y", length=0)
        ax.set_xlabel("Importance (loss increase)")
        ax.set_title("Rashomon importance")
    return fig


def _labels_fit(ends, labels, span, axis_inches, pt=8):
    """True when right-end labels (centred on ``ends``) are far enough apart not to overlap."""
    order = np.argsort(ends)
    width = [0.6 * pt / 72 * len(labels[i]) / axis_inches * span for i in order]
    gaps = np.diff(np.asarray(ends)[order])
    return all(gap > (a + b) / 2 + 0.02 * span for gap, a, b in zip(gaps, width, width[1:]))


def plot_rid_cdf(rid, feature_names=None, *, features=None, ax=None):
    """Step CDFs of the importance of ``features`` (indices or names; default:
    the top 4 by mean, at most 6), one line per feature. Returns the Figure.

    Lines are labelled at their right end when the labels do not overlap;
    otherwise a legend names them.
    """
    matplotlib, Figure = _matplotlib("plot_rid_cdf")
    names = _rid_names(rid, feature_names)
    if features is None:
        features = _by_mean(rid)[:4]
    features = [names.index(f) if isinstance(f, str) else int(f) for f in features]
    if not features:
        raise ValueError("plot_rid_cdf needs at least one feature with a non-empty distribution.")
    if len(features) > 6:
        raise ValueError(f"plot_rid_cdf draws at most 6 features; got {len(features)}.")
    labels = [str(names[j]).replace("_", " ") for j in features]
    with matplotlib.rc_context(_RC):
        fig, ax = _new_axes(Figure, ax, (4.5, 3.0))
        ends = []
        for color, j, label in zip(_OKABE_ITO, features, labels):
            values = np.asarray(rid.values[j], dtype=np.float64)
            if values.size == 0:
                raise ValueError(f"feature {names[j]!r} has an empty importance distribution.")
            cdf = np.cumsum(rid.weights[j])
            ax.step(np.r_[values[0], values], np.r_[0.0, cdf], where="post", color=color,
                    lw=1.5, label=label)
            ends.append(values[-1])
        lo, hi = min(0.0, *[rid.values[j][0] for j in features]), max(ends)
        span = (hi - lo) or 1.0
        ax.set_xlim(lo - 0.03 * span, hi + 0.03 * span)
        width = fig.get_size_inches()[0] * ax.get_position().width
        if _labels_fit(ends, labels, span, width):
            for color, x, label in zip(_OKABE_ITO, ends, labels):
                ax.text(x, 1.03, label, color=color, ha="center", va="bottom", fontsize=8)
            ax.set_ylim(0, 1.15)
        else:
            ax.legend(frameon=False, loc="lower right")
            ax.set_ylim(0, 1.03)
        ax.set_yticks([0, 0.5, 1])
        ax.set_xlabel("Importance (loss increase)")
        ax.set_ylabel("Share of models ≤ x")
        ax.set_title("Importance distribution")
    return fig


def multiplicity_summary(rset, X):
    """Per-row share ``p`` of the reported Rashomon set of a fitted RESPLINTER
    predicting ``classes_[1]`` (exact), the set size, and the shares of rows
    on which the trees disagree (0 < p < 1) or all agree. ``conflicting``
    marks those rows from exact integer counts, so it stays exact for any
    set size."""
    counts, n, _ = rset.predict_class1_counts(X)
    if len(rset.classes_) != 2:
        raise ValueError(f"Multiplicity needs a binary Rashomon set; got {len(rset.classes_)} classes.")
    p = np.array([c / n for c in counts], dtype=np.float64) if n else np.zeros(len(counts))
    conflicting = np.array([0 < c < n for c in counts], dtype=bool)
    share = float(conflicting.mean()) if len(p) else 0.0
    return {"n_trees": int(rset.tree_count_at_target_), "share_conflicting": share,
            "share_unanimous": 1.0 - share, "p": p, "conflicting": conflicting}


def plot_multiplicity(rset, X, *, ax=None):
    """Histogram, over the rows of ``X``, of the share of trees in the reported
    Rashomon set predicting ``classes_[1]``. Returns the Figure.

    Rows where every tree agrees (share exactly 0 or 1) are counted separately
    and drawn as grey end bars just outside [0, 1]; the 18 blue bars split
    (0, 1) evenly and hold the rows where the trees disagree.
    """
    matplotlib, Figure = _matplotlib("plot_multiplicity")
    summary = multiplicity_summary(rset, X)
    p, conflicting = summary["p"], summary["conflicting"]
    counts, edges = np.histogram(np.clip(p[conflicting], 0, 1), bins=np.linspace(0, 1, 19))
    agree_0, agree_1 = np.sum(~conflicting & (p < 0.5)), np.sum(~conflicting & (p >= 0.5))
    with matplotlib.rc_context(_RC):
        fig, ax = _new_axes(Figure, ax, (4.5, 3.0))
        ax.bar([-0.045, 1.045], [agree_0, agree_1], width=0.05, color=_GRAY)
        ax.bar(edges[:-1], counts, width=np.diff(edges), align="edge", color=_ACCENT,
               edgecolor="white", lw=0.5)
        ax.set_xlim(-0.09, 1.09)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1], ["0", "0.25", "0.5", "0.75", "1"])
        ax.set_xlabel(f"Share of trees predicting {rset.classes_[1]}")
        ax.set_ylabel("Rows")
        ax.set_title(f"{summary['share_conflicting']:.0%} of rows get conflicting predictions")
    return fig


_AXIS_LABELS = {"aes": "Avg. explanation size", "acc": "Accuracy", "objective": "Objective",
                "leaves": "Leaves", "n_features": "Features used"}


def plot_tradeoff(table, *, x="aes", y="acc", highlight=None, ax=None):
    """Scatter plot of the trees in a ``splinter.explore.rashomon_table``, ``x`` against ``y``.

    One marker per distinct (x, y) point, with area growing with the number
    of trees there. Trees in ``highlight`` (a table, matched on ``tree``) are
    blue and the rest grey; without ``highlight`` all are blue. A black ring
    marks the tree with the lowest objective. Returns the Figure.
    """
    matplotlib, Figure = _matplotlib("plot_tradeoff")
    if len(table) == 0:
        raise ValueError("plot_tradeoff needs a table with at least one tree.")
    chosen = (np.ones(len(table), dtype=bool) if highlight is None
              else table["tree"].isin(highlight["tree"]).to_numpy())
    points = (table.assign(_chosen=chosen).groupby([x, y, "_chosen"]).size()
              .rename("count").reset_index())
    max_count = points["count"].max()
    area = lambda count: 20 + 180 * count / max_count
    best = table.loc[table["objective"].idxmin()]
    at_best = points[(points[x] == best[x]) & (points[y] == best[y])]["count"].max()
    label = lambda column: _AXIS_LABELS.get(column, column.replace("_", " "))
    n_trees = table.attrs.get("n_trees", len(table))
    with matplotlib.rc_context(_RC):
        fig, ax = _new_axes(Figure, ax, (4.5, 3.2))
        for flag, color, z in ((False, _GRAY, 2), (True, _ACCENT, 3)):
            group = points[points["_chosen"] == flag]
            if len(group):
                ax.scatter(group[x], group[y], s=area(group["count"]), color=color,
                           lw=0, zorder=z)
        ax.scatter([best[x]], [best[y]], s=(np.sqrt(area(at_best)) + 5) ** 2,
                   facecolors="none", edgecolors="black", linewidths=1.2, zorder=4)
        ax.margins(0.08)
        ax.set_xlabel(label(x))
        ax.set_ylabel(label(y))
        if table.attrs.get("mode") == "sample":
            ax.set_title(f"{len(table)} sampled of {n_trees} trees")
        else:
            ax.set_title(f"{len(table)} trees" if len(table) == n_trees
                         else f"{len(table)} of {n_trees} trees")
    return fig
