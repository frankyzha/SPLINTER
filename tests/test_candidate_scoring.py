"""Candidate upper bounds and native objectives must price their witnesses.

These checks route the original rows through the exported partition rules.
They intentionally do not use the estimators' objective/statistics helpers or
the loss values stored on native leaves. Each arity is checked independently;
changing the arity cap can legitimately change the kept candidate pool.
"""

from bisect import bisect_right
from collections import Counter

import numpy as np
import pytest

from splinter.multiway_solver import rashomon_select_debug, solve_branch_and_bound


def _fixture(n_classes):
    # A nonmonotone first feature identifies a class regime. The middle
    # value of the second feature rotates its label, requiring another
    # decision for an exact fit. Replicates keep supports above the minimum.
    X = np.repeat(np.array([(a, b) for a in range(6) for b in range(3)],
                           dtype=np.float64), 2, axis=0)
    y = ((X[:, 0].astype(int) % n_classes) + (X[:, 1] == 1)) % n_classes
    return X, np.ascontiguousarray(y, dtype=np.uint8)


def _thresholds(X):
    result = {}
    for j in range(X.shape[1]):
        occupied = sorted(set(X[:, j]))
        result[j] = [(a + b) / 2 for a, b in zip(occupied, occupied[1:])]
    return result


def _solve(X, y, *, shape, max_k, depth, closure_depth, path_penalty):
    return solve_branch_and_bound(
        x=np.ascontiguousarray(X), y=np.ascontiguousarray(y),
        n_classes=int(y.max()) + 1, reg=0.01, max_depth=depth, max_k=max_k,
        min_samples_leaf=2, feature_exclusion=False, time_limit=10.0,
        initial_used_features=[], feature_names=[f"x{j}" for j in range(X.shape[1])],
        thresholds_per_feature=_thresholds(X), dp_criterion="gini", beam_width=8,
        transition_delta=0.0, atom_mode=False, top_k=8, top_k_min_beam=1,
        reg_alpha=1.0, shape_mode=shape, shape_boundary_tax=0.15,
        shape_sort_per_atom_teacher_prior=shape,
        teacher_logit=np.ascontiguousarray(np.eye(int(y.max()) + 1)[y] * 4),
        path_penalty=path_penalty, closure_depth=closure_depth,
        closure_beam=3, closure_criterion="gini", closure_min_n_for_multiway=1,
    )


def _witness_objective(tree, X, y, *, reg, path_penalty):
    """Evaluate predictions and decision counts from the serialized rules."""
    leaves = []

    def collect(node):
        if node["type"] == "leaf":
            leaves.append(node)
        else:
            for child in node["children"]:
                collect(child)

    collect(tree)
    arrivals = Counter()
    mistakes = decisions = deepest = 0
    for row, label in zip(X, y):
        node, depth = tree, 0
        while node["type"] != "leaf":
            branch = bisect_right(node["thresholds"], float(row[node["feature_index"]]))
            if node.get("group_map") is not None:
                branch = node["group_map"][branch]
            node = node["children"][branch]
            depth += 1
        arrivals[id(node)] += 1
        mistakes += int(node["prediction"] != label)
        decisions += depth
        deepest = max(deepest, depth)
    # Native n_samples is part of the witness too: a score cannot silently
    # refer to a different support from the exported tree's partition.
    for leaf in leaves:
        assert leaf["n_samples"] == arrivals[id(leaf)]
        assert arrivals[id(leaf)] >= 2
    objective = mistakes / len(y) + reg * len(leaves) + path_penalty * decisions / len(y)
    return objective, deepest


@pytest.mark.parametrize("n_classes", [2, 3], ids=["binary", "multiclass"])
@pytest.mark.parametrize("shape", [False, True], ids=["contiguous", "nonlinear"])
@pytest.mark.parametrize("max_k", [3, 4], ids=["arity3", "arity4"])
@pytest.mark.parametrize("depth,closure_depth,path_penalty", [
    (1, 0, 0.0),       # Candidate DP and ordinary depth-zero leaf evaluation.
    (1, 1, 0.0),       # Depth-one closure shortcut.
    (1, 1, 0.035),     # Parent path surcharge with leaf children.
    (3, 2, 0.02),      # Lookahead prefix and recursive closure descendants.
], ids=["prefix", "closure", "closure_path_penalty", "deeper_closure"])
def test_native_objective_equals_exported_tree(n_classes, shape, max_k,
                                               depth, closure_depth, path_penalty):
    X, y = _fixture(n_classes)
    result = _solve(X, y, shape=shape, max_k=max_k, depth=depth,
                    closure_depth=closure_depth, path_penalty=path_penalty)
    assert not result["timed_out"]
    expected, deepest = _witness_objective(result["tree"], X, y, reg=0.01,
                                           path_penalty=path_penalty)
    assert result["objective"] == pytest.approx(expected, rel=0, abs=1e-12)
    assert 0 < deepest <= depth
    if depth > 1:
        assert deepest > 1  # Exercise internal descendants, not just a stump.


@pytest.mark.parametrize("n_classes", [2, 3], ids=["binary", "multiclass"])
@pytest.mark.parametrize("max_k", [3, 4], ids=["arity3", "arity4"])
def test_fragmented_nonlinear_split_prices_only_the_training_objective(n_classes, max_k):
    X = np.repeat(np.arange(6, dtype=np.float64), 6).reshape(-1, 1)
    y = np.ascontiguousarray(X[:, 0].astype(int) % n_classes, dtype=np.uint8)
    result = _solve(X, y, shape=True, max_k=max_k, depth=1,
                    closure_depth=1, path_penalty=0.02)
    assert not result["timed_out"]
    root = result["tree"]
    assert root["type"] != "leaf"
    groups = root.get("group_map")
    assert groups is not None
    changes = sum(a != b for a, b in zip(groups, groups[1:]))
    assert changes > len(root["children"]) - 1
    expected, _ = _witness_objective(root, X, y, reg=0.01, path_penalty=0.02)
    assert result["objective"] == pytest.approx(expected, rel=0, abs=1e-12)


@pytest.mark.parametrize("n_classes", [2, 3], ids=["binary", "multiclass"])
@pytest.mark.parametrize("shape", [False, True], ids=["contiguous", "nonlinear"])
@pytest.mark.parametrize("max_k", [3, 4], ids=["arity3", "arity4"])
def test_debug_split_upper_bound_equals_leaf_completion_cost(n_classes, shape, max_k):
    X, y = _fixture(n_classes)
    support = np.flatnonzero(X[:, 1] != 2).tolist()
    reg, path_penalty = 0.01, 0.035
    candidates = rashomon_select_debug(
        x=X, y=y, n_classes=n_classes, reg=reg, max_depth=3, max_k=max_k,
        rashomon_bound=1.0, support_idx=support, thresholds_per_feature=_thresholds(X),
        beam_width=8, transition_delta=0.0, atom_mode=False, shape_mode=shape,
        min_samples_leaf=2, path_penalty=path_penalty,
    )
    assert candidates
    for candidate in candidates:
        child_labels = [[] for _ in range(candidate["k"])]
        for i in support:
            branch = bisect_right(candidate["thresholds"], float(X[i, candidate["feature"]]))
            if candidate.get("group_map") is not None:
                branch = candidate["group_map"][branch]
            child_labels[branch].append(int(y[i]))
        mistakes = 0
        for branch, labels in enumerate(child_labels):
            counts = [labels.count(c) for c in range(n_classes)]
            assert candidate["child_n"][branch] == len(labels)
            assert candidate["child_counts"][branch] == counts
            mistakes += len(labels) - max(counts)
        expected = (mistakes / len(y) + reg * candidate["k"]
                    + path_penalty * len(support) / len(y))
        assert candidate["split_ub"] == pytest.approx(expected, rel=0, abs=1e-12)
