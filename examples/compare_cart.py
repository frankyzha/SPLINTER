"""SPLINTER vs. scikit-learn's CART decision tree on breast cancer, at matched depths.

Both trees may have leaves of any size (``min_samples_leaf=1``, CART's default).
SPLINTER uses a small leaf penalty (``reg=1e-4``) so that its trees grow with
the maximum depth; its other settings are the defaults. Each cell is the
mean ± standard deviation over 5 stratified train/test splits.
"""
import time

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

from splinter import SPLINTER
from splinter.explore import explanation_size


def cart_explanation_size(tree, X):
    """Distinct features on each row's root-to-leaf path in a fitted CART tree."""
    paths = tree.decision_path(X).tocsr()
    features = tree.tree_.feature  # negative at leaves
    sizes = []
    for i in range(X.shape[0]):
        on_path = features[paths.indices[paths.indptr[i]:paths.indptr[i + 1]]]
        sizes.append(np.unique(on_path[on_path >= 0]).size)
    return np.array(sizes)


def evaluate(make_model, describe, X, y, splits):
    """Test accuracy, leaves, depth and average explanation size, one row per split."""
    rows = []
    for train, test in splits:
        model = make_model().fit(X[train], y[train])
        rows.append((model.score(X[test], y[test]), *describe(model, X[test])))
    return np.array(rows)


def main():
    start = time.perf_counter()
    X, y = load_breast_cancer(return_X_y=True)
    splits = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=0).split(X, y))
    methods = {
        "SPLINTER": (lambda d: SPLINTER(max_depth=d, reg=1e-4, min_samples_leaf=1),
                     lambda m, X: (m.num_leaves(), m.tree_depth(),
                                   explanation_size(m.tree_, X).mean())),
        "CART": (lambda d: DecisionTreeClassifier(max_depth=d, random_state=0),
                 lambda m, X: (m.get_n_leaves(), m.get_depth(),
                               cart_explanation_size(m, X).mean())),
    }
    print(f"{'max depth':>9}  {'method':<9} {'test accuracy':>15} {'leaves':>12} "
          f"{'depth':>11} {'avg. explanation':>17}")
    for depth in (2, 3, 4):
        for name, (make, describe) in methods.items():
            r = evaluate(lambda: make(depth), describe, X, y, splits)
            mean, sd = r.mean(axis=0), r.std(axis=0, ddof=1)
            print(f"{depth:>9}  {name:<9} {mean[0]:>8.3f} ± {sd[0]:.3f} "
                  f"{mean[1]:>6.1f} ± {sd[1]:<4.1f}{mean[2]:>5.1f} ± {sd[2]:<4.1f}"
                  f"{mean[3]:>10.2f} ± {sd[3]:.2f}")
    print("\nAverage explanation size: distinct features on a test row's path, averaged.")
    print(f"done in {time.perf_counter() - start:.1f} s")


if __name__ == "__main__":
    main()
