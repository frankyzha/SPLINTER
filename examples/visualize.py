"""Draw fitted SPLINTER trees with Graphviz and write PNG and PDF files."""
import argparse
from pathlib import Path

import numpy as np
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.model_selection import train_test_split

from splinter import SPLINTER, ShapeSPLINTER
from splinter.viz import plot_tree

PLANS = ["basic", "standard", "premium", "family"]
TEASER = Path(__file__).resolve().parents[1] / "docs" / "images" / "teaser.png"


def churn_data(n=600, seed=0):
    """Plans 0..3 and tenure in months; basic/family plans churn early."""
    rng = np.random.default_rng(seed)
    plan, tenure = rng.integers(4, size=n), rng.integers(73, size=n)
    churn = np.isin(plan, [0, 3]) & (tenure < 12)
    churn ^= rng.random(n) < 0.05
    return np.column_stack([plan, tenure]).astype(float), np.where(churn, "churn", "stay")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", default="tree_figures")
    parser.add_argument("--teaser", action="store_true",
                        help="also write the README image docs/images/teaser.png")
    args = parser.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    figures = {}

    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target_names[data.target], stratify=data.target, random_state=42
    )
    settings = dict(max_depth=3, lookahead_depth=1, min_samples_leaf=5, lgbm_n_estimators=16)
    model = SPLINTER(**settings).fit(X_train, y_train, feature_names=data.feature_names)
    acc = model.score(X_test, y_test)
    figures["breast_cancer_splinter"] = plot_tree(
        model, X_test, y_test, title=f"SPLINTER · depth {model.tree_depth()}\ntest acc. {acc:.3f}")
    if args.teaser:
        small = SPLINTER(max_depth=2, lgbm_n_estimators=16).fit(
            X_train, y_train, feature_names=data.feature_names)
        teaser = plot_tree(small, X_test, y_test)
        teaser.attr(dpi="150")
        TEASER.parent.mkdir(parents=True, exist_ok=True)
        print(teaser.render(TEASER.with_suffix(""), format="png", cleanup=True))
    model = ShapeSPLINTER(**settings).fit(X_train, y_train, feature_names=data.feature_names)
    figures["breast_cancer_nonlinear"] = plot_tree(
        model, X_test, y_test, style="detailed", merge_leaves=True)

    X, y = churn_data()
    model = ShapeSPLINTER(max_depth=2).fit(X, y, feature_names=["plan", "tenure"])
    figures["churn_categorical"] = plot_tree(model, X, y, category_names={"plan": PLANS})

    data = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target_names[data.target], stratify=data.target, random_state=42
    )
    model = SPLINTER(max_depth=2).fit(X_train, y_train, feature_names=data.feature_names)
    figures["iris_multiclass"] = plot_tree(model, X_test, y_test)

    for name, dot in figures.items():
        for fmt in ("png", "pdf"):
            print(dot.render(outdir / name, format=fmt, cleanup=True))


if __name__ == "__main__":
    main()
