"""Fit SPLINTER and ShapeSPLINTER on breast cancer and compare their trees."""
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from splinter import SPLINTER, ShapeSPLINTER
from splinter.explore import explanation_size


def main():
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target_names[data.target], test_size=0.25,
        stratify=data.target, random_state=42,
    )
    # A small LightGBM model (16 rounds) keeps the example fast.
    settings = dict(max_depth=3, lookahead_depth=1, min_samples_leaf=5, lgbm_n_estimators=16)
    print(f"{'model':<14} {'test accuracy':>13} {'leaves':>6} {'depth':>5} {'avg. explanation':>16}")
    models = {}
    for estimator in (SPLINTER, ShapeSPLINTER):
        model = estimator(**settings).fit(X_train, y_train, feature_names=data.feature_names)
        aes = explanation_size(model.tree_, X_test).mean()
        print(f"{estimator.__name__:<14} {model.score(X_test, y_test):>13.3f} "
              f"{model.num_leaves():>6} {model.tree_depth():>5} {aes:>16.2f}")
        models[estimator.__name__] = model
    print("\nAverage explanation size: distinct features on a test row's path, averaged.")
    print("\nThe SPLINTER tree:\n")
    print(models["SPLINTER"])


if __name__ == "__main__":
    main()
