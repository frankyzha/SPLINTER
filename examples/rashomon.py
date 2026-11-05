"""Fit a Rashomon set with RESPLINTER and inspect it."""
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from splinter import RESPLINTER, export_text


def main():
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target_names[data.target], test_size=0.25,
        stratify=data.target, random_state=42,
    )
    # Every depth-2 tree whose objective is within 5% of the best one.
    model = RESPLINTER(max_depth=2, theta=0.05, lgbm_n_estimators=16).fit(
        X_train, y_train, feature_names=data.feature_names)
    print(f"{model.tree_count_} trees in the Rashomon set\n")
    print(model.summary())

    print("\nThe best tree:\n")
    print(export_text(model.get_tree(0), classes=model.classes_))

    report = model.guarantee()
    print(f"\nObjective bound: {report['objective_bound']:.4f}")
    print(f"Set complete for the candidate splits kept by the search: {report['complete']}")
    print(f"Best tree proven optimal: {report['best_tree_certified_optimal']}")
    print(f"\nMajority vote of all trees, test accuracy: {model.score(X_test, y_test):.3f}")


if __name__ == "__main__":
    main()
