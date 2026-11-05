"""LicketySPLINTER: grow a tree one level at a time, with contiguous or nonlinear splits."""
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from splinter import LicketySPLINTER


def main():
    data = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target_names[data.target], test_size=0.3,
        stratify=data.target, random_state=0,
    )
    for shape_mode, label in ((False, "contiguous splits"), (True, "nonlinear splits")):
        model = LicketySPLINTER(
            max_depth=3, min_samples_leaf=3, shape_mode=shape_mode, lgbm_n_estimators=16,
        ).fit(X_train, y_train, feature_names=data.feature_names)
        print(f"LicketySPLINTER, {label}: test accuracy {model.score(X_test, y_test):.3f}, "
              f"{model.num_leaves()} leaves, depth {model.tree_depth()}")


if __name__ == "__main__":
    main()
