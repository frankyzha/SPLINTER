"""Read a SPLINTER tree as text and explain individual predictions."""
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from splinter import SPLINTER, explain, export_text


def condition(feature, label):
    """'worst perimeter > 116.95' for a one-sided branch, 'x in [a, b)' for a range."""
    if label[0] in "<>≤≥":
        return f"{feature} {label}"
    return f"{feature} in {label}"


def main():
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target_names[data.target], test_size=0.25,
        stratify=data.target, random_state=42,
    )
    model = SPLINTER(max_depth=3, lookahead_depth=1, min_samples_leaf=5,
                     lgbm_n_estimators=16).fit(X_train, y_train, feature_names=data.feature_names)
    print(export_text(model))
    print()

    rows = X_test[:3]
    proba = model.predict_proba(rows)
    for i, (row_proba, reason) in enumerate(zip(proba, explain(model, rows))):
        label = reason["prediction"]
        share = row_proba[list(model.classes_).index(label)]
        because = " and ".join(condition(f, c) for f, c in reason["path"])
        print(f"Test row {i}: predicted {label} (probability {share:.2f}) because {because}. "
              f"Its leaf holds {reason['leaf_n']} training rows. True class: {y_test[i]}.")


if __name__ == "__main__":
    main()
