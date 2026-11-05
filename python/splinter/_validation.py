"""Input validation shared by the public Python estimators."""

from numbers import Integral, Real

import numpy as np
from sklearn.utils import check_X_y
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_array, check_is_fitted


def integer_parameter(name, value, minimum):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}; got {value!r}.")


def real_parameter(name, value, minimum=0.0, *, strict=False):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
            or not np.isfinite(value) or value < minimum or (strict and value == minimum)):
        relation = ">" if strict else ">="
        raise ValueError(f"{name} must be finite and {relation} {minimum}; got {value!r}.")


def validate_parameters(estimator):
    for name, minimum in (("max_depth", 0), ("max_k", 2), ("min_samples_leaf", 1),
                          ("beam_width", 1), ("max_bins", 2), ("lgbm_n_estimators", 1),
                          ("lgbm_min_samples_leaf", 1), ("hop_size", 1),
                          ("closure_beam", 1), ("cart_closure_beam", 1)):
        if hasattr(estimator, name):
            value = getattr(estimator, name)
            if value is None and name in getattr(estimator, "_none_allowed_parameters", ()):
                continue
            integer_parameter(name, value, minimum)
    # RESPLINTER cache sizes (interval_cache_size 0 = off).
    for name, minimum in (("lower_bound_memo_size", 1), ("depth1_memo_size", 1),
                          ("interval_cache_size", 0)):
        if hasattr(estimator, name):
            integer_parameter(name, getattr(estimator, name), minimum)
    if hasattr(estimator, "screen_time_cap"):
        real_parameter("screen_time_cap", estimator.screen_time_cap, strict=True)
    for name in ("lookahead_depth", "top_k", "exactify_top_k", "min_n_for_multiway"):
        value = getattr(estimator, name, None)
        if value is not None:
            integer_parameter(name, value, 0 if name == "lookahead_depth" else 1)
    for name in ("reg", "path_penalty", "theta", "transition_delta", "atom_merge_tolerance"):
        if hasattr(estimator, name):
            real_parameter(name, getattr(estimator, name))
    for name in ("time_limit", "ub_margin"):
        value = getattr(estimator, name, None)
        if value is not None:
            real_parameter(name, value, strict=(name == "time_limit"))
    for name in ("reg_alpha", "logit_weight"):
        if hasattr(estimator, name):
            value = getattr(estimator, name)
            real_parameter(name, value)
            if value > 1.0:
                raise ValueError(f"{name} must be in [0, 1].")


def classification_data(X, y, *, binary_only=False):
    """Validate arrays and return (X, encoded labels, original classes)."""
    X, y = check_X_y(X, y, accept_sparse=False, dtype=np.float64)
    check_classification_targets(y)
    classes, encoded = np.unique(y, return_inverse=True)
    maximum = 2 if binary_only else 16
    if classes.size < 2 or classes.size > maximum:
        requirement = "exactly 2" if binary_only else "between 2 and 16"
        # sklearn's estimator checks look for this sentence.
        prefix = ("Only binary classification is supported. "
                  if binary_only and classes.size > 2 else "")
        raise ValueError(f"{prefix}This estimator requires {requirement} classes; got "
                         f"{classes.size} {'class' if classes.size == 1 else 'classes'}.")
    return np.ascontiguousarray(X), np.ascontiguousarray(encoded, dtype=np.uint8), classes


def training_data(estimator, X, y, feature_names=None, *, binary_only=False):
    """Validate inputs and encode labels to native uint8 class indices."""
    validate_parameters(estimator)
    if hasattr(X, "columns") and feature_names is None:
        feature_names = list(X.columns)
    X, encoded, classes = classification_data(X, y, binary_only=binary_only)
    if feature_names is not None and len(feature_names) != X.shape[1]:
        raise ValueError("feature_names length must match the number of columns in X.")
    estimator.classes_ = classes
    estimator.n_classes_ = int(classes.size)
    estimator.n_features_in_ = X.shape[1]
    estimator.n_total_ = X.shape[0]
    estimator.feature_names_ = ([str(name) for name in feature_names] if feature_names is not None
                                else [f"x{i}" for i in range(X.shape[1])])
    if feature_names is not None and all(isinstance(name, str) for name in feature_names):
        estimator.feature_names_in_ = np.asarray(feature_names, dtype=object)
    elif hasattr(estimator, "feature_names_in_"):
        del estimator.feature_names_in_
    return X, encoded


def prediction_data(estimator, X, fitted_attribute="tree_"):
    check_is_fitted(estimator, [fitted_attribute])
    if hasattr(X, "columns") and hasattr(estimator, "feature_names_in_"):
        if list(X.columns) != estimator.feature_names_in_.tolist():
            raise ValueError("Prediction feature names and order must match the fitted data.")
    X = check_array(X, accept_sparse=False, dtype=np.float64)
    if X.shape[1] != estimator.n_features_in_:
        raise ValueError(f"X has {X.shape[1]} features, but {type(estimator).__name__} "
                         f"is expecting {estimator.n_features_in_} features as input.")
    return np.ascontiguousarray(X)


def _class_indices(y, classes):
    y = np.asarray(y).ravel()
    if classes is None:
        if not np.issubdtype(y.dtype, np.integer) or (y.size and y.min() < 0):
            raise ValueError("Without classes, y must hold class indices 0, 1, ...; "
                             "pass classes= to encode labels.")
        return y.astype(np.intp)
    lookup = {label: i for i, label in enumerate(np.asarray(classes).tolist())}
    unknown = sorted({label for label in y.tolist() if label not in lookup}, key=str)
    if unknown:
        raise ValueError(f"y contains labels not in classes: {unknown[:5]}")
    return np.array([lookup[label] for label in y.tolist()], dtype=np.intp)


# Private fitted state whose names do not end in "_".
_PRIVATE_FITTED_ATTRIBUTES = ("_logits", "_logits_full", "_closure_min_n", "_train_X",
                              "_train_y", "_fit_thresholds", "_n_classes",
                              # LicketySPLINTER
                              "_fit_start_time", "_n_solver_calls", "_thresholds_for_rust",
                              "_closure_feature_ranking", "_fit_time")


def reset_fitted_state(estimator):
    """Discard previous fit results, including a failed refit's old tree."""
    for name in list(vars(estimator)):
        if name.endswith("_") or name in _PRIVATE_FITTED_ATTRIBUTES:
            delattr(estimator, name)
