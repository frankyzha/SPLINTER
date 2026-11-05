"""Text rules (`splinter.rules`), the RESPLINTER `guarantee()` report and
`summary()`, and the LightGBM parameters that replaced environment reads.

  * every condition on a row's `explain` path holds for that row (checked
    with the edge-label parser of tests/_helpers.py);
  * `export_text` prints "≤ 0" / "> 0" at LightGBM's zero cutpoint, class
    labels rather than indices, and unions for nonlinear nodes;
  * `guarantee()` uses the user-facing keys and agrees with the fitted
    attributes; `summary()` returns its text.
"""
import numpy as np
import pytest
from sklearn.datasets import load_iris

from splinter import SPLINTER, ShapeSPLINTER, LicketySPLINTER, RESPLINTER, explain, export_text
from splinter.resplinter import _RESPLINTER
from splinter.tree import MWLeaf, MWNode, MWShapeNode, predict_encoded, route_paths

from _helpers import ZERO_SENTINEL, predicate, random_tree, thresholds_of, tiny_data


def churn_data(n=600, seed=0):
    rng = np.random.default_rng(seed)
    plan, tenure = rng.integers(4, size=n), rng.integers(73, size=n)
    churn = np.isin(plan, [0, 3]) & (tenure < 12)
    churn ^= rng.random(n) < 0.05
    return np.column_stack([plan, tenure]).astype(float), np.where(churn, "churn", "stay")


def grid_data():
    X = np.array([[i % 4, i // 4] for i in range(24)], dtype=float)
    y = np.array([int(x[0] >= 2) for x in X])
    y[[5, 18]] ^= 1
    return X, y


def leaves(count):
    return [MWLeaf(i, 1, 0.0) for i in range(count)]


@pytest.fixture(scope="module")
def rset():
    X, y = grid_data()
    model = RESPLINTER(max_depth=2, max_k=3, reg=0.02, theta=0.3, beam_width=4,
                       lgbm_n_estimators=8).fit(X, np.where(y, "yes", "no"))
    assert model.tree_count_at_target_ >= 2
    return model, X


def assert_paths_hold(model_or_tree, X, **kwargs):
    rows = explain(model_or_tree, X, **kwargs)
    tree = model_or_tree.tree_ if hasattr(model_or_tree, "tree_") else model_or_tree
    leaves_reached, paths = route_paths(tree, X)
    names = kwargs.get("feature_names")
    for i, (row, path) in enumerate(zip(rows, paths)):
        assert len(row["path"]) == len(path)
        assert row["leaf_n"] == leaves_reached[i].n_samples
        for (name, condition), (node, _) in zip(row["path"], path):
            expected = names[node.feature_index] if names else node.feature_name
            assert name == expected
            assert predicate(condition)(X[i:i + 1, node.feature_index])[0], (i, name, condition)
    return rows


@pytest.mark.parametrize("kind", [SPLINTER, ShapeSPLINTER, LicketySPLINTER])
def test_explain_conditions_hold_on_training_rows(kind):
    X, y = churn_data()
    model = kind(max_depth=3, lgbm_n_estimators=16).fit(X, y, feature_names=["plan", "tenure"])
    rows = assert_paths_hold(model, X)
    assert [r["prediction"] for r in rows] == model.predict(X).tolist()
    iris_X, iris_y = load_iris(return_X_y=True)
    iris = kind(max_depth=2, lgbm_n_estimators=8).fit(iris_X, iris_y)
    assert_paths_hold(iris, iris_X, feature_names=[f"f{j}" for j in range(4)])


def test_explain_conditions_hold_for_rashomon_trees(rset):
    model, X = rset
    for i in range(min(model.tree_count_at_target_, 10)):
        tree = model.get_tree(i)
        rows = assert_paths_hold(tree, X, classes=model.classes_)
        expected = model.classes_[predict_encoded(tree, X)].tolist()
        assert [r["prediction"] for r in rows] == expected


def test_explain_random_nonlinear_trees_hold():
    rng = np.random.default_rng(3)
    for _ in range(40):
        tree = random_tree(rng, 3)
        cuts = np.array(thresholds_of(tree) + [0.0])
        X = rng.choice(np.r_[cuts, rng.uniform(-6, 6, 40)], size=(300, 3))
        assert_paths_hold(tree, X)


def test_route_paths_matches_predict_encoded():
    X, y = churn_data()
    model = ShapeSPLINTER(max_depth=3, lgbm_n_estimators=16).fit(X, y)
    leaves_reached, paths = route_paths(model.tree_, X)
    np.testing.assert_array_equal([leaf.prediction for leaf in leaves_reached],
                                  predict_encoded(model.tree_, X))
    assert all(path[0][0] is model.tree_ for path in paths)


def test_export_text_zero_cutpoint_prints_zero():
    node = MWNode(0, "x", [ZERO_SENTINEL], leaves(2))
    lines = export_text(node).splitlines()
    assert lines[0] == "x: 2-way split"
    assert lines[1].startswith("├─ ≤ 0 → class = 0")
    assert lines[2].startswith("└─ > 0 → class = 1")
    values = np.array([[-1.0], [0.0], [ZERO_SENTINEL], [1.0]])
    assert [r["path"][0][1] for r in explain(node, values)] == ["≤ 0", "≤ 0", "> 0", "> 0"]


def test_export_text_prints_class_labels_and_counts():
    X, y = churn_data()
    model = SPLINTER(max_depth=2, lgbm_n_estimators=16).fit(X, y, feature_names=["plan", "tenure"])
    text = export_text(model)
    assert "class = churn" in text and "class = stay" in text
    assert "class = 0" not in text and "class = 1" not in text
    assert "churn: " in text and "stay: " in text and "(n = " in text
    plain = export_text(model, show_counts=False)
    assert "churn: " not in plain and "(n = " in plain
    renamed = export_text(model, class_names=["leaves", "stays"])
    assert "class = leaves" in renamed and "class = churn" not in renamed
    assert str(model).splitlines()[1:] == text.splitlines()


def test_export_text_rashomon_tree_uses_classes(rset):
    model, X = rset
    text = export_text(model.get_tree(0), classes=model.classes_, X=X)
    assert "class = yes" in text or "class = no" in text
    with pytest.raises(TypeError, match="get_tree"):
        export_text(model)


def test_export_text_shape_node_prints_unions_and_categories():
    node = MWShapeNode(0, "x", [1.0, 2.0, 3.0, 4.0], [0, 1, 0, 2, 1], leaves(3))
    text = export_text(node)
    assert "< 1 ∪ [2, 3) → class = 0" in text
    assert "[1, 2) ∪ ≥ 4 → class = 1" in text
    named = export_text(MWShapeNode(0, "plan", [0.5, 1.5, 2.5, 3.5], [1, 0, 1, 2, 0], leaves(3)),
                        category_names={"plan": list("abcde")})
    assert "b, e → class = 0" in named and "a, c → class = 1" in named


def test_guarantee_keys_and_values(rset):
    model, _ = rset
    report = model.guarantee()
    assert list(report) == [
        "search_converged", "best_tree_certified_optimal", "optimality_scope",
        "optimality_note", "optimality_gap", "lower_bound", "best_objective",
        "search_objective", "theta", "objective_bound", "inclusion_exact", "coverage_scope",
        "complete", "incomplete_reason", "count_saturated", "completeness_note"]
    assert report["optimality_scope"] == "candidate splits"
    assert report["coverage_scope"] == ("candidate splits kept by the search "
                                        "(beam_width, transition_delta)")
    assert report["best_objective"] == model.opt_
    assert report["search_objective"] == model.fit_info_["first_stage"]["objective"]
    assert report["objective_bound"] == model.rashomon_bound_
    assert report["theta"] == model.theta
    assert report["complete"] == (model.rashomon_complete_ is True)
    assert report["incomplete_reason"] == " ".join(model.rashomon_incomplete_reason_ or [])
    assert report["optimality_note"].endswith(".") and "_" not in report["optimality_note"]
    if not report["best_tree_certified_optimal"]:
        assert report["optimality_gap"] is None and report["lower_bound"] is None


def test_summary_returns_text(rset, capsys):
    model, _ = rset
    text = model.summary()
    assert capsys.readouterr().out == ""
    assert text.startswith("RESPLINTER: ") and "Best objective" in text


def test_lgbm_parameters_replace_environment(monkeypatch):
    import splinter.thresholds as threshold_module
    seen = []
    fit_lgbm = threshold_module.fit_thresholds

    def spy(*args, **kwargs):
        seen.append((kwargs["max_depth"], kwargs["num_leaves"]))
        return fit_lgbm(*args, **kwargs)

    monkeypatch.setattr(threshold_module, "fit_thresholds", spy)
    monkeypatch.setenv("LGBM_MAX_DEPTH", "1")
    monkeypatch.setenv("LGBM_NUM_LEAVES", "2")
    X, y = grid_data()
    kw = dict(max_depth=2, beam_width=2, reg=0.02, lgbm_n_estimators=4)
    assert RESPLINTER().get_params()["lgbm_num_leaves"] == 31
    assert RESPLINTER().get_params()["lgbm_max_depth"] == -1
    RESPLINTER(**kw).fit(X, y)
    model = RESPLINTER(**kw, lgbm_max_depth=3, lgbm_num_leaves=7).fit(X, y)
    assert seen == [(-1, 31), (3, 7)]
    assert model.dag_.fit_config["extras"]["lgbm_max_depth"] == 3
    with pytest.raises(ValueError, match="lgbm_num_leaves"):
        RESPLINTER(**dict(kw, lgbm_num_leaves=None)).fit(X, y)


def test_explain_category_conditions_hold():
    X, y = churn_data()
    plans = ["basic", "standard", "premium", "family"]
    model = ShapeSPLINTER(max_depth=3, lgbm_n_estimators=16).fit(
        X, y, feature_names=["plan", "tenure"])
    rows = explain(model, X, category_names={"plan": plans})
    _, paths = route_paths(model.tree_, X)
    saw_category = False
    for i, (row, path) in enumerate(zip(rows, paths)):
        for (name, condition), (node, _) in zip(row["path"], path):
            if name == "plan":
                saw_category = True
                assert plans[int(X[i, 0])] in condition.split(", "), (i, condition)
            else:
                assert predicate(condition)(X[i:i + 1, node.feature_index])[0], (i, condition)
    assert saw_category


def test_optimality_note_has_no_solver_codes():
    from splinter.resplinter import _OPTIMALITY_ISSUES, _OPTIMALITY_NOTES, _optimality_note
    codes = (list(_OPTIMALITY_NOTES) + list(_OPTIMALITY_ISSUES)
             + ["closure_at_depth_2",
                "reported_upperbound_has_no_matching_tree_witness(root_ub=0.1, witness=0.2)",
                "; ".join(list(_OPTIMALITY_ISSUES) + ["closure_at_depth_3"]), None, "unknown"])
    for code in codes:
        note = _optimality_note(code)
        assert "_" not in note and note.endswith("."), (code, note)
    joined = _optimality_note("candidate_generation_uses_top_b_truncation; closure_at_depth_2")
    assert joined == ("The best tree is not proven optimal. Each node considered only its "
                      "top-ranked candidate splits. Below depth 2 the tree was completed greedily.")


def test_optimality_note_with_lookahead_and_closure():
    X, y = grid_data()
    model = RESPLINTER(max_depth=2, lookahead_depth=1, cart_closure=True,
                       lgbm_n_estimators=8).fit(X, y)
    note = model.guarantee()["optimality_note"]
    assert note and "_" not in note
    assert model.fit_info_["full_depth_tree_in_set"] is True


def test_smaller_beam_can_be_complete_with_fewer_trees():
    X, y = tiny_data(labels=("no", "yes"))
    kw = dict(max_depth=2, max_k=3, reg=0.02, theta=0.3, lgbm_n_estimators=20)
    narrow = RESPLINTER(beam_width=1, **kw).fit(X, y)
    wide = _RESPLINTER(bypass_beam=True, **kw).fit(X, y)  # bypass_beam: a hidden setting
    assert narrow.guarantee()["complete"] and wide.guarantee()["complete"]
    assert narrow.tree_count_ < wide.tree_count_
    assert "beam_width" in narrow.guarantee()["coverage_scope"]
    assert "complete for the splits kept by the search" in narrow.summary()


def test_lgbm_parameters_validated_on_threshold_path():
    X, y = grid_data()
    for bad in (dict(lgbm_num_leaves=1), dict(lgbm_num_leaves=None), dict(lgbm_max_depth=-2),
                dict(lgbm_max_depth=1.5)):
        with pytest.raises(ValueError, match="lgbm_"):
            RESPLINTER(lgbm_n_estimators=4, **bad)._derive_thresholds(X, y)


def test_fixed_thresholds_fit_config_records_lgbm_parameters():
    X, y = grid_data()
    model = RESPLINTER(max_depth=2, lgbm_n_estimators=4, lgbm_max_depth=3, lgbm_num_leaves=7,
                       fixed_thresholds={0: [1.5], 1: [2.5]}).fit(X, y)
    extras = model.dag_.fit_config["extras"]
    assert extras["lgbm_max_depth"] == 3 and extras["lgbm_num_leaves"] == 7
