"""Beam-restricted Rashomon sets of sparse multiway trees.

RESPLINTER returns a factored DAG (a shared graph that encodes many trees
at once) of the trees inside the reported objective threshold.
Completeness is relative to the candidate splits kept by the search and
requires exact cost keys and a completed search; see ``guarantee()``.
"""

import json
import os
import tempfile
import time
import warnings
import zipfile

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted

from .tree import dict_to_tree
from ._validation import (training_data, prediction_data, reset_fitted_state,
                          real_parameter, integer_parameter, classification_data,
                          validate_parameters)
from .thresholds import _normalize_thresholds, _normalize_logits, _discard_fit_only_arrays
from .multiway_solver import solve_rashomon


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

# Pickle state marker for a DAG stored as its v6 bytes.
_DAG_STATE_TAG = "rashomon_dag_v6"
# Version of the saved-file layout; `load` refuses any other.
_SAVE_FORMAT_VERSION = 1
# Fitted attributes saved separately (or rebuilt) by `save` / `load`.
_SAVED_SEPARATELY = ("dag_", "classes_", "feature_names_in_", "lgbm_thresholds_")
# Arrays of a saved file, and the fitted attributes `load` cannot do without.
_SAVED_ARRAYS = ("dag", "X", "y", "n_classes", "threshold_features", "threshold_lengths",
                 "threshold_values", "params_json", "fitted_json", "format_version")
_REQUIRED_FITTED = ("classes_", "fit_info_", "feature_names_", "n_features_in_", "n_classes_",
                    "n_total_", "tree_count_", "tree_count_at_target_", "opt_",
                    "rashomon_bound_", "theta_bound_", "n_dag_nodes_", "fit_time_")
# Earlier format-1 files stored these reports as separate fitted attributes.
# Their values are now held by fit_info_ (including its raw_result).
_OLD_REPORT_ATTRIBUTES = (
    "anchor_mode_", "candidate_ranking_policy_", "fill_mode_", "legacy_delta_metadata_",
    "memo_len_", "n_explored_", "n_leaves_empty_", "n_leaves_filled_", "n_leaves_timed_out_",
    "n_prefix_leaves_", "n_stepb_prefix_leaves_", "opt_dag_", "opt_phase1_",
    "opt_pre_reanchor_", "opt_upfront_", "phase1_certification_mode_",
    "phase1_certification_reason_", "phase1_certified_", "phase1_certified_optimal_",
    "phase1_gap_", "phase1_gap_zero_", "phase1_is_optimal_", "phase1_lowerbound_",
    "phase1_objective_", "phase1_search_converged_", "phase1_termination_reason_",
    "phase1_timed_out_", "phase1_unsafe_heuristics_used_", "phase2_timed_out_",
    "rashomon_bound_enum_", "rashomon_bound_target_dag_", "reanchored_",
    "remaining_depth_", "solve_depth_", "stepb_mode_", "tree_count_at_target_dag_",
    "tree_count_pre_reanchor_", "upfront_anchor_in_dag_",
)


def _saved_value(name, value):
    """`_json_value(value)` for the fitted attribute `name`; ValueError when
    it cannot be stored (nothing is dropped silently)."""
    try:
        return _json_value(value)
    except TypeError as exc:
        raise ValueError(f"Cannot save fitted attribute {name} of type "
                         f"{type(value).__name__}.") from exc


def _json_value(value):
    """`value` as plain JSON types; TypeError for anything else (arrays,
    dicts with non-string keys, native objects)."""
    if value is None or isinstance(value, (bool, str, int, float)):
        return value
    if isinstance(value, (np.bool_, np.integer, np.floating, np.str_)):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    if isinstance(value, dict) and all(isinstance(k, str) for k in value):
        return {k: _json_value(v) for k, v in value.items()}
    raise TypeError(f"{type(value).__name__} is not JSON-serializable")


def _flatten_thresholds(thresholds):
    features = sorted(int(j) for j in thresholds)
    values = [np.asarray(thresholds[j], dtype=np.float64).ravel() for j in features]
    return (np.asarray(features, dtype=np.int64),
            np.asarray([len(v) for v in values], dtype=np.int64),
            np.concatenate(values) if values else np.zeros(0, dtype=np.float64))


def _unflatten_thresholds(features, lengths, values):
    if len(features) != len(lengths) or int(np.sum(lengths)) != len(values):
        raise ValueError("inconsistent threshold arrays")
    out, start = {}, 0
    for j, length in zip(features.tolist(), lengths.tolist()):
        out[int(j)] = values[start:start + length].tolist()
        start += length
    return out


# Plain sentences for the solver's optimality certificate codes. A
# certificate is either one status code or "; "-joined issue codes
# (`assess_certification` in src/solver.rs).
_OPTIMALITY_NOTES = {
    "certified": "The search proved the best tree optimal among trees built from the "
                 "candidate splits.",
    "timed_out_before_certificate": "The search reached its time limit before it could "
                                    "prove the best tree optimal.",
    "search_converged_without_certificate": "The search converged but did not prove the "
                                            "best tree optimal.",
    "search_not_converged": "The search stopped before converging, so the best tree is not "
                            "proven optimal.",
    "parallel_root_uses_candidate_screening_and_top_n_selection":
        "The best tree is not proven optimal: the search screens the candidate splits at "
        "the root and keeps only the top-ranked ones.",
}
_OPTIMALITY_ISSUES = {
    "topk_closure_direct": "The tree was completed directly from the top-ranked splits, "
                           "without a full search.",
    "heuristic_support_bound_transfer_enabled": "Bounds were reused heuristically between "
                                                "similar subproblems.",
    "candidate_generation_uses_top_b_truncation": "Each node considered only its top-ranked "
                                                  "candidate splits.",
    "ub_margin_pruned_surviving_candidates": "Some candidate splits were pruned with a "
                                             "heuristic margin.",
    "bfs_materialized_only_a_subset_of_survivors": "Only part of the surviving candidate "
                                                   "splits was expanded.",
    "greedy_boundary_guidance_enabled": "Greedy guidance chose where to place split "
                                        "thresholds.",
}
# Issue codes that carry a value: prefix -> sentence (with {} for the value).
_OPTIMALITY_ISSUE_PREFIXES = {
    "closure_at_depth_": "Below depth {} the tree was completed greedily.",
    "reported_upperbound_has_no_matching_tree_witness(":
        "The reported objective did not match a tree the search could rebuild.",
}

# Plain sentences for the solver's completeness codes
# (`rashomon_incomplete_reason`, ";"-joined; src/lib.rs).
_INCOMPLETE_REASONS = {
    "phase2_timeout": "The enumeration of the set reached its time limit, so trees may "
                      "be missing.",
    "closure": "The levels below lookahead_depth were completed greedily (cart_closure), "
               "not enumerated.",
    "unfilled_prefix_leaves": "The levels below lookahead_depth were not filled, so the "
                              "trees stop at lookahead_depth.",
    "stepb_timeout": "Filling the levels below lookahead_depth reached its time limit at "
                     "some leaves, so trees may be missing.",
    "rounded_keys": "Tree objectives were compared with the bound after rounding.",
    "non_leaf_objective": "The objective is not a sum of per-leaf costs, so the set cannot "
                          "compare it with the bound exactly.",
    "upfront_anchor_not_in_dag": "The best tree found is not in the set, so the bound is "
                                 "not set from a tree of the set.",
    "no completeness certificate": "The search did not report whether the set is complete.",
}


def _incomplete_sentences(codes):
    """Plain sentences for ";"-joined completeness codes ([] for "")."""
    return [_INCOMPLETE_REASONS.get(code, f"Unrecognized completeness note ({code}).")
            for code in (c.strip() for c in codes.split(";")) if code]


# What `guarantee()["complete"]` is relative to.
_COVERAGE_SCOPE = "candidate splits kept by the search (beam_width, transition_delta)"


def _issue_sentence(code):
    if code in _OPTIMALITY_ISSUES:
        return _OPTIMALITY_ISSUES[code]
    for prefix, sentence in _OPTIMALITY_ISSUE_PREFIXES.items():
        if code.startswith(prefix):
            return sentence.format(code[len(prefix):])
    return f"Unrecognized solver note ({code})."


def _optimality_note(reason):
    """Plain sentences for a solver certification code, or for its
    "; "-joined issue codes."""
    if not reason or reason == "unknown":
        return "No optimality certificate was recorded."
    if reason in _OPTIMALITY_NOTES:
        return _OPTIMALITY_NOTES[reason]
    issues = " ".join(_issue_sentence(code.strip()) for code in reason.split("; "))
    return "The best tree is not proven optimal. " + issues


# Public names of the solver's `stepb_mode` values (`fit_info_["completion"]
# ["mode"]`): 'paper' = the levels below the lookahead were filled by a
# second-level Rashomon search.
_COMPLETION_MODES = {"paper": "recursive", "none": "none"}


def _fit_info(result, fill_mode, n_frontier_leaves):
    """`fit_info_`: the solver's report beyond the documented fitted
    attributes, under plain keys (`raw_result` is the solver's dict without
    the DAG, which is `dag_`). `n_frontier_leaves` = leaves at the
    lookahead frontier."""
    v = result.get("tree_count_at_target_dag", None)
    w = result.get("tree_count_pre_reanchor", None)
    return {
        "candidate_ranking_policy": result.get("candidate_ranking_policy"),
        "fill_mode": fill_mode,
        # The first search stage, which finds the best tree.
        "first_stage": {
            "is_optimal": result["phase1_is_optimal"],
            "certified_optimal": result.get("phase1_certified_optimal",
                                            result["phase1_is_optimal"]),
            "search_converged": result.get("phase1_search_converged",
                                           result["phase1_is_optimal"]),
            "certification_mode": result.get("phase1_certification_mode", "exploratory"),
            "certification_reason": result.get("phase1_certification_reason",
                                               "uncertified"),
            "unsafe_heuristics_used": result.get("phase1_unsafe_heuristics_used", True),
            "gap": result["phase1_gap"],
            # 0 by construction when the search screens candidates; it
            # certifies nothing.
            "zero_gap": result.get("phase1_certified", None),
            "objective": result["phase1_objective"],
            "lower_bound": result.get("phase1_lowerbound"),
            "timed_out": result.get("phase1_timed_out", None),
            "termination_reason": result.get("phase1_termination_reason", None),
            # The best objective of the top levels alone.
            "prefix_objective": result.get("opt_phase1", None),
        },
        # Where `opt_` came from: the first search's best tree of the top
        # levels, or that tree completed to full depth up front; the
        # objective of that completion (None when it did not run) and
        # whether the set's best tree is no worse than the reference tree
        # (that completion, or the first search's tree when it solved every
        # level; None when neither applies).
        "reference_source": result.get("anchor_mode", None),
        "full_depth_objective": result.get("opt_upfront", None),
        "full_depth_tree_in_set": result.get("upfront_anchor_in_dag", None),
        # Whether the threshold was tightened to the set's own best tree
        # after the fill, and the objective, bound and count before that.
        "threshold_tightened": result.get("reanchored", None),
        "objective_before_tightening": result.get("opt_pre_reanchor", None),
        "tree_count_before_tightening": None if w is None else int(w),
        "enumeration_bound": result.get("rashomon_bound_enum", None),
        # The same report relative to the set's own best tree (min cost
        # in the final graph, graph cost units); None on an empty set.
        "set_best_objective": result.get("opt_dag", None),
        "set_theta_bound": result.get("rashomon_bound_target_dag", None),
        "tree_count_at_set_theta_bound": None if v is None else int(v),
        "solve_depth": result["solve_depth"],
        "remaining_depth": result["remaining_depth"],
        # Leaves at the lookahead frontier.
        "n_frontier_leaves": n_frontier_leaves,
        "enumeration_timed_out": result.get("phase2_timed_out", None),
        # ';'-joined failed completeness conditions ("" when complete).
        "incomplete_reason_codes": result.get("rashomon_incomplete_reason", None),
        "exact_cost_keys": result.get("exact_cost_keys", None),
        # The fill of the levels below the lookahead: 'recursive' (the
        # second-level Rashomon search ran) or 'none', and the
        # frontier-leaf counts (filled / empty / timed out are None when it
        # did not run). The solver's own value is raw_result["stepb_mode"].
        "completion": {
            "mode": _COMPLETION_MODES.get(result.get("stepb_mode", None),
                                          result.get("stepb_mode", None)),
            "n_frontier_leaves": result.get("n_prefix_leaves", None),
            "n_filled": result.get("n_leaves_filled", None),
            "n_empty": result.get("n_leaves_empty", None),
            "n_timed_out": result.get("n_leaves_timed_out", None),
        },
        "n_explored": result.get("n_explored", None),
        "memo_size": result.get("memo_len", None),
        "raw_result": {key: value for key, value in result.items() if key != "dag"},
    }


def _restore_dag(data, feature_names, X, y, thresholds, n_classes):
    """A PyRashomonDag from its v6 bytes with the training rows re-attached."""
    from .multiway_solver import PyRashomonDag
    dag = PyRashomonDag.from_bytes(bytes(data), feature_names=list(feature_names))
    dag.attach_training_data(X, y, thresholds, n_classes)
    return dag


def _migrate_saved_fitted(params, fitted):
    """Convert earlier format-1 reports without refitting or changing the set."""
    if "fit_info_" in fitted or "result_" not in fitted:
        return fitted
    result = fitted.pop("result_")
    if not isinstance(result, dict):
        raise ValueError("saved result_ must be an object")
    codes = fitted.get("rashomon_incomplete_reason_",
                       result.get("rashomon_incomplete_reason"))
    fill_mode = fitted.get("fill_mode_", params.get("fill_mode", "full"))
    if ("closure" in (codes or "").split(";")
            or (params.get("cart_closure") and params.get("lookahead_depth") is not None
                and params["lookahead_depth"] < params["max_depth"])):
        fill_mode = "closure"
    fitted["fit_info_"] = _fit_info(result, fill_mode, fitted.get("n_prefix_leaves_"))
    fitted["fit_info_"]["candidate_ranking_policy"] = result.get(
        "candidate_ranking_policy", fitted.get("candidate_ranking_policy_"))
    # A zero gap (formerly phase1_certified_) is not an optimality certificate.
    fitted["fit_info_"]["first_stage"]["certified_optimal"] = result.get(
        "phase1_certified_optimal", False)
    if "rashomon_bound_target_" in fitted:
        fitted["theta_bound_"] = fitted.pop("rashomon_bound_target_")
    fitted["rashomon_incomplete_reason_"] = (None if codes is None
                                             else _incomplete_sentences(codes))
    # None used to request a full-depth search, not today's half-depth default.
    if params.get("lookahead_depth") is None:
        params["lookahead_depth"] = result["solve_depth"]
    for name in _OLD_REPORT_ATTRIBUTES:
        fitted.pop(name, None)
    return fitted


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class _RESPLINTER(ClassifierMixin, BaseEstimator):
    """Rashomon set of near-optimal multiway decision trees.

    The set is stored as a compact graph of trees (a DAG) and holds every
    tree whose objective is within ``1 + theta`` of the best tree found.
    A completed fit is complete relative to the candidate splits kept by
    the search, not the unrestricted class of multiway trees.

    Parameters
    ----------
    max_depth : int, default=4
        Maximum tree depth.
    max_k : int, default=4
        Maximum split arity.
    reg : float, default=0.01
        Per-leaf regularization.
    theta : float, default=0.05
        Relative tolerance around the best objective found. ``guarantee()``
        reports the objective bound used and whether the set is complete.
    min_samples_leaf : int, default=1
        Kept for sklearn API compatibility and saved configs; must be 1.
        RESPLINTER does not enforce a minimum leaf size, so any other value
        raises ValueError. Use ``lgbm_min_samples_leaf`` for the reference
        ensemble and ``splinter.explore.select(table, min_leaf=...)`` to
        filter trees by their smallest leaf.
    feature_exclusion : bool, default=False
        If True, each feature appears at most once per root-to-leaf path.
    time_limit : float or None, default=None
        Maximum solve time per search stage (seconds).
    beam_width : int, default=4
        Partitions kept per interval-DP cell, both when finding the best
        tree and when enumerating the set. Larger values keep more distinct
        candidate splits.
    reg_alpha : float, default=1.0
        Leaf-regularization scale. The factored representation requires 1.0.
    transition_delta : float, default=0.02
        Bin compression delta. Lower = less aggressive compression.
        Used both when finding the best tree and when enumerating the set.
    lgbm_num_leaves : int, default=31
        LightGBM reference-ensemble `num_leaves`.
    lgbm_max_depth : int, default=-1
        LightGBM reference-ensemble `max_depth` (-1 = no limit).
    lgbm_min_samples_leaf : int or None, default=None
        Minimum child samples used only when fitting the LightGBM reference
        ensemble (which supplies the thresholds). None = use
        ``min_samples_leaf`` (i.e. 1).
    binary_threshold_backfill : bool, default=False
        Supply a 0.5 cut for binary features unused by the reference ensemble.
        Disable to use only the LightGBM thresholds.
    threshold_budget : None, float, int, or str, default=None
        Restrict the LightGBM threshold pool, changing the set of
        representable trees. Keep only part of the thresholds before the
        solve, with LicketySPLINTER's semantics -- a float in (0, 1) keeps
        the thresholds carrying that fraction of the LightGBM split-gain
        mass, an int keeps that many (`thresholds._parse_budget`);
        applied before the binary 0.5
        backfill. None / 'full' = every LightGBM threshold (default).
    cart_closure_mode : str, default='greedy'
        Split criterion of the greedy completion below ``lookahead_depth``
        (``cart_closure``): 'gini', 'misclass',
        'rush', 'entropy' or 'hellinger'; the default 'greedy' means Gini.
        Any other value raises ValueError at fit.
    fixed_thresholds : dict or None, default=None
        Candidate thresholds {feature index: thresholds} to solve with
        instead of deriving them from the LightGBM reference ensemble (no
        ensemble is fit; `threshold_source_ = 'fixed'`). Used by RID to fit every
        bootstrap on the thresholds `_derive_thresholds` gives on the full
        data. Every feature index of the fit data must be a key ([] =
        never split on it), and logit_weight must be 1.0 (ValueError
        otherwise). None = derive them on the fit data (default).
    verbose : bool, default=False
        Print progress.
    n_threads : int, default=1
        Threads of the parallel lower-bound prefetch of the enumeration
        (1 = off); the same trees at any value.
    completion_batches : int, default=0
        Split each level of the recursive fill into this many batches
        solved in parallel (0 = unbatched); the same trees at any value.
    completion_batch_threads : int or None, default=None
        ``n_threads`` of each batch solver (None = ``n_threads`` divided by
        the number of batches, at least 1).
    interval_lower_bound : bool, default=True
        Price the remaining-depth-2 completions by an exact interval DP over
        the threshold bins (a tighter-or-equal admissible bound; the set can
        only grow).
    interval_min_arity : int, default=3
        With ``interval_lower_bound``, use the interval DP only when
        ``max_k`` is at least this.
    lower_bound_memo_size : int, default=3000000
        Remaining-depth-1 entries the lower-bound memo holds before it drops
        them (a cache: same trees at any value).
    depth1_memo_size : int, default=16000000
        Entries of the structural depth-1 lower-bound memo (a cache).
    interval_cache_size : int, default=100000
        Entries of the interval DP's per-support cache (0 = off).
    screen_time_cap : float, default=60.0
        Time limit (seconds) of each root-candidate solve when the screen
        of the first search stage is its final solve.
    """

    # lgbm_min_samples_leaf=None means "use min_samples_leaf" (see validate_parameters).
    _none_allowed_parameters = ("lgbm_min_samples_leaf",)

    def __init__(
        self, max_depth=4, lookahead_depth=None, max_k=4, reg=0.01, theta=0.05,
        min_samples_leaf=1, feature_exclusion=False, time_limit=None,
        beam_width=4, reg_alpha=1.0, transition_delta=0.02,
        lgbm_n_estimators=200, lgbm_num_leaves=31, lgbm_max_depth=-1,
        lgbm_min_samples_leaf=None,
        binary_threshold_backfill=False, max_bins=1024, lgbm_random_state=42,
        cost_precision=1e9, path_penalty=0.0, cart_closure=False,
        cart_closure_beam=1, cart_closure_mode='greedy', shape_mode=False,
        atom_mode=True, logit_weight=1.0, logits=None,
        exactify_top_k=None, threshold_budget=None, atom_merge_tolerance=0.0,
        fill_mode='full', leaf_reg_cap=0.1, leaf_short_circuit_eps=1e-4,
        fill_parallel=True, bypass_beam=False, fixed_thresholds=None, verbose=False,
        n_threads=1, completion_batches=0, completion_batch_threads=None,
        interval_lower_bound=True, interval_min_arity=3, lower_bound_memo_size=3_000_000,
        depth1_memo_size=16_000_000, interval_cache_size=100_000, screen_time_cap=60.0,
    ):
        self.max_depth = max_depth
        self.lookahead_depth = lookahead_depth
        self.max_k = max_k
        self.reg = reg
        self.theta = theta
        self.min_samples_leaf = min_samples_leaf
        self.feature_exclusion = feature_exclusion
        self.time_limit = time_limit
        self.beam_width = beam_width
        self.reg_alpha = reg_alpha
        self.transition_delta = transition_delta
        self.lgbm_n_estimators = lgbm_n_estimators
        self.lgbm_num_leaves = lgbm_num_leaves
        self.lgbm_max_depth = lgbm_max_depth
        self.lgbm_min_samples_leaf = lgbm_min_samples_leaf
        self.binary_threshold_backfill = binary_threshold_backfill
        self.max_bins = max_bins
        self.lgbm_random_state = lgbm_random_state
        self.cost_precision = cost_precision
        self.path_penalty = path_penalty
        self.cart_closure = cart_closure
        self.cart_closure_beam = cart_closure_beam
        self.cart_closure_mode = cart_closure_mode
        self.shape_mode = shape_mode
        self.atom_mode = atom_mode
        self.logit_weight = logit_weight
        self.logits = logits
        self.exactify_top_k = exactify_top_k
        self.threshold_budget = threshold_budget
        self.atom_merge_tolerance = atom_merge_tolerance
        self.fill_mode = fill_mode
        self.leaf_reg_cap = leaf_reg_cap
        self.leaf_short_circuit_eps = leaf_short_circuit_eps
        self.fill_parallel = fill_parallel
        self.bypass_beam = bypass_beam
        self.fixed_thresholds = fixed_thresholds
        self.verbose = verbose
        self.n_threads = n_threads
        self.completion_batches = completion_batches
        self.completion_batch_threads = completion_batch_threads
        self.interval_lower_bound = interval_lower_bound
        self.interval_min_arity = interval_min_arity
        self.lower_bound_memo_size = lower_bound_memo_size
        self.depth1_memo_size = depth1_memo_size
        self.interval_cache_size = interval_cache_size
        self.screen_time_cap = screen_time_cap

    def _more_tags(self):
        return {"binary_only": True}

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.classifier_tags.multi_class = False
        return tags

    def __sklearn_clone__(self):
        """Keep restored bin settings when cloning an older saved estimator."""
        model = super().__sklearn_clone__()
        for name in ("atom_mode", "binary_threshold_backfill"):
            setattr(model, name, getattr(self, name))
        return model

    def _validate_leaf_parameters(self):
        """RESPLINTER enforces no minimum leaf size: reject min_samples_leaf != 1
        and validate the reference ensemble's lgbm_min_samples_leaf,
        lgbm_num_leaves and lgbm_max_depth before any training."""
        if self.min_samples_leaf != 1:
            raise ValueError(
                "RESPLINTER does not enforce a minimum leaf size; use "
                "lgbm_min_samples_leaf for the reference ensemble and "
                "splinter.explore.select(table, min_leaf=...) to filter trees.")
        if self.lgbm_min_samples_leaf is not None:
            integer_parameter("lgbm_min_samples_leaf", self.lgbm_min_samples_leaf, 1)
        integer_parameter("lgbm_num_leaves", self.lgbm_num_leaves, 2)
        integer_parameter("lgbm_max_depth", self.lgbm_max_depth, -1)

    def _effective_lookahead_depth(self):
        """The lookahead depth `fit` uses: ``lookahead_depth`` capped at
        ``max_depth``, or ``max_depth // 2`` when it is None."""
        if self.lookahead_depth is None:
            return int(self.max_depth) // 2
        return min(self.lookahead_depth, self.max_depth)

    def _effective_fill_mode(self):
        """The fill mode `fit` uses ('full' becomes 'recursive' when the
        lookahead depth is below max_depth); ValueError for an invalid
        combination."""
        if self.fill_mode not in ("full", "recursive"):
            raise ValueError("fill_mode must be 'full' or 'recursive'.")
        fill_mode = self.fill_mode
        lookahead = self._effective_lookahead_depth()
        if fill_mode == "full" and lookahead < self.max_depth:
            fill_mode = "recursive"
        if fill_mode == "recursive" and lookahead >= self.max_depth:
            raise ValueError("Recursive fill requires lookahead_depth < max_depth.")
        return fill_mode

    def _checked_fixed_thresholds(self, m):
        """`fixed_thresholds` normalized and checked for `m` features:
        ValueError when combined with threshold_budget, or when a feature
        index is out of range or missing."""
        if self.threshold_budget is not None:
            raise ValueError("threshold_budget cannot be combined with fixed_thresholds")
        thresholds = _normalize_thresholds(self.fixed_thresholds)
        # A feature absent from the dict makes the solver re-derive its
        # midpoints from the fit data (a per-fit candidate set), so every
        # feature must be listed ([] = never split on it).
        bad = sorted(j for j in thresholds if not 0 <= j < m)
        if bad:
            raise ValueError(f"fixed_thresholds: feature indices {bad} out of "
                             f"range for {m} features")
        missing = [j for j in range(m) if j not in thresholds]
        if missing:
            raise ValueError(f"fixed_thresholds: features {missing} missing "
                             "(list every feature; [] = never split on it)")
        return thresholds

    def _apply_threshold_budget(self, thresholds, boundary_gain):
        """`thresholds` cut to `threshold_budget` (LicketySPLINTER's rule:
        rank the LightGBM thresholds by split gain, keep the budget's count,
        no per-feature floor), or unchanged (the same object) when the
        budget is None / 'full' / 'all' / 'none'. Records
        `threshold_budget_selected_` / `threshold_budget_total_`."""
        self.threshold_budget_selected_ = None
        self.threshold_budget_total_ = None
        budget = getattr(self, "threshold_budget", None)
        if budget is None or (isinstance(budget, str)
                              and budget.strip().lower() in ("full", "all", "none")):
            return thresholds
        if not thresholds:
            return thresholds
        from .thresholds import _apply_budget, _parse_budget, _gain_per_threshold
        ranked = _gain_per_threshold(thresholds, boundary_gain)
        if ranked is None:
            raise ValueError("threshold_budget requires LightGBM split gains.")
        target = _parse_budget(budget)(ranked)
        out = _apply_budget(thresholds, ranked, target)
        self.threshold_budget_selected_ = sum(len(v) for v in out.values())
        self.threshold_budget_total_ = len(ranked)
        return out

    def _derive_thresholds(self, X, y):
        """Candidate thresholds {feature index: thresholds} for a fit on
        (X, y): LightGBM reference ensemble -> `threshold_budget` -> binary
        0.5 backfill (the default `fit` path). Sets the LightGBM attributes
        (`lgbm_thresholds_`, `lgbm_time_`, `boundary_*_`) and the private
        logits `_logits` and `_logits_full`, and returns
        the dict `fit` passes to the solver. `X` is a float array."""
        validate_parameters(self)
        self._validate_leaf_parameters()
        X, y, _ = classification_data(X, y, binary_only=True)
        # Threshold extraction: RESPLINTER uses the same LightGBM reference cache
        # as the single-tree estimators.
        thresholds_for_rust = None
        self.lgbm_thresholds_ = None
        self.lgbm_time_ = 0.0
        self.boundary_gain_ = None
        self.boundary_cover_ = None
        self.boundary_value_jump_ = None
        self.bin_representatives_ = None
        self.bin_fill_values_ = None
        self.n_bins_per_feature_ = None
        from .thresholds import fit_thresholds
        t_lgbm = time.time()
        _tg_max_depth = int(self.lgbm_max_depth)
        _tg_num_leaves = int(self.lgbm_num_leaves)
        self.lgbm_num_leaves_ = _tg_num_leaves
        self.lgbm_max_depth_ = _tg_max_depth
        self.lgbm_min_samples_leaf_ = int(self.lgbm_min_samples_leaf
                                          if self.lgbm_min_samples_leaf is not None
                                          else self.min_samples_leaf)
        result = fit_thresholds(
            X,
            y,
            max_bins=int(self.max_bins),
            n_estimators=int(self.lgbm_n_estimators),
            min_samples_leaf=self.lgbm_min_samples_leaf_,
            random_state=self.lgbm_random_state,
            num_threads=1,
            max_depth=_tg_max_depth,
            num_leaves=_tg_num_leaves,
        )
        thresholds_for_rust = result.thresholds
        _pre_budget = thresholds_for_rust
        thresholds_for_rust = self._apply_threshold_budget(thresholds_for_rust, result.boundary_gain)
        # Features the budget emptied stay empty (no 0.5 backfill below), as in
        # LicketySPLINTER, which applies the budget with no backfill.
        _budget_had_thresholds = ({j for j, v in _pre_budget.items() if len(v) > 0}
                                  if thresholds_for_rust is not _pre_budget else set())
        self._logits = _normalize_logits(
            self.logits if self.logits is not None else result.logits,
            len(X), 2,
        )
        self.boundary_gain_ = result.boundary_gain
        self.boundary_cover_ = result.boundary_cover
        self.boundary_value_jump_ = result.boundary_value_jump
        self.bin_representatives_ = result.bin_representatives
        self.bin_fill_values_ = result.fill_values
        self.n_bins_per_feature_ = result.n_bins_per_feature
        self.lgbm_time_ = time.time() - t_lgbm
        # Backfill {0.5} threshold for binary 0/1 features that LightGBM did not split on.
        # Without this, Rust treats the empty threshold list as "do not split this feature,"
        # which under-samples the candidate set; {0.5} is used for every binary feature,
        # as in PRAXIS (Heile et al.). binary_threshold_backfill=False skips it.
        for _j in range(X.shape[1] if self.binary_threshold_backfill else 0):
            _uniq = np.unique(X[:, _j])
            _is_binary = len(_uniq) == 2 and set(_uniq.astype(float).tolist()).issubset({0.0, 1.0})
            if (_is_binary and len(thresholds_for_rust.get(_j, [])) == 0
                    and _j not in _budget_had_thresholds):
                thresholds_for_rust[_j] = [0.5]
        self.lgbm_thresholds_ = thresholds_for_rust
        if self.verbose:
            n_thresh = sum(len(v) for v in thresholds_for_rust.values())
            print(f"  Threshold extraction (LightGBM): "
                  f"{len(thresholds_for_rust)} features, {n_thresh} total, "
                  f"time={self.lgbm_time_:.3f}s")
        self._logits_full = result.logits
        return thresholds_for_rust

    def fit(self, X, y, feature_names=None):
        """Fit the Rashomon set: find optimal, then enumerate near-optimal trees.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)
            Any two discrete class labels, encoded internally as 0/1.
        feature_names : list of str, optional
            Names used in the trees; taken from the columns of a DataFrame
            ``X`` when not given.

        Returns
        -------
        self
        """
        reset_fitted_state(self)
        X, y = training_data(self, X, y, feature_names, binary_only=True)
        self._validate_leaf_parameters()
        n, m = X.shape
        if self.path_penalty != 0.0 or self.reg_alpha != 1.0:
            raise ValueError(
                "RESPLINTER supports the leaf-only objective: "
                "path_penalty must be 0 and reg_alpha must be 1. "
                "Use a single-tree estimator for path-penalty optimization."
            )
        real_parameter("cost_precision", self.cost_precision, 1e6)
        real_parameter("leaf_reg_cap", self.leaf_reg_cap)
        real_parameter("leaf_short_circuit_eps", self.leaf_short_circuit_eps)
        fill_mode = self._effective_fill_mode()
        self._logits = None

        if self.verbose:
            print(f"{type(self).__name__}: max_depth={self.max_depth}, max_k={self.max_k}, "
                  f"reg={self.reg}, theta={self.theta}")

        effective_lookahead = self._effective_lookahead_depth()

        if self.fixed_thresholds is not None:
            thresholds_for_rust = self._checked_fixed_thresholds(m)
            if self.logit_weight != 1.0:
                raise ValueError("fixed_thresholds requires logit_weight == 1.0 "
                                 "(the fixed path fits no LightGBM reference ensemble)")
            self.lgbm_thresholds_ = thresholds_for_rust
            self.lgbm_time_ = 0.0
            self.boundary_gain_ = None
            self.boundary_cover_ = None
            self.boundary_value_jump_ = None
            self.bin_representatives_ = None
            self.bin_fill_values_ = None
            self.n_bins_per_feature_ = None
            self.threshold_source_ = 'fixed'
        else:
            thresholds_for_rust = self._derive_thresholds(X, y)
            self.threshold_source_ = 'lightgbm'

        t0 = time.time()
        # Both stages use the same candidate pool.
        rust_kwargs = dict(
            x=np.ascontiguousarray(X, dtype=np.float64),
            y=np.ascontiguousarray(y, dtype=np.uint8),
            n_classes=2,
            reg=self.reg,
            max_depth=self.max_depth,
            max_k=self.max_k,
            theta=self.theta,
            lookahead_depth=effective_lookahead,
            min_samples_leaf=self.min_samples_leaf,
            feature_exclusion=self.feature_exclusion,
            time_limit=self.time_limit,
            feature_names=self.feature_names_,
            beam_width=self.beam_width,
            reg_alpha=self.reg_alpha,
            transition_delta=self.transition_delta,
            thresholds_per_feature=thresholds_for_rust,
            cost_precision=self.cost_precision,
            path_penalty=self.path_penalty,
            closure_depth=(self.max_depth - effective_lookahead
                           if self.cart_closure and effective_lookahead < self.max_depth
                           else None),
            closure_beam=self.cart_closure_beam,
            closure_criterion=self.cart_closure_mode,
            shape_mode=self.shape_mode,
            atom_mode=self.atom_mode,
            teacher_logit=self._logits,
            teacher_alpha=self.logit_weight,
            top_k=self.exactify_top_k,
            atom_merge_tolerance=self.atom_merge_tolerance,
            recursive_fill=(fill_mode == 'recursive'),
            leaf_reg_cap=self.leaf_reg_cap,
            leaf_short_circuit_eps=self.leaf_short_circuit_eps,
            fill_parallel=self.fill_parallel,
            verbose=bool(self.verbose),
            bypass_beam=self.bypass_beam,
            lb_threads=int(self.n_threads),
            stepb_batches=int(self.completion_batches),
            stepb_batch_lb_threads=(None if self.completion_batch_threads is None
                                    else int(self.completion_batch_threads)),
            lb_interval=bool(self.interval_lower_bound),
            lb_interval_mink=int(self.interval_min_arity),
            lb_memo_max_entries=int(self.lower_bound_memo_size),
            lb_d1s_max_entries=int(self.depth1_memo_size),
            lb_interval_cache_max=int(self.interval_cache_size),
            phase1_screen_cap_s=float(self.screen_time_cap),
        )
        result = solve_rashomon(**rust_kwargs)
        self.fit_time_ = time.time() - t0
        # Copies of the exact solver inputs, kept so a pickled or saved model
        # can re-attach its training rows to the reloaded DAG; copied so later
        # edits to the caller's arrays or to `lgbm_thresholds_` cannot change them.
        self._train_X = np.array(rust_kwargs["x"], dtype=np.float64, order="C", copy=True)
        self._train_y = np.array(rust_kwargs["y"], dtype=np.uint8, order="C", copy=True)
        self._fit_thresholds = {j: list(v) for j, v in thresholds_for_rust.items()}
        self._n_classes = rust_kwargs["n_classes"]

        # Store results: the documented attributes, then everything else in
        # `fit_info_` (plain keys; `raw_result` is the solver's dict without
        # the DAG, which is `dag_`).
        self.dag_ = result["dag"]  # PyRashomonDag object (lives in Rust memory)
        self.opt_ = result["opt"]
        self.rashomon_bound_ = result["rashomon_bound"]
        self.tree_count_ = int(result["tree_count"])
        # True when a tree-count op saturated at u128::MAX (counts are then
        # clamped lower bounds; get_tree / sampling raise OverflowError).
        self.count_saturated_ = bool(result.get("count_saturated", False))
        # The bound of the reported set, opt*(1+theta), and its count; they
        # differ from rashomon_bound_ / tree_count_ only when the first search
        # was not certified (its gap > 0 widens the stored bound).
        self.tree_count_at_target_ = int(result.get("tree_count_at_target", self.tree_count_))
        self.theta_bound_ = result.get("rashomon_bound_target", self.rashomon_bound_)
        self.n_dag_nodes_ = result["n_dag_nodes"]
        # Completeness also requires exact cost keys and a finished fill of
        # the levels below the lookahead.
        self.rashomon_complete_ = result.get("rashomon_complete", None)
        codes = result.get("rashomon_incomplete_reason", None)
        self.rashomon_incomplete_reason_ = (None if codes is None
                                            else _incomplete_sentences(codes))
        # With cart_closure the levels below the lookahead are completed
        # greedily, not filled with nested sets, so record 'closure' (the
        # solver still receives fill_mode as computed above).
        recorded_fill_mode = ("closure" if self.cart_closure
                              and effective_lookahead < self.max_depth else fill_mode)
        self.fit_info_ = _fit_info(result, recorded_fill_mode,
                                   int(getattr(self.dag_, "n_prefix_leaves", 0)))
        self._attach_fit_config_to_dag(self.dag_)
        info, first = self.fit_info_, self.fit_info_["first_stage"]
        n_frontier = info["n_frontier_leaves"]

        # Warn when the user asked for more depth than was actually solved
        # AND didn't opt into recursive fill. With fill_mode='recursive',
        # prefix leaves are filled with nested Rashomon sets, so the DAG
        # does cover depth=max_depth (composed) and no warning is needed.
        if (info["remaining_depth"] > 0
                and n_frontier > 0
                and fill_mode != 'recursive'):
            warnings.warn(
                f"DAG covers prefix only: solve_depth={info['solve_depth']}, "
                f"max_depth={self.max_depth} (remaining {info['remaining_depth']} "
                f"levels not solved). The {n_frontier} prefix leaves "
                f"in the DAG terminate as majority-class predictions, NOT as "
                f"Rashomon-fills of the remaining depth. Trees emitted by "
                f"get_tree() are depth={info['solve_depth']}, not "
                f"depth={self.max_depth}. To get a depth={self.max_depth} "
                f"Rashomon set, set lookahead_depth=max_depth "
                f"or fill_mode='recursive' to fill the remaining depth with "
                f"nested Rashomon sets.",
                UserWarning, stacklevel=2,
            )

        if self.verbose:
            if first["certified_optimal"]:
                status = "optimal (certified)"
            elif first["search_converged"]:
                status = "gap closed, not certified"
            else:
                status = "approximate"
            la_info = (f", lookahead={info['solve_depth']}, "
                       f"remaining={info['remaining_depth']}"
                       if info["remaining_depth"] > 0 else "")
            print(f"  Search [{status}]: opt={self.opt_:.6f}, "
                  f"gap={first['gap']:.2e}, "
                  f"cert={first['certification_reason']}{la_info}")
            print(f"  Rashomon set: rashomon_bound={self.rashomon_bound_:.6f}, "
                  f"tree_count={self.tree_count_:,}, "
                  f"DAG nodes={self.n_dag_nodes_}")
            if info["remaining_depth"] > 0 and info["completion"]["mode"] == 'none':
                print(f"  NOTE: DAG covers depth-{info['solve_depth']} prefix only "
                      f"({n_frontier} prefix leaves with majority-class "
                      f"placeholders). Trees emitted are depth-{info['solve_depth']}, "
                      f"not depth-{self.max_depth}.")
            print(f"  Total time: {self.fit_time_:.2f}s")
        # The reference-ensemble arrays (and the logits passed to the solver)
        # are only read during fit; dropping them keeps pickles small.
        _discard_fit_only_arrays(self)
        del self._logits
        return self

    # ------------------------------------------------------------------
    # Rashomon set queries
    # ------------------------------------------------------------------

    def tree_count(self):
        """Number of trees in the Rashomon set; an alias of ``tree_count_``.

        Returns
        -------
        int
            ``tree_count_``: the trees within ``rashomon_bound()``.
        """
        check_is_fitted(self, ["tree_count_"])
        return self.tree_count_

    def optimal_objective(self):
        """Objective of the best tree found, from which the bound of the set
        is computed; an alias of ``opt_``.

        Returns
        -------
        float
            ``opt_``.
        """
        check_is_fitted(self, ["opt_"])
        return self.opt_

    def rashomon_bound(self):
        """Objective bound of the stored trees; an alias of ``rashomon_bound_``.

        Returns
        -------
        float
            ``opt * (1 + theta)``, plus the search gap when that is nonzero.
        """
        check_is_fitted(self, ["rashomon_bound_"])
        return self.rashomon_bound_


    # ------------------------------------------------------------------
    # Tree access
    # ------------------------------------------------------------------

    def get_tree(self, idx):
        """Return tree ``idx`` of the Rashomon set (0 is a best tree).

        Parameters
        ----------
        idx : int
            Tree index in [0, tree_count).

        Returns
        -------
        MWNode, MWShapeNode or MWLeaf
            The root of the tree (MWShapeNode splits appear with
            ``shape_mode=True``).
        """
        check_is_fitted(self, ["dag_"])
        if idx < 0 or idx >= self.tree_count_:
            raise IndexError(
                f"Tree index {idx} out of range [0, {self.tree_count_})"
            )
        return dict_to_tree(self.dag_.get_tree(idx))

    def sample_trees(self, n, seed=None, at_target: bool = False,
                     return_indices: bool = False):
        """Sample n trees uniformly at random from the Rashomon set.

        Parameters
        ----------
        n : int
            Number of trees to sample.
        seed : int or None, default=None
            Seed for the draws (0 <= seed < 2**64). None seeds from the
            system clock (non-reproducible).
        at_target : bool, default=False
            Draw only from the reported set (the `tree_count_at_target_`
            trees with cost <= opt*(1+theta)) instead of all
            `tree_count_` trees.
        return_indices : bool, default=False
            Also return the index of each drawn tree (as taken by
            `get_tree`).

        Returns
        -------
        list of MWNode, MWShapeNode or MWLeaf, or (trees, indices) when
        `return_indices`
        """
        check_is_fitted(self, ["dag_"])
        if not return_indices:
            tree_dicts = self.dag_.sample_trees(n, seed=seed, at_target=at_target)
            return [dict_to_tree(d) for d in tree_dicts]
        # The same indices as the bulk sampler, extracted one at a time.
        idxs = [int(i) for i in self.dag_.sample_indices(n, seed=seed, at_target=at_target)]
        return [dict_to_tree(self.dag_.get_tree(i)) for i in idxs], idxs

    def predict_class1_counts(self, X, at_target=True, max_seconds=None):
        """Exact number of trees in the Rashomon set predicting ``classes_[1]``, per row.

        The counts are exact integers, computed on the compact graph that
        stores the set without listing its trees.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        at_target : bool, default=True
            Count over the trees within ``(1 + theta)`` times ``opt_``
            (``tree_count_at_target_`` trees) instead of all ``tree_count_``.
        max_seconds : float, optional
            Time limit of the count.

        Returns
        -------
        counts : ndarray of Python ints (dtype object), shape (n_samples,)
            Trees predicting ``classes_[1]`` for each row.
        n : int
            Number of trees counted.
        info : dict
            Diagnostics of the count (``complete``, ``seconds``, ...).

        Raises
        ------
        RuntimeError
            When the count did not finish within ``max_seconds`` or failed
            its consistency check.
        """
        check_is_fitted(self, ["dag_"])
        X = prediction_data(self, X, "dag_")
        info = self.dag_.class1_counts(X, at_target=at_target, max_seconds=max_seconds)
        if not info["complete"] or info["count_mismatch"]:
            reason = ("did not finish within max_seconds" if not info["complete"]
                      else "failed its consistency check")
            raise RuntimeError(f"The exact count over the Rashomon set {reason}.")
        counts = np.empty(len(info["counts"]), dtype=object)
        counts[:] = info["counts"]
        return counts, int(info["n_trees"]), info

    def predict_probs_exact(self, X, at_target=True, max_seconds=None):
        """Exact probability of ``classes_[1]`` per row: the fraction of the
        reported Rashomon set predicting ``classes_[1]``
        (`predict_class1_counts` / n; 0 if the set is empty).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        at_target : bool, default=True
            Average over the trees within ``(1 + theta)`` times ``opt_``
            (``tree_count_at_target_`` trees) instead of all ``tree_count_``.
        max_seconds : float, optional
            Time limit of the count; RuntimeError when it is reached.

        Returns
        -------
        p : ndarray of shape (n_samples,)
            Fraction of trees predicting ``classes_[1]``.
        """
        counts, n, _ = self.predict_class1_counts(X, at_target=at_target,
                                                  max_seconds=max_seconds)
        if n == 0:
            return np.zeros(len(counts), dtype=np.float64)
        return np.array([c / n for c in counts], dtype=np.float64)

    def predict_stability(self, X):
        """Per-row prediction stability: max(p, 1 - p), with the exact p of
        `predict_probs_exact` (reported set); a shorthand for that
        expression.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        ndarray of shape (n_samples,)
            Values in [0.5, 1.0]; higher means more trees agree.
        """
        probs = self.predict_probs_exact(X)
        return np.maximum(probs, 1.0 - probs)

    def _predict_majority(self, X):
        """Exact majority vote, mapped back to the original binary labels."""
        check_is_fitted(self, ["dag_"])
        return self.classes_[(self.predict_probs_exact(X) >= 0.5).astype(np.intp)]

    def predict_proba(self, X):
        """Columns ``[1 - p, p]`` (following ``classes_``) with ``p`` from
        `predict_probs_exact`. An exact tie (p = 0.5), which ``predict``
        assigns to ``classes_[1]``, gives ``[0.5 - 1e-12, 0.5 + 1e-12]`` so the
        argmax of every row equals ``predict``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        proba : ndarray of shape (n_samples, 2)
        """
        p = self.predict_probs_exact(X)
        proba = np.column_stack([1.0 - p, p])
        proba[p == 0.5] = [0.5 - 1e-12, 0.5 + 1e-12]
        return proba

    def predict(self, X):
        """Predict with the exact majority vote over the reported set (ties
        go to ``classes_[1]``).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        y : ndarray of shape (n_samples,)
            The predicted labels, from ``classes_``.
        """
        return self._predict_majority(X)

    def _attach_fit_config_to_dag(self, dag):
        """Stash the fit-time config on the Rust DAG so save()/load() carry it."""
        threshold_budget = self.threshold_budget
        if isinstance(threshold_budget, np.generic):
            threshold_budget = threshold_budget.item()
        cfg = {
            "max_depth": int(self.max_depth),
            "max_k": int(self.max_k),
            "min_samples_leaf": int(self.min_samples_leaf),
            "reg_alpha": float(self.reg_alpha),
            "path_penalty": float(self.path_penalty),
            "beam_width": int(self.beam_width),
            "transition_delta": float(self.transition_delta),
            "atom_merge_tolerance": float(self.atom_merge_tolerance),
            "exactify_top_k": (int(self.exactify_top_k)
                               if self.exactify_top_k is not None else None),
            # DAG config key keeps its pre-release name (saved bytes unchanged).
            "teacher_alpha": float(self.logit_weight),
            "feature_exclusion": bool(self.feature_exclusion),
            "shape_mode": bool(self.shape_mode),
            "atom_mode": bool(self.atom_mode),
            "dual_family": False,
            "both_modes": False,
            "cleanup_mode": False,
            "dp_criterion": "rush",
            "atom_merge_rule": "l1",
            "search_strategy": "bfs",
            # Python-only settings that wouldn't be reproducible without this
            # (they govern which thresholds got binned, which LightGBM seed, etc.).
            "extras": {
                "candidate_ranking_policy": self.fit_info_["candidate_ranking_policy"],
                "lgbm_n_estimators": int(self.lgbm_n_estimators),
                "lgbm_num_leaves": int(getattr(self, "lgbm_num_leaves_", self.lgbm_num_leaves)),
                "lgbm_max_depth": int(getattr(self, "lgbm_max_depth_", self.lgbm_max_depth)),
                "lgbm_min_samples_leaf": int(getattr(
                    self, "lgbm_min_samples_leaf_", self.lgbm_min_samples_leaf
                    if self.lgbm_min_samples_leaf is not None else self.min_samples_leaf)),
                "max_bins": int(self.max_bins),
                "lgbm_random_state": (int(self.lgbm_random_state)
                                      if self.lgbm_random_state is not None else None),
                "threshold_source": ("fixed" if self.fixed_thresholds is not None
                                     else "lightgbm"),
                "threshold_budget": threshold_budget,
                "binary_threshold_backfill": bool(self.binary_threshold_backfill),
                "fill_mode": self.fit_info_["fill_mode"],
                "fill_parallel": bool(self.fill_parallel),
                "leaf_reg_cap": float(self.leaf_reg_cap),
                "leaf_short_circuit_eps": float(self.leaf_short_circuit_eps),
                "bypass_beam": bool(self.bypass_beam),
                "cost_precision": float(self.cost_precision),
                "lookahead_depth": int(self._effective_lookahead_depth()),
                "cart_closure": bool(self.cart_closure),
                "cart_closure_beam": int(self.cart_closure_beam),
                "cart_closure_mode": str(self.cart_closure_mode),
            },
        }
        dag.set_fit_config(cfg)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def __getstate__(self):
        """Pickle state: the native DAG becomes its v6 bytes."""
        state = dict(super().__getstate__())
        dag = state.get("dag_")
        if dag is not None:
            state["dag_"] = (_DAG_STATE_TAG, bytes(dag.to_bytes(6)))
        return state

    def __setstate__(self, state):
        stored = state.get("dag_")
        if isinstance(stored, tuple) and len(stored) == 2 and stored[0] == _DAG_STATE_TAG:
            state = dict(state)
            state["dag_"] = _restore_dag(stored[1], state["feature_names_"],
                                         state["_train_X"], state["_train_y"],
                                         state["_fit_thresholds"], state["_n_classes"])
        super().__setstate__(state)

    def save(self, path):
        """Save the fitted estimator to one ``.npz`` file (written atomically).

        The file holds the compact graph that stores the set, the training
        rows and thresholds it was fit on, the parameters, and the fitted
        attributes the query methods need; ``RESPLINTER.load`` reads it
        without pickle. Every
        parameter and fitted attribute must be a plain value (number, string,
        None, list, or dict with string keys); otherwise ``save`` raises
        ValueError. The LightGBM reference-ensemble arrays are
        not kept after fit, so they are not saved.

        The graph alone (``dag_``) does not hold the training rows; without
        them the leaf sample counts of its trees are approximate and RID
        cannot run, so save the estimator with this method.

        Parameters
        ----------
        path : str or path-like
            Destination file.
        """
        check_is_fitted(self, ["dag_"])
        params = {}
        for name, value in self.get_params(deep=False).items():
            if name == "fixed_thresholds" and value is not None:
                value = {str(int(j)): np.asarray(v, dtype=np.float64).ravel().tolist()
                         for j, v in value.items()}
            try:
                params[name] = _json_value(value)
            except TypeError as exc:
                raise ValueError(f"Cannot save parameter {name} of type "
                                 f"{type(value).__name__}: only numbers, strings, None, "
                                 "lists and dicts with string keys can be saved.") from exc
        fitted = {}
        for name, value in vars(self).items():
            if name.endswith("_") and name not in _SAVED_SEPARATELY:
                fitted[name] = _saved_value(name, value)
        try:
            fitted["classes_"] = {"dtype": self.classes_.dtype.str,
                                  "values": _json_value(self.classes_.tolist())}
            if np.asarray(fitted["classes_"]["values"],
                          dtype=self.classes_.dtype).tolist() != self.classes_.tolist():
                raise TypeError("classes_ do not round-trip through JSON")
        except (TypeError, ValueError) as exc:
            raise ValueError("Cannot save classes_ of dtype "
                             f"{self.classes_.dtype}: only number or string labels can be "
                             "saved.") from exc
        if hasattr(self, "feature_names_in_"):
            fitted["feature_names_in_"] = [str(v) for v in self.feature_names_in_]
        features, lengths, values = _flatten_thresholds(self._fit_thresholds)
        arrays = dict(
            dag=np.frombuffer(bytes(self.dag_.to_bytes(6)), dtype=np.uint8),
            X=np.asarray(self._train_X, dtype=np.float64),
            y=np.asarray(self._train_y),
            n_classes=np.asarray(self._n_classes, dtype=np.int64),
            threshold_features=features, threshold_lengths=lengths,
            threshold_values=values,
            params_json=np.asarray(json.dumps(params)),
            fitted_json=np.asarray(json.dumps(fitted)),
            format_version=np.asarray(_SAVE_FORMAT_VERSION, dtype=np.int64),
        )
        path = os.fspath(path)
        fd, tmp = tempfile.mkstemp(dir=os.path.dirname(os.path.abspath(path)),
                                   prefix=".resplinter-", suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as fh:
                np.savez(fh, **arrays)
            os.replace(tmp, path)
            umask = os.umask(0)
            os.umask(umask)
            os.chmod(path, 0o666 & ~umask)
        except BaseException:
            if os.path.exists(tmp):
                os.remove(tmp)
            raise

    @classmethod
    def load(cls, path):
        """Load an estimator written by ``save``; raises ValueError for a
        corrupted, truncated or unsupported file.

        Parameters
        ----------
        path : str or path-like

        Returns
        -------
        RESPLINTER
            The fitted estimator, of the class ``load`` is called on.
        """
        try:
            with np.load(path, allow_pickle=False) as data:
                missing = sorted(set(_SAVED_ARRAYS) - set(data.files))
                if missing:
                    raise ValueError(f"missing arrays {missing}")
                version = int(data["format_version"])
                if version != _SAVE_FORMAT_VERSION:
                    raise ValueError(f"unsupported format_version {version}")
                params = json.loads(str(data["params_json"]))
                fitted = json.loads(str(data["fitted_json"]))
                dag_bytes = data["dag"].tobytes()
                X = np.ascontiguousarray(data["X"], dtype=np.float64)
                y = np.ascontiguousarray(data["y"])
                n_classes = int(data["n_classes"])
                thresholds = _unflatten_thresholds(data["threshold_features"],
                                                   data["threshold_lengths"],
                                                   data["threshold_values"])
            fitted = _migrate_saved_fitted(params, fitted)
            missing = sorted(set(_REQUIRED_FITTED) - set(fitted))
            if missing:
                raise ValueError(f"missing fitted attributes {missing}")
            if params.get("fixed_thresholds") is not None:
                params["fixed_thresholds"] = {int(j): v for j, v
                                              in params["fixed_thresholds"].items()}
            model = cls(**params)
            classes = fitted.pop("classes_")
            model.classes_ = np.asarray(classes["values"], dtype=np.dtype(classes["dtype"]))
            if "feature_names_in_" in fitted:
                model.feature_names_in_ = np.asarray(fitted.pop("feature_names_in_"),
                                                     dtype=object)
            for name, value in fitted.items():
                setattr(model, name, value)
            model.lgbm_thresholds_ = {j: list(v) for j, v in thresholds.items()}
            model._train_X, model._train_y = X, y
            model._fit_thresholds, model._n_classes = thresholds, n_classes
            model.dag_ = _restore_dag(dag_bytes, model.feature_names_, X, y, thresholds,
                                      n_classes)
            # Fixed settings were not constructor parameters in older files.
            # Read them from the saved graph, including after a migrated save.
            config = model.dag_.fit_config
            if "atom_mode" not in params and "atom_mode" in config:
                model.atom_mode = config["atom_mode"]
            extras = config.get("extras", {})
            if "binary_threshold_backfill" not in params and "binary_threshold_backfill" in extras:
                model.binary_threshold_backfill = extras["binary_threshold_backfill"]
        except FileNotFoundError:
            raise
        except (OSError, EOFError, KeyError, AttributeError, TypeError, ValueError,
                zipfile.BadZipFile) as exc:
            raise ValueError(f"{os.fspath(path)!r} is not a valid saved RESPLINTER "
                             f"file: {exc}") from exc
        return model

    # ------------------------------------------------------------------
    # Summary / display
    # ------------------------------------------------------------------

    def guarantee(self):
        """Report what the fit guarantees, in user terms.

        Search convergence, optimality of the best tree and completeness of
        the set are separate facts, each relative to the candidate splits
        (the LightGBM thresholds), not to all multiway trees. Completeness
        is further relative to the splits the search keeps: ``beam_width``
        partitions per interval cell and the ``transition_delta`` bin
        compression (a smaller ``beam_width`` can give a smaller set that
        is still reported complete).

        Returns
        -------
        dict
            The keys below.

        - ``search_converged``: the first search stage converged.
        - ``best_tree_certified_optimal``: that search proved its best tree
          optimal among trees built from the candidate splits.
        - ``optimality_scope``: what optimality is relative to
          (``"candidate splits"``).
        - ``optimality_note``: a sentence on why optimality is or is not
          certified.
        - ``optimality_gap``, ``lower_bound``: the search's gap and lower
          bound when it certified optimality, else None (a search that
          screens candidates reports a nominal zero gap that certifies
          nothing).
        - ``best_objective``: the reference for the objective bound: the
          objective of the best tree found. When the set is flagged
          incomplete it can differ from the minimum objective in the set
          (``incomplete_reason`` then says the best tree is not in the set).
        - ``search_objective``: objective of the best tree the first search
          stage found; with ``lookahead_depth < max_depth`` it is the prefix
          tree's objective, with majority-class leaves below the prefix.
        - ``theta``: the requested relative tolerance.
        - ``objective_bound``: trees with objective at most this bound are
          in the set (``best_objective * (1 + theta)``, plus the search gap
          when that is nonzero).
        - ``inclusion_exact``: tree objectives are compared with the bound
          exactly (exact cost keys and a leaf-only objective), so no tree is
          admitted or left out by rounding.
        - ``coverage_scope``: what completeness is relative to
          (``"candidate splits kept by the search (beam_width,
          transition_delta)"``).
        - ``complete``: the set holds every tree within ``objective_bound``
          that can be built from the candidate splits kept by the search;
          trees whose splits were dropped by ``beam_width`` or merged by
          ``transition_delta`` are not counted.
        - ``incomplete_reason``: the failed completeness conditions, as
          plain sentences ("" when complete; the solver's codes are in
          ``fit_info_["incomplete_reason_codes"]``).
        - ``count_saturated``: a tree count saturated (counts are then lower
          bounds).
        - ``completeness_note``: a caveat on completeness, e.g. for Rashomon
          importance computed on this set (None when the set is complete).
        """
        check_is_fitted(self, ["fit_info_"])
        info, first = self.fit_info_, self.fit_info_["first_stage"]
        rashomon_complete = getattr(self, "rashomon_complete_", None)
        codes = info.get("incomplete_reason_codes") or ""
        if rashomon_complete is None and not codes:
            codes = "no completeness certificate"
        reason = " ".join(_incomplete_sentences(codes))
        inclusion_exact = (
            info.get("exact_cost_keys") is True
            and self.path_penalty == 0.0 and self.reg_alpha == 1.0
            and "non_leaf_objective" not in codes.split(";")
        )
        if rashomon_complete is True:
            caveat = None
        elif rashomon_complete is False:
            caveat = reason or "Enumeration did not establish completeness."
        else:
            caveat = "The backend did not supply a completeness certificate."
        certified = bool(first.get("certified_optimal", False))
        return {
            "search_converged": bool(first.get("search_converged", False)),
            "best_tree_certified_optimal": certified,
            "optimality_scope": "candidate splits",
            "optimality_note": _optimality_note(first.get("certification_reason")),
            "optimality_gap": first.get("gap") if certified else None,
            "lower_bound": first.get("lower_bound") if certified else None,
            "best_objective": float(self.opt_),
            "search_objective": float(first["objective"]),
            "theta": float(self.theta),
            "objective_bound": float(self.rashomon_bound_),
            "inclusion_exact": inclusion_exact,
            "coverage_scope": _COVERAGE_SCOPE,
            "complete": rashomon_complete is True,
            "incomplete_reason": reason,
            "count_saturated": bool(getattr(self, "count_saturated_", False)),
            "completeness_note": caveat,
        }

    def summary(self):
        """Short text report: the set size, the best objective and bound, and
        what is certified (see ``guarantee``).

        Returns
        -------
        str
        """
        check_is_fitted(self, ["fit_info_"])
        report = self.guarantee()
        lines = [
            f"RESPLINTER: {self.tree_count_:,} trees, {self.n_dag_nodes_:,} graph nodes",
            f"  Best objective: {report['best_objective']:.6f}",
            f"  Objective bound: {report['objective_bound']:.6f} "
            f"(exact inclusion: {report['inclusion_exact']})",
            f"  Theta bound: {float(self.theta_bound_):.6f} "
            f"(best objective x (1 + theta))",
            f"  Best tree certified optimal: "
            f"{'yes' if report['best_tree_certified_optimal'] else 'no'} "
            f"(within the candidate splits)",
            f"  Rashomon set: "
            f"{'complete for the splits kept by the search' if report['complete'] else 'completeness unknown' if getattr(self, 'rashomon_complete_', None) is None else 'incomplete'}",
        ]
        if report["incomplete_reason"]:
            lines.append(f"  Incomplete reason: {report['incomplete_reason']}")
        lines.append(f"  Fit time: {self.fit_time_:.2f}s")
        return "\n".join(lines)

    def __repr__(self):
        if hasattr(self, 'tree_count_'):
            return (
                f"{type(self).__name__}(trees={self.tree_count_:,}, "
                f"opt={self.opt_:.4f}, bound={self.rashomon_bound_:.4f}, "
                f"graph_nodes={self.n_dag_nodes_}, "
                f"theta={self.theta}, depth={self.max_depth}, k={self.max_k})"
            )
        return (
            f"{type(self).__name__}(max_depth={self.max_depth}, max_k={self.max_k}, "
            f"reg={self.reg}, theta={self.theta})"
        )
