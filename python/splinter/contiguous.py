"""SPLINTER with contiguous multiway splits, using the Rust backend."""


import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted

from .tree import (dict_to_tree, predict_encoded, annotate_class_counts,
                   predict_proba_encoded,
                   count_leaves, tree_depth, internal_sample_count,
                   objective as tree_objective)
from ._validation import training_data, prediction_data, reset_fitted_state
from .rules import export_text
from .thresholds import _prepare, _discard_fit_only_arrays
from .multiway_solver import solve_branch_and_bound


def _default_lookahead_depth(max_depth):
    """Default exact prefix depth: ``max_depth // 2`` (= ceil((max_depth - 1) / 2))."""
    return max(0, int(max_depth) // 2)


# ---------------------------------------------------------------------------
# Main classifier
# ---------------------------------------------------------------------------

class _SPLINTER(ClassifierMixin, BaseEstimator):
    """Contiguous multiway SPLINTER using the Rust branch-and-bound backend.

    The public API intentionally keeps only the knobs that define the model:
    tree depth, lookahead, arity, penalties, threshold budget, and the local
    candidate beam/top-k limits. Thresholds and reference-ensemble scores are
    learned from the supplied training data unless ``precomputed_thresholds``
    is given.

    Parameters
    ----------
    max_depth : int, default=5
        Maximum tree depth.
    lookahead_depth : int or None, default=None
        Depth of the top levels chosen by the branch-and-bound search over
        the top-ranked candidate splits. ``None`` uses
        ``ceil((max_depth - 1) / 2)``.
    max_k : int, default=3
        Maximum split arity (K = 2..max_k).
    reg : float, default=0.001
        Per-leaf regularization penalty.
    min_samples_leaf : int, default=20
        Minimum child samples enforced by the SPLINTER solver.
    time_limit : float or None, default=None
        Maximum time in seconds for the branch-and-bound search of the top levels.
    max_bins : int, default=1024
        Maximum LightGBM bins per feature.
    lgbm_min_samples_leaf : int, default=20
        Minimum child samples used only when fitting the LightGBM reference
        ensemble. This is intentionally separate from ``min_samples_leaf`` so
        solver leaf-size tuning can reuse the same cached thresholds.
    threshold_budget : None, float, int, or str, default=None
        ``None`` keeps every learned threshold. A float in ``(0, 1)`` keeps
        the shortest LightGBM-gain-ranked prefix covering that cumulative gain
        fraction. An integer keeps exactly that many top-ranked thresholds.
        String forms like ``"0.99"`` and ``"1024"`` are accepted for config
        files.
    precomputed_thresholds : ThresholdCache, dict, path-like, or None, default=None
        Reuse thresholds from a previous LightGBM reference-ensemble run. Bare
        threshold dictionaries are intentionally rejected so SPLINTER does
        not silently run on quantile or hand-authored thresholds.
    precomputed_logits : array-like or None, default=None
        Optional reference-ensemble logits aligned to the rows passed to
        ``fit``. Used only in shape mode (ShapeSPLINTER), to reuse the same
        LightGBM ordering with cached thresholds.
    beam_width : int, default=128
        Number of local DP partitions kept per feature/order/criterion.
    top_k : int, default=3
        Number of local split candidates recursively exactified at each search
        node.
    path_penalty : float, default=0.0
        Per-internal-node path penalty, normalized by the training set size.
    ub_margin : float or None, default=None
        Additional upper-bound sibling-pruning margin. ``None`` disables this
        heuristic pruning.
    verbose : bool, default=False
        Print progress during solving.
    """

    shape_mode = False

    def __init__(
        self,
        max_depth=5,
        lookahead_depth=None,
        max_k=3,
        reg=0.001,
        min_samples_leaf=20,
        time_limit=None,
        max_bins=1024,
        lgbm_n_estimators=512,
        lgbm_min_samples_leaf=20,
        lgbm_random_state=42,
        threshold_budget=None,
        precomputed_thresholds=None,
        precomputed_logits=None,
        beam_width=128,
        top_k=3,
        path_penalty=0.0,
        ub_margin=None,
        verbose=False,
    ):
        self.max_depth = max_depth
        self.lookahead_depth = lookahead_depth
        self.max_k = max_k
        self.reg = reg
        self.min_samples_leaf = min_samples_leaf
        self.time_limit = time_limit
        self.verbose = verbose
        self.max_bins = max_bins
        self.lgbm_n_estimators = lgbm_n_estimators
        self.lgbm_min_samples_leaf = lgbm_min_samples_leaf
        self.lgbm_random_state = lgbm_random_state
        self.beam_width = beam_width
        self.path_penalty = path_penalty
        self.threshold_budget = threshold_budget
        self.precomputed_thresholds = precomputed_thresholds
        self.precomputed_logits = precomputed_logits
        self.top_k = top_k
        self.ub_margin = ub_margin

    def fit(self, X, y, feature_names=None, sample_weight=None):
        """Fit the tree.

        ``sample_weight`` (one finite, nonnegative value per row, positive
        sum) is normalized to sum to 1 before solving, so only relative
        weights matter and ``ones(n)`` gives exactly the unweighted fit.
        The solver counts each row as a whole number of copies: its weight
        rounded to the nearest multiple of the mean weight (half rounds up).
        A weight below half the mean therefore has no effect, and
        ``min_samples_leaf`` counts these copies. Leaf ``class_counts`` hold
        the same copy totals.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)
        feature_names : list of str, optional
        sample_weight : array-like of shape (n_samples,), optional

        Returns
        -------
        self
        """
        reset_fitted_state(self)
        self._logits = None
        X, y = training_data(self, X, y, feature_names)
        n, m = X.shape
        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight, dtype=np.float64)
            if (sample_weight.shape != (n,) or not np.isfinite(sample_weight).all()
                    or np.any(sample_weight < 0) or not sample_weight.sum() > 0):
                raise ValueError("sample_weight must be finite, nonnegative, contain one value "
                                 "per row, and have a positive sum.")
            sample_weight = sample_weight / sample_weight.sum()
        if self.lookahead_depth is None:
            effective_lookahead = _default_lookahead_depth(self.max_depth)
        else:
            effective_lookahead = int(self.lookahead_depth)
        effective_lookahead = min(effective_lookahead, int(self.max_depth))
        self.lookahead_depth_ = effective_lookahead
        thresholds_for_rust = _prepare(self, X, y)

        # Every lookahead depth, including pure greedy (zero), uses the
        # native greedy completion and its feasible leaf incumbent. Rust
        # builds the full tree: exact search above the lookahead frontier,
        # then the greedy completion below it.
        rust_max_depth = self.max_depth
        rust_greedy_extra_depth = 0
        default_closure_depth = self.max_depth - effective_lookahead
        rust_closure_depth = (
            default_closure_depth if default_closure_depth >= 0 else None
        )

        solver_kwargs = dict(
            x=X,
            y=y,
            n_classes=self.n_classes_,
            reg=self.reg,
            max_depth=rust_max_depth,
            max_k=self.max_k,
            min_samples_leaf=self.min_samples_leaf,
            feature_exclusion=False,
            time_limit=self.time_limit,
            initial_used_features=[],
            feature_names=self.feature_names_,
            parallel=False,
            top_n=20,
            thresholds_per_feature=thresholds_for_rust,
            dp_criterion="gini",
            beam_width=self.beam_width,
            reg_alpha=1.0,
            local_thresh_scan=False,
            scan_beam=20,
            transition_delta=0.0,
            materialize_top_n=1,
            path_penalty=self.path_penalty,
            greedy_extra_depth=rust_greedy_extra_depth,
            sample_weights=(
                np.ascontiguousarray(sample_weight, dtype=np.float64)
                if sample_weight is not None else None
            ),
            ub_margin=self.ub_margin,
            n_workers=1,
            lazy_explore_delta=None,
            closure_depth=rust_closure_depth,
            closure_beam=1,
            closure_parallel=False,
            closure_share_memo=False,
            closure_criterion=None,
            closure_split_ub_gap=float('inf'),
            closure_feature_top_k=None,
            closure_min_n_for_multiway=8 * self.min_samples_leaf,
            closure_unsafe_filter_depth_threshold=0,
            phase0_greedy_prerank_depth=0,
            feature_types=None,
            atom_mode=False,
            atom_merge_tolerance=0.0,
            top_k=self.top_k,
            top_k_auto_sqrt_candidates=False,
            top_k_min_beam=1,
            atom_merge_rule='l1',
            atom_wilson_z=1.959964,
            cleanup_mode=False,
            unsafe_filter_depth_threshold=0,
            admissibility_strict=False,
        )
        if self.shape_mode:
            solver_kwargs.update(
                shape_mode=True,
                shape_sort_smoothing=0.0,
                shape_boundary_tax=1.0,
                shape_sort_per_atom_teacher_prior=True,
                teacher_logit=self._logits,
            )

        result = solve_branch_and_bound(**solver_kwargs)

        self.certified_optimal_ = result.get("certified_optimal", result["is_optimal"])
        self.search_converged_ = result.get("search_converged", self.certified_optimal_)
        self.optimality_gap_ = result["final_gap"]
        self.lower_bound_ = result["lowerbound"]
        self.search_objective_ = result["objective"]
        self.timed_out_ = result["timed_out"]
        self.termination_reason_ = result["termination_reason"]
        self.n_subproblems_ = result["n_subproblems"]
        self.search_time_ = result["elapsed"]
        certified = self.certified_optimal_
        self.fit_info_ = {
            "is_optimal": result["is_optimal"],
            "certification_mode": result.get(
                "certification_mode", "exact" if certified else "exploratory"),
            "certification_reason": result.get(
                "certification_reason", "certified" if certified else "uncertified"),
            "unsafe_heuristics_used": result.get("unsafe_heuristics_used", False),
            "n_explored": result["n_explored"],
            "candidate_ranking_policy": result["candidate_ranking_policy"],
            "profile": result.get("profile", None),
        }

        if self.verbose:
            if result["timed_out"]:
                status = "time limit reached"
            elif self.certified_optimal_:
                status = "optimal (certified)"
            elif self.search_converged_:
                status = "gap closed, not certified"
            else:
                status = "best found"
            print(f"  Search [{status}] obj={result['objective']:.6f}, "
                  f"LB={result['lowerbound']:.6f}, gap={result['final_gap']:.2e}, "
                  f"reason={result['termination_reason']}, "
                  f"cert={self.fit_info_['certification_reason']}, "
                  f"subproblems={result['n_subproblems']}, "
                  f"explored={result['n_explored']}, "
                  f"time={result['elapsed']:.2f}s")

        self.tree_ = dict_to_tree(result["tree"])
        copies = None
        if sample_weight is not None:
            # The solver's per-row copies, round(w * n) with halves rounded up
            # (Rust's f64::round on nonnegative values).
            scaled = sample_weight * n
            copies = np.trunc(scaled) + (scaled - np.trunc(scaled) >= 0.5)
        annotate_class_counts(self.tree_, X, y, self.n_classes_, weights=copies)
        _discard_fit_only_arrays(self)
        return self

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, X):
        """Predict the class of each row.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        y : ndarray of shape (n_samples,)
            The predicted labels, from ``classes_``.
        """
        X = prediction_data(self, X)
        return self.classes_[predict_encoded(self.tree_, X)]

    def predict_proba(self, X):
        """Class probabilities (columns follow ``classes_``): the training
        class shares of each row's leaf (weighted when the fit was). A leaf
        without training rows gives one-hot on its prediction; a tied leaf
        gets 1e-12 extra on its predicted class, so the argmax of every row
        equals ``predict``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes)
        """
        X = prediction_data(self, X)
        return predict_proba_encoded(self.tree_, X, self.n_classes_)

    def score(self, X, y, sample_weight=None):
        """Accuracy of ``predict`` on (X, y).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)
        sample_weight : array-like of shape (n_samples,), optional

        Returns
        -------
        float
        """
        return np.average(self.predict(X) == np.asarray(y), weights=sample_weight)

    # ------------------------------------------------------------------
    # Tree statistics
    # ------------------------------------------------------------------

    def objective(self):
        """Training objective of the fitted tree.

        Returns
        -------
        float
            Misclassification rate plus ``reg`` per leaf plus the path
            penalty, on the training rows.
        """
        check_is_fitted(self, ["tree_"])
        return tree_objective(self.tree_, self.reg, self.n_total_, self.path_penalty)

    def num_leaves(self):
        """Number of leaves of the fitted tree.

        Returns
        -------
        int
        """
        check_is_fitted(self, ["tree_"])
        return count_leaves(self.tree_)

    def tree_depth(self):
        """Depth of the fitted tree (0 for a single leaf).

        Returns
        -------
        int
        """
        check_is_fitted(self, ["tree_"])
        return tree_depth(self.tree_)


    def avg_path_length(self):
        """Average number of splits on a training row's root-to-leaf path.

        Returns
        -------
        float
        """
        check_is_fitted(self, ["tree_"])
        return internal_sample_count(self.tree_) / self.n_total_

    # ------------------------------------------------------------------
    # Pretty-printing
    # ------------------------------------------------------------------

    def __str__(self):
        if getattr(self, "tree_", None) is None:
            return repr(self)
        header = (
            f"{self.__class__.__name__}(max_depth={self.max_depth}, max_k={self.max_k}, "
            f"reg={self.reg}): {self.num_leaves()} leaves, depth {self.tree_depth()}, "
            f"training objective {self.objective():.6f}"
        )
        return header + "\n" + export_text(self)
