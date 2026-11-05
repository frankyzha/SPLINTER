"""LicketySPLINTER wrapper using the Rust branch-and-bound backend."""

import time
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted

from .tree import (MWLeaf, MWNode, MWShapeNode, dict_to_tree, predict_encoded,
                   annotate_class_counts, predict_proba_encoded,
                   count_leaves, tree_depth, internal_sample_count,
                   objective as tree_objective)
from ._validation import training_data, prediction_data, reset_fitted_state
from .rules import export_text
from .thresholds import _prepare, _discard_fit_only_arrays
from .multiway_solver import CANDIDATE_RANKING_POLICY, solve_branch_and_bound


class _LicketySPLINTER(ClassifierMixin, BaseEstimator):
    """Recursive per-hop multiway tree using the Rust solver per node.

    Parameters
    ----------
    max_depth : int, default=5
    hop_size : int, default=1
        Number of tree levels solved in each recursive Lickety hop.
    max_k : int, default=3
    reg : float, default=0.001
    min_samples_leaf : int, default=20
        Minimum child samples enforced by the SPLINTER solver.
    lgbm_min_samples_leaf : int, default=20
        Minimum child samples used only when fitting the LightGBM reference
        ensemble.
    time_limit : float or None, default=None
    feature_exclusion : bool, default=False
    ub_margin : float or None, default=None
        Additional upper-bound sibling-pruning margin. ``None`` disables this
        heuristic pruning.
    closure : bool, default=True
        When enabled, each solver call also completes the subtree below the
        current hop greedily, which gives a bound and a candidate tree.
    closure_resplit : bool, default=True
        If true (the default), discard the greedy subtree after each hop and
        search again at each frontier leaf. If false, keep the subtree from a
        single solver call instead.
    verbose : bool, default=False
    """

    def __init__(self, max_depth=5, hop_size=1, max_k=3,
                 reg=0.001, min_samples_leaf=20, time_limit=None,
                 feature_exclusion=False, verbose=False,
                 lgbm_n_estimators=512,
                 lgbm_min_samples_leaf=20,
                 max_bins=1024,
                 lgbm_random_state=42,
                 shape_mode=False,
                 beam_width=128,
                 reg_alpha=1.0,
                 materialize_top_n=1,
                 path_penalty=0.0,
                 ub_margin=None,
                 threshold_budget=None,
                 precomputed_thresholds=None,
                 precomputed_logits=None,
                 closure=True,
                 closure_beam=1,
                 closure_resplit=True,
                 min_n_for_multiway=None,
                 top_k=3,
                 feature_types=None):
        self.max_depth = max_depth
        self.hop_size = hop_size
        self.max_k = max_k
        self.reg = reg
        self.min_samples_leaf = min_samples_leaf
        self.time_limit = time_limit
        self.feature_exclusion = feature_exclusion
        self.verbose = verbose
        self.lgbm_n_estimators = lgbm_n_estimators
        self.lgbm_min_samples_leaf = lgbm_min_samples_leaf
        self.max_bins = max_bins
        self.lgbm_random_state = lgbm_random_state
        self.shape_mode = shape_mode
        self.beam_width = beam_width
        self.reg_alpha = reg_alpha
        self.materialize_top_n = materialize_top_n
        self.path_penalty = path_penalty
        self.ub_margin = ub_margin
        self.threshold_budget = threshold_budget
        self.precomputed_thresholds = precomputed_thresholds
        self.precomputed_logits = precomputed_logits
        self.closure = closure
        self.min_n_for_multiway = min_n_for_multiway
        self.closure_beam = closure_beam
        self.closure_resplit = closure_resplit
        self.top_k = top_k
        self.feature_types = feature_types

    def fit(self, X, y, feature_names=None):
        """Fit the tree.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training rows; finite values.
        y : array-like of shape (n_samples,)
            Class labels (2 to 16 classes).
        feature_names : list of str, optional
            Names used when printing the tree; taken from the columns of a
            DataFrame ``X`` when not given.

        Returns
        -------
        self
        """
        reset_fitted_state(self)
        self._logits = None
        X, y = training_data(self, X, y, feature_names)
        n, m = X.shape

        self.hop_size_ = max(1, min(int(self.hop_size), int(self.max_depth)))

        self._fit_start_time = time.time()
        self._n_solver_calls = 0

        self._closure_min_n = (self.min_n_for_multiway
                               if self.min_n_for_multiway is not None
                               else 8 * self.min_samples_leaf)

        # The shared preparation path resets logits on every fit and reuses
        # the cached reference ensemble without retraining it in recursive hops.
        self._thresholds_for_rust = _prepare(self, X, y)
        self._closure_feature_ranking = sorted(
            range(m), key=lambda j: (-len(self._thresholds_for_rust[j]), j)
        )

        indices = np.arange(n)
        self.tree_ = self._recursive_solve(
            X, y, indices,
            depth_budget=self.max_depth,
            reg_scaled=self.reg,
            n_root=n,
            used_features=set(),
        )

        annotate_class_counts(self.tree_, X, y, self.n_classes_)
        self.fit_info_ = {"candidate_ranking_policy": CANDIDATE_RANKING_POLICY}
        _discard_fit_only_arrays(self)

        self._fit_time = time.time() - self._fit_start_time
        if self.verbose:
            print(f"  LicketySPLINTER fit complete: {self._n_solver_calls} solver calls, "
                  f"time={self._fit_time:.2f}s")
        return self


    # ------------------------------------------------------------------
    # Core recursive method
    # ------------------------------------------------------------------

    def _recursive_solve(self, X, y, indices, depth_budget, reg_scaled,
                         n_root, used_features):
        n_sub = len(indices)

        if depth_budget <= 0 or n_sub < 2:
            return self._make_leaf(y, indices, n_root)

        effective_hop = min(self.hop_size_, depth_budget)

        X_sub = np.ascontiguousarray(X[indices], dtype=np.float64)
        y_sub = np.ascontiguousarray(y[indices], dtype=np.uint8)

        self._n_solver_calls += 1
        if self.verbose:
            print(f"    Solver call #{self._n_solver_calls}: n={n_sub}, "
                  f"hop={effective_hop}, reg={reg_scaled:.6f}, "
                  f"depth_budget={depth_budget}, "
                  f"used_feats={sorted(used_features)}")

        thresholds = self._thresholds_for_rust

        # With closure, Rust also completes the subtree below the hop greedily.
        if self.closure:
            rust_max_depth = depth_budget
            rust_greedy_extra_depth = 0
            closure_depth = depth_budget - effective_hop
            rust_closure_depth = closure_depth if closure_depth >= 0 else None
        else:
            rust_max_depth = effective_hop
            rust_greedy_extra_depth = min(depth_budget - effective_hop, effective_hop)
            rust_closure_depth = None

        # Row-subset the reference-ensemble logits so they align with X_sub.
        # They are root-row indexed; recursive calls operate on sub-supports.
        # Only shape mode reads them (they order the bins).
        logits_sub = (self._logits[indices]
                      if self.shape_mode and self._logits is not None else None)

        result = solve_branch_and_bound(
            x=X_sub,
            y=y_sub,
            n_classes=self.n_classes_,
            reg=reg_scaled,
            max_depth=rust_max_depth,
            max_k=self.max_k,
            min_samples_leaf=self.min_samples_leaf,
            feature_exclusion=self.feature_exclusion,
            time_limit=self.time_limit,
            initial_used_features=sorted(used_features),
            feature_names=self.feature_names_,
            thresholds_per_feature=thresholds,
            dp_criterion="gini",
            beam_width=self.beam_width,
            reg_alpha=self.reg_alpha,
            adaptive_beam=False,
            transition_delta=0.08,
            materialize_top_n=self.materialize_top_n,
            path_penalty=self.path_penalty,
            ub_margin=self.ub_margin,
            greedy_extra_depth=rust_greedy_extra_depth,
            cheap_boundary_greedy=False,
            closure_depth=rust_closure_depth,
            closure_beam=self.closure_beam,
            closure_share_memo=True,
            closure_split_ub_gap=float('inf'),
            closure_min_n_for_multiway=self._closure_min_n,
            closure_feature_top_k=None,
            closure_feature_ranking=self._closure_feature_ranking,
            closure_thresholds_per_feature=self._thresholds_for_rust,
            closure_unsafe_filter_depth_threshold=0,
            unsafe_filter_depth_threshold=0,
            feature_types=self.feature_types,
            shape_mode=self.shape_mode,
            atom_mode=False,
            shape_boundary_tax=(1.0 if self.shape_mode else 0.0),
            shape_sort_per_atom_teacher_prior=bool(self.shape_mode),
            top_k=self.top_k,
            top_k_min_beam=1,
            teacher_logit=logits_sub,
            teacher_alpha=1.0,
        )

        if self.verbose:
            if result["timed_out"]:
                status = "time limit reached"
            elif result.get("certified_optimal", result["is_optimal"]):
                status = "optimal (certified)"
            elif result.get("search_converged", False):
                status = "gap closed, not certified"
            else:
                status = "best found"
            print(f"      Search [{status}] "
                  f"obj={result['objective']:.6f}, "
                  f"LB={result['lowerbound']:.6f}, "
                  f"gap={result['final_gap']:.2e}, "
                  f"reason={result['termination_reason']}, "
                  f"cert={result.get('certification_reason', 'n/a')}, "
                  f"time={result['elapsed']:.2f}s")

        prefix_tree = dict_to_tree(result["tree"])

        # Rust already completed the full subtree; no greedy fill is needed.
        if self.closure:
            if self.closure_resplit and effective_hop < depth_budget:
                # Keep the greedy completion as candidate picker and bound,
                # but discard the subtree and search again on each child's
                # rows, so every hop builds a fresh greedy completion.
                truncated = self._truncate_to_depth(prefix_tree, effective_hop)
                if isinstance(truncated, MWLeaf):
                    self._rescale_losses(prefix_tree, n_sub, n_root)
                    return prefix_tree
                return self._fill_leaves(
                    truncated, X, y, indices,
                    depth_budget, self.reg, n_root, used_features,
                )
            self._rescale_losses(prefix_tree, n_sub, n_root)
            return prefix_tree

        remaining = depth_budget - effective_hop
        if remaining <= 0 and not result["timed_out"]:
            self._rescale_losses(prefix_tree, n_sub, n_root)
            return prefix_tree

        # If solver returned a bare leaf (no viable split found within
        # effective_hop levels), there's nothing to fill — return as-is.
        # Only safe to drop remaining budget if effective_hop == 1 (shallower
        # search can't do better) or remaining == 0 (nothing left).
        # Otherwise re-try with hop_size=1 (single greedy step) so we
        # consume one level at a time and don't silently discard the budget.
        if isinstance(prefix_tree, MWLeaf):
            if remaining <= 0 or effective_hop <= 1:
                if n_sub > 0:
                    prefix_tree.loss = prefix_tree.loss * n_sub / n_root
                return prefix_tree
            # B&B with la>1 found no split; retry one level at a time so
            # we don't jump past the remaining depth.
            reg_scaled = self.reg * (n_root / n_sub)
            return self._recursive_solve(
                X, y, indices,
                depth_budget=remaining,
                reg_scaled=reg_scaled,
                n_root=n_root,
                used_features=used_features,
            )

        # Pass depth_budget and decrement through prefix internal nodes
        # so leaves at any depth get correct remaining budget (important
        # when solver timed out and left shallow leaves).
        return self._fill_leaves(
            prefix_tree, X, y, indices,
            depth_budget, self.reg, n_root, used_features,
        )

    # ------------------------------------------------------------------
    # Recursive leaf expansion
    # ------------------------------------------------------------------

    def _fill_leaves(self, node, X, y, indices,
                     remaining_depth, reg_original, n_root, used_features):
        if isinstance(node, MWLeaf):
            n_sub = len(indices)
            if remaining_depth <= 0 or n_sub < 2:
                if n_sub > 0:
                    node.loss = node.loss * n_sub / n_root
                return node

            reg_scaled = reg_original * (n_root / n_sub)

            return self._recursive_solve(
                X, y, indices,
                depth_budget=remaining_depth,
                reg_scaled=reg_scaled,
                n_root=n_root,
                used_features=used_features,
            )

        feat_vals = X[indices, node.feature_index]
        thresh_arr = np.asarray(node.thresholds)
        bin_assignments = np.searchsorted(thresh_arr, feat_vals, side='right')
        if isinstance(node, MWShapeNode):
            bin_assignments = np.array([node.group_map[b] for b in bin_assignments])

        child_used = (used_features | {node.feature_index}
                      if self.feature_exclusion else used_features)

        new_children = []
        for i, child in enumerate(node.children):
            mask = bin_assignments == i
            child_indices = indices[mask]
            if len(child_indices) == 0:
                if isinstance(child, MWLeaf):
                    child.loss = 0.0
                new_children.append(child)
            else:
                new_children.append(
                    self._fill_leaves(child, X, y, child_indices,
                                      remaining_depth - 1, reg_original,
                                      n_root, child_used)
                )

        if isinstance(node, MWShapeNode):
            return MWShapeNode(
                feature_index=node.feature_index,
                feature_name=node.feature_name,
                thresholds=node.thresholds,
                group_map=node.group_map,
                children=new_children,
            )
        return MWNode(
            feature_index=node.feature_index,
            feature_name=node.feature_name,
            thresholds=node.thresholds,
            children=new_children,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_leaf(self, y, indices, n_root):
        n = len(indices)
        if n == 0:
            return MWLeaf(prediction=0, n_samples=0, loss=0.0)
        counts = np.bincount(y[indices], minlength=self.n_classes_)
        prediction = int(counts.argmax())
        loss = (n - counts.max()) / n_root
        return MWLeaf(prediction=prediction, n_samples=n, loss=loss)

    def _rescale_losses(self, node, n_sub, n_root):
        if isinstance(node, MWLeaf):
            node.loss = node.loss * n_sub / n_root
            return
        for child in node.children:
            self._rescale_losses(child, n_sub, n_root)

    def _truncate_to_depth(self, node, depth):
        """Replace any subtree at depth >= `depth` with a placeholder leaf.
        _fill_leaves will re-solve each placeholder via _recursive_solve
        on the rows routed to it."""
        if isinstance(node, MWLeaf):
            return node
        if depth <= 0:
            return MWLeaf(prediction=0, n_samples=0, loss=0.0)
        new_children = [self._truncate_to_depth(c, depth - 1) for c in node.children]
        if isinstance(node, MWShapeNode):
            return MWShapeNode(
                feature_index=node.feature_index,
                feature_name=node.feature_name,
                thresholds=node.thresholds,
                group_map=node.group_map,
                children=new_children,
            )
        return MWNode(
            feature_index=node.feature_index,
            feature_name=node.feature_name,
            thresholds=node.thresholds,
            children=new_children,
        )

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
        class shares of each row's leaf. A leaf without training rows gives
        one-hot on its prediction; a tied leaf gets 1e-12 extra on its
        predicted class, so the argmax of every row equals ``predict``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes)
        """
        X = prediction_data(self, X)
        return predict_proba_encoded(self.tree_, X, self.n_classes_)

    def score(self, X, y):
        """Accuracy of ``predict`` on (X, y).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)

        Returns
        -------
        float
        """
        return np.mean(self.predict(X) == np.asarray(y))

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
        return tree_objective(self.tree_, self.reg, self.n_total_, self.path_penalty, self.reg_alpha)

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
            f"{type(self).__name__}(max_depth={self.max_depth}, max_k={self.max_k}, "
            f"reg={self.reg}): {self.num_leaves()} leaves, depth {self.tree_depth()}, "
            f"training objective {self.objective():.6f}"
        )
        return header + "\n" + export_text(self)
