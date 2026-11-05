"""Sparse multiway trees with lookahead over candidate splits."""

from importlib.metadata import PackageNotFoundError, version as _version

from .contiguous import _SPLINTER
from .nonlinear import _ShapeSPLINTER
from .lickety_splinter import _LicketySPLINTER
from .resplinter import _RESPLINTER
from .rules import export_text, explain

try:
    __version__ = _version("splinter")
except PackageNotFoundError:  # running from a source tree without installing
    __version__ = "0+unknown"

# Each public class lists only its public parameters. The private base
# classes take more; the public classes fix those at the base defaults as
# class attributes (the base methods read them as ``self.<name>``), so they
# are not parameters: ``get_params``, ``set_params``, ``clone`` and ``repr``
# see only the public ones.


class SPLINTER(_SPLINTER):
    """Sparse multiway decision tree classifier with lookahead search.

    Each split sends a row to one of 2 to ``max_k`` branches by comparing a
    single feature with sorted thresholds. The top ``lookahead_depth`` levels
    are chosen by a branch-and-bound search over the top-ranked candidate
    splits (``certified_optimal_`` reports whether optimality was
    certified); the levels below are completed greedily. The objective is the training misclassification rate
    plus ``reg`` per leaf (plus ``path_penalty`` per internal node on each
    row's path). Candidate thresholds come from a LightGBM model fit on the
    same data.

    Parameters
    ----------
    max_depth : int, default=5
        Maximum depth of the tree.
    lookahead_depth : int or None, default=None
        Number of top levels chosen by the branch-and-bound search over the
        top-ranked candidate splits; the levels below are completed
        greedily. None uses ``max_depth // 2``.
    max_k : int, default=3
        Maximum number of branches per split (splits have 2 to ``max_k``).
    reg : float, default=0.001
        Penalty per leaf, added to the training misclassification rate.
    min_samples_leaf : int, default=20
        Minimum number of training rows in every leaf.
    time_limit : float or None, default=None
        Time limit in seconds for the branch-and-bound search; None for no limit. On
        timeout the best tree found so far is returned.
    max_bins : int, default=1024
        Maximum number of LightGBM bins per feature, which bounds the
        candidate thresholds per feature.
    lgbm_n_estimators : int, default=512
        Number of boosting rounds of the LightGBM model that supplies the
        candidate thresholds.
    lgbm_min_samples_leaf : int, default=20
        Minimum rows per LightGBM leaf. It is separate from
        ``min_samples_leaf`` so that one set of thresholds can be reused
        across tree settings.
    lgbm_random_state : int or None, default=42
        Seed of the LightGBM model.
    threshold_budget : None, float, int or str, default=None
        Keep only part of the LightGBM thresholds, ranked by LightGBM split
        gain. None keeps all. A float in (0, 1) keeps the fewest
        top-ranked thresholds that carry that fraction of the total gain; an
        integer keeps that many. Strings such as ``"0.99"`` or ``"1024"``
        are accepted.
    path_penalty : float, default=0.0
        Penalty per internal node on each row's root-to-leaf path, averaged
        over the training rows. Favors shorter explanations.
    verbose : bool, default=False
        Print search progress.

    Advanced parameters
    -------------------
    beam_width : int, default=128
        Number of partial partitions kept per feature by the dynamic program
        that proposes multiway splits.
    top_k : int, default=3
        Number of best candidate splits per node that the search evaluates
        exactly.
    ub_margin : float or None, default=None
        Margin for pruning a candidate split by its upper bound. None
        disables this heuristic pruning, which can lose optimality.
    precomputed_thresholds : ThresholdCache, dict, path-like or None, default=None
        Thresholds to reuse instead of fitting LightGBM: a cache from
        ``splinter.thresholds.fit_thresholds``, or the path of one saved
        with ``save_threshold_cache``. A plain ``{feature: thresholds}``
        dictionary is rejected.

    Attributes
    ----------
    tree_ : MWNode or MWLeaf
        Root of the fitted tree (see ``splinter.export_text``).
    classes_ : ndarray of shape (n_classes,)
        Class labels.
    n_classes_ : int
        Number of classes.
    n_features_in_ : int
        Number of features seen in ``fit``.
    feature_names_ : list of str
        Feature names used in the tree (``x0``, ``x1``, ... when none given).
    feature_names_in_ : ndarray of shape (n_features_in_,)
        Feature names seen in ``fit``; defined only when they are strings.
    lookahead_depth_ : int
        Number of top levels chosen by the branch-and-bound search.
    lgbm_thresholds_ : dict of int to list of float
        Candidate thresholds per feature index.
    certified_optimal_ : bool
        Whether the search proved the top ``lookahead_depth_`` levels
        optimal among the candidate splits.
    search_converged_ : bool
        Whether the search ran to its end (not stopped early); it does not
        imply ``certified_optimal_``.
    optimality_gap_ : float
        ``search_objective_`` minus ``lower_bound_``.
    lower_bound_ : float
        Lower bound on the objective reported by the search; a proof only
        when ``certified_optimal_`` is True.
    search_objective_ : float
        Objective of the best tree the search found (``objective()`` is the
        objective of the fitted tree).
    timed_out_ : bool
        Whether the search stopped at ``time_limit``.
    termination_reason_ : str
        Why the search stopped.
    n_subproblems_ : int
        Number of subproblems the search created.
    search_time_ : float
        Search time in seconds.
    fit_info_ : dict
        Further details of the search (certification details, solver
        profile, candidate ranking rule), for diagnostics.

    See Also
    --------
    ShapeSPLINTER : Splits that group bins non-contiguously.
    LicketySPLINTER : Faster, level-by-level variant.
    RESPLINTER : Set of all near-optimal trees.

    Examples
    --------
    >>> from sklearn.datasets import load_breast_cancer
    >>> from splinter import SPLINTER, export_text
    >>> X, y = load_breast_cancer(return_X_y=True)
    >>> model = SPLINTER(max_depth=3, lgbm_n_estimators=16).fit(X, y)
    >>> print(export_text(model))  # doctest: +SKIP
    """

    precomputed_logits = None

    def __init__(self, max_depth=5, lookahead_depth=None, max_k=3, reg=0.001,
                 min_samples_leaf=20, time_limit=None, max_bins=1024,
                 lgbm_n_estimators=512, lgbm_min_samples_leaf=20, lgbm_random_state=42,
                 threshold_budget=None, precomputed_thresholds=None, beam_width=128,
                 top_k=3, path_penalty=0.0, ub_margin=None, verbose=False):
        self.max_depth = max_depth
        self.lookahead_depth = lookahead_depth
        self.max_k = max_k
        self.reg = reg
        self.min_samples_leaf = min_samples_leaf
        self.time_limit = time_limit
        self.max_bins = max_bins
        self.lgbm_n_estimators = lgbm_n_estimators
        self.lgbm_min_samples_leaf = lgbm_min_samples_leaf
        self.lgbm_random_state = lgbm_random_state
        self.threshold_budget = threshold_budget
        self.precomputed_thresholds = precomputed_thresholds
        self.beam_width = beam_width
        self.top_k = top_k
        self.path_penalty = path_penalty
        self.ub_margin = ub_margin
        self.verbose = verbose

    def fit(self, X, y, feature_names=None, sample_weight=None):
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
        sample_weight : array-like of shape (n_samples,), optional
            Nonnegative row weights with a positive sum. Only relative
            weights matter: each row counts as its weight divided by the mean
            weight, rounded to a whole number of copies.

        Returns
        -------
        self : SPLINTER
            The fitted estimator.
        """
        return super().fit(X, y, feature_names=feature_names, sample_weight=sample_weight)


class ShapeSPLINTER(_ShapeSPLINTER):
    """SPLINTER with nonlinear splits that group bins non-contiguously.

    Like ``SPLINTER``, but a split may send non-adjacent bins of a feature
    to the same branch (for example, low and high values to one branch and
    middle values to another). Bins are ordered by the LightGBM model's
    scores, and contiguous groups in that order become branches.

    Parameters
    ----------
    max_depth : int, default=5
        Maximum depth of the tree.
    lookahead_depth : int or None, default=None
        Number of top levels chosen by the branch-and-bound search over the
        top-ranked candidate splits; the levels below are completed
        greedily. None uses ``max_depth // 2``.
    max_k : int, default=3
        Maximum number of branches per split (splits have 2 to ``max_k``).
    reg : float, default=0.001
        Penalty per leaf, added to the training misclassification rate.
    min_samples_leaf : int, default=20
        Minimum number of training rows in every leaf.
    time_limit : float or None, default=None
        Time limit in seconds for the branch-and-bound search; None for no limit. On
        timeout the best tree found so far is returned.
    max_bins : int, default=1024
        Maximum number of LightGBM bins per feature, which bounds the
        candidate thresholds per feature.
    lgbm_n_estimators : int, default=512
        Number of boosting rounds of the LightGBM model that supplies the
        candidate thresholds and the bin order.
    lgbm_min_samples_leaf : int, default=20
        Minimum rows per LightGBM leaf. It is separate from
        ``min_samples_leaf`` so that one set of thresholds can be reused
        across tree settings.
    lgbm_random_state : int or None, default=42
        Seed of the LightGBM model.
    threshold_budget : None, float, int or str, default=None
        Keep only part of the LightGBM thresholds, ranked by LightGBM split
        gain. None keeps all. A float in (0, 1) keeps the fewest
        top-ranked thresholds that carry that fraction of the total gain; an
        integer keeps that many. Strings such as ``"0.99"`` or ``"1024"``
        are accepted.
    path_penalty : float, default=0.0
        Penalty per internal node on each row's root-to-leaf path, averaged
        over the training rows. Favors shorter explanations.
    verbose : bool, default=False
        Print search progress.

    Advanced parameters
    -------------------
    beam_width : int, default=128
        Number of partial partitions kept per feature by the dynamic program
        that proposes multiway splits.
    top_k : int, default=3
        Number of best candidate splits per node that the search evaluates
        exactly.
    ub_margin : float or None, default=None
        Margin for pruning a candidate split by its upper bound. None
        disables this heuristic pruning, which can lose optimality.
    precomputed_thresholds : ThresholdCache, dict, path-like or None, default=None
        Thresholds to reuse instead of fitting LightGBM: a cache from
        ``splinter.thresholds.fit_thresholds``, or the path of one saved
        with ``save_threshold_cache``. A plain ``{feature: thresholds}``
        dictionary is rejected.
    precomputed_logits : array-like or None, default=None
        LightGBM raw scores of the training rows, of shape
        (n_samples, n_classes), or (n_samples,) as the class-1 margin for
        binary data. They set the bin order and replace the scores stored
        in ``precomputed_thresholds``.

    Attributes
    ----------
    tree_ : MWShapeNode, MWNode or MWLeaf
        Root of the fitted tree (see ``splinter.export_text``).
    classes_ : ndarray of shape (n_classes,)
        Class labels.
    n_classes_ : int
        Number of classes.
    n_features_in_ : int
        Number of features seen in ``fit``.
    feature_names_ : list of str
        Feature names used in the tree (``x0``, ``x1``, ... when none given).
    feature_names_in_ : ndarray of shape (n_features_in_,)
        Feature names seen in ``fit``; defined only when they are strings.
    lookahead_depth_ : int
        Number of top levels chosen by the branch-and-bound search.
    lgbm_thresholds_ : dict of int to list of float
        Candidate thresholds per feature index.
    certified_optimal_ : bool
        Whether the search proved the top ``lookahead_depth_`` levels
        optimal among the candidate splits.
    search_converged_ : bool
        Whether the search ran to its end (not stopped early); it does not
        imply ``certified_optimal_``.
    optimality_gap_ : float
        ``search_objective_`` minus ``lower_bound_``.
    lower_bound_ : float
        Lower bound on the objective reported by the search; a proof only
        when ``certified_optimal_`` is True.
    search_objective_ : float
        Objective of the best tree the search found (``objective()`` is the
        objective of the fitted tree).
    timed_out_ : bool
        Whether the search stopped at ``time_limit``.
    termination_reason_ : str
        Why the search stopped.
    n_subproblems_ : int
        Number of subproblems the search created.
    search_time_ : float
        Search time in seconds.
    fit_info_ : dict
        Further details of the search (certification details, solver
        profile, candidate ranking rule), for diagnostics.

    See Also
    --------
    SPLINTER : Splits into contiguous intervals only.
    LicketySPLINTER : Faster, level-by-level variant (``shape_mode=True``).

    Examples
    --------
    >>> from sklearn.datasets import load_breast_cancer
    >>> from splinter import ShapeSPLINTER, export_text
    >>> X, y = load_breast_cancer(return_X_y=True)
    >>> model = ShapeSPLINTER(max_depth=3, lgbm_n_estimators=16).fit(X, y)
    >>> print(export_text(model))  # doctest: +SKIP
    """

    def __init__(self, max_depth=5, lookahead_depth=None, max_k=3, reg=0.001,
                 min_samples_leaf=20, time_limit=None, max_bins=1024,
                 lgbm_n_estimators=512, lgbm_min_samples_leaf=20, lgbm_random_state=42,
                 threshold_budget=None, precomputed_thresholds=None, precomputed_logits=None,
                 beam_width=128, top_k=3, path_penalty=0.0, ub_margin=None, verbose=False):
        self.max_depth = max_depth
        self.lookahead_depth = lookahead_depth
        self.max_k = max_k
        self.reg = reg
        self.min_samples_leaf = min_samples_leaf
        self.time_limit = time_limit
        self.max_bins = max_bins
        self.lgbm_n_estimators = lgbm_n_estimators
        self.lgbm_min_samples_leaf = lgbm_min_samples_leaf
        self.lgbm_random_state = lgbm_random_state
        self.threshold_budget = threshold_budget
        self.precomputed_thresholds = precomputed_thresholds
        self.precomputed_logits = precomputed_logits
        self.beam_width = beam_width
        self.top_k = top_k
        self.path_penalty = path_penalty
        self.ub_margin = ub_margin
        self.verbose = verbose

    def fit(self, X, y, feature_names=None, sample_weight=None):
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
        sample_weight : array-like of shape (n_samples,), optional
            Nonnegative row weights with a positive sum. Only relative
            weights matter: each row counts as its weight divided by the mean
            weight, rounded to a whole number of copies.

        Returns
        -------
        self : ShapeSPLINTER
            The fitted estimator.
        """
        return super().fit(X, y, feature_names=feature_names, sample_weight=sample_weight)


class LicketySPLINTER(_LicketySPLINTER):
    """Faster SPLINTER variant that grows the tree one level at a time.

    At each node it runs one search for the best split, looking at the
    whole subtree below (completed greedily) to score the split, then
    repeats on each child. This scales to deeper trees and larger data than
    ``SPLINTER``, but only one level is chosen per search, so the tree is
    not optimal overall.

    Parameters
    ----------
    max_depth : int, default=5
        Maximum depth of the tree.
    max_k : int, default=3
        Maximum number of branches per split (splits have 2 to ``max_k``).
    reg : float, default=0.001
        Penalty per leaf, added to the training misclassification rate.
    min_samples_leaf : int, default=20
        Minimum number of training rows in every leaf.
    time_limit : float or None, default=None
        Time limit in seconds for each search; None for no limit.
    feature_exclusion : bool, default=False
        If True, a feature is used at most once on any root-to-leaf path.
    verbose : bool, default=False
        Print search progress.
    shape_mode : bool, default=False
        If True, allow splits that group bins non-contiguously, as in
        ``ShapeSPLINTER``.
    path_penalty : float, default=0.0
        Penalty per internal node on each row's root-to-leaf path, averaged
        over the training rows. Favors shorter explanations.
    lgbm_n_estimators : int, default=512
        Number of boosting rounds of the LightGBM model that supplies the
        candidate thresholds.
    lgbm_min_samples_leaf : int, default=20
        Minimum rows per LightGBM leaf. It is separate from
        ``min_samples_leaf`` so that one set of thresholds can be reused
        across tree settings.
    max_bins : int, default=1024
        Maximum number of LightGBM bins per feature, which bounds the
        candidate thresholds per feature.
    lgbm_random_state : int or None, default=42
        Seed of the LightGBM model.
    threshold_budget : None, float, int or str, default=None
        Keep only part of the LightGBM thresholds, ranked by LightGBM split
        gain. None keeps all. A float in (0, 1) keeps the fewest
        top-ranked thresholds that carry that fraction of the total gain; an
        integer keeps that many. Strings such as ``"0.99"`` or ``"1024"``
        are accepted.

    Advanced parameters
    -------------------
    beam_width : int, default=128
        Number of partial partitions kept per feature by the dynamic program
        that proposes multiway splits.
    top_k : int, default=3
        Number of best candidate splits per node that the search evaluates
        exactly.
    ub_margin : float or None, default=None
        Margin for pruning a candidate split by its upper bound. None
        disables this heuristic pruning.
    closure : bool, default=True
        If True, each search also completes the subtree below each candidate
        split greedily and uses it to score the split. If False, the search
        looks only one level ahead.
    closure_resplit : bool, default=True
        If True, discard the greedy subtree after each level and search
        again at each child. If False, keep the subtree from a single search.
    precomputed_thresholds : ThresholdCache, dict, path-like or None, default=None
        Thresholds to reuse instead of fitting LightGBM: a cache from
        ``splinter.thresholds.fit_thresholds``, or the path of one saved
        with ``save_threshold_cache``. A plain ``{feature: thresholds}``
        dictionary is rejected.
    precomputed_logits : array-like or None, default=None
        LightGBM raw scores of the training rows, of shape
        (n_samples, n_classes), or (n_samples,) as the class-1 margin for
        binary data. Used only with ``shape_mode=True``, where they set the
        bin order and replace the scores stored in
        ``precomputed_thresholds``.

    Attributes
    ----------
    tree_ : MWNode, MWShapeNode or MWLeaf
        Root of the fitted tree (see ``splinter.export_text``).
    classes_ : ndarray of shape (n_classes,)
        Class labels.
    n_classes_ : int
        Number of classes.
    n_features_in_ : int
        Number of features seen in ``fit``.
    feature_names_ : list of str
        Feature names used in the tree (``x0``, ``x1``, ... when none given).
    feature_names_in_ : ndarray of shape (n_features_in_,)
        Feature names seen in ``fit``; defined only when they are strings.
    lgbm_thresholds_ : dict of int to list of float
        Candidate thresholds per feature index.
    hop_size_ : int
        Number of levels the search decides at a time (always 1).
    fit_info_ : dict
        Further details of the fit (the candidate ranking rule), for
        diagnostics.

    See Also
    --------
    SPLINTER : Branch-and-bound search over the top levels.
    ShapeSPLINTER : Splits that group bins non-contiguously.

    Examples
    --------
    >>> from sklearn.datasets import load_breast_cancer
    >>> from splinter import LicketySPLINTER, export_text
    >>> X, y = load_breast_cancer(return_X_y=True)
    >>> model = LicketySPLINTER(max_depth=3, lgbm_n_estimators=16).fit(X, y)
    >>> print(export_text(model))  # doctest: +SKIP
    """

    hop_size = 1
    materialize_top_n = 1
    closure_beam = 1
    min_n_for_multiway = None
    reg_alpha = 1.0
    feature_types = None

    def __init__(self, max_depth=5, max_k=3, reg=0.001, min_samples_leaf=20, time_limit=None,
                 feature_exclusion=False, verbose=False, lgbm_n_estimators=512,
                 lgbm_min_samples_leaf=20, max_bins=1024, lgbm_random_state=42,
                 shape_mode=False, beam_width=128, path_penalty=0.0, ub_margin=None,
                 threshold_budget=None, precomputed_thresholds=None, precomputed_logits=None,
                 closure=True, closure_resplit=True, top_k=3):
        self.max_depth = max_depth
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
        self.path_penalty = path_penalty
        self.ub_margin = ub_margin
        self.threshold_budget = threshold_budget
        self.precomputed_thresholds = precomputed_thresholds
        self.precomputed_logits = precomputed_logits
        self.closure = closure
        self.closure_resplit = closure_resplit
        self.top_k = top_k


class RESPLINTER(_RESPLINTER):
    """Rashomon set of near-optimal sparse multiway decision trees.

    Finds the best multiway tree, then every tree whose objective is within
    a factor ``1 + theta`` of it (the Rashomon set), built from the same
    candidate splits. The set is stored as a compact graph of trees in
    which subtrees common to many trees are stored once, so very large sets
    fit in memory and can be counted, sampled and averaged exactly. Binary
    classification only. The objective is the
    training misclassification rate plus ``reg`` per leaf.

    The defaults reproduce the configuration of the paper's experiments:
    depth 4, the top ``max_depth // 2`` levels enumerated and each of their
    leaves filled with its own nested Rashomon set (no greedy completion),
    and candidate thresholds from a LightGBM model with 31 leaves per tree
    and 200 trees.

    Parameters
    ----------
    max_depth : int, default=4
        Maximum depth of the trees.
    lookahead_depth : int or None, default=None
        Number of top levels enumerated before the levels below are filled;
        None uses ``max_depth // 2``. Below ``max_depth``, each leaf of the
        top levels is filled with its own nested Rashomon set (see
        ``fill_mode`` and ``cart_closure``); ``lookahead_depth=max_depth``
        enumerates every level in one search. Both give a complete set
        unless ``cart_closure`` or ``time_limit`` cuts the search short, or
        the best full-depth tree found is not in the set
        (``fit_info_["full_depth_tree_in_set"]`` is False, reason code
        ``upfront_anchor_not_in_dag``).
    max_k : int, default=4
        Maximum number of branches per split (splits have 2 to ``max_k``).
    reg : float, default=0.01
        Penalty per leaf, added to the training misclassification rate.
    theta : float, default=0.05
        Relative tolerance: the set holds the trees whose objective is at
        most ``(1 + theta)`` times the best objective (see ``guarantee``).
    feature_exclusion : bool, default=False
        If True, a feature is used at most once on any root-to-leaf path.
    time_limit : float or None, default=None
        Time limit in seconds for each search stage; None for no limit. A
        stage that times out leaves the set incomplete.
    verbose : bool, default=False
        Print progress.
    lgbm_n_estimators : int, default=200
        Number of boosting rounds of the LightGBM model that supplies the
        candidate thresholds.
    lgbm_num_leaves : int, default=31
        Maximum leaves per LightGBM tree.
    lgbm_max_depth : int, default=-1
        Maximum depth of each LightGBM tree; -1 for no limit.
    lgbm_min_samples_leaf : int or None, default=None
        Minimum rows per LightGBM leaf; None means 1. The trees of the set
        have no minimum leaf size; filter them with
        ``splinter.explore.select(table, min_leaf=...)``.
    max_bins : int, default=1024
        Maximum number of LightGBM bins per feature, which bounds the
        candidate thresholds per feature.
    lgbm_random_state : int or None, default=42
        Seed of the LightGBM model.
    threshold_budget : None, float, int or str, default=None
        Keep only part of the LightGBM thresholds, ranked by LightGBM split
        gain. None keeps all. A float in (0, 1) keeps the fewest
        top-ranked thresholds that carry that fraction of the total gain; an
        integer keeps that many. Strings such as ``"0.99"`` or ``"1024"``
        are accepted. Fewer thresholds give fewer representable trees.
    fixed_thresholds : dict or None, default=None
        Candidate thresholds ``{feature index: list of thresholds}`` to use
        instead of fitting LightGBM. Every feature must be a key (``[]``
        means never split on it); cannot be combined with
        ``threshold_budget``. ``splinter.rid.compute_rid`` uses it to fit
        every bootstrap sample on the same thresholds.

    Advanced parameters
    -------------------
    beam_width : int, default=4
        Number of partial partitions kept per interval of the dynamic
        program that proposes multiway splits. Larger values keep more
        candidate splits, so the set can hold more trees.
    transition_delta : float, default=0.02
        Bin compression tolerance; lower values merge fewer neighboring
        bins and keep more candidate splits.
    fill_mode : {'full', 'recursive'}, default='full'
        How the levels below ``lookahead_depth`` are filled: 'recursive'
        fills each leaf of the top levels with its own nested Rashomon set.
        'full' switches to 'recursive' whenever the lookahead depth is below
        ``max_depth`` (as with the default ``lookahead_depth=None``);
        'recursive' raises ValueError otherwise.
    cart_closure : bool, default=False
        With ``lookahead_depth`` below ``max_depth``, complete the levels
        below the lookahead greedily (CART-style) instead of enumerating
        them. Faster, but the set is then reported incomplete.
    shape_mode : bool, default=False
        If True, allow splits that group bins non-contiguously, as in
        ``ShapeSPLINTER``.

    Attributes
    ----------
    dag_ : object
        The compact graph that stores every tree in the set (advanced use).
    tree_count_ : int
        Number of trees stored, those within ``rashomon_bound_``.
    tree_count_at_target_ : int
        Number of trees within ``theta_bound_``; equals ``tree_count_``
        unless the search left an optimality gap.
    opt_ : float
        Objective of the best tree found.
    theta_bound_ : float
        ``opt_ * (1 + theta)``: the bound of the reported set (the trees
        that ``predict``, ``sample_trees(at_target=True)`` and RID use).
    rashomon_bound_ : float
        Objective bound of the stored trees: ``theta_bound_`` plus the
        optimality gap of the search, so at least ``theta_bound_``.
    rashomon_complete_ : bool or None
        Whether the set holds every tree within ``rashomon_bound_`` built
        from the candidate splits kept by the search (see ``guarantee``).
    rashomon_incomplete_reason_ : list of str or None
        Why the set is not complete, one sentence per failed condition
        (empty when complete).
    count_saturated_ : bool
        Whether a tree count overflowed; counts are then lower bounds.
    classes_ : ndarray of shape (2,)
        Class labels.
    n_features_in_ : int
        Number of features seen in ``fit``.
    feature_names_ : list of str
        Feature names used in the trees (``x0``, ``x1``, ... when none given).
    lgbm_thresholds_ : dict of int to list of float
        Candidate thresholds per feature index.
    n_dag_nodes_ : int
        Number of nodes in that graph; a measure of how compactly the set
        is stored.
    fit_time_ : float
        Search time in seconds.
    fit_info_ : dict
        Further details of the search, for diagnostics: ``first_stage``
        (the search for the best tree), ``threshold_tightened`` (whether
        the bound was reset to the best tree of the set), ``completion``
        (the fill below ``lookahead_depth``), ``incomplete_reason_codes``
        (the raw codes behind ``rashomon_incomplete_reason_``) and
        ``raw_result`` (the solver's full report).

    Fixed settings
    --------------
    The class fixes further settings as class attributes. They are not
    parameters, but a subclass can override them:

    - ``binary_threshold_backfill = False``: True gives binary 0/1
      features that LightGBM never splits on the threshold 0.5.
    - ``n_threads = 1``: threads of the lower-bound computation (the same
      trees at any value).
    - ``completion_batches = 0``: solve the fill below ``lookahead_depth``
      in this many parallel batches (0 = one batch; the same trees).
    - ``completion_batch_threads = None``: threads per batch (None =
      ``n_threads`` divided by the batches).
    - ``interval_lower_bound = True``: a tighter lower bound for the last
      two levels (an exact interval search; the set can only grow).
    - ``interval_min_arity = 3``: use that bound only when ``max_k`` is at
      least this.
    - ``lower_bound_memo_size``, ``depth1_memo_size``,
      ``interval_cache_size``: cache sizes (the same trees at any value).
    - ``screen_time_cap = 60.0``: time limit in seconds of each root
      candidate solve when the first search screens candidates.
    - ``cost_precision = 1e9``: resolution of the integer cost keys.
    - ``fill_parallel = True``: fill the frontier leaves in parallel.
    - ``leaf_reg_cap = 0.1``, ``leaf_short_circuit_eps = 1e-4``: recorded in
      saved files; the search does not read them.
    - ``logits = None``, ``logit_weight = 1.0``: LightGBM scores passed to
      the search (None = those of the fitted LightGBM model) and their
      weight.
    - ``cart_closure_beam = 1``, ``cart_closure_mode = "greedy"``: beam and
      split criterion (Gini) of the greedy completion of ``cart_closure``.
    - ``bypass_beam = False``: True ignores ``beam_width`` when enumerating
      the set (many more candidate splits and much more memory).
    - ``atom_mode = True``, ``atom_merge_tolerance = 0.0``: at each node,
      merge neighboring bins with equal class shares, so no split
      separates them (tolerance 0 merges only exact equals).
    - ``exactify_top_k = None``: no cap on the candidate splits searched
      at each node.
    - ``min_samples_leaf = 1``, ``reg_alpha = 1.0``,
      ``path_penalty = 0.0``: the leaf-only objective; other values raise
      ValueError.

    See Also
    --------
    SPLINTER : A single near-optimal tree.
    splinter.rid.compute_rid : Variable importance over the Rashomon set.
    splinter.explore.rashomon_table : One row per tree, for filtering.

    Examples
    --------
    >>> from sklearn.datasets import load_breast_cancer
    >>> from splinter import RESPLINTER
    >>> X, y = load_breast_cancer(return_X_y=True)
    >>> rset = RESPLINTER(max_depth=2, lgbm_n_estimators=16).fit(X, y)
    >>> print(rset.summary())  # doctest: +SKIP
    """

    min_samples_leaf = 1
    reg_alpha = 1.0
    path_penalty = 0.0
    cost_precision = 1e9
    cart_closure_beam = 1
    cart_closure_mode = "greedy"
    atom_mode = True
    atom_merge_tolerance = 0.0
    logit_weight = 1.0
    logits = None
    exactify_top_k = None
    leaf_reg_cap = 0.1
    leaf_short_circuit_eps = 1e-4
    fill_parallel = True
    bypass_beam = False
    binary_threshold_backfill = False
    n_threads = 1
    completion_batches = 0
    completion_batch_threads = None
    interval_lower_bound = True
    interval_min_arity = 3
    lower_bound_memo_size = 3_000_000
    depth1_memo_size = 16_000_000
    interval_cache_size = 100_000
    screen_time_cap = 60.0

    def __init__(self, max_depth=4, lookahead_depth=None, max_k=4, reg=0.01, theta=0.05,
                 feature_exclusion=False, time_limit=None, beam_width=4, transition_delta=0.02,
                 lgbm_n_estimators=200, lgbm_num_leaves=31, lgbm_max_depth=-1,
                 lgbm_min_samples_leaf=None, max_bins=1024, lgbm_random_state=42,
                 cart_closure=False, shape_mode=False, threshold_budget=None, fill_mode="full",
                 fixed_thresholds=None, verbose=False):
        self.max_depth = max_depth
        self.lookahead_depth = lookahead_depth
        self.max_k = max_k
        self.reg = reg
        self.theta = theta
        self.feature_exclusion = feature_exclusion
        self.time_limit = time_limit
        self.beam_width = beam_width
        self.transition_delta = transition_delta
        self.lgbm_n_estimators = lgbm_n_estimators
        self.lgbm_num_leaves = lgbm_num_leaves
        self.lgbm_max_depth = lgbm_max_depth
        self.lgbm_min_samples_leaf = lgbm_min_samples_leaf
        self.max_bins = max_bins
        self.lgbm_random_state = lgbm_random_state
        self.cart_closure = cart_closure
        self.shape_mode = shape_mode
        self.threshold_budget = threshold_budget
        self.fill_mode = fill_mode
        self.fixed_thresholds = fixed_thresholds
        self.verbose = verbose


__all__ = ["SPLINTER", "ShapeSPLINTER", "LicketySPLINTER", "RESPLINTER", "export_text",
           "explain"]
