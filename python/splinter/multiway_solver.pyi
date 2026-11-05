"""Type stub for the compiled Rust extension `splinter.multiway_solver`.

The functions take many keyword arguments that the Python estimators fill in;
see `splinter.contiguous`, `splinter.lickety_splinter` and
`splinter.resplinter` for how they are called.
"""
from typing import Any

import numpy.typing as npt

CANDIDATE_RANKING_POLICY: str

def solve_branch_and_bound(
    x: Any, y: Any, n_classes: Any, reg: Any, max_depth: Any, max_k: Any,
    min_samples_leaf: Any, feature_exclusion: Any, time_limit: Any,
    initial_used_features: Any, feature_names: Any, warm_start_ub: Any = None,
    parallel: Any = False, top_n: Any = 20, thresholds_per_feature: Any = None,
    dp_criterion: Any = "gini", beam_width: Any = 128, n_total_override: Any = None,
    reg_alpha: Any = 1.0, local_thresh_scan: Any = False, scan_beam: Any = 20,
    adaptive_beam: Any = False, transition_delta: Any = 0.08,
    materialize_top_n: Any = 1, shape_mode: Any = False,
    shape_sort_smoothing: Any = 0.0, path_penalty: Any = 0.0,
    greedy_extra_depth: Any = 0, sample_weights: Any = None, ub_margin: Any = None,
    n_workers: Any = None, lazy_explore_delta: Any = None,
    cheap_boundary_greedy: Any = False, closure_depth: Any = None,
    closure_beam: Any = 3, closure_parallel: Any = False,
    closure_share_memo: Any = False, closure_criterion: Any = None,
    closure_thresholds_per_feature: Any = None, closure_split_ub_gap: Any = None,
    closure_feature_top_k: Any = None, closure_feature_ranking: Any = None,
    closure_min_n_for_multiway: Any = None,
    closure_unsafe_filter_depth_threshold: Any = 0, feature_types: Any = None,
    atom_mode: Any = False, shape_boundary_tax: Any = 0.0,
    shape_sort_per_atom_teacher_prior: Any = False, atom_merge_tolerance: Any = 0.0,
    top_k: Any = None, top_k_auto_sqrt_candidates: Any = False, top_k_min_beam: Any = 8,
    atom_merge_rule: Any = "l1", atom_wilson_z: Any = 1.959964,
    cleanup_mode: Any = False, atom_block_compression_purity: Any = 0.0,
    teacher_logit: Any = None, teacher_alpha: Any = 1.0,
    unsafe_filter_depth_threshold: Any = 0, admissibility_strict: Any = False,
    phase0_greedy_prerank_depth: Any = 0, eqp_inline_bounds: Any = True,
) -> dict[str, Any]: ...
def solve_rashomon(
    x: Any, y: Any, n_classes: Any, reg: Any, max_depth: Any, max_k: Any, theta: Any,
    lookahead_depth: Any = None, min_samples_leaf: Any = 1,
    feature_exclusion: Any = False, time_limit: Any = None, feature_names: Any = None,
    beam_width: Any = 4, reg_alpha: Any = 1.0, transition_delta: Any = 0.02,
    thresholds_per_feature: Any = None, cost_precision: Any = 1e9,
    path_penalty: Any = 0.0, closure_depth: Any = None, closure_beam: Any = 1,
    closure_parallel: Any = False, closure_share_memo: Any = False,
    closure_criterion: Any = None, feature_types: Any = None, shape_mode: Any = False,
    shape_sort_smoothing: Any = 0.0, atom_mode: Any = False,
    shape_boundary_tax: Any = 0.0, teacher_logit: Any = None, teacher_alpha: Any = 1.0,
    unsafe_filter_depth_threshold: Any = 0, admissibility_strict: Any = False,
    top_k: Any = None, atom_merge_tolerance: Any = 0.0, recursive_fill: Any = False,
    leaf_reg_cap: Any = 0.1, leaf_short_circuit_eps: Any = 1e-4,
    fill_parallel: Any = True, verbose: Any = False, bypass_beam: Any = False,
    lb_threads: Any = 1, stepb_batches: Any = 0, stepb_batch_lb_threads: Any = None,
    lb_interval: Any = False, lb_interval_mink: Any = 3,
    lb_memo_max_entries: Any = 3000000, lb_d1s_max_entries: Any = 16000000,
    lb_interval_cache_max: Any = 100000, phase1_screen_cap_s: Any = 60.0,
) -> dict[str, Any]: ...
def rashomon_select_debug(
    x: Any, y: Any, n_classes: Any, reg: Any, max_depth: Any, max_k: Any,
    rashomon_bound: Any, support_idx: Any, thresholds_per_feature: Any = None,
    beam_width: Any = 4, reg_alpha: Any = 1.0, transition_delta: Any = 0.02,
    atom_mode: Any = False, atom_merge_tolerance: Any = 0.0, bypass_beam: Any = False,
    shape_mode: Any = False, shape_sort_smoothing: Any = 0.0, path_penalty: Any = 0.0,
    min_samples_leaf: Any = 1, feature_types: Any = None,
) -> Any: ...

class PyRashomonDag:
    @property
    def tree_count(self) -> str: ...
    @property
    def tree_count_u64(self) -> int: ...
    @property
    def tree_count_u128(self) -> int: ...
    @property
    def count_saturated(self) -> bool: ...
    @property
    def n_nodes(self) -> int: ...
    @property
    def opt(self) -> float: ...
    @property
    def rashomon_bound(self) -> float: ...
    @property
    def rashomon_bound_target(self) -> float: ...
    @property
    def tree_count_at_target(self) -> str: ...
    @property
    def theta(self) -> float: ...
    @property
    def reg(self) -> float: ...
    @property
    def exact_cost_keys(self) -> bool: ...
    @property
    def keyed_costs(self) -> bool: ...
    @property
    def n_train(self) -> int | None: ...
    @property
    def n_prefix_leaves(self) -> int: ...
    @property
    def fit_config(self) -> dict[str, Any]: ...
    def attach_training_data(self, x: npt.ArrayLike, y: npt.ArrayLike,
                             thresholds_per_feature: Any, n_classes: int) -> None: ...
    def root_histogram(self) -> Any: ...
    def get_tree(self, idx: int) -> Any: ...
    def sample_trees(self, n: int, seed: int | None = None,
                     at_target: bool = False) -> Any: ...
    def sample_indices(self, n: int, seed: int | None = None,
                       at_target: bool = False) -> Any: ...
    def rid_histograms(self, x: Any = None, y: Any = None, at_target: bool = True,
                       features: Any = None, max_states: int = 5_000_000,
                       max_seconds: float | None = None,
                       max_mask_bytes: int | None = None) -> Any: ...
    def tree_phis(self, indices: Any, x: Any = None, y: Any = None,
                  at_target: bool = True) -> Any: ...
    def predict_probs(self, x: npt.ArrayLike) -> Any: ...
    def class1_counts(self, x: npt.ArrayLike, at_target: bool = True,
                      max_seconds: float | None = None, verify: bool = True,
                      max_states: int = 50_000_000) -> Any: ...
    def to_bytes(self, version: int | None = None) -> bytes: ...
    @staticmethod
    def from_bytes(data: bytes, feature_names: list[str] | None = None) -> PyRashomonDag: ...
    @staticmethod
    def _count_chain_fixture(levels: int, leaf_cost: float = 0.001) -> PyRashomonDag: ...
    def save(self, path: str, version: int | None = None) -> None: ...
    @staticmethod
    def load(path: str, feature_names: list[str] | None = None) -> PyRashomonDag: ...
    def set_fit_config(self, d: dict[str, Any]) -> None: ...
    def summary(self) -> Any: ...
