// Global allocator strategy:
//  - Default (no features): use the system allocator. Avoids interaction
//    with pyarrow / OpenMP-bundled allocators in downstream Python deps,
//    which we observed could segfault when both were live.
//  - `mimalloc-allocator` feature: swap to mimalloc. ~5-15% on alloc-heavy
//    paths (lickety pushes lots of small cost_entries / child_counts Vecs)
//    but only safe in environments without conflicting C-extension allocators.
//  - `alt-allocator` feature: swap to jemalloc with dirty/muzzy decay
//    tuned to 0 ms. Tends to retain less RSS on allocator-heavy BFS workloads.
#[cfg(feature = "alt-allocator")]
#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

#[cfg(all(feature = "mimalloc-allocator", not(feature = "alt-allocator")))]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

pub mod bitmask_ctx;
pub mod candidates;
pub mod dag_predict;
pub mod dag_predict_exact;
pub mod dataset_store;
pub mod factored_dag_io;
pub mod rashomon_anchor;
pub mod rashomon_dag;
pub mod rid;
pub mod scratch;
pub mod shared_bounds;
pub mod solver;
pub mod tree;
pub mod types;

/// Versioned candidate ranking semantics recorded with newly fitted models.
pub const CANDIDATE_RANKING_POLICY: &str = "weighted_gini_leaf_adjacency_v1";

#[cfg(feature = "python")]
use types::{
    DpCriterion, FeatureMask, FeatureType, MsdtPortConfig, RashomonSet, RashomonTuning,
    SolverConfig, SolverProfile, ThresholdInterner, MAX_PROFILE_DEPTH,
};

// ---------------------------------------------------------------------------
// PyO3 wrapper for the Rashomon DAG (gated behind "python" feature)
// ---------------------------------------------------------------------------

#[cfg(feature = "python")]
mod python_bindings {
    use super::*;
    use numpy::{PyReadonlyArray1, PyReadonlyArray2};
    use pyo3::prelude::*;
    use solver::{solve_parallel_root, solve_parallel_root_screen, BranchAndBoundSolver};

    /// Result-dict keys of the completion-lb DFS counters, in the layout of
    /// `types::lb_stats_merge` (entries 0..6 set as is; entries 6..10 give
    /// `LB_STATS_DERIVED_KEYS`).
    const LB_STATS_KEYS: [&str; 6] = [
        "n_lb_selects",
        "n_lb_d1_closed",
        "n_lb_d1_fallback",
        "lb_select_cache_records",
        "n_lb_d1_rowwalk",
        "n_lb_d1_memo_hits",
    ];
    /// Result-dict keys derived from `lb_stats` entries 6..10:
    /// time_lb_d1_closed_s = ns / 1e9, n_store_thresholds_max, and
    /// n_store_thresholds_mean = sum / features with >= 1 threshold.
    const LB_STATS_DERIVED_KEYS: [&str; 3] = [
        "time_lb_d1_closed_s",
        "n_store_thresholds_max",
        "n_store_thresholds_mean",
    ];
    /// Result-dict keys of `lb_stats` entries 10..16 (set as is).
    const LB_STATS_EXTRA_KEYS: [&str; 6] = [
        "n_explore_selects",
        "n_select_shared_hits",
        "n_lb_d1_swept",
        "n_lb_memo_evictions",
        "n_lb_d1_intdp",
        "n_lb_d1_intdp_fallback",
    ];
    use std::collections::HashMap;
    use std::sync::Arc;
    use tree::tree_to_pydict;

    /// Validate the class-indexed input contract before constructing native stores.
    fn validate_training_input(
        n_rows: usize,
        n_features: usize,
        labels: &[u8],
        n_classes: usize,
        reg: f64,
        max_depth: i32,
        max_k: usize,
        min_samples_leaf: usize,
        path_penalty: f64,
        time_limit: Option<f64>,
    ) -> PyResult<()> {
        let invalid = if n_rows == 0 || n_features == 0 {
            Some("X must have at least one row and feature")
        } else if n_features > MAX_FEATURES {
            Some("X must have at most 65535 features")
        } else if labels.len() != n_rows {
            Some("X and y must have the same number of rows")
        } else if !(2..=crate::types::MAX_CLASSES).contains(&n_classes) {
            Some("n_classes must be between 2 and 16")
        } else if labels.iter().any(|&label| label as usize >= n_classes) {
            Some("y must contain encoded labels in 0..n_classes")
        } else if !reg.is_finite() || reg < 0.0 {
            Some("reg must be finite and nonnegative")
        } else if max_depth < 0 || max_k < 2 || min_samples_leaf == 0 {
            Some("max_depth must be nonnegative, max_k >= 2, and min_samples_leaf >= 1")
        } else if max_k > MAX_ARITY {
            Some("max_k must be at most 255")
        } else if !path_penalty.is_finite() || path_penalty < 0.0 {
            Some("path_penalty must be finite and nonnegative")
        } else if time_limit.is_some_and(|t| !t.is_finite() || t < 0.0) {
            Some("time_limit must be finite and nonnegative")
        } else {
            None
        };
        if let Some(message) = invalid {
            return Err(pyo3::exceptions::PyValueError::new_err(message));
        }
        Ok(())
    }

    /// Split feature indices and arities are stored as u16 / u8.
    const MAX_FEATURES: usize = u16::MAX as usize;
    const MAX_ARITY: usize = u8::MAX as usize;
    /// Bin indices of one feature are u16: at most 65535 bins (65534 thresholds).
    const MAX_BINS: usize = u16::MAX as usize;

    fn value_err<T>(message: String) -> PyResult<T> {
        Err(pyo3::exceptions::PyValueError::new_err(message))
    }

    /// Feature types from their names ("num"/"numeric", "cat"/"categorical"),
    /// one per feature; every feature numeric when `None`.
    fn parse_feature_types(
        names: Option<Vec<String>>,
        n_features: usize,
    ) -> PyResult<Vec<FeatureType>> {
        let Some(names) = names else {
            return Ok(vec![FeatureType::Num; n_features]);
        };
        if names.len() != n_features {
            return value_err(format!(
                "feature_types has {} entries for {} features",
                names.len(),
                n_features
            ));
        }
        names
            .iter()
            .map(|s| match s.as_str() {
                "num" | "numeric" => Ok(FeatureType::Num),
                "cat" | "categorical" => Ok(FeatureType::Cat),
                other => value_err(format!(
                    "unknown feature type {:?}; use 'num' or 'cat'",
                    other
                )),
            })
            .collect()
    }

    fn parse_dp_criterion(name: &str) -> PyResult<DpCriterion> {
        Ok(match name {
            "gini" => DpCriterion::Gini,
            "misclass" => DpCriterion::Misclass,
            "depth2" => DpCriterion::Depth2,
            "depth2_misclass" => DpCriterion::Depth2Misclass,
            "rush" => DpCriterion::Rush,
            "twoing" => DpCriterion::Twoing,
            "entropy" => DpCriterion::Entropy,
            "hellinger" => DpCriterion::Hellinger,
            "learned" => DpCriterion::Learned,
            "wasserstein" => DpCriterion::Wasserstein,
            other => {
                return value_err(format!(
                    "unknown dp_criterion {:?}; use 'gini', 'misclass', 'depth2', \
                     'depth2_misclass', 'rush', 'twoing', 'entropy', 'hellinger', \
                     'learned' or 'wasserstein'",
                    other
                ))
            }
        })
    }

    /// Criterion of the greedy completion; "greedy" is Gini.
    fn parse_closure_criterion(name: Option<&str>) -> PyResult<Option<DpCriterion>> {
        name.map(|s| {
            Ok(match s {
                "gini" | "greedy" => DpCriterion::Gini,
                "misclass" => DpCriterion::Misclass,
                "rush" => DpCriterion::Rush,
                "entropy" => DpCriterion::Entropy,
                "hellinger" => DpCriterion::Hellinger,
                other => {
                    return value_err(format!(
                        "unknown closure criterion {:?}; use 'greedy', 'gini', \
                         'misclass', 'rush', 'entropy' or 'hellinger'",
                        other
                    ))
                }
            })
        })
        .transpose()
    }

    fn parse_atom_merge_rule(name: &str) -> PyResult<types::AtomMergeRule> {
        match name {
            "l1" => Ok(types::AtomMergeRule::L1Tolerance),
            "wilson" | "wilson_ci" | "confidence_overlap" => Ok(types::AtomMergeRule::WilsonCI),
            other => value_err(format!(
                "unknown atom_merge_rule {:?}; use 'l1' or 'wilson'",
                other
            )),
        }
    }

    /// Reject features with more than `MAX_BINS` bins: a supplied threshold
    /// list of `MAX_BINS` or more entries, or a categorical feature without
    /// supplied thresholds (one bin per category) with more than `MAX_BINS`
    /// distinct values.
    fn check_bin_counts(
        x: ndarray::ArrayView2<f64>,
        types: &[FeatureType],
        thresholds: Option<&HashMap<usize, Vec<f64>>>,
        closure_thresholds: Option<&HashMap<usize, Vec<f64>>>,
    ) -> PyResult<()> {
        for m in [thresholds, closure_thresholds].into_iter().flatten() {
            if let Some((j, t)) = m.iter().find(|(_, t)| t.len() >= MAX_BINS) {
                return value_err(format!(
                    "feature {} has {} thresholds; at most {} are supported",
                    j,
                    t.len(),
                    MAX_BINS - 1
                ));
            }
        }
        for (j, t) in types.iter().enumerate() {
            let supplied = thresholds.is_some_and(|m| m.contains_key(&j));
            if *t != FeatureType::Cat || supplied || x.nrows() <= MAX_BINS {
                continue;
            }
            let mut v: Vec<f64> = x
                .column(j)
                .iter()
                .copied()
                .filter(|v| !v.is_nan())
                .collect();
            v.sort_by(f64::total_cmp);
            v.dedup();
            if v.len() > MAX_BINS {
                return value_err(format!(
                    "categorical feature {} has {} categories; at most {} are supported",
                    j,
                    v.len(),
                    MAX_BINS
                ));
            }
        }
        Ok(())
    }

    /// `FitConfigBlob::extras` key of the fit-time target bound.
    const TARGET_BOUND_KEY: &str = "rashomon_bound_target";

    /// Largest feature index any split of `dag` uses (pooled and inline
    /// splits); None when it has no split.
    fn max_split_feature(dag: &crate::types::FactoredDag) -> Option<usize> {
        let splits = dag
            .split_pool
            .entries
            .iter()
            .chain(dag.subproblems.iter().flat_map(|sp| sp.splits.iter()));
        splits
            .map(|split| dag.spec_of(split).feature_index as usize)
            .max()
    }

    /// Tree indexing / sampling guard: a `count_saturated` DAG's counts are
    /// clamped at u128::MAX, so its index decoding is not exact.
    fn check_indexable(dag: &crate::types::FactoredDag) -> PyResult<()> {
        if dag.count_saturated {
            return Err(pyo3::exceptions::PyOverflowError::new_err(
                "Rashomon tree count saturated at u128::MAX (count_saturated): \
                 tree indexing and sampling are not exact on this DAG",
            ));
        }
        Ok(())
    }

    /// Get current RSS in MB via /proc (Linux) or `ps` (macOS).
    fn current_rss_mb() -> usize {
        // macOS: read the RSS from `ps`.
        #[cfg(target_os = "macos")]
        {
            if let Ok(output) = std::process::Command::new("ps")
                .args(["-o", "rss=", "-p", &std::process::id().to_string()])
                .output()
            {
                if let Ok(s) = String::from_utf8(output.stdout) {
                    if let Ok(kb) = s.trim().parse::<usize>() {
                        return kb / 1024;
                    }
                }
            }
            0
        }
        #[cfg(not(target_os = "macos"))]
        {
            if let Ok(status) = std::fs::read_to_string("/proc/self/status") {
                for line in status.lines() {
                    if line.starts_with("VmRSS:") {
                        let kb: usize = line
                            .split_whitespace()
                            .nth(1)
                            .and_then(|s| s.parse().ok())
                            .unwrap_or(0);
                        return kb / 1024;
                    }
                }
            }
            0
        }
    }

    #[pyclass(module = "splinter.multiway_solver")]
    pub struct PyRashomonDag {
        // Python receives a materialized set. Native callers can also hold
        // pending snapshots and materialize them on access.
        dag: RashomonSet,
        feature_names: Vec<String>,
        feature_types: Vec<FeatureType>,
        fit_config: Option<factored_dag_io::FitConfigBlob>,
        /// Target bound opt*(1+θ) in DAG cost units, set by `solve_rashomon`.
        /// None for loaded DAGs → `dag.opt * (1 + dag.theta)`.
        target_bound: Option<f64>,
        /// Training rows of the fit (the solver's store). `get_tree` /
        /// `sample_trees` route them through each extracted tree so every
        /// leaf reports its own training-row count
        /// (`rashomon_dag::route_leaf_n_samples`). None for loaded DAGs:
        /// leaves then carry the stored per-vertex value.
        train_store: Option<std::sync::Arc<dataset_store::DatasetStore>>,
        /// Training-row count recorded in v6 files (the fit store's rows, or
        /// the value read back from a v6 file). None when unknown.
        n_train: Option<u64>,
    }

    impl PyRashomonDag {
        fn target_bound_dag(&self) -> f64 {
            let d = self.dag.dag();
            self.target_bound.unwrap_or(d.opt * (1.0 + d.theta))
        }

        /// ValueError unless `x` has one column per feature of the fit
        /// (when the feature names are known) and a column for every
        /// feature a split uses.
        fn check_columns(&self, p: usize) -> PyResult<()> {
            let n_meta = self.feature_names.len();
            if n_meta != 0 && p != n_meta {
                return value_err(format!(
                    "x has {} columns, the DAG was fit on {}",
                    p, n_meta
                ));
            }
            if let Some(f) = max_split_feature(self.dag.dag()).filter(|&f| p <= f) {
                return value_err(format!(
                    "x has {} columns, the DAG splits on feature index {}",
                    p, f
                ));
            }
            Ok(())
        }

        /// Store of the evaluation rows for `rid_histograms` / `tree_phis`:
        /// the fit rows (`train_store`) when `x` and `y` are None, else a
        /// store on (x, y) with the fit store's thresholds and class count,
        /// so every split spec routes rows exactly as on the fit rows.
        fn eval_store(
            &self,
            x: Option<PyReadonlyArray2<f64>>,
            y: Option<Vec<i64>>,
        ) -> PyResult<std::sync::Arc<dataset_store::DatasetStore>> {
            let train = self.train_store.as_ref().ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(
                    "this DAG has no training store (loaded from disk?): RID evaluation \
                     needs the fit-time bins/thresholds",
                )
            })?;
            match (x, y) {
                (None, None) => Ok(train.clone()),
                (Some(x), Some(y)) => {
                    let x = x.as_array().to_owned();
                    let p = train.n_features();
                    if x.ncols() != p {
                        return Err(pyo3::exceptions::PyValueError::new_err(format!(
                            "x has {} columns, the DAG was fit on {}",
                            x.ncols(),
                            p
                        )));
                    }
                    if x.nrows() != y.len() {
                        return Err(pyo3::exceptions::PyValueError::new_err(format!(
                            "x has {} rows but y has {}",
                            x.nrows(),
                            y.len()
                        )));
                    }
                    let n_classes = train.ctx().n_classes;
                    let mut labels = Vec::with_capacity(y.len());
                    for &v in &y {
                        if v < 0 || v as usize >= n_classes {
                            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                                "label {} outside [0, {}) (the DAG's classes)",
                                v, n_classes
                            )));
                        }
                        labels.push(v as u8);
                    }
                    let thresholds: HashMap<usize, Vec<f64>> = train
                        .feature_thresholds()
                        .iter()
                        .enumerate()
                        .filter(|(_, t)| !t.is_empty())
                        .map(|(j, t)| (j, t.to_vec()))
                        .collect();
                    Ok(std::sync::Arc::new(dataset_store::DatasetStore::new(
                        x,
                        labels,
                        None,
                        n_classes,
                        if thresholds.is_empty() {
                            None
                        } else {
                            Some(&thresholds)
                        },
                        None,
                    )))
                }
                _ => Err(pyo3::exceptions::PyValueError::new_err(
                    "pass both x and y, or neither (the fit rows)",
                )),
            }
        }

        /// `factored_dag_io` bytes in `version` (None = the default, v6).
        /// v6 records `n_train`.
        fn serialize(&self, version: Option<u16>) -> PyResult<Vec<u8>> {
            let (dag, names, types, fc) = (
                self.dag.dag(),
                &self.feature_names,
                &self.feature_types,
                self.fit_config.as_ref(),
            );
            match version.unwrap_or(factored_dag_io::DEFAULT_FORMAT_VERSION) {
                6 => Ok(factored_dag_io::serialize_factored_dag_v6(
                    dag,
                    names,
                    types,
                    fc,
                    self.n_train,
                    factored_dag_io::DagCodec::Deflate,
                )),
                v => factored_dag_io::serialize_factored_dag_version(dag, names, types, fc, v)
                    .map_err(pyo3::exceptions::PyValueError::new_err),
            }
        }
    }

    #[pymethods]
    impl PyRashomonDag {
        #[getter]
        fn tree_count(&self) -> String {
            format!("{}", self.dag.dag().tree_count)
        }

        /// The tree count as a u64; raises `OverflowError` when it does not
        /// fit (use `tree_count_u128` or the decimal `tree_count`).
        #[getter]
        fn tree_count_u64(&self) -> PyResult<u64> {
            let tc = self.dag.dag().tree_count;
            u64::try_from(tc).map_err(|_| {
                pyo3::exceptions::PyOverflowError::new_err(format!(
                    "tree_count {} does not fit in u64; use tree_count_u128",
                    tc
                ))
            })
        }

        /// The tree count as a Python int (exact up to u128::MAX; see
        /// `count_saturated`).
        #[getter]
        fn tree_count_u128(&self) -> u128 {
            self.dag.dag().tree_count
        }

        /// True when a count op of the fit saturated at u128::MAX: the
        /// counts are then clamped lower bounds and tree indexing /
        /// sampling raise `OverflowError` instead of decoding indices.
        #[getter]
        fn count_saturated(&self) -> bool {
            self.dag.dag().count_saturated
        }

        #[getter]
        fn n_nodes(&self) -> usize {
            self.dag.dag().subproblems.len()
        }

        #[getter]
        fn opt(&self) -> f64 {
            self.dag.dag().opt
        }

        #[getter]
        fn rashomon_bound(&self) -> f64 {
            self.dag.dag().rashomon_bound
        }

        /// Target bound opt*(1+θ), without the phase-1 gap widening, in
        /// DAG cost units (the units of `rashomon_bound`).
        #[getter]
        fn rashomon_bound_target(&self) -> f64 {
            self.target_bound_dag()
        }

        /// Trees with cost ≤ opt*(1+θ) (decimal string, like tree_count).
        #[getter]
        fn tree_count_at_target(&self) -> String {
            let d = self.dag.dag();
            format!(
                "{}",
                rashomon_dag::tree_count_at_bound(d, self.target_bound_dag())
            )
        }

        #[getter]
        fn theta(&self) -> f64 {
            self.dag.dag().theta
        }

        #[getter]
        fn reg(&self) -> f64 {
            self.dag.dag().reg
        }

        /// True when the counts were built on exact lattice keys.
        #[getter]
        fn exact_cost_keys(&self) -> bool {
            self.dag.dag().exact_cost_keys
        }

        /// True when the counts were built in integer key space.
        #[getter]
        fn keyed_costs(&self) -> bool {
            self.dag.dag().keyed_costs
        }

        /// Training-row count of the fit (recorded in v6 files); None when
        /// unknown (v4 / v5 files, or a v6 file that recorded none).
        #[getter]
        fn n_train(&self) -> Option<usize> {
            self.n_train.map(|n| n as usize)
        }

        /// Re-attach the training rows to a loaded DAG, so `get_tree` /
        /// `sample_trees` report each leaf's training-row count (without
        /// it a loaded DAG reports the stored per-vertex count, which after
        /// merging is the count of one merged support, not necessarily the
        /// tree's own leaf) and
        /// `rid_histograms` / `tree_phis` can run on the fit rows. Builds the
        /// store exactly as `solve_rashomon` does: `x` (float64, n x p), `y`
        /// (encoded labels in 0..n_classes, as passed to `solve_rashomon`),
        /// the same `thresholds_per_feature` dict passed at fit time, and
        /// `n_classes`. Raises ValueError when the row count differs from
        /// the recorded `n_train` (when known), `x` has no column for a
        /// feature some split uses, the feature count differs from the DAG's
        /// feature metadata (when present), or a label is outside
        /// 0..n_classes.
        #[pyo3(signature = (x, y, thresholds_per_feature, n_classes))]
        fn attach_training_data(
            &mut self,
            x: PyReadonlyArray2<f64>,
            y: Vec<i64>,
            thresholds_per_feature: Option<HashMap<usize, Vec<f64>>>,
            n_classes: usize,
        ) -> PyResult<()> {
            let value_error = |m: String| Err(pyo3::exceptions::PyValueError::new_err(m));
            let x = x.as_array().to_owned();
            let (n_rows, p) = (x.nrows(), x.ncols());
            if n_rows == 0 || p == 0 {
                return value_error("x must have at least one row and feature".to_string());
            }
            if y.len() != n_rows {
                return value_error(format!("x has {} rows but y has {}", n_rows, y.len()));
            }
            if let Some(n) = self.n_train {
                if n_rows as u64 != n {
                    return value_error(format!("x has {} rows, the DAG was fit on {}", n_rows, n));
                }
            }
            let max_feature = max_split_feature(self.dag.dag());
            if let Some(f) = max_feature.filter(|&f| p <= f) {
                return value_error(format!(
                    "x has {} columns, the DAG splits on feature index {}",
                    p, f
                ));
            }
            let n_meta = self.feature_names.len();
            if n_meta != 0 && p != n_meta {
                return value_error(format!(
                    "x has {} columns, the DAG was fit on {}",
                    p, n_meta
                ));
            }
            if !(2..=crate::types::MAX_CLASSES).contains(&n_classes) {
                return value_error("n_classes must be between 2 and 16".to_string());
            }
            let mut labels = Vec::with_capacity(n_rows);
            for &v in &y {
                if v < 0 || v as usize >= n_classes {
                    return value_error(format!(
                        "label {} outside [0, {}) (n_classes)",
                        v, n_classes
                    ));
                }
                labels.push(v as u8);
            }
            if let Some(&j) = thresholds_per_feature
                .as_ref()
                .and_then(|t| t.keys().find(|&&j| j >= p))
            {
                return value_error(format!(
                    "thresholds_per_feature has feature index {} for {} features",
                    j, p
                ));
            }
            // As in `solve_rashomon` (`BranchAndBoundSolver::from_raw` or the
            // recursive-fill store): no weights, no reference predictions.
            let store = dataset_store::DatasetStore::new(
                x,
                labels,
                None,
                n_classes,
                thresholds_per_feature.as_ref(),
                None,
            );
            self.n_train = Some(store.n_samples() as u64);
            self.train_store = Some(std::sync::Arc::new(store));
            Ok(())
        }

        /// Return root cost schedule as a list of (cost_float, count) tuples.
        /// `cost_float` is normalized cost (loss + reg); multiply by
        /// n_samples to get the integer objective (as in PRAXIS, Heile et al.).
        /// Entries are sorted ascending by cost.
        fn root_histogram(&self, py: Python<'_>) -> PyResult<PyObject> {
            let entries = &self.dag.dag().root_cost_schedule;
            let list = pyo3::types::PyList::empty_bound(py);
            for e in entries.iter() {
                let tup = pyo3::types::PyTuple::new_bound(
                    py,
                    &[e.cost.into_py(py), e.tree_count().into_py(py)],
                );
                list.append(tup)?;
            }
            Ok(list.into_any().unbind())
        }

        /// Number of subproblems in the DAG that were forced to be leaves by
        /// the configured `solve_depth` (lookahead/closure truncation) and
        /// would have unused depth budget below them. Zero in the typical
        /// `lookahead_depth=None` case. Non-zero means the trees emitted by
        /// `get_tree` terminate at majority-class predictions at those
        /// positions, NOT at a Rashomon-fill of the remaining depth.
        #[getter]
        fn n_prefix_leaves(&self) -> usize {
            self.dag
                .dag()
                .subproblems
                .iter()
                .filter(|sp| sp.prefix_leaf.is_some())
                .count()
        }

        /// Tree at index `idx` in [0, tree_count) (a Python int up to
        /// u128). Raises `IndexError` out of range and `OverflowError` on a
        /// `count_saturated` DAG (its indices are not exact). Leaf
        /// `n_samples` is the tree's own routed training-row count when the
        /// training rows are attached (a fit, or `attach_training_data`),
        /// else the stored per-vertex count (see `attach_training_data`).
        fn get_tree(&self, py: Python<'_>, idx: u128) -> PyResult<PyObject> {
            let dag = self.dag.dag();
            check_indexable(dag)?;
            if idx >= dag.tree_count {
                return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                    "tree index {} out of range [0, {})",
                    idx, dag.tree_count
                )));
            }
            let mut tree = rashomon_dag::get_tree_at_index_factored(dag, idx, &self.feature_names);
            if let Some(store) = self.train_store.as_ref() {
                rashomon_dag::route_leaf_n_samples(store, &mut tree);
            }
            let dict = tree_to_pydict(py, &tree, &self.feature_names, &self.feature_types)?;
            Ok(dict.into_any().unbind())
        }

        /// Uniform draws with replacement. `seed=None` seeds from the clock;
        /// `at_target=True` draws only from the trees counted by
        /// `tree_count_at_target` (cost ≤ opt*(1+θ)).
        #[pyo3(signature = (n, seed=None, at_target=false))]
        fn sample_trees(
            &self,
            py: Python<'_>,
            n: usize,
            seed: Option<u64>,
            at_target: bool,
        ) -> PyResult<Vec<PyObject>> {
            check_indexable(self.dag.dag())?;
            let bound = if at_target {
                Some(self.target_bound_dag())
            } else {
                None
            };
            let mut trees = rashomon_dag::sample_trees_factored(
                self.dag.dag(),
                n,
                &self.feature_names,
                seed,
                bound,
            );
            if let Some(store) = self.train_store.as_ref() {
                for tree in trees.iter_mut() {
                    rashomon_dag::route_leaf_n_samples(store, tree);
                }
            }
            let mut result = Vec::with_capacity(trees.len());
            for tree in &trees {
                let dict = tree_to_pydict(py, tree, &self.feature_names, &self.feature_types)?;
                result.push(dict.into_any().unbind());
            }
            Ok(result)
        }

        /// Tree indices (for `get_tree`) of the draws `sample_trees` would
        /// make with the same arguments.
        #[pyo3(signature = (n, seed=None, at_target=false))]
        fn sample_indices(
            &self,
            n: usize,
            seed: Option<u64>,
            at_target: bool,
        ) -> PyResult<Vec<u128>> {
            check_indexable(self.dag.dag())?;
            let bound = if at_target {
                Some(self.target_bound_dag())
            } else {
                None
            };
            Ok(rashomon_dag::sample_indices_factored(
                self.dag.dag(),
                n,
                seed,
                bound,
            ))
        }

        /// Exact Rashomon Importance Distribution histograms (`rid::
        /// rid_exact_histograms`): per feature j, the numerators N_j =
        /// n²·φ_j of every tree's all-donor subtractive model reliance with
        /// their tree counts, on the fit rows (x, y None) or on (x, y).
        /// `at_target` = the trees of `tree_count_at_target`, else all
        /// `tree_count` trees. `complete` is False when `max_states` or
        /// `max_seconds` stopped the walk, or the interned row masks of one
        /// feature passed `max_mask_bytes` (None = 2 GiB) (histograms then
        /// unusable).
        #[pyo3(signature = (x=None, y=None, at_target=true, features=None, max_states=5_000_000, max_seconds=None, max_mask_bytes=None))]
        fn rid_histograms(
            &self,
            py: Python<'_>,
            x: Option<PyReadonlyArray2<f64>>,
            y: Option<Vec<i64>>,
            at_target: bool,
            features: Option<Vec<usize>>,
            max_states: usize,
            max_seconds: Option<f64>,
            max_mask_bytes: Option<usize>,
        ) -> PyResult<PyObject> {
            // A saturated DAG's clamped counts would pass the exact-RID total check
            // (MAX == MAX) while every bucket is only a lower bound: refuse, as for indexing.
            check_indexable(self.dag.dag())?;
            let store = self.eval_store(x, y)?;
            let p = store.n_features();
            if let Some(fs) = &features {
                if let Some(&bad) = fs.iter().find(|&&j| j >= p) {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "feature {} out of range (p = {})",
                        bad, p
                    )));
                }
            }
            let bound = if at_target {
                Some(self.target_bound_dag())
            } else {
                None
            };
            let dag = self.dag.dag();
            let (res, seconds) = py.allow_threads(|| {
                let t0 = std::time::Instant::now();
                let deadline = max_seconds.and_then(|s| {
                    std::time::Duration::try_from_secs_f64(s.max(0.0))
                        .ok()
                        .and_then(|d| t0.checked_add(d))
                });
                let max_mask_bytes = max_mask_bytes.unwrap_or(rid::DEFAULT_MAX_MASK_BYTES);
                let opts = rid::RidOpts {
                    bound,
                    features,
                    max_states,
                    deadline,
                    max_mask_bytes,
                };
                let res = rid::rid_exact_histograms(dag, &store, &opts);
                (res, t0.elapsed().as_secs_f64())
            });
            let d = pyo3::types::PyDict::new_bound(py);
            let nums = pyo3::types::PyList::empty_bound(py);
            let counts = pyo3::types::PyList::empty_bound(py);
            for h in &res.per_feature {
                let v: Vec<i64> = h.iter().map(|e| e.0).collect();
                nums.append(numpy::PyArray1::from_vec_bound(py, v))?;
                let c: Vec<PyObject> = h.iter().map(|e| e.1.into_py(py)).collect();
                counts.append(pyo3::types::PyList::new_bound(py, c))?;
            }
            d.set_item("nums", nums)?;
            d.set_item("counts", counts)?;
            d.set_item("n_eval", store.n_samples())?;
            d.set_item("n_trees", res.n_trees)?;
            d.set_item("complete", res.complete)?;
            d.set_item("n_states", res.n_states)?;
            d.set_item("count_mismatch", res.count_mismatch)?;
            d.set_item("n_minority_leaf_options", res.n_minority_leaf_options)?;
            d.set_item("mask_bytes_peak", res.mask_bytes_peak)?;
            d.set_item("seconds", seconds)?;
            Ok(d.into_any().unbind())
        }

        /// N_j = n²·φ_j (`rid::tree_phi_numerators`) of the trees at DAG
        /// indices `indices` (as `get_tree` / `sample_indices` index them),
        /// on the fit rows or on (x, y): int64 array (len(indices), p).
        /// Indices must lie in the `at_target` universe
        /// ([0, tree_count_at_target) or [0, tree_count)).
        #[pyo3(signature = (indices, x=None, y=None, at_target=true))]
        fn tree_phis(
            &self,
            py: Python<'_>,
            indices: Vec<u128>,
            x: Option<PyReadonlyArray2<f64>>,
            y: Option<Vec<i64>>,
            at_target: bool,
        ) -> PyResult<PyObject> {
            let store = self.eval_store(x, y)?;
            let dag = self.dag.dag();
            check_indexable(dag)?;
            let universe = if at_target {
                rashomon_dag::tree_count_at_bound(dag, self.target_bound_dag())
            } else {
                dag.tree_count
            };
            if let Some(&bad) = indices.iter().find(|&&i| i >= universe) {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "tree index {} out of range [0, {})",
                    bad, universe
                )));
            }
            let p = store.n_features();
            let names = &self.feature_names;
            let flat: Vec<i64> = py.allow_threads(|| {
                let mut out = Vec::with_capacity(indices.len() * p);
                for &idx in &indices {
                    let tree = rashomon_dag::get_tree_at_index_factored(dag, idx, names);
                    out.extend(rid::tree_phi_numerators(&tree, &store));
                }
                out
            });
            let arr = ndarray::Array2::from_shape_vec((indices.len(), p), flat)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
            Ok(numpy::PyArray2::from_owned_array_bound(py, arr)
                .into_any()
                .unbind())
        }

        /// Approximate P(class=1) for each sample over the Rashomon set.
        ///
        /// Marginalizes over ALL trees in the DAG via a memoized DP, with a
        /// cost-blind approximation: splits are weighted by total tree count,
        /// ignoring which child-cost combinations fit the bound. Exact only
        /// when predictions do not correlate with cost; for the exact uniform
        /// average use `class1_counts`.
        ///
        /// Parameters
        /// ----------
        /// x : numpy array of shape (n_samples, n_features), dtype float64, C-contiguous
        ///
        /// Returns
        /// -------
        /// numpy array of shape (n_samples,), dtype float64
        fn predict_probs(&self, py: Python<'_>, x: PyReadonlyArray2<f64>) -> PyResult<PyObject> {
            let x_array = x.as_array();
            let n_samples = x_array.nrows();
            let n_features = x_array.ncols();
            self.check_columns(n_features)?;

            // Copy to contiguous flat vec for Rust consumption
            let x_flat: Vec<f64> = x_array.iter().copied().collect();

            let probs = py.allow_threads(|| {
                dag_predict::batch_predict_probs(self.dag.dag(), &x_flat, n_samples, n_features)
            });

            // Return as Python list (numpy wrapping done on Python side)
            let list = pyo3::types::PyList::new_bound(py, &probs);
            Ok(list.into_any().unbind())
        }

        /// Exact per-row class counts over the Rashomon set
        /// (`dag_predict_exact::class_counts`): for each row of `x`, the
        /// number of trees predicting class 1 (`counts`) and class 0
        /// (`counts0`), Python ints, over the trees of `tree_count_at_target`
        /// (`at_target`, default) or of `tree_count`. `n_trees` is that set's
        /// size. Binary classes and keyed-cost DAGs only (ValueError
        /// otherwise). `complete` is False when `max_seconds` or
        /// `max_states` stopped the walk; `count_mismatch` > 0 when (with
        /// `verify`) some class total differs from the count pass. Either
        /// way the counts are then not to be used.
        #[pyo3(signature = (x, at_target=true, max_seconds=None, verify=true, max_states=50_000_000))]
        fn class1_counts(
            &self,
            py: Python<'_>,
            x: PyReadonlyArray2<f64>,
            at_target: bool,
            max_seconds: Option<f64>,
            verify: bool,
            max_states: usize,
        ) -> PyResult<PyObject> {
            check_indexable(self.dag.dag())?;
            let x_array = x.as_array();
            let n_samples = x_array.nrows();
            let n_features = x_array.ncols();
            self.check_columns(n_features)?;
            let x_flat: Vec<f64> = x_array.iter().copied().collect();
            let bound = if at_target {
                Some(self.target_bound_dag())
            } else {
                None
            };
            let dag = self.dag.dag();
            let (res, seconds) = py.allow_threads(|| {
                let t0 = std::time::Instant::now();
                let deadline = max_seconds.and_then(|s| {
                    std::time::Duration::try_from_secs_f64(s.max(0.0))
                        .ok()
                        .and_then(|d| t0.checked_add(d))
                });
                let opts = dag_predict_exact::ExactPredictOpts {
                    bound,
                    deadline,
                    verify,
                    max_states,
                };
                let res =
                    dag_predict_exact::class_counts(dag, &x_flat, n_samples, n_features, &opts);
                (res, t0.elapsed().as_secs_f64())
            });
            let res = res.map_err(pyo3::exceptions::PyValueError::new_err)?;
            let d = pyo3::types::PyDict::new_bound(py);
            let c1: Vec<PyObject> = res.counts1.iter().map(|&c| c.into_py(py)).collect();
            let c0: Vec<PyObject> = res.counts0.iter().map(|&c| c.into_py(py)).collect();
            d.set_item("counts", pyo3::types::PyList::new_bound(py, c1))?;
            d.set_item("counts0", pyo3::types::PyList::new_bound(py, c0))?;
            d.set_item("n_trees", res.n_trees)?;
            d.set_item("complete", res.complete)?;
            d.set_item("n_states", res.n_states)?;
            d.set_item("n_unique_rows", res.n_unique_rows)?;
            d.set_item("count_mismatch", res.count_mismatch)?;
            d.set_item("seconds", seconds)?;
            Ok(d.into_any().unbind())
        }

        /// Serialized DAG bytes: format v6 by default; `version=4|5|6`
        /// picks one explicitly.
        #[pyo3(signature = (version=None))]
        fn to_bytes<'py>(
            &self,
            py: Python<'py>,
            version: Option<u16>,
        ) -> PyResult<pyo3::Bound<'py, pyo3::types::PyBytes>> {
            let data = self.serialize(version)?;
            Ok(pyo3::types::PyBytes::new_bound(py, &data))
        }

        /// Deserialize a v4–v6 (`RSMY`) byte stream. If `feature_names` is provided
        /// it overrides whatever feature metadata is embedded in the file; pass
        /// `None` to use the embedded names. `feature_types` always comes from
        /// the file (or defaults to `Num` for every feature if absent).
        #[staticmethod]
        #[pyo3(signature = (data, feature_names=None))]
        fn from_bytes(data: &[u8], feature_names: Option<Vec<String>>) -> PyResult<Self> {
            let factored_dag_io::LoadedDag {
                dag,
                feature_names: embedded_names,
                feature_types: embedded_types,
                fit_config,
                n_train,
            } = factored_dag_io::deserialize_factored_dag_full(data).map_err(|e| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "DAG decode failed (formats v4-v6): {}",
                    e
                ))
            })?;
            let names = feature_names.unwrap_or(embedded_names);
            let types = if embedded_types.is_empty() {
                vec![FeatureType::Num; names.len()]
            } else {
                embedded_types
            };
            // The fit-time target bound (`set_fit_config` stores it under
            // extras); absent → `dag.opt * (1 + dag.theta)`.
            let target_bound = fit_config.as_ref().and_then(|fc| {
                fc.extras
                    .iter()
                    .find(|(k, _)| k == TARGET_BOUND_KEY)
                    .and_then(|(_, v)| v.parse::<f64>().ok())
            });
            Ok(PyRashomonDag {
                dag: RashomonSet::from_built(dag),
                feature_names: names,
                feature_types: types,
                fit_config,
                target_bound,
                train_store: None,
                n_train,
            })
        }

        /// Test fixture (u128 count path): `rashomon_dag::count_chain_fixture`,
        /// a chain DAG with c_i = 1 + c_{i-1}² trees (levels=7 > 2^64,
        /// levels=8 saturates u128). Feature names `["x0"]`.
        #[staticmethod]
        #[pyo3(signature = (levels, leaf_cost=0.001))]
        fn _count_chain_fixture(levels: usize, leaf_cost: f64) -> PyResult<Self> {
            if levels > 9 {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "levels must be <= 9",
                ));
            }
            Ok(PyRashomonDag {
                dag: RashomonSet::from_built(rashomon_dag::count_chain_fixture(levels, leaf_cost)),
                feature_names: vec!["x0".to_string()],
                feature_types: vec![FeatureType::Num],
                fit_config: None,
                target_bound: None,
                train_store: None,
                n_train: None,
            })
        }

        /// Write the DAG to `path`: format v6 by default; `version=4|5|6`
        /// picks one explicitly.
        #[pyo3(signature = (path, version=None))]
        fn save(&self, path: &str, version: Option<u16>) -> PyResult<()> {
            let data = self.serialize(version)?;
            std::fs::write(path, &data)
                .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
            Ok(())
        }

        #[staticmethod]
        #[pyo3(signature = (path, feature_names=None))]
        fn load(path: &str, feature_names: Option<Vec<String>>) -> PyResult<Self> {
            let data = std::fs::read(path)
                .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
            Self::from_bytes(&data, feature_names)
        }

        /// Attach the fit-time configuration that produced this DAG. Call once
        /// from the Python wrapper after `fit()`; the dict is serialized as
        /// part of `save()` and restored by `load()`. Numeric/bool/enum fields
        /// are pulled from named keys (`exactify_top_k`, else `top_k`);
        /// anything else goes under `extras` as JSON-encoded strings, as do
        /// the keys `dual_family`, `both_modes` and `search_strategy` (kept,
        /// not interpreted). The fit-time target bound is added to `extras`
        /// as `rashomon_bound_target`, so a loaded DAG reports the same
        /// `rashomon_bound_target` / `tree_count_at_target`.
        fn set_fit_config(
            &mut self,
            py: Python<'_>,
            d: &pyo3::Bound<'_, pyo3::types::PyDict>,
        ) -> PyResult<()> {
            let get_f64 = |k: &str, default: f64| -> PyResult<f64> {
                match d.get_item(k)? {
                    Some(v) if !v.is_none() => v.extract::<f64>(),
                    _ => Ok(default),
                }
            };
            let get_i32 = |k: &str, default: i32| -> PyResult<i32> {
                match d.get_item(k)? {
                    Some(v) if !v.is_none() => v.extract::<i32>(),
                    _ => Ok(default),
                }
            };
            let get_u32 = |k: &str, default: u32| -> PyResult<u32> {
                match d.get_item(k)? {
                    Some(v) if !v.is_none() => v.extract::<u32>(),
                    _ => Ok(default),
                }
            };
            let get_bool = |k: &str| -> PyResult<bool> {
                match d.get_item(k)? {
                    Some(v) if !v.is_none() => v.extract::<bool>(),
                    _ => Ok(false),
                }
            };
            let get_str = |k: &str, default: &str| -> PyResult<String> {
                match d.get_item(k)? {
                    Some(v) if !v.is_none() => v.extract::<String>(),
                    _ => Ok(default.to_string()),
                }
            };
            let get_opt_u32 = |k: &str| -> PyResult<Option<u32>> {
                match d.get_item(k)? {
                    Some(v) if !v.is_none() => Ok(Some(v.extract::<u32>()?)),
                    _ => Ok(None),
                }
            };

            let mut extras: Vec<(String, String)> = Vec::new();
            if let Some(ex) = d.get_item("extras")? {
                if !ex.is_none() {
                    let ex_dict: &pyo3::Bound<'_, pyo3::types::PyDict> = ex.downcast()?;
                    // Encode each value as JSON via Python's json.dumps so we
                    // don't have to build a JSON serializer in Rust.
                    let json_mod = py.import_bound("json")?;
                    let dumps = json_mod.getattr("dumps")?;
                    for (k, v) in ex_dict.iter() {
                        let key_str: String = k.extract()?;
                        let val_json: String = dumps.call1((v,))?.extract()?;
                        extras.push((key_str, val_json));
                    }
                }
            }
            for key in ["dual_family", "both_modes", "search_strategy"] {
                if let Some(v) = d.get_item(key)? {
                    let dumps = py.import_bound("json")?.getattr("dumps")?;
                    let val_json: String = dumps.call1((v,))?.extract()?;
                    extras.retain(|(k, _)| k != key);
                    extras.push((key.to_string(), val_json));
                }
            }
            extras.retain(|(k, _)| k != TARGET_BOUND_KEY);
            if let Some(tb) = self.target_bound.filter(|tb| tb.is_finite()) {
                // `{:?}` is the shortest decimal that parses back to `tb`.
                extras.push((TARGET_BOUND_KEY.to_string(), format!("{:?}", tb)));
            }

            self.fit_config = Some(factored_dag_io::FitConfigBlob {
                max_depth: get_i32("max_depth", 0)?,
                max_k: get_u32("max_k", 0)?,
                min_samples_leaf: get_u32("min_samples_leaf", 0)?,
                reg_alpha: get_f64("reg_alpha", 1.0)?,
                path_penalty: get_f64("path_penalty", 0.0)?,
                beam_width: get_u32("beam_width", 1)?,
                transition_delta: get_f64("transition_delta", 0.0)?,
                atom_merge_tolerance: get_f64("atom_merge_tolerance", 0.0)?,
                top_k: match get_opt_u32("exactify_top_k")? {
                    Some(v) => Some(v),
                    None => get_opt_u32("top_k")?,
                },
                teacher_alpha: get_f64("teacher_alpha", 1.0)?,
                feature_exclusion: get_bool("feature_exclusion")?,
                shape_mode: get_bool("shape_mode")?,
                atom_mode: get_bool("atom_mode")?,
                cleanup_mode: get_bool("cleanup_mode")?,
                dp_criterion: get_str("dp_criterion", "gini")?,
                atom_merge_rule: get_str("atom_merge_rule", "l1")?,
                extras,
            });
            Ok(())
        }

        /// Return the embedded fit-time config (or `None` when no config was
        /// set with `set_fit_config`, or the loaded file has no FIT_CONFIG
        /// section). Values under
        /// `extras` are returned as the original Python objects (decoded from
        /// the stored JSON).
        #[getter]
        fn fit_config(&self, py: Python<'_>) -> PyResult<PyObject> {
            let Some(fc) = self.fit_config.as_ref() else {
                return Ok(py.None());
            };
            let d = pyo3::types::PyDict::new_bound(py);
            d.set_item("max_depth", fc.max_depth)?;
            d.set_item("max_k", fc.max_k)?;
            d.set_item("min_samples_leaf", fc.min_samples_leaf)?;
            d.set_item("reg_alpha", fc.reg_alpha)?;
            d.set_item("path_penalty", fc.path_penalty)?;
            d.set_item("beam_width", fc.beam_width)?;
            d.set_item("transition_delta", fc.transition_delta)?;
            d.set_item("atom_merge_tolerance", fc.atom_merge_tolerance)?;
            match fc.top_k {
                Some(v) => d.set_item("top_k", v)?,
                None => d.set_item("top_k", py.None())?,
            }
            d.set_item("teacher_alpha", fc.teacher_alpha)?;
            d.set_item("feature_exclusion", fc.feature_exclusion)?;
            d.set_item("shape_mode", fc.shape_mode)?;
            d.set_item("atom_mode", fc.atom_mode)?;
            d.set_item("cleanup_mode", fc.cleanup_mode)?;
            d.set_item("dp_criterion", &fc.dp_criterion)?;
            d.set_item("atom_merge_rule", &fc.atom_merge_rule)?;

            // Decode extras' JSON values back into Python objects.
            let extras_dict = pyo3::types::PyDict::new_bound(py);
            let json_mod = py.import_bound("json")?;
            let loads = json_mod.getattr("loads")?;
            for (k, v) in &fc.extras {
                let val = loads.call1((v.as_str(),))?;
                extras_dict.set_item(k, val)?;
            }
            d.set_item("extras", extras_dict)?;
            Ok(d.into_any().unbind())
        }

        fn summary(&self, py: Python<'_>) -> PyResult<PyObject> {
            let s = rashomon_dag::dag_summary_factored(self.dag.dag());
            let d = pyo3::types::PyDict::new_bound(py);
            d.set_item("n_nodes", s.n_nodes)?;
            d.set_item("n_internal", s.n_internal)?;
            d.set_item("n_leaf_only", s.n_leaf_only)?;
            d.set_item("n_both", s.n_both)?;
            d.set_item("total_split_entries", s.total_split_entries)?;
            d.set_item("tree_count", format!("{}", s.tree_count))?;
            d.set_item("opt", s.opt)?;
            d.set_item("rashomon_bound", s.rashomon_bound)?;
            d.set_item("theta", s.theta)?;
            let delta = pyo3::types::PyDict::new_bound(py);
            delta.set_item("delta_beam", s.delta_budget.delta_beam)?;
            delta.set_item("delta_compress", s.delta_budget.delta_compress)?;
            delta.set_item("delta_quantize", s.delta_budget.delta_quantize)?;
            delta.set_item("delta_criterion", s.delta_budget.delta_criterion)?;
            delta.set_item("delta_total", s.delta_budget.delta_total)?;
            d.set_item("delta_budget", delta)?;
            Ok(d.into_any().unbind())
        }

        fn __repr__(&self) -> String {
            format!(
                "PyRashomonDag(trees={}, subproblems={}, opt={:.4}, bound={:.4})",
                self.dag.dag().tree_count,
                self.dag.dag().subproblems.len(),
                self.dag.dag().opt,
                self.dag.dag().rashomon_bound
            )
        }
    }

    /// Solve the multiway branch-and-bound optimization problem.
    ///
    /// This is the main entry point called from Python. It:
    /// 1. Converts numpy arrays to Rust ndarray
    /// 2. Runs the BFS branch-and-bound solver (with rayon parallelism)
    /// 3. Returns the optimal tree as a nested Python dict
    ///
    /// The GIL is released during the solve phase so rayon threads can run freely.
    #[pyfunction]
    #[pyo3(signature = (x, y, n_classes, reg, max_depth, max_k, min_samples_leaf, feature_exclusion, time_limit, initial_used_features, feature_names, warm_start_ub=None, parallel=false, top_n=20, thresholds_per_feature=None, dp_criterion="gini", beam_width=128, n_total_override=None, reg_alpha=1.0, local_thresh_scan=false, scan_beam=20, adaptive_beam=false, transition_delta=0.08, materialize_top_n=1, shape_mode=false, shape_sort_smoothing=0.0, path_penalty=0.0, greedy_extra_depth=0, sample_weights=None, ub_margin=None, n_workers=None, lazy_explore_delta=None, cheap_boundary_greedy=false, closure_depth=None, closure_beam=3, closure_parallel=false, closure_share_memo=false, closure_criterion=None, closure_thresholds_per_feature=None, closure_split_ub_gap=None, closure_feature_top_k=None, closure_feature_ranking=None, closure_min_n_for_multiway=None, closure_unsafe_filter_depth_threshold=0, feature_types=None, atom_mode=false, shape_boundary_tax=0.0, shape_sort_per_atom_teacher_prior=false, atom_merge_tolerance=0.0, top_k=None, top_k_auto_sqrt_candidates=false, top_k_min_beam=8, atom_merge_rule="l1", atom_wilson_z=1.959964, cleanup_mode=false, atom_block_compression_purity=0.0, teacher_logit=None, teacher_alpha=1.0, unsafe_filter_depth_threshold=0, admissibility_strict=false, phase0_greedy_prerank_depth=0, eqp_inline_bounds=true))]
    fn solve_branch_and_bound(
        py: Python<'_>,
        x: PyReadonlyArray2<f64>,
        y: PyReadonlyArray1<u8>,
        n_classes: usize,
        reg: f64,
        max_depth: i32,
        max_k: usize,
        min_samples_leaf: usize,
        feature_exclusion: bool,
        time_limit: Option<f64>,
        initial_used_features: Vec<usize>,
        feature_names: Vec<String>,
        warm_start_ub: Option<f64>,
        parallel: bool,
        top_n: usize,
        thresholds_per_feature: Option<HashMap<usize, Vec<f64>>>,
        dp_criterion: &str,
        beam_width: usize,
        n_total_override: Option<usize>,
        reg_alpha: f64,
        local_thresh_scan: bool,
        scan_beam: usize,
        adaptive_beam: bool,
        transition_delta: f64,
        materialize_top_n: usize,
        shape_mode: bool,
        shape_sort_smoothing: f64,
        path_penalty: f64,
        greedy_extra_depth: i32,
        sample_weights: Option<PyReadonlyArray1<f64>>,
        ub_margin: Option<f64>,
        n_workers: Option<usize>,
        lazy_explore_delta: Option<f64>,
        cheap_boundary_greedy: bool,
        closure_depth: Option<i32>,
        closure_beam: usize,
        closure_parallel: bool,
        closure_share_memo: bool,
        closure_criterion: Option<&str>,
        closure_thresholds_per_feature: Option<HashMap<usize, Vec<f64>>>,
        closure_split_ub_gap: Option<f64>,
        closure_feature_top_k: Option<usize>,
        closure_feature_ranking: Option<Vec<usize>>,
        closure_min_n_for_multiway: Option<usize>,
        closure_unsafe_filter_depth_threshold: i32,
        feature_types: Option<Vec<String>>,
        atom_mode: bool,
        shape_boundary_tax: f64,
        shape_sort_per_atom_teacher_prior: bool,
        atom_merge_tolerance: f64,
        top_k: Option<usize>,
        top_k_auto_sqrt_candidates: bool,
        top_k_min_beam: usize,
        atom_merge_rule: &str,
        atom_wilson_z: f64,
        cleanup_mode: bool,
        atom_block_compression_purity: f64,
        teacher_logit: Option<PyReadonlyArray2<f64>>,
        teacher_alpha: f64,
        unsafe_filter_depth_threshold: i32,
        admissibility_strict: bool,
        phase0_greedy_prerank_depth: i32,
        eqp_inline_bounds: bool,
    ) -> PyResult<PyObject> {
        let x_array = x.as_array().to_owned();
        let y_vec = y.as_slice()?.to_vec();
        let n_features = x_array.ncols();
        let n_data = y_vec.len();
        validate_training_input(
            x_array.nrows(),
            n_features,
            &y_vec,
            n_classes,
            reg,
            max_depth,
            max_k,
            min_samples_leaf,
            path_penalty,
            time_limit,
        )?;
        if feature_names.len() != n_features {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "feature_names must contain one name per feature",
            ));
        }

        let atom_merge_rule_enum = parse_atom_merge_rule(atom_merge_rule)?;

        // Validate + clamp teacher_alpha. Non-finite is a caller bug.
        if !teacher_alpha.is_finite() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "teacher_alpha must be a finite number in [0.0, 1.0]",
            ));
        }
        let teacher_alpha = teacher_alpha.clamp(0.0, 1.0);

        let weights_vec: Option<Vec<f64>> = sample_weights.map(|w| w.as_array().to_vec());
        if let Some(w) = weights_vec.as_ref().filter(|w| w.len() != n_data) {
            return value_err(format!(
                "sample_weight has {} entries for {} rows",
                w.len(),
                n_data
            ));
        }
        // weight_total for loss normalization: always n_total (or n_data).
        // Integer scaling happens in binning via weight_scale = n_data.
        let weight_total = n_total_override.unwrap_or(n_data) as f64;

        // Reference-ensemble logit -> teacher_prob (softmax-per-row). Expected shape
        // `(n_data, n_classes)`. Binary callers should pass a 2-column array
        // produced by `np.stack([-raw, raw], axis=1)` (the Python wrapper in
        // `contiguous.py` does this conversion). We softmax once here
        // so shape candidate generation can cheaply index class-0 probability per sample.
        let teacher_prob_arc: Option<std::sync::Arc<[f64]>> = match teacher_logit {
            Some(pyarr) => {
                let arr = pyarr.as_array();
                let (nrows, ncols) = (arr.nrows(), arr.ncols());
                if nrows != n_data {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "teacher_logit has {} rows but expected n_samples={}",
                        nrows, n_data
                    )));
                }
                if ncols != n_classes {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "teacher_logit has {} columns but expected n_classes={}",
                        ncols, n_classes
                    )));
                }
                let mut out: Vec<f64> = Vec::with_capacity(nrows * ncols);
                for r in 0..nrows {
                    // Stable softmax: subtract row max before exp.
                    let mut row_max = f64::NEG_INFINITY;
                    for c in 0..ncols {
                        let v = arr[[r, c]];
                        if v > row_max {
                            row_max = v;
                        }
                    }
                    if !row_max.is_finite() {
                        // All −∞ or NaN → uniform prob.
                        for _ in 0..ncols {
                            out.push(1.0 / ncols as f64);
                        }
                        continue;
                    }
                    let mut denom = 0.0f64;
                    let start = out.len();
                    for c in 0..ncols {
                        let e = (arr[[r, c]] - row_max).exp();
                        out.push(e);
                        denom += e;
                    }
                    if denom > 0.0 {
                        for v in &mut out[start..] {
                            *v /= denom;
                        }
                    } else {
                        for v in &mut out[start..] {
                            *v = 1.0 / ncols as f64;
                        }
                    }
                }
                Some(std::sync::Arc::from(out.into_boxed_slice()))
            }
            None => None,
        };

        // Binary convenience slice: extract class-0 probability per sample.
        // The per-node binning loop in `candidates.rs` is binary-specialized
        // and consumes this slice directly.
        let teacher_prob_c0_arc: Option<std::sync::Arc<[f64]>> = teacher_prob_arc
            .as_ref()
            .filter(|_| n_classes == 2)
            .map(|flat| {
                let mut v: Vec<f64> = Vec::with_capacity(n_data);
                for i in 0..n_data {
                    v.push(flat[i * n_classes]);
                }
                std::sync::Arc::from(v.into_boxed_slice())
            });

        let dp_crit = parse_dp_criterion(dp_criterion)?;

        let cc_criterion = parse_closure_criterion(closure_criterion)?;

        let feat_types = parse_feature_types(feature_types, n_features)?;
        check_bin_counts(
            x_array.view(),
            &feat_types,
            thresholds_per_feature.as_ref(),
            closure_thresholds_per_feature.as_ref(),
        )?;

        let config = SolverConfig {
            n_classes,
            n_features,
            reg,
            max_depth,
            max_k,
            min_samples_leaf,
            feature_exclusion,
            time_limit,
            initial_used_features: FeatureMask::all(n_features)
                .without_indices(&initial_used_features),
            feature_names: feature_names.clone(),
            feature_types: feat_types.clone(),
            warm_start_ub,
            n_total_override,
            thresholds_per_feature,
            closure_thresholds_per_feature,
            closure_split_ub_gap: closure_split_ub_gap.unwrap_or(f64::INFINITY),
            closure_feature_top_k,
            closure_feature_ranking: closure_feature_ranking.map(Arc::new),
            closure_min_n_for_multiway,
            closure_unsafe_filter_depth_threshold,
            dp_criterion: dp_crit,
            beam_width: beam_width.max(1),
            reg_alpha: reg_alpha.clamp(0.0, 1.0),
            local_thresh_scan,
            scan_beam: scan_beam.max(1),
            adaptive_beam,
            bypass_beam: false, // single-tree solver: beam always applies

            transition_delta: transition_delta.max(0.0),
            materialize_top_n: materialize_top_n.max(1),
            allow_heuristic_bound_transfer: false,
            shape_mode,
            shape_sort_smoothing,
            path_penalty: path_penalty.max(0.0),
            greedy_extra_depth: greedy_extra_depth.max(0),
            rashomon_mode: false,
            rashomon_bound: f64::INFINITY,
            step_b_remaining: 0,
            weight_total,
            ub_margin: ub_margin.unwrap_or(f64::INFINITY),
            n_workers: n_workers.unwrap_or(1).max(1),
            sequential_candidate_gen: false,
            lazy_explore_delta: lazy_explore_delta.unwrap_or(f64::INFINITY),
            cheap_boundary_greedy,
            closure_depth,
            closure_beam: closure_beam.max(1),
            closure_parallel,
            closure_share_memo,
            closure_shared_memo: None,
            closure_criterion: cc_criterion,
            phase0_greedy_prerank_depth,
            eqp_inline_bounds,
            threshold_interner: Arc::new(ThresholdInterner::new()),
            rash_tuning: RashomonTuning::default(),
            msdt: MsdtPortConfig {
                shape_boundary_tax,
                shape_sort_per_atom_teacher_prior,
                atom_mode,
                atom_merge_rule: atom_merge_rule_enum,
                atom_merge_tolerance: atom_merge_tolerance.max(0.0),
                atom_wilson_z: if atom_wilson_z > 0.0 {
                    atom_wilson_z
                } else {
                    1.959_964
                },
                top_k,
                top_k_auto_sqrt_candidates,
                top_k_min_beam: top_k_min_beam.max(1),
                cleanup_mode,
                atom_block_compression_purity: atom_block_compression_purity.max(0.0).min(1.0),
                teacher_alpha,
                ..MsdtPortConfig::default()
            },
            teacher_prob: teacher_prob_arc.clone(),
            teacher_prob_c0: teacher_prob_c0_arc.clone(),
            reference_preds: None,
            unsafe_filter_depth_threshold,
            admissibility_strict,
        };

        config
            .validate_admissibility()
            .map_err(pyo3::exceptions::PyValueError::new_err)?;

        if parallel {
            // Parallel root: evaluate top-N root candidates, solve children in parallel
            let (result, tree, _root_cands, aggregated_profile) = py
                .allow_threads(|| solve_parallel_root(x_array, y_vec, config, top_n, weights_vec));

            let tree_dict = tree_to_pydict(py, &tree, &feature_names, &feat_types)?;

            let result_dict = pyo3::types::PyDict::new_bound(py);
            result_dict.set_item("candidate_ranking_policy", CANDIDATE_RANKING_POLICY)?;
            result_dict.set_item("tree", tree_dict)?;
            result_dict.set_item("objective", result.objective)?;
            result_dict.set_item("lowerbound", result.lowerbound)?;
            result_dict.set_item("is_optimal", result.is_optimal)?;
            result_dict.set_item("certified_optimal", result.certified_optimal)?;
            result_dict.set_item("search_converged", result.search_converged)?;
            result_dict.set_item("certification_mode", result.certification_mode)?;
            result_dict.set_item("certification_reason", result.certification_reason)?;
            result_dict.set_item("unsafe_heuristics_used", result.unsafe_heuristics_used)?;
            result_dict.set_item("timed_out", result.timed_out)?;
            result_dict.set_item("termination_reason", result.termination_reason)?;
            result_dict.set_item("final_gap", result.final_gap)?;
            result_dict.set_item("n_subproblems", result.n_subproblems)?;
            result_dict.set_item("n_explored", result.n_explored)?;
            result_dict.set_item("elapsed", result.elapsed)?;

            let profile_dict = serialize_profile(py, &aggregated_profile)?;
            result_dict.set_item("profile", profile_dict)?;

            Ok(result_dict.into_any().unbind())
        } else {
            // Standard sequential B&B
            let mut solver =
                BranchAndBoundSolver::from_raw(x_array, y_vec, config, None, weights_vec);

            let result = py.allow_threads(|| solver.solve());

            let tree = solver.extract_tree();
            let tree_dict = tree_to_pydict(py, &tree, &feature_names, &feat_types)?;

            let result_dict = pyo3::types::PyDict::new_bound(py);
            result_dict.set_item("candidate_ranking_policy", CANDIDATE_RANKING_POLICY)?;
            result_dict.set_item("tree", tree_dict)?;
            result_dict.set_item("objective", result.objective)?;
            result_dict.set_item("lowerbound", result.lowerbound)?;
            result_dict.set_item("is_optimal", result.is_optimal)?;
            result_dict.set_item("certified_optimal", result.certified_optimal)?;
            result_dict.set_item("search_converged", result.search_converged)?;
            result_dict.set_item("certification_mode", result.certification_mode)?;
            result_dict.set_item("certification_reason", result.certification_reason)?;
            result_dict.set_item("unsafe_heuristics_used", result.unsafe_heuristics_used)?;
            result_dict.set_item("timed_out", result.timed_out)?;
            result_dict.set_item("termination_reason", result.termination_reason)?;
            result_dict.set_item("final_gap", result.final_gap)?;
            result_dict.set_item("n_subproblems", result.n_subproblems)?;
            result_dict.set_item("n_explored", result.n_explored)?;
            result_dict.set_item("elapsed", result.elapsed)?;

            // Serialize profile
            let profile_dict = serialize_profile(py, &solver.worker.profile)?;
            result_dict.set_item("profile", profile_dict)?;

            Ok(result_dict.into_any().unbind())
        }
    }

    fn serialize_profile(py: Python<'_>, p: &SolverProfile) -> PyResult<PyObject> {
        let d = pyo3::types::PyDict::new_bound(py);

        // Tier 1: Candidate funnel
        d.set_item("n_candidates_generated", p.n_candidates_generated)?;
        d.set_item(
            "n_candidates_survived_phase_a",
            p.n_candidates_survived_phase_a,
        )?;
        d.set_item("n_candidates_materialized", p.n_candidates_materialized)?;
        d.set_item("n_features_available", p.n_features_available)?;
        d.set_item("n_features_filtered_tier1", p.n_features_filtered_tier1)?;

        // Tier 1: Cache hit rates
        d.set_item("n_similar_lb_hits", p.n_similar_lb_hits)?;
        d.set_item("n_similar_lb_misses", p.n_similar_lb_misses)?;
        d.set_item("n_similar_ub_hits", p.n_similar_ub_hits)?;
        d.set_item("n_similar_ub_misses", p.n_similar_ub_misses)?;
        d.set_item("n_inline_bounds_calls", p.n_inline_bounds_calls)?;

        // Tier 1: Heuristics
        d.set_item("n_killer_move_hits", p.n_killer_move_hits)?;
        d.set_item("n_killer_move_misses", p.n_killer_move_misses)?;

        // Tier 1: Search quality
        d.set_item("n_nodes_pruned", p.n_nodes_pruned)?;
        d.set_item("n_nodes_purified", p.n_nodes_purified)?;
        d.set_item("n_nodes_forced_leaf", p.n_nodes_forced_leaf)?;
        d.set_item(
            "n_phase_a_early_terminations",
            p.n_phase_a_early_terminations,
        )?;
        d.set_item("n_propagate_up_calls", p.n_propagate_up_calls)?;
        d.set_item("n_propagate_up_changes", p.n_propagate_up_changes)?;
        d.set_item("n_salvage_rounds", p.n_salvage_rounds)?;
        d.set_item("n_salvage_reenqueued", p.n_salvage_reenqueued)?;
        d.set_item(
            "n_branch_bound_candidates_scope_pruned",
            p.n_branch_bound_candidates_scope_pruned,
        )?;
        d.set_item("n_candidates_beam_truncated", p.n_candidates_beam_truncated)?;
        d.set_item("n_teacher_dp_active_nodes", p.n_teacher_dp_active_nodes)?;
        d.set_item("n_greedy_boundary_nodes", p.n_greedy_boundary_nodes)?;
        d.set_item("n_greedy_score_calls", p.n_greedy_score_calls)?;
        d.set_item("n_greedy_cache_hits", p.n_greedy_cache_hits)?;
        d.set_item("n_closure_nodes", p.n_closure_nodes)?;
        d.set_item("n_closure_fast_leaf", p.n_closure_fast_leaf)?;
        d.set_item("n_closure_shared_hits", p.n_closure_shared_hits)?;
        d.set_item("n_closure_lb_skipped", p.n_closure_lb_skipped)?;
        d.set_item("n_closure_lb_cache_hits", p.n_closure_lb_cache_hits)?;
        d.set_item(
            "n_closure_terminal_shortcircuit",
            p.n_closure_terminal_shortcircuit,
        )?;
        d.set_item(
            "n_closure_alpha_beta_aborted",
            p.n_closure_alpha_beta_aborted,
        )?;
        d.set_item("n_closure_b1_filtered", p.n_closure_b1_filtered)?;
        d.set_item(
            "time_closure_create_sp",
            p.time_closure_create_sp.as_secs_f64(),
        )?;
        d.set_item(
            "time_closure_cand_gen",
            p.time_closure_cand_gen.as_secs_f64(),
        )?;
        d.set_item(
            "time_closure_shortlist",
            p.time_closure_shortlist.as_secs_f64(),
        )?;
        d.set_item(
            "time_closure_sibling_loop",
            p.time_closure_sibling_loop.as_secs_f64(),
        )?;

        // Tier 2: Phase timing
        d.set_item("time_candidate_gen", p.time_candidate_gen.as_secs_f64())?;
        d.set_item("time_phase_a", p.time_phase_a.as_secs_f64())?;
        d.set_item("time_phase_b", p.time_phase_b.as_secs_f64())?;
        d.set_item("time_propagate", p.time_propagate.as_secs_f64())?;
        d.set_item("time_killer_eval", p.time_killer_eval.as_secs_f64())?;
        d.set_item(
            "time_partition_precompute",
            p.time_partition_precompute.as_secs_f64(),
        )?;
        d.set_item("time_greedy_boundary", p.time_greedy_boundary.as_secs_f64())?;
        d.set_item("time_closure", p.time_closure.as_secs_f64())?;

        // Tier 3: Per-depth stats
        let depth_list = pyo3::types::PyList::empty_bound(py);
        for di in 0..MAX_PROFILE_DEPTH {
            if p.nodes_at_depth[di] == 0 {
                continue;
            }
            let ds = pyo3::types::PyDict::new_bound(py);
            let n = p.nodes_at_depth[di];
            ds.set_item("depth", di)?;
            ds.set_item("n_explored", n)?;
            ds.set_item("n_solved", p.n_solved_at_depth[di])?;
            ds.set_item("avg_gap", p.sum_gap_at_depth[di] / n as f64)?;
            let nm = p.n_materialized_at_depth[di];
            ds.set_item(
                "avg_inline_lb_gap",
                if nm > 0 {
                    p.sum_inline_lb_gap[di] / nm as f64
                } else {
                    0.0
                },
            )?;
            ds.set_item("time_candgen", p.time_candgen_per_depth[di].as_secs_f64())?;
            ds.set_item("time_phase_a", p.time_phase_a_per_depth[di].as_secs_f64())?;
            ds.set_item("time_phase_b", p.time_phase_b_per_depth[di].as_secs_f64())?;
            depth_list.append(ds)?;
        }
        d.set_item("depth_stats", depth_list)?;

        // Tier 5: Diagnostic — depth-1 explosion analysis
        d.set_item("n_reexplorations", p.n_reexplorations)?;
        d.set_item("n_depth1_from_rank0", p.n_depth1_from_rank0)?;
        d.set_item("n_depth1_from_rank_n", p.n_depth1_from_rank_n)?;
        d.set_item("n_depth1_lazy_skipped", p.n_depth1_lazy_skipped)?;

        // Tier 6: Beam rejection analysis
        let beam_list = pyo3::types::PyList::empty_bound(py);
        for ki in 0..8usize {
            if p.beam_gap_count_per_arity[ki] == 0
                && p.beam_saturated_cells_total_per_arity[ki] == 0
            {
                continue;
            }
            let bs = pyo3::types::PyDict::new_bound(py);
            bs.set_item("k", ki + 2)?;
            bs.set_item("max_gap", p.beam_gap_max_per_arity[ki])?;
            bs.set_item(
                "avg_gap",
                if p.beam_gap_count_per_arity[ki] > 0 {
                    p.beam_gap_sum_per_arity[ki] / p.beam_gap_count_per_arity[ki] as f64
                } else {
                    0.0
                },
            )?;
            bs.set_item("n_nodes_with_rejections", p.beam_gap_count_per_arity[ki])?;
            bs.set_item(
                "total_saturated_cells",
                p.beam_saturated_cells_total_per_arity[ki],
            )?;
            beam_list.append(bs)?;
        }
        d.set_item("beam_rejection_stats", beam_list)?;

        // Tier 4: Memory
        d.set_item("memo_size", p.memo_size)?;
        d.set_item("similar_lb_size", p.similar_lb_size)?;
        d.set_item("similar_ub_size", p.similar_ub_size)?;
        d.set_item("parent_map_size", p.parent_map_size)?;
        d.set_item("used_features_size", p.used_features_size)?;
        d.set_item("peak_queue_size", p.peak_queue_size)?;

        Ok(d.into_any().unbind())
    }

    /// Inputs of the Phase-2 (Rashomon enumeration) solver config. Shared by
    /// `solve_rashomon` and `rashomon_select_debug` through
    /// `phase2_solver_config`, so the oracle's Select cannot drift from the
    /// search's.
    struct Phase2ConfigArgs {
        n_classes: usize,
        n_features: usize,
        reg: f64,
        solve_depth: i32,
        max_k: usize,
        min_samples_leaf: usize,
        feature_exclusion: bool,
        time_limit: Option<f64>,
        feature_names: Vec<String>,
        feature_types: Vec<FeatureType>,
        warm_start_ub: Option<f64>,
        thresholds_per_feature: Option<HashMap<usize, Vec<f64>>>,
        beam_width: usize,
        reg_alpha: f64,
        bypass_beam: bool,

        transition_delta: f64,
        shape_mode: bool,
        shape_sort_smoothing: f64,
        path_penalty: f64,
        rashomon_bound: f64,
        step_b_remaining: i32,
        n_data: usize,
        closure_depth: Option<i32>,
        closure_beam: usize,
        closure_parallel: bool,
        closure_share_memo: bool,
        closure_criterion: Option<DpCriterion>,
        threshold_interner: Arc<ThresholdInterner>,
        rash_tuning: RashomonTuning,
        msdt: MsdtPortConfig,
        teacher_prob: Option<Arc<[f64]>>,
        teacher_prob_c0: Option<Arc<[f64]>>,
        reference_preds: Option<Arc<[u8]>>,
        unsafe_filter_depth_threshold: i32,
        admissibility_strict: bool,
    }

    /// Phase-2 solver config of `solve_rashomon` (rashomon_mode = true).
    fn phase2_solver_config(a: Phase2ConfigArgs) -> SolverConfig {
        SolverConfig {
            n_classes: a.n_classes,
            n_features: a.n_features,
            reg: a.reg,
            max_depth: a.solve_depth,
            max_k: a.max_k,
            min_samples_leaf: a.min_samples_leaf,
            feature_exclusion: a.feature_exclusion,
            time_limit: a.time_limit,
            initial_used_features: FeatureMask::all(a.n_features),
            feature_names: a.feature_names,
            feature_types: a.feature_types,
            warm_start_ub: a.warm_start_ub,
            n_total_override: None,
            thresholds_per_feature: a.thresholds_per_feature,
            closure_thresholds_per_feature: None,
            closure_split_ub_gap: f64::INFINITY,
            closure_feature_top_k: None,
            closure_feature_ranking: None,
            closure_min_n_for_multiway: None,
            closure_unsafe_filter_depth_threshold: 0,
            dp_criterion: DpCriterion::Rush, // O(L×k) misclass — same as Misclass but faster
            beam_width: a.beam_width.max(1),
            reg_alpha: a.reg_alpha.clamp(0.0, 1.0),
            local_thresh_scan: false,
            scan_beam: 1,
            adaptive_beam: false,
            // Phase 2 is the Rashomon enumeration pass — `bypass_beam` is the
            // user-facing knob that controls whether beam_width is enforced
            // here. Default false = legacy beam-applied behavior. True =
            // emit every (feature, threshold) the per-feature DP would
            // consider, at the cost of ~15× more memo entries on dense data.
            bypass_beam: a.bypass_beam,

            transition_delta: a.transition_delta.max(0.0),
            // Cap on materialised survivors per node for the
            // branch-and-bound search (the Step-A pass materialises every
            // survivor). 100 per node captures the
            // top-100 near-optimal splits; with ~1000 nodes that's ~100K
            // subproblems, manageable.
            materialize_top_n: 100,
            allow_heuristic_bound_transfer: false,
            shape_mode: a.shape_mode,
            shape_sort_smoothing: a.shape_sort_smoothing,
            path_penalty: a.path_penalty.max(0.0),
            greedy_extra_depth: 0,
            rashomon_mode: true,
            rashomon_bound: a.rashomon_bound,
            step_b_remaining: a.step_b_remaining,
            weight_total: a.n_data as f64,
            ub_margin: f64::INFINITY,
            n_workers: 1,
            sequential_candidate_gen: false,
            lazy_explore_delta: f64::INFINITY,
            cheap_boundary_greedy: false,
            closure_depth: a.closure_depth,
            closure_beam: a.closure_beam.max(1),
            closure_parallel: a.closure_parallel,
            closure_share_memo: a.closure_share_memo,
            closure_shared_memo: None,
            closure_criterion: a.closure_criterion,
            phase0_greedy_prerank_depth: 0,
            eqp_inline_bounds: true,
            threshold_interner: a.threshold_interner,
            rash_tuning: a.rash_tuning,
            msdt: a.msdt,
            teacher_prob: a.teacher_prob,
            teacher_prob_c0: a.teacher_prob_c0,
            reference_preds: a.reference_preds,
            unsafe_filter_depth_threshold: a.unsafe_filter_depth_threshold,
            admissibility_strict: a.admissibility_strict,
        }
    }

    /// Debug oracle binding: Select_B(u) of the Step A Phase-2 pass at
    /// the support `support_idx` (row indices into `x`), i.e. the post-dedup
    /// candidate list `rash_explore_node` generates there with ub_hint =
    /// `rashomon_bound`. Built from `phase2_solver_config`, so any kwarg left
    /// at its default matches `solve_rashomon`'s default. Returns a list of
    /// dicts {feature, k, thresholds, group_map, child_n, child_counts,
    /// split_ub}.
    ///
    /// Valid as an oracle only for solves with `feature_exclusion=false`
    /// (the children inherit the full feature mask), closure off
    /// (`closure_depth=None`; closure-owned nodes are not explored) and the
    /// default reference-ensemble settings (`shape_boundary_tax=0`, `teacher_alpha=1`,
    /// `top_k=None`, no `teacher_logit`): those are
    /// hard-wired here and change the candidate list when set.
    #[pyfunction]
    #[pyo3(signature = (x, y, n_classes, reg, max_depth, max_k, rashomon_bound, support_idx, thresholds_per_feature=None, beam_width=4, reg_alpha=1.0, transition_delta=0.02, atom_mode=false, atom_merge_tolerance=0.0, bypass_beam=false, shape_mode=false, shape_sort_smoothing=0.0, path_penalty=0.0, min_samples_leaf=1, feature_types=None))]
    fn rashomon_select_debug(
        py: Python<'_>,
        x: PyReadonlyArray2<f64>,
        y: PyReadonlyArray1<u8>,
        n_classes: usize,
        reg: f64,
        max_depth: i32,
        max_k: usize,
        rashomon_bound: f64,
        support_idx: Vec<u32>,
        thresholds_per_feature: Option<HashMap<usize, Vec<f64>>>,
        beam_width: usize,
        reg_alpha: f64,
        transition_delta: f64,
        atom_mode: bool,
        atom_merge_tolerance: f64,
        bypass_beam: bool,
        shape_mode: bool,
        shape_sort_smoothing: f64,
        path_penalty: f64,
        min_samples_leaf: usize,
        feature_types: Option<Vec<String>>,
    ) -> PyResult<PyObject> {
        let x_array = x.as_array().to_owned();
        let y_vec = y.as_slice()?.to_vec();
        let n_features = x_array.ncols();
        let n_data = y_vec.len();
        validate_training_input(
            x_array.nrows(),
            n_features,
            &y_vec,
            n_classes,
            reg,
            max_depth,
            max_k,
            min_samples_leaf,
            path_penalty,
            None,
        )?;
        if let Some(&bad) = support_idx.iter().find(|&&i| i as usize >= n_data) {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "support_idx {} out of range (n_samples={})",
                bad, n_data
            )));
        }
        let feat_types = parse_feature_types(feature_types, n_features)?;
        check_bin_counts(
            x_array.view(),
            &feat_types,
            thresholds_per_feature.as_ref(),
            None,
        )?;
        let config = phase2_solver_config(Phase2ConfigArgs {
            n_classes,
            n_features,
            reg,
            solve_depth: max_depth,
            max_k,
            min_samples_leaf,
            feature_exclusion: false,
            time_limit: None,
            feature_names: (0..n_features).map(|i| format!("f{}", i)).collect(),
            feature_types: feat_types,
            warm_start_ub: None,
            thresholds_per_feature,
            beam_width,
            reg_alpha,
            bypass_beam,

            transition_delta,
            shape_mode,
            shape_sort_smoothing,
            path_penalty,
            rashomon_bound,
            step_b_remaining: 0,
            n_data,
            closure_depth: None,
            closure_beam: 1,
            closure_parallel: false,
            closure_share_memo: false,
            closure_criterion: None,
            threshold_interner: Arc::new(ThresholdInterner::new()),
            rash_tuning: RashomonTuning::default(),
            msdt: MsdtPortConfig {
                shape_boundary_tax: 0.0,
                atom_mode,
                teacher_alpha: 1.0,
                top_k: None,
                atom_merge_tolerance: atom_merge_tolerance.max(0.0),
                ..MsdtPortConfig::default()
            },
            teacher_prob: None,
            teacher_prob_c0: None,
            reference_preds: None,
            unsafe_filter_depth_threshold: 0,
            admissibility_strict: false,
        });
        let cands = py.allow_threads(|| {
            let mut solver =
                solver::BranchAndBoundSolver::from_raw(x_array, y_vec, config, None, None);
            let support = solver.search.store.support_from_indices(&support_idx);
            solver.rashomon_select(&support)
        });
        let out = pyo3::types::PyList::empty_bound(py);
        for c in &cands {
            let d = pyo3::types::PyDict::new_bound(py);
            d.set_item("feature", c.feature_index)?;
            d.set_item("k", c.k)?;
            d.set_item("thresholds", c.thresholds.to_vec())?;
            d.set_item("group_map", c.group_map.clone())?;
            d.set_item("child_n", c.child_n.clone())?;
            d.set_item("child_counts", c.child_counts.clone())?;
            d.set_item("split_ub", c.split_ub)?;
            out.append(d)?;
        }
        Ok(out.into_any().unbind())
    }

    /// Two-phase Rashomon set solver.
    ///
    /// Phase 1: Find the optimal objective (fast, aggressive pruning).
    /// Phase 2: Re-explore with rashomon_bound = opt*(1+theta) to collect
    ///          all near-optimal splits, then build a compressed DAG.
    ///
    /// Returns a dict with the DAG summary, tree count, and delta budget.
    #[pyfunction]
    #[pyo3(signature = (x, y, n_classes, reg, max_depth, max_k, theta, lookahead_depth=None, min_samples_leaf=1, feature_exclusion=false, time_limit=None, feature_names=None, beam_width=4, reg_alpha=1.0, transition_delta=0.02, thresholds_per_feature=None, cost_precision=1e9, path_penalty=0.0, closure_depth=None, closure_beam=1, closure_parallel=false, closure_share_memo=false, closure_criterion=None, feature_types=None, shape_mode=false, shape_sort_smoothing=0.0, atom_mode=false, shape_boundary_tax=0.0, teacher_logit=None, teacher_alpha=1.0, unsafe_filter_depth_threshold=0, admissibility_strict=false, top_k=None, atom_merge_tolerance=0.0, recursive_fill=false, leaf_reg_cap=0.1, leaf_short_circuit_eps=1e-4, fill_parallel=true, verbose=false, bypass_beam=false, lb_threads=1, stepb_batches=0, stepb_batch_lb_threads=None, lb_interval=false, lb_interval_mink=3, lb_memo_max_entries=3_000_000, lb_d1s_max_entries=16_000_000, lb_interval_cache_max=100_000, phase1_screen_cap_s=60.0))]
    fn solve_rashomon(
        py: Python<'_>,
        x: PyReadonlyArray2<f64>,
        y: PyReadonlyArray1<u8>,
        n_classes: usize,
        reg: f64,
        max_depth: i32,
        max_k: usize,
        theta: f64,
        lookahead_depth: Option<i32>,
        min_samples_leaf: usize,
        feature_exclusion: bool,
        time_limit: Option<f64>,
        feature_names: Option<Vec<String>>,
        beam_width: usize,
        reg_alpha: f64,
        transition_delta: f64,
        thresholds_per_feature: Option<HashMap<usize, Vec<f64>>>,
        cost_precision: f64,
        path_penalty: f64,
        closure_depth: Option<i32>,
        closure_beam: usize,
        closure_parallel: bool,
        closure_share_memo: bool,
        closure_criterion: Option<&str>,
        feature_types: Option<Vec<String>>,
        shape_mode: bool,
        shape_sort_smoothing: f64,
        atom_mode: bool,
        shape_boundary_tax: f64,
        teacher_logit: Option<PyReadonlyArray2<f64>>,
        teacher_alpha: f64,
        unsafe_filter_depth_threshold: i32,
        admissibility_strict: bool,
        top_k: Option<usize>,
        atom_merge_tolerance: f64,
        recursive_fill: bool,
        leaf_reg_cap: f64,
        leaf_short_circuit_eps: f64,
        fill_parallel: bool,
        verbose: bool,
        bypass_beam: bool,
        lb_threads: usize,
        stepb_batches: usize,
        stepb_batch_lb_threads: Option<usize>,
        lb_interval: bool,
        lb_interval_mink: usize,
        lb_memo_max_entries: usize,
        lb_d1s_max_entries: usize,
        lb_interval_cache_max: usize,
        phase1_screen_cap_s: f64,
    ) -> PyResult<PyObject> {
        let rash_tuning = RashomonTuning {
            lb_threads: lb_threads.max(1),
            stepb_batches,
            stepb_batch_lb_threads,
            lb_interval,
            lb_interval_mink,
            lb_memo_max_entries: lb_memo_max_entries.max(1),
            lb_d1s_max_entries: lb_d1s_max_entries.max(1),
            lb_interval_cache_max,
        };
        // Saturation events before this fit; the DAG is flagged
        // `count_saturated` when any count op of the fit clamped at u128::MAX.
        let sat_events0 = crate::types::count_saturation_events();
        let x_array = x.as_array().to_owned();
        let y_vec = y.as_slice()?.to_vec();
        let n_features = x_array.ncols();
        let n_data = y_vec.len();
        validate_training_input(
            x_array.nrows(),
            n_features,
            &y_vec,
            n_classes,
            reg,
            max_depth,
            max_k,
            min_samples_leaf,
            path_penalty,
            time_limit,
        )?;
        if feature_names
            .as_ref()
            .is_some_and(|names| names.len() != n_features)
        {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "feature_names must contain one name per feature",
            ));
        }
        if !theta.is_finite() || theta < 0.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "theta must be finite and nonnegative",
            ));
        }
        if !crate::types::leaf_only_objective(reg, reg_alpha, path_penalty) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "RESPLINTER supports only the leaf-only objective (reg_alpha=1, path_penalty=0)",
            ));
        }

        let fnames =
            feature_names.unwrap_or_else(|| (0..n_features).map(|i| format!("f{}", i)).collect());

        // Validate + clamp teacher_alpha (as in solve_branch_and_bound).
        if !teacher_alpha.is_finite() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "teacher_alpha must be a finite number in [0.0, 1.0]",
            ));
        }
        let teacher_alpha = teacher_alpha.clamp(0.0, 1.0);

        // Reference-ensemble logit → teacher_prob (softmax-per-row). Mirrors the block in
        // solve_branch_and_bound. Shape check: (n_data, n_classes).
        let teacher_prob_arc: Option<std::sync::Arc<[f64]>> = match teacher_logit {
            Some(pyarr) => {
                let arr = pyarr.as_array();
                let (nrows, ncols) = (arr.nrows(), arr.ncols());
                if nrows != n_data {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "teacher_logit has {} rows but expected n_samples={}",
                        nrows, n_data
                    )));
                }
                if ncols != n_classes {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "teacher_logit has {} columns but expected n_classes={}",
                        ncols, n_classes
                    )));
                }
                let mut out: Vec<f64> = Vec::with_capacity(nrows * ncols);
                for r in 0..nrows {
                    let mut row_max = f64::NEG_INFINITY;
                    for c in 0..ncols {
                        let v = arr[[r, c]];
                        if v > row_max {
                            row_max = v;
                        }
                    }
                    if !row_max.is_finite() {
                        for _ in 0..ncols {
                            out.push(1.0 / ncols as f64);
                        }
                        continue;
                    }
                    let mut denom = 0.0f64;
                    let start = out.len();
                    for c in 0..ncols {
                        let e = (arr[[r, c]] - row_max).exp();
                        out.push(e);
                        denom += e;
                    }
                    if denom > 0.0 {
                        for v in &mut out[start..] {
                            *v /= denom;
                        }
                    } else {
                        for v in &mut out[start..] {
                            *v = 1.0 / ncols as f64;
                        }
                    }
                }
                Some(std::sync::Arc::from(out.into_boxed_slice()))
            }
            None => None,
        };
        let teacher_prob_c0_arc: Option<std::sync::Arc<[f64]>> = teacher_prob_arc
            .as_ref()
            .filter(|_| n_classes == 2)
            .map(|flat| {
                let mut v: Vec<f64> = Vec::with_capacity(n_data);
                for i in 0..n_data {
                    v.push(flat[i * n_classes]);
                }
                std::sync::Arc::from(v.into_boxed_slice())
            });

        let cc_criterion = parse_closure_criterion(closure_criterion)?;

        let feat_types = parse_feature_types(feature_types, n_features)?;
        check_bin_counts(
            x_array.view(),
            &feat_types,
            thresholds_per_feature.as_ref(),
            None,
        )?;

        // BnB solves to lookahead_depth. Remaining levels (max_depth - lookahead)
        // are handled by greedy expansion in Python. Default: full depth.
        // With closure, BnB runs at full max_depth (closure handles below frontier).
        // Without, BnB runs at lookahead_depth (remaining levels handled by Python).
        let solve_depth = if closure_depth.is_some() {
            max_depth
        } else {
            lookahead_depth.unwrap_or(max_depth).min(max_depth)
        };

        let interner = Arc::new(ThresholdInterner::new());

        // ── Phase 1: Find optimal ─────────────────────────────────────────
        let phase1_config = SolverConfig {
            n_classes,
            n_features,
            reg,
            max_depth: solve_depth,
            max_k,
            min_samples_leaf,
            feature_exclusion,
            time_limit,
            initial_used_features: FeatureMask::all(n_features),
            feature_names: fnames.clone(),
            feature_types: feat_types.clone(),
            warm_start_ub: None,
            n_total_override: None,
            thresholds_per_feature: thresholds_per_feature.clone(),
            closure_thresholds_per_feature: None,
            closure_split_ub_gap: f64::INFINITY,
            closure_feature_top_k: None,
            closure_feature_ranking: None,
            closure_min_n_for_multiway: None,
            closure_unsafe_filter_depth_threshold: 0,
            // RUSH optimizes exact misclassification in O(L×k) instead of O(L²×k).
            // Same objective as Misclass, same δ_criterion=0, just faster.
            dp_criterion: DpCriterion::Rush,
            beam_width: beam_width.max(1), // match Phase 2 beam for consistent candidates
            reg_alpha: reg_alpha.clamp(0.0, 1.0),
            local_thresh_scan: false,
            scan_beam: 1,
            adaptive_beam: false,
            // Phase 1 finds the optimum; beam doesn't affect Rashomon completeness
            // here (Phase 2 is the enumeration pass). Keep default beam in Phase 1.
            bypass_beam: false,

            transition_delta: transition_delta.max(0.0), // match Phase 2 compression
            materialize_top_n: 5,                        // enough to tighten bounds via salvage
            allow_heuristic_bound_transfer: false,
            shape_mode,
            shape_sort_smoothing,
            path_penalty: path_penalty.max(0.0),
            greedy_extra_depth: 0,
            rashomon_mode: false,
            rashomon_bound: f64::INFINITY,
            step_b_remaining: 0,
            weight_total: n_data as f64,
            ub_margin: f64::INFINITY,
            n_workers: 1,
            sequential_candidate_gen: false,
            lazy_explore_delta: f64::INFINITY,
            cheap_boundary_greedy: false,
            closure_depth,
            closure_beam: closure_beam.max(1),
            closure_parallel,
            closure_share_memo,
            closure_shared_memo: None,
            closure_criterion: cc_criterion,
            phase0_greedy_prerank_depth: 0,
            eqp_inline_bounds: true,
            threshold_interner: interner.clone(),
            rash_tuning: RashomonTuning::default(),
            msdt: MsdtPortConfig {
                shape_boundary_tax,
                atom_mode,
                teacher_alpha,
                top_k,
                atom_merge_tolerance: atom_merge_tolerance.max(0.0),
                ..MsdtPortConfig::default()
            },
            teacher_prob: teacher_prob_arc.clone(),
            teacher_prob_c0: teacher_prob_c0_arc.clone(),
            reference_preds: None,
            unsafe_filter_depth_threshold,
            admissibility_strict,
        };

        phase1_config
            .validate_admissibility()
            .map_err(pyo3::exceptions::PyValueError::new_err)?;

        // Screened Phase 1 supplies a witnessed anchor, not a global certificate.
        let (phase1_result, mut opt, gap, phase1_root_candidates, phase1_tree) = {
            // When the root screen is the final solve its children get
            // min(time_limit, phase1_screen_cap_s) instead of 1 s.
            // Rashomon Phase 1 only.
            let full_screen_cap = Some(phase1_screen_cap_s);
            py.allow_threads(|| {
                // Use parallel root solver for Phase 1 — evaluates top-N root
                // candidates simultaneously, much better gap closure on problems
                // with many features.
                let (result, tree, root_cands, _profile) = solve_parallel_root_screen(
                    x_array.clone(),
                    y_vec.clone(),
                    phase1_config,
                    20,
                    None,
                    full_screen_cap,
                );
                // Phase 1 currently uses screened parallel-root search, so the
                // returned lower bound is not treated as certified. Anchor Rashomon
                // expansion from the best witnessed objective instead; any residual
                // uncertainty only widens the bound.
                let opt = result.objective;
                let gap = if result.certified_optimal {
                    0.0
                } else {
                    result.final_gap
                };
                (result, opt, gap, root_cands, Some(tree))
            })
        };

        // Phase-2 config at bound `rb` (the up-front anchor's Select config at
        // beta_old, and the Phase-2 search config).
        let mk_phase2 = |rb: f64| {
            phase2_solver_config(Phase2ConfigArgs {
                n_classes,
                n_features,
                reg,
                solve_depth,
                max_k,
                min_samples_leaf,
                feature_exclusion,
                time_limit,
                feature_names: fnames.clone(),
                feature_types: feat_types.clone(),
                warm_start_ub: Some(phase1_result.objective),
                thresholds_per_feature: thresholds_per_feature.clone(),
                beam_width,
                reg_alpha,
                bypass_beam,

                transition_delta,
                shape_mode,
                shape_sort_smoothing,
                path_penalty,
                rashomon_bound: rb,
                // Depth still to be completed below the prefix solve.
                step_b_remaining: max_depth - solve_depth,
                n_data,
                closure_depth,
                closure_beam,
                closure_parallel,
                closure_share_memo,
                closure_criterion: cc_criterion,
                threshold_interner: interner.clone(),
                rash_tuning: rash_tuning.clone(),
                msdt: MsdtPortConfig {
                    shape_boundary_tax,
                    atom_mode,
                    teacher_alpha,
                    top_k,
                    atom_merge_tolerance: atom_merge_tolerance.max(0.0),
                    ..MsdtPortConfig::default()
                },
                teacher_prob: teacher_prob_arc.clone(),
                teacher_prob_c0: teacher_prob_c0_arc.clone(),
                reference_preds: None,
                unsafe_filter_depth_threshold,
                admissibility_strict,
            })
        };

        // ── Up-front full-depth anchor ───────────────────────────────────
        // With Step B, Phase 1 solves the depth-d_l prefix only; complete
        // its best tree greedily over the Step-A Select pool to depth d and
        // anchor at that cost (≤ the prefix cost; a pool tree, so ≥ the pool
        // optimum). The anchor's store is reused as the Step-B store.
        let opt_phase1 = opt;
        let upfront_applies = recursive_fill && closure_depth.is_none() && solve_depth < max_depth;
        let mut opt_upfront: Option<f64> = None;
        let mut anchor_completion_s: Option<f64> = None;
        let mut anchor_store: Option<std::sync::Arc<dataset_store::DatasetStore>> = None;
        if upfront_applies {
            if let Some(tree) = phase1_tree.as_ref() {
                let beta_old = opt_phase1 * (1.0 + theta) + gap;
                let cfg_old = mk_phase2(beta_old);
                let reference_preds_vec = cfg_old.reference_preds.as_ref().map(|arc| arc.to_vec());
                let store = std::sync::Arc::new(dataset_store::DatasetStore::new(
                    x_array.clone(),
                    y_vec.clone(),
                    None,
                    n_classes,
                    cfg_old.thresholds_per_feature.as_ref(),
                    reference_preds_vec,
                ));
                let a = py.allow_threads(|| {
                    rashomon_anchor::complete_prefix_with_select(&store, &cfg_old, tree, max_depth)
                });
                debug_assert!(a.cost <= opt_phase1 + 1e-12, "{} > {}", a.cost, opt_phase1);
                if verbose {
                    eprintln!(
                        "  [anchor] up-front: opt_phase1={:.9} completed={:.9} \
                         leaves_completed={} select_calls={} {:.3}s",
                        opt_phase1, a.cost, a.n_leaves_completed, a.n_select_calls, a.elapsed_s
                    );
                }
                // min(opt_phase1, completed); an uncompleted prefix keeps
                // opt_phase1 bit for bit (no float re-summation drift).
                if a.n_leaves_completed > 0 && a.cost < opt_phase1 {
                    opt = a.cost;
                }
                opt_upfront = Some(opt);
                anchor_completion_s = Some(a.elapsed_s);
                anchor_store = Some(store);
            }
        }

        // Reference cost for the check that the cheapest tree in the final
        // DAG costs at most it (`upfront_anchor_in_dag`). The reference
        // tree comes from the first stage, whose search may use splits
        // outside the candidate splits the DAG enumerates; if it does, the
        // DAG can miss trees within the bound and must not be reported
        // complete. With a lookahead below max_depth the reference is the
        // up-front completion above; with a full-depth first stage
        // (solve_depth == max_depth) it is the first-stage tree's own cost.
        let opt_reference: Option<f64> = if opt_upfront.is_some() {
            opt_upfront
        } else if solve_depth == max_depth && phase1_tree.is_some() {
            Some(opt)
        } else {
            None
        };

        let rashomon_bound_raw = opt * (1.0 + theta) + gap;
        let rashomon_bound = rashomon_bound_raw;

        // ── Phase 2: Rashomon exploration ─────────────────────────────────
        let phase2_config = mk_phase2(rashomon_bound);

        phase2_config
            .validate_admissibility()
            .map_err(pyo3::exceptions::PyValueError::new_err)?;

        // Legacy serialized fields remain readable; no empirical approximation
        // constants are manufactured or presented as coverage certificates.
        let delta_budget = crate::types::DeltaBudget::default();
        // The solver's store (training rows), kept on the returned
        // PyRashomonDag for per-tree leaf n_samples.
        let mut train_store: Option<std::sync::Arc<dataset_store::DatasetStore>> = None;
        let (dag, phase2_stats) = py.allow_threads(|| {
            // Phase 2 uses BFS for broad Rashomon exploration. BFS explores via
            // priority queue (gap-ordered), discovering many more near-optimal
            // subproblems than branch-bound's depth-first path. A known BFS scope
            // propagation issue only affects CONVERGENCE (declaring leaf
            // as optimal), not exploration. Enumeration uses its declared
            // anchor scope. BFS with rashomon_mode=true
            // uses rashomon_bound as the exploration boundary, not the current UB.
            let mut phase2_config_fixed = phase2_config;
            phase2_config_fixed.materialize_top_n = 100;

            // For recursive fill, share the dataset across the parent solver
            // and all per-leaf nested solves via an explicit Arc<DatasetStore>.
            // The Phase B orchestration owns the Arc and dispatches per-leaf
            // BranchAndBoundSolver::new() calls against it after Phase 2.
            // When recursive_fill=false we keep the existing simpler path.
            let phase2_config_for_leaves = if recursive_fill {
                Some(phase2_config_fixed.clone())
            } else {
                None
            };
            let store_for_fill: Option<std::sync::Arc<dataset_store::DatasetStore>> =
                if recursive_fill && anchor_store.is_some() {
                    // Same rows, thresholds and reference preds as the
                    // up-front anchor's store.
                    anchor_store.take()
                } else if recursive_fill {
                    let reference_preds_vec = phase2_config_fixed
                        .reference_preds
                        .as_ref()
                        .map(|arc| arc.to_vec());
                    Some(std::sync::Arc::new(dataset_store::DatasetStore::new(
                        x_array.clone(),
                        y_vec.clone(),
                        None,
                        n_classes,
                        phase2_config_fixed.thresholds_per_feature.as_ref(),
                        reference_preds_vec,
                    )))
                } else {
                    None
                };

            let mut solver = if let Some(store) = store_for_fill.as_ref() {
                let _x_array = x_array;
                let _y_vec = y_vec;
                solver::BranchAndBoundSolver::new(store.clone(), phase2_config_fixed, None, None)
            } else {
                solver::BranchAndBoundSolver::from_raw(
                    x_array,
                    y_vec,
                    phase2_config_fixed,
                    None,
                    None,
                )
            };

            // Pre-seed root with Phase 1's root candidates. This ensures Phase 2
            // knows about good root splits that the parallel root found but a
            // sequential solver would miss.
            solver.preseed_root_candidates(&phase1_root_candidates, rashomon_bound);

            // Keep the Phase-2 SolveResult (timeouts are surfaced).
            // The completion search (Step B) shares the Phase-2 budget (`PrefixLeafFillConfig::deadline`).
            let phase2_t0 = std::time::Instant::now();
            let phase2_result = solver.solve();
            // Memo diagnostics
            let memo_len = solver.search.memo.len();
            let mut n_solved = 0usize;
            let mut n_terminal = 0usize;
            let mut n_explored = 0usize;
            let mut total_mat_splits = 0usize;
            let mut total_rash_splits = 0usize;
            let mut total_child_keys = 0usize;
            let mut n_with_best = 0usize;
            for entry in solver.search.memo.iter() {
                let memo_key = entry.key();
                let sp = entry.value();
                if sp.is_solved {
                    n_solved += 1;
                }
                if sp.is_terminal {
                    n_terminal += 1;
                }
                if sp.explored {
                    n_explored += 1;
                }
                if solver.search.best_split_table.contains_key(memo_key) {
                    n_with_best += 1;
                }
                // Materialized splits + best_split live in the global
                // tables; rashomon_indices still lives on SubProblem (duplicate
                // semantics preserved). Telemetry counts via the bound_table
                // iterator + the legacy rashomon_indices Vec.
                let bound_entries = solver.search.bound_features_at(memo_key);
                let mat_pool_idxs: Vec<u32> = bound_entries
                    .iter()
                    .filter(|e| e.is_materialized())
                    .map(|e| e.pool_idx)
                    .collect();
                let best_split_child_count = solver
                    .search
                    .best_split_table
                    .get(memo_key)
                    .and_then(|split_id| {
                        let split_id_v = *split_id;
                        bound_entries.iter().find_map(|e| {
                            let rec = solver.worker.split_pool.get(e.pool_idx);
                            if rec.split_id == split_id_v {
                                Some(rec.child_keys.len())
                            } else {
                                None
                            }
                        })
                    })
                    .unwrap_or(0);
                total_mat_splits += mat_pool_idxs.len();
                total_rash_splits += sp.rashomon_indices.len();
                total_child_keys += mat_pool_idxs
                    .iter()
                    .copied()
                    .chain(sp.rashomon_indices.iter().copied())
                    .map(|idx| solver.worker.split_pool.get(idx).child_keys.len())
                    .sum::<usize>()
                    + best_split_child_count;
            }
            if verbose {
                eprintln!(
                    "  [mem] after the Rashomon-set solve: {} MB, memo entries: {}",
                    current_rss_mb(),
                    memo_len
                );
                eprintln!(
                    "  [memo] solved={} terminal={} explored={} has_best={}",
                    n_solved, n_terminal, n_explored, n_with_best
                );
                eprintln!(
                    "  [memo] materialized_splits={} rashomon_splits={} total_child_keys={}",
                    total_mat_splits, total_rash_splits, total_child_keys
                );
                let p = &solver.worker.profile;
                eprintln!(
                    "  [prof] enumeration time_candidate_gen={:.3}s n_candidates_generated={} \
                     n_candidates_survived_phase_a={} nodes_at_depth={:?} \
                     n_nodes_purified={} n_reexplorations={} n_lb_selects={} \
                     n_lb_d1_closed={} n_lb_d1_fallback={} \
                     lb_select_cache_records={} n_lb_d1_rowwalk={} n_lb_d1_memo_hits={} \
                     time_lb_d1_closed={:.3}s n_store_thresholds_max={} \
                     n_store_thresholds_sum={} n_store_threshold_features={}",
                    p.time_candidate_gen.as_secs_f64(),
                    p.n_candidates_generated,
                    p.n_candidates_survived_phase_a,
                    p.nodes_at_depth,
                    p.n_nodes_purified,
                    p.n_reexplorations,
                    p.n_lb_selects,
                    p.n_lb_d1_closed,
                    p.n_lb_d1_fallback,
                    p.lb_select_cache_records,
                    p.n_lb_d1_rowwalk,
                    p.n_lb_d1_memo_hits,
                    p.time_lb_d1_closed.as_secs_f64(),
                    p.n_store_thresholds_max,
                    p.n_store_thresholds_sum,
                    p.n_store_threshold_features,
                );
            }
            // Full lazy path — solve completes, snapshot owns the
            // build inputs, return `RashomonSet::Pending`. The first downstream
            // `dag_mut()` call (in result_dict population below) triggers
            // `build_dag_from_snapshot`, which:
            //   1. runs `build_factored_dag` from owned memo/pool/arena
            //   2. drops memo/pool/arena
            //   3. runs `quotient_factored_dag`
            //   4. if `prefix_leaf_fill` is Some, runs `fill_prefix_leaves`
            // ...then returns the FactoredDag as the sole resident state.
            // The recursive_fill branch attaches a `PrefixLeafFillConfig`
            // to the snapshot so the fill extraction runs inside ensure_built.
            // Step-A Select / completion-lb times, read before the lb caches
            // move to the shared Step-B solver (shared Step B).
            let phase2_time_candidate_gen_s =
                solver.worker.profile.time_candidate_gen.as_secs_f64();
            let phase2_time_candidate_gen_lb_s =
                solver.worker.profile.time_candidate_gen_lb.as_secs_f64();
            let phase2_lb_stats = solver.worker.profile.lb_stats();
            let prefix_leaf_fill =
                if recursive_fill && store_for_fill.is_some() && phase2_config_for_leaves.is_some()
                {
                    let rash_lb_cache = {
                        let c = solver.take_rash_lb_cache();
                        c.key
                            .is_some()
                            .then(|| std::sync::Arc::new(std::sync::Mutex::new(Some(c))))
                    };
                    Some(crate::types::PrefixLeafFillConfig {
                        store: store_for_fill.as_ref().unwrap().clone(),
                        leaf_base_config: phase2_config_for_leaves.as_ref().unwrap().clone(),
                        reg,
                        leaf_reg_cap,
                        leaf_short_circuit_eps,
                        n_data,
                        fill_parallel,
                        verbose,
                        deadline: time_limit.and_then(|t| {
                            std::time::Duration::try_from_secs_f64(t.max(0.0))
                                .ok()
                                .and_then(|d| phase2_t0.checked_add(d))
                        }),
                        rash_lb_cache,
                    })
                } else {
                    None
                };
            train_store = Some(solver.search.store.clone());
            let mut snapshot =
                solver.into_search_snapshot(opt, theta, delta_budget, cost_precision, max_depth);
            snapshot.prefix_leaf_fill = prefix_leaf_fill;
            let phase2_stats = (
                phase2_result.timed_out,
                phase2_result.n_explored,
                memo_len,
                phase2_time_candidate_gen_s,
                total_rash_splits,
                phase2_time_candidate_gen_lb_s,
                phase2_lb_stats,
            );
            return (
                crate::types::RashomonSet::Pending {
                    snapshot: Some(Box::new(snapshot)),
                },
                Some(phase2_stats),
            );
        });

        // Trigger materialization for downstream code that needs
        // the FactoredDag (summary, result_dict, py_dag construction). For
        // the deferred path (no recursive_fill), this is where the actual
        // build runs — `build_dag_from_snapshot` consumes the snapshot and
        // drops memo+pool+arena before returning the FactoredDag.
        let mut dag_rs = dag;
        dag_rs.ensure_built();
        let mut dag = dag_rs.dag().clone();
        // Scan for clamped counts only when some count op saturated (the
        // global counter also sees concurrent fits in this process).
        dag.count_saturated |= crate::types::count_saturation_events() > sat_events0
            && rashomon_dag::dag_has_saturated_count(&dag);
        // Re-anchor (`reanchor_at_dag_opt`, inside the build): the DAG
        // may now be anchored at its own optimum; every reported anchor,
        // bound and target below describes the re-anchored set.
        let reanchored = dag.delta_budget.reanchored;
        let opt_final = if reanchored { dag.opt } else { opt };
        let rashomon_bound_final = if reanchored {
            dag.rashomon_bound
        } else {
            rashomon_bound
        };

        // ── Build result ────────────────────────────────────────────────
        let summary = rashomon_dag::dag_summary_factored(&dag);

        let result_dict = pyo3::types::PyDict::new_bound(py);
        result_dict.set_item("candidate_ranking_policy", CANDIDATE_RANKING_POLICY)?;
        result_dict.set_item("tree_count", format!("{}", dag.tree_count))?;
        result_dict.set_item("count_saturated", dag.count_saturated)?;
        result_dict.set_item("exact_cost_keys", dag.exact_cost_keys)?;
        // Enumeration runs at the gap-widened bound opt*(1+θ)+gap; also
        // report the count at the target bound opt*(1+θ) and whether
        // phase 1 was certified (gap == 0, so the two bounds coincide).
        // The target is in DAG cost units, built like `rashomon_bound`
        // (RB_FLOOR_INT floor, INT_OBJ round(·n)) so gap == 0 gives the same
        // bound and tree_count_at_target == tree_count.
        let rashomon_bound_target = opt_final * (1.0 + theta);
        let tree_count_at_target = rashomon_dag::tree_count_at_bound(&dag, rashomon_bound_target);
        result_dict.set_item("tree_count_at_target", format!("{}", tree_count_at_target))?;
        result_dict.set_item("rashomon_bound_target", rashomon_bound_target)?;
        // The same report anchored at the DAG's own optimum, opt_dag = min
        // cost in the final root schedule (DAG cost units):
        // rashomon_bound_target_dag = opt_dag*(1+θ) and the count within it
        // (== opt after a re-anchor; without one, opt is the Phase-1 prefix
        // or up-front anchor). None when the DAG holds no tree.
        let opt_dag_val = dag
            .root_cost_schedule
            .iter()
            .map(|e| e.cost)
            .reduce(f64::min);
        match opt_dag_val {
            Some(opt_dag) => {
                let bound_dag = opt_dag * (1.0 + theta);
                result_dict.set_item("opt_dag", opt_dag)?;
                result_dict.set_item("rashomon_bound_target_dag", bound_dag)?;
                result_dict.set_item(
                    "tree_count_at_target_dag",
                    format!("{}", rashomon_dag::tree_count_at_bound(&dag, bound_dag)),
                )?;
            }
            None => {
                for k in [
                    "opt_dag",
                    "rashomon_bound_target_dag",
                    "tree_count_at_target_dag",
                ] {
                    result_dict.set_item(k, py.None())?;
                }
            }
        }
        // `gap` is 0 by construction on the screened Phase-1 path
        // (`solve_parallel_root` hard-codes final_gap = 0): gap == 0 means
        // only "target bound == enumeration bound", it certifies nothing.
        // `phase1_certified` is an alias of `phase1_gap_zero`; the
        // optimality flag is `phase1_certified_optimal`.
        result_dict.set_item("phase1_certified", gap == 0.0)?;
        result_dict.set_item("phase1_gap_zero", gap == 0.0)?;
        // Anchor provenance: the Phase-1 prefix optimum, the up-front
        // full-depth completion (None when not applicable), whether the DAG
        // optimum is within the reference cost `opt_reference` (that
        // completion, or the full-depth first-stage tree), the
        // post-Step-B re-anchor,
        // and the bound Step A/B enumerated at.
        result_dict.set_item("opt_phase1", opt_phase1)?;
        result_dict.set_item("opt_upfront", opt_upfront)?;
        // Whether the DAG minimum is within the reference cost
        // (`opt_reference`: the up-front completion, or the first-stage
        // tree at full depth); None when no first-stage reference applies.
        // false also when the DAG holds no tree; false marks the set
        // incomplete below.
        let upfront_in_dag = match (opt_reference, opt_dag_val) {
            (Some(u), Some(d)) => Some(d <= u + 1e-9),
            (Some(_), None) => Some(false),
            (None, _) => None,
        };
        result_dict.set_item("upfront_anchor_in_dag", upfront_in_dag)?;
        result_dict.set_item("reanchored", reanchored)?;
        result_dict.set_item(
            "opt_pre_reanchor",
            if reanchored {
                dag.delta_budget.reanchor_opt_pre
            } else {
                opt
            },
        )?;
        result_dict.set_item(
            "rashomon_bound_enum",
            if reanchored {
                dag.delta_budget.reanchor_bound_pre
            } else {
                rashomon_bound
            },
        )?;
        result_dict.set_item(
            "tree_count_pre_reanchor",
            format!(
                "{}",
                if reanchored {
                    dag.delta_budget.reanchor_tree_count_pre
                } else {
                    dag.tree_count
                }
            ),
        )?;
        result_dict.set_item(
            "anchor_mode",
            if opt_upfront.is_some() {
                "upfront_select_completion"
            } else {
                "phase1_prefix"
            },
        )?;
        result_dict.set_item("anchor_completion_s", anchor_completion_s)?;
        result_dict.set_item("phase1_timed_out", phase1_result.timed_out)?;
        result_dict.set_item(
            "phase1_termination_reason",
            phase1_result.termination_reason.clone(),
        )?;
        result_dict.set_item("phase1_elapsed_s", phase1_result.elapsed)?;
        result_dict.set_item("n_dag_nodes", summary.n_nodes)?;
        result_dict.set_item("n_quotient_hash_collisions", dag.n_quotient_hash_collisions)?;
        result_dict.set_item("n_internal", summary.n_internal)?;
        result_dict.set_item("n_leaf_only", summary.n_leaf_only)?;
        result_dict.set_item("n_both", summary.n_both)?;
        result_dict.set_item("total_split_entries", summary.total_split_entries)?;
        result_dict.set_item("opt", opt_final)?;
        result_dict.set_item("rashomon_bound", rashomon_bound_final)?;
        result_dict.set_item("theta", theta)?;
        result_dict.set_item("phase1_gap", gap)?;
        result_dict.set_item("phase1_objective", phase1_result.objective)?;
        result_dict.set_item("phase1_lowerbound", phase1_result.lowerbound)?;
        result_dict.set_item("phase1_is_optimal", phase1_result.is_optimal)?;
        result_dict.set_item("phase1_certified_optimal", phase1_result.certified_optimal)?;
        result_dict.set_item("phase1_search_converged", phase1_result.search_converged)?;
        result_dict.set_item(
            "phase1_certification_mode",
            phase1_result.certification_mode,
        )?;
        result_dict.set_item(
            "phase1_certification_reason",
            phase1_result.certification_reason,
        )?;
        result_dict.set_item(
            "phase1_unsafe_heuristics_used",
            phase1_result.unsafe_heuristics_used,
        )?;
        result_dict.set_item("max_depth", max_depth)?;
        result_dict.set_item("solve_depth", solve_depth)?;
        result_dict.set_item("remaining_depth", max_depth - solve_depth)?;
        // Step B mode: 'paper' (`stepb_fill`), 'none' (no prefix
        // truncation, no recursive fill, the lickety path, or no prefix
        // leaf to fill: closure mode and full depth solve to max_depth).
        let n_pl_left = dag
            .subproblems
            .iter()
            .filter(|sp| sp.prefix_leaf.is_some())
            .count();
        let truncated = phase2_stats.is_some() && recursive_fill && solve_depth < max_depth;
        let stepb_mode = if truncated && dag.delta_budget.n_prefix_leaves > 0 {
            "paper"
        } else {
            "none"
        };
        result_dict.set_item("stepb_mode", stepb_mode)?;
        if stepb_mode == "paper" {
            let b = &dag.delta_budget;
            result_dict.set_item("n_prefix_leaves", b.n_prefix_leaves)?;
            result_dict.set_item("n_leaves_filled", b.n_leaves_filled)?;
            result_dict.set_item("n_leaves_empty", b.n_leaves_empty)?;
            result_dict.set_item("n_leaves_timed_out", b.n_leaves_timed_out)?;
            result_dict.set_item(
                "n_stepb_groups_unequal_scope",
                b.n_stepb_groups_unequal_scope,
            )?;
            result_dict.set_item("time_candidate_gen_stepb_s", b.time_candidate_gen_stepb_s)?;
            result_dict.set_item("stepb_shared", b.stepb_shared)?;
        } else {
            result_dict.set_item("n_prefix_leaves", n_pl_left)?;
            for k in [
                "n_leaves_filled",
                "n_leaves_empty",
                "n_leaves_timed_out",
                "n_stepb_groups_unequal_scope",
                "time_candidate_gen_stepb_s",
                "stepb_shared",
            ] {
                result_dict.set_item(k, py.None())?;
            }
        }
        // Phase-2 completeness. `rashomon_complete` = the returned
        // set is exactly the Rashomon set R(D_θ) over the retained pool: the
        // Step-A pass (Step A) ran to completion, no time-out, no
        // closure below the prefix, any depth below the prefix completed by
        // the completion search (Step B) with no nested time-out, exact lattice cost keys and
        // the leaf-only objective (the DAG prices trees by leaf costs).
        // `rashomon_incomplete_reason` lists every failed condition
        // (';'-joined, "" when complete). A time-out makes every count a
        // lower bound. None on the lickety path.
        match phase2_stats {
            Some((
                timed_out,
                n_explored,
                memo_len,
                time_candidate_gen_s,
                n_rashomon_splits,
                time_candidate_gen_lb_s,
                lb_stats_stepa,
            )) => {
                let mut reasons: Vec<&str> = Vec::new();
                if timed_out {
                    reasons.push("phase2_timeout");
                }
                if closure_depth.is_some() {
                    reasons.push("closure");
                }
                if solve_depth < max_depth {
                    // Completion search (Step B) with no placeholder (stepb_mode 'none')
                    // had nothing to complete.
                    if !recursive_fill {
                        reasons.push("unfilled_prefix_leaves");
                    } else if dag.delta_budget.n_leaves_timed_out > 0 {
                        reasons.push("stepb_timeout");
                    }
                }
                if !dag.exact_cost_keys {
                    reasons.push("rounded_keys");
                }
                if !crate::types::leaf_only_objective(reg, reg_alpha, path_penalty) {
                    reasons.push("non_leaf_objective");
                }
                // The DAG minimum exceeds the reference cost (the up-front
                // anchor, or the full-depth first-stage tree): the reference
                // tree is not in the DAG, so the bound is not set from a
                // tree of the set.
                if upfront_in_dag == Some(false) {
                    reasons.push("upfront_anchor_not_in_dag");
                }
                result_dict.set_item("phase2_timed_out", timed_out)?;
                result_dict.set_item("rashomon_complete", reasons.is_empty())?;
                result_dict.set_item("rashomon_incomplete_reason", reasons.join(";"))?;
                result_dict.set_item("n_explored", n_explored)?;
                result_dict.set_item("memo_len", memo_len)?;
                // Step-A (Phase-2 solver) Select time and recorded splits.
                result_dict.set_item("time_candidate_gen_s", time_candidate_gen_s)?;
                result_dict.set_item("n_rashomon_splits", n_rashomon_splits)?;
                // Completion-lb DFS Select time (exact completion lb), Step
                // A + Step B (not in either time_candidate_gen_*).
                let lb_stepb = if stepb_mode == "paper" {
                    dag.delta_budget.time_candidate_gen_lb_stepb_s
                } else {
                    0.0
                };
                result_dict.set_item(
                    "time_candidate_gen_lb_s",
                    time_candidate_gen_lb_s + lb_stepb,
                )?;
                // Completion-lb DFS counters, Step A + Step B
                // (cache records: the peak over the solvers).
                let lb_stats = if stepb_mode == "paper" {
                    crate::types::lb_stats_merge(lb_stats_stepa, dag.delta_budget.lb_stats_stepb)
                } else {
                    lb_stats_stepa
                };
                for (k, v) in LB_STATS_KEYS.iter().zip(lb_stats) {
                    result_dict.set_item(*k, v)?;
                }
                result_dict.set_item(LB_STATS_DERIVED_KEYS[0], lb_stats[6] as f64 / 1e9)?;
                result_dict.set_item(LB_STATS_DERIVED_KEYS[1], lb_stats[7])?;
                let thr_mean = if lb_stats[9] > 0 {
                    lb_stats[8] as f64 / lb_stats[9] as f64
                } else {
                    0.0
                };
                result_dict.set_item(LB_STATS_DERIVED_KEYS[2], thr_mean)?;
                for (k, v) in LB_STATS_EXTRA_KEYS.iter().zip(&lb_stats[10..]) {
                    result_dict.set_item(*k, *v)?;
                }
                // Interval-lb counters (process-wide; `lb_interval` only),
                // except the similar-support skip and depth-3 scan counters
                // (always; the skip counters are 0 with `lb_interval` off).
                let iv_on = lb_interval;
                for (k, v) in crate::solver::rash_iv_stats() {
                    let always = k.contains("_simskip_")
                        || k.ends_with("_scan_d3")
                        || k.ends_with("_scan_d3_ns");
                    if iv_on || always {
                        result_dict.set_item(k, v)?;
                    }
                }
                // Counts of a timed-out run are lower bounds: drivers
                // report such a run as DNF (rashomon_cell.py
                // --no-partial-counts).
                let n_timed_out_stepb = if stepb_mode == "paper" {
                    dag.delta_budget.n_leaves_timed_out
                } else {
                    0
                };
                result_dict.set_item("stepb_partial", timed_out || n_timed_out_stepb > 0)?;
            }
            None => {
                for k in [
                    "phase2_timed_out",
                    "rashomon_complete",
                    "rashomon_incomplete_reason",
                    "n_explored",
                    "memo_len",
                    "time_candidate_gen_s",
                    "n_rashomon_splits",
                    "time_candidate_gen_lb_s",
                    "stepb_partial",
                ]
                .iter()
                .chain(LB_STATS_KEYS.iter())
                .chain(LB_STATS_DERIVED_KEYS.iter())
                .chain(LB_STATS_EXTRA_KEYS.iter())
                {
                    result_dict.set_item(*k, py.None())?;
                }
            }
        }

        let delta_dict = pyo3::types::PyDict::new_bound(py);
        delta_dict.set_item("delta_beam", dag.delta_budget.delta_beam)?;
        delta_dict.set_item("delta_compress", dag.delta_budget.delta_compress)?;
        delta_dict.set_item("delta_quantize", dag.delta_budget.delta_quantize)?;
        delta_dict.set_item("delta_criterion", dag.delta_budget.delta_criterion)?;
        delta_dict.set_item("delta_total", dag.delta_budget.delta_total)?;
        delta_dict.set_item("delta_leaf_max", dag.delta_budget.delta_leaf_max)?;
        delta_dict.set_item("delta_leaf_weighted", dag.delta_budget.delta_leaf_weighted)?;
        delta_dict.set_item("n_leaves_solved", dag.delta_budget.n_leaves_solved)?;
        delta_dict.set_item(
            "n_leaves_short_circuited",
            dag.delta_budget.n_leaves_short_circuited,
        )?;
        result_dict.set_item("delta_budget", delta_dict)?;

        // Wrap the DAG in a PyO3 class so Python can access trees by index.
        // `fit_config` is populated by the Python wrapper post-fit via
        // `set_fit_config`; Rust doesn't know about Python-only LightGBM
        // reference-ensemble-cache kwargs here.
        let py_dag = PyRashomonDag {
            dag: RashomonSet::from_built(dag),
            feature_names: fnames,
            feature_types: feat_types,
            fit_config: None,
            target_bound: Some(rashomon_bound_target),
            n_train: train_store.as_ref().map(|s| s.n_samples() as u64),
            train_store,
        };
        result_dict.set_item("dag", pyo3::Py::new(py, py_dag)?)?;

        Ok(result_dict.into_any().unbind())
    }

    #[pymodule]
    pub fn multiway_solver(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add("CANDIDATE_RANKING_POLICY", CANDIDATE_RANKING_POLICY)?;
        m.add_function(wrap_pyfunction!(solve_branch_and_bound, m)?)?;
        m.add_function(wrap_pyfunction!(solve_rashomon, m)?)?;
        m.add_function(wrap_pyfunction!(rashomon_select_debug, m)?)?;
        m.add_class::<PyRashomonDag>()?;
        Ok(())
    }
} // mod python_bindings
