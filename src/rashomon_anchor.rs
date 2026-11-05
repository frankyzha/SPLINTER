//! Up-front full-depth anchor of the Rashomon bound.
//!
//! With the completion search (Step B), Phase 1 solves only the depth-d_l prefix, so its best
//! tree prices the bound at the depth-d_l optimum. `complete_prefix_with_select`
//! completes every leaf of that prefix to depth `max_depth` with a greedy
//! recursion over the Step-A Select pool (`rashomon_select_depth`, the same
//! Select_B(u) lists Step B enumerates), keeping a leaf whenever no split is
//! cheaper. The completed tree is a member of the retained pool, so its cost
//! is an upper bound on the pool optimum (RESPLIT's "prefix + CART
//! completion" anchor, here over the pool).

use std::sync::Arc;
use std::time::Instant;

use crate::dataset_store::DatasetStore;
use crate::solver::BranchAndBoundSolver;
use crate::tree::TreeNode;
use crate::types::{FeatureMask, SolverConfig, SupportMask};

/// Result of `complete_prefix_with_select`.
pub struct UpfrontAnchor {
    /// Global units: err / n_total + leaf_reg · leaves (+ split_reg · splits).
    pub cost: f64,
    /// The completed depth-≤d tree (debugging / n_leaves export).
    pub tree: TreeNode,
    /// Prefix leaves that the completion replaced by a split.
    pub n_leaves_completed: usize,
    pub n_select_calls: usize,
    pub elapsed_s: f64,
}

struct Walker<'a> {
    store: &'a Arc<DatasetStore>,
    solver: BranchAndBoundSolver,
    n_classes: usize,
    n_total: f64,
    leaf_reg: f64,
    split_reg: f64,
    /// feature_exclusion: the mask shrinks along the path (`rash_explore_node`)
    /// and resets to all features where a Step-B mini is rooted (depths
    /// d_l, 2·d_l, …: `stepb_mini_resplinter` / shared Step B).
    feature_exclusion: bool,
    d_l: i32,
    n_features: usize,
    n_leaves_completed: usize,
    n_select_calls: usize,
}

impl Walker<'_> {
    /// The mask a node at `depth` gets from its parent's mask `parent` after
    /// a split on `feature` (Step-B roots reset it).
    fn child_mask(&self, parent: &FeatureMask, feature: usize, depth: i32) -> FeatureMask {
        if depth > 0 && depth % self.d_l == 0 {
            FeatureMask::all(self.n_features)
        } else if self.feature_exclusion {
            parent.without(feature)
        } else {
            parent.clone()
        }
    }

    /// Leaf-only objective of `support` (the DAG's leaf pricing) and its leaf.
    fn leaf(&self, support: &SupportMask) -> (f64, TreeNode) {
        let (cc, n) = self.store.count_classes(support);
        let counts = &cc[..self.n_classes];
        let max_cc = counts.iter().copied().max().unwrap_or(0);
        // Rust max_by_key: the last maximal class.
        let prediction = (0..self.n_classes).max_by_key(|&c| counts[c]).unwrap_or(0);
        let loss = if n == 0 {
            0.0
        } else {
            (n as f64 - max_cc as f64) / self.n_total
        };
        let leaf = TreeNode::Leaf {
            prediction,
            n_samples: n,
            loss,
        };
        (loss + self.leaf_reg, leaf)
    }

    /// Greedy completion of `support` with `rem` levels: argmin `split_ub`
    /// over Select (ties: `candidate_tiebreak_cmp`), children completed
    /// recursively; the leaf when the split is not cheaper.
    fn greedy_completion(
        &mut self,
        support: &SupportMask,
        mask: &FeatureMask,
        depth: i32,
        rem: i32,
    ) -> (f64, TreeNode) {
        let (leaf_cost, leaf) = self.leaf(support);
        if rem <= 0 || support.n_samples <= 1 {
            return (leaf_cost, leaf);
        }
        // Remaining depth within the solve that owns this node (Step A to
        // d_l, each Step-B mini to the next multiple of d_l).
        let select_depth = (self.d_l - depth % self.d_l).min(rem);
        let cands = self
            .solver
            .rashomon_select_depth(support, mask, select_depth);
        self.n_select_calls += 1;
        let mut best: Option<(&crate::types::SplitCandidate, Vec<SupportMask>)> = None;
        for c in &cands {
            // Mirror explore_node: no candidate with an empty child.
            if c.child_n.iter().any(|&n| n <= 0) {
                continue;
            }
            let better = match &best {
                None => true,
                Some((b, _)) => c
                    .split_ub
                    .total_cmp(&b.split_ub)
                    .then_with(|| crate::solver::candidate_tiebreak_cmp(c, b))
                    .is_lt(),
            };
            if !better {
                continue;
            }
            let parts = self.store.partition_support(
                support,
                c.feature_index,
                &c.thresholds,
                c.group_map.as_deref(),
            );
            if parts.len() != c.k || parts.iter().any(|p| p.n_samples == 0) {
                continue;
            }
            best = Some((c, parts));
        }
        let Some((c, parts)) = best else {
            return (leaf_cost, leaf);
        };
        let mut split_cost = self.split_reg;
        let mut children = Vec::with_capacity(parts.len());
        let cmask = self.child_mask(mask, c.feature_index, depth + 1);
        for p in &parts {
            let (cc, ct) = self.greedy_completion(p, &cmask, depth + 1, rem - 1);
            split_cost += cc;
            children.push(ct);
        }
        if split_cost < leaf_cost {
            let node = TreeNode::Internal {
                feature_index: c.feature_index,
                thresholds: c.thresholds.clone(),
                children,
                group_map: c.group_map.clone(),
            };
            (split_cost, node)
        } else {
            (leaf_cost, leaf)
        }
    }

    /// Walk `node` (routing `support` like `route_leaf_n_samples`), completing
    /// every leaf above `max_depth`; returns the cost and the completed tree.
    fn walk(
        &mut self,
        node: &TreeNode,
        support: &SupportMask,
        mask: &FeatureMask,
        depth: i32,
        max_depth: i32,
    ) -> (f64, TreeNode) {
        match node {
            TreeNode::Internal {
                feature_index,
                thresholds,
                children,
                group_map,
            } => {
                let parts = self.store.partition_support(
                    support,
                    *feature_index,
                    thresholds,
                    group_map.as_deref(),
                );
                assert_eq!(
                    parts.len(),
                    children.len(),
                    "complete_prefix_with_select: split on feature {} routes to {} parts for {} children",
                    feature_index,
                    parts.len(),
                    children.len()
                );
                let mut cost = self.split_reg;
                let mut out = Vec::with_capacity(children.len());
                let cmask = self.child_mask(mask, *feature_index, depth + 1);
                for (child, part) in children.iter().zip(parts.iter()) {
                    let (c, t) = self.walk(child, part, &cmask, depth + 1, max_depth);
                    cost += c;
                    out.push(t);
                }
                let node = TreeNode::Internal {
                    feature_index: *feature_index,
                    thresholds: thresholds.clone(),
                    children: out,
                    group_map: group_map.clone(),
                };
                (cost, node)
            }
            TreeNode::Leaf { .. } => {
                if depth < max_depth {
                    let (c, t) = self.greedy_completion(support, mask, depth, max_depth - depth);
                    if matches!(t, TreeNode::Internal { .. }) {
                        self.n_leaves_completed += 1;
                    }
                    (c, t)
                } else {
                    self.leaf(support)
                }
            }
        }
    }
}

/// Complete every leaf of `prefix` (a depth-≤d_l tree, Phase-1 best) to depth
/// `max_depth` with a greedy recursion over the Step-A Select pool; keep the
/// leaf when no split is cheaper. `phase2_cfg` is the Phase-2 config with
/// `rashomon_bound` = β_old (the Select ub_hint), whose `max_depth` is the
/// prefix depth d_l.
pub fn complete_prefix_with_select(
    store: &Arc<DatasetStore>,
    phase2_cfg: &SolverConfig,
    prefix: &TreeNode,
    max_depth: i32,
) -> UpfrontAnchor {
    let t0 = Instant::now();
    let mut cfg = phase2_cfg.clone();
    cfg.time_limit = None;
    cfg.n_workers = 1;
    cfg.closure_depth = None;
    cfg.closure_shared_memo = None;
    cfg.closure_parallel = false;
    let n_classes = cfg.n_classes;
    let n_total = cfg.n_total_override.unwrap_or(store.n_samples()) as f64;
    let leaf_reg = cfg.reg_alpha * cfg.reg;
    let split_reg = (1.0 - cfg.reg_alpha) * cfg.reg;
    let feature_exclusion = cfg.feature_exclusion;
    let d_l = cfg.max_depth.max(1);
    let n_features = cfg.n_features;
    let root_mask = cfg.initial_used_features.clone();
    let root = store.root_support();
    let solver = BranchAndBoundSolver::new(store.clone(), cfg, None, Some(root.clone()));
    let mut w = Walker {
        store,
        solver,
        n_classes,
        n_total,
        leaf_reg,
        split_reg,
        feature_exclusion,
        d_l,
        n_features,
        n_leaves_completed: 0,
        n_select_calls: 0,
    };
    #[cfg(debug_assertions)]
    {
        let (cc, n) = store.count_classes(&root);
        let reference = w
            .solver
            .leaf_objective_from_counts(&cc[..n_classes], n as f64);
        debug_assert!((w.leaf(&root).0 - reference).abs() <= 1e-12);
    }
    let (cost, tree) = w.walk(prefix, &root, &root_mask, 0, max_depth);
    UpfrontAnchor {
        cost,
        tree,
        n_leaves_completed: w.n_leaves_completed,
        n_select_calls: w.n_select_calls,
        elapsed_s: t0.elapsed().as_secs_f64(),
    }
}
