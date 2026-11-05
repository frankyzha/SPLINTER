"""Anchor: up-front full-depth anchor (Phase-1 prefix completed over the
Step-A Select pool), post-Step-B re-anchor at the DAG optimum and the
full-time Phase-1 screen.

The DAG optimum opt_dag can be strictly cheaper than the up-front anchor;
the reported set is then re-anchored at opt_dag, so the set is always the
pool's trees within opt_dag*(1+theta). Per instance and (d, d_l) in
{(3, 1), (3, 2), (4, 2)} (20 tiny instances, n <= 20):
  1. opt == opt_dag, rashomon_bound == opt_dag*(1+theta), tree_count ==
     tree_count_at_target == tree_count_at_target_dag, reanchored iff
     opt_dag < the up-front anchor, the up-front anchor >= opt_dag
     (upfront_anchor_in_dag);
  2. the brute-force oracle over the same Select pool (Select ub = the
     enumeration bound rashomon_bound_enum; trees at rashomon_bound): count
     and key-set equality with every get_tree(i), every objective <= bound,
     sample_indices(at_target) within [0, tree_count);
  3. to_bytes / from_bytes: same count, opt, target bound, ordered keys;
  4. phase1_timed_out is False;
  5. some instance re-anchors (the re-anchor path is exercised);
  6. with d_l = d (one full-depth search, no up-front completion) the check
     that the DAG minimum is within the first-stage tree's cost still runs
     (upfront_anchor_in_dag is True, not None) and the set is complete.
"""
import numpy as np
import pytest

from splinter.multiway_solver import solve_rashomon
from splinter.resplinter import dict_to_tree

from _helpers import EPS, KEY_CAP, SelectOracle, frac_budget, small_instances, tree_key, tree_objective

N_INST = 20
CONFIGS = [(3, 1), (3, 2), (4, 2)]
FE_CONFIGS = [(3, 1), (4, 2)]  # feature_exclusion=True


def _solve(inst, depth, lookahead, feature_exclusion=False):
    X = np.asarray(inst["X"], dtype=np.float64)
    y = np.asarray(inst["y"], dtype=np.uint8)
    return solve_rashomon(
        X, y, 2, inst["reg"], depth, inst["max_k"], inst["theta"],
        lookahead_depth=lookahead, min_samples_leaf=1, beam_width=inst["beam"],
        feature_exclusion=feature_exclusion, reg_alpha=1.0, transition_delta=0.02, atom_mode=True,
        thresholds_per_feature={f: list(t) for f, t in inst["thresholds"].items()},
        recursive_fill=True, fill_parallel=False)


def _keys_and_checks(res, X, y, reg, bound):
    """Ordered keys of all trees, trees over bound."""
    dag, tc = res["dag"], int(res["tree_count"])
    keys, over = [], 0
    for i in range(tc):
        t = dict_to_tree(dag.get_tree(i))
        if tree_objective(t, X, y, reg) > bound + EPS:
            over += 1
        keys.append(tree_key(t))
    return keys, over


def run_instance(inst, depth, lookahead):
    X = np.asarray(inst["X"], dtype=np.float64)
    y = np.asarray(inst["y"], dtype=np.uint8)
    res = _solve(inst, depth, lookahead)
    theta = inst["theta"]
    tc = int(res["tree_count"])
    fails = []

    def chk(name, cond):
        if not cond:
            fails.append(name)

    if res["opt_dag"] is None:
        chk("empty", tc == 0)
        return dict(fails=fails, tc=tc, reanchored=False)
    od = res["opt_dag"]
    # 1. anchor, bound, counts
    chk("opt==opt_dag", abs(res["opt"] - od) <= 1e-12)
    chk("bound==opt_dag*(1+theta)", abs(res["rashomon_bound"] - od * (1 + theta)) <= EPS)
    chk("tc_at_target==tc", int(res["tree_count_at_target"]) == tc)
    chk("tc_at_target_dag==tc", int(res["tree_count_at_target_dag"]) == tc)
    up = res["opt_upfront"]
    chk("anchor_mode", res["anchor_mode"] == "upfront_select_completion" and up is not None)
    chk("upfront_anchor_in_dag", res["upfront_anchor_in_dag"] is True)
    chk("upfront>=opt_dag", up is not None and up >= od - 1e-12)
    chk("reanchored iff opt_dag<upfront",
        res["reanchored"] == (up is not None and od < res["opt_pre_reanchor"] - EPS))
    chk("opt_pre_reanchor==upfront", abs(res["opt_pre_reanchor"] - up) <= 1e-12)
    # 4. Phase 1
    chk("phase1_not_timed_out", res["phase1_timed_out"] is False)
    # 2. oracle over the pool at the enumeration bound
    bound = float(res["rashomon_bound"])
    checked = tc <= KEY_CAP
    if checked:
        oracle = SelectOracle(X, y, inst["thresholds"], depth, inst["max_k"], inst["reg"],
                              float(res["rashomon_bound_enum"]), beam_width=inst["beam"],
                              transition_delta=0.02, atom_mode=True)
        root = frozenset(range(len(y)))
        okeys = [k for _, k in oracle.trees(root, depth, frac_budget(bound))]
        keys, over = _keys_and_checks(res, X, y.astype(int), inst["reg"], bound)
        chk("oracle_count", len(okeys) == tc)
        chk("oracle_keys", set(keys) == set(okeys))
        chk("distinct", len(set(keys)) == tc)
        chk("objective<=bound", over == 0)
        # 3. save / load
        dag = res["dag"]
        loaded = type(dag).from_bytes(dag.to_bytes())
        chk("load_tc", int(loaded.tree_count) == tc)
        chk("load_opt", loaded.opt == dag.opt == res["opt"])
        chk("load_target", abs(loaded.rashomon_bound_target - dag.rashomon_bound_target) <= 1e-15)
        lkeys = [tree_key(dict_to_tree(loaded.get_tree(i))) for i in range(tc)]
        chk("load_keys_ordered", lkeys == keys)
    if tc > 0:
        idx = res["dag"].sample_indices(50, seed=0, at_target=True)
        chk("sample_in_range", all(0 <= int(i) < tc for i in idx))
    return dict(fails=fails, tc=tc, reanchored=bool(res["reanchored"]), checked=checked)


@pytest.mark.parametrize("depth,lookahead", CONFIGS)
def test_reanchor_and_oracle(depth, lookahead):
    rows = [run_instance(inst, depth, lookahead) for inst in small_instances(N_INST)]
    bad = [(k, r["fails"]) for k, r in enumerate(rows) if r["fails"]]
    assert not bad, bad[:10]


@pytest.mark.parametrize("depth,lookahead", FE_CONFIGS)
def test_reanchor_feature_exclusion(depth, lookahead):
    """feature_exclusion=True: the completion shrinks the mask along the path
    (reset at Step-B roots), so the up-front tree is a pool tree and the set
    is the pool's trees within opt_dag*(1+theta)."""
    bad = []
    for k, inst in enumerate(small_instances(N_INST)):
        r = _solve(inst, depth, lookahead, True)
        if r["opt_dag"] is None:
            continue
        ok = (r["upfront_anchor_in_dag"] is True and r["rashomon_complete"] is True
              and abs(r["opt"] - r["opt_dag"]) <= 1e-12
              and int(r["tree_count"]) == int(r["tree_count_at_target_dag"]))
        if not ok:
            bad.append((k, {q: r.get(q) for q in ("opt", "opt_dag", "tree_count",
                                                   "tree_count_at_target_dag",
                                                   "upfront_anchor_in_dag",
                                                   "rashomon_complete")}))
    assert not bad, bad[:5]


def test_reanchor_cases_exercised():
    """Some instance re-anchors (opt_dag strictly below the up-front anchor)."""
    n_re = sum(bool(_solve(inst, d, dl)["reanchored"])
               for d, dl in CONFIGS for inst in small_instances(N_INST))
    assert n_re > 0


@pytest.mark.parametrize("lookahead", [None, 3])
def test_full_depth_reference_check_runs(lookahead):
    """d_l = d (lookahead None or = max_depth, no recursive fill): no up-front
    completion runs (opt_upfront None, anchor_mode 'phase1_prefix'), but the
    DAG minimum is still checked against the first-stage tree's cost."""
    bad = []
    for k, inst in enumerate(small_instances(N_INST)):
        X = np.asarray(inst["X"], dtype=np.float64)
        y = np.asarray(inst["y"], dtype=np.uint8)
        r = solve_rashomon(
            X, y, 2, inst["reg"], 3, inst["max_k"], inst["theta"],
            lookahead_depth=lookahead, min_samples_leaf=1, beam_width=inst["beam"],
            reg_alpha=1.0, transition_delta=0.02, atom_mode=True,
            thresholds_per_feature={f: list(t) for f, t in inst["thresholds"].items()},
            recursive_fill=False, fill_parallel=False)
        assert r["solve_depth"] == 3 and r["opt_upfront"] is None
        assert r["anchor_mode"] == "phase1_prefix"
        if not (r["upfront_anchor_in_dag"] is True and r["rashomon_complete"] is True):
            bad.append((k, r["upfront_anchor_in_dag"], r["rashomon_incomplete_reason"]))
    assert not bad, bad[:5]
