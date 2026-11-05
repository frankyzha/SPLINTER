"""Shared test helpers: brute-force Rashomon counts, the Select-pool oracle,
tiny random instances, tree keys and objectives, and the edge-label parser.

Imported by the test modules as `from _helpers import ...` (pytest puts
`tests/` on `sys.path`). Never imports a test module.
"""
import collections
import itertools
import operator
import os
import random
import re
from fractions import Fraction as Fr

import numpy as np

from splinter.multiway_solver import rashomon_select_debug, solve_rashomon
from splinter.tree import MWLeaf, MWNode, MWShapeNode

EPS = 1e-9
KEY_CAP = 2000  # sets larger than this are compared on an index sample
N_ORACLE_INST = int(os.environ.get("RASH_ORACLE_N", "30"))

# The 5-row toy: 6 trees of depth <= 2 within theta = 0.5 over every threshold.
TOY = dict(X=[[2, 2], [0, 1], [0, 0], [1, 1], [1, 2]], y=[1, 1, 0, 0, 1],
           thresholds={0: [0.5, 1.5], 1: [0.5, 1.5]}, depth=2, max_k=3, reg=0.05, theta=0.5)


# ------------------------------------------------------------ brute force
def brute_schedule(X, y, thresholds, depth, max_k, reg, by_partition=False):
    """{exact cost: number of trees} over all trees of depth <= `depth`.
    by_partition=True keeps one threshold set per induced partition of the
    support (the rashomon_core.dataset_dag model) instead of every set."""
    n = len(y)
    reg = Fr(reg).limit_denominator(10**6)
    memo = {}

    def leaf_cost(sup):
        cnt = collections.Counter(y[i] for i in sup)
        return Fr(len(sup) - max(cnt.values()), n) + reg

    def sched(sup, d):
        key = (sup, d)
        if key in memo:
            return memo[key]
        out = collections.Counter({leaf_cost(sup): 1})
        if d > 0 and len(sup) >= 2:
            for f, ths in thresholds.items():
                seen_parts = set()
                for r in range(1, max_k):
                    for cut in itertools.combinations(sorted(ths), r):
                        bins = collections.defaultdict(list)
                        for i in sup:
                            bins[int(np.searchsorted(cut, X[i][f], side="right"))].append(i)
                        if len(bins) != r + 1:
                            continue  # an empty child: not a valid split
                        part = tuple(frozenset(bins[b]) for b in range(r + 1))
                        if by_partition and part in seen_parts:
                            continue
                        seen_parts.add(part)
                        acc = collections.Counter({Fr(0): 1})
                        for b in range(r + 1):
                            cs = sched(frozenset(bins[b]), d - 1)
                            nxt = collections.Counter()
                            for a, na in acc.items():
                                for c, nc in cs.items():
                                    nxt[a + c] += na * nc
                            acc = nxt
                        out.update(acc)
        memo[key] = out
        return out

    return sched(frozenset(range(n)), depth)


# ------------------------------------------------------------ trees
def tree_key(t):
    """Exact-structure key of a tree (feature, threshold bits, group map)."""
    if isinstance(t, MWLeaf):
        return ("L", int(t.prediction))
    gm = tuple(int(g) for g in t.group_map) if isinstance(t, MWShapeNode) else None
    return (int(t.feature_index), tuple(float(x).hex() for x in t.thresholds), gm,
            tuple(tree_key(c) for c in t.children))


def tree_objective(t, X, y, reg, idx=None):
    """misclassified/n + reg per leaf, recomputed by routing the rows."""
    idx = np.arange(len(y)) if idx is None else idx
    if isinstance(t, MWLeaf):
        return int((y[idx] != t.prediction).sum()) / len(y) + reg
    b = np.searchsorted(np.asarray(t.thresholds), X[idx, t.feature_index], side="right")
    return sum(tree_objective(c, X, y, reg, idx[b == j]) for j, c in enumerate(t.children))


# ------------------------------------------------------------ Select-pool oracle
class SelectOracle:
    """Brute-force Rashomon set over the solver's own Select pool:

        R_B(sup, d) = {leaf} ∪ ⋃_{c ∈ Select(sup)} ⨂_i R_B(child_i(c), d-1)

    with exact Fraction costs (misclassified/n + reg per leaf; split_reg = 0).
    Select(sup) comes from `rashomon_select_debug` with ub_hint = beta."""

    def __init__(self, X, y, thresholds, depth, max_k, reg, beta, **select_kw):
        self.X = np.asarray(X, dtype=np.float64)
        self.y = np.asarray(y, dtype=np.uint8)
        self.n = len(y)
        self.thresholds = {f: list(t) for f, t in thresholds.items()}
        self.depth, self.max_k, self.beta = depth, max_k, beta
        self.reg = Fr(reg).limit_denominator(10**6)
        self.reg_f = reg
        self.select_kw = select_kw
        self._select = {}
        self._sched = {}

    def select(self, sup):
        # The pool can hold the same split twice (identical feature,
        # thresholds, group_map from two DP passes); the solver keeps one
        # record per split identity, so the oracle dedups the same way.
        if sup not in self._select:
            pool = rashomon_select_debug(
                self.X, self.y, 2, self.reg_f, self.depth, self.max_k, self.beta,
                sorted(sup), thresholds_per_feature=self.thresholds, reg_alpha=1.0,
                **self.select_kw)
            seen, uniq = set(), []
            for c in pool:
                ident = (c["feature"], tuple(float(t).hex() for t in c["thresholds"]),
                         None if c["group_map"] is None else tuple(c["group_map"]))
                if ident not in seen:
                    seen.add(ident)
                    uniq.append(c)
            self._select[sup] = uniq
        return self._select[sup]

    def children(self, sup, c):
        idx = np.array(sorted(sup), dtype=int)
        b = np.searchsorted(np.asarray(c["thresholds"]), self.X[idx, c["feature"]], side="right")
        if c["group_map"] is not None:
            b = np.asarray(c["group_map"])[b]
        return [frozenset(idx[b == g].tolist()) for g in range(c["k"])]

    def leaf(self, sup):
        cnt = [0, 0]
        for i in sup:
            cnt[int(self.y[i])] += 1
        pred = max(range(2), key=lambda c: (cnt[c], c))  # Rust max_by_key: last max
        return Fr(len(sup) - max(cnt), self.n) + self.reg, ("L", pred)

    def splits(self, sup, d):
        if d <= 0 or len(sup) <= 1:
            return []
        out = []
        for c in self.select(sup):
            ch = self.children(sup, c)
            if sum(1 for s in ch if s) <= 1:
                continue
            out.append((c, ch))
        return out

    def sched(self, sup, d):
        """{cost: count} of R_B(sup, d)."""
        key = (sup, d)
        if key not in self._sched:
            cost, _ = self.leaf(sup)
            out = collections.Counter({cost: 1})
            for c, ch in self.splits(sup, d):
                acc = collections.Counter({Fr(0): 1})
                for s in ch:
                    cs = self.sched(s, d - 1)
                    nxt = collections.Counter()
                    for a, na in acc.items():
                        for b, nb in cs.items():
                            nxt[a + b] += na * nb
                    acc = nxt
                out.update(acc)
            self._sched[key] = out
        return self._sched[key]

    def min_cost(self, sup, d):
        return min(self.sched(sup, d))

    def trees(self, sup, d, budget):
        """[(cost, key)] of R_B(sup, d) with cost <= budget (exact)."""
        cost, lk = self.leaf(sup)
        out = [(cost, lk)] if cost <= budget else []
        for c, ch in self.splits(sup, d):
            mins = [self.min_cost(s, d - 1) for s in ch]
            if sum(mins) > budget:
                continue
            partial = [(Fr(0), ())]
            for i, s in enumerate(ch):
                rest = sum(mins[i + 1:])
                nxt = []
                for a, keys in partial:
                    for b, k in self.trees(s, d - 1, budget - a - rest):
                        nxt.append((a + b, keys + (k,)))
                partial = nxt
            th = tuple(float(t).hex() for t in c["thresholds"])
            gm = None if c["group_map"] is None else tuple(int(g) for g in c["group_map"])
            for a, keys in partial:
                out.append((a, (int(c["feature"]), th, gm, keys)))
        return out


def frac_budget(beta):
    return Fr(beta + EPS)


# ------------------------------------------------------------ instances
def random_instances(n_inst=N_ORACLE_INST, seed=0):
    """n <= 40, p <= 4, 3-8 thresholds/feature, d in {2,3}, k in {2,3},
    B in {1,4,16}."""
    rng = random.Random(seed)
    out = []
    while len(out) < n_inst:
        n, p = rng.randint(6, 40), rng.randint(2, 4)
        vmax = rng.randint(3, 8)
        X = [[rng.randint(0, vmax) for _ in range(p)] for _ in range(n)]
        y = [rng.randint(0, 1) for _ in range(n)]
        if len(set(y)) < 2:
            continue
        mids = [v + 0.5 for v in range(vmax)]
        ths = {f: sorted(rng.sample(mids, rng.randint(min(3, len(mids)), min(8, len(mids)))))
               for f in range(p)}
        out.append(dict(X=X, y=y, thresholds=ths, depth=rng.choice([2, 3]),
                        max_k=rng.choice([2, 3]), reg=rng.choice([0.02, 0.05]),
                        theta=rng.choice([0.02, 0.05, 0.1]), beam=rng.choice([1, 4, 16])))
    return out


_SMALL = {}


def small_instances(n_inst, n_max=20, seed=0):
    """The first n_inst random_instances (seed) with n <= n_max (cached)."""
    key = (n_inst, n_max, seed)
    if key not in _SMALL:
        out = [i for i in random_instances(20 * n_inst, seed) if len(i["y"]) <= n_max]
        assert len(out) >= n_inst
        _SMALL[key] = out[:n_inst]
    return _SMALL[key]


def solve_instance(inst, depth, lookahead, fill_parallel=False):
    """solve_rashomon on an instance at the oracle's settings (closure off,
    reg_alpha = 1, transition_delta = 0.02, atom merge); the recursive fill
    runs whenever lookahead < depth."""
    X = np.asarray(inst["X"], dtype=np.float64)
    y = np.asarray(inst["y"], dtype=np.uint8)
    return solve_rashomon(
        X, y, 2, inst["reg"], depth, inst["max_k"], inst["theta"],
        lookahead_depth=lookahead, min_samples_leaf=1, beam_width=inst["beam"],
        reg_alpha=1.0, transition_delta=0.02, atom_mode=True,
        thresholds_per_feature={f: list(t) for f, t in inst["thresholds"].items()},
        recursive_fill=lookahead is not None, fill_parallel=fill_parallel)


def tiny_data(n=120, seed=0, p=3, labels=None):
    """`p` integer features in 0..3; the label is x0 >= 2 with 15% flipped.
    labels=(negative, positive) names the classes (default 0 / 1)."""
    rng = np.random.default_rng(seed)
    X = rng.integers(0, 4, size=(n, p)).astype(np.float64)
    y = (X[:, 0] >= 2) ^ (rng.random(n) < 0.15)
    return X, (y.astype(int) if labels is None else np.where(y, labels[1], labels[0]))


# ------------------------------------------------------------ edge labels and random trees
ZERO_SENTINEL = 1.0000000180025095e-35  # LightGBM's cutpoint at zero
_COMPARE = {"≤": operator.le, "<": operator.lt, "≥": operator.ge, ">": operator.gt}


def _piece(text):
    if text == "any":
        return lambda x: np.ones(len(x), dtype=bool)
    if text[0] in _COMPARE:
        compare, bound = _COMPARE[text[0]], float(text[2:])
        return lambda x: compare(x, bound)
    match = re.fullmatch(r"([\[(])(\S+), (\S+)([\])])", text)
    assert match, text
    low = operator.ge if match[1] == "[" else operator.gt
    high = operator.le if match[4] == "]" else operator.lt
    a, b = float(match[2]), float(match[3])
    return lambda x: low(x, a) & high(x, b)


def predicate(label):
    """Turn an edge label into a vectorized membership test on x."""
    pieces = [_piece(part) for part in label.split(" ∪ ")]
    return lambda x: np.logical_or.reduce([piece(x) for piece in pieces])


def random_tree(rng, depth):
    """A random tree of contiguous and nonlinear (group-map) nodes over 3 features."""
    if depth == 0 or rng.random() < 0.25:
        return MWLeaf(int(rng.integers(3)), int(rng.integers(1, 20)), float(rng.integers(0, 5)))
    feature, k = int(rng.integers(3)), int(rng.integers(2, 6))
    children = [random_tree(rng, depth - 1) for _ in range(k)]
    grid = np.arange(-5.0, 5.5, 0.5)
    if rng.random() < 0.5:
        return MWNode(feature, f"x{feature}", sorted(rng.choice(grid, k - 1, replace=False).tolist()), children)
    n_bins = int(rng.integers(k, k + 4))
    group_map = rng.permutation(np.r_[np.arange(k), rng.integers(k, size=n_bins - k)]).tolist()
    thresholds = sorted(rng.choice(grid, n_bins - 1, replace=False).tolist())
    return MWShapeNode(feature, f"x{feature}", thresholds, group_map, children)


def thresholds_of(tree):
    if isinstance(tree, MWLeaf):
        return []
    return list(tree.thresholds) + [t for child in tree.children for t in thresholds_of(child)]
