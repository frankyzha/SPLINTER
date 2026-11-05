"""get_tree on a DAG file without integer cost keys (`keyed_costs` False).

`data/dag_unkeyed_fixture.bin` is a v6 file (`to_bytes(6)`) written by the
pre-cleanup build (SPLINTER-release) under its former RASH_KEYSPACE=0
switch, which produced unkeyed DAGs: the 6-row fit below, 4 trees. Extraction used
to match child-cost combinations by the key of the float sum of the child
costs while counting adds per-child keys, so index 2 could not be decoded.
Every counted tree must now extract, be distinct, and lie within the bound.
"""
import os

import numpy as np

from splinter.multiway_solver import PyRashomonDag
from splinter.resplinter import dict_to_tree

from _helpers import tree_key, tree_objective

FIXTURE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data",
                       "dag_unkeyed_fixture.bin")
# The fit that wrote the fixture: solve_rashomon(X, y, 2, REG, 3, 2, 1.0,
# thresholds_per_feature={0: [0.5, 1.5], 1: [0.5, 1.5]}, min_samples_leaf=1).
X = np.array([[0, 0], [2, 2], [2, 0], [2, 0], [2, 2], [0, 0]], dtype=np.float64)
Y = np.array([0, 1, 0, 1, 1, 1])
REG = 0.05


def test_unkeyed_dag_extracts_every_counted_tree():
    dag = PyRashomonDag.load(FIXTURE, ["a", "b"])
    assert not dag.keyed_costs
    count = int(dag.tree_count)
    assert count == 4
    trees = [dict_to_tree(dag.get_tree(i)) for i in range(count)]
    assert len({tree_key(t) for t in trees}) == count
    for t in trees:
        assert tree_objective(t, X, Y, REG) <= dag.rashomon_bound + 1e-9
