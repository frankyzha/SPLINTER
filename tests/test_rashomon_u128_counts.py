"""u128 tree counts through the PyO3 API, on the chain fixture
(`PyRashomonDag._count_chain_fixture`, `rashomon_dag::count_chain_fixture`):
vertex v_i has a leaf and one binary split with both children v_{i-1}, so
c_0 = 1 and c_i = 1 + c_{i-1}^2 trees (c_7 ~ 4.4e22 > 2^64, c_8 > 2^128).

  (a) levels=7: tree_count / tree_count_u128 / root_histogram are exact
      Python ints above 2^64; tree_count_u64 raises OverflowError;
      get_tree takes indices above 2^64 and gives distinct trees; an index
      >= tree_count raises IndexError; v4 / v5 round trips keep the counts
      and the trees;
  (b) levels=8: the count saturates at u128::MAX, count_saturated is set,
      and get_tree / sample_indices / sample_trees raise OverflowError.
"""
import numpy as np
import pytest

from splinter.multiway_solver import PyRashomonDag

U64_MAX = 2**64 - 1
U128_MAX = 2**128 - 1


def chain_count(levels):
    c = 1
    for _ in range(levels):
        c = c * c + 1
    return c


def sig(t):
    if t["type"] == "leaf":
        return f"L{t['prediction']}"
    return "(" + ",".join(sig(c) for c in t["children"]) + ")"


def test_counts_and_indexing_above_u64():
    dag = PyRashomonDag._count_chain_fixture(7)
    c7 = chain_count(7)
    assert c7 > U64_MAX
    assert int(dag.tree_count) == c7
    assert dag.tree_count_u128 == c7
    assert dag.count_saturated is False
    with pytest.raises(OverflowError):
        dag.tree_count_u64
    hist = dag.root_histogram()
    assert sum(m for _, m in hist) == c7
    assert max(m for _, m in hist) > U64_MAX

    idxs = [0, 1, 2**64, 2**64 + 1, 2**64 + 12345, c7 - 2, c7 - 1]
    sigs = {sig(dag.get_tree(i)) for i in idxs}
    assert len(sigs) == len(idxs)
    assert sig(dag.get_tree(np.int64(3))) == sig(dag.get_tree(3))
    with pytest.raises(IndexError):
        dag.get_tree(c7)

    for version in (None, 4, 5, 6):
        loaded = PyRashomonDag.from_bytes(dag.to_bytes(version))
        assert loaded.tree_count_u128 == c7
        assert loaded.count_saturated is False
        for i in idxs:
            assert loaded.get_tree(i) == dag.get_tree(i)

    s = dag.sample_indices(16, seed=0)
    assert all(0 <= i < c7 for i in s)
    assert any(i > U64_MAX for i in s)


def test_saturated_count_raises():
    dag = PyRashomonDag._count_chain_fixture(8)
    assert chain_count(8) > U128_MAX
    assert dag.count_saturated is True
    assert dag.tree_count_u128 == U128_MAX
    with pytest.raises(OverflowError):
        dag.get_tree(0)
    with pytest.raises(OverflowError):
        dag.sample_indices(1, seed=0)
    with pytest.raises(OverflowError):
        dag.sample_trees(1, seed=0)
    loaded = PyRashomonDag.from_bytes(dag.to_bytes())
    assert loaded.count_saturated is True
