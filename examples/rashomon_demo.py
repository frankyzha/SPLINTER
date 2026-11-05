"""Rashomon set, prediction multiplicity, variable importance and a per-tree table on planted toy data."""
import argparse
import time
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from splinter import RESPLINTER
from splinter.explore import rashomon_table, select
from splinter.rid import compute_rid
from splinter.viz import (multiplicity_summary, plot_multiplicity, plot_rid, plot_rid_cdf,
                          plot_tradeoff, plot_tree)

FEATURES = ["dose", "age", "dose_noisy", "noise1", "noise2"]
SETTINGS = dict(max_depth=2, lookahead_depth=1, lgbm_min_samples_leaf=5, beam_width=4,
                reg=0.004, theta=0.08, lgbm_n_estimators=32, cart_closure=False)


def toy_data(n=400, seed=0):
    """Label 1 for a middle dose band (30 to 70) below age 60; 5% of labels flipped."""
    rng = np.random.default_rng(seed)
    dose, age = rng.integers(0, 101, n).astype(float), rng.integers(20, 81, n).astype(float)
    dose_noisy = np.round(dose + rng.normal(0, 8, n))
    X = np.column_stack([dose, age, dose_noisy, rng.uniform(size=n), rng.uniform(size=n)])
    y = ((dose >= 30) & (dose < 70) & (age < 60)).astype(int)
    y ^= rng.random(n) < 0.05
    return X, y


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", default="tree_figures")
    outdir = Path(parser.parse_args().outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()

    # 1. Planted toy data.
    X, y = toy_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

    # 2. Fit the Rashomon set.
    rset = RESPLINTER(**SETTINGS).fit(X_train, y_train, feature_names=FEATURES)
    print(rset.summary())

    # 3. The best tree and one sampled tree.
    _, (other,) = rset.sample_trees(1, seed=1, at_target=True, return_indices=True)
    for idx in (0, other):
        dot = plot_tree(rset.get_tree(idx), X_test, y_test, classes=rset.classes_,
                        title=f"Tree {idx}")
        for fmt in ("png", "pdf"):
            print(dot.render(outdir / f"rashomon_tree_{idx}", format=fmt, cleanup=True))

    # 4. Prediction multiplicity on the test rows.
    summary = multiplicity_summary(rset, X_test)
    print(f"{summary['n_trees']} trees; conflicting predictions on "
          f"{summary['share_conflicting']:.1%} of test rows")
    figures = {"rashomon_multiplicity": plot_multiplicity(rset, X_test)}

    # 5. Rashomon importance distribution over 10 bootstraps.
    rid = compute_rid(RESPLINTER(**SETTINGS), X_train, y_train, B=10)
    ranking = sorted(range(len(FEATURES)), key=lambda j: -rid.mean(j))
    print("RID mean importance:", ", ".join(f"{FEATURES[j]} {rid.mean(j):.3f}" for j in ranking))
    figures["rashomon_rid"] = plot_rid(rid, FEATURES)
    figures["rashomon_rid_cdf"] = plot_rid_cdf(rid, FEATURES)

    # 6. Inspect the set: one row per tree; which features does every tree need,
    #    and how good are the trees that split on pure noise?
    table = rashomon_table(rset, X_test, y_test)
    print(f"table: {len(table)} rows ({table.attrs['mode']}) of {table.attrs['n_trees']} trees")
    print(table.head().to_string())
    print(table.features.value_counts().head().to_string())
    for name in ("dose", "age"):
        print(f"trees without {name}: {len(select(table, avoids=[name]))}")
    figures["rashomon_tradeoff"] = plot_tradeoff(table)
    simple = select(table, acc=(0.93, None), aes=(None, 1.7), max_arity=(3, None))
    print(f"accuracy >= 0.93, AES <= 1.7, a split with >= 3 branches: {len(simple)} trees")
    noisy = select(table, uses="noise1")
    print(f"{len(noisy)} of {len(table)} trees split on noise1")
    if noisy.empty:
        raise RuntimeError("no tree splits on noise1; adjust SETTINGS")
    figures["rashomon_tradeoff_selected"] = plot_tradeoff(table, highlight=noisy)
    best = noisy.loc[noisy.objective.idxmin()]
    dot = plot_tree(rset.get_tree(int(best.tree)), X_test, y_test, classes=rset.classes_,
                    title=f"Tree {best.tree}: accuracy {best.acc:.2f}, AES {best.aes:.2f}")
    for fmt in ("png", "pdf"):
        print(dot.render(outdir / "rashomon_tree_selected", format=fmt, cleanup=True))
    # Trees disagree on a test row: which predict each class for it?
    i = int(np.argmin(np.abs(summary["p"] - 0.5)))  # the most evenly split row
    yes = select(table, predicts={i: rset.classes_[1]})
    no = select(table, predicts={i: rset.classes_[0]})
    print(f"test row {i}: {len(yes)} trees predict {rset.classes_[1]}, "
          f"{len(no)} predict {rset.classes_[0]} (of {len(table)})")
    if len(yes) + len(no) != len(table):
        raise RuntimeError("the two predictions should split the table")
    figures["rashomon_tradeoff_disputed"] = plot_tradeoff(table, highlight=yes)

    for name, fig in figures.items():
        for fmt in ("png", "pdf"):
            fig.savefig(outdir / f"{name}.{fmt}", dpi=200)
            print(outdir / f"{name}.{fmt}")
    print(f"done in {time.perf_counter() - start:.1f} s")


if __name__ == "__main__":
    main()
