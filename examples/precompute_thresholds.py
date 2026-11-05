"""Fit LightGBM once, then reuse its threshold cache for several trees.

Pass --cache thresholds.npz to save it and reuse it in later processes.
"""

import argparse
from pathlib import Path
from time import perf_counter

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from splinter import SPLINTER, ShapeSPLINTER, LicketySPLINTER
from splinter.thresholds import fit_thresholds, load_threshold_cache, save_threshold_cache


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path,
                        help="Load this cache if it exists; otherwise fit LightGBM and save it here.")
    args = parser.parse_args(argv)
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target_names[data.target], test_size=0.25,
        stratify=data.target, random_state=42,
    )

    # Fit only on training data. The full cache also holds the row-aligned
    # LightGBM scores that ShapeSPLINTER uses to order feature bins.
    if args.cache is not None and args.cache.exists():
        started = perf_counter()
        cache = load_threshold_cache(args.cache)
        print(f"Loaded threshold cache from {args.cache} in {perf_counter() - started:.3f}s; "
              "LightGBM fitting skipped.")
    else:
        started = perf_counter()
        cache = fit_thresholds(X_train, y_train, n_estimators=16,
                               random_state=42, num_threads=1)
        reference_seconds = perf_counter() - started
        print(f"LightGBM reference fitted in {reference_seconds:.3f}s")
        if args.cache is not None:
            save_threshold_cache(cache, args.cache)
            print(f"Saved threshold cache to {args.cache}")
    n_thresholds = sum(len(edges) for edges in cache.thresholds.values())
    print(f"{n_thresholds} candidate thresholds; one cache for every tree fit below.\n")

    # Every run uses the same training rows and column order, so both
    # the thresholds and the cached LightGBM scores can be reused.
    print(f"{'model':<18} {'depth cap':>9} {'reg':>8} {'tree fit (s)':>12} "
          f"{'test accuracy':>13} {'leaves':>7} {'depth':>6}")
    for estimator in (SPLINTER, ShapeSPLINTER, LicketySPLINTER):
        for depth, reg in ((2, 0.01), (3, 0.001)):
            model = estimator(
                max_depth=depth, reg=reg, min_samples_leaf=5,
                precomputed_thresholds=cache,
            )
            started = perf_counter()
            model.fit(X_train, y_train, feature_names=data.feature_names)
            tree_seconds = perf_counter() - started
            assert model.lgbm_time_ == 0.0  # No reference-model fit in this run.
            print(f"{estimator.__name__:<18} {depth:>9} {reg:>8.3f} {tree_seconds:>12.3f} "
                  f"{model.score(X_test, y_test):>13.3f} {model.num_leaves():>7} "
                  f"{model.tree_depth():>6}")


if __name__ == "__main__":
    main()
