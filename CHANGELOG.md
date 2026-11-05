# Changelog

All notable changes to this project are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/).

## [0.1.0] - 2026-11-04

First public release, accompanying the NeurIPS 2026 paper *Interpretable
Multiway-Split Trees*.

### Fixed

- Load earlier format-1 `RESPLINTER` files after the fitted-attribute renames,
  preserving their saved search settings and reports.

### Added

- `SPLINTER`: decision trees with multiway splits into contiguous ranges of a
  feature. The top levels are found by a branch-and-bound search over the
  best-ranked candidate splits; the levels below are completed greedily.
- `ShapeSPLINTER`: like `SPLINTER`, but one branch can also hold non-adjacent
  ranges of a feature (nonlinear splits).
- `LicketySPLINTER`: grows the tree one level at a time and scores each split
  by a greedy completion of its subtree; faster on deeper trees.
- `RESPLINTER`: binary Rashomon sets of near-optimal trees over the candidate
  splits kept by the search, stored compactly for counting, sampling, and
  prediction. Search completeness and best-tree optimality are reported
  separately.
- A scikit-learn interface for all four estimators (`fit`, `predict`,
  `predict_proba`, `score`, parameter cloning), integer-coded categorical
  features, and sample weights for `SPLINTER` and `ShapeSPLINTER`.
- Candidate thresholds from a LightGBM model, with a cache that can be saved
  and reused (`splinter.thresholds`).
- Tree output as text and per-row explanations (`export_text`, `explain`),
  Graphviz tree drawings and plots (`splinter.viz`).
- Tools for Rashomon sets: a table of all trees with filtering
  (`splinter.explore`), prediction multiplicity, and variable importance across
  all good trees (RID, `splinter.rid`).
- Saving and loading of fitted `RESPLINTER` sets as NumPy `.npz` files that load
  without pickle.
- Type information (`py.typed` and a stub for the compiled extension).
