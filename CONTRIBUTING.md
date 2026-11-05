# Contributing

Bug reports, questions and pull requests are welcome on the issue tracker.

## Development setup

You need Python 3.10–3.12, a [Rust toolchain](https://rustup.rs/) (1.80 or
newer), [uv](https://docs.astral.sh/uv/), and on macOS `brew install libomp`.
From the repository root:

```bash
uv sync --locked --all-extras --no-install-project   # dependencies, with dev and visualization
.venv/bin/maturin develop --release --locked         # build the package and its Rust extension into .venv
```

Rebuild with `maturin develop --release` after every change under `src/`; the
Python files under `python/splinter/` are used in place.

## Tests

```bash
.venv/bin/python -m pytest                       # Python tests
cargo test --release --no-default-features       # Rust tests, without the Python bindings
```

The examples are run in CI as well; run one with
`.venv/bin/python examples/classification.py`.

## Documentation

The [README](README.md) covers installation and core usage. Keep its examples
and the API docstrings in step when changing parameters or defaults.
`tests/test_docstrings.py` checks that every constructor parameter is
documented.

## Code style

- Rust: `cargo fmt`; `cargo clippy --release --no-default-features` should not
  gain new warnings.
- Python: `ruff check python tests examples` (settings in `pyproject.toml`).
- Docstrings use the NumPy style. Public names follow scikit-learn: constructor
  parameters are stored unchanged, and fitted attributes end in `_`.
- New behavior comes with a test. Add an entry to `CHANGELOG.md` for changes
  users will notice.
