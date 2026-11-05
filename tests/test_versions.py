"""The release version is the same in pyproject.toml, Cargo.toml,
CITATION.cff and the installed package."""
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _toml_version(name, table):
    """`version` of `[table]` (regex: tomllib needs Python >= 3.11)."""
    text = (ROOT / name).read_text()
    section = re.search(rf"^\[{re.escape(table)}\]\n(.*?)(?=^\[|\Z)", text, re.M | re.S)
    assert section, f"{name} has no [{table}] table"
    match = re.search(r"^version\s*=\s*\"([^\"]+)\"", section[1], re.M)
    assert match, f"{name} [{table}] has no version"
    return match[1]


def _citation_version():
    match = re.search(r"^version:\s*['\"]?([^'\"\s]+)", (ROOT / "CITATION.cff").read_text(), re.M)
    assert match, "CITATION.cff has no version field"
    return match[1]


def test_pyproject_cargo_and_citation_versions_agree():
    versions = (_toml_version("pyproject.toml", "project"),
                _toml_version("Cargo.toml", "package"), _citation_version())
    assert len(set(versions)) == 1, versions


def test_installed_package_version_matches():
    splinter = pytest.importorskip("splinter")
    assert splinter.__version__ == _toml_version("pyproject.toml", "project")
