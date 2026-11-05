"""Every public constructor parameter is documented in the class docstring,
and nothing else is: the ``Parameters`` and ``Advanced parameters`` sections
list exactly the ``__init__`` parameters.
"""
import inspect
import re

import pytest

from splinter import SPLINTER, ShapeSPLINTER, LicketySPLINTER, RESPLINTER

SECTIONS = ("Parameters", "Advanced parameters")


def documented_parameters(cls):
    """Names of the ``name : type`` entries in the parameter sections."""
    lines = inspect.getdoc(cls).splitlines()
    names, section = [], None
    for i, line in enumerate(lines):
        underline = i + 1 < len(lines) and re.fullmatch(r"-{3,}", lines[i + 1])
        if underline:
            section = line.strip()
            continue
        if section in SECTIONS:
            match = re.match(r"^(\w+) :", line)
            if match:
                names.append(match.group(1))
    return names


@pytest.mark.parametrize("cls", [SPLINTER, ShapeSPLINTER, LicketySPLINTER, RESPLINTER])
def test_signature_matches_documented_parameters(cls):
    signature = [name for name in inspect.signature(cls.__init__).parameters if name != "self"]
    documented = documented_parameters(cls)
    assert len(documented) == len(set(documented)), documented
    assert set(documented) == set(signature)


@pytest.mark.parametrize("cls", [SPLINTER, ShapeSPLINTER, LicketySPLINTER, RESPLINTER])
def test_docstring_has_the_numpydoc_sections(cls):
    doc = inspect.getdoc(cls)
    for section in ("Parameters", "Advanced parameters", "Attributes", "See Also", "Examples"):
        assert f"\n{section}\n{'-' * len(section)}\n" in doc, section
    assert "teacher" not in doc.lower()
