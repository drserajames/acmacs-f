"""The package layout imports cleanly and exposes a version."""

import importlib

import pytest

import af

SUBPACKAGES = [
    "af.util",
    "af.run",
    "af.seq",
    "af.tree",
    "af.tree.build",
    "af.tree.io",
    "af.tree.draw",
    "af.clades",
    "af.chart",
    "af.serology",
    "af.map",
]


def test_version() -> None:
    assert af.__version__


@pytest.mark.parametrize("name", SUBPACKAGES)
def test_subpackage_imports(name: str) -> None:
    module = importlib.import_module(name)
    assert module.__doc__, f"{name} needs a one-line docstring"
