"""The package layout imports cleanly and exposes a version."""

import importlib

import pytest

import af

SUBPACKAGES = [
    "af.util",
    "af.run",
    "af.store",
    "af.pipeline",
    "af.seq",
    "af.tables",
    "af.clades",
    "af.tree",
    "af.tree.build",
    "af.tree.asr",
    "af.tree.draw",
    "af.chart",
    "af.chain",
    "af.map",
    "af.serology",
    "af.geo",
    "af.stat",
    "af.report",
]


def test_version() -> None:
    assert af.__version__


@pytest.mark.parametrize("name", SUBPACKAGES)
def test_subpackage_imports(name: str) -> None:
    module = importlib.import_module(name)
    assert module.__doc__, f"{name} needs a one-line docstring"
