"""The WHO-data gate's allowlist lets placeholders through and nothing that could be real.

Placeholders use the reserved EXAMPLE namespace (tools/who-data-gate-allowlist.txt). Every
widening of that allowlist must keep the second list flagged: each of those strings could
carry a real place, lab code or isolate. Add probes here when the allowlist changes.
"""

import importlib.util
import sys
from types import ModuleType

import pytest

from tests.helpers import REPO_ROOT


def strain(*parts: str) -> str:
    """Join at runtime so this file contains no strain-shaped text for the gate to flag."""
    return "/".join(parts)


ALLOWED = [
    strain("A", "EXAMPLETOWN", "1", "2020"),
    strain("B", "BEXAMPLEVILLE", "7", "17"),
    strain("A", "EXAMPLETOWN_EXAMPLEISLES", "EXAMPLE_6", "2026"),
    strain("A", "EXAMPLE-TOWN", "1", "2020"),
    strain("A", "EXAMPLE TOWN", "1", "2020"),
    strain("a", "example-town", "1", "2020"),
    strain("A", "Example_City", "12", "2026"),
    strain("EXAMPLE-TOWN", "3", "2021"),
    strain("A", "EXAMPLETOWN-EXAMPLEISLES", "7", "2021"),
    strain("A", "EXAMPLETOWN EXAMPLEISLES", "7", "2021"),
    strain("B", "EXAMPLETOWN", "EXAMPLELAB-23", "2021"),
    strain("A", "EXAMPLETOWN", "EXAMPLELAB-EXAMPLEUNIT-1", "2025"),
    strain("A", "EXAMPLETOWN", "0EXAMPLELAB-7", "2025"),
]

# Invented places and ids in real-looking shapes. None is a real strain.
FLAGGED = [
    strain("A", "EXAMPLE FAKE HARBOUR", "45", "2019"),  # a place name after the prefix
    strain("A", "EXAMPLE-FAKEPORT", "6", "2021"),
    strain("A", "FAKE HARBOUR", "45", "2019"),
    strain("A", "TOWN", "1", "2020"),
    strain("A", "SOMEPLACE-TOWN", "1", "2020"),
    strain("A", "CITYEXAMPLE", "1", "2020"),
    strain("A", "EXAMPLETOWN", "1", "2020 A") + "/" + strain("FAKEPORT", "6", "2021"),
    strain("A", "EXAMPLETOWN", "LABX-23", "2021"),  # a lab code in the isolate
    strain("A", "EXAMPLETOWN", "QZ00000-00", "2021"),  # letter-initial isolate formats
    strain("A", "EXAMPLETOWN", "QZX-QZY-0000", "2021"),
    strain("A", "EXAMPLETOWN", "00Q-0000", "2021"),
    strain("A", "EXAMPLETOWN", "EXAMPLELAB-QZX00000", "2021"),
]


@pytest.fixture(scope="module")
def gate_config() -> object:
    spec = importlib.util.spec_from_file_location(
        "who_data_gate", REPO_ROOT / "tools" / "who-data-gate.py"
    )
    assert spec and spec.loader
    module: ModuleType = importlib.util.module_from_spec(spec)
    sys.modules["who_data_gate"] = module
    spec.loader.exec_module(module)
    config = module.Config()
    module.load_allowlist(str(REPO_ROOT / "tools" / "who-data-gate-allowlist.txt"), config)
    return config


@pytest.mark.parametrize("token", ALLOWED)
def test_placeholder_allowed(gate_config: object, token: str) -> None:
    assert gate_config.allowed(token)  # type: ignore[attr-defined]


@pytest.mark.parametrize("token", FLAGGED)
def test_possibly_real_stays_flagged(gate_config: object, token: str) -> None:
    assert not gate_config.allowed(token)  # type: ignore[attr-defined]
