"""Synthetic af tables (interface I2) for the serology tests.

Names are invented and assembled at run time, so no strain-shaped text is committed.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from typing import Any

import pytest

from af.serology.rows import IdentityRules


def virus(place: str, number: int, year: int = 2021, prefix: str = "A(H3N2)") -> str:
    """An invented virus name built from parts."""
    return "/".join([prefix, place, str(number), str(year)])


def designation_identity(
    name: str, reassortant: str, annotations: Sequence[str], fourth: str
) -> tuple[Any, ...] | None:
    """A stand-in for workstream 7's rules, for tests only: DISTINCT or empty -> None."""
    if "DISTINCT" in annotations or not fourth:
        return None
    return (name, reassortant, tuple(annotations), fourth)


TEST_RULES = IdentityRules(
    antigen=designation_identity, serum=designation_identity, version="test-1"
)


def table(
    table_id: str,
    antigens: list[dict[str, Any]],
    sera: list[dict[str, Any]],
    titres: list[list[list[str]]],
    *,
    date: str = "2021-03-04",
    lab: str = "LABX",
    group: str = "h3-hi-labx",
    subtype: str = "A(H3N2)",
    date_suffix: int = 1,
) -> dict[str, Any]:
    body = {
        "format": "af-table-1",
        "table_id": table_id,
        "group": group,
        "lab": lab,
        "subtype": subtype,
        "lineage": "",
        "assay": "HI",
        "rbc": "turkey",
        "date": date,
        "date_suffix": date_suffix,
        "antigens": antigens,
        "sera": sera,
        "titres": titres,
    }
    body["content_hash"] = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
    return body


class Synthetic:
    """The helpers above, handed to tests as the ``syn`` fixture (conftest modules are not
    importable by name here: pytest runs in importlib mode)."""

    virus = staticmethod(virus)
    table = staticmethod(table)
    rules = TEST_RULES


@pytest.fixture
def syn() -> Synthetic:
    return Synthetic()
