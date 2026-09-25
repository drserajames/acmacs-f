"""Synthetic af tables (interface I2) for the serology tests.

Names are invented and assembled at run time, so no strain-shaped text is committed.
"""

from __future__ import annotations

from typing import Any

import pytest

from af.serology.rows import IdentityRules
from af.tables.model import Antigen, Serum, Table


def virus(place: str, number: int, year: int = 2021, prefix: str = "A(H3N2)") -> str:
    """An invented virus name built from parts."""
    return "/".join([prefix, place, str(number), str(year)])


def designation_identity(
    name: str, reassortant: str, annotations: list[str], fourth: str
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
) -> Table:
    """A :class:`Table` from plain dicts; ``raw_name`` defaults to ``name``."""
    return Table(
        table_id=table_id,
        group=group,
        lab=lab,
        subtype=subtype,
        lineage="",
        assay="HI",
        rbc="turkey",
        date=date,
        date_suffix=date_suffix,
        source_key=f"test {table_id}",
        antigens=[Antigen(**{"raw_name": a["name"], **a}) for a in antigens],
        sera=[Serum(**{"raw_name": s["name"], **s}) for s in sera],
        titres=titres,
    )


class Synthetic:
    """The helpers above, handed to tests as the ``syn`` fixture (conftest modules are not
    importable by name here: pytest runs in importlib mode)."""

    virus = staticmethod(virus)
    table = staticmethod(table)
    rules = TEST_RULES


@pytest.fixture
def syn() -> Synthetic:
    return Synthetic()
