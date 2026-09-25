"""Real NIID data (private fixture): af still reads NIID as recorded, and agrees with ae's
committed tables except for the recorded cells. Skipped without the private data repo."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from af.tables import identity
from af.tables.model import Table
from af.tables.rules import Rules

pytest.importorskip("openpyxl")
from af.tables import niid  # noqa: E402


@pytest.fixture(scope="module")
def fixture(af_data: Path) -> Path:
    path = af_data / "fixtures" / "tables" / "niid"
    if not (path / "expected.json").is_file():
        pytest.skip(f"no NIID tables fixture in {path}")
    return path


@pytest.fixture(scope="module")
def expected(fixture: Path) -> dict:
    return json.loads((fixture / "expected.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def tables(fixture: Path, af_data: Path) -> dict[str, Table]:
    books = sorted((fixture / "workbooks").glob("*.xlsx"))
    result = niid.read(books, Rules(af_data / "rules" / "tables"), lab="NIID")
    assert result.errors == []
    assert identity.assign(result.tables, None) == []
    return {t.table_id: t for t in result.tables}


def cells(table: Table) -> dict[str, str]:
    """Keyed as the fixture keys ae's cells: NIID-ID, its occurrence, serum column."""
    out: dict[str, str] = {}
    seen: dict[str, int] = {}
    for i, antigen in enumerate(table.antigens):
        lab_id = antigen.lab_ids[0]
        n = seen.get(lab_id, 0)
        seen[lab_id] = n + 1
        for j, cell in enumerate(table.titres[i]):
            if cell:
                out[f"{lab_id}|{n}|{j}"] = "/".join(cell)
    return out


def test_tables_match_the_golden_record(tables: dict[str, Table], expected: dict) -> None:
    fmt = expected.get("format", "af-table-1")
    got = {
        i: {
            "content_hash": t.content_hash_as(fmt),
            "map_hash": t.map_hash(),
            "antigens": len(t.antigens),
            "sera": len(t.sera),
            "dropped": t.dropped,
            "warnings": len(t.warnings),
        }
        for i, t in tables.items()
    }
    assert sorted(got) == sorted(expected["af"])
    changed = {i: (expected["af"][i], got[i]) for i in got if got[i] != expected["af"][i]}
    assert changed == {}, (
        "af reads these tables differently from the fixture: explain, then regenerate"
    )


def test_agrees_with_ae_except_the_recorded_cells(tables: dict[str, Table], expected: dict) -> None:
    found = {}
    for table_id, theirs in expected["ae_cells"].items():
        mine = cells(tables[table_id])
        diff = {
            k: {"ae": theirs.get(k, "*"), "af": mine.get(k, "*")}
            for k in sorted(theirs.keys() | mine.keys())
            if theirs.get(k) != mine.get(k)
        }
        if diff:
            found[table_id] = diff
    assert found == expected["ae_differences"]
