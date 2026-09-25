"""Real CNIC data (private fixture): af still reads CNIC as recorded, and agrees with ae's
committed tables except for the cells ae dropped. Skipped without the private data repo."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from af.tables import identity
from af.tables.compare import differences, merged_cells
from af.tables.locations import ChineseLocations
from af.tables.model import Table
from af.tables.rules import Rules

pytest.importorskip("openpyxl")
from af.tables import ac21  # noqa: E402


@pytest.fixture(scope="module")
def fixture(af_data: Path) -> Path:
    path = af_data / "fixtures" / "tables" / "cnic"
    if not (path / "expected.json").is_file():
        pytest.skip(f"no CNIC tables fixture in {path}")
    return path


@pytest.fixture(scope="module")
def expected(fixture: Path) -> dict:
    return json.loads((fixture / "expected.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def tables(fixture: Path, af_data: Path) -> dict[str, Table]:
    locations = ChineseLocations.load(
        fixture / "locationdb-subset.json.xz", fixture / "cnic-location-aliases.tsv"
    )
    books = sorted((fixture / "workbooks").glob("*.xlsx"))
    result = ac21.read(books, Rules(af_data / "rules" / "tables"), locations, lab="CNIC")
    assert result.errors == []
    assert identity.assign(result.tables, None) == []
    return {t.table_id: t for t in result.tables}


def test_tables_match_the_golden_record(tables: dict[str, Table], expected: dict) -> None:
    got = {
        i: {
            "content_hash": t.content_hash(),
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
    assert sorted(expected["ae_missing"]) == sorted(set(tables) - set(expected["ae_cells"]))
    found = {}
    for table_id, cells in expected["ae_cells"].items():
        if diff := differences(cells, merged_cells([tables[table_id]])):
            found[table_id] = diff
    assert found == expected["ae_differences"]
    # the recorded differences are all cells only af has, or the one re-keyed CS passage
    for diff in expected["ae_differences"].values():
        assert all(v["ae"] == "*" or v["af"] == "*" for v in diff.values())
