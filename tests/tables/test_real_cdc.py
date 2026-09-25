"""Real CDC data (private fixture): af still reads CDC as recorded, and still agrees with ae.

The fixture (acmacs-f-data ``fixtures/tables/cdc/``, see its README) holds 13 CDC tests
and 3 workbooks. Skipped when the private data repo is absent.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from af.tables import cdc, identity
from af.tables.compare import differences, merged_cells
from af.tables.model import Table
from af.tables.rules import Rules
from af.tables.update import duplicates

pytest.importorskip("openpyxl")
from af.tables import cdc_xlsx  # noqa: E402

DUPLICATE = "h3-hi-guinea-pig-cdc-20240617_002.xlsx"


@pytest.fixture(scope="module")
def fixture(af_data: Path) -> Path:
    path = af_data / "fixtures" / "tables" / "cdc"
    if not (path / "expected.json").is_file():
        pytest.skip(f"no CDC tables fixture in {path}")
    return path


@pytest.fixture(scope="module")
def expected(fixture: Path) -> dict:
    return json.loads((fixture / "expected.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def rules_dir(af_data: Path) -> Path:
    return af_data / "rules" / "tables"


def workbooks(fixture: Path, *, duplicate: bool = False) -> list[Path]:
    return sorted(
        p for p in (fixture / "workbooks").glob("*.xlsx") if (p.name == DUPLICATE) == duplicate
    )


def read_all(fixture: Path, rules_dir: Path, *, drop_flagged: bool = True) -> list[Table]:
    rules = Rules(rules_dir)
    tsv = cdc.read(fixture / "cdc-subset.tsv", rules, drop_flagged=drop_flagged)
    xl = cdc_xlsx.read(workbooks(fixture), rules)
    assert tsv.errors == [] and xl.errors == []
    tables = tsv.tables + xl.tables
    assert identity.assign(tables, None) == []
    return tables


def test_tables_match_the_golden_record(fixture: Path, rules_dir: Path, expected: dict) -> None:
    got = {
        t.table_id: {
            "content_hash": t.content_hash(),
            "map_hash": t.map_hash(),
            "antigens": len(t.antigens),
            "sera": len(t.sera),
            "dropped": t.dropped,
        }
        for t in read_all(fixture, rules_dir)
    }
    assert sorted(got) == sorted(expected["af"])
    changed = {i: (expected["af"][i], got[i]) for i in got if got[i] != expected["af"][i]}
    assert changed == {}, (
        "af reads these tables differently from the fixture: explain, then regenerate"
    )


def test_duplicate_workbook_is_refused(fixture: Path, rules_dir: Path, expected: dict) -> None:
    rules = Rules(rules_dir)
    tsv = cdc.read(fixture / "cdc-subset.tsv", rules)
    dup = cdc_xlsx.read(workbooks(fixture, duplicate=True), rules)
    assert duplicates(tsv.tables, dup.tables) == expected["duplicate_guard"] != []


def test_agrees_with_ae_except_the_recorded_cells(
    fixture: Path, rules_dir: Path, expected: dict
) -> None:
    by_day: dict[str, list[Table]] = {}
    for t in read_all(fixture, rules_dir, drop_flagged=False):
        by_day.setdefault(f"{t.group}-{t.date.replace('-', '')}", []).append(t)
    assert sorted(by_day) == sorted(expected["ae_cells"])
    found = {}
    for day, tables in by_day.items():
        if diff := differences(expected["ae_cells"][day], merged_cells(tables)):
            found[day] = diff
    assert found == expected["ae_differences"]


def test_linares_rename_and_mouse_sera(fixture: Path, rules_dir: Path) -> None:
    """Two decisions the fixture pins: the named rename (with its titre check) and the
    all-mouse table's sera tagged, not dropped."""
    tables = {t.table_id: t for t in read_all(fixture, rules_dir)}
    renamed = [
        a for a in tables["h3-hi-guinea-pig-cdc-20260730"].antigens if "alias_rule" in a.source
    ]
    assert len(renamed) == 1 and renamed[0].name.startswith("A(H3N2)/")
    mouse = tables["h1pdm-hi-turkey-cdc-20210909"].sera
    assert mouse and all(s.species == "MOUSE" for s in mouse)
