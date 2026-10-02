"""A table read from ae's .ace where no workbook exists (invented chart)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from af.tables import ace_import
from af.tables.rules import Rules
from af.tables.update import AceInputs, _add_ace

from .synthetic_rules import write_rules

CHART = {
    "c": {
        "i": {"A": "HI", "D": "20300102", "V": "B", "l": "LABN", "r": "chicken", "s": "VICTORIA"},
        "a": [
            {"N": "B/EXAMPLETOWN/1/2029", "P": "MDCK2/HCK1", "S": "R", "l": ["LABN#29/30 - 1"]},
            {"N": "B/EXAMPLEVILLE/2/2030", "P": "A4/A2/HCK1", "l": ["LABN#29/30 - 2"]},
            {"N": "B/EXAMPLEVILLE/3/2030", "P": "HCK1", "l": ["LABN#29/30 - 3"]},
        ],
        "s": [
            {"N": "B/EXAMPLETOWN/1/2029", "I": "CELL NO.1", "P": "MDCK?"},
            {"N": "B/EXAMPLETOWN/1/2029", "I": "EGG NO.2", "P": "E?"},
        ],
        "t": {"l": [["640", "*"], ["160", "*"], ["*", "*"]]},
    }
}
HEADER = "lab\tkind\tpattern\tevidence\tadded_by\tadded_on\toptional"


def setup(tmp_path: Path, fix: bool = True) -> tuple[Path, Rules]:
    folder = tmp_path / "ace"
    folder.mkdir()
    (folder / "labn-20300102.ace").write_text(json.dumps(CHART))
    directory = write_rules(tmp_path / "rules", all_optional=True)
    (directory / "ace_imports.tsv").write_text(
        f"{HEADER}\nLABN\texact\tlabn-20300102.ace\tno workbook\ttest\t2030-01-01\t\n"
    )
    if fix:
        with (directory / "cell_fixes.tsv").open("a") as f:
            f.write(
                "LABN\tlabn-20300102.ace\tantigens\tD3\tA4/A2/HCK1\tAX42/HCK1\tae misread"
                "\ttest\t2030-01-01\t\n"
            )
    return folder, Rules(directory)


def test_an_ace_import_reads_as_af_writes_tables(tmp_path):
    folder, rules = setup(tmp_path)
    tables: list = []
    report: list[str] = []
    errors: list[str] = []
    _add_ace(AceInputs(lab="LABN", dir=folder), rules, tables, report, errors)
    assert errors == []
    (t,) = tables
    assert (t.group, t.date, t.meta["source"]) == (
        "bvic-hi-chicken-labn",
        "2030-01-02",
        ace_import.SOURCE,
    )
    assert [(a.passage, a.lab_ids, a.reference) for a in t.antigens] == [
        ("MDCK2/HCK1", ["LABN#29/30-1"], True),
        ("AX42/HCK1", ["LABN#29/30-2"], False),  # the cell fix
    ]
    assert (t.sera[0].serum_id, t.sera[0].passage) == ("LABN CELL NO.1", "MDCK?")
    assert t.titres == [[["640"]], [["160"]]]
    assert t.dropped == {"antigens: no readings": 1, "sera: no readings": 1}
    assert any("ACE IMPORT labn-20300102.ace" in line for line in report)


def test_a_workbook_for_the_same_test_retires_the_import(tmp_path):
    folder, rules = setup(tmp_path)
    tables: list = []
    report: list[str] = []
    errors: list[str] = []
    _add_ace(AceInputs(lab="LABN", dir=folder), rules, tables, report, errors)
    workbook = tables[0]
    workbook.source_key = "LABN xlsx labn-20300102.xlsx [S]"
    errors2: list[str] = []
    _add_ace(AceInputs(lab="LABN", dir=folder), rules, [workbook], [], errors2)
    (error,) = errors2
    assert "now reads this test from a workbook" in error


def test_a_stale_cell_fix_is_an_error(tmp_path):
    folder, rules = setup(tmp_path)
    CHART_CHANGED = json.loads(json.dumps(CHART))
    CHART_CHANGED["c"]["a"][1]["P"] = "AX42/HCK1"
    (folder / "labn-20300102.ace").write_text(json.dumps(CHART_CHANGED))
    with pytest.raises(ValueError, match="expects 'A4/A2/HCK1'"):
        _add_ace(AceInputs(lab="LABN", dir=folder), rules, [], [], [])
