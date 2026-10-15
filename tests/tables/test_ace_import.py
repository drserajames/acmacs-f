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
    workbook.meta["file"] = "labn-20300102.xlsx"
    errors2: list[str] = []
    _add_ace(AceInputs(lab="LABN", dir=folder), rules, [workbook], [], errors2)
    (error,) = errors2
    assert "now reads this test from a workbook" in error


def test_a_second_plate_of_the_day_is_not_retired_by_the_first(tmp_path):
    """The import is the lab's second plate (labn-20300102_002.ace, no workbook); the first
    plate's workbook (labn-20300102.xlsx) has the same group and date but is another test."""
    folder, rules = setup(tmp_path)
    (folder / "labn-20300102.ace").rename(folder / "labn-20300102_002.ace")
    (rules.directory / "ace_imports.tsv").write_text(
        f"{HEADER}\nLABN\texact\tlabn-20300102_002.ace\tno workbook\ttest\t2030-01-01\t\n"
    )
    rules = Rules(rules.directory)
    tables: list = []
    _add_ace(AceInputs(lab="LABN", dir=folder), rules, tables, [], [])
    plate1 = tables[0]
    plate1.source_key = "LABN xlsx labn-20300102.xlsx [S]"
    plate1.meta["file"] = "labn-20300102.xlsx"
    errors: list[str] = []
    _add_ace(AceInputs(lab="LABN", dir=folder), rules, [plate1], [], errors)
    assert errors == []


@pytest.mark.parametrize("file", [None, "LABN_season_2030.tsv"])
def test_a_test_from_an_export_retires_by_group_and_date(tmp_path, file):
    """A TSV export (no file of its own, or one file of many tests) pairs as before."""
    folder, rules = setup(tmp_path)
    tables: list = []
    _add_ace(AceInputs(lab="LABN", dir=folder), rules, tables, [], [])
    exported = tables[0]
    exported.source_key = "LABN test_id 1"
    if file is None:
        del exported.meta["file"]
    else:
        exported.meta["file"] = file
    errors: list[str] = []
    _add_ace(AceInputs(lab="LABN", dir=folder), rules, [exported], [], errors)
    (error,) = errors
    assert "now reads this test from a workbook" in error


def test_a_stale_cell_fix_is_an_error(tmp_path):
    folder, rules = setup(tmp_path)
    CHART_CHANGED = json.loads(json.dumps(CHART))
    CHART_CHANGED["c"]["a"][1]["P"] = "AX42/HCK1"
    (folder / "labn-20300102.ace").write_text(json.dumps(CHART_CHANGED))
    with pytest.raises(ValueError, match="expects 'A4/A2/HCK1'"):
        _add_ace(AceInputs(lab="LABN", dir=folder), rules, [], [], [])


def test_an_annotation_of_two_words_stays_one(tmp_path):
    folder, rules = setup(tmp_path)
    chart = json.loads(json.dumps(CHART))
    chart["c"]["a"][0]["a"] = ["ISOLATE 2", "EXAMPLE1X"]
    chart["c"]["s"][0]["a"] = ["ISOLATE 2"]
    (folder / "labn-20300102.ace").write_text(json.dumps(chart))
    tables: list = []
    errors: list[str] = []
    _add_ace(AceInputs(lab="LABN", dir=folder), rules, tables, [], errors)
    assert errors == []
    (t,) = tables
    assert t.antigens[0].annotations == ["ISOLATE 2", "EXAMPLE1X"]
    assert t.antigens[1].annotations == []
    assert t.sera[0].annotations == ["ISOLATE 2"]


def neut_chart(assay: str, **info: str) -> dict:
    """An invented neutralisation chart: ae writes no red-cell field ("r") on these."""
    return {
        "c": {
            "i": {"A": assay, "D": "20300102", "V": "A(H3N2)", "l": "LABV", **info},
            "a": [{"N": "A(H3N2)/EXAMPLETOWN/1/2029", "P": "SIAT1", "S": "R"}],
            "s": [{"N": "A(H3N2)/EXAMPLETOWN/1/2029", "I": "F29-01", "P": "SIAT1"}],
            "t": {"l": [["1280"]]},
        }
    }


def read_neut(tmp_path: Path, chart: dict):  # noqa: ANN201
    path = tmp_path / "labv-20300102.ace"
    path.write_text(json.dumps(chart))
    directory = write_rules(tmp_path / "rules", all_optional=True)
    (directory / "ace_imports.tsv").write_text(
        f"{HEADER}\nLABV\texact\tlabv-20300102.ace\tno workbook\ttest\t2030-01-01\t\n"
    )
    rules = Rules(directory)
    return ace_import.read(path, rules.ace_imports.rules[0], rules, lab="LABV")


@pytest.mark.parametrize(
    ("assay", "info", "code"),
    [
        ("FOCUS REDUCTION", {}, "FRA"),
        ("FRA", {}, "FRA"),
        ("PRN", {}, "PRN"),
        ("HINT", {}, "HINT"),
        ("FOCUS REDUCTION", {"r": "turkey"}, "FRA"),  # a stray red-cell field on a neut chart
    ],
)
def test_a_neutralisation_chart_reads_without_red_cells(tmp_path, assay, info, code):
    (t,) = read_neut(tmp_path, neut_chart(assay, **info)).tables
    assert (t.assay, t.rbc, t.group) == (code, "", f"h3-{code.lower()}-labv")


def test_an_hi_chart_without_red_cells_is_an_error(tmp_path):
    with pytest.raises(ValueError, match="red-cell"):
        read_neut(tmp_path, neut_chart("HI"))


def test_an_assay_af_has_no_code_for_is_an_error(tmp_path):
    with pytest.raises(ValueError, match="EXAMPLE ASSAY"):
        read_neut(tmp_path, neut_chart("EXAMPLE ASSAY"))
