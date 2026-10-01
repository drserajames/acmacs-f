"""CDC's 2013-19 HI workbook layout (one 'tblExcelSpreadsheet' per test), on synthetic sheets
laid out as CDC laid them out then (invented strains)."""

from __future__ import annotations

from pathlib import Path

import pytest

from af.tables.passage import PassageParser
from af.tables.rules import Rules

openpyxl = pytest.importorskip("openpyxl")
from af.tables import cdc_xlsx  # noqa: E402

LETTERS = ["X", "Y", "Z", "AA", "BB"]  # a run continuing from an earlier test, doubled after Z


def workbook(path: Path, *, cells: dict[str, object] | None = None) -> Path:
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "tblExcelSpreadsheet"
    ws["A1"] = "HEMAGGLUTINATION INHIBITION REACTIONS OF INFLUENZA TYPE B VICTORIA LINEAGE VIRUSES"
    ws["B3"] = "TESTED 5/8/31"
    ws["A7"] = "REFERENCE ANTIGENS"
    for i, letter in enumerate(LETTERS):
        ws.cell(7, 6 + i, letter)
        ws.cell(8, 6 + i, f"EX/{i}")
    ws.cell(8, 6 + len(LETTERS), "PASSAGE")  # on the abbreviations row, above the labels
    for col, label in [(3, "STRAIN DESIGNATION"), (4, "CDC ID#"), (5, "DATE COLLECTED")]:
        ws.cell(9, col, label)
    rows: list[tuple[str, int | None, str | None, str, list[int | str]]] = [
        ("B/EXAMPLETOWN/1/2030", 9000000001, "2030-01-02", "CX,C4/C2(8/29/30)", [640, 160]),
        ("B/EXAMPLETOWN/2/2030 (NEW)", 9000000002, "2030-02-03", "E3+3/E2(1/2/31)", [80, 1280]),
        ("TEST ANTIGENS", None, None, "", []),
        ("B/EXAMPLEOTHER/3/2031", 9000000003, "2031-03-04", "C1(4/20/31)", ["QNS", 320]),
    ]
    for r, (name, cdc_id, collected, passage, titres) in enumerate(rows, start=10):
        if name == "TEST ANTIGENS":
            ws.cell(r, 1, name)
            ws.cell(r, 7, "RECEIVED AS EXAMPLE")  # a note over a titre column
            continue
        ws.cell(r, 1, r - 9)
        ws.cell(r, 2, "EX")
        ws.cell(r, 3, name)
        ws.cell(r, 4, cdc_id)
        ws.cell(r, 5, collected)
        for i, titre in enumerate(titres):
            ws.cell(r, 6 + i, titre)
        for i in range(2, len(LETTERS)):  # the dropped sera's columns
            ws.cell(r, 6 + i, 10)
        ws.cell(r, 6 + len(LETTERS), passage)
    ws.cell(15, 3, "SERUM CONTROL")
    ws.cell(18, 2, "REFERENCE ANTISERA")
    for col, label in [
        (5, "LOT #"),
        (8, "SPECIES"),
        (9, "NEW"),
        (10, "BOOSTED"),
        (11, "CONC."),
        (12, "POOL"),
        (14, "PASSAGE"),
    ]:
        ws.cell(18, col, label)
    sera = [
        ("X", "B/EXAMPLETOWN/1/30", "T30-001", "FERRET", "BOOSTED", "", "E3 (1/2/30)"),
        ("Y", "B/EXAMPLETOWN/2/2030", "T30-002, 003", "FERRET", "NOT BOOSTED", "2:1", "C2"),
        ("Z", "FR-1 EXAMPLE LINEAGE", "3031EXAS", "SHEEP", "BOOSTED", "", ""),
        ("AA", "NORMAL FERRET SERUM", "1", "FERRET", "", "", ""),
        ("BB", "B/EXAMPLEOTHER/3/2031", "T31-004", "FERRET", "PRE BOOST BLEED", "", "C1"),
    ]
    for r, (letter, name, lot, species, boosted, conc, passage) in enumerate(sera, start=19):
        for col, value in [
            (1, letter),
            (2, name),
            (5, lot),
            (8, species),
            (10, boosted),
            (11, conc),
            (14, passage),
        ]:
            ws.cell(r, col, value)
    ws.cell(26, 8, "RBC'S USED")
    ws.cell(27, 8, "TURKEY")
    for ref, content in (cells or {}).items():
        ws[ref] = content
    wb.save(path)
    return path


def test_reads_the_2013_19_layout(tmp_path: Path, rules: Rules):
    res = cdc_xlsx.read([workbook(tmp_path / "old.xlsx")], rules)
    assert res.errors == []
    (t,) = res.tables
    assert (t.group, t.date) == ("bvic-hi-turkey-cdc", "2031-05-08")
    assert [a.name for a in t.antigens] == [
        "B/EXAMPLETOWN/1/2030",
        "B/EXAMPLETOWN/2/2030",  # "(NEW)" is description: name_rewrites
        "B/EXAMPLEOTHER/3/2031",
    ]
    assert [(a.passage, a.passage_date) for a in t.antigens] == [
        ("MDCK?/MDCK4/MDCK2", "2030-08-29"),  # "CX,C4/C2": the comma joins steps
        ("E3/E3/E2", "2031-01-02"),  # "E3+3": three more egg passages
        ("MDCK1", "2031-04-20"),
    ]
    assert t.antigens[0].annotations == []
    # the sheep serum and the normal serum are not antisera: dropped by rule, counted
    assert t.dropped["sera: control"] == 2
    assert [s.serum_id for s in t.sera] == ["CDC T30-001", "CDC T30-002, 003", "CDC T31-004"]
    assert t.sera[0].name == "B/EXAMPLETOWN/1/2030"  # a two-digit year, before the test year
    assert [s.annotations for s in t.sera] == [["BOOSTED"], ["CONC 2:1"], []]
    assert t.sera[2].source["boosted"] == "PRE BOOST BLEED"
    assert t.titres[2] == [[], ["320"], ["10"]]  # QNS: not tested
    assert any("note on the section row" in w for w in t.warnings)


def test_a_non_dilution_is_a_typing_error(tmp_path: Path, rules: Rules):
    res = cdc_xlsx.read([workbook(tmp_path / "old.xlsx", cells={"F10": 64})], rules)
    assert len(res.errors) == 1 and "'64' is not a dilution" in res.errors[0]


def test_a_collection_after_the_test_is_a_typing_error(tmp_path: Path, rules: Rules):
    res = cdc_xlsx.read([workbook(tmp_path / "old.xlsx", cells={"E10": "2031-10-02"})], rules)
    assert len(res.errors) == 1 and "after the test date" in res.errors[0]
    assert "!E10:" in res.errors[0]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("CX,C4/C2", "CXC4/C2"),
        ("C1+C1/C1", "C1/C1/C1"),
        ("E2+2/E3", "E2/E2/E3"),
        ("S1(7/19/2030)", "S1(7/19/2030)"),  # the 2024+ notation is untouched
    ],
)
def test_cdc_passage(rules: Rules, text: str, expected: str):
    assert cdc_xlsx.cdc_passage(text, PassageParser(rules.passage_tokens, "CDC")) == expected


@pytest.mark.parametrize(
    ("letter", "expected"), [("A", "B"), ("Y", "Z"), ("Z", "AA"), ("AA", "BB"), ("BB", "CC")]
)
def test_serum_letters_double_after_z(letter: str, expected: str):
    assert cdc_xlsx._next_letter(letter) == expected


def test_a_titre_column_with_no_serum_letter_is_an_error(tmp_path: Path, rules: Rules):
    # serum BB loses its letter, in the header and in the antisera block: its titres remain
    res = cdc_xlsx.read([workbook(tmp_path / "old.xlsx", cells={"J7": None, "A23": None})], rules)
    assert len(res.errors) == 1 and "a titre in a column with no serum letter" in res.errors[0]
