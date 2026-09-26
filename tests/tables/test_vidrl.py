"""VIDRL's workbook layout, on synthetic sheets with invented names."""

from __future__ import annotations

from pathlib import Path

import pytest

from af.tables.passage import PassageParser
from af.tables.rules import Rules

openpyxl = pytest.importorskip("openpyxl")
from af.tables import vidrl  # noqa: E402

SERA = [  # (index, id, passage, abbreviated name)
    ("1", "A0001", "MDCK2", "Exa/12"),
    ("2", "A0002", "E3", "Exl/7"),
    ("3", "A0003", "E3/D1", "BVR-99"),
    ("4", "", "MDCK1", "Exa/5"),  # no id: warned
    ("5", "SH 2030", "CELL", "sera pool"),  # a human pool: dropped
]
ANTIGENS = [  # (index, name, titres, passage, date, lab id)
    ("1", "B/EXAMPLEVILLE/12/2029", ["640", "80", "40", "80", "320"], "MDCK 1, MDCK1", "", ""),
    ("2", "B/EXAMPLELAND/7/2029", ["80", "1280", "40", "40", "160"], "C1+1", "", ""),
    ("3", "BVR-99 (B/EXAMPLELAND/7/2029)", ["40", "640", "640", "40", "160"], "E3/D1", "", ""),
    ("4", "B/EXAMPLEVILLE/5/2029", ["<20", "40", "40", "320", "80"], "X, SIAT1", "", ""),
    (
        "",
        "B/EXAMPLEVILLE/5/2030",
        ["80", "40", "40", "160", "80"],
        "P1 SIAT",
        "01/01/2030",
        "VW10000001",
    ),
    (
        "",
        "B/EXAMPLETOWN/9/2030",
        ["320", "80", "40", "40", "160"],
        "QMC2-HI",
        "31/12/2029",
        "SL10000002",
    ),
]


def workbook(
    path: Path,
    *,
    date: str = "Test Date: 02/01/2030",
    antigens: list[tuple] | None = None,
    extra_sheet: bool = False,
) -> Path:
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Worksheet"
    rows: list[list[str]] = [
        [],
        ["", "", date],
        ["", "", "Tested By: X"],
        ["", "", "RBC Type: Turkey"],
        ["", "", "Test Method: Manual", "", "Reference Antisera"],
        ["", "", "", "", *(s[0] for s in SERA)],
        ["", "", "", "", *(s[1] for s in SERA), "Passage", "Sample"],
        ["", "", "", "", *(s[2] for s in SERA), "Details", "Date", "ID #"],
        ["", "", "", "", *(s[3] for s in SERA)],
        ["", "", "", "Clade>", "X.1", "X.1", "X.1", "X.2", ""],
    ]
    for index, name, titres, passage, sample, lab_id in antigens or ANTIGENS:
        rows.append(["", index, name, "X.1", *titres, passage, sample, lab_id])
    rows.append(["", "", "a note under the table"])
    for r in rows:
        ws.append(r)
    if extra_sheet:  # VIDRL keeps an earlier test in a second sheet
        other = wb.create_sheet("Sheet1")
        for r in rows:
            other.append([v.replace("02/01/2030", "20/12/2029") for v in r])
    wb.save(path)
    return path


def read(path: Path, rules: Rules) -> vidrl.ReadResult:
    return vidrl.read([path], rules, lab="LABV", subtype="B", lineage="VICTORIA")


def one_table(res: vidrl.ReadResult):
    assert res.errors == []
    (table,) = res.tables
    return table


def test_reads_a_sheet(tmp_path, rules):
    t = one_table(read(workbook(tmp_path / "labv-20300102.xlsx"), rules))
    assert (t.group, t.rbc, t.date) == ("bvic-hi-turkey-labv", "turkey", "2030-01-02")
    assert [a.passage for a in t.antigens] == [
        "MDCK1/MDCK1",
        "MDCK1/MDCK1",
        "E3/D1",
        "X?/SIAT1",
        "SIAT1",
        "QMC2",
    ]
    assert [a.reference for a in t.antigens] == [True, True, True, True, False, False]
    assert t.antigens[4].lab_ids == ["LABV#VW10000001"]
    assert t.antigens[4].date == "2030-01-01"
    # abbreviated sera resolve to the table's own antigens
    assert [(s.name, s.reassortant, s.serum_id) for s in t.sera] == [
        ("B/EXAMPLEVILLE/12/2029", "", "LABV A0001"),  # prefix
        ("B/EXAMPLELAND/7/2029", "", "LABV A0002"),  # letters in order
        ("B/EXAMPLELAND/7/2029", "BVR-99", "LABV A0003"),  # a reassortant alone
        ("B/EXAMPLEVILLE/5/2029", "", ""),  # 2029 and 2030 both match: the index decides
    ]
    assert t.dropped["sera: control"] == 1
    assert any("has no id" in w for w in t.warnings)


def test_titre_that_is_not_a_dilution_is_an_error(tmp_path, rules):
    antigens = [
        *ANTIGENS[:5],
        (*ANTIGENS[5][:2], ["302", "80", "40", "40", "160"], *ANTIGENS[5][3:]),
    ]
    res = read(workbook(tmp_path / "labv-20300102.xlsx", antigens=antigens), rules)
    assert res.tables == [] and "not a dilution" in res.errors[0]


def test_a_cell_fix_repairs_a_typo_and_a_stale_one_is_an_error(tmp_path, rules):
    antigens = [(*ANTIGENS[0][:2], ["604", *ANTIGENS[0][2][1:]], *ANTIGENS[0][3:]), *ANTIGENS[1:]]
    t = one_table(read(workbook(tmp_path / "labv-fix-20300102.xlsx", antigens=antigens), rules))
    assert t.titres[0][0] == ["640"]
    assert any("fixed as '640'" in w for w in t.warnings)
    res = read(workbook(tmp_path / "labv-fix-20300102.xlsx"), rules)  # E11 now holds 640
    assert res.tables == [] and "expects '604'" in res.errors[0]


def test_another_tests_sheet_is_reported_not_read(tmp_path, rules):
    res = read(workbook(tmp_path / "labv-20300102.xlsx", extra_sheet=True), rules)
    assert len(res.tables) == 1
    assert any("another test's sheet" in s for s in res.skipped_tests)


def test_the_file_date_must_be_a_sheets_date(tmp_path, rules):
    res = read(workbook(tmp_path / "labv-20300103.xlsx"), rules)
    assert res.tables == [] and any("0 sheets with the file's test date" in e for e in res.errors)


def test_an_ambiguous_abbreviation_is_an_error(tmp_path, rules):
    antigens = [a for a in ANTIGENS if a[1] != "B/EXAMPLEVILLE/5/2029"]
    antigens.append(("", "B/EXAMPLEVILLE/5/2028", ["80", "40", "40", "160", "80"], "E3", "", ""))
    res = read(workbook(tmp_path / "labv-20300102.xlsx", antigens=antigens), rules)
    assert res.tables == [] and "matches 2 antigen names" in res.errors[0]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("MDCK 1, MDCK1", "MDCK1/MDCK1"),
        ("MDCK-1, SIAT1", "MDCK1/SIAT1"),
        ("MDCK#1, MDCK1", "MDCK1/MDCK1"),
        ("C1+1, MDCK1", "MDCK1/MDCK1/MDCK1"),
        ("C2, 2", "MDCK2/MDCK2"),
        ("X, SIAT1", "X?/SIAT1"),
        ("P1 SIAT, MDCK1", "SIAT1/MDCK1"),
        ("QMC2-HI", "QMC2"),
        ("SIAT2 SIAT2", "SIAT2/SIAT2"),
        ("E3/SpE1,E1", "E3/SPE1/E1"),
    ],
)
def test_passages(rules, raw, expected):
    parser = PassageParser(rules.passage_tokens, "LABV")
    assert parser.parse(vidrl._passage_text(raw, parser)).text == expected


@pytest.mark.parametrize(
    ("text", "splits"),
    [
        ("Exa/1234", [("Exa", "1234")]),
        ("EXC07", [("EXC", "07")]),
        ("Exa City 272", [("Exa City", "272")]),
        ("ExaABC1234", [("ExaABC", "1234"), ("Exa", "ABC1234")]),
        ("B/EXA/5", [("EXA", "5")]),
    ],
)
def test_abbreviation_splits(text, splits):
    assert vidrl._splits(text) == splits
