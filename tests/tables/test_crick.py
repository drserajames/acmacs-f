"""Crick's report-table layout, on synthetic sheets with invented names."""

from __future__ import annotations

from pathlib import Path

import pytest

from af.tables import abbrev
from af.tables.model import Antigen
from af.tables.passage import PassageParser
from af.tables.rules import Rules

openpyxl = pytest.importorskip("openpyxl")
from af.tables import crick  # noqa: E402

TITLE = "Table X-1. Antigenic analyses of influenza B viruses (Victoria lineage) 2030-01-02"
SERA = [  # (name row 1, name row 2, passage, ferret number)
    ("B/Exa", "12/29", "Egg", "Sh 5, 6, 7*1,3"),  # sheep pool: marks 1 and 3
    ("B/Exl", "7/29", "MDCK", "F01/29*2"),  # letters in order
    ("BVR-99", "B/Exampleland/7/29", "Egg J141O", "NIB F02/29*2"),  # reassortant over its virus
    ("B/Exa", "5/30", "MDCK", "NEW F03/30*1"),  # 2029 and 2030 on the table: the year decides
    ("B/Exa", "9/30", "MDCK", "TBG"),  # not tested ("*" throughout): dropped
]
REFERENCE = [  # (lab id, name, other information, date, passage, titres)
    ("", "B/Exampleville/12/2029", "", "2029-03-01", "E3/E1 10-3", ["1280", "80", "40", "<", "*"]),
    (
        "",
        "B/Exampleland/7/2029",
        "J141",
        "2029-02-01",
        "MDCK 1/SIAT1",
        ["<", "640", "320", "80", "*"],
    ),
    (
        "",
        "BVR-99 (B/Exampleland/7/2029)",
        "A1B, C2D (J141O)",
        "2029-02-01",
        "E3/D1",
        ["<", "320", "1280", "40", "*"],
    ),
    (
        "",
        "B/Exampleville/5/2029",
        "clone 3",
        "2029-05-01",
        "P1/MDCK1",
        ["40", "<", "80", "160", "*"],
    ),
    ("", "B/Exampleville/5/2030", "A1B, C2D", "", "SIAT P1/SIAT1", ["80", "40", "ND", "640", "*"]),
]
TEST = [
    ("100001", "B/Exampletown/9/2030", "", "01/01/2030", "MDCK1", ["160", "80", "40", "320", "*"])
]
LEGEND = "1 < = <40; 2 < = <10; 3 hyperimmune sheep serum; ND = Not Done"


def workbook(
    path: Path,
    *,
    title: str = TITLE,
    sera: list[tuple] | None = None,
    reference: list[tuple] | None = None,
    stray: list[str] | None = None,
    extra_sheets: bool = False,
) -> Path:
    sera = sera or SERA
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "BX 020130"
    rows: list[list[str]] = [
        [title],
        [],
        ["", "", "", "", "", "", "Haemagglutination inhibition titre"],
        ["", "Viruses", "Other", "", "Collection", "Passage", *(s[0] for s in sera)],
        ["", "", "information", "", "date", "history", *(s[1] for s in sera)],
        ["", "", "", "Passage history", "", "", *(s[2] for s in sera)],
        ["", "", "", "Ferret number", "", "", *(s[3] for s in sera)],
        ["", "", "", "Genetic group", "", "", *("X.1" for _ in sera)],
        ["", "REFERENCE VIRUSES"],
    ]
    for lab_id, name, other, date, passage, titres in reference or REFERENCE:
        rows.append([lab_id, name, other, "X.1", date, passage, *titres])
    # the template's homologous titre left under the last reference antigen
    rows.append(stray if stray is not None else ["", "", "", "", "", "", "", "", "", "640"])
    rows.append(["", "TEST VIRUSES"])
    for lab_id, name, other, date, passage, titres in TEST:
        rows.append([lab_id, name, other, "X.1", date, passage, *titres])
    rows += [[], ["*", "Superscripts refer to antiserum properties"], ["", LEGEND]]
    for r in rows:
        ws.append(r)
    if extra_sheets:
        notes = wb.create_sheet("Sheet1")
        notes.append(["Virus name", "Clade"])
        other = wb.create_sheet("H3")
        for r in rows:
            other.append(
                [
                    v.replace("influenza B viruses (Victoria lineage)", "influenza A(H3N2) viruses")
                    for v in r
                ]
            )
    wb.save(path)
    return path


def read(path: Path, rules: Rules, subtype: str = "B", lineage: str = "VICTORIA"):
    return crick.read([path], rules, lab="LABC", subtype=subtype, lineage=lineage)


def one_table(res):
    assert res.errors == []
    (table,) = res.tables
    return table


def test_reads_a_sheet(tmp_path, rules):
    t = one_table(read(workbook(tmp_path / "labc-20300102.xlsx"), rules))
    assert (t.group, t.rbc, t.date, t.assay) == (
        "bvic-hi-turkey-labc",
        "turkey",
        "2030-01-02",
        "HI",
    )
    assert [
        (s.name, s.reassortant, s.serum_id, s.passage, s.annotations, s.species) for s in t.sera
    ] == [
        ("B/EXAMPLEVILLE/12/2029", "", "LABC SH5/6/7", "E?", [], "SHEEP"),
        ("B/EXAMPLELAND/7/2029", "", "LABC F01/29", "MDCK?", [], ""),
        ("B/EXAMPLELAND/7/2029", "BVR-99", "LABC NIB F02/29", "E?", ["J141O"], ""),
        ("B/EXAMPLEVILLE/5/2030", "", "LABC F03/30", "MDCK?", [], ""),
    ]
    assert t.dropped["sera: no readings"] == 1
    assert [(a.name, a.reassortant, a.annotations, a.passage) for a in t.antigens] == [
        ("B/EXAMPLEVILLE/12/2029", "", [], "E3/E1"),
        ("B/EXAMPLELAND/7/2029", "", ["J141"], "MDCK1/SIAT1"),
        ("B/EXAMPLELAND/7/2029", "BVR-99", ["J141O"], "E3/D1"),
        ("B/EXAMPLEVILLE/5/2029", "", ["CLONE 3"], "X1/MDCK1"),
        ("B/EXAMPLEVILLE/5/2030", "", [], "SIAT1/SIAT1"),
        ("B/EXAMPLETOWN/9/2030", "", [], "MDCK1"),
    ]
    # a list of substitutions describes the virus: kept, but not part of its identity
    assert t.antigens[4].source["other_information"] == "A1B, C2D"
    assert [a.reference for a in t.antigens] == [True] * 5 + [False]
    assert t.antigens[5].lab_ids == ["LABC#100001"]
    assert t.antigens[5].date == "2030-01-01"
    # a bare "<" reads as the serum's footnote: marks 1,3 -> <40; 2 -> <10; 1 -> <40
    assert t.titres[1][0] == ["<40"]
    assert t.titres[3][1] == ["<10"]
    assert t.titres[0][3] == ["<40"]
    assert t.titres[4][2] == []  # ND
    assert t.dropped["cells: no antigen"] == 1


def test_a_bare_less_than_without_a_footnote_is_an_error(tmp_path, rules):
    sera = [*SERA[:3], (*SERA[3][:3], "F03/30"), SERA[4]]
    res = read(workbook(tmp_path / "labc-20300102.xlsx", sera=sera), rules)
    assert res.tables == [] and "0 footnote values" in res.errors[0]


def test_two_titres_without_a_name_are_an_error(tmp_path, rules):
    stray = ["", "", "", "", "", "", "80", "", "", "640"]
    res = read(workbook(tmp_path / "labc-20300102.xlsx", stray=stray), rules)
    assert res.tables == [] and "no strain name" in res.errors[0]


def test_other_sheets_and_subtypes_are_reported_not_read(tmp_path, rules):
    path = workbook(tmp_path / "labc-20300102.xlsx", extra_sheets=True)
    res = read(path, rules)
    assert len(res.tables) == 1
    assert any("no 'Ferret number' row" in s for s in res.skipped_tests)
    assert any("A(H3N2)" in s and "not B" in s for s in res.skipped_tests)
    h3 = one_table(read(path, rules, subtype="A(H3N2)", lineage=""))
    assert h3.group == "h3-hi-turkey-labc"


def test_red_cells_from_the_title(tmp_path, rules):
    title = (
        "Table H3-1. Antigenic analyses of influenza A(H3N2) viruses (Guinea Pig RBC) 2030-01-02"
    )
    res = read(workbook(tmp_path / "labc-20300102.xlsx", title=title), rules, "A(H3N2)", "")
    assert one_table(res).rbc == "guinea-pig"


def test_the_file_date_must_be_a_sheets_date(tmp_path, rules):
    res = read(workbook(tmp_path / "labc-20300103.xlsx"), rules)
    assert res.tables == [] and any("0 sheets with the file's test date" in e for e in res.errors)


def test_a_summary_sheet_without_a_date(tmp_path, rules):
    title = "Table X-2. Antigenic analyses of influenza B viruses (Victoria lineage) - Summary"
    undated = read(workbook(tmp_path / "labc-summary.xlsx", title=title), rules)
    assert undated.tables == [] and undated.errors == []
    assert any("a summary sheet" in s for s in undated.skipped_tests)
    dated = read(workbook(tmp_path / "labc-20300102.xlsx", title=title), rules)
    assert dated.tables == [] and "0 dates in the title" in dated.errors[0]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("MDCK 1/SIAT1", "MDCK1/SIAT1"),
        ("SIAT P1/SIAT1", "SIAT1/SIAT1"),
        ("MDCKP2/MDCK1", "MDCK2/MDCK1"),
        ("MDCK C1/MDCK1", "MDCK1/MDCK1"),
        ("P1/MDCK", "X1/MDCK?"),
        ("Px/MDCK1", "X?/MDCK1"),
        ("E3/E1 10-3", "E3/E1"),
        ("E3 (Am1Al2)", "E3(AM1AL2)"),
        ("C2+1/MDCK1", "MDCK2/MDCK1/MDCK1"),
    ],
)
def test_passages(rules, raw, expected):
    parser = PassageParser(rules.passage_tokens, "LABC")
    assert parser.parse(crick._passage_text(raw, parser)).text == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("F12/28*2", ("F12/28", ["2"])),
        ("Sh 5, 6, 7*1,3", ("SH5/6/7", ["1", "3"])),
        ("NEW F12/30", ("F12/30", [])),
        ("NIB F01/29*1", ("NIB F01/29", ["1"])),
        ("St Exa's F18/29", ("ST EXA'S F18/29", [])),
    ],
)
def test_serum_ids(raw, expected):
    assert crick._serum_id(raw) == expected


@pytest.mark.parametrize(
    ("text", "tags"),
    [
        ("J141O", ["J141O"]),
        ("A1B, C2D (J141)", ["J141"]),
        ("(J141O)", ["J141O"]),
        ("clone 37", ["CLONE 37"]),
        ("Isl 2", ["ISOLATE 2"]),
        ("A1B, C2D", []),
        ("?", []),
        ("J150O grp + O220J", []),
    ],
)
def test_other_information_tags(text, tags):
    assert crick._tags(text) == tags


def test_abbreviation_year_decides():
    antigens = [
        Antigen(name=n, raw_name=n) for n in ("B/EXAMPLEVILLE/5/2029", "B/EXAMPLEVILLE/5/2030")
    ]
    assert list(abbrev.match("Exa", "5", "30", antigens)) == [("B/EXAMPLEVILLE/5/2030", "")]
    assert len(abbrev.match("Exa", "5", "", antigens)) == 2
    assert list(abbrev.match("Exvl", "05", "2029", antigens)) == [("B/EXAMPLEVILLE/5/2029", "")]
