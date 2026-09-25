"""NIID's workbook layout, on synthetic sheets with invented names."""

from __future__ import annotations

from pathlib import Path

import pytest

from af.tables.passage import PassageParser
from af.tables.rules import Rules

openpyxl = pytest.importorskip("openpyxl")
from af.tables import niid  # noqa: E402

TITLE = "Hemagglutination-inhibition test of influenza EXAMPLE H3 viruses - 1.0% GRBC"
SERA = [
    "EXAMPLECITY /1/29 Cell No.101",
    "EXAMPLECITY/ 1/29 Egg SAN-9 #43",
    "EXAMPLEVILLE/12 Cell NIID No.1",  # no year: taken from its antigen
]


def workbook(
    path: Path,
    *,
    test_date: str = "HI test date:2030/1/2",
    sera: list[str] | None = None,
    extra_rows: list[list[str]] | None = None,
    annotations: list[str] | None = None,
) -> Path:
    sera = sera or SERA
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "20300102"
    notes = annotations or ["", ""]
    rows = [
        [TITLE],
        ["", "", "", "", "clade x", "", "", "", "", test_date],
        [
            "NIID-ID",
            "Strains",
            "Passage History",
            "Sample date",
            *sera,
            "Human pool A",
            "Subclade",
            "Amino acid substitution in HA",
        ],
        ["", "REF. Ag."],
        [
            "29/30 - 1",
            "A/EXAMPLECITY/1/2029",
            "MDCK 2 +1",
            "",
            "640",
            "320",
            "40",
            "80",
            "X.1",
            "-",
        ],
        [
            "29/30-2",
            "A/EXAMPLECITY/1/2029",
            "E3 +1",
            "",
            "320",
            "1280",
            "40",
            "80",
            "X.1",
            notes[0],
        ],
        [
            "29/30 - 3",
            "A/EXAMPLEVILLE/12/2029",
            "hCK +2",
            "2029/12/01",
            "80",
            "80",
            "＜10",
            "< 10",
            "X.2",
            notes[1],
        ],
        ["", "TEST Ag."],
        [
            "29/30 - 4",
            "A/EXAMPLETOWN/7/2029",
            "AX-4 1 +hCK1",
            "2029/12/20",
            "< 10",
            "≧640",
            "20",
            "",
            "X.2",
            "",
        ],
        [
            "#29/30 - 5",
            "#A/EXAMPLETOWN/8/2029",
            "#hCK 1",
            "#2029/12/21",
            "#40",
            "#40",
            "#40",
            "",
            "#X.2",
            "",
        ],
        ["29/30 - 6", "A/EXAMPLETOWN/9/2029", "hCK 1", "2092/12/21", "40", "40", "40", "", "", ""],
        *(extra_rows or []),
        [],
        ["", "*Antigenic sites", "", "", "Fold reduction", "< 4-fold"],
    ]
    for r in rows:
        ws.append(r)
    wb.save(path)
    return path


def read(path: Path, rules: Rules) -> niid.ReadResult:
    return niid.read([path], rules, lab="LABN")


def one_table(res: niid.ReadResult):
    assert res.errors == []
    (table,) = res.tables
    return table


def test_reads_a_sheet(tmp_path, rules):
    t = one_table(read(workbook(tmp_path / "labn-20300102.xlsx"), rules))
    assert (t.group, t.subtype, t.rbc, t.date) == (
        "h3-hi-guinea-pig-labn",
        "A(H3N2)",
        "guinea-pig",
        "2030-01-02",
    )
    assert [a.lab_ids for a in t.antigens] == [
        ["LABN#29/30-1"],
        ["LABN#29/30-2"],
        ["LABN#29/30-3"],
        ["LABN#29/30-4"],
        ["LABN#29/30-6"],
    ]
    assert [a.passage for a in t.antigens] == [
        "MDCK2/MDCK1",
        "E3/E1",
        "HCK?/HCK2",
        "AX41/HCK1",
        "HCK1",
    ]
    # sera: ids as ae wrote them, the cell/egg word as the passage, the lab as an annotation
    assert [(s.name, s.serum_id, s.passage, s.reassortant, s.annotations) for s in t.sera] == [
        ("A(H3N2)/EXAMPLECITY/1/2029", "LABN CELL NO.101", "MDCK?", "", []),
        ("A(H3N2)/EXAMPLECITY/1/2029", "LABN EGG #43", "E?", "SAN-9", []),
        ("A(H3N2)/EXAMPLEVILLE/12/2029", "LABN CELL NO.1", "MDCK?", "", ["NIID"]),
    ]
    assert t.titres[2] == [["80"], ["80"], ["<10"]]  # fullwidth < and "< 10" are typing
    assert t.titres[3] == [["<10"], [">320"], ["20"]]  # ≧640 by rule
    assert t.dropped["antigens: commented out by the lab (#)"] == 1
    assert t.dropped["sera: control"] == 1
    assert t.antigens[4].date is None  # 2092: after the test, left out and warned
    assert any("after the test" in w for w in t.warnings)
    assert any("no year; 2029 from its antigen" in w for w in t.warnings)
    assert not any("two-digit year" in w for w in t.warnings)  # headers always abbreviate


def test_test_date_must_be_the_file_date(tmp_path, rules):
    res = read(workbook(tmp_path / "labn-20300103.xlsx"), rules)
    assert res.tables == [] and "not the file's date 2030-01-03" in res.errors[0]


def test_test_date_in_another_order_is_read_as_the_file_date(tmp_path, rules):
    t = one_table(
        read(workbook(tmp_path / "labn-20300102.xlsx", test_date="HI test date: 01/02/2030"), rules)
    )
    assert t.date == "2030-01-02"
    assert any("the file's date" in w for w in t.warnings)


def test_a_column_of_titres_that_is_not_a_serum_is_an_error(tmp_path, rules):
    path = workbook(tmp_path / "labn-20300102.xlsx", annotations=["160", "160"])
    res = read(path, rules)
    assert res.tables == []
    assert "holds titres but does not read as a serum" in res.errors[0]


def test_titres_without_an_id_are_an_error(tmp_path, rules):
    extra = [["", "A/EXAMPLETOWN/10/2029", "hCK 1", "", "40", "40", "40"]]
    res = read(workbook(tmp_path / "labn-20300102.xlsx", extra_rows=extra), rules)
    assert "no NIID-ID" in res.errors[0]


def test_serum_without_a_year_and_no_antigen_is_an_error(tmp_path, rules):
    sera = [*SERA[:2], "EXAMPLENOWHERE/5 Cell No.7"]
    res = read(workbook(tmp_path / "labn-20300102.xlsx", sera=sera), rules)
    assert "has no year and the antigens give none" in res.errors[0]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("EXAMPLECITY /123/29 Cell No.101", ("EXAMPLECITY/123/29", "CELL NO.101", "", "", "")),
        ("EXAMPLECITY/ 9/29 Egg SAN-9 #43", ("EXAMPLECITY/9/29", "EGG #43", "", "SAN-9", "")),
        (
            "EXAMPLEVILLE /1/29 (IVR-99) E3/D9/SpE1/E6 NIID- 2029-101",
            ("EXAMPLEVILLE/1/29", "NO.2029-101", "E3/D9/SpE1/E6", "IVR-99", ""),
        ),
        (
            "156K EXAMPLECITY /12/29 Cell NIID No.1",
            ("EXAMPLECITY/12/29", "CELL NO.1", "", "", "NIID"),
        ),
        (
            "EXAMPLETON/ 10/98 Cell&Egg No.99-2",
            ("EXAMPLETON/10/98", "CELL&EGG NO.99-2", "", "", ""),
        ),
        ("EXAMPLECITY /1/29 Cell CDC No. 343", ("EXAMPLECITY/1/29", "CELL NO.343", "", "", "CDC")),
        ("EXAMPLETON/ 7/29 pdm X-99A Egg No.1", ("EXAMPLETON/7/29", "EGG NO.1", "", "X-99A", "")),
        (
            "EXAMPLEPUR/ EXAMPLEAB 12345/29 Cell No.112",
            ("EXAMPLEPUR/EXAMPLEAB12345/29", "CELL NO.112", "", "", ""),
        ),
        (
            "EXAMPLECITY/ 1/ 29 Cell NIID 2029-1234",
            ("EXAMPLECITY/1/29", "CELL NO.2029-1234", "", "", "NIID"),
        ),
    ],
)
def test_serum_headers(text, expected):
    h = niid.parse_serum_header(text)
    assert h is not None
    assert (h.name, h.serum_id, h.passage, h.reassortant, h.lab) == expected


@pytest.mark.parametrize(
    "text", ["Subclade", "Amino acid substitution in HA", "HA group", "Clade (new)"]
)
def test_annotation_headers_are_not_sera(text):
    assert niid.parse_serum_header(text) is None


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("MDCK 2 +1", "MDCK2/MDCK1"),
        ("hCK +2", "HCK?/HCK2"),
        ("AX-4 1 +hCK1", "AX41/HCK1"),
        ("E0 +2 (Am1Al1)", "E0/E2(AM1AL1)"),
        ("E2(Am1AL1) /SpE1", "E2(AM1AL1)/SPE1"),
        ("MDCKx/1 +2", "MDCK?/MDCK1/MDCK2"),
        ("S1C4/C2 +hCK1", "SIAT1MDCK4/MDCK2/HCK1"),
        ("C3/C3E2 +2", "MDCK3/MDCK3E2/E2"),
        ("E3/D8+1", "E3/D8/D1"),
        ("hCK 0 +1 +MDCK 1", "HCK0/HCK1/MDCK1"),
    ],
)
def test_passages(rules, raw, expected):
    parser = PassageParser(rules.passage_tokens, "LABN")
    assert parser.parse(niid.niid_passage(raw, parser)).text == expected


def test_passage_count_with_nothing_to_repeat(rules):
    parser = PassageParser(rules.passage_tokens, "LABN")
    with pytest.raises(ValueError, match="nothing before"):
        niid.niid_passage("+1", parser)
