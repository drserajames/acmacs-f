"""CDC HI workbooks, on synthetic sheets laid out as CDC lays them out (invented strains)."""

from __future__ import annotations

from pathlib import Path

import pytest

from af.tables.rules import Rules

openpyxl = pytest.importorskip("openpyxl")
from af.tables import cdc_xlsx  # noqa: E402


def workbook(
    path: Path,
    *,
    letters_above: bool = False,
    pool_serum: bool = False,
    serum_label: str = "REFERENCE ANTISERA",
    repeat: bool = False,
) -> Path:
    """A 'RUN' sheet: title, date, serum letters, antigen block, antisera block, RBC footer."""
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "RUN 1"
    ws["G1"] = "HEMAGGLUTINATION INHIBITION REACTIONS OF INFLUENZA H3 VIRUSES"
    ws["A2"] = "WITH 20nM OSELTAMIVIR, 4 HA UNITS/ 50 MICROLITERS"
    ws["C3"] = "DATE TESTED: 9/12/2030"
    letters = ["A", "B", "C"] if pool_serum else ["A", "B"]
    letter_row = 4 if letters_above else 7
    for i, letter in enumerate(letters):
        ws.cell(letter_row, 6 + i, letter)
    ws["A7"] = "REFERENCE VIRUSES"
    for col, label in [
        (2, "PASSAGE"),
        (3, "STRAIN DESIGNATION"),
        (4, "CDC ID#"),
        (5, "DATE COLLECTED"),
        (6 + len(letters), "PASSAGE"),
    ]:
        ws.cell(8, col, label)
    rows: list[tuple[int, str, str, int | None, str | None, list[int | str]]] = [
        (
            1,
            "S1(07/19/2030)<NY>",
            "A/EXAMPLETOWN/1/2030",
            9000000001,
            "2030-06-19",
            [640, "<20", 80],
        ),
        (2, "E3/E1(2/17/2030", "A/EXAMPLETOWN/2/2030", 9000000002, "2030-01-02", [160, 1280, 40]),
        (3, "", "TEST VIRUSES", None, None, []),
        (4, "S2(04/23/2030)", "A/EXAMPLEOTHER/03/2030", 9000000003, "2030-03-19", [320, 320, 20]),
        (5, "", "KIT-1820 CONTROL ANTIGEN", None, None, ["<20", "<20", "<20"]),
    ]
    if repeat:  # the test virus again on the next row, as CDC sheets sometimes list it
        rows[-1] = (
            5,
            "S2(04/23/2030)",
            "A/EXAMPLEOTHER/03/2030",
            9000000003,
            "2030-03-19",
            [640, 160, 20],
        )
    for r, (no, passage, name, cdc_id, collected, titres) in enumerate(rows, start=9):
        ws.cell(r, 1, no)
        ws.cell(r, 2, passage)
        ws.cell(r, 3, name)
        ws.cell(r, 4, cdc_id)
        ws.cell(r, 5, collected)
        for i, titre in enumerate(titres[: len(letters)]):
            ws.cell(r, 6 + i, titre)
        if passage:
            ws.cell(r, 6 + len(letters), passage.split("<")[0])
    ws.cell(14, 3, "SERUM CONTROL")
    ws.cell(16, 1, serum_label)
    for col, label in [(4, "LOT"), (5, "SPECIES"), (7, "BOOSTED"), (9, "PASSAGE"), (10, "POOL")]:
        ws.cell(16, col, label)
    sera = [
        ("A", "A/EXAMPLETOWN/1/2030", "T30-001", "N", "S1(2/14/2030)"),
        ("B", "A/EXAMPLETOWN/2/2030", "T30-002", "Y", "E3/E1(2/17/2030)"),
        ("C", "Human POOL / CSID 1,2", "29/30 H3-CELL HUMAN POOL", "N", ""),
    ][: len(letters)]
    for r, (letter, name, lot, boosted, passage) in enumerate(sera, start=17):
        for col, value in [
            (1, letter),
            (2, name),
            (4, lot),
            (5, "Ferret"),
            (7, boosted),
            (9, passage),
            (10, "N"),
        ]:
            ws.cell(r, col, value)
    ws.cell(22, 7, "RBCS USED")
    ws.cell(23, 7, "GUINEA PIG")
    wb.save(path)
    return path


def test_reads_a_run_sheet(tmp_path: Path, rules: Rules):
    res = cdc_xlsx.read([workbook(tmp_path / "run.xlsx")], rules)
    assert res.errors == []
    (t,) = res.tables
    assert (t.group, t.date, t.meta["test_protocol"]) == (
        "h3-hi-guinea-pig-cdc",
        "2030-09-12",
        "hi_oseltamivir_protocol",
    )
    assert [a.name for a in t.antigens] == [
        "A(H3N2)/EXAMPLETOWN/1/2030",
        "A(H3N2)/EXAMPLETOWN/2/2030",
        "A(H3N2)/EXAMPLEOTHER/3/2030",
    ]
    first = t.antigens[0]
    assert (first.passage, first.passage_date, first.lab_ids, first.source["site"]) == (
        "SIAT1",
        "2030-07-19",
        ["CDC#9000000001"],
        "NY",
    )
    assert first.ae_passage() == "SIAT1 (2030-07-19)"
    assert t.dropped == {"antigens: control": 1}
    assert [s.serum_id for s in t.sera] == ["CDC T30-001", "CDC T30-002"]
    assert t.sera[1].annotations == ["BOOSTED"] and t.sera[1].passage_date == "2030-02-17"
    assert t.titres[0] == [["640"], ["<20"]]
    assert any("no closing parenthesis" in w for w in t.warnings)


def test_layout_variants(tmp_path: Path, rules: Rules):
    path = workbook(
        tmp_path / "run.xlsx",
        letters_above=True,
        pool_serum=True,
        serum_label="REFERENCE ANTISERUM",
    )
    (t,) = cdc_xlsx.read([path], rules).tables
    assert len(t.sera) == 2 and t.dropped["sera: control"] == 1
    assert [row[1] for row in t.titres] == [["<20"], ["1280"], ["320"]]


def test_passage_columns_must_agree(tmp_path: Path, rules: Rules):
    path = workbook(tmp_path / "run.xlsx")
    wb = openpyxl.load_workbook(path)
    wb.active.cell(9, 8, "S2(07/19/2030)")
    wb.save(path)
    res = cdc_xlsx.read([path], rules)
    assert res.tables == [] and "PASSAGE columns disagree" in res.errors[0]


def test_serum_letters_must_match_columns(tmp_path: Path, rules: Rules):
    path = workbook(tmp_path / "run.xlsx")
    wb = openpyxl.load_workbook(path)
    wb.active.cell(18, 1, "D")
    wb.save(path)
    res = cdc_xlsx.read([path], rules)
    assert res.tables == [] and "serum letters" in res.errors[0]


def test_non_titre_sheet_is_reported_not_silently_skipped(tmp_path: Path, rules: Rules):
    path = workbook(tmp_path / "run.xlsx")
    wb = openpyxl.load_workbook(path)
    wb.create_sheet("Sheet1").append(["A/EXAMPLETOWN/1/2030", "S1"])
    wb.save(path)
    res = cdc_xlsx.read([path], rules)
    assert len(res.tables) == 1 and res.skipped_tests == [
        "run.xlsx[Sheet1]!1: no REFERENCE VIRUSES label, not a titre sheet"
    ]


def test_a_repeated_row_merges_into_one_antigen(tmp_path: Path, rules: Rules):
    """As the TSV reader does: one antigen, every row's readings (a chart cannot hold two
    points with one identity)."""
    res = cdc_xlsx.read([workbook(tmp_path / "t.xlsx", repeat=True)], rules)
    assert res.errors == []
    (table,) = res.tables
    names = [a.name for a in table.antigens]
    assert names.count("A(H3N2)/EXAMPLEOTHER/3/2030") == 1
    i = names.index("A(H3N2)/EXAMPLEOTHER/3/2030")
    assert table.titres[i] == [["320", "640"], ["160", "320"]]
    assert table.dropped["antigens: repeated rows merged"] == 1
    assert any("readings merged" in w for w in table.warnings)


def test_a_cell_fix_repairs_a_passage_date(tmp_path: Path, rules: Rules):
    """A named repair of one cell (cell_fixes), e.g. a harvest date typed day-first."""
    (table,) = cdc_xlsx.read([workbook(tmp_path / "fix.xlsx")], rules).tables
    assert table.antigens[0].passage_date == "2030-07-18"
    assert any("fixed as" in w for w in table.warnings)
