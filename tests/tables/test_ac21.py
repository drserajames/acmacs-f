"""AC Excel format 2.1 workbooks (CNIC's format), on synthetic sheets with invented names."""

from __future__ import annotations

import json
import lzma
from pathlib import Path

import pytest

from af.tables.locations import ChineseLocations
from af.tables.rules import Rules

openpyxl = pytest.importorskip("openpyxl")
from af.tables import ac21  # noqa: E402

CJK_PLACE = "测试"  # an invented two-character place name
CJK_KEPT = "样本"  # one locdb names but gives no spelling for


@pytest.fixture
def locations(tmp_path: Path) -> ChineseLocations:
    locdb = tmp_path / "locationdb.json.xz"
    locdb.write_bytes(
        lzma.compress(
            json.dumps(
                {
                    "replacements": {CJK_PLACE: "EXAMPLEPROVINCE", "PLAIN": "IGNORED"},
                    "names": {CJK_KEPT: "EXAMPLEKEY"},
                }
            ).encode()
        )
    )
    aliases = tmp_path / "aliases.tsv"
    aliases.write_text("# chinese\tromanised\tevidence\n")
    return ChineseLocations.load(locdb, aliases)


def workbook(
    path: Path,
    *,
    index_row: list[str] | None = None,
    antisera_names: list[str] | None = None,
    extra_antisera: bool = False,
) -> Path:
    sera = ["A/EXAMPLEREF/1/2029", "A/EXAMPLEREF/2/2029", "A/EXAMPLEREF/3/2029"]
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "2030-1-2"
    rows = [
        ["", "AC Excel format 2.1"],
        ["", "Test date", "2030-01-02"],
        ["", "Tested by (Lab)", "LABX"],
        ["", "Tested by (Person)", "XX"],
        ["", "Assay (HI, VN, etc)", "HI"],
        ["", "RBC species", "Guinea pig "],
        ["", "RBC ID", "x"],
        ["", "Default flu type", "H3N2"],
        ["", "Title / comment", "x"],
        [
            "",
            "TITERS",
            "Strain",
            "Specimen Date",
            *(index_row or ["1", "2", "3"]),
            "Neg.",
            "HA",
            "Back",
            "Passage",
        ],
        ["", "Reference antigens", "ID", "", *sera, "", "", "", "传代史"],
        [
            "1",
            "A/EXAMPLEREF/01/2029",
            "2029-1",
            "2029-05-01",
            "640",
            "80",
            "40",
            "",
            "64",
            "",
            "E3+1",
        ],
        [
            "2",
            f"A/{CJK_PLACE}/7/2030",
            "2030-2",
            "2029-12-01",
            "*",
            "5",
            "160",
            "",
            "64",
            "",
            "C1S2+2",
        ],
        ["", "Test antigens"],
        [
            "3",
            f"А/{CJK_KEPT}/8/29",
            "2030-3",
            "2029-11-01",
            "20",
            "40",
            "<10",
            "",
            "64",
            "",
            "S1+C1",
        ],
        ["", ""],
        ["", "ANTISERA", "Serum", "Date"],
        ["", "Name", "ID", ""],
    ]
    names = antisera_names or sera
    rows += [[str(i + 1), n, f"90{i + 1}", "2029-01-01"] for i, n in enumerate(names)]
    if extra_antisera:
        rows.append(["4", "A/EXAMPLEREF/4/2029", "904", "2029-01-01"])
    for r in rows:
        ws.append(r)
    wb.save(path)
    return path


def read(path: Path, rules: Rules, locations: ChineseLocations) -> ac21.ReadResult:
    return ac21.read([path], rules, locations, lab="LABX")


def test_reads_a_sheet(tmp_path, rules, locations):
    res = read(workbook(tmp_path / "t.xlsx"), rules, locations)
    assert res.errors == [], res.errors
    (t,) = res.tables
    assert (t.group, t.date, t.lab) == ("h3-hi-guinea-pig-labx", "2030-01-02", "LABX")
    names = [a.name for a in t.antigens]
    assert names == [
        "A(H3N2)/EXAMPLEREF/1/2029",
        "A(H3N2)/EXAMPLEPROVINCE/7/2030",
        f"A(H3N2)/{CJK_KEPT}/8/2029",
    ]
    assert [a.passage for a in t.antigens] == ["E3/E1", "MDCK1SIAT2/SIAT2", "SIAT1/MDCK1"]
    assert [s.serum_id for s in t.sera] == ["LABX 901", "LABX 902", "LABX 903"]
    assert t.titres[0] == [["640"], ["80"], ["40"]]
    assert t.titres[1][0] == []  # '*' is not tested (a rule in the test rules)
    joined = " ".join(t.warnings)
    assert (
        "Cyrillic letter" in joined
        and "two-digit year" in joined
        and "Chinese location kept" in joined
    )


def test_index_row_that_reorders_columns_is_followed(tmp_path, rules, locations):
    path = workbook(tmp_path / "t.xlsx", index_row=["1", "3", "2"])
    (t,) = read(path, rules, locations).tables
    assert [s.serum_id for s in t.sera] == ["LABX 901", "LABX 903", "LABX 902"]
    assert any("reorders the columns" in w for w in t.warnings)


def test_column_and_antisera_names_must_agree(tmp_path, rules, locations):
    names = ["A/EXAMPLEREF/1/2029", "A/EXAMPLEOTHER/9/2029", "A/EXAMPLEREF/3/2029"]
    res = read(workbook(tmp_path / "t.xlsx", antisera_names=names), rules, locations)
    assert res.tables == [] and "column 2 says" in res.errors[0]


def test_extra_antisera_rows_are_counted(tmp_path, rules, locations):
    (t,) = read(workbook(tmp_path / "t.xlsx", extra_antisera=True), rules, locations).tables
    assert t.dropped["sera: in ANTISERA without a titre column"] == 1


def test_unmapped_chinese_location_is_an_error(tmp_path, rules, locations):
    path = workbook(tmp_path / "t.xlsx")
    wb = openpyxl.load_workbook(path)
    wb.active.cell(12, 2, "A/未知/1/2030")  # a place no source maps
    wb.save(path)
    res = read(path, rules, locations)
    assert res.tables == [] and "in no alias file and not in locationdb" in res.errors[0]


def test_wrong_lab_is_refused(tmp_path, rules, locations):
    res = ac21.read([workbook(tmp_path / "t.xlsx")], rules, locations, lab="OTHER")
    assert res.tables == [] and "OTHER" in res.errors[0]
