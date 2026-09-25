"""Vaccine selection by name + passage class, on invented strains."""

import pytest

from af.map.vaccines import (
    MapAntigen,
    VaccineChoice,
    VaccineDisable,
    VaccineRow,
    VaccineRuleError,
    passage_class,
    read_org_vaccine_tables,
    select_vaccines,
    strain_name,
)


def strain(place: str, number: int, year: int) -> str:
    """Invented strain names, generated rather than written out (see tools/WHO-DATA-GATE.md)."""
    return "/".join((place, str(number), str(year)))


TV1 = strain("TESTVILLE", 1, 2020)
OT7 = strain("OLDTOWN", 7, 2010)
NW3 = strain("NOWHERE", 3, 2022)
NT1 = strain("NEVERTESTED", 1, 2019)
GH1 = strain("GHOST", 1, 2000)
TIE2 = strain("TIEVILLE", 2, 2021)


@pytest.mark.parametrize(
    ("passage", "reassortant", "expected"),
    [
        ("E3/E2 (2020-01-01)", "", "egg"),
        ("E3/SPE1", "", "egg"),  # egg even though the last step is an SPF egg token
        ("MDCK1/MK2", "", "cell"),  # cell even though the last step is monkey kidney
        ("SIAT1MDCK4/MDCK2", "", "cell"),
        ("E4", "NYMC-999", "reassortant"),
        ("QMC2/SIAT1", "", "cell"),
        ("VW10000001", "", None),  # a specimen id, not a passage
        ("", "", None),
    ],
)
def test_passage_class(passage: str, reassortant: str, expected: str | None) -> None:
    assert passage_class(passage, reassortant) == expected


def test_strain_name_strips_subtype_prefix() -> None:
    assert strain_name("A(H3N2)/" + TV1) == TV1
    assert strain_name("B/" + TV1) == TV1


def ag(index: int, name: str, passage: str, tables: int = 1, reassortant: str = "") -> MapAntigen:
    return MapAntigen(index, name, passage, reassortant, tables, True)


ANTIGENS = [
    ag(0, "A(H3N2)/" + TV1, "MDCK1 (2020-02-01)", tables=3),
    ag(1, "A(H3N2)/" + TV1, "MDCK2 (2021-02-01)", tables=9),
    ag(2, "A(H3N2)/" + TV1, "E3 (2020-03-01)", tables=2),
    ag(3, "A(H3N2)/" + TV1, "E4 (2020-05-01)", tables=2),
    ag(4, "A(H3N2)/" + OT7, "E2", tables=5),
    ag(5, "A(H3N2)/" + NW3, "VW10000001", tables=1),
]
ROWS = [
    VaccineRow(TV1, None, "202109"),
    VaccineRow(OT7, "egg", "201102"),
    VaccineRow(NT1, "cell", "202002"),
]


def test_most_tables_then_latest_date() -> None:
    report = select_vaccines(ANTIGENS, ROWS)
    marks = {(m.row.name, m.passage_class): m.antigen for m in report.marks}
    assert marks[(TV1, "cell")] == 1  # 9 tables beats 3
    assert marks[(TV1, "egg")] == 3  # tie on tables, no table dates: later passage
    assert marks[(OT7, "egg")] == 4
    assert (NT1, "cell") in report.rows_without_antigen
    assert report.unclassified_passages == 1


def test_disable_and_choose_rules() -> None:
    report = select_vaccines(
        ANTIGENS,
        ROWS,
        disable=[VaccineDisable(OT7, "any", "superseded")],
        choose=[VaccineChoice(TV1, "cell", "MDCK1", "the reference preparation")],
    )
    marks = {(m.row.name, m.passage_class): m for m in report.marks}
    assert (OT7, "egg") not in marks
    assert report.disabled == [(OT7, "egg", "superseded")]
    assert marks[(TV1, "cell")].antigen == 0
    assert marks[(TV1, "cell")].chosen_by == "the reference preparation"


def test_dead_rules_are_errors() -> None:
    with pytest.raises(VaccineRuleError, match="disable"):
        select_vaccines(ANTIGENS, ROWS, disable=[VaccineDisable(GH1, "any", "x")])
    with pytest.raises(VaccineRuleError, match="matches 0"):
        select_vaccines(ANTIGENS, ROWS, choose=[VaccineChoice(TV1, "cell", "SIAT9", "x")])


ORG_TEXT = f'''
sData = {{
    "A(H3N2)": org_table_to_dict("""
# -*- Org -*-
| name             | passage | surrogate | year   | comment |
|------------------+---------+-----------+--------+---------|
| {TV1} |         |           | 202109 |         |
| {OT7}   | egg     | True      | 201102 | note    |
# -*-
        """),
}}
'''


def test_read_org_vaccine_tables() -> None:
    tables = read_org_vaccine_tables(ORG_TEXT)
    assert tables["A(H3N2)"] == [
        VaccineRow(TV1, None, "202109", False),
        VaccineRow(OT7, "egg", "201102", True),
    ]


def test_read_org_vaccine_tables_rejects_unknown_passage() -> None:
    with pytest.raises(ValueError, match="unknown passage"):
        read_org_vaccine_tables(ORG_TEXT.replace("| egg     |", "| eggs    |"))


def test_tie_broken_by_latest_table_before_passage_date() -> None:
    import datetime as dt

    early_table = MapAntigen(
        0, "A(H3N2)/" + TIE2, "SIAT1 (2022-01-01)", "", 4, True, dt.date(2021, 5, 1)
    )
    late_table = MapAntigen(
        1, "A(H3N2)/" + TIE2, "SIAT2 (2021-01-01)", "", 4, True, dt.date(2023, 5, 1)
    )
    report = select_vaccines([early_table, late_table], [VaccineRow(TIE2, "cell", "202202")])
    assert [m.antigen for m in report.marks] == [1]
