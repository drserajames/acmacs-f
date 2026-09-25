import math
from typing import Any

import pytest

from af.serology.rows import TableFormatError, rows_from_table
from af.serology.titre import TitreError, parse_reading
from af.tables.model import Table


@pytest.mark.parametrize(
    ("raw", "kind", "log"),
    [("40", "num", 2.0), ("<10", "lt", 0.0), (">1280", "gt", 7.0), ("~80", "dodgy", 3.0)],
)
def test_parse_reading(raw: str, kind: str, log: float) -> None:
    reading = parse_reading(raw)
    assert reading.kind == kind
    assert math.isclose(reading.log, log)


@pytest.mark.parametrize("raw", ["", "*", "abc", "<", "0", "-40", "nan"])
def test_bad_reading_is_an_error(raw: str) -> None:
    with pytest.raises(TitreError):
        parse_reading(raw)


def _two_by_two(syn: Any, table_id: str = "t-2021-03-04") -> Table:
    antigens = [
        {"name": syn.virus("Somewhere", 1), "passage": "MDCK1", "date": "2021-01-05"},
        {"name": syn.virus("Somewhere", 1), "passage": "E3", "annotations": ["DISTINCT"]},
    ]
    sera = [
        {"name": syn.virus("Elsewhere", 2), "serum_id": "S-1", "passage": "E4"},
        {"name": syn.virus("Elsewhere", 3), "serum_id": ""},
    ]
    titres = [[["40", "80"], []], [["<10"], [">1280"]]]
    result: Table = syn.table(table_id, antigens, sera, titres)
    return result


def test_rows_keep_every_reading_and_count_missing_identity(syn: Any) -> None:
    rows = rows_from_table(_two_by_two(syn), syn.rules)
    assert len(rows.antigens) == 2 and len(rows.sera) == 2
    # four readings: two in one cell, none in the untested cell
    assert [(t["antigen_position"], t["serum_position"], t["reading"]) for t in rows.titres] == [
        (0, 0, 0),
        (0, 0, 1),
        (1, 0, 0),
        (1, 1, 0),
    ]
    assert rows.without_identity == {"antigens": 1, "sera": 1}
    # an antigen without identity gets a table-scoped key; one with identity a hash
    assert rows.antigens[1]["antigen_key"] == "t-2021-03-04#a1"
    assert len(rows.antigens[0]["antigen_key"]) == 16


def test_same_identity_same_key_across_tables(syn: Any) -> None:
    a = rows_from_table(_two_by_two(syn), syn.rules)
    b = rows_from_table(_two_by_two(syn, "t-2021-03-05"), syn.rules)
    assert a.antigens[0]["antigen_key"] == b.antigens[0]["antigen_key"]
    assert a.antigens[1]["antigen_key"] != b.antigens[1]["antigen_key"]


def test_wrong_shape_is_refused(syn: Any) -> None:
    bad = _two_by_two(syn)
    bad.titres = bad.titres[:1]
    with pytest.raises(TableFormatError, match="titre rows"):
        rows_from_table(bad, syn.rules)
    bad = _two_by_two(syn)
    bad.titres[1] = bad.titres[1][:1]
    with pytest.raises(TableFormatError, match="titre row 1"):
        rows_from_table(bad, syn.rules)


def test_harvest_date_is_part_of_identity(syn: Any) -> None:
    """af.tables keeps the harvest date apart; identity compares passage *with* the date."""

    def one(table_id: str, harvested: str | None) -> Table:
        antigen = {"name": syn.virus("Somewhere", 1), "passage": "SIAT1", "passage_date": harvested}
        serum = {"name": syn.virus("Elsewhere", 2), "serum_id": "S-1"}
        result: Table = syn.table(table_id, [antigen], [serum], [[["40"]]])
        return result

    a = rows_from_table(one("t1", "2021-01-01"), syn.rules).antigens[0]
    b = rows_from_table(one("t2", "2021-01-01"), syn.rules).antigens[0]
    c = rows_from_table(one("t3", "2021-02-02"), syn.rules).antigens[0]
    assert a["identity_passage"] == "SIAT1 (2021-01-01)"
    assert (a["passage"], a["passage_date"]) == ("SIAT1", "2021-01-01")
    assert a["antigen_key"] == b["antigen_key"] != c["antigen_key"]
