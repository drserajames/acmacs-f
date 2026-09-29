"""A source configured to stand in for a workbook keeps that workbook's table id."""

from __future__ import annotations

import dataclasses

from af.tables import identity
from af.tables.model import Antigen, Serum, Table


def table(source_key: str, meta: dict | None = None) -> Table:
    return Table(
        table_id="",
        group="bvic-hi-turkey-labc",
        lab="LABC",
        subtype="B",
        lineage="VICTORIA",
        assay="HI",
        rbc="turkey",
        date="2030-01-02",
        date_suffix=0,
        source_key=source_key,
        antigens=[Antigen(name="B/EXAMPLEVILLE/1/2029", raw_name="B/Exampleville/1/2029")],
        sera=[Serum(name="B/EXAMPLEVILLE/1/2029", raw_name="B/Exa 1/29")],
        titres=[[["640"]]],
        meta=meta or {},
    )


def previous_with(t: Table) -> identity.Manifest:
    identity.assign([t], None)
    return identity.Manifest.from_tables([t], [])


def test_the_replacing_sheet_keeps_the_id():
    old = table("LABC xlsx labc-20300102.xlsx [BX 020130]")
    previous = previous_with(old)
    new = table("LABC xlsx LABC compilation.xlsx [020130]", {"replaces": "labc-20300102.xlsx"})
    assert identity.assign([new], previous) == []
    assert new.table_id == "bvic-hi-turkey-labc-20300102"


def test_without_replaces_it_is_a_new_table():
    previous = previous_with(table("LABC xlsx labc-20300102.xlsx [BX 020130]"))
    new = table("LABC xlsx LABC compilation.xlsx [020130]")
    identity.assign([new], previous)
    assert new.table_id == "bvic-hi-turkey-labc-20300102.2"


def test_no_takeover_while_the_replaced_workbook_is_still_read():
    old = table("LABC xlsx labc-20300102.xlsx [BX 020130]")
    previous = previous_with(old)
    again = dataclasses.replace(old, table_id="", date_suffix=0)
    new = table("LABC xlsx LABC compilation.xlsx [020130]", {"replaces": "labc-20300102.xlsx"})
    identity.assign([again, new], previous)
    assert again.table_id == "bvic-hi-turkey-labc-20300102"
    assert new.table_id == "bvic-hi-turkey-labc-20300102.2"
