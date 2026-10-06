"""af.tree.report_stage: the report tree derived from a published version. Synthetic only."""

from __future__ import annotations

import dataclasses
import datetime
from collections import Counter
from pathlib import Path

import pytest

from af.seq.locations import LocationTables
from af.store import Store, StoreRef
from af.tables.model import Antigen
from af.tree import report_stage as R
from af.tree.populate import LeafRecord

from .test_report_filter import table

THROUGH = datetime.date(2026, 9, 21)


def record(country: str | None) -> LeafRecord:
    return LeafRecord("EPI_ISL_1", "EPI1", "A/EXAMPLETOWN/1/2024", "ACGT", country=country)


def test_the_name_key_ignores_location_punctuation_and_the_subtype_prefix() -> None:
    assert R.report_name_key("A/EXAMPLETOWN/12/2020") == R.report_name_key(
        "A(H1N1)/EXAMPLE-TOWN/12/2020"
    )
    assert R.report_name_key("A/EXAMPLETOWN/12/2020") != R.report_name_key("A/EXAMPLETOWN/13/2020")


def test_a_name_of_another_shape_only_matches_itself() -> None:
    assert R.report_name_key("odd name") == "unparsed:odd name"
    assert R.report_name_key("odd name") != R.report_name_key("other odd name")


def test_round_tables_keep_the_subtype_up_to_the_round_and_one_b_lineage() -> None:
    old = table(Antigen("B/EXAMPLETOWN/1/2020", "x", lineage="VICTORIA"),
                Antigen("B/EXAMPLETOWN/2/2014", "x", lineage="YAMAGATA"))  # fmt: skip
    old = dataclasses.replace(old, subtype="B", date="2026-01-01")
    late = dataclasses.replace(old, date="2026-10-01")
    other = dataclasses.replace(old, subtype="A(H1N1)")
    kept, counts = R.round_tables([old, late, other], "B", "VICTORIA", THROUGH)
    assert [len(t.antigens) for t in kept] == [1]
    assert counts == {"tables_after_through": 1, "antigens_other_lineage": 1, "tables_used": 1}


def test_no_round_tables_is_an_error() -> None:
    with pytest.raises(R.ReportStageError, match="no A\\(H3N2\\)"):
        R.round_tables([table()], "A(H3N2)", "", datetime.date(2000, 1, 1))


def locations(tmp_path: Path) -> LocationTables:
    head = "code\tspelling\tsource\tevidence\tadded_by\tadded_on\toptional\n"
    rows = [("XAA", "Exampleland"), ("XBB", "Otherland")]
    (tmp_path / "countries.tsv").write_text(
        head + "".join(f"{c}\t{s}\tgisaid\tinvented\ttest\t2026-10-06\t\n" for c, s in rows)
    )
    regions = "scheme\tcountry\tgroup\tevidence\n"
    regions += "continent\tXAA\tEUROPE\tinvented\ncontinent\tXBB\tnot assigned\tinvented\n"
    (tmp_path / "regions.tsv").write_text(regions)
    columns = "location country admin latitude longitude precision source evidence same_as flags"
    (tmp_path / "places.tsv").write_text("\t".join(columns.split()) + "\n")
    return LocationTables.read(tmp_path)


def test_continent_comes_from_the_tables_and_misses_are_counted(tmp_path: Path) -> None:
    continent_of, missing = R.continent_lookup(locations(tmp_path))
    assert continent_of(record("Exampleland")) == "EUROPE"
    assert continent_of(record("Otherland")) is None
    assert continent_of(record("Nowhereland")) is None
    assert continent_of(record(None)) is None
    assert missing == Counter(no_continent_for_country=1, unknown_country_spelling=1, no_country=1)


def test_the_source_must_be_a_named_version(tmp_path: Path) -> None:
    store = Store.create(tmp_path / "store")
    with pytest.raises(R.ReportStageError, match="never CURRENT"):
        R.pinned(store, "trees", "h1/weekly")
    with pytest.raises(R.ReportStageError, match="no such version"):
        R.pinned(store, "trees", "h1/weekly@0123456789abcdef")


@pytest.mark.parametrize("purpose", ["weekly", "report-x"])
def test_a_report_tree_never_replaces_its_source(tmp_path: Path, purpose: str) -> None:
    source = StoreRef("trees", "h1/report-x", "0123456789abcdef", "0123456789abcdef" + "0" * 48)
    with pytest.raises(R.ReportStageError, match="never replaces its source"):
        R.publish(Store.create(tmp_path), None, purpose, source, {}, [], {}, THROUGH)  # type: ignore[arg-type]
