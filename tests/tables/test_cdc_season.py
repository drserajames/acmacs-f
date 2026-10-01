"""CDC per-season titre files, on synthetic rows."""

from __future__ import annotations

from pathlib import Path

from af.tables import cdc_season
from af.tables.rules import Rules

from .synthetic_rules import write_rules


def season_row(**kw: str) -> dict[str, str]:
    base = dict.fromkeys(cdc_season.COLUMNS, "")
    base.update(
        {
            "virus_cdc_id": "100",
            "virus_strain": "A/EXAMPLETOWN/01/2029",
            "virus_collection_date": "2029-05-01",
            "virus_strain_passage": "C1S1",
            "serum_strain": "A/EXAMPLEREF/2/2028",
            "ferret_id": "F0-1;F0-2",
            "lot #": "T28-001;T28-002",
            "serum_antigen_passage": "E3",
            "assay_date": "2029-08-07",
            "subtype": "H1 swl",
            "titer": "640",
            "assay-type": "HI",
        }
    )
    base.update(kw)
    return base


def write(path: Path, rows: list[dict[str, str]]) -> Path:
    path.write_text(
        "\n".join(
            ["\t".join(cdc_season.COLUMNS)]
            + ["\t".join(r[c] for c in cdc_season.COLUMNS) for r in rows]
        )
        + "\n"
    )
    return path


def test_reads_only_the_rule_scope_and_says_flags_are_absent(tmp_path):
    rows = [
        season_row(),
        season_row(titer="5", serum_strain="A/EXAMPLEREF/3/2028", **{"lot #": "T28-003"}),
        season_row(assay_date="2029-07-31"),
        season_row(subtype="B vic", virus_strain="B/EXAMPLEB/1/2029"),
    ]
    res = cdc_season.read(
        write(tmp_path / "season.tsv", rows), Rules(write_rules(tmp_path / "rules"))
    )
    assert res.errors == []
    (t,) = res.tables
    assert (t.group, t.date, t.meta["not_for_use_flags"]) == (
        "h1pdm-hi-turkey-cdc",
        "2029-08-07",
        "absent in source",
    )
    assert t.sera[0].serum_id == "CDC T28-001,T28-002" and t.antigens[0].passage == "MDCK1SIAT1"
    assert t.antigens[0].passage_date is None
    assert t.titres == [[["640"], ["<10"]]]
    assert res.dropped["rows: outside season_files scope"] == 2
    assert res.dropped["rows: read with no not-for-use flags (absent in source)"] == 2


def test_file_without_a_rule_is_an_error(tmp_path):
    import pytest

    with pytest.raises(cdc_season.CDCFormatError, match="no season_files rule"):
        cdc_season.read(
            write(tmp_path / "other.tsv", [season_row()]), Rules(write_rules(tmp_path / "rules"))
        )


def tsv_table(
    date: str, antigens: list[tuple[str, str, str]], sera: list[tuple[str, str, str, str]]
):
    """A main-TSV table reduced to what the join reads: names, passages, harvest dates, lots."""
    from af.tables.model import Antigen, Serum, Table

    return Table(
        table_id="",
        group="h1pdm-hi-turkey-cdc",
        lab="CDC",
        subtype="A(H1N1)",
        lineage="",
        assay="HI",
        rbc="turkey",
        date=date,
        date_suffix=0,
        source_key=f"CDC test_id {date}",
        antigens=[Antigen(name=n, raw_name=n, passage=p, passage_date=d) for n, p, d in antigens],
        sera=[
            Serum(name=n, raw_name=n, passage=p, passage_date=d, serum_id=lot)
            for n, p, d, lot in sera
        ],
        titres=[[["640"] for _ in sera] for _ in antigens],
    )


REF = "A(H1N1)/EXAMPLEREF/2/2028"
TOWN = "A(H1N1)/EXAMPLETOWN/1/2029"


def joined(tmp_path, rows, tsv):
    res = cdc_season.read(
        write(tmp_path / "season-join.tsv", rows), Rules(write_rules(tmp_path / "rules"))
    )
    assert res.errors == []
    counts = cdc_season.join_harvest_dates(res.tables, tsv)
    return res.tables, counts


def test_harvest_dates_join_the_tsv_points(tmp_path):
    rows = [
        season_row(virus_strain="A/EXAMPLEREF/2/2028", virus_strain_passage="E3", virus_cdc_id="7"),
        season_row(),  # a test virus the TSV never has
    ]
    tsv = [
        tsv_table(
            "2029-10-01",
            [(REF, "E3", "2027-01-23"), (REF, "E3", "2029-09-01")],  # re-harvested after the test
            [(REF, "E3", "2027-02-01", "CDC T28-001,T28-002")],
        )
    ]
    (t,), counts = joined(tmp_path, rows, tsv)
    ref = next(a for a in t.antigens if a.name == REF)
    town = next(a for a in t.antigens if a.name == TOWN)
    # several harvest dates: the latest on or before the season test (2029-08-07)
    assert ref.passage_date == "2027-01-23"
    assert "latest of 2" in ref.source["passage_date_from"]
    assert town.passage_date is None
    assert t.sera[0].passage_date == "2027-02-01"  # same name, passage and lot
    assert counts == {
        "antigens: harvest date from the CDC TSV": 1,
        "antigens: no harvest date in the main tables": 1,
        "sera: harvest date from the CDC TSV": 1,
    }


def test_no_join_on_another_lot_or_a_later_harvest_or_without_the_rule(tmp_path):
    rows = [season_row(virus_strain="A/EXAMPLEREF/2/2028", virus_strain_passage="E3")]
    tsv = [
        tsv_table(
            "2029-10-01",
            [(REF, "E3", "2029-09-01")],  # only harvested after the season test
            [(REF, "E3", "2027-02-01", "CDC T28-999")],  # another lot
        )
    ]
    (t,), counts = joined(tmp_path, rows, tsv)
    assert t.antigens[0].passage_date is None and t.sera[0].passage_date is None
    assert counts == {
        "antigens: no harvest date in the main tables": 1,
        "sera: no harvest date in the main tables": 1,
    }
    plain = cdc_season.read(
        write(tmp_path / "season.tsv", rows), Rules(write_rules(tmp_path / "rules2"))
    )
    assert cdc_season.join_harvest_dates(plain.tables, tsv) == {}


def test_an_unknown_harvest_dates_value_is_an_error(tmp_path):
    rules_dir = write_rules(tmp_path / "rules")
    path = rules_dir / "season_files.tsv"
    path.write_text(path.read_text().replace("\tfrom-tsv", "\tfrom-somewhere"))
    res = cdc_season.read(write(tmp_path / "season-join.tsv", [season_row()]), Rules(rules_dir))
    assert any("harvest_dates 'from-somewhere'" in e for e in res.errors)


def test_a_file_with_no_lot_column_takes_lots_from_the_main_tables(tmp_path):
    """CDC's 2015-16 file names sera by ferret id only."""
    columns = cdc_season.COLUMNS_NO_LOT
    rows = [
        season_row(virus_strain="A/EXAMPLEREF/2/2028", virus_strain_passage="E3", virus_cdc_id="7"),
        season_row(ferret_id="F0-9", serum_strain="A/EXAMPLETOWN/1/2029"),  # in no main table
    ]
    path = tmp_path / "season-join.tsv"
    path.write_text(
        "\n".join(["\t".join(columns)] + ["\t".join(r[c] for c in columns) for r in rows]) + "\n"
    )
    res = cdc_season.read(path, Rules(write_rules(tmp_path / "rules")))
    assert res.errors == []
    (t,) = res.tables
    assert [s.serum_id for s in t.sera] == ["", ""]  # no lot yet
    main = [
        tsv_table("2029-06-01", [], [(REF, "E3", "2027-02-01", "CDC T28-005")]),
        tsv_table("2029-08-01", [], [(REF, "E3", "2027-02-01", "CDC T28-001,T28-002")]),
        tsv_table("2029-09-01", [], [(REF, "E3", "2027-02-01", "CDC T28-009")]),  # after it
    ]
    counts = cdc_season.join_lots(res.tables, main)
    assert [s.serum_id for s in t.sera] == ["CDC T28-001,T28-002", ""]  # the latest before
    assert t.sera[0].source["lot_from"] == "main table 2029-08-01, name+passage"
    assert counts == {
        "sera: lot from the main tables": 1,
        "sera: no lot (none, or two, in the main tables)": 1,
    }
    cdc_season.join_harvest_dates(res.tables, main)
    assert t.sera[0].passage_date == "2027-02-01"  # and then, with its lot, the harvest date
