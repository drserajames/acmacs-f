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
