"""Lab id shapes (af.tables.ids), on invented tables."""

from __future__ import annotations

from pathlib import Path

from af.tables import ids
from af.tables.model import Antigen, Serum, Table
from af.tables.rules import Rules

from .synthetic_rules import write_rules

SHAPES = (
    "lab\tfield\tkind\tpattern\tdate_from\tdate_to\tevidence\tadded_by\tadded_on\toptional\n"
    "LABX\tantigen\tregex\t\\d{4}-\\d{3,6}\t2000-01-01\t2099-12-31\tinvented\ttest\t2030-01-01\t\n"
    "LABX\tserum\tregex\t\\d{6}\t2000-01-01\t2099-12-31\tinvented\ttest\t2030-01-01\t\n"
    "LABX\tantigen\tregex\tOLD#\\d+\t2010-01-01\t2012-12-31\tinvented\ttest\t2030-01-01\tyes\n"
)


def table(date: str, antigen_ids: list[str], serum_ids: list[str]) -> Table:
    return Table(
        table_id=f"t-{date}",
        group="h3-hi-turkey-labx",
        lab="LABX",
        subtype="A(H3N2)",
        lineage="",
        assay="HI",
        rbc="turkey",
        date=date,
        date_suffix=0,
        source_key="x",
        antigens=[
            Antigen(name=f"A/EXAMPLETOWN/{i}/2029", raw_name="", lab_ids=[f"LABX#{v}"] if v else [])
            for i, v in enumerate(antigen_ids, 1)
        ],
        sera=[
            Serum(name="A/EXAMPLETOWN/1/2029", raw_name="", serum_id=f"LABX {v}") for v in serum_ids
        ],
        titres=[[["40"] for _ in serum_ids] for _ in antigen_ids],
    )


def rules(tmp_path: Path) -> Rules:
    d = write_rules(tmp_path / "rules")
    (d / "id_shapes.tsv").write_text(SHAPES)
    return Rules(d)


def test_shapes_are_checked_by_field_and_date(tmp_path):
    r = rules(tmp_path)
    tables = [
        table("2030-01-02", ["2029-12345", "300142", ""], ["300142", "2029-1"]),
        table("2011-05-05", ["OLD#12"], ["300001"]),
    ]
    counts, lines = ids.check(tables, r)
    assert counts["LABX antigen ok"] == 2  # the year-serial one and the dated old format
    assert counts["LABX antigen id of an unexpected shape"] == 1  # a serum code as an antigen id
    assert counts["LABX antigen id missing"] == 1
    assert counts["LABX serum id of an unexpected shape"] == 1
    assert any("'300142' fits no id_shapes row" in line for line in lines)


def test_no_rows_no_check(tmp_path):
    counts, lines = ids.check(
        [table("2030-01-02", ["anything"], ["x"])], Rules(write_rules(tmp_path / "r"))
    )
    assert not counts and not lines
