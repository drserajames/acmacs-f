"""The titre check on renamed names: a serum rename may leave its thresholds blank."""

from __future__ import annotations

from af.tables import aliases
from af.tables.model import Antigen, Serum, Table
from af.tables.rules import Rules

from .synthetic_rules import META, write_rules

SERUM_ROW = "LABX\tB\tserum\texact\tB/EXAMPLE/2/30\tB/EXAMPLE/2/2030\t{}\t{}" + META + "yes"


def rules_with(tmp_path, row: str) -> Rules:
    directory = write_rules(tmp_path / "rules")
    path = directory / "strain_aliases.tsv"
    path.write_text(path.read_text() + row + "\n")
    return Rules(directory)


def table_with_renamed_serum(where: str, readings: list[str]) -> Table:
    serum = Serum("B/EXAMPLE/2/2030", "B/EXAMPLE/2/30", source={aliases.SOURCE_KEY: where})
    antigens = [Antigen(f"B/EXAMPLE/{i}/2030", f"B/EXAMPLE/{i}/2030") for i in range(10, 13)]
    return Table(
        table_id="bvic-hi-turkey-labx-20300101",
        group="bvic-hi-turkey-labx",
        lab="LABX",
        subtype="B",
        lineage="VICTORIA",
        assay="HI",
        rbc="turkey",
        date="2030-01-01",
        date_suffix=1,
        source_key="labx-20300101.xlsx",
        antigens=antigens,
        sera=[serum],
        titres=[[[r]] for r in readings],
    )


def test_a_serum_rename_with_blank_thresholds_is_not_titre_checked(tmp_path):
    rules = rules_with(tmp_path, SERUM_ROW.format("", ""))
    where = rules.strain_aliases.rules[-1].where
    table = table_with_renamed_serum(where, ["<10", "<10", "20"])
    assert aliases.check_titres(table, rules) == []


def test_a_serum_rename_with_thresholds_is_still_checked(tmp_path):
    rules = rules_with(tmp_path, SERUM_ROW.format("40", "0.5"))
    where = rules.strain_aliases.rules[-1].where
    table = table_with_renamed_serum(where, ["<10", "<10", "20"])
    (problem,) = aliases.check_titres(table, rules)
    assert f"renamed by {where}" in problem and "0/3" in problem
