"""Vaccines coloured by their named strain's original clade, before passage mutation (Sarah, Q184).
Invented strains and records only."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import duckdb
import pytest

from af.chart.model import Antigen, Chart, Titres
from af.clades.colours import ColourEntry
from af.clades.colours import ColourScheme as CladeColourScheme
from af.clades.sequence import AlignedSequence
from af.geo.colours import BASIS_CLADE, STATE_UNOBSERVED, DotStyle
from af.map.colouring import StoreColours
from af.map.vaccine_colours import (
    StrainRecord,
    original_records,
    resolve,
    vaccine_warning,
)
from af.seq.matching import read_passage_rules
from af.serology.joins import PreparationSequence
from af.serology.outputs import AlignedSequences
from tests.map.test_colouring import caller_setup

RED = DotStyle("Clade P", "#aa0000", BASIS_CLADE)
BLUE = DotStyle("Clade P.1", "#0000aa", BASIS_CLADE)


def rec(n: int, cls: str, evidence: str | None = None) -> StrainRecord:
    return StrainRecord(f"EPI_{n}", f"A{n}", cls, "P", evidence)


def test_the_original_record_comes_first_then_cell_never_egg() -> None:
    egg, cell, original = rec(1, "egg"), rec(2, "cell"), rec(3, "original")
    mixed, unknown = rec(4, "mixed"), rec(5, "unknown")
    assert original_records([egg, cell, original]) == ("original", [original])
    assert original_records([egg, cell, mixed]) == ("cell", [cell])
    assert original_records([egg, mixed, unknown]) == ("", [])


def test_unobserved_then_contradicted_records_abstain_and_the_rest_agree() -> None:
    supported = rec(1, "cell", "supported")
    contradicted = rec(2, "cell", "contradicted")
    unobserved = rec(3, "cell", "unobserved")
    unseen = DotStyle("", None, state=STATE_UNOBSERVED)
    assert resolve([(supported, RED), (contradicted, BLUE), (unobserved, unseen)]) == (RED, "")
    assert resolve([(contradicted, BLUE), (unobserved, unseen)]) == (BLUE, "")
    style, why = resolve([(rec(4, "cell", "supported"), RED), (supported, BLUE)])
    assert style is None and why == "records disagree (Clade P 1, Clade P.1 1)"
    style, why = resolve([(unobserved, unseen)])
    assert style is None and why == f"records give it no colour ({STATE_UNOBSERVED})"


def test_the_warning_is_one_sentence() -> None:
    name = "/".join(("A", "PLACE", "9", "2021"))  # invented, built so it is not a literal
    assert vaccine_warning(name, "egg", "its strain has no original record") == (
        f"WARNING: vaccine {name} (egg) keeps its own preparation's colour, not its "
        "strain's original clade: its strain has no original record."
    )


def vaccine_setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[StoreColours, Chart]:
    """Three egg vaccines: one whose strain has an original record of another clade than its
    own egg record, one with only egg records, and one whose cell records disagree."""
    colours, _, _ = caller_setup(tmp_path)
    passages = tmp_path / "passage-classes.tsv"
    passages.write_text(
        "pattern\tclass\treason\nORIGINAL\toriginal\ttest\nMDCK\tcell\ttest\nE\\d\tegg\ttest\n"
    )
    colours.rules = dataclasses.replace(colours.rules, passage=tuple(read_passage_rules(passages)))
    sub = "A(H3N2)"
    names = ["/".join(("A", "PLACE", str(n), "2021")) for n in (11, 12, 13)]
    chart = Chart({"V": sub}, [Antigen(n, passage="E3") for n in names], [], Titres([[]] * 3))
    colours._sequences = {
        (sub, names[0], "", (), "E3"): PreparationSequence("EPI_1", "A1", "P.1.1", "exact", False),
    }
    colours._strains = {
        "h3": {
            ("PLACE", "11", "2021"): [("EPI_1", "A1", "E3"), ("EPI_2", "A2", "ORIGINAL")],
            ("PLACE", "12", "2021"): [("EPI_3", "A3", "E2")],
            ("PLACE", "13", "2021"): [("EPI_4", "A4", "MDCK1"), ("EPI_5", "A5", "MDCK2")],
        }
    }
    con = duckdb.connect()
    con.execute(
        "CREATE TABLE clade_rows AS SELECT * FROM (VALUES "
        "('EPI_1', 'A1', 'P.1.1', 'supported', NULL), ('EPI_2', 'A2', 'P.2', 'supported', NULL), "
        "('EPI_3', 'A3', 'P.1.1', NULL, NULL), ('EPI_4', 'A4', 'P.1.1', 'supported', NULL), "
        "('EPI_5', 'A5', 'P.2', 'supported', NULL)"
        ") t(epi_isl, accession, clade, clade_evidence, supported_clade)"
    )
    colours._con = con

    def aligned(row: str, records: list[StrainRecord]) -> AlignedSequences:
        return AlignedSequences({(r.epi_isl, r.accession): AlignedSequence("M") for r in records})

    monkeypatch.setattr(colours, "_strain_aligned", aligned)
    return colours, chart


def test_a_vaccine_takes_its_strains_original_clade(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    colours, chart = vaccine_setup(tmp_path, monkeypatch)
    scheme = CladeColourScheme(
        "A(H3N2)",
        "caller",
        (
            ColourEntry(1, "P", "Clade P", "#aa0000", False),
            ColourEntry(2, "P.1", "Clade P.1", "#0000aa", False),
        ),
    )
    own = colours.for_chart(chart, scheme).styles
    assert [s.label for s in own] == ["Clade P.1", "", ""]  # its egg record: P.1.1
    got = colours.for_vaccines(chart, [(0, "egg"), (1, "egg"), (2, "egg")], own, scheme)

    original, egg_only, split = got
    assert (original.style.label, original.labels, original.source, original.changed) == (
        "Clade P", frozenset({"P"}), "original", True
    )  # fmt: skip
    assert original.warning == ""
    assert original.record() == {
        "antigen": "ag0", "name": chart.antigens[0].name, "passage_class": "egg",
        "colour": "Clade P", "source": "original", "records": 1, "changed": True,
    }  # fmt: skip
    assert (egg_only.style, egg_only.changed, egg_only.source) == (own[1], False, "")
    assert egg_only.warning.endswith("its strain has no original or cell sequence record.")
    assert (split.style, split.changed) == (own[2], False)
    assert split.warning.endswith("its strain's cell records disagree (Clade P 1, Clade P.1 1).")
    assert split.record()["warning"] == split.warning


def test_a_serum_takes_the_record_of_its_strain_closest_to_its_passage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Sera are found by strain name, as ae's seqdb lookup does. An egg serum takes the egg
    record, a cell serum the cell one; among records equally far, one whose alignment reads
    (not a partial sequence of gaps) wins over a lower EPI_ISL."""
    from af.chart.model import Serum
    from af.map.colouring import serum_record_rank

    colours, _ = vaccine_setup(tmp_path, monkeypatch)
    names = ["/".join(("A", "PLACE", str(n), "2021")) for n in (11, 13, 13, 14)]
    sera = [
        Serum(names[0], "F1", passage="E?"),  # E3 or an unreadable passage: the egg record
        Serum(names[1], "F2", passage="MDCK1"),  # the exact passage wins, partial or not
        Serum(names[2], "F3"),  # no passage: both rank 90, the one that reads wins
        Serum(names[3], "F4", passage="MDCK1"),  # no record of its strain
    ]
    chart = Chart({"V": "A(H3N2)"}, [], sera, Titres([]))
    full = "MKAG" + "N" * 6  # position 3 is A, position 4 G
    gaps = "-" * 10

    def aligned(row: str, records: list[StrainRecord]) -> AlignedSequences:
        reads = {"EPI_4": gaps, "EPI_5": full}  # the lower EPI_ISL is a partial sequence
        return AlignedSequences(
            {(r.epi_isl, r.accession): AlignedSequence(reads.get(r.epi_isl, full)) for r in records}
        )

    monkeypatch.setattr(colours, "_strain_aligned", aligned)
    egg, exact, unread, none = colours.for_sera(chart, residues=(3, 4))
    assert (egg.epi_isl, egg.clade, egg.lineage) == ("EPI_1", "P.1.1", ("P", "P.1", "P.1.1"))
    assert egg.aa == {3: "A", 4: "G"} and egg.record_passage == "E3"
    assert (exact.epi_isl, exact.aa) == ("EPI_4", {3: "-", 4: "-"})
    assert (unread.epi_isl, unread.clade, unread.lineage) == ("EPI_5", "P.2", ("P", "P.2"))
    assert (none.epi_isl, none.clade, none.lineage, none.aa) == (None, None, None, {})
    assert none.why == "no sequence record"
    passages = colours.rules.passages
    assert serum_record_rank("E2", "E2", passages) == 0
    assert serum_record_rank("MDCK1/SIAT2", "SIAT2", passages) == 10
    assert serum_record_rank("SIAT?", "SIAT1", passages) == 20
    assert serum_record_rank("E?", "E3/E2", passages) == 20
    assert serum_record_rank("SIAT1", "MDCK2", passages) == 50
    assert serum_record_rank("E2", "MDCK2", passages) == 90
    assert serum_record_rank("", "MDCK2", passages) == 90
