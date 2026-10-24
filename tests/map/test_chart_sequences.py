"""StoreColours.for_sequences: a chart's aligned amino acids, antigens and sera, with the reasons
counted where a point has none. Invented strains and records only."""

from __future__ import annotations

from pathlib import Path

import duckdb
import pytest

from af.chart.model import Antigen, Chart, Serum, Titres
from af.clades.sequence import AlignedSequence
from af.map.colouring import (
    AA_NUMBERING,
    NO_AMINO_ACIDS,
    NO_RECORD,
    NO_SEQUENCE,
    TIE,
    ChartSequences,
    MapColouringError,
    PointSequence,
    StoreColours,
)
from af.map.vaccine_colours import StrainRecord
from af.seq.matching import read_passage_rules
from af.serology.joins import PreparationSequence, TiedSequence
from af.serology.outputs import AlignedSequences
from af.store.ref import StoreRef
from tests.map.test_colouring import caller_setup

# an invented protein: the positions a caller tests are read from these, not from any real virus
PROTEIN = "MKT" + "A" * 190 + "NQTN" + "A" * 50
OTHER = "MKT" + "A" * 190 + "NQSN" + "A" * 50


def ref(dataset: str, kind: str = "sequences") -> dict[str, str]:
    version = "0" * 15 + "1"
    return StoreRef(kind, dataset, version, version + "f" * 48).to_json()


def setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[StoreColours, Chart]:
    """Four antigens: sequenced, sequenced-without-protein, tied and unsequenced; two sera, one
    whose strain has two records and one whose strain has none."""
    colours, chart, _ = caller_setup(tmp_path)
    sub = "A(H3N2)"
    names = [a.name for a in chart.antigens]  # PLACE 1 (sequenced), 2 (sequenced), 3 (none)
    tied_name = "/".join(("A", "PLACE", "4", "2021"))
    antigens = [*chart.antigens, Antigen(tied_name, passage="E3")]
    sera = [
        Serum("/".join(("A", "PLACE", "5", "2021")), serum_id="S1", passage="E3"),
        Serum("/".join(("A", "PLACE", "9", "2021")), serum_id="S2", passage="E3"),
    ]
    chart = Chart({"V": sub}, antigens, sera, Titres([[] for _ in antigens]))
    tied = (TiedSequence("EPI_7", "A7", "P.2"), TiedSequence("EPI_8", "A8", "P.2"))
    colours._sequences = {
        (sub, names[0], "", (), "E3"): PreparationSequence("EPI_1", "A1", "P.1.1", "exact", False),
        # a record the join chose whose version holds no aligned protein for it
        (sub, names[1], "", (), "E3"): PreparationSequence("EPI_2", "A2", "P.2", "exact", False),
        (sub, tied_name, "", (), "E3"): PreparationSequence(
            None, None, None, "exact", False, tied=tied, ranked=tied[0]
        ),
    }
    colours._aligned = AlignedSequences({("EPI_1", "A1"): AlignedSequence(PROTEIN)})
    colours._strains = {
        "h3": {
            ("PLACE", "5", "2021"): [("EPI_5", "A5", "E2"), ("EPI_6", "A6", "MDCK1")],
        }
    }
    passages = tmp_path / "passage-classes.tsv"
    passages.write_text("pattern\tclass\treason\nMDCK\tcell\ttest\nE\\d\tegg\ttest\n")
    import dataclasses

    colours.rules = dataclasses.replace(colours.rules, passage=tuple(read_passage_rules(passages)))
    con = duckdb.connect()
    con.execute(
        "CREATE TABLE clade_rows AS SELECT * FROM (VALUES "
        "('EPI_5', 'A5', 'P.1.1', 'supported', NULL), ('EPI_6', 'A6', 'P.2', 'supported', NULL)"
        ") t(epi_isl, accession, clade, clade_evidence, supported_clade)"
    )
    colours._con = con

    def aligned(row: str, records: list[StrainRecord]) -> AlignedSequences:
        # the egg record of the serum's strain carries the protein, its cell record another
        got = {("EPI_5", "A5"): AlignedSequence(PROTEIN), ("EPI_6", "A6"): AlignedSequence(OTHER)}
        asked = {(r.epi_isl, r.accession) for r in records}
        return AlignedSequences({k: v for k, v in got.items() if k in asked})

    monkeypatch.setattr(colours, "_strain_aligned", aligned)
    colours.links.refs = {"sequences": {"h3": ref("h3")}, "clades": [ref("h3", "clades")]}
    colours.serology = StoreRef.from_json(ref("all", "serology"))
    colours.store_read = {"read": "test"}
    return colours, chart


def test_each_antigen_and_serum_strain_gets_its_aligned_protein(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    colours, chart = setup(tmp_path, monkeypatch)
    got = colours.for_sequences(chart)
    assert isinstance(got, ChartSequences) and got.numbering == AA_NUMBERING
    first, no_protein, unsequenced, tied = got.antigens
    assert (first.epi_isl, first.accession) == ("EPI_1", "A1")
    assert first.amino_acids == PROTEIN and first.reason == ""
    # the whole aligned protein, numbered from 1: the position a caller names is [pos - 1], so a
    # caller testing a sequon reads 194..197 as the four amino acids put there
    assert [first.amino_acids[pos - 1] for pos in (194, 195, 196, 197)] == ["N", "Q", "T", "N"]
    assert (no_protein.epi_isl, no_protein.amino_acids) == ("EPI_2", None)
    assert no_protein.reason == NO_AMINO_ACIDS  # a record, and no protein in that version
    assert (unsequenced.amino_acids, unsequenced.reason) == (None, NO_SEQUENCE)
    assert (tied.amino_acids, tied.reason) == (None, TIE)
    # a serum is found by its strain's name, closest passage first: its egg record, not the cell
    with_record, without = got.sera
    assert (with_record.epi_isl, with_record.amino_acids) == ("EPI_5", PROTEIN)
    # its egg record ranks 20 (same passage letter, another number), the cell one 90
    assert with_record.how == "record by name, passage rank 20 of 2 record(s)"
    assert (without.amino_acids, without.reason) == (None, NO_RECORD)


def test_the_reasons_are_counted_and_the_versions_recorded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Design rule 1: a caller can say how many points it tested and why the rest it could not."""
    colours, chart = setup(tmp_path, monkeypatch)
    got = colours.for_sequences(chart)
    assert got.counts() == {
        "antigens": {"sequence": 1, NO_AMINO_ACIDS: 1, NO_SEQUENCE: 1, TIE: 1},
        "sera": {"sequence": 1, NO_RECORD: 1},
    }
    p = got.provenance
    assert p["source"] == "store" and p["numbering"] == AA_NUMBERING
    assert p["store_read"] == {"read": "test"} and p["datasets"] == ["h3"]
    # the versions READ, as a figure cites them: serology, and the join's sequences and clades
    assert [(r["kind"], r["dataset"]) for r in p["store_refs"]] == [
        ("serology", "all"),
        ("sequences", "h3"),
        ("clades", "h3"),
    ]
    assert p["matching_rules"] and p["pins"] == {}


def test_a_point_carries_a_sequence_or_a_reason_never_both_nor_neither() -> None:
    assert PointSequence("EPI_1", "A1", "MKT", "").reason == ""
    assert PointSequence(None, None, None, NO_SEQUENCE).amino_acids is None
    with pytest.raises(MapColouringError, match="exactly one of a sequence and a reason"):
        PointSequence("EPI_1", "A1", "MKT", TIE)
    with pytest.raises(MapColouringError, match="exactly one of a sequence and a reason"):
        PointSequence("EPI_1", "A1", None, "")


def test_a_chart_with_no_sera_reads_no_strain_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The serum side costs a read of the sequence version's isolates; a chart without sera
    must not pay it (the battery's charts are antigens only)."""
    colours, chart = setup(tmp_path, monkeypatch)
    no_sera = Chart({"V": "A(H3N2)"}, chart.antigens, [], Titres([[] for _ in chart.antigens]))

    def refuse(row: str) -> dict[tuple[str, str, str], list[tuple[str, str, str]]]:
        raise AssertionError("a chart with no sera read the strain index")

    monkeypatch.setattr(colours, "_strain_index", refuse)
    got = colours.for_sequences(no_sera)
    assert got.sera == () and len(got.antigens) == 4
    assert got.counts()["sera"] == {}
