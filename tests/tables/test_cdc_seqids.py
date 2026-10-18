"""EPI_ISL ids from CDC's lookups for antigens whose rows carry none (invented ids and names)."""

from __future__ import annotations

from pathlib import Path

import pytest

from af.tables import cdc_seqids
from af.tables.model import Antigen, Table
from af.tables.rules import Rules
from af.tables.update import CDCInputs, _join_sequence_ids

from .synthetic_rules import write_rules

HEADER = "\t".join(cdc_seqids.COLUMNS)
NAME = "A(H3N2)/EXAMPLETOWN/1/2030"
PASSAGES = {  # a lab's passage -> af's reading and class, as PassageParser gives them
    "Original": ("OR", "original"),
    "C1": ("MDCK1", "cell"),
    "C2": ("MDCK2", "cell"),
    "S1": ("SIAT1", "cell"),
    "S3": ("SIAT3", "cell"),
    "E3": ("E3", "egg"),
}


def classify(raw: str) -> tuple[str, str]:
    return PASSAGES[raw]


def lookup_file(tmp_path: Path, *rows: tuple[str, str, str]) -> Path:
    """(CDC virus id, EPI_ISL, passage) rows of an invented *_GISAID.tsv."""
    path = tmp_path / "EXAMPLE_GISAID.tsv"
    lines = [HEADER] + [f"{cdc_id}\t{epi}\t{NAME}\t{p}" for cdc_id, epi, p in rows]
    path.write_text("\n".join(lines) + "\n")
    return path


def table(*antigens: Antigen, source_key: str = "CDC season file EXAMPLE") -> Table:
    return Table(
        table_id="",
        group="h3-hi-guinea-pig-cdc",
        lab="CDC",
        subtype="A(H3N2)",
        lineage="",
        assay="HI",
        rbc="guinea-pig",
        date="2030-01-02",
        date_suffix=0,
        source_key=source_key,
        antigens=list(antigens),
        sera=[],
        titres=[[] for _ in antigens],
    )


def antigen(cdc_id: str, passage: str, passage_class: str = "cell", **kw: str) -> Antigen:
    return Antigen(
        name=NAME,
        raw_name=NAME,
        passage=passage,
        passage_class=passage_class,
        lab_ids=[f"CDC#{cdc_id}"],
        **kw,  # type: ignore[arg-type]
    )


def joined(tmp_path: Path, a: Antigen, rows: list[tuple[str, str, str]], ties: str = "blank"):
    lookup = cdc_seqids.read_lookup(lookup_file(tmp_path, *rows), classify)
    counts = cdc_seqids.join([table(a)], [lookup], ties)
    return a, counts


def test_an_isolate_sequenced_at_the_tested_passage_is_exact(tmp_path):
    a, counts = joined(tmp_path, antigen("100", "MDCK1"), [("100", "EPI_ISL_9001", "C1")])
    assert (a.epi_isl, a.sequence_pairing) == ("EPI_ISL_9001", "exact")
    assert a.source["epi_isl_from"] == "EXAMPLE_GISAID.tsv, exact passage"
    assert counts == {"antigens: exact passage": 1}


def test_the_only_isolate_at_another_passage_is_a_proxy(tmp_path):
    a, counts = joined(tmp_path, antigen("100", "MDCK1"), [("100", "EPI_ISL_9001", "Original")])
    assert (a.epi_isl, a.sequence_pairing) == ("EPI_ISL_9001", "proxy")
    assert "sequenced at OR" in a.source["epi_isl_from"]
    assert counts == {"antigens: one isolate at another passage (proxy)": 1}


SEVERAL = [("100", "EPI_ISL_9001", "Original"), ("100", "EPI_ISL_9009", "S1")]


@pytest.mark.parametrize(
    ("ties", "epi", "category"),
    [
        ("blank", "", "several isolates: left without one (blank)"),
        ("closest-passage", "EPI_ISL_9009", "several isolates: closest passage"),
        ("original", "EPI_ISL_9001", "several isolates: the original specimen"),
    ],
)
def test_several_isolates_follow_the_configured_rule(tmp_path, ties, epi, category):
    """Tested at SIAT3: SIAT1 shares the last step's name (closer than the specimen), and
    its EPI_ISL number is the higher, so the pick is by passage and not by id order."""
    a, counts = joined(tmp_path, antigen("100", "SIAT3"), SEVERAL, ties)
    assert a.epi_isl == epi
    assert a.sequence_pairing == ("proxy" if epi else "")
    assert counts == {f"antigens: {category}": 1}


def test_the_main_tsv_pairing_carries_cdcs_own_status(tmp_path):
    """The same CDC virus id tested again in the main TSV, where CDC called the pairing a proxy
    even at the same passage: the joined antigen keeps CDC's word."""
    main = antigen("100", "MDCK1", epi_isl="EPI_ISL_9001", sequence_pairing="proxy")
    old = antigen("100", "MDCK1")
    lookups = [cdc_seqids.from_tables([table(main, source_key="CDC test_id 1")])]
    cdc_seqids.join([table(old)], lookups, "blank")
    assert (old.epi_isl, old.sequence_pairing) == ("EPI_ISL_9001", "proxy")
    assert old.source["epi_isl_from"] == "CDC TSV, exact passage"


def test_an_antigen_with_its_own_id_and_one_without_a_cdc_id_are_left(tmp_path):
    own = antigen("100", "MDCK1", epi_isl="EPI_ISL_1")
    other = antigen("200", "MDCK1")
    lookup = cdc_seqids.read_lookup(lookup_file(tmp_path, ("100", "EPI_ISL_9001", "C1")), classify)
    counts = cdc_seqids.join([table(own, other)], [lookup], "blank")
    assert (own.epi_isl, other.epi_isl) == ("EPI_ISL_1", "")
    assert counts == {"antigens: no CDC sequence id": 1}


def test_a_record_split_inside_the_strain_name_is_rejoined(tmp_path):
    path = tmp_path / "EXAMPLE_GISAID.tsv"
    path.write_text(f"{HEADER}\n100\tEPI_ISL_9001\tA/EXAMPLE\nTOWN/1/2030\tC1\n")
    assert cdc_seqids.read_lookup(path, classify)["100"][0].epi_isl == "EPI_ISL_9001"


def test_an_unknown_tie_rule_is_an_error():
    with pytest.raises(ValueError, match="sequence_id_ties 'newest'"):
        cdc_seqids.join([], [], "newest")


def test_lookups_without_a_tie_rule_are_a_config_error(tmp_path):
    inputs = CDCInputs(
        tsv=tmp_path / "unused.tsv",
        sequence_ids=[lookup_file(tmp_path, ("100", "EPI_ISL_9001", "C1"))],
    )
    with pytest.raises(ValueError, match="needs sequence_id_ties"):
        _join_sequence_ids(inputs, Rules(write_rules(tmp_path / "rules")), [], [])


def test_the_configured_lookups_join_through_the_cdc_passage_rules(tmp_path):
    inputs = CDCInputs(
        tsv=tmp_path / "unused.tsv",
        sequence_ids=[lookup_file(tmp_path, ("100", "EPI_ISL_9001", "C1"))],
        sequence_id_ties="blank",
    )
    a = antigen("100", "MDCK1")
    line = _join_sequence_ids(inputs, Rules(write_rules(tmp_path / "rules")), [], [table(a)])
    assert a.epi_isl == "EPI_ISL_9001"
    assert line == "sequence ids (blank): antigens: exact passage 1"
