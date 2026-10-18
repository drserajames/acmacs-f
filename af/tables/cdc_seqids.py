"""EPI_ISL ids for CDC antigens whose own rows carry none (Sarah, Q217 follow-up).

CDC's main TSV gives every antigen the GISAID isolate it was paired with
(``ag_epi_isolate_id``). Its older season files and its per-test workbooks do not; but CDC
sent separate lookups beside the season files (``HITest_<season>_GISAID.tsv`` and
``FRA_<season>_GISAID.tsv``), one row per sequenced passage of a CDC virus id. A CDC virus id
can have several rows: the clinical specimen ("Original") and one or more passages.

An antigen without an id takes one by its CDC virus id from those lookups and from the main
TSV's own pairings (the same CDC virus id tested again after September 2019):

- a candidate at the antigen's own passage (af's reading of both, so "C1" equals "MDCK1"):
  ``sequence_pairing`` "exact", or CDC's own status when the main TSV gave the pairing;
- otherwise, one isolate at another passage: "proxy", CDC's own word for a sequence of the
  same isolate at a passage other than the one tested;
- otherwise, several isolates: the run's rule (``sequence_id_ties``, a required config choice
  until Sarah rules on Q221): ``blank`` leaves the antigen without one, ``closest-passage``
  takes the closest by af's passage rank (the Q217 rule, :func:`af.seq.matching.ae_ranked`),
  ``original`` takes the clinical specimen's when there is exactly one.

Every id joined records where it came from in ``source`` ("epi_isl_from"), and the run
reports how many antigens took one each way. EPI_ISL is not map content (``model``): a
table that gains ids changes as metadata only, and restarts no chain.
"""

from __future__ import annotations

import csv
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

from af.seq.matching import Candidate, ae_ranked

from .cdc import EPI_ISL, CDCFormatError
from .cdc_season import _records
from .model import Table

COLUMNS = ["virus_cdc_id", "EPI_ISOLATE_ID", "virus_strain", "virus_strain_passage"]
TIES = ("blank", "closest-passage", "original")
FROM_TSV = "CDC TSV"  # a pairing the main TSV gave for the same CDC virus id
ORIGINAL = "original"  # af's passage class of a clinical specimen
EXACT = "exact passage"
PROXY = "one isolate at another passage (proxy)"

Classify = Callable[[str], tuple[str, str]]  # a lab's passage -> af's passage and its class


@dataclass(frozen=True)
class Sequenced:
    """One sequenced passage of a CDC virus id."""

    epi_isl: str
    passage: str  # af's reading of it
    passage_class: str
    pairing: str  # CDC's own status when the main TSV gave it, else ""
    origin: str  # the lookup file's name, or FROM_TSV


Lookup = dict[str, list[Sequenced]]  # CDC virus id -> its sequenced passages


def read_lookup(path: Path, classify: Classify) -> Lookup:
    """One of CDC's ``*_GISAID.tsv`` files. Like the season files, some records arrive split
    by a line break inside the strain name; they are rejoined (``cdc_season._records``)."""
    lines, _rejoined = _records(path.read_text().splitlines(), path)
    reader = csv.DictReader(lines, delimiter="\t", quoting=csv.QUOTE_NONE)
    if reader.fieldnames != COLUMNS:
        raise CDCFormatError(f"{path}: columns {reader.fieldnames}, expected {COLUMNS}")
    lookup: Lookup = {}
    for row in reader:
        cdc_id, epi = row["virus_cdc_id"].strip(), row["EPI_ISOLATE_ID"].strip()
        if not cdc_id or not EPI_ISL.fullmatch(epi):
            raise CDCFormatError(f"{path}: virus_cdc_id {cdc_id!r}, EPI_ISOLATE_ID {epi!r}")
        passage, passage_class = classify(row["virus_strain_passage"].strip())
        lookup.setdefault(cdc_id, []).append(Sequenced(epi, passage, passage_class, "", path.name))
    return lookup


def from_tables(tables: Iterable[Table]) -> Lookup:
    """The pairings CDC gave in its main TSV, by CDC virus id. An antigen that merges two CDC
    isolate ids (``cdc.MERGED_ISOLATES``) gives none: its id belongs to one of the two."""
    lookup: Lookup = {}
    for table in tables:
        for a in table.antigens:
            ids = [i.removeprefix("CDC#") for i in a.lab_ids if i.startswith("CDC#")]
            if a.epi_isl and len(ids) == 1:
                pairing = a.sequence_pairing
                item = Sequenced(a.epi_isl, a.passage, a.passage_class, pairing, FROM_TSV)
                if item not in lookup.get(ids[0], []):
                    lookup.setdefault(ids[0], []).append(item)
    return lookup


def join(tables: Iterable[Table], lookups: list[Lookup], ties: str) -> Counter[str]:
    """Give each CDC antigen without an EPI_ISL one from ``lookups``, as the module
    docstring says; return the counts for the run's report."""
    if ties not in TIES:
        raise ValueError(f"sequence_id_ties {ties!r}: one of {', '.join(TIES)}")
    counts: Counter[str] = Counter()
    for table in tables:
        for a in table.antigens:
            if a.epi_isl:
                continue
            ids = [i.removeprefix("CDC#") for i in a.lab_ids if i.startswith("CDC#")]
            candidates = [s for lookup in lookups for s in lookup.get(ids[0], [])] if ids else []
            if len(ids) != 1 or not candidates:
                counts["antigens: no CDC sequence id"] += 1
                continue
            chosen, how, detail = _choose(candidates, a.passage, a.passage_class, ties)
            counts[f"antigens: {how}"] += 1
            if chosen is None:
                continue
            a.epi_isl = chosen.epi_isl
            a.sequence_pairing = (chosen.pairing or "exact") if how == EXACT else "proxy"
            a.source["epi_isl_from"] = f"{chosen.origin}, {detail}"
    return counts


def _choose(
    candidates: list[Sequenced], passage: str, passage_class: str, ties: str
) -> tuple[Sequenced | None, str, str]:
    """The chosen isolate (or None), the report's category, and the detail for ``source``."""
    same = _distinct([s for s in candidates if s.passage == passage])
    if len(same) == 1:
        return same[0], EXACT, EXACT
    pool = same or _distinct(candidates)  # several at its own passage tie as well
    if len(pool) == 1:
        return pool[0], PROXY, f"{PROXY}, sequenced at {pool[0].passage or 'no passage'}"
    if ties == "closest-passage":
        best = ae_ranked([_candidate(s) for s in pool], passage, passage_class)
        assert best is not None  # the pool is not empty
        chosen = next(s for s in pool if s.epi_isl == best.epi_isl)
        return (
            chosen,
            "several isolates: closest passage",
            (f"closest passage of {len(pool)} isolates, sequenced at {chosen.passage}"),
        )
    if ties == ORIGINAL:
        originals = [s for s in pool if s.passage_class == ORIGINAL]
        if len(originals) == 1:
            return (
                originals[0],
                "several isolates: the original specimen",
                (f"the original specimen of {len(pool)} isolates"),
            )
    return None, f"several isolates: left without one ({ties})", ""


def _distinct(items: list[Sequenced]) -> list[Sequenced]:
    """One per EPI_ISL: the main TSV's pairing first (it carries CDC's status), then file order."""
    seen: dict[str, Sequenced] = {}
    for s in sorted(items, key=lambda s: s.origin != FROM_TSV):
        seen.setdefault(s.epi_isl, s)
    return list(seen.values())


def _candidate(s: Sequenced) -> Candidate:
    return Candidate(
        epi_isl=s.epi_isl,
        accession="",
        dataset="",
        name="",
        passage=s.passage,
        passage_class=s.passage_class,
        seq_hash="",
    )
