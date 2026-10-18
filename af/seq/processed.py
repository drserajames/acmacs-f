"""The processed sequence store (interface I3): Parquet, one dataset per subtype.

``sequences/<subtype>`` (h1, h3, bvic, byam: one alignment reference each). A version holds
two tables, each partitioned by the pull its rows came from::

    isolates/pull=<pull-id>/part-0.parquet    metadata, one row per (epi_isl, accession)
    sequences/pull=<pull-id>/part-0.parquet   raw and aligned sequence, alignment facts

Adding a pull writes only that pull's two files; every other partition is hard-linked
from the current version, so it stays byte-identical and costs no disk. Every row carries
its own ``source_pull``, so a reader never has to parse directory names.

The store keeps **facts**, not verdicts: alignment coverage, frameshifts, unknown amino
acids and so on, never "passes QC". Thresholds are selection rules (R3), applied when a
tree's sequences are chosen, so changing one never means rebuilding the store.

No record is dropped here. A record that could not be aligned is kept with its error; a
record whose metadata is odd keeps its ``problems`` flags (design rule 1b). The one row
not written is a second, identical copy of a record another pull already holds
(:func:`_drop_already_published`), and the version's provenance counts it.
"""

from __future__ import annotations

import datetime
import hashlib
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from af.seq.dates import CollectionDate
from af.seq.gisaid import SequenceRecord
from af.seq.nextclade import Aligned
from af.store.manifest import Manifest
from af.store.ref import Input, StoreRef
from af.store.store import CURRENT, MANIFEST, Provenance, Store, VersionBuilder

KIND = "sequences"
TABLES = ("isolates", "sequences")
PART = "part-0.parquet"

ISOLATES = pa.schema(
    [
        ("epi_isl", pa.string()),
        ("accession", pa.string()),
        ("gisaid_name", pa.string()),
        ("name", pa.string()),  # af.seq.names.normalise
        ("subtype", pa.string()),  # the dataset: h1, h3, bvic, byam
        ("gisaid_subtype", pa.string()),
        ("lineage", pa.string()),  # as GISAID states it; may be empty
        # The date as stated ("2021", "2021-03", "2021-03-17") and the interval it means.
        # Consumers use first/last; nobody re-derives an interval or assumes 1 January.
        ("collection_date", pa.string()),
        ("date_precision", pa.string()),  # day | month | year, null when unreadable
        ("collection_date_first", pa.date32()),
        ("collection_date_last", pa.date32()),
        ("submission_date", pa.date32()),
        ("update_date", pa.date32()),
        ("location", pa.string()),  # GISAID's whole "region / country / …" text
        ("region", pa.string()),
        ("country", pa.string()),
        ("place", pa.string()),  # what follows the country, " / "-joined; may be empty
        ("originating_lab", pa.string()),
        ("submitting_lab", pa.string()),
        ("passage", pa.string()),
        ("host", pa.string()),
        ("embargoed", pa.bool_()),
        ("embargoed_until", pa.date32()),
        ("problems", pa.list_(pa.string())),
        ("source_pull", pa.string()),
    ]
)

SEQUENCES = pa.schema(
    [
        ("epi_isl", pa.string()),
        ("accession", pa.string()),
        ("nuc_raw", pa.string()),
        # For grouping identical sequences in reports only. Never an identity, never a
        # reason to drop a row: identical sequences from two isolates are two records.
        ("seq_hash", pa.string()),
        ("nuc_aligned", pa.string()),  # mature HA, reference numbering; null if not aligned
        # mature HA protein; null only when no mature CDS translated. A CDS that did not
        # translate is filled with "X" (unobservable) and aa_partial says so.
        ("aa_aligned", pa.string()),
        ("aa_partial", pa.bool_()),
        ("align_error", pa.string()),
        ("alignment_start", pa.int32()),
        ("alignment_end", pa.int32()),
        ("covers_mature", pa.bool_()),
        ("failed_cds", pa.list_(pa.string())),
        ("frameshifts", pa.int32()),
        ("deleted_aa", pa.int32()),
        ("inserted_aa", pa.int32()),
        ("unknown_aa", pa.int32()),
        ("premature_stop", pa.bool_()),
        # Against this dataset's reference, over the aligned range; what B placement compares.
        ("substitutions", pa.int32()),
        ("nextclade_qc_status", pa.string()),  # reported, never used (see af.seq.nextclade)
        # Nextclade's clade calls from the pinned dataset, for af.clades' fallback only.
        ("nextclade_clade", pa.string()),
        ("nextclade_subclade", pa.string()),
        ("source_pull", pa.string()),
    ]
)


class StoreBuildError(RuntimeError):
    """Records that cannot be placed, aligned or stored consistently."""


# ---- placement: which dataset a record belongs to -----------------------------------


@dataclass(frozen=True)
class PlacementRule:
    """GISAID's (Subtype, Lineage) → a dataset, with the reason written in the rule."""

    gisaid_subtype: str
    lineage: str
    dataset: str
    reason: str


def place(
    records: Iterable[SequenceRecord], rules: Sequence[PlacementRule]
) -> dict[str, list[SequenceRecord]]:
    """Group records by dataset. A record no rule places is fatal, with the counts.

    Silently leaving a record out would make it vanish from every tree; guessing a
    dataset would align it against the wrong reference.
    """
    table = {(rule.gisaid_subtype, rule.lineage): rule.dataset for rule in rules}
    if len(table) != len(rules):
        raise StoreBuildError("two placement rules for the same (subtype, lineage)")
    placed: dict[str, list[SequenceRecord]] = {}
    unplaced: Counter[tuple[str, str]] = Counter()
    for record in records:
        dataset = table.get((record.subtype, record.lineage))
        if dataset is None:
            unplaced[(record.subtype, record.lineage)] += 1
        else:
            placed.setdefault(dataset, []).append(record)
    if unplaced:
        shown = ", ".join(f"{s!r}/{lineage!r}: {n}" for (s, lineage), n in sorted(unplaced.items()))
        raise StoreBuildError(f"records no placement rule covers (subtype/lineage): {shown}")
    return placed


@dataclass(frozen=True)
class LineageCheck:
    """Place records of one GISAID subtype by the reference they are nearest, not the label.

    Every record of ``gisaid_subtype`` is aligned against each candidate dataset. The
    nearest (fewest substitutions) wins when it is nearer by at least ``min_margin``; a
    closer call is not evidence either way and falls back to the placement rules.
    """

    gisaid_subtype: str
    candidates: Mapping[str, str]  # GISAID lineage -> dataset
    min_margin: int
    reason: str


def choose_lineage(
    stated: str, results: Mapping[str, Aligned], check: LineageCheck
) -> tuple[str | None, str | None]:
    """``(dataset, flag)`` for one record from its alignment against each candidate.

    ``dataset`` None means "no evidence": the placement rules decide, and the flag says
    why. A clear placement that contradicts GISAID's stated lineage follows the evidence
    (the record is aligned against the reference it actually resembles) and is flagged.
    """
    distances = sorted(
        (result.substitutions, dataset)
        for dataset, result in results.items()
        if result.error is None and result.substitutions is not None
    )
    if not distances:
        return None, "lineage.unaligned"
    best, dataset = distances[0]
    margin = distances[1][0] - best if len(distances) > 1 else None
    if margin is not None and margin < check.min_margin:
        return None, "lineage.ambiguous" if not stated else "lineage.unconfirmed"
    if not stated:
        return dataset, "lineage.from-alignment"
    if stated not in check.candidates:
        return dataset, "lineage.unknown-label"
    if check.candidates[stated] != dataset:
        return dataset, "lineage.disagrees"
    return dataset, None


def place_by_alignment(
    records: Iterable[SequenceRecord],
    rules: Sequence[PlacementRule],
    check: LineageCheck,
    results: Mapping[str, Mapping[str, Aligned]],
) -> tuple[dict[str, list[SequenceRecord]], Counter[str]]:
    """Place checked records (see :class:`LineageCheck`); returns groups and flag counts.

    ``results`` is dataset -> seq id -> alignment, for every candidate dataset. Where the
    alignment gives no answer, the record's placement row decides, and its flag stays on
    the record so a reader can see it was not placed by evidence.
    """
    by_rule = {(rule.gisaid_subtype, rule.lineage): rule.dataset for rule in rules}
    placed: dict[str, list[SequenceRecord]] = {}
    flags: Counter[str] = Counter()
    unplaced: Counter[tuple[str, str]] = Counter()
    for record in records:
        per_dataset = {dataset: found[seq_id(record)] for dataset, found in results.items()}
        dataset, flag = choose_lineage(record.lineage, per_dataset, check)
        if dataset is None:
            dataset = by_rule.get((record.subtype, record.lineage))
        if dataset is None:
            unplaced[(record.subtype, record.lineage)] += 1
            continue
        if flag:
            flags[flag] += 1
            record = replace(record, problems=(*record.problems, flag))
        placed.setdefault(dataset, []).append(record)
    if unplaced:
        shown = ", ".join(f"{s!r}/{lineage!r}: {n}" for (s, lineage), n in sorted(unplaced.items()))
        raise StoreBuildError(f"records neither alignment nor a placement rule places: {shown}")
    return placed, flags


def seq_id(record: SequenceRecord) -> str:
    """The id a record is aligned under: the store key, joined without whitespace."""
    return f"{record.epi_isl}.{record.accession}"


# ---- holding records that are not the dataset's lineage ------------------------------


@dataclass(frozen=True)
class ReferenceCheck:
    """Keep a dataset to records its reference can speak for, by distance to it.

    A dataset pinned to one reference (say a pandemic H1 virus) cannot describe a record
    from another lineage of the same subtype: the alignment succeeds, the clade engine
    names a clade from the wrong tree, and nothing says it is wrong. Records further than
    ``max_substitutions`` from the reference are **held** — placed into no dataset, so
    they stay in the raw pull and out of every tree, clade table and map.

    A count alone lets a short fragment through: a few hundred bases of another lineage
    cannot reach a threshold set for a whole gene, yet the clade engine still names a clade
    for it. ``max_substitution_rate`` is the same distance per aligned base (see
    :func:`substitution_rate`); a record over **either** limit is held, so adding the rate
    can only hold more.

    ``expect_lineages`` are the GISAID lineage labels that should be within the threshold.
    The label never decides (it is blank or wrong too often); it is compared with the
    distance so a disagreement is flagged and counted.

    ``held_dataset`` names a dataset the held records are published to instead of being
    dropped, so they stay queryable as sequences rather than only as a raw pull (Sarah,
    Q196). One row decides both sides deliberately: an inverted second row could carry a
    different threshold, and then a record would land in both datasets or in neither, with
    nothing to catch it. Absent, the held records are discarded exactly as before.
    """

    dataset: str
    max_substitutions: int
    expect_lineages: tuple[str, ...]
    reason: str
    max_substitution_rate: float | None = None
    held_dataset: str | None = None


#: Held: too far from the dataset's reference to be its lineage.
FAR_FROM_REFERENCE = "lineage.far-from-reference"
#: Held: under the count but over the rate, i.e. a short fragment of another lineage.
FAR_BY_RATE = "lineage.far-from-reference.rate"
#: Kept, but GISAID's label is not one the dataset expects.
LABEL_UNEXPECTED = "lineage.label-unexpected"
#: Kept: no alignment, so distance says nothing (see :func:`hold_far_from_reference`).
NO_DISTANCE = "lineage.no-distance"


def hold_far_from_reference(
    records: Iterable[SequenceRecord],
    aligned: Mapping[str, Aligned],
    check: ReferenceCheck,
) -> tuple[list[SequenceRecord], list[SequenceRecord], Counter[str]]:
    """``(kept, held, flags)`` for one dataset's records, by distance to its reference.

    **A record that did not align is kept, not held.** Failing to align is a length
    problem — a short fragment cannot reach the reference — and says nothing about which
    lineage the virus belongs to. Holding those would evict records on no evidence; they
    carry :data:`NO_DISTANCE` instead, and the alignment and qc rules already keep them
    out of trees.

    The count is tested first, so :data:`FAR_BY_RATE` counts only the records the rate
    adds, and a row without a rate gives the counts it always gave.

    **A held record carries the flag that held it**, so a reader of a ``held_dataset`` can
    tell a full-length virus of another lineage from a short fragment the rate caught. The
    flag was only counted while held records were dropped, and a count cannot be read back
    off a record.
    """
    kept: list[SequenceRecord] = []
    held: list[SequenceRecord] = []
    flags: Counter[str] = Counter()
    for record in records:
        result = aligned.get(seq_id(record))
        distance = None if result is None or result.error is not None else result.substitutions
        if distance is None:
            flags[NO_DISTANCE] += 1
            kept.append(replace(record, problems=(*record.problems, NO_DISTANCE)))
        elif distance > check.max_substitutions:
            flags[FAR_FROM_REFERENCE] += 1
            held.append(replace(record, problems=(*record.problems, FAR_FROM_REFERENCE)))
        elif _over_rate(result, check.max_substitution_rate):
            flags[FAR_BY_RATE] += 1
            held.append(replace(record, problems=(*record.problems, FAR_BY_RATE)))
        elif record.lineage not in check.expect_lineages:
            flags[LABEL_UNEXPECTED] += 1
            kept.append(replace(record, problems=(*record.problems, LABEL_UNEXPECTED)))
        else:
            kept.append(record)
    return kept, held, flags


def substitution_rate(result: Aligned) -> float | None:
    """Substitutions per aligned base, or None when there is nothing to divide by.

    The denominator is the A, C, G and T in the aligned mature sequence: gaps and unknown
    bases are places the record says nothing about, so they must not dilute the distance
    of the part that is there.
    """
    if result.substitutions is None or result.nucleotides is None:
        return None
    bases = sum(result.nucleotides.upper().count(base) for base in "ACGT")
    return result.substitutions / bases if bases else None


def _over_rate(result: Aligned | None, limit: float | None) -> bool:
    rate = None if result is None or limit is None else substitution_rate(result)
    return rate is not None and limit is not None and rate > limit


# ---- rows --------------------------------------------------------------------------


def isolates_table(records: Sequence[SequenceRecord], dataset: str, pull_id: str) -> pa.Table:
    rows = [_isolate_row(record, dataset, pull_id) for record in by_key(records)]
    return pa.Table.from_pylist(rows, schema=ISOLATES)


def sequences_table(
    records: Sequence[SequenceRecord], aligned: Mapping[str, Aligned], pull_id: str
) -> pa.Table:
    """Every record must have exactly its own alignment result; a missing one is fatal."""
    missing = [seq_id(record) for record in records if seq_id(record) not in aligned]
    if missing:
        raise StoreBuildError(f"{len(missing)} record(s) have no alignment result: {missing[:5]}")
    rows = [_sequence_row(record, aligned[seq_id(record)], pull_id) for record in by_key(records)]
    return pa.Table.from_pylist(rows, schema=SEQUENCES)


def by_key(records: Sequence[SequenceRecord]) -> list[SequenceRecord]:
    """Row order by key, so the same records always write the same bytes."""
    return sorted(records, key=lambda record: record.key)


def _isolate_row(record: SequenceRecord, dataset: str, pull_id: str) -> dict[str, Any]:
    date: CollectionDate | None = record.collection_date
    region, country, place_ = split_location(record.location)
    return {
        "epi_isl": record.epi_isl,
        "accession": record.accession,
        "gisaid_name": record.gisaid_name,
        "name": record.name,
        "subtype": dataset,
        "gisaid_subtype": record.subtype,
        "lineage": record.lineage,
        "collection_date": str(date) if date else None,
        "date_precision": date.precision.value if date else None,
        "collection_date_first": date.first if date else None,
        "collection_date_last": date.last if date else None,
        "submission_date": _iso_date(record.submission_date, "Submission_Date", record),
        "update_date": _iso_date(record.update_date, "Update_Date", record, optional=True),
        "location": record.location,
        "region": region,
        "country": country,
        "place": place_,
        "originating_lab": record.originating_lab,
        "submitting_lab": record.submitting_lab,
        "passage": record.passage,
        "host": record.host,
        "embargoed": bool(record.embargoed_until),
        "embargoed_until": _iso_date(
            record.embargoed_until, "Publishing_Embargo_Until", record, optional=True
        ),
        "problems": list(record.problems),
        "source_pull": pull_id,
    }


def _sequence_row(record: SequenceRecord, result: Aligned, pull_id: str) -> dict[str, Any]:
    return {
        "epi_isl": record.epi_isl,
        "accession": record.accession,
        "nuc_raw": record.nucleotides,
        "seq_hash": hashlib.sha256(record.nucleotides.encode()).hexdigest()[:16],
        "nuc_aligned": result.nucleotides,
        "aa_aligned": result.amino_acids,
        "aa_partial": result.aa_partial,
        "align_error": result.error,
        "alignment_start": result.alignment_start,
        "alignment_end": result.alignment_end,
        "covers_mature": result.covers_mature,
        "failed_cds": list(result.failed_cds),
        "frameshifts": result.frameshifts,
        "deleted_aa": result.deleted_aa,
        "inserted_aa": result.inserted_aa,
        "unknown_aa": result.unknown_aa,
        "premature_stop": result.premature_stop,
        "substitutions": result.substitutions,
        "nextclade_qc_status": result.qc_status or None,
        "nextclade_clade": result.clade,
        "nextclade_subclade": result.subclade,
        "source_pull": pull_id,
    }


def split_location(location: str) -> tuple[str | None, str | None, str]:
    """GISAID's ``Region / Country / Division / …``: the first two, and the rest joined."""
    parts = [part.strip() for part in location.split(" / ")] if location.strip() else []
    region = parts[0] if parts else None
    country = parts[1] if len(parts) > 1 else None
    return region, country, " / ".join(parts[2:])


def _iso_date(
    text: str, column: str, record: SequenceRecord, *, optional: bool = False
) -> datetime.date | None:
    if not text:
        if optional:
            return None
        raise StoreBuildError(f"{record.epi_isl}: no {column}")
    try:
        return datetime.date.fromisoformat(text)
    except ValueError:
        raise StoreBuildError(f"{record.epi_isl}: {column} {text!r} is not a date") from None


# ---- writing a version ---------------------------------------------------------------


def partition(table: str, pull_id: str) -> str:
    return f"{table}/pull={pull_id}/{PART}"


def publish_pull(
    store: Store,
    dataset: str,
    pull_id: str,
    isolates: pa.Table,
    sequences: pa.Table,
    inputs: Sequence[Input],
    parameters: Mapping[str, Any],
    started: datetime.datetime,
) -> StoreRef:
    """Publish a version of ``sequences/<dataset>`` with this pull's partitions (re)written.

    Every other pull's partitions are hard-linked from the current version. A key in two
    pulls would make one virus two tree leaves, so a key another pull already holds is
    written once: an identical copy is dropped from this pull and counted in the
    provenance (``already_published``), and a copy that differs is refused.
    """
    if isolates.num_rows != sequences.num_rows:
        raise StoreBuildError(
            f"{pull_id}: {isolates.num_rows} isolates, {sequences.num_rows} sequences"
        )
    with store.build(KIND, dataset) as builder:
        others = _link_other_pulls(builder, pull_id)
        isolates, sequences, repeated = _drop_already_published(
            builder.path, pull_id, isolates, sequences
        )
        for name, table in zip(TABLES, (isolates, sequences), strict=True):
            path = builder.path / partition(name, pull_id)
            path.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(table, path, compression="zstd")
        _check_keys_across_pulls(builder.path)
        provenance = Provenance(
            step="seq.store-pull",
            inputs=tuple(inputs),
            parameters={**parameters, "already_published": repeated},
            started=started,
            finished=datetime.datetime.now(datetime.UTC),
        )
        summary = {
            "pull": pull_id,
            "rows": isolates.num_rows,
            "other_pulls": others,
            "already_published": sum(repeated.values()),
        }
        return builder.publish(provenance, summary=summary)


def _link_other_pulls(builder: VersionBuilder, pull_id: str) -> int:
    """Hard-link every file of the current version except this pull's; count the pulls."""
    if not (builder.store.dataset_dir(KIND, builder.dataset) / CURRENT).is_file():
        return 0  # the first pull of this dataset
    current = builder.store.current(KIND, builder.dataset)
    manifest = Manifest.from_bytes((builder.store.version_dir(current) / MANIFEST).read_bytes())
    mine = {partition(name, pull_id) for name in TABLES}
    pulls = set()
    for entry in manifest.files:
        if entry.path not in mine:
            builder.link_unchanged(entry.path)
            pulls.add(entry.path.split("/")[1])
    return len(pulls)


#: Columns that say which pull delivered a row, not what the record is: two pulls' copies
#: of one record differ in these and in nothing else.
_DELIVERY_COLUMNS = frozenset({"source_pull"})

Key = tuple[str, str]


def _drop_already_published(
    version: Path, pull_id: str, isolates: pa.Table, sequences: pa.Table
) -> tuple[pa.Table, pa.Table, dict[str, int]]:
    """This pull's tables without the records another pull already holds, and their count.

    Pulls overlap: a targeted fetch can name a record that a later window pull also
    returns. The copy already published is kept, so no other pull's partition is ever
    rewritten, and when the two copies are identical apart from the delivering pull, which
    one stays changes nothing a reader sees. Copies that differ in anything else (GISAID
    edited the record between the two fetches) are refused: which edit wins is a rule
    nobody has written, and guessing it would change data silently.

    Returns ``{other pull: records dropped because it holds them}``, empty when none.
    """
    if not (version / "isolates").is_dir():
        return isolates, sequences, {}  # the first pull of this dataset
    incoming = set(_keys(isolates))
    held = pq.read_table(version / "isolates", columns=["epi_isl", "accession", "source_pull"],
                         partitioning=None)  # fmt: skip
    holder: dict[Key, str] = {}
    for epi_isl, accession, pull in zip(
        *(held[c].to_pylist() for c in held.column_names), strict=True
    ):
        if (epi_isl, accession) in incoming:
            holder[(epi_isl, accession)] = pull
    if not holder:
        return isolates, sequences, {}
    differ = [
        f"{epi_isl}/{accession} ({name}: {', '.join(columns)}) in {holder[(epi_isl, accession)]}"
        for name, table in zip(TABLES, (isolates, sequences), strict=True)
        for (epi_isl, accession), columns in _differences(version / name, table, holder).items()
    ]
    if differ:
        raise StoreBuildError(
            f"{pull_id}: {len(differ)} record(s) already published by another pull with "
            f"different content; which copy wins is not decided: {sorted(differ)[:3]}"
        )
    return (
        _without(isolates, holder),
        _without(sequences, holder),
        dict(sorted(Counter(holder.values()).items())),
    )


def _keys(table: pa.Table) -> list[Key]:
    return list(zip(table["epi_isl"].to_pylist(), table["accession"].to_pylist(), strict=True))


def _differences(stored: Path, table: pa.Table, keys: Mapping[Key, str]) -> dict[Key, list[str]]:
    """Per key, the columns in which this pull's row and the stored row disagree."""
    ids = sorted({epi_isl for epi_isl, _ in keys})
    theirs = _rows_by_key(
        pq.read_table(stored, partitioning=None, filters=[("epi_isl", "in", ids)])
    )
    mine = _rows_by_key(table)
    found = {}
    for key in keys:
        row, other = mine[key], theirs[key]
        columns = [c for c in row if c not in _DELIVERY_COLUMNS and row[c] != other.get(c)]
        if columns:
            found[key] = columns
    return found


def _rows_by_key(table: pa.Table) -> dict[Key, dict[str, Any]]:
    return {(row["epi_isl"], row["accession"]): row for row in table.to_pylist()}


def _without(table: pa.Table, keys: Mapping[Key, str]) -> pa.Table:
    return table.filter(pa.array([key not in keys for key in _keys(table)], pa.bool_()))


def published_keys(store: Store, dataset: str, *, except_pull: str | None) -> dict[Key, str]:
    """Every key in the current ``sequences/<dataset>``, with the pull that holds it.

    ``except_pull``'s partition is left out: a publish under way replaces it, so what it
    holds now says nothing about the version being built. Empty for a dataset never
    published.
    """
    if not (store.dataset_dir(KIND, dataset) / CURRENT).is_file():
        return {}
    version = store.version_dir(store.current(KIND, dataset))
    skip = partition("isolates", except_pull) if except_pull is not None else None
    keys: dict[Key, str] = {}
    for path in sorted((version / "isolates").glob(f"pull=*/{PART}")):
        if path.relative_to(version).as_posix() == skip:
            continue
        table = pq.read_table(path, columns=["epi_isl", "accession", "source_pull"],
                              partitioning=None)  # fmt: skip
        for epi_isl, accession, pull in zip(
            *(table[c].to_pylist() for c in table.column_names), strict=True
        ):
            keys[(epi_isl, accession)] = pull
    return keys


def _check_keys_across_pulls(version: Path) -> None:
    keys = pq.read_table(version / "isolates", columns=["epi_isl", "accession", "source_pull"],
                         partitioning=None)  # fmt: skip
    seen: dict[tuple[str, str], str] = {}
    clashes: list[str] = []
    for epi_isl, accession, pull in zip(
        *(keys[c].to_pylist() for c in keys.column_names), strict=True
    ):
        other = seen.setdefault((epi_isl, accession), pull)
        if other != pull:
            clashes.append(f"{epi_isl}/{accession} in {other} and {pull}")
    if clashes:
        raise StoreBuildError(
            f"{len(clashes)} key(s) in two pulls after dropping already-published copies: "
            f"{clashes[:3]}"
        )


# ---- reading -------------------------------------------------------------------------


def read_table(
    store: Store, ref: StoreRef, table: str, columns: Sequence[str] | None = None
) -> pa.Table:
    """One of the two tables of a version, every pull's rows together."""
    if table not in TABLES:
        raise ValueError(f"unknown table {table!r}; expected one of {TABLES}")
    directory = store.resolve(ref) / table
    return pq.read_table(directory, columns=list(columns) if columns else None, partitioning=None)
