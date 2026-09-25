"""Publish clade assignments into the store as ``clades/<subtype>`` (interface I4).

One row per sequence, in Parquet beside a small JSON report::

    assignments.parquet   the table consumers read
    report.json           counts, and every clade with how many sequences carry it

Columns (:data:`COLUMNS`) are the key the sequence store uses — ``epi_isl`` plus the
segment's own ``accession``, because EPI_ISL alone is not unique — then the clade, how it
was decided, and how much of the clade's signature the sequence actually showed:

    epi_isl, accession, subtype, clade, method, support, unobservable, tree_node

``clade`` is a single name; ancestry is derived from the clade set, so "is this virus in
D?" is answered with :meth:`CladeSet.is_within` and never by matching name prefixes. An
empty clade means the nomenclature does not name the virus — it is not a failure, and it
must not be filled in with a guess.

Why the store version matters more here than almost anywhere else: in the old system a
clade was baked into a chart at populate time, and an edit to the clade definitions
reached maps only if someone remembered to re-populate. Nothing said otherwise, so maps,
trees and geographic maps could disagree for weeks. Here every version records, in its
provenance, the sequence-store version it labelled, the nomenclature pin it used, and the
engine that did it, so a stale table is detectable rather than merely old.
"""

from __future__ import annotations

import datetime
import json
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from af.clades.assign import Assignment
from af.clades.nomenclature import CladeSet
from af.store import ExternalInput, Provenance, Store, StoreRef
from af.store.ref import Input

ASSIGNMENTS_FILE = "assignments.parquet"
REPORT_FILE = "report.json"
STEP = "clades.assign"

#: The I4 table. Types are DuckDB's, as in the serology store.
COLUMNS: dict[str, str] = {
    "epi_isl": "VARCHAR",
    "accession": "VARCHAR",
    "subtype": "VARCHAR",
    "clade": "VARCHAR",
    "method": "VARCHAR",
    "support": "INTEGER",
    "unobservable": "INTEGER",
    "tree_node": "VARCHAR",
}

#: How a clade was decided. Recorded per row because the two are not equally strong: a
#: tree assignment uses the virus's ancestry, a fallback only its own sequence.
METHODS = ("tree", "fallback")

#: af subtype name -> store dataset key. B/Yamagata is deliberately absent: Sarah decided
#: on 25 Sep 2026 that B/Yam trees and maps carry no clade labels for now, so there is no
#: clades/byam dataset and asking for one must fail rather than return nothing.
DATASETS = {"A(H1N1)": "h1", "A(H3N2)": "h3", "B/Vic": "bvic"}


class CladeStoreError(RuntimeError):
    """The assignments cannot be published as they stand."""


@dataclass(frozen=True)
class CladeRow:
    """One sequence's clade, as it is written to the store."""

    epi_isl: str
    accession: str
    subtype: str
    clade: str | None
    method: str
    support: int = 0
    unobservable: int = 0
    tree_node: str | None = None

    def to_record(self) -> dict[str, Any]:
        return {
            "epi_isl": self.epi_isl,
            "accession": self.accession,
            "subtype": self.subtype,
            "clade": self.clade,
            "method": self.method,
            "support": self.support,
            "unobservable": self.unobservable,
            "tree_node": self.tree_node,
        }


def dataset_for(subtype: str) -> str:
    """The store dataset key for ``subtype``; an unknown one is fatal (design rule 4)."""
    try:
        return DATASETS[subtype]
    except KeyError:
        known = ", ".join(sorted(DATASETS))
        raise CladeStoreError(
            f"no clade dataset for subtype {subtype!r}; there are clade definitions for: {known}"
        ) from None


def rows_from_assignments(
    assignments: Mapping[str, Assignment],
    identities: Mapping[str, tuple[str, str]],
    subtype: str,
    *,
    method: str = "tree",
) -> list[CladeRow]:
    """Turn a tree's assignments into store rows.

    ``identities`` maps a node name to its ``(epi_isl, accession)``; nodes absent from it
    are internal nodes of the tree and are not written, because the store is keyed by
    sequence. A node that *is* in ``identities`` but has no assignment is an error: it
    means the tree and the sequence set disagree, which would otherwise show up much
    later as a virus mysteriously absent from a map.
    """
    if method not in METHODS:
        raise CladeStoreError(f"unknown method {method!r}; expected one of {', '.join(METHODS)}")
    missing = sorted(set(identities) - set(assignments))
    if missing:
        raise CladeStoreError(
            f"{len(missing)} sequences have no assignment, e.g. {missing[:5]}: "
            "the tree and the sequence set do not agree"
        )
    rows = []
    for node, (epi_isl, accession) in sorted(identities.items()):
        assignment = assignments[node]
        rows.append(
            CladeRow(
                epi_isl=epi_isl,
                accession=accession,
                subtype=subtype,
                clade=assignment.clade,
                method=method,
                support=assignment.support,
                unobservable=assignment.unobservable,
                tree_node=node if method == "tree" else None,
            )
        )
    return rows


def build_report(rows: Sequence[CladeRow], clade_set: CladeSet) -> dict[str, Any]:
    """Counts a reviewer needs: how many sequences, how many unnamed, and per clade.

    Clades with no sequences are listed too (design rule 1): a clade that has vanished is
    ordinary as viruses turn over, but it is also what a broken definition looks like, so
    it must be visible rather than merely absent from the counts.
    """
    counts: dict[str, int] = {}
    unnamed = 0
    for row in rows:
        if row.clade is None:
            unnamed += 1
        else:
            counts[row.clade] = counts.get(row.clade, 0) + 1
    return {
        "sequences": len(rows),
        "unnamed": unnamed,
        "clade_set_version": clade_set.version,
        "clades_assigned": len(counts),
        "clades_defined": len(clade_set.names),
        "counts": dict(sorted(counts.items())),
        "clades_without_sequences": sorted(set(clade_set.names) - set(counts)),
        "methods": dict(sorted(_count(row.method for row in rows).items())),
    }


def publish(
    store: Store,
    subtype: str,
    rows: Sequence[CladeRow],
    clade_set: CladeSet,
    *,
    sequences: StoreRef,
    nomenclature: Iterable[ExternalInput],
    started: datetime.datetime,
    engine: str = "tree",
    extra_inputs: Iterable[Input] = (),
) -> StoreRef:
    """Write one version of ``clades/<subtype>`` and make it current.

    ``sequences`` is the sequence-store version these clades label, and ``nomenclature``
    the pinned upstream inputs. Both go into the provenance, which is what makes a stale
    clade table detectable: if either moves, a consumer can see that the clades were
    built from something else.
    """
    if not rows:
        raise CladeStoreError(
            f"{subtype}: refusing to publish an empty clade table; "
            "a run that assigned nothing is a failure, not an empty result"
        )
    wrong = sorted({row.subtype for row in rows} - {subtype})
    if wrong:
        raise CladeStoreError(f"{subtype}: rows carry other subtypes: {wrong}")
    duplicates = _duplicates((row.epi_isl, row.accession) for row in rows)
    if duplicates:
        raise CladeStoreError(
            f"{subtype}: {len(duplicates)} sequences appear more than once, e.g. {duplicates[:3]}"
        )
    dataset = dataset_for(subtype)
    report = build_report(rows, clade_set)
    with store.build("clades", dataset) as builder:
        _write_parquet(builder.path / ASSIGNMENTS_FILE, [row.to_record() for row in rows])
        (builder.path / REPORT_FILE).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        provenance = Provenance(
            step=STEP,
            inputs=(sequences, *tuple(nomenclature), *tuple(extra_inputs)),
            parameters={
                "subtype": subtype,
                "engine": engine,
                "clade_set_version": clade_set.version,
                "sequences": len(rows),
                "unnamed": report["unnamed"],
            },
            started=started,
            finished=datetime.datetime.now(datetime.UTC),
        )
        return builder.publish(provenance, summary=report)


def read_report(store: Store, ref: StoreRef) -> dict[str, Any]:
    """The report of a published version."""
    with (store.resolve(ref) / REPORT_FILE).open() as stream:
        report: dict[str, Any] = json.load(stream)
    return report


def _duplicates(keys: Iterable[tuple[str, str]]) -> list[tuple[str, str]]:
    seen: set[tuple[str, str]] = set()
    repeated: list[tuple[str, str]] = []
    for key in keys:
        if key in seen:
            repeated.append(key)
        else:
            seen.add(key)
    return repeated


def _count(values: Iterable[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return counts


def _write_parquet(path: Path, records: list[dict[str, Any]]) -> None:
    """Write the table with an explicit schema, through a temporary JSON-lines file.

    Why JSON lines rather than CSV: DuckDB then reads the declared column types and keeps
    the difference between an empty string and a missing value, which matters here because
    an absent clade is a real answer ("the nomenclature does not name this virus") and
    must not become the empty string. Same approach as the serology store.
    """
    import duckdb

    with tempfile.TemporaryDirectory() as temporary:
        source = Path(temporary) / "rows.jsonl"
        with source.open("w", encoding="utf-8") as stream:
            for record in records:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        specification = ", ".join(f"'{name}': '{kind}'" for name, kind in COLUMNS.items())
        selection = ", ".join(COLUMNS)
        connection = duckdb.connect()
        query = (
            f"SELECT {selection} FROM read_json('{source.as_posix()}', "
            f"format='newline_delimited', columns={{{specification}}})"
        )
        connection.execute(f"COPY ({query}) TO '{path.as_posix()}' (FORMAT PARQUET)")
        connection.close()
