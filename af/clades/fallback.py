"""Clades for sequences that are not on a tree, from Nextclade's placement.

The tree engine (:mod:`af.clades.assign`) is what af assigns with, but it needs a tree,
and some viruses have none: map antigens labelled before the month's tree is built, and
viruses that never enter one. Sarah chose Nextclade for those (25 Sep 2026). It places a
sequence on the upstream-maintained reference tree, so it answers the same question from
the same nomenclature, and it agreed with the tree engine on 99.6% of H3 leaves, 99.2% of
H1 and 99.8% of B/Vic (``notes/clades/ENGINE-COMPARISON.md``).

**This module does not run Nextclade.** ``af.seq.nextclade`` already does, with the
dataset pinned by hash and the binary's version checked, as part of aligning the
sequences — and its output TSV already carries the clade columns. Running it again here
would mean a second dataset to keep in step and 80 seconds of work per subtype to get an
answer we already have. So this reads that step's output.

What it adds is the guard the engine comparison recommended: a Nextclade dataset and a
nomenclature commit are pinned separately, and **nothing otherwise checks they agree**.
If the dataset is built from a different revision of the nomenclature, the fallback and
the tree engine quietly start naming clades from two different vocabularies — the sort of
divergence that shows up months later as a map and a tree disagreeing.
:func:`check_dataset_agrees` compares the dataset's own subclade set with the pinned one
and refuses a mismatch.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

from af.clades.assign import Assignment
from af.clades.nomenclature import CladeSet
from af.seq.processed import read_table
from af.store import Store, StoreRef

#: Whatever the two assignments are keyed by: a tree node, or a sequence's store key.
K = TypeVar("K")

#: Nextclade's own words for "no clade here". Both mean the nomenclature does not name
#: the virus, which is an answer, not a failure: on the round's H1 tree 10,942 leaves are
#: ancestral to the nomenclature's root clade and every one of them is 'unassigned'.
UNASSIGNED = frozenset({"", "unassigned"})

#: The column holding the Pango-style subclade, and the fallbacks for older datasets.
SUBCLADE_COLUMNS = ("subclade", "clade")
QC_COLUMN = "qc.overallStatus"
NAME_COLUMN = "seqName"


class FallbackError(RuntimeError):
    """The Nextclade output cannot be used as a clade assignment."""


@dataclass(frozen=True)
class FallbackCounts:
    """What the fallback did, for the run's report."""

    sequences: int = 0
    assigned: int = 0
    unassigned: int = 0
    qc_bad: int = 0

    def to_json(self) -> dict[str, Any]:
        return {
            "sequences": self.sequences,
            "assigned": self.assigned,
            "unassigned": self.unassigned,
            "qc_bad": self.qc_bad,
        }


def dataset_subclades(dataset_dir: Path) -> set[str]:
    """Every subclade the dataset's reference tree can assign.

    Read from the tree rather than from a version string: the tree is what Nextclade
    actually places sequences on, so it is the only honest statement of which names the
    dataset can produce.
    """
    tree_file = Path(dataset_dir) / "tree.json"
    if not tree_file.is_file():
        raise FallbackError(f"Nextclade dataset has no tree.json: {dataset_dir}")
    with tree_file.open() as stream:
        tree = json.load(stream)
    found: set[str] = set()
    stack = [tree.get("tree", {})]
    while stack:
        node = stack.pop()
        attributes = node.get("node_attrs", {})
        for key in SUBCLADE_COLUMNS:
            value = attributes.get(key, {}).get("value")
            if value and value not in UNASSIGNED:
                found.add(value)
        stack.extend(node.get("children", []))
    if not found:
        raise FallbackError(f"no clades on the reference tree of {dataset_dir}")
    return found


def check_dataset_agrees(dataset_dir: Path, clade_set: CladeSet) -> set[str]:
    """Refuse a dataset whose clades are not the pinned nomenclature's.

    Returns the names they share. A name on the dataset's tree that the pinned
    nomenclature does not define means the two were built from different revisions, and
    assignments from each would not be comparable.

    The reverse — a pinned clade the dataset cannot assign — is *not* an error: a clade
    designated after the dataset was built simply has no sequences placed in it yet, and
    the tree engine will still assign it. It is reported by the caller, not refused here.
    """
    dataset = dataset_subclades(dataset_dir)
    unknown = sorted(name for name in dataset if name not in clade_set)
    if unknown:
        raise FallbackError(
            f"the Nextclade dataset at {dataset_dir} can assign {len(unknown)} clade(s) that "
            f"{clade_set.subtype} does not define at {clade_set.version}: {unknown[:5]}. "
            "The dataset and the nomenclature pin are out of step; update the pin, or use the "
            "dataset release built from it."
        )
    return dataset & set(clade_set.names)


def read_rows(tsv: Path) -> list[dict[str, str]]:
    """Read a Nextclade TSV, as ``af.seq.nextclade`` writes it."""
    with Path(tsv).open(newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def assign_from_rows(
    rows: Iterable[Mapping[str, str]],
    clade_set: CladeSet,
    *,
    identities: Mapping[str, str] | None = None,
) -> tuple[dict[str, Assignment], FallbackCounts]:
    """Turn Nextclade rows into assignments.

    ``identities`` optionally maps Nextclade's sequence name to the id af keys by; without
    it the sequence name is used. A clade Nextclade reports that the pinned nomenclature
    does not define is fatal — :func:`check_dataset_agrees` should have caught it at the
    dataset, and reaching it here means the output came from a different dataset than the
    one checked.
    """
    assignments: dict[str, Assignment] = {}
    sequences = assigned = unassigned = qc_bad = 0
    for row in rows:
        name = row.get(NAME_COLUMN)
        if not name:
            raise FallbackError(f"a Nextclade row has no {NAME_COLUMN}: {dict(row)}")
        identifier = identities.get(name, name) if identities is not None else name
        if identifier in assignments:
            raise FallbackError(f"{identifier} appears twice in the Nextclade output")
        sequences += 1
        if row.get(QC_COLUMN) == "bad":
            qc_bad += 1
        clade = _clade(row)
        if clade is None:
            unassigned += 1
        else:
            if clade not in clade_set:
                raise FallbackError(
                    f"Nextclade assigned {clade!r} to {name}, which {clade_set.subtype} does not "
                    f"define at {clade_set.version}: this output is not from the checked dataset"
                )
            assigned += 1
        assignments[identifier] = Assignment(node=identifier, clade=clade)
    if not sequences:
        raise FallbackError("the Nextclade output has no rows")
    return assignments, FallbackCounts(sequences, assigned, unassigned, qc_bad)


def assign_from_tsv(
    tsv: Path,
    clade_set: CladeSet,
    *,
    dataset_dir: Path | None = None,
    identities: Mapping[str, str] | None = None,
) -> tuple[dict[str, Assignment], FallbackCounts]:
    """Read ``af.seq.nextclade``'s TSV and assign, checking the dataset first if given."""
    if dataset_dir is not None:
        check_dataset_agrees(dataset_dir, clade_set)
    return assign_from_rows(read_rows(tsv), clade_set, identities=identities)


#: The sequence store's columns this step reads (``af.seq.processed.SEQUENCES``).
#: Only the subclade column: ``nextclade_clade`` holds the older display names, and reading
#: it as a clade is the two-hierarchies mistake (``CladeSet.legacy_name``).
STORE_COLUMNS = ("epi_isl", "accession", "nextclade_subclade", "nextclade_qc_status")


@dataclass(frozen=True)
class StoreFallback:
    """Clade calls for every sequence of one sequence-store version.

    ``assignments`` is keyed by ``(epi_isl, accession)``, the store's key. ``no_call``
    counts sequences Nextclade made no call for at all (the column is null: the sequence
    did not align), which is different from "unassigned" — an answer — and gets no row.
    """

    assignments: Mapping[tuple[str, str], Assignment]
    counts: FallbackCounts
    no_call: int
    dataset: StoreRef


def assign_from_store(store: Store, sequences: StoreRef, clade_set: CladeSet) -> StoreFallback:
    """Read the clade calls ``af.seq.nextclade`` stored with the sequences, and assign.

    The Nextclade dataset that made the calls is the ``raw/nextclade/...`` input in the
    version's provenance, and it is checked against the pinned nomenclature before any
    call is trusted (:func:`dataset_for_calls`).
    """
    dataset = dataset_for_calls(store, sequences, clade_set)
    table = read_table(store, sequences, "sequences", STORE_COLUMNS)
    rows: list[dict[str, str]] = []
    keys: dict[str, tuple[str, str]] = {}
    no_call = 0
    for record in table.to_pylist():
        if record["nextclade_subclade"] is None:
            no_call += 1
            continue
        key = (record["epi_isl"], record["accession"])
        name = f"{key[0]}|{key[1]}"
        keys[name] = key
        rows.append(
            {
                NAME_COLUMN: name,
                "subclade": record["nextclade_subclade"],
                QC_COLUMN: record["nextclade_qc_status"] or "",
            }
        )
    by_name, counts = assign_from_rows(rows, clade_set)
    assignments = {keys[name]: assignment for name, assignment in by_name.items()}
    return StoreFallback(assignments, counts, no_call, dataset)


def dataset_for_calls(store: Store, sequences: StoreRef, clade_set: CladeSet) -> StoreRef:
    """The one Nextclade dataset in ``sequences``' provenance that made its clade calls.

    A version may list more than one: B sequences are aligned against both lineages'
    datasets to place them. The one that made the calls is the one that assigns clades
    at all and agrees with the pinned nomenclature (:func:`check_dataset_agrees`); a
    dataset with no clades cannot have made any. Exactly one must qualify: none means
    the calls came from a dataset out of step with the pin, two means af cannot tell
    which vocabulary the calls are in.
    """
    provenance = json.loads((store.resolve(sequences) / "PROVENANCE.json").read_text())
    qualifying: list[StoreRef] = []
    rejected: list[str] = []
    for item in provenance["inputs"]:
        entry = item.get("store")
        if not entry or entry["kind"] != "raw" or not entry["dataset"].startswith("nextclade/"):
            continue
        ref = StoreRef.from_json(entry)
        try:
            check_dataset_agrees(store.resolve(ref) / "dataset", clade_set)
        except FallbackError as error:
            rejected.append(f"{ref.dataset}: {error}")
            continue
        qualifying.append(ref)
    if len(qualifying) != 1:
        found = ", ".join(ref.dataset for ref in qualifying) or "none"
        detail = "".join(f"\n  {line}" for line in rejected)
        raise FallbackError(
            f"{sequences}: expected one Nextclade dataset that assigns {clade_set.subtype} "
            f"clades at {clade_set.version}, found {found}{detail}"
        )
    return qualifying[0]


def _clade(row: Mapping[str, str]) -> str | None:
    for column in SUBCLADE_COLUMNS:
        if column in row:
            value = (row.get(column) or "").strip()
            return None if value in UNASSIGNED else value
    raise FallbackError(
        f"the Nextclade output has none of the columns {SUBCLADE_COLUMNS}; "
        "it was produced without the dataset's clade attributes"
    )


def disagreements(
    tree_assignments: Mapping[K, Assignment],
    fallback_assignments: Mapping[K, Assignment],
    clade_set: CladeSet,
) -> dict[K, tuple[str | None, str | None]]:
    """Where the tree engine and Nextclade disagree about the same sequences.

    Worth running wherever both are available, and reporting on the tree's review page:
    the two are independent methods over the same nomenclature, so a rise in
    disagreements is a signal about the tree or the dataset rather than noise. A clade
    and an ancestor of it do not count — the tree engine is often more specific, which is
    the point of it — only genuinely different clades do.
    """
    changed: dict[K, tuple[str | None, str | None]] = {}
    for name, tree in tree_assignments.items():
        other = fallback_assignments.get(name)
        if other is None:
            continue
        if tree.clade == other.clade:
            continue
        if (
            tree.clade is not None
            and other.clade is not None
            and clade_set.is_within(tree.clade, other.clade)
        ):
            continue
        changed[name] = (tree.clade, other.clade)
    return changed


def clades_the_dataset_cannot_assign(dataset_dir: Path, clade_set: CladeSet) -> Sequence[str]:
    """Pinned clades absent from the dataset's tree, for the run's report.

    Not an error (see :func:`check_dataset_agrees`), but it says exactly which clades the
    fallback can never produce, which is the difference a reader needs when the fallback
    and the tree engine disagree about a newly designated clade.

    Revoked clades are left out: a dataset not offering a name the nomenclature has
    withdrawn is correct, and listing five of them every run would bury the one that
    matters.
    """
    dataset = dataset_subclades(dataset_dir)
    return sorted(
        name for name in clade_set.names if name not in dataset and not clade_set[name].revoked
    )
