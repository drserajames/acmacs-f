"""Derive ``clades/<subtype>`` from a published tree version (I6), not by assigning again.

The tree stage already runs this package's engine on every node (``af_clades_assigner`` in
``af.tree.populate``) and writes the calls into the tree's ``nodes.parquet``, beside each
leaf's ``epi_isl`` and ``accession`` read from the sequence store. So the clade table is a
projection of the tree version: running the engine a second time would make a second copy of
the same fact, free to drift from the labels the tree figure draws (design rule 6).

What this route cannot label: sequences that are not leaves of the tree. The pre-build filter's
drops are listed in the tree's ``excluded.json`` and counted in the report here; they, and
sequences no tree selected, are labelled by the fallback engine (``af.clades.fallback``), whose
rows carry ``method = "fallback"``.

Refusals, each because the alternative is a table that looks right and is not:

* the tree's clade-set version differs from the clade set given — the table would record one
  pin and carry labels made with another;
* the tree was populated without the clade engine — every leaf would read as unnamed, which is
  an answer, not a gap;
* a leaf without its ids, or whose leaf key does not match them;
* a leaf with no engine call, or with a clade the clade set does not define.
"""

from __future__ import annotations

import dataclasses
import datetime
import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from af.clades.agreement import AgreementCheck, AgreementLimit, check_agreement
from af.clades.assign import Assignment
from af.clades.fallback import StoreFallback, assign_from_store, disagreements
from af.clades.legacy import LegacyRule, legacy_labels
from af.clades.nomenclature import CladeSet
from af.clades.sequence import AlignedSequence
from af.clades.store import CladeRow, CladeStoreError, dataset_for, publish, rows_from_assignments
from af.seq.processed import read_table
from af.store import ExternalInput, Store, StoreRef
from af.store.ref import StoreError
from af.tree.io import i6
from af.tree.populate import leaf_key

#: The nodes.parquet columns this step reads.
NODE_COLUMNS = [
    "node_id",
    "is_leaf",
    "leaf_id",
    "epi_isl",
    "accession",
    "clade",
    "clade_support",
    "clade_unobservable",
    "clade_inherited",
]


def rows_from_tree(directory: Path, subtype: str, clade_set: CladeSet) -> list[CladeRow]:
    """One row per leaf of the tree version in ``directory``, keyed by its sequence ids."""
    meta = i6.read_metadata(directory)
    _check_metadata(meta, subtype, clade_set, directory)
    table = i6.read_nodes(directory, columns=NODE_COLUMNS)
    assignments: dict[str, Assignment] = {}
    identities: dict[str, tuple[str, str]] = {}
    for node in table.to_pylist():
        if not node["is_leaf"]:
            continue
        node_id = node["node_id"]
        epi_isl, accession = node["epi_isl"], node["accession"]
        if not epi_isl or not accession:
            raise CladeStoreError(f"{directory}: leaf {node_id} has no sequence ids")
        if node["leaf_id"] != leaf_key(epi_isl, accession):
            raise CladeStoreError(
                f"{directory}: leaf {node_id} is keyed {node['leaf_id']!r}, "
                f"but its ids are {epi_isl!r} and {accession!r}"
            )
        if node["clade_support"] is None:
            raise CladeStoreError(f"{directory}: leaf {node_id} has no clade call")
        assignments[node_id] = Assignment(
            node_id,
            node["clade"],
            node["clade_support"],
            node["clade_unobservable"],
            bool(node["clade_inherited"]),
        )
        identities[node_id] = (epi_isl, accession)
    unknown = sorted({a.clade for a in assignments.values() if a.clade} - set(clade_set.names))
    if unknown:
        raise CladeStoreError(
            f"{directory}: clades not defined at {clade_set.version}: {unknown[:5]}"
        )
    return rows_from_assignments(assignments, identities, subtype, method="tree")


def publish_from_tree(
    store: Store,
    tree: StoreRef,
    subtype: str,
    clade_set: CladeSet,
    *,
    nomenclature: Iterable[ExternalInput],
    started: datetime.datetime,
    sequences: StoreRef | None = None,
    agreement: AgreementLimit | None = None,
    accept_disagreement: str | None = None,
) -> StoreRef:
    """Publish ``clades/<subtype>`` from the tree version ``tree``, which is its input.

    With ``sequences``, the sequences not on the tree are labelled too, by the fallback, and
    the tree is checked against it under ``agreement`` (:func:`publish_clades`).
    """
    return publish_clades(
        store,
        subtype,
        clade_set,
        tree=tree,
        sequences=sequences,
        nomenclature=nomenclature,
        started=started,
        agreement=agreement,
        accept_disagreement=accept_disagreement,
    )


def publish_clades(
    store: Store,
    subtype: str,
    clade_set: CladeSet,
    *,
    tree: StoreRef | None,
    sequences: StoreRef | None,
    nomenclature: Iterable[ExternalInput],
    started: datetime.datetime,
    agreement: AgreementLimit | None = None,
    accept_disagreement: str | None = None,
    legacy: LegacyRule | None = None,
) -> StoreRef:
    """One ``clades/<subtype>`` table: the tree's calls, and the fallback's for the rest.

    Consumers want one clade per sequence and one place to look it up, so the two engines
    share a table and the ``method`` column says which one decided each row: the tree
    engine for tree leaves (it uses ancestry), Nextclade's stored call for every other
    sequence in ``sequences`` (it uses the sequence alone). Either input may be absent — no
    tree yet (maps are labelled before the month's tree), or no fallback wanted — but not
    both.

    A tree leaf missing from ``sequences`` is an error: the tree was built from sequences
    this version does not hold, so the two inputs do not describe the same set. Where both
    engines label a sequence the tree's call is kept, and the genuine disagreements (a
    different clade, not merely a less specific one) are counted in the report.

    With both, ``agreement`` is required: a tree that disagrees with the fallback beyond it
    is refused unless ``accept_disagreement`` names a reason (:mod:`af.clades.agreement`).

    With ``legacy``, every row the subclades leave unnamed also gets a retrospective label
    from upstream's older definitions (:mod:`af.clades.legacy`); ``nomenclature`` must then
    include the clone's ``clades/`` directory, the definitions it was read from.
    """
    nomenclature = list(nomenclature)
    if legacy is not None and not any(item.path.name == "clades" for item in nomenclature):
        raise CladeStoreError(
            f"{subtype}: a legacy label needs the nomenclature's clades/ directory among the "
            "nomenclature inputs, so the definitions it came from are in the provenance"
        )
    if tree is None and sequences is None:
        raise CladeStoreError(f"{subtype}: give a tree version, a sequence version, or both")
    if tree is not None and sequences is not None and agreement is None:
        raise CladeStoreError(
            f"{subtype}: a tree with a fallback needs an agreement limit (af.clades.agreement); "
            "a tree is never published unchecked when there is something to check it against"
        )
    if accept_disagreement is not None and (tree is None or sequences is None):
        raise CladeStoreError(
            f"{subtype}: accept_disagreement given, but there is no tree and fallback to compare"
        )
    rows: list[CladeRow] = []
    tree_sequences: StoreRef | None = None
    check: AgreementCheck | None = None
    report: dict[str, Any] = {}
    inputs: list[StoreRef] = []
    if tree is not None:
        if tree.kind != "trees":
            raise CladeStoreError(f"{tree}: expected a tree-store version, got kind {tree.kind!r}")
        directory = store.resolve(tree, verify=True)
        rows = rows_from_tree(directory, subtype, clade_set)
        tree_sequences = check_tree_identities(store, tree, rows)
        report = _tree_report(tree, i6.read_metadata(directory), i6.read_excluded(directory))
        inputs.append(tree)
    if sequences is not None:
        if sequences.kind != "sequences":
            raise CladeStoreError(
                f"{sequences}: expected a sequence-store version, got kind {sequences.kind!r}"
            )
        fallback = assign_from_store(store, sequences, clade_set)
        rows, fallback_report, check = _with_fallback(
            rows, fallback, subtype, clade_set, agreement, accept_disagreement
        )
        report["fallback"] = fallback_report
        if check is not None:
            report["agreement"] = check.to_json()
        report.setdefault("not_on_tree", {})["labelled_by"] = "fallback, in this table"
        inputs += [sequences, fallback.dataset]
    if legacy is not None:
        source = sequences or tree_sequences
        assert source is not None  # a tree without sequences still names its sequence version
        rows, report["legacy"] = _with_legacy(store, rows, source, clade_set, legacy)
    return publish(
        store,
        subtype,
        rows,
        clade_set,
        labelled=inputs[0],
        nomenclature=nomenclature,
        started=started,
        engine="+".join(name for name, ref in (("tree", tree), ("fallback", sequences)) if ref),
        extra_inputs=inputs[1:],
        extra_report=report,
        # the limit, its reason and any override go into provenance too: they decided
        # whether this version could exist at all
        extra_parameters={
            **({} if check is None else {"agreement": check.provenance()}),
            **({} if legacy is None else {"legacy": legacy.to_json()}),
        }
        or None,
    )


def check_tree_identities(store: Store, tree: StoreRef, rows: Sequence[CladeRow]) -> StoreRef:
    """Refuse a tree whose leaf ids were never taken from the sequence store.

    Leaf ids that merely look well formed prove nothing: a tree imported from outside af
    can carry stand-in ids, and a clade table keyed by them would replace the real one
    and match nothing in any join. So the tree must name, in its ``PROVENANCE.json``
    ``inputs``, a store input of kind ``sequences``; that version must still be in this
    store; and every leaf must be one of its sequences. Returns that sequence version.
    """
    provenance = json.loads((store.resolve(tree) / "PROVENANCE.json").read_text())
    named = [
        StoreRef.from_json(item["store"])
        for item in provenance.get("inputs", [])
        if isinstance(item, dict) and item.get("store", {}).get("kind") == "sequences"
    ]
    if len(named) != 1:
        raise CladeStoreError(
            f"{tree}: PROVENANCE.json 'inputs' must name exactly one store input of kind "
            f"'sequences' (the version its leaves were read from); found {len(named)}. "
            "A tree not built from the sequence store has no checked sequence identities."
        )
    sequences = named[0]
    try:
        store.resolve(sequences)
    except StoreError as error:
        raise CladeStoreError(
            f"{tree}: its sequences input {sequences.dataset}@{sequences.version} is not in "
            f"this store ({error})"
        ) from error
    table = read_table(store, sequences, "sequences", ["epi_isl", "accession"])
    known = set(zip(table["epi_isl"].to_pylist(), table["accession"].to_pylist(), strict=True))
    unknown = [
        (row.epi_isl, row.accession) for row in rows if (row.epi_isl, row.accession) not in known
    ]
    if unknown:
        raise CladeStoreError(
            f"{tree}: {len(unknown)} of {len(rows)} leaves are not sequences of "
            f"{sequences.dataset}@{sequences.version}, e.g. {unknown[:3]}"
        )
    return sequences


def _with_fallback(
    tree_rows: list[CladeRow],
    fallback: StoreFallback,
    subtype: str,
    clade_set: CladeSet,
    agreement: AgreementLimit | None,
    accept_disagreement: str | None,
) -> tuple[list[CladeRow], dict[str, Any], AgreementCheck | None]:
    """Tree rows plus a fallback row for every sequence not on the tree.

    With tree rows, the tree is first checked against the fallback under ``agreement``
    (which :func:`publish_clades` has made sure is given).
    """
    on_tree = {(row.epi_isl, row.accession): row for row in tree_rows}
    missing = sorted(set(on_tree) - set(fallback.assignments))
    if missing:
        raise CladeStoreError(
            f"{len(missing)} tree leaves are not in the sequence version, e.g. {missing[:3]}: "
            "the tree was built from other sequences"
        )
    tree_calls = {key: Assignment(str(key), row.clade) for key, row in on_tree.items()}
    both = {key: fallback.assignments[key] for key in on_tree}
    disagree = disagreements(tree_calls, both, clade_set)
    check = None
    if tree_rows:
        assert agreement is not None
        check = check_agreement(
            tree_calls, both, clade_set, agreement, accept_disagreement=accept_disagreement
        )
    rows = list(tree_rows)
    for (epi_isl, accession), assignment in sorted(fallback.assignments.items()):
        if (epi_isl, accession) not in on_tree:
            rows.append(CladeRow(epi_isl, accession, subtype, assignment.clade, method="fallback"))
    return (
        rows,
        {
            "dataset": {"dataset": fallback.dataset.dataset, "version": fallback.dataset.version},
            **fallback.counts.to_json(),
            "no_call": fallback.no_call,
            "labelled_here": len(rows) - len(tree_rows),
            "tree_vs_fallback_disagree": len(disagree),
            "tree_vs_fallback_compared": len(both),
        },
        check,
    )


def _with_legacy(
    store: Store,
    rows: list[CladeRow],
    sequences: StoreRef,
    clade_set: CladeSet,
    rule: LegacyRule,
) -> tuple[list[CladeRow], dict[str, Any]]:
    """The rows, with a legacy label on each the subclades leave unnamed, and the counts."""
    unnamed = {(row.epi_isl, row.accession) for row in rows if row.clade is None}
    table = read_table(store, sequences, "sequences", ["epi_isl", "accession", "nuc_aligned"])
    aligned = {
        (epi_isl, accession): nucleotides
        for epi_isl, accession, nucleotides in zip(
            table["epi_isl"].to_pylist(),
            table["accession"].to_pylist(),
            table["nuc_aligned"].to_pylist(),
            strict=True,
        )
        if (epi_isl, accession) in unnamed
    }
    missing = sorted(unnamed - set(aligned))
    if missing:
        raise CladeStoreError(
            f"{len(missing)} unnamed rows are not sequences of {sequences}, e.g. {missing[:3]}"
        )
    calls, counts = legacy_labels(
        {
            key: None if not nucleotides else AlignedSequence.from_nucleotides(nucleotides)
            for key, nucleotides in aligned.items()
        },
        clade_set,
        rule,
    )
    labelled = [
        row
        if row.clade is not None
        else dataclasses.replace(
            row,
            legacy_clade=calls[(row.epi_isl, row.accession)].clade,
            legacy_tolerated=calls[(row.epi_isl, row.accession)].tolerated,
        )
        for row in rows
    ]
    return labelled, {**rule.to_json(), "unnamed": len(unnamed), **counts.to_json()}


def _check_metadata(
    meta: Mapping[str, Any], subtype: str, clade_set: CladeSet, directory: Path
) -> None:
    dataset = dataset_for(subtype)
    if meta["subtype"] != dataset:
        raise CladeStoreError(
            f"{directory}: a {meta['subtype']!r} tree cannot label {subtype} ({dataset!r})"
        )
    version = meta.get("clade_set_version")
    if version is None:
        raise CladeStoreError(
            f"{directory}: the tree was populated without the clade engine; it holds no clades"
        )
    if version != clade_set.version:
        raise CladeStoreError(
            f"{directory}: the tree's clades were assigned at {version}, "
            f"not at the clade set given ({clade_set.version})"
        )


def _tree_report(
    tree: StoreRef, meta: Mapping[str, Any], excluded: list[dict[str, Any]]
) -> dict[str, Any]:
    """What a reader needs to know about the sequences this table does *not* label."""
    return {
        "tree": {
            "dataset": tree.dataset,
            "version": tree.version,
            "purpose": meta["purpose"],
            "leaves": meta["leaves"],
        },
        "not_on_tree": {
            "excluded_before_build": len(excluded),
            "labelled_by": "fallback",
            "why": (
                "only tree leaves are labelled here; the pre-build filter's drops and "
                "sequences no tree selected take their clade from the fallback engine"
            ),
        },
    }
