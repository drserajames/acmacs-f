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

import datetime
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from af.clades.assign import Assignment
from af.clades.fallback import StoreFallback, assign_from_store, disagreements
from af.clades.nomenclature import CladeSet
from af.clades.store import CladeRow, CladeStoreError, dataset_for, publish, rows_from_assignments
from af.store import ExternalInput, Store, StoreRef
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
) -> StoreRef:
    """Publish ``clades/<subtype>`` from the tree version ``tree``, which is its input.

    With ``sequences``, the sequences not on the tree are labelled too, by the fallback
    (:func:`publish_clades`).
    """
    return publish_clades(
        store,
        subtype,
        clade_set,
        tree=tree,
        sequences=sequences,
        nomenclature=nomenclature,
        started=started,
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
    """
    if tree is None and sequences is None:
        raise CladeStoreError(f"{subtype}: give a tree version, a sequence version, or both")
    rows: list[CladeRow] = []
    report: dict[str, Any] = {}
    inputs: list[StoreRef] = []
    if tree is not None:
        if tree.kind != "trees":
            raise CladeStoreError(f"{tree}: expected a tree-store version, got kind {tree.kind!r}")
        directory = store.resolve(tree, verify=True)
        rows = rows_from_tree(directory, subtype, clade_set)
        report = _tree_report(tree, i6.read_metadata(directory), i6.read_excluded(directory))
        inputs.append(tree)
    if sequences is not None:
        if sequences.kind != "sequences":
            raise CladeStoreError(
                f"{sequences}: expected a sequence-store version, got kind {sequences.kind!r}"
            )
        fallback = assign_from_store(store, sequences, clade_set)
        rows, fallback_report = _with_fallback(rows, fallback, subtype, clade_set)
        report["fallback"] = fallback_report
        report.setdefault("not_on_tree", {})["labelled_by"] = "fallback, in this table"
        inputs += [sequences, fallback.dataset]
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
    )


def _with_fallback(
    tree_rows: list[CladeRow], fallback: StoreFallback, subtype: str, clade_set: CladeSet
) -> tuple[list[CladeRow], dict[str, Any]]:
    """Tree rows plus a fallback row for every sequence not on the tree."""
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
    rows = list(tree_rows)
    for (epi_isl, accession), assignment in sorted(fallback.assignments.items()):
        if (epi_isl, accession) not in on_tree:
            rows.append(CladeRow(epi_isl, accession, subtype, assignment.clade, method="fallback"))
    return rows, {
        "dataset": {"dataset": fallback.dataset.dataset, "version": fallback.dataset.version},
        **fallback.counts.to_json(),
        "no_call": fallback.no_call,
        "labelled_here": len(rows) - len(tree_rows),
        "tree_vs_fallback_disagree": len(disagree),
        "tree_vs_fallback_compared": len(both),
    }


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
