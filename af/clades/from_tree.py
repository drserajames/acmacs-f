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
) -> StoreRef:
    """Publish ``clades/<subtype>`` from the tree version ``tree``, which is its input."""
    if tree.kind != "trees":
        raise CladeStoreError(f"{tree}: expected a tree-store version, got kind {tree.kind!r}")
    directory = store.resolve(tree, verify=True)
    rows = rows_from_tree(directory, subtype, clade_set)
    meta = i6.read_metadata(directory)
    excluded = i6.read_excluded(directory)
    return publish(
        store,
        subtype,
        rows,
        clade_set,
        labelled=tree,
        nomenclature=nomenclature,
        started=started,
        engine="tree",
        extra_report=_tree_report(tree, meta, excluded),
    )


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
