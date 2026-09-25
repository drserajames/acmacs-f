"""Deriving ``clades/<subtype>`` from a tree-store version (I6). Synthetic data only.

The tree is workstream 5's five-leaf fixture, populated with a stub engine that names clades
from the synthetic clade set, then published to a scratch store the way the tree stage does.
"""

from __future__ import annotations

import datetime
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import duckdb
import pytest

from af.clades.from_tree import publish_from_tree, rows_from_tree
from af.clades.nomenclature import CladeSet
from af.clades.store import ASSIGNMENTS_FILE, CladeStoreError, publish, read_report
from af.store import ExternalInput, Provenance, Store, StoreRef
from af.tree.io import i6
from af.tree.populate import CladeCall, CladeInput, CladeResult, populate
from tests.tree.tree_fixtures import KEYS, built, records, states_for

from .synthetic import build_clone, load_synthetic

SUBTYPE = "A(H3N2)"
STARTED = datetime.datetime(2026, 9, 25, 9, 0, tzinfo=datetime.UTC)


def clade_set(tmp_path: Path) -> CladeSet:
    return load_synthetic(build_clone(tmp_path / "clone").parent)


def nomenclature_input(tmp_path: Path) -> ExternalInput:
    return ExternalInput.of(tmp_path / "clone" / "synthetic_HA" / "subclades")


def stub_engine(version: str, *, deeper: str = "P.1"):
    """Leaves with the K2E change are in ``deeper``, the one with a deletion is unnamed."""

    def engine(nodes: Sequence[CladeInput]) -> CladeResult:
        def call(node: CladeInput) -> CladeCall:
            if "-" in node.nucleotides:
                return CladeCall(None)
            return CladeCall(deeper if node.nucleotides.startswith("ATGG") else "P", support=1)

        return CladeResult(version, {node.node_id: call(node) for node in nodes})

    return engine


def tree_version(
    store: Store, version: str | None, *, excluded: list[dict[str, Any]] | None = None, **kw: Any
) -> StoreRef:
    tree, ids = built()
    engine = None if version is None else stub_engine(version, **kw)
    populated = populate(
        tree, "h3", records(), states_for(ids), assign_clades=engine, excluded=excluded
    )
    provenance = Provenance("trees.populate", (), {"backend": "stub"}, STARTED, STARTED)
    return i6.publish(store, populated, "report", provenance)


def test_one_row_per_leaf_keyed_by_its_sequence_ids(tmp_path: Path) -> None:
    clades = clade_set(tmp_path)
    store = Store.create(tmp_path / "store")
    tree = tree_version(store, clades.version)
    ref = publish_from_tree(
        store,
        tree,
        SUBTYPE,
        clades,
        nomenclature=[nomenclature_input(tmp_path)],
        started=STARTED,
    )
    table = duckdb.read_parquet(str(store.resolve(ref) / ASSIGNMENTS_FILE)).fetchall()
    by_key = {(row[0], row[1]): row for row in table}
    assert set(by_key) == {tuple(key.split("|")) for key in KEYS.values()}
    assert by_key[("EPI_ISL_900001", "EPI900001")][3] == "P.1"
    assert by_key[("EPI_ISL_900000", "EPI900000")][3] == "P"
    assert by_key[("EPI_ISL_900004", "EPI900004")][3] is None  # unnamed stays null
    assert {row[4] for row in table} == {"tree"}


def test_the_tree_version_is_the_input_and_the_report_says_what_is_not_labelled(
    tmp_path: Path,
) -> None:
    clades = clade_set(tmp_path)
    store = Store.create(tmp_path / "store")
    dropped = [{"leaf_id": "EPI_ISL_900099|EPI900099", "reason": "long_branch"}]
    tree = tree_version(store, clades.version, excluded=dropped)
    ref = publish_from_tree(
        store, tree, SUBTYPE, clades, nomenclature=[nomenclature_input(tmp_path)], started=STARTED
    )
    provenance = json.loads((store.resolve(ref) / "PROVENANCE.json").read_text())
    assert {"store": tree.to_json()} in provenance["inputs"]
    report = read_report(store, ref)
    assert report["sequences"] == 5
    assert report["unnamed"] == 1
    assert report["tree"] == {
        "dataset": "h3/report",
        "version": tree.version,
        "purpose": "report",
        "leaves": 5,
    }
    assert report["not_on_tree"]["excluded_before_build"] == 1
    assert report["not_on_tree"]["labelled_by"] == "fallback"


def test_refuses_a_tree_labelled_at_another_clade_set_version(tmp_path: Path) -> None:
    clades = clade_set(tmp_path)
    store = Store.create(tmp_path / "store")
    tree = tree_version(store, "synthetic_HA@0000000")
    with pytest.raises(CladeStoreError, match="assigned at synthetic_HA@0000000"):
        rows_from_tree(store.resolve(tree), SUBTYPE, clades)


def test_refuses_a_tree_populated_without_the_clade_engine(tmp_path: Path) -> None:
    clades = clade_set(tmp_path)
    store = Store.create(tmp_path / "store")
    tree = tree_version(store, None)
    with pytest.raises(CladeStoreError, match="without the clade engine"):
        rows_from_tree(store.resolve(tree), SUBTYPE, clades)


def test_refuses_a_tree_of_another_subtype(tmp_path: Path) -> None:
    clades = clade_set(tmp_path)
    store = Store.create(tmp_path / "store")
    tree = tree_version(store, clades.version)
    with pytest.raises(CladeStoreError, match="cannot label B/Vic"):
        rows_from_tree(store.resolve(tree), "B/Vic", clades)


def test_refuses_a_clade_the_clade_set_does_not_define(tmp_path: Path) -> None:
    clades = clade_set(tmp_path)
    store = Store.create(tmp_path / "store")
    tree = tree_version(store, clades.version, deeper="P.9")
    with pytest.raises(CladeStoreError, match=r"not defined at .*\['P.9'\]"):
        rows_from_tree(store.resolve(tree), SUBTYPE, clades)


def test_refuses_a_reference_that_is_not_a_tree(tmp_path: Path) -> None:
    clades = clade_set(tmp_path)
    store = Store.create(tmp_path / "store")
    sequences = StoreRef("sequences", "h3", "b" * 16, "b" * 64)
    with pytest.raises(CladeStoreError, match="expected a tree-store version"):
        publish_from_tree(
            store,
            sequences,
            SUBTYPE,
            clades,
            nomenclature=[nomenclature_input(tmp_path)],
            started=STARTED,
        )


def test_extra_report_keys_cannot_replace_the_standard_counts(tmp_path: Path) -> None:
    clades = clade_set(tmp_path)
    store = Store.create(tmp_path / "store")
    tree = tree_version(store, clades.version)
    rows = rows_from_tree(store.resolve(tree), SUBTYPE, clades)
    with pytest.raises(CladeStoreError, match="would replace standard ones"):
        publish(
            store,
            SUBTYPE,
            rows,
            clades,
            labelled=tree,
            nomenclature=[nomenclature_input(tmp_path)],
            started=STARTED,
            extra_report={"unnamed": 0},
        )
