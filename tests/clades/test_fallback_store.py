"""The fallback read from the sequence store, and one clade table from tree + fallback.

Synthetic data only: a scratch store with a Nextclade dataset (just its ``tree.json``), a
sequence version naming it in its provenance, and workstream 5's five-leaf tree.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from af.clades.fallback import FallbackError, assign_from_store, dataset_for_calls
from af.clades.from_tree import publish_clades
from af.clades.store import ASSIGNMENTS_FILE, CladeStoreError, read_report
from af.store import Provenance, Store, StoreRef

from .test_from_tree import STARTED, SUBTYPE, clade_set, nomenclature_input, tree_version

#: (epi_isl, accession, nextclade_subclade, qc). The first five are the tree fixture's
#: leaves; the stub tree engine calls them P, P.1, P.1, P, None.
TREE_LEAVES = [(f"EPI_ISL_90000{i}", f"EPI90000{i}") for i in range(5)]


def raw_dataset(store: Store, name: str, *clades: str) -> StoreRef:
    with store.build("raw", f"nextclade/test/{name}") as builder:
        directory = builder.path / "dataset"
        directory.mkdir()
        children = [{"node_attrs": {"subclade": {"value": clade}}} for clade in clades]
        tree = {"tree": {"node_attrs": {}, "children": children}}
        (directory / "tree.json").write_text(json.dumps(tree))
        return builder.publish(Provenance("test", (), {}, STARTED, STARTED))


def sequence_version(
    store: Store, rows: list[tuple[str, str, str | None, str | None]], *datasets: StoreRef
) -> StoreRef:
    columns: dict[str, list[Any]] = {
        name: [row[index] for row in rows]
        for index, name in enumerate(
            ["epi_isl", "accession", "nextclade_subclade", "nextclade_qc_status"]
        )
    }
    with store.build("sequences", "h3") as builder:
        part = builder.path / "sequences" / "pull=test"
        part.mkdir(parents=True)
        pq.write_table(pa.table(columns), part / "part-0.parquet")
        return builder.publish(Provenance("seq.build", tuple(datasets), {}, STARTED, STARTED))


def standard(store: Store, *, extra: list[tuple[str, str, str | None, str | None]] = ()):
    """The tree leaves as Nextclade sees them (P.2 where the tree says P.1 for leaf 2, a
    genuine disagreement; P where the tree says P.1 for leaf 1, merely less specific), plus
    three sequences off the tree: named, unassigned, no call."""
    dataset = raw_dataset(store, "good", "P", "P.1", "P.1.1", "P.2")
    empty = raw_dataset(store, "other-lineage")  # aligns, assigns no clades at all
    rows = [
        (*TREE_LEAVES[0], "P", "good"),
        (*TREE_LEAVES[1], "P", "good"),
        (*TREE_LEAVES[2], "P.2", "good"),
        (*TREE_LEAVES[3], "P", "good"),
        (*TREE_LEAVES[4], "unassigned", "good"),
        ("EPI_ISL_900010", "EPI900010", "P.1.1", "bad"),
        ("EPI_ISL_900011", "EPI900011", "unassigned", "mediocre"),
        ("EPI_ISL_900012", "EPI900012", None, None),
        *extra,
    ]
    return sequence_version(store, rows, dataset, empty), dataset


def test_assigns_from_the_stored_calls(tmp_path: Path) -> None:
    store = Store.create(tmp_path / "store")
    sequences, dataset = standard(store)
    result = assign_from_store(store, sequences, clade_set(tmp_path))
    assert result.dataset == dataset
    assert result.assignments[("EPI_ISL_900010", "EPI900010")].clade == "P.1.1"
    assert result.assignments[("EPI_ISL_900011", "EPI900011")].clade is None
    assert ("EPI_ISL_900012", "EPI900012") not in result.assignments  # no call, no row
    assert result.no_call == 1
    assert result.counts.to_json() == {"sequences": 7, "assigned": 5, "unassigned": 2, "qc_bad": 1}


def one(tag: str) -> list[tuple[str, str, str | None, str | None]]:
    """A one-row table, distinct per tag: an identical rebuild publishes no new version,
    so versions that should differ in provenance must differ in content too."""
    return [(f"EPI_ISL_7{tag}", f"EPI7{tag}", "P", "good")]


def test_the_dataset_is_the_one_that_assigns_the_pinned_clades(tmp_path: Path) -> None:
    store = Store.create(tmp_path / "store")
    clades = clade_set(tmp_path)
    good = raw_dataset(store, "good", "P", "P.1")
    foreign = raw_dataset(store, "foreign", "Z.1")
    assert (
        dataset_for_calls(store, sequence_version(store, one("1"), good, foreign), clades) == good
    )
    with pytest.raises(FallbackError, match=r"found none[\s\S]*Z\.1"):
        dataset_for_calls(store, sequence_version(store, one("2"), foreign), clades)
    twin = raw_dataset(store, "twin", "P")
    with pytest.raises(FallbackError, match="found nextclade/test/good, nextclade/test/twin"):
        dataset_for_calls(store, sequence_version(store, one("3"), good, twin), clades)


def test_a_call_outside_the_pinned_nomenclature_is_fatal(tmp_path: Path) -> None:
    store = Store.create(tmp_path / "store")
    sequences = sequence_version(
        store, [("EPI_ISL_1", "EPI1", "P.7", "good")], raw_dataset(store, "good", "P")
    )
    with pytest.raises(FallbackError, match="'P.7'"):
        assign_from_store(store, sequences, clade_set(tmp_path))


def read_rows(store: Store, ref: StoreRef) -> dict[tuple[str, str], tuple[str | None, str]]:
    table = duckdb.read_parquet(str(store.resolve(ref) / ASSIGNMENTS_FILE))
    return {(r[0], r[1]): (r[3], r[4]) for r in table.fetchall()}


def test_before_the_tree_every_sequence_is_labelled_by_the_fallback(tmp_path: Path) -> None:
    clades = clade_set(tmp_path)
    store = Store.create(tmp_path / "store")
    sequences, dataset = standard(store)
    ref = publish_clades(
        store,
        SUBTYPE,
        clades,
        tree=None,
        sequences=sequences,
        nomenclature=[nomenclature_input(tmp_path)],
        started=STARTED,
    )
    rows = read_rows(store, ref)
    assert len(rows) == 7 and {method for _, method in rows.values()} == {"fallback"}
    provenance = json.loads((store.resolve(ref) / "PROVENANCE.json").read_text())
    assert {"store": sequences.to_json()} in provenance["inputs"]
    assert {"store": dataset.to_json()} in provenance["inputs"]
    assert provenance["parameters"]["engine"] == "fallback"
    report = read_report(store, ref)
    assert report["fallback"]["no_call"] == 1 and report["fallback"]["labelled_here"] == 7


def test_one_table_tree_calls_for_leaves_fallback_for_the_rest(tmp_path: Path) -> None:
    clades = clade_set(tmp_path)
    store = Store.create(tmp_path / "store")
    tree = tree_version(store, clades.version)
    sequences, _ = standard(store)
    ref = publish_clades(
        store,
        SUBTYPE,
        clades,
        tree=tree,
        sequences=sequences,
        nomenclature=[nomenclature_input(tmp_path)],
        started=STARTED,
    )
    rows = read_rows(store, ref)
    assert [rows[key] for key in TREE_LEAVES] == [
        ("P", "tree"),
        ("P.1", "tree"),
        ("P.1", "tree"),  # the tree's call is kept where the two disagree
        ("P", "tree"),
        (None, "tree"),
    ]
    assert rows[("EPI_ISL_900010", "EPI900010")] == ("P.1.1", "fallback")
    assert ("EPI_ISL_900012", "EPI900012") not in rows
    report = read_report(store, ref)
    assert report["methods"] == {"fallback": 2, "tree": 5}
    # leaf 2 is P.1 on the tree and P.2 by Nextclade: different clades. Leaf 1 (P.1 vs P) is
    # only less specific, and leaf 4 (None vs unassigned) agrees, so neither counts.
    assert report["fallback"]["tree_vs_fallback_disagree"] == 1
    assert report["fallback"]["tree_vs_fallback_compared"] == 5
    assert report["not_on_tree"]["labelled_by"] == "fallback, in this table"
    engine = json.loads((store.resolve(ref) / "PROVENANCE.json").read_text())["parameters"]
    assert engine["engine"] == "tree+fallback"


def test_a_tree_leaf_missing_from_the_sequences_is_fatal(tmp_path: Path) -> None:
    clades = clade_set(tmp_path)
    store = Store.create(tmp_path / "store")
    tree = tree_version(store, clades.version)
    dataset = raw_dataset(store, "good", "P")
    sequences = sequence_version(store, [(*TREE_LEAVES[0], "P", "good")], dataset)
    with pytest.raises(CladeStoreError, match="4 tree leaves are not in the sequence version"):
        publish_clades(
            store,
            SUBTYPE,
            clades,
            tree=tree,
            sequences=sequences,
            nomenclature=[nomenclature_input(tmp_path)],
            started=STARTED,
        )


def test_needs_a_tree_or_sequences_of_the_right_kind(tmp_path: Path) -> None:
    clades = clade_set(tmp_path)
    store = Store.create(tmp_path / "store")
    kwargs: dict[str, Any] = {"nomenclature": [], "started": STARTED}
    with pytest.raises(CladeStoreError, match="give a tree version, a sequence version"):
        publish_clades(store, SUBTYPE, clades, tree=None, sequences=None, **kwargs)
    tree = tree_version(store, clades.version)
    with pytest.raises(CladeStoreError, match="expected a sequence-store version"):
        publish_clades(store, SUBTYPE, clades, tree=None, sequences=tree, **kwargs)
