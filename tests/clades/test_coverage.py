"""A clades table is behind its sequences in content only when a refresh would change it."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from af.clades.coverage import calls_fingerprint, clades_behind_in_content
from af.clades.from_tree import publish_clades
from af.store import Provenance, Store, StoreRef

from .test_fallback_store import raw_dataset
from .test_from_tree import STARTED, SUBTYPE, clade_set, nomenclature_input

ROWS: list[tuple[str, ...]] = [
    ("EPI_ISL_920001", "EPI920001", "P", "good", "virus one"),
    ("EPI_ISL_920002", "EPI920002", "P.1", "good", "virus two"),
]


def version(store: Store, rows: Sequence[tuple[str, ...]], *inputs: StoreRef) -> StoreRef:
    """A sequences version with a name column, so two versions can differ only in a name."""
    names = ["epi_isl", "accession", "nextclade_subclade", "nextclade_qc_status", "name"]
    columns = {name: [row[i] for row in rows] for i, name in enumerate(names)}
    with store.build("sequences", "h3") as builder:
        part = builder.path / "sequences" / "pull=test"
        part.mkdir(parents=True)
        pq.write_table(pa.table(columns), part / "part-0.parquet")
        return builder.publish(Provenance("seq.build", tuple(inputs), {}, STARTED, STARTED))


def labelled(tmp_path: Path) -> tuple[Store, StoreRef, StoreRef, StoreRef]:
    store = Store.create(tmp_path / "store")
    dataset = raw_dataset(store, "good", "P", "P.1")
    first = version(store, ROWS, dataset)
    clades = publish_clades(
        store,
        SUBTYPE,
        clade_set(tmp_path),
        tree=None,
        sequences=first,
        nomenclature=[nomenclature_input(tmp_path)],
        started=STARTED,
    )
    return store, dataset, first, clades


def test_a_republish_that_changes_no_call_is_not_behind(tmp_path: Path) -> None:
    """A fixed name moves the version label (the old signal said "behind") but not a call."""
    store, dataset, first, clades = labelled(tmp_path)
    renamed = version(store, [(*row[:4], row[4].upper()) for row in ROWS], dataset)
    result = clades_behind_in_content(store, clades, renamed)
    assert not result.same_version  # what the label comparison sees
    assert not result.behind  # what a refresh would do: nothing
    assert calls_fingerprint(store, first) == calls_fingerprint(store, renamed)


def test_an_added_call_is_behind(tmp_path: Path) -> None:
    store, dataset, _, clades = labelled(tmp_path)
    grown = version(store, [*ROWS, ("EPI_ISL_920003", "EPI920003", "P", "good", "three")], dataset)
    assert clades_behind_in_content(store, clades, grown).behind


def test_a_changed_call_is_behind(tmp_path: Path) -> None:
    store, dataset, _, clades = labelled(tmp_path)
    moved = version(store, [ROWS[0], (*ROWS[1][:2], "P", *ROWS[1][3:])], dataset)
    assert clades_behind_in_content(store, clades, moved).behind


def test_another_nextclade_dataset_is_behind(tmp_path: Path) -> None:
    """Same calls from a different dataset are not the same input: the vocabulary may differ.

    The rows differ in a name too: the store returns the existing version for identical content,
    whatever the provenance, so a version differing only in its dataset cannot exist."""
    store, _, _, clades = labelled(tmp_path)
    other = raw_dataset(store, "other", "P", "P.1")
    renamed = [(*row[:4], row[4].upper()) for row in ROWS]
    assert clades_behind_in_content(store, clades, version(store, renamed, other)).behind


def test_the_same_version_is_never_behind(tmp_path: Path) -> None:
    store, _, first, clades = labelled(tmp_path)
    result = clades_behind_in_content(store, clades, first)
    assert result.same_version and not result.behind
