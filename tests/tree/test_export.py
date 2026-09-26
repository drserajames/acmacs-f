"""Export (task 5.1): the tree's input files from the sequence store. Everything is invented."""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from af.run.job import JobFailed
from af.seq import select as S
from af.store import Store, StoreRef
from af.tree import export as E
from af.tree import stages
from af.tree.io import i6
from af.tree.io.fasta import read_alignment

from .test_stages import (
    OUTGROUP,
    TREE_STEPS,
    exported_project,
    fill,
    rules,
    statuses,
    with_clades,
)
from .tree_fixtures import KEYS, LEAF_SEQ


@pytest.fixture
def store(tmp_path: Path) -> Store:
    st = Store.create(tmp_path / "store")
    fill(st)
    return st


def test_export_writes_the_selection_with_its_store_version(store: Store, tmp_path: Path) -> None:
    rec = E.export(store, "h3", rules(), tmp_path / "out")
    alignment = read_alignment(tmp_path / "out" / E.ALIGNMENT_FILE)
    assert list(alignment)[0] == KEYS["o"]  # outgroup first
    assert alignment == {KEYS[n]: LEAF_SEQ[n] for n in "oabcd"}
    assert rec.sequences == store.current("sequences", "h3")
    assert (rec.leaves, rec.embargoed, rec.outgroup, rec.test_only) == (5, 1, KEYS["o"], None)
    assert [(c["rule"], c["removed"], c["remaining"]) for c in rec.counts] == [("host", 0, 4)]
    assert E.read_export(tmp_path / "out") == rec


def test_leaves_carry_embargo_and_date_precision(store: Store, tmp_path: Path) -> None:
    E.export(store, "h3", rules(), tmp_path / "out")
    rows = {r["leaf_id"]: r for r in pq.read_table(tmp_path / "out" / E.LEAVES_FILE).to_pylist()}
    assert [k for k, r in rows.items() if r["embargoed"]] == [KEYS["c"]]
    year_only = rows[KEYS["b"]]
    assert year_only["date_precision"] == "year"
    assert (year_only["collection_date_first"].month, year_only["collection_date_last"].month) == (
        1,
        12,
    )


def test_an_export_never_writes_into_a_used_directory(store: Store, tmp_path: Path) -> None:
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / "stray").write_text("x")
    with pytest.raises(E.ExportError, match="not empty"):
        E.export(store, "h3", rules(), tmp_path / "out")


def test_a_missing_outgroup_refuses_unless_a_stand_in_is_labelled(
    store: Store, tmp_path: Path
) -> None:
    absent = S.Outgroup("EPI_ISL_1", "EPI1", "not fetched yet")
    with pytest.raises(S.SelectionError, match="outgroup"):
        E.export(store, "h3", rules(absent), tmp_path / "a")
    with pytest.raises(E.ExportError, match="reason"):
        E.export(store, "h3", rules(absent), tmp_path / "b", test_only=E.TestOnly(OUTGROUP, " "))
    rec = E.export(
        store, "h3", rules(absent), tmp_path / "c", test_only=E.TestOnly(OUTGROUP, "no fetch yet")
    )
    assert rec.test_only == "no fetch yet"
    meta = json.loads((tmp_path / "c" / E.EXPORT_FILE).read_text())
    assert meta["test_only"]["configured_outgroup"] == "EPI_ISL_1|EPI1"


def test_a_ragged_alignment_is_refused(tmp_path: Path) -> None:
    st = Store.create(tmp_path / "store")
    fill(st, ragged=True)
    with pytest.raises(E.ExportError, match="rectangular"):
        E.export(st, "h3", rules(), tmp_path / "out")


def test_files_other_than_the_exported_ones_are_refused(store: Store, tmp_path: Path) -> None:
    rec = E.export(store, "h3", rules(), tmp_path / "out")
    edited = tmp_path / "edited.fasta"
    edited.write_text((tmp_path / "out" / E.ALIGNMENT_FILE).read_text().replace("ATG", "ATA", 1))
    with pytest.raises(E.ExportError, match="alignment.fasta"):
        E.check_matches(rec, edited, tmp_path / "out" / E.LEAVES_FILE)


# ---------------------------------------------------------------------------------------------
# Through the stages


def test_the_tree_records_the_sequence_version_it_was_built_from(tmp_path: Path) -> None:
    config = exported_project(tmp_path / "p")
    assert statuses(config) == dict.fromkeys(TREE_STEPS, "ran")
    store = Store.open(tmp_path / "p" / "store")
    sequences = store.current("sequences", "h3")
    build = json.loads((tmp_path / "p/work/trees/h3/build/build.json").read_text())
    assert build["source"]["sequences"] == sequences.to_json()
    tree = store.current("trees", "h3/weekly")
    version = store.version_dir(tree)
    meta = i6.read_metadata(version)
    assert (meta["embargoed_leaves"], meta["source"]["test_only"]) == (1, None)
    provenance = json.loads((version / "PROVENANCE.json").read_text())
    assert {"store": sequences.to_json()} in provenance["inputs"]
    nodes = i6.read_nodes(version, columns=["leaf_id", "embargoed"]).to_pylist()
    assert [n["leaf_id"] for n in nodes if n["embargoed"]] == [KEYS["c"]]


def test_the_clades_step_labels_from_the_same_sequence_version(tmp_path: Path) -> None:
    config = exported_project(tmp_path / "p")
    with_clades(config)
    assert statuses(config) == dict.fromkeys((*TREE_STEPS, "clades"), "ran")
    store = Store.open(tmp_path / "p" / "store")
    clades = store.version_dir(store.current("clades", "h3"))
    provenance = json.loads((clades / "PROVENANCE.json").read_text())
    assert {"store": store.current("sequences", "h3").to_json()} in provenance["inputs"]


def test_a_test_only_export_builds_only_for_a_test_purpose(tmp_path: Path) -> None:
    config = exported_project(tmp_path / "weekly", test_only=True)
    with pytest.raises(JobFailed, match="test-only"):
        stages.run(config)
    assert not (tmp_path / "weekly/work/trees/h3/build/cmaple").exists()  # before CMAPLE
    config = exported_project(tmp_path / "test", test_only=True, purpose="test-smoke")
    assert statuses(config) == dict.fromkeys(TREE_STEPS, "ran")
    store = Store.open(tmp_path / "test" / "store")
    meta = i6.read_metadata(store.version_dir(store.current("trees", "h3/test-smoke")))
    assert meta["source"]["test_only"] == "stand-in for the test"


def test_an_export_rooted_elsewhere_than_the_tree_config_is_refused(tmp_path: Path) -> None:
    config = exported_project(tmp_path / "p")
    config.write_text(config.read_text().replace(f'outgroup = "{KEYS["o"]}"', 'outgroup = "X|Y"'))
    with pytest.raises(JobFailed, match="rooted on a sequence the rules did not pin"):
        stages.run(config)


def test_a_stage_tree_meets_the_clade_stores_identity_checks(tmp_path: Path) -> None:
    """What WS4's publish_clades requires of a tree before labelling it (their guard, 26 Sep):
    one sequences-store input, which still resolves, and which holds every leaf."""
    config = exported_project(tmp_path / "p")
    statuses(config)
    store = Store.open(tmp_path / "p" / "store")
    version = store.version_dir(store.current("trees", "h3/weekly"))
    inputs = json.loads((version / "PROVENANCE.json").read_text())["inputs"]
    sequences = [i["store"] for i in inputs if i.get("store", {}).get("kind") == "sequences"]
    assert len(sequences) == 1
    held = store.resolve(StoreRef.from_json(sequences[0]))
    rows = pq.read_table(held / "sequences", columns=["epi_isl", "accession"]).to_pylist()
    leaves = i6.read_nodes(version, columns=["is_leaf", "epi_isl", "accession"]).to_pylist()
    on_tree = {(n["epi_isl"], n["accession"]) for n in leaves if n["is_leaf"]}
    assert on_tree <= {(r["epi_isl"], r["accession"]) for r in rows}
    assert len(on_tree) == 5
