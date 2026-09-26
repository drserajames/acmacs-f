"""Populating a tree, and the I6 files it is written to. Synthetic data only.

The tree is described in ``tree_fixtures.py``.
"""

from __future__ import annotations

import datetime
from collections.abc import Sequence
from pathlib import Path

import pytest

from af.store import Provenance, Store
from af.tree.asr.base import GapCapabilityError
from af.tree.io import i6, newick
from af.tree.populate import (
    CladeAssigner,
    CladeCall,
    CladeInput,
    CladeResult,
    PopulateError,
    nucleotide_changes,
    populate,
)

from .tree_fixtures import KEYS, GapBlind, built, records, states_for


def test_changes_ignore_gaps_and_ambiguities() -> None:
    assert nucleotide_changes("ACGTA", "ACGTG") == [(5, "A", "G")]
    assert nucleotide_changes("ACG", "N-G") == []


def test_a_leaf_with_no_record_is_an_error() -> None:
    tree, ids = built()
    partial = records()
    del partial[KEYS["c"]]
    with pytest.raises(PopulateError, match="1 of 5 tree leaves have no sequence-store record"):
        populate(tree, "h3", partial, states_for(ids))


def test_a_node_the_backend_dropped_is_an_error() -> None:
    tree, ids = built()
    states = states_for(ids)
    del states.nucleotides[ids["y"]]
    with pytest.raises(PopulateError, match="1 internal nodes have no ancestral state"):
        populate(tree, "h3", records(), states)


def test_mutation_scale_is_changes_over_length_and_collapses_unchanged_branches() -> None:
    tree, ids = built()
    result = populate(tree, "h3", records(), states_for(ids), branch_scale="mutations")
    assert result.counts["unchanged_branches_collapsed"] == 1  # x->z carried nothing
    by_name = {leaf.name: leaf for leaf in result.tree.leaves()}
    assert by_name[KEYS["a"]].branch_length == 0.0  # y->a: identical
    assert by_name[KEYS["c"]].branch_length == pytest.approx(1 / 9)
    assert by_name[KEYS["d"]].branch_length == 0.0  # gaps are not changes
    y = result.tree.by_id(ids["y"])
    assert y.branch_length == pytest.approx(1 / 9)
    assert result.aa_subs[ids["y"]] == ["K2E"]
    # c's parent is now x, and the ids of the nodes that remain did not change
    assert by_name[KEYS["c"]].parent is result.tree.by_id(ids["x"])
    assert result.ml_lengths[ids["y"]] == pytest.approx(0.05)


def test_ml_scale_keeps_lengths_and_every_node() -> None:
    tree, ids = built()
    result = populate(tree, "h3", records(), states_for(ids), branch_scale="ml")
    assert result.counts["unchanged_branches_collapsed"] == 0
    assert result.tree.by_id(ids["z"]).branch_length == pytest.approx(0.01)
    assert result.nuc_changes[ids["z"]] == 0


def test_mutation_scale_needs_states() -> None:
    tree, _ = built()
    with pytest.raises(PopulateError, match="need ancestral states"):
        populate(tree, "h3", records(), None, branch_scale="mutations")


def test_clades_come_from_the_injected_engine_with_gap_capability_marked() -> None:
    tree, ids = built()
    seen: list[CladeInput] = []

    def engine(nodes: Sequence[CladeInput]) -> CladeResult:
        seen.extend(nodes)
        calls = {n.node_id: CladeCall("J.2", support=2) for n in nodes}
        return CladeResult("nomenclature@abc1234", calls, {"J": None, "J.2": "J"})

    result = populate(
        tree, "h3", records(), states_for(ids), assign_clades=engine, backend=GapBlind()
    )
    assert len(seen) == 9
    assert all(n.gaps_reconstructed for n in seen if n.is_leaf)
    assert not any(n.gaps_reconstructed for n in seen if not n.is_leaf)
    assert result.clade_set_version == "nomenclature@abc1234"
    assert result.clade_parents["J.2"] == "J"
    assert result.counts["leaves_without_clade"] == 0


def test_a_node_the_clade_engine_skipped_is_an_error() -> None:
    tree, ids = built()

    def engine(nodes: Sequence[CladeInput]) -> CladeResult:
        return CladeResult("v", {n.node_id: CladeCall("J") for n in nodes[1:]})

    with pytest.raises(PopulateError, match="no call for 1 nodes"):
        populate(tree, "h3", records(), states_for(ids), assign_clades=engine)


def populated_example():
    tree, ids = built()
    result = populate(
        tree,
        "h3",
        records(),
        states_for(ids),
        branch_scale="mutations",
        continent_of=lambda record: "EUROPE" if record.region == "Europe" else None,
    )
    return result, ids


def test_i6_nodes_are_preorder_and_carry_the_draw_columns(tmp_path: Path) -> None:
    result, ids = populated_example()
    i6.write(result, tmp_path, "report")
    table = i6.read_nodes(tmp_path)
    parent = table["parent"].to_pylist()
    assert parent[0] == -1
    assert all(0 <= p < row for row, p in enumerate(parent) if row)
    columns = i6.draw_columns(tmp_path)
    assert set(columns) >= {"parent", "edge", "leaf_id", "date_precision", "aa_subs", "subtype"}
    b = columns["leaf_id"].index(KEYS["b"])
    assert columns["date_precision"][b] == "year"
    assert columns["continent"][b] == "EUROPE"
    row_y = table["node_id"].to_pylist().index(f"{ids['y']:016x}")
    assert table["aa_subs"][row_y].as_py() == ["K2E"]
    assert table["n_leaves"][0].as_py() == 5
    assert table["titrated_by"][row_y].as_py() == []
    meta = i6.read_metadata(tmp_path)
    assert meta["counts"]["continents"] == {"EUROPE": 5}
    assert meta["branch_scale"] == "mutations" and meta["alignment_length"] == 9
    assert len(i6.read_ancestral(tmp_path)) == meta["internal_nodes"]


def test_the_newick_carries_stable_internal_ids(tmp_path: Path) -> None:
    result, ids = populated_example()
    i6.write(result, tmp_path, "report")
    reread = newick.load(tmp_path / i6.TREE_FILE)
    labels = {node.name for node in reread.internal()}
    assert f"{ids['y']:016x}" in labels


def test_an_identical_rebuild_is_byte_identical_and_the_same_store_version(tmp_path: Path) -> None:
    store = Store.create(tmp_path / "store")
    moment = datetime.datetime(2026, 9, 25, tzinfo=datetime.UTC)
    provenance = Provenance("trees.populate", (), {"backend": "stub"}, moment, moment)
    first = i6.publish(store, populated_example()[0], "report", provenance)
    second = i6.publish(store, populated_example()[0], "report", provenance)
    assert first == second
    assert first.dataset == "h3/report"
    assert (store.resolve(first) / i6.NODES_FILE).exists()


def test_a_gap_blind_backend_reports_no_substitution_into_a_deletion() -> None:
    """raxml-ng puts a residue where leaves have a deletion; that is unobservable, not a change."""
    tree, ids = built()
    d = next(leaf for leaf in tree.leaves() if leaf.name == KEYS["d"])
    capable = populate(tree, "h3", records(), states_for(ids))
    assert capable.aa_subs[d.node_id] == ["P3-"]
    tree, ids = built()
    blind = populate(tree, "h3", records(), states_for(ids), backend=GapBlind())
    assert blind.aa_subs[d.node_id] == []


def test_a_clade_outside_the_hierarchy_is_an_error() -> None:
    tree, ids = built()

    def engine(nodes: Sequence[CladeInput]) -> CladeResult:
        return CladeResult("v", {n.node_id: CladeCall("K") for n in nodes}, {"J": None})

    with pytest.raises(PopulateError, match="not in the clade set's hierarchy"):
        populate(tree, "h3", records(), states_for(ids), assign_clades=engine)


def _deletion_engine(positions: tuple[int, ...]) -> CladeAssigner:
    """A clade engine whose clade set defines a clade by a deletion (as B/Vic's does)."""

    def engine(nodes: Sequence[CladeInput]) -> CladeResult:
        calls = {node.node_id: CladeCall("C.5") for node in nodes}
        return CladeResult("v", calls, {"C.5": None}, positions)

    return engine


def test_a_gap_blind_backend_is_refused_where_deletions_define_clades() -> None:
    """DECISIONS 25 Sep: raxml-fed B/Vic must not be labelled quietly."""
    tree, ids = built()
    with pytest.raises(GapCapabilityError, match="cannot reconstruct deletions"):
        populate(
            tree,
            "bvic",
            records(),
            states_for(ids),
            backend=GapBlind(),
            assign_clades=_deletion_engine((163, 164)),
        )


def test_the_gap_blind_refusal_can_be_overridden_and_is_then_counted() -> None:
    tree, ids = built()
    result = populate(
        tree,
        "bvic",
        records(),
        states_for(ids),
        backend=GapBlind(),
        assign_clades=_deletion_engine((163, 164)),
        allow_gap_blind=True,
    )
    assert result.counts["gap_blind_override"] == 2


def test_a_gap_blind_backend_is_fine_where_no_clade_needs_a_deletion() -> None:
    """H3 has no deletion-defined clade, so the same backend is allowed there."""
    tree, ids = built()
    result = populate(
        tree,
        "h3",
        records(),
        states_for(ids),
        backend=GapBlind(),
        assign_clades=_deletion_engine(()),
    )
    assert "gap_blind_override" not in result.counts


def test_the_prebuild_drops_are_written_beside_the_tree(tmp_path: Path) -> None:
    """A count is not enough: WS11 needs "dropped by the filter" to differ from "lost"."""
    tree, ids = built()
    dropped = [{"leaf_id": "EPI_ISL_9001|EPI9001", "reason": "long_branch", "branch_length": 0.05}]
    result = populate(tree, "h3", records(), states_for(ids), excluded=dropped)
    assert result.counts["excluded_before_build"] == 1
    i6.write(result, tmp_path, "weekly")
    assert i6.read_excluded(tmp_path) == dropped
    assert i6.read_metadata(tmp_path)["excluded_before_build"] == 1


def test_no_excluded_file_is_written_when_nothing_was_dropped(tmp_path: Path) -> None:
    tree, ids = built()
    i6.write(populate(tree, "h3", records(), states_for(ids)), tmp_path, "weekly")
    assert not (tmp_path / i6.EXCLUDED_FILE).exists()
    assert i6.read_excluded(tmp_path) == []


def test_a_continent_outside_the_legend_is_refused() -> None:
    """GISAID's region names are not the figure's vocabulary; grey without a word is a bug."""
    tree, ids = built()
    with pytest.raises(PopulateError, match="outside the figure's legend"):
        populate(tree, "h3", records(), states_for(ids), continent_of=lambda r: r.region)
