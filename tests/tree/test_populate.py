"""Populating a tree, and the I6 files it is written to. Synthetic data only.

The tree used throughout (outgroup ``o``)::

    root ─┬─ o
          └─ x ─┬─ y ─┬─ a
                │     └─ b
                └─ z ─┬─ c
                      └─ d

Internal sequences are chosen so that branch x→z carries no nucleotide change (it must be
collapsed on the mutation scale) and x→y carries one codon change.
"""

from __future__ import annotations

import datetime
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest

from af.store import Provenance, Store
from af.tree.asr.base import AncestralStates
from af.tree.build import finish_tree
from af.tree.io import i6, newick
from af.tree.model import Tree
from af.tree.populate import (
    CladeCall,
    CladeInput,
    LeafRecord,
    PopulateError,
    leaf_key,
    nucleotide_changes,
    populate,
)

KEYS = {name: leaf_key(f"EPI_ISL_90000{i}", f"EPI90000{i}") for i, name in enumerate("oabcd")}
LEAF_SEQ = {
    "o": "ATGAAACCC",
    "a": "ATGGAACCC",  # AAA->GAA on y->a: K2E
    "b": "ATGGAACCN",  # an ambiguity: no change counted, no aa change
    "c": "ATGAAACCG",  # synonymous CCC->CCG on z->c
    "d": "ATGAAA---",  # a deletion: gaps never count as nucleotide changes
}
INTERNAL_SEQ = {"root": "ATGAAACCT", "x": "ATGAAACCC", "y": "ATGGAACCC", "z": "ATGAAACCC"}


def records() -> dict[str, LeafRecord]:
    out = {}
    for index, name in enumerate("oabcd"):
        precision = "year" if name == "b" else "day"
        date = datetime.date(2020 + index, 1, 1 if name == "b" else 15)
        out[KEYS[name]] = LeafRecord(
            epi_isl=f"EPI_ISL_90000{index}",
            accession=f"EPI90000{index}",
            name=f"A/EXAMPLETOWN/{index}/2020",
            nucleotides=LEAF_SEQ[name],
            collection_date=date,
            date_precision=precision,  # type: ignore[arg-type]
            collection_date_first=date if name != "b" else datetime.date(2021, 1, 1),
            collection_date_last=date if name != "b" else datetime.date(2021, 12, 31),
            country="EXAMPLELAND",
            region="Europe",
        )
    return out


def built() -> tuple[Tree, dict[str, int]]:
    """The finished tree, and each internal node's id by its letter."""
    k = KEYS
    text = (
        f"({k['o']}:0.1,(({k['a']}:0.2,{k['b']}:0.3):0.05,({k['c']}:0.1,{k['d']}:0.1):0.01):0.02);"
    )
    tree = finish_tree(newick.loads(text), outgroup=k["o"]).tree
    ids = {}
    for node in tree.internal():
        leaves = {leaf.name for leaf in _leaves_under(node)}
        if node.parent is None:
            ids["root"] = node.node_id
        elif leaves == {k["a"], k["b"]}:
            ids["y"] = node.node_id
        elif leaves == {k["c"], k["d"]}:
            ids["z"] = node.node_id
        else:
            ids["x"] = node.node_id
    return tree, ids


def _leaves_under(node):
    stack = [node]
    while stack:
        item = stack.pop()
        if item.is_leaf:
            yield item
        stack.extend(item.children)


def states_for(ids: Mapping[str, int]) -> AncestralStates:
    return AncestralStates(
        nucleotides={ids[name]: INTERNAL_SEQ[name] for name in ids},
        backend="stub",
        backend_version="0",
        seconds=0.0,
    )


class GapBlind:
    name = "gapblind"
    reconstructs_gaps = False
    optimises_branch_lengths = False

    def version(self) -> str:
        return "0"

    def reconstruct(self, tree, alignment, work_dir, threads=1):
        raise NotImplementedError


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

    def engine(nodes: Sequence[CladeInput]) -> tuple[str, Mapping[str, CladeCall]]:
        seen.extend(nodes)
        return "nomenclature@abc1234", {n.node_id: CladeCall("J.2", support=2) for n in nodes}

    result = populate(
        tree, "h3", records(), states_for(ids), assign_clades=engine, backend=GapBlind()
    )
    assert len(seen) == 9
    assert all(n.gaps_reconstructed for n in seen if n.is_leaf)
    assert not any(n.gaps_reconstructed for n in seen if not n.is_leaf)
    assert result.clade_set_version == "nomenclature@abc1234"
    assert result.counts["leaves_without_clade"] == 0


def test_a_node_the_clade_engine_skipped_is_an_error() -> None:
    tree, ids = built()

    def engine(nodes: Sequence[CladeInput]) -> tuple[str, Mapping[str, CladeCall]]:
        return "v", {n.node_id: CladeCall("J") for n in nodes[1:]}

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
    meta = i6.read_metadata(tmp_path)
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
