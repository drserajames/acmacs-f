"""The engine's known limitation, made visible: sibling conflicts and name-decided ties."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, cast

from af.clades.assign import Node, assign_tree, conflict_review, sibling_conflicts
from af.clades.nomenclature import CladeSet

from .synthetic import build_clone, commit_command, load_synthetic, write_clade
from .test_assign import sequence


def _synthetic(tmp_path: Path) -> CladeSet:
    return load_synthetic(build_clone(tmp_path).parent)


def test_a_node_matching_a_sibling_of_its_inherited_clade_is_reported(tmp_path: Path) -> None:
    """Below a node labelled P.2, a node gains P.1's own markers and matches P.1's whole
    signature with more support than P.2. The engine keeps P.2 (candidates are the inherited
    clade's descendants), and the conflict is reported, not resolved."""
    clade_set = _synthetic(tmp_path)
    nodes = [
        Node("root", None, sequence(p5="K")),
        Node("p2", "root", sequence(p5="K", p7="-")),
        Node("gains-p1", "p2", sequence(p5="K", p7="-", p9="T", p331="W")),
        Node("leaf-a", "gains-p1", sequence(p5="K", p7="-", p9="T", p331="W")),
        Node("leaf-b", "gains-p1", sequence(p5="K", p7="-", p9="T", p331="W")),
    ]
    result = assign_tree(nodes, clade_set)
    assert result.clade("gains-p1") == "P.2"  # labelling unchanged
    conflicts = sibling_conflicts(nodes, result, clade_set)
    [top] = [c for c in conflicts if c.topmost]
    assert (top.node, top.label, top.other) == ("gains-p1", "P.2", "P.1")
    assert top.leaves == 2 and top.gained_on_edge
    assert top.other_support >= top.label_support
    # the leaves below repeat the conflict, but are not topmost
    assert {c.node for c in conflicts if not c.topmost} == {"leaf-a", "leaf-b"}


def test_no_conflict_within_the_labels_own_lineage(tmp_path: Path) -> None:
    """Matching an ancestor or a descendant of the label is the normal case, not a conflict."""
    clade_set = _synthetic(tmp_path)
    nodes = [
        Node("root", None, sequence(p5="K")),
        Node("p1", "root", sequence(p5="K", p9="T", p331="W")),
        Node("leaf", "p1", sequence(p5="K", p9="T", p331="W")),
    ]
    result = assign_tree(nodes, clade_set)
    assert sibling_conflicts(nodes, result, clade_set) == ()


def test_a_tie_decided_by_the_name_is_recorded(tmp_path: Path) -> None:
    """Two sibling clades with the same depth and the same support at a node: the name
    decides, deterministically, and the tie is recorded so it is not silent."""
    clone = build_clone(tmp_path)
    write_clade(
        clone / "subclades",
        "P.4",
        """
name: P.4
parent: P
defining_mutations:
- locus: HA1
  position: 9
  state: T
- locus: HA2
  position: 2
  state: W
""",
    )
    subprocess.run(["git", "-C", str(clone), "add", "-A"], check=True)
    subprocess.run(commit_command(clone, "a sibling with P.1's signature"), check=True)
    clade_set = load_synthetic(clone.parent)
    nodes = [
        Node("root", None, sequence(p5="K")),
        Node("leaf", "root", sequence(p5="K", p9="T", p331="W")),
    ]
    result = assign_tree(nodes, clade_set)
    assert result.clade("leaf") == "P.4"  # unchanged rule: the later name
    [tie] = result.ties
    assert (tie.node, tie.chosen, tie.others) == ("leaf", "P.4", ("P.1",))


def test_no_tie_when_support_differs(tmp_path: Path) -> None:
    clade_set = _synthetic(tmp_path)
    nodes = [
        Node("root", None, sequence(p5="K")),
        Node("leaf", "root", sequence(p5="K", p9="T", p331="W")),
    ]
    assert assign_tree(nodes, clade_set).ties == ()


# ------------------------------------- sibling recovery (Sarah, 11 Oct 2026, Q138)


def recovery_tree() -> list[Node]:
    """P → P.1, then an internal node whose own sequence reverts P.1's own 9T and matches
    sister P.2 (its deletion at 7), with a leaf below it."""
    reverted = sequence(p5="K", p7="-", p9="A")
    return [
        Node("root", None, sequence(p5="K")),
        Node("in-p1", "root", sequence(p5="K", p9="T", p331="W")),
        Node("reverts", "in-p1", reverted),
        Node("leaf", "reverts", reverted),
    ]


def test_an_internal_node_contradicting_its_clade_is_recovered_to_a_sibling(tmp_path: Path) -> None:
    """The node reverts P.1's own 9T and matches P.2 instead. Without recovery it would keep
    P.1 and take its leaves with it, and P.2 would be assigned nowhere."""
    clade_set = _synthetic(tmp_path)
    nodes = recovery_tree()
    result = assign_tree(nodes, clade_set)
    assert result.clade("in-p1") == "P.1"
    assert result.clade("reverts") == "P.2"
    assert result.clade("leaf") == "P.2"
    assert [(r.node, r.inherited, r.recovered) for r in result.recoveries] == [
        ("reverts", "P.1", "P.2")
    ]
    assert not result.assignments["reverts"].inherited


def test_a_leaf_is_never_recovered(tmp_path: Path) -> None:
    """A tip cannot fall out of its clade: the same sequence as a leaf keeps P.1."""
    clade_set = _synthetic(tmp_path)
    nodes = [node for node in recovery_tree() if node.name != "leaf"]
    nodes[-1] = Node("reverts", "in-p1", nodes[-1].sequence)  # now a leaf: no children
    result = assign_tree(nodes, clade_set)
    assert result.clade("reverts") == "P.1"
    assert result.assignments["reverts"].inherited
    assert result.recoveries == ()


def test_recovery_never_climbs_above_the_parent_clade(tmp_path: Path) -> None:
    """Contradicting P as well as P.1 (P's own 5K reverted) keeps P.1: recovering into P's
    subtree on a sequence that contradicts P itself would be a guess."""
    clade_set = _synthetic(tmp_path)
    nodes = [
        Node("root", None, sequence(p5="K")),
        Node("in-p1", "root", sequence(p5="K", p9="T", p331="W")),
        Node("reverts", "in-p1", sequence(p5="E", p7="-", p9="A")),
        Node("leaf", "reverts", sequence(p5="E", p7="-", p9="A")),
    ]
    result = assign_tree(nodes, clade_set)
    assert result.clade("reverts") == "P.1"
    assert result.recoveries == ()


def test_a_conflict_counts_only_the_leaves_still_in_its_label(tmp_path: Path) -> None:
    """Below a node labelled P.2 that also matches P.1, one subtree is recovered to P.1. Those
    leaves are no longer mislabelled, so the guard must not count them (Q138)."""
    clade_set = _synthetic(tmp_path)
    nodes = [
        Node("root", None, sequence(p5="K")),
        Node("p2", "root", sequence(p5="K", p7="-")),
        Node("gains-p1", "p2", sequence(p5="K", p7="-", p9="T", p331="W")),
        Node("stays", "gains-p1", sequence(p5="K", p7="-", p9="T", p331="W")),
        Node("moved", "gains-p1", sequence(p5="K", p7="A", p9="T", p331="W")),
        Node("moved-leaf", "moved", sequence(p5="K", p7="A", p9="T", p331="W")),
    ]
    result = assign_tree(nodes, clade_set)
    assert result.clade("gains-p1") == "P.2"
    assert result.clade("moved") == "P.1" and result.clade("moved-leaf") == "P.1"
    conflict = next(c for c in sibling_conflicts(nodes, result, clade_set) if c.node == "gains-p1")
    assert (conflict.leaves, conflict.leaves_in_label) == (2, 1)
    review = conflict_review(
        sibling_conflicts(nodes, result, clade_set), result.ties, result.recoveries
    )
    topmost = cast("list[dict[str, Any]]", review["topmost"])
    assert topmost[0]["leaves_in_label"] == 1
    assert topmost[0]["leaves"] == 2  # kept: 05-trees reads it
    assert cast("dict[str, int]", review["counts"])["recovered_nodes"] == 1
    assert review["recovered"] == [{"node": "moved", "inherited": "P.2", "recovered": "P.1"}]
