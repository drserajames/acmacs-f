"""The engine's known limitation, made visible: sibling conflicts and name-decided ties."""

from __future__ import annotations

import subprocess
from pathlib import Path

from af.clades.assign import Node, assign_tree, sibling_conflicts
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
