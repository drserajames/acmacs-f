"""The tree engine: descent, reversions, deletions, and what it refuses to guess.

Sequences are built to the synthetic nomenclature in ``synthetic.py``: position 5 K makes
a virus P, 9 T and 331 W make it P.1, 12 N and nucleotide 6 A make it P.1.1.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from af.clades.assign import AssignmentError, Node, assign_sequences, assign_tree
from af.clades.nomenclature import CladeSet
from af.clades.sequence import AlignedSequence, GapSupport

from .synthetic import build_clone, load_synthetic

SUBTYPE = "A(H3N2)"


def _synthetic(tmp_path: Path) -> CladeSet:
    return load_synthetic(build_clone(tmp_path).parent)


def protein(**states: str) -> str:
    """An amino-acid string of 400 alanines with the given 1-based positions set."""
    residues = ["A"] * 400
    for position, state in states.items():
        residues[int(position[1:]) - 1] = state
    return "".join(residues)


def sequence(gaps: GapSupport = GapSupport.OBSERVED, **states: str) -> AlignedSequence:
    return AlignedSequence(amino_acids=protein(**states), nucleotides="A" * 1200, gaps=gaps)


def test_deepest_matching_clade_wins(tmp_path: Path) -> None:
    clade_set = _synthetic(tmp_path)
    nodes = [
        Node("root", None, sequence(p5="K")),
        Node("leaf", "root", sequence(p5="K", p9="T", p331="W")),
    ]
    result = assign_tree(nodes, clade_set)
    assert result.clade("root") == "P"
    assert result.clade("leaf") == "P.1"


def test_a_tip_reversion_does_not_drop_the_clade(tmp_path: Path) -> None:
    """The reason for assigning on a tree at all: a virus that loses a defining residue
    keeps the clade its ancestors established, where signature matching moved it to the
    parent clade."""
    clade_set = _synthetic(tmp_path)
    nodes = [
        Node("root", None, sequence(p5="K")),
        Node("founder", "root", sequence(p5="K", p9="T", p331="W")),
        Node("reverted", "founder", sequence(p5="K", p9="A", p331="W")),
    ]
    result = assign_tree(nodes, clade_set)
    assert result.clade("founder") == "P.1"
    assert result.clade("reverted") == "P.1"
    assert result.assignments["reverted"].inherited


def test_a_clade_is_never_assigned_outside_its_parent(tmp_path: Path) -> None:
    """A virus carrying P.1.1's own mutations but not P.1's must not become P.1.1."""
    clade_set = _synthetic(tmp_path)
    nodes = [
        Node("root", None, sequence(p5="K")),
        Node("leaf", "root", sequence(p5="K", p12="N")),
    ]
    assert assign_tree(nodes, clade_set).clade("leaf") == "P"


def test_unobservable_positions_do_not_rule_a_clade_out(tmp_path: Path) -> None:
    clade_set = _synthetic(tmp_path)
    nodes = [
        Node("root", None, sequence(p5="K")),
        Node("leaf", "root", sequence(p5="K", p9="T", p331="X")),
    ]
    result = assign_tree(nodes, clade_set)
    assert result.clade("leaf") == "P.1"
    assert result.assignments["leaf"].unobservable == 1
    assert result.assignments["leaf"].support == 2


def test_revoked_clades_are_not_assigned(tmp_path: Path) -> None:
    clade_set = _synthetic(tmp_path)
    nodes = [Node("root", None, sequence(p5="K"))]
    assert assign_tree(nodes, clade_set).clade("root") == "P"


def test_gap_blind_reconstruction_is_refused_when_clades_need_deletions(tmp_path: Path) -> None:
    """B/Vic defines six clades by deletions; raxml-ng and IQ-TREE cannot represent one,
    so af says so rather than quietly never assigning those clades."""
    clade_set = _synthetic(tmp_path)
    nodes = [Node("root", None, sequence(GapSupport.GAP_BLIND, p5="K"))]
    with pytest.raises(AssignmentError, match="cannot represent a gap"):
        assign_tree(nodes, clade_set)


def test_gap_blind_may_be_accepted_knowingly(tmp_path: Path) -> None:
    clade_set = _synthetic(tmp_path)
    nodes = [Node("root", None, sequence(GapSupport.GAP_BLIND, p5="K"))]
    assert assign_tree(nodes, clade_set, require_gap_support=False).clade("root") == "P"


def test_deletion_defined_clade_is_assigned_from_an_observed_gap(tmp_path: Path) -> None:
    clade_set = _synthetic(tmp_path)
    nodes = [
        Node("root", None, sequence(p5="K")),
        Node("leaf", "root", sequence(p5="K", p7="-")),
    ]
    assert assign_tree(nodes, clade_set).clade("leaf") == "P.2"


def test_result_carries_the_clade_set_version(tmp_path: Path) -> None:
    clade_set = _synthetic(tmp_path)
    result = assign_tree([Node("root", None, sequence(p5="K"))], clade_set)
    assert result.clade_set_version == clade_set.version
    assert result.subtype == SUBTYPE


def test_convergent_clades_are_reported(tmp_path: Path) -> None:
    """One clade established twice is either real convergence or a loose definition; a
    reviewer should see it either way."""
    clade_set = _synthetic(tmp_path)
    nodes = [
        Node("root", None, sequence(p5="K")),
        Node("branch-a", "root", sequence(p5="K", p9="T", p331="W")),
        Node("branch-b", "root", sequence(p5="K", p9="T", p331="W")),
    ]
    result = assign_tree(nodes, clade_set)
    assert set(result.convergent()["P.1"]) == {"branch-a", "branch-b"}


def test_assignment_is_deterministic(tmp_path: Path) -> None:
    clade_set = _synthetic(tmp_path)
    nodes = [
        Node("root", None, sequence(p5="K")),
        Node("leaf", "root", sequence(p5="K", p9="T", p331="W", p12="N")),
    ]
    first = assign_tree(nodes, clade_set).counts()
    second = assign_tree(list(reversed(nodes)), clade_set).counts()
    assert first == second


def test_empty_tree_is_fatal(tmp_path: Path) -> None:
    clade_set = _synthetic(tmp_path)
    with pytest.raises(AssignmentError, match="no nodes"):
        assign_tree([], clade_set)


def test_unreachable_node_is_fatal(tmp_path: Path) -> None:
    """A node whose parent is missing from the tree would otherwise never be assigned,
    and its absence from the output would look like an unassigned virus."""
    clade_set = _synthetic(tmp_path)
    nodes = [
        Node("root", None, sequence(p5="K")),
        Node("orphan-a", "orphan-b", sequence(p5="K")),
        Node("orphan-b", "orphan-a", sequence(p5="K")),
    ]
    with pytest.raises(AssignmentError, match="not reachable"):
        assign_tree(nodes, clade_set)


def test_sequences_off_the_tree_use_the_chosen_fallback(tmp_path: Path) -> None:
    """Map antigens need clades before the tree exists; Sarah chose Nextclade for that."""
    clade_set = _synthetic(tmp_path)
    sequences = {"virus-1": sequence(p5="K"), "virus-2": sequence(p5="A")}
    result = assign_sequences(
        sequences, clade_set, fallback=lambda seqs, _: {"virus-1": "P", "virus-2": None}
    )
    assert result["virus-1"].clade == "P"
    assert result["virus-2"].clade is None


def test_fallback_returning_an_unknown_sequence_is_fatal(tmp_path: Path) -> None:
    clade_set = _synthetic(tmp_path)
    with pytest.raises(AssignmentError, match="unknown sequences"):
        assign_sequences(
            {"virus-1": sequence(p5="K")}, clade_set, fallback=lambda seqs, _: {"other": "P"}
        )
