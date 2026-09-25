"""Newick parsing and writing, including the shapes real tools emit."""

from __future__ import annotations

import pytest

from af.tree.io import newick
from af.tree.model import TreeError


def test_round_trip_keeps_leaves_and_lengths() -> None:
    text = "((a:0.1,b:0.2):0.3,c:0.4);"
    tree = newick.loads(text)
    again = newick.loads(newick.dumps(tree))
    assert sorted(leaf.name or "" for leaf in again.leaves()) == ["a", "b", "c"]
    lengths = {leaf.name: leaf.branch_length for leaf in again.leaves()}
    assert lengths == pytest.approx({"a": 0.1, "b": 0.2, "c": 0.4})


def test_internal_labels_can_be_written_as_stable_ids() -> None:
    tree = newick.loads("((a:1,b:1):1,c:1);")
    tree.assign_ids()
    written = newick.dumps(tree, with_internal_labels=True)
    parsed = newick.loads(written)
    labels = {node.name for node in parsed.internal() if node.name}
    assert labels == {node.id_hex for node in tree.internal()}


def test_writing_ids_before_assigning_them_is_an_error() -> None:
    tree = newick.loads("((a:1,b:1):1,c:1);")
    with pytest.raises(TreeError, match="assign_ids"):
        newick.dumps(tree, with_internal_labels=True)


def test_comments_are_skipped() -> None:
    """TreeTime writes NEXUS-flavoured Newick with [&mutations=...] comments."""
    tree = newick.loads('((a:1[&mutations="A1T"],b:1):1,c:1);')
    assert sorted(leaf.name or "" for leaf in tree.leaves()) == ["a", "b", "c"]


def test_quoted_labels_survive_a_round_trip() -> None:
    # Quoting matters because real leaf labels contain spaces, slashes and commas.
    tree = newick.loads("('label with spaces/and/slashes':1,'comma,inside':1);")
    names = sorted(leaf.name or "" for leaf in newick.loads(newick.dumps(tree)).leaves())
    assert names == ["comma,inside", "label with spaces/and/slashes"]


def test_polytomies_are_kept() -> None:
    tree = newick.loads("(a:1,b:1,c:1,d:1);")
    assert len(tree.root.children) == 4


def test_a_leaf_without_a_label_is_an_error() -> None:
    """Addressing a leaf by position is exactly what af does not do (design rule 2)."""
    with pytest.raises(TreeError, match="no label"):
        newick.loads("((a:1,:1):1,c:1);")


def test_unbalanced_parentheses_are_an_error() -> None:
    with pytest.raises(TreeError, match="unbalanced"):
        newick.loads("((a:1,b:1):1,c:1;")


def test_bad_branch_length_is_an_error() -> None:
    with pytest.raises(TreeError, match="bad branch length"):
        newick.loads("(a:zzz,b:1);")


def test_empty_input_is_an_error() -> None:
    with pytest.raises(TreeError, match="empty"):
        newick.loads("   ")


def test_a_deep_tree_does_not_overflow_the_stack() -> None:
    """Real trees are >100,000 leaves; a recursive parser dies on this."""
    depth = 5000
    text = "(" * depth + "a:1" + ",b:1):1" * depth
    tree = newick.loads(text + ";")
    assert sum(1 for _ in tree.leaves()) == depth + 1
    assert newick.dumps(tree).count("(") == depth
