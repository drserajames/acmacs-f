"""The tree model: stable ids, and the shape operations the build step depends on."""

from __future__ import annotations

import pytest

from af.tree.io import newick
from af.tree.model import Tree, TreeError, leaf_id

from .tree_fixtures import built


def tree_of(text: str) -> Tree:
    tree = newick.loads(text)
    tree.assign_ids()
    return tree


def test_ids_are_the_same_whatever_the_order_of_the_children() -> None:
    """The whole point: redrawing the tree must not renumber it."""
    one = tree_of("((a:1,b:1):1,(c:1,d:1):1);")
    two = tree_of("((d:1,c:1):1,(b:1,a:1):1);")
    assert {node.node_id for node in one.internal()} == {node.node_id for node in two.internal()}
    assert one.root.node_id == two.root.node_id


def test_ladderizing_does_not_change_ids() -> None:
    tree = tree_of("((a:1,(b:1,c:1):1):1,(d:1,(e:1,f:1):1):1);")
    before = sorted(node.node_id for node in tree.internal())
    tree.ladderize()
    after = sorted(node.node_id for node in tree.internal())
    assert before == after


def test_a_leaf_id_depends_only_on_its_key() -> None:
    assert tree_of("(a:1,b:1);").by_id(leaf_id("a")).name == "a"


def test_unary_node_is_rejected_not_tolerated() -> None:
    """CMAPLE segfaults on these, and ae's Tree::remove leaves them behind (INVENTORY §5)."""
    with pytest.raises(TreeError, match="unary node"):
        tree_of("((a:1):1,b:1);")


def test_remove_unary_keeps_the_total_branch_length() -> None:
    tree = newick.loads("((a:1):2,b:1);")
    assert tree.remove_unary() == 1
    tree.assign_ids()
    leaf = next(node for node in tree.leaves() if node.name == "a")
    assert leaf.branch_length == pytest.approx(3.0)


def test_duplicate_leaf_names_are_rejected() -> None:
    with pytest.raises(TreeError, match="duplicate leaf"):
        tree_of("((a:1,a:1):1,b:1);")


def test_collapse_only_touches_internal_branches() -> None:
    tree = tree_of("((a:0,b:0):0,(c:1,d:1):1);")
    assert tree.collapse_short_branches(0.0) == 1
    tree.assign_ids()
    names = sorted(leaf.name or "" for leaf in tree.leaves())
    assert names == ["a", "b", "c", "d"]  # no leaf was collapsed away
    assert len(tree.root.children) == 3  # a, b and the (c,d) clade


def test_collapse_respects_the_tolerance() -> None:
    tree = tree_of("((a:1,b:1):0.00001,(c:1,d:1):1);")
    assert tree.collapse_short_branches(0.0) == 0  # nothing is exactly zero
    assert tree.collapse_short_branches(5e-5) == 1


def test_reroot_puts_the_outgroup_below_the_root() -> None:
    tree = tree_of("(((a:1,b:1):1,c:1):1,outgroup:1);")
    tree.reroot_on_outgroup("outgroup")
    tree.assign_ids()
    assert "outgroup" in {child.name for child in tree.root.children}
    assert sorted(leaf.name or "" for leaf in tree.leaves()) == ["a", "b", "c", "outgroup"]


def test_reroot_keeps_the_clades_that_do_not_contain_the_outgroup() -> None:
    tree = tree_of("(((a:1,b:1):1,c:1):1,outgroup:1);")
    ab = next(
        node.node_id
        for node in tree.internal()
        if {leaf.name for leaf in tree_leaves(node)} == {"a", "b"}
    )
    tree.reroot_on_outgroup("outgroup")
    tree.assign_ids()
    assert ab in {node.node_id for node in tree.internal()}


def test_reroot_on_a_missing_outgroup_is_an_error() -> None:
    tree = tree_of("((a:1,b:1):1,c:1);")
    with pytest.raises(TreeError, match="not a leaf"):
        tree.reroot_on_outgroup("nope")


def test_prune_removes_leaves_and_the_nodes_left_unary() -> None:
    tree = tree_of("((a:1,b:1):1,(c:1,d:1):1);")
    assert tree.prune(["a", "c", "d"]) == 1
    tree.assign_ids()
    assert sorted(leaf.name or "" for leaf in tree.leaves()) == ["a", "c", "d"]
    assert all(len(node.children) != 1 for node in tree.internal())


def test_prune_rejects_names_that_are_not_there() -> None:
    tree = tree_of("((a:1,b:1):1,c:1);")
    with pytest.raises(TreeError, match="not in the tree"):
        tree.prune(["a", "zzz"])


def test_cumulative_lengths_are_root_to_node() -> None:
    tree = tree_of("((a:1,b:2):3,c:4);")
    lengths = tree.cumulative_lengths()
    leaf_a = next(leaf for leaf in tree.leaves() if leaf.name == "a")
    assert lengths[id(leaf_a)] == pytest.approx(4.0)


def tree_leaves(node):  # noqa: ANN001, ANN201 - test helper
    stack = [node]
    while stack:
        current = stack.pop()
        if current.is_leaf:
            yield current
        stack.extend(current.children)


def test_prune_counts_leaves_not_emptied_internal_nodes() -> None:
    """Regression: dropping a whole clade counted its internal nodes as leaves too.

    On a real H1 tree that overstated 13,930 removed leaves as 15,933 — the kind of quiet
    miscount that would go straight into a provenance count and be believed.
    """
    tree = tree_of("(((a:1,b:1):1,(c:1,d:1):1):1,(e:1,f:1):1);")
    assert tree.prune(["e", "f"]) == 4


def test_prune_to_a_single_clade_leaves_a_usable_tree() -> None:
    tree = tree_of("(((a:1,b:1):1,(c:1,d:1):1):1,(e:1,f:1):1);")
    tree.prune(["a", "b"])
    tree.assign_ids()
    assert sorted(leaf.name or "" for leaf in tree.leaves()) == ["a", "b"]
    assert all(len(node.children) != 1 for node in tree.internal())


def _clades(tree) -> set[frozenset[str]]:
    """Every internal node's leaf set."""
    out = set()
    for node in tree.internal():
        stack, leaves = [node], set()
        while stack:
            current = stack.pop()
            if current.is_leaf:
                leaves.add(current.name)
            else:
                stack.extend(current.children)
        out.add(frozenset(leaves))
    return out


def test_binary_copy_resolves_multifurcations_without_changing_the_leaf_sets() -> None:
    """raxml-ng refuses a multifurcating tree, and af's finished trees usually are one."""
    tree, _ = built()
    tree.collapse_short_branches(1.0)  # force a multifurcation
    assert max(len(node.children) for node in tree.preorder()) > 2
    binary, added = tree.binary_copy()
    assert added > 0
    assert max(len(node.children) for node in binary.preorder()) == 2
    assert {leaf.name for leaf in binary.leaves()} == {leaf.name for leaf in tree.leaves()}
    # Every original clade survives, so match_states_by_clade maps the states back.
    assert _clades(tree) <= _clades(binary)
    # The nodes it added carry no distance.
    original = _clades(tree)
    added_nodes = [
        n
        for n in binary.internal()
        if frozenset(leaf.name for leaf in binary.leaves() if _under(n, leaf)) not in original
    ]
    assert all(node.branch_length == 0.0 for node in added_nodes)


def _under(node, leaf) -> bool:
    current = leaf
    while current is not None:
        if current is node:
            return True
        current = current.parent
    return False


def test_binary_copy_leaves_an_already_binary_tree_alone() -> None:
    tree, _ = built()
    assert max(len(node.children) for node in tree.preorder()) == 2
    binary, added = tree.binary_copy()
    assert added == 0
    assert newick.dumps(binary) == newick.dumps(tree)
