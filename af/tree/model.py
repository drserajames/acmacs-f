"""The tree model: nodes, and node ids that stay the same when the tree is redrawn.

Why stable ids. In the old pipeline a node was addressed by its position (`node_id`), so
re-ladderizing or rebuilding renumbered everything and every hand-placed label silently moved or
stopped applying (INVENTORY §2.2 C2, "per-node entries keyed by node_id silently stop applying").
Here a node's id is derived from *which leaves are under it*, not from where it sits in the file:

    leaf id      = 64-bit BLAKE2b of the leaf's key (its name, or EPI_ISL+accession)
    internal id  = XOR of the ids of its children

XOR is commutative and associative, so the id survives re-ladderizing, re-rooting anywhere outside
the clade, and a rebuild that keeps the same leaves. It is also cheap: one pass, no sorting.

Two consequences worth knowing. A node with exactly one child has the same id as that child: unary
nodes are therefore *detected* rather than silently tolerated — which matters, because ae's
`Tree::remove` leaves them behind and CMAPLE segfaults on them (INVENTORY §5). And two distinct
nodes cannot share an id unless they have the same leaf set, so a collision is a real structural
problem and is raised, never worked around.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any

ROOT_NAME = "__root__"


class TreeError(ValueError):
    """The tree is structurally wrong (unary node, duplicate leaf, id collision)."""


def leaf_id(key: str) -> int:
    """A leaf's 64-bit id, from its key. Stable across runs and machines."""
    return int.from_bytes(hashlib.blake2b(key.encode(), digest_size=8).digest(), "big")


@dataclass(eq=False)
class Node:
    """One node. ``name`` is the leaf label, or None for an internal node."""

    name: str | None = None
    branch_length: float = 0.0
    children: list[Node] = field(default_factory=list)
    parent: Node | None = None
    node_id: int = 0
    annotations: dict[str, Any] = field(default_factory=dict)

    @property
    def is_leaf(self) -> bool:
        return not self.children

    @property
    def id_hex(self) -> str:
        return f"{self.node_id:016x}"

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        what = self.name if self.is_leaf else f"<{len(self.children)} children>"
        return f"Node({what}, l={self.branch_length:g}, id={self.id_hex})"


class Tree:
    """A rooted tree. Traversals are iterative: real trees here have >100,000 leaves."""

    def __init__(self, root: Node) -> None:
        self.root = root
        self._by_id: dict[int, Node] = {}

    # ---- traversal ----------------------------------------------------------------

    def preorder(self) -> Iterator[Node]:
        stack = [self.root]
        while stack:
            node = stack.pop()
            yield node
            stack.extend(reversed(node.children))

    def postorder(self) -> Iterator[Node]:
        out: list[Node] = []
        stack = [(self.root, False)]
        while stack:
            node, expanded = stack.pop()
            if expanded or node.is_leaf:
                out.append(node)
            else:
                stack.append((node, True))
                stack.extend((child, False) for child in node.children)
        return iter(out)

    def leaves(self) -> Iterator[Node]:
        return (node for node in self.preorder() if node.is_leaf)

    def internal(self) -> Iterator[Node]:
        return (node for node in self.preorder() if not node.is_leaf)

    def count(self) -> tuple[int, int]:
        """(leaves, internal nodes)."""
        leaves = internal = 0
        for node in self.preorder():
            if node.is_leaf:
                leaves += 1
            else:
                internal += 1
        return leaves, internal

    # ---- identity -----------------------------------------------------------------

    def assign_ids(self, key: Callable[[Node], str] | None = None) -> None:
        """Compute every node's id bottom-up, and verify the tree's structure.

        Raises TreeError on a duplicate leaf key, a unary node, or an id collision.
        """
        key = key or (lambda node: node.name or "")
        seen_leaves: dict[str, int] = {}
        by_id: dict[int, Node] = {}
        for node in self.postorder():
            if node.is_leaf:
                leaf_key = key(node)
                if not leaf_key:
                    raise TreeError("a leaf has no name; every leaf needs a key for its id")
                if leaf_key in seen_leaves:
                    raise TreeError(f"duplicate leaf key {leaf_key!r}")
                seen_leaves[leaf_key] = 1
                node.node_id = leaf_id(leaf_key)
            else:
                if len(node.children) == 1:
                    child = node.children[0]
                    what = child.name or "an internal node"
                    raise TreeError(
                        f"unary node above {what}: one child means it has the same leaf set as its "
                        "child, so it has no identity of its own. CMAPLE also segfaults on these; "
                        "collapse it (Tree.remove_unary) before continuing."
                    )
                value = 0
                for child in node.children:
                    value ^= child.node_id
                node.node_id = value
            if node.node_id in by_id and by_id[node.node_id] is not node:
                raise TreeError(
                    f"two nodes share id {node.id_hex}: they have the same leaf set, which a tree "
                    "should not allow"
                )
            by_id[node.node_id] = node
        self._by_id = by_id

    def by_id(self, node_id: int | str) -> Node:
        if isinstance(node_id, str):
            node_id = int(node_id, 16)
        if not self._by_id:
            raise TreeError("ids have not been assigned; call assign_ids() first")
        return self._by_id[node_id]

    def ids_assigned(self) -> bool:
        return bool(self._by_id)

    # ---- shape --------------------------------------------------------------------

    def remove_unary(self) -> int:
        """Splice out nodes with one child, adding their branch length to the child.

        Returns how many were removed, for the step's counts. ae's `Tree::remove` leaves these
        behind and CMAPLE then segfaults reading the tree (INVENTORY §5).
        """
        removed = 0
        for node in list(self.postorder()):
            while len(node.children) == 1:
                child = node.children[0]
                child.branch_length += node.branch_length
                child.parent = node.parent
                if node.parent is None:
                    self.root = child
                    child.branch_length = 0.0
                else:
                    index = node.parent.children.index(node)
                    node.parent.children[index] = child
                removed += 1
                node = child
        if removed:
            self._by_id = {}
        return removed

    def ladderize(self, smallest_first: bool = True) -> None:
        """Order children by how many leaves they carry. Ids do not change."""
        sizes: dict[int, int] = {}
        for node in self.postorder():
            sizes[id(node)] = (
                1 if node.is_leaf else sum(sizes[id(child)] for child in node.children)
            )
        for node in self.preorder():
            if node.children:
                node.children.sort(key=lambda child: sizes[id(child)], reverse=not smallest_first)

    def collapse_short_branches(self, tolerance: float = 0.0) -> int:
        """Merge an internal branch into its parent when its length is <= tolerance.

        Returns the number collapsed. With ``tolerance=0.0`` only exactly-zero branches go, which
        measurement shows is all the delivered trees' `di2multi(tol=5e-5)` ever did
        (notes/trees/COMPARISON.md §3). Leaf branches are never collapsed: a leaf is a datum.
        """
        collapsed = 0
        for node in list(self.postorder()):
            if node.is_leaf or node.parent is None:
                continue
            if node.branch_length <= tolerance:
                parent = node.parent
                index = parent.children.index(node)
                for child in node.children:
                    child.parent = parent
                parent.children[index : index + 1] = node.children
                collapsed += 1
        if collapsed:
            self._by_id = {}
        return collapsed

    def reroot_on_outgroup(self, outgroup: str) -> None:
        """Root the tree on the branch above the named leaf.

        The outgroup ends up as a child of the root, as `rotate.R`'s `ape::root(t, outgroup)` does
        (it makes the root a multifurcation at the outgroup's parent, and af matches that so
        Robinson-Foulds comparisons with the old trees are not off by one split).
        """
        target = next((leaf for leaf in self.leaves() if leaf.name == outgroup), None)
        if target is None:
            raise TreeError(f"outgroup {outgroup!r} is not a leaf of this tree")
        if target.parent is self.root:
            return
        path: list[Node] = []
        node: Node | None = target.parent
        while node is not None:
            path.append(node)
            node = node.parent
        # Reverse the parent chain between the old root and the outgroup's parent.
        new_root = path[0]
        for child, parent in zip(path, path[1:], strict=False):
            parent.children.remove(child)
            child.children.append(parent)
            parent.parent = child
            parent.branch_length = child.branch_length
        new_root.parent = None
        new_root.branch_length = 0.0
        self.root = new_root
        self.remove_unary()
        self._by_id = {}

    def prune(self, keep: Iterable[str]) -> int:
        """Keep only the named leaves. Returns the number of leaves removed."""
        wanted = set(keep)
        missing = wanted - {leaf.name for leaf in self.leaves() if leaf.name}
        if missing:
            raise TreeError(
                f"{len(missing)} leaves to keep are not in the tree, e.g. {sorted(missing)[:3]}"
            )
        removed = 0
        for node in list(self.postorder()):
            if node.is_leaf and node.name not in wanted and node.parent is not None:
                node.parent.children.remove(node)
                removed += 1
            elif not node.is_leaf and not node.children and node.parent is not None:
                node.parent.children.remove(node)
        self.remove_unary()
        self._by_id = {}
        return removed

    def cumulative_lengths(self) -> dict[int, float]:
        """Root-to-node distance for every node, keyed by ``id(node)``."""
        out = {id(self.root): 0.0}
        for node in self.preorder():
            if node.parent is not None:
                out[id(node)] = out[id(node.parent)] + node.branch_length
        return out
