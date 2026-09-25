"""The tree as the figure needs it: an I6-shaped, pre-order node table.

Why a separate model: the figure must not depend on how the tree store is written (I6 is
still being settled by workstream 5), and every figure step needs the same few columns. An
adapter from the store fills :class:`DrawTree`; tests build small ones by hand.

Nodes are in pre-order (a parent before its children, children in drawing order), so the
tree must already be ladderized. Index 0 is the root. Node indices are internal to one
figure and never used to select anything (design rule 2); leaves are addressed by ``leaf_id``
or ``name``.
"""

from __future__ import annotations

from dataclasses import dataclass, field


class TreeModelError(ValueError):
    """The node table is inconsistent (wrong lengths, not pre-order, duplicate ids)."""


@dataclass
class DrawTree:
    """Columns per node. Leaf-only columns are ``None`` on internal nodes.

    ``clade`` is the single clade label a leaf was assigned (I6); membership of ancestor
    clades is derived from the nomenclature hierarchy, not stored per leaf.
    ``aa_subs`` holds the ASR substitutions on the branch *into* each node, e.g. ``"N145S"``.
    ``date_precision`` is ``"day"``, ``"month"`` or ``"year"`` (from the original date string).
    """

    parent: list[int]
    edge: list[float]
    leaf_id: list[str | None]
    name: list[str | None]
    date: list[str | None]
    date_precision: list[str | None]
    continent: list[str | None]
    clade: list[str | None]
    aa: list[str | None]
    aa_subs: list[list[str]]
    subtype: str = ""
    children: list[list[int]] = field(init=False)

    def __post_init__(self) -> None:
        n = len(self.parent)
        columns: dict[str, list] = {
            "edge": self.edge,
            "leaf_id": self.leaf_id,
            "name": self.name,
            "date": self.date,
            "date_precision": self.date_precision,
            "continent": self.continent,
            "clade": self.clade,
            "aa": self.aa,
            "aa_subs": self.aa_subs,
        }
        for column, values in columns.items():
            if len(values) != n:
                raise TreeModelError(f"column {column!r} has {len(values)} rows, parent has {n}")
        if n == 0 or self.parent[0] != -1:
            raise TreeModelError("node 0 must be the root (parent -1)")
        self.children = [[] for _ in range(n)]
        for i in range(1, n):
            p = self.parent[i]
            if not 0 <= p < i:
                raise TreeModelError(f"node {i}: parent {p} is not an earlier node (not pre-order)")
            self.children[p].append(i)
        seen: set[str] = set()
        for i in self.leaves():
            leaf_id = self.leaf_id[i]
            if not leaf_id:
                raise TreeModelError(f"leaf node {i} has no leaf_id")
            if leaf_id in seen:
                raise TreeModelError(f"duplicate leaf_id {leaf_id!r}")
            seen.add(leaf_id)

    def __len__(self) -> int:
        return len(self.parent)

    def is_leaf(self, i: int) -> bool:
        return not self.children[i]

    def leaves(self) -> list[int]:
        return [i for i in range(len(self.parent)) if not self.children[i]]

    def ancestors(self, i: int) -> list[int]:
        """Proper ancestors of node ``i``, nearest first."""
        out = []
        p = self.parent[i]
        while p >= 0:
            out.append(p)
            p = self.parent[p]
        return out
