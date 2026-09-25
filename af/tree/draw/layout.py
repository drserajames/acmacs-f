"""Which leaves are drawn, and where: the layout other figures reuse.

The tree arrives ladderized, so the layout is a pre-order walk over the drawn leaves. Rows are
consecutive integers over drawn leaves only; an inode sits midway between its first and last
drawn child; x is cumulative branch length. Signature pages reuse the same :class:`Layout`, so
it is a plain result with no drawing in it.

Hiding rules are counted and a named hide that matches no leaf is an error (design rule 1),
because a stale name otherwise silently leaves the leaf drawn.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np

from .model import DrawTree


class HideRuleError(ValueError):
    """A hide override names a leaf that is not in the tree."""


@dataclass
class HideRules:
    """What is not drawn. Everything here is counted in :class:`Layout.hidden`.

    Year-only dates are not a hide rule: such a leaf is drawn, and only its month-matrix bar is
    left out (:mod:`.timeseries`), because a year-only date stored as 1 January would put it in
    the wrong month.
    ``min_edge``: a node whose branch is at least this long is hidden with its whole subtree
    (long-branch outliers); ``None`` switches the rule off.
    ``names``: hand overrides by leaf name, each must match a leaf.
    """

    min_edge: float | None = None
    names: frozenset[str] = frozenset()


@dataclass
class Layout:
    leaf_nodes: np.ndarray  # node index per row, top to bottom
    node_y: np.ndarray  # row coordinate per node (float), NaN when not drawn
    node_x: np.ndarray  # cumulative branch length per node
    first_row: np.ndarray  # first drawn row under each node, -1 when none
    last_row: np.ndarray  # last drawn row under each node, -1 when none
    hidden: dict[str, int] = field(default_factory=dict)  # rule -> leaves hidden by it

    @property
    def n_rows(self) -> int:
        return len(self.leaf_nodes)

    def rows_of(self, node: int) -> int:
        """Number of drawn rows under ``node`` (0 when none)."""
        f = int(self.first_row[node])
        return 0 if f < 0 else int(self.last_row[node]) - f + 1

    def row_ids(self, tree: DrawTree) -> list[str]:
        """Leaf ids in row order: the reusable part (signature pages, the I7 JSON)."""
        return [str(tree.leaf_id[i]) for i in self.leaf_nodes]


def shown_leaves(tree: DrawTree, rules: HideRules) -> tuple[np.ndarray, dict[str, int]]:
    """Per node, whether it is a drawn leaf; plus counts per rule (first matching rule wins)."""
    n = len(tree)
    leaf = np.array([tree.is_leaf(i) for i in range(n)])
    counts = {"long branch": 0, "named override": 0}
    names = {tree.name[i] for i in range(n) if leaf[i]}
    missing = sorted(set(rules.names) - names)
    if missing:
        raise HideRuleError(f"{len(missing)} hide override(s) match no leaf: {missing[:5]}")
    under_long = np.zeros(n, bool)
    if rules.min_edge is not None:
        for i in range(1, n):
            under_long[i] = under_long[tree.parent[i]] or tree.edge[i] >= rules.min_edge
    shown = leaf.copy()
    for i in np.flatnonzero(leaf).tolist():
        if under_long[i]:
            counts["long branch"] += 1
        elif tree.name[i] in rules.names:
            counts["named override"] += 1
        else:
            continue
        shown[i] = False
    return shown, counts


def compute_layout(tree: DrawTree, rules: HideRules | None = None) -> Layout:
    shown, counts = shown_leaves(tree, rules or HideRules())
    n = len(tree)
    x = np.zeros(n)
    for i in range(1, n):  # pre-order: the parent is already placed
        x[i] = x[tree.parent[i]] + tree.edge[i]
    first = np.full(n, -1, dtype=np.int64)
    last = np.full(n, -1, dtype=np.int64)
    y = np.full(n, np.nan)
    rows: list[int] = []
    for i in range(n):
        if shown[i]:
            first[i] = last[i] = len(rows)
            y[i] = len(rows)
            rows.append(i)
    for i in range(n - 1, -1, -1):  # reverse pre-order: children before parents
        drawn = [c for c in tree.children[i] if first[c] >= 0]
        if drawn:
            first[i], last[i] = first[drawn[0]], last[drawn[-1]]
            y[i] = (y[drawn[0]] + y[drawn[-1]]) / 2
    if not rows:
        raise ValueError("no leaf left to draw after the hide rules")
    return Layout(np.array(rows, dtype=np.int64), y, x, first, last, counts)


def rows_matching(tree: DrawTree, layout: Layout, names: Iterable[str]) -> np.ndarray:
    """Bool per row: the leaf's id or name is in ``names`` (e.g. a centre's antigens)."""
    wanted = set(names)
    return np.array(
        [tree.leaf_id[i] in wanted or tree.name[i] in wanted for i in layout.leaf_nodes], bool
    )
