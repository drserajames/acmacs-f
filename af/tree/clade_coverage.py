"""Which clades the leaves say are there but no node of the tree carries.

A tree can be wrong in a way every other check passes. The H3 tree published on 30 September 2026
had a positive clock (slope +0.0054/yr, r 0.982), sound placement (correlation 0.982) and 97.76%
agreement with the independent clade calls — and yet **no node of it was labelled G.2.2**, so 3,697
sequences lost their clade. The agreement metric nearly let it through at 1.96% against a 2% limit,
because one vanished clade is a small fraction of a large tree
(`notes/trees/G2-SPLIT.md`, `notes/clades/TREE-H3-E99A0165.md`).

This reports that directly: a clade the leaves' own calls give to many leaves, which no node of the
tree carries. For each one it names the node holding most of those leaves and what that node is
labelled instead, and the **sibling groups beside it**, because the explanation is usually a small
group whose state the reconstruction has taken for the whole clade's. In the G.2.2 case the answer
was a two-leaf sister of a 3,700-leaf clade, and this would have printed it in one run.

It is a **diagnostic, not a guard**: it needs the expected calls, which come from outside this
package (the sequence store's Nextclade calls, or any other independent labelling), and the caller
supplies them, so `af.tree` keeps no dependency on `af.clades`. Deciding whether a vanished clade
should refuse a publish belongs with whoever publishes the clades.
"""

from __future__ import annotations

import collections
import csv
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from af.tree.io import i6


@dataclass(frozen=True)
class SiblingGroup:
    """One child of the holder's parent: its size, and what the expected calls say it holds."""

    node_id: str
    leaves: int
    expected: dict[str, int]  # the expected calls under it, commonest first

    def to_json(self) -> dict[str, Any]:
        return {"node_id": self.node_id, "leaves": self.leaves, "expected": dict(self.expected)}


@dataclass(frozen=True)
class VanishedClade:
    """A clade the expected calls give to many leaves, which no node of the tree carries."""

    clade: str
    expected_leaves: int  # leaves the expected calls put in this clade
    holder_node_id: str | None  # the node holding most of them
    holder_label: str | None  # what the tree labels that node instead
    holder_leaves: int  # leaves under the holder
    holder_from_this_clade: int  # how many of them the expected calls put in this clade
    siblings: list[SiblingGroup]  # the holder's parent's other children, largest first

    @property
    def purity(self) -> float:
        """How cleanly the holder is this clade: 1.0 means it holds nothing else."""
        return self.holder_from_this_clade / self.holder_leaves if self.holder_leaves else 0.0

    def to_json(self) -> dict[str, Any]:
        return {
            "clade": self.clade,
            "expected_leaves": self.expected_leaves,
            "holder_node_id": self.holder_node_id,
            "holder_label": self.holder_label,
            "holder_leaves": self.holder_leaves,
            "holder_from_this_clade": self.holder_from_this_clade,
            "holder_purity": round(self.purity, 4),
            "siblings": [sibling.to_json() for sibling in self.siblings],
        }

    def describe(self) -> str:
        """One paragraph a reader can act on."""
        if self.holder_node_id is None:
            return f"{self.clade}: {self.expected_leaves} leaves expected, none grouped in the tree"
        smallest = min(self.siblings, key=lambda s: s.leaves, default=None)
        beside = ""
        if smallest is not None:
            kinds = ", ".join(f"{n} {k}" for k, n in list(smallest.expected.items())[:2])
            beside = (
                f" Beside it sits a {smallest.leaves}-leaf sibling group ({kinds or 'no calls'}), "
                f"the smallest of {len(self.siblings)}."
            )
        return (
            f"{self.clade}: {self.expected_leaves} leaves expected, no node labelled it. "
            f"{self.holder_from_this_clade} of them sit under node {self.holder_node_id} "
            f"({self.holder_leaves} leaves, {100 * self.purity:.0f}% this clade), which the tree "
            f"labels {self.holder_label!r}.{beside}"
        )


def find_vanished_clades(
    directory: Path,
    expected: Mapping[str, str],
    *,
    min_leaves: int = 100,
    max_siblings: int = 5,
) -> list[VanishedClade]:
    """Clades with at least ``min_leaves`` expected leaves that no node of the tree carries.

    ``expected`` maps leaf key to the clade an independent method calls it. Leaves the tree does not
    have are ignored; leaves with no call are ignored. Largest first.
    """
    table = i6.read_nodes(
        Path(directory),
        columns=["node_id", "parent", "is_leaf", "n_leaves", "leaf_id", "clade"],
    )
    rows = table.to_pydict()
    count = len(rows["node_id"])
    labelled = {clade for clade in rows["clade"] if clade}

    children: list[list[int]] = [[] for _ in range(count)]
    for row in range(1, count):
        children[rows["parent"][row]].append(row)
    subtree = [1] * count  # rows are pre-order, so a subtree is a contiguous block
    for row in range(count - 1, 0, -1):
        subtree[rows["parent"][row]] += subtree[row]

    def calls_under(row: int) -> collections.Counter[str]:
        found: collections.Counter[str] = collections.Counter()
        for other in range(row, row + subtree[row]):
            if rows["is_leaf"][other]:
                call = expected.get(rows["leaf_id"][other] or "")
                if call:
                    found[call] += 1
        return found

    in_tree = collections.Counter(
        call
        for row in range(count)
        if rows["is_leaf"][row] and (call := expected.get(rows["leaf_id"][row] or ""))
    )
    vanished: list[VanishedClade] = []
    for clade, leaves in in_tree.most_common():
        if leaves < min_leaves or clade in labelled:
            continue
        holder, best = None, -1
        for row in range(count):
            if rows["is_leaf"][row]:
                continue
            mine = calls_under(row)[clade]
            if mine < min_leaves:
                continue
            score = mine - (subtree[row] - mine)  # the cleanest large grouping
            if score > best:
                holder, best = row, score
        if holder is None:
            vanished.append(VanishedClade(clade, leaves, None, None, 0, 0, []))
            continue
        parent = rows["parent"][holder]
        siblings = [
            SiblingGroup(
                node_id=rows["node_id"][row],
                leaves=sum(1 for o in range(row, row + subtree[row]) if rows["is_leaf"][o]),
                expected=dict(calls_under(row).most_common()),
            )
            for row in (children[parent] if parent >= 0 else [])
            if row != holder
        ]
        siblings.sort(key=lambda sibling: -sibling.leaves)
        vanished.append(
            VanishedClade(
                clade=clade,
                expected_leaves=leaves,
                holder_node_id=rows["node_id"][holder],
                holder_label=rows["clade"][holder],
                holder_leaves=sum(
                    1 for o in range(holder, holder + subtree[holder]) if rows["is_leaf"][o]
                ),
                holder_from_this_clade=calls_under(holder)[clade],
                siblings=siblings[:max_siblings],
            )
        )
    return vanished


def read_expected(path: Path) -> dict[str, str]:
    """A two-column TSV of leaf key and the clade an independent method calls it."""
    with Path(path).open(newline="") as stream:
        rows = list(csv.reader(stream, delimiter="\t"))
    if not rows:
        raise ValueError(f"{path}: no rows")
    start = 1 if rows[0] and rows[0][0].strip().lower() in {"leaf_id", "leaf", "key"} else 0
    return {
        row[0].strip(): row[1].strip() for row in rows[start:] if len(row) >= 2 and row[1].strip()
    }


def main(argv: list[str] | None = None) -> int:
    import argparse
    import json

    parser = argparse.ArgumentParser(prog="python -m af.tree.clade_coverage", description=__doc__)
    parser.add_argument("version", type=Path, help="a tree-store version directory (I6)")
    parser.add_argument(
        "expected", type=Path, help="TSV: leaf key, the clade an independent method calls it"
    )
    parser.add_argument("--min-leaves", type=int, default=100)
    parser.add_argument("--json", action="store_true", help="machine-readable instead of prose")
    args = parser.parse_args(argv)

    found = find_vanished_clades(
        args.version, read_expected(args.expected), min_leaves=args.min_leaves
    )
    if args.json:
        print(json.dumps([item.to_json() for item in found], indent=1))
    elif not found:
        print(f"no clade with {args.min_leaves}+ expected leaves is missing from this tree")
    else:
        for item in found:
            print(item.describe())
    return 1 if found else 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["SiblingGroup", "VanishedClade", "find_vanished_clades", "read_expected"]
