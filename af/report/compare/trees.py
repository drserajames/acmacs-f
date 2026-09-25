"""Same-science comparison of two phylogenetic trees: topology and clade agreement.

Leaves are matched by strain name. ae leaf names end in ``_<passage>_<hash>``, and ae keeps one
leaf per identical sequence where af keeps every isolate. So only leaves present on both sides
are compared, with each tree restricted to them.

Topology: normalised Robinson-Foulds distance over the non-trivial splits of the restricted
trees, unrooted, polytomies kept. Each split is hashed as the XOR of random 64-bit leaf keys, so a
100k-leaf tree needs no per-split leaf sets; a split and its complement are one split (the
smaller of ``h`` and ``h ^ total``). Clades: the finest clade per leaf, compared by label and
by adjusted Rand index.
"""

from __future__ import annotations

import json
import lzma
import random
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from af.report.compare.maps import adjusted_rand

AE_HASH_SUFFIX = re.compile(r"_[A-Z0-9]*_?[0-9A-F]{8}$")


def strain_key(name: str) -> str:
    """Strain name without ae's ``_<passage>_<hash>`` suffix (unchanged if there is none)."""
    return AE_HASH_SUFFIX.sub("", name)


@dataclass
class Tree:
    """Minimal rooted tree. Invariant: a node's parent has a smaller index than the node."""

    parent: list[int] = field(default_factory=list)
    leaf_name: dict[int, str] = field(default_factory=dict)
    leaf_clades: dict[int, list[str]] = field(default_factory=dict)  # coarse -> fine

    def add(self, parent: int) -> int:
        if parent >= len(self.parent):
            raise ValueError("a parent must be added before its children")
        self.parent.append(parent)
        return len(self.parent) - 1


def read_ae_tjz(path: Path) -> Tree:
    """Read an ae ``phylogenetic-tree-v3`` file (xz JSON). Iterative: trees are deep."""
    raw = path.read_bytes()
    doc = json.loads(lzma.decompress(raw) if raw[:6] == b"\xfd7zXZ\x00" else raw)
    tree = Tree()
    stack: list[tuple[dict[str, Any], int]] = [(doc["tree"], -1)]
    while stack:
        node, parent = stack.pop()
        me = tree.add(parent)
        children = node.get("t")
        if children:
            stack.extend((child, me) for child in children)
        else:
            tree.leaf_name[me] = node["n"]
            tree.leaf_clades[me] = list(node.get("L", []))
    return tree


def _unique_keys(tree: Tree) -> tuple[dict[int, str], int]:
    counts = Counter(strain_key(v) for v in tree.leaf_name.values())
    keep = {n: strain_key(v) for n, v in tree.leaf_name.items() if counts[strain_key(v)] == 1}
    return keep, sum(k for k in counts.values() if k > 1)


def _splits(tree: Tree, keep: dict[int, str], leaf_hash: dict[str, int], total: int) -> set[int]:
    n = len(tree.parent)
    h = [0] * n
    count = [0] * n
    for node, key in keep.items():
        h[node] = leaf_hash[key]
        count[node] = 1
    for node in range(n - 1, 0, -1):  # children before parents
        p = tree.parent[node]
        h[p] ^= h[node]
        count[p] += count[node]
    n_leaves = count[0]
    return {
        min(h[node], h[node] ^ total) for node in range(1, n) if 2 <= count[node] <= n_leaves - 2
    }


def compare(ref: Tree, new: Tree, seed: int = 1) -> dict[str, Any]:
    rk, rdup = _unique_keys(ref)
    nk, ndup = _unique_keys(new)
    common = set(rk.values()) & set(nk.values())
    rng = random.Random(seed)
    leaf_hash = {k: rng.getrandbits(64) for k in sorted(common)}
    total = 0
    for value in leaf_hash.values():
        total ^= value
    rkeep = {n: k for n, k in rk.items() if k in common}
    nkeep = {n: k for n, k in nk.items() if k in common}
    rs = _splits(ref, rkeep, leaf_hash, total)
    ns = _splits(new, nkeep, leaf_hash, total)
    rf = len(rs ^ ns)

    def finest(tree: Tree, node: int) -> str:
        clades = tree.leaf_clades.get(node) or [""]
        return clades[-1]

    rc = {k: finest(ref, n) for n, k in rkeep.items()}
    nc = {k: finest(new, n) for n, k in nkeep.items()}
    keys = sorted(common)
    lr, ln = [rc[k] for k in keys], [nc[k] for k in keys]
    disagree = Counter((a, b) for a, b in zip(lr, ln, strict=True) if a != b)
    return {
        "ref_leaves": len(ref.leaf_name), "new_leaves": len(new.leaf_name),
        "duplicate_names_dropped": {"ref": rdup, "new": ndup},
        "common_leaves": len(common),
        "splits": {"ref": len(rs), "new": len(ns), "shared": len(rs & ns)},
        "rf": rf, "rf_normalised": rf / max(1, len(rs) + len(ns)),
        "clade": {
            "label_agreement": sum(a == b for a, b in zip(lr, ln, strict=True)) / max(1, len(keys)),
            "adjusted_rand": adjusted_rand(lr, ln),
            "top_disagreements": [f"{a} -> {b}: {n}" for (a, b), n in disagree.most_common(5)],
        },
    }  # fmt: skip
