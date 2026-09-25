"""Same-science comparison of two phylogenetic trees: topology and clade agreement.

Leaves are matched by strain name. ae leaf names end in ``_<passage>_<hash>``, and ae keeps one
leaf per identical sequence where af keeps every isolate. So only leaves present on both sides
are compared, with each tree restricted to them.

Topology: normalised Robinson-Foulds distance and reference-split recovery over the
non-trivial splits of the restricted trees (unrooted, polytomies kept), reported by the size of
each split's smaller side (see :func:`split_agreement` for why). Each split is hashed as the
XOR of random 64-bit leaf keys, so a 100k-leaf tree needs no per-split leaf sets; a split and
its complement are one split (the smaller of ``h`` and ``h ^ total``).

Clades: the finest clade per leaf, compared by label and by adjusted Rand index.
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


def read_newick(path: Path, collapse_at_most: float | None = None) -> Tree:
    """Read a Newick tree. Iterative, so 100k-leaf ladder-shaped trees do not hit recursion limits.

    ``collapse_at_most``: internal branches with length <= this are collapsed (their children move
    up to the grandparent). A zero-length branch is an arbitrary resolution of a polytomy, so two
    runs that differ only by seed resolve it differently, and counting it would measure the seed.
    Leaf clades are not in Newick; they stay empty (compare clades from the annotation table).
    """
    text = path.read_text().strip()
    if not text.endswith(";"):
        raise ValueError(f"{path}: Newick does not end with ';'")
    parents: list[int] = []
    lengths: list[float | None] = []
    names: dict[int, str] = {}
    stack: list[int] = []
    current = -1
    i, n = 0, len(text) - 1

    def new_node(parent: int) -> int:
        parents.append(parent)
        lengths.append(None)
        return len(parents) - 1

    def read_label(i: int) -> tuple[str, int]:
        if text[i] == "'":
            end = text.index("'", i + 1)
            return text[i + 1 : end], end + 1
        j = i
        while j < n and text[j] not in ",():;[":
            j += 1
        return text[i:j].strip(), j

    while i < n:
        c = text[i]
        if c == "(":
            current = new_node(stack[-1] if stack else -1)
            stack.append(current)
            i += 1
        elif c == ",":
            i += 1
        elif c == ")":
            current = stack.pop()
            i += 1
            label, i = read_label(i)  # internal labels (support values) are ignored
        elif c == ":":
            j = i + 1
            while j < n and text[j] not in ",();[":
                j += 1
            lengths[current] = float(text[i + 1 : j])
            i = j
        elif c == "[":
            i = text.index("]", i) + 1  # comments
        else:
            current = new_node(stack[-1] if stack else -1)
            label, i = read_label(i)
            names[current] = label
    if stack:
        raise ValueError(f"{path}: unbalanced parentheses")
    return _build(parents, lengths, names, collapse_at_most)


def _build(
    parents: list[int], lengths: list[float | None], names: dict[int, str],
    collapse_at_most: float | None,
) -> Tree:  # fmt: skip
    """Renumber into a :class:`Tree`, skipping collapsed internal nodes (parents first)."""
    tree = Tree()
    new_index: dict[int, int] = {}
    for old, parent in enumerate(parents):  # a parent always precedes its children here
        target = -1 if parent < 0 else new_index[parent]
        length = lengths[old]
        collapse = (
            collapse_at_most is not None and old not in names and parent >= 0
            and length is not None and length <= collapse_at_most
        )  # fmt: skip
        if collapse:
            new_index[old] = target  # children attach to the grandparent
            continue
        new_index[old] = tree.add(target)
        if old in names:
            tree.leaf_name[new_index[old]] = names[old]
            tree.leaf_clades[new_index[old]] = []
    return tree


def _unique_keys(tree: Tree) -> tuple[dict[int, str], int]:
    counts = Counter(strain_key(v) for v in tree.leaf_name.values())
    keep = {n: strain_key(v) for n, v in tree.leaf_name.items() if counts[strain_key(v)] == 1}
    return keep, sum(k for k in counts.values() if k > 1)


def _splits(
    tree: Tree, keep: dict[int, str], leaf_hash: dict[str, int], total: int
) -> dict[int, int]:
    """Split hash -> size of its smaller side, for every non-trivial split."""
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
        min(h[node], h[node] ^ total): min(count[node], n_leaves - count[node])
        for node in range(1, n)
        if 2 <= count[node] <= n_leaves - 2
    }


SPLIT_SIZES = (2, 10, 100, 1000)


def split_agreement(rs: dict[int, int], ns: dict[int, int], min_size: int) -> dict[str, Any]:
    """RF and recovery over splits whose smaller side has at least ``min_size`` leaves.

    Why by size: two CMAPLE runs on identical input differ only by seed, yet disagree on 13-22%
    of all splits, while large clades are stable (WS5, 25 Sep 2026: B/Vic splits above 1,000
    leaves identical). Fine topology is not reproducible, so a threshold is only meaningful for
    large splits. ``ref_recovered`` is the fraction of the reference's splits present in the
    new tree. Unlike RF, it does not count against a new tree for resolving a polytomy the
    reference left collapsed (ae trees collapse many zero-length branches).
    """
    r = {h for h, size in rs.items() if size >= min_size}
    n = {h for h, size in ns.items() if size >= min_size}
    rf = len(r ^ n)
    return {
        "ref": len(r), "new": len(n), "shared": len(r & n), "rf": rf,
        "rf_normalised": rf / max(1, len(r) + len(n)),
        "ref_recovered": len(r & n) / len(r) if r else float("nan"),
    }  # fmt: skip


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
    by_size = {f">={m}": split_agreement(rs, ns, m) for m in SPLIT_SIZES}

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
        "tip_sets_identical": set(rk.values()) == set(nk.values()) and not rdup and not ndup,
        "splits_by_size": by_size,
        "rf_normalised": by_size[">=2"]["rf_normalised"],
        "clade": {
            "label_agreement": sum(a == b for a, b in zip(lr, ln, strict=True)) / max(1, len(keys)),
            "adjusted_rand": adjusted_rand(lr, ln),
            "top_disagreements": [f"{a} -> {b}: {n}" for (a, b), n in disagree.most_common(5)],
        },
    }  # fmt: skip


# ---- tree figures (I7) ----------------------------------------------------------------------


def _figure_leaves(doc: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], int]:
    """Drawn leaves keyed by spelling-normalised strain name; ambiguous keys dropped, counted."""
    from af.report.compare.maps import spelling_key

    drawn = [leaf for leaf in doc["tree"]["leaves"] if leaf["shown"]]
    counts = Counter(spelling_key(strain_key(leaf["name"])) for leaf in drawn)
    unique = {
        spelling_key(strain_key(leaf["name"])): leaf
        for leaf in drawn
        if counts[spelling_key(strain_key(leaf["name"]))] == 1
    }
    return unique, sum(n for n in counts.values() if n > 1)


def _spearman(x: list[float], y: list[float]) -> float:
    """Rank correlation (no ties expected: positions are distinct)."""
    n = len(x)
    if n < 3:
        return float("nan")

    def ranks(v: list[float]) -> list[int]:
        order = sorted(range(n), key=lambda i: v[i])
        r = [0] * n
        for rank, i in enumerate(order):
            r[i] = rank
        return r

    rx, ry = ranks(x), ranks(y)
    d2 = sum((a - b) ** 2 for a, b in zip(rx, ry, strict=True))
    return 1 - 6 * d2 / (n * (n * n - 1))


def compare_figures(ref: dict[str, Any], new: dict[str, Any]) -> dict[str, Any]:
    """Same-science comparison of two tree-figure I7s: which leaves, their order, clades, sections.

    Order is the rank correlation of the drawn position of every leaf both figures draw: 1 means
    the same top-to-bottom order. Sections are matched by clade label; each is compared by the
    leaves (drawn on both sides) that fall inside it.
    """
    ri, rdup = _figure_leaves(ref)
    ni, ndup = _figure_leaves(new)
    common = sorted(ri.keys() & ni.keys())
    out: dict[str, Any] = {
        "ref_drawn": len(ri), "new_drawn": len(ni), "common": len(common),
        "ambiguous_dropped": {"ref": rdup, "new": ndup},
        "jaccard": len(common) / max(1, len(ri.keys() | ni.keys())),
        "only_ref_keys": sorted(ri[k]["name"] for k in ri.keys() - ni.keys()),
        "only_new_keys": sorted(ni[k]["name"] for k in ni.keys() - ri.keys()),
        "order_spearman": _spearman(
            [float(ri[k]["order"]) for k in common], [float(ni[k]["order"]) for k in common]
        ),
    }  # fmt: skip
    lr = [str(ri[k]["clade"]) for k in common]
    ln = [str(ni[k]["clade"]) for k in common]
    out["clade"] = {
        "label_agreement": sum(a == b for a, b in zip(lr, ln, strict=True)) / max(1, len(common)),
        "adjusted_rand": adjusted_rand(lr, ln),
    }
    out["sections"] = _compare_sections(ref, new, ri, ni, set(common))
    rts, nts = ref["tree"]["time_series"], new["tree"]["time_series"]
    out["time_series"] = {"ref": rts, "new": nts, "same": rts == nts}
    return out


def _section_members(
    doc: dict[str, Any], index: dict[str, dict[str, Any]], common: set[str]
) -> tuple[dict[str, set[str]], list[str]]:
    """Clade label -> the common leaves drawn between the section's first and last leaf.

    Section bounds are matched like leaves (ae hash suffix removed, spelling-normalised). A
    section whose bounds are not drawn leaves is returned as unresolved, never skipped silently.
    """
    from af.report.compare.maps import spelling_key

    position = {k: leaf["order"] for k, leaf in index.items() if k in common}
    by_name = {
        spelling_key(strain_key(leaf["name"])): leaf["order"]
        for leaf in doc["tree"]["leaves"]
        if leaf["shown"]
    }
    out: dict[str, set[str]] = {}
    unresolved: list[str] = []
    for section in doc["tree"]["sections"]:
        first = by_name.get(spelling_key(strain_key(section["first_leaf"])))
        last = by_name.get(spelling_key(strain_key(section["last_leaf"])))
        if first is None or last is None:
            unresolved.append(
                f"{section['clade']} ({section['first_leaf']} .. {section['last_leaf']})"
            )
            continue
        lo, hi = min(first, last), max(first, last)
        members = {k for k, p in position.items() if lo <= p <= hi}
        out.setdefault(section["clade"], set()).update(members)  # a split clade: union of parts
    return out, unresolved


def _compare_sections(
    ref: dict[str, Any], new: dict[str, Any], ri: dict[str, dict[str, Any]],
    ni: dict[str, dict[str, Any]], common: set[str],
) -> dict[str, Any]:  # fmt: skip
    (rs, r_unresolved), (ns, n_unresolved) = (
        _section_members(ref, ri, common),
        _section_members(new, ni, common),
    )
    matched = {}
    for label in sorted(rs.keys() & ns.keys()):
        a, b = rs[label], ns[label]
        matched[label] = {"ref": len(a), "new": len(b),
                          "jaccard": len(a & b) / max(1, len(a | b))}  # fmt: skip
    worst = min((v["jaccard"] for v in matched.values()), default=float("nan"))
    return {"matched": matched, "only_ref": sorted(rs.keys() - ns.keys()),
            "only_new": sorted(ns.keys() - rs.keys()), "min_jaccard": worst,
            "unresolved": {"ref": r_unresolved, "new": n_unresolved}}  # fmt: skip
