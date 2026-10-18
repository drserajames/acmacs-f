"""aa-transition labels from the ASR, filtered to the ones worth printing.

The ASR gives substitutions on every branch; a report tree can carry a few dozen. The filters
are the old renderer's two (subtree size, leaf consensus: ae cc/tree/aa-transitions.cc) plus two
that replace hand curation measured on a real round (notes/tree-figure/COMPARISON.md §4):

- size: the subtree has at least ``min_share`` of the drawn rows; or, with ``target_labels``
  set, the threshold follows label density: it is lowered or raised until about that many
  labels survive the other filters (a figure with dense changes gets a higher threshold, a
  sparse one a lower threshold, down to ``min_rows_floor``). A per-subtype constant would bake
  in whatever density one round happened to have;
- near-root: a change carried by more than ``max_share`` of the drawn rows says nothing;
- leaf consensus: the derived residue is the most common one (> ``consensus``) among the
  node's drawn leaves, not counting leaves under a later change at the same position, so a
  substitution that later reverted in a large sub-clade is still judged on the leaves that
  kept it;
- reversions: Y->X below an ancestor's X->Y is not printed unless it covers at least
  ``backbone_reversion`` of the ancestor's rows (a real back-and-forth on the backbone), or
  sits on the stem of a drawn clade band (one of the clade's defining changes). A partial
  reversion is a side branch and would crowd the figure.

Leaf consensus and reversions are af's own (``leaf_filters``). The old renderer's report rule
(AD acmacs-tal DrawAATransitions, notes 09-06) is available too, for reproducing its pages:

- ``min_rows_per_pos``: a smaller subtree is enough at chosen positions (the .tal's
  minimum-number-leaves-in-subtree-per-pos): a node passes the size rule when its rows reach
  the lower of the global minimum and any of its positions' minimums;
- ``flip_removal``: AD's eu-20200915 stage 3. A change X->Y whose descendants revert it (Y->X)
  close below (``flip_max_levels``), over more than ``flip_min_share`` of its rows, is a
  back-and-forth, not a lineage change: it is removed, and the reverts with it.

Named hides (:class:`LabelHide`) take out individual labels a round chooses not to print; each
must match a label (design rule 1) and they are counted.

Labels are keyed by their node's first and last drawn leaf ids, which survive a rebuild; node
indices do not.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .layout import Layout
from .model import DrawTree

SUBSTITUTION = re.compile(r"^([A-Z*-])(\d+)([A-Z*-])$")


@dataclass
class AALabel:
    node: int
    subs: list[str]
    first: str  # leaf id of the first drawn leaf under the node
    last: str  # leaf id of the last drawn leaf under the node
    rows: int


class LabelRuleError(ValueError):
    """A label setting or named label hide that cannot apply (design rule 1)."""


@dataclass
class LabelParams:
    """Values and their reasons: defaults.toml [labels]."""

    target_labels: int  # about this many labels; 0 = use min_share instead
    min_rows_floor: int  # with a target, never label a subtree smaller than this
    min_share: float  # without a target, a label needs this share of the drawn rows
    max_share: float
    consensus: float
    backbone_reversion: float
    stem_overlap: float
    hide_aa: str
    leaf_filters: bool  # af's leaf-consensus and partial-reversion filters
    min_rows_per_pos: dict[str, float]  # position -> minimum rows (< 1: share); needs no target
    flip_removal: bool  # AD eu-20200915 stage 3
    flip_max_levels: int  # a revert this many inode levels below at most (AD: < 3)
    flip_min_share: float  # reverting rows above this share of the change's rows (AD: 0.005)


@dataclass(frozen=True)
class LabelHide:
    """One label not printed: the label on the node that is the common ancestor of two leaf ids.

    ``text`` is the label as printed ("S8N T9A"); it must match, so a hide cannot silently move
    to another label when the tree changes.
    """

    first: str
    last: str
    text: str
    reason: str


def leaf_consensus(seqs: Sequence[str | None], pos1: int) -> tuple[str, float]:
    """Most common residue at 1-based ``pos1`` and its share, ignoring X and gaps."""
    c = Counter(s[pos1 - 1] for s in seqs if s and len(s) >= pos1 and s[pos1 - 1] not in "X-")
    if not c:
        return "", 0.0
    aa, k = c.most_common(1)[0]
    return aa, k / sum(c.values())


def _overlap(a: tuple[int, int], b: tuple[int, int]) -> float:
    inter = min(a[1], b[1]) - max(a[0], b[0]) + 1
    if inter <= 0:
        return 0.0
    return inter / (max(a[1], b[1]) - min(a[0], b[0]) + 1)


def _rows_threshold(value: float, n_rows: int) -> int:
    """AD's reading of a leaf minimum: below 1 a share of the drawn rows, else a count."""
    return int(n_rows * value) if value < 1 else int(value)


def _per_pos_minimum(p: LabelParams, n_rows: int) -> dict[int, int]:
    out = {}
    for key, value in p.min_rows_per_pos.items():
        if not key.isdigit() or int(key) < 1 or value <= 0:
            raise LabelRuleError(
                f"min_rows_per_pos: {key!r} = {value!r} is not a position and a size"
            )
        out[int(key)] = _rows_threshold(value, n_rows)
    if out and p.target_labels:
        raise LabelRuleError("min_rows_per_pos needs target_labels = 0 (a share rule, as in AD)")
    return out


def remove_flips(tree: DrawTree, layout: Layout, p: LabelParams) -> tuple[list[list[str]], int]:
    """AD eu-20200915 stage 3 over the drawn tree's substitutions; returns them and the count of
    changes removed.

    At each position a substitution's ancestral residue is the derived residue of the nearest
    ancestor changing that position. A change X->Y whose descendants revert it, the nearest at most
    ``flip_max_levels - 1`` inode levels below and together holding more than ``flip_min_share``
    of its rows, is removed. Its descendants at that position are then re-anchored to the residue
    above it: a revert becomes X->X and goes, any other change keeps its derived residue.
    """
    subs = [
        list(s) if not tree.is_leaf(i) and layout.first_row[i] >= 0 else []
        for i, s in enumerate(tree.aa_subs)
    ]
    removed = 0
    at_pos: dict[int, set[int]] = {}
    for node, ss in enumerate(subs):
        for s in ss:
            if m := SUBSTITUTION.match(s):
                at_pos.setdefault(int(m.group(2)), set()).add(node)

    def change(node: int, pos: int) -> tuple[str, str] | None:
        for s in subs[node]:
            if (m := SUBSTITUTION.match(s)) and int(m.group(2)) == pos:
                return m.group(1), m.group(3)
        return None

    for pos in sorted(at_pos):
        while True:
            # nearest changing ancestor of each changing node, and how many levels up it is
            anchor: dict[int, tuple[int, int]] = {}
            for d in at_pos[pos]:
                for k, a in enumerate(tree.ancestors(d)):
                    if a in at_pos[pos]:
                        anchor[d] = (a, k)
                        break
            flips: dict[int, list[tuple[int, int]]] = {}
            for d, (a, k) in anchor.items():
                cd, ca = change(d, pos), change(a, pos)
                if cd and ca and cd[1] == ca[0] and ca[1] != cd[1]:
                    flips.setdefault(a, []).append((d, k))
            dropped = next(
                (
                    a
                    for a in sorted(flips, reverse=True)  # deepest first, as AD's post-order
                    if min(k for _, k in flips[a]) < p.flip_max_levels
                    and sum(layout.rows_of(d) for d, _ in flips[a])
                    > p.flip_min_share * layout.rows_of(a)
                ),
                None,
            )
            if dropped is None:
                break
            before, _ = change(dropped, pos) or ("", "")
            subs[dropped] = [s for s in subs[dropped] if not _at(s, pos)]
            at_pos[pos].discard(dropped)
            removed += 1
            for d, (a, _) in anchor.items():
                if a != dropped:
                    continue
                cd = change(d, pos)
                rest = [s for s in subs[d] if not _at(s, pos)]
                if cd and cd[1] != before:
                    rest.append(f"{before}{pos}{cd[1]}")
                else:
                    at_pos[pos].discard(d)
                    removed += 1
                subs[d] = rest
    return subs, removed


def _at(sub: str, pos: int) -> bool:
    m = SUBSTITUTION.match(sub)
    return m is not None and int(m.group(2)) == pos


def select_labels(
    tree: DrawTree,
    layout: Layout,
    clade_bands: Sequence[tuple[int, int]],
    p: LabelParams,
) -> tuple[list[AALabel], dict[str, int]]:
    counts: Counter[str] = Counter()
    rows_aa = [tree.aa[i] for i in layout.leaf_nodes]
    per_pos = _per_pos_minimum(p, layout.n_rows)
    all_subs: Sequence[Sequence[str]] = tree.aa_subs
    if p.flip_removal:
        all_subs, counts["removed subs: back-and-forth (flip)"] = remove_flips(tree, layout, p)
    by_pos: dict[int, list[int]] = {}
    for node, subs in enumerate(all_subs):
        for s in subs:
            if m := SUBSTITUTION.match(s):
                by_pos.setdefault(int(m.group(2)), []).append(node)

    def rows_kept(node: int, pos: int) -> list[str | None]:
        f, last = int(layout.first_row[node]), int(layout.last_row[node])
        keep = np.ones(last - f + 1, bool)
        for other in by_pos.get(pos, []):
            of, ol = int(layout.first_row[other]), int(layout.last_row[other])
            if other != node and of >= 0 and f <= of and ol <= last:
                keep[of - f : ol - f + 1] = False
        return [rows_aa[f + k] for k in np.flatnonzero(keep)]

    min_rows = p.min_rows_floor if p.target_labels else p.min_share * layout.n_rows
    labels = []
    for node, subs in enumerate(all_subs):
        if not subs or tree.is_leaf(node) or layout.first_row[node] < 0:
            continue
        counts["candidate nodes"] += 1
        nrows = layout.rows_of(node)
        positions = [int(m.group(2)) for s in subs if (m := SUBSTITUTION.match(s))]
        need = min([min_rows, *(per_pos[q] for q in positions if q in per_pos)])
        if nrows < need:
            counts["dropped: small subtree"] += 1
            continue
        if nrows < min_rows:
            counts["kept by a position's own minimum"] += 1
        if nrows > p.max_share * layout.n_rows:
            counts["dropped: on nearly every drawn leaf"] += 1
            continue
        span = (int(layout.first_row[node]), int(layout.last_row[node]))
        is_stem = any(_overlap(span, b) >= p.stem_overlap for b in clade_bands)
        keep = []
        for s in subs:
            m = SUBSTITUTION.match(s)
            if not m or m.group(1) in p.hide_aa or m.group(3) in p.hide_aa:
                counts["dropped subs: X or unparsable"] += 1
                continue
            before, pos, after = m.group(1), int(m.group(2)), m.group(3)
            back = f"{after}{pos}{before}"
            forward = next((a for a in tree.ancestors(node) if back in all_subs[a]), None)
            partial = forward is not None and nrows < p.backbone_reversion * layout.rows_of(forward)
            if p.leaf_filters and partial and not is_stem:
                counts["dropped subs: partial reversion"] += 1
                continue
            if p.leaf_filters:
                aa, share = leaf_consensus(rows_kept(node, pos), pos)
                if aa != after or share <= p.consensus:
                    counts["dropped subs: leaf consensus"] += 1
                    continue
            keep.append(s)
        if keep:
            first = str(tree.leaf_id[layout.leaf_nodes[span[0]]])
            last = str(tree.leaf_id[layout.leaf_nodes[span[1]]])
            labels.append(AALabel(node, keep, first, last, nrows))
    if p.target_labels:
        labels, threshold = _to_target(labels, p.target_labels)
        counts["size threshold (rows)"] = threshold
        counts["dropped: below density threshold"] = counts["candidate nodes"] - len(labels)
    counts["labels"] = len(labels)
    return labels, dict(counts)


def hide_labels(
    tree: DrawTree, layout: Layout, labels: list[AALabel], hides: Sequence[LabelHide]
) -> tuple[list[AALabel], int]:
    """Drop the named labels. A hide whose leaves are not drawn, or whose node carries no label
    with that text, is an error: it names nothing on this tree (design rule 1)."""
    if not hides:
        return labels, 0
    node_of = {str(tree.leaf_id[i]): i for i in layout.leaf_nodes}
    by_node = {lab.node: lab for lab in labels}
    drop, problems = set(), []
    for h in hides:
        a, b = node_of.get(h.first), node_of.get(h.last)
        if a is None or b is None:
            problems.append(f"{h.text}: leaf {h.first if a is None else h.last} is not drawn")
            continue
        up = set(tree.ancestors(a)) | {a}
        node = b
        while node not in up:
            node = tree.parent[node]
        lab = by_node.get(node)
        if lab is None or " ".join(lab.subs) != h.text:
            found = " ".join(lab.subs) if lab else "no label"
            problems.append(f"{h.text}: the node of {h.first} .. {h.last} has {found}")
            continue
        drop.add(node)
    if problems:
        raise LabelRuleError("label hide(s) match no label:\n  " + "\n  ".join(problems))
    return [lab for lab in labels if lab.node not in drop], len(drop)


def _to_target(labels: list[AALabel], target: int) -> tuple[list[AALabel], int]:
    """Keep the ``target`` largest labels; ties at the cut stay in, so the count can exceed it.

    Returns the kept labels in their original (tree) order and the row threshold used.
    """
    if len(labels) <= target:
        return labels, min((lab.rows for lab in labels), default=0)
    threshold = sorted((lab.rows for lab in labels), reverse=True)[target - 1]
    return [lab for lab in labels if lab.rows >= threshold], threshold
