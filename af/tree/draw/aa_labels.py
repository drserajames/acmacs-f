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


@dataclass
class LabelParams:
    min_share: float = 0.00767  # used when target_labels is None
    target_labels: int | None = None  # label count the size threshold is set to reach
    min_rows_floor: int = 20  # with a target, never label a subtree smaller than this
    max_share: float = 0.99
    consensus: float = 0.6
    backbone_reversion: float = 0.9
    stem_overlap: float = 0.9
    hide_aa: str = "X"


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


def select_labels(
    tree: DrawTree,
    layout: Layout,
    clade_bands: Sequence[tuple[int, int]] = (),
    p: LabelParams | None = None,
) -> tuple[list[AALabel], dict[str, int]]:
    p = p or LabelParams()
    counts: Counter[str] = Counter()
    rows_aa = [tree.aa[i] for i in layout.leaf_nodes]
    by_pos: dict[int, list[int]] = {}
    for node, subs in enumerate(tree.aa_subs):
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
    for node, subs in enumerate(tree.aa_subs):
        if not subs or tree.is_leaf(node) or layout.first_row[node] < 0:
            continue
        counts["candidate nodes"] += 1
        nrows = layout.rows_of(node)
        if nrows < min_rows:
            counts["dropped: small subtree"] += 1
            continue
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
            forward = next((a for a in tree.ancestors(node) if back in tree.aa_subs[a]), None)
            partial = forward is not None and nrows < p.backbone_reversion * layout.rows_of(forward)
            if partial and not is_stem:
                counts["dropped subs: partial reversion"] += 1
                continue
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


def _to_target(labels: list[AALabel], target: int) -> tuple[list[AALabel], int]:
    """Keep the ``target`` largest labels; ties at the cut stay in, so the count can exceed it.

    Returns the kept labels in their original (tree) order and the row threshold used.
    """
    if len(labels) <= target:
        return labels, min((lab.rows for lab in labels), default=0)
    threshold = sorted((lab.rows for lab in labels), reverse=True)[target - 1]
    return [lab for lab in labels if lab.rows >= threshold], threshold
