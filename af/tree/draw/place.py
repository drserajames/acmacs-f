"""Place aa-transition labels: an ink occupancy grid, a greedy pass and local repair.

Why not adjustText: it pushes labels apart by repulsion and knows nothing about staying off the
tree lines or keeping a leader short. The past report trees put each label in the white space
to the left of its branch point, with a leader to the node; this search does that directly,
with the old renderer's hard rules as heavy costs: no label on tree ink, no leader crossing
another leader or passing through another label, no dead-flat leader (it reads as a branch).

Deterministic: no randomness, candidates in a fixed order, ties broken by that order.
Units are page points, y down; a box is (x0, y0, x1, y1).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

Box = tuple[float, float, float, float]
Point = tuple[float, float]


@dataclass
class Placed:
    key: int  # the caller's id (the node)
    text: str
    anchor: Point  # where the leader ends (the node)
    box: Box
    cost: float


class Grid:
    """Occupancy of tree ink (and anything else to keep labels off), in ``cell``-point cells."""

    def __init__(self, width: float, height: float, cell: float = 1.0) -> None:
        self.cell = cell
        self.ink = np.zeros((int(height / cell) + 2, int(width / cell) + 2), dtype=np.int32)

    def _ij(self, box: Box) -> tuple[int, int, int, int]:
        x0, y0, x1, y1 = box
        n_i, n_j = self.ink.shape
        c = self.cell
        i0, i1 = max(0, int(y0 / c)), min(n_i, int(np.ceil(y1 / c)))
        j0, j1 = max(0, int(x0 / c)), min(n_j, int(np.ceil(x1 / c)))
        return i0, i1, j0, j1

    def add_box(self, box: Box) -> None:
        i0, i1, j0, j1 = self._ij(box)
        self.ink[i0:i1, j0:j1] += 1

    def add_hline(self, x0: float, x1: float, y: float) -> None:
        self.add_box((min(x0, x1), y - 0.25, max(x0, x1), y + 0.25))

    def add_vline(self, x: float, y0: float, y1: float) -> None:
        self.add_box((x - 0.25, min(y0, y1), x + 0.25, max(y0, y1)))

    def cost(self, box: Box) -> int:
        """Number of inked cells under the box."""
        i0, i1, j0, j1 = self._ij(box)
        return int((self.ink[i0:i1, j0:j1] > 0).sum())

    def line_cost(self, p: Point, q: Point) -> int:
        """Number of inked cells the segment p-q passes over (endpoints excluded)."""
        n = max(2, int(np.hypot(q[0] - p[0], q[1] - p[1]) / self.cell))
        xs = np.linspace(p[0], q[0], n)[1:-1]
        ys = np.linspace(p[1], q[1], n)[1:-1]
        i = np.clip((ys / self.cell).astype(int), 0, self.ink.shape[0] - 1)
        j = np.clip((xs / self.cell).astype(int), 0, self.ink.shape[1] - 1)
        return int((self.ink[i, j] > 0).sum())


def box_overlap(a: Box, b: Box) -> float:
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    return w * h if w > 0 and h > 0 else 0.0


def segment_hits_box(p: Point, q: Point, box: Box, pad: float = 0.5) -> bool:
    """Does segment p-q pass through the padded box? (Liang-Barsky clipping.)"""
    x0, y0, x1, y1 = box[0] - pad, box[1] - pad, box[2] + pad, box[3] + pad
    dx, dy = q[0] - p[0], q[1] - p[1]
    t0, t1 = 0.0, 1.0
    for pp, qq in ((-dx, p[0] - x0), (dx, x1 - p[0]), (-dy, p[1] - y0), (dy, y1 - p[1])):
        if pp == 0:
            if qq < 0:
                return False
            continue
        t = qq / pp
        if pp < 0:
            t0 = max(t0, t)
        else:
            t1 = min(t1, t)
        if t0 > t1:
            return False
    return True


def segments_cross(p1: Point, p2: Point, q1: Point, q2: Point) -> bool:
    def orient(a: Point, b: Point, c: Point) -> float:
        return float(np.sign((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])))

    return orient(p1, p2, q1) != orient(p1, p2, q2) and orient(q1, q2, p1) != orient(q1, q2, p2)


def leader_end(box: Box) -> Point:
    """The leader leaves the box at the middle of its right edge (labels sit left of the node)."""
    return (box[2], (box[1] + box[3]) / 2)


@dataclass
class Weights:
    ink: float = 1000.0  # per inked cell under the box: effectively a hard rule
    leader_ink: float = 12.0  # per inked cell the leader passes over
    overlap: float = 400.0  # per square point of overlap with another label or obstacle
    crossing: float = 3000.0  # leader crossing a leader, or passing through a label: hard
    length: float = 0.6  # per point of leader length
    flat: float = 120.0  # leader within 8 degrees of horizontal reads as a branch


@dataclass
class Search:
    dxs: Sequence[float] = (2, 4, 7, 10, 14, 19, 25, 32, 40, 50, 62, 76, 92, 110, 130, 155, 185)
    dys: Sequence[float] = (
        *(0, -3, 3, -6, 6, -10, 10, -15, 15, -21, 21),
        *(-28, 28, -36, 36, -46, 46, -58, 58),
    )
    passes: int = 3
    left: float = 0.0
    top: float = 0.0
    bottom: float = 1e9


def place_labels(
    items: Sequence[tuple[int, str, Point]],
    grid: Grid,
    text_width: Callable[[str], float],
    line_height: float,
    obstacles: Sequence[Box] = (),
    weights: Weights | None = None,
    search: Search | None = None,
) -> list[Placed]:
    """Place ``items`` = [(key, text, anchor)]; a multi-line text uses '\\n'.

    Greedy top to bottom, each label taking its cheapest candidate given those already placed;
    then up to ``passes`` rounds re-placing each label against all the others.
    """
    w, s = weights or Weights(), search or Search()
    boxes: dict[int, Box] = {}
    order = sorted(range(len(items)), key=lambda k: (items[k][2][1], items[k][0]))

    def candidates(k: int) -> list[Box]:
        _, text, (ax, ay) = items[k]
        lines = text.split("\n")
        bw = max(text_width(t) for t in lines) + 2
        bh = line_height * len(lines) + 1
        out = []
        for dx in s.dxs:
            for dy in s.dys:
                box = (ax - dx - bw, ay + dy - bh / 2, ax - dx, ay + dy + bh / 2)
                if box[0] >= s.left and box[1] >= s.top and box[3] <= s.bottom:
                    out.append(box)
        if not out:
            raise ValueError(f"label {text!r} has no candidate position inside the page limits")
        return out

    def cost_of(k: int, box: Box) -> float:
        anchor = items[k][2]
        end = leader_end(box)
        length = float(np.hypot(anchor[0] - end[0], anchor[1] - end[1]))
        c = w.ink * grid.cost(box) + w.leader_ink * grid.line_cost(end, anchor) + w.length * length
        if length > 3 and abs(anchor[1] - end[1]) < np.tan(np.radians(8)) * abs(anchor[0] - end[0]):
            c += w.flat
        for o in obstacles:
            c += w.overlap * box_overlap(box, o)
            c += w.crossing * segment_hits_box(end, anchor, o)
        for j, b in boxes.items():
            if j == k:
                continue
            other_anchor = items[j][2]
            c += w.overlap * box_overlap(box, b)
            c += w.crossing * segments_cross(end, anchor, leader_end(b), other_anchor)
            through = segment_hits_box(end, anchor, b) or segment_hits_box(
                leader_end(b), other_anchor, box
            )
            c += w.crossing * through
        return c

    def best(k: int) -> tuple[float, Box]:
        return min(((cost_of(k, b), b) for b in candidates(k)), key=lambda t: t[0])

    def moved(box: Box, like: Box) -> Box:
        """``box`` re-anchored at ``like``'s leader end (right edge, vertical centre)."""
        ex, ey = leader_end(like)
        h, wd = box[3] - box[1], box[2] - box[0]
        return (ex - wd, ey - h / 2, ex, ey + h / 2)

    def swap_crossed() -> bool:
        """Two crossed leaders uncross when their labels trade places; one label moving
        alone cannot do that, so the greedy passes stall on crowded anchors."""
        swapped = False
        for a_i, a in enumerate(order):
            for b in order[a_i + 1 :]:
                ba, bb = boxes[a], boxes[b]
                if not segments_cross(leader_end(ba), items[a][2], leader_end(bb), items[b][2]):
                    continue
                before = cost_of(a, ba) + cost_of(b, bb)
                boxes[a], boxes[b] = moved(ba, bb), moved(bb, ba)
                if cost_of(a, boxes[a]) + cost_of(b, boxes[b]) < before - 1e-9:
                    swapped = True
                else:
                    boxes[a], boxes[b] = ba, bb
        return swapped

    for k in order:
        boxes[k] = best(k)[1]
    for _ in range(s.passes):
        changed = False
        for k in order:
            c, b = best(k)
            if b != boxes[k] and c < cost_of(k, boxes[k]) - 1e-9:
                boxes[k], changed = b, True
        changed |= swap_crossed()
        if not changed:
            break
    return [
        Placed(items[k][0], items[k][1], items[k][2], boxes[k], cost_of(k, boxes[k])) for k in order
    ]


def placement_metrics(placed: Sequence[Placed], grid: Grid, obstacles: Sequence[Box] = ()) -> dict:
    """What the comparison reports: overlaps, labels on ink, leader crossings, leader length."""
    n_box = n_ink = n_cross = n_through = 0
    lengths = []
    for i, a in enumerate(placed):
        end = leader_end(a.box)
        n_ink += grid.cost(a.box) > 0
        lengths.append(float(np.hypot(a.anchor[0] - end[0], a.anchor[1] - end[1])))
        n_box += sum(box_overlap(a.box, o) > 0 for o in obstacles)
        for b in placed[i + 1 :]:
            n_box += box_overlap(a.box, b.box) > 0
            n_cross += segments_cross(end, a.anchor, leader_end(b.box), b.anchor)
        n_through += sum(segment_hits_box(end, a.anchor, b.box) for b in placed if b is not a)
    return {
        "labels": len(placed),
        "box_overlaps": int(n_box),
        "labels_on_tree_ink": int(n_ink),
        "leader_crossings": int(n_cross),
        "leaders_through_labels": int(n_through),
        "median_leader_pt": round(float(np.median(lengths)), 1) if lengths else 0.0,
    }
