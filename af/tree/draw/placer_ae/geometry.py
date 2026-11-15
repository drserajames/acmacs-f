"""ae's geometry predicates (draw-tree.cc M:2468-2512, ink branch I:2686-2740), one pair at a time.

These are the reference: pairs.py builds its matrices with vectorised versions and is tested
against these, and measure.py reports with these, so what is optimised is what is reported.
Touching counts as crossing (ae's orientation test): two leaders from one node always cross
unless collinear, as af's place.segments_cross.
"""

from __future__ import annotations

import math

from .scene import CAP, ROW, Box, Point

ATTACH_INSET = 0.23  # x fs: a leader leaving a box's top or bottom starts this far inside it


def orient(p: Point, q: Point, r: Point) -> int:
    v = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
    return (v > 0) - (v < 0)


def segs_cross(a: Point, b: Point, c: Point, d: Point) -> bool:
    return orient(a, b, c) != orient(a, b, d) and orient(c, d, a) != orient(c, d, b)


def inside(p: Point, box: Box) -> bool:
    return box[0] <= p[0] <= box[2] and box[1] <= p[1] <= box[3]


def seg_box(p0: Point, p1: Point, box: Box) -> bool:
    """Does the segment touch the box: an end inside (inclusive) or crossing an edge."""
    if inside(p0, box) or inside(p1, box):
        return True
    x0, y0, x1, y1 = box
    corners = ((x0, y0), (x1, y0), (x1, y1), (x0, y1))
    return any(segs_cross(p0, p1, corners[k], corners[(k + 1) % 4]) for k in range(4))


def pt_seg_d(p: Point, a: Point, b: Point) -> float:
    dx, dy = b[0] - a[0], b[1] - a[1]
    l2 = dx * dx + dy * dy
    t = 0.0 if l2 == 0 else max(0.0, min(1.0, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / l2))
    return math.hypot(p[0] - (a[0] + t * dx), p[1] - (a[1] + t * dy))


def seg_seg_d(a: Point, b: Point, c: Point, d: Point) -> float:
    if segs_cross(a, b, c, d):
        return 0.0
    return min(pt_seg_d(a, c, d), pt_seg_d(b, c, d), pt_seg_d(c, a, b), pt_seg_d(d, a, b))


def seg_box_len(a: Point, b: Point, box: Box) -> float:
    """Length of segment ab inside the box (Liang-Barsky)."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    t0, t1 = 0.0, 1.0
    for p, q in (
        (-dx, a[0] - box[0]),
        (dx, box[2] - a[0]),
        (-dy, a[1] - box[1]),
        (dy, box[3] - a[1]),
    ):
        if p == 0:
            if q < 0:
                return 0.0
            continue
        t = q / p
        if p < 0:
            t0 = max(t0, t)
        else:
            t1 = min(t1, t)
        if t0 > t1:
            return 0.0
    return (t1 - t0) * math.hypot(dx, dy)


def grow(box: Box, by: float) -> Box:
    return (box[0] - by, box[1] - by, box[2] + by, box[3] + by)


def attach_pt(end: Point, fs: float, box: Box) -> Point:
    """Where a label's leader leaves it (M:2468-2483): the end of its text at mid-height when the
    box is wholly left of the leader end; otherwise the middle of the box, on the bottom cap edge
    if the box is wholly above the branch, else the top cap edge."""
    x0, y0, x1, y1 = box
    if end[0] > x1:
        return (x1, (y0 + y1) / 2)
    ink = ATTACH_INSET * fs
    return ((x0 + x1) / 2, y1 - ink if y1 < end[1] else y0 + ink)


def glyph_rects(box: Box, lw: tuple[float, ...], fs: float) -> list[Box]:
    """One rect per line (I:2690-2697): the line's own advance wide, a cap tall, centred in its
    row."""
    x0, y0 = box[0], box[1]
    out = []
    for k, w in enumerate(lw):
        y = y0 + (k + 0.5) * ROW * fs - CAP / 2 * fs
        out.append((x0, y, x0 + w, y + CAP * fs))
    return out


def rects_clash(a: Box, b: Box, m: float, vsep: float, tolerance: float = 0.0) -> bool:
    """ae's ink rule for two glyph lines: overlapping; stacked closer than ``vsep``; side by side
    or diagonally closer than ``m``. ``tolerance`` loosens the stacked test (Options)."""
    gx = max(b[0] - a[2], a[0] - b[2])
    gy = max(b[1] - a[3], a[1] - b[3])
    if gx < 0 and gy < 0:
        return True
    if gx < 0:
        return gy < vsep - tolerance
    if gy < 0:
        return gx < m
    return math.hypot(gx, gy) < m
