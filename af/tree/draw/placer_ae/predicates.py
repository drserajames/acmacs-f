"""ae's pairwise terms between two placed labels (draw-tree.cc M:2812-2856, ink branch I:2678-2777),
one pair at a time: the conflict count, the soft penetration, the vertical-order inversion and
the leader angle.

This is the one measure: pairs.py vectorises exactly these (and is tested against them), the
search minimises them, measure.py reports them. A placed label is (its Label, its text box, its
attach point); its leader runs from ``label.end`` to the attach point.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

from .geometry import glyph_rects, grow, rects_clash, seg_box, seg_box_len, seg_seg_d, segs_cross
from .scene import CAP, ROW, Box, Label, Options, Point, Scene


@dataclass(frozen=True)
class Consts:
    m: float  # label from label (box margin; glyph side and diagonal gap)
    mt: float  # leader from text
    vsep: float  # glyph lines stacked: a stack's row gap
    leadthr: float  # leaders closer than this pay (soft)
    fs: float  # mrca_fs: the soft terms' unit
    side: float  # side-by-side glyph gap: m, or af's beside gap if wider (Options.beside_gap)
    tolerance: float  # stacked knife edge (Options.stack_tolerance)
    ink: bool  # ae's ink rule (glyph lines) or main's box rule
    wt: int = 1  # a glyph overlap counts this many conflicts (Options.overlap_weight; 1: ae)


def consts(scene: Scene, opts: Options) -> Consts:
    fs, w = scene.mrca_fs, opts.weights
    if not 1 <= opts.overlap_weight <= 100:  # the pair tables are int8: 100 + 3 still fits
        raise ValueError(f"overlap_weight {opts.overlap_weight}: 1 to 100")
    m = w.m * fs
    return Consts(
        m=m,
        mt=w.mt * fs,
        vsep=fs * (ROW - CAP),
        leadthr=w.leadthr * fs,
        fs=fs,
        side=max(m, opts.beside_gap or 0.0),
        tolerance=opts.stack_tolerance,
        ink=opts.label_clash == "ink",
        wt=opts.overlap_weight,
    )


Placed = tuple[Label, Box, Point]  # label, text box, attach point


def _boxes_near(a: Box, b: Box, mx: float, my: float) -> bool:
    return a[0] - mx < b[2] and b[0] - mx < a[2] and a[1] - my < b[3] and b[1] - my < a[3]


def text_clash(p: Placed, q: Placed, k: Consts) -> bool:
    """Label text on label text: main, the boxes within m; ink, any two glyph lines clash."""
    (la, a, _), (lb, b, _) = p, q
    if not k.ink:
        if _boxes_near(a, b, k.m, k.m):
            return True
        return k.side > k.m and _beside_boxes(a, b, k.side)
    if not _boxes_near(a, b, k.side, k.vsep):
        return False
    return any(
        rects_clash(ra, rb, k.side, k.vsep, k.tolerance)
        for ra in glyph_rects(a, la.lw, la.fs)
        for rb in glyph_rects(b, lb.lw, lb.fs)
    )


def _beside_boxes(a: Box, b: Box, gap: float) -> bool:
    """af's R8 rule on boxes (main mode with a beside gap): rows overlapping, closer than gap."""
    return min(a[3], b[3]) > max(a[1], b[1]) and max(a[0], b[0]) - min(a[2], b[2]) < gap


def text_conflicts(p: Placed, q: Placed, k: Consts) -> int:
    """text_clash as a count: ae's 0 or 1, or with ``k.wt`` > 1, wt for glyphs that clash at
    ae's own gap m and 1 for a pair that clashes only by the beside gap."""
    if k.wt == 1:
        return int(text_clash(p, q, k))
    if text_clash(p, q, replace(k, side=k.m)):
        return k.wt
    return int(k.side > k.m and text_clash(p, q, k))


def leader_over_text(p: Placed, q: Placed, k: Consts) -> bool:
    """p's leader over q's text: main, q's box grown by mt; ink, any of q's glyph lines grown."""
    lp, _, ap = p
    lq, b, _ = q
    if not seg_box(lp.end, ap, grow(b, k.mt)):
        return False
    if not k.ink:
        return True
    return any(seg_box(lp.end, ap, grow(r, k.mt)) for r in glyph_rects(b, lq.lw, lq.fs))


def leaders_cross(p: Placed, q: Placed) -> bool:
    return segs_cross(p[0].end, p[2], q[0].end, q[2])


def pconf(p: Placed, q: Placed, k: Consts) -> int:
    """The hard conflict count, 0-4 (weight wc); with ``k.wt`` > 1 a glyph overlap counts wt
    and a pair only inside the wider beside gap 1 (:func:`text_conflicts`)."""
    return (
        text_conflicts(p, q, k)
        + int(leaders_cross(p, q))
        + int(leader_over_text(p, q, k))
        + int(leader_over_text(q, p, k))
    )


def psoft(p: Placed, q: Placed, k: Consts) -> float:
    """Soft penetration (weight wsoft, descent only): how far into each other they go."""
    (la, a, aa), (lb, b, ab) = p, q
    pen = 0.0
    if not k.ink:
        ox = min(a[2], b[2]) - max(a[0], b[0]) + k.m
        oy = min(a[3], b[3]) - max(a[1], b[1]) + k.m
        if ox > 0 and oy > 0:
            pen += 6 * (ox / k.fs) * (oy / k.fs)
        pen += 5 * seg_box_len(la.end, aa, grow(b, k.mt)) / k.fs
        pen += 5 * seg_box_len(lb.end, ab, grow(a, k.mt)) / k.fs
    else:
        if _boxes_near(a, b, k.m, k.vsep):
            for ra in glyph_rects(a, la.lw, la.fs):
                for rb in glyph_rects(b, lb.lw, lb.fs):
                    ox = min(ra[2], rb[2]) - max(ra[0], rb[0]) + k.m
                    oy = min(ra[3], rb[3]) - max(ra[1], rb[1]) + k.vsep
                    if ox > 0 and oy > 0:
                        pen += 6 * (ox / k.fs) * (oy / k.fs)
        pen += _leader_into_lines(la.end, aa, lb, b, k)
        pen += _leader_into_lines(lb.end, ab, la, a, k)
    d = seg_seg_d(la.end, aa, lb.end, ab)
    if d < k.leadthr:
        pen += 4 * (k.leadthr - d) / k.fs
    return pen


def _leader_into_lines(s: Point, e: Point, lb: Label, b: Box, k: Consts) -> float:
    if not seg_box(s, e, grow(b, k.mt)):
        return 0.0
    return sum(5 * seg_box_len(s, e, grow(r, k.mt)) / k.fs for r in glyph_rects(b, lb.lw, lb.fs))


def inv(p: Placed, q: Placed) -> int:
    """1 when the two labels' boxes are in the opposite vertical order to their branches. On one
    node (equal y) it is asymmetric, as ae's: [p's box centre above q's]."""
    cp, cq = (p[1][1] + p[1][3]) / 2, (q[1][1] + q[1][3]) / 2
    return int((p[0].node[1] < q[0].node[1]) != (cp < cq))


def angle(p: Placed) -> float:
    """The leader's signed angle, radians (negative: the label below its branch)."""
    label, _, (cx, cy) = p
    return math.atan2(label.end[1] - cy, label.end[0] - cx)
