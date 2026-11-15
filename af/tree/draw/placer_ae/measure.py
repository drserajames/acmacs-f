"""What a layout is worth, in ae's terms: the drawn leader, the residual conflicts, the metrics
line, the Phase-A objective of any layout, and af's own counters for an ae layout.

Why this module reports with predicates.py and never with geometry of its own: on ae main the
residual WARNING (box + m) and the metrics line (bare boxes) disagreed by construction, so a
layout could be "clean" by one and not the other. ae's ink branch (ed4a2ab) counts both with the
tests the search minimised; so does this, in either clash mode. Only text-over-branch and
text-over-name use the text box, as ae's do: the search vetoes those at generation, it never
costs them pairwise.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from ..place import box_overlap, segment_hits_box, segments_cross
from . import predicates as pr
from .geometry import glyph_rects
from .scene import ROW, Box, Label, Options, Point, Scene, Weights

LEADER_GAP = 0.12  # x the label's fs: the drawn leader stops this short of the text
NO_LEADER = 0.4  # x the label's fs: no leader drawn within this of the end, on both axes
SHALLOW = 22.0  # degrees: ae's metrics count leaders flatter than this


# ---- the drawn leader (spec §5) -----------------------------------------------------------------


def drawn_leader(label: Label, attach: Point) -> tuple[Point, Point] | None:
    """The leader as ae draws it (M:3344-3362): from the end on the tree to the attach point,
    pulled back by min(0.12 fs, 0.4 L) so it does not touch the text; none when the attach point
    is within 0.4 fs of the end on both axes (Chebyshev), where a leader would be a speck. The
    search always costs the full segment; this is drawing only."""
    (ex, ey), (cx, cy), fs = label.end, attach, label.fs
    if abs(cx - ex) <= NO_LEADER * fs and abs(cy - ey) <= NO_LEADER * fs:
        return None
    lx, ly = ex - cx, ey - cy
    ll = math.hypot(lx, ly)
    gap = min(fs * LEADER_GAP, ll * 0.4)
    return (ex, ey), (cx + lx / ll * gap, cy + ly / ll * gap)


# ---- residual conflicts (spec §7) ---------------------------------------------------------------


def glyph_gap(p: pr.Placed, q: pr.Placed) -> float:
    """Closest approach of two labels' glyph lines, ae's way (I:2725): over line pairs, the
    largest axis gap (not Euclidean), 0 when they touch or overlap."""
    (la, a, _), (lb, b, _) = p, q
    return min(
        max(rb[0] - ra[2], ra[0] - rb[2], rb[1] - ra[3], ra[1] - rb[3], 0.0)
        for ra in glyph_rects(a, la.lw, la.fs)
        for rb in glyph_rects(b, lb.lw, lb.fs)
    )


@dataclass(frozen=True)
class Conflict:
    """One pair i < j with pconf > 0, and which of pconf's four tests it failed."""

    i: int
    j: int
    kinds: tuple[str, ...]  # text/text (box/box in main mode), leader/leader, i-leader/j-text, ...
    glyph_gap: float | None  # with a text/text kind in ink mode
    texts: tuple[str, str]
    ends: tuple[Point, Point]
    boxes: tuple[Box, Box]
    pinned: tuple[bool, bool]
    hand: tuple[bool, bool]
    tiers: tuple[int | None, int | None]

    @property
    def count(self) -> int:
        """What it adds to the conflict total: pconf, one per failed test."""
        return len(self.kinds)

    @property
    def both_pinned(self) -> bool:
        return all(self.pinned)

    def __str__(self) -> str:
        kinds = ", ".join(
            f"{k} (glyph gap {self.glyph_gap:.2f}pt)" if k == "text/text" else k for k in self.kinds
        )
        tail = ""
        if all(self.hand):
            tail = "; both placed by hand: move one in the label editor"
        elif self.both_pinned:
            tail = "; both pinned, which the placer cannot move"
        return f"residual conflict [{kinds}]: {self._side(0)} vs {self._side(1)}{tail}"

    def _side(self, s: int) -> str:
        (ex, ey), (x0, y0, x1, y1) = self.ends[s], self.boxes[s]
        who = " pinned by hand" if self.hand[s] else " pinned" if self.pinned[s] else ""
        tier = "" if self.tiers[s] is None else f" tier {self.tiers[s]}"
        text = self.texts[s].replace("\n", " ")
        return (
            f"'{text}'{who} end({ex:.1f},{ey:.1f}) box({x0:.1f},{y0:.1f} "
            f"{x1 - x0:.1f}x{y1 - y0:.1f}){tier}"
        )


def residuals(
    placed: Sequence[pr.Placed], k: pr.Consts, tiers: Sequence[int] | None = None
) -> list[Conflict]:
    """Every pair i < j that pconf counts, with its kinds from the same four tests (I:3082-3102).
    Their counts sum to the search's conflict total; nothing here is refused, only reported."""
    n = len(placed)
    found = (_conflict(placed, i, j, k, tiers) for i in range(n) for j in range(i + 1, n))
    return [c for c in found if c is not None]


def _conflict(
    placed: Sequence[pr.Placed], i: int, j: int, k: pr.Consts, tiers: Sequence[int] | None
) -> Conflict | None:
    p, q = placed[i], placed[j]
    kinds = []
    if pr.text_clash(p, q, k):
        kinds.append("text/text" if k.ink else "box/box")
    if pr.leaders_cross(p, q):
        kinds.append("leader/leader")
    if pr.leader_over_text(p, q, k):
        kinds.append("i-leader/j-text")
    if pr.leader_over_text(q, p, k):
        kinds.append("j-leader/i-text")
    if not kinds:
        return None
    (la, a, _), (lb, b, _) = p, q
    return Conflict(
        i=i,
        j=j,
        kinds=tuple(kinds),
        glyph_gap=glyph_gap(p, q) if "text/text" in kinds else None,
        texts=(la.text, lb.text),
        ends=(la.end, lb.end),
        boxes=(a, b),
        pinned=(la.pinned is not None, lb.pinned is not None),
        hand=(la.hand, lb.hand),
        tiers=(None, None) if tiers is None else (tiers[i], tiers[j]),
    )


def warning(conflicts: Sequence[Conflict]) -> str | None:
    """ae's closing WARNING line: the conflict total (pconf summed, so a pair can count up to
    4) and how much of it lies between two pinned labels, which no search can move."""
    total = sum(c.count for c in conflicts)
    if total == 0:
        return None
    pinned = sum(c.count for c in conflicts if c.both_pinned)
    return (
        f"{total} residual conflict(s) could not be removed; "
        f"{pinned} of them between two pinned labels, which the placer cannot move"
    )


# ---- the metrics line (spec §7) -----------------------------------------------------------------


def metrics(
    placed: Sequence[pr.Placed],
    scene: Scene,
    k: pr.Consts,
    leader_cells: Callable[[Point, Point], int] | None = None,
    tiers: Sequence[int] | None = None,
    weights: Weights | None = None,
) -> dict[str, Any]:
    """ae's ink-branch metrics line (I:3104-3151) as a dict, floats to 0.01. text-over-text,
    leader-over-text and leader-over-leader are pconf's three parts, so their sum is the
    residual total. ``leader_cells`` is the occupancy-grid walk (uncapped), ``tiers`` the
    chosen candidates' tiers; without them those fields are left out, not zeroed."""
    w = weights or Weights()
    len_max = w.len_max * scene.height
    lens = sorted(_length(p) for p in placed)
    angs = sorted(_angle_deg(p) for p in placed)
    out: dict[str, Any] = {
        "n": len(placed),
        **_pair_counts(placed, scene, k),
        "leader_len_med_pct": _r(100 * _upper_median(lens) / scene.height),
        "leader_len_max_pct": _r(100 * lens[-1] / scene.height if lens else 0.0),
        "leader_len_med_pt": _r(_upper_median(lens)),
        "leader_len_max_pt": _r(lens[-1] if lens else 0.0),
        "len_max_pt": _r(len_max),
        "over_len_max": sum(v > len_max for v in lens),
        "angle_min_deg": _r(angs[0] if angs else 0.0),
        "angle_med_deg": _r(_upper_median(angs)),
        "below_22deg": sum(a < SHALLOW for a in angs),
        "not_ne_sw": sum(att[1] < lab.end[1] for lab, _, att in placed),
    }
    if leader_cells is not None:
        cells = sorted(leader_cells(lab.end, att) for lab, _, att in placed)
        out["leader_cells_med"] = _upper_median(cells)
        out["leader_cells_max"] = cells[-1] if cells else 0
    if tiers is not None:
        _check_len(tiers, placed, "tiers")
        out["off_envelope"] = sum(t == 2 for t in tiers)
        out["band_sweep"] = sum(t == 3 for t in tiers)
    out["pinned"] = sum(lab.pinned is not None for lab, _, _ in placed)
    out["min_glyph_gap_pt"] = _r(_min_glyph_gap(placed))
    return out


def _pair_counts(placed: Sequence[pr.Placed], scene: Scene, k: pr.Consts) -> dict[str, int]:
    """The five counters. Text-over-name is at most one per label (ae breaks on the first)."""
    on_branch = sum(any(_overlaps(b, s) for s in scene.strokes) for _, b, _ in placed)
    on_name = sum(any(_overlaps(b, o) for o in scene.obstacles) for _, b, _ in placed)
    text = cross = through = 0
    for i, p in enumerate(placed):
        for j, q in enumerate(placed):
            if j == i:
                continue
            through += pr.leader_over_text(p, q, k)
            if j > i:
                text += pr.text_clash(p, q, k)
                cross += pr.leaders_cross(p, q)
    return {
        "text_over_branch": on_branch,
        "text_over_name": on_name,
        "text_over_text": text,
        "leader_over_text": through,
        "leader_over_leader": cross,
    }


def _overlaps(a: Box, b: Box) -> bool:
    """Strict overlap, ae's box_hits_ink: boxes that only touch do not count."""
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def _length(p: pr.Placed) -> float:
    lab, _, (cx, cy) = p
    return math.hypot(lab.end[0] - cx, lab.end[1] - cy)


def _angle_deg(p: pr.Placed) -> float:
    """The leader's slope from horizontal, 0-90 degrees, whichever way it points (ae's
    atan2(|dy|, max(|dx|, 1e-9)))."""
    lab, _, (cx, cy) = p
    dx, dy = lab.end[0] - cx, lab.end[1] - cy
    return math.degrees(math.atan2(abs(dy), max(abs(dx), 1e-9)))


def _upper_median(v: Sequence[float]) -> float:
    """ae's median, v[n/2] of the sorted list: the upper one of an even count."""
    return v[len(v) // 2] if v else 0.0


def _min_glyph_gap(placed: Sequence[pr.Placed]) -> float:
    """The tightest any two labels' glyphs came (0 with fewer than two labels, as ae prints)."""
    gaps = [glyph_gap(p, q) for i, p in enumerate(placed) for q in placed[i + 1 :]]
    return min(gaps) if gaps else 0.0


def _r(v: float) -> float:
    return round(float(v), 2)


def _check_len(v: Sequence[Any], placed: Sequence[pr.Placed], name: str) -> None:
    if len(v) != len(placed):
        raise ValueError(f"{name}: {len(v)} values for {len(placed)} labels")


# ---- the Phase-A objective of any layout (spec §3.4, AEL_REF) -----------------------------------


def score(
    placed: Sequence[pr.Placed], k: pr.Consts, opts: Options, base: Sequence[float]
) -> dict[str, float]:
    """ae's Phase-A score of a layout, split into its parts, so a hand layout or the grid
    placer's can be judged by the objective the ae search minimised (ae's AEL_REF, 4309703).
    ``base[i]`` is label i's own cost at its box (the caller costs it; 0 for a pinned label).
    Sums run i < j in scene order, as ae's; floats are not rounded."""
    _check_len(base, placed, "base")
    conf = inversions = 0
    soft = 0.0
    for i, p in enumerate(placed):
        for q in placed[i + 1 :]:
            conf += pr.pconf(p, q, k)
            soft += pr.psoft(p, q, k)
            inversions += pr.inv(p, q)
    base_total = float(sum(base))
    w = opts.weights
    return {
        "conf_total": conf,
        "soft_total": soft,
        "inv_total": inversions,
        "base_total": base_total,
        "score_a": w.wc * conf + w.wsoft * soft + w.wo * inversions + base_total,
    }


# ---- af's own counters for an ae layout ---------------------------------------------------------

AF_PAD_X = 1.0  # af's placer box: the text plus a unit left and right ...
AF_PAD_Y = 0.5  # ... and half a unit above and below (render: line_height * n + 1)
AF_LEADER_PAD = 0.5  # place.segment_hits_box's default pad


def af_counters(
    placed: Sequence[pr.Placed],
    scene: Scene,
    ink: Callable[[Box], int] | None = None,
    beside_gap: float | None = None,
) -> dict[str, Any]:
    """place.placement_metrics' and render.glyph_metrics' counters, re-derived for an ae layout,
    so an ae render and a grid render can be compared on af's numbers too.

    The mapping (af's own rule kept; ae's geometry put in):
    - af's box is ae's text box grown by 1 pt left and right and 0.5 pt above and below: ae's
      pen advance and rows (1.18 fs) are kept, where af's box has the TextPath width and 1.2 fs
      rows (render._af_box), so an af box here is a little narrower and 0.02 fs a line shorter;
    - the leader is ae's costed segment, end to attach point (the one pconf tests, drawn or
      not), not af's box-right-edge-middle to the node;
    - labels_on_tree_ink is ``ink(box) > 0`` with ``ink`` af's ``Grid.cost`` if given, else a
      strict overlap with any exact stroke rect (no 1 pt raster, no 0.25 pt stroke widening);
    - box_overlaps counts label pairs only, as render passes no obstacles;
    - glyph counters use ae's glyph block (first line's cap top to the last's cap bottom, the
      widest line's advance), af's pitch being ae's 1.18 and the stack row gap 0.46 fs; they are
      given only with ``beside_gap``, as render gives them only with label_beside_gap.
    """
    boxes = [_af_box(b) for _, b, _ in placed]
    segs = [(lab.end, att) for lab, _, att in placed]
    out: dict[str, Any] = {
        "labels": len(placed),
        "box_overlaps": _count_pairs(boxes, lambda a, b: box_overlap(a, b) > 0),
        "labels_on_tree_ink": sum(_on_ink(b, scene, ink) for b in boxes),
        "leader_crossings": _leader_crossings(segs),
        "leaders_through_labels": _through(segs, boxes, AF_LEADER_PAD),
        "median_leader_pt": round(_np_median([_length(p) for p in placed]), 1),
    }
    if beside_gap is not None:
        glyphs = [_glyph_block(p) for p in placed]
        row_gap = scene.mrca_fs * (ROW - 0.72)
        out["labels_beside"] = _count_pairs(
            glyphs, lambda a, b: _hits(a, b, beside_gap, row_gap) and not _hits(a, b, 0.0)
        )
        out["leaders_through_glyphs"] = _through(segs, glyphs, 0.0)
    return out


def _af_box(b: Box) -> Box:
    return (b[0] - AF_PAD_X, b[1] - AF_PAD_Y, b[2] + AF_PAD_X, b[3] + AF_PAD_Y)


def _on_ink(box: Box, scene: Scene, ink: Callable[[Box], int] | None) -> bool:
    if ink is not None:
        return ink(box) > 0
    return any(_overlaps(box, s) for s in scene.strokes)


def _count_pairs(items: Sequence[Box], test: Callable[[Box, Box], bool]) -> int:
    return sum(test(a, b) for i, a in enumerate(items) for b in items[i + 1 :])


def _leader_crossings(segs: Sequence[tuple[Point, Point]]) -> int:
    """place.segments_cross over pairs (the same tie rule as ae's: a shared end crosses)."""
    return sum(
        segments_cross(*segs[i], *segs[j])
        for i in range(len(segs))
        for j in range(i + 1, len(segs))
    )


def _through(segs: Sequence[tuple[Point, Point]], boxes: Sequence[Box], pad: float) -> int:
    """place.segment_hits_box(leader_i, box_j, pad) over ordered pairs j != i."""
    return sum(
        segment_hits_box(s, e, b, pad)
        for i, (s, e) in enumerate(segs)
        for j, b in enumerate(boxes)
        if j != i
    )


def _glyph_block(p: pr.Placed) -> Box:
    lab, box, _ = p
    rects = glyph_rects(box, lab.lw, lab.fs)
    return (box[0], rects[0][1], box[0] + lab.tw, rects[-1][3])


def _hits(a: Box, b: Box, gap: float, gap_y: float | None = None) -> bool:
    """render._hits: within ``gap`` across and ``gap_y`` (default ``gap``) up and down."""
    gy = gap if gap_y is None else gap_y
    return min(a[2], b[2]) + gap > max(a[0], b[0]) and min(a[3], b[3]) + gy > max(a[1], b[1])


def _np_median(v: Sequence[float]) -> float:
    """numpy's median (the mean of the middle two), as placement_metrics reports; 0 if empty."""
    if not v:
        return 0.0
    s, n = sorted(v), len(v)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2
