"""What the ae placer works on: the labels, the tree's ink, the band labels may use, the options.

Why a scene of its own and not af's Grid: ae vetoes a box against the exact stroke rectangles of
the tree, not a 1 pt raster, and measures label sizes from the global label size (``mrca_fs``)
and each label's own (``Label.fs``), which differ only when a label has its own size.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

Box = tuple[float, float, float, float]
Point = tuple[float, float]

ROW = 1.18  # a line's row, in its label's size; the cap is 0.72 of it, centred
CAP = 0.72


@dataclass(frozen=True)
class Label:
    """One aa label. ``end`` is where its leader meets the tree, which every cost measures from:
    ae's middle of the node's horizontal edge, or the node itself (Options.leader_end)."""

    key: int  # the caller's id (the node)
    text: str  # lines joined by "\n"
    lines: tuple[str, ...]
    lw: tuple[float, ...]  # each line's advance, in points
    fs: float  # this label's size, in points
    node: Point  # the branch point
    end: Point  # where the leader ends
    pinned: Box | None = None  # a fixed text box: one candidate, never moved
    hand: bool = False  # pinned by hand (af's rules refuse clashes; ae only reports them)

    @property
    def tw(self) -> float:
        return max(self.lw)

    @property
    def th(self) -> float:
        return len(self.lines) * (self.fs * ROW)  # ae's order (M:2587), to the last bit


@dataclass(frozen=True)
class Scene:
    labels: tuple[Label, ...]
    strokes: tuple[Box, ...]  # the tree's ink, one rect per stroke (exact; also rasterised)
    obstacles: tuple[Box, ...]  # in the grid only: strain labels, title, inset
    band: Box  # (gx0, gy0, gx1, gy1): where a padded label box may lie
    height: float  # page height: ae's lengths are fractions of it
    mrca_fs: float  # the global label size: grid cell, pads, margins, normalisations


@dataclass(frozen=True)
class Weights:
    """ae's constants (draw-tree.cc M:2519-2575, M:2812-2848, M:2941); fractions of the page
    height are given as fractions, sizes as multiples of a label size."""

    gap_l: float = 0.45  # x mrca_fs: a box's right edge at least this left of the leader end
    pad: float = 0.3  # x mrca_fs: around a box, against tree ink and the grid
    len_soft: float = 0.018  # x height
    len_max: float = 0.110  # x height: tier 1's longest leader
    len_max2: float = 0.30  # x height: tier 2's
    ang_short: float = 10.0  # degrees: the angle aimed at up to ang_l1
    ang_long: float = 30.0  # ... and from ang_l2
    ang_l1: float = 0.018  # x height
    ang_l2: float = 0.045  # x height
    ang_min: float = 4.0  # tier 1 samples ang_min..ang_max degrees
    ang_max: float = 72.0
    ang_min2: float = 8.0  # tier 2
    ang_max2: float = 86.0
    noleader: float = 120.0  # a leader under 0.6 sizes
    above: float = 250.0  # K_nw: a box sampled above its branch
    tier2: float = 700.0
    wlen: float = 2.5  # per point of leader
    wquad: float = 6.0  # per ((L - len_soft) / fs)^2
    wang: float = 10.0  # per degree off the target angle
    t3d: float = 4.0  # tier 3: per point from the leader end to the box centre
    wink: float = 22.0  # per grid cell the leader crosses (capped at 80 cells)
    m: float = 0.08  # x mrca_fs: label from label, side by side and diagonally
    mt: float = 0.05  # x mrca_fs: leader from text
    leadthr: float = 0.5  # x mrca_fs: leaders closer than this pay
    wc: float = 1_000_000.0  # one conflict
    wo: float = 1100.0  # one vertical-order inversion
    wa: float = 130.0  # per radian between neighbours' leader angles (tidy pass)
    wsoft: float = 5000.0  # soft penetration (descent)


@dataclass(frozen=True)
class Schedule:
    """ae's search schedule (M:2954-3072)."""

    seed: int = 2463534242
    restarts: int = 400  # random restarts after the ordered seed
    sweeps: int = 60
    stale: int = 120  # restarts without improvement, once the score is under wc
    pool: int = 40  # a random restart picks among each label's pool cheapest
    pair_passes: int = 12  # A2
    window_passes: int = 8  # A3
    window: int = 6  # ranks either side of the conflicting pair
    window_restarts: int = 251
    window_sweeps: int = 40
    tidy_sweeps: int = 200  # B


@dataclass(frozen=True)
class Options:
    """Where af departs from ae, each a switch (Sarah chooses: plan rulings 1 and 2)."""

    # Plan ruling 1: ae vetoes a padded box on tree ink; R6 made af's a cost per inked cell.
    tree_ink: Literal["veto", "soft"] = "veto"
    soft_ink_weight: float = 1000.0  # per grid cell under the padded box, with tree_ink "soft"
    # Plan ruling 2: ae's glyph rule between labels ("ink", ae ed4a2ab: 0.08 sizes side by side,
    # a stack's row gap stacked) or ae main's box rule ("main"); beside_gap adds af's R8 M3 rule
    # (two labels' lines on overlapping rows closer than this, glyph to glyph: a conflict).
    label_clash: Literal["ink", "main"] = "ink"
    beside_gap: float | None = None
    # Two boxes that touch vertically are a row gap apart to the last bit: ae's float arithmetic
    # calls that a clash, af's hand rule (render._ink_clash) does not. af's is taken.
    stack_tolerance: float = 1e-6
    leader_end: Literal["mid", "node"] = "mid"
    weights: Weights = field(default_factory=Weights)
    schedule: Schedule = field(default_factory=Schedule)


@dataclass
class Cands:
    """One label's candidates in the search's order, as arrays: the text box, the attach point
    the leader is costed to, the cost of the label alone (``base``) and its tier (0 pinned,
    1-2 sampled, 3 swept or last resort)."""

    x0: np.ndarray
    y0: np.ndarray
    x1: np.ndarray
    y1: np.ndarray
    cx: np.ndarray
    cy: np.ndarray
    base: np.ndarray
    tier: np.ndarray

    def __len__(self) -> int:
        return len(self.base)

    def box(self, c: int) -> Box:
        return (float(self.x0[c]), float(self.y0[c]), float(self.x1[c]), float(self.y1[c]))

    @staticmethod
    def of(rows: list[tuple[float, float, float, float, float, float, float, int]]) -> Cands:
        """From (x0, y0, x1, y1, cx, cy, base, tier) rows, in their order."""
        a = np.array([r[:7] for r in rows], dtype=np.float64).reshape(-1, 7)
        tiers = np.array([r[7] for r in rows], dtype=np.int8)
        col = [a[:, k].copy() for k in range(7)]
        return Cands(col[0], col[1], col[2], col[3], col[4], col[5], col[6], tiers)
