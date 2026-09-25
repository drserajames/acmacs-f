"""Automatic framing of a map: a square of fixed size, placed so nothing important is hidden.

Why: every lab folder carries a hand-written ``viewport()`` that has to be re-derived when points
move, and a hand frame silently hides points (on the Sep 2026 round: 5 antigens from the last six
months and 14 from the last twelve were off the frame or under the legend, across 18 maps). With the
same frame sizes, the search below hid none and one respectively.

The size is fixed per subtype/assay in config, so maps of one subtype share a scale (Sarah, 25 Sep
2026). Only the position is chosen. Frames are stored in absolute displayed coordinates, never
relative to the layout's hull (ae's convention, under which one new far-away antigen moves every
frame).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]
Mask = NDArray[np.bool_]


class FrameError(ValueError):
    """The points that must be shown cannot all be shown in a frame of the configured size."""


@dataclass(frozen=True)
class Frame:
    """Square frame in displayed map coordinates (y grows downward): top-left corner and side."""

    x: float
    y: float
    size: float

    def page(self, xy: Array) -> Array:
        """Positions as fractions of the page, (0, 0) top-left, (1, 1) bottom-right."""
        return (xy - [self.x, self.y]) / self.size


@dataclass(frozen=True)
class Box:
    """Page furniture (legend, title) as page fractions: left, top, right, bottom."""

    name: str
    left: float
    top: float
    right: float
    bottom: float


@dataclass(frozen=True)
class Priority:
    """A group of points and how much hiding it matters; earlier priorities dominate later ones."""

    name: str
    mask: Mask


@dataclass(frozen=True)
class FrameChoice:
    frame: Frame
    hidden: dict[str, int]  # per priority: points off the page or under furniture


def hidden_mask(page: Array, furniture: tuple[Box, ...]) -> Mask:
    """Points off the page or covered by a furniture box. NaN positions count as not hidden:
    points without coordinates are reported elsewhere, not here."""
    with np.errstate(invalid="ignore"):
        x, y = page[:, 0], page[:, 1]
        hidden = (x < 0) | (x > 1) | (y < 0) | (y > 1)
        for box in furniture:
            hidden |= (x >= box.left) & (x <= box.right) & (y >= box.top) & (y <= box.bottom)
    return hidden


def choose_frame(
    xy: Array,
    priorities: tuple[Priority, ...],
    *,
    size: float,
    furniture: tuple[Box, ...],
    step: float = 0.1,
) -> FrameChoice:
    """Place a ``size`` x ``size`` frame over ``xy`` (displayed coordinates of drawn points).

    Candidate corners lie on a ``step`` grid. They are compared lexicographically: hidden count of
    each priority in order, then the distance between the frame centre and the centre of the first
    priority's points (so the recent antigens sit in the middle when nothing else decides). The
    first priority is required: if even the best frame hides one of its points, :class:`FrameError`.
    """
    if not priorities:
        raise ValueError("choose_frame needs at least one priority group")
    for p in priorities:
        if p.mask.shape != (len(xy),):
            raise ValueError(f"priority {p.name!r}: mask length {p.mask.shape} != {len(xy)} points")
    ok = ~np.isnan(xy).any(axis=1)
    if not ok.any():
        raise FrameError("no drawn point has coordinates")
    pts = xy[ok]
    masks = [p.mask[ok] for p in priorities]
    focus = pts[masks[0]] if masks[0].any() else pts
    centre = (focus.min(axis=0) + focus.max(axis=0)) / 2
    lo = np.minimum(pts.min(axis=0), pts.max(axis=0) - size) - 1.0
    hi = np.maximum(pts.min(axis=0), pts.max(axis=0) - size) + 1.0
    best: tuple[tuple[float, ...], Frame] | None = None
    for x in np.arange(lo[0], hi[0] + step / 2, step):
        for y in np.arange(lo[1], hi[1] + step / 2, step):
            frame = Frame(round(float(x), 6), round(float(y), 6), size)
            hidden = hidden_mask(frame.page(pts), furniture)
            offset = float(np.hypot(*(centre - [x + size / 2, y + size / 2])))
            cost = (*(float((hidden & m).sum()) for m in masks), offset)
            if best is None or cost < best[0]:
                best = (cost, frame)
    assert best is not None
    cost, frame = best
    counts = {p.name: int(c) for p, c in zip(priorities, cost, strict=False)}
    if counts[priorities[0].name]:
        raise FrameError(
            f"{counts[priorities[0].name]} point(s) of {priorities[0].name!r} cannot be shown in a "
            f"{size} x {size} frame with this furniture; enlarge the configured size"
        )
    return FrameChoice(frame, counts)
