"""Automatic framing of a map: a square of fixed size, placed so nothing important is hidden.

Why: every lab folder carries a hand-written ``viewport()`` that has to be re-derived when points
move, and a hand frame silently hides points (on the Sep 2026 round: 5 antigens from the last six
months and 14 from the last twelve were off the frame or under the legend, across 18 maps). With the
same frame sizes, the search below hid none and one respectively.

The size is fixed per subtype/assay in config, so maps of one subtype share a scale (Sarah, 25 Sep
2026). Only the position is chosen. Frames are stored in absolute displayed coordinates, never
relative to the layout's hull (ae's convention, under which one new far-away antigen moves every
frame).

Displayed map coordinates have y increasing UPWARDS, as R/Racmacs draws maps (Sarah, Q105, 2 Oct
2026: "y-up everywhere"). The page has y increasing downwards, as pages do. :meth:`Frame.page` is
the one place one becomes the other; nothing else flips.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from af.map.orient import rows_with_coordinates

Array = NDArray[np.float64]
Mask = NDArray[np.bool_]


def from_y_down(xy: Array) -> Array:
    """Coordinates of a picture drawn with y DOWN (ae and kateri draw ``.ace`` maps that way) as
    af's displayed coordinates (y up), so the picture looks the same: ``(x, y) -> (x, -y)``.

    For a reference that is a drawing, such as the previous round's map as it was shipped. An
    ``.ace`` layout itself is never flipped: af draws it y up, as R does.
    """
    return np.asarray(xy, dtype=float) * [1.0, -1.0]


class FrameError(ValueError):
    """The points that must be shown cannot all be shown in a frame of the configured size."""


@dataclass(frozen=True)
class Frame:
    """A frame in displayed map coordinates (y up): its top-left corner as drawn, which is
    (smallest x, LARGEST y), its width ``size`` and, for a rectangular frame, its ``height``.

    The report's frames are square (``height`` None). A rectangular frame (Sarah, 2 Oct: her R
    figures used data-derived limits) is opt-in, for callers that choose their own frames.
    """

    x: float
    y: float
    size: float
    height: float | None = None

    def __post_init__(self) -> None:
        if self.size <= 0 or (self.height is not None and self.height <= 0):
            raise ValueError(f"frame sides must be positive, got {self.size} x {self.height}")

    @property
    def tall(self) -> float:
        """The frame's height in map units (its width for a square frame)."""
        return self.size if self.height is None else self.height

    @property
    def aspect(self) -> float:
        """Height over width: 1.0 for a square frame."""
        return 1.0 if self.height is None else self.height / self.size

    def page(self, xy: Array) -> Array:
        """Positions as fractions of the page, (0, 0) top-left, (1, 1) bottom-right: x over
        the width, y over the height.

        The one conversion from displayed map coordinates (y up) to the page (y down).
        """
        return np.column_stack([(xy[:, 0] - self.x) / self.size, (self.y - xy[:, 1]) / self.tall])


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
    ok = rows_with_coordinates(xy)
    if not ok.any():
        raise FrameError("no drawn point has coordinates")
    pts = xy[ok]
    masks = [p.mask[ok] for p in priorities]
    # Candidates are searched in page orientation (x right, y DOWN, so ``-y``), stepping the
    # frame's top-left corner as drawn, so the search does not depend on which way map y grows.
    down = pts * [1.0, -1.0]
    focus = down[masks[0]] if masks[0].any() else down
    centre = (focus.min(axis=0) + focus.max(axis=0)) / 2
    lo = np.minimum(down.min(axis=0), down.max(axis=0) - size) - 1.0
    hi = np.maximum(down.min(axis=0), down.max(axis=0) - size) + 1.0
    best: tuple[tuple[float, ...], Frame] | None = None
    for x in np.arange(lo[0], hi[0] + step / 2, step):
        for y in np.arange(lo[1], hi[1] + step / 2, step):
            frame = Frame(round(float(x), 6), -round(float(y), 6), size)
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
