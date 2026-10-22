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


def hidden_mask(page: Array, furniture: tuple[Box, ...], radii: Array | float = 0.0) -> Mask:
    """Points off the page or covered by a furniture box. NaN positions count as not hidden:
    points without coordinates are reported elsewhere, not here.

    ``radii`` (page fractions, one per point or one for all) is how far each marker reaches
    from its centre: a point whose marker crosses the page edge or reaches into a box counts as
    hidden, as the PDF shows it cut or under the title. Testing centres alone put coloured
    markers over the title (R4 review, 18 Oct 2026). The marker is taken as the square round
    it, so a box's corner hides a little early; with 0 only centres are tested.
    """
    with np.errstate(invalid="ignore"):
        x, y = page[:, 0], page[:, 1]
        r = np.broadcast_to(np.asarray(radii, dtype=float), x.shape)
        hidden = (x - r < 0) | (x + r > 1) | (y - r < 0) | (y + r > 1)
        for box in furniture:
            hidden |= (
                (x + r >= box.left)
                & (x - r <= box.right)
                & (y + r >= box.top)
                & (y - r <= box.bottom)
            )
    return hidden


@dataclass(frozen=True)
class View:
    """One page a frame must serve: its priority groups and its furniture. A map's windows
    share one frame (as shipped: the all, 12m and 6m pages of a map show the same square), but
    each window greys its own points and has its own title and legend."""

    priorities: tuple[Priority, ...]
    furniture: tuple[Box, ...]


def choose_frame(
    xy: Array,
    priorities: tuple[Priority, ...],
    *,
    size: float,
    furniture: tuple[Box, ...],
    step: float = 0.1,
    radii: Array | float = 0.0,
    also: tuple[View, ...] = (),
) -> FrameChoice:
    """Place a ``size`` x ``size`` frame over ``xy`` (displayed coordinates of drawn points).

    Candidate corners lie on a ``step`` grid. They are compared lexicographically: hidden count of
    each priority in order, then the distance between the frame centre and the centre of the first
    priority's points (so the recent antigens sit in the middle when nothing else decides). The
    first priority is required: if even the best frame hides one of its points, :class:`FrameError`.
    ``radii``: each point's marker extent as a page fraction (:func:`hidden_mask`). The required
    group's CENTRES must all be visible; after that, frames are compared by whole markers, so a
    marker cut by the edge or reaching under the title counts as hidden in every group (``hidden``
    reports those counts). Requiring whole markers outright made a frame impossible where every
    centre fits (h1-vidrl, R4, 15-unit frame).

    ``also``: further pages the same frame must serve, each with as many priority groups, in the
    same ranks. The required group must then be visible on every page; after it, the pages are
    compared in turn, every rank of the first page before the next page's. Choosing a frame per
    window moved h1-cnic's frame 3 units between its all and 12m pages (R5 review, 19 Oct 2026).
    Summing each rank over the pages instead let the all page, where every antigen is in the
    window, outweigh the windows' own: h1-cnic's shared frame was the all page's, 2.9 units above
    its 12m and 6m frames (R6 review, 20 Oct 2026). ``hidden`` reports the first page.
    """
    views = (View(priorities, furniture), *also)
    if not priorities:
        raise ValueError("choose_frame needs at least one priority group")
    for v in views:
        if len(v.priorities) != len(priorities):
            raise ValueError(
                f"every page needs {len(priorities)} priority groups, got {len(v.priorities)}"
            )
        for p in v.priorities:
            if p.mask.shape != (len(xy),):
                raise ValueError(
                    f"priority {p.name!r}: mask length {p.mask.shape} != {len(xy)} points"
                )
    ok = rows_with_coordinates(xy)
    if not ok.any():
        raise FrameError("no drawn point has coordinates")
    pts = xy[ok]
    rad = np.broadcast_to(np.asarray(radii, dtype=float), (len(xy),))[ok]
    masks = [[p.mask[ok] for p in v.priorities] for v in views]
    # Candidates are searched in page orientation (x right, y DOWN, so ``-y``), stepping the
    # frame's top-left corner as drawn, so the search does not depend on which way map y grows.
    down = pts * [1.0, -1.0]
    first = masks[0][0]
    focus = down[first] if first.any() else down
    centre = (focus.min(axis=0) + focus.max(axis=0)) / 2
    lo = np.minimum(down.min(axis=0), down.max(axis=0) - size) - 1.0
    hi = np.maximum(down.min(axis=0), down.max(axis=0) - size) + 1.0
    best: tuple[tuple[float, ...], Frame, list[float]] | None = None
    for x in np.arange(lo[0], hi[0] + step / 2, step):
        for y in np.arange(lo[1], hi[1] + step / 2, step):
            frame = Frame(round(float(x), 6), -round(float(y), 6), size)
            page = frame.page(pts)
            required = 0.0
            ranks: list[float] = []  # page by page, each page's ranks in order
            for v, ms in zip(views, masks, strict=True):
                required += float((hidden_mask(page, v.furniture) & ms[0]).sum())
                hidden = hidden_mask(page, v.furniture, rad)
                ranks += [float((hidden & m).sum()) for m in ms]
            own = ranks[: len(priorities)]
            offset = float(np.hypot(*(centre - [x + size / 2, y + size / 2])))
            cost = (required, *ranks, offset)
            if best is None or cost < best[0]:
                best = (cost, frame, own)
    assert best is not None
    cost, frame, own = best
    report = {p.name: int(c) for p, c in zip(priorities, own, strict=True)}
    if cost[0]:
        raise FrameError(
            f"{int(cost[0])} point(s) of {priorities[0].name!r} cannot be shown in a "
            f"{size} x {size} frame with this furniture; enlarge the configured size"
        )
    return FrameChoice(frame, report)


def hidden_counts(
    frame: Frame,
    xy: Array,
    priorities: tuple[Priority, ...],
    furniture: tuple[Box, ...],
    radii: Array | float = 0.0,
    *,
    strict: bool = True,
) -> FrameChoice:
    """A given frame's hidden counts per priority, as :func:`choose_frame` reports them (whole
    markers), for a page drawn in a frame chosen for several (:class:`View`). Raises
    :class:`FrameError` if a centre of the first, required group is hidden, unless not
    ``strict``: a figure drawn in another page's frame with its own furniture (an addendum's
    taller title) counts what its furniture covers instead."""
    page = frame.page(xy)
    if strict and (hidden_mask(page, furniture) & priorities[0].mask).any():
        raise FrameError(f"the given frame hides points of {priorities[0].name!r}")
    hidden = hidden_mask(page, furniture, radii)
    return FrameChoice(frame, {p.name: int((hidden & p.mask).sum()) for p in priorities})
