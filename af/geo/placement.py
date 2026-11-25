"""Where a geo page's dots go: each location's own pile, or one shared lattice for the page (Q280).

Two named rules, chosen per report slot in the round's figure-style file (:mod:`af.geo.style`):

- **piles** (what af drew before Q280, and still the default): each location's dots are packed
  around its own point (:data:`af.geo.render.PACKINGS`). Neighbouring piles overlap, and at this
  scale (a dot is 3.6 degrees wide) a crowded region hides whole locations behind each other.
- **lattice** (Sarah, Q280, 25 Nov 2026: the uncapped shared lattice): the page, not the location,
  carries one hexagonal lattice. Each location first takes one cell nearest its own point
  (``reserve_point``), then every remaining dot takes the free cell nearest its place, the
  nearest offer on the whole page winning first. No two dots are ever nearer than the pitch, so
  at the ring step no dot is covered more than a dot inside today's own pile is, and no
  location can be drawn under another. The price is position: a dot can end up further from
  its place, which :func:`displacement` measures so every page says what it paid.

``max_move`` caps how far any dot may move; a dot with no free cell within it stays where its
pile would put it. Unset is uncapped (the decided rule); it exists so trying the cap is one
config line.

The allocation is the Q280 bench's (``placement.py`` in the 10-serology notes), ported as is so
the pages it measured are the pages this draws: ties break on the location, then the cell, so
the answer is the same every run (design rule 8).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

PILES = "piles"
LATTICE = "lattice"
RULES = (PILES, LATTICE)
NEAREST_OFFER_FIRST = "nearest-offer-first"
ORDERS = (NEAREST_OFFER_FIRST,)

#: How many of a dot's nearest cells it offers for before the page-wide sort; a dot whose offers
#: are all taken searches outward from its place, up to :data:`REACH`. The bench's values: they
#: bound the search and decide which dot of a tie is placed by offer and which by search.
CANDIDATES = 160
REACH = 200.0  # figure points
EARTH_RADIUS_KM = 6371.0088


class PlacementError(RuntimeError):
    """A dot the lattice could not place: never dropped silently (design rule 3)."""


@dataclass(frozen=True)
class Placement:
    """The placement rule a slot is drawn with, and the lattice's parameters when it is one."""

    rule: str = PILES
    pitch: float | None = None  # figure points between neighbouring cells
    order: str | None = None  # how cells are handed out
    reserve_point: bool | None = None  # one cell per location before any pile spreads
    max_move: float | None = None  # figure points; None is uncapped

    def to_json(self) -> dict[str, Any]:
        if self.rule == PILES:
            return {"rule": PILES}
        return {
            "rule": self.rule,
            "pitch": self.pitch,
            "order": self.order,
            "reserve_point": self.reserve_point,
            "max_move": self.max_move,
        }


@dataclass(frozen=True)
class Dot:
    """One dot in figure points: where its pile puts it, and the place it stands for."""

    location: str
    x: float
    y: float
    hx: float
    hy: float


Cell = tuple[int, int]


def cell_centre(i: int, j: int, pitch: float) -> tuple[float, float]:
    """Lattice cell ``(i, j)`` on the page: rows offset by half a pitch, hex-packed. Anchored to
    the page, not to a location, so neighbouring locations draw from the same cells."""
    return (i * pitch + (pitch / 2 if j % 2 else 0.0), j * pitch * math.sqrt(3) / 2)


def _cell_offsets(pitch: float, reach: float) -> list[tuple[int, int, float]]:
    """Cell-index offsets out to ``reach``, nearest first, each with its distance."""
    rows = int(reach / (pitch * math.sqrt(3) / 2)) + 2
    cols = int(reach / pitch) + 2
    out = []
    for j in range(-rows, rows + 1):
        for i in range(-cols, cols + 1):
            dx, dy = cell_centre(i, j, pitch)
            distance = math.hypot(dx, dy)
            if distance <= reach:
                out.append((i, j, distance))
    return sorted(out, key=lambda c: (round(c[2], 6), c[1], c[0]))


def _nearest_cell(x: float, y: float, pitch: float) -> Cell:
    """The lattice cell nearest ``(x, y)``."""
    row_height = pitch * math.sqrt(3) / 2
    best: tuple[float, int, int] | None = None
    for j in (math.floor(y / row_height), math.ceil(y / row_height)):
        i = round((x - (pitch / 2 if j % 2 else 0.0)) / pitch)
        gap = math.dist((x, y), cell_centre(i, j, pitch))
        if best is None or gap < best[0]:
            best = (gap, i, j)
    assert best is not None
    return best[1], best[2]


def _nearest_free(
    home: tuple[float, float],
    anchor: Cell,
    offsets: Sequence[tuple[int, int, float]],
    pitch: float,
    free: Callable[[Cell], bool],
) -> Cell | None:
    """The free cell nearest ``home``. Offsets are ordered from ``anchor``, which sits up to a
    covering radius off ``home``, so the search keeps the best true distance and stops only
    once no unsearched cell could beat it."""
    best: tuple[float, int, int] | None = None
    for di, dj, lower in offsets:
        if best is not None and lower > best[0] + pitch:
            break
        cell = (anchor[0] + di, anchor[1] + dj)
        if not free(cell):
            continue
        gap = math.dist(home, cell_centre(cell[0], cell[1], pitch))
        if best is None or gap < best[0]:
            best = (gap, cell[0], cell[1])
    return None if best is None else (best[1], best[2])


def lattice_positions(
    dots: Sequence[Dot],
    pitch: float,
    *,
    reserve_point: bool,
    max_move: float | None,
    inside: Callable[[float, float], bool],
) -> tuple[list[tuple[float, float]], int]:
    """Each dot's position on the shared lattice, in the order given; and how many dots the cap
    kept where their pile put them.

    ``inside(x, y)``: whether a dot centred there is drawn whole on the page; a cell that is not
    is never handed out, so the frame cuts no dot.
    """
    if pitch <= 0:
        raise PlacementError(f"lattice pitch {pitch}: must be positive")
    offsets = _cell_offsets(pitch, REACH)
    near = offsets[:CANDIDATES]
    anchors = {i: _nearest_cell(d.hx, d.hy, pitch) for i, d in enumerate(dots)}

    def gap(i: int, cell: Cell) -> float:
        return math.dist((dots[i].hx, dots[i].hy), cell_centre(cell[0], cell[1], pitch))

    def allocate(wanted: list[int], fixed: list[Dot]) -> tuple[dict[int, Cell], list[int]]:
        taken: set[Cell] = set()

        def free(cell: Cell) -> bool:
            if cell in taken:
                return False
            centre = cell_centre(cell[0], cell[1], pitch)
            if not inside(*centre):
                return False
            return all(math.dist(centre, (d.x, d.y)) >= pitch for d in fixed)

        def within_cap(i: int, cell: Cell) -> bool:
            return max_move is None or gap(i, cell) <= max_move

        at: dict[int, Cell] = {}

        def hand_out(bidders: list[int]) -> None:
            bids = [
                (gap(i, cell), i, cell)
                for i in bidders
                for cell in [(anchors[i][0] + di, anchors[i][1] + dj) for di, dj, _ in near]
            ]
            # nearest offer first; ties on the location, then the cell (stable for the rest)
            bids.sort(key=lambda b: (round(b[0], 6), dots[b[1]].location, b[2]))
            for _, i, cell in bids:
                if i in at or not free(cell) or not within_cap(i, cell):
                    continue
                at[i] = cell
                taken.add(cell)

        if reserve_point:
            # one cell per location first; its dots share a home, so one offers for all
            first: dict[str, int] = {}
            for i in wanted:
                first.setdefault(dots[i].location, i)
            hand_out(list(first.values()))
        hand_out([i for i in wanted if i not in at])
        shut_out = []
        for i in wanted:  # every offer lost: search further out, inside the cap
            if i in at:
                continue
            cell = _nearest_free((dots[i].hx, dots[i].hy), anchors[i], offsets, pitch, free)
            if cell is None or not within_cap(i, cell):
                shut_out.append(i)
                continue
            at[i] = cell
            taken.add(cell)
        return at, shut_out

    # A dot the cap shuts out stays in its pile and then holds that space against the lattice,
    # which can shut out another: settle it (the kept set only grows).
    kept: set[int] = set()
    at: dict[int, Cell] = {}
    for _ in range(4):
        at, shut_out = allocate(
            [i for i in range(len(dots)) if i not in kept], [dots[i] for i in sorted(kept)]
        )
        if not shut_out:
            break
        if max_move is None:
            raise PlacementError(
                f"{len(shut_out)} dot(s) found no free lattice cell within {REACH} points"
            )
        kept |= set(shut_out)
    else:
        if shut_out:
            raise PlacementError(f"the {max_move} point cap did not settle: {len(shut_out)} dots")
    out = []
    for i, dot in enumerate(dots):
        out.append((dot.x, dot.y) if i in kept else cell_centre(*at[i], pitch))
    return out, len(kept)


def great_circle_km(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Kilometres between two (longitude, latitude) points."""
    lon1, lat1, lon2, lat2 = (math.radians(v) for v in (a[0], a[1], b[0], b[1]))
    d = (
        math.sin((lat2 - lat1) / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    )
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(d))


def displacement(
    dots: Sequence[Dot],
    latitudes: dict[str, float],
    lon_per_point: float,
    lat_per_point: float,
    radius: float,
) -> dict[str, Any]:
    """What a page's placement costs in position, whichever rule drew it.

    A dot's move is measured on the globe from its place to where it is drawn. A location is
    *off its point* when none of its dots touches the point; a dot is *in another's place* when
    another location's point is nearer it than its own, which is how a reader attributes it.
    Neither is zero for piles: a big pile already spills past its neighbours.
    """
    if not dots:
        return {"locations": 0, "locations_off_point": 0, "dots_in_another_place": 0,
                "move_km": {"median": 0.0, "p90": 0.0, "max": 0.0}}  # fmt: skip
    places = list({(d.location, d.hx, d.hy): None for d in dots})
    moves = []
    nearest: dict[str, float] = {}
    elsewhere = 0
    for d in dots:
        lat = latitudes[d.location]
        moves.append(
            great_circle_km(
                (d.hx * lon_per_point, lat),
                (d.x * lon_per_point, lat + (d.y - d.hy) * lat_per_point),
            )
        )
        closest = min(places, key=lambda p: math.dist((d.x, d.y), (p[1], p[2])))
        elsewhere += closest[0] != d.location
        away = math.dist((d.x, d.y), (d.hx, d.hy)) - radius
        nearest[d.location] = min(nearest.get(d.location, math.inf), away)
    array = np.array(moves)
    return {
        "locations": len(nearest),
        "locations_off_point": sum(1 for away in nearest.values() if away > 0.0),
        "dots_in_another_place": elsewhere,
        "move_km": {
            "median": round(float(np.median(array)), 1),
            "p90": round(float(np.percentile(array, 90)), 1),
            "max": round(float(array.max()), 1),
        },
    }
