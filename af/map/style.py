"""Style a map: what each point looks like, which points are greyed or hidden, and the legend.

Why a separate, data-only step: today a map's look is spread over inherited plot-spec layers
(``-reset``, ``-clades``, ``-o12m-grey``, ``-vaccines``...) resolved at render time, so nobody can
say what a PDF shows without re-running the renderer. Here styling produces a :class:`Scene`, a
plain list of points with their final colour, visibility and reason. The renderer draws it and the
I7 JSON is written from it, so the picture and its description cannot disagree.

Colours come from a user colour scheme (workstream 4). A scheme is an ordered list of rows; a point
is painted by the *last* row whose labels it carries all of, as today's clade layers paint (later
wins). The legend counts every shown antigen painted by a row, the same as today's report maps.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]
PointKind = Literal["antigen", "serum"]


@dataclass(frozen=True)
class ColourRow:
    """One legend row: points carrying every label in ``labels`` are painted ``colour``."""

    legend: str
    colour: str
    labels: frozenset[str]


@dataclass(frozen=True)
class ColourScheme:
    name: str
    rows: tuple[ColourRow, ...]

    def paint(self, labels: frozenset[str]) -> ColourRow | None:
        """Last matching row wins (painting order); None = unpainted."""
        hit = None
        for row in self.rows:
            if row.labels <= labels:
                hit = row
        return hit


@dataclass(frozen=True)
class Window:
    """Time window of a map: antigens isolated before ``since`` are greyed. ``since`` None = all."""

    name: str
    since: dt.date | None


@dataclass(frozen=True)
class PointIn:
    """What styling needs to know about one point (built by the caller from the chart)."""

    id: str
    name: str
    kind: PointKind
    xy: tuple[float, float] | None  # displayed coordinates; None = no coordinates
    labels: frozenset[str] = frozenset()  # clade / aa labels, for the colour scheme
    date: dt.date | None = None
    passage_class: str | None = None
    reference: bool = False
    serum_id: str | None = None
    hide: str | None = None  # a named hide rule that selected this point


@dataclass(frozen=True)
class ScenePoint:
    id: str
    name: str
    kind: PointKind
    xy: tuple[float, float] | None
    shown: bool
    hidden_reason: str | None
    legend: str | None
    colour: str | None
    greyed: bool
    date: dt.date | None
    passage_class: str | None
    reference: bool
    serum_id: str | None
    vaccine: str | None = None  # label text if marked as a vaccine


@dataclass
class Scene:
    title: str
    window: Window
    scheme: str
    points: list[ScenePoint]
    legend: list[tuple[str, str, int]]  # (row text, colour, count)
    undated_antigens: int = 0
    notes: list[str] = field(default_factory=list)

    def xy(self) -> Array:
        return np.array([p.xy if p.xy is not None else (np.nan, np.nan) for p in self.points])


def style_points(
    points: Sequence[PointIn],
    scheme: ColourScheme,
    window: Window,
    *,
    title: str,
    vaccines: dict[str, str] | None = None,
) -> Scene:
    """Build the :class:`Scene` for one map and window.

    ``vaccines`` maps point id to label text. A vaccine is never greyed (it is a reference point
    for the reader, whatever its date), matching today's maps. An antigen without an isolation date
    is greyed in a windowed map (it cannot be shown to be recent) and counted in
    ``Scene.undated_antigens``.
    """
    vaccines = vaccines or {}
    unknown = set(vaccines) - {p.id for p in points}
    if unknown:
        raise ValueError(f"vaccine ids not among the points: {sorted(unknown)}")
    out: list[ScenePoint] = []
    counts = {row.legend: 0 for row in scheme.rows}
    undated = 0
    for p in points:
        row = scheme.paint(p.labels) if p.kind == "antigen" else None
        shown = p.xy is not None and p.hide is None
        reason = None if shown else ("no_coordinates" if p.xy is None else f"override:{p.hide}")
        greyed = False
        if p.kind == "antigen" and window.since is not None and p.id not in vaccines:
            if p.date is None:
                greyed = True
                undated += 1
            else:
                greyed = p.date < window.since
        if shown and row is not None and p.kind == "antigen":
            counts[row.legend] += 1
        out.append(
            ScenePoint(
                id=p.id,
                name=p.name,
                kind=p.kind,
                xy=p.xy,
                shown=shown,
                hidden_reason=reason,
                legend=row.legend if row else None,
                colour=row.colour if row else None,
                greyed=greyed,
                date=p.date,
                passage_class=p.passage_class,
                reference=p.reference,
                serum_id=p.serum_id,
                vaccine=vaccines.get(p.id),
            )
        )
    legend = [(row.legend, row.colour, counts[row.legend]) for row in scheme.rows]
    return Scene(title, window, scheme.name, out, legend, undated)
