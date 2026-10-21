"""Serum circles on a map page, drawn as the circle addenda drew them.

What a circle is (its homologous antigens and radii) is :mod:`af.chart.serum_circles`'. This
module decides only how a record is drawn, as ae's ``ae.report.multiple_circles`` drew the
shipped circle addenda:

- the circle is filled with the map colour of the antigen its radius comes from: under the
  best-match rule (Sarah, Q228) the one titre that gives the circle and the coverage colouring.
  ae coloured by the first homologous antigen in chart order with a theoretical radius, which
  can be another preparation of the virus. An unpainted antigen gives ae's grey;
- the circled serum is drawn black, with a 3-point outline in that colour, at 1.3 times a
  serum's size (ae: 26 points on the 800-point page, af's serum is 20);
- a serum with no radius gets ae's dashed circle of radius fold.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from af.chart.serum_circles import SerumCircle
from af.map.render import CircleMark, PointStyle
from af.map.vaccines import display_serum_id
from af.tables.labs import Lab

RADII = ("theoretical", "empirical")
NO_COLOUR = "#808080"  # ae multiple_circles' GREY: a homologous antigen with no map colour


class CircleError(ValueError):
    pass


@dataclass(frozen=True)
class CircleLook:
    """How circles and their sera are drawn. The defaults are the circle addenda's (ae)."""

    radius: str = "theoretical"  # the circle addenda draw theoretical radii; sig pages empirical
    fill_alpha: float = 0.2  # ae: 0x33 over 0xFF
    outline_width: float = 1.0
    serum_outline_width: float = 3.0
    serum_scale: float = 1.3

    def __post_init__(self) -> None:
        if self.radius not in RADII:
            raise CircleError(f"circle radius {self.radius!r}: not one of {RADII}")


DEFAULT_CIRCLES = CircleLook()


def serum_point(record: SerumCircle) -> str:
    """The map's id of the record's serum point (af.map.figure.chart_points)."""
    return f"sr{record.serum}"


def circle_colour(
    record: SerumCircle, painted: Mapping[str, str | None], radius: str = "theoretical"
) -> str:
    """A serum circle's colour: the map colour of the antigen its ``radius`` comes from.
    ``painted`` is each antigen point's map colour."""
    antigen = getattr(record.circle, f"{radius}_antigen")
    if antigen is None:
        return NO_COLOUR
    return painted.get(f"ag{antigen}") or NO_COLOUR


def serum_circle_marks(
    records: Sequence[SerumCircle],
    painted: Mapping[str, str | None],
    look: CircleLook = DEFAULT_CIRCLES,
) -> tuple[list[CircleMark], dict[str, PointStyle]]:
    """The circles and the restyled serum points for ``records`` (the sera to circle).

    A serum with no coordinates is an error: there is nothing to centre its circle on, and
    dropping it silently would drop it from the page."""
    marks: list[CircleMark] = []
    styles: dict[str, PointStyle] = {}
    for rec in records:
        point = serum_point(rec)
        if rec.xy is None:
            raise CircleError(f"{rec.designation}: the serum has no coordinates on this map")
        colour = circle_colour(rec, painted, look.radius)
        radius = getattr(rec.circle, look.radius)
        marks.append(
            CircleMark(
                point,
                radius if radius is not None else rec.fold,
                colour,
                fill_alpha=look.fill_alpha,
                outline_width=look.outline_width,
                dashed=radius is None,
            )
        )
        styles[point] = PointStyle(
            fill="black",
            outline=colour,
            outline_width=look.serum_outline_width,
            scale=look.serum_scale,
            on_top=True,
        )
    return marks, styles


def names_lines(records: Sequence[SerumCircle], lab: Lab) -> list[str]:
    """The circled sera listed under the title, as the shipped addenda listed them (Sarah, Q247
    a): name, serum id as maps print it (``lab``'s rule, Q123) and passage, the passage with its
    harvest date where the lab gives one (CDC's are the only dated ones). The last serum first,
    as ae's list ran."""
    return [
        " ".join(x for x in (r.name, display_serum_id(r.serum_id, lab), r.passage) if x)
        for r in reversed(records)
    ]
