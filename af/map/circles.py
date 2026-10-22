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
from typing import Any

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


# ---------------------------------------------------------------- the serum-coverage look

#: The coverage maps' circle, by the serum's passage class (ae serum_coverage_circle_style):
#: the outline in the colour, the fill the same colour at 0x18 of 0xFF.
PASSAGE_CIRCLE = {"cell": "#0000ff", "egg": "#ff0000", "reassortant": "#ffa500"}
COVERAGE_FILL_ALPHA = 0x18 / 0xFF
WITHIN_OUTLINE = "#ffc0cb"  # pink: titrated within fold of the homologous titre
OUTSIDE_OUTLINE = "#000000"
COVERAGE_OUTLINE_WIDTH = 3.0
COVERAGE_SERUM_SCALE = 1.8  # ae's 36 points over af's 20
PALE_FACTOR = 0.4  # kateri's :pale


def pale(colour: str) -> str:
    """kateri's ":pale" (ColorAndModifier, factor 0.4): saturation down, value up; a grey is
    only lightened, and black becomes mid grey. "none" and "transparent" stay as they are."""
    import colorsys

    from matplotlib.colors import to_hex, to_rgb

    if colour in ("none", "transparent"):
        return colour
    h, s, v = colorsys.rgb_to_hsv(*to_rgb(colour))
    if s > 0.0:
        s, v = s * PALE_FACTOR, v + (1.0 - v) * (1.0 - PALE_FACTOR)
    else:
        v = 0.5 if v == 0.0 else min(1.0, v / PALE_FACTOR)
    return str(to_hex(colorsys.hsv_to_rgb(h, s, v)))


def coverage_marks(
    record: SerumCircle, scene: Any, radius: str = "empirical"
) -> tuple[list[CircleMark], dict[str, PointStyle]]:
    """The serum-coverage look of one serum on its map's ``scene`` (ae's sc-* maps): every
    point pale; the antigens titrated against the serum in full colour, outlined pink within
    fold of the homologous titre and black outside it, raised (outside over within); the serum
    a black square on top; its circle in its passage class's colour (dashed at radius fold when
    it has none)."""
    from af.map.render import point_colours

    if radius not in RADII:
        raise CircleError(f"circle radius {radius!r}: not one of {RADII}")
    within, outside = set(record.coverage.within), set(record.coverage.outside)
    styles: dict[str, PointStyle] = {}
    for p in scene.points:
        fill, edge = point_colours(p)
        k = int(p.id[2:]) if p.kind == "antigen" and p.id[2:].isdigit() else -1
        if k in within or k in outside:
            styles[p.id] = PointStyle(
                fill=fill,
                outline=WITHIN_OUTLINE if k in within else OUTSIDE_OUTLINE,
                outline_width=COVERAGE_OUTLINE_WIDTH,
                on_top=True,
                raise_rank=0 if k in within else 1,
            )
        else:
            styles[p.id] = PointStyle(fill=pale(fill), outline=pale(edge))
    point = serum_point(record)
    if record.xy is None:
        raise CircleError(f"{record.designation}: the serum has no coordinates on this map")
    # a square whatever the serum's passage, as ae's coverage maps marked every serum
    styles[point] = PointStyle(
        fill="black", outline="black", scale=COVERAGE_SERUM_SCALE, on_top=True, raise_rank=2,
        marker="box",
    )  # fmt: skip
    colour = PASSAGE_CIRCLE.get(record.passage_class or "", NO_COLOUR)
    r = getattr(record.circle, radius)
    circle = CircleMark(
        point,
        r if r is not None else record.fold,
        colour,
        fill_alpha=COVERAGE_FILL_ALPHA,
        outline=colour,
        dashed=r is None,
    )
    return [circle], styles
