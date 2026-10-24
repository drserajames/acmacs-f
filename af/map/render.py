"""Draw a styled map scene: to a PDF page, or onto any matplotlib Axes.

The renderer knows shapes, colours and coordinates, never what a point is: each scene point
carries its marker shape ("circle", "egg", "box", "uglyegg"), chosen when the scene is built.
The I7 description of a figure is written by :mod:`af.map.i7`.

Look: similar to today's report maps (DECISIONS: "similar look, a little flexibility"): square
page, light 1-unit grid, open serum markers, greyed points drawn first and small-looking, legend
box bottom-left with counts, bold title top-left, marked points enlarged with labels.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from af.map.labels import Placed
from af.map.orient import rows_with_coordinates
from af.map.style import MARKERS, Marker, Scene, ScenePoint
from af.map.viewport import Box, Frame

GREY = "#cccccc"  # an antigen with no colour (unpainted, or unsequenced)
GREY_WINDOW = "#bebebe"  # an antigen outside the map's window, greyed by it (never a vaccine mark)
OPEN = "transparent"  # no fill, GREY outline: a reference antigen with no colour
SERUM_OUTLINE = "#cccccc"  # as the shipped report maps
TITLE_PITCH = 1.2  # a title's line pitch, in title sizes (several lines: the coverage maps)


def grey_of(p: ScenePoint) -> str | None:
    """The grey a point is drawn in, or None for its own colour. As the shipped maps (kateri's
    default point specs and its window style): a darker grey for an antigen the window greys,
    a lighter one for a test antigen with no colour, and no fill (:data:`OPEN`, a GREY ring) for
    a reference antigen with no colour. A vaccine with no colour is filled, as it was marked."""
    if p.greyed:
        return GREY_WINDOW
    if p.colour is None:
        return OPEN if p.reference and not p.mark else GREY
    return None


@dataclass(frozen=True)
class Look:
    """Sizes as fractions of the page side (so the look does not depend on the PDF size)."""

    page_points: float = 800.0
    antigen_radius: float = 0.0125
    mark_scale: float = 2.0  # a marked point's size over a plain one's
    serum_half_side: float = 0.0125
    title_size: float = 25.5
    label_size: float = 30.0
    label_halo: float = 1.22  # width of the white outline round each label's glyphs
    label_gap: float = (
        2.0  # points kept clear between two labels (abutting boxes looked stacked, R4 review)
    )
    # points kept clear right of the title's last letter: 6 (R5 review) let markers 8 and 10
    # points away read as part of the title (R6 review, 20 Oct 2026)
    title_clear: float = 12.0
    caption_size: float = 7.0  # the circle addenda's list of sera under the title (ae: 7 points)
    legend_pad: float = 0.0125  # the legend box's distance from the page's left and bottom
    # The legend, in points of the 800-point page, as the shipped report draws it (style 08-07/08,
    # measured from its figures): text size and row pitch, the text's and the marker's distance
    # from the box's left edge, the marker's size and outline, the gap before the count column
    # (in units of the text size), the count column's inset from the box's right edge, and the
    # box height beyond its rows. B/Vic's legend is larger; a round sets that per subtype.
    legend_text_size: float = 12.23
    legend_row_pitch: float = 24.5
    legend_text_left: float = 43.0
    legend_marker_left: float = 17.5
    legend_marker_size: float = 15.0
    legend_marker_outline: float = 1.0
    legend_count_gap: float = 1.0
    legend_count_inset: float = 10.0
    legend_padding: float = 2.26
    # Opt-in, for other figures than the report (pyacmapcheck draws in R's style); the defaults
    # are the report's look.
    alpha: float | None = None  # every point's fill and outline opacity; None: opaque
    grid_colour: str = "#cccccc"  # 1.0 wide, as the shipped report maps


DEFAULT_LOOK = Look()
#: The Look fields a caller sets for another figure style; the I7 records those not at default.
LOOK_OPT_INS = ("alpha", "grid_colour")


def look_opt_ins(look: Look) -> dict[str, Any]:
    """The opt-in Look settings that differ from the report's, for the figure's I7."""
    return {
        f: getattr(look, f) for f in LOOK_OPT_INS if getattr(look, f) != getattr(DEFAULT_LOOK, f)
    }


@dataclass(frozen=True)
class PointStyle:
    """One point's drawing, overriding the scene's (opt-in; None keeps the scene's own).

    ``outline_width`` is in points on the report page, scaled like every other line width
    (the default is 1.0 for a point and for a box, as on the shipped report maps). ``alpha`` is
    the opacity of the point's fill and outline, over :attr:`Look.alpha`. ``scale`` multiplies
    the point's size
    (the Look's antigen radius or serum half-side, and a mark's enlargement on top), as R draws
    removed points larger. ``on_top`` draws the point over every other point (ae's raise: the
    circled sera of the circle addenda); among raised points a higher ``raise_rank`` is drawn
    later (the coverage maps raise the within antigens, then the outside ones, then the serum).
    A ``raise_rank`` without ``on_top`` raises an antigen over the other antigens but under the
    vaccine marks (the new antigens of an interim report, as ae raised them).
    """

    fill: str | None = None
    outline: str | None = None
    outline_width: float | None = None
    alpha: float | None = None
    scale: float | None = None
    on_top: bool | None = None
    raise_rank: int | None = None
    marker: Marker | None = None  # another shape than the point's own (a coverage map's serum)

    def __post_init__(self) -> None:
        if self.scale is not None and self.scale <= 0:
            raise ValueError(f"PointStyle scale must be positive, got {self.scale}")

    def recorded(self) -> dict[str, Any]:
        """The fields set, for the figure's I7 (an unset field is the scene's own drawing)."""
        return {k: v for k, v in vars(self).items() if v is not None}


@dataclass(frozen=True)
class CircleMark:
    """A circle round a point, such as a serum circle: centred on point ``point`` (its id),
    ``radius`` in map units. Drawn as ae's circle addenda draw them: over the points, under the
    labels, legend and title, a translucent fill and then a solid outline. ``dashed`` is ae's
    drawing of a serum with no radius (a circle of radius fold)."""

    point: str
    radius: float
    fill: str
    fill_alpha: float = 0.2
    outline: str = "black"
    outline_width: float = 1.0
    dashed: bool = False

    def __post_init__(self) -> None:
        if not self.radius > 0:
            raise ValueError(f"CircleMark {self.point}: radius must be positive, got {self.radius}")

    def recorded(self) -> dict[str, Any]:
        """For the figure's I7."""
        return {
            "point": self.point,
            "radius": round(self.radius, 4),
            "fill": self.fill,
            "fill_alpha": self.fill_alpha,
            "outline": self.outline,
            "outline_width": self.outline_width,
            "dashed": self.dashed,
        }


def ugly_egg_path(x: float, y: float, r: float) -> Any:
    """The "uglyegg" marker (kateri draw_on_pdf.dart, PointShape.uglyegg): a hexagon with
    vertices (0, r), (r, 0.6r), (0.8r, -0.6r), (0, -r), (-0.8r, -0.6r), (-r, 0.6r). It fills the
    same box as a "box" marker of half-width ``r``, wide end DOWN on the page."""
    from matplotlib.path import Path as MplPath

    verts = [
        (x, y + r), (x + r, y + 0.6 * r), (x + 0.8 * r, y - 0.6 * r), (x, y - r),
        (x - 0.8 * r, y - 0.6 * r), (x - r, y + 0.6 * r), (x, y + r),
    ]  # fmt: skip
    codes = [MplPath.MOVETO] + [MplPath.LINETO] * 5 + [MplPath.CLOSEPOLY]
    return MplPath(verts, codes)


def egg_path(x: float, y: float, r: float) -> Any:
    """The "egg" marker (kateri draw_on_pdf.dart, PointShape.egg): two mirrored cubic Bezier
    curves between apexes ``r`` above and below the centre, control points at (+-1.4r, 0.95r) by
    the wide end and (+-0.8r, -0.98r) by the narrow end. As tall as a circle marker is wide (2r),
    1.191 times as tall as wide, wide end DOWN on the page (page y grows downward here)."""
    from matplotlib.path import Path as MplPath

    verts = [
        (x, y + r),
        (x + 1.4 * r, y + 0.95 * r), (x + 0.8 * r, y - 0.98 * r), (x, y - r),
        (x - 0.8 * r, y - 0.98 * r), (x - 1.4 * r, y + 0.95 * r), (x, y + r),
        (x, y + r),
    ]  # fmt: skip
    codes = [MplPath.MOVETO] + [MplPath.CURVE4] * 6 + [MplPath.CLOSEPOLY]
    return MplPath(verts, codes)


def legend_box(scene: Scene, look: Look) -> Box:
    """Where the legend will be drawn, bottom-left; the frame search and label placer avoid it.
    Its width is the widest row text and the widest count, measured in the figure font."""
    size, pts = look.legend_text_size, look.page_points
    text = max((text_size(t, size)[0] for t, _, _ in scene.legend), default=0.0)
    count = max((text_size(str(n), size)[0] for _, _, n in scene.legend), default=0.0)
    gap = look.legend_count_gap * size
    width = (look.legend_text_left + text + gap + count + look.legend_count_inset) / pts
    height = (len(scene.legend) * look.legend_row_pitch + look.legend_padding) / pts
    return Box(
        "legend",
        look.legend_pad,
        1 - look.legend_pad - height,
        look.legend_pad + width,
        1 - look.legend_pad,
    )


def text_size(text: str, size: float, *, bold: bool = False) -> tuple[float, float]:
    """Width and height of ``text`` at ``size`` points in the CURRENT matplotlib font (the
    figure fonts' context while a map is built, af.util.fonts), as matplotlib lays it out."""
    width, height, _ = text_metrics(text, size, bold=bold)
    return width, height


def text_metrics(text: str, size: float, *, bold: bool = False) -> tuple[float, float, float]:
    """:func:`text_size` and the descent: how far ``text``'s glyphs reach below its baseline
    (points). The height runs from the lowest descender to the highest ascender of THIS text."""
    from matplotlib.font_manager import FontProperties
    from matplotlib.textpath import text_to_path

    prop = FontProperties(size=size, weight="bold" if bold else "normal")
    width, height, descent = text_to_path.get_text_width_height_descent(text, prop, ismath=False)
    return float(width), float(height), float(descent)


def title_box(scene: Scene, look: Look) -> Box:
    """The title's room, measured in the figure font: an estimate from the character count
    ran up to 13 u short of the drawn title, which then sat on points (style 08-20). It ends
    ``look.title_clear`` past the last letter: ending AT it, a marker could touch the title's
    "e" (R5 review, 19 Oct 2026), and ending 6 points past it, markers 8 points from the title
    sat in its line (R6 review, 20 Oct 2026)."""
    lines = scene.title.split("\n")
    width = max(text_size(line, look.title_size, bold=True)[0] for line in lines)
    right = 0.024 + (width + look.title_clear) / look.page_points
    # one line keeps the 0.06 the report maps have always had; each further line adds its pitch
    bottom = 0.06 + (len(lines) - 1) * TITLE_PITCH * look.title_size / look.page_points
    return Box("title", 0.0, 0.0, min(0.95, right), bottom)


def draw_pdf(
    scene: Scene,
    frame: Frame,
    labels: Mapping[str, Placed],
    out_pdf: Path,
    look: Look = DEFAULT_LOOK,
    styles: Mapping[str, PointStyle] | None = None,
    circles: Sequence[CircleMark] = (),
    caption: Sequence[str] = (),
) -> None:
    """Draw ``scene`` inside ``frame`` to ``out_pdf`` (one page). The drawing itself is
    :func:`draw_scene`, shared with :func:`af.map.figure.draw_axes`."""
    import matplotlib

    matplotlib.use("pdf")
    import matplotlib.pyplot as plt

    side = look.page_points / 72.0
    fig = plt.figure(figsize=(side, side * frame.aspect), dpi=72)
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
    draw_scene(ax, scene, frame, labels, look, styles=styles, circles=circles, caption=caption)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf)
    plt.close(fig)
    if not out_pdf.is_file() or out_pdf.stat().st_size == 0:
        raise RuntimeError(f"map PDF not written: {out_pdf}")


def draw_scene(
    ax: Any,
    scene: Scene,
    frame: Frame,
    labels: Mapping[str, Placed],
    look: Look,
    *,
    scale: float = 1.0,
    legend: bool = True,
    title: bool = True,
    styles: Mapping[str, PointStyle] | None = None,
    circles: Sequence[CircleMark] = (),
    caption: Sequence[str] = (),
) -> tuple[str, ...]:
    """Draw a scene on an Axes whose data limits become the page (0..1, y down).

    ``styles`` override single points' drawing, by point id. ``circles`` are drawn round their
    points (a point not shown, or with no coordinates, is an error: nothing to centre on).
    ``caption`` lines are printed small under the title, as the circle addenda list their sera.
    Returns the ids of shown points
    that lie outside the frame, which the Axes clips: a caller can mark where they went.

    ``scale`` multiplies every text size and line width: 1.0 on the report page; the Axes'
    width over ``look.page_points`` for a panel (:func:`af.map.figure.draw_axes`), so a panel is
    the page shrunk. Everything else is in fractions of the page's WIDTH: a rectangular frame's
    page runs 0..1 across and 0..aspect down, so a circle stays round and a map unit is the same
    length both ways. A square frame's aspect is exactly 1, and its page exactly the square one.
    """
    a = frame.aspect
    ax.set_xlim(0, 1)
    ax.set_ylim(a, 0)  # the PAGE grows downward; map y (up) became page y in Frame.page
    ax.axis("off")
    across, down = int(frame.size), int(frame.tall)
    for k in range(max(across, down) + 1):
        g = k / frame.size
        if k <= across:
            ax.plot([g, g], [0, a], color=look.grid_colour, lw=1.0 * scale, zorder=0)
        if k <= down:
            ax.plot([0, 1], [g, g], color=look.grid_colour, lw=1.0 * scale, zorder=0)
    page = frame.page(scene.xy())
    drawn = page * [1.0, a]  # in width units: what the Axes draws
    styles = styles or {}

    def order(p: ScenePoint) -> float:
        own = styles.get(p.id)
        if own is not None and own.on_top:
            return 5 + 0.1 * min(own.raise_rank or 0, 9)  # still under circles (7)
        if p.kind == "serum":
            return 0
        if p.mark:
            return 4
        if own is not None and own.raise_rank is not None:
            return 3 + 0.1 * (1 + min(own.raise_rank, 8))  # raised, still under the marks (4)
        return 1 if (p.greyed or p.colour is None) else (2 if p.reference else 3)

    for i in sorted(range(len(scene.points)), key=lambda i: order(scene.points[i])):
        p = scene.points[i]
        if not p.shown:
            continue
        x, y = drawn[i]
        z = order(p) + 1
        own = styles.get(p.id, PointStyle())
        s = marker_extent(p, look, own)
        fill, edge = point_colours(p)
        if p.kind == "serum":
            style: dict[str, Any] = {
                "fc": own.fill or fill,
                "ec": own.outline or edge,
                "lw": (own.outline_width if own.outline_width is not None else 1.0) * scale,
                "zorder": z,
            }
        else:
            style = {
                "fc": own.fill or fill,
                "ec": own.outline or edge,
                "lw": (own.outline_width if own.outline_width is not None else 1.0) * scale,
                "zorder": z,
            }
        alpha = own.alpha if own.alpha is not None else look.alpha
        if alpha is not None:
            style["alpha"] = alpha
        ax.add_patch(_marker_patch(own.marker or p.marker, x, y, s, style))
    _draw_circles(ax, scene, drawn, frame, circles, scale)
    from matplotlib import patheffects

    for lab in labels.values():
        # On the baseline that puts the glyphs in the placed box. Aligned "bottom", matplotlib
        # keeps room for a descender a label without one does not have, and drew "-cell" labels
        # up to 6 points above their box, over the label placed above (R5 review, 19 Oct 2026).
        descent = text_metrics(lab.text, look.label_size)[2] / look.page_points
        ax.text(
            lab.box[0],
            (lab.box[3] - descent) * a,
            lab.text,
            fontsize=look.label_size * scale,
            va="baseline",
            ha="left",
            zorder=8,
            path_effects=[
                patheffects.withStroke(linewidth=look.label_halo * scale, foreground="white")
            ],
        )
    for k, line in enumerate(caption):
        # ae's names overlay: 7 points, 1.25 line pitch, its first line's top 0.115 of the page down
        ax.text(
            0.024,
            0.115 * a + k * look.caption_size * 1.25 / look.page_points,
            line,
            fontsize=look.caption_size * scale,
            va="top",
            ha="left",
            zorder=9,
        )
    if title:
        # The shipped maps' title: baseline 37.5 and left edge 19 page units from the top-left
        # corner. The title box (furniture) is unchanged: the title stays inside its 0.06. A
        # title of several lines (the coverage maps': map, serum, clade) runs on down.
        for k, line in enumerate(scene.title.split("\n")):
            ax.text(
                19.0 / look.page_points,
                (37.5 + k * TITLE_PITCH * look.title_size) / look.page_points,
                line,
                fontsize=look.title_size * scale,
                weight="bold",
                va="baseline",
                zorder=9,
            )
    if legend:
        _draw_legend(ax, scene, look, scale, drop=a - 1.0)
    # Spelled out with np.asarray (as rows_with_coordinates is): older numpy stubs type a reduced
    # comparison as possibly a scalar, which fails type checking on Python 3.11 (CI).
    with np.errstate(invalid="ignore"):
        inside = np.asarray(((page >= 0) & (page <= 1)).all(axis=1), dtype=np.bool_)
    outside = rows_with_coordinates(page) & ~inside
    return tuple(
        p.id for p, out in zip(scene.points, outside.tolist(), strict=True) if p.shown and out
    )


def point_colours(p: ScenePoint) -> tuple[str, str]:
    """A point's fill and outline as the page draws them, before any :class:`PointStyle`: a
    serum is an open grey box; a greyed or uncoloured antigen its grey (an uncoloured
    reference antigen an open grey ring); a marked one has a black outline."""
    if p.kind == "serum":
        return "none", SERUM_OUTLINE
    grey = grey_of(p)
    fill = "none" if grey == OPEN else (grey or p.colour)
    assert fill is not None  # grey_of gives a grey whenever the colour is None
    edge = "black" if p.mark else (GREY if grey == OPEN else (grey or "black"))
    return fill, edge


def marker_extent(p: ScenePoint, look: Look, own: PointStyle | None = None) -> float:
    """How far a point's marker reaches from its centre, as a fraction of the page side: the
    radius of a circle or egg (an egg is as tall as a circle is wide, and narrower), the
    half-side of a box. The one place a marker's size is worked out, so the frame search tests
    the marker the PDF draws (a centre-only test let markers sit over the title, R4 review)."""
    own = own or PointStyle()
    if p.kind == "serum":
        s = look.serum_half_side
    else:
        s = look.antigen_radius * (look.mark_scale if p.mark else 1.0)
    return s * (own.scale if own.scale is not None else 1.0)


def _marker_patch(marker: str, x: float, y: float, size: float, style: dict[str, Any]) -> Any:
    """One marker. ``size`` is the radius ("circle", "egg") or half-side ("box", "uglyegg")."""
    from matplotlib.patches import Circle, PathPatch, Rectangle

    if marker == "circle":
        return Circle((x, y), size, **style)
    if marker == "egg":
        return PathPatch(egg_path(x, y, size), **style)
    if marker == "box":
        return Rectangle((x - size, y - size), 2 * size, 2 * size, **style)
    if marker == "uglyegg":
        return PathPatch(ugly_egg_path(x, y, size), **style)
    raise ValueError(f"unknown marker shape {marker!r} (known: {', '.join(MARKERS)})")


def _draw_circles(
    ax: Any,
    scene: Scene,
    drawn: np.ndarray,
    frame: Frame,
    circles: Sequence[CircleMark],
    scale: float,
) -> None:
    """Each circle's fill (translucent), then its outline (solid), over every point."""
    from matplotlib.patches import Circle

    at = {p.id: i for i, p in enumerate(scene.points)}
    for c in circles:
        i = at.get(c.point)
        if i is None or not scene.points[i].shown or not np.isfinite(drawn[i]).all():
            raise ValueError(f"circle round {c.point}: no drawn point to centre it on")
        centre, r = (float(drawn[i][0]), float(drawn[i][1])), c.radius / frame.size
        ax.add_patch(Circle(centre, r, fc=c.fill, ec="none", alpha=c.fill_alpha, zorder=7))
        ax.add_patch(
            Circle(
                centre, r, fc="none", ec=c.outline, lw=c.outline_width * scale, zorder=7,
                linestyle=(0, (3.0, 3.0)) if c.dashed else "solid",
            )
        )  # fmt: skip


def _draw_legend(
    ax: Any, scene: Scene, look: Look, scale: float = 1.0, *, drop: float = 0.0
) -> None:
    """The legend, bottom-left; ``drop`` moves it down to the bottom of a taller page."""
    from matplotlib.patches import Circle, Rectangle

    box = legend_box(scene, look)
    box = Box(box.name, box.left, box.top + drop, box.right, box.bottom + drop)
    ax.add_patch(
        Rectangle(
            (box.left, box.top),
            box.right - box.left,
            box.bottom - box.top,
            fc="white",
            ec="black",
            lw=1 * scale,
            zorder=10,
        )
    )
    pts = look.page_points
    font = look.legend_text_size * scale
    first = box.top + (look.legend_padding / 2 + look.legend_row_pitch / 2) / pts
    for k, (text, colour, count) in enumerate(scene.legend):
        y = first + k * look.legend_row_pitch / pts
        ax.add_patch(
            Circle(
                (box.left + look.legend_marker_left / pts, y),
                look.legend_marker_size / 2 / pts,
                fc=colour,
                ec="black",
                lw=look.legend_marker_outline * scale,
                zorder=11,
            )
        )
        ax.text(
            box.left + look.legend_text_left / pts, y, text, fontsize=font, va="center", zorder=11
        )
        ax.text(
            box.right - look.legend_count_inset / pts,
            y,
            str(count),
            fontsize=font,
            va="center",
            ha="right",
            zorder=11,
        )


def recent_hidden(scene: Scene, frame: Frame, furniture: Sequence[Box], since: dt.date) -> int:
    """Shown antigens isolated on or after ``since`` that end up off the page or under furniture
    (should be 0)."""
    from af.map.viewport import hidden_mask

    page = frame.page(scene.xy())
    hidden = hidden_mask(page, tuple(furniture))
    return sum(
        1
        for i, p in enumerate(scene.points)
        if p.kind == "antigen"
        and p.shown
        and p.date is not None
        and p.date >= since
        and bool(hidden[i])
    )
