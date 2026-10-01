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

from af.map.labels import Placed
from af.map.style import MARKERS, Scene, ScenePoint
from af.map.viewport import Box, Frame

GREY = "#c8c8c8"
SERUM_OUTLINE = "#9a9a9a"


@dataclass(frozen=True)
class Look:
    """Sizes as fractions of the page side (so the look does not depend on the PDF size)."""

    page_points: float = 800.0
    antigen_radius: float = 0.0125
    vaccine_scale: float = 2.0
    serum_half_side: float = 0.0125
    title_size: float = 25.0
    label_size: float = 22.0
    legend_row_height: float = 0.0275
    legend_char_width: float = 0.0105
    legend_pad: float = 0.0125


DEFAULT_LOOK = Look()


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
    """Where the legend will be drawn, bottom-left; the frame search and label placer avoid it."""
    rows = len(scene.legend)
    text = max((len(t) for t, _, _ in scene.legend), default=0)
    count = max((len(str(n)) for _, _, n in scene.legend), default=0)
    width = (
        look.legend_pad * 4 + look.legend_row_height + (text + count + 2) * look.legend_char_width
    )
    height = rows * look.legend_row_height + 2 * look.legend_pad
    return Box(
        "legend",
        look.legend_pad,
        1 - look.legend_pad - height,
        look.legend_pad + width,
        1 - look.legend_pad,
    )


def title_box(scene: Scene, look: Look) -> Box:
    width = min(0.95, 0.024 + len(scene.title) * look.title_size * 0.6 / look.page_points)
    return Box("title", 0.0, 0.0, width, 0.06)


def draw_pdf(
    scene: Scene,
    frame: Frame,
    labels: Mapping[str, Placed],
    out_pdf: Path,
    look: Look = DEFAULT_LOOK,
) -> None:
    """Draw ``scene`` inside ``frame`` to ``out_pdf`` (one page). The drawing itself is
    :func:`draw_scene`, shared with :func:`af.map.figure.draw_axes`."""
    import matplotlib

    matplotlib.use("pdf")
    import matplotlib.pyplot as plt

    side = look.page_points / 72.0
    fig = plt.figure(figsize=(side, side), dpi=72)
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
    draw_scene(ax, scene, frame, labels, look)
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
) -> None:
    """Draw a scene on an Axes whose data limits become the page (0..1, y down).

    ``scale`` multiplies every text size and line width: 1.0 on the report page; the Axes'
    side over ``look.page_points`` for a panel (:func:`af.map.figure.draw_axes`), so a panel is
    the page shrunk. Everything else is in page fractions already.
    """
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)  # y grows downward, as in the map frame
    ax.axis("off")
    for k in range(int(frame.size) + 1):
        g = k / frame.size
        ax.plot([g, g], [0, 1], color="#dddddd", lw=0.8 * scale, zorder=0)
        ax.plot([0, 1], [g, g], color="#dddddd", lw=0.8 * scale, zorder=0)
    page = frame.page(scene.xy())

    def order(p: ScenePoint) -> int:
        if p.kind == "serum":
            return 0
        if p.vaccine:
            return 4
        return 1 if (p.greyed or p.colour is None) else (2 if p.reference else 3)

    for i in sorted(range(len(scene.points)), key=lambda i: order(scene.points[i])):
        p = scene.points[i]
        if not p.shown:
            continue
        x, y = page[i]
        z = order(p) + 1
        if p.kind == "serum":
            s = look.serum_half_side
            style: dict[str, Any] = {"fc": "none", "ec": SERUM_OUTLINE, "lw": scale, "zorder": z}
        else:
            grey = p.greyed or p.colour is None
            edge = "black" if p.vaccine else (GREY if grey else "black")
            s = look.antigen_radius * (look.vaccine_scale if p.vaccine else 1.0)
            style = {"fc": GREY if grey else p.colour, "ec": edge, "lw": 0.8 * scale, "zorder": z}
        ax.add_patch(_marker_patch(p.marker, x, y, s, style))
    for lab in labels.values():
        ax.text(
            lab.box[0],
            lab.box[3],
            lab.text,
            fontsize=look.label_size * scale,
            va="bottom",
            ha="left",
            zorder=8,
        )
    if title:
        ax.text(
            0.024,
            0.015,
            scene.title,
            fontsize=look.title_size * scale,
            weight="bold",
            va="top",
            zorder=9,
        )
    if legend:
        _draw_legend(ax, scene, look, scale)


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


def _draw_legend(ax: Any, scene: Scene, look: Look, scale: float = 1.0) -> None:
    from matplotlib.patches import Circle, Rectangle

    box = legend_box(scene, look)
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
    h = look.legend_row_height
    font = h * look.page_points * 0.62 * scale
    for k, (text, colour, count) in enumerate(scene.legend):
        y = box.top + look.legend_pad + h * (k + 0.5)
        ax.add_patch(
            Circle(
                (box.left + look.legend_pad + h / 2, y),
                h * 0.35,
                fc=colour,
                ec="black",
                lw=0.6 * scale,
                zorder=11,
            )
        )
        ax.text(box.left + 2 * look.legend_pad + h, y, text, fontsize=font, va="center", zorder=11)
        ax.text(
            box.right - look.legend_pad,
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
