"""An af map figure from a chart, for any caller: the points, the styled scene, and a drawing.

This is the consumer entry point for drawing a chart the way af's report maps are drawn,
outside :mod:`af.map.build` (which adds a round's config, curation and orientation on top):

1. colours, optional: :class:`af.map.colouring.StoreColours` ``.for_chart(chart, scheme)`` gives
   the clade colouring of the reports (call :func:`af.serology.update.require_current` on the
   store first, as the build does);
2. :func:`chart_scene` turns the chart, a layout and those colours into a
   :class:`~af.map.style.Scene`;
3. :func:`frame_around` gives a square frame over the shown points (the build chooses frames by
   its own rules; this is the simple one);
4. :func:`draw_axes` draws the scene on a matplotlib Axes. It is the same code that draws the
   report PDF (:func:`af.map.render.draw_pdf` calls it), so a panel is the report page scaled,
   not a lookalike.

Also public: :func:`marker_for_passage`, a point's marker shape from its passage, for a caller
that builds points itself (:func:`chart_scene` already sets every point's ``marker``).

Displayed coordinates grow upwards, as in R (Q105): pass a chart's layout as stored, and the map
looks as R draws it.

Everything else in af.map is the build's own machinery; pin only to the names above.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from af.chart.model import Chart
from af.map.style import Marker, PointIn, Scene, Window, style_points
from af.map.vaccines import marker_for, passage_class
from af.map.viewport import Frame

if TYPE_CHECKING:
    from af.map.colouring import ChartColours
    from af.map.labels import Placed
    from af.map.render import Look


class FigureError(ValueError):
    """A figure cannot be drawn as asked: a layout of the wrong shape, a non-square Axes."""


def parse_date(text: str) -> dt.date | None:
    """An isolation date, parsed. Never inferred from anything else (design rule 7)."""
    try:
        return dt.date.fromisoformat(text[:10])
    except (ValueError, TypeError):
        return None


def marker_for_passage(kind: str, passage: str, reassortant: str = "") -> Marker:
    """The marker shape af draws for a point (public): ``kind`` is "antigen" or "serum",
    ``passage`` and ``reassortant`` as the chart records them. Egg-grown antigens are eggs and
    egg-grown sera ugly eggs; every other antigen is a circle, every other serum a box.

    Supported for callers: the same rule :func:`chart_scene` applies, decided in one place
    (:func:`af.map.vaccines.marker_for`).
    """
    if kind not in ("antigen", "serum"):
        raise FigureError(f"marker_for_passage: kind {kind!r} is not 'antigen' or 'serum'")
    return marker_for(kind, passage_class(passage, reassortant))


def chart_points(
    chart: Chart,
    xy: NDArray[np.float64],
    *,
    labels: Sequence[frozenset[str]] | None = None,
    sequenced: Sequence[bool] | None = None,
    hidden: Mapping[int, str] | None = None,
) -> list[PointIn]:
    """One :class:`PointIn` per antigen then serum, ids "ag<i>" / "sr<j>".

    ``xy`` is the displayed layout (antigens then sera, NaN rows for points with no
    coordinates). ``labels`` are each antigen's colour labels (from
    :class:`~af.map.colouring.ChartColours` or a stand-in chart); none means unpainted.
    ``hidden`` names the rule that hides an antigen.
    """
    n = chart.n_antigens + chart.n_sera
    if xy.ndim != 2 or xy.shape[0] != n:
        raise FigureError(f"layout has shape {xy.shape} for {n} points")
    labels = labels if labels is not None else [frozenset()] * chart.n_antigens
    if sequenced is None:
        sequenced = [bool(a.extra.get("A")) for a in chart.antigens]
    hidden = hidden or {}
    points: list[PointIn] = []
    for i, a in enumerate(chart.antigens):
        ok = bool(np.isfinite(xy[i]).all())
        points.append(
            PointIn(
                f"ag{i}",
                a.name,
                "antigen",
                (float(xy[i, 0]), float(xy[i, 1])) if ok else None,
                labels[i],
                parse_date(a.date),
                passage_class(a.passage, a.reassortant),
                bool((a.extra.get("T") or {}).get("R")),
                sequenced=sequenced[i],
                hide=hidden.get(i),
                marker=marker_for("antigen", passage_class(a.passage, a.reassortant)),
            )
        )
    for j, s in enumerate(chart.sera):
        k = chart.n_antigens + j
        ok = bool(np.isfinite(xy[k]).all())
        points.append(
            PointIn(
                f"sr{j}",
                s.name,
                "serum",
                (float(xy[k, 0]), float(xy[k, 1])) if ok else None,
                passage_class=passage_class(s.passage, s.reassortant),
                serum_id=s.serum_id,
                marker=marker_for("serum", passage_class(s.passage, s.reassortant)),
            )
        )
    return points


def chart_scene(
    chart: Chart,
    xy: NDArray[np.float64],
    colours: ChartColours,
    *,
    title: str,
    window: Window | None = None,
    vaccines: Mapping[str, str] | None = None,
) -> Scene:
    """The styled scene for a chart drawn at ``xy`` with ``colours``: the report maps' colours,
    greying (``window``; default: nothing greyed) and vaccine marks (point id -> label)."""
    points = chart_points(chart, xy, labels=colours.labels, sequenced=colours.sequenced)
    return style_points(
        points,
        colours.scheme,
        window or Window("all", None),
        title=title,
        vaccines=dict(vaccines or {}),
    )


def frame_around(scene: Scene, *, margin: float = 1.0, size: float | None = None) -> Frame:
    """A square frame over every shown point, ``margin`` map units beyond them; or of a fixed
    ``size`` (the report maps' frame per subtype), centred on them."""
    xy = np.array([p.xy for p in scene.points if p.shown and p.xy is not None], dtype=float)
    if len(xy) == 0:
        raise FigureError("no shown point has coordinates: nothing to frame")
    lo, hi = xy.min(axis=0), xy.max(axis=0)
    side = float(size) if size is not None else float((hi - lo).max()) + 2 * margin
    centre = (lo + hi) / 2
    return Frame(float(centre[0] - side / 2), float(centre[1] + side / 2), side)  # top-left


def draw_axes(
    ax: Any,
    scene: Scene,
    frame: Frame,
    *,
    labels: Mapping[str, Placed] | None = None,
    look: Look | None = None,
    legend: bool = True,
    title: bool = True,
) -> None:
    """Draw ``scene`` inside ``frame`` on a matplotlib Axes, exactly as the report PDF draws it.

    The Axes must be square on the figure: the map frame is square, and a stretched panel would
    misrepresent distances. Marker sizes, grid and legend box are fractions of the Axes; text
    sizes and line widths are the report page's, scaled by the Axes' side over the page's
    (``look.page_points``), so a panel is the report page shrunk. ``legend`` and ``title`` can be
    left off for small panels; nothing else differs from the PDF.
    """
    from af.map.render import DEFAULT_LOOK, draw_scene

    look = look or DEFAULT_LOOK
    fig = ax.get_figure()
    box = ax.get_position()
    w_in, h_in = box.width * fig.get_figwidth(), box.height * fig.get_figheight()
    if abs(w_in - h_in) > 1e-3 * max(w_in, h_in):
        raise FigureError(
            f"the Axes is {w_in:.3f} x {h_in:.3f} in: a map panel must be square, as the frame is"
        )
    scale = w_in * 72.0 / look.page_points
    draw_scene(ax, scene, frame, labels or {}, look, scale=scale, legend=legend, title=title)
