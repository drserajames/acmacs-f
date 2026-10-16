"""Draw geo maps: one PDF per month, dots packed around each location.

Input is the I7 ``geo`` document (:func:`af.geo.records.to_i7`), so what is drawn and what
the comparison reads are the same data. The look is ae's ``geo-draw`` (the shipped report's
``geo/<st>-YYYY-MM.pdf``): an equirectangular world centred at 11.76 E with no frame, the month
top-left, and at each location its dots packed in ae's rings, the largest colour group at the
centre. Sizes are in the figure's own points at ae's map width, so a panel set at the shipped
width shows ae's sizes at ae's scale.

Two inputs are explicit rather than fetched or guessed:

- **The coastline** is a Natural Earth shapefile at a path from config. cartopy would
  otherwise download it on first use, a network input the pipeline cannot pin or check.
  A missing file is an error.
- **Coordinates** come from a function ``location -> (longitude, latitude)``, the sequence
  workstream's location lookup. A location it cannot place is not drawn and is reported,
  never dropped silently.

The returned :class:`RenderReport` says how many dots each month drew and which
locations had no coordinates, so a thin map says why.
"""

from __future__ import annotations

import functools
import math
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

Coordinates = Callable[[str], "tuple[float, float] | None"]


class GeoRenderError(RuntimeError):
    """An input the renderer needs is missing or unusable."""


@dataclass(frozen=True)
class Look:
    """Drawing parameters, in the figure's points. The defaults are ae geo-draw's
    (ae cc/geo/geographic-map.cc, geo-draw-main.cc and the report's geographic settings)."""

    width_points: float = 800.0  # ae's time-series width
    aspect: float = 632.591 / 1261.3  # ae's world outline, height / width
    central_longitude: float = 11.762094  # ae's map spans 168.24 W to 191.76 E
    dot_diameter: float = 8.0  # ae point_size
    ring_step: float = 6.4  # ae point_size x density (0.8): ring r is r x ring_step out
    outline_colour: str = "#a1a1a1"
    outline_width: float = 0.5  # ae: width / 1600
    uncoloured_edge: str = "#a0a0a0"
    title_size: float = 20.0  # ae: width / 40
    title_inset: float = 10.0  # ae: width x 0.0125, to the top-left of the title's ink


@dataclass
class RenderReport:
    drawn: dict[str, int] = field(default_factory=dict)  # period -> dots drawn
    no_coordinates: Counter[str] = field(default_factory=Counter)  # location -> dots
    clipped: Counter[str] = field(default_factory=Counter)  # location -> dots cut by the edge
    files: list[Path] = field(default_factory=list)


def render_geo(
    doc: Mapping[str, Any],
    coordinates: Coordinates,
    coastline: Path,
    out_dir: Path,
    prefix: str,
    look: Look | None = None,
) -> RenderReport:
    """Write ``<out_dir>/<prefix>-<YYYY-MM>.pdf`` for every period of ``doc``."""
    look = look if look is not None else Look()
    shapes = _coastline(coastline)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = RenderReport()
    for period in doc["periods"]:
        path = out_dir / f"{prefix}-{period['period']}.pdf"
        report.drawn[period["period"]] = _render_period(
            period, coordinates, shapes, path, look, report.no_coordinates, report.clipped
        )
        report.files.append(path)
    return report


def render_period(
    period: Mapping[str, Any],
    coordinates: Coordinates,
    coastline: Path,
    path: Path,
    look: Look | None = None,
    no_coordinates: Counter[str] | None = None,
    clipped: Counter[str] | None = None,
) -> int:
    """Draw one period of an I7 ``geo`` document to ``path``; the number of dots drawn.

    For a caller that files each month separately (the report's one figure per folder).
    Locations it cannot place are added to ``no_coordinates``; dots the map's edge cuts, to
    ``clipped``.
    """
    unplaced = no_coordinates if no_coordinates is not None else Counter()
    cut = clipped if clipped is not None else Counter()
    look = look if look is not None else Look()
    return _render_period(period, coordinates, _coastline(coastline), path, look, unplaced, cut)


def ring_capacity(ring: int) -> int:
    """Dots on ring ``ring`` in ae's packing: floor(2 pi ring^2), so 6, 25, 56, 100 ... From ring 2
    on they overlap (about 0.4 of a diameter apart), which is what makes a big cluster a disc."""
    return int(2.0 * math.pi * ring * ring)


def packed_offsets(n: int, step: float) -> list[tuple[float, float]]:
    """ae's packing (geo-draw ``pack_colored_points``, a port of AD's): the centre first, then
    ring r at ``r * step`` holding :func:`ring_capacity` dots, evenly spaced from angle 0; the
    last ring holds what is left. The same for every n, in the units of ``step``."""
    offsets = [(0.0, 0.0)]
    ring = 1
    while len(offsets) < n:
        k = min(ring_capacity(ring), n - len(offsets))
        for i in range(k):
            angle = 2 * math.pi * i / k
            offsets.append((ring * step * math.cos(angle), ring * step * math.sin(angle)))
        ring += 1
    return offsets[:n]


def wrap_longitude(longitude: float, central: float) -> float:
    """Degrees east of the map's centre, in [-180, 180): a place west of the left edge is drawn
    at the right, never lost (ae drew it off the canvas)."""
    return (longitude - central + 180.0) % 360.0 - 180.0


def _coastline(path: Path) -> list[Any]:
    return list(_read_coastline(str(Path(path).resolve())))


@functools.cache
def _read_coastline(path: str) -> tuple[Any, ...]:
    """Read once per process: every month of every subtype draws the same outline."""
    if not Path(path).is_file():
        raise GeoRenderError(f"coastline shapefile not found: {path}")
    from cartopy.io import shapereader

    geometries = list(shapereader.Reader(str(path)).geometries())
    if not geometries:
        raise GeoRenderError(f"coastline shapefile has no shapes: {path}")
    return tuple(geometries)


def _render_period(
    period: Mapping[str, Any],
    coordinates: Coordinates,
    shapes: list[Any],
    path: Path,
    look: Look,
    no_coordinates: Counter[str],
    clipped: Counter[str],
) -> int:
    import cartopy.crs as ccrs
    import matplotlib

    matplotlib.use("pdf")
    import matplotlib.pyplot as plt

    plate = ccrs.PlateCarree()
    # Dots are placed in the map's own coordinates (degrees from its centre), so a cluster is
    # packed in one piece and an edge cuts it rather than splitting it across the map.
    centred = ccrs.PlateCarree(central_longitude=look.central_longitude)
    height = look.width_points * look.aspect
    fig = plt.figure(figsize=(look.width_points / 72.0, height / 72.0))
    # figure points to degrees: the map fills the figure, as ae's does
    lon_per_point, lat_per_point = 360.0 / look.width_points, 180.0 / height
    radius = look.dot_diameter / 2.0 * lon_per_point
    try:
        # A GeoAxes: matplotlib's stubs type add_axes as plain Axes.
        ax: Any = fig.add_axes((0.0, 0.0, 1.0, 1.0), projection=centred)
        ax.spines["geo"].set_visible(False)
        ax.set_extent((-180.0, 180.0, -90.0, 90.0), crs=centred)
        ax.set_aspect("auto")  # ae's outline is 0.3 % taller than 2:1; fill the figure as it does
        ax.add_geometries(
            shapes, crs=plate, facecolor="none", edgecolor=look.outline_colour,
            linewidth=look.outline_width,
        )  # fmt: skip
        drawn = 0
        for location in period["locations"]:
            dots = [p for p in location["points"] for _ in range(int(p["count"]))]
            where = coordinates(location["name"])
            if where is None:
                no_coordinates[location["name"]] += len(dots)
                continue
            x0 = wrap_longitude(where[0], look.central_longitude)
            xs, ys, fills, edges = [], [], [], []
            for (dx, dy), point in zip(
                packed_offsets(len(dots), look.ring_step), dots, strict=True
            ):
                xs.append(x0 + dx * lon_per_point)
                ys.append(where[1] + dy * lat_per_point)
                coloured = point["color"] != "transparent"
                fills.append(point["color"] if coloured else "none")
                edges.append("#808080" if coloured else look.uncoloured_edge)
            cut = sum(1 for x in xs if abs(x) + radius > 180.0)
            if cut:
                clipped[location["name"]] += cut
            ax.scatter(
                xs, ys, s=look.dot_diameter**2, c=fills, edgecolors=edges, linewidths=0.6,
                transform=centred, zorder=3,
            )  # fmt: skip
            drawn += len(dots)
        _title(fig, ax, period.get("title", period["period"]), look)
        fig.savefig(path)
    finally:
        plt.close(fig)
    if not path.is_file() or path.stat().st_size == 0:
        raise GeoRenderError(f"no map written: {path}")
    return drawn


def _title(fig: Any, ax: Any, title: str, look: Look) -> None:
    """The month, as ae places it: the top-left of its ink at ``title_inset`` from the map's
    top-left corner, whatever the font's side bearing and ascent."""
    from matplotlib.font_manager import FontProperties
    from matplotlib.textpath import TextPath
    from matplotlib.transforms import offset_copy

    ink = TextPath((0, 0), title, size=look.title_size, prop=FontProperties()).get_extents()
    at = offset_copy(
        ax.transAxes, fig=fig, x=look.title_inset - ink.x0,
        y=-(look.title_inset + ink.y1), units="points",
    )  # fmt: skip
    ax.text(0.0, 1.0, title, transform=at, fontsize=look.title_size, ha="left", va="baseline")
