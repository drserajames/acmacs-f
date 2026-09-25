"""Draw geo maps: one PDF per month, dots packed around each location.

Input is the I7 ``geo`` document (:func:`af.geo.records.to_i7`), so what is drawn and what
the comparison reads are the same data. Look follows today's ``geo/<st>-YYYY-MM.pdf``:
an equirectangular world outline with no frame, the month top-left, and at each location
its dots packed in hexagonal rings, the largest colour group at the centre.

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
    """Drawing parameters; the defaults follow today's maps."""

    width_inches: float = 10.0
    dot_size: float = 46.0  # matplotlib scatter size, points^2
    spacing_degrees: float = 3.0  # between dot centres in a packed cluster
    outline_colour: str = "#a1a1a1"
    outline_width: float = 0.5
    uncoloured_edge: str = "#a0a0a0"
    title_size: float = 16.0


@dataclass
class RenderReport:
    drawn: dict[str, int] = field(default_factory=dict)  # period -> dots drawn
    no_coordinates: Counter[str] = field(default_factory=Counter)  # location -> dots
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
            period, coordinates, shapes, path, look, report.no_coordinates
        )
        report.files.append(path)
    return report


def packed_offsets(n: int, spacing: float) -> list[tuple[float, float]]:
    """Centre first, then hexagonal rings of 6k points: dense, and the same for every n."""
    offsets = [(0.0, 0.0)]
    ring = 1
    while len(offsets) < n:
        for i in range(6 * ring):
            angle = 2 * math.pi * i / (6 * ring)
            offsets.append((ring * spacing * math.cos(angle), ring * spacing * math.sin(angle)))
        ring += 1
    return offsets[:n]


def _coastline(path: Path) -> list[Any]:
    if not Path(path).is_file():
        raise GeoRenderError(f"coastline shapefile not found: {path}")
    from cartopy.io import shapereader

    geometries = list(shapereader.Reader(str(path)).geometries())
    if not geometries:
        raise GeoRenderError(f"coastline shapefile has no shapes: {path}")
    return geometries


def _render_period(
    period: Mapping[str, Any],
    coordinates: Coordinates,
    shapes: list[Any],
    path: Path,
    look: Look,
    no_coordinates: Counter[str],
) -> int:
    import cartopy.crs as ccrs
    import matplotlib

    matplotlib.use("pdf")
    import matplotlib.pyplot as plt

    plate = ccrs.PlateCarree()
    fig = plt.figure(figsize=(look.width_inches, look.width_inches / 2))
    try:
        # A GeoAxes: matplotlib's stubs type add_axes as plain Axes.
        ax: Any = fig.add_axes((0.0, 0.0, 1.0, 1.0), projection=plate)
        ax.spines["geo"].set_visible(False)
        ax.set_extent((-180.0, 180.0, -90.0, 90.0), crs=plate)
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
            xs, ys, fills, edges = [], [], [], []
            for (dx, dy), point in zip(
                packed_offsets(len(dots), look.spacing_degrees), dots, strict=True
            ):
                xs.append(where[0] + dx)
                ys.append(where[1] + dy)
                coloured = point["color"] != "transparent"
                fills.append(point["color"] if coloured else "none")
                edges.append("#808080" if coloured else look.uncoloured_edge)
            ax.scatter(
                xs, ys, s=look.dot_size, c=fills, edgecolors=edges, linewidths=0.6,
                transform=plate, zorder=3,
            )  # fmt: skip
            drawn += len(dots)
        ax.text(
            0.01, 0.985, period.get("title", period["period"]), transform=ax.transAxes,
            fontsize=look.title_size, va="top",
        )  # fmt: skip
        fig.savefig(path)
    finally:
        plt.close(fig)
    if not path.is_file() or path.stat().st_size == 0:
        raise GeoRenderError(f"no map written: {path}")
    return drawn
