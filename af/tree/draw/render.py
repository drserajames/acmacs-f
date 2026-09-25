"""Draw the report tree figure as a vector PDF (matplotlib).

Page geometry follows the past report trees (701 x 1000 pt): the tree, an optional column for
one centre's antigens, the month x leaf matrix, the clade brackets, then the aa dash-bars.
Everything is placed in page points. The function returns what it drew, for the I7 JSON and the
draw report; the caller writes those.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from importlib import resources
from pathlib import Path

import matplotlib

matplotlib.use("pdf")
import matplotlib.pyplot as plt  # noqa: E402  (after the backend is chosen)
import numpy as np  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.collections import LineCollection, PolyCollection  # noqa: E402
from matplotlib.font_manager import FontProperties  # noqa: E402
from matplotlib.textpath import TextPath  # noqa: E402

from .aa_labels import AALabel  # noqa: E402
from .layout import Layout  # noqa: E402
from .model import DrawTree  # noqa: E402
from .place import Box, Grid, Search, leader_end, place_labels, placement_metrics  # noqa: E402
from .sections import HzBand, Selection  # noqa: E402
from .timeseries import TimeSeries  # noqa: E402

CONTINENT_COLOURS = {  # the past reports' continent legend (ae cc/tal/draw-tree.cc)
    "EUROPE": "#00FF00",
    "CENTRAL-AMERICA": "#AAF9FF",
    "MIDDLE-EAST": "#8000FF",
    "NORTH-AMERICA": "#00008B",
    "AFRICA": "#FF8000",
    "ASIA": "#FF0000",
    "RUSSIA": "#B03060",
    "AUSTRALIA-OCEANIA": "#FF69B4",
    "SOUTH-AMERICA": "#40E0D0",
    "ANTARCTICA": "#808080",
}
UNKNOWN_COLOUR = "#808080"
INSET_CONTINENTS = [c for c in CONTINENT_COLOURS if c != "ANTARCTICA"]
# Placeholder palette for the clade style until the user colour schemes (workstream 4) land.
PALETTE = [
    "#1b9e77",
    "#d95f02",
    "#7570b3",
    "#e7298a",
    "#66a61e",
    "#e6ab02",
    "#a6761d",
    "#1f78b4",
    "#b2df8a",
    "#fb9a99",
    "#cab2d6",
    "#6a3d9a",
    "#ff7f00",
]
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


@dataclass
class Geometry:
    width: float = 700.952
    height: float = 1000.0
    top: float = 27.0
    bottom: float = 976.0
    tree_left: float = 22.0
    tree_right: float = 372.0
    centre_col: float = 0.0  # width of the centre-marks column; 0 = none
    ts_left: float = 382.0
    ts_slot: float = 7.5
    clade_gap: float = 10.0
    clade_slot: float = 9.0
    bar_width: float = 9.0
    bar_gap: float = 4.5
    label_font: float = 6.8
    strain_font: float = 6.2
    # Fixed scales, for figures compared side by side (e.g. a tree before and after pruning):
    # the branch length drawn across the tree's width, and the rows the page height holds.
    # None = fit this tree.
    x_max: float | None = None
    row_capacity: int | None = None


@dataclass
class DashBar:
    """Amino acid at one position, one colour per residue; 'transparent' is not drawn."""

    pos: int
    colours: dict[str, str]
    legend: list[tuple[str, str]] = field(default_factory=list)  # (text, colour)


@dataclass
class FigureSpec:
    title: str
    tree: DrawTree
    layout: Layout
    selection: Selection
    hz: list[HzBand]
    timeseries: TimeSeries
    labels: list[AALabel]
    dash_bars: list[DashBar] = field(default_factory=list)
    strains: list[tuple[str, str]] = field(default_factory=list)  # (leaf id, label text)
    marked_rows: np.ndarray | None = None  # centre-marked variant: bool per row
    marked_name: str = ""
    marked_colour: str = "#d62728"
    colour_by: str = "continent"  # or "clade": matrix and branches by lettered band
    clade_colours: dict[str, str] = field(default_factory=dict)
    geometry: Geometry = field(default_factory=Geometry)


@dataclass
class Drawn:
    """What the renderer drew, for I7 and the draw report."""

    placed: list  # place.Placed
    label_metrics: dict
    strains: list[dict]
    continents_not_in_legend: dict[str, int]  # value -> rows drawn grey for it


class _Page:
    """Row and branch-length coordinates -> page points, shared by the panels."""

    def __init__(self, spec: FigureSpec) -> None:
        g, lay = spec.geometry, spec.layout
        self.g = g
        capacity = g.row_capacity or lay.n_rows
        if capacity < lay.n_rows:
            raise ValueError(f"row_capacity {capacity} < {lay.n_rows} rows to draw")
        self.row_h = (g.bottom - g.top) / capacity
        self.bottom = g.top + lay.n_rows * self.row_h  # where the drawn rows end
        self.tree_right = g.tree_right - g.centre_col
        xmax = g.x_max or float(np.nanmax(lay.node_x[lay.leaf_nodes])) or 1.0
        self.nx = g.tree_left + lay.node_x / xmax * (self.tree_right - g.tree_left)
        self.ny = g.top + (lay.node_y + 0.5) * self.row_h

    def row_y(self, rows: np.ndarray | int) -> np.ndarray:
        return self.g.top + (np.asarray(rows) + 0.5) * self.row_h

    def row_edge(self, row: int) -> float:
        return self.g.top + row * self.row_h

    @property
    def dash_width(self) -> float:
        """A row is far thinner than a printable line on a big tree; never go below 0.25 pt."""
        return max(0.25, self.row_h)


def _text_width(font: FontProperties):
    cache: dict[str, float] = {}

    def width(s: str) -> float:
        if s not in cache:
            cache[s] = TextPath((0, 0), s, prop=font).get_extents().width
        return cache[s]

    return width


def clade_colour_scheme(spec: FigureSpec) -> dict[str, str]:
    names = sorted({h.clade for h in spec.hz})
    return {c: spec.clade_colours.get(c, PALETTE[k % len(PALETTE)]) for k, c in enumerate(names)}


def continents_not_in_legend(spec: FigureSpec) -> dict[str, int]:
    """Rows whose continent the legend does not know, by value ("" = none): they draw grey,
    so a vocabulary mismatch upstream shows up as a count instead of silently."""
    out: dict[str, int] = {}
    for i in spec.layout.leaf_nodes:
        value = spec.tree.continent[i] or ""
        if value not in CONTINENT_COLOURS:
            out[value] = out.get(value, 0) + 1
    return out


def row_colours(spec: FigureSpec) -> np.ndarray:
    """Colour per row for the matrix: by continent, or by lettered clade band."""
    t, lay = spec.tree, spec.layout
    if spec.colour_by == "clade":
        scheme = clade_colour_scheme(spec)
        out = np.full(lay.n_rows, "#bdbdbd", dtype=object)
        for h in spec.hz:
            out[h.first : h.last + 1] = scheme[h.clade]
        return out
    if spec.colour_by != "continent":
        raise ValueError(f"unknown colour_by {spec.colour_by!r}")
    return np.array(
        [CONTINENT_COLOURS.get(t.continent[i] or "", UNKNOWN_COLOUR) for i in lay.leaf_nodes],
        dtype=object,
    )


def _draw_tree(ax: Axes, spec: FigureSpec, pg: _Page, grid: Grid, colours: np.ndarray) -> None:
    t, lay = spec.tree, spec.layout
    segs, seg_rows = [], []
    for i in range(len(t)):
        if lay.first_row[i] < 0:
            continue
        rows = (int(lay.first_row[i]), int(lay.last_row[i]))
        p = t.parent[i]
        if p >= 0:
            segs.append(((pg.nx[p], pg.ny[i]), (pg.nx[i], pg.ny[i])))
            seg_rows.append(rows)
        drawn = [c for c in t.children[i] if lay.first_row[c] >= 0]
        if len(drawn) > 1:
            segs.append(((pg.nx[i], pg.ny[drawn[0]]), (pg.nx[i], pg.ny[drawn[-1]])))
            seg_rows.append(rows)
    if spec.colour_by == "clade":
        seg_colours = [colours[a] if colours[a] == colours[b] else "black" for a, b in seg_rows]
    else:
        seg_colours = ["black"] * len(segs)
    ax.add_collection(LineCollection(segs, colors=seg_colours, linewidths=0.12))
    for (x0, y0), (x1, y1) in segs:
        if y0 == y1:
            grid.add_hline(x0, x1, y0)
        else:
            grid.add_vline(x0, y0, y1)


def _draw_marks(ax: Axes, spec: FigureSpec, pg: _Page) -> None:
    if spec.marked_rows is None:
        return
    g, lay = pg.g, spec.layout
    rows = np.flatnonzero(spec.marked_rows)
    ys = pg.row_y(rows)
    ax.scatter(pg.nx[lay.leaf_nodes[rows]], ys, s=1.2, c=spec.marked_colour, linewidths=0, zorder=5)
    x0 = pg.tree_right + 3
    ax.add_collection(
        LineCollection(
            [((x0, y), (x0 + g.centre_col - 5, y)) for y in ys],
            colors=spec.marked_colour,
            linewidths=max(0.3, pg.row_h),
        )
    )
    ax.text(
        x0 + (g.centre_col - 5) / 2,
        g.top - 3,
        spec.marked_name,
        rotation=90,
        ha="center",
        va="bottom",
        fontsize=6,
        color=spec.marked_colour,
    )


def _draw_matrix(ax: Axes, spec: FigureSpec, pg: _Page, colours: np.ndarray) -> float:
    g, ts = pg.g, spec.timeseries
    right = g.ts_left + len(ts.months) * g.ts_slot
    for k in range(len(ts.months) + 1):
        x = g.ts_left + k * g.ts_slot
        ax.plot([x, x], [g.top, pg.bottom], color="black", lw=0.5)
    for k, (y, m) in enumerate(ts.months):
        x = g.ts_left + (k + 0.5) * g.ts_slot
        text = f"{MONTHS[m - 1]} {y % 100:02d}"
        for yy, va in ((g.top - 2, "bottom"), (pg.bottom + 2, "top")):
            ax.text(
                x, yy, text, rotation=-90, ha="center", va=va, fontsize=5.8, rotation_mode="anchor"
            )
    rows = np.flatnonzero(ts.in_window)
    x0 = g.ts_left + (ts.column[rows] + 0.15) * g.ts_slot
    ys = pg.row_y(rows)
    segs = [((a, y), (a + 0.7 * g.ts_slot, y)) for a, y in zip(x0, ys, strict=True)]
    ax.add_collection(LineCollection(segs, colors=list(colours[rows]), linewidths=pg.dash_width))
    return right


def _draw_brackets(ax: Axes, spec: FigureSpec, pg: _Page, left: float) -> float:
    g, sel = pg.g, spec.selection
    names: list[tuple[float, float, str]] = []
    for cs in sel.shown:
        x = left + sel.slots[cs.clade] * g.clade_slot + 2
        for b in cs.bands:
            y0, y1 = pg.row_edge(b.first), pg.row_edge(b.last + 1)
            ax.annotate(
                "",
                xy=(x, y0),
                xytext=(x, y1),
                arrowprops={
                    "arrowstyle": "<->",
                    "lw": 0.6,
                    "shrinkA": 0,
                    "shrinkB": 0,
                    "mutation_scale": 4,
                },
            )
            for yy in (y0, y1):
                ax.plot([x - 2.5, x], [yy, yy], color="black", lw=0.5)
            names.append((x, (y0 + y1) / 2, cs.clade))
    _place_clade_names(ax, names)
    return left + (max(sel.slots.values(), default=0) + 1) * g.clade_slot + 8


def _place_clade_names(ax: Axes, names: list[tuple[float, float, str]], size: float = 7) -> None:
    """Write each clade's name beside its bracket, centred on the band.

    Two small bands next to each other in one column would print their names on top of each
    other; a name that would overlap the one above slides down just past it, with a thin
    leader back to its band's middle, so every name stays readable.
    """
    width = _text_width(FontProperties(size=size))
    last_end: dict[float, float] = {}
    for x, mid, text in sorted(names, key=lambda t: (t[0], t[1])):
        half = width(text) / 2 + 3  # a clear gap between stacked names
        centre = max(mid, last_end.get(x, -1e9) + half)
        last_end[x] = centre + half
        if centre != mid:
            ax.plot([x, x + 3.2], [mid, centre - half + 1], color="#7f7f7f", lw=0.3)
        ax.text(
            x + 1.2,
            centre,
            text,
            rotation=-90,
            ha="center",
            va="bottom",
            fontsize=size,
            rotation_mode="anchor",
        )


def _draw_hz_rules(ax: Axes, spec: FigureSpec, pg: _Page, right: float) -> None:
    rows = sorted({h.first for h in spec.hz} | {h.last + 1 for h in spec.hz})
    for r in rows:
        if 0 < r < spec.layout.n_rows:
            y = pg.row_edge(r)
            ax.plot([pg.g.ts_left, right - 6], [y, y], color="#7f7f7f", lw=0.5)


def _draw_dash_bars(ax: Axes, spec: FigureSpec, pg: _Page, left: float) -> None:
    """As many bars as configured, narrowed to fit the width left on the page."""
    if not spec.dash_bars:
        return
    g = pg.g
    pitch = min(g.bar_width + g.bar_gap, (g.width - 6 - left) / len(spec.dash_bars))
    width = pitch * g.bar_width / (g.bar_width + g.bar_gap)
    t, lay = spec.tree, spec.layout
    x = left
    for bar in spec.dash_bars:
        segs, cols = [], []
        for r, i in enumerate(lay.leaf_nodes):
            a = t.aa[i]
            colour = bar.colours.get(a[bar.pos - 1]) if a and len(a) >= bar.pos else None
            if colour is None or colour == "transparent":
                continue
            y = float(pg.row_y(r))
            segs.append(((x, y), (x + width, y)))
            cols.append(colour)
        ax.add_collection(LineCollection(segs, colors=cols, linewidths=pg.dash_width))
        for k, (text, colour) in enumerate(bar.legend):
            ax.text(
                x + width / 2,
                g.top - 16 + k * 6.5,
                text,
                color=colour,
                fontsize=5,
                ha="center",
                va="bottom",
            )
        x += pitch


def _draw_strains(
    ax: Axes, spec: FigureSpec, pg: _Page, grid: Grid
) -> tuple[list[Box], list[dict]]:
    """Strain labels right-aligned just left of the tree ink on their row, never overlapping."""
    g, t, lay = pg.g, spec.tree, spec.layout
    font = FontProperties(family="DejaVu Sans", size=g.strain_font)
    width = _text_width(font)
    row_of = {t.leaf_id[i]: r for r, i in enumerate(lay.leaf_nodes)}
    boxes: list[Box] = []
    report = []
    for leaf_id, text in spec.strains:
        r = row_of.get(leaf_id)
        if r is None:
            report.append({"leaf": leaf_id, "text": text, "drawn": False})
            continue
        y = float(pg.row_y(r))
        near = grid.ink[max(0, int(y) - 3) : int(y) + 4, : int(pg.tree_right)].any(axis=0)
        cols = np.flatnonzero(near)
        x_right = max(
            float(cols.min() - 3) if len(cols) else pg.nx[lay.leaf_nodes[r]] - 3, 2 + width(text)
        )
        h = g.strain_font * 1.1
        box: Box = (x_right - width(text), y - h / 2, x_right, y + h / 2)
        for dy in (0, -h, h, -2 * h, 2 * h, -3 * h, 3 * h):
            box = (x_right - width(text), y + dy - h / 2, x_right, y + dy + h / 2)
            while grid.cost(box) and box[0] > 2:  # slide left off the tree lines
                box = (box[0] - 2, box[1], box[2] - 2, box[3])
            x_right = box[2]
            if not any(
                min(box[2], o[2]) > max(box[0], o[0]) and min(box[3], o[3]) > max(box[1], o[1])
                for o in boxes
            ):
                break
        ax.text(box[2], (box[1] + box[3]) / 2, text, ha="right", va="center", fontproperties=font)
        boxes.append(box)
        report.append({"leaf": leaf_id, "text": text, "drawn": True, "row": int(r)})
    missing = [s["leaf"] for s in report if not s["drawn"]]
    if missing:
        raise ValueError(f"{len(missing)} strain label(s) name no drawn leaf: {missing[:5]}")
    return boxes, report


def _draw_labels(ax: Axes, spec: FigureSpec, pg: _Page, grid: Grid, obstacles: list[Box]):
    g = pg.g
    font = FontProperties(family="DejaVu Sans Mono", size=g.label_font)
    for o in obstacles:
        grid.add_box(o)
    items = [
        (lab.node, "\n".join(lab.subs), (float(pg.nx[lab.node]), float(pg.ny[lab.node])))
        for lab in spec.labels
    ]
    search = Search(left=2.0, top=g.top - 4, bottom=pg.bottom + 2)
    placed = place_labels(items, grid, _text_width(font), g.label_font * 1.2, search=search)
    for p in placed:
        x0, y0, x1, y1 = p.box
        ax.text(
            x1 - 1,
            (y0 + y1) / 2,
            p.text,
            ha="right",
            va="center",
            fontproperties=font,
            color="#4d4d4d",
            linespacing=1.15,
        )
        ex, ey = leader_end(p.box)
        ax.plot([ex, p.anchor[0]], [ey, p.anchor[1]], color="#4d4d4d", lw=0.35)
    return placed, placement_metrics(placed, grid)


def _draw_key(ax: Axes, spec: FigureSpec, pg: _Page) -> None:
    """Continent inset (doubles as the continent legend), or a clade key in the clade style."""
    g = pg.g
    if spec.colour_by == "clade":
        scheme = clade_colour_scheme(spec)
        for k, (clade, colour) in enumerate(scheme.items()):
            y = g.bottom - 12 * (len(scheme) - k)
            ax.add_patch(plt.Rectangle((22, y), 8, 8, color=colour))
            ax.text(34, y + 4, clade, fontsize=7, va="center")
        return
    data = json.loads(resources.files(__package__).joinpath("continents.json").read_text())
    x0, y0, scale = 22.0, g.bottom - 80, 140.0 / data["size"][0]
    for continent in INSET_CONTINENTS:
        path = data["paths"][continent]
        polys: list[list[tuple[float, float]]] = []
        cur: list[tuple[float, float]] = []
        for k in range(0, len(path) - 1, 2):
            px, py = path[k], path[k + 1]
            if px < 0 and len(cur) > 2:  # negative x marks a move-to
                polys.append(cur)
            if px < 0:
                cur = []
            cur.append((x0 + abs(px) * scale, y0 + abs(py) * scale))
        if len(cur) > 2:
            polys.append(cur)
        ax.add_collection(
            PolyCollection(polys, facecolors=CONTINENT_COLOURS[continent], edgecolors="none")
        )


def render(spec: FigureSpec, pdf_path: Path) -> Drawn:
    """Write the PDF and return what was drawn."""
    g = spec.geometry
    pg = _Page(spec)
    fig = plt.figure(figsize=(g.width / 72, g.height / 72))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, g.width)
    ax.set_ylim(g.height, 0)
    ax.axis("off")
    grid = Grid(g.width, g.height, cell=1.0)
    colours = row_colours(spec)

    _draw_tree(ax, spec, pg, grid, colours)
    _draw_marks(ax, spec, pg)
    matrix_right = _draw_matrix(ax, spec, pg, colours)
    brackets_right = _draw_brackets(ax, spec, pg, matrix_right + g.clade_gap)
    _draw_hz_rules(ax, spec, pg, brackets_right)
    _draw_dash_bars(ax, spec, pg, brackets_right)
    strain_boxes, strain_report = _draw_strains(ax, spec, pg, grid)
    placed, metrics = _draw_labels(ax, spec, pg, grid, strain_boxes)
    _draw_key(ax, spec, pg)
    ax.text(g.tree_left + 30, 16, spec.title, fontsize=15, ha="left", va="center")

    fig.savefig(pdf_path, metadata={"CreationDate": None, "Creator": "acmacs-f"})
    plt.close(fig)
    return Drawn(placed, metrics, strain_report, continents_not_in_legend(spec))


def with_centre_column(g: Geometry, width: float = 14.0) -> Geometry:
    """Geometry for the centre-marked variant: the tree gives up ``width`` points."""
    return replace(g, centre_col=width)
