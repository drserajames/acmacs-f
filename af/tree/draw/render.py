"""Draw the report tree figure as a vector PDF (matplotlib).

Page geometry follows the past report trees (701 x 1000 pt): the tree, an optional column for
one centre's antigens, the month x leaf matrix, the clade brackets, then the aa dash-bars.
Everything is placed in page points. The function returns what it drew, for the I7 JSON and the
draw report; the caller writes those.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping, Sequence
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
from matplotlib.patches import ArrowStyle, FancyArrowPatch  # noqa: E402
from matplotlib.textpath import TextPath  # noqa: E402

from .aa_labels import AALabel  # noqa: E402
from .layout import Layout  # noqa: E402
from .model import DrawTree  # noqa: E402
from .place import (  # noqa: E402
    Box,
    Grid,
    Placed,
    Search,
    Weights,
    leader_end,
    place_labels,
    placement_metrics,
    segment_hits_box,
)
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
    """Figure layout in figure units (1/72 in before the report scales the page).

    Defaults are the Sep 2026 report's H3 page (ae tal-draw), measured from the shipped PDF;
    a round sets the per-subtype values (H1's wider canvas, each tree's width, text sizes) in
    its config. Text sizes are in figure units too.
    """

    width: float = 700.952
    height: float = 1000.0
    top: float = 25.0
    bottom: float = 975.0
    tree_left: float = 22.0
    tree_right: float = 372.0
    centre_col: float = 0.0  # width of the centre-marks column; 0 = none
    ts_left: float = 382.0
    ts_slot: float = 7.5
    clade_gap: float = 11.4  # matrix's right edge -> first bracket column (+2)
    clade_slot: float = 15.0  # bracket column pitch
    clade_name_offset: float = 4.5  # bracket -> clade name baseline
    # Clear space between two names stacked in one bracket column. Too little, and a column of
    # names reads as one string (R5 review: two sibling sub-clades at 6); too much pushes a
    # name past the end of its small band (another tree's pair at 10).
    clade_name_gap: float = 6.0
    bracket_width: float = 1.0
    bars_gap: float = 16.3  # last bracket column -> first dash bar
    bar_width: float = 8.7
    bar_gap: float = 5.8
    row_stroke: float = 0.15  # matrix ticks and dash-bar rows (a row is thinner on big trees)
    branch_stroke: float = 0.5  # branch line width, as a fraction of the row height
    tick_start: float = 0.25  # matrix tick: where it starts in its month slot, and its length
    tick_length: float = 0.5
    month_gap: float = 1.5  # month labels end this far above the matrix (start below it)
    rule_colour: str = "#bebebe"  # separators between clade bands
    label_font: float = 9.5  # aa-substitution labels
    strain_font: float = 6.8
    clade_font: float = 10.0  # clade names beside the brackets (per clade: FigureSpec)
    month_font: float = 5.8  # month labels above and below the matrix
    legend_font: float = 6.1  # dash-bar legend
    title_font: float = 15.0
    inset_gap: float = 7.5  # continent inset: its lowest edge sits this far above the last row
    # Fixed scales, for figures compared side by side (e.g. a tree before and after pruning):
    # the branch length drawn across the tree's width, and the rows the page height holds.
    # None = fit this tree.
    x_max: float | None = None
    row_capacity: int | None = None
    # Shipped placements (R4 review, 18 Oct 2026); None keeps af's own rule.
    # Strain (vaccine) labels left-aligned at this x, in the margin where no substitution leader
    # runs (leaders always run right, to their node); None: right-aligned just left of the tree.
    strain_x: float | None = None
    # A margin strain label whose own row is taken moves off it in steps of this (None: the
    # label's height), so two strains on near rows sit this far apart.
    strain_step: float | None = None
    # Margin strain labels placed before the aa labels, which then keep off them. For a tree
    # whose strain labels sit at fixed (shipped) offsets: placed last, an automatic aa label
    # takes their spot and pushes them off their rows (R7, B/Vic).
    strains_first: bool = False
    # The continent inset's preferred corner (left x, bottom y): the clear spot nearest to it is
    # taken, never one over ink; None: the lowest, then leftmost clear spot. And its width.
    inset_left: float | None = None
    inset_bottom: float | None = None
    inset_width: float = 140.0
    # Dash-bar legend rows: the first row's bottom (None: 16 above the rows) and the row pitch.
    legend_bottom: float | None = None
    legend_pitch: float = 6.5
    # No aa label starts left of this x. With margin strain labels, set it right of their
    # column, so the margin is theirs alone as shipped (R5 review: aa labels from x 3).
    label_left: float = 2.0
    # The label placer's weights (place.Weights) for this tree, where they differ from its
    # defaults (None keeps a default): a dense tree's labels sit beside their branches, over
    # thin leaf lines, only when ink costs less and leader length more (R6 review, H3).
    label_ink_weight: float | None = None
    label_length_weight: float | None = None
    label_flat_weight: float | None = None
    # Clade bracket arrowheads, in points (FancyArrowPatch head length and half-width), and
    # the largest share of its bracket a head may take (None: no cap); past it the head
    # shrinks, so a short bracket's two heads never meet (R6 review: a bow-tie on J.1.1).
    bracket_head_length: float = 5.76
    bracket_head_width: float = 1.44
    bracket_head_share: float | None = None
    # How far each bracket end is moved out, so the drawn tip lands on its band's edge:
    # matplotlib stops an arrowhead's tip short of its end point (about 1.2 points for these
    # heads at a 1-point line), which left adjacent brackets' tips 2 points apart where the
    # shipped pages' touch (R6 review). Scaled down with the head on a short bracket.
    bracket_tip_extend: float = 0.0


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
    # Clade name -> label size, for clades drawn at other than ``geometry.clade_font`` (the
    # shipped H1 page draws the D sub-clades smaller). Every name must be a shown clade.
    clade_label_sizes: dict[str, float] = field(default_factory=dict)
    # Every drawn leaf's name as text at its row: too small to read (a row is ~0.01 pt on a big
    # tree) but searchable, so a reader can find a strain on the page. The Sep 2026 report's H1
    # and H3 tree pages carried the same layer.
    leaf_names: bool = False
    drawn_title: str | None = None  # drawn instead of ``title`` ("" = no title); see FigureConfig
    # Bracket text per clade where it differs from the name (FigureConfig.clade_display).
    clade_display: dict[str, str] = field(default_factory=dict)
    # Drawn in a round's figure fonts (FigureConfig.fonts, af.util.fonts): text uses the rc sans
    # and mono families. False: DejaVu Sans and DejaVu Sans Mono, as before the figure fonts.
    figure_fonts: bool = False
    # Strain label -> how far below its row it sits, in figure units (the shipped .tal's per
    # strain offsets: they set the labels' order where two strains are on near rows).
    strain_dy: dict[str, float] = field(default_factory=dict)
    # Label index (into ``labels``) -> (dx, dy, x_by): labels pinned by hand
    # (figure.LabelPin); the placer keeps them where they are and places the rest around them.
    label_pins: dict[int, tuple[float, float, str]] = field(default_factory=dict)


@dataclass
class Drawn:
    """What the renderer drew, for I7 and the draw report."""

    placed: list  # place.Placed
    label_metrics: dict
    strains: list[dict]
    continents_not_in_legend: dict[str, int]  # value -> rows drawn grey for it
    leaf_names: int | None = None  # names written (None: layer off)
    inset: Box | None = None  # continent inset (x0, y0, x1, y1), figure units; None: clade key


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
        """A row is far thinner than a printable line on a big tree; never go below the
        geometry's row stroke (the shipped pages draw 0.15 units whatever the row count)."""
        return max(self.g.row_stroke, self.row_h)


def _families(spec: FigureSpec) -> tuple[str, str]:
    """(sans, mono) family names: the figure fonts' rc families, else DejaVu."""
    return ("sans-serif", "monospace") if spec.figure_fonts else ("DejaVu Sans", "DejaVu Sans Mono")


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
    # Hairlines in proportion to the row height, as on the shipped pages (0.005 units on a
    # 100,000-leaf tree): a fixed width painted big trees solid black.
    width = min(spec.geometry.branch_stroke * pg.row_h, 0.5)  # small trees: a normal line
    ax.add_collection(LineCollection(segs, colors=seg_colours, linewidths=width))
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
        # Rotated to read downwards: the top label ends just above the matrix, the bottom one
        # starts just below it, each centred in its slot, clear of the slot's lines.
        for yy, ha in ((g.top - g.month_gap, "right"), (pg.bottom + g.month_gap, "left")):
            ax.text(
                x,
                yy,
                text,
                rotation=-90,
                ha=ha,
                va="center",
                fontsize=g.month_font,
                rotation_mode="anchor",
            )
    rows = np.flatnonzero(ts.in_window)
    x0 = g.ts_left + (ts.column[rows] + g.tick_start) * g.ts_slot
    ys = pg.row_y(rows)
    segs = [((a, y), (a + g.tick_length * g.ts_slot, y)) for a, y in zip(x0, ys, strict=True)]
    ax.add_collection(LineCollection(segs, colors=list(colours[rows]), linewidths=pg.dash_width))
    return right


def _draw_brackets(ax: Axes, spec: FigureSpec, pg: _Page, left: float) -> dict[int, float]:
    """One double-headed arrow per band, filled heads touching the band's edges (the shipped
    style: 5.76 x 2.88 units). Returns, per band-edge row, the x of the outermost bracket that
    ends there, so the separator rule at that row can stop at it."""
    g, sel = pg.g, spec.selection
    names: list[tuple[float, float, str]] = []
    edges: dict[int, float] = {}
    style = ArrowStyle("<|-|>", head_length=g.bracket_head_length, head_width=g.bracket_head_width)
    for cs in sel.shown:
        x = left + sel.slots[cs.clade] * g.clade_slot + 2
        for b in cs.bands:
            y0, y1 = pg.row_edge(b.first), pg.row_edge(b.last + 1)
            scale = _head_scale(g, y1 - y0)
            tip = g.bracket_tip_extend * scale
            ax.add_patch(
                FancyArrowPatch(
                    (x, y1 + tip),
                    (x, y0 - tip),
                    arrowstyle=style,
                    mutation_scale=scale,
                    lw=g.bracket_width,
                    color="black",
                    shrinkA=0,
                    shrinkB=0,
                )
            )
            for r in (b.first, b.last + 1):
                edges[r] = max(edges.get(r, x), x)
            names.append((x + g.clade_name_offset, (y0 + y1) / 2, cs.clade))
    _place_clade_names(
        ax, names, g.clade_font, spec.clade_label_sizes, spec.clade_display, g.clade_name_gap
    )
    return edges


def _head_scale(g: Geometry, span: float) -> float:
    """1, or less where the bracket is too short for two whole heads (bracket_head_share)."""
    if g.bracket_head_share is None or g.bracket_head_length <= g.bracket_head_share * span:
        return 1
    return g.bracket_head_share * span / g.bracket_head_length


def _place_clade_names(
    ax: Axes,
    names: list[tuple[float, float, str]],
    size: float,
    sizes: Mapping[str, float] | None = None,
    display: Mapping[str, str] | None = None,
    gap: float = 6.0,
) -> None:
    """Write each clade's name beside its bracket, centred on the band.

    Two small bands next to each other in one column would print their names on top of each
    other; a name that would overlap the one above slides down just past it, with a thin
    leader back to its band's middle, so every name stays readable, ``gap`` clear of the one
    above.
    """
    sizes, display = sizes or {}, display or {}
    widths = {s: _text_width(FontProperties(size=s)) for s in {size, *sizes.values()}}
    last_end: dict[float, float] = {}
    for x, mid, clade in sorted(names, key=lambda t: (t[0], t[1])):
        text_size = sizes.get(clade, size)
        text = display.get(clade, clade)
        half = widths[text_size](text) / 2
        clear = half + gap / 2
        centre = max(mid, last_end.get(x, -1e9) + clear)
        last_end[x] = centre + clear
        if centre != mid:
            ax.plot([x - 3, x], [mid, centre - clear + (gap / 2 - 2)], color="#7f7f7f", lw=0.3)
        ax.text(
            x,
            centre,
            text,
            rotation=-90,
            ha="center",
            va="baseline",
            fontsize=text_size,
            rotation_mode="anchor",
        )


def _draw_hz_rules(
    ax: Axes, spec: FigureSpec, pg: _Page, matrix_right: float, edges: Mapping[int, float]
) -> None:
    """A rule across the matrix at every band edge; where a clade bracket ends at that row the
    rule carries on to that bracket and stops there, as on the shipped pages."""
    g = pg.g
    rows = sorted({h.first for h in spec.hz} | {h.last + 1 for h in spec.hz} | set(edges))
    for r in rows:
        if 0 < r < spec.layout.n_rows:
            y = pg.row_edge(r)
            ax.plot([g.ts_left, edges.get(r, matrix_right)], [y, y], color=g.rule_colour, lw=0.5)


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
        first = g.top - 16 if g.legend_bottom is None else g.legend_bottom
        for k, (text, colour) in enumerate(bar.legend):
            ax.text(
                x + width / 2,
                first + k * g.legend_pitch,
                text,
                color=colour,
                fontsize=g.legend_font,
                ha="center",
                va="bottom",
            )
        x += pitch


def _draw_strains(
    ax: Axes, spec: FigureSpec, pg: _Page, grid: Grid, placed: Sequence[Placed] = ()
) -> tuple[list[Box], list[dict]]:
    """Strain labels right-aligned just left of the tree ink on their row, never overlapping."""
    g, t, lay = pg.g, spec.tree, spec.layout
    font = FontProperties(family=_families(spec)[0], size=g.strain_font)
    width = _text_width(font)
    row_of = {t.leaf_id[i]: r for r, i in enumerate(lay.leaf_nodes)}
    boxes: list[Box] = []
    report = []
    for leaf_id, text in spec.strains:
        r = row_of.get(leaf_id)
        if r is None:
            report.append({"leaf": leaf_id, "text": text, "drawn": False})
            continue
        y = float(pg.row_y(r)) + spec.strain_dy.get(leaf_id, 0.0)
        if g.strain_x is not None:
            at = _margin_strain_box(
                g.strain_x, y, width(text), g.strain_font * 1.1, boxes, grid, placed, g.strain_step
            )
            ax.text(at[0], (at[1] + at[3]) / 2, text, ha="left", va="center", fontproperties=font)
            boxes.append(at)
            report.append({"leaf": leaf_id, "text": text, "drawn": True, "row": int(r)})
            continue
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


def _margin_strain_box(
    x: float,
    y: float,
    w: float,
    h: float,
    boxes: list[Box],
    grid: Grid,
    placed: Sequence[Placed] = (),
    step: float | None = None,
) -> Box:
    """A strain label in the left margin at ``x``: on its row, or the nearest offset (in
    ``step``s, default ``h``) that overlaps no other strain label, no ink, no placed aa label
    and no aa leader. Off a row another strain label sits on, it moves away from that label
    first, so the labels keep their rows' order. Placed after the aa labels, so they keep the
    spots they would have had (R4 review: margin labels placed first pushed aa labels into each
    other's leaders). None clear is an error, not an overlap."""

    def hits(box: Box, others: Iterable[Box]) -> bool:
        return any(
            min(box[2], o[2]) > max(box[0], o[0]) and min(box[3], o[3]) > max(box[1], o[1])
            for o in others
        )

    on_row = [o for o in boxes if hits((x, y - h / 2, x + w, y + h / 2), [o])]
    away = 1 if any((o[1] + o[3]) / 2 <= y for o in on_row) else -1  # down if one is above
    for k in range(13):
        dy = (k + 1) // 2 * (h if step is None else step) * (away if k % 2 == 1 else -away)
        box: Box = (x, y + dy - h / 2, x + w, y + dy + h / 2)
        clear = not hits(box, [*boxes, *(p.box for p in placed)])
        crossed = any(segment_hits_box(leader_end(p.box), p.anchor, box) for p in placed)
        if clear and not crossed and not grid.cost(box):
            return box
    raise ValueError(f"no clear margin space for a strain label at x {x}, y {y:.0f}")


def _draw_leaf_names(ax: Axes, spec: FigureSpec, pg: _Page) -> int:
    """One text per drawn row, "<name> <leaf id>" (names repeat across passages; the id makes
    each search hit one leaf), left-aligned just right of the leaf's tip at the row's height.

    Fully transparent: matplotlib cannot write text under 1 pt (FreeType), and 1 pt names on
    rows ~0.01 pt apart paint the tree over (7.6% of an H1 page's pixels changed). Transparent
    text is still found by a viewer's search, which highlights the row. Drawn last and kept out
    of the ink grid: it must not move any visible label."""
    t, lay = spec.tree, spec.layout
    for r, i in enumerate(lay.leaf_nodes):
        ax.text(
            float(pg.nx[i]) + 0.5,
            float(pg.row_y(r)),
            f"{t.name[i]} {t.leaf_id[i]}",
            fontsize=1.0,
            family=_families(spec)[0],
            alpha=0.0,
            ha="left",
            va="center",
        )
    return len(lay.leaf_nodes)


def _draw_labels(ax: Axes, spec: FigureSpec, pg: _Page, grid: Grid, obstacles: list[Box]):
    g = pg.g
    font = FontProperties(family=_families(spec)[1], size=g.label_font)
    for o in obstacles:
        grid.add_box(o)
    items = [
        (lab.node, "\n".join(lab.subs), (float(pg.nx[lab.node]), float(pg.ny[lab.node])))
        for lab in spec.labels
    ]
    # The candidate offsets were tuned for 6.8-unit labels; bigger labels also get the same
    # search in proportion (the tuned offsets stay, so a label near the margin keeps its
    # tight positions).
    k = max(1.0, g.label_font / 6.8)
    base = Search()
    search = Search(
        dxs=sorted({*base.dxs, *(d * k for d in base.dxs)}),
        dys=sorted({*base.dys, *(d * k for d in base.dys)}, key=lambda d: (abs(d), d)),
        left=g.label_left,
        top=g.top - 4,
        bottom=pg.bottom + 2,
    )
    width, line_height = _text_width(font), g.label_font * 1.2
    fixed = {k: _pinned_box(spec, pg, k, width, line_height) for k in spec.label_pins}
    placed = place_labels(
        items, grid, width, line_height, search=search, weights=_weights(g), fixed=fixed or None
    )
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


def _pinned_box(
    spec: FigureSpec, pg: _Page, k: int, width: Callable[[str], float], line_height: float
) -> Box:
    """A pinned label's box: its top-left at the node point plus the pin's offset times the
    page (ae's convention), sized as the placer sizes every label. The text is drawn one unit
    in from the box's right edge, so the box starts one unit left of ae's text."""
    g, lab = pg.g, spec.labels[k]
    dx, dy, x_by = spec.label_pins[k]
    bw = max(width(t) for t in lab.subs) + 2
    bh = line_height * len(lab.subs) + 1
    x0 = float(pg.nx[lab.node]) + dx * (g.height if x_by == "height" else g.width) - 1
    y0 = float(pg.ny[lab.node]) + dy * g.height
    return (x0, y0, x0 + bw, y0 + bh)


def _weights(g: Geometry) -> Weights | None:
    """The placer's weights with this tree's own values, or None for its defaults."""
    given = {
        "ink": g.label_ink_weight,
        "length": g.label_length_weight,
        "flat": g.label_flat_weight,
    }
    own: dict[str, float] = {k: v for k, v in given.items() if v is not None}
    return Weights(**own) if own else None


class InsetPlacementError(ValueError):
    """No clear space on the tree side of the page for the continent inset."""


def _inset_polygons(
    width: float = 140.0,
) -> tuple[dict[str, list[list[tuple[float, float]]]], float, float]:
    """Continent outlines scaled to the inset's width, origin at the inset's corner; plus the
    inset's width and height."""
    data = json.loads(resources.files(__package__).joinpath("continents.json").read_text())
    scale = width / data["size"][0]
    out: dict[str, list[list[tuple[float, float]]]] = {}
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
            cur.append((abs(px) * scale, abs(py) * scale))
        if len(cur) > 2:
            polys.append(cur)
        out[continent] = polys
    return out, data["size"][0] * scale, data["size"][1] * scale


def place_inset(
    grid: Grid,
    w: float,
    h: float,
    lowest: float,
    right: float,
    pad: float = 2.0,
    near: tuple[float, float] | None = None,
) -> Box:
    """The lowest, then leftmost, ``w`` x ``h`` box with no ink under it (``pad`` around it),
    whose bottom is at most ``lowest`` and right edge at most ``right``.

    A fixed corner put the inset over the tree whenever the tree's lower clades reached the
    left margin (the af H1 page drew 3,723 branch strokes under it); the shipped pages always
    found it clear space beside the lower tree. No clear space at all is an error, never an
    overlap drawn silently.
    """
    c = grid.cell
    occupied = (grid.ink > 0).astype(np.int64)
    sat = np.zeros((occupied.shape[0] + 1, occupied.shape[1] + 1), dtype=np.int64)
    sat[1:, 1:] = occupied.cumsum(0).cumsum(1)
    bw, bh = int(np.ceil((w + 2 * pad) / c)), int(np.ceil((h + 2 * pad) / c))
    i_last = min(int((lowest + pad) / c) - bh, occupied.shape[0] - bh)
    j_last = min(int((right + pad) / c) - bw, occupied.shape[1] - bw)
    if i_last < 0 or j_last < 0:
        raise InsetPlacementError(f"no room for a {w:.0f} x {h:.0f} inset")
    i = np.arange(i_last + 1)[:, None]
    j = np.arange(j_last + 1)[None, :]
    inked = sat[i + bh, j + bw] - sat[i, j + bw] - sat[i + bh, j] + sat[i, j]
    clear_rows = np.flatnonzero((inked == 0).any(axis=1))
    if not len(clear_rows):
        raise InsetPlacementError(f"no clear {w:.0f} x {h:.0f} space left of x {right:.0f}")
    if near is not None:  # the clear corner nearest the wanted one (left x, top y)
        ti, tj = (near[1] - pad) / c, (near[0] - pad) / c
        distance = np.where(inked == 0, (i - ti) ** 2 + (j - tj) ** 2, np.inf)
        bi, bj = (int(k) for k in np.unravel_index(int(np.argmin(distance)), distance.shape))
    else:
        bi = int(clear_rows[-1])
        bj = int(np.flatnonzero(inked[bi] == 0)[0])
    x0, y0 = bj * c + pad, bi * c + pad
    return (x0, y0, x0 + w, y0 + h)


def _draw_key(ax: Axes, spec: FigureSpec, pg: _Page, grid: Grid) -> Box | None:
    """Continent inset (doubles as the continent legend), or a clade key in the clade style.

    The inset goes in clear space beside the lower tree and is then inked on the grid, so the
    strain and aa labels placed after it keep off it."""
    g = pg.g
    if spec.colour_by == "clade":
        scheme = clade_colour_scheme(spec)
        for k, (clade, colour) in enumerate(scheme.items()):
            y = g.bottom - 12 * (len(scheme) - k)
            ax.add_patch(plt.Rectangle((22, y), 8, 8, color=colour))
            ax.text(34, y + 4, clade, fontsize=7, va="center")
        return None
    polys, w, h = _inset_polygons(g.inset_width)
    near = None
    if g.inset_left is not None and g.inset_bottom is not None:
        near = (g.inset_left, g.inset_bottom - h)
    box = place_inset(grid, w, h, lowest=pg.bottom - g.inset_gap, right=pg.tree_right, near=near)
    for continent, parts in polys.items():
        ax.add_collection(
            PolyCollection(
                [[(box[0] + x, box[1] + y) for x, y in poly] for poly in parts],
                facecolors=CONTINENT_COLOURS[continent],
                edgecolors="none",
            )
        )
    grid.add_box(box)
    return box


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
    bracket_left = matrix_right + g.clade_gap
    edges = _draw_brackets(ax, spec, pg, bracket_left)
    _draw_hz_rules(ax, spec, pg, matrix_right, edges)
    last_column = bracket_left + max(spec.selection.slots.values(), default=0) * g.clade_slot + 2
    _draw_dash_bars(ax, spec, pg, last_column + g.bars_gap)
    inset = _draw_key(ax, spec, pg, grid)
    if g.strain_x is None or g.strains_first:  # strain labels first: the aa labels keep off them
        strain_boxes, strain_report = _draw_strains(ax, spec, pg, grid)
        placed, metrics = _draw_labels(ax, spec, pg, grid, strain_boxes)
    else:  # margin strain labels go where the aa labels and their leaders leave room
        placed, metrics = _draw_labels(ax, spec, pg, grid, [])
        strain_boxes, strain_report = _draw_strains(ax, spec, pg, grid, placed)
    # The shipped pages start the title just left of the tree's root (H1 4, H3 4 units).
    title = spec.title if spec.drawn_title is None else spec.drawn_title
    if title:
        ax.text(g.tree_left - 4, 16, title, fontsize=g.title_font, ha="left", va="center")
    names = _draw_leaf_names(ax, spec, pg) if spec.leaf_names else None

    # Without figure fonts the name layer is written Type 42 (a Unicode map, so viewers can search
    # it), and only with the layer, so other figures keep their bytes. The figure fonts set their
    # own type (af.util.fonts.pdf_fonttype; TeX Gyre is Type 3, whose
    # text stays searchable): forcing 42 there only makes readers flag a font type mismatch.
    searchable = spec.leaf_names and not spec.figure_fonts
    with plt.rc_context({"pdf.fonttype": 42} if searchable else {}):
        fig.savefig(pdf_path, metadata={"CreationDate": None, "Creator": "acmacs-f"})
    plt.close(fig)
    return Drawn(placed, metrics, strain_report, continents_not_in_legend(spec), names, inset)


def with_centre_column(g: Geometry, width: float = 14.0) -> Geometry:
    """Geometry for the centre-marked variant: the tree gives up ``width`` points."""
    return replace(g, centre_col=width)
