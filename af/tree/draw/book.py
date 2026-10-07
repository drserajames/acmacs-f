"""The leaf book: the report tree over many pages, one readable row per drawn leaf.

Why: the report figure fits ~100,000 leaves on one page, so a leaf there is ~0.01 pt and no
name can be read. The book draws the same rows (same layout, hides, overrides and lettered
bands as the figure, via :func:`~af.tree.draw.figure.arrange`) at a readable pitch: each row
gives the leaf's name, leaf id, collection date and clade, with its continent colour, beside
its part of the tree, and each page brackets the lettered bands it shows. Index pages first
list every lettered band and every leaf clade with the pages they run over. Text is embedded
with Unicode maps (Type 42 fonts), so the PDF is searchable.

Each page zooms the branch-length axis to its own leaves (one scale for the whole tree left a
page's ~127 close relatives in a few points of width), with a scale bar in the tree's depth
unit. Ancestors to the left of the window are cut at its left edge: their lines continue on
other pages.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("pdf")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402

from af.store import StoreRef  # noqa: E402

from .figure import FigureConfig, arrange  # noqa: E402
from .layout import Layout  # noqa: E402
from .model import DrawTree  # noqa: E402
from .render import CONTINENT_COLOURS, UNKNOWN_COLOUR  # noqa: E402
from .sections import HzBand  # noqa: E402

MONO = "DejaVu Sans Mono"
CHAR_W = 0.602  # DejaVu Sans Mono advance width per point of font size


@dataclass(frozen=True)
class BookGeometry:
    """A4 portrait, in points. ``row_pitch`` sets rows per page; the text column takes what
    the widest row needs at ``font`` and the tree gets the rest."""

    width: float = 595.28
    height: float = 841.89
    top: float = 46.0
    bottom: float = 28.0
    left: float = 22.0
    right: float = 22.0
    row_pitch: float = 6.0
    font: float = 4.6
    bracket_width: float = 54.0
    min_tree_width: float = 150.0

    @property
    def rows_per_page(self) -> int:
        return int((self.height - self.top - self.bottom) // self.row_pitch)


class BookLayoutError(ValueError):
    """The rows do not fit the page at this font (the tree would get too little width)."""


def _row_text(tree: DrawTree, i: int, widths: tuple[int, int, int]) -> str:
    name = tree.name[i] or ""
    leaf_id = tree.leaf_id[i] or ""
    date = tree.date[i] or ""
    clade = tree.clade[i] or ""
    return f"{name:<{widths[0]}}  {leaf_id:<{widths[1]}}  {date:<{widths[2]}}  {clade}"


def _columns(tree: DrawTree, rows: Sequence[int]) -> tuple[tuple[int, int, int], int]:
    """Column widths (characters) for name, leaf id and date; and the widest row."""
    w = (
        max(len(tree.name[i] or "") for i in rows),
        max(len(tree.leaf_id[i] or "") for i in rows),
        max(len(tree.date[i] or "") for i in rows),
    )
    longest_clade = max(len(tree.clade[i] or "") for i in rows)
    return w, w[0] + w[1] + w[2] + longest_clade + 6


def page_ranges(n_rows: int, per_page: int) -> list[tuple[int, int]]:
    """Inclusive row ranges, one per tree page."""
    return [(r, min(r + per_page, n_rows) - 1) for r in range(0, n_rows, per_page)]


def _index(
    tree: DrawTree, layout: Layout, hz: list[HzBand], pages: list[tuple[int, int]], first: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Lettered bands and leaf clades with the (1-based) pages they run over."""
    starts = np.array([a for a, _ in pages])

    def page_of(row: int) -> int:
        return int(np.searchsorted(starts, row, side="right")) + first - 1

    bands = [
        {
            "letter": h.letter,
            "clade": h.clade,
            "rows": h.last - h.first + 1,
            "first_page": page_of(h.first),
            "last_page": page_of(h.last),
        }
        for h in hz
    ]
    rows_of: dict[str, list[int]] = {}
    for r, i in enumerate(layout.leaf_nodes):
        rows_of.setdefault(tree.clade[i] or "", []).append(r)
    clades = sorted(
        (
            {
                "clade": c or "(none)",
                "leaves": len(rs),
                "first_page": page_of(rs[0]),
                "last_page": page_of(rs[-1]),
            }
            for c, rs in rows_of.items()
        ),
        key=lambda x: (x["first_page"], x["clade"]),
    )
    return bands, clades


def _index_lines(title: str, bands: list[dict], clades: list[dict], n_rows: int) -> list[str]:
    out = [f"{title}: every drawn leaf, {n_rows:,} rows", ""]
    out.append("Lettered bands (as on the report tree page)")
    out.append(f"  {'band':<5} {'clade':<14} {'rows':>8}   pages")
    for b in bands:
        out.append(
            f"  {b['letter']:<5} {b['clade']:<14} {b['rows']:>8,}   "
            f"{b['first_page']}-{b['last_page']}"
        )
    out += ["", "Leaf clades (each leaf's assigned clade), by first page"]
    out.append(f"  {'clade':<20} {'leaves':>8}   pages")
    for c in clades:
        out.append(f"  {c['clade']:<20} {c['leaves']:>8,}   {c['first_page']}-{c['last_page']}")
    return out


def _new_page(g: BookGeometry) -> tuple[Any, Any]:
    fig = plt.figure(figsize=(g.width / 72, g.height / 72))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, g.width)
    ax.set_ylim(g.height, 0)
    ax.axis("off")
    return fig, ax


def make_book(
    tree: DrawTree,
    parents: Mapping[str, str | None],
    config: FigureConfig,
    pdf: Path,
    inputs: Mapping[str, str],
    flags: Mapping[str, list[str]] | None = None,
    store_refs: Sequence[StoreRef] = (),
    geometry: BookGeometry | None = None,
    depth_unit: str = "",
) -> dict[str, Any]:
    """Write the leaf book ``pdf`` and its report (``<stem>.book.json``); return the report."""
    g = geometry or BookGeometry()
    started = time.monotonic()
    layout, _ts, _selection, hz = arrange(tree, parents, config, flags)
    rows = [int(i) for i in layout.leaf_nodes]
    widths, longest = _columns(tree, rows)
    text_w = longest * CHAR_W * g.font
    text_left = g.width - g.right - g.bracket_width - text_w
    tree_right = text_left - 4
    if tree_right - g.left < g.min_tree_width:
        raise BookLayoutError(
            f"rows of {longest} characters at {g.font} pt leave {tree_right - g.left:.0f} pt "
            f"for the tree (< {g.min_tree_width})"
        )
    per_page = g.rows_per_page
    pages = page_ranges(layout.n_rows, per_page)

    bands_index, clades_index = _index(tree, layout, hz, pages, first=0)
    index_lines = _index_lines(config.title, bands_index, clades_index, layout.n_rows)
    lines_per_index_page = int((g.height - g.top - g.bottom) // (g.row_pitch * 1.25))
    n_index = -(-len(index_lines) // lines_per_index_page)
    bands_index, clades_index = _index(tree, layout, hz, pages, first=n_index + 1)
    index_lines = _index_lines(config.title, bands_index, clades_index, layout.n_rows)
    total = n_index + len(pages)

    parent = np.asarray(tree.parent)
    drawn = layout.first_row >= 0
    # vertical connector per internal node: from its first to its last drawn child's row
    lo = np.full(len(tree), np.nan)
    hi = np.full(len(tree), np.nan)
    for node in np.flatnonzero(drawn):
        kids = [c for c in tree.children[node] if layout.first_row[c] >= 0]
        if len(kids) > 1:
            lo[node], hi[node] = layout.node_y[kids[0]], layout.node_y[kids[-1]]
    h_nodes = np.flatnonzero(drawn & (parent >= 0))
    v_nodes = np.flatnonzero(~np.isnan(lo))
    colours = [CONTINENT_COLOURS.get(tree.continent[i] or "", UNKNOWN_COLOUR) for i in rows]

    def y_of(row: np.ndarray | float, first: int) -> np.ndarray:
        return g.top + (np.asarray(row) - first + 0.5) * g.row_pitch

    with (
        plt.rc_context({"pdf.fonttype": 42}),
        PdfPages(pdf, metadata={"CreationDate": None, "Creator": "acmacs-f"}) as out,
    ):
        for k in range(n_index):
            fig, ax = _new_page(g)
            chunk = index_lines[k * lines_per_index_page : (k + 1) * lines_per_index_page]
            for j, line in enumerate(chunk):
                y = g.top + j * g.row_pitch * 1.25
                ax.text(g.left, y, line, family=MONO, fontsize=g.font + 1, va="center")
            _footer(ax, g, config.title, k + 1, total, store_refs)
            out.savefig(fig)
            plt.close(fig)
        for p, (r0, r1) in enumerate(pages):
            fig, ax = _new_page(g)
            nx = _tree_slice(
                ax,
                g,
                tree_right,
                layout,
                parent,
                h_nodes,
                v_nodes,
                lo,
                hi,
                r0,
                r1,
                y_of,
                depth_unit,
            )
            for r in range(r0, r1 + 1):
                i = rows[r]
                y = float(y_of(r, r0))
                ax.plot([nx(layout.node_x[i]), text_left - 2], [y, y], color="#d9d9d9", lw=0.15)
                ax.add_patch(
                    plt.Rectangle(
                        (text_left - 1.2, y - g.row_pitch * 0.35),
                        1.0,
                        g.row_pitch * 0.7,
                        color=colours[r],
                        lw=0,
                    )
                )
                ax.text(
                    text_left + 1,
                    y,
                    _row_text(tree, i, widths),
                    family=MONO,
                    fontsize=g.font,
                    va="center",
                )
            _page_brackets(ax, g, hz, r0, r1, y_of)
            ax.text(
                g.left,
                g.top - 14,
                f"{config.title}   rows {r0 + 1:,}-{r1 + 1:,} of {layout.n_rows:,}",
                fontsize=8,
                va="center",
            )
            _footer(ax, g, config.title, n_index + p + 1, total, store_refs)
            out.savefig(fig)
            plt.close(fig)

    report = {
        "rows": layout.n_rows,
        "rows_per_page": per_page,
        "index_pages": n_index,
        "tree_pages": len(pages),
        "pages": total,
        "hidden": layout.hidden,
        "bands": bands_index,
        "clades": clades_index,
        "continents_not_in_legend": _unknown_continents(tree, rows),
        "font_pt": g.font,
        "x_scale": "per page, scale bar in " + (depth_unit or "tree units"),
        "seconds": round(time.monotonic() - started, 1),
        "inputs": dict(inputs),
        "store_refs": [str(r) for r in store_refs],
    }
    pdf.with_suffix(".book.json").write_text(json.dumps(report, indent=1, default=str))
    return report


def _nice(span: float) -> float:
    """A round scale-bar length near a fifth of ``span``: 1, 2 or 5 times a power of ten."""
    target = span / 5
    base = 10 ** np.floor(np.log10(target))
    return float(max(m * base for m in (1, 2, 5) if m * base <= target))


def _tree_slice(
    ax, g, tree_right, layout, parent, h_nodes, v_nodes, lo, hi, r0, r1, y_of, depth_unit
) -> Any:
    """The branches crossing rows r0..r1, zoomed to the nodes on these rows; returns the page's
    branch-length -> x mapping. Horizontal edges into nodes on the rows, and the part of each
    vertical connector inside them (it continues on the neighbouring page). An edge whose
    parent lies left of the window starts at the window's left edge."""
    top, bottom = r0 - 0.5, r1 + 0.5
    y, x = layout.node_y, layout.node_x
    h = h_nodes[(y[h_nodes] >= top) & (y[h_nodes] <= bottom)]
    x_lo, x_hi = float(x[h].min()), float(x[h].max())
    span = max(x_hi - x_lo, 1e-12)
    x_lo -= 0.02 * span
    span *= 1.02
    width = tree_right - g.left

    def nx(v: Any) -> Any:
        return g.left + (np.maximum(np.asarray(v), x_lo) - x_lo) / span * width

    segs = [((nx(x[parent[i]]), y_of(y[i], r0)), (nx(x[i]), y_of(y[i], r0))) for i in h]
    v = v_nodes[(hi[v_nodes] >= top) & (lo[v_nodes] <= bottom) & (x[v_nodes] >= x_lo)]
    segs += [
        ((nx(x[i]), y_of(max(lo[i], top), r0)), (nx(x[i]), y_of(min(hi[i], bottom), r0))) for i in v
    ]
    ax.add_collection(LineCollection(segs, colors="black", linewidths=0.3))
    bar = _nice(span)
    x0, yb = g.left, g.top - 5
    ax.plot([x0, x0 + bar / span * width], [yb, yb], color="black", lw=0.6)
    ax.text(
        x0 + bar / span * width + 3, yb, f"{bar:g} {depth_unit}".rstrip(), fontsize=5, va="center"
    )
    return nx


def _page_brackets(ax, g: BookGeometry, hz: list[HzBand], r0: int, r1: int, y_of) -> None:
    """Each lettered band on the page as a bracket with its letter and clade; an open end
    (no tick) where the band continues on the previous or next page."""
    x = g.width - g.right - g.bracket_width + 6
    for h in hz:
        if h.last < r0 or h.first > r1:
            continue
        a, b = max(h.first, r0), min(h.last, r1)
        ya, yb = float(y_of(a - 0.4, r0)), float(y_of(b + 0.4, r0))
        ax.plot([x, x], [ya, yb], color="black", lw=0.6)
        if h.first >= r0:
            ax.plot([x - 3, x], [ya, ya], color="black", lw=0.6)
        if h.last <= r1:
            ax.plot([x - 3, x], [yb, yb], color="black", lw=0.6)
        label = f"{h.letter} {h.clade}" + ("" if h.first >= r0 else " (cont.)")
        ax.text(x + 3, (ya + yb) / 2, label, fontsize=6, va="center", rotation=90)


def _footer(ax, g: BookGeometry, title: str, page: int, total: int, refs) -> None:
    ref = f"   tree {refs[0]}" if refs else ""
    ax.text(
        g.left,
        g.height - g.bottom / 2,
        f"{title} leaf book{ref}",
        fontsize=5,
        color="#4d4d4d",
        va="center",
    )
    ax.text(
        g.width - g.right,
        g.height - g.bottom / 2,
        f"{page} / {total}",
        fontsize=6,
        ha="right",
        va="center",
    )


def _unknown_continents(tree: DrawTree, rows: Sequence[int]) -> dict[str, int]:
    out: dict[str, int] = {}
    for i in rows:
        value = tree.continent[i] or ""
        if value not in CONTINENT_COLOURS:
            out[value] = out.get(value, 0) + 1
    return out
