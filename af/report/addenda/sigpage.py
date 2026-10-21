"""Signature pages (addendum 4): a lab's map, one small map per lettered band of the tree.

Each page puts a subtype's report tree (with two extra columns) beside a grid of small maps of
one lab's chart, one per lettered band of the tree (its "sections"). Each small map draws the
whole chart in grey and that band's antigens on top, coloured by date. The page shows where each
part of the tree sits on the map.

What the delivered pages (Sep 2026, ae's port of AD's signature page) draw, and how this module
draws it:

- The tree panel is the report tree figure (af.tree.draw, the round's own config) with two more
  columns after the clade brackets: a grey bar on the rows whose leaf is an antigen in this
  lab's chart, and the lettered bands. It is drawn at report size by ``make_figure``, so it keeps
  its own I7 and draw report, and the page scales it down.
- The sections are the tree's own lettered bands (``hz``), the ones the report tree page letters.
  AD derived its sections from per-clade tolerances in the tree ``.tal``; af does not port them
  (00, 20 Oct 2026: measure the difference first).
- A small map's points come from the lab's map figure record (af.map's figure.i7.json, the
  "all" window: every antigen), in its own frame, the frame all its windows share.
- An antigen belongs to a section when a leaf it matches is in the section's rows. The match is
  by identity (the antigen's sequence, as the map's colouring chose it), never by name. Its
  caller supplies it (``antigen_leaves``).
- Colours and sizes are AD's (ae's port, ae/py/ae/tal/section_maps.py): sizes in points for a
  map cell ``AD_CELL_PT`` wide, scaled with the cell; the light grey #e0e0e0 for the chart's
  points, a darker #a1a1a1 for antigens on the tree but outside the section, the section's
  antigens filled by month on AD's bezier "viridis" scale over the tree's time window, black
  outline, on top. A section antigen dated outside the window keeps the darker grey.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("pdf")
import matplotlib.pyplot as plt  # noqa: E402  (after the backend is chosen)

from af.tree.draw.figure import FigureConfig, Prepared, RowBarIds  # noqa: E402

#: AD's three anchors of its time-series colour scale (acmacs color bezier gradient).
VIRIDIS_ANCHORS = (0x440154, 0x40FFFF, 0xFDE725)
GREY88 = "#e0e0e0"  # the chart's points
GRAY63 = "#a1a1a1"  # antigens on the tree, outside the section
WHITE = "#ffffff"
#: AD's map cell width on its own signature page: its point sizes are points on a cell this wide.
AD_CELL_PT = 171.617

MATCHES_BAR = "matches chart antigen"


class SigPageError(ValueError):
    """A signature page cannot be drawn from what it was given."""


# ----------------------------------------------------------------------------------------------
# the date colours


def bezier_gradient(c1: int, c2: int, c3: int, n: int) -> list[int]:
    """AD's quadratic bezier gradient through three colours, ``n`` steps: the integer
    truncation per term is AD's, so the colours are AD's to the bit."""
    anchors, fact = (c1, c2, c3), (1.0, 1.0, 2.0)

    def bern(t: float, deg: int, i: int) -> float:
        return fact[deg] / (fact[i] * fact[deg - i]) * ((1.0 - t) ** (deg - i)) * (t**i)

    out, denom = [], (n - 1) if n > 1 else 1
    for index in range(n):
        t = index / denom
        r = g = b = 0
        for i, col in enumerate(anchors):
            weight = bern(t, len(anchors) - 1, i)
            r += int(((col >> 16) & 0xFF) * weight)
            g += int(((col >> 8) & 0xFF) * weight)
            b += int((col & 0xFF) * weight)
        out.append((r << 16) | (g << 8) | b)
    return out


@dataclass(frozen=True)
class DateScale:
    """A colour per month of ``[start, end)`` (``YYYY-MM``); a date outside it has none."""

    start: tuple[int, int]
    colours: tuple[str, ...]

    @classmethod
    def for_window(cls, start: str, end: str) -> DateScale:
        (y0, m0), (y1, m1) = _ym(start), _ym(end)
        n = (y1 - y0) * 12 + (m1 - m0)
        if n < 1:
            raise SigPageError(f"time window {start}..{end} holds no month")
        return cls((y0, m0), tuple(f"#{c:06x}" for c in bezier_gradient(*VIRIDIS_ANCHORS, n)))

    def colour(self, date: str | None) -> str | None:
        if not date or len(date) < 7:
            return None
        y, m = _ym(date[:7])
        slot = (y - self.start[0]) * 12 + (m - self.start[1])
        return self.colours[slot] if 0 <= slot < len(self.colours) else None


def _ym(text: str) -> tuple[int, int]:
    year, month = text.split("-")[:2]
    return int(year), int(month)


# ----------------------------------------------------------------------------------------------
# sections and their antigens


@dataclass(frozen=True)
class Section:
    letter: str
    clade: str
    leaf_ids: frozenset[str]  # the drawn leaves in its rows


def sections_of(prepared: Prepared) -> list[Section]:
    """The tree's lettered bands, each with the leaf ids of its rows."""
    tree, rows = prepared.spec.tree, prepared.layout.leaf_nodes
    return [
        Section(
            band.letter,
            band.clade,
            frozenset(str(tree.leaf_id[rows[r]]) for r in range(band.first, band.last + 1)),
        )
        for band in prepared.hz
    ]


def section_members(
    sections: Sequence[Section], antigen_leaves: Mapping[str, Iterable[str]]
) -> dict[str, list[str]]:
    """Section letter -> the antigen ids (map record ``id``) one of whose leaves is in it."""
    return {
        s.letter: [ag for ag, leaves in antigen_leaves.items() if s.leaf_ids.intersection(leaves)]
        for s in sections
    }


def antigen_leaves_from_record(record: Mapping[str, Any]) -> dict[str, frozenset[str]]:
    """Antigen id -> its tree leaf id, from the map record's ``sequence`` field (af.map: the
    sequence its colouring chose). An af tree's leaf id is ``<epi_isl>|<accession>``."""
    out = {}
    for ag in record["antigens"]:
        if "sequence" not in ag:
            raise SigPageError(
                "the map record has no antigen sequences (an af.map from before the field): "
                "redraw the map"
            )
        seq = ag["sequence"]
        if seq:
            out[ag["id"]] = frozenset({f"{seq['epi_isl']}|{seq['accession']}"})
    return out


# ----------------------------------------------------------------------------------------------
# the tree panel


def panel_config(
    config: FigureConfig,
    matched_leaf_ids: Iterable[str],
    title: str,
    section_marker_width: float = 22.0,
    section_font: float = 16.0,
) -> FigureConfig:
    """The report tree's config, plus the signature page's two columns and its title."""
    geometry = dataclasses.replace(
        config.geometry, section_marker_width=section_marker_width, section_font=section_font
    )
    bar = RowBarIds(MATCHES_BAR, frozenset(matched_leaf_ids))
    return dataclasses.replace(
        config,
        geometry=geometry,
        row_bars=[*config.row_bars, bar],
        drawn_title=title,
    )


# ----------------------------------------------------------------------------------------------
# the section maps


@dataclass(frozen=True)
class MapLook:
    """AD's signature-page map look; sizes are diameters in points on an ``AD_CELL_PT`` cell."""

    test: float = 2.5
    reference: float = 3.0
    serum: float = 3.0
    section: float = 3.5
    outline: float = 1.0  # the grey points' outline
    section_outline: float = 0.5
    grid_colour: str = "#cccccc"
    grid_width: float = 0.96
    border_width: float = 1.0
    title_size: float = 10.0
    title_offset: tuple[float, float] = (3.0, 3.0)


@dataclass
class MapsLayout:
    """The map grid: ``rows`` maps down, as many columns as the sections need; cells square."""

    width: float  # points
    height: float
    rows: int = 3
    gap: float = 5.3
    columns: int = field(init=False, default=0)
    cell: float = field(init=False, default=0.0)

    def fit(self, n: int) -> MapsLayout:
        self.columns = max(1, math.ceil(n / self.rows))
        self.cell = min(
            (self.width - (self.columns - 1) * self.gap) / self.columns,
            (self.height - (self.rows - 1) * self.gap) / self.rows,
        )
        return self

    def origin(self, k: int) -> tuple[float, float]:
        """Cell ``k``'s top-left, in points from the grid's top-left: row by row, left to
        right, as the delivered pages fill it."""
        row, col = divmod(k, self.columns)
        return col * (self.cell + self.gap), row * (self.cell + self.gap)


def draw_section_maps(
    record: Mapping[str, Any],
    sections: Sequence[Section],
    members: Mapping[str, Sequence[str]],
    in_tree: Iterable[str],
    scale: DateScale,
    pdf: Path,
    layout: MapsLayout,
    look: MapLook = MapLook(),
) -> dict[str, Any]:
    """One small map per section on one PDF the size of the grid; returns what each drew."""
    if not sections:
        raise SigPageError("no sections to draw")
    layout.fit(len(sections))
    x0, top, size, tall = record["viewport"]
    height = tall if tall is not None else size
    k = layout.cell / AD_CELL_PT
    ms = lambda d: (d * k) ** 2  # noqa: E731  (scatter sizes are areas in points^2)
    tree_ids = set(in_tree)
    antigens = [a for a in record["antigens"] if a.get("shown", True)]
    sera = [s for s in record["sera"] if s.get("shown", True)]
    fig = plt.figure(figsize=(layout.width / 72, layout.height / 72))
    report: dict[str, Any] = {"cell_pt": round(layout.cell, 2), "columns": layout.columns}
    for n, section in enumerate(sections):
        cx, cy = layout.origin(n)
        ax = fig.add_axes(
            (
                cx / layout.width,
                1 - (cy + layout.cell) / layout.height,
                layout.cell / layout.width,
                layout.cell / layout.height,
            )
        )
        ax.set_xlim(x0, x0 + size)
        ax.set_ylim(top - height, top)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(look.border_width * k)
        for g in range(math.ceil(x0), math.floor(x0 + size) + 1):
            ax.axvline(g, color=look.grid_colour, lw=look.grid_width * k, zorder=0)
        for g in range(math.ceil(top - height), math.floor(top) + 1):
            ax.axhline(g, color=look.grid_colour, lw=look.grid_width * k, zorder=0)
        mine = set(members.get(section.letter, ()))
        off = [a for a in antigens if a["id"] in tree_ids and a["id"] not in mine]
        rest = [a for a in antigens if a["id"] not in tree_ids]
        tests = [a for a in rest if not a.get("reference")]
        refs = [a for a in rest if a.get("reference")]
        inside = [a for a in antigens if a["id"] in mine]
        _scatter(ax, off, ms(look.test), GRAY63, WHITE, 0.5 * k, 1)  # under the light grey
        _scatter(ax, tests, ms(look.test), GREY88, GREY88, look.outline * k, 2)
        _scatter(ax, refs, ms(look.reference), "none", GREY88, look.outline * k, 3)
        _scatter(ax, sera, ms(look.serum), "none", GREY88, look.outline * k, 3, marker="s")
        dated = [(a, scale.colour(a.get("date"))) for a in inside]
        ax.scatter(
            [a["xy"][0] for a, _ in dated],
            [a["xy"][1] for a, _ in dated],
            s=ms(look.section),
            c=[c or GRAY63 for _, c in dated],
            edgecolors="black",
            linewidths=look.section_outline * k,
            zorder=4,
        )
        ax.text(
            look.title_offset[0] * k / layout.cell,
            1 - look.title_offset[1] * k / layout.cell,
            f"{section.letter}. {section.clade}",
            transform=ax.transAxes,
            fontsize=look.title_size * k,
            ha="left",
            va="top",
            zorder=5,
        )
        report[section.letter] = {
            "clade": section.clade,
            "antigens": len(inside),
            "dated": sum(c is not None for _, c in dated),
        }
    fig.savefig(pdf, metadata={"CreationDate": None, "Creator": "acmacs-f"})
    plt.close(fig)
    return report


def _scatter(
    ax: Any,
    points: Sequence[Mapping[str, Any]],
    size: float,
    fill: str,
    edge: str,
    width: float,
    z: int,
    marker: str = "o",
) -> None:
    if points:
        ax.scatter(
            [p["xy"][0] for p in points],
            [p["xy"][1] for p in points],
            s=size,
            c=fill if fill != "none" else "none",
            edgecolors=edge,
            linewidths=width,
            marker=marker,
            zorder=z,
        )
