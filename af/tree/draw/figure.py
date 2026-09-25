"""Make one report tree figure: the steps in order, with their counts in one report.

Input is a :class:`~af.tree.draw.model.DrawTree` (filled from the tree store, I6), the clade
hierarchy, and a :class:`FigureConfig`. Output: the PDF, its I7 JSON and a draw report
(``<stem>.draw.json``) listing every automatic decision and every override, counted.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .aa_labels import LabelParams, select_labels
from .defaults import load_defaults
from .i7 import tree_block, write_i7
from .layout import HideRules, compute_layout, rows_matching
from .model import DrawTree
from .render import DashBar, FigureSpec, Geometry, render, with_centre_column
from .sections import (
    BandParams,
    SectionOverrideError,
    SelectParams,
    clade_membership,
    hz_partition,
    select_clades,
)
from .timeseries import compute as compute_timeseries


@dataclass
class Overrides:
    """Hand overrides: named data, never indices. Each must match something (design rule 1)."""

    hide_leaves: frozenset[str] = frozenset()  # leaf names not drawn
    show_clades: frozenset[str] = frozenset()
    hide_clades: frozenset[str] = frozenset()
    hide_hz_starting_at: frozenset[str] = frozenset()  # leaf name that starts a lettered band


@dataclass
class FigureConfig:
    title: str
    window_start: str  # 'YYYY-MM', inclusive
    window_end: str  # 'YYYY-MM', exclusive
    hide: HideRules = field(default_factory=HideRules)
    # Rule thresholds: defaults.toml unless a config supplies its own (load_defaults(path)).
    select: SelectParams = field(default_factory=lambda: load_defaults().clades)
    bands: BandParams = field(default_factory=lambda: load_defaults().bands)
    labels: LabelParams = field(default_factory=lambda: load_defaults().labels)
    overrides: Overrides = field(default_factory=Overrides)
    dash_bars: list[DashBar] = field(default_factory=list)
    strains: list[tuple[str, str]] = field(default_factory=list)  # (leaf id, label text)
    colour_by: str = "continent"
    clade_colours: dict[str, str] = field(default_factory=dict)
    marked_ids: frozenset[str] = frozenset()  # centre-marked variant (leaf ids)
    marked_name: str = ""
    geometry: Geometry = field(default_factory=Geometry)


def make_figure(
    tree: DrawTree,
    parents: Mapping[str, str | None],
    config: FigureConfig,
    pdf: Path,
    inputs: Mapping[str, str],
    flags: Mapping[str, list[str]] | None = None,
) -> dict[str, Any]:
    """Render ``pdf`` and write its I7 and draw report; return the draw report.

    ``flags`` (leaf id -> reasons, from the tree store) are counted, never acted on: whether a
    flagged leaf is drawn is a hide rule's decision, not the flag's.
    """
    hide = HideRules(
        config.hide.min_edge,
        config.hide.names | config.overrides.hide_leaves,
        config.hide.flag_reasons,
    )
    layout = compute_layout(tree, hide, flags)
    ts = compute_timeseries(
        [tree.date[i] for i in layout.leaf_nodes],
        [tree.date_precision[i] for i in layout.leaf_nodes],
        config.window_start,
        config.window_end,
    )
    member = clade_membership([tree.clade[i] for i in layout.leaf_nodes], parents)
    selection = select_clades(
        member,
        parents,
        ts.in_window,
        config.select,
        config.bands,
        config.overrides.show_clades,
        config.overrides.hide_clades,
    )
    hz_first_rows = _rows_of_names(tree, layout, config.overrides.hide_hz_starting_at)
    hz = hz_partition(
        selection, parents, ts.in_window, config.select, hide_first_rows=frozenset(hz_first_rows)
    )
    bands = [(b.first, b.last) for cs in selection.shown for b in cs.bands]
    labels, label_counts = select_labels(tree, layout, bands, config.labels)

    marked = None
    geometry = config.geometry
    if config.marked_ids:
        marked = rows_matching(tree, layout, config.marked_ids)
        geometry = with_centre_column(geometry)
    spec = FigureSpec(
        config.title,
        tree,
        layout,
        selection,
        hz,
        ts,
        labels,
        config.dash_bars,
        config.strains,
        marked,
        config.marked_name,
        colour_by=config.colour_by,
        clade_colours=config.clade_colours,
        geometry=geometry,
    )
    drawn = render(spec, pdf)
    write_i7(pdf, config.title, tree_block(tree, layout, hz, ts, config.marked_ids), inputs)

    names = {i: tree.leaf_id[i] for i in layout.leaf_nodes}
    report = {
        "rows": layout.n_rows,
        "hidden": layout.hidden,
        "time_series": ts.counts,
        "clades_shown": {
            cs.clade: {
                "slot": selection.slots[cs.clade],
                "rows": cs.total,
                "bands": len(cs.bands),
                "dropped_bands": len(cs.dropped),
                "dropped_rows": sum(b.members for b in cs.dropped),
            }
            for cs in selection.shown
        },
        "clades_not_shown": selection.rejected,
        "hz": [
            {
                "letter": h.letter,
                "clade": h.clade,
                "first": names[layout.leaf_nodes[h.first]],
                "rows": h.last - h.first + 1,
            }
            for h in hz
        ],
        "aa_labels": [
            {"subs": lab.subs, "first": lab.first, "last": lab.last, "rows": lab.rows}
            for lab in labels
        ],
        "aa_label_filter": label_counts,
        "aa_label_placement": drawn.label_metrics,
        "strains": drawn.strains,
        "continents_not_in_legend": drawn.continents_not_in_legend,
        "flagged_drawn": _count_flags(
            flags or {}, [str(tree.leaf_id[i]) for i in layout.leaf_nodes]
        ),
        "marked_rows": int(marked.sum()) if marked is not None else None,
        "overrides": {k: len(v) for k, v in asdict(config.overrides).items()},
    }
    pdf.with_suffix(".draw.json").write_text(json.dumps(report, indent=1, default=str))
    return report


def _rows_of_names(tree: DrawTree, layout, names: frozenset[str]) -> set[int]:
    row_of = {tree.name[i]: r for r, i in enumerate(layout.leaf_nodes)}
    missing = sorted(n for n in names if n not in row_of)
    if missing:
        raise SectionOverrideError(f"hz override(s) name no drawn leaf: {missing[:5]}")
    return {row_of[n] for n in names}


def _count_flags(flags: Mapping[str, list[str]], drawn_ids: list[str]) -> dict[str, int]:
    """Drawn rows per flag reason."""
    out: dict[str, int] = {}
    for leaf in drawn_ids:
        for reason in flags.get(leaf, []):
            out[reason] = out.get(reason, 0) + 1
    return dict(sorted(out.items()))
