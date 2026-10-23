"""Make one report tree figure: the steps in order, with their counts in one report.

Input is a :class:`~af.tree.draw.model.DrawTree` (filled from the tree store, I6), the clade
hierarchy, and a :class:`FigureConfig`. Output: the PDF, its I7 JSON and a draw report
(``<stem>.draw.json``) listing every automatic decision and every override, counted.
"""

from __future__ import annotations

import contextlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

from af.store import StoreRef
from af.util.fonts import FigureFonts

from .aa_labels import AALabel, LabelHide, LabelParams, hide_labels, select_labels
from .defaults import load_defaults
from .i7 import tree_block, write_i7
from .layout import HideRules, Layout, compute_layout, rows_matching
from .model import DrawTree
from .render import DashBar, FigureSpec, Geometry, render, with_centre_column
from .sections import (
    BandParams,
    HzBand,
    SectionOverrideError,
    Selection,
    SelectParams,
    clade_membership,
    hz_partition,
    select_clades,
)
from .timeseries import TimeSeries
from .timeseries import compute as compute_timeseries


@dataclass
class Overrides:
    """Hand overrides: named data, never indices. Each must match something (design rule 1)."""

    hide_leaves: frozenset[str] = frozenset()  # leaf names not drawn
    show_clades: frozenset[str] = frozenset()
    hide_clades: frozenset[str] = frozenset()
    hide_hz_starting_at: frozenset[str] = frozenset()  # leaf name that starts a lettered band
    reasons: dict[str, str] = field(default_factory=dict)  # "hide:<clade>" -> why, who decided


@dataclass(frozen=True)
class LabelPin:
    """A substitution label held where a hand put it (Sarah, M1: the shipped report's
    positions), as ae's label editor keys it: the node is the MRCA of two leaves, and the label
    box's top-left sits at the node point plus ``dx``, ``dy`` times the page (x by the page's
    height, or by its width when ``x_by`` is "width", as the old ``offset`` did; y by height).
    Named data (design rule 9): counted in the draw report. A pin whose node carries no label of
    its text is an error unless ``optional`` (the label is drawn elsewhere, or not at all, on
    this tree), and an optional one left unused is listed with its reason."""

    first: str  # leaf id
    last: str  # leaf id
    text: str  # the substitutions, space separated, as the label prints them
    dx: float
    dy: float
    reason: str
    x_by: str = "height"
    optional: bool = False


class LabelPinError(ValueError):
    """A label pin that cannot apply: a leaf the tree does not have, a pin matching no label
    and not optional, or two pins on one label."""


@dataclass
class FigureConfig:
    title: str  # the figure's name: I7 title, report heading and caption
    window_start: str  # 'YYYY-MM', inclusive
    window_end: str  # 'YYYY-MM', exclusive
    hide: HideRules = field(default_factory=HideRules)
    # Rule thresholds: defaults.toml unless a config supplies its own (load_defaults(path)).
    select: SelectParams = field(default_factory=lambda: load_defaults().clades)
    bands: BandParams = field(default_factory=lambda: load_defaults().bands)
    labels: LabelParams = field(default_factory=lambda: load_defaults().labels)
    # Labels a round chooses not to print, by name (design rule 9): counted in the draw report.
    hide_labels: list[LabelHide] = field(default_factory=list)
    overrides: Overrides = field(default_factory=Overrides)
    dash_bars: list[DashBar] = field(default_factory=list)
    strains: list[tuple[str, str]] = field(default_factory=list)  # (leaf id, label text)
    # Strain label leaf id -> its offset below its row, in figure units: the shipped .tal's
    # hand offsets, which set two near strains' order (named data, rule 9; an id that is not
    # a strain row's is an error).
    strain_dy: dict[str, float] = field(default_factory=dict)
    # Substitution labels pinned by hand (LabelPin); the others are placed automatically.
    label_pins: list[LabelPin] = field(default_factory=list)
    colour_by: str = "continent"
    clade_colours: dict[str, str] = field(default_factory=dict)
    marked_ids: frozenset[str] = frozenset()  # centre-marked variant (leaf ids)
    marked_name: str = ""
    geometry: Geometry = field(default_factory=Geometry)
    clade_label_sizes: dict[str, float] = field(default_factory=dict)  # clade -> label size
    leaf_names: bool = False  # searchable name layer, one per drawn leaf (render.FigureSpec)
    # The title drawn on the figure, when it differs from ``title`` ("" = none). Separate, so
    # matching the shipped page (no B/Vic title, "A(H1N1)" on H1) never renames the figure
    # in the I7, the report heading or the caption.
    drawn_title: str | None = None
    # Bracket text per clade, where a figure prints more than the clade's name: the shipped pages
    # print the older name upstream records beside the current one, "<name> (<older name>)"
    # (legacy_clade_display). A clade not drawn is not printed, so an entry for it is unused.
    clade_display: dict[str, str] = field(default_factory=dict)
    # The round's figure fonts (af.util.fonts, figures.toml): the whole figure, label placement
    # included, is drawn in them and their files are recorded in the I7. None: DejaVu, as before.
    fonts: FigureFonts | None = None


def make_figure(
    tree: DrawTree,
    parents: Mapping[str, str | None],
    config: FigureConfig,
    pdf: Path,
    inputs: Mapping[str, str],
    flags: Mapping[str, list[str]] | None = None,
    store_refs: Sequence[StoreRef] = (),
    depth_unit: str = "",
) -> dict[str, Any]:
    """Render ``pdf`` and write its I7 and draw report; return the draw report.

    ``flags`` (leaf id -> reasons, from the tree store) are counted, never acted on: whether a
    flagged leaf is drawn is a hide rule's decision, not the flag's.
    """
    layout, ts, selection, hz = arrange(tree, parents, config, flags)
    shown = {cs.clade for cs in selection.shown}
    unknown = sorted(set(config.clade_label_sizes) - shown)
    if unknown:  # design rule 1: a size for a clade that is not drawn is a stale setting
        raise SectionOverrideError(f"clade label size(s) for clade(s) not shown: {unknown}")
    bands = [(b.first, b.last) for cs in selection.shown for b in cs.bands]
    labels, label_counts = select_labels(tree, layout, bands, config.labels)
    if config.hide_labels:
        labels, label_counts["hidden by name"] = hide_labels(
            tree, layout, labels, config.hide_labels
        )
        label_counts["labels"] = len(labels)

    dash_bars = dash_bar_counts(tree, layout.leaf_nodes, config.dash_bars)
    pins, pin_report = resolve_pins(tree, labels, config.label_pins)

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
        clade_label_sizes=dict(config.clade_label_sizes),
        leaf_names=config.leaf_names,
        drawn_title=config.drawn_title,
        clade_display=dict(config.clade_display),
        figure_fonts=config.fonts is not None,
        strain_dy=_strain_dy(config),
        label_pins=pins,
    )
    # Every text, label placement included, is measured and drawn in the figure fonts. They are
    # registered here, in the process that draws: a spawned worker (af.tree.draw.stage) starts
    # with matplotlib's own fonts only, and would draw in its fallback with just a warning.
    fonts = config.fonts.files.resolve() if config.fonts is not None else None
    with fonts.context() if fonts is not None else contextlib.nullcontext():
        drawn = render(spec, pdf)
    if config.fonts is not None:  # the font files are inputs (rule 5)
        inputs = {**inputs, **{k: v["sha256"] for k, v in config.fonts.inputs().items()}}
    write_i7(
        pdf,
        config.title,
        tree_block(tree, layout, hz, ts, config.marked_ids, depth_unit),
        inputs,
        store_refs=store_refs,
    )

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
        "aa_label_hides": [asdict(h) for h in config.hide_labels],
        "aa_label_placement": drawn.label_metrics,
        "dash_bars": dash_bars,
        "strains": drawn.strains,
        "leaf_names": drawn.leaf_names,
        "clade_label_sizes": dict(sorted(config.clade_label_sizes.items())),
        "drawn_title": config.title if config.drawn_title is None else config.drawn_title,
        "inset": [round(v, 2) for v in drawn.inset] if drawn.inset else None,
        "continents_not_in_legend": drawn.continents_not_in_legend,
        "flagged_drawn": _count_flags(
            flags or {}, [str(tree.leaf_id[i]) for i in layout.leaf_nodes]
        ),
        "marked_rows": int(marked.sum()) if marked is not None else None,
        "overrides": {k: len(v) for k, v in asdict(config.overrides).items() if k != "reasons"},
        "override_reasons": dict(config.overrides.reasons),
    }
    if config.label_pins:  # only when there are pins, so other figures keep their bytes
        if drawn.pin_moves is not None:  # Geometry.pin_clearance set
            pin_report = {
                **pin_report,
                "moved to clear": drawn.pin_moves,
                "glyph overprints": drawn.overprints,
            }
        report["label_pins"] = pin_report
    pdf.with_suffix(".draw.json").write_text(json.dumps(report, indent=1, default=str))
    return report


def resolve_pins(
    tree: DrawTree, labels: Sequence[AALabel], pins: Sequence[LabelPin]
) -> tuple[dict[int, tuple[float, float, str]], dict[str, Any]]:
    """Label index -> (dx, dy, x_by) for each pinned label, and what the pins did.

    A pin applies to the label on the MRCA of its two leaves whose substitutions are the
    pin's (in any order). It is never moved to another node to fit (00, 21 Oct 2026)."""
    node_of = {str(tree.leaf_id[i]): i for i in tree.leaves()}
    missing = sorted({x for p in pins for x in (p.first, p.last) if x not in node_of})
    if missing:  # design rule 1
        raise LabelPinError(f"{len(missing)} label pin leaf id(s) not in the tree: {missing[:5]}")
    by_key: dict[tuple[int, tuple[str, ...]], int] = {}
    for k, lab in enumerate(labels):
        by_key.setdefault((lab.node, tuple(sorted(lab.subs))), k)
    out: dict[int, tuple[float, float, str]] = {}
    unused: list[dict[str, str]] = []
    errors: list[str] = []
    for p in pins:
        if p.x_by not in ("height", "width"):
            errors.append(f"{p.text}: x_by {p.x_by!r} is neither 'height' nor 'width'")
            continue
        node = _mrca(tree, node_of[p.first], node_of[p.last])
        found = by_key.get((node, tuple(sorted(p.text.split()))))
        if found is None:
            if p.optional:
                unused.append({"text": p.text, "reason": p.reason})
            else:
                errors.append(f"{p.text}: no label of that text on the MRCA of {p.first}, {p.last}")
        elif found in out:
            errors.append(f"{p.text}: a second pin on the same label")
        else:
            out[found] = (p.dx, p.dy, p.x_by)
    if errors:
        raise LabelPinError(f"{len(errors)} label pin(s) refused:\n  " + "\n  ".join(errors))
    report = {
        "pins": len(pins),
        "pinned": len(out),
        "optional pins unused": unused,
        "placed automatically": [
            " ".join(lab.subs) for k, lab in enumerate(labels) if k not in out
        ],
    }
    return out, report


def _mrca(tree: DrawTree, a: int, b: int) -> int:
    up = set()
    while a != -1:
        up.add(a)
        a = tree.parent[a]
    while b not in up:
        b = tree.parent[b]
    return b


def _strain_dy(config: FigureConfig) -> dict[str, float]:
    unknown = sorted(set(config.strain_dy) - {leaf for leaf, _ in config.strains})
    if unknown:  # design rule 1
        raise ValueError(f"strain offsets for leaves with no strain label: {unknown}")
    return dict(config.strain_dy)


class OlderNames(Protocol):
    """What legacy_clade_display asks of a clade set (af.clades.nomenclature.CladeSet)."""

    @property
    def version(self) -> str: ...

    def __contains__(self, name: object) -> bool: ...

    def legacy_name(self, name: str) -> str | None: ...


def legacy_clade_display(clade_set: OlderNames, clades: Iterable[str]) -> dict[str, str]:
    """``"<name> (<older name>)"`` for each clade upstream records an older name for, as the
    report pages have always printed a clade (af.clades.nomenclature.CladeSet.legacy_name).
    A clade not in ``clade_set`` is an error, never a name printed bare by default."""
    out = {}
    for clade in clades:
        if clade not in clade_set:
            raise ValueError(f"clade {clade!r} is not in {clade_set.version}")
        older = clade_set.legacy_name(clade)
        if older and older != clade:
            out[clade] = f"{clade} ({older})"
    return out


def arrange(
    tree: DrawTree,
    parents: Mapping[str, str | None],
    config: FigureConfig,
    flags: Mapping[str, list[str]] | None = None,
) -> tuple[Layout, TimeSeries, Selection, list[HzBand]]:
    """Rows, time window, clade selection and lettered bands: shared by the figure and the
    leaf book, so a row and a band letter mean the same leaf in both."""
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
    return layout, ts, selection, hz


class DashBarError(ValueError):
    """A dash-bar colour key that no drawn leaf has (design rule 1)."""


def dash_bar_counts(
    tree: DrawTree, leaf_nodes: Iterable[int], bars: Sequence[DashBar]
) -> list[dict[str, Any]]:
    """Drawn rows per residue for each bar: coloured, transparent, and with no colour.

    A colour key that matches no drawn row is an error: it means the bar was configured for
    other residues (or another numbering), and the figure would show an empty or wrong bar with
    a legend that looks right. A residue with no colour is counted, not an error: it is simply
    not drawn. ``""`` counts rows whose sequence does not reach the position.
    """
    out, problems = [], []
    for bar in bars:
        rows: dict[str, int] = {}
        for i in leaf_nodes:
            a = tree.aa[i]
            residue = a[bar.pos - 1] if a and len(a) >= bar.pos else ""
            rows[residue] = rows.get(residue, 0) + 1
        unmatched = sorted(k for k in bar.colours if rows.get(k, 0) == 0)
        if unmatched:
            seen = ", ".join(f"{k or '(none)'} {n}" for k, n in sorted(rows.items()))
            problems.append(
                f"position {bar.pos}: colour key(s) {unmatched} match no drawn leaf ({seen})"
            )
        kind = {
            k: "no_colour"
            if k not in bar.colours
            else "transparent"
            if bar.colours[k] == "transparent"
            else "coloured"
            for k in rows
        }
        entry: dict[str, Any] = {"pos": bar.pos, "coloured": {}, "transparent": {}, "no_colour": {}}
        for k, n in sorted(rows.items()):
            entry[kind[k]][k] = n
        out.append(entry)
    if problems:
        raise DashBarError("dash bars: " + "; ".join(problems))
    return out


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
