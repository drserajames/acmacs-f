"""Build a round's antigenic map figures from its config.

One command, every input explicit, nothing read from the environment (design rule 4):

    python -m af.map.build --config <round>/config/maps.toml --store <root> --out <figures>

Writes ``map/<folder>/<window>/<version>/figure.pdf`` and ``figure.i7.json`` under ``--out``,
writes nothing outside it, and exits non-zero if any map fails. Each map prints one line.

What happens to a map, in the order the steps require:
layout (a chain's last step, or a bring-up stand-in) → named moves → named hides → orientation
to the previous round's drawn points, plus any named rotation → vaccines from the curated list →
style → frame (avoiding the legend it will draw) → labels → PDF → I7.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import sys
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from af.chart.ace import content_hash, read_chart
from af.chart.column_bases import (
    ColumnBaseOverride,
    ColumnBaseOverrideError,
    apply_column_base_override,
)
from af.chart.model import Chart
from af.chart.sera import FERRET_ONLY, Marker, non_ferret, read_markers
from af.map.colouring import MapColouringError, StoreColours
from af.map.config import MapConfig, MapsConfig
from af.map.curate import (
    BlockOffset,
    CurationError,
    MoveOverride,
    apply_block_offset,
    apply_move,
)
from af.map.figure import chart_points
from af.map.finish import finish_map
from af.map.labels import label_text
from af.map.orient import Orientation, RotationOverride, drawn_pairs, orient
from af.map.style import ColourRow, ColourScheme, Window
from af.map.vaccines import (
    MapAntigen,
    VaccineChoice,
    VaccineDisable,
    select_vaccines,
)
from af.map.viewport import from_y_down
from af.store.busy import ReadGuard
from af.store.ref import StoreError, StoreRef
from af.store.store import Store
from af.util.subtypes import Subtype

FloatArray = NDArray[np.float64]

#: Where the non-ferret sera markers live under the acmacs-f-data checkout (af_data).
SERA_MARKERS = Path("rules/sera/non_ferret_markers.tsv")

_DATE_IN_PASSAGE = re.compile(r" \(\d{4}-\d{2}-\d{2}\)")


class BuildError(RuntimeError):
    """A map could not be built. The message says which map and why."""


# ---------------------------------------------------------------- chart helpers


def designation(a: Any) -> str:
    """A point as a person reads it, with the passage's date stripped: the form overrides use."""
    return " ".join(
        p for p in (a.name, a.reassortant, *a.annotations, _DATE_IN_PASSAGE.sub("", a.passage)) if p
    )


def clade_labels(a: Any) -> frozenset[str]:
    """Clade and aa labels the chart carries for this antigen (semantic attribute "C")."""
    return frozenset((a.extra.get("T") or {}).get("C") or ())


def table_counts(chart: Chart) -> tuple[list[int], list[dt.date | None]]:
    """Per antigen: in how many source tables it has a titre, and the latest such table's date."""
    n = chart.n_antigens
    counts = [0] * n
    latest: list[dt.date | None] = [None] * n
    sources = chart.info.get("S") or []
    for k, layer in enumerate(chart.titres.layers):
        when = _compact(sources[k]) if k < len(sources) else None
        # ONE per table, not one per titre: this decides which preparation of a vaccine strain is
        # marked, and counting titres would let a strain tested against many sera in one table
        # outrank one tested in many tables.
        in_this_table = {i for i, _j in layer}
        for i in in_this_table:
            counts[i] += 1
            previous = latest[i]
            if when is not None and (previous is None or when > previous):
                latest[i] = when
    return counts, latest


def _compact(source: dict[str, Any]) -> dt.date | None:
    """Table dates appear as 20260909, 20260909.002 or 2026-09-09; parse, never sort as text."""
    raw = str(source.get("D", "")).split(".")[0].replace("-", "")
    if len(raw) < 8 or not raw[:8].isdigit():
        return None
    return dt.date(int(raw[:4]), int(raw[4:6]), int(raw[6:8]))


def scheme_from_chart(chart: Chart, name: str) -> ColourScheme:
    """Bring-up: read the colour rows a chart already carries (`R` plot specs).

    Workstream 4's user colour schemes replace this; until then the round's own scheme is the
    only source, and the figure's provenance says so.
    """
    styles = chart.extra.get("R") or {}
    style = styles.get("-" + name) or styles.get(name)
    if style is None:
        raise BuildError(f"chart has no colour scheme {name!r}")
    rows = []
    for m in style.get("A", ()):
        sel = m.get("T") or {}
        if m.get("A") in (1, True) and set(sel) == {"C"} and "L" in m and "F" in m:
            rows.append(ColourRow(m["L"]["t"], m["F"], frozenset({sel["C"]})))
    if not rows:
        raise BuildError(f"colour scheme {name!r} has no clade rows")
    return ColourScheme(f"chart {name}", tuple(rows))


def _labels_by_designation(chart: Chart) -> dict[str, frozenset[str]]:
    """Clade labels from a chart, keyed by designation (bring-up stand-in for assignment)."""
    out: dict[str, frozenset[str]] = {}
    for a in chart.antigens:
        labels = clade_labels(a)
        if labels:
            out[designation(a)] = labels
    return out


def chart_subtype(chart: Chart) -> Subtype:
    """The chart's row in af's subtype table: :func:`af.map.colouring.chart_subtype`, the one
    copy of the rule (majority lineage), with its error as a build error."""
    from af.map.colouring import chart_subtype as from_chart

    try:
        return from_chart(chart)
    except MapColouringError as exc:
        raise BuildError(str(exc)) from exc


def lineage_minority(chart: Chart, row: Subtype) -> dict[str, Any]:
    """Antigens whose lineage code is not the map's, by code, with a few names: drawn as the
    chart has them, and reported (flagged means reported, not excluded)."""
    other = [a for a in chart.antigens if str(a.extra.get("L", "")) != row.ace_lineage]
    if not other:
        return {}
    by_code = Counter(str(a.extra.get("L", "")) or "none" for a in other)
    return {
        "map_lineage": row.ace_lineage,
        "other": dict(sorted(by_code.items())),
        "examples": sorted({a.name for a in other})[:5],
    }


def map_title(chart: Chart, configured: str | None) -> str:
    if configured:
        return configured
    info = chart.info
    assay = {"HI": "", "HINT": " HINT", "PRN": " neut", "FOCUS REDUCTION+FRA": " neut"}.get(
        info.get("A", ""), f" {info.get('A', '')}".rstrip()
    )
    return f"{info.get('l', '')} {chart_subtype(chart).name}{assay} by clade"


# ---------------------------------------------------------------- steps


def chain_chart(
    store: Store, dataset: str, until_table: str | None = None
) -> tuple[Chart, StoreRef, Path]:
    """The chart of a chain's CURRENT version: its last step, or the step after ``until_table``.

    Resolved at run time: a pinned version goes stale the moment the chain is rebuilt, and the
    report builder refuses a stale figure. ``until_table`` names a table id; it must match
    exactly one step (design rule 1), and a step index is never accepted (rule 2).
    """
    ref = store.current("chains", dataset)
    root = store.resolve(ref)
    steps = json.loads((root / "chain.json").read_text())["steps"]
    if not steps:
        raise BuildError(f"chain {dataset} has no steps")
    if until_table is None:
        step = steps[-1]
    else:
        found = [s for s in steps if s["table_id"] == until_table]
        if len(found) != 1:
            raise BuildError(
                f"chain {dataset}@{ref.version}: chain_until table {until_table!r} matches "
                f"{len(found)} steps (needs exactly 1)"
            )
        step = found[0]
    ace = root / step["directory"] / step["chosen_file"]
    return read_chart(ace), ref, ace


def apply_moves(
    chart: Chart,
    layout: np.ndarray,
    cfg: MapConfig,
    scheme: ColourScheme,
    painted: Sequence[str | None],
    relax: Any,
    stress: Any,
) -> tuple[np.ndarray, list[str], list[dict[str, Any]]]:
    """Run the map's kept moves in order. A refused move leaves the layout alone and is flagged
    on the figure rather than stopping the build: the reader should see that it was refused."""
    designations = [designation(a) for a in chart.antigens]
    flags: list[str] = []
    reports: list[dict[str, Any]] = []
    current = stress(layout)
    for m in cfg.moves:
        row = next((r for r in scheme.rows if r.legend == m.target_legend), None)
        if row is None:
            raise BuildError(
                f"{cfg.folder}: move {m.name!r} targets unknown legend row {m.target_legend!r}"
            )
        rule = MoveOverride(
            m.name,
            m.reason,
            m.decided.isoformat(),
            m.load_movers(),
            scheme.name,
            row.colour,
            m.max_stress_rise,
            m.max_from_target,
            m.min_target_points,
        )
        try:
            result = apply_move(
                rule, layout, designations, painted, stress_before=current, relax=relax
            )
        except CurationError as exc:
            flags.append(str(exc).split(";")[0])
            reports.append(
                {
                    "override": m.name,
                    "applied": False,
                    "why": str(exc).split(";")[0],
                    **exc.fields(),
                }
            )
            continue
        layout, current = result.layout, result.stress_after
        report = {**result.report(rule), "applied": True}
        # A move that reports success and changes nothing is a known failure mode of the scripts
        # this replaces (one ran for weeks moving nothing). Flag it: either the map already had
        # the arrangement, or the rule is no longer doing what it was written to do.
        if result.movers_moved < 0.01:
            note = f"move {m.name!r} changed nothing (furthest mover {result.movers_moved:.4f} u)"
            flags.append(note)
            report["no_op"] = True
        reports.append(report)
    return layout, flags, reports


def hidden_by_rules(chart: Chart, cfg: MapConfig) -> dict[int, str]:
    """Named hides, by designation. Every listed designation must match, or the build fails:
    a hide list that silently stops matching is how the old index-based hides went wrong."""
    hidden: dict[int, str] = {}
    by_designation: dict[str, list[int]] = {}
    for i, a in enumerate(chart.antigens):
        by_designation.setdefault(designation(a), []).append(i)
    for rule in cfg.hides:
        wanted = rule.load()
        missing = [d for d in wanted if d not in by_designation]
        if missing:
            raise BuildError(
                f"{cfg.folder}: hide {rule.name!r} lists {len(missing)} designation(s) that match "
                f"no antigen, e.g. {missing[0]!r}"
            )
        for d in wanted:
            for i in by_designation[d]:
                hidden[i] = rule.name
    return hidden


def orientation_for(
    chart: Chart, layout: np.ndarray, cfg: MapConfig, config: MapsConfig, scheme_name: str
) -> Orientation | None:
    """Fit to the previous round's map, over the points that map actually DREW."""
    if config.defaults.previous_round is None:
        return None
    previous = config.defaults.previous_round / cfg.folder / "styled.ace"
    if not previous.is_file():
        return None
    prev = read_chart(previous)
    pairs, prev_drawn = common_points(chart, prev, scheme_name)
    overrides = tuple(
        RotationOverride(r.name, r.degrees, r.reason, r.decided.isoformat(), r.reflect)
        for r in cfg.rotations
    )
    # The previous round's map is an ae round's styled.ace, which kateri drew with y DOWN. Fit to
    # it as it was DRAWN, so this round's map looks like last round's rather than mirrored (Q105).
    return orient(
        layout,
        from_y_down(prev.projections[0].transformed_layout()),
        drawn_pairs(pairs, prev_drawn),
        reference=f"{cfg.folder} previous round ({previous.parent.parent.name})",
        min_common=config.defaults.min_common_points,
        overrides=overrides,
    )


def common_points(chart: Chart, other: Chart, scheme_name: str) -> tuple[np.ndarray, np.ndarray]:
    """Points that are the same virus or serum in both charts, and which of `other`'s are drawn."""

    def key(p: Any, kind: str) -> tuple:
        return (kind, p.name, p.reassortant, p.annotations, getattr(p, "serum_id", ""), p.passage)

    mine = [key(a, "a") for a in chart.antigens] + [key(s, "s") for s in chart.sera]
    theirs = [key(a, "a") for a in other.antigens] + [key(s, "s") for s in other.sera]
    from collections import Counter

    cm, ct = Counter(mine), Counter(theirs)
    index = {k: i for i, k in enumerate(theirs) if ct[k] == 1}
    pairs = np.array(
        [(i, index[k]) for i, k in enumerate(mine) if cm[k] == 1 and k in index], dtype=np.intp
    )
    if len(pairs) == 0:
        pairs = np.empty((0, 2), dtype=np.intp)
    drawn = np.ones(other.n_points, dtype=np.bool_)
    hidden = _hidden_in_style(other, scheme_name)
    # A reference chart's index hides can point past its own end: the shipped rounds select by
    # antigen index, and an index list written one round is applied to the next chart unchanged
    # (the failure af removes by never selecting on index). Out-of-range entries are ignored here
    # rather than crashing, because this is only deciding which points steer an orientation fit.
    inside = [i for i in hidden if 0 <= i < other.n_points]
    drawn[inside] = False
    return pairs, drawn


def _hidden_in_style(chart: Chart, scheme_name: str) -> set[int]:
    """Points a chart's own style hides (`"-": true` modifiers), so they do not steer the fit."""
    styles = chart.extra.get("R") or {}
    hidden: set[int] = set()
    seen: set[str] = set()

    def walk(name: str) -> None:
        if name in seen:
            return
        seen.add(name)
        style = styles.get(name)
        if not style:
            return
        for m in style.get("A", ()):
            if "R" in m:
                walk(m["R"])
            elif m.get("-") is True:
                sel = m.get("T") or {}
                if "!i" in sel:
                    hidden.add(int(sel["!i"]))

    walk(scheme_name)
    return hidden


def _stand_in_colours(
    chart: Chart, cfg: MapConfig, inputs: dict[str, Any]
) -> tuple[ColourScheme, list[frozenset[str]]]:
    """Bring-up colours: the rows and clade labels a chart already carries, painted in row order.

    Replaced by the store colouring (``[colouring] source = "store"``).
    """
    scheme_chart = read_chart(cfg.scheme_stand_in) if cfg.scheme_stand_in else chart
    scheme = scheme_from_chart(scheme_chart, cfg.clade_scheme)
    if cfg.scheme_stand_in:
        inputs["colour_scheme_chart"] = {
            "path": str(cfg.scheme_stand_in),
            "sha256": content_hash(cfg.scheme_stand_in),
            "note": "STAND-IN: colour rows AND clade labels, until clade assignment is wired in",
        }
    # Clade labels come from the same chart as the colour rows: a chart written before the round's
    # clade step carries neither.
    labels_of = _labels_by_designation(scheme_chart) if cfg.scheme_stand_in else None
    # Resolved ONCE, for every antigen, and used for both the colours and the points. Computing
    # it in two places is how the substitution-qualified clades ("K 96R") were lost from the
    # points while the colours still had them: one call site was updated and the other was not.
    labels = [
        clade_labels(a) if labels_of is None else labels_of.get(designation(a), frozenset())
        for a in chart.antigens
    ]
    return scheme, labels


# ---------------------------------------------------------------- one map


@dataclass(frozen=True)
class MapResult:
    folder: str
    figures: tuple[Path, ...]
    flags: tuple[str, ...]
    seconds: float
    # Antigens of a lineage other than the map's, by code: drawn, and counted in the build's own
    # report so nobody has to open a figure to learn of them.
    lineage_minority: dict[str, int] = field(default_factory=dict)
    # Sera on the chart that are not ferret (species or marker): drawn, reported, counted here.
    non_ferret_sera: int = 0


def build_map(
    cfg: MapConfig,
    config: MapsConfig,
    *,
    store: Store | None,
    out_root: Path,
    vaccine_table: Mapping[str, Sequence[Any]] | Sequence[Any],
    vaccine_defaults: dict[str, tuple[VaccineDisable, ...]],
    created: dt.datetime,
    colours: StoreColours | None = None,
    sera_markers: list[Marker] | None = None,
    guard: ReadGuard | None = None,
) -> MapResult:
    """Build every window of one map. Raises :class:`BuildError` with the folder named.

    ``colours`` is the run's store colouring when the round's ``[colouring] source`` is
    "store"; without it the map is painted from its stand-in chart, as in bring-up.
    ``sera_markers`` (the non-ferret markers) make the map REPORT any serum that is not ferret
    (Sarah, 1 Oct: ferret sera only). It reports and does not refuse: the build draws whatever
    chart it is given, and refusing is a deliberate later change, once every map comes from a
    path that already filters.
    """
    started = time.monotonic()
    inputs: dict[str, Any] = {}
    stand_in: dict[str, str] = {}
    store_refs: list[dict[str, str]] = []
    if cfg.chain:
        if store is None:
            raise BuildError(f"{cfg.folder}: chain configured but no --store given")
        until = cfg.chain_until
        chart, ref, ace = chain_chart(store, cfg.chain, until.table if until else None)
        store_refs.append(ref.to_json())
        inputs["chain"] = {**ref.to_json(), "sha256": content_hash(ace)}
        if until is not None:
            inputs["chain"]["until_table"] = until.table
    else:
        assert cfg.layout_stand_in is not None
        chart = read_chart(cfg.layout_stand_in)
        inputs["chart"] = {
            "path": str(cfg.layout_stand_in),
            "sha256": content_hash(cfg.layout_stand_in),
        }
        stand_in["layout"] = str(cfg.layout_stand_in)

    if not chart.projections:
        raise BuildError(f"{cfg.folder}: chart has no projection to draw")
    sequenced: list[bool]
    if colours is not None:
        try:
            coloured = colours.for_chart(chart, cfg.clade_scheme)
        except (MapColouringError, ValueError) as exc:
            raise BuildError(f"{cfg.folder}: {exc}") from exc
        scheme = coloured.scheme
        labels = list(coloured.labels)
        sequenced = list(coloured.sequenced)
        store_refs.extend(colours.store_refs())
        colour_note: dict[str, Any] = coloured.provenance
    else:
        scheme, labels = _stand_in_colours(chart, cfg, inputs)
        sequenced = [bool(a.extra.get("A")) for a in chart.antigens]
        colour_note = {
            "source": "stand-in",
            "note": "STAND-IN: the chart's own rows, until user colour schemes are wired in",
        }

    inputs["colour_scheme"] = {
        "name": scheme.name,
        "sha256": hashlib.sha256(
            json.dumps([[r.legend, r.colour, sorted(r.labels)] for r in scheme.rows]).encode()
        ).hexdigest(),
        **colour_note,
    }
    layout = chart.projections[0].layout.copy()
    painted = [
        (lambda row: row.colour if row else None)(scheme.paint(labels[i]))
        for i in range(chart.n_antigens)
    ]
    # What was DECIDED about this map, as opposed to what it was built from: these carry reasons,
    # not content hashes, so they are not inputs.
    decisions: dict[str, Any] = {}
    non_ferret_count = 0
    if sera_markers is not None:
        sera = non_ferret(chart, sera_markers)
        non_ferret_count = len(sera.non_ferret)
        decisions["sera"] = {
            "rule": FERRET_ONLY,
            **sera.to_json(),
            "verification": sera.verification(),
        }
    if minority := lineage_minority(chart, chart_subtype(chart)):
        decisions["lineage_minority"] = minority
    if cfg.chain_until is not None:
        decisions["chain_until"] = {
            "table": cfg.chain_until.table,
            "reason": cfg.chain_until.reason,
            "decided": cfg.chain_until.decided.isoformat(),
        }
    projection = chart.projections[0]
    for cb in cfg.column_bases:
        rule_cb = ColumnBaseOverride(
            cb.name, cb.reason, cb.decided.isoformat(), tuple(cb.sera), cb.value
        )
        try:
            result_cb = apply_column_base_override(chart, projection, rule_cb)
        except ColumnBaseOverrideError as exc:
            raise BuildError(f"{cfg.folder}: {exc}") from exc
        projection = result_cb.projection
        decisions.setdefault("column_bases", []).append(result_cb.report(rule_cb))
    relax, stress = relaxer_for(chart, projection=projection)
    layout, flags, move_reports = apply_moves(chart, layout, cfg, scheme, painted, relax, stress)
    if move_reports:
        decisions["moves"] = move_reports
    for b in cfg.blocks:
        row_b = next((r for r in scheme.rows if r.legend == b.target_legend), None)
        if row_b is None:
            raise BuildError(
                f"{cfg.folder}: block {b.name!r} targets unknown legend row {b.target_legend!r}"
            )
        rule_b = BlockOffset(
            b.name,
            b.reason,
            b.decided.isoformat(),
            b.load_movers(),
            b.shift,
            b.derived_from,
            scheme.name,
            row_b.colour,
            b.max_stress_rise,
            b.settled_within,
            b.min_settled,
            b.max_other_move,
        )
        try:
            result_b = apply_block_offset(
                rule_b,
                layout,
                [designation(a) for a in chart.antigens],
                painted,
                stress_before=stress(layout),
                relax=relax,
            )
        except CurationError as exc:
            flags.append(str(exc).split(";")[0])
            decisions.setdefault("blocks", []).append(
                {
                    "override": b.name,
                    "applied": False,
                    "why": str(exc).split(";")[0],
                    **exc.fields(),
                }
            )
            continue
        layout = result_b.layout
        decisions.setdefault("blocks", []).append({**result_b.report(rule_b), "applied": True})

    hidden = hidden_by_rules(chart, cfg)
    if cfg.hides:
        decisions["hides"] = [
            {
                "name": r.name,
                "reason": r.reason,
                "decided": r.decided.isoformat(),
                "count": len(r.load()),
            }
            for r in cfg.hides
        ]

    ori = orientation_for(chart, layout, cfg, config, cfg.clade_scheme)
    xy = layout @ ori.matrix + ori.translation if ori else layout

    counts, latest = table_counts(chart)
    ags = [
        MapAntigen(
            i,
            a.name,
            a.passage,
            a.reassortant,
            counts[i],
            bool(np.isfinite(layout[i]).all()),
            latest[i],
        )
        for i, a in enumerate(chart.antigens)
    ]
    # Subtype defaults are chosen by the chart's OWN subtype, never guessed from the folder name:
    # folder naming is a local convention, and af must not depend on one (design rule 10).
    subtype = chart.info.get("V", "")
    disable: list[VaccineDisable] = list(vaccine_defaults.get(subtype, ()))
    n_defaults = len(disable)
    if vaccine_defaults and subtype not in vaccine_defaults:
        decisions["vaccine_defaults"] = f"no subtype defaults for {subtype!r}"
    for d in cfg.vaccine_disable:
        disable.append(
            VaccineDisable(d.name, _passage(d.passage), d.reason, d.optional, _iso(d.decided))
        )
    choose = [
        VaccineChoice(
            c.name, _class(c.passage_class), c.passage, c.reason, c.optional, _iso(c.decided)
        )
        for c in cfg.vaccine_choose
    ]
    vrep = select_vaccines(
        ags, vaccine_table_for(vaccine_table, chart_subtype(chart)), disable=disable, choose=choose
    )
    ids = [f"ag{i}" for i in range(chart.n_antigens)] + [f"sr{j}" for j in range(chart.n_sera)]
    vlabels = {
        ids[m.antigen]: label_text(
            chart.antigens[m.antigen].name,
            m.passage_class,
            chart.antigens[m.antigen].reassortant,
            {},
        )
        for m in vrep.marks
    }
    if vrep.unused_optional_rules:
        decisions["vaccine_rules_unused_optional"] = list(vrep.unused_optional_rules)
    if disable or choose:
        decisions["vaccines"] = vaccine_rule_records(disable, choose, n_defaults, vrep.used_rules)

    points = chart_points(chart, xy, labels=labels, sequenced=sequenced, hidden=hidden)

    size = config.frame_size(chart.info.get("V", ""), chart.info.get("A", ""))
    title_base = map_title(chart, cfg.title)
    provenance: dict[str, Any] = {"inputs": inputs}
    if decisions:
        provenance["decisions"] = decisions
    if store_refs:
        provenance["store_refs"] = store_refs
    if guard is not None:  # the guarded read this figure came from (af.store.busy)
        provenance["store_read"] = guard.to_json()
    if stand_in:
        provenance["stand_in"] = stand_in
    figures = []
    for w in config.defaults.windows:
        suffix = "" if w.since is None else f" (since {w.since.strftime('%B %Y')})"
        out = out_root / "map" / cfg.folder / w.name / "v1" / "figure.pdf"
        result = finish_map(
            points,
            chart=cfg.folder,
            scheme=scheme,
            window=Window(w.name, w.since),
            title=title_base + suffix,
            frame_size=size,
            must_show_since=config.defaults.must_show_since,
            vaccine_labels=vlabels,
            out_pdf=out,
            created=created,
            provenance=provenance,
            orientation=ori.report() if ori else None,
            flags=flags,
        )
        figures.append(result.i7)
    return MapResult(
        cfg.folder,
        tuple(figures),
        tuple(flags),
        time.monotonic() - started,
        dict(decisions.get("lineage_minority", {}).get("other", {})),
        non_ferret_count,
    )


# ---------------------------------------------------------------- the command


def _iso(day: dt.date | None) -> str | None:
    return day.isoformat() if day is not None else None


def vaccine_rule_records(
    disable: Sequence[VaccineDisable],
    choose: Sequence[VaccineChoice],
    n_defaults: int,
    used: set[VaccineDisable | VaccineChoice],
) -> list[dict[str, Any]]:
    """Every vaccine rule this map was built with, in order (the subtype defaults, then the
    map's own), each with whether it matched anything on this chart. A rule with no recorded
    date has ``decided`` null, so a report can say the date is missing rather than omit it."""
    out: list[dict[str, Any]] = []
    for n, d in enumerate(disable):
        out.append(
            {
                "rule": "disable",
                "scope": "subtype default" if n < n_defaults else "map",
                "name": d.name,
                "passage": d.passage,
                "reason": d.reason,
                "decided": d.decided,
                "optional": d.optional,
                "used": d in used,
            }
        )
    for c in choose:
        out.append(
            {
                "rule": "choose",
                "scope": "map",
                "name": c.name,
                "passage": c.passage,
                "passage_class": c.passage_class,
                "reason": c.reason,
                "decided": c.decided,
                "optional": c.optional,
                "used": c in used,
            }
        )
    return out


def load_vaccine_defaults(path: Path | None) -> dict[str, tuple[VaccineDisable, ...]]:
    """Subtype-wide vaccine rules shared by every round (one editable copy, design rule 6)."""
    if path is None:
        return {}
    import tomllib

    with path.open("rb") as f:
        data = tomllib.load(f)
    out: dict[str, tuple[VaccineDisable, ...]] = {}
    for subtype, rows in (data.get("disable") or {}).items():
        out[subtype] = tuple(
            VaccineDisable(
                r["name"],
                r.get("passage", "any"),
                r["reason"],
                bool(r.get("optional", True)),
                dt.date.fromisoformat(str(r["decided"])).isoformat() if "decided" in r else None,
            )
            for r in rows
        )
    return out


def build(
    config: MapsConfig,
    *,
    store_root: Path | None,
    out_root: Path,
    vaccine_list: Mapping[str, Sequence[Any]] | Sequence[Any],
    vaccine_defaults: dict[str, tuple[VaccineDisable, ...]],
    only: Sequence[str] = (),
    created: dt.datetime | None = None,
    log: Any = print,
    ignore_busy: bool = False,
) -> list[MapResult]:
    """Build every configured map (or just ``only``). Raises on the first failure.

    ``ignore_busy`` builds while a batch is writing the store, for diagnosis only (the command
    line's explicit ``--ignore-busy``); every figure's provenance then says it did."""
    created = created or dt.datetime.now(dt.UTC)
    store = Store(store_root) if store_root is not None else None
    wanted = [m for m in config.maps if not only or m.folder in only]
    missing = sorted(set(only) - {m.folder for m in config.maps})
    if missing:
        raise BuildError(f"no such map(s) in the config: {', '.join(missing)}")
    with ExitStack() as stack:
        # One guarded read of the store for the whole run: no build while a batch is writing
        # it, and none whose inputs moved while it ran (af.store.busy). Each figure records it.
        guard = (
            stack.enter_context(store.reading("map-build", override=ignore_busy))
            if store is not None
            else None
        )
        colours = None
        if any(config.colour_source(m) == "store" for m in wanted):
            if store is None:
                raise BuildError("store colouring configured but no --store given")
            started = time.monotonic()
            from af.seq.matching_rules import matching_rules
            from af.serology.update import require_current
            from af.store import StoreError

            # A serology store built before a lab's latest tables leaves that lab's newest antigens
            # uncoloured, and nothing on the figure would say why. Refuse instead (10-serology's
            # guard, the same one geo and stat use).
            try:
                require_current(store)
            except StoreError as exc:
                raise BuildError(str(exc)) from exc
            if config.af_data is None:
                raise BuildError("store colouring needs af_data (the acmacs-f-data checkout)")
            colours = StoreColours(
                store, config.colouring, matching_rules(config.af_data), ignore_busy=ignore_busy
            )
            log(f"{'colouring':24s} {time.monotonic() - started:5.1f}s  store join and user tables")
        # Read once per run. Every map built from a round config reports its non-ferret sera; a
        # config made in code without af_data reports none.
        markers = (
            read_markers(config.af_data / SERA_MARKERS) if config.af_data is not None else None
        )
        results = []
        for cfg in wanted:
            result = build_map(
                cfg,
                config,
                store=store,
                out_root=out_root,
                vaccine_table=vaccine_list,
                vaccine_defaults=vaccine_defaults,
                created=created,
                colours=colours if config.colour_source(cfg) == "store" else None,
                sera_markers=markers,
                guard=guard,
            )
            results.append(result)
            flags = f" flags={len(result.flags)}" if result.flags else ""
            other = sum(result.lineage_minority.values())
            lineage = f" other-lineage antigens={other} {result.lineage_minority}" if other else ""
            if result.non_ferret_sera:
                lineage += f" NON-FERRET SERA={result.non_ferret_sera}"
            n = len(result.figures)
            log(f"{cfg.folder:24s} {result.seconds:5.1f}s  {n} figures{flags}{lineage}")
            for f in result.flags:
                log(f"    flag: {f}")
        return results


def main(argv: Sequence[str] | None = None) -> int:
    """CLI. Every input is explicit; nothing is read from the environment (design rule 4)."""
    parser = argparse.ArgumentParser(
        prog="python -m af.map.build", description="Build a round's antigenic map figures."
    )
    parser.add_argument("--config", type=Path, required=True, help="round config/maps.toml")
    parser.add_argument("--out", type=Path, required=True, help="figure root to write under")
    parser.add_argument("--store", type=Path, help="store root (needed for maps built from chains)")
    parser.add_argument("--only", action="append", default=[], metavar="FOLDER")
    parser.add_argument(
        "--ignore-busy",
        action="store_true",
        help="build while a batch is writing the store (diagnosis only; recorded in provenance)",
    )
    args = parser.parse_args(argv)
    try:
        from af.map.roundconfig import load_maps_config, load_vaccine_list

        config, vaccine_list_path = load_maps_config(args.config)
        results = build(
            config,
            store_root=args.store,
            out_root=args.out,
            vaccine_list=load_vaccine_list(vaccine_list_path),
            vaccine_defaults=load_vaccine_defaults(config.vaccine_defaults),
            only=args.only,
            ignore_busy=args.ignore_busy,
        )
    except (BuildError, StoreError, ValueError, OSError) as exc:
        print(f"af.map.build: {exc}", file=sys.stderr)
        return 1
    print(f"{len(results)} map(s), {sum(len(r.figures) for r in results)} figures -> {args.out}")
    return 0


def vaccine_table_for(
    tables: Mapping[str, Sequence[Any]] | Sequence[Any], row: Subtype
) -> Sequence[Any]:
    """The curated rows for one subtype, under the key acmacs-data uses for it (the subtype
    table's ``acmacs_data``: "BV" for B/Victoria). The list also carries "-disabled",
    "-seasonal" and historical tables that are NOT a current map's vaccines; only the
    subtype's own table is read. A subtype with no table is an error, not an empty map."""
    if not isinstance(tables, Mapping):
        return list(tables)  # already a flat list (a caller that selected for us)
    if row.acmacs_data not in tables:
        raise BuildError(
            f"the curated vaccine list has no table {row.acmacs_data!r} for {row.name} "
            f"(it has {', '.join(sorted(tables))})"
        )
    return tables[row.acmacs_data]


def _passage(word: str) -> Any:
    """A config passage word, checked against the classes af knows (never a silent 'any')."""
    if word not in ("any", "cell", "egg", "reassortant"):
        raise BuildError(f"unknown passage {word!r}: expected any, cell, egg or reassortant")
    return word


def _class(word: str) -> Any:
    if word not in ("cell", "egg", "reassortant"):
        raise BuildError(f"unknown passage class {word!r}: expected cell, egg or reassortant")
    return word


def relaxer_for(chart: Chart, projection_no: int = 0, projection: Any = None) -> tuple[Any, Any]:
    """`(relax, stress)` bound to this chart's optimiser problem.

    `relax(start, movable)` minimises from `start` holding everything outside `movable`, which is
    what a curation move needs: the moved points settle and the rest of the map follows. Built
    from the projection's own arrays, so a projection carrying its own forced column bases (a
    chain stage that lowered some) is measured against those, not the chart's.
    """
    from af.map.optimise import MapProblem, relax, stress

    if projection is None:
        projection = chart.projections[projection_no]
    arrays = chart.optimiser_arrays(projection.minimum_column_basis, projection)
    problem = MapProblem(**{k: v for k, v in arrays.items() if k != "start_layout"})

    def do_relax(start: FloatArray, movable: Any) -> tuple[FloatArray, float]:
        # A disconnected point takes no part in the stress, so it must not ALSO be called
        # unmovable: the optimiser rejects that combination, and "held still" means nothing for a
        # point with no coordinates.
        disconnected = np.asarray(arrays.get("disconnected"), dtype=np.bool_)
        held_still = ~np.asarray(movable, dtype=np.bool_) & ~disconnected
        held = MapProblem(
            **{k: v for k, v in arrays.items() if k not in ("start_layout", "unmovable")},
            unmovable=held_still,
        )
        result = relax(held, n_starts=1, seed=0, start_layout=np.asarray(start, dtype=float))
        best = result.projections[0]
        return np.asarray(best.layout, dtype=float), float(best.stress)

    def do_stress(layout: FloatArray) -> float:
        return stress(problem, np.asarray(layout, dtype=float))

    return do_relax, do_stress


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
