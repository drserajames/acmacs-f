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
import dataclasses
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
from af.geo.colours import DotStyle
from af.map.addenda import (
    AddendaConfig,
    AddendaContext,
    MapPage,
    load_addenda,
    run_addenda,
)
from af.map.colouring import MapColouringError, StoreColours
from af.map.config import (
    TARGET_SHIPPED,
    FrameShiftConfig,
    MapConfig,
    MapsConfig,
    MoveToConfig,
    TitlesConfig,
)
from af.map.curate import (
    SETTLED_AT_START,
    BlockOffset,
    CurationError,
    MoveOverride,
    MoveToOverride,
    MoveToResult,
    apply_block_offset,
    apply_move,
    apply_move_to,
    near_cap_flags,
)
from af.map.editor import editor_path, save_name
from af.map.figure import chart_points
from af.map.finish import EditorRequest, finish_map, shared_frame
from af.map.hand import load_hand_labels
from af.map.labels import LabelAbbreviations, label_text
from af.map.new_antigens import new_antigen_styles, outlines_on_page
from af.map.orient import (
    Orientation,
    OrientationError,
    RotationOverride,
    drawn_pairs,
    orient,
)
from af.map.render import DEFAULT_LOOK, PointStyle
from af.map.style import ColourRow, ColourScheme, PointIn, SequenceLink, Window
from af.map.vaccines import (
    MapAntigen,
    VaccineChoice,
    VaccineDisable,
    select_vaccines,
)
from af.map.viewport import Frame, from_y_down
from af.store.busy import ReadGuard
from af.store.ref import StoreError, StoreRef
from af.store.store import Store
from af.util.fonts import FigureFonts, load_fonts
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


def mover_designations(chart: Chart) -> list[str]:
    """How a move or block names its movers: each antigen's designation, date stripped, except
    where several preparations share it, which are told apart by their passage date as the chart
    writes it ("<name> MDCK2 (2020-01-12)"). A shared name could only ever match several
    antigens, which a mover refuses (design rule 1), so a group with twin preparations could not
    all be moved (Sarah, Q284 b, 31 Oct 2026: move all 48). Hides keep the dateless form: a hide
    by name means every preparation of it."""
    plain = [designation(a) for a in chart.antigens]
    shared = {d for d, n in Counter(plain).items() if n > 1}
    return [
        " ".join(p for p in (a.name, a.reassortant, *a.annotations, a.passage) if p)
        if d in shared else d
        for a, d in zip(chart.antigens, plain, strict=True)
    ]  # fmt: skip


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


def map_title(chart: Chart, configured: str | None, titles: TitlesConfig | None = None) -> str:
    """The map's own title (a map's configured one wins), named as the round's [titles] say."""
    if configured:
        return configured
    titles = titles or TitlesConfig()
    info = chart.info
    subtype = chart_subtype(chart).name
    lab = titles.labs.get(info.get("l", ""), info.get("l", ""))
    assay = ""
    if subtype in titles.assay_subtypes:
        code = info.get("A", "")
        if code not in titles.assays:
            raise BuildError(f"titles: assay {code!r} of a {subtype} map has no word in assays")
        assay = f" {titles.assays[code]}"
    return f"{lab} {subtype}{assay} by clade"


# ---------------------------------------------------------------- steps


def chain_chart(
    store: Store, dataset: str, until_table: str | None = None
) -> tuple[Chart, StoreRef, Path]:
    """The chart of a chain's CURRENT version: its last step, or the step after ``until_table``.

    Resolved at run time: a pinned version goes stale the moment the chain is rebuilt, and the
    report builder refuses a stale figure. ``until_table`` names a table id; it must match
    exactly one step (design rule 1), and a step index is never accepted (rule 2).
    """
    return _chain_step_chart(store, dataset, store.current("chains", dataset), until_table)


def chain_version_chart(store: Store, dataset: str, version: str) -> tuple[Chart, StoreRef, Path]:
    """The chart of one named chain version (its last step): a report's map as it was drawn,
    which a later report compares with (af.map.new_antigens). The version is pinned, not CURRENT."""
    from af.store.manifest import MANIFEST, Manifest
    from af.store.store import VERSIONS

    manifest = store.dataset_dir("chains", dataset) / VERSIONS / version / MANIFEST
    if not manifest.is_file():
        raise BuildError(f"no chain version chains/{dataset}@{version} in the store")
    ref = StoreRef("chains", dataset, version, Manifest.from_bytes(manifest.read_bytes()).sha256())
    return _chain_step_chart(store, dataset, ref, None)


def _chain_step_chart(
    store: Store, dataset: str, ref: StoreRef, until_table: str | None
) -> tuple[Chart, StoreRef, Path]:
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
    designations = mover_designations(chart)
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
            flags.extend(near_cap_flags("move", m.name, False, exc.checks))
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
        flags.extend(near_cap_flags("move", m.name, True, result.checks))
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


@dataclass(frozen=True)
class PendingMoveTo:
    """A move_to that has been applied, waiting for the map's final layout to be measured in the
    frame its target was stated in (check_moves_to)."""

    config: MoveToConfig
    result: MoveToResult
    movers: tuple[str, ...]
    requested: np.ndarray  # shipped: reference_round's DRAWN positions; [x, y]: this map's drawn
    placement_rmsd: float  # the fit that carried the targets into the raw layout
    placement_common: int
    frame: str  # which frame requested/achieved are in
    # shipped only: the reference's drawn layout and the (row here, row there) pairs, movers
    # excluded, to refit the final layout onto; and what was fitted to
    reference_drawn: np.ndarray | None = None
    reference_pairs: NDArray[np.intp] | None = None
    reference_record: dict[str, str] | None = None


def _serum_movers(chart: Chart, wanted: Sequence[str], designations: Sequence[str]) -> list[str]:
    """Movers that name no antigen but do name a serum: moves_to moves antigens only (Q354 Q6),
    and saying so beats "matches 0 antigens"."""
    sera = {designation(s) for s in chart.sera} | {s.name for s in chart.sera}
    return [w for w in wanted if w not in designations and w in sera]


def _shipped_reference(
    chart: Chart, m: MoveToConfig, folder: str, scheme_name: str, at: str
) -> tuple[np.ndarray, NDArray[np.intp], dict[str, str]]:
    """The reference round's drawn map, the drawn common points with it, and the record of what
    was read. Missing folder, missing styled.ace: BuildError (design rule 4)."""
    assert m.reference_round is not None
    if not m.reference_round.is_dir():
        raise BuildError(f"{at}: reference_round {m.reference_round} is not a directory")
    styled = m.reference_round / folder / "styled.ace"
    if not styled.is_file():
        raise BuildError(f"{at}: reference_round has no {folder}/styled.ace ({styled})")
    ref = read_chart(styled)
    pairs, drawn = common_points(chart, ref, scheme_name)
    record = {
        "round": m.reference_round.name,
        "path": str(styled),
        "sha256": content_hash(styled),
    }
    return from_y_down(ref.projections[0].transformed_layout()), drawn_pairs(pairs, drawn), record


def _refit(
    layout: np.ndarray, reference_drawn: np.ndarray, pairs: NDArray[np.intp], at: str, need: int
) -> Orientation:
    try:
        return orient(layout, reference_drawn, pairs, reference=at, min_common=need)
    except OrientationError as exc:
        raise BuildError(f"{at}: {exc}") from exc


def apply_moves_to(
    chart: Chart,
    layout: np.ndarray,
    cfg: MapConfig,
    config: MapsConfig,
    relax: Any,
    stress: Any,
) -> tuple[np.ndarray, list[PendingMoveTo]]:
    """Run the map's moves_to in order (after moves, before blocks).

    Each target is carried into the raw layout by a rigid fit that leaves this move's movers
    out, so they do not steer the fit that places them:

    * "shipped": a fit of this map onto ``reference_round``'s drawn map over the points both
      draw. That round need not be the one the build orients to (Q354 Q1').
    * [x, y]: the map's own orientation (orientation_for: the same previous round and rotations),
      computed without the movers. The final orientation is made later, unchanged.

    check_moves_to measures the result in the same frame after every other step.

    Unlike moves and blocks, which flag a refusal on the figure and carry on, every failure here
    is a BuildError (00-v21, Q354 plan Q2). This form exists to put a point somewhere specific,
    so a refused or empty move means the figure is not the one that was decided.
    """
    if not cfg.moves_to:
        return layout, []
    where = cfg.folder
    designations = mover_designations(chart)
    moved_by_moves = {d for m in cfg.moves for d in m.load_movers()}
    need = config.defaults.min_common_points
    pending: list[PendingMoveTo] = []
    for m in cfg.moves_to:
        at = f"{where}: move_to {m.name!r}"
        movers = m.load_movers()
        if both := sorted(moved_by_moves & set(movers)):
            raise BuildError(f"{at}: {both[0]!r} is also moved by a moves rule")
        if sera := _serum_movers(chart, movers, designations):
            raise BuildError(f"{at}: {sera[0]!r} is a serum; moves_to moves antigens only")
        rows = []
        for want in movers:
            hits = [i for i, d in enumerate(designations) if d == want]
            if len(hits) != 1:
                raise BuildError(f"{at}: mover {want!r} matches {len(hits)} antigens")
            rows.append(hits[0])
        extra: dict[str, Any] = {}
        if m.target == TARGET_SHIPPED:
            ref_drawn, ref_pairs, record = _shipped_reference(chart, m, where, cfg.clade_scheme, at)
            there = {int(i): int(j) for i, j in ref_pairs}
            if missing := [w for w, r in zip(movers, rows, strict=True) if r not in there]:
                raise BuildError(
                    f"{at}: mover {missing[0]!r} is not drawn, once, on {record['round']}'s map"
                )
            requested = ref_drawn[[there[r] for r in rows]]
            if not np.isfinite(requested).all():
                raise BuildError(f"{at}: a mover has no position on {record['round']}'s map")
            without = ref_pairs[~np.isin(ref_pairs[:, 0], rows)]
            fit = _refit(layout, ref_drawn, without, at, need).fit
            matrix, translation, rmsd, common = fit.matrix, fit.translation, fit.rmsd, fit.n_common
            frame = f"{record['round']} drawn"
            extra = dict(
                reference_drawn=ref_drawn, reference_pairs=without, reference_record=record
            )
        else:
            assert isinstance(m.target, tuple)
            previous_round = config.defaults.previous_round
            if previous_round is None or not (previous_round / where / "styled.ace").is_file():
                raise BuildError(
                    f"{at}: an [x, y] target is read in the frame of the previous round it was "
                    "decided against, but this build orients to none"
                )
            if m.reference != previous_round.name:
                raise BuildError(
                    f"{at} was decided against {m.reference}, but this build orients to "
                    f"{previous_round.name}: re-decide it against {previous_round.name}"
                )
            in_force = tuple(r.name for r in cfg.rotations)
            if m.frame_rotations != in_force:
                raise BuildError(
                    f"{at}: its coordinates were read with rotations "
                    f"{list(m.frame_rotations or ())}, but the map now has {list(in_force)}: "
                    "re-read them in the current frame"
                )
            ori = orientation_for(chart, layout, cfg, config, cfg.clade_scheme, exclude=rows)
            assert ori is not None  # a previous round exists, checked above
            requested = np.tile(np.asarray(m.target, dtype=float), (len(rows), 1))
            matrix, translation = ori.matrix, ori.translation
            rmsd, common, frame = ori.fit.rmsd, ori.fit.n_common, "this map drawn"
        targets = (requested - translation) @ matrix.T  # orthogonal: the inverse is the transpose
        rule = MoveToOverride(
            m.name,
            m.reason,
            m.decided.isoformat(),
            tuple(movers),
            m.max_stress_rise,
            m.max_from_target,
            m.relax,
        )
        try:
            result = apply_move_to(
                rule,
                layout,
                designations,
                targets,
                stress_before=stress(layout),
                stress=stress,
                relax=relax,
            )
        except CurationError as exc:
            raise BuildError(f"{where}: {str(exc).split('; layout left')[0]}") from exc
        layout = result.layout
        pending.append(
            PendingMoveTo(m, result, tuple(movers), requested, rmsd, common, frame, **extra)
        )
    return layout, pending


def check_moves_to(
    pending: Sequence[PendingMoveTo],
    layout: np.ndarray,
    ori: Orientation | None,
    folder: str,
    min_common: int,
) -> list[dict[str, Any]]:
    """Measure every applied move_to in its target's frame, from the map's FINAL layout, and
    record it (design rule 9).

    from_target_final is the distance between where the mover finally sits and where it was asked
    to be. For a shipped target, that is after refitting the final layout onto the reference's
    drawn map, movers left out. For an [x, y] target, it is through the map's final orientation,
    rotations included, which is where it is drawn. It is checked against the same
    max_from_target as the relative distance (apply_move_to), because blocks, or the final
    orientation, which keeps the movers in it, can still pull a mover off.
    """
    out = []
    for p in pending:
        at = f"{folder}: move_to {p.config.name!r}"
        if p.reference_drawn is not None:
            assert p.reference_pairs is not None
            final = _refit(layout, p.reference_drawn, p.reference_pairs, at, min_common).fit
            m, t, final_rmsd, final_common = (
                final.matrix,
                final.translation,
                final.rmsd,
                final.n_common,
            )
        else:
            if ori is None:  # cannot happen: an [x, y] move_to needs the previous round
                raise BuildError(f"{at}: no final orientation")
            m, t, final_rmsd, final_common = (
                ori.matrix,
                ori.translation,
                ori.fit.rmsd,
                ori.fit.n_common,
            )
        achieved = layout[list(p.result.rows)] @ m + t
        from_target = np.linalg.norm(achieved - p.requested, axis=1)
        if (worst := float(from_target.max())) > p.config.max_from_target:
            k = int(np.argmax(from_target))
            raise BuildError(
                f"{at}: mover {p.movers[k]!r} ends {worst:.3f} from its target in the final "
                f"layout (cap {p.config.max_from_target}; relative to the rest it is "
                f"{p.result.relative_from_target[k]:.3f})"
            )
        r = p.result
        rise = r.stress_after - r.stress_before
        shipped = p.config.target == TARGET_SHIPPED
        out.append(
            {
                "override": p.config.name,
                "reason": p.config.reason,
                "decided": p.config.decided.isoformat(),
                "applied": True,
                "target": p.config.target if shipped else list(p.config.target),
                "reference": p.reference_record if shipped else p.config.reference,
                "relax": p.config.relax,
                "movers": len(p.movers),
                "stress_before": round(float(r.stress_before), 4),
                "stress_after": round(float(r.stress_after), 4),
                "stress_rise": round(float(rise), 4),
                "stress_rise_pct": round(100 * float(rise) / float(r.stress_before), 4)
                if r.stress_before
                else None,
                # the frame every relative distance is in: the rest of the map, aligned
                "rest_alignment": {
                    "rmsd": round(r.rest_fit_rmsd, 4),
                    "det": round(r.rest_fit_det, 4),
                    "largest_other_move": round(r.largest_other_move, 4),
                },
                # requested/achieved are in this frame: the one the target was stated in
                "frame": p.frame,
                "fit": {
                    "placement_rmsd": round(p.placement_rmsd, 4),
                    "placement_common": p.placement_common,
                    "final_rmsd": round(float(final_rmsd), 4),
                    "final_common": final_common,
                },
                "per_mover": [
                    {
                        "designation": p.movers[k],
                        "requested": [round(float(v), 4) for v in p.requested[k]],
                        "achieved": [round(float(v), 4) for v in achieved[k]],
                        "moved_relative": round(r.relative_moved[k], 4),
                        "from_target_relative": round(r.relative_from_target[k], 4),
                        "from_target_final": round(float(from_target[k]), 4),
                    }
                    for k in range(len(p.movers))
                ],
                "checks": [c.fields() for c in r.checks]
                + [
                    {
                        "guard": "from_target_final",
                        "measured": round(worst, 4),
                        "limit": p.config.max_from_target,
                        "bound": "max",
                    }
                ],
            }
        )
    return out


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
    chart: Chart,
    layout: np.ndarray,
    cfg: MapConfig,
    config: MapsConfig,
    scheme_name: str,
    exclude: Sequence[int] = (),
) -> Orientation | None:
    """Fit to the previous round's map, over the points that map actually DREW.

    A hand rotation turns the map from where the fit to ITS reference leaves it, so it is refused
    against any other reference, which may already carry the turn (M2's B/Vic map drew turned
    132° when one was carried to a round oriented on the map that held it), and refused when
    there is no reference at all, where it would silently not be applied.

    ``exclude`` (rows of this chart) leaves points out of the fit: a move_to carries its targets
    with a fit that the movers themselves must not steer (apply_moves_to). Empty for the map's
    own orientation, which is unchanged by it."""
    previous = (
        None
        if config.defaults.previous_round is None
        else config.defaults.previous_round / cfg.folder / "styled.ace"
    )
    if previous is None or not previous.is_file():
        if cfg.rotations:
            raise BuildError(
                f"{cfg.folder}: rotation {cfg.rotations[0].name!r} needs the previous round it was "
                f"decided against ({cfg.rotations[0].reference}), but this build orients to none"
            )
        return None
    round_name = previous.parent.parent.name
    for r in cfg.rotations:
        if r.reference != round_name:
            raise BuildError(
                f"{cfg.folder}: rotation {r.name!r} was decided against {r.reference}, but this "
                f"build orients to {round_name}, which may already carry it: re-decide it against "
                f"{round_name} or drop it"
            )
    prev = read_chart(previous)
    pairs, prev_drawn = common_points(chart, prev, scheme_name)
    if len(exclude):
        pairs = pairs[~np.isin(pairs[:, 0], np.asarray(exclude, dtype=np.intp))]
    overrides = tuple(
        RotationOverride(r.name, r.degrees, r.reason, r.decided.isoformat(), r.reflect, r.reference)
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
    # Sentences for antigens left unpainted by clade evidence (af.geo.colours.unpainted_warning),
    # also in each figure's decisions.unpainted_clades; printed in the build log.
    warnings: tuple[str, ...] = ()
    # What an addendum draws from the map (af.map.addenda); kept only when addenda are built.
    page: MapPage | None = None


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
    fonts: FigureFonts | None = None,
    abbreviations: LabelAbbreviations | None = None,
    keep_page: bool = False,
) -> MapResult:
    """Build every window of one map. Raises :class:`BuildError` with the folder named.

    ``keep_page`` keeps what the map's addenda draw from it (:class:`af.map.addenda.MapPage`).

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
        dots: Sequence[DotStyle] = coloured.styles
        sequenced = list(coloured.sequenced)
        sequences: Sequence[SequenceLink] | None = coloured.sequences
        store_refs.extend(colours.store_refs(coloured.provenance["datasets"]))
        colour_note: dict[str, Any] = coloured.provenance
    else:
        dots = ()
        scheme, labels = _stand_in_colours(chart, cfg, inputs)
        sequenced = [bool(a.extra.get("A")) for a in chart.antigens]
        sequences = None  # no join: a stand-in chart's own rows colour it
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
    layout, moves_to = apply_moves_to(chart, layout, cfg, config, relax, stress)
    for b in cfg.blocks:
        row_b = next((r for r in scheme.rows if r.legend == b.target_legend), None)
        if row_b is None and not (b.settled_against == SETTLED_AT_START and not b.target_legend):
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
            row_b.colour if row_b is not None else "",
            b.max_stress_rise,
            b.settled_within,
            b.min_settled,
            b.max_other_move,
            to=b.to,
            settled_against=b.settled_against,
        )
        try:
            result_b = apply_block_offset(
                rule_b,
                layout,
                mover_designations(chart),
                painted,
                stress_before=stress(layout),
                relax=relax,
            )
        except CurationError as exc:
            flags.append(str(exc).split(";")[0])
            flags.extend(near_cap_flags("block", b.name, False, exc.checks))
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
        flags.extend(near_cap_flags("block", b.name, True, result_b.checks))
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
    if moves_to:
        decisions["moves_to"] = check_moves_to(
            moves_to, layout, ori, cfg.folder, config.defaults.min_common_points
        )

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
            # the shipped maps label a reassortant vaccine "egg" (style 08-27); its reassortant
            # name stays in the I7
            "egg" if m.passage_class == "reassortant" else m.passage_class,
            chart.antigens[m.antigen].reassortant,
            abbreviations.table if abbreviations is not None else {},
        )
        for m in vrep.marks
    }
    if abbreviations is not None:  # the place abbreviations the labels used (rule 5)
        inputs["label_abbreviations"] = abbreviations.input()
    if vrep.unused_optional_rules:
        decisions["vaccine_rules_unused_optional"] = list(vrep.unused_optional_rules)
    if disable or choose:
        decisions["vaccines"] = vaccine_rule_records(disable, choose, n_defaults, vrep.used_rules)
    vaccine_flags: list[str] = []
    if colours is not None and vrep.marks:
        # Sarah, Q184: a vaccine is coloured by its named strain's original clade, before
        # passage mutation. Store colours only: a stand-in chart's rows are the round's own.
        # Recoloured after the layout's curation, which reads the colours before vaccines exist.
        try:
            found = colours.for_vaccines(
                chart, [(m.antigen, m.passage_class) for m in vrep.marks], dots, cfg.clade_scheme
            )
        except (MapColouringError, ValueError) as exc:
            raise BuildError(f"{cfg.folder}: vaccine colours: {exc}") from exc
        recoloured = list(dots)
        for v in found:
            recoloured[v.antigen], labels[v.antigen] = v.style, v.labels
            sequenced[v.antigen] = sequenced[v.antigen] or v.style.colour is not None
        dots = recoloured
        decisions["vaccine_colours"] = [v.record() for v in found]
        vaccine_flags = [v.warning for v in found if v.warning]

    # A vaccine recoloured from its named strain keeps its own preparation's sequence: the link
    # says which virus the point is, not where its colour came from.
    points = chart_points(
        chart, xy, labels=labels, sequenced=sequenced, hidden=hidden, sequences=sequences
    )
    unpainted = unpainted_clades(chart, points, dots, scheme.name)
    if unpainted:
        decisions["unpainted_clades"] = unpainted
    placed = placement_clades(chart, points, dots)
    if placed:
        decisions["painted_by_placement"] = placed

    size = config.frame_size(chart.info.get("V", ""), chart.info.get("A", ""))
    legend = config.legend(chart.info.get("V", ""), chart.info.get("A", ""))
    look = DEFAULT_LOOK
    if legend is not None:
        look = dataclasses.replace(
            DEFAULT_LOOK,
            legend_text_size=legend.text_size,
            legend_row_pitch=legend.row_pitch,
            legend_text_left=legend.text_left,
            legend_padding=legend.padding,
        )
    if legend is not None:  # the round's own legend for this subtype, recorded (rule 5)
        decisions["legend"] = legend.look()
    title_base = map_title(chart, cfg.title, config.titles)
    if fonts is not None:  # the font files the figure is drawn in (rule 5)
        inputs.update(fonts.inputs())
    hand = None
    if cfg.hand_labels is not None:
        # the map's labels placed by hand (af.map.hand): an input, and its rows recorded (rule 9)
        hand = load_hand_labels(cfg.hand_labels)
        inputs["hand_labels"] = {
            "path": str(cfg.hand_labels),
            "sha256": hashlib.sha256(cfg.hand_labels.read_bytes()).hexdigest(),
        }
        decisions["hand_labels"] = hand.record(cfg.hand_labels)
    note: tuple[str, ...] = ()
    if cfg.note is not None:  # a short note in the frame, on every page (Q323)
        note = cfg.note.text
        decisions["note"] = cfg.note.record()
    outlines: dict[str, PointStyle] | None = None
    outline_record: list[dict[str, Any]] = []
    if config.new_antigen_outlines:
        # the interim reports' outlines for antigens new since the previous reports (Q264 b);
        # each page keeps those it draws in colour, and records its own counts
        outlines, outline_record = new_antigen_outlines(
            chart, points, cfg, config, store, frozenset(vlabels), inputs, store_refs
        )
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
    windows = [
        (
            Window(w.name, w.since),
            title_base + ("" if w.since is None else f" (since {w.since.strftime('%B %Y')})"),
        )
        for w in config.defaults.windows
    ]
    # one frame on all of the map's pages, as shipped (R5 review, 19 Oct 2026)
    frame = shared_frame(
        points,
        scheme=scheme,
        windows=windows,
        frame_size=size,
        must_show_since=config.defaults.must_show_since,
        vaccine_labels=vlabels,
        look=look,
        note=note,
    )
    if cfg.frame_shift is not None:
        # every page then checks the moved frame still shows the recent antigens
        frame, record = shift_frame(frame, cfg.frame_shift)
        provenance.setdefault("decisions", {})["frame_shift"] = record
    editor = None
    if hand is not None:
        # the map's label editor, from its first window's page (af.map.editor)
        chain_input = inputs.get("chain") or {}
        editor = EditorRequest(
            editor_path(out_root, cfg.folder),
            cfg.folder,
            f"{chain_input.get('dataset', cfg.chain)}@{chain_input.get('version', '?')}",
            save_name(cfg.hand_labels, cfg.folder),
        )
    for window, title in windows:
        out = out_root / "map" / cfg.folder / window.name / "v1" / "figure.pdf"
        page_styles, page_provenance = outlines, provenance
        if outlines is not None:
            page_styles, page_record = outlines_on_page(
                points, scheme, window, vlabels, outlines, outline_record
            )
            page_decisions = {
                **provenance.get("decisions", {}),
                "new_antigen_outlines": page_record,
            }
            page_provenance = {**provenance, "decisions": page_decisions}
        result = finish_map(
            points,
            chart=cfg.folder,
            scheme=scheme,
            window=window,
            title=title,
            frame_size=size,
            frame=frame,
            must_show_since=config.defaults.must_show_since,
            vaccine_labels=vlabels,
            out_pdf=out,
            created=created,
            provenance=page_provenance,
            orientation=ori.report() if ori else None,
            flags=flags,
            look=look,
            styles=page_styles,
            hand=hand,
            editor=editor,
            note=note,
        )
        editor = None  # one editor per map
        figures.append(result.i7)
    return MapResult(
        cfg.folder,
        tuple(figures),
        tuple(flags),
        time.monotonic() - started,
        dict(decisions.get("lineage_minority", {}).get("other", {})),
        non_ferret_count,
        (*(u["warning"] for u in [*unpainted, *placed]), *vaccine_flags),
        _map_page(
            cfg,
            chart,
            projection,
            points,
            scheme,
            frame,
            look,
            windows,
            config,
            provenance,
            vlabels,
        )  # fmt: skip
        if keep_page
        else None,
    )


def shift_frame(frame: Frame, shift: FrameShiftConfig) -> tuple[Frame, dict[str, Any]]:
    """A named hand move of the searched frame (design rule 9), and the record every figure of
    the map carries: the move, its reason, and the frame searched and drawn."""
    moved = dataclasses.replace(frame, x=frame.x + shift.dx, y=frame.y + shift.dy)
    return moved, {
        "name": shift.name,
        "reason": shift.reason,
        "decided": shift.decided.isoformat(),
        "dx": shift.dx,
        "dy": shift.dy,
        "searched": [frame.x, frame.y, frame.size],
        "drawn": [moved.x, moved.y, moved.size],
    }


def new_antigen_outlines(
    chart: Chart,
    points: Sequence[PointIn],
    cfg: MapConfig,
    config: MapsConfig,
    store: Store | None,
    marks: frozenset[str],
    inputs: dict[str, Any],
    store_refs: list[dict[str, str]],
) -> tuple[dict[str, PointStyle], list[dict[str, Any]]]:
    """Each tier's previous report map, read from its pinned chain version and recorded as an
    input of the figure (design rule 5), then the outlines (af.map.new_antigens)."""
    if store is None:
        raise BuildError(f"{cfg.folder}: new-antigen outlines need --store (the previous maps)")
    loaded: dict[tuple[str, str], Chart] = {}
    previous = []
    for tier in config.new_antigen_outlines:
        try:
            dataset, version = tier.previous_map(cfg.folder)
        except ValueError as exc:
            raise BuildError(f"{cfg.folder}: {exc}") from exc
        if (dataset, version) not in loaded:
            prev, ref, ace = chain_version_chart(store, dataset, version)
            loaded[dataset, version] = prev
            store_refs.append(ref.to_json())
            inputs[f"previous_map {dataset}@{version}"] = {
                **ref.to_json(),
                "sha256": content_hash(ace),
            }
        previous.append(loaded[dataset, version])
    return new_antigen_styles(chart, points, config.new_antigen_outlines, previous, marks)


def _map_page(
    cfg: MapConfig,
    chart: Chart,
    projection: Any,
    points: Sequence[PointIn],
    scheme: ColourScheme,
    frame: Any,
    look: Any,
    windows: Sequence[tuple[Window, str]],
    config: MapsConfig,
    provenance: dict[str, Any],
    vaccine_labels: Mapping[str, str],
) -> MapPage:
    if not cfg.chain:
        raise BuildError(f"{cfg.folder}: an addendum needs a map drawn from a chain")
    return MapPage(
        folder=cfg.folder,
        centre=cfg.chain.split("/")[0],
        chain_ref=dict(provenance["inputs"]["chain"]),
        chart=chart,
        projection=projection,
        points=tuple(points),
        scheme=scheme,
        frame=frame,
        look=look,
        windows=tuple(windows),
        must_show_since=config.defaults.must_show_since,
        provenance=provenance,
        vaccine_labels=dict(vaccine_labels),
        clade_scheme=cfg.clade_scheme,
        colour_source=config.colour_source(cfg),
    )


def unpainted_clades(
    chart: Chart, points: Sequence[PointIn], styles: Sequence[DotStyle], scheme: str
) -> list[dict[str, Any]]:
    """One record per supported clade the scheme cannot paint (and one for antigens whose
    sequences support no clade), over the antigens this map draws: Sarah, 8 Oct 2026, a warning,
    never a silent count. The sentence is af.geo.colours.unpainted_warning's, copied verbatim
    downstream (11-reports). Drawn = has coordinates and no hide rule; every window draws the
    same antigens (windows only grey), so one record serves all of them."""
    from af.geo.colours import STATE_NO_SCHEME_ROW, STATE_NO_SUPPORTED, unpainted_warning

    groups: dict[str | None, dict[str, Any]] = {}
    for p, dot in zip(points[: chart.n_antigens], styles, strict=False):
        if dot.state not in (STATE_NO_SCHEME_ROW, STATE_NO_SUPPORTED) or p.xy is None or p.hide:
            continue
        clade = dot.clade if dot.state == STATE_NO_SCHEME_ROW else None
        g = groups.setdefault(clade, {"points": 0, "reference": 0, "tree": Counter()})
        g["points"] += 1
        g["reference"] += bool(p.reference)
        g["tree"][dot.tree_clade] += 1
    out = []
    for clade, g in sorted(groups.items(), key=lambda kv: (kv[0] is None, kv[0] or "")):
        out.append(
            {
                "clade": clade,
                "tree_clades": dict(sorted(g["tree"].items())),
                "points": g["points"],
                "reference_antigens": g["reference"],
                "scheme": scheme,
                "warning": unpainted_warning(
                    g["points"], clade, g["tree"], scheme, reference=g["reference"]
                ),
            }
        )
    return out


def placement_clades(
    chart: Chart, points: Sequence[PointIn], styles: Sequence[DotStyle]
) -> list[dict[str, Any]]:
    """The antigens this map draws painted by tree placement alone (Sarah, Q183 b, 14 Oct 2026:
    a contradicted clade with no supported ancestor is coloured by placement, flagged): one
    record with af.geo.colours.placement_warning's sentence, copied verbatim downstream
    (11-reports), or none. Drawn as for :func:`unpainted_clades`."""
    from af.geo.colours import BASIS_PLACEMENT, placement_warning

    n, reference, tree = 0, 0, Counter[str]()
    for p, dot in zip(points[: chart.n_antigens], styles, strict=False):
        if dot.basis != BASIS_PLACEMENT or p.xy is None or p.hide:
            continue
        n += 1
        reference += bool(p.reference)
        tree[dot.tree_clade] += 1
    if not n:
        return []
    return [
        {
            "tree_clades": dict(sorted(tree.items())),
            "points": n,
            "reference_antigens": reference,
            "warning": placement_warning(n, tree, reference=reference),
        }
    ]


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
    addenda: Path | None = None,
) -> list[MapResult]:
    """Build every configured map (or just ``only``). Raises on the first failure.

    ``addenda`` (a round's addenda.toml): after the maps, draw the addenda from them
    (:mod:`af.map.addenda`). Without it nothing about the maps changes.

    ``ignore_busy`` builds while a batch is writing the store, for diagnosis only (the command
    line's explicit ``--ignore-busy``); every figure's provenance then says it did."""
    created = created or dt.datetime.now(dt.UTC)
    store = Store(store_root) if store_root is not None else None
    wanted = [m for m in config.maps if not only or m.folder in only]
    missing = sorted(set(only) - {m.folder for m in config.maps})
    if missing:
        raise BuildError(f"no such map(s) in the config: {', '.join(missing)}")
    addenda_config = load_addenda(addenda) if addenda is not None else None  # fail early
    with ExitStack() as stack:
        # One guarded read of the store for the whole run: no build while a batch is writing
        # it, and none whose inputs moved while it ran (af.store.busy). Each figure records it.
        guard = (
            stack.enter_context(store.reading("map-build", override=ignore_busy))
            if store is not None
            else None
        )
        colours = None
        # An addendum reads its sera's strains from the store join, whatever colours its maps
        # take: a stand-in map keeps its stand-in colours (only the maps below get colours).
        if addenda_config is not None or any(config.colour_source(m) == "store" for m in wanted):
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
        # The whole drawing, label placement included, runs in the round's fonts (af.util.fonts).
        fonts = load_fonts(config.figures) if config.figures is not None else None
        if fonts is not None:
            stack.enter_context(fonts.context())
        abbreviations = (
            LabelAbbreviations.read(config.label_abbreviations)
            if config.label_abbreviations is not None
            else None
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
                fonts=fonts,
                abbreviations=abbreviations,
                keep_page=addenda_config is not None,
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
            for w in result.warnings:
                log(f"    {w}")
        if addenda_config is not None:
            _build_addenda(addenda_config, results, colours, config, out_root, created, log)
        return results


def _build_addenda(
    addenda_config: AddendaConfig,
    results: Sequence[MapResult],
    colours: StoreColours | None,
    config: MapsConfig,
    out_root: Path,
    created: dt.datetime,
    log: Any,
) -> None:
    from af.seq.matching_rules import LABS
    from af.tables.labs import read_labs

    assert colours is not None  # built whenever addenda are
    if config.af_data is None:
        raise BuildError("addenda need af_data (the labs table)")
    pages = {r.folder: r.page for r in results if r.page is not None}
    ctx = AddendaContext(
        colours=colours,
        since=addenda_config.since,
        out_root=out_root,
        created=created,
        labs={lab.code: lab for lab in read_labs(config.af_data / LABS)},
    )
    started = time.monotonic()
    for a in run_addenda(addenda_config, pages, ctx):
        log(
            f"{a.directory.name:24s} {time.monotonic() - started:5.1f}s  "
            f"{len(a.figures)} figures, sera {a.selection['sections']}"
        )


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
    parser.add_argument(
        "--addenda", type=Path, help="round config/addenda.toml: also draw the addenda's figures"
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
            addenda=args.addenda,
        )
    except (BuildError, StoreError, ValueError, OSError) as exc:  # FontError is a ValueError
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
