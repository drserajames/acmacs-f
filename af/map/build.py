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
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from af.chart.ace import content_hash, read_chart
from af.chart.model import Chart
from af.map.config import MapConfig, MapsConfig
from af.map.curate import CurationError, MoveOverride, apply_move
from af.map.finish import finish_map
from af.map.labels import label_text
from af.map.orient import Orientation, RotationOverride, drawn_pairs, orient
from af.map.style import ColourRow, ColourScheme, PointIn, Window
from af.map.vaccines import (
    MapAntigen,
    VaccineChoice,
    VaccineDisable,
    passage_class,
    select_vaccines,
)
from af.store.ref import StoreRef
from af.store.store import Store

FloatArray = NDArray[np.float64]

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


def parse_date(text: str) -> dt.date | None:
    """An isolation date, parsed. Never inferred from anything else (design rule 7)."""
    try:
        return dt.date.fromisoformat(text[:10])
    except (ValueError, TypeError):
        return None


def table_counts(chart: Chart) -> tuple[list[int], list[dt.date | None]]:
    """Per antigen: in how many source tables it has a titre, and the latest such table's date."""
    n = chart.n_antigens
    counts = [0] * n
    latest: list[dt.date | None] = [None] * n
    seen_per_layer: set[tuple[int, int]] = set()
    del seen_per_layer
    sources = chart.info.get("S") or []
    for k, layer in enumerate(chart.titres.layers):
        when = _compact(sources[k]) if k < len(sources) else None
        for i, _j in layer:
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


def map_title(chart: Chart, configured: str | None) -> str:
    if configured:
        return configured
    info = chart.info
    assay = {"HI": "", "HINT": " HINT", "PRN": " neut", "FOCUS REDUCTION+FRA": " neut"}.get(
        info.get("A", ""), f" {info.get('A', '')}".rstrip()
    )
    subtype = {"B": "B/Vic"}.get(info.get("V", ""), info.get("V", ""))
    return f"{info.get('l', '')} {subtype}{assay} by clade"


# ---------------------------------------------------------------- steps


def chain_chart(store: Store, dataset: str) -> tuple[Chart, StoreRef, Path]:
    """The last step of a chain's CURRENT version. Resolved at run time: a pinned version goes
    stale the moment the chain is rebuilt, and the report builder refuses a stale figure."""
    ref = store.current("chains", dataset)
    root = store.resolve(ref)
    steps = json.loads((root / "chain.json").read_text())["steps"]
    if not steps:
        raise BuildError(f"chain {dataset} has no steps")
    last = steps[-1]
    ace = root / last["directory"] / last["chosen_file"]
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
            tuple(m.movers),
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
            reports.append({"override": m.name, "applied": False, "why": str(exc).split(";")[0]})
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
    return orient(
        layout,
        prev.projections[0].transformed_layout(),
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


# ---------------------------------------------------------------- one map


@dataclass(frozen=True)
class MapResult:
    folder: str
    figures: tuple[Path, ...]
    flags: tuple[str, ...]
    seconds: float


def build_map(
    cfg: MapConfig,
    config: MapsConfig,
    *,
    store: Store | None,
    out_root: Path,
    vaccine_table: Sequence[Any],
    vaccine_defaults: dict[str, tuple[VaccineDisable, ...]],
    created: dt.datetime,
) -> MapResult:
    """Build every window of one map. Raises :class:`BuildError` with the folder named."""
    started = time.monotonic()
    inputs: dict[str, Any] = {}
    stand_in: dict[str, str] = {}
    store_refs: list[dict[str, str]] = []
    if cfg.chain:
        if store is None:
            raise BuildError(f"{cfg.folder}: chain configured but no --store given")
        chart, ref, ace = chain_chart(store, cfg.chain)
        store_refs.append(ref.to_json())
        inputs["chain"] = {**ref.to_json(), "sha256": content_hash(ace)}
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
    scheme_chart = read_chart(cfg.scheme_stand_in) if cfg.scheme_stand_in else chart
    scheme = scheme_from_chart(scheme_chart, cfg.clade_scheme)
    if cfg.scheme_stand_in:
        inputs["colour_scheme_chart"] = {
            "path": str(cfg.scheme_stand_in),
            "sha256": content_hash(cfg.scheme_stand_in),
            "note": "STAND-IN: colour rows AND clade labels, until clade assignment is wired in",
        }
    # Clade labels come from the same chart as the colour rows: a chart written before the round's
    # clade step carries neither. Workstream 4's assignment replaces both.
    labels_of = _labels_by_designation(scheme_chart) if cfg.scheme_stand_in else None

    def labels_for(index: int, antigen: Any) -> frozenset[str]:
        if labels_of is None:
            return clade_labels(antigen)
        return labels_of.get(designation(antigen), frozenset())

    inputs["colour_scheme"] = {
        "name": scheme.name,
        "sha256": hashlib.sha256(
            json.dumps([[r.legend, r.colour, sorted(r.labels)] for r in scheme.rows]).encode()
        ).hexdigest(),
        "note": "STAND-IN: the chart's own rows, until user colour schemes are wired in",
    }
    layout = chart.projections[0].layout.copy()
    painted = [
        (lambda row: row.colour if row else None)(scheme.paint(labels_for(i, a)))
        for i, a in enumerate(chart.antigens)
    ]
    relax, stress = relaxer_for(chart)
    layout, flags, move_reports = apply_moves(chart, layout, cfg, scheme, painted, relax, stress)
    # What was decided about this map, as opposed to what it was built FROM: these carry reasons,
    # not content hashes, so they are not inputs.
    decisions: dict[str, Any] = {}
    if move_reports:
        decisions["moves"] = move_reports
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
    if vaccine_defaults and subtype not in vaccine_defaults:
        decisions["vaccine_defaults"] = f"no subtype defaults for {subtype!r}"
    for d in cfg.vaccine_disable:
        disable.append(VaccineDisable(d.name, _passage(d.passage), d.reason, d.optional))
    choose = [
        VaccineChoice(c.name, _class(c.passage_class), c.passage, c.reason, c.optional)
        for c in cfg.vaccine_choose
    ]
    vrep = select_vaccines(ags, vaccine_table, disable=disable, choose=choose)
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

    points: list[PointIn] = []
    for i, a in enumerate(chart.antigens):
        ok = bool(np.isfinite(xy[i]).all())
        points.append(
            PointIn(
                ids[i],
                a.name,
                "antigen",
                (float(xy[i, 0]), float(xy[i, 1])) if ok else None,
                clade_labels(a),
                parse_date(a.date),
                passage_class(a.passage, a.reassortant),
                bool((a.extra.get("T") or {}).get("R")),
                sequenced=bool(a.extra.get("A")),
                hide=hidden.get(i),
            )
        )
    for j, s in enumerate(chart.sera):
        k = chart.n_antigens + j
        ok = bool(np.isfinite(xy[k]).all())
        points.append(
            PointIn(
                ids[k],
                s.name,
                "serum",
                (float(xy[k, 0]), float(xy[k, 1])) if ok else None,
                passage_class=passage_class(s.passage, s.reassortant),
                serum_id=s.serum_id,
            )
        )

    size = config.frame_size(chart.info.get("V", ""), chart.info.get("A", ""))
    title_base = map_title(chart, cfg.title)
    provenance: dict[str, Any] = {"inputs": inputs}
    if decisions:
        provenance["decisions"] = decisions
    if store_refs:
        provenance["store_refs"] = store_refs
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
    return MapResult(cfg.folder, tuple(figures), tuple(flags), time.monotonic() - started)


# ---------------------------------------------------------------- the command


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
            )
            for r in rows
        )
    return out


def build(
    config: MapsConfig,
    *,
    store_root: Path | None,
    out_root: Path,
    vaccine_list: Sequence[Any],
    vaccine_defaults: dict[str, tuple[VaccineDisable, ...]],
    only: Sequence[str] = (),
    created: dt.datetime | None = None,
    log: Any = print,
) -> list[MapResult]:
    """Build every configured map (or just ``only``). Raises on the first failure."""
    created = created or dt.datetime.now(dt.UTC)
    store = Store(store_root) if store_root is not None else None
    wanted = [m for m in config.maps if not only or m.folder in only]
    missing = sorted(set(only) - {m.folder for m in config.maps})
    if missing:
        raise BuildError(f"no such map(s) in the config: {', '.join(missing)}")
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
        )
        results.append(result)
        flags = f" flags={len(result.flags)}" if result.flags else ""
        log(f"{cfg.folder:24s} {result.seconds:5.1f}s  {len(result.figures)} figures{flags}")
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
        )
    except (BuildError, ValueError, OSError) as exc:
        print(f"af.map.build: {exc}", file=sys.stderr)
        return 1
    print(f"{len(results)} map(s), {sum(len(r.figures) for r in results)} figures -> {args.out}")
    return 0


def _passage(word: str) -> Any:
    """A config passage word, checked against the classes af knows (never a silent 'any')."""
    if word not in ("any", "cell", "egg", "reassortant"):
        raise BuildError(f"unknown passage {word!r}: expected any, cell, egg or reassortant")
    return word


def _class(word: str) -> Any:
    if word not in ("cell", "egg", "reassortant"):
        raise BuildError(f"unknown passage class {word!r}: expected cell, egg or reassortant")
    return word


def relaxer_for(chart: Chart, projection_no: int = 0) -> tuple[Any, Any]:
    """`(relax, stress)` bound to this chart's optimiser problem.

    `relax(start, movable)` minimises from `start` holding everything outside `movable`, which is
    what a curation move needs: the moved points settle and the rest of the map follows. Built
    from the projection's own arrays, so a projection carrying its own forced column bases (a
    chain stage that lowered some) is measured against those, not the chart's.
    """
    from af.map.optimise import MapProblem, relax, stress

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
