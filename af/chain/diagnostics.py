"""Per-step diagnostics: measured when a step runs, judged when the review page is built.

`step_diagnostics` records numbers only (in `step.json`), with generous recording floors.
`flags` turns them into warnings using `Thresholds`. Keeping the judgement out of the step
means a threshold can be tuned without remaking any map, and there is one copy of each
fact: the measurement.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Any

import numpy as np

import af.chart.control as control
from af.chart.model import Chart
from af.chart.procrustes import procrustes
from af.chart.titre import Titre, TitreType

# what a step records (lower than any sensible flag threshold)
RECORD_MOVED = 0.5  # map units between consecutive steps
RECORD_SLACK = 0.5  # column-basis slack


@dataclass(frozen=True)
class Thresholds:
    moved_far: float = 1.0  # map units a common point may move between steps
    moved_far_count: int = 3  # a step is flagged when at least this many points moved far
    stress_jump: float = 0.15  # relative rise in stress per titre vs the previous step
    scratch_beats_incremental: float = 0.01  # scratch lower by this: incremental is stuck
    basin_rmsd: float = 0.5  # reported with a scratch win: the maps are in different basins
    column_basis_slack: float = 1.0  # a forced base this far above the table-only base


THRESHOLDS = Thresholds()


def flags(d: dict[str, Any], t: Thresholds = THRESHOLDS) -> list[str]:
    """Warnings for one step's recorded diagnostics."""
    out = []
    if new := d.get("newly_disconnected"):
        out.append(f"{len(new)} newly disconnected")
    far = [m for m in d.get("moved_far", []) if m["distance"] > t.moved_far]
    if len(far) >= t.moved_far_count:
        out.append(f"{len(far)} points moved > {t.moved_far}")
    rise = d.get("stress_per_term_change")
    if rise is not None and rise > t.stress_jump:
        out.append(f"stress per titre +{rise:.0%}")
    gap = d.get("incremental_minus_scratch_relative")
    if gap is not None and gap > t.scratch_beats_incremental:
        # Incremental below scratch is normal (it starts from a good map). The reverse means the
        # carried-forward layout is stuck in a worse basin.
        rmsd = d.get("incremental_vs_scratch_rmsd", 0.0)
        basin = f", different basin (RMSD {rmsd:.2f})" if rmsd > t.basin_rmsd else ""
        out.append(f"scratch beat incremental by {gap:.1%}{basin}")
    slack = [s for s in d.get("column_basis_slack", []) if s["slack"] >= t.column_basis_slack]
    if slack:
        out.append(f"{len(slack)} sera with column-basis slack ≥ {t.column_basis_slack}")
    if trapped := d.get("trapped"):  # after the core's trapped-point loop, any left is a problem
        out.append(f"{trapped} trapped")
    if new_control := d.get("control_flags"):  # already only the ones this step completes
        out.append(f"{len(new_control)} new control-chart flags")
    for g in d.get("group_moves", []):  # a group move changed the map: say so, one flag each
        if g["kept"]:
            # Sarah, 26 Sep 2026: groups may mix antigens and sera, but a mixed one is flagged as
            # such: a serum carried along by antigens whose titres to it are being fitted.
            mixed = (
                f"MIXED ({g['antigens']} antigens, {g['sera']} sera) "
                if g.get("sera") and g.get("antigens")
                else ""
            )
            out.append(
                f"{mixed}group of {g['size']} moved {g['distance']:.2f} "
                f"(stress {g['stress_after'] - g['stress_before']:+.2f})"
            )
    return out  # hemisphering is listed in the details but not flagged: most maps have some


def _cell_readings(chart: Chart, i: int, j: int) -> list[Titre]:
    """One cell's readings, one per layer (table) that has it, in table order."""
    return [layer[(i, j)] for layer in chart.titres.layers or [] if (i, j) in layer]


def _dropped_cells(chart: Chart, dropped: list[tuple[int, int, str]]) -> list[dict]:
    """Cells the layer merge turned into `*` although they have readings, and why."""
    return [
        {
            "cell": f"{_names(chart, [i])[0]} × {_names(chart, [chart.n_antigens + j])[0]}",
            "readings": [str(t) for t in _cell_readings(chart, i, j)],
            "outcome": outcome,
        }
        for i, j, outcome in dropped
    ]


# hicontrol (af.chart.control) on the series of readings of a cell across tables. Measurement
# only (Sarah, Q64, 26 Sep 2026: "just measure at present, keep sd_limit"): nothing here changes
# a titre. As in hicontrol's ref_panel_plot: centre = median, threshold = 1 log2 unit.
CONTROL_MIN_N = 5
# Trend 4 and alternation 10 compare successive readings exactly; continuous titres (off the
# two-fold series, e.g. HINT) almost never tie, so these fire by chance there (CONTROL-VS-SD.md).
EXACT_COMPARISON_RULES = (5, 6)  # indices into control.RULE_NAMES: "alt 10", "trend 4"
# Rules that flag single readings ("1 >=3", "2/3 >=2", "diff of 3") are new whenever their count
# rises: each new outlier is news. Run rules are new only when they first fire for the cell, or a
# run that grows by one point would be reported again at every step.
POINT_RULES = (0, 1, 7)


def _two_fold(t: Titre) -> bool:
    return t.type != TitreType.REGULAR or math.log2(t.value / 10).is_integer()


def _rules_firing(series: list[Titre], skip: tuple[int, ...]) -> dict[int, int]:
    dat = control.log_num([str(t) for t in series])
    finite = [v for v in dat if not math.isnan(v)]
    result = control.hi_rules(dat, statistics.median(finite) if finite else 0.0, 1.0)
    return {k: c for k, c in enumerate(result.counts) if c and k not in skip}


def control_flags(chart: Chart, min_n: int = CONTROL_MIN_N) -> list[dict]:
    """Control-chart rules this step's table newly completes: fired on the cell's series with
    the new table's reading, not without it. Each flag is reported once, where it appears."""
    layers = chart.titres.layers or []
    if len(layers) < 2:
        return []
    out = []
    for i, j in sorted(layers[-1]):
        series = _cell_readings(chart, i, j)
        if len(series) < min_n:
            continue
        continuous = not all(_two_fold(t) for t in series)
        skip = EXACT_COMPARISON_RULES if continuous else ()
        now = _rules_firing(series, skip)
        before = _rules_firing(series[:-1], skip) if len(series) - 1 >= min_n else {}
        new = [
            k
            for k, count in now.items()
            if k not in before or (k in POINT_RULES and count > before[k])
        ]
        for k in sorted(new):
            out.append(
                {
                    "rule": control.RULE_NAMES[k],
                    "cell": f"{_names(chart, [i])[0]} × {_names(chart, [chart.n_antigens + j])[0]}",
                    "readings": [str(t) for t in series],
                    "continuous": continuous,
                }
            )
    return out


def group_moves(chart: Chart, groups: list[dict] | None) -> list[dict]:
    """The trapped loop's group moves (`move_groups`), in reader's terms: which points, how far
    they moved together, and what it did to the stress. Kept moves changed the map."""
    return [
        {
            "size": len(g["members"]),
            "antigens": sum(1 for m in g["members"] if m < chart.n_antigens),
            "sera": sum(1 for m in g["members"] if m >= chart.n_antigens),
            "distance": math.hypot(*g["shift"]),
            "shift": g["shift"],
            "stress_before": g["stress_before"],
            "stress_after": g["stress_after"],
            "kept": g["kept"],
            "points": _names(chart, g["members"]),
        }
        for g in groups or []
    ]


def _names(chart: Chart, points: Any) -> list[str]:
    out = []
    for p in points:
        p = int(p)
        if p < chart.n_antigens:
            out.append("AG " + chart.antigens[p].designation())
        else:
            out.append("SR " + chart.sera[p - chart.n_antigens].designation())
    return out


def previous_in_current(previous: Chart, current: Chart) -> np.ndarray:
    """Index in `current` of each point of `previous` (the merge keeps previous indexes)."""
    return np.concatenate(
        [np.arange(previous.n_antigens), current.n_antigens + np.arange(previous.n_sera)]
    )


def n_stress_terms(arrays: dict) -> int:
    n_ag = arrays["titre_type"].shape[0]
    kinds = [1, 2, 4] if arrays.get("dodgy_is_regular") else [1, 2]
    use = np.isin(arrays["titre_type"], kinds)
    use &= ~arrays["disconnected"][:n_ag, None] & ~arrays["disconnected"][None, n_ag:]
    return int(use.sum())


def step_diagnostics(
    chosen: Chart,
    previous: Chart | None,
    incremental: Chart | None,
    scratch: Chart | None,
    arrays: dict | None,
    optimiser: Any,
    cfg: Any,
    report: Any = None,
) -> dict:
    d: dict[str, Any] = {"antigens": chosen.n_antigens, "sera": chosen.n_sera}
    best = chosen.best()
    if arrays is not None:
        _disconnection(d, chosen, previous, arrays)
    if previous is not None:
        _movement(d, chosen, previous, cfg)
    if incremental is not None and scratch is not None:
        si, ss = incremental.best().stress_value, scratch.best().stress_value
        d["incremental_minus_scratch_relative"] = (si - ss) / ss
        d["incremental_vs_scratch_rmsd"] = procrustes(
            incremental.best().layout, scratch.best().layout
        ).rmsd
    if report is not None:
        slack = {int(k): v for k, v in report.column_basis_slack.items() if v >= RECORD_SLACK}
        names = _names(chosen, [chosen.n_antigens + j for j in slack])
        d["column_basis_slack"] = [
            {"serum": n, "slack": slack[j]} for j, n in zip(slack, names, strict=True)
        ]
        d["sd_too_big_cells"] = report.outcomes.get("sd-too-big", 0)
        d["dropped_cells"] = _dropped_cells(chosen, report.dropped)
        if not report.cheating_assay:  # a skipped table adds no layer, so no new readings
            d["control_flags"] = control_flags(chosen)
    if optimiser is not None and arrays is not None and cfg.options.grid_test:
        gt = optimiser.grid_test(best.layout, arrays)
        d["grid_test"] = [{**g, "name": _names(chosen, [g["point"]])[0]} for g in gt]
        d["trapped"] = sum(1 for g in gt if g["diagnosis"] == "trapped")
        d["hemisphering"] = sum(1 for g in gt if g["diagnosis"] == "hemisphering")
    return d


def _disconnection(d: dict, chosen: Chart, previous: Chart | None, arrays: dict) -> None:
    terms = n_stress_terms(arrays)
    d["stress_terms"] = terms
    d["stress_per_term"] = chosen.best().stress_value / terms if terms else None
    disc = np.nonzero(arrays["disconnected"])[0]
    d["disconnected"] = _names(chosen, disc)
    if previous is not None:
        was = previous_in_current(previous, chosen)[list(previous.best().disconnected)]
        d["newly_disconnected"] = _names(chosen, [p for p in disc if p not in set(was)])


def _movement(d: dict, chosen: Chart, previous: Chart, cfg: Any) -> None:
    """Procrustes to the previous step over common points; points that moved; stress change."""
    best, prev_best = chosen.best(), previous.best()
    a = np.full_like(best.layout, np.nan)
    a[previous_in_current(previous, chosen)] = prev_best.layout
    fit = procrustes(a, best.layout)
    d["procrustes_to_previous_rmsd"] = fit.rmsd
    moved = np.nonzero(np.nan_to_num(fit.distances) > RECORD_MOVED)[0]
    order = moved[np.argsort(-fit.distances[moved])]
    d["moved_far"] = [
        {"point": n, "distance": float(fit.distances[p])}
        for n, p in zip(_names(chosen, order), order, strict=True)
    ]
    if d.get("stress_per_term") is not None:
        prev_arrays = previous.optimiser_arrays(
            cfg.options.minimum_column_basis, disconnect_threshold=cfg.options.disconnect_threshold
        )
        pt = n_stress_terms(prev_arrays)
        if pt and prev_best.stress_value:
            d["stress_per_term_change"] = d["stress_per_term"] / (prev_best.stress_value / pt) - 1
