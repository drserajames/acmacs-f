"""Per-step diagnostics: the numbers the review page uses to flag problem maps.

Each check returns plain JSON-able values, recorded in the step's `step.json`, so the
review page and any later tool read the same numbers. Thresholds are one dataclass so
they can come from config.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from af.chart.model import Chart
from af.chart.procrustes import procrustes


@dataclass(frozen=True)
class Thresholds:
    moved_far: float = 1.0  # map units a common point may move between steps before it is listed
    moved_far_count: int = 1  # a step is flagged when at least this many points moved far
    stress_jump: float = 0.15  # relative rise in stress per titre vs the previous step
    scratch_beats_incremental: float = (
        0.01  # scratch lower by this much: the incremental map is stuck
    )
    basin_rmsd: float = 0.5  # recorded; flagged only together with a scratch win
    column_basis_slack: float = 1.0  # a forced base this far above the table-only base


THRESHOLDS = Thresholds()


def _names(chart: Chart, points) -> list[str]:
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
    optimiser,
    cfg,
    report=None,
    thresholds: Thresholds = THRESHOLDS,
) -> dict:
    d: dict = {"thresholds": asdict(thresholds), "flags": []}
    best = chosen.best()
    d["antigens"], d["sera"] = chosen.n_antigens, chosen.n_sera
    if arrays is not None:
        terms = n_stress_terms(arrays)
        d["stress_terms"] = terms
        d["stress_per_term"] = best.stress_value / terms if terms else None
        disc = np.nonzero(arrays["disconnected"])[0]
        d["disconnected"] = _names(chosen, disc)
        if previous is not None:
            prev_disc = (
                set(previous_in_current(previous, chosen)[list(previous.best().disconnected)])
                if previous.best().disconnected
                else set()
            )
            new_disc = [p for p in disc if p not in prev_disc]
            d["newly_disconnected"] = _names(chosen, new_disc)
            if new_disc:
                d["flags"].append(f"{len(new_disc)} newly disconnected")

    if previous is not None:
        prev_best = previous.best()
        idx = previous_in_current(previous, chosen)
        a = np.full_like(best.layout, np.nan)
        a[idx] = prev_best.layout
        fit = procrustes(a, best.layout)
        d["procrustes_to_previous_rmsd"] = fit.rmsd
        far = np.nonzero(np.nan_to_num(fit.distances) > thresholds.moved_far)[0]
        order = far[np.argsort(-fit.distances[far])]
        d["moved_far"] = [
            {"point": n, "distance": float(fit.distances[p])}
            for n, p in zip(_names(chosen, order), order, strict=True)
        ]
        if len(far) >= thresholds.moved_far_count:
            d["flags"].append(f"{len(far)} points moved > {thresholds.moved_far}")
        if prev_best.stress_value and "stress_per_term" in d and d["stress_per_term"] is not None:
            prev_terms = previous.optimiser_arrays(
                cfg.options.minimum_column_basis,
                disconnect_threshold=cfg.options.disconnect_threshold,
            )
            pt = n_stress_terms(prev_terms)
            if pt:
                rise = d["stress_per_term"] / (prev_best.stress_value / pt) - 1.0
                d["stress_per_term_change"] = rise
                if rise > thresholds.stress_jump:
                    d["flags"].append(f"stress per titre +{rise:.0%}")

    if incremental is not None and scratch is not None:
        si, ss = incremental.best().stress_value, scratch.best().stress_value
        gap = (si - ss) / ss
        d["incremental_minus_scratch_relative"] = gap
        basin = procrustes(incremental.best().layout, scratch.best().layout)
        d["incremental_vs_scratch_rmsd"] = basin.rmsd
        # Incremental below scratch is the normal case (it starts from a good map). The
        # problem is the reverse: the carried-forward layout is stuck in a worse basin.
        if gap > thresholds.scratch_beats_incremental:
            extra = (
                f", different basin (RMSD {basin.rmsd:.2f})"
                if basin.rmsd > thresholds.basin_rmsd
                else ""
            )
            d["flags"].append(f"scratch beat incremental by {gap:.1%}{extra}")

    if report is not None:
        slack = {
            int(k): v
            for k, v in report.column_basis_slack.items()
            if v >= thresholds.column_basis_slack
        }
        d["column_basis_slack"] = [
            {"serum": n, "slack": slack[j]}
            for j, n in zip(
                slack, _names(chosen, [chosen.n_antigens + j for j in slack]), strict=True
            )
        ]
        if slack:
            d["flags"].append(
                f"{len(slack)} sera with column-basis slack ≥ {thresholds.column_basis_slack}"
            )
        sd = report.outcomes.get("sd-too-big", 0)
        d["sd_too_big_cells"] = sd

    if optimiser is not None and arrays is not None and cfg.options.grid_test:
        gt = optimiser.grid_test(best.layout, arrays)
        d["grid_test"] = [{**g, "name": _names(chosen, [g["point"]])[0]} for g in gt]
        trapped = sum(1 for g in gt if g["diagnosis"] == "trapped")
        hemi = sum(1 for g in gt if g["diagnosis"] == "hemisphering")
        d["trapped"], d["hemisphering"] = trapped, hemi
        if trapped:  # after the core's trapped-point loop, any left is a problem
            d["flags"].append(f"{trapped} trapped")
        # hemisphering is recorded (details) but not flagged: most maps have some
    return d
