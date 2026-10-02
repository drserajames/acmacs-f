"""Named column-basis adjustments: changes a round made to a merged chart's column bases before
optimising, kept as data with their reasons and counted (design rule 9).

Sarah, 2 Oct 2026: a round's adjustment-stage changes become named, per-map options applied
before the map is optimised ("Yes, as named config"). Two rules:

lower          the column bases of `sera` (every serum when none are named) lowered by `by` log2:
               the chart's forced bases when it has them (a merge derives them), the titre-implied
               ones otherwise. A named serum that matches no serum, or several, is an error: a
               silent no-op would publish a map taken to be adjusted that is not.
remove_forced  the chart's forced column bases removed, so every serum's basis comes from its
               titres in the merged table.

Applied to a merge_all map only: in a chain each step's merge would meet the adjustment again.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from af.chart.model import Chart

RULES = ("lower", "remove_forced")


class AdjustError(ValueError):
    pass


@dataclass(frozen=True)
class ColumnBasisAdjustment:
    """One `[[adjust_column_bases]]` row of a chain file."""

    rule: str
    reason: str
    decided: str  # who decided and when, e.g. "Sarah, 2 Oct 2026"
    by: float = 0.0  # log2 units, for "lower"
    sera: list[str] = field(default_factory=list)  # designations, for "lower"; empty = all

    def __post_init__(self) -> None:
        if self.rule not in RULES:
            raise AdjustError(f"column-basis adjustment rule must be one of {RULES}: {self.rule!r}")
        if not self.reason.strip() or not self.decided.strip():
            raise AdjustError(f"column-basis adjustment {self.rule!r} needs a reason and decided")
        if self.rule == "lower" and not self.by > 0:
            raise AdjustError(f"lower needs by > 0, not {self.by}")
        if self.rule == "remove_forced" and (self.by or self.sera):
            raise AdjustError("remove_forced takes no by or sera")

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {"rule": self.rule, "reason": self.reason, "decided": self.decided}
        if self.rule == "lower":
            out["by"] = self.by
            if self.sera:
                out["sera"] = list(self.sera)
        return out


def apply(rules: list[ColumnBasisAdjustment], chart: Chart) -> tuple[Chart, list[dict]]:
    """`chart` with each rule applied in order, and per rule what it changed: the count, the
    largest change and every serum changed (designation, before, after), so a reader can check
    one serum. Bases are compared as the optimiser sees them with no minimum column basis."""
    reports = []
    for rule in rules:
        before = np.asarray(chart.column_bases("none"), dtype=float)
        if rule.rule == "lower":
            which = _select(chart, rule.sera)
            forced = before.copy()
            forced[which] -= rule.by
            chart = replace(chart, forced_column_bases=forced)
        else:
            chart = replace(chart, forced_column_bases=None)
        after = np.asarray(chart.column_bases("none"), dtype=float)
        changed = np.nonzero(np.abs(after - before) > 1e-9)[0]
        reports.append(
            {
                **rule.to_json(),
                "sera_changed": len(changed),
                "max_change": float(np.abs(after - before)[changed].max()) if len(changed) else 0.0,
                "changed": [
                    {
                        "serum": chart.sera[j].designation(),
                        "before": float(before[j]),
                        "after": float(after[j]),
                    }
                    for j in changed
                ],
            }
        )
    return chart, reports


def _select(chart: Chart, names: list[str]) -> list[int]:
    if not names:
        return list(range(chart.n_sera))
    out = []
    for name in names:
        hits = [j for j, s in enumerate(chart.sera) if s.designation() == name]
        if len(hits) != 1:
            raise AdjustError(
                f"column-basis adjustment: {name!r} matches {len(hits)} sera, expected exactly one"
            )
        out.append(hits[0])
    return out
