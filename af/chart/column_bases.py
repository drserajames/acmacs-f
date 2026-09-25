"""Forced column bases set by hand, per stage, as named and counted data.

Why: a serum's column base is the log2 of its highest titre, and one discarded `>` titre can set
it higher than every titre the map keeps, so the serum sits well away from its antigens. Today a
folder script lowers such bases between two finishing stages, which changes the stress every
later stage is measured with. af keeps the bases on the stage's `Projection`
(`forced_column_bases` wins over the chart's, as in ae), so each stage is optimised and measured
with its own bases. This module is the override that sets them.

Sera are selected by designation (design rule 2) and each must match exactly one serum, so a
renamed or duplicated serum is an error rather than a silent partial change.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

from af.chart.model import Chart, Projection


class ColumnBaseOverrideError(ValueError):
    pass


@dataclass(frozen=True)
class ColumnBaseOverride:
    """Force the column base of the named sera to ``value`` (log2 units: 8 is a titre of 2560)."""

    name: str
    reason: str
    decided: str
    sera: tuple[str, ...]  # Serum.designation(), each matching exactly one serum
    value: float


@dataclass(frozen=True)
class ColumnBaseChange:
    serum: int
    designation: str
    before: float
    after: float


@dataclass(frozen=True)
class ColumnBaseResult:
    projection: Projection
    changes: tuple[ColumnBaseChange, ...]

    def report(self, rule: ColumnBaseOverride) -> dict[str, object]:
        return {
            "override": rule.name,
            "reason": rule.reason,
            "decided": rule.decided,
            "sera": len(self.changes),
            "changed": sum(1 for c in self.changes if c.before != c.after),
            "changes": [
                {"designation": c.designation, "before": c.before, "after": c.after}
                for c in self.changes
            ],
        }


def apply_column_base_override(
    chart: Chart, projection: Projection, rule: ColumnBaseOverride
) -> ColumnBaseResult:
    """A copy of ``projection`` whose forced bases are its effective bases with ``rule`` applied.

    The copy has no stress: the old value was computed with the old bases. Applying the same
    rule twice changes nothing and reports before == after.
    """
    if not rule.sera:
        raise ColumnBaseOverrideError(f"{rule.name}: no sera named")
    bases = chart.column_bases(projection.minimum_column_basis, projection)
    designations = [serum.designation() for serum in chart.sera]
    problems = []
    rows = []
    for wanted in rule.sera:
        matches = [j for j, d in enumerate(designations) if d == wanted]
        if len(matches) != 1:
            problems.append(f"{wanted!r} matches {len(matches)} sera")
        else:
            rows.append(matches[0])
    if problems:
        raise ColumnBaseOverrideError(f"{rule.name}: " + "; ".join(problems))
    if len(set(rows)) != len(rows):
        raise ColumnBaseOverrideError(f"{rule.name}: a serum is named twice")
    new = bases.copy()
    changes = []
    for j in rows:
        changes.append(ColumnBaseChange(j, designations[j], float(bases[j]), float(rule.value)))
        new[j] = rule.value
    return ColumnBaseResult(
        dataclasses.replace(projection, forced_column_bases=new, stress=None), tuple(changes)
    )
