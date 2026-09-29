"""Points a chain leaves out: named rules applied to every table as it enters the merge.

A round's map is often a selection of the lab's tables, not all of them (today's `0do` files
drop, e.g., every egg antigen and serum from an egg-free map). Here that selection is data in
the chain config, one rule per `[[select.remove]]` entry, each with its reason:

    [[select.remove]]
    what = "antigens"             # or "sera"
    passage = "egg"               # ANY passage step in eggs (af.map.vaccines.passage_class)
    reason = "egg-free map (Sarah, 29 Sep 2026)"

A rule has exactly one selector: `passage` (the class of the whole passage history, see below),
`name` (a substring of the virus name) or `designation` (a substring of the designation, which
adds the passage). `passage = "egg"` means ANY step was in eggs, so E3/E2SIAT1 is egg. ae's
`is_egg()` looks at the LAST step only and would call it cell; the two differ on egg-then-cell
passages (Sarah chose any-step, 29 Sep 2026).

Why applied per table, before the merge: a removed point then never reaches the merged titres,
the column bases or the map, which is what the round's per-merge removal does. A rule that
matches nothing in any of the chain's tables is an error (a typo would otherwise remove
nothing, silently); the matches are counted per step.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from af.chart.ace import read_chart
from af.chart.model import Antigen, Chart, Serum, Titres
from af.map.vaccines import passage_class

WHAT = ("antigens", "sera")
PASSAGE_CLASSES = ("egg", "cell")


class SelectError(ValueError):
    pass


@dataclass(frozen=True)
class RemoveRule:
    what: str  # "antigens" or "sera"
    reason: str
    passage: str | None = None  # "egg" (any step in eggs) or "cell"
    name: str | None = None  # substring of the name
    designation: str | None = None  # substring of the designation

    def __post_init__(self) -> None:
        if self.what not in WHAT:
            raise SelectError(f"select.remove: what must be antigens or sera, not {self.what!r}")
        if not self.reason.strip():
            raise SelectError("select.remove: every rule needs a reason")
        chosen = [k for k in ("passage", "name", "designation") if getattr(self, k)]
        if len(chosen) != 1:
            raise SelectError(
                f"select.remove ({self.reason}): exactly one of passage, name, designation"
                f" (got {chosen or 'none'})"
            )
        if self.passage is not None and self.passage not in PASSAGE_CLASSES:
            raise SelectError(f"select.remove: passage must be one of {PASSAGE_CLASSES}")

    def matches(self, point: Antigen | Serum) -> bool:
        if self.passage is not None:
            # the passage history alone: a reassortant grown in eggs is egg here, as in ae
            return passage_class(point.passage, "") == self.passage
        if self.name is not None:
            return self.name in point.name
        assert self.designation is not None
        return self.designation in point.designation()

    def to_json(self) -> dict[str, str]:
        return {k: v for k, v in vars(self).items() if v is not None}


@dataclass(frozen=True)
class Selection:
    """The `[select]` table of a chain config."""

    remove: list[RemoveRule] = field(default_factory=list)


def removed_by(rules: list[RemoveRule], chart: Chart) -> list[dict]:
    """Per rule, the designations it removes from `chart` (empty lists included)."""
    out = []
    for rule in rules:
        points = chart.antigens if rule.what == "antigens" else chart.sera
        out.append(
            {**rule.to_json(), "points": [p.designation() for p in points if rule.matches(p)]}
        )
    return out


def apply(rules: list[RemoveRule], chart: Chart) -> tuple[Chart, list[dict]]:
    """`chart` without the points any rule removes, and what each rule removed.

    Only a table (no projections) is accepted: removing points from a map would also need its
    layout and per-point settings remapped, and chains apply rules to tables only."""
    report = removed_by(rules, chart)
    if not any(r["points"] for r in report):
        return chart, report
    if chart.projections:
        raise SelectError("select.remove applies to tables, not to charts with projections")
    drop_ag = {i for i, a in enumerate(chart.antigens) if _removed(rules, "antigens", a)}
    drop_sr = {j for j, s in enumerate(chart.sera) if _removed(rules, "sera", s)}
    keep_ag = [i for i in range(chart.n_antigens) if i not in drop_ag]
    keep_sr = [j for j in range(chart.n_sera) if j not in drop_sr]
    new_ag = {old: new for new, old in enumerate(keep_ag)}
    new_sr = {old: new for new, old in enumerate(keep_sr)}
    titres = Titres(
        [[chart.titres.table[i][j] for j in keep_sr] for i in keep_ag],
        [
            {
                (new_ag[i], new_sr[j]): t
                for (i, j), t in layer.items()
                if i in new_ag and j in new_sr
            }
            for layer in chart.titres.layers
        ],
    )
    forced = (
        None
        if chart.forced_column_bases is None
        else np.asarray(chart.forced_column_bases)[keep_sr]
    )
    # per-point plot styles ("p") would no longer line up; a chain restyles anyway
    extra = {k: v for k, v in chart.extra.items() if k != "p"}
    kept = Chart(
        chart.info,
        [chart.antigens[i] for i in keep_ag],
        [chart.sera[j] for j in keep_sr],
        titres,
        forced,
        [],
        extra,
    )
    return kept, report


def _removed(rules: list[RemoveRule], what: str, point: Antigen | Serum) -> bool:
    return any(r.what == what and r.matches(point) for r in rules)


def check_rules_match(rules: list[RemoveRule], tables: list[Path]) -> None:
    """Every rule must remove something from at least one of the chain's tables (design rule 1).
    Checked before any step runs, so a typo fails in seconds, not after hours of maps."""
    unmatched = list(rules)
    for path in tables:
        if not unmatched:
            return
        chart = read_chart(path)
        unmatched = [rule for rule in unmatched if not removed_by([rule], chart)[0]["points"]]
    if unmatched:
        raise SelectError(
            "select.remove rules match nothing in the chain's tables: "
            + "; ".join(str(r.to_json()) for r in unmatched)
        )
