"""Outlines that mark the antigens newly titrated since a report's cut-offs (the interim reports).

Sarah, for M2 (23 Oct 2026): antigens newly titrated since the previous VCM have a thin black
outline, thicker than the default; those titrated since the previous interim report (or the VCM,
for the first interim) a thick bold one, as ae's interim (TC) reports drew them. ae's widths are
1, 3 and 6 points on the 800-point page (ae.report.chart_modifier.style_new_compared_to), black,
the new antigens raised over the others.

A round configures the tiers (``[[new_antigen_outlines]]``): each a cut-off date and a width. An
antigen is new for a tier when the first table it was titrated in is dated after the tier's
cut-off; one new for several tiers takes the latest cut-off's (the thick one). ae marked an
antigen new when the previous round's chart lacked it; that count is recorded beside, for a
tier that names the round to compare with. A tier that matches nothing is reported, not an
error: a quiet month is possible. Vaccine marks keep their own outline, as in ae.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from af.chart.model import Chart
from af.map.render import PointStyle
from af.map.style import PointIn


@dataclass(frozen=True)
class NewAntigenTier:
    """Antigens first titrated after ``since`` get an outline ``outline_width`` points wide."""

    name: str
    since: dt.date
    outline_width: float
    reason: str
    colour: str = "black"
    # an ae round folder whose <map>/styled.ace is the previous report's chart, for counting ae's
    # rule (antigens that chart lacks) beside this one; None: not counted
    compare_round: Path | None = None

    def __post_init__(self) -> None:
        if not self.outline_width > 0:
            raise ValueError(f"new antigen outline {self.name!r}: width must be positive")


def first_titrated(chart: Chart) -> list[dt.date | None]:
    """Each antigen's first titration: the earliest date of the source tables in which it has a
    titre (a merged chart's layers and their sources), or the chart's own date for a single
    table. None where no table is dated."""
    from af.map.build import _compact

    if not chart.titres.layers:
        when = _compact(chart.info)
        return [when] * chart.n_antigens
    sources = chart.info.get("S") or []
    first: list[dt.date | None] = [None] * chart.n_antigens
    for k, layer in enumerate(chart.titres.layers):
        when = _compact(sources[k]) if k < len(sources) else None
        if when is None:
            continue
        for (ag, _sr), titre in layer.items():
            before = first[ag]
            if not titre.is_missing and (before is None or when < before):
                first[ag] = when
    return first


def _not_in_previous(chart: Chart, folder: str, round_dir: Path) -> set[int] | None:
    """Antigens the previous round's chart lacks (ae's rule), or None when it has no chart."""
    from af.chart.ace import read_chart

    path = round_dir / folder / "styled.ace"
    if not path.is_file():
        return None

    def key(a: Any) -> tuple:
        return (a.name, a.reassortant, a.annotations, a.passage)

    theirs = {key(a) for a in read_chart(path).antigens}
    return {i for i, a in enumerate(chart.antigens) if key(a) not in theirs}


def new_antigen_styles(
    chart: Chart,
    points: Sequence[PointIn],
    tiers: Sequence[NewAntigenTier],
    folder: str,
    marks: frozenset[str] = frozenset(),
) -> tuple[dict[str, PointStyle], list[dict[str, Any]]]:
    """The outline of each new antigen point (by id), and per tier what it matched, for the
    figure's record. ``marks``: point ids drawn as vaccine marks, which keep their outline."""
    first = first_titrated(chart)
    order = sorted(range(len(tiers)), key=lambda k: (tiers[k].since, k))  # the latest wins
    tier_of: dict[int, int] = {}
    for i, when in enumerate(first):
        for k in order:
            if when is not None and when > tiers[k].since:
                tier_of[i] = k
    styles: dict[str, PointStyle] = {}
    drawn = [0] * len(tiers)
    for p in points:
        if p.kind != "antigen" or not p.id[2:].isdigit() or p.id in marks or p.xy is None:
            continue
        tier = tier_of.get(int(p.id[2:]))
        if tier is None:
            continue
        styles[p.id] = PointStyle(
            outline=tiers[tier].colour, outline_width=tiers[tier].outline_width, raise_rank=tier
        )
        drawn[tier] += 1
    record = []
    for k, t in enumerate(tiers):
        new = [i for i, when in enumerate(first) if when is not None and when > t.since]
        ae_rule = _not_in_previous(chart, folder, t.compare_round) if t.compare_round else None
        record.append(
            {
                "name": t.name,
                "since": t.since.isoformat(),
                "outline_width": t.outline_width,
                "reason": t.reason,
                "first_titrated_after": len(new),
                "outlined": drawn[k],
                "ae_rule_not_in_previous_chart": None if ae_rule is None else len(ae_rule),
                "compare_round": str(t.compare_round) if t.compare_round else None,
            }
        )
    return styles, record
