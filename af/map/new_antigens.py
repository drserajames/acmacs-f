"""Outlines that mark the antigens new since the previous reports (the interim reports).

Sarah, for M2 (23 Oct 2026): antigens newly titrated since the previous VCM have a thin black
outline, thicker than the default; those titrated since the previous interim report (or the VCM,
for the first interim) a thick bold one, as ae's interim (TC) reports drew them. ae's widths are
1, 3 and 6 points on the 800-point page (ae.report.chart_modifier.style_new_compared_to), black,
the new antigens raised over the others.

A round configures the tiers (``[[new_antigen_outlines]]``): each a cut-off, a width, and per
map the previous report's map as af drew it (a store chain version, ``dataset@version``). An
antigen is new for a tier when that map lacks it (Sarah, Q264 b, 24 Oct 2026: what is new to the
reader, as ae's rule, against af's own chain version); one new for several tiers takes the latest
cut-off's (the thick one). How many were first titrated after the cut-off is recorded beside: a
table dated before the cut-off can arrive after it. A tier that matches nothing is reported, not
an error: a quiet month is possible. Vaccine marks keep their own outline, as in ae.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from af.chart.model import Chart
from af.map.render import PointStyle
from af.map.style import PointIn


@dataclass(frozen=True)
class NewAntigenTier:
    """Antigens the previous report's map lacks get an outline ``outline_width`` points wide;
    ``since`` is that report's cut-off, which orders the tiers and counts first titrations."""

    name: str
    since: dt.date
    outline_width: float
    reason: str
    # per map folder: the previous report's map, the store chain version it was drawn from
    previous_maps: Mapping[str, str] = field(default_factory=dict)
    colour: str = "black"

    def __post_init__(self) -> None:
        if not self.outline_width > 0:
            raise ValueError(f"new antigen outline {self.name!r}: width must be positive")
        for folder, ref in self.previous_maps.items():
            dataset, sep, version = ref.partition("@")
            if not sep or not dataset or not version:
                raise ValueError(
                    f"new antigen outline {self.name!r}: {folder} = {ref!r} is not dataset@version"
                )

    def previous_map(self, folder: str) -> tuple[str, str]:
        """The previous report's chain dataset and version for ``folder``; a map the tier does
        not name is an error (design rule 4: no silent default)."""
        if folder not in self.previous_maps:
            raise ValueError(
                f"new antigen outline {self.name!r} names no previous map for {folder}"
            )
        dataset, _, version = self.previous_maps[folder].partition("@")
        return dataset, version


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


def _key(a: Any) -> tuple:
    return (a.name, a.reassortant, a.annotations, a.passage)


def new_antigen_styles(
    chart: Chart,
    points: Sequence[PointIn],
    tiers: Sequence[NewAntigenTier],
    previous: Sequence[Chart],
    marks: frozenset[str] = frozenset(),
) -> tuple[dict[str, PointStyle], list[dict[str, Any]]]:
    """The outline of each new antigen point (by id), and per tier what it matched, for the
    figure's record. ``previous[k]`` is tier k's previous report map (its chart). ``marks``:
    point ids drawn as vaccine marks, which keep their outline."""
    if len(previous) != len(tiers):
        raise ValueError(f"{len(previous)} previous maps for {len(tiers)} tiers")
    new_for = []
    for prev in previous:
        theirs = {_key(a) for a in prev.antigens}
        new_for.append({i for i, a in enumerate(chart.antigens) if _key(a) not in theirs})
    order = sorted(range(len(tiers)), key=lambda k: (tiers[k].since, k))  # the latest wins
    tier_of: dict[int, int] = {}
    for i in range(chart.n_antigens):
        for k in order:
            if i in new_for[k]:
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
    first = first_titrated(chart)
    record = []
    for k, t in enumerate(tiers):
        record.append(
            {
                "name": t.name,
                "since": t.since.isoformat(),
                "outline_width": t.outline_width,
                "reason": t.reason,
                "new_since_previous_map": len(new_for[k]),
                "outlined": drawn[k],
                "first_titrated_after": sum(1 for w in first if w is not None and w > t.since),
            }
        )
    return styles, record
