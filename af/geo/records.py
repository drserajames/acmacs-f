"""Geo records: how many dots each location gets in each month.

Production rule (Sarah, 25 Sep 2026, and today's report): **one dot per antigen
preparation** (name, reassortant, annotations, passage) tested in any table, placed in the
month of its earliest reported collection date, at the location of its name. So one virus
titrated as an egg and a cell isolate is two dots.

Other dot rules Sarah asked for as options (one dot per virus with egg and cell kept
apart; merging genetically identical preparations; QC before any merge) come later, as
further values of ``DotRule``.

Where a location comes from is the caller's business (workstream 2's lookup), passed in as
a function, so this module holds no location data. A preparation with no collection date
or no resolvable location cannot be drawn; both are counted in the result, never dropped
silently.
"""

from __future__ import annotations

import calendar
import datetime
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from af.geo.colours import UNCOLOURED, DotStyle
from af.serology.query import Preparation


class DotRule(Enum):
    PREPARATION = "preparation"


@dataclass(frozen=True)
class Month:
    year: int
    month: int

    @classmethod
    def of(cls, date: datetime.date) -> Month:
        return cls(date.year, date.month)

    def __str__(self) -> str:
        return f"{self.year:04d}-{self.month:02d}"

    def next(self) -> Month:
        return Month(self.year + self.month // 12, self.month % 12 + 1)


def months(first: Month, last: Month) -> list[Month]:
    """Every month from ``first`` to ``last`` inclusive; empty months still get a map."""
    if (last.year, last.month) < (first.year, first.month):
        raise ValueError(f"month range ends before it starts: {first}..{last}")
    out = [first]
    while out[-1] != last:
        out.append(out[-1].next())
    return out


@dataclass
class GeoCounts:
    """Dots per (subtype, month, location), plus what could not be placed."""

    dots: Counter[tuple[str, Month, str, DotStyle]] = field(default_factory=Counter)
    months: list[Month] = field(default_factory=list)
    undated: Counter[str] = field(default_factory=Counter)  # subtype -> preparations
    no_location: Counter[tuple[str, str]] = field(default_factory=Counter)  # (subtype, name)


def geo_counts(
    preparations: Iterable[Preparation],
    first: Month,
    last: Month,
    location_of: Callable[[str], str | None],
    rule: DotRule = DotRule.PREPARATION,
    style_of: Callable[[Preparation], DotStyle] | None = None,
) -> GeoCounts:
    """Count dots for each month in ``first..last`` under ``rule``.

    ``style_of`` colours each dot (:func:`af.geo.colours.dot_styles`); without it every
    dot is drawn uncoloured.
    """
    if rule is not DotRule.PREPARATION:
        raise NotImplementedError(rule)
    window = months(first, last)
    wanted = set(window)
    result = GeoCounts(months=window)
    for prep in preparations:
        if prep.collection_date is None:
            result.undated[prep.subtype] += 1
            continue
        month = Month.of(prep.collection_date)
        if month not in wanted:
            continue
        location = location_of(prep.name)
        if location is None:
            result.no_location[prep.subtype, prep.name] += 1
            continue
        style = style_of(prep) if style_of is not None else UNCOLOURED
        result.dots[prep.subtype, month, location, style] += 1
    return result


def to_i7(counts: GeoCounts, subtype: str) -> dict[str, Any]:
    """One subtype in the I7 ``geo`` shape (ae's ``geo/<st>-records.json``).

    periods -> locations -> point groups with a count. Every month of the window is listed,
    empty ones included, so a month with no data is still a map. Until clade colours are
    joined, points are ``"transparent"`` with no ``clade``, and ``af.report.compare.geo``
    then compares locations only. A coloured point carries its legend label as ``clade``.
    What could not be placed is listed with the document.
    """
    by_month: dict[Month, dict[str, Counter[DotStyle]]] = {m: {} for m in counts.months}
    for (s, month, location, style), n in counts.dots.items():
        if s == subtype:
            by_month[month].setdefault(location, Counter())[style] += n
    return {
        "subtype": subtype,
        "periods": [
            {
                "period": str(month),
                "title": f"{calendar.month_name[month.month]} {month.year}",
                "locations": [
                    {"name": name, "points": _points(styles)}
                    for name, styles in sorted(locations.items())
                ],
            }
            for month, locations in by_month.items()
        ],
        "undated": counts.undated[subtype],
        "no_location": sorted(name for s, name in counts.no_location if s == subtype),
    }


def _points(styles: Counter[DotStyle]) -> list[dict[str, Any]]:
    """Point groups at one location, the largest first (drawn at the centre)."""
    points = []
    for style, n in sorted(styles.items(), key=lambda kv: (-kv[1], kv[0].label)):
        point: dict[str, Any] = {"color": style.colour or "transparent", "count": n}
        if style.label:
            point["clade"] = style.label
        points.append(point)
    return points
