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

    dots: Counter[tuple[str, Month, str]] = field(default_factory=Counter)
    months: list[Month] = field(default_factory=list)
    undated: Counter[str] = field(default_factory=Counter)  # subtype -> preparations
    no_location: Counter[tuple[str, str]] = field(default_factory=Counter)  # (subtype, name)


def geo_counts(
    preparations: Iterable[Preparation],
    first: Month,
    last: Month,
    location_of: Callable[[str], str | None],
    rule: DotRule = DotRule.PREPARATION,
) -> GeoCounts:
    """Count dots for each month in ``first..last`` under ``rule``."""
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
        result.dots[prep.subtype, month, location] += 1
    return result


UNCOLOURED = "unassigned"


def to_i7(counts: GeoCounts, subtype: str) -> dict[str, Any]:
    """One subtype in the I7 ``geo`` shape (ae's ``geo/<st>-records.json``).

    periods -> locations -> point groups with a count. Every month of the window is listed,
    empty ones included, so a month with no data is still a map. Until clade colours are
    joined (workstream 4) each location has one point group coloured ``"unassigned"``, and
    ``af.report.compare.geo`` then compares locations only. What could not be placed is
    listed with the document.
    """
    by_month: dict[Month, dict[str, int]] = {month: {} for month in counts.months}
    for (s, month, location), n in counts.dots.items():
        if s == subtype:
            by_month[month][location] = n
    return {
        "subtype": subtype,
        "periods": [
            {
                "period": str(month),
                "title": f"{calendar.month_name[month.month]} {month.year}",
                "locations": [
                    {"name": name, "points": [{"color": UNCOLOURED, "count": n}]}
                    for name, n in sorted(locations.items())
                ],
            }
            for month, locations in by_month.items()
        ],
        "undated": counts.undated[subtype],
        "no_location": sorted(name for s, name in counts.no_location if s == subtype),
    }
