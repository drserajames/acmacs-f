"""Stat counts: antigens and sera by subtype, lab, period and continent.

Rules (Sarah, 25 Sep 2026):

- An **antigen** is a preparation (see :mod:`af.serology.query`), counted in the month of
  its earliest collection date, for the **lab of the earliest table** that titrated it.
  (Today's tool uses whichever lab sorts first in hidb's table order, which is arbitrary.)
- A **serum** is counted in the month of the **isolation date of its strain**, for the
  lab of its earliest table. ``sera`` counts each serum name once per subtype, and
  ``sera_unique`` counts every serum, as today's stat page does. (Today's tool dates a
  serum by a homologous antigen hidb almost never records, so its serum counts are empty.)

Every count is added across {subtype, "all"} x {lab, "all"} x {month, year, "all"} x
{continent, "all"}, so any total can be read directly. For the subtypes named in
``split_by_lineage`` a second row per lineage (``<subtype>/<lineage>``, or
``<subtype>/UNKNOWN``) is added too, without an extra "all" subtype roll-up.

Periods are only those inside the window. Items with no date are counted apart
(``undated``), and a location with no continent counts under ``UNKNOWN``, reported.
"""

from __future__ import annotations

import datetime
from collections import Counter
from collections.abc import Callable, Collection, Iterable
from dataclasses import dataclass, field

from af.geo.records import Month, months
from af.serology.query import Preparation, SerumRecord

ALL = "all"
UNKNOWN = "UNKNOWN"

Key = tuple[str, str, str, str]  # (subtype, lab, period, continent)


@dataclass
class StatCounts:
    antigens: Counter[Key] = field(default_factory=Counter)
    sera: Counter[Key] = field(default_factory=Counter)
    sera_unique: Counter[Key] = field(default_factory=Counter)
    undated: Counter[str] = field(default_factory=Counter)  # "antigens"/"sera" -> n
    unknown_continent: Counter[str] = field(default_factory=Counter)  # location -> n


@dataclass(frozen=True)
class _Item:
    subtype: str
    lineage: str
    lab: str
    date: datetime.date | None
    name: str


def stat_counts(
    preparations: Iterable[Preparation],
    sera: Iterable[SerumRecord],
    first: Month,
    last: Month,
    location_of: Callable[[str], str | None],
    continent_of: Callable[[str], str | None],
    split_by_lineage: Collection[str] = (),
) -> StatCounts:
    """Count antigens and sera in the months ``first..last``."""
    window = set(months(first, last))
    result = StatCounts()

    def place(item: _Item, kind: str) -> tuple[Month, str] | None:
        """The (month, continent) an item counts in, or None if outside the window."""
        if item.date is None:
            result.undated[kind] += 1
            return None
        month = Month.of(item.date)
        if month not in window:
            return None
        location = location_of(item.name)
        continent = continent_of(location) if location else None
        if continent is None:
            result.unknown_continent[location or item.name] += 1
            continent = UNKNOWN
        return month, continent

    for p in preparations:
        item = _Item(p.subtype, p.lineage, p.first_lab, p.collection_date, p.name)
        if where := place(item, "antigens"):
            _bucket(result.antigens, item, *where, split_by_lineage)
    seen_names: set[tuple[str, str]] = set()
    for s in sera:
        item = _Item(s.subtype, s.lineage, s.first_lab, s.strain_collection_date, s.name)
        if where := place(item, "sera"):
            _bucket(result.sera_unique, item, *where, split_by_lineage)
            if (s.subtype, s.name) not in seen_names:
                seen_names.add((s.subtype, s.name))
                _bucket(result.sera, item, *where, split_by_lineage)
    return result


def _bucket(
    target: Counter[Key], item: _Item, month: Month, continent: str, split: Collection[str]
) -> None:
    subtypes = [item.subtype, ALL]
    lineage_row = f"{item.subtype}/{item.lineage or UNKNOWN}" if item.subtype in split else None
    for subtype in subtypes + ([lineage_row] if lineage_row else []):
        for lab in (item.lab, ALL):
            for period in (str(month), f"{month.year:04d}", ALL):
                for ct in (continent, ALL):
                    target[subtype, lab, period, ct] += 1
