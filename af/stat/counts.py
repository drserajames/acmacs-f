"""Stat counts: antigens and sera by subtype, lab, period and continent.

Rules (Sarah, 25 Sep 2026):

- An **antigen** is a preparation (see :mod:`af.serology.query`), counted in the month of
  its earliest collection date, for the **lab of the earliest table** that titrated it.
  (Today's tool uses whichever lab sorts first in hidb's table order, which is arbitrary.)
- Sera are counted two ways, from the tables where a serum has at least one reading:
  - ``sera_new``: each serum once, in the month of its **first titration**, for that
    table's lab (the earliest table wins, as for antigens);
  - ``sera_used``: each serum in every month it was **used in a titration**, for each lab
    that used it. Within a year or the whole window a serum counts once, however many
    months it was used in, so these cells are numbers of distinct sera, not of uses.
  (Today's tool dates a serum by a homologous antigen hidb almost never records, so its
  serum counts are empty.)

Every count is added across {subtype, "all"} x {lab, "all"} x {month, year, "all"} x
{continent, "all"}, so any total can be read directly. For the subtypes named in
``split_by_lineage`` a second row per lineage (``<subtype>/<lineage>``, or
``<subtype>/UNKNOWN``) is added too, without an extra "all" subtype roll-up. A serum's
continent is that of its strain's location.

Periods are only those inside the window. Antigens with no collection date are counted
apart (``undated``), and a location with no continent counts under ``UNKNOWN``, reported.
"""

from __future__ import annotations

import datetime
from collections import Counter, defaultdict
from collections.abc import Callable, Collection, Iterable, Iterator
from dataclasses import dataclass, field

from af.geo.records import Month, months
from af.serology.query import Preparation, SerumUse

ALL = "all"
UNKNOWN = "UNKNOWN"

Key = tuple[str, str, str, str]  # (subtype, lab, period, continent)


@dataclass
class StatCounts:
    antigens: Counter[Key] = field(default_factory=Counter)
    sera_new: Counter[Key] = field(default_factory=Counter)
    sera_used: Counter[Key] = field(default_factory=Counter)
    undated: Counter[str] = field(default_factory=Counter)  # "antigens" -> n
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
    serum_uses: Iterable[SerumUse],
    first: Month,
    last: Month,
    location_of: Callable[[str], str | None],
    continent_of: Callable[[str], str | None],
    split_by_lineage: Collection[str] = (),
) -> StatCounts:
    """Count antigens, new sera and sera used in the months ``first..last``."""
    window = set(months(first, last))
    result = StatCounts()

    def place(item: _Item) -> tuple[Month, str] | None:
        """The (month, continent) an item counts in, or None if outside the window."""
        if item.date is None:
            result.undated["antigens"] += 1
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
        if where := place(item):
            result.antigens.update(_keys(item, *where, split_by_lineage))

    uses = list(serum_uses)
    for use in _first_uses(uses):
        item = _Item(use.subtype, use.lineage, use.lab, use.table_date, use.name)
        if where := place(item):
            result.sera_new.update(_keys(item, *where, split_by_lineage))

    distinct: defaultdict[Key, set[tuple[str, str]]] = defaultdict(set)
    for use in uses:
        item = _Item(use.subtype, use.lineage, use.lab, use.table_date, use.name)
        if where := place(item):
            for key in _keys(item, *where, split_by_lineage):
                distinct[key].add((use.subtype, use.serum_key))
    result.sera_used = Counter({key: len(sera) for key, sera in distinct.items()})
    return result


def _first_uses(uses: Iterable[SerumUse]) -> list[SerumUse]:
    """Each serum's earliest use, ordered by (date, date_suffix, lab, table_id)."""
    first: dict[tuple[str, str], SerumUse] = {}
    for use in uses:
        key = (use.subtype, use.serum_key)
        if key not in first or _order(use) < _order(first[key]):
            first[key] = use
    return list(first.values())


def _order(use: SerumUse) -> tuple[datetime.date, int, str, str]:
    return (use.table_date, use.date_suffix, use.lab, use.table_id)


def _keys(item: _Item, month: Month, continent: str, split: Collection[str]) -> Iterator[Key]:
    """Every cell an item counts in: its own values and each "all" roll-up."""
    subtypes = [item.subtype, ALL]
    if item.subtype in split:
        subtypes.append(f"{item.subtype}/{item.lineage or UNKNOWN}")
    for subtype in subtypes:
        for lab in (item.lab, ALL):
            for period in (str(month), f"{month.year:04d}", ALL):
                for ct in (continent, ALL):
                    yield subtype, lab, period, ct
