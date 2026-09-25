"""Collection dates that keep their precision.

A submitted collection date may be a day, a month or only a year. The old pipeline
turned all three into a day — a year-only date became 1 January — and from then on
nothing could tell a virus genuinely collected on 1 January from one collected at an
unknown point in that year. That mattered: those false 1-January dates cluster on the
tree's time axis, and one lab's habit of submitting year-only dates produced a visible
January spike that had to be hidden by hand, year after year.

So a date here is a value *and* its precision, and the pair travels together. A date is
an interval: ``2024`` means 1 January to 31 December 2024, ``2024-03`` means the whole of
March. Comparisons are interval comparisons, and :func:`on_or_after` requires the *whole*
interval to satisfy a floor (Sarah, 25 Sep 2026), so a year-only 2018 does not slip past
a March 2018 floor on the strength of a 1 January that was never stated.

No function here looks at today's date: a rebuild must give the same answer next year
(design rule 7). Two-digit years are not accepted at all — the old pipeline's pivot read
them against the current year, so the same input changed meaning as time passed.
"""

from __future__ import annotations

import datetime
import re
from dataclasses import dataclass
from enum import Enum


class Precision(Enum):
    """How much of a date was actually stated."""

    DAY = "day"
    MONTH = "month"
    YEAR = "year"


class DateProblem(ValueError):
    """A collection date that cannot be read as a day, a month or a year."""


_DAY = re.compile(r"(\d{4})-(\d{2})-(\d{2})")
_MONTH = re.compile(r"(\d{4})-(\d{2})")
_YEAR = re.compile(r"(\d{4})")


@dataclass(frozen=True, order=True)
class CollectionDate:
    """A collection date and the precision it was given to.

    ``first`` and ``last`` bound what the date can mean. For a full day they are equal.
    Ordering is by ``first`` then ``last``, which is a stable sort key, not a claim that
    one virus was collected before another.
    """

    first: datetime.date
    last: datetime.date
    precision: Precision

    def __str__(self) -> str:
        if self.precision is Precision.DAY:
            return self.first.isoformat()
        if self.precision is Precision.MONTH:
            return f"{self.first.year:04d}-{self.first.month:02d}"
        return f"{self.first.year:04d}"

    @property
    def year(self) -> int:
        return self.first.year

    def on_or_after(self, floor: CollectionDate | str) -> bool:
        """Is the whole interval on or after ``floor``?

        A year-only date fails a floor inside that year, because the record does not
        say the virus was collected after it. Being explicit beats assuming 1 January.
        """
        other = parse(floor) if isinstance(floor, str) else floor
        return self.first >= other.first

    def overlaps(self, other: CollectionDate) -> bool:
        return self.first <= other.last and other.first <= self.last


def parse(text: str) -> CollectionDate:
    """Read ``YYYY-MM-DD``, ``YYYY-MM`` or ``YYYY``. Anything else raises.

    Raising rather than returning a default is deliberate: a date af cannot read is a
    record someone should look at, and a silent 1 January hides it forever.
    """
    value = text.strip()
    if (match := _DAY.fullmatch(value)) is not None:
        year, month, day = (int(part) for part in match.groups())
        try:
            exact = datetime.date(year, month, day)
        except ValueError as err:  # 31 February, month 13, …
            raise DateProblem(f"collection date {text!r}: {err}") from err
        return CollectionDate(exact, exact, Precision.DAY)
    if (match := _MONTH.fullmatch(value)) is not None:
        year, month = (int(part) for part in match.groups())
        if not 1 <= month <= 12:
            raise DateProblem(f"collection date {text!r}: month {month} is not 1-12")
        return CollectionDate(
            datetime.date(year, month, 1),
            datetime.date(year, month, _days_in_month(year, month)),
            Precision.MONTH,
        )
    if (match := _YEAR.fullmatch(value)) is not None:
        year = int(match.group(1))
        return CollectionDate(
            datetime.date(year, 1, 1), datetime.date(year, 12, 31), Precision.YEAR
        )
    raise DateProblem(
        f"collection date {text!r}: expected YYYY-MM-DD, YYYY-MM or YYYY"
        " (a two-digit year is refused: its meaning would depend on when af ran)"
    )


def _days_in_month(year: int, month: int) -> int:
    if month == 12:
        return 31
    return (datetime.date(year, month + 1, 1) - datetime.timedelta(days=1)).day
