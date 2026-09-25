"""Dates written by labs, parsed without guessing.

A lab's convention (``date_order`` in the ``lab_conventions`` rule table: MDY, DMY or YMD)
decides slash dates. A date that is impossible in the lab's order but valid in exactly one
other order is accepted with a warning (CDC writes "E3/E1(17/02/23)" once among thousands of
M/D/Y dates). A date that is valid in no order is an error. A two-digit year is 20yy unless
that would be after ``not_after`` (the table's test date), then 19yy: the pivot comes from
the table, never from today's date (D-ingestion §2.4).
"""

from __future__ import annotations

import datetime as dt
import re

_SLASH = re.compile(r"(\d{1,4})[/.-](\d{1,2})[/.-](\d{2,4})")
ORDERS = {"MDY": (0, 1, 2), "DMY": (1, 0, 2), "YMD": (1, 2, 0)}  # (month, day, year) positions


class DateError(ValueError):
    pass


def parse(text: str, order: str, not_after: dt.date | None = None) -> tuple[str, str | None]:
    """Return (ISO date, warning or None)."""
    value = text.strip()
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}( 00:00:00)?", value):  # an Excel date cell
        return dt.date.fromisoformat(value[:10]).isoformat(), None
    m = _SLASH.fullmatch(value)
    if m is None:
        raise DateError(f"not a date: {text!r}")
    parts = m.groups()
    order = "YMD" if len(parts[0]) == 4 else order
    first = _build(parts, order, not_after)
    if first is not None:
        return first, None
    others = {o: d for o in ORDERS if o != order and (d := _build(parts, o, not_after)) is not None}
    if len(set(others.values())) == 1:
        (o, d), *_ = others.items()
        return d, f"date {text!r} is not {order}; read as {o} ({d})"
    raise DateError(
        f"date {text!r}: invalid as {order}, and {'ambiguous' if others else 'invalid'} otherwise"
    )


def _build(parts: tuple[str, ...], order: str, not_after: dt.date | None) -> str | None:
    mi, di, yi = ORDERS[order]
    if order == "YMD" and len(parts[0]) != 4:
        return None
    year = int(parts[yi])
    if len(parts[yi]) == 2:
        year += 2000 if not_after is None or 2000 + year <= not_after.year else 1900
    elif len(parts[yi]) != 4:
        return None
    try:
        return dt.date(year, int(parts[mi]), int(parts[di])).isoformat()
    except ValueError:
        return None
