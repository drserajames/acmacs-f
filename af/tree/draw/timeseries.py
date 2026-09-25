"""Month x row matrix: the month each drawn leaf was collected, within the report window.

The window comes from the report config (start inclusive, end exclusive), never from today's
date (design rule 7). Dates are parsed; a leaf whose date lacks a month cannot be placed and is
counted, not guessed. A date known only to the year gets no bar either: the store keeps its
precision, and the date string (often YYYY-01-01) would put every such leaf in January. The
test is the precision flag, never the string: a real 1 January keeps its bar.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def parse_month(s: str | None) -> tuple[int, int] | None:
    """(year, month) from 'YYYY-MM' or 'YYYY-MM-DD'; None if there is no month."""
    if not s or len(s) < 7 or s[4] != "-":
        return None
    try:
        y, m = int(s[:4]), int(s[5:7])
    except ValueError:
        return None
    return (y, m) if 1 <= m <= 12 else None


def months(start: str, end: str) -> list[tuple[int, int]]:
    """Months in [start, end), e.g. months('2024-10', '2026-10') has 24 entries."""
    s, e = parse_month(start), parse_month(end)
    if s is None or e is None or s >= e:
        raise ValueError(f"bad time-series window {start!r}..{end!r}")
    out, (y, m) = [], s
    while (y, m) < e:
        out.append((y, m))
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


@dataclass
class TimeSeries:
    months: list[tuple[int, int]]
    column: np.ndarray  # per row: month column, -1 when outside the window or undated
    counts: dict[str, int]

    @property
    def in_window(self) -> np.ndarray:
        return self.column >= 0


def compute(
    dates_by_row: list[str | None],
    precision_by_row: list[str | None],
    start: str,
    end: str,
) -> TimeSeries:
    if len(precision_by_row) != len(dates_by_row):
        raise ValueError("one date precision per row is required")
    ms = months(start, end)
    index = {m: k for k, m in enumerate(ms)}
    col = np.full(len(dates_by_row), -1)
    counts = {
        "rows": len(dates_by_row),
        "in window": 0,
        "outside window": 0,
        "year-only date (no bar)": 0,
        "no month": 0,
    }
    for r, (d, precision) in enumerate(zip(dates_by_row, precision_by_row, strict=True)):
        ym = parse_month(d)
        if precision == "year":
            counts["year-only date (no bar)"] += 1
        elif ym is None:
            counts["no month"] += 1
        elif ym in index:
            col[r] = index[ym]
            counts["in window"] += 1
        else:
            counts["outside window"] += 1
    return TimeSeries(ms, col, counts)
