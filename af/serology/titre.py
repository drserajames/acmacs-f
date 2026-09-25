"""One titre reading: its kind, value and log.

Readings arrive from the tables store already normalised (``"40"``, ``"<10"``, ``">1280"``,
``"~80"``); an untested cell is an empty list there, never a token, so there is no
"missing" kind here. Anything else is an error rather than a guess (design rule 4).

``log`` is log2(value / 10), the acmacs convention, for every kind. It is the face value:
``<10`` logs to 0, not -1. Consumers that average titres decide how to treat censored
readings; the store records what the lab reported.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

KINDS = {"<": "lt", ">": "gt", "~": "dodgy"}


class TitreError(ValueError):
    """A reading that is not a titre token."""


@dataclass(frozen=True)
class Reading:
    raw: str
    kind: str  # "num", "lt", "gt" or "dodgy"
    value: float
    log: float


def parse_reading(raw: str) -> Reading:
    """Parse one reading token; raise :class:`TitreError` for anything that is not one."""
    kind = KINDS.get(raw[:1], "num")
    number = raw[1:] if kind != "num" else raw
    try:
        value = float(number)
    except ValueError:
        raise TitreError(f"not a titre reading: {raw!r}") from None
    if not math.isfinite(value) or value <= 0:
        raise TitreError(f"not a titre reading: {raw!r}")
    return Reading(raw=raw, kind=kind, value=value, log=math.log2(value / 10.0))
