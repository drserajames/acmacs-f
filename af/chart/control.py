"""Control-chart rules for repeated HI titres: a port of the R package hicontrol.

The method is Sarah James and Yi Dong's (hicontrol, GPL-3,
https://github.com/drserajames/hicontrol, ported from commit
aca6dd822e920f333f9c320685fb8f89c29b8fc2). Eight Westgard/Nelson-style rules are adapted to
interval- and left-censored HI titres on the log2(titre/10) scale, and are applied to the
series of readings of one antigen/serum cell across tables (one reading per layer of a merged
chart). This module is a line-by-line port kept equivalent to the R code:
`tests/chart/hicontrol/` runs every case in `hicontrol-cases.json`, which the R package itself
produced (`make_fixtures.R`), and requires identical results.

Why a port and not a reimplementation: the rules are the evidence Sarah weighs against the
inherited `sd_limit` merge rule, so the numbers must be hicontrol's numbers.

Conventions kept from R, on purpose:
* Indices are 0-based here (R's are 1-based); nothing else about them changes.
* `None` means "the series is too short for this rule" (R `NULL`); `[]` means "tested, nothing
  found" (R `integer(0)` or a 0-column matrix). Several rules also return `None` when they
  find nothing, exactly as R does.
* Missing readings are NaN and behave as R's NA: a comparison with NA is NA, a cumulative sum
  stays NA after the first NA (so `rule_xyz` and `rule_alt` see nothing past a missing
  reading), and every NA is a run of its own in `rle`.
* `rule_trend` reproduces an R quirk: when the first qualifying run starts the series, only
  that run's start is recorded and the starts of later runs of the same direction are `None`.
* `log_num` maps `<x` to half the detection limit and anything it cannot read (`*`, `>x`)
  to NaN, as the R function does.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

HICONTROL_COMMIT = "aca6dd822e920f333f9c320685fb8f89c29b8fc2"

Series = Sequence[float]  # NaN = missing (R NA)
Bool3 = bool | None  # R logical with NA


def log_num(titres: Sequence[str]) -> list[float]:
    """HI titres to log2(titre/10); `<x` is half the detection limit, log2(x/20)."""
    out = []
    for t in titres:
        if "<" in t:
            out.append(_log2_or_nan(t.replace("<", ""), 20.0))
        else:
            out.append(_log2_or_nan(t, 10.0))
    return out


def _log2_or_nan(text: str, divisor: float) -> float:
    try:
        value = float(text)
    except ValueError:
        return math.nan
    if value == 0:
        return -math.inf  # R: log(0) is -Inf
    return math.log2(value / divisor) if value > 0 else math.nan


# ----------------------------------------------------------------------
# R semantics


def _na(v: float) -> bool:
    return math.isnan(v)


def _cmp(values: Series, predicate: object) -> list[Bool3]:
    return [None if _na(v) else bool(predicate(v)) for v in values]  # type: ignore[operator]


def _cumsum_from_zero(flags: Sequence[float | None]) -> list[float | None]:
    """R `c(0, cumsum(x))`: NA from the first NA on."""
    out: list[float | None] = [0.0]
    total: float | None = 0.0
    for f in flags:
        total = None if total is None or f is None else total + f
        out.append(total)
    return out


def _rle(values: Sequence[object]) -> tuple[list[int], list[object]]:
    """R `rle`: runs break wherever neighbours differ or either is NA (None or NaN)."""
    n = len(values)
    if n == 0:
        return [], []

    def is_na(v: object) -> bool:
        return v is None or (isinstance(v, float) and math.isnan(v))

    ends = [
        i
        for i in range(n - 1)
        if is_na(values[i]) or is_na(values[i + 1]) or values[i] != values[i + 1]
    ]
    ends.append(n - 1)
    lengths, vals, prev = [], [], -1
    for e in ends:
        lengths.append(e - prev)
        vals.append(values[e])
        prev = e
    return lengths, vals


def _run_starts(lengths: Sequence[int], chosen: Sequence[int]) -> list[int]:
    """0-based start of each chosen run (R: `sum(lengths[1:(k - 1)]) + 1`)."""
    return [sum(lengths[:k]) for k in chosen]


def _diff(dat: Series) -> list[float]:
    """R `dat[1:(len-1)] - dat[2:len]` for len >= 2."""
    return [dat[i] - dat[i + 1] for i in range(len(dat) - 1)]


# ----------------------------------------------------------------------
# the rules

Runs = list[tuple[int, int]]  # (start, length)


def rule_xyz(
    dat: Series, centre: float, threshold: float, x: int, y: int, z: float
) -> list[int] | None:
    """Windows of `y` points with at least `x` strictly beyond centre +/- z*threshold."""
    if len(dat) < y:
        return None
    upp, low = centre + z * threshold, centre - z * threshold
    hits = []
    for flags in (_cmp(dat, lambda v: v > upp), _cmp(dat, lambda v: v < low)):
        c = _cumsum_from_zero([None if f is None else float(f) for f in flags])
        for k in range(len(c) - y):
            a, b = c[k + y], c[k]
            if a is not None and b is not None and (a - b) / y >= x / y:
                hits.append(k)
    return hits


def rule_trend(dat: Series, n: int) -> list[tuple[int | None, int, int]] | None:
    """Runs of >= n strictly monotone points: (start, length, direction -1 down / +1 up)."""
    if len(dat) < n:
        return None
    diff = _diff(dat)
    result: list[tuple[int | None, int, int]] = []
    for direction, flags in ((-1, _cmp(diff, lambda v: v > 0)), (1, _cmp(diff, lambda v: v < 0))):
        lengths, values = _rle(flags)
        chosen = [
            k
            for k, (ln, v) in enumerate(zip(lengths, values, strict=True))
            if ln >= n - 1 and v is True
        ]
        starts: list[int | None] = [None] * len(chosen)
        for i in range(len(chosen)):
            if chosen[0] == 0:  # R checks run_n[1] == 1 for every i (the quirk above)
                starts[0] = 0
            else:
                starts[i] = sum(lengths[: chosen[i]])
        result += [(s, lengths[k] + 1, direction) for s, k in zip(starts, chosen, strict=True)]
    return result or None


def _zone_runs(
    dat: Series, centre: float, threshold: float, n: int, outside: object, want: bool
) -> Runs | None:
    if len(dat) < n:
        return None
    lengths, values = _rle(_cmp(dat, lambda v: outside(abs(v - centre), threshold)))  # type: ignore[operator]
    chosen = [
        k for k, (ln, v) in enumerate(zip(lengths, values, strict=True)) if ln >= n and v is want
    ]
    if not chosen:
        return None
    return list(zip(_run_starts(lengths, chosen), [lengths[k] for k in chosen], strict=True))


def rule_noC(dat: Series, centre: float, threshold: float, n: int) -> Runs | None:  # noqa: N802
    """Runs of >= n points all strictly outside centre +/- threshold (mixture)."""
    return _zone_runs(dat, centre, threshold, n, lambda d, t: d > t, True)


def rule_onlyC(dat: Series, centre: float, threshold: float, n: int) -> Runs | None:  # noqa: N802
    """Runs of >= n points all within centre +/- threshold (stratification)."""
    return _zone_runs(dat, centre, threshold, n, lambda d, t: d > t, False)


def rule_noCrelax(dat: Series, centre: float, threshold: float, n: int) -> Runs | None:  # noqa: N802
    """rule_noC with points on the boundary counted as outside (>=)."""
    return _zone_runs(dat, centre, threshold, n, lambda d, t: d >= t, True)


def rule_alt(dat: Series, n: int) -> Runs | None:
    """Runs of >= n points alternating up and down (over-control)."""
    if len(dat) < n:
        return None
    diff = _diff(dat)
    signs = [None if _na(d) else float((d > 0) - (d < 0)) for d in diff]
    c = _cumsum_from_zero(signs)
    rsum = [
        None if c[k + 2] is None or c[k] is None else (c[k + 2] - c[k]) / 2  # type: ignore[operator]
        for k in range(len(c) - 2)
    ]
    lengths, values = _rle(rsum)
    chosen = [
        k for k, (ln, v) in enumerate(zip(lengths, values, strict=True)) if ln >= n - 2 and v == 0
    ]
    if not chosen:
        return None
    starts = _run_starts(lengths, chosen)
    # R: match(which(diff != 0), ind) with NAs dropped, i.e. keep runs whose start is a real step
    nonzero = [i for i, d in enumerate(diff) if not _na(d) and d != 0]
    keep = [starts.index(p) for p in nonzero if p in starts]
    return [(starts[j], lengths[chosen[j]] + 2) for j in keep]


def rule_nodiff(dat: Series, n: int) -> Runs | None:
    """Runs of >= n consecutive identical values."""
    if len(dat) < n:
        return None
    lengths, values = _rle(_diff(dat))
    chosen = [
        k for k, (ln, v) in enumerate(zip(lengths, values, strict=True)) if ln >= n - 1 and v == 0
    ]
    if not chosen:
        return None
    return list(zip(_run_starts(lengths, chosen), [lengths[k] + 1 for k in chosen], strict=True))


def rule_diff(dat: Series, d: float) -> list[int]:
    """Indices i with |dat[i] - dat[i+1]| >= d. An empty series is an error, as in R."""
    if not dat:
        raise ValueError(
            "rule_diff: empty series (R: only 0's may be mixed with negative subscripts)"
        )
    return [i for i, v in enumerate(_diff(dat)) if not _na(v) and abs(v) >= d]


# ----------------------------------------------------------------------
# the eight HI rules together

RULE_NAMES = (
    "1 >=3",
    "2/3 >=2",
    "8 >=1",
    "9 >=1 both",
    "no diff 25",
    "alt 10",
    "trend 4",
    "diff of 3",
)
_N_MINUS = (1, 3, 8, 9, 25, 10, 4, 3)


@dataclass(frozen=True)
class RulesResult:
    """hicontrol's `hi_rules2`: each rule's raw result and the number of points it flags."""

    results: tuple[object, ...]
    counts: tuple[int, ...]

    def rates(self, n_points: int) -> tuple[float, ...]:
        """hicontrol's `hi_rules`: the counts as proportions of the series length."""
        return tuple(c / n_points for c in self.counts)


def hi_rules(dat: Series, centre: float, threshold: float) -> RulesResult:
    """All eight rules; `threshold` is one SD unit (hicontrol's reference panel uses 1)."""
    results: tuple[object, ...] = (
        rule_xyz(dat, centre, threshold, 1, 1, 3 - 1e-5),
        rule_xyz(dat, centre, threshold, 2, 3, 2 - 1e-5),
        rule_xyz(dat, centre, threshold, 8, 8, 0),
        rule_noCrelax(dat, centre, threshold, 9),
        rule_nodiff(dat, 25),
        rule_alt(dat, 10),
        rule_trend(dat, 4),
        rule_diff(dat, 3),
    )
    counts = []
    for i, r in enumerate(results):
        if r is None:
            counts.append(0)
        elif i < 3 or i == 7:
            counts.append(len(r))  # type: ignore[arg-type]
        else:
            counts.append(sum(run[1] - _N_MINUS[i] + 1 for run in r))  # type: ignore[attr-defined]
    return RulesResult(results, tuple(counts))
