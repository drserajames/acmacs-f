"""Orient one chart onto another: match their points, fit by Procrustes, report what matched.

Public, for anything that needs "chart A in chart B's frame" (pyacmapcheck, chain pages, map
reviews; Sarah, 1 Oct 2026). Three pieces, each usable alone:

- :func:`chart_points` - one plain record per antigen and serum. Needs no projection.
- :func:`match_points` - which point of A is which point of B, with every count.
- :func:`orient` - the rigid fit of A's layout onto B's (rotation, optional reflection,
  translation; never scaling, since map units are log2 fold), with RMSD and per-point
  distances.

How points are matched: the CALLER says, always (the same principle as column bases and
disconnected points: the caller resolves identity). In order of precedence:

1. ``match``: a :class:`PointMatch` the caller built (af's flu identity matching does this:
   :func:`af.map.matching.orient_by_identity`).
2. ``pairs``: the caller says which rows are the same point.
3. ``key``: a callable from a point record to a key string, or None for "matches nothing". Two
   points pair when their keys are equal and each key occurs once on its side. Nothing falls
   back: a None key is unmatched and counted.

With none of them the aligner refuses (:class:`AlignmentError`): it never picks a matching rule
of its own, because any rule that pairs viruses across charts knows how they are named.

In every mode a key that occurs twice on one side is dropped and counted, never merged.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from af.chart.model import Chart
from af.chart.procrustes import ProcrustesResult, procrustes

Point = dict[str, Any]


#: Why the aligner refuses without a matching rule, and where af's rule is.
NO_MATCHING = (
    "orient/match_points need pairs=, key= or match=: the aligner has no matching rule of its own. "
    "For af's influenza identity matching (isolate/year/passage class/date, as the report "
    "comparison uses), call af.map.matching.orient_by_identity or match_by_identity."
)


class AlignmentError(ValueError):
    """The two charts cannot be fitted: layouts of different dimension, or too few matched
    points with coordinates. A subclass of ValueError, so existing handlers still catch it, but
    distinct, so a caller can tell "these charts do not align" from any other failure."""


MatchKey = Callable[[Mapping[str, Any]], "str | None"]


def chart_points(chart: Chart) -> list[Point]:
    """One record per point, antigens first then sera, in chart order.

    ``index`` is the row in the chart's layout (sera follow the antigens), so a pair of records
    gives the two layout rows directly. Fields: ``kind`` ("antigen"/"serum"), ``id`` ("ag<i>" /
    "sr<j>"), ``name``, ``reassortant``, ``annotations``, ``passage`` (as written),
    ``date`` (antigens; None if unknown) and ``serum_id`` (sera). Nothing interpreted: a key
    that needs more (a passage class, a parsed strain name) computes it itself.
    """
    out: list[Point] = []
    for i, a in enumerate(chart.antigens):
        out.append({
            "kind": "antigen", "index": i, "id": f"ag{i}", "name": a.name,
            "reassortant": a.reassortant, "annotations": tuple(a.annotations),
            "passage": a.passage,
            "date": a.date[:10] or None, "serum_id": None,
        })  # fmt: skip
    for j, s in enumerate(chart.sera):
        out.append({
            "kind": "serum", "index": chart.n_antigens + j, "id": f"sr{j}", "name": s.name,
            "reassortant": s.reassortant, "annotations": tuple(s.annotations),
            "passage": s.passage,
            "date": None, "serum_id": s.serum_id or None,
        })  # fmt: skip
    return out


@dataclass(frozen=True)
class PointMatch:
    """Which point of A is which point of B, and how every point fared.

    ``pairs`` rows are (layout row in A, layout row in B). ``counts`` always holds, per kind,
    how many matched, how many on each side found no partner, and how many were dropped because
    their key occurred twice on their side; plus, for the default identity matching, how many
    points were keyed by name instead (``fallback_*``).
    """

    pairs: NDArray[np.intp]
    keys: tuple[str, ...]
    mode: str  # "identity", "key" or "pairs"
    counts: dict[str, int] = field(default_factory=dict)


def match_points(a: Chart, b: Chart, *, key: MatchKey | None = None) -> PointMatch:
    """Match A's points to B's by ``key`` (see the module docstring). ``key`` is required: the
    aligner has no matching rule of its own, and refuses rather than guess one."""
    if key is None:
        raise AlignmentError(NO_MATCHING)
    pa, pb = chart_points(a), chart_points(b)
    pairs: list[tuple[int, int]] = []
    keys: list[str] = []
    counts: dict[str, int] = {}
    for kind in ("antigen", "serum"):
        ka = [(key(p), p) for p in pa if p["kind"] == kind]
        kb = [(key(p), p) for p in pb if p["kind"] == kind]
        kp, kk, kc = pair_keyed(ka, kb, kind)
        pairs += kp
        keys += kk
        counts.update(kc)
    return PointMatch(np.array(pairs, dtype=np.intp).reshape(-1, 2), tuple(keys), "key", counts)


Keyed = Sequence[tuple[str | None, Point]]


def pair_keyed(
    keyed_a: Keyed, keyed_b: Keyed, kind: str
) -> tuple[list[tuple[int, int]], list[str], dict[str, int]]:
    """Pair two sides' keyed records of one kind: equal keys that occur once on each side.

    Returns the (row in A, row in B) pairs, their keys, and the counts: matched, unmatched per
    side, keys dropped as duplicates per side, records with no key per side.
    """
    ua, dup_a, none_a = _unique(keyed_a)
    ub, dup_b, none_b = _unique(keyed_b)
    common = sorted(ua.keys() & ub.keys())
    counts = {
        f"matched_{kind}": len(common),
        f"unmatched_a_{kind}": len(ua) - len(common) + dup_a + none_a,
        f"unmatched_b_{kind}": len(ub) - len(common) + dup_b + none_b,
        f"duplicate_key_a_{kind}": dup_a,
        f"duplicate_key_b_{kind}": dup_b,
        f"no_key_a_{kind}": none_a,
        f"no_key_b_{kind}": none_b,
    }
    return [(ua[k]["index"], ub[k]["index"]) for k in common], common, counts


def _unique(
    keyed: Sequence[tuple[str | None, Point]],
) -> tuple[dict[str, Point], int, int]:
    """Key -> point for keys occurring once; points dropped as duplicates; points with no key."""
    counts = Counter(k for k, _ in keyed if k is not None)
    unique = {k: p for k, p in keyed if k is not None and counts[k] == 1}
    duplicated = sum(n for n in counts.values() if n > 1)
    return unique, duplicated, sum(1 for k, _ in keyed if k is None)


@dataclass(frozen=True)
class Alignment:
    """A's layout fitted onto B's. ``fit.distances`` follows ``match.pairs`` row for row (NaN
    where either side has no coordinates); ``apply`` puts any layout in A's frame into B's."""

    match: PointMatch
    fit: ProcrustesResult

    def apply(self, layout: NDArray[np.float64]) -> NDArray[np.float64]:
        return self.fit.apply(layout)


def orient(
    a: Chart,
    b: Chart,
    *,
    layout_a: NDArray[np.float64] | None = None,
    layout_b: NDArray[np.float64] | None = None,
    key: MatchKey | None = None,
    pairs: Sequence[tuple[int, int]] | NDArray[np.intp] | None = None,
    match: PointMatch | None = None,
    allow_reflection: bool = True,
) -> Alignment:
    """Fit A's layout onto B's over their matched points.

    Layouts default to each chart's first projection as drawn (its transformation applied). A
    chart with no projection needs its layout passed in; it is never guessed. Points are paired
    by ``match``, ``pairs`` or ``key``, in that order; with none of them this refuses (see the
    module docstring).
    """
    la = _layout(a, layout_a, "A")
    lb = _layout(b, layout_b, "B")
    if match is not None:
        pairs = match.pairs
    if pairs is not None:
        arr = np.asarray(pairs, dtype=np.intp).reshape(-1, 2)
        for side, col, n in (("A", 0, len(la)), ("B", 1, len(lb))):
            if len(arr) and (arr[:, col].min() < 0 or arr[:, col].max() >= n):
                raise ValueError(f"a supplied pair names a row outside chart {side}'s layout")
        if match is None:
            supplied = {"supplied": len(arr)}
            match = PointMatch(arr, tuple(f"{i}-{j}" for i, j in arr), "pairs", supplied)
    elif key is not None:
        match = match_points(a, b, key=key)
    else:
        raise AlignmentError(NO_MATCHING)
    if la.shape[1] != lb.shape[1]:
        raise AlignmentError(
            f"chart A's layout is {la.shape[1]}-D and chart B's is {lb.shape[1]}-D: "
            "a rigid fit needs both in the same dimension"
        )
    usable = int(
        (np.isfinite(la[match.pairs[:, 0]]).all(axis=1)
         & np.isfinite(lb[match.pairs[:, 1]]).all(axis=1)).sum()
    ) if len(match.pairs) else 0  # fmt: skip
    if usable < la.shape[1] + 1:
        raise AlignmentError(
            f"only {usable} matched points have coordinates on both sides; a {la.shape[1]}-D "
            f"fit needs at least {la.shape[1] + 1} ({match.mode} matching: {dict(match.counts)})"
        )
    fit = procrustes(lb[match.pairs[:, 1]], la[match.pairs[:, 0]], allow_reflection)
    counts = dict(match.counts)
    counts["fitted"] = fit.n_common
    counts["no_coordinates"] = len(match.pairs) - fit.n_common
    return Alignment(PointMatch(match.pairs, match.keys, match.mode, counts), fit)


def _layout(chart: Chart, given: NDArray[np.float64] | None, side: str) -> NDArray[np.float64]:
    if given is not None:
        layout = np.asarray(given, dtype=float)
    elif chart.projections:
        layout = chart.projections[0].transformed_layout()
    else:
        raise ValueError(f"chart {side} has no projection: pass its layout explicitly")
    n = chart.n_antigens + chart.n_sera
    if layout.ndim != 2 or layout.shape[0] != n:
        raise AlignmentError(f"chart {side}: layout has shape {layout.shape} for {n} points")
    return layout
