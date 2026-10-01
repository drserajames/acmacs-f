"""af's influenza point matching for orienting one chart onto another (af side).

The core aligner (:mod:`af.map.align`) has no matching rule of its own: any rule that pairs
viruses across charts knows how they are named, and that is flu knowledge. This module is af's
rule: the report comparison's identity matching (:func:`af.report.compare.maps.keyed_points`,
mode ``identity``): isolate number, year, passage class and isolation date (sera: isolate,
year, serum id), the location left out because af keeps each lab's spelling of a place. A point
with no identity, or one its identity cannot pair, is keyed by its spelling-normalised name,
on both sides, and counted (``fallback_*``); that fallback is part of the comparison's rule.

Behaviour is exactly what :func:`af.map.align.orient` did by default before the core stopped
choosing a rule for its callers.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from af.chart.model import Chart
from af.map.align import Alignment, Point, PointMatch, chart_points, orient, pair_keyed
from af.map.vaccines import passage_class
from af.report.compare.maps import keyed_points


def identity_points(chart: Chart) -> list[Point]:
    """:func:`af.map.align.chart_points` plus each point's passage class, which the identity
    key reads."""
    out = []
    for p in chart_points(chart):
        out.append({**p, "passage_class": passage_class(p["passage"], p["reassortant"])})
    return out


def match_by_identity(a: Chart, b: Chart) -> PointMatch:
    """A's points paired with B's by af's identity matching, with every count."""
    pa, pb = identity_points(a), identity_points(b)
    pairs: list[tuple[int, int]] = []
    keys: list[str] = []
    counts: dict[str, int] = {}
    for kind in ("antigen", "serum"):
        ka, kb, fallback = keyed_points(
            [p for p in pa if p["kind"] == kind], [p for p in pb if p["kind"] == kind], "identity"
        )
        for reason, n in fallback.items():
            counts[f"fallback_{reason}_{kind}"] = counts.get(f"fallback_{reason}_{kind}", 0) + n
        kp, kk, kc = pair_keyed(list(ka), list(kb), kind)
        pairs += kp
        keys += kk
        counts.update({k: v for k, v in kc.items() if not k.startswith("no_key_")})
    arr = np.array(pairs, dtype=np.intp).reshape(-1, 2)
    return PointMatch(arr, tuple(keys), "identity", counts)


def orient_by_identity(
    a: Chart,
    b: Chart,
    *,
    layout_a: NDArray[np.float64] | None = None,
    layout_b: NDArray[np.float64] | None = None,
    allow_reflection: bool = True,
) -> Alignment:
    """:func:`af.map.align.orient` with af's identity matching."""
    return orient(
        a,
        b,
        layout_a=layout_a,
        layout_b=layout_b,
        match=match_by_identity(a, b),
        allow_reflection=allow_reflection,
    )
