"""Hand curation of a map layout, as named, guarded, counted data: moves and continuity.

Why: today a map's layout is adjusted by per-folder scripts. The older ones move whatever falls in
a polygon drawn in a frame that changes every round, without a guard (one swept 414 antigens). The
newer ones (22-23 Sep 2026) select points by designation, move them to the median of a group in the
raw layout, relax, and refuse the result if stress rises too much or a point relaxes back. This
module keeps only the newer shape, as data:

* :class:`MoveOverride`: named points go to the median of a *painted colour* group (the colour a
  point gets from the map's colour scheme), then the whole map relaxes. Selecting by painted
  colour replaces the hand-copied list of clade exclusions the scripts carried, which silently
  drifted when the colour scheme changed. On the Sep 2026 round, h3 HI NIID's curated move is
  reproduced to RMSD 0.001 over 1065 points, stress identical.
* :func:`continuity_layout`: a relax that starts from the previous round's positions. Most large
  hand moves on that round restored last round's arrangement, and this reproduces two of them
  (RMSD 0.13 and 0.04 to the shipped maps). Sarah, 25 Sep 2026: it is never applied automatically.
  It goes on the chain review page and she picks per map.

The relaxation itself is injected (:class:`Relax`), so the optimiser core (``af.map._core``,
workstream 6) plugs in without this module knowing how stress is minimised.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from af.map.orient import procrustes, rows_with_coordinates

Array = NDArray[np.float64]
Mask = NDArray[np.bool_]


class Relax(Protocol):
    """Minimise stress from ``start`` moving only ``movable`` points; return (layout, stress).
    Rows of ``start`` that are NaN are points without coordinates and stay NaN."""

    def __call__(self, start: Array, movable: Mask) -> tuple[Array, float]: ...


class CurationError(ValueError):
    """A curation rule matched the wrong points, or its result broke a guard."""


@dataclass(frozen=True)
class MoveOverride:
    """Move named antigens into the group painted ``target_colour`` by ``colour_scheme``.

    ``movers`` are designations (name, reassortant and passage without its date) and must each
    match exactly one antigen. The target is the per-axis median, in the raw layout, of the other
    points painted that colour. Guards: stress may rise by at most ``max_stress_rise`` and every
    mover must end within ``max_from_target`` of the target after the relax.
    """

    name: str
    reason: str
    decided: str
    movers: tuple[str, ...]
    colour_scheme: str
    target_colour: str
    max_stress_rise: float
    max_from_target: float
    min_target_points: int = 5


@dataclass(frozen=True)
class MoveResult:
    layout: Array
    stress_before: float
    stress_after: float
    target: tuple[float, float]
    mover_rows: tuple[int, ...]
    target_points: int
    worst_from_target: float
    largest_other_move: float

    def report(self, rule: MoveOverride) -> dict[str, object]:
        return {
            "override": rule.name,
            "reason": rule.reason,
            "decided": rule.decided,
            "movers": len(self.mover_rows),
            "target_points": self.target_points,
            "stress_before": round(float(self.stress_before), 4),
            "stress_after": round(float(self.stress_after), 4),
            "worst_from_target": round(float(self.worst_from_target), 4),
            "largest_other_move": round(float(self.largest_other_move), 4),
        }


def apply_move(
    rule: MoveOverride,
    layout: Array,
    designations: Sequence[str],
    painted: Sequence[str | None],
    *,
    stress_before: float,
    relax: Relax,
) -> MoveResult:
    """Apply one :class:`MoveOverride` to ``layout`` (raw coordinates, antigens then sera).

    ``designations`` and ``painted`` describe the antigens (rows ``0..len-1``): each antigen's
    designation, and the colour the scheme ``rule.colour_scheme`` paints it (None = unpainted).
    Raises :class:`CurationError` before returning anything if a mover matches 0 or several
    antigens, the target group is too small, or a guard fails, so a failed rule can never leave a
    half-curated layout behind.
    """
    rows = []
    for want in rule.movers:
        hits = [i for i, d in enumerate(designations) if d == want]
        if len(hits) != 1:
            raise CurationError(f"move {rule.name!r}: mover {want!r} matches {len(hits)} antigens")
        rows.append(hits[0])
    has_xy = rows_with_coordinates(layout)
    movers = set(rows)
    group = [
        i
        for i, colour in enumerate(painted)
        if colour is not None
        and colour.lower() == rule.target_colour.lower()
        and i not in movers
        and has_xy[i]
    ]
    if len(group) < rule.min_target_points:
        raise CurationError(
            f"move {rule.name!r}: {len(group)} points painted {rule.target_colour} by "
            f"{rule.colour_scheme}, need at least {rule.min_target_points}"
        )
    target = np.median(layout[group], axis=0)
    start = layout.copy()
    start[rows] = target
    out, stress_after = relax(start, has_xy)
    worst = float(np.linalg.norm(out[rows] - target, axis=1).max())
    others = np.array(
        [i for i in range(len(layout)) if i not in movers and has_xy[i]], dtype=np.intp
    )
    largest_other = float(np.linalg.norm(out[others] - layout[others], axis=1).max())
    if stress_after - stress_before > rule.max_stress_rise:
        raise CurationError(
            f"move {rule.name!r}: stress rose {stress_after - stress_before:.3f} "
            f"(cap {rule.max_stress_rise}); layout left unchanged"
        )
    if worst > rule.max_from_target:
        raise CurationError(
            f"move {rule.name!r}: a mover relaxed back {worst:.3f} from the target "
            f"(cap {rule.max_from_target}); layout left unchanged"
        )
    return MoveResult(
        out,
        stress_before,
        stress_after,
        (float(target[0]), float(target[1])),
        tuple(rows),
        len(group),
        worst,
        largest_other,
    )


@dataclass(frozen=True)
class ContinuityResult:
    """A candidate layout for the review page; never applied without a named decision."""

    layout: Array
    stress_chain: float
    stress_continuity: float
    rmsd_chain_to_previous: float
    rmsd_continuity_to_previous: float
    rmsd_continuity_to_chain: float
    common_points: int

    def report(self) -> dict[str, object]:
        rise = self.stress_continuity - self.stress_chain
        return {
            "common_points": self.common_points,
            "stress_chain": round(float(self.stress_chain), 4),
            "stress_continuity": round(float(self.stress_continuity), 4),
            "stress_delta_pct": round(float(100 * rise / self.stress_chain), 3)
            if self.stress_chain
            else math.nan,
            "rmsd_chain_to_previous": round(float(self.rmsd_chain_to_previous), 4),
            "rmsd_continuity_to_previous": round(float(self.rmsd_continuity_to_previous), 4),
            "rmsd_continuity_to_chain": round(float(self.rmsd_continuity_to_chain), 4),
        }


def continuity_layout(
    chain: Array,
    previous_displayed: Array,
    pairs: NDArray[np.intp],
    *,
    stress_chain: float,
    relax: Relax,
) -> ContinuityResult:
    """Relax from the previous round's arrangement.

    Points present last round start where they were drawn then; new points start where the chain
    put them, after a rigid fit of the chain onto the previous map. ``pairs`` holds (row in chain,
    row in previous) for the same virus or serum.
    """
    fit = procrustes(chain[pairs[:, 0]], previous_displayed[pairs[:, 1]])
    start = chain @ fit.matrix + fit.translation
    prev = previous_displayed[pairs[:, 1]]
    known = rows_with_coordinates(prev)
    start[pairs[known, 0]] = prev[known]
    has_xy = rows_with_coordinates(chain)
    start[~has_xy] = np.nan
    out, stress = relax(start, has_xy)
    to_prev = procrustes(out[pairs[:, 0]], prev)
    to_chain = procrustes(out, chain)
    return ContinuityResult(
        out, stress_chain, stress, fit.rmsd, to_prev.rmsd, to_chain.rmsd, fit.n_common
    )
