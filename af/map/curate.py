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
from typing import Literal, Protocol

import numpy as np
from numpy.typing import NDArray

from af.map.orient import decompose, procrustes, rows_with_coordinates

Array = NDArray[np.float64]
Mask = NDArray[np.bool_]


class Relax(Protocol):
    """Minimise stress from ``start`` moving only ``movable`` points; return (layout, stress).
    Rows of ``start`` that are NaN are points without coordinates and stay NaN."""

    def __call__(self, start: Array, movable: Mask) -> tuple[Array, float]: ...


class CurationError(ValueError):
    """A curation rule matched the wrong points, or its result broke a guard.

    ``guard`` names the check that refused, with what it ``measured`` and the ``limit`` it was
    held to (``bound``: "max", "min" or "exact"), so a report can state the refusal from data
    rather than by parsing the message.
    """

    def __init__(
        self,
        message: str,
        *,
        guard: str,
        measured: float,
        limit: float,
        bound: Literal["max", "min", "exact"],
    ) -> None:
        super().__init__(message)
        self.guard = guard
        self.measured = measured
        self.limit = limit
        self.bound = bound

    def fields(self) -> dict[str, object]:
        """The refusal as plain data: guard, measured, limit, bound."""
        return {
            "guard": self.guard,
            "measured": round(float(self.measured), 4),
            "limit": self.limit,
            "bound": self.bound,
        }


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
    movers_moved: float  # how far the furthest mover actually travelled
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
            "movers_moved": round(float(self.movers_moved), 4),
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
    if not rule.movers:
        raise CurationError(
            f"move {rule.name!r}: no movers", guard="movers", measured=0, limit=1, bound="min"
        )
    rows = []
    for want in rule.movers:
        hits = [i for i, d in enumerate(designations) if d == want]
        if len(hits) != 1:
            raise CurationError(
                f"move {rule.name!r}: mover {want!r} matches {len(hits)} antigens",
                guard="mover_match",
                measured=len(hits),
                limit=1,
                bound="exact",
            )
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
            f"{rule.colour_scheme}, need at least {rule.min_target_points}",
            guard="target_points",
            measured=len(group),
            limit=rule.min_target_points,
            bound="min",
        )
    target = np.median(layout[group], axis=0)
    start = layout.copy()
    start[rows] = target
    out, stress_after = relax(start, has_xy)
    worst = float(np.linalg.norm(out[rows] - target, axis=1).max())
    moved = float(np.linalg.norm(out[rows] - layout[rows], axis=1).max())
    others = np.array(
        [i for i in range(len(layout)) if i not in movers and has_xy[i]], dtype=np.intp
    )
    largest_other = float(np.linalg.norm(out[others] - layout[others], axis=1).max())
    if stress_after - stress_before > rule.max_stress_rise:
        raise CurationError(
            f"move {rule.name!r}: stress rose {stress_after - stress_before:.3f} "
            f"(cap {rule.max_stress_rise}); layout left unchanged",
            guard="stress_rise",
            measured=stress_after - stress_before,
            limit=rule.max_stress_rise,
            bound="max",
        )
    if worst > rule.max_from_target:
        raise CurationError(
            f"move {rule.name!r}: a mover relaxed back {worst:.3f} from the target "
            f"(cap {rule.max_from_target}); layout left unchanged",
            guard="from_target",
            measured=worst,
            limit=rule.max_from_target,
            bound="max",
        )
    return MoveResult(
        layout=out,
        stress_before=stress_before,
        stress_after=stress_after,
        target=(float(target[0]), float(target[1])),
        movers_moved=moved,
        mover_rows=tuple(rows),
        target_points=len(group),
        worst_from_target=worst,
        largest_other_move=largest_other,
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


@dataclass(frozen=True)
class BlockOffset:
    """Shift named points by a fixed offset, keeping their shape, then relax the whole map.

    Why a fixed offset rather than a target: for a large, weakly constrained group the relax is
    sensitive to where it starts, so a shift recomputed from the current layout gives a visibly
    different result from one a person reviewed. The number is therefore part of the decision,
    like the reason, and is **round-bound**: it is derived from one round's layout and must be
    re-derived when the map is rebuilt. `derived_from` records how it was obtained, so the next
    round can repeat the derivation rather than guess.

    ``to = "target-median"`` instead of a shift (Sarah, 7 Oct 2026): the block is shifted so its
    median lands on the median of the target points, computed from the layout it is applied to.
    A fixed offset is tied to one layout's raw coordinates and points somewhere else once the map
    is re-laid out (a from-scratch map, a new chain); a target survives that. The shift actually
    used is reported with the result, so the figure still records the number.

    Guards, all checked before anything is returned: stress may rise by at most
    ``max_stress_rise``; at least ``min_settled`` of the movers must end within ``settled_within``
    of the median of the points painted ``target_colour``; and no other point may move more than
    ``max_other_move`` once the map is rigidly fitted back onto its old self.
    """

    name: str
    reason: str
    decided: str
    movers: tuple[str, ...]
    shift: tuple[float, float] | None
    derived_from: str | None
    colour_scheme: str
    target_colour: str
    max_stress_rise: float
    settled_within: float
    min_settled: int
    max_other_move: float
    to: str | None = None  # "target-median": shift computed from the layout, not given

    def __post_init__(self) -> None:
        if self.to is None:
            if self.shift is None or not self.derived_from:
                raise ValueError(f"block {self.name!r}: a fixed shift needs shift and derived_from")
        elif self.to != TO_TARGET_MEDIAN:
            raise ValueError(
                f"block {self.name!r}: to must be {TO_TARGET_MEDIAN!r}, not {self.to!r}"
            )
        elif self.shift is not None or self.derived_from:
            raise ValueError(
                f"block {self.name!r}: to = {TO_TARGET_MEDIAN!r} computes the shift; "
                "give no shift or derived_from"
            )


TO_TARGET_MEDIAN = "target-median"


@dataclass(frozen=True)
class BlockResult:
    layout: Array
    stress_before: float
    stress_after: float
    mover_rows: tuple[int, ...]
    settled: int
    largest_other_move: float
    rotation_degrees: float
    shift: tuple[float, float] = (0.0, 0.0)  # the shift applied (computed for a target rule)

    def report(self, rule: BlockOffset) -> dict[str, object]:
        placed = {"to": rule.to} if rule.to else {"derived_from": rule.derived_from}
        return {
            "override": rule.name,
            "reason": rule.reason,
            "decided": rule.decided,
            "shift": [round(float(self.shift[0]), 4), round(float(self.shift[1]), 4)],
            **placed,
            "movers": len(self.mover_rows),
            "settled": self.settled,
            "stress_before": round(float(self.stress_before), 4),
            "stress_after": round(float(self.stress_after), 4),
            "largest_other_move": round(float(self.largest_other_move), 4),
            "rotation_degrees": round(float(self.rotation_degrees), 4),
        }


def apply_block_offset(
    rule: BlockOffset,
    layout: Array,
    designations: Sequence[str],
    painted: Sequence[str | None],
    *,
    stress_before: float,
    relax: Relax,
) -> BlockResult:
    """Apply one :class:`BlockOffset`. Raises :class:`CurationError` if a guard fails."""
    if not rule.movers:
        raise CurationError(
            f"block {rule.name!r}: no movers", guard="movers", measured=0, limit=1, bound="min"
        )
    rows = []
    for want in rule.movers:
        hits = [i for i, d in enumerate(designations) if d == want]
        if len(hits) != 1:
            raise CurationError(
                f"block {rule.name!r}: mover {want!r} matches {len(hits)} antigens",
                guard="mover_match",
                measured=len(hits),
                limit=1,
                bound="exact",
            )
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
    if not group:
        raise CurationError(
            f"block {rule.name!r}: no points painted {rule.target_colour} by {rule.colour_scheme}",
            guard="target_points",
            measured=0,
            limit=1,
            bound="min",
        )
    if rule.to == TO_TARGET_MEDIAN:
        shift = np.median(layout[group], axis=0) - np.median(layout[rows], axis=0)
    else:
        assert rule.shift is not None  # checked by BlockOffset
        shift = np.asarray(rule.shift, dtype=float)
    if not np.isfinite(shift).all():
        raise CurationError(
            f"block {rule.name!r}: a mover has no coordinates; cannot place the block",
            guard="mover_coordinates",
            measured=0,
            limit=len(rows),
            bound="min",
        )
    start = layout.copy()
    start[rows] = start[rows] + shift
    out, stress_after = relax(start, has_xy)

    # Compare positions only after fitting the new map rigidly back onto the old one: a relax can
    # turn the whole map, and an unfitted comparison reads that rotation as every point moving.
    others = np.array(
        [i for i in range(len(layout)) if i not in movers and has_xy[i]], dtype=np.intp
    )
    fit = procrustes(out[others], layout[others], reflection=False)
    fitted = out @ fit.matrix + fit.translation
    largest_other = float(np.linalg.norm(fitted[others] - layout[others], axis=1).max())
    centre = np.median(fitted[group], axis=0)
    settled = int((np.linalg.norm(fitted[rows] - centre, axis=1) <= rule.settled_within).sum())
    degrees = decompose(fit.matrix)[0]

    if stress_after - stress_before > rule.max_stress_rise:
        raise CurationError(
            f"block {rule.name!r}: stress rose {stress_after - stress_before:.3f} "
            f"(cap {rule.max_stress_rise}); layout left unchanged",
            guard="stress_rise",
            measured=stress_after - stress_before,
            limit=rule.max_stress_rise,
            bound="max",
        )
    if settled < rule.min_settled:
        raise CurationError(
            f"block {rule.name!r}: only {settled} of {len(rows)} movers settled within "
            f"{rule.settled_within} of the target (need {rule.min_settled}); layout left unchanged",
            guard="settled",
            measured=settled,
            limit=rule.min_settled,
            bound="min",
        )
    if largest_other > rule.max_other_move:
        raise CurationError(
            f"block {rule.name!r}: another point moved {largest_other:.3f} "
            f"(cap {rule.max_other_move}); layout left unchanged",
            guard="other_move",
            measured=largest_other,
            limit=rule.max_other_move,
            bound="max",
        )
    return BlockResult(
        layout=out,
        stress_before=stress_before,
        stress_after=stress_after,
        mover_rows=tuple(rows),
        settled=settled,
        largest_other_move=largest_other,
        rotation_degrees=degrees,
        shift=(float(shift[0]), float(shift[1])),
    )
