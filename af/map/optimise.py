"""Place antigens and sera on a map: the Python face of the C++ optimiser core.

The chart model (``af.chart``) turns a chart into a :class:`MapProblem` (interface I1):
titres as ``log2(titre/10)`` with a type code, column bases and disconnected points
already resolved. This module checks those arrays, runs the core and returns
:class:`Projection` objects sorted by stress.

Every run is seeded. Start *i* of a run seeded with *s* draws its random layout from its
own generator (SplitMix64 of *s* and *i*), so the same seed gives identical maps with any
number of threads, and identical starting layouts on any machine (a different compiler or
maths library can change the last bits of the minimised maps). ae shared one generator
between threads, so its seeded runs were not reproducible.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Literal

import numpy as np
import numpy.typing as npt

from af.map import _core

Method = Literal["cg", "lbfgs"]
Precision = Literal["fine", "rough", "very_rough"]
Diagnosis = Literal["excluded", "normal", "trapped", "hemisphering"]

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]


class TitreType(IntEnum):
    """Titre type codes in ``MapProblem.titre_type`` (I1)."""

    MISSING = 0
    REGULAR = 1
    LESS_THAN = 2
    MORE_THAN = 3
    DODGY = 4


@dataclass(frozen=True)
class MapProblem:
    """One table ready for the optimiser (interface I1). Points are antigens then sera.

    Everything that needs knowledge of the chart is decided by the caller, so the optimiser
    never guesses: ``column_bases`` are the ones to use (forced, minimum or merged),
    ``disconnected`` marks points the caller has decided to leave off the map.
    """

    titre_value: FloatArray  # [n_ag, n_sr] log2(titre/10); NaN when missing
    titre_type: npt.NDArray[np.int8]  # [n_ag, n_sr] TitreType codes
    column_bases: FloatArray  # [n_sr]
    disconnected: BoolArray  # [n_points]
    dodgy_is_regular: bool
    weights: FloatArray | None = None  # [n_ag, n_sr]
    avidity_adjust: FloatArray | None = None  # [n_points], raw factors (.ace projection "f")
    gradient_multipliers: FloatArray | None = None  # [n_points]; only 1.0 is supported
    unmovable: BoolArray | None = None  # [n_points]
    _core_problem: _core.Problem = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        # The C++ side re-checks everything; these checks give Python-side messages for the
        # common mistakes (wrong dtype that would silently truncate, wrong lengths).
        titre_type = np.asarray(self.titre_type)
        if not np.issubdtype(titre_type.dtype, np.integer):
            raise TypeError("titre_type must be an integer array of TitreType codes")
        disconnected = np.asarray(self.disconnected)
        if disconnected.dtype != np.bool_:
            raise TypeError("disconnected must be a boolean array")
        if self.unmovable is not None and np.asarray(self.unmovable).dtype != np.bool_:
            raise TypeError("unmovable must be a boolean array")
        core = _core.Problem(
            np.asarray(self.titre_value, dtype=np.float64),
            titre_type.astype(np.int8),
            np.asarray(self.column_bases, dtype=np.float64),
            disconnected,
            dodgy_is_regular=bool(self.dodgy_is_regular),
            weights=_optional_float(self.weights),
            avidity_adjust=_optional_float(self.avidity_adjust),
            gradient_multipliers=_optional_float(self.gradient_multipliers),
            unmovable=None if self.unmovable is None else np.asarray(self.unmovable),
        )
        object.__setattr__(self, "_core_problem", core)

    @property
    def n_antigens(self) -> int:
        return int(self._core_problem.n_antigens)

    @property
    def n_sera(self) -> int:
        return int(self._core_problem.n_sera)

    @property
    def n_points(self) -> int:
        return int(self._core_problem.n_points)


@dataclass(frozen=True)
class Projection:
    """One map. ``layout`` rows of disconnected points are NaN."""

    layout: FloatArray  # [n_points, dimensions]
    stress: float
    dimensions: int
    n_iterations: int
    start_seed: int  # seed of this start's generator (0 for optimise())
    start_index: int
    termination: int  # alglib termination type (>0); 0 if the layout was not minimised


@dataclass(frozen=True)
class RelaxResult:
    """Projections sorted by stress, and the column bases they were made with (echoed so
    the chart can record them faithfully)."""

    projections: list[Projection]
    column_bases: FloatArray

    @property
    def best(self) -> Projection:
        return self.projections[0]


@dataclass(frozen=True)
class GridResult:
    """Grid-test verdict for one point. ``stress_diff`` < 0 means moving it helps."""

    point: int
    diagnosis: Diagnosis
    position: FloatArray | None
    distance: float
    stress_diff: float


INCREMENTAL_REFINED = 5  # ae relax_incremental minimises its best five starts again, finely


def relax(
    problem: MapProblem,
    *,
    n_starts: int,
    seed: int,
    first_start: int = 0,
    dimensions: int = 2,
    start_layout: FloatArray | None = None,
    method: Method = "cg",
    precision: Precision = "fine",
    dimension_annealing: bool = False,
    keep: int | None = None,
    threads: int = 0,
) -> RelaxResult:
    """Make ``n_starts`` maps and return them sorted by stress.

    Without ``start_layout`` every point starts at random (a chain's *scratch* map).
    With it, only the NaN rows are placed at random and the whole map is then minimised
    (a chain's *incremental* map): every start is minimised roughly and, if ``precision``
    is fine, the best five again finely (:func:`refine`), as ae ``relax_incremental`` does.

    ``first_start`` runs starts ``first_start .. first_start + n_starts - 1``, each with the
    seed it has in a single run, so a run can be split into jobs. For an incremental map
    split into jobs, run each job with ``precision="rough"``, combine the projections, and
    call :func:`refine` once on the combination: refining each job's own best five would
    make the result depend on how the run was split.

    Random positions are drawn from a cube sized by a very rough map of the table. As in
    ae, ``unmovable`` points are held in that map (Sarah, 25 Sep 2026). Against sizing with
    every point free, this changed the best stress in only 9 of 72 incremental runs on real
    tables, in both directions (-1.7% to +0.3%): it matches ae rather than improving on it.
    """
    _require_positive("n_starts", n_starts)
    _require_seed(seed)
    _require_non_negative("first_start", first_start)
    keep_n = 0 if keep is None else _require_positive("keep", keep)
    core = problem._core_problem
    if start_layout is None:
        _require_positive("dimensions", dimensions)
        raw = _core.relax(
            core,
            dimensions=dimensions,
            n_starts=n_starts,
            first_start=first_start,
            seed=seed,
            method=method,
            precision=precision,
            dimension_annealing=dimension_annealing,
            keep=keep_n,
            threads=threads,
        )
        projections = [_projection(entry) for entry in raw]
    else:
        if dimension_annealing:
            raise ValueError("dimension_annealing applies only to maps from scratch")
        layout = _layout(problem, start_layout)
        if layout.shape[1] != dimensions:
            raise ValueError(f"start_layout has {layout.shape[1]} dimensions, not {dimensions}")
        rough = precision == "fine"
        raw = _core.relax_incremental(
            core,
            layout,
            n_starts=n_starts,
            first_start=first_start,
            seed=seed,
            method=method,
            precision="rough" if rough else precision,
            keep=0 if rough else keep_n,
            threads=threads,
        )
        projections = [_projection(entry) for entry in raw]
        if rough:
            projections = refine(problem, projections, method=method, threads=threads)
            if keep_n:
                projections = projections[:keep_n]
    return RelaxResult(projections=projections, column_bases=np.array(core.column_bases))


def refine(
    problem: MapProblem,
    projections: list[Projection],
    *,
    n_best: int = INCREMENTAL_REFINED,
    method: Method = "cg",
    threads: int = 0,
) -> list[Projection]:
    """Minimise the ``n_best`` lowest-stress projections again at fine precision and
    return all of them, re-sorted. The second stage of an incremental relax; run it once
    on the projections of all jobs when a run is split. Each projection keeps its start
    seed and index, so the order (stress, then start index) is the same however the starts
    were grouped."""
    _require_positive("n_best", n_best)
    ordered = sort_projections(projections)
    best, rest = ordered[:n_best], ordered[n_best:]
    raw = _core.refine(
        problem._core_problem,
        [_layout(problem, p.layout) for p in best],
        method=method,
        precision="fine",
        threads=threads,
    )
    refined = [
        dataclasses.replace(
            before,
            layout=np.asarray(after["layout"]),
            stress=float(after["stress"]),
            n_iterations=before.n_iterations + int(after["n_iterations"]),
            termination=int(after["termination"]),
        )
        for before, after in zip(best, raw, strict=True)
    ]
    return sort_projections(refined + rest)


def sort_projections(projections: list[Projection]) -> list[Projection]:
    """Stress ascending, NaN stress last, ties by start index: the core's order, so
    projections combined from several jobs sort exactly as one run's would."""
    return sorted(projections, key=lambda p: (math.isnan(p.stress), p.stress, p.start_index))


def optimise(
    problem: MapProblem,
    layout: FloatArray,
    *,
    method: Method = "cg",
    precision: Precision = "fine",
) -> Projection:
    """Minimise one given layout, without randomising anything (ae ``Projection.relax``)."""
    return _projection(
        _core.optimise(
            problem._core_problem, _layout(problem, layout), method=method, precision=precision
        )
    )


def grid_test(
    problem: MapProblem, layout: FloatArray, *, step: float = 0.1, threads: int = 0
) -> list[GridResult]:
    """Look for trapped and hemisphering points (one result per point)."""
    raw = _core.grid_test(
        problem._core_problem, _layout(problem, layout), step=step, threads=threads
    )
    return [
        GridResult(
            point=int(entry["point"]),
            diagnosis=entry["diagnosis"],
            position=None if entry["position"] is None else np.asarray(entry["position"]),
            distance=float(entry["distance"]),
            stress_diff=float(entry["stress_diff"]),
        )
        for entry in raw
    ]


@dataclass(frozen=True)
class GroupMove:
    """One group tried by :func:`resolve_trapped` with ``move_groups=True``: points whose
    grid-test better positions share a displacement, shifted together."""

    members: list[int]  # point indices
    shift: tuple[float, ...]  # the rigid displacement tried
    stress_before: float
    stress_after: float  # after moving the group and re-minimising
    kept: bool  # only a move that lowers the stress is kept


@dataclass(frozen=True)
class TrappedResolution:
    """Result of :func:`resolve_trapped`."""

    projection: Projection
    rounds: int  # grid tests run
    moved: int  # single-point moves applied in total
    last_grid: list[GridResult]  # the grid test of the returned map: no trapped points, unless
    # rounds ran out or the trapped points left all have a worse position (stress_diff > 0)
    groups: list[GroupMove] = field(default_factory=list)  # empty unless move_groups=True


GROUP_TOLERANCE = 0.5  # moves within this distance of each other form one group
GROUP_MIN_GAIN = 1e-6  # a group move must lower the stress by more than this to be kept


def resolve_trapped(
    problem: MapProblem,
    layout: FloatArray,
    *,
    max_rounds: int = 20,
    method: Method = "cg",
    step: float = 0.1,
    threads: int = 0,
    move_groups: bool = False,
    group_tolerance: float = GROUP_TOLERANCE,
) -> TrappedResolution:
    """Grid-test, move the points that have a better place, re-minimise; repeat until no
    point is trapped or ``max_rounds`` grid tests have run (ae ``chart-relax-grid``: 20).

    As in ae, every point whose better position lowers the stress is moved, including
    hemisphering ones, but only trapped points keep the loop going. A point is trapped when
    its move changes the stress by more than 0.25 either way; one whose move would *raise*
    the stress is reported but never moved. ae then repeats the loop on an unchanged map
    until the rounds run out; here a round that moves nothing ends the loop.

    ``move_groups`` (off by default; on is Sarah's decision) adds one pass at the end for a
    blind spot ae shares: several points stuck together in a worse place. Each point's own
    gain is below the trap threshold, so none is trapped and the loop above never moves them.
    Points whose better positions share a displacement (within ``group_tolerance``) are
    shifted together and the map re-minimised; a move is kept only if the stress drops, so
    the map can only improve. Measured (notes/optimiser/GROUP-TRAPS.md): on a CDC B/Vic map
    it found the misplaced block of six antigens unprompted, and three more, moved them to
    within 0.07 of ae's positions (map RMSD to ae 0.156 -> 0.063); it found no group in 72
    other real maps; it costs about 0.1% of a chain step.
    """
    _require_positive("max_rounds", max_rounds)
    if not group_tolerance > 0:
        raise ValueError(f"group_tolerance must be positive, not {group_tolerance!r}")
    # The layout is taken as given (normally the best projection of a relax), as ae does.
    start = _layout(problem, layout)
    current = Projection(start, stress(problem, start), start.shape[1], 0, 0, 0, 0)
    moved = 0
    grid: list[GridResult] = []
    rounds = max_rounds
    grid_is_stale = False  # rounds ran out: grid describes the map before the last move
    for round_no in range(1, max_rounds + 1):
        grid = grid_test(problem, current.layout, step=step, threads=threads)
        moves = [r for r in grid if r.position is not None and r.stress_diff < 0.0]
        if not any(result.diagnosis == "trapped" for result in grid) or not moves:
            rounds = round_no
            break
        new_layout = current.layout.copy()
        for result in moves:
            new_layout[result.point] = result.position
        moved += len(moves)
        current = optimise(problem, new_layout, method=method, precision="fine")
    else:
        grid_is_stale = True
    if not move_groups:
        return TrappedResolution(current, rounds, moved, grid)
    if grid_is_stale:
        grid = grid_test(problem, current.layout, step=step, threads=threads)
    current, groups = _move_groups(problem, current, grid, group_tolerance, method)
    if any(g.kept for g in groups):
        grid = grid_test(problem, current.layout, step=step, threads=threads)
    return TrappedResolution(current, rounds, moved, grid, groups)


def _move_groups(
    problem: MapProblem,
    current: Projection,
    grid: list[GridResult],
    tolerance: float,
    method: Method,
) -> tuple[Projection, list[GroupMove]]:
    """Shift each group of points with a shared better displacement; keep improvements."""
    tried: list[GroupMove] = []
    for members, shift in _displacement_groups(grid, current.layout, tolerance):
        trial = current.layout.copy()
        trial[members] += shift
        result = optimise(problem, trial, method=method, precision="fine")
        kept = result.stress < current.stress - GROUP_MIN_GAIN
        tried.append(
            GroupMove(members, tuple(float(x) for x in shift), current.stress, result.stress, kept)
        )
        if kept:
            current = result
    return current, tried


def _displacement_groups(
    grid: list[GridResult], layout: FloatArray, tolerance: float
) -> list[tuple[list[int], FloatArray]]:
    """Points with a better position (stress_diff < 0, a real move) grouped by displacement:
    connected components of "moves within ``tolerance``", two points or more, largest first
    (ties by lowest point index, so the order is deterministic)."""
    moves = {
        r.point: np.asarray(r.position) - layout[r.point]
        for r in grid
        if r.position is not None and r.stress_diff < 0.0 and r.distance > 0.0
    }
    points = sorted(moves)
    group_of: dict[int, int] = {}
    groups: list[list[int]] = []
    for point in points:
        if point in group_of:
            continue
        component, queue = [point], [point]
        group_of[point] = len(groups)
        while queue:
            here = queue.pop()
            for other in points:
                if other not in group_of and np.linalg.norm(moves[here] - moves[other]) < tolerance:
                    group_of[other] = len(groups)
                    component.append(other)
                    queue.append(other)
        groups.append(sorted(component))
    result = [(g, np.mean([moves[p] for p in g], axis=0)) for g in groups if len(g) >= 2]
    return sorted(result, key=lambda item: (-len(item[0]), item[0][0]))


def stress(problem: MapProblem, layout: FloatArray) -> float:
    """Stress of a layout (disconnected rows ignored)."""
    return float(problem._core_problem.stress(_layout(problem, layout)))


def gradient(problem: MapProblem, layout: FloatArray) -> FloatArray:
    """Analytic gradient of the stress (zero rows for unmovable and disconnected points)."""
    return np.asarray(problem._core_problem.gradient(_layout(problem, layout)))


def column_bases(
    titre_value: FloatArray, titre_type: npt.NDArray[np.int8], minimum: float = 0.0
) -> FloatArray:
    """Unforced column bases as ae computes them. For tests and comparisons only:
    production column bases are resolved by the chart model."""
    return np.asarray(
        _core.column_bases(
            np.asarray(titre_value, dtype=np.float64),
            np.asarray(titre_type, dtype=np.int8),
            minimum,
        )
    )


def _projection(entry: dict) -> Projection:
    return Projection(
        layout=np.asarray(entry["layout"]),
        stress=float(entry["stress"]),
        dimensions=int(entry["dimensions"]),
        n_iterations=int(entry["n_iterations"]),
        start_seed=int(entry["start_seed"]),
        start_index=int(entry["start_index"]),
        termination=int(entry["termination"]),
    )


def _layout(problem: MapProblem, layout: FloatArray) -> FloatArray:
    array = np.ascontiguousarray(layout, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] != problem.n_points:
        raise ValueError(
            f"layout must have shape ({problem.n_points}, dimensions), not {array.shape}"
        )
    return array


def _optional_float(array: FloatArray | None) -> FloatArray | None:
    return None if array is None else np.asarray(array, dtype=np.float64)


def _require_positive(name: str, value: int) -> int:
    if not isinstance(value, (int, np.integer)) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer, not {value!r}")
    return int(value)


def _require_non_negative(name: str, value: int) -> int:
    if not isinstance(value, (int, np.integer)) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer, not {value!r}")
    return int(value)


def _require_seed(seed: int) -> None:
    # Design rule 8: deterministic when seeded, so the seed is required and must fit uint64.
    if not isinstance(seed, (int, np.integer)) or isinstance(seed, bool) or not 0 <= seed < 2**64:
        raise ValueError(f"seed must be an integer in [0, 2**64), not {seed!r}")
