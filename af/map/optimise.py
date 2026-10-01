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
import functools
import logging
import math
import os
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Literal

import numpy as np
import numpy.typing as npt

from af.map import _core

log = logging.getLogger(__name__)

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
    threads: int  # threads the starts ran on (resolved from the `threads` argument)

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
    return RelaxResult(
        projections=projections,
        column_bases=np.array(core.column_bases),
        threads=threads_used("relax", threads),
    )


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
    n_antigens: int  # members that are antigens (point index < MapProblem.n_antigens)
    n_sera: int  # members that are sera
    # Antigens and sera moved together. Allowed, but flagged for review (Sarah, 26 Sep 2026):
    # the groups measured so far were all antigens.
    mixed: bool


@dataclass(frozen=True)
class TrappedResolution:
    """Result of :func:`resolve_trapped`."""

    projection: Projection
    rounds: int  # grid tests run
    moved: int  # single-point moves applied in total
    last_grid: list[GridResult]  # the grid test of the returned map: no trapped points, unless
    # rounds ran out or the trapped points left all have a worse position (stress_diff > 0)
    groups: list[GroupMove] = field(default_factory=list)  # empty unless move_groups=True
    threads: int = 1  # threads the grid tests ran on (resolved from the `threads` argument)


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

    ``move_groups`` (off by default here; the chains turn it on through their options, as
    Sarah decided, 26 Sep 2026) adds one pass at the end for a
    blind spot ae shares: several points stuck together in a worse place. Each point's own
    gain is below the trap threshold, so none is trapped and the loop above never moves them.
    Points whose better positions share a displacement (within ``group_tolerance``) are
    shifted together and the map re-minimised; a move is kept only if the stress drops, so
    the map can only improve. Measured (notes/optimiser/GROUP-TRAPS.md): on a CDC B/Vic map
    it found the misplaced block of six antigens unprompted, and three more, moved them to
    within 0.07 of ae's positions (map RMSD to ae 0.156 -> 0.063); it found no group in 72
    other real maps; it costs about 0.1% of a chain step. Each group tried is reported as a
    :class:`GroupMove`, with ``mixed`` set when it holds both antigens and sera: allowed, but
    to be flagged for review (Sarah); every group measured so far was all antigens.
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
        return TrappedResolution(
            current, rounds, moved, grid, threads=threads_used("grid test", threads)
        )
    if grid_is_stale:
        grid = grid_test(problem, current.layout, step=step, threads=threads)
    current, groups = _move_groups(problem, current, grid, group_tolerance, method)
    if any(g.kept for g in groups):
        grid = grid_test(problem, current.layout, step=step, threads=threads)
    return TrappedResolution(
        current, rounds, moved, grid, groups, threads_used("grid test", threads)
    )


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
        tried.append(_group_move(problem, members, shift, current.stress, result.stress, kept))
        if kept:
            current = result
    return current, tried


def _group_move(
    problem: MapProblem,
    members: list[int],
    shift: FloatArray,
    stress_before: float,
    stress_after: float,
    kept: bool,
) -> GroupMove:
    n_antigens = sum(1 for point in members if point < problem.n_antigens)
    n_sera = len(members) - n_antigens
    return GroupMove(
        members=members,
        shift=tuple(float(x) for x in shift),
        stress_before=stress_before,
        stress_after=stress_after,
        kept=kept,
        n_antigens=n_antigens,
        n_sera=n_sera,
        mixed=n_antigens > 0 and n_sera > 0,
    )


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


def threads_used(what: str, threads: int) -> int:
    """The threads a call with ``threads`` runs on, logged. ``threads=0`` means OpenMP's
    default, which honours an inherited ``OMP_NUM_THREADS``: a login environment that sets it to
    1 makes every run single-threaded, silently (measured on an HPC: 7.2 s instead of 0.95 s).
    So the resolved count is recorded in every result and a shortfall is warned about."""
    resolved = _core.resolve_threads(threads)
    log.info("%s: %d thread(s) (requested %d)", what, resolved, threads)
    available = available_cpus()
    # without OpenMP a run is single-threaded by build, not by environment (CMake warns then)
    if threads == 0 and _core.openmp and resolved < available:
        _warn_fewer_threads(resolved, available, os.environ.get("OMP_NUM_THREADS"))
    return resolved


def available_cpus() -> int:
    """CPUs this process may run on: its affinity mask (what SLURM or a cgroup allows) where
    the platform reports one, otherwise all CPUs."""
    affinity = getattr(os, "sched_getaffinity", None)
    return len(affinity(0)) if affinity is not None else (os.cpu_count() or 1)


@functools.cache
def _warn_fewer_threads(resolved: int, available: int, omp_num_threads: str | None) -> None:
    # Once per combination, so a loop of grid tests does not repeat it.
    log.warning(
        "threads=0 resolved to %d thread(s) but %d CPUs are available (OMP_NUM_THREADS=%s); "
        "pass threads explicitly or unset OMP_NUM_THREADS",
        resolved,
        available,
        omp_num_threads,
    )


def stress(problem: MapProblem, layout: FloatArray) -> float:
    """Stress of a layout (disconnected rows ignored)."""
    return float(problem._core_problem.stress(_layout(problem, layout)))


def stress_table(problem: MapProblem, layout: FloatArray) -> FloatArray:
    """Each titre's own stress term, ``[n_antigens, n_sera]``: what :func:`stress` adds up.

    For diagnostics (which titres a map fits badly). A regular titre contributes
    ``w * (d - D)**2``, a ``<`` titre ``w * (d + 1 - D)**2 * sigmoid(10 * (d + 1 - D))``, with
    ``d`` its table distance (column basis minus log2(titre/10) minus the avidity
    adjustments, clipped at 0) and ``D`` the map distance. Cells are
    NaN where a titre is not fitted: missing, ``>``, dodgy unless ``dodgy_is_regular``, or
    touching a disconnected point. ``np.nansum(stress_table(...)) == stress(...)``.
    """
    return np.asarray(problem._core_problem.stress_table(_layout(problem, layout)))


def point_stress(problem: MapProblem, layout: FloatArray) -> FloatArray:
    """Each point's share of the stress, ``[n_points]`` (antigens then sera): the sum of the
    stress terms of the titres it takes part in (row sums of :func:`stress_table` for antigens,
    column sums for sera; 0 for a point with no fitted titre).

    Every titre joins one antigen and one serum, so it counts at both ends:
    ``point_stress(...).sum() == 2 * stress(...)``. This is the quantity the grid test minimises
    for one point at a time.
    """
    table = stress_table(problem, layout)
    return np.concatenate([np.nansum(table, axis=1), np.nansum(table, axis=0)])


def table_distances(problem: MapProblem) -> FloatArray:
    """The target map distance of each fitted titre, ``[n_antigens, n_sera]``: column basis
    minus log2(titre/10) minus the two avidity adjustments, clipped at 0 (ae's rule). NaN in
    exactly the cells where :func:`stress_table` is NaN.

    :func:`stress_table` is squared, so it says how badly a titre is fitted but not which way.
    The signed residual is **target minus map distance**,
    ``table_distances(problem) - D`` with ``D[ag, sr]`` the antigen-serum map distance:

    - positive: the map puts the pair **too close** for its titre;
    - negative: **too far apart**.

    For a ``<`` titre the target is a lower bound: the map distance should be at least the
    target plus 1, so only ``target + 1 - D > 0`` (too close) is an error. ae's error lines
    draw that as ``(target + 1 - D) * sqrt(sigmoid(10 * (target + 1 - D)))``.
    """
    return np.asarray(problem._core_problem.table_distances())


ColumnBasesMode = Literal["fixed", "recompute"]

REACTIVITY_PENALTY = (
    1.0  # Racmacs's default weight. UNCALIBRATED: a placeholder (see fit docstring)
)
AE_REACTIVITY_GRID = (
    1.0,
    2.0,
    3.0,
    4.0,
    5.0,
    6.0,
    -1.0,
    -2.0,
    -3.0,
    -4.0,
    -5.0,
    -6.0,
)  # ae's order


@dataclass(frozen=True)
class ReactivityFit:
    """Result of :func:`fit_antigen_reactivity`. Nothing in af applies it: to use it, a caller
    passes ``avidity_adjust = 2 ** reactivity`` for the antigens (and, if ``column_bases`` was
    ``"recompute"``, the returned ``column_bases``) in a new :class:`MapProblem`."""

    # [n_antigens], log2: the adjustment ADDED to each antigen's log titres (Racmacs's
    # agReactivityAdjustments, ae's logged avidity adjust). An antigen whose titres all read
    # too high for its position gets a NEGATIVE adjustment.
    reactivity: FloatArray
    projection: Projection  # the map relaxed with these reactivities
    stress: float  # map stress with the reactivities (penalty excluded)
    objective: float  # stress + sum((penalty * reactivity) ** 2)
    stress_before: float  # stress of the given layout without reactivity
    column_bases: FloatArray  # the column bases the fit uses
    penalty: float
    column_bases_mode: ColumnBasesMode
    clip: bool


def fit_antigen_reactivity(
    problem: MapProblem,
    layout: FloatArray,
    *,
    penalty: float = REACTIVITY_PENALTY,
    column_bases: ColumnBasesMode = "fixed",
    clip: bool = True,
    minimum_column_basis: float = 0.0,
    fixed: FloatArray | None = None,
    start: FloatArray | None = None,
    method: Method = "cg",
) -> ReactivityFit:
    """Fit one reactivity adjustment per antigen jointly with the map.

    The adjustment ``r_i`` is added to all of antigen i's log titres (Racmacs's convention,
    and ae's avidity adjust), so its target distances become ``cb - log2(titre/10) - r_i``.
    An antigen whose titres are uniformly too high for its position gets a negative ``r_i``.

    Minimises ``stress + sum((penalty * r) ** 2)`` over the layout and all antigens'
    reactivities together, from ``layout`` (normally a relaxed map), in one minimisation with
    an analytic gradient. This is Racmacs's ``optimizeAgReactivity`` objective; Racmacs
    re-relaxes the map for every finite-difference step, which this avoids.

    **Overfitting.** Each reactivity is a free parameter, and a free parameter can only lower
    the fitted stress. ae's ``avidity_test`` (reproduced by :func:`antigen_reactivity_scan`)
    flags an adjustment whenever the stress drops *at all*, judged one antigen at a time on
    the same titres it was fitted to: on real tables that flags noise for most antigens. It is
    harmless in ae only because ae never applies the result, so do not read it as a safe rule
    for applying adjustments. The penalty is the brake here, and its default (1, Racmacs's) is
    **uncalibrated**, a placeholder until a held-out-titre comparison decides it
    (notes/optimiser/REACTIVITY.md).

    ``column_bases``: ``"fixed"`` keeps the problem's column bases (af, interface I1, ae).
    ``"recompute"`` recomputes them from the adjusted titres, as Racmacs does, so raising the
    antigen that sets a serum's basis raises the basis too (``minimum_column_basis`` applies
    then). ``clip``: ae and af clip target distances at 0; Racmacs does not.

    Antigens only, as both references: a serum's overall reactivity is already absorbed by its
    column basis, so a serum offset would only trade off against the basis.

    ``fixed`` ([n_antigens]): NaN marks an antigen to fit, a finite value holds it there.
    Nothing in af applies the result automatically; see :class:`ReactivityFit`.
    """
    n_ag = problem.n_antigens
    fixed_arr = np.full(n_ag, np.nan) if fixed is None else np.asarray(fixed, dtype=np.float64)
    start_arr = np.zeros(n_ag) if start is None else np.asarray(start, dtype=np.float64)
    if fixed_arr.shape != (n_ag,) or start_arr.shape != (n_ag,):
        raise ValueError(f"fixed and start must have one entry per antigen ({n_ag})")
    layout_arr = _layout(problem, layout)
    raw = _core.fit_reactivity(
        problem._core_problem,
        layout_arr,
        start_arr,
        fixed_arr,
        penalty=float(penalty),
        column_bases=column_bases,
        clip=bool(clip),
        minimum_column_basis=float(minimum_column_basis),
        method=method,
    )
    return ReactivityFit(
        reactivity=np.asarray(raw["reactivity"]),
        projection=Projection(
            np.asarray(raw["layout"]),
            float(raw["stress"]),
            layout_arr.shape[1],
            int(raw["n_iterations"]),
            0,
            0,
            int(raw["termination"]),
        ),
        stress=float(raw["stress"]),
        objective=float(raw["objective"]),
        stress_before=stress(problem, layout_arr),
        column_bases=np.asarray(raw["column_bases"]),
        penalty=float(penalty),
        column_bases_mode=column_bases,
        clip=bool(clip),
    )


@dataclass(frozen=True)
class ReactivityTest:
    """One antigen in :func:`antigen_reactivity_scan`."""

    antigen: int
    best: float  # the grid adjustment with the lowest stress, if below the original; else 0
    stress_diff: dict[float, float]  # adjustment -> relaxed stress minus original stress


def antigen_reactivity_scan(
    problem: MapProblem,
    layout: FloatArray,
    *,
    adjustments: tuple[float, ...] = AE_REACTIVITY_GRID,
    antigens: list[int] | None = None,
    column_bases: ColumnBasesMode = "fixed",
    minimum_column_basis: float = 0.0,
    method: Method = "cg",
) -> list[ReactivityTest]:
    """ae's ``projection.avidity_test``, for comparison with ae: each antigen alone, each
    grid adjustment set on that antigen only, one fine relax of the whole map from ``layout``,
    ``best`` = the adjustment with the lowest stress if it is below the original.

    A report, not a rule: the "any decrease" choice is fitted and judged on the same titres
    and flags noise (see :func:`fit_antigen_reactivity`). O(antigens x adjustments) relaxes.
    Target distances are always clipped at 0 here (as ae); ``column_bases="recompute"``
    recomputes the bases from the adjusted titres (not something ae does).
    """
    layout_arr = _layout(problem, layout)
    original = stress(problem, layout_arr)
    base = (
        np.ones(problem.n_points)
        if problem.avidity_adjust is None
        else np.asarray(problem.avidity_adjust)
    )
    results = []
    for antigen in range(problem.n_antigens) if antigens is None else antigens:
        diffs: dict[float, float] = {}
        for adjust in adjustments:
            avidity = base.copy()
            avidity[antigen] *= 2.0**adjust
            changes: dict[str, object] = {"avidity_adjust": avidity}
            if column_bases == "recompute":
                reactivity = np.zeros(problem.n_antigens)
                reactivity[antigen] = adjust
                changes["column_bases"] = _core.reactivity_column_bases(
                    problem._core_problem,
                    reactivity,
                    column_bases="recompute",
                    minimum_column_basis=float(minimum_column_basis),
                )
            adjusted = dataclasses.replace(problem, **changes)  # type: ignore[arg-type]
            diffs[adjust] = optimise(adjusted, layout_arr, method=method).stress - original
        best_adjust, best_diff = min(diffs.items(), key=lambda item: item[1])
        results.append(ReactivityTest(antigen, best_adjust if best_diff < 0 else 0.0, diffs))
    return results


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
