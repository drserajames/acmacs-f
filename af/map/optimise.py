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


def relax(
    problem: MapProblem,
    *,
    n_starts: int,
    seed: int,
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
    (a chain's *incremental* map): every start is minimised roughly and the best five
    again at ``precision``, as ae ``relax_incremental`` does.
    """
    _require_positive("n_starts", n_starts)
    _require_seed(seed)
    keep_n = 0 if keep is None else _require_positive("keep", keep)
    core = problem._core_problem
    if start_layout is None:
        _require_positive("dimensions", dimensions)
        raw = _core.relax(
            core,
            dimensions=dimensions,
            n_starts=n_starts,
            seed=seed,
            method=method,
            precision=precision,
            dimension_annealing=dimension_annealing,
            keep=keep_n,
            threads=threads,
        )
    else:
        if dimension_annealing:
            raise ValueError("dimension_annealing applies only to maps from scratch")
        layout = _layout(problem, start_layout)
        if layout.shape[1] != dimensions:
            raise ValueError(f"start_layout has {layout.shape[1]} dimensions, not {dimensions}")
        raw = _core.relax_incremental(
            core,
            layout,
            n_starts=n_starts,
            seed=seed,
            method=method,
            precision=precision,
            keep=keep_n,
            threads=threads,
        )
    return RelaxResult(
        projections=[_projection(entry) for entry in raw],
        column_bases=np.array(core.column_bases),
    )


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
class TrappedResolution:
    """Result of :func:`resolve_trapped`."""

    projection: Projection
    rounds: int  # grid tests run
    moved: int  # point moves applied in total
    last_grid: list[GridResult]  # the final grid test (no trapped points unless rounds ran out)


def resolve_trapped(
    problem: MapProblem,
    layout: FloatArray,
    *,
    max_rounds: int = 20,
    method: Method = "cg",
    step: float = 0.1,
    threads: int = 0,
) -> TrappedResolution:
    """Grid-test, move the points that have a better place, re-minimise; repeat until no
    point is trapped or ``max_rounds`` grid tests have run (ae ``chart-relax-grid``: 20).

    As in ae, every point whose better position lowers the stress is moved, including
    hemisphering ones, but only trapped points keep the loop going.
    """
    _require_positive("max_rounds", max_rounds)
    # The layout is taken as given (normally the best projection of a relax), as ae does.
    start = _layout(problem, layout)
    current = Projection(start, stress(problem, start), start.shape[1], 0, 0, 0, 0)
    moved = 0
    grid: list[GridResult] = []
    for round_no in range(1, max_rounds + 1):
        grid = grid_test(problem, current.layout, step=step, threads=threads)
        if not any(result.diagnosis == "trapped" for result in grid):
            return TrappedResolution(current, round_no, moved, grid)
        new_layout = current.layout.copy()
        for result in grid:
            if result.position is not None and result.stress_diff < 0.0:
                new_layout[result.point] = result.position
                moved += 1
        current = optimise(problem, new_layout, method=method, precision="fine")
    return TrappedResolution(current, max_rounds, moved, grid)


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


def _require_seed(seed: int) -> None:
    # Design rule 8: deterministic when seeded, so the seed is required and must fit uint64.
    if not isinstance(seed, (int, np.integer)) or isinstance(seed, bool) or not 0 <= seed < 2**64:
        raise ValueError(f"seed must be an integer in [0, 2**64), not {seed!r}")
