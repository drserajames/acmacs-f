"""Type stub for the compiled optimiser core (cpp/optimiser/bindings.cc).

Use af.map.optimise, which checks its arguments and returns dataclasses; these are the raw
bindings it calls.
"""

from typing import Any

import numpy as np
import numpy.typing as npt

openmp: bool

class Problem:
    def __init__(
        self,
        titre_value: npt.NDArray[np.float64],
        titre_type: npt.NDArray[np.int8],
        column_bases: npt.NDArray[np.float64],
        disconnected: npt.NDArray[np.bool_],
        *,
        dodgy_is_regular: bool,
        weights: npt.NDArray[np.float64] | None = None,
        avidity_adjust: npt.NDArray[np.float64] | None = None,
        gradient_multipliers: npt.NDArray[np.float64] | None = None,
        unmovable: npt.NDArray[np.bool_] | None = None,
    ) -> None: ...
    @property
    def n_antigens(self) -> int: ...
    @property
    def n_sera(self) -> int: ...
    @property
    def n_points(self) -> int: ...
    @property
    def column_bases(self) -> npt.NDArray[np.float64]: ...
    def stress(self, layout: npt.NDArray[np.float64]) -> float: ...
    def gradient(self, layout: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]: ...
    def n_table_distances(self) -> tuple[int, int]: ...
    def max_table_distance(self) -> float: ...

def relax(
    problem: Problem,
    *,
    dimensions: int,
    n_starts: int,
    first_start: int = 0,
    seed: int,
    method: str = "cg",
    precision: str = "fine",
    dimension_annealing: bool = False,
    keep: int = 0,
    threads: int = 0,
) -> list[dict[str, Any]]: ...
def relax_incremental(
    problem: Problem,
    start_layout: npt.NDArray[np.float64],
    *,
    n_starts: int,
    first_start: int = 0,
    seed: int,
    method: str = "cg",
    precision: str = "rough",
    keep: int = 0,
    threads: int = 0,
) -> list[dict[str, Any]]: ...
def refine(
    problem: Problem,
    layouts: list[npt.NDArray[np.float64]],
    *,
    method: str = "cg",
    precision: str = "fine",
    threads: int = 0,
) -> list[dict[str, Any]]: ...
def optimise(
    problem: Problem,
    layout: npt.NDArray[np.float64],
    *,
    method: str = "cg",
    precision: str = "fine",
) -> dict[str, Any]: ...
def grid_test(
    problem: Problem, layout: npt.NDArray[np.float64], *, step: float = 0.1, threads: int = 0
) -> list[dict[str, Any]]: ...
def column_bases(
    titre_value: npt.NDArray[np.float64], titre_type: npt.NDArray[np.int8], minimum: float = 0.0
) -> npt.NDArray[np.float64]: ...
def start_seed(seed: int, index: int) -> int: ...
def resolve_threads(requested: int) -> int: ...
