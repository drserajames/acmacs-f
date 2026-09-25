"""Which optimiser the chain uses: workstream 6's core (`af.map.optimise`), or a stub.

Both return every start as a dict {layout, stress, dimensions, n_iterations, start_seed},
sorted by stress. The engine decides what to keep.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np


class Optimiser(Protocol):
    name: str

    def optimise(
        self, arrays: dict, n_starts: int, dim: int, seed: int, start_layout: np.ndarray | None
    ) -> list[dict]: ...

    def grid_test(self, layout: np.ndarray, arrays: dict) -> list[dict]: ...


class CoreOptimiser:
    """ae's chain maps: N starts (incremental: from the previous layout), then the trapped-point
    loop on the best (ae `chart-relax-grid` / `chart-relax-incremental --grid-test`)."""

    name = "af.map.optimise (alglib CG) + resolve_trapped"

    def __init__(self, threads: int = 0):
        self.threads = threads

    @staticmethod
    def _problem(arrays: dict):
        from af.map.optimise import MapProblem

        return MapProblem(
            titre_value=arrays["titre_value"],
            titre_type=arrays["titre_type"],
            column_bases=np.asarray(arrays["column_bases"], dtype=float),
            disconnected=arrays["disconnected"],
            dodgy_is_regular=bool(arrays.get("dodgy_is_regular", False)),
            weights=arrays.get("weights"),
            avidity_adjust=arrays.get("avidity_adjust"),
            unmovable=arrays.get("unmovable"),
        )

    def optimise(self, arrays, n_starts, dim, seed, start_layout):
        from af.map.optimise import relax, resolve_trapped

        problem = self._problem(arrays)
        result = relax(
            problem,
            n_starts=n_starts,
            seed=seed & (2**64 - 1),
            dimensions=dim,
            start_layout=start_layout,
            keep=None,
            threads=self.threads,
        )
        out = [
            {
                "layout": p.layout,
                "stress": p.stress,
                "dimensions": p.dimensions,
                "n_iterations": p.n_iterations,
                "start_seed": p.start_index,
                "resolved_rounds": 0,
            }
            for p in result.projections
        ]
        fixed = resolve_trapped(problem, out[0]["layout"], threads=self.threads)
        if fixed.moved:
            p = fixed.projection
            out[0] = {
                **out[0],
                "layout": p.layout,
                "stress": p.stress,
                "resolved_rounds": fixed.rounds,
                "resolved_moves": fixed.moved,
            }
            out.sort(key=lambda r: r["stress"])
        return out

    def grid_test(self, layout, arrays):
        from af.map.optimise import grid_test

        results = grid_test(self._problem(arrays), layout, threads=self.threads)
        return [
            {
                "point": int(g.point),
                "diagnosis": g.diagnosis,
                "distance": float(g.distance),
                "contribution_diff": float(g.stress_diff),
            }
            for g in results
            if g.diagnosis in ("trapped", "hemisphering")
        ]


class StubOptimiser:
    name = "stub-scipy-lbfgs"

    def optimise(self, arrays, n_starts, dim, seed, start_layout):
        from af.chain import stub_optimiser

        return stub_optimiser.optimise(
            arrays, n_starts=n_starts, dim=dim, seed=seed, start_layout=start_layout, keep=n_starts
        )

    def grid_test(self, layout, arrays):
        from af.chain import stub_optimiser

        return stub_optimiser.grid_test(layout, arrays)


def default_optimiser() -> Optimiser:
    try:
        import af.map.optimise  # noqa: F401
    except ImportError:
        return StubOptimiser()
    return CoreOptimiser()
