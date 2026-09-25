"""Which optimiser the chain uses: workstream 6's core (`af.map.optimise`), or a stand-in.

Every backend returns maps as dicts {layout, stress, dimensions, n_iterations, start_seed}
sorted by stress, and can run a map's starts in chunks (`relax_chunk`) and put the chunks
back together (`combine`) so that the result is the same however the starts were split
(start i always draws from its own seed; COORDINATION.md I1).
"""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np

MapResult = dict[str, Any]


class Optimiser(Protocol):
    name: str
    key: str  # the name `python -m af.chain.starts` knows it by

    def optimise(
        self, arrays: dict, n_starts: int, dim: int, seed: int, start_layout: np.ndarray | None
    ) -> list[MapResult]: ...

    def relax_chunk(
        self,
        arrays: dict,
        first_start: int,
        n_starts: int,
        dim: int,
        seed: int,
        start_layout: np.ndarray | None,
    ) -> list[MapResult]: ...

    def combine(
        self, arrays: dict, chunks: list[MapResult], incremental: bool
    ) -> list[MapResult]: ...

    def grid_test(self, layout: np.ndarray, arrays: dict) -> list[dict]: ...


class CoreOptimiser:
    """ae's chain maps: N starts (incremental: from the previous layout, rough, then the best 5
    fine), then the trapped-point loop on the best (ae `chart-relax-grid` /
    `chart-relax-incremental --grid-test`)."""

    name = "af.map.optimise (alglib CG) + resolve_trapped"
    key = "core"

    def __init__(self, threads: int = 0):
        self.threads = threads

    @staticmethod
    def _problem(arrays: dict) -> Any:
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

    @staticmethod
    def _as_dict(p: Any) -> MapResult:
        return {
            "layout": p.layout,
            "stress": p.stress,
            "dimensions": p.dimensions,
            "n_iterations": p.n_iterations,
            "start_seed": p.start_index,
            "rng_seed": p.start_seed,
            "termination": p.termination,
        }

    @staticmethod
    def _as_projection(r: MapResult) -> Any:
        from af.map.optimise import Projection

        return Projection(
            layout=r["layout"],
            stress=r["stress"],
            dimensions=r["dimensions"],
            n_iterations=r["n_iterations"],
            start_seed=r["rng_seed"],
            start_index=r["start_seed"],
            termination=r["termination"],
        )

    def optimise(self, arrays, n_starts, dim, seed, start_layout):
        from af.map.optimise import relax

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
        return self._resolve(problem, [self._as_dict(p) for p in result.projections])

    def relax_chunk(self, arrays, first_start, n_starts, dim, seed, start_layout):
        """One job's share of a map's starts. Incremental chunks stay rough: the fine stage
        must see the best starts of all chunks (see `combine`)."""
        from af.map.optimise import relax

        result = relax(
            self._problem(arrays),
            n_starts=n_starts,
            first_start=first_start,
            seed=seed & (2**64 - 1),
            dimensions=dim,
            start_layout=start_layout,
            precision="rough" if start_layout is not None else "fine",
            keep=None,
            threads=self.threads,
        )
        return [self._as_dict(p) for p in result.projections]

    def combine(self, arrays, chunks, incremental):
        from af.map.optimise import refine, sort_projections

        problem = self._problem(arrays)
        projections = sort_projections([self._as_projection(r) for r in chunks])
        if incremental:
            projections = refine(problem, projections, n_best=5, threads=self.threads)
        return self._resolve(problem, [self._as_dict(p) for p in projections])

    def _resolve(self, problem: Any, maps: list[MapResult]) -> list[MapResult]:
        from af.map.optimise import resolve_trapped

        fixed = resolve_trapped(problem, maps[0]["layout"], threads=self.threads)
        if fixed.moved:
            p = fixed.projection
            maps[0] = {
                **maps[0],
                "layout": p.layout,
                "stress": p.stress,
                "resolved_rounds": fixed.rounds,
                "resolved_moves": fixed.moved,
            }
            maps.sort(key=lambda r: r["stress"])
        return maps

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
    """Numpy stand-in (af.chain.stub_optimiser): for tests and machines without the core."""

    name = "stub-numpy-gradient-descent"
    key = "stub"

    def optimise(self, arrays, n_starts, dim, seed, start_layout):
        return self.relax_chunk(arrays, 0, n_starts, dim, seed, start_layout)

    def relax_chunk(self, arrays, first_start, n_starts, dim, seed, start_layout):
        from af.chain import stub_optimiser

        return stub_optimiser.optimise(
            arrays,
            n_starts=n_starts,
            dim=dim,
            seed=seed,
            start_layout=start_layout,
            keep=n_starts,
            first_start=first_start,
        )

    def combine(self, arrays, chunks, incremental):
        return sorted(chunks, key=lambda r: (r["stress"], r["start_seed"]))

    def grid_test(self, layout, arrays):
        from af.chain import stub_optimiser

        return stub_optimiser.grid_test(layout, arrays)


BACKENDS = {"core": CoreOptimiser, "stub": StubOptimiser}


def optimiser_by_key(key: str, threads: int = 0) -> Optimiser:
    if key == "core":
        return CoreOptimiser(threads=threads)
    if key == "stub":
        return StubOptimiser()
    raise ValueError(f"unknown optimiser {key!r}; known: {sorted(BACKENDS)}")


def default_optimiser() -> Optimiser:
    try:
        import af.map.optimise  # noqa: F401
    except ImportError:
        return StubOptimiser()
    return CoreOptimiser()
