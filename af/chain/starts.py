"""Run one chunk of a map's starts: the unit of work a SLURM array task (or a local job) does.

    python -m af.chain.starts <problem.npz> <first_start> <n_starts> <result.npz>

The problem file holds the I1 arrays, the start layout (incremental maps), the seed, the
dimensions and the optimiser key, written by `write_problem`. The result file holds the
chunk's maps, read back by `read_result`. Chunks are combined by the optimiser's
`combine`, and the maps are the same however the starts were split (per-start seeds).

Why a file-based command rather than in-process threads: the same step then runs on a
laptop (af.run LocalRunner) and as a SLURM array (af.run SlurmRunner) unchanged.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

from af.chain.backend import MapResult, optimiser_by_key

_ARRAYS = ("titre_value", "titre_type", "column_bases", "disconnected")
_OPTIONAL = ("weights", "avidity_adjust", "unmovable")


def write_problem(
    path: Path,
    arrays: dict,
    *,
    seed: int,
    dimensions: int,
    optimiser: str,
    start_layout: np.ndarray | None,
) -> Path:
    data: dict[str, Any] = {k: arrays[k] for k in _ARRAYS}
    data.update({k: arrays[k] for k in _OPTIONAL if arrays.get(k) is not None})
    data["dodgy_is_regular"] = np.bool_(arrays.get("dodgy_is_regular", False))
    data["seed"] = np.uint64(seed & (2**64 - 1))
    data["dimensions"] = np.int64(dimensions)
    data["optimiser"] = np.str_(optimiser)
    if start_layout is not None:
        data["start_layout"] = start_layout
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez(tmp, **data)
    tmp.replace(path)
    return path


def read_problem(path: Path) -> tuple[dict, int, int, str, np.ndarray | None]:
    with np.load(path, allow_pickle=False) as f:
        arrays = {k: f[k] for k in (*_ARRAYS, *_OPTIONAL) if k in f}
        arrays["dodgy_is_regular"] = bool(f["dodgy_is_regular"])
        start = f.get("start_layout", None)
        return arrays, int(f["seed"]), int(f["dimensions"]), str(f["optimiser"]), start


def write_result(path: Path, maps: list[MapResult]) -> Path:
    keys = ("stress", "dimensions", "n_iterations", "start_seed")
    data: dict[str, Any] = {"layouts": np.stack([m["layout"] for m in maps])}
    data.update({k: np.array([m[k] for m in maps]) for k in keys})
    if "rng_seed" in maps[0]:
        data["rng_seed"] = np.array([m["rng_seed"] for m in maps], dtype=np.uint64)
        data["termination"] = np.array([m["termination"] for m in maps])
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez(tmp, **data)
    tmp.replace(path)
    return path


def read_result(path: Path) -> list[MapResult]:
    with np.load(path, allow_pickle=False) as f:
        n = len(f["stress"])
        maps = []
        for i in range(n):
            m: MapResult = {
                "layout": f["layouts"][i],
                "stress": float(f["stress"][i]),
                "dimensions": int(f["dimensions"][i]),
                "n_iterations": int(f["n_iterations"][i]),
                "start_seed": int(f["start_seed"][i]),
            }
            if "rng_seed" in f:
                m["rng_seed"] = int(f["rng_seed"][i])
                m["termination"] = int(f["termination"][i])
            maps.append(m)
        return maps


def run_chunk(
    problem: Path, first_start: int, n_starts: int, result: Path, threads: int = 0
) -> Path:
    arrays, seed, dim, key, start = read_problem(problem)
    opt = optimiser_by_key(key, threads=threads)
    maps = opt.relax_chunk(arrays, first_start, n_starts, dim, seed, start)
    if len(maps) != n_starts:
        raise RuntimeError(f"{problem}: asked for {n_starts} starts, got {len(maps)}")
    return write_result(result, maps)


def main(argv: list[str]) -> int:
    if len(argv) not in (4, 5):
        print(__doc__, file=sys.stderr)
        return 2
    problem, first, n, result = Path(argv[0]), int(argv[1]), int(argv[2]), Path(argv[3])
    run_chunk(problem, first, n, result, threads=int(argv[4]) if len(argv) == 5 else 0)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
