"""Run one chunk of a map's starts: the unit of work a SLURM array task (or a local job) does.

    python -m af.chain.starts <problem.npz> <first_start> <n_starts> <result.npz>

The problem file holds the I1 arrays, the start layout (incremental maps), the seed, the
dimensions, the optimiser key and the arithmetic it was written for (`Optimiser.numerics`),
written by `write_problem`. The result file holds the chunk's maps and the arithmetic that made
them, read back by `read_result` and `result_numerics`. Chunks are combined by the optimiser's
`combine`, and the maps are the same however the starts were split (per-start seeds).

Why a file-based command rather than in-process threads: the same step then runs on a
laptop (af.run LocalRunner) and as a SLURM array (af.run SlurmRunner) unchanged.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Any

import numpy as np

from af.chain.backend import MapResult, optimiser_by_key

_ARRAYS = ("titre_value", "titre_type", "column_bases", "disconnected")
_OPTIONAL = ("weights", "avidity_adjust", "unmovable")
# how a map from scratch is relaxed (MapOptions.scratch_precision), carried to every chunk
_SETTINGS = ("scratch_precision",)
_SETTING_ORIGINAL = {"scratch_precision": "fine"}  # what a problem without the key means


def write_problem(
    path: Path,
    arrays: dict,
    *,
    seed: int,
    dimensions: int,
    optimiser: str,
    numerics: str | None,
    start_layout: np.ndarray | None,
) -> Path:
    data = _problem_data(
        arrays,
        seed=seed,
        dimensions=dimensions,
        optimiser=optimiser,
        numerics=numerics,
        start_layout=start_layout,
    )
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez(tmp, **data)
    tmp.replace(path)
    return path


def problem_digest(
    arrays: dict,
    *,
    seed: int,
    dimensions: int,
    optimiser: str,
    numerics: str | None,
    start_layout: np.ndarray | None,
) -> str:
    """sha256 of everything a chunk's maps depend on (the problem file's content, not its bytes:
    an .npz holds zip timestamps). Per-start seeding makes a chunk's maps a function of this and
    its start range alone, so a finished chunk of the same problem can be reused exactly.
    `numerics` is part of it: chunks made under another arithmetic are another problem's."""
    data = _problem_data(
        arrays,
        seed=seed,
        dimensions=dimensions,
        optimiser=optimiser,
        numerics=numerics,
        start_layout=start_layout,
    )
    h = hashlib.sha256()
    for key in sorted(data):
        a = np.asarray(data[key])
        h.update(f"{key}|{a.dtype.str}|{a.shape}|".encode())
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def _problem_data(
    arrays: dict,
    *,
    seed: int,
    dimensions: int,
    optimiser: str,
    numerics: str | None,
    start_layout: np.ndarray | None,
) -> dict[str, Any]:
    data: dict[str, Any] = {k: arrays[k] for k in _ARRAYS}
    data.update({k: arrays[k] for k in _OPTIONAL if arrays.get(k) is not None})
    data["dodgy_is_regular"] = np.bool_(arrays.get("dodgy_is_regular", False))
    # a setting at its original value is left out, as a problem written before it existed has
    # it: the file means the same, and its digest matches the older run's (chunks resume)
    data.update(
        {
            k: np.str_(arrays[k])
            for k in _SETTINGS
            if arrays.get(k) is not None and arrays[k] != _SETTING_ORIGINAL[k]
        }
    )
    data["seed"] = np.uint64(seed & (2**64 - 1))
    data["dimensions"] = np.int64(dimensions)
    data["optimiser"] = np.str_(optimiser)
    # an optimiser that names no arithmetic leaves the key out, as every problem written before
    # the key existed has it: such a problem's digest is what it always was
    if numerics is not None:
        data["numerics"] = np.str_(numerics)
    if start_layout is not None:
        data["start_layout"] = start_layout
    return data


def read_problem(path: Path) -> tuple[dict, int, int, str, str | None, np.ndarray | None]:
    """arrays, seed, dimensions, optimiser key, numerics (None: none named), start layout."""
    with np.load(path, allow_pickle=False) as f:
        arrays = {k: f[k] for k in (*_ARRAYS, *_OPTIONAL) if k in f}
        arrays["dodgy_is_regular"] = bool(f["dodgy_is_regular"])
        arrays.update({k: str(f[k]) for k in _SETTINGS if k in f})
        start = f.get("start_layout", None)
        numerics = str(f["numerics"]) if "numerics" in f else None
        return arrays, int(f["seed"]), int(f["dimensions"]), str(f["optimiser"]), numerics, start


def write_result(path: Path, maps: list[MapResult], numerics: str | None = None) -> Path:
    """`numerics` is the arithmetic that made these maps (None: none named, and no key, as a
    result written before the key existed). The driver reads it back (`result_numerics`) and
    refuses a chunk made under another: a task can run a build other than the driver's."""
    keys = ("stress", "dimensions", "n_iterations", "start_seed")
    data: dict[str, Any] = {"layouts": np.stack([m["layout"] for m in maps])}
    data.update({k: np.array([m[k] for m in maps]) for k in keys})
    if "rng_seed" in maps[0]:
        data["rng_seed"] = np.array([m["rng_seed"] for m in maps], dtype=np.uint64)
        data["termination"] = np.array([m["termination"] for m in maps])
    if "threads" in maps[0]:  # measured in this task: the driver cannot see the node's CPUs
        data["threads"] = np.array([m["threads"] for m in maps])
        data["cpus"] = np.array([m["cpus"] for m in maps])
    if numerics is not None:
        data["numerics"] = np.str_(numerics)
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez(tmp, **data)
    tmp.replace(path)
    return path


def finished(path: Path, first: int, count: int) -> bool:
    """Whether `path` holds this chunk's maps: readable, and exactly its starts (first ..
    first + count - 1). Results are written atomically, so a present file is whole unless
    something else broke it."""
    try:
        maps = read_result(path)
    except Exception:  # noqa: BLE001  (unreadable: rerun the chunk)
        return False
    return sorted(m["start_seed"] for m in maps) == list(range(first, first + count))


def result_numerics(path: Path) -> str | None:
    """The arithmetic a chunk's maps were made under (`write_result`); None when it names none.
    Reads that one entry, not the layouts."""
    with np.load(path, allow_pickle=False) as f:
        return str(f["numerics"]) if "numerics" in f else None


def read_result(path: Path) -> list[MapResult]:
    """The chunk's maps. Each array is read from the file ONCE and each map's layout is its own
    copy: indexing `f["layouts"][i]` per map decompressed the whole chunk again for every map, and
    each layout kept a view on its own full copy (2 Oct 2026: a 13,095-point merge_all driver held
    ~99 GB resident, ~300 GB virtual, for 2 GB of layouts, and spent hours reading and swapping)."""
    with np.load(path, allow_pickle=False) as f:
        data = {k: f[k] for k in f.files if k != "numerics"}
    n = len(data["stress"])
    maps = []
    for i in range(n):
        m: MapResult = {
            "layout": data["layouts"][i].copy(),
            "stress": float(data["stress"][i]),
            "dimensions": int(data["dimensions"][i]),
            "n_iterations": int(data["n_iterations"][i]),
            "start_seed": int(data["start_seed"][i]),
        }
        if "rng_seed" in data:
            m["rng_seed"] = int(data["rng_seed"][i])
            m["termination"] = int(data["termination"][i])
        if "threads" in data:
            m["threads"] = int(data["threads"][i])
            m["cpus"] = int(data["cpus"][i])
        maps.append(m)
    return maps


def run_chunk(
    problem: Path, first_start: int, n_starts: int, result: Path, threads: int = 0
) -> Path:
    arrays, seed, dim, key, numerics, start = read_problem(problem)
    opt = optimiser_by_key(key, threads=threads)
    if opt.numerics != numerics:
        # the driver and this task run different builds (an environment changed under a run):
        # the chunk's maps would not be the ones the problem's digest stands for
        raise RuntimeError(
            f"{problem}: written for numerics {numerics!r}, but the {key!r} optimiser of this"
            f" build has {opt.numerics!r} ({sys.executable}); run the chunk with the build that"
            " wrote the problem"
        )
    maps = opt.relax_chunk(arrays, first_start, n_starts, dim, seed, start)
    if len(maps) != n_starts:
        raise RuntimeError(f"{problem}: asked for {n_starts} starts, got {len(maps)}")
    return write_result(result, maps, numerics)


def main(argv: list[str]) -> int:
    if len(argv) not in (4, 5):
        print(__doc__, file=sys.stderr)
        return 2
    problem, first, n, result = Path(argv[0]), int(argv[1]), int(argv[2]), Path(argv[3])
    run_chunk(problem, first, n, result, threads=int(argv[4]) if len(argv) == 5 else 0)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
