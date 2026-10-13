#!/usr/bin/env python3
"""Does this build's ORIGINAL optimiser give the same bits as an earlier release on this machine?

The original arithmetic (``MapProblem(numerics="original")``, af.map._core_original) is kept so that
maps made before the arithmetic contract stay reproducible where they were built. That holds only
if it really is the arithmetic those releases had: this tool runs the same problems under a
reference release's python and under the python running it, and compares every bit.

    <new python> tools/check-original-arithmetic.py --reference <release>/env/bin/python
    ... --chart NAME=chart.ace:STARTS [--chart ...] [--trapped-loop] [--threads N]

With no --chart the problem is the integer-only invented table of tests/map/test_numerics.py (no
data, no network), run as a chain makes a scratch map: 40 rough starts, the best 5 refined, the
trapped loop on the first. With --chart, real maps: STARTS rough starts, the best 5 refined, and
the trapped loop on the first only with --trapped-loop (it is slow on large maps). A chart's
arrays are made once, by the python running this, and both sides run those same arrays: what is
compared is the optimiser, not how a chart becomes arrays.

The reference may be a release from before the contract (its only optimiser is the original) or a
later one (its original core is used). The result does not depend on the thread count.

Exit status: 0 every bit the same; 1 compared, and some bit differs; 2 could not compare (the
reference would not run, a build without the original, a missing input). One summary line on
stdout; one line per problem before it, with its iterations and layout hash.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

SEED = 20261013
SAME, DIFFERENT, COULD_NOT_COMPARE = 0, 1, 2


class CannotCompare(Exception):
    pass


def sha(array: Any) -> str:
    import numpy as np

    return hashlib.sha256(np.ascontiguousarray(array, dtype="<f8").tobytes()).hexdigest()


def original_options(opt: Any) -> dict[str, Any]:
    """How to ask this af for the original arithmetic: by name where it is offered; nothing to
    say before the contract, when it was the only one."""
    offered = getattr(opt, "NUMERICS_OFFERED", None)
    if offered is not None:
        if "original" not in offered:
            raise CannotCompare(f"this build offers {offered}, not the original arithmetic")
        return {"numerics": "original"}
    if hasattr(opt, "NUMERICS"):
        raise CannotCompare("this build has an arithmetic contract but no original core")
    return {}


def integer_problem(options: dict[str, Any]) -> Any:
    """The invented table of tests/map/test_numerics.py, found beside this file (one copy)."""
    path = Path(__file__).resolve().parents[1] / "tests" / "map" / "test_numerics.py"
    if not path.is_file():
        raise CannotCompare(f"{path} is missing: run from a source tree, or give --chart")
    spec = importlib.util.spec_from_file_location("af_numerics_table", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.integer_table(**options)


def chart_arrays(path: str, out: Path) -> None:
    """A chart's optimiser arrays, written once by THIS python. Both sides then run these same
    arrays, so the comparison is of the optimiser and not of how a chart becomes arrays (which
    has changed on its own: Titre.logged is a correctly rounded log2 since October 2026)."""
    import numpy as np

    from af.chart.ace import read_chart

    chart = read_chart(path)
    arrays = chart.optimiser_arrays(projection=chart.best())
    kept = {
        "titre_value": np.asarray(arrays["titre_value"], dtype=float),
        "titre_type": np.asarray(arrays["titre_type"]),
        "column_bases": np.asarray(arrays["column_bases"], dtype=float),
        "disconnected": np.asarray(arrays["disconnected"]),
        "dodgy_is_regular": np.asarray(bool(arrays.get("dodgy_is_regular", False))),
    }
    if arrays.get("avidity_adjust") is not None:
        kept["avidity_adjust"] = np.asarray(arrays["avidity_adjust"], dtype=float)
    np.savez(out, **kept)


def arrays_problem(path: str, options: dict[str, Any]) -> Any:
    import numpy as np

    from af.map.optimise import MapProblem

    with np.load(path) as z:
        return MapProblem(
            titre_value=z["titre_value"],
            titre_type=z["titre_type"],
            column_bases=z["column_bases"],
            disconnected=z["disconnected"],
            dodgy_is_regular=bool(z["dodgy_is_regular"]),
            avidity_adjust=z["avidity_adjust"] if "avidity_adjust" in z.files else None,
            **options,
        )


def measure(problem: Any, starts: int, threads: int, trapped_loop: bool) -> dict[str, Any]:
    import numpy as np

    from af.map import optimise as opt

    rough = opt.relax(problem, n_starts=starts, seed=SEED, precision="rough", threads=threads)
    refined = opt.refine(problem, rough.projections, n_best=5, threads=threads)
    result = {
        "starts": starts,
        "iterations": int(sum(p.n_iterations for p in refined)),
        "rough stresses": sha(np.array([p.stress for p in rough.projections])),
        "refined layouts": sha(np.array([p.layout for p in refined[:5]])),
        "best refined stress": float(refined[0].stress).hex(),
        "layout": sha(refined[0].layout),
    }
    if trapped_loop:
        resolved = opt.resolve_trapped(problem, refined[0].layout, threads=threads)
        result["trapped loop"] = f"{resolved.rounds} rounds, {resolved.moved} moves"
        result["resolved stress"] = float(resolved.projection.stress).hex()
        result["layout"] = sha(resolved.projection.layout)
    return result


def emit(args: argparse.Namespace) -> dict[str, Any]:
    from af.map import optimise as opt

    options = original_options(opt)
    core = getattr(opt, "_core_original", opt._core)  # the module that will do the work
    out: dict[str, Any] = {"python": sys.executable, "core": core.__file__, "problems": {}}
    if args.problem:
        for spec in args.problem:
            name, rest = spec.split("=", 1)
            path, starts = rest.rsplit(":", 1)
            problem = arrays_problem(path, options)
            out["problems"][name] = measure(problem, int(starts), args.threads, args.trapped_loop)
    else:
        out["problems"]["invented table"] = measure(
            integer_problem(options), 40, args.threads, True
        )
    return out


def reference(args: argparse.Namespace) -> dict[str, Any]:
    # -I: isolated. The reference must import its OWN af, whatever PYTHONPATH or directory the
    # caller has: a release's python refuses an af from elsewhere, and so it should.
    command = [
        args.reference,
        "-I",
        str(Path(__file__).resolve()),
        "--emit",
        "--threads",
        str(args.threads),
    ]
    for spec in args.problem or []:
        command += ["--problem", spec]
    if args.trapped_loop:
        command.append("--trapped-loop")
    try:
        done = subprocess.run(command, capture_output=True, text=True, check=False)
    except OSError as err:
        raise CannotCompare(f"the reference python would not start: {err}") from err
    lines = [line for line in done.stdout.splitlines() if line.startswith("JSON ")]
    if done.returncode != 0 or not lines:
        tail = (done.stderr or done.stdout).strip().splitlines()[-1:] or ["no output"]
        raise CannotCompare(f"the reference run failed (exit {done.returncode}): {tail[0]}")
    return json.loads(lines[-1][5:])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--reference", help="python of the release to compare with")
    parser.add_argument("--chart", action="append", metavar="NAME=PATH:STARTS")
    parser.add_argument("--trapped-loop", action="store_true")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--emit", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--problem", action="append", help=argparse.SUPPRESS
    )  # NAME=ARRAYS.npz:STARTS
    args = parser.parse_args(argv)
    try:
        if args.emit:
            print("JSON " + json.dumps(emit(args)))
            return SAME
        if not args.reference:
            parser.error("--reference is required")
        with tempfile.TemporaryDirectory() as scratch:
            args.problem = []
            for number, spec in enumerate(args.chart or []):
                name, rest = spec.split("=", 1)
                path, starts = rest.rsplit(":", 1)
                arrays = Path(scratch) / f"problem{number}.npz"
                chart_arrays(path, arrays)
                args.problem.append(f"{name}={arrays}:{starts}")
            mine, theirs = emit(args), reference(args)
    except CannotCompare as err:
        print(f"original arithmetic: COULD NOT COMPARE: {err}")
        return COULD_NOT_COMPARE
    different = []
    for name, a in mine["problems"].items():
        b = theirs["problems"][name]
        same = a == b
        if not same:
            different.append(name)
        loop = f"; trapped loop {a['trapped loop']}" if "trapped loop" in a else ""
        print(
            f"{name}: {a['starts']} starts, {a['iterations']} iterations{loop}; layout sha256 "
            f"{a['layout'][:16]}; reference {b['iterations']} iterations, {b['layout'][:16]}: "
            f"{'the same bits' if same else 'DIFFERENT'}"
        )
    where = f"this build {mine['core']} against {theirs['core']}"
    if different:
        print(f"original arithmetic: DIFFERENT bits on {', '.join(different)} ({where})")
        return DIFFERENT
    print(f"original arithmetic: the same bits on {len(mine['problems'])} problem(s) ({where})")
    return SAME


if __name__ == "__main__":
    sys.exit(main())
