"""The optimiser's arithmetic is the same on every machine (cpp/optimiser/numerics.hh).

Sarah, 13 Oct 2026: "the algorithm should be reproducible regardless of machine." A map used to
depend on where it was built: arm64 fuses a*b+c into one rounding, x86_64 does not, and the system
exp differs in the last bit between libms; the minimiser turns one bit into another local minimum.

The layout tests here are the proof that it no longer does. They build a table with integer
arithmetic only, run the optimiser as a chain does, and compare the sha256 of the raw coordinates
with hashes recorded in this file. CI runs them on Linux x86_64 and macOS arm64, so one committed
hash passing on both IS the cross-machine check. If a hash changes, the arithmetic changed:
`NUMERICS` must change with it, and every published map will differ.

A release build runs this file as a gate, so it needs no data and no network, uses fixed thread
counts (2; and 1 against 5 to show the count does not matter), takes about a second, and has no
skip: a skipped check would read as a passed one.

The original arithmetic is kept as an option (af.map._core_original). Its bits belong to one
machine, so the tests here only check that it is offered, repeats itself, and cannot be mixed with
the contract's core; tools/check-original-arithmetic.py compares it with an earlier release.
"""

from __future__ import annotations

import hashlib
import math
import random
import struct
import sys
from typing import Any

import numpy as np
import pytest

from af.map import optimise as opt
from af.map.optimise import MapProblem, TitreType

core = opt._core


def bits(x: float) -> str:
    return struct.pack(">d", x).hex()


def ulps_apart(a: float, b: float) -> int:
    ia, ib = (struct.unpack("<q", struct.pack("<d", v))[0] for v in (a, b))
    return abs(ia - ib)


# ---------------------------------------------------------------------------------------
# the contract


def test_the_build_keeps_the_arithmetic_contract():
    assert opt.NUMERICS == core.NUMERICS == "strict-fp-1"
    assert core.contracts_multiply_add() is False


EXP_BITS = {
    0.0: "3ff0000000000000",
    1.0: "4005bf0a8b145769",
    -1.0: "3fd78b56362cef37",
    0.5: "3ffa61298e1e069c",
    -37.25: "3c93278bcd70e981",
    12.345678901234567: "410c12618f0f1566",
    100.0: "48f3494a9b171bf4",
    -700.0: "00d14f2b0fb9307f",
    700.0: "7f0d945df4f8ec8e",
    1e-9: "3ff000000044b830",
    -0.0027: "3fefe9e953ac0bf4",
    36.9: "4342d61a967da146",
}
LOG2_BITS = {
    1.0: "0000000000000000",
    2.0: "3ff0000000000000",
    0.5: "bff0000000000000",
    3.0: "3ff95c01a39fbd69",
    10.0: "400a934f0979a371",
    0.7071: "bfe0001d03eb60d8",
    1.4142135623730951: "3fe0000000000001",
    1e-300: "c08f24a09f1a8b89",
    5e-324: "c090c80000000000",
    123456.789: "4030e9e4bf2a1cb2",
}


def test_portable_exp_and_log2_give_their_recorded_bits():
    """Recorded on macOS arm64; the same bits are required everywhere."""
    assert {x: bits(core.portable_exp(x)) for x in EXP_BITS} == EXP_BITS
    assert {x: bits(core.portable_log2(x)) for x in LOG2_BITS} == LOG2_BITS


def test_portable_log2_is_exact_for_powers_of_two():
    # a 10 x 2^k titre and a 2^k avidity factor must give whole numbers, as with any libm
    for k in range(-1074, 1024):
        assert core.portable_log2(math.ldexp(1.0, k)) == k


def test_portable_functions_at_the_edges():
    assert core.portable_exp(math.inf) == math.inf
    assert core.portable_exp(-math.inf) == 0.0
    assert math.isnan(core.portable_exp(math.nan))
    assert core.portable_exp(710.0) == math.inf and core.portable_exp(-746.0) == 0.0
    assert core.portable_log2(0.0) == -math.inf
    assert math.isnan(core.portable_log2(-1.0))
    assert core.portable_log2(math.inf) == math.inf


@pytest.mark.parametrize(
    ("name", "low", "high", "limit"),
    [("exp", -40.0, 40.0, 2), ("exp", -700.0, 700.0, 2), ("log2", 0.25, 4.0, 3)],
)
def test_portable_functions_stay_close_to_the_system_ones(name, low, high, limit):
    """They replace libm for sameness, not accuracy, and must not drift from it.

    Measured on macOS arm64: exp over 4,000,000 arguments in each of [-40, 40], [-700, 700],
    [-1, 1] and [-400, 40] differs from the system exp by 0 ulp on 74% and by 1 ulp on the rest,
    never more; log2 over 400,000 arguments by 0 (86%), 1 (14%) or 2 ulp (5 of 400,000). The
    limits here are one ulp wider, because the system functions themselves differ by an ulp
    between machines.
    """
    rng = random.Random(1)
    mine, system = (
        (core.portable_exp, math.exp) if name == "exp" else (core.portable_log2, math.log2)
    )
    worst = max(
        ulps_apart(mine(x), system(x)) for x in (rng.uniform(low, high) for _ in range(20_000))
    )
    assert worst <= limit


# ---------------------------------------------------------------------------------------
# the same layout on every machine


def integer_table(
    n_antigens: int = 90, n_sera: int = 14, seed: int = 20261013, *, numerics: str | None = None
) -> MapProblem:
    """A noisy titre table made with integer arithmetic only.

    Positions are whole eighths, column bases and logged titres whole numbers, the random numbers
    a 64-bit linear congruential generator: nothing here goes through libm or numpy's random
    module, so the PROBLEM is the same bits on every machine and only the optimiser is on trial.
    """
    state = seed

    def draw(n: int) -> int:
        nonlocal state
        state = (state * 6364136223846793005 + 1442695040888963407) % 2**64
        return (state >> 33) % n

    points = [(draw(64) / 8.0 - 4.0, draw(64) / 8.0 - 4.0) for _ in range(n_antigens + n_sera)]
    colbase = [float(6 + draw(4)) for _ in range(n_sera)]
    value = np.full((n_antigens, n_sera), np.nan)
    kind = np.full((n_antigens, n_sera), TitreType.MISSING, dtype=np.int8)
    for ag in range(n_antigens):
        for sr in range(n_sera):
            dx = points[ag][0] - points[n_antigens + sr][0]
            dy = points[ag][1] - points[n_antigens + sr][1]
            distance = math.sqrt(dx * dx + dy * dy)  # sqrt is correctly rounded everywhere
            logged = (
                colbase[sr] - math.floor(distance + 0.5) + (draw(5) - 2)
            )  # +-2 dilutions of noise
            if draw(8) == 0:
                continue  # missing
            if logged < 0:
                value[ag, sr], kind[ag, sr] = 0.0, TitreType.LESS_THAN
            else:
                value[ag, sr], kind[ag, sr] = float(min(logged, 10.0)), TitreType.REGULAR
    # numerics left unsaid is the build's default; tools/check-original-arithmetic.py uses this
    # table too, also with releases from before MapProblem had the argument
    options: dict[str, Any] = {} if numerics is None else {"numerics": numerics}
    return MapProblem(
        titre_value=value,
        titre_type=kind,
        column_bases=np.array(colbase),
        disconnected=np.zeros(n_antigens + n_sera, dtype=bool),
        dodgy_is_regular=False,
        **options,
    )


def layout_hash(layout: np.ndarray) -> str:
    """sha256 of the raw coordinates: before any orientation, which goes through LAPACK."""
    assert sys.byteorder == "little"
    return hashlib.sha256(np.ascontiguousarray(layout, dtype="<f8").tobytes()).hexdigest()


def test_the_problem_itself_is_the_same_everywhere():
    problem = integer_table()
    blob = (
        np.nan_to_num(problem.titre_value, nan=-1.0).astype("<f8").tobytes()
        + np.asarray(problem.titre_type).astype("<i1").tobytes()
        + np.asarray(problem.column_bases).astype("<f8").tobytes()
    )
    assert hashlib.sha256(blob).hexdigest() == PROBLEM_HASH
    assert int((np.asarray(problem.titre_type) == TitreType.LESS_THAN).sum()) > 50  # exp is used


THREADS = 2  # fixed, not the machine's: a release gate runs this on 14, 16 and 32+ cores


def chain_like_map(problem: MapProblem, *, threads: int = THREADS):
    """As a chain makes a scratch map: rough starts, the best 5 refined, trapped points resolved."""
    rough = opt.relax(problem, n_starts=40, seed=20261013, precision="rough", threads=threads)
    refined = opt.refine(problem, rough.projections, n_best=5, threads=threads)
    resolved = opt.resolve_trapped(problem, refined[0].layout, threads=threads)
    return rough, refined, resolved


def test_a_scratch_map_is_the_same_on_every_machine():
    problem = integer_table()
    rough, refined, resolved = chain_like_map(problem)
    assert resolved.moved > 0  # the trapped loop really moved points: that path is on trial too
    got = {
        "rough stresses": hashlib.sha256(
            np.array([p.stress for p in rough.projections], dtype="<f8").tobytes()
        ).hexdigest(),
        "refined #0": layout_hash(refined[0].layout),
        "refined #4": layout_hash(refined[4].layout),
        "resolved": layout_hash(resolved.projection.layout),
        "resolved stress": bits(resolved.projection.stress),
    }
    assert got == SCRATCH


def test_the_hashes_do_not_depend_on_the_thread_count():
    problem = integer_table()
    one, many = chain_like_map(problem, threads=1), chain_like_map(problem, threads=5)
    assert layout_hash(one[2].projection.layout) == layout_hash(many[2].projection.layout)
    assert [p.stress for p in one[0].projections] == [p.stress for p in many[0].projections]


def test_an_incremental_map_is_the_same_on_every_machine():
    problem = integer_table()
    start = np.array(chain_like_map(problem)[1][0].layout)
    start[::7] = np.nan  # every seventh point is new: placed at random, then all relaxed
    result = opt.relax(problem, n_starts=20, seed=7, start_layout=start, threads=THREADS)
    assert layout_hash(result.best.layout) == INCREMENTAL


def test_annealed_and_lbfgs_maps_are_the_same_on_every_machine():
    problem = integer_table()
    annealed = opt.relax(
        problem, n_starts=10, seed=3, dimension_annealing=True, threads=THREADS
    )  # with PCA
    lbfgs = opt.relax(problem, n_starts=10, seed=3, method="lbfgs", threads=THREADS)
    got = {"annealed": layout_hash(annealed.best.layout), "lbfgs": layout_hash(lbfgs.best.layout)}
    assert got == OTHER


def test_ties_between_starts_are_broken_by_start_number():
    # the published map must not depend on which thread finished first, or on a sort's whim
    problem = integer_table()
    rough = opt.relax(problem, n_starts=6, seed=1, precision="rough", threads=THREADS).projections
    twins = [opt.Projection(p.layout, 100.0, 2, 0, p.start_seed, p.start_index, 0) for p in rough]
    ordered = opt.sort_projections(list(reversed(twins)))
    assert [p.start_index for p in ordered] == sorted(p.start_index for p in twins)


# ---------------------------------------------------------------------------------------
# the original arithmetic, kept as an option


def test_both_arithmetics_are_offered_and_the_default_is_the_contract():
    """Sarah, 13 Oct 2026: "please keep the original optimiser (non-reproducible on a different
    machine) as an option". It is chosen by name, per problem, and never by default."""
    assert opt.NUMERICS_OFFERED == ("strict-fp-1", "original")
    assert integer_table().numerics == opt.NUMERICS
    assert opt._core_original.NUMERICS is None  # no name: its bits belong to one machine
    with pytest.raises(ValueError, match="offers"):
        integer_table(numerics="fast")  # an unknown name is an error, not a fallback


def test_a_problem_is_run_by_the_core_it_was_made_for():
    strict, original = integer_table(), integer_table(numerics=opt.ORIGINAL)
    assert type(strict._core_problem).__module__ == "af.map._core"
    assert type(original._core_problem).__module__ == "af.map._core_original"
    # an object of one core handed to the other is refused, never quietly run under the wrong one
    with pytest.raises(TypeError):
        opt._core.relax(original._core_problem, dimensions=2, n_starts=1, seed=1)
    with pytest.raises(TypeError):
        opt._core_original.relax(strict._core_problem, dimensions=2, n_starts=1, seed=1)
    with pytest.raises(TypeError):
        opt._core.grid_test(
            original._core_problem, np.zeros((strict.n_points, 2)), step=0.1, threads=1
        )


def test_the_original_repeats_itself_on_one_machine():
    """Its bits are this machine's, so none is recorded here (tools/check-original-arithmetic.py
    compares them with an earlier release on the same machine). Only: the same bits every time,
    at any thread count."""
    problem = integer_table(numerics=opt.ORIGINAL)
    first, again, more_threads = (
        chain_like_map(problem),
        chain_like_map(problem),
        chain_like_map(problem, threads=5),
    )
    hashes = {layout_hash(run[2].projection.layout) for run in (first, again, more_threads)}
    assert len(hashes) == 1
    assert [p.stress for p in first[0].projections] == [
        p.stress for p in more_threads[0].projections
    ]


def test_the_contract_holds_with_the_original_loaded_and_used():
    # the two cores share a process, each with its own alglib: neither may reach the other's
    original = chain_like_map(integer_table(numerics=opt.ORIGINAL))
    strict = chain_like_map(integer_table())
    assert layout_hash(strict[2].projection.layout) == SCRATCH["resolved"]
    # the same problem under both: near-equal minima, whatever the last bits
    assert original[2].projection.stress == pytest.approx(strict[2].projection.stress, rel=0.05)


# Recorded with NUMERICS "strict-fp-1" on macOS arm64 (Apple clang). The same values are required on
# every platform; see the module docstring before changing one.
PROBLEM_HASH = "1de121d83411b9083fff2704eb7fcc6452b722cd7fe1c14ba9180a3c729d313c"
SCRATCH = {
    "rough stresses": "e7f51dba92e6d651262b8c36c951b15abbcb67f9ba987b60bd0c431040843d40",
    "refined #0": "da6a2e8c56e102adbad924bd4db61f183c222d1a2581f9663acaa53921cc3f42",
    "refined #4": "bb0bf7853a864a3e9378bf49c3ae4ccc5be87311ed99e1994bbd055e335c433b",
    "resolved": "3a5263eac14dd1f892536120bdcd49eca8dcf2487985d5f97cb1a3f4262c0031",
    "resolved stress": "4099dcb1d11f3830",
}
INCREMENTAL = "e2f51a2d2080adf1a21555052d8a813fd5a6c43838d4e5906f8efd694ca54cd5"
OTHER = {
    "annealed": "bd8e42caae529680c1ce8a7c5ff0bd07b47539aad52485ba28b777ffa95a2d6c",
    "lbfgs": "8b54c2f64f7a68cef4a761e887d2ab9060d73072278599e324a3d3cf8cb41d72",
}
