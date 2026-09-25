"""Tests of the optimiser core on synthetic tables (generated here; no real data)."""

from __future__ import annotations

import dataclasses
import os
import subprocess
import sys
import tempfile
from typing import Any

import numpy as np
import pytest

from af.map import optimise as opt
from af.map.optimise import MapProblem, TitreType

# ---------------------------------------------------------------------------------------
# synthetic tables


def synthetic_table(
    n_antigens: int = 30,
    n_sera: int = 8,
    *,
    seed: int = 1,
    noise: float = 0.0,
    missing: float = 0.1,
    thresholds: bool = True,
) -> tuple[MapProblem, np.ndarray]:
    """A table made from known 2-D positions, so the right answer is known.

    Titres are log2(titre/10) = column basis - map distance (+ noise), rounded to whole
    dilutions as a lab reports them; below 0 they become "<10" and above 10 ">10240".
    The column bases given are the true ones (as if forced).
    Returns the problem and the true layout.
    """
    rng = np.random.default_rng(seed)
    truth = rng.uniform(-4.0, 4.0, size=(n_antigens + n_sera, 2))
    ag, sr = truth[:n_antigens], truth[n_antigens:]
    distance = np.linalg.norm(ag[:, None, :] - sr[None, :, :], axis=2)
    colbase = rng.uniform(6.0, 9.0, size=n_sera)
    logged = np.round(colbase[None, :] - distance + rng.normal(0.0, noise, distance.shape))
    titre_type = np.full(logged.shape, TitreType.REGULAR, dtype=np.int8)
    if thresholds:
        titre_type[logged < 0] = TitreType.LESS_THAN
        logged[logged < 0] = 0.0
        titre_type[logged > 10] = TitreType.MORE_THAN
        logged[logged > 10] = 10.0
    drop = rng.random(logged.shape) < missing
    titre_type[drop] = TitreType.MISSING
    logged[drop] = np.nan
    problem = MapProblem(
        titre_value=logged,
        titre_type=titre_type,
        column_bases=colbase,  # the true ones, so the true layout fits
        disconnected=np.zeros(n_antigens + n_sera, dtype=bool),
        dodgy_is_regular=False,
    )
    return problem, truth


def with_changes(problem: MapProblem, **changes: Any) -> MapProblem:
    """A copy of `problem` with some inputs replaced (re-validated by the core)."""
    return dataclasses.replace(problem, **changes)


def procrustes_rmsd(reference: np.ndarray, layout: np.ndarray) -> float:
    """RMSD after the best rotation/reflection + translation (no scaling), NaN rows skipped."""
    rows = ~(np.isnan(reference).any(axis=1) | np.isnan(layout).any(axis=1))
    a = reference[rows] - reference[rows].mean(axis=0)
    b = layout[rows] - layout[rows].mean(axis=0)
    u, _, vt = np.linalg.svd(b.T @ a)
    return float(np.sqrt(np.mean(np.sum((b @ (u @ vt) - a) ** 2, axis=1))))


def reference_stress(problem: MapProblem, layout: np.ndarray) -> float:
    """The stress formula written out independently of the C++ code."""
    n_ag = problem.n_antigens
    adjust = (
        np.zeros(problem.n_points)
        if problem.avidity_adjust is None
        else np.log2(problem.avidity_adjust)
    )
    weights = np.ones(problem.titre_value.shape) if problem.weights is None else problem.weights
    total = 0.0
    for ag in range(n_ag):
        for sr in range(problem.n_sera):
            kind = problem.titre_type[ag, sr]
            if problem.disconnected[ag] or problem.disconnected[n_ag + sr]:
                continue
            if kind == TitreType.DODGY and problem.dodgy_is_regular:
                kind = TitreType.REGULAR
            if kind not in (TitreType.REGULAR, TitreType.LESS_THAN):
                continue
            target = max(
                0.0,
                problem.column_bases[sr]
                - problem.titre_value[ag, sr]
                - adjust[ag]
                - adjust[n_ag + sr],
            )
            dist = np.linalg.norm(layout[ag] - layout[n_ag + sr])
            if kind == TitreType.REGULAR:
                total += weights[ag, sr] * (target - dist) ** 2
            else:
                diff = target - dist + 1.0
                total += weights[ag, sr] * diff**2 / (1.0 + np.exp(-10.0 * diff))
    return total


def numeric_gradient(problem: MapProblem, layout: np.ndarray, h: float = 1e-6) -> np.ndarray:
    result = np.zeros_like(layout)
    for index in np.ndindex(layout.shape):
        plus, minus = layout.copy(), layout.copy()
        plus[index] += h
        minus[index] -= h
        result[index] = (opt.stress(problem, plus) - opt.stress(problem, minus)) / (2.0 * h)
    return result


# ---------------------------------------------------------------------------------------
# stress and gradient


def test_stress_matches_the_written_out_formula():
    problem, truth = synthetic_table(noise=0.7)
    layout = truth + np.random.default_rng(3).normal(0.0, 1.0, truth.shape)
    assert opt.stress(problem, layout) == pytest.approx(
        reference_stress(problem, layout), rel=1e-12
    )


def test_true_layout_of_a_noise_free_table_has_low_stress():
    problem, truth = synthetic_table(noise=0.0, thresholds=False)
    # rounding to whole dilutions is the only error: each term is at most 0.5^2
    n_regular, _ = problem._core_problem.n_table_distances()
    assert opt.stress(problem, truth) <= 0.25 * n_regular


@pytest.mark.parametrize(
    "variant",
    ["plain", "weights", "avidity", "dodgy_regular", "disconnected"],
)
def test_analytic_gradient_matches_finite_differences(variant):
    problem, truth = synthetic_table(n_antigens=12, n_sera=5, noise=0.8, seed=7)
    rng = np.random.default_rng(11)
    if variant == "weights":
        problem = with_changes(problem, weights=rng.uniform(0.2, 2.0, problem.titre_value.shape))
    elif variant == "avidity":
        problem = with_changes(problem, avidity_adjust=rng.uniform(0.5, 2.0, problem.n_points))
    elif variant == "dodgy_regular":
        types = problem.titre_type.copy()
        regular = np.argwhere(types == TitreType.REGULAR)[:6]
        types[tuple(regular.T)] = TitreType.DODGY
        problem = with_changes(problem, titre_type=types, dodgy_is_regular=True)
    elif variant == "disconnected":
        disconnected = np.zeros(problem.n_points, dtype=bool)
        disconnected[[2, 13]] = True
        problem = with_changes(problem, disconnected=disconnected)
    layout = truth + rng.normal(0.0, 1.5, truth.shape)
    analytic = opt.gradient(problem, layout)
    numeric = numeric_gradient(problem, layout)
    assert np.max(np.abs(analytic - numeric)) < 1e-5 * max(1.0, np.max(np.abs(numeric)))
    assert opt.stress(problem, layout) == pytest.approx(
        reference_stress(problem, layout), rel=1e-12
    )


def test_less_than_gradient_across_the_sigmoid():
    """One "<" titre, scanned through the region where the sigmoid switches on."""
    value = np.array([[2.0]])
    kind = np.array([[TitreType.LESS_THAN]], dtype=np.int8)
    problem = MapProblem(
        value, kind, np.array([5.0]), np.zeros(2, dtype=bool), dodgy_is_regular=False
    )
    for dist in np.linspace(0.1, 6.0, 40):  # target 3, +1: switch near 4
        layout = np.array([[0.0, 0.0], [dist, 0.0]])
        analytic = opt.gradient(problem, layout)
        numeric = numeric_gradient(problem, layout, h=1e-7)
        assert np.max(np.abs(analytic - numeric)) < 1e-6


def test_unmovable_points_get_zero_gradient():
    problem, truth = synthetic_table(n_antigens=10, n_sera=4)
    unmovable = np.zeros(problem.n_points, dtype=bool)
    unmovable[[0, 11]] = True
    grad = opt.gradient(with_changes(problem, unmovable=unmovable), truth + 0.5)
    assert np.all(grad[[0, 11]] == 0.0)
    assert np.any(grad[1] != 0.0)


# ---------------------------------------------------------------------------------------
# relax


def test_relax_recovers_the_true_map():
    problem, truth = synthetic_table(noise=0.0, thresholds=False, missing=0.0)
    result = opt.relax(problem, n_starts=20, seed=42)
    best = result.best
    assert best.stress == pytest.approx(opt.stress(problem, best.layout), rel=1e-9)
    assert best.stress <= opt.stress(problem, truth) + 1e-6
    assert procrustes_rmsd(truth, best.layout) < 0.3
    stresses = [p.stress for p in result.projections]
    assert stresses == sorted(stresses)
    np.testing.assert_array_equal(result.column_bases, problem.column_bases)


def test_seeded_relax_is_independent_of_thread_count():
    problem, _ = synthetic_table(noise=0.8, seed=5)
    one = opt.relax(problem, n_starts=12, seed=2026, threads=1)
    many = opt.relax(problem, n_starts=12, seed=2026, threads=4)
    assert [p.start_index for p in one.projections] == [p.start_index for p in many.projections]
    for a, b in zip(one.projections, many.projections, strict=True):
        np.testing.assert_array_equal(a.layout, b.layout)
        assert a.stress == b.stress
        assert a.start_seed == b.start_seed == opt._core.start_seed(2026, a.start_index)


def test_different_seeds_give_different_starts():
    problem, _ = synthetic_table(noise=0.8, seed=5)
    a = opt.relax(problem, n_starts=3, seed=1)
    b = opt.relax(problem, n_starts=3, seed=2)
    assert not np.array_equal(a.projections[0].layout, b.projections[0].layout)


def test_keep_and_lbfgs_and_annealing():
    """A table with several basins. Measured (60 starts, seed 3): best CG 25.614,
    L-BFGS 26.166, 5-D annealing 25.337 (reached by most of its starts)."""
    problem, truth = synthetic_table(noise=0.3, seed=9)
    cg = opt.relax(problem, n_starts=60, seed=3)
    lbfgs = opt.relax(problem, n_starts=60, seed=3, method="lbfgs", keep=4)
    annealed = opt.relax(problem, n_starts=60, seed=3, dimension_annealing=True)
    assert len(lbfgs.projections) == 4
    assert annealed.best.layout.shape == truth.shape
    assert annealed.best.stress <= cg.best.stress
    for result in (cg, lbfgs):
        assert result.best.stress == pytest.approx(annealed.best.stress, rel=0.05)
    assert all(p.stress < opt.stress(problem, truth) for p in (cg.best, lbfgs.best, annealed.best))


def test_disconnected_points_come_back_as_nan():
    problem, _ = synthetic_table(n_antigens=15, n_sera=5)
    disconnected = np.zeros(problem.n_points, dtype=bool)
    disconnected[[3, 16]] = True
    result = opt.relax(with_changes(problem, disconnected=disconnected), n_starts=4, seed=1)
    for projection in result.projections:
        assert np.isnan(projection.layout[[3, 16]]).all()
        assert np.isfinite(np.delete(projection.layout, [3, 16], axis=0)).all()


def test_incremental_relax_places_new_points_and_keeps_unmovable_ones():
    problem, truth = synthetic_table(n_antigens=25, n_sera=6, noise=0.3, seed=4)
    base = opt.relax(problem, n_starts=10, seed=8).best.layout
    start = base.copy()
    new_points = [5, 6, 27]
    start[new_points] = np.nan
    result = opt.relax(problem, n_starts=10, seed=9, start_layout=start)
    assert np.isfinite(result.best.layout).all()
    assert result.best.stress == pytest.approx(opt.stress(problem, base), abs=1e-3) or (
        result.best.stress < opt.stress(problem, base)
    )

    unmovable = ~np.isnan(start).any(axis=1)
    fixed = opt.relax(
        with_changes(problem, unmovable=unmovable), n_starts=6, seed=9, start_layout=start
    )
    np.testing.assert_array_equal(fixed.best.layout[unmovable], start[unmovable])
    assert np.isfinite(fixed.best.layout[new_points]).all()


def test_incremental_relax_is_seeded_and_thread_independent():
    problem, _ = synthetic_table(n_antigens=20, n_sera=6, noise=0.5, seed=12)
    start = opt.relax(problem, n_starts=4, seed=1).best.layout
    start[[0, 21]] = np.nan
    a = opt.relax(problem, n_starts=8, seed=77, start_layout=start, threads=1)
    b = opt.relax(problem, n_starts=8, seed=77, start_layout=start, threads=3)
    for x, y in zip(a.projections, b.projections, strict=True):
        np.testing.assert_array_equal(x.layout, y.layout)


def combine(results: list[opt.RelaxResult]) -> list[opt.Projection]:
    return opt.sort_projections([p for r in results for p in r.projections])


def assert_same_projections(a: list[opt.Projection], b: list[opt.Projection]) -> None:
    assert [p.start_index for p in a] == [p.start_index for p in b]
    for x, y in zip(a, b, strict=True):
        np.testing.assert_array_equal(x.layout, y.layout)
        assert x.stress == y.stress and x.start_seed == y.start_seed


def test_scratch_run_split_into_jobs_equals_one_run():
    problem, _ = synthetic_table(noise=0.8, seed=5)
    whole = opt.relax(problem, n_starts=12, seed=31)
    parts = [
        opt.relax(problem, n_starts=n, first_start=f, seed=31) for f, n in ((0, 5), (5, 4), (9, 3))
    ]
    assert_same_projections(whole.projections, combine(parts))
    assert sorted(p.start_index for p in whole.projections) == list(range(12))


def test_incremental_run_split_into_jobs_equals_one_run():
    """Jobs run rough only; refine() on the combination gives the single run's result."""
    problem, _ = synthetic_table(n_antigens=25, n_sera=6, noise=0.5, seed=14)
    start = opt.relax(problem, n_starts=4, seed=1).best.layout
    start[[2, 3, 26]] = np.nan
    whole = opt.relax(problem, n_starts=12, seed=8, start_layout=start)
    parts = [
        opt.relax(problem, n_starts=n, first_start=f, seed=8, start_layout=start, precision="rough")
        for f, n in ((0, 7), (7, 5))
    ]
    assert_same_projections(whole.projections, opt.refine(problem, combine(parts)))


def test_refine_keeps_start_identity_and_only_touches_the_best():
    problem, _ = synthetic_table(n_antigens=20, n_sera=6, noise=0.5, seed=15)
    start = opt.relax(problem, n_starts=4, seed=1).best.layout
    start[[0, 21]] = np.nan
    rough = opt.relax(
        problem, n_starts=10, seed=3, start_layout=start, precision="rough"
    ).projections
    refined = opt.refine(problem, rough, n_best=3)
    by_index = {p.start_index: p for p in rough}
    for p in refined:
        before = by_index[p.start_index]
        assert p.start_seed == before.start_seed
        assert p.stress <= before.stress + 1e-12
    untouched = {p.start_index for p in rough[3:]}
    for p in refined:
        if p.start_index in untouched:
            np.testing.assert_array_equal(p.layout, by_index[p.start_index].layout)
    stresses = [p.stress for p in refined]
    assert stresses == sorted(stresses)


def test_first_start_is_checked():
    problem, _ = synthetic_table(n_antigens=6, n_sera=3)
    with pytest.raises(ValueError, match="first_start"):
        opt.relax(problem, n_starts=2, seed=1, first_start=-1)


# ---------------------------------------------------------------------------------------
# grid test


def reflected_antigen(seed: int):
    """A table in which antigen 0 is titrated against three sera only, its best map, and
    that map with antigen 0 reflected across the line through the first two sera and
    re-minimised: a local minimum that can be wrong."""
    problem, _ = synthetic_table(
        n_antigens=20, n_sera=6, noise=0.0, thresholds=False, missing=0.0, seed=seed
    )
    value, kind = problem.titre_value.copy(), problem.titre_type.copy()
    value[0, 3:] = np.nan
    kind[0, 3:] = TitreType.MISSING
    problem = with_changes(problem, titre_value=value, titre_type=kind)
    best = opt.relax(problem, n_starts=10, seed=1).best
    n_ag = problem.n_antigens
    a, b = best.layout[n_ag], best.layout[n_ag + 1]
    along = (b - a) / np.linalg.norm(b - a)
    offset = best.layout[0] - a
    layout = best.layout.copy()
    layout[0] = a + 2.0 * (offset @ along) * along - offset
    return problem, best, opt.optimise(problem, layout)


def test_grid_test_finds_and_resolves_a_trapped_point():
    problem, best, trapped = reflected_antigen(seed=4)
    assert trapped.stress > best.stress + 1.0  # measured: 9.002 vs 7.119
    assert all(r.diagnosis != "trapped" for r in opt.grid_test(problem, best.layout))
    grid = opt.grid_test(problem, trapped.layout)
    assert grid[0].diagnosis == "trapped"
    assert grid[0].stress_diff < -1.0
    resolved = opt.resolve_trapped(problem, trapped.layout)
    assert resolved.moved >= 1
    assert resolved.projection.stress == pytest.approx(best.stress, rel=1e-4)
    assert all(r.diagnosis != "trapped" for r in resolved.last_grid)


def test_grid_test_reports_hemisphering():
    problem, best, reflected = reflected_antigen(seed=2)
    # measured: the reflected position is 2.9 units away and only 0.01 worse
    grid = opt.grid_test(problem, reflected.layout)
    assert grid[0].diagnosis == "hemisphering"
    assert grid[0].distance > 1.0 and abs(grid[0].stress_diff) < 0.25


def test_grid_test_excludes_disconnected_and_unmovable():
    problem, truth = synthetic_table(n_antigens=10, n_sera=4)
    disconnected = np.zeros(problem.n_points, dtype=bool)
    disconnected[1] = True
    unmovable = np.zeros(problem.n_points, dtype=bool)
    unmovable[2] = True
    layout = truth.copy()
    layout[1] = np.nan
    grid = opt.grid_test(
        with_changes(problem, disconnected=disconnected, unmovable=unmovable), layout
    )
    assert grid[1].diagnosis == "excluded" and grid[2].diagnosis == "excluded"
    assert len(grid) == problem.n_points


# ---------------------------------------------------------------------------------------
# input checks


def test_rejects_bad_inputs():
    problem, truth = synthetic_table(n_antigens=6, n_sera=3)
    with pytest.raises(ValueError, match="gradient_multipliers"):
        with_changes(problem, gradient_multipliers=np.full(problem.n_points, 2.0))
    both = np.zeros(problem.n_points, dtype=bool)
    both[0] = True
    with pytest.raises(ValueError, match="both unmovable and disconnected"):
        with_changes(problem, unmovable=both, disconnected=both)
    with pytest.raises(ValueError, match="column_bases"):
        with_changes(problem, column_bases=problem.column_bases[:-1])
    bad = problem.titre_type.copy()
    bad[0, 0] = 7
    with pytest.raises(ValueError, match="not 0..4"):
        with_changes(problem, titre_type=bad)
    with pytest.raises(TypeError):
        with_changes(problem, disconnected=np.zeros(problem.n_points, dtype=int))
    with pytest.raises(ValueError, match="seed"):
        opt.relax(problem, n_starts=2, seed=-1)
    with pytest.raises(ValueError, match="unmovable"):
        opt.relax(with_changes(problem, unmovable=both), n_starts=2, seed=1)
    partial = truth.copy()
    partial[0, 0] = np.nan
    with pytest.raises(ValueError, match="some but not all"):
        opt.relax(problem, n_starts=2, seed=1, start_layout=partial)


def test_too_few_connected_points():
    problem, _ = synthetic_table(n_antigens=3, n_sera=2)
    disconnected = np.ones(problem.n_points, dtype=bool)
    disconnected[:2] = False
    with pytest.raises(ValueError, match="fewer than 3 connected"):
        opt.relax(with_changes(problem, disconnected=disconnected), n_starts=1, seed=1)


def test_column_bases_helper():
    value = np.array([[3.0, np.nan], [5.0, 2.0], [1.0, 6.0]])
    kind = np.array(
        [
            [TitreType.REGULAR, TitreType.MISSING],
            [TitreType.LESS_THAN, TitreType.MORE_THAN],
            [TitreType.REGULAR, TitreType.DODGY],
        ],
        dtype=np.int8,
    )
    np.testing.assert_array_equal(opt.column_bases(value, kind), [5.0, 3.0])
    np.testing.assert_array_equal(opt.column_bases(value, kind, minimum=4.0), [5.0, 4.0])


# ---------------------------------------------------------------------------------------
# exceptions across pybind11 modules

_MATPLOTLIB_FIRST = """
import matplotlib._path  # a pybind11 module, loaded before af.map._core
import numpy as np
import pytest
from af.map import optimise as opt
from af.map.optimise import MapProblem

value = np.zeros((3, 2))
kind = np.ones((3, 2), dtype=np.int8)
disconnected = np.zeros(5, dtype=bool)
with pytest.raises(ValueError, match="gradient_multipliers"):
    MapProblem(value, kind, np.full(2, 4.0), disconnected, dodgy_is_regular=False,
               gradient_multipliers=np.full(5, 2.0))
disconnected[:3] = True
problem = MapProblem(value, kind, np.full(2, 4.0), disconnected, dodgy_is_regular=False)
with pytest.raises(ValueError, match="fewer than 3 connected"):
    opt.relax(problem, n_starts=1, seed=1)
with pytest.raises(TypeError):
    opt._core.start_seed("not a number", 0)
print("ok")
"""


def test_exception_types_survive_matplotlib_loaded_first():
    """A pybind11 module loaded before _core (matplotlib's) once turned every C++ error into
    "RuntimeError: Caught an unknown exception!". Run in a fresh interpreter so the import
    order is certain whatever other tests have imported."""
    env = dict(os.environ, MPLBACKEND="Agg")
    env.setdefault("MPLCONFIGDIR", tempfile.mkdtemp())
    result = subprocess.run(
        [sys.executable, "-c", _MATPLOTLIB_FIRST], capture_output=True, text=True, env=env
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert result.stdout.strip() == "ok"
