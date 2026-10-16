"""Tests of the optimiser core on synthetic tables (generated here; no real data)."""

from __future__ import annotations

import dataclasses
import json
import math
import os
import subprocess
import sys
import tempfile
import typing
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
    # no trapped point is left with a move that would lower the stress
    assert not any(r.diagnosis == "trapped" and r.stress_diff < 0.0 for r in resolved.last_grid)


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


# ---------------------------------------------------------------------------------------
# groups stuck together (resolve_trapped move_groups)


def stuck_block(seed: int = 11, copies: int = 6):
    """Antigen 0 repeated `copies` times (identical titres against three sera only), all
    copies reflected together across the line through the first two sera and re-minimised.
    Measured (seed 11, 6 copies): each copy alone is only hemisphering, so the plain loop
    moves nothing (stress 6.708); moved together they return to the best map (6.065)."""
    problem, _ = synthetic_table(
        n_antigens=20, n_sera=6, noise=0.0, thresholds=False, missing=0.0, seed=seed
    )
    value, kind = problem.titre_value.copy(), problem.titre_type.copy()
    value[0, 3:] = np.nan
    kind[0, 3:] = TitreType.MISSING
    value = np.vstack([np.repeat(value[:1], copies, axis=0), value[1:]])
    kind = np.vstack([np.repeat(kind[:1], copies, axis=0), kind[1:]])
    n_points = value.shape[0] + value.shape[1]
    block = MapProblem(
        value, kind, problem.column_bases, np.zeros(n_points, dtype=bool), dodgy_is_regular=False
    )
    best = opt.relax(block, n_starts=10, seed=1).best
    n_ag = value.shape[0]
    a, b = best.layout[n_ag], best.layout[n_ag + 1]
    along = (b - a) / np.linalg.norm(b - a)
    layout = best.layout.copy()
    for i in range(copies):
        offset = layout[i] - a
        layout[i] = a + 2.0 * (offset @ along) * along - offset
    return block, best, opt.optimise(block, layout), list(range(copies))


def test_group_pass_moves_a_block_the_plain_loop_leaves():
    block, best, stuck, copies = stuck_block()
    plain = opt.resolve_trapped(block, stuck.layout)
    assert plain.moved == 0 and plain.groups == []
    assert plain.projection.stress > best.stress + 0.5
    assert {plain.last_grid[i].diagnosis for i in copies} == {"hemisphering"}

    grouped = opt.resolve_trapped(block, stuck.layout, move_groups=True)
    kept = [g for g in grouped.groups if g.kept]
    assert len(kept) == 1
    assert kept[0].members == copies
    assert kept[0].stress_before == pytest.approx(plain.projection.stress)
    assert kept[0].stress_after == pytest.approx(grouped.projection.stress)
    assert len(kept[0].shift) == 2
    assert grouped.projection.stress == pytest.approx(best.stress, rel=1e-4)
    # the block is all antigens: composition reported, not flagged as mixed
    assert (kept[0].n_antigens, kept[0].n_sera, kept[0].mixed) == (len(copies), 0, False)
    # last_grid describes the returned map
    fresh = opt.grid_test(block, grouped.projection.layout)
    assert [(r.diagnosis, r.stress_diff) for r in grouped.last_grid] == [
        (r.diagnosis, r.stress_diff) for r in fresh
    ]


def test_move_groups_is_off_by_default_and_changes_nothing_then():
    block, _, stuck, _ = stuck_block()
    default = opt.resolve_trapped(block, stuck.layout)
    off = opt.resolve_trapped(block, stuck.layout, move_groups=False)
    np.testing.assert_array_equal(default.projection.layout, off.projection.layout)
    assert default.groups == off.groups == []


@pytest.mark.parametrize("seed", [3, 5, 9, 12])
def test_group_pass_never_raises_the_stress(seed):
    problem, _ = synthetic_table(n_antigens=25, n_sera=6, noise=0.8, seed=seed)
    for projection in opt.relax(problem, n_starts=20, seed=seed, keep=5).projections:
        plain = opt.resolve_trapped(problem, projection.layout)
        grouped = opt.resolve_trapped(problem, projection.layout, move_groups=True)
        assert grouped.projection.stress <= plain.projection.stress
        for group in grouped.groups:
            assert group.kept == (group.stress_after < group.stress_before - opt.GROUP_MIN_GAIN)
        if not any(g.kept for g in grouped.groups):
            np.testing.assert_array_equal(grouped.projection.layout, plain.projection.layout)


# ---------------------------------------------------------------------------------------
# hemisphering points whose other position is lower (resolve_trapped move_lower)


def assert_move_lower_record(resolved: opt.TrappedResolution, problem: MapProblem) -> None:
    """What every move-lower result must say: kept rounds lower the stress by more than the
    rule's margin, one after another; a rejected round is the last and changes nothing; the
    last grid test describes the returned map."""
    rule = resolved.move_lower_rule or opt.MOVE_LOWER_RULE
    stress = None
    for n, r in enumerate(resolved.move_lower, start=1):
        assert r.round == n
        if stress is not None:
            assert r.stress_before == stress
        assert r.kept == (r.stress_after < r.stress_before - rule.min_gain)
        assert r.kept == bool(r.moved)
        assert set(r.moved) <= set(r.candidates)
        if not r.kept:
            assert n == len(resolved.move_lower) and r.stress_after == r.stress_before
            assert resolved.move_lower_stop == "stress did not fall"
        stress = r.stress_after
    if stress is not None:
        assert resolved.projection.stress == stress
    fresh = opt.grid_test(problem, resolved.projection.layout)
    assert [(r.diagnosis, r.stress_diff) for r in resolved.last_grid] == [
        (r.diagnosis, r.stress_diff) for r in fresh
    ]


@pytest.mark.parametrize("numerics", opt.NUMERICS_OFFERED)
def test_move_lower_moves_a_block_the_plain_loop_leaves(numerics):
    block, best, stuck, copies = stuck_block()
    block = with_changes(block, numerics=numerics)
    stuck = opt.optimise(block, stuck.layout)
    best = opt.optimise(block, best.layout)
    plain = opt.resolve_trapped(block, stuck.layout)
    assert plain.projection.stress > best.stress + 0.5
    assert plain.move_lower == [] and plain.move_lower_stop is None

    lower = opt.resolve_trapped(block, stuck.layout, move_lower=True)
    assert lower.move_lower_stop == "nothing lower"
    first = lower.move_lower[0]
    assert set(copies) <= set(first.candidates)
    assert first.how == "jointly" and first.kept
    assert first.stress_before == pytest.approx(plain.projection.stress)
    assert lower.projection.stress == pytest.approx(best.stress, rel=1e-4)
    assert_move_lower_record(lower, block)
    # nothing the rule would move is left
    assert not [
        r
        for r in lower.last_grid
        if r.position is not None
        and r.stress_diff < 0.0
        and r.diagnosis in opt.MOVE_LOWER_RULE.diagnoses
    ]


def test_move_lower_moves_one_hemisphering_point():
    # measured (seed 6): the plain loop moves nothing and leaves antigen 0 hemisphering, its
    # other position 9 units away and 0.04 lower; the rule moves it alone, gaining 0.05
    problem, _, reflected = reflected_antigen(seed=6)
    plain = opt.resolve_trapped(problem, reflected.layout)
    assert plain.moved == 0
    assert plain.last_grid[0].diagnosis == "hemisphering" and plain.last_grid[0].stress_diff < 0
    lower = opt.resolve_trapped(problem, reflected.layout, move_lower=True)
    first = lower.move_lower[0]
    assert (first.candidates, first.how, first.moved) == ((0,), "jointly", (0,))
    assert lower.projection.stress < plain.projection.stress - opt.MOVE_LOWER_RULE.min_gain
    assert lower.move_lower_stop == "nothing lower"
    assert_move_lower_record(lower, problem)


def test_move_lower_is_off_by_default_and_changes_nothing_then():
    block, _, stuck, _ = stuck_block()
    default = opt.resolve_trapped(block, stuck.layout)
    off = opt.resolve_trapped(block, stuck.layout, move_lower=False)
    np.testing.assert_array_equal(default.projection.layout, off.projection.layout)
    assert default.move_lower == off.move_lower == []
    assert default.move_lower_stop is None and off.move_lower_stop is None


@pytest.mark.parametrize("seed", [3, 5, 9, 12])
def test_move_lower_never_raises_the_stress(seed):
    problem, _ = synthetic_table(n_antigens=25, n_sera=6, noise=0.8, seed=seed)
    for projection in opt.relax(problem, n_starts=20, seed=seed, keep=5).projections:
        plain = opt.resolve_trapped(problem, projection.layout)
        lower = opt.resolve_trapped(problem, projection.layout, move_lower=True)
        assert lower.projection.stress <= plain.projection.stress
        assert lower.move_lower_stop is not None
        assert_move_lower_record(lower, problem)
        if not any(r.kept for r in lower.move_lower):
            np.testing.assert_array_equal(lower.projection.layout, plain.projection.layout)


@pytest.mark.parametrize("case", ["one point", "block"])
def test_move_lower_tries_one_at_a_time_when_the_joint_move_fails(monkeypatch, case):
    if case == "one point":  # the seed-6 table above: its one candidate, alone, is kept
        problem, _, start = reflected_antigen(seed=6)
    else:
        problem, _, start, _ = stuck_block()
    expected = opt.resolve_trapped(problem, start.layout, move_lower=True)
    real = opt.optimise
    calls = []

    def joint_fails(problem, layout, **kwargs):
        # the first minimise the rule asks for is its joint move (the loop moves nothing here)
        result = real(problem, layout, **kwargs)
        calls.append(result.stress)
        return dataclasses.replace(result, stress=math.inf) if len(calls) == 1 else result

    monkeypatch.setattr(opt, "optimise", joint_fails)
    lower = opt.resolve_trapped(problem, start.layout, move_lower=True)
    first = lower.move_lower[0]
    assert first.joint_stress == math.inf and first.how == "one at a time"
    assert len(calls) == 1 + len(first.candidates) + sum(
        1 + len(r.candidates) * (r.how == "one at a time") for r in lower.move_lower[1:]
    )
    assert_move_lower_record(lower, problem)
    assert lower.projection.stress <= start.stress
    if case == "one point":
        # alone, the move is the joint move: kept, and the same map as without the failure
        assert first.moved == (0,) and first.kept
        assert lower.projection.stress == expected.projection.stress


def test_move_lower_stops_at_max_rounds():
    block, _, stuck, _ = stuck_block()
    lower = opt.resolve_trapped(block, stuck.layout, move_lower=True, max_rounds=1)
    assert len(lower.move_lower) == 1
    assert lower.move_lower_stop in ("round limit", "nothing lower")
    assert_move_lower_record(lower, block)


def test_move_lower_rule_is_a_record():
    rule = dataclasses.asdict(opt.MOVE_LOWER_RULE)
    assert json.loads(json.dumps(rule)) == {**rule, "diagnoses": list(rule["diagnoses"])}
    assert rule["name"] == "move-lower-1"
    assert set(rule["diagnoses"]) <= set(typing.get_args(opt.Diagnosis))


# AD's last round (AD_LAST_ROUND, Q198): one joint round, kept if the stress falls at all


def ad_last_round(problem: MapProblem, plain: opt.TrappedResolution) -> np.ndarray:
    """AD chart-relax-grid's last round, written out independently of resolve_trapped
    (acmacs-chart-2 grid-test.cc): on the map the trapped loop finished, move every trapped or
    hemisphering point whose other position lowers the stress, minimise once (fine), and keep the
    moved map only if its stress is lower (AD keeps both projections and sorts them by stress)."""
    layout = plain.projection.layout
    moves = [
        r
        for r in plain.last_grid
        if r.diagnosis in ("trapped", "hemisphering")
        and r.position is not None
        and r.stress_diff < 0.0
    ]
    if not moves:
        return layout
    trial = layout.copy()
    for r in moves:
        trial[r.point] = r.position
    moved = opt.optimise(problem, trial, precision="fine")
    return moved.layout if moved.stress < plain.projection.stress else layout


def ad_cases():
    block, _, stuck, _ = stuck_block()
    yield "block", block, stuck.layout
    problem, _, reflected = reflected_antigen(seed=6)
    yield "one point", problem, reflected.layout
    for seed in (3, 5, 9):
        problem, _ = synthetic_table(n_antigens=25, n_sera=6, noise=0.8, seed=seed)
        for n, projection in enumerate(
            opt.relax(problem, n_starts=12, seed=seed, keep=3).projections
        ):
            yield f"seed {seed} #{n}", problem, projection.layout


def test_ad_last_round_is_ads_last_round():
    tried = moved = 0
    for name, problem, layout in ad_cases():
        plain = opt.resolve_trapped(problem, layout)
        ad = opt.resolve_trapped(
            problem, layout, move_lower=True, move_lower_rule=opt.AD_LAST_ROUND
        )
        np.testing.assert_array_equal(
            ad.projection.layout, ad_last_round(problem, plain), err_msg=name
        )
        assert ad.move_lower_rule is opt.AD_LAST_ROUND
        assert len(ad.move_lower) <= 1 and ad.move_lower_stop is not None
        assert_move_lower_record(ad, problem)
        tried += 1
        moved += any(r.kept for r in ad.move_lower)
    assert tried >= 10 and moved >= 2  # the comparison saw moves kept, not only empty rounds


def test_ad_last_round_is_the_first_round_of_move_lower():
    # where move-lower-1's joint round 1 is kept, AD's round is that round; measured the same on
    # 15 chain maps (notes/optimiser/hemi-lower/why-ae/)
    for name in ("block", "one point"):
        _, problem, layout = next(c for c in ad_cases() if c[0] == name)
        default = opt.resolve_trapped(problem, layout, move_lower=True)
        ad = opt.resolve_trapped(
            problem, layout, move_lower=True, move_lower_rule=opt.AD_LAST_ROUND
        )
        assert default.move_lower[0].how == "jointly" and default.move_lower[0].kept
        assert ad.move_lower[0] == default.move_lower[0]
        assert ad.projection.stress == default.move_lower[0].stress_after


def test_ad_last_round_never_tries_one_at_a_time(monkeypatch):
    block, _, stuck, _ = stuck_block()
    plain = opt.resolve_trapped(block, stuck.layout)
    real = opt.optimise
    calls = []

    def joint_fails(problem, layout, **kwargs):
        # the loop moves nothing here, so the first minimise asked for is the joint move
        result = real(problem, layout, **kwargs)
        calls.append(result.stress)
        return dataclasses.replace(result, stress=math.inf) if len(calls) == 1 else result

    monkeypatch.setattr(opt, "optimise", joint_fails)
    ad = opt.resolve_trapped(
        block, stuck.layout, move_lower=True, move_lower_rule=opt.AD_LAST_ROUND
    )
    assert len(calls) == 1
    (only,) = ad.move_lower
    assert (only.how, only.moved, only.kept) == ("jointly", (), False)
    assert ad.move_lower_stop == "stress did not fall"
    np.testing.assert_array_equal(ad.projection.layout, plain.projection.layout)


def test_ad_last_round_keeps_any_fall_in_stress(monkeypatch):
    # AD sorts its projections by stress, so a move that gains 1e-6 wins; move-lower-1 wants 1e-3
    problem, _, reflected = reflected_antigen(seed=6)
    plain = opt.resolve_trapped(problem, reflected.layout)
    real = opt.optimise

    def tiny_gain(problem, layout, **kwargs):
        return dataclasses.replace(
            real(problem, layout, **kwargs), stress=plain.projection.stress - 1e-6
        )

    monkeypatch.setattr(opt, "optimise", tiny_gain)
    for rule, kept in ((opt.AD_LAST_ROUND, True), (opt.MOVE_LOWER_RULE, False)):
        result = opt.resolve_trapped(
            problem, reflected.layout, move_lower=True, move_lower_rule=rule
        )
        assert result.move_lower[0].kept is kept


def test_ad_last_round_runs_one_round_whatever_max_rounds_says(monkeypatch):
    # no small invented table needs a second round (the chain maps did: up to 6), so this checks
    # the cap the rounds are given
    block, _, stuck, _ = stuck_block()
    real = opt._move_lower
    caps = []

    def capture(*args, max_rounds, **kwargs):
        caps.append(max_rounds)
        return real(*args, max_rounds=max_rounds, **kwargs)

    monkeypatch.setattr(opt, "_move_lower", capture)
    for rule in (opt.AD_LAST_ROUND, opt.MOVE_LOWER_RULE):
        result = opt.resolve_trapped(
            block, stuck.layout, max_rounds=7, move_lower=True, move_lower_rule=rule
        )
        assert len(result.move_lower) == 1
    assert caps == [1, 7]


def test_ad_last_round_rule_is_its_own_record():
    rule = dataclasses.asdict(opt.AD_LAST_ROUND)
    assert json.loads(json.dumps(rule)) == {**rule, "diagnoses": list(rule["diagnoses"])}
    assert rule["name"] == "ad-last-round-1" != opt.MOVE_LOWER_RULE.name
    assert (rule["max_rounds"], rule["min_gain"], rule["one_at_a_time"]) == (1, 0.0, False)
    assert rule["diagnoses"] == opt.MOVE_LOWER_RULE.diagnoses
    block, _, stuck, _ = stuck_block()
    assert opt.resolve_trapped(block, stuck.layout).move_lower_rule is None
    assert (
        opt.resolve_trapped(block, stuck.layout, move_lower=True).move_lower_rule
        is opt.MOVE_LOWER_RULE
    )
    with pytest.raises(ValueError, match="move_lower_rule.max_rounds"):
        opt.resolve_trapped(
            block, stuck.layout, move_lower=True, move_lower_rule=opt.MoveLowerRule(max_rounds=0)
        )


def test_group_tolerance_is_checked():
    problem, truth = synthetic_table(n_antigens=6, n_sera=3)
    with pytest.raises(ValueError, match="group_tolerance"):
        opt.resolve_trapped(problem, truth, move_groups=True, group_tolerance=0.0)


def _groups_from(problem: MapProblem, layout: np.ndarray, members: list[int], shift: np.ndarray):
    """Run the real grouping and move code on hand-made grid results in which `members` all
    have a better position at the same displacement (a mixed trap is hard to build from a
    table: a serum stuck with its antigens holds them in place, so their own grid tests
    find nothing)."""
    current = opt.Projection(layout, opt.stress(problem, layout), layout.shape[1], 0, 0, 0, 0)
    grid = [
        opt.GridResult(
            point=i,
            diagnosis="hemisphering",
            position=layout[i] + shift,
            distance=float(np.linalg.norm(shift)),
            stress_diff=-0.1,
        )
        for i in members
    ]
    _, tried = opt._move_groups(problem, current, grid, opt.GROUP_TOLERANCE, "cg")
    return tried


def test_group_with_antigens_and_sera_is_flagged_mixed():
    problem, _ = synthetic_table(n_antigens=10, n_sera=4, seed=3)
    layout = opt.relax(problem, n_starts=5, seed=1).best.layout
    antigens_and_serum = [2, 3, problem.n_antigens + 1]
    (group,) = _groups_from(problem, layout, antigens_and_serum, np.array([1.0, -0.5]))
    assert group.members == antigens_and_serum
    assert (group.n_antigens, group.n_sera, group.mixed) == (2, 1, True)


def test_group_of_antigens_only_is_not_flagged():
    problem, _ = synthetic_table(n_antigens=10, n_sera=4, seed=3)
    layout = opt.relax(problem, n_starts=5, seed=1).best.layout
    (group,) = _groups_from(problem, layout, [2, 3, 7], np.array([1.0, -0.5]))
    assert (group.n_antigens, group.n_sera, group.mixed) == (3, 0, False)


# ---------------------------------------------------------------------------------------
# threads actually used


def test_resolved_thread_count_is_recorded():
    problem, truth = synthetic_table(n_antigens=10, n_sera=4)
    expected = 2 if opt._core.openmp else 1
    assert opt.relax(problem, n_starts=4, seed=1, threads=2).threads == expected
    assert opt.resolve_trapped(problem, truth, threads=2).threads == expected
    assert opt.threads_used("test", 0) >= 1


_INHERITED_ONE_THREAD = """
import logging, sys
import numpy as np
from af.map import optimise as opt
from af.map.optimise import MapProblem
logging.basicConfig(level=logging.WARNING, stream=sys.stderr, format="%(levelname)s %(message)s")
rng = np.random.default_rng(1)
value = np.round(rng.uniform(0, 6, (8, 4)))
kind = np.ones((8, 4), dtype=np.int8)
problem = MapProblem(value, kind, np.full(4, 7.0), np.zeros(12, dtype=bool), dodgy_is_regular=False)
print("threads=0", opt.relax(problem, n_starts=2, seed=1, threads=0).threads)
print("threads=2", opt.relax(problem, n_starts=2, seed=1, threads=2).threads)
"""


def test_inherited_omp_num_threads_is_warned_about():
    """An HPC login environment with OMP_NUM_THREADS=1, inherited by jobs, made threads=0 run
    single-threaded with no sign. It must show in the result and in a warning; an explicit
    thread count must still win."""
    if not opt._core.openmp or opt.available_cpus() < 2:
        pytest.skip("needs an OpenMP build and at least 2 CPUs")
    env = dict(os.environ, OMP_NUM_THREADS="1")
    result = subprocess.run(
        [sys.executable, "-c", _INHERITED_ONE_THREAD], capture_output=True, text=True, env=env
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert result.stdout.split() == ["threads=0", "1", "threads=2", "2"]
    assert "threads=0 resolved to 1 thread(s)" in result.stderr
    assert "OMP_NUM_THREADS=1" in result.stderr
    assert result.stderr.count("WARNING") == 1


# ---------------------------------------------------------------------------------------
# per-titre and per-point stress


def reference_terms(problem: MapProblem, layout: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-titre stress terms and table distances written out independently of the C++ code
    (NaN where a titre is not fitted)."""
    n_ag, n_sr = problem.titre_value.shape
    adjust = (
        np.zeros(problem.n_points)
        if problem.avidity_adjust is None
        else np.log2(problem.avidity_adjust)
    )
    weights = np.ones((n_ag, n_sr)) if problem.weights is None else problem.weights
    terms, targets = np.full((n_ag, n_sr), np.nan), np.full((n_ag, n_sr), np.nan)
    for ag in range(n_ag):
        for sr in range(n_sr):
            kind = problem.titre_type[ag, sr]
            if kind == TitreType.DODGY and problem.dodgy_is_regular:
                kind = TitreType.REGULAR
            if kind not in (TitreType.REGULAR, TitreType.LESS_THAN):
                continue
            if problem.disconnected[ag] or problem.disconnected[n_ag + sr]:
                continue
            target = max(
                0.0,
                problem.column_bases[sr]
                - problem.titre_value[ag, sr]
                - adjust[ag]
                - adjust[n_ag + sr],
            )
            dist = np.linalg.norm(layout[ag] - layout[n_ag + sr])
            targets[ag, sr] = target
            if kind == TitreType.REGULAR:
                terms[ag, sr] = weights[ag, sr] * (target - dist) ** 2
            else:
                diff = target - dist + 1.0
                terms[ag, sr] = weights[ag, sr] * diff**2 / (1.0 + np.exp(-10.0 * diff))
    return terms, targets


def mixed_table(seed: int = 7) -> tuple[MapProblem, np.ndarray]:
    """A table exercising every case: regular, '<', '>', missing and dodgy titres, weights,
    avidity adjustments, and one disconnected antigen and one disconnected serum."""
    problem, truth = synthetic_table(n_antigens=14, n_sera=6, noise=0.8, seed=seed)
    rng = np.random.default_rng(seed)
    kind = problem.titre_type.copy()
    regular = np.argwhere(kind == TitreType.REGULAR)[:4]
    kind[tuple(regular.T)] = TitreType.DODGY
    value = problem.titre_value.copy()
    more = np.argwhere(kind == TitreType.REGULAR)[4:7]
    kind[tuple(more.T)] = TitreType.MORE_THAN  # '>' titres: never fitted
    disconnected = np.zeros(problem.n_points, dtype=bool)
    disconnected[[3, problem.n_antigens + 2]] = True
    problem = with_changes(
        problem,
        titre_value=value,
        titre_type=kind,
        disconnected=disconnected,
        weights=rng.uniform(0.5, 2.0, problem.titre_value.shape),
        avidity_adjust=rng.uniform(0.7, 1.4, problem.n_points),
    )
    layout = truth + rng.normal(0.0, 1.0, truth.shape)
    layout[[3, problem.n_antigens + 2]] = np.nan
    return problem, layout


@pytest.mark.parametrize("dodgy_is_regular", [False, True])
def test_stress_table_matches_the_written_out_terms(dodgy_is_regular):
    problem, layout = mixed_table()
    problem = with_changes(problem, dodgy_is_regular=dodgy_is_regular)
    kinds = set(problem.titre_type.ravel().tolist())
    assert {
        TitreType.REGULAR,
        TitreType.LESS_THAN,
        TitreType.MORE_THAN,
        TitreType.MISSING,
        TitreType.DODGY,
    } <= kinds
    expected_terms, _ = reference_terms(problem, layout)
    terms = opt.stress_table(problem, layout)
    assert terms.shape == problem.titre_value.shape
    np.testing.assert_array_equal(np.isnan(terms), np.isnan(expected_terms))
    np.testing.assert_allclose(terms, expected_terms, rtol=1e-12, equal_nan=True)
    assert np.nansum(terms) == pytest.approx(opt.stress(problem, layout), rel=1e-12)


def test_point_stress_counts_each_titre_at_both_ends():
    problem, layout = mixed_table()
    table, points = opt.stress_table(problem, layout), opt.point_stress(problem, layout)
    assert points.shape == (problem.n_points,)
    np.testing.assert_allclose(points[: problem.n_antigens], np.nansum(table, axis=1))
    np.testing.assert_allclose(points[problem.n_antigens :], np.nansum(table, axis=0))
    assert points.sum() == pytest.approx(2.0 * opt.stress(problem, layout), rel=1e-12)
    assert points[3] == 0.0 and points[problem.n_antigens + 2] == 0.0  # the disconnected points


def test_stress_table_agrees_term_by_term_with_the_stub_optimiser():
    """af.chain.stub_optimiser keeps its own numpy copy of the formula so the chain runs without
    the compiled core; this keeps the two implementations in step, titre by titre."""
    from af.chain import stub_optimiser

    problem, layout = mixed_table(seed=11)
    arrays = {
        "titre_value": problem.titre_value,
        "titre_type": problem.titre_type,
        "column_bases": problem.column_bases,
        "disconnected": problem.disconnected,
        "dodgy_is_regular": problem.dodgy_is_regular,
        "weights": problem.weights,
        "avidity_adjust": problem.avidity_adjust,
    }
    t = stub_optimiser.stress_terms(arrays)
    ag, sr = t.p1, t.p2 - problem.n_antigens
    terms = opt.stress_table(problem, layout)
    _, targets = reference_terms(problem, layout)  # independently written formula
    # the stub fits exactly the titres the core fits
    stub_cells = np.zeros(terms.shape, dtype=bool)
    stub_cells[ag, sr] = True
    np.testing.assert_array_equal(stub_cells, ~np.isnan(terms))
    np.testing.assert_allclose(t.distance, targets[ag, sr], rtol=1e-12)
    assert np.array_equal(t.less_than, problem.titre_type[ag, sr] == TitreType.LESS_THAN)
    # per-term stress from the stub's own fields and sigmoid
    filled = np.where(np.isnan(layout), 0.0, layout)
    diff = t.distance - np.linalg.norm(filled[t.p1] - filled[t.p2], axis=1)
    shifted = diff + 1.0
    stub_terms = np.where(
        t.less_than,
        t.weight
        * shifted**2
        * stub_optimiser._sigmoid(stub_optimiser.SIGMOID_MULTIPLIER * shifted),
        t.weight * diff**2,
    )
    np.testing.assert_allclose(stub_terms, terms[ag, sr], rtol=1e-12)
    stub_total, _ = stub_optimiser.stress_and_gradient(filled.ravel(), t, layout.shape[1])
    assert stub_total == pytest.approx(opt.stress(problem, layout), rel=1e-12)


def test_table_distances_match_the_written_out_targets_and_stress():
    problem, layout = mixed_table()
    targets = opt.table_distances(problem)
    terms = opt.stress_table(problem, layout)
    _, expected = reference_terms(problem, layout)
    assert targets.shape == problem.titre_value.shape
    np.testing.assert_array_equal(np.isnan(targets), np.isnan(terms))
    np.testing.assert_allclose(targets, expected, rtol=1e-12, equal_nan=True)
    assert np.nanmin(targets) >= 0.0  # clipped at 0, as ae
    # the documented sign convention: residual = target - map distance, and for regular,
    # unweighted titres its square is the stress term
    n_ag = problem.n_antigens
    plain = with_changes(problem, weights=None)
    regular = (plain.titre_type == TitreType.REGULAR) & ~np.isnan(targets)
    D = np.linalg.norm(layout[:n_ag, None, :] - layout[None, n_ag:, :], axis=2)
    residual = targets - D
    np.testing.assert_allclose(
        residual[regular] ** 2, opt.stress_table(plain, layout)[regular], rtol=1e-12
    )


# ---------------------------------------------------------------------------------------
# antigen reactivity


def planted_reactivity(
    seed: int = 1, *, n_ag: int = 40, n_sr: int = 10, n_reactive: int = 8, size: float = 1.5
):
    """An invented table whose antigens' titres are shifted by known amounts: titre = column
    basis - distance + shift. The fitted adjustment that undoes a shift is minus it."""
    rng = np.random.default_rng(seed)
    truth = rng.uniform(-4.0, 4.0, (n_ag + n_sr, 2))
    dist = np.linalg.norm(truth[:n_ag, None] - truth[None, n_ag:], axis=2)
    colbase = rng.uniform(7.0, 9.0, n_sr)
    shift = np.zeros(n_ag)
    reactive = rng.choice(n_ag, n_reactive, replace=False)
    shift[reactive] = rng.choice([-1.0, 1.0], n_reactive) * size
    value = colbase[None, :] - dist + shift[:, None]
    kind = np.ones(value.shape, dtype=np.int8)
    problem = MapProblem(
        value, kind, colbase, np.zeros(n_ag + n_sr, dtype=bool), dodgy_is_regular=False
    )
    return problem, opt.relax(problem, n_starts=50, seed=seed).best.layout, shift


@pytest.mark.parametrize("column_bases", ["fixed", "recompute"])
@pytest.mark.parametrize("clip", [True, False])
def test_reactivity_objective_gradient_matches_finite_differences(column_bases, clip):
    problem, layout = mixed_table(seed=5)
    rng = np.random.default_rng(2)
    reactivity = rng.normal(0.0, 0.7, problem.n_antigens)
    kw: dict[str, Any] = dict(
        penalty=0.6, column_bases=column_bases, clip=clip, minimum_column_basis=1.0
    )

    def value(lay, r):
        return opt._core.reactivity_objective(problem._core_problem, lay, r, **kw)[0]

    _, g_layout, g_r = opt._core.reactivity_objective(
        problem._core_problem, layout, reactivity, **kw
    )
    h = 1e-6
    for index in map(tuple, np.argwhere(np.isfinite(layout))):
        plus, minus = layout.copy(), layout.copy()
        plus[index] += h
        minus[index] -= h
        assert g_layout[index] == pytest.approx(
            (value(plus, reactivity) - value(minus, reactivity)) / (2 * h), abs=1e-5
        )
    for i in range(problem.n_antigens):
        plus, minus = reactivity.copy(), reactivity.copy()
        plus[i] += h
        minus[i] -= h
        assert g_r[i] == pytest.approx(
            (value(layout, plus) - value(layout, minus)) / (2 * h), abs=1e-5
        )


def test_reactivity_objective_without_adjustment_is_the_stress():
    problem, layout = mixed_table()
    zero = np.zeros(problem.n_antigens)
    value = opt._core.reactivity_objective(
        problem._core_problem,
        layout,
        zero,
        penalty=1.0,
        column_bases="fixed",
        clip=True,
        minimum_column_basis=0.0,
    )[0]
    assert value == pytest.approx(opt.stress(problem, layout), rel=1e-12)


def test_fit_recovers_planted_shifts_on_clean_data():
    problem, layout, shift = planted_reactivity()
    fit = opt.fit_antigen_reactivity(problem, layout, penalty=0.0)
    # measured: RMSE 0.000 on noise-free data
    np.testing.assert_allclose(fit.reactivity, -shift, atol=1e-3)
    assert fit.stress < 1e-6 < fit.stress_before
    assert fit.objective == pytest.approx(fit.stress)


def test_a_large_penalty_gives_the_plain_map():
    problem, layout, _ = planted_reactivity(seed=2)
    fit = opt.fit_antigen_reactivity(problem, layout, penalty=1e3)
    assert np.max(np.abs(fit.reactivity)) < 1e-3
    assert fit.stress == pytest.approx(opt.optimise(problem, layout).stress, rel=1e-4)


def test_held_antigens_keep_their_values():
    problem, layout, _ = planted_reactivity(seed=3)
    fixed = np.full(problem.n_antigens, np.nan)
    fixed[[0, 5]] = [0.7, -1.2]
    fit = opt.fit_antigen_reactivity(problem, layout, penalty=0.5, fixed=fixed)
    assert fit.reactivity[0] == 0.7 and fit.reactivity[5] == -1.2
    assert np.any(fit.reactivity[np.isnan(fixed)] != 0.0)


@pytest.mark.parametrize("column_bases", ["fixed", "recompute"])
def test_applying_the_fit_reproduces_its_stress(column_bases):
    """The documented way to use a fit: avidity_adjust = 2 ** reactivity for the antigens, and the
    returned column bases."""
    problem, layout, _ = planted_reactivity(seed=4)
    fit = opt.fit_antigen_reactivity(problem, layout, penalty=0.5, column_bases=column_bases)
    avidity = np.concatenate([2.0**fit.reactivity, np.ones(problem.n_sera)])
    applied = with_changes(problem, avidity_adjust=avidity, column_bases=fit.column_bases)
    assert opt.stress(applied, fit.projection.layout) == pytest.approx(fit.stress, rel=1e-9)
    if column_bases == "fixed":
        np.testing.assert_array_equal(fit.column_bases, problem.column_bases)


def test_recomputed_column_bases_follow_the_adjusted_titres():
    problem, _ = mixed_table()
    reactivity = np.random.default_rng(4).normal(0.0, 1.0, problem.n_antigens)
    bases = opt._core.reactivity_column_bases(
        problem._core_problem, reactivity, column_bases="recompute", minimum_column_basis=2.0
    )
    adjusted = problem.titre_value + reactivity[:, None]
    np.testing.assert_allclose(bases, opt.column_bases(adjusted, problem.titre_type, minimum=2.0))


def test_clip_matters_only_when_a_target_is_negative():
    # no shifts: every target is a true distance, so none is negative
    problem, layout, _ = planted_reactivity(seed=6, n_reactive=0)
    kw: dict[str, Any] = dict(penalty=0.0, column_bases="fixed", minimum_column_basis=0.0)
    zero = np.zeros(problem.n_antigens)
    core = problem._core_problem
    clipped = opt._core.reactivity_objective(core, layout, zero, clip=True, **kw)[0]
    assert clipped == opt._core.reactivity_objective(core, layout, zero, clip=False, **kw)[0]
    value = problem.titre_value.copy()
    value[0, 0] = problem.column_bases[0] + 1.5  # a titre above its column basis: target -1.5
    above = with_changes(problem, titre_value=value)._core_problem
    with_clip = opt._core.reactivity_objective(above, layout, zero, clip=True, **kw)[0]
    without = opt._core.reactivity_objective(above, layout, zero, clip=False, **kw)[0]
    assert with_clip != without


def test_reactivity_scan_follows_ae():
    problem, layout, _ = planted_reactivity(seed=7, n_ag=12, n_sr=5, n_reactive=2)
    grid = (1.0, -1.0, 2.0)
    (result,) = opt.antigen_reactivity_scan(problem, layout, adjustments=grid, antigens=[3])
    original = opt.stress(problem, layout)
    for adjust in grid:
        avidity = np.ones(problem.n_points)
        avidity[3] = 2.0**adjust
        expected = (
            opt.optimise(with_changes(problem, avidity_adjust=avidity), layout).stress - original
        )
        assert result.stress_diff[adjust] == pytest.approx(expected, rel=1e-12)
    best_adjust, best_diff = min(result.stress_diff.items(), key=lambda item: item[1])
    assert result.best == (best_adjust if best_diff < 0 else 0.0)


def test_reactivity_inputs_are_checked():
    problem, layout, _ = planted_reactivity(seed=8, n_ag=10, n_sr=4, n_reactive=1)
    with pytest.raises(ValueError, match="one entry per antigen"):
        opt.fit_antigen_reactivity(problem, layout, fixed=np.zeros(3))
    with pytest.raises(ValueError, match="penalty"):
        opt.fit_antigen_reactivity(problem, layout, penalty=-1.0)
    with pytest.raises(ValueError, match="column_bases"):
        opt.fit_antigen_reactivity(problem, layout, column_bases="moving")  # type: ignore[arg-type]
