"""Two sharper checks of maps from scratch made as AD's chart-relax-grid does (every start rough,
then the best 5 fine; test_scratch_precision.py has the rest).

1. Exactly the best five rough starts are refined: every other start comes back identical to a
   plain rough relax (layout, stress, iterations, termination), not merely close in stress.
2. The same best as all-fine on surfaces with many minima. Measured on invented tables
   (30 antigens x 8 sera, rounded titres with noise, "<" where below the threshold): rough + refine
   found the all-fine best on 12 of 12 seeds. Seeds 5 and 6 are the hardest: 55 and 58 distinct
   minima among 60 fine starts, and only 2% of starts reach the best.
"""

import numpy as np
import pytest

from af.chain import backend

N_STARTS = 60
SEED = 3


def many_minima(seed: int, n_ag: int = 30, n_sr: int = 8, noise: float = 0.3) -> dict:
    """I1 arrays of an invented table: titres from known positions, rounded, with noise."""
    rng = np.random.default_rng(seed)
    truth = rng.uniform(-4.0, 4.0, (n_ag + n_sr, 2))
    dist = np.linalg.norm(truth[:n_ag, None] - truth[None, n_ag:], axis=2)
    colbase = rng.uniform(6.0, 9.0, n_sr)
    value = np.round(colbase[None, :] - dist + rng.normal(0.0, noise, dist.shape))
    kind = np.ones(value.shape, dtype=np.int8)
    kind[value < 0] = 2  # "<": below the lowest dilution
    value[value < 0] = 0.0
    return dict(
        titre_value=value,
        titre_type=kind,
        column_bases=colbase,
        disconnected=np.zeros(n_ag + n_sr, dtype=bool),
        dodgy_is_regular=False,
    )


def _scratch(arrays: dict, precision: str) -> list[dict]:
    opt = backend.CoreOptimiser(threads=1)
    return opt.optimise({**arrays, "scratch_precision": precision}, N_STARTS, 2, SEED, None)


def test_only_the_best_five_rough_starts_are_refined_the_rest_come_back_unchanged():
    pytest.importorskip("af.map.optimise")
    from af.map.optimise import relax

    arrays = many_minima(6)
    maps = {m["start_seed"]: m for m in _scratch(arrays, "rough")}
    plain = relax(
        backend.CoreOptimiser._problem(arrays),
        n_starts=N_STARTS,
        seed=SEED,
        dimensions=2,
        precision="rough",
        keep=None,
        threads=1,
    )
    best_five = {p.start_index for p in plain.projections[:5]}
    assert len(maps) == N_STARTS
    for p in plain.projections:
        m = maps[p.start_index]
        if p.start_index in best_five:
            assert m["n_iterations"] > p.n_iterations  # refined: more iterations, fine stop
        else:
            np.testing.assert_array_equal(m["layout"], p.layout)
            assert (m["stress"], m["n_iterations"], m["termination"]) == (
                p.stress,
                p.n_iterations,
                p.termination,
            )


@pytest.mark.parametrize("seed", [5, 6])
def test_rough_finds_the_all_fine_best_where_minima_are_many(seed):
    pytest.importorskip("af.map.optimise")
    arrays = many_minima(seed)
    fine = _scratch(arrays, "fine")
    # the surface really is hard: most starts end in a minimum of their own
    assert len({round(m["stress"], 3) for m in fine}) > 40
    rough = _scratch(arrays, "rough")
    assert rough[0]["stress"] == pytest.approx(fine[0]["stress"], rel=1e-6)
