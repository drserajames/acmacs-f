"""Maps from scratch as AD's chart-relax-grid makes them: every start rough, then the best 5 fine.

Sarah, 2 Oct 2026 ("Yes, match AD"): af had relaxed every scratch start fine, 2.8x the cost per
start on a 7,094-point H1 merge for the same best (06-optimiser).
"""

import numpy as np
import pytest

from af.chain import backend
from af.chain.config import MapOptions, option_parameters
from af.chain.starts import read_problem, write_problem
from af.chain.synthetic import make_tables
from af.chart.ace import read_chart

from .test_engine import GROUP, OPTS

N_STARTS = 12


@pytest.fixture
def arrays(tmp_path):
    pytest.importorskip("af.map.optimise")
    make_tables(tmp_path / "tables", GROUP, n_tables=1)
    [path] = sorted((tmp_path / "tables").iterdir())
    return read_chart(path).optimiser_arrays(
        OPTS.minimum_column_basis, disconnect_threshold=OPTS.disconnect_threshold
    )


def _scratch(arrays, precision):
    opt = backend.CoreOptimiser(threads=1)
    return opt.optimise({**arrays, "scratch_precision": precision}, N_STARTS, 2, 7, None)


def test_rough_is_the_default_and_a_step_parameter_fine_reproduces_old_maps():
    assert MapOptions().scratch_precision == "rough"
    assert option_parameters(MapOptions())["scratch_precision"] == "rough"
    # set to the old method it leaves the parameters as they were: old chains are reused
    assert "scratch_precision" not in option_parameters(MapOptions(scratch_precision="fine"))
    with pytest.raises(ValueError, match="rough or fine"):
        MapOptions(scratch_precision="medium")


def test_rough_refines_only_the_best_five(arrays):
    from af.map.optimise import MapProblem, relax

    rough = _scratch(arrays, "rough")
    problem = backend.CoreOptimiser._problem(arrays)
    assert isinstance(problem, MapProblem)
    plain = relax(problem, n_starts=N_STARTS, seed=7, dimensions=2, precision="rough", keep=None)
    plain_by_start = {p.start_index: p.stress for p in plain.projections}
    refined = {
        m["start_seed"]
        for m in rough
        if m["stress"] != pytest.approx(plain_by_start[m["start_seed"]])
    }
    # the refine (and the trapped pass on the best) change at most the 5 best starts
    assert 1 <= len(refined) <= 5
    assert len(rough) - len(refined) >= N_STARTS - 5


def test_rough_finds_the_same_best_as_all_fine(arrays):
    rough, fine = _scratch(arrays, "rough"), _scratch(arrays, "fine")
    assert rough[0]["stress"] == pytest.approx(fine[0]["stress"], rel=1e-6)


def test_split_gives_the_unsplit_map(arrays):
    opt = backend.CoreOptimiser(threads=1)
    a = {**arrays, "scratch_precision": "rough"}
    whole = opt.optimise(a, N_STARTS, 2, 7, None)
    chunks = opt.relax_chunk(a, 0, 5, 2, 7, None) + opt.relax_chunk(a, 5, N_STARTS - 5, 2, 7, None)
    split = opt.combine(a, chunks, incremental=False)
    assert split[0]["stress"] == whole[0]["stress"]
    assert np.array_equal(split[0]["layout"], whole[0]["layout"])


def test_the_problem_file_carries_it_to_every_chunk(arrays, tmp_path):
    path = write_problem(
        tmp_path / "p.npz",
        {**arrays, "scratch_precision": "rough"},
        seed=7,
        dimensions=2,
        optimiser="core",
        start_layout=None,
    )
    read, *_ = read_problem(path)
    assert read["scratch_precision"] == "rough"
    old = write_problem(
        tmp_path / "q.npz", arrays, seed=7, dimensions=2, optimiser="core", start_layout=None
    )
    assert "scratch_precision" not in read_problem(old)[0]  # absent: every start fine, as before
