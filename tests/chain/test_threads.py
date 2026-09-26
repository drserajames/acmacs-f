"""step.json records the threads each relax and trapped pass ran on, and the CPUs it had."""

import numpy as np
import pytest

from af.chain import backend
from af.chain.diagnostics import flags, run_threads
from af.chain.starts import read_result, write_result
from af.chain.synthetic import make_tables
from af.chart.ace import read_chart

from .test_engine import GROUP, OPTS


def one(threads, cpus):
    return {
        "relax": [{"threads": threads, "cpus": cpus, "starts": 4}],
        "trapped": {"threads": threads, "cpus": cpus},
    }


def test_single_thread_with_cpus_free_is_flagged():
    d = run_threads({"incremental": one(1, 8), "scratch": one(4, 4)})
    assert flags(d) == [
        "single-threaded with CPUs free: incremental relax (8 CPUs), incremental trapped (8 CPUs)"
    ]


def test_single_thread_on_one_cpu_is_not_flagged():
    assert flags(run_threads({"scratch": one(1, 1)})) == []


def test_nothing_recorded_reads_as_before():
    assert run_threads({"incremental": None, "scratch": None}) == {}


def test_count_runs_groups_chunks_by_node_kind():
    runs = [{"threads": 4, "cpus": 4}] * 3 + [{"threads": 1, "cpus": 32}]
    assert backend._count_runs(runs) == [
        {"threads": 1, "cpus": 32, "starts": 1},
        {"threads": 4, "cpus": 4, "starts": 3},
    ]


def test_chunk_file_keeps_threads_and_cpus(tmp_path):
    maps = [
        {
            "layout": np.zeros((2, 2)),
            "stress": s,
            "dimensions": 2,
            "n_iterations": 1,
            "start_seed": i,
            "threads": 4,
            "cpus": 16,
        }
        for i, s in enumerate([1.0, 2.0])
    ]
    back = read_result(write_result(tmp_path / "r.npz", maps))
    assert [(m["threads"], m["cpus"]) for m in back] == [(4, 16), (4, 16)]


@pytest.fixture
def arrays(tmp_path):
    pytest.importorskip("af.map.optimise")
    make_tables(tmp_path / "tables", GROUP, n_tables=1)
    [path] = sorted((tmp_path / "tables").iterdir())
    return read_chart(path).optimiser_arrays(
        OPTS.minimum_column_basis, disconnect_threshold=OPTS.disconnect_threshold
    )


def test_core_records_threads_unsplit_and_split(arrays):
    from af.map.optimise import available_cpus

    opt = backend.CoreOptimiser(threads=2)
    whole = opt.optimise(arrays, 6, 2, 7, None)
    chunks = opt.relax_chunk(arrays, 0, 4, 2, 7, None) + opt.relax_chunk(arrays, 4, 2, 2, 7, None)
    split = opt.combine(arrays, chunks, incremental=False)
    cpus = available_cpus()
    for maps in (whole, split):
        [best] = [m for m in maps if "run_threads" in m]
        assert best["stress"] == min(m["stress"] for m in maps)
        t = best["run_threads"]
        assert [r["starts"] for r in t["relax"]] == [6]
        assert t["relax"][0]["cpus"] == cpus and t["trapped"]["cpus"] == cpus
        assert t["relax"][0]["threads"] >= 1 and t["trapped"]["threads"] >= 1
