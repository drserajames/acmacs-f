"""A split map reuses the finished chunks of the same problem (2 Oct 2026: a run killed after its
array, in the driver's refine, paid for the whole array again; ~830 core-h on one map)."""

import numpy as np
import pytest

from af.chain.backend import StubOptimiser
from af.chain.engine import Mapper, SplitStarts
from af.chain.synthetic import make_tables
from af.chart.ace import read_chart
from af.run.local import LocalRunner

from .test_engine import GROUP, OPTS


class CountingRunner(LocalRunner):
    def __init__(self) -> None:
        super().__init__(max_parallel=2)
        self.ran: list[str] = []

    def run_many(self, jobs):
        self.ran += [j.name for j in jobs]
        return super().run_many(jobs)


@pytest.fixture
def arrays(tmp_path):
    make_tables(tmp_path / "tables", GROUP, n_tables=1)
    [path] = sorted((tmp_path / "tables").iterdir())
    return read_chart(path).optimiser_arrays(
        OPTS.minimum_column_basis, disconnect_threshold=OPTS.disconnect_threshold
    )


def _make(mapper, runner, arrays, seed=11):
    return mapper.make(runner, "testmap/0000/scratch", arrays, 9, 2, seed, None)


def _same(a, b):
    return [m["stress"] for m in a] == [m["stress"] for m in b] and all(
        np.array_equal(x["layout"], y["layout"]) for x, y in zip(a, b, strict=True)
    )


def test_a_rerun_of_the_same_problem_runs_no_job(arrays, tmp_path):
    mapper = Mapper(StubOptimiser(), SplitStarts(chunks=3, work_dir=tmp_path / "work"))
    runner = CountingRunner()
    first = _make(mapper, runner, arrays)
    assert len(runner.ran) == 3
    runner.ran.clear()
    again = _make(mapper, runner, arrays)
    assert runner.ran == [] and _same(first, again)


def test_only_missing_or_broken_chunks_rerun(arrays, tmp_path):
    mapper = Mapper(StubOptimiser(), SplitStarts(chunks=3, work_dir=tmp_path / "work"))
    runner = CountingRunner()
    first = _make(mapper, runner, arrays)
    work = tmp_path / "work" / "testmap" / "0000" / "scratch"
    files = sorted(work.glob("starts-*.npz"))
    files[0].unlink()  # killed before this task finished
    files[2].write_bytes(b"not an npz")  # broken
    runner.ran.clear()
    again = _make(mapper, runner, arrays)
    assert sorted(runner.ran) == ["testmap-0000-scratch-000", "testmap-0000-scratch-002"]
    assert _same(first, again)


def test_another_problem_starts_afresh(arrays, tmp_path):
    mapper = Mapper(StubOptimiser(), SplitStarts(chunks=3, work_dir=tmp_path / "work"))
    runner = CountingRunner()
    _make(mapper, runner, arrays)
    runner.ran.clear()
    _make(mapper, runner, arrays, seed=12)  # a different seed is a different problem
    assert len(runner.ran) == 3
    changed = {**arrays, "titre_value": arrays["titre_value"] + 0.5}
    runner.ran.clear()
    _make(mapper, runner, changed, seed=12)
    assert len(runner.ran) == 3


def test_a_run_killed_after_its_array_reuses_the_chunks(arrays, tmp_path, monkeypatch):
    optimiser = StubOptimiser()
    mapper = Mapper(optimiser, SplitStarts(chunks=3, work_dir=tmp_path / "work"))
    runner = CountingRunner()
    real = optimiser.combine

    def killed(*a, **k):
        raise KeyboardInterrupt  # SIGTERM in the driver's refine, after the array returned

    monkeypatch.setattr(optimiser, "combine", killed)
    with pytest.raises(KeyboardInterrupt):
        _make(mapper, runner, arrays)
    assert len(runner.ran) == 3
    monkeypatch.setattr(optimiser, "combine", real)
    runner.ran.clear()
    resumed = _make(mapper, runner, arrays)
    assert runner.ran == []  # the array is not paid for again
    fresh = Mapper(StubOptimiser(), SplitStarts(chunks=3, work_dir=tmp_path / "fresh"))
    assert _same(resumed, _make(fresh, CountingRunner(), arrays))
