"""The chain keeps maps made under different arithmetics apart.

The optimiser names its arithmetic (`af.map.optimise.NUMERICS`): builds with the same name give the
same bits from the same problem on any machine, builds with different names need not. So the name
is part of what a chunk, a step and a reference check are reused on, and every record says it. An
optimiser that names none (the numpy stand-in) keeps every digest, parameter and record it had.
"""

import json

import numpy as np
import pytest

from af.chain import backend
from af.chain.backend import StubOptimiser, map_order
from af.chain.engine import ChainError, Mapper, SplitStarts, chain_steps, platform_info, run_chain
from af.chain.starts import (
    problem_digest,
    read_problem,
    result_numerics,
    run_chunk,
    write_problem,
    write_result,
)
from af.run.job import JobFailed
from af.run.local import LocalRunner

from .test_engine import GROUP, config, tables  # noqa: F401  (tables is a fixture)

NAMES = ("plain", "named")

# what the digest of this problem was before the arithmetic had a name (master dc1ded4)
BEFORE = {
    "stub": "3fe260b7dd375e4e50100f3592704e09316eae311cd2e9c39a7eaa32794e1ff1",
    "core": "31312c661ec5fa1f0038b960d7d671cdfcdac399949dff934f5a7fb2bd5879a9",
}


class Named(StubOptimiser):
    """The stand-in, claiming an arithmetic: what the core is to the chain, without the core."""

    numerics = "test-fp-1"


@pytest.fixture
def arrays():
    return {
        "titre_value": np.array([[5.0, 3.0], [2.0, 6.0], [4.0, 1.0]]),
        "titre_type": np.array([[1, 1], [1, 2], [1, 1]], dtype=np.int8),
        "column_bases": np.array([5.0, 6.0]),
        "disconnected": np.zeros(5, dtype=np.bool_),
    }


def _digest(arrays, optimiser, numerics):
    return problem_digest(
        arrays, seed=3, dimensions=2, optimiser=optimiser, numerics=numerics, start_layout=None
    )


@pytest.mark.parametrize("optimiser", ["stub", "core"])
def test_a_problem_that_names_no_arithmetic_has_the_digest_it_always_had(arrays, optimiser):
    assert _digest(arrays, optimiser, None) == BEFORE[optimiser]


def test_another_arithmetic_is_another_problem(arrays):
    one, two = _digest(arrays, "core", "test-fp-1"), _digest(arrays, "core", "test-fp-2")
    assert len({one, two, BEFORE["core"]}) == 3


def _problem(path, arrays, numerics):
    return write_problem(
        path, arrays, seed=3, dimensions=2, optimiser="stub", numerics=numerics, start_layout=None
    )


def test_the_problem_file_says_which_arithmetic_it_was_written_for(arrays, tmp_path):
    assert read_problem(_problem(tmp_path / "a.npz", arrays, "test-fp-1"))[4] == "test-fp-1"
    assert read_problem(_problem(tmp_path / "b.npz", arrays, None))[4] is None


def test_a_chunk_refuses_a_problem_written_for_another_arithmetic(arrays, tmp_path):
    # the task's build is not the driver's: an environment that changed under a run
    problem = _problem(tmp_path / "p.npz", arrays, "test-fp-1")
    with pytest.raises(RuntimeError, match="written for numerics 'test-fp-1'.*has None"):
        run_chunk(problem, 0, 2, tmp_path / "r.npz")
    assert not (tmp_path / "r.npz").exists()
    done = run_chunk(_problem(tmp_path / "q.npz", arrays, None), 0, 2, tmp_path / "s.npz")
    assert result_numerics(done) is None


def _make(mapper, arrays):
    return mapper.make(LocalRunner(max_parallel=2), "testmap/0000/scratch", arrays, 6, 2, 3, None)


def test_the_driver_refuses_a_chunk_made_under_another_arithmetic(arrays, tmp_path):
    mapper = Mapper(StubOptimiser(), SplitStarts(chunks=3, work_dir=tmp_path / "work"))
    first = _make(mapper, arrays)
    [*_, last] = sorted((tmp_path / "work/testmap/0000/scratch").glob("starts-*.npz"))
    maps = [m for m in first if m["start_seed"] >= 4]
    write_result(last, maps, "test-fp-1")  # the same starts, from a task on another build
    with pytest.raises(ChainError, match=r"1 of 3 chunk\(s\) were made under numerics"):
        _make(mapper, arrays)


def test_chunks_made_before_the_name_are_not_reused_under_it(arrays, tmp_path):
    ran: list[str] = []

    class Counting(LocalRunner):
        def run_many(self, jobs):
            ran.extend(j.name for j in jobs)
            return super().run_many(jobs)

    split = SplitStarts(chunks=3, work_dir=tmp_path / "work")
    Mapper(StubOptimiser(), split).make(Counting(), "m/0000/scratch", arrays, 6, 2, 3, None)
    assert len(ran) == 3
    # the same problem under a named arithmetic: the directory starts afresh, and its tasks (the
    # plain stand-in, which names none) are refused rather than combined
    with pytest.raises(JobFailed):
        Mapper(Named(), split).make(Counting(), "m/0000/scratch", arrays, 6, 2, 3, None)
    assert len(ran) == 6


def test_a_step_is_remade_under_another_arithmetic(tables, tmp_path):  # noqa: F811
    cfg = config(tables)
    plain = chain_steps(cfg, tmp_path / "x", Mapper(StubOptimiser(), None))
    named = chain_steps(cfg, tmp_path / "x", Mapper(Named(), None))
    assert all("numerics" not in s.parameters for s in plain)  # parameters as they always were
    assert [s.parameters["numerics"] for s in named] == ["test-fp-1"] * len(plain)
    assert [{k: v for k, v in s.parameters.items() if k != "numerics"} for s in named] == [
        dict(s.parameters) for s in plain
    ]
    root = tmp_path / "s"
    assert [r.reused for r in run_chain(cfg, root, optimiser=StubOptimiser())] == [False] * 4
    assert [r.reused for r in run_chain(cfg, root, optimiser=Named())] == [False] * 4
    assert [r.reused for r in run_chain(cfg, root, optimiser=Named())] == [True] * 4


def _records(root):
    chain = json.loads(next(root.rglob("chain.json")).read_text())
    return chain, [json.loads(p.read_text())["platform"] for p in sorted(root.rglob("step.json"))]


def test_every_record_names_the_arithmetic_when_the_optimiser_does(tables, tmp_path):  # noqa: F811
    cfg = config(tables)
    run_chain(cfg, tmp_path / "plain", optimiser=StubOptimiser())
    run_chain(cfg, tmp_path / "named", optimiser=Named())
    chain, platforms = _records(tmp_path / "plain")
    assert "numerics" not in chain and all("numerics" not in p for p in platforms)
    chain, platforms = _records(tmp_path / "named")
    assert chain["numerics"] == "test-fp-1"
    assert [p["numerics"] for p in platforms] == ["test-fp-1"] * 4
    assert "numerics" not in platform_info("site-a")
    assert platform_info(None, "test-fp-1")["numerics"] == "test-fp-1"


def test_a_published_chain_carries_it_in_its_parameters(tables, tmp_path):  # noqa: F811
    from af.chain.publish import publish_chain
    from af.chain.review import build_review
    from af.store import Store

    def parameters(optimiser, name):
        cfg = config(tables)
        run_chain(cfg, tmp_path / name, optimiser=optimiser)
        build_review(tmp_path / name / cfg.name)
        store = Store.create(tmp_path / f"store-{name}")
        ref = publish_chain(
            tmp_path / f"store-{name}", "testlab/h9-hi-test-lab/main", tmp_path / name / cfg.name
        )
        return json.loads((store.resolve(ref) / "PROVENANCE.json").read_text())["parameters"]

    assert "numerics" not in parameters(StubOptimiser(), "plain")
    assert parameters(Named(), "named")["numerics"] == "test-fp-1"
    pages = {n: (tmp_path / n / GROUP / "review" / "index.html").read_text() for n in NAMES}
    assert "numerics" not in pages["plain"] and "numerics test-fp-1" in pages["named"]


def test_the_core_names_the_compiled_arithmetic(arrays, tmp_path):
    optimise = pytest.importorskip("af.map.optimise")
    assert backend.CoreOptimiser().numerics == optimise.NUMERICS != ""
    assert StubOptimiser().numerics is None
    path = write_problem(
        tmp_path / "p.npz",
        arrays,
        seed=3,
        dimensions=2,
        optimiser="core",
        numerics=optimise.NUMERICS,
        start_layout=None,
    )
    assert (
        result_numerics(run_chunk(path, 0, 2, tmp_path / "r.npz", threads=1)) == optimise.NUMERICS
    )
    before = write_problem(
        tmp_path / "q.npz",
        arrays,
        seed=3,
        dimensions=2,
        optimiser="core",
        numerics=None,
        start_layout=None,
    )
    with pytest.raises(RuntimeError, match="written for numerics None"):  # a problem from before
        run_chunk(before, 0, 2, tmp_path / "s.npz", threads=1)


def _map(stress, start):
    return {"layout": np.zeros((1, 2)), "stress": stress, "start_seed": start}


def test_tied_maps_rank_by_start_index_whatever_order_they_arrive_in():
    maps = [_map(2.0, 7), _map(float("nan"), 0), _map(1.0, 5), _map(1.0, 2)]
    order = [m["start_seed"] for m in sorted(maps, key=map_order)]
    assert order == [2, 5, 7, 0]  # stress, ties by start index, NaN last: the core's order
    assert [m["start_seed"] for m in sorted(reversed(maps), key=map_order)] == order
    assert [m["start_seed"] for m in StubOptimiser().combine({}, maps, False)] == order


def test_the_map_kept_after_the_trapped_loop_does_not_depend_on_arrival_order(monkeypatch):
    optimise = pytest.importorskip("af.map.optimise")
    from types import SimpleNamespace

    def resolve(problem, layout, **kwargs):  # every map resolves to the same stress: a tie
        return SimpleNamespace(
            projection=SimpleNamespace(layout=layout + 1.0, stress=1.0),
            rounds=1,
            moved=1,
            groups=[],
            threads=1,
        )

    monkeypatch.setattr(optimise, "resolve_trapped", resolve)
    maps = [{**_map(3.0, 9), "threads": 1, "cpus": 1}, {**_map(4.0, 4), "threads": 1, "cpus": 1}]
    out = backend.CoreOptimiser(threads=1)._resolve(None, maps, False, resolve_best=2)
    assert [m["start_seed"] for m in out] == [4, 9]  # tied at 1.0: the lower start index first
    assert "run_threads" in out[0] and out[0]["refined_rank"] == 1
