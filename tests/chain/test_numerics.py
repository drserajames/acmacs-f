"""The chain keeps maps made under different arithmetics apart.

The optimiser names its arithmetic (`af.map.optimise.NUMERICS`): builds with the same name give the
same bits from the same problem on any machine, builds with different names need not. So the name
is part of what a chunk, a step and a reference check are reused on, and every record says it.

One build offers two arithmetics: the named one and the original, which has no name because its
results belong to one machine. A run says which in its config. Nothing is written for the original
(or for the numpy stand-in), so maps made before arithmetics had names keep every digest, step
parameter and record they had, and a run under the original reuses them.
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
    with pytest.raises(RuntimeError, match="written for numerics 'test-fp-1', which this build"):
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


def _core_problem(path, arrays, numerics):
    return write_problem(
        path, arrays, seed=3, dimensions=2, optimiser="core", numerics=numerics, start_layout=None
    )


def test_the_core_offers_the_named_arithmetic_and_the_original(arrays, tmp_path):
    optimise = pytest.importorskip("af.map.optimise")
    strict, original = backend.CoreOptimiser(), backend.CoreOptimiser(arithmetic=optimise.ORIGINAL)
    assert strict.numerics == optimise.NUMERICS != ""  # the default, as MapProblem's
    assert backend.CoreOptimiser(arithmetic=optimise.NUMERICS).numerics == optimise.NUMERICS
    assert (
        original.numerics is None and StubOptimiser().numerics is None
    )  # no name: nothing written
    assert strict._problem(arrays).numerics == optimise.NUMERICS
    assert original._problem(arrays).numerics == optimise.ORIGINAL
    with pytest.raises(ValueError, match="unknown optimiser arithmetic 'other-fp-9'"):
        backend.CoreOptimiser(arithmetic="other-fp-9")
    # the original's problems are the problems of before: the same digest, so their chunks resume
    assert _digest(arrays, "core", original.numerics) == BEFORE["core"]
    assert _digest(arrays, "core", strict.numerics) != BEFORE["core"]


def test_a_chunk_runs_the_core_its_problem_was_written_for(arrays, tmp_path):
    optimise = pytest.importorskip("af.map.optimise")
    named = _core_problem(tmp_path / "p.npz", arrays, optimise.NUMERICS)
    assert (
        result_numerics(run_chunk(named, 0, 2, tmp_path / "r.npz", threads=1)) == optimise.NUMERICS
    )
    # no name: the original arithmetic, as a problem written before names existed
    before = _core_problem(tmp_path / "q.npz", arrays, None)
    assert result_numerics(run_chunk(before, 0, 2, tmp_path / "s.npz", threads=1)) is None
    other = _core_problem(tmp_path / "x.npz", arrays, "other-fp-9")  # a build this one is not
    with pytest.raises(RuntimeError, match="written for numerics 'other-fp-9', which this build"):
        run_chunk(other, 0, 2, tmp_path / "t.npz", threads=1)
    assert not (tmp_path / "t.npz").exists()


def _core_records(root):
    chain = json.loads(next(root.rglob("chain.json")).read_text())
    return chain, [json.loads(p.read_text()) for p in sorted(root.rglob("step.json"))]


def test_a_run_under_the_original_writes_what_runs_always_wrote(tables, tmp_path):  # noqa: F811
    optimise = pytest.importorskip("af.map.optimise")
    cfg = config(tables)
    original = backend.CoreOptimiser(threads=1, arithmetic=optimise.ORIGINAL)
    strict = backend.CoreOptimiser(threads=1)
    plain = chain_steps(cfg, tmp_path / "x", Mapper(original, None))
    named = chain_steps(cfg, tmp_path / "x", Mapper(strict, None))
    assert all("numerics" not in s.parameters for s in plain)  # so existing steps are reused
    assert [s.parameters["numerics"] for s in named] == [optimise.NUMERICS] * len(named)
    assert all("refine_best" not in s.parameters for s in (*plain, *named))  # never a parameter

    run_chain(cfg, tmp_path / "original", optimiser=original)
    run_chain(cfg, tmp_path / "strict", optimiser=strict)
    chain, steps = _core_records(tmp_path / "original")
    assert "numerics" not in chain and all("numerics" not in s["platform"] for s in steps)
    chain, steps = _core_records(tmp_path / "strict")
    assert chain["numerics"] == optimise.NUMERICS
    assert [s["platform"]["numerics"] for s in steps] == [optimise.NUMERICS] * 4
    # one root, the arithmetic changed: every step is made again, then reused
    root = tmp_path / "original"
    assert [r.reused for r in run_chain(cfg, root, optimiser=original)] == [True] * 4
    assert [r.reused for r in run_chain(cfg, root, optimiser=strict)] == [False] * 4


def test_a_step_says_how_its_start_stresses_are_to_be_read(tables, tmp_path):  # noqa: F811
    optimise = pytest.importorskip("af.map.optimise")
    core = backend.CoreOptimiser(threads=1)
    assert core.refine_best == optimise.INCREMENTAL_REFINED == 5  # one copy: the optimiser's
    assert StubOptimiser().refine_best is None
    run_chain(config(tables), tmp_path / "core", optimiser=core)
    chain, [first, *later] = _core_records(tmp_path / "core")
    assert chain["refine_best"] == 5
    # 4 starts from scratch, rough, then the best 5 refined fine: here all 4
    assert first["fine_starts"] == {"scratch": 4} and first["refine_best"] == 5
    for step in later:  # and 3 incremental starts, likewise
        assert step["fine_starts"] == {"incremental": 3, "scratch": 4}
        assert step["refine_best"] == 5
    for step in (first, *later):
        for name, loop in step["trapped_loop"].items():
            # before the loop the kept map was no lower than after, and is what the list held
            assert loop["stress_before"] >= step["stress"][name]
            if loop["moves"] == 0:
                assert loop["stress_before"] == step["stress"][name]
    run_chain(config(tables), tmp_path / "stub", optimiser=StubOptimiser())
    chain, steps = _core_records(tmp_path / "stub")
    assert "refine_best" not in chain
    for step in steps:  # the stand-in records neither: not recorded, never zero
        assert "fine_starts" not in step and "refine_best" not in step
        assert all("stress_before" not in loop for loop in step["trapped_loop"].values())


def test_rough_starts_say_how_many_were_refined(arrays):
    pytest.importorskip("af.map.optimise")
    core = backend.CoreOptimiser(threads=1)
    rough = core.optimise({**arrays, "scratch_precision": "rough"}, 8, 2, 3, None)
    assert rough[0]["fine_starts"] == 5 and rough[0]["refine_best"] == 5
    fine = core.optimise(arrays, 8, 2, 3, None)
    assert fine[0]["fine_starts"] == 8 and "refine_best" not in fine[0]
    chunks = core.relax_chunk({**arrays, "scratch_precision": "rough"}, 0, 8, 2, 3, None)
    split = core.combine({**arrays, "scratch_precision": "rough"}, chunks, incremental=False)
    assert split[0]["fine_starts"] == 5 and split[0]["refine_best"] == 5
    assert split[0]["stress_before_resolve"] == rough[0]["stress_before_resolve"]


def test_a_run_with_the_core_must_name_its_arithmetic():
    optimise = pytest.importorskip("af.map.optimise")
    from types import SimpleNamespace

    from af.chain.__main__ import _optimiser, arithmetic
    from af.chain.config import ChainConfigError

    def run(optimiser, numerics):
        return SimpleNamespace(optimiser=optimiser, threads=1, numerics=numerics)

    with pytest.raises(ChainConfigError, match="must name the optimiser's arithmetic"):
        _optimiser(run("core", None))
    with pytest.raises(ChainConfigError, match="numerics = 'other-fp-9': this build offers"):
        arithmetic(run("core", "other-fp-9"))
    assert _optimiser(run("core", optimise.NUMERICS)).numerics == optimise.NUMERICS
    assert _optimiser(run("core", optimise.ORIGINAL)).numerics is None
    assert _optimiser(run("stub", None)).numerics is None  # the stand-in has none to name


def test_a_reference_check_is_cached_on_the_named_arithmetic_only(tables, tmp_path):  # noqa: F811
    optimise = pytest.importorskip("af.map.optimise")
    import dataclasses
    from types import SimpleNamespace

    from af.chain.__main__ import measure_references
    from af.chain.config import Reference
    from af.chart.ace import read_chart, write_chart

    from .test_reference import _merged_map

    cfg, root, chosen = _merged_map(tables, tmp_path / "s")
    write_chart(read_chart(chosen), tmp_path / "ref.ace")
    cfg = dataclasses.replace(cfg, references=[Reference("the round", "ref.ace", starts=2)])

    def measured(numerics):
        run = SimpleNamespace(references=tmp_path, threads=1, numerics=numerics)
        measure_references(cfg, run, root)  # type: ignore[arg-type]
        return json.loads((root / "chain.json").read_text())["references"][0]

    assert "numerics" not in measured(optimise.ORIGINAL)  # as a check cached before names existed
    assert measured(optimise.NUMERICS)["numerics"] == optimise.NUMERICS  # and measured again


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
    # each map the loop ran on keeps the stress it had before: what the other starts compare to
    assert [m["stress_before_resolve"] for m in out] == [4.0, 3.0]
    assert "fine_starts" not in out[0]  # the caller did not say how the maps were made
