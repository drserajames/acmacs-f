"""af.pipeline runs independent steps side by side, up to max_parallel, in dependency order."""

import os
import signal
import threading
import time
from pathlib import Path

import pytest

from af.pipeline import Pipeline, Step, StepContext
from af.run import Job, LocalRunner
from af.run.job import Terminated
from af.util.artefacts import Artefact


class Tracker:
    """Records when each step's action runs, and the most running at once."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.running = 0
        self.peak = 0
        self.spans: dict[str, tuple[float, float]] = {}

    def step(
        self,
        name: str,
        out: Path,
        *,
        inputs: list[Path] | None = None,
        seconds: float = 0.5,
        fail: bool = False,
    ) -> Step:
        def action(context: StepContext) -> None:
            with self.lock:
                self.running += 1
                self.peak = max(self.peak, self.running)
            start = time.monotonic()
            try:
                time.sleep(seconds)
                if fail:
                    raise RuntimeError(f"{name} failed")
                out.write_text(name)
            finally:
                with self.lock:
                    self.running -= 1
                self.spans[name] = (start, time.monotonic())

        return Step(name, action, outputs=[Artefact(out)], inputs=inputs or [])


def pipeline(steps: list[Step], tmp_path: Path, max_parallel: int) -> Pipeline:
    return Pipeline(
        steps,
        state_dir=tmp_path / "state",
        runner=LocalRunner(),
        root=tmp_path,
        max_parallel=max_parallel,
    )


def trees(tracker: Tracker, tmp_path: Path) -> list[Step]:
    return [tracker.step(f"tree-{s}", tmp_path / f"{s}.nwk") for s in ("h1", "h3", "bvic")]


def test_independent_steps_run_side_by_side(tmp_path: Path) -> None:
    tracker = Tracker()
    started = time.monotonic()
    outcomes = pipeline(trees(tracker, tmp_path), tmp_path, max_parallel=3).run()
    elapsed = time.monotonic() - started
    assert [o.status for o in outcomes] == ["ran"] * 3
    assert tracker.peak == 3
    assert elapsed < 1.2, f"three 0.5 s steps took {elapsed:.2f} s: not parallel"


def test_serial_by_default(tmp_path: Path) -> None:
    tracker = Tracker()
    steps = trees(tracker, tmp_path)
    Pipeline(steps, state_dir=tmp_path / "state", runner=LocalRunner(), root=tmp_path).run()
    assert tracker.peak == 1


def test_max_parallel_is_a_cap(tmp_path: Path) -> None:
    tracker = Tracker()
    steps = [tracker.step(f"s{i}", tmp_path / f"s{i}", seconds=0.2) for i in range(5)]
    pipeline(steps, tmp_path, max_parallel=2).run()
    assert tracker.peak == 2


def test_dependencies_wait(tmp_path: Path) -> None:
    tracker = Tracker()
    steps = trees(tracker, tmp_path)
    figure = tracker.step(
        "figure",
        tmp_path / "fig.pdf",
        inputs=[tmp_path / "h1.nwk", tmp_path / "h3.nwk"],
        seconds=0.1,
    )
    run = pipeline([figure, *steps], tmp_path, max_parallel=4)
    outcomes = run.run()
    assert [o.name for o in outcomes] == [step.name for step in run.order()], "stable order"
    names = [o.name for o in outcomes]
    assert names.index("figure") > max(names.index("tree-h1"), names.index("tree-h3"))
    figure_start = tracker.spans["figure"][0]
    assert figure_start >= tracker.spans["tree-h1"][1]
    assert figure_start >= tracker.spans["tree-h3"][1]


def test_failure_stops_new_steps_but_lets_running_ones_finish(tmp_path: Path) -> None:
    tracker = Tracker()
    fails = tracker.step("tree-h1", tmp_path / "h1.nwk", seconds=0.1, fail=True)
    slow = tracker.step("tree-h3", tmp_path / "h3.nwk", seconds=0.6)
    after = tracker.step("figure", tmp_path / "fig.pdf", inputs=[tmp_path / "h1.nwk"], seconds=0.1)
    with pytest.raises(RuntimeError, match="tree-h1 failed"):
        pipeline([fails, slow, after], tmp_path, max_parallel=2).run()
    assert "figure" not in tracker.spans, "a step depending on the failure never started"
    assert (tmp_path / "h3.nwk").exists() and (tmp_path / "state" / "tree-h3.json").exists()

    # The next run re-runs only what failed: the finished tree is skipped.
    tracker2 = Tracker()
    fixed = tracker2.step("tree-h1", tmp_path / "h1.nwk", seconds=0.1)
    slow2 = tracker2.step("tree-h3", tmp_path / "h3.nwk", seconds=0.6)
    after2 = tracker2.step(
        "figure", tmp_path / "fig.pdf", inputs=[tmp_path / "h1.nwk"], seconds=0.1
    )
    outcomes = pipeline([fixed, slow2, after2], tmp_path, max_parallel=2).run()
    assert [(o.name, o.status) for o in outcomes] == [
        ("tree-h1", "ran"),
        ("tree-h3", "skipped"),
        ("figure", "ran"),
    ]


def test_sigterm_during_parallel_steps_stops_their_jobs(tmp_path: Path) -> None:
    """Steps wait on runner jobs in worker threads; SIGTERM must still stop the jobs."""

    def tree(subtype: str) -> Step:
        out = tmp_path / f"{subtype}.nwk"

        def action(context: StepContext) -> None:
            context.runner.run(
                Job(
                    f"build-{subtype}",
                    ["/bin/sh", "-c", f"sleep 60; echo x > {out}"],
                    tmp_path,
                    tmp_path / f"{subtype}.log",
                    [Artefact(out)],
                )
            )

        return Step(f"tree-{subtype}", action, outputs=[Artefact(out)])

    threading.Timer(1.0, os.kill, (os.getpid(), signal.SIGTERM)).start()
    started = time.monotonic()
    with pytest.raises(Terminated):
        Pipeline(
            [tree("h1"), tree("h3")],
            state_dir=tmp_path / "state",
            runner=LocalRunner(max_parallel=2),
            root=tmp_path,
            max_parallel=2,
        ).run()
    assert time.monotonic() - started < 20
    time.sleep(0.5)
    assert not list(tmp_path.glob("*.nwk")), "the tree jobs were stopped"
