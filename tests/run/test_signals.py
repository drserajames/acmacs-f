"""A driver told to stop (Ctrl-C, SIGTERM, SIGHUP) stops its jobs before it exits.

Real signals are sent to the test process while a runner waits. The waits run in
worker threads and the signal arrives in the main thread; cancelling must still reach
every job in flight.
"""

import os
import signal
import threading
import time
from pathlib import Path

import pytest

from af.run import Job, LocalRunner, SlurmRunner
from af.run.job import Terminated
from af.util.artefacts import Artefact

SIGNALS = [signal.SIGTERM, signal.SIGHUP, signal.SIGINT]
EXPECTED = (Terminated, KeyboardInterrupt)


def sleeper(tmp_path: Path, name: str) -> Job:
    return Job(
        name,
        ["/bin/sh", "-c", "sleep 60; echo late > late-" + name],
        tmp_path,
        tmp_path / f"{name}.log",
        [Artefact(tmp_path / ("late-" + name))],
    )


def signal_after(seconds: float, signum: int) -> threading.Timer:
    timer = threading.Timer(seconds, os.kill, (os.getpid(), signum))
    timer.start()
    return timer


@pytest.mark.parametrize("signum", SIGNALS, ids=lambda s: signal.Signals(s).name)
def test_slurm_driver_cancels_jobs_on_signal(
    tmp_path: Path, fake_sbatch: Path, fake_scancel: Path, signum: int
) -> None:
    runner = SlurmRunner(
        work_dir=tmp_path / "slurm", sbatch=str(fake_sbatch), scancel=str(fake_scancel)
    )
    jobs = [sleeper(tmp_path, "a"), sleeper(tmp_path, "b")]
    jobs[1] = Job(
        jobs[1].name,
        jobs[1].command,
        jobs[1].cwd,
        jobs[1].log,
        jobs[1].outputs,
        resources=jobs[1].resources.__class__(threads=2),
    )  # two separate arrays
    started = time.monotonic()
    signal_after(1.0, signum)
    with pytest.raises(EXPECTED):
        runner.run_many(jobs)
    assert time.monotonic() - started < 20, "the driver waited for the jobs instead of cancelling"
    cancelled = (fake_sbatch.parent / "scancel-calls.txt").read_text().split()
    assert len(cancelled) == 2, "both in-flight arrays cancelled"
    assert all(name.startswith("--name=af-") for name in cancelled)
    assert not list(tmp_path.glob("late-*")), "no job ran on after the driver stopped"


@pytest.mark.parametrize("signum", SIGNALS, ids=lambda s: signal.Signals(s).name)
def test_local_driver_terminates_children_on_signal(tmp_path: Path, signum: int) -> None:
    runner = LocalRunner(max_parallel=2)
    started = time.monotonic()
    signal_after(1.0, signum)
    with pytest.raises(EXPECTED):
        runner.run_many([sleeper(tmp_path, "a"), sleeper(tmp_path, "b")])
    assert time.monotonic() - started < 20
    time.sleep(0.5)
    assert not list(tmp_path.glob("late-*"))


def test_handlers_restored_after_run(tmp_path: Path) -> None:
    before = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGHUP)}
    out = tmp_path / "x"
    LocalRunner().run(
        Job("x", ["/bin/sh", "-c", "echo 1 > x"], tmp_path, tmp_path / "x.log", [Artefact(out)])
    )
    assert {sig: signal.getsignal(sig) for sig in before} == before
