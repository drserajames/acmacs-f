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


def two_arrays(tmp_path: Path) -> list[Job]:
    """Sleepers "a" and "b" with different resources, so they go as two separate arrays."""
    b = sleeper(tmp_path, "b")
    b = Job(b.name, b.command, b.cwd, b.log, b.outputs, resources=b.resources.__class__(threads=2))
    return [sleeper(tmp_path, "a"), b]


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
    started = time.monotonic()
    signal_after(1.0, signum)
    with pytest.raises(EXPECTED):
        runner.run_many(two_arrays(tmp_path))
    assert time.monotonic() - started < 20, "the driver waited for the jobs instead of cancelling"
    cancelled = scancel_counts(fake_sbatch)
    assert len(cancelled) == 2, "both in-flight arrays cancelled"
    assert all(name.startswith("--name=af-") for name in cancelled)
    assert not list(tmp_path.glob("late-*")), "no job ran on after the driver stopped"


def scancel_counts(fake_sbatch: Path) -> dict[str, int]:
    calls = (fake_sbatch.parent / "scancel-calls.txt").read_text().split()
    return {name: calls.count(name) for name in calls}


def marker_passes(tmp_path: Path, job: str) -> list[str]:
    (marker,) = [
        m
        for m in (tmp_path / "slurm").glob("batch-*/CANCELLED")
        if f"scancel --name=af-{job}-" in m.read_text()
    ]
    return [line for line in marker.read_text().splitlines() if ", pass " in line]


@pytest.mark.parametrize("signum", SIGNALS, ids=lambda s: signal.Signals(s).name)
def test_slurm_cancel_repeats_until_a_late_job_stops(
    tmp_path: Path,
    fake_sbatch: Path,
    fake_scancel: Path,
    signum: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A job the first scancel misses is cancelled on a later pass, and only that job.

    Array "b" records its pid 3 s after it starts, so the cancel at 2 s cannot reach it:
    the same miss as a real scancel that arrives before the controller has registered
    a just-submitted job. With a single scancel the driver waited out the whole job.
    Array "a" stopped at the first cancel and must not be cancelled again.
    """
    monkeypatch.setenv("LATE_PIDFILE_JOB", "af-b-")
    monkeypatch.setenv("LATE_PIDFILE_SECONDS", "3")
    runner = SlurmRunner(
        work_dir=tmp_path / "slurm",
        sbatch=str(fake_sbatch),
        scancel=str(fake_scancel),
        cancel_interval_seconds=1.0,
    )
    started = time.monotonic()
    signal_after(2.0, signum)
    with pytest.raises(EXPECTED):
        runner.run_many(two_arrays(tmp_path))
    assert time.monotonic() - started < 20, "the driver waited for the late job"
    counts = scancel_counts(fake_sbatch)
    (a,) = [name for name in counts if name.startswith("--name=af-a-")]
    (b,) = [name for name in counts if name.startswith("--name=af-b-")]
    assert counts[a] == 1, "a job that stopped is not cancelled again"
    assert counts[b] >= 2, "the job the first scancel missed was cancelled again"
    assert len(marker_passes(tmp_path, "a")) == 1
    assert len(marker_passes(tmp_path, "b")) == counts[b], "CANCELLED records every pass"
    assert not list(tmp_path.glob("late-*")), "no job ran on after the driver stopped"


@pytest.mark.parametrize("signum", SIGNALS, ids=lambda s: signal.Signals(s).name)
def test_slurm_cancel_gives_up_on_an_sbatch_that_never_returns(
    tmp_path: Path,
    fake_sbatch: Path,
    fake_scancel: Path,
    signum: int,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Cancelling is bounded: after cancel_retry_seconds the driver stops waiting, names
    the jobs that may still be running, and exits instead of hanging on their sbatch."""
    monkeypatch.setenv("SBATCH_IGNORES_CANCEL", "1")
    runner = SlurmRunner(
        work_dir=tmp_path / "slurm",
        sbatch=str(fake_sbatch),
        scancel=str(fake_scancel),
        cancel_interval_seconds=0.5,
        cancel_retry_seconds=2.0,
    )
    job = Job("h", ["/bin/sh", "-c", "sleep 30"], tmp_path, tmp_path / "h.log", [])
    started = time.monotonic()
    signal_after(1.0, signum)
    with pytest.raises(EXPECTED):
        runner.run_many([job])
    elapsed = time.monotonic() - started
    assert 2.5 < elapsed < 15, f"stopped after {elapsed:.1f} s, bound is 1 s + 2 s"
    (name,) = scancel_counts(fake_sbatch)
    assert scancel_counts(fake_sbatch)[name] >= 3
    (marker,) = (tmp_path / "slurm").glob("batch-*/CANCELLED")
    text = marker.read_text()
    assert "gave up" in text and f"{name.removeprefix('--')} by hand" in text
    assert "cancel by hand" in caplog.text


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


DRIVER = """\
import sys
from pathlib import Path
from af.pipeline import Pipeline, Step
from af.run import Job, SlurmRunner
from af.util.artefacts import Artefact

root, sbatch, scancel = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
runner = SlurmRunner(work_dir=root / "slurm", sbatch=sbatch, scancel=scancel)
def action(context):
    context.runner.run_many([
        Job(f"s{i}", ["/bin/sh", "-c", "sleep 60; echo x > s%d" % i], root, root / f"s{i}.log",
            [Artefact(root / f"s{i}")]) for i in range(2)
    ])
print("started", flush=True)
Pipeline([Step("relax", action, outputs=[Artefact(root / "s0")])], state_dir=root / "state",
         runner=runner, root=root).run()
"""

LAUNCH_BLOCKED = """\
import os, signal, sys
signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGTERM})
os.execv(sys.argv[1], sys.argv[1:])
"""


def test_driver_launched_with_sigterm_blocked_still_stops(
    tmp_path: Path, fake_sbatch: Path, fake_scancel: Path
) -> None:
    """A chain-shaped driver (pipeline step -> SlurmRunner -> sbatch --wait) whose launcher
    left SIGTERM blocked: before the fix it ignored kill until kill -9 (seen on o)."""
    import subprocess
    import sys

    driver = tmp_path / "driver.py"
    driver.write_text(DRIVER)
    launch = tmp_path / "launch.py"
    launch.write_text(LAUNCH_BLOCKED)
    work = tmp_path / "work"
    work.mkdir()
    process = subprocess.Popen(
        [
            sys.executable,
            str(launch),
            sys.executable,
            str(driver),
            str(work),
            str(fake_sbatch),
            str(fake_scancel),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=tmp_path,
    )
    try:
        assert process.stdout is not None
        assert process.stdout.readline().strip() == "started"
        time.sleep(2)  # into sbatch --wait
        process.send_signal(signal.SIGTERM)
        output, _ = process.communicate(timeout=20)
    finally:
        if process.poll() is None:
            process.kill()
    assert process.returncode != 0
    assert "terminated by SIGTERM" in output
    cancelled = (fake_sbatch.parent / "scancel-calls.txt").read_text().split()
    assert len(cancelled) == 1 and cancelled[0].startswith("--name=af-s0+1-")
    markers = list((work / "slurm").glob("batch-*/CANCELLED"))
    assert len(markers) == 1 and "cancelled" in markers[0].read_text()


BUSY_DRIVER = """\
import sys, time
from pathlib import Path
from af.pipeline import Pipeline, Step
from af.run import LocalRunner
from af.run.job import run_main
from af.util.artefacts import Artefact

root = Path(sys.argv[1])
def busy(context):
    time.sleep(120)  # stands in for a long C++ call (resolve_trapped) Python can't interrupt
    (root / "out").write_text("x")
def main():
    Pipeline([Step("refine", busy, outputs=[Artefact(root / "out")])], state_dir=root / "state",
             runner=LocalRunner(), root=root).run()
    return 0
print("started", flush=True)
run_main(main)
"""


def test_driver_busy_in_process_stops_at_once(tmp_path: Path) -> None:
    """SIGTERM while a step is minutes into in-process work: exit now with 143, not after it.

    Before the fix the main thread handled the signal but then joined the busy step thread,
    so the driver outlived SIGTERM by minutes (chain drivers on o, 30 Sep 2026).
    """
    import subprocess
    import sys

    driver = tmp_path / "busy.py"
    driver.write_text(BUSY_DRIVER)
    work = tmp_path / "work"
    work.mkdir()
    process = subprocess.Popen(
        [sys.executable, str(driver), str(work)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=tmp_path,
    )
    try:
        assert process.stdout is not None
        assert process.stdout.readline().strip() == "started"
        time.sleep(1)
        started = time.monotonic()
        process.send_signal(signal.SIGTERM)
        output, _ = process.communicate(timeout=20)
        elapsed = time.monotonic() - started
    finally:
        if process.poll() is None:
            process.kill()
    assert process.returncode == 128 + signal.SIGTERM
    assert elapsed < 5, f"took {elapsed:.1f} s to stop"
    assert "terminated by SIGTERM" in output and "step reruns next time" in output
    assert not (work / "state" / "refine.json").exists(), "an interrupted step keeps no record"


PUBLISHING_DRIVER = """\
import datetime, sys, time
from pathlib import Path
import af.store.store as store_module
from af.pipeline import Pipeline, Step
from af.run import LocalRunner
from af.run.job import run_main
from af.store import Provenance, Store
from af.util.artefacts import Artefact

root = Path(sys.argv[1])
store = Store.create(root / "store")
append_history = store_module._append_history
def slow_append_history(*args, **kwargs):
    (root / "in-publish").write_text("x")  # renamed into place; HISTORY and CURRENT not yet
    time.sleep(2)
    append_history(*args, **kwargs)
store_module._append_history = slow_append_history
def publish(context):
    with store.build("tables", "labx/h3") as build:
        (build.path / "t.txt").write_text("synthetic")
        now = datetime.datetime.now(datetime.UTC)
        build.publish(Provenance(step="publish", inputs=(), parameters={}, started=now,
                                 finished=now))
    time.sleep(120)
    (root / "out").write_text("x")
def main():
    Pipeline([Step("publish", publish, outputs=[Artefact(root / "out")])],
             state_dir=root / "state", runner=LocalRunner(), root=root).run()
    return 0
run_main(main)
"""


def test_driver_stopped_mid_publish_finishes_the_publish(tmp_path: Path) -> None:
    """SIGTERM between a version's rename and its HISTORY/CURRENT writes (07-chains, 30 Sep).

    The driver exits at once otherwise; here it waits for the publish to finish, so HISTORY
    and CURRENT agree, and then exits with 143.
    """
    import json
    import subprocess
    import sys

    driver = tmp_path / "publishing.py"
    driver.write_text(PUBLISHING_DRIVER)
    work = tmp_path / "work"
    work.mkdir()
    process = subprocess.Popen(
        [sys.executable, str(driver), str(work)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=tmp_path,
    )
    try:
        deadline = time.monotonic() + 20
        while not (work / "in-publish").exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert (work / "in-publish").exists(), "the step never reached the publish"
        started = time.monotonic()
        process.send_signal(signal.SIGTERM)
        output, _ = process.communicate(timeout=20)
        elapsed = time.monotonic() - started
    finally:
        if process.poll() is None:
            process.kill()
    assert process.returncode == 128 + signal.SIGTERM, output
    assert 1 < elapsed < 10, f"stopped after {elapsed:.1f} s: should wait out the 2 s publish"
    dataset = work / "store" / "tables" / "labx" / "h3"
    history = [json.loads(line) for line in (dataset / "HISTORY.jsonl").read_text().splitlines()]
    assert [entry["event"] for entry in history] == ["published"]
    assert (dataset / "CURRENT").read_text().strip() == history[0]["version"]
    assert (dataset / "versions" / history[0]["version"]).is_dir()
