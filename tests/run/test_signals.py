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
