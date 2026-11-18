"""Fixtures for af.run tests.

``fake_sbatch``: a fake ``sbatch`` that runs array tasks locally, as SLURM would.

It sets SLURM_ARRAY_TASK_ID, runs the batch script for each task, and exits with the
highest task status. KILL_TASKS (comma-separated task ids) simulates tasks SLURM
killed before they could record a status.

Cancelling (``fake_scancel``) reaches a fake sbatch through the pid file it records
under its job name. LATE_PIDFILE_JOB (a job-name prefix) makes those sbatches record it
LATE_PIDFILE_SECONDS late, like a job the controller has not registered yet when the
first scancel arrives. SBATCH_IGNORES_CANCEL=1 makes every sbatch ignore the cancel and
never return on its own, like one that hangs.
"""

import stat
import sys
from pathlib import Path

import pytest

FAKE_SBATCH = """\
#!{python}
# Fake sbatch: run array tasks locally. KILL_TASKS lists task ids to "kill" before they run.
import os, signal, subprocess, sys, time
args = sys.argv[1:]
if args == ["--version"]:
    print("slurm-fake 0.0")
    sys.exit(0)
with open(os.path.join(os.path.dirname(sys.argv[0]), "sbatch-calls.txt"), "a") as log:
    log.write(" ".join(args) + "\\n")
script = args[-1]
array = next(a.split("=", 1)[1] for a in args if a.startswith("--array="))
last = int(array.split("%")[0].split("-")[1])
killed = {{int(x) for x in os.environ.get("KILL_TASKS", "").split(",") if x}}
# Like SLURM, a cancelled job ends: record our pid under the job name for fake scancel,
# and on SIGTERM stop the running task and exit.
name = next(a.split("=", 1)[1] for a in args if a.startswith("--job-name="))
pidfile = os.path.join(os.path.dirname(sys.argv[0]), "running-" + name)
child = None
def cancelled(signum, frame):
    if child is not None:
        child.terminate()
    sys.exit(143)
if os.environ.get("SBATCH_IGNORES_CANCEL"):
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
else:
    signal.signal(signal.SIGTERM, cancelled)
late = os.environ.get("LATE_PIDFILE_JOB")
if late and name.startswith(late):
    time.sleep(float(os.environ.get("LATE_PIDFILE_SECONDS", "1.5")))
with open(pidfile, "w") as f:
    f.write(str(os.getpid()))
worst = 0
for task in range(last + 1):
    if task in killed:
        worst = max(worst, 1)
        continue
    env = {{**os.environ, "SLURM_ARRAY_TASK_ID": str(task)}}
    child = subprocess.Popen(["/bin/sh", script], env=env)
    worst = max(worst, child.wait())
print("12345")
sys.exit(worst)
"""


@pytest.fixture
def fake_sbatch(tmp_path: Path) -> Path:
    path = tmp_path / "bin" / "sbatch"
    path.parent.mkdir()
    path.write_text(FAKE_SBATCH.format(python=sys.executable))
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return path


FAKE_SCANCEL = """\
#!/bin/sh
# Fake scancel: --name=<job name> -> SIGTERM the fake sbatch that recorded that name.
here="$(dirname "$0")"
echo "$@" >> "$here/scancel-calls.txt"
name="${{1#--name=}}"
[ -f "$here/running-$name" ] && kill -TERM "$(cat "$here/running-$name")"
exit 0
"""


@pytest.fixture
def fake_scancel(fake_sbatch: Path) -> Path:
    path = fake_sbatch.parent / "scancel"
    path.write_text(FAKE_SCANCEL.format())
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return path
