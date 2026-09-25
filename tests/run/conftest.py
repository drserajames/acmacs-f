"""Fixtures for af.run tests.

``fake_sbatch``: a fake ``sbatch`` that runs array tasks locally, as SLURM would.

It sets SLURM_ARRAY_TASK_ID, runs the batch script for each task, and exits with the
highest task status. KILL_TASKS (comma-separated task ids) simulates tasks SLURM
killed before they could record a status.
"""

import stat
import sys
from pathlib import Path

import pytest

FAKE_SBATCH = """\
#!{python}
# Fake sbatch: run array tasks locally. KILL_TASKS lists task ids to "kill" before they run.
import os, subprocess, sys
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
worst = 0
for task in range(last + 1):
    if task in killed:
        worst = max(worst, 1)
        continue
    env = {{**os.environ, "SLURM_ARRAY_TASK_ID": str(task)}}
    rc = subprocess.run(["/bin/sh", script], env=env).returncode
    worst = max(worst, rc)
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
