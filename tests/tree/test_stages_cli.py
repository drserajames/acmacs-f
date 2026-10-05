"""The driver as it runs on a server: ``python -m af.tree.stages <config>``, jobs and all.

Every other stage test calls ``stages.run`` in-process, where the module's ``__name__`` is its
real name. Started with ``-m`` it is ``__main__``, which is what broke the first real build.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from af.store import Store
from af.tree import stages

from .test_stages import FIXTURE_PURPOSE, make_project


def test_the_driver_runs_its_jobs_when_started_with_dash_m(tmp_path: Path) -> None:
    config = make_project(tmp_path / "p")
    result = subprocess.run(
        [sys.executable, "-m", "af.tree.stages", str(config)],
        cwd=tmp_path,  # outside any checkout, as on the server
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    store = Store.open(tmp_path / "p" / "store")
    assert store.current("trees", f"h3/{FIXTURE_PURPOSE}") is not None
    job_log = (tmp_path / "p/work/trees/h3/build/job.log").read_text()
    assert "__main__" not in job_log


def test_jobs_name_the_real_module() -> None:
    assert stages.JOB_MODULE == stages.__name__ == "af.tree.stages"


def test_a_relative_config_path_reaches_the_jobs(tmp_path: Path) -> None:
    """Jobs run in their own output directory; the driver must hand them an absolute path."""
    make_project(tmp_path / "p")
    result = subprocess.run(
        [sys.executable, "-m", "af.tree.stages", "p/trees.toml"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert (
        Store.open(tmp_path / "p" / "store").current("trees", f"h3/{FIXTURE_PURPOSE}") is not None
    )
