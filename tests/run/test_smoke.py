"""af.run.smoke runs its checks end to end (locally, and against the fake sbatch)."""

import json
from pathlib import Path

from af.run import smoke


def test_smoke_local(tmp_path: Path) -> None:
    assert smoke.main(["--work-dir", str(tmp_path), "--local"]) == 0
    report = json.loads((tmp_path / "smoke-report.json").read_text())
    assert [check["passed"] for check in report["checks"]] == [True] * 5


def test_smoke_against_fake_sbatch(tmp_path: Path, fake_sbatch: Path) -> None:
    code = smoke.main(
        ["--work-dir", str(tmp_path), "--sbatch", str(fake_sbatch), "--output-wait-seconds", "0"]
    )
    report = json.loads((tmp_path / "smoke-report.json").read_text())
    assert code == 0, report
    assert report["sbatch"] == "slurm-fake 0.0"
    calls = (fake_sbatch.parent / "sbatch-calls.txt").read_text().splitlines()
    arrays = [c for c in calls if "--array=0-3%3" in c.split()]
    assert arrays, "the 6-task array is split at max-array-size 4 and throttled to 3"


def test_smoke_reports_a_misbehaving_cluster(tmp_path: Path, fake_sbatch: Path) -> None:
    """If the cluster loses a task's status, the smoke test must say FAIL, not PASS."""
    import os

    os.environ["KILL_TASKS"] = "0"
    try:
        code = smoke.main(
            [
                "--work-dir",
                str(tmp_path),
                "--sbatch",
                str(fake_sbatch),
                "--output-wait-seconds",
                "0",
            ]
        )
    finally:
        del os.environ["KILL_TASKS"]
    report = json.loads((tmp_path / "smoke-report.json").read_text())
    assert code == 1
    assert not report["checks"][0]["passed"]
