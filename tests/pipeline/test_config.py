"""make_runner: where a SLURM runner keeps its job files."""

from pathlib import Path

import pytest

from af.pipeline.config import RunnerSettings, SlurmSettings, make_runner
from af.run import SlurmRunner
from af.util.config import ConfigError


def test_slurm_work_dir_from_config_wins(tmp_path):
    settings = RunnerSettings(kind="slurm", slurm=SlurmSettings(work_dir=tmp_path / "set"))
    runner = make_runner(settings, default_work_dir=tmp_path / "default")
    assert isinstance(runner, SlurmRunner) and runner.work_dir == tmp_path / "set"


def test_slurm_work_dir_falls_back_to_the_callers_default(tmp_path):
    settings = RunnerSettings(kind="slurm", slurm=SlurmSettings())
    runner = make_runner(settings, default_work_dir=tmp_path / "jobs")
    assert isinstance(runner, SlurmRunner) and runner.work_dir == tmp_path / "jobs"


def test_slurm_work_dir_is_required_without_a_default():
    settings = RunnerSettings(kind="slurm", slurm=SlurmSettings())
    with pytest.raises(ConfigError, match="work_dir: required"):
        make_runner(settings, Path("run.toml"))
