"""af.store.work: the configured work area for step state, apart from the store."""

from dataclasses import dataclass
from pathlib import Path

import pytest

from af.store import PathsConfig, StoreError, Work
from af.util.config import ConfigError, load_config


@dataclass(frozen=True)
class StepConfig:
    paths: PathsConfig


def test_paths_config_requires_both_roots(tmp_path: Path) -> None:
    config = tmp_path / "step.toml"
    config.write_text('[paths]\nstore = "store"\nwork = "~/elsewhere/work"\n')
    paths = load_config(config, StepConfig).paths
    assert paths.store == (tmp_path / "store").resolve()
    assert paths.work == Path("~/elsewhere/work").expanduser()

    config.write_text('[paths]\nstore = "store"\n')
    with pytest.raises(ConfigError) as caught:
        load_config(config, StepConfig)
    assert caught.value.problems == ["paths.work: required key missing"]


def test_create_open_and_dataset_dirs(tmp_path: Path) -> None:
    root = tmp_path / "work"
    Work.create(root)
    work = Work.open(root)
    chain = work.dataset("chains", "labx/h3-hi-labx/main")
    assert chain.state == root / "chains" / "labx" / "h3-hi-labx" / "main" / "state"
    assert chain.state.is_dir() and chain.tmp.is_dir()


def test_open_never_creates(tmp_path: Path) -> None:
    """A typo in the configured path must fail, not start an empty work area."""
    with pytest.raises(StoreError, match="not a work area"):
        Work.open(tmp_path / "typo")
    assert not (tmp_path / "typo").exists()


def test_create_refuses_non_empty_and_bad_keys(tmp_path: Path) -> None:
    (tmp_path / "busy").mkdir()
    (tmp_path / "busy" / "file").write_text("x")
    with pytest.raises(StoreError, match="non-empty"):
        Work.create(tmp_path / "busy")
    work = Work.create(tmp_path / "work")
    with pytest.raises(StoreError):
        work.dataset("chain", "labx/h3")
    with pytest.raises(StoreError):
        work.dataset("chains", "../escape")


def test_unknown_layout(tmp_path: Path) -> None:
    root = tmp_path / "work"
    Work.create(root)
    (root / "WORK.toml").write_text("layout = 7\n")
    with pytest.raises(StoreError, match="layout 7"):
        Work.open(root)
