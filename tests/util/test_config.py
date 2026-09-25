"""af.util.config: unknown and missing keys are errors; paths resolve against the file."""

from dataclasses import dataclass, field
from pathlib import Path

import pytest

from af.util.config import ConfigError, load_config, parse_config


@dataclass(frozen=True)
class Store:
    root: Path


@dataclass(frozen=True)
class Step:
    name: str
    threads: int = 1
    tolerance: float = 1e-6


@dataclass(frozen=True)
class Settings:
    store: Store
    steps: list[Step]
    labels: dict[str, str] = field(default_factory=dict)
    note: str | None = None


def write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "af.toml"
    path.write_text(text)
    return path


def test_valid_config(tmp_path: Path) -> None:
    path = write(
        tmp_path,
        """
        [store]
        root = "store"

        [[steps]]
        name = "build"
        threads = 4
        tolerance = 1

        [labels]
        example = "Example"
        """,
    )
    settings = load_config(path, Settings)
    assert settings.store.root == (tmp_path / "store").resolve()
    assert settings.steps == [Step(name="build", threads=4, tolerance=1.0)]
    assert settings.labels == {"example": "Example"}
    assert settings.note is None


def test_relative_path_ignores_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    path = write(config_dir, 'steps = []\n[store]\nroot = "../store"\n')
    monkeypatch.chdir(tmp_path.parent)
    assert load_config(path, Settings).store.root == (tmp_path / "store").resolve()


def test_absolute_path_kept(tmp_path: Path) -> None:
    path = write(tmp_path, f'steps = []\n[store]\nroot = "{tmp_path / "elsewhere"}"\n')
    assert load_config(path, Settings).store.root == tmp_path / "elsewhere"


def test_unknown_and_missing_keys_all_reported(tmp_path: Path) -> None:
    path = write(
        tmp_path,
        """
        stepz = []

        [store]
        rot = "store"
        """,
    )
    with pytest.raises(ConfigError) as caught:
        load_config(path, Settings)
    assert sorted(caught.value.problems) == [
        "steps: required key missing",
        "stepz: unknown key",
        "store.root: required key missing",
        "store.rot: unknown key",
    ]


def test_wrong_types(tmp_path: Path) -> None:
    path = write(
        tmp_path,
        """
        [store]
        root = 3

        [[steps]]
        name = "build"
        threads = true
        """,
    )
    with pytest.raises(ConfigError) as caught:
        load_config(path, Settings)
    assert caught.value.problems == [
        "store.root: expected a path string, got int 3",
        "steps[0].threads: expected int, got bool",
    ]


def test_missing_file_is_fatal(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "absent.toml", Settings)


def test_syntax_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="TOML syntax"):
        load_config(write(tmp_path, "[store\n"), Settings)


def test_strings_stay_strings(tmp_path: Path) -> None:
    """The YAML traps: NO, 3.10 and dates must not change type."""

    @dataclass(frozen=True)
    class Labels:
        code: str
        version: str
        date: str

    labels = parse_config(
        {"code": "NO", "version": "3.10", "date": "2026-09-24"}, Labels, base_dir=Path("/")
    )
    assert labels == Labels(code="NO", version="3.10", date="2026-09-24")


def test_schema_must_be_dataclass() -> None:
    with pytest.raises(TypeError):
        parse_config({}, dict, base_dir=Path("/"))
