"""af.util.running: which af is running, from a release, from a checkout, or not known."""

import socket
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import duckdb
import pyarrow
import pyarrow.parquet as pq
import pytest

import af
from af.util import running

COMMIT = "0123456789abcdef0123456789abcdef01234567"


@pytest.fixture(autouse=True)
def asked_afresh() -> Iterator[None]:
    """running_af() answers once per process: each test asks afresh, and leaves no answer."""
    running._running_af.cache_clear()
    yield
    running._running_af.cache_clear()


def af_at(directory: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Pretend the af imported lives in ``directory/af``."""
    package = directory / "af"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    monkeypatch.setattr(af, "__file__", str(package / "__init__.py"))
    return package


def release_at(root: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A release directory with its RELEASE.toml, and this python made to look like its own."""
    prefix = root / "env"
    prefix.mkdir(parents=True)
    (root / "RELEASE.toml").write_text(f'commit = "{COMMIT}"\ncxx = "c++"\n')
    monkeypatch.setattr(sys, "prefix", str(prefix))
    return prefix


def git(repo: Path, *args: str) -> str:
    quiet = [
        "-c",
        "user.name=t",
        "-c",
        "user.email=t@example.invalid",
        "-c",
        "commit.gpgsign=false",
    ]
    hooks = ["-c", "core.hooksPath=/dev/null"]  # a test repository runs nobody's hooks
    return subprocess.run(
        ["git", "-C", str(repo), *quiet, *hooks, *args], capture_output=True, text=True, check=True
    ).stdout.strip()


def test_from_a_release_the_commit_is_the_one_in_its_release_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prefix = release_at(tmp_path / "0123456789ab", monkeypatch)
    af_at(prefix / "lib" / "site-packages", monkeypatch)
    record = running.running_af()
    assert record["commit"] == running.release_info()["commit"] == COMMIT  # type: ignore[index]
    assert record["dirty"] is False  # a release is read-only
    assert record["release"] == "0123456789ab"  # its directory name


def test_a_release_python_importing_another_af_is_not_that_release(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    release_at(tmp_path / "0123456789ab", monkeypatch)
    af_at(tmp_path / "elsewhere", monkeypatch)
    record = running.running_af()
    assert record["release"] is None and record["commit"] is None


def test_from_a_checkout_the_commit_is_head_and_a_changed_file_says_dirty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sys, "prefix", str(tmp_path / "devenv"))  # not a release
    repo = tmp_path / "checkout"
    package = af_at(repo, monkeypatch)
    git(repo, "init", "-q")
    git(repo, "add", "af/__init__.py")
    git(repo, "commit", "-q", "-m", "first")
    clean = running.running_af()
    assert clean["commit"] == git(repo, "rev-parse", "HEAD") and len(clean["commit"]) == 40
    assert clean["dirty"] is False and clean["release"] is None

    (repo / "notes.txt").write_text("untracked: not part of the code that ran")
    running._running_af.cache_clear()
    assert running.running_af()["dirty"] is False

    (package / "__init__.py").write_text("changed = True\n")
    assert running.running_af()["dirty"] is False  # asked once per process: still the first answer
    running._running_af.cache_clear()
    dirty = running.running_af()
    assert dirty["dirty"] is True and dirty["commit"] == clean["commit"]


def test_an_af_inside_some_other_repository_is_not_given_its_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sys, "prefix", str(tmp_path / "devenv"))
    repo = tmp_path / "other-project"
    af_at(repo / "vendored", monkeypatch)
    git(repo, "init", "-q")
    git(repo, "add", ".")
    git(repo, "commit", "-q", "-m", "first")
    assert running.running_af()["commit"] is None


def test_neither_a_release_nor_a_checkout_is_not_known_rather_than_guessed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sys, "prefix", str(tmp_path / "devenv"))
    af_at(tmp_path / "site-packages", monkeypatch)
    assert running.running_af() == {
        "version": af.__version__,
        "commit": None,
        "dirty": None,
        "release": None,
        "python": ".".join(str(n) for n in sys.version_info[:3]),
        "libraries": {"duckdb": duckdb.__version__, "pyarrow": pyarrow.__version__},
    }


def test_no_git_is_not_known(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def no_git(*args: object, **kwargs: object) -> None:
        raise FileNotFoundError("git")

    monkeypatch.setattr(sys, "prefix", str(tmp_path / "devenv"))
    monkeypatch.setattr(subprocess, "run", no_git)
    record = running.running_af()
    assert record["commit"] is None and record["dirty"] is None


def test_the_record_names_no_host_and_no_path() -> None:
    record = running.running_af()  # this test run's own af, whatever it is
    assert set(record) == {"version", "commit", "dirty", "release", "python", "libraries"}
    text = [v for v in [*record.values(), *record["libraries"].values()] if isinstance(v, str)]
    assert not any("/" in value or socket.gethostname() in value for value in text)


def test_each_caller_gets_its_own_copy() -> None:
    record = running.running_af()
    record["commit"] = record["libraries"]["pyarrow"] = "edited by a caller"
    again = running.running_af()
    assert "edited by a caller" not in (again["commit"], again["libraries"]["pyarrow"])


def test_the_library_versions_are_the_ones_parquet_files_are_stamped_with(tmp_path: Path) -> None:
    path = tmp_path / "t.parquet"
    pq.write_table(pyarrow.table({"n": [1]}), path)
    stamped = pq.ParquetFile(path).metadata.created_by  # "parquet-cpp-arrow version <x.y.z>"
    assert stamped.endswith(" " + running.running_af()["libraries"]["pyarrow"])


def test_a_library_not_installed_is_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(running, "BYTE_DECIDING_LIBRARIES", ("no-such-distribution-af-test",))
    assert running.running_af()["libraries"] == {"no-such-distribution-af-test": None}
