"""af.run.runtime: a run pinned to a release refuses any other interpreter."""

import sys
from pathlib import Path

import pytest

from af.run import runtime


def test_the_running_interpreter_passes() -> None:
    runtime.require_python(sys.executable)


def test_a_symlink_to_the_running_interpreter_passes(tmp_path: Path) -> None:
    link = tmp_path / "bin" / "python"
    link.parent.mkdir()
    link.symlink_to(sys.executable)
    runtime.require_python(link)


def test_another_interpreter_is_refused(tmp_path: Path) -> None:
    other = tmp_path / "python"
    other.write_text("#!/bin/sh\n")
    with pytest.raises(runtime.WrongInterpreter, match="must be started by"):
        runtime.require_python(other)


def test_a_missing_interpreter_is_refused(tmp_path: Path) -> None:
    with pytest.raises(runtime.WrongInterpreter, match="does not exist"):
        runtime.require_python(tmp_path / "absent" / "python")


def test_release_info(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "prefix", str(tmp_path / "env"))
    assert runtime.release_info() is None
    (tmp_path / "RELEASE.toml").write_text('commit = "0123456789ab"\ncxx = "c++"\n')
    assert runtime.release_info() == {"commit": "0123456789ab", "cxx": "c++"}
