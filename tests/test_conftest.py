"""The af_data fixture resolves AF_DATA against the repo root and skips when absent."""

from pathlib import Path

import pytest

from tests.helpers import REPO_ROOT, resolve_af_data


def test_default_is_sibling_of_repo(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AF_DATA", raising=False)
    assert resolve_af_data() == (REPO_ROOT.parent / "acmacs-f-data").resolve()


def test_relative_af_data_is_relative_to_repo(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AF_DATA", "some/where")
    assert resolve_af_data() == (REPO_ROOT / "some" / "where").resolve()


def test_absolute_af_data(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("AF_DATA", str(tmp_path))
    assert resolve_af_data() == tmp_path.resolve()
