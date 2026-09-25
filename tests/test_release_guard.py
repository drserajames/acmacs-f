"""A release's python refuses an af from outside the release (the cwd-shadowing trap)."""

from pathlib import Path

import pytest

from af import _check_release


def release(tmp_path: Path) -> Path:
    env = tmp_path / "rel" / "env"
    (env / "lib").mkdir(parents=True)
    (tmp_path / "rel" / "RELEASE.toml").write_text('commit = "x"\n')
    return env


def test_af_inside_the_release_is_fine(tmp_path: Path) -> None:
    env = release(tmp_path)
    _check_release(env, env / "lib" / "af" / "__init__.py")


def test_af_from_a_checkout_is_refused(tmp_path: Path) -> None:
    env = release(tmp_path)
    with pytest.raises(ImportError, match="frozen af release .* run it from outside"):
        _check_release(env, tmp_path / "checkout" / "af" / "__init__.py")


def test_ordinary_python_is_unaffected(tmp_path: Path) -> None:
    _check_release(tmp_path / "venv", tmp_path / "checkout" / "af" / "__init__.py")
