"""No source file under af/ or tests/ is matched by .gitignore.

An unanchored `build/` pattern once silently ignored the af/tree/build package: the
code existed on disk, tests passed locally, and nothing reached the repository.
"""

import shutil
import subprocess
from pathlib import Path

import pytest
from conftest import REPO_ROOT

SKIP_PARTS = {"__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache"}


def source_files() -> list[str]:
    files = []
    for top in ("af", "tests"):
        for path in (REPO_ROOT / top).rglob("*"):
            if path.is_file() and not SKIP_PARTS & set(path.parts) and path.suffix != ".pyc":
                files.append(path.relative_to(REPO_ROOT).as_posix())
    return sorted(files)


def test_no_source_file_is_ignored() -> None:
    if not shutil.which("git") or not (REPO_ROOT / ".git").exists():
        pytest.skip("not a git checkout")
    files = source_files()
    assert "af/tree/build/__init__.py" in files
    # --no-index: test the patterns even against files that are already tracked.
    checked = subprocess.run(
        ["git", "check-ignore", "--no-index", "--verbose", "--stdin"],
        cwd=REPO_ROOT,
        input="\n".join(files) + "\n",
        capture_output=True,
        text=True,
    )
    assert checked.returncode in (0, 1), checked.stderr  # 128 = git error
    assert checked.stdout == "", f"ignored source files:\n{checked.stdout}"
