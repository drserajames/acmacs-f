"""Which af is running: pin a run to a frozen release, and say which one it is.

Long runs (chain replays, tree builds, reports) launch with a release's python, made by
``tools/make-release.py``. A run config names it, e.g. ``python = "<release>/bin/python"``,
and the driver calls :func:`require_python` first. Started by any other interpreter (a
dev venv, a shared editable clone that someone may pull mid-run), it refuses rather than
running code nobody pinned. :func:`release_info` reads the release's RELEASE.toml, so a
run can log exactly which commit produced it.
"""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path
from typing import Any

RELEASE_FILE = "RELEASE.toml"


class WrongInterpreter(RuntimeError):
    """The run was started by a different python from the one its config names."""


def require_python(expected: Path | str) -> None:
    """Stop unless this process runs the configured interpreter (symlinks resolved)."""
    wanted = Path(expected).expanduser()
    if not wanted.exists():
        raise WrongInterpreter(f"configured python does not exist: {wanted}")
    running = Path(sys.executable)
    if running.resolve() != wanted.resolve():
        raise WrongInterpreter(
            f"this run must be started by {wanted} (a frozen af release), "
            f"but it was started by {running}"
        )


def release_info() -> dict[str, Any] | None:
    """The running release's RELEASE.toml, or None when not running from a release."""
    path = Path(sys.prefix).parent / RELEASE_FILE
    if not path.is_file():
        return None
    return tomllib.loads(path.read_text())
