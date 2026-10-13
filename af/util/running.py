"""Which af is running: the commit, whether it is a release, and the python.

Why: ``af.__version__`` stays the same for months, so a record holding only it cannot say which
code wrote a store version. Three things went unanswerable on one day (13 Oct 2026): tree files
written from an environment outside the releases, datasets published from development
environments, and maps that repeat on one machine but not across machines. :func:`running_af`
is the one answer, written into every record af keeps of its own work.

What it says, and how it knows:

- From a frozen release (``tools/make-release.py``): the commit in the release's
  ``RELEASE.toml``, ``dirty`` false since a release is read-only, and ``release`` its id.
- From a git checkout (an editable install): ``HEAD``, and ``dirty`` when a tracked file differs
  from it. ``release`` is null.
- Otherwise ``commit`` and ``dirty`` are null: not known, said rather than guessed.

It never records a host name or a path: a record is read on other machines, and by other people.
"""

from __future__ import annotations

import functools
import platform
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any

import af

RELEASE_FILE = "RELEASE.toml"
RELEASE_ID_LENGTH = 12  # a release's directory name: the first 12 characters of its commit


def release_info() -> dict[str, Any] | None:
    """The running release's RELEASE.toml, or None when not running from a release."""
    path = Path(sys.prefix).parent / RELEASE_FILE
    if not path.is_file():
        return None
    return tomllib.loads(path.read_text())


def running_af() -> dict[str, Any]:
    """``version``, ``commit``, ``dirty``, ``release`` and ``python`` of this process's af.

    Asked once per process: a long run records the af it started with, also when the checkout
    it runs from is edited meanwhile.
    """
    return dict(_running_af())


@functools.cache
def _running_af() -> dict[str, Any]:
    record: dict[str, Any] = {
        "version": af.__version__,
        "commit": None,
        "dirty": None,
        "release": None,
        "python": platform.python_version(),
    }
    source = Path(af.__file__).resolve().parent
    release = release_info()
    # Only when the af imported is the release's own: a release's python can be made to import
    # another af (PYTHONPATH), and that af is not the release.
    if release and release.get("commit") and source.is_relative_to(Path(sys.prefix).resolve()):
        commit = str(release["commit"])
        record.update(commit=commit, dirty=False, release=commit[:RELEASE_ID_LENGTH])
        return record
    try:
        # The checkout must be af's own: a copy of af inside some other repository would
        # otherwise be given that repository's commit.
        if Path(_git(source, "rev-parse", "--show-toplevel")).resolve() == source.parent:
            record["commit"] = _git(source, "rev-parse", "HEAD")
            record["dirty"] = bool(_git(source, "status", "--porcelain", "--untracked-files=no"))
    except (OSError, subprocess.CalledProcessError):
        pass  # no git, or not a checkout: not known
    return record


def _git(source: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(source), *args], capture_output=True, text=True, check=True
    ).stdout.strip()
