#!/usr/bin/env python3
"""Run ruff, mypy and pytest on exactly what is committed, the way CI sees it.

Why: a worktree is not a clean checkout. Untracked helper files, tool caches and the
editable install all make local runs pass where CI fails (a test helper that only
resolved because of an untracked file; import sorting that depended on the
environment). This script exports HEAD with `git archive` into a temporary directory
and runs every check there, with no caches:

- `af` is imported from the export, not from the worktree. Python runs with `-S`
  (so the editable install's import hook is not loaded) and the current
  environment's site-packages is put on PYTHONPATH for the dependencies.
- Child pythons that tests start without `-S` (``sys.executable -m af...``) would load
  that hook and import af from wherever the environment was installed from: an older
  worktree, say, so those tests silently checked other code (found 1 Oct 2026, when a
  driver test failed on a config key HEAD supports). A ``sitecustomize`` put first on
  PYTHONPATH drops every import finder that resolves af outside the export, and the
  check stops if a child still imports af from elsewhere.
- The environment must have everything CI installs: the core dependencies and every
  extra (`pip install -e '.[dev,geo]'`). The check reads HEAD's pyproject.toml and
  stops, naming what is missing, before a test fails at collection for want of it.
- Real-data tests run against the private data repo beside *this checkout* (or
  ``$AF_DATA`` if set): the export lives in a temporary directory, where the tests'
  default ``../acmacs-f-data`` would not exist and they would silently skip.
- The compiled optimiser (`af/map/_core*.so`) is not in git. It is copied from the
  current environment into the export, so the map tests still run, but only if it was
  built from the same C++ as HEAD: the check compares the `cpp/` sources of the checkout
  the environment's af was installed from with HEAD's, and stops if they differ
  (a stale binary silently tests old C++). Rebuild with `pip install -e .` here.

Usage (from a checkout, with the dev environment's python):

    python tools/check-committed.py            # checks HEAD
    python tools/check-committed.py --keep     # keep the export for inspection

Exit status is non-zero if any check fails. Uncommitted changes are NOT checked;
commit (on your branch) first.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import shutil
import site
import subprocess
import sys
import tempfile
import tomllib
from importlib import metadata
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--keep", action="store_true", help="keep the exported tree")
    args = parser.parse_args()

    repo = Path(git("rev-parse", "--show-toplevel"))
    head = git("rev-parse", "--short", "HEAD")
    dirty = git("status", "--porcelain", "--untracked-files=no")
    export = Path(tempfile.mkdtemp(prefix=f"af-check-{head}-"))
    shim: Path | None = None
    try:
        archive = subprocess.run(
            ["git", "-C", str(repo), "archive", "--format=tar", "HEAD"],
            check=True,
            capture_output=True,
        )
        subprocess.run(["tar", "-x", "-C", str(export)], input=archive.stdout, check=True)
        missing = missing_requirements(export)
        if missing:
            print("STOP: this environment lacks what CI installs:")
            for line in missing:
                print(f"  - {line}")
            extras = ",".join(optional_extras(export))
            print(f"Install as CI does: pip install -e '.[{extras}]'")
            return 1
        stale = stale_extension_reason(export)
        if stale:
            print(f"STOP: {stale}")
            return 1
        copied = copy_compiled_extensions(export)
        shim = write_site_shim()
        env = check_environment(export, shim)
        wrong = child_af_outside_export(export, env)
        if wrong:
            print(f"STOP: a child python without -S imports af from {wrong}, not the export")
            return 1
        print(f"checking committed HEAD {head} in {export}")
        if dirty:
            print("note: uncommitted changes in the worktree are NOT checked")
        if copied:
            print("compiled extensions copied in:", ", ".join(copied))
        af_data = os.environ.get("AF_DATA") or str((repo.parent / "acmacs-f-data").resolve())
        print(
            f"AF_DATA={af_data}"
            + ("" if Path(af_data).is_dir() else " (absent: real-data tests skip)")
        )
        failed = [
            name for name, command in checks(export) if not run(name, command, export, env, af_data)
        ]
    finally:
        if shim is not None:
            shutil.rmtree(shim, ignore_errors=True)
        if args.keep:
            print(f"kept {export}")
        else:
            shutil.rmtree(export, ignore_errors=True)
    if failed:
        print(f"\nFAILED on committed HEAD {head}: {', '.join(failed)}")
        return 1
    print(f"\nall checks passed on committed HEAD {head}")
    return 0


def checks(export: Path) -> list[tuple[str, list[str]]]:
    python = [sys.executable, "-S"]
    return [
        ("ruff check", [*python, "-m", "ruff", "check", "--no-cache", "."]),
        ("ruff format", [*python, "-m", "ruff", "format", "--no-cache", "--check", "."]),
        ("mypy", [*python, "-m", "mypy", "--cache-dir", str(export / ".mypy-cache")]),
        ("pytest", [*python, "-m", "pytest", "-q", "-p", "no:cacheprovider"]),
    ]


def run(name: str, command: list[str], export: Path, env: dict[str, str], af_data: str) -> bool:
    print(f"\n== {name}", flush=True)
    result = subprocess.run(command, cwd=export, env={**env, "AF_DATA": af_data})
    return result.returncode == 0


def check_environment(export: Path, shim: Path) -> dict[str, str]:
    env = dict(os.environ)
    env["AF_CHECK_EXPORT"] = str(export)
    env["PYTHONPATH"] = os.pathsep.join([str(shim), str(export), *site.getsitepackages()])
    return env


SITE_SHIM = '''"""Written by af's tools/check-committed.py for one run; see its docstring.

Drops every import finder that would import af from outside the export under test (the
editable install's redirecting finder), so child pythons started without -S test HEAD.
"""

import os
import sys

_export = os.path.realpath(os.environ["AF_CHECK_EXPORT"]) + os.sep


def _imports_af_elsewhere(finder):
    if isinstance(finder, type) or not hasattr(finder, "find_spec"):
        return False  # the class-level finders (PathFinder) follow sys.path: export first
    try:
        spec = finder.find_spec("af", None)
    except Exception:
        return False
    origin = (spec.origin or "") if spec is not None else ""
    return spec is not None and not os.path.realpath(origin).startswith(_export)


sys.meta_path[:] = [f for f in sys.meta_path if not _imports_af_elsewhere(f)]
'''


def write_site_shim() -> Path:
    """A directory holding only the sitecustomize above (kept out of the export, which is HEAD)."""
    shim = Path(tempfile.mkdtemp(prefix="af-check-site-"))
    (shim / "sitecustomize.py").write_text(SITE_SHIM)
    return shim


def child_af_outside_export(export: Path, env: dict[str, str]) -> str | None:
    """Where a child python started WITHOUT -S imports af from, if not the export."""
    result = subprocess.run(
        [sys.executable, "-c", "import af; print(af.__file__)"],
        cwd=export.parent,
        env=env,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return f"nowhere (import failed: {result.stderr.strip()[-300:]})"
    location = Path(result.stdout.strip()).resolve()
    return None if location.is_relative_to(export.resolve()) else str(location)


def optional_extras(export: Path) -> list[str]:
    project = tomllib.loads((export / "pyproject.toml").read_text())["project"]
    return sorted(project.get("optional-dependencies", {}))


def missing_requirements(export: Path) -> list[str]:
    """Requirements from HEAD's pyproject (core + every extra) not installed here."""
    project = tomllib.loads((export / "pyproject.toml").read_text())["project"]
    wanted = [(requirement, "core") for requirement in project.get("dependencies", [])]
    for extra, requirements in project.get("optional-dependencies", {}).items():
        wanted += [(requirement, f"extra {extra}") for requirement in requirements]
    missing = []
    for requirement, source in wanted:
        name = re.split(r"[\s<>=!~;\[]", requirement, maxsplit=1)[0]
        try:
            installed = metadata.version(name)
        except metadata.PackageNotFoundError:
            missing.append(f"{requirement} ({source}): not installed")
            continue
        if not version_satisfies(installed, requirement):
            missing.append(f"{requirement} ({source}): {installed} installed")
    return missing


def version_satisfies(installed: str, requirement: str) -> bool:
    """True unless `packaging` is available and says the version is out of range."""
    try:
        from packaging.requirements import Requirement
    except ImportError:
        return True  # presence was checked; the range needs packaging
    return Requirement(requirement).specifier.contains(installed, prereleases=True)


def stale_extension_reason(export: Path) -> str | None:
    """Why the environment's compiled af would not match HEAD's C++, or None if it does."""
    try:
        import af
    except ImportError:
        return None  # no af installed: nothing to copy, the map tests will say so
    installed_from = Path(af.__file__).resolve().parent.parent
    if not (installed_from / "cpp").is_dir():
        return None  # a non-editable install: cannot tell where it was built from
    if tree_hash(installed_from / "cpp") != tree_hash(export / "cpp"):
        return (
            f"this environment's af (and its compiled optimiser) was installed from "
            f"{installed_from}, whose cpp/ differs from HEAD's. Rebuild into this environment "
            f"from this checkout (pip install -e .) and run the check again."
        )
    return None


def tree_hash(directory: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(p for p in directory.rglob("*") if p.is_file()):
        digest.update(path.relative_to(directory).as_posix().encode() + b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def copy_compiled_extensions(export: Path) -> list[str]:
    """Copy af's built extension modules from this environment into the export."""
    copied = []
    for base in site.getsitepackages():
        for built in Path(base, "af").rglob("*.so") if Path(base, "af").is_dir() else []:
            relative = built.relative_to(base)
            target = export / relative
            if target.parent.is_dir() and not target.exists():
                shutil.copy2(built, target)
                copied.append(str(relative))
    return copied


def git(*args: str) -> str:
    return subprocess.run(["git", *args], check=True, capture_output=True, text=True).stdout.strip()


if __name__ == "__main__":
    sys.exit(main())
