#!/usr/bin/env python3
"""Make a development environment with exactly a release's packages, af editable from a worktree.

Why: a venv made with ``python -m venv`` and pip takes whatever python and libraries are
newest. On 6 Oct 2026 that was python 3.14 + matplotlib 3.11.2 against the releases' 3.12 +
3.10.9, and 57 of 75 round figures came out 0.5-2.8% of pixels different. A figure checked in
a worktree was not the figure the round ships. This env replays the release's own locks, so
the only difference from the release is the worktree's af.

    python3 tools/dev-env.py --release <releases>/<sha12> --prefix <new env dir> \\
        --worktree <acmacs-f worktree> --micromamba <path> --mamba-root <dir> \\
        [--cxx /usr/bin/clang++] [--extras dev,geo]

The release's lock must be for this machine's platform (an osx-arm64 lock is never replayed
on linux-64). Afterwards the env's conda lock and pip freeze (af aside) are checked against
the release's, so a env that differs is refused, not handed over. The last line printed is
the env's python. The editable install points at ``--worktree``: after that worktree is
removed or switched, make the env again.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import platform
import sys
from pathlib import Path
from types import ModuleType


def make_release_module() -> ModuleType:
    """tools/make-release.py, whose lock replay this shares (one copy of that logic)."""
    path = Path(__file__).with_name("make-release.py")
    spec = importlib.util.spec_from_file_location("make_release", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["make_release"] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    mr = make_release_module()
    args = parse_args()
    release, prefix, worktree = (
        args.release.resolve(),
        args.prefix.resolve(),
        args.worktree.resolve(),
    )
    lock_file, pip_file = release / mr.LOCK_FILE, release / mr.PIP_FILE
    for path in (release / mr.RELEASE_FILE, lock_file, pip_file, worktree / "pyproject.toml"):
        if not path.is_file():
            raise SystemExit(f"{path} is missing")
    if prefix.exists():
        raise SystemExit(f"{prefix} exists; remove it or choose another --prefix")
    here, locked = mr.conda_platform(), mr.lock_platform(lock_file.read_text())
    if here != locked:
        raise SystemExit(f"{release.name}'s lock is for {locked}; this machine is {here}")

    base_env = mr.clean_environment(args)
    mr.step(f"creating {prefix} from {release.name}'s lock ({here})", "environment")
    create = [str(args.micromamba), "create", "--yes", "--quiet", "--prefix", str(prefix)]
    mr.run([*create, "--file", str(lock_file)], base_env)
    python = str(prefix / "bin" / "python")
    build_env = {**base_env, "PATH": f"{prefix / 'bin'}{os.pathsep}{base_env['PATH']}"}
    mr.step("installing the locked pip-only packages", "pip_lock")
    mr.install_pip_pins(python, pip_file.read_text(), build_env, worktree)
    mr.step(f"installing af editable from {worktree}", "install")
    if platform.system() == "Darwin":
        openmp = [f"--config-settings=cmake.define.AF_OPENMP_PREFIX={prefix}"]
    else:
        openmp = [f"--config-settings=cmake.define.CMAKE_INSTALL_RPATH={prefix / 'lib'}"]
    install = [python, "-m", "pip", "install", "--no-deps", "--no-build-isolation", *openmp]
    mr.run([*install, "-e", f"{worktree}[{args.extras}]"], build_env, cwd=worktree)

    mr.step("checking the env against the release", "verify")
    lock = mr.capture(
        [str(args.micromamba), "env", "export", "--explicit", "--prefix", str(prefix)], base_env
    )
    pip = mr.capture([python, "-m", "pip", "freeze"], build_env)
    mr.check_replay(release, lock, pip)
    mr.step(f"dev env ready: {release.name}'s packages, af from {worktree}")
    print(f"== stage times: {mr.stage_summary()}", flush=True)
    print(python)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--release", type=Path, required=True, help="a release directory")
    parser.add_argument("--prefix", type=Path, required=True, help="where to make the env")
    parser.add_argument("--worktree", type=Path, required=True, help="acmacs-f checkout for af")
    parser.add_argument("--micromamba", type=Path, required=True, help="micromamba executable")
    parser.add_argument("--mamba-root", type=Path, required=True, help="MAMBA_ROOT_PREFIX")
    parser.add_argument("--cxx", help="C++ compiler for the optimiser; default c++")
    parser.add_argument("--extras", default="dev,geo", help="af extras (their packages are locked)")
    return parser.parse_args()


if __name__ == "__main__":
    sys.exit(main())
