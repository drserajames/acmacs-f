#!/usr/bin/env python3
"""Make a frozen, self-contained af install for one commit: a *release*.

Why: a long run (a chain replay, a tree build, a report) must use the same af code
from its first job to its last. A shared editable clone that someone pulls mid-run
mixes new Python with an old compiled optimiser. That happened on `o` on 25 Sep 2026,
and an H1 chain had to be rerun from scratch. A release can't change after it is made.

    python3 tools/make-release.py --repo <git clone> --rev <commit> \\
        --releases <dir> --micromamba <path> --mamba-root <dir> \\
        [--cxx /usr/bin/g++] [--extras dev,geo] [--link <path>]

It makes ``<releases>/<sha12>/`` containing:

- ``src/``: the commit's files (``git archive``; not a clone, so it cannot be pulled);
- ``env/``: its own micromamba environment from that commit's environment.yml, which
  also freezes the tool versions. micromamba hard-links from its package cache, so a
  release costs little disk;
- af installed into ``env`` **non-editable**, with the C++ optimiser built there using
  ``--cxx`` (recorded), in a clean environment. Compiler flags exported by your shell
  (LDFLAGS, CPPFLAGS, …) are not inherited;
- checks: af and its compiled optimiser import from inside the release; every tool test
  passes with ``--require-tools`` (pdflatex excepted); ``af.run.smoke --local`` passes;
- ``RELEASE.toml`` (commit, compiler, versions, host, time), ``conda-explicit.txt`` and
  ``pip-freeze.txt``, written last. The whole tree is then made read-only.

The last line printed is the release's python, ``<release>/bin/python`` (``bin`` links
to ``env/bin``). Launch runs with it, and every Python job
they submit inherits it (``Job.python_module`` uses ``sys.executable``). Point tool paths
in configs at ``<release>/env/bin/``.

Run a release's python from outside any acmacs-f checkout (or with ``python -P``):
``python -c``/``-m`` put the current directory first on sys.path, so inside a checkout it
would import that checkout's af. af refuses that when its python is a release.

Running it again for a commit that already has a release just verifies it and prints the
python. A half-made release (no RELEASE.toml) is refused unless ``--replace-incomplete``.
``--link`` points a stable symlink (e.g. the laptop's ``~/AC/eu/af-env``) at the release.
To delete one: ``chmod -R u+w <release> && rm -rf <release>``.
"""

from __future__ import annotations

import argparse
import datetime
import fcntl
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

RELEASE_FILE = "RELEASE.toml"


def main() -> int:
    args = parse_args()
    repo = args.repo.resolve()
    sha = git(repo, "rev-parse", "--verify", f"{args.rev}^{{commit}}")
    releases = args.releases.resolve()
    releases.mkdir(parents=True, exist_ok=True)
    release = releases / sha[:12]
    with locked(releases / ".lock"):
        if (release / RELEASE_FILE).is_file():
            print(f"release {sha[:12]} exists; verifying", flush=True)
            verify_imports(release)
        else:
            build(args, repo, sha, release)
    if args.link:
        relink(args.link, release)
        print(public_python(args.link))
    else:
        print(public_python(release))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--repo", type=Path, required=True, help="a git clone holding --rev")
    parser.add_argument("--rev", required=True, help="commit, tag or branch to release")
    parser.add_argument("--releases", type=Path, required=True, help="directory of releases")
    parser.add_argument("--micromamba", type=Path, required=True, help="micromamba executable")
    parser.add_argument("--mamba-root", type=Path, required=True, help="MAMBA_ROOT_PREFIX")
    parser.add_argument("--cxx", help="C++ compiler for the optimiser (recorded); default c++")
    parser.add_argument("--extras", default="dev,geo", help="af extras to install")
    parser.add_argument("--link", type=Path, help="point this symlink at the release")
    parser.add_argument("--replace-incomplete", action="store_true")
    return parser.parse_args()


def build(args: argparse.Namespace, repo: Path, sha: str, release: Path) -> None:
    if release.exists():
        if not args.replace_incomplete:
            raise SystemExit(
                f"{release} exists but has no {RELEASE_FILE} (an interrupted build). "
                "Delete it, or pass --replace-incomplete."
            )
        remove_tree(release)
    started = now()
    src, env = release / "src", release / "env"
    src.mkdir(parents=True)
    step(f"exporting {sha[:12]} from {repo}")
    archive = subprocess.run(
        ["git", "-C", str(repo), "archive", sha], check=True, capture_output=True
    )
    subprocess.run(["tar", "-x", "-C", str(src)], input=archive.stdout, check=True)

    base_env = clean_environment(args)
    step("creating the environment from environment.yml")
    run(
        [
            str(args.micromamba),
            "create",
            "--yes",
            "--quiet",
            "--prefix",
            str(env),
            "--file",
            str(src / "environment.yml"),
        ],
        base_env,
    )
    python = python_of(release)
    build_env = {**base_env, "PATH": f"{env / 'bin'}{os.pathsep}{base_env['PATH']}"}
    # The first launch of a freshly installed cmake can take longer than scikit-build-core's
    # probe allows (macOS scans a new binary; NFS is slow on first read): it then reports
    # "Could not find CMake". Launch the build tools once, untimed.
    for tool in ("cmake", "ninja"):
        run([str(env / "bin" / tool), "--version"], build_env)
    step(f"installing af (non-editable, extras {args.extras}) with {build_env.get('CXX', 'c++')}")
    run(
        [python, "-m", "pip", "install", "--no-build-isolation", f"{src}[{args.extras}]"],
        build_env,
        cwd=release,
    )

    step("verifying")
    verify_imports(release)
    with tempfile.TemporaryDirectory() as scratch:
        smoke = [python, "-m", "af.run.smoke", "--local", "--work-dir", scratch]
        run(smoke, build_env, cwd=scratch)
        # The tests without the source tree beside them, so they exercise the installed af
        # (with its compiled optimiser), not src/af.
        (Path(scratch) / "tests").symlink_to(src / "tests")
        shutil.copy2(src / "pyproject.toml", Path(scratch) / "pyproject.toml")
        tool_tests = [python, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-m", "tool"]
        run([*tool_tests, "--require-tools", "--skip-tool", "pdflatex"], build_env, cwd=scratch)

    step("recording")
    (release / "conda-explicit.txt").write_text(
        capture(
            [str(args.micromamba), "env", "export", "--explicit", "--prefix", str(env)], base_env
        )
    )
    (release / "pip-freeze.txt").write_text(capture([python, "-m", "pip", "freeze"], build_env))
    write_release_file(args, release, sha, started, build_env)
    (release / "bin").symlink_to("env/bin")  # so <release>/bin/python, like o's env/bin/python
    make_read_only(release)
    step(f"release {sha[:12]} ready in {release}")


def clean_environment(args: argparse.Namespace) -> dict[str, str]:
    """Only what the build needs: no compiler flags leak in from the caller's shell."""
    env = {
        "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
        "HOME": os.environ["HOME"],
        "MAMBA_ROOT_PREFIX": str(args.mamba_root.resolve()),
        "LANG": "C.UTF-8",
    }
    for name in ("TMPDIR", "PIP_CACHE_DIR"):
        if name in os.environ:
            env[name] = os.environ[name]
    if args.cxx:
        env["CXX"] = args.cxx
    return env


def verify_imports(release: Path) -> None:
    """af and its compiled optimiser must import from inside this release's env."""
    probe = "import af, af.map._core as c; print(af.__file__); print(c.__file__)"
    lines = capture([python_of(release), "-I", "-c", probe], {"PATH": "/usr/bin:/bin"}).split()
    outside = [line for line in lines if not Path(line).resolve().is_relative_to(release)]
    if len(lines) != 2 or outside:
        raise SystemExit(f"af does not import from inside {release}: {lines}")


def write_release_file(
    args: argparse.Namespace, release: Path, sha: str, started: str, env: dict[str, str]
) -> None:
    python = python_of(release)
    af_version = capture([python, "-I", "-c", "import af; print(af.__version__)"], env).strip()
    cxx = env.get("CXX", "c++")
    cxx_version = capture([cxx, "--version"], env).splitlines()[0]
    fields = {
        "commit": sha,
        "rev": args.rev,
        "af_version": af_version,
        "python": python,
        "cxx": cxx,
        "cxx_version": cxx_version,
        "micromamba": capture([str(args.micromamba), "--version"], env).strip(),
        "extras": args.extras,
        "host": platform.node(),
        "started": started,
        "finished": now(),
    }
    body = "".join(
        f'{key} = "{str(value).replace(chr(34), chr(39))}"\n' for key, value in fields.items()
    )
    (release / RELEASE_FILE).write_text("# af release: frozen; do not edit\n" + body)


def python_of(release: Path) -> str:
    return str(release / "env" / "bin" / "python")


def public_python(release: Path) -> str:
    """The path to give people and configs: <release>/bin/python (through the link)."""
    return str(release / "bin" / "python")


def relink(link: Path, release: Path) -> None:
    """Point ``link`` at ``release`` atomically (a new symlink renamed over the old)."""
    if link.exists() and not link.is_symlink():
        raise SystemExit(f"--link {link} exists and is not a symlink; not replacing it")
    temporary = link.with_name(f".{link.name}.{os.getpid()}")
    temporary.symlink_to(release)
    temporary.replace(link)
    step(f"{link} -> {release}")


def make_read_only(root: Path) -> None:
    for path in [root, *root.rglob("*")]:
        if path.is_symlink():
            continue
        mode = path.stat().st_mode
        path.chmod(mode & 0o555)


def remove_tree(root: Path) -> None:
    for path in [root, *root.rglob("*")]:
        if not path.is_symlink() and path.is_dir():
            path.chmod(path.stat().st_mode | 0o700)
    shutil.rmtree(root)


class locked:
    """Serialise releases in one directory (two builds of one commit would collide)."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def __enter__(self) -> None:
        self.handle = self.path.open("w")
        fcntl.flock(self.handle, fcntl.LOCK_EX)

    def __exit__(self, *exc: object) -> None:
        fcntl.flock(self.handle, fcntl.LOCK_UN)
        self.handle.close()


def run(command: list[str], env: dict[str, str], cwd: Path | str | None = None) -> None:
    result = subprocess.run(command, env=env, cwd=cwd)
    if result.returncode != 0:
        raise SystemExit(f"failed ({result.returncode}): {' '.join(command)}")


def capture(command: list[str], env: dict[str, str]) -> str:
    return subprocess.run(command, env=env, check=True, capture_output=True, text=True).stdout


def git(repo: Path, *args: str) -> str:
    result = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit(
            f"git {' '.join(args)} failed in {repo}: {result.stderr.strip()} "
            "(fetch the commit into that clone first)"
        )
    return result.stdout.strip()


def step(message: str) -> None:
    print(f"== {message}", flush=True)


def now() -> str:
    return datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds")


if __name__ == "__main__":
    sys.exit(main())
