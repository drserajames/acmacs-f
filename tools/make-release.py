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
  ``pip-freeze.txt``, written last. The whole tree is then made read-only;
- ``CONTENTS.txt``: every pull request merged into the commit's history, newest first, one
  per line (``#182<TAB>maps-align<TAB><merge sha><TAB><date><TAB><title>``). A project that
  pins a release can see whether it has the change it needs with ``grep '^#182' CONTENTS.txt``,
  without a clone or a git walk.

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
import re
import shutil
import subprocess
import sys
import tempfile
import time
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
    step(f"exporting {sha[:12]} from {repo}", "export")
    archive = subprocess.run(
        ["git", "-C", str(repo), "archive", sha], check=True, capture_output=True
    )
    subprocess.run(["tar", "-x", "-C", str(src)], input=archive.stdout, check=True)

    base_env = clean_environment(args)
    step("creating the environment from environment.yml", "environment")
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
    step("first launch of cmake and ninja", "tool_warmup")
    for tool in ("cmake", "ninja"):
        run([str(env / "bin" / tool), "--version"], build_env)
    step(
        f"installing af (non-editable, extras {args.extras}) with {build_env.get('CXX', 'c++')}",
        "install",
    )
    # On macOS, link the optimiser to the env's own libomp: numpy's OpenBLAS already loads
    # it, and a second libomp copy (e.g. Homebrew's) aborts the process at run time.
    # On Linux, a RUNPATH to the env's lib: _core's libgomp and libstdc++ then come from the
    # release whichever module loads first (o, 28 Sep: they came from the env only because
    # numpy happened to load first; imported first, _core would have pulled in the system's).
    openmp: list[str] = []
    if platform.system() == "Darwin":
        openmp = [f"--config-settings=cmake.define.AF_OPENMP_PREFIX={env}"]
    else:
        openmp = [f"--config-settings=cmake.define.CMAKE_INSTALL_RPATH={env / 'lib'}"]
    run(
        [python, "-m", "pip", "install", "--no-build-isolation", *openmp, f"{src}[{args.extras}]"],
        build_env,
        cwd=release,
    )

    step("verifying imports, linkage and the OpenMP runtime", "verify")
    verify_imports(release)
    verify_linkage(release)
    verify_one_openmp(python, build_env)
    with tempfile.TemporaryDirectory() as scratch:
        step("smoke test", "smoke")
        smoke = [python, "-m", "af.run.smoke", "--local", "--work-dir", scratch]
        run(smoke, build_env, cwd=scratch)
        step("tool tests", "tool_tests")
        # The tests without the source tree beside them, so they exercise the installed af
        # (with its compiled optimiser), not src/af.
        (Path(scratch) / "tests").symlink_to(src / "tests")
        shutil.copy2(src / "pyproject.toml", Path(scratch) / "pyproject.toml")
        tool_tests = [python, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-m", "tool"]
        run([*tool_tests, "--require-tools", "--skip-tool", "pdflatex"], build_env, cwd=scratch)

    step("recording", "record")
    (release / "conda-explicit.txt").write_text(
        capture(
            [str(args.micromamba), "env", "export", "--explicit", "--prefix", str(env)], base_env
        )
    )
    (release / "pip-freeze.txt").write_text(capture([python, "-m", "pip", "freeze"], build_env))
    (release / "CONTENTS.txt").write_text(contents(args.repo, sha))
    write_release_file(args, release, sha, started, build_env)
    (release / "bin").symlink_to("env/bin")  # so <release>/bin/python, like o's env/bin/python
    make_read_only(release)
    step(f"release {sha[:12]} ready in {release}")
    print(f"== stage times: {stage_summary()}", flush=True)


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


def verify_linkage(release: Path) -> None:
    """af's compiled extensions must take their libraries from the release.

    A library from outside the release and the operating system (e.g. /opt/homebrew) means
    the release is not self-contained. On Linux also: a library the release itself ships
    (by soname, e.g. libgomp.so.1, libstdc++.so.6) must resolve to the release's copy, not
    the system's, or which copy a process uses depends on import order.
    A scan that finds no extension, or reads no libraries from one, fails: a check that
    examined nothing must not pass.
    """
    extensions = sorted((release / "env" / "lib").glob("python3.*/site-packages/af/**/*.so"))
    if not extensions:
        raise SystemExit(f"no compiled af extension found in {release}: cannot check linkage")
    darwin = platform.system() == "Darwin"
    shipped = {path.name for path in (release / "env" / "lib").iterdir()}
    for extension in extensions:
        libraries = _linked_libraries(extension, darwin)
        if not libraries:
            raise SystemExit(f"read no linked libraries from {extension}: cannot check linkage")
        foreign = foreign_libraries(libraries, release, shipped, darwin=darwin)
        if foreign:
            raise SystemExit(
                f"{extension.name} links libraries from outside the release: {foreign}"
            )


MACOS_SYSTEM = ("/usr/lib/", "/System/", "@rpath/", "@loader_path/")
LINUX_SYSTEM = ("/lib/", "/lib64/", "/usr/lib/", "/usr/lib64/")


def foreign_libraries(
    libraries: list[tuple[str, str]], release: Path, shipped: set[str], *, darwin: bool
) -> list[str]:
    """The resolved paths among ``(name, path)`` pairs that break self-containment.

    macOS: anything outside the release and the OS (conda ships a libc++ too, but a
    system libc++ beside it is normal there: two-level namespaces keep them apart).
    Linux: also anything the release ships by the same soname but resolved elsewhere.
    """
    foreign = []
    for name, path in libraries:
        if path.startswith("@") or Path(path).resolve().is_relative_to(release.resolve()):
            continue
        if darwin:
            if not path.startswith(MACOS_SYSTEM):
                foreign.append(path)
        elif not path.startswith(LINUX_SYSTEM) or name in shipped:
            foreign.append(path)
    return foreign


def _linked_libraries(extension: Path, darwin: bool) -> list[tuple[str, str]]:
    tool = ["otool", "-L"] if darwin else ["ldd"]
    output = capture([*tool, str(extension)], {"PATH": "/usr/bin:/bin"})
    return parse_otool(output) if darwin else parse_ldd(output)


def parse_otool(output: str) -> list[tuple[str, str]]:
    """``otool -L`` lines after the first: the install name is both name and path."""
    paths = [line.split()[0] for line in output.splitlines()[1:] if line.strip()]
    return [(Path(path).name, path) for path in paths]


def parse_ldd(output: str) -> list[tuple[str, str]]:
    """``ldd`` lines ``soname => /resolved/path (0x…)``; unresolved or vdso lines are skipped."""
    pairs = []
    for line in output.splitlines():
        if "=>" not in line:
            continue
        name, _, rest = line.partition("=>")
        path = rest.split()[0] if rest.split() else ""
        if path.startswith("/"):
            pairs.append((name.strip(), path))
    return pairs


OPENMP_PROBE = """
import numpy as np
from af.map.optimise import MapProblem, TitreType, relax
a = np.random.default_rng(0).random((600, 600)); a @ a   # numpy's BLAS starts its threads
rng = np.random.default_rng(1)
truth = rng.uniform(-4, 4, (30, 2)); d = np.linalg.norm(truth[:24, None] - truth[None, 24:], axis=2)
cb = rng.uniform(6, 9, 6); v = np.round(cb[None] - d)
t = np.full(v.shape, TitreType.REGULAR, dtype=np.int8)
p = MapProblem(titre_value=v, titre_type=t, column_bases=cb,
               disconnected=np.zeros(30, dtype=bool), dodgy_is_regular=False)
relax(p, n_starts=8, seed=1, threads=4)                  # and the optimiser starts its own
print("one OpenMP runtime")
"""


def verify_one_openmp(python: str, env: dict[str, str]) -> None:
    """numpy's BLAS threads and the optimiser's threads in one process must not abort.

    Two OpenMP runtimes in one process abort it ("OMP: Error #15", exit 134). Imports
    alone don't show it; only running both thread pools does, so run both.
    """
    with tempfile.TemporaryDirectory() as scratch:
        result = subprocess.run(
            [python, "-I", "-c", OPENMP_PROBE], env=env, cwd=scratch, capture_output=True, text=True
        )
    if result.returncode != 0:
        raise SystemExit(
            f"numpy + optimiser threads in one process failed (exit {result.returncode}); "
            f"two OpenMP runtimes? {result.stderr.strip()[-400:]}"
        )


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
        "stage_seconds": stage_summary(),
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


MERGED_PR = re.compile(r"Merge pull request (#\d+) from [^/\s]+/(\S+)")


def contents(repo: Path, sha: str) -> str:
    """The pull requests merged into ``sha``'s history: the merges on its first-parent line.

    Every change reaches master through a pull request merged with a merge commit, whose
    subject names the PR and branch and whose body is the PR title.
    """
    log = git(repo, "log", "--first-parent", "--merges", "--format=%h%x1f%ci%x1f%s%x1f%b%x1e", sha)
    lines = []
    for record in filter(None, (part.strip() for part in log.split("\x1e"))):
        short, committed, subject, body = (record.split("\x1f") + ["", "", "", ""])[:4]
        date = committed[:10]  # %ci, not %cs: older gits (o's) lack %cs
        title = body.strip().splitlines()[0] if body.strip() else ""
        found = MERGED_PR.match(subject)
        if found:
            lines.append("\t".join((found[1], found[2], short, date, title)))
        else:
            lines.append("\t".join(("-", subject, short, date, title)))
    header = f"# pull requests merged into {sha} (newest first): pr, branch, merge, date, title\n"
    return header + "\n".join(lines) + "\n"


def git(repo: Path, *args: str) -> str:
    result = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit(
            f"git {' '.join(args)} failed in {repo}: {result.stderr.strip()} "
            "(fetch the commit into that clone first)"
        )
    return result.stdout.strip()


# Stages so far, as (key, start), for the timings printed and kept in RELEASE.toml. A release
# build on CSD3 took 7, 12 and then 36 min with the same environment.yml (csd3-v2, 2 Oct
# 2026), apparently all in "creating the environment", but unmeasured. Each stage now logs
# its wall time, so the next slow build says where the time went.
_stages: list[tuple[str, float]] = []


def step(message: str, key: str | None = None) -> None:
    """Log a stage (with UTC time); with ``key``, time it until the next step."""
    if _stages and _stages[-1][0]:
        name, started = _stages[-1]
        print(f"   ({name}: {time.monotonic() - started:.0f} s)", flush=True)
    _stages.append((key or "", time.monotonic()))
    print(f"== {now()} {message}", flush=True)


def stage_summary() -> str:
    """Seconds per timed stage, in order, e.g. "export=2 environment=118 ..."; the stage
    still running is counted up to now."""
    ends = [start for _, start in _stages[1:]] + [time.monotonic()]
    return " ".join(
        f"{key}={end - start:.0f}" for (key, start), end in zip(_stages, ends, strict=True) if key
    )


def now() -> str:
    return datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds")


if __name__ == "__main__":
    sys.exit(main())
