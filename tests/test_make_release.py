"""tools/make-release.py's linkage gate, on real otool/ldd output (Linux can't run here)."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

from tests.helpers import REPO_ROOT

RELEASE = Path("/syn/acmacs-f/releases/abc")


@pytest.fixture(scope="module")
def mr() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "make_release", REPO_ROOT / "tools" / "make-release.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["make_release"] = module
    spec.loader.exec_module(module)
    return module


# o's release b6da3f4e06df (28 Sep): _core built with system g++, no RUNPATH.
O_LDD = """\
\tlinux-vdso.so.1 (0x00007ffd1d7f8000)
\tlibgomp.so.1 => /lib/x86_64-linux-gnu/libgomp.so.1 (0x0000748df217d000)
\tlibstdc++.so.6 => /lib/x86_64-linux-gnu/libstdc++.so.6 (0x0000748df1e00000)
\tlibc.so.6 => /lib/x86_64-linux-gnu/libc.so.6 (0x0000748df1a00000)
\t/lib64/ld-linux-x86-64.so.2 (0x0000748df21e0000)
"""


def test_parse_ldd(mr: ModuleType) -> None:
    assert mr.parse_ldd(O_LDD) == [
        ("libgomp.so.1", "/lib/x86_64-linux-gnu/libgomp.so.1"),
        ("libstdc++.so.6", "/lib/x86_64-linux-gnu/libstdc++.so.6"),
        ("libc.so.6", "/lib/x86_64-linux-gnu/libc.so.6"),
    ]


def test_linux_system_copy_of_a_shipped_library_is_foreign(mr: ModuleType) -> None:
    """o's case: the env ships libgomp and libstdc++, ldd resolved the system's."""
    shipped = {"libgomp.so.1", "libgomp.so.1.0.0", "libstdc++.so.6", "libpython3.12.so"}
    foreign = mr.foreign_libraries(mr.parse_ldd(O_LDD), RELEASE, shipped, darwin=False)
    assert foreign == ["/lib/x86_64-linux-gnu/libgomp.so.1", "/lib/x86_64-linux-gnu/libstdc++.so.6"]


def test_linux_runpath_build_passes(mr: ModuleType) -> None:
    fixed = O_LDD.replace("/lib/x86_64-linux-gnu/libgomp.so.1", f"{RELEASE}/env/lib/libgomp.so.1")
    fixed = fixed.replace(
        "/lib/x86_64-linux-gnu/libstdc++.so.6", f"{RELEASE}/env/lib/libstdc++.so.6"
    )
    shipped = {"libgomp.so.1", "libstdc++.so.6"}
    assert mr.foreign_libraries(mr.parse_ldd(fixed), RELEASE, shipped, darwin=False) == []


def test_linux_non_system_path_is_foreign(mr: ModuleType) -> None:
    libs = [("libfoo.so.1", "/opt/other/lib/libfoo.so.1")]
    assert mr.foreign_libraries(libs, RELEASE, set(), darwin=False) == [
        "/opt/other/lib/libfoo.so.1"
    ]


def test_macos_rules(mr: ModuleType) -> None:
    otool = (
        "_core.so:\n"
        "\t@rpath/libomp.dylib (compatibility version 5.0.0)\n"
        "\t/usr/lib/libc++.1.dylib (compatibility version 1.0.0)\n"
    )
    shipped = {"libomp.dylib", "libc++.1.dylib"}  # conda ships libc++ too: fine on macOS
    assert mr.foreign_libraries(mr.parse_otool(otool), RELEASE, shipped, darwin=True) == []
    homebrew = otool.replace("@rpath/libomp.dylib", "/opt/homebrew/opt/libomp/lib/libomp.dylib")
    assert mr.foreign_libraries(mr.parse_otool(homebrew), RELEASE, shipped, darwin=True) == [
        "/opt/homebrew/opt/libomp/lib/libomp.dylib"
    ]


def test_contents_lists_merged_pull_requests_newest_first(mr: ModuleType, tmp_path: Path) -> None:
    """CONTENTS.txt: a consumer greps for a PR number instead of walking git."""
    import subprocess

    def git(*args: str) -> None:
        subprocess.run(
            ["git", "-C", str(tmp_path), "-c", "user.name=t", "-c", "user.email=t@t", *args],
            check=True,
            capture_output=True,
        )

    git("init", "-q")
    git("symbolic-ref", "HEAD", "refs/heads/master")  # not `init -b`: git 2.23 lacks it
    git("commit", "-q", "--allow-empty", "-m", "start")
    for number, branch, title in [(1, "first-thing", "Add a"), (2, "second-thing", "Add b")]:
        git("checkout", "-q", "-b", branch)
        git("commit", "-q", "--allow-empty", "-m", f"work on {branch}")
        git("checkout", "-q", "master")
        message = f"Merge pull request #{number} from someone/{branch}\n\n{title}"
        git("merge", "-q", "--no-ff", branch, "-m", message)
    lines = mr.contents(tmp_path, "HEAD").splitlines()
    assert lines[0].startswith("# pull requests merged into HEAD")
    rows = [line.split("\t") for line in lines[1:]]
    assert [(r[0], r[1], r[4]) for r in rows] == [
        ("#2", "second-thing", "Add b"),
        ("#1", "first-thing", "Add a"),
    ]
    assert all(len(r[3]) == 10 and r[3][4] == "-" for r in rows), "dates as YYYY-MM-DD"


def test_stage_times_are_logged_and_summarised(
    mr: ModuleType, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Each timed stage reports its wall time, so a slow build says where the time went."""
    clock = iter([0.0, 2.0, 2.0, 120.0, 120.0, 125.0, 125.0, 140.0])
    monkeypatch.setattr(mr.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(mr, "_stages", [])
    mr.step("exporting", "export")  # t=0
    mr.step("creating the environment", "environment")  # export took 2 s
    mr.step("verifying", "verify")  # environment took 118 s
    mr.step("untimed note")  # verify took 5 s; no key: not timed
    assert mr.stage_summary() == "export=2 environment=118 verify=5"
    log = capsys.readouterr().out
    assert "(export: 2 s)" in log and "(environment: 118 s)" in log and "(verify: 5 s)" in log


# Package locks. Explicit-lock lines in micromamba's format; the packages are public conda
# packages, the versions invented.
def lock_text(platform_name: str, packages: dict[str, str]) -> str:
    head = f"# created by micromamba\n# platform: {platform_name}\n@EXPLICIT\n"
    lines = [
        f"https://conda.anaconda.org/conda-forge/{platform_name}/{name}-{vb}.conda#00{i}"
        for i, (name, vb) in enumerate(sorted(packages.items()))
    ]
    return head + "\n".join(lines) + "\n"


FREEZE = "acmacs-f @ file:///r/src\nnumpy @ file:///b/numpy\npytest==9.0.0\nruff==0.16.0\n"


def make_release(
    root: Path,
    name: str,
    *,
    platform_name: str,
    env_yml: str,
    finished: str,
    packages: dict[str, str],
    freeze: str = FREEZE,
) -> Path:
    release = root / name
    (release / "src").mkdir(parents=True)
    (release / "src" / "environment.yml").write_text(env_yml)
    (release / "conda-explicit.txt").write_text(lock_text(platform_name, packages))
    (release / "pip-freeze.txt").write_text(freeze)
    (release / "RELEASE.toml").write_text(f'commit = "{name}"\nfinished = "{finished}"\n')
    return release


PKGS = {"python": "3.12.1-h0_0", "nextclade": "3.23.0-h1_0", "sqlite": "3.53.4-h2_1"}


@pytest.fixture
def releases(tmp_path: Path) -> Path:
    """Three releases: two osx-arm64 with one environment.yml, one linux-64, and a
    half-made one (no RELEASE.toml) that must never be chosen."""
    make_release(
        tmp_path,
        "aaa",
        platform_name="osx-arm64",
        env_yml="e1",
        finished="2026-10-01T10:00:00+00:00",
        packages=PKGS,
    )
    make_release(
        tmp_path,
        "bbb",
        platform_name="osx-arm64",
        env_yml="e1",
        finished="2026-10-02T09:00:00+00:00",
        packages={**PKGS, "nextclade": "3.24.0-h1_0"},
    )
    make_release(
        tmp_path,
        "ccc",
        platform_name="linux-64",
        env_yml="e1",
        finished="2026-10-03T09:00:00+00:00",
        packages=PKGS,
    )
    half = make_release(
        tmp_path,
        "ddd",
        platform_name="osx-arm64",
        env_yml="e1",
        finished="2026-10-04T09:00:00+00:00",
        packages=PKGS,
    )
    (half / "RELEASE.toml").unlink()
    return tmp_path


def test_lock_platform_is_read_from_the_lock(mr: ModuleType) -> None:
    assert mr.lock_platform(lock_text("linux-64", PKGS)) == "linux-64"
    with pytest.raises(SystemExit, match="platform"):
        mr.lock_platform("@EXPLICIT\nhttps://x/osx-arm64/a-1-0.conda\n")


def test_earlier_releases_are_this_platforms_complete_ones_newest_first(
    mr: ModuleType, releases: Path
) -> None:
    assert [r.name for r in mr.earlier_releases(releases, "osx-arm64")] == ["bbb", "aaa"]
    assert [r.name for r in mr.earlier_releases(releases, "linux-64")] == ["ccc"]


def test_same_environment_yml_replays_the_newest_lock_on_this_platform(
    mr: ModuleType, releases: Path
) -> None:
    choice = mr.choose_lock(releases, b"e1", "osx-arm64", resolve=False)
    assert (choice.replay.name, choice.previous.name) == ("bbb", "bbb")
    assert "replayed the lock of bbb (osx-arm64)" in choice.reason


def test_a_lock_is_never_replayed_on_another_platform(mr: ModuleType, releases: Path) -> None:
    """ccc (linux-64) is the newest release with this environment.yml; osx must not take it,
    and the first build on a new platform solves afresh and says so."""
    assert mr.choose_lock(releases, b"e1", "osx-arm64", resolve=False).replay.name == "bbb"
    first = mr.choose_lock(releases, b"e1", "linux-aarch64", resolve=False)
    assert (first.replay, first.previous) == (None, None)
    assert "first release on linux-aarch64" in first.reason


def test_changed_environment_yml_or_resolve_solves_afresh(mr: ModuleType, releases: Path) -> None:
    changed = mr.choose_lock(releases, b"e2", "osx-arm64", resolve=False)
    assert changed.replay is None and changed.previous.name == "bbb"
    assert "environment.yml changed since bbb" in changed.reason
    forced = mr.choose_lock(releases, b"e1", "osx-arm64", resolve=True)
    assert forced.replay is None and forced.previous.name == "bbb" and "--resolve" in forced.reason


def test_package_section_is_generated_from_the_two_locks(mr: ModuleType) -> None:
    old = lock_text("osx-arm64", PKGS)
    new = lock_text(
        "osx-arm64",
        {**PKGS, "nextclade": "3.24.0-h1_0", "sqlite": "3.53.4-h2_102", "zlib": "1.3-h3_0"},
    )
    choice = mr.LockChoice(
        None, Path("/r/bbb"), "osx-arm64", "solved afresh (osx-arm64): --resolve"
    )
    section = mr.package_section(
        choice, (old, FREEZE), new, FREEZE.replace("ruff==0.16.0", "ruff==0.16.1")
    )
    assert section.splitlines() == [
        "# packages: solved afresh (osx-arm64): --resolve",
        "# packages vs bbb: conda 3 changed/added/removed of 4; pip pins 1 changed/added/removed",
        "#   conda nextclade 3.23.0 h1_0 -> 3.24.0 h1_0",
        "#   conda sqlite 3.53.4 h2_1 -> 3.53.4 h2_102",
        "#   conda + zlib 1.3 h3_0",
        "#   pip ruff 0.16.0 -> 0.16.1",
    ]
    first = mr.LockChoice(
        None, None, "linux-64", "solved afresh: the first release on linux-64 in /r"
    )
    assert "no earlier release on linux-64" in mr.package_section(first, None, new, FREEZE)


def test_contents_pr_lines_still_grep_cleanly_under_the_package_header(mr: ModuleType) -> None:
    """Consumers grep '^#182' in CONTENTS.txt: package lines start '# ', never '#<digit>'."""
    choice = mr.LockChoice(None, Path("/r/bbb"), "osx-arm64", "x")
    new = lock_text("osx-arm64", {**PKGS, "python": "3.12.2-h0_0"})
    section = mr.package_section(choice, (lock_text("osx-arm64", PKGS), FREEZE), new, FREEZE)
    assert all(line.startswith("# ") for line in section.splitlines())


def test_pip_pins_skip_url_lines_and_names_normalise(mr: ModuleType) -> None:
    pins = mr.pip_pins("acmacs-f @ file:///s\nPygments==2.21.0\nast_serialize==0.12.1\n")
    assert pins == {"pygments": "Pygments==2.21.0", "ast-serialize": "ast_serialize==0.12.1"}


def test_replay_check_ignores_where_af_came_from_but_nothing_else(
    mr: ModuleType, releases: Path
) -> None:
    bbb = releases / "bbb"
    lock = (bbb / "conda-explicit.txt").read_text()
    editable = "# Editable install with no version control (acmacs-f==0.1)\n-e /w/acmacs-f-x\n"
    mr.check_replay(bbb, lock, editable + FREEZE.replace("acmacs-f @ file:///r/src\n", ""))
    with pytest.raises(SystemExit, match="conda-explicit.txt differs"):
        mr.check_replay(bbb, lock.replace("3.24.0", "3.25.0"), FREEZE)
    with pytest.raises(SystemExit, match="pip-freeze.txt differs"):
        mr.check_replay(bbb, lock, FREEZE.replace("ruff==0.16.0", "ruff==0.16.1"))


def test_dev_env_refuses_a_lock_for_another_platform(mr: ModuleType, releases: Path) -> None:
    """Checked before anything is created, so no micromamba is needed to see the refusal."""
    import subprocess

    other = "ccc" if mr.conda_platform() != "linux-64" else "bbb"
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tools" / "dev-env.py"),
            "--release",
            str(releases / other),
            "--prefix",
            str(releases / "env"),
            "--worktree",
            str(REPO_ROOT),
            "--micromamba",
            "/nonexistent/micromamba",
            "--mamba-root",
            str(releases / "m"),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0 and "this machine is" in result.stderr
    assert not (releases / "env").exists()
