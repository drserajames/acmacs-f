"""tools/make-release.py's linkage gate, on real otool/ldd output (Linux can't run here)."""

import argparse
import importlib.util
import os
import shutil
import subprocess
import sys
import tomllib
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


def git_in(repo: Path, *args: str) -> str:
    done = subprocess.run(
        ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t", *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return done.stdout.strip()


def repo_with_two_pull_requests(repo: Path) -> None:
    """master: a first commit, then two pull requests merged as GitHub merges them."""
    repo.mkdir(exist_ok=True)
    git_in(repo, "init", "-q")
    git_in(repo, "symbolic-ref", "HEAD", "refs/heads/master")  # not `init -b`: git 2.23 lacks it
    git_in(repo, "commit", "-q", "--allow-empty", "-m", "start")
    for number, branch, title in [(1, "first-thing", "Add a"), (2, "second-thing", "Add b")]:
        git_in(repo, "checkout", "-q", "-b", branch)
        git_in(repo, "commit", "-q", "--allow-empty", "-m", f"work on {branch}")
        git_in(repo, "checkout", "-q", "master")
        message = f"Merge pull request #{number} from someone/{branch}\n\n{title}"
        git_in(repo, "merge", "-q", "--no-ff", branch, "-m", message)


def test_contents_lists_merged_pull_requests_newest_first(mr: ModuleType, tmp_path: Path) -> None:
    """CONTENTS.txt: a consumer greps for a PR number instead of walking git."""
    repo_with_two_pull_requests(tmp_path)
    lines = mr.contents(tmp_path, "HEAD").splitlines()
    assert lines[0].startswith("# pull requests merged into HEAD")
    rows = [line.split("\t") for line in lines[1:]]
    assert [(r[0], r[1], r[4]) for r in rows] == [
        ("#2", "second-thing", "Add b"),
        ("#1", "first-thing", "Add a"),
        ("-", "start", ""),  # the first commit was made on master itself
    ]
    assert all(len(r[3]) == 10 and r[3][4] == "-" for r in rows), "dates as YYYY-MM-DD"


def test_contents_lists_a_release_branchs_cherry_picks_with_their_sources(
    mr: ModuleType, tmp_path: Path
) -> None:
    """A release cut from an older commit plus picks: CONTENTS must not read as the older one."""
    repo_with_two_pull_requests(tmp_path)
    git_in(tmp_path, "checkout", "-q", "-b", "later", "master")
    git_in(tmp_path, "commit", "-q", "--allow-empty", "-m", "fix one thing")
    fix = git_in(tmp_path, "rev-parse", "HEAD")
    git_in(tmp_path, "commit", "-q", "--allow-empty", "-m", "and another")
    git_in(tmp_path, "checkout", "-q", "-b", "release", "master~1")  # #1 only
    git_in(tmp_path, "cherry-pick", "-x", "--allow-empty", fix)
    git_in(tmp_path, "commit", "-q", "--allow-empty", "-m", "a commit of the branch's own")
    rows = [line.split("\t") for line in mr.contents(tmp_path, "HEAD").splitlines()[1:]]
    assert [(r[0], r[1], r[4]) for r in rows] == [
        ("-", "a commit of the branch's own", ""),
        ("-", "fix one thing", f"cherry picked from {fix}"),
        ("#1", "first-thing", "Add a"),
        ("-", "start", ""),
    ]
    picked = git_in(tmp_path, "rev-parse", "--short", "HEAD~1")
    assert rows[1][2] == picked, "the commit in the release, not its source"


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


# The optimiser's arithmetic. A release's compiler is outside its package locks, so the build
# itself is tested; the stand-in test files and tools here are run for real, by a real python.
PASSING = "def test_a():\n    assert True\n\n\ndef test_b():\n    assert True\n"
SKIPPING = PASSING + "\n\nimport pytest\n\n\ndef test_c():\n    pytest.skip('not here')\n"
EXPECTED_TO_FAIL = (
    PASSING + "\n\nimport pytest\n\n\n@pytest.mark.xfail\ndef test_c():\n    assert False\n"
)
FAILING = PASSING + "\n\ndef test_c():\n    assert False\n"
BOTH = ("invented-fp-9", "original")


def commit_with(
    tmp_path: Path, numerics_tests: str | None, original_tool: str | None = None
) -> tuple[Path, Path]:
    """A commit's files (``src``) and the build's scratch directory, arranged as build() does."""
    src, scratch = tmp_path / "src", tmp_path / "scratch"
    (src / "tests" / "map").mkdir(parents=True)
    (src / "tools").mkdir()
    scratch.mkdir()
    if numerics_tests is not None:
        (src / "tests" / "map" / "test_numerics.py").write_text(numerics_tests)
    if original_tool is not None:
        (src / "tools" / "check-original-arithmetic.py").write_text(original_tool)
    (scratch / "tests").symlink_to(src / "tests")
    return src, scratch


def declaring(
    mr: ModuleType, monkeypatch: pytest.MonkeyPatch, default: str, offered: tuple[str, ...]
) -> None:
    declared = mr.Numerics(default, offered)
    monkeypatch.setattr(mr, "declared_numerics", lambda python, env: declared)


def test_a_build_that_keeps_its_arithmetic_is_released_under_its_names(
    mr: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> None:
    declaring(mr, monkeypatch, "invented-fp-9", BOTH)
    src, scratch = commit_with(tmp_path, PASSING)
    found = mr.verify_numerics(sys.executable, src, dict(os.environ), scratch)
    assert (found.default, found.offered) == ("invented-fp-9", BOTH)
    log = capfd.readouterr().out
    assert "invented-fp-9 (offered: invented-fp-9, original): 2 tests passed, none skipped" in log


@pytest.mark.parametrize(
    ("tests", "message"),
    [
        (SKIPPING, "'skipped': 1"),  # pytest itself exits 0 on a skip: only the count shows it
        (EXPECTED_TO_FAIL, "'skipped': 1"),  # an expected failure is a check that did not pass
        (FAILING, r"failed \(1\)"),
        ("", r"failed \(5\)"),  # a file with no test in it: nothing ran
    ],
)
def test_a_skipped_failed_or_absent_arithmetic_test_stops_the_release(
    mr: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tests: str, message: str
) -> None:
    declaring(mr, monkeypatch, "invented-fp-9", BOTH)
    src, scratch = commit_with(tmp_path, tests)
    with pytest.raises(SystemExit, match=message):
        mr.verify_numerics(sys.executable, src, dict(os.environ), scratch)


@pytest.mark.parametrize(
    "offered", [(), ("invented-fp-9",), ("original",), ("invented-fp-9", "invented-fp-9")]
)
def test_a_build_under_the_contract_must_offer_its_default_and_another(
    mr: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, offered: tuple[str, ...]
) -> None:
    declaring(mr, monkeypatch, "invented-fp-9", offered)
    src, scratch = commit_with(tmp_path, PASSING)
    with pytest.raises(SystemExit, match="must offer its default and the original"):
        mr.verify_numerics(sys.executable, src, dict(os.environ), scratch)


def test_a_commit_from_before_the_contract_is_released_with_no_name_and_says_so(
    mr: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    declaring(mr, monkeypatch, "", ())
    src, scratch = commit_with(tmp_path, None)
    found = mr.verify_numerics(sys.executable, src, dict(os.environ), scratch)
    assert (found.default, found.offered) == ("", ())
    assert "predates the arithmetic contract" in capsys.readouterr().out


def test_a_declared_arithmetic_and_its_tests_come_together_or_not_at_all(
    mr: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    declaring(mr, monkeypatch, "invented-fp-9", BOTH)
    src, scratch = commit_with(tmp_path / "declared", None)
    with pytest.raises(SystemExit, match="has no tests/map/test_numerics.py to check it with"):
        mr.verify_numerics(sys.executable, src, dict(os.environ), scratch)
    for offered in ((), BOTH):  # no default, with the tests or with names offered
        declaring(mr, monkeypatch, "", offered)
        src, scratch = commit_with(tmp_path / f"undeclared-{len(offered)}", PASSING)
        with pytest.raises(SystemExit, match="declares no default arithmetic"):
            mr.verify_numerics(sys.executable, src, dict(os.environ), scratch)


def test_a_report_with_no_suite_in_it_is_not_read_as_nothing_wrong(mr: ModuleType) -> None:
    with pytest.raises(SystemExit, match="no test suite"):
        mr.junit_counts("<testsuites></testsuites>")


def test_this_commits_own_arithmetic_tests_pass_the_gate_as_a_release_build_runs_them(
    mr: ModuleType, tmp_path: Path
) -> None:
    """The real file, the real conftest and pyproject, and the af this test run imports."""
    from af.map import optimise

    (tmp_path / "tests").symlink_to(REPO_ROOT / "tests")
    shutil.copy2(REPO_ROOT / "pyproject.toml", tmp_path / "pyproject.toml")
    found = mr.verify_numerics(sys.executable, REPO_ROOT, dict(os.environ), tmp_path)
    assert (found.default, found.offered) == (optimise.NUMERICS, optimise.NUMERICS_OFFERED)
    assert found.default and len(found.offered) == 2


# The original arithmetic against the release before. Stand-ins for the commit's tool: what
# it prints last, and its exit status.
def tool_saying(summary: str, status: int) -> str:
    return f"import sys\nprint('one problem: details')\nprint({summary!r})\nsys.exit({status})\n"


SAME = tool_saying("original arithmetic: the same bits on 1 problem(s) (here against there)", 0)
DIFFERENT = tool_saying("original arithmetic: DIFFERENT bits on one problem (here, there)", 1)
BEFORE = Path("/releases/0123456789ab")  # the release before; the stand-ins never run its python


def compare(
    mr: ModuleType,
    tmp_path: Path,
    tool: str | None,
    *,
    previous: Path | None = BEFORE,
    accept_change: bool = False,
) -> tuple[str, str]:
    src, scratch = commit_with(tmp_path, None, tool)
    found = mr.Tool(src / mr.ORIGINAL_TOOL, "c" * 40)
    answer = mr.compare_original(
        sys.executable, found, dict(os.environ), scratch, previous, accept_change=accept_change
    )
    assert answer.tool_commit == ("c" * 40 if previous else "")
    return answer.reference, answer.result


def test_the_same_original_bits_as_the_release_before_are_recorded_with_its_name(
    mr: ModuleType, tmp_path: Path, capfd: pytest.CaptureFixture[str]
) -> None:
    assert compare(mr, tmp_path, SAME) == ("0123456789ab", "same")
    assert "   original arithmetic: the same bits" in capfd.readouterr().out  # shown in the log


def test_different_original_bits_stop_the_release_unless_accepted_and_then_it_is_recorded(
    mr: ModuleType, tmp_path: Path
) -> None:
    with pytest.raises(SystemExit, match="--accept-original-change"):
        compare(mr, tmp_path / "refused", DIFFERENT)
    reference, result = compare(mr, tmp_path / "accepted", DIFFERENT, accept_change=True)
    assert reference == "0123456789ab" and result.startswith("different: accepted")


@pytest.mark.parametrize(
    "tool",
    [
        tool_saying("original arithmetic: COULD NOT COMPARE: the reference would not run", 2),
        "raise RuntimeError('died before comparing')\n",  # python exits 1 here too
        tool_saying("original arithmetic: the same bits on 1 problem(s)", 1),  # they disagree
        tool_saying("original arithmetic: DIFFERENT bits on one problem", 0),
        "",  # says nothing, exits 0
    ],
)
def test_a_comparison_that_was_not_made_stops_the_release_even_when_a_change_is_accepted(
    mr: ModuleType, tmp_path: Path, tool: str
) -> None:
    with pytest.raises(SystemExit, match="could not compare the original arithmetic"):
        compare(mr, tmp_path, tool, accept_change=True)


def test_the_first_release_on_a_platform_says_the_original_was_not_compared(
    mr: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    reference, result = compare(mr, tmp_path, SAME, previous=None)
    assert reference == "" and result == mr.NOT_COMPARED
    assert "not compared: no earlier release" in capsys.readouterr().out


def test_a_commit_without_the_tool_cannot_have_its_original_compared(
    mr: ModuleType, tmp_path: Path
) -> None:
    with pytest.raises(SystemExit, match=r"no tools/check-original-arithmetic\.py"):
        compare(mr, tmp_path, None)


def test_this_commits_own_tool_answers_the_gate_in_the_words_it_reads(
    mr: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The real tool, with this python as the release before: the original against itself.

    The gate reads the tool's summary line as well as its exit status, so a reworded summary
    must fail here, in CI, not in the next release build.
    """
    monkeypatch.setattr(mr, "python_of", lambda release: sys.executable)
    tool = mr.Tool(REPO_ROOT / mr.ORIGINAL_TOOL, "c" * 40)
    result = mr.compare_original(
        sys.executable, tool, dict(os.environ), tmp_path, BEFORE, accept_change=False
    )
    assert result == mr.Original("0123456789ab", "same", "c" * 40)


def tools_repo(repo: Path, files: dict[str, str]) -> str:
    """A clone whose one commit holds ``files``; its sha."""
    for name, text in files.items():
        (repo / name).parent.mkdir(parents=True, exist_ok=True)
        (repo / name).write_text(text)
    git_in(repo, "init", "-q")
    git_in(repo, "add", ".")
    git_in(repo, "commit", "-q", "-m", "tools")
    return git_in(repo, "rev-parse", "HEAD")


def test_a_commit_under_the_contract_is_compared_by_its_own_tool(
    mr: ModuleType, tmp_path: Path
) -> None:
    src = tmp_path / "src"
    tool = mr.original_tool(src, "a" * 40, mr.Numerics("invented-fp-9", BOTH), tmp_path, "x", src)
    assert tool == mr.Tool(src / mr.ORIGINAL_TOOL, "a" * 40)  # no git asked: "x" is no rev


def test_a_commit_from_before_the_contract_is_compared_by_the_tools_revs_tool_and_table(
    mr: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Both files, beside each other as in a checkout: the tool finds its table by its own path."""
    repo, scratch = tmp_path / "clone", tmp_path / "scratch"
    repo.mkdir()
    scratch.mkdir()
    tools_sha = tools_repo(repo, {mr.ORIGINAL_TOOL: SAME, mr.NUMERICS_TESTS: "TABLE = 1\n"})
    tool = mr.original_tool(tmp_path / "src", "a" * 40, mr.Numerics("", ()), repo, "HEAD", scratch)
    assert tool.commit == tools_sha
    assert tool.path.read_text() == SAME
    assert (tool.path.parents[1] / mr.NUMERICS_TESTS).read_text() == "TABLE = 1\n"
    assert f"compared by {mr.ORIGINAL_TOOL} from HEAD ({tools_sha[:12]})" in capsys.readouterr().out
    answer = mr.compare_original(
        sys.executable, tool, dict(os.environ), scratch, BEFORE, accept_change=False
    )
    assert answer == mr.Original("0123456789ab", "same", tools_sha)


def test_a_tools_rev_without_the_tool_and_table_stops_the_release(
    mr: ModuleType, tmp_path: Path
) -> None:
    for name, files in [
        ("neither", {"README": "x\n"}),
        ("no-table", {"tools/check-original-arithmetic.py": SAME}),
    ]:
        repo = tmp_path / name
        repo.mkdir()
        tools_repo(repo, files)
        scratch = tmp_path / f"scratch-{name}"
        scratch.mkdir()
        with pytest.raises(SystemExit, match=r"--tools-rev HEAD .* has no tools/check-original"):
            mr.original_tool(tmp_path / "src", "a" * 40, mr.Numerics("", ()), repo, "HEAD", scratch)


def test_the_real_tool_and_table_answer_the_gate_when_taken_from_a_tools_rev(
    mr: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """This commit's tool and table, archived from a clone as a release build takes them."""
    repo, scratch = tmp_path / "clone", tmp_path / "scratch"
    repo.mkdir()
    scratch.mkdir()
    files = {name: (REPO_ROOT / name).read_text() for name in (mr.ORIGINAL_TOOL, mr.NUMERICS_TESTS)}
    tools_sha = tools_repo(repo, files)
    tool = mr.original_tool(tmp_path / "src", "a" * 40, mr.Numerics("", ()), repo, "HEAD", scratch)
    monkeypatch.setattr(mr, "python_of", lambda release: sys.executable)
    answer = mr.compare_original(
        sys.executable, tool, dict(os.environ), scratch, BEFORE, accept_change=False
    )
    assert answer == mr.Original("0123456789ab", "same", tools_sha)


def test_the_release_file_names_the_arithmetics_and_the_comparison(
    mr: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(mr, "capture", lambda command, env: "something 1.0\n")
    monkeypatch.setattr(mr, "_stages", [("record", mr.time.monotonic())])  # as within a build
    args = argparse.Namespace(rev="a-branch", micromamba=Path("micromamba"), extras="dev")
    cases = [
        (mr.Numerics("invented-fp-9", BOTH), mr.Original("0123456789ab", "same", "0" * 40)),
        # a commit from before the contract, compared by a later commit's tool
        (mr.Numerics("", ()), mr.Original("0123456789ab", "same", "f" * 40)),
        (mr.Numerics("", ()), mr.Original("", mr.NOT_COMPARED, "")),
    ]
    for numerics, original in cases:
        mr.write_release_file(args, tmp_path, "0" * 40, "then", {"CXX": "c++"}, numerics, original)
        written = tomllib.loads((tmp_path / "RELEASE.toml").read_text())
        assert written["numerics_default"] == numerics.default
        assert written["numerics_offered"] == list(numerics.offered)
        assert (
            written["original_reference"],
            written["original_result"],
            written["original_tool_commit"],
        ) == (original.reference, original.result, original.tool_commit)
        assert written["commit"] == "0" * 40 and mr.finished_at(
            tmp_path
        )  # the old lines still read


def test_the_imports_check_wants_every_core_the_install_has_inside_the_release(
    mr: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    release = tmp_path / "0123456789ab"
    inside = [str(release / "env" / name) for name in ("af/x.py", "af/map/a.so", "af/map/b.so")]
    for lines in (inside[:2], inside):  # before the contract: one core; after: two
        monkeypatch.setattr(mr, "capture", lambda command, env, lines=lines: "\n".join(lines))
        mr.verify_imports(release)
    for lines in (inside[:1], [*inside[:2], "/elsewhere/af/map/b.so"]):
        monkeypatch.setattr(mr, "capture", lambda command, env, lines=lines: "\n".join(lines))
        with pytest.raises(SystemExit, match="does not import from inside"):
            mr.verify_imports(release)
