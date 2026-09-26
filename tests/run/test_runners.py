"""af.run: a failing tool propagates failure, a missing artefact is fatal, for both runners.

The SLURM runner is tested against a fake ``sbatch`` (a Python script written by the
test) that runs each array task locally, in the same way SLURM would: it sets
SLURM_ARRAY_TASK_ID, runs the batch script, and exits with the highest task status.
It can also simulate a task killed by SLURM before it records its status.
"""

import json
import shutil
from pathlib import Path

import pytest

from af.run import Job, JobFailed, LocalRunner, Resources, Runner, SlurmRunner
from af.util.artefacts import Artefact


@pytest.fixture(params=["local", "slurm"])
def runner(request: pytest.FixtureRequest, tmp_path: Path) -> Runner:
    if request.param == "local":
        return LocalRunner(max_parallel=2)
    sbatch = request.getfixturevalue("fake_sbatch")
    return SlurmRunner(work_dir=tmp_path / "slurm", sbatch=str(sbatch), partition="example")


def shell_job(tmp_path: Path, name: str, script: str, outputs: list[str]) -> Job:
    return Job(
        name=name,
        command=["/bin/sh", "-c", script],
        cwd=tmp_path,
        log=tmp_path / "logs" / f"{name}.log",
        outputs=[Artefact(tmp_path / output) for output in outputs],
    )


def test_success(runner: Runner, tmp_path: Path) -> None:
    job = shell_job(tmp_path, "write", "echo hello > out.txt; echo done", ["out.txt"])
    result = runner.run(job)
    assert result.returncode == 0
    assert [artefact.path for artefact in result.artefacts] == [tmp_path / "out.txt"]
    assert (tmp_path / "logs" / "write.log").read_text() == "done\n"


def test_failing_tool_propagates_failure(runner: Runner, tmp_path: Path) -> None:
    job = shell_job(tmp_path, "fails", "echo written > out.txt; echo boom >&2; exit 3", ["out.txt"])
    with pytest.raises(JobFailed) as caught:
        runner.run(job)
    assert [failure.reason for failure in caught.value.failures] == ["exit status 3"]
    assert "boom" in str(caught.value), "the log tail belongs in the error"


def test_missing_artefact_is_fatal(runner: Runner, tmp_path: Path) -> None:
    job = shell_job(tmp_path, "lazy", "exit 0", ["never-written.txt"])
    expected = "exit status 0 but bad output: .*never-written.txt: missing"
    with pytest.raises(JobFailed, match=expected):
        runner.run(job)


def test_empty_artefact_is_fatal(runner: Runner, tmp_path: Path) -> None:
    job = shell_job(tmp_path, "empty", ": > out.txt", ["out.txt"])
    with pytest.raises(JobFailed, match="empty file"):
        runner.run(job)


def test_run_many_waits_for_all_and_reports_every_failure(runner: Runner, tmp_path: Path) -> None:
    jobs = [
        shell_job(tmp_path, "a", "exit 2", ["a.txt"]),
        shell_job(tmp_path, "b", "sleep 0.2; echo b > b.txt", ["b.txt"]),
        shell_job(tmp_path, "c", "exit 0", ["c.txt"]),
    ]
    with pytest.raises(JobFailed) as caught:
        runner.run_many(jobs)
    assert {failure.job.name: failure.reason for failure in caught.value.failures} == {
        "a": "exit status 2",
        "c": f"exit status 0 but bad output: {tmp_path / 'c.txt'}: missing",
    }
    assert (tmp_path / "b.txt").read_text() == "b\n", "the slow job still finished"


def test_run_many_results_in_job_order(runner: Runner, tmp_path: Path) -> None:
    jobs = [shell_job(tmp_path, f"j{i}", f"echo {i} > {i}.txt", [f"{i}.txt"]) for i in range(4)]
    results = runner.run_many(jobs)
    assert [result.job.name for result in results] == ["j0", "j1", "j2", "j3"]


def test_duplicate_job_names_rejected(runner: Runner, tmp_path: Path) -> None:
    job = shell_job(tmp_path, "same", "exit 0", ["x"])
    with pytest.raises(ValueError, match="unique"):
        runner.run_many([job, job])


def test_env_is_added(runner: Runner, tmp_path: Path) -> None:
    job = Job(
        name="env",
        command=["/bin/sh", "-c", 'printf "%s" "$AF_EXAMPLE" > out.txt'],
        cwd=tmp_path,
        log=tmp_path / "env.log",
        outputs=[Artefact(tmp_path / "out.txt")],
        env={"AF_EXAMPLE": "it's set"},
    )
    runner.run(job)
    assert (tmp_path / "out.txt").read_text() == "it's set"


def test_local_tool_not_found(tmp_path: Path) -> None:
    job = Job("missing-tool", ["/nonexistent/tool"], tmp_path, tmp_path / "l.log", [])
    with pytest.raises(JobFailed, match="could not start"):
        LocalRunner().run(job)


def test_slurm_killed_task_is_a_failure(
    fake_sbatch: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A task SLURM kills never records a status, even if a partial output exists."""
    (tmp_path / "b.txt").write_text("stale output from an earlier run")
    monkeypatch.setenv("KILL_TASKS", "1")
    runner = SlurmRunner(work_dir=tmp_path / "slurm", sbatch=str(fake_sbatch))
    jobs = [
        shell_job(tmp_path, "a", "echo a > a.txt", ["a.txt"]),
        shell_job(tmp_path, "b", "echo b > b.txt", ["b.txt"]),
    ]
    with pytest.raises(JobFailed) as caught:
        runner.run_many(jobs)
    assert [(f.job.name, f.reason) for f in caught.value.failures] == [
        ("b", "no exit status recorded (killed, or hit its time limit?)")
    ]


def test_slurm_sbatch_missing(tmp_path: Path) -> None:
    runner = SlurmRunner(work_dir=tmp_path / "slurm", sbatch="/nonexistent/sbatch")
    with pytest.raises(JobFailed, match="could not run sbatch"):
        runner.run(shell_job(tmp_path, "a", "exit 0", ["a.txt"]))


def test_slurm_sbatch_rejects_submission(tmp_path: Path) -> None:
    sbatch = tmp_path / "sbatch"
    sbatch.write_text("#!/bin/sh\necho 'invalid partition' >&2\nexit 1\n")
    sbatch.chmod(0o755)
    runner = SlurmRunner(work_dir=tmp_path / "slurm", sbatch=str(sbatch))
    with pytest.raises(JobFailed, match=r"sbatch failed \(1\): invalid partition"):
        runner.run(shell_job(tmp_path, "a", "exit 0", ["a.txt"]))


def test_slurm_arguments(fake_sbatch: Path, tmp_path: Path) -> None:
    """One array per distinct resource request; resources map to sbatch options."""
    runner = SlurmRunner(
        work_dir=tmp_path / "slurm",
        sbatch=str(fake_sbatch),
        partition="part",
        account="acct",
        extra_args=("--qos=example",),
        max_parallel_tasks=5,
    )
    big = Resources(threads=8, memory_gb=2, time_limit_minutes=90)
    jobs = [
        shell_job(tmp_path, "small1", "echo > s1.txt", ["s1.txt"]),
        shell_job(tmp_path, "small2", "echo > s2.txt", ["s2.txt"]),
        Job(
            "big",
            ["/bin/sh", "-c", "echo > big.txt"],
            tmp_path,
            tmp_path / "big.log",
            [Artefact(tmp_path / "big.txt")],
            resources=big,
        ),
    ]
    runner.run_many(jobs)
    calls = sorted((fake_sbatch.parent / "sbatch-calls.txt").read_text().splitlines())
    assert len(calls) == 2
    big_call = next(call for call in calls if "--cpus-per-task=8" in call)
    small_call = next(call for call in calls if "--cpus-per-task=1" in call)
    for option in [
        "--wait",
        "--array=0-0%5",
        "--mem=2048M",
        "--time=90",
        "--partition=part",
        "--account=acct",
        "--qos=example",
    ]:
        assert option in big_call.split(), option
    assert "--array=0-1%5" in small_call.split()
    assert "--mem" not in small_call and "--time" not in small_call
    names = [next(a for a in call.split() if a.startswith("--job-name=")) for call in calls]
    assert names[0] != names[1], "each submission has its own job name, for scancel --name"
    assert any(name.startswith("--job-name=af-big-") for name in names)
    assert any(name.startswith("--job-name=af-small1+1-") for name in names)


def test_slurm_splits_arrays_at_max_array_size(fake_sbatch: Path, tmp_path: Path) -> None:
    runner = SlurmRunner(work_dir=tmp_path / "slurm", sbatch=str(fake_sbatch), max_array_size=2)
    jobs = [shell_job(tmp_path, f"j{i}", f"echo {i} > {i}.txt", [f"{i}.txt"]) for i in range(5)]
    results = runner.run_many(jobs)
    assert [result.job.name for result in results] == ["j0", "j1", "j2", "j3", "j4"]
    calls = (fake_sbatch.parent / "sbatch-calls.txt").read_text().splitlines()
    arrays = sorted(next(a for a in c.split() if a.startswith("--array=")) for c in calls)
    assert arrays == ["--array=0-0", "--array=0-1", "--array=0-1"]


def test_slurm_interrupt_cancels_submitted_jobs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ctrl-C while sbatch --wait blocks must scancel, or the jobs run on unwatched."""
    scancel = tmp_path / "scancel"
    scancel.write_text(f'#!/bin/sh\necho "$@" >> {tmp_path}/scancel-calls.txt\n')
    scancel.chmod(0o755)
    import af.run.slurm as slurm_module

    real_run = slurm_module.subprocess.run

    def fake_run(argv: list[str], **kwargs: object) -> object:
        if argv[0] == "sbatch":
            raise KeyboardInterrupt
        return real_run(argv, **kwargs)  # type: ignore[call-overload]

    monkeypatch.setattr(slurm_module.subprocess, "run", fake_run)
    runner = SlurmRunner(work_dir=tmp_path / "slurm", scancel=str(scancel))
    with pytest.raises(KeyboardInterrupt):
        runner.run(shell_job(tmp_path, "long", "sleep 1000", ["x.txt"]))
    cancelled = (tmp_path / "scancel-calls.txt").read_text().split()
    assert len(cancelled) == 1 and cancelled[0].startswith("--name=af-long-")
    batch = next((tmp_path / "slurm").iterdir())
    assert cancelled[0] == f"--name=af-long-{batch.name.removeprefix('batch-')}"


def test_python_module_job(runner: Runner, tmp_path: Path) -> None:
    """Python work runs as `python -m module` with this interpreter, on any runner."""
    out = tmp_path / "probe.json"
    job = Job.python_module(
        "probe",
        "af.run.smoke",
        ["--child", str(out)],
        cwd=tmp_path,
        log=tmp_path / "probe.log",
        outputs=[Artefact(out)],
    )
    runner.run(job)
    assert json.loads(out.read_text())["af_version"]


@pytest.mark.slurm
def test_real_slurm(tmp_path_factory: pytest.TempPathFactory) -> None:
    """On a real cluster. pytest's temporary directory must be on a filesystem the
    compute nodes share (run with --basetemp=<shared dir> if it is not)."""
    if not shutil.which("sbatch"):
        pytest.skip("sbatch not on PATH")
    base = tmp_path_factory.mktemp("slurm-real")
    runner: Runner = SlurmRunner(work_dir=base / "work", output_wait_seconds=30)
    with pytest.raises(JobFailed):
        runner.run(shell_job(base, "fails", "exit 7", ["x.txt"]))
    assert runner.run(shell_job(base, "ok", "hostname > h.txt", ["h.txt"])).returncode == 0


def test_python_module_keeps_the_no_site_flag(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import af.run.job as job_module

    class Flags:
        no_site = 1

    monkeypatch.setattr(job_module.sys, "flags", Flags())
    job = Job.python_module("p", "af.run.smoke", [], cwd=tmp_path, log=tmp_path / "l", outputs=[])
    assert job.argv()[1:4] == ["-S", "-m", "af.run.smoke"]


def test_jobs_get_omp_num_threads_from_their_resources(
    runner: Runner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A preset OMP_NUM_THREADS=1 (CSD3's login env) must not starve a job of its threads."""
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    job = Job(
        "threads",
        ["/bin/sh", "-c", 'printf "%s" "$OMP_NUM_THREADS" > t.txt'],
        tmp_path,
        tmp_path / "t.log",
        [Artefact(tmp_path / "t.txt")],
        resources=Resources(threads=3),
    )
    runner.run(job)
    assert (tmp_path / "t.txt").read_text() == "3"


def test_explicit_env_overrides_omp_num_threads(runner: Runner, tmp_path: Path) -> None:
    job = Job(
        "override",
        ["/bin/sh", "-c", 'printf "%s" "$OMP_NUM_THREADS" > t.txt'],
        tmp_path,
        tmp_path / "t.log",
        [Artefact(tmp_path / "t.txt")],
        resources=Resources(threads=3),
        env={"OMP_NUM_THREADS": "2"},
    )
    runner.run(job)
    assert (tmp_path / "t.txt").read_text() == "2"


def test_slurm_time_limit_kill_is_not_a_failed_submission(
    fake_sbatch: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A job killed at its limit makes sbatch --wait exit nonzero with no status file.

    It was submitted (sbatch printed a job id), so it must be reported as killed, not as
    "sbatch failed", with SLURM's state when sacct can say.
    """
    monkeypatch.setenv("KILL_TASKS", "0")
    sacct = tmp_path / "sacct"
    sacct.write_text('#!/bin/sh\nprintf "12345|TIMEOUT\\n12345_0|TIMEOUT\\n"\n')
    sacct.chmod(0o755)
    runner = SlurmRunner(work_dir=tmp_path / "slurm", sbatch=str(fake_sbatch), sacct=str(sacct))
    with pytest.raises(JobFailed) as caught:
        runner.run(shell_job(tmp_path, "slow", "echo x > x.txt", ["x.txt"]))
    reason = caught.value.failures[0].reason
    assert "sbatch failed" not in reason
    assert (
        reason == "no exit status recorded (killed, or hit its time limit?) [SLURM state TIMEOUT]"
    )


def test_slurm_kill_without_sacct_still_fails_plainly(
    fake_sbatch: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("KILL_TASKS", "0")
    runner = SlurmRunner(
        work_dir=tmp_path / "slurm", sbatch=str(fake_sbatch), sacct="/nonexistent/sacct"
    )
    with pytest.raises(JobFailed, match=r"no exit status recorded \(killed"):
        runner.run(shell_job(tmp_path, "slow", "echo x > x.txt", ["x.txt"]))
