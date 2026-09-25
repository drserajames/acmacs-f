"""Smoke test for af.run on a real cluster: does SLURM behave the way af assumes?

Run it from a login node, with a work directory on a filesystem the compute nodes
share::

    python -m af.run.smoke --work-dir /shared/scratch/af-smoke \\
        [--partition P] [--account A] [--extra-arg=--qos=Q] [--max-array-size N] \\
        [--with-timeout]

Each check submits real jobs and compares what af reports with what should happen:

1. a shell job that succeeds and writes its output;
2. a Python job (``python -m af.run.smoke --child``) on a compute node, which records
   the node's hostname, af version and SLURM variables, proving the environment
   and interpreter reach the nodes;
3. a job that exits 3, which must be reported as ``exit status 3``;
4. a job that exits 0 but writes nothing, which must fail on its missing output;
5. an array of 6 tasks, run at most 3 at a time and split into arrays of at most 4
   (or ``--max-array-size``), which must all succeed with their own outputs;
6. with ``--with-timeout``, a job killed by its 1-minute time limit, which must fail
   with "no exit status recorded" (takes about 2 minutes).

A table is printed, and ``smoke-report.json`` is written in the work directory. Exit
status 0 only if every check behaved as expected. ``--local`` runs the same checks
with the local runner, to test this script without a cluster.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import af
from af.run.job import Job, JobFailed, Resources, Runner
from af.run.local import LocalRunner
from af.run.slurm import SlurmRunner
from af.util.artefacts import Artefact


@dataclass
class Check:
    name: str
    expected: str
    passed: bool = False
    detail: str = ""


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.child:
        write_probe(Path(args.child))
        return 0
    work = Path(args.work_dir).resolve()
    work.mkdir(parents=True, exist_ok=True)
    runner = make_runner(args, work)
    checks = run_checks(runner, work, with_timeout=args.with_timeout and not args.local)
    report(checks, work, args)
    return 0 if all(check.passed for check in checks) else 1


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m af.run.smoke", description=__doc__)
    parser.add_argument("--work-dir", help="on a filesystem the compute nodes share")
    parser.add_argument("--partition")
    parser.add_argument("--account")
    parser.add_argument("--extra-arg", action="append", default=[], help="passed to sbatch")
    parser.add_argument("--max-array-size", type=int, default=4)
    parser.add_argument("--output-wait-seconds", type=float, default=30.0)
    parser.add_argument("--with-timeout", action="store_true", help="also test a killed job")
    parser.add_argument("--local", action="store_true", help="local runner (no cluster)")
    parser.add_argument("--sbatch", default="sbatch", help=argparse.SUPPRESS)
    parser.add_argument("--child", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if not args.child and not args.work_dir:
        parser.error("--work-dir is required")
    return args


def make_runner(args: argparse.Namespace, work: Path) -> Runner:
    if args.local:
        return LocalRunner(max_parallel=3)
    return SlurmRunner(
        work_dir=work / "slurm",
        partition=args.partition,
        account=args.account,
        extra_args=tuple(args.extra_arg),
        max_parallel_tasks=3,
        output_wait_seconds=args.output_wait_seconds,
        max_array_size=args.max_array_size,
        sbatch=args.sbatch,
    )


def run_checks(runner: Runner, work: Path, *, with_timeout: bool) -> list[Check]:
    def shell(name: str, script: str, outputs: list[str], **kw: object) -> Job:
        return Job(
            name,
            ["/bin/sh", "-c", script],
            work,
            work / "logs" / f"{name}.log",
            [Artefact(work / output) for output in outputs],
            **kw,  # type: ignore[arg-type]
        )

    probe = work / "probe.json"
    checks: list[tuple[Check, Callable[[], str]]] = [
        (
            Check("shell job", "succeeds"),
            lambda: ok(runner.run(shell("ok", "hostname > ok.txt", ["ok.txt"]))),
        ),
        (
            Check("python job on a node", "succeeds, probe written"),
            lambda: python_probe(runner, work, probe),
        ),
        (
            Check("failing job", "JobFailed: exit status 3"),
            lambda: expect_failure(
                runner, shell("fails", "exit 3", ["never.txt"]), "exit status 3"
            ),
        ),
        (
            Check("missing output", "JobFailed: bad output"),
            lambda: expect_failure(runner, shell("lazy", "true", ["absent.txt"]), "bad output"),
        ),
        (
            Check("array of 6", "all succeed, 6 distinct outputs"),
            lambda: array(runner, shell),
        ),
    ]
    if with_timeout:
        killed = shell(
            "killed",
            "sleep 150; echo late > late.txt",
            ["late.txt"],
            resources=Resources(time_limit_minutes=1),
        )
        checks.append(
            (
                Check("time limit kill", "JobFailed: no exit status recorded"),
                lambda: expect_failure(runner, killed, "no exit status recorded"),
            )
        )
    results = []
    for check, action in checks:
        print(f"running: {check.name} ...", flush=True)
        try:
            check.detail = action()
            check.passed = True
        except Exception as error:  # a check failing is a result, not a crash
            check.detail = f"{type(error).__name__}: {error}".splitlines()[0][:300]
        results.append(check)
    return results


def ok(result: object) -> str:
    return "ok"


def python_probe(runner: Runner, work: Path, probe: Path) -> str:
    probe.unlink(missing_ok=True)
    runner.run(
        Job.python_module(
            "probe",
            "af.run.smoke",
            ["--child", str(probe)],
            cwd=work,
            log=work / "logs" / "probe.log",
            outputs=[Artefact(probe, parse=lambda p: json.loads(p.read_text()))],
        )
    )
    data = json.loads(probe.read_text())
    return f"node {data['host']}, af {data['af_version']}, job {data['slurm_job_id']}"


def expect_failure(runner: Runner, job: Job, reason: str) -> str:
    try:
        runner.run(job)
    except JobFailed as error:
        got = error.failures[0].reason
        if reason not in got:
            raise AssertionError(f"failed, but with {got!r}") from error
        return got
    raise AssertionError("reported success")


def array(runner: Runner, shell: Callable[..., Job]) -> str:
    jobs = [shell(f"task{i}", f"echo {i} > task{i}.txt", [f"task{i}.txt"]) for i in range(6)]
    results = runner.run_many(jobs)
    names = [result.job.name for result in results]
    if names != [job.name for job in jobs]:
        raise AssertionError(f"results out of order: {names}")
    return f"{len(results)} ok"


def write_probe(path: Path) -> None:
    """Run on a compute node: record where and with what we ran."""
    path.write_text(
        json.dumps(
            {
                "host": socket.gethostname(),
                "af_version": af.__version__,
                "python": sys.executable,
                "platform": platform.platform(),
                "cwd": str(Path.cwd()),
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
                "cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
            },
            indent=2,
        )
        + "\n"
    )


def report(checks: list[Check], work: Path, args: argparse.Namespace) -> None:
    version = "local runner" if args.local else sbatch_version(args.sbatch)
    print(
        f"\naf.run smoke test: af {af.__version__}, {version}, submit host {socket.gethostname()}"
    )
    for check in checks:
        mark = "PASS" if check.passed else "FAIL"
        print(f"  {mark}  {check.name:22s} expected: {check.expected}\n        {check.detail}")
    (work / "smoke-report.json").write_text(
        json.dumps(
            {
                "af_version": af.__version__,
                "sbatch": version,
                "submit_host": socket.gethostname(),
                "checks": [vars(check) for check in checks],
            },
            indent=2,
        )
        + "\n"
    )


def sbatch_version(sbatch: str) -> str:
    try:
        return subprocess.run([sbatch, "--version"], capture_output=True, text=True).stdout.strip()
    except OSError as error:
        return f"sbatch unavailable: {error}"


if __name__ == "__main__":
    sys.exit(main())
