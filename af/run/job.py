"""What a runner runs (:class:`Job`), what it returns, and how failure is reported.

A job is one external command with its working directory, a log file and the
artefacts it must produce. Runners differ only in *where* the command runs; the
contract is the same for all of them:

- ``run`` returns only after the command has finished, its exit status was 0 and every
  declared artefact checks out. Otherwise it raises :class:`JobFailed`.
- ``run_many`` runs independent jobs, waits for **all** of them, and raises one
  :class:`JobFailed` listing every failure. It never returns early while work continues.
"""

from __future__ import annotations

import datetime
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from af.util.artefacts import Artefact, ArtefactError, CheckedArtefact, check_artefacts

LOG_TAIL_LINES = 20


@dataclass(frozen=True)
class Resources:
    """What a job asks the scheduler for. The local runner only uses ``threads``."""

    threads: int = 1
    memory_gb: float | None = None
    time_limit_minutes: int | None = None


@dataclass(frozen=True)
class Job:
    """One external command.

    ``env`` holds variables to *add* to the inherited environment, set explicitly by
    the step; nothing is looked up implicitly.
    """

    name: str
    command: Sequence[str | Path]
    cwd: Path
    log: Path
    outputs: Sequence[Artefact]
    resources: Resources = field(default_factory=Resources)
    env: Mapping[str, str] = field(default_factory=dict)

    def argv(self) -> list[str]:
        return [str(part) for part in self.command]

    @classmethod
    def python_module(
        cls,
        name: str,
        module: str,
        args: Sequence[str | Path],
        *,
        cwd: Path,
        log: Path,
        outputs: Sequence[Artefact],
        resources: Resources | None = None,
        env: Mapping[str, str] | None = None,
    ) -> Job:
        """A job that runs ``python -m module args`` with *this* interpreter.

        This is how Python work (a chain step, a relax batch) runs on compute nodes:
        the same interpreter, so the same af and dependencies, provided it sits on a
        filesystem the nodes share. Pass thread counts in ``args`` as well as in
        ``resources``: SLURM allocates the CPUs, but the program must be told to use them.
        """
        return cls(
            name=name,
            command=[sys.executable, "-m", module, *args],
            cwd=cwd,
            log=log,
            outputs=outputs,
            resources=resources or Resources(),
            env=dict(env or {}),
        )


@dataclass(frozen=True)
class JobResult:
    job: Job
    returncode: int
    started: datetime.datetime
    finished: datetime.datetime
    artefacts: list[CheckedArtefact]


@dataclass(frozen=True)
class Failure:
    job: Job
    reason: str


class JobFailed(RuntimeError):
    """One or more jobs failed: non-zero exit, killed, or bad artefacts."""

    def __init__(self, failures: list[Failure]) -> None:
        self.failures = failures
        parts = [f"{len(failures)} job(s) failed:"]
        for failure in failures:
            parts.append(f"  - {failure.job.name}: {failure.reason} (log: {failure.job.log})")
            tail = log_tail(failure.job.log)
            if tail:
                parts.append("      " + "\n      ".join(tail))
        super().__init__("\n".join(parts))


class Runner(Protocol):
    """Where jobs run. Both implementations honour the contract in the module docstring."""

    def run(self, job: Job) -> JobResult: ...

    def run_many(self, jobs: Sequence[Job]) -> list[JobResult]: ...


def finish(job: Job, returncode: int | None, started: datetime.datetime) -> JobResult | Failure:
    """Turn a finished command into a result, checking its exit status and artefacts.

    ``returncode`` is None when no exit status was recorded (the job was killed, e.g. by
    a time limit), which is a failure, never a pass.
    """
    finished = now()
    if returncode is None:
        return Failure(job, "no exit status recorded (killed, or hit its time limit?)")
    if returncode != 0:
        return Failure(job, f"exit status {returncode}")
    try:
        artefacts = check_artefacts(job.name, job.outputs)
    except ArtefactError as error:
        reasons = "; ".join(f"{path}: {reason}" for path, reason in error.failures)
        return Failure(job, f"exit status 0 but bad output: {reasons}")
    return JobResult(job, returncode, started, finished, artefacts)


def collect(outcomes: Sequence[JobResult | Failure]) -> list[JobResult]:
    """Raise one JobFailed for every failure, or return the results in job order."""
    failures = [outcome for outcome in outcomes if isinstance(outcome, Failure)]
    if failures:
        raise JobFailed(failures)
    return [outcome for outcome in outcomes if isinstance(outcome, JobResult)]


def check_unique_names(jobs: Sequence[Job]) -> None:
    names = [job.name for job in jobs]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"job names must be unique; repeated: {duplicates}")


def log_tail(path: Path, lines: int = LOG_TAIL_LINES) -> list[str]:
    try:
        return path.read_text(errors="replace").splitlines()[-lines:]
    except OSError:
        return []


def now() -> datetime.datetime:
    return datetime.datetime.now(datetime.UTC)
