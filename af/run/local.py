"""Run jobs as local subprocesses.

The exit status comes straight from the child process. Nothing is piped through
``tee`` or wrapped in ``time``, either of which can replace the tool's own status
with theirs. stdout and stderr both go to the job's log file.
"""

from __future__ import annotations

import logging
import os
import subprocess
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from af.run.job import Failure, Job, JobResult, check_unique_names, collect, finish, now

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class LocalRunner:
    """Runs jobs on this machine, at most ``max_parallel`` at a time in ``run_many``."""

    max_parallel: int = 1

    def __post_init__(self) -> None:
        if self.max_parallel < 1:
            raise ValueError("max_parallel must be at least 1")

    def run(self, job: Job) -> JobResult:
        return collect([self._run_one(job)])[0]

    def run_many(self, jobs: Sequence[Job]) -> list[JobResult]:
        check_unique_names(jobs)
        with ThreadPoolExecutor(max_workers=self.max_parallel) as pool:
            outcomes = list(pool.map(self._run_one, jobs))  # waits for every job
        return collect(outcomes)

    def _run_one(self, job: Job) -> JobResult | Failure:
        job.log.parent.mkdir(parents=True, exist_ok=True)
        env = {**os.environ, **job.env} if job.env else None
        log.info("%s: running %s", job.name, subprocess.list2cmdline(job.argv()))
        started = now()
        try:
            with job.log.open("w") as stream:
                completed = subprocess.run(
                    job.argv(), cwd=job.cwd, stdout=stream, stderr=subprocess.STDOUT, env=env
                )
        except OSError as error:  # e.g. the tool is not installed
            return Failure(job, f"could not start: {error}")
        outcome = finish(job, completed.returncode, started)
        log.info("%s: %s", job.name, "ok" if isinstance(outcome, JobResult) else outcome.reason)
        return outcome
