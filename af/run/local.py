"""Run jobs as local subprocesses.

The exit status comes straight from the child process. On Ctrl-C, SIGTERM or SIGHUP
every running child is terminated, so none outlives the driver. Nothing is piped through
``tee`` or wrapped in ``time``, either of which can replace the tool's own status
with theirs. stdout and stderr both go to the job's log file.
"""

from __future__ import annotations

import logging
import os
import subprocess
import threading
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from af.run.job import (
    Failure,
    Job,
    JobResult,
    check_unique_names,
    collect,
    finish,
    now,
    signals_as_exceptions,
)

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class LocalRunner:
    """Runs jobs on this machine, at most ``max_parallel`` at a time in ``run_many``."""

    max_parallel: int = 1
    _active: set[subprocess.Popen[bytes]] = field(
        default_factory=set, init=False, repr=False, compare=False
    )
    _lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if self.max_parallel < 1:
            raise ValueError("max_parallel must be at least 1")

    def run(self, job: Job) -> JobResult:
        return self.run_many([job])[0]

    def run_many(self, jobs: Sequence[Job]) -> list[JobResult]:
        check_unique_names(jobs)
        with signals_as_exceptions(), ThreadPoolExecutor(max_workers=self.max_parallel) as pool:
            try:
                outcomes = list(pool.map(self._run_one, jobs))  # waits for every job
            except BaseException:
                self.cancel_active()
                raise
        return collect(outcomes)

    def cancel_active(self) -> None:
        """Terminate every child still running."""
        with self._lock:
            children = list(self._active)
        for child in children:
            child.terminate()

    def _run_one(self, job: Job) -> JobResult | Failure:
        job.log.parent.mkdir(parents=True, exist_ok=True)
        env = {**os.environ, **job.environment()}
        log.info("%s: running %s", job.name, subprocess.list2cmdline(job.argv()))
        started = now()
        try:
            with job.log.open("w") as stream:
                child = subprocess.Popen(
                    job.argv(), cwd=job.cwd, stdout=stream, stderr=subprocess.STDOUT, env=env
                )
                with self._lock:
                    self._active.add(child)
                try:
                    returncode = child.wait()
                finally:
                    with self._lock:
                        self._active.discard(child)
        except OSError as error:  # e.g. the tool is not installed
            return Failure(job, f"could not start: {error}")
        outcome = finish(job, returncode, started)
        log.info("%s: %s", job.name, "ok" if isinstance(outcome, JobResult) else outcome.reason)
        return outcome
