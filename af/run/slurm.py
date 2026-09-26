"""Run jobs under SLURM with ``sbatch --wait``: one job, or a job array for many.

How success is established, without trusting any single signal:

1. ``sbatch --wait`` blocks until the job (or every task of the array) has ended.
2. Each task runs a small wrapper script that runs the command, then writes the
   command's own exit status to a status file (written to a temporary name, then
   renamed). A task killed by SLURM (time limit, memory, preemption) never writes
   one, and a missing status file is a failure.
3. The declared artefacts are checked (exists, non-empty, parses, hash).

sbatch's own exit code is logged but not relied on: for an array it is the highest
task exit code, which cannot say *which* task failed.

If the driver is interrupted (Ctrl-C), terminated (SIGTERM, e.g. by a job
scheduler or ``kill``), loses its terminal (SIGHUP, a dropped ssh session) or fails
while waiting, every job it has in flight is cancelled (``scancel --name``: each
submission has a unique job name), so no orphaned job keeps writing into a directory
the next run will use. The waits run in worker threads but signals arrive in the
main thread, which is why cancelling works from a shared registry of in-flight
submissions. Only SIGKILL cannot be handled; run long pipelines in tmux anyway.

Arrays longer than ``max_array_size`` (the cluster's MaxArraySize, often 1001) are
split into several arrays, submitted together.

Requirements on the cluster: ``work_dir``, each job's ``cwd``, log and outputs, and
the Python interpreter must be on a filesystem the compute nodes share. Jobs inherit
the submitting environment (sbatch's default ``--export=ALL``).

No host, partition, account or path is built in. They come from the runner's fields,
which come from config. ``work_dir`` holds the generated scripts and status files and
must be on a filesystem the compute nodes can see.
"""

from __future__ import annotations

import logging
import shlex
import subprocess
import threading
import time
import uuid
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from af.run.job import (
    Failure,
    Job,
    JobResult,
    Resources,
    check_unique_names,
    collect,
    finish,
    now,
    signals_as_exceptions,
)

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class SlurmRunner:
    """Submits jobs with ``sbatch --wait``.

    ``output_wait_seconds``: on some shared filesystems files written on a compute node
    appear on the submit host a little later. If a status file or an artefact is not
    visible yet, poll for up to this long before declaring failure. 0 = no waiting.
    """

    work_dir: Path
    partition: str | None = None
    account: str | None = None
    extra_args: tuple[str, ...] = ()
    max_parallel_tasks: int | None = None
    output_wait_seconds: float = 0.0
    max_array_size: int | None = None
    sbatch: str = "sbatch"
    scancel: str = "scancel"
    sacct: str = "sacct"
    _active: set[str] = field(default_factory=set, init=False, repr=False, compare=False)
    _lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if self.max_array_size is not None and self.max_array_size < 1:
            raise ValueError("max_array_size must be at least 1")

    def run(self, job: Job) -> JobResult:
        return self.run_many([job])[0]

    def run_many(self, jobs: Sequence[Job]) -> list[JobResult]:
        """Submit one array per distinct resource request, wait for all, then check."""
        check_unique_names(jobs)
        by_resources: dict[Resources, list[Job]] = {}
        for job in jobs:
            by_resources.setdefault(job.resources, []).append(job)
        arrays = [chunk for group in by_resources.values() for chunk in self._chunks(group)]
        with signals_as_exceptions(), ThreadPoolExecutor(max_workers=max(1, len(arrays))) as pool:
            try:
                grouped = list(pool.map(self._submit_array, arrays))
            except BaseException:
                # Ctrl-C, SIGTERM/SIGHUP or an error arrives here, in the main thread,
                # while workers still block in sbatch --wait: cancel what they wait on.
                self.cancel_active()
                raise
        outcome_by_name = {
            outcome.job.name: outcome for outcomes in grouped for outcome in outcomes
        }
        return collect([outcome_by_name[job.name] for job in jobs])

    def cancel_active(self) -> None:
        """scancel every submission still in flight. Their sbatch --wait then returns."""
        with self._lock:
            names = sorted(self._active)
        for name in names:
            self._cancel(name)

    def _chunks(self, jobs: list[Job]) -> list[list[Job]]:
        size = self.max_array_size or len(jobs)
        return [jobs[start : start + size] for start in range(0, len(jobs), size)]

    def _submit_array(self, jobs: list[Job]) -> list[JobResult | Failure]:
        batch = self.work_dir / f"batch-{uuid.uuid4().hex[:12]}"
        batch.mkdir(parents=True)
        for index, job in enumerate(jobs):
            job.log.parent.mkdir(parents=True, exist_ok=True)
            _write_task_script(batch, index, job)
        array_script = batch / "array.sh"
        array_script.write_text(
            '#!/bin/sh\nexec /bin/sh "' + str(batch) + '/task-${SLURM_ARRAY_TASK_ID}.sh"\n'
        )
        argv = self._sbatch_argv(jobs, batch, array_script)
        log.info("submitting %d task(s): %s", len(jobs), shlex.join(argv))
        started = now()
        name = _job_name(jobs, batch)
        with self._lock:
            self._active.add(name)
        try:
            submitted = subprocess.run(argv, capture_output=True, text=True)
        except OSError as error:
            return [Failure(job, f"could not run sbatch: {error}") for job in jobs]
        except BaseException:
            # Failed while waiting, in this thread: don't leave the jobs running.
            self._cancel(name)
            raise
        finally:
            with self._lock:
                self._active.discard(name)
        (batch / "sbatch.out").write_text(submitted.stdout + submitted.stderr)
        log.info("sbatch exited %d for %s", submitted.returncode, batch.name)
        job_id = _job_id(submitted.stdout)
        if submitted.returncode != 0 and job_id is None:
            # No job id printed: sbatch itself failed (rejected, or not installed), no task ran.
            # (With a job id, a nonzero exit means a task failed or was killed, e.g. at its
            # time limit, which SLURM reports this way without any status file.)
            reason = f"sbatch failed ({submitted.returncode}): {submitted.stderr.strip()}"
            return [Failure(job, reason) for job in jobs]
        self._wait_for_files(batch, jobs)
        outcomes = [
            finish(job, _read_status(_status_path(batch, index)), started)
            for index, job in enumerate(jobs)
        ]
        return self._explain_kills(outcomes, job_id)

    def _explain_kills(
        self, outcomes: list[JobResult | Failure], job_id: str | None
    ) -> list[JobResult | Failure]:
        """Add SLURM's own state (TIMEOUT, OUT_OF_MEMORY, …) to tasks that left no status.

        Best effort: without sacct, or when accounting doesn't answer, the reason stays
        "no exit status recorded", which is still a failure.
        """
        killed = [
            i
            for i, o in enumerate(outcomes)
            if isinstance(o, Failure) and o.reason.startswith("no exit status")
        ]
        if not killed or job_id is None:
            return outcomes
        states = self._task_states(job_id)
        for index in killed:
            state = states.get(index)
            if state:
                failure = outcomes[index]
                assert isinstance(failure, Failure)
                outcomes[index] = Failure(failure.job, f"{failure.reason} [SLURM state {state}]")
        return outcomes

    def _task_states(self, job_id: str) -> dict[int, str]:
        """Array task index -> SLURM state, from sacct; empty if it can't be read."""
        try:
            result = subprocess.run(
                [self.sacct, "-X", "-n", "-P", "-j", job_id, "--format=JobID,State"],
                capture_output=True,
                text=True,
                timeout=60,
            )
        except (OSError, subprocess.TimeoutExpired):
            return {}
        states: dict[int, str] = {}
        for line in result.stdout.splitlines():
            ident, _, state = line.partition("|")
            head, _, index = ident.partition("_")
            if head == job_id and index.isdigit():
                states[int(index)] = state.split()[0] if state else ""
        return states

    def _sbatch_argv(self, jobs: list[Job], batch: Path, script: Path) -> list[str]:
        resources = jobs[0].resources
        array = f"0-{len(jobs) - 1}"
        if self.max_parallel_tasks:
            array += f"%{self.max_parallel_tasks}"
        argv = [
            self.sbatch,
            "--wait",
            "--parsable",
            f"--array={array}",
            f"--job-name={_job_name(jobs, batch)}",
            f"--output={batch}/slurm-%A_%a.out",
            f"--cpus-per-task={resources.threads}",
        ]
        if resources.memory_gb is not None:
            argv.append(f"--mem={int(resources.memory_gb * 1024)}M")
        if resources.time_limit_minutes is not None:
            argv.append(f"--time={resources.time_limit_minutes}")
        if self.partition:
            argv.append(f"--partition={self.partition}")
        if self.account:
            argv.append(f"--account={self.account}")
        return [*argv, *self.extra_args, str(script)]

    def _cancel(self, job_name: str) -> None:
        log.warning("cancelling SLURM jobs named %s", job_name)
        try:
            subprocess.run([self.scancel, f"--name={job_name}"], capture_output=True, timeout=60)
        except (OSError, subprocess.TimeoutExpired) as error:
            log.error("could not cancel %s: %s; cancel it by hand", job_name, error)

    def _wait_for_files(self, batch: Path, jobs: list[Job]) -> None:
        """Give a lagging shared filesystem up to output_wait_seconds to show outputs."""
        deadline = time.monotonic() + self.output_wait_seconds
        expected = [_status_path(batch, i) for i in range(len(jobs))]
        expected += [artefact.path for job in jobs for artefact in job.outputs]
        while time.monotonic() < deadline and not all(path.exists() for path in expected):
            time.sleep(1)


def _job_id(stdout: str) -> str | None:
    """The job id sbatch --parsable prints on submission ("123" or "123;cluster")."""
    for line in stdout.splitlines():
        head = line.strip().split(";")[0]
        if head.isdigit():
            return head
    return None


def _job_name(jobs: list[Job], batch: Path) -> str:
    """Unique per submission (the batch id), so scancel --name hits exactly these jobs."""
    first = jobs[0].name if len(jobs) == 1 else f"{jobs[0].name}+{len(jobs) - 1}"
    return f"af-{first}-{batch.name.removeprefix('batch-')}"


def _write_task_script(batch: Path, index: int, job: Job) -> None:
    """The wrapper: run the command with its log, then record its exit status."""
    status = _status_path(batch, index)
    exports = "".join(
        f"export {name}={shlex.quote(value)}\n" for name, value in job.environment().items()
    )
    script = (
        "#!/bin/sh\n"
        f"# af job: {job.name}\n"
        f"cd {shlex.quote(str(job.cwd))} || exit 1\n"
        f"{exports}"
        f"{shlex.join(job.argv())} > {shlex.quote(str(job.log))} 2>&1\n"
        "rc=$?\n"
        f'printf "%s\\n" "$rc" > {shlex.quote(str(status))}.tmp'
        f" && mv {shlex.quote(str(status))}.tmp {shlex.quote(str(status))}\n"
        'exit "$rc"\n'
    )
    (batch / f"task-{index}.sh").write_text(script)


def _status_path(batch: Path, index: int) -> Path:
    return batch / f"task-{index}.status"


def _read_status(path: Path) -> int | None:
    try:
        return int(path.read_text().strip())
    except (OSError, ValueError):
        return None
