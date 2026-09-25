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

If the driver is interrupted (Ctrl-C) or fails while waiting, every job it submitted
is cancelled (``scancel --name``: each submission has a unique job name), so no
orphaned job keeps writing into a directory the next run will use. A driver killed
outright (SIGKILL, a dropped ssh session) cannot clean up. Run long pipelines
inside tmux, or submit the driver itself as a SLURM job.

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
import time
import uuid
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
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
        with ThreadPoolExecutor(max_workers=max(1, len(arrays))) as pool:
            grouped = list(pool.map(self._submit_array, arrays))
        outcome_by_name = {
            outcome.job.name: outcome for outcomes in grouped for outcome in outcomes
        }
        return collect([outcome_by_name[job.name] for job in jobs])

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
        try:
            submitted = subprocess.run(argv, capture_output=True, text=True)
        except OSError as error:
            return [Failure(job, f"could not run sbatch: {error}") for job in jobs]
        except BaseException:
            # Interrupted or failed while waiting: don't leave the jobs running.
            self._cancel(_job_name(jobs, batch))
            raise
        (batch / "sbatch.out").write_text(submitted.stdout + submitted.stderr)
        log.info("sbatch exited %d for %s", submitted.returncode, batch.name)
        if submitted.returncode != 0 and not any(
            _status_path(batch, i).exists() for i in range(len(jobs))
        ):
            # sbatch itself failed (rejected, or not installed): no task ran.
            reason = f"sbatch failed ({submitted.returncode}): {submitted.stderr.strip()}"
            return [Failure(job, reason) for job in jobs]
        self._wait_for_files(batch, jobs)
        return [
            finish(job, _read_status(_status_path(batch, index)), started)
            for index, job in enumerate(jobs)
        ]

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


def _job_name(jobs: list[Job], batch: Path) -> str:
    """Unique per submission (the batch id), so scancel --name hits exactly these jobs."""
    first = jobs[0].name if len(jobs) == 1 else f"{jobs[0].name}+{len(jobs) - 1}"
    return f"af-{first}-{batch.name.removeprefix('batch-')}"


def _write_task_script(batch: Path, index: int, job: Job) -> None:
    """The wrapper: run the command with its log, then record its exit status."""
    status = _status_path(batch, index)
    exports = "".join(f"export {name}={shlex.quote(value)}\n" for name, value in job.env.items())
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
