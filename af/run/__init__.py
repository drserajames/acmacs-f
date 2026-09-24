"""Runners for external tools: local subprocess and SLURM, behind one call site.

Pick the runner from config, never by probing the machine. Then call
``runner.run(job)`` or ``runner.run_many(jobs)``. Neither returns until every job has
finished, exited 0 and produced its declared artefacts; otherwise they raise
:class:`JobFailed`.
"""

from af.run.job import Job, JobFailed, JobResult, Resources, Runner
from af.run.local import LocalRunner
from af.run.slurm import SlurmRunner

__all__ = ["Job", "JobFailed", "JobResult", "LocalRunner", "Resources", "Runner", "SlurmRunner"]
