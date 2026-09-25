"""Run named steps in dependency order, skipping those whose inputs are unchanged.

A :class:`Step` is a Python function plus what it reads (``inputs``), what it must
produce (``outputs``), its ``parameters`` and any explicit ordering (``after``).
Dependencies are the ``after`` names plus, automatically, every step that produces
one of this step's inputs.

For each step, in order, :meth:`Pipeline.run`:

1. hashes the inputs and parameters (a missing input is fatal),
2. skips the step if its record says nothing changed (:mod:`af.pipeline.record`),
3. otherwise drops the old record, runs the action, checks every output artefact,
   writes a provenance file beside each output, and saves the new record.

Independent steps run in parallel, up to ``max_parallel`` at a time (1, the default,
is serial): a step starts as soon as every step it depends on has finished. So the
three subtypes' trees build side by side, each through the runner (local or SLURM).

The first failure stops new steps from starting. Steps already running are left to
finish, so their records are saved, and then the first exception propagates. Steps that
finished keep their records, so the next run starts from the failed step. A Ctrl-C,
SIGTERM or SIGHUP cancels the runner's jobs in flight before the run stops. A step
whose upstream re-ran but produced identical output is correctly skipped, because
its input hashes did not change.

The action does its work in-process or through ``context.runner`` (local or SLURM),
which is how the same pipeline runs on a laptop and on a cluster.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Collection, Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from af.pipeline.record import (
    Inputs,
    check_up_to_date,
    fingerprint,
    forget,
    input_roles,
    output_roles,
    save_record,
)
from af.run.job import Runner, now, signals_as_exceptions
from af.util.artefacts import Artefact, check_artefacts, write_provenance

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class StepContext:
    step: Step
    runner: Runner


@dataclass(frozen=True)
class Step:
    """One step. ``inputs`` is either a list of paths or a mapping of role name to path.

    Named inputs (``{"table": path, "previous": path}``) make the step's record
    independent of where the files live. A list of paths gets roles relative to the
    pipeline's ``root``.
    """

    name: str
    action: Callable[[StepContext], None]
    outputs: Sequence[Artefact]
    inputs: Inputs = ()
    parameters: Mapping[str, Any] = field(default_factory=dict)
    after: Sequence[str] = ()


@dataclass(frozen=True)
class StepOutcome:
    name: str
    status: Literal["ran", "skipped"]
    reason: str


class PipelineError(ValueError):
    """The pipeline definition is inconsistent (unknown names, cycles, duplicates)."""


class Pipeline:
    def __init__(
        self,
        steps: Sequence[Step],
        *,
        state_dir: Path,
        runner: Runner,
        root: Path | None = None,
        max_parallel: int = 1,
    ) -> None:
        """``root``: unnamed inputs and outputs under it are recorded relative to it, so
        the whole tree can move (or sync to another machine) without re-running steps.
        ``max_parallel``: how many independent steps may run at once.
        """
        if max_parallel < 1:
            raise PipelineError("max_parallel must be at least 1")
        self.max_parallel = max_parallel
        self.steps = {step.name: step for step in steps}
        self.state_dir = state_dir
        self.runner = runner
        self.root = root
        if len(self.steps) != len(steps):
            raise PipelineError(f"duplicate step names in {[step.name for step in steps]}")
        self._producer = self._output_producers(steps)
        self._dependencies = {step.name: self._depends_on(step) for step in steps}
        self._check_acyclic()

    def order(self, targets: Sequence[str] | None = None) -> list[Step]:
        """The steps needed for ``targets`` (default: all), dependencies first.

        Ties are broken by the order the steps were given in, so the order is stable.
        """
        wanted = list(self.steps) if targets is None else list(targets)
        unknown = [name for name in wanted if name not in self.steps]
        if unknown:
            raise PipelineError(f"unknown target step(s): {unknown}")
        needed: set[str] = set()
        stack = list(wanted)
        while stack:
            name = stack.pop()
            if name not in needed:
                needed.add(name)
                stack.extend(self._dependencies[name])
        ordered: list[Step] = []
        done: set[str] = set()
        while len(ordered) < len(needed):
            for name in self.steps:
                if name in needed and name not in done and self._dependencies[name] <= done:
                    ordered.append(self.steps[name])
                    done.add(name)
                    break
        return ordered

    def run(
        self, targets: Sequence[str] | None = None, *, force: Collection[str] = ()
    ) -> list[StepOutcome]:
        """Run what is out of date for ``targets``. ``force`` names steps to re-run anyway."""
        unknown = sorted(set(force) - set(self.steps))
        if unknown:
            raise PipelineError(f"unknown step(s) to force: {unknown}")
        planned = self.order(targets)
        outcomes: dict[str, StepOutcome] = {}
        with signals_as_exceptions(), ThreadPoolExecutor(max_workers=self.max_parallel) as pool:
            try:
                failure = self._schedule(pool, planned, outcomes, force)
            except BaseException:
                # Ctrl-C / SIGTERM / SIGHUP arrive here, in the main thread, while steps
                # wait on jobs in worker threads: stop those jobs, then stop.
                self.runner.cancel_active()
                raise
        if failure is not None:
            raise failure
        return [outcomes[step.name] for step in planned]

    def _schedule(
        self,
        pool: ThreadPoolExecutor,
        planned: list[Step],
        outcomes: dict[str, StepOutcome],
        force: Collection[str],
    ) -> BaseException | None:
        """Run ``planned`` as their dependencies allow; return the first failure, if any."""
        pending = list(planned)
        running: dict[Future[StepOutcome], Step] = {}
        failure: BaseException | None = None
        while pending or running:
            if failure is None:
                for step in [s for s in pending if self._dependencies[s.name] <= outcomes.keys()]:
                    if len(running) >= self.max_parallel:
                        break
                    pending.remove(step)
                    running[pool.submit(self._run_step, step, forced=step.name in force)] = step
            if not running:
                break  # a failure stopped new steps and nothing is left running
            done, _ = wait(running, return_when=FIRST_COMPLETED)
            for future in done:
                step = running.pop(future)
                try:
                    outcome = future.result()
                except Exception as error:
                    log.error("%s: failed: %s", step.name, error)
                    failure = failure or error
                    continue
                log.info("%s: %s (%s)", step.name, outcome.status, outcome.reason)
                outcomes[step.name] = outcome
        return failure

    def _run_step(self, step: Step, *, forced: bool) -> StepOutcome:
        try:
            inputs = input_roles(step.name, step.inputs, self.root)
            outputs = output_roles(step.name, step.outputs, self.root)
        except ValueError as error:
            raise PipelineError(str(error)) from error
        current = fingerprint(step.name, inputs, step.parameters)
        if forced:
            reason = "forced"
        else:
            up_to_date, reason = check_up_to_date(self.state_dir, step.name, current, outputs)
            if up_to_date:
                return StepOutcome(step.name, "skipped", reason)
        forget(self.state_dir, step.name)
        started = now()
        step.action(StepContext(step=step, runner=self.runner))
        checked = check_artefacts(step.name, step.outputs)
        finished = now()
        for output in checked:
            write_provenance(
                step.name,
                output,
                inputs=list(inputs.values()),
                parameters=current.parameters,
                started=started,
                finished=finished,
            )
        by_role = dict(zip(outputs, checked, strict=True))
        save_record(self.state_dir, step.name, current, by_role, started, finished)
        return StepOutcome(step.name, "ran", reason)

    @staticmethod
    def _output_producers(steps: Sequence[Step]) -> dict[Path, str]:
        producer: dict[Path, str] = {}
        for step in steps:
            for artefact in step.outputs:
                if artefact.path in producer:
                    raise PipelineError(
                        f"{artefact.path} is an output of both "
                        f"{producer[artefact.path]!r} and {step.name!r}"
                    )
                producer[artefact.path] = step.name
        return producer

    def _depends_on(self, step: Step) -> set[str]:
        unknown = [name for name in step.after if name not in self.steps]
        if unknown:
            raise PipelineError(f"step {step.name!r}: 'after' names unknown step(s) {unknown}")
        paths = step.inputs.values() if isinstance(step.inputs, Mapping) else step.inputs
        produced = {self._producer[path] for path in paths if path in self._producer}
        dependencies = set(step.after) | produced
        if step.name in dependencies:
            raise PipelineError(f"step {step.name!r} depends on itself")
        return dependencies

    def _check_acyclic(self) -> None:
        visiting: set[str] = set()
        finished: set[str] = set()

        def visit(name: str, path: list[str]) -> None:
            if name in finished:
                return
            if name in visiting:
                cycle = [*path[path.index(name) :], name]
                raise PipelineError(f"dependency cycle: {' -> '.join(cycle)}")
            visiting.add(name)
            for dependency in sorted(self._dependencies[name]):
                visit(dependency, [*path, name])
            visiting.discard(name)
            finished.add(name)

        for name in self.steps:
            visit(name, [])
