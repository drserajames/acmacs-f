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

The first failure stops the run, and the exception propagates. Steps that already
finished keep their records, so the next run starts from the failed step. A step
whose upstream re-ran but produced identical output is correctly skipped, because
its input hashes did not change.

The action does its work in-process or through ``context.runner`` (local or SLURM),
which is how the same pipeline runs on a laptop and on a cluster.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from af.pipeline.record import (
    check_up_to_date,
    fingerprint,
    forget,
    save_record,
)
from af.run.job import Runner, now
from af.util.artefacts import Artefact, check_artefacts, write_provenance

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class StepContext:
    step: Step
    runner: Runner


@dataclass(frozen=True)
class Step:
    name: str
    action: Callable[[StepContext], None]
    outputs: Sequence[Artefact]
    inputs: Sequence[Path] = ()
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
    def __init__(self, steps: Sequence[Step], *, state_dir: Path, runner: Runner) -> None:
        self.steps = {step.name: step for step in steps}
        self.state_dir = state_dir
        self.runner = runner
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
        outcomes = []
        for step in self.order(targets):
            outcome = self._run_step(step, forced=step.name in force)
            log.info("%s: %s (%s)", step.name, outcome.status, outcome.reason)
            outcomes.append(outcome)
        return outcomes

    def _run_step(self, step: Step, *, forced: bool) -> StepOutcome:
        current = fingerprint(step.name, step.inputs, step.parameters)
        if forced:
            reason = "forced"
        else:
            up_to_date, reason = check_up_to_date(self.state_dir, step.name, current, step.outputs)
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
                inputs=step.inputs,
                parameters=current.parameters,
                started=started,
                finished=finished,
            )
        save_record(self.state_dir, step.name, current, checked, started, finished)
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
        produced = {self._producer[path] for path in step.inputs if path in self._producer}
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
