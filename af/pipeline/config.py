"""Build and run a pipeline from a TOML config file.

The config chooses the steps, their parameters and the runner. The step code comes
from a *registry*: a mapping from step name to a factory that takes that step's
parameter table and returns a :class:`~af.pipeline.driver.Step`. Each workstream
supplies its own factories.

Example (all paths relative to the config file)::

    [pipeline]
    state_dir = "state"
    root = "."                # record paths relative to the config's directory
    steps = ["export", "build-tree"]

    [runner]
    kind = "slurm"            # or "local"

    [runner.slurm]
    work_dir = "slurm"
    partition = "example"

    [parameters.build-tree]
    threads = 8

A step listed in ``steps`` that the registry doesn't know, or a ``[parameters.X]``
table for a step not listed, is an error (design rule 1).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from af.pipeline.driver import Pipeline, Step, StepOutcome
from af.run import LocalRunner, Runner, SlurmRunner
from af.util.config import ConfigError, load_config

StepFactory = Callable[[Mapping[str, Any]], Step]


@dataclass(frozen=True)
class LocalSettings:
    max_parallel: int = 1


@dataclass(frozen=True)
class SlurmSettings:
    work_dir: Path
    partition: str | None = None
    account: str | None = None
    extra_args: list[str] = field(default_factory=list)
    max_parallel_tasks: int | None = None
    output_wait_seconds: float = 0.0
    max_array_size: int | None = None  # the cluster's MaxArraySize (scontrol show config)


@dataclass(frozen=True)
class RunnerSettings:
    kind: str
    local: LocalSettings = field(default_factory=LocalSettings)
    slurm: SlurmSettings | None = None


@dataclass(frozen=True)
class PipelineSettings:
    state_dir: Path
    steps: list[str]
    # Unnamed step inputs/outputs under `root` are recorded relative to it, so the tree
    # can move or sync to another machine without re-running. Omit it only for pipelines
    # that never move (their records then hold absolute paths).
    root: Path | None = None
    # How many independent steps may run at once (1 = one after another).
    max_parallel: int = 1


@dataclass(frozen=True)
class PipelineConfig:
    pipeline: PipelineSettings
    runner: RunnerSettings
    parameters: dict[str, dict[str, Any]] = field(default_factory=dict)


def make_runner(settings: RunnerSettings, source: Path | str = "<config>") -> Runner:
    if settings.kind == "local":
        return LocalRunner(max_parallel=settings.local.max_parallel)
    if settings.kind == "slurm":
        if settings.slurm is None:
            raise ConfigError(source, ["runner.slurm: required when runner.kind = 'slurm'"])
        slurm = settings.slurm
        return SlurmRunner(
            work_dir=slurm.work_dir,
            partition=slurm.partition,
            account=slurm.account,
            extra_args=tuple(slurm.extra_args),
            max_parallel_tasks=slurm.max_parallel_tasks,
            output_wait_seconds=slurm.output_wait_seconds,
            max_array_size=slurm.max_array_size,
        )
    raise ConfigError(source, [f"runner.kind: expected 'local' or 'slurm', got {settings.kind!r}"])


def build_pipeline(
    config_path: Path, registry: Mapping[str, StepFactory]
) -> tuple[Pipeline, PipelineConfig]:
    config = load_config(config_path, PipelineConfig)
    problems = [
        f"pipeline.steps: unknown step {name!r}"
        for name in config.pipeline.steps
        if name not in registry
    ]
    problems += [
        f"parameters.{name}: no such step in pipeline.steps"
        for name in config.parameters
        if name not in config.pipeline.steps
    ]
    if problems:
        raise ConfigError(config_path, problems)
    steps = [registry[name](config.parameters.get(name, {})) for name in config.pipeline.steps]
    for name, step in zip(config.pipeline.steps, steps, strict=True):
        if step.name != name:
            raise ConfigError(
                config_path, [f"factory for {name!r} built a step named {step.name!r}"]
            )
    runner = make_runner(config.runner, config_path)
    pipeline = Pipeline(
        steps,
        state_dir=config.pipeline.state_dir,
        runner=runner,
        root=config.pipeline.root,
        max_parallel=config.pipeline.max_parallel,
    )
    return pipeline, config


def run_from_config(
    config_path: Path,
    registry: Mapping[str, StepFactory],
    *,
    targets: Sequence[str] | None = None,
    force: Sequence[str] = (),
) -> list[StepOutcome]:
    pipeline, _ = build_pipeline(config_path, registry)
    return pipeline.run(targets, force=force)
