"""af.pipeline: dependency order, incremental skipping, failure propagation, config."""

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from af.pipeline import (
    Pipeline,
    PipelineError,
    Step,
    StepContext,
    StepInputMissing,
    build_pipeline,
    run_from_config,
)
from af.run import Job, LocalRunner
from af.util.artefacts import Artefact, ArtefactError, read_provenance
from af.util.config import ConfigError


class Chain:
    """Three steps: raw.txt -> upper.txt -> count.txt, recording which actions ran."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.raw = root / "raw.txt"
        self.upper = root / "upper.txt"
        self.count = root / "count.txt"
        self.raw.write_text("abc\n")
        self.ran: list[str] = []

    def steps(self, repeat: int = 1) -> list[Step]:
        def upper(context: StepContext) -> None:
            self.ran.append("upper")
            times = int(context.step.parameters["repeat"])
            self.upper.write_text(self.raw.read_text().upper() * times)

        def count(context: StepContext) -> None:
            self.ran.append("count")
            self.count.write_text(str(len(self.upper.read_text())))

        # Declared out of order on purpose: dependencies come from inputs/outputs.
        return [
            Step("count", count, outputs=[Artefact(self.count)], inputs=[self.upper]),
            Step(
                "upper",
                upper,
                outputs=[Artefact(self.upper)],
                inputs=[self.raw],
                parameters={"repeat": repeat},
            ),
        ]

    def pipeline(self, repeat: int = 1) -> Pipeline:
        return Pipeline(self.steps(repeat), state_dir=self.root / "state", runner=LocalRunner())


def statuses(outcomes: list[Any]) -> list[tuple[str, str]]:
    return [(outcome.name, outcome.status) for outcome in outcomes]


def test_order_follows_inputs(tmp_path: Path) -> None:
    chain = Chain(tmp_path)
    assert [step.name for step in chain.pipeline().order()] == ["upper", "count"]
    assert [step.name for step in chain.pipeline().order(["upper"])] == ["upper"]


def test_second_run_skips_everything(tmp_path: Path) -> None:
    chain = Chain(tmp_path)
    assert statuses(chain.pipeline().run()) == [("upper", "ran"), ("count", "ran")]
    outcomes = chain.pipeline().run()
    assert statuses(outcomes) == [("upper", "skipped"), ("count", "skipped")]
    assert chain.ran == ["upper", "count"]
    assert chain.count.read_text() == "4"


def test_changed_input_reruns_downstream(tmp_path: Path) -> None:
    chain = Chain(tmp_path)
    chain.pipeline().run()
    chain.raw.write_text("abcdef\n")
    outcomes = chain.pipeline().run()
    assert statuses(outcomes) == [("upper", "ran"), ("count", "ran")]
    assert outcomes[0].reason == f"inputs changed: {chain.raw}"
    assert chain.count.read_text() == "7"


def test_changed_parameter_reruns(tmp_path: Path) -> None:
    chain = Chain(tmp_path)
    chain.pipeline().run()
    outcomes = chain.pipeline(repeat=2).run()
    assert outcomes[0].reason == "parameters changed: repeat"
    assert chain.count.read_text() == "8"


def test_identical_upstream_output_skips_downstream(tmp_path: Path) -> None:
    """Forcing upstream to re-run with the same result must not cascade."""
    chain = Chain(tmp_path)
    chain.pipeline().run()
    outcomes = chain.pipeline().run(force=["upper"])
    assert statuses(outcomes) == [("upper", "ran"), ("count", "skipped")]


def test_edited_output_reruns(tmp_path: Path) -> None:
    chain = Chain(tmp_path)
    chain.pipeline().run()
    chain.count.write_text("edited by hand")
    outcomes = chain.pipeline().run()
    assert outcomes[1].reason == f"output modified since last run: {chain.count}"
    assert chain.count.read_text() == "4"


def test_deleted_output_reruns(tmp_path: Path) -> None:
    chain = Chain(tmp_path)
    chain.pipeline().run()
    chain.upper.unlink()
    outcomes = chain.pipeline().run()
    assert outcomes[0].reason == f"output missing: {chain.upper}"


def test_record_and_provenance_written(tmp_path: Path) -> None:
    chain = Chain(tmp_path)
    chain.pipeline().run()
    record = json.loads((tmp_path / "state" / "upper.json").read_text())
    assert record["parameters"] == {"repeat": 1}
    assert list(record["inputs"]) == [str(chain.raw)]
    assert list(record["outputs"]) == [str(chain.upper)]
    provenance = read_provenance(chain.upper)
    assert provenance["step"] == "upper"
    assert provenance["inputs"][0]["path"] == str(chain.raw)


def test_missing_input_is_fatal(tmp_path: Path) -> None:
    chain = Chain(tmp_path)
    chain.raw.unlink()
    with pytest.raises(StepInputMissing, match="step 'upper': input missing"):
        chain.pipeline().run()


def test_step_that_writes_nothing_fails_and_is_not_recorded(tmp_path: Path) -> None:
    out = tmp_path / "out.txt"
    step = Step("lazy", lambda context: None, outputs=[Artefact(out)])
    pipeline = Pipeline([step], state_dir=tmp_path / "state", runner=LocalRunner())
    with pytest.raises(ArtefactError, match="missing"):
        pipeline.run()
    assert not (tmp_path / "state" / "lazy.json").exists()


def test_failure_stops_and_restart_resumes(tmp_path: Path) -> None:
    """A failed step stops the run; the next run skips what already finished."""
    chain = Chain(tmp_path)
    steps = chain.steps()
    failing = Step(
        "count",
        lambda context: (_ for _ in ()).throw(RuntimeError("tool crashed")),
        outputs=[Artefact(chain.count)],
        inputs=[chain.upper],
    )
    broken = Pipeline([failing, steps[1]], state_dir=tmp_path / "state", runner=LocalRunner())
    with pytest.raises(RuntimeError, match="tool crashed"):
        broken.run()
    outcomes = chain.pipeline().run()
    assert statuses(outcomes) == [("upper", "skipped"), ("count", "ran")]


def test_step_using_the_runner(tmp_path: Path) -> None:
    """A step that runs an external tool: a failing tool fails the pipeline."""
    out = tmp_path / "tool.txt"

    def action(context: StepContext) -> None:
        script = context.step.parameters["script"]
        context.runner.run(
            Job("tool", ["/bin/sh", "-c", script], tmp_path, tmp_path / "tool.log", [Artefact(out)])
        )

    def pipeline(script: str) -> Pipeline:
        step = Step("tool", action, outputs=[Artefact(out)], parameters={"script": script})
        return Pipeline([step], state_dir=tmp_path / "state", runner=LocalRunner())

    with pytest.raises(Exception, match="exit status 5"):
        pipeline("exit 5").run()
    assert statuses(pipeline("echo ok > tool.txt").run()) == [("tool", "ran")]


def test_definition_errors(tmp_path: Path) -> None:
    runner = LocalRunner()
    state = tmp_path / "state"
    a = tmp_path / "a"
    b = tmp_path / "b"

    def noop(context: StepContext) -> None:
        pass

    with pytest.raises(PipelineError, match="duplicate step names"):
        Pipeline(
            [Step("x", noop, [Artefact(a)]), Step("x", noop, [Artefact(b)])],
            state_dir=state,
            runner=runner,
        )
    with pytest.raises(PipelineError, match="output of both"):
        Pipeline(
            [Step("x", noop, [Artefact(a)]), Step("y", noop, [Artefact(a)])],
            state_dir=state,
            runner=runner,
        )
    with pytest.raises(PipelineError, match="unknown step"):
        Pipeline([Step("x", noop, [Artefact(a)], after=["nope"])], state_dir=state, runner=runner)
    with pytest.raises(PipelineError, match="dependency cycle: x -> y -> x"):
        Pipeline(
            [
                Step("x", noop, [Artefact(a)], inputs=[b]),
                Step("y", noop, [Artefact(b)], inputs=[a]),
            ],
            state_dir=state,
            runner=runner,
        )
    good = Pipeline([Step("x", noop, [Artefact(a)])], state_dir=state, runner=runner)
    with pytest.raises(PipelineError, match="unknown target"):
        good.run(["y"])
    with pytest.raises(PipelineError, match="unknown step"):
        good.run(force=["y"])


def registry_for(chain: Chain) -> dict[str, Any]:
    def upper(parameters: Mapping[str, Any]) -> Step:
        return next(s for s in chain.steps(int(parameters.get("repeat", 1))) if s.name == "upper")

    def count(parameters: Mapping[str, Any]) -> Step:
        return next(s for s in chain.steps() if s.name == "count")

    return {"upper": upper, "count": count}


def test_run_from_config(tmp_path: Path) -> None:
    chain = Chain(tmp_path)
    config = tmp_path / "pipeline.toml"
    config.write_text(
        """
        [pipeline]
        state_dir = "state"
        steps = ["upper", "count"]

        [runner]
        kind = "local"

        [parameters.upper]
        repeat = 3
        """
    )
    outcomes = run_from_config(config, registry_for(chain))
    assert statuses(outcomes) == [("upper", "ran"), ("count", "ran")]
    assert chain.count.read_text() == "12"
    assert (tmp_path / "state" / "count.json").exists()


def test_config_errors(tmp_path: Path) -> None:
    chain = Chain(tmp_path)
    config = tmp_path / "pipeline.toml"
    config.write_text(
        """
        [pipeline]
        state_dir = "state"
        steps = ["upper", "unknown"]

        [runner]
        kind = "local"

        [parameters.count]
        x = 1
        """
    )
    with pytest.raises(ConfigError) as caught:
        build_pipeline(config, registry_for(chain))
    assert caught.value.problems == [
        "pipeline.steps: unknown step 'unknown'",
        "parameters.count: no such step in pipeline.steps",
    ]


def test_slurm_runner_needs_its_table(tmp_path: Path) -> None:
    chain = Chain(tmp_path)
    config = tmp_path / "pipeline.toml"
    config.write_text('[pipeline]\nstate_dir = "s"\nsteps = []\n[runner]\nkind = "slurm"\n')
    with pytest.raises(ConfigError, match="runner.slurm: required"):
        build_pipeline(config, registry_for(chain))
    config.write_text('[pipeline]\nstate_dir = "s"\nsteps = []\n[runner]\nkind = "cloud"\n')
    with pytest.raises(ConfigError, match="expected 'local' or 'slurm'"):
        build_pipeline(config, registry_for(chain))
