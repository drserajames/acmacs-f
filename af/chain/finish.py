"""Finish a published merged map: the next version of the dataset, made from the CURRENT
version's chosen layout by the trapped-point loop under a named rule, without new starts.

    python -m af.chain.finish <chain.toml> <run.toml> --version <id> --move-groups

Why: a published map can sit in a local minimum that the loop's own rule left, with a lower one a
few moves away (Q195, 14 Oct 2026: a merged map whose block of points the loop could not move).
Finishing takes the lower one from the SAME layout, so the map changes only where the rule moved
it, and the version records the rule, the stress before and after, and the version it came from.

It refuses, before anything is computed, unless:
- the version is the dataset's CURRENT and a one-step merge_all version;
- today's input table files have the names and sha256 its build recorded, and today's step
  parameters are the ones its build recorded (steps/0000/chosen.ace.provenance.json): the same
  chain, options, seed, tables and arithmetic. So a version made under the original arithmetic
  is finished only by a run config that says numerics = "original";
- today's tables merge to its merge.ace byte for byte.
A rule that leaves the layout unchanged makes no version. Otherwise the new version is published
only when the run config says publish, with the finished version as its parent.
"""

from __future__ import annotations

import argparse
import functools
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from af.chain.__main__ import (
    RunSettings,
    _optimiser,
    check_tables_dataset,
    dataset_from_path,
    measure_references,
)
from af.chain.backend import FinishRule, Optimiser
from af.chain.config import ChainConfig, ChainConfigError, load_chain_config
from af.chain.diagnostics import step_diagnostics
from af.chain.engine import (
    CHOSEN,
    STEP_RECORD,
    Mapper,
    StepResult,
    _ace,
    _action,
    _json,
    _map_chart,
    _merge_all_record,
    _projections,
    _write_chain_record,
    _write_step_record,
    chain_steps,
    merge_all_tables,
    step_dir,
)
from af.chain.publish import publish_chain
from af.chain.review import build_review
from af.chart.ace import read_chart, write_chart
from af.pipeline import Pipeline, Step
from af.pipeline.record import canonical, fingerprint, input_roles
from af.run.job import Runner, run_main
from af.run.local import LocalRunner
from af.run.runtime import require_python
from af.store.store import Store
from af.store.work import Work
from af.util.artefacts import Artefact, sha256_path
from af.util.config import load_config

log = logging.getLogger(__name__)


class FinishError(RuntimeError):
    pass


@dataclass(frozen=True)
class Source:
    """The published version being finished."""

    version: str
    directory: Path  # its steps/0000
    map_sha256: str
    stress: float

    @property
    def chosen(self) -> Path:
        return self.directory / CHOSEN

    def to_json(self) -> dict[str, Any]:
        return {"version": self.version, "map_sha256": self.map_sha256, "stress": self.stress}


def check_source(
    cfg: ChainConfig, store_root: Path, dataset: str, version: str, step: Step, root: Path
) -> Source:
    """The version, once it is shown to be today's chain's own map (see the module docstring)."""
    store = Store.open(store_root)
    current = store.current("chains", dataset)
    if current.version != version:
        raise FinishError(f"{dataset}: {version} is not CURRENT ({current.version})")
    vdir = store.resolve(current)
    doc = json.loads((vdir / "chain.json").read_text())
    if doc.get("mode") != "merge_all" or [s["directory"] for s in doc["steps"]] != ["steps/0000"]:
        raise FinishError(f"{dataset}@{version} is not a one-step merge_all version")
    sdir = vdir / "steps" / "0000"
    built = json.loads((sdir / f"{CHOSEN}.provenance.json").read_text())
    current_fp = fingerprint(step.name, input_roles(step.name, step.inputs, root), step.parameters)
    was = [(Path(i["path"]).name, i["sha256"]) for i in built["inputs"]]
    now = [(Path(current_fp.paths[r]).name, current_fp.inputs[r]) for r in current_fp.inputs]
    if was != now:
        differ = [a[0] for a, b in zip(was, now, strict=False) if a != b]
        raise FinishError(
            f"{dataset}@{version}: today's input tables are not the ones it was built from"
            f" ({len(was)} then, {len(now)} now; first differing: {differ[:3]})"
        )
    params, recorded = canonical(step.parameters), built["parameters"]
    changed = sorted(k for k in set(params) | set(recorded) if params.get(k) != recorded.get(k))
    if "numerics" in changed:
        made, asked = recorded.get("numerics", "original"), params.get("numerics", "original")
        raise FinishError(
            f"{dataset}@{version} was made under numerics {made!r}; the run config asks for"
            f" {asked!r}: a finish must not mix arithmetics"
        )
    if changed:
        raise FinishError(f"{dataset}@{version}: step parameters differ from its build: {changed}")
    merged = merge_all_tables(cfg)
    check = root / "tmp" / "merge-check.ace"
    check.parent.mkdir(parents=True, exist_ok=True)
    write_chart(merged.merged, check)
    if sha256_path(check) != sha256_path(sdir / "merge.ace"):
        raise FinishError(f"{dataset}@{version}: today's tables do not merge to its merge.ace")
    check.unlink()
    chart = read_chart(sdir / CHOSEN)
    return Source(version, sdir, sha256_path(sdir / CHOSEN), chart.best().stress_value)


def finish_step(
    cfg: ChainConfig, root: Path, step: Step, source: Source, rule: FinishRule, mapper: Mapper
) -> Step:
    """The finish as a pipeline step: the chain's own inputs and parameters plus the source map
    and the rule, so it is never taken for a search, and a rerun that changed nothing reuses it."""
    d0 = step_dir(root, 0)
    return Step(
        name="step-0000",
        action=_action(
            functools.partial(_finish, cfg, source=source, rule=rule, mapper=mapper), d0
        ),
        inputs={**dict(step.inputs), "source": source.chosen},  # type: ignore[arg-type]
        outputs=[Artefact(d0 / n, parse=_ace) for n in ("merge.ace", CHOSEN)]
        + [Artefact(d0 / STEP_RECORD, parse=_json)],
        parameters={
            **step.parameters,
            "finished_from": {"version": source.version, "map_sha256": source.map_sha256},
            "finish": rule.to_json(),
        },
    )


def _finish(
    cfg: ChainConfig,
    directory: Path,
    runner: Runner,
    *,
    source: Source,
    rule: FinishRule,
    mapper: Mapper,
) -> None:
    m = merge_all_tables(cfg)
    write_chart(m.merged, directory / "merge.ace")
    if sha256_path(directory / "merge.ace") != sha256_path(source.directory / "merge.ace"):
        raise FinishError(f"{source.version}: the tables no longer merge to its merge.ace")
    src = read_chart(source.chosen)
    best = src.best()
    finished = mapper.optimiser.finish(m.arrays, np.asarray(best.layout), rule)
    rest = [p for p in src.projections if p is not best]  # the other basins kept, unchanged
    projections = [*_projections([finished], m.arrays, cfg.options.minimum_column_basis), *rest]
    chart = _map_chart(m.chart, projections)
    write_chart(chart, directory / CHOSEN)
    base = _merge_all_record(cfg, m)
    record = {
        **{k: base[k] for k in ("index", "table_id", "merge_all")},
        "optimiser": mapper.optimiser.name,
        "chosen": "finished",
        "stress": {"finished": finished["stress"], "source": source.stress},
        "finished_from": source.to_json(),
        "finish": finished["finish"],
        **{k: v for k, v in base.items() if k not in ("index", "table_id", "merge_all")},
    }
    diagnostics = step_diagnostics(
        chart, None, None, None, m.arrays, mapper.optimiser, cfg, m.report
    )
    diagnostics.pop("control_flags", None)  # as for any merge_all map
    record["diagnostics"] = diagnostics
    _write_step_record(directory, record, mapper)


def finish_chain(
    cfg: ChainConfig,
    root: Path,
    source: Source,
    optimiser: Optimiser,
    rule: FinishRule,
    step: Step,
    site: str | None = None,
) -> StepResult:
    """Run the finish into `root` (its own chain root) and write its chain.json."""
    fin = finish_step(cfg, root, step, source, rule, Mapper(optimiser, None, site))
    [outcome] = Pipeline([fin], state_dir=root / "state", runner=LocalRunner(), root=root).run()
    d0 = step_dir(root, 0)
    result = StepResult(
        0, d0, outcome.status == "skipped", json.loads((d0 / STEP_RECORD).read_text())
    )
    _write_chain_record(root, cfg, [result], optimiser, n_steps=1)
    path = root / "chain.json"
    doc = json.loads(path.read_text())
    doc["finished_from"] = source.to_json()
    doc["finish"] = rule.to_json()
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(doc, indent=1))
    tmp.replace(path)
    return result


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m af.chain.finish", description=__doc__.split("\n")[0]
    )
    parser.add_argument("chain", type=Path, help="chain config (TOML)")
    parser.add_argument("run", type=Path, help="run config (TOML)")
    parser.add_argument("--version", required=True, help="the dataset's CURRENT version to finish")
    parser.add_argument(
        "--move-groups", action="store_true", help="the loop's group pass (resolve_trapped)"
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    rule = FinishRule(move_groups=args.move_groups)
    dataset = dataset_from_path(args.chain)
    check_tables_dataset(args.chain, dataset)
    run = load_config(args.run, RunSettings)
    if run.python is not None:
        require_python(run.python)
    if run.optimiser != "core":
        raise ChainConfigError(f"finishing needs the core optimiser, not {run.optimiser!r}")
    optimiser = _optimiser(run)
    work = Work.open(run.paths.work).dataset("chains", dataset)
    cfg = load_chain_config(
        args.chain, inputs_dir=work.root / "inputs", tables_store=run.paths.store
    )
    [step] = chain_steps(cfg, work.root, Mapper(optimiser, None, run.site))
    root = work.root / "finish" / args.version
    root.mkdir(parents=True, exist_ok=True)
    source = check_source(cfg, Path(run.paths.store), dataset, args.version, step, work.root)
    result = finish_chain(cfg, root, source, optimiser, rule, step, run.site)
    log.info(
        "%s@%s finished: stress %.6f -> %.6f (%s)",
        dataset,
        args.version,
        source.stress,
        result.record["stress"]["finished"],
        "reused" if result.reused else "made",
    )
    if sha256_path(step_dir(root, 0) / CHOSEN) == source.map_sha256 or (
        _same_layout(step_dir(root, 0) / CHOSEN, source.chosen)
    ):
        log.info("%s@%s: the rule moved nothing; no version is made", dataset, args.version)
        return 0
    if cfg.references:
        measure_references(cfg, run, root)
    page = build_review(root)
    log.info("review page: %s", page)
    if run.publish:
        ref = publish_chain(Path(run.paths.store), dataset, root)
        log.info("published %s (parent %s)", ref, args.version)
    return 0


def _same_layout(a: Path, b: Path) -> bool:
    return bool(
        np.array_equal(read_chart(a).best().layout, read_chart(b).best().layout, equal_nan=True)
    )


if __name__ == "__main__":
    run_main(lambda: main(sys.argv[1:]))
