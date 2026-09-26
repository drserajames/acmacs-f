"""The chain engine: merge tables one at a time and keep a map at every step.

Step 0 maps the first table (or takes a seed chart). Step n merges table n into step
n-1's chosen map, then makes an **incremental** map (step n-1's layout, new points
random) and a **scratch** map (random starts), and carries forward the one with the
lower stress (today's rule, ae `chains.py:choose_between_incremental_scratch`).

Each step is an `af.pipeline` step whose inputs are the table and the previous step's
`chosen.ace`. So the pipeline's content hashes decide what to remake:

* a changed table k re-runs step k; if its chosen map changes, step k+1's input changed
  and it re-runs too, and so on. Steps before k are skipped;
* a step whose output was edited or lost is remade; if the remade map is identical
  (maps are seeded from the inputs' hashes), the steps after it are still skipped;
* a step is never half-old: its directory is emptied before it runs, and its record is
  written only after every output checks out (today's chains kept merges from an earlier
  run beside maps from a later one).
"""

from __future__ import annotations

import functools
import hashlib
import json
import logging
import shutil
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from af.chain.backend import Optimiser, default_optimiser
from af.chain.config import ChainConfig, TableRef, config_to_json, option_parameters
from af.chain.diagnostics import group_moves, step_diagnostics
from af.chain.starts import read_result, write_problem
from af.chart.ace import read_chart, read_json, write_chart
from af.chart.merge import ColumnBasisConvention, MergeOptions, MergeType, merge
from af.chart.model import Chart, Projection
from af.chart.procrustes import procrustes
from af.chart.titre import MergeSettings
from af.pipeline import Pipeline, Step, StepContext
from af.run.job import Job, Resources, Runner
from af.run.local import LocalRunner
from af.util.artefacts import Artefact, sha256_path

log = logging.getLogger(__name__)

STEP_RECORD = "step.json"
CHOSEN = "chosen.ace"
CHAIN_RECORD = "chain.json"


class ChainError(RuntimeError):
    pass


@dataclass
class StepResult:
    index: int
    directory: Path
    reused: bool
    record: dict[str, Any]


@dataclass(frozen=True)
class SplitStarts:
    """Run a map's starts as `chunks` jobs through the step's runner (af.run: local or SLURM).

    Not part of any step's key: per-start seeding makes the maps the same however the
    starts are split. `work_dir` holds the jobs' problem and result files (outside the
    store), `threads` is each job's thread count.
    """

    chunks: int
    work_dir: Path
    threads: int = 1


class Mapper:
    """Makes a map's starts, in one process or split into jobs."""

    def __init__(self, optimiser: Optimiser, split: SplitStarts | None) -> None:
        self.optimiser = optimiser
        self.split = split

    def make(
        self,
        runner: Runner,
        label: str,
        arrays: dict,
        n_starts: int,
        dim: int,
        seed: int,
        start_layout: np.ndarray | None,
        move_groups: bool = False,
    ) -> list[dict]:
        if self.split is None or self.split.chunks <= 1:
            return self.optimiser.optimise(
                arrays, n_starts, dim, seed, start_layout, move_groups=move_groups
            )
        work = self.split.work_dir / label
        if work.exists():
            shutil.rmtree(work)
        work.mkdir(parents=True)
        problem = write_problem(
            work / "problem.npz",
            arrays,
            seed=seed,
            dimensions=dim,
            optimiser=self.optimiser.key,
            start_layout=start_layout,
        )
        jobs = []
        for k, (first, count) in enumerate(_chunks(n_starts, self.split.chunks)):
            out = work / f"starts-{first:06d}-{count:06d}.npz"
            jobs.append(
                Job(
                    name=f"{label.replace('/', '-')}-{k:03d}",
                    command=[
                        sys.executable,
                        "-m",
                        "af.chain.starts",
                        problem,
                        str(first),
                        str(count),
                        out,
                        str(self.split.threads),
                    ],
                    cwd=work,
                    log=work / f"starts-{k:03d}.log",
                    outputs=[Artefact(out)],
                    resources=Resources(threads=self.split.threads),
                )
            )
        runner.run_many(jobs)
        chunks = [m for job in jobs for m in read_result(Path(job.outputs[0].path))]
        return self.optimiser.combine(
            arrays, chunks, incremental=start_layout is not None, move_groups=move_groups
        )


def _chunks(n: int, k: int) -> list[tuple[int, int]]:
    """Split starts 0..n-1 into k contiguous (first, count) runs, sizes differing by at most one."""
    k = min(k, n)
    base, extra = divmod(n, k)
    out, first = [], 0
    for i in range(k):
        count = base + (1 if i < extra else 0)
        out.append((first, count))
        first += count
    return out


def step_dir(root: Path, index: int) -> Path:
    return root / "steps" / f"{index:04d}"


def _ace(path: Path) -> object:
    return read_json(path)


def _json(path: Path) -> object:
    return json.loads(path.read_text())


# ----------------------------------------------------------------------


def chain_steps(cfg: ChainConfig, root: Path, mapper: Mapper) -> list[Step]:
    """One pipeline step per chain step, each depending on the previous step's chosen map."""
    first_source = cfg.first_map if cfg.first_map else cfg.tables[0].path
    merged_tables = cfg.tables if cfg.first_map else cfg.tables[1:]
    common = {
        "chain": cfg.name,
        "options": option_parameters(cfg.options),
        "optimiser": mapper.optimiser.name,
        "seed": cfg.seed,
    }
    steps = []
    d0 = step_dir(root, 0)
    first_outputs = [d0 / CHOSEN, d0 / STEP_RECORD] + (
        [] if cfg.first_map else [d0 / "scratch.ace"]
    )
    steps.append(
        Step(
            name="step-0000",
            action=_action(functools.partial(_first_step, cfg, mapper=mapper), d0),
            inputs={"source": Path(first_source)},
            outputs=[
                Artefact(p, parse=_ace if p.suffix == ".ace" else _json) for p in first_outputs
            ],
            parameters={**common, "index": 0, "source": Path(first_source).name},
        )
    )
    for index, table in enumerate(merged_tables, start=1):
        d = step_dir(root, index)
        previous = step_dir(root, index - 1) / CHOSEN
        outputs = [d / n for n in ("merge.ace", "incremental.ace", "scratch.ace", CHOSEN)]
        steps.append(
            Step(
                name=f"step-{index:04d}",
                action=_action(
                    functools.partial(
                        _merge_step,
                        cfg,
                        index=index,
                        previous_path=previous,
                        table_ref=table,
                        mapper=mapper,
                    ),
                    d,
                ),
                inputs={"previous": previous, "table": table.path},
                outputs=[Artefact(p, parse=_ace) for p in outputs]
                + [Artefact(d / STEP_RECORD, parse=_json)],
                parameters={**common, "index": index, "table_id": table.table_id},
            )
        )
    return steps


def _action(work: Callable[[Path, Runner], None], directory: Path) -> Callable[[StepContext], None]:
    def run(context: StepContext) -> None:
        if directory.exists():
            shutil.rmtree(directory)  # never leave an earlier run's files beside this one's
        directory.mkdir(parents=True)
        work(directory, context.runner)

    return run


def run_chain(
    cfg: ChainConfig,
    store_root: Path,
    optimiser: Optimiser | None = None,
    runner: Runner | None = None,
    until_step: int | None = None,
    split: SplitStarts | None = None,
    root: Path | None = None,
) -> list[StepResult]:
    """Run or resume a chain into `store_root/<name>/`; returns every step, skipped or remade.

    `runner` runs the start chunks when `split` is given (af.run LocalRunner or SlurmRunner).
    """
    optimiser = optimiser or default_optimiser()
    # `root` is the chain's own directory (af.store.Work: <work>/chains/<dataset>); without
    # it, <store_root>/<name>. Pipeline state goes in <root>/state (DatasetWork.state).
    root = Path(root) if root is not None else Path(store_root) / cfg.name
    steps = chain_steps(cfg, root, Mapper(optimiser, split))
    # Inputs are named and outputs recorded relative to the chain root, so a store or a tables
    # checkout moved to another root (a sync to o or the HPC) re-runs nothing.
    pipeline = Pipeline(steps, state_dir=root / "state", runner=runner or LocalRunner(), root=root)
    targets = None if until_step is None else [steps[until_step].name]
    outcomes = pipeline.run(targets)
    results = [
        StepResult(
            i,
            step_dir(root, i),
            o.status == "skipped",
            json.loads((step_dir(root, i) / STEP_RECORD).read_text()),
        )
        for i, o in enumerate(outcomes)
    ]
    _write_chain_record(root, cfg, results, optimiser, n_steps=len(steps))
    return results


def _step_seed(cfg: ChainConfig, index: int, inputs: list[Path]) -> int:
    """Seeds follow the inputs' content, so the same inputs always give the same maps."""
    h = hashlib.sha256(f"{cfg.seed}:{index}".encode())
    for p in inputs:
        h.update(sha256_path(Path(p)).encode())
    return int(h.hexdigest()[:16], 16)


# ----------------------------------------------------------------------
# steps


def _merge_options(cfg: ChainConfig) -> MergeOptions:
    o = cfg.options
    return MergeOptions(
        merge_type=MergeType.INCREMENTAL,
        combine_cheating_assays=o.combine_cheating_assays,
        titres=MergeSettings(sd_limit=o.sd_limit),
        column_bases=ColumnBasisConvention(o.column_bases),
    )


def distinct_basins(maps: list[dict], keep: int, min_rmsd: float = 0.5) -> list[dict]:
    """Up to `keep` maps, best first, each more than `min_rmsd` (Procrustes) from those kept.

    Ten copies of one basin tell a reviewer nothing; the alternatives are what matter.
    """
    kept: list[dict] = []
    for m in sorted(maps, key=lambda r: r["stress"]):
        if len(kept) >= keep:
            break
        if all(procrustes(k["layout"], m["layout"]).rmsd > min_rmsd for k in kept):
            kept.append(m)
    return kept


def _projections(results: list[dict], arrays: dict, mcb: str) -> list[Projection]:
    disconnected = tuple(int(i) for i in np.nonzero(arrays["disconnected"])[0])
    return [
        Projection(
            layout=r["layout"],
            stress=r["stress"],
            minimum_column_basis=mcb,
            forced_column_bases=np.asarray(arrays["column_bases"], dtype=float),
            disconnected=disconnected,
            comment=f"start {r['start_seed']}, {r['n_iterations']} iterations",
        )
        for r in results
    ]


def _map_chart(chart: Chart, projections: list[Projection]) -> Chart:
    return Chart(
        chart.info,
        chart.antigens,
        chart.sera,
        chart.titres,
        chart.forced_column_bases,
        projections,
        dict(chart.extra),
    )


def _loop(maps: list[dict]) -> dict:
    best = min(maps, key=lambda r: r["stress"])
    loop = {"rounds": best.get("resolved_rounds", 0), "moves": best.get("resolved_moves", 0)}
    if "resolved_groups" in best:
        loop["groups"] = best["resolved_groups"]
    return loop


def _write_step_record(directory: Path, record: dict) -> None:
    (directory / STEP_RECORD).write_text(json.dumps(record, indent=1, default=_json_default))


def _json_default(o: object) -> object:
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def _first_step(cfg: ChainConfig, directory: Path, runner: Runner, *, mapper: Mapper) -> None:
    optimiser = mapper.optimiser
    o = cfg.options
    if cfg.first_map:
        chart = read_chart(cfg.first_map)
        if not chart.projections:
            raise ChainError(f"{cfg.first_map}: seed chart has no projections")
        write_chart(chart, directory / CHOSEN)
        record = {
            "index": 0,
            "table_id": Path(cfg.first_map).name,
            "source": str(cfg.first_map),
            "chosen": "seed",
            "stress": {"seed": chart.best().stress_value},
        }
        record["diagnostics"] = step_diagnostics(chart, None, None, None, None, None, cfg)
        _write_step_record(directory, record)
        return
    t = cfg.tables[0]
    table = read_chart(t.path)
    arrays = table.optimiser_arrays(
        o.minimum_column_basis, disconnect_threshold=o.disconnect_threshold
    )
    seed = _step_seed(cfg, 0, [t.path])
    all_maps = mapper.make(
        runner,
        f"{cfg.name}/0000/scratch",
        arrays,
        o.scratch_starts,
        o.dimensions,
        seed,
        None,
        move_groups=o.move_groups,
    )
    chart = _map_chart(
        table,
        _projections(
            distinct_basins(all_maps, o.projections_to_keep), arrays, o.minimum_column_basis
        ),
    )
    write_chart(chart, directory / "scratch.ace")
    shutil.copyfile(directory / "scratch.ace", directory / CHOSEN)
    record = {
        "index": 0,
        "table_id": t.table_id,
        "table_sha256": sha256_path(t.path),
        "optimiser": optimiser.name,
        "chosen": "scratch",
        "stress": {"scratch": all_maps[0]["stress"]},
        "start_stresses": {"scratch": [r["stress"] for r in all_maps]},
        "trapped_loop": {"scratch": _loop(all_maps)},
    }
    diagnostics = step_diagnostics(chart, None, None, None, arrays, optimiser, cfg)
    if o.move_groups:
        diagnostics["group_moves"] = group_moves(chart, _loop(all_maps).get("groups"))
    record["diagnostics"] = diagnostics
    _write_step_record(directory, record)


def _merge_step(
    cfg: ChainConfig,
    directory: Path,
    runner: Runner,
    *,
    index: int,
    previous_path: Path,
    table_ref: TableRef,
    mapper: Mapper,
) -> None:
    o = cfg.options
    optimiser = mapper.optimiser
    previous = read_chart(previous_path)
    merged, report = merge(previous, read_chart(table_ref.path), _merge_options(cfg))
    write_chart(merged, directory / "merge.ace")
    arrays = merged.optimiser_arrays(
        o.minimum_column_basis, disconnect_threshold=o.disconnect_threshold
    )
    seed = _step_seed(cfg, index, [previous_path, table_ref.path])
    start = merged.projections[0].layout.copy()
    start[arrays["disconnected"]] = np.nan
    label = f"{cfg.name}/{index:04d}"
    all_incremental = mapper.make(
        runner,
        f"{label}/incremental",
        arrays,
        o.incremental_starts,
        o.dimensions,
        seed,
        start,
        move_groups=o.move_groups,
    )
    all_scratch = mapper.make(
        runner,
        f"{label}/scratch",
        arrays,
        o.scratch_starts,
        o.dimensions,
        seed ^ 0x5DEECE66D,
        None,
        move_groups=o.move_groups,
    )
    inc_chart = _map_chart(
        merged,
        _projections(
            distinct_basins(all_incremental, o.projections_to_keep), arrays, o.minimum_column_basis
        ),
    )
    scr_chart = _map_chart(
        merged,
        _projections(
            distinct_basins(all_scratch, o.projections_to_keep), arrays, o.minimum_column_basis
        ),
    )
    write_chart(inc_chart, directory / "incremental.ace")
    write_chart(scr_chart, directory / "scratch.ace")
    s_inc, s_scr = all_incremental[0]["stress"], all_scratch[0]["stress"]
    chosen_name = "incremental" if s_inc <= s_scr else "scratch"  # ties go to incremental, as today
    chosen = inc_chart if chosen_name == "incremental" else scr_chart
    shutil.copyfile(directory / f"{chosen_name}.ace", directory / CHOSEN)
    record = {
        "index": index,
        "table_id": table_ref.table_id,
        "table_sha256": sha256_path(table_ref.path),
        "seeded_from_sha256": sha256_path(previous_path),
        "optimiser": optimiser.name,
        "chosen": chosen_name,
        "stress": {"incremental": s_inc, "scratch": s_scr},
        "start_stresses": {
            "incremental": [r["stress"] for r in all_incremental],
            "scratch": [r["stress"] for r in all_scratch],
        },
        "trapped_loop": {"incremental": _loop(all_incremental), "scratch": _loop(all_scratch)},
        "merge": {
            "antigens": merged.n_antigens,
            "sera": merged.n_sera,
            "layers": len(merged.titres.layers),
            "common_antigens": report.common_antigens,
            "common_sera": report.common_sera,
            "new_antigens": report.new_antigens,
            "new_sera": report.new_sera,
            "cheating_assay": report.cheating_assay,
            "skipped_reference_antigens": report.skipped_reference_antigens,
            "outcomes": dict(report.outcomes),
            "column_basis_slack": {str(k): v for k, v in report.column_basis_slack.items()},
        },
    }
    diagnostics = step_diagnostics(
        chosen, previous, inc_chart, scr_chart, arrays, optimiser, cfg, report
    )
    if o.move_groups:
        chosen_maps = all_incremental if chosen_name == "incremental" else all_scratch
        diagnostics["group_moves"] = group_moves(chosen, _loop(chosen_maps).get("groups"))
    record["diagnostics"] = diagnostics
    _write_step_record(directory, record)


def _write_chain_record(
    root: Path, cfg: ChainConfig, results: list[StepResult], optimiser: Optimiser, n_steps: int
) -> None:
    """chain.json lists the current steps in order: consumers never glob or sort directories."""
    doc = {
        "config": config_to_json(cfg),
        "optimiser": optimiser.name,
        "complete": len(results) == n_steps,
        "steps": [
            {
                "index": r.index,
                "directory": str(r.directory.relative_to(root)),
                "table_id": r.record["table_id"],
                "chosen": r.record["chosen"],
                "chosen_file": CHOSEN,
                "reused": r.reused,
            }
            for r in results
        ],
    }
    # directories left from a longer chain (a table removed): not part of this chain
    doc["stale_step_directories"] = sorted(
        p.name
        for p in (root / "steps").iterdir()
        if p.is_dir() and not p.name.startswith(".") and int(p.name) >= n_steps
    )
    tmp = root / (CHAIN_RECORD + ".tmp")
    tmp.write_text(json.dumps(doc, indent=1))
    tmp.replace(root / CHAIN_RECORD)
