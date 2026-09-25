"""Run a chain from the command line.

    python -m af.chain <chain.toml> <run.toml>          run or resume, review page, publish
    python -m af.chain <chain.toml> <run.toml> --review  only rebuild the review page

`chain.toml` says what the chain is (tables, map options, seed; `af.chain.config`).
`run.toml` says where and how it runs, so the same chain runs on a laptop, the HPC or `o`:

    dataset = "labx/h3-hi-turkey-labx/main"   # chains/<dataset> in the work area and the store
    optimiser = "core"            # "core" (af.map.optimise) or "stub"
    threads = 0                   # per process; 0 = all cores
    publish = true                # publish the finished chain to the store

    [paths]                       # af.store.PathsConfig
    store = "~/AC/eu/store"       # published versions (tables are read from here too)
    work = "~/AC/eu/work"         # resumable state: <work>/chains/<dataset>/

    [runner]                      # af.pipeline runner settings
    kind = "slurm"                # or "local"
    [runner.slurm]
    work_dir = "/scratch/af-jobs"

    [split]                       # optional: run each map's starts as array jobs
    chunks = 20
    threads = 4

The chain's working directory is `<work>/chains/<dataset>/`: `state/` (pipeline records),
`steps/`, `review/`, `chain.json`, `inputs/` (tables as charts, by map hash) and `tmp/`
(the split jobs' files). The work area must already exist (af.store.Work.open never
creates one), so a mistyped path fails instead of re-running every step.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

from af.chain.backend import optimiser_by_key
from af.chain.config import load_chain_config
from af.chain.engine import SplitStarts, run_chain
from af.chain.publish import publish_chain
from af.chain.review import build_review
from af.pipeline.config import RunnerSettings, make_runner
from af.store.work import PathsConfig, Work
from af.util.config import load_config


@dataclass(frozen=True)
class SplitSettings:
    chunks: int
    threads: int = 1


@dataclass(frozen=True)
class RunSettings:
    dataset: str
    optimiser: str
    paths: PathsConfig
    threads: int = 0
    publish: bool = True
    runner: RunnerSettings = field(default_factory=lambda: RunnerSettings(kind="local"))
    split: SplitSettings | None = None


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="python -m af.chain", description=__doc__.split("\n")[0])
    parser.add_argument("chain", type=Path, help="chain config (TOML)")
    parser.add_argument("run", type=Path, help="run config (TOML)")
    parser.add_argument("--review", action="store_true", help="only rebuild the review page")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    run = load_config(args.run, RunSettings)
    work = Work.open(run.paths.work).dataset("chains", run.dataset)
    cfg = load_chain_config(args.chain, inputs_dir=work.root / "inputs")
    if not args.review:
        split = None
        if run.split is not None:
            split = SplitStarts(run.split.chunks, work.tmp, run.split.threads)
        results = run_chain(
            cfg,
            work.root.parent,
            optimiser=optimiser_by_key(run.optimiser, threads=run.threads),
            runner=make_runner(run.runner, args.run),
            split=split,
            root=work.root,
        )
        remade = sum(1 for r in results if not r.reused)
        logging.info(
            "%s: %d steps, %d remade, %d reused",
            run.dataset,
            len(results),
            remade,
            len(results) - remade,
        )
    page = build_review(work.root)
    logging.info("review page: %s", page)
    if run.publish and not args.review:
        ref = publish_chain(run.paths.store, run.dataset, work.root)
        logging.info("published %s", ref)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
