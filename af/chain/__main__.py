"""Run a chain from the command line.

    python -m af.chain <chain.toml> <run.toml>          run or resume, then write the review page
    python -m af.chain <chain.toml> <run.toml> --review  only rebuild the review page

`chain.toml` says what the chain is (tables, map options, seed; `af.chain.config`).
`run.toml` says where and how it runs, so the same chain runs on a laptop, the HPC or `o`:

    store_root = "/path/to/store/chains"
    optimiser = "core"            # "core" (af.map.optimise) or "stub"
    threads = 0                   # per process; 0 = all cores

    [runner]                      # af.pipeline runner settings
    kind = "slurm"                # or "local"
    [runner.slurm]
    work_dir = "/scratch/af-jobs"

    publish_store = "/path/to/store"   # optional: publish the finished chain ...
    publish_as = "labx/h3-hi-turkey-labx/main"  # ... as chains/<publish_as> (af.store)

    [split]                       # optional: run each map's starts as array jobs
    chunks = 20
    threads = 4
    work_dir = "/scratch/af-starts"
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
from af.util.config import load_config


@dataclass(frozen=True)
class SplitSettings:
    chunks: int
    work_dir: Path
    threads: int = 1


@dataclass(frozen=True)
class RunSettings:
    store_root: Path
    optimiser: str
    threads: int = 0
    runner: RunnerSettings = field(default_factory=lambda: RunnerSettings(kind="local"))
    split: SplitSettings | None = None
    publish_store: Path | None = None  # af store root; publish as chains/<publish_as>
    publish_as: str | None = None  # e.g. "labx/h3-hi-turkey-labx/main"


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="python -m af.chain", description=__doc__.split("\n")[0])
    parser.add_argument("chain", type=Path, help="chain config (TOML)")
    parser.add_argument("run", type=Path, help="run config (TOML)")
    parser.add_argument("--review", action="store_true", help="only rebuild the review page")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    run = load_config(args.run, RunSettings)
    cfg = load_chain_config(args.chain, inputs_dir=run.store_root / "inputs")
    root = run.store_root / cfg.name
    if not args.review:
        split = None
        if run.split is not None:
            split = SplitStarts(run.split.chunks, run.split.work_dir / cfg.name, run.split.threads)
        results = run_chain(
            cfg,
            run.store_root,
            optimiser=optimiser_by_key(run.optimiser, threads=run.threads),
            runner=make_runner(run.runner, args.run),
            split=split,
        )
        remade = sum(1 for r in results if not r.reused)
        logging.info(
            "%s: %d steps, %d remade, %d reused",
            cfg.name,
            len(results),
            remade,
            len(results) - remade,
        )
    page = build_review(root)
    logging.info("review page: %s", page)
    if run.publish_store is not None and run.publish_as is not None and not args.review:
        ref = publish_chain(run.publish_store, run.publish_as, root)
        logging.info("published %s", ref)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
