"""Build a tree with CMAPLE.

Sarah's requirement (INVENTORY §0): keep using CMAPLE, and no deduplication of identical
sequences — CMAPLE handles them, and dropping them costs the leaves that a report has to show.

The command goes through ``af.run``, so the same code runs locally or under SLURM and the runner
checks the output before reporting success. That matters here more than anywhere: production's
`make-cmaple` counts a failed run, logs it, mails "completed" and exits 0, leaving
`cmaple_tree.txt` pointing at a tree that does not exist — and the next week's build then links to
the missing file (INVENTORY §5). Here the treefile is a declared artefact, parsed before the step
is allowed to succeed.

Incremental builds pass the previous tree with ``-t``. Its tips are pruned to the new selection
first: ae's `Tree::remove` leaves unary nodes behind and CMAPLE segfaults on them, so af prunes
with its own model, which cannot leave one (see af/tree/model.py).
"""

from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from af.run import Job, LocalRunner, Resources, Runner
from af.tree.io import newick
from af.tree.model import Tree

SEARCH_TYPES = ("FAST", "NORMAL", "EXHAUSTIVE")


@dataclass(frozen=True)
class CmapleSettings:
    """Everything CMAPLE is told. From config; nothing is read from the environment."""

    executable: str = "cmaple"
    model: str = "GTR"
    search: str = "EXHAUSTIVE"
    min_branch_length: float = 1e-4
    seed: int = 1
    threads: int = 4

    def __post_init__(self) -> None:
        if self.search not in SEARCH_TYPES:
            raise ValueError(f"search must be one of {SEARCH_TYPES}, not {self.search!r}")
        if self.min_branch_length <= 0:
            raise ValueError("min_branch_length must be positive")


def version(executable: str = "cmaple") -> str:
    """CMAPLE's own version string, for the provenance."""
    if shutil.which(executable) is None:
        raise FileNotFoundError(f"{executable} is not on PATH")
    done = subprocess.run([executable, "--help"], capture_output=True, text=True, check=False)
    match = re.search(r"CMAPLE version (\S+)", done.stdout + done.stderr)
    return f"cmaple/{match.group(1)}" if match else "cmaple/unknown"


def build_job(
    alignment: Path,
    out_dir: Path,
    settings: CmapleSettings,
    starting_tree: Path | None = None,
    name: str = "cmaple",
) -> tuple[Job, Path]:
    """The CMAPLE job, and the treefile it must produce."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = out_dir / "cmaple"
    treefile = prefix.with_suffix(".treefile")

    command: list[str | Path] = [
        settings.executable,
        "-aln",
        alignment,
        "-m",
        settings.model,
        "-st",
        "DNA",
        "--search",
        settings.search,
        "--min-blength",
        str(settings.min_branch_length),
        "-nt",
        str(settings.threads),
        "--seed",
        str(settings.seed),
        "--prefix",
        prefix,
        "--overwrite",
    ]
    if starting_tree is not None:
        command += ["-t", starting_tree]

    from af.util.artefacts import Artefact

    job = Job(
        name=name,
        command=command,
        cwd=out_dir,
        log=out_dir / "cmaple.log",
        outputs=[Artefact(treefile, parse=_parse_tree)],
        resources=Resources(threads=settings.threads),
    )
    return job, treefile


def _parse_tree(path: Path) -> object:
    """Artefact check: the treefile must be a tree, not just a non-empty file."""
    tree = newick.loads(path.read_text())
    leaves, _ = tree.count()
    if leaves < 2:
        raise ValueError(f"{path}: parsed to {leaves} leaves")
    return tree


def run_cmaple(
    alignment: Path,
    out_dir: Path,
    settings: CmapleSettings | None = None,
    starting_tree: Path | None = None,
    runner: Runner | None = None,
) -> Tree:
    """Build a tree and return it. Raises JobFailed if CMAPLE fails or writes no usable tree."""
    settings = settings or CmapleSettings()
    job, treefile = build_job(alignment, out_dir, settings, starting_tree)
    (runner or LocalRunner()).run(job)
    return newick.load(treefile)


def prune_starting_tree(previous: Tree, keep: list[str], out_path: Path) -> tuple[Path, int]:
    """Write the previous tree cut down to the current selection, for an incremental build.

    Leaves that are gone are removed and any node left with one child is spliced out, because
    CMAPLE segfaults on unary nodes. Returns the path and how many leaves were dropped.
    """
    removed = previous.prune(keep)
    newick.dump(previous, out_path)
    return out_path, removed
