"""raxml-ng ancestral reconstruction: the reference, kept for comparison.

This is what production runs (`ae/proj/weekly-tree/raxml-asr`), and af keeps it so a tree can be
reconstructed the old way and compared. It is not a candidate for the default: measured at 2-4
hours per subtype on `o`, against TreeTime's 8 minutes for the same tree.

Two differences from production, both deliberate:

- ``--opt-branches off``. raxml-ng ``--ancestral`` re-optimises branch lengths by default (its log
  says "branch lengths: proportional (ML estimate)"), so production's reconstruction silently
  discards the CMAPLE lengths it was given.
- ``GTR+G`` is offered alongside production's ``GTR+G+I`` because ``+I`` produced an identical
  reconstruction for more runtime (notes/trees/ASR.md). The default here stays ``GTR+G+I`` so a
  comparison against the delivered trees is like for like.

raxml-ng reconstructs no gaps at all, so it is refused where deletions define clades.

It also refuses a multifurcating tree ("Binary tree expected"), which af's finished trees normally
are, so it is given a zero-length-resolved copy; the count is in the reconstruction's parameters.
"""

from __future__ import annotations

import re
import subprocess
import time
from pathlib import Path

from af.tree.asr.base import AncestralStates, BackendUnavailable
from af.tree.asr.matching import match_states_by_clade
from af.tree.io import newick
from af.tree.model import Tree


class RaxmlBackend:
    """`raxml-ng --ancestral`, the production reference."""

    name = "raxml"
    reconstructs_gaps = False
    optimises_branch_lengths = False

    def __init__(
        self,
        executable: str | Path = "raxml-ng",
        model: str = "GTR+G+I",
        seed: int = 1,
        optimise_branches: bool = False,
    ) -> None:
        self.executable = str(executable)
        self.model = model
        self.seed = seed
        self.optimise_branches = optimise_branches

    def version(self) -> str:
        try:
            done = subprocess.run(
                [self.executable, "--version"], capture_output=True, text=True, check=False
            )
        except OSError as error:
            raise BackendUnavailable(f"cannot run {self.executable!r}: {error}") from error
        match = re.search(r"RAxML-NG v\. (\S+)", done.stdout + done.stderr)
        return f"raxml-ng/{match.group(1)}" if match else "raxml-ng/unknown"

    def reconstruct(
        self, tree: Tree, alignment: Path, work_dir: Path, threads: int = 1
    ) -> AncestralStates:
        if not tree.ids_assigned():
            tree.assign_ids()
        work_dir = Path(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        tree_path = work_dir / "input.nwk"
        # raxml-ng refuses a multifurcating tree, and af's finished trees usually are one (zero-
        # length branches are collapsed, as the delivered trees do). The extra nodes sit on
        # zero-length branches and are dropped again by the clade matching below, which keys on
        # leaf sets. Production never hit this: it fed raxml its own binary best tree.
        binary, resolved = tree.binary_copy()
        newick.dump(binary, tree_path, with_internal_labels=True)
        prefix = work_dir / "raxml"

        command = [
            self.executable,
            "--ancestral",
            "--model",
            self.model,
            "--msa",
            str(alignment),
            "--msa-format",
            "FASTA",
            "--tree",
            str(tree_path),
            "--threads",
            str(threads),
            "--prefix",
            str(prefix),
            "--seed",
            str(self.seed),
            "--redo",
        ]
        if not self.optimise_branches:
            command += ["--opt-branches", "off"]
        started = time.monotonic()
        try:
            done = subprocess.run(command, capture_output=True, text=True, check=False)
        except OSError as error:
            raise BackendUnavailable(f"cannot run {command[0]!r}: {error}") from error
        seconds = time.monotonic() - started
        states_path = Path(f"{prefix}.raxml.ancestralStates")
        tree_out = Path(f"{prefix}.raxml.ancestralTree")
        if done.returncode != 0 or not states_path.exists() or not tree_out.exists():
            tail = "\n".join((done.stdout + done.stderr).splitlines()[-20:])
            raise BackendUnavailable(
                f"raxml-ng --ancestral failed (exit {done.returncode}); last output:\n{tail}"
            )

        by_name: dict[str, str] = {}
        with states_path.open() as handle:
            for line in handle:
                fields = line.rstrip("\n").split("\t")
                if len(fields) >= 2:
                    by_name[fields[0]] = fields[1]
        states = match_states_by_clade(tree, tree_out, by_name)
        return AncestralStates(
            nucleotides=states,
            backend=self.name,
            backend_version=self.version(),
            seconds=seconds,
            parameters={
                "model": self.model,
                "seed": self.seed,
                "branch_lengths": "optimised" if self.optimise_branches else "fixed",
                "multifurcations_resolved": resolved,
            },
        )
