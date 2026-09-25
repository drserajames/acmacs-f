"""TreeTime ancestral reconstruction: af's default.

Chosen by Sarah on measurement (notes/trees/ASR.md): on the full 93,676-leaf H3 tree it took 8
minutes against IQ-TREE's 32 and production raxml-ng's 2-4 hours, and matched IQ-TREE's accuracy
to three decimal places. It is also the only backend measured here that reconstructs deletions,
which B/Vic's clades need.

TreeTime names internal nodes itself, so states come back under its names. af writes the topology
with its own stable ids as internal labels and matches them back by **clade** — the set of leaves
below a node — rather than by name or position, so a backend that renames or re-roots is still
matched correctly, and a backend that silently drops a node is caught instead of hidden.
"""

from __future__ import annotations

import re
import subprocess
import time
from pathlib import Path

from af.tree.asr.base import AncestralStates, BackendUnavailable
from af.tree.asr.matching import match_states_by_clade
from af.tree.io import newick
from af.tree.io.fasta import iter_fasta
from af.tree.model import Tree


class TreeTimeBackend:
    """`treetime ancestral` on a fixed topology."""

    name = "treetime"
    reconstructs_gaps = True
    optimises_branch_lengths = False

    def __init__(self, executable: str | Path = "treetime", gtr: str = "infer") -> None:
        self.executable = str(executable)
        self.gtr = gtr

    def version(self) -> str:
        try:
            done = subprocess.run(
                [self.executable, "--version"], capture_output=True, text=True, check=False
            )
        except OSError as error:
            raise BackendUnavailable(f"cannot run {self.executable!r}: {error}") from error
        text = (done.stdout + done.stderr).strip()
        match = re.search(r"\d+\.\d+\.\d+", text)
        return f"treetime/{match.group(0)}" if match else f"treetime/{text[:40] or 'unknown'}"

    def reconstruct(
        self, tree: Tree, alignment: Path, work_dir: Path, threads: int = 1
    ) -> AncestralStates:
        del threads  # TreeTime's ancestral pass is single-threaded
        if not tree.ids_assigned():
            tree.assign_ids()
        work_dir = Path(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        tree_path = work_dir / "input.nwk"
        newick.dump(tree, tree_path, with_internal_labels=True)
        out_dir = work_dir / "treetime"

        command = [
            self.executable,
            "ancestral",
            "--aln",
            str(alignment),
            "--tree",
            str(tree_path),
            "--outdir",
            str(out_dir),
            "--gtr",
            self.gtr,
            # No branch-length optimisation: the lengths come from CMAPLE and stay as they are.
            "--keep-overhangs",
        ]
        started = time.monotonic()
        try:
            done = subprocess.run(command, capture_output=True, text=True, check=False)
        except OSError as error:
            raise BackendUnavailable(f"cannot run {command[0]!r}: {error}") from error
        seconds = time.monotonic() - started
        states_path = out_dir / "ancestral_sequences.fasta"
        tree_out = out_dir / "annotated_tree.nexus"
        if done.returncode != 0 or not states_path.exists() or not tree_out.exists():
            tail = "\n".join((done.stdout + done.stderr).splitlines()[-20:])
            raise BackendUnavailable(
                f"treetime ancestral failed (exit {done.returncode}); last output:\n{tail}"
            )

        by_name = dict(iter_fasta(states_path))
        states = match_states_by_clade(tree, tree_out, by_name)
        return AncestralStates(
            nucleotides=states,
            backend=self.name,
            backend_version=self.version(),
            seconds=seconds,
            parameters={"gtr": self.gtr, "keep_overhangs": True},
        )
