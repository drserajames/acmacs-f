"""IQ-TREE 3 ancestral reconstruction: the swappable alternative Sarah asked for.

Two flags matter, both measured (notes/trees/ASR.md):

``-blfix`` — do not touch the branch lengths. It makes IQ-TREE both faster *and* closer to the
reference (0.006 vs 0.019 aa mismatches per node on the 3,000-leaf subtree), because the lengths it
was given came from CMAPLE and re-estimating them moves the reconstruction away from the tree that
was actually built.

``GTR+G4`` rather than production's ``GTR+G+I`` — ``+I`` changed nothing at all (identical
reconstructions, same substitution sets) for ten times the runtime, almost all of it in
"thoroughly optimizing +I+G parameters from 10 start values".

Caveat recorded in the interface: IQ-TREE's ``.state`` file has p_A/p_C/p_G/p_T columns only, so it
**cannot represent a deletion**. For a subtype whose clades are defined by deletions (B/Vic) this
backend is refused — see :func:`af.tree.asr.base.check_gap_capability`. Its per-site output is also
very large (7.4 GB for the full H3 tree), so it is streamed to per-node sequences, never held.
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


class IqTreeBackend:
    """`iqtree3 --ancestral` with fixed branch lengths."""

    name = "iqtree"
    reconstructs_gaps = False
    optimises_branch_lengths = False

    def __init__(self, executable: str | Path = "iqtree3", model: str = "GTR+G4") -> None:
        self.executable = str(executable)
        self.model = model

    def version(self) -> str:
        try:
            done = subprocess.run(
                [self.executable, "--version"], capture_output=True, text=True, check=False
            )
        except OSError as error:
            raise BackendUnavailable(f"cannot run {self.executable!r}: {error}") from error
        match = re.search(r"version (\S+)", done.stdout + done.stderr)
        return f"iqtree/{match.group(1)}" if match else "iqtree/unknown"

    def reconstruct(
        self, tree: Tree, alignment: Path, work_dir: Path, threads: int = 1
    ) -> AncestralStates:
        if not tree.ids_assigned():
            tree.assign_ids()
        work_dir = Path(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        tree_path = work_dir / "input.nwk"
        newick.dump(tree, tree_path, with_internal_labels=True)
        prefix = work_dir / "iqtree"

        command = [
            self.executable,
            "-s",
            str(alignment),
            "-te",
            str(tree_path),  # fixed topology
            "-m",
            self.model,
            "--ancestral",
            "-blfix",  # and fixed branch lengths
            "-T",
            str(threads),
            "--prefix",
            str(prefix),
            "-redo",
        ]
        started = time.monotonic()
        try:
            done = subprocess.run(command, capture_output=True, text=True, check=False)
        except OSError as error:
            raise BackendUnavailable(f"cannot run {command[0]!r}: {error}") from error
        seconds = time.monotonic() - started
        state_file = prefix.with_suffix(".state")
        tree_out = prefix.with_suffix(".treefile")
        if done.returncode != 0 or not state_file.exists() or not tree_out.exists():
            tail = "\n".join((done.stdout + done.stderr).splitlines()[-20:])
            raise BackendUnavailable(
                f"iqtree --ancestral failed (exit {done.returncode}); last output:\n{tail}"
            )

        by_name = _read_state_file(state_file)
        states = match_states_by_clade(tree, tree_out, by_name)
        return AncestralStates(
            nucleotides=states,
            backend=self.name,
            backend_version=self.version(),
            seconds=seconds,
            parameters={"model": self.model, "branch_lengths": "fixed (-blfix)"},
        )


def _read_state_file(path: Path) -> dict[str, str]:
    """Stream IQ-TREE's per-site `.state` into one sequence per node.

    The file is one line per (node, site) with four posterior columns, so it reaches several GB on
    a real tree. It is read once, in order, keeping only the chosen state.
    """
    out: dict[str, list[str]] = {}
    with path.open() as handle:
        for line in handle:
            if not line or line[0] == "#":
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 3 or fields[0] == "Node":
                continue
            out.setdefault(fields[0], []).append(fields[2])
    return {name: "".join(states) for name, states in out.items()}
