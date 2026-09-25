"""Fitch parsimony, in pure Python over bit masks.

The cheapest backend by a wide margin (measured: 1.3 s of reconstruction on a 93,676-leaf H3 tree
against TreeTime's 8 minutes), and the least accurate: it invents about twice as many branch
substitutions as the ML backends. So it is not the default for the figure's aa-transition labels,
but it is the right tool for a single-codon question such as the trim step's 223V / 127T, and for a
cheap sanity check before paying for ML.

Unlike raxml-ng and IQ-TREE this backend *can* carry deletions: '-' is a fifth state, so a clade
defined by a deletion is representable.
"""

from __future__ import annotations

import time
from pathlib import Path

from af.tree.asr.base import AncestralStates
from af.tree.io.fasta import read_alignment
from af.tree.model import Tree, TreeError

# Bit per state; '-' is a state of its own, everything unknown is "any".
_BITS = {"A": 1, "C": 2, "G": 4, "T": 8, "U": 8, "-": 16}
_ANY = 31
_DECODE = {1: "A", 2: "C", 4: "G", 8: "T", 16: "-"}


def _encode(sequence: str) -> list[int]:
    return [_BITS.get(char.upper(), _ANY) for char in sequence]


def _resolve(mask: int, preferred: int) -> int:
    """One state from a mask: keep the parent's if allowed, else the lowest bit (A<C<G<T<-)."""
    if mask & preferred:
        return preferred
    return mask & -mask


class ParsimonyBackend:
    """Fitch parsimony. Deterministic: ties resolve towards the parent, then A<C<G<T<-."""

    name = "parsimony"
    reconstructs_gaps = True
    optimises_branch_lengths = False

    def version(self) -> str:
        from af import __version__

        return f"af-fitch/{__version__}"

    def reconstruct(
        self, tree: Tree, alignment: Path, work_dir: Path, threads: int = 1
    ) -> AncestralStates:
        del work_dir, threads  # in-process; nothing to write, nothing to parallelise
        started = time.monotonic()
        sequences = read_alignment(alignment)
        if not tree.ids_assigned():
            tree.assign_ids()
        length = len(next(iter(sequences.values())))

        masks: dict[int, list[int]] = {}
        for node in tree.postorder():
            if node.is_leaf:
                observed = sequences.get(node.name or "")
                if observed is None:
                    raise TreeError(f"leaf {node.name!r} has no sequence in {alignment}")
                masks[id(node)] = _encode(observed)
                continue
            child_masks = [masks[id(child)] for child in node.children]
            combined: list[int] = []
            for site in range(length):
                intersection = _ANY
                union = 0
                for child_mask in child_masks:
                    intersection &= child_mask[site]
                    union |= child_mask[site]
                combined.append(intersection if intersection else union)
            masks[id(node)] = combined

        states: dict[int, str] = {}
        resolved: dict[int, list[int]] = {}
        for node in tree.preorder():
            if node.is_leaf:
                continue
            mask = masks[id(node)]
            parent = resolved.get(id(node.parent)) if node.parent is not None else None
            picked = [_resolve(mask[site], parent[site] if parent else 0) for site in range(length)]
            resolved[id(node)] = picked
            states[node.node_id] = "".join(_DECODE.get(bit, "N") for bit in picked)
            # A leaf's own mask is never overwritten, so drop what we no longer need.
            for child in node.children:
                if child.is_leaf:
                    masks.pop(id(child), None)

        return AncestralStates(
            nucleotides=states,
            backend=self.name,
            backend_version=self.version(),
            seconds=time.monotonic() - started,
            parameters={"ties": "parent-then-ACGT-gap", "gap_is_a_state": True},
        )
