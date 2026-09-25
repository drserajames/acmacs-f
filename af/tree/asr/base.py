"""The swappable ancestral-reconstruction interface.

Sarah's requirement (INVENTORY §0): raxml-ng is very slow, find something better, and make the
algorithm easy to swap. So every backend implements :class:`Backend` and is chosen from config —
never from an environment variable (design rule 4).

Two properties of a backend are not preferences but facts about what it can represent, and af
records them so a wrong choice fails loudly instead of quietly producing the wrong science:

``reconstructs_gaps``
    Whether the backend can put a deletion at an internal node. raxml-ng and IQ-TREE cannot: their
    output has A/C/G/T only. B/Vic's C-lineage clades are *defined* by the HA1 163/164 deletions,
    so a gap-blind backend silently loses them. Measured: TreeTime gives the full deletion at 6,393
    of 6,421 B/Vic internal nodes, raxml-ng at none (notes/trees/ASR.md).

``optimises_branch_lengths``
    Whether the backend rewrites the branch lengths it was given. The topology *and* the lengths
    come from CMAPLE, so re-optimising them is wasted work and a departure from the tree that was
    built. raxml-ng ``--ancestral`` does it by default; af turns it off everywhere.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

from af.tree.model import Tree

# A codon of gaps is a real deletion; a part-gapped codon is unknown.
_GAP_CODON = "-"
_UNKNOWN_AA = "X"

CODON_TABLE = {
    "TTT": "F",
    "TTC": "F",
    "TTA": "L",
    "TTG": "L",
    "CTT": "L",
    "CTC": "L",
    "CTA": "L",
    "CTG": "L",
    "ATT": "I",
    "ATC": "I",
    "ATA": "I",
    "ATG": "M",
    "GTT": "V",
    "GTC": "V",
    "GTA": "V",
    "GTG": "V",
    "TCT": "S",
    "TCC": "S",
    "TCA": "S",
    "TCG": "S",
    "CCT": "P",
    "CCC": "P",
    "CCA": "P",
    "CCG": "P",
    "ACT": "T",
    "ACC": "T",
    "ACA": "T",
    "ACG": "T",
    "GCT": "A",
    "GCC": "A",
    "GCA": "A",
    "GCG": "A",
    "TAT": "Y",
    "TAC": "Y",
    "TAA": "*",
    "TAG": "*",
    "CAT": "H",
    "CAC": "H",
    "CAA": "Q",
    "CAG": "Q",
    "AAT": "N",
    "AAC": "N",
    "AAA": "K",
    "AAG": "K",
    "GAT": "D",
    "GAC": "D",
    "GAA": "E",
    "GAG": "E",
    "TGT": "C",
    "TGC": "C",
    "TGA": "*",
    "TGG": "W",
    "CGT": "R",
    "CGC": "R",
    "CGA": "R",
    "CGG": "R",
    "AGT": "S",
    "AGC": "S",
    "AGA": "R",
    "AGG": "R",
    "GGT": "G",
    "GGC": "G",
    "GGA": "G",
    "GGG": "G",
}


def translate(nucleotides: str) -> str:
    """Translate codon by codon, keeping deletions.

    Biopython's ``translate`` turns ``---`` into ``X`` and destroys the deletion — which is exactly
    the signal B/Vic's clades are defined by, and it does it silently. A whole deleted codon
    becomes ``-``; a codon with any other ambiguity becomes ``X``.
    """
    out: list[str] = []
    for start in range(0, len(nucleotides) - 2, 3):
        codon = nucleotides[start : start + 3].upper()
        if codon == "---":
            out.append(_GAP_CODON)
        else:
            out.append(CODON_TABLE.get(codon, _UNKNOWN_AA))
    return "".join(out)


@dataclass(frozen=True)
class Substitution:
    """One amino-acid (or nucleotide) change on a branch. 1-based position."""

    position: int
    from_state: str
    to_state: str

    def __str__(self) -> str:
        return f"{self.from_state}{self.position}{self.to_state}"


@dataclass(frozen=True)
class AncestralStates:
    """What every backend returns: a state per internal node, keyed by stable node id.

    ``nucleotides`` maps node id -> aligned nucleotide sequence. ``aa`` is derived, not supplied by
    the backend, so every backend's protein is translated the same way (including deletions).
    """

    nucleotides: dict[int, str]
    backend: str
    backend_version: str
    seconds: float
    parameters: Mapping[str, object] = field(default_factory=dict)

    def aa(self) -> dict[int, str]:
        return {node_id: translate(sequence) for node_id, sequence in self.nucleotides.items()}

    def substitutions(
        self, tree: Tree, leaf_sequences: Mapping[str, str], positions: Sequence[int] | None = None
    ) -> dict[tuple[int, int], list[Substitution]]:
        """Amino-acid substitutions on every branch, keyed by (parent id, child id).

        Leaves are included, using their observed sequence. A position where either end is unknown
        (``X``) yields no substitution: an ambiguity is not a change.
        """
        wanted = set(positions) if positions else None
        proteins = self.aa()
        out: dict[tuple[int, int], list[Substitution]] = {}
        for node in tree.preorder():
            parent = node.parent
            if parent is None:
                continue
            above = proteins.get(parent.node_id)
            if above is None:
                continue
            if node.is_leaf:
                raw = leaf_sequences.get(node.name or "")
                below = translate(raw) if raw is not None else None
            else:
                below = proteins.get(node.node_id)
            if below is None:
                continue
            changes = [
                Substitution(index + 1, first, second)
                for index, (first, second) in enumerate(zip(above, below, strict=False))
                if first != second
                and first != _UNKNOWN_AA
                and second != _UNKNOWN_AA
                and (wanted is None or index + 1 in wanted)
            ]
            if changes:
                out[(parent.node_id, node.node_id)] = changes
        return out


@runtime_checkable
class Backend(Protocol):
    """One ancestral-reconstruction method."""

    name: str
    reconstructs_gaps: bool
    optimises_branch_lengths: bool

    def version(self) -> str:
        """The tool's own version string, for the provenance."""

    def reconstruct(
        self, tree: Tree, alignment: Path, work_dir: Path, threads: int = 1
    ) -> AncestralStates:
        """Reconstruct states on the given topology, leaving its branch lengths alone."""


class BackendUnavailable(RuntimeError):
    """The backend's external tool is not installed or not runnable."""


class GapCapabilityError(ValueError):
    """A gap-blind backend was chosen for a subtype whose clades are defined by deletions."""


def check_gap_capability(backend: Backend, deletion_defined_positions: Sequence[int]) -> None:
    """Refuse a gap-blind backend where deletions define clades.

    Recorded as a constraint on the swap, not a preference: B/Vic has 6 defining mutations whose
    state is '-', H3 has none, and a backend that cannot represent a gap will silently mislabel
    them (notes/trees/ASR.md).
    """
    if deletion_defined_positions and not backend.reconstructs_gaps:
        raise GapCapabilityError(
            f"backend {backend.name!r} cannot reconstruct deletions, but "
            f"{len(deletion_defined_positions)} clade-defining mutations are deletions "
            f"(positions {sorted(deletion_defined_positions)[:5]}...). Those clades would be "
            "silently lost. Use a gap-capable backend (treetime) for this subtype."
        )
