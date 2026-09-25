"""Aligned sequences as the clade engine sees them, and what "unknown" means.

Two distinctions here decide whether a clade rule is tested correctly, and both have
already produced wrong clade assignments when muddled (``notes/clades/ENGINE-COMPARISON.md``):

* **Unknown depends on the alphabet.** ``X`` is an unknown amino acid, ``N`` an unknown
  base — but ``N`` is *asparagine*, a perfectly ordinary amino acid, and treating it as
  unknown makes every clade defined by an N pass on sequences that do not have it. That
  mislabelled 2,179 H3 leaves in a first run of the engine.
* **A gap is a state, not an absence.** Several B/Vic clades are defined by the HA1
  163/164 deletion. In an observed sequence ``-`` is evidence of that deletion. In an
  *ancestral reconstruction* it depends on the backend: TreeTime reconstructs gaps,
  raxml-ng and IQ-TREE do not, so for those a deletion locus is unobservable rather than
  absent (:class:`GapSupport`).

A test of one locus therefore has three outcomes, not two: it matches, it contradicts,
or the sequence carries no evidence. Only a contradiction rules a clade out.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass

from af.clades.coordinates import Alphabet

UNKNOWN: dict[Alphabet, str] = {"aa": "X", "nuc": "N"}
GAP = "-"

#: Codons translate to one amino acid; a wholly deleted codon is a deletion, and any
#: codon that is part gap or part ambiguity is unknown.
_BASES = "TCAG"
_AMINO_ACIDS = "FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG"
CODON_TABLE: dict[str, str] = {
    first + second + third: _AMINO_ACIDS[16 * i + 4 * j + k]
    for i, first in enumerate(_BASES)
    for j, second in enumerate(_BASES)
    for k, third in enumerate(_BASES)
}


class GapSupport(enum.Enum):
    """Whether the source of a sequence can represent a deletion at all."""

    #: An observed sequence, or a reconstruction that models indels (TreeTime).
    OBSERVED = "observed"
    #: A reconstruction that never emits a gap (raxml-ng, IQ-TREE): deletion loci are
    #: unobservable, and testing them would rule out clades that are in fact present.
    GAP_BLIND = "gap-blind"


class Evidence(enum.Enum):
    """The result of testing one locus against a sequence."""

    MATCHES = "matches"
    CONTRADICTS = "contradicts"
    UNOBSERVABLE = "unobservable"


def translate(nucleotides: str) -> str:
    """Translate codon-wise, keeping deletions.

    ``---`` becomes ``-`` and any other codon containing a gap or an ambiguity becomes
    ``X``. The obvious alternative (``Bio.Seq.translate``) turns ``---`` into ``X``,
    which silently destroys exactly the signal the B/Vic C-lineage clades are defined by.
    """
    return "".join(
        GAP if (codon := nucleotides[index : index + 3]) == "---" else CODON_TABLE.get(codon, "X")
        for index in range(0, len(nucleotides) - 2, 3)
    )


@dataclass(frozen=True)
class AlignedSequence:
    """One virus's aligned mature HA, as amino acids and optionally nucleotides.

    ``gaps`` says whether a ``-`` in these strings means "deleted" or "the source cannot
    say". It is required rather than defaulted: the answer depends on where the sequence
    came from, and guessing it wrong changes clade assignments.
    """

    amino_acids: str
    nucleotides: str | None = None
    gaps: GapSupport = GapSupport.OBSERVED

    @classmethod
    def from_nucleotides(
        cls, nucleotides: str, *, gaps: GapSupport = GapSupport.OBSERVED
    ) -> AlignedSequence:
        return cls(amino_acids=translate(nucleotides), nucleotides=nucleotides, gaps=gaps)

    def evidence(self, alphabet: Alphabet, position: int, state: str) -> Evidence:
        """Test one locus: does this sequence match ``state`` at ``position``?"""
        sequence = self.amino_acids if alphabet == "aa" else self.nucleotides
        if sequence is None or position > len(sequence) or position < 1:
            return Evidence.UNOBSERVABLE
        observed = sequence[position - 1]
        if observed == UNKNOWN[alphabet]:
            return Evidence.UNOBSERVABLE
        if self.gaps is GapSupport.GAP_BLIND and (state == GAP or observed == GAP):
            return Evidence.UNOBSERVABLE
        return Evidence.MATCHES if observed == state else Evidence.CONTRADICTS
