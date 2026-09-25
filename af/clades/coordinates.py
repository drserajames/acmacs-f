"""Convert upstream nomenclature loci to the coordinates af stores sequences in.

The upstream nomenclature states a defining mutation as a locus (``HA1``, ``HA2``,
``SigPep`` or ``nuc``) plus a position within that locus. af stores an aligned amino-acid
string over the **mature HA** (HA1 then HA2, continuous) and the matching nucleotide
string starting at the first base of the mature HA. So every comparison needs two
conversions, and getting either wrong produces a rule that quietly matches nothing:

* ``HA2 p`` is amino-acid position ``ha1_length + p``;
* ``nuc p`` is nucleotide position ``p - nuc_offset``.

**This module is the only place those conversions happen.** INVENTORY E §2.4 records two
rules in the old system that were written with the wrong offset: one was corrected in
September 2026, the other still matches nothing.

Some loci cannot be expressed at all against a mature-HA sequence: the signal peptide
precedes it, and a few nucleotide positions fall before its first base. Those are not
silently dropped — :func:`convert` returns them as :class:`Unexpressible`, and callers
report them (design rule 1). They are unobservable rather than false: a sequence that
starts at the mature HA carries no evidence either way.

Offsets were derived empirically and re-checked in September 2026 against the WHO CC
trees and against Nextclade's placements; see ``notes/clades/ENGINE-COMPARISON.md``.
B/Vic is 347, not the 346 recorded in earlier notes: of 1,242 sequences Nextclade places
in B/Vic A.1, 1,157 carry that clade's defining HA2 151K at mature position 347+151 and
none at 346+151.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Alphabet = Literal["aa", "nuc"]

#: Loci an upstream repository may use. Anything else is a hard error: a new locus name
#: means the nomenclature has grown a coordinate system af does not understand yet.
KNOWN_LOCI = frozenset({"HA1", "HA2", "SigPep", "nuc"})


@dataclass(frozen=True)
class Coordinates:
    """How one subtype's mature-HA sequence lines up with the upstream loci.

    ``ha1_length`` is the number of amino acids in HA1, which is also the offset added to
    an upstream HA2 position. ``nuc_offset`` is subtracted from an upstream nucleotide
    position to index af's mature-HA nucleotide string.
    """

    ha1_length: int
    nuc_offset: int


#: Per subtype, as used by the WHO CC trees and seqdb. Keys are af subtype names.
COORDINATES: dict[str, Coordinates] = {
    "A(H1N1)": Coordinates(ha1_length=327, nuc_offset=71),
    "A(H3N2)": Coordinates(ha1_length=329, nuc_offset=65),
    "B/Vic": Coordinates(ha1_length=347, nuc_offset=78),
}


@dataclass(frozen=True)
class Position:
    """A defining mutation expressed in af's mature-HA coordinates."""

    alphabet: Alphabet
    position: int
    state: str

    def __str__(self) -> str:
        prefix = "" if self.alphabet == "aa" else "nuc "
        return f"{prefix}{self.position}{self.state}"


@dataclass(frozen=True)
class Unexpressible:
    """An upstream locus that a mature-HA sequence cannot carry, and why."""

    locus: str
    position: int
    state: str
    reason: str

    def __str__(self) -> str:
        return f"{self.locus} {self.position}{self.state} ({self.reason})"


class UnknownLocusError(ValueError):
    """An upstream file used a locus name this module does not know."""


def coordinates_for(subtype: str) -> Coordinates:
    """Coordinates for ``subtype``; an unknown subtype is fatal (design rule 4)."""
    try:
        return COORDINATES[subtype]
    except KeyError:
        known = ", ".join(sorted(COORDINATES))
        raise KeyError(f"no clade coordinates for subtype {subtype!r}; known: {known}") from None


def convert(
    locus: str, position: int, state: str, coordinates: Coordinates
) -> Position | Unexpressible:
    """Convert one upstream locus to a mature-HA :class:`Position`.

    Returns :class:`Unexpressible` for loci outside the mature HA rather than raising,
    because such a rule is not wrong, only unobservable in the sequences af stores.
    """
    if locus not in KNOWN_LOCI:
        known = ", ".join(sorted(KNOWN_LOCI))
        raise UnknownLocusError(f"unknown locus {locus!r} at position {position}; known: {known}")
    if position < 1:
        raise ValueError(f"{locus} position must be 1-based, got {position}")
    if locus == "SigPep":
        return Unexpressible(locus, position, state, "signal peptide precedes the mature HA")
    if locus == "HA1":
        return Position("aa", position, state)
    if locus == "HA2":
        return Position("aa", coordinates.ha1_length + position, state)
    converted = position - coordinates.nuc_offset
    if converted < 1:
        return Unexpressible(locus, position, state, "nucleotide precedes the mature HA")
    return Position("nuc", converted, state)
