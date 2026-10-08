"""Mark tree calls that rest on ancestry alone (Sarah, 8 Oct 2026, Q124/Q124b).

A tree engine labels a leaf by where it sits: a virus inside a clade's subtree carries the
clade even when its own sequence cannot show the clade's defining changes (the loci fall in an
unsequenced stretch) or shows something else there (a reversion). That is a claim about
ancestry, not about the virus's sequence. Sarah's ruling: publish such calls, record them as
ancestry only, and do not let them colour a point on a map.

The mark is general, never a hand list. A tree row is ``ancestry_only`` when both hold:

* the tree's clade differs from the fallback's (Nextclade's) call for the same sequence, so
  the sequence-only engine did not reach it; and
* not every one of that clade's **own** defining loci is observed and matching in the row's
  sequence.

Rows where the two engines agree are not marked: the sequence alone already supports the call.
Fallback rows are never marked. ``ancestry_reason`` names the own loci that were unobservable
and those that were contradicted, so a reader sees why without re-running anything.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence

from af.clades.nomenclature import CladeSet
from af.clades.sequence import AlignedSequence, Evidence
from af.clades.store import CladeRow

Key = tuple[str, str]


@dataclasses.dataclass(frozen=True)
class OwnLoci:
    """How a sequence stands on one clade's own defining loci."""

    unobservable: tuple[str, ...]
    contradicted: tuple[str, ...]

    @property
    def supported(self) -> bool:
        return not self.unobservable and not self.contradicted

    def reason(self) -> str:
        parts = []
        if self.contradicted:
            parts.append(f"own loci contradicted: {','.join(self.contradicted)}")
        if self.unobservable:
            parts.append(f"own loci unobservable: {','.join(self.unobservable)}")
        return "; ".join(parts)


def own_loci(sequence: AlignedSequence, clade_set: CladeSet, clade: str) -> OwnLoci:
    """The clade's own loci (its branch, not its ancestors') the sequence does not show."""
    unobservable, contradicted = [], []
    for locus in clade_set[clade].mutations:
        evidence = sequence.evidence(locus.alphabet, locus.position, locus.state)
        token = f"{locus.position}{locus.state}"
        token = token if locus.alphabet == "aa" else f"nuc{token}"
        if evidence is Evidence.CONTRADICTS:
            contradicted.append(token)
        elif evidence is not Evidence.MATCHES:
            unobservable.append(token)
    return OwnLoci(tuple(unobservable), tuple(contradicted))


def candidates(rows: Sequence[CladeRow], fallback: Mapping[Key, str | None]) -> set[Key]:
    """Tree rows naming a clade the fallback did not give: the only ones that can be marked."""
    return {
        (row.epi_isl, row.accession)
        for row in rows
        if row.method == "tree"
        and row.clade is not None
        and fallback.get((row.epi_isl, row.accession)) != row.clade
    }


def mark(
    rows: Sequence[CladeRow],
    fallback: Mapping[Key, str | None],
    sequences: Mapping[Key, AlignedSequence],
    clade_set: CladeSet,
) -> list[CladeRow]:
    """``rows`` with ``ancestry_only`` and ``ancestry_reason`` set (see the module docstring).

    ``sequences`` must hold every candidate (:func:`candidates`): a candidate whose sequence
    is missing cannot be judged, and leaving it unmarked would let it colour (design rule 4).
    """
    wanted = candidates(rows, fallback)
    missing = sorted(wanted - set(sequences))
    if missing:
        raise ValueError(
            f"{len(missing)} tree rows to judge have no aligned sequence, e.g. {missing[:3]}"
        )
    marked = []
    for row in rows:
        key = (row.epi_isl, row.accession)
        if key in wanted:
            assert row.clade is not None
            loci = own_loci(sequences[key], clade_set, row.clade)
            if not loci.supported:
                row = dataclasses.replace(row, ancestry_only=True, ancestry_reason=loci.reason())
        marked.append(row)
    return marked


def summary(rows: Sequence[CladeRow]) -> dict[str, int]:
    """Counts for the report: how many rows are marked, and by which kind of evidence."""
    flagged = [row for row in rows if row.ancestry_only]
    contradicted = sum("contradicted" in (row.ancestry_reason or "") for row in flagged)
    return {
        "rows": len(flagged),
        "with_contradicted_loci": contradicted,
        "unobservable_loci_only": len(flagged) - contradicted,
    }
