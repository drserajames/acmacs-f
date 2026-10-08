"""What a tree-labelled virus's own sequence shows of its clade (Sarah, 8 Oct 2026, Q124/Q124c).

A tree engine labels a leaf by where it sits. A virus inside a clade's subtree carries the
clade even when its own sequence cannot show the clade's defining changes (the loci fall in an
unsequenced or ambiguous stretch) or shows something else there (a reversion). The table keeps
the tree's label in every case; this module records, per tree row, what the sequence itself
supports, so that a figure can colour by evidence rather than by placement alone:

* ``supported``: every own defining locus of the clade is observed and matches. The clade's
  colour.
* ``unobserved``: none contradicted, at least one not observable (strict: one is enough). The
  clade is not known from the sequence, so the point is drawn as an unsequenced antigen.
* ``contradicted``: at least one own locus observed with another state. The point takes the
  colour of ``supported_clade``: the deepest clade above it on its own lineage whose own loci
  are all observed and matching. ``None`` when no clade on the lineage is supported.

Only rows the tree labelled (``method = "tree"``) are judged, against the sequence version the
tree was built from. Fallback rows and rows with no clade carry no evidence state.
"""

from __future__ import annotations

import dataclasses
from collections import Counter
from collections.abc import Mapping, Sequence

from af.clades.nomenclature import CladeSet
from af.clades.sequence import AlignedSequence, Evidence
from af.clades.store import CladeRow

Key = tuple[str, str]
SUPPORTED, UNOBSERVED, CONTRADICTED = "supported", "unobserved", "contradicted"
STATES = (SUPPORTED, UNOBSERVED, CONTRADICTED)


@dataclasses.dataclass(frozen=True)
class OwnLoci:
    """How a sequence stands on one clade's own defining loci."""

    unobservable: tuple[str, ...]
    contradicted: tuple[str, ...]

    @property
    def state(self) -> str:
        if self.contradicted:
            return CONTRADICTED
        return UNOBSERVED if self.unobservable else SUPPORTED

    def reason(self) -> str | None:
        parts = []
        if self.contradicted:
            parts.append(f"own loci contradicted: {','.join(self.contradicted)}")
        if self.unobservable:
            parts.append(f"own loci unobservable: {','.join(self.unobservable)}")
        return "; ".join(parts) or None


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


def supported_ancestor(sequence: AlignedSequence, clade_set: CladeSet, clade: str) -> str | None:
    """The deepest clade above ``clade`` on its lineage whose own loci the sequence all shows."""
    for ancestor in clade_set.ancestors(clade):
        if own_loci(sequence, clade_set, ancestor).state == SUPPORTED:
            return ancestor
    return None


def tree_keys(rows: Sequence[CladeRow]) -> set[Key]:
    """The rows to judge: the tree's calls that name a clade."""
    return {
        (row.epi_isl, row.accession)
        for row in rows
        if row.method == "tree" and row.clade is not None
    }


def judge(
    rows: Sequence[CladeRow], sequences: Mapping[Key, AlignedSequence], clade_set: CladeSet
) -> list[CladeRow]:
    """``rows`` with ``clade_evidence``, ``supported_clade`` and ``clade_evidence_reason`` set.

    ``sequences`` must hold every tree row with a clade: one that cannot be judged would be
    coloured on its placement alone, which is what the evidence state exists to prevent
    (design rule 4).
    """
    wanted = tree_keys(rows)
    missing = sorted(wanted - set(sequences))
    if missing:
        raise ValueError(
            f"{len(missing)} tree rows to judge have no aligned sequence, e.g. {missing[:3]}"
        )
    judged = []
    for row in rows:
        key = (row.epi_isl, row.accession)
        if key in wanted:
            assert row.clade is not None
            sequence = sequences[key]
            loci = own_loci(sequence, clade_set, row.clade)
            row = dataclasses.replace(
                row,
                clade_evidence=loci.state,
                supported_clade=(
                    supported_ancestor(sequence, clade_set, row.clade)
                    if loci.state == CONTRADICTED
                    else None
                ),
                clade_evidence_reason=loci.reason(),
            )
        judged.append(row)
    return judged


def summary(rows: Sequence[CladeRow]) -> dict[str, object]:
    """Counts for the report: rows per state, and where contradicted rows take their colour."""
    states = Counter(row.clade_evidence for row in rows if row.clade_evidence)
    moves = Counter(
        f"{row.clade} -> {row.supported_clade or 'none supported'}"
        for row in rows
        if row.clade_evidence == CONTRADICTED
    )
    return {
        **{state: states.get(state, 0) for state in STATES},
        "contradicted_without_supported_clade": sum(
            1 for row in rows if row.clade_evidence == CONTRADICTED and row.supported_clade is None
        ),
        "contradicted_colour_from": dict(moves.most_common()),
    }
