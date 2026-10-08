"""Colour geo dots by clade: one rule set, the clade workstream's colour schemes.

Today's geo maps carry a second copy of the clade rules (``conference_data.py``'s
``geographic_coloring``: aa substitutions per colour, "the later row wins"), separate
from the tables that colour trees and maps. Here a dot takes its colour from the same
:class:`af.clades.colours.ColourScheme` everything else uses, through its
:meth:`~af.clades.colours.ColourScheme.entry_for` (the last matching row wins, as in the
round's tables; Sarah, Q80). There is no geo-specific rule table.

A preparation gets a colour only when it has one sequence (:mod:`af.serology.joins`),
that sequence has a clade assignment, and the scheme has an entry for it. Every other
case is drawn with the uncoloured style and counted by reason, so a thin map says why.
Proxy pairings (the lab paired the antigen with a related isolate's sequence) are used
by default and counted; ``use_proxies=False`` leaves them uncoloured instead.

A preparation whose name found several different sequences (a refused tie, no sequence
chosen) is coloured when every tied sequence gives the same style, and otherwise by the one
ae's rank takes (Sarah, Q81: "Agree + ae's rank for splits"). Both are counted in
``ColourCounts.ties``, so a figure says how many of its dots came from ties. Dots coloured
through a doubtful match ae uses (egg antigen with only a cell sequence, reassortant, lab
EPI_ISL whose name differs; Sarah, Q81 D) are counted per doubt in ``ColourCounts.doubtful``.

A preparation whose table rows name two GISAID records of one virus is coloured by the record
whose passage matches its own; when none or several match, by the records only if they agree
(Sarah, Q81, 30 Sep: "Passage-matched record"). Counted in ``ColourCounts.rows``. Passage stays
part of antigen identity: egg and cell preparations are never merged; this chooses only among
the records one preparation's rows already name.

A sequence the nomenclature names no clade for can still be coloured by a GROUP that needs no
clade (one with an empty anchor: substitutions alone), because the scheme's own rows may say so
(Sarah, 2 Oct 2026: "af will need to fix this going forward"). No clade row and no anchored
group can match such a virus, and the scheme's row order still decides (Q80). Each coloured dot
records its ``basis``: by its clade, or by a group with no clade named.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

from af.clades.colours import ColourScheme
from af.clades.groups import GroupSet
from af.clades.nomenclature import CladeSet
from af.clades.sequence import AlignedSequence
from af.seq.matching import TIE_AGREES, TIE_RANKED
from af.serology.joins import (
    ROWS_PASSAGE_MATCHED,
    PreparationKey,
    PreparationSequence,
    TiedSequence,
)
from af.serology.query import Preparation

#: How a coloured dot got its colour: the virus's clade (any row matched with a clade named) ...
BASIS_CLADE = "clade"
#: ... or a group needing no clade, for a sequence the nomenclature names no clade for.
BASIS_GROUP_NO_CLADE = "group, no clade named"
#: ... or the deepest clade its sequence supports, where the tree's clade is contradicted by
#: the sequence at that clade's own loci (Sarah, 8 Oct 2026: clade_evidence "contradicted").
BASIS_SUPPORTED_CLADE = "supported clade, tree clade contradicted"

# Why an uncoloured dot is uncoloured, beyond "no sequence" (states carried on the DotStyle so
# the maps can act on them; reasons counted in ColourCounts.uncoloured). Tree-derived clades
# only: 04-clades' clade_evidence on method="tree" rows (Sarah, 8 Oct 2026).
#: The tree's clade cannot be shown from the sequence (an own locus unobserved, strict): the
#: clade is not known, so the dot is drawn as an unsequenced one.
STATE_UNOBSERVED = "clade not observable in its sequence"
#: The tree's clade is contradicted and its sequence supports no clade it descends from.
STATE_NO_SUPPORTED = "contradicted, no supported clade"
#: The supported clade has no row in the colour scheme: a scheme gap, reported as a warning.
STATE_NO_SCHEME_ROW = "supported clade not in the colour scheme"


@dataclass(frozen=True)
class DotStyle:
    """How a dot is drawn. ``label`` is the legend text; the uncoloured style has none.

    ``basis`` says how a coloured dot got its colour (:data:`BASIS_CLADE` or
    :data:`BASIS_GROUP_NO_CLADE`); it is part of equality, so tied sequences reaching one colour
    by different routes do not count as agreeing.
    """

    label: str
    colour: str | None  # None: outline only
    basis: str = ""
    # for an uncoloured dot, a STATE_* when clade evidence left it so; with STATE_NO_SCHEME_ROW,
    # ``clade`` names the supported clade the scheme has no row for
    state: str = ""
    clade: str = ""
    tree_clade: str = ""  # with STATE_NO_SCHEME_ROW / STATE_NO_SUPPORTED: the tree's label


UNCOLOURED = DotStyle(label="", colour=None)


@dataclass
class ColourCounts:
    coloured: Counter[str] = field(default_factory=Counter)  # legend label -> preparations
    uncoloured: Counter[str] = field(default_factory=Counter)  # reason -> preparations
    ties: Counter[str] = field(default_factory=Counter)  # TIE_AGREES / TIE_RANKED -> preparations
    doubtful: Counter[str] = field(default_factory=Counter)  # doubt flag -> coloured preparations
    rows: Counter[str] = field(default_factory=Counter)  # ROWS_* resolution -> preparations
    basis: Counter[str] = field(default_factory=Counter)  # BASIS_* -> coloured preparations
    # clade_evidence of each preparation's matched record ("supported", "unobserved",
    # "contradicted"); tree-derived records only
    evidence: Counter[str] = field(default_factory=Counter)
    # supported clade -> preparations left unpainted because the scheme has no row for it
    unpainted_clades: Counter[str] = field(default_factory=Counter)


def dot_styles(
    sequences_of: Mapping[PreparationKey, PreparationSequence],
    aligned: Callable[[str, str], AlignedSequence | None],
    scheme: ColourScheme,
    clade_set: CladeSet,
    group_set: GroupSet | None = None,
    *,
    use_proxies: bool = True,
) -> tuple[Callable[[Preparation | PreparationKey], DotStyle], ColourCounts]:
    """A function giving each preparation its style, and the counts it keeps as it goes.

    The function takes a serology :class:`~af.serology.query.Preparation` (geo) or its key
    directly (:func:`af.serology.joins.preparation_key`, the maps): one key, one copy.

    ``aligned(epi_isl, accession)`` returns the sequence the groups are tested against
    (from the sequence store); None means the sequence store has no aligned sequence.
    """
    counts = ColourCounts()
    cache: dict[PreparationKey, DotStyle] = {}

    def style(prep: Preparation | PreparationKey) -> DotStyle:
        key = prep if isinstance(prep, tuple) else prep.key()
        if key not in cache:
            cache[key], reason = _style(key)
            linked = sequences_of.get(key)
            if linked is not None and linked.clade_evidence:
                counts.evidence[linked.clade_evidence] += 1
            if cache[key].state == STATE_NO_SCHEME_ROW:
                counts.unpainted_clades[cache[key].clade] += 1
            if reason:
                counts.uncoloured[reason] += 1
            else:
                counts.coloured[cache[key].label] += 1
                counts.basis[cache[key].basis] += 1
                linked = sequences_of.get(key)
                counts.doubtful.update(linked.doubts if linked is not None else ())
        return cache[key]

    def _style(key: PreparationKey) -> tuple[DotStyle, str]:
        linked = sequences_of.get(key)
        if linked is None:
            return UNCOLOURED, "no sequence"
        if linked.conflict:
            return _rows(linked)
        if linked.pairing == "proxy" and not use_proxies:
            return UNCOLOURED, "proxy pairing not used"
        if linked.tied:
            return _tie(linked)
        return _sequence_style(linked)

    def _rows(linked: PreparationSequence) -> tuple[DotStyle, str]:
        """Rows naming different records: the passage-matched one, else only if they agree."""
        if linked.resolution == ROWS_PASSAGE_MATCHED:
            counts.rows[ROWS_PASSAGE_MATCHED] += 1
            return _sequence_style(linked)
        results = {_sequence_style(a) for a in linked.alternatives}
        if len(results) == 1 and linked.resolution:
            counts.rows[linked.resolution] += 1
            return results.pop()
        return UNCOLOURED, "rows name different sequences"

    def _tie(linked: PreparationSequence) -> tuple[DotStyle, str]:
        """Every tied sequence styled as if it were the match; agree, or ae's rank decides."""
        results = {_sequence_style(t) for t in linked.tied}
        if len(results) == 1:
            counts.ties[TIE_AGREES] += 1
            return results.pop()
        if linked.ranked is None:
            return UNCOLOURED, "tie with no ranked sequence"
        counts.ties[TIE_RANKED] += 1
        return _sequence_style(linked.ranked)

    def _sequence_style(record: PreparationSequence | TiedSequence) -> tuple[DotStyle, str]:
        """One record's style. Its clade evidence (tree-derived clades) decides which clade
        paints it: unobserved, none; contradicted, the deepest supported clade; else its own.
        The evidence is 04-clades' and is read, never recomputed here."""
        epi_isl, accession, clade = record.epi_isl, record.accession, record.clade
        assert epi_isl is not None and accession is not None
        basis, tree_clade = BASIS_CLADE, ""
        if record.clade_evidence == "unobserved":
            return DotStyle("", None, state=STATE_UNOBSERVED), STATE_UNOBSERVED
        if record.clade_evidence == "contradicted":
            if not record.supported_clade:
                none = DotStyle("", None, state=STATE_NO_SUPPORTED, tree_clade=clade or "")
                return none, STATE_NO_SUPPORTED
            tree_clade, clade, basis = clade or "", record.supported_clade, BASIS_SUPPORTED_CLADE
        if clade is None:
            return UNCOLOURED, "no clade assignment"
        if clade == "":
            # Known to have no clade: only a group needing none can colour it. The reason for
            # one that stays uncoloured is unchanged, so earlier counts stay comparable.
            no_clade = aligned(epi_isl, accession)
            found = scheme.entry_for(None, no_clade, clade_set, group_set) if no_clade else None
            if found is None:
                return UNCOLOURED, "nomenclature names no clade"
            return DotStyle(found.legend, found.colour, BASIS_GROUP_NO_CLADE), ""
        sequence = aligned(epi_isl, accession)
        if sequence is None:
            return UNCOLOURED, "no aligned sequence"
        entry = scheme.entry_for(clade, sequence, clade_set, group_set)
        if entry is None:
            if basis == BASIS_SUPPORTED_CLADE:
                gap = DotStyle(
                    "", None, state=STATE_NO_SCHEME_ROW, clade=clade, tree_clade=tree_clade
                )
                return gap, STATE_NO_SCHEME_ROW
            return UNCOLOURED, "not in the colour scheme"
        return DotStyle(entry.legend, entry.colour, basis), ""

    return style, counts
