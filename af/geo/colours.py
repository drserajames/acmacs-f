"""Colour geo dots by clade: one rule set, the clade workstream's colour schemes.

Today's geo maps carry a second copy of the clade rules (``conference_data.py``'s
``geographic_coloring``: aa substitutions per colour, "the later row wins"), separate
from the tables that colour trees and maps. Here a dot takes its colour from the same
:class:`af.clades.colours.ColourScheme` everything else uses, through its
:meth:`~af.clades.colours.ColourScheme.entry_for` (most specific entry wins; groups
before clades). There is no geo-specific rule table.

A preparation gets a colour only when it has one sequence (:mod:`af.serology.joins`),
that sequence has a clade assignment, and the scheme has an entry for it. Every other
case is drawn with the uncoloured style and counted by reason, so a thin map says why.
Proxy pairings (the lab paired the antigen with a related isolate's sequence) are used
by default and counted; ``use_proxies=False`` leaves them uncoloured instead.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

from af.clades.colours import ColourScheme
from af.clades.groups import GroupSet
from af.clades.nomenclature import CladeSet
from af.clades.sequence import AlignedSequence
from af.serology.joins import PreparationKey, PreparationSequence
from af.serology.query import Preparation


@dataclass(frozen=True)
class DotStyle:
    """How a dot is drawn. ``label`` is the legend text; the uncoloured style has none."""

    label: str
    colour: str | None  # None: outline only


UNCOLOURED = DotStyle(label="", colour=None)


@dataclass
class ColourCounts:
    coloured: Counter[str] = field(default_factory=Counter)  # legend label -> preparations
    uncoloured: Counter[str] = field(default_factory=Counter)  # reason -> preparations


def dot_styles(
    sequences_of: Mapping[PreparationKey, PreparationSequence],
    aligned: Callable[[str, str], AlignedSequence | None],
    scheme: ColourScheme,
    clade_set: CladeSet,
    group_set: GroupSet | None = None,
    *,
    use_proxies: bool = True,
) -> tuple[Callable[[Preparation], DotStyle], ColourCounts]:
    """A function giving each preparation its style, and the counts it keeps as it goes.

    ``aligned(epi_isl, accession)`` returns the sequence the groups are tested against
    (from the sequence store); None means the sequence store has no aligned sequence.
    """
    counts = ColourCounts()
    cache: dict[PreparationKey, DotStyle] = {}

    def style(prep: Preparation) -> DotStyle:
        key = (prep.subtype, prep.name, prep.reassortant, prep.annotations, prep.passage)
        if key not in cache:
            cache[key], reason = _style(key)
            if reason:
                counts.uncoloured[reason] += 1
            else:
                counts.coloured[cache[key].label] += 1
        return cache[key]

    def _style(key: PreparationKey) -> tuple[DotStyle, str]:
        linked = sequences_of.get(key)
        if linked is None:
            return UNCOLOURED, "no sequence"
        if linked.conflict:
            return UNCOLOURED, "rows name different sequences"
        if linked.pairing == "proxy" and not use_proxies:
            return UNCOLOURED, "proxy pairing not used"
        if linked.clade is None:
            return UNCOLOURED, "no clade assignment"
        if linked.clade == "":
            return UNCOLOURED, "nomenclature names no clade"
        assert linked.epi_isl is not None and linked.accession is not None
        sequence = aligned(linked.epi_isl, linked.accession)
        if sequence is None:
            return UNCOLOURED, "no aligned sequence"
        entry = scheme.entry_for(linked.clade, sequence, clade_set, group_set)
        if entry is None:
            return UNCOLOURED, "not in the colour scheme"
        return DotStyle(label=entry.legend, colour=entry.colour), ""

    return style, counts
