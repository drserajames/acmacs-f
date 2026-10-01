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
from af.serology.joins import ROWS_PASSAGE_MATCHED, PreparationKey, PreparationSequence
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
    ties: Counter[str] = field(default_factory=Counter)  # TIE_AGREES / TIE_RANKED -> preparations
    doubtful: Counter[str] = field(default_factory=Counter)  # doubt flag -> coloured preparations
    rows: Counter[str] = field(default_factory=Counter)  # ROWS_* resolution -> preparations


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
            if reason:
                counts.uncoloured[reason] += 1
            else:
                counts.coloured[cache[key].label] += 1
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
        assert linked.epi_isl is not None and linked.accession is not None
        return _sequence_style(linked.epi_isl, linked.accession, linked.clade)

    def _rows(linked: PreparationSequence) -> tuple[DotStyle, str]:
        """Rows naming different records: the passage-matched one, else only if they agree."""
        if linked.resolution == ROWS_PASSAGE_MATCHED:
            assert linked.epi_isl is not None and linked.accession is not None
            counts.rows[ROWS_PASSAGE_MATCHED] += 1
            return _sequence_style(linked.epi_isl, linked.accession, linked.clade)
        results = {_sequence_style(a.epi_isl, a.accession, a.clade) for a in linked.alternatives}
        if len(results) == 1 and linked.resolution:
            counts.rows[linked.resolution] += 1
            return results.pop()
        return UNCOLOURED, "rows name different sequences"

    def _tie(linked: PreparationSequence) -> tuple[DotStyle, str]:
        """Every tied sequence styled as if it were the match; agree, or ae's rank decides."""
        results = {_sequence_style(t.epi_isl, t.accession, t.clade) for t in linked.tied}
        if len(results) == 1:
            counts.ties[TIE_AGREES] += 1
            return results.pop()
        if linked.ranked is None:
            return UNCOLOURED, "tie with no ranked sequence"
        counts.ties[TIE_RANKED] += 1
        ranked = linked.ranked
        return _sequence_style(ranked.epi_isl, ranked.accession, ranked.clade)

    def _sequence_style(epi_isl: str, accession: str, clade: str | None) -> tuple[DotStyle, str]:
        if clade is None:
            return UNCOLOURED, "no clade assignment"
        if clade == "":
            return UNCOLOURED, "nomenclature names no clade"
        sequence = aligned(epi_isl, accession)
        if sequence is None:
            return UNCOLOURED, "no aligned sequence"
        entry = scheme.entry_for(clade, sequence, clade_set, group_set)
        if entry is None:
            return UNCOLOURED, "not in the colour scheme"
        return DotStyle(label=entry.legend, colour=entry.colour), ""

    return style, counts
