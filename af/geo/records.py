"""Geo records: how many dots each location gets in each month.

Production rule (Sarah, 7 Oct 2026: "One dot per virus name - as it is meant to show the number
of isolates (i.e. merge egg & cell)"): **one dot per virus**, :data:`DotRule.VIRUS`. A virus is
its table subtype, name, reassortant and annotations; its preparations (passages, egg or cell)
are one dot, in the month of the earliest collection date any of them reports, at the location
of its name. Its colour (Sarah, 7 Oct): the colour its coloured preparations agree on; where
they disagree, the colour of its cell or original preparations; where those disagree too (or
it has none), uncoloured. Every case is counted, per subtype (:attr:`GeoCounts.merges`).

:data:`DotRule.PREPARATION`, one dot per preparation (name, reassortant, annotations,
passage), is what the shipped round drew (ae), and stays for comparison with it.

Where a location comes from is the caller's business (workstream 2's lookup), passed in as
a function, so this module holds no location data. A preparation with no collection date
or no resolvable location cannot be drawn; both are counted in the result, never dropped
silently.
"""

from __future__ import annotations

import calendar
import datetime
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from af.geo.colours import STATE_NO_SCHEME_ROW, STATE_NO_SUPPORTED, UNCOLOURED, DotStyle
from af.serology.query import Preparation


class DotRule(Enum):
    PREPARATION = "preparation"
    VIRUS = "virus"


# How a virus's dot got its colour from its preparations' (GeoCounts.merges keys)
SINGLE = "one preparation"
AGREE = "preparations agree"
COLOURED_OVER_UNCOLOURED = "coloured preparations agree, others uncoloured"
CELL_CHOSEN = "preparations disagree: cell or original preparation's colour"
DISAGREE = "preparations disagree"  # uncoloured
CELL_CLASSES = frozenset({"cell", "original"})

VirusKey = tuple[str, str, str, tuple[str, ...]]

# Uncoloured because of tree-clade evidence and the scheme, not for want of a sequence: each such
# drawn dot is warned about (Sarah, 8 Oct 2026; af.geo.colours.unpainted_warning)
UNPAINTED_STATES = frozenset({STATE_NO_SCHEME_ROW, STATE_NO_SUPPORTED})


@dataclass(frozen=True)
class Month:
    year: int
    month: int

    @classmethod
    def of(cls, date: datetime.date) -> Month:
        return cls(date.year, date.month)

    def __str__(self) -> str:
        return f"{self.year:04d}-{self.month:02d}"

    def next(self) -> Month:
        return Month(self.year + self.month // 12, self.month % 12 + 1)


def months(first: Month, last: Month) -> list[Month]:
    """Every month from ``first`` to ``last`` inclusive; empty months still get a map."""
    if (last.year, last.month) < (first.year, first.month):
        raise ValueError(f"month range ends before it starts: {first}..{last}")
    out = [first]
    while out[-1] != last:
        out.append(out[-1].next())
    return out


@dataclass
class GeoCounts:
    """Dots per (subtype, month, location), plus what could not be placed."""

    dots: Counter[tuple[str, Month, str, DotStyle]] = field(default_factory=Counter)
    months: list[Month] = field(default_factory=list)
    undated: Counter[str] = field(default_factory=Counter)  # subtype -> preparations (or viruses)
    no_location: Counter[tuple[str, str]] = field(default_factory=Counter)  # (subtype, name)
    # DotRule.VIRUS: subtype -> how each drawn virus's colour was decided -> viruses
    merges: dict[str, Counter[str]] = field(default_factory=dict)
    merged_preparations: Counter[str] = field(default_factory=Counter)  # subtype -> drawn preps
    # drawn dots left unpainted by clade evidence: (subtype, month, supported clade or None for
    # "no supported clade") -> Unpainted
    unpainted: dict[tuple[str, Month, str | None], Unpainted] = field(default_factory=dict)


@dataclass
class Unpainted:
    """Drawn dots left unpainted for one supported clade (None: their sequences support none).

    ``points`` are dots (viruses under :data:`DotRule.VIRUS`); ``preparations`` are those dots'
    preparations whose own sequence left them so; ``tree_clades`` counts the tree's label of
    each dot (a virus's first such preparation's, by passage order).
    """

    points: int = 0
    preparations: int = 0
    tree_clades: Counter[str] = field(default_factory=Counter)


def geo_counts(
    preparations: Iterable[Preparation],
    first: Month,
    last: Month,
    location_of: Callable[[str], str | None],
    rule: DotRule = DotRule.PREPARATION,
    style_of: Callable[[Preparation], DotStyle] | None = None,
    class_of: Callable[[Preparation], str] | None = None,
) -> GeoCounts:
    """Count dots for each month in ``first..last`` under ``rule``.

    ``style_of`` colours each preparation (:func:`af.geo.colours.dot_styles`); without it
    every dot is drawn uncoloured. ``class_of`` gives a preparation's passage class ("egg",
    "cell", "original", ...); :data:`DotRule.VIRUS` needs it to settle a disagreement.
    """
    window = months(first, last)
    result = GeoCounts(months=window)
    style = style_of if style_of is not None else (lambda _: UNCOLOURED)
    if rule is DotRule.PREPARATION:
        for prep in preparations:
            dot = style(prep)
            if _place(result, prep.subtype, prep.name, prep.collection_date, dot, location_of):
                _note_unpainted(result, prep.subtype, prep.collection_date, dot, [dot])
        return result
    if rule is not DotRule.VIRUS:
        raise NotImplementedError(rule)
    if class_of is None:
        raise ValueError("DotRule.VIRUS needs class_of (each preparation's passage class)")
    viruses: dict[VirusKey, list[Preparation]] = {}
    for prep in preparations:
        viruses.setdefault(virus_key(prep), []).append(prep)
    for (subtype, name, _, _), preps in viruses.items():
        dates = [p.collection_date for p in preps if p.collection_date is not None]
        first_date = min(dates) if dates else None
        styles = [(style(p), class_of(p)) for p in preps]
        dot, case = merged_style(styles)
        if _place(result, subtype, name, first_date, dot, location_of):
            result.merges.setdefault(subtype, Counter())[case] += 1
            result.merged_preparations[subtype] += len(preps)
            _note_unpainted(result, subtype, first_date, dot, [s for s, _ in styles])
    return result


def _note_unpainted(
    result: GeoCounts,
    subtype: str,
    date: datetime.date | None,
    dot: DotStyle,
    styles: list[DotStyle],
) -> None:
    """Count a drawn dot that is uncoloured because some of its preparations' clades could not
    be painted (and none was coloured). A dot whose preparations disagree on a colour is not
    one: it was paintable. Its clade is its first such preparation's; preparations of one
    virus supporting different clades are rare enough to be counted under the first."""
    if dot.colour is not None or date is None:
        return
    gaps = [s for s in styles if s.state in UNPAINTED_STATES]
    if not gaps or any(s.colour is not None for s in styles):
        return
    first = gaps[0]
    clade = first.clade if first.state == STATE_NO_SCHEME_ROW else None
    entry = result.unpainted.setdefault((subtype, Month.of(date), clade), Unpainted())
    entry.points += 1
    entry.preparations += len(gaps)
    entry.tree_clades[first.tree_clade] += 1


def virus_key(prep: Preparation) -> VirusKey:
    """A virus: its preparations without their passage (Sarah, 7 Oct: egg and cell merge)."""
    return (prep.subtype, prep.name, prep.reassortant, prep.annotations)


def merged_style(styles: list[tuple[DotStyle, str]]) -> tuple[DotStyle, str]:
    """One virus's dot from its preparations' (style, passage class), and how it was decided."""
    if len(styles) == 1:
        return styles[0][0], SINGLE
    coloured = {s for s, _ in styles if s.colour is not None}
    if len(coloured) == 1:
        (only,) = coloured
        agree = all(s.colour is not None for s, _ in styles)
        return only, AGREE if agree else COLOURED_OVER_UNCOLOURED
    if not coloured:
        return UNCOLOURED, AGREE
    cell = {s for s, kind in styles if s.colour is not None and kind in CELL_CLASSES}
    if len(cell) == 1:
        return next(iter(cell)), CELL_CHOSEN
    return UNCOLOURED, DISAGREE


def _place(
    result: GeoCounts,
    subtype: str,
    name: str,
    date: datetime.date | None,
    style: DotStyle,
    location_of: Callable[[str], str | None],
) -> bool:
    """Add one dot if it falls in the window and can be placed; whether it was drawn."""
    if date is None:
        result.undated[subtype] += 1
        return False
    month = Month.of(date)
    if month not in result.months:
        return False
    location = location_of(name)
    if location is None:
        result.no_location[subtype, name] += 1
        return False
    result.dots[subtype, month, location, style] += 1
    return True


def to_i7(counts: GeoCounts, subtype: str) -> dict[str, Any]:
    """One subtype in the I7 ``geo`` shape (ae's ``geo/<st>-records.json``).

    periods -> locations -> point groups with a count. Every month of the window is listed,
    empty ones included, so a month with no data is still a map. Until clade colours are
    joined, points are ``"transparent"`` with no ``clade``, and ``af.report.compare.geo``
    then compares locations only. A coloured point carries its legend label as ``clade``.
    What could not be placed is listed with the document.
    """
    by_month: dict[Month, dict[str, Counter[DotStyle]]] = {m: {} for m in counts.months}
    for (s, month, location, style), n in counts.dots.items():
        if s == subtype:
            by_month[month].setdefault(location, Counter())[style] += n
    return {
        "subtype": subtype,
        "periods": [
            {
                "period": str(month),
                "title": f"{calendar.month_name[month.month]} {month.year}",
                "locations": [
                    {"name": name, "points": _points(styles)}
                    for name, styles in sorted(locations.items())
                ],
                "unpainted": _unpainted(counts, subtype, month),
            }
            for month, locations in by_month.items()
        ],
        "undated": counts.undated[subtype],
        "no_location": sorted(name for s, name in counts.no_location if s == subtype),
    }


def _unpainted(counts: GeoCounts, subtype: str, month: Month) -> list[dict[str, Any]]:
    """The month's unpainted dots per supported clade (None last), for the figure's decisions."""
    found = [(c, u) for (s, m, c), u in counts.unpainted.items() if s == subtype and m == month]
    return [
        {
            "clade": clade,
            "tree_clades": dict(sorted(u.tree_clades.items())),
            "points": u.points,
            "preparations": u.preparations,
        }
        for clade, u in sorted(found, key=lambda cu: (cu[0] is None, cu[0] or ""))
    ]


def _points(styles: Counter[DotStyle]) -> list[dict[str, Any]]:
    """Point groups at one location, the largest first (drawn at the centre). Dots that look
    the same are one group: an uncoloured dot's evidence state is not drawn."""
    looks: Counter[DotStyle] = Counter()
    for style, n in styles.items():
        looks[DotStyle(style.label, style.colour)] += n
    points = []
    for style, n in sorted(looks.items(), key=lambda kv: (-kv[1], kv[0].label)):
        point: dict[str, Any] = {"color": style.colour or "transparent", "count": n}
        if style.label:
            point["clade"] = style.label
        points.append(point)
    return points
