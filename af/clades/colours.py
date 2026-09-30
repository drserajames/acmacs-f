"""User-defined colour schemes: which clades and groups are drawn, and in what colour.

Colours are the user's choice, not the nomenclature's (Sarah, 24 Sep 2026), so they live
in acmacs-f-data as plain tables, one per scheme::

    colours/<subtype>/<scheme>.tsv
    order   key        legend       colour
    1       C.1        C.1          #98e1d7
    2       D          D            #5b00d3
    3       D 139N     D.3.1 139N   #00bfff

``key`` is a clade name, a group name (:mod:`af.clades.groups`), or the full name of an older
upstream clade with its own definitions (a *legacy* entry, :mod:`af.clades.legacy`, which only
matches viruses the subclades leave unnamed); ``legend`` is what the
figure prints, which is not always the key — today's tables relabel clades on the figure
while keeping the old key. ``order`` is the row order: it orders the legend **and decides
which entry colours a virus**.

**The last matching row wins** (Sarah, 29 Sep 2026, Q80: "keep the order, maps & geo
should match"). Entries are walked in ``order``; an entry matches a virus when it is a
group the virus belongs to, or a clade the virus's clade lies within; the last one that
matches colours it, and a virus no entry matches is not drawn. This is the rule the round's
tables were written for, so maps and geo draw what the round drew. A rule chosen by
specificity instead (groups first, then the deepest clade) was built first and measured
against the round on 18 maps: it repainted most of one current clade with a legacy group
that the tables deliberately list above it.

The cost is the trap this rule has always had (INVENTORY E, trap T9): a row listed above a
broader row can never win — a child clade above its parent is always drawn in the parent's
colour. That is the user's content to order, so it is not an error; but a row that can
never colour anything is reported (:func:`shadowed_entries`), so it cannot go unnoticed.
Two rows with the same ``order`` would leave the winner undefined, and are a load error.
"""

from __future__ import annotations

import csv
import re
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from af.clades.groups import Group, GroupSet
from af.clades.nomenclature import CladeSet
from af.clades.sequence import AlignedSequence

COLOUR = re.compile(r"^#[0-9a-fA-F]{6}$")
COLUMNS = ("order", "key", "legend", "colour")


class ColourSchemeError(ValueError):
    """A colour scheme is malformed, or names a clade or group that does not exist."""

    def __init__(self, source: Path | str, problems: Sequence[str]) -> None:
        self.source = source
        self.problems = list(problems)
        lines = "\n".join(f"  - {problem}" for problem in self.problems)
        super().__init__(f"invalid colour scheme {source}:\n{lines}")


@dataclass(frozen=True)
class ColourEntry:
    """One row: what to draw, how to label it, and in what colour."""

    order: int
    key: str
    legend: str
    colour: str
    is_group: bool
    is_legacy: bool = False

    def __str__(self) -> str:
        return f"{self.key} ({self.colour})"


@dataclass(frozen=True)
class ColourScheme:
    """One scheme for one subtype, in legend order."""

    subtype: str
    name: str
    entries: tuple[ColourEntry, ...]
    source: Path | None = None

    def __iter__(self) -> Iterator[ColourEntry]:
        return iter(self.entries)

    def __len__(self) -> int:
        return len(self.entries)

    @property
    def keys(self) -> tuple[str, ...]:
        return tuple(entry.key for entry in self.entries)

    def legend(self) -> tuple[tuple[str, str], ...]:
        """(label, colour) in the order the file gives, for drawing the legend."""
        return tuple((entry.legend, entry.colour) for entry in self.in_order())

    def entry_for(
        self,
        clade: str | None,
        sequence: AlignedSequence,
        clade_set: CladeSet,
        group_set: GroupSet | None = None,
        *,
        legacy_clade: str | None = None,
    ) -> ColourEntry | None:
        """The entry that colours this virus: the last matching row wins (Q80).

        A scheme with group entries needs the groups it was loaded with; without them a
        group row would silently never match, so that is an error. ``legacy_clade`` is the
        virus's retrospective label, given only when the subclades do not name it; a legacy
        row matches it or anything below it, and no clade row can then match.
        """
        if clade is not None and legacy_clade is not None:
            raise ValueError("a virus has a subclade or a legacy label, never both")
        has_groups = any(entry.is_group for entry in self.entries)
        if has_groups and group_set is None:
            raise ValueError(
                f"colour scheme {self.subtype} {self.name} has group entries; "
                "entry_for needs the group set it was loaded with"
            )
        in_groups = (
            set(group_set.matching(clade, sequence, clade_set))
            if has_groups and group_set is not None
            else set()
        )
        chosen: ColourEntry | None = None
        for entry in self.in_order():
            if entry.is_group:
                matched = entry.key in in_groups
            elif entry.is_legacy:
                matched = legacy_clade is not None and clade_set.legacy_is_within(
                    legacy_clade, entry.key
                )
            else:
                matched = clade is not None and clade_set.is_within(clade, entry.key)
            if matched:
                chosen = entry
        return chosen

    def in_order(self) -> tuple[ColourEntry, ...]:
        """Entries by ``order``: the legend's order and the order precedence is decided in."""
        return tuple(sorted(self.entries, key=lambda entry: entry.order))


def load_colour_scheme(
    path: Path,
    subtype: str,
    clade_set: CladeSet,
    group_set: GroupSet | None = None,
) -> ColourScheme:
    """Read one scheme and check every key against the clade set and the groups.

    An unknown key is an error, not an empty legend row: today a scheme may name a clade
    that no longer exists and the only symptom is a colour nobody ever sees
    (INVENTORY E §2.3 counts four such rows).
    """
    path = Path(path)
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        fields = reader.fieldnames or []
        missing = [column for column in COLUMNS if column not in fields]
        if missing:
            raise ColourSchemeError(path, [f"missing column(s): {', '.join(missing)}"])
        unknown_columns = sorted(set(fields) - set(COLUMNS))
        if unknown_columns:
            raise ColourSchemeError(path, [f"unknown column(s): {', '.join(unknown_columns)}"])
        rows = [(f"line {line}", row) for line, row in enumerate(reader, start=2)]
    return scheme_from_rows(rows, subtype, path.stem, clade_set, group_set, path)


def scheme_from_rows(
    rows: Iterable[tuple[str, Mapping[str, str | None]]],
    subtype: str,
    name: str,
    clade_set: CladeSet,
    group_set: GroupSet | None = None,
    source: Path | None = None,
) -> ColourScheme:
    """Check scheme rows, each ``(where, {order, key, legend, colour})``, and build it.

    The one place a scheme is validated, whether its rows came from a file here or were
    read at run time from the user's tables elsewhere (:mod:`af.clades.importer`), so the
    two routes cannot drift apart in what they accept.
    """
    problems: list[str] = []
    entries: list[ColourEntry] = []
    seen_keys: set[str] = set()
    seen_orders: dict[int, str] = {}
    group_names = set(group_set.names) if group_set else set()
    legacy_names = set(clade_set.legacy_defining())
    for where, row in rows:
        values = {key: (row.get(key) or "").strip() for key in COLUMNS}
        if not any(values.values()):
            continue
        key = values["key"]
        if not key:
            problems.append(f"{where}: no key")
            continue
        is_group = key in group_names
        is_legacy = not is_group and key not in clade_set and key in legacy_names
        if not is_group and not is_legacy and key not in clade_set:
            problems.append(
                f"{where}: key {key!r} is neither a clade of {subtype} at "
                f"{clade_set.version}, nor a legacy clade with its own definitions, nor a known "
                "group"
            )
            continue
        if key in seen_keys:
            problems.append(f"{where}: duplicate key {key!r}")
            continue
        seen_keys.add(key)
        if not COLOUR.fullmatch(values["colour"]):
            problems.append(f"{where}: colour {values['colour']!r} is not '#rrggbb'")
            continue
        try:
            order = int(values["order"])
        except ValueError:
            problems.append(f"{where}: order {values['order']!r} is not a number")
            continue
        if order in seen_orders:
            problems.append(f"{where}: order {order} is already used by {seen_orders[order]!r}")
            continue
        seen_orders[order] = key
        entries.append(
            ColourEntry(
                order, key, values["legend"] or key, values["colour"].lower(), is_group, is_legacy
            )
        )
    label = source if source is not None else f"{subtype} {name}"
    if not entries and not problems:
        problems.append("scheme has no entries")
    if problems:
        raise ColourSchemeError(label, problems)
    return ColourScheme(subtype, name, tuple(entries), source)


def load_colour_schemes(
    directory: Path,
    subtype: str,
    clade_set: CladeSet,
    group_set: GroupSet | None = None,
) -> dict[str, ColourScheme]:
    """Every ``*.tsv`` scheme in one subtype's directory, keyed by file stem."""
    directory = Path(directory)
    if not directory.is_dir():
        raise ColourSchemeError(directory, ["no such directory"])
    schemes = {
        path.stem: load_colour_scheme(path, subtype, clade_set, group_set)
        for path in sorted(directory.glob("*.tsv"))
    }
    if not schemes:
        raise ColourSchemeError(directory, ["no colour schemes (*.tsv) found"])
    return schemes


def unused_entries(scheme: ColourScheme, used: Mapping[str, int]) -> tuple[str, ...]:
    """Keys of a scheme that coloured nothing in a run.

    Reported, not fatal: a scheme kept across seasons will legitimately contain clades
    that have died out. But it is also what a typo looks like, so it must be visible
    (design rule 1).
    """
    return tuple(entry.key for entry in scheme.entries if not used.get(entry.key))


@dataclass(frozen=True)
class Shadowed:
    """A row that can never colour anything, and the later row that always overrides it."""

    entry: ColourEntry
    by: ColourEntry

    def __str__(self) -> str:
        return (
            f"row {self.entry.order} {self.entry.key!r} is always overridden by "
            f"row {self.by.order} {self.by.key!r}"
        )


def shadowed_entries(
    scheme: ColourScheme, clade_set: CladeSet, group_set: GroupSet | None = None
) -> tuple[Shadowed, ...]:
    """Rows that can never win under "the last matching row wins", each with its overrider.

    A row is shadowed when a later row matches every virus it matches: a later clade that
    contains the row's clade (or the group's anchor), or a later group whose anchor
    contains the row's anchor and whose substitutions are a subset of the row's. Reported,
    not fatal — ordering is the user's (Q80) — but a shadowed row is dead weight in the
    legend and is how trap T9 shows itself.
    """
    groups = {group.name: group for group in group_set} if group_set is not None else {}
    ordered = scheme.in_order()
    found: list[Shadowed] = []
    for index, entry in enumerate(ordered):
        for later in reversed(ordered[index + 1 :]):
            if _covers(later, entry, clade_set, groups):
                found.append(Shadowed(entry, later))
                break
    return tuple(found)


def _covers(
    later: ColourEntry,
    earlier: ColourEntry,
    clade_set: CladeSet,
    groups: Mapping[str, Group],
) -> bool:
    """True when every virus ``earlier`` matches is also matched by ``later``."""
    if earlier.is_legacy or later.is_legacy:
        # legacy rows match only viruses the subclades leave unnamed, so a legacy row covers
        # (and is covered by) nothing but a legacy row at or above it
        return (
            earlier.is_legacy
            and later.is_legacy
            and clade_set.legacy_is_within(earlier.key, later.key)
        )
    earlier_group = groups.get(earlier.key) if earlier.is_group else None
    earlier_anchor = earlier_group.anchor if earlier_group is not None else earlier.key
    if earlier.is_group and earlier_group is None:
        return False
    if not later.is_group:
        # a clade row covers anything confined to a clade within it; an unanchored group
        # (substitutions alone) is confined to no clade, so no clade row covers it
        return earlier_anchor is not None and clade_set.is_within(earlier_anchor, later.key)
    later_group = groups.get(later.key)
    if later_group is None:
        return False
    if later_group.anchor is not None and (
        earlier_anchor is None or not clade_set.is_within(earlier_anchor, later_group.anchor)
    ):
        return False
    required = set(earlier_group.substitutions) if earlier_group is not None else set()
    return set(later_group.substitutions) <= required
