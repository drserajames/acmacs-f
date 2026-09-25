"""User-defined colour schemes: which clades and groups are drawn, and in what colour.

Colours are the user's choice, not the nomenclature's (Sarah, 24 Sep 2026), so they live
in acmacs-f-data as plain tables, one per scheme::

    colours/<subtype>/<scheme>.tsv
    order   key        legend       colour
    1       C.1        C.1          #98e1d7
    2       D          D            #5b00d3
    3       D 139N     D.3.1 139N   #00bfff

``key`` is a clade name or a group name (:mod:`af.clades.groups`); ``legend`` is what the
figure prints, which is not always the key — today's tables relabel clades on the figure
while keeping the old key. ``order`` is the legend's order and nothing else.

**Which entry colours a virus is decided by specificity, not by row order.** Today the
rule is "the later row wins", so a scheme that lists a child clade above its parent
silently draws the child in the parent's colour, and the only defence is remembering to
order the file correctly (INVENTORY E, trap T9; the H1 table carries a comment doing
exactly this by hand). Here the most specific entry wins:

1. a group the virus matches, most specific group first (groups are a deliberate,
   narrower statement than a clade);
2. otherwise the deepest clade entry the virus's clade lies within;
3. otherwise nothing — the virus is not drawn by this scheme.

Order in the file therefore cannot change which colour a virus gets, only the order of
the legend. Two entries that would tie are a load error rather than a silent winner.
"""

from __future__ import annotations

import csv
import re
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from af.clades.groups import GroupSet
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
        return tuple(
            (entry.legend, entry.colour)
            for entry in sorted(self.entries, key=lambda entry: entry.order)
        )

    def entry_for(
        self,
        clade: str | None,
        sequence: AlignedSequence,
        clade_set: CladeSet,
        group_set: GroupSet | None = None,
    ) -> ColourEntry | None:
        """The entry that colours this virus: most specific wins, never row order."""
        by_key = {entry.key: entry for entry in self.entries}
        if group_set is not None:
            for name in group_set.matching(clade, sequence, clade_set):
                if name in by_key:
                    return by_key[name]
        if clade is None:
            return None
        candidates = [
            entry
            for entry in self.entries
            if not entry.is_group
            and entry.key in clade_set
            and clade_set.is_within(clade, entry.key)
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda entry: clade_set.depth(entry.key))


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
    problems: list[str] = []
    entries: list[ColourEntry] = []
    seen_keys: set[str] = set()
    seen_orders: dict[int, str] = {}
    group_names = set(group_set.names) if group_set else set()

    with path.open(newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        fields = reader.fieldnames or []
        missing = [column for column in COLUMNS if column not in fields]
        if missing:
            raise ColourSchemeError(path, [f"missing column(s): {', '.join(missing)}"])
        unknown_columns = sorted(set(fields) - set(COLUMNS))
        if unknown_columns:
            raise ColourSchemeError(path, [f"unknown column(s): {', '.join(unknown_columns)}"])
        for line, row in enumerate(reader, start=2):
            values = {key: (row.get(key) or "").strip() for key in COLUMNS}
            if not any(values.values()):
                continue
            where = f"line {line}"
            key = values["key"]
            if not key:
                problems.append(f"{where}: no key")
                continue
            is_group = key in group_names
            if not is_group and key not in clade_set:
                problems.append(
                    f"{where}: key {key!r} is neither a clade of {subtype} at "
                    f"{clade_set.version} nor a known group"
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
                ColourEntry(order, key, values["legend"] or key, values["colour"].lower(), is_group)
            )
    if not entries and not problems:
        problems.append("scheme has no entries")
    if problems:
        raise ColourSchemeError(path, problems)
    return ColourScheme(subtype, path.stem, tuple(entries), path)


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
