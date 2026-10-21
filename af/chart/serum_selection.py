"""Which sera an addendum shows, section by section, each with the reason it was chosen.

The serum-coverage and serum-circle addenda pick their sera the same way, so they share this:
a round's section table (config, not code: the strains named in it are the round's) is matched
against every serum of the labs the addendum covers. A section selects by designation substring
(the vaccine sera), by clade, by amino acid, or by clade and amino acid; then:

- **recency**: only sera titrated on or after ``since`` (the round gives the date, e.g. 12
  months before the meeting), as the Sep 2026 addenda did; a serum with no titration date
  is kept and counted;
- **passage**: only sera of the given passage classes (the circle addenda show cell sera);
- **one section per serum**: a serum matching several sections goes to the one with the
  lowest ``priority``, then the fewest matches (a child clade before its parent), then the
  first in the table; ``drop`` claims sera and shows none of them, and ``extra`` shows its
  sera without claiming them.

A section that matches nothing is an error unless it is marked optional (design rule 1).
"""

from __future__ import annotations

import datetime as dt
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

_RESIDUE = re.compile(r"^(?P<pos>[1-9][0-9]*)(?P<aa>[A-Z-])$")


class SelectionError(ValueError):
    """A section that is malformed, or that matches no serum and is not optional."""


@dataclass(frozen=True)
class Section:
    """One section of an addendum. Exactly one kind of selector: ``strains`` (designation
    substrings), or ``clade`` and/or ``aa`` (residues such as "155E", all required)."""

    label: str
    strains: tuple[str, ...] = ()
    clade: str | None = None
    aa: tuple[str, ...] = ()
    exclude: tuple[str, ...] = ()
    priority: int = 0
    drop: bool = False
    extra: bool = False
    optional: bool = False

    def __post_init__(self) -> None:
        if bool(self.strains) == bool(self.clade or self.aa):
            raise SelectionError(f"section {self.label!r}: give strains, or clade and/or aa")
        for r in self.aa:
            if not _RESIDUE.match(r):
                raise SelectionError(f"section {self.label!r}: residue {r!r} is not like 155E")
        if self.drop and self.extra:
            raise SelectionError(f"section {self.label!r}: drop and extra together")

    @classmethod
    def from_config(cls, entry: Mapping[str, Any]) -> Section:
        known = {
            "label",
            "strains",
            "clade",
            "aa",
            "exclude",
            "priority",
            "drop",
            "extra",
            "optional",
        }
        if unknown := sorted(set(entry) - known):
            raise SelectionError(f"section {entry.get('label')!r}: unknown keys {unknown}")
        if "label" not in entry:
            raise SelectionError(f"section without a label: {dict(entry)}")
        return cls(
            label=str(entry["label"]),
            strains=tuple(entry.get("strains", ())),
            clade=entry.get("clade"),
            aa=tuple(entry.get("aa", ())),
            exclude=tuple(entry.get("exclude", ())),
            priority=int(entry.get("priority", 0)),
            drop=bool(entry.get("drop", False)),
            extra=bool(entry.get("extra", False)),
            optional=bool(entry.get("optional", False)),
        )

    def matches(self, serum: Candidate) -> str | None:
        """Why the serum is in this section, or None. Exclusions are applied by the caller."""
        if self.strains:
            designation = serum.designation.upper()
            hit = next((s for s in self.strains if s.upper() in designation), None)
            return None if hit is None else f"designation contains {hit!r}"
        if self.clade is not None and self.clade not in serum.clades:
            return None
        for r in self.aa:
            m = _RESIDUE.match(r)
            assert m is not None  # checked in __post_init__
            if serum.aa.get(int(m["pos"])) != m["aa"]:
                return None
        return " + ".join(
            [*([f"clade {self.clade}"] if self.clade else []), *(f"aa {r}" for r in self.aa)]
        )


@dataclass(frozen=True)
class Candidate:
    """A serum of one lab's chart, as selection sees it. ``key`` is the caller's identity for
    it (never a chart index on its own: design rule 2). ``clades`` holds every clade label of
    the serum's strain, ancestors included, so a section for a parent clade matches its
    children; ``aa`` maps a position to the residue there."""

    key: tuple[str, ...]
    designation: str
    passage_class: str | None
    clades: frozenset[str] = frozenset()
    aa: Mapping[int, str] = field(default_factory=dict)
    last_titrated: dt.date | None = None


@dataclass(frozen=True)
class Chosen:
    serum: Candidate
    reason: str


@dataclass(frozen=True)
class Selection:
    """The sera each shown section holds, in the candidates' order, with what was left out and
    why (counted, design rule 1)."""

    sections: dict[str, list[Chosen]]
    dropped: dict[str, list[Chosen]]
    excluded: dict[str, int]  # reason -> number of sera

    def report(self) -> dict[str, Any]:
        return {
            "sections": {k: len(v) for k, v in self.sections.items()},
            "dropped": {k: len(v) for k, v in self.dropped.items()},
            "excluded": dict(self.excluded),
        }


def select_sera(
    candidates: Sequence[Candidate],
    sections: Sequence[Section],
    *,
    since: dt.date | None,
    passage_classes: frozenset[str] | None = None,
) -> Selection:
    """The sera for each section. ``since``: keep sera titrated on or after it (None: no
    recency filter). ``passage_classes``: keep sera of these classes only (None: all)."""
    labels = [s.label for s in sections]
    if len(set(labels)) != len(labels):
        raise SelectionError(f"section labels repeat: {labels}")
    excluded: dict[str, int] = {}

    def skip(why: str) -> None:
        excluded[why] = excluded.get(why, 0) + 1

    eligible = []
    for c in candidates:
        if passage_classes is not None and c.passage_class not in passage_classes:
            skip(f"passage class {c.passage_class}")
        elif since is not None and c.last_titrated is not None and c.last_titrated < since:
            skip(f"last titrated before {since.isoformat()}")
        else:
            if since is not None and c.last_titrated is None:
                skip("no titration date (kept)")
            eligible.append(c)

    matched: dict[str, dict[tuple[str, ...], str]] = {}
    for s in sections:
        found = {}
        for c in eligible:
            reason = s.matches(c)
            if reason is None:
                continue
            if any(x.upper() in c.designation.upper() for x in s.exclude):
                skip(f"excluded from {s.label!r} by designation")
                continue
            found[c.key] = reason
        if not found and not s.optional:
            raise SelectionError(f"section {s.label!r} matches no serum")
        matched[s.label] = found

    exclusive = [s for s in sections if not s.extra]
    owner: dict[tuple[str, ...], str] = {}
    for c in eligible:
        claims = [s for s in exclusive if c.key in matched[s.label]]
        if claims:
            best = min(
                claims, key=lambda s: (s.priority, len(matched[s.label]), labels.index(s.label))
            )
            owner[c.key] = best.label

    shown: dict[str, list[Chosen]] = {}
    dropped: dict[str, list[Chosen]] = {}
    for s in sections:
        mine = [
            Chosen(c, matched[s.label][c.key])
            for c in eligible
            if c.key in matched[s.label] and (s.extra or owner.get(c.key) == s.label)
        ]
        (dropped if s.drop else shown)[s.label] = mine
    return Selection(shown, dropped, excluded)
