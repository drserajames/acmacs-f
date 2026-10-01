"""af.clades' facts about each subtype, read from the shared subtype table.

Which subtypes have clade labels, which upstream nomenclature clone names them, and how its
loci convert to af's mature-HA coordinates are data, not code (design rule 10; Sarah, 1 Oct
2026). They live in the ``clades`` section of each row of ``af/subtypes.toml``
(:mod:`af.util.subtypes`), the one table every part of af reads subtypes from, each with the
source that justifies it.

The loader passes sections through unchecked; this module validates them, every row at once
on first use, so a malformed table fails whichever subtype is asked for:

* a row with no ``clades`` section is an error: no clade facts is not the same as "no labels";
* ``labels`` is required. With ``true``: ``repository``, ``ha1_length`` and ``nuc_offset``
  are required. With ``false`` they must be absent, and asking for that subtype's clades
  fails with the row's source as the reason (B/Yam: Sarah, 25 Sep 2026);
* ``source`` is required on every row, and an unknown key is refused.
"""

from __future__ import annotations

import functools
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from af.clades.coordinates import Coordinates
from af.util.subtypes import Subtype, Subtypes, subtypes

SECTION = "clades"
_KEYS = frozenset({"labels", "repository", "ha1_length", "nuc_offset", "source"})
_LABELLED = ("repository", "ha1_length", "nuc_offset")


class CladeSubtypeError(LookupError):
    """A subtype has no clade facts, has no clade labels, or its section is malformed."""


@dataclass(frozen=True)
class CladeFacts:
    """One subtype's clade facts. ``labels`` False: the others are None, ``source`` says why."""

    name: str
    dataset: str
    labels: bool
    repository: str | None
    coordinates: Coordinates | None
    source: str


def clade_facts(name: str, table: Subtypes | None = None) -> CladeFacts:
    """The facts for af subtype ``name`` (e.g. ``"A(H3N2)"``), labelled or not."""
    facts = _validated(table or subtypes())
    try:
        return facts[name]
    except KeyError:
        known = ", ".join(sorted(facts))
        raise CladeSubtypeError(
            f"no subtype {name!r} in the subtype table (it has {known})"
        ) from None


def labelled(name: str, table: Subtypes | None = None) -> CladeFacts:
    """The facts for a subtype that has clade labels; one without them is an error, with why."""
    facts = clade_facts(name, table)
    if not facts.labels:
        raise CladeSubtypeError(f"{name} has no clade labels: {facts.source}")
    return facts


def coordinates_for(name: str, table: Subtypes | None = None) -> Coordinates:
    """How ``name``'s upstream loci convert to af's mature-HA coordinates."""
    coordinates = labelled(name, table).coordinates
    assert coordinates is not None  # labelled rows always have them
    return coordinates


def nomenclature_repository(name: str, table: Subtypes | None = None) -> str:
    """The influenza-clade-nomenclature clone that names ``name``'s clades."""
    repository = labelled(name, table).repository
    assert repository is not None
    return repository


def clade_dataset(name: str, table: Subtypes | None = None) -> str:
    """The ``clades/<key>`` store dataset of a labelled subtype: its row key."""
    return labelled(name, table).dataset


def clade_subtypes(table: Subtypes | None = None) -> tuple[str, ...]:
    """Every subtype that has clade labels, in table order."""
    return tuple(name for name, facts in _validated(table or subtypes()).items() if facts.labels)


def subtype_for_dataset(dataset: str, table: Subtypes | None = None) -> str:
    """The labelled subtype whose clades live in ``clades/<dataset>``."""
    for name, facts in _validated(table or subtypes()).items():
        if facts.labels and facts.dataset == dataset:
            return name
    raise CladeSubtypeError(f"no labelled subtype has clade dataset {dataset!r}")


@functools.cache
def _validated(table: Subtypes) -> Mapping[str, CladeFacts]:
    problems: list[str] = []
    facts: dict[str, CladeFacts] = {}
    for row in table:
        fact = _facts(row, problems)
        if fact is not None:
            facts[row.name] = fact
    if problems:
        raise CladeSubtypeError(f"{table.source}: " + "; ".join(problems))
    return facts


def _facts(row: Subtype, problems: list[str]) -> CladeFacts | None:
    where = f"[subtype.{row.key}.{SECTION}]"
    section: Mapping[str, Any] | None = row.sections.get(SECTION)
    if section is None:
        problems.append(f"{where} is missing: every subtype must say whether it has clade labels")
        return None
    unknown = sorted(set(section) - _KEYS)
    if unknown:
        problems.append(f"{where}: unknown key(s) {', '.join(unknown)}")
    source = section.get("source")
    if not isinstance(source, str) or not source.strip():
        problems.append(f"{where}: no source")
    labels = section.get("labels")
    if not isinstance(labels, bool):
        problems.append(f"{where}: labels must be true or false")
        return None
    if not labels:
        present = [key for key in _LABELLED if key in section]
        if present:
            problems.append(f"{where}: labels = false, but {', '.join(present)} given")
        return CladeFacts(row.name, row.key, False, None, None, str(source))
    missing = [key for key in _LABELLED if key not in section]
    if missing:
        problems.append(f"{where}: labels = true needs {', '.join(missing)}")
        return None
    repository, ha1, nuc = section["repository"], section["ha1_length"], section["nuc_offset"]
    if not isinstance(repository, str) or not repository:
        problems.append(f"{where}: repository must be a directory name")
    for key, value in (("ha1_length", ha1), ("nuc_offset", nuc)):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            problems.append(f"{where}: {key} must be a positive whole number, got {value!r}")
    return CladeFacts(
        row.name, row.key, True, str(repository), Coordinates(int(ha1), int(nuc)), str(source)
    )


__all__ = [
    "CladeFacts",
    "CladeSubtypeError",
    "clade_dataset",
    "clade_facts",
    "clade_subtypes",
    "coordinates_for",
    "labelled",
    "nomenclature_repository",
    "subtype_for_dataset",
]
