"""Clades af defines itself, on top of the published nomenclature.

Some names the reports need are not upstream's to publish and never will be: the historical
H3 antigenic clusters, sub-lineages the group has named for its own use, and the older
lineages that predate the nomenclature's root. Upstream also has no B/Yamagata repository
at all. These stay local (DECISIONS, 24 Sep 2026).

They are written the same way upstream writes a subclade — a parent and the mutations
that define the branch — so the tree engine treats them identically and nothing needs a
second matching mechanism:

    clades/local.tsv
    subtype   name    parent   mutations            scope        note
    A(H3N2)   XX68    -        145S 155T 156K       historical   antigenic cluster
    A(H3N2)   J.2.x   J.2      346M 8D              active

``parent`` empty (or ``-``) means the clade attaches at the root, which is how a
pre-nomenclature lineage is expressed: upstream's own root clade is itself defined by
substitutions, so viruses ancestral to it sit outside every upstream clade and can only be
named locally. ``mutations`` are positions in af's mature-HA amino-acid numbering, or
``nuc123A`` for a nucleotide.

``scope`` is ``historical`` or ``active``, and is documentation for the user rather than
something the engine acts on: a historical cluster still has to be assignable, because old
viruses are still drawn.

**The local layer is part of the clade-set version.** A clade table built with local
definitions records a version naming both the upstream commit and the content of this
file, so a change to either makes existing assignments detectably stale (design rule 5).
That is the failure the old system had no defence against: an edit to the local tables
reached maps only on a re-populate, and nothing said so.
"""

from __future__ import annotations

import csv
import hashlib
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

from af.clades.coordinates import Position
from af.clades.nomenclature import CladeSet, Subclade

COLUMNS = ("subtype", "name", "parent", "mutations", "scope", "note")
SCOPES = ("historical", "active")
MUTATION = re.compile(r"^(?:(?P<nuc>nuc)\s*)?(?P<position>[0-9]+)(?P<state>[A-Z-])$", re.IGNORECASE)
ROOT = {"", "-", "none", "root"}


class LocalCladeError(ValueError):
    """The local clade file is malformed, or conflicts with the nomenclature."""

    def __init__(self, source: Path | str, problems: Sequence[str]) -> None:
        self.source = source
        self.problems = list(problems)
        lines = "\n".join(f"  - {problem}" for problem in self.problems)
        super().__init__(f"invalid local clades {source}:\n{lines}")


@dataclass(frozen=True)
class LocalClade:
    """One locally defined clade, in the same shape as an upstream subclade."""

    subtype: str
    name: str
    parent: str | None
    mutations: tuple[Position, ...]
    scope: str = "active"
    note: str = ""
    source_line: int = 0


def parse_mutations(text: str, problems: list[str], where: str) -> tuple[Position, ...]:
    """Parse ``145S nuc123A`` into mature-HA positions."""
    positions: list[Position] = []
    for token in text.split():
        matched = MUTATION.fullmatch(token)
        if matched is None:
            problems.append(
                f"{where}: {token!r} is not a mutation like '145S', '163-' or 'nuc123A'"
            )
            continue
        positions.append(
            Position(
                "nuc" if matched.group("nuc") else "aa",
                int(matched.group("position")),
                matched.group("state").upper(),
            )
        )
    return tuple(positions)


def load_local_clades(path: Path) -> dict[str, list[LocalClade]]:
    """Read ``local.tsv``. Every problem is reported together, each with its line."""
    path = Path(path)
    problems: list[str] = []
    by_subtype: dict[str, list[LocalClade]] = {}
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        fields = reader.fieldnames or []
        missing = [column for column in COLUMNS if column not in fields]
        if missing:
            raise LocalCladeError(path, [f"missing column(s): {', '.join(missing)}"])
        unknown = sorted(set(fields) - set(COLUMNS))
        if unknown:
            raise LocalCladeError(path, [f"unknown column(s): {', '.join(unknown)}"])
        for line, row in enumerate(reader, start=2):
            values = {key: (row.get(key) or "").strip() for key in COLUMNS}
            if not any(values.values()):
                continue
            where = f"line {line}"
            if not values["subtype"] or not values["name"]:
                problems.append(f"{where}: subtype and name are required")
                continue
            scope = values["scope"] or "active"
            if scope not in SCOPES:
                problems.append(f"{where}: scope {scope!r} must be one of {', '.join(SCOPES)}")
                continue
            mutations = parse_mutations(values["mutations"], problems, where)
            if not mutations:
                problems.append(
                    f"{where}: local clade {values['name']!r} defines no mutations, so nothing "
                    "could ever be assigned to it"
                )
                continue
            parent = None if values["parent"].lower() in ROOT else values["parent"]
            by_subtype.setdefault(values["subtype"], []).append(
                LocalClade(
                    values["subtype"],
                    values["name"],
                    parent,
                    mutations,
                    scope,
                    values["note"],
                    line,
                )
            )
    if problems:
        raise LocalCladeError(path, problems)
    return by_subtype


def content_version(path: Path) -> str:
    """A short hash of the file's contents, for the clade-set version."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]


def extend(
    clade_set: CladeSet,
    local: Sequence[LocalClade],
    *,
    version_suffix: str,
    source: Path | str = "local.tsv",
) -> CladeSet:
    """Return ``clade_set`` with the local clades added.

    A local clade may not take a name the nomenclature already uses, and may not attach
    to a parent that does not exist: either would make a local edit quietly change what a
    published clade name means, which is the one thing the local layer must not do.
    """
    problems: list[str] = []
    subclades = dict(clade_set.subclades)
    names = {entry.name for entry in local}
    for entry in local:
        where = f"line {entry.source_line}"
        if entry.name in clade_set:
            problems.append(
                f"{where}: local clade {entry.name!r} is already defined by the nomenclature "
                f"at {clade_set.version}; a local definition may not redefine a published clade"
            )
            continue
        if entry.parent is not None and entry.parent not in clade_set and entry.parent not in names:
            problems.append(
                f"{where}: local clade {entry.name!r} has parent {entry.parent!r}, which is "
                "neither a published clade nor another local one"
            )
            continue
        subclades[entry.name] = Subclade(
            name=entry.name,
            parent=entry.parent,
            mutations=entry.mutations,
            comment=entry.note or None,
            source=f"{source}:{entry.source_line}",
        )
    if problems:
        raise LocalCladeError(source, problems)
    ancestors = _ancestry(subclades)
    return replace(
        clade_set,
        subclades=subclades,
        version=f"{clade_set.version}+local:{version_suffix}",
        local_names=frozenset(clade_set.local_names | names),
        _ancestors=ancestors,
    )


def extend_from_file(clade_set: CladeSet, path: Path) -> CladeSet:
    """Load ``path`` and extend ``clade_set`` with this subtype's local clades."""
    path = Path(path)
    local = load_local_clades(path).get(clade_set.subtype, [])
    return extend(clade_set, local, version_suffix=content_version(path), source=path)


def _ancestry(subclades: Mapping[str, Subclade]) -> dict[str, tuple[str, ...]]:
    """Recompute ancestry over the combined set; a cycle through a local parent is fatal."""
    ancestors: dict[str, tuple[str, ...]] = {}
    for name in subclades:
        chain: list[str] = []
        seen = {name}
        current = subclades[name].parent
        while current is not None and current in subclades:
            if current in seen:
                raise LocalCladeError(
                    "local.tsv", [f"clade parentage forms a cycle at {current!r}"]
                )
            seen.add(current)
            chain.append(current)
            current = subclades[current].parent
        ancestors[name] = tuple(chain)
    return ancestors
