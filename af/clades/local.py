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
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

from af.clades.coordinates import Position
from af.clades.nomenclature import CladeSet, Subclade

COLUMNS = ("subtype", "name", "parent", "mutations", "scope", "note")
#: Columns a file may leave out. Without ``mutations`` every row takes its signature from
#: the source it was carried over from (:func:`from_signatures`).
OPTIONAL = ("mutations",)
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
        missing = [column for column in COLUMNS if column not in fields and column not in OPTIONAL]
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
            # empty means "from the source signature": resolved by from_signatures, and
            # refused by extend if it never was
            mutations = parse_mutations(values["mutations"], problems, where)
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
        if not entry.mutations:
            problems.append(
                f"{where}: local clade {entry.name!r} defines no mutations, so nothing could "
                "ever be assigned to it; give them, or read its signature from its source "
                "(from_signatures)"
            )
            continue
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


def extend_from_file(
    clade_set: CladeSet,
    path: Path,
    *,
    signatures: Mapping[str, Sequence[str]] | None = None,
) -> CladeSet:
    """Load ``path`` and extend ``clade_set`` with this subtype's local clades.

    ``signatures`` (name -> tokens, see :func:`from_signatures`) supplies the mutations of
    rows that give none. The clade-set version then covers the signatures used as well as
    the file, since a change to either changes what the local clades match.
    """
    path = Path(path)
    local = load_local_clades(path).get(clade_set.subtype, [])
    suffix = content_version(path)
    if signatures is not None:
        local = from_signatures(local, clade_set, signatures, source=path)
        suffix = f"{suffix}.{signatures_version(local, signatures)}"
    return extend(clade_set, local, version_suffix=suffix, source=path)


def from_signatures(
    local: Sequence[LocalClade],
    clade_set: CladeSet,
    signatures: Mapping[str, Sequence[str]],
    *,
    source: Path | str = "local.tsv",
) -> list[LocalClade]:
    """Give each row without mutations its own mutations, from a source signature.

    This is how a local clade is carried over while its definition still lives elsewhere
    (``acmacs-data``'s ``clades.json``, read by :func:`af.clades.importer.clades_json_signatures`):
    the local file adds only what the source lacks — the parent, scope and reason — and the
    source stays the one editable copy of the signature (design rule 6).

    The old signatures are not per-branch: they repeat positions the parent already has. A
    row's own mutations are therefore its signature minus its parent's cumulative
    signature, which is also exactly what reproduces upstream's own children from theirs.

    Refused, with every problem listed: a row whose name the source lacks; a row that also
    gives its own mutations (two copies); a parent that is not a published clade (a local
    parent has no cumulative signature to subtract); a signature that adds nothing to its
    parent's, which would make the clade indistinguishable from it.
    """
    problems: list[str] = []
    resolved: list[LocalClade] = []
    for entry in local:
        where = f"line {entry.source_line}"
        if entry.mutations:
            if entry.name in signatures:
                problems.append(
                    f"{where}: {entry.name!r} gives mutations here and has a source signature; "
                    "keep one copy"
                )
            resolved.append(entry)
            continue
        if entry.name not in signatures:
            problems.append(
                f"{where}: {entry.name!r} gives no mutations and has no source signature"
            )
            continue
        if (
            entry.parent is None
            or entry.parent not in clade_set
            or clade_set.is_local(entry.parent)
        ):
            problems.append(
                f"{where}: {entry.name!r} takes its signature from the source, so its parent "
                f"must be a published clade, not {entry.parent!r}"
            )
            continue
        signature = parse_mutations(" ".join(signatures[entry.name]), problems, where)
        parent = clade_set.cumulative(entry.parent)
        own = tuple(m for m in signature if parent.get((m.alphabet, m.position)) != m.state)
        if not own:
            problems.append(
                f"{where}: {entry.name!r}'s source signature adds nothing to {entry.parent!r}'s"
            )
            continue
        resolved.append(replace(entry, mutations=own))
    if problems:
        raise LocalCladeError(source, problems)
    return resolved


def signatures_version(local: Sequence[LocalClade], signatures: Mapping[str, Sequence[str]]) -> str:
    """A short hash of the source signatures these rows used, and only those: an edit to an
    unrelated clade in the source must not make every clade table look stale."""
    used = {entry.name: list(signatures[entry.name]) for entry in local if entry.name in signatures}
    text = json.dumps(used, sort_keys=True)
    return hashlib.sha256(text.encode()).hexdigest()[:12]


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
