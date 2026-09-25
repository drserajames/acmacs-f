"""Local clade groups: "this clade, or anything under it, plus these substitutions".

The reports do not label viruses by clade alone. They also label sub-groupings such as
"clade K with 145R", which are local choices about what is worth showing this season, not
part of the published nomenclature. Today those live in ``semantic_clades.py`` as
``attributes`` rows of ``name | clade | aa``, and they fail in two silent ways
(INVENTORY E §2.3, traps T5 and T6):

* the ``clade`` column has to equal a clade name exactly, so when the nomenclature renames
  a clade every row naming the old name matches nothing — 24 such rows are inert today;
* a row anchored on a clade misses the viruses in that clade's *children*, so a row has to
  name the parent to catch them, which is easy to get wrong in the other direction.

Both are fixed here by construction. An anchor is checked against the clade set when the
file is read, so an unknown one is an error rather than an empty label (design rule 1),
and a group matches the anchor **or any clade below it**, so children are included
without naming them.

The file (``clades/groups.tsv`` in acmacs-f-data), tab-separated with a header::

    subtype   group      anchor   substitutions   note
    A(H3N2)   K 145R     K        145R            for the September map
    A(H1N1)   D 139N     D        139N 155E

``substitutions`` are positions in af's mature-HA amino-acid numbering, the same as
everywhere else in af. An empty ``anchor`` means "any clade", for a group defined by
substitutions alone; an empty ``substitutions`` would make the group identical to its
anchor clade, so it is rejected.
"""

from __future__ import annotations

import csv
import re
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from af.clades.nomenclature import CladeSet
from af.clades.sequence import AlignedSequence, Evidence

SUBSTITUTION = re.compile(r"^(?P<position>[0-9]+)(?P<state>[A-Z-])$")
COLUMNS = ("subtype", "group", "anchor", "substitutions", "note")


class GroupError(ValueError):
    """A group file is malformed, or names something the clade set does not have."""

    def __init__(self, source: Path | str, problems: Sequence[str]) -> None:
        self.source = source
        self.problems = list(problems)
        lines = "\n".join(f"  - {problem}" for problem in self.problems)
        super().__init__(f"invalid clade groups {source}:\n{lines}")


@dataclass(frozen=True)
class Substitution:
    """One position and the residue a member of the group carries there."""

    position: int
    state: str

    def __str__(self) -> str:
        return f"{self.position}{self.state}"


@dataclass(frozen=True)
class Group:
    """A named sub-grouping: a clade (or its descendants) plus required substitutions."""

    subtype: str
    name: str
    anchor: str | None
    substitutions: tuple[Substitution, ...]
    note: str = ""
    source_line: int = 0

    def matches(self, clade: str | None, sequence: AlignedSequence, clade_set: CladeSet) -> bool:
        """True when a virus in ``clade`` with this ``sequence`` belongs to the group.

        An unobservable position does not match: a group is a positive claim about what a
        virus carries, and "we cannot see position 145" is not evidence that it is 145R.
        """
        if self.anchor is not None and (
            clade is None or not clade_set.is_within(clade, self.anchor)
        ):
            return False
        return all(
            sequence.evidence("aa", substitution.position, substitution.state) is Evidence.MATCHES
            for substitution in self.substitutions
        )


@dataclass(frozen=True)
class GroupSet:
    """Every group of one subtype, in file order."""

    subtype: str
    groups: tuple[Group, ...]

    def __post_init__(self) -> None:
        # A second group of one name would be silently shadowed by the first in every
        # lookup; the file loader refuses it, and so must a set built in memory.
        seen: set[str] = set()
        repeated: list[str] = []
        for group in self.groups:
            if group.name in seen:
                repeated.append(group.name)
            seen.add(group.name)
        if repeated:
            raise GroupError(self.subtype, [f"duplicate group {name!r}" for name in repeated])

    def __iter__(self) -> Iterator[Group]:
        return iter(self.groups)

    def __len__(self) -> int:
        return len(self.groups)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(group.name for group in self.groups)

    def matching(
        self, clade: str | None, sequence: AlignedSequence, clade_set: CladeSet
    ) -> tuple[str, ...]:
        """Every group a virus belongs to, most specific first.

        Specificity is the number of substitutions, then the depth of the anchor: a virus
        that is both "D 139N" and "D 139N 155E" is listed under the more specific one
        first, so a caller wanting one label can take the first without a tie-break rule
        living in the caller.
        """
        matched = [group for group in self.groups if group.matches(clade, sequence, clade_set)]
        return tuple(
            group.name
            for group in sorted(
                matched,
                key=lambda group: (
                    len(group.substitutions),
                    clade_set.depth(group.anchor) if group.anchor else -1,
                ),
                reverse=True,
            )
        )


def parse_substitutions(text: str, problems: list[str], where: str) -> tuple[Substitution, ...]:
    """Parse ``145R 155E``. A token that is not a position and a residue is an error."""
    substitutions: list[Substitution] = []
    for token in text.split():
        matched = SUBSTITUTION.fullmatch(token)
        if matched is None:
            problems.append(f"{where}: {token!r} is not a substitution like '145R' or '163-'")
            continue
        substitutions.append(Substitution(int(matched.group("position")), matched.group("state")))
    return tuple(substitutions)


def load_groups(path: Path, clade_sets: Mapping[str, CladeSet]) -> dict[str, GroupSet]:
    """Read ``groups.tsv`` and check it against the clade sets it refers to.

    Every problem in the file is reported together, each with its line, so one run shows
    everything wrong with it rather than one error per run.
    """
    path = Path(path)
    problems: list[str] = []
    rows = _read_rows(path, problems)
    by_subtype: dict[str, list[Group]] = {subtype: [] for subtype in clade_sets}
    seen: set[tuple[str, str]] = set()
    for line, row in rows:
        where = f"line {line}"
        subtype = row["subtype"]
        if subtype not in clade_sets:
            known = ", ".join(sorted(clade_sets))
            problems.append(f"{where}: unknown subtype {subtype!r}; loaded: {known}")
            continue
        clade_set = clade_sets[subtype]
        name = row["group"]
        if not name:
            problems.append(f"{where}: no group name")
            continue
        if (subtype, name) in seen:
            problems.append(f"{where}: duplicate group {name!r} for {subtype}")
            continue
        seen.add((subtype, name))
        anchor = row["anchor"] or None
        if anchor is not None and anchor not in clade_set:
            problems.append(
                f"{where}: group {name!r} is anchored on {anchor!r}, which {subtype} does not "
                f"define at {clade_set.version}"
            )
            continue
        before = len(problems)
        substitutions = parse_substitutions(row["substitutions"], problems, where)
        if len(problems) > before:
            # the tokens were already reported as unreadable; saying "no substitutions"
            # as well would report one mistake twice
            continue
        if not substitutions:
            problems.append(
                f"{where}: group {name!r} has no substitutions, so it would be identical to "
                f"its anchor clade"
            )
            continue
        by_subtype[subtype].append(Group(subtype, name, anchor, substitutions, row["note"], line))
    if problems:
        raise GroupError(path, problems)
    return {subtype: GroupSet(subtype, tuple(groups)) for subtype, groups in by_subtype.items()}


def _read_rows(path: Path, problems: list[str]) -> list[tuple[int, dict[str, str]]]:
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        missing = [column for column in COLUMNS if column not in (reader.fieldnames or [])]
        if missing:
            raise GroupError(path, [f"missing column(s): {', '.join(missing)}"])
        unknown = sorted(set(reader.fieldnames or []) - set(COLUMNS))
        if unknown:
            raise GroupError(path, [f"unknown column(s): {', '.join(unknown)}"])
        rows: list[tuple[int, dict[str, str]]] = []
        for line, row in enumerate(reader, start=2):
            if all(not (value or "").strip() for value in row.values()):
                continue
            rows.append((line, {key: (row.get(key) or "").strip() for key in COLUMNS}))
    if not rows:
        problems.append("file has no rows")
    return rows


def count_matches(
    group_set: GroupSet,
    viruses: Iterable[tuple[str | None, AlignedSequence]],
    clade_set: CladeSet,
) -> dict[str, int]:
    """How many viruses each group matches.

    Reported so a group that matches nothing is visible (design rule 1). It is not an
    error by itself — a season's grouping may legitimately have no viruses yet — but it
    is exactly what a mistyped substitution looks like, so it must never pass unseen.
    """
    counts = dict.fromkeys(group_set.names, 0)
    for clade, sequence in viruses:
        for name in group_set.matching(clade, sequence, clade_set):
            counts[name] += 1
    return counts
