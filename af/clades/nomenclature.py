"""Read the upstream influenza clade nomenclature at a pinned commit.

The nomenclature (Neher et al. 2026) is a Pango-style hierarchy published as one YAML
file per subclade in the ``influenza-clade-nomenclature`` repositories. af takes clade
names, parentage and defining mutations from there and holds nothing of its own
(DECISIONS, 24 Sep 2026): local additions are a separate, explicitly local layer.

Two things this module insists on, both because their absence has cost real time before:

* **A pin.** The clones move under you: an audit run a week apart gives different
  answers. :class:`Pin` names the exact commit per repository, and
  :func:`load_clade_set` checks the clone is at that commit before reading it.
* **One coordinate conversion**, done in :mod:`af.clades.coordinates`, with loci that a
  mature-HA sequence cannot carry reported rather than dropped.

The YAML files are small and very regular, so they are parsed here rather than adding a
YAML dependency for them. The parser is deliberately strict: a line it does not
recognise is an error naming the file and the line, because a silently skipped
``defining_mutations`` entry becomes a clade rule that matches the wrong viruses.
"""

from __future__ import annotations

import subprocess
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path

from af.clades.coordinates import (
    Alphabet,
    Coordinates,
    Position,
    Unexpressible,
    convert,
    coordinates_for,
)

#: The upstream repository for each subtype's HA segment.
HA_REPOSITORIES: dict[str, str] = {
    "A(H1N1)": "seasonal_A-H1N1pdm_HA",
    "A(H3N2)": "seasonal_A-H3N2_HA",
    "B/Vic": "seasonal_B-Vic_HA",
}

_SCALAR_KEYS = frozenset(
    {"name", "parent", "alias_of", "unaliased_name", "clade", "short_name", "revoked", "comment"}
)
_LIST_KEYS = frozenset({"defining_mutations", "representatives"})
_NONE = frozenset({"none", "None", ""})


class NomenclatureError(ValueError):
    """The upstream data could not be read, or is not what the pin says it is."""


@dataclass(frozen=True)
class Pin:
    """The exact upstream commit af reads for one subtype.

    ``commit`` is a full or abbreviated SHA. It is checked, not assumed: a clone that has
    moved on is a hard error, so a clade set cannot silently change between runs.
    """

    subtype: str
    repository: str
    commit: str


@dataclass(frozen=True)
class Subclade:
    """One node of the nomenclature, with its own branch's defining mutations.

    ``mutations`` are **per-branch** and already in af's mature-HA coordinates; the
    cumulative signature of a node is its own plus every ancestor's
    (:meth:`CladeSet.cumulative`). ``unexpressible`` lists defining mutations that a
    mature-HA sequence cannot carry (signal peptide, nucleotides before the mature HA).
    """

    name: str
    parent: str | None
    mutations: tuple[Position, ...]
    unexpressible: tuple[Unexpressible, ...] = ()
    alias_of: str | None = None
    unaliased_name: str | None = None
    clade: str | None = None
    short_name: str | None = None
    revoked: bool = False
    comment: str | None = None
    source: str = ""

    @property
    def is_pointer(self) -> bool:
        """True for a clade file that only points at a subclade and defines nothing."""
        return self.alias_of is not None and not self.mutations and not self.unexpressible


@dataclass(frozen=True)
class CladeSet:
    """The subclade hierarchy of one subtype at one pinned commit.

    Ancestry is derived here and nowhere else: af stores a single clade per sequence and
    answers "is this virus in D?" with :meth:`is_within` (Sarah, 25 Sep 2026), rather
    than storing every ancestor's name against every virus as the old system did.

    Upstream publishes **two** naming hierarchies: ``subclades/`` holds the short
    Pango-style names af assigns, and ``clades/`` holds the older, longer names people
    still read on figures. Only subclades are assignable; the clade files are display
    names, reached through :meth:`legacy_name`. Mixing the two silently labels every
    virus with a legacy name instead of its clade.
    """

    subtype: str
    pin: Pin
    coordinates: Coordinates
    subclades: Mapping[str, Subclade]
    version: str
    legacy_clades: Mapping[str, Subclade] = field(default_factory=dict)
    local_names: frozenset[str] = frozenset()
    """Clades af defines itself (:mod:`af.clades.local`); empty for a pure upstream set."""
    _ancestors: Mapping[str, tuple[str, ...]] = field(repr=False, default_factory=dict)

    def __iter__(self) -> Iterator[Subclade]:
        return iter(self.subclades.values())

    def __contains__(self, name: object) -> bool:
        return name in self.subclades

    def __getitem__(self, name: str) -> Subclade:
        try:
            return self.subclades[name]
        except KeyError:
            raise KeyError(f"{self.subtype}: no clade named {name!r} at {self.version}") from None

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self.subclades)

    @property
    def live(self) -> tuple[Subclade, ...]:
        """Subclades that are not revoked."""
        return tuple(clade for clade in self.subclades.values() if not clade.revoked)

    @property
    def roots(self) -> tuple[str, ...]:
        return tuple(
            name for name, clade in self.subclades.items() if clade.parent not in self.subclades
        )

    def ancestors(self, name: str) -> tuple[str, ...]:
        """Every ancestor of ``name``, nearest first. Unknown names are fatal."""
        self[name]
        return self._ancestors.get(name, ())

    def is_within(self, name: str, ancestor: str) -> bool:
        """True when ``name`` is ``ancestor`` or descends from it.

        This is how every selector should ask the question. Matching clade names as
        strings is what made a rule anchored on a parent clade silently match nothing
        once its children were renamed.
        """
        self[ancestor]
        if name == ancestor:
            return True
        return ancestor in self.ancestors(name)

    def descendants(self, name: str) -> tuple[str, ...]:
        """``name`` and every clade below it."""
        self[name]
        return tuple(other for other in self.subclades if self.is_within(other, name))

    def children(self, name: str) -> tuple[str, ...]:
        self[name]
        return tuple(other for other, clade in self.subclades.items() if clade.parent == name)

    def depth(self, name: str) -> int:
        """Number of ancestors; the deepest matching clade is the most specific one."""
        return len(self.ancestors(name))

    def is_local(self, name: str) -> bool:
        return name in self.local_names

    def upstream_depth(self, name: str) -> int:
        """Depth of the deepest published clade at or above ``name``; -1 if there is none.

        This is what stops a local clade overriding a published one. A local clade sitting
        under a published parent inherits that parent's standing and refines it; one
        attached at the root has no published ancestor, so it ranks below every published
        clade and can only be assigned where the nomenclature says nothing at all.
        """
        for candidate in (name, *self.ancestors(name)):
            if candidate not in self.local_names:
                return self.depth(candidate)
        return -1

    def cumulative(self, name: str) -> dict[tuple[Alphabet, int], str]:
        """The full signature of ``name``: its own branch's mutations and its ancestors'.

        A descendant's state at a position replaces an ancestor's, which is what makes
        the nomenclature's per-branch mutations into something a sequence can be tested
        against.
        """
        signature: dict[tuple[Alphabet, int], str] = {}
        for ancestor in reversed((name, *self.ancestors(name))):
            for position in self.subclades[ancestor].mutations:
                signature[(position.alphabet, position.position)] = position.state
        return signature

    def legacy_name(self, name: str) -> str | None:
        """The older display name for a subclade, if upstream records one.

        Taken from the subclade's own ``clade:`` field, else from a clade file that
        aliases it. Report figures have always shown these alongside the current name,
        so they are kept as labels — never as something to assign or select by.
        """
        clade = self[name]
        if clade.clade:
            return clade.clade
        for legacy in self.legacy_clades.values():
            if legacy.alias_of == name:
                return legacy.short_name or legacy.name
        return None

    def unexpressible(self) -> dict[str, tuple[Unexpressible, ...]]:
        """Clades with defining mutations no mature-HA sequence can carry (design rule 1)."""
        return {
            clade.name: clade.unexpressible
            for clade in self.subclades.values()
            if clade.unexpressible
        }


def load_clade_set(
    subtype: str,
    clones: Path,
    pin: Pin | None = None,
    *,
    repository: str | None = None,
    check_commit: bool = True,
) -> CladeSet:
    """Load ``subtype``'s subclades from the clone under ``clones``.

    ``clones`` is the directory holding the ``influenza-clade-nomenclature`` clones. The
    clone must be at ``pin.commit``; pass ``check_commit=False`` only in tests that build
    a directory by hand.
    """
    name = repository or (pin.repository if pin else HA_REPOSITORIES.get(subtype))
    if name is None:
        known = ", ".join(sorted(HA_REPOSITORIES))
        raise NomenclatureError(f"no upstream repository known for subtype {subtype!r}; {known}")
    root = Path(clones) / name
    if not root.is_dir():
        raise NomenclatureError(f"upstream clone not found: {root}")
    if pin is None:
        pin = Pin(subtype=subtype, repository=name, commit=head_commit(root))
    elif check_commit:
        _check_commit(root, pin)
    coordinates = coordinates_for(subtype)
    subclades = _read_directory(root / "subclades", coordinates, root)
    if not subclades:
        raise NomenclatureError(f"no subclade files under {root / 'subclades'}")
    legacy = _read_directory(root / "clades", coordinates, root)
    ancestors = _ancestry(subclades, root)
    return CladeSet(
        subtype=subtype,
        pin=pin,
        coordinates=coordinates,
        subclades=subclades,
        version=f"{name}@{pin.commit}",
        legacy_clades=legacy,
        _ancestors=ancestors,
    )


def head_commit(repository: Path) -> str:
    """The commit a clone is on. A directory that is not a git clone is fatal."""
    try:
        finished = subprocess.run(
            ["git", "-C", str(repository), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise NomenclatureError(f"cannot read the commit of {repository}: {error}") from error
    return finished.stdout.strip()


def _check_commit(repository: Path, pin: Pin) -> None:
    head = head_commit(repository)
    if not head.startswith(pin.commit) and not pin.commit.startswith(head):
        raise NomenclatureError(
            f"{repository} is at {head[:12]}, not the pinned {pin.commit[:12]}. "
            "Check out the pinned commit, or update the pin deliberately."
        )


def _read_directory(directory: Path, coordinates: Coordinates, root: Path) -> dict[str, Subclade]:
    """Read every ``*.yml`` in one directory. A missing directory is empty, not an error:
    the NA repositories have no ``clades/`` level."""
    if not directory.is_dir():
        return {}
    clades: dict[str, Subclade] = {}
    for path in sorted(directory.glob("*.yml")):
        clade = _subclade_from_file(path, coordinates, root)
        if clade.name in clades:
            raise NomenclatureError(f"{path}: duplicate clade name {clade.name!r}")
        clades[clade.name] = clade
    return clades


def _subclade_from_file(path: Path, coordinates: Coordinates, root: Path) -> Subclade:
    fields = _parse_yaml(path)
    name = _scalar(fields.get("name"))
    if not name:
        raise NomenclatureError(f"{path}: no name")
    mutations: list[Position] = []
    unexpressible: list[Unexpressible] = []
    declared = fields.get("defining_mutations") or []
    if not isinstance(declared, list):
        raise NomenclatureError(f"{path}: defining_mutations is not a list")
    for entry in declared:
        if not isinstance(entry, dict):
            raise NomenclatureError(f"{path}: malformed defining_mutations entry {entry!r}")
        missing = {"locus", "position", "state"} - set(entry)
        if missing:
            raise NomenclatureError(f"{path}: defining mutation missing {sorted(missing)}")
        try:
            position = int(entry["position"])
        except ValueError:
            raise NomenclatureError(
                f"{path}: position {entry['position']!r} is not a number"
            ) from None
        converted = convert(entry["locus"], position, entry["state"], coordinates)
        (mutations if isinstance(converted, Position) else unexpressible).append(converted)  # type: ignore[arg-type]
    return Subclade(
        name=name,
        parent=_scalar(fields.get("parent")),
        mutations=tuple(mutations),
        unexpressible=tuple(unexpressible),
        alias_of=_scalar(fields.get("alias_of")),
        unaliased_name=_scalar(fields.get("unaliased_name")),
        clade=_scalar(fields.get("clade")),
        short_name=_scalar(fields.get("short_name")),
        revoked=str(fields.get("revoked", "")).lower() == "true",
        comment=_scalar(fields.get("comment")),
        source=str(path.relative_to(root)),
    )


def _ancestry(subclades: Mapping[str, Subclade], root: Path) -> dict[str, tuple[str, ...]]:
    """Ancestors of every node, nearest first. A parent cycle is fatal."""
    ancestors: dict[str, tuple[str, ...]] = {}
    for name in subclades:
        chain: list[str] = []
        seen = {name}
        current = subclades[name].parent
        while current is not None and current in subclades:
            if current in seen:
                raise NomenclatureError(f"{root}: clade parentage forms a cycle at {current!r}")
            seen.add(current)
            chain.append(current)
            current = subclades[current].parent
        ancestors[name] = tuple(chain)
    return ancestors


def _scalar(value: object) -> str | None:
    """A YAML scalar, with upstream's ``none`` spellings turned into ``None``."""
    if value is None or isinstance(value, list):
        return None
    text = str(value).strip()
    return None if text in _NONE else text


def _parse_yaml(path: Path) -> dict[str, object]:
    """Parse one nomenclature YAML file. Anything unexpected raises, naming the line."""
    fields: dict[str, object] = {}
    current_list: list[dict[str, str]] | None = None
    current_item: dict[str, str] | None = None
    for number, raw in enumerate(path.read_text().splitlines(), start=1):
        line = raw.rstrip()
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        stripped = line.lstrip()
        indent = len(line) - len(stripped)
        if indent == 0 and not stripped.startswith("- "):
            key, separator, value = stripped.partition(":")
            key = key.strip()
            if not separator:
                raise NomenclatureError(f"{path}:{number}: expected 'key: value', got {raw!r}")
            if key not in _SCALAR_KEYS and key not in _LIST_KEYS:
                raise NomenclatureError(f"{path}:{number}: unknown key {key!r}")
            text = value.strip()
            if key in _LIST_KEYS:
                if text not in ("", "[]"):
                    raise NomenclatureError(f"{path}:{number}: expected a list under {key!r}")
                current_list = [] if text == "" else []
                fields[key] = current_list
                current_item = None
            else:
                fields[key] = _unquote(text)
                current_list, current_item = None, None
        elif stripped.startswith("- "):
            if current_list is None:
                raise NomenclatureError(f"{path}:{number}: list item outside a list: {raw!r}")
            current_item = {}
            current_list.append(current_item)
            key, separator, value = stripped[2:].partition(":")
            if not separator:
                raise NomenclatureError(f"{path}:{number}: expected 'key: value', got {raw!r}")
            current_item[key.strip()] = _unquote(value.strip())
        elif current_item is not None:
            key, separator, value = stripped.partition(":")
            if not separator:
                raise NomenclatureError(f"{path}:{number}: expected 'key: value', got {raw!r}")
            current_item[key.strip()] = _unquote(value.strip())
        else:
            raise NomenclatureError(f"{path}:{number}: unexpected line {raw!r}")
    return fields


def _unquote(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        return value[1:-1]
    return value


def load_clade_sets(
    clones: Path, pins: Iterable[Pin], *, check_commit: bool = True
) -> dict[str, CladeSet]:
    """Load every pinned subtype. Two pins for one subtype is a configuration error."""
    sets: dict[str, CladeSet] = {}
    for pin in pins:
        if pin.subtype in sets:
            raise NomenclatureError(f"two pins for subtype {pin.subtype!r}")
        sets[pin.subtype] = load_clade_set(
            pin.subtype, clones, pin, repository=pin.repository, check_commit=check_commit
        )
    return sets
