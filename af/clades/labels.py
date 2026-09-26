"""Resolve clade labels written by other systems onto the upstream canonical scheme.

Comparing af's figures with the ones they replace needs both sides' clade labels at one
granularity, and the old labels come in several spellings of the same fact: a current
subclade name, a subclade with its older display name in brackets, the older name alone,
a name only the local layer defines. This module is the one place that knows how those
spellings relate, and it reads every relation from the clade set — the pinned upstream
YAML plus ``clades/local.tsv`` — so no table of names exists anywhere else.

A label resolves to one upstream subclade name, or to ``None`` when it names something
the subclade hierarchy does not contain. The routes, tried in order:

``subclade``
    The label is an upstream subclade name.
``revoked``
    The label is a subclade upstream has revoked and renamed. It resolves to the live
    subclade with the same ``unaliased_name`` — the same clade under its current name
    (upstream renames an alias, and its children, this way). The structured
    field is used, not the revocation comment, which can be wrong. Also applied inside the
    ``display`` form and to the older names a revoked subclade claims.
``local``
    The label is a local clade (:mod:`af.clades.local`); it folds to its deepest upstream
    ancestor, because a local name refines a published clade and never replaces it. A
    local clade with no upstream ancestor resolves to ``None``.
``display``
    ``"<name> (<legacy>)"``, the way figures have always written a clade. The bracket must
    be *that* clade's legacy name: a bracket naming anything else is legend text, and
    guessing it away would hide a real mismatch.
``legacy``
    An older name alone. Upstream records it on the subclade (its ``clade:`` field) or as
    a legacy clade file aliasing the subclade.
``outside``
    An older name upstream still publishes but aliases to no subclade: a lineage outside
    the subclade hierarchy, typically ancestral to its root. It resolves to ``None``, as
    af's engine labels those viruses.

Anything else is unmapped, and unmapped labels are an error unless the caller asks for
them to be reported instead (design rule 1). Names are never matched by prefix: whether
one clade is within another is :meth:`CladeSet.is_within`'s question, not this module's.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

from af.clades.nomenclature import CladeSet

ROUTES = ("unnamed", "subclade", "revoked", "local", "display", "legacy", "outside")

#: ``name (bracketed)`` with exactly one bracket, as figures write a clade and its old name.
_DISPLAY = re.compile(r"^(?P<name>[^\s()]+) \((?P<bracket>[^()]+)\)$")


class LabelError(ValueError):
    """Labels that resolve to no clade, or a clade set whose older names collide."""


@dataclass(frozen=True)
class Resolved:
    """One label's upstream canonical clade, and the route that got there."""

    label: str | None
    clade: str | None
    route: str


@dataclass(frozen=True)
class LabelMap:
    """Every label given, resolved or not, against one clade-set version."""

    clade_set_version: str
    resolved: Mapping[str | None, Resolved]
    unmapped: Mapping[str, str] = field(default_factory=dict)
    """label -> why it did not resolve"""

    def clade(self, label: str | None) -> str | None:
        try:
            return self.resolved[label].clade
        except KeyError:
            raise KeyError(f"label {label!r} was not resolved") from None

    def counts(self) -> dict[str, int]:
        """Labels per route, plus ``unmapped``: what a comparison report should print."""
        counts = {route: 0 for route in ROUTES}
        for item in self.resolved.values():
            counts[item.route] += 1
        counts["unmapped"] = len(self.unmapped)
        return counts


def canonical_labels(
    labels: Iterable[str | None], clade_set: CladeSet, *, allow_unmapped: bool = False
) -> LabelMap:
    """Resolve each distinct label; see the module docstring for the routes.

    ``None`` stands for "no label" and resolves to ``None``. With ``allow_unmapped`` the
    unresolved labels are returned in :attr:`LabelMap.unmapped` for the caller to report;
    without it they are fatal, all listed together.
    """
    older = older_names(clade_set)
    resolved: dict[str | None, Resolved] = {}
    unmapped: dict[str, str] = {}
    for label in dict.fromkeys(labels):
        if label is None:
            resolved[None] = Resolved(None, None, "unnamed")
            continue
        outcome = _resolve(label, clade_set, older)
        if isinstance(outcome, Resolved):
            resolved[label] = outcome
        else:
            unmapped[label] = outcome
    if unmapped and not allow_unmapped:
        lines = "\n".join(f"  {label!r}: {why}" for label, why in sorted(unmapped.items()))
        raise LabelError(
            f"{len(unmapped)} label(s) name no clade of {clade_set.subtype} at "
            f"{clade_set.version}:\n{lines}"
        )
    return LabelMap(clade_set.version, resolved, unmapped)


def older_names(clade_set: CladeSet) -> dict[str, tuple[str, ...] | None]:
    """Every older name upstream publishes -> the subclade(s) claiming it (``None``: none).

    Two sources, both upstream's: a subclade's own ``clade:`` field, and the legacy clade
    files' ``alias_of``, keyed by their full name (their short names, such as a bare digit,
    are display abbreviations too short to identify anything).

    Claims by a revoked subclade count for its successor (:func:`successor`), so a clade
    renamed upstream claims its older name once, under its current name. A name still
    claimed by more than one subclade is ambiguous at the canonical granularity, and
    :func:`canonical_labels` reports it rather than choosing.
    A legacy file without an alias is the old clade's own definition, not a pointer; it
    maps to ``None`` ("outside") only when no subclade claims the name. An alias naming a
    clade that does not exist is a defect in the clade set and is fatal.
    """
    claims: dict[str, set[str]] = {}
    unaliased: list[str] = []
    problems: list[str] = []
    for subclade in clade_set:
        if subclade.clade and subclade.clade not in clade_set:
            current = successor(subclade.name, clade_set)
            if current is not None:  # a revoked clade with no successor claims nothing
                claims.setdefault(subclade.clade, set()).add(current)
    for legacy in clade_set.legacy_clades.values():
        if legacy.name in clade_set:
            continue  # a current subclade name always means that subclade
        if legacy.alias_of is None:
            unaliased.append(legacy.name)
        elif legacy.alias_of not in clade_set:
            problems.append(f"{legacy.name!r} aliases {legacy.alias_of!r}, which is not defined")
        else:
            current = successor(legacy.alias_of, clade_set)
            if current is not None:
                claims.setdefault(legacy.name, set()).add(current)
    if problems:
        raise LabelError(f"{clade_set.version}: " + "; ".join(problems))
    older: dict[str, tuple[str, ...] | None] = {
        name: tuple(sorted(targets)) for name, targets in claims.items()
    }
    for name in unaliased:
        older.setdefault(name, None)
    return older


def _resolve(
    label: str, clade_set: CladeSet, older: Mapping[str, tuple[str, ...] | None]
) -> Resolved | str:
    """The label resolved, or the reason it is not."""
    if label in clade_set:
        if clade_set.is_local(label):
            return Resolved(label, _upstream_anchor(label, clade_set), "local")
        if clade_set[label].revoked:
            current = successor(label, clade_set)
            if current is None:
                return f"{label!r} is revoked and no live subclade has its unaliased name"
            return Resolved(label, current, "revoked")
        return Resolved(label, label, "subclade")
    display = _DISPLAY.fullmatch(label)
    if display is not None:
        name, bracket = display["name"], display["bracket"]
        if name not in clade_set:
            return f"{name!r} is not a clade"
        expected = clade_set.legacy_name(name)
        if bracket != expected:
            return f"the bracket {bracket!r} is not {name!r}'s older name ({expected!r})"
        current = successor(name, clade_set)
        if current is None:
            return f"{name!r} is revoked and no live subclade has its unaliased name"
        return Resolved(label, _upstream_anchor(current, clade_set), "display")
    if label in older:
        targets = older[label]
        if targets is None:
            return Resolved(label, None, "outside")
        if len(targets) > 1:
            return f"an older name shared by {', '.join(targets)}; it names none of them alone"
        return Resolved(label, targets[0], "legacy")
    return "not a subclade, a local clade, or an older name upstream publishes"


def successor(name: str, clade_set: CladeSet) -> str | None:
    """``name``, or for a revoked subclade the live one with the same ``unaliased_name``.

    ``None`` when a revoked subclade has no such successor, or more than one: then there is
    no single current name for it, and guessing one would hide that.
    """
    clade = clade_set[name]
    if not clade.revoked:
        return name
    if not clade.unaliased_name:
        return None
    live = [
        other.name
        for other in clade_set
        if not other.revoked and other.unaliased_name == clade.unaliased_name
    ]
    return live[0] if len(live) == 1 else None


def _upstream_anchor(name: str, clade_set: CladeSet) -> str | None:
    """``name`` itself if upstream publishes it, else its nearest upstream ancestor."""
    for candidate in (name, *clade_set.ancestors(name)):
        if not clade_set.is_local(candidate):
            return candidate
    return None
