"""A retrospective label for viruses the subclade nomenclature does not name (task 4.10).

Upstream's subclades begin at a recent root, so an older lineage gets no subclade: by design,
not by fault. Upstream also publishes the older ("legacy") clades, and for some subtypes those
files carry their own defining mutations. Sarah chose (Q85, 30 Sep 2026) to label such viruses
from those definitions, **only where the subclade is null**, for display and colouring. The
label is never something to assign or select by: ``clade`` stays the one label for that.

The label is a signature match, which is fragile on old definitions in one known way: a site
that changed later in a lineage contradicts an *ancestor's* marker without unmaking the lineage
(a later change at 216 inside a lineage defined below the clade that introduced 216T). So one
stated, counted tolerance applies, read from data with its reason:

* every defining locus of the labelled clade's **own** branch must MATCH;
* at most ``max_ancestor_contradictions`` loci of its legacy **ancestors** may fail to match
  (contradicted, or unobservable — neither is evidence *for* the lineage), each recorded;
* ``tolerate_root`` says whether a locus of the root legacy clade may be among them (with
  ``false``, a failed root locus disqualifies, however many others are allowed).

The deepest candidate wins. Two equally deep candidates that are not nested are labelled with
their deepest common legacy ancestor (none: no label), and counted.
"""

from __future__ import annotations

import csv
from collections import Counter
from collections.abc import Hashable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

from af.clades.nomenclature import CladeSet
from af.clades.sequence import AlignedSequence, Evidence

K = TypeVar("K", bound=Hashable)

COLUMNS = ("subtype", "max_ancestor_contradictions", "tolerate_root", "reason")
_TRUE, _FALSE = {"true", "yes", "1"}, {"false", "no", "0"}


class LegacyError(ValueError):
    """The legacy settings are malformed, or a subtype has none."""


@dataclass(frozen=True)
class LegacyRule:
    """How far a legacy label may tolerate a changed ancestral site, and why."""

    subtype: str
    max_ancestor_contradictions: int
    tolerate_root: bool
    reason: str

    def to_json(self) -> dict[str, Any]:
        return {
            "max_ancestor_contradictions": self.max_ancestor_contradictions,
            "tolerate_root": self.tolerate_root,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class LegacyCall:
    """One sequence's legacy label (None: none), and the ancestral loci it was forgiven."""

    clade: str | None
    tolerated: tuple[str, ...] = ()
    ambiguous: tuple[str, ...] = ()
    """The tied candidates, when the label is their common ancestor (or None)."""


@dataclass
class LegacyCounts:
    labelled: int = 0
    strict: int = 0
    tolerated: Counter[str] = field(default_factory=Counter)
    ambiguous: Counter[str] = field(default_factory=Counter)
    none: int = 0
    no_sequence: int = 0

    def to_json(self) -> dict[str, Any]:
        return {
            "labelled": self.labelled,
            "strict": self.strict,
            "tolerated_loci": dict(self.tolerated.most_common()),
            "ambiguous": dict(self.ambiguous.most_common()),
            "none": self.none,
            "no_aligned_sequence": self.no_sequence,
        }


def load_legacy_rules(path: Path) -> dict[str, LegacyRule]:
    """Read ``subtype, max_ancestor_contradictions, tolerate_root, reason`` rows.

    Every problem is reported at once; a row without a reason is refused (design rule 11).
    """
    path = Path(path)
    if not path.is_file():
        raise LegacyError(f"legacy rules not found: {path}")
    problems: list[str] = []
    rules: dict[str, LegacyRule] = {}
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if tuple(reader.fieldnames or ()) != COLUMNS:
            raise LegacyError(f"{path}: columns must be {', '.join(COLUMNS)}")
        for line, row in enumerate(reader, start=2):
            where = f"{path} line {line}"
            subtype = (row["subtype"] or "").strip()
            reason = (row["reason"] or "").strip()
            raw = (row["max_ancestor_contradictions"] or "").strip()
            flag = (row["tolerate_root"] or "").strip().lower()
            if not subtype:
                problems.append(f"{where}: no subtype")
            elif subtype in rules:
                problems.append(f"{where}: {subtype!r} is listed twice")
            elif not raw.isdigit():
                problems.append(f"{where}: max_ancestor_contradictions {raw!r} is not a count")
            elif flag not in _TRUE | _FALSE:
                problems.append(f"{where}: tolerate_root {flag!r} is not true or false")
            elif not reason:
                problems.append(f"{where}: {subtype!r} has no reason")
            else:
                rules[subtype] = LegacyRule(subtype, int(raw), flag in _TRUE, reason)
    if problems:
        raise LegacyError("\n".join(problems))
    if not rules:
        raise LegacyError(f"{path}: no rules")
    return rules


def rule_for(rules: Mapping[str, LegacyRule], subtype: str) -> LegacyRule:
    try:
        return rules[subtype]
    except KeyError:
        raise LegacyError(
            f"no legacy rule for {subtype!r} (there are: {', '.join(sorted(rules))})"
        ) from None


def legacy_label(sequence: AlignedSequence, clade_set: CladeSet, rule: LegacyRule) -> LegacyCall:
    """The deepest legacy clade the sequence belongs to under ``rule``."""
    candidates: dict[str, tuple[str, ...]] = {}
    for name in clade_set.legacy_defining():
        forgiven = _forgiven(sequence, clade_set, name, rule)
        if forgiven is not None:
            candidates[name] = forgiven
    if not candidates:
        return LegacyCall(None)
    depth = {name: len(clade_set.legacy_ancestors(name)) for name in candidates}
    deepest = max(depth.values())
    top = sorted(name for name in candidates if depth[name] == deepest)
    if len(top) == 1:
        return LegacyCall(top[0], candidates[top[0]])
    return LegacyCall(clade_set.legacy_common_ancestor(top), (), tuple(top))


def _forgiven(
    sequence: AlignedSequence, clade_set: CladeSet, name: str, rule: LegacyRule
) -> tuple[str, ...] | None:
    """The ancestral loci ``name`` needs forgiven; None when it is not a candidate at all.

    Its own loci must all match. An ancestor's locus that does not match is forgiven, up to
    the rule's budget, unless it is the root's and the rule does not tolerate the root.
    """
    if not all(_matches(sequence, m) for m in clade_set.legacy_clades[name].mutations):
        return None
    failed: list[str] = []
    for ancestor in clade_set.legacy_ancestors(name):
        is_root = clade_set.legacy_parent(ancestor) is None
        for mutation in clade_set.legacy_clades[ancestor].mutations:
            if _matches(sequence, mutation):
                continue
            if is_root and not rule.tolerate_root:
                return None
            failed.append(f"{ancestor}:{_locus(mutation)}")
            if len(failed) > rule.max_ancestor_contradictions:
                return None
    return tuple(failed)


def legacy_labels(
    sequences: Mapping[K, AlignedSequence | None],
    clade_set: CladeSet,
    rule: LegacyRule,
) -> tuple[dict[K, LegacyCall], LegacyCounts]:
    """Label every sequence; one with no aligned sequence (None) is counted, not labelled."""
    counts = LegacyCounts()
    calls: dict[K, LegacyCall] = {}
    for key, sequence in sequences.items():
        if sequence is None:
            counts.no_sequence += 1
            calls[key] = LegacyCall(None)
            continue
        call = legacy_label(sequence, clade_set, rule)
        calls[key] = call
        if call.ambiguous:
            counts.ambiguous[" / ".join(call.ambiguous)] += 1
        if call.clade is None:
            counts.none += 1
            continue
        counts.labelled += 1
        if call.tolerated:
            counts.tolerated.update(call.tolerated)
        else:
            counts.strict += 1
    return calls, counts


def _matches(sequence: AlignedSequence, mutation: Any) -> bool:
    return (
        sequence.evidence(mutation.alphabet, mutation.position, mutation.state) is Evidence.MATCHES
    )


def _locus(mutation: Any) -> str:
    prefix = "" if mutation.alphabet == "aa" else "nuc"
    return f"{prefix}{mutation.position}{mutation.state}"
