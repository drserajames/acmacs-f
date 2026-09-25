"""Dropping sequences before the tree is built (Sarah, 25 Sep 2026).

    "Drop long branches before making trees as they can influence topology & speed.
     report all. further exclusions in the round"

So the long-branch rule is not a figure-time hide, as ae's ``.tal`` edge rule was, and not a
post-build flag: the sequences leave the **alignment**, before CMAPLE sees it. Clock outliers are
reported only, and any further exclusion is a round's decision, named per reason in its config.

**Pre-build there is no tree**, so a branch length has to come from a tree built earlier. There are
two sources, and both reduce to the same function of a tree:

* the **previous cycle's tree**, for an incremental build — what production does weekly, and the
  normal case;
* a **first-pass tree**, for a from-scratch build or a replay: build once, measure, drop, rebuild.

A sequence that is in neither — genuinely new, never yet placed — has no branch length and cannot
be judged here. It is kept, and judged at the next build. That is the same fail-safe direction as
the cut node (``notes/trees/CUT-NODE.md`` §3c): never drop a virus af has not already looked at.

Every dropped key is counted and named, with the threshold and the tree it was measured on, so a
reader of the figure can always be told what left and why.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable
from dataclasses import dataclass, field
from pathlib import Path

from af.tree.io import fasta
from af.tree.model import Tree

LONG_BRANCH = "long_branch"

#: ae's `.tal` edge rule, in substitutions per site. Measured against the September round's H1
#: hand hides: it flags 9 leaves and all 9 are hand-hidden (notes/trees/CLOCK.md).
DEFAULT_LONG_BRANCH = 0.01


class PrebuildError(ValueError):
    """The exclusion cannot be applied to this alignment."""


@dataclass(frozen=True)
class ExclusionRule:
    """One reason sequences are dropped before the build, with the reason recorded as data."""

    reason: str
    threshold: float
    why: str
    """Why this threshold, in words. Design rule: a rule carries its justification, not just a
    number, because the number outlives whoever chose it."""


LONG_BRANCH_RULE = ExclusionRule(
    reason=LONG_BRANCH,
    threshold=DEFAULT_LONG_BRANCH,
    why=(
        "ae's .tal edge rule. A leaf on a long branch of its own is unlike anything else in the "
        "tree and drags the topology; Sarah, 25 Sep 2026: drop before building, for topology and "
        "speed. Measured on the 2026-0921 H1 tree: 9 flagged, 9 of 9 hand-hidden."
    ),
)


@dataclass
class ExclusionPlan:
    """What will be dropped, why, and from which tree it was measured."""

    drop: dict[str, str] = field(default_factory=dict)
    """leaf key -> reason."""

    lengths: dict[str, float] = field(default_factory=dict)
    counts: dict[str, object] = field(default_factory=dict)

    def keys(self) -> set[str]:
        return set(self.drop)

    def by_reason(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for key, reason in sorted(self.drop.items()):
            out.setdefault(reason, []).append(key)
        return out


def long_branch_leaves(tree: Tree, threshold: float = DEFAULT_LONG_BRANCH) -> dict[str, float]:
    """Leaf key -> terminal branch length, for leaves at or over ``threshold``."""
    return {
        leaf.name or "": leaf.branch_length
        for leaf in tree.leaves()
        if leaf.branch_length >= threshold
    }


def plan(
    tree: Tree,
    rule: ExclusionRule = LONG_BRANCH_RULE,
    keep: Collection[str] = (),
    source: str = "previous tree",
) -> ExclusionPlan:
    """Which sequences to leave out of the next build, measured on ``tree``.

    ``keep`` is exempt whatever its branch length — the outgroup belongs there, since the build has
    nothing to root on without it (``notes/trees/CUT-NODE.md`` §3d).
    """
    if rule.reason != LONG_BRANCH:
        raise PrebuildError(f"no pre-build rule named {rule.reason!r}")
    exempt = set(keep)
    found = long_branch_leaves(tree, rule.threshold)
    result = ExclusionPlan()
    for key, length in sorted(found.items()):
        if key in exempt:
            continue
        result.drop[key] = rule.reason
        result.lengths[key] = length
    longest = max((leaf.branch_length for leaf in tree.leaves()), default=0.0)
    leaves, _ = tree.count()
    result.counts = {
        "rule": rule.reason,
        "threshold": rule.threshold,
        "why": rule.why,
        "measured_on": source,
        "leaves_measured": leaves,
        "dropped": len(result.drop),
        "exempt_but_over_threshold": sorted(set(found) & exempt),
        "longest_terminal_branch": longest,
    }
    # An inert threshold must be visible, not implied. On the `mutations` scale a branch is
    # (changes / alignment length), so the longest one possible is small: on the 2026-0921 H3 tree
    # the maximum is 16/1650 = 0.0097, just under ae's 0.01, and the rule can never fire. ae's
    # threshold was set against ML lengths, and it does not carry over to the other scale.
    if rule.threshold > longest:
        result.counts["inert"] = (
            f"threshold {rule.threshold:g} is above the longest terminal branch in this tree "
            f"({longest:g}), so it cannot fire; is it the right scale for these lengths?"
        )
    return result


def filter_alignment(source: Path, target: Path, drop: Collection[str]) -> dict[str, object]:
    """Copy ``source`` to ``target`` without the dropped keys. Returns counts.

    A key that is not in the alignment is an **error**, not a silent no-op (design rule 1): it means
    the plan was measured against a different dataset, and quietly building the full tree would hide
    that. Writing an alignment with fewer than two sequences is refused for the same reason.
    """
    wanted = set(drop)
    kept: dict[str, str] = {}
    seen: set[str] = set()
    for name, sequence in fasta.iter_fasta(Path(source)):
        seen.add(name)
        if name not in wanted:
            kept[name] = sequence
    missing = wanted - seen
    if missing:
        raise PrebuildError(
            f"{len(missing)} sequences to drop are not in {source}, e.g. {sorted(missing)[:3]}; "
            "the exclusion plan was measured on a different dataset"
        )
    if len(kept) < 2:
        raise PrebuildError(f"{source}: dropping would leave {len(kept)} sequences")
    fasta.write_alignment(Path(target), kept)
    return {"sequences_in": len(seen), "dropped": len(seen) - len(kept), "sequences_out": len(kept)}


def apply_plan(source: Path, target: Path, plan_: ExclusionPlan) -> dict[str, object]:
    """:func:`filter_alignment` for a plan, merging both sets of counts."""
    counts = dict(plan_.counts)
    counts.update(filter_alignment(source, target, plan_.keys()))
    for reason, keys in plan_.by_reason().items():
        counts[f"dropped_{reason}"] = len(keys)
    return counts


def names_dropped(plan_: ExclusionPlan, names: Iterable[tuple[str, str]]) -> list[str]:
    """Report the dropped sequences by strain name, for the round's written record.

    ``names`` is (leaf key, strain name). Reporting by key alone is not enough for a person
    checking what left the tree.
    """
    by_key = dict(names)
    return sorted(f"{by_key.get(key, '?')} ({plan_.lengths[key]:.4g})" for key in plan_.drop)


__all__ = [
    "DEFAULT_LONG_BRANCH",
    "LONG_BRANCH",
    "LONG_BRANCH_RULE",
    "ExclusionPlan",
    "ExclusionRule",
    "PrebuildError",
    "apply_plan",
    "filter_alignment",
    "long_branch_leaves",
    "names_dropped",
    "plan",
]
