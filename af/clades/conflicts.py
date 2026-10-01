"""Refuse a tree whose clade labels hide a large sibling conflict (Sarah, 1 Oct 2026).

The engine cannot recover a sibling clade once its walk has gone down the wrong branch
(:mod:`af.clades.assign`, known limitation), so a wrongly reconstructed ancestor can hand a
whole clade's leaves to its sister, silently. :func:`af.clades.assign.sibling_conflicts`
finds such nodes; the tree's metadata carries their review (``counts.clade_review``). A
publish from a tree is refused when its largest *topmost* conflict covers at least the
subtype's limit of leaves.

Measured on five trees before Sarah chose the limit (``notes/clades/GUARDS-MEASURED.md``):
the largest topmost conflict was at most 83 leaves on the three reference trees, 3,700 on a
tree that lost a clade, and 23,024 on an inverted one. The limit is data, one row per
subtype with its reason and evidence, read by the caller (this package reads no config).
A named ``accept_conflict`` reason publishes anyway and is recorded.
"""

from __future__ import annotations

import csv
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

COLUMNS = ("subtype", "max_topmost_conflict_leaves", "reason", "evidence")


class ConflictError(ValueError):
    """The limits file is malformed, a tree has no review, or a conflict is over the limit."""


@dataclass(frozen=True)
class ConflictLimit:
    """The largest topmost sibling conflict (in leaves) a tree-based publish accepts."""

    subtype: str
    max_topmost_conflict_leaves: int
    reason: str
    evidence: str

    def to_json(self) -> dict[str, Any]:
        return {
            "max_topmost_conflict_leaves": self.max_topmost_conflict_leaves,
            "reason": self.reason,
            "evidence": self.evidence,
        }


def load_conflict_limits(path: Path) -> dict[str, ConflictLimit]:
    """Read ``subtype, max_topmost_conflict_leaves, reason, evidence``; all problems at once.

    A row without a reason or evidence is refused: a threshold must say why it is where it
    is, and what it was measured on, so it can be revisited (design rule 11).
    """
    path = Path(path)
    if not path.is_file():
        raise ConflictError(f"sibling-conflict limits not found: {path}")
    problems: list[str] = []
    limits: dict[str, ConflictLimit] = {}
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if tuple(reader.fieldnames or ()) != COLUMNS:
            raise ConflictError(f"{path}: columns must be {', '.join(COLUMNS)}")
        for line, row in enumerate(reader, start=2):
            where = f"{path} line {line}"
            subtype = (row["subtype"] or "").strip()
            raw = (row["max_topmost_conflict_leaves"] or "").strip()
            reason = (row["reason"] or "").strip()
            evidence = (row["evidence"] or "").strip()
            if not subtype:
                problems.append(f"{where}: no subtype")
            elif subtype in limits:
                problems.append(f"{where}: {subtype!r} is listed twice")
            elif not raw.isdigit() or int(raw) < 1:
                problems.append(f"{where}: max_topmost_conflict_leaves {raw!r} is not a count >= 1")
            elif not reason or not evidence:
                problems.append(f"{where}: {subtype!r} needs both a reason and its evidence")
            else:
                limits[subtype] = ConflictLimit(subtype, int(raw), reason, evidence)
    if problems:
        raise ConflictError("\n".join(problems))
    if not limits:
        raise ConflictError(f"{path}: no limits")
    return limits


def conflict_limit_for(limits: Mapping[str, ConflictLimit], subtype: str) -> ConflictLimit:
    try:
        return limits[subtype]
    except KeyError:
        raise ConflictError(
            f"no sibling-conflict limit for {subtype!r} (there are: {', '.join(sorted(limits))})"
        ) from None


def check_conflicts(
    review: Mapping[str, Any] | None,
    limit: ConflictLimit,
    tree: str,
    *,
    accept_conflict: str | None = None,
) -> dict[str, Any]:
    """The review checked against the limit, for the report; over it, refused unless accepted.

    A tree with clade calls but no review was labelled before the review existed, or by an
    engine that does not make one: it is refused rather than treated as conflict-free.
    """
    if accept_conflict is not None and not accept_conflict.strip():
        raise ConflictError("accept_conflict needs a reason, not an empty string")
    if review is None:
        raise ConflictError(
            f"{tree}: no sibling-conflict review in its metadata (counts.clade_review); "
            "it cannot be checked, so it is not published"
        )
    over = [
        c for c in review.get("topmost", []) if c["leaves"] >= limit.max_topmost_conflict_leaves
    ]
    result = {**limit.to_json(), "over_limit": len(over), "accepted": accept_conflict}
    if over and accept_conflict is None:
        listed = "; ".join(
            f"node {c['node']} labelled {c['label']} also matches {c['other']} "
            f"over {c['leaves']:,} leaves"
            for c in over[:5]
        )
        # the list is the largest topmost conflicts first, so ones over the limit can lie beyond
        # it only when even its smallest entry is over the limit
        kept = review.get("topmost", [])
        beyond = bool(review.get("topmost_truncated")) and len(over) == len(kept)
        count = f"at least {len(over)}" if beyond else str(len(over))
        raise ConflictError(
            f"{limit.subtype}: {tree} has {count} sibling conflict(s) covering at least "
            f"{limit.max_topmost_conflict_leaves:,} leaves: {listed}. The tree's ancestral "
            "states put a clade's marker on the wrong branch, and the engine cannot recover the "
            f"sibling (af.clades.assign). Limit: {limit.reason}. To publish anyway, give "
            "accept_conflict='<reason>'."
        )
    return result
