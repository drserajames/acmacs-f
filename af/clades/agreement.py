"""Refuse a tree whose clade calls disagree with Nextclade's beyond a set limit.

The tree engine and the fallback (Nextclade's stored call) are independent methods over one
pinned nomenclature. On correctly built trees they disagree on well under 1% of the leaves
both call; on a tree whose topology runs backwards inside a lineage they disagree on a third
of them, and the clade table built from it would relabel tens of thousands of sequences.
So a publish that has both compares them first, and refuses when the disagreement exceeds a
limit (Sarah, 29 Sep 2026: 2%).

**Disagreement** is :func:`af.clades.fallback.disagreements`: a different lineage, or a tree
call *shallower* than Nextclade's. A deeper tree call is not counted — being more specific is
what the tree engine is for, and a tree may correctly label a clade newer than the Nextclade
dataset. The fraction is over the leaves both engines call.

The limit is data, one row per subtype with its reason (``clades/agreement.tsv`` in the data
repo), read by the caller: this package reads no config. A publish over the limit can still
go ahead with a named reason (``accept_disagreement``), which is written into the version's
provenance and report — never silently (design rule 9).
"""

from __future__ import annotations

import csv
import dataclasses
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from af.clades.assign import Assignment
from af.clades.fallback import disagreements
from af.clades.nomenclature import CladeSet

COLUMNS = ("subtype", "max_disagreement", "reason")
#: How many of the commonest disagreements an error and a report name.
TOP_TRANSITIONS = 10


class AgreementError(ValueError):
    """The limits file is malformed, or a tree disagrees with the fallback beyond its limit."""


@dataclass(frozen=True)
class AgreementLimit:
    """The largest fraction of disagreeing leaves a tree-based publish accepts, and why."""

    subtype: str
    max_disagreement: float
    reason: str

    def to_json(self) -> dict[str, Any]:
        return {"max_disagreement": self.max_disagreement, "reason": self.reason}


@dataclass(frozen=True)
class AgreementCheck:
    """What the comparison found, for the version's report and provenance."""

    limit: AgreementLimit
    compared: int
    disagree: int
    transitions: tuple[tuple[str | None, str | None, int], ...]
    """The commonest disagreements as (fallback clade, tree clade, leaves), most first."""
    accepted: str | None = None
    """The reason given to publish over the limit, as given (``exceeded`` says if it was needed)."""

    @property
    def fraction(self) -> float:
        return self.disagree / self.compared

    @property
    def exceeded(self) -> bool:
        return self.fraction > self.limit.max_disagreement

    def provenance(self) -> dict[str, Any]:
        """What decided that the version may exist: the limit, its reason, any override."""
        return {**self.limit.to_json(), "accepted": self.accepted}

    def to_json(self) -> dict[str, Any]:
        return {
            **self.limit.to_json(),
            "compared": self.compared,
            "disagree": self.disagree,
            "fraction": round(self.fraction, 6),
            "exceeded": self.exceeded,
            "accepted": self.accepted,
            "top_transitions": [
                {"fallback": fallback, "tree": tree, "leaves": leaves}
                for fallback, tree, leaves in self.transitions
            ],
        }


def load_agreement_limits(path: Path) -> dict[str, AgreementLimit]:
    """Read ``subtype, max_disagreement, reason`` rows; every problem is reported at once.

    A row without a reason is refused: a limit nobody can explain cannot be revisited
    (design rule 11).
    """
    path = Path(path)
    if not path.is_file():
        raise AgreementError(f"agreement limits not found: {path}")
    problems: list[str] = []
    limits: dict[str, AgreementLimit] = {}
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if tuple(reader.fieldnames or ()) != COLUMNS:
            raise AgreementError(f"{path}: columns must be {', '.join(COLUMNS)}")
        for line, row in enumerate(reader, start=2):
            subtype = (row["subtype"] or "").strip()
            reason = (row["reason"] or "").strip()
            raw = (row["max_disagreement"] or "").strip()
            where = f"{path} line {line}"
            if not subtype:
                problems.append(f"{where}: no subtype")
                continue
            if subtype in limits:
                problems.append(f"{where}: {subtype!r} is listed twice")
                continue
            try:
                value = float(raw)
            except ValueError:
                problems.append(f"{where}: max_disagreement {raw!r} is not a number")
                continue
            if not 0 <= value < 1:
                problems.append(f"{where}: max_disagreement {value} is not a fraction in [0, 1)")
                continue
            if not reason:
                problems.append(f"{where}: {subtype!r} has no reason")
                continue
            limits[subtype] = AgreementLimit(subtype, value, reason)
    if problems:
        raise AgreementError("\n".join(problems))
    if not limits:
        raise AgreementError(f"{path}: no limits")
    return limits


def limit_for(limits: Mapping[str, AgreementLimit], subtype: str) -> AgreementLimit:
    """The subtype's limit; a subtype without one is an error, never "no limit"."""
    try:
        return limits[subtype]
    except KeyError:
        raise AgreementError(
            f"no agreement limit for {subtype!r} (there are: {', '.join(sorted(limits))})"
        ) from None


def check_agreement(
    tree_calls: Mapping[Any, Assignment],
    fallback_calls: Mapping[Any, Assignment],
    clade_set: CladeSet,
    limit: AgreementLimit,
    *,
    accept_disagreement: str | None = None,
) -> AgreementCheck:
    """Compare the two engines on the leaves both call; refuse over the limit unless accepted.

    ``fallback_calls`` are the fallback's calls for tree leaves only. No overlap at all is an
    error: a tree none of whose leaves Nextclade called cannot be checked (design rule 1).
    """
    if accept_disagreement is not None and not accept_disagreement.strip():
        raise AgreementError("accept_disagreement needs a reason, not an empty string")
    compared = [key for key in tree_calls if key in fallback_calls]
    if not compared:
        raise AgreementError(
            f"{limit.subtype}: no tree leaf has a fallback call, so the tree cannot be checked"
        )
    disagreeing = disagreements(
        {key: tree_calls[key] for key in compared},
        {key: fallback_calls[key] for key in compared},
        clade_set,
    )
    counts = Counter((fallback, tree) for tree, fallback in disagreeing.values())
    check = AgreementCheck(
        limit,
        len(compared),
        len(disagreeing),
        tuple((fb, tree, n) for (fb, tree), n in counts.most_common(TOP_TRANSITIONS)),
    )
    if check.exceeded and accept_disagreement is None:
        transitions = ", ".join(f"{fb} -> {tree} {n:,}" for fb, tree, n in check.transitions)
        raise AgreementError(
            f"{limit.subtype}: the tree disagrees with the fallback on {check.disagree:,} of "
            f"{check.compared:,} leaves ({100 * check.fraction:.2f}%), over the limit of "
            f"{100 * limit.max_disagreement:g}% ({limit.reason}). Commonest (fallback -> tree): "
            f"{transitions}. A disagreement this large means the tree is wrong (topology, "
            "rooting or ancestral states), not that the clades moved; to publish anyway, give "
            "accept_disagreement='<reason>'."
        )
    return dataclasses.replace(check, accepted=accept_disagreement)
