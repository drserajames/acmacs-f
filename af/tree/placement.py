"""Refuse a tree whose leaves are not placed where their sequences say they belong.

For each leaf, the path through the tree to the outgroup should be about as long as the raw
difference between the two sequences. Patristic path length does not depend on where the tree is
rooted, so the ratio of the two measures *placement*, and the **correlation** between them over a
sample of leaves is what separates a sound tree from a broken one. Measured (30 Sep 2026,
``notes/trees/PLACEMENT-GUARD.md``): ae's three round trees 0.966, 0.987 and 0.903; af's seeded H3
tree 0.982; af's from-scratch H3 tree, which placed the pre-2023 lineages inside the modern
radiation, **0.206**.

The median ratio does not discriminate (1.25 against ae's 1.08-1.15): a minority of badly placed
leaves barely moves a median over 193,185 of them. So the correlation is the refusal statistic, and
the median, p95, maximum and the per-date-band table are reported beside it, never refused on.

The limit is **data**, one row per subtype with its reason and how many trees it rests on
(``trees/placement.tsv`` in the data repo, read by the caller: this module reads no config), as
Sarah settled on 30 Sep 2026 — A(H3N2) 0.85, A(H1N1) 0.85, B/Vic 0.75. Two of those rest on a
single tree each and say so; they are to be revisited when that subtype has an af-built tree.

This is a different question from the clock guard (:func:`af.tree.clock.check_clock_direction`),
which asks whether root-to-tip distance grows with date. A tree can pass that and fail this: the
from-scratch H3 tree's clock slope was positive while its placement correlation was 0.206.
"""

from __future__ import annotations

import csv
import random
import statistics
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from af.tree.populate import PopulatedTree

COLUMNS = ("subtype", "min_correlation", "reason", "trees")

SAMPLE = 4000
"""Leaves sampled. The whole-tree calculation is O(n) per leaf for the path to the outgroup, and a
4,000-leaf sample already separates 0.206 from 0.903 by a wide margin."""

SEED = 30
"""Fixed, so the same tree always gives the same number (design rule 8). It is the seed the limits
were calibrated with; changing it would invalidate them."""

MIN_LEAVES = 200
"""Below this a correlation over the sample says little, so the check reports no verdict and says
so rather than refusing or quietly passing. Synthetic trees in tests sit here."""

DEFINITE = frozenset("ACGT")


class PlacementError(ValueError):
    """The limits file is malformed, or a tree's leaves are not placed where their sequences say."""


@dataclass(frozen=True)
class PlacementLimit:
    """One subtype's minimum placement correlation, with why it is that number."""

    subtype: str
    min_correlation: float
    reason: str
    trees: int  # how many trees the number was calibrated on; 1 means treat it as provisional

    def to_json(self) -> dict[str, Any]:
        return {
            "subtype": self.subtype,
            "min_correlation": self.min_correlation,
            "reason": self.reason,
            "calibrated_on_trees": self.trees,
        }


@dataclass(frozen=True)
class PlacementResult:
    """What the check measured. ``correlation`` is None when there were too few leaves to judge."""

    correlation: float | None
    median_ratio: float | None
    p95_ratio: float | None
    max_ratio: float | None
    leaves: int
    by_band: dict[str, float]
    not_judged: str | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "correlation": self.correlation,
            "median_ratio": self.median_ratio,
            "p95_ratio": self.p95_ratio,
            "max_ratio": self.max_ratio,
            "sampled_leaves": self.leaves,
            "ratio_by_date_band": dict(self.by_band),
            "not_judged": self.not_judged,
        }


def load_placement_limits(path: Path) -> dict[str, PlacementLimit]:
    """Read ``trees/placement.tsv``: one row per subtype, every column required."""
    path = Path(path)
    # Comment lines are dropped before the header is read, so the file can carry its preamble:
    # where the numbers came from and who decided them (design rule 11). csv.DictReader would
    # otherwise take the first comment line as the header.
    body = [line for line in path.read_text().splitlines() if not line.lstrip().startswith("#")]
    rows = list(csv.DictReader(body, delimiter="\t"))
    if not rows:
        raise PlacementError(f"{path}: no rows")
    missing = [c for c in COLUMNS if rows and c not in rows[0]]
    if missing:
        raise PlacementError(f"{path}: missing column(s) {missing}; expected {list(COLUMNS)}")
    limits: dict[str, PlacementLimit] = {}
    for row in rows:
        subtype = (row["subtype"] or "").strip()
        reason = (row["reason"] or "").strip()
        if not subtype:
            raise PlacementError(f"{path}: a row has no subtype")
        if subtype in limits:
            raise PlacementError(f"{path}: {subtype} appears twice")
        if not reason:
            raise PlacementError(f"{path}: {subtype} has no reason (design rule 11)")
        try:
            value = float(row["min_correlation"])
            trees = int(row["trees"])
        except (TypeError, ValueError) as error:
            raise PlacementError(f"{path}: {subtype}: {error}") from error
        if not -1.0 <= value <= 1.0:
            raise PlacementError(f"{path}: {subtype}: min_correlation {value} is not a correlation")
        if trees < 1:
            raise PlacementError(f"{path}: {subtype}: trees must be at least 1")
        limits[subtype] = PlacementLimit(subtype, value, reason, trees)
    return limits


def limit_for(limits: Mapping[str, PlacementLimit], subtype: str) -> PlacementLimit:
    """The subtype's limit. A subtype with no row is an error, never a default (design rule 4)."""
    try:
        return limits[subtype]
    except KeyError:
        known = ", ".join(sorted(limits)) or "none"
        raise PlacementError(f"no placement limit for {subtype!r}; the file has: {known}") from None


def measure(populated: PopulatedTree, outgroup: str) -> PlacementResult:
    """Measure placement, on CMAPLE's own lengths, so the result does not depend on the ASR."""
    tree = populated.tree
    depth: dict[int, int] = {id(tree.root): 0}
    distance: dict[int, float] = {id(tree.root): 0.0}
    parent: dict[int, Any] = {}
    for node in tree.preorder():
        if node.parent is not None:
            parent[id(node)] = node.parent
            depth[id(node)] = depth[id(node.parent)] + 1
            length = populated.ml_lengths.get(node.node_id, node.branch_length or 0.0)
            distance[id(node)] = distance[id(node.parent)] + length
    leaves = {leaf.name: leaf for leaf in tree.leaves() if leaf.name}
    target = leaves.get(outgroup)
    if target is None:
        raise PlacementError(f"outgroup {outgroup!r} is not a leaf of this tree")
    og_record = populated.leaves.get(outgroup)
    if og_record is None:
        raise PlacementError(f"outgroup {outgroup!r} has no leaf record, so no sequence")
    og_seq = og_record.nucleotides

    def patristic(leaf: Any) -> float:
        a, b = leaf, target
        while a is not b:
            if depth[id(a)] >= depth[id(b)]:
                a = parent[id(a)]
            else:
                b = parent[id(b)]
        return distance[id(leaf)] + distance[id(target)] - 2 * distance[id(a)]

    candidates = [name for name in leaves if name != outgroup and name in populated.leaves]
    chosen = random.Random(SEED).sample(candidates, min(SAMPLE, len(candidates)))
    rows: list[tuple[float, float, int | None]] = []
    for name in chosen:
        record = populated.leaves[name]
        raw = _p_distance(record.nucleotides, og_seq)
        if raw is None:
            continue
        day = record.collection_date
        rows.append((patristic(leaves[name]), raw, day.year if day else None))
    if len(rows) < MIN_LEAVES:
        return PlacementResult(
            None, None, None, None, len(rows), {},
            not_judged=f"only {len(rows)} comparable leaves; {MIN_LEAVES} needed for a verdict",
        )  # fmt: skip
    ratios = sorted(tree_d / raw for tree_d, raw, _ in rows)
    return PlacementResult(
        correlation=_correlation([t for t, _, _ in rows], [r for _, r, _ in rows]),
        median_ratio=statistics.median(ratios),
        p95_ratio=ratios[int(0.95 * len(ratios))],
        max_ratio=ratios[-1],
        leaves=len(rows),
        by_band=_by_band(rows),
    )


def check_placement(
    populated: PopulatedTree, outgroup: str, limit: PlacementLimit
) -> PlacementResult:
    """Measure, and refuse when the correlation is under the subtype's limit."""
    result = measure(populated, outgroup)
    if result.correlation is not None and result.correlation < limit.min_correlation:
        bands = ", ".join(f"{band} {ratio:.2f}" for band, ratio in sorted(result.by_band.items()))
        raise PlacementError(
            f"{limit.subtype}: leaves are not placed where their sequences say. The correlation "
            f"between path length to the outgroup and raw sequence distance to it is "
            f"{result.correlation:.3f}, under the limit {limit.min_correlation} "
            f"({limit.reason}). Median ratio {result.median_ratio:.2f}, p95 "
            f"{result.p95_ratio:.2f}, max {result.max_ratio:.2f}, over {result.leaves} leaves. "
            f"By date band: {bands}. A ratio that grows with age means the older lineages sit "
            f"inside the newer ones (notes/trees/H3-OLD-LINEAGE-PLACEMENT.md)"
        )
    return result


def _p_distance(a: str, b: str) -> float | None:
    """Difference per site over the positions where both have a definite base. None if too few."""
    if len(a) != len(b):
        return None
    same = total = 0
    for x, y in zip(a, b, strict=True):
        if x in DEFINITE and y in DEFINITE:
            total += 1
            same += x == y
    if total < 500 or total == same:
        return None
    return (total - same) / total


def _correlation(xs: list[float], ys: list[float]) -> float | None:
    try:
        return statistics.correlation(xs, ys)
    except statistics.StatisticsError:  # one of them is constant: no correlation is defined
        return None


BANDS = ((0, 2019), (2020, 2022), (2023, 2024), (2025, 9999))


def _by_band(rows: list[tuple[float, float, int | None]]) -> dict[str, float]:
    """Median ratio per date band. It is the grading by age that makes a fault legible."""
    out: dict[str, float] = {}
    for low, high in BANDS:
        got = [t / r for t, r, year in rows if year and low <= year <= high]
        if len(got) >= 20:
            name = f"{low or 'up to'}-{high if high < 9999 else 'on'}"
            out[name] = statistics.median(got)
    return out


__all__ = [
    "COLUMNS",
    "PlacementError",
    "PlacementLimit",
    "PlacementResult",
    "check_placement",
    "limit_for",
    "load_placement_limits",
    "measure",
]
