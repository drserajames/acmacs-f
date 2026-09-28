"""Clock and long-branch outliers: the tree rule that replaces part of the hand hides (task 5.9).

Ownership (DECISIONS, 25 Sep 2026). The report trees' hand hides split in two: 97-100% are
year-precision dates, which WS9 handles in the figure, and **the remainder are clock or
long-branch outliers, which are WS5's**. Measured by eu-56 on the September round: H3 22, H1 4 plus
17 caught by the `.tal` edge>=0.01 rule, B/Vic 0.

Two independent checks, because they catch different things:

**Root-to-tip clock.** A virus whose genetic distance from the root does not fit its collection date
is either misdated or contaminated. Distance is regressed on date and the residual is scored. The
fit is **robust** (Theil-Sen slope, median intercept, MAD-scaled residuals) and not least squares:
the round's own trees carry collection dates in 1480-1481 (an ae bug, `COMPARISON.md` §3c), and one
leaf five centuries from the data is enough to drag a least-squares line and inflate its residual
scale, so every other leaf is then scored against the wrong ruler.

**Long terminal branch.** A leaf on a very long branch of its own is a sequence unlike anything else
in the tree, whatever its date says. This is ae's `.tal` edge rule, which caught 17 H1 leaves the
clock check alone did not.

What the rules do *not* do is decide silently. A leaf with an impossible date is its own reported
category, never a clock outlier: the clock has nothing to say about a date that cannot be true.
Every leaf carries its reason and its score, and `flags` travels into I6 for the figure to count.
"""

from __future__ import annotations

import datetime
import random
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from af.tree.model import Tree
from af.tree.populate import PopulatedTree

CLOCK_OUTLIER = "clock_outlier"
LONG_BRANCH = "long_branch"
IMPOSSIBLE_DATE = "impossible_date"

#: Influenza A/H1N1pdm09 emerged in 2009; sequencing of seasonal influenza predates that by
#: decades, so anything before this is a data error rather than an old virus. Config, not code.
DEFAULT_EARLIEST = datetime.date(1977, 1, 1)


@dataclass(frozen=True)
class ClockSettings:
    """Thresholds, each with the reason it is set where it is (design rule: state the reason)."""

    z_threshold: float = 4.0
    """Robust z above which a leaf is a clock outlier. eu-56 measured the H3 hand-hidden batch at
    +4.8 to +5.4, and ordinary leaves well inside +-4, so 4.0 separates them with margin."""

    branch_threshold: float | None = None
    """Terminal branch length counted as long. None = use ae's .tal rule of 0.01 substitutions per
    site, which caught 17 H1 leaves the clock missed."""

    earliest: datetime.date = DEFAULT_EARLIEST
    """Collection dates before this are impossible, reported as such, and left out of the fit."""

    min_leaves: int = 20
    """Below this many dated leaves a robust fit is not meaningful; the clock check is skipped and
    says so, rather than scoring a handful of points against each other."""


@dataclass
class ClockResult:
    """Scores and reasons per leaf, plus the fit, for the provenance and the review page."""

    slope: float | None
    intercept: float | None
    residual_scale: float | None
    z: dict[str, float] = field(default_factory=dict)
    flags: dict[str, list[str]] = field(default_factory=dict)
    counts: dict[str, object] = field(default_factory=dict)

    @property
    def flagged(self) -> list[str]:
        return sorted(self.flags)


def _decimal_year(day: datetime.date) -> float:
    start = datetime.date(day.year, 1, 1)
    length = (datetime.date(day.year + 1, 1, 1) - start).days
    return day.year + (day - start).days / length


def theil_sen(xs: Sequence[float], ys: Sequence[float], pairs: int = 200_000) -> float:
    """Median of pairwise slopes: unaffected by a minority of wild points, unlike least squares.

    All pairs is O(n^2), which on 90,000 leaves is 4e9 of them and takes minutes, so at most
    ``pairs`` are sampled. The sample is drawn from a fixed seed, never from system randomness, so
    the same tree always gives the same slope (design rule 8).
    """
    n = len(xs)
    if n < 2:
        raise ValueError("a slope needs at least two points")
    slopes: list[float] = []
    if n * (n - 1) // 2 <= pairs:
        candidates = ((i, j) for i in range(n - 1) for j in range(i + 1, n))
    else:
        rng = random.Random(20260925)
        candidates = ((rng.randrange(n), rng.randrange(n)) for _ in range(pairs))
    for i, j in candidates:
        run = xs[j] - xs[i]
        if run:
            slopes.append((ys[j] - ys[i]) / run)
    if not slopes:
        raise ValueError("every pair of points shares an x value; no slope is defined")
    return statistics.median(slopes)


def root_to_tip(tree: Tree) -> dict[str, float]:
    """Cumulative branch length from the root to each leaf, by leaf name."""
    lengths = tree.cumulative_lengths()
    return {leaf.name or "": lengths[id(leaf)] for leaf in tree.leaves()}


def find_outliers(
    populated: PopulatedTree,
    settings: ClockSettings | None = None,
) -> ClockResult:
    """Score every leaf and flag the clock and long-branch outliers.

    Branch lengths must be on a per-site scale for the thresholds to mean what they say; with the
    ``mutations`` scale they are (changes / alignment length), which is what ae's rule assumed.
    """
    settings = settings or ClockSettings()
    tree = populated.tree
    distances = root_to_tip(tree)
    result = ClockResult(slope=None, intercept=None, residual_scale=None)

    dated: list[tuple[str, float, float]] = []
    impossible: list[str] = []
    undated = 0
    for key, record in populated.leaves.items():
        day = record.collection_date
        if day is None:
            undated += 1
            continue
        if day < settings.earliest:
            impossible.append(key)
            result.flags.setdefault(key, []).append(IMPOSSIBLE_DATE)
            continue
        dated.append((key, _decimal_year(day), distances[key]))

    result.counts["leaves"] = len(populated.leaves)
    result.counts["undated"] = undated
    result.counts["impossible_date"] = len(impossible)

    if len(dated) >= settings.min_leaves:
        xs = [x for _, x, _ in dated]
        ys = [y for _, _, y in dated]
        slope = theil_sen(xs, ys)
        intercept = statistics.median(y - slope * x for x, y in zip(xs, ys, strict=True))
        residuals = [y - (intercept + slope * x) for x, y in zip(xs, ys, strict=True)]
        # MAD, scaled to be comparable with a standard deviation on normal data.
        scale = statistics.median(abs(r) for r in residuals) * 1.4826
        result.slope, result.intercept, result.residual_scale = slope, intercept, scale
        if scale > 0:
            for (key, _, _), residual in zip(dated, residuals, strict=True):
                z = residual / scale
                result.z[key] = z
                if abs(z) >= settings.z_threshold:
                    result.flags.setdefault(key, []).append(CLOCK_OUTLIER)
        else:
            result.counts["clock_skipped"] = "residual scale is zero"
    else:
        result.counts["clock_skipped"] = f"only {len(dated)} dated leaves"

    threshold = settings.branch_threshold
    if threshold is None:
        threshold = 0.01
    for leaf in tree.leaves():
        if leaf.branch_length >= threshold:
            result.flags.setdefault(leaf.name or "", []).append(LONG_BRANCH)

    result.counts["clock_outliers"] = sum(CLOCK_OUTLIER in v for v in result.flags.values())
    result.counts["long_branches"] = sum(LONG_BRANCH in v for v in result.flags.values())
    result.counts["flagged"] = len(result.flags)
    result.counts["branch_threshold"] = threshold
    result.counts["z_threshold"] = settings.z_threshold
    if result.slope is not None:
        result.counts["substitutions_per_site_per_year"] = result.slope
    return result


class InvertedClockError(ValueError):
    """Root-to-tip distance falls with collection date: the tree is rooted in the wrong place."""


@dataclass(frozen=True)
class ClockDirection:
    """Whether distance from the root grows with date, measured on CMAPLE's own lengths."""

    slope: float | None  # per year, Theil-Sen; None when too few dated leaves to say
    correlation: float | None  # Pearson, reported alongside so a reader can judge the slope
    leaves: int  # dated leaves the fit used
    nearest_root: list[tuple[str, float, str]]  # (leaf key, ML distance, ISO date), closest first

    def to_json(self) -> dict[str, object]:
        return {
            "slope_ml_per_year": self.slope,
            "correlation": self.correlation,
            "dated_leaves": self.leaves,
            "nearest_root": [list(item) for item in self.nearest_root],
        }


def check_clock_direction(
    populated: PopulatedTree, *, min_leaves: int = 20, nearest: int = 10
) -> ClockDirection:
    """Refuse a tree whose whole clock is inverted; this is an artefact check, not curation.

    Individual clock outliers are reported and kept (Sarah, 25 Sep). A tree whose root-to-tip
    slope is not positive is different: it is rooted among recent viruses, so every distance from
    the root means the opposite of what it should. The first full H3 build (28 Sep 2026) was, with
    slope -0.0051/yr: a cluster of recent sequences almost identical to the 2009 outgroup joined the
    tree at its recent end. CMAPLE's lengths are used, not the published scale, so the check does
    not depend on ancestral reconstruction.
    """
    tree = populated.tree
    distance: dict[int, float] = {id(tree.root): 0.0}
    for node in tree.preorder():
        if node.parent is not None:
            length = populated.ml_lengths.get(node.node_id, node.branch_length or 0.0)
            distance[id(node)] = distance[id(node.parent)] + length
    dated: list[tuple[str, float, float, datetime.date]] = []
    for leaf in tree.leaves():
        record = populated.leaves.get(leaf.name or "")
        if record is not None and record.collection_date is not None:
            day = record.collection_date
            dated.append((leaf.name or "", _decimal_year(day), distance[id(leaf)], day))
    closest = sorted(dated, key=lambda item: (item[2], item[0]))[:nearest]
    nearest_root = [(key, round(d, 6), day.isoformat()) for key, _, d, day in closest]
    xs = [x for _, x, _, _ in dated]
    if len(dated) < min_leaves or len(set(xs)) < 2:
        return ClockDirection(None, None, len(dated), nearest_root)
    ys = [y for _, _, y, _ in dated]
    slope = theil_sen(xs, ys)
    correlation = statistics.correlation(xs, ys) if len(set(ys)) > 1 else None
    direction = ClockDirection(slope, correlation, len(dated), nearest_root)
    if slope <= 0:
        shown = ", ".join(f"{key} {day} ({d:g})" for key, d, day in nearest_root[:5])
        corr = "n/a" if correlation is None else f"{correlation:.3f}"
        raise InvertedClockError(
            f"root-to-tip distance does not grow with date on CMAPLE's lengths: slope "
            f"{slope:.5g}/yr, correlation {corr}, over {len(dated)} dated leaves. The tree is "
            f"rooted among recent sequences. Nearest the root: {shown}"
        )
    return direction


def apply_flags(populated: PopulatedTree, result: ClockResult) -> dict[str, object]:
    """Write the flags onto the tree's leaves, for I6's ``flags`` column. Returns the counts.

    Flagging is reporting, not exclusion (DECISIONS, 25 Sep). Removing these leaves is a separate,
    explicit step, so a reader of the figure can always be shown what was dropped and why.
    """
    by_name = {leaf.name or "": leaf for leaf in populated.tree.leaves()}
    for key, reasons in result.flags.items():
        node = by_name.get(key)
        if node is not None:
            populated.flags.setdefault(node.node_id, []).extend(reasons)
    populated.counts.update({f"clock_{k}": v for k, v in result.counts.items()})
    return dict(result.counts)


def excluded(result: ClockResult, reasons: Mapping[str, bool] | None = None) -> list[str]:
    """The leaves an explicit exclusion step would remove, by reason (default: clock outliers).

    DECISIONS, 25 Sep: "flagged means reported, not excluded ... exclusion stays with WS5's
    clock-outlier check". So exclusion is possible here, but it is opt-in and named per reason.
    """
    wanted = dict(reasons or {CLOCK_OUTLIER: True})
    return sorted(
        key for key, flags in result.flags.items() if any(wanted.get(flag) for flag in flags)
    )


__all__ = [
    "CLOCK_OUTLIER",
    "IMPOSSIBLE_DATE",
    "LONG_BRANCH",
    "ClockResult",
    "ClockSettings",
    "apply_flags",
    "excluded",
    "find_outliers",
    "root_to_tip",
    "theil_sen",
    "ClockDirection",
    "InvertedClockError",
    "check_clock_direction",
]
