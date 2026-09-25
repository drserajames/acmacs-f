"""The tree stage's settings, per subtype, from TOML (design rules 4 and 10).

Everything the tree stage does was previously a dataclass default in code: the CMAPLE flags, the
branch scale, the ASR backend, the outlier thresholds and the outgroups. Defaults in code are not
configuration — a run cannot be reproduced from a file, and a site-specific choice (which outgroup,
which scale) ends up compiled in. This module makes them a config file, and keeps the *measured*
defaults so a minimal file still does the right thing.

Per subtype, because almost every one of these genuinely differs between subtypes:

* **branch scale** — Sarah, 24 Sep: both, chosen per subtype. The delivered h3 and B/Vic trees carry
  mutation counts / L; h1 carries CMAPLE's ML lengths.
* **outgroup** — a different strain for each, and the build cannot root without it.
* **ASR backend** — B/Vic's clades are defined by deletions, so it needs a gap-capable backend;
  af refuses a gap-blind one there (``af.tree.populate``).
* **long-branch threshold** — in the units of the branch scale, so it is set **per scale**, in
  config, with its reason in the same row (Sarah, 25 Sep, Q9): ``[long_branch.<scale>]``. ae's 0.01
  was set against ML lengths; on the mutations scale the longest branch possible is about
  16/1650 = 0.0097, so carried over it could never fire. A scale with no row drops nothing, and
  says so; :func:`af.tree.prebuild.plan` reports a threshold that cannot fire as inert.

Nothing here reads the environment, and an unknown key in the file is an error, not a typo that
silently leaves a default in place.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import cast

from af.run import Resources
from af.tree.asr import DEFAULT_BACKEND, available_backends
from af.tree.build.cmaple import SEARCH_TYPES, CmapleSettings
from af.tree.clock import ClockSettings
from af.tree.populate import BRANCH_SCALES, BranchScale
from af.tree.prebuild import DEFAULT_LONG_BRANCH, LONG_BRANCH, LONG_BRANCH_RULE, ExclusionRule
from af.util.config import load_config


class TreeConfigError(ValueError):
    """The tree config is not usable as written."""


JOB_STAGES = ("build", "asr", "populate")
"""The stages that run as a job each (under SLURM, one allocation per stage per subtype).
Publish and clades are short store writes and run in the driver."""


@dataclass(frozen=True)
class LongBranchSetting:
    """One ``[long_branch.<scale>]`` row: the threshold, and why it is that number.

    The reason is required (design rule 11): a reader of the tree must be able to tell why a
    sequence is missing from it. What is known, for whoever writes the rows:

    ``ml``: ae's 0.01 is calibrated — on the 2026-0921 H1 tree it drops 9 leaves and all 9 were
    hand-hidden (notes/trees/CLOCK.md).

    ``mutations``: there is no ground truth; the only hand-curated long-branch set is H1's, which is
    on the ML scale. It is a different quantity, not a small adjustment of ae's number. Leaves over
    the threshold on the 2026-0921 trees, by changes on the terminal branch (x / alignment length):

        changes:   4     5     6     8    10
        H3:     1324   447   168    35    11
        B/Vic:   549   196    82    16     1
    """

    threshold: float
    reason: str

    def __post_init__(self) -> None:
        if self.threshold <= 0:
            raise TreeConfigError("long_branch threshold must be positive")
        if not self.reason.strip():
            raise TreeConfigError("a long_branch row needs its reason (design rule 11)")


@dataclass(frozen=True)
class SubtypeSettings:
    """One subtype's tree settings. Only ``outgroup`` has no sensible default."""

    outgroup: str
    branch_scale: str = "ml"  # a BranchScale; str because the config loader has no Literal support
    asr_backend: str = DEFAULT_BACKEND
    collapse_tolerance: float = 0.0
    long_branch_threshold: float | None = None
    """In the units of ``branch_scale``. Unset: the ``[long_branch.<scale>]`` row, if any.
    Set here only for a subtype that genuinely differs from its scale's row."""
    long_branch_reason: str | None = None
    """Required with ``long_branch_threshold``. Filled from the scale's row otherwise."""
    clock_z_threshold: float = 4.0
    drop_long_branches: bool = True
    """Sarah, 25 Sep: long branches go before the tree is built. Off only for a deliberate
    comparison with a tree that kept them."""
    report_cutoff: str | None = None
    """ISO date. Leaves collected wholly before it are kept only if titrated (task 5.5)."""
    resources: dict[str, Resources] = field(default_factory=dict)
    """Per stage (``build``, ``asr``, ``populate``): what its job asks the scheduler for. The three
    subtypes differ several-fold in size, so wall-times and memory are set per subtype. A stage
    with no entry gets ``TreeSettings.threads`` and no memory or time limit."""
    exclude_reasons: list[str] = field(default_factory=list)
    """Further exclusions this round opts into, by reason (Sarah: "further exclusions in the
    round"). Empty means report only, which is the default everywhere."""

    def __post_init__(self) -> None:
        problems = []
        if self.branch_scale not in BRANCH_SCALES:
            problems.append(f"branch_scale {self.branch_scale!r} is not one of {BRANCH_SCALES}")
        if self.asr_backend not in available_backends():
            problems.append(
                f"asr_backend {self.asr_backend!r} is not one of {available_backends()}"
            )
        if not self.outgroup:
            problems.append("outgroup is empty; the build cannot root the tree without one")
        if self.long_branch_threshold is not None and self.long_branch_threshold <= 0:
            problems.append("long_branch_threshold must be positive")
        unknown = sorted(set(self.resources) - set(JOB_STAGES))
        if unknown:
            problems.append(f"resources for unknown stage(s) {unknown}; stages: {JOB_STAGES}")
        if any(r.threads < 1 for r in self.resources.values()):
            problems.append("resources: threads must be at least 1")
        if self.long_branch_threshold is not None and not (self.long_branch_reason or "").strip():
            problems.append("long_branch_threshold needs a long_branch_reason (design rule 11)")
        if problems:
            raise TreeConfigError("; ".join(problems))

    @property
    def scale(self) -> BranchScale:
        """``branch_scale`` as the typed value populate() takes. Validated in __post_init__."""
        return cast("BranchScale", self.branch_scale)

    @property
    def long_branch(self) -> float | None:
        """The threshold to use, or None when config sets none for this scale or subtype."""
        return self.long_branch_threshold

    def long_branch_rule(self) -> ExclusionRule | None:
        """The pre-build rule, or None when config sets no threshold — never an invented number."""
        threshold = self.long_branch
        if threshold is None:
            return None
        why = f"{LONG_BRANCH_RULE.why} {self.branch_scale!r} scale: {self.long_branch_reason}"
        return ExclusionRule(reason=LONG_BRANCH, threshold=threshold, why=why)

    def no_long_branch_reason(self) -> str | None:
        """Why no pre-build drop will happen, for the counts. None when one will."""
        if not self.drop_long_branches:
            return "drop_long_branches is off in config"
        if self.long_branch is None:
            return (
                f"no [long_branch.{self.branch_scale}] row in config, so nothing is dropped; af "
                "will not invent a threshold (Sarah, 25 Sep: set per scale, in config)"
            )
        return None

    def clock_settings(self) -> ClockSettings:
        """The long-branch check reports at the threshold the build drops at, or at ae's ML number
        when the scale has none, so a scale without a drop threshold is still reported on."""
        return ClockSettings(
            z_threshold=self.clock_z_threshold,
            branch_threshold=self.long_branch or DEFAULT_LONG_BRANCH,
        )


@dataclass(frozen=True)
class TreeSettings:
    """The TOML schema: ``[cmaple]``, ``[defaults]`` and one ``[subtypes.<name>]`` per subtype."""

    subtypes: dict[str, SubtypeSettings]
    cmaple: CmapleSettings = field(default_factory=CmapleSettings)
    long_branch: dict[str, LongBranchSetting] = field(default_factory=dict)
    """Keyed by branch scale. A scale with no row drops no long branches, and says so."""
    defaults: dict[str, str] = field(default_factory=dict)
    """Applied to every subtype that does not set the key itself. Strings, converted per field."""

    threads: int = 4
    store: Path | None = None
    """Tree-store root. Absent when a run writes plain directories instead of store versions."""

    def __post_init__(self) -> None:
        if not self.subtypes:
            raise TreeConfigError("no [subtypes.<name>] section: there is nothing to build")
        if self.cmaple.search not in SEARCH_TYPES:  # CmapleSettings checks this too; be explicit
            raise TreeConfigError(f"cmaple.search must be one of {SEARCH_TYPES}")
        unknown = sorted(set(self.long_branch) - set(BRANCH_SCALES))
        if unknown:
            raise TreeConfigError(
                f"[long_branch.X] for unknown scale(s) {unknown}: {BRANCH_SCALES}"
            )

    def for_subtype(self, subtype: str) -> SubtypeSettings:
        """One subtype's settings, with its scale's long-branch row applied unless it sets its own.

        An unknown subtype is an error, never a default (rule 4).
        """
        try:
            settings = self.subtypes[subtype]
        except KeyError:
            known = ", ".join(sorted(self.subtypes))
            raise TreeConfigError(
                f"no settings for subtype {subtype!r}; the config names: {known}"
            ) from None
        row = self.long_branch.get(settings.branch_scale)
        if settings.long_branch_threshold is not None or row is None:
            return settings
        return replace(
            settings,
            long_branch_threshold=row.threshold,
            long_branch_reason=f"[long_branch.{settings.branch_scale}] {row.reason}",
        )

    def resources_for(self, subtype: str, stage: str) -> Resources:
        """What ``stage``'s job for ``subtype`` asks for. An unknown stage is an error."""
        if stage not in JOB_STAGES:
            raise TreeConfigError(f"{stage!r} does not run as a job; job stages: {JOB_STAGES}")
        configured = self.for_subtype(subtype).resources.get(stage)
        return configured if configured is not None else Resources(threads=self.threads)

    def cmaple_for(self, subtype: str) -> CmapleSettings:
        """CMAPLE settings with the build job's thread count, so it uses what it was given."""
        return replace(self.cmaple, threads=self.resources_for(subtype, "build").threads)


def load_tree_config(path: Path) -> TreeSettings:
    """Read a tree config. A missing file, an unknown key or a bad value is fatal."""
    return load_config(Path(path), TreeSettings)


__all__ = [
    "JOB_STAGES",
    "LongBranchSetting",
    "SubtypeSettings",
    "TreeConfigError",
    "TreeSettings",
    "load_tree_config",
]
