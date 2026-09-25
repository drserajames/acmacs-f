"""Building a tree: CMAPLE, then root, ladderize and collapse — with every step counted.

    result = build(alignment, outgroup="...", out_dir=..., settings=CmapleSettings())
    result.tree          the finished tree, ids assigned
    result.counts        what each step did, for the provenance

The order matches today's pipeline (`ae/proj/weekly-tree`: make-cmaple, rotate.R, collapse) so the
output is comparable with the delivered trees, but each step reports what it did instead of
succeeding silently. Measurement behind the collapse default is in notes/trees/COMPARISON.md §3:
`di2multi(tol=5e-5)` never once fired on a non-zero branch in any delivered tree, so af collapses
exactly-zero branches by default and keeps the tolerance in config, where an inert setting is
visible rather than implied.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from af.run import Runner
from af.tree.build.cmaple import (
    CmapleSettings,
    build_job,
    prune_starting_tree,
    run_cmaple,
    version,
)
from af.tree.model import Tree


@dataclass
class BuildResult:
    """The tree, and what each step did to it."""

    tree: Tree
    counts: dict[str, int | float | str] = field(default_factory=dict)


def finish_tree(
    tree: Tree,
    outgroup: str,
    collapse_tolerance: float = 0.0,
    smallest_first: bool = True,
) -> BuildResult:
    """Root on the outgroup, collapse short branches, ladderize, and assign ids.

    Rooting comes first because collapsing changes which branches exist, and ladderizing last
    because it is presentation. Ids are assigned at the end and are unaffected by the ladderizing
    (that is the point of them), so the same tree drawn differently keeps its labels.
    """
    counts: dict[str, int | float | str] = {}
    leaves_before, internal_before = tree.count()
    counts["leaves"] = leaves_before
    counts["internal_nodes_before"] = internal_before

    counts["unary_nodes_removed"] = tree.remove_unary()
    tree.reroot_on_outgroup(outgroup)
    counts["outgroup"] = outgroup
    counts["branches_collapsed"] = tree.collapse_short_branches(collapse_tolerance)
    counts["collapse_tolerance"] = collapse_tolerance
    tree.ladderize(smallest_first=smallest_first)
    tree.assign_ids()

    leaves_after, internal_after = tree.count()
    counts["internal_nodes_after"] = internal_after
    if leaves_after != leaves_before:
        raise AssertionError(
            f"finishing the tree changed the leaf count ({leaves_before} -> {leaves_after}); "
            "rooting, collapsing and ladderizing must never lose a leaf"
        )
    return BuildResult(tree=tree, counts=counts)


def build(
    alignment: Path,
    outgroup: str,
    out_dir: Path,
    settings: CmapleSettings | None = None,
    starting_tree: Path | None = None,
    collapse_tolerance: float = 0.0,
    runner: Runner | None = None,
) -> BuildResult:
    """CMAPLE, then finish. The whole build step."""
    settings = settings or CmapleSettings()
    tree = run_cmaple(alignment, out_dir, settings, starting_tree, runner)
    result = finish_tree(tree, outgroup, collapse_tolerance)
    result.counts["builder"] = version(settings.executable)
    result.counts["search"] = settings.search
    result.counts["seed"] = settings.seed
    result.counts["from_scratch"] = "no" if starting_tree else "yes"
    return result


__all__ = [
    "BuildResult",
    "CmapleSettings",
    "build",
    "build_job",
    "finish_tree",
    "prune_starting_tree",
    "run_cmaple",
    "version",
]
