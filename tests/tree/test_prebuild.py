"""Dropping long-branch sequences before the tree is built. Synthetic data only."""

from __future__ import annotations

from pathlib import Path

import pytest

from af.tree.io import fasta, newick
from af.tree.prebuild import (
    LONG_BRANCH,
    LONG_BRANCH_RULE,
    ExclusionRule,
    PrebuildError,
    apply_plan,
    filter_alignment,
    long_branch_leaves,
    names_dropped,
    plan,
)

# b sits on a branch 20x longer than anyone else's; outgroup is long too, deliberately.
TREE = "((a:0.001,b:0.05):0.002,(c:0.002,d:0.001):0.001,outgroup:0.4);"


def tree_of(text: str = TREE):
    tree = newick.loads(text)
    tree.assign_ids()
    return tree


def alignment(tmp_path: Path, names: tuple[str, ...] = ("a", "b", "c", "d", "outgroup")) -> Path:
    path = tmp_path / "input.fasta"
    fasta.write_alignment(path, {name: "ACGTACGT" for name in names})
    return path


def test_only_leaves_over_the_threshold_are_found() -> None:
    assert sorted(long_branch_leaves(tree_of(), 0.01)) == ["b", "outgroup"]
    assert long_branch_leaves(tree_of(), 0.5) == {}


def test_the_outgroup_is_exempt_and_the_exemption_is_reported() -> None:
    """Without it the rebuild has nothing to root on (CUT-NODE.md §3d)."""
    result = plan(tree_of(), keep=["outgroup"])
    assert result.keys() == {"b"}
    assert result.counts["exempt_but_over_threshold"] == ["outgroup"]


def test_the_plan_records_the_threshold_and_why() -> None:
    result = plan(tree_of(), keep=["outgroup"])
    assert result.counts["threshold"] == 0.01
    assert "topology and speed" in str(result.counts["why"])
    assert result.counts["measured_on"] == "previous tree"
    assert result.counts["dropped"] == 1


def test_a_rule_this_module_does_not_implement_is_refused() -> None:
    bogus = ExclusionRule(reason="vibes", threshold=1.0, why="no")
    with pytest.raises(PrebuildError, match="no pre-build rule named"):
        plan(tree_of(), bogus)


def test_the_filtered_alignment_loses_exactly_the_dropped_sequences(tmp_path: Path) -> None:
    source = alignment(tmp_path)
    target = tmp_path / "filtered.fasta"
    counts = filter_alignment(source, target, ["b"])
    assert sorted(fasta.read_alignment(target)) == ["a", "c", "d", "outgroup"]
    assert counts == {"sequences_in": 5, "dropped": 1, "sequences_out": 4}
    # the export itself is untouched
    assert sorted(fasta.read_alignment(source)) == ["a", "b", "c", "d", "outgroup"]


def test_dropping_a_key_the_alignment_does_not_have_is_an_error(tmp_path: Path) -> None:
    """It means the plan came from a different dataset; building the full tree would hide that."""
    with pytest.raises(PrebuildError, match="measured on a different dataset"):
        filter_alignment(alignment(tmp_path), tmp_path / "out.fasta", ["b", "nobody"])


def test_dropping_almost_everything_is_refused(tmp_path: Path) -> None:
    source = alignment(tmp_path, ("a", "b", "c"))
    with pytest.raises(PrebuildError, match="would leave 1 sequences"):
        filter_alignment(source, tmp_path / "out.fasta", ["b", "c"])


def test_apply_plan_merges_both_sets_of_counts(tmp_path: Path) -> None:
    counts = apply_plan(
        alignment(tmp_path), tmp_path / "filtered.fasta", plan(tree_of(), keep=["outgroup"])
    )
    assert counts["dropped_long_branch"] == 1
    assert counts["sequences_out"] == 4
    assert counts["rule"] == LONG_BRANCH


def test_dropped_sequences_are_reportable_by_strain_name() -> None:
    result = plan(tree_of(), keep=["outgroup"])
    assert names_dropped(result, [("b", "A/EXAMPLETOWN/2/2024")]) == ["A/EXAMPLETOWN/2/2024 (0.05)"]


def test_the_default_rule_carries_its_evidence() -> None:
    """A threshold outlives whoever chose it, so it travels with its reason."""
    assert LONG_BRANCH_RULE.threshold == 0.01
    assert "9 of 9 hand-hidden" in LONG_BRANCH_RULE.why


def test_a_threshold_that_cannot_fire_says_so() -> None:
    """On the mutations scale ae's 0.01 exceeds the longest branch possible: inert, not clean."""
    result = plan(tree_of("((a:0.001,b:0.002):0.001,c:0.0015,outgroup:0.003);"), keep=["outgroup"])
    assert result.drop == {}
    assert "cannot fire" in str(result.counts["inert"])
    assert result.counts["longest_terminal_branch"] == 0.003


def test_a_threshold_that_can_fire_is_not_called_inert() -> None:
    assert "inert" not in plan(tree_of(), keep=["outgroup"]).counts


def test_records_name_each_dropped_sequence_for_the_comparison() -> None:
    """WS11 must be able to tell "dropped by the filter" from "lost", so identity travels."""
    result = plan(tree_of(), keep=["outgroup"])
    result.drop.clear()
    result.lengths.clear()
    result.drop["EPI_ISL_9001|EPI9001"] = LONG_BRANCH
    result.lengths["EPI_ISL_9001|EPI9001"] = 0.05
    (row,) = result.records({"EPI_ISL_9001|EPI9001": "A/EXAMPLETOWN/2/2024"})
    assert row["epi_isl"] == "EPI_ISL_9001"
    assert row["accession"] == "EPI9001"
    assert row["name"] == "A/EXAMPLETOWN/2/2024"
    assert row["reason"] == LONG_BRANCH
    assert row["branch_length"] == 0.05
    assert row["threshold"] == 0.01


def test_records_work_without_a_name_lookup() -> None:
    result = plan(tree_of(), keep=["outgroup"])
    assert all(row["name"] is None for row in result.records())
    assert [row["leaf_id"] for row in result.records()] == ["b"]
