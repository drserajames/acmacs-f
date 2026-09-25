"""The clock and long-branch outlier rules (task 5.9). Synthetic data only.

A clean molecular clock is built by hand — leaf i is collected in year 2000+i and sits exactly
i * rate from the root — and single leaves are then broken in known ways.
"""

from __future__ import annotations

import datetime

import pytest

from af.tree.clock import (
    CLOCK_OUTLIER,
    IMPOSSIBLE_DATE,
    LONG_BRANCH,
    ClockSettings,
    apply_flags,
    excluded,
    find_outliers,
    root_to_tip,
    theil_sen,
)
from af.tree.model import Node, Tree
from af.tree.populate import LeafRecord, PopulatedTree, leaf_key

RATE = 0.005


def clock_tree(count: int = 40) -> PopulatedTree:
    """A caterpillar: leaf i hangs off the spine at depth i, collected in year 2000 + i."""
    root = Node()
    spine = root
    leaves: dict[str, LeafRecord] = {}

    # Deterministic scatter. A *perfect* clock has a zero residual scale, so a single stray leaf
    # would score as infinitely outlying; real trees always have some, and so must the fixture.
    spread = (-0.5, -0.3, -0.1, 0.1, 0.3, 0.5)
    jitter = [0.1 * RATE * (1 + spread[i % len(spread)]) for i in range(count + 1)]

    def add_leaf(index: int, parent: Node, depth: int) -> None:
        """Dated by its DEPTH, not its index: two leaves on one spine node are contemporaries."""
        key = leaf_key(f"EPI_ISL_{80000 + index}", f"EPI{80000 + index}")
        parent.children.append(Node(name=key, branch_length=jitter[index], parent=parent))
        leaves[key] = LeafRecord(
            f"EPI_ISL_{80000 + index}",
            f"EPI{80000 + index}",
            f"A/EXAMPLETOWN/{index}/{2000 + depth}",
            "",
            collection_date=datetime.date(2000 + depth, 7, 1),
            date_precision="day",
        )

    # Each spine node carries one leaf; the last carries two, so no node is left unary.
    for index in range(count - 2):
        add_leaf(index, spine, index)
        nxt = Node(branch_length=RATE, parent=spine)
        spine.children.append(nxt)
        spine = nxt
    add_leaf(count - 2, spine, count - 2)
    add_leaf(count - 1, spine, count - 2)
    tree = Tree(root)
    tree.assign_ids()
    return PopulatedTree(
        tree=tree,
        subtype="h3",
        alignment_length=1650,
        branch_scale="mutations",
        leaves=leaves,
        states=None,
        ml_lengths={},
        nuc_changes={},
        aa={},
        aa_subs={},
        nuc_subs={},
    )


def leaf_named(populated: PopulatedTree, key: str) -> Node:
    return next(leaf for leaf in populated.tree.leaves() if leaf.name == key)


def keys(populated: PopulatedTree) -> list[str]:
    return list(populated.leaves)


def test_theil_sen_ignores_a_wild_minority() -> None:
    xs = [float(i) for i in range(20)]
    ys = [2.0 * x for x in xs]
    ys[0] = 500.0  # one catastrophic point
    assert theil_sen(xs, ys) == pytest.approx(2.0)


def test_theil_sen_is_deterministic_on_a_large_input() -> None:
    xs = [float(i % 97) for i in range(1200)]
    ys = [3.0 * x for x in xs]
    assert theil_sen(xs, ys) == theil_sen(xs, ys)


def test_a_clean_clock_flags_nothing() -> None:
    populated = clock_tree()
    result = find_outliers(populated)
    assert result.counts["clock_outliers"] == 0
    assert result.slope == pytest.approx(RATE, rel=0.05)


def test_a_leaf_far_off_the_clock_is_flagged() -> None:
    populated = clock_tree()
    key = keys(populated)[10]
    leaf_named(populated, key).branch_length = 0.08  # ~16 years of divergence in one branch
    # branch_threshold raised so only the clock rule can speak; 0.08 would also be a long branch.
    result = find_outliers(populated, ClockSettings(branch_threshold=1.0))
    assert result.flags[key] == [CLOCK_OUTLIER]
    assert result.counts["clock_outliers"] == 1
    assert abs(result.z[key]) > 4


def test_an_impossible_date_is_its_own_category_not_a_clock_outlier() -> None:
    """The clock has nothing to say about a date that cannot be true (COMPARISON.md §3c)."""
    populated = clock_tree()
    key = keys(populated)[5]
    populated.leaves[key] = LeafRecord(
        populated.leaves[key].epi_isl,
        populated.leaves[key].accession,
        populated.leaves[key].name,
        "",
        collection_date=datetime.date(1481, 8, 4),
    )
    result = find_outliers(populated)
    assert result.flags[key] == [IMPOSSIBLE_DATE]
    assert key not in result.z


def test_one_impossible_date_does_not_move_the_fit() -> None:
    """The point of the robust fit: a leaf five centuries away must not rescale everyone else."""
    clean = find_outliers(clock_tree())
    populated = clock_tree()
    key = keys(populated)[5]
    populated.leaves[key] = LeafRecord(
        populated.leaves[key].epi_isl,
        populated.leaves[key].accession,
        populated.leaves[key].name,
        "",
        collection_date=datetime.date(1481, 8, 4),
    )
    spoiled = find_outliers(populated)
    assert spoiled.slope == pytest.approx(clean.slope, rel=0.01)
    assert spoiled.counts["clock_outliers"] == 0


def test_a_long_terminal_branch_is_flagged_even_when_the_date_fits() -> None:
    """ae's .tal edge rule: it caught 17 H1 leaves the clock did not."""
    populated = clock_tree()
    key = keys(populated)[3]
    leaf_named(populated, key).branch_length = 0.02
    result = find_outliers(populated, ClockSettings(z_threshold=100.0, branch_threshold=0.01))
    assert result.flags[key] == [LONG_BRANCH]
    assert result.counts["clock_outliers"] == 0


def test_a_short_tree_says_the_clock_was_skipped_rather_than_scoring_it() -> None:
    result = find_outliers(clock_tree(5))
    assert "only 5 dated leaves" in str(result.counts["clock_skipped"])
    assert result.counts["clock_outliers"] == 0


def test_undated_leaves_are_counted_not_flagged() -> None:
    populated = clock_tree()
    key = keys(populated)[7]
    populated.leaves[key] = LeafRecord(
        populated.leaves[key].epi_isl, populated.leaves[key].accession, "A/EXAMPLETOWN/7/2007", ""
    )
    result = find_outliers(populated)
    assert result.counts["undated"] == 1
    assert key not in result.flags


def test_flags_reach_the_tree_for_i6_and_exclusion_is_opt_in() -> None:
    populated = clock_tree()
    key = keys(populated)[10]
    leaf_named(populated, key).branch_length = 0.08  # ~16 years of divergence in one branch
    result = find_outliers(populated, ClockSettings(branch_threshold=1.0))
    apply_flags(populated, result)
    assert populated.flags[leaf_named(populated, key).node_id] == [CLOCK_OUTLIER]
    assert populated.counts["clock_clock_outliers"] == 1
    # Flagging is reporting; removal is a separate, named decision (DECISIONS 25 Sep).
    assert excluded(result) == [key]
    assert excluded(result, {LONG_BRANCH: True}) == []


def test_root_to_tip_is_the_cumulative_length() -> None:
    populated = clock_tree(4)
    distances = root_to_tip(populated.tree)
    # Leaf i hangs off spine depth i, except the last two, which share the final spine node.
    depths = sorted(round(value, 6) for value in distances.values())
    assert depths[0] == pytest.approx(0.0, abs=RATE / 2)
    assert depths[-1] == pytest.approx(2 * RATE, abs=RATE / 2)
    assert len(depths) == 4
