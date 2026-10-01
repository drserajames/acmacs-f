"""The placement guard (af.tree.placement). Synthetic sequences and trees only.

A leaf's path to the outgroup should grow with how different its sequence is from the outgroup's.
The fixtures build a tree where that holds and one where it is inverted, which is the fault the
guard exists to catch (notes/trees/H3-OLD-LINEAGE-PLACEMENT.md: correlation 0.206).
"""

from __future__ import annotations

import datetime
from pathlib import Path

import pytest

from af.tree.model import Node, Tree
from af.tree.placement import (
    PlacementError,
    PlacementLimit,
    check_placement,
    limit_for,
    load_placement_limits,
    measure,
)
from af.tree.populate import LeafRecord, PopulatedTree, leaf_key

LENGTH = 1200
OUTGROUP = leaf_key("EPI_ISL_70000", "EPI70000")


def sequence(mutations: int) -> str:
    """The outgroup's sequence with `mutations` sites changed, so p-distance grows with it."""
    base = list("ACGT" * (LENGTH // 4))
    for site in range(mutations):
        base[site * 3] = "T" if base[site * 3] != "T" else "G"
    return "".join(base)


def ladder(count: int = 300, *, inverted: bool = False) -> PopulatedTree:
    """A ladder: the leaf at depth i carries i mutations, so distance tracks divergence.

    ``inverted`` hangs the same leaves in the opposite order, which is what a tree looks like when
    the builder has put the divergent sequences nearest the root.
    """
    root = Node()
    outgroup = Node(name=OUTGROUP, branch_length=0.0, parent=root)
    root.children.append(outgroup)
    leaves = {
        OUTGROUP: LeafRecord(
            "EPI_ISL_70000",
            "EPI70000",
            "A/EXAMPLETOWN/0/2009",
            sequence(0),
            collection_date=datetime.date(2009, 1, 1),
            date_precision="day",
        )  # fmt: skip
    }
    spine = root
    order = range(count - 1, 0, -1) if inverted else range(1, count)
    for index in order:
        step = Node(branch_length=1.0 / LENGTH, parent=spine)
        spine.children.append(step)
        spine = step
        key = leaf_key(f"EPI_ISL_7{index:04d}", f"EPI7{index:04d}")
        spine.children.append(Node(name=key, branch_length=0.0, parent=spine))
        year = 2009 + min(index // 20, 16)
        leaves[key] = LeafRecord(
            f"EPI_ISL_7{index:04d}", f"EPI7{index:04d}", f"A/EXAMPLETOWN/{index}/{year}",
            sequence(index), collection_date=datetime.date(year, 6, 1), date_precision="day",
        )  # fmt: skip
    # The last spine node needs a second leaf: a node with one child is unary and refused.
    extra = leaf_key("EPI_ISL_79999", "EPI79999")
    last = count - 1 if not inverted else 1
    spine.children.append(Node(name=extra, branch_length=0.0, parent=spine))
    leaves[extra] = LeafRecord(
        "EPI_ISL_79999", "EPI79999", "A/EXAMPLETOWN/9999/2025", sequence(last),
        collection_date=datetime.date(2025, 6, 1), date_precision="day",
    )  # fmt: skip
    tree = Tree(root)
    tree.assign_ids()
    return PopulatedTree(
        tree=tree, subtype="h3", alignment_length=LENGTH, branch_scale="ml", leaves=leaves,
        states=None, ml_lengths={}, nuc_changes={}, aa={}, aa_subs={}, nuc_subs={},
    )  # fmt: skip


LIMIT = PlacementLimit("h3", 0.85, "calibrated on two trees", 2)


def test_a_tree_whose_leaves_sit_where_their_sequences_say_passes() -> None:
    result = check_placement(ladder(), OUTGROUP, LIMIT)
    assert result.correlation is not None and result.correlation > 0.95
    assert result.leaves == 300
    assert result.median_ratio is not None and 0.5 < result.median_ratio < 2.0


def test_an_inverted_tree_is_refused() -> None:
    with pytest.raises(PlacementError, match="not placed where their sequences say") as refused:
        check_placement(ladder(inverted=True), OUTGROUP, LIMIT)
    assert "is -1.000" in str(refused.value)  # the ladder hung in reverse: perfectly inverted
    assert "By date band" in str(refused.value)


def test_the_same_tree_always_gives_the_same_number() -> None:
    assert measure(ladder(), OUTGROUP).correlation == measure(ladder(), OUTGROUP).correlation


def test_a_small_tree_is_not_judged() -> None:
    result = check_placement(ladder(count=40), OUTGROUP, LIMIT)
    assert result.correlation is None
    assert result.not_judged is not None and "verdict" in result.not_judged


def test_an_outgroup_that_is_not_a_leaf_is_an_error() -> None:
    with pytest.raises(PlacementError, match="not a leaf"):
        measure(ladder(), "EPI_ISL_9|EPI9")


def write(path: Path, body: str) -> Path:
    path.write_text("subtype\tmin_correlation\treason\ttrees\n" + body)
    return path


def test_limits_load_with_their_reasons(tmp_path: Path) -> None:
    path = write(
        tmp_path / "p.tsv",
        "h3\t0.85\ttwo trees: ae round 0.966, af seeded 0.982\t2\n"
        "B/Vic\t0.75\tone tree only: ae round 0.903, the noisiest\t1\n",
    )
    limits = load_placement_limits(path)
    assert sorted(limits) == ["B/Vic", "h3"]
    assert limits["h3"].min_correlation == 0.85
    assert limits["B/Vic"].trees == 1
    assert "noisiest" in limits["B/Vic"].reason


def test_a_row_without_a_reason_is_refused(tmp_path: Path) -> None:
    with pytest.raises(PlacementError, match="no reason"):
        load_placement_limits(write(tmp_path / "p.tsv", "h3\t0.85\t\t2\n"))


def test_a_repeated_subtype_is_refused(tmp_path: Path) -> None:
    body = "h3\t0.85\tfirst\t2\nh3\t0.70\tsecond\t1\n"
    with pytest.raises(PlacementError, match="appears twice"):
        load_placement_limits(write(tmp_path / "p.tsv", body))


def test_a_value_that_is_not_a_correlation_is_refused(tmp_path: Path) -> None:
    with pytest.raises(PlacementError, match="not a correlation"):
        load_placement_limits(write(tmp_path / "p.tsv", "h3\t1.5\twhy\t2\n"))


def test_a_subtype_with_no_row_is_an_error(tmp_path: Path) -> None:
    limits = load_placement_limits(write(tmp_path / "p.tsv", "h3\t0.85\twhy\t2\n"))
    with pytest.raises(PlacementError, match="no placement limit for 'h1'"):
        limit_for(limits, "h1")


def test_a_preamble_before_the_header_is_skipped(tmp_path: Path) -> None:
    """The real file explains where its numbers came from, above the header (design rule 11)."""
    path = tmp_path / "p.tsv"
    path.write_text(
        "# Minimum placement correlation per subtype.\n"
        "# Sarah, 30 September 2026.\n"
        "subtype\tmin_correlation\treason\ttrees\n"
        "h3\t0.85\ttwo trees\t2\n"
        "# bvic is still being calibrated\n"
    )
    limits = load_placement_limits(path)
    assert sorted(limits) == ["h3"]
    assert limits["h3"].min_correlation == 0.85
