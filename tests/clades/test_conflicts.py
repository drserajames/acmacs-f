"""The sibling-conflict guard: a tree that hides a sibling clade over many leaves is refused."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from af.clades.assign import conflict_review
from af.clades.conflicts import (
    ConflictError,
    ConflictLimit,
    check_conflicts,
    conflict_limit_for,
    load_conflict_limits,
)
from af.clades.from_tree import publish_from_tree
from af.clades.nomenclature import CladeSet
from af.clades.store import read_report
from af.store import Store, StoreRef
from af.tree.io import i6

from .test_from_tree import STARTED, SUBTYPE, clade_set, nomenclature_input, tree_version

HEADER = "subtype\tmax_topmost_conflict_leaves\treason\tevidence\n"
LIMIT = ConflictLimit(SUBTYPE, 100, "the limit", "five trees")


def review_with(*leaves: int, truncated: int = 0) -> dict[str, Any]:
    review = conflict_review((), ())
    review["topmost"] = [
        {
            "node": f"n{i}",
            "label": "P.1",
            "other": "P.2",
            "leaves": n,
            "label_support": 3,
            "other_support": 3,
            "gained_on_edge": True,
        }
        for i, n in enumerate(leaves)
    ]
    review["topmost_truncated"] = truncated
    return review


# ------------------------------------------------------------------ the limits file


def limits_file(tmp_path: Path, *rows: str, header: str = HEADER) -> Path:
    path = tmp_path / "conflicts.tsv"
    path.write_text(header + "".join(row + "\n" for row in rows))
    return path


def test_reads_a_limit_with_its_reason_and_evidence(tmp_path: Path) -> None:
    limits = load_conflict_limits(limits_file(tmp_path, f"{SUBTYPE}\t100\twhy\tmeasured on"))
    assert conflict_limit_for(limits, SUBTYPE) == ConflictLimit(SUBTYPE, 100, "why", "measured on")


@pytest.mark.parametrize(
    ("row", "message"),
    [
        (f"{SUBTYPE}\t100\twhy\t", "needs both a reason and its evidence"),
        (f"{SUBTYPE}\t100\t\tmeasured", "needs both a reason and its evidence"),
        (f"{SUBTYPE}\t0\twhy\tm", "is not a count >= 1"),
        (f"{SUBTYPE}\tmany\twhy\tm", "is not a count >= 1"),
        ("\t100\twhy\tm", "no subtype"),
    ],
)
def test_a_bad_row_is_refused(tmp_path: Path, row: str, message: str) -> None:
    with pytest.raises(ConflictError, match=message):
        load_conflict_limits(limits_file(tmp_path, row))


def test_duplicates_columns_absence_and_unknown_subtype_are_refused(tmp_path: Path) -> None:
    with pytest.raises(ConflictError, match="listed twice"):
        load_conflict_limits(limits_file(tmp_path, f"{SUBTYPE}\t1\ta\tb", f"{SUBTYPE}\t2\ta\tb"))
    with pytest.raises(ConflictError, match="columns must be"):
        load_conflict_limits(limits_file(tmp_path, "x\t1", header="subtype\tlimit\n"))
    with pytest.raises(ConflictError, match="not found"):
        load_conflict_limits(tmp_path / "absent.tsv")
    with pytest.raises(ConflictError, match="no sibling-conflict limit for 'B/Vic'"):
        conflict_limit_for(
            load_conflict_limits(limits_file(tmp_path, f"{SUBTYPE}\t1\ta\tb")), "B/Vic"
        )


# ------------------------------------------------------------------ the check


def test_conflicts_under_the_limit_pass_and_are_counted() -> None:
    result = check_conflicts(review_with(99, 10), LIMIT, "tree")
    assert result["over_limit"] == 0 and result["accepted"] is None


def test_a_conflict_at_the_limit_is_refused_and_named() -> None:
    with pytest.raises(ConflictError, match=r"node n0 labelled P\.1 also matches P\.2 over 3,700"):
        check_conflicts(review_with(3700, 5), LIMIT, "tree")
    with pytest.raises(ConflictError, match="1 sibling conflict"):
        check_conflicts(review_with(100), LIMIT, "tree")  # at the limit counts


def test_the_count_says_at_least_only_when_more_can_lie_beyond_the_list() -> None:
    """The review keeps the largest topmost conflicts first: a truncated list whose smallest
    entry is under the limit holds every conflict over it."""
    with pytest.raises(ConflictError, match=r"has 1 sibling conflict"):
        check_conflicts(review_with(3700, 69, truncated=30), LIMIT, "tree")
    with pytest.raises(ConflictError, match=r"has at least 2 sibling conflict"):
        check_conflicts(review_with(3700, 500, truncated=30), LIMIT, "tree")


def test_a_named_reason_publishes_over_the_limit_and_is_kept() -> None:
    result = check_conflicts(review_with(3700), LIMIT, "tree", accept_conflict="understood")
    assert (result["over_limit"], result["accepted"]) == (1, "understood")
    with pytest.raises(ConflictError, match="needs a reason"):
        check_conflicts(review_with(3700), LIMIT, "tree", accept_conflict=" ")


def test_a_tree_without_a_review_cannot_be_checked() -> None:
    """An absent review is not "checked, nothing found"."""
    with pytest.raises(ConflictError, match="no sibling-conflict review"):
        check_conflicts(None, LIMIT, "tree")


# ------------------------------------------------------------------ through publishing


def publish(
    tmp_path: Path, store: Store, tree: StoreRef, clades: CladeSet, **kwargs: Any
) -> StoreRef:
    return publish_from_tree(
        store,
        tree,
        SUBTYPE,
        clades,
        nomenclature=[nomenclature_input(tmp_path)],
        started=STARTED,
        **kwargs,
    )


def test_a_tree_hiding_a_large_sibling_conflict_is_not_published(tmp_path: Path) -> None:
    store = Store.create(tmp_path / "store")
    clades = clade_set(tmp_path)
    tree = tree_version(store, clades.version, review=review_with(3700))
    with pytest.raises(ConflictError, match="covering at least 100 leaves"):
        publish(tmp_path, store, tree, clades, conflicts=LIMIT)
    ref = publish(
        tmp_path, store, tree, clades, conflicts=LIMIT, accept_conflict="known and understood"
    )
    report = read_report(store, ref)["sibling_conflicts"]
    assert report["topmost"][0]["leaves"] == 3700
    assert report["check"]["accepted"] == "known and understood"
    provenance = json.loads((store.resolve(ref) / "PROVENANCE.json").read_text())
    assert provenance["parameters"]["sibling_conflicts"] == {
        "max_topmost_conflict_leaves": 100,
        "accepted": "known and understood",
    }


def test_a_tree_publish_needs_a_conflict_limit(tmp_path: Path) -> None:
    from af.clades.store import CladeStoreError

    store = Store.create(tmp_path / "store")
    clades = clade_set(tmp_path)
    tree = tree_version(store, clades.version)
    with pytest.raises(CladeStoreError, match="needs a sibling-conflict limit"):
        publish(tmp_path, store, tree, clades)


def test_a_tree_without_a_clade_set_carries_no_review(tmp_path: Path) -> None:
    """No clade set: no key at all, so it cannot be mistaken for a review that found nothing."""
    store = Store.create(tmp_path / "store")
    tree = tree_version(store, None)
    assert "clade_review" not in i6.read_metadata(store.resolve(tree))["counts"]
