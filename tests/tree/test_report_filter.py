"""The report tree: every leaf of its source, no date rule and no tables. Synthetic only.

Sarah, 25 Nov 2026: a report tree must not drop sequences that were not titrated, and "the tables
shoud not affect the trees". A tree is made smaller only by a node cut (tests/tree/test_trim.py).
"""

from __future__ import annotations

import inspect

from af.tree import report_filter
from af.tree.populate import CladeCall, CladeResult, populate
from af.tree.report_filter import report_tree

from .tree_fixtures import KEYS, built, records, states_for


def full(assign: bool = False):
    tree, ids = built()

    def engine(nodes):
        calls = {n.node_id: CladeCall("J" if n.is_leaf else "J.2") for n in nodes}
        return CladeResult("v1", calls, {"J": None, "J.2": "J"})

    return populate(
        tree,
        "h3",
        records(),
        states_for(ids),
        branch_scale="ml",
        assign_clades=engine if assign else None,
        outgroup=KEYS["o"],
    ), ids


def test_every_leaf_is_kept_whatever_its_date() -> None:
    """o (2020), a (2021), b (year-only 2021) are as old as the fixture gets; none is dropped."""
    populated, _ = full()
    result = report_tree(populated)
    assert {leaf.name for leaf in result.tree.leaves()} == set(KEYS.values())
    for gone in ("removed_before_cutoff", "kept_titrated_before_cutoff", "report_cutoff"):
        assert gone not in result.counts
    # the input is untouched
    assert len(list(populated.tree.leaves())) == 5


def test_no_leaf_is_marked_titrated_so_i6_writes_null_as_on_a_weekly_tree() -> None:
    populated, _ = full()
    result = report_tree(populated)
    assert result.titrated == {} and result.titrated_by == {}
    assert not any(key.startswith("titrated") or key.startswith("titre_") for key in result.counts)


def test_nothing_here_takes_a_titre_table() -> None:
    """The tables cannot reach the tree through this function: it has no parameter for them."""
    assert set(inspect.signature(report_tree).parameters) == {
        "populated",
        "backend",
        "assign_clades",
        "continent_of",
        "trim",
    }
    assert "af.tables" not in inspect.getsource(report_filter)


def test_without_a_trim_the_tree_is_its_source_leaf_for_leaf() -> None:
    populated, _ = full()
    result = report_tree(populated)
    by_name = {leaf.name: leaf for leaf in populated.tree.leaves()}
    for leaf in result.tree.leaves():
        source = by_name[leaf.name]
        assert leaf.branch_length == source.branch_length
        assert result.aa_subs[leaf.node_id] == populated.aa_subs[source.node_id]
    assert result.tree.count() == populated.tree.count()


def test_clades_are_carried_without_rerunning_the_engine() -> None:
    populated, ids = full(assign=True)
    result = report_tree(populated)
    x = next(child for child in result.tree.root.children if not child.is_leaf)
    assert result.clades[x.id_hex].clade == "J.2"
    assert result.clade_set_version == "v1"
    assert result.clade_parents == {"J": None, "J.2": "J"}
    assert result.counts["leaves_without_clade"] == 0


def test_the_outgroup_is_kept_without_a_trim() -> None:
    populated, _ = full()
    result = report_tree(populated)
    assert KEYS["o"] in {leaf.name for leaf in result.tree.leaves()}
    assert result.counts["outgroup"] == KEYS["o"]
