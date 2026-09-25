"""The report tree: older untitrated leaves pruned, topology and states kept. Synthetic only."""

from __future__ import annotations

import datetime

import pytest

from af.tables.model import Antigen, Table
from af.tree.populate import CladeCall, CladeResult, populate
from af.tree.report_filter import TitratedIndex, collected_before, report_tree
from tree.tree_fixtures import KEYS, built, records, states_for

CUTOFF = datetime.date(2023, 6, 1)


def table(*antigens: Antigen) -> Table:
    return Table(
        table_id="h3-hint-examplelab-20260901",
        group="h3-hint-examplelab",
        lab="EXAMPLELAB",
        subtype="A(H3N2)",
        lineage="",
        assay="HINT",
        rbc="",
        date="2026-09-01",
        date_suffix=1,
        source_key="example 1",
        antigens=list(antigens),
        sera=[],
        titres=[],
    )


def index() -> TitratedIndex:
    recs = records()
    by_name = Antigen(name=recs[KEYS["a"]].name.lower(), raw_name="x")  # matched by name
    by_epi = Antigen(
        name="A/EXAMPLEVILLE/9/2020", raw_name="x", source={"epi": recs[KEYS["c"]].epi_isl}
    )
    return TitratedIndex.from_tables(
        [table(by_name, by_epi)], name_key=str.upper, epi_fields=["epi"]
    )


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


def test_the_interval_decides_before_the_cutoff() -> None:
    recs = records()
    assert collected_before(recs[KEYS["b"]], datetime.date(2022, 1, 1)) is True
    assert collected_before(recs[KEYS["b"]], datetime.date(2021, 6, 1)) is False  # year-only 2021


def test_untitrated_old_leaves_go_and_the_rest_stay() -> None:
    populated, _ = full()
    result = report_tree(populated, CUTOFF, index())
    kept = {leaf.name for leaf in result.tree.leaves()}
    assert kept == {KEYS["o"], KEYS["a"], KEYS["c"], KEYS["d"]}
    assert result.counts["removed_before_cutoff"] == 1
    assert result.counts["kept_titrated_before_cutoff"] == 2
    assert result.counts["titrated_by_epi_isl"] == 1 and result.counts["titrated_by_name"] == 1
    assert result.titrated[KEYS["a"]] and not result.titrated[KEYS["d"]]
    assert result.titrated_by[KEYS["c"]] == ["EXAMPLELAB"] and result.titrated_by[KEYS["d"]] == []
    assert result.counts["titrated_leaves_EXAMPLELAB"] == 2
    # the input is untouched
    assert len(list(populated.tree.leaves())) == 5


def test_a_spliced_branch_carries_the_net_change_and_its_length() -> None:
    """y lost b, so a hangs from x: its branch is y's plus a's, and K2E moves onto it."""
    populated, ids = full()
    result = report_tree(populated, CUTOFF, index())
    a = next(leaf for leaf in result.tree.leaves() if leaf.name == KEYS["a"])
    x = next(child for child in result.tree.root.children if not child.is_leaf)
    assert a.parent is x
    assert a.branch_length == pytest.approx(0.05 + 0.2)
    assert result.aa_subs[a.node_id] == ["K2E"]


def test_clades_are_carried_without_rerunning_the_engine() -> None:
    populated, ids = full(assign=True)
    result = report_tree(populated, CUTOFF, index())
    x = next(child for child in result.tree.root.children if not child.is_leaf)
    assert result.clades[x.id_hex].clade == "J.2"
    assert result.clade_set_version == "v1"
    assert result.clade_parents == {"J": None, "J.2": "J"}
    assert result.counts["leaves_without_clade"] == 0


def test_the_outgroup_is_kept_even_when_old_and_untitrated() -> None:
    populated, _ = full()
    result = report_tree(populated, datetime.date(2030, 1, 1), index())
    assert KEYS["o"] in {leaf.name for leaf in result.tree.leaves()}
