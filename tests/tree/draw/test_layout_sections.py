"""Layout, hiding rules, clade membership, bands, selection and lettered bands."""

import pytest

np = pytest.importorskip(
    "numpy", reason="numpy not installed: af.tree.draw needs it (pyproject, WS1)"
)

from tree.draw.synthetic import PARENTS, build, inner, leaf, standard_tree  # noqa: E402

from af.tree.draw.layout import HideRuleError, HideRules, compute_layout  # noqa: E402
from af.tree.draw.model import TreeModelError  # noqa: E402
from af.tree.draw.sections import (  # noqa: E402
    SectionOverrideError,
    SelectParams,
    _letter,
    clade_bands,
    clade_membership,
    hz_partition,
    select_clades,
)
from af.tree.draw.timeseries import compute, months, parse_month  # noqa: E402


def test_model_rejects_duplicate_leaf_ids():
    with pytest.raises(TreeModelError, match="duplicate"):
        build(inner(leaf(1, "X"), leaf(1, "X")))


def test_layout_rows_follow_preorder_and_inodes_sit_midway():
    t = standard_tree()
    lay = compute_layout(t)
    assert lay.n_rows == 50
    assert [t.name[i] for i in lay.leaf_nodes[:3]] == [
        "leaf-000",
        "leaf-001",
        "leaf-002",
    ]
    assert (lay.first_row[0], lay.last_row[0]) == (0, 49) and lay.rows_of(0) == 50
    # an inode sits midway between its first and last drawn child, as in the past reports
    first_child, last_child = t.children[0][0], t.children[0][-1]
    assert lay.node_y[0] == pytest.approx((lay.node_y[first_child] + lay.node_y[last_child]) / 2)


def test_hide_rules_are_counted_and_year_precision_is_not_the_day():
    t = build(
        inner(
            leaf(1, "X", date="2025-01-01", precision="year"),
            leaf(2, "X", date="2025-01-01", precision="day"),  # a real 1 January
            leaf(3, "X"),
            leaf(4, "X"),
        )
    )
    lay = compute_layout(t, HideRules(names=frozenset({"leaf-004"})))
    assert lay.hidden == {"year-precision date": 1, "long branch": 0, "named override": 1}
    assert [t.name[i] for i in lay.leaf_nodes] == ["leaf-002", "leaf-003"]


def test_named_hide_that_matches_nothing_is_an_error():
    with pytest.raises(HideRuleError):
        compute_layout(standard_tree(), HideRules(names=frozenset({"no-such-leaf"})))


def test_long_branch_rule_hides_the_subtree():
    t = build(inner(inner(leaf(1, "X"), leaf(2, "X"), edge=0.5), leaf(3, "X")))
    lay = compute_layout(t, HideRules(min_edge=0.1))
    assert lay.hidden["long branch"] == 2 and lay.n_rows == 1


def test_membership_resolves_ancestry_from_single_labels():
    m = clade_membership(["X.1.1", "X.2", None], PARENTS)
    assert m["X"].tolist() == [True, True, False]
    assert m["ROOTCL"].tolist() == [True, True, False]
    assert m["X.1"].tolist() == [True, False, False]


def test_membership_unknown_label_is_an_error():
    with pytest.raises(SectionOverrideError):
        clade_membership(["Y.9"], PARENTS)


def test_bands_merge_small_gaps_but_not_through_a_sibling():
    m = np.zeros(200, bool)
    m[0:50] = m[52:100] = True  # 2-row gap: merged
    m[150:200] = True  # 50-row gap of another clade: 50 > 5% of 148, stays separate
    cs = clade_bands(m, "C")
    assert [(b.first, b.last) for b in cs.bands] == [(0, 99), (150, 199)]


def test_bands_drop_small_strays_and_count_them():
    m = np.zeros(300, bool)
    m[0:200] = True
    m[290] = True
    cs = clade_bands(m, "C")
    assert [(b.first, b.last) for b in cs.bands] == [(0, 199)]
    assert len(cs.dropped) == 1 and cs.dropped[0].members == 1


def test_selection_rules_and_slots():
    t = standard_tree()
    lay = compute_layout(t)
    member = clade_membership([t.clade[i] for i in lay.leaf_nodes], PARENTS)
    sel = select_clades(member, PARENTS, lay.n_rows, SelectParams(min_share=0.1))
    shown = {cs.clade for cs in sel.shown}
    assert shown == {"X", "X.1", "X.1.1", "X.2"}
    assert "ROOTCL" in sel.rejected  # on every row
    assert sel.slots == {"X": 2, "X.1": 1, "X.2": 1, "X.1.1": 0}


def test_clade_override_must_name_a_clade():
    t = standard_tree()
    lay = compute_layout(t)
    member = clade_membership([t.clade[i] for i in lay.leaf_nodes], PARENTS)
    with pytest.raises(SectionOverrideError):
        select_clades(member, PARENTS, lay.n_rows, force_hide=frozenset({"Q.7"}))


def test_hz_letters_old_nested_band_stays_inside_parent():
    t = standard_tree()
    lay = compute_layout(t)
    member = clade_membership([t.clade[i] for i in lay.leaf_nodes], PARENTS)
    sel = select_clades(member, PARENTS, lay.n_rows, SelectParams(min_share=0.1))
    ts = compute([t.date[i] for i in lay.leaf_nodes], "2024-10", "2026-10")
    hz = hz_partition(sel, PARENTS, ts.in_window)
    # X.1.1 (in window) cuts X.1; X.2 (collected 2023) stays inside X's letter
    assert [(h.letter, h.clade, h.first, h.last) for h in hz] == [
        ("A", "X.1", 0, 9),
        ("B", "X.1.1", 10, 19),
        ("C", "X", 20, 44),
    ]


def test_hz_override_by_row_must_match_a_band_start():
    t = standard_tree()
    lay = compute_layout(t)
    member = clade_membership([t.clade[i] for i in lay.leaf_nodes], PARENTS)
    sel = select_clades(member, PARENTS, lay.n_rows, SelectParams(min_share=0.1))
    inw = np.ones(lay.n_rows, bool)
    with pytest.raises(SectionOverrideError):
        hz_partition(sel, PARENTS, inw, hide_first_rows=frozenset({3}))


def test_letters_continue_past_z():
    assert [_letter(k) for k in (0, 25, 26, 27)] == ["A", "Z", "AA", "AB"]


def test_months_window_end_is_exclusive_and_parsed():
    ms = months("2024-10", "2026-10")
    assert len(ms) == 24 and ms[0] == (2024, 10) and ms[-1] == (2026, 9)
    assert parse_month("2025") is None and parse_month("2025-13-01") is None
    with pytest.raises(ValueError):
        months("2026-10", "2024-10")
