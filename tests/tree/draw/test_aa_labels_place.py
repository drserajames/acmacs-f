"""aa-transition label filters and label placement."""

from dataclasses import replace

import pytest

np = pytest.importorskip(
    "numpy", reason="numpy not installed: af.tree.draw needs it (pyproject, WS1)"
)

from af.tree.draw.aa_labels import (  # noqa: E402
    LabelHide,
    LabelRuleError,
    hide_labels,
    leaf_consensus,
    select_labels,
)
from af.tree.draw.defaults import load_defaults  # noqa: E402
from af.tree.draw.layout import compute_layout  # noqa: E402
from af.tree.draw.place import (  # noqa: E402
    Grid,
    Search,
    place_labels,
    placement_metrics,
    segment_hits_box,
)

from .synthetic import (  # noqa: E402
    BASE,
    build,
    clade_block,
    inner,
    leaf,
    mutate,
    standard_tree,
)

D = load_defaults()
LOW = replace(D.labels, target_labels=0, min_share=0.05, max_share=0.99)


def labels_of(tree, **kw):
    lay = compute_layout(tree)
    labels, counts = select_labels(tree, lay, (), p=replace(LOW, **kw))
    return {" ".join(lab.subs) for lab in labels}, counts


def test_stem_changes_are_labelled_and_keyed_by_leaf_ids():
    t = standard_tree()
    lay = compute_layout(t)
    labels, _ = select_labels(t, lay, (), LOW)
    by_text = {" ".join(lab.subs): lab for lab in labels}
    assert set(by_text) == {"K2N", "L7F", "A6S"}
    assert by_text["L7F"].first == "EPI_ISL_0000010" and by_text["L7F"].rows == 10


def test_small_subtrees_are_not_labelled():
    texts, counts = labels_of(standard_tree(), min_share=0.3)
    assert texts == {"K2N", "A6S"} and counts["dropped: small subtree"] == 1


def test_change_on_nearly_every_leaf_is_not_labelled():
    seq = mutate(BASE, "T3S")
    t = build(inner(inner(*clade_block(0, 99, "X", aa=seq), subs=("T3S",)), leaf(99, "X")))
    texts, counts = labels_of(t, max_share=0.95)
    assert "T3S" not in texts and counts["dropped: on nearly every drawn leaf"] == 1


def reversion_tree(back_rows: int, total: int = 40):
    fwd = mutate(BASE, "S8N")
    back = [leaf(100 + k, "X", aa=BASE) for k in range(back_rows)]
    kept = [leaf(200 + k, "X", aa=fwd) for k in range(total - back_rows)]
    return build(
        inner(
            inner(inner(*back, subs=("N8S",)), inner(*kept), subs=("S8N",)),
            *clade_block(300, 20, "X"),
        )
    )


def test_partial_reversion_is_hidden_and_forward_change_kept():
    texts, counts = labels_of(reversion_tree(back_rows=25))
    # 25 of 40 rows reverted: the forward change is judged on the 15 that kept it
    assert texts == {"S8N"} and counts["dropped subs: partial reversion"] == 1


def test_backbone_reversion_is_shown():
    texts, _ = labels_of(reversion_tree(back_rows=38))
    assert texts == {"S8N", "N8S"}


def test_leaf_consensus_ignores_x_and_gaps():
    assert leaf_consensus(["AXC", "A-C", "AGC"], 2) == ("G", 1.0)
    assert leaf_consensus([None, "A"], 3) == ("", 0.0)


def test_segment_box_intersection():
    box = (10.0, 10.0, 20.0, 20.0)
    assert segment_hits_box((0, 15), (30, 15), box)
    assert not segment_hits_box((0, 0), (5, 30), box)


def test_placement_avoids_ink_and_other_labels_and_is_deterministic():
    grid = Grid(200, 200)
    for y in range(40, 160, 4):  # a block of "tree" to the right of x = 120
        grid.add_hline(120, 190, y)
    items = [(k, f"A{k}B", (120.0, 60.0 + 3 * k)) for k in range(8)]  # crowded anchors
    width = lambda s: 5.0 * len(s)  # noqa: E731
    a = place_labels(items, grid, width, 7.0, search=Search(left=2))
    b = place_labels(items, grid, width, 7.0, search=Search(left=2))
    assert [p.box for p in a] == [p.box for p in b]
    m = placement_metrics(a, grid)
    assert m["box_overlaps"] == 0 and m["labels_on_tree_ink"] == 0 and m["leader_crossings"] == 0


def test_target_label_count_sets_the_size_threshold():
    t = standard_tree()
    lay = compute_layout(t)
    # stems: K2N 20 rows, A6S 20 rows, L7F 10 rows
    two, counts = select_labels(t, lay, (), replace(D.labels, target_labels=2, min_rows_floor=5))
    assert {" ".join(lab.subs) for lab in two} == {"K2N", "A6S"}
    assert counts["size threshold (rows)"] == 20
    three, _ = select_labels(t, lay, (), replace(D.labels, target_labels=3, min_rows_floor=5))
    assert len(three) == 3
    floor, _ = select_labels(t, lay, (), replace(D.labels, target_labels=3, min_rows_floor=15))
    assert len(floor) == 2  # the floor wins over the target


def test_a_position_minimum_lets_a_smaller_subtree_through():
    # stems: K2N 20 rows, A6S 20, L7F 10 (of 50); a 15-row minimum keeps only the first two
    texts, _ = labels_of(standard_tree(), min_share=0.3)
    assert texts == {"K2N", "A6S"}
    texts, counts = labels_of(standard_tree(), min_share=0.3, min_rows_per_pos={"7": 5})
    assert texts == {"K2N", "A6S", "L7F"}
    assert counts["kept by a position's own minimum"] == 1
    texts, _ = labels_of(standard_tree(), min_share=0.3, min_rows_per_pos={"7": 0.1})  # 5 rows
    assert "L7F" in texts


@pytest.mark.parametrize(
    "kw",
    [
        {"min_rows_per_pos": {"7": 5}, "target_labels": 3},  # a share rule only, as in AD
        {"min_rows_per_pos": {"x7": 5}},
        {"min_rows_per_pos": {"7": 0}},
    ],
)
def test_a_position_minimum_that_cannot_apply_is_an_error(kw):
    with pytest.raises(LabelRuleError):
        labels_of(standard_tree(), **kw)


def flip_tree(levels: int):
    """S8N on a 40-row node; 10 of its rows revert (N8S) ``levels`` inode levels below it, and
    5 others change on (N8T)."""
    fwd, on = mutate(BASE, "S8N"), mutate(BASE, "S8T")
    back = inner(*[leaf(100 + k, "X", aa=BASE) for k in range(10)], subs=("N8S",))
    for _ in range(levels):
        back = inner(back)
    kept = inner(*[leaf(200 + k, "X", aa=fwd) for k in range(25)])
    onward = inner(*[leaf(300 + k, "X", aa=on) for k in range(5)], subs=("N8T",))
    return build(inner(inner(back, kept, onward, subs=("S8N",)), *clade_block(400, 20, "X")))


def test_a_close_revert_removes_the_change_and_the_revert():
    kw = {"leaf_filters": False, "flip_removal": True, "min_share": 0.05}
    texts, counts = labels_of(flip_tree(levels=0), **kw)
    # the onward change is re-anchored to the residue above the removed one
    assert texts == {"S8T"} and counts["removed subs: back-and-forth (flip)"] == 2
    texts, _ = labels_of(flip_tree(levels=0), **{**kw, "flip_removal": False})
    assert texts == {"S8N", "N8S", "N8T"}


def test_a_revert_far_below_or_on_few_rows_is_not_a_flip():
    kw = {"leaf_filters": False, "flip_removal": True, "min_share": 0.05}
    texts, _ = labels_of(flip_tree(levels=3), **kw)  # 3 levels below: AD's limit is < 3
    assert texts == {"S8N", "N8S", "N8T"}
    texts, _ = labels_of(flip_tree(levels=0), **{**kw, "flip_min_share": 0.5})  # 10 of 40 rows
    assert texts == {"S8N", "N8S", "N8T"}


def test_leaf_filters_off_keeps_a_partial_reversion():
    texts, _ = labels_of(reversion_tree(back_rows=25), leaf_filters=False)
    assert texts == {"S8N", "N8S"}


def test_named_hide_drops_its_label_and_must_match_one():
    t = standard_tree()
    lay = compute_layout(t)
    labels, _ = select_labels(t, lay, (), LOW)
    l7f = next(lab for lab in labels if lab.subs == ["L7F"])
    hide = LabelHide(l7f.first, l7f.last, "L7F", "invented")
    kept, n = hide_labels(t, lay, labels, [hide])
    assert n == 1 and {" ".join(lab.subs) for lab in kept} == {"K2N", "A6S"}
    # any two leaves under the node name it: the common ancestor is the labelled node
    inside = LabelHide("EPI_ISL_0000012", "EPI_ISL_0000017", "L7F", "invented")
    assert hide_labels(t, lay, labels, [inside])[1] == 1
    for bad in (
        LabelHide(l7f.first, l7f.last, "L7A", "text differs"),
        LabelHide("EPI_ISL_0000000", "EPI_ISL_0000045", "L7F", "node has no label"),
        LabelHide("EPI_ISL_9999999", l7f.last, "L7F", "leaf not on the tree"),
    ):
        with pytest.raises(LabelRuleError):
            hide_labels(t, lay, labels, [bad])


def _crowded(seed: int):
    """Many labels anchored close together beside random ink: where single moves stall."""
    rng = np.random.default_rng(seed)
    grid = Grid(400, 400)
    for _ in range(120):
        x, y = rng.uniform(120, 380), rng.uniform(0, 390)
        grid.add_box((x, y, x + rng.uniform(5, 60), y + 0.5))
    items = [
        (k, f"A{k}B", (float(rng.uniform(150, 220)), float(rng.uniform(60, 340))))
        for k in range(40)
    ]
    return grid, items


@pytest.mark.parametrize("seed", [3, 4, 7])
def test_pair_repair_untangles_what_single_moves_leave(seed):
    def width(s: str) -> float:
        return 6.0 * len(s)

    def clashes(placed) -> int:
        m = placement_metrics(placed, grid)
        return m["leader_crossings"] + m["leaders_through_labels"] + m["box_overlaps"]

    grid, items = _crowded(seed)
    single = place_labels(items, grid, width, 10.0, search=Search(pair_candidates=0))
    paired = place_labels(items, grid, width, 10.0, search=Search(pair_candidates=40))
    assert clashes(paired) < clashes(single)  # when written: 7 -> 2, 6 -> 0, 13 -> 4
