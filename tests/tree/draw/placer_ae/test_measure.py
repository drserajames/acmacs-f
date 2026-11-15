"""ae's reporting on hand-made layouts: the drawn leader (oracles/small-cases "draw"), residual
conflicts and the metrics line counted with the search's own predicates, the Phase-A score of
a layout, and af's counters for an ae layout. Invented labels and coordinates only."""

import pytest

from af.tree.draw.placer_ae import predicates as pr
from af.tree.draw.placer_ae.measure import (
    af_counters,
    drawn_leader,
    glyph_gap,
    metrics,
    residuals,
    score,
    warning,
)
from af.tree.draw.placer_ae.predicates import consts
from af.tree.draw.placer_ae.scene import Label, Options, Scene

FS = 9.5
ADV = 0.6 * FS  # the oracles' monospace advance per character


def _scene(strokes=(), obstacles=()):
    return Scene((), tuple(strokes), tuple(obstacles), (0.0, 0.0, 1000.0, 1000.0), 1000.0, FS)


K = consts(_scene(), Options())


def _label(text, end, pinned=None, hand=False):
    lines = tuple(text.split("\n"))
    lw = tuple(ADV * len(t) for t in lines)
    return Label(0, text, lines, lw, FS, end, end, pinned, hand)


def _at(text, box, end, pinned=False, hand=False):
    """A placed label; its attach point is ae's for the box (the text's end, mid-height)."""
    lab = _label(text, end, box if pinned else None, hand)
    return (lab, box, (box[2], (box[1] + box[3]) / 2))


def _box(x0, y0, lines=1, chars=3):
    return (x0, y0, x0 + ADV * chars, y0 + lines * 1.18 * FS)


# Three pairs, far apart, each with one conflict of its own kind.
TEXT_A = _at("K2N", _box(100.0, 100.0), (150.0, 115.0))
TEXT_B = _at("A6S", _box(90.0, 106.0), (140.0, 125.0))  # glyphs on A's
OVER_C = _at("L7F", _box(100.0, 200.0), (200.0, 205.605))  # flat leader through D's glyphs
OVER_D = _at("K2N", _box(150.0, 198.0), (200.0, 180.0))
NODE_E = _at("A6S", _box(250.0, 380.0), (300.0, 400.0))  # E and F share a node
NODE_F = _at("L7F", _box(250.0, 410.0), (300.0, 400.0))
LAYOUT = [TEXT_A, TEXT_B, OVER_C, OVER_D, NODE_E, NODE_F]
CLEAN = [TEXT_A, OVER_C, NODE_E]


# ---- the drawn leader ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "attach, end",
    [
        ((145.0, 500.0), (146.14, 500.0)),  # L30: drawn, gap min(1.14, 12)
        ((172.0, 503.0), None),  # dx3_dy3: L 4.24, but both axes within 3.8
        ((171.0, 500.0), (172.14, 500.0)),  # dx4_dy0: dx 4 > 3.8
        ((172.1, 500.0), None),  # L2_9
    ],
)
def test_drawn_leader_matches_the_oracle(attach, end):
    seg = drawn_leader(_label("K2N", (175.0, 500.0)), attach)
    if end is None:
        assert seg is None
        return
    assert seg is not None
    assert seg[0] == (175.0, 500.0)
    assert seg[1] == pytest.approx(end)


def test_the_gap_scales_with_the_label_size():
    # A drawn leader is longer than 0.4 fs, so 0.4 L > 0.16 fs and the gap is always 0.12 fs:
    # ae's 0.4 L cap never binds. At size 2 a 4 pt leader is drawn, 0.24 short; at size 20 the
    # same leader is within 0.4 sizes and is not drawn.
    small = Label(0, "K2N", ("K2N",), (3.6,), 2.0, (175.0, 500.0), (175.0, 500.0))
    seg = drawn_leader(small, (171.0, 500.0))
    assert seg is not None and seg[1][0] - 171.0 == pytest.approx(0.24)
    big = Label(0, "K2N", ("K2N",), (36.0,), 20.0, (175.0, 500.0), (175.0, 500.0))
    assert drawn_leader(big, (171.0, 500.0)) is None


# ---- residual conflicts -------------------------------------------------------------------------


def test_each_conflict_is_counted_once_with_its_kind():
    found = residuals(LAYOUT, K)
    assert [(c.i, c.j, c.kinds) for c in found] == [
        (0, 1, ("text/text",)),
        (2, 3, ("i-leader/j-text",)),
        (4, 5, ("leader/leader",)),  # one node: ae's leaders always cross
    ]
    assert sum(c.count for c in found) == sum(
        pr.pconf(p, q, K) for i, p in enumerate(LAYOUT) for q in LAYOUT[i + 1 :]
    )


def test_a_clean_layout_has_no_residuals_and_no_warning():
    assert residuals(CLEAN, K) == []
    assert warning([]) is None


def test_main_mode_names_the_box_rule():
    k = consts(_scene(), Options(label_clash="main"))
    # main tests whole boxes: B's leader starts inside A's box grown by mt, ink's glyph line not
    assert residuals([TEXT_A, TEXT_B], k)[0].kinds == ("box/box", "j-leader/i-text")


def test_the_line_names_the_glyph_gap_pins_and_tiers():
    c = residuals([TEXT_A, TEXT_B], K, tiers=[1, 3])[0]
    line = str(c)
    assert line.startswith("residual conflict [text/text (glyph gap 0.00pt)]: 'K2N' end(150.0")
    assert "tier 1 vs 'A6S'" in line and line.endswith("tier 3")
    assert "pinned" not in line


def test_two_hand_pinned_labels_say_which_tool_moves_them():
    a = _at("K2N", TEXT_A[1], TEXT_A[0].end, pinned=True, hand=True)
    b = _at("A6S", TEXT_B[1], TEXT_B[0].end, pinned=True, hand=True)
    c = residuals([a, b], K)[0]
    assert c.both_pinned and c.pinned == (True, True) and c.hand == (True, True)
    assert str(c).endswith("; both placed by hand: move one in the label editor")
    assert "'K2N' pinned by hand" in str(c)
    assert warning([c]) == (
        "1 residual conflict(s) could not be removed; 1 of them between two pinned labels, "
        "which the placer cannot move"
    )


def test_two_pinned_labels_not_by_hand_get_no_editor_advice():
    a = _at("K2N", TEXT_A[1], TEXT_A[0].end, pinned=True)
    b = _at("A6S", TEXT_B[1], TEXT_B[0].end, pinned=True)
    line = str(residuals([a, b], K)[0])
    assert "label editor" not in line and line.endswith("both pinned, which the placer cannot move")


# ---- metrics ------------------------------------------------------------------------------------


def _pairs(layout, test):
    return sum(test(p, q) for i, p in enumerate(layout) for q in layout[i + 1 :])


def test_metrics_count_with_the_search_predicates():
    m = metrics(LAYOUT, _scene(), K)
    assert (m["text_over_text"], m["leader_over_text"], m["leader_over_leader"]) == (1, 1, 1)
    assert m["text_over_text"] == _pairs(LAYOUT, lambda p, q: pr.text_clash(p, q, K))
    assert m["leader_over_leader"] == _pairs(LAYOUT, pr.leaders_cross)
    assert m["leader_over_text"] == sum(
        pr.leader_over_text(p, q, K) for p in LAYOUT for q in LAYOUT if q is not p
    )
    clean = metrics(CLEAN, _scene(), K)
    assert (clean["text_over_text"], clean["leader_over_text"], clean["leader_over_leader"]) == (
        0,
        0,
        0,
    )


def test_text_over_branch_and_name_use_the_bare_box_strictly():
    a = TEXT_A[1]
    strokes = [
        (a[0] - 5, a[1] + 2, a[0] + 1, a[1] + 2.5),
        (a[2], a[1], a[2] + 4, a[3]),
    ]  # 2nd touches
    names = [(a[0] + 1, a[1] + 1, a[0] + 3, a[1] + 3), (a[0] + 2, a[1] + 2, a[0] + 4, a[1] + 4)]
    m = metrics([TEXT_A, OVER_C], _scene(strokes, names), K)
    assert m["text_over_branch"] == 1  # one label, the touching stroke not counted
    assert m["text_over_name"] == 1  # two names on one label count once


def test_lengths_and_angles_use_ae_upper_median():
    end = (500.0, 500.0)
    labs = []
    for n, (dx, dy) in enumerate([(10.0, 0.0), (20.0, 0.0), (30.0, 30.0), (40.0, -5.0)]):
        lab = _label("K2N", (end[0], end[1] + 100 * n))
        labs.append((lab, (0.0, 0.0, 1.0, 1.0), (lab.end[0] - dx, lab.end[1] - dy)))
    m = metrics(labs, _scene(), K, tiers=[1, 2, 3, 2])
    lens = sorted([10.0, 20.0, 30.0 * 2**0.5, (40.0**2 + 25.0) ** 0.5])
    assert m["leader_len_med_pt"] == round(lens[2], 2)  # v[n/2], not numpy's mean of two
    assert m["leader_len_max_pct"] == round(100 * lens[3] / 1000.0, 2)
    assert m["angle_min_deg"] == 0.0 and m["angle_med_deg"] == 7.13  # atan(5/40)
    assert m["below_22deg"] == 3
    assert m["not_ne_sw"] == 1  # the one whose attach point is above its end (dy > 0)
    assert (m["off_envelope"], m["band_sweep"]) == (2, 1)
    assert m["over_len_max"] == 0 and m["len_max_pt"] == 110.0
    assert "leader_cells_med" not in m


def test_leader_cells_and_pinned():
    pinned = _at("K2N", TEXT_A[1], TEXT_A[0].end, pinned=True)
    cells = iter([5, 1, 9])
    m = metrics([pinned, OVER_C, NODE_E], _scene(), K, leader_cells=lambda a, b: next(cells))
    assert (m["leader_cells_med"], m["leader_cells_max"], m["pinned"]) == (5, 9, 1)


def test_metrics_refuse_tiers_of_the_wrong_length():
    with pytest.raises(ValueError, match="tiers"):
        metrics(CLEAN, _scene(), K, tiers=[1])


# ---- glyph gap ----------------------------------------------------------------------------------

P = (100.0, 100.0, 117.1, 111.21)  # glyph line y 102.185-109.025


@pytest.mark.parametrize(
    "other, gap",
    [
        ((100.0, 111.71, 117.1, 122.92), 4.87),  # stacked: box gap 0.5 + 2 x 0.23 fs
        ((118.1, 100.0, 135.2, 111.21), 1.0),  # side by side
        ((120.0, 115.0, 137.1, 126.21), 8.16),  # diagonal: the larger axis gap, not hypot
        ((115.1, 108.21, 132.2, 119.42), 1.37),  # boxes overlap 2 x 3, glyphs do not
        ((110.0, 104.0, 127.1, 115.21), 0.0),  # glyphs overlap
    ],
)
def test_glyph_gap(other, gap):
    p, q = _at("K2N", P, (300.0, 0.0)), _at("A6S", other, (300.0, 300.0))
    assert glyph_gap(p, q) == pytest.approx(gap)
    assert metrics([p, q], _scene(), K)["min_glyph_gap_pt"] == round(gap, 2)


def test_glyph_gap_beside_a_short_line():
    two = _at("K2N\nA", (100.0, 100.0, 117.1, 122.42), (300.0, 0.0))  # 2nd line 5.7 wide
    other = _at("L7F", (107.0, 112.0, 124.1, 123.21), (300.0, 300.0))
    # vs line 1 (cap bottom 109.025): gy = 114.185 - 109.025; vs line 2 (x1 105.7): gx = 1.3
    assert glyph_gap(two, other) == pytest.approx(1.3)


# ---- score --------------------------------------------------------------------------------------


def test_score_is_the_weighted_sum_of_its_parts():
    # A and B clash (text); C sits above B's branch with its box below B's: an inversion
    c = _at("L7F", _box(60.0, 140.0), (120.0, 110.0))
    layout, base = [TEXT_A, TEXT_B, c], [10.5, 0.0, 3.25]
    opts = Options()
    s = score(layout, K, opts, base)
    conf = _pairs(layout, lambda p, q: pr.pconf(p, q, K))
    soft = sum(pr.psoft(p, q, K) for i, p in enumerate(layout) for q in layout[i + 1 :])
    inv = _pairs(layout, pr.inv)
    assert (s["conf_total"], s["inv_total"]) == (conf, inv)
    assert conf >= 1 and inv >= 1 and soft > 0
    assert s["soft_total"] == pytest.approx(soft)
    assert s["base_total"] == 13.75
    assert s["score_a"] == pytest.approx(1e6 * conf + 5000 * soft + 1100 * inv + 13.75)


def test_score_refuses_a_base_per_label_mismatch():
    with pytest.raises(ValueError, match="base"):
        score(CLEAN, K, Options(), [0.0])


# ---- af's counters ------------------------------------------------------------------------------


def test_af_counters_pad_boxes_as_af_does():
    a = _at("K2N", P, (300.0, 105.605))
    b = _at("A6S", (118.6, 140.0, 135.7, 151.21), (300.0, 160.0))  # 1.5 right, far below
    side = _at("L7F", (118.6, 100.0, 135.7, 111.21), (118.0, 300.0))  # 1.5 pt beside a
    m = af_counters([a, b, side], _scene())
    assert m["labels"] == 3
    assert m["box_overlaps"] == 1  # a and side: 1 pt pads overlap across a 1.5 pt gap
    assert m["leader_crossings"] == _pairs([a, b, side], lambda p, q: pr.leaders_cross(p, q))
    assert "labels_beside" not in m


def test_af_counters_glyph_rules_with_a_beside_gap():
    a = _at("K2N", P, (300.0, 0.0))
    near = _at("A6S", (125.0, 100.0, 142.1, 111.21), (300.0, 50.0))  # 7.9 pt beside
    far = _at("L7F", (127.0, 300.0, 144.1, 311.21), (300.0, 350.0))
    m = af_counters([a, near, far], _scene(), beside_gap=8.5)
    assert m["labels_beside"] == 1
    assert m["leaders_through_glyphs"] == 0


def test_af_counters_ink_from_a_grid_or_the_strokes():
    stroke = (P[2] + 0.5, P[1], P[2] + 0.8, P[3])  # clear of the text box, under af's pad
    a = _at("K2N", P, (300.0, 105.605))
    assert metrics([a], _scene([stroke]), K)["text_over_branch"] == 0
    assert af_counters([a], _scene([stroke]))["labels_on_tree_ink"] == 1
    assert af_counters([a], _scene([stroke]), ink=lambda box: 0)["labels_on_tree_ink"] == 0
