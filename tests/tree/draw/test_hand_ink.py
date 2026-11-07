"""Hand labels checked glyph line to glyph line, as ae's ink rule (ae ed4a2ab; Q306, Q307):
stacked one stack's row gap (0.46 sizes at a 1.18 pitch), side by side and diagonally 0.08
sizes, leaders within 0.05 sizes of a line. Pin clearing and the placer keep their own rule."""

from dataclasses import replace

import pytest

pytest.importorskip("numpy", reason="numpy not installed: af.tree.draw needs it (pyproject, WS1)")
pytest.importorskip("matplotlib", reason="matplotlib not installed: af.tree.draw needs it")

from af.tree.draw.place import Placed  # noqa: E402
from af.tree.draw.render import (  # noqa: E402
    Geometry,
    HandLabelClash,
    _glyph_box,
    _glyph_lines,
    _hits,
    _ink_clash,
    hand_clashes,
    hand_leaders,
)

from .test_label_editor import _layout, _pin_at  # noqa: E402

FS = 10.0
G = Geometry(label_font=FS, label_pitch=1.18, pin_clearance=1.0, label_beside_gap=9.0)
EPS = 0.01


def width(text: str) -> float:  # an invented mono font: 6 units a character
    return 6.0 * len(text)


def _box(x_right: float, yc: float, text: str) -> tuple[float, float, float, float]:
    """A placer box whose glyphs end at ``x_right``, centred on ``yc``."""
    n = len(text.split("\n"))
    h = FS * 1.2 * n + 1
    w = max(width(t) for t in text.split("\n"))
    return (x_right + 1 - w - 2, yc - h / 2, x_right + 1, yc + h / 2)


def _lines(x_right: float, yc: float, text: str):
    return _glyph_lines(_box(x_right, yc, text), text, width, G)


def test_each_line_is_its_own_width_capitals_high_a_pitch_apart():
    a, b = _lines(100.0, 50.0, "K2N\nA6S L7F")
    cy = 50 - 0.59 * FS  # half a pitch above the centre
    assert a == pytest.approx((100.0 - 18, cy - 0.36 * FS, 100.0, cy + 0.36 * FS))
    assert b[0] == pytest.approx(100.0 - 42) and b[1] - a[1] == pytest.approx(1.18 * FS)
    # the lines sit inside the one rect pin clearing uses
    block = _glyph_box(_box(100.0, 50.0, "K2N\nA6S L7F"), "K2N\nA6S L7F", width, G)
    assert block[0] == pytest.approx(b[0]) and block[1] == pytest.approx(a[1])
    assert block[3] == pytest.approx(b[3])


@pytest.mark.parametrize("gap, clash", [(0.46 * FS - EPS, True), (0.46 * FS + EPS, False)])
def test_stacked_labels_keep_a_stacks_row_gap(gap, clash):
    upper = _lines(100.0, 50.0, "K2N")
    lower = _lines(100.0, 50.0 + 0.72 * FS + gap, "K2N")
    assert _ink_clash(upper, lower, G) is clash


@pytest.mark.parametrize("gap, clash", [(0.08 * FS - EPS, True), (0.08 * FS + EPS, False)])
def test_labels_side_by_side_keep_0_08_sizes(gap, clash):
    left = _lines(100.0, 50.0, "K2N")
    right = _lines(100.0 + gap + 18, 50.0, "A6S")
    assert _ink_clash(left, right, G) is clash
    # pin clearing's beside rule still calls the same pair too close
    assert _hits(left[0], right[0], G.label_beside_gap or 0.0, 0.46 * FS)


@pytest.mark.parametrize("k, clash", [(0.9, True), (1.1, False)])
def test_diagonal_neighbours_keep_0_08_sizes_corner_to_corner(k, clash):
    d = 0.08 * FS * k / 2**0.5
    a = _lines(100.0, 50.0, "K2N")
    b = _lines(100.0 + d + 18, 50.0 + 0.72 * FS + d, "A6S")
    assert _ink_clash(a, b, G) is clash


def test_a_label_may_sit_beside_a_short_line_of_a_longer_label():
    # the two-line label's first line is 18 wide, its second 42: a label ending 2 units left of
    # the short line, on its row, is inside the label's one rect but clear of every line
    two = _lines(100.0, 50.0, "K2N\nA6S L7F")
    beside = _lines(100.0 - 18 - 2, two[0][1] + 0.36 * FS, "L7F")
    assert not _ink_clash(two, beside, G)
    block = _glyph_box(_box(100.0, 50.0, "K2N\nA6S L7F"), "K2N\nA6S L7F", width, G)
    assert _hits(block, beside[0], 0.0)


def _placed(key, text, x_right, yc, anchor):
    return Placed(key, text, anchor, _box(x_right, yc, text), 0.0)


def test_hand_clashes_name_touching_lines_only():
    a = _placed(1, "K2N", 100.0, 50.0, (130.0, 50.0))
    near = _placed(2, "A6S", 100.0 + 3 + 18, 50.0, (160.0, 50.0))  # 3 apart: was refused at 9
    on = _placed(3, "L7F", 100.0 + 0.5 + 18, 80.0, (160.0, 80.0))
    b = _placed(4, "K2N", 100.0, 80.0, (130.0, 80.0))
    hand = {a.box, b.box}
    assert hand_clashes([a, near, b, on], hand, [], width, G) == ["K2N: against L7F"]
    strain = (60.0, 46.0, 81.5, 54.0)  # 0.5 left of the label's glyph line, inside the clearance
    assert hand_clashes([a], {a.box}, [strain], width, G) == ["K2N: against a strain label"]


@pytest.mark.parametrize("off, through", [(0.05 * FS - EPS, True), (0.05 * FS + EPS, False)])
def test_a_leader_is_through_a_label_within_0_05_sizes_of_a_line(off, through):
    label = _placed(1, "K2N", 100.0, 50.0, (130.0, 50.0))
    top = _glyph_lines(label.box, "K2N", width, G)[0][1]
    # a leader running level just above the label's capitals, from a label far to the left
    other = Placed(2, "A6S", (110.0, top - off), (10.0, top - off - 6, 40.0, top - off + 6), 0.0)
    found = hand_leaders([label, other], {label.box}, width, G)
    assert (found == ["A6S's leader through K2N"]) is through


def test_two_hand_labels_side_by_side_3_units_apart_are_drawn(tmp_path):
    # Q306: refused under the 9-unit beside rule, drawn under the ink rule; 0.5 apart: refused
    _, rows = _layout(tmp_path)
    height = Geometry().height
    geometry = replace(Geometry(), label_beside_gap=9.0, pin_clearance=1.0)
    k2n = _pin_at(rows["K2N"], 20.0, 150.0, height)
    w = rows["K2N"]["glyph_w"]
    for gap, refused in ((3.0, False), (0.5, True)):
        l7f = _pin_at(rows["L7F"], 20.0 + w + gap, 150.0, height)
        if refused:
            with pytest.raises(HandLabelClash, match="K2N: against L7F"):
                _layout(tmp_path, [k2n, l7f], geometry=geometry)
            continue
        report, after = _layout(tmp_path, [k2n, l7f], geometry=geometry)
        assert after["L7F"]["box"][0] == pytest.approx(20.0 + w + gap, abs=2e-3)
        assert report["label_pins"]["pinned by hand"] == 2
