"""ae's pairwise terms on the spec's worked cases (oracles/small-cases, derived from
draw-tree.cc)."""

import math

import pytest

from af.tree.draw.placer_ae.predicates import (
    angle,
    consts,
    inv,
    leader_over_text,
    leaders_cross,
    pconf,
    psoft,
    text_clash,
)
from af.tree.draw.placer_ae.scene import Label, Options, Scene

FS = 9.5


def _scene():
    return Scene((), (), (), (0.0, 0.0, 1000.0, 1000.0), 1000.0, FS)


def _label(text="K2N", end=(500.0, 100.0), lw=None):
    lines = tuple(text.split("\n"))
    return Label(0, text, lines, lw or tuple(0.6 * FS * len(t) for t in lines), FS, end, end)


K_MAIN = consts(_scene(), Options(label_clash="main"))
K_INK = consts(_scene(), Options())
P = (100.0, 100.0, 117.1, 111.21)


def _placed(box, end=(500.0, 100.0), attach=None, text="K2N", lw=None):
    return (_label(text, end, lw), box, attach or (box[2], (box[1] + box[3]) / 2))


@pytest.mark.parametrize(
    "other, main, ink",
    [
        ((117.6, 100.0, 134.7, 111.21), True, True),
        ((118.1, 100.0, 135.2, 111.21), False, False),
        # stacked: main's box margin calls it a clash, ink's glyph gap does not
        ((100.0, 111.71, 117.1, 122.92), True, False),
        ((115.1, 108.21, 132.2, 119.42), True, True),
    ],
)
def test_text_on_text_main_and_ink(other, main, ink):
    assert text_clash(_placed(P), _placed(other), K_MAIN) is main
    assert text_clash(_placed(P), _placed(other), K_INK) is ink


def test_a_beside_gap_widens_the_side_by_side_rule_only():
    k = consts(_scene(), Options(beside_gap=9.0))
    assert text_clash(_placed(P), _placed((125.0, 100.0, 142.1, 111.21)), k)  # 7.9 apart
    assert not text_clash(_placed(P), _placed((127.0, 100.0, 144.1, 111.21)), k)
    assert not text_clash(_placed(P), _placed((100.0, 111.71, 117.1, 122.92)), k)  # stacked


def test_vertical_order_and_angle():
    # i: branch at (175, 500), box below-left; j: branch at (260, 520), box below it (ok) or
    # above it (inverted)
    i = _placed((128.0, 505.0, 145.1, 516.21), end=(175.0, 500.0), attach=(145.1, 510.605))
    ok = _placed((210.0, 525.0, 227.1, 536.21), end=(260.0, 520.0), attach=(227.1, 530.605))
    bad = _placed((210.0, 480.0, 227.1, 491.21), end=(260.0, 520.0), attach=(227.1, 485.605))
    assert (inv(i, ok), inv(i, bad)) == (0, 1)
    assert angle(i) == pytest.approx(-0.3408400053823867)
    assert int(130 * abs(angle(i) - angle(ok))) == 3
    assert int(130 * abs(angle(i) - angle(bad))) == 149


def test_conflicts_count_each_kind_once():
    # two labels on one node: their leaders always "cross" (ae's touching rule)
    a = _placed((100.0, 100.0, 117.1, 111.21), end=(175.0, 120.0))
    b = _placed((100.0, 130.0, 117.1, 141.21), end=(175.0, 120.0))
    assert leaders_cross(a, b) and pconf(a, b, K_INK) == 1
    # a leader through the other's text
    c = _placed((150.0, 140.0, 167.1, 151.21), end=(200.0, 100.0), attach=(150.0, 145.6))
    d = _placed((160.0, 115.0, 177.1, 126.21), end=(300.0, 300.0))
    assert leader_over_text(c, d, K_INK)


def test_soft_penetration_is_zero_far_apart_and_grows_with_overlap():
    far = psoft(_placed(P, end=(200.0, 50.0)), _placed((300.0, 300.0, 317.1, 311.21)), K_MAIN)
    assert far == 0.0
    near = psoft(
        _placed(P, end=(200.0, 50.0)),
        _placed((115.1, 108.21, 132.2, 119.42), end=(400.0, 400.0)),
        K_MAIN,
    )
    assert near > 0.6899  # the area term alone (oracle) plus leader terms
    assert math.isfinite(near)
