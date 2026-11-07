"""ae's geometry predicates, on the worked cases of the port's spec (09's notes,
oracles/small-cases: derived by hand from draw-tree.cc, not run in ae). Invented coordinates;
label size 9.5."""

import math

import pytest

from af.tree.draw.placer_ae.geometry import (
    attach_pt,
    glyph_rects,
    grow,
    pt_seg_d,
    rects_clash,
    seg_box,
    seg_box_len,
    seg_seg_d,
    segs_cross,
)

FS = 9.5
M, VSEP = 0.08 * FS, FS * (1.18 - 0.72)


@pytest.mark.parametrize(
    "a, b, c, d, crossed",
    [
        ((175, 500), (145, 511), (175, 500), (140, 530), True),  # one node: always "cross"
        ((175, 500), (145, 510), (175, 500), (115, 520), False),  # ... unless collinear
        ((175, 500), (145, 511), (175, 520), (145, 531), False),
        ((175, 500), (145, 530), (175, 530), (145, 500), True),
        ((100, 100), (200, 100), (150, 100), (150, 50), True),  # touching counts
    ],
)
def test_segments_cross_as_ae_counts_it(a, b, c, d, crossed):
    assert segs_cross(a, b, c, d) is crossed


@pytest.mark.parametrize(
    "box, expected",
    [
        ((60.0, 40.0, 80.0, 51.21), (80.0, 45.605)),  # left of the branch: end of text, mid-height
        ((90.0, 30.0, 110.0, 41.21), (100.0, 39.025)),  # wholly above: bottom cap edge
        ((90.0, 55.0, 110.0, 66.21), (100.0, 57.185)),  # below: top cap edge
        ((90.0, 45.0, 110.0, 56.21), (100.0, 47.185)),  # straddling: top cap edge
    ],
)
def test_attach_point(box, expected):
    assert attach_pt((100.0, 50.0), FS, box) == pytest.approx(expected)


P = (100.0, 100.0, 117.1, 111.21)


def _ink(a, wa, b, wb, tolerance=0.0):
    if not (a[0] - M < b[2] and b[0] - M < a[2] and a[1] - VSEP < b[3] and b[1] - VSEP < a[3]):
        return False
    return any(
        rects_clash(ra, rb, M, VSEP, tolerance)
        for ra in glyph_rects(a, wa, FS)
        for rb in glyph_rects(b, wb, FS)
    )


@pytest.mark.parametrize(
    "other, clash",
    [
        ((117.6, 100.0, 134.7, 111.21), True),  # 0.5 apart side by side: under m
        ((118.1, 100.0, 135.2, 111.21), False),  # 1.0 apart
        ((100.0, 111.71, 117.1, 122.92), False),  # boxes 0.5 apart stacked: glyphs 4.87 apart
        ((115.1, 108.21, 132.2, 119.42), True),
        ((100.0, 111.11, 117.1, 122.32), True),  # boxes overlapping by 0.1 stacked
    ],
)
def test_the_ink_rule_between_glyph_lines(other, clash):
    assert _ink(P, (17.1,), other, (17.1,)) is clash


def test_touching_boxes_stacked_sit_on_a_float_knife_edge():
    # glyph gap 2 x 0.23 sizes = vsep, to the last bit: ae's arithmetic calls it a clash, af's
    # tolerance (Options.stack_tolerance, as render._ink_clash) does not
    touching = (100.0, 111.21, 117.1, 122.42)
    assert _ink(P, (17.1,), touching, (17.1,)) is True
    assert _ink(P, (17.1,), touching, (17.1,), tolerance=1e-6) is False


def test_a_label_may_sit_beside_a_short_line():
    two, beside = (100.0, 100.0, 117.1, 122.42), (107.0, 112.0, 124.1, 123.21)
    assert _ink(two, (17.1, 5.7), beside, (17.1,)) is False
    a, b = grow(two, M), beside
    # ae main's box rule calls it a clash
    assert a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def test_glyph_rects_are_capitals_high_a_row_apart():
    a, b = glyph_rects((10.0, 20.0, 40.0, 42.42), (30.0, 12.0), FS)
    assert a == pytest.approx(
        (10.0, 20.0 + 0.59 * FS - 0.36 * FS, 40.0, 20.0 + 0.59 * FS + 0.36 * FS)
    )
    assert b[1] - a[1] == pytest.approx(1.18 * FS) and b[2] == pytest.approx(22.0)


def test_distances_and_lengths():
    assert pt_seg_d((0.0, 5.0), (-10.0, 0.0), (10.0, 0.0)) == pytest.approx(5.0)
    assert pt_seg_d((3.0, 4.0), (0.0, 0.0), (0.0, 0.0)) == pytest.approx(5.0)  # a point segment
    assert seg_seg_d((0, 0), (10, 0), (0, 3), (10, 3)) == pytest.approx(3.0)
    assert seg_seg_d((0, 0), (10, 10), (0, 10), (10, 0)) == 0.0
    assert seg_box_len((0.0, 5.0), (20.0, 5.0), (5.0, 0.0, 10.0, 10.0)) == pytest.approx(5.0)
    assert seg_box_len((0.0, 0.0), (10.0, 10.0), (20.0, 20.0, 30.0, 30.0)) == 0.0
    diag = seg_box_len((0.0, 0.0), (10.0, 10.0), (0.0, 0.0, 5.0, 5.0))
    assert diag == pytest.approx(5 * math.sqrt(2))
    assert seg_box((0.0, 0.0), (1.0, 1.0), (0.5, 0.5, 2.0, 2.0))
    assert not seg_box((0.0, 0.0), (1.0, 0.0), (0.0, 1.0, 1.0, 2.0))
