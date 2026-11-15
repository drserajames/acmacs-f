"""The pair matrices equal predicates.pconf / psoft element for element, on random scenes and on
the knife edges where rounding decides; pruned pairs really are zero."""

from __future__ import annotations

import math
import time

import numpy as np
import pytest

from af.tree.draw.placer_ae.geometry import attach_pt, glyph_rects, grow
from af.tree.draw.placer_ae.pairs import Pairs, _hypot_below, build_pairs, reach
from af.tree.draw.placer_ae.predicates import Placed, consts, pconf, psoft
from af.tree.draw.placer_ae.scene import ROW, Cands, Label, Options, Scene

FS = 9.5
TEXTS = ("K2N", "A6S", "L7F\nT9I", "K2N\nA6S\nL7F", "S3P", "N1D\nE5K")
Row = tuple[float, float, float, float, float, float, float, int]


def _label(key: int, text: str, end: tuple[float, float], fs: float = FS, lw=None) -> Label:
    lines = tuple(text.split("\n"))
    lw = lw or tuple(0.6 * fs * len(t) for t in lines)
    return Label(key, text, lines, lw, fs, end, end)


def _scene(labels, fs: float = FS) -> Scene:
    return Scene(tuple(labels), (), (), (0.0, 0.0, 2000.0, 2000.0), 1000.0, fs)


def _row(label: Label, box, attach=None) -> Row:
    cx, cy = attach or attach_pt(label.end, label.fs, box)
    return (box[0], box[1], box[2], box[3], cx, cy, 0.0, 1)


def _placed(scene: Scene, cands, i: int, c: int) -> Placed:
    return (scene.labels[i], cands[i].box(c), (float(cands[i].cx[c]), float(cands[i].cy[c])))


def _reference(scene, cands, opts, i, j):
    k = consts(scene, opts)
    shape = (len(cands[i]), len(cands[j]))
    conf, soft = np.zeros(shape, dtype=np.int8), np.zeros(shape)
    for a in range(shape[0]):
        for b in range(shape[1]):
            p, q = _placed(scene, cands, i, a), _placed(scene, cands, j, b)
            conf[a, b], soft[a, b] = pconf(p, q, k), psoft(p, q, k)
    return conf, soft


def _check_all(scene, cands, opts, lazy: bool = False) -> Pairs:
    """Every ordered pair, neighbours or not, against the predicates."""
    pairs = build_pairs(scene, cands, opts, lazy_soft=lazy)
    for i in range(pairs.n):
        for j in range(pairs.n):
            if i == j:
                continue
            conf, soft = _reference(scene, cands, opts, i, j)
            np.testing.assert_array_equal(pairs.conf(i, j), conf, err_msg=f"conf {i},{j}")
            np.testing.assert_allclose(
                pairs.soft(i, j), soft, rtol=1e-9, atol=1e-12, err_msg=f"soft {i},{j}"
            )
            assert pairs.conf(i, j).dtype == np.int8
            if j not in pairs.neighbours(i):
                assert not conf.any() and not soft.any(), f"pruned {i},{j} is not zero"
    return pairs


def _random_scene(seed: int, n: int = 7, per: int = 14, snap: bool = False):
    """Labels on a short column of branches, candidates scattered around each: dense enough
    that most kinds of term fire, with a far label so that something is pruned. Snapped
    coordinates (quarter points) make exact ties: touching boxes, collinear leaders."""
    rng = np.random.default_rng(seed)
    labels, cands = [], []
    for i in range(n):
        fs = FS if i % 3 else 8.0
        end = (200.0 + rng.uniform(-10, 10), 100.0 + 9.0 * i + rng.uniform(-3, 3))
        if i == n - 1:
            end = (900.0, 900.0)  # far from everything
        if snap:
            end = (round(end[0] * 4) / 4, round(end[1] * 4) / 4)
        label = _label(i, TEXTS[i % len(TEXTS)], end, fs)
        rows = []
        for _ in range(per):
            x0 = end[0] - rng.uniform(5, 60) - label.tw
            y0 = end[1] + rng.uniform(-35, 25)
            if snap:
                x0, y0 = round(x0 * 4) / 4, round(y0 * 4) / 4
            box = (x0, y0, x0 + label.tw, y0 + label.th)
            attach = None
            if rng.random() < 0.3:
                attach = (x0 + rng.uniform(-5, label.tw + 5), y0 + rng.uniform(-5, label.th + 5))
            rows.append(_row(label, box, attach))
        labels.append(label)
        cands.append(Cands.of(rows))
    return _scene(labels), cands


OPTS = [
    Options(label_clash=mode, beside_gap=gap, stack_tolerance=tol)
    for mode in ("ink", "main")
    for gap in (None, 9.0)
    for tol in (0.0, 1e-6)
] + [
    Options(label_clash=mode, beside_gap=gap, overlap_weight=8)
    for mode in ("ink", "main")
    for gap in (None, 9.0)
]


@pytest.mark.parametrize(
    "opts",
    OPTS,
    ids=lambda o: f"{o.label_clash}-{o.beside_gap}-{o.stack_tolerance}-{o.overlap_weight}",
)
@pytest.mark.parametrize("seed", [1, 2, 3])
@pytest.mark.parametrize("snap", [False, True])
@pytest.mark.parametrize("lazy", [False, True])
def test_random_scenes_equal_the_predicates(opts, seed, snap, lazy):
    scene, cands = _random_scene(seed, snap=snap)
    pairs = _check_all(scene, cands, opts, lazy)
    assert len(pairs.neighbours(pairs.n - 1)) == 0  # the far label was pruned
    hits = sum(int(pairs.conf(i, j).any()) for i in range(pairs.n) for j in pairs.neighbours(i))
    assert hits > 0


def _two(boxes_attaches, ends, fs: float = FS, texts=("K2N", "A6S"), lws=(None, None)):
    """Two labels with one or more hand-made candidates each."""
    labels = [_label(n, texts[n], ends[n], fs, lws[n]) for n in range(2)]
    cands = [
        Cands.of([_row(labels[n], box, att) for box, att in boxes_attaches[n]]) for n in range(2)
    ]
    return _scene(labels, fs), cands


def _each_opts(scene, cands):
    for opts in OPTS:
        _check_all(scene, cands, opts)
        _check_all(scene, cands, opts, lazy=True)


def test_touching_stacked_boxes():
    # 2 x 0.23 fs of leading: the glyph gap is vsep to the last bit (spec §3.3, oracle §8.5)
    a = (100.0, 100.0, 117.1, 111.21)
    b = (100.0, 111.21, 117.1, 122.42)
    scene, cands = _two([[(a, None)], [(b, None)]], [(175.0, 100.0), (300.0, 160.0)])
    k0 = consts(scene, Options(stack_tolerance=0.0))
    k1 = consts(scene, Options(stack_tolerance=1e-6))
    p, q = _placed(scene, cands, 0, 0), _placed(scene, cands, 1, 0)
    assert pconf(p, q, k0) != pconf(p, q, k1)  # the case really is on the edge
    _each_opts(scene, cands)


def test_shared_leader_end_and_collinear_leaders():
    end = (175.0, 120.0)
    a = (100.0, 100.0, 117.1, 111.21)
    b = (100.0, 130.0, 117.1, 141.21)
    on_line = [((90.0, 140.0, 107.1, 151.21), (150.0, 120.0))]  # attach on the end's row
    scene, cands = _two(
        [[(a, None), on_line[0]], [(b, None), ((60.0, 140.0, 77.1, 151.21), (130.0, 120.0))]],
        [end, end],
    )
    k = consts(scene, Options())
    p, q = _placed(scene, cands, 0, 1), _placed(scene, cands, 1, 1)
    assert pconf(p, q, k) == 0  # collinear leaders from one end do not cross
    assert pconf(_placed(scene, cands, 0, 0), _placed(scene, cands, 1, 0), k) == 1
    _each_opts(scene, cands)


def test_collinear_leaders_on_one_line_from_two_ends():
    a = ((40.0, 140.0, 57.1, 151.21), (100.0, 120.0))
    b = ((10.0, 160.0, 27.1, 171.21), (80.0, 120.0))
    scene, cands = _two([[a], [b]], [(175.0, 120.0), (110.0, 120.0)])
    _each_opts(scene, cands)


def test_leader_ending_exactly_on_a_box_edge():
    scene0, _ = _two([[((0, 0, 1, 1), None)], [((0, 0, 1, 1), None)]], [(0, 0), (0, 0)])
    k = consts(scene0, Options())
    b = (100.0, 100.0, 117.1, 111.21)
    g = grow(b, k.mt)
    rect = grow(glyph_rects(b, (0.6 * FS * 3,), FS)[0], k.mt)
    attaches = [
        (g[2], 105.0),  # on the grown box's right edge
        (rect[2], rect[1]),  # on a grown glyph line's corner
        (g[2], g[3]),  # on the grown box's corner, outside the grown glyph line
        (math.nextafter(g[2], 1e9), 105.0),  # one ulp off
    ]
    rows = [((130.0, 60.0 + 15 * n, 147.1, 71.21 + 15 * n), att) for n, att in enumerate(attaches)]
    scene, cands = _two([rows, [(b, None)]], [(200.0, 105.0), (250.0, 200.0)])
    _each_opts(scene, cands)


def test_leaders_exactly_leadthr_apart():
    k = consts(
        _two([[((0, 0, 1, 1), None)], [((0, 0, 1, 1), None)]], [(0, 0), (0, 0)])[0], Options()
    )
    assert k.leadthr == 4.75
    a = ((100.0, 70.0, 117.1, 81.21), (150.0, 100.0))
    b = [
        ((100.0, 130.0, 117.1, 141.21), (150.0, 104.75)),  # parallel, exactly leadthr
        ((100.0, 150.0, 117.1, 161.21), (150.0, math.nextafter(104.75, 0))),
    ]
    scene, cands = _two([[a], b], [(200.0, 100.0), (200.0, 104.75)])
    _each_opts(scene, cands)


def _nudge_to(target: float, y: float, f) -> float:
    """The y whose computed f(y) is exactly target, searching a few hundred ulps."""
    for step in range(400):
        for v in (y + step * math.ulp(y), y - step * math.ulp(y)):
            if f(v) == target:
                return v
    raise AssertionError("no exact knife edge found")


@pytest.mark.parametrize("ulps", [-1, 0, 1])
def test_diagonal_glyph_gap_on_the_side_threshold(ulps):
    # ink, beside gap 5: two glyph lines 3 apart in x and ~4 in y, hypot on 5 to the last bit
    fs, lw = 10.0, (16.0,)
    a = (100.0, 100.0, 116.0, 100.0 + ROW * fs)
    ra = glyph_rects(a, lw, fs)[0]

    def gy(y0: float) -> float:
        return glyph_rects((119.0, y0, 135.0, y0 + ROW * fs), lw, fs)[0][1] - ra[3]

    y0 = _nudge_to(4.0, ra[3] + 4.0 - 0.23 * fs, gy)
    y0 += ulps * math.ulp(y0)
    b = (119.0, y0, 135.0, y0 + ROW * fs)
    scene, cands = _two(
        [[(a, (116.0, 105.0))], [(b, (135.0, y0 + 5.0))]],
        [(180.0, 60.0), (260.0, 250.0)],
        fs=fs,
        lws=(lw, lw),
    )
    for opts in (Options(beside_gap=5.0, stack_tolerance=t) for t in (0.0, 1e-6)):
        _check_all(scene, cands, opts)


def test_hypot_near_a_threshold_rounds_as_math_hypot():
    rng = np.random.default_rng(11)
    x = rng.uniform(0, 5, 20000)
    y = np.sqrt(np.maximum(25.0 - x * x, 0.0)) + rng.integers(-3, 4, x.size) * 1e-15
    h, below = _hypot_below(x, y, 5.0)
    exact = np.array([math.hypot(a, b) for a, b in zip(x.tolist(), y.tolist(), strict=True)])
    np.testing.assert_array_equal(below, exact < 5.0)
    near = np.abs(exact - 5.0) <= 1e-14 * 5.0
    np.testing.assert_array_equal(h[near], exact[near])


def test_pruning_bound_is_conservative():
    """A second label swept out past the reach of every term: whenever the pair is pruned the
    predicates give zero, and it is kept while a term can still fire."""
    for opts in OPTS:
        k = consts(_scene([]), opts)
        r = reach(k)
        a = _label(0, "K2N\nA6S", (200.0, 100.0))
        box_a = (150.0, 90.0, 150.0 + a.tw, 90.0 + a.th)
        for dx in np.linspace(r * 0.5, r * 1.5, 21):
            b = _label(1, "L7F", (400.0, 100.0))
            x0 = 150.0 + a.tw + dx
            box_b = (x0, 90.0, x0 + b.tw, 90.0 + b.th)
            scene = _scene([a, b])
            cands = [
                Cands.of([_row(a, box_a, (box_a[2], 95.0))]),
                Cands.of([_row(b, box_b, (box_b[0], 95.0))]),
            ]
            _check_all(scene, cands, opts)


def test_symmetry():
    scene, cands = _random_scene(5)
    pairs = build_pairs(scene, cands, Options())
    for i in range(pairs.n):
        for j in pairs.neighbours(i):
            assert i in pairs.neighbours(j)
            np.testing.assert_array_equal(pairs.conf(j, i), pairs.conf(i, j).T)
            np.testing.assert_array_equal(pairs.soft(j, i), pairs.soft(i, j).T)
        assert list(pairs.neighbours(i)) == sorted(pairs.neighbours(i))
    with pytest.raises(ValueError):
        pairs.conf(0, 0)
    i, j = 0, pairs.neighbours(0)[0]
    with pytest.raises(ValueError):
        pairs.conf(i, j)[0, 0] = 9  # read-only: the tables are shared


def test_a_label_without_candidates_has_no_neighbours():
    scene, cands = _random_scene(4, n=3)
    cands[1] = Cands.of([])
    pairs = _check_all(scene, cands, Options())
    assert pairs.neighbours(1) == ()
    assert pairs.conf(0, 1).shape == (len(cands[0]), 0)


def _synthetic(n: int, per: int, seed: int = 7):
    """A report-sized scene: n labels down a tree column, per candidates each, spread as ae's
    sampling spreads them (left of the branch, up to ~100 pt away)."""
    rng = np.random.default_rng(seed)
    labels, cands = [], []
    for i in range(n):
        end = (300.0 + rng.uniform(-80, 80), 40.0 + 15.0 * i)
        label = _label(i, TEXTS[i % len(TEXTS)], end)
        rows = []
        for _ in range(per):
            ang = math.radians(rng.uniform(-80, 80))
            length = rng.uniform(5, 110)
            ax, ay = end[0] - length * math.cos(ang), end[1] - length * math.sin(ang)
            box = (ax - label.tw, ay - label.th / 2, ax, ay + label.th / 2)
            rows.append(_row(label, box))
        labels.append(label)
        cands.append(Cands.of(rows))
    return _scene(labels), cands


def test_build_time_report_sized():
    """Prints (does not assert) the build time and size for 60 labels x 190 candidates."""
    scene, cands = _synthetic(60, 190)
    for mode in ("ink", "main"):
        for lazy in (False, True):
            t = time.perf_counter()
            pairs = build_pairs(scene, cands, Options(label_clash=mode), lazy_soft=lazy)
            dt = time.perf_counter() - t
            npairs = sum(len(pairs.neighbours(i)) for i in range(pairs.n)) // 2
            print(
                f"\npairs {mode} lazy={lazy}: 60 labels x 190 candidates: {npairs} pairs, "
                f"{dt:.2f} s, {pairs.nbytes / 1e6:.1f} MB"
            )
    # spot-check one stored pair against the predicates
    i = next(i for i in range(pairs.n) if pairs.neighbours(i))
    j = pairs.neighbours(i)[0]
    small = [Cands.of([_row_of(cands[n], c) for c in range(0, 190, 19)]) for n in range(60)]
    sub = build_pairs(scene, small, Options(label_clash="main"))
    conf, soft = _reference(scene, small, Options(label_clash="main"), i, j)
    np.testing.assert_array_equal(sub.conf(i, j), conf)
    np.testing.assert_allclose(sub.soft(i, j), soft, rtol=1e-9, atol=1e-12)


def _row_of(c: Cands, n: int) -> Row:
    return (*c.box(n), float(c.cx[n]), float(c.cy[n]), float(c.base[n]), int(c.tier[n]))
