"""ae's candidate generation (spec §2, §3.1) on the port's oracles (oracles/derive.md: an invented
3-stroke tree re-derived from the spec, not run in ae), plus the grid primitives against brute
force, the pinned / soft-ink / tier-3 / last-resort branches, determinism and a timing print.
Invented labels and coordinates only."""

from __future__ import annotations

import math
import time

import numpy as np
import pytest

from af.tree.draw.placer_ae import candidates as C
from af.tree.draw.placer_ae.geometry import attach_pt
from af.tree.draw.placer_ae.scene import Box, Cands, Label, Options, Scene, Weights

FS = 9.5
H = 1000.0
HW = 0.5
BAND = (30.0, 20.0, 400.0, 980.0)
# node N (200, 500), parent column x 150, sibling leaf (210, 560)
STROKES: tuple[Box, ...] = (
    (150.0, 500.0 - HW, 200.0, 500.0 + HW),
    (150.0, 560.0 - HW, 210.0, 560.0 + HW),
    (150.0 - HW, 500.0, 150.0 + HW, 560.0),
)
END = (175.0, 500.0)  # middle of N's horizontal edge

# oracles/candidates-one-label.json, rounded to 6 places as written there:
# (x0, y0, cx, cy, base, tier); x1 = x0 + 17.1, y1 = y0 + 11.21
FIRST12 = [
    (128.061333, 505.848991, 145.161333, 511.453991, 165.444016, 1),
    (123.558176, 510.335716, 140.658176, 515.940716, 210.754936, 1),
    (120.775815, 514.250639, 137.875815, 519.855639, 212.742268, 1),
    (116.497849, 517.81921, 133.597849, 523.42421, 248.050905, 1),
    (122.407676, 505.146142, 139.507676, 510.751142, 255.773985, 1),
    (125.463108, 515.298232, 142.563108, 520.903232, 266.132486, 1),
    (123.289971, 519.581412, 140.389971, 525.186412, 268.619768, 1),
    (128.061333, 500.243991, 145.161333, 505.848991, 277.244245, 1),
    (119.077214, 508.606804, 136.177214, 514.211804, 277.424055, 1),
    (111.858969, 519.553633, 128.958969, 525.158633, 289.603519, 1),
    (116.497849, 523.42421, 133.597849, 529.02921, 291.279244, 1),
    (116.085437, 512.584755, 133.185437, 518.189755, 295.545706, 1),
]
LAST3 = [
    (82.640557, 315.70513, 99.740557, 321.31013, 3885.150666, 2),
    (109.79719, 303.420042, 126.89719, 309.025042, 4051.692645, 2),
    (144.01811, 301.479729, 161.11811, 307.084729, 4060.034716, 2),
]


def label(
    text: str = "K2N", end: tuple[float, float] = END, key: int = 1, pinned: Box | None = None
) -> Label:
    """Lines split on spaces, 0.6 fs advance per character (the oracles' assumption)."""
    lines = tuple(text.split())
    return Label(
        key=key,
        text="\n".join(lines),
        lines=lines,
        lw=tuple(len(t) * (0.6 * FS) for t in lines),
        fs=FS,
        node=(end[0] + 25.0, end[1]),
        end=end,
        pinned=pinned,
    )


def scene(
    labels: tuple[Label, ...] = (),
    strokes: tuple[Box, ...] = STROKES,
    obstacles: tuple[Box, ...] = (),
) -> Scene:
    return Scene(
        labels=labels or (label(),),
        strokes=strokes,
        obstacles=obstacles,
        band=BAND,
        height=H,
        mrca_fs=FS,
    )


def rows(c: Cands) -> list[tuple[float, ...]]:
    """(x0, y0, cx, cy, base, tier) per candidate, rounded as the oracle file is."""
    cols = (c.x0, c.y0, c.cx, c.cy, c.base)
    return [tuple(round(float(v[i]), 6) for v in cols) + (int(c.tier[i]),) for i in range(len(c))]


# --------------------------------------------------------------------------- the oracle


def test_one_label_reproduces_the_oracle():
    sc = scene()
    stats: dict[str, int] = {}
    c = C.label_candidates(C.context(sc, Options()), sc.labels[0], stats)
    assert stats == {"raw": 3814, "near": 110, "final": 188}
    assert len(c) == 188
    assert [int((c.tier == t).sum()) for t in (1, 2)] == [153, 35]
    got = rows(c)
    assert got[:12] == FIRST12
    assert got[-3:] == LAST3
    np.testing.assert_allclose(c.x1 - c.x0, 17.1, rtol=0, atol=1e-12)
    np.testing.assert_allclose(c.y1 - c.y0, 11.21, rtol=0, atol=1e-12)
    assert np.all(np.diff(c.base) > 0)  # no ties in this oracle, and sorted


def test_make_candidates_is_label_candidates_per_label():
    sc = scene()
    (c,) = C.make_candidates(sc, Options())
    assert rows(c)[:12] == FIRST12


# --------------------------------------------------------------------------- shape cost

SHAPE = [  # oracles/small-cases.json "shape": (L, deg, down, base)
    (5.0, 20.0, True, 232.5),
    (10.0, 20.0, True, 125.0),
    (18.0, 20.0, True, 145.0),
    (31.5, 20.0, True, 90.86634349030474),
    (45.0, 20.0, True, 260.9653739612188),
    (60.0, 20.0, True, 367.27423822714684),
    (30.0, 10.0, False, 423.4622960911049),
]


@pytest.mark.parametrize("length, deg, down, base", SHAPE)
def test_shape_cost_rows(length, deg, down, base):
    got = C.shape_cost(length, deg * C.PI / 180.0, FS, down, Weights(), H)
    assert float(got) == pytest.approx(base, rel=1e-12)


def test_shape_cost_vectorises():
    ls = np.array([r[0] for r in SHAPE])
    th = np.array([r[1] for r in SHAPE]) * C.PI / 180.0
    dn = np.array([r[2] for r in SHAPE])
    got = C.shape_cost(ls, th, FS, dn, Weights(), H)
    np.testing.assert_allclose(got, [r[3] for r in SHAPE], rtol=1e-12)


# --------------------------------------------------------------------------- grid primitives


def _brute_free(g: C.OccGrid, rx: float, ry: float, rw: float, rh: float) -> bool:
    if rx < g.gx0 or rx + rw > g.gx1 or ry < g.gy0 or ry + rh > g.gy1:
        return False
    sub = g.occ[g.row(ry) : g.row(ry + rh) + 1, g.col(rx) : g.col(rx + rw) + 1]
    return not sub.any()


def _brute_walk(g: C.OccGrid, x0: float, y0: float, x1: float, y1: float) -> int:
    """oracles/derive.md's leader_ink_cells, line for line."""
    steps = min(max(int(math.hypot(x1 - x0, y1 - y0) / g.cell) + 1, 1), 2000)
    hits, pr, pc = 0, -1, -1
    for k in range(steps + 1):
        t = k / steps
        r, c = g.row(y0 + (y1 - y0) * t), g.col(x0 + (x1 - x0) * t)
        if r == pr and c == pc:
            continue
        pr, pc = r, c
        hits += int(g.occ[r, c] != 0)
    return hits


def test_grid_geometry_as_ae():
    g = C.tree_grid(scene())
    assert (g.GX, g.GY) == (486, 1263)
    assert g.cell == pytest.approx(0.76)
    # mark_h: one row at the stroke's centre; mark_v: one column
    assert g.occ[g.row(500.0), g.col(150.0) : g.col(200.0) + 1].all()
    assert g.occ[:, g.col(175.0)].sum() == 2  # N's edge and the sibling's
    assert g.occ[g.row(530.0), :].sum() == 1


def test_box_free_matches_brute_force():
    rng = np.random.default_rng(7)
    sc = scene(obstacles=((60.0, 100.0, 90.0, 110.0),))
    g = C.grids(sc, Options())[2]
    rx = rng.uniform(20, 390, 3000)
    ry = rng.uniform(10, 960, 3000)
    rw, rh = 22.8, 16.9
    fast = g.boxes_free(rx, ry, rw, rh)
    slow = [_brute_free(g, float(x), float(y), rw, rh) for x, y in zip(rx, ry, strict=True)]
    assert fast.tolist() == slow
    assert 0 < fast.sum() < len(fast)


def test_leader_walk_matches_brute_force():
    g = C.grids(scene(obstacles=((60.0, 480.0, 90.0, 520.0),)), Options())[2]
    rng = np.random.default_rng(3)
    x1, y1 = rng.uniform(30, 400, 300), rng.uniform(20, 980, 300)
    fast = g.leaders_ink_cells(175.0, 500.0, x1, y1)
    slow = [_brute_walk(g, 175.0, 500.0, float(a), float(b)) for a, b in zip(x1, y1, strict=True)]
    assert fast.tolist() == slow
    assert g.leader_ink_cells(175.0, 500.0, 145.161333, 511.453991) == 3  # the oracle's first


def test_stroke_index_matches_brute_force():
    rng = np.random.default_rng(11)
    strokes = [_rand_stroke(rng) for _ in range(500)]
    idx = C.StrokeIndex(strokes)
    bx0, by0 = rng.uniform(0, 400, 4000), rng.uniform(0, 1000, 4000)
    bx1, by1 = bx0 + rng.uniform(0, 30, 4000), by0 + rng.uniform(0, 30, 4000)
    a = np.array(strokes)
    slow = [
        bool(np.any((a[:, 0] < x1) & (x0 < a[:, 2]) & (a[:, 1] < y1) & (y0 < a[:, 3])))
        for x0, y0, x1, y1 in zip(bx0, by0, bx1, by1, strict=True)
    ]
    assert idx.hits(bx0, by0, bx1, by1).tolist() == slow


def _rand_stroke(rng: np.random.Generator) -> Box:
    x, y = float(rng.uniform(0, 400)), float(rng.uniform(0, 1000))
    if rng.random() < 0.5:
        return (x, y - HW, x + float(rng.uniform(0, 80)), y + HW)
    return (x - HW, y, x + HW, y + float(rng.uniform(0, 300)))


def test_exact_ink_is_strict_overlap():
    idx = C.StrokeIndex(STROKES)
    assert idx.box_hits_ink((140.0, 495.0, 160.0, 505.0))
    assert not idx.box_hits_ink((140.0, 480.0, 160.0, 499.5))  # touching the stroke's top edge


# --------------------------------------------------------------------------- branches


def test_pinned_label_one_candidate_and_reserved():
    box = (120.0, 505.0, 137.1, 516.21)
    pin = label(key=2, end=(176.0, 500.0), pinned=box)
    free = label(key=1)
    sc = scene(labels=(pin, free))
    cp, cf = C.make_candidates(sc, Options())
    assert len(cp) == 1 and int(cp.tier[0]) == 0 and float(cp.base[0]) == 0.0
    assert cp.box(0) == box
    assert (float(cp.cx[0]), float(cp.cy[0])) == attach_pt(pin.end, FS, box)
    # the pinned label's padded box is in the obstacle grid, and the other label keeps off it
    pad = 0.3 * FS
    obs = C.obstacle_grid(sc, Options())
    g0 = C.OccGrid(sc.band, sc.mrca_fs)
    assert obs.occ[
        g0.row(box[1] - pad) : g0.row(box[3] + pad) + 1,
        g0.col(box[0] - pad) : g0.col(box[2] + pad) + 1,
    ].all()
    sel = cf.tier != 3
    w, h = 17.1 + 2 * pad, 11.21 + 2 * pad
    assert not obs.counts(cf.x0[sel] - pad, cf.y0[sel] - pad, w, h).any()
    # the oracle's best box overlaps the pinned box, so it is gone
    assert rows(cf)[0][:2] != FIRST12[0][:2]


def test_soft_tree_ink_is_a_cost_veto_is_a_veto():
    # a connector at x 135 through the oracle's best box (and not along its leader, x 145-175)
    extra = (135.0 - HW, 505.0, 135.0 + HW, 520.0)
    sc = scene(strokes=STROKES + (extra,))
    lab = sc.labels[0]
    x0, y0 = FIRST12[0][0], FIRST12[0][1]

    def find(raw: C.Raw) -> list[int]:
        a = raw.arrays()
        return np.flatnonzero((np.abs(a[0] - x0) < 1e-6) & (np.abs(a[1] - y0) < 1e-6)).tolist()

    veto = C.context(sc, Options(tree_ink="veto"))
    assert find(C.raw_candidates(veto, lab)) == []
    final = C.label_candidates(veto, lab)
    pad = 0.3 * FS
    hit = veto.ink.hits(final.x0 - pad, final.y0 - pad, final.x1 + pad, final.y1 + pad)
    assert not hit[final.tier != 3].any()

    w = 7.0
    soft = C.context(sc, Options(tree_ink="soft", soft_ink_weight=w))
    raw = C.raw_candidates(soft, lab)
    (i,) = find(raw)
    g = soft.tree
    # the connector's rows under the padded box, one column
    cells = min(g.row(520.0), g.row(y0 + 11.21 + pad)) - max(g.row(505.0), g.row(y0 - pad)) + 1
    base0 = FIRST12[0][4] - 3 * 22.0  # the oracle's base before the leader-ink charge
    assert float(raw.arrays()[4][i]) == pytest.approx(base0 + w * cells, abs=1e-6)
    assert len(raw) > len(C.raw_candidates(veto, lab))


def test_tier3_when_tiers_12_are_blocked():
    # everything left of the branch is an obstacle: tiers 1-2 find nothing, the sweep finds
    # the band's right, charged for being right of the branch
    sc = scene(obstacles=((30.0, 20.0, 172.0, 980.0),))
    lab = sc.labels[0]
    ctx = C.context(sc, Options())
    raw = C.raw_candidates(ctx, lab)
    tiers = raw.arrays()[5]
    assert len(tiers) and set(tiers.tolist()) == {3}
    assert len(tiers) <= C.T3_PICK
    c = C.label_candidates(ctx, lab)
    assert set(c.tier.tolist()) == {3}
    assert np.all(c.base >= C.T3_BASE + C.T3_RIGHT)
    assert np.all(c.x0 >= 172.0)


def test_tier3_tops_up_a_starved_label():
    # a pocket left of the branch with a few tier-1/2 spots only
    obstacles = (
        (30.0, 20.0, 400.0, 498.0),
        (30.0, 532.0, 400.0, 980.0),
        (30.0, 498.0, 130.0, 532.0),
    )
    sc = scene(obstacles=obstacles)
    raw = C.raw_candidates(C.context(sc, Options()), sc.labels[0])
    tiers = raw.arrays()[5].tolist()
    n12 = sum(t in (1, 2) for t in tiers)
    assert 0 < n12 < C.TIER3_BELOW
    assert 3 in tiers


def test_last_resort_when_everything_is_blocked():
    sc = scene(obstacles=(BAND,))
    lab = sc.labels[0]
    raw = C.raw_candidates(C.context(sc, Options()), lab)
    a = raw.arrays()
    assert len(a[0]) == 3 * 25  # every column x row: all left of the branch
    assert set(a[5].tolist()) == {3}
    assert np.all(a[4] >= C.LR_BASE)
    tw = 17.1
    assert set(np.round(a[0], 9).tolist()) == {round(30.0 + xs * tw * 0.35, 9) for xs in range(3)}


def test_last_resort_single_box_when_even_the_column_is_right():
    # leader end at the band's left edge: no column box is left of it
    lab = label(end=(31.0, 500.0))
    sc = scene(labels=(lab,), obstacles=(BAND,))
    c = C.label_candidates(C.context(sc, Options()), lab)
    assert len(c) == 1 and int(c.tier[0]) == 3
    assert c.box(0)[:2] == (30.0, 500.0 - 11.21 * 0.5)
    # base 1e5 plus the leader-ink charge (the obstacle fills the grid)
    assert float(c.base[0]) >= C.LR_BASE


def test_multi_line_label_height_in_aes_order():
    lab = label("K2N A6S L7F")
    assert C.text_height(lab) == 3 * (FS * 1.18)


# --------------------------------------------------------------------------- determinism, time


def _synthetic(n_leaves: int, n_labels: int, seed: int) -> Scene:
    """A random binary tree laid out ae-style: leaves 1.4 pt apart, one rect per edge and per
    connector, labels on random inner nodes (invented texts)."""
    rng = np.random.default_rng(seed)
    strokes: list[Box] = []
    inner: list[tuple[float, float, float]] = []  # (x, y, parent x)
    step = 950.0 / n_leaves
    hw = 0.35

    def build(lo: int, hi: int, px: float, x: float) -> float:
        if hi - lo == 1:
            y = 25.0 + lo * step
            strokes.append((px, y - hw, x, y + hw))
            return y
        mid = int(rng.integers(lo + 1, hi))
        ya = build(lo, mid, x, x + float(rng.uniform(2, 15)))
        yb = build(mid, hi, x, x + float(rng.uniform(2, 15)))
        y = (ya + yb) / 2
        strokes.append((x - hw, ya, x + hw, yb))
        strokes.append((px, y - hw, x, y + hw))
        inner.append((x, y, px))
        return y

    build(0, n_leaves, 60.0, 60.0)
    texts = ["K2N", "A6S", "L7F", "K2N A6S", "A6S L7F K2N"]
    picks = rng.choice(len(inner), size=n_labels, replace=False)
    labels = []
    for j, p in enumerate(picks):
        x, y, px = inner[int(p)]
        lines = tuple(texts[j % len(texts)].split())
        labels.append(
            Label(
                key=j,
                text="\n".join(lines),
                lines=lines,
                lw=tuple(len(t) * (0.6 * FS) for t in lines),
                fs=FS,
                node=(x, y),
                end=((px + x) * 0.5, y),
            )
        )
    return Scene(
        labels=tuple(labels),
        strokes=tuple(strokes),
        obstacles=((60.0, 20.0, 160.0, 35.0),),
        band=(30.0, 20.0, 560.0, 980.0),
        height=H,
        mrca_fs=FS,
    )


@pytest.mark.parametrize("mode", ["veto", "soft"])
def test_deterministic(mode):
    sc = _synthetic(200, 12, seed=5)
    a = C.make_candidates(sc, Options(tree_ink=mode))
    b = C.make_candidates(sc, Options(tree_ink=mode))
    for ca, cb in zip(a, b, strict=True):
        for f in ("x0", "y0", "x1", "y1", "cx", "cy", "base", "tier"):
            assert np.array_equal(getattr(ca, f), getattr(cb, f))


def test_timing_synthetic_tree():
    sc = _synthetic(680, 60, seed=1)
    assert 1900 <= len(sc.strokes) <= 2100
    for mode in ("veto", "soft"):
        t0 = time.perf_counter()
        cands = C.make_candidates(sc, Options(tree_ink=mode))
        dt = time.perf_counter() - t0
        sizes = [len(c) for c in cands]
        tiers = np.bincount(np.concatenate([c.tier for c in cands]), minlength=4)
        print(
            f"\nmake_candidates[{mode}]: {len(sc.labels)} labels, {len(sc.strokes)} strokes: "
            f"{dt:.2f} s; candidates {sum(sizes)} (min {min(sizes)}, max {max(sizes)}); "
            f"tiers {tiers.tolist()}"
        )
        assert all(s > 0 for s in sizes)
