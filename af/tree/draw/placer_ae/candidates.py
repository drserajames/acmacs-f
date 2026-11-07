"""Each label's candidate positions and their cost alone, as ae generates them (spec §2, §3.1-3.2;
draw-tree.cc M:2345-2420, M:2586-2809).

Why ae's own float order everywhere: the candidates are thinned greedily (attach points closer
than half a size, one per angle and log-length bin), so a last-bit difference in a sampled
position or a cost can move a candidate across a bin edge or reorder two in the sort, and every
later step (the search's order, its random restarts among the 40 cheapest) inherits that. Hence
the formulas below are written in ae's evaluation order, hypot and atan2 come from libm (numpy's,
not Python's own ``math.hypot``), and every sort is stable (spec §4.7 reading (a)).

Why vectorised: a label samples up to 5508 boxes in tiers 1-2 and ~10^4 in the tier-3 sweep, each
tested against a ~10^6-cell grid and thousands of stroke rects (spec §10). The grid test is O(1)
per box through a summed-area table; the exact stroke test looks only at strokes in the box's
range, found by sorting.

Two grids, not ae's one, because of the soft tree-ink option (plan ruling 1): tree strokes and
obstacles (strain labels, title, inset, pinned labels' padded boxes) must be told apart when tree
ink is a cost but obstacles stay a veto. Leaders are charged across both, as in ae.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .geometry import attach_pt
from .scene import ROW, Box, Cands, Label, Options, Scene, Weights

PI = 3.14159265358979323846  # ae's literal (== math.pi)
DEG = 180.0 / PI

# ae's structure constants: sample counts, caps and the size multiples that are not cost weights
# (those are in scene.Weights). Cited to draw-tree.cc.
CELL_FS, CELL_MIN = 0.08, 0.2  # grid cell, x mrca_fs, floor in points (M:2345)
GX_MAX, GY_MAX = 2600, 3400  # grid size caps (M:2347-2348)
STEPS_MAX = 2000  # leader walk samples (M:2362)
NTH, NL = 16, 26  # tiers 1-2: angles and lengths per side (M:2643)
LMIN_FS = 0.9  # shortest sampled leader, x fs (M:2648)
NOLEADER_FS = 0.6  # a costed leader shorter than this, x fs, pays K_noleader (M:2623)
TIER3_BELOW = 24  # tier 3 runs when fewer candidates than this (M:2683)
T3_STEP_FS, T3_STEP_MIN = 0.6, 2.0  # sweep step, x fs, floor (M:2684)
T3_DX_TW, T3_DY_TH = 0.8, 2.0  # sweep spots closer than this (x tw, x th) are one place (M:2700)
T3_PICK = 40  # sweep spots kept (M:2702)
T3_BASE = 1.0e4  # sweep surcharge (M:2715)
T3_RIGHT, T3_RIGHT_PER = 5000.0, 10.0  # sweep box right of the branch: flat, per point (M:2724)
T3_FLAT, T3_FLAT_PER = 600.0, 200.0  # sweep leader under ang_min: flat, per degree (M:2728)
LR_XS, LR_DX_TW = 3, 0.35  # last resort: columns, their spacing x tw (M:2741-2742)
LR_K, LR_DY_TH = 12, 1.3  # last resort: rows either side, spacing x th (M:2743-2744)
LR_GAP = 0.5  # last resort: attach at least this x gapL left of the branch (M:2747)
LR_BASE = 1.0e5  # last-resort surcharge (M:2750)
SEP_FS = 0.5  # thinning pass 1: attach points closer than this, x fs, are one (M:2779)
KEEP_NEAR = 110  # thinning pass 1 cap (M:2786)
ABINS, LBINS = 10, 8  # thinning pass 2: angle and log-length bins per side (M:2788)
INK_CAP = 80  # leader ink cells charged at most (M:2806)
INK_PAD = 1.0  # points of slack in the stroke range search (the exact test follows)
CHUNK = 1_000_000  # pair tests per batch in the stroke test (memory bound)


# --------------------------------------------------------------------------- occupancy grid


class OccGrid:
    """ae's occupancy grid over the band (spec §3.1): cells ``max(0.08 mrca_fs, 0.2)`` nominal,
    actual width (band / count). Marks are inclusive; ``box_free`` also vetoes any box leaving the
    band. Mark everything first: the summed-area table that makes the box queries O(1) is built at
    the first query and marking after that is refused."""

    def __init__(self, band: Box, mrca_fs: float) -> None:
        self.gx0, self.gy0, self.gx1, self.gy1 = (float(v) for v in band)
        self.cell = max(mrca_fs * CELL_FS, CELL_MIN)
        self.GX = min(max(int((self.gx1 - self.gx0) / self.cell), 1), GX_MAX)
        self.GY = min(max(int((self.gy1 - self.gy0) / self.cell), 1), GY_MAX)
        self.occ = np.zeros((self.GY, self.GX), dtype=np.uint8)
        self._sat: np.ndarray | None = None

    # ae's col/row, one value: int() truncates toward zero as static_cast<int> does
    def col(self, x: float) -> int:
        c = int(_trunc_arg((x - self.gx0) / (self.gx1 - self.gx0) * self.GX))
        return min(max(c, 0), self.GX - 1)

    def row(self, y: float) -> int:
        r = int(_trunc_arg((y - self.gy0) / (self.gy1 - self.gy0) * self.GY))
        return min(max(r, 0), self.GY - 1)

    def cols(self, x: np.ndarray) -> np.ndarray:
        return _clamp_index((x - self.gx0) / (self.gx1 - self.gx0) * self.GX, self.GX)

    def rows(self, y: np.ndarray) -> np.ndarray:
        return _clamp_index((y - self.gy0) / (self.gy1 - self.gy0) * self.GY, self.GY)

    def _writable(self) -> np.ndarray:
        if self._sat is not None:
            raise RuntimeError("OccGrid marked after it was queried")
        return self.occ

    def mark_h(self, xa: float, xb: float, y: float) -> None:
        """A horizontal stroke: one row, whatever its width (M:2352)."""
        r = self.row(y)
        self._writable()[r, self.col(min(xa, xb)) : self.col(max(xa, xb)) + 1] = 1

    def mark_v(self, x: float, ya: float, yb: float) -> None:
        """A vertical stroke: one column (M:2353)."""
        c = self.col(x)
        self._writable()[self.row(min(ya, yb)) : self.row(max(ya, yb)) + 1, c] = 1

    def mark_box(self, rx: float, ry: float, rw: float, rh: float) -> None:
        """A box given by corner and size, as ae passes it (M:2354)."""
        occ = self._writable()
        occ[self.row(ry) : self.row(ry + rh) + 1, self.col(rx) : self.col(rx + rw) + 1] = 1

    def sat(self) -> np.ndarray:
        if self._sat is None:
            s = np.zeros((self.GY + 1, self.GX + 1), dtype=np.int64)
            s[1:, 1:] = self.occ.cumsum(axis=0, dtype=np.int64).cumsum(axis=1)
            self._sat = s
        return self._sat

    def counts(
        self, rx: np.ndarray, ry: np.ndarray, rw: np.ndarray | float, rh: np.ndarray | float
    ) -> np.ndarray:
        """Cells set in rows row(ry)..row(ry+rh), cols col(rx)..col(rx+rw), inclusive."""
        s = self.sat()
        r0, r1 = self.rows(ry), self.rows(ry + rh) + 1
        c0, c1 = self.cols(rx), self.cols(rx + rw) + 1
        return s[r1, c1] - s[r0, c1] - s[r1, c0] + s[r0, c0]

    def in_band(
        self, rx: np.ndarray, ry: np.ndarray, rw: np.ndarray | float, rh: np.ndarray | float
    ) -> np.ndarray:
        """box_free's bounds test, ae's out-of-area veto."""
        return ~((rx < self.gx0) | (rx + rw > self.gx1) | (ry < self.gy0) | (ry + rh > self.gy1))

    def boxes_free(
        self, rx: np.ndarray, ry: np.ndarray, rw: np.ndarray | float, rh: np.ndarray | float
    ) -> np.ndarray:
        return self.in_band(rx, ry, rw, rh) & (self.counts(rx, ry, rw, rh) == 0)

    def box_free(self, rx: float, ry: float, rw: float, rh: float) -> bool:
        a = np.array([rx]), np.array([ry])
        return bool(self.boxes_free(a[0], a[1], rw, rh)[0])

    def leaders_ink_cells(self, x0: float, y0: float, x1: np.ndarray, y1: np.ndarray) -> np.ndarray:
        """ae's leader walk (M:2361-2372) for many leaders from one start: steps+1 samples, ends
        included, skipping a cell only when it repeats the previous sample's."""
        x1, y1 = np.asarray(x1, dtype=np.float64), np.asarray(y1, dtype=np.float64)
        n = len(x1)
        if n == 0:
            return np.zeros(0, dtype=np.int64)
        dx, dy = x1 - x0, y1 - y0
        lin = np.hypot(dx, dy) / self.cell
        steps = np.clip(_trunc(np.minimum(lin, STEPS_MAX)) + 1, 1, STEPS_MAX)
        size = steps + 1
        starts = np.cumsum(size) - size
        owner = np.repeat(np.arange(n), size)
        k = np.arange(int(size.sum())) - np.repeat(starts, size)
        t = k / np.repeat(steps, size).astype(np.float64)
        r = self.rows(y0 + dy[owner] * t)
        c = self.cols(x0 + dx[owner] * t)
        new = np.ones(len(k), dtype=bool)
        new[1:] = (r[1:] != r[:-1]) | (c[1:] != c[:-1])
        new[starts] = True
        hit = (self.occ[r, c] != 0) & new
        return np.bincount(owner[hit], minlength=n).astype(np.int64)

    def leader_ink_cells(self, x0: float, y0: float, x1: float, y1: float) -> int:
        return int(self.leaders_ink_cells(x0, y0, np.array([x1]), np.array([y1]))[0])

    def union(self, other: OccGrid) -> OccGrid:
        """A grid with the cells of both (same band and cell): what ae's single grid holds."""
        if (self.GX, self.GY) != (other.GX, other.GY):
            raise ValueError("grids of different shapes")
        g = OccGrid.__new__(OccGrid)
        g.__dict__.update(self.__dict__)
        g.occ = self.occ | other.occ
        g._sat = None
        return g


def _trunc_arg(v: float) -> float:
    """Keep a huge or infinite coordinate finite before truncation (ae's cast is then UB)."""
    return min(max(v, -1.0), 1e9)


def _trunc(v: np.ndarray) -> np.ndarray:
    return v.astype(np.int64)  # toward zero, as static_cast<int>


def _clamp_index(v: np.ndarray, n: int) -> np.ndarray:
    v = np.clip(np.asarray(v, dtype=np.float64), -1.0, float(n))
    return np.clip(_trunc(v), 0, n - 1)


def stroke_is_horizontal(r: Box) -> bool:
    """A stroke rect is a branch's horizontal edge when at least as wide as tall, else a
    connector: the Scene carries rects, not ae's two kinds (see grids())."""
    return (r[2] - r[0]) >= (r[3] - r[1])


def tree_grid(scene: Scene) -> OccGrid:
    """The strokes rasterised as ae does (M:2392-2413): a horizontal edge with mark_h at its
    centre row, a connector with mark_v at its centre column."""
    g = OccGrid(scene.band, scene.mrca_fs)
    for r in scene.strokes:
        if stroke_is_horizontal(r):
            g.mark_h(r[0], r[2], (r[1] + r[3]) * 0.5)
        else:
            g.mark_v((r[0] + r[2]) * 0.5, r[1], r[3])
    return g


def obstacle_grid(scene: Scene, opts: Options) -> OccGrid:
    """Strain labels, title, inset (each as ae's mark_box of corner and size), then every pinned
    label's padded box (M:2580-2585), all before any candidate."""
    g = OccGrid(scene.band, scene.mrca_fs)
    for b in scene.obstacles:
        g.mark_box(b[0], b[1], b[2] - b[0], b[3] - b[1])
    pad = scene.mrca_fs * opts.weights.pad
    for lab in scene.labels:
        if lab.pinned is not None:
            tw, th = lab.tw, text_height(lab)
            g.mark_box(lab.pinned[0] - pad, lab.pinned[1] - pad, tw + 2.0 * pad, th + 2.0 * pad)
    return g


# --------------------------------------------------------------------------- exact tree ink


class StrokeIndex:
    """ae's exact ink test (M:2415-2419): a box hits ink when it overlaps a stroke rect strictly.
    Horizontal edges are sorted by top, connectors by left, so a box is tested only against the
    strokes in its own band of y (or x)."""

    def __init__(self, strokes: Sequence[Box]) -> None:
        a = np.array(strokes, dtype=np.float64).reshape(-1, 4)
        wide = (a[:, 2] - a[:, 0]) >= (a[:, 3] - a[:, 1])
        h, v = a[wide], a[~wide]
        self.h = h[np.argsort(h[:, 1], kind="stable")]
        self.v = v[np.argsort(v[:, 0], kind="stable")]
        self.h_ext = float((self.h[:, 3] - self.h[:, 1]).max()) if len(self.h) else 0.0
        self.v_ext = float((self.v[:, 2] - self.v[:, 0]).max()) if len(self.v) else 0.0

    def hits(
        self, bx0: np.ndarray, by0: np.ndarray, bx1: np.ndarray, by1: np.ndarray
    ) -> np.ndarray:
        box = [np.asarray(v, dtype=np.float64) for v in (bx0, by0, bx1, by1)]
        out = np.zeros(len(box[0]), dtype=bool)
        if len(self.h):
            out |= _hits_sorted(self.h, 1, self.h_ext, box)
        if len(self.v):
            out |= _hits_sorted(self.v, 0, self.v_ext, box)
        return out

    def box_hits_ink(self, box: Box) -> bool:
        return bool(self.hits(*(np.array([v]) for v in box))[0])


def _hits_sorted(rects: np.ndarray, axis: int, ext: float, box: list[np.ndarray]) -> np.ndarray:
    """Strict overlap against rects sorted by their low edge on ``axis`` (1: y, 0: x), each at
    most ``ext`` long on it: only rects starting in (box low - ext, box high) can overlap."""
    keys = rects[:, axis]
    lo_edge, hi_edge = (box[0], box[2]) if axis == 0 else (box[1], box[3])
    lo = np.searchsorted(keys, lo_edge - ext - INK_PAD, side="left")
    hi = np.searchsorted(keys, hi_edge, side="left")
    cnt = np.maximum(hi - lo, 0)
    out = np.zeros(len(cnt), dtype=bool)
    todo = np.flatnonzero(cnt)
    csum = np.cumsum(cnt[todo])
    start = 0
    while start < len(todo):
        base = csum[start - 1] if start else 0
        stop = int(np.searchsorted(csum, base + CHUNK, side="right"))
        stop = max(stop, start + 1)
        idx = todo[start:stop]
        c = cnt[idx]
        owner = np.repeat(idx, c)
        first = np.cumsum(c) - c
        ridx = np.repeat(lo[idx], c) + (np.arange(int(c.sum())) - np.repeat(first, c))
        r = rects[ridx]
        hit = (
            (r[:, 0] < box[2][owner])
            & (box[0][owner] < r[:, 2])
            & (r[:, 1] < box[3][owner])
            & (box[1][owner] < r[:, 3])
        )
        out[np.unique(owner[hit])] = True
        start = stop
    return out


# --------------------------------------------------------------------------- shape cost


@dataclass(frozen=True)
class Consts:
    """ae's candidate constants for one scene, in points and radians, computed as ae computes
    them (M:2519-2575) so the products round alike."""

    gap_l: float
    pad: float
    len_soft: float
    len_max: float
    len_max2: float
    ang_short: float
    ang_long: float
    ang_l1: float
    ang_l2: float
    ang_min: float
    ang_max: float
    ang_min2: float
    ang_max2: float


def consts(w: Weights, height: float, mrca_fs: float) -> Consts:
    fs, h = mrca_fs, height
    return Consts(
        gap_l=fs * w.gap_l,
        pad=fs * w.pad,
        len_soft=w.len_soft * h,
        len_max=w.len_max * h,
        len_max2=w.len_max2 * h,
        ang_short=w.ang_short * PI / 180.0,
        ang_long=w.ang_long * PI / 180.0,
        ang_l1=w.ang_l1 * h,
        ang_l2=w.ang_l2 * h,
        ang_min=w.ang_min * PI / 180.0,
        ang_max=w.ang_max * PI / 180.0,
        ang_min2=w.ang_min2 * PI / 180.0,
        ang_max2=w.ang_max2 * PI / 180.0,
    )


def ang_target(length: np.ndarray | float, k: Consts) -> np.ndarray:
    """The angle a leader of this length aims at (M:2550-2553): ang_short up to ang_l1, ang_long
    from ang_l2, linear between."""
    t = np.clip((np.asarray(length) - k.ang_l1) / max(k.ang_l2 - k.ang_l1, 1e-9), 0.0, 1.0)
    return k.ang_short + (k.ang_long - k.ang_short) * t


def shape_cost(
    length: np.ndarray | float,
    theta: np.ndarray | float,
    fs: float,
    down: np.ndarray | bool,
    weights: Weights,
    height: float,
) -> np.ndarray:
    """emit_at's cost of a leader alone (M:2620-2624), tier surcharge excluded: linear in its
    length, quadratic past len_soft in units of the label's own size, the angle off the target
    for that length, a leader too short to draw, a sample taken above the branch. ``theta`` in
    radians; ``down`` is the SAMPLING direction (spec §2.1), not where the box ended up."""
    w = weights
    k = consts(w, height, 0.0)  # the mrca_fs-scaled ones (gap_l, pad) are not used here
    la = np.asarray(length, dtype=np.float64)
    base = la * w.wlen
    o = (la - k.len_soft) / fs
    base = base + np.where(la > k.len_soft, o * o * w.wquad, 0.0)
    base = base + np.abs(np.asarray(theta) - ang_target(la, k)) * DEG * w.wang
    base = base + np.where(la < fs * NOLEADER_FS, w.noleader, 0.0)
    return base + np.where(np.asarray(down), 0.0, w.above)


# --------------------------------------------------------------------------- generation


def text_height(label: Label) -> float:
    """``nlines * (1.18 fs)``, in ae's order (M:2587): Label.th multiplies the other way, which
    differs in the last bit from three lines up."""
    return len(label.lines) * (label.fs * ROW)


@dataclass
class Context:
    """What every label's generation shares: the constants, the grids, the stroke index."""

    scene: Scene
    opts: Options
    k: Consts
    veto: OccGrid  # what vetoes a box: everything (ae), or obstacles only (soft tree ink)
    tree: OccGrid  # tree strokes alone: the soft cost's count
    both: OccGrid  # everything: the leader walk
    ink: StrokeIndex


def grids(scene: Scene, opts: Options) -> tuple[OccGrid, OccGrid, OccGrid]:
    """(tree, obstacles, both)."""
    tree, obs = tree_grid(scene), obstacle_grid(scene, opts)
    return tree, obs, tree.union(obs)


def context(scene: Scene, opts: Options) -> Context:
    tree, obs, both = grids(scene, opts)
    return Context(
        scene=scene,
        opts=opts,
        k=consts(opts.weights, scene.height, scene.mrca_fs),
        veto=both if opts.tree_ink == "veto" else obs,
        tree=tree,
        both=both,
        ink=StrokeIndex(scene.strokes),
    )


def make_candidates(scene: Scene, opts: Options) -> list[Cands]:
    """Every label's candidates in the search's order (spec §2.6): pinned labels one, the rest
    tiers 1-2, tier 3 if starved, the last resort if still empty, thinned and charged for the
    ink their leaders cross."""
    ctx = context(scene, opts)
    return [label_candidates(ctx, lab) for lab in scene.labels]


@dataclass
class Raw:
    """Accepted candidates before thinning, in emission order."""

    x0: list[np.ndarray]
    y0: list[np.ndarray]
    cx: list[np.ndarray]
    cy: list[np.ndarray]
    base: list[np.ndarray]
    tier: list[np.ndarray]

    def add(self, *cols: np.ndarray) -> None:
        for dst, src in zip(
            (self.x0, self.y0, self.cx, self.cy, self.base, self.tier), cols, strict=True
        ):
            dst.append(np.asarray(src))

    def __len__(self) -> int:
        return sum(len(b) for b in self.base)

    def arrays(self) -> list[np.ndarray]:
        out = []
        for col, dt in zip(
            (self.x0, self.y0, self.cx, self.cy, self.base, self.tier),
            (np.float64,) * 5 + (np.int8,),
            strict=True,
        ):
            out.append(np.concatenate(col).astype(dt) if col else np.zeros(0, dtype=dt))
        return out


def label_candidates(ctx: Context, lab: Label, stats: dict[str, int] | None = None) -> Cands:
    """One label's candidates; ``stats`` (if given) gets the counts at each stage: raw (accepted
    before thinning), near (pass 1), final."""
    tw, th = lab.tw, text_height(lab)
    if lab.pinned is not None:
        return _pinned(lab)
    return _thin(ctx, lab, tw, th, raw_candidates(ctx, lab), stats)


def raw_candidates(ctx: Context, lab: Label) -> Raw:
    """An unpinned label's accepted candidates before thinning, in ae's emission order: tiers 1-2,
    tier 3 if fewer than TIER3_BELOW, the last resort if none."""
    tw, th = lab.tw, text_height(lab)
    raw = Raw([], [], [], [], [], [])
    _tiers12(ctx, lab, tw, th, raw)
    if len(raw) < TIER3_BELOW:
        _tier3(ctx, lab, tw, th, raw)
    if len(raw) == 0:
        _last_resort(ctx, lab, tw, th, raw)
    return raw


def _pinned(lab: Label) -> Cands:
    """One fixed candidate (M:2588-2596): no checks, no cost, no thinning."""
    assert lab.pinned is not None
    box = lab.pinned
    cx, cy = attach_pt(lab.end, lab.fs, box)
    return Cands.of([(box[0], box[1], box[2], box[3], cx, cy, 0.0, 0)])


def _box_test(
    ctx: Context, x0: np.ndarray, y0: np.ndarray, tw: float, th: float
) -> tuple[np.ndarray, np.ndarray]:
    """(accepted, extra cost) for boxes with these top-lefts, padded as ae pads them
    (M:2612-2613). Veto: the padded box clear of every grid cell and of the exact strokes.
    Soft: clear of the obstacle grid only, paying per tree cell under it."""
    pad = ctx.k.pad
    rx, ry, rw, rh = x0 - pad, y0 - pad, tw + 2.0 * pad, th + 2.0 * pad
    ok = ctx.veto.boxes_free(rx, ry, rw, rh)
    extra = np.zeros(len(x0))
    if ctx.opts.tree_ink == "veto":
        idx = np.flatnonzero(ok)
        if len(idx):
            x, y = x0[idx], y0[idx]
            ok[idx] = ~ctx.ink.hits(x - pad, y - pad, x + tw + pad, y + th + pad)
    else:
        idx = np.flatnonzero(ok)
        extra[idx] = ctx.opts.soft_ink_weight * ctx.tree.counts(rx[idx], ry[idx], rw, rh)
    return ok, extra


def _samples(
    ctx: Context, lab: Label, tw: float, th: float, tier: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Tier 1 or 2's emit calls in ae's order (s, theta, length, then centred / below / above):
    the sampled right-edge point (cx, cy), the box top y0, and down."""
    k, fs = ctx.k, lab.fs
    ax, ay = lab.end
    a0, a1, lmax = (
        (k.ang_min, k.ang_max, k.len_max) if tier == 1 else (k.ang_min2, k.ang_max2, k.len_max2)
    )
    ks = np.arange(NL + 1)
    cxs, cys, downs = [], [], []
    for s in range(2):
        sgn = 1.0 if s == 0 else -1.0
        for t in range(NTH + 1):
            theta = a0 + (a1 - a0) * t / NTH
            lmin = max(k.gap_l / math.cos(theta), fs * LMIN_FS)
            if lmin > lmax:
                continue
            length = lmin + (lmax - lmin) * ks / NL
            cxs.append(ax - length * math.cos(theta))
            cys.append(ay + sgn * length * math.sin(theta))
            downs.append(np.full(NL + 1, s == 0))
    if not cxs:
        e = np.zeros(0)
        return e, e, e, np.zeros(0, dtype=bool)
    cx, cy, down = (np.repeat(np.concatenate(v), 3) for v in (cxs, cys, downs))
    variant = np.tile(np.arange(3), len(cx) // 3)
    y0 = np.where(variant == 0, cy - th * 0.5, np.where(variant == 1, cy, cy - th))
    return cx, cy, y0, down.astype(bool)


def _tiers12(ctx: Context, lab: Label, tw: float, th: float, raw: Raw) -> None:
    """emit_at over tiers 1 and 2 (M:2609-2671), all boxes at once."""
    ax, ay = lab.end
    w = ctx.opts.weights
    for tier, extra in ((1, 0.0), (2, w.tier2)):
        cx, cy, y0, down = _samples(ctx, lab, tw, th, tier)
        keep = ~(cx > ax - ctx.k.gap_l)
        cx, y0, down = cx[keep], y0[keep], down[keep]
        x0 = cx - tw
        ok, soft = _box_test(ctx, x0, y0, tw, th)
        x0, y0, down, soft = x0[ok], y0[ok], down[ok], soft[ok]
        x1, y1 = x0 + tw, y0 + th
        tx, ty = _attach(ax, ay, lab.fs, x0, y0, x1, y1)
        la = np.hypot(ax - tx, ay - ty)
        tha = np.arctan2(np.abs(ay - ty), np.maximum(np.abs(ax - tx), 1e-9))
        base = shape_cost(la, tha, lab.fs, down, w, ctx.scene.height) + extra + soft
        t = np.full(len(base), 2 if extra > 0.0 else 1, dtype=np.int8)
        raw.add(x0, y0, tx, ty, base, t)


def _attach(
    ax: float, ay: float, fs: float, x0: np.ndarray, y0: np.ndarray, x1: np.ndarray, y1: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """geometry.attach_pt over arrays."""
    left = ax > x1
    ink = fs * 0.23
    cx = np.where(left, x1, (x0 + x1) * 0.5)
    cy = np.where(left, (y0 + y1) * 0.5, np.where(y1 < ay, y1 - ink, y0 + ink))
    return cx, cy


def _accumulate(start: float, size: float, stop: float, step: float) -> np.ndarray:
    """ae's ``for (v = start; v + size <= stop; v += step)``, accumulated as it does."""
    out = []
    v = start
    while v + size <= stop:
        out.append(v)
        v += step
    return np.array(out, dtype=np.float64)


def _tier3(ctx: Context, lab: Label, tw: float, th: float, raw: Raw) -> None:
    """The band sweep for a starved label (M:2683-2731): free boxes nearest the leader end
    first, spread wide, charged the tiers' shape cost by geometry."""
    sc, fs = ctx.scene, lab.fs
    gx0, gy0, gx1, gy1 = sc.band
    ax, ay = lab.end
    step = max(fs * T3_STEP_FS, T3_STEP_MIN)
    xs, ys = _accumulate(gx0, tw, gx1, step), _accumulate(gy0, th, gy1, step)
    if not len(xs) or not len(ys):
        return
    x0, y0 = np.tile(xs, len(ys)), np.repeat(ys, len(xs))
    ok, soft = _box_test(ctx, x0, y0, tw, th)
    x0, y0, soft = x0[ok], y0[ok], soft[ok]
    d = np.hypot(ax - (x0 + tw * 0.5), ay - (y0 + th * 0.5))
    order = np.argsort(d, kind="stable")
    rows = [
        _tier3_cand(ctx, lab, tw, th, float(x0[i]), float(y0[i]), float(d[i]), float(soft[i]))
        for i in _spread(x0[order], y0[order], tw * T3_DX_TW, th * T3_DY_TH, order)
    ]
    if rows:
        raw.add(*(np.array(c) for c in zip(*rows, strict=True)))


def _spread(x: np.ndarray, y: np.ndarray, dx: float, dy: float, order: np.ndarray) -> list[int]:
    """ae's greedy pick (M:2696-2703) in sorted order: a spot is taken unless an earlier pick is
    within dx and dy of it. Returns the picks' original indices."""
    alive = np.ones(len(x), dtype=bool)
    picks: list[int] = []
    i = 0
    while len(picks) < T3_PICK:
        nxt = np.flatnonzero(alive[i:])
        if not len(nxt):
            break
        i += int(nxt[0])
        picks.append(int(order[i]))
        alive &= ~((np.abs(x - x[i]) < dx) & (np.abs(y - y[i]) < dy))
        i += 1
    return picks


def _tier3_cand(
    ctx: Context, lab: Label, tw: float, th: float, x0: float, y0: float, d: float, soft: float
) -> tuple[float, float, float, float, float, int]:
    k, w, fs = ctx.k, ctx.opts.weights, lab.fs
    ax, ay = lab.end
    cx, cy = attach_pt(lab.end, fs, (x0, y0, x0 + tw, y0 + th))
    theta = math.atan2(abs(ay - cy), max(abs(ax - cx), 1e-9))
    base = T3_BASE + d * w.t3d + abs(theta - float(ang_target(d, k))) * DEG * w.wang
    if d > k.len_soft:
        o = (d - k.len_soft) / fs
        base += o * o * w.wquad
    if cy < ay:
        base += w.above
    if x0 + tw > ax:
        base += T3_RIGHT + (x0 + tw - ax) * T3_RIGHT_PER
    if theta < k.ang_min:
        base += T3_FLAT + (k.ang_min - theta) * DEG * T3_FLAT_PER
    return (x0, y0, cx, cy, base + soft, 3)


def _last_resort(ctx: Context, lab: Label, tw: float, th: float, raw: Raw) -> None:
    """A column of boxes down the band's left edge, ignoring ink and the grid (M:2740-2759)."""
    k, w, fs = ctx.k, ctx.opts.weights, lab.fs
    gx0, gy0, _, gy1 = ctx.scene.band
    ax, ay = lab.end
    rows: list[tuple[float, float, float, float, float, int]] = []
    for xs in range(LR_XS):
        x0 = gx0 + xs * tw * LR_DX_TW
        for j in range(-LR_K, LR_K + 1):
            y0 = min(max(ay - th * 0.5 + j * th * LR_DY_TH, gy0), gy1 - th)
            cx, cy = attach_pt(lab.end, fs, (x0, y0, x0 + tw, y0 + th))
            if cx > ax - k.gap_l * LR_GAP:
                continue
            theta = math.atan2(abs(ay - cy), max(ax - cx, 1e-9))
            lr = float(np.hypot(ax - cx, ay - cy))
            base = LR_BASE + lr * w.wlen + abs(theta - float(ang_target(lr, k))) * DEG * w.wang
            if cy < ay:
                base += w.above
            rows.append((x0, y0, cx, cy, base, 3))
    if not rows:
        x0, y0 = gx0, min(max(ay - th * 0.5, gy0), gy1 - th)
        cx, cy = attach_pt(lab.end, fs, (x0, y0, x0 + tw, y0 + th))
        rows.append((x0, y0, cx, cy, LR_BASE, 3))
    raw.add(*(np.array(c) for c in zip(*rows, strict=True)))


# --------------------------------------------------------------------------- thinning


def _thin(
    ctx: Context, lab: Label, tw: float, th: float, raw: Raw, stats: dict[str, int] | None
) -> Cands:
    """Sort by base, keep a near field and one per bin, charge the leaders' ink, sort again
    (M:2760-2809)."""
    x0, y0, cx, cy, base, tier = raw.arrays()
    order = np.argsort(base, kind="stable")
    x0, y0, cx, cy, base, tier = (a[order] for a in (x0, y0, cx, cy, base, tier))
    keep = _near_field(cx, cy, lab.fs * SEP_FS)
    n_near = len(keep)
    keep = _far_field(ctx, lab, cx, cy, keep)
    if stats is not None:
        stats.update(raw=len(base), near=n_near, final=len(keep))
    sel = np.array(keep, dtype=np.int64)
    ax, ay = lab.end
    cells = ctx.both.leaders_ink_cells(ax, ay, cx[sel], cy[sel])
    b = base[sel] + np.minimum(cells, INK_CAP) * ctx.opts.weights.wink
    fin = sel[np.argsort(b, kind="stable")]
    b = np.sort(b, kind="stable")
    return Cands(
        x0[fin].copy(),
        y0[fin].copy(),
        x0[fin] + tw,
        y0[fin] + th,
        cx[fin].copy(),
        cy[fin].copy(),
        b,
        tier[fin].copy(),
    )


def _near_field(cx: np.ndarray, cy: np.ndarray, sep: float) -> list[int]:
    """Pass 1 (M:2779-2787): in base order, keep an attach point unless one kept is closer than
    sep; stop at KEEP_NEAR. Kept points are hashed into sep-sized bins, so only the 3 x 3 bins
    around a point are compared, and the distance is libm's hypot as ae's, consulted only when
    the squared distance is near sep squared."""
    xs, ys = cx.tolist(), cy.tolist()
    bins: dict[tuple[int, int], list[int]] = {}
    keep: list[int] = []
    s2, tol = sep * sep, 1e-9 * sep * sep
    for i, (x, y) in enumerate(zip(xs, ys, strict=True)):
        bx, by = math.floor(x / sep), math.floor(y / sep)
        if not _near_kept(xs, ys, bins, bx, by, x, y, sep, s2, tol):
            keep.append(i)
            bins.setdefault((bx, by), []).append(i)
        if len(keep) >= KEEP_NEAR:
            break
    return keep


def _near_kept(
    xs: list[float],
    ys: list[float],
    bins: dict[tuple[int, int], list[int]],
    bx: int,
    by: int,
    x: float,
    y: float,
    sep: float,
    s2: float,
    tol: float,
) -> bool:
    for gx in (bx - 1, bx, bx + 1):
        for gy in (by - 1, by, by + 1):
            for j in bins.get((gx, gy), ()):
                dx, dy = x - xs[j], y - ys[j]
                q = dx * dx + dy * dy
                if q < s2 - tol or (q <= s2 + tol and float(np.hypot(dx, dy)) < sep):
                    return True
    return False


def _far_field(
    ctx: Context, lab: Label, cx: np.ndarray, cy: np.ndarray, keep: list[int]
) -> list[int]:
    """Pass 2 (M:2788-2804): over the whole sorted list, the first candidate in each (side,
    angle, log-length) cell not already held."""
    fs = lab.fs
    ax, ay = lab.end
    lo = math.log(max(fs * LMIN_FS, 1.0))
    hi = math.log(max(ctx.k.len_max2, fs * 2.0))
    ddx, ddy = ax - cx, ay - cy
    th_ = np.arctan2(np.abs(ddy), np.maximum(np.abs(ddx), 1e-9))
    ab = np.clip(_trunc(th_ / (PI * 0.5) * ABINS), 0, ABINS - 1)
    ln = np.log(np.maximum(np.hypot(ddx, ddy), 1.0))
    lb = np.clip(_trunc(np.clip((ln - lo) / (hi - lo) * LBINS, -1.0, LBINS)), 0, LBINS - 1)
    side = np.where(cy > ay, 0, 1)
    cell = ((side * ABINS + ab) * LBINS + lb).tolist()
    have = {cell[i] for i in keep}
    out = list(keep)
    for i, c in enumerate(cell):
        if c not in have:
            have.add(c)
            out.append(i)
    return out
