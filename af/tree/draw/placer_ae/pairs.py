"""The pairwise conflict and penetration matrices: predicates.pconf and psoft for every candidate
pair of every two labels that can interact, built once before the search.

Why matrices: the search evaluates a label's cost against its neighbours' current choices
millions of times (spec §10), and both terms depend only on the two candidates, so they are
tabulated here and the search only indexes them. They must equal predicates.py element for
element, because measure.py reports with predicates.py: what the search minimises is what is
reported. So every expression below repeats predicates.py's operations in its order (float
results that sit on a threshold must round the same way), and the one libm call whose rounding
numpy may not share, ``hypot`` in a comparison, is redone with ``math.hypot`` near the threshold.

Pairs whose envelopes (all boxes, glyph lines and leaders, grown by the widest reach of any
term) do not meet contribute exactly zero and are not stored; the same bound prunes the rows and
columns inside a stored pair.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TypeVar

import numpy as np
from numpy.typing import NDArray

from .predicates import Consts, consts
from .scene import CAP, ROW, Cands, Options, Scene

Arr = NDArray[np.float64]
Mask = NDArray[np.bool_]
V = Arr | np.float64  # a coordinate: per candidate, or a leader end shared by all
Quad = tuple[Arr, Arr, Arr, Arr]  # x0, y0, x1, y1, broadcastable
Env = tuple[float, float, float, float]
_A = TypeVar("_A", bound=np.ndarray)

_NEAR = 1e-14  # relative: np.hypot and math.hypot agree to a few ulps, so recheck within this


@dataclass(frozen=True)
class _Side:
    """One label's candidates as arrays, with what every pair needs precomputed once."""

    ex: float  # the leader end (label.end)
    ey: float
    box: Quad  # (n,)
    gbox: Quad  # the box grown by mt: the leader-over-text test
    cx: Arr  # the attach points
    cy: Arr
    llen: Arr  # each leader's length, math.hypot as seg_box_len's
    rects: list[Quad]  # per glyph line, (n,) each
    grects: list[Quad]  # the same grown by mt
    cenv: Quad  # each candidate's envelope: box, lines, leader
    env: Env | None  # the label's: union of cenv (None with no candidates)


def _side(scene: Scene, i: int, c: Cands, k: Consts) -> _Side:
    label = scene.labels[i]
    ex, ey = float(label.end[0]), float(label.end[1])
    box = (c.x0, c.y0, c.x1, c.y1)
    rects = []
    for n, w in enumerate(label.lw):
        # glyph_rects: y = y0 + (k + 0.5) * ROW * fs - CAP / 2 * fs, in that order
        y = c.y0 + (n + 0.5) * ROW * label.fs - CAP / 2 * label.fs
        rects.append((c.x0, y, c.x0 + w, y + CAP * label.fs))
    llen = np.array(
        [math.hypot(float(x) - ex, float(y) - ey) for x, y in zip(c.cx, c.cy, strict=True)],
        dtype=np.float64,
    )
    cenv = _cand_envelopes(ex, ey, box, rects, c.cx, c.cy)
    env = None
    if len(c):
        env = (
            float(cenv[0].min()),
            float(cenv[1].min()),
            float(cenv[2].max()),
            float(cenv[3].max()),
        )
    return _Side(
        ex, ey, box, _grow(box, k.mt), c.cx, c.cy, llen, rects, [_grow(r, k.mt) for r in rects],
        cenv, env,
    )  # fmt: skip


def _grow(b: Quad, by: float) -> Quad:
    """geometry.grow on arrays (the same subtraction and addition)."""
    return (b[0] - by, b[1] - by, b[2] + by, b[3] + by)


def _cand_envelopes(ex: float, ey: float, box: Quad, rects: list[Quad], cx: Arr, cy: Arr) -> Quad:
    """Each candidate's extent: its box, its glyph lines (which a hand-made box need not hold)
    and its leader from the end to the attach point (leaders cross far from either box)."""
    x0 = np.minimum(np.minimum(box[0], cx), ex)
    y0 = np.minimum(np.minimum(box[1], cy), ey)
    x1 = np.maximum(np.maximum(box[2], cx), ex)
    y1 = np.maximum(np.maximum(box[3], cy), ey)
    for r in rects:
        x0, y0 = np.minimum(x0, r[0]), np.minimum(y0, r[1])
        x1, y1 = np.maximum(x1, r[2]), np.maximum(y1, r[3])
    return (x0, y0, x1, y1)


def reach(k: Consts) -> float:
    """How far apart two labels' ink and leaders can be and still score: the widest threshold
    of any term, with slack for the rounding in comparisons like ``a.x0 - m < b.x1``."""
    r = max(k.m, k.mt, k.vsep, k.leadthr, k.side) + abs(k.tolerance)
    return r * (1 + 1e-9) + 1e-9


def _meets(a: Quad | Env, b: Quad | Env, r: float) -> Mask:
    """Envelopes within r of each other (inclusive): a necessary condition for any term."""
    return np.asarray(
        (a[0] - r <= b[2]) & (b[0] - r <= a[2]) & (a[1] - r <= b[3]) & (b[1] - r <= a[3])
    )


# --- geometry.py on arrays --------------------------------------------------------------------


def _orient(px: V, py: V, qx: V, qy: V, rx: V, ry: V) -> Arr:
    v = (qy - py) * (rx - qx) - (qx - px) * (ry - qy)
    return np.sign(v)


def _cross(ax: V, ay: V, bx: V, by: V, cx: V, cy: V, dx: V, dy: V) -> Mask:
    """segs_cross: touching counts (an orientation of 0 against a non-zero)."""
    first = _orient(ax, ay, bx, by, cx, cy) != _orient(ax, ay, bx, by, dx, dy)
    return first & (_orient(cx, cy, dx, dy, ax, ay) != _orient(cx, cy, dx, dy, bx, by))


def _inside(px: V, py: V, b: Quad) -> Mask:
    return (b[0] <= px) & (px <= b[2]) & (b[1] <= py) & (py <= b[3])


def _seg_box(sx: V, sy: V, ex: V, ey: V, b: Quad) -> Mask:
    """seg_box: an end inside (inclusive) or the segment crossing one of the four edges."""
    hit = _inside(sx, sy, b) | _inside(ex, ey, b)
    corners = ((b[0], b[1]), (b[2], b[1]), (b[2], b[3]), (b[0], b[3]))
    for n in range(4):
        (c0x, c0y), (c1x, c1y) = corners[n], corners[(n + 1) % 4]
        hit = hit | _cross(sx, sy, ex, ey, c0x, c0y, c1x, c1y)
    return hit


def _seg_box_len(sx: float, sy: float, ex: Arr, ey: Arr, length: Arr, b: Quad) -> Arr:
    """seg_box_len (Liang-Barsky). ae returns as soon as t0 > t1; t0 only grows and t1 only
    shrinks, so testing once at the end is the same."""
    dx, dy = ex - sx, ey - sy
    t0: Arr = np.zeros(np.broadcast_shapes(np.shape(ex), np.shape(b[0])))
    t1: Arr = np.ones_like(t0)
    dead = np.zeros(t0.shape, dtype=bool)
    with np.errstate(divide="ignore", invalid="ignore"):
        for p, q in ((-dx, sx - b[0]), (dx, b[2] - sx), (-dy, sy - b[1]), (dy, b[3] - sy)):
            zero = p == 0
            dead |= zero & (q < 0)
            t = q / p
            t0 = np.where(~zero & (p < 0), np.maximum(t0, t), t0)
            t1 = np.where(~zero & (p > 0), np.minimum(t1, t), t1)
    return np.where(dead | (t0 > t1), 0.0, (t1 - t0) * length)


def _hypot_below(x: Arr, y: Arr, thr: float) -> tuple[Arr, Mask]:
    """hypot(x, y) and hypot < thr, with math.hypot's own rounding wherever it is near thr."""
    x, y = np.broadcast_arrays(x, y)
    h = np.hypot(x, y)
    near = np.nonzero(np.abs(h - thr) <= _NEAR * thr)
    if len(near[0]):
        h = h.copy()
        h[near] = [
            math.hypot(a, b) for a, b in zip(x[near].tolist(), y[near].tolist(), strict=True)
        ]
    return h, h < thr


def _pt_seg_d(px: V, py: V, ax: V, ay: V, bx: V, by: V, thr: float) -> Arr:
    """pt_seg_d; exact (math.hypot) near ``thr``, the threshold its caller compares with."""
    dx, dy = bx - ax, by - ay
    l2 = dx * dx + dy * dy
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.maximum(0.0, np.minimum(1.0, ((px - ax) * dx + (py - ay) * dy) / l2))
    t = np.where(l2 == 0, 0.0, t)
    return _hypot_below(px - (ax + t * dx), py - (ay + t * dy), thr)[0]


def _boxes_near(a: Quad, b: Quad, mx: float, my: float) -> Mask:
    return (a[0] - mx < b[2]) & (b[0] - mx < a[2]) & (a[1] - my < b[3]) & (b[1] - my < a[3])


def _rects_clash(a: Quad, b: Quad, m: float, vsep: float, tolerance: float) -> Mask:
    gx = np.maximum(b[0] - a[2], a[0] - b[2])
    gy = np.maximum(b[1] - a[3], a[1] - b[3])
    ox, oy = gx < 0, gy < 0
    diag = _hypot_below(gx, gy, m)[1]
    return (
        (ox & oy)
        | (ox & ~oy & (gy < vsep - tolerance))
        | (~ox & oy & (gx < m))
        | (~ox & ~oy & diag)
    )


# --- one pair ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class _View:
    """A side restricted to some candidates and shaped to broadcast: rows (n, 1) for the first
    label of a pair, columns (1, n) for the second."""

    ex: float
    ey: float
    box: Quad
    gbox: Quad
    cx: Arr
    cy: Arr
    llen: Arr
    rects: list[Quad]
    grects: list[Quad]


def _view(s: _Side, idx: NDArray[np.intp], rows: bool) -> _View:
    def sh(a: Arr) -> Arr:
        v = a[idx]
        return v[:, None] if rows else v[None, :]

    def q(b: Quad) -> Quad:
        return (sh(b[0]), sh(b[1]), sh(b[2]), sh(b[3]))

    return _View(
        s.ex, s.ey, q(s.box), q(s.gbox), sh(s.cx), sh(s.cy), sh(s.llen),
        [q(r) for r in s.rects], [q(r) for r in s.grects],
    )  # fmt: skip


def _text_clash(p: _View, q: _View, k: Consts) -> Mask:
    a, b = p.box, q.box
    if not k.ink:
        out = _boxes_near(a, b, k.m, k.m)
        if k.side > k.m:  # _beside_boxes
            rows = np.minimum(a[3], b[3]) > np.maximum(a[1], b[1])
            out = out | (rows & (np.maximum(a[0], b[0]) - np.minimum(a[2], b[2]) < k.side))
        return out
    hit = np.zeros(np.broadcast_shapes(a[0].shape, b[0].shape), dtype=bool)
    for ra in p.rects:
        for rb in q.rects:
            hit |= _rects_clash(ra, rb, k.side, k.vsep, k.tolerance)
    return hit & _boxes_near(a, b, k.side, k.vsep)


def _leader_over_text(p: _View, q: _View, k: Consts) -> Mask:
    """p's leader over q's text (box grown by mt; ink: also one of q's grown glyph lines)."""
    hit = _seg_box(np.float64(p.ex), np.float64(p.ey), p.cx, p.cy, q.gbox)
    if not k.ink:
        return hit
    lines = np.zeros_like(hit)
    for r in q.grects:
        lines |= _seg_box(np.float64(p.ex), np.float64(p.ey), p.cx, p.cy, r)
    return hit & lines


def _leaders_cross(p: _View, q: _View) -> Mask:
    ex, ey, fx, fy = (np.float64(v) for v in (p.ex, p.ey, q.ex, q.ey))
    return _cross(ex, ey, p.cx, p.cy, fx, fy, q.cx, q.cy)


def _conf(p: _View, q: _View, k: Consts) -> NDArray[np.int8]:
    """pconf: the int sum of the four kinds."""
    total = (
        _text_clash(p, q, k).astype(np.int8)
        + _leaders_cross(p, q).astype(np.int8)
        + _leader_over_text(p, q, k).astype(np.int8)
        + _leader_over_text(q, p, k).astype(np.int8)
    )
    return np.asarray(total, dtype=np.int8)


def _leader_into(p: _View, q: _View, k: Consts) -> Arr:
    """Main: 5 * seg_box_len(p's leader, q's box grown by mt) / fs."""
    return 5 * _seg_box_len(p.ex, p.ey, p.cx, p.cy, p.llen, q.gbox) / k.fs


def _leader_into_lines(p: _View, q: _View, k: Consts) -> Arr:
    """Ink: the same over q's grown lines, summed in line order from 0, gated by the box."""
    total: Arr = np.zeros(np.broadcast_shapes(p.cx.shape, q.cx.shape))
    for r in q.grects:
        total = total + 5 * _seg_box_len(p.ex, p.ey, p.cx, p.cy, p.llen, r) / k.fs
    gate = _seg_box(np.float64(p.ex), np.float64(p.ey), p.cx, p.cy, q.gbox)
    return np.where(gate, total, 0.0)


def _area(a: Quad, b: Quad, mx: float, my: float, fs: float) -> Arr:
    ox = np.minimum(a[2], b[2]) - np.maximum(a[0], b[0]) + mx
    oy = np.minimum(a[3], b[3]) - np.maximum(a[1], b[1]) + my
    return np.where((ox > 0) & (oy > 0), 6 * (ox / fs) * (oy / fs), 0.0)


def _soft(p: _View, q: _View, k: Consts) -> Arr:
    """psoft, its terms added to ``pen`` in psoft's order (adding a skipped term's 0.0 to a
    non-negative sum changes nothing)."""
    if not k.ink:
        pen = _area(p.box, q.box, k.m, k.m, k.fs)
        pen = pen + _leader_into(p, q, k)
        pen = pen + _leader_into(q, p, k)
    else:
        pen = np.zeros(np.broadcast_shapes(p.cx.shape, q.cx.shape))
        near = _boxes_near(p.box, q.box, k.m, k.vsep)
        for ra in p.rects:
            for rb in q.rects:
                pen = pen + np.where(near, _area(ra, rb, k.m, k.vsep, k.fs), 0.0)
        pen = pen + _leader_into_lines(p, q, k)
        pen = pen + _leader_into_lines(q, p, k)
    d = _seg_seg_d(p, q, k.leadthr)
    return pen + np.where(d < k.leadthr, 4 * (k.leadthr - d) / k.fs, 0.0)


def _seg_seg_d(p: _View, q: _View, thr: float) -> Arr:
    ax, ay, cx, cy = (np.float64(v) for v in (p.ex, p.ey, q.ex, q.ey))
    bx, by, dx, dy = p.cx, p.cy, q.cx, q.cy
    d = np.minimum(
        np.minimum(_pt_seg_d(ax, ay, cx, cy, dx, dy, thr), _pt_seg_d(bx, by, cx, cy, dx, dy, thr)),
        np.minimum(_pt_seg_d(cx, cy, ax, ay, bx, by, thr), _pt_seg_d(dx, dy, ax, ay, bx, by, thr)),
    )
    return np.where(_cross(ax, ay, bx, by, cx, cy, dx, dy), 0.0, d)


# --- the table --------------------------------------------------------------------------------


Key = tuple[int, int]  # i < j


class Pairs:
    """predicates.pconf and psoft for every candidate pair of every two labels that interact
    (the PairTable the search reads). Every returned array is read-only (the tables are shared).

    Stored once per unordered pair, i < j; the other order is the transpose, so conf(j, i) ==
    conf(i, j).T exactly, and soft(j, i) is psoft(i, j) transposed (psoft adds its two leader
    terms in argument order, so the two orders can differ in the last bit; one is kept so that
    both labels see the same cost). All-zero tables are a broadcast zero, not stored.

    Eager (the default), neighbours are exactly the labels with a non-zero entry. With a lazy
    soft table (``build_pairs(lazy_soft=True)``), a pair's soft table is computed on first use
    and cached, and neighbours are every pair whose envelopes meet, since the soft terms are not
    yet known to be zero."""

    def __init__(
        self,
        sizes: Sequence[int],
        conf: dict[Key, NDArray[np.int8] | None],
        soft: dict[Key, Arr | None],
        later: Callable[[int, int], Arr] | None = None,
    ) -> None:
        self.n = len(sizes)
        self._sizes = tuple(sizes)
        self._conf = conf
        self._soft = soft
        self._later = later
        nb: list[list[int]] = [[] for _ in sizes]
        for i, j in conf:
            nb[i].append(j)
            nb[j].append(i)
        self._nb = tuple(tuple(sorted(v)) for v in nb)

    def neighbours(self, i: int) -> Sequence[int]:
        return self._nb[i]

    def conf(self, i: int, j: int) -> NDArray[np.int8]:
        return self._oriented(i, j, self._conf.get(_key(i, j)), np.int8(0))

    def soft(self, i: int, j: int) -> Arr:
        key = _key(i, j)
        if key not in self._soft and key in self._conf and self._later is not None:
            self._soft[key] = _kept(self._later(*key))
        return self._oriented(i, j, self._soft.get(key), np.float64(0.0))

    def _oriented(self, i: int, j: int, table: _A | None, zero: np.generic) -> _A:
        if table is None:
            return np.broadcast_to(zero, (self._sizes[i], self._sizes[j]))  # type: ignore[return-value]
        return table if i < j else table.T

    @property
    def nbytes(self) -> int:
        """The memory the stored (or so far cached) tables take."""
        tables = [*self._conf.values(), *self._soft.values()]
        return sum(t.nbytes for t in tables if t is not None)


def _key(i: int, j: int) -> Key:
    if i == j:
        raise ValueError(f"no pair table for a label with itself ({i})")
    return (i, j) if i < j else (j, i)


def _kept(a: _A) -> _A | None:
    """None for an all-zero table; otherwise the table, read-only, since every caller shares
    it and none may change another's costs."""
    if not a.any():
        return None
    a.setflags(write=False)
    return a


def build_pairs(
    scene: Scene, cands: Sequence[Cands], opts: Options, *, lazy_soft: bool = False
) -> Pairs:
    """Tabulate every pair whose envelopes meet; pairs that do not are exactly zero. Soft
    tables are the bulk of the memory (float64: float32 would change sums and so ties); a
    caller that may not need them all can defer them with ``lazy_soft``."""
    if len(cands) != len(scene.labels):
        raise ValueError(f"{len(cands)} candidate lists for {len(scene.labels)} labels")
    k = consts(scene, opts)
    r = reach(k)
    sides = [_side(scene, i, c, k) for i, c in enumerate(cands)]
    conf: dict[Key, NDArray[np.int8] | None] = {}
    soft: dict[Key, Arr | None] = {}
    for i, j in _candidate_pairs(sides, r):
        conf[(i, j)] = _kept(_pair(sides[i], sides[j], k, r, _conf))
        if lazy_soft:
            continue
        soft[(i, j)] = _kept(_pair(sides[i], sides[j], k, r, _soft))
        if conf[(i, j)] is None and soft[(i, j)] is None:  # scores nothing: not a neighbour
            del conf[(i, j)], soft[(i, j)]

    def later(i: int, j: int) -> Arr:
        return _pair(sides[i], sides[j], k, r, _soft)

    return Pairs([len(c) for c in cands], conf, soft, later if lazy_soft else None)


def _candidate_pairs(sides: list[_Side], r: float) -> list[Key]:
    live = [n for n, s in enumerate(sides) if s.env is not None]
    if not live:
        return []
    env = np.array([sides[n].env for n in live], dtype=np.float64)
    e = (env[:, 0:1], env[:, 1:2], env[:, 2:3], env[:, 3:4])
    et = (env[:, 0], env[:, 1], env[:, 2], env[:, 3])
    meet = _meets(e, et, r)
    return [(live[a], live[b]) for a, b in zip(*np.nonzero(np.triu(meet, 1)), strict=True)]


def _pair(
    p: _Side, q: _Side, k: Consts, r: float, term: Callable[[_View, _View, Consts], _A]
) -> _A:
    """One pair's matrix of one term, computed only on the rows and columns whose candidate
    envelope reaches the other label at all (the rest are zero by the same bound)."""
    assert p.env is not None and q.env is not None
    rows = np.nonzero(_meets(p.cenv, q.env, r))[0]
    cols = np.nonzero(_meets(q.cenv, p.env, r))[0]
    pv, qv = _view(p, rows, True), _view(q, cols, False)
    block = term(pv, qv, k)
    out = np.zeros((len(p.cx), len(q.cx)), dtype=block.dtype)
    out[np.ix_(rows, cols)] = block
    return out  # type: ignore[return-value]
