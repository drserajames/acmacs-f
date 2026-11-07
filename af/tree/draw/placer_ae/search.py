"""ae's seeded search over all labels at once (draw-tree.cc M:2858-3072; spec §4).

Order of play, each phase as ae's: an ordered seed down the tree; Phase A, a descent from the
seed and from many random restarts on the soft objective; A2, moving two conflicting labels
together; A3, re-solving a window of labels around a stubborn conflict; Phase B, a tidy that
never adds a conflict. Every argmin keeps the first minimum and every comparison is strict, as
ae's loops are, so equal-cost candidates resolve to the earliest in the candidate order.

Why not ae's loops as written: each of ae's per-candidate costs is a sum over all labels, which
in Python is ~10^9 pair tests. The pair terms depend only on the two candidates, so they come
precomputed (PairTable) and a label's cost over all its candidates is a few vector adds, one per
neighbour. Sums whose float order could split a tie are added in ae's order (neighbours by index,
pairs i < j), so a cost here is the same double ae would compute from the same pair values.
"""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from .scene import Cands, Label, Options

MASK32 = 0xFFFFFFFF
# The seed's surcharge for a candidate above the previous label's (M:2966): ae's literal, not one
# of its named weights, so not in Weights either.
SEED_DESCEND = 3000.0
_BIG = np.iinfo(np.int64).max


class PairTable(Protocol):
    """The pairwise terms, precomputed (pairs.py). ``conf(i, j)[ci, cj]`` is pconf(i, ci, j, cj)
    and ``soft`` psoft likewise, each symmetric under transposition; a pair not listed in
    ``neighbours`` contributes exactly 0 to both."""

    n: int

    def neighbours(self, i: int) -> Sequence[int]: ...

    def conf(self, i: int, j: int) -> np.ndarray: ...

    def soft(self, i: int, j: int) -> np.ndarray: ...


class XorShift32:
    """ae's restart RNG (M:2862-2863): Marsaglia xorshift32 on a uint32, ``state % mm`` after
    each update. A seed parameter, not an environment variable, so a layout is reproducible
    from its options alone (design rule 8)."""

    def __init__(self, seed: int) -> None:
        self.state = seed & MASK32
        self.draws = 0

    def draw(self, mm: int) -> int:
        s = self.state
        s ^= (s << 13) & MASK32
        s ^= s >> 17
        s ^= (s << 5) & MASK32
        self.state = s
        self.draws += 1
        return s % mm


@dataclass
class SearchReport:
    """What each phase did, for the report: how hard the search had to work and where the
    remaining conflicts were (or were not) removed."""

    restarts: int = 0  # random restarts run in Phase A (after the seed's descent)
    best_restart: int = 0  # the restart whose layout Phase A kept (0: the seed's)
    sweeps_a: int = 0
    pair_passes: int = 0  # A2
    pair_moves: int = 0
    window_passes: int = 0  # A3
    window_restarts: int = 0
    sweeps_b: int = 0
    draws: int = 0  # RNG draws, both phases
    conflicts: dict[str, int] = field(default_factory=dict)  # conf_total after each phase
    conf_total: int = 0  # the final layout's scoreA components
    soft_total: float = 0.0
    inv_total: int = 0
    base_total: float = 0.0
    score_a: float = 0.0
    seconds: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class _Problem:
    """The static inputs, in the shapes the phases index: per label its candidates' base, box
    centre y and leader angle; per neighbour pair the two matrices."""

    base: list[np.ndarray]
    base_l: list[np.ndarray]  # ae's static_cast<long>(base): truncated toward zero
    cy: list[np.ndarray]  # box centre y, (y0 + y1) / 2: vertical order
    ang: list[np.ndarray]  # leader angle from label.end to the attach point: tidy pass
    ny: np.ndarray  # each label's node y
    nbrs: list[np.ndarray]  # sorted ascending
    conf: list[dict[int, np.ndarray]]  # conf[i][j]: (C_i, C_j) int8
    soft: list[dict[int, np.ndarray]]
    stack: list[np.ndarray]  # stack[i]: the conf[i][j].T blocks, rows = j's candidates
    offs: list[np.ndarray]  # where each neighbour's block starts in stack[i]
    pairs: list[tuple[int, int]]  # neighbour pairs, i < j, in index order
    ord_a: np.ndarray
    rank_a: np.ndarray

    @property
    def n(self) -> int:
        return len(self.base)


def search(
    labels: Sequence[Label], cands: Sequence[Cands], pairs: PairTable, opts: Options
) -> np.ndarray:
    """The chosen candidate per label (indices into each Cands)."""
    return search_with_report(labels, cands, pairs, opts)[0]


def search_with_report(
    labels: Sequence[Label], cands: Sequence[Cands], pairs: PairTable, opts: Options
) -> tuple[np.ndarray, SearchReport]:
    prob = _problem(labels, cands, pairs)
    st = _State(prob, opts)
    rep = SearchReport()
    clock = time.perf_counter()
    for name, phase in (
        ("seed", _seed),
        ("A", _phase_a),
        ("A2", _phase_a2),
        ("A3", _phase_a3),
        ("B", _phase_b),
    ):
        phase(st, rep)
        rep.conflicts[name] = st.conf_total()
        now = time.perf_counter()
        rep.seconds[name] = round(now - clock, 3)
        clock = now
    rep.draws = st.rng.draws
    rep.conf_total, rep.soft_total, rep.inv_total, rep.base_total = st.components()
    rep.score_a = st.score_a()
    return st.choice.copy(), rep


# ---- inputs -------------------------------------------------------------------------------


def _problem(labels: Sequence[Label], cands: Sequence[Cands], pairs: PairTable) -> _Problem:
    n = len(labels)
    if len(cands) != n or pairs.n != n:
        raise ValueError(f"{n} labels, {len(cands)} candidate lists, a pair table for {pairs.n}")
    for i, c in enumerate(cands):
        if len(c) == 0:
            raise ValueError(f"label {i} has no candidates")
        if not np.all(np.isfinite(c.base)) or np.any(c.base < 0):
            raise ValueError(f"label {i}: base costs must be finite and >= 0")
    nbrs = [
        np.array(sorted({int(j) for j in pairs.neighbours(i)}), dtype=np.int64) for i in range(n)
    ]
    conf: list[dict[int, np.ndarray]] = [{} for _ in range(n)]
    soft: list[dict[int, np.ndarray]] = [{} for _ in range(n)]
    for i in range(n):
        for j in (int(x) for x in nbrs[i]):
            if j == i or i not in set(nbrs[j].tolist()):
                raise ValueError(f"neighbours not symmetric (or self) at labels {i}, {j}")
            if j < i:
                continue
            shape = (len(cands[i]), len(cands[j]))
            c_ij = np.asarray(pairs.conf(i, j), dtype=np.int8)
            s_ij = np.asarray(pairs.soft(i, j), dtype=np.float64)
            if c_ij.shape != shape or s_ij.shape != shape:
                raise ValueError(
                    f"pair ({i}, {j}): matrices {c_ij.shape}/{s_ij.shape}, not {shape}"
                )
            conf[i][j], conf[j][i] = c_ij, c_ij.T
            soft[i][j], soft[j][i] = s_ij, s_ij.T
    stack, offs = _stacks(cands, nbrs, conf)
    ny = np.array([lb.node[1] for lb in labels], dtype=np.float64)
    ord_a = np.argsort(ny, kind="stable")  # spec §4.7 reading (a): ties in index order
    rank_a = np.empty(n, dtype=np.int64)
    rank_a[ord_a] = np.arange(n)
    return _Problem(
        base=[c.base for c in cands],
        base_l=[c.base.astype(np.int64) for c in cands],
        cy=[(c.y0 + c.y1) / 2 for c in cands],
        ang=[
            np.arctan2(lb.end[1] - c.cy, lb.end[0] - c.cx)
            for lb, c in zip(labels, cands, strict=True)
        ],
        ny=ny,
        nbrs=nbrs,
        conf=conf,
        soft=soft,
        stack=stack,
        offs=offs,
        pairs=[(i, int(j)) for i in range(n) for j in nbrs[i] if j > i],
        ord_a=ord_a,
        rank_a=rank_a,
    )


def _stacks(
    cands: Sequence[Cands], nbrs: list[np.ndarray], conf: list[dict[int, np.ndarray]]
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Each label's neighbour conf blocks stacked by the neighbour's candidate, so its conflict
    vector over its own candidates is one row gather and one sum (int: order is immaterial)."""
    stack, offs = [], []
    for i, nb in enumerate(nbrs):
        sizes = [len(cands[int(j)]) for j in nb]
        offs.append(np.concatenate([[0], np.cumsum(sizes)[:-1]]).astype(np.int64))
        blocks = [conf[i][int(j)].T for j in nb]
        stack.append(
            np.ascontiguousarray(np.concatenate(blocks))
            if blocks
            else np.zeros((0, len(cands[i])), dtype=np.int8)
        )
    return stack, offs


# ---- state --------------------------------------------------------------------------------


class _State:
    """The current choice and, per label, its conflict and soft vectors against the others'
    current choices. A label's vectors are recomputed only after a neighbour moved: late
    sweeps move few labels, so most evaluations reuse them."""

    def __init__(self, prob: _Problem, opts: Options) -> None:
        self.p = prob
        w = opts.weights
        self.sched = opts.schedule
        self.wc = float(w.wc)  # Phase A and the seed: ae's double 1.0e6
        self.wc_l = int(w.wc)  # A3 and B: ae's long WC
        self.wo = int(w.wo)  # ae's long WO, used as a double in Phase A as well
        self.wa = float(w.wa)
        self.wsoft = float(w.wsoft)
        self.rng = XorShift32(self.sched.seed)
        n = prob.n
        self.choice = np.zeros(n, dtype=np.int64)
        self.cur_cy = np.array([cy[0] for cy in prob.cy])
        self._conf: list[np.ndarray | None] = [None] * n
        self._soft: list[np.ndarray | None] = [None] * n
        self.above = prob.ny[:, None] < prob.ny[None, :]  # above[i, j]: ny_i < ny_j

    def set(self, i: int, c: int) -> None:
        self.choice[i] = c
        self.cur_cy[i] = self.p.cy[i][c]
        for j in self.p.nbrs[i]:
            self._conf[j] = None
            self._soft[j] = None

    def set_all(self, choice: np.ndarray) -> None:
        self.choice = np.array(choice, dtype=np.int64)
        self.cur_cy = np.array([self.p.cy[i][c] for i, c in enumerate(self.choice)])
        self._conf = [None] * self.p.n
        self._soft = [None] * self.p.n

    def conf_vec(self, i: int) -> np.ndarray:
        """conf_i(i, ci) for every ci: int64."""
        v = self._conf[i]
        if v is None:
            p = self.p
            rows = p.stack[i][p.offs[i] + self.choice[p.nbrs[i]]]
            v = rows.sum(axis=0, dtype=np.int64)
            self._conf[i] = v
        return v

    def soft_vec(self, i: int) -> np.ndarray:
        """soft_i(i, ci) for every ci, summed over neighbours by index from 0.0, as ae's loop
        (a non-neighbour adds exactly 0.0, which changes nothing)."""
        v = self._soft[i]
        if v is None:
            v = np.zeros(len(self.p.base[i]))
            s = self.p.soft[i]
            for j in self.p.nbrs[i]:
                v += s[int(j)][:, self.choice[j]]
            self._soft[i] = v
        return v

    def inv_vec(self, i: int) -> np.ndarray:
        """inv_i(i, ci) for every ci, against all other labels (inv is global, not a neighbour
        term). The j = i column is [cy_i(ci) < cur_cy_i] and is taken off again."""
        cy = self.p.cy[i]
        lower = cy[:, None] < self.cur_cy[None, :]
        full = (lower != self.above[i][None, :]).sum(axis=1)
        return full - (cy < self.cur_cy[i])

    def conf_total(self) -> int:
        ch = self.choice
        return sum(int(self.p.conf[i][j][ch[i], ch[j]]) for i, j in self.p.pairs)

    def components(self) -> tuple[int, float, int, float]:
        """conf_total, soft_total, inv_total, base_total, each summed as ae does (i < j,
        i outer; labels by index)."""
        p, ch = self.p, self.choice
        conf, soft = 0, 0.0
        for i, j in p.pairs:
            conf += int(p.conf[i][j][ch[i], ch[j]])
            soft += float(p.soft[i][j][ch[i], ch[j]])
        inv = self.above != (self.cur_cy[:, None] < self.cur_cy[None, :])
        base = 0.0
        for i in range(p.n):
            base += float(p.base[i][ch[i]])
        return conf, soft, int(np.triu(inv, 1).sum()), base

    def score_a(self) -> float:
        conf, soft, inv, base = self.components()
        return float(conf) * self.wc + soft * self.wsoft + float(inv) * float(self.wo) + base


def _step(cost: np.ndarray, cur: int) -> int:
    """ae's argmin loop: start from the current candidate, scan all, replace on strictly less.
    The result is the first minimum, unless it only equals the current candidate's cost."""
    m = int(np.argmin(cost))
    return m if cost[m] < cost[cur] else cur


# ---- phases -------------------------------------------------------------------------------


def _seed(st: _State, rep: SearchReport) -> None:
    """Down the tree by node y, each label takes its cheapest candidate that clashes with none
    placed before it, with a surcharge for rising above the previous one (M:2954-2971). Ordered
    labels over ordered branches cannot cross, so this starts Phase A at or near feasibility."""
    p = st.p
    seed = np.zeros(p.n, dtype=np.int64)
    placed = np.zeros(p.n, dtype=bool)
    prev_cy = -math.inf
    for i in (int(x) for x in p.ord_a):
        v = p.base[i].copy()
        before = [int(j) for j in p.nbrs[i] if placed[j]]
        for j in sorted(before, key=lambda k: p.rank_a[k]):
            v += p.conf[i][j][:, seed[j]] * st.wc
        v += np.where(p.cy[i] < prev_cy, SEED_DESCEND, 0.0)
        seed[i] = int(np.argmin(v))
        placed[i] = True
        prev_cy = float(p.cy[i][seed[i]])
    st.set_all(seed)


def _acost(st: _State, i: int) -> np.ndarray:
    """Phase A's per-label cost, in ae's order: conf*1e6 + soft*5000 + inv*1100 + base."""
    return (
        st.conf_vec(i) * st.wc
        + st.soft_vec(i) * st.wsoft
        + st.inv_vec(i) * float(st.wo)
        + st.p.base[i]
    )


def _sweep_a(st: _State) -> bool:
    improved = False
    for i in range(st.p.n):
        cur = int(st.choice[i])
        bc = _step(_acost(st, i), cur)
        if bc != cur:
            st.set(i, bc)
            improved = True
    return improved


def _phase_a(st: _State, rep: SearchReport) -> None:
    """The seed's descent, then random restarts among each label's ``pool`` cheapest; keep the
    best scoreA seen after any sweep. The stall counter only runs once the best is under one
    conflict's weight (M:2995), so an infeasible tree uses every restart."""
    sch, n = st.sched, st.p.n
    best, best_s = st.choice.copy(), st.score_a()
    stale = 0
    for restart in range(sch.restarts + 1):
        if restart > 0:
            rep.restarts += 1
            st.set_all(np.array([st.rng.draw(min(sch.pool, len(st.p.base[i]))) for i in range(n)]))
        improved_here = False
        for _ in range(sch.sweeps):
            moved = _sweep_a(st)
            rep.sweeps_a += 1
            s = st.score_a()
            if s < best_s:
                best_s, best, improved_here = s, st.choice.copy(), True
                rep.best_restart = restart
            if not moved:
                break
        if best_s < st.wc:
            stale = 0 if improved_here else stale + 1
            if stale >= sch.stale:
                break
    st.set_all(best)


def _phase_a2(st: _State, rep: SearchReport) -> None:
    """Two labels in conflict only with each other cannot be freed one at a time; move both to
    the cheapest pair of candidates clean against each other and everyone else (M:3003-3020).
    Order and soft terms are ignored, as ae's are."""
    p = st.p
    for _ in range(st.sched.pair_passes):
        rep.pair_passes += 1
        moved = False
        for i, j in p.pairs:
            if _repair_pair(st, i, j):
                rep.pair_moves += 1
                moved = True
        if not moved:
            break


def _repair_pair(st: _State, i: int, j: int) -> bool:
    p = st.p
    ci, cj = int(st.choice[i]), int(st.choice[j])
    m = p.conf[i][j]
    if m[ci, cj] == 0:
        return False
    free_i = (st.conf_vec(i) - m[:, cj]) == 0  # conf_excl(i, :, j) == 0
    free_j = (st.conf_vec(j) - m[ci, :]) == 0
    ok = (m == 0) & free_i[:, None] & free_j[None, :]
    if not ok.any():
        return False
    v = np.where(ok, p.base_l[i][:, None] + p.base_l[j][None, :], _BIG)
    bci, bcj = divmod(int(np.argmin(v)), m.shape[1])  # row-major: ae's ci-outer loop
    st.set(i, bci)
    st.set(j, bcj)
    return True


def _phase_a3(st: _State, rep: SearchReport) -> None:
    """Re-solve the labels within ``window`` ranks of the first conflicting pair, together,
    from random restarts over each one's full candidate list (M:3027-3054): a congested
    cluster needs its innocent neighbours to move too, which no single or pair move tries."""
    sch = st.sched
    for _ in range(sch.window_passes):
        before = st.conf_total()
        if before == 0:
            return
        rep.window_passes += 1
        win = _window(st)
        best_win, best_wc = st.choice[win].copy(), before
        for rs in range(sch.window_restarts):
            if best_wc == 0:
                break
            if rs > 0:
                rep.window_restarts += 1
                for w in win:
                    st.set(int(w), st.rng.draw(len(st.p.base[w])))
            best_win, best_wc = _descend_window(st, win, best_win, best_wc)
        for w, c in zip(win, best_win, strict=True):
            st.set(int(w), int(c))
        if st.conf_total() >= before:
            return


def _window(st: _State) -> np.ndarray:
    p = st.p
    pi, pj = next((i, j) for i, j in p.pairs if p.conf[i][j][st.choice[i], st.choice[j]] > 0)
    w = st.sched.window
    lo = max(0, int(min(p.rank_a[pi], p.rank_a[pj])) - w)
    hi = min(p.n - 1, int(max(p.rank_a[pi], p.rank_a[pj])) + w)
    return p.ord_a[lo : hi + 1]


def _descend_window(
    st: _State, win: np.ndarray, best_win: np.ndarray, best_wc: int
) -> tuple[np.ndarray, int]:
    for _ in range(st.sched.window_sweeps):
        moved = False
        for w in (int(x) for x in win):
            cur = int(st.choice[w])
            bc = _step(st.conf_vec(w) * st.wc_l + st.p.base_l[w], cur)
            if bc != cur:
                st.set(w, bc)
                moved = True
        s = st.conf_total()
        if s < best_wc:
            best_wc, best_win = s, st.choice[win].copy()
        if best_wc == 0 or not moved:
            break
    return best_win, best_wc


def _icost(st: _State, i: int, conf: np.ndarray) -> np.ndarray:
    """Phase B's long cost: base, conflicts, order, and the leader angle against the labels
    either side in branch order, each truncated to long as ae's (M:2850-2857)."""
    p = st.p
    c = p.base_l[i] + conf * st.wc_l + st.inv_vec(i) * st.wo
    r = int(p.rank_a[i])
    for k in (r - 1, r + 1):
        if 0 <= k < p.n:
            nb = int(p.ord_a[k])
            c = c + (st.wa * np.abs(p.ang[i] - p.ang[nb][st.choice[nb]])).astype(np.int64)
    return c


def _phase_b(st: _State, rep: SearchReport) -> None:
    """Tidy order, leader length and neighbour angles, never letting a label take on more
    conflicts than it has (M:3059-3072); pconf is symmetric, so conf_total cannot rise."""
    for _ in range(st.sched.tidy_sweeps):
        rep.sweeps_b += 1
        moved = False
        for i in range(st.p.n):
            cur = int(st.choice[i])
            conf = st.conf_vec(i)
            cost = np.where(conf > conf[cur], _BIG, _icost(st, i, conf))
            bc = _step(cost, cur)
            if bc != cur:
                st.set(i, bc)
                moved = True
        if not moved:
            break
