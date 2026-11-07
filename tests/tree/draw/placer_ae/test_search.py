"""The seeded search (spec §4) on hand-made and random synthetic problems: the RNG against the
spec's oracle, each phase doing its one job, determinism, and a brute-force optimum on tiny
problems. Labels and costs are invented; the pair terms come from explicit matrices."""

from __future__ import annotations

import itertools
import math
import os
import time
from dataclasses import replace

import numpy as np
import pytest

from af.tree.draw.placer_ae.predicates import angle, inv
from af.tree.draw.placer_ae.scene import Cands, Label, Options, Schedule
from af.tree.draw.placer_ae.search import (
    XorShift32,
    _acost,
    _icost,
    _problem,
    _State,
    search,
    search_with_report,
)

FS = 9.5
TEXTS = ("K2N", "A6S", "L7F", "T9I", "N4D", "G5R")


class DensePairs:
    """PairTable from explicit (i < j) -> (conf, soft) matrices; every other pair is zero."""

    def __init__(self, n: int, mats: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]]):
        self.n = n
        self.mats = {
            k: (np.asarray(c, dtype=np.int8), np.asarray(s, dtype=float))
            for k, (c, s) in mats.items()
        }
        self.nb: list[set[int]] = [set() for _ in range(n)]
        for i, j in mats:
            assert i < j
            self.nb[i].add(j)
            self.nb[j].add(i)

    def neighbours(self, i: int) -> list[int]:
        return sorted(self.nb[i])

    def _get(self, i: int, j: int, k: int) -> np.ndarray:
        return self.mats[(i, j)][k] if i < j else self.mats[(j, i)][k].T

    def conf(self, i: int, j: int) -> np.ndarray:
        return self._get(i, j, 0)

    def soft(self, i: int, j: int) -> np.ndarray:
        return self._get(i, j, 1)


def _label(k: int, ny: float, x: float = 200.0) -> Label:
    text = TEXTS[k % len(TEXTS)]
    return Label(k, text, (text,), (0.6 * FS * len(text),), FS, (x, ny), (x - 6.0, ny))


def _cands(ys: list[float], bases: list[float], x1: float = 150.0) -> Cands:
    """Boxes left of the branch centred on each y; the leader attaches at the text's end."""
    h = 1.18 * FS
    return Cands.of(
        [(x1 - 20, y - h / 2, x1, y + h / 2, x1, y, b, 1) for y, b in zip(ys, bases, strict=True)]
    )


def _opts(**sched: int) -> Options:
    return Options(schedule=replace(Schedule(), **sched))


# Only the seed: no descent, no repair, no tidy.
SEED_ONLY = {"restarts": 0, "sweeps": 0, "pair_passes": 0, "window_passes": 0, "tidy_sweeps": 0}


# ---- reference objectives (plain loops over the matrices and predicates.inv / angle) --------


def _placed(labels, cands, i, c):
    return (labels[i], cands[i].box(c), (float(cands[i].cx[c]), float(cands[i].cy[c])))


def _ref_score_a(labels, cands, pairs, choice) -> float:
    w = Options().weights
    n = len(labels)
    conf = soft = 0.0
    invs = 0
    for i in range(n):
        for j in range(i + 1, n):
            if j in pairs.nb[i]:
                conf += int(pairs.conf(i, j)[choice[i], choice[j]])
                soft += float(pairs.soft(i, j)[choice[i], choice[j]])
            invs += inv(_placed(labels, cands, i, choice[i]), _placed(labels, cands, j, choice[j]))
    base = sum(float(cands[i].base[choice[i]]) for i in range(n))
    return conf * w.wc + soft * w.wsoft + invs * w.wo + base


def _ref_conf_i(pairs, choice, i, ci) -> int:
    return sum(int(pairs.conf(i, j)[ci, choice[j]]) for j in pairs.nb[i])


def _ref_icost(labels, cands, pairs, choice, i, ci) -> int:
    n = len(labels)
    order = sorted(range(n), key=lambda k: labels[k].node[1])
    rank = order.index(i)
    p = _placed(labels, cands, i, ci)
    c = int(cands[i].base[ci]) + _ref_conf_i(pairs, choice, i, ci) * 1_000_000
    c += sum(inv(p, _placed(labels, cands, j, choice[j])) * 1100 for j in range(n) if j != i)
    for r in (rank - 1, rank + 1):
        if 0 <= r < n:
            nb = order[r]
            c += int(130 * abs(angle(p) - angle(_placed(labels, cands, nb, choice[nb]))))
    return c


def _random_problem(seed: int, n: int, nc: int, density: float, reach: int = 3, soft: bool = True):
    """Labels down a column, candidates scattered round each node, conflicts between labels
    within ``reach`` places of each other."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    labels = [_label(k, 30.0 * float(order[k])) for k in range(n)]
    cands = []
    for k in range(n):
        ys = list(labels[k].node[1] + rng.uniform(-40, 40, nc))
        cands.append(_cands(ys, list(np.round(rng.uniform(0, 500, nc), 1))))
    mats = {}
    for i in range(n):
        for j in range(i + 1, n):
            if abs(int(order[i]) - int(order[j])) <= reach:
                c = (rng.random((nc, nc)) < density) * rng.integers(1, 3, (nc, nc))
                s = rng.random((nc, nc)) * (c > 0) * 0.3 if soft else np.zeros((nc, nc))
                mats[(i, j)] = (c, s)
    return labels, cands, DensePairs(n, mats)


# ---- RNG ------------------------------------------------------------------------------------

ORACLE_STATES = (
    723471715,
    2497366906,
    2064144800,
    2008045182,
    3532304609,
    374114282,
    1350636274,
    691148861,
    746858951,
    2653896249,
)
ORACLE_MOD40 = (35, 26, 0, 22, 9, 2, 34, 21, 31, 9)


def test_rng_matches_oracle():
    rng = XorShift32(Schedule().seed)
    mods = []
    for state in ORACLE_STATES:
        mods.append(rng.draw(40))
        assert rng.state == state
    assert tuple(mods) == ORACLE_MOD40


def test_rng_draw_one_is_zero_and_still_advances():
    rng = XorShift32(Schedule().seed)
    assert rng.draw(1) == 0
    assert rng.state == ORACLE_STATES[0]


# ---- seed -----------------------------------------------------------------------------------


def test_seed_goes_down_by_node_y_and_avoids_placed_labels():
    # label 1 hangs higher, so is placed first and takes its cheapest; label 0 then skips the
    # candidate that clashes with it, and the one above label 1's box (the +3000).
    labels = [_label(0, 100.0), _label(1, 50.0)]
    cands = [_cands([110, 105, 40, 120], [1, 2, 0, 9]), _cands([60, 50], [3, 1])]
    conf = np.zeros((4, 2))
    conf[0, 1] = 1  # label 0's first candidate on label 1's second
    pairs = DensePairs(2, {(0, 1): (conf, np.zeros((4, 2)))})
    choice, rep = search_with_report(labels, cands, pairs, _opts(**SEED_ONLY))
    assert list(choice) == [1, 1]
    assert rep.conflicts["seed"] == 0


def test_seed_takes_first_of_equal_costs():
    labels = [_label(0, 10.0), _label(1, 20.0)]
    cands = [_cands([10, 11, 12], [5, 5, 5]), _cands([20, 21], [7, 7])]
    pairs = DensePairs(2, {})
    assert list(search(labels, cands, pairs, _opts(**SEED_ONLY))) == [0, 0]


# ---- Phase A, A2, A3 ------------------------------------------------------------------------


def test_phase_a_fixes_a_conflict_one_move_removes():
    # The seed gives label 0 its cheapest; every candidate of label 1 clashes with it. Moving
    # label 0 alone fixes it.
    labels = [_label(0, 10.0), _label(1, 40.0)]
    cands = [_cands([10, 12], [1, 50]), _cands([40, 42], [1, 2])]
    conf = np.array([[1, 1], [0, 0]])
    pairs = DensePairs(2, {(0, 1): (conf, np.zeros((2, 2)))})
    opts = _opts(restarts=0, pair_passes=0, window_passes=0, tidy_sweeps=0)
    choice, rep = search_with_report(labels, cands, pairs, opts)
    assert rep.conflicts["seed"] == 1
    assert rep.conflicts["A"] == 0
    assert list(choice) == [1, 0]


def _both_must_move():
    labels = [_label(0, 10.0), _label(1, 40.0)]
    cands = [_cands([10, 12], [1, 2]), _cands([40, 42], [1, 2])]
    conf = np.array([[1, 1], [1, 0]])  # clean only when both take their second candidate
    return labels, cands, DensePairs(2, {(0, 1): (conf, np.zeros((2, 2)))})


def test_a2_moves_two_labels_together():
    labels, cands, pairs = _both_must_move()
    opts = _opts(restarts=0, window_passes=0, tidy_sweeps=0)
    choice, rep = search_with_report(labels, cands, pairs, opts)
    assert rep.conflicts["A"] == 1  # no single move helps
    assert rep.conflicts["A2"] == 0
    assert rep.pair_moves == 1
    assert list(choice) == [1, 1]


def test_a3_moves_a_window_together():
    # Three labels, each pair clean only when both take candidate 1: from all-zero no single
    # move helps and no pair is clean against the third, so only the window's restarts find it.
    labels = [_label(k, 10.0 + 30 * k) for k in range(3)]
    cands = [_cands([10 + 30 * k, 12 + 30 * k], [1, 2]) for k in range(3)]
    conf = np.array([[1, 1], [1, 0]])
    zero = np.zeros((2, 2))
    pairs = DensePairs(3, {(0, 1): (conf, zero), (0, 2): (conf, zero), (1, 2): (conf, zero)})
    opts = _opts(restarts=0, tidy_sweeps=0)
    choice, rep = search_with_report(labels, cands, pairs, opts)
    assert rep.conflicts["A2"] > 0
    assert rep.conflicts["A3"] == 0
    assert list(choice) == [1, 1, 1]
    assert rep.draws == 3 * rep.window_restarts


# ---- Phase B --------------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(6))
def test_phase_b_never_adds_conflicts_and_ends_at_its_local_optimum(seed):
    labels, cands, pairs = _random_problem(seed, n=10, nc=8, density=0.5, soft=False)
    opts = _opts(restarts=20)
    choice, rep = search_with_report(labels, cands, pairs, opts)
    assert rep.conflicts["B"] <= rep.conflicts["A3"]
    assert rep.sweeps_b < opts.schedule.tidy_sweeps  # converged, not cut off
    ch = list(choice)
    for i in range(len(labels)):
        cur = _ref_icost(labels, cands, pairs, ch, i, ch[i])
        cf = _ref_conf_i(pairs, ch, i, ch[i])
        for ci in range(len(cands[i])):
            if _ref_conf_i(pairs, ch, i, ci) <= cf:
                assert _ref_icost(labels, cands, pairs, ch, i, ci) >= cur


def test_phase_b_lowers_order_and_angle_cost():
    # Without conflicts Phase A already optimised scoreA; B trades by icost, never raising it
    # for the label that moves: here the sum over labels falls.
    labels, cands, pairs = _random_problem(3, n=12, nc=10, density=0.05)
    before = search(labels, cands, pairs, _opts(restarts=10, tidy_sweeps=0))
    after = search(labels, cands, pairs, _opts(restarts=10))

    def total(ch):
        return sum(_ref_icost(labels, cands, pairs, list(ch), i, ch[i]) for i in range(len(ch)))

    assert total(after) <= total(before)


# ---- pinned, determinism --------------------------------------------------------------------


def test_pinned_labels_never_move_and_still_draw():
    labels, cands, pairs = _random_problem(7, n=6, nc=6, density=0.4)
    cands[2] = _cands([float(cands[2].cy[3])], [0.0])
    mats = {k: v for k, v in pairs.mats.items() if 2 not in k}
    for k, (c, s) in pairs.mats.items():
        if 2 in k:
            mats[k] = (c[3:4], s[3:4]) if k[0] == 2 else (c[:, 3:4], s[:, 3:4])
    pairs = DensePairs(6, mats)
    choice, rep = search_with_report(
        labels, cands, pairs, _opts(restarts=5, window_passes=0, stale=1000)
    )
    assert choice[2] == 0
    assert rep.draws == 5 * 6  # n draws per restart, the pinned label's included


def test_deterministic_for_a_seed_and_seed_dependent_with_ties():
    labels, cands, pairs = _random_problem(11, n=8, nc=6, density=0.6, soft=False)
    for c in cands:
        c.base[:] = 10.0 * (np.arange(len(c)) % 2)  # many equal-cost layouts
    first = search_with_report(labels, cands, pairs, _opts(restarts=30))
    again = search_with_report(labels, cands, pairs, _opts(restarts=30))
    assert list(first[0]) == list(again[0])
    assert first[1].score_a == again[1].score_a
    seen = {tuple(search(labels, cands, pairs, _opts(restarts=30, seed=s))) for s in range(1, 9)}
    assert len(seen) > 1


# ---- brute force ----------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(24))
def test_tiny_problems_reach_the_brute_force_optimum(seed):
    """Seeds 0-15 have a conflict-free layout, 16-23 (dense) mostly do not."""
    n, nc = 3 + seed % 2, 4 + seed % 2
    density = 0.35 if seed < 16 else 0.85
    labels, cands, pairs = _random_problem(100 + seed, n=n, nc=nc, density=density, reach=n)
    every = list(itertools.product(range(nc), repeat=n))
    scores = [_ref_score_a(labels, cands, pairs, ch) for ch in every]
    feasible = any(sum(_ref_conf_i(pairs, ch, i, ch[i]) for i in range(n)) == 0 for ch in every)
    best = min(scores)

    choice, rep = search_with_report(labels, cands, pairs, Options())
    if feasible:
        assert rep.conflicts["B"] == 0
    # Without the tidy, the result is Phase A's best (A2 and A3 move only conflicting labels):
    # on a problem this small its 401 restarts reach the optimum.
    choice_a, rep_a = search_with_report(labels, cands, pairs, _opts(tidy_sweeps=0, stale=1000))
    if rep_a.conflicts["A"] == rep_a.conflicts["A3"]:
        assert rep_a.score_a <= best + 1e-9 * abs(best)
    assert math.isclose(
        rep_a.score_a, _ref_score_a(labels, cands, pairs, list(choice_a)), rel_tol=1e-12
    )
    assert math.isclose(
        rep.score_a, _ref_score_a(labels, cands, pairs, list(choice)), rel_tol=1e-12
    )


def test_inputs_are_checked():
    labels = [_label(0, 10.0), _label(1, 20.0)]
    cands = [_cands([10], [1]), _cands([20, 21], [1, 2])]
    bad = DensePairs(2, {(0, 1): (np.zeros((2, 2)), np.zeros((2, 2)))})
    with pytest.raises(ValueError, match="matrices"):
        search(labels, cands, bad, Options())
    with pytest.raises(ValueError, match="candidate lists"):
        search(labels[:1], cands, DensePairs(2, {}), Options())


# ---- timing ---------------------------------------------------------------------------------


def _timing_problem(n: int = 60, nc: int = 190, k: int = 10):
    """Real-tree-sized: n labels, nc candidates each, about k neighbours each, conflicts sparse
    enough that a conflict-free layout exists (the stall rule then applies, as on a real tree)."""
    return _random_problem(2026, n=n, nc=nc, density=0.04, reach=k // 2)


def test_timing_60_labels(capsys):
    """Prints the full schedule's runtime (no assertion). The full run takes tens of seconds, so
    it runs only with AF_PLACER_TIMING=1; otherwise a 20-restart run, to keep the path tested."""
    labels, cands, pairs = _timing_problem()
    full = os.environ.get("AF_PLACER_TIMING") == "1"
    opts = Options() if full else _opts(restarts=20)
    t0 = time.perf_counter()
    choice, rep = search_with_report(labels, cands, pairs, opts)
    dt = time.perf_counter() - t0
    with capsys.disabled():
        print(
            f"\nsearch, {len(labels)} labels x {len(cands[0])} candidates, "
            f"{sum(len(pairs.nb[i]) for i in range(len(labels))) / len(labels):.1f} neighbours: "
            f"{dt:.2f} s ({'full' if full else '20 restarts'}); restarts {rep.restarts}, "
            f"sweeps A {rep.sweeps_a}, B {rep.sweeps_b}; conflicts {rep.conflicts}; "
            f"seconds {rep.seconds}"
        )
    assert len(choice) == len(labels)


# ---- the vectorised costs are ae's loops ----------------------------------------------------


def test_vector_costs_equal_the_loops_exactly():
    """Phase A's and B's cost vectors, against plain loops over the reference predicates, to
    the last bit; two labels share a node (inv's asymmetric tie, spec §3.3)."""
    labels, cands, pairs = _random_problem(5, n=9, nc=7, density=0.4)
    labels[4] = replace(labels[4], node=labels[1].node, end=labels[1].end)
    prob = _problem(labels, cands, pairs)
    st = _State(prob, Options())
    rng = np.random.default_rng(0)
    for _ in range(5):
        ch = [int(rng.integers(len(c))) for c in cands]
        st.set_all(np.array(ch))
        for i in range(len(labels)):
            for ci in range(len(cands[i])):
                p = _placed(labels, cands, i, ci)
                soft = 0.0
                for j in range(len(labels)):
                    if j in pairs.nb[i]:
                        soft += float(pairs.soft(i, j)[ci, ch[j]])
                invs = sum(
                    inv(p, _placed(labels, cands, j, ch[j])) for j in range(len(labels)) if j != i
                )
                conf = _ref_conf_i(pairs, ch, i, ci)
                want = float(conf) * 1.0e6 + soft * 5000.0 + float(invs) * 1100.0
                want += float(cands[i].base[ci])
                assert _acost(st, i)[ci] == want
                assert _icost(st, i, st.conf_vec(i))[ci] == _ref_icost(
                    labels, cands, pairs, ch, i, ci
                )
