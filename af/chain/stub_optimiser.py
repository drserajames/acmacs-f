"""A pure-Python stand-in for the optimiser core (workstream 6), for the engine's plumbing.

It takes the interface I1 arrays and returns projections in I1's shape, so replacing it
with the compiled core (`af.map.optimise`) is a one-line change in `backend.py`. The stress
and gradient copy
ae `cc/chart/v3/stress.cc` (regular: (d-D)^2; `<`: (d-D+1)^2 * sigmoid(10(d-D+1)); `>`
ignored; table distance cb - log - avidity, clipped at 0). The minimiser is
a plain gradient descent with Barzilai-Borwein steps (no scipy), so basins differ from
ae; it is for tests and for running without the compiled core, not for replay comparisons.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SIGMOID_MULTIPLIER = 10.0


@dataclass
class StressTerms:
    """The titres that enter stress, as flat arrays (built once per chart)."""

    p1: np.ndarray  # antigen point index
    p2: np.ndarray  # serum point index
    distance: np.ndarray  # table distance
    less_than: np.ndarray  # bool: `<` titre
    weight: np.ndarray
    n_points: int


def stress_terms(arrays: dict) -> StressTerms:
    value, kind = arrays["titre_value"], arrays["titre_type"]
    n_ag, n_sr = value.shape
    cb = arrays["column_bases"]
    disconnected = arrays["disconnected"]
    regular_kinds = [1, 4] if arrays.get("dodgy_is_regular") else [1]
    use = np.isin(kind, regular_kinds + [2])
    use &= ~disconnected[:n_ag, None] & ~disconnected[None, n_ag:]
    ag, sr = np.nonzero(use)
    adjust = np.zeros(len(ag))
    if arrays.get("avidity_adjust") is not None:
        logged = np.log2(arrays["avidity_adjust"])
        adjust = logged[ag] + logged[n_ag + sr]
    distance = np.maximum(cb[sr] - value[ag, sr] - adjust, 0.0)
    weights = arrays.get("weights")
    weight = np.ones(len(ag)) if weights is None else weights[ag, sr]
    return StressTerms(ag, n_ag + sr, distance, kind[ag, sr] == 2, weight, n_ag + n_sr)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    # numerically stable logistic: never exponentiates a large positive number
    e = np.exp(-np.abs(x))
    return np.where(x >= 0, 1.0 / (1.0 + e), e / (1.0 + e))


def stress_and_gradient(flat: np.ndarray, t: StressTerms, dim: int) -> tuple[float, np.ndarray]:
    x = flat.reshape(t.n_points, dim)
    delta = x[t.p1] - x[t.p2]
    map_d = np.sqrt((delta**2).sum(axis=1))
    diff = t.distance - map_d
    lt = t.less_than
    diff_lt = diff[lt] + 1.0
    s_lt = _sigmoid(diff_lt * SIGMOID_MULTIPLIER)
    stress = float(
        (t.weight[~lt] * diff[~lt] ** 2).sum() + (t.weight[lt] * diff_lt**2 * s_lt).sum()
    )
    safe = np.where(map_d == 0.0, 1e-5, map_d)  # ae non_zero()
    inc = np.empty_like(map_d)
    inc[~lt] = t.weight[~lt] * diff[~lt] * 2.0 / safe[~lt]
    inc[lt] = (
        t.weight[lt]
        * (diff_lt * 2.0 * s_lt + diff_lt**2 * s_lt * (1.0 - s_lt) * SIGMOID_MULTIPLIER)
        / safe[lt]
    )
    g = inc[:, None] * delta
    grad = np.zeros_like(x)
    np.add.at(grad, t.p1, -g)
    np.add.at(grad, t.p2, g)
    return stress, grad.reshape(-1)


def stress(layout: np.ndarray, arrays: dict) -> float:
    t = stress_terms(arrays)
    x = np.nan_to_num(layout, nan=0.0)
    return stress_and_gradient(x.reshape(-1), t, layout.shape[1])[0]


def optimise(
    arrays: dict,
    n_starts: int,
    dim: int = 2,
    seed: int = 0,
    start_layout: np.ndarray | None = None,
    keep: int = 10,
) -> list[dict]:
    """Run `n_starts` minimisations; projections sorted by stress (I1 output shape).

    Start i draws from its own generator seeded (seed, i), so a start's result does not
    depend on how starts are split across jobs or threads.
    """
    t = stress_terms(arrays)
    disconnected = arrays["disconnected"]
    unmovable = arrays.get("unmovable")
    span = max(float(t.distance.max(initial=1.0)), 1.0)
    results = []
    for i in range(n_starts):
        rng = np.random.default_rng([seed, i])
        if start_layout is not None:
            x0 = start_layout.copy()
            missing = np.isnan(x0).any(axis=1)
            x0[missing] = _random_near(
                rng, x0[~missing & ~disconnected], int(missing.sum()), dim, span
            )
        else:
            x0 = rng.uniform(-span, span, size=(t.n_points, dim))
        x0[disconnected] = 0.0
        fixed = disconnected.copy() if unmovable is None else (disconnected | unmovable)

        x, value, iterations = _minimise(_objective(t, dim, fixed), x0.reshape(-1))
        layout = x.reshape(-1, dim)
        layout[disconnected] = np.nan
        results.append(
            {
                "layout": layout,
                "stress": value,
                "dimensions": dim,
                "n_iterations": iterations,
                "start_seed": i,
            }
        )
    results.sort(key=lambda r: float(r["stress"]))  # type: ignore[arg-type]
    return results[:keep]


def _objective(t: StressTerms, dim: int, fixed: np.ndarray):
    """Stress and gradient with fixed (disconnected, unmovable) points held still."""

    def fun(flat: np.ndarray) -> tuple[float, np.ndarray]:
        s, g = stress_and_gradient(flat, t, dim)
        g = g.reshape(-1, dim)
        g[fixed] = 0.0
        return s, g.reshape(-1)

    return fun


def _minimise(
    fun, x: np.ndarray, max_iter: int = 5000, gtol: float = 1e-6
) -> tuple[np.ndarray, float, int]:
    """Gradient descent with Barzilai-Borwein step lengths and a backtracking safeguard."""
    f, g = fun(x)
    step = 1e-3
    for it in range(1, max_iter + 1):
        if np.linalg.norm(g) < gtol:
            return x, float(f), it
        while True:
            x_new = x - step * g
            f_new, g_new = fun(x_new)
            if f_new <= f or step < 1e-12:
                break
            step *= 0.5
        s_vec, y_vec = x_new - x, g_new - g
        sy = float(s_vec @ y_vec)
        step = float(s_vec @ s_vec) / sy if sy > 0 else step * 2.0
        if abs(f - f_new) <= 1e-12 * max(1.0, abs(f)):
            return x_new, float(f_new), it
        x, f, g = x_new, f_new, g_new
    return x, float(f), max_iter


def _random_near(
    rng: np.random.Generator, placed: np.ndarray, n: int, dim: int, span: float
) -> np.ndarray:
    """New points of an incremental relax: uniform over the placed points' bounding box, widened."""
    if len(placed) == 0:
        return rng.uniform(-span, span, size=(n, dim))
    lo, hi = placed.min(axis=0), placed.max(axis=0)
    pad = 0.5 * (hi - lo) + 1.0
    return rng.uniform(lo - pad, hi + pad, size=(n, dim))


# ----------------------------------------------------------------------
# grid test (simplified ae `grid-test.cc`: no rough re-optimisation after the move)

HEMISPHERING_DISTANCE = 1.0
HEMISPHERING_STRESS = 0.25


def _contribution(
    pos: np.ndarray, oc: np.ndarray, dist: np.ndarray, lt: np.ndarray, w: np.ndarray
) -> np.ndarray:
    """One point's stress contribution at each candidate position (rows of `pos`)."""
    md = np.sqrt(((pos[:, None, :] - oc[None, :, :]) ** 2).sum(axis=2))
    diff = dist[None, :] - md
    d_lt = diff + 1.0
    c = np.where(lt[None, :], d_lt**2 * _sigmoid(d_lt * SIGMOID_MULTIPLIER), diff**2)
    return (w[None, :] * c).sum(axis=1)


def grid_test(layout: np.ndarray, arrays: dict, step: float = 0.25) -> list[dict]:
    """Points whose own stress contribution is lower elsewhere on a grid.

    trapped: a grid cell lowers the point's contribution by more than 0.25;
    hemisphering: no better cell, but one more than 1 unit away is within 0.5 of it
    (ae's rough threshold, 2 x 0.25). ae then re-optimises and re-measures; the core
    (workstream 6) will; this stub reports the grid result only.
    """
    t = stress_terms(arrays)
    x = np.nan_to_num(layout, nan=0.0)
    out = []
    partners: dict[int, list[int]] = {}
    for k, (a, b) in enumerate(zip(t.p1, t.p2, strict=True)):
        partners.setdefault(int(a), []).append(k)
        partners.setdefault(int(b), []).append(k)
    for point, partner_terms in partners.items():
        ks = np.array(partner_terms)
        other = np.where(t.p1[ks] == point, t.p2[ks], t.p1[ks])
        oc, dist, lt, w = x[other], t.distance[ks], t.less_than[ks], t.weight[ks]
        lo = (oc - dist[:, None]).min(axis=0)
        hi = (oc + dist[:, None]).max(axis=0)
        gx = np.arange(lo[0], hi[0] + step, step)
        gy = np.arange(lo[1], hi[1] + step, step)
        grid = np.stack(np.meshgrid(gx, gy), axis=-1).reshape(-1, 2)

        current = float(_contribution(x[point][None, :], oc, dist, lt, w)[0])
        chunks = [grid[i : i + 4096] for i in range(0, len(grid), 4096)]
        values = np.concatenate([_contribution(c, oc, dist, lt, w) for c in chunks])
        best = int(values.argmin())
        moved = float(np.linalg.norm(grid[best] - x[point]))
        if values[best] < current - HEMISPHERING_STRESS:
            out.append(
                {
                    "point": point,
                    "diagnosis": "trapped",
                    "distance": moved,
                    "contribution_diff": float(values[best] - current),
                }
            )
            continue
        far = np.linalg.norm(grid - x[point], axis=1) > HEMISPHERING_DISTANCE
        if far.any() and values[far].min() < current + 2 * HEMISPHERING_STRESS:
            j = np.nonzero(far)[0][values[far].argmin()]
            out.append(
                {
                    "point": point,
                    "diagnosis": "hemisphering",
                    "distance": float(np.linalg.norm(grid[j] - x[point])),
                    "contribution_diff": float(values[j] - current),
                }
            )
    return out
