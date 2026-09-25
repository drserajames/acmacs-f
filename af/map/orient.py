"""Orient a map: Procrustes to a reference map, plus named rotation overrides.

Why: today each lab folder is rotated by hand (baked ``chart-rotate`` angles, ``orient.py``,
``align_to_previous.py``), and the angles have to be re-derived every round. Measured on the
Sep 2026 round, a Procrustes fit (rotation, optional reflection, translation, no scaling) of each
map to the *same lab's previous-round map* reproduces the hand orientation to within 0.5 degrees
in 15 of 18 folders. The other three were deliberate one-off rotations, which is what
:class:`RotationOverride` records (Sarah, 25 Sep 2026: previous-map default, an optional
reference-lab rule behind a quality gate, and named overrides).

Conventions (the same as ``.ace`` projections): points are rows, ``displayed = raw @ M + t``,
the y axis grows downward on the page, and an angle is measured in the displayed frame,
positive from +x towards +y.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


class OrientationError(ValueError):
    """The reference fit is not trustworthy enough to use (too few common points, ambiguous)."""


@dataclass(frozen=True)
class Fit:
    """A rigid fit ``src @ matrix + translation ~= dst`` over ``n_common`` points."""

    matrix: Array
    translation: Array
    rmsd: float
    n_common: int

    @property
    def degrees(self) -> float:
        return decompose(self.matrix)[0]

    @property
    def reflected(self) -> bool:
        return decompose(self.matrix)[1]


@dataclass(frozen=True)
class RotationOverride:
    """A hand rotation applied after the automatic fit, kept as named data (design rule 9).

    ``degrees`` rotates the displayed map; ``reflect`` mirrors it first (x -> -x). ``reason`` and
    ``decided`` say who wanted it and when, so the report can list every override in use.
    """

    name: str
    degrees: float
    reason: str
    decided: str
    reflect: bool = False


def rotation(degrees: float, reflect: bool = False) -> Array:
    """Row-vector matrix that (optionally) mirrors x, then rotates by ``degrees``."""
    a = math.radians(degrees)
    r = np.array([[math.cos(a), math.sin(a)], [-math.sin(a), math.cos(a)]])
    return (np.diag([-1.0, 1.0]) @ r) if reflect else r


def decompose(matrix: Array) -> tuple[float, bool]:
    """Split an orthogonal 2x2 row-vector matrix into (degrees, reflected).

    Inverse of :func:`rotation`: ``rotation(*decompose(m))`` equals ``m``.
    """
    reflected = bool(np.linalg.det(matrix) < 0)
    r = (np.diag([-1.0, 1.0]) @ matrix) if reflected else matrix
    return math.degrees(math.atan2(r[0, 1], r[0, 0])), reflected


def procrustes(src: Array, dst: Array, *, reflection: bool = True, translation: bool = True) -> Fit:
    """Least-squares rigid fit of ``src`` onto ``dst`` (rows paired). No scaling: map units are
    antigenic units and must keep their meaning. Rows with NaN on either side are ignored."""
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 2:
        raise ValueError(f"procrustes needs two (n, 2) arrays, got {src.shape} and {dst.shape}")
    ok = ~(np.isnan(src).any(axis=1) | np.isnan(dst).any(axis=1))
    x, y = src[ok], dst[ok]
    if len(x) < 3:
        raise OrientationError(f"procrustes needs at least 3 common points, got {len(x)}")
    mx = x.mean(axis=0) if translation else np.zeros(2)
    my = y.mean(axis=0) if translation else np.zeros(2)
    u, _, vt = np.linalg.svd((x - mx).T @ (y - my))
    m = u @ vt
    if not reflection and np.linalg.det(m) < 0:
        u[:, -1] *= -1
        m = u @ vt
    t = my - mx @ m
    rmsd = float(np.sqrt(((x @ m + t - y) ** 2).sum(axis=1).mean()))
    return Fit(matrix=m, translation=t, rmsd=rmsd, n_common=len(x))


@dataclass(frozen=True)
class Orientation:
    """Result of :func:`orient`: the matrix to store on the projection and what it came from."""

    matrix: Array
    translation: Array
    fit: Fit
    reference: str
    overrides: tuple[RotationOverride, ...]

    def report(self) -> dict[str, object]:
        """Plain-data summary for the I7 JSON ``map.orientation`` block and the review page."""
        degrees, reflected = decompose(self.matrix)
        return {
            "reference": self.reference,
            "common_points": self.fit.n_common,
            "rmsd": round(self.fit.rmsd, 4),
            "fit_degrees": round(self.fit.degrees, 3),
            "fit_reflected": self.fit.reflected,
            "overrides": [
                {"name": o.name, "degrees": o.degrees, "reflect": o.reflect, "reason": o.reason}
                for o in self.overrides
            ],
            "degrees": round(degrees, 3),
            "reflected": reflected,
        }


def drawn_pairs(pairs: NDArray[np.intp], reference_shown: NDArray[np.bool_]) -> NDArray[np.intp]:
    """Keep only pairs whose reference point was actually DRAWN on the reference map.

    Orientation is presentation, so it should follow what a reader can compare. A point the
    reference map hid still has coordinates, and including it turns the visible part of the new
    map against the old one. Measured on the Sep 2026 round: the only chart with hidden points in
    its previous map (242 of 2636) fits 7.3 degrees differently, and the drawn-only fit is the
    better one (RMSD 1.04 against 1.55). Every other chart is unaffected.
    """
    if reference_shown.ndim != 1:
        raise ValueError("reference_shown must be a 1-D mask over the reference's points")
    return pairs[reference_shown[pairs[:, 1]]]


def orient(
    raw: Array,
    reference_displayed: Array,
    pairs: NDArray[np.intp],
    *,
    reference: str,
    min_common: int,
    min_reflection_margin: float = 0.0,
    overrides: tuple[RotationOverride, ...] = (),
) -> Orientation:
    """Orient ``raw`` (this map's layout) to ``reference_displayed`` (the reference map as drawn).

    ``pairs`` holds (row in raw, row in reference) for points that are the same virus or serum;
    matching is the caller's job (chart identity keys). Pass them through :func:`drawn_pairs`
    first: a point the reference map did not draw should not decide how this map is turned.

    Fails, rather than guessing, when fewer
    than ``min_common`` pairs have coordinates, or when the best fit with a reflection and the
    best without differ in RMSD by less than ``min_reflection_margin``: cross-lab references
    measured on the Sep 2026 round flip chirality with small changes to the matching, and a
    silently mirrored map is worse than an error.
    """
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError("pairs must be an (n, 2) index array")
    src, dst = raw[pairs[:, 0]], reference_displayed[pairs[:, 1]]
    fit = procrustes(src, dst)
    if fit.n_common < min_common:
        raise OrientationError(
            f"orientation to {reference}: {fit.n_common} common points with coordinates, "
            f"need at least {min_common}"
        )
    if min_reflection_margin > 0:
        other = procrustes(src, dst, reflection=False) if fit.reflected else _mirrored_fit(src, dst)
        if abs(other.rmsd - fit.rmsd) < min_reflection_margin:
            raise OrientationError(
                f"orientation to {reference}: reflected and unreflected fits are too close "
                f"(RMSD {fit.rmsd:.3f} vs {other.rmsd:.3f}, margin {min_reflection_margin})"
            )
    matrix, t = fit.matrix, fit.translation
    for override in overrides:
        extra = rotation(override.degrees, override.reflect)
        matrix, t = matrix @ extra, t @ extra
    return Orientation(matrix, t, fit, reference, tuple(overrides))


def _mirrored_fit(src: Array, dst: Array) -> Fit:
    """Best fit forced to include a reflection (for the chirality margin test)."""
    flip = np.diag([-1.0, 1.0])
    f = procrustes(src @ flip, dst, reflection=False)
    return Fit(flip @ f.matrix, f.translation, f.rmsd, f.n_common)
