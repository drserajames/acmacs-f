"""Procrustes: fit one layout onto another over common points (rotation/reflection + translation).

No scaling, as ae (`procrustes.cc`): map units are log2 titre units and must not change.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ProcrustesResult:
    rotation: np.ndarray  # dim x dim, applied as x @ rotation
    translation: np.ndarray  # dim
    rmsd: float
    distances: np.ndarray  # per common point, after the fit
    n_common: int

    def apply(self, layout: np.ndarray) -> np.ndarray:
        return layout @ self.rotation + self.translation


def procrustes(
    primary: np.ndarray, secondary: np.ndarray, allow_reflection: bool = True
) -> ProcrustesResult:
    """Fit `secondary` onto `primary`; rows are the same points, NaN rows are skipped."""
    ok = ~(np.isnan(primary).any(axis=1) | np.isnan(secondary).any(axis=1))
    a, b = primary[ok], secondary[ok]
    if len(a) < primary.shape[1] + 1:
        raise ValueError(
            f"procrustes needs at least {primary.shape[1] + 1} common points, got {len(a)}"
        )
    ca, cb = a.mean(axis=0), b.mean(axis=0)
    u, _, vt = np.linalg.svd((b - cb).T @ (a - ca))
    rot = u @ vt
    if not allow_reflection and np.linalg.det(rot) < 0:
        u[:, -1] *= -1
        rot = u @ vt
    trans = ca - cb @ rot
    fitted = b @ rot + trans
    dist = np.sqrt(((fitted - a) ** 2).sum(axis=1))
    full = np.full(len(primary), np.nan)
    full[ok] = dist
    return ProcrustesResult(rot, trans, float(np.sqrt((dist**2).mean())), full, int(ok.sum()))
