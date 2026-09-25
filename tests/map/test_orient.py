"""Orientation: Procrustes to a reference plus named overrides, on synthetic layouts."""

import numpy as np
import pytest

from af.map.orient import (
    OrientationError,
    RotationOverride,
    decompose,
    orient,
    procrustes,
    rotation,
)


def cloud(n: int = 60, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    # anisotropic so the fit is well determined
    return rng.normal(size=(n, 2)) * [4.0, 1.5]


def pairs(n: int) -> np.ndarray:
    return np.stack([np.arange(n), np.arange(n)], axis=1)


@pytest.mark.parametrize("degrees", [0.0, 37.5, -120.0, 179.0])
@pytest.mark.parametrize("reflect", [False, True])
def test_rotation_decompose_round_trip(degrees: float, reflect: bool) -> None:
    got_degrees, got_reflect = decompose(rotation(degrees, reflect))
    assert got_reflect is reflect
    assert got_degrees == pytest.approx(degrees)


@pytest.mark.parametrize("reflect", [False, True])
def test_procrustes_recovers_rigid_motion(reflect: bool) -> None:
    raw = cloud()
    m = rotation(-73.0, reflect)
    ref = raw @ m + [5.0, -2.0]
    fit = procrustes(raw, ref)
    assert fit.rmsd == pytest.approx(0.0, abs=1e-9)
    assert fit.reflected is reflect
    assert fit.degrees == pytest.approx(-73.0)
    np.testing.assert_allclose(fit.translation, [5.0, -2.0], atol=1e-9)


def test_procrustes_never_scales() -> None:
    raw = cloud()
    fit = procrustes(raw, raw * 2.0)
    assert np.linalg.det(fit.matrix) == pytest.approx(1.0)
    assert fit.rmsd > 0.5


def test_procrustes_ignores_rows_without_coordinates() -> None:
    raw = cloud()
    ref = raw @ rotation(20.0)
    raw[3] = np.nan
    ref[7] = np.nan
    fit = procrustes(raw, ref)
    assert fit.n_common == len(raw) - 2
    assert fit.degrees == pytest.approx(20.0)


def test_orient_reports_and_applies_override() -> None:
    raw = cloud()
    ref = raw @ rotation(30.0)
    override = RotationOverride("line up with another lab", -8.0, "reviewer asked", "2026-09-18")
    result = orient(
        raw, ref, pairs(len(raw)), reference="prev", min_common=10, overrides=(override,)
    )
    assert decompose(result.matrix)[0] == pytest.approx(22.0)
    report = result.report()
    assert report["fit_degrees"] == pytest.approx(30.0)
    assert report["degrees"] == pytest.approx(22.0)
    assert report["common_points"] == len(raw)
    overrides = report["overrides"]
    assert isinstance(overrides, list)
    assert [o["name"] for o in overrides] == ["line up with another lab"]


def test_orient_refuses_too_few_common_points() -> None:
    raw = cloud()
    with pytest.raises(OrientationError, match="need at least 100"):
        orient(raw, raw, pairs(len(raw)), reference="other lab", min_common=100)


def test_orient_refuses_ambiguous_chirality() -> None:
    # Nearly collinear points are almost mirror-symmetric about their own axis, so the mirrored
    # and unmirrored fits are about as good: chirality is not determined.
    rng = np.random.default_rng(3)
    raw = np.column_stack([np.linspace(-5, 5, 40), rng.normal(scale=0.05, size=40)])
    ref = raw @ rotation(10.0) + rng.normal(scale=0.3, size=raw.shape)
    with pytest.raises(OrientationError, match="too close"):
        orient(raw, ref, pairs(40), reference="other lab", min_common=10, min_reflection_margin=0.5)
    # a clear-cut fit passes the same gate
    clean = cloud()
    orient(
        clean,
        clean @ rotation(10.0),
        pairs(len(clean)),
        reference="prev",
        min_common=10,
        min_reflection_margin=0.5,
    )
