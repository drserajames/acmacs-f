"""Named moves and continuity layouts, with stand-in relaxers (the optimiser has its own tests)."""

import numpy as np
import pytest

from af.map.curate import CurationError, MoveOverride, apply_move, continuity_layout
from af.map.orient import rotation

BLUE, RED = "#0000ff", "#ff0000"


def no_relax(start: np.ndarray, movable: np.ndarray) -> tuple[np.ndarray, float]:
    """Stand-in: keeps the start; stress = squared distance of every point from the origin."""
    return start.copy(), float(np.nansum(start**2))


def rule(**changes: object) -> MoveOverride:
    base: dict[str, object] = dict(
        name="pull strays in",
        reason="test",
        decided="2026-01-01",
        movers=("STRAY A", "STRAY B"),
        colour_scheme="scheme-1",
        target_colour=BLUE,
        max_stress_rise=1e9,
        max_from_target=0.5,
        min_target_points=3,
    )
    base.update(changes)
    return MoveOverride(**base)  # type: ignore[arg-type]


LAYOUT = np.array(
    [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [10.0, 10.0], [9.0, 9.0], [5.0, 5.0], [np.nan, np.nan]]
)
NAMES = ["IN 1", "IN 2", "IN 3", "STRAY A", "STRAY B", "OTHER", "NOCOORDS"]
PAINTED = [BLUE, BLUE, BLUE, BLUE, BLUE, RED, BLUE]


def test_movers_go_to_median_of_painted_group() -> None:
    result = apply_move(rule(), LAYOUT, NAMES, PAINTED, stress_before=0.0, relax=no_relax)
    assert result.target == (1.0, 0.0)  # median of IN 1-3; movers and NaN rows excluded
    assert result.target_points == 3
    np.testing.assert_allclose(result.layout[[3, 4]], [[1.0, 0.0], [1.0, 0.0]])
    assert result.largest_other_move == 0.0
    report = result.report(rule())
    assert report["movers"] == 2
    assert all(type(v) in (str, int, float) for v in report.values())  # plain JSON types


def test_mover_must_match_exactly_one() -> None:
    with pytest.raises(CurationError, match="matches 0"):
        apply_move(
            rule(movers=("GHOST",)), LAYOUT, NAMES, PAINTED, stress_before=0.0, relax=no_relax
        )
    with pytest.raises(CurationError, match="matches 2"):
        apply_move(
            rule(movers=("IN 1",)),
            LAYOUT,
            [*NAMES[:1], "IN 1", *NAMES[2:]],
            PAINTED,
            stress_before=0.0,
            relax=no_relax,
        )


def test_target_group_too_small() -> None:
    with pytest.raises(CurationError, match="need at least 3"):
        apply_move(
            rule(target_colour=RED), LAYOUT, NAMES, PAINTED, stress_before=0.0, relax=no_relax
        )


def test_guards() -> None:
    with pytest.raises(CurationError, match="stress rose"):
        apply_move(
            rule(max_stress_rise=0.0), LAYOUT, NAMES, PAINTED, stress_before=0.0, relax=no_relax
        )

    def springs_back(start: np.ndarray, movable: np.ndarray) -> tuple[np.ndarray, float]:
        return LAYOUT.copy(), 0.0

    with pytest.raises(CurationError, match="relaxed back"):
        apply_move(rule(), LAYOUT, NAMES, PAINTED, stress_before=0.0, relax=springs_back)


def test_continuity_starts_from_previous_positions() -> None:
    rng = np.random.default_rng(5)
    previous = rng.normal(size=(30, 2)) * [3.0, 1.0]
    # this round: the old points drifted a little and were rotated; two new points appended
    chain = np.vstack(
        [
            (previous + rng.normal(scale=0.3, size=previous.shape)) @ rotation(70.0),
            [[4.0, 4.0], [-4.0, 4.0]],
        ]
    )
    pairs = np.stack([np.arange(30), np.arange(30)], axis=1)
    result = continuity_layout(chain, previous, pairs, stress_chain=100.0, relax=no_relax)
    np.testing.assert_allclose(result.layout[:30], previous)  # identity relax: old points as drawn
    assert result.rmsd_continuity_to_previous == pytest.approx(0.0, abs=1e-9)
    assert result.rmsd_chain_to_previous == pytest.approx(0.3 * np.sqrt(2), rel=0.35)
    report = result.report()
    assert report["common_points"] == 30
    assert set(report) >= {"stress_delta_pct", "rmsd_continuity_to_chain"}
