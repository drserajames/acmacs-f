"""Named moves and continuity layouts, with stand-in relaxers (the optimiser has its own tests)."""

import numpy as np
import pytest

from af.map.curate import (
    BlockOffset,
    CurationError,
    MoveOverride,
    apply_block_offset,
    apply_move,
    continuity_layout,
)
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
    with pytest.raises(CurationError, match="matches 0") as none:
        apply_move(
            rule(movers=("GHOST",)), LAYOUT, NAMES, PAINTED, stress_before=0.0, relax=no_relax
        )
    assert none.value.fields() == {
        "guard": "mover_match",
        "measured": 0,
        "limit": 1,
        "bound": "exact",
    }
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
    with pytest.raises(CurationError, match="need at least 3") as small:
        apply_move(
            rule(target_colour=RED), LAYOUT, NAMES, PAINTED, stress_before=0.0, relax=no_relax
        )
    assert small.value.guard == "target_points" and small.value.bound == "min"
    assert small.value.limit == 3 and small.value.measured < 3


def test_guards() -> None:
    with pytest.raises(CurationError, match="stress rose"):
        apply_move(
            rule(max_stress_rise=0.0), LAYOUT, NAMES, PAINTED, stress_before=0.0, relax=no_relax
        )

    def springs_back(start: np.ndarray, movable: np.ndarray) -> tuple[np.ndarray, float]:
        return LAYOUT.copy(), 0.0

    with pytest.raises(CurationError, match="relaxed back") as back:
        apply_move(rule(), LAYOUT, NAMES, PAINTED, stress_before=0.0, relax=springs_back)
    # The refusal as data, so a report states it without parsing the message.
    fields = back.value.fields()
    assert fields["guard"] == "from_target" and fields["bound"] == "max"
    assert fields["limit"] == rule().max_from_target
    assert isinstance(fields["measured"], float) and fields["measured"] > fields["limit"]


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


def block_rule(**changes: object) -> BlockOffset:
    base: dict[str, object] = dict(
        name="shift the stray cluster in",
        reason="reviewed at the meeting",
        decided="2026-09-23",
        movers=("STRAY A", "STRAY B"),
        shift=(-9.0, -9.0),
        derived_from="median of the blue group minus the mean of the strays, 22 Sep layout",
        colour_scheme="scheme-1",
        target_colour=BLUE,
        max_stress_rise=1e9,
        settled_within=1.0,
        min_settled=2,
        max_other_move=1e9,
    )
    base.update(changes)
    return BlockOffset(**base)  # type: ignore[arg-type]


def test_block_offset_shifts_the_group_and_keeps_its_shape() -> None:
    result = apply_block_offset(
        block_rule(), LAYOUT, NAMES, PAINTED, stress_before=0.0, relax=no_relax
    )
    # the two movers keep their separation: a block move, not a move to a point
    before = LAYOUT[4] - LAYOUT[3]
    after = result.layout[4] - result.layout[3]
    np.testing.assert_allclose(after, before)
    np.testing.assert_allclose(result.layout[3], LAYOUT[3] + [-9.0, -9.0])
    assert result.settled == 2
    report = result.report(block_rule())
    assert report["shift"] == [-9.0, -9.0]
    assert "derived_from" in report  # the number is round-bound; say where it came from
    assert all(type(v) in (str, int, float, list) for v in report.values())


def test_block_offset_guards() -> None:
    with pytest.raises(CurationError, match="stress rose"):
        apply_block_offset(
            block_rule(max_stress_rise=0.0),
            LAYOUT,
            NAMES,
            PAINTED,
            stress_before=-1000.0,
            relax=no_relax,
        )
    with pytest.raises(CurationError, match="settled within"):
        apply_block_offset(
            block_rule(shift=(0.0, 0.0)),
            LAYOUT,
            NAMES,
            PAINTED,
            stress_before=0.0,
            relax=no_relax,
        )
    with pytest.raises(CurationError, match="matches 0"):
        apply_block_offset(
            block_rule(movers=("GHOST",)),
            LAYOUT,
            NAMES,
            PAINTED,
            stress_before=0.0,
            relax=no_relax,
        )


def test_block_offset_ignores_a_rotation_when_measuring_collateral() -> None:
    """A relax can turn the whole map; unfitted, that reads as every point having moved."""

    def turns_the_map(start: np.ndarray, movable: np.ndarray) -> tuple[np.ndarray, float]:
        return start @ rotation(25.0), 0.0

    result = apply_block_offset(
        block_rule(max_other_move=1e-6, settled_within=1e9),
        LAYOUT,
        NAMES,
        PAINTED,
        stress_before=0.0,
        relax=turns_the_map,
    )
    assert result.largest_other_move < 1e-6  # the rotation is fitted out, not counted as movement
    assert abs(result.rotation_degrees) == pytest.approx(25.0, abs=0.01)


def target_rule(**changes: object) -> BlockOffset:
    return block_rule(shift=None, derived_from=None, to="target-median", **changes)


def test_a_target_block_lands_its_median_on_the_target_median() -> None:
    """Sarah, 7 Oct 2026: the shift comes from the layout, so the block lands on its target."""
    result = apply_block_offset(
        target_rule(), LAYOUT, NAMES, PAINTED, stress_before=0.0, relax=no_relax
    )
    # target = median of IN 1-3 = (1, 0); movers' median = (9.5, 9.5)
    np.testing.assert_allclose(result.shift, (-8.5, -9.5))
    np.testing.assert_allclose(np.median(result.layout[[3, 4]], axis=0), (1.0, 0.0))
    np.testing.assert_allclose(result.layout[4] - result.layout[3], LAYOUT[4] - LAYOUT[3])
    report = result.report(target_rule())
    assert report["shift"] == [-8.5, -9.5] and report["to"] == "target-median"
    assert "derived_from" not in report  # nothing round-bound to explain


def test_a_target_block_survives_a_re_layout_a_fixed_offset_does_not() -> None:
    """The same map turned and moved (a from-scratch re-layout): the fixed offset, derived on
    the old layout, sends the block elsewhere; the target still lands it."""
    moved = LAYOUT @ rotation(120.0) + np.array([30.0, -4.0])
    fixed = block_rule(shift=(-8.5, -9.5), derived_from="the old layout")
    with pytest.raises(CurationError, match="settled within"):
        apply_block_offset(fixed, moved, NAMES, PAINTED, stress_before=0.0, relax=no_relax)
    result = apply_block_offset(
        target_rule(), moved, NAMES, PAINTED, stress_before=0.0, relax=no_relax
    )
    assert result.settled == 2


def test_a_block_is_a_fixed_shift_or_a_target_never_both() -> None:
    with pytest.raises(ValueError, match="computes the shift"):
        block_rule(to="target-median")  # still has shift and derived_from
    with pytest.raises(ValueError, match="needs shift and derived_from"):
        block_rule(shift=None)
    with pytest.raises(ValueError, match="to must be"):
        block_rule(shift=None, derived_from=None, to="somewhere")
