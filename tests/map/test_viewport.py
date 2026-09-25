"""Automatic framing on synthetic layouts."""

import numpy as np
import pytest

from af.map.viewport import Box, FrameError, Priority, choose_frame, hidden_mask

LEGEND = Box("legend", 0.0, 0.6, 0.3, 1.0)  # bottom-left, like today's report maps


def test_recent_points_kept_clear_of_the_legend() -> None:
    rng = np.random.default_rng(0)
    old = rng.normal(size=(200, 2)) * 2.0
    recent = rng.normal(size=(40, 2)) * 1.0 + [3.0, 3.0]
    xy = np.vstack([old, recent])
    is_recent = np.r_[np.zeros(200, bool), np.ones(40, bool)]
    choice = choose_frame(
        xy,
        (Priority("recent", is_recent), Priority("all", np.ones(240, bool))),
        size=14.0,
        furniture=(LEGEND,),
    )
    assert choice.hidden["recent"] == 0
    page = choice.frame.page(xy)
    assert not hidden_mask(page[is_recent], (LEGEND,)).any()


def test_frame_error_when_required_points_do_not_fit() -> None:
    xy = np.array([[0.0, 0.0], [20.0, 0.0], [10.0, 5.0]])
    with pytest.raises(FrameError, match="cannot be shown"):
        choose_frame(xy, (Priority("recent", np.ones(3, bool)),), size=10.0, furniture=())


def test_points_without_coordinates_are_ignored() -> None:
    xy = np.array([[0.0, 0.0], [np.nan, np.nan], [3.0, 3.0]])
    choice = choose_frame(xy, (Priority("all", np.ones(3, bool)),), size=6.0, furniture=())
    assert choice.hidden["all"] == 0


def test_lower_priorities_only_break_ties() -> None:
    # Recent points force the frame to the right; an old point on the left cannot be saved.
    xy = np.array([[0.0, 0.0], [10.0, 0.0], [14.0, 0.0], [10.0, 4.0]])
    recent = np.array([False, True, True, True])
    choice = choose_frame(
        xy,
        (Priority("recent", recent), Priority("all", np.ones(4, bool))),
        size=6.0,
        furniture=(),
    )
    assert choice.hidden == {"recent": 0, "all": 1}


def test_mask_length_checked() -> None:
    with pytest.raises(ValueError, match="mask length"):
        choose_frame(np.zeros((3, 2)), (Priority("x", np.ones(2, bool)),), size=1.0, furniture=())
