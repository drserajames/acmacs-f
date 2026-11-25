"""Where a geo page's dots go (Q280): each location's pile, or one lattice shared by the page.

All places, counts and colours are invented.
"""

import json
import math
from pathlib import Path
from typing import Any

import pytest

from af.geo.figures import write_geo_figures
from af.geo.placement import (
    LATTICE,
    NEAREST_OFFER_FIRST,
    PILES,
    Dot,
    Placement,
    PlacementError,
    cell_centre,
    displacement,
    lattice_positions,
)
from af.geo.render import Look, render_period

from .test_render import _coastline

PITCH = 6.4
RULE = Placement(LATTICE, pitch=PITCH, order=NEAREST_OFFER_FIRST, reserve_point=True)


def _everywhere(x: float, y: float) -> bool:
    return True


def _crowd(*places: tuple[str, int, float, float]) -> list[Dot]:
    """Dots piled exactly on their place: (name, dots, x, y) in figure points."""
    return [Dot(name, x, y, x, y) for name, n, x, y in places for _ in range(n)]


def _period(*places: tuple[str, int]) -> dict[str, Any]:
    return {
        "period": "2021-01",
        "title": "January 2021",
        "locations": [
            {"name": name, "points": [{"color": "#aa0000", "count": n, "clade": "Clade R"}]}
            for name, n in places
        ],
    }


def test_no_two_lattice_dots_are_nearer_than_the_pitch() -> None:
    dots = _crowd(("ALPHA", 30, 0.0, 0.0), ("BETA", 20, 1.0, 0.5), ("GAMMA", 7, -2.0, 1.0))
    positions, kept = lattice_positions(
        dots, PITCH, reserve_point=True, max_move=None, inside=_everywhere
    )
    assert kept == 0 and len(positions) == len(dots)
    gaps = [math.dist(a, b) for i, a in enumerate(positions) for b in positions[i + 1 :]]
    assert min(gaps) >= PITCH - 1e-9


def test_reserving_a_cell_keeps_a_dot_on_every_place() -> None:
    """Three crowded places: handed out nearest offer first, one loses every cell near its
    point to its neighbours' piles; with a cell reserved per place first, none does. (A layout
    found by search; most layouts need no reserve, which is why it is cheap.)"""
    dots = _crowd(("ALPHA", 7, -0.7, -6.1), ("BETA", 6, -5.7, -5.6), ("GAMMA", 3, 5.0, -2.1))
    off = {}
    for reserve in (True, False):
        positions, _ = lattice_positions(
            dots, PITCH, reserve_point=reserve, max_move=None, inside=_everywhere
        )
        moved = [
            Dot(d.location, x, y, d.hx, d.hy) for d, (x, y) in zip(dots, positions, strict=True)
        ]
        cost = displacement(moved, dict.fromkeys(("ALPHA", "BETA", "GAMMA"), 0.0), 0.45, 0.45, 4.0)
        off[reserve] = cost["locations_off_point"]
    assert off == {True: 0, False: 1}


def test_the_lattice_is_the_same_every_run() -> None:
    dots = _crowd(("ALPHA", 12, 0.0, 0.0), ("BETA", 12, 2.0, 2.0))
    runs = {
        tuple(lattice_positions(dots, PITCH, reserve_point=True, max_move=None,
                                inside=_everywhere)[0])
        for _ in range(3)
    }  # fmt: skip
    assert len(runs) == 1


def test_a_cap_keeps_dots_that_would_move_too_far_in_their_pile() -> None:
    dots = _crowd(("ALPHA", 60, 0.0, 0.0))
    positions, kept = lattice_positions(
        dots, PITCH, reserve_point=True, max_move=2 * PITCH, inside=_everywhere
    )
    assert kept > 0
    for dot, at in zip(dots, positions, strict=True):
        moved = math.dist(at, (dot.hx, dot.hy))
        assert moved <= 2 * PITCH + 1e-9 or at == (dot.x, dot.y)


def test_no_cell_the_frame_would_cut_is_handed_out() -> None:
    def inside(x: float, y: float) -> bool:
        return x <= 0.0  # a frame just right of the place

    dots = _crowd(("EDGE", 10, 0.0, 0.0))
    positions, _ = lattice_positions(dots, PITCH, reserve_point=True, max_move=None, inside=inside)
    assert all(x <= 0.0 for x, _ in positions)


def test_a_bad_pitch_or_no_room_is_an_error() -> None:
    dots = _crowd(("ALPHA", 3, 0.0, 0.0))
    with pytest.raises(PlacementError, match="pitch"):
        lattice_positions(dots, 0.0, reserve_point=True, max_move=None, inside=_everywhere)
    with pytest.raises(PlacementError, match="no free lattice cell"):
        lattice_positions(
            dots, PITCH, reserve_point=True, max_move=None, inside=lambda x, y: (x, y) == (0, 0)
        )


def test_cells_are_hex_packed() -> None:
    assert cell_centre(1, 0, 2.0) == (2.0, 0.0)
    assert cell_centre(0, 1, 2.0) == pytest.approx((1.0, math.sqrt(3)))


def test_displacement_counts_dots_drawn_nearer_another_place() -> None:
    dots = [Dot("ALPHA", 9.0, 0.0, 0.0, 0.0), Dot("BETA", 10.0, 0.0, 10.0, 0.0)]
    cost = displacement(dots, {"ALPHA": 0.0, "BETA": 0.0}, 0.45, 0.45, 4.0)
    assert cost["dots_in_another_place"] == 1
    assert cost["locations_off_point"] == 1  # ALPHA's only dot is 9 pt off, past its radius
    assert cost["move_km"]["max"] == pytest.approx(9 * 0.45 * 111.195, rel=1e-3)
    assert displacement([], {}, 0.45, 0.45, 4.0)["locations"] == 0


def test_a_crowded_page_drawn_on_the_lattice_hides_nothing(tmp_path: Path) -> None:
    # three places a pitch or more apart (3 degrees is 6.7 points): their piles overlap, but
    # each has a cell of its own on its point
    near = {"ALPHA": (10.0, 20.0), "BETA": (13.0, 20.0), "GAMMA": (10.0, 23.0)}
    period = _period(("ALPHA", 25), ("BETA", 25), ("GAMMA", 5))
    piles = render_period(period, near.get, _coastline(tmp_path), tmp_path / "piles.pdf")
    lattice = render_period(
        period, near.get, _coastline(tmp_path), tmp_path / "lattice.pdf", placement=RULE
    )
    assert piles.hiding.mostly_covered > 0 and piles.placement is None
    assert lattice.dots == piles.dots == 55
    assert lattice.hiding.half_covered == 0 and lattice.hiding.invisible_locations == 0
    record = lattice.placement
    assert record is not None
    assert (record["rule"], record["pitch"], record["max_move"]) == (LATTICE, PITCH, None)
    assert record["unreadable"] == 0 and record["locations"] == 3
    assert record["locations_off_point"] == 0  # one cell reserved on each place first


def test_piles_named_draw_the_same_page_and_record_what_they_cost(tmp_path: Path) -> None:
    near = {"ALPHA": (10.0, 20.0), "BETA": (10.5, 20.2)}
    period = _period(("ALPHA", 9), ("BETA", 9))
    coast = _coastline(tmp_path)
    unnamed = render_period(period, near.get, coast, tmp_path / "a.pdf")
    named = render_period(period, near.get, coast, tmp_path / "b.pdf", placement=Placement(PILES))
    assert (tmp_path / "a.pdf").read_bytes() == (tmp_path / "b.pdf").read_bytes()
    assert named.hiding == unnamed.hiding
    assert named.placement is not None and named.placement["rule"] == PILES
    assert named.placement["unreadable"] == unnamed.hiding.mostly_covered


def test_a_lattice_page_with_no_dots_is_drawn(tmp_path: Path) -> None:
    got = render_period(
        _period(), lambda _: None, _coastline(tmp_path), tmp_path / "e.pdf", placement=RULE
    )
    assert got.dots == 0 and got.placement is not None and got.placement["locations"] == 0


def test_the_figure_records_its_placement_only_when_one_is_given(tmp_path: Path) -> None:
    doc = {"subtype": "A(H3N2)", "periods": [_period(("ALPHA", 4))], "undated": 0,
           "no_location": []}  # fmt: skip
    coast = _coastline(tmp_path)
    for placement, folder in ((None, "plain"), (RULE, "lattice")):
        write_geo_figures(doc, "h3", lambda _: (1.0, 2.0), coast, tmp_path / folder, {},
                          look=Look(), placement=placement)  # fmt: skip
        i7 = json.loads(
            (tmp_path / folder / "geo" / "h3" / "2021-01" / "af" / "figure.i7.json").read_text()
        )
        assert ("placement" in i7["provenance"]) is (placement is not None)
    assert i7["provenance"]["placement"]["rule"] == LATTICE
