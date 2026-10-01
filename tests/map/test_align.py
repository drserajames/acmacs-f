"""Orienting one chart onto another: matching, counts, the fit, and the three ways to match."""

import numpy as np
import pytest

from af.chart.model import Antigen, Chart, Projection, Serum, Titres
from af.map.align import chart_points, match_points, orient


def name(place: str, n: int) -> str:
    return "/".join(("A(H3N2)", place, str(n), "2021"))


def chart(places: list[str], dates: list[str], sera: list[str], layout: np.ndarray | None) -> Chart:
    antigens = [
        Antigen(name(p, i + 1), passage="SIAT1", date=d)
        for i, (p, d) in enumerate(zip(places, dates, strict=True))
    ]
    sr = [Serum(name("OLDTOWN", 90 + j), serum_id=sid) for j, sid in enumerate(sera)]
    c = Chart({"V": "A(H3N2)"}, antigens, sr, Titres([[] for _ in antigens]))
    if layout is not None:
        c.projections.append(Projection(layout=layout))
    return c


RNG = np.random.default_rng(3)
BASE = RNG.normal(size=(6, 2)) * 3
DATES = [f"2021-01-0{i + 1}" for i in range(4)]


def rotated(layout: np.ndarray, degrees: float, shift: tuple[float, float]) -> np.ndarray:
    t = np.radians(degrees)
    r = np.array([[np.cos(t), np.sin(t)], [-np.sin(t), np.cos(t)]])
    return layout @ r + np.array(shift)


def test_identity_matches_across_place_spellings_and_fits_exactly() -> None:
    """One virus spelled two ways (the lab's spelling vs a rewritten one) still matches."""
    a = chart(["NEWTOWN"] * 4, DATES, ["S1", "S2"], rotated(BASE, 30, (2, -1)))
    b = chart(["NEW TOWN"] * 4, DATES, ["S1", "S2"], BASE)
    al = orient(a, b)
    assert al.match.mode == "identity"
    assert al.match.counts["matched_antigen"] == 4 and al.match.counts["matched_serum"] == 2
    assert al.fit.rmsd < 1e-9
    assert np.allclose(al.apply(a.projections[0].layout), BASE)


def test_counts_unmatched_and_duplicates() -> None:
    a = chart(["NEWTOWN"] * 4, DATES, ["S1", "S1"], BASE)
    a.sera[1] = Serum(a.sera[0].name, serum_id="S1")  # one serum twice: same name, same id
    b = chart(["NEWTOWN"] * 3, DATES[:3], ["S1", "S9"], BASE[[0, 1, 2, 4, 5]])
    m = match_points(a, b)
    assert m.counts["matched_antigen"] == 3
    assert m.counts["unmatched_a_antigen"] == 1  # the fourth antigen has no partner
    assert m.counts["duplicate_key_a_serum"] == 2  # dropped, never merged
    assert m.counts["matched_serum"] == 0


def test_works_without_projections_and_refuses_to_guess_a_layout() -> None:
    a = chart(["NEWTOWN"] * 4, DATES, ["S1", "S2"], None)
    b = chart(["NEWTOWN"] * 4, DATES, ["S1", "S2"], BASE)
    assert len(match_points(a, b).pairs) == 6
    with pytest.raises(ValueError, match="chart A has no projection"):
        orient(a, b)
    al = orient(a, b, layout_a=rotated(BASE, -45, (0, 0)))
    assert al.fit.rmsd < 1e-9


def test_a_caller_key_has_no_fallback() -> None:
    """A key returning None matches nothing; the count says so."""
    a = chart(["NEWTOWN"] * 4, DATES, ["S1", "S2"], BASE)
    b = chart(["NEWTOWN"] * 4, DATES, ["S1", "S2"], BASE)
    m = match_points(a, b, key=lambda p: p["name"] if p["kind"] == "antigen" else None)
    assert m.mode == "key"
    assert m.counts["matched_antigen"] == 4 and m.counts["matched_serum"] == 0
    assert m.counts["no_key_a_serum"] == 2


def test_supplied_pairs_replace_matching() -> None:
    a = chart(["NEWTOWN"] * 4, DATES, ["S1", "S2"], rotated(BASE, 90, (1, 1)))
    b = chart(["ELSEWHERE"] * 4, ["2020-01-01"] * 4, ["X", "Y"], BASE)  # nothing would match
    al = orient(a, b, pairs=[(i, i) for i in range(6)])
    assert al.match.mode == "pairs" and al.match.counts["supplied"] == 6
    assert al.fit.rmsd < 1e-9
    with pytest.raises(ValueError, match="outside chart B"):
        orient(a, b, pairs=[(0, 0), (1, 1), (2, 99)])


def test_records_follow_layout_rows() -> None:
    c = chart(["NEWTOWN"] * 4, DATES, ["S1", "S2"], BASE)
    rows = [p["index"] for p in chart_points(c)]
    assert rows == list(range(6))
    assert chart_points(c)[4]["kind"] == "serum" and chart_points(c)[4]["serum_id"] == "S1"
