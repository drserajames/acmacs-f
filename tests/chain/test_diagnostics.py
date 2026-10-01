"""Flags are judged from recorded numbers with thresholds that can change without remapping."""

from af.chain.diagnostics import Thresholds, flags


def test_flags_from_recorded_numbers():
    d = {
        "moved_far": [{"point": "a", "distance": 2.0}, {"point": "b", "distance": 1.5}]
        + [{"point": "c", "distance": 0.7}],
        "stress_per_term_change": 0.2,
        "incremental_minus_scratch_relative": 0.03,
        "incremental_vs_scratch_rmsd": 0.9,
        "column_basis_slack": [{"serum": "s", "slack": 2.0}, {"serum": "t", "slack": 0.6}],
        "trapped": 1,
        "hemisphering": 4,
        "newly_disconnected": ["x"],
    }
    assert flags(d) == [
        "1 newly disconnected",
        "stress per titre +20%",
        "scratch beat incremental by 3.0%, different basin (RMSD 0.90)",
        "1 sera with column-basis slack ≥ 1.0",
        "1 trapped",
    ]
    loose = Thresholds(moved_far=0.5, moved_far_count=3, stress_jump=0.5)
    assert "3 points moved > 0.5" in flags(d, loose)
    assert not any("stress per titre" in f for f in flags(d, loose))


def test_incremental_below_scratch_is_not_a_problem():
    assert (
        flags({"incremental_minus_scratch_relative": -0.05, "incremental_vs_scratch_rmsd": 3.0})
        == []
    )


def _chart(n_ag: int, n_sr: int):
    from af.chart.model import Antigen, Chart, Serum, Titres, empty_table

    return Chart(
        info={"D": "20210115"},
        antigens=[Antigen(f"TEST-{i}", passage="MDCK1") for i in range(n_ag)],
        sera=[Serum(f"TEST-S{j}", serum_id=f"S{j}") for j in range(n_sr)],
        titres=Titres(empty_table(n_ag, n_sr), []),
    )


def _maps(layouts_stresses):
    return [{"layout": lay, "stress": s} for lay, s in layouts_stresses]


def test_two_position_points_same_basin_near_equal_only():
    import numpy as np

    from af.chain.diagnostics import two_position_points

    chart = _chart(196, 4)  # real maps have thousands: one 0.5 move barely shifts the RMSD
    rng = np.random.default_rng(1)
    best = rng.normal(size=(200, 2)) * 3
    other = best.copy()
    other[2] += [0.5, 0.0]  # one antigen in its second place, every other point where it was
    theta = 0.7  # the same map rotated and shifted: Procrustes removes that
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    other = other @ rot.T + [4.0, -1.0]
    far_basin = rng.normal(size=(200, 2)) * 3  # near-equal stress but another basin: not compared
    worse = best.copy()
    worse[5] += [2.0, 0.0]  # same basin but not near-equal stress: not compared
    d = two_position_points(
        chart, _maps([(best, 100.0), (other, 100.05), (far_basin, 100.06), (worse, 120.0)])
    )
    assert d["near_best_maps"] == 1
    [p] = d["two_position_points"]
    assert p["point"].startswith("AG ") and p["maps"] == 1
    assert abs(p["max_distance"] - 0.5) < 0.05
    flag = flags(d)
    assert flag and flag[0].startswith("1 point(s) with two near-equal positions")
    assert flags(d, Thresholds(two_position=0.6)) == []


def test_two_position_points_none_when_maps_agree():
    import numpy as np

    from af.chain.diagnostics import two_position_points

    chart = _chart(6, 4)
    best = np.arange(20, dtype=float).reshape(10, 2)
    d = two_position_points(chart, _maps([(best, 50.0), (best + 1.0, 50.01)]))
    assert d == {"near_best_maps": 1, "two_position_points": []}
