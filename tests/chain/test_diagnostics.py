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
