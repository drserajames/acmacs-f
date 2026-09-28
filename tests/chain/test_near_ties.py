"""Near-tied incremental/scratch choices are flagged, not changed; each step says what made it."""

import pytest

from af.chain.diagnostics import flags

from .test_engine import config, run, tables  # noqa: F401  (tables is a fixture)


@pytest.mark.parametrize("gap", [0.0005, -0.0005])  # scratch won narrowly / incremental did
def test_near_tie_with_different_maps_is_flagged(gap):
    d = {"incremental_minus_scratch_relative": gap, "incremental_vs_scratch_rmsd": 0.3}
    assert flags(d) == [f"near tie: incremental {gap:+.3%} vs scratch, maps differ (RMSD 0.30)"]


def test_same_map_both_ways_is_not_a_near_tie():
    assert (
        flags({"incremental_minus_scratch_relative": 0.0, "incremental_vs_scratch_rmsd": 0.01})
        == []
    )


def test_clear_scratch_win_is_reported_once_not_as_a_tie():
    d = {"incremental_minus_scratch_relative": 0.02, "incremental_vs_scratch_rmsd": 0.3}
    assert flags(d) == ["scratch beat incremental by 2.0%"]


def test_outside_the_margin_is_not_flagged():
    assert (
        flags({"incremental_minus_scratch_relative": 0.005, "incremental_vs_scratch_rmsd": 1.0})
        == []
    )


def test_every_step_records_its_platform(tables, tmp_path):  # noqa: F811
    for r in run(config(tables), tmp_path / "s"):
        pf = r.record["platform"]
        assert set(pf) == {"release", "cxx_version", "machine", "system"}
        assert pf["machine"] and pf["system"]
