"""The `move_groups` option: off leaves chains as they are; on reruns steps and reports groups."""

import dataclasses
from types import SimpleNamespace

import numpy as np
import pytest

from af.chain import backend
from af.chain.config import MapOptions, config_to_json, option_parameters
from af.chain.diagnostics import flags, group_moves
from af.chart.model import Antigen, Chart, Serum, Titres, empty_table

from .test_engine import config, reused, run, tables  # noqa: F401  (tables is a fixture)


def test_option_left_at_its_old_value_is_not_a_parameter():
    assert "move_groups" not in option_parameters(MapOptions())
    assert option_parameters(MapOptions(move_groups=True))["move_groups"] is True


def test_off_changes_nothing_on_reruns_every_step(tables, tmp_path):  # noqa: F811
    cfg = config(tables)
    assert "move_groups" not in config_to_json(cfg)["options"]
    assert reused(run(cfg, tmp_path / "s")) == [False] * 4
    on = dataclasses.replace(cfg, options=dataclasses.replace(cfg.options, move_groups=True))
    results = run(on, tmp_path / "s")
    assert reused(results) == [False] * 4  # a map option changed: every step is remade
    # the stand-in optimiser tries no groups, but every step says it looked
    assert all(r.record["diagnostics"]["group_moves"] == [] for r in results[1:])


def test_flags_name_each_kept_group_move():
    d = {
        "group_moves": [
            {
                "size": 6,
                "distance": 2.81,
                "stress_before": 5996.66,
                "stress_after": 5994.85,
                "kept": True,
            },
            {
                "size": 2,
                "distance": 0.7,
                "stress_before": 10.0,
                "stress_after": 10.2,
                "kept": False,
            },
        ]
    }
    assert flags(d) == ["group of 6 moved 2.81 (stress -1.81)"]


def test_group_moves_names_the_points():
    chart = Chart(
        info={},
        antigens=[Antigen("TEST-A", passage="P1"), Antigen("TEST-B", passage="P1")],
        sera=[Serum("TEST-S", serum_id="S1")],
        titres=Titres(empty_table(2, 1)),
    )
    groups = [
        {
            "members": [0, 2],
            "shift": [3.0, 4.0],
            "stress_before": 9.0,
            "stress_after": 7.5,
            "kept": True,
        }
    ]
    [g] = group_moves(chart, groups)
    assert (g["size"], g["distance"], g["points"]) == (2, 5.0, ["AG TEST-A P1", "SR TEST-S S1"])
    assert (g["antigens"], g["sera"]) == (1, 1)
    assert flags({"group_moves": [g]}) == [
        "MIXED (1 antigens, 1 sera) group of 2 moved 5.00 (stress -1.50)"
    ]


def fake_resolution(layout, groups, moved=0):
    return SimpleNamespace(
        projection=SimpleNamespace(layout=layout, stress=1.0),
        rounds=1,
        moved=moved,
        groups=[SimpleNamespace(**g) for g in groups],
    )


@pytest.mark.parametrize("kept", [True, False])
def test_core_keeps_a_kept_group_move_and_records_every_group(monkeypatch, kept):
    pytest.importorskip("af.map.optimise")
    import af.map.optimise as optimise

    moved_layout = np.array([[9.0, 9.0]])
    group = {
        "members": [0],
        "shift": (1.0, 0.0),
        "stress_before": 2.0,
        "stress_after": 1.0,
        "kept": kept,
    }
    monkeypatch.setattr(
        optimise, "resolve_trapped", lambda *a, **k: fake_resolution(moved_layout, [group])
    )
    maps = [{"layout": np.array([[0.0, 0.0]]), "stress": 2.0}]
    out = backend.CoreOptimiser()._resolve(None, maps, move_groups=True)
    assert out[0]["resolved_groups"] == [
        {
            "members": [0],
            "shift": [1.0, 0.0],
            "stress_before": 2.0,
            "stress_after": 1.0,
            "kept": kept,
        }
    ]
    assert np.array_equal(out[0]["layout"], moved_layout if kept else np.array([[0.0, 0.0]]))


def test_core_records_nothing_when_off(monkeypatch):
    pytest.importorskip("af.map.optimise")
    import af.map.optimise as optimise

    seen = {}

    def resolve(*a, **k):
        seen.update(k)
        return fake_resolution(np.zeros((1, 2)), [])

    monkeypatch.setattr(optimise, "resolve_trapped", resolve)
    out = backend.CoreOptimiser()._resolve(
        None, [{"layout": np.zeros((1, 2)), "stress": 2.0}], move_groups=False
    )
    assert seen["move_groups"] is False and "resolved_groups" not in out[0]
