"""The `move_lower` option: off leaves chains as they are; on runs the trapped-point loop's
move-lower rule, makes the steps again, and records the rule and its rounds."""

import dataclasses
from types import SimpleNamespace

import numpy as np
import pytest

from af.chain import backend
from af.chain.config import MapOptions, config_to_json, option_parameters
from af.chain.engine import ChainError, run_chain

from .test_engine import config, reused, run, tables  # noqa: F401  (tables is a fixture)


def test_option_left_at_its_old_value_is_not_a_parameter():
    assert "move_lower" not in option_parameters(MapOptions())
    assert option_parameters(MapOptions(move_lower=True))["move_lower"] is True


def test_the_stand_in_refuses_it(tables, tmp_path):  # noqa: F811
    cfg = config(tables)
    assert "move_lower" not in config_to_json(cfg)["options"]
    on = dataclasses.replace(cfg, options=dataclasses.replace(cfg.options, move_lower=True))
    with pytest.raises(ChainError, match="move_lower needs an optimiser with the rule"):
        run(on, tmp_path / "s")
    with pytest.raises(ValueError, match="no move-lower rule"):
        backend.StubOptimiser().optimise({}, 1, 2, 0, None, move_lower=True)


def test_the_rule_is_recorded_with_its_round_cap():
    pytest.importorskip("af.map.optimise")
    import af.map.optimise as optimise

    rule = backend.move_lower_rule()
    assert rule["name"] == optimise.MOVE_LOWER_RULE.name
    assert rule["diagnoses"] == list(optimise.MOVE_LOWER_RULE.diagnoses)  # JSON: a list
    assert rule["max_rounds"] == 20  # resolve_trapped's own default, which the chain never sets
    assert backend.CoreOptimiser().move_lower_rule == rule


def _round(kept: bool) -> SimpleNamespace:
    return SimpleNamespace(
        round=1,
        candidates=(2, 0),
        joint_stress=1.5 if kept else 2.5,
        how="jointly" if kept else "one at a time",
        moved=(2, 0) if kept else (),
        stress_before=2.0,
        stress_after=1.5 if kept else 2.0,
        kept=kept,
    )


@pytest.mark.parametrize("kept", [True, False])
def test_core_records_the_rounds_on_the_kept_map(monkeypatch, kept):
    pytest.importorskip("af.map.optimise")
    import af.map.optimise as optimise

    seen = {}
    start = np.zeros((3, 2))

    def resolve(problem, layout, **k):
        seen.update(k)
        p = SimpleNamespace(layout=layout + 1.0 if kept else layout, stress=1.5 if kept else 2.0)
        return SimpleNamespace(
            projection=p,
            rounds=1,
            moved=0,  # nothing trapped: only the rule moved points
            groups=[],
            threads=1,
            move_lower=[_round(kept)],
            move_lower_stop="nothing lower" if kept else "stress did not fall",
        )

    monkeypatch.setattr(optimise, "resolve_trapped", resolve)
    maps = [{"layout": start, "stress": 2.0, "start_seed": 0}]
    [out] = backend.CoreOptimiser()._resolve(None, maps, False, move_lower=True)
    assert seen["move_lower"] is True
    assert out["resolved_lower"]["stop"] == ("nothing lower" if kept else "stress did not fall")
    [r] = out["resolved_lower"]["rounds"]
    assert r["candidates"] == [2, 0] and r["moved"] == ([2, 0] if kept else [])
    assert out["stress"] == (1.5 if kept else 2.0)
    assert np.array_equal(out["layout"], start + 1.0 if kept else start)

    from af.chain.engine import _loop

    assert _loop([out])["move_lower"] == out["resolved_lower"]


def test_core_records_nothing_when_off(monkeypatch):
    pytest.importorskip("af.map.optimise")
    import af.map.optimise as optimise

    seen = {}

    def resolve(problem, layout, **k):
        seen.update(k)
        return SimpleNamespace(
            projection=SimpleNamespace(layout=layout, stress=2.0),
            rounds=1,
            moved=0,
            groups=[],
            threads=1,
            move_lower=[],
            move_lower_stop=None,
        )

    monkeypatch.setattr(optimise, "resolve_trapped", resolve)
    maps = [{"layout": np.zeros((1, 2)), "stress": 2.0, "start_seed": 0}]
    [out] = backend.CoreOptimiser()._resolve(None, maps, False)
    assert seen["move_lower"] is False and "resolved_lower" not in out


def test_on_reruns_every_step_and_records_the_rule(tables, tmp_path):  # noqa: F811
    pytest.importorskip("af.map.optimise")
    cfg = config(tables)
    core = backend.CoreOptimiser(threads=1)
    assert reused(run_chain(cfg, tmp_path / "s", optimiser=core)) == [False] * 4
    on = dataclasses.replace(cfg, options=dataclasses.replace(cfg.options, move_lower=True))
    results = run_chain(on, tmp_path / "s", optimiser=core)
    assert reused(results) == [False] * 4  # a map option changed: every step is remade
    for r in results:
        for kind, loop in r.record["trapped_loop"].items():
            assert loop["move_lower"]["stop"] in ("nothing lower", "stress did not fall"), kind
    import json

    from af.chain.engine import CHOSEN, step_dir

    sidecar = step_dir(tmp_path / "s" / cfg.name, 0) / f"{CHOSEN}.provenance.json"
    params = json.loads(sidecar.read_text())["parameters"]
    assert params["options"]["move_lower"] is True
    assert params["move_lower_rule"] == backend.move_lower_rule()
    again = run_chain(on, tmp_path / "s", optimiser=core)
    assert reused(again) == [True] * 4
