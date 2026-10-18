"""The `move_lower` option: off leaves chains as they are; on runs the trapped-point loop's
move-lower rule, makes the steps again, and records the rule and its rounds."""

import dataclasses
from types import SimpleNamespace

import numpy as np
import pytest

from af.chain import backend
from af.chain.config import ChainConfigError, MapOptions, config_to_json, option_parameters
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
    assert backend.CoreOptimiser().move_lower_rule() == rule


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
            move_lower_rule=optimise.MOVE_LOWER_RULE,
        )

    monkeypatch.setattr(optimise, "resolve_trapped", resolve)
    maps = [{"layout": start, "stress": 2.0, "start_seed": 0}]
    [out] = backend.CoreOptimiser()._resolve(None, maps, False, move_lower=True)
    assert seen["move_lower"] is True
    assert out["resolved_lower"]["rule"] == "move-lower-1"
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


# AD's last round (Q198): the rule is chosen by name, recorded with its own cap of one round, and
# reaches resolve_trapped as the af.map.optimise constant of that name.

AD = "ad-last-round-1"


def test_every_rule_name_is_the_name_its_rule_records():
    pytest.importorskip("af.map.optimise")
    import af.map.optimise as optimise

    for name, constant in backend.MOVE_LOWER_RULES.items():
        assert getattr(optimise, constant).name == name
    assert backend.MOVE_LOWER_RULES[backend.DEFAULT_MOVE_LOWER_RULE] == "MOVE_LOWER_RULE"


def test_ad_last_round_is_recorded_with_one_round():
    pytest.importorskip("af.map.optimise")
    rule = backend.move_lower_rule(AD)
    assert rule["name"] == AD and rule["max_rounds"] == 1 and rule["one_at_a_time"] is False
    assert backend.CoreOptimiser().move_lower_rule(AD) == rule
    assert backend.StubOptimiser().move_lower_rule(AD) is None
    with pytest.raises(ValueError, match="unknown move-lower rule"):
        backend.move_lower_rule("move-lower-2")


def test_the_rule_option_is_a_parameter_only_when_not_the_old_one():
    assert "move_lower_rule" not in option_parameters(MapOptions(move_lower=True))
    params = option_parameters(MapOptions(move_lower=True, move_lower_rule=AD))
    assert params["move_lower"] is True and params["move_lower_rule"] == AD


def test_a_rule_alone_or_unknown_is_refused():
    with pytest.raises(ChainConfigError, match="needs move_lower = true"):
        MapOptions(move_lower_rule=AD)
    with pytest.raises(ChainConfigError, match="move_lower_rule must be one of"):
        MapOptions(move_lower=True, move_lower_rule="ad-last-round")


def test_the_rule_is_a_search_setting_in_continuity():
    from af.chain.continuity import _search_apart

    rest, search = _search_apart(
        {"seed": 3, "options": {"move_lower": True, "move_lower_rule": AD}}
    )
    assert rest == {"seed": 3}
    assert search == {"resolve_best": 1, "move_lower": True, "move_lower_rule": AD}
    _, old = _search_apart({"options": {}})
    assert old["move_lower_rule"] == backend.DEFAULT_MOVE_LOWER_RULE


@pytest.mark.parametrize("name, constant", [(None, "MOVE_LOWER_RULE"), (AD, "AD_LAST_ROUND")])
def test_core_hands_resolve_trapped_the_rule_named(monkeypatch, name, constant):
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
            move_lower_stop="nothing lower",
            move_lower_rule=k.get("move_lower_rule"),
        )

    monkeypatch.setattr(optimise, "resolve_trapped", resolve)
    maps = [{"layout": np.zeros((1, 2)), "stress": 2.0, "start_seed": 0}]
    named = {} if name is None else {"move_lower_rule": name}
    backend.CoreOptimiser()._resolve(None, maps, False, move_lower=True, **named)
    assert seen["move_lower_rule"] is getattr(optimise, constant)


@pytest.mark.parametrize("split", [False, True])
def test_ad_last_round_reaches_every_maps_loop(tables, tmp_path, monkeypatch, split):  # noqa: F811
    """What ran, not what was asked: every trapped-point loop of every step (scratch, incremental,
    and after a split relax the parent's combine) is handed AD_LAST_ROUND, and records it."""
    pytest.importorskip("af.map.optimise")
    import functools
    import json

    import af.map.optimise as optimise
    from af.chain.engine import CHOSEN, SplitStarts, step_dir
    from af.run.local import LocalRunner

    real = optimise.resolve_trapped
    rules = []

    @functools.wraps(real)  # the default round cap is read from resolve_trapped's signature
    def spy(problem, layout, **kwargs):
        rules.append(kwargs.get("move_lower_rule"))
        return real(problem, layout, **kwargs)

    monkeypatch.setattr(optimise, "resolve_trapped", spy)
    cfg = config(tables)
    on = dataclasses.replace(
        cfg, options=dataclasses.replace(cfg.options, move_lower=True, move_lower_rule=AD)
    )
    core = backend.CoreOptimiser(threads=1)  # strict-fp-1, the core's default arithmetic
    chunks = SplitStarts(chunks=2, work_dir=tmp_path / "w") if split else None
    runner = LocalRunner() if split else None
    results = run_chain(on, tmp_path / "s", optimiser=core, runner=runner, split=chunks)
    assert reused(results) == [False] * 4
    # step 0 maps from scratch, each later step incrementally and from scratch: 7 loops
    assert len(rules) == 7 and all(r is optimise.AD_LAST_ROUND for r in rules), rules
    loops = [loop for r in results for loop in r.record["trapped_loop"].values()]
    assert len(loops) == 7 and all(loop["move_lower"]["rule"] == AD for loop in loops)
    assert all(len(loop["move_lower"]["rounds"]) <= 1 for loop in loops)
    sidecar = step_dir(tmp_path / "s" / cfg.name, 0) / f"{CHOSEN}.provenance.json"
    params = json.loads(sidecar.read_text())["parameters"]
    assert params["options"]["move_lower_rule"] == AD
    assert params["move_lower_rule"] == backend.move_lower_rule(AD)
    assert params["numerics"] == core.numerics == "strict-fp-1"  # recorded together


def test_ad_last_round_is_another_map_than_move_lower(tables, tmp_path):  # noqa: F811
    pytest.importorskip("af.map.optimise")
    cfg = config(tables)
    core = backend.CoreOptimiser(threads=1)

    def options(**k):
        return dataclasses.replace(cfg, options=dataclasses.replace(cfg.options, **k))

    lower, ad = options(move_lower=True), options(move_lower=True, move_lower_rule=AD)
    assert reused(run_chain(lower, tmp_path / "s", optimiser=core)) == [False] * 4
    assert reused(run_chain(ad, tmp_path / "s", optimiser=core)) == [False] * 4
    assert reused(run_chain(ad, tmp_path / "s", optimiser=core)) == [True] * 4
