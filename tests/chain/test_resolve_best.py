"""MapOptions.resolve_best: the trapped-point loop on the best K refined maps, the lowest kept.

06-optimiser, gp-cnic (12 Oct 2026): the best refined layout resolved to 26,882 while the next
four resolved to ~26,236. Opt-in; Sarah (Q155): "Keep as is", so the default stays 1 and must
change nothing, which these tests hold to.
"""

import dataclasses
from types import SimpleNamespace

import numpy as np
import pytest

from af.chain import backend
from af.chain.config import ChainConfigError, MapOptions, option_parameters
from af.chain.engine import _loop

from .test_scratch_rough_core import many_minima


def test_the_default_changes_no_parameters_and_k_is_bounded():
    assert option_parameters(MapOptions()) == option_parameters(MapOptions(resolve_best=1))
    assert "resolve_best" not in option_parameters(MapOptions())
    assert option_parameters(MapOptions(resolve_best=3))["resolve_best"] == 3
    for k in (0, 6):
        with pytest.raises(ChainConfigError, match="resolve_best must be 1 to 5"):
            MapOptions(resolve_best=k)


def _fake_resolve(lower_by: dict[float, float], calls: list):
    """resolve_trapped standing in: the map whose stress is in `lower_by` drops by that much."""

    def resolve(
        problem, layout, threads=0, move_groups=False, move_lower=False, move_lower_rule=None
    ):
        key = round(float(layout[0, 0]), 6)
        calls.append(key)
        drop = lower_by.get(key, 0.0)
        p = SimpleNamespace(layout=layout + 1.0, stress=STRESS[key] - drop)
        return SimpleNamespace(projection=p, moved=int(drop > 0), rounds=1, groups=[], threads=1)

    return resolve


STRESS = {0.0: 10.0, 1.0: 11.0, 2.0: 12.0, 3.0: 13.0}


def _maps() -> list[dict]:
    return [
        {"layout": np.full((3, 2), x), "stress": s, "start_seed": int(x), "threads": 1, "cpus": 1}
        for x, s in STRESS.items()
    ]


def test_k_resolves_the_best_k_and_keeps_the_lowest(monkeypatch):
    import af.map.optimise as optimise

    calls: list = []
    monkeypatch.setattr(optimise, "resolve_trapped", _fake_resolve({0.0: 0.5, 2.0: 5.0}, calls))
    opt = backend.CoreOptimiser(threads=1)

    one = opt._resolve(None, _maps(), False)
    assert calls == [0.0] and one[0]["stress"] == 9.5 and "refined_rank" not in one[0]
    assert "maps" not in one[0]["run_threads"]["trapped"]

    calls.clear()
    three = opt._resolve(None, _maps(), False, resolve_best=3)
    assert calls == [0.0, 1.0, 2.0]  # the best three by refined stress, in that order
    assert [m["stress"] for m in three] == [7.0, 9.5, 11.0, 13.0]
    assert three[0]["refined_rank"] == 2 and three[0]["resolved_moves"] == 1
    assert "resolved_moves" not in three[2]  # resolved, nothing moved: kept as it was
    assert three[0]["run_threads"]["trapped"]["maps"] == 3
    loop = _loop(three)
    assert loop["refined_rank"] == 2
    assert [r["refined_rank"] for r in loop["resolved"]] == [0, 1, 2]


def test_k_one_is_the_chain_as_it_was():
    """K=1 by default and explicitly: the same maps, field for field, on a real relax."""
    pytest.importorskip("af.map.optimise")
    arrays = {**many_minima(5), "scratch_precision": "rough"}
    opt = backend.CoreOptimiser(threads=1)
    default = opt.optimise(arrays, 20, 2, 3, None)
    one = opt.optimise(arrays, 20, 2, 3, None, resolve_best=1)
    assert len(default) == len(one)
    for a, b in zip(default, one, strict=True):
        assert a.keys() == b.keys() and "refined_rank" not in a
        np.testing.assert_array_equal(a["layout"], b["layout"])
        assert {k: v for k, v in a.items() if k != "layout"} == {
            k: v for k, v in b.items() if k != "layout"
        }


def test_k_above_one_never_ends_higher_than_k_one():
    pytest.importorskip("af.map.optimise")
    arrays = {**many_minima(6), "scratch_precision": "rough"}
    opt = backend.CoreOptimiser(threads=1)
    one = opt.optimise(arrays, 20, 2, 3, None)
    five = opt.optimise(arrays, 20, 2, 3, None, resolve_best=5)
    assert five[0]["stress"] <= one[0]["stress"] + 1e-9
    assert "refined_rank" in five[0]


def test_the_stub_takes_the_option():
    arrays = many_minima(5)
    stub = backend.StubOptimiser()
    assert stub.optimise(arrays, 3, 2, 1, None, resolve_best=3)


def test_options_carry_it():
    o = dataclasses.replace(MapOptions(), resolve_best=2)
    assert o.resolve_best == 2
