"""Ferret sera only (Sarah, 1 Oct 2026): the default for every chain, opt-in to keep, reported."""

import dataclasses
import json

import pytest

from af.chain.config import ChainConfig
from af.chain.select import SelectError, Selection, SeraPolicy
from af.chart.ace import read_chart, write_chart

from .test_engine import config, reused, run, tables  # noqa: F401  (tables is a fixture)


def with_mouse_serum(tables_dir):
    """Make the second table's serum TEST-2 a mouse serum (its own id: a different serum)."""
    path = sorted(tables_dir.iterdir())[1]
    chart = read_chart(path)
    chart.sera[1].species = "MOUSE"
    chart.sera[1].serum_id = "M001"
    write_chart(chart, path)


def test_ferret_only_chains_keep_their_steps(tables, tmp_path):  # noqa: F811
    cfg = config(tables)
    first = run(cfg, tmp_path / "s")
    assert all(
        "non_ferret_sera" not in r.record or not r.record["non_ferret_sera"]["removed"]
        for r in first
    )
    assert reused(run(cfg, tmp_path / "s")) == [True] * 4


def test_a_mouse_serum_is_removed_and_reported(tables, tmp_path):  # noqa: F811
    with_mouse_serum(tables)
    results = run(config(tables), tmp_path / "s")
    step1 = results[1].record["non_ferret_sera"]
    assert step1["removed"] and step1["non_ferret"][0]["caught_by"] == "species MOUSE"
    final = read_chart(results[-1].directory / "chosen.ace")
    assert all(s.species != "MOUSE" for s in final.sera)
    doc = json.loads((results[-1].directory.parent.parent / "chain.json").read_text())
    summary = doc["non_ferret_sera"]
    assert summary["removed_rows_by_test"] == {"species MOUSE": 1}
    assert summary["non_ferret_on_final_map"] == 0 and "ferret sera only" in summary["policy"]


def test_keep_needs_a_reason_and_keeps(tables, tmp_path):  # noqa: F811
    with pytest.raises(SelectError, match="non_ferret_reason"):
        Selection(non_ferret_sera="keep")
    with_mouse_serum(tables)
    cfg = dataclasses.replace(config(tables), sera_policy=SeraPolicy("keep", "test map needs it"))
    results = run(cfg, tmp_path / "s")
    assert not results[1].record["non_ferret_sera"]["removed"]
    final = read_chart(results[-1].directory / "chosen.ace")
    assert any(s.species == "MOUSE" for s in final.sera)


def test_the_policy_is_a_parameter_only_when_it_removes(tables, tmp_path):  # noqa: F811
    from af.chain.backend import StubOptimiser
    from af.chain.engine import Mapper, chain_steps

    def params(cfg: ChainConfig) -> dict:
        return dict(chain_steps(cfg, tmp_path / "x", Mapper(StubOptimiser(), None))[0].parameters)

    assert "non_ferret_sera" not in params(config(tables))
    with_mouse_serum(tables)
    assert "non_ferret_sera" in params(config(tables))
