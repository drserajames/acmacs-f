"""`[options] merge_all`: all tables merged, then one map from scratch (as ae download_full)."""

import dataclasses
import json

import pytest

from af.chain.config import ChainConfig, ChainConfigError, MapOptions, option_parameters
from af.chain.review import build_review
from af.chain.select import RemoveRule
from af.chart.ace import read_chart

from .test_engine import config, reused, run, tables  # noqa: F401  (tables is a fixture)


def merge_all(cfg: ChainConfig, **kw) -> ChainConfig:
    return dataclasses.replace(cfg, options=dataclasses.replace(cfg.options, merge_all=True), **kw)


def test_off_is_not_a_parameter_so_existing_chains_keep_their_steps():
    assert "merge_all" not in option_parameters(MapOptions())
    assert option_parameters(MapOptions(merge_all=True))["merge_all"] is True


def test_one_step_holds_every_table(tables, tmp_path):  # noqa: F811
    cfg = config(tables)
    chain = run(cfg, tmp_path / "chain")
    results = run(merge_all(cfg), tmp_path / "all")
    assert len(results) == 1
    rec = results[0].record
    assert [t["table_id"] for t in rec["merge_all"]] == [t.table_id for t in cfg.tables]
    assert rec["chosen"] == "scratch" and rec["table_id"] == cfg.tables[-1].table_id
    merged = read_chart(results[0].directory / "chosen.ace")
    by_chain = read_chart(chain[-1].directory / "chosen.ace")
    assert merged.n_antigens == by_chain.n_antigens and merged.n_sera == by_chain.n_sera
    assert merged.projections and merged.projections[0].stress is not None
    doc = json.loads((results[0].directory.parent.parent / "chain.json").read_text())
    assert doc["complete"] is True and len(doc["steps"]) == 1
    assert build_review(results[0].directory.parent.parent).exists()


def test_rerun_reuses_and_any_changed_table_remakes(tables, tmp_path):  # noqa: F811
    cfg = merge_all(config(tables))
    run(cfg, tmp_path / "s")
    assert reused(run(cfg, tmp_path / "s")) == [True]
    last = sorted(tables.iterdir())[-1]
    chart = read_chart(last)
    chart.antigens[-1].passage = "MDCK2"  # a table's content changes
    from af.chart.ace import write_chart

    write_chart(chart, last)
    assert reused(run(merge_all(config(tables)), tmp_path / "s")) == [False]


def test_rules_still_apply_and_are_recorded(tables, tmp_path):  # noqa: F811
    cfg = merge_all(config(tables), remove=[RemoveRule("antigens", "test", name="TEST-201")])
    [result] = run(cfg, tmp_path / "s")
    assert "TEST-201" not in {a.name for a in read_chart(result.directory / "chosen.ace").antigens}
    removed = result.record["removed"]
    assert sum(len(r["points"]) for rules in removed.values() for r in rules) == 1


def test_merge_all_and_a_seed_map_do_not_mix(tables, tmp_path):  # noqa: F811
    cfg = config(tables)
    with pytest.raises(ChainConfigError, match="merge_all"):
        ChainConfig(
            cfg.name,
            cfg.tables,
            MapOptions(merge_all=True),
            first_map=tmp_path / "seed.ace",
        )
