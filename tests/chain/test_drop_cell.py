"""One reading kept out of the map by a named rule, the table keeping the lab's value
(Sarah, 2 Oct 2026, Q98)."""

import dataclasses
import json

import pytest

from af.chain.config import MapOptions
from af.chain.select import DropCell, SelectError, check_cells_match, drop_cells
from af.chart.ace import read_chart

from .test_engine import config, run, tables  # noqa: F401  (tables is a fixture)

WHY = {"reason": "implausible reading; lab asked", "decided": "test"}


def _present_cell(chart):
    for i, row in enumerate(chart.titres.table):
        for j, t in enumerate(row):
            if not t.is_missing:
                return i, j
    raise AssertionError("no reading")


def _rule(cfg, k=0, **kw):
    t = cfg.tables[k]
    chart = read_chart(t.path)
    i, j = _present_cell(chart)
    fields = {
        "table": t.table_id,
        "antigen": chart.antigens[i].designation(),
        "serum": chart.sera[j].designation(),
        **WHY,
    }
    return DropCell(**{**fields, **kw}), chart, (i, j)


def test_the_named_reading_goes_missing_and_is_reported(tables):  # noqa: F811
    cfg = config(tables)
    rule, chart, (i, j) = _rule(cfg)
    out, [r] = drop_cells([rule], rule.table, chart)
    assert out.titres.table[i][j].is_missing
    assert r["titre"] == str(chart.titres.table[i][j]) and r["reason"] == WHY["reason"]
    others = [
        (a, b) for a in range(chart.n_antigens) for b in range(chart.n_sera) if (a, b) != (i, j)
    ]
    assert all(out.titres.table[a][b] == chart.titres.table[a][b] for a, b in others)
    assert drop_cells([rule], "another-table", chart) == (chart, [])


def test_a_rule_must_match_exactly_one_present_reading(tables):  # noqa: F811
    cfg = config(tables)
    rule, chart, _ = _rule(cfg)
    with pytest.raises(SelectError, match="matches 0"):
        drop_cells([dataclasses.replace(rule, antigen="NO-SUCH")], rule.table, chart)
    out, _ = drop_cells([rule], rule.table, chart)
    with pytest.raises(SelectError, match="already missing"):
        drop_cells([rule], rule.table, out)
    with pytest.raises(SelectError, match="not one of the chain's tables"):
        check_cells_match([dataclasses.replace(rule, table="x-19990101")], {})
    with pytest.raises(SelectError, match="empty reason"):
        dataclasses.replace(rule, reason=" ")


def test_a_merge_all_map_leaves_it_out_and_records_it(tables, tmp_path):  # noqa: F811
    base = config(tables)
    rule, chart, (i, j) = _rule(base, k=1)
    cfg = dataclasses.replace(
        base, options=MapOptions(merge_all=True, scratch_starts=4), drop_cells=[rule]
    )
    [result] = run(cfg, tmp_path / "s")
    rec = json.loads((result.directory / "step.json").read_text())
    [d] = rec["dropped_cells_by_rule"]
    assert (d["table"], d["antigen"], d["serum"]) == (rule.table, rule.antigen, rule.serum)
    merged = read_chart(result.directory / "merge.ace")
    a = next(k for k, x in enumerate(merged.antigens) if x.designation() == rule.antigen)
    s = next(k for k, x in enumerate(merged.sera) if x.designation() == rule.serum)
    layer = merged.source_dates().index(rule.table.rsplit("-", 1)[-1])
    assert (a, s) not in merged.titres.layers[layer]  # the map never sees that reading


def test_a_rule_naming_nothing_stops_the_chain_before_any_step(tables, tmp_path):  # noqa: F811
    base = config(tables)
    rule, _, _ = _rule(base)
    cfg = dataclasses.replace(base, drop_cells=[dataclasses.replace(rule, serum="NO-SUCH")])
    with pytest.raises(SelectError, match="matches 0"):
        run(cfg, tmp_path / "s")
    assert not (tmp_path / "s").exists() or not any((tmp_path / "s").rglob("step.json"))


def test_chain_file_rows(tmp_path):
    from af.chain.config import ChainSettings
    from af.util.config import load_config

    path = tmp_path / "chain.toml"
    path.write_text(
        'name = "x"\nseed = 1\n[tables]\ndataset = "labx/h9"\n'
        '[options]\nminimum_column_basis = "none"\n'
        '[[select.drop_cell]]\ntable = "h9-20221208"\nantigen = "A"\nserum = "S"\n'
        'reason = "r"\ndecided = "d"\n'
    )
    [rule] = load_config(path, ChainSettings).select.drop_cell
    assert (rule.table, rule.antigen, rule.serum) == ("h9-20221208", "A", "S")
