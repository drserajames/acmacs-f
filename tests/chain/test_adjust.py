"""Named column-basis adjustments (Sarah, 2 Oct 2026: a round's adjustment-stage changes as named
per-map options, applied before optimising and counted)."""

import dataclasses
import json

import numpy as np
import pytest

from af.chain.adjust import AdjustError, ColumnBasisAdjustment, apply
from af.chain.config import ChainConfigError, MapOptions
from af.chart.model import Antigen, Chart, Serum, Titres, empty_table
from af.chart.titre import Titre

from .test_engine import config, run, tables  # noqa: F401  (tables is a fixture)


def _rule(rule: str, **kw) -> ColumnBasisAdjustment:
    return ColumnBasisAdjustment(rule, reason="as the round's adjusted map", decided="test", **kw)


def _chart(forced=None):
    table = empty_table(3, 3)
    for i in range(3):
        for j in range(3):
            table[i][j] = Titre.parse(str(40 * 2 ** (i + j)))
    return Chart(
        info={"D": "20210115"},
        antigens=[Antigen(f"TEST-{i}", passage="MDCK1") for i in range(3)],
        sera=[Serum(f"TEST-S{j}", serum_id=f"S{j}") for j in range(3)],
        titres=Titres(table, []),
        forced_column_bases=None if forced is None else np.asarray(forced, dtype=float),
    )


def test_lower_every_serum_from_the_forced_bases():
    chart, [r] = apply([_rule("lower", by=1.0)], _chart([6.0, 7.5, 9.0]))
    assert list(chart.column_bases("none")) == [5.0, 6.5, 8.0]
    assert r["sera_changed"] == 3 and r["max_change"] == pytest.approx(1.0)
    assert (r["changed"][1]["before"], r["changed"][1]["after"]) == (7.5, 6.5)


def test_lower_from_the_titres_when_nothing_is_forced():
    plain = _chart()
    titre_bases = np.asarray(plain.column_bases("none"), dtype=float)
    chart, [r] = apply([_rule("lower", by=0.5)], plain)
    assert np.allclose(chart.column_bases("none"), titre_bases - 0.5)


def test_lower_named_sera_only_and_a_name_must_match_once():
    designation = _chart().sera[2].designation()
    rule = _rule("lower", by=1.0, sera=[designation])
    chart, [r] = apply([rule], _chart([6.0, 7.0, 8.0]))
    assert list(chart.column_bases("none")) == [6.0, 7.0, 7.0] and r["sera_changed"] == 1
    with pytest.raises(AdjustError, match="matches 0 sera"):
        apply([_rule("lower", by=1.0, sera=["NO-SUCH"])], _chart([6.0] * 3))


def test_remove_forced_returns_to_the_titres_and_lists_what_changed():
    titre_bases = np.asarray(_chart().column_bases("none"), dtype=float)
    forced = titre_bases.copy()
    forced[0] += 2.0  # one serum's merged base differs from its titres
    chart, [r] = apply([_rule("remove_forced")], _chart(forced))
    assert chart.forced_column_bases is None
    assert np.allclose(chart.column_bases("none"), titre_bases)
    assert r["sera_changed"] == 1 and r["changed"][0]["serum"] == chart.sera[0].designation()


@pytest.mark.parametrize(
    "kw",
    [
        {"rule": "raise", "reason": "r", "decided": "d"},
        {"rule": "lower", "by": 1.0, "reason": "", "decided": "x"},
        {"rule": "lower", "by": 0.0, "reason": "r", "decided": "d"},
        {"rule": "remove_forced", "by": 1.0, "reason": "r", "decided": "d"},
    ],
)
def test_bad_rows_are_refused(kw):
    with pytest.raises(AdjustError):
        ColumnBasisAdjustment(**kw)


def test_a_merge_all_map_is_optimised_under_them_and_records_them(tables, tmp_path):  # noqa: F811
    from af.chart.ace import read_chart

    cfg = config(tables)
    base = dataclasses.replace(cfg, options=MapOptions(merge_all=True, scratch_starts=4))
    lowered = dataclasses.replace(base, column_basis_adjustments=[_rule("lower", by=1.0)])
    [plain] = run(base, tmp_path / "a")
    [result] = run(lowered, tmp_path / "b")
    merged = read_chart(result.directory / "merge.ace")
    chosen = read_chart(result.directory / "chosen.ace")
    assert np.allclose(
        chosen.column_bases("none", chosen.best()), np.asarray(merged.column_bases("none")) - 1.0
    )
    rec = json.loads((result.directory / "step.json").read_text())
    [r] = rec["column_basis_adjustments"]
    assert r["rule"] == "lower" and r["sera_changed"] == merged.n_sera
    assert "column_basis_adjustments" not in json.loads((plain.directory / "step.json").read_text())
    with pytest.raises(ChainConfigError, match="merge_all map only"):
        dataclasses.replace(cfg, column_basis_adjustments=lowered.column_basis_adjustments)


def test_chain_file_rows(tmp_path):
    from af.chain.config import ChainSettings
    from af.util.config import load_config

    path = tmp_path / "chain.toml"
    path.write_text(
        'name = "x"\nseed = 1\n[tables]\ndataset = "labx/h9"\n'
        '[options]\nminimum_column_basis = "none"\nmerge_all = true\n'
        '[[adjust_column_bases]]\nrule = "lower"\nby = 1.0\nreason = "r"\ndecided = "d"\n'
        '[[adjust_column_bases]]\nrule = "remove_forced"\nreason = "r2"\ndecided = "d2"\n'
    )
    s = load_config(path, ChainSettings)
    assert [(a.rule, a.by) for a in s.adjust_column_bases] == [
        ("lower", 1.0),
        ("remove_forced", 0.0),
    ]
