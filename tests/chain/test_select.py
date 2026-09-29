"""`[[select.remove]]`: named, reasoned rules that leave points out of every table of a chain."""

import dataclasses

import pytest

from af.chain.config import ChainConfig, ChainConfigError, config_to_json
from af.chain.select import RemoveRule, SelectError, apply, check_rules_match
from af.chart.ace import read_chart
from af.chart.model import Antigen, Chart, Serum, Titres, empty_table
from af.chart.titre import Titre

from .test_engine import config, reused, run, tables  # noqa: F401  (tables is a fixture)


def rule(**kw):
    return RemoveRule(**{"what": "antigens", "reason": "test", **kw})


@pytest.mark.parametrize(
    "kw",
    [
        {"reason": " ", "name": "X"},  # no reason
        {"name": "X", "passage": "egg"},  # two selectors
        {},  # no selector
        {"what": "points", "name": "X"},
        {"passage": "reassortant"},
    ],
)
def test_bad_rules_are_refused(kw):
    with pytest.raises(SelectError):
        rule(**kw)


@pytest.mark.parametrize(
    ("passage", "egg"),
    [("E3", True), ("E3/E2SIAT1", True), ("MDCK1", False), ("SIAT1 (2021-01-01)", False)],
)
def test_passage_egg_means_any_step_in_eggs(passage, egg):
    # any step, not ae's last step: E3/E2SIAT1 is egg here (Sarah, 29 Sep 2026)
    assert rule(passage="egg").matches(Antigen("TEST-A", passage=passage)) is egg


def small_chart():
    table = empty_table(3, 2)
    for i in range(3):
        for j in range(2):
            table[i][j] = Titre.parse(str(10 * 2 ** (i + j)))
    layer = {(i, j): table[i][j] for i in range(3) for j in range(2)}
    return Chart(
        info={"D": "20210115"},
        antigens=[Antigen(f"TEST-{i}", passage="MDCK1") for i in range(3)],
        sera=[Serum("TEST-S0", serum_id="S0", passage="E3"), Serum("TEST-S1", serum_id="S1")],
        titres=Titres(table, [layer]),
    )


def test_apply_removes_points_and_their_titres():
    chart = small_chart()
    kept, report = apply([rule(name="TEST-1"), rule(what="sera", passage="egg")], chart)
    assert [a.name for a in kept.antigens] == ["TEST-0", "TEST-2"]
    assert [s.name for s in kept.sera] == ["TEST-S1"]
    assert [[t.value for t in row] for row in kept.titres.table] == [[20], [80]]
    assert {k: v.value for k, v in kept.titres.layers[0].items()} == {(0, 0): 20, (1, 0): 80}
    assert [len(r["points"]) for r in report] == [1, 1]


def test_a_rule_matching_nothing_in_any_table_is_an_error(tables):  # noqa: F811
    paths = sorted(tables.iterdir())
    check_rules_match([rule(name="TEST-201")], paths)  # in the second table only: fine
    with pytest.raises(SelectError, match="match nothing"):
        check_rules_match([rule(name="TEST-201"), rule(name="NO-SUCH-VIRUS")], paths)


def test_chain_leaves_the_points_out_and_counts_them(tables, tmp_path):  # noqa: F811
    cfg = config(tables)
    plain = run(cfg, tmp_path / "plain")
    run(cfg, tmp_path / "s")
    selected = dataclasses.replace(cfg, remove=[rule(name="TEST-201")])
    results = run(selected, tmp_path / "s")
    assert reused(results) == [False] * 4  # a new selection is a new parameter: all remade
    assert results[1].record["removed"][0]["points"] == ["TEST-201 MDCK1"]
    assert results[2].record["removed"][0]["points"] == []
    final = read_chart(results[-1].directory / "chosen.ace")
    assert "TEST-201" not in {a.name for a in final.antigens}
    assert final.n_antigens == read_chart(plain[-1].directory / "chosen.ace").n_antigens - 1
    assert config_to_json(selected)["select_remove"] == [
        {"what": "antigens", "reason": "test", "name": "TEST-201"}
    ]
    assert "select_remove" not in config_to_json(cfg)  # no rules: the config reads as before
    assert reused(run(cfg, tmp_path / "s")) == [False] * 4  # back to no rules: remade again


def test_rules_and_a_seed_map_do_not_mix(tables, tmp_path):  # noqa: F811
    cfg = config(tables)
    with pytest.raises(ChainConfigError, match="first_map"):
        ChainConfig(cfg.name, cfg.tables, first_map=tmp_path / "seed.ace", remove=[rule(name="X")])


def test_toml_rules(tables, tmp_path):  # noqa: F811
    from af.chain.config import load_chain_config

    path = tmp_path / "chain.toml"
    head = f'name = "h9-hi-test-lab"\nseed = 3\n[tables]\ndirectory = "{tables}"\n'
    path.write_text(
        head
        + 'group = "h9-hi-test-lab"\n'
        + '[[select.remove]]\nwhat = "sera"\npassage = "egg"\nreason = "egg-free map"\n'
        + '[[select.remove]]\nwhat = "antigens"\ndesignation = "TEST-201 MDCK1"\n'
        + 'reason = "curated out"\n'
    )
    cfg = load_chain_config(path)
    assert [(r.what, r.passage, r.designation) for r in cfg.remove] == [
        ("sera", "egg", None),
        ("antigens", None, "TEST-201 MDCK1"),
    ]
