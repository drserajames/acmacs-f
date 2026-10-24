"""The tables after a seed (gaps 6 and 7 of the chains-from-shipped plan).

A chain seeded from a map made elsewhere must merge only the tables that map does not already
hold, keep its published order from month to month, and let `[select]` rules act on those later
tables. Its own tables are its bridge, and the seed's points are the shipped map's.

Invented names only (acmacs-f is public).
"""

import dataclasses
import json
from types import SimpleNamespace

import pytest

from af.chain.__main__ import main as chain_main
from af.chain.append import SettingsChangedError
from af.chain.backend import StubOptimiser
from af.chain.config import (
    ChainConfig,
    ChainConfigError,
    MapOptions,
    SeedChart,
    load_chain_config,
    tables_from_directory,
)
from af.chain.engine import CHOSEN, run_chain, stepped_tables
from af.chain.select import RemoveRule, SelectError
from af.chain.synthetic import make_tables
from af.chart.ace import read_chart, write_chart
from af.store.store import Store
from af.store.work import Work

from .sera_markers import write_markers
from .test_tables import DATASET, make_table, publish

GROUP = "h9-hi-test-lab"
OPTS = MapOptions(scratch_starts=4, incremental_starts=3, grid_test=False)


# ----------------------------------------------------------------------
# a seed made by a chain of the first three tables, spelled as the store spells them


@pytest.fixture
def seeded(tmp_path):
    """Five tables; the first three make the seed. No respelling: the identities already match,
    so the seed needs no bridge and the chain's only question is which tables are steps."""
    make_tables(tmp_path / "t", GROUP, n_tables=5)
    refs = tables_from_directory(tmp_path / "t", GROUP)
    run_chain(ChainConfig(GROUP, refs[:3], OPTS, seed=3), tmp_path / "pre", StubOptimiser())
    made = read_chart(tmp_path / "pre" / GROUP / "steps" / "0002" / CHOSEN)
    write_chart(made, tmp_path / "seed.ace")
    return SimpleNamespace(root=tmp_path, refs=refs, seed=tmp_path / "seed.ace")


PLAIN_SEED = SeedChart(bridge=False)  # identities already match: the seed needs no bridge


def config(s, *, seed_chart=PLAIN_SEED, **kw):
    return ChainConfig(GROUP, s.refs, OPTS, seed=3, first_map=s.seed, seed_chart=seed_chart, **kw)


def test_a_seed_without_a_bridge_steps_only_the_tables_after_it(seeded):
    """Gap 6: the seed holds three tables, so only the last two are steps. Before this, every
    selected table merged onto the seed and its own three would have merged a second time."""
    out = run_chain(config(seeded), seeded.root / "s", StubOptimiser())
    assert [r.record["table_id"] for r in out] == ["seed.ace"] + [
        t.table_id for t in seeded.refs[3:]
    ]


def test_the_step_tables_come_from_the_seeds_own_source_list(seeded):
    """The split is taken from the seed chart, not from config, so it cannot drift from the file
    it describes: drop a source and that table becomes one the store has and the seed lacks."""
    assert [t.table_id for t in stepped_tables(config(seeded))] == [
        t.table_id for t in seeded.refs[3:]
    ]
    seed = read_chart(seeded.seed)
    # drop the MIDDLE source: that table is one the store has and the seed lacks. Dropping the
    # last would instead move it after the seed's dates, which is a different thing.
    seed.info["S"] = [s for k, s in enumerate(seed.info["S"]) if k != 1]
    seed.titres.layers = [x for k, x in enumerate(seed.titres.layers) if k != 1]
    write_chart(seed, seeded.seed)
    cfg = config(seeded, seed_chart=SeedChart(bridge=True, af_only_tables="append"))
    # the third table is now af-only: appended at the END, after the tables that follow the seed
    assert [t.table_id for t in stepped_tables(cfg)] == [
        t.table_id for t in (*seeded.refs[3:], seeded.refs[1])
    ]


def test_a_first_map_without_seed_chart_merges_every_table(seeded):
    """Pinned: a `first_map` chain with no `[seed_chart]` keeps the meaning it had before, every
    selected table merged onto the seed. Only `[seed_chart]` says the seed holds tables."""
    cfg = ChainConfig(GROUP, seeded.refs[3:], OPTS, seed=3, first_map=seeded.seed)
    assert [t.table_id for t in stepped_tables(cfg)] == [t.table_id for t in seeded.refs[3:]]


def test_a_chain_without_a_seed_steps_every_table_but_the_first(seeded):
    """Pinned: with no seed at all, the first table is step 0's own map."""
    cfg = ChainConfig(GROUP, seeded.refs, OPTS, seed=3)
    assert [t.table_id for t in stepped_tables(cfg)] == [t.table_id for t in seeded.refs[1:]]


# ----------------------------------------------------------------------
# gap 7: rules on the tables after the seed


def _rule(path, what="antigens"):
    chart = read_chart(path)
    point = chart.antigens[-1] if what == "antigens" else chart.sera[-1]
    return RemoveRule(what=what, name=point.name, reason="a test rule")


def test_a_rule_applies_to_the_tables_after_the_seed(seeded):
    """Gap 7: with a seed the rules were refused outright. They act on each table as it enters a
    merge, so with `[seed_chart]` they have the post-seed tables to act on, and only those."""
    rule = _rule(seeded.refs[4].path)
    out = run_chain(config(seeded, remove=[rule]), seeded.root / "r", StubOptimiser())
    removed = [r.record.get("removed") for r in out[1:]]
    assert any(any(x["points"] for x in step) for step in removed if step), removed
    # the seed's own points are untouched: step 0 is the shipped map
    assert read_chart(out[0].directory / CHOSEN).n_antigens == read_chart(seeded.seed).n_antigens


def test_a_rule_matching_only_the_seeds_own_tables_is_an_error(seeded):
    """Design rule 1: a rule that can never fire is a mistake, and the refusal says why rather
    than reading like a typo. The point is in the seed's tables only."""
    first = read_chart(seeded.refs[0].path)
    only_early = next(
        a.name
        for a in first.antigens
        if all(a.name not in [x.name for x in read_chart(t.path).antigens] for t in seeded.refs[3:])
    )
    rule = RemoveRule(what="antigens", name=only_early, reason="only in the seed's tables")
    with pytest.raises(SelectError, match="only tables the seed already holds"):
        run_chain(config(seeded, remove=[rule]), seeded.root / "x", StubOptimiser())


def test_rules_still_need_a_seed_chart_to_be_used_with_first_map(seeded):
    """Without `[seed_chart]` there are no post-seed tables to act on, so the refusal stands —
    and now says what to add."""
    with pytest.raises(ChainConfigError, match=r"select.remove needs \[seed_chart\]"):
        ChainConfig(
            GROUP,
            seeded.refs,
            OPTS,
            seed=3,
            first_map=seeded.seed,
            remove=[_rule(seeded.refs[4].path)],
        )


def test_append_still_needs_a_seed_chart_to_be_used_with_first_map(seeded):
    with pytest.raises(ChainConfigError, match=r"append-to-published needs \[seed_chart\]"):
        ChainConfig(
            GROUP,
            seeded.refs,
            OPTS,
            seed=3,
            first_map=seeded.seed,
            append={"late_tables": {}},
        )


def test_a_step_order_naming_an_unselected_table_is_an_error(seeded):
    with pytest.raises(ChainConfigError, match="not selected"):
        ChainConfig(
            GROUP,
            seeded.refs,
            OPTS,
            seed=3,
            first_map=seeded.seed,
            seed_chart=SeedChart(bridge=False),
            stepped=("no-such-table",),
        )


# ----------------------------------------------------------------------
# gap 6, through the store: a seeded month keeps its published order


@pytest.fixture
def monthly(tmp_path):
    """A seeded monthly chain: three tables published, the first two held by the seed, so the
    chain's first version steps the third alone."""
    publish(tmp_path / "store", [make_table(k) for k in range(3)])
    Work.create(tmp_path / "work")
    write_markers(tmp_path / "rules" / "sera" / "non_ferret_markers.tsv")
    seed = _make_seed(tmp_path, keep=2)
    chain = tmp_path / "chains" / "testlab" / "h9-hi-turkey-testlab" / "monthly.toml"
    chain.parent.mkdir(parents=True)
    chain.write_text(
        f'name = "h9-hi-turkey-testlab"\nseed = 1\nfirst_map = "{seed}"\n'
        f'[tables]\ndataset = "{DATASET}"\norder = "append-to-published"\n'
        '[options]\nminimum_column_basis = "none"\n'
        "scratch_starts = 3\nincremental_starts = 3\ngrid_test = false\n"
        "[seed_chart]\nbridge = false\n"
    )
    run = tmp_path / "run.toml"
    run.write_text(
        f'optimiser = "stub"\npublish = true\n[paths]\nstore = "{tmp_path / "store"}"\n'
        f'work = "{tmp_path / "work"}"\n[runner]\nkind = "local"\n'
    )
    assert chain_main([str(chain), str(run)]) == 0
    return SimpleNamespace(root=tmp_path, chain=chain, run=run, seed=seed)


def _make_seed(tmp_path, keep: int, name: str = "seed.ace"):
    """A seed over the store's first `keep` tables, made by a chain of its own (not published)."""
    store = Store.open(tmp_path / "store")
    inputs = tmp_path / "inputs"
    from af.chain.tables import tables_from_store

    _, refs = tables_from_store(store.root, DATASET, inputs, set())
    held = refs[:keep]
    out = tmp_path / f"pre-{name}"
    run_chain(ChainConfig("h9-hi-turkey-testlab", held, OPTS, seed=1), out, StubOptimiser())
    made = read_chart(out / "h9-hi-turkey-testlab" / "steps" / f"{keep - 1:04d}" / CHOSEN)
    path = tmp_path / name
    write_chart(made, path)
    return path


def _published(m):
    store = Store.open(m.root / "store")
    ref = store.current("chains", "testlab/h9-hi-turkey-testlab/monthly")
    return json.loads((store.resolve(ref) / "chain.json").read_text()), store.resolve(ref)


def test_a_seeded_month_steps_only_after_the_seed_then_appends(monthly):
    """The first version is the seed plus the one table it lacks; next month's table appends and
    every published step is reused."""
    first, _ = _published(monthly)
    assert [s["table_id"] for s in first["steps"]][1:] == ["h9-hi-turkey-testlab-20210315"]
    assert first["steps"][0]["chosen"] == "seed"
    publish(monthly.root / "store", [make_table(k) for k in range(4)])
    assert chain_main([str(monthly.chain), str(monthly.run)]) == 0
    doc, _ = _published(monthly)
    assert [s["table_id"] for s in doc["steps"]][1:] == [
        "h9-hi-turkey-testlab-20210315",
        "h9-hi-turkey-testlab-20210415",
    ]
    assert [s["reused"] for s in doc["steps"]] == [True, True, False]
    assert doc["config"]["append"]["prefix"] == 1
    assert doc["config"]["append"]["seed_tables"] == 2


def test_a_back_dated_table_appends_at_the_end_of_a_seeded_chain(monthly):
    """Q164 "cut-off only": an arrival dated before the published chain's last table joins after
    the published prefix rather than where its date falls, and is recorded as late. It must fall
    after the seed's last source to be a step at all, so it is late and post-seed at once."""
    late = dataclasses.replace(
        make_table(9), table_id="h9-hi-turkey-testlab-20210301", date="2021-03-01"
    )
    publish(monthly.root / "store", [*(make_table(k) for k in range(4)), late])
    assert chain_main([str(monthly.chain), str(monthly.run)]) == 0
    doc, vdir = _published(monthly)
    assert [s["table_id"][-4:] for s in doc["steps"]][1:] == ["0315", "0301", "0415"]
    assert set(doc["config"]["append"]["late_tables"]) == {"h9-hi-turkey-testlab-20210301"}
    late_step = json.loads((vdir / "steps/0002/step.json").read_text())["late"]
    assert late_step["date"] == "2021-03-01" and late_step["published_until"] == "2021-03-15"


def test_a_different_seed_stops_the_month(monthly):
    """Every step stands on the seed, so appending to a chain built on another map would rebase
    the published steps on a map they were not made from. Refused, with both hashes."""
    other = _make_seed(monthly.root, keep=3, name="other.ace")
    monthly.chain.write_text(monthly.chain.read_text().replace(f'"{monthly.seed}"', f'"{other}"'))
    publish(monthly.root / "store", [make_table(k) for k in range(4)])
    with pytest.raises(ChainConfigError, match="not the one the published chain"):
        load_chain_config(monthly.chain, monthly.root / "inputs", monthly.root / "store")


def test_the_same_seed_from_another_path_is_accepted(monthly):
    """The seed is checked by its content, not its path: a store synced to another machine moves
    the file and the chain goes on. Step 0 itself is remade, because its input is that path, but
    it is a copy of the seed with no optimisation and its output is the same bytes, so every
    table step is reused and the published maps stand."""
    moved = monthly.root / "moved-seed.ace"
    moved.write_bytes(monthly.seed.read_bytes())
    monthly.chain.write_text(monthly.chain.read_text().replace(f'"{monthly.seed}"', f'"{moved}"'))
    publish(monthly.root / "store", [make_table(k) for k in range(4)])
    assert chain_main([str(monthly.chain), str(monthly.run)]) == 0
    doc, _ = _published(monthly)
    assert [s["table_id"] for s in doc["steps"]][1:] == [
        "h9-hi-turkey-testlab-20210315",
        "h9-hi-turkey-testlab-20210415",
    ]
    assert [s["reused"] for s in doc["steps"]][1:] == [True, False]


def test_changed_step_parameters_stop_a_seeded_month(monthly):
    """check_settings compares each published step with today's. A seeded chain's tables start at
    step 1, so the comparison must be offset: without that it would read step 0, the seed, and
    report every parameter as changed."""
    monthly.chain.write_text(
        monthly.chain.read_text().replace("scratch_starts = 3", "scratch_starts = 5")
    )
    publish(monthly.root / "store", [make_table(k) for k in range(4)])
    with pytest.raises(SettingsChangedError, match="options") as e:
        chain_main([str(monthly.chain), str(monthly.run)])
    assert 'rerun_settings = [ { keys = ["options"]' in str(e.value)
    doc, _ = _published(monthly)  # nothing was published: the first version still stands
    assert [s["table_id"] for s in doc["steps"]][1:] == ["h9-hi-turkey-testlab-20210315"]
