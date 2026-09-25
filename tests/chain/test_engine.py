"""Chain engine on synthetic tables (stub optimiser: fast and deterministic)."""

import json
import shutil

import numpy as np
import pytest

from af.chain.backend import StubOptimiser
from af.chain.config import (
    ChainConfig,
    ChainConfigError,
    MapOptions,
    load_chain_config,
    tables_from_directory,
)
from af.chain.engine import run_chain
from af.chain.review import build_review
from af.chain.stub_optimiser import stress_and_gradient, stress_terms
from af.chain.synthetic import make_tables
from af.chart.ace import read_chart, write_chart
from af.chart.titre import Titre

GROUP = "h9-hi-test-lab"
OPTS = MapOptions(scratch_starts=4, incremental_starts=3, grid_test=False)


def config(d, **kw):
    return ChainConfig(GROUP, tables_from_directory(d, GROUP, **kw), OPTS, seed=3)


@pytest.fixture
def tables(tmp_path):
    make_tables(tmp_path / "tables", GROUP, n_tables=4)
    return tmp_path / "tables"


def run(cfg, store):
    return run_chain(cfg, store, optimiser=StubOptimiser())


def reused(results):
    return [r.reused for r in results]


def test_rerun_reuses_everything(tables, tmp_path):
    assert reused(run(config(tables), tmp_path / "s")) == [False] * 4
    assert reused(run(config(tables), tmp_path / "s")) == [True] * 4


def test_changed_table_restarts_at_its_step(tables, tmp_path):
    first = run(config(tables), tmp_path / "s")
    path = sorted(tables.glob("*.ace"))[2]
    c = read_chart(path)
    c.titres.table[7][1] = Titre.parse("2560")
    write_chart(c, path)
    second = run(config(tables), tmp_path / "s")
    assert reused(second) == [True, True, False, False]
    assert first[1].record == second[1].record
    assert second[2].record["table_sha256"] != first[2].record["table_sha256"]


def test_inserted_table_restarts_from_its_date(tables, tmp_path):
    run(config(tables), tmp_path / "s")
    extra = tmp_path / "extra"
    make_tables(extra, GROUP, n_tables=6, seed=9)
    inserted = sorted(extra.glob("*.ace"))[5]  # a date after the others ...
    shutil.copy(
        inserted, tables / f"{GROUP}-20210315.2.ace"
    )  # ... filed as a second table on step 2's date
    second = run(config(tables), tmp_path / "s")
    assert [r.record["table_id"] for r in second][2:4] == [
        f"{GROUP}-20210315",
        f"{GROUP}-20210315.2",
    ]
    assert reused(second) == [True, True, True, False, False]


def test_removed_table_leaves_a_stale_directory_listed(tables, tmp_path):
    run(config(tables), tmp_path / "s")
    run(config(tables, exclude={f"{GROUP}-20210415"}), tmp_path / "s")
    doc = json.loads((tmp_path / "s" / GROUP / "chain.json").read_text())
    assert [s["table_id"] for s in doc["steps"]][-1] == f"{GROUP}-20210315"
    assert doc["stale_step_directories"] == ["0003"] and doc["complete"]


def test_same_inputs_same_maps(tables, tmp_path):
    a = run(config(tables), tmp_path / "a")
    b = run(config(tables), tmp_path / "b")
    assert [r.record["stress"] for r in a] == [r.record["stress"] for r in b]


def test_damaged_output_is_remade_and_later_steps_kept(tables, tmp_path):
    first = run(config(tables), tmp_path / "s")
    (first[1].directory / "scratch.ace").write_bytes(b"{}")
    again = run(config(tables), tmp_path / "s")
    # step 1 is remade identically (seeded from its inputs), so steps 2-3 keep their input hashes
    assert reused(again) == [True, False, True, True]


def test_exclusion_matching_nothing_is_an_error(tables):
    with pytest.raises(ChainConfigError):
        tables_from_directory(tables, GROUP, exclude={f"{GROUP}-19990101"})


def test_toml_config(tables, tmp_path):
    (tmp_path / "chain.toml").write_text(
        f'name = "{GROUP}"\nseed = 3\n[tables]\ndirectory = "tables"\ngroup = "{GROUP}"\n'
        'date_from = "2021-02-01"\n[options]\nscratch_starts = 4\nincremental_starts = 3\n'
    )
    cfg = load_chain_config(tmp_path / "chain.toml")
    assert [t.table_id for t in cfg.tables][0] == f"{GROUP}-20210215" and len(cfg.tables) == 3
    assert cfg.options.scratch_starts == 4 and cfg.options.dimensions == 2


def test_review_page(tables, tmp_path):
    run(config(tables), tmp_path / "s")
    page = build_review(tmp_path / "s" / GROUP)
    text = page.read_text()
    assert "Flagged steps" in text and text.count('<tr id="step-') == 4
    assert len(list((page.parent / "thumbs").glob("*.png"))) == 4


def test_stub_gradient_matches_finite_differences(tables):
    c = read_chart(sorted(tables.glob("*.ace"))[0])
    arrays = c.optimiser_arrays(disconnect_threshold=None)
    t = stress_terms(arrays)
    x = np.random.default_rng(0).normal(0, 2, 2 * c.n_points)
    _, g = stress_and_gradient(x, t, 2)
    h = 1e-6
    for i in range(0, len(x), 5):
        e = np.zeros_like(x)
        e[i] = h
        num = (stress_and_gradient(x + e, t, 2)[0] - stress_and_gradient(x - e, t, 2)[0]) / (2 * h)
        assert num == pytest.approx(g[i], rel=1e-5, abs=1e-6)
