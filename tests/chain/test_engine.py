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


def test_split_starts_give_the_same_chain(tables, tmp_path):
    """Starts run as 3 local jobs through af.run give exactly the single-process maps."""
    from af.chain.engine import SplitStarts, _chunks
    from af.run.local import LocalRunner

    assert _chunks(10, 3) == [(0, 4), (4, 3), (7, 3)]
    assert _chunks(2, 5) == [(0, 1), (1, 1)]
    single = run(config(tables), tmp_path / "a")
    split = run_chain(
        config(tables),
        tmp_path / "b",
        optimiser=StubOptimiser(),
        runner=LocalRunner(max_parallel=2),
        split=SplitStarts(chunks=3, work_dir=tmp_path / "work"),
    )
    assert [r.record["start_stresses"] for r in split] == [
        r.record["start_stresses"] for r in single
    ]
    assert [r.record["stress"] for r in split] == [r.record["stress"] for r in single]


def chain_toml(tmp_path, dataset="testlab/h9/main", extra=""):
    """A directory-tables chain config at <tmp>/chains/<dataset>.toml."""
    path = tmp_path / "chains" / f"{dataset}.toml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f'name = "{GROUP}"\nseed = 3\n[tables]\ndirectory = "{tmp_path / "tables"}"\n'
        f'group = "{GROUP}"\n{extra}'
        "[options]\nscratch_starts = 4\nincremental_starts = 3\ngrid_test = false\n"
    )
    return path


RUN_TOML = (
    'optimiser = "stub"\n'
    '[paths]\nstore = "store"\nwork = "work"\n[runner]\nkind = "local"\n[split]\nchunks = 2\n'
)


def test_command_line(tables, tmp_path):
    from af.chain.__main__ import main
    from af.store.store import Store
    from af.store.work import Work

    Work.create(tmp_path / "work")
    Store.create(tmp_path / "store")
    chain = chain_toml(tmp_path)
    (tmp_path / "run.toml").write_text(RUN_TOML)
    assert main([str(chain), str(tmp_path / "run.toml")]) == 0
    root = tmp_path / "work" / "chains" / "testlab" / "h9" / "main"
    doc = json.loads((root / "chain.json").read_text())
    assert doc["complete"] and len(doc["steps"]) == 4
    assert (root / "state").is_dir() and (root / "review" / "index.html").exists()
    ref = Store.open(tmp_path / "store").current("chains", "testlab/h9/main")
    assert (Store.open(tmp_path / "store").resolve(ref) / "steps" / "0003" / "chosen.ace").exists()
    assert main([str(chain), str(tmp_path / "run.toml"), "--review"]) == 0


def test_command_line_refuses_a_missing_work_area(tables, tmp_path):
    from af.chain.__main__ import main
    from af.store.ref import StoreError

    (tmp_path / "run.toml").write_text(
        'optimiser = "stub"\n[paths]\nstore = "store"\nwork = "typo"\n'
    )
    with pytest.raises(StoreError):
        main([str(chain_toml(tmp_path)), str(tmp_path / "run.toml")])


def test_dataset_comes_from_the_config_path(tmp_path):
    from af.chain.__main__ import dataset_from_path

    assert dataset_from_path(tmp_path / "chains" / "labx" / "h9" / "main.toml") == "labx/h9/main"
    # the nearest chains/ above the file counts, wherever the checkout lives
    nested = tmp_path / "chains" / "repo" / "chains" / "labx" / "h9" / "o-test.toml"
    assert dataset_from_path(nested) == "labx/h9/o-test"
    for bad in ("chain.toml", "chains/labx/main.toml", "chains/labx/h9/main/extra.toml"):
        with pytest.raises(ChainConfigError, match="must be at"):
            dataset_from_path(tmp_path / bad)


def test_tables_dataset_must_match_the_path(tables, tmp_path):
    from af.chain.__main__ import check_tables_dataset

    path = tmp_path / "chains" / "labx" / "h9" / "main.toml"
    path.parent.mkdir(parents=True)
    path.write_text('name = "x"\nseed = 1\n[tables]\ndataset = "labx/h9"\n')
    check_tables_dataset(path, "labx/h9/main")
    with pytest.raises(ChainConfigError, match="not this chain's"):
        check_tables_dataset(path, "laby/h9/main")


def test_run_toml_no_longer_names_the_dataset(tables, tmp_path):
    from af.chain.__main__ import main
    from af.store.work import Work
    from af.util.config import ConfigError

    Work.create(tmp_path / "work")
    (tmp_path / "run.toml").write_text('dataset = "testlab/h9/main"\n' + RUN_TOML)
    with pytest.raises(ConfigError, match="dataset: unknown key"):
        main([str(chain_toml(tmp_path)), str(tmp_path / "run.toml")])


def test_moved_store_and_tables_rerun_nothing(tables, tmp_path):
    run(config(tables), tmp_path / "s")
    shutil.move(tmp_path / "s", tmp_path / "moved-store")
    shutil.move(tables, tmp_path / "moved-tables")
    again = run(config(tmp_path / "moved-tables"), tmp_path / "moved-store")
    assert reused(again) == [True] * 4
