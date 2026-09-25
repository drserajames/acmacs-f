"""Chain input from the tables store (synthetic af tables, invented names)."""

import datetime
import json

import pytest

from af.chain.backend import StubOptimiser
from af.chain.config import ChainConfig, ChainConfigError, MapOptions
from af.chain.engine import run_chain
from af.chain.tables import cell_titre, table_chart, tables_from_store
from af.store.store import Provenance, Store
from af.tables.model import Antigen, Serum, Table, canonical_json

DATASET = "testlab/h9-hi-turkey-testlab"


def make_table(k: int, titre_shift: int = 0, note: str = "") -> Table:
    sera = [
        Serum(f"TEST-SR-{j}", f"TEST-SR-{j}", serum_id=f"LOT{j}", passage="E3") for j in range(4)
    ]
    antigens = [
        Antigen(
            f"TEST-{i}", f"TEST-{i}", passage="MDCK1", passage_date="2021-01-0" + str(1 + i % 5)
        )
        for i in range(6)
    ] + [Antigen(f"TEST-{100 * k + i}", f"TEST-{100 * k + i}", passage="E2") for i in range(3)]
    for a in antigens:
        a.source = {"note": note}
    titres = [
        [[str(10 * 2 ** ((i + j + titre_shift) % 7 + 1))] for j in range(4)]
        for i in range(len(antigens))
    ]
    titres[0][0] = ["40", "80"]  # a repeat reading
    return Table(
        table_id=f"h9-hi-turkey-testlab-2021{k + 1:02d}15",
        group="h9-hi-turkey-testlab",
        lab="TESTLAB",
        subtype="A(H9N2)",
        lineage="",
        assay="HI",
        rbc="turkey",
        date=f"2021-{k + 1:02d}-15",
        date_suffix=1,
        source_key=f"test {k}",
        antigens=antigens,
        sera=sera,
        titres=titres,
    )


def publish(root, tables):
    store = Store.open(root) if (root / "STORE.toml").exists() else Store.create(root)
    with store.build("tables", DATASET) as build:
        index = {}
        for t in tables:
            (build.path / "tables").mkdir(exist_ok=True)
            (build.path / "tables" / f"{t.table_id}.json").write_text(canonical_json(t.to_json()))
            index[t.table_id] = {
                "table_id": t.table_id,
                "lab": t.lab,
                "hash": t.content_hash(),
                "map_hash": t.map_hash(),
                "source_key": t.source_key,
                "group": t.group,
                "date": t.date,
                "date_suffix": t.date_suffix,
            }
        (build.path / "index.json").write_text(
            json.dumps({"format": 1, "tables": index, "retired": {}})
        )
        now = datetime.datetime.now(datetime.UTC)
        build.publish(Provenance(step="test", inputs=(), parameters={}, started=now, finished=now))


def test_cell_titre_merges_repeats():
    assert str(cell_titre([])) == "*"
    assert str(cell_titre(["<10"])) == "<10"
    assert str(cell_titre(["40", "80"])) == "57"


def test_table_chart_uses_ae_passages():
    chart = table_chart(make_table(0))
    assert chart.antigens[0].passage == "MDCK1 (2021-01-01)" and chart.antigens[6].passage == "E2"
    assert str(chart.titres.table[0][0]) == "57" and chart.info["D"] == "20210115"


def test_only_map_changes_restart(tmp_path):
    publish(tmp_path / "store", [make_table(k) for k in range(4)])
    opts = MapOptions(scratch_starts=3, incremental_starts=2, grid_test=False)

    def run():
        _, tables = tables_from_store(tmp_path / "store", DATASET, tmp_path / "inputs")
        cfg = ChainConfig("h9-hi-turkey-testlab", tables, opts, seed=1)
        return [r.reused for r in run_chain(cfg, tmp_path / "chains", optimiser=StubOptimiser())]

    assert run() == [False] * 4
    # lab metadata changes (content hash) but the map does not: nothing re-runs
    publish(
        tmp_path / "store", [make_table(k, note="sequenced" if k == 1 else "") for k in range(4)]
    )
    assert run() == [True] * 4
    # a titre changes in table 2: steps 2 and 3 re-run
    publish(tmp_path / "store", [make_table(k, titre_shift=1 if k == 2 else 0) for k in range(4)])
    assert run() == [True, True, False, False]


def test_store_exclusion_must_match(tmp_path):
    publish(tmp_path / "store", [make_table(0)])
    with pytest.raises(ChainConfigError):
        tables_from_store(tmp_path / "store", DATASET, tmp_path / "inputs", {"no-such-table"})
