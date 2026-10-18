"""`[tables] order = "append-to-published"` (Q164, Q224): a monthly chain keeps its published order,
appends new tables (late ones recorded), and stops on a revised table until a re-run is decided."""

import dataclasses
import datetime
import json
from types import SimpleNamespace

import pytest

from af.chain.__main__ import main as chain_main
from af.chain.append import (
    OrderError,
    Published,
    Rerun,
    RevisedTablesError,
    append_order,
    table_changes,
)
from af.chain.config import ChainConfigError, TableSelection
from af.store.store import Store
from af.store.work import Work

from .sera_markers import write_markers
from .test_tables import DATASET, make_table, publish


def _t(table_id, month, day=15, suffix=1):
    return SimpleNamespace(table_id=table_id, date=datetime.date(2021, month, day), suffix=suffix)


JAN, FEB, MAR = _t("t-jan", 1), _t("t-feb", 2), _t("t-mar", 3)
FEB20, APR = _t("t-feb20", 2, 20), _t("t-apr", 4)
HASHES = {"t-jan": "h1", "t-feb": "h2", "t-mar": "h3", "t-feb20": "h4", "t-apr": "h5"}
PUBLISHED = Published("v1", [("t-jan", "h1"), ("t-feb", "h2"), ("t-mar", "h3")])


def _order(today, hashes=HASHES, reruns=(), published=PUBLISHED):
    return append_order("lab/x/monthly", published, today, hashes, list(reruns), lambda t: {})


# ---- the order ----------------------------------------------------------------------------------


def test_the_first_version_is_in_date_order():
    o = _order([JAN, FEB, MAR], published=None)
    assert [t.table_id for t in o.tables] == ["t-jan", "t-feb", "t-mar"] and o.late == {}


def test_new_tables_follow_the_published_order_and_late_ones_are_recorded():
    o = _order([JAN, FEB, FEB20, MAR, APR])  # today's selection, in date order
    assert [t.table_id for t in o.tables] == ["t-jan", "t-feb", "t-mar", "t-feb20", "t-apr"]
    assert set(o.late) == {"t-feb20"}  # dated before the published chain's latest table
    assert o.late["t-feb20"] == {"date": "2021-02-20", "published_until": "2021-03-15",
                                 "published_version": "v1"}  # fmt: skip


def test_a_published_table_no_longer_selected_stops_the_run():
    with pytest.raises(OrderError, match="1 table\\(s\\) of the published chain v1 are no longer"):
        _order([JAN, MAR, APR])


def test_a_revised_table_stops_the_run_and_says_what_a_rerun_costs():
    with pytest.raises(RevisedTablesError) as e:
        _order([JAN, FEB, MAR, APR], hashes={**HASHES, "t-feb": "h2-new"})
    [r] = e.value.revisions
    assert (r.table_id, r.position, r.from_end, r.rerun_steps) == ("t-feb", 1, 1, 3)
    assert 'map_hash = "h2-new"' in str(e.value) and "rerun_from = [" in str(e.value)
    assert "makes 3 steps, from the earliest, t-feb" in str(e.value)


def test_a_decided_rerun_keeps_the_table_in_place():
    revised = {**HASHES, "t-feb": "h2-new"}
    o = _order([JAN, FEB, MAR, APR], hashes=revised, reruns=[Rerun("t-feb", "h2-new", "test")])
    assert [t.table_id for t in o.tables] == ["t-jan", "t-feb", "t-mar", "t-apr"]
    [done] = o.reruns
    assert done["table_id"] == "t-feb" and done["decided"] == "test"


def test_an_acknowledgement_that_matches_nothing_is_an_error():
    with pytest.raises(OrderError, match="no such revision"):
        _order([JAN, FEB, MAR], reruns=[Rerun("t-feb", "h2-new", "test")])  # not revised
    with pytest.raises(OrderError, match="no such revision"):  # revised, but to another hash
        _order([JAN, FEB, MAR], hashes={**HASHES, "t-feb": "h2-new"},
               reruns=[Rerun("t-feb", "h2-other", "test")])  # fmt: skip


def test_the_setting_is_checked():
    with pytest.raises(ChainConfigError, match="order must be one of"):
        TableSelection(dataset="x", order="arrival")
    with pytest.raises(ChainConfigError, match="rerun_from goes with"):
        TableSelection(dataset="x", rerun_from=[Rerun("t", "h", "d")])


def test_what_changed_in_a_revised_table_is_counted():
    old = make_table(1)
    new = make_table(1, titre_shift=1)
    new.antigens = new.antigens[:-1]
    new.titres = new.titres[:-1]
    got = table_changes(old, new)
    assert got["antigens removed"] == 1 and got["cells changed"] > 0
    assert table_changes(old, make_table(1)) == {}


def test_a_field_the_map_reads_is_named_in_the_changes():
    """A serum's species, which the ferret-only policy reads, changes the map hash without
    changing a cell or a point: the refusal says which field, not just "map content"."""
    old, new = make_table(1), make_table(1)
    new.sera[0] = dataclasses.replace(new.sera[0], species="HUMAN")
    assert old.map_hash() != new.map_hash()
    assert table_changes(old, new) == {"sera species": 1}


# ---- end to end, through the store ---------------------------------------------------------------


@pytest.fixture
def monthly(tmp_path):
    """A monthly chain over three published tables, run once (its first version)."""
    publish(tmp_path / "store", [make_table(k) for k in range(3)])
    Work.create(tmp_path / "work")
    write_markers(tmp_path / "rules" / "sera" / "non_ferret_markers.tsv")
    chain = tmp_path / "chains" / "testlab" / "h9-hi-turkey-testlab" / "monthly.toml"
    chain.parent.mkdir(parents=True)
    chain.write_text(
        f'name = "h9-hi-turkey-testlab"\nseed = 1\n[tables]\ndataset = "{DATASET}"\n'
        'order = "append-to-published"\n[options]\nminimum_column_basis = "none"\n'
        "scratch_starts = 3\nincremental_starts = 3\ngrid_test = false\n"
    )
    run = tmp_path / "run.toml"
    run.write_text(
        f'optimiser = "stub"\npublish = true\n[paths]\nstore = "{tmp_path / "store"}"\n'
        f'work = "{tmp_path / "work"}"\n[runner]\nkind = "local"\n'
    )
    assert chain_main([str(chain), str(run)]) == 0
    return SimpleNamespace(root=tmp_path, chain=chain, run=run)


def _published(m):
    store = Store.open(m.root / "store")
    ref = store.current("chains", "testlab/h9-hi-turkey-testlab/monthly")
    return json.loads((store.resolve(ref) / "chain.json").read_text())


def _late_table():
    t = make_table(9)  # its own test antigens; dated between the published tables
    return dataclasses.replace(t, table_id="h9-hi-turkey-testlab-20210220", date="2021-02-20")


def test_a_month_appends_and_records_the_late_table(monthly):
    first = _published(monthly)
    assert [s["table_id"][-4:] for s in first["steps"]] == ["0115", "0215", "0315"]
    publish(monthly.root / "store", [*(make_table(k) for k in range(4)), _late_table()])
    assert chain_main([str(monthly.chain), str(monthly.run)]) == 0
    doc = _published(monthly)
    assert [s["table_id"][-4:] for s in doc["steps"]] == ["0115", "0215", "0315", "0220", "0415"]
    assert [s["reused"] for s in doc["steps"]] == [True, True, True, False, False]
    assert set(doc["config"]["append"]["late_tables"]) == {"h9-hi-turkey-testlab-20210220"}
    store = Store.open(monthly.root / "store")
    vdir = store.resolve(store.current("chains", "testlab/h9-hi-turkey-testlab/monthly"))
    late = json.loads((vdir / "steps/0003/step.json").read_text())["late"]
    assert late["date"] == "2021-02-20" and late["published_until"] == "2021-03-15"


def test_a_metadata_only_change_stops_nothing(monthly):
    publish(monthly.root / "store", [make_table(k, note="Q221 ids") for k in range(3)])
    assert chain_main([str(monthly.chain), str(monthly.run)]) == 0


def test_a_revised_table_stops_the_month_until_a_rerun_is_decided(monthly):
    tables = [make_table(0), make_table(1, titre_shift=1), make_table(2)]
    publish(monthly.root / "store", tables)
    with pytest.raises(RevisedTablesError) as e:
        chain_main([str(monthly.chain), str(monthly.run)])
    [r] = e.value.revisions
    assert r.position == 1 and r.from_end == 1 and r.rerun_steps == 2
    assert r.changes.get("cells changed", 0) > 0
    text = monthly.chain.read_text().replace(
        "[options]",
        f'[[tables.rerun_from]]\ntable = "{r.table_id}"\nmap_hash = "{r.new_hash}"\n'
        'decided = "test: re-run"\n\n[options]',
    )
    monthly.chain.write_text(text)
    assert chain_main([str(monthly.chain), str(monthly.run)]) == 0
    doc = _published(monthly)
    assert [s["reused"] for s in doc["steps"]] == [True, False, False]
    assert doc["config"]["append"]["reruns"][0]["decided"] == "test: re-run"
