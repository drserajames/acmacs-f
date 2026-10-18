"""`[tables] order = "append-to-published"` (Q164, Q224): a monthly chain keeps its published order,
appends new tables (late ones recorded), and stops on anything that would remake a published
step (a revised table, a dropped table, changed step parameters) until a re-run is decided."""

import dataclasses
import datetime
import json
from types import SimpleNamespace

import pytest

from af.chain.__main__ import main as chain_main
from af.chain.append import (
    DroppedTablesError,
    OrderError,
    Published,
    Rerun,
    RevisedTablesError,
    SettingsChangedError,
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
SHAS = {"t-jan": "s1", "t-feb": "s2", "t-mar": "s3", "t-feb20": "s4", "t-apr": "s5"}
PUBLISHED = Published("v1", [("t-jan", "s1"), ("t-feb", "s2"), ("t-mar", "s3")])


def _order(today, shas=SHAS, reruns=(), published=PUBLISHED, why=("retired", "gone")):
    return append_order("lab/x/monthly", published, today, shas, list(reruns),
                        lambda t: {"cells changed": 1}, lambda t: why)  # fmt: skip


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


def test_a_revised_table_stops_the_run_and_says_what_a_rerun_costs():
    with pytest.raises(RevisedTablesError) as e:
        _order([JAN, FEB, MAR, APR], shas={**SHAS, "t-feb": "s2-new"})
    [r] = e.value.revisions
    assert (r.table_id, r.position, r.from_end, r.rerun_steps) == ("t-feb", 1, 1, 3)
    assert 'table_sha256 = "s2-new"' in str(e.value) and "rerun_from = [" in str(e.value)
    assert "makes 3 steps, from the earliest, t-feb" in str(e.value)


def test_a_decided_rerun_keeps_the_revised_table_in_place():
    revised = {**SHAS, "t-feb": "s2-new"}
    o = _order([JAN, FEB, MAR, APR], shas=revised,
               reruns=[Rerun("t-feb", "test", table_sha256="s2-new")])  # fmt: skip
    assert [t.table_id for t in o.tables] == ["t-jan", "t-feb", "t-mar", "t-apr"]
    [done] = o.reruns
    assert done["table_id"] == "t-feb" and done["decided"] == "test"


def test_a_dropped_table_stops_the_run_with_its_kind_until_acknowledged():
    with pytest.raises(DroppedTablesError) as e:
        _order([JAN, MAR, APR], why=("excluded", "outside the dates"))
    [d] = e.value.dropped
    assert d["table_id"] == "t-feb" and d["kind"] == "excluded" and d["position"] == 1
    assert 'dropped = "excluded"' in str(e.value)
    o = _order([JAN, MAR, APR], why=("excluded", "outside the dates"),
               reruns=[Rerun("t-feb", "test", dropped="excluded")])  # fmt: skip
    assert [t.table_id for t in o.tables] == ["t-jan", "t-mar", "t-apr"] and o.prefix == 2
    with pytest.raises(OrderError):  # acknowledged as the wrong kind: matches nothing pending
        _order([JAN, MAR], why=("excluded", "x"), reruns=[Rerun("t-feb", "t", dropped="retired")])


def test_an_acknowledgement_that_matches_nothing_is_an_error():
    with pytest.raises(OrderError, match="nothing such is pending"):
        _order([JAN, FEB, MAR], reruns=[Rerun("t-feb", "test", table_sha256="s2-new")])
    with pytest.raises(OrderError, match="nothing such is pending"):  # to another chart
        _order([JAN, FEB, MAR], shas={**SHAS, "t-feb": "s2-new"},
               reruns=[Rerun("t-feb", "test", table_sha256="s2-other")])  # fmt: skip


def test_the_settings_are_checked():
    with pytest.raises(ChainConfigError, match="order must be one of"):
        TableSelection(dataset="x", order="arrival")
    with pytest.raises(ChainConfigError, match="go with order"):
        TableSelection(dataset="x", rerun_from=[Rerun("t", "d", table_sha256="h")])
    with pytest.raises(OrderError, match="exactly one of"):
        Rerun("t", "d")
    with pytest.raises(OrderError, match="dropped must be one of"):
        Rerun("t", "d", dropped="gone")


# ---- what changed --------------------------------------------------------------------------------


def test_what_changed_in_a_revised_table_is_counted():
    old, new = make_table(1), make_table(1, titre_shift=1)
    new.antigens, new.titres = new.antigens[:-1], new.titres[:-1]
    got = table_changes(old, new)
    assert got["antigens removed"] == 1 and got["cells changed"] > 0
    assert table_changes(old, make_table(1)) == {}


def test_a_field_the_map_reads_is_named_in_the_changes():
    old, new = make_table(1), make_table(1)
    new.sera[0] = dataclasses.replace(new.sera[0], species="HUMAN")
    assert table_changes(old, new) == {"sera species": 1}


def test_a_corrected_name_reads_as_the_name_not_a_point_swapped():
    old, new = make_table(1), make_table(1)
    new.antigens[7] = dataclasses.replace(new.antigens[7], name="TEST-RENAMED")
    assert table_changes(old, new) == {"antigens name": 1}


def test_a_repeat_in_the_test_is_compared_too():
    """Two antigens with one key: the second copy's change is not lost behind the first's."""
    old, new = make_table(1), make_table(1)
    for t in (old, new):
        t.antigens.append(dataclasses.replace(t.antigens[0]))
        t.titres.append([list(r) for r in t.titres[0]])
    new.titres[-1][1] = ["5120"]
    assert table_changes(old, new) == {"cells changed": 1}


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


def _run(m):
    return chain_main([str(m.chain), str(m.run)])


def _published(m):
    store = Store.open(m.root / "store")
    ref = store.current("chains", "testlab/h9-hi-turkey-testlab/monthly")
    return json.loads((store.resolve(ref) / "chain.json").read_text()), store.resolve(ref)


def _tables_block(m, text):
    m.chain.write_text(m.chain.read_text().replace("[options]", text + "\n[options]"))


def test_a_month_appends_and_records_the_late_table(monthly):
    first, _ = _published(monthly)
    assert [s["table_id"][-4:] for s in first["steps"]] == ["0115", "0215", "0315"]
    late = dataclasses.replace(make_table(9), table_id="h9-hi-turkey-testlab-20210220",
                               date="2021-02-20")  # fmt: skip
    publish(monthly.root / "store", [*(make_table(k) for k in range(4)), late])
    assert _run(monthly) == 0
    doc, vdir = _published(monthly)
    assert [s["table_id"][-4:] for s in doc["steps"]] == ["0115", "0215", "0315", "0220", "0415"]
    assert [s["reused"] for s in doc["steps"]] == [True, True, True, False, False]
    assert set(doc["config"]["append"]["late_tables"]) == {"h9-hi-turkey-testlab-20210220"}
    record = json.loads((vdir / "steps/0003/step.json").read_text())["late"]
    assert record["date"] == "2021-02-20" and record["published_until"] == "2021-03-15"


def test_a_change_the_chart_never_sees_stops_nothing(monthly):
    """Metadata (03's EPI_ISL ids) and map-hash fields the chart does not read (an antigen's
    reference flag) leave every published step's key alone."""
    tables = [make_table(k, note="Q221 ids") for k in range(3)]
    tables[1].antigens[0] = dataclasses.replace(tables[1].antigens[0], reference=True)
    assert tables[1].map_hash() != make_table(1).map_hash()
    publish(monthly.root / "store", tables)
    assert _run(monthly) == 0
    doc, _ = _published(monthly)
    assert all(s["reused"] for s in doc["steps"])


def test_a_revised_table_stops_the_month_until_a_rerun_is_decided(monthly):
    publish(monthly.root / "store", [make_table(0), make_table(1, titre_shift=1), make_table(2)])
    with pytest.raises(RevisedTablesError) as e:
        _run(monthly)
    [r] = e.value.revisions
    assert r.position == 1 and r.from_end == 1 and r.rerun_steps == 2
    assert r.changes.get("cells changed", 0) > 0
    ack = f'table = "{r.table_id}"\ntable_sha256 = "{r.new_sha256}"\ndecided = "test: re-run"\n'
    _tables_block(monthly, "[[tables.rerun_from]]\n" + ack)
    assert _run(monthly) == 0
    doc, _ = _published(monthly)
    assert [s["reused"] for s in doc["steps"]] == [True, False, False]
    assert doc["config"]["append"]["reruns"][0]["decided"] == "test: re-run"


def test_an_excluded_table_stops_the_month_until_acknowledged(monthly):
    first, _ = _published(monthly)
    gone = first["steps"][1]["table_id"]
    monthly.chain.write_text(
        monthly.chain.read_text().replace(
            'order = "append-to-published"', f'order = "append-to-published"\nexclude = ["{gone}"]'
        )
    )
    with pytest.raises(DroppedTablesError, match="excluded by the chain file"):
        _run(monthly)
    _tables_block(monthly, f'[[tables.rerun_from]]\ntable = "{gone}"\ndropped = "excluded"\n'
                           'decided = "test: leave it out"\n')  # fmt: skip
    assert _run(monthly) == 0
    doc, _ = _published(monthly)
    assert [s["table_id"] for s in doc["steps"]] == [first["steps"][0]["table_id"],
                                                     first["steps"][2]["table_id"]]  # fmt: skip
    assert [s["reused"] for s in doc["steps"]] == [True, False]


def test_changed_step_parameters_stop_the_month_until_accepted(monthly):
    monthly.chain.write_text(monthly.chain.read_text().replace("scratch_starts = 3",
                                                               "scratch_starts = 4"))  # fmt: skip
    with pytest.raises(SettingsChangedError, match="options") as e:
        _run(monthly)
    assert 'rerun_settings = [ { keys = ["options"]' in str(e.value)
    _tables_block(monthly, '[[tables.rerun_settings]]\nkeys = ["options"]\ndecided = "test"\n')
    assert _run(monthly) == 0
    doc, _ = _published(monthly)
    assert not any(s["reused"] for s in doc["steps"])
    with pytest.raises(OrderError, match="did not change"):  # accepted again, unchanged now
        _run(monthly)


def test_a_moved_sequence_decision_is_named_in_the_settings_refusal():
    """03's case: a metadata version or a pin moves a remove_by_sequence decision, which re-keys
    every step; the refusal names what moved."""
    from af.chain.append import _setting_detail

    was = {"rules": [1], "removed": [["antigen", "TEST-1 E1", "not-deleted"]]}
    now = {"rules": [1], "removed": [["antigen", "TEST-2 E1", "tie-none-deleted"]]}
    text = _setting_detail("select_remove_by_sequence", was, now)
    assert "2 decision(s) moved" in text
    assert "now removed: antigen | TEST-2 E1 | tie-none-deleted" in text
    assert "now kept: antigen | TEST-1 E1 | not-deleted" in text
