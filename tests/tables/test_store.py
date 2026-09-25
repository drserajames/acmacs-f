"""The tables store on af.store: per-group datasets, reconfirmation, stable ids across runs."""

from __future__ import annotations

import json
from pathlib import Path

from af.store import PathsConfig, Store, Work
from af.tables import cdc, identity
from af.tables.rules import Rules
from af.tables.store import INDEX, KIND, read_table
from af.tables.update import PUBLISHED, CDCInputs, TablesSettings, make_step, state_dir, update

from .conftest import write_rules
from .test_cdc import row, write_tsv


def settings(tmp_path: Path, rows: list[dict[str, str]]) -> tuple[TablesSettings, PathsConfig]:
    paths = PathsConfig(store=tmp_path / "store", work=tmp_path / "work")
    if not paths.store.exists():
        Store.create(paths.store)
        Work.create(paths.work)
    tables = TablesSettings(
        rules=write_rules(tmp_path / "rules", all_optional=True),
        run="cdc/all",
        cdc=CDCInputs(tsv=write_tsv(tmp_path / "cdc.tsv", rows)),
    )
    return tables, paths


H3 = dict(test_subtype="H3")
H1 = dict(test_subtype="H1 swl", test_id="9", test_date="2030-03-01")


def events(s: tuple[TablesSettings, PathsConfig]) -> dict[str, str]:
    data = json.loads((state_dir(s[1], s[0]) / PUBLISHED).read_text())
    return {entry["ref"]["dataset"]: entry["event"] for entry in data}


def test_first_run_publishes_one_dataset_per_group(tmp_path):
    s = settings(tmp_path, [row(**H3), row(**H1)])
    status, report = update(*s)
    assert status == 0, report
    assert events(s) == {
        "cdc/h1pdm-hi-turkey-cdc": "published",
        "cdc/h3-hi-guinea-pig-cdc": "published",
    }
    store = Store.open(s[1].store)
    ref = store.current(KIND, "cdc/h3-hi-guinea-pig-cdc")
    version = store.resolve(ref, verify=True)
    assert (version / "tables" / "h3-hi-guinea-pig-cdc-20300102.json").is_file()
    assert (version / "dumps" / "h3-hi-guinea-pig-cdc-20300102.txt").is_file()
    assert list(json.loads((version / INDEX).read_text())["tables"]) == [
        "h3-hi-guinea-pig-cdc-20300102"
    ]


def test_identical_rerun_is_reconfirmed(tmp_path):
    s = settings(tmp_path, [row(**H3), row(**H1)])
    update(*s)
    status, report = update(*s)
    assert status == 0 and "new 0, changed 0, removed 0, metadata only 0" in report
    assert set(events(s).values()) == {"reconfirmed"}


def test_one_changed_group_only(tmp_path):
    s = settings(tmp_path, [row(**H3), row(**H1)])
    update(*s)
    s = settings(tmp_path, [row(**H3, titer_value="320"), row(**H1)])
    status, report = update(*s)
    assert status == 0
    assert events(s) == {
        "cdc/h1pdm-hi-turkey-cdc": "reconfirmed",
        "cdc/h3-hi-guinea-pig-cdc": "published",
    }
    assert "  restart  h3-hi-guinea-pig-cdc from 2030-01-02" in report


def test_suffix_and_retired_ids_survive_across_runs(tmp_path):
    s = settings(tmp_path, [row(**H3, test_id="20"), row(**H3, test_id="30", titer_value="80")])
    update(*s)
    # test 20 disappears; test 30 keeps .2, and the base id is retired, never reused
    s = settings(
        tmp_path,
        [row(**H3, test_id="30", titer_value="80"), row(**H3, test_id="40", titer_value="40")],
    )
    status, report = update(*s)
    assert status == 0, report
    store = Store.open(s[1].store)
    index = json.loads(
        (store.resolve(store.current(KIND, "cdc/h3-hi-guinea-pig-cdc")) / INDEX).read_text()
    )
    ids = {e["source_key"]: t for t, e in index["tables"].items()}
    assert ids == {
        "CDC test_id 30": "h3-hi-guinea-pig-cdc-20300102.2",
        "CDC test_id 40": "h3-hi-guinea-pig-cdc-20300102.3",
    }
    assert list(index["retired"]) == ["h3-hi-guinea-pig-cdc-20300102"]


def test_errors_publish_nothing(tmp_path):
    s = settings(tmp_path, [row(**H3, titer_value="0")])  # HI 0: no rule, an error
    status, report = update(*s)
    assert status == 1 and any(line.startswith("ERROR") for line in report)
    assert Store.open(s[1].store).list_datasets(KIND) == []


def test_stored_table_reads_back_with_its_hash(tmp_path):
    s = settings(tmp_path, [row(**H3)])
    update(*s)
    store = Store.open(s[1].store)
    version = store.resolve(store.current(KIND, "cdc/h3-hi-guinea-pig-cdc"))
    stored = read_table(version / "tables" / "h3-hi-guinea-pig-cdc-20300102.json")
    (fresh,) = cdc.read(s[0].cdc.tsv, Rules(s[0].rules)).tables
    identity.assign([fresh], None)
    assert stored.content_hash() == fresh.content_hash()


def test_pipeline_step_names_its_inputs_by_role(tmp_path):
    tables, paths = settings(tmp_path, [row(**H3)])
    parameters = {"rules": str(tables.rules), "run": "cdc/all", "cdc": {"tsv": str(tables.cdc.tsv)}}
    step = make_step(parameters, base_dir=tmp_path, paths=paths)
    assert step.name == "tables-update" and set(step.inputs) == {"rules", "cdc_tsv"}
    assert step.outputs[0].path == paths.work / "tables" / "cdc" / "all" / "state" / PUBLISHED
