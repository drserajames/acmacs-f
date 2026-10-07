"""The serology step against a real af.store and af.tables publish, all in tmp_path."""

import datetime
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from af.serology import query
from af.serology.rows import IdentityRules
from af.serology.update import update
from af.store import Provenance, Store, StoreError
from af.tables.identity import Manifest
from af.tables.model import Table
from af.tables.store import publish


def _tables(syn: Any) -> list[Table]:
    serum = {"name": syn.virus("Elsewhere", 2), "serum_id": "S-1"}
    a = {"name": syn.virus("Somewhere", 1), "passage": "MDCK1", "date": "2021-01-05"}
    b = {"name": syn.virus("Somewhere", 5), "passage": "SIAT1", "date": "2021-02-01"}
    return [
        syn.table("h3-hi-labx-20210304", [a], [serum], [[["80"]]]),
        syn.table("h3-hi-labx-20220110", [b], [serum], [[["40"]]], date="2022-01-10"),
        syn.table(
            "h3-hi-laby-20210201",
            [a, b],
            [serum],
            [[["160"]], [["<10"]]],
            date="2021-02-01",
            lab="LABY",
            group="h3-hi-laby",
        ),
    ]


def _publish(store: Store, tables: list[Table]) -> None:
    now = datetime.datetime.now(datetime.UTC)
    publish(
        store,
        tables,
        Manifest.from_tables(tables, inputs=[]),
        Provenance(step="tables-test", inputs=(), parameters={}, started=now, finished=now),
    )


@pytest.fixture
def store(tmp_path: Path) -> Store:
    return Store.create(tmp_path / "store")


def test_first_build_then_skip_then_incremental(store: Store, syn: Any) -> None:
    tables = _tables(syn)
    _publish(store, tables)

    first = update(store, syn.rules)
    assert first.built and first.reason == "no serology version yet"
    assert first.report["tables"] == 3 and first.report["readings"] == 4
    con = query.connect(store.resolve(first.ref))
    assert {p.first_lab for p in query.preparations(con)} == {"LABY"}

    again = update(store, syn.rules)
    assert not again.built and again.ref == first.ref and again.reason == "inputs unchanged"

    tables[1] = replace(tables[1], titres=[[[">1280"]]])  # one LABX table changes
    _publish(store, tables)
    third = update(store, syn.rules)
    assert third.built and third.changed == ["labx/h3-hi-labx"]
    assert third.report["rebuilt"] == ["h3-hi-labx/2022"]
    assert sorted(third.report["reused"]) == ["h3-hi-labx/2021", "h3-hi-laby/2021"]
    assert third.ref != first.ref
    assert store.current("serology", "all") == third.ref


def test_new_rules_rebuild_and_force(store: Store, syn: Any) -> None:
    _publish(store, _tables(syn))
    update(store, syn.rules)
    rules2 = IdentityRules(syn.rules.antigen, syn.rules.serum, version="test-2")
    result = update(store, rules2)
    assert result.built and result.reason == "identity rules changed"
    assert result.report["reused"] == []
    forced = update(store, rules2, force=True)
    assert forced.built and forced.reason == "forced"
    assert forced.ref == result.ref  # identical files: same version, nothing new published


def test_no_tables_is_an_error(store: Store, syn: Any) -> None:
    with pytest.raises(StoreError, match="no tables datasets"):
        update(store, syn.rules)


def test_chain_rules_are_the_chain_engines(syn: Any) -> None:
    from af.chart import identity
    from af.serology.rules import chain_rules

    rules = chain_rules()
    assert rules.antigen is identity.antigen_identity
    assert rules.serum is identity.serum_identity
    assert rules.version.startswith("af.chart.identity:") and rules.version == chain_rules().version
    # empty passage and DISTINCT mean no identity, as the chains treat them
    name = syn.virus("Somewhere", 1)
    assert rules.antigen(name, "", [], "") is None
    assert rules.antigen(name, "", ["DISTINCT"], "E3") is None
    assert rules.antigen(name, "", [], "E3 (2021-01-01)") is not None


def test_the_guard_names_each_stale_dataset_and_update_clears_it(store: Store, syn: Any) -> None:
    """A lab whose tables arrive after serology/all was built is refused by name, not dropped."""
    from af.serology.update import require_current, stale

    tables = _tables(syn)
    _publish(store, tables[:2])  # LABX only
    update(store, syn.rules)
    assert stale(store, syn.rules) == []
    require_current(store, syn.rules)

    _publish(store, tables)  # LABY's tables are published after serology/all was built
    reasons = stale(store, syn.rules)
    assert reasons == [
        f"tables laby/h3-hi-laby: new since {store.current('serology', 'all').version}"
    ]
    with pytest.raises(
        StoreError, match=r"behind the tables store \(run af.serology.update\).*laby"
    ):
        require_current(store, syn.rules)

    result = update(store, syn.rules)
    assert result.built and result.changed == ["laby/h3-hi-laby"]
    require_current(store, syn.rules)
    # the identity rules are part of what "current" means
    other = IdentityRules(syn.rules.antigen, syn.rules.serum, version="test-2")
    assert stale(store, other) == ["identity rules changed"]


def test_identical_rebuilds_leave_serology_current(store: Store, syn: Any) -> None:
    """An identical rebuild publishes no new version (the store is content-addressed). With
    the tables refs recorded as content, that version's record is still exactly right:
    forced with nothing changed, and tables changed then changed back (A -> B -> A)."""
    from af.serology.update import stale

    tables = _tables(syn)
    _publish(store, tables)
    first = update(store, syn.rules)
    forced = update(store, syn.rules, force=True)
    assert forced.ref == first.ref and stale(store, syn.rules) == []

    original = tables[1]
    tables[1] = replace(original, titres=[[[">1280"]]])  # B
    _publish(store, tables)
    changed = update(store, syn.rules)
    assert changed.ref != first.ref and stale(store, syn.rules) == []
    tables[1] = original  # back to A: the first version's files exactly
    _publish(store, tables)
    assert stale(store, syn.rules) == [
        f"tables labx/h3-hi-labx: {_labx(store, changed.ref)}"
        f" -> {store.current('tables', 'labx/h3-hi-labx').version}"
    ]
    restored = update(store, syn.rules)
    assert restored.built and restored.ref == first.ref  # republished, not a new version
    assert store.current("serology", "all") == first.ref and stale(store, syn.rules) == []


def _labx(store: Store, serology: Any) -> str:
    """The labx tables version a serology version records."""
    import json

    doc = json.loads((store.resolve(serology) / "inputs.json").read_text())
    version: str = doc["tables"]["labx/h3-hi-labx"]
    return version


def test_same_tables_give_the_same_version_byte_for_byte(tmp_path: Path, syn: Any) -> None:
    """Two stores built from the same tables agree on the version id and every file."""
    refs, trees = [], []
    for name in ("one", "two"):
        store = Store.create(tmp_path / name)
        _publish(store, _tables(syn))
        refs.append(update(store, syn.rules).ref)
        root = store.resolve(refs[-1])
        trees.append(
            {
                p.relative_to(root).as_posix(): p.read_bytes()
                for p in sorted(root.rglob("*"))
                if p.is_file() and p.name != "PROVENANCE.json"  # timestamps
            }
        )
    assert refs[0].version == refs[1].version
    assert trees[0] == trees[1] and "inputs.json" in trees[0]


def test_a_version_without_recorded_inputs_is_stale(store: Store, syn: Any) -> None:
    """Versions built before inputs.json existed count as stale; one update catches up."""
    from af.serology.store import build
    from af.serology.update import stale
    from af.store import Provenance as P

    tables = _tables(syn)
    _publish(store, tables)
    now = datetime.datetime.now(datetime.UTC)
    with store.build("serology", "all") as builder:
        build(tables, builder.path, syn.rules)  # the old layout: no inputs.json
        old = builder.publish(P(step="old", inputs=(), parameters={}, started=now, finished=now))
    assert stale(store, syn.rules) == [
        f"{old} has no inputs.json (built before af recorded its inputs)"
    ]
    caught_up = update(store, syn.rules)
    assert caught_up.built and caught_up.ref != old
    again = update(store, syn.rules)
    assert not again.built and again.reason == "inputs unchanged"


def _store_read(store: Store, ref: Any) -> dict[str, Any]:
    provenance = json.loads((store.resolve(ref) / "PROVENANCE.json").read_text())
    store_read: dict[str, Any] = provenance["parameters"]["store_read"]
    return store_read


def test_update_refuses_while_tables_are_being_published(store: Store, syn: Any) -> None:
    """A start in the middle of a several-dataset tables publish would build serology/all from
    a half-published set, and record that set as its inputs, so require_current could never
    catch it later. The guarded read refuses to start instead, and publishes nothing."""
    from af.store import StoreBusy

    _publish(store, _tables(syn))
    sweep = store.batch("tables-sweep", ["tables/labx/h3-hi-labx"])
    with sweep, pytest.raises(StoreBusy, match="tables-sweep"):
        update(store, syn.rules)
    assert not (store.dataset_dir("serology", "all") / "CURRENT").exists()


def test_a_batch_over_other_kinds_does_not_hold_the_update_off(store: Store, syn: Any) -> None:
    """The update reads only tables and serology/all, so a sequences sweep (which can hold the
    store for hours) must not hold it off; the provenance says which kinds were guarded and
    names the batch that was running, so a later reader sees it was not overridden."""
    _publish(store, _tables(syn))
    sweep = store.batch("sequences-sweep", ["sequences/h3", "sequences/h1"])
    with sweep:
        built = update(store, syn.rules)
    assert built.built
    seen = _store_read(store, built.ref)
    assert seen["kinds"] == ["serology", "tables"]
    assert seen["overrode_batches"] == []
    assert [m["name"] for m in seen["batches_of_other_kinds"]] == ["sequences-sweep"]


def test_provenance_says_guarded_or_overridden(store: Store, syn: Any) -> None:
    """The recorded guard is what tells a later reader a guarded build from an overridden one."""
    tables = _tables(syn)
    _publish(store, tables)
    guarded = update(store, syn.rules)
    seen = _store_read(store, guarded.ref)
    assert seen["read"] == "serology-update"
    assert seen["overrode_batches"] == []
    labx = store.current("tables", "labx/h3-hi-labx").version
    assert seen["currents_read"]["tables/labx/h3-hi-labx"] == [labx]

    tables[1] = replace(tables[1], titres=[[[">1280"]]])
    _publish(store, tables)
    with store.batch("tables-sweep", ["tables/laby/h3-hi-laby"]):
        overridden = update(store, syn.rules, ignore_busy=True)
    assert overridden.built
    names = [m["name"] for m in _store_read(store, overridden.ref)["overrode_batches"]]
    assert names == ["tables-sweep"]


def test_a_tables_current_moving_mid_update_fails_naming_it(
    store: Store, syn: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Tables published while the update reads: it fails at the end naming the dataset that
    moved and both its versions (the 2 Oct incident was only diagnosable because a report
    said which versions it read), and serology/all stays where it was."""
    import af.serology.update as step
    from af.store import StoreBusy

    tables = _tables(syn)
    _publish(store, tables)
    first = update(store, syn.rules)
    tables[1] = replace(tables[1], titres=[[[">1280"]]])
    _publish(store, tables)
    was = store.current("tables", "labx/h3-hi-labx").version

    real_build = step.build

    def build_while_tables_move(*args: Any, **kwargs: Any) -> Any:
        tables[1] = replace(tables[1], titres=[[["20"]]])
        _publish(store, tables)  # a lab's tables land mid-update
        return real_build(*args, **kwargs)

    monkeypatch.setattr(step, "build", build_while_tables_move)
    with pytest.raises(StoreBusy) as raised:
        update(store, syn.rules)
    now = store.current("tables", "labx/h3-hi-labx").version
    assert now != was
    assert f"tables/labx/h3-hi-labx moved from {was} to {now}" in str(raised.value)
    assert "serology-update" in str(raised.value)
    assert store.current("serology", "all") == first.ref  # nothing published
