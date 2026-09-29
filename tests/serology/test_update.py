"""The serology step against a real af.store and af.tables publish, all in tmp_path."""

import datetime
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
