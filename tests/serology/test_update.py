"""The serology step against a real af.store and af.tables publish, all in tmp_path."""

import datetime
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from af.serology import query
from af.serology.rows import IdentityRules
from af.serology.update import update
from af.store import Provenance, Store, StoreError, Work
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
def roots(tmp_path: Path) -> tuple[Store, Work]:
    return Store.create(tmp_path / "store"), Work.create(tmp_path / "work")


def test_first_build_then_skip_then_incremental(roots: tuple[Store, Work], syn: Any) -> None:
    store, work = roots
    tables = _tables(syn)
    _publish(store, tables)

    first = update(store, work, syn.rules)
    assert first.built and first.reason == "no serology version yet"
    assert first.report["tables"] == 3 and first.report["readings"] == 4
    con = query.connect(store.resolve(first.ref))
    assert {p.first_lab for p in query.preparations(con)} == {"LABY"}

    again = update(store, work, syn.rules)
    assert not again.built and again.ref == first.ref and again.reason == "inputs unchanged"

    tables[1] = replace(tables[1], titres=[[[">1280"]]])  # one LABX table changes
    _publish(store, tables)
    third = update(store, work, syn.rules)
    assert third.built and third.changed == ["labx/h3-hi-labx"]
    assert third.report["rebuilt"] == ["h3-hi-labx/2022"]
    assert sorted(third.report["reused"]) == ["h3-hi-labx/2021", "h3-hi-laby/2021"]
    assert third.ref != first.ref
    assert store.current("serology", "all") == third.ref


def test_new_rules_rebuild_and_force(roots: tuple[Store, Work], syn: Any) -> None:
    store, work = roots
    _publish(store, _tables(syn))
    update(store, work, syn.rules)
    rules2 = IdentityRules(syn.rules.antigen, syn.rules.serum, version="test-2")
    result = update(store, work, rules2)
    assert result.built and result.reason == "identity rules changed"
    assert result.report["reused"] == []
    forced = update(store, work, rules2, force=True)
    assert forced.built and forced.reason == "forced"
    assert forced.ref == result.ref  # identical files: same version, nothing new published


def test_no_tables_is_an_error(roots: tuple[Store, Work], syn: Any) -> None:
    store, work = roots
    with pytest.raises(StoreError, match="no tables datasets"):
        update(store, work, syn.rules)


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
