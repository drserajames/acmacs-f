"""The serology build step: current tables store -> a new ``serology/all`` version.

Inputs are the CURRENT refs of every ``tables`` dataset, and the identity rules' version.
Each version records them as part of its own content, in ``inputs.json``: the tables
``dataset -> version`` it covers and the identity rules it was built with. So the store alone
says whether ``serology/all`` is current (:func:`stale`), wherever it is read: a report or map
build on another machine needs no work area of this step's (Sarah: refuse when stale).

Why content rather than PROVENANCE.json: store versions are content-addressed, and an
identical rebuild publishes no new version, so a version's provenance keeps the inputs of the
build that *first* made it and can lag behind the tables it covers. With the inputs in the
content, a new tables version always makes a new serology version (cheaply: unchanged
partitions are hard-linked), and the same tables in give the same version out.

Inside the build, unchanged partitions are hard-linked from the CURRENT serology version
(:mod:`af.serology.store`), so a new version costs only the partitions that changed.
"""

from __future__ import annotations

import datetime
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from af.serology.rows import IdentityRules
from af.serology.store import build
from af.store import Provenance, Store, StoreError, StoreRef
from af.tables.store import current_tables, read_table

KIND = "serology"
DATASET = "all"
STEP = "serology"
INPUTS = "inputs.json"
FIX = "run af.serology.update"


@dataclass
class UpdateResult:
    ref: StoreRef
    built: bool  # False when serology/all was already current
    reason: str
    changed: list[str] = field(default_factory=list)  # tables datasets new or changed
    report: dict[str, Any] = field(default_factory=dict)


def update(store: Store, rules: IdentityRules, *, force: bool = False) -> UpdateResult:
    """Bring ``serology/all`` up to date with the current tables store."""
    inputs = store.list_datasets("tables")
    if not inputs:
        raise StoreError("no tables datasets in the store: nothing to build serology from")
    current = _current_or_none(store)
    reasons = stale(store, rules)
    if not force and not reasons:
        assert current is not None  # stale() says so when there is none
        return UpdateResult(current, built=False, reason="inputs unchanged")
    covered = _covered(store, current)
    before = covered["tables"] if covered is not None else {}
    changed = [ref.dataset for ref in inputs if before.get(ref.dataset) != ref.version]

    started = _now()
    tables = [read_table(path) for _, path in current_tables(store).values()]
    previous = store.resolve(current) if current is not None else None
    with store.build(KIND, DATASET) as builder:
        report = build(tables, builder.path, rules, previous=previous)
        _write_inputs(builder.path / INPUTS, inputs, rules)
        ref = builder.publish(
            Provenance(
                step=STEP,
                inputs=tuple(inputs),
                parameters={"identity_version": rules.version},
                started=started,
                finished=_now(),
            ),
            summary={
                "tables": report.tables,
                "readings": report.readings,
                "rebuilt": len(report.rebuilt),
                "reused": len(report.reused),
            },
        )
    reason = "forced" if force else "; ".join(reasons)
    return UpdateResult(ref, built=True, reason=reason, changed=changed, report=report.__dict__)


def stale(store: Store, rules: IdentityRules) -> list[str]:
    """Why ``serology/all`` CURRENT does not cover the current tables store; empty if it does.

    One reason per tables dataset that is new, changed or removed since CURRENT was built,
    so a refusal can name them. The update step skips on exactly this test, so the guard and
    the step cannot disagree about what "current" means.
    """
    current = _current_or_none(store)
    if current is None:
        return ["no serology version yet"]
    covered = _covered(store, current)
    if covered is None:
        return [f"{current} has no {INPUTS} (built before af recorded its inputs)"]
    reasons = []
    if covered.get("identity_version") != rules.version:
        reasons.append("identity rules changed")
    now = {ref.dataset: ref.version for ref in store.list_datasets("tables")}
    was: dict[str, str] = covered["tables"]
    for dataset in sorted(now.keys() | was.keys()):
        if dataset not in was:
            reasons.append(f"tables {dataset}: new since {current.version}")
        elif dataset not in now:
            reasons.append(f"tables {dataset}: removed")
        elif now[dataset] != was[dataset]:
            reasons.append(f"tables {dataset}: {was[dataset]} -> {now[dataset]}")
    return reasons


def require_current(store: Store, rules: IdentityRules | None = None) -> None:
    """Refuse to read ``serology/all`` when it is behind the tables store.

    For every consumer (geo, stat, maps): a stale serology store silently leaves out a lab
    whose tables were published after it was built. ``rules`` default to the chain rules,
    which are what the serology step builds with.
    """
    if rules is None:
        from af.serology.rules import chain_rules

        rules = chain_rules()
    if reasons := stale(store, rules):
        raise StoreError(
            f"{KIND}/{DATASET} is behind the tables store ({FIX}): " + "; ".join(reasons)
        )


def _covered(store: Store, ref: StoreRef | None) -> dict[str, Any] | None:
    """The inputs a serology version records, or None for a version without them."""
    if ref is None:
        return None
    path = store.resolve(ref) / INPUTS
    if not path.is_file():
        return None
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return data


def _write_inputs(path: Path, inputs: list[StoreRef], rules: IdentityRules) -> None:
    """Sorted and without timestamps: the same inputs must give the same bytes (and version)."""
    doc = {
        "identity_version": rules.version,
        "tables": {ref.dataset: ref.version for ref in sorted(inputs, key=lambda r: r.dataset)},
    }
    path.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n", encoding="utf-8")


def _current_or_none(store: Store) -> StoreRef | None:
    if not (store.dataset_dir(KIND, DATASET) / "CURRENT").is_file():
        return None
    return store.current(KIND, DATASET)


def _now() -> datetime.datetime:
    return datetime.datetime.now(datetime.UTC)
