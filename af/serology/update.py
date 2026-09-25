"""The serology build step: current tables store -> a new ``serology/all`` version.

Inputs are the CURRENT refs of every ``tables`` dataset, and the identity rules' version.
The step keeps its own small record in the work area (``Work.dataset("serology",
"all").state``): the refs and rules it last built from. A run whose inputs equal that
record, with the serology version it produced still CURRENT, does nothing.

Why a record of its own rather than PROVENANCE.json: an identical rebuild publishes no new
version, so a version's provenance keeps the inputs of the build that *first* made it.
Diffing against it would rebuild on every run after one identical rebuild. The record is
rewritten on every successful run. It compares refs, not file hashes, so the log can say
which tables datasets changed.

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
from af.store import Provenance, Store, StoreError, StoreRef, Work
from af.tables.store import current_tables, read_table

KIND = "serology"
DATASET = "all"
STEP = "serology"
RECORD = "serology.json"


@dataclass
class UpdateResult:
    ref: StoreRef
    built: bool  # False when the record showed nothing had changed
    reason: str
    changed: list[str] = field(default_factory=list)  # tables datasets new or changed
    report: dict[str, Any] = field(default_factory=dict)


def update(store: Store, work: Work, rules: IdentityRules, *, force: bool = False) -> UpdateResult:
    """Bring ``serology/all`` up to date with the current tables store."""
    inputs = store.list_datasets("tables")
    if not inputs:
        raise StoreError("no tables datasets in the store: nothing to build serology from")
    state = work.dataset(KIND, DATASET).state
    record = _read_record(state)
    current = _current_or_none(store)
    wanted = _record(inputs, rules)
    changed = _changed(record, inputs)
    if not force and current is not None and record == {**wanted, "output": current.to_json()}:
        return UpdateResult(current, built=False, reason="inputs unchanged")
    reason = _why(force, current, record, wanted, changed)

    started = _now()
    tables = [read_table(path) for _, path in current_tables(store).values()]
    previous = store.resolve(current) if current is not None else None
    with store.build(KIND, DATASET) as builder:
        report = build(tables, builder.path, rules, previous=previous)
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
    _write_record(state, {**wanted, "output": ref.to_json()})
    return UpdateResult(ref, built=True, reason=reason, changed=changed, report=report.__dict__)


def _record(inputs: list[StoreRef], rules: IdentityRules) -> dict[str, Any]:
    return {"inputs": [ref.to_json() for ref in inputs], "identity_version": rules.version}


def _changed(record: dict[str, Any] | None, inputs: list[StoreRef]) -> list[str]:
    """Tables datasets that are new or at a different version since the last build."""
    before = {r["dataset"]: r["version"] for r in (record or {}).get("inputs", [])}
    return [ref.dataset for ref in inputs if before.get(ref.dataset) != ref.version]


def _why(
    force: bool,
    current: StoreRef | None,
    record: dict[str, Any] | None,
    wanted: dict[str, Any],
    changed: list[str],
) -> str:
    if force:
        return "forced"
    if current is None:
        return "no serology version yet"
    if record is None:
        return "no record of the last build"
    if record.get("identity_version") != wanted["identity_version"]:
        return "identity rules changed"
    if record.get("output") != current.to_json():
        return "CURRENT serology version is not the one last built"
    removed = {r["dataset"] for r in record["inputs"]} - {r["dataset"] for r in wanted["inputs"]}
    parts = []
    if changed:
        parts.append(f"tables changed: {', '.join(changed)}")
    if removed:
        parts.append(f"tables removed: {', '.join(sorted(removed))}")
    return "; ".join(parts) or "inputs changed"


def _current_or_none(store: Store) -> StoreRef | None:
    if not (store.dataset_dir(KIND, DATASET) / "CURRENT").is_file():
        return None
    return store.current(KIND, DATASET)


def _read_record(state: Path) -> dict[str, Any] | None:
    path = state / RECORD
    if not path.is_file():
        return None
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return data


def _write_record(state: Path, record: dict[str, Any]) -> None:
    """Write via a temporary file and rename, so a crash never leaves half a record."""
    path = state / RECORD
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(record, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _now() -> datetime.datetime:
    return datetime.datetime.now(datetime.UTC)
