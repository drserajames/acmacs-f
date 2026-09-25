"""Stable table ids, the tables manifest, and the diff between two manifests.

A table's id is ``<group>-<YYYYMMDD>`` plus ``.2``, ``.3`` ... when several tables in a group
share a test date. The suffix is fixed the first time a source table is seen and recorded in
the manifest, so a table arriving later (or read in a different order) never renumbers the
ones already there. With no previous manifest, same-day tables are numbered in source-key
order (CDC test_id; xlsx file + sheet).

A suffix, once given, is never reused, even after its table disappears from the source: a
reused id would make a removed table look "changed" instead.

The diff says which tables are new, changed (same id, different content hash) or removed,
and for each group the earliest date affected: the chain engine restarts from there.
"""

from __future__ import annotations

import datetime as dt
import json
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .model import Table

MANIFEST_FORMAT = "af-tables-manifest-1"


def base_id(group: str, date: str) -> str:
    return f"{group}-{date.replace('-', '')}"


def make_id(group: str, date: str, suffix: int) -> str:
    return base_id(group, date) + ("" if suffix == 1 else f".{suffix}")


def _source_order(key: str) -> tuple[Any, ...]:
    """Natural order: "CDC test_id 987" before "CDC test_id 1001"."""
    return tuple(int(p) if p.isdigit() else p for p in re.split(r"(\d+)", key))


def assign(tables: list[Table], previous: Manifest | None) -> list[str]:
    """Give every table its stable id and suffix, in place. Returns problems found."""
    problems = []
    known: dict[str, str] = {}  # source_key -> table_id, from the previous manifest
    used: dict[str, set[int]] = {}  # base id -> suffixes ever given
    if previous is not None:
        for entry in [*previous.tables.values(), *previous.retired.values()]:
            known[entry["source_key"]] = entry["table_id"]
            used.setdefault(base_id(entry["group"], entry["date"]), set()).add(entry["date_suffix"])
    seen: dict[str, Table] = {}
    for table in sorted(tables, key=lambda t: (t.group, t.date, _source_order(t.source_key))):
        if table.source_key in seen:
            problems.append(f"source key {table.source_key!r} read twice")
            continue
        seen[table.source_key] = table
        base = base_id(table.group, table.date)
        if (old_id := known.get(table.source_key)) is not None:
            old_base, _, old_suffix = old_id.partition(".")
            if old_base == base:
                table.date_suffix = int(old_suffix or 1)
                table.table_id = old_id
                continue
            # The source moved the test to another date or group: it becomes a new table
            # there; the old id is reported removed by the diff.
            problems.append(f"{table.source_key}: was {old_id}, now in {base}")
        taken = used.setdefault(base, set())
        suffix = 1
        while suffix in taken:
            suffix += 1
        taken.add(suffix)
        table.date_suffix = suffix
        table.table_id = make_id(table.group, table.date, suffix)
    return problems


@dataclass
class Manifest:
    """What the store holds: table id -> hash and identity fields. JSON on disk."""

    tables: dict[str, dict[str, Any]] = field(default_factory=dict)
    retired: dict[str, dict[str, Any]] = field(default_factory=dict)  # ids that must not be reused
    inputs: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_tables(
        cls, tables: Iterable[Table], inputs: list[dict[str, Any]], previous: Manifest | None = None
    ) -> Manifest:
        manifest = cls(inputs=inputs)
        for t in tables:
            manifest.tables[t.table_id] = {
                "table_id": t.table_id,
                "hash": t.content_hash(),
                "map_hash": t.map_hash(),
                "source_key": t.source_key,
                "group": t.group,
                "date": t.date,
                "date_suffix": t.date_suffix,
            }
        if previous is not None:
            for table_id, entry in [*previous.tables.items(), *previous.retired.items()]:
                if table_id not in manifest.tables:
                    manifest.retired[table_id] = entry
        return manifest

    def write(self, path: Path) -> None:
        data = {
            "format": MANIFEST_FORMAT,
            "inputs": self.inputs,
            "tables": dict(sorted(self.tables.items())),
            "retired": dict(sorted(self.retired.items())),
        }
        path.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    @classmethod
    def read(cls, path: Path) -> Manifest:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("format") != MANIFEST_FORMAT:
            raise ValueError(f"{path}: not an {MANIFEST_FORMAT} file")
        return cls(tables=data["tables"], retired=data["retired"], inputs=data["inputs"])


@dataclass
class Diff:
    new: list[str]
    changed: list[str]  # the map content changed: chains restart here
    removed: list[str]
    metadata: list[str]  # only non-map fields changed (e.g. EPI_ISL filled in): no restart
    restart: dict[str, str]  # group -> earliest date with a new, changed or removed table

    def is_empty(self) -> bool:
        return not (self.new or self.changed or self.removed or self.metadata)

    def report(self) -> list[str]:
        lines = [
            f"new {len(self.new)}, changed {len(self.changed)}, removed {len(self.removed)}, "
            f"metadata only {len(self.metadata)}"
        ]
        for what, ids in (
            ("new", self.new),
            ("changed", self.changed),
            ("removed", self.removed),
            ("metadata", self.metadata),
        ):
            lines.extend(f"  {what:8s} {i}" for i in ids)
        lines.extend(f"  restart  {g} from {d}" for g, d in sorted(self.restart.items()))
        return lines


def diff(old: Manifest, new: Manifest) -> Diff:
    new_ids = sorted(set(new.tables) - set(old.tables))
    removed = sorted(set(old.tables) - set(new.tables))
    both = set(old.tables) & set(new.tables)
    changed = sorted(i for i in both if old.tables[i]["map_hash"] != new.tables[i]["map_hash"])
    metadata = sorted(
        i for i in both if old.tables[i]["hash"] != new.tables[i]["hash"] and i not in changed
    )
    restart: dict[str, str] = {}
    for table_id in (*new_ids, *changed, *removed):
        entry = new.tables.get(table_id) or old.tables[table_id]
        group, date = entry["group"], entry["date"]
        if group not in restart or dt.date.fromisoformat(date) < dt.date.fromisoformat(
            restart[group]
        ):
            restart[group] = date
    return Diff(new=new_ids, changed=changed, removed=removed, metadata=metadata, restart=restart)
