"""The tables store: af tables published as :mod:`af.store` datasets, one per lab and group.

Dataset key ``<lab>/<group>`` (e.g. ``labx/h3-hi-guinea-pig-labx``): the tables one chain
is built from. A version holds::

    tables/<table_id>.json    one af table (I2), canonical JSON
    dumps/<table_id>.txt      its readable dump (review and diff only)
    index.json                table_id -> content_hash, map_hash, source_key, date, suffix;
                              plus retired ids, which must never be reused
    report.txt                what the tables of this group dropped and warned about

Everything in a version is a pure function of its tables, so re-reading unchanged inputs
reproduces the same files, the same version id, and ``af.store`` records a reconfirmation
rather than a new version. Tables whose content did not change are hard-linked from the
current version. (``index.json``, not ``manifest.json``: af.store reserves
``MANIFEST.json``, and macOS file systems are case-insensitive.)
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from af.store import Provenance, Store, StoreError, StoreRef

from .dump import dump
from .identity import MANIFEST_FORMAT, Manifest
from .model import Table, canonical_json

KIND = "tables"
INDEX = "index.json"


def dataset_key(lab: str, group: str) -> str:
    return f"{lab.lower()}/{group}"


def read_table(path: Path) -> Table:
    return Table.from_json(json.loads(path.read_text(encoding="utf-8")))


def previous_index(store: Store) -> Manifest | None:
    """Every tables dataset's CURRENT index, combined; None before the first publish.

    Stable suffixes are decided across all groups at once (identity.assign), so the whole
    previous state is read before anything is built.
    """
    refs = store.list_datasets(KIND)
    if not refs:
        return None
    combined = Manifest()
    for ref in refs:
        index = Manifest.read(store.resolve(ref) / INDEX)
        combined.tables.update(index.tables)
        combined.retired.update(index.retired)
    return combined


def current_tables(store: Store) -> dict[str, tuple[StoreRef, Path]]:
    """table_id -> (the dataset ref holding it, the path of its JSON), for CURRENT versions."""
    out: dict[str, tuple[StoreRef, Path]] = {}
    for ref in store.list_datasets(KIND):
        directory = store.resolve(ref)
        for table_id in Manifest.read(directory / INDEX).tables:
            out[table_id] = (ref, directory / "tables" / f"{table_id}.json")
    return out


def publish(
    store: Store, tables: Iterable[Table], manifest: Manifest, provenance: Provenance
) -> list[tuple[StoreRef, str]]:
    """Publish every group touched by ``manifest`` (tables or retired ids).

    Returns (ref, event) per dataset, event being "published" for a new version or
    "reconfirmed" when the files were identical to CURRENT.
    """
    by_dataset: dict[str, list[Table]] = defaultdict(list)
    for table in tables:
        by_dataset[dataset_key(table.lab, table.group)].append(table)
    entries: dict[str, Manifest] = defaultdict(Manifest)
    for table_id, entry in manifest.tables.items():
        entries[dataset_key(entry["lab"], entry["group"])].tables[table_id] = entry
    for table_id, entry in manifest.retired.items():
        entries[dataset_key(entry["lab"], entry["group"])].retired[table_id] = entry
    out = []
    for key in sorted(entries):
        out.append(_publish_one(store, key, by_dataset.get(key, []), entries[key], provenance))
    return out


def _publish_one(
    store: Store, key: str, tables: list[Table], index: Manifest, provenance: Provenance
) -> tuple[StoreRef, str]:
    try:
        before = store.current(KIND, key)
        old = Manifest.read(store.resolve(before) / INDEX).tables
    except StoreError:
        before, old = None, {}
    with store.build(KIND, key) as build:
        for table in sorted(tables, key=lambda t: t.order_key):
            name = f"{table.table_id}.json"
            unchanged = old.get(table.table_id, {}).get("hash") == table.content_hash()
            if unchanged:
                build.link_unchanged(f"tables/{name}")
                build.link_unchanged(f"dumps/{table.table_id}.txt")
            else:
                _write(build.path / "tables" / name, canonical_json(table.to_json()) + "\n")
                _write(build.path / "dumps" / f"{table.table_id}.txt", dump(table))
        _write(build.path / INDEX, _index_text(index))
        _write(build.path / "report.txt", _report(key, tables))
        ref = build.publish(
            provenance, summary={"tables": len(tables), "retired": len(index.retired)}
        )
    event = "reconfirmed" if before is not None and before.version == ref.version else "published"
    return ref, event


def _index_text(index: Manifest) -> str:
    data: dict[str, Any] = {
        "format": MANIFEST_FORMAT,
        "tables": dict(sorted(index.tables.items())),
        "retired": dict(sorted(index.retired.items())),
    }
    return json.dumps(data, indent=1, ensure_ascii=False) + "\n"


def _report(key: str, tables: list[Table]) -> str:
    lines = [f"dataset tables/{key}: {len(tables)} tables"]
    dropped: dict[str, int] = defaultdict(int)
    for table in tables:
        for reason, n in table.dropped.items():
            dropped[reason] += n
    lines.extend(f"dropped  {reason}: {n}" for reason, n in sorted(dropped.items()))
    for table in sorted(tables, key=lambda t: t.order_key):
        lines.extend(f"warning  {table.table_id}: {w}" for w in table.warnings)
    return "\n".join(lines) + "\n"


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
