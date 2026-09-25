"""The tables store: tables by content hash, a manifest, readable dumps, run reports.

Layout under the store's ``tables/`` root (a proposal until workstream 1 fixes I9)::

    objects/<h[:2]>/<hash>.json   one table, content-addressed; never rewritten
    manifest.json                 the current tables (id -> hash); see identity.Manifest
    manifests/<run>.json          every earlier manifest, for diffs and reproducibility
    dumps/<group>/<table_id>.txt  readable dump of the current version of each table
    reports/<run>.txt             what a run read, dropped, changed and got wrong
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import shutil
from pathlib import Path

from .dump import dump
from .identity import Manifest
from .model import Table, canonical_json


class Store:
    def __init__(self, root: Path):
        self.root = root

    @property
    def manifest_path(self) -> Path:
        return self.root / "manifest.json"

    def object_path(self, content_hash: str) -> Path:
        return self.root / "objects" / content_hash[:2] / f"{content_hash}.json"

    def previous_manifest(self) -> Manifest | None:
        return Manifest.read(self.manifest_path) if self.manifest_path.exists() else None

    def write_table(self, table: Table) -> Path:
        """Write the table's object (if new) and its dump; verify what landed on disk."""
        h = table.content_hash()
        path = self.object_path(h)
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(canonical_json(table.to_json()) + "\n", encoding="utf-8")
            tmp.replace(path)
        if Table.from_json(json.loads(path.read_text(encoding="utf-8"))).content_hash() != h:
            raise RuntimeError(f"{path}: stored table does not reproduce hash {h}")
        dump_path = self.root / "dumps" / table.group / f"{table.table_id}.txt"
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        dump_path.write_text(dump(table), encoding="utf-8")
        return path

    def read_table(self, content_hash: str) -> Table:
        return Table.from_json(
            json.loads(self.object_path(content_hash).read_text(encoding="utf-8"))
        )

    def commit(self, manifest: Manifest, report: list[str], run: str | None = None) -> str:
        """Archive the old manifest, install the new one, write the report. Returns the run name."""
        run = run or dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
        (self.root / "manifests").mkdir(parents=True, exist_ok=True)
        (self.root / "reports").mkdir(parents=True, exist_ok=True)
        for table_id, entry in manifest.tables.items():
            if not self.object_path(entry["hash"]).exists():
                raise RuntimeError(
                    f"manifest names {table_id} {entry['hash']}, which is not in the store"
                )
        if self.manifest_path.exists():
            shutil.copy2(self.manifest_path, self.root / "manifests" / f"before-{run}.json")
        tmp = self.manifest_path.with_suffix(".tmp")
        manifest.write(tmp)
        tmp.replace(self.manifest_path)
        shutil.copy2(self.manifest_path, self.root / "manifests" / f"{run}.json")
        (self.root / "reports" / f"{run}.txt").write_text(
            "\n".join(report) + "\n", encoding="utf-8"
        )
        return run


def file_input(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
