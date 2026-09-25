"""Write the serology store: Parquet partitions, rebuilt only where tables changed.

Layout of one store version (the directory ``af.store`` publishes)::

    partitions/<group>/<year>/{tables,antigens,sera,titres}.parquet
    partitions.json     partition -> {table_id: content_hash}, plus the identity-rules version
    report.json         counts: tables, rows, readings, rows without identity, rebuilt/reused

Why partitions by table group and year: a changed table rewrites only its own partition,
and every other partition is hard-linked from the previous version, so a new version costs
little disk and syncs cheaply. One file per table would be simpler to update but thousands
of tiny files make queries slower and manifests bigger.

A partition is reused only when its set of (table_id, content_hash) and the identity-rules
version are both unchanged; anything else rebuilds it. Nothing is updated in place: the
caller gives a fresh output directory, and the previous version is only read.
"""

from __future__ import annotations

import datetime
import json
import os
import shutil
import tempfile
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from af.serology.rows import (
    ANTIGEN_COLUMNS,
    SERUM_COLUMNS,
    TABLE_COLUMNS,
    TITRE_COLUMNS,
    IdentityRules,
    TableFormatError,
    TableRows,
    rows_from_table,
)

KINDS: dict[str, dict[str, str]] = {
    "tables": TABLE_COLUMNS,
    "antigens": ANTIGEN_COLUMNS,
    "sera": SERUM_COLUMNS,
    "titres": TITRE_COLUMNS,
}
PARTITIONS_FILE = "partitions.json"
REPORT_FILE = "report.json"


class StoreError(RuntimeError):
    """The input tables or the previous store version are not usable."""


@dataclass
class BuildReport:
    tables: int = 0
    antigens: int = 0
    sera: int = 0
    readings: int = 0
    without_identity: dict[str, int] = field(default_factory=lambda: {"antigens": 0, "sera": 0})
    rebuilt: list[str] = field(default_factory=list)
    reused: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)


def partition_of(table: Mapping[str, Any]) -> str:
    """``<group>/<year>`` from the table's group and ISO date."""
    try:
        year = datetime.date.fromisoformat(table["date"]).year
    except (KeyError, TypeError, ValueError):
        raise TableFormatError(
            f"{table.get('table_id', '<no table_id>')}: date {table.get('date')!r} is not ISO"
        ) from None
    return f"{table['group']}/{year}"


def build(
    tables: Iterable[Mapping[str, Any]],
    out_dir: Path,
    rules: IdentityRules,
    previous: Path | None = None,
) -> BuildReport:
    """Write a complete store version into the empty directory ``out_dir``.

    ``tables`` is every table the store should contain (I2 dicts). ``previous`` is the
    last store version, whose unchanged partitions are hard-linked instead of rebuilt.
    """
    out_dir = Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise StoreError(f"output directory is not empty: {out_dir}")
    grouped = _group(tables)
    old = _read_partitions(previous) if previous is not None else None
    report = BuildReport()
    partitions: dict[str, dict[str, str]] = {}
    all_counts: dict[str, dict[str, int]] = {}
    for partition, members in sorted(grouped.items()):
        wanted = {t["table_id"]: t["content_hash"] for t in members}
        partitions[partition] = wanted
        target = out_dir / "partitions" / partition
        reusable = (
            old is not None
            and previous is not None
            and old["identity_version"] == rules.version
            and old["partitions"].get(partition) == wanted
        )
        if reusable:
            assert previous is not None
            _link_partition(previous / "partitions" / partition, target)
            report.reused.append(partition)
            counts = old["counts"][partition] if old else {}
        else:
            counts = _write_partition(members, target, rules)
            report.rebuilt.append(partition)
        all_counts[partition] = dict(counts)
        _add_counts(report, counts)
    if old is not None:
        report.removed = sorted(set(old["partitions"]) - set(partitions))
    report.tables = sum(len(p) for p in partitions.values())
    _write_json(
        out_dir / PARTITIONS_FILE,
        {
            "identity_version": rules.version,
            "partitions": partitions,
            "counts": all_counts,
        },
    )
    _write_json(out_dir / REPORT_FILE, report.__dict__)
    return report


def _group(tables: Iterable[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    seen: dict[str, str] = {}
    for table in tables:
        table_id = table["table_id"]
        if table_id in seen:
            raise StoreError(f"table {table_id!r} given twice")
        seen[table_id] = table["content_hash"]
        grouped[partition_of(table)].append(table)
    if not grouped:
        raise StoreError("no tables given: an empty serology store is never intended")
    return grouped


def _write_partition(
    members: list[Mapping[str, Any]], target: Path, rules: IdentityRules
) -> dict[str, int]:
    """Flatten the partition's tables and write its four Parquet files."""
    rows = [rows_from_table(t, rules) for t in members]
    target.mkdir(parents=True)
    _write_parquet(target / "tables.parquet", KINDS["tables"], [r.table for r in rows])
    for kind in ("antigens", "sera", "titres"):
        records = [record for r in rows for record in getattr(r, kind)]
        _write_parquet(target / f"{kind}.parquet", KINDS[kind], records)
    return _counts(rows)


def _counts(rows: list[TableRows]) -> dict[str, int]:
    return {
        "antigens": sum(len(r.antigens) for r in rows),
        "sera": sum(len(r.sera) for r in rows),
        "readings": sum(len(r.titres) for r in rows),
        "antigens_without_identity": sum(r.without_identity["antigens"] for r in rows),
        "sera_without_identity": sum(r.without_identity["sera"] for r in rows),
    }


def _add_counts(report: BuildReport, counts: Mapping[str, int]) -> None:
    report.antigens += counts.get("antigens", 0)
    report.sera += counts.get("sera", 0)
    report.readings += counts.get("readings", 0)
    report.without_identity["antigens"] += counts.get("antigens_without_identity", 0)
    report.without_identity["sera"] += counts.get("sera_without_identity", 0)


def _write_parquet(path: Path, columns: Mapping[str, str], records: list[dict[str, Any]]) -> None:
    """Write records with an explicit schema, via a temporary JSON-lines file.

    Why JSON lines: DuckDB reads it with the declared column types and keeps the
    difference between an empty string and a missing value, which CSV loses.
    """
    import duckdb

    with tempfile.TemporaryDirectory() as tmp:
        source = Path(tmp) / "rows.jsonl"
        with source.open("w", encoding="utf-8") as stream:
            for record in records:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        spec = ", ".join(f"'{name}': '{kind}'" for name, kind in columns.items())
        select = ", ".join(columns)
        con = duckdb.connect()
        query = (
            f"SELECT {select} FROM read_json('{source.as_posix()}', format='newline_delimited', "
            f"columns={{{spec}}})"
        )
        if not records:
            query = f"SELECT {select} FROM ({query}) LIMIT 0"
        con.execute(f"COPY ({query}) TO '{path.as_posix()}' (FORMAT parquet, COMPRESSION zstd)")


def _link_partition(source: Path, target: Path) -> None:
    """Hard-link every file of a reused partition; copy where links are not possible."""
    if not source.is_dir():
        raise StoreError(f"previous version lacks partition {source}")
    target.mkdir(parents=True)
    for file in sorted(source.iterdir()):
        try:
            os.link(file, target / file.name)
        except OSError:
            shutil.copy2(file, target / file.name)


def _read_partitions(previous: Path) -> dict[str, Any]:
    path = Path(previous) / PARTITIONS_FILE
    if not path.is_file():
        raise StoreError(f"previous store version has no {PARTITIONS_FILE}: {previous}")
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return data


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n", encoding="utf-8")
