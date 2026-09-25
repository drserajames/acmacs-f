"""Queries over a serology store version.

DuckDB runs in memory over the version's Parquet partitions; nothing is written. The
views are ``tables``, ``antigens``, ``sera`` and ``titres``, one row each per table, antigen
in a table, serum in a table and titre reading (see :mod:`af.serology.rows`).

Preparations: geo and stat count antigen *preparations* (name, reassortant, annotations,
passage), as today's report does (Sarah, 25 Sep 2026). That is not the chain identity:
an antigen with no passage is still one preparation here. DISTINCT points are left out,
as they are duplicates the lab asked to keep apart inside one table.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from af.serology.store import KINDS, PARTITIONS_FILE, StoreError

DISTINCT = "DISTINCT"


def connect(version_dir: Path) -> Any:
    """An in-memory DuckDB connection with the store's four views defined."""
    import duckdb

    version_dir = Path(version_dir)
    if not (version_dir / PARTITIONS_FILE).is_file():
        raise StoreError(f"not a serology store version: {version_dir}")
    con = duckdb.connect()
    for kind in KINDS:
        files = sorted((version_dir / "partitions").glob(f"*/*/{kind}.parquet"))
        if not files:
            raise StoreError(f"{version_dir}: no {kind}.parquet partitions")
        listing = ", ".join(f"'{f.as_posix()}'" for f in files)
        con.execute(f"CREATE VIEW {kind} AS SELECT * FROM read_parquet([{listing}])")
    return con


@dataclass(frozen=True)
class Preparation:
    """One antigen preparation across all tables, with what geo and stat need."""

    subtype: str
    lineage: str
    name: str
    reassortant: str
    annotations: tuple[str, ...]
    passage: str
    collection_date: datetime.date | None  # earliest reported
    first_lab: str  # lab of the earliest table that titrated it
    first_table_date: datetime.date


def preparations(con: Any) -> list[Preparation]:
    """Every antigen preparation, DISTINCT points excluded.

    The earliest table is ordered by (date, date_suffix, lab, table_id): the last two only
    break ties between tables on the same day, so the choice never depends on row order.
    """
    rows = con.execute(
        f"""
        SELECT t.subtype, max(coalesce(a.lineage, '')), a.name, a.reassortant, a.annotations,
               a.passage, min(a.collection_date),
               arg_min(t.lab, (t.date, coalesce(t.date_suffix, 0), t.lab, t.table_id)),
               min(t.date)
        FROM antigens a JOIN tables t USING (table_id)
        WHERE NOT list_contains(a.annotations, '{DISTINCT}')
        GROUP BY t.subtype, a.name, a.reassortant, a.annotations, a.passage
        ORDER BY ALL
        """
    ).fetchall()
    return [
        Preparation(
            subtype=r[0],
            lineage=r[1],
            name=r[2],
            reassortant=r[3],
            annotations=tuple(r[4]),
            passage=r[5],
            collection_date=r[6],
            first_lab=r[7],
            first_table_date=r[8],
        )
        for r in rows
    ]


@dataclass(frozen=True)
class SerumRecord:
    """One serum across all tables (by chain identity, or per table when it has none)."""

    subtype: str
    lineage: str
    serum_key: str
    name: str
    first_lab: str
    strain_collection_date: datetime.date | None  # earliest antigen of the same name


def sera(con: Any) -> list[SerumRecord]:
    """Every serum, with its strain's isolation date taken from antigens of the same name.

    Sarah (25 Sep 2026): a serum is counted by the isolation date of its strain. Where no
    antigen of that name has a collection date the date is None, and stat reports it.
    """
    rows = con.execute(
        f"""
        WITH strain AS (
            SELECT t.subtype, a.name, min(a.collection_date) AS collected
            FROM antigens a JOIN tables t USING (table_id)
            GROUP BY ALL
        )
        SELECT t.subtype, max(coalesce(s.lineage, '')), s.serum_key, any_value(s.name),
               arg_min(t.lab, (t.date, coalesce(t.date_suffix, 0), t.lab, t.table_id)),
               min(strain.collected)
        FROM sera s JOIN tables t USING (table_id)
        LEFT JOIN strain ON strain.subtype = t.subtype AND strain.name = s.name
        WHERE NOT list_contains(s.annotations, '{DISTINCT}')
        GROUP BY t.subtype, s.serum_key
        ORDER BY ALL
        """
    ).fetchall()
    return [SerumRecord(*r) for r in rows]


def strains_with_titres(con: Any, since: datetime.date) -> list[tuple[str, str]]:
    """(subtype, antigen name) for every antigen with at least one reading in a table
    dated ``since`` or later. The tree figure uses this to keep titrated strains."""
    rows = con.execute(
        """
        SELECT DISTINCT t.subtype, a.name
        FROM titres x
        JOIN antigens a ON a.table_id = x.table_id AND a.position = x.antigen_position
        JOIN tables t ON t.table_id = x.table_id
        WHERE t.date >= ?
        ORDER BY ALL
        """,
        [since],
    ).fetchall()
    return [(r[0], r[1]) for r in rows]


def centre_month(con: Any, lab: str, year: int, month: int) -> list[dict[str, Any]]:
    """The tables a centre produced in one month: the core of the monthly report."""
    first = datetime.date(year, month, 1)
    after = datetime.date(year + month // 12, month % 12 + 1, 1)
    cursor = con.execute(
        """
        SELECT table_id, group_key, subtype, assay, rbc, date, date_suffix
        FROM tables WHERE lab = ? AND date >= ? AND date < ?
        ORDER BY date, date_suffix, table_id
        """,
        [lab, first, after],
    )
    columns = [d[0] for d in cursor.description]
    return [dict(zip(columns, r, strict=True)) for r in cursor.fetchall()]
