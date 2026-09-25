"""Turn one af table (interface I2, ``format: "af-table-1"``) into long-form store rows.

The serology store keeps one row per table, per antigen in a table, per serum in a table
and per titre reading. Nothing is merged here: several readings in one cell stay several
rows, and the same virus in two tables stays two antigen rows. Cross-table identity is a
column filled by the identity rules the caller passes in; those rules belong to the chain
engine (``af.chart.identity``), so the store and the chains agree on what "the same
antigen" means and there is one copy of the rules (design rule 6).

An antigen or serum the rules give no identity to (``None``: a DISTINCT point, an antigen
with no passage, a serum with no serum id) gets a table-scoped key instead, so it is never
merged with anything. How many rows that happened to is counted.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from af.serology.titre import parse_reading

TABLE_FORMAT = "af-table-1"

AntigenIdentity = Callable[[str, str, Sequence[str], str], "tuple[Any, ...] | None"]
SerumIdentity = Callable[[str, str, Sequence[str], str], "tuple[Any, ...] | None"]


@dataclass(frozen=True)
class IdentityRules:
    """The cross-table identity functions, and a version string recorded with the store.

    ``antigen(name, reassortant, annotations, passage)`` and
    ``serum(name, reassortant, annotations, serum_id)`` return a hashable key or None.
    ``version`` must change whenever the rules change: it is part of what decides
    whether stored partitions are still valid.
    """

    antigen: AntigenIdentity
    serum: SerumIdentity
    version: str


class TableFormatError(ValueError):
    """A table that is not in the af table format this store reads."""


@dataclass
class TableRows:
    """Every store row that one table contributes."""

    table: dict[str, Any]
    antigens: list[dict[str, Any]] = field(default_factory=list)
    sera: list[dict[str, Any]] = field(default_factory=list)
    titres: list[dict[str, Any]] = field(default_factory=list)
    without_identity: dict[str, int] = field(default_factory=lambda: {"antigens": 0, "sera": 0})


TABLE_COLUMNS = {
    "table_id": "VARCHAR",
    "content_hash": "VARCHAR",
    "group_key": "VARCHAR",
    "lab": "VARCHAR",
    "subtype": "VARCHAR",
    "lineage": "VARCHAR",
    "assay": "VARCHAR",
    "rbc": "VARCHAR",
    "date": "DATE",
    "date_suffix": "INTEGER",
}
ANTIGEN_COLUMNS = {
    "table_id": "VARCHAR",
    "position": "INTEGER",
    "name": "VARCHAR",
    "reassortant": "VARCHAR",
    "annotations": "VARCHAR[]",
    "passage": "VARCHAR",
    "collection_date": "DATE",
    "lineage": "VARCHAR",
    "lab_ids": "VARCHAR[]",
    "reference": "BOOLEAN",
    "identity": "VARCHAR",
    "antigen_key": "VARCHAR",
}
SERUM_COLUMNS = {
    "table_id": "VARCHAR",
    "position": "INTEGER",
    "name": "VARCHAR",
    "reassortant": "VARCHAR",
    "annotations": "VARCHAR[]",
    "passage": "VARCHAR",
    "serum_id": "VARCHAR",
    "species": "VARCHAR",
    "lineage": "VARCHAR",
    "identity": "VARCHAR",
    "serum_key": "VARCHAR",
}
TITRE_COLUMNS = {
    "table_id": "VARCHAR",
    "antigen_position": "INTEGER",
    "serum_position": "INTEGER",
    "reading": "INTEGER",
    "raw": "VARCHAR",
    "kind": "VARCHAR",
    "value": "DOUBLE",
    "log": "DOUBLE",
}


def rows_from_table(table: Mapping[str, Any], rules: IdentityRules) -> TableRows:
    """Flatten one I2 table into store rows. Raises :class:`TableFormatError` on bad input."""
    _check_format(table)
    table_id = table["table_id"]
    rows = TableRows(table={column: table.get(column) for column in TABLE_COLUMNS})
    rows.table["group_key"] = table["group"]
    for position, antigen in enumerate(table["antigens"]):
        identity = rules.antigen(
            antigen["name"],
            antigen.get("reassortant", ""),
            antigen.get("annotations", []),
            antigen.get("passage", ""),
        )
        if identity is None:
            rows.without_identity["antigens"] += 1
        rows.antigens.append(
            {
                "table_id": table_id,
                "position": position,
                "name": antigen["name"],
                "reassortant": antigen.get("reassortant", ""),
                "annotations": list(antigen.get("annotations", [])),
                "passage": antigen.get("passage", ""),
                "collection_date": antigen.get("date") or None,
                "lineage": antigen.get("lineage", ""),
                "lab_ids": list(antigen.get("lab_ids", [])),
                "reference": bool(antigen.get("reference", False)),
                "identity": _identity_text(identity),
                "antigen_key": _key(identity, table_id, "a", position),
            }
        )
    for position, serum in enumerate(table["sera"]):
        identity = rules.serum(
            serum["name"],
            serum.get("reassortant", ""),
            serum.get("annotations", []),
            serum.get("serum_id", ""),
        )
        if identity is None:
            rows.without_identity["sera"] += 1
        rows.sera.append(
            {
                "table_id": table_id,
                "position": position,
                "name": serum["name"],
                "reassortant": serum.get("reassortant", ""),
                "annotations": list(serum.get("annotations", [])),
                "passage": serum.get("passage", ""),
                "serum_id": serum.get("serum_id", ""),
                "species": serum.get("species", ""),
                "lineage": serum.get("lineage", ""),
                "identity": _identity_text(identity),
                "serum_key": _key(identity, table_id, "s", position),
            }
        )
    rows.titres = list(_titre_rows(table))
    return rows


def _titre_rows(table: Mapping[str, Any]) -> Any:
    table_id = table["table_id"]
    for ag, row in enumerate(table["titres"]):
        for sr, readings in enumerate(row):
            for n, raw in enumerate(readings):
                reading = parse_reading(raw)
                yield {
                    "table_id": table_id,
                    "antigen_position": ag,
                    "serum_position": sr,
                    "reading": n,
                    "raw": reading.raw,
                    "kind": reading.kind,
                    "value": reading.value,
                    "log": reading.log,
                }


def _check_format(table: Mapping[str, Any]) -> None:
    """Refuse a table whose shape does not match I2, naming what is wrong."""
    where = table.get("table_id", "<no table_id>")
    if table.get("format") != TABLE_FORMAT:
        raise TableFormatError(f"{where}: format {table.get('format')!r}, expected {TABLE_FORMAT}")
    for key in ("table_id", "content_hash", "group", "lab", "subtype", "assay", "date"):
        if not table.get(key):
            raise TableFormatError(f"{where}: missing {key!r}")
    n_ag, n_sr = len(table["antigens"]), len(table["sera"])
    titres = table["titres"]
    if len(titres) != n_ag or any(len(row) != n_sr for row in titres):
        raise TableFormatError(f"{where}: titre matrix is not {n_ag} x {n_sr}")


def _identity_text(identity: tuple[Any, ...] | None) -> str | None:
    return None if identity is None else json.dumps(list(identity), ensure_ascii=False)


def _key(identity: tuple[Any, ...] | None, table_id: str, kind: str, position: int) -> str:
    """A stable key: hash of the identity, or table-scoped when there is none.

    The table-scoped form uses the lab's row/column position inside that one table, which
    is fixed for a given table content; it is a storage key, never a selector.
    """
    if identity is None:
        return f"{table_id}#{kind}{position}"
    text = json.dumps(list(identity), ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(text.encode()).hexdigest()[:16]
