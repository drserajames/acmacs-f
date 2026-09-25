"""Turn one af table (:class:`af.tables.model.Table`, interface I2) into long-form rows.

The serology store keeps one row per table, per antigen in a table, per serum in a table
and per titre reading. Nothing is merged here: several readings in one cell stay several
rows, and the same virus in two tables stays two antigen rows. Cross-table identity is a
column filled by the identity rules the caller passes in; those rules belong to the chain
engine (``af.chart.identity``), so the store and the chains agree on what "the same
antigen" means and there is one copy of the rules (design rule 6).

An antigen or serum the rules give no identity to (``None``: a DISTINCT point, an antigen
with no passage, a serum with no serum id) gets a table-scoped key instead, so it is never
merged with anything. How many rows that happened to is counted.

Passage: the identity rules compare the passage *with* its harvest date, as ae writes it
("MDCK2/SIAT1 (2016-05-12)"); ``af.tables`` keeps the date apart and builds that string in
``Antigen.ae_passage()``. The store keeps both the lab's passage and the date, and records
the combined string in ``identity_passage`` so preparations use the same rule.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

from af.serology.titre import parse_reading
from af.tables.model import Table

# Annotations are passed as the table's list; af.chart.identity accepts a list or a tuple.
AntigenIdentity = Callable[[str, str, list[str], str], "tuple[Any, ...] | None"]
SerumIdentity = Callable[[str, str, list[str], str], "tuple[Any, ...] | None"]


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
    """A table whose structure the store cannot hold (titre matrix shape, no rows)."""


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
    "passage_date": "DATE",
    "identity_passage": "VARCHAR",
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
    "passage_date": "DATE",
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


def rows_from_table(table: Table, rules: IdentityRules) -> TableRows:
    """Flatten one table into store rows. Raises :class:`TableFormatError` if it is unsound."""
    _check_shape(table)
    table_id = table.table_id
    rows = TableRows(
        table={
            "table_id": table_id,
            "content_hash": table.content_hash(),
            "group_key": table.group,
            "lab": table.lab,
            "subtype": table.subtype,
            "lineage": table.lineage,
            "assay": table.assay,
            "rbc": table.rbc,
            "date": table.date,
            "date_suffix": table.date_suffix,
        }
    )
    for position, antigen in enumerate(table.antigens):
        passage = antigen.ae_passage()
        identity = rules.antigen(antigen.name, antigen.reassortant, antigen.annotations, passage)
        if identity is None:
            rows.without_identity["antigens"] += 1
        rows.antigens.append(
            {
                "table_id": table_id,
                "position": position,
                "name": antigen.name,
                "reassortant": antigen.reassortant,
                "annotations": list(antigen.annotations),
                "passage": antigen.passage,
                "passage_date": antigen.passage_date,
                "identity_passage": passage,
                "collection_date": antigen.date or None,
                "lineage": antigen.lineage,
                "lab_ids": list(antigen.lab_ids),
                "reference": antigen.reference,
                "identity": _identity_text(identity),
                "antigen_key": _key(identity, table_id, "a", position),
            }
        )
    for position, serum in enumerate(table.sera):
        identity = rules.serum(serum.name, serum.reassortant, serum.annotations, serum.serum_id)
        if identity is None:
            rows.without_identity["sera"] += 1
        rows.sera.append(
            {
                "table_id": table_id,
                "position": position,
                "name": serum.name,
                "reassortant": serum.reassortant,
                "annotations": list(serum.annotations),
                "passage": serum.passage,
                "passage_date": serum.passage_date,
                "serum_id": serum.serum_id,
                "species": serum.species,
                "lineage": serum.lineage,
                "identity": _identity_text(identity),
                "serum_key": _key(identity, table_id, "s", position),
            }
        )
    rows.titres = list(_titre_rows(table))
    return rows


def _titre_rows(table: Table) -> Iterator[dict[str, Any]]:
    for ag, row in enumerate(table.titres):
        for sr, readings in enumerate(row):
            for n, raw in enumerate(readings):
                reading = parse_reading(raw)
                yield {
                    "table_id": table.table_id,
                    "antigen_position": ag,
                    "serum_position": sr,
                    "reading": n,
                    "raw": reading.raw,
                    "kind": reading.kind,
                    "value": reading.value,
                    "log": reading.log,
                }


def _check_shape(table: Table) -> None:
    """Only the matrix shape stops flattening. Antigens or sera with no titres are the
    tables store's warnings, and the store holds such tables as they are."""
    n_ag, n_sr = len(table.antigens), len(table.sera)
    if len(table.titres) != n_ag:
        raise TableFormatError(
            f"{table.table_id}: {len(table.titres)} titre rows for {n_ag} antigens"
        )
    for no, row in enumerate(table.titres):
        if len(row) != n_sr:
            raise TableFormatError(
                f"{table.table_id}: titre row {no}: {len(row)} cells, {n_sr} sera"
            )


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
