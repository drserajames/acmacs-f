"""The subtype table: the one copy of what af knows about each influenza subtype.

The facts live in ``af/subtypes.toml`` (shipped inside the package, so every checkout, CI
run and frozen release has it). Before it, the same maps (subtype name -> store dataset,
table subtype and lineage -> chart group prefix, ...) were copied into ten places, and a
new subtype or a renamed prefix meant finding all of them (design rule 6).

The names involved:

- **key**: the row key and store dataset key: ``h1``, ``h3``, ``bvic``, ``byam``.
- **name**: af's subtype name for sequences, clades and trees: ``"A(H3N2)"``, ``"B/Vic"``.
- **table subtype** and **lineage**: what titre tables (and their antigens) record:
  ``("B", "VICTORIA")``. A B table or antigen may leave the lineage unknown (``""``).
- **ace lineage**: the one-letter lineage code .ace charts store per antigen (``"V"``),
  derived as the lineage's first letter, the way af writes it.

Every lookup that matches nothing raises :class:`SubtypeError`, naming what exists
(design rule 1). A row's sub-tables (``[subtype.h3.clades]``) are passed through raw in
:attr:`Subtype.sections`; the workstream that owns a section validates it.
"""

from __future__ import annotations

import functools
import tomllib
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from importlib import resources
from typing import Any

TABLE_FILE = "subtypes.toml"
_ROW_FIELDS = ("name", "table_subtype", "lineage", "group_prefix", "acmacs_data")
_TABLE_SUBTYPE_FIELDS = {"geo": True, "unresolved_group_prefix": False}  # name -> required


class SubtypeError(LookupError):
    """A subtype that the table does not list, or a table that is inconsistent."""


@dataclass(frozen=True)
class Subtype:
    """One row of the table."""

    key: str
    name: str
    table_subtype: str
    lineage: str
    group_prefix: str
    acmacs_data: str
    sections: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)

    @property
    def ace_lineage(self) -> str:
        """The .ace per-antigen lineage code ("V"); "" when the table subtype has no lineages."""
        return self.lineage[:1]


@dataclass(frozen=True)
class TableSubtype:
    """A titre-table subtype and the facts that belong to it rather than to one lineage."""

    name: str
    geo: str
    unresolved_group_prefix: str | None
    subtypes: tuple[Subtype, ...]  # in table order

    @property
    def has_lineages(self) -> bool:
        return len(self.subtypes) > 1


class Subtypes:
    """The validated table. Get the packaged one with :func:`subtypes`."""

    def __init__(self, data: Mapping[str, Any], source: str) -> None:
        self.source = source
        self._rows, self._table_subtypes = _parse(data, source)

    def __iter__(self) -> Iterator[Subtype]:
        return iter(self._rows)

    def keys(self) -> tuple[str, ...]:
        """Every row key (store dataset key), in table order."""
        return tuple(row.key for row in self._rows)

    def by_key(self, key: str) -> Subtype:
        return self._one(lambda row: row.key == key, f"key {key!r}", "key")

    def by_name(self, name: str) -> Subtype:
        return self._one(lambda row: row.name == name, f"name {name!r}", "name")

    def by_acmacs_data(self, key: str) -> Subtype:
        return self._one(
            lambda row: row.acmacs_data == key, f"acmacs-data key {key!r}", "acmacs_data"
        )

    def table_subtype(self, name: str) -> TableSubtype:
        try:
            return self._table_subtypes[name]
        except KeyError:
            raise SubtypeError(
                f"table subtype {name!r} is not in {self.source} "
                f"(it lists {', '.join(self._table_subtypes)})"
            ) from None

    def split_by_lineage(self) -> tuple[str, ...]:
        """Table subtypes that have several lineages: ("B",)."""
        return tuple(name for name, entry in self._table_subtypes.items() if entry.has_lineages)

    def for_table(self, subtype: str, lineage: str) -> tuple[Subtype, ...]:
        """The rows a titre table's (subtype, lineage) can mean, in table order.

        One row when the lineage is known (or the subtype has no lineages); every row of the
        subtype when a subtype with lineages leaves it unknown: ("B", "") -> (bvic, byam).
        Callers rely on the order (the first dataset that matches wins), so it is part of
        the contract.
        """
        entry = self.table_subtype(subtype)
        if lineage == "":
            return entry.subtypes
        rows = tuple(row for row in entry.subtypes if row.lineage == lineage)
        if not rows:
            raise SubtypeError(
                f"table subtype {subtype!r} has no lineage {lineage!r} in {self.source} "
                f"(it has {_lineages(entry)})"
            )
        return rows

    def group_prefix(self, subtype: str, lineage: str) -> str:
        """The chart group prefix: "h1pdm"; for a B table of unknown lineage, "b"."""
        rows = self.for_table(subtype, lineage)
        if len(rows) == 1:
            return rows[0].group_prefix
        prefix = self.table_subtype(subtype).unresolved_group_prefix
        assert prefix is not None  # guaranteed by _parse for subtypes with lineages
        return prefix

    def for_chart(self, subtype: str, ace_lineage: str) -> Subtype:
        """The row for a chart's subtype and its antigens' .ace lineage code: ("B", "V")."""
        entry = self.table_subtype(subtype)
        rows = [row for row in entry.subtypes if row.ace_lineage == ace_lineage]
        if len(rows) != 1:
            codes = ", ".join(repr(row.ace_lineage) for row in entry.subtypes)
            raise SubtypeError(
                f"table subtype {subtype!r} has no .ace lineage code {ace_lineage!r} "
                f"in {self.source} (it has {codes})"
            )
        return rows[0]

    def _one(self, match: Callable[[Subtype], bool], what: str, column: str) -> Subtype:
        for row in self._rows:
            if match(row):
                return row
        known = ", ".join(repr(getattr(row, column)) for row in self._rows)
        raise SubtypeError(f"no subtype with {what} in {self.source} (it has {known})")


@functools.cache
def subtypes() -> Subtypes:
    """The packaged table, read and validated once per process."""
    text = resources.files("af").joinpath(TABLE_FILE).read_text()
    return Subtypes(tomllib.loads(text), f"af/{TABLE_FILE}")


# ---- parsing and validation -----------------------------------------------------------


def _parse(
    data: Mapping[str, Any], source: str
) -> tuple[tuple[Subtype, ...], dict[str, TableSubtype]]:
    """Every problem is collected and reported together, like af.util.config."""
    problems: list[str] = []
    for key in data:
        if key not in ("table_subtype", "subtype"):
            problems.append(f"unknown top-level key {key!r}")
    table_data = _table(data, "table_subtype", problems)
    row_data = _table(data, "subtype", problems)

    entries: dict[str, dict[str, str | None]] = {}
    for name, values in table_data.items():
        entries[name] = _fields(f"table_subtype.{name!r}", values, _TABLE_SUBTYPE_FIELDS, problems)

    rows: list[Subtype] = []
    for key, values in row_data.items():
        row = _row(key, values, problems)
        if row is not None:
            rows.append(row)
    if not rows:
        problems.append("no [subtype.<key>] rows")

    for column in ("name", "acmacs_data", "group_prefix"):
        _unique(column, [getattr(row, column) for row in rows], problems)
    _unique(
        "(table_subtype, lineage)", [(row.table_subtype, row.lineage) for row in rows], problems
    )

    table_subtypes: dict[str, TableSubtype] = {}
    for name, fields in entries.items():
        members = tuple(row for row in rows if row.table_subtype == name)
        table_subtypes[name] = _table_subtype(name, fields, members, problems)
    for row in rows:
        if row.table_subtype not in entries:
            problems.append(
                f"subtype.{row.key}: table_subtype {row.table_subtype!r} "
                "has no [table_subtype] entry"
            )
    unresolved = [entry.unresolved_group_prefix for entry in table_subtypes.values()]
    for row in rows:
        if row.group_prefix in unresolved:
            problems.append(
                f"subtype.{row.key}: group_prefix {row.group_prefix!r} "
                "is also an unresolved_group_prefix"
            )

    if problems:
        lines = "\n".join(f"  - {problem}" for problem in problems)
        raise SubtypeError(f"invalid subtype table {source}:\n{lines}")
    return tuple(rows), table_subtypes


def _table(data: Mapping[str, Any], key: str, problems: list[str]) -> Mapping[str, Any]:
    value = data.get(key)
    if value is None:
        problems.append(f"missing [{key}.*] entries")
        return {}
    if not isinstance(value, Mapping):
        problems.append(f"{key} must be a table")
        return {}
    return value


def _fields(
    where: str, values: Any, spec: Mapping[str, bool], problems: list[str]
) -> dict[str, str | None]:
    if not isinstance(values, Mapping):
        problems.append(f"{where} must be a table")
        return {}
    out: dict[str, str | None] = {}
    for name, required in spec.items():
        value = values.get(name)
        if value is None:
            if required:
                problems.append(f"{where}: missing {name}")
            out[name] = None
        elif not isinstance(value, str) or (value == "" and name != "lineage"):
            problems.append(f"{where}: {name} must be a non-empty string, not {value!r}")
            out[name] = None
        else:
            out[name] = value
    for name in values:
        if name not in spec:
            problems.append(f"{where}: unknown key {name!r}")
    return out


def _row(key: str, values: Any, problems: list[str]) -> Subtype | None:
    where = f"subtype.{key}"
    if not isinstance(values, Mapping):
        problems.append(f"{where} must be a table")
        return None
    sections = {name: value for name, value in values.items() if isinstance(value, Mapping)}
    scalars = {name: value for name, value in values.items() if name not in sections}
    fields = _fields(where, scalars, dict.fromkeys(_ROW_FIELDS, True), problems)
    if any(fields.get(name) is None for name in _ROW_FIELDS):
        return None
    return Subtype(key=key, sections=sections, **{name: str(fields[name]) for name in _ROW_FIELDS})


def _table_subtype(
    name: str, fields: Mapping[str, str | None], members: tuple[Subtype, ...], problems: list[str]
) -> TableSubtype:
    where = f"table_subtype.{name!r}"
    unresolved = fields.get("unresolved_group_prefix")
    if not members:
        problems.append(f"{where}: no [subtype] row has this table_subtype")
    elif len(members) == 1:
        if members[0].lineage:
            problems.append(f'{where}: its only row, {members[0].key}, must have lineage ""')
        if unresolved is not None:
            problems.append(
                f"{where}: unresolved_group_prefix is only for subtypes with several lineages"
            )
    else:
        empty = [row.key for row in members if not row.lineage]
        if empty:
            problems.append(
                f"{where}: has several rows, so each needs a lineage ({', '.join(empty)} has none)"
            )
        _unique(
            f"{where} .ace lineage code",
            [row.ace_lineage for row in members if row.lineage],
            problems,
        )
        if unresolved is None:
            problems.append(f"{where}: has several lineages, so it needs unresolved_group_prefix")
    return TableSubtype(name, str(fields.get("geo")), unresolved, members)


def _unique(column: str, values: list[Any], problems: list[str]) -> None:
    repeated = sorted({repr(value) for value in values if values.count(value) > 1})
    if repeated:
        problems.append(f"{column} must be unique; repeated: {', '.join(repeated)}")


def _lineages(entry: TableSubtype) -> str:
    return ", ".join(repr(row.lineage) for row in entry.subtypes)
