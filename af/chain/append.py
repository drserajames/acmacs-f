"""A monthly chain that keeps its published order: `[tables] order = "append-to-published"`.

Sarah, Q164 (13 Oct 2026), "cut-off only": a table that arrives after a month's cut-off joins the
chain at its END, whatever its test date, and is recorded as late; earlier months' maps never
change. So the chain's published CURRENT version fixes the order of every table it holds (the
prefix), and the tables selected today that it does not hold follow, in date order, as this
month's arrivals. One that is dated before the prefix's latest table is LATE. "Arrival order" is
thereby the run a table first appeared in; no arrival time is needed. Every prefix step is reused
untouched (a step is keyed by the one before it), so a month costs one step per new table.

A prefix table that comes back REVISED stops the run (Sarah, Q224 a, 18 Oct 2026: "if the table
is recent then likely will want to re-run"). Revised means its map hash changed: the tables
store's hash of what a map depends on (`af.tables.model.Table.map_hash`), which leaves out the
sequence links, the passage class and the provenance fields, so a metadata-only version (03's
EPI_ISL ids, Q221) does not stop anything. The refusal says, per revised table, what changed
(cells, points), its position from the chain's end and how many steps a re-run from it costs.
A re-run is then chosen per case and never automatically, by naming the table and its new map
hash in the chain file:

    [[tables.rerun_from]]
    table = "<table id>"
    map_hash = "<the new map hash, as the refusal printed it>"
    decided = "Sarah, <date>: <why>"

The revised table then stays where it is in the order, and every step from it is remade. An
acknowledgement that matches no revision, or names another hash, is an error (design rule 1).
A prefix table no longer selected stops the run too: the published order cannot be kept.
"""

from __future__ import annotations

import datetime
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

ORDERS = ("date", "append-to-published")


class OrderError(ValueError):
    pass


@dataclass(frozen=True)
class Rerun:
    """`[[tables.rerun_from]]`: one revised table re-run from its own position, decided."""

    table: str
    map_hash: str
    decided: str

    def __post_init__(self) -> None:
        if not (self.table.strip() and self.map_hash.strip() and self.decided.strip()):
            raise OrderError("tables.rerun_from needs table, map_hash and decided")


@dataclass(frozen=True)
class Published:
    """The chain's published CURRENT version as the order needs it."""

    version: str
    tables: Sequence[tuple[str, str]]  # (table id, map hash it was built from), in chain order


@dataclass(frozen=True)
class Revision:
    table_id: str
    position: int  # in the published order, from 0
    from_end: int  # 0 = the published chain's last table
    rerun_steps: int  # steps a re-run from it makes: it, the rest of the prefix, and the new tables
    old_hash: str
    new_hash: str
    changes: Mapping[str, Any] = field(default_factory=dict)

    def describe(self) -> str:
        what = ", ".join(f"{k} {v}" for k, v in self.changes.items()) or "map content"
        return (
            f"  {self.table_id}: position {self.position} ({self.from_end} from the end); a"
            f" re-run from it makes {self.rerun_steps} steps; changed: {what}"
        )

    def acknowledgement(self) -> str:
        return (
            f'  {{ table = "{self.table_id}", map_hash = "{self.new_hash}",'
            ' decided = "<who, date: why>" },'
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "table_id": self.table_id,
            "position": self.position,
            "from_end": self.from_end,
            "rerun_steps": self.rerun_steps,
            "old_map_hash": self.old_hash,
            "new_map_hash": self.new_hash,
            "changes": dict(self.changes),
        }


class RevisedTablesError(OrderError):
    """A prefix table changed: the monthly run stops and shows it (Q224 a)."""

    def __init__(self, dataset: str, version: str, revisions: Sequence[Revision]) -> None:
        self.revisions = tuple(revisions)
        first = min(revisions, key=lambda r: r.position)
        super().__init__(
            f"{dataset}: {len(revisions)} table(s) of the published chain {version} were revised"
            " since it was built; nothing was run. Each re-run is a decision (Q224). Re-running"
            f" makes {first.rerun_steps} steps, from the earliest, {first.table_id}:\n"
            + "\n".join(r.describe() for r in revisions)
            + "\nTo re-run, every revised table is acknowledged under [tables]:\nrerun_from = [\n"
            + "\n".join(r.acknowledgement() for r in revisions)
            + "\n]"
        )


@dataclass(frozen=True)
class Order:
    """The chain's table order today, and what it says about each table."""

    tables: list[Any]  # TableRefs
    late: dict[str, dict[str, Any]]  # table id -> its lateness record
    reruns: tuple[dict[str, Any], ...] = ()  # acknowledged revisions, as recorded

    def to_json(self, published: Published | None) -> dict[str, Any]:
        return {
            "order": "append-to-published",
            "published_version": None if published is None else published.version,
            "prefix": 0 if published is None else len(published.tables),
            "late_tables": self.late,
            "reruns": list(self.reruns),
        }


def append_order(
    dataset: str,
    published: Published | None,
    today: Sequence[Any],
    today_hashes: Mapping[str, str],
    reruns: Sequence[Rerun],
    changes_of: Callable[[str], Mapping[str, Any]],
) -> Order:
    """The published order, then today's new tables in date order (see the module docstring).
    `today` are the tables selected today (TableRefs in date order), `today_hashes` their map
    hashes, `changes_of(table_id)` what changed in a revised table (for the refusal)."""
    if published is None:  # the first version: nothing to keep, the order is the dates'
        if reruns:
            raise OrderError(f"{dataset}: tables.rerun_from with no published version")
        return Order(list(today), {})
    by_id = {t.table_id: t for t in today}
    gone = [tid for tid, _ in published.tables if tid not in by_id]
    if gone:
        raise OrderError(
            f"{dataset}: {len(gone)} table(s) of the published chain {published.version} are no"
            f" longer selected ({', '.join(gone[:3])}{', ...' if len(gone) > 3 else ''}); the"
            " published order cannot be kept"
        )
    held = {tid for tid, _ in published.tables}
    tail = [t for t in today if t.table_id not in held]
    n = len(published.tables)
    revisions = [
        Revision(tid, i, n - 1 - i, n - i + len(tail), old, today_hashes[tid], changes_of(tid))
        for i, (tid, old) in enumerate(published.tables)
        if today_hashes[tid] != old
    ]
    acked = {r.table: r for r in reruns}
    unknown = [
        r.table for r in reruns
        if not any(v.table_id == r.table and v.new_hash == r.map_hash for v in revisions)
    ]  # fmt: skip
    if unknown:
        raise OrderError(
            f"{dataset}: tables.rerun_from names {', '.join(unknown)}, but no such revision (table"
            " and new map hash) is pending: an acknowledgement that matches nothing is an error"
        )
    pending = [v for v in revisions if v.table_id not in acked]
    if pending:
        raise RevisedTablesError(dataset, published.version, pending)
    prefix = [by_id[tid] for tid, _ in published.tables]
    latest = max((t.date, t.suffix) for t in prefix)
    late = {
        t.table_id: {
            "date": t.date.isoformat(),
            "published_until": datetime.date.isoformat(latest[0]),
            "published_version": published.version,
        }
        for t in tail
        if (t.date, t.suffix) < latest
    }
    done = tuple(
        {**v.to_json(), "decided": acked[v.table_id].decided} for v in revisions
    )  # fmt: skip
    return Order([*prefix, *tail], late, done)


def table_changes(old: Any, new: Any) -> dict[str, int]:
    """What changed between two versions of one table (af.tables.model.Table), counted for the
    refusal, in the terms the map hash uses (`Table.map_content`): points added or removed by
    identity; per field, how many points both hold changed it (a serum's species, which the
    ferret-only policy reads, is one); cells among the points both hold; other table fields."""
    a, b = old.map_content(), new.map_content()
    out: dict[str, int] = {}
    matched: dict[str, list[tuple[int, int]]] = {}
    for what, key in (("antigens", _antigen_key), ("sera", _serum_key)):
        ka = {key(p): i for i, p in enumerate(a[what])}
        kb = {key(p): i for i, p in enumerate(b[what])}
        if added := len(kb.keys() - ka.keys()):
            out[f"{what} added"] = added
        if removed := len(ka.keys() - kb.keys()):
            out[f"{what} removed"] = removed
        pairs = [(ka[k], kb[k]) for k in ka.keys() & kb.keys()]
        matched[what] = pairs
        fields: dict[str, int] = {}
        for i, j in pairs:
            x, y = a[what][i], b[what][j]
            for f in sorted(set(x) | set(y)):
                if x.get(f) != y.get(f):
                    fields[f] = fields.get(f, 0) + 1
        for f, n in fields.items():
            out[f"{what} {f}"] = n
    cells = sum(
        a["titres"][i][j] != b["titres"][k][m]
        for i, k in matched["antigens"]
        for j, m in matched["sera"]
    )  # fmt: skip
    if cells:
        out["cells changed"] = cells
    for f in sorted((set(a) | set(b)) - {"antigens", "sera", "titres"}):
        if a.get(f) != b.get(f):
            out[f"table {f}"] = 1
    return out


def _antigen_key(p: Mapping[str, Any]) -> tuple:
    return (p.get("name"), p.get("reassortant"), tuple(p.get("annotations") or ()),
            p.get("passage"), p.get("passage_date"))  # fmt: skip


def _serum_key(p: Mapping[str, Any]) -> tuple:
    return (p.get("name"), p.get("reassortant"), tuple(p.get("annotations") or ()),
            p.get("serum_id"))  # fmt: skip
